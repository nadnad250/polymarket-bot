"""Cycle CI — 1 itération collect + simulate, version lean pour GitHub Actions.

Exécuté toutes les 5 min par `.github/workflows/collect.yml`.
- Fetch 1 snapshot Binance + Polymarket
- Insère dans SQLite
- Si modèle dispo : inférence + simule trade si edge
- Résout les trades expirés (event clos)
- Output : DB mise à jour, trades + summary écrits
"""
from __future__ import annotations

import json
import pickle
import sqlite3
import time
from contextlib import closing
from datetime import datetime
from pathlib import Path

import numpy as np

from config import DB_PATH, INITIAL_CAPITAL, MAX_POSITION_PCT
from src.fetchers.btc import BTCFetcher
from src.fetchers.collector import init_db, insert_tick
from src.fetchers.polymarket import PolymarketClient
from src.simulator.fees import DEFAULT_FEES
from src.simulator.live_loop import (
    LiveTrade, TRADES_DB, init_trades_db, resolve_trade, save_trade,
    write_summary,
)

MODEL_PATH = Path("data/model_ensemble.pkl")
MIN_EDGE = 0.03
KELLY_CAP = MAX_POSITION_PCT


def _load_model() -> dict | None:
    if not MODEL_PATH.exists():
        return None
    try:
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"[ci] échec chargement modèle: {e}")
        return None


def _predict_for_latest(model_payload: dict) -> float | None:
    """Construit les features depuis les derniers ticks et prédit P(up)."""
    import pandas as pd
    from src.models.features import FEATURE_COLS, build_features

    with closing(sqlite3.connect(str(DB_PATH))) as conn:
        df = pd.read_sql(
            "SELECT * FROM ticks ORDER BY ts DESC LIMIT 200", conn,
        ).sort_values("ts").reset_index(drop=True)
    if len(df) < 60:
        return None

    feats = build_features(df)
    X = feats[FEATURE_COLS].replace([np.inf, -np.inf], 0).fillna(0).iloc[-1:]
    from src.models.ensemble import predict_proba
    return float(predict_proba(model_payload, X)[0])


def _cash_from_trades() -> float:
    """Recalcule le cash à partir de tous les trades clos."""
    if not TRADES_DB.exists():
        return INITIAL_CAPITAL
    with closing(sqlite3.connect(str(TRADES_DB))) as conn:
        r = conn.execute(
            "SELECT COALESCE(SUM(pnl), 0) FROM trades WHERE outcome IS NOT NULL"
        ).fetchone()
    return INITIAL_CAPITAL + float(r[0] if r else 0)


def _open_trade_for(event_slug: str) -> LiveTrade | None:
    if not TRADES_DB.exists():
        return None
    with closing(sqlite3.connect(str(TRADES_DB))) as conn:
        r = conn.execute(
            "SELECT * FROM trades WHERE event_slug=? AND outcome IS NULL", (event_slug,)
        ).fetchone()
    if not r:
        return None
    return LiveTrade(
        event_slug=r[0], opened_at=r[1], side=r[2], entry_price=r[3],
        size_usd=r[4], btc_entry=r[5], momentum=r[6], imbalance=r[7],
        resolved_at=r[8], outcome=r[9], pnl=r[10], btc_exit=r[11],
    )


def _resolve_expired_trades(binance: BTCFetcher, poly: PolymarketClient) -> None:
    """Résout les trades dont l'event est clos."""
    if not TRADES_DB.exists():
        return
    with closing(sqlite3.connect(str(TRADES_DB))) as conn:
        opens = conn.execute(
            "SELECT event_slug FROM trades WHERE outcome IS NULL"
        ).fetchall()

    for (slug,) in opens:
        event = poly.get_event(slug)
        if event is None:
            continue
        if not event.get("closed"):
            continue
        trade = _open_trade_for(slug)
        if trade is None:
            continue
        btc_exit = binance.get_price()
        trade = resolve_trade(trade, btc_exit)
        save_trade(trade)
        status = "WIN" if trade.outcome else "LOSS"
        print(f"[ci] résolu {status} {slug} pnl={trade.pnl:+.2f}")


def run_cycle() -> None:
    init_db()
    init_trades_db()
    binance = BTCFetcher()
    poly = PolymarketClient()

    try:
        # 1) Tick Binance + Polymarket
        event = poly.find_btc_updown_event()
        if event is None:
            print("[ci] aucun event actif, skip.")
            return

        btc = binance.get_book_ticker()
        imb = binance.orderbook_imbalance(levels=10)
        snap = poly.snapshot_event(event)
        if snap is None:
            print("[ci] snapshot Polymarket indisponible, skip.")
            return

        with closing(sqlite3.connect(str(DB_PATH))) as conn:
            insert_tick(conn, (
                int(time.time() * 1000),
                btc.price, btc.bid, btc.ask, imb,
                snap.event_slug, snap.yes_price, snap.no_price,
                snap.volume_24h, snap.question[:100],
            ))
        print(f"[ci] tick {datetime.utcnow().isoformat()}Z | BTC ${btc.price:,.2f} | YES={snap.yes_price:.3f}")

        # 2) Résolution des trades expirés
        _resolve_expired_trades(binance, poly)

        # 3) Inférence + trade si edge
        model = _load_model()
        if model is None:
            print("[ci] pas de modèle → pas de trade cette itération.")
        else:
            # Skip si trade déjà ouvert pour cet event
            if _open_trade_for(snap.event_slug) is None:
                p_up = _predict_for_latest(model)
                if p_up is None:
                    print("[ci] pas assez de ticks pour prédire.")
                else:
                    edge_yes = p_up - snap.yes_price
                    edge_no = (1 - p_up) - snap.no_price
                    if edge_yes > edge_no:
                        side, price, edge = "YES", snap.yes_price, edge_yes
                    else:
                        side, price, edge = "NO", snap.no_price, edge_no

                    if edge >= MIN_EDGE and 0.05 < price < 0.95:
                        cash = _cash_from_trades()
                        # Kelly fractionnaire cap
                        size_pct = min(KELLY_CAP, max(0.0, edge))
                        size_usd = cash * size_pct
                        eff_price, _ = DEFAULT_FEES.apply_entry(price, size_usd)
                        trade = LiveTrade(
                            event_slug=snap.event_slug,
                            opened_at=int(time.time() * 1000),
                            side=side,
                            entry_price=eff_price,
                            size_usd=size_usd,
                            btc_entry=btc.price,
                            momentum=0.0,
                            imbalance=imb,
                        )
                        save_trade(trade)
                        print(
                            f"[ci] ▶ OPEN {side} @{eff_price:.3f} size=${size_usd:.2f} "
                            f"p_up={p_up:.3f} edge={edge:+.3f}"
                        )
                    else:
                        print(f"[ci] pas d'edge (edge={edge:+.3f} price={price:.3f})")

        # 4) Résumé pour dashboard
        cash = _cash_from_trades()
        all_trades = []
        with closing(sqlite3.connect(str(TRADES_DB))) as conn:
            rows = conn.execute("SELECT * FROM trades").fetchall()
        for r in rows:
            all_trades.append(LiveTrade(
                event_slug=r[0], opened_at=r[1], side=r[2], entry_price=r[3],
                size_usd=r[4], btc_entry=r[5], momentum=r[6], imbalance=r[7],
                resolved_at=r[8], outcome=r[9], pnl=r[10], btc_exit=r[11],
            ))
        write_summary(cash, all_trades)
    finally:
        binance.close()
        poly.close()


if __name__ == "__main__":
    run_cycle()
