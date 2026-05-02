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
import os
import pickle
import sqlite3
import sys
import time
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (
    ALLOW_BASELINE_TRADES,
    DB_PATH,
    INITIAL_CAPITAL,
    MAX_MODEL_BRIER,
    MAX_MODEL_LOGLOSS,
    MAX_OPEN_POSITIONS,
    MAX_POSITION_PCT,
    MIN_MODEL_AUC,
    MIN_MODEL_TEST_ROWS,
    MIN_SECONDS_TO_CLOSE,
    MIN_TRADE_EDGE,
)
from src.fetchers.btc import BTCFetcher
from src.fetchers.collector import init_db, insert_tick
from src.fetchers.polymarket import PolymarketClient
from src.simulator.fees import DEFAULT_FEES
from src.simulator.live_loop import (
    LiveTrade, TRADES_DB, init_trades_db, resolve_trade, save_trade,
    write_summary,
)
from src.simulator.paper import kelly_size

MODEL_PATH = Path("data/model_ensemble.pkl")
METRICS_PATH = Path("data/model_metrics.json")
MIN_CONFIDENCE = 0.10  # |p_up - 0.5| min ; sous ce seuil le modèle n'est pas confiant
KELLY_CAP = MAX_POSITION_PCT
PRICE_MIN = 0.08
PRICE_MAX = 0.92
FOLLOW_EVENT_UNTIL_CLOSE = os.getenv("FOLLOW_EVENT_UNTIL_CLOSE", "1").lower() in {"1", "true", "yes"}
FOLLOW_POLL_SEC = int(os.getenv("FOLLOW_POLL_SEC", "15"))


def _compute_momentum(binance: BTCFetcher) -> float:
    """Momentum 1min : (close_now / close_il_y_a_1min) - 1."""
    try:
        klines = binance.get_klines(limit=2)
        if len(klines) < 2:
            return 0.0
        old_close = float(klines[0][4])
        new_close = float(klines[1][4])
        return (new_close - old_close) / old_close if old_close > 0 else 0.0
    except Exception:
        return 0.0


def _load_model() -> dict | None:
    metrics_override = None
    if METRICS_PATH.exists():
        try:
            metrics_payload = json.loads(METRICS_PATH.read_text())
            metrics_override = metrics_payload.get("metrics")
            if isinstance(metrics_override, dict) and metrics_override.get("tradeable") is False:
                return {"metrics": metrics_override}
        except Exception as e:
            print(f"[ci] métriques modèle ignorées: {e}")

    if not MODEL_PATH.exists():
        return None
    try:
        with open(MODEL_PATH, "rb") as f:
            payload = pickle.load(f)
        if metrics_override:
            payload["metrics"] = metrics_override
        return payload
    except Exception as e:
        print(f"[ci] échec chargement modèle: {e}")
        return None


def _model_is_tradeable(model_payload: dict) -> tuple[bool, str]:
    metrics = model_payload.get("metrics") or {}
    if metrics.get("tradeable") is False:
        return False, str(metrics.get("trade_block_reason") or "metrics marked untradeable")
    if int(metrics.get("n_test") or 0) < MIN_MODEL_TEST_ROWS:
        return False, f"n_test < {MIN_MODEL_TEST_ROWS}"
    auc = metrics.get("auc")
    if auc is None or float(auc) < MIN_MODEL_AUC:
        return False, f"auc < {MIN_MODEL_AUC}"
    if float(metrics.get("brier") or 1.0) > MAX_MODEL_BRIER:
        return False, f"brier > {MAX_MODEL_BRIER}"
    if float(metrics.get("logloss") or 99.0) > MAX_MODEL_LOGLOSS:
        return False, f"logloss > {MAX_MODEL_LOGLOSS}"
    return True, "ok"


def _event_close_ms_from_slug(slug: str) -> int | None:
    try:
        return int(str(slug).rsplit("-", 1)[-1]) * 1000
    except Exception:
        return None


def _seconds_to_close(end_date: str | None, fallback_slug: str = "") -> float | None:
    if end_date:
        try:
            end_dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
            return (end_dt - datetime.now(tz=timezone.utc)).total_seconds()
        except Exception:
            pass
    close_ms = _event_close_ms_from_slug(fallback_slug)
    if close_ms is None:
        return None
    return close_ms / 1000.0 - time.time()


def _btc_price_near_ts(
    target_ms: int,
    max_skew_ms: int = 180_000,
    after_only: bool = False,
) -> float | None:
    if not Path(DB_PATH).exists():
        return None
    with closing(sqlite3.connect(str(DB_PATH))) as conn:
        if after_only:
            row = conn.execute(
                """
                SELECT btc_price FROM ticks
                WHERE ts BETWEEN ? AND ?
                ORDER BY ts ASC
                LIMIT 1
                """,
                (target_ms, target_ms + max_skew_ms),
            ).fetchone()
        else:
            row = conn.execute(
                """
                SELECT btc_price FROM ticks
                WHERE ts BETWEEN ? AND ?
                ORDER BY ABS(ts - ?) ASC
                LIMIT 1
                """,
                (target_ms - max_skew_ms, target_ms + max_skew_ms, target_ms),
            ).fetchone()
    return float(row[0]) if row else None


def _insert_snapshot_tick(
    conn: sqlite3.Connection,
    btc,
    imbalance: float,
    snap,
) -> None:
    insert_tick(conn, (
        int(time.time() * 1000),
        btc.price, btc.bid, btc.ask, imbalance,
        snap.event_slug, snap.yes_price, snap.no_price,
        snap.volume_24h, snap.question[:100],
    ))


def _follow_event_until_close(
    binance: BTCFetcher,
    poly: PolymarketClient,
    event: dict,
    last_snap,
) -> None:
    if not FOLLOW_EVENT_UNTIL_CLOSE:
        return
    close_ms = _event_close_ms_from_slug(last_snap.event_slug)
    if close_ms is None:
        return

    now_ms = int(time.time() * 1000)
    remaining_ms = close_ms - now_ms
    if remaining_ms < 20_000 or remaining_ms > 330_000:
        return

    print(f"[ci] suivi dense jusqu'a la cloture ({remaining_ms/1000:.0f}s restants)")
    deadline_ms = close_ms + 35_000
    with closing(sqlite3.connect(str(DB_PATH))) as conn:
        while int(time.time() * 1000) < deadline_ms:
            sleep_s = min(FOLLOW_POLL_SEC, max(1, (deadline_ms - int(time.time() * 1000)) / 1000))
            time.sleep(sleep_s)
            try:
                btc = binance.get_book_ticker()
                imb = binance.orderbook_imbalance(levels=10)
                snap = poly.snapshot_event(event) or last_snap
                _insert_snapshot_tick(conn, btc, imb, snap)
                last_snap = snap
                print(f"[ci] dense tick BTC ${btc.price:,.2f} | YES={snap.yes_price:.3f}")
            except Exception as e:
                print(f"[ci] dense tick skip: {e}")


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
    """Recalcule le cash en tenant compte des positions encore ouvertes."""
    if not TRADES_DB.exists():
        return INITIAL_CAPITAL
    with closing(sqlite3.connect(str(TRADES_DB))) as conn:
        closed = conn.execute(
            "SELECT COALESCE(SUM(pnl), 0) FROM trades WHERE outcome IS NOT NULL"
        ).fetchone()
        open_size = conn.execute(
            "SELECT COALESCE(SUM(size_usd), 0) FROM trades WHERE outcome IS NULL"
        ).fetchone()
    cash = INITIAL_CAPITAL + float(closed[0] if closed else 0) - float(open_size[0] if open_size else 0)
    return max(0.0, cash)


def _open_trades() -> list[LiveTrade]:
    if not TRADES_DB.exists():
        return []
    with closing(sqlite3.connect(str(TRADES_DB))) as conn:
        rows = conn.execute("SELECT * FROM trades WHERE outcome IS NULL").fetchall()
    return [
        LiveTrade(
            event_slug=r[0], opened_at=r[1], side=r[2], entry_price=r[3],
            size_usd=r[4], btc_entry=r[5], momentum=r[6], imbalance=r[7],
            resolved_at=r[8], outcome=r[9], pnl=r[10], btc_exit=r[11],
        )
        for r in rows
    ]


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
    """Résout les trades dont l'event est clos OU dont endDate est passé.

    Polymarket prend 5-10 min pour passer closed=True après la fin du marché.
    On résout dès que endDate est passé (le résultat est déterminé par BTC à ce
    moment-là, peu importe quand Polymarket l'enregistre).
    """
    if not TRADES_DB.exists():
        return
    with closing(sqlite3.connect(str(TRADES_DB))) as conn:
        opens = conn.execute(
            "SELECT event_slug FROM trades WHERE outcome IS NULL"
        ).fetchall()

    now = datetime.now(tz=timezone.utc)

    for (slug,) in opens:
        event = poly.get_event(slug)
        is_closed = bool(event.get("closed", False)) if event else False
        end_date_str = (event.get("endDate") or "") if event else ""
        is_expired = False
        close_ms = None
        if end_date_str:
            try:
                end_dt = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
                close_ms = int(end_dt.timestamp() * 1000)
                # 30s de marge pour s'assurer que la résolution est définitive
                if (now - end_dt).total_seconds() > 30:
                    is_expired = True
            except Exception:
                pass
        if close_ms is None:
            close_ms = _event_close_ms_from_slug(slug)
            if close_ms is not None and (time.time() * 1000 - close_ms) > 30_000:
                is_expired = True

        if not (is_closed or is_expired):
            continue

        trade = _open_trade_for(slug)
        if trade is None:
            continue
        btc_exit = _btc_price_near_ts(close_ms, after_only=True) if close_ms is not None else None
        if btc_exit is None:
            btc_exit = binance.get_price()
        trade = resolve_trade(trade, btc_exit)
        save_trade(trade)
        status = "WIN" if trade.outcome else "LOSS"
        reason = "closed" if is_closed else "expired"
        print(f"[ci] résolu [{reason}] {status} {slug} pnl={trade.pnl:+.2f}")


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
            _insert_snapshot_tick(conn, btc, imb, snap)
        print(f"[ci] tick {datetime.utcnow().isoformat()}Z | BTC ${btc.price:,.2f} | YES={snap.yes_price:.3f}")

        # 2) Résolution des trades expirés
        _resolve_expired_trades(binance, poly)

        # 3) Décision de trade — modèle ML uniquement si les garde-fous passent
        sec_to_close = _seconds_to_close(snap.end_date, snap.event_slug)
        open_positions = _open_trades()
        decision = None  # (side, price, edge_or_score, source, p_up)

        if _open_trade_for(snap.event_slug) is not None:
            print(f"[ci] position déjà ouverte sur {snap.event_slug}, skip.")
        elif len(open_positions) >= MAX_OPEN_POSITIONS:
            print(f"[ci] {len(open_positions)} position ouverte non résolue, skip nouveau trade.")
        elif sec_to_close is not None and sec_to_close < MIN_SECONDS_TO_CLOSE:
            print(f"[ci] trop proche de la clôture ({sec_to_close:.0f}s), skip.")
        else:
            model = _load_model()

            if model is not None:
                ok, reason = _model_is_tradeable(model)
                if not ok:
                    print(f"[ci] modèle non tradable: {reason}")
                else:
                    p_up = _predict_for_latest(model)
                    if p_up is not None:
                        confidence = abs(p_up - 0.5)
                        if confidence < MIN_CONFIDENCE:
                            print(
                                f"[ci] modèle peu confiant (|p_up-0.5|={confidence:.3f} "
                                f"< {MIN_CONFIDENCE}), skip ML"
                            )
                        else:
                            edge_yes = p_up - snap.yes_price
                            edge_no = (1 - p_up) - snap.no_price
                            if edge_yes > edge_no:
                                side, price, edge = "YES", snap.yes_price, edge_yes
                            else:
                                side, price, edge = "NO", snap.no_price, edge_no
                            required_edge = (
                                MIN_TRADE_EDGE
                                + DEFAULT_FEES.spread_pct / 2
                                + DEFAULT_FEES.slippage_pct
                            )
                            if edge >= required_edge and PRICE_MIN < price < PRICE_MAX:
                                decision = (side, price, edge, "ml", p_up)
                            else:
                                print(
                                    f"[ci] edge net insuffisant ({edge:.3f} < "
                                    f"{required_edge:.3f}) ou prix hors borne ({price:.3f})"
                                )
            else:
                print("[ci] aucun modèle disponible.")

            if decision is None and ALLOW_BASELINE_TRADES:
                momentum = _compute_momentum(binance)
                signal = momentum * 1000 + imb * 0.5
                if momentum > 0.0005 and imb > 0.35 and snap.yes_price < 0.52:
                    decision = ("YES", snap.yes_price, abs(signal), "baseline", None)
                elif momentum < -0.0005 and imb < -0.35 and snap.no_price < 0.52:
                    decision = ("NO", snap.no_price, abs(signal), "baseline", None)
            elif decision is None:
                print("[ci] baseline désactivée; paper trading en mode protection.")

        if decision is not None:
            side, price, score, src, p_up = decision
            cash = _cash_from_trades()
            if src == "ml":
                size_pct = min(KELLY_CAP, kelly_size(score, price))
            else:
                size_pct = min(KELLY_CAP / 2, 0.0025)

            if cash <= 0 or size_pct <= 0:
                print(f"[ci] sizing nul (cash={cash:.2f}, size_pct={size_pct:.4f}), skip.")
            else:
                size_usd = cash * size_pct
                eff_price, _ = DEFAULT_FEES.apply_entry(price, size_usd)
                trade = LiveTrade(
                    event_slug=snap.event_slug,
                    opened_at=int(time.time() * 1000),
                    side=side,
                    entry_price=eff_price,
                    size_usd=size_usd,
                    btc_entry=btc.price,
                    momentum=score if src == "baseline" else (p_up or 0.0),
                    imbalance=imb,
                )
                save_trade(trade)
                p_str = f"p_up={p_up:.3f}" if p_up is not None else f"score={score:+.3f}"
                print(
                    f"[ci] OPEN [{src.upper()}] {side} @{eff_price:.3f} "
                    f"size=${size_usd:.2f} {p_str}"
                )

        _follow_event_until_close(binance, poly, event, snap)
        _resolve_expired_trades(binance, poly)

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
