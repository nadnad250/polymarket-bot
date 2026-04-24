"""Simulateur live — trade en continu pendant que les data s'accumulent.

Stratégie baseline (sans ML, juste heuristique) :
  - Si momentum 1min > 0 ET orderbook imbalance > 0 → BUY YES
  - Si momentum 1min < 0 ET orderbook imbalance < 0 → BUY NO
  - Sinon pas de trade
  - Sizing : 2% du bankroll par trade (fixe, pas Kelly tant que pas de modèle)

Un trade par event (5 min). Résolution automatique au changement d'event.
Sauvegarde résumé dans data/sim_summary.json pour le dashboard.
"""
from __future__ import annotations

import json
import signal
import sqlite3
import time
from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from config import DB_PATH, INITIAL_CAPITAL
from src.fetchers.binance import BinanceClient
from src.fetchers.polymarket import PolymarketClient

POLL_SEC = 10
SIM_SUMMARY_PATH = Path("data/sim_summary.json")
TRADES_DB = Path("data/trades.db")
POSITION_PCT = 0.02  # 2% du bankroll par trade


@dataclass
class LiveTrade:
    event_slug: str
    opened_at: int
    side: str               # "YES" ou "NO"
    entry_price: float
    size_usd: float
    btc_entry: float
    momentum: float
    imbalance: float
    resolved_at: int | None = None
    outcome: int | None = None   # 1 gagné, 0 perdu
    pnl: float = 0.0
    btc_exit: float | None = None


def init_trades_db() -> None:
    with closing(sqlite3.connect(str(TRADES_DB))) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS trades (
                event_slug TEXT PRIMARY KEY,
                opened_at INTEGER,
                side TEXT,
                entry_price REAL,
                size_usd REAL,
                btc_entry REAL,
                momentum REAL,
                imbalance REAL,
                resolved_at INTEGER,
                outcome INTEGER,
                pnl REAL,
                btc_exit REAL
            );
            """
        )
        conn.commit()


def save_trade(trade: LiveTrade) -> None:
    with closing(sqlite3.connect(str(TRADES_DB))) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO trades VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                trade.event_slug, trade.opened_at, trade.side, trade.entry_price,
                trade.size_usd, trade.btc_entry, trade.momentum, trade.imbalance,
                trade.resolved_at, trade.outcome, trade.pnl, trade.btc_exit,
            ),
        )
        conn.commit()


def compute_momentum(binance: BinanceClient, window_sec: int = 60) -> float:
    """Momentum simple = (prix_now / prix_il_y_a_60s) - 1."""
    klines = binance.get_klines(interval="1m", limit=2)
    if len(klines) < 2:
        return 0.0
    old_close = float(klines[0][4])
    new_close = float(klines[1][4])
    return (new_close - old_close) / old_close


def decide_side(momentum: float, imbalance: float) -> str | None:
    """Heuristique simple : momentum + imbalance alignés."""
    if momentum > 0.0001 and imbalance > 0.05:
        return "YES"
    if momentum < -0.0001 and imbalance < -0.05:
        return "NO"
    return None


def resolve_trade(trade: LiveTrade, btc_exit: float) -> LiveTrade:
    """Résolution : BTC a-t-il monté depuis entry ?"""
    went_up = btc_exit > trade.btc_entry
    won = (trade.side == "YES" and went_up) or (trade.side == "NO" and not went_up)
    trade.outcome = 1 if won else 0
    trade.resolved_at = int(time.time() * 1000)
    trade.btc_exit = btc_exit
    # Simplification : payout 1:1 à la cote d'entrée
    if won:
        payout = trade.size_usd / trade.entry_price
        trade.pnl = payout - trade.size_usd
    else:
        trade.pnl = -trade.size_usd
    return trade


def write_summary(cash: float, trades: list[LiveTrade]) -> None:
    closed = [t for t in trades if t.outcome is not None]
    wins = sum(t.outcome for t in closed) if closed else 0
    total_pnl = sum(t.pnl for t in closed)
    roi = (cash - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    summary = {
        "cash": round(cash, 2),
        "total_trades": len(closed),
        "open_positions": len(trades) - len(closed),
        "win_rate": round(wins / len(closed), 3) if closed else 0,
        "total_pnl": round(total_pnl, 2),
        "roi_pct": round(roi, 2),
        "updated_at": datetime.now().isoformat(),
    }
    SIM_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SIM_SUMMARY_PATH.write_text(json.dumps(summary, indent=2))


def run() -> None:
    init_trades_db()
    binance = BinanceClient()
    poly = PolymarketClient()
    stop = {"flag": False}
    signal.signal(signal.SIGINT, lambda s, f: stop.update(flag=True))

    trades: list[LiveTrade] = []
    cash = INITIAL_CAPITAL
    open_trade: LiveTrade | None = None
    current_slug: str | None = None

    print(f"[sim] démarrage | capital ${cash:.2f} | position {POSITION_PCT*100}% par trade")

    try:
        while not stop["flag"]:
            try:
                event = poly.find_btc_updown_event()
                if event is None:
                    time.sleep(POLL_SEC)
                    continue

                slug = event.get("slug")
                snap = poly.snapshot_event(event)
                if snap is None:
                    time.sleep(POLL_SEC)
                    continue

                # Changement d'event → on résout le trade ouvert
                if open_trade and current_slug and slug != current_slug:
                    btc_now = binance.get_price()
                    open_trade = resolve_trade(open_trade, btc_now)
                    cash += open_trade.size_usd + open_trade.pnl
                    save_trade(open_trade)
                    status = "WIN " if open_trade.outcome else "LOSS"
                    print(
                        f"[{datetime.now():%H:%M:%S}] ✓ {status} "
                        f"{open_trade.side} @{open_trade.entry_price:.3f} → "
                        f"BTC {open_trade.btc_entry:.0f}→{btc_now:.0f} | "
                        f"PnL ${open_trade.pnl:+.2f} | cash ${cash:.2f}"
                    )
                    open_trade = None
                    current_slug = None
                    write_summary(cash, trades)

                # Ouvre un nouveau trade sur le nouvel event
                if open_trade is None and slug != current_slug:
                    momentum = compute_momentum(binance)
                    imbalance = binance.orderbook_imbalance()
                    btc_price = binance.get_price()
                    side = decide_side(momentum, imbalance)

                    if side is None:
                        print(
                            f"[{datetime.now():%H:%M:%S}] skip "
                            f"(momentum={momentum:+.4f} imb={imbalance:+.2f}) "
                            f"event={slug}"
                        )
                        current_slug = slug
                        time.sleep(POLL_SEC)
                        continue

                    entry_price = snap.yes_price if side == "YES" else snap.no_price
                    if entry_price <= 0.02 or entry_price >= 0.98:
                        print(f"[{datetime.now():%H:%M:%S}] skip (prix extrême {entry_price})")
                        current_slug = slug
                        time.sleep(POLL_SEC)
                        continue

                    size_usd = cash * POSITION_PCT
                    cash -= size_usd
                    open_trade = LiveTrade(
                        event_slug=slug,
                        opened_at=int(time.time() * 1000),
                        side=side,
                        entry_price=entry_price,
                        size_usd=size_usd,
                        btc_entry=btc_price,
                        momentum=momentum,
                        imbalance=imbalance,
                    )
                    trades.append(open_trade)
                    current_slug = slug
                    print(
                        f"[{datetime.now():%H:%M:%S}] ▶ BUY {side} @{entry_price:.3f} "
                        f"size=${size_usd:.2f} | BTC={btc_price:.0f} "
                        f"mom={momentum:+.4f} imb={imbalance:+.2f}"
                    )
                    write_summary(cash, trades)

            except Exception as e:
                print(f"[sim] erreur: {e}")
            time.sleep(POLL_SEC)
    finally:
        write_summary(cash, trades)
        binance.close()
        poly.close()


if __name__ == "__main__":
    run()
