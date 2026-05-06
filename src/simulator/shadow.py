"""Separate paper-only shadow trades for strategy research."""
from __future__ import annotations

import sqlite3
import time
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from config import INITIAL_CAPITAL, SHADOW_POSITION_PCT
from src.simulator.fees import DEFAULT_FEES

SHADOW_TRADES_DB = Path("data/shadow_trades.db")


@dataclass
class ShadowTrade:
    event_slug: str
    opened_at: int
    side: str
    entry_price: float
    size_usd: float
    btc_entry: float
    p_up: float | None
    edge: float
    score: float
    source: str
    decision_reason: str
    resolved_at: int | None = None
    outcome: int | None = None
    pnl: float = 0.0
    btc_exit: float | None = None


def init_shadow_db() -> None:
    SHADOW_TRADES_DB.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(str(SHADOW_TRADES_DB))) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS shadow_trades (
                event_slug TEXT PRIMARY KEY,
                opened_at INTEGER,
                side TEXT,
                entry_price REAL,
                size_usd REAL,
                btc_entry REAL,
                p_up REAL,
                edge REAL,
                score REAL,
                source TEXT,
                decision_reason TEXT,
                resolved_at INTEGER,
                outcome INTEGER,
                pnl REAL,
                btc_exit REAL
            );
            """
        )
        conn.commit()


def save_shadow_trade(trade: ShadowTrade) -> None:
    init_shadow_db()
    with closing(sqlite3.connect(str(SHADOW_TRADES_DB))) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO shadow_trades VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                trade.event_slug,
                trade.opened_at,
                trade.side,
                trade.entry_price,
                trade.size_usd,
                trade.btc_entry,
                trade.p_up,
                trade.edge,
                trade.score,
                trade.source,
                trade.decision_reason,
                trade.resolved_at,
                trade.outcome,
                trade.pnl,
                trade.btc_exit,
            ),
        )
        conn.commit()


def open_shadow_trade_for(event_slug: str) -> ShadowTrade | None:
    if not SHADOW_TRADES_DB.exists():
        return None
    with closing(sqlite3.connect(str(SHADOW_TRADES_DB))) as conn:
        row = conn.execute(
            "SELECT * FROM shadow_trades WHERE event_slug=? AND outcome IS NULL",
            (event_slug,),
        ).fetchone()
    return _from_row(row) if row else None


def open_shadow_trades() -> list[ShadowTrade]:
    if not SHADOW_TRADES_DB.exists():
        return []
    with closing(sqlite3.connect(str(SHADOW_TRADES_DB))) as conn:
        rows = conn.execute("SELECT * FROM shadow_trades WHERE outcome IS NULL").fetchall()
    return [_from_row(row) for row in rows]


def all_shadow_trades() -> list[ShadowTrade]:
    if not SHADOW_TRADES_DB.exists():
        return []
    with closing(sqlite3.connect(str(SHADOW_TRADES_DB))) as conn:
        rows = conn.execute("SELECT * FROM shadow_trades ORDER BY opened_at DESC").fetchall()
    return [_from_row(row) for row in rows]


def resolve_shadow_trade(trade: ShadowTrade, btc_exit: float) -> ShadowTrade:
    went_up = btc_exit > trade.btc_entry
    won = (trade.side == "YES" and went_up) or (trade.side == "NO" and not went_up)
    trade.outcome = 1 if won else 0
    trade.resolved_at = int(time.time() * 1000)
    trade.btc_exit = btc_exit
    gas = DEFAULT_FEES.gas_fee_usd * 2
    if won:
        payout = trade.size_usd / trade.entry_price
        trade.pnl = payout - trade.size_usd - gas
    else:
        trade.pnl = -trade.size_usd - gas
    return trade


def default_shadow_size(cash: float | None = None) -> float:
    base = INITIAL_CAPITAL if cash is None else max(0.0, cash)
    return max(0.0, base * SHADOW_POSITION_PCT)


def _from_row(row) -> ShadowTrade:
    return ShadowTrade(
        event_slug=row[0],
        opened_at=row[1],
        side=row[2],
        entry_price=row[3],
        size_usd=row[4],
        btc_entry=row[5],
        p_up=row[6],
        edge=row[7],
        score=row[8],
        source=row[9],
        decision_reason=row[10],
        resolved_at=row[11],
        outcome=row[12],
        pnl=row[13],
        btc_exit=row[14],
    )
