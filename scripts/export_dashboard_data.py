"""Exporte data pour dashboard static (GitHub Pages).

Génère des fichiers JSON lisibles côté navigateur :
- public/data/latest.json    — dernier tick + résumé
- public/data/ticks.json     — 24h de ticks (downsamplé)
- public/data/trades.json    — tous les trades + métriques
- public/data/metrics.json   — métriques modèle courant
- public/data/equity.json    — courbe de capital
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from config import DB_PATH, INITIAL_CAPITAL
from src.simulator.live_loop import TRADES_DB

PUBLIC_DATA = Path("public/data")
PUBLIC_DATA.mkdir(parents=True, exist_ok=True)


def _load_ticks(hours: int = 24) -> pd.DataFrame:
    if not Path(DB_PATH).exists():
        return pd.DataFrame()
    with closing(sqlite3.connect(str(DB_PATH))) as conn:
        since = int(datetime.now(tz=timezone.utc).timestamp() * 1000) - hours * 3600 * 1000
        df = pd.read_sql(
            "SELECT * FROM ticks WHERE ts > ? ORDER BY ts ASC",
            conn, params=(since,),
        )
    return df


def _load_trades() -> pd.DataFrame:
    if not Path(TRADES_DB).exists():
        return pd.DataFrame()
    with closing(sqlite3.connect(str(TRADES_DB))) as conn:
        return pd.read_sql("SELECT * FROM trades ORDER BY opened_at DESC", conn)


def _downsample(df: pd.DataFrame, target_points: int = 500) -> pd.DataFrame:
    if len(df) <= target_points:
        return df
    step = max(1, len(df) // target_points)
    return df.iloc[::step].reset_index(drop=True)


def export_latest(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    last = df.iloc[-1]
    return {
        "ts": int(last["ts"]),
        "btc_price": float(last["btc_price"]),
        "poly_yes": float(last["poly_yes"] or 0.5),
        "poly_no": float(last["poly_no"] or 0.5),
        "poly_market": str(last["poly_market"]),
        "poly_question": str(last["poly_question"] or ""),
        "ob_imb": float(last["btc_ob_imb"] or 0),
        "spread_bps": round(((last["btc_ask"] or 0) - (last["btc_bid"] or 0)) / (last["btc_price"] or 1) * 10000, 2),
        "updated_at": datetime.now(tz=timezone.utc).isoformat(),
    }


def export_ticks(df: pd.DataFrame) -> list[dict]:
    if df.empty:
        return []
    df = _downsample(df, target_points=600)
    return [
        {
            "ts": int(r["ts"]),
            "btc": round(float(r["btc_price"]), 2),
            "yes": round(float(r["poly_yes"] or 0.5), 4),
        }
        for _, r in df.iterrows()
    ]


def export_trades(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"trades": [], "metrics": {}}
    closed = trades.dropna(subset=["outcome"])

    def _safe_int(v):
        return int(v) if pd.notna(v) else None

    def _safe_round(v, n=2):
        return round(float(v), n) if pd.notna(v) else None

    trades_list = [
        {
            "event": str(r["event_slug"]),
            "ts": int(r["opened_at"]),
            "side": str(r["side"]),
            "entry": _safe_round(r["entry_price"], 4),
            "size": _safe_round(r["size_usd"], 2),
            "btc_entry": _safe_round(r["btc_entry"], 2),
            "btc_exit": _safe_round(r["btc_exit"], 2),
            "momentum": _safe_round(r["momentum"], 5),
            "imbalance": _safe_round(r["imbalance"], 3),
            "outcome": _safe_int(r["outcome"]),
            "pnl": _safe_round(r["pnl"], 2) or 0.0,
        }
        for _, r in trades.head(100).iterrows()
    ]

    # Equity curve reconstruction
    closed_sorted = closed.sort_values("resolved_at")
    equity = []
    eq = INITIAL_CAPITAL
    for _, r in closed_sorted.iterrows():
        eq += float(r["pnl"])
        equity.append({"ts": int(r["resolved_at"]), "equity": round(eq, 2)})

    metrics = {
        "total_trades": int(len(closed)),
        "win_rate": round(float(closed["outcome"].mean()), 3) if len(closed) else 0,
        "total_pnl": round(float(closed["pnl"].sum()), 2),
        "best_trade": round(float(closed["pnl"].max()), 2) if len(closed) else 0,
        "worst_trade": round(float(closed["pnl"].min()), 2) if len(closed) else 0,
        "avg_pnl": round(float(closed["pnl"].mean()), 2) if len(closed) else 0,
        "roi_pct": round(float(closed["pnl"].sum()) / INITIAL_CAPITAL * 100, 2),
        "capital": round(INITIAL_CAPITAL + float(closed["pnl"].sum()), 2),
    }
    return {"trades": trades_list, "metrics": metrics, "equity": equity}


def export_model_metrics() -> dict:
    path = Path("data/model_metrics.json")
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}
    return {}


def main() -> None:
    print("[export] chargement data...")
    ticks = _load_ticks(hours=24)
    trades = _load_trades()
    print(f"[export] ticks={len(ticks)} trades={len(trades)}")

    latest = export_latest(ticks)
    ticks_data = export_ticks(ticks)
    trades_data = export_trades(trades)
    model_metrics = export_model_metrics()

    (PUBLIC_DATA / "latest.json").write_text(json.dumps(latest, indent=2))
    (PUBLIC_DATA / "ticks.json").write_text(json.dumps(ticks_data))
    (PUBLIC_DATA / "trades.json").write_text(json.dumps(trades_data, indent=2))
    (PUBLIC_DATA / "metrics.json").write_text(json.dumps(model_metrics, indent=2))

    print(f"[export] ✓ public/data/ mis à jour ({len(ticks_data)} ticks, {len(trades_data.get('trades', []))} trades)")


if __name__ == "__main__":
    main()
