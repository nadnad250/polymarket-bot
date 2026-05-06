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
import sys
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (
    DB_PATH,
    INITIAL_CAPITAL,
    LIVE_READY_MAX_SHADOW_DD_PCT,
    LIVE_READY_MIN_SHADOW_ROI_PCT,
    LIVE_READY_MIN_SHADOW_TRADES,
    LIVE_READY_MIN_SHADOW_WIN_RATE,
)
from src.simulator.live_loop import TRADES_DB
from src.simulator.shadow import SHADOW_TRADES_DB

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


def _load_shadow_trades() -> pd.DataFrame:
    if not Path(SHADOW_TRADES_DB).exists():
        return pd.DataFrame()
    with closing(sqlite3.connect(str(SHADOW_TRADES_DB))) as conn:
        return pd.read_sql("SELECT * FROM shadow_trades ORDER BY opened_at DESC", conn)


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


def _max_drawdown_pct(pnls: pd.Series) -> float:
    if pnls.empty:
        return 0.0
    equity = INITIAL_CAPITAL + pnls.cumsum()
    peak = equity.cummax()
    dd = (equity - peak) / peak.replace(0, np.nan)
    return round(abs(float(dd.min() or 0.0)) * 100, 2)


def export_shadow_trades(shadow: pd.DataFrame) -> dict:
    if shadow.empty:
        return {"trades": [], "metrics": _empty_shadow_metrics(), "readiness": _readiness(_empty_shadow_metrics())}

    closed = shadow.dropna(subset=["outcome"]).copy()

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
            "p_up": _safe_round(r["p_up"], 4),
            "edge": _safe_round(r["edge"], 4),
            "score": _safe_round(r["score"], 1),
            "source": str(r["source"]),
            "reason": str(r["decision_reason"] or ""),
            "outcome": _safe_int(r["outcome"]),
            "pnl": _safe_round(r["pnl"], 2) or 0.0,
        }
        for _, r in shadow.head(100).iterrows()
    ]

    if closed.empty:
        metrics = _empty_shadow_metrics()
    else:
        wins = closed[closed["pnl"] > 0]["pnl"].sum()
        losses = abs(closed[closed["pnl"] < 0]["pnl"].sum())
        profit_factor = float(wins / losses) if losses > 0 else (float("inf") if wins > 0 else 0.0)
        total_pnl = float(closed["pnl"].sum())
        metrics = {
            "total_trades": int(len(closed)),
            "open_positions": int(shadow["outcome"].isna().sum()),
            "win_rate": round(float(closed["outcome"].mean()), 3),
            "total_pnl": round(total_pnl, 2),
            "roi_pct": round(total_pnl / INITIAL_CAPITAL * 100, 2),
            "avg_pnl": round(float(closed["pnl"].mean()), 3),
            "avg_edge": round(float(closed["edge"].mean()), 4),
            "avg_score": round(float(closed["score"].mean()), 1),
            "profit_factor": round(profit_factor, 2) if np.isfinite(profit_factor) else 99.0,
            "max_drawdown_pct": _max_drawdown_pct(closed.sort_values("resolved_at")["pnl"]),
            "capital": round(INITIAL_CAPITAL + total_pnl, 2),
        }
    return {"trades": trades_list, "metrics": metrics, "readiness": _readiness(metrics)}


def _empty_shadow_metrics() -> dict:
    return {
        "total_trades": 0,
        "open_positions": 0,
        "win_rate": 0,
        "total_pnl": 0,
        "roi_pct": 0,
        "avg_pnl": 0,
        "avg_edge": 0,
        "avg_score": 0,
        "profit_factor": 0,
        "max_drawdown_pct": 0,
        "capital": INITIAL_CAPITAL,
    }


def _readiness(metrics: dict) -> dict:
    checks = [
        {
            "name": "sample",
            "ok": int(metrics.get("total_trades") or 0) >= LIVE_READY_MIN_SHADOW_TRADES,
            "actual": metrics.get("total_trades", 0),
            "target": LIVE_READY_MIN_SHADOW_TRADES,
        },
        {
            "name": "win_rate",
            "ok": float(metrics.get("win_rate") or 0) >= LIVE_READY_MIN_SHADOW_WIN_RATE,
            "actual": metrics.get("win_rate", 0),
            "target": LIVE_READY_MIN_SHADOW_WIN_RATE,
        },
        {
            "name": "roi",
            "ok": float(metrics.get("roi_pct") or 0) >= LIVE_READY_MIN_SHADOW_ROI_PCT,
            "actual": metrics.get("roi_pct", 0),
            "target": LIVE_READY_MIN_SHADOW_ROI_PCT,
        },
        {
            "name": "drawdown",
            "ok": float(metrics.get("max_drawdown_pct") or 0) <= LIVE_READY_MAX_SHADOW_DD_PCT,
            "actual": metrics.get("max_drawdown_pct", 0),
            "target": LIVE_READY_MAX_SHADOW_DD_PCT,
        },
    ]
    passed = sum(1 for c in checks if c["ok"])
    if passed == len(checks):
        status = "candidate"
        label = "Candidat reel"
    elif passed >= 2:
        status = "watch"
        label = "A surveiller"
    else:
        status = "blocked"
        label = "Demo seulement"
    return {"status": status, "label": label, "passed": passed, "total": len(checks), "checks": checks}


def export_model_metrics() -> dict:
    path = Path("data/model_metrics.json")
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}
    return {}


def export_decision_latest() -> dict:
    path = Path("data/decision_latest.json")
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
    shadow = _load_shadow_trades()
    print(f"[export] ticks={len(ticks)} trades={len(trades)} shadow={len(shadow)}")

    latest = export_latest(ticks)
    ticks_data = export_ticks(ticks)
    trades_data = export_trades(trades)
    shadow_data = export_shadow_trades(shadow)
    model_metrics = export_model_metrics()
    decision_latest = export_decision_latest()

    (PUBLIC_DATA / "latest.json").write_text(json.dumps(latest, indent=2))
    (PUBLIC_DATA / "ticks.json").write_text(json.dumps(ticks_data))
    (PUBLIC_DATA / "trades.json").write_text(json.dumps(trades_data, indent=2))
    (PUBLIC_DATA / "shadow.json").write_text(json.dumps(shadow_data, indent=2))
    (PUBLIC_DATA / "metrics.json").write_text(json.dumps(model_metrics, indent=2))
    (PUBLIC_DATA / "decision.json").write_text(json.dumps(decision_latest, indent=2))

    print(f"[export] ok public/data/ mis a jour ({len(ticks_data)} ticks, {len(trades_data.get('trades', []))} trades)")


if __name__ == "__main__":
    main()
