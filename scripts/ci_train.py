"""Entraînement CI — training complet ensemble ML puis commit.

Lancé toutes les 6h par `.github/workflows/train.yml`.
- Charge tous les ticks de la DB
- Entraîne LightGBM + XGBoost + LSTM ensemble
- Sauvegarde modèle + metrics JSON
"""
from __future__ import annotations

import json
import sqlite3
import sys
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (
    DB_PATH,
    MAX_MODEL_BRIER,
    MAX_MODEL_LOGLOSS,
    MIN_MODEL_AUC,
    MIN_MODEL_TEST_ROWS,
    MIN_TRAIN_LABELS,
)
from src.models.ensemble import save_ensemble, train_ensemble
from src.models.features import build_features, get_xy

MIN_TICKS = 100
LSTM_MIN_LABELS = 2000
MODEL_PATH = Path("data/model_ensemble.pkl")
METRICS_PATH = Path("data/model_metrics.json")
PUBLIC_METRICS_PATH = Path("public/data/metrics.json")


def _is_tradeable(metrics: dict) -> tuple[bool, str]:
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


def _write_metrics(payload: dict) -> None:
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PUBLIC_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2)
    METRICS_PATH.write_text(text)
    PUBLIC_METRICS_PATH.write_text(text)


def _write_skip(reason: str, n_ticks: int, n_labels: int = 0) -> None:
    payload = {
        "trained_at": datetime.now(tz=timezone.utc).isoformat(),
        "metrics": {
            "tradeable": False,
            "trade_block_reason": reason,
            "n_trainable_labels": n_labels,
        },
        "weights": {},
        "n_ticks_used": n_ticks,
    }
    _write_metrics(payload)
    print(f"[train] skip: {reason}")


def main() -> None:
    if not Path(DB_PATH).exists():
        print("[train] DB absente, skip.")
        return

    with closing(sqlite3.connect(str(DB_PATH))) as conn:
        df = pd.read_sql("SELECT * FROM ticks ORDER BY ts", conn)

    print(f"[train] {len(df)} ticks chargés")
    if len(df) < MIN_TICKS:
        _write_skip(f"pas assez de ticks ({len(df)} < {MIN_TICKS})", len(df))
        return

    feats = build_features(df)
    print(f"[train] features shape: {feats.shape}")
    X, _ = get_xy(feats)
    n_labels = len(X)
    print(f"[train] labels valides: {n_labels}")
    if n_labels < MIN_TRAIN_LABELS:
        _write_skip(
            f"pas assez de labels valides ({n_labels} < {MIN_TRAIN_LABELS})",
            len(df),
            n_labels,
        )
        return

    use_lstm = n_labels >= LSTM_MIN_LABELS
    try:
        result = train_ensemble(feats, use_lstm=use_lstm)
    except Exception as e:
        _write_skip(f"training failed: {e}", len(df), n_labels)
        return
    tradeable, reason = _is_tradeable(result.metrics)
    result.metrics["tradeable"] = bool(tradeable)
    result.metrics["trade_block_reason"] = reason
    print(f"[train] ok metrics: {result.metrics}")
    print(f"[train] ok weights: {result.weights}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_ensemble(result, MODEL_PATH)

    payload = {
        "trained_at": datetime.now(tz=timezone.utc).isoformat(),
        "metrics": result.metrics,
        "weights": result.weights,
        "n_ticks_used": len(df),
        "n_trainable_labels": n_labels,
    }
    _write_metrics(payload)

    print(f"[train] ok saved {MODEL_PATH} + {METRICS_PATH}")


if __name__ == "__main__":
    main()
