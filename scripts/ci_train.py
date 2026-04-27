"""Entraînement CI — training complet ensemble ML puis commit.

Lancé toutes les 6h par `.github/workflows/train.yml`.
- Charge tous les ticks de la DB
- Entraîne LightGBM + XGBoost + LSTM ensemble
- Sauvegarde modèle + metrics JSON
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config import DB_PATH
from src.models.ensemble import save_ensemble, train_ensemble
from src.models.features import build_features

MIN_TICKS = 100   # on attend au moins 100 ticks avant de train (LSTM désactivé sous 200)
LSTM_MIN_TICKS = 250
MODEL_PATH = Path("data/model_ensemble.pkl")
METRICS_PATH = Path("data/model_metrics.json")
PUBLIC_METRICS_PATH = Path("public/data/metrics.json")


def main() -> None:
    if not Path(DB_PATH).exists():
        print("[train] DB absente, skip.")
        return

    with closing(sqlite3.connect(str(DB_PATH))) as conn:
        df = pd.read_sql("SELECT * FROM ticks ORDER BY ts", conn)

    print(f"[train] {len(df)} ticks chargés")
    if len(df) < MIN_TICKS:
        print(f"[train] pas assez de ticks ({len(df)} < {MIN_TICKS}), skip.")
        return

    feats = build_features(df)
    print(f"[train] features shape: {feats.shape}")

    use_lstm = len(df) >= LSTM_MIN_TICKS
    result = train_ensemble(feats, use_lstm=use_lstm)
    print(f"[train] ✓ métriques: {result.metrics}")
    print(f"[train] ✓ poids ensemble: {result.weights}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_ensemble(result, MODEL_PATH)

    payload = {
        "trained_at": datetime.now(tz=timezone.utc).isoformat(),
        "metrics": result.metrics,
        "weights": result.weights,
        "n_ticks_used": len(df),
    }
    METRICS_PATH.write_text(json.dumps(payload, indent=2))
    PUBLIC_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PUBLIC_METRICS_PATH.write_text(json.dumps(payload, indent=2))

    print(f"[train] ✓ sauvegardé {MODEL_PATH} + {METRICS_PATH}")


if __name__ == "__main__":
    main()
