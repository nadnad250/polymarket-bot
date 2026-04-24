"""Baseline LightGBM — classifier binaire up/down + calibration probabilité.

Walk-forward split (pas de data leakage) + calibration isotonique pour obtenir
des probabilités utilisables par le simulateur Kelly.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from src.models.features import FEATURE_COLS, build_features, get_xy


@dataclass
class TrainResult:
    model: object
    metrics: dict
    feature_importance: pd.DataFrame


def walk_forward_split(
    X: pd.DataFrame, y: pd.Series, n_splits: int = 5
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Retourne des (train_idx, test_idx) séquentiels, pas aléatoires.

    Critique pour les séries temporelles : on entraîne sur passé, on teste sur futur.
    """
    n = len(X)
    fold_size = n // (n_splits + 1)
    splits = []
    for i in range(n_splits):
        train_end = fold_size * (i + 1)
        test_end = train_end + fold_size
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(train_end, min(test_end, n))
        if len(test_idx) == 0:
            break
        splits.append((train_idx, test_idx))
    return splits


def train(
    df_features: pd.DataFrame,
    calibrate: bool = True,
    n_estimators: int = 300,
) -> TrainResult:
    """Entraîne un LightGBM classifier avec calibration isotonique."""
    X, y = get_xy(df_features)
    if len(X) < 100:
        raise ValueError(
            f"Pas assez de données pour entraîner ({len(X)} lignes). "
            "Collecte au moins quelques heures de ticks avant."
        )

    # Last 20% = test hold-out
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    base = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=0.1,
        objective="binary",
        verbose=-1,
    )

    if calibrate:
        model = CalibratedClassifierCV(base, method="isotonic", cv=3)
    else:
        model = base

    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs > 0.5).astype(int)

    metrics = {
        "n_train": len(X_train),
        "n_test": len(X_test),
        "accuracy": float((preds == y_test.values).mean()),
        "auc": float(roc_auc_score(y_test, probs)) if y_test.nunique() > 1 else None,
        "logloss": float(log_loss(y_test, probs.clip(1e-6, 1 - 1e-6))),
        "brier": float(brier_score_loss(y_test, probs)),
        "base_rate": float(y_train.mean()),
    }

    if hasattr(base, "booster_"):
        importance = base.booster_.feature_importance(importance_type="gain")
    elif calibrate:
        est = model.calibrated_classifiers_[0].estimator
        importance = est.booster_.feature_importance(importance_type="gain")
    else:
        importance = np.zeros(len(FEATURE_COLS))

    fi = pd.DataFrame(
        {"feature": FEATURE_COLS, "gain": importance}
    ).sort_values("gain", ascending=False).reset_index(drop=True)

    return TrainResult(model=model, metrics=metrics, feature_importance=fi)


def save_model(model: object, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path: str | Path) -> object:
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_proba(model: object, df_features: pd.DataFrame) -> np.ndarray:
    """Prédit P(up) pour chaque ligne."""
    X = df_features[FEATURE_COLS].replace([np.inf, -np.inf], 0).fillna(0)
    return model.predict_proba(X)[:, 1]


if __name__ == "__main__":
    import sqlite3
    from config import DB_PATH

    conn = sqlite3.connect(str(DB_PATH))
    df = pd.read_sql("SELECT * FROM ticks ORDER BY ts", conn)
    conn.close()
    print(f"Ticks chargés: {len(df)}")

    feats = build_features(df)
    print(f"Features construites: {feats.shape}")

    result = train(feats)
    print(f"\nMétriques: {result.metrics}")
    print(f"\nTop features:\n{result.feature_importance.head(10)}")

    save_model(result.model, "data/model_lgbm.pkl")
    print("\nModèle sauvegardé dans data/model_lgbm.pkl")
