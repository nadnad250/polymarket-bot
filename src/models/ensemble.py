"""Ensemble — LightGBM + XGBoost + LSTM avec moyenne pondérée.

Agrégation des probas → meilleure calibration + robustesse.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from src.models.features import FEATURE_COLS, build_features, get_xy


@dataclass
class EnsembleResult:
    lgbm: object
    xgb: object
    lstm: object | None
    weights: dict
    metrics: dict


def _train_lgbm(X_tr, y_tr):
    base = lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.03, num_leaves=31,
        min_child_samples=20, reg_alpha=0.1, reg_lambda=0.1,
        objective="binary", verbose=-1,
    )
    clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
    clf.fit(X_tr, y_tr)
    return clf


def _train_xgb(X_tr, y_tr):
    base = xgb.XGBClassifier(
        n_estimators=400, learning_rate=0.03, max_depth=5,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", verbosity=0,
    )
    clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
    clf.fit(X_tr, y_tr)
    return clf


def train_ensemble(df_features: pd.DataFrame, use_lstm: bool = True) -> EnsembleResult:
    X, y = get_xy(df_features)
    if len(X) < 150:
        raise ValueError(f"Pas assez de data ensemble ({len(X)} < 150)")

    split = int(len(X) * 0.8)
    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    y_tr, y_te = y.iloc[:split], y.iloc[split:]

    print("[ensemble] entraînement LightGBM...")
    m_lgbm = _train_lgbm(X_tr, y_tr)
    p_lgbm = m_lgbm.predict_proba(X_te)[:, 1]

    print("[ensemble] entraînement XGBoost...")
    m_xgb = _train_xgb(X_tr, y_tr)
    p_xgb = m_xgb.predict_proba(X_te)[:, 1]

    m_lstm = None
    p_lstm = None
    if use_lstm:
        try:
            from src.models.lstm import train_lstm, make_windows
            print("[ensemble] entraînement LSTM...")
            m_lstm, _ = train_lstm(df_features, epochs=20)
            X_arr = X.to_numpy(dtype=np.float32)
            Xw, _ = make_windows(X_arr, y.to_numpy(dtype=np.float32), window=m_lstm.window)
            split_w = int(len(Xw) * 0.8)
            p_lstm_full = m_lstm.predict_proba_window(Xw[split_w:])
            # Align sur X_te (peut différer de longueur)
            n = min(len(p_lstm_full), len(p_lgbm))
            p_lstm = p_lstm_full[-n:]
            p_lgbm = p_lgbm[-n:]
            p_xgb = p_xgb[-n:]
            y_te = y_te.iloc[-n:]
        except Exception as e:
            print(f"[ensemble] LSTM skipped: {e}")
            m_lstm = None

    # Pondération optimale : minimise brier sur test
    if m_lstm is not None and p_lstm is not None:
        weights = _optimize_weights([p_lgbm, p_xgb, p_lstm], y_te.to_numpy())
        p_ens = weights[0] * p_lgbm + weights[1] * p_xgb + weights[2] * p_lstm
        w_dict = {"lgbm": weights[0], "xgb": weights[1], "lstm": weights[2]}
    else:
        weights = _optimize_weights([p_lgbm, p_xgb], y_te.to_numpy())
        p_ens = weights[0] * p_lgbm + weights[1] * p_xgb
        w_dict = {"lgbm": weights[0], "xgb": weights[1]}

    preds = (p_ens > 0.5).astype(int)
    metrics = {
        "n_train": len(X_tr), "n_test": len(y_te),
        "accuracy": float((preds == y_te.values).mean()),
        "auc": float(roc_auc_score(y_te, p_ens)) if y_te.nunique() > 1 else None,
        "brier": float(brier_score_loss(y_te, p_ens)),
        "logloss": float(log_loss(y_te, p_ens.clip(1e-6, 1 - 1e-6))),
        "base_rate": float(y_tr.mean()),
        "brier_lgbm": float(brier_score_loss(y_te, p_lgbm)),
        "brier_xgb": float(brier_score_loss(y_te, p_xgb)),
    }
    if p_lstm is not None:
        metrics["brier_lstm"] = float(brier_score_loss(y_te, p_lstm))

    return EnsembleResult(lgbm=m_lgbm, xgb=m_xgb, lstm=m_lstm, weights=w_dict, metrics=metrics)


def _optimize_weights(probs_list: list[np.ndarray], y: np.ndarray) -> list[float]:
    """Grid search simple sur les poids."""
    best_score = float("inf")
    best = [1.0 / len(probs_list)] * len(probs_list)
    n = len(probs_list)

    if n == 2:
        for w1 in np.arange(0.1, 1.0, 0.1):
            w2 = 1 - w1
            p = w1 * probs_list[0] + w2 * probs_list[1]
            s = brier_score_loss(y, p)
            if s < best_score:
                best_score = s
                best = [w1, w2]
    else:
        for w1 in np.arange(0.1, 0.9, 0.1):
            for w2 in np.arange(0.1, 1.0 - w1, 0.1):
                w3 = 1 - w1 - w2
                if w3 < 0.05:
                    continue
                p = w1 * probs_list[0] + w2 * probs_list[1] + w3 * probs_list[2]
                s = brier_score_loss(y, p)
                if s < best_score:
                    best_score = s
                    best = [w1, w2, w3]
    return best


def save_ensemble(result: EnsembleResult, path: str | Path) -> None:
    """Sauvegarde l'ensemble — tensors LSTM convertis en numpy pour torch-free load."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lstm_state = None
    if result.lstm is not None:
        # Convertit tensors en numpy → pickle peut se charger sans torch
        sd = {k: v.detach().cpu().numpy() for k, v in result.lstm.net.state_dict().items()}
        lstm_state = {
            "state_dict_np": sd,
            "window": result.lstm.window,
            "feature_cols": result.lstm.feature_cols,
        }
    with open(p, "wb") as f:
        pickle.dump({
            "lgbm": result.lgbm,
            "xgb": result.xgb,
            "lstm_state": lstm_state,
            "weights": result.weights,
            "metrics": result.metrics,
        }, f)


def predict_proba(ensemble_payload: dict, X: pd.DataFrame) -> np.ndarray:
    """Inférence ensemble — torch est optionnel (CI lean sans torch).

    Si torch n'est pas dispo et qu'un poids LSTM existe, on renormalise les
    poids LGBM + XGB pour qu'ils somment à 1.
    """
    X = X[FEATURE_COLS].replace([np.inf, -np.inf], 0).fillna(0)
    p_lgbm = ensemble_payload["lgbm"].predict_proba(X)[:, 1]
    p_xgb = ensemble_payload["xgb"].predict_proba(X)[:, 1]
    w = ensemble_payload["weights"]
    w_lgbm = float(w.get("lgbm", 0.5))
    w_xgb = float(w.get("xgb", 0.5))
    w_lstm = float(w.get("lstm", 0.0))

    # Tente d'utiliser le LSTM si présent + torch dispo
    probs_lstm = None
    if ensemble_payload.get("lstm_state") and w_lstm > 0:
        try:
            import torch
            from src.models.lstm import LSTMClassifier, make_windows
            state = ensemble_payload["lstm_state"]
            net = LSTMClassifier(n_features=len(state["feature_cols"]))
            # Reconvertit numpy → tensors
            sd_np = state.get("state_dict_np") or state.get("state_dict") or {}
            sd = {k: torch.from_numpy(v) if hasattr(v, "shape") else v for k, v in sd_np.items()}
            net.load_state_dict(sd)
            net.eval()
            X_arr = X.to_numpy(dtype=np.float32)
            Xw, _ = make_windows(X_arr, np.zeros(len(X_arr), dtype=np.float32), window=state["window"])
            if len(Xw) > 0:
                with torch.no_grad():
                    probs_lstm = torch.sigmoid(net(torch.from_numpy(Xw))).numpy()
                pad = np.full(len(X) - len(probs_lstm), 0.5)
                probs_lstm = np.concatenate([pad, probs_lstm])
        except ImportError:
            print("[ensemble] torch absent — LSTM ignoré, poids renormalisés sur LGBM+XGB")
            probs_lstm = None
        except Exception as e:
            print(f"[ensemble] LSTM inference failed: {e} — fallback LGBM+XGB")
            probs_lstm = None

    if probs_lstm is not None:
        return w_lgbm * p_lgbm + w_xgb * p_xgb + w_lstm * probs_lstm

    # Fallback : renormalise sans LSTM
    total = w_lgbm + w_xgb
    if total <= 0:
        return 0.5 * p_lgbm + 0.5 * p_xgb
    return (w_lgbm / total) * p_lgbm + (w_xgb / total) * p_xgb
