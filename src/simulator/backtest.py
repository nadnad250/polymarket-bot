"""Backtester walk-forward avec frais réalistes.

Simule le bot sur data historique :
- Entraîne sur fenêtre passée
- Trade la fenêtre suivante
- Fait rouler sur toute la data
- Calcule courbe de capital, drawdown, Sharpe, win rate
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from config import INITIAL_CAPITAL, KELLY_FRACTION, MAX_POSITION_PCT
from src.models.ensemble import predict_proba, train_ensemble
from src.models.features import FEATURE_COLS, build_features
from src.simulator.fees import DEFAULT_FEES, FeeModel


@dataclass
class BacktestResult:
    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict


def kelly(edge: float, price: float, fraction: float = KELLY_FRACTION) -> float:
    if edge <= 0 or price <= 0 or price >= 1:
        return 0.0
    b = (1 - price) / price
    p = price + edge
    q = 1 - p
    k = (b * p - q) / b
    return max(0.0, min(k * fraction, MAX_POSITION_PCT))


def backtest(
    df_ticks: pd.DataFrame,
    min_edge: float = 0.02,
    initial_capital: float = INITIAL_CAPITAL,
    fees: FeeModel = DEFAULT_FEES,
    n_retrains: int = 3,
) -> BacktestResult:
    """Walk-forward backtest avec re-entraînement périodique."""
    feats = build_features(df_ticks)
    feats = feats.dropna(subset=["future_price"]).reset_index(drop=True)
    if len(feats) < 300:
        raise ValueError(f"Pas assez de data pour backtest ({len(feats)})")

    # Découpe en N+1 blocs : train sur 1er, test sur suivant, retrain, etc.
    block = len(feats) // (n_retrains + 1)
    cash = initial_capital
    equity = [{"ts": int(feats.iloc[0]["ts"]), "cash": cash, "equity": cash}]
    trades = []

    for k in range(n_retrains):
        train_end = block * (k + 1)
        test_end = min(train_end + block, len(feats))
        train_df = feats.iloc[:train_end]
        test_df = feats.iloc[train_end:test_end]
        if len(test_df) == 0:
            break

        print(f"[backtest] block {k+1}/{n_retrains} — train={len(train_df)} test={len(test_df)}")
        try:
            result = train_ensemble(train_df, use_lstm=False)  # LSTM off pour vitesse
        except Exception as e:
            print(f"  skip train: {e}")
            continue

        payload = {
            "lgbm": result.lgbm, "xgb": result.xgb,
            "lstm_state": None, "weights": result.weights,
        }

        X_test = test_df[FEATURE_COLS].replace([np.inf, -np.inf], 0).fillna(0)
        probs = predict_proba(payload, X_test)

        for i, (_, row) in enumerate(test_df.iterrows()):
            p_up = float(probs[i])
            poly_yes = float(row.get("poly_yes", 0.5))
            # Side : achète le côté qui a l'edge max
            edge_yes = p_up - poly_yes
            edge_no = (1 - p_up) - (1 - poly_yes)
            if edge_yes > edge_no:
                side, price, edge = "YES", poly_yes, edge_yes
            else:
                side, price, edge = "NO", 1 - poly_yes, edge_no

            if edge < min_edge:
                equity.append({"ts": int(row["ts"]), "cash": cash, "equity": cash})
                continue

            size_pct = kelly(edge, price)
            if size_pct == 0:
                equity.append({"ts": int(row["ts"]), "cash": cash, "equity": cash})
                continue

            size_usd = cash * size_pct
            eff_price, entry_fees = fees.apply_entry(price, size_usd)
            cash -= size_usd + entry_fees

            went_up = bool(row["label"])
            won = (side == "YES" and went_up) or (side == "NO" and not went_up)
            pnl = fees.net_pnl(won, size_usd, eff_price)
            cash += size_usd + pnl + entry_fees  # refund size + pnl net

            trades.append({
                "ts": int(row["ts"]),
                "side": side,
                "entry_price": eff_price,
                "market_price": price,
                "model_prob": p_up,
                "edge": edge,
                "size_usd": size_usd,
                "outcome": int(won),
                "pnl": pnl,
                "cash_after": cash,
            })
            equity.append({"ts": int(row["ts"]), "cash": cash, "equity": cash})

    eq_df = pd.DataFrame(equity)
    tr_df = pd.DataFrame(trades)

    metrics = _compute_metrics(eq_df, tr_df, initial_capital)
    return BacktestResult(equity_curve=eq_df, trades=tr_df, metrics=metrics)


def _compute_metrics(eq: pd.DataFrame, trades: pd.DataFrame, capital_0: float) -> dict:
    if len(trades) == 0:
        return {"n_trades": 0, "roi_pct": 0, "win_rate": 0}
    returns = eq["equity"].pct_change().dropna()
    peak = eq["equity"].cummax()
    dd = (eq["equity"] - peak) / peak
    max_dd = float(dd.min()) if len(dd) else 0
    sharpe = float(returns.mean() / returns.std() * np.sqrt(365 * 24 * 12)) if returns.std() > 0 else 0
    return {
        "n_trades": int(len(trades)),
        "win_rate": round(float(trades["outcome"].mean()), 3),
        "avg_edge": round(float(trades["edge"].mean()), 4),
        "avg_pnl": round(float(trades["pnl"].mean()), 3),
        "total_pnl": round(float(trades["pnl"].sum()), 2),
        "roi_pct": round((float(eq["equity"].iloc[-1]) - capital_0) / capital_0 * 100, 2),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "sharpe_annualized": round(sharpe, 2),
        "final_equity": round(float(eq["equity"].iloc[-1]), 2),
    }


def save_backtest(result: BacktestResult, outdir: str | Path) -> None:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    result.equity_curve.to_csv(outdir / "equity.csv", index=False)
    result.trades.to_csv(outdir / "trades.csv", index=False)
    (outdir / "metrics.json").write_text(json.dumps(result.metrics, indent=2))
