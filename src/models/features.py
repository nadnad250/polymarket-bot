"""Feature engineering pour prédiction BTC 5min up/down.

Entrée  : DataFrame ticks (ts, btc_price, btc_bid, btc_ask, btc_ob_imb, poly_yes, ...)
Sortie  : DataFrame features + label (1 si BTC monte sur horizon, 0 sinon)

Features utilisées :
- Returns multi-horizons (30s, 1m, 3m, 5m)
- Volatilité réalisée (std des returns)
- Momentum / RSI simplifié
- Orderbook imbalance (instantané + moyennes)
- Spread Binance
- Probabilité implicite Polymarket (signal de la foule)
- Gap prob. marché vs momentum (détecte mispricing)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


HORIZON_SEC = 300  # 5 minutes


def _rolling_std(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).std().fillna(0)


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0).rolling(window, min_periods=1).mean()
    down = (-delta.clip(upper=0)).rolling(window, min_periods=1).mean()
    rs = up / down.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def build_features(df: pd.DataFrame, poll_sec: int = 5) -> pd.DataFrame:
    """Construit les features depuis un DataFrame de ticks bruts.

    Le DataFrame doit être trié par ts croissant et contenir :
    ts, btc_price, btc_bid, btc_ask, btc_ob_imb, poly_yes, poly_no
    """
    df = df.sort_values("ts").reset_index(drop=True).copy()
    df["ts"] = pd.to_numeric(df["ts"])
    df["dt"] = pd.to_datetime(df["ts"], unit="ms")

    # --- Returns multi-horizons ---
    for sec in (30, 60, 180, 300):
        n = max(1, sec // poll_sec)
        df[f"ret_{sec}s"] = df["btc_price"].pct_change(n).fillna(0)

    # --- Volatilité ---
    for sec in (60, 300):
        n = max(1, sec // poll_sec)
        df[f"vol_{sec}s"] = _rolling_std(df["ret_30s"], n)

    # --- Momentum ---
    df["rsi_14"] = _rsi(df["btc_price"], 14)
    df["mom_1m"] = df["btc_price"] - df["btc_price"].shift(60 // poll_sec).fillna(df["btc_price"])
    df["mom_5m"] = df["btc_price"] - df["btc_price"].shift(300 // poll_sec).fillna(df["btc_price"])

    # --- Orderbook / microstructure ---
    df["ob_imb"] = df["btc_ob_imb"].fillna(0)
    df["ob_imb_avg_30s"] = df["ob_imb"].rolling(30 // poll_sec, min_periods=1).mean()
    df["spread"] = (df["btc_ask"] - df["btc_bid"]).fillna(0)
    df["spread_pct"] = df["spread"] / df["btc_price"]

    # --- Signal Polymarket (la foule) ---
    df["poly_yes"] = df["poly_yes"].fillna(0.5)
    df["poly_edge_vs_5050"] = df["poly_yes"] - 0.5

    # --- Mispricing : le marché dit X mais le momentum dit Y ---
    df["poly_vs_momentum"] = df["poly_yes"] - (df["ret_60s"] * 10 + 0.5).clip(0, 1)

    # --- Label : BTC a-t-il monté dans HORIZON_SEC secondes ? ---
    shift_n = HORIZON_SEC // poll_sec
    df["future_price"] = df["btc_price"].shift(-shift_n)
    df["label"] = (df["future_price"] > df["btc_price"]).astype(int)
    df["future_return"] = (df["future_price"] - df["btc_price"]) / df["btc_price"]

    return df


FEATURE_COLS = [
    "ret_30s", "ret_60s", "ret_180s", "ret_300s",
    "vol_60s", "vol_300s",
    "rsi_14", "mom_1m", "mom_5m",
    "ob_imb", "ob_imb_avg_30s", "spread_pct",
    "poly_yes", "poly_edge_vs_5050", "poly_vs_momentum",
]


def get_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Extrait X, y prêts pour training en gardant uniquement lignes avec label."""
    clean = df.dropna(subset=["future_price"]).copy()
    X = clean[FEATURE_COLS].replace([np.inf, -np.inf], 0).fillna(0)
    y = clean["label"]
    return X, y
