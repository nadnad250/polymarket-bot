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
- Bandes de Bollinger (distance bornes upper/lower)
- MACD (12-26) + signal (9)
- VWAP cumulé + déviation
- Z-score du return 60s
- Variations + vélocité Polymarket
- Encoding cyclique de l'heure (sin/cos)
- Ratio de régime de volatilité (60s / 300s)
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd


HORIZON_SEC = 300  # 5 minutes
LABEL_MIN_ENTRY_SEC = 30
LABEL_MAX_DELAY_SEC = 180
_EVENT_CLOSE_RE = re.compile(r"(\d{10})$")


def _infer_poll_sec(df: pd.DataFrame, default: int = 5) -> int:
    """Infer the dominant sampling cadence from timestamps."""
    if "ts" not in df or len(df) < 2:
        return default
    ts = pd.to_numeric(df["ts"], errors="coerce").dropna().sort_values().to_numpy()
    gaps = np.diff(ts) / 1000.0
    gaps = gaps[np.isfinite(gaps) & (gaps > 0)]
    if len(gaps) == 0:
        return default
    usable = gaps[gaps <= 600]
    if len(usable) < max(10, int(len(gaps) * 0.1)):
        usable = gaps
    rounded = np.maximum(1, np.rint(usable)).astype(int)
    values, counts = np.unique(rounded, return_counts=True)
    return int(values[np.argmax(counts)]) if len(values) else default


def _rows_for(seconds: int, poll_sec: int) -> int:
    return max(1, int(round(seconds / max(1, poll_sec))))


def _parse_event_close_ms(value: object) -> float:
    if not isinstance(value, str):
        return np.nan
    match = _EVENT_CLOSE_RE.search(value)
    if not match:
        return np.nan
    return float(int(match.group(1)) * 1000)


def _future_price_from_event_close(df: pd.DataFrame) -> pd.Series:
    """Use BTC price near the actual Polymarket event close for labels."""
    if "poly_market" not in df:
        return pd.Series(np.nan, index=df.index, dtype=float)

    ts = pd.to_numeric(df["ts"], errors="coerce").to_numpy(dtype=float)
    price = pd.to_numeric(df["btc_price"], errors="coerce").to_numpy(dtype=float)
    close_ms = df["poly_market"].map(_parse_event_close_ms).to_numpy(dtype=float)
    future = np.full(len(df), np.nan, dtype=float)

    for i, close_ts in enumerate(close_ms):
        if not np.isfinite(close_ts) or not np.isfinite(ts[i]):
            continue
        seconds_to_close = (close_ts - ts[i]) / 1000.0
        if seconds_to_close < LABEL_MIN_ENTRY_SEC:
            continue
        j = int(np.searchsorted(ts, close_ts, side="left"))
        if j >= len(df):
            continue
        delay_sec = (ts[j] - close_ts) / 1000.0
        if 0 <= delay_sec <= LABEL_MAX_DELAY_SEC and np.isfinite(price[j]):
            future[i] = price[j]

    return pd.Series(future, index=df.index, dtype=float)


def _rolling_std(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).std().fillna(0)


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0).rolling(window, min_periods=1).mean()
    down = (-delta.clip(upper=0)).rolling(window, min_periods=1).mean()
    rs = up / down.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=1).mean()


def build_features(df: pd.DataFrame, poll_sec: int | None = None) -> pd.DataFrame:
    """Construit les features depuis un DataFrame de ticks bruts.

    Le DataFrame doit être trié par ts croissant et contenir :
    ts, btc_price, btc_bid, btc_ask, btc_ob_imb, poly_yes, poly_no
    """
    df = df.sort_values("ts").reset_index(drop=True).copy()
    if poll_sec is None:
        poll_sec = _infer_poll_sec(df)
    df["ts"] = pd.to_numeric(df["ts"])
    df["dt"] = pd.to_datetime(df["ts"], unit="ms")

    # --- Returns multi-horizons ---
    for sec in (30, 60, 180, 300):
        n = _rows_for(sec, poll_sec)
        df[f"ret_{sec}s"] = df["btc_price"].pct_change(n).fillna(0)

    # --- Volatilité ---
    for sec in (60, 300):
        n = _rows_for(sec, poll_sec)
        df[f"vol_{sec}s"] = _rolling_std(df["ret_30s"], n)

    # --- Momentum ---
    df["rsi_14"] = _rsi(df["btc_price"], 14)
    df["mom_1m"] = df["btc_price"] - df["btc_price"].shift(_rows_for(60, poll_sec)).fillna(df["btc_price"])
    df["mom_5m"] = df["btc_price"] - df["btc_price"].shift(_rows_for(300, poll_sec)).fillna(df["btc_price"])

    # --- Orderbook / microstructure ---
    df["ob_imb"] = df["btc_ob_imb"].fillna(0)
    df["ob_imb_avg_30s"] = df["ob_imb"].rolling(_rows_for(30, poll_sec), min_periods=1).mean()
    df["spread"] = (df["btc_ask"] - df["btc_bid"]).fillna(0)
    df["spread_pct"] = df["spread"] / df["btc_price"]

    # --- Signal Polymarket (la foule) ---
    df["poly_yes"] = df["poly_yes"].fillna(0.5)
    df["poly_edge_vs_5050"] = df["poly_yes"] - 0.5

    # --- Mispricing : le marché dit X mais le momentum dit Y ---
    df["poly_vs_momentum"] = df["poly_yes"] - (df["ret_60s"] * 10 + 0.5).clip(0, 1)

    # --- Bandes de Bollinger (window=20) ---
    bb_window = 20
    bb_mean = df["btc_price"].rolling(bb_window, min_periods=1).mean()
    bb_std = df["btc_price"].rolling(bb_window, min_periods=1).std().fillna(0)
    bb_upper = bb_mean + 2 * bb_std
    bb_lower = bb_mean - 2 * bb_std
    # Distance relative au prix (positif = au-dessus de la borne)
    safe_price = df["btc_price"].replace(0, np.nan)
    df["bb_upper_pct"] = ((df["btc_price"] - bb_upper) / safe_price).fillna(0)
    df["bb_lower_pct"] = ((df["btc_price"] - bb_lower) / safe_price).fillna(0)

    # --- MACD (12-26) + signal (9) ---
    ema_12 = _ema(df["btc_price"], 12)
    ema_26 = _ema(df["btc_price"], 26)
    df["macd"] = (ema_12 - ema_26).fillna(0)
    df["macd_signal"] = _ema(df["macd"], 9).fillna(0)

    # --- VWAP cumulé + déviation ---
    # Sans volume explicite, on approxime avec un poids unitaire (TWAP cumulé)
    cum_price = df["btc_price"].expanding(min_periods=1).mean()
    df["vwap_dev_pct"] = ((df["btc_price"] - cum_price) / safe_price).fillna(0)

    # --- Z-score du return 60s sur fenêtre 60s ---
    n60 = _rows_for(60, poll_sec)
    ret60_mean = df["ret_60s"].rolling(n60, min_periods=1).mean()
    ret60_std = df["ret_60s"].rolling(n60, min_periods=1).std().replace(0, np.nan)
    df["ret_zscore_60"] = ((df["ret_60s"] - ret60_mean) / ret60_std).fillna(0)

    # --- Variations + vélocité Polymarket ---
    n30p = _rows_for(30, poll_sec)
    n60p = _rows_for(60, poll_sec)
    df["poly_yes_diff_30s"] = df["poly_yes"].diff(n30p).fillna(0)
    df["poly_yes_diff_60s"] = df["poly_yes"].diff(n60p).fillna(0)
    df["poly_yes_velocity"] = (df["poly_yes_diff_60s"] / 60.0).fillna(0)

    # --- Encoding cyclique de l'heure ---
    hours = df["dt"].dt.hour + df["dt"].dt.minute / 60.0
    df["tod_sin"] = np.sin(2 * np.pi * hours / 24.0)
    df["tod_cos"] = np.cos(2 * np.pi * hours / 24.0)

    # --- Ratio de régime de volatilité ---
    safe_vol_300 = df["vol_300s"].replace(0, np.nan)
    df["volatility_ratio"] = (df["vol_60s"] / safe_vol_300).fillna(1.0)

    # --- Label : BTC a-t-il monte a la cloture du marche 5m ? ---
    event_future = _future_price_from_event_close(df)
    has_event_closes = (
        "poly_market" in df
        and df["poly_market"].map(_parse_event_close_ms).notna().any()
    )
    if has_event_closes:
        df["future_price"] = event_future
    else:
        shift_n = _rows_for(HORIZON_SEC, poll_sec)
        df["future_price"] = df["btc_price"].shift(-shift_n)

    df["label"] = np.where(
        df["future_price"].notna(),
        (df["future_price"] > df["btc_price"]).astype(int),
        np.nan,
    )
    df["future_return"] = (df["future_price"] - df["btc_price"]) / df["btc_price"]

    return df


FEATURE_COLS = [
    "ret_30s", "ret_60s", "ret_180s", "ret_300s",
    "vol_60s", "vol_300s",
    "rsi_14", "mom_1m", "mom_5m",
    "ob_imb", "ob_imb_avg_30s", "spread_pct",
    "poly_yes", "poly_edge_vs_5050", "poly_vs_momentum",
    # --- Nouvelles features (10) ---
    "bb_upper_pct", "bb_lower_pct",
    "macd", "macd_signal",
    "vwap_dev_pct",
    "ret_zscore_60",
    "poly_yes_diff_30s", "poly_yes_diff_60s", "poly_yes_velocity",
    "tod_sin", "tod_cos",
    "volatility_ratio",
]


def get_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Extrait X, y prêts pour training en gardant uniquement lignes avec label."""
    clean = df.dropna(subset=["future_price"]).copy()
    X = clean[FEATURE_COLS].replace([np.inf, -np.inf], 0).fillna(0)
    y = clean["label"].astype(int)
    return X, y
