import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.features import build_features, get_xy


def _row(ts_ms: int, price: float, close_sec: int) -> dict:
    return {
        "ts": ts_ms,
        "btc_price": price,
        "btc_bid": price - 0.5,
        "btc_ask": price + 0.5,
        "btc_ob_imb": 0.0,
        "poly_market": f"btc-updown-5m-{close_sec}",
        "poly_yes": 0.5,
        "poly_no": 0.5,
    }


class FeatureLabelTests(unittest.TestCase):
    def test_event_close_label_uses_tick_near_close(self):
        close_sec = 1_800_000_000
        close_ms = close_sec * 1000
        df = pd.DataFrame([
            _row(close_ms - 60_000, 100.0, close_sec),
            _row(close_ms - 30_000, 101.0, close_sec),
            _row(close_ms + 5_000, 103.0, close_sec),
        ])

        features = build_features(df)
        X, y = get_xy(features)

        self.assertEqual(len(X), 2)
        self.assertEqual(y.tolist(), [1, 1])
        self.assertEqual(features.loc[0, "future_price"], 103.0)

    def test_sparse_tick_after_close_is_not_labeled(self):
        close_sec = 1_800_000_000
        close_ms = close_sec * 1000
        df = pd.DataFrame([
            _row(close_ms - 120_000, 100.0, close_sec),
            _row(close_ms + 20 * 60_000, 110.0, close_sec),
        ])

        features = build_features(df)
        X, _ = get_xy(features)

        self.assertEqual(len(X), 0)
        self.assertTrue(features["future_price"].isna().all())


if __name__ == "__main__":
    unittest.main()
