"""Fetcher Binance — prix BTC/USDT spot + orderbook.

API gratuite, pas de clé requise pour la lecture.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import httpx

from config import BINANCE_REST_API


@dataclass
class PriceSnapshot:
    symbol: str
    price: float
    bid: float
    ask: float
    bid_qty: float
    ask_qty: float
    timestamp: int


class BinanceClient:
    """Client REST synchrone Binance (lecture seule)."""

    def __init__(self, timeout: float = 10.0) -> None:
        self._client = httpx.Client(timeout=timeout, base_url=BINANCE_REST_API)

    def close(self) -> None:
        self._client.close()

    def get_price(self, symbol: str = "BTCUSDT") -> float:
        r = self._client.get("/api/v3/ticker/price", params={"symbol": symbol})
        r.raise_for_status()
        return float(r.json()["price"])

    def get_book_ticker(self, symbol: str = "BTCUSDT") -> PriceSnapshot:
        """Meilleur bid/ask — utile pour prix instantané + spread."""
        r = self._client.get("/api/v3/ticker/bookTicker", params={"symbol": symbol})
        r.raise_for_status()
        d = r.json()
        return PriceSnapshot(
            symbol=d["symbol"],
            price=(float(d["bidPrice"]) + float(d["askPrice"])) / 2,
            bid=float(d["bidPrice"]),
            ask=float(d["askPrice"]),
            bid_qty=float(d["bidQty"]),
            ask_qty=float(d["askQty"]),
            timestamp=int(time.time() * 1000),
        )

    def get_klines(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "1m",
        limit: int = 500,
    ) -> list[list]:
        """OHLCV candles. interval: 1s, 1m, 5m, 15m, 1h, 4h, 1d..."""
        r = self._client.get(
            "/api/v3/klines",
            params={"symbol": symbol, "interval": interval, "limit": limit},
        )
        r.raise_for_status()
        return r.json()

    def get_depth(self, symbol: str = "BTCUSDT", limit: int = 20) -> dict:
        """Orderbook — calcul imbalance feature."""
        r = self._client.get(
            "/api/v3/depth",
            params={"symbol": symbol, "limit": limit},
        )
        r.raise_for_status()
        return r.json()

    def orderbook_imbalance(self, symbol: str = "BTCUSDT", levels: int = 10) -> float:
        """(bids - asks) / (bids + asks) sur N niveaux. Feature importante."""
        book = self.get_depth(symbol, limit=levels)
        bid_vol = sum(float(b[1]) for b in book["bids"][:levels])
        ask_vol = sum(float(a[1]) for a in book["asks"][:levels])
        total = bid_vol + ask_vol
        if total == 0:
            return 0.0
        return (bid_vol - ask_vol) / total


if __name__ == "__main__":
    client = BinanceClient()
    try:
        snap = client.get_book_ticker()
        print(f"BTC price: ${snap.price:,.2f} | spread: {snap.ask - snap.bid:.2f}")
        imb = client.orderbook_imbalance()
        print(f"Orderbook imbalance: {imb:+.3f}")
        klines = client.get_klines(interval="1m", limit=5)
        print(f"Dernières 5 bougies 1m: {len(klines)} reçues")
    finally:
        client.close()
