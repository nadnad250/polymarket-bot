"""Fetcher BTC multi-source avec fallback automatique.

Binance bloque les IPs US (HTTP 451), donc en CI (GitHub Actions US) on utilise
Coinbase ou Kraken. En local (non-US) Binance a la meilleure liquidité.

Ordre de fallback : Binance → Coinbase → Kraken.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import httpx


@dataclass
class PriceSnapshot:
    symbol: str
    price: float
    bid: float
    ask: float
    bid_qty: float
    ask_qty: float
    timestamp: int


class BTCFetcher:
    """Fetcher BTC résistant aux blocages géographiques."""

    def __init__(self, timeout: float = 10.0) -> None:
        self._client = httpx.Client(
            timeout=timeout,
            headers={"User-Agent": "polymarket-bot/0.1"},
        )
        self._source: str | None = None  # cache source qui marche

    def close(self) -> None:
        self._client.close()

    @property
    def source(self) -> str | None:
        return self._source

    # ---------- BINANCE ----------
    def _binance_ticker(self) -> PriceSnapshot:
        r = self._client.get(
            "https://api.binance.com/api/v3/ticker/bookTicker",
            params={"symbol": "BTCUSDT"},
        )
        r.raise_for_status()
        d = r.json()
        return PriceSnapshot(
            symbol="BTCUSDT",
            price=(float(d["bidPrice"]) + float(d["askPrice"])) / 2,
            bid=float(d["bidPrice"]), ask=float(d["askPrice"]),
            bid_qty=float(d["bidQty"]), ask_qty=float(d["askQty"]),
            timestamp=int(time.time() * 1000),
        )

    def _binance_imbalance(self, levels: int = 10) -> float:
        r = self._client.get(
            "https://api.binance.com/api/v3/depth",
            params={"symbol": "BTCUSDT", "limit": levels},
        )
        r.raise_for_status()
        book = r.json()
        bid = sum(float(b[1]) for b in book["bids"][:levels])
        ask = sum(float(a[1]) for a in book["asks"][:levels])
        return (bid - ask) / (bid + ask) if (bid + ask) > 0 else 0.0

    # ---------- COINBASE ----------
    def _coinbase_ticker(self) -> PriceSnapshot:
        r = self._client.get(
            "https://api.exchange.coinbase.com/products/BTC-USD/ticker"
        )
        r.raise_for_status()
        d = r.json()
        bid = float(d["bid"])
        ask = float(d["ask"])
        return PriceSnapshot(
            symbol="BTC-USD",
            price=(bid + ask) / 2,
            bid=bid, ask=ask,
            bid_qty=float(d.get("size", 0)),
            ask_qty=float(d.get("size", 0)),
            timestamp=int(time.time() * 1000),
        )

    def _coinbase_imbalance(self, levels: int = 10) -> float:
        r = self._client.get(
            "https://api.exchange.coinbase.com/products/BTC-USD/book",
            params={"level": 2},
        )
        r.raise_for_status()
        book = r.json()
        bid = sum(float(b[1]) for b in book["bids"][:levels])
        ask = sum(float(a[1]) for a in book["asks"][:levels])
        return (bid - ask) / (bid + ask) if (bid + ask) > 0 else 0.0

    # ---------- KRAKEN ----------
    def _kraken_ticker(self) -> PriceSnapshot:
        r = self._client.get(
            "https://api.kraken.com/0/public/Ticker",
            params={"pair": "XBTUSD"},
        )
        r.raise_for_status()
        result = r.json().get("result", {})
        key = next(iter(result)) if result else None
        if not key:
            raise RuntimeError("Kraken ticker vide")
        d = result[key]
        bid = float(d["b"][0])
        ask = float(d["a"][0])
        return PriceSnapshot(
            symbol="XBTUSD",
            price=(bid + ask) / 2,
            bid=bid, ask=ask,
            bid_qty=float(d["b"][2]),
            ask_qty=float(d["a"][2]),
            timestamp=int(time.time() * 1000),
        )

    def _kraken_imbalance(self, levels: int = 10) -> float:
        r = self._client.get(
            "https://api.kraken.com/0/public/Depth",
            params={"pair": "XBTUSD", "count": levels},
        )
        r.raise_for_status()
        result = r.json().get("result", {})
        key = next(iter(result)) if result else None
        if not key:
            return 0.0
        book = result[key]
        bid = sum(float(b[1]) for b in book["bids"][:levels])
        ask = sum(float(a[1]) for a in book["asks"][:levels])
        return (bid - ask) / (bid + ask) if (bid + ask) > 0 else 0.0

    # ---------- FAÇADE AVEC FALLBACK ----------
    def get_book_ticker(self) -> PriceSnapshot:
        sources = [("binance", self._binance_ticker),
                   ("coinbase", self._coinbase_ticker),
                   ("kraken", self._kraken_ticker)]
        # Si on a déjà une source qui marche, on la retente en premier
        if self._source:
            sources.sort(key=lambda s: 0 if s[0] == self._source else 1)

        last_err: Exception | None = None
        for name, fn in sources:
            try:
                snap = fn()
                self._source = name
                return snap
            except Exception as e:
                last_err = e
                continue
        raise RuntimeError(f"Toutes les sources BTC ont échoué: {last_err}")

    def orderbook_imbalance(self, levels: int = 10) -> float:
        src = self._source or "binance"
        fn = {
            "binance": self._binance_imbalance,
            "coinbase": self._coinbase_imbalance,
            "kraken": self._kraken_imbalance,
        }.get(src, self._binance_imbalance)
        try:
            return fn(levels)
        except Exception:
            return 0.0

    def get_price(self) -> float:
        return self.get_book_ticker().price

    def get_klines(self, limit: int = 60) -> list[list]:
        """1min klines pour calcul momentum."""
        if self._source == "coinbase":
            r = self._client.get(
                "https://api.exchange.coinbase.com/products/BTC-USD/candles",
                params={"granularity": 60},
            )
            r.raise_for_status()
            return r.json()[:limit]  # [time, low, high, open, close, volume]
        try:
            r = self._client.get(
                "https://api.binance.com/api/v3/klines",
                params={"symbol": "BTCUSDT", "interval": "1m", "limit": limit},
            )
            r.raise_for_status()
            return r.json()
        except Exception:
            return []


if __name__ == "__main__":
    f = BTCFetcher()
    try:
        snap = f.get_book_ticker()
        print(f"Source: {f.source}")
        print(f"BTC ${snap.price:,.2f} | bid={snap.bid:,.2f} ask={snap.ask:,.2f}")
        print(f"Imbalance: {f.orderbook_imbalance():+.3f}")
    finally:
        f.close()
