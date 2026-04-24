"""Fetcher Polymarket — Gamma Events API + CLOB prices.

Les marchés BTC Up/Down 5m sont des events qui se renouvellent toutes les 5 min.
On cherche l'event actif via tag crypto + slug pattern, on suit ses marchés jusqu'à
clôture, puis on passe au suivant.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx

from config import POLYMARKET_CLOB_API, POLYMARKET_GAMMA_API


@dataclass
class MarketSnapshot:
    event_slug: str
    question: str
    yes_price: float
    no_price: float
    yes_token_id: str
    no_token_id: str
    volume_24h: float
    timestamp: int
    end_date: str


class PolymarketClient:
    """Client REST synchrone pour Polymarket (lecture seule, pas de clé requise)."""

    def __init__(self, timeout: float = 10.0) -> None:
        self._client = httpx.Client(timeout=timeout, headers={"User-Agent": "polymarket-bot/0.1"})

    def close(self) -> None:
        self._client.close()

    def get_events_by_series(
        self,
        series_slug: str = "btc-updown-5m",
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Liste les events d'une série (ex: btc-updown-5m, eth-updown-5m)."""
        url = f"{POLYMARKET_GAMMA_API}/events"
        params = {
            "limit": limit,
            "closed": "false",
            "series_slug": series_slug,
            "order": "endDate",
            "ascending": "true",
        }
        r = self._client.get(url, params=params)
        r.raise_for_status()
        return r.json()

    def find_btc_updown_event(self) -> dict[str, Any] | None:
        """Trouve l'event BTC Up/Down 5m actif.

        Les events ont un slug déterministe : `btc-updown-5m-{close_unix_seconds}`
        où close_unix_seconds est divisible par 300 (5 min).
        On essaie les 3 prochaines fenêtres pour tomber sur un event actif.
        """
        import time as _t
        now = int(_t.time())
        # Prochaine fenêtre 5-min + celle d'après
        next_close = ((now // 300) + 1) * 300
        candidates = [next_close, next_close + 300, next_close + 600, next_close - 300]

        for close_ts in candidates:
            slug = f"btc-updown-5m-{close_ts}"
            event = self.get_event(slug)
            if event and not event.get("closed"):
                return event
        return None

    def get_event(self, slug: str) -> dict[str, Any] | None:
        """Récupère un event par slug."""
        url = f"{POLYMARKET_GAMMA_API}/events"
        r = self._client.get(url, params={"slug": slug})
        r.raise_for_status()
        items = r.json()
        return items[0] if items else None

    def get_price(self, token_id: str, side: str = "BUY") -> float:
        """Prix actuel d'un outcome token via CLOB."""
        url = f"{POLYMARKET_CLOB_API}/price"
        params = {"token_id": token_id, "side": side}
        r = self._client.get(url, params=params)
        r.raise_for_status()
        return float(r.json().get("price", 0))

    def get_midpoint(self, token_id: str) -> float:
        """Mid-price CLOB (meilleur prix)."""
        url = f"{POLYMARKET_CLOB_API}/midpoint"
        r = self._client.get(url, params={"token_id": token_id})
        if r.status_code != 200:
            return self.get_price(token_id)
        return float(r.json().get("mid", 0))

    def get_orderbook(self, token_id: str) -> dict[str, Any]:
        url = f"{POLYMARKET_CLOB_API}/book"
        r = self._client.get(url, params={"token_id": token_id})
        r.raise_for_status()
        return r.json()

    def get_prices_history(
        self,
        market_id: str,
        interval: str = "1m",
        fidelity: int = 1,
    ) -> list[dict[str, Any]]:
        url = f"{POLYMARKET_CLOB_API}/prices-history"
        params = {"market": market_id, "interval": interval, "fidelity": fidelity}
        r = self._client.get(url, params=params)
        r.raise_for_status()
        return r.json().get("history", [])

    def _extract_tokens(self, market: dict[str, Any]) -> tuple[str, str] | None:
        """Extrait (yes_token_id, no_token_id) depuis un market."""
        # Gamma API renvoie clobTokenIds sous forme de string JSON parfois
        clob_ids = market.get("clobTokenIds")
        if isinstance(clob_ids, str):
            import json
            try:
                clob_ids = json.loads(clob_ids)
            except Exception:
                return None
        if not clob_ids or len(clob_ids) < 2:
            return None

        outcomes = market.get("outcomes")
        if isinstance(outcomes, str):
            import json
            try:
                outcomes = json.loads(outcomes)
            except Exception:
                outcomes = ["Yes", "No"]

        # Match outcome → token_id
        if outcomes and len(outcomes) >= 2:
            yes_idx = next(
                (i for i, o in enumerate(outcomes) if o.lower() in ("yes", "up")),
                0,
            )
            no_idx = 1 - yes_idx if len(clob_ids) >= 2 else 0
            return clob_ids[yes_idx], clob_ids[no_idx]
        return clob_ids[0], clob_ids[1]

    def snapshot_event(self, event: dict[str, Any]) -> MarketSnapshot | None:
        """Construit un snapshot depuis un event (prend le 1er market)."""
        markets = event.get("markets") or []
        if not markets:
            return None
        market = markets[0]
        tokens = self._extract_tokens(market)
        if not tokens:
            return None
        yes_id, no_id = tokens
        try:
            yes_price = self.get_midpoint(yes_id)
            no_price = self.get_midpoint(no_id)
        except Exception:
            return None

        return MarketSnapshot(
            event_slug=event.get("slug", ""),
            question=event.get("title", "") or market.get("question", ""),
            yes_price=yes_price,
            no_price=no_price,
            yes_token_id=yes_id,
            no_token_id=no_id,
            volume_24h=float(event.get("volume24hr") or 0),
            timestamp=int(time.time() * 1000),
            end_date=event.get("endDate", "") or market.get("endDate", ""),
        )


if __name__ == "__main__":
    client = PolymarketClient()
    try:
        event = client.find_btc_updown_event()
        if not event:
            print("Aucun event BTC trouvé.")
        else:
            print(f"Event: {event.get('title')}")
            print(f"Slug : {event.get('slug')}")
            print(f"Markets: {len(event.get('markets') or [])}")
            snap = client.snapshot_event(event)
            print(f"\nSnapshot: {snap}")
    finally:
        client.close()
