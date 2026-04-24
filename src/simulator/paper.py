"""Simulateur paper trading $1000.

Gère :
- Bankroll initiale (INITIAL_CAPITAL)
- Positions YES/NO avec sizing Kelly fractionnaire
- Frais + spread simulés
- Historique complet des trades + P&L
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from config import (
    ASSUMED_SPREAD,
    INITIAL_CAPITAL,
    KELLY_FRACTION,
    MAX_POSITION_PCT,
    POLYMARKET_FEE,
)

Side = Literal["YES", "NO"]


@dataclass
class Trade:
    ts: int
    market: str
    side: Side
    price: float
    size_usd: float
    model_prob: float
    outcome: int | None = None  # 1 si gagné, 0 si perdu, None en cours
    pnl: float = 0.0


@dataclass
class Portfolio:
    cash: float = INITIAL_CAPITAL
    trades: list[Trade] = field(default_factory=list)
    open_positions: list[Trade] = field(default_factory=list)

    @property
    def total_trades(self) -> int:
        return len([t for t in self.trades if t.outcome is not None])

    @property
    def win_rate(self) -> float:
        closed = [t for t in self.trades if t.outcome is not None]
        if not closed:
            return 0.0
        return sum(t.outcome for t in closed) / len(closed)

    @property
    def total_pnl(self) -> float:
        return sum(t.pnl for t in self.trades)

    @property
    def roi(self) -> float:
        return (self.cash - INITIAL_CAPITAL) / INITIAL_CAPITAL


def kelly_size(edge: float, price: float, fraction: float = KELLY_FRACTION) -> float:
    """
    Kelly fractionnaire pour un pari binaire.
    edge = model_prob - market_price (probabilité additionnelle)
    Retourne la fraction du bankroll à miser (0-1).
    """
    if edge <= 0 or price <= 0 or price >= 1:
        return 0.0
    b = (1 - price) / price  # odds net
    p = price + edge
    q = 1 - p
    kelly = (b * p - q) / b
    return max(0.0, min(kelly * fraction, MAX_POSITION_PCT))


class PaperSimulator:
    def __init__(self, initial_capital: float = INITIAL_CAPITAL) -> None:
        self.portfolio = Portfolio(cash=initial_capital)

    def place_trade(
        self,
        ts: int,
        market: str,
        side: Side,
        price: float,
        model_prob: float,
    ) -> Trade | None:
        """Place un trade si l'edge le justifie. None si pas d'edge."""
        edge = (model_prob if side == "YES" else 1 - model_prob) - price
        size_pct = kelly_size(edge, price)
        if size_pct == 0:
            return None

        size_usd = self.portfolio.cash * size_pct
        effective_price = price + ASSUMED_SPREAD / 2  # slippage côté achat
        shares = size_usd / effective_price

        trade = Trade(
            ts=ts,
            market=market,
            side=side,
            price=effective_price,
            size_usd=size_usd,
            model_prob=model_prob,
        )
        trade._shares = shares  # type: ignore[attr-defined]
        self.portfolio.cash -= size_usd
        self.portfolio.open_positions.append(trade)
        self.portfolio.trades.append(trade)
        return trade

    def resolve(self, trade: Trade, btc_went_up: bool) -> float:
        """Résolution du marché : payout 1.0 si correct, 0.0 sinon."""
        won = (trade.side == "YES" and btc_went_up) or (trade.side == "NO" and not btc_went_up)
        trade.outcome = 1 if won else 0
        shares = getattr(trade, "_shares", trade.size_usd / trade.price)
        payout = shares * 1.0 if won else 0.0
        fee = payout * POLYMARKET_FEE
        payout -= fee
        trade.pnl = payout - trade.size_usd
        self.portfolio.cash += payout
        if trade in self.portfolio.open_positions:
            self.portfolio.open_positions.remove(trade)
        return trade.pnl

    def summary(self) -> dict:
        p = self.portfolio
        return {
            "cash": round(p.cash, 2),
            "total_trades": p.total_trades,
            "win_rate": round(p.win_rate, 3),
            "total_pnl": round(p.total_pnl, 2),
            "roi_pct": round(p.roi * 100, 2),
            "open_positions": len(p.open_positions),
        }


if __name__ == "__main__":
    sim = PaperSimulator()
    t = sim.place_trade(ts=0, market="BTC-5m", side="YES", price=0.48, model_prob=0.58)
    print(f"Trade placé: {t}")
    sim.resolve(t, btc_went_up=True)
    print(f"Résumé: {sim.summary()}")
