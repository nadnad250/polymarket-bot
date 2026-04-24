"""Modèle de frais Polymarket — réaliste pour backtest/live.

Polymarket (Polygon) :
- Pas de frais directs de trading
- Gas fee Polygon : ~$0.01-0.10 par tx (matic)
- Spread bid/ask : 1-3% typique sur les marchés 5m (illiquides)
- Slippage si ordre > top-of-book (rare pour $20-$50)
- USDC→USDC donc pas de FX
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FeeModel:
    gas_fee_usd: float = 0.05       # coût gas Polygon par tx (approx)
    spread_pct: float = 0.02        # spread moyen YES/NO (2%)
    slippage_pct: float = 0.005     # slippage supplémentaire sur ordre
    maker_rebate: float = 0.0       # Polymarket pas de rebate actuellement

    def apply_entry(self, price: float, size_usd: float) -> tuple[float, float]:
        """
        Prix effectif d'entrée + frais totaux pour un achat.
        Retourne (effective_price, total_fees_usd).
        """
        effective = price * (1 + self.spread_pct / 2 + self.slippage_pct)
        effective = min(effective, 0.99)
        fees = self.gas_fee_usd
        return effective, fees

    def apply_exit(self, won: bool, size_usd: float, entry_price: float) -> tuple[float, float]:
        """
        Payout + frais à la sortie (résolution).
        Retourne (payout_usd, exit_fees_usd).
        """
        if won:
            # Payout nominal 1.0 par share. shares = size / entry_price
            shares = size_usd / entry_price
            payout = shares * 1.0
        else:
            payout = 0.0
        fees = self.gas_fee_usd  # claim gas
        return payout, fees

    def net_pnl(self, won: bool, size_usd: float, entry_price: float) -> float:
        """PnL net après TOUS les frais (gas entrée + gas sortie + spread déjà dans prix)."""
        _, entry_fees = self.apply_entry(entry_price, size_usd)
        payout, exit_fees = self.apply_exit(won, size_usd, entry_price)
        return payout - size_usd - entry_fees - exit_fees


DEFAULT_FEES = FeeModel()
