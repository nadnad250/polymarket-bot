"""Decision engine for protected paper trades and exploratory shadow trades."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from src.simulator.fees import DEFAULT_FEES, FeeModel


@dataclass
class DecisionPlan:
    action: str = "skip"
    source: str = "none"
    side: str | None = None
    price: float | None = None
    edge: float | None = None
    p_up: float | None = None
    confidence: float | None = None
    required_edge: float | None = None
    quality_score: float = 0.0
    size_pct: float = 0.0
    reasons: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def should_open(self) -> bool:
        return self.action == "open" and self.side is not None and self.price is not None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["should_open"] = self.should_open
        return payload


def choose_side(p_up: float, yes_price: float, no_price: float) -> tuple[str, float, float, dict[str, float]]:
    """Choose the Polymarket side with the strongest model-vs-price edge."""
    edge_yes = p_up - yes_price
    edge_no = (1.0 - p_up) - no_price
    if edge_yes >= edge_no:
        return "YES", yes_price, edge_yes, {"YES": edge_yes, "NO": edge_no}
    return "NO", no_price, edge_no, {"YES": edge_yes, "NO": edge_no}


def required_edge(
    base_edge: float,
    price: float,
    fees: FeeModel = DEFAULT_FEES,
    seconds_to_close: float | None = None,
) -> float:
    """Dynamic edge hurdle: fees plus small penalties for fragile market states."""
    hurdle = base_edge + fees.spread_pct / 2 + fees.slippage_pct
    if seconds_to_close is not None:
        if seconds_to_close < 90:
            hurdle += 0.015
        elif seconds_to_close < 150:
            hurdle += 0.005
    if price < 0.15 or price > 0.85:
        hurdle += 0.010
    return float(hurdle)


def quality_score(
    edge: float,
    hurdle: float,
    confidence: float,
    price: float,
    seconds_to_close: float | None = None,
    model_auc: float | None = None,
) -> float:
    """Convert decision inputs into a 0-100 quality score for audit/UI."""
    edge_component = _clip01(edge / max(hurdle * 1.5, 0.01))
    confidence_component = _clip01(confidence / 0.25)
    price_component = _clip01(1.0 - abs(price - 0.50) / 0.45)
    time_component = 0.75
    if seconds_to_close is not None:
        time_component = _clip01((seconds_to_close - 30.0) / 210.0)
    auc_component = 0.50
    if model_auc is not None:
        auc_component = _clip01((model_auc - 0.50) / 0.20)

    score = (
        0.38 * edge_component
        + 0.24 * confidence_component
        + 0.16 * price_component
        + 0.12 * time_component
        + 0.10 * auc_component
    )
    return round(float(score * 100.0), 2)


def evaluate_ml_decision(
    *,
    p_up: float | None,
    yes_price: float,
    no_price: float,
    min_confidence: float,
    min_edge: float,
    min_quality: float,
    price_min: float,
    price_max: float,
    seconds_to_close: float | None,
    model_metrics: dict[str, Any] | None = None,
    fees: FeeModel = DEFAULT_FEES,
) -> DecisionPlan:
    """Strict ML decision used by the protected main paper portfolio."""
    plan = DecisionPlan(source="ml", p_up=p_up, required_edge=None)
    if p_up is None:
        plan.reasons.append("no_model_prediction")
        return plan

    confidence = abs(float(p_up) - 0.5)
    side, price, edge, edges = choose_side(float(p_up), float(yes_price), float(no_price))
    hurdle = required_edge(min_edge, price, fees=fees, seconds_to_close=seconds_to_close)
    auc = _safe_float((model_metrics or {}).get("auc"))
    score = quality_score(edge, hurdle, confidence, price, seconds_to_close, auc)

    plan.side = side
    plan.price = price
    plan.edge = edge
    plan.confidence = confidence
    plan.required_edge = hurdle
    plan.quality_score = score
    plan.details.update({"edges": edges, "model_auc": auc})

    if confidence < min_confidence:
        plan.reasons.append(f"confidence {confidence:.3f} < {min_confidence:.3f}")
    if edge < hurdle:
        plan.reasons.append(f"edge {edge:.3f} < required {hurdle:.3f}")
    if not (price_min < price < price_max):
        plan.reasons.append(f"price {price:.3f} outside {price_min:.2f}-{price_max:.2f}")
    if score < min_quality:
        plan.reasons.append(f"quality {score:.1f} < {min_quality:.1f}")

    if not plan.reasons:
        plan.action = "open"
    return plan


def evaluate_shadow_decision(
    *,
    p_up: float | None,
    yes_price: float,
    no_price: float,
    momentum: float,
    imbalance: float,
    min_confidence: float,
    min_edge: float,
    price_min: float,
    price_max: float,
    seconds_to_close: float | None,
) -> DecisionPlan:
    """Exploratory paper decision for the Shadow Lab.

    This records small notional demo trades to gather statistics. It is
    intentionally looser than the protected portfolio and is never a real order.
    """
    if p_up is not None:
        confidence = abs(float(p_up) - 0.5)
        side, price, edge, edges = choose_side(float(p_up), float(yes_price), float(no_price))
        score = quality_score(
            edge=edge,
            hurdle=max(min_edge, 0.005),
            confidence=confidence,
            price=price,
            seconds_to_close=seconds_to_close,
        )
        plan = DecisionPlan(
            source="shadow_ml",
            side=side,
            price=price,
            edge=edge,
            p_up=float(p_up),
            confidence=confidence,
            required_edge=min_edge,
            quality_score=score,
            details={"edges": edges},
        )
        if confidence >= min_confidence and edge >= min_edge and price_min < price < price_max:
            plan.action = "open"
            return plan
        plan.reasons.append(
            f"shadow_ml weak conf={confidence:.3f} edge={edge:.3f} price={price:.3f}"
        )
    else:
        plan = DecisionPlan(source="shadow_baseline")

    baseline = _baseline_shadow(yes_price, no_price, momentum, imbalance, price_min, price_max)
    if baseline.should_open:
        baseline.required_edge = min_edge
        baseline.quality_score = max(
            baseline.quality_score,
            quality_score(
                edge=baseline.edge or 0.0,
                hurdle=max(min_edge, 0.005),
                confidence=min(abs(momentum) * 250.0 + abs(imbalance) * 0.25, 0.25),
                price=baseline.price or 0.5,
                seconds_to_close=seconds_to_close,
            ),
        )
        return baseline

    if p_up is not None:
        plan.reasons.extend(baseline.reasons)
        return plan
    return baseline


def _baseline_shadow(
    yes_price: float,
    no_price: float,
    momentum: float,
    imbalance: float,
    price_min: float,
    price_max: float,
) -> DecisionPlan:
    aligned_up = momentum > 0.00015 and imbalance > 0.15
    aligned_down = momentum < -0.00015 and imbalance < -0.15
    pseudo_edge = min(0.08, abs(momentum) * 45.0 + abs(imbalance) * 0.025)

    if aligned_up:
        side, price = "YES", float(yes_price)
    elif aligned_down:
        side, price = "NO", float(no_price)
    else:
        return DecisionPlan(
            source="shadow_baseline",
            edge=pseudo_edge,
            reasons=[f"baseline not aligned mom={momentum:+.5f} imb={imbalance:+.3f}"],
            details={"momentum": momentum, "imbalance": imbalance},
        )

    plan = DecisionPlan(
        action="open",
        source="shadow_baseline",
        side=side,
        price=price,
        edge=pseudo_edge,
        confidence=min(abs(momentum) * 250.0 + abs(imbalance) * 0.25, 0.25),
        details={"momentum": momentum, "imbalance": imbalance},
    )
    if not (price_min < price < price_max):
        plan.action = "skip"
        plan.reasons.append(f"baseline price {price:.3f} outside {price_min:.2f}-{price_max:.2f}")
    return plan


def _clip01(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(min(1.0, max(0.0, value)))


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        out = float(value)
        return out if np.isfinite(out) else None
    except Exception:
        return None
