import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.simulator.decision import evaluate_ml_decision, evaluate_shadow_decision


class DecisionEngineTests(unittest.TestCase):
    def test_ml_decision_opens_only_when_edge_confidence_and_quality_pass(self):
        plan = evaluate_ml_decision(
            p_up=0.72,
            yes_price=0.50,
            no_price=0.51,
            min_confidence=0.10,
            min_edge=0.08,
            min_quality=55,
            price_min=0.08,
            price_max=0.92,
            seconds_to_close=180,
            model_metrics={"auc": 0.62},
        )

        self.assertTrue(plan.should_open)
        self.assertEqual(plan.side, "YES")
        self.assertGreater(plan.quality_score, 55)

    def test_ml_decision_skips_low_confidence_even_with_small_positive_edge(self):
        plan = evaluate_ml_decision(
            p_up=0.508,
            yes_price=0.50,
            no_price=0.50,
            min_confidence=0.10,
            min_edge=0.08,
            min_quality=55,
            price_min=0.08,
            price_max=0.92,
            seconds_to_close=180,
            model_metrics={"auc": 0.61},
        )

        self.assertFalse(plan.should_open)
        self.assertTrue(any("confidence" in r for r in plan.reasons))

    def test_shadow_can_record_looser_demo_signal(self):
        plan = evaluate_shadow_decision(
            p_up=0.508,
            yes_price=0.50,
            no_price=0.50,
            momentum=0.0,
            imbalance=0.0,
            min_confidence=0.005,
            min_edge=0.005,
            price_min=0.08,
            price_max=0.92,
            seconds_to_close=180,
        )

        self.assertTrue(plan.should_open)
        self.assertEqual(plan.source, "shadow_ml")

    def test_shadow_baseline_requires_aligned_momentum_and_imbalance(self):
        weak = evaluate_shadow_decision(
            p_up=None,
            yes_price=0.50,
            no_price=0.50,
            momentum=0.0003,
            imbalance=-0.25,
            min_confidence=0.005,
            min_edge=0.005,
            price_min=0.08,
            price_max=0.92,
            seconds_to_close=180,
        )
        aligned = evaluate_shadow_decision(
            p_up=None,
            yes_price=0.50,
            no_price=0.50,
            momentum=0.0003,
            imbalance=0.25,
            min_confidence=0.005,
            min_edge=0.005,
            price_min=0.08,
            price_max=0.92,
            seconds_to_close=180,
        )

        self.assertFalse(weak.should_open)
        self.assertTrue(aligned.should_open)
        self.assertEqual(aligned.side, "YES")


if __name__ == "__main__":
    unittest.main()
