import unittest

import pandas as pd

from sports_trading.evaluation import analyze_model_predictions


class EvaluationTests(unittest.TestCase):
    def test_analyzes_prediction_log(self):
        predictions = pd.DataFrame(
            [
                {
                    "model_probability": 0.60,
                    "odds": 2.0,
                    "stake": 100,
                    "result": "win",
                    "closing_odds": 1.85,
                },
                {
                    "model_probability": 0.48,
                    "odds": 2.2,
                    "stake": 100,
                    "result": "loss",
                    "closing_odds": 2.0,
                },
                {
                    "model_probability": 0.52,
                    "odds": 1.95,
                    "stake": 100,
                    "result": "push",
                    "closing_odds": 1.9,
                },
            ]
        )

        summary, analyzed, calibration, edge_buckets = analyze_model_predictions(predictions)

        self.assertEqual(summary.bets, 3)
        self.assertEqual(summary.total_staked, 300)
        self.assertAlmostEqual(summary.profit, 0)
        self.assertGreater(summary.brier_score, 0)
        self.assertGreater(summary.log_loss, 0)
        self.assertIsNotNone(summary.average_closing_line_value)
        self.assertIn("edge", analyzed.columns)
        self.assertFalse(calibration.empty)
        self.assertFalse(edge_buckets.empty)

    def test_requires_core_columns(self):
        with self.assertRaises(ValueError):
            analyze_model_predictions(pd.DataFrame([{"odds": 2.0}]))

    def test_rejects_probability_outside_decimal_range(self):
        predictions = pd.DataFrame(
            [
                {
                    "model_probability": 60,
                    "odds": 2.0,
                    "stake": 100,
                    "result": "win",
                }
            ]
        )

        with self.assertRaises(ValueError):
            analyze_model_predictions(predictions)


if __name__ == "__main__":
    unittest.main()
