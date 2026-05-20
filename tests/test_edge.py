import unittest

from sports_trading.edge import calculate_edge


class EdgeTests(unittest.TestCase):
    def test_positive_edge_and_kelly(self):
        result = calculate_edge(
            decimal_odds=2.1,
            model_probability=0.52,
            stake=100,
            bankroll=1000,
            kelly_multiplier=0.25,
        )

        self.assertAlmostEqual(result.market_probability, 1 / 2.1)
        self.assertGreater(result.edge_probability, 0)
        self.assertGreater(result.expected_value_for_stake, 0)
        self.assertGreater(result.recommended_stake, 0)

    def test_negative_edge_has_zero_kelly(self):
        result = calculate_edge(
            decimal_odds=1.8,
            model_probability=0.50,
            bankroll=1000,
        )

        self.assertLess(result.expected_value_per_unit, 0)
        self.assertEqual(result.recommended_stake, 0)

    def test_rejects_invalid_probability(self):
        with self.assertRaises(ValueError):
            calculate_edge(decimal_odds=2.0, model_probability=1.0)


if __name__ == "__main__":
    unittest.main()
