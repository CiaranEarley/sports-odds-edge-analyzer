import unittest

from sports_trading.odds import (
    OddsFormat,
    decimal_to_american,
    decimal_to_fractional,
    implied_probability,
    overround,
    remove_overround,
    to_decimal_odds,
)


class OddsTests(unittest.TestCase):
    def test_decimal_odds_conversion(self):
        self.assertEqual(to_decimal_odds("2.5", OddsFormat.DECIMAL), 2.5)
        self.assertEqual(to_decimal_odds("+150", OddsFormat.AMERICAN), 2.5)
        self.assertEqual(to_decimal_odds("-200", OddsFormat.AMERICAN), 1.5)
        self.assertEqual(to_decimal_odds("3/2", OddsFormat.FRACTIONAL), 2.5)

    def test_implied_probability_and_overround(self):
        probabilities = [implied_probability(1.91), implied_probability(1.91)]

        self.assertAlmostEqual(overround(probabilities), 0.0471, places=4)
        self.assertEqual(remove_overround(probabilities), [0.5, 0.5])

    def test_decimal_to_american_and_fractional(self):
        self.assertEqual(decimal_to_american(2.5), 150)
        self.assertEqual(decimal_to_american(1.5), -200)
        self.assertEqual(decimal_to_fractional(2.5), "3/2")

    def test_rejects_invalid_odds(self):
        with self.assertRaises(ValueError):
            to_decimal_odds("1.0", OddsFormat.DECIMAL)


if __name__ == "__main__":
    unittest.main()
