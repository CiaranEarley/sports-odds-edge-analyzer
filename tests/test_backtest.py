import unittest

import pandas as pd

from sports_trading.backtest import analyze_bets


class BacktestTests(unittest.TestCase):
    def test_analyzes_bet_log(self):
        bets = pd.DataFrame(
            [
                {"stake": 100, "odds": 2.0, "result": "win", "closing_odds": 1.9},
                {"stake": 100, "odds": 2.0, "result": "loss", "closing_odds": 2.1},
                {"stake": 100, "odds": 1.8, "result": "push", "closing_odds": 1.8},
            ]
        )

        summary, analyzed = analyze_bets(bets)

        self.assertEqual(summary.bets, 3)
        self.assertEqual(summary.total_staked, 300)
        self.assertEqual(summary.profit, 0)
        self.assertEqual(summary.hit_rate, 1 / 3)
        self.assertIn("equity", analyzed.columns)
        self.assertIsNotNone(summary.average_closing_line_value)

    def test_requires_core_columns(self):
        with self.assertRaises(ValueError):
            analyze_bets(pd.DataFrame([{"stake": 100}]))


if __name__ == "__main__":
    unittest.main()
