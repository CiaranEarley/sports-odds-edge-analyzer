import unittest

from sports_trading.market_data import (
    MarketProbabilityRow,
    market_rows_to_frame,
    polymarket_events_to_frame,
    sample_market_rows,
)


class MarketDataTests(unittest.TestCase):
    def test_sample_market_rows_have_expected_columns(self):
        frame = sample_market_rows()

        self.assertFalse(frame.empty)
        self.assertIn("Market probability", frame.columns)
        self.assertTrue(frame["Market probability"].between(0, 1).all())

    def test_polymarket_events_to_frame_parses_outcome_price_strings(self):
        events = [
            {
                "title": "Test event",
                "slug": "test-event",
                "volume": "1000",
                "liquidity": "250",
                "markets": [
                    {
                        "question": "Will Team A win?",
                        "outcomes": '["Yes", "No"]',
                        "outcomePrices": '["0.62", "0.38"]',
                    }
                ],
            }
        ]

        frame = polymarket_events_to_frame(events)

        self.assertEqual(len(frame), 2)
        self.assertEqual(frame.loc[0, "Outcome"], "Yes")
        self.assertAlmostEqual(frame.loc[0, "Market probability"], 0.62)
        self.assertEqual(frame.loc[0, "Event URL"], "https://polymarket.com/event/test-event")

    def test_market_rows_to_frame_drops_invalid_probabilities(self):
        frame = market_rows_to_frame(
            [
                MarketProbabilityRow(
                    source="Test",
                    event="Event",
                    market="Market",
                    outcome="Valid",
                    probability=0.5,
                    event_url="",
                ),
                MarketProbabilityRow(
                    source="Test",
                    event="Event",
                    market="Market",
                    outcome="Invalid",
                    probability=1.2,
                    event_url="",
                ),
            ]
        )

        self.assertEqual(len(frame), 1)
        self.assertEqual(frame.loc[0, "Outcome"], "Valid")


if __name__ == "__main__":
    unittest.main()
