import unittest

from sports_trading.models import (
    davidson_three_way_probabilities,
    elo_win_probability,
    football_poisson_markets,
    football_score_probabilities,
    poisson_goal_probability,
)


class ModelTests(unittest.TestCase):
    def test_elo_probability_is_symmetric_at_equal_ratings(self):
        self.assertAlmostEqual(elo_win_probability(rating_a=1500, rating_b=1500), 0.5)

    def test_elo_probability_increases_with_rating_advantage(self):
        self.assertGreater(
            elo_win_probability(rating_a=1600, rating_b=1500),
            elo_win_probability(rating_a=1500, rating_b=1600),
        )

    def test_davidson_probabilities_sum_to_one(self):
        probabilities = davidson_three_way_probabilities(
            home_rating=1500,
            away_rating=1480,
            home_advantage=65,
            draw_strength=0.85,
        )

        self.assertAlmostEqual(sum(probabilities), 1.0)
        self.assertTrue(all(probability > 0 for probability in probabilities))

    def test_poisson_goal_probabilities_sum_close_to_one(self):
        probability_mass = sum(
            poisson_goal_probability(expected_goals=1.8, goals=goals)
            for goals in range(20)
        )

        self.assertAlmostEqual(probability_mass, 1.0, places=6)

    def test_football_poisson_market_probabilities_are_coherent(self):
        markets = football_poisson_markets(
            home_expected_goals=1.6,
            away_expected_goals=1.1,
            max_goals=16,
        )

        self.assertAlmostEqual(markets.home_win + markets.draw + markets.away_win, 1.0)
        self.assertAlmostEqual(markets.over_2_5 + markets.under_2_5, 1.0)
        self.assertAlmostEqual(markets.btts_yes + markets.btts_no, 1.0)
        self.assertGreater(markets.home_win, markets.away_win)
        self.assertGreater(markets.score_coverage, 0.999999)

    def test_football_score_grid_size(self):
        scores = football_score_probabilities(
            home_expected_goals=1.5,
            away_expected_goals=1.2,
            max_goals=5,
        )

        self.assertEqual(len(scores), 36)

    def test_football_poisson_rejects_negative_expected_goals(self):
        with self.assertRaises(ValueError):
            football_poisson_markets(home_expected_goals=-1.0, away_expected_goals=1.2)


if __name__ == "__main__":
    unittest.main()
