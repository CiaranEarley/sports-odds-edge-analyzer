"""Simple sports probability models used by the dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, factorial


@dataclass(frozen=True)
class ScoreProbability:
    """Probability for one exact football scoreline."""

    home_goals: int
    away_goals: int
    probability: float


@dataclass(frozen=True)
class FootballPoissonMarkets:
    """Derived football markets from independent home/away goal processes."""

    home_win: float
    draw: float
    away_win: float
    over_2_5: float
    under_2_5: float
    btts_yes: float
    btts_no: float
    score_coverage: float
    scores: list[ScoreProbability]


def elo_win_probability(
    *,
    rating_a: float,
    rating_b: float,
    rating_adjustment_a: float = 0.0,
    scale: float = 400.0,
) -> float:
    """Return two-way win probability from an Elo-style rating difference."""

    if scale <= 0:
        raise ValueError("scale must be positive.")

    rating_difference = rating_a + rating_adjustment_a - rating_b
    return 1.0 / (1.0 + 10.0 ** (-rating_difference / scale))


def davidson_three_way_probabilities(
    *,
    home_rating: float,
    away_rating: float,
    home_advantage: float = 65.0,
    draw_strength: float = 0.85,
    scale: float = 400.0,
) -> tuple[float, float, float]:
    """Estimate home/draw/away probabilities with a Davidson tie model.

    The ratings are converted to positive team strengths. A separate draw
    strength controls the mass assigned to the draw outcome.
    """

    if scale <= 0:
        raise ValueError("scale must be positive.")
    if draw_strength < 0:
        raise ValueError("draw_strength cannot be negative.")

    home_strength = 10.0 ** ((home_rating + home_advantage) / scale)
    away_strength = 10.0 ** (away_rating / scale)
    draw_component = draw_strength * (home_strength * away_strength) ** 0.5
    denominator = home_strength + away_strength + draw_component

    return (
        home_strength / denominator,
        draw_component / denominator,
        away_strength / denominator,
    )


def poisson_goal_probability(*, expected_goals: float, goals: int) -> float:
    """Return the Poisson probability of a team scoring exactly `goals`."""

    if expected_goals < 0:
        raise ValueError("expected_goals cannot be negative.")
    if goals < 0:
        raise ValueError("goals cannot be negative.")

    return exp(-expected_goals) * expected_goals**goals / factorial(goals)


def football_score_probabilities(
    *,
    home_expected_goals: float,
    away_expected_goals: float,
    max_goals: int = 10,
) -> list[ScoreProbability]:
    """Return a finite grid of exact-score probabilities."""

    _validate_poisson_inputs(
        home_expected_goals=home_expected_goals,
        away_expected_goals=away_expected_goals,
        max_goals=max_goals,
    )

    home_probabilities = [
        poisson_goal_probability(expected_goals=home_expected_goals, goals=goals)
        for goals in range(max_goals + 1)
    ]
    away_probabilities = [
        poisson_goal_probability(expected_goals=away_expected_goals, goals=goals)
        for goals in range(max_goals + 1)
    ]

    return [
        ScoreProbability(
            home_goals=home_goals,
            away_goals=away_goals,
            probability=home_probability * away_probability,
        )
        for home_goals, home_probability in enumerate(home_probabilities)
        for away_goals, away_probability in enumerate(away_probabilities)
    ]


def football_poisson_markets(
    *,
    home_expected_goals: float,
    away_expected_goals: float,
    max_goals: int = 16,
) -> FootballPoissonMarkets:
    """Price common football markets from independent Poisson xG assumptions."""

    scores = football_score_probabilities(
        home_expected_goals=home_expected_goals,
        away_expected_goals=away_expected_goals,
        max_goals=max_goals,
    )
    coverage = sum(score.probability for score in scores)
    if coverage <= 0:
        raise ValueError("Score probability coverage must be positive.")

    home_win = sum(score.probability for score in scores if score.home_goals > score.away_goals)
    draw = sum(score.probability for score in scores if score.home_goals == score.away_goals)
    away_win = sum(score.probability for score in scores if score.home_goals < score.away_goals)

    total_expected_goals = home_expected_goals + away_expected_goals
    under_2_5 = sum(
        poisson_goal_probability(expected_goals=total_expected_goals, goals=goals)
        for goals in range(3)
    )
    over_2_5 = 1.0 - under_2_5
    btts_no = (
        exp(-home_expected_goals)
        + exp(-away_expected_goals)
        - exp(-(home_expected_goals + away_expected_goals))
    )
    btts_yes = 1.0 - btts_no

    return FootballPoissonMarkets(
        home_win=home_win / coverage,
        draw=draw / coverage,
        away_win=away_win / coverage,
        over_2_5=over_2_5,
        under_2_5=under_2_5,
        btts_yes=btts_yes,
        btts_no=btts_no,
        score_coverage=coverage,
        scores=scores,
    )


def _validate_poisson_inputs(
    *,
    home_expected_goals: float,
    away_expected_goals: float,
    max_goals: int,
) -> None:
    if home_expected_goals < 0:
        raise ValueError("home_expected_goals cannot be negative.")
    if away_expected_goals < 0:
        raise ValueError("away_expected_goals cannot be negative.")
    if max_goals < 1:
        raise ValueError("max_goals must be at least 1.")
