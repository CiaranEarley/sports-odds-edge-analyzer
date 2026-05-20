"""Odds conversion and market margin helpers."""

from __future__ import annotations

from enum import Enum
from fractions import Fraction


class OddsFormat(str, Enum):
    """Supported input odds formats."""

    DECIMAL = "decimal"
    AMERICAN = "american"
    FRACTIONAL = "fractional"


def to_decimal_odds(value: str | float | int, odds_format: str | OddsFormat) -> float:
    """Convert decimal, American, or fractional odds to decimal odds."""

    selected_format = OddsFormat(odds_format)
    if selected_format == OddsFormat.DECIMAL:
        decimal_odds = float(value)
    elif selected_format == OddsFormat.AMERICAN:
        american_odds = float(value)
        if american_odds == 0:
            raise ValueError("American odds cannot be zero.")
        if american_odds > 0:
            decimal_odds = 1.0 + american_odds / 100.0
        else:
            decimal_odds = 1.0 + 100.0 / abs(american_odds)
    else:
        fraction = Fraction(str(value))
        decimal_odds = 1.0 + fraction.numerator / fraction.denominator

    if decimal_odds <= 1.0:
        raise ValueError("Decimal odds must be greater than 1.0.")
    return decimal_odds


def implied_probability(decimal_odds: float) -> float:
    """Convert decimal odds to implied probability."""

    if decimal_odds <= 1.0:
        raise ValueError("Decimal odds must be greater than 1.0.")
    return 1.0 / decimal_odds


def remove_overround(implied_probabilities: list[float]) -> list[float]:
    """Remove bookmaker overround with proportional normalization."""

    if not implied_probabilities:
        raise ValueError("At least one implied probability is required.")
    if any(probability <= 0 for probability in implied_probabilities):
        raise ValueError("All implied probabilities must be positive.")

    total_probability = sum(implied_probabilities)
    if total_probability <= 0:
        raise ValueError("Total implied probability must be positive.")
    return [probability / total_probability for probability in implied_probabilities]


def overround(implied_probabilities: list[float]) -> float:
    """Return market overround as total implied probability minus 100%."""

    if not implied_probabilities:
        raise ValueError("At least one implied probability is required.")
    return sum(implied_probabilities) - 1.0


def decimal_to_american(decimal_odds: float) -> int:
    """Convert decimal odds to American odds."""

    if decimal_odds <= 1.0:
        raise ValueError("Decimal odds must be greater than 1.0.")
    if decimal_odds >= 2.0:
        return round((decimal_odds - 1.0) * 100.0)
    return round(-100.0 / (decimal_odds - 1.0))


def decimal_to_fractional(decimal_odds: float, *, max_denominator: int = 100) -> str:
    """Convert decimal odds to a compact fractional-odds string."""

    if decimal_odds <= 1.0:
        raise ValueError("Decimal odds must be greater than 1.0.")
    fraction = Fraction(decimal_odds - 1.0).limit_denominator(max_denominator)
    return f"{fraction.numerator}/{fraction.denominator}"
