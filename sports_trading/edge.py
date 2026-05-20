"""Expected value and Kelly staking helpers."""

from __future__ import annotations

from dataclasses import dataclass

from sports_trading.odds import implied_probability


@dataclass(frozen=True)
class EdgeResult:
    """Pricing edge and stake sizing for one selection."""

    decimal_odds: float
    model_probability: float
    market_probability: float
    edge_probability: float
    expected_value_per_unit: float
    expected_value_for_stake: float
    break_even_probability: float
    fair_decimal_odds: float
    full_kelly_fraction: float
    fractional_kelly_fraction: float
    recommended_stake: float


def calculate_edge(
    *,
    decimal_odds: float,
    model_probability: float,
    stake: float = 100.0,
    bankroll: float = 1000.0,
    kelly_multiplier: float = 0.25,
) -> EdgeResult:
    """Calculate edge, EV, and fractional Kelly stake for one bet."""

    _validate_inputs(
        decimal_odds=decimal_odds,
        model_probability=model_probability,
        stake=stake,
        bankroll=bankroll,
        kelly_multiplier=kelly_multiplier,
    )
    market_probability = implied_probability(decimal_odds)
    profit_if_win = decimal_odds - 1.0
    expected_value_per_unit = model_probability * profit_if_win - (1.0 - model_probability)
    full_kelly_fraction = max(
        (decimal_odds * model_probability - 1.0) / profit_if_win,
        0.0,
    )
    fractional_kelly_fraction = full_kelly_fraction * kelly_multiplier

    return EdgeResult(
        decimal_odds=decimal_odds,
        model_probability=model_probability,
        market_probability=market_probability,
        edge_probability=model_probability - market_probability,
        expected_value_per_unit=expected_value_per_unit,
        expected_value_for_stake=expected_value_per_unit * stake,
        break_even_probability=market_probability,
        fair_decimal_odds=1.0 / model_probability,
        full_kelly_fraction=full_kelly_fraction,
        fractional_kelly_fraction=fractional_kelly_fraction,
        recommended_stake=bankroll * fractional_kelly_fraction,
    )


def _validate_inputs(
    *,
    decimal_odds: float,
    model_probability: float,
    stake: float,
    bankroll: float,
    kelly_multiplier: float,
) -> None:
    if decimal_odds <= 1.0:
        raise ValueError("decimal_odds must be greater than 1.0.")
    if not 0 < model_probability < 1:
        raise ValueError("model_probability must be between 0 and 1.")
    if stake < 0:
        raise ValueError("stake cannot be negative.")
    if bankroll < 0:
        raise ValueError("bankroll cannot be negative.")
    if not 0 <= kelly_multiplier <= 1:
        raise ValueError("kelly_multiplier must be between 0 and 1.")
