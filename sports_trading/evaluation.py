"""Model evaluation helpers for sports prediction logs."""

from __future__ import annotations

from dataclasses import dataclass
from math import log

import pandas as pd

from sports_trading.odds import implied_probability


@dataclass(frozen=True)
class ModelEvaluationSummary:
    """Headline model quality and betting performance metrics."""

    bets: int
    total_staked: float
    profit: float
    roi: float
    hit_rate: float
    average_model_probability: float
    average_edge: float
    brier_score: float
    log_loss: float
    max_drawdown: float
    average_closing_line_value: float | None


def analyze_model_predictions(
    predictions: pd.DataFrame,
    *,
    calibration_bins: int = 5,
) -> tuple[ModelEvaluationSummary, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate model probabilities against outcomes and market prices.

    Required columns: model_probability, odds, stake, result.
    Optional columns: event, market, selection, closing_odds.
    """

    missing = {"model_probability", "odds", "stake", "result"} - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    if calibration_bins < 2:
        raise ValueError("calibration_bins must be at least 2.")

    analyzed = predictions.copy()
    analyzed["model_probability"] = pd.to_numeric(
        analyzed["model_probability"],
        errors="coerce",
    )
    analyzed["odds"] = pd.to_numeric(analyzed["odds"], errors="coerce")
    analyzed["stake"] = pd.to_numeric(analyzed["stake"], errors="coerce")
    analyzed["result_value"] = analyzed["result"].map(_result_to_value)
    analyzed = analyzed.dropna(
        subset=["model_probability", "odds", "stake", "result_value"]
    ).reset_index(drop=True)

    if analyzed.empty:
        raise ValueError("No valid prediction rows found.")
    if not analyzed["model_probability"].between(0, 1, inclusive="neither").all():
        raise ValueError("model_probability values must be decimals between 0 and 1.")
    if (analyzed["odds"] <= 1).any():
        raise ValueError("Odds must be decimal odds greater than 1.0.")
    if (analyzed["stake"] < 0).any():
        raise ValueError("Stake values cannot be negative.")

    analyzed["market_probability"] = analyzed["odds"].map(implied_probability)
    analyzed["edge"] = analyzed["model_probability"] - analyzed["market_probability"]
    analyzed["profit"] = analyzed.apply(_profit_for_row, axis=1)
    analyzed["equity"] = analyzed["profit"].cumsum()
    analyzed["running_peak"] = analyzed["equity"].cummax().clip(lower=0)
    analyzed["drawdown"] = analyzed["running_peak"] - analyzed["equity"]
    analyzed["brier_component"] = (
        analyzed["model_probability"] - analyzed["result_value"]
    ) ** 2
    analyzed["log_loss_component"] = analyzed.apply(_log_loss_for_row, axis=1)

    average_clv = None
    if "closing_odds" in analyzed:
        analyzed["closing_odds"] = pd.to_numeric(analyzed["closing_odds"], errors="coerce")
        valid_closing = analyzed.dropna(subset=["closing_odds"])
        valid_closing = valid_closing[valid_closing["closing_odds"] > 1.0].copy()
        if not valid_closing.empty:
            valid_closing["closing_line_value"] = (
                valid_closing["closing_odds"].map(implied_probability)
                - valid_closing["market_probability"]
            )
            analyzed.loc[valid_closing.index, "closing_line_value"] = valid_closing[
                "closing_line_value"
            ]
            average_clv = float(valid_closing["closing_line_value"].mean())

    calibration = _calibration_table(analyzed, bins=calibration_bins)
    edge_buckets = _edge_bucket_table(analyzed)
    total_staked = float(analyzed["stake"].sum())
    profit = float(analyzed["profit"].sum())

    summary = ModelEvaluationSummary(
        bets=len(analyzed),
        total_staked=total_staked,
        profit=profit,
        roi=profit / total_staked if total_staked else 0.0,
        hit_rate=float((analyzed["result_value"] == 1.0).mean()),
        average_model_probability=float(analyzed["model_probability"].mean()),
        average_edge=float(analyzed["edge"].mean()),
        brier_score=float(analyzed["brier_component"].mean()),
        log_loss=float(analyzed["log_loss_component"].mean()),
        max_drawdown=float(analyzed["drawdown"].max()),
        average_closing_line_value=average_clv,
    )
    return summary, analyzed, calibration, edge_buckets


def _calibration_table(analyzed: pd.DataFrame, *, bins: int) -> pd.DataFrame:
    labels = [f"{int(left)}-{int(right)}%" for left, right in _bucket_edges(bins)]
    analyzed = analyzed.copy()
    analyzed["probability_bucket"] = pd.cut(
        analyzed["model_probability"],
        bins=bins,
        labels=labels,
        include_lowest=True,
        duplicates="drop",
    )
    return (
        analyzed.groupby("probability_bucket", observed=True)
        .agg(
            bets=("result_value", "size"),
            average_model_probability=("model_probability", "mean"),
            observed_rate=("result_value", "mean"),
            brier_score=("brier_component", "mean"),
        )
        .reset_index()
    )


def _edge_bucket_table(analyzed: pd.DataFrame) -> pd.DataFrame:
    labels = ["<=0%", "0-2.5%", "2.5-5%", "5-10%", "10%+"]
    analyzed = analyzed.copy()
    analyzed["edge_bucket"] = pd.cut(
        analyzed["edge"],
        bins=[float("-inf"), 0.0, 0.025, 0.05, 0.10, float("inf")],
        labels=labels,
        include_lowest=True,
    )
    grouped = (
        analyzed.groupby("edge_bucket", observed=True)
        .agg(
            bets=("result_value", "size"),
            average_edge=("edge", "mean"),
            total_staked=("stake", "sum"),
            profit=("profit", "sum"),
        )
        .reset_index()
    )
    grouped["roi"] = grouped["profit"] / grouped["total_staked"]
    return grouped


def _bucket_edges(bins: int) -> list[tuple[float, float]]:
    step = 100 / bins
    return [(index * step, (index + 1) * step) for index in range(bins)]


def _result_to_value(value) -> float | None:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"win", "won", "w", "1"}:
            return 1.0
        if normalized in {"loss", "lost", "lose", "l", "0"}:
            return 0.0
        if normalized in {"push", "void", "refund", "0.5"}:
            return 0.5
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return None
    if numeric_value in {0.0, 0.5, 1.0}:
        return numeric_value
    return None


def _profit_for_row(row) -> float:
    if row["result_value"] == 1.0:
        return row["stake"] * (row["odds"] - 1.0)
    if row["result_value"] == 0.5:
        return 0.0
    return -row["stake"]


def _log_loss_for_row(row) -> float:
    probability = min(max(row["model_probability"], 1e-12), 1.0 - 1e-12)
    outcome = row["result_value"]
    return -(outcome * log(probability) + (1.0 - outcome) * log(1.0 - probability))
