"""Bet log analysis helpers."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from sports_trading.odds import implied_probability


@dataclass(frozen=True)
class BacktestSummary:
    """Summary metrics for a historical bet log."""

    bets: int
    total_staked: float
    profit: float
    roi: float
    hit_rate: float
    average_odds: float
    max_drawdown: float
    average_closing_line_value: float | None


def analyze_bets(bets: pd.DataFrame) -> tuple[BacktestSummary, pd.DataFrame]:
    """Analyze a bet log.

    Required columns: stake, odds, result.
    result accepts win/loss/push or 1/0/0.5.
    Optional column: closing_odds.
    """

    missing = {"stake", "odds", "result"} - set(bets.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    analyzed = bets.copy()
    analyzed["stake"] = pd.to_numeric(analyzed["stake"], errors="coerce")
    analyzed["odds"] = pd.to_numeric(analyzed["odds"], errors="coerce")
    analyzed["result_value"] = analyzed["result"].map(_result_to_value)
    analyzed = analyzed.dropna(subset=["stake", "odds", "result_value"]).reset_index(drop=True)

    if analyzed.empty:
        raise ValueError("No valid bets found.")
    if (analyzed["stake"] < 0).any():
        raise ValueError("Stake values cannot be negative.")
    if (analyzed["odds"] <= 1).any():
        raise ValueError("Odds must be decimal odds greater than 1.0.")

    analyzed["profit"] = analyzed.apply(_profit_for_row, axis=1)
    analyzed["equity"] = analyzed["profit"].cumsum()
    analyzed["running_peak"] = analyzed["equity"].cummax().clip(lower=0)
    analyzed["drawdown"] = analyzed["running_peak"] - analyzed["equity"]

    average_clv = None
    if "closing_odds" in analyzed:
        analyzed["closing_odds"] = pd.to_numeric(analyzed["closing_odds"], errors="coerce")
        valid_closing = analyzed.dropna(subset=["closing_odds"])
        valid_closing = valid_closing[valid_closing["closing_odds"] > 1.0].copy()
        if not valid_closing.empty:
            valid_closing["closing_line_value"] = (
                valid_closing["closing_odds"].map(implied_probability)
                - valid_closing["odds"].map(implied_probability)
            )
            analyzed.loc[valid_closing.index, "closing_line_value"] = valid_closing[
                "closing_line_value"
            ]
            average_clv = float(valid_closing["closing_line_value"].mean())

    total_staked = float(analyzed["stake"].sum())
    profit = float(analyzed["profit"].sum())
    summary = BacktestSummary(
        bets=len(analyzed),
        total_staked=total_staked,
        profit=profit,
        roi=profit / total_staked if total_staked else 0.0,
        hit_rate=float((analyzed["result_value"] == 1.0).mean()),
        average_odds=float(analyzed["odds"].mean()),
        max_drawdown=float(analyzed["drawdown"].max()),
        average_closing_line_value=average_clv,
    )
    return summary, analyzed


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
