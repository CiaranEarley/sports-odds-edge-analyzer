"""Streamlit dashboard for sports odds edge analysis."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sports_trading.backtest import analyze_bets
from sports_trading.edge import calculate_edge
from sports_trading.evaluation import analyze_model_predictions
from sports_trading.market_data import (
    MarketDataError,
    fetch_polymarket_events,
    polymarket_events_to_frame,
    sample_market_rows,
)
from sports_trading.models import (
    davidson_three_way_probabilities,
    elo_win_probability,
    football_poisson_markets,
)
from sports_trading.odds import (
    OddsFormat,
    decimal_to_american,
    decimal_to_fractional,
    implied_probability,
    overround,
    remove_overround,
    to_decimal_odds,
)


DEFAULT_MARKETS = {
    "Football 2-way": ["Team A", "Team B"],
    "Football 3-way": ["Home", "Draw", "Away"],
    "Basketball spread": ["Home - spread", "Away + spread"],
    "Tennis match winner": ["Player A", "Player B"],
}


def main() -> None:
    st.set_page_config(
        page_title="Sports Odds Edge Analyzer",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    _apply_styles()
    st.title("Sports Odds Edge Analyzer")

    config = _sidebar_inputs()
    market_tab, data_tab, football_tab, evaluation_tab, methodology_tab = st.tabs(
        [
            "Market Edge",
            "Market Data",
            "Football Poisson Model",
            "Model Evaluation",
            "Methodology",
        ]
    )

    with market_tab:
        _render_market_edge_tab(config)

    with data_tab:
        _render_market_data_tab(config)

    with football_tab:
        _render_football_poisson_tab(
            stake=config["stake"],
            bankroll=config["bankroll"],
            kelly_multiplier=config["kelly_multiplier"],
        )

    with evaluation_tab:
        _render_model_evaluation_tab()

    with methodology_tab:
        _render_methodology_tab()


def _render_market_edge_tab(config: dict) -> None:
    _render_probability_model_helper(config["market_template"])
    selections = _market_inputs(config["market_template"], config["odds_format"])
    try:
        market = _build_market(selections)
    except ValueError as error:
        st.error(f"Check the market inputs: {error}")
        return
    bankroll = config["bankroll"]
    stake = config["stake"]
    kelly_multiplier = config["kelly_multiplier"]

    _render_market_summary(market)
    _render_edge_table(
        market=market,
        bankroll=bankroll,
        stake=stake,
        kelly_multiplier=kelly_multiplier,
    )
    _render_line_shopping(config["odds_format"])
    _render_pnl_chart(market, stake=stake)
    _render_backtest()
    _render_model_notes()


def _render_market_data_tab(config: dict) -> None:
    st.subheader("Market Data")
    st.markdown(
        """
        Pull market-implied probabilities into the same edge framework. The
        Polymarket mode is read-only and uses public market data endpoints.
        """
    )

    source = st.radio(
        "Source",
        options=["Sample data", "Polymarket public markets"],
        horizontal=True,
        key="market_data_source",
    )
    if source == "Polymarket public markets":
        market_frame = _load_polymarket_market_data()
    else:
        market_frame = sample_market_rows()

    if market_frame.empty:
        st.warning("No market probabilities found. Try a broader search query.")
        return

    _render_market_data_summary(market_frame)
    _render_market_data_table(market_frame)
    _render_market_data_comparison(market_frame, config)
    _render_market_data_chart(market_frame)


def _load_polymarket_market_data() -> pd.DataFrame:
    query = st.text_input(
        "Polymarket search query",
        value="football",
        key="polymarket_search_query",
    )
    limit = st.slider(
        "Event limit",
        min_value=1,
        max_value=20,
        value=8,
        step=1,
        key="polymarket_event_limit",
    )

    if not st.button("Fetch Polymarket markets", key="fetch_polymarket_markets"):
        st.info("Click fetch to query live public Polymarket market data.")
        return sample_market_rows()

    try:
        events = _cached_polymarket_events(query=query, limit=limit)
        market_frame = polymarket_events_to_frame(events)
    except MarketDataError as error:
        st.warning(
            "Polymarket could not be reached from this environment. "
            f"Showing sample data instead. Details: {error}"
        )
        return sample_market_rows()

    if market_frame.empty:
        st.warning("Polymarket returned events, but no outcome prices were available.")
        return sample_market_rows()
    return market_frame


@st.cache_data(ttl=60)
def _cached_polymarket_events(*, query: str, limit: int) -> list[dict]:
    return fetch_polymarket_events(query=query, limit=limit)


def _render_market_data_summary(market_frame: pd.DataFrame) -> None:
    cols = st.columns(4)
    cols[0].metric("Outcomes", f"{len(market_frame)}")
    cols[1].metric("Events", f"{market_frame['Event'].nunique()}")
    cols[2].metric("Markets", f"{market_frame['Market'].nunique()}")
    cols[3].metric(
        "Average probability",
        f"{market_frame['Market probability'].mean():.2%}",
    )


def _render_market_data_table(market_frame: pd.DataFrame) -> None:
    st.subheader("Market-Implied Probabilities")
    display = market_frame.copy()
    display["Market probability"] = display["Market probability"].map(lambda value: f"{value:.2%}")
    display["Market fair odds"] = display["Market fair odds"].map(lambda value: f"{value:.3f}")
    for column in ["Volume", "Liquidity"]:
        if column in display:
            display[column] = display[column].map(
                lambda value: "" if pd.isna(value) else f"{value:,.0f}"
            )
    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        column_config={"Event URL": st.column_config.LinkColumn("Event URL")},
    )


def _render_market_data_comparison(market_frame: pd.DataFrame, config: dict) -> None:
    st.subheader("Compare Against Your Model")
    labels = [
        (
            f"{row['Event']} | {row['Market']} | "
            f"{row['Outcome']} ({row['Market probability']:.1%})"
        )
        for _, row in market_frame.iterrows()
    ]
    selected_index = st.selectbox(
        "Outcome",
        options=list(range(len(market_frame))),
        format_func=lambda index: labels[index],
        key="market_data_selected_outcome",
    )
    selected = market_frame.iloc[selected_index]
    default_model_probability = min(
        max(float(selected["Market probability"]) + 0.03, 0.01),
        0.99,
    )
    model_probability = st.number_input(
        "Your model probability (%)",
        min_value=0.01,
        max_value=99.99,
        value=round(default_model_probability * 100, 2),
        step=0.25,
        format="%.2f",
        key=f"market_data_model_probability_{selected_index}",
    ) / 100.0

    edge = calculate_edge(
        decimal_odds=float(selected["Market fair odds"]),
        model_probability=model_probability,
        stake=config["stake"],
        bankroll=config["bankroll"],
        kelly_multiplier=config["kelly_multiplier"],
    )

    cols = st.columns(5)
    cols[0].metric("Market probability", f"{selected['Market probability']:.2%}")
    cols[1].metric("Model probability", f"{model_probability:.2%}")
    cols[2].metric("Edge", f"{edge.edge_probability:.2%}")
    cols[3].metric(f"EV / {config['stake']:,.0f}", f"{edge.expected_value_for_stake:,.2f}")
    cols[4].metric("Suggested stake", f"{edge.recommended_stake:,.2f}")


def _render_market_data_chart(market_frame: pd.DataFrame) -> None:
    st.subheader("Probability Snapshot")
    chart_frame = market_frame.head(12).copy()
    chart_frame["Label"] = chart_frame["Outcome"] + " | " + chart_frame["Event"]
    figure = go.Figure(
        data=go.Bar(
            x=chart_frame["Label"],
            y=chart_frame["Market probability"],
            marker_color="#38bdf8",
            hovertemplate=(
                "%{x}<br>"
                "Market probability: %{y:.2%}<br>"
                "<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        height=420,
        xaxis_title="Outcome",
        yaxis_title="Market probability",
        yaxis_tickformat=".0%",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(figure, use_container_width=True)


def _render_football_poisson_tab(
    *,
    stake: float,
    bankroll: float,
    kelly_multiplier: float,
) -> None:
    st.subheader("Football Poisson Model")
    inputs = _football_poisson_inputs()
    markets = football_poisson_markets(
        home_expected_goals=inputs["home_expected_goals"],
        away_expected_goals=inputs["away_expected_goals"],
        max_goals=16,
    )

    metric_cols = st.columns(4)
    metric_cols[0].metric(
        "Total expected goals",
        f"{inputs['home_expected_goals'] + inputs['away_expected_goals']:.2f}",
    )
    metric_cols[1].metric(f"{inputs['home_team']} win", f"{markets.home_win:.2%}")
    metric_cols[2].metric("Draw", f"{markets.draw:.2%}")
    metric_cols[3].metric(f"{inputs['away_team']} win", f"{markets.away_win:.2%}")

    book_odds = _football_book_odds_inputs(inputs["home_team"], inputs["away_team"])
    _render_poisson_market_prices(
        markets=markets,
        book_odds=book_odds,
        home_team=inputs["home_team"],
        away_team=inputs["away_team"],
        stake=stake,
        bankroll=bankroll,
        kelly_multiplier=kelly_multiplier,
    )
    _render_score_probability_heatmap(
        markets=markets,
        home_team=inputs["home_team"],
        away_team=inputs["away_team"],
        max_goals=inputs["display_max_goals"],
    )
    _render_top_correct_scores(
        markets=markets,
        home_team=inputs["home_team"],
        away_team=inputs["away_team"],
    )


def _render_model_evaluation_tab() -> None:
    st.subheader("Model Evaluation")
    uploaded_file = st.file_uploader(
        "Prediction log CSV",
        type=["csv"],
        key="prediction_log_upload",
    )
    predictions = (
        pd.read_csv(uploaded_file)
        if uploaded_file is not None
        else _sample_prediction_log()
    )

    try:
        summary, analyzed, calibration, edge_buckets = analyze_model_predictions(
            predictions,
            calibration_bins=5,
        )
    except ValueError as error:
        st.error(f"Could not evaluate prediction log: {error}")
        return

    _render_evaluation_summary(summary)
    _render_calibration_chart(calibration)
    _render_edge_bucket_chart(edge_buckets)
    _render_evaluation_curves(analyzed)
    _render_prediction_log(analyzed)


def _render_methodology_tab() -> None:
    st.subheader("Methodology")
    st.markdown(
        """
        This app follows the same basic loop a sports trader uses: estimate the
        true probability, compare it with the market price, size the position,
        and then evaluate whether the edge was real over a sample of bets.
        """
    )

    _render_methodology_workflow()
    _render_methodology_market_data()
    _render_methodology_pricing()
    _render_methodology_poisson()
    _render_methodology_evaluation()


def _render_methodology_workflow() -> None:
    st.subheader("Trading Workflow")
    workflow = pd.DataFrame(
        [
            {
                "Step": "1. Price the event",
                "Question": "What probability does my model assign?",
                "Output": "Model probability and fair odds",
            },
            {
                "Step": "2. Read the market",
                "Question": "What probability is implied by bookmaker odds?",
                "Output": "Market probability and overround",
            },
            {
                "Step": "3. Compare",
                "Question": "Is my probability higher than the break-even probability?",
                "Output": "Edge and expected value",
            },
            {
                "Step": "4. Size",
                "Question": "How much should I risk if the edge is positive?",
                "Output": "Fractional Kelly stake",
            },
            {
                "Step": "5. Evaluate",
                "Question": "Did the model stay accurate and profitable over time?",
                "Output": "Calibration, scoring rules, ROI, CLV",
            },
        ]
    )
    st.dataframe(workflow, use_container_width=True, hide_index=True)


def _render_methodology_market_data() -> None:
    st.subheader("Market Data")
    st.markdown(
        """
        Market prices can be treated as probabilities. On Polymarket, a binary
        contract trading near `0.62` means the market is roughly pricing that
        outcome at 62%. In sportsbook decimal odds, the equivalent break-even
        probability is `1 / odds`. Once everything is converted into
        probabilities, we can compare different markets and data sources on the
        same scale.
        """
    )
    rows = pd.DataFrame(
        [
            {
                "Source": "Prediction market",
                "Raw price": "Contract price, e.g. 0.62",
                "Probability": "0.62 or 62%",
            },
            {
                "Source": "Sportsbook",
                "Raw price": "Decimal odds, e.g. 2.10",
                "Probability": "1 / 2.10 = 47.62%",
            },
            {
                "Source": "Model",
                "Raw price": "Estimated probability, e.g. 0.55",
                "Probability": "0.55 or 55%",
            },
        ]
    )
    st.dataframe(rows, use_container_width=True, hide_index=True)


def _render_methodology_pricing() -> None:
    st.subheader("Odds, Edge, and Staking")
    formulas = pd.DataFrame(
        [
            {
                "Concept": "Implied probability",
                "Formula": "1 / decimal odds",
                "Meaning": "The break-even probability before adjusting for bookmaker margin.",
            },
            {
                "Concept": "Overround",
                "Formula": "sum(implied probabilities) - 1",
                "Meaning": "The bookmaker margin embedded in all prices for a market.",
            },
            {
                "Concept": "No-vig probability",
                "Formula": "implied probability / sum(implied probabilities)",
                "Meaning": "A simple normalized estimate of the market's fair probability.",
            },
            {
                "Concept": "Edge",
                "Formula": "model probability - implied probability",
                "Meaning": "Positive when your model is more confident than the market price.",
            },
            {
                "Concept": "Expected value",
                "Formula": "p * (odds - 1) - (1 - p)",
                "Meaning": "Expected profit per unit staked using your model probability.",
            },
            {
                "Concept": "Kelly fraction",
                "Formula": "(odds * p - 1) / (odds - 1)",
                "Meaning": "The growth-optimal bankroll fraction before applying a safety haircut.",
            },
        ]
    )
    st.dataframe(formulas, use_container_width=True, hide_index=True)


def _render_methodology_poisson() -> None:
    st.subheader("Football Poisson Model")
    st.markdown(
        """
        The Poisson tab assumes each team's goals are generated by an independent
        Poisson process. The key inputs are expected goals for the home and away
        teams. From those two numbers, the app builds a grid of exact-score
        probabilities and then sums that grid into markets such as home win,
        draw, away win, over/under 2.5, and both teams to score.
        """
    )
    poisson_rows = pd.DataFrame(
        [
            {
                "Output": "Correct score",
                "Calculation": "P(home goals = h) * P(away goals = a)",
            },
            {
                "Output": "Home win",
                "Calculation": "Sum all scorelines where home goals > away goals",
            },
            {
                "Output": "Draw",
                "Calculation": "Sum all scorelines where home goals = away goals",
            },
            {
                "Output": "Over 2.5",
                "Calculation": "Sum all scorelines where total goals >= 3",
            },
            {
                "Output": "BTTS yes",
                "Calculation": "Sum all scorelines where both teams score at least once",
            },
        ]
    )
    st.dataframe(poisson_rows, use_container_width=True, hide_index=True)


def _render_methodology_evaluation() -> None:
    st.subheader("Model Evaluation")
    metrics = pd.DataFrame(
        [
            {
                "Metric": "Brier score",
                "What it checks": "Probability accuracy",
                "Good sign": "Lower is better; zero is perfect.",
            },
            {
                "Metric": "Log loss",
                "What it checks": "Whether the model is confidently wrong",
                "Good sign": "Lower is better; big misses are punished heavily.",
            },
            {
                "Metric": "Calibration",
                "What it checks": "Whether 60% predictions win about 60% of the time",
                "Good sign": "The model line sits close to the perfect calibration line.",
            },
            {
                "Metric": "ROI by edge bucket",
                "What it checks": "Whether larger quoted edges lead to better returns",
                "Good sign": "Higher edge buckets perform better over a large sample.",
            },
            {
                "Metric": "Equity curve",
                "What it checks": "Profit and drawdown path through time",
                "Good sign": "Upward trend without unacceptable drawdowns.",
            },
            {
                "Metric": "Closing-line value",
                "What it checks": "Whether your entry price beat the closing market",
                "Good sign": "Positive average CLV over many bets.",
            },
        ]
    )
    st.dataframe(metrics, use_container_width=True, hide_index=True)

    st.markdown(
        """
        A profitable short sample is not enough by itself. For a sports trading
        model, the stronger evidence is a combination of positive expected value,
        sensible calibration, disciplined staking, and repeated closing-line
        value across many independent bets.
        """
    )


def _render_evaluation_summary(summary) -> None:
    first_row = st.columns(4)
    first_row[0].metric("Brier score", f"{summary.brier_score:.3f}")
    first_row[1].metric("Log loss", f"{summary.log_loss:.3f}")
    first_row[2].metric("ROI", f"{summary.roi:.2%}")
    first_row[3].metric("Profit", f"{summary.profit:,.2f}")

    second_row = st.columns(4)
    second_row[0].metric("Bets", f"{summary.bets}")
    second_row[1].metric("Hit rate", f"{summary.hit_rate:.2%}")
    second_row[2].metric("Average edge", f"{summary.average_edge:.2%}")
    clv = (
        f"{summary.average_closing_line_value:.2%}"
        if summary.average_closing_line_value is not None
        else "N/A"
    )
    second_row[3].metric("Average CLV", clv)


def _render_calibration_chart(calibration: pd.DataFrame) -> None:
    st.subheader("Calibration")
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Perfect calibration",
            line={"dash": "dash", "color": "#94a3b8"},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=calibration["average_model_probability"],
            y=calibration["observed_rate"],
            mode="markers+lines",
            name="Model",
            marker={
                "size": (calibration["bets"] * 5 + 12).tolist(),
                "color": "#22c55e",
                "line": {"color": "#f8fafc", "width": 1},
            },
            text=[
                f"{bucket}: {bets} bets"
                for bucket, bets in zip(
                    calibration["probability_bucket"],
                    calibration["bets"],
                    strict=True,
                )
            ],
            hovertemplate=(
                "%{text}<br>"
                "Average model probability: %{x:.2%}<br>"
                "Observed hit rate: %{y:.2%}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        height=420,
        xaxis_title="Average model probability",
        yaxis_title="Observed hit rate",
        xaxis={"tickformat": ".0%", "range": [0, 1]},
        yaxis={"tickformat": ".0%", "range": [0, 1]},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(figure, use_container_width=True)

    display = calibration.copy()
    display["average_model_probability"] = display["average_model_probability"].map(
        lambda value: f"{value:.2%}"
    )
    display["observed_rate"] = display["observed_rate"].map(lambda value: f"{value:.2%}")
    display["brier_score"] = display["brier_score"].map(lambda value: f"{value:.3f}")
    st.dataframe(display, use_container_width=True, hide_index=True)


def _render_edge_bucket_chart(edge_buckets: pd.DataFrame) -> None:
    st.subheader("ROI by Edge Bucket")
    figure = go.Figure(
        data=go.Bar(
            x=edge_buckets["edge_bucket"].astype(str),
            y=edge_buckets["roi"],
            text=[f"{value:.1%}" for value in edge_buckets["roi"]],
            textposition="outside",
            marker_color=[
                "#22c55e" if value >= 0 else "#ef4444" for value in edge_buckets["roi"]
            ],
            customdata=edge_buckets["bets"],
            hovertemplate=(
                "Edge bucket: %{x}<br>"
                "ROI: %{y:.2%}<br>"
                "Bets: %{customdata}<extra></extra>"
            ),
        )
    )
    figure.add_hline(y=0, line_dash="dot", line_color="#94a3b8")
    figure.update_layout(
        height=380,
        xaxis_title="Model edge",
        yaxis_title="ROI",
        yaxis_tickformat=".0%",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(figure, use_container_width=True)


def _render_evaluation_curves(analyzed: pd.DataFrame) -> None:
    st.subheader("Performance Curves")
    equity_col, clv_col = st.columns(2)
    with equity_col:
        figure = go.Figure(
            data=go.Scatter(
                x=list(range(1, len(analyzed) + 1)),
                y=analyzed["equity"],
                mode="lines+markers",
                name="Equity",
                line={"color": "#22c55e"},
            )
        )
        figure.add_hline(y=0, line_dash="dot", line_color="#94a3b8")
        figure.update_layout(
            height=360,
            xaxis_title="Bet number",
            yaxis_title="Cumulative profit",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(figure, use_container_width=True)

    with clv_col:
        if "closing_line_value" not in analyzed:
            st.metric("Average CLV", "N/A")
            return
        clv_values = analyzed["closing_line_value"].dropna()
        figure = go.Figure(
            data=go.Histogram(
                x=clv_values,
                nbinsx=12,
                marker_color="#38bdf8",
                hovertemplate="CLV: %{x:.2%}<br>Bets: %{y}<extra></extra>",
            )
        )
        figure.add_vline(x=0, line_dash="dot", line_color="#94a3b8")
        figure.update_layout(
            height=360,
            xaxis_title="Closing-line value",
            yaxis_title="Bets",
            xaxis_tickformat=".1%",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(figure, use_container_width=True)


def _render_prediction_log(analyzed: pd.DataFrame) -> None:
    with st.expander("Prediction Log"):
        display = analyzed.copy()
        for column in [
            "model_probability",
            "market_probability",
            "edge",
            "closing_line_value",
        ]:
            if column in display:
                display[column] = display[column].map(
                    lambda value: "" if pd.isna(value) else f"{value:.2%}"
                )
        for column in ["odds", "closing_odds", "profit", "equity", "drawdown"]:
            if column in display:
                display[column] = display[column].map(
                    lambda value: "" if pd.isna(value) else f"{value:,.2f}"
                )
        columns = [
            column
            for column in [
                "date",
                "event",
                "market",
                "selection",
                "model_probability",
                "odds",
                "market_probability",
                "edge",
                "stake",
                "result",
                "profit",
                "equity",
                "closing_odds",
                "closing_line_value",
            ]
            if column in display
        ]
        st.dataframe(display[columns], use_container_width=True, hide_index=True)


def _sample_prediction_log() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["2026-01-06", "Arsenal vs Brighton", "1X2", "Arsenal", 0.58, 1.95, 1.82, 100, "win"],
            ["2026-01-07", "Napoli vs Lazio", "Over/Under", "Over 2.5", 0.54, 2.02, 1.91, 100, "loss"],
            ["2026-01-08", "Valencia vs Sevilla", "BTTS", "Yes", 0.57, 1.92, 1.84, 100, "win"],
            ["2026-01-09", "Lyon vs Lille", "1X2", "Draw", 0.30, 3.65, 3.45, 75, "loss"],
            ["2026-01-10", "Milan vs Torino", "1X2", "Milan", 0.62, 1.80, 1.72, 100, "win"],
            ["2026-01-11", "Leeds vs Norwich", "Over/Under", "Under 2.5", 0.51, 2.05, 2.12, 75, "loss"],
            ["2026-01-12", "Dortmund vs Mainz", "BTTS", "Yes", 0.61, 1.78, 1.70, 100, "win"],
            ["2026-01-13", "Betis vs Getafe", "1X2", "Betis", 0.46, 2.32, 2.20, 75, "loss"],
            ["2026-01-14", "Spurs vs Everton", "Over/Under", "Over 2.5", 0.59, 1.86, 1.76, 100, "win"],
            ["2026-01-15", "Nice vs Rennes", "BTTS", "No", 0.53, 2.00, 1.95, 100, "push"],
            ["2026-01-16", "Chelsea vs Fulham", "1X2", "Chelsea", 0.55, 2.05, 1.92, 100, "loss"],
            ["2026-01-17", "Roma vs Bologna", "Over/Under", "Under 2.5", 0.56, 1.94, 1.88, 100, "win"],
            ["2026-01-18", "Villarreal vs Girona", "BTTS", "Yes", 0.63, 1.74, 1.68, 100, "win"],
            ["2026-01-19", "Bayern vs Freiburg", "1X2", "Bayern", 0.71, 1.50, 1.44, 100, "win"],
            ["2026-01-20", "West Ham vs Palace", "1X2", "Draw", 0.29, 3.85, 3.70, 75, "loss"],
            ["2026-01-21", "Atalanta vs Genoa", "Over/Under", "Over 2.5", 0.57, 1.95, 1.86, 100, "win"],
            ["2026-01-22", "Marseille vs Nantes", "BTTS", "No", 0.49, 2.12, 2.04, 75, "loss"],
            ["2026-01-23", "Sociedad vs Osasuna", "1X2", "Sociedad", 0.52, 2.10, 2.02, 100, "win"],
            ["2026-01-24", "Inter vs Udinese", "Over/Under", "Under 2.5", 0.47, 2.25, 2.35, 75, "loss"],
            ["2026-01-25", "PSG vs Reims", "BTTS", "Yes", 0.56, 1.98, 1.88, 100, "win"],
            ["2026-01-26", "Celtic vs Hearts", "1X2", "Celtic", 0.69, 1.55, 1.49, 100, "win"],
            ["2026-01-27", "Ajax vs Utrecht", "Over/Under", "Over 2.5", 0.64, 1.72, 1.66, 100, "loss"],
            ["2026-01-28", "Porto vs Braga", "BTTS", "No", 0.52, 2.02, 1.96, 100, "win"],
            ["2026-01-29", "Monaco vs Lens", "1X2", "Monaco", 0.50, 2.18, 2.06, 75, "loss"],
            ["2026-01-30", "Liverpool vs Villa", "Over/Under", "Over 2.5", 0.60, 1.84, 1.75, 100, "win"],
            ["2026-01-31", "Fiorentina vs Empoli", "BTTS", "Yes", 0.48, 2.18, 2.08, 75, "loss"],
            ["2026-02-01", "Benfica vs Rio Ave", "1X2", "Benfica", 0.67, 1.62, 1.55, 100, "win"],
            ["2026-02-02", "Leicester vs Hull", "Over/Under", "Under 2.5", 0.55, 1.98, 1.90, 100, "win"],
        ],
        columns=[
            "date",
            "event",
            "market",
            "selection",
            "model_probability",
            "odds",
            "closing_odds",
            "stake",
            "result",
        ],
    )


def _sidebar_inputs() -> dict:
    with st.sidebar:
        st.header("Setup")
        market_template = st.selectbox("Market template", options=list(DEFAULT_MARKETS))
        odds_format = st.radio(
            "Odds format",
            options=list(OddsFormat),
            format_func=lambda value: value.value.title(),
            horizontal=True,
        )
        bankroll = st.number_input(
            "Bankroll",
            min_value=0.0,
            value=1000.0,
            step=100.0,
            format="%.2f",
        )
        stake = st.number_input(
            "Reference stake",
            min_value=0.0,
            value=100.0,
            step=10.0,
            format="%.2f",
        )
        kelly_multiplier = st.slider(
            "Kelly fraction",
            min_value=0.0,
            max_value=1.0,
            value=0.25,
            step=0.05,
        )

    return {
        "market_template": market_template,
        "odds_format": odds_format,
        "bankroll": bankroll,
        "stake": stake,
        "kelly_multiplier": kelly_multiplier,
    }


def _football_poisson_inputs() -> dict:
    first_row = st.columns(2)
    home_team = first_row[0].text_input(
        "Home team",
        value="Home",
        key="poisson_home_team",
    )
    away_team = first_row[1].text_input(
        "Away team",
        value="Away",
        key="poisson_away_team",
    )

    second_row = st.columns(3)
    home_expected_goals = second_row[0].number_input(
        "Home expected goals",
        min_value=0.05,
        max_value=6.0,
        value=1.55,
        step=0.05,
        format="%.2f",
        key="poisson_home_xg",
    )
    away_expected_goals = second_row[1].number_input(
        "Away expected goals",
        min_value=0.05,
        max_value=6.0,
        value=1.10,
        step=0.05,
        format="%.2f",
        key="poisson_away_xg",
    )
    display_max_goals = second_row[2].slider(
        "Scoreline range",
        min_value=4,
        max_value=10,
        value=6,
        step=1,
        key="poisson_scoreline_range",
    )

    return {
        "home_team": home_team or "Home",
        "away_team": away_team or "Away",
        "home_expected_goals": home_expected_goals,
        "away_expected_goals": away_expected_goals,
        "display_max_goals": display_max_goals,
    }


def _football_book_odds_inputs(home_team: str, away_team: str) -> dict[str, float]:
    st.subheader("Book Prices")
    result_col, total_col, btts_col = st.columns(3)
    with result_col:
        st.caption("1X2")
        home_odds = st.number_input(
            f"{home_team} odds",
            min_value=1.01,
            value=2.10,
            step=0.01,
            format="%.2f",
            key="poisson_home_odds",
        )
        draw_odds = st.number_input(
            "Draw odds",
            min_value=1.01,
            value=3.55,
            step=0.01,
            format="%.2f",
            key="poisson_draw_odds",
        )
        away_odds = st.number_input(
            f"{away_team} odds",
            min_value=1.01,
            value=3.60,
            step=0.01,
            format="%.2f",
            key="poisson_away_odds",
        )
    with total_col:
        st.caption("Goals")
        over_odds = st.number_input(
            "Over 2.5 odds",
            min_value=1.01,
            value=1.95,
            step=0.01,
            format="%.2f",
            key="poisson_over_odds",
        )
        under_odds = st.number_input(
            "Under 2.5 odds",
            min_value=1.01,
            value=1.92,
            step=0.01,
            format="%.2f",
            key="poisson_under_odds",
        )
    with btts_col:
        st.caption("BTTS")
        btts_yes_odds = st.number_input(
            "BTTS yes odds",
            min_value=1.01,
            value=1.88,
            step=0.01,
            format="%.2f",
            key="poisson_btts_yes_odds",
        )
        btts_no_odds = st.number_input(
            "BTTS no odds",
            min_value=1.01,
            value=1.95,
            step=0.01,
            format="%.2f",
            key="poisson_btts_no_odds",
        )

    return {
        "home": home_odds,
        "draw": draw_odds,
        "away": away_odds,
        "over_2_5": over_odds,
        "under_2_5": under_odds,
        "btts_yes": btts_yes_odds,
        "btts_no": btts_no_odds,
    }


def _render_poisson_market_prices(
    *,
    markets,
    book_odds: dict[str, float],
    home_team: str,
    away_team: str,
    stake: float,
    bankroll: float,
    kelly_multiplier: float,
) -> None:
    st.subheader("Model Prices and Edge")
    rows = [
        ("1X2", home_team, markets.home_win, book_odds["home"]),
        ("1X2", "Draw", markets.draw, book_odds["draw"]),
        ("1X2", away_team, markets.away_win, book_odds["away"]),
        ("Total goals", "Over 2.5", markets.over_2_5, book_odds["over_2_5"]),
        ("Total goals", "Under 2.5", markets.under_2_5, book_odds["under_2_5"]),
        ("BTTS", "Yes", markets.btts_yes, book_odds["btts_yes"]),
        ("BTTS", "No", markets.btts_no, book_odds["btts_no"]),
    ]

    display_rows = []
    for market_name, selection, probability, odds in rows:
        edge = calculate_edge(
            decimal_odds=odds,
            model_probability=probability,
            stake=stake,
            bankroll=bankroll,
            kelly_multiplier=kelly_multiplier,
        )
        display_rows.append(
            {
                "Market": market_name,
                "Selection": selection,
                "Model probability": f"{probability:.2%}",
                "Model fair odds": f"{1.0 / probability:.3f}",
                "Book odds": f"{odds:.2f}",
                "Edge": f"{edge.edge_probability:.2%}",
                f"EV / {stake:,.0f} stake": f"{edge.expected_value_for_stake:,.2f}",
                "Recommended stake": f"{edge.recommended_stake:,.2f}",
            }
        )

    st.dataframe(pd.DataFrame(display_rows), use_container_width=True, hide_index=True)


def _render_score_probability_heatmap(
    *,
    markets,
    home_team: str,
    away_team: str,
    max_goals: int,
) -> None:
    st.subheader("Correct Score Probability Map")
    scores = [
        {
            "Home goals": score.home_goals,
            "Away goals": score.away_goals,
            "Probability": score.probability,
        }
        for score in markets.scores
        if score.home_goals <= max_goals and score.away_goals <= max_goals
    ]
    score_frame = pd.DataFrame(scores)
    matrix = score_frame.pivot(
        index="Home goals",
        columns="Away goals",
        values="Probability",
    ).sort_index(ascending=False)
    text = [[f"{value:.1%}" for value in row] for row in matrix.to_numpy()]
    heatmap_coverage = score_frame["Probability"].sum()

    figure = go.Figure(
        data=go.Heatmap(
            z=matrix.to_numpy(),
            x=[str(column) for column in matrix.columns],
            y=[str(index) for index in matrix.index],
            colorscale=[
                [0.0, "#111827"],
                [0.45, "#334155"],
                [1.0, "#22c55e"],
            ],
            text=text,
            texttemplate="%{text}",
            hovertemplate=(
                f"{home_team} %{{y}} - %{{x}} {away_team}<br>"
                "Probability: %{z:.2%}<extra></extra>"
            ),
            colorbar={"title": "Probability"},
        )
    )
    figure.update_layout(
        height=520,
        xaxis_title=f"{away_team} goals",
        yaxis_title=f"{home_team} goals",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(figure, use_container_width=True)

    cols = st.columns(2)
    cols[0].metric("Heatmap coverage", f"{heatmap_coverage:.2%}")
    cols[1].metric("Pricing grid coverage", f"{markets.score_coverage:.2%}")


def _render_top_correct_scores(*, markets, home_team: str, away_team: str) -> None:
    st.subheader("Most Likely Correct Scores")
    top_scores = sorted(markets.scores, key=lambda score: score.probability, reverse=True)[:10]
    rows = [
        {
            "Score": f"{home_team} {score.home_goals}-{score.away_goals} {away_team}",
            "Probability": f"{score.probability:.2%}",
            "Fair odds": f"{1.0 / score.probability:.2f}",
        }
        for score in top_scores
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _market_inputs(market_template: str, odds_format: OddsFormat) -> list[dict]:
    st.subheader("Market Prices and Model Probabilities")
    defaults = _default_market_rows(market_template, odds_format)
    rows = []
    columns = st.columns(len(defaults))
    for column, default in zip(columns, defaults, strict=True):
        with column:
            selection = st.text_input(
                "Selection",
                value=default["selection"],
                key=f"{market_template}_{default['selection']}_name",
            )
            odds_value = st.text_input(
                "Book odds",
                value=default["odds"],
                key=f"{market_template}_{default['selection']}_odds",
            )
            probability_key = f"{market_template}_{default['selection']}_probability"
            if probability_key not in st.session_state:
                st.session_state[probability_key] = default["model_probability_pct"]
            model_probability_pct = st.number_input(
                "Model probability (%)",
                min_value=0.01,
                max_value=99.99,
                step=0.25,
                format="%.2f",
                key=probability_key,
            )
        rows.append(
            {
                "selection": selection,
                "odds_value": odds_value,
                "odds_format": odds_format,
                "model_probability": model_probability_pct / 100.0,
            }
        )
    return rows


def _render_probability_model_helper(market_template: str) -> None:
    defaults = _default_market_rows(market_template, OddsFormat.DECIMAL)
    with st.expander("Quick Rating Model"):
        if len(defaults) == 2:
            rating_a, rating_b, adjustment, scale = _two_way_rating_inputs()
            probability_a = elo_win_probability(
                rating_a=rating_a,
                rating_b=rating_b,
                rating_adjustment_a=adjustment,
                scale=scale,
            )
            probabilities = [probability_a, 1.0 - probability_a]
        else:
            home_rating, away_rating, home_advantage, draw_strength, scale = (
                _three_way_rating_inputs()
            )
            probabilities = list(
                davidson_three_way_probabilities(
                    home_rating=home_rating,
                    away_rating=away_rating,
                    home_advantage=home_advantage,
                    draw_strength=draw_strength,
                    scale=scale,
                )
            )

        model_rows = pd.DataFrame(
            {
                "Selection": [row["selection"] for row in defaults],
                "Suggested model probability": [f"{value:.2%}" for value in probabilities],
                "Model fair odds": [f"{1.0 / value:.3f}" for value in probabilities],
            }
        )
        st.dataframe(model_rows, use_container_width=True, hide_index=True)

        if st.button("Apply probabilities to market inputs"):
            for row, probability in zip(defaults, probabilities, strict=True):
                key = f"{market_template}_{row['selection']}_probability"
                st.session_state[key] = round(probability * 100.0, 2)
            st.rerun()


def _two_way_rating_inputs() -> tuple[float, float, float, float]:
    cols = st.columns(4)
    rating_a = cols[0].number_input(
        "Selection A rating",
        value=1500.0,
        step=10.0,
        format="%.0f",
    )
    rating_b = cols[1].number_input(
        "Selection B rating",
        value=1460.0,
        step=10.0,
        format="%.0f",
    )
    adjustment = cols[2].number_input(
        "Adjustment A",
        value=35.0,
        step=5.0,
        format="%.0f",
    )
    scale = cols[3].number_input(
        "Elo scale",
        min_value=1.0,
        value=400.0,
        step=25.0,
        format="%.0f",
    )
    return rating_a, rating_b, adjustment, scale


def _three_way_rating_inputs() -> tuple[float, float, float, float, float]:
    first_row = st.columns(3)
    home_rating = first_row[0].number_input(
        "Home rating",
        value=1500.0,
        step=10.0,
        format="%.0f",
    )
    away_rating = first_row[1].number_input(
        "Away rating",
        value=1460.0,
        step=10.0,
        format="%.0f",
    )
    home_advantage = first_row[2].number_input(
        "Home advantage",
        value=65.0,
        step=5.0,
        format="%.0f",
    )
    second_row = st.columns(2)
    draw_strength = second_row[0].slider(
        "Draw strength",
        min_value=0.0,
        max_value=2.0,
        value=0.85,
        step=0.05,
    )
    scale = second_row[1].number_input(
        "Elo scale",
        min_value=1.0,
        value=400.0,
        step=25.0,
        format="%.0f",
    )
    return home_rating, away_rating, home_advantage, draw_strength, scale


def _build_market(rows: list[dict]) -> pd.DataFrame:
    records = []
    for row in rows:
        decimal_odds = to_decimal_odds(row["odds_value"], row["odds_format"])
        records.append(
            {
                "Selection": row["selection"],
                "Book odds": decimal_odds,
                "American": decimal_to_american(decimal_odds),
                "Fractional": decimal_to_fractional(decimal_odds),
                "Market probability": implied_probability(decimal_odds),
                "Model probability": row["model_probability"],
            }
        )

    dataframe = pd.DataFrame(records)
    fair_probabilities = remove_overround(dataframe["Market probability"].tolist())
    dataframe["No-vig probability"] = fair_probabilities
    dataframe["No-vig fair odds"] = 1.0 / dataframe["No-vig probability"]
    dataframe["Model fair odds"] = 1.0 / dataframe["Model probability"]
    return dataframe


def _render_market_summary(market: pd.DataFrame) -> None:
    market_overround = overround(market["Market probability"].tolist())
    best_edge_row = market.assign(
        Edge=market["Model probability"] - market["Market probability"]
    ).sort_values("Edge", ascending=False).iloc[0]

    overround_col, edge_col, best_col = st.columns(3)
    overround_col.metric("Book overround", f"{market_overround:.2%}")
    edge_col.metric("Top model edge", f"{best_edge_row['Edge']:.2%}")
    best_col.metric("Best selection", best_edge_row["Selection"])

    display = market.copy()
    percent_columns = ["Market probability", "No-vig probability", "Model probability"]
    odds_columns = ["Book odds", "No-vig fair odds", "Model fair odds"]
    for column in percent_columns:
        display[column] = display[column].map(lambda value: f"{value:.2%}")
    for column in odds_columns:
        display[column] = display[column].map(lambda value: f"{value:.3f}")
    st.dataframe(display, use_container_width=True, hide_index=True)


def _render_edge_table(
    *,
    market: pd.DataFrame,
    bankroll: float,
    stake: float,
    kelly_multiplier: float,
) -> None:
    st.subheader("Edge and Stake Sizing")
    rows = []
    for _, row in market.iterrows():
        edge = calculate_edge(
            decimal_odds=row["Book odds"],
            model_probability=row["Model probability"],
            stake=stake,
            bankroll=bankroll,
            kelly_multiplier=kelly_multiplier,
        )
        rows.append(
            {
                "Selection": row["Selection"],
                "Edge": f"{edge.edge_probability:.2%}",
                "EV / unit": f"{edge.expected_value_per_unit:.3f}",
                f"EV / {stake:,.0f} stake": f"{edge.expected_value_for_stake:,.2f}",
                "Break-even": f"{edge.break_even_probability:.2%}",
                "Full Kelly": f"{edge.full_kelly_fraction:.2%}",
                "Fractional Kelly": f"{edge.fractional_kelly_fraction:.2%}",
                "Recommended stake": f"{edge.recommended_stake:,.2f}",
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_line_shopping(odds_format: OddsFormat) -> None:
    with st.expander("Line Shopping"):
        st.caption("Compare prices for one selection across books or exchanges.")
        rows = []
        columns = st.columns(4)
        try:
            for index, column in enumerate(columns, start=1):
                with column:
                    book = st.text_input("Book", value=f"Book {index}", key=f"book_{index}")
                    odds_value = st.text_input(
                        "Odds",
                        value=["2.05", "2.00", "1.98", "2.10"][index - 1],
                        key=f"book_odds_{index}",
                    )
                decimal_odds = to_decimal_odds(odds_value, odds_format)
                rows.append(
                    {
                        "Book": book,
                        "Decimal odds": decimal_odds,
                        "Implied probability": implied_probability(decimal_odds),
                    }
                )
        except ValueError as error:
            st.error(f"Check the line shopping odds: {error}")
            return

        dataframe = pd.DataFrame(rows).sort_values("Decimal odds", ascending=False)
        dataframe["Best price"] = ["Yes" if index == 0 else "" for index in range(len(dataframe))]
        display = dataframe.copy()
        display["Decimal odds"] = display["Decimal odds"].map(lambda value: f"{value:.3f}")
        display["Implied probability"] = display["Implied probability"].map(
            lambda value: f"{value:.2%}"
        )
        st.dataframe(display, use_container_width=True, hide_index=True)


def _render_pnl_chart(market: pd.DataFrame, *, stake: float) -> None:
    st.subheader("Outcome PnL")
    selections = market["Selection"].tolist()
    figure = go.Figure()
    for selected in selections:
        row = market[market["Selection"] == selected].iloc[0]
        pnl = [
            stake * (row["Book odds"] - 1.0) if outcome == selected else -stake
            for outcome in selections
        ]
        figure.add_trace(
            go.Bar(
                x=selections,
                y=pnl,
                name=selected,
            )
        )
    figure.add_hline(y=0, line_dash="dot", line_color="#94a3b8")
    figure.update_layout(
        barmode="group",
        xaxis_title="Actual outcome",
        yaxis_title="PnL",
        height=440,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(figure, use_container_width=True)


def _render_backtest() -> None:
    with st.expander("Backtest Bet Log"):
        st.caption("Upload a CSV with columns: stake, odds, result, and optionally closing_odds.")
        uploaded_file = st.file_uploader("Bet log CSV", type=["csv"])
        if uploaded_file is None:
            example = pd.DataFrame(
                [
                    {"stake": 100, "odds": 2.10, "result": "win", "closing_odds": 1.95},
                    {"stake": 100, "odds": 1.80, "result": "loss", "closing_odds": 1.75},
                    {"stake": 100, "odds": 2.40, "result": "win", "closing_odds": 2.20},
                    {"stake": 100, "odds": 1.95, "result": "loss", "closing_odds": 1.90},
                ]
            )
            st.dataframe(example, use_container_width=True, hide_index=True)
            return

        bets = pd.read_csv(uploaded_file)
        try:
            summary, analyzed = analyze_bets(bets)
        except ValueError as error:
            st.error(f"Could not analyze the uploaded bet log: {error}")
            return
        cols = st.columns(5)
        cols[0].metric("Bets", f"{summary.bets}")
        cols[1].metric("Profit", f"{summary.profit:,.2f}")
        cols[2].metric("ROI", f"{summary.roi:.2%}")
        cols[3].metric("Hit rate", f"{summary.hit_rate:.2%}")
        cols[4].metric("Max drawdown", f"{summary.max_drawdown:,.2f}")

        figure = go.Figure(
            data=go.Scatter(
                x=list(range(1, len(analyzed) + 1)),
                y=analyzed["equity"],
                mode="lines+markers",
                name="Equity",
            )
        )
        figure.add_hline(y=0, line_dash="dot", line_color="#94a3b8")
        figure.update_layout(
            xaxis_title="Bet number",
            yaxis_title="Cumulative profit",
            height=360,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(figure, use_container_width=True)
        st.dataframe(analyzed, use_container_width=True, hide_index=True)


def _render_model_notes() -> None:
    with st.expander("Model Notes"):
        st.markdown(
            """
            - Book odds imply probabilities, but those probabilities include bookmaker
              margin. This app removes margin with proportional normalization.
            - Positive expected value appears when your model probability exceeds the
              break-even probability implied by the available price.
            - Kelly sizing is sensitive to model error, so the default uses quarter
              Kelly rather than full Kelly.
            - Closing-line value is treated as improvement in implied probability from
              your entry price to the closing price.
            """
        )


def _default_market_rows(market_template: str, odds_format: OddsFormat) -> list[dict]:
    selections = DEFAULT_MARKETS[market_template]
    if len(selections) == 2:
        decimal_odds = [2.05, 1.87]
        probabilities = [52.0, 50.0]
    else:
        decimal_odds = [2.20, 3.45, 3.20]
        probabilities = [47.0, 27.0, 31.0]

    return [
        {
            "selection": selection,
            "odds": _format_odds(decimal_odds[index], odds_format),
            "model_probability_pct": probabilities[index],
        }
        for index, selection in enumerate(selections)
    ]


def _format_odds(decimal_odds: float, odds_format: OddsFormat) -> str:
    if odds_format == OddsFormat.DECIMAL:
        return f"{decimal_odds:.2f}"
    if odds_format == OddsFormat.AMERICAN:
        return str(decimal_to_american(decimal_odds))
    return decimal_to_fractional(decimal_odds)


def _apply_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 100%;
            padding-top: 1rem;
            padding-right: 1rem;
            padding-bottom: 2rem;
            padding-left: 1rem;
        }
        header[data-testid="stHeader"],
        div[data-testid="stToolbar"],
        div[data-testid="stDecoration"] {
            display: none;
            height: 0;
        }
        div[data-testid="stMetric"] {
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 1rem;
            background: #111827;
            box-shadow: none;
        }
        div[data-testid="stMetric"] * {
            color: #f8fafc !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
