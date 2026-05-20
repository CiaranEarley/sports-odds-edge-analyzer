"""Market data helpers for external probability feeds."""

from __future__ import annotations

import json
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


POLYMARKET_GAMMA_URL = "https://gamma-api.polymarket.com"


class MarketDataError(RuntimeError):
    """Raised when external market data cannot be fetched or parsed."""


@dataclass(frozen=True)
class MarketProbabilityRow:
    """One outcome probability from a market data source."""

    source: str
    event: str
    market: str
    outcome: str
    probability: float
    event_url: str
    volume: float | None = None
    liquidity: float | None = None
    end_date: str | None = None


def sample_market_rows() -> pd.DataFrame:
    """Return a small static dataset for demos and offline work."""

    rows = [
        MarketProbabilityRow(
            source="Sample",
            event="Arsenal vs Brighton",
            market="Match winner",
            outcome="Arsenal",
            probability=0.56,
            volume=125000,
            liquidity=18000,
            end_date="2026-03-14",
            event_url="",
        ),
        MarketProbabilityRow(
            source="Sample",
            event="Arsenal vs Brighton",
            market="Match winner",
            outcome="Brighton",
            probability=0.44,
            volume=125000,
            liquidity=18000,
            end_date="2026-03-14",
            event_url="",
        ),
        MarketProbabilityRow(
            source="Sample",
            event="Liverpool vs Aston Villa",
            market="Over/Under 2.5",
            outcome="Over 2.5",
            probability=0.59,
            volume=87000,
            liquidity=12500,
            end_date="2026-03-15",
            event_url="",
        ),
        MarketProbabilityRow(
            source="Sample",
            event="Liverpool vs Aston Villa",
            market="Over/Under 2.5",
            outcome="Under 2.5",
            probability=0.41,
            volume=87000,
            liquidity=12500,
            end_date="2026-03-15",
            event_url="",
        ),
        MarketProbabilityRow(
            source="Sample",
            event="Bayern vs Freiburg",
            market="Match winner",
            outcome="Bayern",
            probability=0.70,
            volume=190000,
            liquidity=24000,
            end_date="2026-03-16",
            event_url="",
        ),
        MarketProbabilityRow(
            source="Sample",
            event="Bayern vs Freiburg",
            market="Match winner",
            outcome="Freiburg",
            probability=0.30,
            volume=190000,
            liquidity=24000,
            end_date="2026-03-16",
            event_url="",
        ),
    ]
    return market_rows_to_frame(rows)


def fetch_polymarket_events(
    *,
    query: str = "",
    limit: int = 10,
    timeout: float = 10.0,
) -> list[dict]:
    """Fetch active Polymarket events from public Gamma endpoints."""

    if limit < 1:
        raise ValueError("limit must be at least 1.")

    if query.strip():
        payload = _get_json(
            "/public-search",
            {
                "q": query.strip(),
                "events_status": "active",
                "limit_per_type": limit,
                "search_profiles": "false",
            },
            timeout=timeout,
        )
        events = payload.get("events", []) if isinstance(payload, dict) else []
        return events[:limit]

    payload = _get_json(
        "/events",
        {
            "active": "true",
            "closed": "false",
            "order": "volume_24hr",
            "ascending": "false",
            "limit": limit,
        },
        timeout=timeout,
    )
    if not isinstance(payload, list):
        raise MarketDataError("Polymarket events response was not a list.")
    return payload


def polymarket_events_to_frame(events: list[dict]) -> pd.DataFrame:
    """Convert Polymarket event payloads into an outcome-probability table."""

    rows: list[MarketProbabilityRow] = []
    for event in events:
        event_title = str(event.get("title") or event.get("ticker") or "Untitled event")
        event_url = _event_url(event)
        event_markets = event.get("markets") or []
        if not isinstance(event_markets, list):
            event_markets = []

        for market in event_markets:
            rows.extend(_polymarket_market_rows(event, market, event_title, event_url))

    return market_rows_to_frame(rows)


def market_rows_to_frame(rows: list[MarketProbabilityRow]) -> pd.DataFrame:
    """Convert market rows into a consistently shaped DataFrame."""

    records = [
        {
            "Source": row.source,
            "Event": row.event,
            "Market": row.market,
            "Outcome": row.outcome,
            "Market probability": row.probability,
            "Market fair odds": 1.0 / row.probability,
            "Volume": row.volume,
            "Liquidity": row.liquidity,
            "End date": row.end_date,
            "Event URL": row.event_url,
        }
        for row in rows
        if 0 < row.probability < 1
    ]
    return pd.DataFrame.from_records(records)


def _get_json(path: str, params: dict[str, object], *, timeout: float) -> object:
    url = f"{POLYMARKET_GAMMA_URL}{path}?{urlencode(params, doseq=True)}"
    request = Request(url, headers={"User-Agent": "sports-odds-edge-analyzer/0.1"})
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except HTTPError as error:
        raise MarketDataError(f"Polymarket returned HTTP {error.code}.") from error
    except URLError as error:
        raise MarketDataError(f"Could not reach Polymarket: {error.reason}") from error
    except json.JSONDecodeError as error:
        raise MarketDataError("Polymarket response was not valid JSON.") from error


def _polymarket_market_rows(
    event: dict,
    market: dict,
    event_title: str,
    event_url: str,
) -> list[MarketProbabilityRow]:
    outcomes = _parse_sequence(market.get("outcomes"))
    prices = _parse_sequence(market.get("outcomePrices"))
    if not outcomes or not prices or len(outcomes) != len(prices):
        return []

    volume = _first_number(market, event, keys=["volumeNum", "volume", "volume24hr"])
    liquidity = _first_number(market, event, keys=["liquidityNum", "liquidity"])
    market_title = str(market.get("question") or market.get("title") or event_title)
    end_date = str(market.get("endDate") or event.get("endDate") or "")

    rows = []
    for outcome, price in zip(outcomes, prices, strict=True):
        probability = _to_float(price)
        if probability is None or not 0 < probability < 1:
            continue
        rows.append(
            MarketProbabilityRow(
                source="Polymarket",
                event=event_title,
                market=market_title,
                outcome=str(outcome),
                probability=probability,
                volume=volume,
                liquidity=liquidity,
                end_date=end_date,
                event_url=event_url,
            )
        )
    return rows


def _event_url(event: dict) -> str:
    slug = event.get("slug")
    if not slug:
        return ""
    return f"https://polymarket.com/event/{slug}"


def _parse_sequence(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def _first_number(*records: dict, keys: list[str]) -> float | None:
    for record in records:
        for key in keys:
            value = _to_float(record.get(key))
            if value is not None:
                return value
    return None


def _to_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
