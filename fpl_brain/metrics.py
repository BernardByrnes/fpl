"""Transparent derived metrics that read from SQLite only."""

from __future__ import annotations

import sqlite3
from collections import Counter
from statistics import mean
from typing import Any

from . import repositories as repo
from .utils import parse_utc, subtract_days


def fixture_outlook(conn: sqlite3.Connection, team_id: int, from_event: int, horizon: int) -> dict[str, Any]:
    """Return each scheduled fixture and its raw FDR for a team."""

    fixtures = []
    for row in repo.fixture_rows(conn, from_event, horizon):
        if row["team_h"] == team_id:
            fixtures.append(
                {
                    "fixture_id": row["id"],
                    "event": row["event"],
                    "opponent_team": row["team_a"],
                    "home": True,
                    "difficulty": row["team_h_difficulty"],
                    "kickoff_time": row["kickoff_time"],
                }
            )
        elif row["team_a"] == team_id:
            fixtures.append(
                {
                    "fixture_id": row["id"],
                    "event": row["event"],
                    "opponent_team": row["team_h"],
                    "home": False,
                    "difficulty": row["team_a_difficulty"],
                    "kickoff_time": row["kickoff_time"],
                }
            )
    difficulties = [item["difficulty"] for item in fixtures if item["difficulty"] is not None]
    event_counts = Counter(item["event"] for item in fixtures)
    return {
        "fixtures": fixtures,
        "mean_fdr": mean(difficulties) if difficulties else None,
        "fixture_count": len(fixtures),
        "home_count": sum(1 for item in fixtures if item["home"]),
        "blank_count": max(0, int(horizon) - len(fixtures)),
        "double_events": sorted(event for event, count in event_counts.items() if count > 1),
        "horizon": horizon,
        "from_event": from_event,
    }


def attacking_and_defensive_outlook(
    conn: sqlite3.Connection,
    team_id: int,
    from_event: int,
    horizon: int,
) -> dict[str, Any]:
    """Average the opponent's context-aware defence and attack strengths."""

    values: list[tuple[int | float, int | float]] = []
    for row in repo.fixture_rows(conn, from_event, horizon):
        if row["team_h"] == team_id:
            opponent_id = row["team_a"]
            opponent_side = "away"
        elif row["team_a"] == team_id:
            opponent_id = row["team_h"]
            opponent_side = "home"
        else:
            continue
        opponent = repo.team_row(conn, opponent_id) if opponent_id is not None else None
        if opponent is None:
            continue
        defence = opponent[f"strength_defence_{opponent_side}"]
        attack = opponent[f"strength_attack_{opponent_side}"]
        if defence is not None and attack is not None:
            values.append((defence, attack))
    attacking = mean(value[0] for value in values) if values else None
    defensive = mean(value[1] for value in values) if values else None
    return {
        "attacking_outlook": attacking,
        "defensive_outlook": defensive,
        "mean_opponent_defence": attacking,
        "mean_opponent_attack": defensive,
        "fixture_count": len(values),
    }


def points_per_million(conn: sqlite3.Connection, player_id: int) -> float | None:
    """Compute latest total points divided by current price in millions."""

    snapshot = repo.latest_snapshot(conn, player_id)
    if not snapshot or snapshot["total_points"] is None or not snapshot["now_cost"]:
        return None
    return float(snapshot["total_points"]) / (float(snapshot["now_cost"]) / 10.0)


def trend(
    conn: sqlite3.Connection,
    player_id: int,
    field: str,
    lookback_days: int,
) -> dict[str, Any] | None:
    """Compare the first and last non-null snapshot values in a time window."""

    all_rows = repo.snapshot_history(conn, player_id, field)
    if not all_rows:
        return None
    latest_time = all_rows[-1]["captured_at"]
    since = subtract_days(latest_time, lookback_days)
    rows = repo.snapshot_history(conn, player_id, field, since)
    rows = [row for row in rows if row["value"] is not None]
    if len(rows) < 2:
        return None
    first = rows[0]["value"]
    last = rows[-1]["value"]
    delta = last - first
    percentage = None if first == 0 else (delta / first) * 100.0
    return {
        "field": field,
        "first": first,
        "last": last,
        "first_value": first,
        "last_value": last,
        "absolute_delta": delta,
        "percentage_delta": percentage,
        "sample_count": len(rows),
        "from": rows[0]["captured_at"],
        "to": rows[-1]["captured_at"],
    }


def minutes_reliability(
    conn: sqlite3.Connection,
    player_id: int,
    last_n_events: int,
) -> dict[str, Any] | None:
    """Summarise starts, appearances, and minutes from finished fixtures only."""

    rows = repo.gameweek_rows(conn, player_id, last_n_events)
    if not rows:
        return None
    minutes = [row["minutes"] for row in rows if row["minutes"] is not None]
    return {
        "starts": sum(1 for row in rows if row["starts"] not in (None, 0)),
        "appearances": sum(1 for row in rows if row["minutes"] not in (None, 0)),
        "total_minutes": sum(minutes),
        "mean_minutes": mean(minutes) if minutes else None,
        "events_sampled": len(rows),
    }


def _expected_minutes_stub(*_: Any, **__: Any) -> None:
    """Intended definition: a scouting-backed expected minutes estimate."""


def _start_probability_stub(*_: Any, **__: Any) -> None:
    """Intended definition: a scouting-backed probability of starting."""


def _defcon_potential_stub(*_: Any, **__: Any) -> None:
    """Intended definition: a scouting-backed DEFCON contribution expectation."""


def _route_diversity_stub(*_: Any, **__: Any) -> None:
    """Intended definition: count of independent routes to points from official/scouting evidence."""


def _role_security_stub(*_: Any, **__: Any) -> None:
    """Intended definition: a scouting-backed role security assessment over five gameweeks."""


def _squad_flexibility_stub(*_: Any, **__: Any) -> None:
    """Intended definition: legal squad formations and price-point flexibility around a player."""


def _future_transfer_pressure_stub(*_: Any, **__: Any) -> None:
    """Intended definition: upcoming fixtures, price, ownership, and squad constraints that create pressure."""


def _replacement_options_stub(*_: Any, **__: Any) -> None:
    """Intended definition: transparent same-position alternatives without ranking them."""


METRIC_REGISTRY: dict[str, Any] = {
    "fixture_outlook": fixture_outlook,
    "attacking_and_defensive_outlook": attacking_and_defensive_outlook,
    "points_per_million": points_per_million,
    "trend": trend,
    "minutes_reliability": minutes_reliability,
    "expected_minutes": None,
    "start_probability": None,
    "defcon_potential": None,
    "route_diversity": None,
    "role_security": None,
    "squad_flexibility": None,
    "future_transfer_pressure": None,
    "replacement_options": None,
}
