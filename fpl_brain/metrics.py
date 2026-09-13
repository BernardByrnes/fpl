"""Transparent derived metrics that read from SQLite only."""

from __future__ import annotations

import sqlite3
from collections import Counter
from statistics import mean
from typing import Any, Sequence

from . import repositories as repo
from .utils import parse_utc, subtract_days


def fixture_outlook(conn: sqlite3.Connection, team_id: int, from_event: int, horizon: int) -> dict[str, Any]:
    """Return each scheduled fixture and its raw FDR for a team."""

    fixtures = []
    for row in repo.future_fixture_rows(conn, from_event, horizon):
        if row["team_h"] == team_id:
            fixtures.append(
                {
                    "fixture_id": row["id"],
                    "event": row["event"],
                    "horizon_event": row.get("horizon_event", row["event"]),
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
                    "horizon_event": row.get("horizon_event", row["event"]),
                    "opponent_team": row["team_h"],
                    "home": False,
                    "difficulty": row["team_a_difficulty"],
                    "kickoff_time": row["kickoff_time"],
                }
            )
    difficulties = [item["difficulty"] for item in fixtures if item["difficulty"] is not None]
    event_counts = Counter(item["horizon_event"] for item in fixtures)
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
    for row in repo.future_fixture_rows(conn, from_event, horizon):
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


def transfer_affordability(
    conn: sqlite3.Connection,
    entry_id: int,
    event: int,
    outgoing_player_ids: Sequence[int],
    incoming_player_ids: Sequence[int],
) -> dict[str, Any]:
    """Compare effective outgoing selling values with official incoming prices."""

    state = repo.manager_planning_state(conn, entry_id, event)
    official_market_prices: dict[int, int] = {}
    data_gaps: list[str] = []
    mismatches: list[str] = []
    effective_prices: dict[int, int] = {}
    selling_sources: dict[int, str] = {}

    bank = state.get("bank")
    if bank is None:
        data_gaps.append(f"bank missing for entry {entry_id}, GW{event}")

    for player_id in outgoing_player_ids:
        player_id = int(player_id)
        value = effective_selling_price(conn, entry_id, event, player_id)
        if value["status"] == "MISMATCH":
            mismatches.append(f"player {player_id}: SELLING PRICE MISMATCH")
        elif value["effective_selling_price"] is None:
            data_gaps.append(f"selling price unavailable for outgoing player {player_id}")
        else:
            effective_prices[player_id] = int(value["effective_selling_price"])
            selling_sources[player_id] = str(value["source"])

    for player_id in incoming_player_ids:
        player_id = int(player_id)
        snapshot = repo.latest_snapshot(conn, player_id)
        if not snapshot or snapshot.get("now_cost") is None:
            data_gaps.append(f"official market price missing for incoming player {player_id}")
        else:
            official_market_prices[player_id] = int(snapshot["now_cost"])

    result: dict[str, Any] = {
        "status": "SELLING_PRICE_MISMATCH" if mismatches else "DATA_GAP" if data_gaps else "AFFORDABLE",
        "bank": bank,
        "free_transfers": state.get("free_transfers"),
        "free_transfers_source": state.get("free_transfers_source"),
        "selling_prices": effective_prices,
        "effective_selling_prices": effective_prices,
        "selling_price_sources": selling_sources,
        "official_market_prices": official_market_prices,
        "outgoing_selling_value": None,
        "incoming_market_value": None,
        "available_funds": None,
        "remaining_bank": None,
        "shortfall": None,
        "data_gaps": data_gaps,
        "mismatches": mismatches,
    }
    if data_gaps or mismatches:
        return result

    outgoing_value = sum(effective_prices[int(player_id)] for player_id in outgoing_player_ids)
    incoming_value = sum(official_market_prices[int(player_id)] for player_id in incoming_player_ids)
    remaining_bank = int(bank) + outgoing_value - incoming_value
    result.update(
        {
            "status": "AFFORDABLE" if remaining_bank >= 0 else "INSUFFICIENT_FUNDS",
            "outgoing_selling_value": outgoing_value,
            "incoming_market_value": incoming_value,
            "available_funds": remaining_bank,
            "remaining_bank": remaining_bank,
            "shortfall": max(0, -remaining_bank),
        }
    )
    return result


def calculate_selling_price(purchase_price: int, official_market_price: int) -> int:
    """Apply the FPL integer-tenths selling-price rule."""

    if isinstance(purchase_price, bool) or isinstance(official_market_price, bool):
        raise ValueError("prices must be integer tenths values")
    if not isinstance(purchase_price, int) or not isinstance(official_market_price, int):
        raise ValueError("prices must be integer tenths values")
    if purchase_price < 0 or official_market_price < 0:
        raise ValueError("prices must not be negative")
    if official_market_price <= purchase_price:
        return official_market_price
    return purchase_price + (official_market_price - purchase_price) // 2


def effective_selling_price(
    conn: sqlite3.Connection,
    entry_id: int,
    event: int,
    player_id: int,
) -> dict[str, Any]:
    """Resolve one player's current realisable selling price with provenance."""

    acquisition = next(
        (row for row in repo.active_manager_acquisitions(conn, int(entry_id)) if int(row["player_id"]) == int(player_id)),
        None,
    )
    snapshot = repo.latest_snapshot(conn, int(player_id)) or {}
    manual = repo.get_manager_selling_price(conn, int(entry_id), int(event), int(player_id))
    official = snapshot.get("now_cost")
    result: dict[str, Any] = {
        "player_id": int(player_id),
        "purchase_price": acquisition.get("purchase_price") if acquisition else None,
        "official_market_price": official,
        "calculated_selling_price": None,
        "manual_selling_price": manual.get("selling_price") if manual else None,
        "manual_market_price_at_capture": manual.get("market_price_at_capture") if manual else None,
        "effective_selling_price": None,
        "source": None,
        "status": "DATA_GAP",
    }
    if official is None:
        result["source"] = "official_market_price_missing"
        return result

    comparable = manual is not None and (
        manual.get("market_price_at_capture") is None
        or int(manual["market_price_at_capture"]) == int(official)
    )
    if acquisition is not None:
        calculated = calculate_selling_price(int(acquisition["purchase_price"]), int(official))
        result["calculated_selling_price"] = calculated
        if manual is not None and comparable:
            if int(manual["selling_price"]) != calculated:
                result.update({"source": "manual_verification", "status": "MISMATCH"})
                return result
            result.update(
                {
                    "effective_selling_price": calculated,
                    "source": "acquisition_formula+manual_verification",
                    "status": "VERIFIED_BY_MANUAL",
                }
            )
            return result
        result.update(
            {
                "effective_selling_price": calculated,
                "source": "acquisition_formula",
                "status": "AUTO_CALCULATED",
            }
        )
        return result

    if manual is not None and comparable:
        result.update(
            {
                "effective_selling_price": int(manual["selling_price"]),
                "source": "manual_snapshot",
                "status": "MANUAL_FALLBACK",
            }
        )
    elif manual is not None:
        result["source"] = "manual_snapshot_stale"
    else:
        result["source"] = "acquisition_history_missing"
    return result


def realisable_squad_value(conn: sqlite3.Connection, entry_id: int, event: int) -> dict[str, Any]:
    """Sum effective prices for a verified current squad, preserving data gaps."""

    exact = repo.complete_squad_rows(conn, int(entry_id), int(event))
    squad_event = int(event) if exact else None
    squad = exact
    squad_source = "exact_target_event"
    transfers_available, transfers = repo.manager_transfer_history(conn, int(entry_id))
    free_hit_events = repo.manager_chip_events(conn, int(entry_id), "freehit")
    if not squad:
        fallback = repo.latest_complete_squad_rows(
            conn,
            int(entry_id),
            int(event),
            ascending=False,
            exclude_events=free_hit_events,
        )
        if fallback is None or not transfers_available:
            return {
                "status": "DATA_GAP",
                "entry_id": int(entry_id),
                "event": int(event),
                "squad_source": None,
                "squad_event": None,
                "player_count": 0,
                "official_market_value": None,
                "realisable_selling_value": None,
                "players": [],
                "data_gaps": ["verified current squad unavailable or transfer history unavailable"],
                "mismatches": [],
            }
        squad_event, squad = fallback
        player_ids = {int(row["player_id"]) for row in squad}
        # Free-Hit transfers are temporary and must not rewrite the permanent squad.
        for transfer in sorted(
            (row for row in transfers if int(row.get("event", 0)) not in free_hit_events),
            key=lambda row: (int(row.get("event", 0)), row.get("time") or ""),
        ):
            if int(transfer.get("event", 0)) > int(squad_event) and int(transfer.get("event", 0)) <= int(event):
                player_ids.discard(int(transfer["element_out"]))
                player_ids.add(int(transfer["element_in"]))
        if len(player_ids) != 15:
            return {
                "status": "DATA_GAP",
                "entry_id": int(entry_id),
                "event": int(event),
                "squad_source": "latest_complete_snapshot+transfer_history",
                "squad_event": squad_event,
                "player_count": len(player_ids),
                "official_market_value": None,
                "realisable_selling_value": None,
                "players": [],
                "data_gaps": ["transfer history does not reconcile to a 15-player squad"],
                "mismatches": [],
            }
        known = {int(row["player_id"]): row for row in squad}
        squad = [known[player_id] for player_id in sorted(player_ids) if player_id in known]
        for player_id in sorted(player_ids - set(known)):
            player = repo.get_player(conn, player_id) or {"player_id": player_id, "id": player_id}
            squad.append({"player_id": player_id, **player})
        squad_source = "latest_complete_snapshot+transfer_history"

    players: list[dict[str, Any]] = []
    data_gaps: list[str] = []
    mismatches: list[str] = []
    official_total = 0
    selling_total = 0
    for row in squad:
        player_id = int(row["player_id"])
        value = effective_selling_price(conn, int(entry_id), int(event), player_id)
        if value["official_market_price"] is None:
            data_gaps.append(f"official market price missing for player {player_id}")
        else:
            official_total += int(value["official_market_price"])
        if value["status"] == "MISMATCH":
            mismatches.append(f"player {player_id}: SELLING PRICE MISMATCH")
        if value["effective_selling_price"] is None:
            if value["status"] != "MISMATCH":
                data_gaps.append(f"effective selling price missing for player {player_id}")
        else:
            selling_total += int(value["effective_selling_price"])
        players.append({**row, **value})
    status = "SELLING_PRICE_MISMATCH" if mismatches else "DATA_GAP" if data_gaps else "OK"
    return {
        "status": status,
        "entry_id": int(entry_id),
        "event": int(event),
        "squad_source": squad_source,
        "squad_event": squad_event,
        "player_count": len(squad),
        "official_market_value": official_total if not data_gaps else None,
        "realisable_selling_value": selling_total if not data_gaps and not mismatches else None,
        "players": players,
        "data_gaps": data_gaps,
        "mismatches": mismatches,
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
    "transfer_affordability": transfer_affordability,
    "realisable_squad_value": realisable_squad_value,
    "expected_minutes": None,
    "start_probability": None,
    "defcon_potential": None,
    "route_diversity": None,
    "role_security": None,
    "squad_flexibility": None,
    "future_transfer_pressure": None,
    "replacement_options": None,
}
