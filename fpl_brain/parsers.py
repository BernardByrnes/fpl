"""The single module that knows the public FPL JSON field names."""

from __future__ import annotations

from typing import Any, Callable

from .models import (
    BootstrapRecords,
    ChipRecord,
    EntryRecord,
    EventRecord,
    FixtureRecord,
    HistoryRow,
    ManagerChipRecord,
    ManagerHistory,
    ManagerTransferRecord,
    PickRecord,
    PicksRecord,
    PlayerGameweekRecord,
    PlayerRecord,
    PlayerSnapshotRecord,
    PositionRecord,
    TeamRecord,
)
from .utils import flag, normalise_name, safe_get, to_float, to_int, to_text


def _map(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _items(value: Any, key: str) -> list[dict[str, Any]]:
    candidate = safe_get(_map(value), key, default=[])
    if not isinstance(candidate, list):
        return []
    return [item for item in candidate if isinstance(item, dict)]


def _read(value: Any, key: str, default: Any = None, cast: Callable[[Any], Any] | None = None) -> Any:
    return safe_get(_map(value), key, default=default, cast=cast)


def _integer(value: Any, key: str, default: Any = None) -> int | None:
    return _read(value, key, default=default, cast=to_int)


def _number(value: Any, key: str, default: Any = None) -> float | None:
    return _read(value, key, default=default, cast=to_float)


def _string(value: Any, key: str, default: Any = None) -> str | None:
    return _read(value, key, default=default, cast=to_text)


def _flag(value: Any, key: str, default: Any = None) -> int | None:
    return _read(value, key, default=default, cast=flag)


class BootstrapValidationError(ValueError):
    """The bootstrap payload is too incomplete to replace the active catalog."""


def validate_bootstrap_payload(payload: Any, records: BootstrapRecords | None = None) -> None:
    """Reject catastrophic bootstrap shapes before absence marking or writes.

    The public API is allowed to add fields, but a successful catalog refresh
    must contain a credible, non-empty team and element collection.  Every
    element must have an ID so a truncated/malformed response cannot make all
    previously active players look absent.
    """

    if not isinstance(payload, dict):
        raise BootstrapValidationError("bootstrap-static response must be an object")
    for key in ("elements", "teams", "events"):
        if key not in payload:
            raise BootstrapValidationError(f"bootstrap-static missing top-level key: {key}")
    elements = payload.get("elements")
    teams = payload.get("teams")
    events = payload.get("events")
    if not isinstance(elements, list) or not elements:
        raise BootstrapValidationError("bootstrap-static elements must be a non-empty array")
    if not isinstance(teams, list) or not teams:
        raise BootstrapValidationError("bootstrap-static teams must be a non-empty array")
    if not isinstance(events, list):
        raise BootstrapValidationError("bootstrap-static events must be an array")
    if any(not isinstance(item, dict) or _integer(item, "id") is None or not _string(item, "web_name") for item in elements):
        raise BootstrapValidationError("bootstrap-static contains malformed elements")
    if not any(isinstance(item, dict) and _integer(item, "id") is not None and _string(item, "name") for item in teams):
        raise BootstrapValidationError("bootstrap-static contains no credible teams")
    if records is not None and len(records.players) != len(elements):
        raise BootstrapValidationError("bootstrap-static player parsing was incomplete")


def parse_event(value: dict[str, Any]) -> EventRecord | None:
    event_id = _integer(value, "id")
    if event_id is None:
        return None
    return EventRecord(
        id=event_id,
        name=_string(value, "name"),
        deadline_time=_string(value, "deadline_time"),
        deadline_time_epoch=_integer(value, "deadline_time_epoch"),
        finished=_flag(value, "finished"),
        data_checked=_flag(value, "data_checked"),
        is_previous=_flag(value, "is_previous"),
        is_current=_flag(value, "is_current"),
        is_next=_flag(value, "is_next"),
        average_entry_score=_integer(value, "average_entry_score"),
        highest_score=_integer(value, "highest_score"),
        most_selected=_integer(value, "most_selected"),
        most_transferred_in=_integer(value, "most_transferred_in"),
        most_captained=_integer(value, "most_captained"),
        most_vice_captained=_integer(value, "most_vice_captained"),
        top_element=_integer(value, "top_element"),
        transfers_made=_integer(value, "transfers_made"),
        released=_flag(value, "released"),
        raw_json=dict(value),
    )


def parse_chip(value: dict[str, Any]) -> ChipRecord | None:
    chip_id = _integer(value, "id")
    name = _string(value, "name")
    if chip_id is None or name is None:
        return None
    return ChipRecord(
        id=chip_id,
        name=name,
        number=_integer(value, "number"),
        chip_type=_string(value, "chip_type"),
        start_event=_integer(value, "start_event"),
        stop_event=_integer(value, "stop_event"),
    )


def parse_team(value: dict[str, Any]) -> TeamRecord | None:
    team_id = _integer(value, "id")
    if team_id is None:
        return None
    return TeamRecord(
        id=team_id,
        code=_integer(value, "code"),
        name=_string(value, "name"),
        short_name=_string(value, "short_name"),
        strength=_integer(value, "strength"),
        strength_overall_home=_integer(value, "strength_overall_home"),
        strength_overall_away=_integer(value, "strength_overall_away"),
        strength_attack_home=_integer(value, "strength_attack_home"),
        strength_attack_away=_integer(value, "strength_attack_away"),
        strength_defence_home=_integer(value, "strength_defence_home"),
        strength_defence_away=_integer(value, "strength_defence_away"),
        raw_json=dict(value),
    )


def parse_position(value: dict[str, Any]) -> PositionRecord | None:
    position_id = _integer(value, "id")
    if position_id is None:
        return None
    return PositionRecord(
        id=position_id,
        singular_name=_string(value, "singular_name"),
        singular_name_short=_string(value, "singular_name_short"),
        plural_name=_string(value, "plural_name"),
        squad_select=_integer(value, "squad_select"),
        squad_min_play=_integer(value, "squad_min_play"),
        squad_max_play=_integer(value, "squad_max_play"),
        raw_json=dict(value),
    )


def parse_player(value: dict[str, Any], captured_at: str) -> PlayerRecord | None:
    player_id = _integer(value, "id")
    if player_id is None:
        return None
    first_name = _string(value, "first_name")
    second_name = _string(value, "second_name")
    web_name = _string(value, "web_name")
    name_parts = [part for part in (first_name, second_name) if part]
    full_name = " ".join(name_parts) if name_parts else web_name
    return PlayerRecord(
        id=player_id,
        code=_integer(value, "code"),
        first_name=first_name,
        second_name=second_name,
        web_name=web_name,
        full_name=full_name,
        norm_name=normalise_name(full_name) or None,
        team_id=_integer(value, "team"),
        element_type=_integer(value, "element_type"),
        squad_number=_integer(value, "squad_number"),
        opta_code=_string(value, "opta_code"),
        raw_json=dict(value),
    )


def parse_snapshot(value: dict[str, Any], captured_at: str, event_context: int | None = None) -> PlayerSnapshotRecord | None:
    player_id = _integer(value, "id")
    if player_id is None:
        return None
    integer_fields = (
        "cost_change_event",
        "cost_change_start",
        "transfers_in_event",
        "transfers_out_event",
        "transfers_in",
        "transfers_out",
        "chance_of_playing_this_round",
        "chance_of_playing_next_round",
        "total_points",
        "event_points",
        "minutes",
        "starts",
        "goals_scored",
        "assists",
        "clean_sheets",
        "goals_conceded",
        "saves",
        "bonus",
        "bps",
        "yellow_cards",
        "red_cards",
        "penalties_saved",
        "penalties_missed",
        "defensive_contribution",
        "clearances_blocks_interceptions",
        "recoveries",
        "tackles",
    )
    float_fields = (
        "selected_by_percent",
        "points_per_game",
        "form",
        "influence",
        "creativity",
        "threat",
        "ict_index",
        "expected_goals",
        "expected_assists",
        "expected_goal_involvements",
        "expected_goals_conceded",
        "expected_goals_per_90",
        "expected_assists_per_90",
        "expected_goal_involvements_per_90",
        "expected_goals_conceded_per_90",
        "ep_this",
        "ep_next",
        "value_form",
        "value_season",
    )
    values: dict[str, Any] = {field: _integer(value, field) for field in integer_fields}
    values.update({field: _number(value, field) for field in float_fields})
    values.update(
        {
            "now_cost": _integer(value, "now_cost"),
            "status": _string(value, "status"),
            "news": _string(value, "news"),
            "news_added": _string(value, "news_added"),
        }
    )
    return PlayerSnapshotRecord(
        player_id=player_id,
        captured_at=captured_at,
        event_context=event_context,
        raw_json=dict(value),
        **values,
    )


def parse_bootstrap(payload: dict[str, Any], captured_at: str, event_context: int | None = None) -> BootstrapRecords:
    events = [record for item in _items(payload, "events") if (record := parse_event(item)) is not None]
    chips = [record for item in _items(payload, "chips") if (record := parse_chip(item)) is not None]
    teams = [record for item in _items(payload, "teams") if (record := parse_team(item)) is not None]
    positions = [record for item in _items(payload, "element_types") if (record := parse_position(item)) is not None]
    players = [record for item in _items(payload, "elements") if (record := parse_player(item, captured_at)) is not None]
    snapshots = [
        record
        for item in _items(payload, "elements")
        if (record := parse_snapshot(item, captured_at, event_context)) is not None
    ]
    return BootstrapRecords(events, chips, teams, positions, players, snapshots)


def parse_fixtures(payload: list[Any]) -> list[FixtureRecord]:
    records: list[FixtureRecord] = []
    for value in payload if isinstance(payload, list) else []:
        if not isinstance(value, dict):
            continue
        fixture_id = _integer(value, "id")
        if fixture_id is None:
            continue
        stats = _read(value, "stats", default=[])
        records.append(
            FixtureRecord(
                id=fixture_id,
                code=_integer(value, "code"),
                event=_integer(value, "event"),
                kickoff_time=_string(value, "kickoff_time"),
                team_h=_integer(value, "team_h"),
                team_a=_integer(value, "team_a"),
                team_h_score=_integer(value, "team_h_score"),
                team_a_score=_integer(value, "team_a_score"),
                team_h_difficulty=_integer(value, "team_h_difficulty"),
                team_a_difficulty=_integer(value, "team_a_difficulty"),
                started=_flag(value, "started"),
                finished=_flag(value, "finished"),
                finished_provisional=_flag(value, "finished_provisional"),
                provisional_start_time=_flag(value, "provisional_start_time"),
                minutes=_integer(value, "minutes"),
                stats_json=stats,
                raw_json=dict(value),
            )
        )
    return records


def _gameweek_record(
    player_id: int,
    value: dict[str, Any],
    event: int | None,
    fixture_id: int | None,
    source: str,
    was_home_key: str = "was_home",
    opponent_key: str = "opponent_team",
    stats_value: dict[str, Any] | None = None,
) -> PlayerGameweekRecord | None:
    if event is None:
        return None
    has_separate_stats = stats_value is not None
    stats = stats_value if isinstance(stats_value, dict) else value

    def live_or_row(key: str, cast: Callable[[Any], Any] | None = None) -> Any:
        found = safe_get(stats, key, cast=cast)
        if found is not None:
            return found
        if has_separate_stats:
            return None
        return safe_get(value, key, cast=cast)

    was_home = _flag(value, was_home_key)
    opponent = _integer(value, opponent_key)
    if opponent is None:
        home_team = _integer(value, "team_h")
        away_team = _integer(value, "team_a")
        if was_home == 1:
            opponent = away_team
        elif was_home == 0:
            opponent = home_team
    fields_int = (
        "minutes",
        "starts",
        "total_points",
        "goals_scored",
        "assists",
        "clean_sheets",
        "goals_conceded",
        "saves",
        "bonus",
        "bps",
        "yellow_cards",
        "red_cards",
        "penalties_saved",
        "penalties_missed",
        "own_goals",
        "defensive_contribution",
        "value",
        "selected",
        "transfers_in",
        "transfers_out",
        "transfers_balance",
    )
    fields_float = (
        "influence",
        "creativity",
        "threat",
        "ict_index",
        "expected_goals",
        "expected_assists",
        "expected_goal_involvements",
        "expected_goals_conceded",
    )
    values = {field: live_or_row(field, to_int) for field in fields_int}
    values.update({field: live_or_row(field, to_float) for field in fields_float})
    return PlayerGameweekRecord(
        player_id=player_id,
        event=event,
        fixture_id=fixture_id,
        opponent_team=opponent,
        was_home=was_home,
        kickoff_time=_string(value, "kickoff_time"),
        source=source,
        raw_json=dict(value),
        **values,
    )


def parse_element_summary(payload: dict[str, Any], player_id: int) -> list[PlayerGameweekRecord]:
    records: list[PlayerGameweekRecord] = []
    for value in _items(payload, "fixtures"):
        event = _integer(value, "event")
        fixture_id = _integer(value, "id")
        record = _gameweek_record(
            player_id,
            value,
            event,
            fixture_id,
            "element_summary",
            was_home_key="is_home",
        )
        if record is not None:
            record.raw_json = dict(value)
            records.append(record)
    for value in _items(payload, "history"):
        event = _integer(value, "event")
        if event is None:
            event = _integer(value, "round")
        fixture_id = _integer(value, "fixture")
        record = _gameweek_record(player_id, value, event, fixture_id, "element_summary")
        if record is not None:
            records.append(record)
    return records


_LIVE_INT_FIELDS = {
    "minutes",
    "starts",
    "total_points",
    "goals_scored",
    "assists",
    "clean_sheets",
    "goals_conceded",
    "saves",
    "bonus",
    "bps",
    "yellow_cards",
    "red_cards",
    "penalties_saved",
    "penalties_missed",
    "own_goals",
    "defensive_contribution",
    "value",
    "selected",
    "transfers_in",
    "transfers_out",
    "transfers_balance",
}
_LIVE_FLOAT_FIELDS = {
    "influence",
    "creativity",
    "threat",
    "ict_index",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "expected_goals_conceded",
}


def _parse_live_explanation_stats(value: dict[str, Any]) -> dict[str, Any]:
    """Convert one explain[] fixture's stat entries into row fields.

    The live response's top-level stats are event totals.  For a double
    gameweek those totals must not be copied to every fixture row.  The
    explain[] entries contain the only fixture-specific values we use here.
    """

    entries = _read(value, "stats", default=[])
    if not isinstance(entries, list):
        return {}
    parsed: dict[str, Any] = {}
    points_total = 0
    point_entries = 0
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        identifier = _string(entry, "identifier")
        if identifier is None:
            continue
        raw_value = _read(entry, "value")
        if raw_value is not None:
            if identifier in _LIVE_INT_FIELDS:
                parsed[identifier] = to_int(raw_value)
            elif identifier in _LIVE_FLOAT_FIELDS:
                parsed[identifier] = to_float(raw_value)
        points = _integer(entry, "points")
        if points is not None:
            points_total += points
            point_entries += 1
    if "total_points" not in parsed and point_entries:
        parsed["total_points"] = points_total
    return parsed


def parse_live_event_totals(value: dict[str, Any]) -> dict[str, Any]:
    """The EVENT totals one ``event/live`` element states at its top level.

    ``stats`` is the endpoint's own statement of the player's total for the
    event, which is exactly the number that must not be duplicated once per
    double-gameweek fixture; the ``explain`` legs below it are the per-fixture
    shares of that same total.  Only fields the payload actually states are
    returned, so an absent field stays absent and a reader is left to treat it
    as missing rather than read a zero into it.
    """

    stats = _read(value, "stats", default={})
    stats_map = stats if isinstance(stats, dict) else {}
    totals: dict[str, Any] = {}
    for name in sorted(_LIVE_INT_FIELDS):
        if name in stats_map:
            totals[name] = to_int(stats_map[name])
    for name in sorted(_LIVE_FLOAT_FIELDS):
        if name in stats_map:
            totals[name] = to_float(stats_map[name])
    return totals


def parse_event_live(payload: dict[str, Any], event: int) -> list[PlayerGameweekRecord]:
    records: list[PlayerGameweekRecord] = []
    for value in _items(payload, "elements"):
        player_id = _integer(value, "id")
        if player_id is None:
            player_id = _integer(value, "element")
        if player_id is None:
            continue
        stats = _read(value, "stats", default={})
        stats_map = stats if isinstance(stats, dict) else {}
        explanations = _read(value, "explain", default=[])
        explanation_rows: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
        if isinstance(explanations, list):
            seen_fixtures: set[int] = set()
            for explanation in explanations:
                if not isinstance(explanation, dict):
                    continue
                explanation_fixture = _integer(explanation, "fixture")
                if explanation_fixture is None or explanation_fixture in seen_fixtures:
                    continue
                seen_fixtures.add(explanation_fixture)
                context = dict(value)
                for key in ("fixture", "opponent_team", "was_home", "is_home", "team_h", "team_a", "kickoff_time"):
                    if key in explanation:
                        context[key] = explanation[key]
                if "was_home" not in context and "is_home" in context:
                    context["was_home"] = context["is_home"]
                # The element kept in raw_json is the id this row was actually
                # read for, so downstream readers can tell whose row it is from
                # the payload rather than from what they asked the parser for.
                context["element"] = player_id
                context["id"] = player_id
                explanation_rows.append((explanation_fixture, context, _parse_live_explanation_stats(explanation)))
        if explanation_rows:
            for fixture_id, context, fixture_stats in explanation_rows:
                # An empty explain.stats for a single fixture can safely use
                # the live stats object.  With multiple fixtures it must stay
                # empty, otherwise event totals would be duplicated.
                if not fixture_stats and len(explanation_rows) > 1:
                    stats_value: dict[str, Any] | None = {}
                else:
                    stats_value = fixture_stats or (stats_map if stats_map else None)
                record = _gameweek_record(
                    player_id,
                    context,
                    event,
                    fixture_id,
                    "event_live",
                    stats_value=stats_value,
                )
                if record is not None:
                    record.raw_json = dict(value)
                    records.append(record)
            continue

        fixture_id = safe_get(stats_map, "fixture", cast=to_int)
        if fixture_id is None:
            fixture_id = _integer(value, "fixture")
        if fixture_id is None:
            fixture_id = -1
        record = _gameweek_record(
            player_id,
            value,
            event,
            fixture_id,
            "event_live",
            stats_value=stats_map if stats_map else None,
        )
        if record is not None:
            record.raw_json = {**dict(value), "element": player_id, "id": player_id}
            records.append(record)
    return records


def parse_element_history_past(payload: dict[str, Any], player_id: int) -> list["PlayerSeasonHistoryRecord"]:
    """Parse official prior-season rows exactly as the endpoint provides them."""

    from .models import PlayerSeasonHistoryRecord  # local import avoids module cycles

    records: list[PlayerSeasonHistoryRecord] = []
    for value in _items(payload, "history_past"):
        season_name = _string(value, "season_name")
        if season_name is None:
            continue
        records.append(
            PlayerSeasonHistoryRecord(
                player_id=int(player_id),
                season_name=str(season_name),
                minutes=_integer(value, "minutes"),
                starts=_integer(value, "starts"),
                total_points=_integer(value, "total_points"),
                goals_scored=_integer(value, "goals_scored"),
                assists=_integer(value, "assists"),
                clean_sheets=_integer(value, "clean_sheets"),
                bonus=_integer(value, "bonus"),
                saves=_integer(value, "saves"),
                raw_json=dict(value),
            )
        )
    return records


def parse_entry(payload: dict[str, Any]) -> EntryRecord | None:
    entry_id = _integer(payload, "id")
    if entry_id is None:
        return None
    first = _string(payload, "player_first_name")
    last = _string(payload, "player_last_name")
    player_name = " ".join(part for part in (first, last) if part) or None
    return EntryRecord(
        entry_id=entry_id,
        player_name=player_name,
        team_name=_string(payload, "name"),
        summary_overall_points=_integer(payload, "summary_overall_points"),
        summary_overall_rank=_integer(payload, "summary_overall_rank"),
        summary_event_points=_integer(payload, "summary_event_points"),
        summary_event_rank=_integer(payload, "summary_event_rank"),
        current_event=_integer(payload, "current_event"),
        bank=_integer(payload, "last_deadline_bank"),
        team_value=_integer(payload, "last_deadline_value"),
        total_transfers=_integer(payload, "last_deadline_total_transfers"),
        raw_json=dict(payload),
    )


def parse_history_row(value: dict[str, Any]) -> HistoryRow:
    overall_rank = _integer(value, "overall_rank")
    if overall_rank is None:
        overall_rank = _integer(value, "rank")
    return HistoryRow(
        event=_integer(value, "event"),
        bank=_integer(value, "bank"),
        value=_integer(value, "value"),
        total_transfers=_integer(value, "total_transfers"),
        event_transfers=_integer(value, "event_transfers"),
        event_transfers_cost=_integer(value, "event_transfers_cost"),
        points_on_bench=_integer(value, "points_on_bench"),
        overall_rank=overall_rank,
        raw_json=dict(value),
    )


def parse_entry_history(payload: dict[str, Any]) -> ManagerHistory:
    current = [parse_history_row(value) for value in _items(payload, "current")]
    past = [parse_history_row(value) for value in _items(payload, "past")]
    chips = []
    for value in _items(payload, "chips"):
        name = _string(value, "name")
        event = _integer(value, "event")
        if name is not None and event is not None:
            chips.append(ManagerChipRecord(name=name, event=event, time=_string(value, "time")))
    return ManagerHistory(current=current, past=past, chips=chips)


def parse_entry_transfers(payload: list[Any]) -> list[ManagerTransferRecord]:
    """Parse exact public transfer-history rows without reconstructing costs."""

    if not isinstance(payload, list):
        raise ValueError("Transfer history response must be a list")
    records: list[ManagerTransferRecord] = []
    for value in payload:
        if not isinstance(value, dict):
            raise ValueError("Transfer history rows must be objects")
        required = {
            "entry_id": _integer(value, "entry"),
            "element_in": _integer(value, "element_in"),
            "element_out": _integer(value, "element_out"),
            "event": _integer(value, "event"),
            "element_in_cost": _integer(value, "element_in_cost"),
            "element_out_cost": _integer(value, "element_out_cost"),
        }
        if any(item is None for item in required.values()):
            raise ValueError("Transfer history row is missing an exact required field")
        records.append(
            ManagerTransferRecord(
                entry_id=int(required["entry_id"]),
                element_in=int(required["element_in"]),
                element_out=int(required["element_out"]),
                event=int(required["event"]),
                time=_string(value, "time"),
                element_in_cost=int(required["element_in_cost"]),
                element_out_cost=int(required["element_out_cost"]),
                raw_json=dict(value),
            )
        )
    return records


def parse_entry_picks(payload: dict[str, Any]) -> PicksRecord:
    picks: list[PickRecord] = []
    for value in _items(payload, "picks"):
        player_id = _integer(value, "element")
        position = _integer(value, "position")
        if player_id is None or position is None:
            continue
        picks.append(
            PickRecord(
                player_id=player_id,
                position=position,
                multiplier=_integer(value, "multiplier"),
                is_captain=_flag(value, "is_captain"),
                is_vice_captain=_flag(value, "is_vice_captain"),
                raw_json=dict(value),
            )
        )
    entry_history = _read(payload, "entry_history", default={})
    entry_history_row = parse_history_row(entry_history) if isinstance(entry_history, dict) else None
    automatic_subs = _read(payload, "automatic_subs", default=[])
    return PicksRecord(
        picks=picks,
        entry_history=entry_history_row,
        active_chip=_string(payload, "active_chip"),
        automatic_subs=[dict(value) for value in automatic_subs if isinstance(value, dict)] if isinstance(automatic_subs, list) else [],
        raw_json=dict(payload),
    )


# Descriptive alias used by callers that mirror the endpoint name.
parse_bootstrap_static = parse_bootstrap
