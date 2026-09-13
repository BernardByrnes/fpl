"""Read-only, deterministic Post-GW Market Report V1."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import tempfile
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

from . import repositories as repo
from .utils import json_text


DEFAULT_MARKET_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "post_gw_market_report.default.json"
MARKET_CONFIG_SCHEMA_VERSION = "post_gw_market_report.config.v1"
MARKET_REPORT_SCHEMA_VERSION = "post_gw_market_report.v1"


class MarketReportError(ValueError):
    """Base error for Post-GW Market Report V1."""


class MarketReportConfigError(MarketReportError):
    """The Market Report configuration is invalid."""


class MarketReportDataError(MarketReportError):
    """The local database cannot safely supply a requested report fact."""


class IncompleteGameweekError(MarketReportError):
    """Normal output was requested for an incomplete or unverifiable gameweek."""


class MarketReportOutputError(MarketReportError):
    """Report artifacts could not be written atomically."""


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise MarketReportConfigError(f"Could not read {label}: {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise MarketReportConfigError(f"Invalid JSON in {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise MarketReportConfigError(f"{label} must be a JSON object: {path}")
    return value


def _deep_merge(defaults: dict[str, Any], supplied: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(defaults)
    for key, value in supplied.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _reject_unknown_keys(value: Any, template: Any, path: str = "") -> None:
    if not isinstance(value, dict) or not isinstance(template, dict):
        return
    for key, supplied in value.items():
        location = f"{path}.{key}" if path else key
        if key not in template:
            raise MarketReportConfigError(f"Unknown Market Report config key: {location}")
        _reject_unknown_keys(supplied, template[key], location)


def _is_int(value: Any, minimum: int | None = None) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and (minimum is None or value >= minimum)


def _is_number(value: Any, minimum: float | None = None) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value)) and (minimum is None or float(value) >= minimum)


def _require_bool(value: Any, name: str) -> None:
    if not isinstance(value, bool):
        raise MarketReportConfigError(f"{name} must be boolean")


def _require_int(value: Any, name: str, minimum: int = 0) -> None:
    if not _is_int(value, minimum):
        raise MarketReportConfigError(f"{name} must be an integer >= {minimum}")


def _require_number(value: Any, name: str, minimum: float = 0.0) -> None:
    if not _is_number(value, minimum):
        raise MarketReportConfigError(f"{name} must be a finite number >= {minimum}")


def validate_market_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate and return a fully normalized Market Report configuration."""

    if not isinstance(config, dict):
        raise MarketReportConfigError("Market Report configuration root must be an object")
    if config.get("schema_version") != MARKET_CONFIG_SCHEMA_VERSION:
        raise MarketReportConfigError(f"schema_version must be {MARKET_CONFIG_SCHEMA_VERSION}")

    output = config.get("output")
    completion = config.get("gameweek_completion")
    numeric = config.get("numeric_format")
    metrics = config.get("metrics")
    sections = config.get("sections")
    for name, value in (
        ("output", output),
        ("gameweek_completion", completion),
        ("numeric_format", numeric),
        ("metrics", metrics),
        ("sections", sections),
    ):
        if not isinstance(value, dict):
            raise MarketReportConfigError(f"{name} must be an object")

    if not isinstance(output.get("base_dir"), str) or not output["base_dir"].strip():
        raise MarketReportConfigError("output.base_dir must be a non-empty string")
    for key in ("write_json", "write_markdown", "deterministic", "include_wall_clock_generated_at"):
        _require_bool(output.get(key), f"output.{key}")
    if not output["write_json"] and not output["write_markdown"]:
        raise MarketReportConfigError("At least one of output.write_json or output.write_markdown must be true")
    if not output["deterministic"] or output["include_wall_clock_generated_at"]:
        raise MarketReportConfigError("V1 requires deterministic output without a wall-clock generated timestamp")

    for key in (
        "require_complete",
        "require_event_finished_if_available",
        "require_event_data_checked_if_available",
        "require_fixtures_finished",
    ):
        _require_bool(completion.get(key), f"gameweek_completion.{key}")
        if not completion[key]:
            raise MarketReportConfigError(f"V1 safety requires gameweek_completion.{key}=true")

    for key in ("price_decimals", "ownership_decimals", "x_metric_decimals"):
        _require_int(numeric.get(key), f"numeric_format.{key}", 0)
    if metrics.get("null_xg_policy") != "unknown_not_zero":
        raise MarketReportConfigError("metrics.null_xg_policy must be unknown_not_zero")
    if metrics.get("xgi_preference") != "official_then_xg_plus_xa":
        raise MarketReportConfigError("metrics.xgi_preference must be official_then_xg_plus_xa")

    gw_stars = sections.get("gw_stars")
    underlying = sections.get("underlying_leaders")
    under_return = sections.get("attacking_process_under_return")
    ahead = sections.get("attacking_output_ahead_of_process")
    defcon = sections.get("defensive_contributions")
    minutes = sections.get("minutes_watch")
    squad = sections.get("our_squad")
    candidates = sections.get("scout_candidates")
    for name, value in (
        ("sections.gw_stars", gw_stars),
        ("sections.underlying_leaders", underlying),
        ("sections.attacking_process_under_return", under_return),
        ("sections.attacking_output_ahead_of_process", ahead),
        ("sections.defensive_contributions", defcon),
        ("sections.minutes_watch", minutes),
        ("sections.our_squad", squad),
        ("sections.scout_candidates", candidates),
    ):
        if not isinstance(value, dict):
            raise MarketReportConfigError(f"{name} must be an object")

    for name, value, keys in (
        ("sections.gw_stars", gw_stars, ("enabled",)),
        ("sections.underlying_leaders", underlying, ("enabled",)),
        ("sections.attacking_process_under_return", under_return, ("enabled",)),
        ("sections.attacking_output_ahead_of_process", ahead, ("enabled",)),
        ("sections.defensive_contributions", defcon, ("enabled", "require_per_fixture_provenance")),
        ("sections.minutes_watch", minutes, ("enabled",)),
        ("sections.our_squad", squad, ("enabled", "include_market_section_flags")),
        ("sections.scout_candidates", candidates, ("enabled",)),
    ):
        for key in keys:
            _require_bool(value.get(key), f"{name}.{key}")
    if not defcon["require_per_fixture_provenance"]:
        raise MarketReportConfigError(
            "V1 requires sections.defensive_contributions.require_per_fixture_provenance=true"
        )

    _require_int(gw_stars.get("rows"), "sections.gw_stars.rows", 1)
    _require_int(gw_stars.get("minimum_minutes"), "sections.gw_stars.minimum_minutes", 1)

    _require_int(underlying.get("rows_per_metric"), "sections.underlying_leaders.rows_per_metric", 1)
    _require_int(underlying.get("minimum_minutes"), "sections.underlying_leaders.minimum_minutes", 1)
    if not isinstance(underlying.get("metrics"), list) or not underlying["metrics"]:
        raise MarketReportConfigError("sections.underlying_leaders.metrics must be a non-empty list")
    if any(metric not in {"xg", "xa", "xgi"} for metric in underlying["metrics"]):
        raise MarketReportConfigError("sections.underlying_leaders.metrics supports only xg, xa, xgi")

    for name, section in (
        ("sections.attacking_process_under_return", under_return),
        ("sections.attacking_output_ahead_of_process", ahead),
    ):
        _require_int(section.get("rows"), f"{name}.rows", 1)
        _require_int(section.get("minimum_minutes"), f"{name}.minimum_minutes", 1)
        if not isinstance(section.get("positions"), list) or not section["positions"]:
            raise MarketReportConfigError(f"{name}.positions must be a non-empty list")
        if any(position not in {"DEF", "MID", "FWD"} for position in section["positions"]):
            raise MarketReportConfigError(f"{name}.positions must contain only DEF, MID, FWD")
    _require_number(under_return.get("minimum_xgi"), "sections.attacking_process_under_return.minimum_xgi", 0)
    _require_int(under_return.get("maximum_actual_gi"), "sections.attacking_process_under_return.maximum_actual_gi", 0)
    _require_int(ahead.get("minimum_actual_gi"), "sections.attacking_output_ahead_of_process.minimum_actual_gi", 0)
    _require_number(ahead.get("minimum_attacking_residual"), "sections.attacking_output_ahead_of_process.minimum_attacking_residual", 0)

    _require_int(defcon.get("rows_per_position_group"), "sections.defensive_contributions.rows_per_position_group", 1)
    _require_int(defcon.get("minimum_minutes_per_fixture"), "sections.defensive_contributions.minimum_minutes_per_fixture", 1)
    _require_int(defcon.get("near_threshold_margin"), "sections.defensive_contributions.near_threshold_margin", 0)
    rules = defcon.get("position_rules")
    if not isinstance(rules, dict) or set(rules) != {"DEF", "MID", "FWD"}:
        raise MarketReportConfigError("sections.defensive_contributions.position_rules must define DEF, MID, FWD")
    for position, expected_family in (("DEF", "CBIT"), ("MID", "CBIRT"), ("FWD", "CBIRT")):
        rule = rules[position]
        if not isinstance(rule, dict):
            raise MarketReportConfigError(f"sections.defensive_contributions.position_rules.{position} must be an object")
        _require_int(rule.get("threshold"), f"sections.defensive_contributions.position_rules.{position}.threshold", 1)
        if rule.get("action_family") != expected_family:
            raise MarketReportConfigError(f"{position} action_family must be {expected_family}")

    for key in ("rows_per_subsection", "lookback_completed_fixtures", "minimum_baseline_fixtures"):
        _require_int(minutes.get(key), f"sections.minutes_watch.{key}", 1)
    for key in (
        "baseline_average_minutes_per_fixture",
        "baseline_starts",
        "low_minutes_threshold",
        "limited_minutes_maximum",
        "limited_minutes_minimum_points",
    ):
        _require_int(minutes.get(key), f"sections.minutes_watch.{key}", 0)
    _require_number(minutes.get("limited_minutes_minimum_xgi"), "sections.minutes_watch.limited_minutes_minimum_xgi", 0)
    if minutes["limited_minutes_maximum"] < minutes["low_minutes_threshold"]:
        raise MarketReportConfigError("sections.minutes_watch.limited_minutes_maximum must be >= low_minutes_threshold")
    if minutes["minimum_baseline_fixtures"] > minutes["lookback_completed_fixtures"]:
        raise MarketReportConfigError("minimum_baseline_fixtures cannot exceed lookback_completed_fixtures")

    if squad.get("squad_resolution") != "target_gw_only":
        raise MarketReportConfigError("sections.our_squad.squad_resolution must be target_gw_only")
    _require_int(candidates.get("maximum_market_triggered_candidates"), "sections.scout_candidates.maximum_market_triggered_candidates", 1)
    _require_int(candidates.get("maximum_questions_per_player"), "sections.scout_candidates.maximum_questions_per_player", 1)
    return config


def load_market_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the tracked default config plus an optional strict override."""

    defaults = _read_json_object(DEFAULT_MARKET_CONFIG_PATH, "default Market Report configuration")
    if path is None:
        return validate_market_config(defaults)
    source = _read_json_object(Path(path), "Market Report configuration")
    _reject_unknown_keys(source, defaults)
    return validate_market_config(_deep_merge(defaults, source))


def market_config_hash(config: dict[str, Any]) -> str:
    """Return a SHA-256 hash of the normalized configuration."""

    normalized = validate_market_config(deepcopy(config))
    digest = hashlib.sha256(json_text(normalized).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def open_read_only_database(path: str | Path) -> sqlite3.Connection:
    """Open the local database in read-only/query-only mode.

    This connection never runs migrations or collection and cannot perform a
    logical database write. Run the separate collector to completion before
    building a report so the source data represents a coherent local state.
    """

    source = Path(path)
    if not source.exists():
        raise MarketReportDataError(f"Database does not exist: {source}")
    try:
        conn = sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True)
    except sqlite3.Error as exc:
        raise MarketReportDataError(f"Could not open database read-only: {source}: {exc}") from exc
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA query_only=ON")
    return conn


def _rows(cursor: sqlite3.Cursor) -> list[dict[str, Any]]:
    return [dict(row) for row in cursor.fetchall()]


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _whole_or_float(value: float | None) -> int | float | None:
    if value is None:
        return None
    return int(value) if value.is_integer() else value


def _rounded(value: float | int | None, decimals: int) -> float | int | None:
    if value is None:
        return None
    rounded = round(float(value), decimals)
    return int(rounded) if rounded.is_integer() else rounded


def _sum_field(rows: Iterable[dict[str, Any]], field: str, flags: set[str]) -> int | float | None:
    values: list[float] = []
    missing = False
    for row in rows:
        raw_value = row.get(field)
        value = _number(raw_value)
        if value is None:
            missing = True
            if raw_value is not None:
                flags.add(f"malformed_{field}")
        else:
            values.append(value)
    if missing:
        flags.add(f"missing_{field}")
        return None
    return _whole_or_float(sum(values))


def _fixture_xgi(row: dict[str, Any], decimals: int) -> tuple[float | int | None, str | None]:
    official = _number(row.get("expected_goal_involvements"))
    if official is not None:
        return _rounded(official, decimals), "FACT:official_expected_goal_involvements"
    xg = _number(row.get("expected_goals"))
    xa = _number(row.get("expected_assists"))
    if xg is None or xa is None:
        return None, None
    return _rounded(xg + xa, decimals), "DERIVED:xg_plus_xa"


def _player_rows(conn: sqlite3.Connection) -> dict[int, dict[str, Any]]:
    rows = _rows(
        conn.execute(
            """SELECT p.id, p.web_name, p.full_name, p.team_id, p.element_type, p.is_active,
                      t.name AS team_name, t.short_name AS team_short_name,
                      pos.singular_name_short AS position_short_name
               FROM players p
               LEFT JOIN teams t ON t.id=p.team_id
               LEFT JOIN positions pos ON pos.id=p.element_type
               ORDER BY p.id"""
        )
    )
    return {int(row["id"]): row for row in rows}


def _identity(player: dict[str, Any]) -> dict[str, Any]:
    name = player.get("full_name") or player.get("web_name") or f"Player {player['id']}"
    return {
        "id": int(player["id"]),
        "name": str(name),
        "team": player.get("team_short_name") or player.get("team_name") or "UNK",
        "position": player.get("position_short_name") or "UNK",
    }


def _event_row(conn: sqlite3.Connection, gw: int) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM events WHERE id=?", (int(gw),)).fetchone()
    return dict(row) if row else None


def assess_gameweek_completion(conn: sqlite3.Connection, gw: int, config: dict[str, Any]) -> dict[str, Any]:
    """Establish target-GW completion from stored event and fixture facts."""

    event = _event_row(conn, gw)
    fixtures = repo.fixture_rows(conn, int(gw), 1)
    completion = config["gameweek_completion"]
    blockers: list[str] = []
    basis: list[str] = []
    if event is None:
        blockers.append(f"GW{gw} is not present in events")
    else:
        event_finished = event.get("finished")
        if event_finished is None:
            basis.append("events.finished unavailable")
        elif event_finished == 1:
            basis.append("events.finished=true")
        else:
            blockers.append("events.finished is not true")
        data_checked = event.get("data_checked")
        if data_checked is None:
            basis.append("events.data_checked unavailable")
        elif data_checked == 1:
            basis.append("events.data_checked=true")
        else:
            blockers.append("events.data_checked is not true")
    if not fixtures:
        blockers.append(f"GW{gw} has no stored fixtures and cannot be proven complete")
    finished_count = sum(1 for fixture in fixtures if fixture.get("finished") == 1)
    provisional_count = sum(1 for fixture in fixtures if fixture.get("finished_provisional") == 1)
    started_unfinished = sum(
        1 for fixture in fixtures if fixture.get("started") == 1 and fixture.get("finished") != 1
    )
    if fixtures:
        basis.append(f"{finished_count}/{len(fixtures)} fixtures finished")
        basis.append(f"{provisional_count} fixtures with finished_provisional=true (metadata only)")
    if completion["require_fixtures_finished"] and finished_count != len(fixtures):
        blockers.append(f"{len(fixtures) - finished_count} target-GW fixtures are unfinished")
    if started_unfinished:
        blockers.append(f"{started_unfinished} target-GW fixtures are started but unfinished")
    return {
        "status": "complete" if not blockers else "incomplete_or_unverifiable",
        "completion_basis": basis,
        "blockers": blockers,
        "fixtures": fixtures,
        "event": event,
        "fixtures_total": len(fixtures),
        "fixtures_finished": finished_count,
        "fixtures_provisional": provisional_count,
    }


def _team_fixture_counts(fixtures: Iterable[dict[str, Any]]) -> dict[int, int]:
    counts: dict[int, int] = defaultdict(int)
    for fixture in fixtures:
        for key in ("team_h", "team_a"):
            value = fixture.get(key)
            if value is not None:
                counts[int(value)] += 1
    return dict(counts)


_CREDIBLE_PERFORMANCE_FIELDS = (
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
    "influence",
    "creativity",
    "threat",
    "ict_index",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "expected_goals_conceded",
    "defensive_contribution",
)

_NON_NEGATIVE_PLAYER_FIELDS = (
    "minutes",
    "starts",
    "goals_scored",
    "assists",
    "clean_sheets",
    "goals_conceded",
    "saves",
    "bonus",
    "expected_goals",
    "expected_assists",
    "expected_goal_involvements",
    "expected_goals_conceded",
    "defensive_contribution",
)


def _is_credible_performance(row: dict[str, Any]) -> bool:
    """Identify a non-placeholder row without treating zero as performance."""

    for field in _CREDIBLE_PERFORMANCE_FIELDS:
        value = _number(row.get(field))
        if value is not None and value != 0:
            return True
    return False


def _validate_target_performance_provenance(
    conn: sqlite3.Connection,
    gw: int,
    gaps: list[str],
) -> None:
    """Reject credible performance tied to an unproven fixture.

    Calculations still use only repo.completed_player_fixture_rows. This
    guard merely reports stored contradictions rather than letting a preview
    present them as ordinary schedule placeholders.
    """

    rows = _rows(
        conn.execute(
            """SELECT p.*, f.id AS joined_fixture_id, f.event AS fixture_event,
                      f.finished AS fixture_finished
                 FROM player_gameweeks p
                 LEFT JOIN fixtures f ON f.id=p.fixture_id
                WHERE p.event=?
                ORDER BY p.player_id,p.fixture_id""",
            (gw,),
        )
    )
    excluded_placeholders = 0
    for row in rows:
        fixture_id = row.get("fixture_id")
        if fixture_id in (None, -1):
            excluded_placeholders += 1
            continue
        if row.get("joined_fixture_id") is None:
            if _is_credible_performance(row):
                raise MarketReportDataError(
                    f"Player {row['player_id']} GW{gw} has credible performance for unknown fixture {fixture_id}"
                )
            excluded_placeholders += 1
            continue
        fixture_event = row.get("fixture_event")
        fixture_finished = row.get("fixture_finished")
        if fixture_event != gw or fixture_finished != 1:
            if _is_credible_performance(row):
                raise MarketReportDataError(
                    f"Player {row['player_id']} GW{gw} has credible performance for unproven fixture {fixture_id}"
                )
            excluded_placeholders += 1
    if excluded_placeholders:
        _add_gap(
            gaps,
            f"{excluded_placeholders} target-GW schedule/sentinel/unproven player rows were excluded from performance calculations.",
        )


def _validate_non_negative_player_fields(row: dict[str, Any]) -> None:
    for field in _NON_NEGATIVE_PLAYER_FIELDS:
        value = _number(row.get(field))
        if value is not None and value < 0:
            raise MarketReportDataError(
                f"Negative {field} for player {row['player_id']} fixture {row['fixture_id']}"
            )


def _normalise_market_rows(
    conn: sqlite3.Connection,
    gw: int,
    fixtures: list[dict[str, Any]],
    config: dict[str, Any],
    gaps: list[str],
) -> list[dict[str, Any]]:
    """Aggregate completed target-GW player fixture facts without approximations."""

    decimals = int(config["numeric_format"]["x_metric_decimals"])
    price_decimals = int(config["numeric_format"]["price_decimals"])
    ownership_decimals = int(config["numeric_format"]["ownership_decimals"])
    players = _player_rows(conn)
    completed_rows = repo.completed_player_fixture_rows(conn, event=gw)
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in completed_rows:
        if row.get("fixture_event") != gw:
            raise MarketReportDataError(
                f"Player {row['player_id']} GW{gw} row references fixture {row['fixture_id']} in event {row.get('fixture_event')}"
            )
        if row.get("fixture_finished") != 1:
            raise MarketReportDataError(
                f"Player {row['player_id']} fixture {row['fixture_id']} was not proven finished"
            )
        _validate_non_negative_player_fields(row)
        grouped[int(row["player_id"])].append(row)
    if not grouped:
        _add_gap(gaps, f"No completed player-fixture performance rows are stored for GW{gw}.")

    team_fixture_counts = _team_fixture_counts(fixtures)
    normalized: list[dict[str, Any]] = []
    for player_id in sorted(grouped):
        player = players.get(player_id)
        if player is None:
            raise MarketReportDataError(f"Completed performance row references unknown player {player_id}")
        rows = grouped[player_id]
        flags: set[str] = set()
        minutes = _sum_field(rows, "minutes", flags)
        points = _sum_field(rows, "total_points", flags)
        goals = _sum_field(rows, "goals_scored", flags)
        assists = _sum_field(rows, "assists", flags)
        clean_sheets = _sum_field(rows, "clean_sheets", flags)
        bonus = _sum_field(rows, "bonus", flags)
        bps = _sum_field(rows, "bps", flags)
        xg = _sum_field(rows, "expected_goals", flags)
        xa = _sum_field(rows, "expected_assists", flags)
        official_xgi = _sum_field(rows, "expected_goal_involvements", flags)
        xg = _rounded(xg, decimals)
        xa = _rounded(xa, decimals)
        official_xgi = _rounded(official_xgi, decimals)
        xgi = official_xgi
        xgi_source = "FACT:official_expected_goal_involvements" if xgi is not None else None
        if xgi is None and xg is not None and xa is not None:
            xgi = _rounded(float(xg) + float(xa), decimals)
            xgi_source = "DERIVED:xg_plus_xa"
        if xgi is None:
            flags.add("xgi_unknown")
        actual_gi = None if goals is None or assists is None else _whole_or_float(float(goals) + float(assists))
        residual = None if actual_gi is None or xgi is None else _rounded(float(actual_gi) - float(xgi), decimals)
        goal_residual = None if goals is None or xg is None else _rounded(float(goals) - float(xg), decimals)
        assist_residual = None if assists is None or xa is None else _rounded(float(assists) - float(xa), decimals)

        snapshot = repo.latest_snapshot(conn, player_id)
        price: float | int | None = None
        ownership: float | int | None = None
        snapshot_as_of = None
        if snapshot is None:
            flags.add("snapshot_missing")
        else:
            snapshot_as_of = snapshot.get("captured_at")
            cost = _number(snapshot.get("now_cost"))
            if cost is None or cost < 0:
                flags.add("price_unknown")
            else:
                price = _rounded(cost / 10.0, price_decimals)
            selected = _number(snapshot.get("selected_by_percent"))
            if selected is None or selected < 0 or selected > 100:
                flags.add("ownership_unknown")
            else:
                ownership = _rounded(selected, ownership_decimals)

        team_id = player.get("team_id")
        team_fixtures = team_fixture_counts.get(int(team_id), 0) if team_id is not None else 0
        fixture_ids = {int(row["fixture_id"]) for row in rows}
        if team_fixtures and len(fixture_ids) > team_fixtures:
            raise MarketReportDataError(
                f"Player {player_id} has {len(fixture_ids)} completed fixture rows but team has only {team_fixtures} GW{gw} fixtures"
            )
        per_fixture = []
        for row in rows:
            fixture_xgi, fixture_xgi_source = _fixture_xgi(row, decimals)
            per_fixture.append(
                {
                    "fixture_id": int(row["fixture_id"]),
                    "event": int(row["event"]),
                    "minutes": _whole_or_float(_number(row.get("minutes"))),
                    "starts": _whole_or_float(_number(row.get("starts"))),
                    "points": _whole_or_float(_number(row.get("total_points"))),
                    "goals": _whole_or_float(_number(row.get("goals_scored"))),
                    "assists": _whole_or_float(_number(row.get("assists"))),
                    "xg": _rounded(_number(row.get("expected_goals")), decimals),
                    "xa": _rounded(_number(row.get("expected_assists")), decimals),
                    "xgi": fixture_xgi,
                    "xgi_source": fixture_xgi_source,
                    "defensive_contribution": _whole_or_float(_number(row.get("defensive_contribution"))),
                    "source": row.get("source"),
                }
            )
        normalized.append(
            {
                "player": _identity(player),
                "player_id": player_id,
                "team_id": int(team_id) if team_id is not None else None,
                "facts": {
                    "price": price,
                    "ownership_pct": ownership,
                    "snapshot_as_of": snapshot_as_of,
                    "minutes": minutes,
                    "points": points,
                    "goals": goals,
                    "assists": assists,
                    "clean_sheets": clean_sheets,
                    "bonus": bonus,
                    "bps": bps,
                    "xg": xg,
                    "xa": xa,
                    "xgi_official": official_xgi,
                },
                "derived": {
                    "xgi": xgi,
                    "xgi_source": xgi_source,
                    "actual_gi": actual_gi,
                    "attacking_residual": residual,
                    "goal_residual": goal_residual,
                    "assist_residual": assist_residual,
                    "team_fixtures_in_gw": team_fixtures,
                    "fixtures_played": len(fixture_ids),
                    "is_blank_team": team_fixtures == 0,
                    "is_dgw_team": team_fixtures > 1,
                    "is_zero_minute_player": minutes == 0,
                },
                "data_flags": sorted(flags),
                "fixture_rows": per_fixture,
            }
        )
    return normalized


def _reason(source: str, metric: str, operator: str, threshold: Any, value: Any, message: str) -> dict[str, Any]:
    return {
        "source": source,
        "metric": metric,
        "operator": operator,
        "threshold": threshold,
        "value": value,
        "message": message,
    }


def _public_market_row(row: dict[str, Any], reasons: list[dict[str, Any]], rank: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "player": dict(row["player"]),
        "facts": dict(row["facts"]),
        "derived": dict(row["derived"]),
        "trigger_reasons": reasons,
        "data_flags": list(row["data_flags"]),
    }
    if rank is not None:
        result["rank"] = rank
    return result


def _descending(value: Any) -> float:
    number = _number(value)
    return -number if number is not None else math.inf


def _add_gap(gaps: list[str], message: str) -> None:
    if message not in gaps:
        gaps.append(message)


def _gw_stars(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    settings = config["sections"]["gw_stars"]
    if not settings["enabled"]:
        return {"title": "GW Stars", "status": "disabled", "rows": []}
    eligible = [
        row
        for row in rows
        if _number(row["facts"]["minutes"]) is not None
        and row["facts"]["minutes"] >= settings["minimum_minutes"]
        and _number(row["facts"]["points"]) is not None
    ]
    eligible.sort(
        key=lambda row: (
            _descending(row["facts"]["points"]),
            _descending(row["facts"]["bonus"]),
            _descending(row["facts"]["bps"]),
            _descending(row["facts"]["minutes"]),
            row["player_id"],
        )
    )
    result = []
    for rank, row in enumerate(eligible[: settings["rows"]], start=1):
        result.append(
            _public_market_row(
                row,
                [
                    _reason(
                        "FACT",
                        "points",
                        "rank",
                        f"top_{settings['rows']}",
                        row["facts"]["points"],
                        f"Selected because player ranked in top {settings['rows']} by GW points using deterministic tie-breakers.",
                    )
                ],
                rank,
            )
        )
    return {
        "title": "GW Stars",
        "basis": "FACT official FPL GW totals",
        "thresholds": {"rows": settings["rows"], "minimum_minutes": settings["minimum_minutes"]},
        "rows": result,
    }


def _underlying_leaders(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    settings = config["sections"]["underlying_leaders"]
    result: dict[str, Any] = {
        "title": "Underlying Leaders",
        "basis": "FACT official target-GW totals; no per-90 metrics",
    }
    if not settings["enabled"]:
        result["status"] = "disabled"
        result["xg_leaders"] = {"rows": []}
        result["xa_leaders"] = {"rows": []}
        result["xgi_leaders"] = {"rows": []}
        return result
    source_fields = {
        "xg": ("xg", "xG"),
        "xa": ("xa", "xA"),
        "xgi": ("xgi", "xGI"),
    }
    for metric in ("xg", "xa", "xgi"):
        key = f"{metric}_leaders"
        if metric not in settings["metrics"]:
            result[key] = {"rows": []}
            continue
        if metric == "xgi":
            values = [(row, row["derived"]["xgi"]) for row in rows]
        else:
            values = [(row, row["facts"][metric]) for row in rows]
        eligible = [
            (row, value)
            for row, value in values
            if _number(row["facts"]["minutes"]) is not None
            and row["facts"]["minutes"] >= settings["minimum_minutes"]
            and _number(value) is not None
        ]
        eligible.sort(
            key=lambda item: (
                _descending(item[1]),
                _descending(item[0]["facts"]["minutes"]),
                _descending(item[0]["facts"]["points"]),
                item[0]["player_id"],
            )
        )
        leader_rows = []
        for rank, (row, value) in enumerate(eligible[: settings["rows_per_metric"]], start=1):
            display = source_fields[metric][1]
            leader_rows.append(
                _public_market_row(
                    row,
                    [
                        _reason(
                            "FACT" if metric != "xgi" or row["derived"]["xgi_source"] == "FACT:official_expected_goal_involvements" else "DERIVED",
                            metric,
                            "rank",
                            f"top_{settings['rows_per_metric']}",
                            value,
                            f"Selected because player ranked in top {settings['rows_per_metric']} by GW {display} using deterministic tie-breakers.",
                        )
                    ],
                    rank,
                )
            )
        result[key] = {
            "metric": metric,
            "minimum_minutes": settings["minimum_minutes"],
            "rows": leader_rows,
        }
    return result


def _attacking_under_return(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    settings = config["sections"]["attacking_process_under_return"]
    if not settings["enabled"]:
        return {"title": "Attacking Process Under-Return Signals", "status": "disabled", "rows": []}
    eligible = [
        row
        for row in rows
        if row["player"]["position"] in settings["positions"]
        and _number(row["facts"]["minutes"]) is not None
        and row["facts"]["minutes"] >= settings["minimum_minutes"]
        and _number(row["derived"]["xgi"]) is not None
        and row["derived"]["xgi"] >= settings["minimum_xgi"]
        and row["derived"]["actual_gi"] == settings["maximum_actual_gi"]
    ]
    eligible.sort(
        key=lambda row: (
            _number(row["derived"]["attacking_residual"]) if _number(row["derived"]["attacking_residual"]) is not None else math.inf,
            _descending(row["derived"]["xgi"]),
            _descending(row["facts"]["minutes"]),
            row["player_id"],
        )
    )
    result = []
    for row in eligible[: settings["rows"]]:
        result.append(
            _public_market_row(
                row,
                [
                    _reason(
                        "DERIVED",
                        "attacking_residual",
                        "<=",
                        -settings["minimum_xgi"],
                        row["derived"]["attacking_residual"],
                        f"actual_gi {row['derived']['actual_gi']} with xGI {row['derived']['xgi']}; attacking_residual {row['derived']['attacking_residual']} meets configured under-return criteria.",
                    )
                ],
            )
        )
    return {
        "title": "Attacking Process Under-Return Signals",
        "basis": "DERIVED actual_gi minus xGI; FPL points are context only",
        "thresholds": {
            "minimum_minutes": settings["minimum_minutes"],
            "minimum_xgi": settings["minimum_xgi"],
            "maximum_actual_gi": settings["maximum_actual_gi"],
        },
        "rows": result,
    }


def _attacking_ahead(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    settings = config["sections"]["attacking_output_ahead_of_process"]
    if not settings["enabled"]:
        return {"title": "Attacking Output Ahead of Process Signals", "status": "disabled", "rows": []}
    eligible = [
        row
        for row in rows
        if row["player"]["position"] in settings["positions"]
        and _number(row["facts"]["minutes"]) is not None
        and row["facts"]["minutes"] >= settings["minimum_minutes"]
        and _number(row["derived"]["xgi"]) is not None
        and _number(row["derived"]["actual_gi"]) is not None
        and row["derived"]["actual_gi"] >= settings["minimum_actual_gi"]
        and _number(row["derived"]["attacking_residual"]) is not None
        and row["derived"]["attacking_residual"] >= settings["minimum_attacking_residual"]
    ]
    eligible.sort(
        key=lambda row: (
            _descending(row["derived"]["attacking_residual"]),
            _descending(row["derived"]["actual_gi"]),
            _number(row["derived"]["xgi"]) if _number(row["derived"]["xgi"]) is not None else math.inf,
            row["player_id"],
        )
    )
    result = []
    for row in eligible[: settings["rows"]]:
        result.append(
            _public_market_row(
                row,
                [
                    _reason(
                        "DERIVED",
                        "attacking_residual",
                        ">=",
                        settings["minimum_attacking_residual"],
                        row["derived"]["attacking_residual"],
                        f"actual_gi {row['derived']['actual_gi']} with xGI {row['derived']['xgi']}; attacking_residual {row['derived']['attacking_residual']} meets configured ahead-of-process criteria.",
                    )
                ],
            )
        )
    return {
        "title": "Attacking Output Ahead of Process Signals",
        "basis": "DERIVED actual_gi minus xGI; FPL points are context only",
        "thresholds": {
            "minimum_minutes": settings["minimum_minutes"],
            "minimum_actual_gi": settings["minimum_actual_gi"],
            "minimum_attacking_residual": settings["minimum_attacking_residual"],
        },
        "rows": result,
    }


def _defensive_contributions(
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    gw: int,
    gaps: list[str],
) -> dict[str, Any]:
    settings = config["sections"]["defensive_contributions"]
    position_rules = settings["position_rules"]
    empty = {
        "title": "Defensive Contributions",
        "basis": "FACT official per-fixture defensive-contribution counts vs positional per-match thresholds",
        "position_rules": {
            "DEF": position_rules["DEF"],
            "MID_FWD": {
                "threshold": position_rules["MID"]["threshold"],
                "action_family": position_rules["MID"]["action_family"],
            },
        },
        "position_groups": {"DEF": {"rows": []}, "MID_FWD": {"rows": []}},
        "gw_summary": [],
        "caveats": [
            "Thresholds are per match. DGW raw sums are never compared to a single threshold.",
            "Position groups are not comparable; the MID/FWD action family includes ball recoveries.",
        ],
    }
    if not settings["enabled"]:
        empty["status"] = "disabled"
        return empty

    evaluated: list[dict[str, Any]] = []
    missing_count = 0
    for player_row in rows:
        position = player_row["player"]["position"]
        if position not in {"DEF", "MID", "FWD"}:
            continue
        for fixture_row in player_row["fixture_rows"]:
            actions = _number(fixture_row.get("defensive_contribution"))
            if actions is None:
                missing_count += 1
                continue
            minutes = _number(fixture_row.get("minutes"))
            if minutes is None:
                missing_count += 1
                continue
            if actions < 0 or minutes < 0:
                raise MarketReportDataError(
                    f"Negative defensive contribution/minutes for player {player_row['player_id']} fixture {fixture_row['fixture_id']}"
                )
            rule = position_rules[position]
            threshold = int(rule["threshold"])
            margin = int(actions) - threshold
            hit = int(actions) >= threshold
            group = "DEF" if position == "DEF" else "MID_FWD"
            evaluated.append(
                {
                    "player": dict(player_row["player"]),
                    "player_id": player_row["player_id"],
                    "fixture_id": fixture_row["fixture_id"],
                    "minutes": _whole_or_float(minutes),
                    "relevant_actions": int(actions),
                    "action_family": rule["action_family"],
                    "positional_threshold": threshold,
                    "actions_vs_threshold": margin,
                    "threshold_hit": hit,
                    "official_defcon_points": None,
                    "group": group,
                    "trigger_reasons": [
                        _reason(
                            "FACT",
                            "relevant_actions",
                            ">=",
                            "threshold" if hit else "threshold - near_margin",
                            int(actions),
                            f"{int(actions)} {rule['action_family']} actions in fixture {fixture_row['fixture_id']} against threshold {threshold}.",
                        )
                    ],
                    "data_flags": list(player_row["data_flags"]),
                }
            )
    if not evaluated:
        empty["status"] = "data_gap"
        empty["data_gap"] = f"Defensive-contribution per-fixture player_gameweeks.defensive_contribution data unavailable for completed GW{gw} fixtures."
        _add_gap(gaps, empty["data_gap"])
        return empty
    if missing_count:
        _add_gap(
            gaps,
            f"Defensive-contribution data was unavailable for {missing_count} completed player-fixture rows and was not treated as zero.",
        )

    summaries: dict[int, dict[str, Any]] = {}
    group_rows: dict[str, list[dict[str, Any]]] = {"DEF": [], "MID_FWD": []}
    for item in evaluated:
        summary = summaries.setdefault(
            int(item["player_id"]),
            {
                "player_id": int(item["player_id"]),
                "player_name": item["player"]["name"],
                "threshold_hits_in_gw": 0,
                "official_defcon_points": None,
            },
        )
        if item["threshold_hit"]:
            summary["threshold_hits_in_gw"] += 1
        if item["minutes"] >= settings["minimum_minutes_per_fixture"] and (
            item["threshold_hit"]
            or item["relevant_actions"] >= item["positional_threshold"] - settings["near_threshold_margin"]
        ):
            public = dict(item)
            public.pop("group")
            group_rows[item["group"]].append(public)
    for group, values in group_rows.items():
        values.sort(
            key=lambda item: (
                0 if item["threshold_hit"] else 1,
                -item["actions_vs_threshold"],
                -float(item["minutes"]),
                item["player_id"],
                item["fixture_id"],
            )
        )
        empty["position_groups"][group]["rows"] = values[: settings["rows_per_position_group"]]
    empty["gw_summary"] = sorted(summaries.values(), key=lambda item: item["player_id"])
    empty["status"] = "ok"
    return empty


def _baseline_rows(
    conn: sqlite3.Connection,
    player_id: int,
    team_id: int | None,
    gw: int,
    config: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    settings = config["sections"]["minutes_watch"]
    if team_id is None:
        return None, "NO BASELINE AVAILABLE"
    team_fixtures = _rows(
        conn.execute(
            """SELECT id,event
                 FROM fixtures
                WHERE event IS NOT NULL AND event<? AND finished=1
                  AND (team_h=? OR team_a=?)
                ORDER BY event DESC, kickoff_time DESC, id DESC
                LIMIT ?""",
            (gw, int(team_id), int(team_id), settings["lookback_completed_fixtures"]),
        )
    )
    if not team_fixtures:
        return None, "NO BASELINE AVAILABLE"
    team_fixtures.reverse()
    fixture_ids = [int(fixture["id"]) for fixture in team_fixtures]
    previous_rows = repo.completed_player_fixture_rows(
        conn,
        player_id=player_id,
        fixture_ids=fixture_ids,
    )
    by_fixture = {int(row["fixture_id"]): row for row in previous_rows}
    if len(by_fixture) != len(fixture_ids):
        return None, "INSUFFICIENT BASELINE"
    previous = []
    for fixture in team_fixtures:
        row = by_fixture[int(fixture["id"])]
        if row.get("fixture_event") != fixture.get("event") or row.get("event") != fixture.get("event"):
            raise MarketReportDataError(
                f"Player {player_id} baseline row does not match completed team fixture {fixture['id']}"
            )
        previous.append(row)
    minutes = [_number(row.get("minutes")) for row in previous]
    if any(value is None for value in minutes):
        return None, "INSUFFICIENT BASELINE"
    if len(previous) < settings["minimum_baseline_fixtures"]:
        return None, "INSUFFICIENT BASELINE"
    starts = [_number(row.get("starts")) for row in previous]
    starts_available = all(value is not None for value in starts)
    start_count = sum(1 for value in starts if value not in (None, 0))
    total_minutes = sum(float(value) for value in minutes if value is not None)
    return (
        {
            "fixtures_considered": len(previous),
            "fixture_ids": fixture_ids,
            "total_minutes": _whole_or_float(total_minutes),
            "mean_minutes_per_fixture": _whole_or_float(total_minutes / len(previous)),
            "starts_available": starts_available,
            "starts": start_count if starts_available else None,
        },
        None,
    )


def _minutes_watch(
    conn: sqlite3.Connection,
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    gw: int,
    gaps: list[str],
) -> dict[str, Any]:
    settings = config["sections"]["minutes_watch"]
    result: dict[str, Any] = {
        "title": "Minutes Watch",
        "baseline_policy": "previous_completed_team_fixtures",
        "subsections": {
            "possible_minutes_drop": {"rows": []},
            "zero_minute_non_appearance": {"status": "enabled", "rows": []},
            "limited_minute_production": {"rows": []},
            "limited_minute_appearance": {"rows": []},
        },
    }
    if not settings["enabled"]:
        result["status"] = "disabled"
        return result
    if gw == 1:
        result["early_season_status"] = "NO BASELINE AVAILABLE"
        _add_gap(gaps, "Minutes Watch: NO BASELINE AVAILABLE for GW1.")
    else:
        result["early_season_status"] = None

    insufficient_players = 0
    for player_row in rows:
        baseline, baseline_status = _baseline_rows(
            conn,
            player_row["player_id"],
            player_row["team_id"],
            gw,
            config,
        )
        if baseline_status:
            insufficient_players += 1
        regular = bool(
            baseline
            and (
                baseline["mean_minutes_per_fixture"] >= settings["baseline_average_minutes_per_fixture"]
                or (baseline["starts_available"] and baseline["starts"] >= settings["baseline_starts"])
            )
        )
        for fixture_row in player_row["fixture_rows"]:
            minutes = _number(fixture_row.get("minutes"))
            if minutes is None or minutes < 0:
                continue
            points = _number(fixture_row.get("points"))
            fixture_xgi = _number(fixture_row.get("xgi"))
            base_item = {
                "player": dict(player_row["player"]),
                "player_id": player_row["player_id"],
                "fixture_id": fixture_row["fixture_id"],
                "minutes": _whole_or_float(minutes),
                "points": _whole_or_float(points),
                "xgi": _whole_or_float(fixture_xgi),
                "baseline": baseline,
                "baseline_status": baseline_status,
                "data_flags": list(player_row["data_flags"]),
            }
            if minutes == 0 and regular:
                item = dict(base_item)
                item["trigger_reasons"] = [
                    _reason(
                        "DERIVED",
                        "minutes",
                        "=",
                        0,
                        0,
                        "Non-appearance despite prior minutes baseline. Investigate availability or selection context.",
                    )
                ]
                result["subsections"]["zero_minute_non_appearance"]["rows"].append(item)
            elif minutes <= settings["low_minutes_threshold"] and regular:
                item = dict(base_item)
                item["trigger_reasons"] = [
                    _reason(
                        "DERIVED",
                        "minutes",
                        "<=",
                        settings["low_minutes_threshold"],
                        _whole_or_float(minutes),
                        "Completed-fixture minutes were low against the transparent prior-fixture baseline.",
                    )
                ]
                result["subsections"]["possible_minutes_drop"]["rows"].append(item)
            if 1 <= minutes <= settings["limited_minutes_maximum"]:
                limited_item = dict(base_item)
                limited_item["trigger_reasons"] = [
                    _reason(
                        "FACT",
                        "minutes",
                        "between",
                        [1, settings["limited_minutes_maximum"]],
                        _whole_or_float(minutes),
                        "Completed-fixture limited-minute appearance.",
                    )
                ]
                result["subsections"]["limited_minute_appearance"]["rows"].append(limited_item)
                if (
                    (points is not None and points >= settings["limited_minutes_minimum_points"])
                    or (fixture_xgi is not None and fixture_xgi >= settings["limited_minutes_minimum_xgi"])
                ):
                    production_item = dict(base_item)
                    production_item["trigger_reasons"] = [
                        _reason(
                            "DERIVED",
                            "limited_minutes_production",
                            ">=",
                            {
                                "points": settings["limited_minutes_minimum_points"],
                                "xgi": settings["limited_minutes_minimum_xgi"],
                            },
                            {"points": _whole_or_float(points), "xgi": _whole_or_float(fixture_xgi)},
                            "Strong production in limited minutes is a small-sample scout question, not a role conclusion.",
                        )
                    ]
                    result["subsections"]["limited_minute_production"]["rows"].append(production_item)
    if gw > 1 and insufficient_players:
        _add_gap(gaps, f"Minutes Watch: INSUFFICIENT BASELINE for {insufficient_players} players with completed GW{gw} rows.")
    for subsection in result["subsections"].values():
        subsection["rows"].sort(key=lambda item: (item["minutes"], item["player_id"], item["fixture_id"]))
        subsection["rows"] = subsection["rows"][: settings["rows_per_subsection"]]
    return result


def _our_squad(
    conn: sqlite3.Connection,
    brain_config: dict[str, Any],
    gw: int,
    market_rows: list[dict[str, Any]],
    memberships: dict[int, set[str]],
    config: dict[str, Any],
    gaps: list[str],
) -> tuple[dict[str, Any], set[int]]:
    settings = config["sections"]["our_squad"]
    result: dict[str, Any] = {
        "title": "Our Squad Context",
        "source": "MANUAL",
        "squad_resolution": "target_gw_only",
        "available": False,
        "rows": [],
    }
    if not settings["enabled"]:
        result["status"] = "disabled"
        return result, set()
    entry_id = brain_config.get("fpl_entry_id")
    if entry_id is None:
        result["data_gap"] = "Target-GW squad snapshot unavailable because Team ID is not configured."
        _add_gap(gaps, result["data_gap"])
        return result, set()
    picks = repo.squad_rows(conn, int(entry_id), gw)
    if not picks:
        result["data_gap"] = f"Target-GW squad snapshot unavailable for entry {entry_id}, GW{gw}; no latest-squad fallback was used."
        _add_gap(gaps, result["data_gap"])
        return result, set()
    market_by_player = {row["player_id"]: row for row in market_rows}
    result["available"] = True
    squad_ids: set[int] = set()
    for pick in picks:
        player_id = int(pick["player_id"])
        squad_ids.add(player_id)
        player = {
            "id": player_id,
            "name": pick.get("full_name") or pick.get("web_name") or f"Player {player_id}",
            "team": pick.get("team_short_name") or pick.get("team_name") or "UNK",
            "position": pick.get("singular_name_short") or "UNK",
        }
        market = market_by_player.get(player_id)
        result["rows"].append(
            {
                "position": pick.get("position"),
                "multiplier": pick.get("multiplier"),
                "is_captain": pick.get("is_captain") == 1,
                "is_vice_captain": pick.get("is_vice_captain") == 1,
                "player": player,
                "gw_facts": dict(market["facts"]) if market else None,
                "market_sections": sorted(memberships.get(player_id, set())) if settings["include_market_section_flags"] else [],
                "data_flags": list(market["data_flags"]) if market else ["target_gw_performance_row_missing"],
            }
        )
    result["rows"].sort(key=lambda item: (item.get("position") if item.get("position") is not None else 999, item["player"]["id"]))
    return result, squad_ids


def _add_memberships(memberships: dict[int, set[str]], source: str, rows: Iterable[dict[str, Any]]) -> None:
    for row in rows:
        player = row.get("player")
        if isinstance(player, dict) and player.get("id") is not None:
            memberships[int(player["id"])].add(source)


def _market_memberships(
    gw_stars: dict[str, Any],
    underlying: dict[str, Any],
    under_return: dict[str, Any],
    ahead: dict[str, Any],
    defcon: dict[str, Any],
    minutes: dict[str, Any],
) -> dict[int, set[str]]:
    memberships: dict[int, set[str]] = defaultdict(set)
    _add_memberships(memberships, "gw_stars", gw_stars.get("rows", []))
    for metric in ("xg", "xa", "xgi"):
        _add_memberships(memberships, f"underlying_leaders.{metric}", underlying.get(f"{metric}_leaders", {}).get("rows", []))
    _add_memberships(memberships, "attacking_process_under_return", under_return.get("rows", []))
    _add_memberships(memberships, "attacking_output_ahead_of_process", ahead.get("rows", []))
    for group in ("DEF", "MID_FWD"):
        _add_memberships(
            memberships,
            "defensive_contributions",
            defcon.get("position_groups", {}).get(group, {}).get("rows", []),
        )
    for name, subsection in minutes.get("subsections", {}).items():
        _add_memberships(memberships, f"minutes_watch.{name}", subsection.get("rows", []))
    return memberships


def _questions_for_sources(sources: Iterable[str]) -> list[str]:
    source_set = set(sources)
    questions: list[str] = []
    mappings = (
        (
            {"minutes_watch.possible_minutes_drop"},
            "Investigate expected minutes after the completed-fixture minutes change.",
        ),
        (
            {"minutes_watch.zero_minute_non_appearance"},
            "Check availability, suspension, matchday squad status, or selection context.",
        ),
        (
            {"minutes_watch.limited_minute_production"},
            "Investigate whether the role and minutes are likely to expand or remain limited.",
        ),
        (
            {"attacking_process_under_return"},
            "Investigate whether the underlying actions were repeatable or isolated events.",
        ),
        (
            {"attacking_output_ahead_of_process"},
            "Review whether attacking returns included events not captured by xGI; FPL assists may not map exactly to xA.",
        ),
        (
            {"underlying_leaders.xg", "underlying_leaders.xa", "underlying_leaders.xgi"},
            "Investigate role, chance quality, set-piece involvement, and game-state context behind the underlying numbers.",
        ),
        (
            {"defensive_contributions"},
            "Investigate whether the defensive-contribution role appears stable.",
        ),
    )
    for trigger_sources, question in mappings:
        if source_set.intersection(trigger_sources) and question not in questions:
            questions.append(question)
    if not questions:
        questions.append("Review the completed-GW statistical context before drawing tactical conclusions.")
    return questions


def _scout_candidates(
    market_rows: list[dict[str, Any]],
    memberships: dict[int, set[str]],
    squad: dict[str, Any],
    squad_ids: set[int],
    config: dict[str, Any],
    gw: int,
    gaps: list[str],
) -> dict[str, Any]:
    settings = config["sections"]["scout_candidates"]
    result: dict[str, Any] = {
        "title": "Statistical Market Triggers / Scout Candidates",
        "market_triggered_cap": settings["maximum_market_triggered_candidates"],
        "market_triggered_count": 0,
        "candidates": [],
        "squad_uncertainties": [],
    }
    if not settings["enabled"]:
        result["status"] = "disabled"
        return result
    if not squad.get("available"):
        result["status"] = "data_gap"
        result["data_gap"] = "Market-triggered non-squad candidates withheld because the exact target-GW squad snapshot is unavailable."
        _add_gap(gaps, result["data_gap"])
        return result

    rows_by_id = {row["player_id"]: row for row in market_rows}
    uncertainty_sources = {
        "minutes_watch.possible_minutes_drop",
        "minutes_watch.zero_minute_non_appearance",
        "minutes_watch.limited_minute_production",
    }
    for player_id in sorted(squad_ids):
        sources = memberships.get(player_id, set())
        active_sources = sorted(sources.intersection(uncertainty_sources))
        if not active_sources or player_id not in rows_by_id:
            continue
        player = rows_by_id[player_id]["player"]
        result["squad_uncertainties"].append(
            {
                "player_id": player_id,
                "player_name": player["name"],
                "source_sections": active_sources,
                "research_questions": _questions_for_sources(active_sources)[: settings["maximum_questions_per_player"]],
            }
        )

    ranked: list[tuple[int, float, float, int, dict[str, Any]]] = []
    for player_id, sources in memberships.items():
        if player_id in squad_ids or player_id not in rows_by_id:
            continue
        row = rows_by_id[player_id]
        source_list = sorted(sources)
        if not source_list:
            continue
        if "minutes_watch.limited_minute_production" in sources:
            priority, category = 4, "LIMITED_MINUTES_PRODUCTION"
        elif "defensive_contributions" in sources:
            priority, category = 5, "DEFCON_STABILITY"
        elif (
            len(source_list) >= 2
            or (
                "attacking_process_under_return" in sources
                and any(source.startswith("underlying_leaders.") for source in sources)
            )
        ):
            priority, category = 3, "STRONG_STATISTICAL_TRIGGER"
        else:
            priority, category = 6, "MARKET_TRIGGER"
        xgi = _number(row["derived"]["xgi"]) or 0.0
        residual = abs(_number(row["derived"]["attacking_residual"]) or 0.0)
        evidence = [
            f"GW{gw} {source.replace('_', ' ')} signal present."
            for source in source_list
        ]
        if row["derived"]["xgi"] is not None:
            evidence.append(f"GW{gw} xGI {row['derived']['xgi']}.")
        if row["derived"]["actual_gi"] is not None:
            evidence.append(f"GW{gw} actual_gi {row['derived']['actual_gi']}.")
        if row["derived"]["attacking_residual"] is not None:
            evidence.append(f"GW{gw} attacking_residual {row['derived']['attacking_residual']}.")
        candidate = {
            "player_id": player_id,
            "player_name": row["player"]["name"],
            "priority_category": category,
            "source_sections": source_list,
            "evidence": evidence,
            "research_questions": _questions_for_sources(source_list)[: settings["maximum_questions_per_player"]],
            "inclusion_reason": "Completed-GW statistical signal selected by deterministic category and evidence ordering.",
            "priority_category_reason": f"Category {priority}: {category}.",
        }
        ranked.append((priority, -xgi, -residual, player_id, candidate))
    ranked.sort(key=lambda item: item[:4])
    for index, (_, _, _, player_id, candidate) in enumerate(
        ranked[: settings["maximum_market_triggered_candidates"]], start=1
    ):
        candidate["candidate_id"] = f"gw{gw:02d}-player{player_id}-market-{index:03d}"
        result["candidates"].append(candidate)
    result["market_triggered_count"] = len(result["candidates"])
    return result


def _snapshot_state(rows: list[dict[str, Any]], db_path: str | Path, config: dict[str, Any]) -> dict[str, Any]:
    timestamps = sorted({row["facts"]["snapshot_as_of"] for row in rows if row["facts"]["snapshot_as_of"]})
    caveat = None
    if not timestamps:
        caveat = "No player snapshot is available for the target-GW performance rows."
    elif len(timestamps) > 1:
        caveat = "Latest available snapshot is selected per player; captured_at values differ across players."
    return {
        "official_source": "local_sqlite",
        "db_identity": str(Path(db_path).resolve()),
        "config_hash": market_config_hash(config),
        "price_ownership_snapshot": {
            "source": "player_snapshots",
            "as_of": timestamps[-1] if timestamps else None,
            "caveat": caveat,
        },
    }


def _standing_caveats() -> list[str]:
    return [
        "Single Gameweek samples are weak.",
        "This report does not recommend transfers, captaincy, or chips.",
        "FPL points and xGI are not comparable dimensions; attacking signals use goal involvements only.",
        "Ownership is context only, not player quality.",
        "Defensive-contribution groups are positional and per-match; raw counts are not comparable across groups.",
        "Tactical explanations require Scout Operations research.",
    ]


def build_market_report(
    conn: sqlite3.Connection,
    brain_config: dict[str, Any],
    market_config: dict[str, Any],
    gw: int,
    *,
    db_path: str | Path,
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    """Build the canonical in-memory Market Report object from local facts."""

    market_config = validate_market_config(deepcopy(market_config))
    if not _is_int(gw, 1):
        raise MarketReportDataError("gw must be a positive integer")
    completion = assess_gameweek_completion(conn, gw, market_config)
    if completion["status"] != "complete" and not allow_incomplete:
        detail = "; ".join(completion["blockers"]) or "completion could not be proven"
        raise IncompleteGameweekError(f"GW{gw} incomplete or unverifiable: {detail}")

    gaps: list[str] = []
    if completion["status"] != "complete":
        _add_gap(gaps, f"PREVIEW ONLY: GW{gw} completion is unverified: " + "; ".join(completion["blockers"]))
    _validate_target_performance_provenance(conn, gw, gaps)
    market_rows = _normalise_market_rows(conn, gw, completion["fixtures"], market_config, gaps)
    if market_rows:
        xg_unknown = sum(1 for row in market_rows if row["facts"]["xg"] is None)
        xa_unknown = sum(1 for row in market_rows if row["facts"]["xa"] is None)
        xgi_unknown = sum(1 for row in market_rows if row["derived"]["xgi"] is None)
        price_unknown = sum(1 for row in market_rows if row["facts"]["price"] is None)
        ownership_unknown = sum(1 for row in market_rows if row["facts"]["ownership_pct"] is None)
        if xg_unknown:
            _add_gap(gaps, f"xG unavailable for {xg_unknown} completed-GW player rows.")
        if xa_unknown:
            _add_gap(gaps, f"xA unavailable for {xa_unknown} completed-GW player rows.")
        if xgi_unknown:
            _add_gap(gaps, f"xGI unavailable for {xgi_unknown} completed-GW player rows.")
        if price_unknown:
            _add_gap(gaps, f"Price unavailable for {price_unknown} completed-GW player rows.")
        if ownership_unknown:
            _add_gap(gaps, f"Ownership unavailable for {ownership_unknown} completed-GW player rows.")

    gw_stars = _gw_stars(market_rows, market_config)
    underlying = _underlying_leaders(market_rows, market_config)
    under_return = _attacking_under_return(market_rows, market_config)
    ahead = _attacking_ahead(market_rows, market_config)
    defcon = _defensive_contributions(market_rows, market_config, gw, gaps)
    minutes = _minutes_watch(conn, market_rows, market_config, gw, gaps)
    memberships = _market_memberships(gw_stars, underlying, under_return, ahead, defcon, minutes)
    squad, squad_ids = _our_squad(conn, brain_config, gw, market_rows, memberships, market_config, gaps)
    candidates = _scout_candidates(market_rows, memberships, squad, squad_ids, market_config, gw, gaps)

    players = _player_rows(conn)
    all_team_ids = {row.get("team_id") for row in players.values() if row.get("team_id") is not None}
    team_counts = _team_fixture_counts(completion["fixtures"])
    contains_blank = any(int(team_id) not in team_counts for team_id in all_team_ids)
    report_status = "complete" if completion["status"] == "complete" else "preview_incomplete"
    return {
        "schema_version": MARKET_REPORT_SCHEMA_VERSION,
        "report_key": f"gw{gw:02d}",
        "gameweek": {
            "gw": gw,
            "status": report_status,
            "completion_basis": completion["completion_basis"],
            "completion_blockers": completion["blockers"],
            "fixtures_total": completion["fixtures_total"],
            "fixtures_finished": completion["fixtures_finished"],
            "fixtures_provisional": completion["fixtures_provisional"],
            "contains_dgw_teams": any(count > 1 for count in team_counts.values()),
            "contains_blank_teams": contains_blank,
        },
        "source_state": _snapshot_state(market_rows, db_path, market_config),
        "report_policy": {
            "no_recommendations": True,
            "no_composite_score": True,
            "no_points_vs_xgi_comparison": True,
            "no_single_gw_per90": True,
            "defcon_per_fixture_only": True,
            "squad_resolution": "target_gw_only",
            "single_gw_sample_warning": True,
        },
        "sections": {
            "gw_stars": gw_stars,
            "underlying_leaders": underlying,
            "attacking_process_under_return": under_return,
            "attacking_output_ahead_of_process": ahead,
            "defensive_contributions": defcon,
            "minutes_watch": minutes,
        },
        "our_squad": squad,
        "scout_candidates": candidates,
        "data_gaps": gaps,
        "caveats": _standing_caveats(),
    }


def render_market_json(report: dict[str, Any]) -> str:
    """Render canonical deterministic JSON from the report object."""

    return json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _format(value: Any, decimals: int = 2) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    number = _number(value)
    if number is not None:
        if number.is_integer():
            return str(int(number))
        return f"{number:.{decimals}f}"
    return str(value)


def _markdown_table(headers: list[str], values: list[list[Any]]) -> list[str]:
    if not values:
        return ["No rows."]
    clean_headers = [str(header) for header in headers]
    lines = [
        "| " + " | ".join(clean_headers) + " |",
        "| " + " | ".join("---" for _ in clean_headers) + " |",
    ]
    for row in values:
        cells = [str(value).replace("|", "&#124;").replace("\n", " ") for value in row]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _trigger_text(row: dict[str, Any]) -> str:
    reasons = row.get("trigger_reasons", [])
    if not reasons:
        return "—"
    return "; ".join(str(reason.get("message", "")) for reason in reasons)


def _signal_table(rows: list[dict[str, Any]], include_rank: bool = True) -> list[list[Any]]:
    values = []
    for row in rows:
        facts = row["facts"]
        derived = row["derived"]
        cells = []
        if include_rank:
            cells.append(row.get("rank", "—"))
        cells.extend(
            [
                row["player"]["name"],
                row["player"]["team"],
                row["player"]["position"],
                _format(facts.get("price"), 1),
                _format(facts.get("ownership_pct"), 1),
                _format(facts.get("minutes")),
                _format(facts.get("points")),
                _format(facts.get("goals")),
                _format(facts.get("assists")),
                _format(facts.get("clean_sheets")),
                _format(facts.get("bonus")),
                _format(facts.get("bps")),
                _format(facts.get("xg")),
                _format(facts.get("xa")),
                _format(derived.get("xgi")),
                _format(derived.get("actual_gi")),
                _format(derived.get("attacking_residual")),
                _trigger_text(row),
            ]
        )
        values.append(cells)
    return values


def render_market_markdown(report: dict[str, Any]) -> str:
    """Render Markdown solely from the canonical in-memory report object."""

    gw = report["gameweek"]["gw"]
    sections = report["sections"]
    lines = [
        f"# FPL Brain — Post-GW Market Report — GW{gw:02d}",
        "",
        f"Status: {report['gameweek']['status']}",
        "Source: Local official FPL database",
        "Decision policy: No transfers, captaincy, chips, buys, sells, or holds are generated by this report.",
        "",
        "## Caveats",
        "",
    ]
    lines.extend(f"- {caveat}" for caveat in report["caveats"])
    lines.extend(["", "## GW Stars", ""])
    stars = sections["gw_stars"]
    lines.append(f"Rows: {len(stars.get('rows', []))}")
    lines.extend(
        _markdown_table(
            [
                "Rank", "Player", "Team", "Pos", "£", "Own%", "Min", "Pts", "G", "A", "CS", "Bonus",
                "BPS", "xG", "xA", "xGI", "GI", "Residual", "Trigger reasons",
            ],
            _signal_table(stars.get("rows", [])),
        )
    )

    lines.extend(["", "## Underlying Leaders", ""])
    underlying = sections["underlying_leaders"]
    for metric, title in (("xg", "xG Leaders"), ("xa", "xA Leaders"), ("xgi", "xGI Leaders")):
        leader_rows = underlying.get(f"{metric}_leaders", {}).get("rows", [])
        lines.extend([f"### {title}", "", f"Rows: {len(leader_rows)}"])
        lines.extend(
            _markdown_table(
                [
                    "Rank", "Player", "Team", "Pos", "£", "Own%", "Min", "Pts", "G", "A", "CS", "Bonus",
                    "BPS", "xG", "xA", "xGI", "GI", "Residual", "Trigger reasons",
                ],
                _signal_table(leader_rows),
            )
        )
        lines.append("")

    for key, title in (
        ("attacking_process_under_return", "Attacking Process Under-Return Signals"),
        ("attacking_output_ahead_of_process", "Attacking Output Ahead of Process Signals"),
    ):
        section = sections[key]
        lines.extend([f"## {title}", "", section.get("basis", ""), f"Rows: {len(section.get('rows', []))}"])
        lines.extend(
            _markdown_table(
                [
                    "Player", "Team", "Pos", "£", "Own%", "Min", "Pts", "G", "A", "CS", "Bonus", "BPS",
                    "xG", "xA", "xGI", "GI", "Residual", "Trigger reasons",
                ],
                _signal_table(section.get("rows", []), include_rank=False),
            )
        )
        lines.append("")

    defcon = sections["defensive_contributions"]
    lines.extend(["## Defensive Contributions", "", defcon["basis"]])
    if defcon.get("status") == "data_gap":
        lines.append(f"DATA GAP: {defcon['data_gap']}")
    for group, title in (("DEF", "DEF scoring group"), ("MID_FWD", "MID/FWD scoring group")):
        group_rows = defcon.get("position_groups", {}).get(group, {}).get("rows", [])
        rule = defcon["position_rules"][group]
        lines.extend(
            [
                "",
                f"### {title} — threshold {rule['threshold']}, action family {rule['action_family']}",
                "",
                f"Rows: {len(group_rows)}",
            ]
        )
        lines.extend(
            _markdown_table(
                ["Player", "Team", "Fixture", "Min", "Actions", "Threshold", "Margin", "Hit", "DC Pts"],
                [
                    [
                        row["player"]["name"],
                        row["player"]["team"],
                        row["fixture_id"],
                        _format(row["minutes"]),
                        row["relevant_actions"],
                        row["positional_threshold"],
                        row["actions_vs_threshold"],
                        _format(row["threshold_hit"]),
                        _format(row["official_defcon_points"]),
                    ]
                    for row in group_rows
                ],
            )
        )
    summary_rows = sorted(
        (
            row
            for row in defcon.get("gw_summary", [])
            if row.get("threshold_hits_in_gw", 0) > 0
        ),
        key=lambda row: (-row["threshold_hits_in_gw"], row["player_id"]),
    )
    lines.extend(
        [
            "",
            "GW summary: threshold_hits_in_gw is counted from fixture-level hits only.",
            "",
            f"Rows: {len(summary_rows)}",
        ]
    )
    if summary_rows:
        lines.extend(
            _markdown_table(
                ["Player ID", "Player", "Threshold Hits", "Official DC Pts"],
                [
                    [
                        row["player_id"],
                        row["player_name"],
                        row["threshold_hits_in_gw"],
                        _format(row["official_defcon_points"]),
                    ]
                    for row in summary_rows
                ],
            )
        )
    else:
        lines.append("No players recorded a defensive-contribution threshold hit in this Gameweek.")

    minutes = sections["minutes_watch"]
    lines.extend(["", "## Minutes Watch", "", "Baseline: previous completed team fixtures; blanks are excluded and DGW fixtures count separately."])
    if minutes.get("early_season_status"):
        lines.append(f"Status: {minutes['early_season_status']}")
    labels = {
        "possible_minutes_drop": "Possible Minutes Drop",
        "zero_minute_non_appearance": "Zero-Minute Non-Appearance",
        "limited_minute_production": "Limited-Minute Production",
        "limited_minute_appearance": "Limited-Minute Appearance",
    }
    for key, title in labels.items():
        subsection = minutes["subsections"][key]
        rows = subsection.get("rows", [])
        lines.extend(["", f"### {title}", "", f"Rows: {len(rows)}"])
        lines.extend(
            _markdown_table(
                ["Player", "Fixture", "Min", "Pts", "xGI", "Baseline Mean", "Baseline Status"],
                [
                    [
                        row["player"]["name"],
                        row["fixture_id"],
                        _format(row["minutes"]),
                        _format(row["points"]),
                        _format(row["xgi"]),
                        _format((row.get("baseline") or {}).get("mean_minutes_per_fixture")),
                        row.get("baseline_status") or "available",
                    ]
                    for row in rows
                ],
            )
        )

    squad = report["our_squad"]
    lines.extend(["", "## Our Squad Context", "", "Context only. This section does not generate transfer action."])
    if not squad["available"]:
        lines.append(f"DATA GAP: {squad.get('data_gap', 'Target-GW squad snapshot unavailable.')}")
    else:
        lines.append(f"Rows: {len(squad['rows'])}")
        lines.extend(
            _markdown_table(
                ["Pick", "Player", "Team", "Pos", "Min", "Pts", "Market Sections"],
                [
                    [
                        row.get("position"),
                        row["player"]["name"],
                        row["player"]["team"],
                        row["player"]["position"],
                        _format((row.get("gw_facts") or {}).get("minutes")),
                        _format((row.get("gw_facts") or {}).get("points")),
                        ", ".join(row["market_sections"]) or "—",
                    ]
                    for row in squad["rows"]
                ],
            )
        )

    candidates = report["scout_candidates"]
    lines.extend(
        [
            "",
            "## Statistical Market Triggers / Scout Candidates",
            "",
            f"Market-triggered non-squad candidates: {candidates['market_triggered_count']}/{candidates['market_triggered_cap']}",
        ]
    )
    if candidates.get("data_gap"):
        lines.append(f"DATA GAP: {candidates['data_gap']}")
    lines.extend(
        _markdown_table(
            ["ID", "Player", "Priority", "Source Sections", "Research Questions"],
            [
                [
                    row["candidate_id"],
                    row["player_name"],
                    row["priority_category"],
                    ", ".join(row["source_sections"]),
                    " ".join(row["research_questions"]),
                ]
                for row in candidates["candidates"]
            ],
        )
    )
    if candidates["squad_uncertainties"]:
        lines.extend(["", "Squad uncertainties:"])
        for item in candidates["squad_uncertainties"]:
            lines.append(f"- {item['player_name']}: " + " ".join(item["research_questions"]))

    lines.extend(["", "## Data Gaps / Caveats", ""])
    if report["data_gaps"]:
        lines.extend(f"- {gap}" for gap in report["data_gaps"])
    else:
        lines.append("- No known data gaps.")
    return "\n".join(lines).rstrip() + "\n"


def _write_temp(directory: Path, name: str, content: str) -> Path:
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
        prefix=f".{name}.",
        suffix=".tmp",
        dir=directory,
        delete=False,
    )
    try:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
        return Path(handle.name)
    finally:
        handle.close()


def write_market_report(
    report: dict[str, Any],
    market_config: dict[str, Any],
    out_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Atomically write all configured final artifacts or restore prior files."""

    config = validate_market_config(deepcopy(market_config))
    output_root = Path(out_dir if out_dir is not None else config["output"]["base_dir"])
    gw = int(report["gameweek"]["gw"])
    directory = output_root / f"gw{gw:02d}"
    targets: list[tuple[Path, str]] = []
    if config["output"]["write_json"]:
        targets.append((directory / "post_gw_market_report.json", render_market_json(report)))
    if config["output"]["write_markdown"]:
        targets.append((directory / "post_gw_market_report.md", render_market_markdown(report)))
    if not targets:
        raise MarketReportOutputError("No output artifact is enabled")

    temp_paths: dict[Path, Path] = {}
    backups: dict[Path, Path] = {}
    replaced: list[Path] = []
    try:
        directory.mkdir(parents=True, exist_ok=True)
        for target, content in targets:
            temp_paths[target] = _write_temp(directory, target.name, content)
        for target, _ in targets:
            if target.exists():
                backup = directory / f".{target.name}.{os.getpid()}.bak"
                backup.unlink(missing_ok=True)
                os.replace(target, backup)
                backups[target] = backup
        for target, _ in targets:
            os.replace(temp_paths[target], target)
            replaced.append(target)
        for backup in backups.values():
            backup.unlink(missing_ok=True)
    except OSError as exc:
        for target in replaced:
            target.unlink(missing_ok=True)
        for target, backup in backups.items():
            if backup.exists():
                os.replace(backup, target)
        raise MarketReportOutputError(f"Could not write Market Report artifacts atomically: {exc}") from exc
    finally:
        for temp in temp_paths.values():
            temp.unlink(missing_ok=True)
        for backup in backups.values():
            backup.unlink(missing_ok=True)
    return {target.suffix.lstrip("."): target for target, _ in targets}
