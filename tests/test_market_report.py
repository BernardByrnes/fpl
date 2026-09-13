from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from fpl_brain import repositories as repo
import fpl_brain.market_report as market_report_module
from fpl_brain.config import DEFAULT_CONFIG
from fpl_brain.database import connect_database
from fpl_brain.market_report import (
    assess_gameweek_completion,
    build_market_report,
    IncompleteGameweekError,
    load_market_config,
    MarketReportDataError,
    MarketReportOutputError,
    open_read_only_database,
    render_market_json,
    render_market_markdown,
    write_market_report,
)
from fpl_brain.models import (
    EntryRecord,
    EventRecord,
    FixtureRecord,
    PickRecord,
    PlayerGameweekRecord,
    PlayerRecord,
    PlayerSnapshotRecord,
    PositionRecord,
    TeamRecord,
)


def _brain_config(tmp_path, entry_id: int | None = 99):
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["fpl_entry_id"] = entry_id
    config["paths"]["database"] = str(tmp_path / "fpl.db")
    config["paths"]["raw_dir"] = str(tmp_path / "raw")
    config["paths"]["exports_dir"] = str(tmp_path / "exports")
    return config


def _seed_reference(conn, player_specs):
    team_ids = sorted({spec.get("team_id", 1) for spec in player_specs})
    with conn:
        repo.upsert_teams(
            conn,
            [TeamRecord(id=team_id, name=f"Team {team_id}", short_name=f"T{team_id}") for team_id in team_ids],
        )
        repo.upsert_positions(
            conn,
            [
                PositionRecord(id=1, singular_name_short="GKP"),
                PositionRecord(id=2, singular_name_short="DEF"),
                PositionRecord(id=3, singular_name_short="MID"),
                PositionRecord(id=4, singular_name_short="FWD"),
            ],
        )
        repo.upsert_players(
            conn,
            [
                PlayerRecord(
                    id=spec["id"],
                    web_name=spec.get("name", f"P{spec['id']}"),
                    full_name=spec.get("name", f"Player {spec['id']}"),
                    team_id=spec.get("team_id", 1),
                    element_type=spec.get("position_id", 3),
                )
                for spec in player_specs
            ],
        )


def _insert_snapshots(conn, player_specs, captured_at="2026-08-25T22:00:00Z"):
    with conn:
        run = repo.create_fetch_run(conn, "fetch_fpl")
        repo.insert_snapshots(
            conn,
            [
                PlayerSnapshotRecord(
                    player_id=spec["id"],
                    captured_at=captured_at,
                    now_cost=spec.get("cost", 75),
                    selected_by_percent=spec.get("ownership", 10.0),
                    defensive_contribution=spec.get("snapshot_defcon"),
                    raw_json={},
                )
                for spec in player_specs
            ],
            run,
        )
        repo.finish_fetch_run(conn, run, "success", 1, ["bootstrap-static", "fixtures"], [])


def _build(path, brain_config, market_config, gw):
    conn = open_read_only_database(path)
    try:
        return build_market_report(conn, brain_config, market_config, gw, db_path=path)
    finally:
        conn.close()


def _complete_event(conn, event, fixtures):
    fixture_team_ids = sorted(
        {
            int(team_id)
            for fixture in fixtures
            for team_id in (fixture.team_h, fixture.team_a)
            if team_id is not None
        }
    )
    with conn:
        repo.upsert_teams(
            conn,
            [TeamRecord(id=team_id, name=f"Team {team_id}", short_name=f"T{team_id}") for team_id in fixture_team_ids],
        )
        repo.upsert_events(
            conn,
            [EventRecord(id=event, name=f"Gameweek {event}", finished=1, data_checked=1, is_previous=1)],
        )
        repo.upsert_fixtures(conn, fixtures)


def _assess_completion(tmp_path, event, fixture):
    config = _brain_config(tmp_path, entry_id=None)
    path = config["paths"]["database"]
    conn = connect_database(path)
    fixture_team_ids = sorted({int(fixture.team_h), int(fixture.team_a)})
    with conn:
        repo.upsert_teams(
            conn,
            [TeamRecord(id=team_id, name=f"Team {team_id}", short_name=f"T{team_id}") for team_id in fixture_team_ids],
        )
        repo.upsert_events(conn, [event])
        repo.upsert_fixtures(conn, [fixture])
    conn.close()

    read_conn = open_read_only_database(path)
    try:
        return assess_gameweek_completion(read_conn, event.id, load_market_config())
    finally:
        read_conn.close()


def test_completion_accepts_finished_provisional_fixture_after_event_is_final(tmp_path):
    result = _assess_completion(
        tmp_path,
        EventRecord(id=1, name="Gameweek 1", finished=1, data_checked=1),
        FixtureRecord(id=101, event=1, team_h=1, team_a=2, finished=1, finished_provisional=1, raw_json={}),
    )

    assert result["status"] == "complete"
    assert result["blockers"] == []
    assert result["fixtures_provisional"] == 1


def test_completion_rejects_unfinished_provisional_fixture(tmp_path):
    result = _assess_completion(
        tmp_path,
        EventRecord(id=1, name="Gameweek 1", finished=0, data_checked=0),
        FixtureRecord(id=101, event=1, team_h=1, team_a=2, finished=0, finished_provisional=1, raw_json={}),
    )

    assert result["status"] == "incomplete_or_unverifiable"
    assert "events.finished is not true" in result["blockers"]
    assert "events.data_checked is not true" in result["blockers"]
    assert "1 target-GW fixtures are unfinished" in result["blockers"]


def test_completion_rejects_unfinished_event_even_when_fixture_is_finished(tmp_path):
    result = _assess_completion(
        tmp_path,
        EventRecord(id=1, name="Gameweek 1", finished=0, data_checked=1),
        FixtureRecord(id=101, event=1, team_h=1, team_a=2, finished=1, finished_provisional=1, raw_json={}),
    )

    assert result["status"] == "incomplete_or_unverifiable"
    assert result["blockers"] == ["events.finished is not true"]


def test_completion_rejects_unchecked_event_with_finished_fixture(tmp_path):
    result = _assess_completion(
        tmp_path,
        EventRecord(id=1, name="Gameweek 1", finished=1, data_checked=0),
        FixtureRecord(id=101, event=1, team_h=1, team_a=2, finished=1, finished_provisional=1, raw_json={}),
    )

    assert result["status"] == "incomplete_or_unverifiable"
    assert result["blockers"] == ["events.data_checked is not true"]


def test_market_report_uses_completed_fixture_rows_for_zero_minute_nonappearance_only(tmp_path):
    config = _brain_config(tmp_path)
    path = config["paths"]["database"]
    conn = connect_database(path)
    players = [{"id": 1, "name": "Proven Zero", "team_id": 1, "position_id": 3}]
    _seed_reference(conn, players)
    _complete_event(
        conn,
        1,
        [FixtureRecord(id=101, event=1, team_h=1, team_a=2, finished=1, raw_json={})],
    )
    _complete_event(
        conn,
        2,
        [FixtureRecord(id=201, event=2, team_h=1, team_a=2, finished=1, raw_json={})],
    )
    with conn:
        repo.upsert_events(conn, [EventRecord(id=3, name="Gameweek 3", finished=0, data_checked=0, is_next=1)])
        repo.upsert_fixtures(conn, [FixtureRecord(id=301, event=3, team_h=1, team_a=2, finished=0, raw_json={})])
        repo.upsert_player_gameweeks(
            conn,
            [
                PlayerGameweekRecord(
                    player_id=1,
                    event=1,
                    fixture_id=101,
                    minutes=90,
                    starts=1,
                    total_points=6,
                    goals_scored=0,
                    assists=0,
                    clean_sheets=0,
                    bonus=0,
                    bps=20,
                    expected_goals=0.1,
                    expected_assists=0.1,
                    expected_goal_involvements=0.2,
                    raw_json={},
                ),
                PlayerGameweekRecord(
                    player_id=1,
                    event=2,
                    fixture_id=201,
                    minutes=0,
                    starts=0,
                    total_points=0,
                    goals_scored=0,
                    assists=0,
                    clean_sheets=0,
                    bonus=0,
                    bps=0,
                    expected_goals=0,
                    expected_assists=0,
                    expected_goal_involvements=0,
                    raw_json={},
                ),
                PlayerGameweekRecord(
                    player_id=1,
                    event=3,
                    fixture_id=301,
                    minutes=0,
                    source="element_summary",
                    raw_json={"minutes": 0},
                ),
                PlayerGameweekRecord(
                    player_id=1,
                    event=2,
                    fixture_id=-1,
                    minutes=0,
                    source="event_live",
                    raw_json={"sentinel": True},
                ),
            ],
        )
        conn.execute(
            """INSERT INTO player_gameweeks(
                 player_id,event,fixture_id,minutes,source,raw_json,updated_at
               ) VALUES (?,?,?,?,?,?,?)""",
            (1, 2, None, 0, "event_live", "{}", "2026-08-25T22:00:00Z"),
        )
    _insert_snapshots(conn, players)
    repo.upsert_squad_picks(conn, 99, 2, [PickRecord(player_id=1, position=1, raw_json={})])
    conn.close()

    read_conn = open_read_only_database(path)
    try:
        completed = repo.completed_player_fixture_rows(read_conn, event=2)
    finally:
        read_conn.close()
    assert [(row["fixture_id"], row["minutes"]) for row in completed] == [(201, 0)]

    market_config = load_market_config()
    market_config["sections"]["minutes_watch"]["minimum_baseline_fixtures"] = 1
    report = _build(path, config, market_config, 2)
    zero_rows = report["sections"]["minutes_watch"]["subsections"]["zero_minute_non_appearance"]["rows"]
    assert [(row["player_id"], row["fixture_id"], row["minutes"]) for row in zero_rows] == [(1, 201, 0)]
    assert report["sections"]["minutes_watch"]["subsections"]["possible_minutes_drop"]["rows"] == []
    target = report["sections"]["gw_stars"]["rows"]
    assert target == []


def test_market_report_defcon_is_per_fixture_and_never_uses_snapshot_or_gw_sum(tmp_path):
    config = _brain_config(tmp_path)
    path = config["paths"]["database"]
    conn = connect_database(path)
    players = [
        {"id": 1, "name": "Near Defender", "team_id": 1, "position_id": 2, "snapshot_defcon": 999},
        {"id": 2, "name": "Hitting Mid", "team_id": 2, "position_id": 3, "snapshot_defcon": 999},
        {"id": 3, "name": "DGW Mid", "team_id": 3, "position_id": 3, "snapshot_defcon": 999},
    ]
    _seed_reference(conn, players)
    _complete_event(
        conn,
        2,
        [
            FixtureRecord(id=201, event=2, team_h=1, team_a=4, finished=1, raw_json={}),
            FixtureRecord(id=202, event=2, team_h=2, team_a=4, finished=1, raw_json={}),
            FixtureRecord(id=203, event=2, team_h=3, team_a=4, finished=1, raw_json={}),
            FixtureRecord(id=204, event=2, team_h=3, team_a=5, finished=1, raw_json={}),
            FixtureRecord(id=205, event=2, team_h=2, team_a=5, finished=1, raw_json={}),
        ],
    )
    with conn:
        repo.upsert_player_gameweeks(
            conn,
            [
                PlayerGameweekRecord(player_id=1, event=2, fixture_id=201, minutes=90, total_points=6, goals_scored=0, assists=0, clean_sheets=1, bonus=0, bps=20, expected_goals=0, expected_assists=0, expected_goal_involvements=0, defensive_contribution=9, raw_json={}),
                PlayerGameweekRecord(player_id=2, event=2, fixture_id=202, minutes=90, total_points=8, goals_scored=1, assists=0, clean_sheets=0, bonus=0, bps=20, expected_goals=0.2, expected_assists=0, expected_goal_involvements=0.2, defensive_contribution=13, raw_json={}),
                PlayerGameweekRecord(player_id=2, event=2, fixture_id=205, minutes=90, total_points=2, goals_scored=0, assists=0, clean_sheets=0, bonus=0, bps=10, expected_goals=0, expected_assists=0, expected_goal_involvements=0, defensive_contribution=12, raw_json={}),
                PlayerGameweekRecord(player_id=3, event=2, fixture_id=203, minutes=90, total_points=2, goals_scored=0, assists=0, clean_sheets=0, bonus=0, bps=10, expected_goals=0, expected_assists=0, expected_goal_involvements=0, defensive_contribution=7, raw_json={}),
                PlayerGameweekRecord(player_id=3, event=2, fixture_id=204, minutes=90, total_points=2, goals_scored=0, assists=0, clean_sheets=0, bonus=0, bps=10, expected_goals=0, expected_assists=0, expected_goal_involvements=0, defensive_contribution=7, raw_json={}),
            ],
        )
    _insert_snapshots(conn, players)
    repo.upsert_squad_picks(conn, 99, 2, [PickRecord(player_id=1, position=1, raw_json={})])
    conn.close()

    report = _build(path, config, load_market_config(), 2)
    section = report["sections"]["defensive_contributions"]
    assert section["status"] == "ok"
    defender = section["position_groups"]["DEF"]["rows"][0]
    assert defender["relevant_actions"] == 9
    assert defender["actions_vs_threshold"] == -1
    assert defender["threshold_hit"] is False
    mid = section["position_groups"]["MID_FWD"]["rows"][0]
    assert mid["player"]["id"] == 2
    assert mid["relevant_actions"] == 13
    assert mid["threshold_hit"] is True
    summary = {row["player_id"]: row for row in section["gw_summary"]}
    assert summary[2]["threshold_hits_in_gw"] == 2
    assert summary[3]["threshold_hits_in_gw"] == 0
    assert summary[1]["official_defcon_points"] is None
    assert len(json.loads(render_market_json(report))["sections"]["defensive_contributions"]["gw_summary"]) == 3

    markdown = render_market_markdown(report)
    summary_markdown = markdown.split("GW summary:", 1)[1].split("\n## Minutes Watch", 1)[0]
    assert "Hitting Mid" in summary_markdown
    assert "Near Defender" not in summary_markdown
    assert "DGW Mid" not in summary_markdown
    assert "### DEF scoring group" in markdown
    assert "### MID/FWD scoring group" in markdown

    empty_report = copy.deepcopy(report)
    for row in empty_report["sections"]["defensive_contributions"]["gw_summary"]:
        row["threshold_hits_in_gw"] = 0
    empty_summary_markdown = render_market_markdown(empty_report).split("GW summary:", 1)[1].split("\n## Minutes Watch", 1)[0]
    assert "No players recorded a defensive-contribution threshold hit in this Gameweek." in empty_summary_markdown


def test_market_report_defcon_and_squad_emit_data_gaps_when_source_data_is_missing(tmp_path):
    config = _brain_config(tmp_path, entry_id=99)
    path = config["paths"]["database"]
    conn = connect_database(path)
    players = [{"id": 1, "name": "No DEFCON", "team_id": 1, "position_id": 3}]
    _seed_reference(conn, players)
    _complete_event(
        conn,
        1,
        [FixtureRecord(id=101, event=1, team_h=1, team_a=2, finished=1, raw_json={})],
    )
    with conn:
        repo.upsert_player_gameweeks(
            conn,
            [
                PlayerGameweekRecord(
                    player_id=1,
                    event=1,
                    fixture_id=101,
                    minutes=90,
                    total_points=5,
                    goals_scored=0,
                    assists=0,
                    clean_sheets=0,
                    bonus=0,
                    bps=10,
                    expected_goals=0.2,
                    expected_assists=0.1,
                    expected_goal_involvements=0.3,
                    defensive_contribution=None,
                    raw_json={},
                )
            ],
        )
        repo.insert_manager_state(
            conn,
            entry=EntryRecord(entry_id=99, player_name="Manager", team_name="Team"),
            history_row=None,
            active_chip=None,
            free_transfers_manual=None,
            fetch_run_id=None,
            event=1,
            raw_json={},
            captured_at="2026-08-25T22:00:00Z",
        )
    _insert_snapshots(conn, players)
    conn.close()

    report = _build(path, config, load_market_config(), 1)
    assert report["sections"]["defensive_contributions"]["status"] == "data_gap"
    assert report["our_squad"]["available"] is False
    assert "no latest-squad fallback" in report["our_squad"]["data_gap"]
    assert report["scout_candidates"]["status"] == "data_gap"
    assert any("Defensive-contribution" in gap for gap in report["data_gaps"])
    assert any("Target-GW squad snapshot unavailable" in gap for gap in report["data_gaps"])


def test_market_report_is_deterministic_renders_agree_and_caps_non_squad_candidates(tmp_path):
    config = _brain_config(tmp_path)
    path = config["paths"]["database"]
    conn = connect_database(path)
    players = [
        {"id": player_id, "name": f"Player {player_id}", "team_id": 1, "position_id": 3, "cost": 70 + player_id}
        for player_id in range(1, 26)
    ]
    _seed_reference(conn, players)
    _complete_event(
        conn,
        1,
        [FixtureRecord(id=101, event=1, team_h=1, team_a=2, finished=1, raw_json={})],
    )
    with conn:
        repo.upsert_player_gameweeks(
            conn,
            [
                PlayerGameweekRecord(
                    player_id=spec["id"],
                    event=1,
                    fixture_id=101,
                    minutes=90,
                    total_points=2,
                    goals_scored=0,
                    assists=0,
                    clean_sheets=0,
                    bonus=0,
                    bps=10,
                    expected_goals=0.1 * spec["id"],
                    expected_assists=0.05,
                    expected_goal_involvements=0.1 * spec["id"] + 0.05,
                    defensive_contribution=1,
                    raw_json={},
                )
                for spec in players
            ],
        )
        repo.upsert_squad_picks(
            conn,
            99,
            1,
            [PickRecord(player_id=player_id, position=player_id, raw_json={}) for player_id in range(1, 16)],
        )
    _insert_snapshots(conn, players)
    conn.close()

    market_config = load_market_config()
    first = _build(path, config, market_config, 1)
    second = _build(path, config, market_config, 1)
    json_first = render_market_json(first)
    markdown_first = render_market_markdown(first)
    assert json_first == render_market_json(second)
    assert markdown_first == render_market_markdown(second)
    assert "generated" not in first
    assert len(first["sections"]["defensive_contributions"]["gw_summary"]) == len(players)
    assert all(row["threshold_hits_in_gw"] == 0 for row in first["sections"]["defensive_contributions"]["gw_summary"])
    assert "No players recorded a defensive-contribution threshold hit in this Gameweek." in markdown_first
    candidates = first["scout_candidates"]["candidates"]
    assert first["scout_candidates"]["market_triggered_count"] == 8
    assert len(candidates) == 8
    assert all(candidate["player_id"] > 15 for candidate in candidates)
    assert len({candidate["candidate_id"] for candidate in candidates}) == 8
    assert json.loads(json_first) == first
    assert f"Rows: {len(first['sections']['gw_stars']['rows'])}" in markdown_first
    assert f"Rows: {len(first['sections']['underlying_leaders']['xg_leaders']['rows'])}" in markdown_first
    assert f"Rows: {len(first['sections']['underlying_leaders']['xa_leaders']['rows'])}" in markdown_first
    assert f"Rows: {len(first['sections']['underlying_leaders']['xgi_leaders']['rows'])}" in markdown_first
    assert f"Rows: {len(first['sections']['attacking_process_under_return']['rows'])}" in markdown_first
    assert f"Rows: {len(first['sections']['attacking_output_ahead_of_process']['rows'])}" in markdown_first
    assert all(candidate["candidate_id"] in markdown_first for candidate in candidates)

    out_dir = tmp_path / "market_outputs"
    written = write_market_report(first, market_config, out_dir)
    assert set(written) == {"json", "md"}
    assert written["json"].read_text(encoding="utf-8") == json_first
    assert written["md"].read_text(encoding="utf-8") == markdown_first
    assert not list((out_dir / "gw01").glob("*.tmp"))
    assert not list((out_dir / "gw01").glob("*.bak"))
    assert json.loads(json_first)["scout_candidates"]["market_triggered_count"] == 8


def test_minutes_watch_baseline_uses_completed_team_fixtures_and_counts_dgw_rows(tmp_path):
    config = _brain_config(tmp_path)
    path = config["paths"]["database"]
    conn = connect_database(path)
    players = [{"id": 1, "name": "Regular Starter", "team_id": 1, "position_id": 3}]
    _seed_reference(conn, players)
    _complete_event(
        conn,
        1,
        [FixtureRecord(id=101, event=1, team_h=1, team_a=2, finished=1, raw_json={})],
    )
    _complete_event(
        conn,
        3,
        [
            FixtureRecord(id=301, event=3, team_h=1, team_a=2, finished=1, raw_json={}),
            FixtureRecord(id=302, event=3, team_h=1, team_a=3, finished=1, raw_json={}),
        ],
    )
    _complete_event(
        conn,
        4,
        [FixtureRecord(id=401, event=4, team_h=1, team_a=2, finished=1, raw_json={})],
    )
    with conn:
        repo.upsert_player_gameweeks(
            conn,
            [
                PlayerGameweekRecord(player_id=1, event=1, fixture_id=101, minutes=90, starts=1, total_points=4, goals_scored=0, assists=0, clean_sheets=0, bonus=0, bps=10, expected_goals=0, expected_assists=0, expected_goal_involvements=0, raw_json={}),
                PlayerGameweekRecord(player_id=1, event=3, fixture_id=301, minutes=90, starts=1, total_points=4, goals_scored=0, assists=0, clean_sheets=0, bonus=0, bps=10, expected_goals=0, expected_assists=0, expected_goal_involvements=0, raw_json={}),
                PlayerGameweekRecord(player_id=1, event=3, fixture_id=302, minutes=90, starts=1, total_points=4, goals_scored=0, assists=0, clean_sheets=0, bonus=0, bps=10, expected_goals=0, expected_assists=0, expected_goal_involvements=0, raw_json={}),
                PlayerGameweekRecord(player_id=1, event=4, fixture_id=401, minutes=0, starts=0, total_points=0, goals_scored=0, assists=0, clean_sheets=0, bonus=0, bps=0, expected_goals=0, expected_assists=0, expected_goal_involvements=0, raw_json={}),
            ],
        )
        repo.upsert_squad_picks(conn, 99, 4, [PickRecord(player_id=1, position=1, raw_json={})])
    _insert_snapshots(conn, players)
    conn.close()

    report = _build(path, config, load_market_config(), 4)
    rows = report["sections"]["minutes_watch"]["subsections"]["zero_minute_non_appearance"]["rows"]
    assert len(rows) == 1
    assert rows[0]["fixture_id"] == 401
    assert rows[0]["baseline"] == {
        "fixtures_considered": 3,
        "fixture_ids": [101, 301, 302],
        "total_minutes": 270,
        "mean_minutes_per_fixture": 90,
        "starts_available": True,
        "starts": 3,
    }


def test_market_report_rejects_credible_unfinished_fixture_in_preview(tmp_path):
    config = _brain_config(tmp_path)
    path = config["paths"]["database"]
    conn = connect_database(path)
    players = [{"id": 1, "name": "Contradiction", "team_id": 1, "position_id": 3}]
    _seed_reference(conn, players)
    with conn:
        repo.upsert_events(conn, [EventRecord(id=1, name="Gameweek 1", finished=0, data_checked=0, is_next=1)])
        repo.upsert_teams(conn, [TeamRecord(id=2, name="Team 2", short_name="T2")])
        repo.upsert_fixtures(conn, [FixtureRecord(id=101, event=1, team_h=1, team_a=2, finished=0, raw_json={})])
        repo.upsert_player_gameweeks(
            conn,
            [PlayerGameweekRecord(player_id=1, event=1, fixture_id=101, minutes=45, total_points=3, raw_json={})],
        )
    conn.close()

    read_conn = open_read_only_database(path)
    try:
        with pytest.raises(MarketReportDataError, match="credible performance"):
            build_market_report(
                read_conn,
                config,
                load_market_config(),
                1,
                db_path=path,
                allow_incomplete=True,
            )
    finally:
        read_conn.close()


def test_market_report_marks_malformed_xg_unknown_without_zero_coercion(tmp_path):
    config = _brain_config(tmp_path)
    path = config["paths"]["database"]
    conn = connect_database(path)
    players = [{"id": 1, "name": "Malformed xG", "team_id": 1, "position_id": 3}]
    _seed_reference(conn, players)
    _complete_event(
        conn,
        1,
        [FixtureRecord(id=101, event=1, team_h=1, team_a=2, finished=1, raw_json={})],
    )
    with conn:
        repo.upsert_player_gameweeks(
            conn,
            [
                PlayerGameweekRecord(
                    player_id=1,
                    event=1,
                    fixture_id=101,
                    minutes=90,
                    total_points=2,
                    goals_scored=0,
                    assists=0,
                    clean_sheets=0,
                    bonus=0,
                    bps=10,
                    expected_goals="not-a-number",
                    expected_assists=0.3,
                    expected_goal_involvements=None,
                    raw_json={},
                )
            ],
        )
        repo.upsert_squad_picks(conn, 99, 1, [PickRecord(player_id=1, position=1, raw_json={})])
    _insert_snapshots(conn, players)
    conn.close()

    report = _build(path, config, load_market_config(), 1)
    star = report["sections"]["gw_stars"]["rows"][0]
    assert star["facts"]["xg"] is None
    assert star["derived"]["xgi"] is None
    assert "malformed_expected_goals" in star["data_flags"]
    assert report["sections"]["underlying_leaders"]["xgi_leaders"]["rows"] == []
    assert any("xG unavailable" in gap for gap in report["data_gaps"])
    assert any("xGI unavailable" in gap for gap in report["data_gaps"])


def test_market_report_atomic_write_restores_existing_paired_artifacts_on_failure(tmp_path, monkeypatch):
    config = _brain_config(tmp_path)
    path = config["paths"]["database"]
    conn = connect_database(path)
    players = [{"id": 1, "name": "Atomic Player", "team_id": 1, "position_id": 3}]
    _seed_reference(conn, players)
    _complete_event(
        conn,
        1,
        [FixtureRecord(id=101, event=1, team_h=1, team_a=2, finished=1, raw_json={})],
    )
    with conn:
        repo.upsert_player_gameweeks(
            conn,
            [
                PlayerGameweekRecord(
                    player_id=1,
                    event=1,
                    fixture_id=101,
                    minutes=90,
                    total_points=5,
                    goals_scored=0,
                    assists=0,
                    clean_sheets=0,
                    bonus=0,
                    bps=10,
                    expected_goals=0.1,
                    expected_assists=0.1,
                    expected_goal_involvements=0.2,
                    defensive_contribution=1,
                    raw_json={},
                )
            ],
        )
    _insert_snapshots(conn, players)
    conn.close()

    market_config = load_market_config()
    report = _build(path, config, market_config, 1)
    output_root = tmp_path / "reports"
    written = write_market_report(report, market_config, output_root)
    previous_json = written["json"].read_text(encoding="utf-8")
    previous_markdown = written["md"].read_text(encoding="utf-8")
    actual_replace = market_report_module.os.replace

    def fail_markdown_install(source, destination):
        source_path = Path(source)
        destination_path = Path(destination)
        if destination_path.name == "post_gw_market_report.md" and source_path.suffix == ".tmp":
            raise OSError("simulated markdown replacement failure")
        return actual_replace(source, destination)

    monkeypatch.setattr(market_report_module.os, "replace", fail_markdown_install)
    with pytest.raises(MarketReportOutputError, match="atomically"):
        write_market_report(report, market_config, output_root)

    assert written["json"].read_text(encoding="utf-8") == previous_json
    assert written["md"].read_text(encoding="utf-8") == previous_markdown
    assert not list(written["json"].parent.glob("*.tmp"))
    assert not list(written["json"].parent.glob("*.bak"))
