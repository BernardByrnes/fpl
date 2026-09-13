from __future__ import annotations

import copy
import json

from fpl_brain import repositories as repo
from fpl_brain.config import DEFAULT_CONFIG
from fpl_brain.database import connect_database
from fpl_brain.models import EntryRecord, EventRecord, FixtureRecord, HistoryRow, PickRecord, PlayerRecord, PlayerSnapshotRecord, PositionRecord, TeamRecord
from fpl_brain.reports import build_report, render_json, render_markdown


def _config(tmp_path):
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["paths"]["database"] = str(tmp_path / "fpl.db")
    config["paths"]["raw_dir"] = str(tmp_path / "raw")
    config["paths"]["exports_dir"] = str(tmp_path / "exports")
    return config


def _structural_setup(tmp_path, player_team_ids, event_ids=(3,)):
    config = _config(tmp_path)
    config["fpl_entry_id"] = 99
    conn = connect_database(config["paths"]["database"])
    with conn:
        repo.upsert_events(
            conn,
            [EventRecord(id=event_id, name=f"Gameweek {event_id}") for event_id in event_ids],
        )
        team_ids = sorted(set(player_team_ids.values()))
        repo.upsert_teams(
            conn,
            [TeamRecord(id=team_id, name=f"Team {team_id}", short_name=f"T{team_id}") for team_id in team_ids],
        )
        repo.upsert_positions(conn, [PositionRecord(id=1, singular_name_short="GKP")])
        repo.upsert_players(
            conn,
            [
                PlayerRecord(
                    id=player_id,
                    web_name=f"P{player_id}",
                    full_name=f"Player {player_id}",
                    team_id=team_id,
                    element_type=1,
                )
                for player_id, team_id in player_team_ids.items()
            ],
        )
    return config, conn


def test_empty_db_report_has_all_sections_and_gaps(tmp_path):
    config = _config(tmp_path)
    conn = connect_database(config["paths"]["database"])
    report = build_report(conn, config, 1)
    markdown = render_markdown(report)
    document = json.loads(render_json(report))
    assert "## DATA GAPS AND CAVEATS" in markdown
    assert "No GW performance data" in markdown
    headings = {line[3:] for line in markdown.splitlines() if line.startswith("## ")}
    assert headings == set(document["sections"])
    conn.close()


def test_scouting_notes_render_confidence_observation_and_stale(tmp_path):
    config = _config(tmp_path)
    conn = connect_database(config["paths"]["database"])
    with conn:
        repo.upsert_teams(conn, [TeamRecord(id=1, name="Arsenal", short_name="ARS")])
        repo.upsert_positions(conn, [PositionRecord(id=3, singular_name_short="MID")])
        repo.upsert_players(conn, [PlayerRecord(id=1, web_name="Raya", full_name="David Raya", team_id=1, element_type=3)])
        run = repo.create_fetch_run(conn, "fetch_fpl")
        repo.insert_snapshots(conn, [PlayerSnapshotRecord(player_id=1, captured_at="2026-08-19T00:00:00Z", now_cost=80, selected_by_percent=10, total_points=50, raw_json={})], run)
        repo.add_watchlist(conn, 1, "WATCH", "test")
        import_id = repo.insert_scouting_import(conn, {"source_file": "test", "file_sha256": "x", "players_total": 1, "players_resolved": 1, "notes_inserted": 1})
        repo.insert_scouting_note(
            conn,
            {
                "import_id": import_id,
                "player_id": 1,
                "key": "rotation_risk",
                "value_text": "high",
                "confidence": "medium",
                "observed_at": "2020-01-01T00:00:00Z",
                "expires_at": "2020-01-02T00:00:00Z",
            },
        )
    markdown = render_markdown(build_report(conn, config, 1))
    note_lines = [line for line in markdown.splitlines() if "[SCOUTING]" in line and "rotation_risk" in line]
    assert note_lines and "confidence medium" in note_lines[0] and "observed 2020-01-01T00:00:00Z" in note_lines[0]
    assert "STALE" in note_lines[0]
    conn.close()


def test_populated_report_uses_manager_team_name_fixture_names_strength_outlook_and_legal_risk(tmp_path):
    config = _config(tmp_path)
    config["fpl_entry_id"] = 99
    conn = connect_database(config["paths"]["database"])
    with conn:
        repo.upsert_events(conn, [EventRecord(id=1, name="Gameweek 1", is_current=1, deadline_time="2026-08-20T00:00:00Z")])
        repo.upsert_teams(
            conn,
            [
                TeamRecord(id=1, name="Arsenal", short_name="ARS", strength_defence_home=50, strength_defence_away=55, strength_attack_home=60, strength_attack_away=65),
                TeamRecord(id=2, name="Chelsea", short_name="CHE", strength_defence_home=30, strength_defence_away=35, strength_attack_home=40, strength_attack_away=45),
            ],
        )
        repo.upsert_positions(conn, [PositionRecord(id=1, singular_name_short="GKP")])
        repo.upsert_players(
            conn,
            [PlayerRecord(id=player_id, web_name=f"P{player_id}", full_name=f"Player {player_id}", team_id=1 if player_id <= 3 else 2, element_type=1) for player_id in range(1, 16)],
        )
        repo.upsert_fixtures(conn, [FixtureRecord(id=1, event=1, team_h=1, team_a=2, team_h_difficulty=3, team_a_difficulty=4, raw_json={})])
        repo.insert_manager_state(
            conn,
            EntryRecord(entry_id=99, player_name="Ada Lovelace", team_name="The Analytical XI"),
            None,
            None,
            None,
            None,
            1,
            {},
            "2026-08-19T00:00:00Z",
        )
        repo.upsert_squad_picks(conn, 99, 1, [PickRecord(player_id=player_id, position=player_id, raw_json={}) for player_id in range(1, 16)])
        run = repo.create_fetch_run(conn, "fetch_fpl")
        repo.finish_fetch_run(conn, run, "partial", 1, ["bootstrap-static"], [{"endpoint": "fixtures", "error": "timeout"}])
    markdown = render_markdown(build_report(conn, config, 1))
    assert "FPL team: The Analytical XI" in markdown
    assert "Manager: Ada Lovelace" in markdown
    assert "ARS vs CHE" in markdown
    assert "1 vs 2" not in markdown
    assert "attacking uses opponent strength_defence" in markdown
    assert "defensive/clean-sheet uses opponent strength_attack" in markdown
    assert "not xG or probability" in markdown
    assert "3 players from ARS" in markdown
    assert "maximum club allocation reached" in markdown
    assert "status partial" in markdown
    assert "some FACT data may be stale" in markdown
    conn.close()


def test_structural_limit_is_data_gap_without_target_squad_even_with_same_club_watchlist(tmp_path):
    config, conn = _structural_setup(tmp_path, {1: 1, 2: 1, 3: 1})
    with conn:
        for player_id in (1, 2, 3):
            repo.add_watchlist(conn, player_id, "WATCH", "same-club test")

    risks = build_report(conn, config, 3)["sections"]["RISKS"]["lines"]
    assert "[DERIVED] Structural: DATA GAP — verified complete current squad unavailable; club concentration not evaluated." in risks
    assert not any("no club has reached the three-player maximum" in line for line in risks)
    conn.close()


def test_structural_limit_is_data_gap_for_partial_target_squad(tmp_path):
    mapping = {player_id: 1 if player_id <= 3 else 2 for player_id in range(1, 16)}
    config, conn = _structural_setup(tmp_path, mapping)
    with conn:
        repo.upsert_squad_picks(
            conn,
            99,
            3,
            [PickRecord(player_id=player_id, position=player_id, raw_json={}) for player_id in range(1, 15)],
        )

    risks = build_report(conn, config, 3)["sections"]["RISKS"]["lines"]
    assert "[DERIVED] Structural: DATA GAP — verified complete current squad unavailable; club concentration not evaluated." in risks
    conn.close()


def test_structural_limit_preserves_concentration_warning_for_complete_target_squad(tmp_path):
    mapping = {player_id: 1 if player_id <= 3 else 2 for player_id in range(1, 16)}
    config, conn = _structural_setup(tmp_path, mapping)
    with conn:
        repo.upsert_squad_picks(
            conn,
            99,
            3,
            [PickRecord(player_id=player_id, position=player_id, raw_json={}) for player_id in range(1, 16)],
        )

    risks = build_report(conn, config, 3)["sections"]["RISKS"]["lines"]
    assert any("3 players from T1" in line and "maximum club allocation reached" in line for line in risks)
    assert not any("Structural: DATA GAP" in line for line in risks)
    conn.close()


def test_structural_limit_does_not_reuse_complete_previous_gameweek_squad(tmp_path):
    mapping = {player_id: 1 if player_id <= 3 else 2 for player_id in range(1, 16)}
    config, conn = _structural_setup(tmp_path, mapping, event_ids=(2, 3))
    with conn:
        repo.upsert_squad_picks(
            conn,
            99,
            2,
            [PickRecord(player_id=player_id, position=player_id, raw_json={}) for player_id in range(1, 16)],
        )

    risks = build_report(conn, config, 3)["sections"]["RISKS"]["lines"]
    assert "[DERIVED] Structural: DATA GAP — verified complete current squad unavailable; club concentration not evaluated." in risks
    assert not any("Concentration risk:" in line for line in risks)
    conn.close()


def test_structural_limit_ignores_watchlist_and_report_player_universe(tmp_path):
    mapping = {
        1: 1,
        2: 1,
        3: 2,
        4: 2,
        5: 3,
        6: 3,
        7: 4,
        8: 4,
        9: 5,
        10: 5,
        11: 6,
        12: 6,
        13: 7,
        14: 7,
        15: 8,
        16: 1,
        17: 1,
        18: 1,
    }
    config, conn = _structural_setup(tmp_path, mapping)
    config["report"]["include_all_players"] = True
    with conn:
        repo.upsert_squad_picks(
            conn,
            99,
            3,
            [PickRecord(player_id=player_id, position=player_id, raw_json={}) for player_id in range(1, 16)],
        )
        for player_id in (16, 17, 18):
            repo.add_watchlist(conn, player_id, "WATCH", "outside-squad test")

    risks = build_report(conn, config, 3)["sections"]["RISKS"]["lines"]
    assert "[DERIVED] Structural: no club has reached the three-player maximum." in risks
    assert not any("Concentration risk:" in line for line in risks)
    conn.close()


def test_structural_limit_keeps_no_club_message_for_complete_squad_without_three(tmp_path):
    mapping = {
        1: 1,
        2: 1,
        3: 2,
        4: 2,
        5: 3,
        6: 3,
        7: 4,
        8: 4,
        9: 5,
        10: 5,
        11: 6,
        12: 6,
        13: 7,
        14: 7,
        15: 8,
    }
    config, conn = _structural_setup(tmp_path, mapping)
    with conn:
        repo.upsert_squad_picks(
            conn,
            99,
            3,
            [PickRecord(player_id=player_id, position=player_id, raw_json={}) for player_id in range(1, 16)],
        )

    risks = build_report(conn, config, 3)["sections"]["RISKS"]["lines"]
    assert "[DERIVED] Structural: no club has reached the three-player maximum." in risks
    conn.close()


def test_report_future_horizons_exclude_finalized_gameweek_and_use_future_fdr(tmp_path):
    config = _config(tmp_path)
    config["report"]["include_all_players"] = True
    conn = connect_database(config["paths"]["database"])
    with conn:
        repo.upsert_events(conn, [EventRecord(id=2, name="Gameweek 2", finished=1, data_checked=1)])
        repo.upsert_teams(
            conn,
            [
                TeamRecord(id=1, name="Arsenal", short_name="ARS"),
                TeamRecord(id=2, name="Aston Villa", short_name="AVL"),
                TeamRecord(id=3, name="Chelsea", short_name="CHE"),
                TeamRecord(id=4, name="Sunderland", short_name="SUN"),
            ],
        )
        repo.upsert_positions(conn, [PositionRecord(id=1, singular_name_short="GKP")])
        repo.upsert_players(conn, [PlayerRecord(id=1, web_name="Raya", full_name="David Raya", team_id=1, element_type=1)])
        repo.upsert_fixtures(
            conn,
            [
                FixtureRecord(id=40, event=2, team_h=2, team_a=1, team_a_difficulty=4, started=1, finished=1, raw_json={}),
                FixtureRecord(id=41, event=3, team_h=1, team_a=3, team_h_difficulty=4, started=0, finished=0, raw_json={}),
                FixtureRecord(id=42, event=4, team_h=4, team_a=1, team_a_difficulty=3, started=0, finished=0, raw_json={}),
            ],
        )
    markdown = render_markdown(build_report(conn, config, 2))
    arsenal_line = next(line for line in markdown.splitlines() if "Arsenal next 8:" in line)
    assert arsenal_line.startswith("  Arsenal next 8: CHE(H, 4), SUN(A, 3)")
    assert "AVL" not in arsenal_line
    assert "FDR next horizons: 3.5 / 3.5 / 3.5" in markdown
    conn.close()


def test_report_flags_unplaced_pending_fixture_excluded_from_future_fdr(tmp_path):
    config = _config(tmp_path)
    config["report"]["include_all_players"] = True
    conn = connect_database(config["paths"]["database"])
    with conn:
        repo.upsert_events(
            conn,
            [
                EventRecord(id=2, name="Gameweek 2", finished=1, data_checked=1),
                EventRecord(id=3, name="Gameweek 3", deadline_time="2026-09-01T00:00:00Z"),
            ],
        )
        repo.upsert_teams(conn, [TeamRecord(id=1, name="Arsenal", short_name="ARS"), TeamRecord(id=2, name="Chelsea", short_name="CHE")])
        repo.upsert_positions(conn, [PositionRecord(id=1, singular_name_short="GKP")])
        repo.upsert_players(conn, [PlayerRecord(id=1, web_name="Raya", full_name="David Raya", team_id=1, element_type=1)])
        repo.upsert_fixtures(
            conn,
            [
                FixtureRecord(id=80, event=1, team_h=1, team_a=2, team_h_difficulty=5, started=0, finished=0, raw_json={}),
                FixtureRecord(id=81, event=3, team_h=1, team_a=2, team_h_difficulty=3, kickoff_time="2026-09-02T12:00:00Z", started=0, finished=0, raw_json={}),
            ],
        )

    markdown = render_markdown(build_report(conn, config, 2))
    assert "Pending fixture schedule unresolved; excluded from future FDR horizons" in markdown
    assert "fixture 80 (official event GW1; kickoff unknown)" in markdown
    conn.close()


def test_fetch_health_warns_only_for_running_or_failed_latest_fetch(tmp_path):
    config = _config(tmp_path)
    conn = connect_database(config["paths"]["database"])
    with conn:
        running = repo.create_fetch_run(conn, "fetch_fpl")
    running_report = render_markdown(build_report(conn, config, 1))
    assert f"Latest fetch_fpl run {running}: status running" in running_report
    assert "Latest fetch_fpl run is running" in running_report

    with conn:
        repo.finish_fetch_run(conn, running, "success", 1, ["bootstrap-static", "fixtures"])
        # A newer run from another trigger must not displace the latest
        # fetch_fpl record selected by the report.
        repo.create_fetch_run(conn, "sync_manager")
    completed_report = render_markdown(build_report(conn, config, 1))
    assert f"Latest fetch_fpl run {running}: status success" in completed_report
    assert "Latest fetch_fpl run is running" not in completed_report
    assert "Latest fetch_fpl run is success" not in completed_report

    with conn:
        failed = repo.create_fetch_run(conn, "fetch_fpl")
        repo.finish_fetch_run(
            conn,
            failed,
            "failed",
            1,
            ["bootstrap-static"],
            [{"endpoint": "fixtures", "error": "fixture request failed"}],
            "fixture request failed",
        )
    failed_report = render_markdown(build_report(conn, config, 1))
    assert f"Latest fetch_fpl run {failed}: status failed" in failed_report
    assert "Latest fetch_fpl run is failed; some FACT data may be stale." in failed_report
    conn.close()


def test_report_formats_percentage_point_deltas_without_binary_float_noise(tmp_path):
    config = _config(tmp_path)
    config["report"]["include_all_players"] = True
    conn = connect_database(config["paths"]["database"])
    with conn:
        repo.upsert_players(conn, [PlayerRecord(id=1, web_name="Player", full_name="Player")])
        run1 = repo.create_fetch_run(conn, "fetch_fpl")
        repo.insert_snapshots(conn, [PlayerSnapshotRecord(player_id=1, captured_at="2026-08-01T00:00:00Z", selected_by_percent=34.7, now_cost=80, raw_json={})], run1)
        run2 = repo.create_fetch_run(conn, "fetch_fpl")
        repo.insert_snapshots(conn, [PlayerSnapshotRecord(player_id=1, captured_at="2026-08-10T00:00:00Z", selected_by_percent=38.0, now_cost=80, raw_json={})], run2)
    markdown = render_markdown(build_report(conn, config, 1))
    assert "(3.3pp)" in markdown
    assert "3.299999" not in markdown
    conn.close()


def test_report_renders_event_scoped_manual_state_and_separate_prices(tmp_path):
    config = _config(tmp_path)
    config["fpl_entry_id"] = 241392
    conn = connect_database(config["paths"]["database"])
    with conn:
        repo.upsert_events(conn, [EventRecord(id=4, name="Gameweek 4")])
        repo.upsert_players(
            conn,
            [
                PlayerRecord(id=1, web_name="Calafiori", full_name="Riccardo Calafiori"),
                PlayerRecord(id=2, web_name="Joao Pedro", full_name="Joao Pedro"),
            ],
        )
        run = repo.create_fetch_run(conn, "fetch_fpl")
        repo.insert_snapshots(
            conn,
            [
                PlayerSnapshotRecord(player_id=1, captured_at="2026-09-07T00:00:00Z", now_cost=57, raw_json={}),
                PlayerSnapshotRecord(player_id=2, captured_at="2026-09-07T00:00:00Z", now_cost=77, raw_json={}),
            ],
            run,
        )
        repo.upsert_manual_manager_state(conn, 241392, 4, 3, 0, captured_at="2026-09-07T12:00:00Z")
        repo.upsert_manager_selling_prices(
            conn,
            241392,
            4,
            {1: 56, 2: 76},
            captured_at="2026-09-07T12:00:00Z",
        )

    markdown = render_markdown(build_report(conn, config, 4))
    assert "Manual manager state: GW4 | free transfers 3 | bank" in markdown
    assert "selling-value snapshot" in markdown
    assert "Riccardo Calafiori: official market" in markdown
    assert "manager selling" in markdown
    assert "5.7m" in markdown
    assert "5.6m" in markdown
    assert "7.7m" in markdown
    assert "7.6m" in markdown
    assert repo.latest_snapshot(conn, 1)["now_cost"] == 57
    assert repo.latest_snapshot(conn, 2)["now_cost"] == 77
    conn.close()


def test_report_renders_self_adjusting_manager_value_state(tmp_path):
    config = _config(tmp_path)
    config["fpl_entry_id"] = 99
    conn = connect_database(config["paths"]["database"])
    with conn:
        repo.upsert_events(conn, [EventRecord(id=2, name="Gameweek 2")])
        repo.upsert_positions(conn, [PositionRecord(id=1, singular_name_short="GKP")])
        repo.upsert_players(
            conn,
            [PlayerRecord(id=player_id, web_name=f"P{player_id}", full_name=f"Player {player_id}", element_type=1) for player_id in range(1, 16)],
        )
        run = repo.create_fetch_run(conn, "fetch_fpl")
        repo.insert_snapshots(
            conn,
            [PlayerSnapshotRecord(player_id=player_id, captured_at="2026-09-07T00:00:00Z", now_cost=55, raw_json={}) for player_id in range(1, 16)],
            run,
        )
        repo.insert_manager_state(
            conn,
            EntryRecord(entry_id=99, player_name="Manager", team_name="Team", bank=0, team_value=825, total_transfers=0),
            HistoryRow(event=2, bank=0, value=825, total_transfers=0),
            None,
            3,
            None,
            2,
            {"transfers": [], "transfers_endpoint_available": True},
            "2026-09-07T01:00:00Z",
        )
        repo.upsert_squad_picks(conn, 99, 2, [PickRecord(player_id=player_id, position=player_id, raw_json={}) for player_id in range(1, 16)])
        for player_id in range(1, 16):
            repo.insert_manager_acquisition(conn, 99, player_id, 1, 55, source="verified_initial_squad")
        repo.upsert_manual_manager_state(conn, 99, 2, 3, 0, captured_at="2026-09-07T02:00:00Z")
        repo.upsert_manager_selling_prices(conn, 99, 2, {player_id: 55 for player_id in range(1, 16)}, market_prices_at_capture={player_id: 55 for player_id in range(1, 16)})
    markdown = render_markdown(build_report(conn, config, 2))
    assert "Manager value state: OK" in markdown
    assert "Official squad market value: £82.5m | Realisable selling value: £82.5m" in markdown
    assert "purchase £5.5m | market £5.5m | calculated sell £5.5m | manual snapshot £5.5m | effective sell £5.5m" in markdown
    conn.close()
