from __future__ import annotations

import copy
import json

from fpl_brain import repositories as repo
from fpl_brain.config import DEFAULT_CONFIG
from fpl_brain.database import connect_database
from fpl_brain.models import EntryRecord, EventRecord, FixtureRecord, PickRecord, PlayerRecord, PlayerSnapshotRecord, PositionRecord, TeamRecord
from fpl_brain.reports import build_report, render_json, render_markdown


def _config(tmp_path):
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["paths"]["database"] = str(tmp_path / "fpl.db")
    config["paths"]["raw_dir"] = str(tmp_path / "raw")
    config["paths"]["exports_dir"] = str(tmp_path / "exports")
    return config


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
