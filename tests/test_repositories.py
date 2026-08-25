from __future__ import annotations

from fpl_brain import repositories as repo
from fpl_brain.database import connect_database
import sqlite3

import pytest

from fpl_brain.models import FixtureRecord, PickRecord, PlayerGameweekRecord, PlayerRecord, PositionRecord, TeamRecord


def _seed(conn):
    with conn:
        repo.upsert_teams(conn, [TeamRecord(id=1, name="One"), TeamRecord(id=2, name="Two")])
        repo.upsert_positions(conn, [PositionRecord(id=1, singular_name_short="GKP")])
        repo.upsert_players(
            conn,
            [PlayerRecord(id=10, web_name="Old", full_name="Old Name", norm_name="old name", team_id=1, element_type=1)],
            "2026-08-19T00:00:00Z",
        )


def test_identity_upsert_absence_and_fixture_gameweek_upserts(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    _seed(conn)
    with conn:
        repo.upsert_players(
            conn,
            [PlayerRecord(id=10, web_name="New", full_name="New Name", norm_name="new name", team_id=2, element_type=1)],
            "2026-08-20T00:00:00Z",
        )
        repo.mark_absent_players(conn, set())
    player = repo.get_player(conn, 10)
    assert player["web_name"] == "New"
    assert player["team_id"] == 2
    assert player["is_active"] == 1

    fixture = FixtureRecord(id=1, team_h=1, team_a=2, event=1, team_h_score=0, team_a_score=0, raw_json={})
    with conn:
        repo.upsert_fixtures(conn, [fixture])
        repo.upsert_fixtures(conn, [FixtureRecord(id=1, team_h=1, team_a=2, event=1, team_h_score=2, team_a_score=1, raw_json={})])
    assert conn.execute("SELECT team_h_score FROM fixtures WHERE id=1").fetchone()[0] == 2

    with conn:
        repo.upsert_player_gameweeks(
            conn,
            [PlayerGameweekRecord(player_id=10, event=1, fixture_id=1, minutes=90, total_points=6, raw_json={})],
        )
        repo.upsert_player_gameweeks(
            conn,
            [PlayerGameweekRecord(player_id=10, event=1, fixture_id=1, minutes=90, total_points=8, raw_json={})],
        )
    assert conn.execute("SELECT COUNT(*) FROM player_gameweeks").fetchone()[0] == 1
    assert conn.execute("SELECT total_points FROM player_gameweeks").fetchone()[0] == 8
    conn.close()


def test_gameweek_source_precedence_preserves_performance_and_reconciles_sentinel(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    _seed(conn)
    with conn:
        repo.upsert_player_gameweeks(
            conn,
            [PlayerGameweekRecord(player_id=10, event=1, fixture_id=1, minutes=90, total_points=6, source="event_live", raw_json={"live": 1})],
        )
        # A future schedule row can add context but cannot blank performance.
        repo.upsert_player_gameweeks(
            conn,
            [PlayerGameweekRecord(player_id=10, event=1, fixture_id=1, minutes=0, kickoff_time="2026-08-20T15:00:00Z", source="element_summary", raw_json={"schedule": 1})],
        )
        row = conn.execute("SELECT * FROM player_gameweeks WHERE player_id=10 AND event=1 AND fixture_id=1").fetchone()
        assert row["minutes"] == 90 and row["total_points"] == 6
        assert row["kickoff_time"] == "2026-08-20T15:00:00Z"

        # A completed summary can enrich/replace the live observation.
        repo.upsert_player_gameweeks(
            conn,
            [PlayerGameweekRecord(player_id=10, event=1, fixture_id=1, minutes=80, total_points=5, source="element_summary", raw_json={"summary": 1})],
        )
        row = conn.execute("SELECT * FROM player_gameweeks WHERE player_id=10 AND event=1 AND fixture_id=1").fetchone()
        assert row["minutes"] == 80 and row["total_points"] == 5

        # Later live data remains an explicit performance observation and is idempotent.
        repo.upsert_player_gameweeks(
            conn,
            [PlayerGameweekRecord(player_id=10, event=1, fixture_id=1, minutes=90, total_points=7, source="event_live", raw_json={"live": 2})],
        )
        repo.upsert_player_gameweeks(
            conn,
            [PlayerGameweekRecord(player_id=10, event=1, fixture_id=-1, minutes=90, total_points=7, source="event_live", raw_json={"sentinel": 1})],
        )
        repo.upsert_player_gameweeks(
            conn,
            [PlayerGameweekRecord(player_id=10, event=1, fixture_id=2, minutes=90, total_points=8, source="element_summary", raw_json={"completed": 1})],
        )
        assert conn.execute("SELECT COUNT(*) FROM player_gameweeks WHERE player_id=10 AND event=1 AND fixture_id=-1").fetchone()[0] == 0
        count_before = conn.execute("SELECT COUNT(*) FROM player_gameweeks").fetchone()[0]
        repo.upsert_player_gameweeks(
            conn,
            [PlayerGameweekRecord(player_id=10, event=1, fixture_id=2, minutes=90, total_points=8, source="element_summary", raw_json={"completed": 1})],
        )
        assert conn.execute("SELECT COUNT(*) FROM player_gameweeks").fetchone()[0] == count_before
    conn.close()


def test_squad_picks_are_atomic_exact_event_roster_replacements(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    with conn:
        repo.upsert_teams(conn, [TeamRecord(id=1, name="One")])
        repo.upsert_positions(conn, [PositionRecord(id=1, singular_name_short="GKP")])
        repo.upsert_players(
            conn,
            [PlayerRecord(id=player_id, web_name=f"P{player_id}", team_id=1, element_type=1) for player_id in range(1, 33)],
        )
        original = [PickRecord(player_id=player_id, position=player_id, raw_json={}) for player_id in range(1, 16)]
        other_event = [PickRecord(player_id=player_id, position=player_id, raw_json={}) for player_id in range(16, 31)]
        repo.upsert_squad_picks(conn, 99, 1, original)
        repo.upsert_squad_picks(conn, 99, 2, other_event)
        replacement = [PickRecord(player_id=player_id, position=player_id, raw_json={}) for player_id in range(2, 17)]
        assert repo.upsert_squad_picks(conn, 99, 1, replacement) == 15
        current = {row[0] for row in conn.execute("SELECT player_id FROM squad_picks WHERE entry_id=99 AND event=1")}
        assert current == set(range(2, 17))
        assert conn.execute("SELECT COUNT(*) FROM squad_picks WHERE entry_id=99 AND event=1").fetchone()[0] == 15
        assert conn.execute("SELECT COUNT(*) FROM squad_picks WHERE entry_id=99 AND event=2").fetchone()[0] == 15
        with pytest.raises(sqlite3.IntegrityError):
            repo.upsert_squad_picks(conn, 99, 1, [PickRecord(player_id=999, position=1, raw_json={})])
        retained = {row[0] for row in conn.execute("SELECT player_id FROM squad_picks WHERE entry_id=99 AND event=1")}
        assert retained == set(range(2, 17))
    conn.close()


def test_scouting_current_tie_break_is_single_and_deterministic(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    with conn:
        repo.upsert_teams(conn, [TeamRecord(id=1, name="One")])
        repo.upsert_players(conn, [PlayerRecord(id=10, web_name="Player", full_name="Player", team_id=1)])
        import_id = repo.insert_scouting_import(conn, {"source_file": "tie", "file_sha256": "tie", "players_total": 1, "players_resolved": 1, "notes_inserted": 2})
        for value in ("old", "new"):
            repo.insert_scouting_note(
                conn,
                {
                    "import_id": import_id,
                    "player_id": 10,
                    "key": "likely_role",
                    "value_text": value,
                    "confidence": "medium",
                    "observed_at": "2026-08-19T00:00:00Z",
                },
            )
    rows = repo.scouting_current_rows(conn, [10])
    assert len(rows) == 1
    assert rows[0]["value_text"] == "new"
    conn.close()
