from __future__ import annotations

from fpl_brain import repositories as repo
from fpl_brain.database import connect_database
import sqlite3

import pytest

from fpl_brain.models import FixtureRecord, PickRecord, PlayerGameweekRecord, PlayerRecord, PlayerSnapshotRecord, PositionRecord, TeamRecord


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


def test_manual_manager_state_and_selling_prices_are_event_and_entry_scoped(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    with conn:
        repo.upsert_players(
            conn,
            [
                PlayerRecord(id=10, web_name="Maguire", full_name="Harry Maguire"),
                PlayerRecord(id=11, web_name="Rodon", full_name="Joe Rodon"),
            ],
        )
        run = repo.create_fetch_run(conn, "fetch_fpl")
        repo.insert_snapshots(
            conn,
            [
                PlayerSnapshotRecord(player_id=10, captured_at="2026-09-07T00:00:00Z", now_cost=51, raw_json={}),
                PlayerSnapshotRecord(player_id=11, captured_at="2026-09-07T00:00:00Z", now_cost=45, raw_json={}),
            ],
            run,
        )
        state = repo.upsert_manual_manager_state(
            conn,
            241392,
            4,
            3,
            0,
            captured_at="2026-09-07T12:00:00Z",
        )
        assert state == {
            "entry_id": 241392,
            "event": 4,
            "free_transfers": 3,
            "bank": 0,
            "source": "manual",
            "captured_at": "2026-09-07T12:00:00Z",
            "event_start_free_transfers": None,
        }
        # The event-start FT bank is explicit provenance for the Wildcard/Free-Hit
        # FT transition; it is stored and returned verbatim (kept on a separate
        # event so this test's other scope assertions stay independent).
        scoped = repo.upsert_manual_manager_state(
            conn, 241392, 5, 0, 7, source="user_confirmed_override",
            captured_at="2026-09-07T13:00:00Z", event_start_free_transfers=2,
        )
        assert scoped["event_start_free_transfers"] == 2
        assert scoped["free_transfers"] == 0
        assert repo.upsert_manager_selling_prices(
            conn,
            241392,
            4,
            {10: 50, 11: 44},
            captured_at="2026-09-07T12:00:00Z",
        ) == 2
        repo.upsert_manual_manager_state(conn, 241392, 3, 2, 7, captured_at="2026-09-06T12:00:00Z")
        repo.upsert_manager_selling_prices(conn, 999, 4, {10: 99}, captured_at="2026-09-07T12:00:00Z")

    assert repo.get_manual_manager_state(conn, 241392, 4)["free_transfers"] == 3
    assert repo.get_manual_manager_state(conn, 241392, 3)["bank"] == 7
    assert repo.get_manual_manager_state(conn, 999, 4) is None
    assert [row["selling_price"] for row in repo.get_manager_selling_prices(conn, 241392, 4)] == [50, 44]
    assert repo.get_manager_selling_price(conn, 241392, 4, 10)["selling_price"] == 50
    assert repo.get_manager_selling_price(conn, 241392, 3, 10) is None
    assert repo.get_manager_selling_price(conn, 999, 4, 10)["selling_price"] == 99
    assert repo.latest_snapshot(conn, 10)["now_cost"] == 51
    # events 4, 3 and the explicit event-start-FT row on event 5
    assert repo.count_rows(conn, "manager_manual_state") == 3
    assert repo.count_rows(conn, "manager_selling_prices") == 3
    assert repo.count_rows(conn, "manager_player_acquisitions") == 0
    conn.close()


def test_acquisition_ledger_preserves_stints_and_only_allows_one_active_row(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    with conn:
        repo.upsert_players(conn, [PlayerRecord(id=10, web_name="Player", full_name="Player")])
        first = repo.insert_manager_acquisition(
            conn, 99, 10, 1, 55, source="verified_initial_squad", created_at="2026-08-20T00:00:00Z"
        )
        with pytest.raises(sqlite3.IntegrityError):
            repo.insert_manager_acquisition(conn, 99, 10, 2, 60, source="official_transfer_history")
        repo.close_manager_acquisition(conn, first, 10, "2026-10-01T00:00:00Z")
        second = repo.insert_manager_acquisition(
            conn, 99, 10, 16, 78, source="official_transfer_history", acquired_at="2026-12-01T00:00:00Z"
        )
    rows = repo.list_manager_acquisitions(conn, 99, 10)
    assert len(rows) == 2
    assert rows[0]["sold_event"] == 10
    assert rows[1]["id"] == second and rows[1]["sold_event"] is None
    assert repo.active_manager_acquisitions(conn, 99)[0]["purchase_price"] == 78
    conn.close()


def test_reconcile_transfer_history_is_idempotent_and_supports_second_stint(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    with conn:
        repo.upsert_positions(conn, [PositionRecord(id=1, singular_name_short="GKP")])
        repo.upsert_players(
            conn,
            [PlayerRecord(id=player_id, web_name=f"P{player_id}", full_name=f"Player {player_id}", element_type=1) for player_id in range(1, 17)],
        )
        run = repo.create_fetch_run(conn, "fetch_fpl")
        repo.insert_snapshots(
            conn,
            [PlayerSnapshotRecord(player_id=player_id, captured_at="2026-08-20T00:00:00Z", now_cost=55, cost_change_start=0, raw_json={}) for player_id in range(1, 17)],
            run,
        )
        repo.upsert_squad_picks(conn, 99, 1, [PickRecord(player_id=player_id, position=player_id, raw_json={}) for player_id in range(1, 16)])
        repo.upsert_squad_picks(conn, 99, 2, [PickRecord(player_id=16 if player_id == 1 else player_id, position=player_id, raw_json={}) for player_id in range(1, 16)])
        first_transfer = [{"entry": 99, "element_in": 16, "element_out": 1, "event": 2, "time": "2026-09-01T10:00:00Z", "element_in_cost": 55, "element_out_cost": 55}]
        result = repo.reconcile_manager_acquisitions(conn, 99, 2, first_transfer, True, captured_at="2026-09-01T12:00:00Z")
        assert result["status"] == "success"
        repeated = repo.reconcile_manager_acquisitions(conn, 99, 2, first_transfer, True, captured_at="2026-09-01T13:00:00Z")
        assert repeated["status"] == "success"
        repo.upsert_squad_picks(conn, 99, 3, [PickRecord(player_id=1 if player_id == 2 else (16 if player_id == 1 else player_id), position=player_id, raw_json={}) for player_id in range(1, 16)])
        second_transfer = first_transfer + [{"entry": 99, "element_in": 1, "element_out": 2, "event": 3, "time": "2026-09-08T10:00:00Z", "element_in_cost": 60, "element_out_cost": 55}]
        result = repo.reconcile_manager_acquisitions(conn, 99, 3, second_transfer, True, captured_at="2026-09-08T12:00:00Z")
        assert result["status"] == "success"
    player_one = repo.list_manager_acquisitions(conn, 99, 1)
    assert [(row["acquired_event"], row["purchase_price"], row["sold_event"]) for row in player_one] == [(1, 55, 2), (3, 60, None)]
    assert len(repo.active_manager_acquisitions(conn, 99)) == 15
    conn.close()
