from __future__ import annotations

import sqlite3

import pytest

from fpl_brain import repositories as repo
from fpl_brain.database import connect_database
from fpl_brain.models import PlayerRecord, PlayerSnapshotRecord, PositionRecord, TeamRecord


def test_snapshots_are_append_only_per_run(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    with conn:
        repo.upsert_teams(conn, [TeamRecord(id=1, name="One")])
        repo.upsert_positions(conn, [PositionRecord(id=1)])
        repo.upsert_players(conn, [PlayerRecord(id=1, web_name="One", full_name="One", team_id=1, element_type=1)])
        run1 = repo.create_fetch_run(conn, "fetch_fpl")
        snapshot = PlayerSnapshotRecord(player_id=1, captured_at="2026-08-19T00:00:00Z", now_cost=80, raw_json={})
        repo.insert_snapshots(conn, [snapshot], run1)
        run2 = repo.create_fetch_run(conn, "fetch_fpl")
        repo.insert_snapshots(conn, [snapshot], run2)
    rows = conn.execute("SELECT id,fetch_run_id FROM player_snapshots ORDER BY id").fetchall()
    assert len(rows) == 2
    ids = [row[0] for row in rows]
    with pytest.raises(sqlite3.IntegrityError):
        with conn:
            repo.insert_snapshots(conn, [snapshot], run2)
    assert [row[0] for row in conn.execute("SELECT id FROM player_snapshots ORDER BY id").fetchall()] == ids

    run3 = repo.create_fetch_run(conn, "fetch_fpl")
    with pytest.raises(sqlite3.IntegrityError):
        with conn:
            repo.insert_snapshots(conn, [PlayerSnapshotRecord(player_id=1, captured_at="later", raw_json={}), snapshot], run3)
    assert conn.execute("SELECT COUNT(*) FROM player_snapshots").fetchone()[0] == 2
    conn.close()

