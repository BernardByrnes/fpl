from __future__ import annotations

import sqlite3

import pytest

import fpl_brain.database as database


def test_schema_is_idempotent_and_enforces_pragmas(tmp_path):
    path = tmp_path / "fpl.db"
    conn = database.connect_database(path)
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    assert conn.execute("SELECT value FROM schema_meta WHERE key='schema_version'").fetchone()[0] == str(database.SCHEMA_VERSION)
    assert conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='manager_state_observations'").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='manager_selling_price_observations'").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='manager_manual_state'").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='manager_selling_prices'").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='manager_player_acquisitions'").fetchone()[0] == 1
    columns = {row["name"] for row in conn.execute("PRAGMA table_info(manager_selling_prices)")}
    assert "market_price_at_capture" in columns
    conn.close()
    conn = database.connect_database(path)
    assert conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'").fetchone()[0] >= 18
    conn.close()


def test_additive_migration_bumps_once(tmp_path):
    path = tmp_path / "fpl.db"
    conn = database.connect_database(path)
    baseline_version = int(conn.execute("SELECT value FROM schema_meta WHERE key='schema_version'").fetchone()[0])

    def migration_two(connection):
        connection.execute("CREATE TABLE IF NOT EXISTS migration_probe (id INTEGER PRIMARY KEY)")

    database.MIGRATIONS.append(migration_two)
    try:
        database.initialize_database(conn)
        assert conn.execute("SELECT value FROM schema_meta WHERE key='schema_version'").fetchone()[0] == str(baseline_version + 1)
        database.initialize_database(conn)
        assert conn.execute("SELECT COUNT(*) FROM migration_probe").fetchone()[0] == 0
    finally:
        database.MIGRATIONS.pop()
        conn.close()


def test_foreign_keys_are_enforced(tmp_path):
    conn = database.connect_database(tmp_path / "fpl.db")
    with pytest.raises(sqlite3.IntegrityError):
        with conn:
            conn.execute(
                "INSERT INTO players(id,web_name,team_id,first_seen_at,last_seen_at,raw_json,updated_at) VALUES (?,?,?,?,?,?,?)",
                (1, "Missing team", 999, "now", "now", "{}", "now"),
            )
    conn.close()


def test_phase2_projection_tables_and_widened_run_family(tmp_path):
    conn = database.connect_database(tmp_path / "fpl.db")
    for table in ("team_fixture_projections", "player_rate_projections"):
        assert conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()[0] == 1
    # The widened CHECK admits the new families and still rejects unknown ones.
    with conn:
        conn.execute(
            """INSERT INTO projection_runs(model_family,model_version,generated_at,planning_event,data_cutoff,status)
               VALUES ('team_strength_v1','t','2026-09-11T00:00:00Z',1,'2026-09-11T00:00:00Z','running')"""
        )
    with pytest.raises(sqlite3.IntegrityError):
        with conn:
            conn.execute(
                """INSERT INTO projection_runs(model_family,model_version,generated_at,planning_event,data_cutoff,status)
                   VALUES ('not_a_family','t','2026-09-11T00:00:00Z',1,'2026-09-11T00:00:00Z','running')"""
            )
    triggers = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='trigger'")}
    assert {
        "team_fixture_projections_no_update",
        "team_fixture_projections_no_delete",
        "player_rate_projections_no_update",
        "player_rate_projections_no_delete",
    } <= triggers
    conn.close()
