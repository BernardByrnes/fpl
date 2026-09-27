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


def test_schema_19_adds_immutable_snapshot_provenance_without_backfill(tmp_path):
    path = tmp_path / "schema18.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute("INSERT INTO schema_meta(key,value) VALUES ('schema_version','0')")
    for version, migration in enumerate(database.MIGRATIONS[:-1], start=1):
        with conn:
            migration(conn)
            conn.execute(
                "UPDATE schema_meta SET value=? WHERE key='schema_version'", (str(version),)
            )
    with conn:
        cursor = conn.execute(
            "INSERT INTO projection_runs(model_family,model_version,generated_at,planning_event,"
            "data_cutoff,status) VALUES ('minutes_v1','legacy','2026-09-19T11:00:00Z',5,"
            "'2026-09-19T11:00:00Z','complete')"
        )
        legacy_id = int(cursor.lastrowid)

    database.initialize_database(conn)
    columns = {row[1] for row in conn.execute("PRAGMA table_info(projection_runs)")}
    assert {
        "data_snapshot_sha256", "execution_run_uuid", "planning_context_inputs_json",
    } <= columns
    assert conn.execute(
        "SELECT data_snapshot_sha256,execution_run_uuid,planning_context_inputs_json "
        "FROM projection_runs WHERE id=?", (legacy_id,),
    ).fetchone() == (None, None, None)
    assert int(conn.execute(
        "SELECT value FROM schema_meta WHERE key='schema_version'"
    ).fetchone()[0]) == database.SCHEMA_VERSION
    with pytest.raises(sqlite3.IntegrityError, match="PE9_PROVENANCE_IMMUTABLE"):
        conn.execute(
            "UPDATE projection_runs SET data_snapshot_sha256=? WHERE id=?",
            ("a" * 64, legacy_id),
        )
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
