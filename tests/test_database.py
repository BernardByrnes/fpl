from __future__ import annotations

import sqlite3

import pytest

import fpl_brain.database as database


def test_schema_is_idempotent_and_enforces_pragmas(tmp_path):
    path = tmp_path / "fpl.db"
    conn = database.connect_database(path)
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    assert conn.execute("SELECT value FROM schema_meta WHERE key='schema_version'").fetchone()[0] == "1"
    conn.close()
    conn = database.connect_database(path)
    assert conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'").fetchone()[0] >= 18
    conn.close()


def test_additive_migration_bumps_once(tmp_path):
    path = tmp_path / "fpl.db"
    conn = database.connect_database(path)

    def migration_two(connection):
        connection.execute("CREATE TABLE IF NOT EXISTS migration_probe (id INTEGER PRIMARY KEY)")

    database.MIGRATIONS.append(migration_two)
    try:
        database.initialize_database(conn)
        assert conn.execute("SELECT value FROM schema_meta WHERE key='schema_version'").fetchone()[0] == "2"
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

