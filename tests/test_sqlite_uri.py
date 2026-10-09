from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from fpl_brain.database import connect_readonly_database
from fpl_brain.sqlite_uri import readonly_sqlite_uri


def test_readonly_sqlite_uri_escapes_path_and_opens_the_exact_database(tmp_path):
    target = tmp_path / "source#fragment%name.sqlite"
    with sqlite3.connect(target) as writer:
        writer.execute("CREATE TABLE identity (value TEXT NOT NULL)")
        writer.execute("INSERT INTO identity VALUES ('intended-db')")

    uri = readonly_sqlite_uri(target)
    assert "%23" in uri
    assert "%25" in uri
    assert uri.endswith("?mode=ro")

    # Exercise mode=ro itself: query_only on the application factory must not
    # be what makes a missing URI read-only.
    conn = sqlite3.connect(uri, uri=True)
    try:
        assert Path(next(row[2] for row in conn.execute("PRAGMA database_list") if row[1] == "main")).resolve() == target.resolve()
        assert conn.execute("SELECT value FROM identity").fetchone()[0] == "intended-db"
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("INSERT INTO identity VALUES ('must-not-write')")
    finally:
        conn.close()

    # The production factory still layers query_only over the URI guarantee.
    conn = connect_readonly_database(target)
    try:
        assert conn.execute("SELECT value FROM identity").fetchone()[0] == "intended-db"
    finally:
        conn.close()


def test_readonly_uri_refuses_absent_target_without_creating_it(tmp_path):
    target = tmp_path / "absent#source.sqlite"
    with pytest.raises(sqlite3.OperationalError):
        sqlite3.connect(readonly_sqlite_uri(target), uri=True)
    assert not target.exists()


def test_readonly_factory_preserves_explicit_memory_database_readonly_behavior():
    conn = connect_readonly_database(":memory:")
    try:
        assert conn.execute("PRAGMA query_only").fetchone()[0] == 1
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("CREATE TABLE local_fixture (value INTEGER)")
    finally:
        conn.close()


def test_readonly_sqlite_uri_is_absolute_and_escaped(tmp_path):
    target = tmp_path / "nested#db.sqlite"
    assert readonly_sqlite_uri(target).startswith("file:")
    assert "%23" in readonly_sqlite_uri(target)
