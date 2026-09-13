"""Append-only manager state and selling-price observation semantics."""

from __future__ import annotations

import sqlite3

from fpl_brain import metrics, repositories as repo
from fpl_brain.database import SCHEMA_VERSION, connect_database
from fpl_brain.models import PlayerRecord, PlayerSnapshotRecord


def test_multiple_manager_states_inside_one_event_are_all_preserved(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    # A GW where the manager had 3FT/£0.0m early in the week and then made a
    # real transfer, after which the FPL UI showed 2FT/£0.3m bank.
    with conn:
        first = repo.upsert_manual_manager_state(conn, 241392, 4, 3, 0, captured_at="2026-09-07T13:34:29Z")
        second = repo.upsert_manual_manager_state(conn, 241392, 4, 2, 3, captured_at="2026-09-09T18:00:00Z")
    assert first["free_transfers"] == 3
    assert second["free_transfers"] == 2
    current = repo.get_manual_manager_state(conn, 241392, 4)
    assert current["free_transfers"] == 2 and current["bank"] == 3
    observations = repo.list_manager_state_observations(conn, 241392, 4)
    assert [(row["free_transfers"], row["bank"]) for row in observations] == [(3, 0), (2, 3)]
    # as-of lookup can reconstruct the pre-transfer state.
    earlier = repo.manual_manager_state_as_of(conn, 241392, 4, as_of="2026-09-08T00:00:00Z")
    assert earlier["free_transfers"] == 3 and earlier["bank"] == 0
    assert repo.manual_manager_state_as_of(conn, 241392, 4, as_of="2026-09-06T00:00:00Z") is None
    latest = repo.manual_manager_state_as_of(conn, 241392, 4)
    assert latest["free_transfers"] == 2
    conn.close()


def test_manual_selling_price_observations_keep_history(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    with conn:
        repo.upsert_players(conn, [PlayerRecord(id=8, web_name="De Cuyper", full_name="Maxim De Cuyper")])
        # Manager records the transfer-screen selling price right after the
        # transfer and again inside the same event.
        repo.upsert_manager_selling_prices(
            conn, 241392, 4, {8: 47}, captured_at="2026-09-07T13:34:29Z", market_prices_at_capture={8: 47}
        )
        repo.upsert_manager_selling_prices(
            conn, 241392, 4, {8: 47}, captured_at="2026-09-08T12:00:00Z", market_prices_at_capture={8: 48}
        )
    current = repo.get_manager_selling_price(conn, 241392, 4, 8)
    assert current["selling_price"] == 47 and current["market_price_at_capture"] == 48
    observations = repo.list_manager_selling_price_observations(conn, 241392, 4, 8)
    assert [row["market_price_at_capture"] for row in observations] == [47, 48]
    as_of_rows = repo.list_manager_selling_price_observations(conn, 241392, 4, 8, as_of="2026-09-07T20:00:00Z")
    assert len(as_of_rows) == 1 and as_of_rows[0]["market_price_at_capture"] == 47
    conn.close()


def test_stale_manual_price_never_reapplies_after_market_movement(tmp_path):
    """A manual snapshot captured at an earlier market price cannot freeze the
    automatic selling price after the official price moves."""

    conn = connect_database(tmp_path / "fpl.db")
    with conn:
        repo.upsert_players(conn, [PlayerRecord(id=8, web_name="De Cuyper", full_name="Maxim De Cuyper")])
        run = repo.create_fetch_run(conn, "fetch_fpl")
        # Acquisition purchase price £4.7m (47 tenths), market at purchase 47.
        repo.insert_manager_acquisition(
            conn, 241392, 8, 4, 47, source="official_transfer_history", created_at="2026-09-07T13:00:00Z"
        )
        repo.insert_snapshots(
            conn,
            [PlayerSnapshotRecord(player_id=8, captured_at="2026-09-07T13:00:00Z", now_cost=47, raw_json={})],
            run,
        )
        repo.upsert_manager_selling_prices(
            conn, 241392, 4, {8: 47}, captured_at="2026-09-07T13:34:29Z", market_prices_at_capture={8: 47}
        )
        value = metrics.effective_selling_price(conn, 241392, 4, 8)
        assert value["effective_selling_price"] == 47 and value["status"] == "VERIFIED_BY_MANUAL"

        # Market rises to £4.9m (49): the manual snapshot (captured at 47) is
        # stale; the automatic acquisition formula becomes sell £4.8m.
        run2 = repo.create_fetch_run(conn, "fetch_fpl")
        repo.insert_snapshots(
            conn,
            [PlayerSnapshotRecord(player_id=8, captured_at="2026-09-09T18:00:00Z", now_cost=49, raw_json={})],
            run2,
        )
        # Re-entering the OLD selling price against the NEW market is a
        # same-market disagreement: explicit MISMATCH, never a silent freeze.
        manually_reapplied = repo.upsert_manager_selling_prices(
            conn, 241392, 4, {8: 47}, captured_at="2026-09-09T18:05:00Z", market_prices_at_capture={8: 49}
        )
        assert manually_reapplied == 1
        value = metrics.effective_selling_price(conn, 241392, 4, 8)
        assert value["effective_selling_price"] is None and value["status"] == "MISMATCH"
        # When manual agree with the fresh market they verify again.
        repo.upsert_manager_selling_prices(
            conn, 241392, 4, {8: 48}, captured_at="2026-09-09T18:10:00Z", market_prices_at_capture={8: 49}
        )
        value = metrics.effective_selling_price(conn, 241392, 4, 8)
        assert value["status"] == "VERIFIED_BY_MANUAL" and value["effective_selling_price"] == 48
        observations = repo.list_manager_selling_price_observations(conn, 241392, 4, 8)
        assert [row["selling_price"] for row in observations] == [47, 47, 48]
    conn.close()


def test_migration_from_schema_version_three_preserves_existing_data(tmp_path):
    """Upgrading an existing real-schema database keeps every fact row."""

    database = tmp_path / "upgraded.db"
    legacy_max = SCHEMA_VERSION - 1
    raw_conn = sqlite3.connect(database)
    raw_conn.execute("CREATE TABLE IF NOT EXISTS schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    raw_conn.execute("INSERT INTO schema_meta VALUES ('schema_version', ?)", (str(legacy_max),))
    raw_conn.close()

    seed = connect_database_raw_until_m003(database)
    with seed:
        seed.execute("INSERT INTO schema_meta(key, value) VALUES ('schema_version', '3') ON CONFLICT(key) DO UPDATE SET value=excluded.value")
        seed.execute(
            "INSERT INTO manager_manual_state(entry_id, event, free_transfers, bank, source, captured_at)"
            " VALUES (241392, 4, 3, 0, 'manual', '2026-09-07T13:34:29Z')"
        )
        seed.execute(
            "INSERT INTO manager_selling_prices(entry_id, event, player_id, selling_price, source, captured_at)"
            " VALUES (241392, 4, 8, 47, 'manual', '2026-09-07T13:34:29Z')"
        )
    seed.close()

    migrated = connect_database(database)
    with migrated:
        repo.upsert_manual_manager_state(migrated, 241392, 4, 2, 3, captured_at="2026-09-09T18:00:00Z")
    observations = repo.list_manager_state_observations(migrated, 241392, 4)
    assert [(row["free_transfers"], row["bank"]) for row in observations] == [(3, 0), (2, 3)]
    price_observations = repo.list_manager_selling_price_observations(migrated, 241392, 4, 8)
    assert [row["selling_price"] for row in price_observations] == [47]
    version = migrated.execute("SELECT value FROM schema_meta WHERE key='schema_version'").fetchone()[0]
    assert int(version) == SCHEMA_VERSION
    migrated.close()


def connect_database_raw_until_m003(database):
    """Migrate a fresh database to exactly schema v3 (pre-m004 real schema)."""

    from fpl_brain.database import MIGRATIONS

    conn = sqlite3.connect(database)
    conn.execute("CREATE TABLE IF NOT EXISTS schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    for version, migration in enumerate(MIGRATIONS, start=1):
        if version > 3:
            break
        with conn:
            migration(conn)
    return conn
