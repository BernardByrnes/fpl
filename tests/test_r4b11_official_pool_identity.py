"""R4B.1.1 — certified official-pool identity tests.

Synthetic-first and hermetic: every test that needs a database builds its own
temporary SQLite file (or an in-memory database), so the live ``fpl.db``,
``projection_runs`` and any production artifact are never touched.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from fpl_brain import database as db
from fpl_brain import ingest_provenance as prov
from fpl_brain import repositories as repo

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "scripts" / "run_four_gw_decision.py"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _elements(ids, status="a", team=1):
    return {"elements": [{"id": pid, "web_name": f"P{pid}", "status": status, "team": team,
                          "element_type": 3, "now_cost": 50} for pid in ids]}


def _temp_db(tmp_path) -> sqlite3.Connection:
    """A fresh, fully migrated database (all migrations, including m015)."""

    conn = db.connect_database(tmp_path / "fpl.db")
    return conn


def _seed_players(conn, ids, active=()):
    """Insert players, marking ``active`` (default: all) active."""

    active_set = set(ids) if not active else set(active)
    for pid in ids:
        conn.execute(
            "INSERT INTO players(id, web_name, full_name, first_seen_at, last_seen_at, is_active, raw_json, updated_at)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (pid, f"P{pid}", f"Player {pid}", "2026-09-12T00:00:00Z", "2026-09-12T00:00:00Z",
             1 if pid in active_set else 0, "{}", "2026-09-12T00:00:00Z"),
        )
    conn.commit()


def _record_generation(conn, ids, *, accepted=True, capture="2026-09-12T10:00:00Z", statuses=None):
    generation = prov.build_generation(
        payload={"elements": [{"id": pid, "web_name": f"P{pid}", "team": 1,
                               "status": (statuses or {}).get(pid, "a"),
                               "element_type": 3, "now_cost": 50} for pid in ids]},
        parsed_count=len(ids), persisted_count=len(ids), captured_at=capture, run_id=1,
    )
    assert generation.accepted is accepted, generation.rejection_reasons
    row_id = repo.record_bootstrap_generation(
        conn,
        captured_at=generation.captured_at,
        accepted=generation.accepted,
        official_element_count=generation.official_element_count,
        parsed_count=generation.parsed_count,
        persisted_count=generation.persisted_count,
        element_ids=generation.element_ids,
        element_ids_sha256=generation.element_id_sha256,
        availability_counts=generation.availability_counts,
        club_player_counts=generation.club_player_counts,
        acceptance_rule=generation.acceptance_rule,
        acceptance_rule_version=prov.ACCEPTANCE_RULE_VERSION,
        rejection_reasons=generation.rejection_reasons,
        fetch_run_id=generation.run_id,
    )
    conn.commit()
    return generation, row_id


def _assert_identity(conn, *, enforce=True):
    return prov.assert_official_pool_identity(
        accepted_generation=repo.latest_accepted_bootstrap_generation(conn),
        snapshot_player_ids=repo.active_player_ids(conn),
        enforce=enforce,
    )


# ---------------------------------------------------------------------------
# Migration / schema
# ---------------------------------------------------------------------------
def test_m015_migration_creates_the_generation_table(tmp_path):
    conn = _temp_db(tmp_path)
    try:
        row = conn.execute("SELECT value FROM schema_meta WHERE key='schema_version'").fetchone()
        assert int(row[0]) == db.SCHEMA_VERSION == 15
        cols = {r[1] for r in conn.execute("PRAGMA table_info(bootstrap_generations)")}
        for column in ("id", "fetch_run_id", "captured_at", "accepted", "official_element_count",
                       "parsed_count", "persisted_count", "element_ids_sha256", "element_ids_json",
                       "acceptance_rule", "acceptance_rule_version", "recorded_at"):
            assert column in cols
        # The migration is additive: projection_runs is untouched.
        assert conn.execute("SELECT COUNT(*) FROM projection_runs").fetchone()[0] == 0
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# A — exact match PASSES
# ---------------------------------------------------------------------------
def test_A_exact_generation_matches_snapshot_pool(tmp_path):
    conn = _temp_db(tmp_path)
    try:
        ids = list(range(1, 26))
        _record_generation(conn, ids)
        _seed_players(conn, ids)
        report = _assert_identity(conn)
        assert report["official_pool_identity_match"] is True
        assert report["official_generation_count"] == len(ids)
        assert report["snapshot_pool_count"] == len(ids)
        assert report["official_generation_ids_sha256"] == report["snapshot_pool_ids_sha256"]
        assert report["missing_from_snapshot_pool"] == []
        assert report["extra_in_snapshot_pool"] == []
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# B / C / D — any identity deviation FAILS
# ---------------------------------------------------------------------------
def test_B_one_player_missing_from_snapshot_pool_fails(tmp_path):
    conn = _temp_db(tmp_path)
    try:
        ids = list(range(1, 26))
        _record_generation(conn, ids)
        _seed_players(conn, ids, active=[pid for pid in ids if pid != 7])
        with pytest.raises(prov.OfficialPoolIncomplete, match=prov.DIAG_INGEST_INCOMPLETE):
            _assert_identity(conn)
        report = _assert_identity(conn, enforce=False)
        assert report["official_pool_identity_match"] is False
        assert 7 in report["missing_from_snapshot_pool"]
        assert "COUNT_MISMATCH" in " ".join(report["identity_reasons"])
    finally:
        conn.close()


def test_C_one_extra_player_in_snapshot_pool_fails(tmp_path):
    conn = _temp_db(tmp_path)
    try:
        ids = list(range(1, 26))
        _record_generation(conn, ids)
        _seed_players(conn, ids + [999])          # 999 is not in the generation
        with pytest.raises(prov.OfficialPoolIncomplete):
            _assert_identity(conn)
        report = _assert_identity(conn, enforce=False)
        assert report["official_pool_identity_match"] is False
        assert 999 in report["extra_in_snapshot_pool"]
    finally:
        conn.close()


def test_D_equal_count_but_different_id_fails(tmp_path):
    """The whole point: counts alone cannot prove identity."""

    conn = _temp_db(tmp_path)
    try:
        ids = list(range(1, 26))
        _record_generation(conn, ids)
        swapped = [pid for pid in ids if pid != 25] + [777]   # same SIZE, one different id
        _seed_players(conn, swapped)
        report = _assert_identity(conn, enforce=False)
        assert report["official_generation_count"] == report["snapshot_pool_count"] == 25
        assert report["official_pool_identity_match"] is False
        assert "ID_HASH_MISMATCH" in report["identity_reasons"]
        assert 25 in report["missing_from_snapshot_pool"]
        assert 777 in report["extra_in_snapshot_pool"]
        with pytest.raises(prov.OfficialPoolIncomplete):
            _assert_identity(conn)
    finally:
        conn.close()


def test_no_accepted_generation_fails_closed(tmp_path):
    conn = _temp_db(tmp_path)
    try:
        _seed_players(conn, list(range(1, 6)))
        report = _assert_identity(conn, enforce=False)
        assert report["official_pool_identity_match"] is False
        assert "NO_ACCEPTED_OFFICIAL_GENERATION_IN_SNAPSHOT" in report["identity_reasons"]
        with pytest.raises(prov.OfficialPoolIncomplete):
            _assert_identity(conn)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# E — a rejected generation never becomes authoritative
# ---------------------------------------------------------------------------
def test_E_rejected_generation_never_becomes_authoritative(tmp_path):
    conn = _temp_db(tmp_path)
    try:
        good = list(range(1, 501))
        _record_generation(conn, good)
        _seed_players(conn, good)

        # A rejected (truncated) generation, recorded later with a higher id.
        truncated_ids = list(range(1, 101))
        rejected = prov.build_generation(
            payload=_elements(truncated_ids), parsed_count=len(truncated_ids),
            persisted_count=len(truncated_ids), captured_at="2026-09-12T12:00:00Z", run_id=2,
            previous=prov.build_generation(payload=_elements(good), parsed_count=len(good),
                                           persisted_count=len(good),
                                           captured_at="2026-09-12T10:00:00Z", run_id=1),
        )
        assert rejected.accepted is False
        repo.record_bootstrap_generation(
            conn, captured_at=rejected.captured_at, accepted=False,
            official_element_count=rejected.official_element_count,
            parsed_count=rejected.parsed_count, persisted_count=rejected.persisted_count,
            element_ids=rejected.element_ids, element_ids_sha256=rejected.element_id_sha256,
            acceptance_rule=rejected.acceptance_rule,
            acceptance_rule_version=prov.ACCEPTANCE_RULE_VERSION,
            rejection_reasons=rejected.rejection_reasons, fetch_run_id=2,
        )
        conn.commit()

        # The rejected row exists (audit) but is NOT the authoritative identity.
        latest_any = repo.latest_bootstrap_generation_attempt(conn)
        assert latest_any is not None and int(latest_any["accepted"]) == 0
        accepted = repo.latest_accepted_bootstrap_generation(conn)
        assert accepted is not None
        assert int(accepted["official_element_count"]) == len(good)
        # And the assertion still passes against the FULL pool.
        assert _assert_identity(conn)["official_pool_identity_match"] is True

        # If the pool had actually been truncated by that rejected generation,
        # the identity check would fail rather than accept the smaller pool.
        conn.execute("UPDATE players SET is_active=0 WHERE id > 100")
        conn.commit()
        assert _assert_identity(conn, enforce=False)["official_pool_identity_match"] is False
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# F — availability status is not membership
# ---------------------------------------------------------------------------
def test_F_injured_and_suspended_players_remain_in_official_identity(tmp_path):
    conn = _temp_db(tmp_path)
    try:
        ids = [1, 2, 3, 4]
        statuses = {2: "i", 3: "s", 4: "d"}   # injured, suspended, doubtful
        generation, _ = _record_generation(conn, ids, statuses=statuses)
        assert set(generation.element_ids) == set(ids)
        assert generation.availability_counts["i"] == 1
        _seed_players(conn, ids)
        report = _assert_identity(conn)
        assert report["official_pool_identity_match"] is True
        assert report["official_generation_count"] == 4
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# G — the decision uses the snapshot row, not the external JSON
# ---------------------------------------------------------------------------
def test_G_decision_reads_the_snapshot_row_not_external_json(tmp_path):
    """A lying external JSON report must not affect the decision's identity."""

    conn = _temp_db(tmp_path)
    try:
        ids = list(range(1, 21))
        _record_generation(conn, ids)
        _seed_players(conn, ids)

        # Write a deliberately WRONG external report next to a raw dir.
        raw_dir = tmp_path / "raw"
        (raw_dir / prov.GENERATIONS_DIRNAME).mkdir(parents=True, exist_ok=True)
        (raw_dir / prov.GENERATIONS_DIRNAME / prov.GENERATION_FILENAME).write_text(
            json.dumps({"element_ids": [1, 2, 3], "official_element_count": 3}), encoding="utf-8"
        )

        # The assertion takes the accepted generation from SQLite, so the wrong
        # JSON cannot influence it.
        report = _assert_identity(conn)
        assert report["official_pool_identity_match"] is True
        assert report["official_generation_count"] == len(ids)

        # Provenance/wiring: the runner reads the generation from source_conn.
        source = RUNNER.read_text(encoding="utf-8")
        assert "repo.latest_accepted_bootstrap_generation(source_conn)" in source
        assert "repo.active_player_ids(source_conn)" in source
        assert "load_latest_generation" not in source
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# H — post-certification ingestion cannot alter the certified identity
# ---------------------------------------------------------------------------
def test_H_post_certification_ingest_cannot_alter_the_certified_identity(tmp_path):
    """The snapshot is immutable: later live ingest must not change its verdict."""

    live_path = tmp_path / "live.db"
    snap_path = tmp_path / "snapshot.db"

    live = db.connect_database(live_path)
    try:
        ids = list(range(1, 31))
        _record_generation(live, ids)
        _seed_players(live, ids)

        # Certify: capture an immutable snapshot (VACUUM INTO), exactly as R4A.3 does.
        live.execute("PRAGMA busy_timeout=30000")
        live.execute("VACUUM INTO ?", (str(snap_path),))
        live.commit()

        # POST-CERTIFICATION live ingest: deactivate some players, add a new one,
        # and record a newer accepted generation against a DIFFERENT pool.
        live.execute("UPDATE players SET is_active=0 WHERE id IN (1,2,3,4,5)")
        _seed_players(live, [555])
        live.commit()
        conn2 = db.connect_database(live_path)
        conn2.execute("DELETE FROM bootstrap_generations")
        conn2.commit()
        _record_generation(conn2, list(range(1, 31)) + [555])
        conn2.close()

        # The snapshot still holds the ORIGINAL identity and pool, and passes.
        snap = sqlite3.connect(f"file:{snap_path.as_posix()}?mode=ro", uri=True)
        snap.row_factory = sqlite3.Row
        try:
            report = prov.assert_official_pool_identity(
                accepted_generation=repo.latest_accepted_bootstrap_generation(snap),
                snapshot_player_ids=repo.active_player_ids(snap),
            )
            assert report["official_pool_identity_match"] is True
            assert report["official_generation_count"] == 30
            assert report["snapshot_pool_count"] == 30
            assert 555 not in repo.active_player_ids(snap)
        finally:
            snap.close()

        # Whereas the LIVE database now disagrees with its own accepted generation.
        live2 = sqlite3.connect(f"file:{live_path.as_posix()}?mode=ro", uri=True)
        live2.row_factory = sqlite3.Row
        try:
            live_report = prov.assert_official_pool_identity(
                accepted_generation=repo.latest_accepted_bootstrap_generation(live2),
                snapshot_player_ids=repo.active_player_ids(live2),
                enforce=False,
            )
            assert live_report["official_pool_identity_match"] is False
        finally:
            live2.close()
    finally:
        live.close()


# ---------------------------------------------------------------------------
# I — projection_runs is not touched
# ---------------------------------------------------------------------------
def test_I_projection_runs_remain_212():
    db_path = REPO_ROOT / "fpl.db"
    if not db_path.exists():
        pytest.skip("no local database")
    conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    try:
        count, max_id = conn.execute("SELECT COUNT(*), MAX(id) FROM projection_runs").fetchone()
    finally:
        conn.close()
    assert count == 212, f"expected 212 projection runs, found {count}"
    assert max_id == 212


def test_I_migration_does_not_grow_projection_runs(tmp_path):
    """Migrating a fresh database (through m015) creates no projection runs."""

    conn = _temp_db(tmp_path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM projection_runs").fetchone()[0] == 0
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Artifact shape (§4)
# ---------------------------------------------------------------------------
def test_artifact_exposes_the_identity_fields(tmp_path):
    conn = _temp_db(tmp_path)
    try:
        ids = list(range(1, 11))
        _, row_id = _record_generation(conn, ids)
        _seed_players(conn, ids)
        report = _assert_identity(conn)
        for field in ("official_generation_id", "official_generation_captured_at",
                      "official_generation_count", "official_generation_ids_sha256",
                      "snapshot_pool_count", "snapshot_pool_ids_sha256",
                      "official_pool_identity_match"):
            assert field in report, field
        assert report["official_generation_id"] == row_id
        # Identity is primary; the pool accounting is explicitly secondary.
        assert report["official_pool_identity_match"] is True
    finally:
        conn.close()


def test_acceptance_rule_version_is_persisted(tmp_path):
    conn = _temp_db(tmp_path)
    try:
        _record_generation(conn, [1, 2, 3])
        row = conn.execute(
            "SELECT acceptance_rule_version, acceptance_rule FROM bootstrap_generations ORDER BY id DESC LIMIT 1"
        ).fetchone()
        assert row[0] == prov.ACCEPTANCE_RULE_VERSION
        assert row[1] == "FIRST_GENERATION"
    finally:
        conn.close()
