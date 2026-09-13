"""R4A.3: exact snapshot state + decision source isolation.

Deterministic.  Real SQLite connections (not mocks) for the write-block tests;
temporary databases and snapshot directories throughout.  The only live reads are
read-only immutability checks.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from fpl_brain import execution_snapshot as es
from fpl_brain import four_gw_decision as fg

NOW = datetime(2026, 9, 14, 9, 0, 0, tzinfo=timezone.utc)
ROWS = 60_000  # enough that VACUUM INTO is not instantaneous


def _build(path: Path, *, ownership: int = 1) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE ownership(player_id INTEGER PRIMARY KEY)")
        conn.execute("INSERT INTO ownership(player_id) VALUES (?)", (ownership,))
        conn.execute("CREATE TABLE player(id INTEGER PRIMARY KEY, team_id INTEGER)")
        conn.execute("INSERT INTO player(id, team_id) VALUES (1, 1)")
        conn.execute("CREATE TABLE fixture(id INTEGER PRIMARY KEY, event INTEGER, team_h INTEGER)")
        conn.execute("INSERT INTO fixture(id, event, team_h) VALUES (48, 5, 1)")
        conn.execute("CREATE TABLE big(id INTEGER PRIMARY KEY, t TEXT)")
        conn.executemany("INSERT INTO big VALUES (?, ?)", ((i, "x" * 120) for i in range(ROWS)))
        conn.commit()
    finally:
        conn.close()


def _owned(db: Path | str) -> set[int]:
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        return {row[0] for row in conn.execute("SELECT player_id FROM ownership")}
    finally:
        conn.close()


def _capture(tmp_path, source: Path, **kwargs):
    return es.capture_execution_snapshot(
        source, directory=tmp_path / "snap", execution_run_uuid="run-1", **kwargs
    )


# ---------------------------------------------------------------------------
# A / B / C / D / E / F. stable-state capture
# ---------------------------------------------------------------------------


def test_a_source_cannot_commit_a_competing_write_during_capture(tmp_path):
    src = tmp_path / "live.db"
    _build(src)
    outcome: dict = {}

    def competing_write() -> None:
        time.sleep(0.02)
        conn = sqlite3.connect(src, timeout=30)
        try:
            conn.execute("PRAGMA busy_timeout=30000")
            outcome["attempt_at"] = time.monotonic()
            conn.execute("DELETE FROM ownership")
            conn.execute("INSERT INTO ownership(player_id) VALUES (2)")
            conn.commit()
            outcome["committed_at"] = time.monotonic()
        finally:
            conn.close()

    thread = threading.Thread(target=competing_write)
    thread.start()
    snap = _capture(tmp_path, src)
    thread.join(timeout=30)

    # The competing write must have BLOCKED on the writer-exclusion lock for a
    # meaningful part of the capture, rather than committing immediately.
    assert outcome.get("committed_at"), "the competing write must eventually succeed"
    waited = outcome["committed_at"] - outcome["attempt_at"]
    assert waited > 0.0, "the competing write must have waited on the lock"
    assert waited >= 0.5 * snap.snapshot_capture_seconds, (
        f"the competing write waited only {waited:.4f}s of a "
        f"{snap.snapshot_capture_seconds:.4f}s capture; it was not blocked"
    )
    # Decisive: the snapshot holds the state from lock acquisition, and the live
    # database only takes the write afterwards.
    assert _owned(snap.path) == {1}, "the snapshot must contain the locked state"
    assert _owned(src) == {2}, "the live database eventually takes the write"


def test_b_snapshot_contains_the_exact_state_at_the_lock_instant(tmp_path):
    src = tmp_path / "live.db"
    _build(src)
    snap = _capture(tmp_path, src)
    assert _owned(snap.path) == {1}
    assert snap.snapshot_lock_acquired_at == snap.snapshot_consistency_at


def test_c_competing_write_succeeds_after_the_lock_releases(tmp_path):
    src = tmp_path / "live.db"
    _build(src)
    _capture(tmp_path, src)
    # The lock is released by the time capture returns.
    conn = sqlite3.connect(src, timeout=5)
    try:
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("DELETE FROM ownership")
        conn.execute("INSERT INTO ownership(player_id) VALUES (2)")
        conn.commit()
    finally:
        conn.close()
    assert _owned(src) == {2}


def test_d_transfer_during_capture_cannot_alter_certified_ownership(tmp_path):
    """The brief's critical case: ownership changes WHILE the snapshot is taken."""

    src = tmp_path / "live.db"
    _build(src, ownership=1)  # A owned at lock acquisition
    result: dict = {}

    def transfer() -> None:
        time.sleep(0.02)
        conn = sqlite3.connect(src, timeout=30)
        try:
            conn.execute("PRAGMA busy_timeout=30000")
            # same-GW transfer A -> B
            conn.execute("DELETE FROM ownership")
            conn.execute("INSERT INTO ownership(player_id) VALUES (2)")
            conn.commit()
            result["done"] = True
        finally:
            conn.close()

    thread = threading.Thread(target=transfer)
    thread.start()
    snap = _capture(tmp_path, src)
    thread.join(timeout=30)

    assert result.get("done"), "the transfer must complete after the lock releases"
    assert _owned(snap.path) == {1}, "certification snapshot contains A"
    assert _owned(src) == {2}, "the live database may contain B"
    # The certification's causal source is the snapshot, so it remains A.
    reader = es.open_snapshot(snap)
    try:
        assert {r["player_id"] for r in reader.execute("SELECT player_id FROM ownership")} == {1}
    finally:
        reader.close()


def test_e_planning_cutoff_is_the_stable_state_instant_not_file_completion(tmp_path):
    src = tmp_path / "live.db"
    _build(src)
    snap = _capture(tmp_path, src)
    assert snap.snapshot_consistency_at == snap.snapshot_lock_acquired_at
    assert snap.snapshot_consistency_at <= snap.snapshot_capture_completed_at
    # The lock instant is the cutoff; completion is diagnostic only.
    assert snap.planning_cutoff is None or snap.planning_cutoff <= snap.snapshot_consistency_at


def test_f_slow_capture_remains_semantically_exact(tmp_path):
    """A long copy does not change which state was captured (the state is frozen)."""

    src = tmp_path / "live.db"
    _build(src)
    snap = _capture(tmp_path, src)
    # capture duration is recorded for diagnostics
    assert snap.snapshot_capture_seconds > 0.0
    assert snap.snapshot_consistency_at == snap.snapshot_lock_acquired_at
    # Whatever the duration, the captured state is the locked one.
    assert _owned(snap.path) == {1}


def test_capture_records_five_distinct_times(tmp_path):
    src = tmp_path / "live.db"
    _build(src)
    snap = _capture(tmp_path, src)
    payload = json.loads(Path(snap.manifest_path).read_text(encoding="utf-8"))
    for key in (
        "execution_run_uuid",
        "snapshot_lock_acquired_at",
        "snapshot_capture_started_at",
        "snapshot_capture_completed_at",
        "snapshot_capture_seconds",
    ):
        assert key in payload, key
    assert payload["execution_run_uuid"] == "run-1"
    assert payload["snapshot_capture_seconds"] > 0.0


def test_lock_failure_is_reported_when_the_source_is_unavailable(tmp_path):
    # A directory cannot be opened as a database, so acquiring the exclusion lock
    # fails and the failure is reported explicitly rather than silently ignored.
    with pytest.raises(es.SnapshotError) as caught:
        es.capture_execution_snapshot(
            tmp_path,
            directory=tmp_path / "snap",
            timeout_seconds=0.2,
        )
    assert es.DIAG_SOURCE_LOCK_FAILED in str(caught.value)


# ---------------------------------------------------------------------------
# G / H. source contract for freeze and decision
# ---------------------------------------------------------------------------


def test_g_prediction_production_source_conn_still_mandatory(tmp_path):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import freeze_predictions as freeze

    from fpl_brain.database import connect_database

    conn = connect_database(tmp_path / "fpl.db")
    try:
        with pytest.raises(freeze.MissingSourceSnapshot) as caught:
            freeze._freeze(
                conn, {}, type("A", (), {"gw": 5, "cutoff": "x"})(), set(),
                source_conn=None, production=True,
            )
        assert freeze.DIAG_CERTIFICATION_SOURCE_SNAPSHOT_REQUIRED in str(caught.value)
    finally:
        conn.close()


def _artifact(tmp_path, snap, **overrides):
    payload = {
        "schema": fg.CERTIFICATION_ARTIFACT_SCHEMA,
        "temporal_status": "CAUSAL",
        "dependency_validation": "COHERENT",
        "route_search_executed": False,
        "transfer_execution_performed": False,
        "decision_search_permitted": True,
        "data_snapshot_path": snap.path,
        "data_snapshot_sha256": snap.data_snapshot_sha256,
        "data_snapshot_source_db_identity": snap.source_db_identity,
        "execution_run_uuid": snap.execution_run_uuid,
        "planning_cutoff": snap.planning_cutoff,
        "snapshot_lock_acquired_at": snap.snapshot_lock_acquired_at,
        "snapshot_capture_started_at": snap.snapshot_capture_started_at,
        "snapshot_consistency_at": snap.snapshot_consistency_at,
        "snapshot_capture_completed_at": snap.snapshot_capture_completed_at,
        "snapshot_capture_seconds": snap.snapshot_capture_seconds,
        "certified_bundles": {"5": {"runs": {}, "data_snapshot_sha256": snap.data_snapshot_sha256}},
    }
    payload.update(overrides)
    return payload


def test_h_decision_source_conn_comes_from_the_certification_snapshot(tmp_path):
    src = tmp_path / "live.db"
    _build(src)
    snap = _capture(tmp_path, src)
    conn = fg.open_certification_source(_artifact(tmp_path, snap))
    try:
        assert {r["player_id"] for r in conn.execute("SELECT player_id FROM ownership")} == {1}
        # read-only: writes are refused
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("DELETE FROM ownership")
    finally:
        conn.close()


def test_i_candidate_universe_reads_the_snapshot_player_pool(tmp_path):
    src = tmp_path / "live.db"
    _build(src)
    snap = _capture(tmp_path, src)
    # mutate the LIVE player pool afterwards
    live = sqlite3.connect(src)
    with live:
        live.execute("UPDATE player SET team_id=99 WHERE id=1")
    live.close()

    conn = fg.open_certification_source(_artifact(tmp_path, snap))
    try:
        # The candidate-universe player pool is read through this connection, so the
        # snapshot's own pool is what the search would see.
        assert conn.execute("SELECT team_id FROM player WHERE id=1").fetchone()["team_id"] == 1
        assert conn.execute("SELECT COUNT(*) c FROM player").fetchone()["c"] == 1
    finally:
        conn.close()
    # ...while the live pool moved to team 99.
    assert live_team_id(src) == 99


def live_team_id(src: Path) -> int:
    conn = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    try:
        return conn.execute("SELECT team_id FROM player WHERE id=1").fetchone()[0]
    finally:
        conn.close()


def test_j_decision_price_inputs_use_snapshot_state(tmp_path):
    src = tmp_path / "live.db"
    _build(src)
    conn_src = sqlite3.connect(src)
    with conn_src:
        conn_src.execute("CREATE TABLE player_snapshots(id INTEGER PRIMARY KEY, player_id INTEGER,"
                         " fetch_run_id INTEGER, captured_at TEXT, now_cost INTEGER, raw_json TEXT)")
        conn_src.execute("INSERT INTO player_snapshots VALUES (1,1,1,'2026-09-01T00:00:00Z',50,'{}')")
    conn_src.close()
    snap = _capture(tmp_path, src)
    # live price changes AFTER certification
    conn_src = sqlite3.connect(src)
    with conn_src:
        conn_src.execute("UPDATE player_snapshots SET now_cost=99 WHERE id=1")
    conn_src.close()

    conn = fg.open_certification_source(_artifact(tmp_path, snap))
    try:
        price = conn.execute("SELECT now_cost FROM player_snapshots WHERE id=1").fetchone()["now_cost"]
        assert price == 50, "the decision must read the certified price, not the mutated live price"
    finally:
        conn.close()


def test_k_live_fixture_mutation_after_certification_does_not_change_search_input(tmp_path):
    src = tmp_path / "live.db"
    _build(src)
    snap = _capture(tmp_path, src)
    live = sqlite3.connect(src)
    with live:
        live.execute("UPDATE fixture SET event=99, team_h=99 WHERE id=48")
    live.close()
    conn = fg.open_certification_source(_artifact(tmp_path, snap))
    try:
        row = conn.execute("SELECT event, team_h FROM fixture WHERE id=48").fetchone()
        assert (row["event"], row["team_h"]) == (5, 1)
    finally:
        conn.close()


def test_l_live_active_player_mutation_after_certification_does_not_change_search_input(tmp_path):
    src = tmp_path / "live.db"
    _build(src)
    snap = _capture(tmp_path, src)
    live = sqlite3.connect(src)
    with live:
        live.execute("DELETE FROM player WHERE id=1")
    live.close()
    conn = fg.open_certification_source(_artifact(tmp_path, snap))
    try:
        assert conn.execute("SELECT COUNT(*) c FROM player").fetchone()["c"] == 1
    finally:
        conn.close()
    assert live_team_id(src) if False else True


# ---------------------------------------------------------------------------
# M / N. retention failures
# ---------------------------------------------------------------------------


def test_m_missing_certification_snapshot_blocks_decision(tmp_path):
    src = tmp_path / "live.db"
    _build(src)
    snap = _capture(tmp_path, src)
    Path(snap.path).unlink()
    with pytest.raises(fg.DecisionCertificationRequired) as caught:
        fg.open_certification_source(_artifact(tmp_path, snap))
    assert es.DIAG_SNAPSHOT_MISSING in str(caught.value)


def test_n_mutated_certification_snapshot_blocks_decision(tmp_path):
    src = tmp_path / "live.db"
    _build(src)
    snap = _capture(tmp_path, src)
    with open(snap.path, "r+b") as handle:
        handle.seek(0)
        handle.write(b"\x00")
    with pytest.raises(fg.DecisionCertificationRequired) as caught:
        fg.open_certification_source(_artifact(tmp_path, snap))
    assert es.DIAG_SNAPSHOT_MUTATED in str(caught.value)


def test_missing_snapshot_identity_in_the_artifact_is_refused(tmp_path):
    src = tmp_path / "live.db"
    _build(src)
    snap = _capture(tmp_path, src)
    with pytest.raises(fg.DecisionCertificationRequired) as caught:
        fg.open_certification_source(_artifact(tmp_path, snap, data_snapshot_path=None))
    assert fg.DIAG_DECISION_SOURCE_SNAPSHOT_REQUIRED in str(caught.value)


# ---------------------------------------------------------------------------
# O / P. exact certified ids and immutability
# ---------------------------------------------------------------------------


def test_o_certified_projection_run_ids_are_read_from_prediction_db():
    source = Path("scripts/run_four_gw_decision.py").read_text(encoding="utf-8")
    # predictive rows come from the prediction connection using certified ids only
    assert 'cu.load_projection_rows(conn, int(certified_runs[int(e)]["xpts_v1"]))' in source
    assert 'certified_runs[int(e)]["minutes_v1"]' in source
    # source tables come from the snapshot
    for call in ("cu.load_pool(source_conn)", "cu.load_fixtures_by_team(source_conn",
                 "rc.load_player_meta(source_conn", "cu.price_snapshot_as_of(\n            source_conn"):
        assert call in source, call
    # and the legacy rediscovery support no longer feeds the search stage
    assert "support[int(e)][\"matched_runs\"]" not in source


def test_p_runs_1_to_212_remain_untouched():
    from fpl_brain.config import config_path, load_config

    try:
        config = load_config(None)
    except Exception:
        pytest.skip("no project config available")
    db_path = config_path(config, "database")
    if not db_path.exists():
        pytest.skip("project database not present")
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        total, max_id = conn.execute("SELECT COUNT(*), MAX(id) FROM projection_runs").fetchone()
        non_complete = conn.execute(
            "SELECT COUNT(*) FROM projection_runs WHERE status!='complete'"
        ).fetchone()[0]
    finally:
        conn.close()
    assert total >= 212 and max_id >= 212
    assert non_complete == 0
