"""R4A.2: snapshot timepoint + fail-closed source contract.

Deterministic.  Temporary databases and snapshot directories; the only live reads
are read-only immutability checks.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from fpl_brain import causality as cx
from fpl_brain import execution_snapshot as es
from fpl_brain import four_gw_decision as fg
from fpl_brain import repositories as repo
from fpl_brain.database import connect_database

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import freeze_predictions as freeze  # noqa: E402


class StepClock:
    """A clock that advances a fixed step on every read, so windows are visible."""

    def __init__(self, start: datetime, step_seconds: float = 0.0) -> None:
        self.now = start
        self.step = step_seconds

    def __call__(self) -> datetime:
        moment = self.now
        self.now = self.now + timedelta(seconds=self.step)
        return moment


def _world(path: Path):
    conn = connect_database(path)
    with conn:
        conn.execute("INSERT INTO positions(id, singular_name_short, raw_json, updated_at)"
                     " VALUES (2,'DEF','{}','2026-09-01T00:00:00Z')")
        conn.execute("INSERT INTO teams(id, name, short_name, raw_json, updated_at)"
                     " VALUES (1,'One','ONE','{}','2026-09-01T00:00:00Z')")
        conn.execute("INSERT INTO events(id, name, deadline_time, finished, raw_json, updated_at)"
                     " VALUES (5,'GW5','2026-09-18T17:30:00Z',0,'{}','2026-09-01T00:00:00Z')")
        conn.execute("INSERT INTO fixture_runs_probe" if False else
                     "INSERT INTO fetch_runs(id, started_at, status, trigger)"
                     " VALUES (1,'2026-09-01T00:00:00Z','success','test')")
        for pid, name in ((1, "A"), (2, "B")):
            conn.execute(
                "INSERT INTO players(id, web_name, team_id, element_type, is_active, first_seen_at,"
                " last_seen_at, raw_json, updated_at)"
                " VALUES (?,?,1,2,1,'2026-09-01T00:00:00Z','2026-09-01T00:00:00Z','{}',"
                "'2026-09-01T00:00:00Z')",
                (pid, name),
            )
        # Initial squad: A owned since event 1.  No acquired_at timestamp, matching
        # the live store where 15 of 17 rows have none.
        conn.execute(
            "INSERT INTO manager_player_acquisitions(id, entry_id, player_id, acquired_event,"
            " purchase_price, sold_event, source, created_at, updated_at)"
            " VALUES (1,241392,1,1,50,NULL,'verified_initial_squad',"
            "'2026-09-07T10:00:00Z','2026-09-07T10:00:00Z')"
        )
    return conn


def _capture(tmp_path, *, cutoff=None, clock=None, directory=None):
    return es.capture_execution_snapshot(
        tmp_path / "live.db",
        directory=directory or (tmp_path / "snap"),
        execution_run_uuid="run-1",
        planning_cutoff=cutoff,
        clock=clock or (lambda: datetime(2026, 9, 14, 9, 0, 0, tzinfo=timezone.utc)),
    )


# ---------------------------------------------------------------------------
# 2. three distinct times / SQLite consistency semantics
# ---------------------------------------------------------------------------


def test_snapshot_records_three_distinct_times(tmp_path):
    conn = _world(tmp_path / "live.db")
    clock = StepClock(datetime(2026, 9, 14, 9, 0, 0, tzinfo=timezone.utc), step_seconds=3.0)
    snap = _capture(tmp_path, clock=clock)
    # R4A.3 changed the semantics: the source is under SQLite writer exclusion for
    # the whole copy, so the state is provably constant and the causal instant is the
    # LOCK ACQUISITION time, not the (later) file-completion time.
    assert snap.snapshot_lock_acquired_at == "2026-09-14T09:00:00Z"
    assert snap.snapshot_capture_started_at == snap.snapshot_lock_acquired_at
    assert snap.snapshot_consistency_at == snap.snapshot_lock_acquired_at
    assert snap.snapshot_capture_completed_at == "2026-09-14T09:00:03Z"
    assert snap.snapshot_consistency_at <= snap.snapshot_capture_completed_at
    assert snap.created_at == snap.snapshot_consistency_at  # backwards-compatible alias
    assert es.snapshot_consistency_window(snap) == pytest.approx(3.0)
    payload = json.loads(Path(snap.manifest_path).read_text(encoding="utf-8"))
    for key in (
        "snapshot_lock_acquired_at",
        "snapshot_capture_started_at",
        "snapshot_consistency_at",
        "snapshot_capture_completed_at",
        "snapshot_capture_seconds",
    ):
        assert key in payload, key
    conn.close()


def test_consistency_instant_is_the_conservative_upper_bound(tmp_path):
    """Everything in the snapshot existed at or before the consistency instant."""

    conn = _world(tmp_path / "live.db")
    # A write lands inside the capture window (after start, before completion).
    clock = StepClock(datetime(2026, 9, 14, 9, 0, 0, tzinfo=timezone.utc), step_seconds=5.0)
    snap = _capture(tmp_path, clock=clock)
    # Claiming the cutoff to be the START would over-claim if the write were
    # captured; claiming the COMPLETED time can never over-claim, because the true
    # read instant is >= start and <= completed.
    assert snap.snapshot_consistency_at >= snap.snapshot_capture_started_at
    cx.assert_causal_cutoff(snap.snapshot_consistency_at, snap.snapshot_capture_completed_at)
    conn.close()


# ---------------------------------------------------------------------------
# A. live cutoff == snapshot consistency instant
# ---------------------------------------------------------------------------


def test_a_live_certification_cutoff_equals_the_consistency_instant(tmp_path):
    conn = _world(tmp_path / "live.db")
    snap = _capture(tmp_path)
    # No requested cutoff -> minted from the snapshot.
    assert (
        es.require_live_cutoff_matches_snapshot(snap.snapshot_consistency_at, None)
        == snap.snapshot_consistency_at
    )
    # Requested cutoff within the mechanical tolerance -> accepted, still minted.
    assert (
        es.require_live_cutoff_matches_snapshot(snap.snapshot_consistency_at, snap.snapshot_consistency_at)
        == snap.snapshot_consistency_at
    )
    assert es.CUTOFF_MATCH_TOLERANCE_SECONDS <= 2.0
    conn.close()


# ---------------------------------------------------------------------------
# B / D. arbitrary older cutoff vs a fresh snapshot
# ---------------------------------------------------------------------------


def test_b_arbitrary_older_cutoff_requires_a_historical_snapshot(tmp_path):
    conn = _world(tmp_path / "live.db")
    snap = _capture(tmp_path)  # consistency 2026-09-14T09:00:00Z
    with pytest.raises(es.SnapshotError) as caught:
        es.require_live_cutoff_matches_snapshot(snap.snapshot_consistency_at, "2026-09-14T08:00:00Z")
    assert es.DIAG_HISTORICAL_SNAPSHOT_REQUIRED in str(caught.value)
    assert "immutable snapshot" in str(caught.value)
    conn.close()


def test_d_later_snapshot_cannot_falsely_represent_an_earlier_cutoff(tmp_path):
    """The exact case from the brief: 10:00 cutoff, 11:00 transfer, 12:00 snapshot."""

    conn = _world(tmp_path / "live.db")
    # 11:00 same-GW transfer: A sold, B bought
    with conn:
        conn.execute("UPDATE manager_player_acquisitions SET sold_event=5 WHERE id=1")
        conn.execute(
            "INSERT INTO manager_player_acquisitions(id, entry_id, player_id, acquired_event,"
            " purchase_price, sold_event, source, created_at, updated_at)"
            " VALUES (2,241392,2,5,45,NULL,'manual','2026-09-14T11:00:00Z','2026-09-14T11:00:00Z')"
        )
    # 12:00 snapshot therefore contains B
    snap = _capture(
        tmp_path,
        clock=lambda: datetime(2026, 9, 14, 12, 0, 0, tzinfo=timezone.utc),
    )
    reader = es.open_snapshot(snap)
    try:
        owned = {r["player_id"] for r in reader.execute(
            "SELECT player_id FROM manager_player_acquisitions WHERE sold_event IS NULL")}
        assert owned == {2}, "the 12:00 snapshot legitimately contains B"
    finally:
        reader.close()
    # ...but it must NOT be allowed to claim it represents a 10:00 cutoff.
    with pytest.raises(es.SnapshotError) as caught:
        es.require_live_cutoff_matches_snapshot(snap.snapshot_consistency_at, "2026-09-14T10:00:00Z")
    assert es.DIAG_HISTORICAL_SNAPSHOT_REQUIRED in str(caught.value)
    conn.close()


# ---------------------------------------------------------------------------
# C / 5. same-GW transfer after the snapshot cannot leak backward
# ---------------------------------------------------------------------------


def test_c_same_gw_transfer_after_snapshot_cannot_leak_backward(tmp_path):
    conn = _world(tmp_path / "live.db")
    # T = 10:00: snapshot captured with A owned
    snap = _capture(
        tmp_path,
        clock=lambda: datetime(2026, 9, 14, 10, 0, 0, tzinfo=timezone.utc),
    )
    reader = es.open_snapshot(snap)
    before = {r["player_id"] for r in reader.execute(
        "SELECT player_id FROM manager_player_acquisitions WHERE sold_event IS NULL")}
    reader.close()
    assert before == {1}, "the snapshot at T must show A"

    # T+1h same-GW transfer
    with conn:
        conn.execute("UPDATE manager_player_acquisitions SET sold_event=5 WHERE id=1")
        conn.execute(
            "INSERT INTO manager_player_acquisitions(id, entry_id, player_id, acquired_event,"
            " purchase_price, sold_event, source, created_at, updated_at)"
            " VALUES (2,241392,2,5,45,NULL,'manual','2026-09-14T11:00:00Z','2026-09-14T11:00:00Z')"
        )
    es.assert_snapshot_unchanged(snap)

    reader = es.open_snapshot(snap)
    try:
        after = {r["player_id"] for r in reader.execute(
            "SELECT player_id FROM manager_player_acquisitions WHERE sold_event IS NULL")}
        # CERTIFICATION STILL SEES THE T SQUAD
        assert after == {1}
        # The event-level resolver on the LIVE db cannot help: it sees only B.
        live_event_level = repo.active_manager_acquisitions_as_of(conn, 241392, 5)
        assert {r["player_id"] for r in live_event_level} == {2}
    finally:
        reader.close()
    conn.close()


# ---------------------------------------------------------------------------
# E / F. slow snapshot does not break causal semantics
# ---------------------------------------------------------------------------


def test_e_slow_snapshot_completion_does_not_violate_causality(tmp_path):
    """A capture whose file copy finishes long after the UUID exists is still causal.

    The cutoff is minted from the consistency instant, so a long capture simply
    moves the cutoff later rather than invalidating the ordering.
    """

    conn = _world(tmp_path / "live.db")
    clock = StepClock(datetime(2026, 9, 14, 9, 0, 0, tzinfo=timezone.utc), step_seconds=180.0)
    snap = _capture(tmp_path, clock=clock)
    assert es.snapshot_consistency_window(snap) == pytest.approx(180.0)
    # execution_started_at is AFTER the consistency instant, so the ordering holds
    # even though the copy took 3 minutes.
    execution_started_at = "2026-09-14T09:05:00Z"
    cx.assert_causal_cutoff(snap.snapshot_consistency_at, execution_started_at)
    teardown = cx._aware(execution_started_at, label="exec")
    assert teardown >= cx._aware(snap.snapshot_consistency_at, label="consistency")
    conn.close()


def test_f_execution_start_is_after_the_planning_cutoff(tmp_path):
    conn = _world(tmp_path / "live.db")
    snap = _capture(tmp_path)
    # The production sequence mints the cutoff from the snapshot and only then
    # creates the run, so cutoff <= started_at holds by construction.
    started_at = "2026-09-14T09:00:30Z"
    cx.assert_causal_cutoff(snap.snapshot_consistency_at, started_at)
    with pytest.raises(cx.PlanningCutoffInFuture):
        cx.assert_causal_cutoff(started_at, snap.snapshot_consistency_at)
    conn.close()


# ---------------------------------------------------------------------------
# G / H. source_conn fail-closed contract
# ---------------------------------------------------------------------------


def test_g_production_freeze_without_source_conn_hard_fails(tmp_path):
    conn = _world(tmp_path / "live.db")
    args = type("A", (), {"gw": 5, "cutoff": "2026-09-14T09:00:00Z", "dry_run": False})()
    with pytest.raises(freeze.MissingSourceSnapshot) as caught:
        freeze._freeze(conn, {}, args, set(), source_conn=None, production=True)
    assert freeze.DIAG_CERTIFICATION_SOURCE_SNAPSHOT_REQUIRED in str(caught.value)
    conn.close()


def test_h_legacy_caller_default_is_explicitly_non_production(tmp_path):
    """The permissive default remains, but only for a non-production caller."""

    import inspect

    signature = inspect.signature(freeze._freeze)
    assert signature.parameters["source_conn"].default is None
    assert signature.parameters["production"].default is False
    source = Path(freeze.__file__).read_text(encoding="utf-8")
    assert "CERTIFICATION_SOURCE_SNAPSHOT_REQUIRED" in source
    assert "production freeze mode requires an" in source
    assert "explicit immutable source connection" in source
    assert "refusing to read the live database" in source


def test_h_production_with_a_snapshot_is_accepted(tmp_path):
    conn = _world(tmp_path / "live.db")
    snap = _capture(tmp_path)
    reader = es.open_snapshot(snap)
    try:
        # A production call WITH a source_conn passes the guard (it then proceeds to
        # build, which needs real config; the guard itself is what is under test).
        with pytest.raises(Exception) as caught:
            freeze._freeze(conn, {}, type("A", (), {"gw": 5, "cutoff": "x"})()[0:0] or
                           type("A", (), {"gw": 5, "cutoff": "x"})(), set(),
                           source_conn=reader, production=True)
        assert not isinstance(caught.value, freeze.MissingSourceSnapshot)
    finally:
        reader.close()
        conn.close()


# ---------------------------------------------------------------------------
# I / J. artifact flag semantics
# ---------------------------------------------------------------------------


def _artifact(**overrides):
    payload = {
        "schema": fg.CERTIFICATION_ARTIFACT_SCHEMA,
        "temporal_status": "CAUSAL",
        "dependency_validation": "COHERENT",
        "data_snapshot_sha256": "d" * 64,
        "code_snapshot_sha256": "codehash",
        "route_search_executed": False,
        "transfer_execution_performed": False,
        "decision_search_permitted": True,
        # Schema v2 (history-completeness contract): a NEW certification must carry
        # the audit and declare the gated entry point among its covered code.
        "history_completeness": {"complete": True, "blocker": None, "reasons": []},
        "certification_wiring": fg.certification_wiring_identity(),
        "certified_bundles": {
            "5": {
                "runs": {"minutes_v1": 1, "team_strength_v1": 2, "player_rates_v1": 3,
                         "xpts_v1": 4, "monte_carlo_v1": 5},
                "data_snapshot_sha256": "d" * 64,
            }
        },
    }
    payload.update(overrides)
    return payload


def _write(path: Path, payload) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_i_decision_runner_refuses_when_search_is_not_permitted(tmp_path):
    path = _write(tmp_path / "art.json", _artifact(decision_search_permitted=False,
                                                   decision_search_permitted_reasons=["horizon incomplete"]))
    with pytest.raises(fg.DecisionCertificationRequired) as caught:
        fg.load_certification_artifact(path)
    assert "horizon incomplete" in str(caught.value)

    # A missing flag is equally refused (no implicit permission).
    payload = _artifact()
    payload.pop("decision_search_permitted")
    path = _write(tmp_path / "art2.json", payload)
    with pytest.raises(fg.DecisionCertificationRequired):
        fg.load_certification_artifact(path)


def test_i_decision_runner_refuses_an_artifact_recording_prior_execution(tmp_path):
    for overrides in ({"route_search_executed": True}, {"transfer_execution_performed": True}):
        path = _write(tmp_path / "art.json", _artifact(**overrides))
        with pytest.raises(fg.DecisionCertificationRequired, match="prior execution"):
            fg.load_certification_artifact(path)


def test_j_valid_certification_flags_are_factual_and_authorising(tmp_path):
    path = _write(tmp_path / "art.json", _artifact())
    artifact = fg.load_certification_artifact(path)
    assert artifact["route_search_executed"] is False
    assert artifact["transfer_execution_performed"] is False
    assert artifact["decision_search_permitted"] is True
    # The old misleading fleet is gone from the contract.
    assert "route_search_permitted" not in artifact


def test_j_certifier_computes_the_authorisation_flag():
    """The flag is computed from the four conditions, never asserted."""

    source = Path("scripts/certify_gw5_gw8.py").read_text(encoding="utf-8")
    assert "decision_search_permitted" in source
    assert "permit_reasons" in source
    for condition in ("temporal_status", "dependency_validation", "horizon_status",
                      "data_snapshot_sha256", "assert_snapshot_unchanged"):
        assert condition in source, condition
    # The factual fields are present and false.
    assert '"route_search_executed": False' in source
    assert '"transfer_execution_performed": False' in source


def test_j_certifier_authorisation_is_executable_not_merely_present():
    """The four-condition rule must RUN, not merely appear in the source.

    The pre-existing check above is a source-text assertion, so it passed while the
    inline rule read an undefined local ``horizon_status`` -- and the certification
    artifact could never be written. Executing the extracted rule pins the behaviour
    the source text alone could not.
    """

    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import certify_gw5_gw8 as certifier

    base = dict(
        temporal_status="CAUSAL",
        dependency_validation="COHERENT",
        horizon_status=fg.DECISION_HORIZON_COMPLETE,
        data_snapshot_sha256="d" * 64,
        # The history-completeness audit is a REQUIRED argument (F5): permission
        # can never be obtained by simply not evaluating the gate.
        history_completeness={"complete": True, "blocker": None, "reasons": []},
    )
    permitted, reasons = certifier.decide_search_permission(**base)
    assert permitted is True, reasons
    assert reasons == []

    for overrides, needle in (
        ({"temporal_status": "HISTORICAL"}, "temporal_status"),
        ({"dependency_validation": "INCOHERENT"}, "dependency_validation"),
        ({"horizon_status": fg.DECISION_HORIZON_INCOMPLETE}, "horizon status"),
        ({"data_snapshot_sha256": None}, "no data snapshot identity"),
        ({"snapshot_error": "CERTIFICATION_SNAPSHOT_MUTATED"}, "CERTIFICATION_SNAPSHOT_MUTATED"),
    ):
        kwargs = dict(base)
        kwargs.update(overrides)
        permitted, reasons = certifier.decide_search_permission(**kwargs)
        assert permitted is False, (overrides, reasons)
        assert any(needle in reason for reason in reasons), (overrides, reasons)


# ---------------------------------------------------------------------------
# K / L. immutability
# ---------------------------------------------------------------------------


def test_k_snapshot_hash_mutation_still_hard_fails(tmp_path):
    conn = _world(tmp_path / "live.db")
    snap = _capture(tmp_path)
    es.assert_snapshot_unchanged(snap)
    with open(snap.path, "r+b") as handle:
        handle.seek(0)
        handle.write(b"\x00")
    with pytest.raises(es.SnapshotError) as caught:
        es.assert_snapshot_unchanged(snap)
    assert es.DIAG_SNAPSHOT_MUTATED in str(caught.value)
    with pytest.raises(es.SnapshotError):
        es.open_snapshot(snap)
    conn.close()


def test_l_runs_1_to_212_remain_immutable():
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
        run71 = conn.execute("SELECT model_version FROM projection_runs WHERE id=71").fetchone()
    finally:
        conn.close()
    assert total >= 212 and max_id >= 212
    assert non_complete == 0
    assert run71[0] == "mc_v1.2.1"
