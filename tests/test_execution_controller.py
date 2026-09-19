"""R2A tests for the production execution controller.

These tests are deterministic (injected wall clock), operate only on a temporary
database, and assert that the projection spine receives **no writes**: the
controller is operational infrastructure and must never author predictive
provenance.
"""

from __future__ import annotations

import hashlib
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone

import pytest

import fpl_brain.database as database
from fpl_brain import execution as ex
from fpl_brain.process_control import IS_WINDOWS, OwnedProcessTree, spawn_sleep_child

PROJECTION_TABLES = (
    "projection_runs",
    "frozen_predictions",
    "player_rate_projections",
    "player_fixture_xpts_projections",
    "monte_carlo_distributions",
    "team_fixture_projections",
    "team_minutes_coherence",
    "team_substitution_profiles",
)


class FakeClock:
    """A mutable aware-UTC wall clock."""

    def __init__(self, start: datetime) -> None:
        self.moment = start

    def __call__(self) -> datetime:
        return self.moment

    def advance(self, seconds: float) -> datetime:
        self.moment = self.moment + timedelta(seconds=float(seconds))
        return self.moment


@pytest.fixture
def db_conn(tmp_path):
    conn = database.connect_database(tmp_path / "fpl.db")
    yield conn
    conn.close()


@pytest.fixture
def open_controllers():
    """Track created controllers so job handles and children are always closed."""

    created: list[ex.ExecutionController] = []

    def make(conn, **kwargs):
        controller = ex.ExecutionController(conn, **kwargs)
        created.append(controller)
        return controller

    yield make
    for controller in created:
        try:
            controller.terminate_children(grace_seconds=0.2)
        except Exception:
            pass
        controller.close()


def _start_run(controller, *, hard_stop_seconds=3600, event=4, cutoff="2026-09-12T10:40:04Z", **kwargs):
    hard_stop = controller.now_dt() + timedelta(seconds=hard_stop_seconds)
    controller.create_run(
        planning_event=event, planning_cutoff=cutoff, hard_stop_at=hard_stop, **kwargs
    )
    controller.start()
    return controller.run()


def _projection_content_digest(conn) -> str:
    """Digest of every projection table's contents (used to prove no writes)."""

    digest = hashlib.sha256()
    for table in PROJECTION_TABLES:
        rows = list(conn.execute(f"SELECT * FROM {table}"))
        digest.update(f"{table}:{len(rows)}".encode())
        for row in rows:
            digest.update(repr(tuple(row)).encode())
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# A. second runner cannot acquire an active lease
# ---------------------------------------------------------------------------


def test_a_second_runner_cannot_acquire_active_lease(db_conn, open_controllers):
    first = open_controllers(db_conn)
    identity = _start_run(first)
    lease_id = first.acquire_run_lease()

    second = open_controllers(db_conn, run_uuid="second-runner")
    _start_run(second)
    with pytest.raises(ex.LeaseError, match="lease already held"):
        second.acquire_run_lease(lease_key=identity.semantic_run_key)

    # The first owner still holds it, and the second recorded nothing.
    active = first.active_lease(ex.LEASE_KIND_RUN, identity.semantic_run_key)
    assert active is not None and int(active["id"]) == lease_id
    assert second.active_lease(ex.LEASE_KIND_RUN, identity.semantic_run_key) is not None


def test_a_released_lease_frees_the_key(db_conn, open_controllers):
    first = open_controllers(db_conn)
    identity = _start_run(first)
    lease_id = first.acquire_run_lease()
    first.release_lease(lease_id)

    second = open_controllers(db_conn, run_uuid="second-runner")
    _start_run(second)
    second_lease = second.acquire_run_lease(lease_key=identity.semantic_run_key)
    assert second_lease != lease_id


# ---------------------------------------------------------------------------
# B. stale lease reclaim rule
# ---------------------------------------------------------------------------


def test_b_expired_lease_with_dead_owner_is_reclaimable(db_conn, open_controllers):
    clock = FakeClock(datetime(2026, 9, 12, 10, 0, tzinfo=timezone.utc))
    stale = open_controllers(db_conn, config=ex.ExecutionConfig(lease_ttl_seconds=60), wall_clock=clock)
    identity = _start_run(stale)
    stale.acquire_run_lease()

    clock.advance(600)  # TTL long expired
    fresh = open_controllers(db_conn, run_uuid="fresh", wall_clock=clock)
    _start_run(fresh)
    reclaimed = fresh.reclaim_stale_leases(
        ex.LEASE_KIND_RUN, identity.semantic_run_key, liveness_probe=lambda pid: False
    )
    assert len(reclaimed) == 1
    row = db_conn.execute(
        "SELECT status, note FROM execution_leases WHERE id=?", (reclaimed[0],)
    ).fetchone()
    assert row["status"] == ex.LEASE_RECLAIMED
    assert "ttl_expired_owner_dead" in row["note"]
    # The key is now acquirable.
    assert fresh.acquire_run_lease(lease_key=identity.semantic_run_key) > 0


def test_b_expired_lease_with_live_owner_is_not_reclaimed(db_conn, open_controllers):
    clock = FakeClock(datetime(2026, 9, 12, 10, 0, tzinfo=timezone.utc))
    holder = open_controllers(
        db_conn, config=ex.ExecutionConfig(lease_ttl_seconds=60), wall_clock=clock
    )
    identity = _start_run(holder)
    holder.acquire_run_lease()

    clock.advance(600)
    other = open_controllers(db_conn, run_uuid="other", wall_clock=clock)
    _start_run(other)
    # A live owner that merely missed a heartbeat must NOT be duplicated.
    reclaimed = other.reclaim_stale_leases(
        ex.LEASE_KIND_RUN, identity.semantic_run_key, liveness_probe=lambda pid: True
    )
    assert reclaimed == []
    with pytest.raises(ex.LeaseError):
        other.acquire_run_lease(lease_key=identity.semantic_run_key)


def test_b_unexpired_lease_is_never_reclaimed(db_conn, open_controllers):
    clock = FakeClock(datetime(2026, 9, 12, 10, 0, tzinfo=timezone.utc))
    holder = open_controllers(db_conn, wall_clock=clock)
    identity = _start_run(holder)
    holder.acquire_run_lease()
    other = open_controllers(db_conn, run_uuid="other", wall_clock=clock)
    _start_run(other)
    assert (
        other.reclaim_stale_leases(
            ex.LEASE_KIND_RUN, identity.semantic_run_key, liveness_probe=lambda pid: False
        )
        == []
    )


# ---------------------------------------------------------------------------
# C. hard_stop prevents new stage launch
# ---------------------------------------------------------------------------


def test_c_stage_refused_when_remaining_budget_too_small(db_conn, open_controllers):
    clock = FakeClock(datetime(2026, 9, 12, 10, 0, tzinfo=timezone.utc))
    controller = open_controllers(db_conn, wall_clock=clock)
    # 120s of budget left; MONTE_CARLO declares a 180s minimum.
    _start_run(controller, hard_stop_seconds=120)

    with pytest.raises(ex.StageSkippedInsufficientTime) as caught:
        with controller.stage(ex.MONTE_CARLO):
            pytest.fail("stage body must never run")
    assert "STAGE_SKIPPED_INSUFFICIENT_TIME" in str(caught.value)

    recorded = db_conn.execute(
        "SELECT status FROM execution_stages WHERE run_uuid=? AND stage=?",
        (controller.run_uuid, ex.MONTE_CARLO),
    ).fetchone()
    assert recorded["status"] == ex.STAGE_SKIPPED_INSUFFICIENT_TIME


def test_c_hard_stop_blocks_all_new_stages(db_conn, open_controllers):
    clock = FakeClock(datetime(2026, 9, 12, 10, 0, tzinfo=timezone.utc))
    controller = open_controllers(db_conn, wall_clock=clock)
    _start_run(controller, hard_stop_seconds=120)

    clock.advance(121)  # past the hard stop
    assert controller.remaining_seconds() < 0
    with pytest.raises(ex.DeadlineExceeded):
        controller.ensure_can_start(ex.PREFLIGHT)
    with pytest.raises(ex.DeadlineExceeded):
        with controller.stage(ex.PREFLIGHT):
            pytest.fail("stage body must never run after the hard stop")


def test_c_stage_starts_when_budget_is_sufficient(db_conn, open_controllers):
    clock = FakeClock(datetime(2026, 9, 12, 10, 0, tzinfo=timezone.utc))
    controller = open_controllers(db_conn, wall_clock=clock)
    _start_run(controller, hard_stop_seconds=3600)
    with controller.stage(ex.PREFLIGHT):
        pass
    row = db_conn.execute(
        "SELECT status FROM execution_stages WHERE run_uuid=? AND stage=?",
        (controller.run_uuid, ex.PREFLIGHT),
    ).fetchone()
    assert row["status"] == ex.STAGE_COMPLETE


def test_c_safety_margin_shrinks_remaining_budget(db_conn, open_controllers):
    clock = FakeClock(datetime(2026, 9, 12, 10, 0, tzinfo=timezone.utc))
    controller = open_controllers(
        db_conn, wall_clock=clock, config=ex.ExecutionConfig(safety_margin_seconds=50.0)
    )
    _start_run(controller, hard_stop_seconds=200)  # 150s effective
    with pytest.raises(ex.StageSkippedInsufficientTime):
        controller.ensure_can_start(ex.MONTE_CARLO)  # needs 180s


# ---------------------------------------------------------------------------
# D. cancellation stops progression to the next stage
# ---------------------------------------------------------------------------


def test_d_cancellation_stops_next_stage(db_conn, open_controllers):
    controller = open_controllers(db_conn)
    _start_run(controller, hard_stop_seconds=3600)

    with controller.stage(ex.PREFLIGHT):
        pass
    controller.request_cancel("operator stop")

    with pytest.raises(ex.RunCancelled):
        with controller.stage(ex.BASELINE):
            pytest.fail("cancelled run must not start a new stage")

    stages = {
        row["stage"]: row["status"]
        for row in db_conn.execute(
            "SELECT stage, status FROM execution_stages WHERE run_uuid=?", (controller.run_uuid,)
        )
    }
    assert stages[ex.PREFLIGHT] == ex.STAGE_COMPLETE
    assert stages[ex.BASELINE] == ex.STAGE_CANCELLED
    assert ex.STAGE_RUNNING not in stages.values()


def test_d_cancel_is_observable_from_another_process_view(db_conn, open_controllers):
    """A second controller object over the same DB sees the request."""

    runner = open_controllers(db_conn)
    _start_run(runner, hard_stop_seconds=3600)
    observer = open_controllers(db_conn, run_uuid=runner.run_uuid)
    assert observer.cancel_requested() is False
    observer.request_cancel("external stop")
    assert runner.cancel_requested() is True
    with pytest.raises(ex.RunCancelled):
        runner.check_cancel()


def test_d_cancellation_inside_stage_marks_cancelled(db_conn, open_controllers):
    controller = open_controllers(db_conn)
    _start_run(controller, hard_stop_seconds=3600)
    with pytest.raises(ex.RunCancelled):
        with controller.stage(ex.ROUTE_SEARCH):
            raise ex.RunCancelled("stop between optimizer batches")
    row = db_conn.execute(
        "SELECT status FROM execution_stages WHERE run_uuid=? AND stage=?",
        (controller.run_uuid, ex.ROUTE_SEARCH),
    ).fetchone()
    assert row["status"] == ex.STAGE_CANCELLED


# ---------------------------------------------------------------------------
# E. owned child tree terminated, unrelated process untouched
# ---------------------------------------------------------------------------


def test_e_owned_child_tree_terminated_and_unrelated_untouched(open_controllers, db_conn):
    controller = open_controllers(db_conn)
    _start_run(controller, hard_stop_seconds=3600)

    owned = controller.tree.spawn(spawn_sleep_child(60))
    controller.register_child(owned.pid, spawn_sleep_child(60))
    unrelated = subprocess.Popen(spawn_sleep_child(60))

    try:
        time.sleep(0.4)
        assert OwnedProcessTree.is_alive(owned.pid)
        assert OwnedProcessTree.is_alive(unrelated.pid)

        audit = controller.terminate_children(grace_seconds=1.0)

        assert own_pid_recorded(audit, owned.pid)
        assert not OwnedProcessTree.is_alive(owned.pid), "owned child must be reaped"
        assert OwnedProcessTree.is_alive(unrelated.pid), "unrelated process must survive"
    finally:
        unrelated.kill()
        unrelated.wait(timeout=10)


def own_pid_recorded(audit, pid):
    return any(int(entry["pid"]) == int(pid) for entry in audit)


def test_e_no_children_means_no_signals(open_controllers, db_conn):
    controller = open_controllers(db_conn)
    _start_run(controller, hard_stop_seconds=3600)
    assert controller.terminate_children(grace_seconds=0) == []


def test_e_child_audit_list_has_pid_and_argv(open_controllers, db_conn):
    controller = open_controllers(db_conn)
    _start_run(controller, hard_stop_seconds=3600)
    child = controller.tree.spawn(spawn_sleep_child(30))
    controller.register_child(child.pid, spawn_sleep_child(30))
    audit = controller.child_audit_list()
    assert audit and int(audit[0]["pid"]) == child.pid
    assert audit[0]["argv"], "argv must be recorded for auditability"
    controller.terminate_children(grace_seconds=0.5)


@pytest.mark.skipif(IS_WINDOWS, reason="POSIX reap semantics are not used on Windows")
def test_e_already_exited_owned_child_is_reaped_and_audited():
    tree = OwnedProcessTree(label="already-exited")
    child = tree.spawn([sys.executable, "-c", "pass"])
    try:
        child.wait(timeout=10)
        audit = tree.terminate_all(grace_seconds=0)
        record = next(entry for entry in audit if int(entry["pid"]) == child.pid)
        assert record["exit_status"] == "exited"
        assert not OwnedProcessTree.is_alive(child.pid)
    finally:
        tree.close()


@pytest.mark.skipif(IS_WINDOWS, reason="POSIX forced-group-kill semantics are not used on Windows")
def test_e_forced_kill_path_is_bounded_and_reaps_owned_child():
    tree = OwnedProcessTree(label="forced-kill")
    code = (
        "import signal,sys,time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "print('ready', flush=True); time.sleep(60)"
    )
    child = tree.spawn([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "ready"
        started = time.monotonic()
        audit = tree.terminate_all(grace_seconds=0.05)
        elapsed = time.monotonic() - started
        record = next(entry for entry in audit if int(entry["pid"]) == child.pid)
        assert elapsed < 5.0
        assert record["exit_status"] == "terminated"
        assert child.poll() is not None
        assert not OwnedProcessTree.is_alive(child.pid)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)
        tree.close()


# ---------------------------------------------------------------------------
# F. second DB writer is serialized/refused
# ---------------------------------------------------------------------------


def test_f_second_writer_lease_refused_then_granted_after_release(db_conn, open_controllers):
    first = open_controllers(db_conn)
    _start_run(first)
    first_lease = first.acquire_writer_lease()

    second = open_controllers(db_conn, run_uuid="writer-two")
    _start_run(second)
    with pytest.raises(ex.LeaseError):
        second.acquire_writer_lease()

    first.release_lease(first_lease)
    assert second.acquire_writer_lease() > 0


def test_f_distinct_writer_keys_coexist(db_conn, open_controllers):
    """Read-only workers are not the writer; only the writer key is exclusive."""

    writer = open_controllers(db_conn)
    _start_run(writer)
    writer.acquire_writer_lease("sqlite-writer")
    other = open_controllers(db_conn, run_uuid="reader")
    _start_run(other)
    # A different key is a different lease, so it is granted.
    assert other.acquire_writer_lease("report-reader") > 0


# ---------------------------------------------------------------------------
# G. restart reuses a completed stage rather than duplicating it
# ---------------------------------------------------------------------------


def test_g_restart_reuses_completed_stage(db_conn, open_controllers):
    first = open_controllers(db_conn)
    identity = _start_run(first, event=4)
    with first.stage(ex.MINUTES):
        pass
    first.finish(ex.RUN_COMPLETE)

    restarted = open_controllers(db_conn, run_uuid="restarted")
    _start_run(restarted, event=4, semantic_key=identity.semantic_run_key)
    assert restarted.classify_stage(ex.MINUTES) == ex.REUSE_COMPLETE
    assert restarted.classify_stage(ex.ROUTE_SEARCH) == ex.NOT_STARTED


def test_g_projection_probe_avoids_recomputing_predictive_work(db_conn, open_controllers):
    controller = open_controllers(db_conn)
    _start_run(controller, event=4)
    calls: list[str] = []

    def probe(family: str):
        calls.append(family)
        return 133 if family == "xpts_v1" else None

    plan = controller.plan_restart(
        (ex.PREFLIGHT, ex.BASELINE, ex.XPTS, ex.ROUTE_SEARCH), projection_probe=probe
    )
    assert plan[ex.XPTS] == ex.REUSE_COMPLETE
    assert plan[ex.BASELINE] == ex.NOT_STARTED
    assert plan[ex.PREFLIGHT] == ex.NOT_STARTED
    assert "xpts_v1" in calls


def test_g_interrupted_writer_is_blocked_not_silently_retried(db_conn, open_controllers):
    controller = open_controllers(db_conn)
    _start_run(controller, event=4)
    key = ex.stage_semantic_key(controller.run().semantic_run_key, ex.MINUTES)
    db_conn.execute(
        """INSERT INTO execution_stages(run_uuid, stage, semantic_key, status, created_at)
           VALUES (?,?,?,?,?)""",
        (controller.run_uuid, ex.MINUTES, key, ex.STAGE_RUNNING, controller.now()),
    )
    db_conn.commit()
    assert controller.classify_stage(ex.MINUTES) == ex.BLOCKED_PARTIAL_STATE


def test_g_completed_stage_identity_is_unique(db_conn, open_controllers):
    """Two runs cannot both record COMPLETE for the same semantic stage."""

    first = open_controllers(db_conn)
    identity = _start_run(first, event=4)
    with first.stage(ex.PLAYER_RATES):
        pass

    second = open_controllers(db_conn, run_uuid="second")
    _start_run(second, event=4, semantic_key=identity.semantic_run_key)
    with pytest.raises(sqlite3.IntegrityError):
        with second.stage(ex.PLAYER_RATES):
            pass


# ---------------------------------------------------------------------------
# H. cancellation leaves no execution state marked RUNNING
# ---------------------------------------------------------------------------


def test_h_cancellation_leaves_nothing_running(db_conn, open_controllers):
    controller = open_controllers(db_conn)
    _start_run(controller, hard_stop_seconds=3600)
    with controller.stage(ex.PREFLIGHT):
        pass
    with controller.stage(ex.BASELINE):
        pass
    controller.request_cancel("deadline overrun")
    with pytest.raises(ex.RunCancelled):
        with controller.stage(ex.MINUTES):
            pytest.fail("unreachable")
    controller.acknowledge_cancel()

    run = controller.run()
    assert run.status == ex.RUN_CANCELLED
    assert run.finished_at is not None

    running_runs = db_conn.execute(
        "SELECT COUNT(*) FROM execution_runs WHERE status IN ('RUNNING','CANCEL_REQUESTED')"
    ).fetchone()[0]
    assert running_runs == 0

    running_stages = db_conn.execute(
        "SELECT COUNT(*) FROM execution_stages WHERE status='RUNNING'"
    ).fetchone()[0]
    assert running_stages == 0

    assert controller.active_lease(ex.LEASE_KIND_RUN, run.semantic_run_key) is None


def test_h_failed_run_releases_leases_and_is_terminal(db_conn, open_controllers):
    controller = open_controllers(db_conn)
    identity = _start_run(controller, hard_stop_seconds=3600)
    controller.acquire_run_lease()
    controller.acquire_writer_lease()
    controller.finish(ex.RUN_FAILED, "stage exploded")
    assert controller.run().status == ex.RUN_FAILED
    assert controller.active_lease(ex.LEASE_KIND_RUN, identity.semantic_run_key) is None
    assert controller.active_lease(ex.LEASE_KIND_WRITER, "sqlite-writer") is None
    with pytest.raises(ex.ExecutionError):
        controller.finish("SOMETHING_ELSE")


def test_h_run_status_check_constraint_is_enforced(db_conn, open_controllers):
    controller = open_controllers(db_conn)
    _start_run(controller, hard_stop_seconds=3600)
    with pytest.raises(sqlite3.IntegrityError):
        db_conn.execute(
            "UPDATE execution_runs SET status='NONSENSE' WHERE run_uuid=?", (controller.run_uuid,)
        )
    db_conn.rollback()


# ---------------------------------------------------------------------------
# I. projection tables receive no writes
# ---------------------------------------------------------------------------


def test_i_projection_spine_is_untouched_by_controller_operations(db_conn, open_controllers):
    before = _projection_content_digest(db_conn)

    controller = open_controllers(db_conn)
    identity = _start_run(controller, hard_stop_seconds=3600)
    controller.acquire_run_lease()
    controller.acquire_writer_lease()
    child = controller.tree.spawn(spawn_sleep_child(20))
    controller.register_child(child.pid, spawn_sleep_child(20))
    with controller.stage(ex.PREFLIGHT):
        pass
    with controller.stage(ex.MINUTES):
        pass
    controller.request_cancel("test")
    controller.acknowledge_cancel()
    controller.terminate_children(grace_seconds=0.5)
    controller.classify_stage(ex.XPTS)
    controller.plan_restart((ex.PREFLIGHT, ex.MINUTES))

    after = _projection_content_digest(db_conn)
    assert before == after
    for table in PROJECTION_TABLES:
        assert db_conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0

    # ...while operational provenance was written.
    assert db_conn.execute("SELECT COUNT(*) FROM execution_runs").fetchone()[0] >= 1
    assert db_conn.execute("SELECT COUNT(*) FROM execution_leases").fetchone()[0] >= 2
    assert db_conn.execute("SELECT COUNT(*) FROM execution_stages").fetchone()[0] >= 2
    assert db_conn.execute("SELECT COUNT(*) FROM execution_children").fetchone()[0] >= 1
    assert identity.semantic_run_key


def test_i_projection_probe_is_read_only(db_conn, open_controllers):
    before = _projection_content_digest(db_conn)
    assert (
        ex.find_completed_projection_run(
            db_conn, model_family="xpts_v1", planning_event=4, data_cutoff="2026-09-12T10:40:04Z"
        )
        is None
    )
    assert _projection_content_digest(db_conn) == before


# ---------------------------------------------------------------------------
# J. UTC timestamps are aware and never naive-Z
# ---------------------------------------------------------------------------


def test_j_format_utc_refuses_naive_datetime():
    with pytest.raises(ValueError, match="naive"):
        ex.format_utc(datetime(2026, 9, 12, 10, 40, 0))
    with pytest.raises(TypeError):
        ex.format_utc("2026-09-12T10:40:00Z")


def test_j_format_utc_normalises_to_z_and_roundtrips():
    aware = datetime(2026, 9, 12, 12, 40, 0, tzinfo=timezone(timedelta(hours=2)))
    text = ex.format_utc(aware)
    assert text == "2026-09-12T10:40:00Z"
    parsed = ex.parse_utc_dt(text)
    assert parsed is not None and parsed.utcoffset() == timedelta(0)


def test_j_utc_now_is_aware():
    moment = ex.utc_now_dt()
    assert moment.tzinfo is not None
    assert moment.utcoffset() == timedelta(0)
    assert ex.format_utc(moment).endswith("Z")


def test_j_controller_timestamps_are_utc_aware(db_conn, open_controllers):
    controller = open_controllers(db_conn)
    _start_run(controller, hard_stop_seconds=3600)
    stamp = controller.now()
    assert stamp.endswith("Z")
    assert ex.parse_utc_dt(stamp).utcoffset() == timedelta(0)
    row = controller.run()
    for field in (row.started_at, row.heartbeat_at, row.hard_stop_at):
        assert field.endswith("Z")
        assert ex.parse_utc_dt(field).utcoffset() == timedelta(0)


def test_j_controller_rejects_a_naive_wall_clock(db_conn):
    naive_controller = ex.ExecutionController(db_conn, wall_clock=lambda: datetime(2026, 9, 12, 10, 0))
    with pytest.raises(ValueError, match="aware"):
        naive_controller.now_dt()
    naive_controller.close()


def test_j_naive_hard_stop_text_is_read_as_utc_not_local(db_conn, open_controllers):
    controller = open_controllers(db_conn)
    controller.create_run(planning_event=4, planning_cutoff="2026-09-12T10:40:04Z",
                          hard_stop_at="2026-09-12T12:30:00Z")
    assert controller.run().hard_stop_at == "2026-09-12T12:30:00Z"
    assert controller.hard_stop().utcoffset() == timedelta(0)


# ---------------------------------------------------------------------------
# Regression for the Sep-12 incident shape
# ---------------------------------------------------------------------------


def test_regression_cancel_after_gw4_blocks_gw6_and_gw7(db_conn, open_controllers):
    """The Sep-12 shape: cancel after GW4 -> GW6/GW7 must not begin afterwards.

    Runs 135-147 (GW6) and 148-160 (GW7) were generated after the GW4 task was
    cancelled because nothing checked a cancellation token.  Here the token is
    checked before every event, so GW6 and GW7 never start.
    """

    controller = open_controllers(db_conn)
    _start_run(controller, event=4, hard_stop_seconds=7200)

    def event_stage(event: int):
        return controller.stage(ex.MINUTES, discriminator=f"event={event}")

    started: list[int] = []
    with event_stage(4):
        started.append(4)

    # The operator cancels once GW4 work is done.
    controller.request_cancel("GW4 complete; stop the batch")

    for event in (6, 7):
        with pytest.raises(ex.RunCancelled):
            with event_stage(event):
                started.append(event)

    assert started == [4], "no GW6/GW7 work may start after cancellation"

    recorded = {
        (row["stage"], row["detail_json"] or "", row["status"])
        for row in db_conn.execute(
            "SELECT stage, detail_json, status FROM execution_stages WHERE run_uuid=?",
            (controller.run_uuid,),
        )
    }
    statuses = [
        row["status"]
        for row in db_conn.execute(
            "SELECT status FROM execution_stages WHERE run_uuid=? ORDER BY id",
            (controller.run_uuid,),
        )
    ]
    assert statuses.count(ex.STAGE_COMPLETE) == 1
    assert statuses.count(ex.STAGE_CANCELLED) == 2
    assert ex.STAGE_RUNNING not in statuses

    controller.acknowledge_cancel()
    assert controller.run().status == ex.RUN_CANCELLED
    assert recorded  # both blocked events were recorded, not silently dropped


def test_regression_duplicate_runner_is_stopped_by_the_lease(db_conn, open_controllers):
    """A second identical batch invocation cannot start while the first holds the key."""

    first = open_controllers(db_conn)
    identity = _start_run(first, event=4, cutoff="2026-09-12T10:40:04Z")
    first.acquire_run_lease()

    duplicate = open_controllers(db_conn, run_uuid="duplicate-batch")
    duplicate.create_run(
        planning_event=4,
        planning_cutoff="2026-09-12T10:40:04Z",
        hard_stop_at=duplicate.now_dt() + timedelta(hours=2),
    )
    duplicate.start()
    assert duplicate.run().semantic_run_key == identity.semantic_run_key
    with pytest.raises(ex.LeaseError):
        duplicate.acquire_run_lease()


def test_semantic_key_discriminates_distinct_work():
    """Distinct semantic work must not collide (the R1 baseline-kind lesson)."""

    raw = ex.stage_semantic_key("run-key", ex.PLAYER_RATES, "RAW_CURRENT_RATE")
    prior = ex.stage_semantic_key("run-key", ex.PLAYER_RATES, "PRIOR_ONLY_RATE")
    assert raw != prior
    assert ex.stage_semantic_key("run-key", ex.PLAYER_RATES, "RAW_CURRENT_RATE") == raw


def test_schema_migration_is_additive_and_idempotent(tmp_path):
    path = tmp_path / "fpl.db"
    conn = database.connect_database(path)
    assert conn.execute("SELECT value FROM schema_meta WHERE key='schema_version'").fetchone()[0] == str(
        database.SCHEMA_VERSION
    )
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'execution%'")
    }
    assert tables == {"execution_runs", "execution_leases", "execution_children", "execution_stages"}
    before = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    conn.close()

    conn = database.connect_database(path)
    after = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert before == after
    conn.close()


def test_busy_timeout_and_readonly_factory(tmp_path):
    path = tmp_path / "fpl.db"
    conn = database.connect_database(path)
    assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 30000
    conn.close()

    reader = database.connect_readonly_database(path)
    try:
        assert reader.execute("PRAGMA query_only").fetchone()[0] == 1
        reader.execute("SELECT COUNT(*) FROM projection_runs").fetchone()
        with pytest.raises(sqlite3.OperationalError):
            reader.execute("CREATE TABLE nope (id INTEGER)")
    finally:
        reader.close()


# ---------------------------------------------------------------------------
# production_run_guard — the single required entry point
# ---------------------------------------------------------------------------


def test_guard_acquires_leases_completes_and_releases(db_conn):
    with ex.production_run_guard(
        db_conn,
        planning_event=4,
        planning_cutoff="2026-09-12T10:40:04Z",
        label="test-guard",
        families=["minutes_v1"],
    ) as guard:
        run_uuid = guard.run_uuid
        assert guard.run().status == ex.RUN_RUNNING
        assert guard.active_lease(ex.LEASE_KIND_RUN, guard.run().semantic_run_key) is not None
        assert guard.active_lease(ex.LEASE_KIND_WRITER, "sqlite-writer") is not None

    row = db_conn.execute("SELECT status FROM execution_runs WHERE run_uuid=?", (run_uuid,)).fetchone()
    assert row["status"] == ex.RUN_COMPLETE
    assert (
        db_conn.execute("SELECT COUNT(*) FROM execution_leases WHERE status='ACTIVE'").fetchone()[0] == 0
    )


def test_guard_marks_failed_and_releases_on_exception(db_conn):
    run_uuid = None
    with pytest.raises(RuntimeError, match="boom"):
        with ex.production_run_guard(db_conn, planning_event=4, label="test-guard-fail") as guard:
            run_uuid = guard.run_uuid
            raise RuntimeError("boom")
    row = db_conn.execute("SELECT status, failure_reason FROM execution_runs WHERE run_uuid=?", (run_uuid,)).fetchone()
    assert row["status"] == ex.RUN_FAILED
    assert "RuntimeError" in row["failure_reason"]
    assert db_conn.execute("SELECT COUNT(*) FROM execution_leases WHERE status='ACTIVE'").fetchone()[0] == 0


def test_guard_refuses_a_second_runner_for_same_semantic_work(db_conn):
    """A duplicate top-level runner cannot start while the first holds the key."""

    with ex.production_run_guard(
        db_conn,
        planning_event=4,
        planning_cutoff="2026-09-12T10:40:04Z",
        families=["minutes_v1"],
    ):
        with pytest.raises(ex.LeaseError):
            with ex.production_run_guard(
                db_conn,
                planning_event=4,
                planning_cutoff="2026-09-12T10:40:04Z",
                families=["minutes_v1"],
            ):
                pytest.fail("duplicate guard body must never run")


def test_guard_hard_stop_is_the_earlier_of_deadline_and_ceiling(db_conn):
    clock = FakeClock(datetime(2026, 9, 12, 10, 0, tzinfo=timezone.utc))
    with ex.production_run_guard(
        db_conn,
        planning_event=4,
        label="deadline-bound",
        official_deadline="2026-09-12T10:30:00Z",  # 30 min away
        max_wall_seconds=6 * 3600,
        wall_clock=clock,
    ) as guard:
        # The official deadline wins over the 6h ceiling.
        assert guard.run().hard_stop_at == "2026-09-12T10:30:00Z"
        assert guard.remaining_seconds() == pytest.approx(1800.0, abs=1.0)


def test_guard_allow_late_uses_the_wall_ceiling(db_conn):
    clock = FakeClock(datetime(2026, 9, 12, 14, 0, tzinfo=timezone.utc))
    with ex.production_run_guard(
        db_conn,
        planning_event=4,
        label="late-repro",
        official_deadline="2026-09-12T12:30:00Z",  # already passed
        max_wall_seconds=3600,
        allow_late=True,
        wall_clock=clock,
    ) as guard:
        assert guard.run().hard_stop_at == "2026-09-12T15:00:00Z"


def test_guard_never_touches_the_projection_spine(db_conn):
    before = _projection_content_digest(db_conn)
    with ex.production_run_guard(db_conn, planning_event=4, label="guard-no-writes"):
        pass
    assert _projection_content_digest(db_conn) == before
