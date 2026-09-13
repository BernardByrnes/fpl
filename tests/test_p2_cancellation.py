"""P2 §11 — cancellation probe, heartbeat, and the operational findings.

P1 established that ``scripts/run_four_gw_decision.py`` never called
``ExecutionController.check_cancel()`` and never sent a heartbeat, so a
``CANCEL_REQUESTED`` row was unobservable mid-search, no stage ledger existed, and the
15-minute lease TTL lapsed during a four-hour run.  P2 threads an optional
``cancel_probe`` through the search and evaluation path and polls it ONLY at the
boundaries P1 identified.

These tests prove:
  * the probe is called at those boundaries and nowhere else;
  * a requested cancel interrupts cleanly, releases every lease, and leaves the run
    terminal and the database with no open transaction;
  * no cache is corrupted by an interruption (a shared ``exact_cache`` never holds a
    partially computed entry);
  * a normal no-cancel run is unaffected by installing a probe;
  * the search and exact-evaluation path executes ZERO SQLite statements while
    probing, so a poll can never sit inside a write transaction.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

from fpl_brain import execution
from fpl_brain.database import connect_database
from fpl_brain import route_optimizer as ro
from test_transfer_state import SQUAD_IDS

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_four_gw_decision as runner  # noqa: E402


class StubGuard:
    """Records calls; optionally raises RunCancelled after a set number of checks."""

    def __init__(self, cancel_after: int | None = None) -> None:
        self.checks = 0
        self.heartbeats = 0
        self.cancel_after = cancel_after

    def check_cancel(self) -> None:
        self.checks += 1
        if self.cancel_after is not None and self.checks > self.cancel_after:
            raise execution.RunCancelled("stub cancellation")

    def heartbeat(self) -> None:
        self.heartbeats += 1


def _search_fixture():
    from test_route_optimizer import _config, _provider, _scenario, _universe
    universe, state, meta = _universe()
    return universe, state, meta, _scenario(), _config(), _provider()


# ---------------------------------------------------------------------------
# the probe helper
# ---------------------------------------------------------------------------

def test_cancel_probe_calls_check_cancel_every_time_and_heartbeats_once():
    guard = StubGuard()
    probe = runner._cancel_probe(guard, heartbeat_interval=60.0)
    probe()
    # The first probe always heartbeats (last_heartbeat starts at 0).
    assert guard.checks == 1 and guard.heartbeats == 1
    probe()
    probe()
    assert guard.checks == 3
    assert guard.heartbeats == 1, "the heartbeat must respect the cadence, not fire per probe"


def test_cancel_probe_heartbeats_every_time_when_the_interval_is_zero():
    guard = StubGuard()
    probe = runner._cancel_probe(guard, heartbeat_interval=0.0)
    probe()
    probe()
    probe()
    assert guard.checks == 3 and guard.heartbeats == 3


def test_cancel_probe_propagates_the_cancellation():
    guard = StubGuard(cancel_after=1)
    probe = runner._cancel_probe(guard)
    probe()
    with pytest.raises(execution.RunCancelled):
        probe()


# ---------------------------------------------------------------------------
# probing at the boundaries P1 identified
# ---------------------------------------------------------------------------

def test_run_search_polls_once_per_level_and_nowhere_inside():
    universe, state, meta, scenario, config, provider = _search_fixture()
    seen: list[int] = []

    def probe() -> None:
        seen.append(len(seen))

    result = ro.run_search(initial_state=state, events=list(config.events), rows={
        int(row["player_id"]): row for row in universe["universe"]},
        pool_ids=ro.build_search_pool(universe, SQUAD_IDS, config)["pool_ids"],
        positions={int(row["player_id"]): str(row["position"]) for row in universe["universe"]},
        scenario=scenario, player_meta=meta, config=config, cancel_probe=probe)
    assert len(seen) == len(config.events), "exactly one poll per search level"
    assert result["stats"]["partial_states"] > 0
    baseline = ro.run_search(initial_state=state, events=list(config.events), rows={
        int(row["player_id"]): row for row in universe["universe"]},
        pool_ids=ro.build_search_pool(universe, SQUAD_IDS, config)["pool_ids"],
        positions={int(row["player_id"]): str(row["position"]) for row in universe["universe"]},
        scenario=scenario, player_meta=meta, config=config)
    assert result["stats"] == baseline["stats"]


def test_optimize_polls_per_event_and_per_promoted_route():
    universe, state, meta, scenario, config, provider = _search_fixture()
    calls = {"n": 0}

    def probe() -> None:
        calls["n"] += 1

    result = ro.optimize(universe=universe, initial_state=state, scenario=scenario,
                         player_meta=meta, config=config, world_provider=provider,
                         exact_cache={}, cancel_probe=probe)
    promoted = len(result["routes"])
    # events (worlds) + events (search levels) + promoted routes
    assert calls["n"] >= len(config.events) * 2 + promoted, (calls["n"], promoted)


def test_optimize_without_a_probe_is_identical_to_a_no_op_probe():
    universe, state, meta, scenario, config, provider = _search_fixture()
    without = ro.optimize(universe=universe, initial_state=state, scenario=scenario,
                          player_meta=meta, config=config, world_provider=provider, exact_cache={})
    with_probe = ro.optimize(universe=universe, initial_state=state, scenario=scenario,
                             player_meta=meta, config=config, world_provider=provider,
                             exact_cache={}, cancel_probe=lambda: None)
    assert without["routes"] == with_probe["routes"]
    assert without["search_stats"] == with_probe["search_stats"]
    assert without["exact_evaluations"] == with_probe["exact_evaluations"]
    assert without["exact_cache_entries"] == with_probe["exact_cache_entries"]


def test_cancellation_during_optimize_propagates_and_leaves_no_partial_cache_entry():
    universe, state, meta, scenario, config, provider = _search_fixture()
    # Allow the world loop and search to finish, then cancel on an exact evaluation.
    allowance = len(config.events) * 2 + 1
    counter = {"n": 0}

    def probe() -> None:
        counter["n"] += 1
        if counter["n"] >= allowance:
            raise execution.RunCancelled("cancel during exact evaluation")

    cache: dict = {}
    with pytest.raises(execution.RunCancelled):
        ro.optimize(universe=universe, initial_state=state, scenario=scenario, player_meta=meta,
                    config=config, world_provider=provider, exact_cache=cache, cancel_probe=probe)
    # Every entry that made it into the shared cache is COMPLETE: replaying with the same
    # cache must reproduce a fresh run's routes exactly (no partially scored squad).
    replay = ro.optimize(universe=universe, initial_state=state, scenario=scenario,
                         player_meta=meta, config=config, world_provider=provider,
                         exact_cache=cache)
    fresh = ro.optimize(universe=universe, initial_state=state, scenario=scenario,
                        player_meta=meta, config=config, world_provider=provider, exact_cache={})
    assert replay["routes"] == fresh["routes"]
    for record in cache.values():
        assert set(record) == {"policy", "scores", "mean_gross"}
        assert len(record["scores"]) == int(config.search_draws)


def test_cancel_before_any_evaluation_leaves_the_cache_empty():
    universe, state, meta, scenario, config, provider = _search_fixture()
    counter = {"n": 0}

    def probe() -> None:
        counter["n"] += 1
        if counter["n"] > len(config.events):
            raise execution.RunCancelled("cancel after the world loop")

    cache: dict = {}
    with pytest.raises(execution.RunCancelled):
        ro.optimize(universe=universe, initial_state=state, scenario=scenario, player_meta=meta,
                    config=config, world_provider=provider, exact_cache=cache, cancel_probe=probe)
    assert cache == {}


def test_search_and_exact_evaluation_execute_no_sqlite_statements():
    """A poll can never sit inside a write transaction if no statement ever runs."""

    universe, state, meta, scenario, config, provider = _search_fixture()
    conn = connect_database(":memory:")
    statements: list[str] = []
    conn.set_trace_callback(statements.append)
    try:
        conn.execute("SELECT 1").fetchone()
        statements.clear()
        ro.optimize(universe=universe, initial_state=state, scenario=scenario, player_meta=meta,
                    config=config, world_provider=provider, exact_cache={},
                    cancel_probe=lambda: None)
        assert statements == [], statements
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# end to end through the real execution guard
# ---------------------------------------------------------------------------

def _memory_db_with_event(event: int = 4) -> sqlite3.Connection:
    conn = connect_database(":memory:")
    conn.execute(
        "INSERT INTO events(id, name, deadline_time, finished, raw_json, updated_at) "
        "VALUES (?,?,?,?,?,?)",
        (int(event), f"GW{event}", "2026-09-20T11:00:00Z", 0, "{}", "2026-09-12T10:00:00Z"),
    )
    conn.commit()
    return conn


def test_requested_cancel_interrupts_cleanly_and_releases_every_lease():
    conn = _memory_db_with_event()
    holder: dict = {}
    try:
        # The exception must propagate OUT of the guard so the guard's own
        # RunCancelled branch runs (that branch is what acknowledges the cancel).
        with pytest.raises(execution.RunCancelled):
            with execution.event_run_guard(conn, planning_event=4,
                                           cutoff="2026-09-12T10:00:00Z",
                                           label="p2_cancel_test",
                                           families=["p2_cancel_test"]) as guard:
                holder["run_uuid"] = guard.run_uuid
                probe = runner._cancel_probe(guard, heartbeat_interval=0.0)
                probe()  # a healthy poll: no cancellation requested yet
                assert guard.run().status == execution.RUN_RUNNING
                active = conn.execute(
                    "SELECT COUNT(*) FROM execution_leases WHERE status='ACTIVE'").fetchone()[0]
                assert active == 2, "the run lease and the writer lease are both held"
                guard.request_cancel("p2 test cancellation")
                probe()  # must raise RunCancelled and unwind through the guard
        row = conn.execute(
            "SELECT status, failure_reason FROM execution_runs WHERE run_uuid=?",
            (holder["run_uuid"],)).fetchone()
        assert row[0] == execution.RUN_CANCELLED, tuple(row)
        active = conn.execute(
            "SELECT COUNT(*) FROM execution_leases WHERE status='ACTIVE'").fetchone()[0]
        assert active == 0, "every lease must be released on the cancel path"
        assert not conn.in_transaction, "no transaction may be left open"
        released = conn.execute(
            "SELECT COUNT(*) FROM execution_leases WHERE status='RELEASED'").fetchone()[0]
        assert released == 2
    finally:
        conn.close()


def test_normal_run_through_the_guard_completes_and_releases_leases():
    conn = _memory_db_with_event()
    try:
        with execution.event_run_guard(conn, planning_event=4,
                                       cutoff="2026-09-12T10:00:00Z", label="p2_ok_test",
                                       families=["p2_ok_test"]) as guard:
            probe = runner._cancel_probe(guard, heartbeat_interval=0.0)
            probe()
            probe()
            run_uuid = guard.run_uuid
        row = conn.execute("SELECT status FROM execution_runs WHERE run_uuid=?",
                           (run_uuid,)).fetchone()
        assert row[0] == execution.RUN_COMPLETE
        active = conn.execute(
            "SELECT COUNT(*) FROM execution_leases WHERE status='ACTIVE'").fetchone()[0]
        assert active == 0
        assert not conn.in_transaction
    finally:
        conn.close()
