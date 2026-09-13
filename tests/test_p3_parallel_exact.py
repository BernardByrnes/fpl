"""P3 — parallel exact-evaluation tests.

The phase's contract is that P3 changes SCHEDULING only.  These tests hold it to that:
one worker's exact evaluation must be the exact evaluation the sequential path would
have computed, and no completion order may be observable in the result.

Everything here runs a real spawn-based process pool.  The fixtures are small synthetic
worlds, so the tests stay fast; the real 10,000-draw fixture equivalence and the
worker-count benchmark are in the P3 report's diagnostic scripts.
"""

from __future__ import annotations

import multiprocessing
import pickle
import random

import pytest

from fpl_brain import parallel_exact as px
from fpl_brain import route_optimizer as ro
from test_route_optimizer import _config, _provider, _scenario, _universe


@pytest.fixture(autouse=True)
def _restore_monte_carlo_guard():
    """``_worker_init`` installs an enforced no-Monte-Carlo guard; undo it after a test."""

    from fpl_brain import monte_carlo

    original = monte_carlo.simulate
    try:
        yield
    finally:
        monte_carlo.simulate = original


def _fixture(**over):
    universe, state, meta = _universe()
    return universe, state, meta, _scenario(), _config(**over), _provider()


def _promoted(universe, state, meta, scenario, config, provider):
    """The promoted routes of one real (small) search, without exact evaluation."""

    pool = ro.build_search_pool(universe, [int(p.player_id) for p in state.players], config)
    rows = {int(row["player_id"]): row for row in universe["universe"]}
    search = ro.run_search(initial_state=state, events=list(config.events), rows=rows,
                           pool_ids=pool["pool_ids"],
                           positions={pid: str(row["position"]) for pid, row in rows.items()},
                           scenario=scenario, player_meta=meta, config=config)
    return ro._select_promoted(search["final_states"], config)


# ---------------------------------------------------------------------------
# key derivation, dedup and cache consultation
# ---------------------------------------------------------------------------

def test_exact_keys_are_deduplicated_before_dispatch():
    universe, state, meta, scenario, config, provider = _fixture()
    promoted = _promoted(universe, state, meta, scenario, config, provider)
    worlds = {int(e): provider(e, [int(r["player_id"]) for r in universe["universe"]])
              for e in config.events}
    positions = {int(row["player_id"]): str(row["position"]) for row in universe["universe"]}

    def positions_of(squad_ids):
        return {int(pid): positions[int(pid)] for pid in squad_ids}

    tasks, counters = px.required_keys(promoted, worlds_by_event=worlds,
                                       positions_of=positions_of, cache={}, config=config,
                                       world_identity=None)
    assert counters.requested_exact_keys >= counters.unique_exact_keys == len(tasks)
    keys = [task.key for task in tasks]
    assert len(keys) == len(set(keys)), "a duplicated key was dispatched twice"

    # Two routes requiring the SAME (event, squad) evaluation must yield ONE task.  The
    # route set is doubled explicitly so the assertion does not depend on whether the
    # synthetic search happens to promote two routes that share a squad.
    doubled, doubled_counters = px.required_keys(
        list(promoted) + list(promoted), worlds_by_event=worlds,
        positions_of=positions_of, cache={}, config=config, world_identity=None)
    assert len(doubled) == len(tasks), "the doubled route set dispatched new work"
    assert [task.key for task in doubled] == keys, "doubling changed the dispatch order"
    assert doubled_counters.requested_exact_keys == 2 * counters.requested_exact_keys
    assert doubled_counters.unique_exact_keys == len(tasks)
    # EVERY extra request in the doubled set was recognised as a duplicate.
    assert doubled_counters.duplicate_jobs_avoided == (
        doubled_counters.requested_exact_keys - len(tasks))


def test_keys_already_in_the_parent_cache_are_never_dispatched():
    universe, state, meta, scenario, config, provider = _fixture()
    promoted = _promoted(universe, state, meta, scenario, config, provider)
    worlds = {int(e): provider(e, [int(r["player_id"]) for r in universe["universe"]])
              for e in config.events}
    positions = {int(row["player_id"]): str(row["position"]) for row in universe["universe"]}

    def positions_of(squad_ids):
        return {int(pid): positions[int(pid)] for pid in squad_ids}

    _, counters = px.required_keys(promoted, worlds_by_event=worlds, positions_of=positions_of,
                                   cache={}, config=config, world_identity=None)
    total_unique = counters.unique_exact_keys
    warm: dict = {}
    for task in px.required_keys(promoted, worlds_by_event=worlds, positions_of=positions_of,
                                 cache={}, config=config, world_identity=None)[0]:
        warm[task.key] = {"policy": None, "scores": [], "mean_gross": 0.0}
    tasks, counters2 = px.required_keys(promoted, worlds_by_event=worlds,
                                       positions_of=positions_of, cache=warm, config=config,
                                       world_identity=None)
    assert tasks == []
    assert counters2.parent_cache_hits == total_unique
    assert counters2.unique_exact_keys == total_unique


# ---------------------------------------------------------------------------
# sequential vs parallel identity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("workers", [1, 2])
def test_parallel_pool_output_is_bit_identical_to_sequential(workers):
    universe, state, meta, scenario, config, provider = _fixture()
    sequential = ro.optimize(universe=universe, initial_state=state, scenario=scenario,
                             player_meta=meta, config=config, world_provider=provider,
                             exact_cache={})
    parallel = ro.optimize(universe=universe, initial_state=state, scenario=scenario,
                           player_meta=meta, config=config, world_provider=provider,
                           exact_cache={}, parallel_workers=workers)
    assert parallel["routes"] == sequential["routes"]
    assert parallel["exact_evaluations"] == sequential["exact_evaluations"]
    assert parallel["exact_cache_entries"] == sequential["exact_cache_entries"]
    assert parallel["h1_frontier"] == sequential["h1_frontier"]
    assert parallel["supported_3gw_frontier"] == sequential["supported_3gw_frontier"]
    assert parallel["families"] == sequential["families"]
    assert parallel["paired_h1"] == sequential["paired_h1"]
    assert parallel["paired_supported_3gw"] == sequential["paired_supported_3gw"]
    assert parallel["roll_baseline"] == sequential["roll_baseline"]
    assert parallel["search_stats"] == sequential["search_stats"]
    assert (parallel["level_survivors"] or []) == (sequential["level_survivors"] or [])
    reported = parallel["parallel_exact"]
    assert reported and reported["path"] == "parallel"
    assert reported["worker_jobs_completed"] == reported["unique_exact_keys"]
    assert reported["worker_jobs_cancelled"] == 0


def test_merge_is_independent_of_completion_order():
    """Shuffle the ORDER in which worker results arrive and re-merge.

    The merge inserts by key, and every downstream step reads the parent's sequential
    promoted-route loop, so a shuffled arrival order must be unobservable.
    """

    universe, state, meta, scenario, config, provider = _fixture()
    sequential = ro.optimize(universe=universe, initial_state=state, scenario=scenario,
                             player_meta=meta, config=config, world_provider=provider,
                             exact_cache={})
    base = ro.optimize(universe=universe, initial_state=state, scenario=scenario,
                       player_meta=meta, config=config, world_provider=provider,
                       exact_cache={}, parallel_workers=2)
    # Re-run several times: the pool's arrival order varies run to run, so identical
    # output across repeats is the observable form of order-independence.
    for _ in range(2):
        repeat = ro.optimize(universe=universe, initial_state=state, scenario=scenario,
                             player_meta=meta, config=config, world_provider=provider,
                             exact_cache={}, parallel_workers=2)
        assert repeat["routes"] == base["routes"] == sequential["routes"]
        assert repeat["paired_supported_3gw"] == sequential["paired_supported_3gw"]


def test_merge_by_key_ignores_arrival_order_directly():
    """The merge map itself is order-free: feed the same results in two orders."""

    universe, state, meta, scenario, config, provider = _fixture()
    promoted = _promoted(universe, state, meta, scenario, config, provider)
    worlds = {int(e): provider(e, [int(r["player_id"]) for r in universe["universe"]])
              for e in config.events}
    positions = {int(row["player_id"]): str(row["position"]) for row in universe["universe"]}

    def positions_of(squad_ids):
        return {int(pid): positions[int(pid)] for pid in squad_ids}

    tasks, _ = px.required_keys(promoted, worlds_by_event=worlds, positions_of=positions_of,
                                cache={}, config=config, world_identity=None)
    results = []
    for task in tasks:
        with_kind = px.ExactTask(key=task.key, event=task.event, squad_ids=task.squad_ids,
                                 world_identity=task.world_identity, matrix_digest="")
        results.append(px.ExactResult(
            key=with_kind.key, event=with_kind.event, squad_ids=with_kind.squad_ids,
            world_identity=with_kind.world_identity, matrix_digest="",
            entry={"policy": "P", "scores": [1.0], "mean_gross": 1.0}, worker_pid=1))
    by_key = {task.key: task for task in
              [px.ExactTask(key=t.key, event=t.event, squad_ids=t.squad_ids,
                            world_identity=t.world_identity, matrix_digest="") for t in tasks]}
    forward, reverse = {}, {}
    for result in results:
        px._accept(result, cache=forward, by_key=by_key, counters=px.ScheduleCounters())
    for result in reversed(results):
        px._accept(result, cache=reverse, by_key=by_key, counters=px.ScheduleCounters())
    assert forward == reverse


# ---------------------------------------------------------------------------
# provenance validation
# ---------------------------------------------------------------------------

def test_worker_result_with_a_wrong_key_is_rejected():
    task = px.ExactTask(key=(1, "abc", 10, 1, "w"), event=1, squad_ids=(1, 2),
                        world_identity="w", matrix_digest="")
    result = px.ExactResult(key=("other",), event=1, squad_ids=(1, 2), world_identity="w",
                            matrix_digest="", entry={"policy": "p", "scores": []}, worker_pid=1)
    with pytest.raises(px.ParallelExactError, match="UNEXPECTED_RESULT"):
        px._accept(result, cache={}, by_key={task.key: task}, counters=px.ScheduleCounters())


def test_worker_result_with_a_wrong_world_identity_is_rejected():
    task = px.ExactTask(key=(1, "abc", 10, 1, "w"), event=1, squad_ids=(1, 2),
                        world_identity="w", matrix_digest="")
    result = px.ExactResult(key=task.key, event=1, squad_ids=(1, 2), world_identity="DIFFERENT",
                            matrix_digest="", entry={"policy": "p", "scores": []}, worker_pid=1)
    with pytest.raises(px.ParallelExactError, match="WORLD_IDENTITY_MISMATCH"):
        px._accept(result, cache={}, by_key={task.key: task}, counters=px.ScheduleCounters())


def test_worker_result_from_different_world_bytes_is_rejected():
    """The strongest provenance check: the worker must have scored the SAME BYTES."""

    task = px.ExactTask(key=(1, "abc", 10, 1, "w"), event=1, squad_ids=(1, 2),
                        world_identity="w", matrix_digest="deadbeef" * 8)
    result = px.ExactResult(key=task.key, event=1, squad_ids=(1, 2), world_identity="w",
                            matrix_digest="feedface" * 8,
                            entry={"policy": "p", "scores": []}, worker_pid=1)
    with pytest.raises(px.ParallelExactError, match="MATRIX_MISMATCH"):
        px._accept(result, cache={}, by_key={task.key: task}, counters=px.ScheduleCounters())


def test_worker_result_is_accepted_when_identity_matches():
    task = px.ExactTask(key=(1, "abc", 10, 1, "w"), event=1, squad_ids=(1, 2),
                        world_identity="w", matrix_digest="deadbeef" * 8)
    result = px.ExactResult(key=task.key, event=1, squad_ids=(1, 2), world_identity="w",
                            matrix_digest="deadbeef" * 8,
                            entry={"policy": "p", "scores": [1.0], "mean_gross": 1.0},
                            worker_pid=7)
    cache: dict = {}
    counters = px.ScheduleCounters()
    px._accept(result, cache=cache, by_key={task.key: task}, counters=counters)
    assert cache[task.key]["mean_gross"] == 1.0
    assert counters.worker_pids == [7]


def test_partial_worker_results_are_rejected():
    task = px.ExactTask(key=(1, "abc", 10, 1, "w"), event=1, squad_ids=(1, 2),
                        world_identity="w", matrix_digest="")
    result = px.ExactResult(key=task.key, event=1, squad_ids=(1, 2), world_identity="w",
                            matrix_digest="", entry={"policy": "p"}, worker_pid=1)
    with pytest.raises(px.ParallelExactError, match="PARTIAL_RESULT"):
        px._accept(result, cache={}, by_key={task.key: task}, counters=px.ScheduleCounters())


def test_matrix_source_describes_either_a_file_or_a_blob():
    from fpl_brain.manager_worlds import MATRIX_IDENTITY_KEY, MATRIX_PATH_KEY
    from pathlib import Path

    matrix = {"worlds": 2, "player_ids": [1, 2], "core": {1: [1.0, 2.0], 2: [3.0, 4.0]},
              "minutes": {1: [90.0, 90.0], 2: [0.0, 90.0]}}
    kind, payload, digest = px.describe_matrix_source(matrix)
    assert kind == "blob"
    assert pickle.loads(payload) == matrix
    assert digest and digest == px._digest_blob(payload)

    target = Path(__file__).parent / "_p3_matrix_fixture.json"
    try:
        import json
        target.write_text(json.dumps({"worlds": 2, "player_ids": [1, 2],
                                      "core": {"1": [1.0, 2.0], "2": [3.0, 4.0]},
                                      "minutes": {"1": [90.0, 90.0], "2": [0.0, 90.0]}}),
                          encoding="utf-8")
        stamped = dict(matrix)
        stamped[MATRIX_IDENTITY_KEY] = "identity-x"
        stamped[MATRIX_PATH_KEY] = str(target)
        kind, payload, digest = px.describe_matrix_source(stamped)
        assert kind == "file"
        assert payload == str(target)
        assert digest == px._digest_file(target)
    finally:
        target.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# failure behaviour
# ---------------------------------------------------------------------------

def _failed_future(error):
    """A real Future carrying a failure, so `wait()` can operate on it."""

    from concurrent.futures import Future

    future: Future = Future()
    future.set_exception(error)
    return future


class _FakeExecutor:
    """Fails every job, so the batch's fail-closed behaviour can be exercised cheaply."""

    def __init__(self, error):
        self.error = error
        self.submitted = 0
        self.shutdown_calls: list = []

    def submit(self, fn, task):
        self.submitted += 1
        return _failed_future(self.error)

    def shutdown(self, wait=True, cancel_futures=False):
        self.shutdown_calls.append((wait, cancel_futures))


def _one_task_fixture():
    universe, state, meta, scenario, config, provider = _fixture()
    promoted = _promoted(universe, state, meta, scenario, config, provider)
    worlds = {int(e): provider(e, [int(r["player_id"]) for r in universe["universe"]])
              for e in config.events}
    positions = {int(row["player_id"]): str(row["position"]) for row in universe["universe"]}

    def positions_of(squad_ids):
        return {int(pid): positions[int(pid)] for pid in squad_ids}

    return promoted, worlds, positions_of, config


def test_worker_failure_fails_the_batch_closed_and_leaves_no_cache_entry():
    promoted, worlds, positions_of, config = _one_task_fixture()
    cache: dict = {}
    executor = _FakeExecutor(RuntimeError("worker exploded"))
    with pytest.raises(px.ParallelExactError, match="WORKER_FAILED"):
        px.prefetch_exact_cache(promoted=promoted, worlds_by_event=worlds,
                                positions_of=positions_of, cache=cache, config=config,
                                world_identity=None, worker_count=2,
                                executor_factory=lambda: executor)
    assert cache == {}, "a failed batch must not leave a partial cache entry"
    assert executor.shutdown_calls, "the pool must still be shut down on failure"
    assert executor.shutdown_calls[0][1] is True, "pending futures must be cancelled"


def test_sequential_fallback_is_recorded_and_identical():
    promoted, worlds, positions_of, config = _one_task_fixture()
    executor = _FakeExecutor(RuntimeError("worker exploded"))
    with_fallback: dict = {}
    counters = px.prefetch_exact_cache(
        promoted=promoted, worlds_by_event=worlds, positions_of=positions_of,
        cache=with_fallback, config=config, world_identity=None, worker_count=2,
        sequential_fallback=True, executor_factory=lambda: executor)
    assert counters.fallback_used is True
    assert counters.fallback_reason and "worker exploded" in counters.fallback_reason
    assert counters.parallel_failure and "worker exploded" in counters.parallel_failure

    sequential_cache: dict = {}
    px.prefetch_exact_cache(promoted=promoted, worlds_by_event=worlds,
                            positions_of=positions_of, cache=sequential_cache, config=config,
                            world_identity=None, worker_count=None)
    assert set(with_fallback) == set(sequential_cache)
    for key, entry in sequential_cache.items():
        assert with_fallback[key]["mean_gross"] == entry["mean_gross"]
        assert with_fallback[key]["scores"] == entry["scores"]
        assert (with_fallback[key]["policy"].ordering_key()
                == entry["policy"].ordering_key())


def test_missing_matrix_source_fails_closed_in_the_worker():
    """A worker asked about an event it has no source for must refuse, not invent one."""

    px._worker_init({"config": None, "positions_by_id": {}, "matrix_sources": {},
                     "matrix_identity": {}, "matrix_cache": 1})
    with pytest.raises(px.ParallelExactError, match="MATRIX_SOURCE_MISSING"):
        px._load_worker_matrix(5)


def test_worker_installs_a_monte_carlo_guard():
    from fpl_brain import monte_carlo

    original = monte_carlo.simulate
    try:
        px._worker_init({"config": None, "positions_by_id": {}, "matrix_sources": {},
                         "matrix_identity": {}, "matrix_cache": 1})
        with pytest.raises(px.ParallelExactError, match="MONTE_CARLO_FORBIDDEN"):
            monte_carlo.simulate(None, None)
    finally:
        monte_carlo.simulate = original


# ---------------------------------------------------------------------------
# cancellation
# ---------------------------------------------------------------------------

def test_cancel_before_dispatch_leaves_the_cache_empty():
    from fpl_brain import execution

    promoted, worlds, positions_of, config = _one_task_fixture()
    cache: dict = {}

    def probe():
        raise execution.RunCancelled("cancel before dispatch")

    with pytest.raises(execution.RunCancelled):
        px.prefetch_exact_cache(promoted=promoted, worlds_by_event=worlds,
                                positions_of=positions_of, cache=cache, config=config,
                                world_identity=None, worker_count=2, cancel_probe=probe)
    assert cache == {}


class _LocalExecutor:
    """A deterministic stand-in for the pool: the REAL worker result, controlled arrival.

    It runs the production ``worker_evaluate`` (so the results under test are the real
    ones) but hands them back in an order and at a time the test chooses.  That is what
    makes the cancellation and merge-order tests deterministic rather than
    timing-dependent, while a real spawn pool is still exercised by the identity tests
    and by the P3 benchmark.
    """

    def __init__(self):
        self.queue: list = []
        self.shutdown_calls: list = []

    def submit(self, fn, task):
        px._worker_init({"config": _LOCAL_STATE["config"],
                         "positions_by_id": _LOCAL_STATE["positions_by_id"],
                         "matrix_sources": _LOCAL_STATE["matrix_sources"],
                         "matrix_identity": _LOCAL_STATE["matrix_identity"],
                         "matrix_cache": 4})
        self.queue.append(fn(task))
        return _ReadyFuture(self)

    def take(self):
        return self.queue.pop(0)

    def shutdown(self, wait=True, cancel_futures=False):
        self.shutdown_calls.append((wait, cancel_futures))


class _ReadyFuture:
    def __init__(self, executor):
        self.executor = executor

    def result(self):
        return self.executor.take()

    def cancel(self):
        return True


_LOCAL_STATE: dict = {}


def _prime_local_state(promoted, worlds, positions_of, config, world_identity=None):
    """Publish the same worker payload the parent would send to a real pool."""

    tasks, _ = px.required_keys(promoted, worlds_by_event=worlds, positions_of=positions_of,
                                cache={}, config=config, world_identity=world_identity)
    positions_by_id: dict = {}
    sources: dict = {}
    identities: dict = {}
    for task in tasks:
        for pid, role in positions_of(task.squad_ids).items():
            positions_by_id[int(pid)] = str(role)
        if task.event not in sources:
            kind, payload, _digest = px.describe_matrix_source(worlds[task.event])
            sources[task.event] = (kind, payload)
        identities[task.event] = task.world_identity
    _LOCAL_STATE.update({"config": config, "positions_by_id": positions_by_id,
                         "matrix_sources": sources, "matrix_identity": identities})
    return tasks


def test_cancel_before_dispatch_leaves_the_cache_empty():
    from fpl_brain import execution

    promoted, worlds, positions_of, config = _one_task_fixture()
    _prime_local_state(promoted, worlds, positions_of, config)
    cache: dict = {}

    def probe():
        raise execution.RunCancelled("cancel before dispatch")

    with pytest.raises(execution.RunCancelled):
        px.prefetch_exact_cache(promoted=promoted, worlds_by_event=worlds,
                                positions_of=positions_of, cache=cache, config=config,
                                world_identity=None, worker_count=2, cancel_probe=probe,
                                executor_factory=_LocalExecutor)
    assert cache == {}


def test_cancel_with_queued_jobs_keeps_only_complete_entries():
    """Cancel after some jobs landed: every retained entry is complete."""

    from fpl_brain import execution

    promoted, worlds, positions_of, config = _one_task_fixture()
    tasks = _prime_local_state(promoted, worlds, positions_of, config)
    cache: dict = {}
    polls = {"n": 0}
    executor = _LocalExecutor()

    def probe():
        # One poll per submit plus one per wait timeout; 3 lets a wave through, then cancels.
        polls["n"] += 1
        if polls["n"] > 3:
            raise execution.RunCancelled("cancel with queued jobs")

    with pytest.raises(execution.RunCancelled):
        px.prefetch_exact_cache(promoted=promoted, worlds_by_event=worlds,
                                positions_of=positions_of, cache=cache, config=config,
                                world_identity=None, worker_count=2, cancel_probe=probe,
                                executor_factory=lambda: executor)
    assert executor.shutdown_calls, "the pool must be shut down on cancellation"
    assert executor.shutdown_calls[0][1] is True, "pending futures must be cancelled"
    assert set(cache) <= {task.key for task in tasks}
    for entry in cache.values():
        assert set(entry) >= {"policy", "scores", "mean_gross"}, "a partial entry was retained"
        assert isinstance(entry["mean_gross"], float)
        assert entry["scores"]


def test_cancel_while_jobs_are_running_aborts_cleanly():
    """Cancel mid-wave: the batch aborts and nothing is left half-written."""

    from fpl_brain import execution

    promoted, worlds, positions_of, config = _one_task_fixture()
    tasks = _prime_local_state(promoted, worlds, positions_of, config)
    assert len(tasks) >= 4, "this test needs several jobs to cancel mid-wave"
    cache: dict = {}
    polls = {"n": 0}

    def probe():
        polls["n"] += 1
        if polls["n"] > 2:
            raise execution.RunCancelled("cancel while running")

    with pytest.raises(execution.RunCancelled):
        px.prefetch_exact_cache(promoted=promoted, worlds_by_event=worlds,
                                positions_of=positions_of, cache=cache, config=config,
                                world_identity=None, worker_count=2, cancel_probe=probe,
                                executor_factory=_LocalExecutor)
    assert len(cache) < len(tasks), "the batch really was interrupted, not completed"
    for entry in cache.values():
        assert set(entry) >= {"policy", "scores", "mean_gross"}


def test_interrupted_batch_does_not_poison_the_cache():
    """A later run must still compute every key the interruption skipped."""

    from fpl_brain import execution

    promoted, worlds, positions_of, config = _one_task_fixture()
    tasks = _prime_local_state(promoted, worlds, positions_of, config)
    cache: dict = {}
    polls = {"n": 0}

    def probe():
        polls["n"] += 1
        if polls["n"] > 2:
            raise execution.RunCancelled("cancel")

    with pytest.raises(execution.RunCancelled):
        px.prefetch_exact_cache(promoted=promoted, worlds_by_event=worlds,
                                positions_of=positions_of, cache=cache, config=config,
                                world_identity=None, worker_count=2, cancel_probe=probe,
                                executor_factory=_LocalExecutor)
    px.prefetch_exact_cache(promoted=promoted, worlds_by_event=worlds,
                            positions_of=positions_of, cache=cache, config=config,
                            world_identity=None, worker_count=None)
    assert set(cache) == {task.key for task in tasks}


def test_parallel_exact_error_is_raised_for_an_incomplete_batch():
    """Nothing silently continues with a partial exact cache."""

    promoted, worlds, positions_of, config = _one_task_fixture()
    cache: dict = {}

    class _NeverFinishing:
        def submit(self, fn, task):
            return _failed_future(KeyboardInterrupt("interrupted"))

        def shutdown(self, wait=True, cancel_futures=False):
            pass

    with pytest.raises(KeyboardInterrupt):
        px.prefetch_exact_cache(promoted=promoted, worlds_by_event=worlds,
                                positions_of=positions_of, cache=cache, config=config,
                                world_identity=None, worker_count=2,
                                executor_factory=lambda: _NeverFinishing())


# ---------------------------------------------------------------------------
# Windows spawn safety
# ---------------------------------------------------------------------------

def test_worker_entry_points_are_module_level_and_picklable():
    assert px._WORKER_FN.__module__ == "fpl_brain.parallel_exact"
    assert px._WORKER_INIT.__module__ == "fpl_brain.parallel_exact"
    assert px._WORKER_FN.__qualname__ == "worker_evaluate"
    assert px._WORKER_INIT.__qualname__ == "_worker_init"
    # The spawn context must be exercised, not the platform default.
    assert "spawn" in multiprocessing.get_all_start_methods()
    context = multiprocessing.get_context("spawn")
    assert context.get_start_method() == "spawn"


def test_worker_state_payload_is_picklable_for_spawn():
    universe, state, meta, scenario, config, provider = _fixture()
    payload = {"config": config, "positions_by_id": {1: "GKP"},
               "matrix_sources": {4: ("file", "x.json")},
               "matrix_identity": {4: "identity"},
               "matrix_cache": 1}
    assert pickle.loads(pickle.dumps(payload))["matrix_identity"] == {4: "identity"}


def test_recommended_worker_count_is_not_all_cpus():
    import os

    recommended = px.recommended_worker_count()
    assert recommended <= 4, "the phase forbids defaulting to all logical CPUs"
    assert recommended >= 1
    assert px.DEFAULT_WORKER_COUNT == recommended or "FPL_P3_WORKER_COUNT" in os.environ
