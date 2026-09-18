"""P3 — parallel exact (event, squad) evaluation, bit-identical by construction.

The exact lineup evaluation is the only remaining bottleneck (98% of Stage 2): one
call costs ~190-210 s at 10,000 draws and Stage 2 needs 75-84 independent calls.  This
module parallelises THOSE CALLS and nothing else.

What it does not do
-------------------
It never touches the algorithm.  A worker calls exactly the functions the sequential
path calls, in exactly the same order, on exactly the same worlds:

* one task == one unique ``(event, canonical squad, draws, seed, world provenance)``
  evaluation -- the key ``route_optimizer.exact_evaluation_key`` already defines;
* the worker runs ``route_optimizer._subsample`` ->
  ``manager_lineup.rank_policies`` -> ``route_comparator.policy_world_scores`` and
  returns the COMPLETE cache entry;
* no world, skeleton, captain, autosub or float reduction is split across processes, so
  no summation order changes anywhere.  Workers perform zero stochastic generation.

Why the merge cannot depend on completion order
-----------------------------------------------
The scheduler's only effect on the parent is to PRE-POPULATE the run-scoped
``exact_cache`` by key.  ``route_optimizer.optimize`` then walks the promoted routes in
its existing order and calls ``exact_evaluate``, which now hits the cache.  Route
ranking, tie-breaks, paired comparison, the preferred route, the runner-up and
confidence all read that unchanged sequential loop, so a shuffled completion order is
unobservable by construction -- not merely tested.

Workers
-------
Spawn-safe (Windows): the entry point is a module-level picklable function, the pool
uses an explicit ``spawn`` context, and the worker keeps a bounded, process-local
matrix cache so resident memory stays small.  A worker never opens a database, never
writes anything, and never mutates the parent's cache.

The semantic matrix contract
----------------------------
A worker receives a matrix by one of two transports: a certified cache FILE, or a
pickled BLOB when the parent holds a matrix that has no cache path (a provider-injected
or prebuilt one).  Both must hand exact evaluation the SAME decision problem, so both
pass through :func:`normalise_semantic_matrix`, which requires every
:data:`SEMANTIC_MATRIX_BLOCKS` entry and FAILS CLOSED when one is absent.  A worker is
never the place where a missing policy block is defaulted: ``manager_lineup`` reads an
absent ``expected_bonus`` as "score the armband on CORE alone" and an absent
``role_actionability`` as "nothing is restricted", and either default silently replaces
the parent's decision with a different one.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import pickle
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

PARALLEL_EXACT_VERSION = "parallel_exact_p3_2.0.0"

#: Sequential (in-parent) evaluation.  ``optimize(parallel_workers=None)`` uses it.
SEQUENTIAL = None

#: Declared default worker count.  Deliberately NOT "all logical CPUs": the phase asks
#: for a measured choice with an initial preference of 4 or fewer.  The measured
#: evidence and the recommended production value are in the P3 report.
DEFAULT_WORKER_COUNT = 1

#: How many event matrices a worker keeps resident.  1 bounds worker RSS to roughly one
#: matrix plus its derived memos; larger values trade memory for fewer cold
#: ``captain_terms`` recomputations when tasks for different events interleave.
DEFAULT_WORKER_MATRIX_CACHE = 1

#: The blocks exact evaluation actually consumes.  A world matrix is only usable for a
#: decision when ALL of them are present: ``expected_bonus`` drives the armband objective
#: and ``role_actionability`` restricts the legal policy set, so a matrix missing either
#: describes a DIFFERENT decision problem rather than a degraded one.  The two leading
#: underscores in ``manager_worlds``' names are internal transport metadata
#: (``_p2_matrix_identity`` / ``_p2_matrix_path``) and are deliberately NOT part of this
#: contract: they travel separately and are re-stamped by the loader.
SEMANTIC_MATRIX_BLOCKS = ("worlds", "player_ids", "core", "minutes",
                          "expected_bonus", "role_actionability")


@dataclass(frozen=True)
class ExactTask:
    """One unique exact evaluation, identified by its full cache key."""

    key: tuple
    event: int
    squad_ids: tuple
    world_identity: str
    matrix_digest: str


@dataclass(frozen=True)
class ExactResult:
    """The COMPLETE result of one exact evaluation (never partial)."""

    key: tuple
    event: int
    squad_ids: tuple
    world_identity: str
    matrix_digest: str
    entry: dict
    worker_pid: int
    evaluated_policy_count: int = -1
    matrix_load_seconds: float = 0.0


@dataclass
class ScheduleCounters:
    """Every count the phase asks for, plus the measured overheads."""

    requested_exact_keys: int = 0
    unique_exact_keys: int = 0
    parent_cache_hits: int = 0
    worker_jobs_dispatched: int = 0
    worker_jobs_completed: int = 0
    worker_jobs_cancelled: int = 0
    duplicate_jobs_avoided: int = 0
    worker_matrix_cache_hits: int = 0
    worker_matrix_cache_misses: int = 0
    worker_matrix_loads: int = 0
    evaluated_policy_counts: list = field(default_factory=list)
    worker_matrix_load_seconds: list = field(default_factory=list)
    executor_startup_seconds: float = 0.0
    task_serialize_seconds: float = 0.0
    dispatch_and_collect_seconds: float = 0.0
    pool_shutdown_seconds: float = 0.0
    wall_seconds: float = 0.0
    path: str = "sequential"
    worker_count: int = 0
    worker_pids: list = field(default_factory=list)
    fallback_used: bool = False
    fallback_reason: str | None = None
    parallel_failure: str | None = None

    def as_dict(self) -> dict:
        return {name: value for name, value in self.__dict__.items()}


class ParallelExactError(RuntimeError):
    """A worker failed, or a worker returned a result for a different evaluation."""


# ---------------------------------------------------------------------------
# worker state and entry point (module level => spawn-picklable)
# ---------------------------------------------------------------------------

_WORKER_STATE: dict[str, Any] = {}


def _worker_init(payload: dict) -> None:
    """Load immutable worker state once per process.

    ``matrix_sources`` maps ``event -> ("file", path)`` or ``("blob", pickled bytes)``.
    The payload deliberately does not embed a world matrix in the production path: a
    worker loads the one event it is asked about and keeps it resident.
    """

    _WORKER_STATE.clear()
    _WORKER_STATE.update(payload)
    _WORKER_STATE["matrices"] = {}
    _WORKER_STATE["digests"] = {}
    _WORKER_STATE["order"] = []
    _WORKER_STATE["hits"] = 0
    _WORKER_STATE["misses"] = 0
    _WORKER_STATE["loads"] = 0
    _forbid_monte_carlo()


def _forbid_monte_carlo() -> None:
    """Enforce §12 in the WORKER, not merely assert it.

    A worker must perform zero stochastic generation: it scores preserved worlds, it
    never creates them.  If any code path inside a worker ever reached the Monte Carlo
    engine, the run would silently stop being common-random-numbers and the parallel
    output would stop being bit-identical to the sequential output.  So the worker
    installs a hard error rather than trusting the call graph.
    """

    from . import monte_carlo

    def _forbidden(*args, **kwargs):  # pragma: no cover - only reachable on a defect
        raise ParallelExactError(
            "WORKER_MONTE_CARLO_FORBIDDEN: a worker attempted to generate football worlds; "
            "workers must only score matrices supplied by the parent"
        )

    monte_carlo.simulate = _forbidden  # type: ignore[assignment]


def _digest_blob(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _digest_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1 << 20)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def normalise_semantic_matrix(raw, *, event: int | None = None, source: str = "") -> dict[str, Any]:
    """The ONE reader for a world matrix entering exact evaluation.

    BOTH transports — a certified cache FILE and a pickled BLOB — pass through here, so a
    worker can never evaluate a matrix whose meaning depends on how it arrived.  Blob mode
    previously bypassed validation entirely, which is how a field loss could hide.

    A missing semantic block FAILS CLOSED.  ``manager_lineup`` reads an absent
    ``expected_bonus`` as "the captain is chosen on CORE alone" and an absent
    ``role_actionability`` as "nothing is restricted"; both are correct defaults for a
    caller that knows it has no policy layer, and both are wrong for a worker that is
    supposed to be reproducing the parent's decision.  A worker must never be the place
    where that default gets applied, so the absence is an error, not a fallback.

    Normalisation is value-preserving: it only fixes key types and rebuilds the canonical
    shape, never the numbers.  ``core``/``minutes`` values are carried by reference.
    """

    missing = [name for name in SEMANTIC_MATRIX_BLOCKS if name not in raw]
    if missing:
        raise ParallelExactError(
            f"PARALLEL_EXACT_SEMANTIC_BLOCK_MISSING: the {source or 'supplied'} matrix for "
            f"event {event} is missing {missing}. A worker scores the matrix the parent "
            "scored; it refuses to substitute a default for a missing policy block, "
            "because that would silently change the armband objective or the legal policy "
            "set. Provision the matrix through the current cache schema instead."
        )
    return {
        "worlds": int(raw["worlds"]),
        "player_ids": [int(pid) for pid in raw["player_ids"]],
        "core": {int(pid): series for pid, series in raw["core"].items()},
        "minutes": {int(pid): series for pid, series in raw["minutes"].items()},
        "expected_bonus": {int(pid): float(value) for pid, value in raw["expected_bonus"].items()},
        "role_actionability": {int(pid): bool(value)
                               for pid, value in raw["role_actionability"].items()},
    }


def _load_worker_matrix(event: int) -> tuple[Mapping[str, Any], str]:
    """Load (or reuse) the certified world matrix for one event, worker-locally.

    Returns the matrix and the digest the worker OBSERVED for it, so the parent can
    assert the worker scored the same bytes it dispatched.
    """

    from . import manager_worlds as mw

    matrices = _WORKER_STATE["matrices"]
    cached = matrices.get(event)
    if cached is not None:
        _WORKER_STATE["hits"] += 1
        return cached, _WORKER_STATE["digests"][event]

    _WORKER_STATE["misses"] += 1
    source = _WORKER_STATE["matrix_sources"].get(event)
    if source is None:
        raise ParallelExactError(
            f"WORKER_MATRIX_SOURCE_MISSING: no matrix source for event {event}; a worker "
            "refuses to evaluate against anything it was not given"
        )
    kind, payload = source
    from . import route_optimizer as ro
    load_started = time.perf_counter()
    if kind == "file":
        data = json.loads(Path(str(payload)).read_text(encoding="utf-8"))
        matrix = normalise_semantic_matrix(data, event=event, source=f"file {payload}")
        ro._stamp_matrix_path(matrix, payload)
        digest = _digest_file(payload)
    elif kind == "blob":
        raw = pickle.loads(payload)
        matrix = normalise_semantic_matrix(raw, event=event, source="blob")
        path = raw.get(mw.MATRIX_PATH_KEY) if hasattr(raw, "get") else None
        if path:
            ro._stamp_matrix_path(matrix, path)
        digest = _digest_blob(payload)
    else:  # pragma: no cover - defensive
        raise ParallelExactError(f"WORKER_MATRIX_SOURCE_UNKNOWN: {kind!r}")
    # The identity is transport metadata, not matrix content, so it is re-stamped here for
    # BOTH transports rather than inherited from a pickle.  A worker's memos must key on
    # the identity the parent dispatched, whichever way the bytes arrived.
    identity = _WORKER_STATE["matrix_identity"].get(event)
    if identity:
        ro._stamp_matrix_identity(matrix, identity)
    _WORKER_STATE["last_load_seconds"] = time.perf_counter() - load_started

    # Bound resident memory: evict the oldest event (and the process-local matrix-keyed
    # memos derived from it) so a worker holds at most ``matrix_cache`` matrices.
    limit = max(1, int(_WORKER_STATE.get("matrix_cache", DEFAULT_WORKER_MATRIX_CACHE)))
    order = _WORKER_STATE["order"]
    order.append(event)
    while len(order) > limit:
        evicted = order.pop(0)
        matrices.pop(evicted, None)
        _WORKER_STATE["digests"].pop(evicted, None)
    mw.clear_matrix_memos()
    matrices[event] = matrix
    _WORKER_STATE["digests"][event] = digest
    _WORKER_STATE["loads"] += 1
    return matrix, digest


def worker_evaluate(task: ExactTask) -> ExactResult:
    """One exact evaluation, exactly as the sequential path performs it."""

    from . import manager_lineup as ml, route_comparator as rc, route_optimizer as ro

    config = _WORKER_STATE["config"]
    positions_by_id = _WORKER_STATE["positions_by_id"]

    recomputed = ro.exact_evaluation_key(task.event, task.squad_ids, config,
                                        task.world_identity or None)
    if recomputed != task.key:
        raise ParallelExactError(
            f"WORKER_KEY_MISMATCH: dispatched key {task.key!r} does not match the key "
            f"recomputed from the task's own inputs {recomputed!r}"
        )

    matrix, digest = _load_worker_matrix(int(task.event))
    from .manager_worlds import MATRIX_IDENTITY_KEY

    stamped = matrix.get(MATRIX_IDENTITY_KEY) if hasattr(matrix, "get") else None
    if task.world_identity and str(stamped) != str(task.world_identity):
        raise ParallelExactError(
            f"WORKER_WORLD_IDENTITY_MISMATCH: the worker holds matrix identity {stamped!r} for "
            f"event {task.event} but the task was dispatched with {task.world_identity!r}"
        )

    squad_ids = tuple(int(pid) for pid in task.squad_ids)
    positions = {int(pid): positions_by_id[int(pid)] for pid in squad_ids}
    selection = ro._subsample(matrix, int(config.policy_selection_worlds))
    ranked = ml.rank_policies(list(squad_ids), positions, selection, top_k=1)
    if not ranked["top_policies"]:
        raise ParallelExactError(f"no legal manager policy for squad at event {task.event}")
    policy = ranked["top_policies"][0]
    scores = rc.policy_world_scores(policy, matrix, positions)
    # The cache entry is EXACTLY what the sequential path writes.  Job diagnostics travel
    # out of band on the result, never inside the entry, so the two paths cannot differ
    # even structurally.
    entry = {"policy": policy, "scores": scores, "mean_gross": sum(scores) / len(scores)}
    return ExactResult(key=task.key, event=int(task.event), squad_ids=squad_ids,
                       world_identity="" if task.world_identity is None else str(task.world_identity),
                       matrix_digest=digest, entry=entry, worker_pid=os.getpid(),
                       evaluated_policy_count=int(ranked["evaluated_policies"]),
                       matrix_load_seconds=float(_WORKER_STATE.get("last_load_seconds", 0.0)))


_WORKER_INIT = _worker_init
_WORKER_FN = worker_evaluate


# ---------------------------------------------------------------------------
# parent side
# ---------------------------------------------------------------------------


def describe_matrix_source(matrix) -> tuple[str, Any, str]:
    """A spawn-safe description of how a worker can obtain this matrix.

    ``("file", path, sha256-of-file)`` whenever the matrix came from (or was written to)
    the certified world cache; otherwise ``("blob", pickled bytes, sha256-of-bytes)``.
    Either way the digest is computed over exactly the bytes the worker will consume.
    """

    from .manager_worlds import MATRIX_PATH_KEY

    path = matrix.get(MATRIX_PATH_KEY) if hasattr(matrix, "get") else None
    if path and Path(str(path)).exists():
        return ("file", str(path), _digest_file(path))
    payload = pickle.dumps(dict(matrix))
    return ("blob", payload, _digest_blob(payload))


def required_keys(promoted: Sequence[Any], *, worlds_by_event, positions_of, cache,
                  config, world_identity) -> tuple[list[ExactTask], ScheduleCounters]:
    """The unique, still-missing exact evaluations the promoted routes require.

    Deduplicated on the FULL key, walked in the promoted-route order the sequential path
    uses, so the task list is a deterministic function of the route set.
    """

    from . import route_optimizer as ro

    counters = ScheduleCounters()
    tasks: list[ExactTask] = []
    seen: set = set()
    for partial in promoted:
        for event, squad_ids in ro.route_event_squads(partial):
            counters.requested_exact_keys += 1
            identity = None if world_identity is None else world_identity.get(int(event))
            key = ro.exact_evaluation_key(event, squad_ids, config, identity)
            if key in seen:
                counters.duplicate_jobs_avoided += 1
                continue
            seen.add(key)
            counters.unique_exact_keys += 1
            if key in cache:
                counters.parent_cache_hits += 1
                continue
            tasks.append(ExactTask(key=key, event=int(event), squad_ids=tuple(squad_ids),
                                  world_identity="" if identity is None else str(identity),
                                  matrix_digest=""))
    return tasks, counters


def prefetch_exact_cache(*, promoted, worlds_by_event, positions_of, cache, config,
                         world_identity, worker_count: int | None = SEQUENTIAL,
                         matrix_cache: int = DEFAULT_WORKER_MATRIX_CACHE,
                         cancel_probe=None, wave_multiplier: int = 2,
                         sequential_fallback: bool = False,
                         executor_factory=None, result_sink: dict | None = None,
                         positions_by_id: Mapping[int, str] | None = None) -> ScheduleCounters:
    """Populate ``cache`` with the missing exact evaluations of ``promoted``.

    Only COMPLETE, identity-verified results are ever inserted.  ``worker_count=None``
    runs the same evaluator sequentially in the parent (the P2 path).  ``worker_count=1``
    runs it in a real one-process pool, which must produce identical output.
    """

    started = time.perf_counter()
    tasks, counters = required_keys(promoted, worlds_by_event=worlds_by_event,
                                   positions_of=positions_of, cache=cache, config=config,
                                   world_identity=world_identity)
    sink = result_sink if result_sink is not None else {}
    sink.clear()
    sink.update({"worker_count": 0 if worker_count is None else int(worker_count),
                 "missing_tasks": len(tasks)})

    if not tasks or worker_count is None:
        for task in tasks:
            _sequential_task(task, worlds_by_event=worlds_by_event, positions_of=positions_of,
                             cache=cache, counters=counters, config=config)
        counters.path = "sequential"
        counters.wall_seconds = time.perf_counter() - started
        sink["path"] = counters.path
        return counters

    sources: dict[int, tuple] = {}
    identities: dict[int, str] = {}
    digests: dict[int, str] = {}
    for event in sorted({task.event for task in tasks}):
        kind, payload, digest = describe_matrix_source(worlds_by_event[int(event)])
        sources[int(event)] = (kind, payload)
        digests[int(event)] = digest
        identities[int(event)] = ("" if world_identity is None
                                  else str(world_identity.get(int(event), "")))
    dispatched = [ExactTask(key=task.key, event=task.event, squad_ids=task.squad_ids,
                            world_identity=task.world_identity,
                            matrix_digest=digests[task.event]) for task in tasks]

    if positions_by_id is None:
        positions_by_id = {int(pid): str(role) for pid, role in
                           _positions_from(positions_of, dispatched)}
    payload = {
        "config": config,
        "positions_by_id": dict(positions_by_id),
        "matrix_sources": sources,
        "matrix_identity": identities,
        "matrix_cache": int(matrix_cache),
    }
    multiprocessing.freeze_support()
    context = multiprocessing.get_context("spawn")
    factory = executor_factory or (
        lambda: ProcessPoolExecutor(max_workers=int(worker_count), mp_context=context,
                                    initializer=_WORKER_INIT, initargs=(payload,)))

    counters.worker_count = int(worker_count)
    aborted = False
    serialize_started = time.perf_counter()
    executor = None
    try:
        executor = factory()
        counters.executor_startup_seconds = time.perf_counter() - serialize_started
        counters.task_serialize_seconds = time.perf_counter() - serialize_started
        counters = _run_waves(executor, dispatched, cache=cache, counters=counters,
                              cancel_probe=cancel_probe, worker_count=int(worker_count),
                              wave_multiplier=int(wave_multiplier),
                              sequential_fallback=sequential_fallback,
                              worlds_by_event=worlds_by_event, positions_of=positions_of,
                              config=config)
        counters.path = "parallel"
    except BaseException as failure:
        aborted = True
        if not isinstance(failure, ParallelExactError):
            counters.worker_jobs_cancelled = max(
                0, len(dispatched) - counters.worker_jobs_completed)
        sink["aborted"] = True
        raise
    finally:
        sink["path"] = counters.path if not aborted else "aborted"
        if executor is not None:
            shutdown_started = time.perf_counter()
            try:
                # Never kill workers arbitrarily: cancel what has not started and let the
                # pool wind down.  `wait=False` on an abnormal exit keeps the cancel path
                # responsive; a worker owns no database transaction, so an in-flight task
                # is harmless and simply finishes.
                executor.shutdown(wait=not aborted, cancel_futures=True)
            except Exception:  # pragma: no cover - defensive
                executor.shutdown(wait=False)
            counters.pool_shutdown_seconds = time.perf_counter() - shutdown_started
            counters.wall_seconds = time.perf_counter() - started
    sink.update(counters.as_dict())
    return counters


def _positions_from(positions_of, tasks) -> list[tuple[int, str]]:
    wanted: dict[int, str] = {}
    for task in tasks:
        for pid, role in positions_of(task.squad_ids).items():
            wanted[int(pid)] = str(role)
    return list(wanted.items())


def _sequential_task(task: ExactTask, *, worlds_by_event, positions_of, cache, counters,
                     config) -> None:
    from . import manager_lineup as ml, route_comparator as rc, route_optimizer as ro

    key = ro.exact_evaluation_key(task.event, task.squad_ids, config,
                                 task.world_identity or None)
    squad_ids = tuple(int(pid) for pid in task.squad_ids)
    positions = positions_of(squad_ids)
    matrix = worlds_by_event[int(task.event)]
    selection = ro._subsample(matrix, int(config.policy_selection_worlds))
    ranked = ml.rank_policies(list(squad_ids), positions, selection, top_k=1)
    if not ranked["top_policies"]:
        raise ParallelExactError(f"no legal manager policy for squad at event {task.event}")
    policy = ranked["top_policies"][0]
    scores = rc.policy_world_scores(policy, matrix, positions)
    cache[key] = {"policy": policy, "scores": scores,
                  "mean_gross": sum(scores) / len(scores)}
    # Recorded out of band exactly as the worker does, so the two paths are comparable.
    counters.evaluated_policy_counts.append(int(ranked["evaluated_policies"]))
    counters.worker_matrix_load_seconds.append(0.0)
    counters.worker_jobs_dispatched += 1
    counters.worker_jobs_completed += 1


def _run_waves(executor, tasks, *, cache, counters, cancel_probe, worker_count,
               wave_multiplier, sequential_fallback, worlds_by_event, positions_of,
               config) -> ScheduleCounters:
    """Submit in bounded waves; insert each COMPLETE, verified result BY KEY.

    Waves are ordered by event so a worker's matrix memo stays warm, and bounded so the
    parent stays responsive to cancellation and the number of resident matrices stays
    small.
    """

    by_key = {task.key: task for task in tasks}
    ordered = sorted(tasks, key=lambda task: (task.event, str(task.key)))
    wave_size = max(1, int(worker_count) * max(1, int(wave_multiplier)))
    pending: set = set()
    index = 0
    loop_started = time.perf_counter()
    while index < len(ordered) or pending:
        while index < len(ordered) and len(pending) < wave_size:
            if cancel_probe is not None:
                cancel_probe()
            pending.add(executor.submit(_WORKER_FN, ordered[index]))
            counters.worker_jobs_dispatched += 1
            index += 1
        if not pending:
            break
        done, pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)
        if not done:
            # Nothing finished within the timeout: stay responsive to cancellation.
            if cancel_probe is not None:
                cancel_probe()
            continue
        for future in done:
            try:
                result = future.result()
            except BaseException as failure:
                if not isinstance(failure, Exception):
                    # KeyboardInterrupt / SystemExit are control flow, not a worker
                    # defect: never wrap them into a "worker failed" error.
                    raise
                counters.parallel_failure = f"{type(failure).__name__}: {failure}"
                if sequential_fallback:
                    counters.fallback_used = True
                    counters.fallback_reason = counters.parallel_failure
                    continue
                raise ParallelExactError(
                    f"PARALLEL_EXACT_WORKER_FAILED: {type(failure).__name__}: {failure}"
                ) from failure
            _accept(result, cache=cache, by_key=by_key, counters=counters)
            counters.worker_jobs_completed += 1
    counters.dispatch_and_collect_seconds = time.perf_counter() - loop_started

    missing = [task for task in ordered if task.key not in cache]
    if missing and sequential_fallback:
        for task in missing:
            _sequential_task(task, worlds_by_event=worlds_by_event, positions_of=positions_of,
                             cache=cache, counters=counters, config=config)
        missing = [task for task in ordered if task.key not in cache]
    if missing:
        raise ParallelExactError(
            f"PARALLEL_EXACT_INCOMPLETE: {len(missing)} of {len(ordered)} exact evaluations "
            f"were not completed (first: event {missing[0].event}, squad "
            f"{missing[0].squad_ids[:3]}…); refusing to continue with a partial exact cache"
        )
    return counters


def _accept(result: ExactResult, *, cache, by_key, counters) -> None:
    """Validate a worker result's identity, then insert it.  A mismatch is a hard failure."""

    expected = by_key.get(result.key)
    if expected is None:
        raise ParallelExactError(
            f"PARALLEL_EXACT_UNEXPECTED_RESULT: worker returned key {result.key!r} which was "
            "never dispatched"
        )
    if str(result.world_identity) != str(expected.world_identity):
        raise ParallelExactError(
            "PARALLEL_EXACT_WORLD_IDENTITY_MISMATCH: worker reported "
            f"{result.world_identity!r} but the parent dispatched {expected.world_identity!r}"
        )
    if expected.matrix_digest and str(result.matrix_digest) != str(expected.matrix_digest):
        raise ParallelExactError(
            "PARALLEL_EXACT_MATRIX_MISMATCH: worker scored a matrix with digest "
            f"{str(result.matrix_digest)[:16]}… but the parent dispatched "
            f"{str(expected.matrix_digest)[:16]}…"
        )
    entry = result.entry
    if not isinstance(entry, dict) or "policy" not in entry or "scores" not in entry:
        raise ParallelExactError(
            "PARALLEL_EXACT_PARTIAL_RESULT: worker returned an incomplete cache entry"
        )
    if int(result.worker_pid) not in counters.worker_pids:
        counters.worker_pids.append(int(result.worker_pid))
    counters.evaluated_policy_counts.append(int(result.evaluated_policy_count))
    counters.worker_matrix_load_seconds.append(float(result.matrix_load_seconds))
    cache[result.key] = entry


def recommended_worker_count() -> int:
    """The production default worker count (see the P3 report for the measured basis)."""

    return int(os.environ.get("FPL_P3_WORKER_COUNT", DEFAULT_WORKER_COUNT))


__all__ = [
    "DEFAULT_WORKER_COUNT",
    "DEFAULT_WORKER_MATRIX_CACHE",
    "ExactResult",
    "ExactTask",
    "PARALLEL_EXACT_VERSION",
    "ParallelExactError",
    "SEQUENTIAL",
    "SEMANTIC_MATRIX_BLOCKS",
    "ScheduleCounters",
    "describe_matrix_source",
    "normalise_semantic_matrix",
    "prefetch_exact_cache",
    "recommended_worker_count",
    "required_keys",
    "worker_evaluate",
]
