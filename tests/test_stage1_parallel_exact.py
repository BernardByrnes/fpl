"""Performance Spike B1 — Stage-1 exact-evaluation parallelism.

Stage 1 previously ran its exact evaluations sequentially, by an explicit in-tree decision:
a 2,000-draw evaluation is much shorter than a 10,000-draw one, so pool overhead might
outweigh the gain, and the P3 bit-identity proof covered only the 10,000-draw path.  This
module supplies the missing equivalence proof and asserts the PRODUCTION WIRING, not a
helper: the runner's own Stage-1 call site is read from its AST, because that call cannot be
driven without a database.

The optimization is SCHEDULING ONLY.  Every assertion below is exact `==` on
decision-affecting output, because the exact evaluation is governed by the repository's
standing bit-identity contract.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

from fpl_brain import parallel_exact as px
from fpl_brain import route_optimizer as ro
from test_route_optimizer import _config, _provider, _scenario, _universe

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_four_gw_decision as runner  # noqa: E402

RUNNER_SOURCE = (SCRIPTS / "run_four_gw_decision.py").read_text(encoding="utf-8")
RUNNER_TREE = ast.parse(RUNNER_SOURCE)


def _stage1_optimize_call():
    """The runner's real `stage1_result = ...optimize(...)` call node."""

    for node in ast.walk(RUNNER_TREE):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            func = node.value.func
            if "stage1_result" in targets and isinstance(func, ast.Attribute) \
                    and func.attr == "optimize":
                return node.value
    raise AssertionError("the runner has no `stage1_result = ...optimize(...)` call")


def _rich_provider(*, bonus=None, unresolved=(), events=("5",), worlds=6):
    """A Stage-1-shaped provider that also carries the two policy-level blocks.

    ``expected_bonus`` exercises the corrected armband arithmetic and
    ``role_actionability`` exercises the policy-set restriction, so a scheduling change that
    reordered or dropped a matrix field would show up as a value difference.
    """

    base = _provider()

    def provider(event, union_ids):
        matrix = dict(base(event, union_ids))
        if bonus is not None:
            matrix["expected_bonus"] = {int(p): float(bonus(p)) for p in union_ids}
        if unresolved:
            matrix["role_actionability"] = {int(p): (int(p) in set(unresolved)) for p in union_ids}
        return matrix

    return provider


def _run(universe, state, meta, scenario, config, provider, *, workers):
    cache: dict = {}
    result = ro.optimize(
        universe=universe, initial_state=state, scenario=scenario, player_meta=meta,
        config=config, world_provider=provider, exact_cache=cache,
        parallel_workers=workers,
    )
    return result, cache


# ---------------------------------------------------------------------------
# WIRING — the production call site, read from the real source
# ---------------------------------------------------------------------------


def test_the_runner_passes_the_configured_worker_count_to_stage1():
    """PREDECESSOR KILL: on 8468e92 the Stage-1 call passed no worker count at all.

    Asserted on the runner's own AST rather than on a helper, because the production wiring
    IS the defect this slice removes.
    """

    keywords = {keyword.arg for keyword in _stage1_optimize_call().keywords}
    assert "parallel_workers" in keywords, (
        "the Stage-1 call must forward parallel_workers; without it Stage 1 is sequential"
    )


def test_the_runner_does_not_hard_code_the_stage1_worker_count():
    """The count must come from the CLI, so `--parallel-workers 1` stays a real rollback."""

    keywords = {k.arg: k.value for k in _stage1_optimize_call().keywords}
    value = keywords["parallel_workers"]
    assert isinstance(value, ast.Name) and value.id == "parallel_workers", (
        "the Stage-1 call must forward the resolved CLI value, not a literal"
    )


def test_no_silent_sequential_fallback_is_enabled_for_stage1():
    """Speed must never be bought by weakening the failure contract."""

    keywords = {k.arg for k in _stage1_optimize_call().keywords}
    assert "sequential_fallback" not in keywords


# ---------------------------------------------------------------------------
# A / B / E / F — exactness across the real Stage-1 path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("workers", [2, 4])
def test_stage1_serial_and_parallel_are_bit_identical(workers):
    """Serial vs parallel on the same deterministic input: exact `==` throughout."""

    universe, state, meta = _universe()
    scenario, config = _scenario(), _config()
    provider = _rich_provider(bonus=lambda p: 0.25 if int(p) % 2 else 0.0,
                              unresolved={next(iter(state.players)).player_id})

    sequential, seq_cache = _run(universe, state, meta, scenario, config, provider, workers=None)
    parallel, par_cache = _run(universe, state, meta, scenario, config, provider, workers=workers)

    # A. every decision-affecting output, exact equality (no tolerance).
    #
    # ``exact_cache_hits`` is deliberately NOT compared: it counts how many evaluations were
    # SERVED FROM the cache rather than computed, so the parallel path -- which pre-populates
    # the whole batch first -- necessarily reports more of them.  It is a scheduling counter,
    # it is not persisted into the artifact, and `exact_evaluations` (which IS persisted) is
    # asserted equal below.
    assert parallel["exact_evaluations"] == sequential["exact_evaluations"]
    assert parallel["exact_cache_entries"] == sequential["exact_cache_entries"]
    assert set(parallel["routes"]) == set(sequential["routes"])
    for name in sequential["routes"]:
        assert parallel["routes"][name] == sequential["routes"][name], name
    assert parallel["h1_frontier"] == sequential["h1_frontier"]
    assert parallel["supported_3gw_frontier"] == sequential["supported_3gw_frontier"]
    assert parallel["families"] == sequential["families"]
    assert parallel["roll_baseline"] == sequential["roll_baseline"]
    assert parallel["paired_supported_3gw"] == sequential["paired_supported_3gw"]
    assert parallel["search_stats"] == sequential["search_stats"]

    # B. the cache contains the same keys AND the same values
    assert set(par_cache) == set(seq_cache)
    for key in seq_cache:
        assert par_cache[key]["scores"] == seq_cache[key]["scores"], key
        assert par_cache[key]["mean_gross"] == seq_cache[key]["mean_gross"], key
        assert par_cache[key]["policy"] == seq_cache[key]["policy"], key

    # the parallel run really used the pool, so this is not an accidental sequential label
    reported = parallel["parallel_exact"]
    assert reported["worker_count"] == workers
    assert reported["path"] == "parallel"
    assert reported["worker_jobs_completed"] == reported["unique_exact_keys"]


def test_stage1_parallel_reports_more_than_one_worker_process():
    """A parallel label must mean more than one process actually did the work."""

    universe, state, meta = _universe()
    scenario, config = _scenario(), _config()
    provider = _rich_provider(bonus=lambda p: 0.1)
    result, _cache = _run(universe, state, meta, scenario, config, provider, workers=4)
    reported = result["parallel_exact"]
    assert reported["unique_exact_keys"] > 1, "the fixture must present several exact units"
    assert len(reported.get("worker_pids") or []) > 1


def test_role_actionability_restriction_survives_parallel_scheduling():
    """E. The policy set a conflicted role removes must be the same set under the pool."""

    universe, state, meta = _universe()
    scenario, config = _scenario(), _config()
    # ONE player is role-conflicted.  Flagging the whole squad would leave no legal policy
    # (captain and vice must come from the XI) and the evaluator would raise, which is the
    # correct fail-closed answer but not the contract under test here.
    unresolved = {int(next(iter(state.players)).player_id)}
    provider = _rich_provider(unresolved=unresolved)

    sequential, seq_cache = _run(universe, state, meta, scenario, config, provider, workers=None)
    parallel, par_cache = _run(universe, state, meta, scenario, config, provider, workers=4)

    for name in sequential["routes"]:
        assert parallel["routes"][name] == sequential["routes"][name], name
    for key in seq_cache:
        policy = seq_cache[key]["policy"]
        assert int(policy.captain_id) not in unresolved
        assert int(policy.vice_captain_id) not in unresolved
        assert par_cache[key]["policy"] == policy


def test_captain_and_vice_terms_survive_parallel_scheduling():
    """F. The corrected bonus-inclusive armband and its fallback are unchanged by the pool."""

    universe, state, meta = _universe()
    scenario, config = _scenario(), _config()
    # A large bonus on one player makes the armband ordering bonus-sensitive, so a
    # dropped or reordered expected_bonus block would change the chosen captain.
    provider = _rich_provider(bonus=lambda p: 3.0 if int(p) in {int(x.player_id) for x in state.players} else 0.0)

    sequential, seq_cache = _run(universe, state, meta, scenario, config, provider, workers=None)
    parallel, par_cache = _run(universe, state, meta, scenario, config, provider, workers=4)
    for key in seq_cache:
        assert par_cache[key]["policy"].captain_id == seq_cache[key]["policy"].captain_id, key
        assert par_cache[key]["policy"].vice_captain_id == seq_cache[key]["policy"].vice_captain_id, key
    assert parallel["routes"] == sequential["routes"]


# ---------------------------------------------------------------------------
# C — completion order cannot be observed
# ---------------------------------------------------------------------------


def test_completion_order_cannot_change_the_stage1_result():
    """C. Different worker counts -- hence different completion interleavings -- agree exactly.

    The scheduler merges strictly by evaluation key (``_accept``), so arrival order is
    unobservable; the scheduler-level reverse-order proof already lives in
    ``test_p3_parallel_exact.py``.  Here the claim is asserted through the real Stage-1 call
    with every scheduling path the production runner can select, including a genuine
    one-process pool and the sequential fallback.
    """

    universe, state, meta = _universe()
    scenario, config = _scenario(), _config()
    provider = _rich_provider(bonus=lambda p: 0.2)

    baseline, baseline_cache = _run(universe, state, meta, scenario, config, provider, workers=None)
    for workers in (1, 2, 4):
        other, other_cache = _run(universe, state, meta, scenario, config, provider, workers=workers)
        assert set(other["routes"]) == set(baseline["routes"])
        for name in baseline["routes"]:
            assert other["routes"][name] == baseline["routes"][name], (workers, name)
        assert other["h1_frontier"] == baseline["h1_frontier"]
        assert other["roll_baseline"] == baseline["roll_baseline"]
        assert set(other_cache) == set(baseline_cache)
        for key in baseline_cache:
            assert other_cache[key]["scores"] == baseline_cache[key]["scores"], (workers, key)
        reported = other["parallel_exact"]
        assert reported["worker_count"] == workers
        assert reported["path"] == "parallel"


# ---------------------------------------------------------------------------
# D — a worker failure fails closed
# ---------------------------------------------------------------------------


def test_a_worker_failure_fails_the_stage1_batch_closed(monkeypatch):
    """D. One worker fails -> the Stage-1 batch raises and leaves NO cache entry behind.

    The promoted states are the REAL Stage-1 population captured from a real run (through
    ``_select_promoted``), so this asserts the contract on the tasks production would
    actually dispatch rather than on a hand-made stub.  ``ro.optimize`` exposes no executor
    seam, so the failing executor is injected at the scheduler entry point that Stage 1 now
    calls with exactly those tasks.
    """

    from fpl_brain import manager_lineup as ml  # noqa: F401  (import side effects)

    universe, state, meta = _universe()
    scenario, config = _scenario(), _config()
    provider = _rich_provider(bonus=lambda p: 0.2)

    captured = {}
    real_select = ro._select_promoted

    def capturing_select(final_states, config, **kwargs):
        promoted = real_select(final_states, config, **kwargs)
        captured["promoted"] = promoted
        return promoted

    monkeypatch.setattr(ro, "_select_promoted", capturing_select)
    worlds = {int(e): provider(e, [int(p.player_id) for p in state.players]) for e in (4, 5, 6, 7)}
    result = ro.optimize(
        universe=universe, initial_state=state, scenario=scenario, player_meta=meta,
        config=config, world_provider=provider, parallel_workers=None,
    )
    monkeypatch.undo()
    promoted = captured.get("promoted") or []
    assert promoted, "the fixture must produce promoted exact units"

    # Position metadata exactly as the optimizer derives it, so the scheduler sees the same
    # policy set it would in a real Stage-1 dispatch.
    rows = {int(row["player_id"]): row for row in universe["universe"]}
    positions_by_id = {int(pid): str(row["position"]) for pid, row in rows.items()}

    class _ExplodingExecutor:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def submit(self, *args, **kwargs):
            raise px.ParallelExactError("WORKER_DIED")

        def shutdown(self, *args, **kwargs):
            return None

    cache: dict = {}
    with pytest.raises(px.ParallelExactError):
        px.prefetch_exact_cache(
            promoted=promoted, worlds_by_event=worlds,
            positions_of=lambda ids: {int(p): positions_by_id.get(int(p), "MID") for p in ids},
            cache=cache, config=config, world_identity={int(e): "w" for e in worlds},
            worker_count=2, executor_factory=lambda *a, **k: _ExplodingExecutor(),
            positions_by_id=positions_by_id,
        )
    assert cache == {}, "a failed batch must leave no cache entry behind"
