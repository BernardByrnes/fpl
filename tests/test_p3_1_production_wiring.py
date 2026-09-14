"""P3.1 — production parallelism wiring, tested through the REAL call graph.

P3 proved the scheduler.  P3.1 must prove the PRODUCTION RUNNER actually requests it, on the
two 10,000-draw paths it was proven for, and that Stage 1 does not silently acquire it.

These tests therefore drive ``scripts/run_four_gw_decision.py`` and
``fpl_brain/finalist_refinement.py`` — not ``parallel_exact`` directly:

  * Stage-2 ``optimize`` receives ``parallel_workers = PRODUCTION_PARALLEL_EXACT_WORKERS``;
  * the ONE stability escalation receives the same value;
  * the SAME run-scoped ``exact_cache`` object is retained across both;
  * Stage 1 does NOT receive it (asserted on the runner's own AST, because that call site
    cannot be driven without a database);
  * the library defaults stay conservative, so no other caller inherits a pool;
  * the parallel path is genuinely REACHED and reports its worker processes — i.e. there is
    no silent fallback to sequential when the pool is healthy.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

from fpl_brain import finalist_refinement as fr
from fpl_brain import parallel_exact as px
from fpl_brain import route_optimizer as ro
from test_route_optimizer import _config, _provider, _scenario, _universe

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_four_gw_decision as runner  # noqa: E402

RUNNER_SOURCE = (SCRIPTS / "run_four_gw_decision.py").read_text(encoding="utf-8")
RUNNER_TREE = ast.parse(RUNNER_SOURCE)
OPTIMIZER_SOURCE = (Path(__file__).resolve().parents[1] / "fpl_brain"
                    / "route_optimizer.py").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# AST helpers: assert on the real call sites, not on a copy of them
# ---------------------------------------------------------------------------

def _calls(named: str):
    """Every ``... .named(...)`` call node in the runner, as (call, keywords)."""

    found = []
    for node in ast.walk(RUNNER_TREE):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr == named:
            found.append(node)
    return found


def _stage1_optimize_call():
    """The runner's ``stage1_result = ro.optimize(...)`` call node."""

    for node in ast.walk(RUNNER_TREE):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            func = node.value.func
            if "stage1_result" in targets and isinstance(func, ast.Attribute) \
                    and func.attr == "optimize":
                return node.value
    raise AssertionError("the runner has no `stage1_result = ...optimize(...)` call")


def _keywords(call) -> set:
    return {keyword.arg for keyword in call.keywords}


# ---------------------------------------------------------------------------
# the declared production constants
# ---------------------------------------------------------------------------

def test_production_constant_is_the_measured_four():
    assert runner.PRODUCTION_PARALLEL_EXACT_WORKERS == 4, (
        "the production runner must request the worker count P3 measured (4); "
        "see the P3 report's worker-count table"
    )


def test_library_defaults_stay_conservative():
    """The opt-in must be explicit: nothing else inherits a process pool."""

    import inspect

    signature = inspect.signature(ro.optimize)
    assert signature.parameters["parallel_workers"].default is None, (
        "route_optimizer.optimize must keep a sequential default; production opts in "
        "explicitly so the choice is auditable"
    )
    assert px.DEFAULT_WORKER_COUNT == 1, (
        "parallel_exact.DEFAULT_WORKER_COUNT must stay conservative (never all logical CPUs)"
    )


def test_the_cli_default_is_the_production_constant():
    defaults = {}
    for node in ast.walk(RUNNER_TREE):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr == "add_argument":
            if node.args and isinstance(node.args[0], ast.Constant):
                for keyword in node.keywords:
                    if keyword.arg == "default":
                        defaults[node.args[0].value] = ast.unparse(keyword.value)
    assert defaults.get("--parallel-workers") == "PRODUCTION_PARALLEL_EXACT_WORKERS", defaults


def test_stage1_call_site_does_not_request_parallelism():
    """Stage 1 stays sequential: the 10k target was Stage-2 and the escalation only."""

    assert "parallel_workers" not in _keywords(_stage1_optimize_call()), (
        "the Stage-1 optimize call must not pass parallel_workers"
    )


def test_runner_never_enables_a_silent_sequential_fallback():
    """P3 fails the batch closed; production must not be able to switch that off."""

    for call in _calls("optimize") + _calls("refine_finalists"):
        assert "sequential_fallback" not in _keywords(call)
    assert "prefetch_exact_cache(" in OPTIMIZER_SOURCE
    prefetch = [node for node in ast.walk(ast.parse(OPTIMIZER_SOURCE))
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "prefetch_exact_cache"]
    assert prefetch, "optimize no longer calls the scheduler"
    for call in prefetch:
        assert "sequential_fallback" not in _keywords(call), (
            "optimize must not expose or enable the sequential fallback: a worker failure has "
            "to fail the batch closed"
        )


def test_runner_artifact_records_the_scheduling_choice():
    assert "parallel_exact_scheduling" in RUNNER_SOURCE
    assert "SCHEDULING_ONLY_BIT_IDENTICAL" in RUNNER_SOURCE


# ---------------------------------------------------------------------------
# the real production call graph, with a spy around the REAL optimize
# ---------------------------------------------------------------------------

@pytest.fixture
def spy_optimize(monkeypatch):
    """Wrap the real ``ro.optimize`` so its kwargs can be asserted without faking it."""

    recorded: list[dict] = []
    real = ro.optimize

    def spy(*args, **kwargs):
        recorded.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(ro, "optimize", spy)
    return recorded


def _stage1_result(universe, state, meta, scenario, config, provider):
    return ro.optimize(universe=universe, initial_state=state, scenario=scenario,
                       player_meta=meta, config=config, world_provider=provider)


def test_stage2_refinement_receives_the_production_worker_count_and_cache(spy_optimize):
    universe, state, meta = _universe()
    scenario, config, provider = _scenario(), _config(), _provider()
    stage1 = _stage1_result(universe, state, meta, scenario, config, provider)
    spy_optimize.clear()          # the Stage-1 call above is covered by its own test
    cache: dict = {}
    refinement = fr.refine_finalists(
        universe=universe, initial_state=state, scenario=scenario, player_meta=meta,
        base_config=config, stage1_result=stage1, world_provider=provider,
        verify_prefix=False, exact_cache=cache,
        parallel_workers=runner.PRODUCTION_PARALLEL_EXACT_WORKERS)
    assert spy_optimize, "the refinement must have called optimize"
    assert len(spy_optimize) == 1, "the refinement must call optimize exactly once"
    for kwargs in spy_optimize:
        assert kwargs.get("parallel_workers") == 4
        assert kwargs.get("exact_cache") is cache
    refined = refinement["refined"]
    reported = refined.get("parallel_exact") or {}
    assert reported.get("path") == "parallel", "the pool was not used"
    assert reported.get("worker_count") == 4
    assert reported.get("worker_jobs_cancelled") == 0


def test_escalation_receives_the_same_worker_count_and_the_same_cache(spy_optimize):
    universe, state, meta = _universe()
    scenario, config, provider = _scenario(), _config(), _provider()
    stage1 = _stage1_result(universe, state, meta, scenario, config, provider)
    selection = fr.select_finalists(stage1)
    partials = fr.finalist_partials(stage1, selection)
    cache: dict = {}
    # prime the cache AND capture the worlds the way the production Stage-2 refinement does
    refinement = fr.refine_finalists(
        universe=universe, initial_state=state, scenario=scenario, player_meta=meta,
        base_config=config, stage1_result=stage1, world_provider=provider,
        verify_prefix=False, exact_cache=cache, stage2_draws=config.search_draws,
        parallel_workers=runner.PRODUCTION_PARALLEL_EXACT_WORKERS)
    spy_optimize.clear()

    escalation = runner._escalation_runner(
        universe=universe, initial_state=state, scenario=scenario, player_meta=meta,
        bundles=None, conn=None, base_config=config, stage1_result=stage1,
        stage2_draws=config.search_draws,
        prebuilt_worlds=refinement["prebuilt_worlds"], finalist_partials=partials,
        exact_cache=cache, cancel_probe=None,
        parallel_workers=runner.PRODUCTION_PARALLEL_EXACT_WORKERS)
    result = escalation(12)
    assert spy_optimize, "the escalation must have called optimize"
    for kwargs in spy_optimize:
        assert kwargs.get("parallel_workers") == 4
        assert kwargs.get("exact_cache") is cache, (
            "the escalation must reuse the SAME run-scoped cache as the Stage-2 refinement"
        )
    reported = result.get("parallel_exact") or {}
    assert reported.get("worker_count") == 4
    # The escalation re-scores the finalists, whose entries Stage 2 already produced, so it
    # must report cache hits rather than recomputing them from scratch.
    assert reported.get("parent_cache_hits", 0) > 0


def test_parallel_stage2_is_bit_identical_to_sequential_stage2():
    """The production wiring changes scheduling only."""

    universe, state, meta = _universe()
    scenario, config, provider = _scenario(), _config(), _provider()
    stage1 = _stage1_result(universe, state, meta, scenario, config, provider)

    sequential = fr.refine_finalists(
        universe=universe, initial_state=state, scenario=scenario, player_meta=meta,
        base_config=config, stage1_result=stage1, world_provider=provider,
        verify_prefix=False, exact_cache={}, parallel_workers=None)["refined"]
    parallel = fr.refine_finalists(
        universe=universe, initial_state=state, scenario=scenario, player_meta=meta,
        base_config=config, stage1_result=stage1, world_provider=provider,
        verify_prefix=False, exact_cache={},
        parallel_workers=runner.PRODUCTION_PARALLEL_EXACT_WORKERS)["refined"]
    assert parallel["routes"] == sequential["routes"]
    assert parallel["supported_3gw_frontier"] == sequential["supported_3gw_frontier"]
    assert parallel["h1_frontier"] == sequential["h1_frontier"]
    assert parallel["paired_supported_3gw"] == sequential["paired_supported_3gw"]
    assert parallel["families"] == sequential["families"]
    assert parallel["roll_baseline"] == sequential["roll_baseline"]


def test_stage1_stays_sequential_in_the_real_call_graph(spy_optimize):
    """The Stage-1 call the runner makes carries no worker count."""

    universe, state, meta = _universe()
    scenario, config, provider = _scenario(), _config(), _provider()
    _stage1_result(universe, state, meta, scenario, config, provider)
    for kwargs in spy_optimize:
        assert kwargs.get("parallel_workers") is None
    # and the Stage-1 result therefore reports no parallel scheduling at all
    result = _stage1_result(universe, state, meta, scenario, config, provider)
    assert result.get("parallel_exact") is None


def test_production_worker_count_is_a_real_pool_not_a_sequential_label(spy_optimize):
    """``worker_count=4`` must mean MORE THAN ONE process actually did the work."""

    universe, state, meta = _universe()
    scenario, config, provider = _scenario(), _config(), _provider()
    stage1 = _stage1_result(universe, state, meta, scenario, config, provider)
    refinement = fr.refine_finalists(
        universe=universe, initial_state=state, scenario=scenario, player_meta=meta,
        base_config=config, stage1_result=stage1, world_provider=provider,
        verify_prefix=False, exact_cache={},
        parallel_workers=runner.PRODUCTION_PARALLEL_EXACT_WORKERS)
    reported = (refinement["refined"].get("parallel_exact") or {})
    assert reported.get("path") == "parallel"
    assert reported.get("worker_jobs_completed") == reported.get("unique_exact_keys")
    # Either several workers did work, or the batch had a single unique key (which the
    # synthetic fixture does not have); assert the stronger form and say why.
    assert reported.get("unique_exact_keys", 0) > 1
    assert len(reported.get("worker_pids") or []) > 1, (
        "a healthy 4-worker pool over several units must show more than one worker PID"
    )
