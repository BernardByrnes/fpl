"""Phase 8B.1 — search-stability ladder and pre-declared certification.

Builds a NESTED budget ladder (each larger budget inherits the smaller budget's
survivors and exact-evaluated leaders) and applies the pre-declared certification
rule.  It never changes the optimizer's objective, scoring, pool or thresholds.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

from . import route_optimizer as ro

PHASE8B1_VERSION = "route_stability_v8b1_1.0.0"
LADDER_BUDGETS = (12, 24, 48)
MATERIAL_CORE = ro.MATERIAL_FRONTIER_CHANGE_CORE  # 0.25, unchanged


def budget_config(budget: int, base: ro.OptimizerConfig) -> ro.OptimizerConfig:
    return ro.OptimizerConfig(
        events=base.events, search_draws=base.search_draws, seed=base.seed,
        beam_width=int(budget), exact_evaluation_budget=int(base.exact_evaluation_budget),
        policy_selection_worlds=base.policy_selection_worlds,
        max_auto_hit_points_per_event=base.max_auto_hit_points_per_event,
        singles_per_out=base.singles_per_out, max_transfers_per_event=base.max_transfers_per_event,
        rescue_top_k_per_position=base.rescue_top_k_per_position,
        search_n_per_criterion=base.search_n_per_criterion, retention_lenses=base.retention_lenses,
    )


def summarize(result: Mapping[str, Any], budget: int) -> dict[str, Any]:
    stats = result.get("search_stats") or {}
    h1 = ro.best_route_family(result, "h1_net_core")
    w3 = ro.best_route_family(result, "supported_3gw_net_core")
    return {
        "budget": int(budget),
        "states_generated": int(stats.get("partial_states") or 0),
        "safe_dedup": int(stats.get("safe_dedup") or 0),
        "heuristic_pruned": int(stats.get("heuristic_pruned") or 0),
        "heuristic_retained": int(stats.get("heuristic_retained") or 0),
        "nested_inherited": int(stats.get("nested_inherited") or 0),
        "levels": stats.get("levels") or [],
        "promoted_routes": int(result.get("promoted_route_count") or 0),
        "exact_evaluations": int(result.get("exact_evaluations") or 0),
        "exact_cache_entries": int(result.get("exact_cache_entries") or 0),
        "h1_leader_signature": (h1 or {}).get("signature"),
        "h1_leader_value": (h1 or {}).get("h1_net_core"),
        "supported_3gw_leader_signature": (w3 or {}).get("signature"),
        "supported_3gw_leader_value": (w3 or {}).get("supported_3gw_net_core"),
        "h1_frontier_size": len(result.get("h1_frontier") or []),
        "supported_3gw_frontier_size": len(result.get("supported_3gw_frontier") or []),
        "h1_frontier_signatures": ro.frontier_family_signatures(result),
        "timing_s": result.get("timing_s"),
        "roll_baseline": result.get("roll_baseline"),
    }


def new_b48_family_materially_better(b24: Mapping[str, Any], b48: Mapping[str, Any]) -> dict[str, Any]:
    """E: a completely new B48 family beating the B24 leader by more than 0.25 CORE."""

    small = set(ro.route_family_scores(b24))
    large = ro.route_family_scores(b48)
    leader = ro.best_route_family(b24, "supported_3gw_net_core") or {}
    leader_value = float(leader.get("supported_3gw_net_core", 0.0))
    offenders = [
        {"signature": signature, "value": large[signature]["supported_3gw_net_core"],
         "margin": large[signature]["supported_3gw_net_core"] - leader_value}
        for signature in sorted(set(large) - small)
        if large[signature]["supported_3gw_net_core"] > leader_value + MATERIAL_CORE
    ]
    return {"status": "PASS" if not offenders else "FAIL", "offenders": offenders,
            "b24_leader_value": leader_value, "threshold_core": MATERIAL_CORE}


def frontier_signatures_for(result: Mapping[str, Any], objective_key: str) -> set[str]:
    """Canonical frontier signatures for ONE objective (H1 or SUPPORTED_3GW)."""

    lookup = {
        str(record.get("route_id", key)): str(record.get("canonical_family_signature") or record.get("family_signature"))
        for key, record in (result.get("routes") or {}).items()
    }
    return {lookup[name] for name in (result.get(objective_key) or []) if name in lookup}


def frontier_stability(smaller: Mapping[str, Any], larger: Mapping[str, Any]) -> dict[str, Any]:
    """G: frontier strategies stable, evaluated PER OBJECTIVE.

    A frontier addition is material only if it materially improves that
    objective's frontier (exceeds the smaller run's frontier best by more than
    the unchanged 0.25 CORE threshold); near-equivalent variants that appear or
    disappear are benign.  Comparing the H1 and SUPPORTED_3GW frontiers on a
    single mixed value would flag every H1-specialist addition spuriously.
    """

    small = ro.route_family_scores(smaller)
    large = ro.route_family_scores(larger)
    report: dict[str, Any] = {}
    material_any = False
    for objective, key in (("h1_net_core", "h1_frontier"),
                           ("supported_3gw_net_core", "supported_3gw_frontier")):
        small_sigs = frontier_signatures_for(smaller, key)
        large_sigs = frontier_signatures_for(larger, key)
        added = sorted(large_sigs - small_sigs)
        removed = sorted(small_sigs - large_sigs)

        def value(source, signature):
            return source.get(signature, {}).get(objective)

        best_small = max((value(small, s) for s in small_sigs if value(small, s) is not None), default=float("-inf"))
        best_large = max((value(large, s) for s in large_sigs if value(large, s) is not None), default=float("-inf"))
        material_added = [
            s for s in added
            if value(large, s) is not None and best_small is not float("-inf")
            and value(large, s) > best_small + MATERIAL_CORE
        ]
        material_removed = [
            s for s in removed
            if value(small, s) is not None and best_large is not float("-inf")
            and value(small, s) > best_large + MATERIAL_CORE
        ]
        objective_material = bool(material_added) or bool(material_removed)
        material_any = material_any or objective_material
        report[objective] = {
            "status": "FAIL" if objective_material else "PASS",
            "smaller_frontier_size": len(small_sigs), "larger_frontier_size": len(large_sigs),
            "smaller_best": None if best_small is float("-inf") else best_small,
            "larger_best": None if best_large is float("-inf") else best_large,
            "added_signatures": added, "removed_signatures": removed,
            "material_added": material_added, "material_removed": material_removed,
            "benign_added_count": len(added) - len(material_added),
            "benign_removed_count": len(removed) - len(material_removed),
        }
    return {"status": "FAIL" if material_any else "PASS", "per_objective": report,
            "threshold_core": MATERIAL_CORE}


def certify(ladder: Sequence[Mapping[str, Any]], budgets: Sequence[int]) -> dict[str, Any]:
    """Apply the pre-declared certification rule to a nested ladder."""

    criteria: dict[str, Any] = {}
    monotonic = [ro.monotonic_check(ladder[i - 1], ladder[i]) for i in range(1, len(ladder))]
    identity = [ro.cross_budget_score_identity(ladder[i - 1], ladder[i]) for i in range(1, len(ladder))]
    criteria["A_monotonic"] = {
        "status": "PASS" if all(item["status"] == "PASS" for item in monotonic) else "FAIL",
        "details": monotonic,
    }
    criteria["B_score_identity"] = {
        "status": "PASS" if all(item["status"] == "PASS" for item in identity) else "FAIL",
        "details": identity,
    }

    if len(ladder) >= 2:
        small, large = ladder[-2], ladder[-1]
        small_3gw = ro.best_route_family(small, "supported_3gw_net_core") or {}
        large_3gw = ro.best_route_family(large, "supported_3gw_net_core") or {}
        small_h1 = ro.best_route_family(small, "h1_net_core") or {}
        large_h1 = ro.best_route_family(large, "h1_net_core") or {}
        delta_3gw = abs(float(large_3gw.get("supported_3gw_net_core", 0.0)) - float(small_3gw.get("supported_3gw_net_core", 0.0)))
        delta_h1 = abs(float(large_h1.get("h1_net_core", 0.0)) - float(small_h1.get("h1_net_core", 0.0)))
        leader_equal = small_3gw.get("signature") == large_3gw.get("signature")
        criteria["C_last_pair_3gw_delta"] = {
            "status": "PASS" if delta_3gw <= MATERIAL_CORE else "FAIL",
            "delta": delta_3gw, "threshold_core": MATERIAL_CORE, "pair": [int(budgets[-2]), int(budgets[-1])],
        }
        criteria["D_last_pair_h1_delta"] = {
            "status": "PASS" if delta_h1 <= MATERIAL_CORE else "FAIL",
            "delta": delta_h1, "threshold_core": MATERIAL_CORE,
        }
        criteria["E_new_family"] = new_b48_family_materially_better(small, large)
        criteria["F_leader"] = {
            "status": "PASS" if (leader_equal or delta_3gw <= MATERIAL_CORE) else "FAIL",
            "leader_equal": leader_equal, "delta": delta_3gw, "threshold_core": MATERIAL_CORE,
        }
        criteria["G_frontier"] = frontier_stability(small, large)
    else:
        for key in ("C_last_pair_3gw_delta", "D_last_pair_h1_delta", "E_new_family", "F_leader", "G_frontier"):
            criteria[key] = {"status": "FAIL", "reason": "INSUFFICIENT_BUDGETS"}

    overall = "COMPLETE" if all(item["status"] == "PASS" for item in criteria.values()) else "PARTIAL"
    status_flag = (ro.SEARCH_STABLE if overall == "COMPLETE" else ro.SEARCH_UNSTABLE)
    return {"criteria": criteria, "overall": overall, "status_flag": status_flag,
            "material_threshold_core": MATERIAL_CORE, "budgets": [int(b) for b in budgets]}


def run_ladder(
    *,
    universe: Mapping[str, Any],
    initial_state,
    scenario,
    player_meta: Mapping[int, Any],
    base_config: ro.OptimizerConfig,
    budgets: Sequence[int] = LADDER_BUDGETS,
    bundles: Mapping[int, Any] | None = None,
    conn=None,
    prebuilt_worlds: Mapping[int, Any] | None = None,
    cache_dir=None,
    world_provider=None,
    exact_cache: dict | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run the nested budget ladder and certify it."""

    import time

    exact_cache = exact_cache if exact_cache is not None else {}
    ladder: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    prior_view: dict[str, Any] | None = None
    prior_result: Mapping[str, Any] | None = None
    for budget in budgets:
        config = budget_config(int(budget), base_config)
        required = ro.required_routes_from(prior_result) if prior_result is not None else None
        started = time.time()
        result = ro.optimize(
            universe=universe, initial_state=initial_state, scenario=scenario, player_meta=player_meta,
            bundles=bundles, conn=conn, config=config, cache_dir=cache_dir, world_provider=world_provider,
            prebuilt_worlds=prebuilt_worlds, exact_cache=exact_cache,
            nested_prior=prior_view, required_routes=required,
        )
        result["ladder_budget"] = int(budget)
        result["ladder_runtime_s"] = round(time.time() - started, 3)
        if progress:
            progress(f"B{budget}: promoted={result.get('promoted_route_count')} "
                     f"exact={result.get('exact_evaluations')} runtime={result['ladder_runtime_s']}s")
        ladder.append(result)
        summaries.append({**summarize(result, int(budget)), "ladder_runtime_s": result["ladder_runtime_s"]})
        prior_view = ro.nested_budget_view(result)
        prior_result = result

    certification = certify(ladder, budgets)
    monotonic = [ro.monotonic_check(ladder[i - 1], ladder[i]) for i in range(1, len(ladder))]
    identity = [ro.cross_budget_score_identity(ladder[i - 1], ladder[i]) for i in range(1, len(ladder))]
    return {
        "phase_version": PHASE8B1_VERSION,
        "budgets": [int(b) for b in budgets],
        "summaries": summaries,
        "ladder_results": ladder,
        "monotonic": monotonic,
        "cross_budget_score_identity": identity,
        "certification": certification,
        "material_threshold_core": MATERIAL_CORE,
        "no_recommendation": True,
    }
