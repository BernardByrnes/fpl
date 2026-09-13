#!/usr/bin/env python3
"""Phase 8B.1 — nested-budget search-stability ladder and certification.

Narrow remediation of SEARCH_NOT_STABLE_AT_CURRENT_BUDGET.  No redesign, no new
predictive runs, no threshold change, no recommendation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fpl_brain import candidate_universe as cu
from fpl_brain import manager_worlds, route_comparator as rc
from fpl_brain import route_optimizer as ro, route_stability as rs
from fpl_brain.config import config_path, load_config
from fpl_brain.database import connect_database
from fpl_brain.planning import get_planning_context

PHASE8B1 = "phase8b1_search_stability_v1.0.0"
PLANNING_CUTOFF = "2026-09-11T10:16:51Z"
EVENTS = (4, 5, 6)
RUNS = {4: {"minutes": 64, "team": 65, "rate": 67, "xpts": 70, "mc": 71},
        5: {"minutes": 80, "team": 81, "rate": 83, "xpts": 86, "mc": 87},
        6: {"minutes": 88, "team": 89, "rate": 91, "xpts": 94, "mc": 95}}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gw", type=int, default=4)
    parser.add_argument("--config")
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--exact-budget", type=int, default=30)
    parser.add_argument("--policy-worlds", type=int, default=300)
    parser.add_argument("--search-n", type=int, default=20)
    parser.add_argument("--rescue-k", type=int, default=25)
    parser.add_argument("--budgets", default="12,24,48")
    parser.add_argument("--cache-dir", default="data/cache/manager_worlds")
    parser.add_argument("--out")
    args = parser.parse_args(argv)

    budgets = [int(part) for part in str(args.budgets).split(",") if part.strip()]
    config = load_config(args.config)
    conn = connect_database(config_path(config, "database"))
    try:
        started = time.time()
        context = get_planning_context(conn, int(config.get("fpl_entry_id")), int(args.gw),
                                       as_of=PLANNING_CUTOFF, season=config.get("season"))
        squad = manager_worlds.resolve_squad(context, conn)
        universe_path = config_path(config, "exports_dir") / "candidates" / f"gw{int(args.gw):02d}" / "candidate_universe.json"
        universe = json.loads(universe_path.read_text(encoding="utf-8"))

        base_config = ro.OptimizerConfig(
            events=EVENTS, search_draws=int(args.draws), beam_width=budgets[0],
            exact_evaluation_budget=int(args.exact_budget), policy_selection_worlds=int(args.policy_worlds),
            search_n_per_criterion=int(args.search_n), rescue_top_k_per_position=int(args.rescue_k),
        )
        initial_state = rc.build_route_state(conn, context, squad)
        pool = ro.build_search_pool(universe, squad["squad_ids"], base_config)
        union = ro.union_player_ids(initial_state, pool["pool_ids"])
        bundles = {event: rc.EventBundle(
            event=event, minutes_run_id=RUNS[event]["minutes"], team_run_id=RUNS[event]["team"],
            rate_run_id=RUNS[event]["rate"], xpts_run_id=RUNS[event]["xpts"], mc_run_id=RUNS[event]["mc"],
            simulations=int(args.draws), seed=base_config.seed, planning_cutoff=PLANNING_CUTOFF,
        ) for event in EVENTS}
        cache_dir = Path(args.cache_dir) if args.cache_dir else None
        prebuilt = {}
        world_info = {}
        for event in EVENTS:
            matrix, info = ro.build_event_worlds(conn, bundles, event, union, base_config, cache_dir=cache_dir)
            prebuilt[event] = matrix
            world_info[str(event)] = {**info, "worlds": int(matrix["worlds"]), "union_players": len(union)}

        prices = cu.latest_price_snapshot(conn, int(args.gw))
        scenario = rc.flat_current_price_scenario(prices, EVENTS)
        player_meta = rc.load_player_meta(conn, union)
        exact_cache: dict = {}

        def run(budget, nested_prior=None, required=None):
            cfg = rs.budget_config(int(budget), base_config)
            mark = time.time()
            result = ro.optimize(universe=universe, initial_state=initial_state, scenario=scenario,
                                 player_meta=player_meta, bundles=bundles, config=cfg,
                                 prebuilt_worlds=prebuilt, exact_cache=exact_cache,
                                 nested_prior=nested_prior, required_routes=required)
            result["ladder_runtime_s"] = round(time.time() - mark, 3)
            return result

        # (1) Reproduce the ORIGINAL non-nested behaviour at B12 and B24.
        legacy_small = run(budgets[0])
        legacy_large = run(budgets[1] if len(budgets) > 1 else budgets[0] * 2)
        legacy_comparison = ro.search_stability(legacy_small, legacy_large)
        legacy_monotonic = ro.monotonic_check(legacy_small, legacy_large)

        # (2) Nested ladder: B12 is the same run; B24/B48 inherit smaller survivors.
        ladder = [legacy_small]
        summaries = [{**rs.summarize(legacy_small, budgets[0]), "ladder_runtime_s": legacy_small["ladder_runtime_s"]}]
        prior = ro.nested_budget_view(legacy_small)
        prior_result = legacy_small
        for budget in budgets[1:]:
            result = run(budget, nested_prior=prior, required=ro.required_routes_from(prior_result))
            ladder.append(result)
            summaries.append({**rs.summarize(result, budget), "ladder_runtime_s": result["ladder_runtime_s"]})
            prior = ro.nested_budget_view(result)
            prior_result = result
        certification = rs.certify(ladder, budgets)
        monotonic = [ro.monotonic_check(ladder[i - 1], ladder[i]) for i in range(1, len(ladder))]
        identity = [ro.cross_budget_score_identity(ladder[i - 1], ladder[i]) for i in range(1, len(ladder))]

        artifact = {
            "phase_version": PHASE8B1,
            "planning_cutoff": PLANNING_CUTOFF,
            "supported_events": list(EVENTS),
            "flags": sorted(set(ro.OptimizerConfig().as_dict()["labels"]) | {certification["status_flag"]}),
            "candidate_artifact": str(universe_path),
            "world_info": world_info,
            "union_players": len(union),
            "budgets": budgets,
            "original_non_nested": {
                "comparison": legacy_comparison, "monotonic": legacy_monotonic,
                "b12_best_3gw": (ro.best_route_family(legacy_small, "supported_3gw_net_core") or {}).get("supported_3gw_net_core"),
                "b24_best_3gw": (ro.best_route_family(legacy_large, "supported_3gw_net_core") or {}).get("supported_3gw_net_core"),
            },
            "ladder_summaries": summaries,
            "ladder_records": [
                {key: value for key, value in result.items() if key not in ("promoted_routes", "level_survivors")}
                for result in ladder
            ],
            "monotonic": monotonic,
            "cross_budget_score_identity": identity,
            "certification": certification,
            "timing_s": {"total": round(time.time() - started, 3)},
            "no_recommendation": True,
        }
        out_dir = Path(args.out) if args.out else config_path(config, "exports_dir") / "optimizer" / f"gw{int(args.gw):02d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / "stability_ladder.json"
        target.write_text(json.dumps(cu.jsonable(artifact), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                          encoding="utf-8")

        print(f"phase 8b.1 complete: GW{args.gw} budgets={budgets} union={len(union)} cutoff={PLANNING_CUTOFF}")
        print(f"  original non-nested: B{budgets[0]}={artifact['original_non_nested']['b12_best_3gw']:.4f} "
              f"B{budgets[1]}={artifact['original_non_nested']['b24_best_3gw']:.4f} "
              f"leader_changed={legacy_comparison['leader_changed']} delta={legacy_comparison['value_delta']:.4f}")
        cache_sources = ", ".join(f"{e}:{world_info[str(e)]['source']}" for e in EVENTS)
        print(f"  world cache: {cache_sources}")
        for summary in summaries:
            print(f"  B{summary['budget']}: 3GW={summary['supported_3gw_leader_value']:.4f} "
                  f"H1={summary['h1_leader_value']:.4f} states={summary['states_generated']} "
                  f"inherited={summary['nested_inherited']} pruned={summary['heuristic_pruned']} "
                  f"promoted={summary['promoted_routes']} exact={summary['exact_evaluations']} "
                  f"frontier3gw={summary['supported_3gw_frontier_size']} "
                  f"runtime={summary.get('ladder_runtime_s') or (summary.get('timing_s') or {}).get('total')}s")
        print("  monotonic: " + ", ".join(f"{m['status']}({k})" for m in monotonic for k in ('h1_net_core','supported_3gw_net_core')))
        print(f"  certification: {certification['overall']} {certification['status_flag']}")
        for name, item in certification["criteria"].items():
            print(f"    {name}: {item['status']}")
        print(f"  artifact={target} total={artifact['timing_s']['total']}s")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
