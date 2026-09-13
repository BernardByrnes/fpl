#!/usr/bin/env python3
"""Phase 8B — bounded multi-Gameweek route optimizer (engineering validation).

Descriptive only: no recommendation, no execution, no chip.  It searches GW4-GW6
(the certified contiguous window) and reports an auditable Pareto set.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fpl_brain import candidate_universe as cu
from fpl_brain import manager_worlds, packet as packet_mod, route_comparator as rc
from fpl_brain import route_optimizer as ro
from fpl_brain import transfer_state as ts
from fpl_brain.config import config_path, load_config
from fpl_brain.database import connect_database
from fpl_brain.planning import get_planning_context

PHASE8B = "phase8b_route_optimizer_v1.0.0"
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
    parser.add_argument("--beam", type=int, default=12)
    parser.add_argument("--exact-budget", type=int, default=30)
    parser.add_argument("--policy-worlds", type=int, default=300)
    parser.add_argument("--search-n", type=int, default=20)
    parser.add_argument("--rescue-k", type=int, default=25)
    parser.add_argument("--cache-dir")
    parser.add_argument("--out")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    conn = connect_database(config_path(config, "database"))
    try:
        started = time.time()
        context = get_planning_context(conn, int(config.get("fpl_entry_id")), int(args.gw),
                                       as_of=PLANNING_CUTOFF, season=config.get("season"))
        squad = manager_worlds.resolve_squad(context, conn)

        universe_path = config_path(config, "exports_dir") / "candidates" / f"gw{int(args.gw):02d}" / "candidate_universe.json"
        if not universe_path.exists():
            print(f"phase 8b failed: candidate universe artifact missing at {universe_path}", file=sys.stderr)
            return 3
        universe = json.loads(universe_path.read_text(encoding="utf-8"))

        base_config = ro.OptimizerConfig(
            events=EVENTS, search_draws=int(args.draws), beam_width=int(args.beam),
            exact_evaluation_budget=int(args.exact_budget), policy_selection_worlds=int(args.policy_worlds),
            search_n_per_criterion=int(args.search_n), rescue_top_k_per_position=int(args.rescue_k),
        )
        expanded_budget_config = ro.OptimizerConfig(
            events=EVENTS, search_draws=int(args.draws), beam_width=int(args.beam) * 2,
            exact_evaluation_budget=int(args.exact_budget), policy_selection_worlds=int(args.policy_worlds),
            search_n_per_criterion=int(args.search_n), rescue_top_k_per_position=int(args.rescue_k),
        )
        expanded_pool_config = ro._with_n(base_config, int(args.search_n) + 10)

        initial_state = rc.build_route_state(conn, context, squad)

        # One world set per event, wide enough for BOTH pools (capture superset).
        pool_base = ro.build_search_pool(universe, squad["squad_ids"], base_config)
        pool_expanded = ro.build_search_pool(universe, squad["squad_ids"], expanded_pool_config)
        union = ro.union_player_ids(initial_state, sorted(set(pool_base["pool_ids"]) | set(pool_expanded["pool_ids"])))

        bundles = {event: rc.EventBundle(
            event=event, minutes_run_id=RUNS[event]["minutes"], team_run_id=RUNS[event]["team"],
            rate_run_id=RUNS[event]["rate"], xpts_run_id=RUNS[event]["xpts"], mc_run_id=RUNS[event]["mc"],
            simulations=int(args.draws), seed=base_config.seed, planning_cutoff=PLANNING_CUTOFF,
        ) for event in EVENTS}

        cache_dir = Path(args.cache_dir) if args.cache_dir else None
        world_info = {}
        prebuilt = {}
        for event in EVENTS:
            matrix, info = ro.build_event_worlds(conn, bundles, event, union, base_config, cache_dir=cache_dir)
            prebuilt[event] = matrix
            world_info[str(event)] = {**info, "worlds": int(matrix["worlds"]), "union_players": len(union)}

        prices = cu.latest_price_snapshot(conn, int(args.gw))
        scenario = rc.flat_current_price_scenario(prices, EVENTS)
        player_meta = rc.load_player_meta(conn, union)

        shared_cache: dict = {}
        base = ro.optimize(universe=universe, initial_state=initial_state, scenario=scenario,
                           player_meta=player_meta, bundles=bundles, config=base_config,
                           prebuilt_worlds=prebuilt, exact_cache=shared_cache)
        budget = ro.optimize(universe=universe, initial_state=initial_state, scenario=scenario,
                             player_meta=player_meta, bundles=bundles, config=expanded_budget_config,
                             prebuilt_worlds=prebuilt, exact_cache=shared_cache)
        pool = ro.optimize(universe=universe, initial_state=initial_state, scenario=scenario,
                           player_meta=player_meta, bundles=bundles, config=expanded_pool_config,
                           prebuilt_worlds=prebuilt, exact_cache=shared_cache)

        for result, cutoff in ((base, PLANNING_CUTOFF), (budget, PLANNING_CUTOFF), (pool, PLANNING_CUTOFF)):
            result["planning_cutoff"] = cutoff
            result["planning_context_hash"] = _context_hash(context)
            result["candidate_artifact"] = str(universe_path)
            result["full_universe_count"] = len(universe["universe"])
            result["phase"] = PHASE8B

        stability_budget = ro.search_stability(base, budget)
        stability_pool = ro.search_stability(base, pool)
        overall = (ro.SEARCH_STABLE if stability_budget["status"] == ro.SEARCH_STABLE
                   and stability_pool["status"] == ro.SEARCH_STABLE else ro.SEARCH_UNSTABLE)
        base["stability"] = {"budget_expansion": stability_budget, "candidate_pool_expansion": stability_pool,
                             "overall": overall, "overall_flag": overall}
        base["search_n_base"] = int(args.search_n)
        base["search_n_expanded"] = int(args.search_n) + 10
        base["world_info"] = world_info
        base["timing_s"]["total_with_stability"] = round(time.time() - started, 3)
        base["flags"] = sorted(set(base["flags"] + [overall]))

        out_dir = Path(args.out) if args.out else config_path(config, "exports_dir") / "optimizer"
        target = out_dir / f"gw{int(args.gw):02d}"
        target.mkdir(parents=True, exist_ok=True)
        artifact = target / "optimizer_result.json"
        artifact.write_text(json.dumps(cu.jsonable(base), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")

        section = _optimizer_section(base, artifact)
        built = packet_mod.build_decision_packet(conn, config, event=int(args.gw), optimizer=section)
        (target / "optimizer_packet.json").write_text(packet_mod.packet_to_json(built["packet"]), encoding="utf-8")
        (target / "optimizer_packet.md").write_text(packet_mod.render_packet_markdown(built["packet"]), encoding="utf-8")

        print(f"phase 8b complete: GW{args.gw} events={list(EVENTS)} cutoff={PLANNING_CUTOFF}")
        print(f"  universe={base['full_universe_count']} base_view={base['base_search_view_count']} "
              f"pool={base['search_pool_count']} rescued={base['rescue']['rescued_count']} "
              f"(outside view {base['rescue']['rescued_outside_search_view']})")
        print(f"  promoted={base['promoted_route_count']} exact_evals={base['exact_evaluations']} "
              f"cache={base['exact_cache_entries']} union={len(union)}")
        print(f"  H1 frontier={len(base['h1_frontier'])} 3GW frontier={len(base['supported_3gw_frontier'])}")
        rb = base.get("roll_baseline") or {}
        h1 = rb.get("h1_net_core")
        w3 = rb.get("supported_3gw_net_core")
        print(f"  ROLL baseline: H1={h1 if h1 is None else round(h1, 3)} "
              f"3GW={w3 if w3 is None else round(w3, 3)} FT={rb.get('terminal_ft')} "
              f"bank={rb.get('terminal_bank_tenths')}")
        print("  top families (descriptive):")
        for family in base["families"][:5]:
            print(f"    {family['representative_route_id']} 3GW={family['supported_3gw_net_core']:.3f} "
                  f"H1={family['h1_net_core']:.3f} hits={family['cumulative_hits']} "
                  f"FT={family['terminal_ft']} bank={family['terminal_bank_tenths']}")
        print(f"  stability budget={stability_budget['status']} pool={stability_pool['status']} overall={overall}")
        print(f"  timing={base['timing_s']} artifact={artifact}")
        return 0
    finally:
        conn.close()


def _optimizer_section(result, artifact) -> dict:
    families = result.get("families") or []
    return {
        "phase_version": result.get("phase_version"),
        "optimizer_artifact": str(artifact),
        "planning_cutoff": result.get("planning_cutoff"),
        "supported_events": result.get("supported_events"),
        "search_scope": "BOUNDED_SEARCH_GW4_GW6",
        "full_universe_count": result.get("full_universe_count"),
        "search_pool_count": result.get("search_pool_count"),
        "promoted_route_count": result.get("promoted_route_count"),
        "exact_evaluations": result.get("exact_evaluations"),
        "h1_frontier": result.get("h1_frontier"),
        "supported_3gw_frontier": result.get("supported_3gw_frontier"),
        "top_families": [
            {"representative_route_id": f["representative_route_id"],
             "supported_3gw_net_core": f["supported_3gw_net_core"], "h1_net_core": f["h1_net_core"],
             "cumulative_hits": f["cumulative_hits"], "terminal_ft": f["terminal_ft"],
             "terminal_bank_tenths": f["terminal_bank_tenths"]}
            for f in families[:5]
        ],
        "roll_baseline": result.get("roll_baseline"),
        "stability": result.get("stability"),
        "risk_flags": result.get("flags"),
        "no_recommendation": True,
    }


def _context_hash(context) -> str:
    from fpl_brain import analytics
    return analytics.canonical_hash({"entry_id": context.entry_id, "event": context.planning_event,
                                     "squad": context.squad, "official_runs": context.official_runs})


if __name__ == "__main__":
    raise SystemExit(main())
