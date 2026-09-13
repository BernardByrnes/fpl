#!/usr/bin/env python3
"""Phase 6A — build the fixed-15 manager policy evaluation and decision packet.

Descriptive validation only: no lineup change, transfer or chip is executed.
Consumes the canonical PlanningContext and the FROZEN Phase-5 predictive runs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fpl_brain import manager_lineup, manager_worlds, packet as packet_mod
from fpl_brain.config import config_path, load_config
from fpl_brain.database import connect_database
from fpl_brain.planning import get_planning_context

PHASE6A_VERSION = "phase6a_manager_lineup_v1.0.0"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Phase 6A fixed-15 manager lineup evaluation")
    parser.add_argument("--gw", type=int, required=True)
    parser.add_argument("--config")
    parser.add_argument("--minutes-run", type=int, default=64)
    parser.add_argument("--xpts-run", type=int, default=70)
    parser.add_argument("--team-run", type=int, default=65)
    parser.add_argument("--monte-carlo-run", type=int, default=71)
    parser.add_argument("--simulations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--out-dir")
    parser.add_argument("--cutoff", help="planning as-of cutoff for coherent pre-deadline provenance")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    conn = connect_database(config_path(config, "database"))
    try:
        started = time.time()
        entry_id = config.get("fpl_entry_id")
        context = get_planning_context(
            conn, int(entry_id), int(args.gw), as_of=args.cutoff, season=config.get("season"),
            official_price_stale_after_hours=config.get("report", {}).get("official_price_stale_after_hours"),
        )
        squad = manager_worlds.resolve_squad(context, conn)
        if squad["player_count"] != 15:
            print(f"phase 6a failed: PlanningContext squad has {squad['player_count']} players, not 15", file=sys.stderr)
            return 3
        if not all(position in ("GKP", "DEF", "MID", "FWD") for position in squad["positions"].values()):
            print("phase 6a failed: unresolved player position in squad", file=sys.stderr)
            return 3

        timed = {}
        mark = time.time()
        built = manager_worlds.build_manager_worlds(
            conn, planning_event=int(args.gw),
            minutes_run_id=int(args.minutes_run), xpts_run_id=int(args.xpts_run),
            team_run_id=int(args.team_run), squad_ids=squad["squad_ids"],
            simulations=int(args.simulations), seed=int(args.seed), occupancy_audit=True,
        )
        world_matrix = built["world_matrix"]
        timed["manager_world_generation_s"] = round(time.time() - mark, 2)

        mark = time.time()
        skeletons = list(manager_lineup.enumerate_skeletons(squad["squad_ids"], squad["positions"]))
        timed["enumeration_s"] = round(time.time() - mark, 3)

        mark = time.time()
        ranked = manager_lineup.rank_policies(
            squad["squad_ids"], squad["positions"], world_matrix, top_k=int(args.top_k)
        )
        timed["ranking_s"] = round(time.time() - mark, 2)
        top_policies = ranked["top_policies"]

        mark = time.time()
        top_rows = []
        for policy in top_policies:
            metrics = manager_lineup.evaluate_policy(policy, world_matrix, squad["positions"])
            row = {
                **policy.as_dict(squad["names"]),
                **{key: value for key, value in metrics.items() if key != "final_counted_player_probability"},
                "mean_core_rank_only": ranked["top_mean_by_key"].get(policy.ordering_key()),
            }
            top_rows.append(row)
        top_rows.sort(key=lambda row: -row["mean_core"])
        timed["top_k_distribution_s"] = round(time.time() - mark, 2)
        timed["total_s"] = round(time.time() - started, 1)

        manager_section = {
            "phase_version": PHASE6A_VERSION,
            "lineup_model_version": manager_lineup.MANAGER_LINEUP_VERSION,
            "worlds_model_version": manager_worlds.MANAGER_WORLDS_VERSION,
            "planning_context_hash": _context_hash(context),
            "predictive_runs": {
                "minutes": int(args.minutes_run), "minutes_version": "minutes_v1.5.2",
                "xpts": int(args.xpts_run), "xpts_version": "xpts_v1.4.1",
                "monte_carlo": int(args.monte_carlo_run),
                "monte_carlo_version": built["simulation"]["mc_model_version"],
            },
            "simulation": built["simulation"],
            "input_run_ids": built["input_run_ids"],
            "squad_ids": squad["squad_ids"],
            "squad_state": squad["squad_state"],
            "squad_source": squad["squad_source"],
            "skeleton_count": ranked["skeletons"],
            "evaluated_policy_count": ranked["evaluated_policies"],
            "timing": timed,
            "scoring_basis": "CORE",
            "bonus_handling_status": "BONUS_MEAN_MANAGER_INTEGRATION_DEFERRED",
            "risk_flags": [
                "CORE_BASED_NO_BONUS_VARIANCE",
                "MANAGER_LAYER_FIXED_15_ONLY",
                "DESCRIPTIVE_ONLY_NO_EXECUTION",
            ],
            "top_policies": top_rows[:20],
        }
        built_packet = packet_mod.build_decision_packet(
            conn, config, event=int(args.gw), as_of=args.cutoff, manager=manager_section
        )
        packet = built_packet["packet"]

        out_dir = Path(args.out_dir) if args.out_dir else config_path(config, "exports_dir") / "manager"
        target = out_dir / f"gw{int(args.gw):02d}"
        target.mkdir(parents=True, exist_ok=True)
        (target / "manager_lineup_packet.json").write_text(
            packet_mod.packet_to_json(packet), encoding="utf-8"
        )
        (target / "manager_lineup_packet.md").write_text(
            packet_mod.render_packet_markdown(packet), encoding="utf-8"
        )
        elapsed = time.time() - started
        print(
            f"phase 6a complete: GW{args.gw} squad={len(squad['squad_ids'])} worlds={built['simulation']['worlds']} "
            f"skeletons={ranked['skeletons']} policies={ranked['evaluated_policies']} "
            f"occ_viol={built['simulation']['occupancy_violations']} elapsed={elapsed:.1f}s artifact={target}"
        )
        for index, row in enumerate(top_rows[:5], start=1):
            print(
                f"  #{index} mean={row['mean_core']:.3f} cap={row['captain_name']} vice={row['vice_captain_name']} "
                f"XI={row['starter_names']} bench={row['bench_names']}"
            )
        return 0
    finally:
        conn.close()


def _context_hash(context) -> str:
    from fpl_brain import analytics
    return analytics.canonical_hash({
        "entry_id": context.entry_id, "event": context.planning_event,
        "squad": context.squad, "official_runs": context.official_runs,
    })


if __name__ == "__main__":
    raise SystemExit(main())
