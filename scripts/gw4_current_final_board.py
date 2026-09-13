#!/usr/bin/env python3
"""FINAL pre-deadline GW4 current-GW (H1) manager board.

Uses the certified fresh GW4 bundle at cutoff 2026-09-12T10:40:04Z
(minutes 127 / team 128 / rates 130 / xPts 133 / MC 134) and the accepted
Phase-6 exact manager-policy enumeration over the GW4 worlds.

CURRENT_GW_H1_ONLY — the four-GW transfer horizon is INCOMPLETE, so there is no
normal transfer recommendation, no chip recommendation and no route search.
Nothing is executed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fpl_brain import four_gw_decision as fg, manager_lineup as ml, manager_worlds, route_comparator as rc
from fpl_brain import route_optimizer as ro, transfer_state as ts
from fpl_brain.config import config_path, load_config
from fpl_brain.database import connect_database
from fpl_brain.planning import get_planning_context
from fpl_brain.utils import utc_now

CUTOFF = "2026-09-12T10:40:04Z"
SEED = 20260911
DRAWS = 10_000
RUNS = {"minutes_v1": 127, "team_strength_v1": 128, "player_rates_v1": 130,
        "xpts_v1": 133, "monte_carlo_v1": 134}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Final GW4 current-GW H1 board")
    parser.add_argument("--config")
    parser.add_argument("--cache-dir", default="data/cache/manager_worlds")
    parser.add_argument("--out", default="data/exports/four_gw/gw04/current_gw_h1_decision.json")
    args = parser.parse_args(argv)

    started = time.time()
    config = load_config(args.config)
    conn = connect_database(config_path(config, "database"))
    try:
        entry_id = int(config["fpl_entry_id"])
        context = get_planning_context(conn, entry_id, 4, as_of=CUTOFF, season=config.get("season"))
        squad = manager_worlds.resolve_squad(context, conn)
        squad_ids = [int(pid) for pid in squad["squad_ids"]]
        positions = {int(pid): str(pos) for pid, pos in squad["positions"].items()}
        state = rc.build_route_state(conn, context, squad)
        meta = rc.load_player_meta(conn, squad_ids)
        names = {}
        prices = {}
        for pid in squad_ids:
            row = conn.execute("SELECT web_name, full_name FROM players WHERE id=?", (pid,)).fetchone()
            names[pid] = row["web_name"] or row["full_name"] or str(pid)
            prices[pid] = int(meta[pid].club_id)
        selling = {int(r["player_id"]): r for r in context.selling_prices}
        clubs = {int(r["id"]): r["short_name"] for r in conn.execute("SELECT id, short_name FROM teams")}

        # --- GW4 worlds: reuse the accepted cache when present ----------------
        bundle = rc.EventBundle(event=4, minutes_run_id=RUNS["minutes_v1"], team_run_id=RUNS["team_strength_v1"],
                                rate_run_id=RUNS["player_rates_v1"], xpts_run_id=RUNS["xpts_v1"],
                                mc_run_id=RUNS["monte_carlo_v1"], simulations=DRAWS, seed=SEED,
                                planning_cutoff=CUTOFF)
        wconfig = ro.OptimizerConfig(events=(4,), search_draws=DRAWS, seed=SEED, policy_selection_worlds=0)
        key = ro.world_cache_key(event=4, bundle=bundle, config=wconfig, union_ids=squad_ids)
        cache_path = Path(args.cache_dir) / f"{key}.json"
        t0 = time.time()
        worlds, info = ro.build_event_worlds(conn, {4: bundle}, 4, squad_ids, wconfig, cache_dir=Path(args.cache_dir))
        world_seconds = time.time() - t0

        # --- exact Phase-6 policy selection over ALL draws --------------------
        t1 = time.time()
        ranked = ml.rank_policies(squad_ids, positions, worlds, top_k=5)
        policy = ranked["top_policies"][0]
        scores = rc.policy_world_scores(policy, worlds, positions)
        scores_sorted = sorted(scores)
        n = len(scores)
        policy_mean = sum(scores) / n
        median = scores_sorted[n // 2]
        policy_seconds = time.time() - t1

        def nm(pid):
            return names[int(pid)]

        xi_by_position = {}
        for pid in policy.starter_ids:
            xi_by_position.setdefault(positions[int(pid)], []).append(int(pid))
        order = ["GKP", "DEF", "MID", "FWD"]
        formation = "".join(str(len(xi_by_position.get(p, []))) for p in ("DEF", "MID", "FWD"))

        # --- availability warnings from the CERTIFIED GW4 minutes/xpts runs ----
        warnings = []
        per_player = {}
        for pid in squad_ids:
            mrow = conn.execute(
                "SELECT payload_json FROM frozen_predictions WHERE projection_run_id=? AND kind=? AND player_id=?",
                (RUNS["minutes_v1"], __import__("fpl_brain.analytics", fromlist=["x"]).MINUTES_V1_KIND, pid),
            ).fetchall()
            rows = [json.loads(r["payload_json"]) for r in mrow]
            avail = min((float(r.get("joint_availability") or 0.0) for r in rows), default=None)
            minutes = sum(float(r.get("expected_minutes") or 0.0) for r in rows) if rows else None
            p_start = max((float(r.get("p_start") or 0.0) for r in rows), default=None)
            per_player[pid] = {"availability": avail, "expected_minutes": minutes, "p_start": p_start,
                               "fixtures": len(rows)}
            label = f"{nm(pid)} ({positions[pid]}, {clubs.get(int(meta[pid].club_id), '?')})"
            if avail is not None and avail < 0.999:
                warnings.append(f"AVAILABILITY {avail:.2f}: {label} — reduced availability in the certified GW4 minutes run")
            if minutes is not None and minutes < 45.0:
                warnings.append(f"LOW_MINUTES {minutes:.1f}: {label} — certified GW4 expected minutes below 45")
            elif p_start is not None and p_start < 0.50:
                warnings.append(f"LOW_START {p_start:.2f}: {label} — certified GW4 P(start) below 0.50")

        next_ft = ts.next_event_free_transfers(state.free_transfers, 0)
        starter_set = {int(p) for p in policy.starter_ids}
        board = {
            "label": ["CURRENT_GW_H1_ONLY", "TRANSFER_DECISION_HORIZON_INCOMPLETE",
                      "NO_ADDITIONAL_TRANSFER", "NO_CHIP_RECOMMENDATION"],
            "planning_event": 4, "planning_cutoff": CUTOFF,
            "generated_at_utc": utc_now(),
            "horizon": {"decision_events": [4, 5, 6, 7], "status": fg.DECISION_HORIZON_INCOMPLETE,
                        "blocked_events": [5, 6, 7]},
            "transfer_recommendation": {"status": fg.RECOMMENDATION_SUPPRESSED,
                                        "reason": fg.DECISION_HORIZON_INCOMPLETE,
                                        "preferred_route_id": None, "route_search_run": False},
            "chip_recommendation": {"status": "NO_CHIP_RECOMMENDATION",
                                    "wildcard": "no supported four-GW Wildcard evaluation exists"},
            "squad": [
                {"player_id": pid, "name": nm(pid), "position": positions[pid],
                 "club": clubs.get(int(meta[pid].club_id)),
                 "purchase_tenths": int((selling.get(pid, {}) or {}).get("purchase_price") or 0),
                 "market_tenths": int((selling.get(pid, {}) or {}).get("official_market_price") or 0),
                 "selling_tenths": int((selling.get(pid, {}) or {}).get("effective_selling_price") or 0)}
                for pid in squad_ids
            ],
            "manager_state": {"free_transfers_remaining": int(state.free_transfers), "bank_tenths": int(state.bank_tenths),
                              "event_start_free_transfers": state.event_start_free_transfers,
                              "authoritative_source": (context.manager_state or {}).get("authoritative_source"),
                              "health": context.health.get("status")},
            "expected_free_transfers_entering_gw5_if_no_action": int(next_ft),
            "xi": {pos: [nm(p) for p in xi_by_position.get(pos, [])] for pos in order},
            "xi_player_ids": {pos: xi_by_position.get(pos, []) for pos in order},
            "formation": formation,
            "bench_gk": nm(policy.bench_gk_id),
            "bench_order": [nm(p) for p in policy.bench_outfield_order],
            "captain": nm(policy.captain_id),
            "vice_captain": nm(policy.vice_captain_id),
            "autosub_expected_points_added": ranked["base_stats"][
                next(i for i, (s, _b, _o) in enumerate(ml.enumerate_skeletons(squad_ids, positions))
                     if tuple(s) == tuple(policy.starter_ids) and int(_b) == int(policy.bench_gk_id)
                     and tuple(_o) == tuple(policy.bench_outfield_order))
            ].get("mean_autosub_core") if ranked.get("base_stats") else None,
            "policy_expectation": {"worlds": n, "expected_core": policy_mean, "median_core": median,
                                   "q10": scores_sorted[int(0.10 * (n - 1))], "q90": scores_sorted[int(0.90 * (n - 1))],
                                   "evaluated_policies": ranked["evaluated_policies"], "skeletons": ranked["skeletons"]},
            "world_provenance": {"source": info.get("source"), "cache_key": key, "exists_before": cache_path.exists(),
                                 "worlds": int(worlds["worlds"]), "union_players": len(squad_ids),
                                 "run_ids": RUNS, "seed": SEED, "seconds": round(world_seconds, 1),
                                 "generation": info.get("source")},
            "player_availability": {str(pid): per_player[pid] for pid in squad_ids},
            "warnings": warnings,
            "timing_seconds": {"worlds": round(world_seconds, 1), "policy": round(policy_seconds, 1),
                               "total": round(time.time() - started, 1)},
            "no_execution": True,
        }
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(board, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

        print("=" * 78)
        print("FINAL GW4 CURRENT-GW BOARD   " + " | ".join(board["label"]))
        print("=" * 78)
        print(f"cutoff {CUTOFF}   worlds source={info.get('source')} n={worlds['worlds']} seed={SEED} "
              f"worlds={world_seconds:.1f}s policy={policy_seconds:.1f}s")
        print(f"SQUAD 15: FT remaining={state.free_transfers}  bank=£{state.bank_tenths/10:.1f}m  "
              f"event-start FT={state.event_start_free_transfers}  health={context.health.get('status')}")
        print(f"expected FT entering GW5 if no action = {next_ft}")
        print()
        for pos in order:
            print(f"{pos}: " + ", ".join(nm(p) for p in xi_by_position.get(pos, [])))
        print(f"Formation {formation}")
        print(f"Bench GK: {nm(policy.bench_gk_id)}")
        print(f"Bench order: " + ", ".join(nm(p) for p in policy.bench_outfield_order))
        print(f"Captain: {nm(policy.captain_id)}   Vice: {nm(policy.vice_captain_id)}")
        print(f"XI expected CORE {policy_mean:.4f} (median {median:.1f}, q10 {scores_sorted[int(0.10*(n-1))]:.1f}, "
              f"q90 {scores_sorted[int(0.90*(n-1))]:.1f}) over {n} draws")
        print()
        print("MATERIAL AVAILABILITY WARNINGS")
        if warnings:
            for w in warnings:
                print("  - " + w)
        else:
            print("  - none")
        print(f"\nTRANSFER: {board['transfer_recommendation']['status']} (horizon {board['horizon']['status']})")
        print(f"CHIP: {board['chip_recommendation']['status']}")
        print(f"artifact: {out}")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
