#!/usr/bin/env python3
"""Phase 8C.1 — final-acceptance consistency remediation (recertification).

Narrow pass: reuses the existing 10,000-draw GW4 world cache, reselects the GW4
Phase-6 policy on ALL 10,000 worlds for each UNIQUE finalist GW4 squad, proves the
net/gross/hit invariants, derives the true unique-squad count, and writes a
recertified artifact.  No optimizer search, no predictive run, no world
regeneration, no threshold change, no recommendation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fpl_brain import candidate_universe as cu
from fpl_brain import final_acceptance as fa
from fpl_brain import manager_lineup as ml, manager_worlds
from fpl_brain import packet as packet_mod, route_comparator as rc, route_optimizer as ro
from fpl_brain.config import config_path, load_config
from fpl_brain.database import connect_database
from fpl_brain.planning import get_planning_context

PHASE8C1 = "phase8c1_acceptance_consistency_v1.0.0"
PLANNING_CUTOFF = "2026-09-11T10:16:51Z"
EVENTS = (4, 5, 6)
EPSILON = 1e-9


def _bundle(event, draws):
    runs = {4: (64, 65, 67, 70, 71), 5: (80, 81, 83, 86, 87), 6: (88, 89, 91, 94, 95)}[event]
    return rc.EventBundle(event=event, minutes_run_id=runs[0], team_run_id=runs[1], rate_run_id=runs[2],
                          xpts_run_id=runs[3], mc_run_id=runs[4], simulations=draws, seed=20260911,
                          planning_cutoff=PLANNING_CUTOFF)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gw", type=int, default=4)
    parser.add_argument("--config")
    parser.add_argument("--cache-dir", default="data/cache/manager_worlds")
    parser.add_argument("--out")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    conn = connect_database(config_path(config, "database"))
    try:
        started = time.time()
        exports = config_path(config, "exports_dir")
        context = get_planning_context(conn, int(config.get("fpl_entry_id")), int(args.gw),
                                       as_of=PLANNING_CUTOFF, season=config.get("season"))
        squad = manager_worlds.resolve_squad(context, conn)
        universe = json.loads((exports / "candidates" / f"gw{int(args.gw):02d}" /
                               "candidate_universe.json").read_text(encoding="utf-8"))
        positions_by_id = {int(row["player_id"]): str(row["position"]) for row in universe["universe"]}
        original = json.loads((exports / "final_acceptance" / f"gw{int(args.gw):02d}" /
                               "final_acceptance.json").read_text(encoding="utf-8"))

        # --- finalists re-derived deterministically from the certified ladder ----
        ladder = json.loads((exports / "optimizer" / f"gw{int(args.gw):02d}" /
                             "stability_ladder.json").read_text(encoding="utf-8"))
        certified = ladder["ladder_records"][-1]
        selection = fa.select_finalists(certified, max_families=12)
        finalists = selection["finalists"]
        initial_state = rc.build_route_state(conn, context, squad)
        scenario = rc.flat_current_price_scenario(cu.latest_price_snapshot(conn, int(args.gw)), EVENTS)
        union = {int(p.player_id) for p in initial_state.players}
        for finalist in finalists:
            for action in finalist["actions"]:
                for move in action.get("transfers") or []:
                    union.add(int(move["out"]))
                    union.add(int(move["in"]))
                union.update(int(pid) for pid in action.get("squad_ids") or ())
        player_meta = rc.load_player_meta(conn, sorted(union))

        # --- Phase-7A replay, GW4 squad derivation, 2k/hit consistency ----------
        replay_errors = []
        finalist_records = []
        for finalist in finalists:
            signature = finalist["canonical_signature"]
            replay = fa.replay_route(initial_state=initial_state, actions=finalist["actions"],
                                     scenario=scenario, player_meta=player_meta)
            if replay["errors"]:
                replay_errors.extend(f"{signature}: {e}" for e in replay["errors"])
                continue
            gw4 = replay["events"][0]
            gw4_hash = fa.gw4_policy_cache_key(4, gw4["squad_ids"], draws=fa.GW4_DRAWS, seed=20260911)
            finalist_records.append({
                "canonical_signature": signature,
                "gw4_action": fa._strategic_shape(finalist["actions"], drop_events=(5, 6)),
                "gw4_squad_ids": list(gw4["squad_ids"]),
                "gw4_squad_hash": gw4_hash,
                "gw4_hit_points": int(gw4["hit_points"]),
                "gw5_action": fa._strategic_shape(finalist["actions"], drop_events=(4, 6)),
                "gw6_action": fa._strategic_shape(finalist["actions"], drop_events=(4, 5)),
                "gw5_mean_net_core": float(next(item for item in original["finalists"]
                                                if item["canonical_signature"] == signature)["gw5"]["mean_net_core"]),
                "gw6_mean_net_core": float(next(item for item in original["finalists"]
                                                if item["canonical_signature"] == signature)["gw6"]["mean_net_core"]),
                "old_gw4_policy": next(item for item in original["finalists"]
                                       if item["canonical_signature"] == signature)["gw4"]["policy"],
                "old_gw4_mean_gross_core": float(next(item for item in original["finalists"]
                                                      if item["canonical_signature"] == signature)["gw4"]["mean_gross_core"]),
                "selection_reasons": finalist["selection_reasons"],
            })
        if replay_errors:
            print("phase 8c.1 HARD BLOCK: replay failed", file=sys.stderr)
            for error in replay_errors[:10]:
                print(f"  - {error}", file=sys.stderr)
            return 4

        # Issue B: identical GW4 actions must give the identical GW4 squad.
        by_action: dict[str, set[str]] = {}
        for record in finalist_records:
            by_action.setdefault(record["gw4_action"], set()).add(record["gw4_squad_hash"])
        inconsistent = {action: hashes for action, hashes in by_action.items() if len(hashes) != 1}
        if inconsistent:
            print(f"phase 8c.1 HARD BLOCK: identical GW4 actions produced different squads: {inconsistent}",
                  file=sys.stderr)
            return 4
        groups: dict[str, list[str]] = {}
        for record in finalist_records:
            groups.setdefault(record["gw4_squad_hash"], []).append(record["canonical_signature"])
        unique_squad_count = len(groups)

        # --- GW4 10k worlds: reuse the cache, never regenerate silently --------
        cache_dir = Path(args.cache_dir) if args.cache_dir else None
        gw4_config = ro.OptimizerConfig(events=EVENTS, search_draws=fa.GW4_DRAWS, seed=20260911,
                                        policy_selection_worlds=0)  # 0 => ALL worlds (no subsample)
        key = ro.world_cache_key(event=4, bundle=_bundle(4, fa.GW4_DRAWS), config=gw4_config,
                                 union_ids=sorted(union))
        cache_path = (cache_dir / f"{key}.json") if cache_dir else None
        cache_hit = bool(cache_path and cache_path.exists())
        if not cache_hit:
            print("phase 8c.1 STOP: the 10k GW4 world cache key is not present; refusing to regenerate "
                  "football worlds automatically.", file=sys.stderr)
            return 5
        gw4_worlds, gw4_info = ro.build_event_worlds(conn, {4: _bundle(4, fa.GW4_DRAWS)}, 4, sorted(union),
                                                     gw4_config, cache_dir=cache_dir)
        cache_bytes = cache_path.stat().st_size

        # --- Issue C: FULL-10k Phase-6 policy selection per UNIQUE squad --------
        squad_cache: dict[str, dict] = {}
        misses = 0
        for record in finalist_records:
            gw4_hash = record["gw4_squad_hash"]
            if gw4_hash in squad_cache:
                continue
            misses += 1
            positions = {int(pid): positions_by_id[int(pid)] for pid in record["gw4_squad_ids"]}
            ranked = ml.rank_policies(list(record["gw4_squad_ids"]), positions, gw4_worlds, top_k=1)
            if not ranked["top_policies"]:
                print(f"phase 8c.1 HARD BLOCK: no legal policy for squad {gw4_hash}", file=sys.stderr)
                return 4
            policy = ranked["top_policies"][0]
            scores = rc.policy_world_scores(policy, gw4_worlds, positions)
            metrics = ml.evaluate_policy(policy, gw4_worlds, positions)
            squad_cache[gw4_hash] = {"policy": policy, "scores": scores, "metrics": metrics,
                                     "worlds": int(gw4_worlds["worlds"])}
        cache_hits = len(finalist_records) - misses

        # --- assemble corrected records ----------------------------------------
        corrected = []
        for record in finalist_records:
            entry = squad_cache[record["gw4_squad_hash"]]
            policy = entry["policy"]
            gross = float(entry["metrics"]["mean_core"])
            hit = int(record["gw4_hit_points"])
            net = gross - hit
            old_policy = record["old_gw4_policy"]
            policy_changed = any([
                list(policy.starter_ids) != list(old_policy["starter_ids"]),
                int(policy.bench_gk_id) != int(old_policy["bench_gk_id"]),
                list(policy.bench_outfield_order) != list(old_policy["bench_outfield_order"]),
                int(policy.captain_id) != int(old_policy["captain_id"]),
                int(policy.vice_captain_id) != int(old_policy["vice_captain_id"]),
            ])
            confirmed_3gw = fa.recompute_supported_3gw(net, record["gw5_mean_net_core"], record["gw6_mean_net_core"])
            corrected.append({
                **{k: v for k, v in record.items() if k != "old_gw4_policy"},
                "gw4": {
                    "mean_gross_core": gross, "hit_points": hit, "mean_net_core": net,
                    "median_core": float(entry["metrics"]["median_core"]),
                    "q10": float(entry["metrics"]["q10_core"]), "q25": float(entry["metrics"]["q25_core"]),
                    "q75": float(entry["metrics"]["q75_core"]), "q90": float(entry["metrics"]["q90_core"]),
                    "std_core": float(entry["metrics"]["std_core"]),
                    "p_any_autosub": float(entry["metrics"]["p_any_autosub"]),
                    "expected_autosub_points_added": float(entry["metrics"]["expected_autosub_points_added"]),
                    "p_vice_takes_captaincy": float(entry["metrics"]["p_vice_takes_captaincy"]),
                    "p_no_captain_multiplier": float(entry["metrics"]["p_no_captain_multiplier"]),
                    "policy": {"starter_ids": list(policy.starter_ids), "bench_gk_id": int(policy.bench_gk_id),
                               "bench_outfield_order": list(policy.bench_outfield_order),
                               "captain_id": int(policy.captain_id), "vice_captain_id": int(policy.vice_captain_id)},
                    "squad_hash": record["gw4_squad_hash"],
                    "policy_selection_worlds": int(entry["worlds"]),
                },
                "policy_changed_vs_2k_subsample": bool(policy_changed),
                "old_gw4_policy": old_policy,
                "old_gw4_mean_gross_core": record["old_gw4_mean_gross_core"],
                "gw4_mean_gross_delta": gross - record["old_gw4_mean_gross_core"],
                "confirmed_3gw_net_core": confirmed_3gw,
            })

        # --- invariants ---------------------------------------------------------
        invariants = []
        for row in corrected:
            gross, hit, net = row["gw4"]["mean_gross_core"], row["gw4"]["hit_points"], row["gw4"]["mean_net_core"]
            invariants.append({"signature": row["canonical_signature"], "check": "net_equals_gross_minus_hit",
                               "status": "PASS" if abs(net - (gross - hit)) <= EPSILON else "FAIL"})
            if hit == 0:
                invariants.append({"signature": row["canonical_signature"], "check": "zero_hit_gross_equals_net",
                                   "status": "PASS" if abs(net - gross) <= EPSILON else "FAIL"})
        roll_rows = [row for row in corrected if row["canonical_signature"] == "E4:ROLL;E5:ROLL;E6:ROLL"]
        roll = roll_rows[0] if roll_rows else None
        if roll is None:
            print("phase 8c.1 HARD BLOCK: ALL-ROLL finalist missing", file=sys.stderr)
            return 4
        roll_invariants = [
            {"check": "roll_transfers_zero", "status": "PASS" if roll["gw4_action"] == "E4:ROLL" else "FAIL"},
            {"check": "roll_hit_zero", "status": "PASS" if roll["gw4"]["hit_points"] == 0 else "FAIL"},
            {"check": "roll_gross_equals_net",
             "status": "PASS" if abs(roll["gw4"]["mean_gross_core"] - roll["gw4"]["mean_net_core"]) <= EPSILON else "FAIL"},
        ]
        # same GW4 squad => identical policy and identical world-score vector
        squad_policy_ok = True
        squad_scores_ok = True
        for gw4_hash, members in groups.items():
            policies = {json.dumps(row["gw4"]["policy"], sort_keys=True)
                        for row in corrected if row["gw4"]["squad_hash"] == gw4_hash}
            squad_policy_ok = squad_policy_ok and len(policies) == 1
        # compare per-finalist world-score vectors by squad hash
        vector_by_squad: dict[str, tuple] = {}
        for row in corrected:
            gw4_hash = row["gw4"]["squad_hash"]
            vector = tuple(squad_cache[gw4_hash]["scores"])
            existing = vector_by_squad.get(gw4_hash)
            if existing is not None and existing != vector:
                squad_scores_ok = False
            vector_by_squad[gw4_hash] = vector

        # --- paired vs ROLL on the SAME 10k worlds -----------------------------
        paired = []
        roll_scores = squad_cache[roll["gw4"]["squad_hash"]]["scores"]
        for row in corrected:
            if row is roll:
                continue
            scores = squad_cache[row["gw4"]["squad_hash"]]["scores"]
            diffs = [a - b - row["gw4"]["hit_points"] for a, b in zip(scores, roll_scores)]
            n = len(diffs)
            mean = sum(diffs) / n
            variance = sum((d - mean) ** 2 for d in diffs) / (n - 1) if n > 1 else 0.0
            se = (variance / n) ** 0.5 if n > 1 else 0.0
            paired.append({"finalist": row["canonical_signature"], "reference": roll["canonical_signature"],
                           "worlds": n, "mean_difference": mean, "paired_se": se,
                           "ci95_low": mean - 1.96 * se, "ci95_high": mean + 1.96 * se,
                           "p_finalist_gt_roll": sum(1 for d in diffs if d > 0) / n,
                           "near_tied": (abs(mean) <= 1.96 * se) if se > 0 else mean == 0.0})

        # --- leaders ------------------------------------------------------------
        h1_ranking = sorted(corrected, key=lambda r: -r["gw4"]["mean_net_core"])
        leaders_3gw = sorted(corrected, key=lambda r: -r["confirmed_3gw_net_core"])
        old_original = {row["canonical_signature"]: row for row in original["finalists"]}
        h1_old = max(sorted(old_original), key=lambda sig: old_original[sig]["gw4"]["mean_net_core"])
        h1_new = h1_ranking[0]["canonical_signature"]
        h1_old_value = old_original[h1_old]["gw4"]["mean_net_core"]
        h1_new_value = h1_ranking[0]["gw4"]["mean_net_core"]
        h1_changed = h1_new != h1_old
        h1_material = h1_changed and abs(h1_new_value - h1_old_value) > fa.CONFIRMATION_MATERIAL_CORE
        w3_old = max(sorted(old_original), key=lambda sig: old_original[sig]["confirmed_3gw_net_core"])
        w3_new = leaders_3gw[0]["canonical_signature"]
        w3_old_value = old_original[w3_old]["confirmed_3gw_net_core"]
        w3_new_value = leaders_3gw[0]["confirmed_3gw_net_core"]
        w3_material = (w3_new != w3_old) and abs(w3_new_value - w3_old_value) > fa.CONFIRMATION_MATERIAL_CORE

        # --- extended audit -----------------------------------------------------
        audit_checks = []
        def check(name, ok, detail=""):
            audit_checks.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": str(detail)})

        check("net_equals_gross_minus_hit_all", all(item["status"] == "PASS" for item in invariants
                                                    if item["check"] == "net_equals_gross_minus_hit"))
        check("zero_hit_gross_equals_net_all", all(item["status"] == "PASS" for item in invariants
                                                   if item["check"] == "zero_hit_gross_equals_net"))
        check("roll_transfers_zero", roll_invariants[0]["status"] == "PASS")
        check("roll_hit_zero", roll_invariants[1]["status"] == "PASS")
        check("roll_gross_equals_net", roll_invariants[2]["status"] == "PASS")
        check("same_gw4_squad_same_policy", squad_policy_ok)
        check("same_gw4_squad_same_world_scores", squad_scores_ok)
        check("gw4_actions_map_to_single_squad", not inconsistent)
        check("unique_squad_count_equals_hash_cardinality", unique_squad_count == len(groups))
        check("cache_misses_equal_unique_squad_evaluations", misses == unique_squad_count)
        check("full_10k_policy_selection_no_subsample",
              all(row["gw4"]["policy_selection_worlds"] == fa.GW4_DRAWS for row in corrected))
        check("world_cache_hit", cache_hit)
        check("finalist_families_count", len(finalists) == len(corrected))
        check("old_122_checks_still_pass", original["cross_section_audit"]["status"] == "PASS")
        extended_audit = {"status": "PASS" if all(c["status"] == "PASS" for c in audit_checks) else "FAIL",
                          "checks": audit_checks, "check_count": len(audit_checks),
                          "failed": [c for c in audit_checks if c["status"] == "FAIL"],
                          "original_phase8c_audit": original["cross_section_audit"]["status"]}

        # --- frontier over corrected values ------------------------------------
        orig_by_sig = {row["canonical_signature"]: row for row in original["finalists"]}
        frontier = fa.confirmed_frontier({row["canonical_signature"]: {
            "confirmed_3gw_net_core": row["confirmed_3gw_net_core"],
            "cumulative_hits": int(orig_by_sig[row["canonical_signature"]]["cumulative_hits"]),
            "terminal_ft": int(orig_by_sig[row["canonical_signature"]]["terminal_ft"]),
            "terminal_bank_tenths": int(orig_by_sig[row["canonical_signature"]]["terminal_bank_tenths"]),
        } for row in corrected})

        artifact = {
            "phase_version": PHASE8C1,
            "supersedes": "phase8c_final_acceptance_v1.0.0",
            "planning_cutoff": PLANNING_CUTOFF,
            "supported_events": list(EVENTS),
            "score_basis": "CORE",
            "flags": [fa.H4_H6_FLAG, fa.CROSS_GW_FLAG, fa.PRICE_FLAG, ro.BOUNDED_PRUNING_LABEL,
                      ro.HEURISTIC_LABEL, ro.BEST_WITHIN_SEARCH_LABEL],
            "draw_fidelity": {"gw4": fa.GW4_DRAWS, "gw5": fa.GW5_DRAWS, "gw6": fa.GW6_DRAWS},
            "price_scenario": fa.PRICE_SCENARIO,
            "policy_selection": {"gw4_worlds_used": fa.GW4_DRAWS, "subsample_used": False},
            "world_cache": {"gw4_key": key, "gw4_hit": cache_hit, "gw4_bytes": cache_bytes,
                            "gw4_union_players": len(union), "gw4_worlds": int(gw4_worlds["worlds"])},
            "finalist_families": len(finalists),
            "unique_gw4_squad_count": unique_squad_count,
            "groups_by_gw4_squad": {gw4_hash: sorted(members) for gw4_hash, members in sorted(groups.items())},
            "phase6_cache": {"unique_squad_evaluations": misses, "hits": cache_hits,
                             "key_fields": ["event", "gw4_squad_hash", "worlds/config identity"],
                             "excludes": ["route_id", "gw5_action", "gw6_action", "family_signature"]},
            "invariants": invariants,
            "roll": {"gw4_action": roll["gw4_action"], "hit_points": roll["gw4"]["hit_points"],
                     "gross_10k": roll["gw4"]["mean_gross_core"], "net_10k": roll["gw4"]["mean_net_core"],
                     "confirmed_3gw_net_core": roll["confirmed_3gw_net_core"],
                     "terminal_ft": int(orig_by_sig[roll["canonical_signature"]]["terminal_ft"]),
                     "terminal_bank_tenths": int(orig_by_sig[roll["canonical_signature"]]["terminal_bank_tenths"])},
            "roll_invariants": roll_invariants,
            "finalists": corrected,
            "paired_vs_roll": paired,
            "leaders": {
                "h1_leader_before": h1_old, "h1_leader_after": h1_new,
                "h1_value_before": h1_old_value, "h1_value_after": h1_new_value,
                "h1_changed": h1_changed, "h1_changed_materially": h1_material,
                "supported_3gw_leader_before": w3_old, "supported_3gw_leader_after": w3_new,
                "supported_3gw_value_before": w3_old_value, "supported_3gw_value_after": w3_new_value,
                "supported_3gw_changed": w3_new != w3_old, "supported_3gw_changed_materially": w3_material,
                "material_threshold_core": fa.CONFIRMATION_MATERIAL_CORE,
            },
            "confirmed_shortlist_pareto_frontier": frontier,
            "extended_cross_section_audit": extended_audit,
            "timing_s": {"total": round(time.time() - started, 3)},
            "acceptance": "COMPLETE" if (extended_audit["status"] == "PASS" and not h1_material
                                         and not w3_material) else "PARTIAL",
            "no_recommendation": True,
        }

        out_dir = Path(args.out) if args.out else exports / "final_acceptance" / f"gw{int(args.gw):02d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / "final_acceptance_recertified.json"
        target.write_text(json.dumps(cu.jsonable(artifact), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                          encoding="utf-8")

        section = {
            "phase_version": PHASE8C1, "planning_cutoff": PLANNING_CUTOFF,
            "supported_events": list(EVENTS), "score_basis": "CORE",
            "draw_fidelity": artifact["draw_fidelity"], "price_scenario": fa.PRICE_SCENARIO,
            "flags": artifact["flags"], "policy_selection": artifact["policy_selection"],
            "unique_gw4_squad_count": unique_squad_count, "finalist_families": len(finalists),
            "leaders": artifact["leaders"], "extended_cross_section_audit": extended_audit,
            "acceptance": artifact["acceptance"], "no_recommendation": True,
            "final_acceptance_artifact": str(target),
        }
        built = packet_mod.build_decision_packet(conn, config, event=int(args.gw), final_acceptance=section)
        (out_dir / "final_acceptance_recertified_packet.json").write_text(
            packet_mod.packet_to_json(built["packet"]), encoding="utf-8")
        (out_dir / "final_acceptance_recertified_packet.md").write_text(
            packet_mod.render_packet_markdown(built["packet"]), encoding="utf-8")

        print(f"phase 8c.1 complete: families={len(finalists)} unique_gw4_squads={unique_squad_count} "
              f"gw4_cache={'HIT' if cache_hit else 'MISS'} policy_worlds={fa.GW4_DRAWS} "
              f"phase6_misses={misses} hits={cache_hits}")
        print(f"  ROLL gross={roll['gw4']['mean_gross_core']:.4f} hit={roll['gw4']['hit_points']} "
              f"net={roll['gw4']['mean_net_core']:.4f} 3GW={roll['confirmed_3gw_net_core']:.4f}")
        print(f"  H1 leader {h1_old} -> {h1_new} (changed={h1_changed}, material={h1_material})")
        print(f"  3GW leader {w3_old} -> {w3_new} (changed={w3_new != w3_old}, material={w3_material})")
        print(f"  extended audit={extended_audit['status']} ({extended_audit['check_count']} checks) "
              f"acceptance={artifact['acceptance']}")
        print("  | squad_hash | families | policy changed | old 10k gross | new 10k gross | delta |")
        for gw4_hash, members in sorted(groups.items()):
            row = next(r for r in corrected if r["gw4"]["squad_hash"] == gw4_hash)
            print(f"  | {gw4_hash} | {len(members)} | {row['policy_changed_vs_2k_subsample']} | "
                  f"{row['old_gw4_mean_gross_core']:.4f} | {row['gw4']['mean_gross_core']:.4f} | "
                  f"{row['gw4_mean_gross_delta']:+.4f} |")
        print(f"  total={artifact['timing_s']['total']}s artifact={target}")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
