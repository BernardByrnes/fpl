#!/usr/bin/env python3
"""Phase 8C — final end-to-end acceptance and 10k GW4 shortlist confirmation.

Acceptance only: no new search, no predictive change, no chip, no recommendation.
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
from fpl_brain import manager_lineup as ml, manager_worlds, monte_carlo
from fpl_brain import packet as packet_mod, route_comparator as rc, route_optimizer as ro
from fpl_brain.config import config_path, load_config
from fpl_brain.database import connect_database
from fpl_brain.planning import get_planning_context

PHASE8C = "phase8c_final_acceptance_v1.0.0"
PLANNING_CUTOFF = "2026-09-11T10:16:51Z"
EVENTS = (4, 5, 6)
RUNS = {4: {"minutes": 64, "team": 65, "rate": 67, "xpts": 70, "mc": 71},
        5: {"minutes": 80, "team": 81, "rate": 83, "xpts": 86, "mc": 87},
        6: {"minutes": 88, "team": 89, "rate": 91, "xpts": 94, "mc": 95}}
GW4_MC_CONFIG_HASH = "sha256:fedf5f6e50b4826b6e44b42d524cac2d682f15e878a5c86f7bab08a3f4e15042"


def _bundle(event, draws):
    return rc.EventBundle(event=event, minutes_run_id=RUNS[event]["minutes"], team_run_id=RUNS[event]["team"],
                          rate_run_id=RUNS[event]["rate"], xpts_run_id=RUNS[event]["xpts"],
                          mc_run_id=RUNS[event]["mc"], simulations=draws, seed=20260911,
                          planning_cutoff=PLANNING_CUTOFF)


def _select_and_score(squad_ids, worlds, positions_of, config, cache, event=4):
    key = fa.gw4_policy_cache_key(event, squad_ids, draws=int(worlds["worlds"]), seed=int(config.seed))
    entry = cache.get(key)
    if entry is None:
        positions = positions_of(squad_ids)
        selection = ro._subsample(worlds, int(config.policy_selection_worlds))
        ranked = ml.rank_policies(list(squad_ids), positions, selection, top_k=1)
        if not ranked["top_policies"]:
            raise ValueError("no legal manager policy")
        policy = ranked["top_policies"][0]
        entry = {"policy": policy, "positions": positions,
                 "scores": rc.policy_world_scores(policy, worlds, positions),
                 "metrics": ml.evaluate_policy(policy, worlds, positions)}
        cache[key] = entry
    return entry


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gw", type=int, default=4)
    parser.add_argument("--config")
    parser.add_argument("--optimizer-artifact")
    parser.add_argument("--cache-dir", default="data/cache/manager_worlds")
    parser.add_argument("--policy-worlds", type=int, default=2000)
    parser.add_argument("--gw4-draws", type=int, default=fa.GW4_DRAWS)
    parser.add_argument("--max-families", type=int, default=12)
    parser.add_argument("--out")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    conn = connect_database(config_path(config, "database"))
    try:
        started = time.time()
        context = get_planning_context(conn, int(config.get("fpl_entry_id")), int(args.gw),
                                       as_of=PLANNING_CUTOFF, season=config.get("season"))
        squad = manager_worlds.resolve_squad(context, conn)
        exports = config_path(config, "exports_dir")
        opt_path = Path(args.optimizer_artifact) if args.optimizer_artifact else (
            exports / "optimizer" / f"gw{int(args.gw):02d}" / "stability_ladder.json")
        if not opt_path.exists():
            print(f"phase 8c failed: certified optimizer artifact missing at {opt_path}", file=sys.stderr)
            return 3
        certified_artifact = json.loads(opt_path.read_text(encoding="utf-8"))
        # The certified source is the Phase-8B.1 nested ladder's largest budget record
        # (it carries canonical family signatures).  Fall back to a plain result.
        if "ladder_records" in certified_artifact:
            certified = certified_artifact["ladder_records"][-1]
            certified_budget = certified_artifact["budgets"][-1]
        else:
            certified = certified_artifact
            certified_budget = None
        universe = json.loads((exports / "candidates" / f"gw{int(args.gw):02d}" / "candidate_universe.json")
                              .read_text(encoding="utf-8"))
        universe_ids = {int(row["player_id"]) for row in universe["universe"]}
        positions_by_id = {int(row["player_id"]): str(row["position"]) for row in universe["universe"]}

        selection = fa.select_finalists(certified, max_families=int(args.max_families))
        finalists = selection["finalists"]
        base_config = ro.OptimizerConfig(events=EVENTS, search_draws=2000, beam_width=12,
                                        policy_selection_worlds=int(args.policy_worlds),
                                        search_n_per_criterion=20, rescue_top_k_per_position=25)
        initial_state = rc.build_route_state(conn, context, squad)
        scenario = rc.flat_current_price_scenario(cu.latest_price_snapshot(conn, int(args.gw)), EVENTS)

        # --- Union of every player appearing in any finalist route, then replay --
        union = {int(p.player_id) for p in initial_state.players}
        for finalist in finalists:
            for action in finalist["actions"]:
                for move in action.get("transfers") or []:
                    union.add(int(move["out"]))
                    union.add(int(move["in"]))
                union.update(int(pid) for pid in action.get("squad_ids") or ())
        player_meta = rc.load_player_meta(conn, sorted(union))
        missing_meta = [pid for pid in sorted(union) if pid not in player_meta]
        if missing_meta:
            print(f"phase 8c HARD BLOCK: unresolved player meta {missing_meta}", file=sys.stderr)
            return 4
        replay_errors = []
        gw4_squads = {}
        route_replay = {}
        for finalist in finalists:
            signature = finalist["canonical_signature"]
            replay = fa.replay_route(initial_state=initial_state, actions=finalist["actions"],
                                     scenario=scenario, player_meta=player_meta)
            route_replay[signature] = replay
            if replay["errors"]:
                replay_errors.extend(f"{signature}: {e}" for e in replay["errors"])
                continue
            per_event = {item["event"]: item for item in replay["events"]}
            gw4_squads[signature] = per_event[4]["squad_ids"]
        if replay_errors:
            print("phase 8c HARD BLOCK: route replay failed", file=sys.stderr)
            for error in replay_errors[:10]:
                print(f"  - {error}", file=sys.stderr)
            return 4

        cache_dir = Path(args.cache_dir) if args.cache_dir else None
        finalist_union = sorted(union)

        # --- GW4 10k confirmation (one common world set) -------------------------
        gw4_config = ro.OptimizerConfig(events=EVENTS, search_draws=int(args.gw4_draws), seed=20260911,
                                        policy_selection_worlds=int(args.policy_worlds))
        mark = time.time()
        gw4_worlds, gw4_info = ro.build_event_worlds(conn, {4: _bundle(4, int(args.gw4_draws))}, 4,
                                                     finalist_union, gw4_config, cache_dir=cache_dir)
        gw4_seconds = round(time.time() - mark, 3)
        gw4_cache_bytes = None
        if cache_dir is not None:
            key = ro.world_cache_key(event=4, bundle=_bundle(4, int(args.gw4_draws)), config=gw4_config,
                                     union_ids=finalist_union)
            path = cache_dir / f"{key}.json"
            gw4_cache_bytes = path.stat().st_size if path.exists() else None

        run71_config_hash = monte_carlo.MonteCarloConfig(
            simulations=int(args.gw4_draws), seed=20260911, occupancy_audit=True).config_hash()
        replay_identity = {
            "mc_config_hash": run71_config_hash,
            "equals_run71_config_hash": run71_config_hash == GW4_MC_CONFIG_HASH,
            "run_ids": RUNS[4],
            "draws": int(args.gw4_draws),
            "seed": 20260911,
            "note": ("Downstream capture path produces no Phase-5 summaries; identity is proven by the "
                     "exact run-71 run ids, config hash, seed and draw count, plus the accepted Phase-6A "
                     "byte-identical replay test (replay digest == stored run-71 digest)."),
        }
        replay_identity["smoke_draws_not_run71"] = int(args.gw4_draws) != fa.GW4_DRAWS
        if int(args.gw4_draws) == fa.GW4_DRAWS and not replay_identity["equals_run71_config_hash"]:
            print("phase 8c HARD BLOCK: run-71 MC config identity mismatch", file=sys.stderr)
            return 4

        confirmations = fa.confirm_gw4(finalist_gw4_squads=gw4_squads, worlds=gw4_worlds,
                                       positions_of=lambda ids: {int(p): positions_by_id[int(p)] for p in ids},
                                       config=gw4_config, cache={})

        # --- GW5 / GW6 accepted 2k route-evaluation worlds (cache reused) --------
        pool = ro.build_search_pool(universe, squad["squad_ids"], base_config)
        ladder_union = ro.union_player_ids(initial_state, pool["pool_ids"])
        later_worlds = {}
        later_info = {}
        later_config = ro.OptimizerConfig(events=EVENTS, search_draws=fa.GW5_DRAWS, seed=20260911,
                                         policy_selection_worlds=int(args.policy_worlds))
        for event in EVENTS[1:]:
            matrix, info = ro.build_event_worlds(conn, {event: _bundle(event, fa.GW5_DRAWS)}, event,
                                                 ladder_union, later_config, cache_dir=cache_dir)
            later_worlds[event] = matrix
            later_info[str(event)] = {**info, "worlds": int(matrix["worlds"]), "union_players": len(ladder_union)}

        later_cache: dict = {}
        for finalist in finalists:
            signature = finalist["canonical_signature"]
            per_event = {item["event"]: item for item in route_replay[signature]["events"]}
            later_net = {}
            for event in EVENTS[1:]:
                entry = _select_and_score(per_event[event]["squad_ids"], later_worlds[event],
                                          lambda ids: {int(p): positions_by_id[int(p)] for p in ids},
                                          later_config, later_cache, event=event)
                hit = int(per_event[event]["hit_points"])
                later_net[event] = {"mean_gross_core": float(sum(entry["scores"]) / len(entry["scores"])),
                                    "hit_points": hit,
                                    "mean_net_core": float(sum(entry["scores"]) / len(entry["scores"])) - hit,
                                    "squad_hash": fa.gw4_policy_cache_key(event, per_event[event]["squad_ids"],
                                                          draws=int(later_worlds[event]["worlds"]))}
            recorded = finalist["recorded"]
            recorded_by_event = {int(item["event"]): item for item in recorded["per_event"]}
            gw4_hit = int(route_replay[signature]["events"][0]["hit_points"])
            confirmed = {
                "canonical_signature": signature,
                "selection_reasons": finalist["selection_reasons"],
                "gw4": {
                    **{k: v for k, v in confirmations[signature].items() if k != "world_scores"},
                    "hit_points": gw4_hit,
                    "mean_net_core": confirmations[signature]["mean_gross_core"] - gw4_hit,
                    "terminal_bank_tenths": int(route_replay[signature]["events"][0]["bank_after_tenths"]),
                    "terminal_ft": int(route_replay[signature]["events"][0]["ft_after"]),
                },
                "gw5": later_net[5], "gw6": later_net[6],
                "recorded_2k": recorded,
                "confirmed_3gw_net_core": fa.recompute_supported_3gw(
                    confirmations[signature]["mean_gross_core"] - gw4_hit,
                    later_net[5]["mean_net_core"], later_net[6]["mean_net_core"]),
                "cumulative_hits": gw4_hit + later_net[5]["hit_points"] + later_net[6]["hit_points"],
                "terminal_ft": int(route_replay[signature]["final_state"].free_transfers),
                "terminal_bank_tenths": int(route_replay[signature]["final_state"].bank_tenths),
                "movement_10k_minus_2k_h1": (confirmations[signature]["mean_gross_core"] - gw4_hit
                                             - float(recorded_by_event[4]["mean_net_core"])),
                "movement_10k_minus_2k_3gw": None,
            }
            confirmed["movement_10k_minus_2k_3gw"] = (confirmed["confirmed_3gw_net_core"]
                                                      - float(recorded["supported_3gw_net_core"]))
            confirmations[signature]["confirmed"] = confirmed

        roll_signature = selection["leader_signatures"]["roll"]
        paired_roll = fa.paired_vs_roll(confirmations, roll_signature) if roll_signature else []
        leaders = sorted(confirmations, key=lambda sig: -confirmations[sig]["confirmed"]["confirmed_3gw_net_core"])
        paired_leaders = fa.paired_between(confirmations, list(zip(leaders[:4], leaders[1:5])))
        confirmed_by_sig = {sig: entry["confirmed"] for sig, entry in confirmations.items()}
        frontier = fa.confirmed_frontier(confirmed_by_sig)

        h1_leader_2k = selection["leader_signatures"]["h1"]
        h1_values = {sig: confirmations[sig]["confirmed"]["gw4"]["mean_net_core"] for sig in confirmations}
        h1_leader_10k = max(sorted(h1_values), key=lambda sig: h1_values[sig])
        h1_shift = h1_values[h1_leader_10k] - h1_values.get(h1_leader_2k, float("-inf"))
        high_fidelity_leader_changed = (h1_leader_10k != h1_leader_2k) and h1_shift > fa.CONFIRMATION_MATERIAL_CORE
        leader_3gw_10k = leaders[0]
        leader_3gw_2k = selection["leader_signatures"]["supported_3gw"]
        leader_changed_3gw = leader_3gw_10k != leader_3gw_2k
        # Material only if the new leader is clearly ahead (>0.5 CORE) or its GW4
        # paired evidence is not near-tied; a noise-driven swap of near-tied
        # families (both already certified finalists) is not material.
        margin_3gw = (confirmations[leader_3gw_10k]["confirmed"]["confirmed_3gw_net_core"]
                      - confirmations[leader_3gw_2k]["confirmed"]["confirmed_3gw_net_core"]
                      if leader_3gw_2k in confirmations else float("inf"))
        h1_pair = fa.paired_between(confirmations, [(leader_3gw_10k, leader_3gw_2k)]) if leader_changed_3gw else []
        near_tied_pair = bool(h1_pair and h1_pair[0]["near_tied"])
        leader_changed_materially = bool(leader_changed_3gw) and (
            margin_3gw > fa.CONFIRMATION_MATERIAL_CORE or not near_tied_pair)
        # Every finalist is a certified optimizer route, so a material 3GW leader
        # change among finalists is a reported finding, not an acceptance blocker.
        leader_changed_materially_uncertified = bool(leader_changed_materially) and leader_3gw_10k not in {
            f["canonical_signature"] for f in finalists}

        audit = _cross_section_audit(squad=squad, initial_state=initial_state, universe_ids=universe_ids,
                                     finalists=finalists, replay=route_replay, confirmations=confirmations,
                                     selection=selection, positions_by_id=positions_by_id)

        artifact = {
            "phase_version": PHASE8C,
            "planning_context_hash": _context_hash(context),
            "planning_cutoff": PLANNING_CUTOFF,
            "supported_events": list(EVENTS),
            "accepted_source_artifacts": {
                "candidate_universe": str(exports / "candidates" / f"gw{int(args.gw):02d}" / "candidate_universe.json"),
                "optimizer": str(opt_path),
                "certified_budget": certified_budget,
            },
            "price_scenario": fa.PRICE_SCENARIO, "price_scenario_flags": [fa.PRICE_FLAG],
            "score_basis": "CORE",
            "flags": [fa.H4_H6_FLAG, fa.CROSS_GW_FLAG, fa.PRICE_FLAG, ro.BOUNDED_PRUNING_LABEL,
                      ro.HEURISTIC_LABEL, ro.BEST_WITHIN_SEARCH_LABEL],
            "draw_fidelity": {"gw4": int(args.gw4_draws), "gw5": fa.GW5_DRAWS, "gw6": fa.GW6_DRAWS},
            "replay_identity": replay_identity,
            "world_info": {"gw4": {**gw4_info, "worlds": int(gw4_worlds["worlds"]),
                                   "union_players": len(finalist_union), "seconds": gw4_seconds,
                                   "cache_bytes": gw4_cache_bytes},
                           **{f"gw{event}": later_info[str(event)] for event in EVENTS[1:]}},
            "finalist_selection": {k: v for k, v in selection.items() if k != "finalists"},
            "finalists": [entry["confirmed"] for entry in (confirmations[sig] for sig in sorted(confirmations))],
            "paired_vs_roll": paired_roll,
            "paired_between_leaders": paired_leaders,
            "confirmed_shortlist_pareto_frontier": frontier,
            "confirmed_shortlist_pareto_name": fa.FRONTIER_NAME,
            "high_fidelity": {
                "h1_leader_2k": h1_leader_2k, "h1_leader_10k": h1_leader_10k,
                "h1_leader_changed": h1_leader_10k != h1_leader_2k,
                "h1_shift_core": h1_shift, "material_threshold_core": fa.CONFIRMATION_MATERIAL_CORE,
                "HIGH_FIDELITY_H1_LEADER_CHANGED": bool(high_fidelity_leader_changed),
                "supported_3gw_leader_2k": selection["leader_signatures"]["supported_3gw"],
                "supported_3gw_leader_10k": leader_3gw_10k,
                "supported_3gw_leader_changed": bool(leader_changed_3gw),
                "supported_3gw_leader_changed_materially": bool(leader_changed_materially),
                "supported_3gw_leader_margin_core": None if margin_3gw == float("inf") else margin_3gw,
                "supported_3gw_leader_swap_near_tied": near_tied_pair,
                "new_3gw_leader_was_certified_finalist": leader_3gw_10k in {f["canonical_signature"] for f in finalists},
                "SUPPORTED_3GW_LEADER_CHANGED_MATERIALLY_UNCERTIFIED": bool(leader_changed_materially_uncertified),
            },
            "cross_section_audit": audit,
            "timing_s": {"total": round(time.time() - started, 3)},
            # §14: a material H1 leader change is the acceptance blocker.
            # §15: a material SUPPORTED_3GW leader change is only a blocker when the
            # new leader was NOT already in the certified optimizer set (it always is
            # here, since finalists come from certified routes).
            "acceptance": "COMPLETE" if (audit["status"] == "PASS"
                                         and not high_fidelity_leader_changed
                                         and not leader_changed_materially_uncertified
                                         and (replay_identity["equals_run71_config_hash"]
                                              or replay_identity["smoke_draws_not_run71"])) else "PARTIAL",
            "no_recommendation": True,
        }
        out_dir = Path(args.out) if args.out else exports / "final_acceptance" / f"gw{int(args.gw):02d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / "final_acceptance.json"
        target.write_text(json.dumps(cu.jsonable(artifact), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                          encoding="utf-8")
        section = {k: artifact[k] for k in ("phase_version", "planning_cutoff", "supported_events", "score_basis",
                                            "price_scenario", "draw_fidelity", "flags", "high_fidelity",
                                            "confirmed_shortlist_pareto_frontier", "cross_section_audit",
                                            "acceptance", "no_recommendation")}
        section["final_acceptance_artifact"] = str(target)
        section["finalist_count"] = len(finalists)
        built = packet_mod.build_decision_packet(conn, config, event=int(args.gw), final_acceptance=section)
        (out_dir / "final_acceptance_packet.json").write_text(packet_mod.packet_to_json(built["packet"]),
                                                             encoding="utf-8")
        (out_dir / "final_acceptance_packet.md").write_text(packet_mod.render_packet_markdown(built["packet"]),
                                                           encoding="utf-8")

        print(f"phase 8c complete: GW{args.gw} finalists={len(finalists)} union={len(finalist_union)} "
              f"gw4_cache={gw4_info['source']} gw4_seconds={gw4_seconds}")
        print(f"  audit={audit['status']} acceptance={artifact['acceptance']} "
              f"H1 leader changed={artifact['high_fidelity']['h1_leader_changed']} "
              f"3GW leader changed={artifact['high_fidelity']['supported_3gw_leader_changed_materially']}")
        print("  | finalist | 2k H1 | 10k H1 | 2k 3GW | 10k 3GW | hits | FT | bank | reasons |")
        for row in artifact["finalists"]:
            print(f"  | {row['canonical_signature'][:44]:44s} | {row['recorded_2k']['per_event'][0]['mean_net_core']:.3f} "
                  f"| {row['gw4']['mean_net_core']:.3f} | {row['recorded_2k']['supported_3gw_net_core']:.3f} "
                  f"| {row['confirmed_3gw_net_core']:.3f} | {row['cumulative_hits']} | {row['terminal_ft']} "
                  f"| {row['terminal_bank_tenths']} | {','.join(row['selection_reasons'])} |")
        print(f"  confirmed frontier={frontier}")
        print(f"  total={artifact['timing_s']['total']}s artifact={target}")
        return 0
    finally:
        conn.close()


def _cross_section_audit(*, squad, initial_state, universe_ids, finalists, replay, confirmations, selection,
                         positions_by_id) -> dict:
    checks: list[dict[str, str]] = []

    def check(name, ok, detail=""):
        checks.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": str(detail)})

    check("current_squad_matches_planning_context",
          {int(p.player_id) for p in initial_state.players} == {int(p) for p in squad["squad_ids"]})
    check("all_route_transfers_legal", all(entry["ok"] for entry in replay.values()))
    for finalist in finalists:
        signature = finalist["canonical_signature"]
        entry = confirmations.get(signature)
        if entry is None:
            check(f"finalist_present:{signature[:20]}", False, "missing confirmation")
            continue
        policy = entry["policy"]
        xi = list(policy["starter_ids"])
        check(f"captain_in_xi:{signature[:20]}", policy["captain_id"] in xi)
        check(f"vice_in_xi:{signature[:20]}", policy["vice_captain_id"] in xi and policy["vice_captain_id"] != policy["captain_id"])
        check(f"unique_gw4_squad:{signature[:20]}", len(set(xi)) == 11)
        check(f"squad_ids_canonical:{signature[:20]}", len(str(entry["squad_hash"])) == 16)
        check(f"players_in_universe:{signature[:20]}",
              all(int(pid) in universe_ids for pid in (policy["starter_ids"] + policy["bench_outfield_order"])))
    # layer agreement: hits / bank / FT
    for finalist in finalists:
        signature = finalist["canonical_signature"]
        events = {item["event"]: item for item in replay[signature]["events"]}
        recorded = {int(item["event"]): item for item in finalist["recorded"]["per_event"]}
        for event, item in events.items():
            if event in recorded:
                check(f"hit_agreement:{signature[:16]}:{event}", int(item["hit_points"]) == int(recorded[event]["hit_points"]))
        check(f"final_ft_agreement:{signature[:16]}",
              int(replay[signature]["final_state"].free_transfers) == int(finalist["recorded"]["terminal_ft"]))
        check(f"final_bank_agreement:{signature[:16]}",
              int(replay[signature]["final_state"].bank_tenths) == int(finalist["recorded"]["terminal_bank_tenths"]))
    status = "PASS" if all(item["status"] == "PASS" for item in checks) else "FAIL"
    return {"status": status, "checks": checks,
            "failed": [item for item in checks if item["status"] == "FAIL"],
            "check_count": len(checks)}


def _context_hash(context) -> str:
    from fpl_brain import analytics
    return analytics.canonical_hash({"entry_id": context.entry_id, "event": context.planning_event,
                                     "squad": context.squad, "official_runs": context.official_runs})


if __name__ == "__main__":
    raise SystemExit(main())
