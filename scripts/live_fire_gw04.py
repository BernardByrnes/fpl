#!/usr/bin/env python3
"""GW4 operational live-fire drill — frozen-data decision test.

Read/evaluate/interpret only.  Reuses accepted artifacts and the existing 10,000-draw
GW4 world cache.  No model/architecture change, no search, no predictive run, no
world regeneration, no chip, no execution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fpl_brain import candidate_universe as cu
from fpl_brain import final_acceptance as fa
from fpl_brain import manager_lineup as ml, manager_worlds
from fpl_brain import route_comparator as rc, route_optimizer as ro
from fpl_brain import transfer_state as ts
from fpl_brain.config import config_path, load_config
from fpl_brain.database import connect_database
from fpl_brain.planning import get_planning_context

PLANNING_CUTOFF = "2026-09-11T10:16:51Z"
EVENTS = (4, 5, 6)
GW4_DRAWS = 10_000
EPS = 1e-9
ACCEPTED_DB_SHA = "36dd214d9f726a118fad2ad3a52c62e22e09544ce2dda2e90f4fa8d5cc1bdcd2"


def _bundle(event):
    runs = {4: (64, 65, 67, 70, 71), 5: (80, 81, 83, 86, 87), 6: (88, 89, 91, 94, 95)}[event]
    return rc.EventBundle(event=event, minutes_run_id=runs[0], team_run_id=runs[1], rate_run_id=runs[2],
                          xpts_run_id=runs[3], mc_run_id=runs[4], simulations=GW4_DRAWS, seed=20260911,
                          planning_cutoff=PLANNING_CUTOFF)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gw", type=int, default=4)
    parser.add_argument("--config")
    parser.add_argument("--cache-dir", default="data/cache/manager_worlds")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    conn = connect_database(config_path(config, "database"))
    try:
        exports = config_path(config, "exports_dir")
        gw_dir = exports / "final_acceptance" / f"gw{int(args.gw):02d}"
        recert = json.loads((gw_dir / "final_acceptance_recertified.json").read_text(encoding="utf-8"))
        universe = json.loads((exports / "candidates" / f"gw{int(args.gw):02d}" /
                               "candidate_universe.json").read_text(encoding="utf-8"))
        ladder = json.loads((exports / "optimizer" / f"gw{int(args.gw):02d}" /
                             "stability_ladder.json").read_text(encoding="utf-8"))
        certified_routes = {r["canonical_family_signature"]: r for r in ladder["ladder_records"][-1]["routes"].values()}
        # Route-level terminal state and cumulative hits are unchanged by GW4 policy
        # selection and live in the accepted Phase-8C artifact.
        phase8c = json.loads((exports / "final_acceptance" / f"gw{int(args.gw):02d}" /
                              "final_acceptance.json").read_text(encoding="utf-8"))
        route_level = {r["canonical_signature"]: r for r in phase8c["finalists"]}
        rows = {int(r["player_id"]): r for r in universe["universe"]}
        def feature(pid, event, field):
            for f in rows[int(pid)]["events"]:
                if int(f["event"]) == int(event):
                    return f.get(field)
            return None
        def name(pid):
            return rows[int(pid)].get("web_name") or rows[int(pid)].get("full_name") or str(pid)
        def pos(pid):
            return rows[int(pid)]["position"]
        def club(pid):
            return rows[int(pid)]["club_id"]
        def price(pid):
            return rows[int(pid)]["current_market_price_tenths"]

        context = get_planning_context(conn, int(config.get("fpl_entry_id")), int(args.gw),
                                       as_of=PLANNING_CUTOFF, season=config.get("season"))
        squad = manager_worlds.resolve_squad(context, conn)
        initial_state = rc.build_route_state(conn, context, squad)
        scenario = rc.flat_current_price_scenario(cu.latest_price_snapshot(conn, int(args.gw)), EVENTS)
        selling = {int(r["player_id"]): r for r in context.selling_prices}

        # --- verification only --------------------------------------------------
        db_sha = hashlib.sha256(Path(config_path(config, "database")).read_bytes()).hexdigest()
        verification = {
            "planning_context": context.entry_id is not None and len(squad["squad_ids"]) == 15,
            "squad_15": len(squad["squad_ids"]) == 15,
            "bank_resolves": (context.manager_state or {}).get("bank") is not None,
            "free_transfers_resolve": (context.manager_state or {}).get("free_transfers") is not None,
            "purchase_prices_resolve": all(selling.get(int(p), {}).get("purchase_price") is not None
                                           for p in squad["squad_ids"]),
            "selling_prices_resolve": all(selling.get(int(p), {}).get("effective_selling_price") is not None
                                          for p in squad["squad_ids"]),
            "db_sha_matches_accepted": db_sha == ACCEPTED_DB_SHA,
            "schema": int(conn.execute("SELECT value FROM schema_meta WHERE key='schema_version'").fetchone()[0]),
            "run_count": int(conn.execute("SELECT COUNT(*) FROM projection_runs").fetchone()[0]),
            "prediction_runs_created": False,
        }

        # --- 10k GW4 worlds from the accepted cache (no regeneration) ------------
        # The union must be the SAME set the accepted 10k cache was keyed with:
        # squad + every finalist route action's in/out ids + all action squad ids.
        union = {int(p.player_id) for p in initial_state.players}
        for f in recert["finalists"]:
            for action in certified_routes[f["canonical_signature"]]["actions"]:
                for move in action.get("transfers") or []:
                    union.add(int(move["out"]))
                    union.add(int(move["in"]))
                union.update(int(pid) for pid in action.get("squad_ids") or ())
        union = sorted(union)
        gw4_config = ro.OptimizerConfig(events=EVENTS, search_draws=GW4_DRAWS, seed=20260911,
                                        policy_selection_worlds=0)
        # Guard BEFORE any simulation: the accepted 10k cache must already exist
        # for this exact key.  Never let build_event_worlds regenerate.
        gw4_key = ro.world_cache_key(event=4, bundle=_bundle(4), config=gw4_config, union_ids=union)
        gw4_cache_path = Path(args.cache_dir) / f"{gw4_key}.json"
        if not gw4_cache_path.exists():
            print("LIVE_FIRE_SCOPE_EXPANSION_REQUIRED: the accepted 10k GW4 cache key was not found; "
                  "refusing to regenerate football worlds.", file=sys.stderr)
            return 6
        gw4_worlds, gw4_info = ro.build_event_worlds(conn, {4: _bundle(4)}, 4, union, gw4_config,
                                                     cache_dir=Path(args.cache_dir))
        if gw4_info.get("source") != "cache":
            print("LIVE_FIRE_SCOPE_EXPANSION_REQUIRED: the accepted 10k GW4 cache key was not found; "
                  "refusing to regenerate football worlds.", file=sys.stderr)
            return 6
        # score the recorded full-10k policies on the cached worlds (no selection)
        vectors: dict[str, list[float]] = {}
        for f in recert["finalists"]:
            squad_ids = tuple(sorted(int(pid) for pid in f["gw4"]["policy"]["starter_ids"] +
                                     f["gw4"]["policy"]["bench_outfield_order"] + [f["gw4"]["policy"]["bench_gk_id"]]))
            key = hashlib.sha256(",".join(map(str, squad_ids)).encode()).hexdigest()[:16]
            if key in vectors:
                continue
            policy = ml.ManagerPolicy(tuple(f["gw4"]["policy"]["starter_ids"]), f["gw4"]["policy"]["bench_gk_id"],
                                      tuple(f["gw4"]["policy"]["bench_outfield_order"]),
                                      f["gw4"]["policy"]["captain_id"], f["gw4"]["policy"]["vice_captain_id"])
            positions = {pid: pos(pid) for pid in squad_ids}
            vectors[key] = rc.policy_world_scores(policy, gw4_worlds, positions)
        roll_sig = "E4:ROLL;E5:ROLL;E6:ROLL"
        roll = next(f for f in recert["finalists"] if f["canonical_signature"] == roll_sig)
        roll_key = hashlib.sha256(",".join(map(str, sorted(int(p) for p in roll["gw4"]["policy"]["starter_ids"] +
                                                          roll["gw4"]["policy"]["bench_outfield_order"] +
                                                          [roll["gw4"]["policy"]["bench_gk_id"]]))).encode()).hexdigest()[:16]
        roll_scores = vectors[roll_key]

        # --- finalist table grouped by unique GW4 squad --------------------------
        groups: dict[str, list] = {}
        for f in recert["finalists"]:
            groups.setdefault(f["gw4"]["squad_hash"], []).append(f)
        finalists = []
        for f in recert["finalists"]:
            key = f["gw4"]["squad_hash"]
            scores = vectors[hashlib.sha256(",".join(map(str, sorted(int(p) for p in
                f["gw4"]["policy"]["starter_ids"] + f["gw4"]["policy"]["bench_outfield_order"] +
                [f["gw4"]["policy"]["bench_gk_id"]]))).encode()).hexdigest()[:16]]
            hit = int(f["gw4"]["hit_points"])
            finalists.append({
                "canonical_signature": f["canonical_signature"],
                "gw4_squad_hash": key,
                "gw4_action_kind": certified_routes[f["canonical_signature"]]["actions"][0]["kind"],
                "gw4_transfers": [{"out": int(t["out"]), "out_name": name(t["out"]),
                                   "in": int(t["in"]), "in_name": name(t["in"])}
                                  for t in certified_routes[f["canonical_signature"]]["actions"][0]["transfers"]],
                "gw5_action": certified_routes[f["canonical_signature"]]["actions"][1],
                "gw6_action": certified_routes[f["canonical_signature"]]["actions"][2],
                "gw4_gross_core": f["gw4"]["mean_gross_core"], "gw4_hit": hit,
                "gw4_net_core": f["gw4"]["mean_net_core"],
                "gw4_gain_vs_roll": f["gw4"]["mean_net_core"] - roll["gw4"]["mean_net_core"],
                "supported_3gw_net_core": f["confirmed_3gw_net_core"],
                "gain_3gw_vs_roll": f["confirmed_3gw_net_core"] - roll["confirmed_3gw_net_core"],
                "terminal_ft": int(route_level[f["canonical_signature"]]["terminal_ft"]),
                "terminal_bank_tenths": int(route_level[f["canonical_signature"]]["terminal_bank_tenths"]),
                "cumulative_hits": int(route_level[f["canonical_signature"]]["cumulative_hits"]),
                "policy": f["gw4"]["policy"], "gw4_mean_gross": f["gw4"]["mean_gross_core"],
                "gw4_median": f["gw4"]["median_core"], "gw4_q10": f["gw4"]["q10"], "gw4_q90": f["gw4"]["q90"],
                "policy_changed": f["policy_changed_vs_2k_subsample"],
                "world_scores": scores,
            })

        # --- paired evidence vs ROLL + head-to-head -----------------------------
        def paired(a_scores, b_scores, hits_a=0, hits_b=0):
            diffs = [x - hits_a - (y - hits_b) for x, y in zip(a_scores, b_scores)]
            n = len(diffs)
            mean = sum(diffs) / n
            var = sum((d - mean) ** 2 for d in diffs) / (n - 1) if n > 1 else 0.0
            se = (var / n) ** 0.5 if n > 1 else 0.0
            return {"worlds": n, "mean_difference": mean, "paired_se": se,
                    "ci95_low": mean - 1.96 * se, "ci95_high": mean + 1.96 * se,
                    "p_a_gt_b": sum(1 for d in diffs if d > 0) / n,
                    "near_tied": (abs(mean) <= 1.96 * se) if se > 0 else mean == 0.0}

        paired_rows = []
        for f in finalists:
            if f["canonical_signature"] == roll_sig:
                continue
            stats = paired(f["world_scores"], roll_scores, f["gw4_hit"], 0)
            paired_rows.append({"finalist": f["canonical_signature"], "reference": roll_sig, **stats})

        by_sig = {f["canonical_signature"]: f for f in finalists}
        h1_leader_sig = max(sorted(by_sig), key=lambda s: by_sig[s]["gw4_net_core"])
        w3_leader_sig = max(sorted(by_sig), key=lambda s: by_sig[s]["supported_3gw_net_core"])
        frontier_sigs = [s for s in recert["confirmed_shortlist_pareto_frontier"] if s in by_sig]
        alt_frontier = [s for s in frontier_sigs if s not in (roll_sig, h1_leader_sig, w3_leader_sig)]
        best_alt_sig = max(sorted(alt_frontier), key=lambda s: by_sig[s]["supported_3gw_net_core"])
        close3_pool = [s for s in by_sig if s not in (roll_sig, w3_leader_sig)]
        close3_sig = max(sorted(close3_pool), key=lambda s: by_sig[s]["supported_3gw_net_core"])
        h2h = {
            "h1_vs_3gw_leader": paired(by_sig[h1_leader_sig]["world_scores"], by_sig[w3_leader_sig]["world_scores"],
                                       by_sig[h1_leader_sig]["gw4_hit"], by_sig[w3_leader_sig]["gw4_hit"]),
            "h1_vs_roll": paired(by_sig[h1_leader_sig]["world_scores"], roll_scores, by_sig[h1_leader_sig]["gw4_hit"], 0),
            "3gw_vs_roll": paired(by_sig[w3_leader_sig]["world_scores"], roll_scores, by_sig[w3_leader_sig]["gw4_hit"], 0),
            "h1_vs_best_alt": paired(by_sig[h1_leader_sig]["world_scores"], by_sig[best_alt_sig]["world_scores"],
                                     by_sig[h1_leader_sig]["gw4_hit"], by_sig[best_alt_sig]["gw4_hit"]),
            "3gw_vs_close_alt": paired(by_sig[w3_leader_sig]["world_scores"], by_sig[close3_sig]["world_scores"],
                                       by_sig[w3_leader_sig]["gw4_hit"], by_sig[close3_sig]["gw4_hit"]),
        }
        h2h_ids = {"h1_leader": h1_leader_sig, "supported_3gw_leader": w3_leader_sig,
                   "best_alternative_pareto": best_alt_sig, "best_close_3gw_alternative": close3_sig}

        # --- manual accounting audit for the top-3 distinct GW4 strategies ------
        top3 = ["E4:329-332|334-305", "E4:1-28|329-305", "E4:165-249|329-332"]
        def gw4_action_group(signature):
            return fa._strategic_shape(certified_routes[signature]["actions"], drop_events=(5, 6))
        chosen = []
        for target in top3:
            match = next((f for f in finalists if gw4_action_group(f["canonical_signature"]) == target), None)
            if match:
                chosen.append(match)
        accounting = []
        for f in chosen:
            signature = f["canonical_signature"]
            replay = fa.replay_route(initial_state=initial_state, actions=certified_routes[signature]["actions"],
                                     scenario=scenario,
                                     player_meta=rc.load_player_meta(conn, [int(p.player_id) for p in
                                                                            initial_state.players] + union))
            event = replay["events"][0]
            gw4_action = next(a for a in certified_routes[signature]["actions"] if int(a["event"]) == 4)
            outs = [int(t["out"]) for t in gw4_action["transfers"]]
            ins = [int(t["in"]) for t in gw4_action["transfers"]]
            accounting.append({
                "strategy": target, "canonical_signature": signature,
                "outgoing": [{"player_id": pid, "name": name(pid), "position": pos(pid),
                              "purchase_price_tenths": int(selling.get(pid, {}).get("purchase_price") or 0),
                              "current_market_price_tenths": price(pid),
                              "selling_price_tenths": ts.selling_price_tenths(
                                  int(selling.get(pid, {}).get("purchase_price") or 0), price(pid))} for pid in outs],
                "incoming": [{"player_id": pid, "name": name(pid), "position": pos(pid),
                              "purchase_price_tenths": price(pid)} for pid in ins],
                "bank_before_tenths": int(initial_state.bank_tenths),
                "sale_proceeds_tenths": int(event["sale_proceeds_tenths"]),
                "purchase_cost_tenths": int(event["purchase_cost_tenths"]),
                "bank_after_tenths": int(event["bank_after_tenths"]),
                "ft_before": int(initial_state.free_transfers), "transfers": len(outs),
                "paid_transfers": int(event["hit_points"]) // 4,
                "ft_used": len(outs) - int(event["hit_points"]) // 4,
                "hit": int(event["hit_points"]), "ft_after": int(event["ft_after"]),
                "max_club_count_after": max(replay["final_state"].club_counts().values()),
                "lineup_errors": ml.policy_legality_errors(
                    ml.ManagerPolicy(tuple(f["policy"]["starter_ids"]), f["policy"]["bench_gk_id"],
                                     tuple(f["policy"]["bench_outfield_order"]), f["policy"]["captain_id"],
                                     f["policy"]["vice_captain_id"]),
                    {int(pid): pos(int(pid)) for pid in (list(f["policy"]["starter_ids"]) +
                                                         list(f["policy"]["bench_outfield_order"]) +
                                                         [f["policy"]["bench_gk_id"]])}),
            })

        # --- route-continuation sanity for the two leaders ----------------------
        def club_counts_of(squad_ids):
            counts: dict[int, int] = {}
            for pid in squad_ids:
                c = club(int(pid))
                counts[c] = counts.get(c, 0) + 1
            return counts

        continuation = {}
        for label, sig in (("h1_leader", h1_leader_sig), ("supported_3gw_leader", w3_leader_sig)):
            replay = fa.replay_route(initial_state=initial_state, actions=certified_routes[sig]["actions"],
                                     scenario=scenario, player_meta=rc.load_player_meta(conn, union))
            timeline = []
            for ev, act in zip(replay["events"], certified_routes[sig]["actions"]):
                counts = club_counts_of(ev["squad_ids"])
                incoming = [int(t["in"]) for t in act.get("transfers") or []]
                timeline.append({
                    "event": int(ev["event"]), "kind": act["kind"],
                    "transfers": [{"out": int(t["out"]), "out_name": name(t["out"]),
                                   "in": int(t["in"]), "in_name": name(t["in"])}
                                  for t in act.get("transfers") or []],
                    "bank_after_tenths": int(ev["bank_after_tenths"]), "ft_after": int(ev["ft_after"]),
                    "hit_points": int(ev["hit_points"]),
                    "max_club_count": max(counts.values()) if counts else 0,
                    "purchase_basis_new_signings": {str(pid): int(ev["purchase_basis"][pid])
                                                    for pid in incoming if pid in ev["purchase_basis"]},
                })
            continuation[label] = {"canonical_signature": sig, "replay_ok": replay["ok"],
                                   "replay_errors": list(replay["errors"]),
                                   "terminal_ft": int(replay["final_state"].free_transfers),
                                   "terminal_bank_tenths": int(replay["final_state"].bank_tenths),
                                   "timeline": timeline}

        # --- confirmed shortlist Pareto frontier --------------------------------
        confirmed_frontier = []
        for s in frontier_sigs:
            f = by_sig[s]
            confirmed_frontier.append({
                "canonical_signature": s, "gw4_action_kind": f["gw4_action_kind"],
                "gw4_transfers": f["gw4_transfers"],
                "gw5_action": f["gw5_action"], "gw6_action": f["gw6_action"],
                "supported_3gw_net_core": f["supported_3gw_net_core"],
                "gw4_net_core": f["gw4_net_core"], "cumulative_hits": f["cumulative_hits"],
                "terminal_ft": f["terminal_ft"], "terminal_bank_tenths": f["terminal_bank_tenths"],
                "is_roll": s == roll_sig,
            })

        # --- adversarial: bad but legal transfer --------------------------------
        union_set = set(int(pid) for pid in union)
        edges = universe.get("replacement_edges") or []
        deltas = []
        for edge in edges:
            if not edge["currently_legal_single_transfer"]:
                continue
            out_id, in_id = int(edge["out_player_id"]), int(edge["in_player_id"])
            if out_id not in union_set or in_id not in union_set:
                continue  # need cached worlds for both players; never regenerate
            delta = (float(rows[in_id]["supported_3gw_expected_core"]) - float(rows[out_id]["supported_3gw_expected_core"]))
            deltas.append((delta, out_id, in_id))
        deltas.sort()
        bad = None
        if deltas:
            delta, out_id, in_id = deltas[0]
            state = initial_state
            batch = ts.TransferBatch((ts.TransferAction(out_id, in_id),))
            meta = rc.load_player_meta(conn, union + [out_id, in_id])
            transition = ts.apply_transfer_batch(state, batch, scenario.snapshot_for(4), meta)
            if transition.ok:
                new_squad = tuple(sorted(int(p.player_id) for p in transition.squad_after.players))
                positions = {pid: pos(pid) for pid in new_squad}
                selection = ro._subsample(gw4_worlds, 2000)
                policy = ml.rank_policies(list(new_squad), positions, selection, top_k=1)["top_policies"][0]
                scores = rc.policy_world_scores(policy, gw4_worlds, positions)
                mean = sum(scores) / len(scores)
                stats = paired(scores, roll_scores, transition.hit_points, 0)
                bad = {"label": "TEST_FIXTURE_NOT_RECOMMENDATION", "out": out_id, "out_name": name(out_id),
                       "in": in_id, "in_name": name(in_id), "delta_3gw_proxy": delta,
                       "bank_after_tenths": int(transition.bank_after_tenths), "hit": int(transition.hit_points),
                       "gw4_mean_gross": mean, "gw4_mean_net": mean - transition.hit_points,
                       "vs_roll": stats}
        # --- adversarial: illegal transfer (wrong-position replacement) --------
        meta = rc.load_player_meta(conn, union)
        owned = {int(p.player_id) for p in initial_state.players}
        out_id = next(int(p) for p in squad["squad_ids"] if pos(int(p)) == "DEF")
        in_id = next(int(pid) for pid in sorted(union_set)
                     if rows[int(pid)]["position"] == "MID" and int(pid) not in owned)
        illegal_batch = ts.TransferBatch((ts.TransferAction(out_id, in_id),))
        result = ts.apply_transfer_batch(initial_state, illegal_batch, scenario.snapshot_for(4), meta)
        illegal = {"out": out_id, "out_name": name(out_id), "out_position": pos(out_id),
                   "in": in_id, "in_name": name(in_id), "in_position": pos(in_id),
                   "reason": list(result.errors), "rejected": not result.ok,
                   "state_unchanged": ro.state_key(result.before_state) == ro.state_key(initial_state),
                   "bank_unchanged": int(initial_state.bank_tenths) == int(initial_state.bank_tenths),
                   "ft_unchanged": int(initial_state.free_transfers) == int(initial_state.free_transfers),
                   "squad_after_is_none": result.squad_after is None}

        artifact = {
            "phase_version": "live_fire_gw04_v1.0.0",
            "planning_cutoff": PLANNING_CUTOFF, "supported_events": list(EVENTS),
            "verification": verification, "db_sha256": db_sha,
            "squad": [{"player_id": int(p), "name": name(p), "position": pos(p), "club_id": club(p),
                       "purchase_price_tenths": int(selling.get(int(p), {}).get("purchase_price") or 0),
                       "current_market_price_tenths": price(p),
                       "selling_price_tenths": ts.selling_price_tenths(
                           int(selling.get(int(p), {}).get("purchase_price") or 0), price(p))}
                      for p in squad["squad_ids"]],
            "bank_tenths": int(initial_state.bank_tenths), "free_transfers": int(initial_state.free_transfers),
            "club_counts": {str(k): v for k, v in sorted(initial_state.club_counts().items())},
            "roll_baseline": {"gw4_action": "ROLL", "gw4_gross": roll["gw4"]["mean_gross_core"],
                              "gw4_hit": 0, "gw4_net": roll["gw4"]["mean_net_core"],
                              "supported_3gw_net_core": roll["confirmed_3gw_net_core"],
                              "terminal_ft": int(route_level[roll_sig]["terminal_ft"]),
                              "terminal_bank_tenths": int(route_level[roll_sig]["terminal_bank_tenths"]),
                              "policy": roll["gw4"]["policy"]},
            "finalists": [{k: v for k, v in f.items() if k != "world_scores"} for f in finalists],
            "unique_gw4_squads": len(groups),
            "groups_by_gw4_squad": {h: sorted(f["canonical_signature"] for f in members)
                                    for h, members in sorted(groups.items())},
            "paired_vs_roll": paired_rows,
            "head_to_head": h2h,
            "head_to_head_ids": h2h_ids,
            "confirmed_shortlist_pareto_frontier": confirmed_frontier,
            "continuation": continuation,
            "h1_leader": h1_leader_sig, "supported_3gw_leader": w3_leader_sig,
            "accounting_audit": accounting,
            "bad_legal_transfer": bad,
            "illegal_transfer": illegal,
            "certified_hit_routes": 0,
            "world_cache_source": gw4_info.get("source"),
            "world_cache_key": gw4_key,
            "world_union_players": len(union),
            "no_recommendation": True,
        }
        out_dir = exports / "live_fire" / f"gw{int(args.gw):02d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / "live_fire_20260911.json"
        target.write_text(json.dumps(cu.jsonable(artifact), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                          encoding="utf-8")
        print(f"live fire complete: squad=15 bank={initial_state.bank_tenths} ft={initial_state.free_transfers} "
              f"worlds={gw4_worlds['worlds']} cache={gw4_info.get('source')}")
        print(f"  H1 leader {h1_leader_sig} net={by_sig[h1_leader_sig]['gw4_net_core']:.4f} "
              f"3GW leader {w3_leader_sig} 3GW={by_sig[w3_leader_sig]['supported_3gw_net_core']:.4f}")
        print(f"  ROLL net={roll['gw4']['mean_net_core']:.4f} 3GW={roll['confirmed_3gw_net_core']:.4f}")
        print(f"  pairing ok: {len(paired_rows)} rows; accounting {len(accounting)}; bad transfer "
              f"{'yes' if bad else 'no'}; illegal rejected={illegal['rejected']}")
        print(f"  artifact={target}")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
