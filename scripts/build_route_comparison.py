#!/usr/bin/env python3
"""Phase 7B — compare explicit transfer routes in shared football worlds.

Descriptive only: no candidate generation, no optimization, no recommendation.
Routes are supplied by the caller (``--routes FILE``); the default is ROUTE_ROLL
(ROLL every event), which is the only real-squad engineering validation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fpl_brain import manager_worlds, packet as packet_mod, route_comparator as rc, transfer_state as ts
from fpl_brain.config import config_path, load_config
from fpl_brain.database import connect_database
from fpl_brain.planning import get_planning_context

PHASE7B_VERSION = "phase7b_route_comparator_v1.0.0"
HORIZON_EVENTS = 6  # H6 = G .. G+5

FAMILY_VERSION_PREFIX = {
    "minutes_v1": "minutes_v1.5",
    "team_strength_v1": None,
    "player_rates_v1": None,
    "xpts_v1": None,
    "monte_carlo_v1": None,
}


def _latest_runs(conn, events) -> dict[int, dict[str, int]]:
    out: dict[int, dict[str, int]] = {}
    for event in events:
        row: dict[str, int] = {}
        for family, prefix in FAMILY_VERSION_PREFIX.items():
            sql = ("SELECT id, model_version FROM projection_runs WHERE model_family=? AND planning_event=? "
                   "AND status='complete'")
            params = [family, int(event)]
            if prefix:
                sql += " AND model_version LIKE ?"
                params.append(prefix + "%")
            sql += " ORDER BY id DESC LIMIT 1"
            found = conn.execute(sql, params).fetchone()
            if found:
                row[family] = int(found["id"])
        out[int(event)] = row
    return out


def _current_prices(conn, player_ids) -> dict[int, int]:
    ids = sorted({int(pid) for pid in player_ids})
    prices = {}
    if not ids:
        return prices
    for pid in ids:
        row = conn.execute(
            "SELECT now_cost FROM player_snapshots WHERE player_id=? ORDER BY captured_at DESC LIMIT 1", (pid,)
        ).fetchone()
        if row is not None:
            prices[pid] = int(row["now_cost"])
    return prices


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gw", type=int, default=4)
    parser.add_argument("--config")
    parser.add_argument("--simulations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--routes", help="JSON file with an explicit route list (list of route specs)")
    parser.add_argument("--out")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    conn = connect_database(config_path(config, "database"))
    try:
        started = time.time()
        context = get_planning_context(conn, int(config.get("fpl_entry_id")), int(args.gw),
                                       season=config.get("season"))
        squad = manager_worlds.resolve_squad(context, conn)
        # Longest CONTIGUOUS prefix of events that has a complete predictive bundle.
        all_events = list(range(int(args.gw), int(args.gw) + HORIZON_EVENTS))
        runs = _latest_runs(conn, all_events)
        families = ("minutes_v1", "team_strength_v1", "player_rates_v1", "xpts_v1", "monte_carlo_v1")
        events, blocked = [], []
        for event in all_events:
            if all(k in runs[event] for k in families):
                if not blocked:
                    events.append(event)
            else:
                blocked.append(event)
        if not events:
            print(f"phase 7b failed: no complete predictive bundle at GW{args.gw}", file=sys.stderr)
            return 3
        if blocked:
            print(f"phase 7b note: events {blocked} have no certified predictive bundle; "
                  f"route comparison limited to contiguous {events}")
        runs = {event: runs[event] for event in events}

        initial_state = rc.build_route_state(conn, context, squad)

        if args.routes:
            specs = json.loads(Path(args.routes).read_text(encoding="utf-8"))
            routes = [rc.parse_route_spec(spec) for spec in specs]
        else:
            routes = [rc.TransferRoute(
                route_id="ROUTE_ROLL", label="ROLL every event",
                steps=tuple(rc.RouteStep(event=e, transfer_batch=ts.TransferBatch.roll()) for e in events),
            )]

        union_ids = {int(p.player_id) for p in initial_state.players}
        for route in routes:
            for step in route.steps:
                union_ids.update(int(a.out_player_id) for a in step.transfer_batch.actions)
                union_ids.update(int(a.in_player_id) for a in step.transfer_batch.actions)
        player_meta = rc.load_player_meta(conn, union_ids)
        unknown = [pid for pid in union_ids if pid not in player_meta]
        if unknown:
            print(f"phase 7b failed: unknown player ids {unknown}", file=sys.stderr)
            return 3

        prices = _current_prices(conn, union_ids)
        missing_prices = [pid for pid in union_ids if pid not in prices]
        if missing_prices:
            print(f"phase 7b failed: missing current price for {missing_prices}", file=sys.stderr)
            return 3
        base_snapshot = ts.PriceSnapshot(event=int(args.gw), prices=prices)
        scenario = rc.flat_current_price_scenario(base_snapshot, events)

        bundles = {
            event: rc.EventBundle(
                event=event, minutes_run_id=runs[event]["minutes_v1"], team_run_id=runs[event]["team_strength_v1"],
                rate_run_id=runs[event]["player_rates_v1"], xpts_run_id=runs[event]["xpts_v1"],
                mc_run_id=runs[event].get("monte_carlo_v1"), simulations=int(args.simulations),
                seed=int(args.seed), planning_cutoff=context.as_of,
            )
            for event in events
        }

        result = rc.compare_routes(
            conn=conn, bundles=bundles, routes=routes, initial_state=initial_state,
            scenario=scenario, player_meta=player_meta, simulations=int(args.simulations),
            seed=int(args.seed), planning_cutoff=context.as_of,
        )
        result["phase"] = PHASE7B_VERSION
        result["simulations"] = int(args.simulations)
        result["predictive_bundles"] = {
            str(event): {**runs[event], "cutoff": context.as_of, "versions": {
                "minutes": "minutes_v1.5.2", "xpts": "xpts_v1.4.1", "mc": "mc_v1.2.1"}}
            for event in events
        }
        result["blocked_events"] = blocked
        result["route_table"] = _route_table(result)

        out_dir = Path(args.out) if args.out else config_path(config, "exports_dir") / "routes"
        target = out_dir / f"gw{int(args.gw):02d}"
        target.mkdir(parents=True, exist_ok=True)
        (target / "route_comparison.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        built = packet_mod.build_decision_packet(conn, config, event=int(args.gw), routes=result)
        (target / "route_comparison_packet.json").write_text(
            packet_mod.packet_to_json(built["packet"]), encoding="utf-8")
        (target / "route_comparison_packet.md").write_text(
            packet_mod.render_packet_markdown(built["packet"]), encoding="utf-8")

        elapsed = time.time() - started
        print(f"phase 7b complete: GW{args.gw} events={events} routes={result['route_count']} "
              f"valid={result['valid_route_count']} invalid={result['invalid_route_count']} "
              f"union={result['union_player_count']} worlds/event={args.simulations} "
              f"policy_cache={result['policy_cache_entries']} elapsed={elapsed:.1f}s artifact={target}")
        print("| route | valid | H1 net | H4 net | H6 net | cum hits | term FT | term bank |")
        for row in result["route_table"]:
            print(f"| {row['route_id']} | {row['valid']} | {row['H1_net']} | {row['H4_net']} | {row['H6_net']} | "
                  f"{row['cumulative_hits']} | {row['terminal_ft']} | {row['terminal_bank_tenths']} |")
        if result.get("worlds_generation"):
            print("worlds:", json.dumps(result["worlds_generation"]))
        return 0
    finally:
        conn.close()


def _route_table(result) -> list[dict]:
    rows = []
    for route_id, body in sorted(result["routes"].items()):
        horizons = body["horizons"]
        rows.append({
            "route_id": route_id,
            "valid": body["valid"],
            "valid_through_event": body["valid_through_event"],
            "failure": body["failure"],
            "H1_net": horizons.get("H1", {}).get("net_core"),
            "H4_net": horizons.get("H4", {}).get("net_core"),
            "H6_net": horizons.get("H6", {}).get("net_core"),
            "cumulative_hits": (horizons.get("H6") or horizons.get("H4") or horizons.get("H1") or {}).get("cumulative_hits"),
            "terminal_ft": (horizons.get("H6") or horizons.get("H4") or horizons.get("H1") or {}).get("terminal_ft"),
            "terminal_bank_tenths": (horizons.get("H6") or horizons.get("H4") or horizons.get("H1") or {}).get("terminal_bank_tenths"),
        })
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
