#!/usr/bin/env python3
"""Phase 7B — compare explicit transfer routes in shared football worlds.

Descriptive only: no candidate generation, no optimization, no recommendation.
Routes are supplied by the caller (``--routes FILE``); the default is ROUTE_ROLL
(ROLL every event), which is the only real-squad engineering validation.

PE-9: the worlds are simulated from a CERTIFIED GENERATION.  The exact certified
run ids are read from the generation's digest-verified manifest, so this script
never rediscovers "the newest run per family" and cannot silently construct an
alternate predictive world.  ``--generation`` selects a specific certified
generation; omitting it resolves the ``current_generation`` pointer, and an unset
pointer refuses rather than falling back to a rediscovery.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fpl_brain import generation_store as gs
from fpl_brain import manager_worlds, packet as packet_mod, route_comparator as rc, transfer_state as ts
from fpl_brain.config import config_path, load_config
from fpl_brain.database import connect_database
from fpl_brain.planning import get_planning_context

PHASE7B_VERSION = "phase7b_route_comparator_v1.0.0"


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
    parser.add_argument(
        "--generation", default=None,
        help="explicit certified generation id SELECTOR.  Omit to resolve the current_generation "
             "pointer for this event; an unset pointer refuses rather than rediscovering runs",
    )
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
        # PE-9: the certified generation is the ONLY source of predictive run ids.
        try:
            generation = gs.resolve_generation(
                conn, planning_event=int(args.gw), generation_id=args.generation
            )
        except gs.GenerationRefused as refusal:
            print(f"phase 7b failed: {refusal}", file=sys.stderr)
            return 3
        gs.require_snapshot_retained(generation)
        gs.assert_generation_bundles_valid(conn, generation)
        events = [int(event) for event in generation.events]
        cutoff = str(generation.cutoff)

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

        result = rc.compare_routes(
            conn=conn, generation=generation, routes=routes, initial_state=initial_state,
            scenario=scenario, player_meta=player_meta, simulations=int(args.simulations),
            seed=int(args.seed), planning_cutoff=cutoff,
        )
        result["phase"] = PHASE7B_VERSION
        result["simulations"] = int(args.simulations)
        result["certified_generation"] = {
            "generation_id": generation.generation_id,
            "planning_event": int(generation.planning_event),
            "horizon_kind": generation.horizon_kind,
            "cutoff": cutoff,
            "events": events,
            "certified_runs_by_event": {str(event): generation.runs_for(event) for event in events},
            "model_versions_by_event": {
                str(event): generation.model_versions_by_event.get(int(event), {}) for event in events
            },
            "snapshot_path": generation.snapshot.get("path"),
            "snapshot_sha256": generation.snapshot.get("sha256"),
            "snapshot_source_db_identity": generation.snapshot.get("source_db_identity"),
            "execution_run_uuid": generation.snapshot.get("execution_run_uuid"),
        }
        result["blocked_events"] = []
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
