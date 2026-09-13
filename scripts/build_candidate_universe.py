#!/usr/bin/env python3
"""Phase 8A — build the candidate universe artifact (fast, no Monte Carlo).

Supplies Phase 8B with candidate targets only.  No recommendation, no ranking of
final routes, no execution.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fpl_brain import candidate_universe as cu
from fpl_brain import manager_worlds, packet as packet_mod, route_comparator as rc, transfer_state as ts
from fpl_brain.config import config_path, load_config
from fpl_brain.database import connect_database
from fpl_brain.planning import get_planning_context

PHASE8A = "phase8a_candidate_universe_v1.0.0"
SUPPORTED_EVENTS = (4, 5, 6)
XPTS_RUNS = {4: 70, 5: 86, 6: 94}
MINUTES_RUNS = {4: 64, 5: 80, 6: 88}
PLANNING_CUTOFF = "2026-09-11T10:16:51Z"  # the frozen route-comparison cutoff; no later info may leak


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gw", type=int, default=4)
    parser.add_argument("--config")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--out")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    conn = connect_database(config_path(config, "database"))
    try:
        started = time.time()
        context = get_planning_context(conn, int(config.get("fpl_entry_id")), int(args.gw),
                                       as_of=PLANNING_CUTOFF, season=config.get("season"))
        squad = manager_worlds.resolve_squad(context, conn)
        events = list(SUPPORTED_EVENTS)

        mark = time.time()
        pool = cu.load_pool(conn)
        events_fixtures = cu.load_fixtures_by_team(conn, events)
        from fpl_brain import analytics
        xpts_rows = {event: cu.load_projection_rows(conn, XPTS_RUNS[event]) for event in events}
        # Minutes rows come from the MINUTES_V1 frozen kind, not the xPts table.
        minutes_rows = {
            event: cu.load_projection_rows(conn, MINUTES_RUNS[event], analytics.MINUTES_V1_KIND)
            for event in events
        }
        snapshot = cu.latest_price_snapshot(conn, int(args.gw))
        build_s = time.time() - mark

        mark = time.time()
        universe = cu.build_universe(
            pool=pool, events_fixtures=events_fixtures, xpts_rows_by_event=xpts_rows,
            minutes_rows_by_event=minutes_rows, events=events, owned_ids=squad["squad_ids"],
            price_snapshot=snapshot, config=cu.CandidateConfig(top_n_per_criterion=int(args.top_n)),
            planning_cutoff=PLANNING_CUTOFF,
            run_refs={"xpts": XPTS_RUNS, "minutes": MINUTES_RUNS,
                      "events": events, "versions": {"minutes": "minutes_v1.5.2",
                                                     "xpts": "xpts_v1.4.1", "mc": "mc_v1.2.1"}},
        )
        universe["phase"] = PHASE8A
        universe_s = time.time() - mark

        # Current-squad coverage HARD FAIL.
        universe_ids = {int(row["player_id"]) for row in universe["universe"]}
        missing_owned = [pid for pid in squad["squad_ids"] if int(pid) not in universe_ids]
        if missing_owned:
            print(f"phase 8a failed: owned players missing from universe: {missing_owned}", file=sys.stderr)
            return 3

        mark = time.time()
        initial_state = rc.build_route_state(conn, context, squad, snapshot)
        player_meta = rc.load_player_meta(conn, [int(row["player_id"]) for row in universe["universe"]])
        edges = cu.build_replacement_edges(
            universe_rows=universe["universe"], owned_ids=squad["squad_ids"], state=initial_state,
            price_snapshot=snapshot, player_meta=player_meta,
        )
        edges_s = time.time() - mark
        universe["replacement_edges"] = edges
        universe["replacement_edge_count"] = len(edges)
        universe["legal_single_transfer_count"] = sum(1 for e in edges if e["currently_legal_single_transfer"])
        universe["illegal_single_transfer_count"] = sum(1 for e in edges if not e["currently_legal_single_transfer"])
        universe["squad_coverage"] = {"owned": len(squad["squad_ids"]), "present": len(squad["squad_ids"]) - len(missing_owned)}
        universe["timing_s"] = {"universe_build": round(build_s, 3), "universe_assemble": round(universe_s, 3),
                                "replacement_edges": round(edges_s, 3), "total": round(time.time() - started, 3)}

        out_dir = Path(args.out) if args.out else config_path(config, "exports_dir") / "candidates"
        target = out_dir / f"gw{int(args.gw):02d}"
        target.mkdir(parents=True, exist_ok=True)
        artifact = target / "candidate_universe.json"
        artifact.write_text(json.dumps(cu.jsonable(universe), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")

        section = cu.universe_metadata(universe)
        section["candidate_universe_artifact"] = str(artifact)
        section["replacement_edge_count"] = len(edges)
        section["legal_single_transfer_count"] = universe["legal_single_transfer_count"]
        section["illegal_single_transfer_count"] = universe["illegal_single_transfer_count"]
        section["squad_coverage"] = universe["squad_coverage"]
        built = packet_mod.build_decision_packet(conn, config, event=int(args.gw), candidate_universe=section)
        (target / "candidate_universe_packet.json").write_text(
            packet_mod.packet_to_json(built["packet"]), encoding="utf-8")
        (target / "candidate_universe_packet.md").write_text(
            packet_mod.render_packet_markdown(built["packet"]), encoding="utf-8")

        audit = universe["completeness_audit"]
        print(f"phase 8a complete: GW{args.gw} events={events} cutoff={context.as_of}")
        print(f"  universe={universe['universe_count']} {universe['counts_by_position']} "
              f"search_view={universe['search_view_count']} {universe['search_view_counts_by_position']}")
        print(f"  squad coverage {universe['squad_coverage']['present']}/15 | edges={len(edges)} "
              f"legal={universe['legal_single_transfer_count']} illegal={universe['illegal_single_transfer_count']}")
        for event, statuses in sorted(audit["by_event"].items()):
            print(f"  event {event}: {statuses}")
        print(f"  missing_projection rows={len(audit['missing_projection_players'])}")
        print(f"  timing={universe['timing_s']} artifact={artifact}")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
