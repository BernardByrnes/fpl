#!/usr/bin/env python3
"""ARCHIVAL GW4/H1 diagnostic capture over the accepted FPL Brain components.

SCOPE: HISTORICAL GW4 / H1 DESCRIPTIVE DIAGNOSTIC — NOT the 4GW production runner.

* ``EVENTS = (4,)`` — a single GW4 world bundle only.
* The route table is the historical Rodon-era four-route comparison; Rodon is
  now sold, Davis is already owned, and those routes are no longer actionable.
* It passes ``routes=None`` into the four-GW decision output, so it can never
  emit a normal transfer recommendation.

The real rolling four-Gameweek (GW4-GW7) run must feed GW4/5/6/7 into the
accepted multi-event optimizer / route-comparator path and then into
``fpl_brain.four_gw_decision.evaluate_four_gw_decision``.  This script is kept
for archival/diagnostic value and must not be retrofitted into that role.

It still performs verification only: no route search, no search-stability
ladders, no H4/H6 fabrication, and no transfer/chip execution.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fpl_brain import analytics, candidate_universe as cu, four_gw_decision as fg, manager_worlds
from fpl_brain import packet as packet_mod
from fpl_brain import repositories as repo
from fpl_brain import route_comparator as rc
from fpl_brain import route_optimizer as ro
from fpl_brain import transfer_state as ts
from fpl_brain import execution
from fpl_brain.config import config_path, load_config
from fpl_brain.database import connect_database


def _refuse_execution(guard, code: int, reason: str) -> int:
    """Record a deliberate refusal as a FAILED execution run, then return the code."""

    guard.finish(execution.RUN_FAILED, reason)
    return code
from fpl_brain.planning import get_planning_context


FRESH_CUTOFF = "2026-09-11T22:46:18Z"
EVENT = 4
EVENTS = (4,)
# The canonical normal-transfer horizon: current GW + the next three.
DECISION_EVENTS = fg.decision_events(EVENT)
SEED = 20260911
SIMULATIONS = 10_000
RUNS = {
    "minutes": 114,
    "team": 115,
    "rates": 117,
    "xpts": 120,
    "monte_carlo": 121,
}

ROUTE_A = "A_RODON_JUSTIN_MUHAREMOVIC_DAVIS"
ROUTE_B = "B_RAYA_MARTINEZ_RODON_DAVIS"
ROUTE_RODON = "RODON_ONLY_DAVIS"
ROUTE_ROLL = "ROLL"


def _name(conn, player_id: int) -> str:
    row = conn.execute("SELECT web_name, full_name FROM players WHERE id=?", (int(player_id),)).fetchone()
    if row is None:
        return str(player_id)
    return str(row["web_name"] or row["full_name"] or player_id)


def _money_tenths(value) -> str:
    return "n/a" if value is None else f"£{int(value) / 10:.1f}m"


def _route(route_id: str, label: str, transfers: list[tuple[int, int]]) -> rc.TransferRoute:
    return rc.TransferRoute(
        route_id=route_id,
        label=label,
        steps=(rc.RouteStep(event=EVENT, transfer_batch=ts.TransferBatch(
            tuple(ts.TransferAction(out_id, in_id) for out_id, in_id in transfers)
        )),),
    )


def _policy_names(conn, policy: dict | None) -> dict | None:
    if not policy:
        return None
    return {
        **policy,
        "starter_names": [_name(conn, int(pid)) for pid in policy.get("starter_ids", [])],
        "bench_gk_name": _name(conn, int(policy["bench_gk_id"])) if policy.get("bench_gk_id") is not None else None,
        "bench_outfield_names": [_name(conn, int(pid)) for pid in policy.get("bench_outfield_order", [])],
        "captain_name": _name(conn, int(policy["captain_id"])) if policy.get("captain_id") is not None else None,
        "vice_captain_name": _name(conn, int(policy["vice_captain_id"])) if policy.get("vice_captain_id") is not None else None,
    }


def _fresh_candidate_universe(conn, context, squad, snapshot) -> dict:
    from fpl_brain import analytics as analytics_module

    pool = cu.load_pool(conn)
    fixtures = cu.load_fixtures_by_team(conn, EVENTS)
    xpts_rows = {EVENT: cu.load_projection_rows(conn, RUNS["xpts"])}
    minutes_rows = {
        EVENT: cu.load_projection_rows(conn, RUNS["minutes"], analytics_module.MINUTES_V1_KIND)
    }
    universe = cu.build_universe(
        pool=pool,
        events_fixtures=fixtures,
        xpts_rows_by_event=xpts_rows,
        minutes_rows_by_event=minutes_rows,
        events=EVENTS,
        owned_ids=squad["squad_ids"],
        price_snapshot=snapshot,
        config=cu.CandidateConfig(top_n_per_criterion=20),
        planning_cutoff=FRESH_CUTOFF,
        run_refs={
            "xpts": {str(EVENT): RUNS["xpts"]},
            "minutes": {str(EVENT): RUNS["minutes"]},
            "events": list(EVENTS),
            "versions": {"minutes": "minutes_v1.5.2", "xpts": "xpts_v1.4.1", "mc": "mc_v1.2.1"},
        },
    )
    universe["scope"] = "FRESH_GW4_ONLY"
    universe["scope_note"] = (
        "Future GW5/GW6 projections were not certified after the fresh refresh; this candidate screen is descriptive GW4-only."
    )
    universe["flags"] = ["FRESH_GW4_ONLY", "FUTURE_PROJECTIONS_UNSUPPORTED", cu.VALUE_LABEL]
    return universe


def _frozen_comparison(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    artifact = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict] = {}
    for finalist in artifact.get("finalists", []):
        transfers = tuple(sorted(
            (int(item["out"]), int(item["in"])) for item in finalist.get("gw4_transfers", [])
        ))
        if transfers == ((329, 332), (334, 305)) and ROUTE_A not in out:
            out[ROUTE_A] = {
                "planning_cutoff": artifact.get("planning_cutoff"),
                "gw4_net_core": finalist.get("gw4_net_core"),
                "supported_3gw_net_core": finalist.get("supported_3gw_net_core"),
                "canonical_signature": finalist.get("canonical_signature"),
            }
        elif transfers == ((1, 28), (329, 305)) and ROUTE_B not in out:
            out[ROUTE_B] = {
                "planning_cutoff": artifact.get("planning_cutoff"),
                "gw4_net_core": finalist.get("gw4_net_core"),
                "supported_3gw_net_core": finalist.get("supported_3gw_net_core"),
                "canonical_signature": finalist.get("canonical_signature"),
            }
        elif transfers == ((329, 305),) and ROUTE_RODON not in out:
            out[ROUTE_RODON] = {
                "planning_cutoff": artifact.get("planning_cutoff"),
                "gw4_net_core": finalist.get("gw4_net_core"),
                "supported_3gw_net_core": finalist.get("supported_3gw_net_core"),
                "canonical_signature": finalist.get("canonical_signature"),
            }
        elif not transfers and ROUTE_ROLL not in out:
            roll = artifact.get("roll_baseline") or {}
            out[ROUTE_ROLL] = {
                "planning_cutoff": artifact.get("planning_cutoff"),
                "gw4_net_core": roll.get("gw4_net"),
                "supported_3gw_net_core": roll.get("supported_3gw_net_core"),
                "canonical_signature": "E4:ROLL;E5:ROLL;E6:ROLL",
            }
    return out


def _rodon_projection(conn) -> dict:
    rows = conn.execute(
        "SELECT fixture_id, payload_json FROM frozen_predictions WHERE projection_run_id=? AND kind=? AND player_id=? ORDER BY fixture_id",
        (RUNS["minutes"], analytics.MINUTES_V1_KIND, 329),
    ).fetchall()
    projections = []
    for row in rows:
        payload = json.loads(row["payload_json"])
        projections.append({
            "fixture_id": int(row["fixture_id"]),
            "joint_availability": payload.get("joint_availability"),
            "expected_minutes": payload.get("expected_minutes"),
            "p_start": payload.get("p_start"),
            "p_60_plus": payload.get("p_60_plus"),
            "availability_source_summary": payload.get("availability_source_summary"),
            "modifier_ids": payload.get("modifier_ids"),
            "risk_flags": payload.get("risk_flags"),
        })
    return {
        "player_id": 329,
        "name": "Rodon",
        "projections": projections,
        "all_expected_minutes_zero": bool(projections) and all(
            float(row.get("expected_minutes") or 0.0) == 0.0 for row in projections
        ),
        "all_joint_availability_zero": bool(projections) and all(
            float(row.get("joint_availability") or 0.0) == 0.0 for row in projections
        ),
    }


def _future_block(horizon: dict) -> dict:
    blocked = [int(event) for event in horizon["blocked_events"]]
    supported = [int(event) for event in horizon["supported_events"]]
    return {
        "decision_events": horizon["decision_events"],
        "decision_horizon_status": horizon["status"],
        "supported_fresh_events": supported,
        "unsupported_fresh_events": blocked,
        "same_cutoff_required": horizon["same_cutoff_required"],
        "events": horizon["events"],
        "unsupported_horizons": ["3GW", "H4", "H6"],
        "numeric_values_reported": False,
        "note": (
            "No future numeric value is carried into the corrected decision. "
            "The earlier GW5 readiness attempts produced no committed projection rows; "
            "their failure is retained as a readiness block only, not as a forecast."
        ),
        "gw5": {
            "status": "READINESS_FAIL_NO_WRITE",
            "attempt_context": "earlier fresh pre-correction diagnostic; not used as decision data",
            "attempts": [
                {
                    "simulations": 2000,
                    "calibration_max_iterations": 250,
                    "reasons": [
                        "INDIVIDUAL_RECONCILIATION_MATERIAL: 19 failures",
                        "SCORER_CALIBRATION_NON_CONVERGENCE: fixture 41 team 4 residual 0.1661590914",
                    ],
                },
                {
                    "simulations": 10000,
                    "calibration_max_iterations": 1000,
                    "reasons": [
                        "INDIVIDUAL_RECONCILIATION_MATERIAL: 38 failures",
                        "SCORER_CALIBRATION_NON_CONVERGENCE: fixture 41 team 4 residual 0.1662911747",
                    ],
                },
            ],
            "stale_runs_not_used": [80, 81, 83, 86, 87],
        },
        "gw6": {
            "status": "NOT_RUN_BLOCKED_BY_GW5",
            "stale_runs_not_used": [88, 89, 91, 94, 95],
        },
        "architecture_changed": False,
        "search_stability_ladders_run": 0,
        "h4_h6_fabricated": False,
    }


def _weak_slot_screen(universe: dict, squad, events) -> dict:
    """Bounded, descriptive Wildcard-trigger evidence (no fabricated points).

    A slot is 'weak' when the owned player's four-GW expected CORE proxy is
    below the position's median legal replacement inside the fresh universe.
    """

    rows = {int(row["player_id"]): row for row in universe["universe"]}
    events = list(events)
    owned = [int(pid) for pid in squad["squad_ids"]]
    by_position: dict[str, list[float]] = {}
    for row in universe["universe"]:
        value = fg.four_gw_expected_core(row, events)
        by_position.setdefault(str(row["position"]), []).append(value)
    medians = {
        position: (sorted(values)[len(values) // 2] if values else 0.0)
        for position, values in by_position.items()
    }
    weak = 0
    availability = 0
    for pid in owned:
        row = rows.get(pid)
        if row is None:
            continue
        value = fg.four_gw_expected_core(row, events)
        if value < medians.get(str(row["position"]), 0.0):
            weak += 1
        if all(fg.event_availability(row, event) == 0.0 for event in events):
            availability += 1
    return {
        "weak_slot_count": weak,
        "position_four_gw_median_core": {position: round(value, 4) for position, value in medians.items()},
        "availability_problems": availability,
    }



def _markdown(report: dict) -> str:
    official = report["official_refresh"]
    rows = report["route_comparison"]["routes"]
    horizon = report["decision_layer"]["horizon"]
    decision = report["decision_layer"]
    h1 = report["h1_descriptive_route"]
    lines = [
        "# GW4 Final Pre-Deadline Operational Refresh",
        "",
        f"- Planning cutoff: `{report['planning_cutoff']}`",
        f"- Official GW4 deadline: `{report['planning_context']['deadline']}`",
        f"- Decision horizon: **{horizon['decision_events']}** (current GW + next three)",
        f"- Decision horizon status: **{horizon['status']}**",
        f"- Fresh support: **{horizon['supported_events']}**; blocked: **{horizon['blocked_events']}**",
        "- Transfers executed: **No**; chips executed: **No**",
        "- Architecture changes: **No**; new search-stability ladders: **No**",
        "",
        "## Outcome",
        "",
        f"**Normal transfer recommendation: SUPPRESSED ({horizon['status']}).** "
        "The rolling four-Gameweek transfer horizon is incomplete, so no transfer route is presented as a recommendation. "
        "A current-GW-only (H1) result is never the best normal transfer recommendation.",
        "",
        f"> {decision.get('operator_summary', '')}",
        "",
        f"Current-GW lineup/captain analysis remains supported (GW{horizon['decision_events'][0]} only) and is labelled lineup-only: "
        f"**{h1['policy']['captain_name']}** (C) / **{h1['policy']['vice_captain_name']}** (V).",
        "",
        "## Blocked horizon events",
        "",
        "| Event | Supported | Reason | Observed cutoff | Missing families |",
        "|---|---|---|---|---|",
    ]
    for event in horizon["decision_events"]:
        record = horizon["events"][str(event)]
        lines.append(
            f"| GW{event} | {'yes' if record['supported'] else 'no'} | {record.get('reason') or '-'} | "
            f"`{record.get('data_cutoff')}` | {', '.join(record.get('missing_families') or []) or '-'} |"
        )
    lines += [
        "",
        "## Full legal action screen (before any route promotion)",
        "",
        f"- Transfer pairs enumerated: {decision['screened_actions']['enumerated_transfer_pairs']} "
        f"(exhaustive: {decision['screened_actions']['enumeration_exhaustive']})",
        f"- Legal single transfers screened: {decision['screened_actions']['screened_legal_actions']}",
        f"- Illegal single transfers: {decision['screened_actions']['illegal_single_transfers']}",
        f"- Bounded promotion pool: {decision['screened_actions']['promotion_pool_size']} "
        f"(limit {decision['screened_actions']['promotion_pool_limit']})",
        "- Every legal action is screened from the fresh CandidateUniverse; a player does not need to be named first.",
        "",
        "## Wildcard trigger screen",
        "",
        f"- Status: **{decision['wildcard_screen']['status']}**",
        f"- Signals: {', '.join(decision['wildcard_screen']['signals']) or 'none'}",
        f"- Full Wildcard optimizer implemented: **No** (no points are fabricated)",
        f"- Same-GW Wildcard/hit rule: **VERIFIED 2026/27** — {decision['wildcard_screen'].get('hit_rule')} "
        f"(net hit for the Gameweek with the chip active: "
        f"{decision['wildcard_screen'].get('same_gameweek_hit_rule', {}).get('net_hit')})",
        "",
        "## Fresh GW4 route comparison (descriptive, lineup-support only)",
        "",
        "| Route | Transfers | Sale proceeds | Purchase cost | Hit | Fresh GW4 net CORE | Frozen GW4 net CORE | Delta | Bank after | FT after | Legal |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for route_id in (ROUTE_A, ROUTE_B, ROUTE_RODON, ROUTE_ROLL):
        row = rows[route_id]
        accounting = row["transfer_accounting"]
        frozen = row.get("frozen_comparison", {}).get("gw4_net_core")
        delta = None if frozen is None else row["gw4_net_core"] - float(frozen)
        hit_points = int(accounting.get("hit_points") or 0)
        hit_label = "0" if hit_points == 0 else f"-{hit_points}"
        lines.append(
            f"| {route_id} | {row['transfer_label']} | {_money_tenths(accounting.get('sale_proceeds_tenths'))} | "
            f"{_money_tenths(accounting.get('purchase_cost_tenths'))} | {hit_label} | "
            f"{row['gw4_net_core']:.4f} | {float(frozen):.4f} | {delta:+.4f} | "
            f"{_money_tenths(accounting.get('bank_after_tenths'))} | {accounting.get('next_free_transfers')} | "
            f"{'PASS' if row['valid'] else 'FAIL'} |"
        )
    ranking = report["route_comparison"].get("ranking") or []
    if ranking:
        lines += [
            "",
            "Descriptive fresh ranking by current-GW net CORE (NOT a transfer recommendation):",
            *[
                f"{row['rank']}. **{row['route_id']}** — {row['gw4_net_core']:.4f}"
                for row in ranking
            ],
        ]
    lines += [
        "",
        "Transfer accounting is from the Phase-7A state engine; expected values are CORE means from one shared fresh GW4 world set. "
        "FT after is the stored free-transfer count entering GW5.",
        "",
        "## Fresh GW4 XI / bench / armbands (lineup only)",
        "",
        f"- Route context: **{h1['route_id']}** (descriptive)",
        f"- XI: {', '.join(h1['policy']['starter_names'])}",
        f"- Bench GK: {h1['policy']['bench_gk_name']}",
        f"- Bench order: {', '.join(h1['policy']['bench_outfield_names'])}",
        f"- Captain: **{h1['policy']['captain_name']}**",
        f"- Vice-captain: **{h1['policy']['vice_captain_name']}**",
        "",
        "## Future-horizon readiness",
        "",
        "GW5 was attempted twice in the earlier pre-correction diagnostic and failed the existing readiness gate on fixture 41 scorer calibration; "
        "both attempts rolled back with no projection rows. GW6 was not run because the contiguous fresh bundle was already blocked. "
        "GW5/GW6 accepted runs exist only at the older cutoff `2026-09-11T10:16:51Z` and are therefore STALE for this decision; GW7 has no runs. "
        "The four-GW transfer horizon is incomplete and the transfer recommendation is suppressed by code.",
        "",
        "## Artifacts",
        "",
        f"- Operational JSON: `{report['artifacts']['operational_json']}`",
        f"- This report: `{report['artifacts']['operational_markdown']}`",
        f"- Fresh candidate screen: `{report['artifacts']['candidate_universe']}`",
        f"- Fresh manager-policy packet: `{report['artifacts']['manager_packet']}`",
        f"- Fresh GW4 world cache: `{report['artifacts']['world_cache']}`",
        f"- Late-news scouting input: `{report['artifacts']['scouting_input']}`",
    ]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    global FRESH_CUTOFF, RUNS, SIMULATIONS
    parser = argparse.ArgumentParser(description="Capture the fresh GW4 operational decision report")
    parser.add_argument("--config")
    parser.add_argument("--out-dir", default="data/exports/operational_refresh/gw04")
    parser.add_argument("--cache-dir", default="data/cache/manager_worlds")
    parser.add_argument("--cutoff", default=FRESH_CUTOFF)
    parser.add_argument("--minutes-run", type=int, default=RUNS["minutes"])
    parser.add_argument("--team-run", type=int, default=RUNS["team"])
    parser.add_argument("--rates-run", type=int, default=RUNS["rates"])
    parser.add_argument("--xpts-run", type=int, default=RUNS["xpts"])
    parser.add_argument("--monte-carlo-run", type=int, default=RUNS["monte_carlo"])
    parser.add_argument("--simulations", type=int, default=SIMULATIONS)
    parser.add_argument("--manager-packet", help="exact manager-policy packet to reference")
    args = parser.parse_args(argv)

    FRESH_CUTOFF = str(args.cutoff)
    RUNS = {
        "minutes": int(args.minutes_run),
        "team": int(args.team_run),
        "rates": int(args.rates_run),
        "xpts": int(args.xpts_run),
        "monte_carlo": int(args.monte_carlo_run),
    }
    SIMULATIONS = int(args.simulations)

    config = load_config(args.config)
    conn = connect_database(config_path(config, "database"))
    entered_guard = None
    try:
        guard_ctx = execution.event_run_guard(
            conn, planning_event=int(EVENT), cutoff=str(args.cutoff),
            label="final_operational_refresh_gw04", families=["operational_refresh"],
        )
        guard = guard_ctx.__enter__()
        entered_guard = guard_ctx
        print(f"execution guard: run_uuid={guard.run_uuid} hard_stop={guard.run().hard_stop_at}")
        entry_id = int(config["fpl_entry_id"])
        context = get_planning_context(
            conn, entry_id, EVENT, as_of=FRESH_CUTOFF, season=config.get("season"),
            official_price_stale_after_hours=config.get("report", {}).get("official_price_stale_after_hours"),
        )
        if context.health["status"] == "FAIL":
            print("operational refresh failed: PlanningContext health FAIL", file=sys.stderr)
            for reason in context.health["fail_reasons"]:
                print(f"  - {reason}", file=sys.stderr)
            return _refuse_execution(guard, 2, "PlanningContext health FAIL")
        squad = manager_worlds.resolve_squad(context, conn)
        if squad["player_count"] != 15:
            print(f"operational refresh failed: expected 15-player squad, got {squad['player_count']}", file=sys.stderr)
            return _refuse_execution(guard, 2, "squad size != 15")

        # --- cutoff / override safety (fail fast, never silently use stale state)
        # Compare against the LATEST recorded override for the event (as-of-now),
        # not the one resolved at this cutoff — otherwise an old cutoff would
        # trivially "cover" the older observation it resolves to.
        latest_override = repo.get_manual_manager_state(conn, entry_id, EVENT) or {}
        cutoff_check = fg.verify_cutoff_covers_override(
            planning_cutoff=FRESH_CUTOFF, override_captured_at=latest_override.get("captured_at"),
        )
        if not cutoff_check["pass"]:
            print(f"operational refresh refused: {cutoff_check['status']}", file=sys.stderr)
            print(f"  {cutoff_check['detail']}", file=sys.stderr)
            print("  establish a planning cutoff at or after the manager-state override and re-run", file=sys.stderr)
            return _refuse_execution(guard, 5, "cutoff precedes manager-state override")

        snapshot = cu.price_snapshot_as_of(
            conn, EVENT, FRESH_CUTOFF,
            required_player_ids=[int(pid) for pid in squad["squad_ids"]],
        )
        initial_state = rc.build_route_state(conn, context, squad, snapshot)
        universe = _fresh_candidate_universe(conn, context, squad, snapshot)
        universe_ids = {int(row["player_id"]) for row in universe["universe"]}
        missing_owned = sorted(int(pid) for pid in squad["squad_ids"] if int(pid) not in universe_ids)
        if missing_owned:
            print(f"operational refresh failed: owned players missing from fresh candidate screen {missing_owned}", file=sys.stderr)
            return _refuse_execution(guard, 2, "owned players missing from candidate screen")
        player_meta = rc.load_player_meta(conn, universe_ids)
        universe["replacement_edges"] = cu.build_replacement_edges(
            universe_rows=universe["universe"], owned_ids=squad["squad_ids"], state=initial_state,
            price_snapshot=snapshot, player_meta=player_meta,
        )
        universe["replacement_edge_count"] = len(universe["replacement_edges"])
        universe["legal_single_transfer_count"] = sum(
            1 for edge in universe["replacement_edges"] if edge["currently_legal_single_transfer"]
        )
        universe["illegal_single_transfer_count"] = len(universe["replacement_edges"]) - universe["legal_single_transfer_count"]
        universe["squad_coverage"] = {"owned": len(squad["squad_ids"]), "present": len(squad["squad_ids"]) - len(missing_owned)}

        # --- canonical four-GW decision horizon + hard readiness gate -----------
        season_last_event = fg.season_last_event_from_db(conn)
        horizon_support = fg.event_support_from_db(conn, DECISION_EVENTS, FRESH_CUTOFF)
        horizon = fg.evaluate_horizon(
            planning_event=EVENT, support_by_event=horizon_support, cutoff=FRESH_CUTOFF,
            last_event=season_last_event,
        )

        # --- screen EVERY legal single transfer before any route promotion ------
        action_screen = fg.screen_legal_actions(
            universe_rows=universe["universe"],
            replacement_edges=universe["replacement_edges"],
            owned_ids=squad["squad_ids"],
            decision_events_window=DECISION_EVENTS,
        )

        weak = _weak_slot_screen(universe, squad, horizon["decision_events"])
        # Squad-change units, NOT the size of the legal action space.  The
        # candidate-edge count is reported separately and never used as the
        # number of desired moves.
        desired = action_screen["desired_squad_changes"]
        event_start_ft = (context.manager_state or {}).get("event_start_free_transfers")
        remaining_ft = (context.manager_state or {}).get("free_transfers")
        transfers_already_executed = sum(
            1 for row in repo.active_manager_acquisitions(conn, int(config["fpl_entry_id"]))
            if int(row.get("acquired_event") or 0) == EVENT
        )
        prior_paid_transfers = (
            None if event_start_ft is None
            else max(0, transfers_already_executed - int(event_start_ft))
        )
        wildcard_screen = fg.wildcard_trigger_screen(
            weak_slot_count=weak["weak_slot_count"],
            availability_problems=weak["availability_problems"],
            desired_transfer_count=desired["desired_transfer_count"],
            prior_paid_transfers=int(prior_paid_transfers or 0),
            event_start_free_transfers=event_start_ft,
            free_transfers_available=remaining_ft,
        )
        wildcard_screen["input_audit"] = {
            "desired_transfer_count_source": desired["desired_transfer_source"],
            "desired_transfer_count": desired["desired_transfer_count"],
            "legal_action_space_size_not_used": action_screen["legal_single_transfers"],
            "transfers_already_executed_this_event": transfers_already_executed,
            "event_start_free_transfers": event_start_ft,
            "free_transfers_remaining": remaining_ft,
            "prior_paid_transfers": prior_paid_transfers,
        }

        routes = [
            _route(ROUTE_A, "Rodon → Justin + Muharemović → Davis", [(329, 332), (334, 305)]),
            _route(ROUTE_B, "Raya → Martinez + Rodon → Davis", [(1, 28), (329, 305)]),
            _route(ROUTE_RODON, "Rodon → Davis only", [(329, 305)]),
            _route(ROUTE_ROLL, "ROLL", []),
        ]
        union_ids = {int(pid) for pid in squad["squad_ids"]}
        for route in routes:
            for step in route.steps:
                union_ids.update(int(action.out_player_id) for action in step.transfer_batch.actions)
                union_ids.update(int(action.in_player_id) for action in step.transfer_batch.actions)
        player_meta = rc.load_player_meta(conn, union_ids)
        missing_meta = sorted(int(pid) for pid in union_ids if int(pid) not in player_meta)
        if missing_meta:
            print(f"operational refresh failed: missing player metadata {missing_meta}", file=sys.stderr)
            return _refuse_execution(guard, 2, "missing player metadata")

        scenario = rc.flat_current_price_scenario(snapshot, EVENTS)
        bundle = rc.EventBundle(
            event=EVENT, minutes_run_id=RUNS["minutes"], team_run_id=RUNS["team"], rate_run_id=RUNS["rates"],
            xpts_run_id=RUNS["xpts"], mc_run_id=RUNS["monte_carlo"], simulations=SIMULATIONS,
            seed=SEED, planning_cutoff=FRESH_CUTOFF,
        )
        world_config = ro.OptimizerConfig(events=EVENTS, search_draws=SIMULATIONS, seed=SEED, policy_selection_worlds=0)
        cache_dir = Path(args.cache_dir)
        matrix, world_info = ro.build_event_worlds(
            conn, {EVENT: bundle}, EVENT, sorted(union_ids), world_config, cache_dir=cache_dir,
        )
        comparison = rc.compare_routes(
            conn=conn, bundles={EVENT: bundle}, routes=routes, initial_state=initial_state,
            scenario=scenario, player_meta=player_meta,
            # The matrix was fetched by the certified loader above, so it is
            # presented under an explicit NON-PRODUCTION declaration: this is a
            # historical replay, and no certification artifact exists for it.
            non_production_worlds=ro.NonProductionWorlds(
                declaration="scripts/final_operational_refresh_gw04.py: historical GW4 replay, "
                            "world loaded from a hand-assembled bundle",
                matrices={EVENT: matrix},
            ),
            simulations=SIMULATIONS, seed=SEED, planning_cutoff=FRESH_CUTOFF,
        )
        comparison["fresh_world_info"] = {**world_info, "worlds": int(matrix["worlds"]), "union_players": len(union_ids)}
        comparison["route_search_run"] = False
        comparison["search_stability_ladders_run"] = 0

        frozen = _frozen_comparison(config_path(config, "exports_dir") / "live_fire" / "gw04" / "live_fire_20260911.json")
        route_rows: dict[str, dict] = {}
        for route in routes:
            body = comparison["routes"][route.route_id]
            record = body["events"][0] if body.get("events") else {}
            h1 = body["horizons"].get("H1") or {}
            transfers = record.get("selected_policy")
            transfer_actions = route.steps[0].transfer_batch.actions
            state_transition = ts.apply_transfer_batch(
                initial_state, route.steps[0].transfer_batch, snapshot, player_meta
            )
            transfer_label = "ROLL" if not transfer_actions else "; ".join(
                f"{_name(conn, int(action.out_player_id))} → {_name(conn, int(action.in_player_id))}"
                for action in transfer_actions
            )
            route_rows[route.route_id] = {
                **body,
                "label": route.label,
                "gw4_net_core": float(h1.get("net_core")),
                "gw4_gross_core": float(record.get("mean_gross_core")),
                "gw4_hit_points": int(record.get("hit_points", 0)),
                "transfer_label": transfer_label,
                "transfer_accounting": {
                    "bank_before_tenths": int(state_transition.bank_before_tenths),
                    "sale_proceeds_tenths": int(state_transition.sale_proceeds_tenths),
                    "purchase_cost_tenths": int(state_transition.purchase_cost_tenths),
                    "bank_after_tenths": int(state_transition.bank_after_tenths),
                    "free_transfers_before": int(state_transition.ft_before),
                    "free_transfers_used": int(state_transition.ft_used),
                    "paid_transfers": int(state_transition.paid_transfers),
                    "next_free_transfers": int(state_transition.next_event_state.free_transfers)
                    if state_transition.next_event_state is not None else None,
                    "next_bank_tenths": int(state_transition.next_event_state.bank_tenths)
                    if state_transition.next_event_state is not None else None,
                    "hit_points": int(state_transition.hit_points),
                    "valid": bool(body.get("valid")) and bool(state_transition.ok),
                    "state_engine_errors": list(state_transition.errors),
                    "errors": list(body.get("errors") or []) + list(state_transition.errors),
                },
                "policy": _policy_names(conn, record.get("selected_policy")),
                "frozen_comparison": frozen.get(route.route_id),
            }
        preferred_id = max(route_rows, key=lambda route_id: route_rows[route_id]["gw4_net_core"])
        ranking = [
            {
                "rank": index,
                "route_id": route_id,
                "label": route_rows[route_id]["label"],
                "gw4_net_core": route_rows[route_id]["gw4_net_core"],
                "hit_points": route_rows[route_id]["transfer_accounting"]["hit_points"],
                "bank_after_tenths": route_rows[route_id]["transfer_accounting"]["bank_after_tenths"],
                "next_free_transfers": route_rows[route_id]["transfer_accounting"]["next_free_transfers"],
                "valid": route_rows[route_id]["valid"],
            }
            for index, route_id in enumerate(
                sorted(route_rows, key=lambda route_id: route_rows[route_id]["gw4_net_core"], reverse=True),
                start=1,
            )
        ]

        # --- the decision under the product rule -------------------------------
        # Current-GW lineup support comes from the fresh GW4 world set; the
        # normal transfer block is emitted only if the four-GW horizon is complete.
        gw4_supported = bool((horizon["events"].get(str(EVENT)) or {}).get("supported"))
        lineup_payload = None
        if gw4_supported:
            lineup_payload = {
                "status": fg.LINEUP_ONLY,
                "policy": route_rows[preferred_id]["policy"],
                "policy_route_id": preferred_id,
                "source": "fresh GW4 world set, current-GW manager policy",
            }
        decision = fg.evaluate_four_gw_decision(
            planning_event=EVENT,
            support_by_event=horizon_support,
            cutoff=FRESH_CUTOFF,
            last_event=season_last_event,
            screened_actions=action_screen,
            routes=None,
            lineup=lineup_payload,
            wildcard=wildcard_screen,
        )

        projection_runs = []
        for run_id in RUNS.values():
            row = conn.execute(
                "SELECT id, model_family, model_version, planning_event, data_cutoff, deadline_status, status, generated_at "
                "FROM projection_runs WHERE id=?", (int(run_id),)
            ).fetchone()
            if row is not None:
                projection_runs.append(dict(row))

        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        candidate_path = out_dir / "fresh_candidate_universe_gw04.json"
        candidate_path.write_text(json.dumps(cu.jsonable(universe), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        manager_packet = (
            Path(args.manager_packet)
            if args.manager_packet
            else out_dir / "gw04" / "manager_lineup_packet.json"
        )
        manager_packet_md = manager_packet.with_suffix(".md")
        world_cache_path = cache_dir / f"{world_info.get('key')}.json" if world_info.get("key") else None

        report = {
            "schema": "fpl_brain.final_operational_refresh.v1",
            "phase": "GW4_FINAL_PRE_DEADLINE_OPERATIONAL_REFRESH",
            "runner_scope": "ARCHIVAL_GW4_H1_DESCRIPTIVE_DIAGNOSTIC_NOT_A_4GW_RUNNER",
            "production_runner_note": (
                "the rolling 4GW (GW4-GW7) production decision must use the multi-event optimizer / "
                "route-comparator path feeding fpl_brain.four_gw_decision.evaluate_four_gw_decision"
            ),
            "planning_cutoff": FRESH_CUTOFF,
            "planning_context": context.to_dict(),
            "corrected_manager_state": {
                "free_transfers_before": int(initial_state.free_transfers),
                "bank_before_tenths": int(initial_state.bank_tenths),
                "bank_before": f"£{initial_state.bank_tenths / 10:.1f}m",
                "executed_transfer_already_in_squad": "Maguire → De Cuyper",
                "de_cuyper_player_id": 115,
                "maguire_player_id": 418,
                "de_cuyper_in_active_squad": 115 in set(int(pid) for pid in squad["squad_ids"]),
                "maguire_in_active_squad": 418 in set(int(pid) for pid in squad["squad_ids"]),
                "source": context.manager_state.get("manual", {}).get("source"),
                "captured_at": context.manager_state.get("manual", {}).get("captured_at"),
                "prior_two_ft_artifacts_excluded": True,
                "preexisting_transfer_hit_recharged": False,
            },
            "official_refresh": {
                "fetch_run": context.official_runs.get("fetch"),
                "manager_sync": context.official_runs.get("manager_sync"),
                "event_data_state": context.event_data_state,
                "price_snapshot_id": snapshot.identity(),
                "price_freshness": context.official_price_freshness,
                "projection_runs": projection_runs,
            },
            "late_news_input": {
                "source": "user-provided video transcript",
                "import_path": "data/scouting/inbox/gw04_final_refresh_20260911.json",
                "notes_imported": 8,
                "players_resolved": 7,
                "operational_hard_signal": "Rodon availability=unavailable",
                "qualitative_context": [
                    "Justin viable short-term Leeds option",
                    "Muharemović safer longer-term Leeds defender",
                    "Mitoma out until after the international break; De Cuyper advanced-role context",
                    "DCL considered a good starter this week",
                    "Groß strong fixture; benching merits scrutiny",
                ],
            },
            "rodon_signal": _rodon_projection(conn),
            "candidate_universe": {
                "artifact": str(candidate_path),
                "scope": universe["scope"],
                "universe_count": universe["universe_count"],
                "search_view_count": universe["search_view_count"],
                "replacement_edge_count": universe["replacement_edge_count"],
                "legal_single_transfer_count": universe["legal_single_transfer_count"],
                "illegal_single_transfer_count": universe["illegal_single_transfer_count"],
                "completeness_audit": universe["completeness_audit"],
                "no_recommendation": True,
            },
            "route_comparison": {
                **{key: value for key, value in comparison.items() if key != "routes"},
                "ranking": ranking,
                "routes": route_rows,
                "scope_note": (
                    "Descriptive current-GW (H1) analysis only. This comparison runs over the one "
                    "supported event and is NOT a transfer recommendation."
                ),
            },
            "h1_descriptive_route": {
                "route_id": preferred_id,
                "label": route_rows[preferred_id]["label"],
                "gw4_net_core": route_rows[preferred_id]["gw4_net_core"],
                "gw4_gross_core": route_rows[preferred_id]["gw4_gross_core"],
                "hit_points": route_rows[preferred_id]["gw4_hit_points"],
                "policy": route_rows[preferred_id]["policy"],
                "scope": "CURRENT_GW_ONLY_DESCRIPTIVE_NOT_A_TRANSFER_RECOMMENDATION",
            },
            "preferred_route": None,
            "preferred_route_suppressed_reason": fg.DECISION_HORIZON_INCOMPLETE,
            "future_horizon": _future_block(horizon),
            "decision_layer": {
                "horizon": horizon,
                "operator_summary": decision["operator_summary"],
                "screened_actions": {
                    key: value for key, value in action_screen.items() if key != "promotion_pool"
                },
                "wildcard_screen": wildcard_screen,
                "transfer_decision": decision["transfer_decision"],
                "transfer_recommendation": decision["transfer_recommendation"],
                "lineup_decision": decision["lineup_decision"],
                "lineup_recommendation": decision["lineup_recommendation"],
                "decision_board": decision["decision_board"],
                "product_rule": "normal_transfers_are_four_gameweek_decisions",
            },
            "decision_status": {
                "decision_horizon_status": horizon["status"],
                "transfer_recommendation_status": decision["transfer_recommendation"]["status"],
                "lineup_recommendation_status": decision["lineup_decision"]["status"],
                "rodon_out_dominant_signal": None,
                "preferred_route_changes_from_frozen": None,
                "no_transfer_execution": True,
                "no_chip_execution": True,
                "architecture_changed": False,
            },
            "artifacts": {
                "operational_json": str(out_dir / "final_operational_refresh_gw04.json"),
                "operational_markdown": str(out_dir / "final_operational_refresh_gw04.md"),
                "candidate_universe": str(candidate_path),
                "manager_packet": str(manager_packet),
                "manager_packet_markdown": str(manager_packet_md),
                "world_cache": None if world_cache_path is None else str(world_cache_path),
                "scouting_input": "data/scouting/inbox/gw04_final_refresh_20260911.json",
            },
            "no_recommendation": not fg.transfer_recommendation_allowed(horizon),
            "no_execution": True,
        }
        # Fill derived decision flags after the complete fresh route table exists.
        report["decision_status"]["rodon_out_dominant_signal"] = bool(
            report["rodon_signal"]["all_expected_minutes_zero"]
            and report["rodon_signal"]["all_joint_availability_zero"]
        )
        frozen_h1_preferred = max(
            frozen,
            key=lambda route_id: float(frozen[route_id].get("gw4_net_core") or float("-inf")),
        ) if frozen else None
        report["decision_status"]["preferred_route_changes_from_frozen"] = (
            None if frozen_h1_preferred is None else preferred_id != frozen_h1_preferred
        )
        report["h1_descriptive_route"]["frozen_h1_preferred_route"] = frozen_h1_preferred
        report["h1_descriptive_route"]["route_changed_from_frozen_leaders"] = (
            None if frozen_h1_preferred is None else preferred_id != frozen_h1_preferred
        )

        json_path = out_dir / "final_operational_refresh_gw04.json"
        md_path = out_dir / "final_operational_refresh_gw04.md"
        report["artifacts"]["operational_json"] = str(json_path)
        report["artifacts"]["operational_markdown"] = str(md_path)
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        md_path.write_text(_markdown(report), encoding="utf-8")

        print(f"operational refresh complete: cutoff={FRESH_CUTOFF} "
              f"horizon={horizon['decision_events']} status={horizon['status']}")
        print(f"  transfer_recommendation={decision['transfer_recommendation']['status']} "
              f"blocked={horizon['blocked_events']}")
        print(f"  screened legal single transfers={action_screen['screened_legal_actions']} "
              f"exhaustive={action_screen['enumeration_exhaustive']} pool={action_screen['promotion_pool_size']}")
        print(f"  wildcard={wildcard_screen['status']} signals={wildcard_screen['signals']}")
        print(f"  lineup_status={decision['lineup_decision']['status']} "
              f"h1_descriptive={preferred_id} net={route_rows[preferred_id]['gw4_net_core']:.4f}")
        for route_id in (ROUTE_A, ROUTE_B, ROUTE_RODON, ROUTE_ROLL):
            row = route_rows[route_id]
            print(f"  {route_id}: valid={row['valid']} GW4={row['gw4_net_core']:.4f} bank={row['transfer_accounting']['bank_after_tenths']} FT={row['transfer_accounting']['next_free_transfers']}")
        print(f"  json={json_path}")
        print(f"  markdown={md_path}")
        return 0
    finally:
        if entered_guard is not None:
            entered_guard.__exit__(*sys.exc_info())
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
