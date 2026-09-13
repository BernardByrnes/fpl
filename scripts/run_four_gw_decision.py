#!/usr/bin/env python3
"""Production rolling four-Gameweek (GW4-GW7) decision runner.

Feeds GW4, GW5, GW6, GW7 into the accepted multi-event optimizer path and then
into ``fpl_brain.four_gw_decision.evaluate_four_gw_decision``.

Stages
------
``bundle``  verify coherent accepted predictive support for every decision event
            at ONE cutoff, and write the four-event bundle artifact.
``search``  run the accepted bounded multi-event route search (ROLL + non-chip
            routes), adapt the results, and emit the four-GW decision board.
``all``     bundle then search (search refuses if the horizon is incomplete).

Read/write only: no transfer or chip is executed, no readiness gate is weakened,
and no search-stability ladder is re-run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fpl_brain import analytics, candidate_universe as cu, decision_confidence as dc
from fpl_brain import four_gw_decision as fg
from fpl_brain import certified_bundle
from fpl_brain import manager_worlds, route_optimizer as ro, transfer_state as ts
from fpl_brain import execution
from fpl_brain import ingest_provenance as provenance
from fpl_brain import repositories as repo
from fpl_brain.config import config_path, load_config
from fpl_brain.database import connect_database
from fpl_brain.planning import get_planning_context
from fpl_brain.utils import parse_utc, utc_now


def _refuse_execution(guard, code: int, reason: str) -> int:
    """Record a deliberate refusal as a FAILED execution run, then return the code."""

    guard.finish(execution.RUN_FAILED, reason)
    return code

SEED = 20260911
#: Stage-1 (screening) draw count.  This is the count ACTUALLY used to build the
#: shared per-event world matrices for every event, and it is what the artifact
#: reports.  The two-stage finalist refinement belongs to R4B.2; until then no
#: artifact may advertise a higher per-event fidelity than was run.
STAGE1_DRAWS = 2_000
DIAG_PREDICTIVE_GENERATION_MISMATCH = "PREDICTIVE_GENERATION_MISMATCH"
DIAG_DECISION_EVENT_MISMATCH = "DECISION_EVENT_MISMATCH"
FAMILY_KINDS = {
    "minutes_v1": analytics.MINUTES_V1_KIND,
    "xpts_v1": None,
}
FAMILY_TO_BUNDLE = {
    "minutes_v1": "minutes_run_id",
    "team_strength_v1": "team_run_id",
    "player_rates_v1": "rate_run_id",
    "xpts_v1": "xpts_run_id",
    "monte_carlo_v1": "mc_run_id",
}


def _money(tenths) -> str:
    return "n/a" if tenths is None else f"£{int(tenths) / 10:.1f}m"


def bundle_for(event: int, runs: dict, cutoff: str, draws: int):
    from fpl_brain import route_comparator as rc

    return rc.EventBundle(
        event=int(event),
        minutes_run_id=int(runs["minutes_v1"]),
        team_run_id=int(runs["team_strength_v1"]),
        rate_run_id=int(runs["player_rates_v1"]),
        xpts_run_id=int(runs["xpts_v1"]),
        mc_run_id=int(runs["monte_carlo_v1"]),
        simulations=int(draws), seed=SEED, planning_cutoff=cutoff,
    )


def draws_for(event: int, decision_events) -> int:
    """Stage-1 draws for every event.  Identical for all events: one shared
    draw count is what actually runs, so the artifact must not claim otherwise.
    """

    return STAGE1_DRAWS


def _certification_events(path) -> list[int] | None:
    """Read the certified decision events from an artifact WITHOUT validating it.

    Used only to resolve the authoritative planning event before the execution
    guard exists.  Full validation still happens through
    ``four_gw_decision.load_certification_artifact`` on the decision path.
    """

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None
    events = payload.get("events") or []
    try:
        resolved = [int(event) for event in events]
    except (TypeError, ValueError):
        return None
    return resolved or None


def _certified_role_evidence(conn, certified_runs, player_ids, decision_events, certification):
    """Role evidence from the CERTIFIED minutes runs ONLY (R4B.2c).

    Reads the frozen MINUTES_V1 payloads of the exact minutes run ids named by the
    certification bundles.  It never reads the latest minutes run, live scouting, or
    any post-certification role evidence.
    """

    minutes_run_ids = sorted(
        {
            int(certified_runs[int(event)]["minutes_v1"])
            for event in decision_events
            if certified_runs.get(int(event), {}).get("minutes_v1") is not None
        }
    )
    evidence: dict[int, dict] = {}
    for run_id in minutes_run_ids:
        for record in analytics.frozen_predictions(conn, run_id, [analytics.MINUTES_V1_KIND]):
            pid = int(record["player_id"])
            if pid not in set(int(p) for p in player_ids):
                continue
            block = (record.get("payload") or {}).get("role_evidence")
            if isinstance(block, dict):
                evidence[pid] = block
    source = {
        "certification_identity": certification.get("four_gw_certification_identity"),
        "minutes_run_ids": minutes_run_ids,
        "player_ids": sorted(int(p) for p in player_ids),
        "provenance": "certified_minutes_run_only",
    }
    return evidence, source


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Rolling four-GW production decision runner")
    parser.add_argument("--stage", choices=("bundle", "search", "all"), default="all")
    parser.add_argument("--config")
    parser.add_argument(
        "--event",
        type=int,
        default=None,
        help="planning Gameweek. Omit to derive it from the certification artifact (authoritative); "
             "if supplied it must match the certification's first event",
    )
    parser.add_argument("--cutoff", required=True)
    parser.add_argument("--cache-dir", default="data/cache/manager_worlds")
    parser.add_argument("--out-dir", default="data/exports/four_gw")
    parser.add_argument("--beam", type=int, default=8)
    parser.add_argument("--exact-budget", type=int, default=20)
    parser.add_argument("--search-n", type=int, default=12)
    parser.add_argument("--singles-per-out", type=int, default=4)
    parser.add_argument("--max-transfers-per-event", type=int, default=2)
    parser.add_argument(
        "--certification",
        help="path to the authoritative certification artifact (REQUIRED for --stage search|all)",
    )
    args = parser.parse_args(argv)

    started = time.time()
    config = load_config(args.config)

    # --- Event safety -------------------------------------------------------
    # The certification artifact is AUTHORITATIVE for the planning event.  There
    # is deliberately no GW4 default: omitting --event without a certification is
    # an error, and a supplied --event that contradicts the certification is a
    # hard stop (DECISION_EVENT_MISMATCH).
    cert_events = _certification_events(args.certification) if args.certification else None
    if args.event is None:
        if not cert_events:
            print(
                "decision refused: --event is required when no readable --certification artifact is "
                "supplied (there is no default Gameweek)",
                file=sys.stderr,
            )
            return 2
        planning_event = int(cert_events[0])
    else:
        planning_event = int(args.event)
        if cert_events and int(cert_events[0]) != planning_event:
            print(
                f"decision refused: {DIAG_DECISION_EVENT_MISMATCH}: --event {planning_event} does not "
                f"match the certification's first event {int(cert_events[0])}; the certification is "
                "authoritative",
                file=sys.stderr,
            )
            return 2

    conn = connect_database(config_path(config, "database"))
    entered_guard = None
    try:
        guard_ctx = execution.event_run_guard(
            conn, planning_event=planning_event, cutoff=str(args.cutoff),
            label="run_four_gw_decision", families=["four_gw_decision"],
        )
        guard = guard_ctx.__enter__()
        entered_guard = guard_ctx
        print(f"execution guard: run_uuid={guard.run_uuid} hard_stop={guard.run().hard_stop_at}")
        entry_id = int(config["fpl_entry_id"])
        cutoff = str(args.cutoff)
        deadline_of = {
            int(row[0]): row[1] for row in conn.execute("SELECT id, deadline_time FROM events")
        }
        last_event = fg.season_last_event_from_db(conn)
        decision_events = fg.decision_events(planning_event, last_event=last_event)
        context = get_planning_context(conn, entry_id, planning_event, as_of=cutoff, season=config.get("season"))
        override = fg.verify_cutoff_covers_override(
            planning_cutoff=cutoff,
            override_captured_at=(context.manager_state or {}).get("override", {}).get("captured_at"),
        )
        squad = manager_worlds.resolve_squad(context, conn)
        initial_state = __import__("fpl_brain.route_comparator", fromlist=["x"]).build_route_state(conn, context, squad)

        # --- Stage A: coherence of the four-event predictive bundle -----------
        # NON-PRODUCTION readiness display only.  This legacy path picks the newest
        # same-cutoff run per family and validates no dependency edges, so it is
        # used for the readiness PRINT and never for the production decision below.
        support = fg.event_support_from_db(conn, decision_events, cutoff)
        print("readiness support source: NON-PRODUCTION latest-per-family rediscovery")
        horizon = fg.evaluate_horizon(planning_event=planning_event, support_by_event=support,
                                      cutoff=cutoff, last_event=last_event)
        bundle_artifact = {
            "schema": "fpl_brain.four_gw_bundle.v1",
            "planning_event": planning_event,
            "planning_cutoff": cutoff,
            "decision_events": list(decision_events),
            "deadlines": {str(event): deadline_of.get(int(event)) for event in decision_events},
            "cutoff_guard": override,
            "manager_state": {
                "free_transfers": (context.manager_state or {}).get("free_transfers"),
                "event_start_free_transfers": (context.manager_state or {}).get("event_start_free_transfers"),
                "bank_tenths": (context.manager_state or {}).get("bank"),
                "authoritative_source": (context.manager_state or {}).get("authoritative_source"),
                "health": context.health.get("status"),
            },
            "horizon": horizon,
            "draw_fidelity": {str(event): draws_for(event, decision_events) for event in decision_events},
            "families_required": list(fg.REQUIRED_HORIZON_FAMILIES),
            "no_execution": True,
            "no_recommendation": True,
        }
        out_dir = Path(args.out_dir) / f"gw{planning_event:02d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "four_gw_bundle.json").write_text(
            json.dumps(bundle_artifact, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        print(f"STAGE A bundle: cutoff={cutoff} events={list(decision_events)} status={horizon['status']}")
        for event in decision_events:
            record = horizon["events"][str(event)]
            print(f"  GW{event}: supported={record['supported']} runs={support[int(event)]['matched_runs']} "
                  f"missing={support[int(event)]['missing_families']} "
                  f"stale={support[int(event)]['stale_families']}")
        if not horizon["complete"]:
            print(f"STAGE A BLOCKED: {horizon['status']} blocked_events={horizon['blocked_events']}")
            print(f"  artifact={out_dir / 'four_gw_bundle.json'}")
            return _refuse_execution(guard, 4, "decision horizon incomplete")
        if args.stage == "bundle":
            print(f"  readiness complete; artifact={out_dir / 'four_gw_bundle.json'}")
            return 0

        # --- Production decision gate: a certified bundle is MANDATORY ----------
        # No fallback to latest-per-family discovery.  Without an authoritative
        # certification artifact this refuses with DECISION_CERTIFICATION_REQUIRED.
        if not args.certification:
            print(
                "decision refused: DECISION_CERTIFICATION_REQUIRED: --certification <artifact> is "
                "required for --stage search|all; a production decision must consume certified run ids",
                file=sys.stderr,
            )
            return _refuse_execution(guard, 6, "DECISION_CERTIFICATION_REQUIRED: no artifact supplied")
        try:
            certification = fg.load_certification_artifact(args.certification)
            certified_support = fg.event_support_from_certification(
                conn, certification, events=decision_events, cutoff=cutoff
            )
        except fg.DecisionCertificationRequired as failure:
            print(f"decision refused: {failure}", file=sys.stderr)
            return _refuse_execution(guard, 6, str(failure))
        except certified_bundle.BundleIncoherent as failure:
            print(f"decision refused: {failure}", file=sys.stderr)
            return _refuse_execution(guard, 6, str(failure))
        print(
            f"certified bundles accepted: snapshot={str(certification.get('data_snapshot_sha256'))[:16]}… "
            f"identity={str(certification.get('four_gw_certification_identity'))[:16]}…"
        )
        for event in decision_events:
            record = certified_support[int(event)]
            print(f"  GW{event}: certified runs={record['matched_runs']} "
                  f"bundle={record['bundle_identity'][:16]}…")

        # --- ONE certified generation, everywhere ------------------------------
        # Discovery/screening, exact evaluation and the decision board must all
        # describe the SAME certified generation.  The readiness-only `support`
        # above is a NON-PRODUCTION rediscovery and is never used for the
        # decision.  The equivalence of the two generations actually consumed is
        # asserted below, once the bundles exist.
        certified_runs = {
            int(event): certified_support[int(event)]["matched_runs"] for event in decision_events
        }
        certified_horizon = fg.evaluate_horizon(
            planning_event=planning_event, support_by_event=certified_support,
            cutoff=cutoff, last_event=last_event,
        )
        if not certified_horizon["complete"]:
            print(
                f"decision refused: {certified_horizon['status']}: the CERTIFIED horizon is incomplete "
                f"blocked_events={certified_horizon['blocked_events']}",
                file=sys.stderr,
            )
            return _refuse_execution(guard, 6, "certified decision horizon incomplete")
        search_provenance = {
            "planning_cutoff": cutoff,
            "planning_context_hash": analytics.canonical_hash({
                "entry_id": entry_id,
                "planning_event": planning_event,
                "cutoff": cutoff,
                "manager_state": (context.manager_state or {}),
            }),
            "four_gw_certification_identity": certification.get("four_gw_certification_identity"),
            "data_snapshot_sha256": certification.get("data_snapshot_sha256"),
            "decision_events": list(decision_events),
            "certified_runs_by_event": {str(e): dict(certified_runs[int(e)]) for e in decision_events},
        }
        # The discovery/exact generation identities are added below, once the
        # bundles exist and both generations can be compared.

        # --- Stage B: bounded non-chip route search ---------------------------
        import fpl_brain.route_comparator as rc

        # SOURCE vs PREDICTION split.  All causal source state comes from the
        # certification's immutable snapshot; predictive rows come from the live
        # prediction DB using ONLY the certified exact run ids.
        source_conn = fg.open_certification_source(certification)
        try:
            source_context = get_planning_context(
                source_conn, entry_id, planning_event, as_of=cutoff, season=config.get("season")
            )
            source_squad = manager_worlds.resolve_squad(source_context, source_conn)
            source_state = rc.build_route_state(source_conn, source_context, source_squad)
        except Exception:
            source_conn.close()
            raise
        if [int(pid) for pid in source_squad["squad_ids"]] != [int(pid) for pid in squad["squad_ids"]]:
            print(
                "decision refused: the certification snapshot's squad differs from the live squad; "
                "the snapshot is the causal source for this decision",
                file=sys.stderr,
            )
            source_conn.close()
            return _refuse_execution(guard, 6, "snapshot squad differs from live squad")

        bundles = {
            int(event): bundle_for(
                event, certified_runs[int(event)], cutoff, draws_for(event, decision_events)
            )
            for event in decision_events
        }
        snapshot = cu.price_snapshot_as_of(
            source_conn, planning_event, cutoff,
            required_player_ids=[int(pid) for pid in source_squad["squad_ids"]],
        )
        scenario = rc.flat_current_price_scenario(snapshot, decision_events)
        pool = cu.load_pool(source_conn)
        # --- Certified official-pool identity (R4B.1.1) ------------------------
        # The snapshot's active pool must BE the latest accepted official
        # bootstrap generation, by exact ID identity.  This reads ONLY the
        # immutable snapshot (never live JSON, never the live DB) and refuses
        # before candidate promotion, route generation or optimisation.
        try:
            pool_identity = provenance.assert_official_pool_identity(
                accepted_generation=repo.latest_accepted_bootstrap_generation(source_conn),
                snapshot_player_ids=repo.active_player_ids(source_conn),
                enforce=True,
            )
        except provenance.OfficialPoolIncomplete as failure:
            print(f"decision refused: {failure}", file=sys.stderr)
            source_conn.close()
            return _refuse_execution(guard, 6, str(failure))
        print(
            f"official pool identity: generation={pool_identity['official_generation_id']} "
            f"count={pool_identity['snapshot_pool_count']} "
            f"sha={str(pool_identity['snapshot_pool_ids_sha256'])[:16]}… "
            f"match={pool_identity['official_pool_identity_match']}"
        )
        fixtures = cu.load_fixtures_by_team(source_conn, decision_events)
        xpts_rows = {int(e): cu.load_projection_rows(conn, int(certified_runs[int(e)]["xpts_v1"]))
                     for e in decision_events}
        minutes_rows = {int(e): cu.load_projection_rows(
            conn, int(certified_runs[int(e)]["minutes_v1"]), FAMILY_KINDS["minutes_v1"])
            for e in decision_events}
        # The generation actually consumed by DISCOVERY/SCREENING.
        discovery_runs = {int(e): dict(certified_runs[int(e)]) for e in decision_events}
        # The generation actually consumed by EXACT EVALUATION (the bundles).
        exact_runs = {
            int(e): {
                "minutes_v1": int(bundles[int(e)].minutes_run_id),
                "team_strength_v1": int(bundles[int(e)].team_run_id),
                "player_rates_v1": int(bundles[int(e)].rate_run_id),
                "xpts_v1": int(bundles[int(e)].xpts_run_id),
                "monte_carlo_v1": int(bundles[int(e)].mc_run_id),
            }
            for e in decision_events
        }
        discovery_identity = analytics.canonical_hash(
            {"cutoff": cutoff, "runs": {str(e): discovery_runs[int(e)] for e in decision_events}}
        )
        exact_identity = analytics.canonical_hash(
            {"cutoff": cutoff, "runs": {str(e): exact_runs[int(e)] for e in decision_events}}
        )
        if discovery_identity != exact_identity:
            print(
                f"decision refused: {DIAG_PREDICTIVE_GENERATION_MISMATCH}: discovery generation "
                f"{discovery_identity} != exact-evaluation generation {exact_identity}",
                file=sys.stderr,
            )
            source_conn.close()
            return _refuse_execution(guard, 6, DIAG_PREDICTIVE_GENERATION_MISMATCH)
        universe = cu.build_universe(
            pool=pool, events_fixtures=fixtures, xpts_rows_by_event=xpts_rows,
            minutes_rows_by_event=minutes_rows, events=list(decision_events),
            owned_ids=source_squad["squad_ids"], price_snapshot=snapshot,
            config=cu.CandidateConfig(top_n_per_criterion=20), planning_cutoff=cutoff,
            run_refs={"events": list(decision_events),
                      "runs": {str(e): certified_runs[int(e)] for e in decision_events}},
        )
        meta_ids = {int(row["player_id"]) for row in universe["universe"]}
        player_meta = rc.load_player_meta(source_conn, meta_ids)
        universe["replacement_edges"] = cu.build_replacement_edges(
            universe_rows=universe["universe"], owned_ids=source_squad["squad_ids"], state=source_state,
            price_snapshot=snapshot, player_meta=player_meta,
        )
        screen = fg.screen_legal_actions(
            universe_rows=universe["universe"], replacement_edges=universe["replacement_edges"],
            owned_ids=source_squad["squad_ids"], decision_events_window=decision_events,
        )
        optimizer_config = ro.OptimizerConfig(
            events=tuple(decision_events), search_draws=STAGE1_DRAWS, seed=SEED,
            beam_width=int(args.beam), exact_evaluation_budget=int(args.exact_budget),
            search_n_per_criterion=int(args.search_n), singles_per_out=int(args.singles_per_out),
            max_transfers_per_event=int(args.max_transfers_per_event),
            policy_selection_worlds=0,
        )
        # All-player discovery accounting, asserted BEFORE any route search.
        discovery = cu.discovery_completeness(
            pool=pool,
            universe_rows=universe["universe"],
            excluded=universe.get("excluded") or [],
            replacement_edges=universe["replacement_edges"],
            screen=screen,
            enforce=True,
        )
        t0 = time.time()
        result = ro.optimize(
            universe=universe, initial_state=source_state, scenario=scenario, player_meta=player_meta,
            bundles=bundles, conn=conn, config=optimizer_config, cache_dir=Path(args.cache_dir),
            provenance={**search_provenance,
                        "discovery_certification_identity": discovery_identity,
                        "exact_evaluation_certification_identity": exact_identity},
        )
        search_seconds = time.time() - t0
        source_conn.close()

        transfers_by_route = {}
        for route_id, record in (result.get("routes") or {}).items():
            transfers_by_route[route_id] = {
                int(action["event"]): [{"out": int(m["out"]), "in": int(m["in"])} for m in action.get("transfers") or []]
                for action in (record.get("actions") or [])
            }
        routes = fg.routes_for_decision(result.get("routes") or {}, transfers_by_route=transfers_by_route)
        baseline = next((row["route_id"] for row in routes if not any(r["transfers"] for r in row["per_event"])), None)
        # First pass: the transfer decision.  The H1 lineup is then taken from the
        # PREFERRED route (or, when suppressed, from the baseline) so the
        # operator-visible lineup always belongs to the route being recommended.
        decision = fg.evaluate_four_gw_decision(
            planning_event=planning_event, support_by_event=certified_support, cutoff=cutoff,
            last_event=last_event, screened_actions=screen, routes=routes,
            baseline_route_id=baseline, lineup=None,
        )
        preferred_route_id = (decision.get("transfer_recommendation") or {}).get("preferred_route_id")
        lineup_route_id = preferred_route_id or baseline
        lineup_policy = fg.lineup_policy_for_route(
            routes=routes, route_id=lineup_route_id, decision_events_window=decision_events,
        )
        if lineup_policy is not None:
            decision = fg.evaluate_four_gw_decision(
                planning_event=planning_event, support_by_event=certified_support, cutoff=cutoff,
                last_event=last_event, screened_actions=screen, routes=routes,
                baseline_route_id=baseline,
                lineup={"status": fg.LINEUP_ONLY, "policy": lineup_policy,
                        "lineup_route_id": lineup_route_id, "lineup_basis": "CURRENT_GW_H1"},
            )
        else:
            lineup_route_id = None
        # --- R4B.2c: four-GW fixture horizon (from the CERTIFICATION snapshot) --
        fixture_horizon = fg.classify_fixture_horizon(
            source_conn, decision_events,
            last_event=fg.season_last_event_from_db(source_conn),
        )
        # --- R4B.2c: decision confidence (computed AFTER ranking; never ranks) --
        # Paired CRN near-tie is CONSUMED from the route comparison when the result
        # exposes it; otherwise it is reported unavailable and the state cannot claim
        # a near tie.
        # --- ROLE-RELEVANT PLAYER SET (R4B.2c integration correction 1) ---------
        # NOT the certified squad.  After the preferred route is known, the players
        # whose role drives the recommendation's thesis are: everyone transferred IN,
        # everyone transferred OUT, and the preferred route's captain and vice.
        # A transfer-IN player is usually NOT in the pre-transfer certified squad and
        # must nevertheless be evaluated.
        role_relevant_ids: set[int] = set()
        preferred_key = preferred_route_id or lineup_route_id
        for event_transfers in (transfers_by_route.get(preferred_key) or {}).values():
            for move in event_transfers:
                role_relevant_ids.add(int(move["in"]))
                role_relevant_ids.add(int(move["out"]))
        if lineup_policy is not None:
            for attribute in ("captain_id", "vice_captain_id"):
                value = (
                    getattr(lineup_policy, attribute, None)
                    if not isinstance(lineup_policy, dict)
                    else lineup_policy.get(attribute)
                )
                if value is not None:
                    role_relevant_ids.add(int(value))
        role_relevant_players = sorted(role_relevant_ids)

        # --- CANONICAL PAIRED CRN DIAGNOSTIC (integration correction 2) --------
        # Exactly one canonical record: the FINAL preferred leader versus its
        # relevant runner-up/comparator.  Legacy keys are accepted only as a
        # fallback; when no canonical record exists, confidence must NOT infer
        # "not near tied".
        paired_record = result.get(dc.CANONICAL_PAIRED_DIAGNOSTIC_KEY)
        if not isinstance(paired_record, dict):
            paired_record = None
            for key in ("leader_paired", "paired_leader_vs_runner_up"):
                candidate = result.get(key)
                if isinstance(candidate, dict) and (
                    "mean_difference" in candidate or "near_tied" in candidate
                ):
                    paired_record = candidate
                    break
        if not isinstance(paired_record, dict) or not paired_record:
            print(
                f"confidence: {dc.DIAG_PAIRED_DIAGNOSTIC_REQUIRED}: route comparison published no "
                f"canonical paired diagnostic ({dc.CANONICAL_PAIRED_DIAGNOSTIC_KEY}); confidence "
                "cannot be decisive",
                file=sys.stderr,
            )

        role_evidence, role_source = _certified_role_evidence(
            conn, certified_runs, role_relevant_players, decision_events, certification
        )
        confidence = dc.classify_decision_confidence(
            paired=paired_record,
            role_evidence=role_evidence,
            focus_player_ids=role_relevant_players,
            role_evidence_source=role_source,
        )
        dc.assert_confidence_invariants(confidence)

        artifact = {
            "schema": "fpl_brain.four_gw_decision.v1",
            "planning_event": planning_event,
            "planning_cutoff": cutoff,
            "decision_events": list(decision_events),
            "cutoff_guard": override,
            "search": {
                "engine": "route_optimizer.optimize (accepted bounded multi-event search)",
                "config": optimizer_config.as_dict(),
                "seconds": round(search_seconds, 1),
                "world_info": result.get("world_info"),
                "search_stats": result.get("search_stats"),
                "exact_evaluations": result.get("exact_evaluations"),
                "promoted_route_count": result.get("promoted_route_count"),
                "flags": result.get("flags"),
                "no_stability_ladder_rerun": True,
                # Truthful draw provenance: one Stage-1 count is used for EVERY
                # event.  No per-event 10k claim until R4B.2 implements the real
                # two-stage finalist refinement.
                "stage1_search_draws": STAGE1_DRAWS,
                "draw_fidelity": {str(event): STAGE1_DRAWS for event in decision_events},
            },
            "fixture_horizon": fixture_horizon,
            "decision_confidence": confidence,
            "provenance": {
                **search_provenance,
                "search_artifact_cutoff": result.get("planning_cutoff"),
                "search_artifact_context_hash": result.get("planning_context_hash"),
                "search_supported_events": result.get("supported_events"),
                "lineup_route_id": lineup_route_id,
                "lineup_basis": "CURRENT_GW_H1",
            },
            "discovery_completeness": discovery,
            "official_pool_identity": pool_identity,
            "screened_actions": {k: v for k, v in screen.items() if k != "promotion_pool"},
            "decision": decision,
            "no_execution": True,
        }
        (out_dir / "four_gw_decision.json").write_text(
            json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True, default=cu.jsonable) + "\n",
            encoding="utf-8",
        )
        # R4B.2c decision gate: an unresolved fixture that could alter any team's
        # fixture set inside the four-GW window blocks the normal transfer
        # recommendation.  It is SUPPRESSED, never replaced by a "best H1 transfer".
        if not fixture_horizon["complete"]:
            suppressed = dict(decision.get("transfer_recommendation") or {})
            suppressed.update(
                {
                    "status": fg.RECOMMENDATION_SUPPRESSED,
                    "reason": fg.DECISION_HORIZON_INCOMPLETE,
                    "preferred_route_id": None,
                    "fixture_horizon_blocking_reasons": fixture_horizon["blocking_reasons"],
                }
            )
            decision["transfer_recommendation"] = suppressed
            artifact["transfer_recommendation"] = suppressed
            print(
                "transfer recommendation SUPPRESSED: fixture horizon incomplete "
                f"({len(fixture_horizon['blocking_reasons'])} blocking reason(s))"
            )
        block = decision["transfer_recommendation"]
        print(f"STAGE B decision: {block['status']} preferred={block.get('preferred_route_id')} "
              f"eligible={block.get('eligible_route_count')} excluded={len(block.get('excluded_routes') or [])}")
        for row in block.get("ranking", [])[:6]:
            print(f"  #{row['rank']} {row['route_id']} 4GW net={row['four_gw_net_core']:.3f} "
                  f"hits={row['total_hit_points']} terminal FT={row['terminal_ft']} bank={_money(row['terminal_bank_tenths'])}")
        print(f"  search {search_seconds:.1f}s  decisions in {time.time() - started:.1f}s  "
              f"artifact={out_dir / 'four_gw_decision.json'}")
        return 0
    finally:
        if entered_guard is not None:
            entered_guard.__exit__(*sys.exc_info())
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
