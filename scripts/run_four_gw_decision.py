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
from fpl_brain import finalist_refinement as fr
from fpl_brain import four_gw_decision as fg
from fpl_brain import certified_bundle
from fpl_brain import manager_worlds, route_optimizer as ro, transfer_state as ts
from fpl_brain import execution
from fpl_brain import ingest_provenance as provenance
from fpl_brain import repositories as repo
from fpl_brain import route_stability as rs
from fpl_brain.config import config_path, load_config
from fpl_brain.database import connect_database
from fpl_brain.planning import get_planning_context
from fpl_brain.utils import parse_utc, utc_now


def _refuse_execution(guard, code: int, reason: str) -> int:
    """Record a deliberate refusal as a FAILED execution run, then return the code."""

    guard.finish(execution.RUN_FAILED, reason)
    return code


def _suppress_transfer_recommendation(decision: dict, *, reason: str, extra: dict | None = None) -> dict:
    """Replace the normal transfer recommendation with an explicit suppression.

    ONE implementation for every suppression path (fixture horizon, search
    instability, ...).  The route table and the ranking stay in the artifact; only
    decisiveness is removed, and ``preferred_route_id`` is cleared so no downstream
    consumer can mistake the numerically highest route for a recommendation.  There
    is deliberately no best-current-GW-transfer fallback.
    """

    suppressed = dict(decision.get("transfer_recommendation") or {})
    suppressed.update(
        {
            "status": fg.RECOMMENDATION_SUPPRESSED,
            "reason": str(reason),
            "preferred_route_id": None,
        }
    )
    suppressed.update(extra or {})
    decision["transfer_recommendation"] = suppressed
    return suppressed


HEARTBEAT_INTERVAL_SECONDS = 60.0


def _stage_timings(*, search_seconds: float, refine_started: float, refine_finished: float,
                   stability_seconds: float) -> dict:
    """The artifact's per-stage wall times.

    ``stage2_refinement_seconds`` must measure the Stage-2 refinement ITSELF.  The previous
    expression subtracted the Stage-1 search time from a timestamp taken BEFORE the
    refinement, so it reported the (near-zero) gap between the two statements instead of the
    refinement; the R5 final acceptance run consequently recorded 0.0 s for a ~59-minute
    Stage 2 (telemetry only - nothing downstream read it).
    """

    return {
        "stage1_seconds": round(float(search_seconds), 3),
        "stage2_refinement_seconds": round(max(0.0, float(refine_finished) - float(refine_started)), 3),
        "stability_seconds": round(float(stability_seconds), 3),
    }


def _scheduled_workers(result) -> int | None:
    """The worker count an optimizer result actually ran with (None when sequential).

    Read from the result's own ``parallel_exact`` block, so the artifact records what
    RAN rather than what was requested.
    """

    if not isinstance(result, dict):
        return None
    block = result.get("parallel_exact")
    return None if not isinstance(block, dict) else block.get("worker_count")


def _cancel_probe(guard, *, heartbeat_interval: float = HEARTBEAT_INTERVAL_SECONDS):
    """A safe-boundary probe: canonical cancel check + periodic lease heartbeat.

    Returns a zero-argument callable to be invoked ONLY at the boundaries P1
    identified (top of a search level, between world-matrix builds, between exact
    finalist-route evaluations, immediately before the escalation, between
    prefix-invariance events).  At every one of those points no write transaction is
    open — ``run_search`` and ``optimize`` issue no SQLite writes at all, and the
    world builders have fully returned — so the heartbeat can never widen a write
    lock window.

    ``check_cancel`` re-reads the run row, so a ``CANCEL_REQUESTED`` written by
    another process is observed; it raises ``RunCancelled``, which unwinds through
    the runner's existing ``finally`` into ``production_run_guard``'s cancel branch
    (``acknowledge_cancel`` + lease release).
    """

    state = {"last_heartbeat": 0.0, "checks": 0, "heartbeats": 0}

    def probe() -> None:
        guard.check_cancel()
        state["checks"] += 1
        now = time.monotonic()
        if now - state["last_heartbeat"] >= float(heartbeat_interval):
            guard.heartbeat()
            state["last_heartbeat"] = now
            state["heartbeats"] += 1

    probe.state = state  # type: ignore[attr-defined]
    return probe


def _role_relevant_ids(transfers_by_route, preferred_key, lineup_policy) -> list[int]:
    """Role-relevant player ids for the FINAL preferred route, sorted.

    Exactly: every player transferred IN, every player transferred OUT, and the
    preferred route's captain and vice.  Deliberately NOT the certified squad
    (R4B.2c integration correction 1).  Extracted so the caller cannot
    accidentally derive it from a pre-escalation route: the only input that names
    the route is ``preferred_key``.
    """

    ids: set[int] = set()
    for event_transfers in (transfers_by_route.get(preferred_key) or {}).values():
        for move in event_transfers:
            ids.add(int(move["in"]))
            ids.add(int(move["out"]))
    if lineup_policy is not None:
        for attribute in ("captain_id", "vice_captain_id"):
            value = (
                getattr(lineup_policy, attribute, None)
                if not isinstance(lineup_policy, dict)
                else lineup_policy.get(attribute)
            )
            if value is not None:
                ids.add(int(value))
    return sorted(ids)


SEED = 20260911
#: Stage-1 (screening) draw count.  This is the count ACTUALLY used to build the
#: shared per-event world matrices for every event, and it is what the artifact
#: reports.
STAGE1_DRAWS = 2_000
#: Stage-2 (finalist precision) draw count.  R4B.2b re-evaluates ONLY the
#: finalists at this budget, with the same seed and the same certified inputs, and
#: the artifact reports the count that actually ran.
STAGE2_DRAWS = fr.STAGE2_DRAWS

#: P3.1 — production worker count for the 10,000-draw exact-evaluation paths ONLY.
#:
#: One exact (event, squad) evaluation at 10,000 draws costs ~190-230 s and Stage 2 needs 84
#: of them, so the finalist refinement plus the ONE bounded escalation dominate the decision
#: run.  P3 proved a 4-worker pool reproduces those evaluations BIT-IDENTICALLY on the real
#: certified fixture (18 real unit comparisons, 0 mismatches) and measured 53.9 s/unit against
#: 230.5 s sequential (4.28x), which brings the modelled full Stage-2 from 3.91 h to ~70 min.
#:
#: This constant is deliberately explicit rather than a library default:
#: ``route_optimizer.optimize(parallel_workers=...)`` still defaults to ``None`` (sequential)
#: and ``parallel_exact.DEFAULT_WORKER_COUNT`` still defaults to 1, so no caller inherits a
#: process pool by accident.
#:
#: STAGE 1 ALSO USES THE POOL, now that it has its own equivalence proof.  The original
#: decision kept Stage 1 sequential on two grounds: a 2,000-draw evaluation is much shorter
#: than a 10,000-draw one, so spawn/matrix-load overhead might outweigh the gain; and the
#: P3 bit-identity proof covered only the 10,000-draw path.  Performance Spike B1 supplied
#: the missing proof (serial == parallel across the real Stage-1 unit population, including
#: the exact cache keys and values, out-of-order completion, worker failure and the
#: role-actionability policy set) and measured the scaling, so both grounds are now resolved
#: by evidence rather than by assumption.  The Stage-1 `stage1_result` search call below
#: passes `parallel_workers` for that reason; `worker_count <= 1` still selects the
#: unchanged sequential evaluator.
#:
#: (Note on wording: this block deliberately does not spell the Stage-1 call as source
#: text, because `tests/test_r4b2a_search_coverage.py` greps this file for the first
#: occurrence of the search call to assert that screening and discovery run before it.
#: A comment mentioning the call would satisfy that grep earlier than the real call.)
PRODUCTION_PARALLEL_EXACT_WORKERS = 4

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


def bundle_for(event: int, runs: dict, cutoff: str, draws: int, *, conn=None, certification: dict | None = None):
    """The CERTIFIED bundle for one event, carrying its provenance.

    The run ids come from the certification artifact, and so do the model versions
    the bundle DECLARES: the artifact's recorded versions are declared here, and the
    canonical identity is computed by the one shared algorithm.  A bundle that
    cannot declare both is refused by the loader rather than simulated, so this is
    also where the exact certified generation is proved against the artifact.
    """

    from fpl_brain import certified_bundle as cb
    from fpl_brain import route_comparator as rc

    artifact = dict(certification or {})
    certified = (artifact.get("certified_bundles") or {}).get(str(int(event))) or {}
    versions = dict(certified.get("model_versions") or {}) or dict(
        artifact.get("required_model_versions") or {}
    )
    bundle = rc.certified_event_bundle(
        event=int(event),
        runs=runs,
        cutoff=cutoff,
        model_versions=versions,
        simulations=int(draws),
        seed=SEED,
        code_snapshot_sha256=certified.get("code_snapshot_sha256")
        or artifact.get("code_snapshot_sha256"),
        data_snapshot_sha256=certified.get("data_snapshot_sha256")
        or artifact.get("data_snapshot_sha256"),
        planning_context_hash=certified.get("planning_context_hash"),
    )
    # The LOADED artifact is passed on, not the copy made above for reading: a copy is
    # a raw mapping again and would only be revalidated at the boundary, while the
    # loaded value is the authorisation itself.
    cb.assert_event_bundle_certified(
        conn, bundle, event=int(event), certification=certification or artifact or None
    )
    return bundle


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
        "--parallel-workers", type=int, default=PRODUCTION_PARALLEL_EXACT_WORKERS,
        help="P3.1 worker processes for the 10,000-draw FINALIST and ESCALATION exact "
             "evaluations only (Stage 1 stays sequential).  1 disables the pool and runs the "
             "sequential evaluator in-process. The exact evaluations are bit-identical either "
             "way; this changes scheduling only.",
    )
    parser.add_argument(
        "--stage2-draws", type=int, default=STAGE2_DRAWS,
        help="Monte Carlo draws per event for the Stage-2 FINALIST-ONLY refinement. "
             "Must be strictly greater than the Stage-1 draw count; the artifact "
             "reports the count that actually ran.",
    )
    parser.add_argument(
        "--certification",
        help="path to the authoritative certification artifact (REQUIRED for --stage search|all)",
    )
    parser.add_argument(
        "--calibration",
        help="optional path to a PE-8 calibration evidence artifact.  PE-9 CONSULTS it at the "
             "certification boundary: its identity, cutoff and per-event run ids must belong to the "
             "certified bundle, and its evidence states are recorded.  Omitting it does not refuse the "
             "decision -- it means no calibration claim is made, which is reported as EVIDENCE_LIMITED "
             "rather than presented as a clean result.",
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
        parallel_workers = int(args.parallel_workers)
        print(f"P3.1 parallel exact evaluation: stage2+escalation workers={parallel_workers} "
              f"(stage1 sequential); "
              f"{'pool' if parallel_workers > 1 else 'sequential evaluator'}")
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
        # --- PE-9 reconciliation of the two views ------------------------------
        # Stage A's horizon is a NON-PRODUCTION latest-per-family rediscovery.  The
        # decision below never consumes it, but it IS published as this run's
        # readiness artifact, so PE-9 reconciles the certification against it rather
        # than leaving the two published views free to describe different predictive
        # worlds.  A divergence is RECORDED, not fatal: the certified run ids are
        # authoritative, and a newer same-cutoff rerun must never replace them (nor
        # block the decision by merely existing).
        readiness_runs = {
            int(e): {str(f): int(r) for f, r in (support[int(e)].get("matched_runs") or {}).items()}
            for e in decision_events
        }
        readiness_divergence = {
            str(e): {
                "readiness_runs": readiness_runs[int(e)],
                "certified_runs": {str(f): int(r) for f, r in certified_runs[int(e)].items()},
            }
            for e in decision_events
            if readiness_runs[int(e)]
            != {str(f): int(r) for f, r in certified_runs[int(e)].items()}
        }
        if readiness_divergence:
            print(
                "readiness view reconciled: the NON-PRODUCTION latest-per-family readiness view names a "
                "different generation for "
                + ", ".join(f"GW{event}" for event in sorted(readiness_divergence))
                + "; the certified run ids are authoritative and are what this decision consumes"
            )
            for event in sorted(readiness_divergence):
                record = readiness_divergence[event]
                print(
                    f"  GW{event}: readiness={record['readiness_runs']} "
                    f"certified={record['certified_runs']}"
                )
        else:
            print("readiness view reconciled: the readiness view names the certified generation for every event")
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
            # The reconciliation itself is persisted, so the readiness artifact and
            # the decision artifact cannot silently disagree about which generation
            # each of them described.
            "readiness_source": "NON_PRODUCTION_LATEST_PER_FAMILY",
            "readiness_reconciled_against_certification": True,
            "readiness_generation_divergence": readiness_divergence,
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
                event, certified_runs[int(event)], cutoff, draws_for(event, decision_events),
                conn=conn, certification=certification,
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

        # --- PE-9: certify the horizon and persist the certification result -----
        # The horizon is certified AS a horizon, from the ONE artifact, over the exact
        # ids the decision is about to consume.  The result carries the per-bundle
        # states, the per-event bundle identities and model versions, the code and data
        # snapshots and the calibration identity consulted, and it is persisted WITH
        # the decision artifact below -- so what a later reader can verify is the
        # authorisation, not a narrative about it.
        calibration_artifact = None
        if args.calibration:
            try:
                calibration_artifact = json.loads(
                    Path(args.calibration).read_text(encoding="utf-8")
                )
            except Exception as failure:
                print(
                    f"decision refused: the calibration artifact at {args.calibration} could not be "
                    f"read: {failure}",
                    file=sys.stderr,
                )
                source_conn.close()
                return _refuse_execution(guard, 6, "unreadable calibration artifact")
        try:
            pe9_certification = certified_bundle.certify_decision_horizon(
                conn,
                certification=certification,
                events=decision_events,
                cutoff=cutoff,
                required_versions=certification.get("required_model_versions") or None,
                calibration=calibration_artifact,
                last_event=last_event,
            )
        except certified_bundle.CertificationRefused as failure:
            print(f"decision refused: {failure}", file=sys.stderr)
            source_conn.close()
            return _refuse_execution(guard, 6, f"{failure.token}")
        except certified_bundle.BundleIncoherent as failure:
            print(f"decision refused: {failure}", file=sys.stderr)
            source_conn.close()
            return _refuse_execution(guard, 6, "certified bundle incoherent")
        pe9_certification["certification_result_identity"] = (
            certified_bundle.certification_result_identity(pe9_certification)
        )
        # The CERTIFIED horizon is the gate, not a report.  A bundle the calibration
        # evidence could not certify is a blocked event, so one blocked event makes the
        # required horizon incomplete and the decision is refused rather than taken on
        # the remaining events: there is no partial-horizon transfer recommendation.
        if pe9_certification["horizon_state"] == fg.DECISION_HORIZON_INCOMPLETE:
            print(
                f"decision refused: {fg.DECISION_HORIZON_INCOMPLETE}: PE-9 certification blocked "
                f"event(s) {pe9_certification['horizon']['blocked_events']}; the required horizon cannot "
                "be certified",
                file=sys.stderr,
            )
            source_conn.close()
            return _refuse_execution(guard, 6, f"{fg.DECISION_HORIZON_INCOMPLETE}")
        for event in decision_events:
            record = pe9_certification["per_event"][str(event)]
            if record["bundle_identity"] != certified_support[int(event)]["bundle_identity"]:
                print(
                    f"decision refused: {certified_bundle.STATE_PREDICTIVE_BUNDLE_INCOHERENT}: GW{event} "
                    "certifies to a different bundle identity than the support the decision consumed",
                    file=sys.stderr,
                )
                source_conn.close()
                return _refuse_execution(guard, 6, "certification identity mismatch")
        print(
            f"PE-9 certification: horizon={pe9_certification['horizon_state']} "
            f"phase={pe9_certification['phase_terminal_state']} "
            f"states={ {event: pe9_certification['per_event'][str(event)]['state'] for event in decision_events} } "
            f"identity={pe9_certification['certification_result_identity'][:24]}…"
        )
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
        # A "refinement" that is not higher fidelity would still be reported as one,
        # so it is refused before any work is done.
        if int(args.stage2_draws) <= int(STAGE1_DRAWS):
            print(
                f"decision refused: --stage2-draws {int(args.stage2_draws)} must exceed the Stage-1 "
                f"draw count {STAGE1_DRAWS}",
                file=sys.stderr,
            )
            source_conn.close()
            return _refuse_execution(guard, 6, "stage2 draws not higher than stage1")
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
        stage1_result = ro.optimize(
            universe=universe, initial_state=source_state, scenario=scenario, player_meta=player_meta,
            bundles=bundles, conn=conn, config=optimizer_config, cache_dir=Path(args.cache_dir),
            certification=certification,
            provenance={**search_provenance,
                        "discovery_certification_identity": discovery_identity,
                        "exact_evaluation_certification_identity": exact_identity},
            parallel_workers=parallel_workers,
        )
        search_seconds = time.time() - t0

        # --- R4B.2b Stage 2: FINALIST-ONLY precision refinement ----------------
        # Same seed, same certified bundles, same route legality, same discovery
        # universe; the ONLY changed input is the shared-world draw count.  The
        # refinement also publishes the one canonical paired record below.
        t_refine = time.time()
        # ONE run-scoped exact-evaluation cache, shared by the Stage-2 refinement and the
        # single stability escalation.  Reuse is keyed on the COMPLETE evaluation
        # identity (event, canonical squad, draws, seed, world provenance), and both
        # calls run at the same draw count, seed and certified worlds, so a shared entry
        # is literally the same evaluation.  The cache never leaves the run.
        cancel_probe = _cancel_probe(guard)
        run_exact_cache: dict = {}
        refinement = fr.refine_finalists(
            universe=universe, initial_state=source_state, scenario=scenario,
            player_meta=player_meta, bundles=bundles, conn=conn, base_config=optimizer_config,
            stage1_result=stage1_result, stage2_draws=int(args.stage2_draws),
            certification=certification, cache_dir=Path(args.cache_dir),
            exact_cache=run_exact_cache, cancel_probe=cancel_probe,
            parallel_workers=parallel_workers,
        )
        t_refine_finished = time.time()
        stage2_result = refinement["refined"]
        stage2_parallel = (stage2_result.get("parallel_exact") or {}).get("worker_count")
        print(f"exact cache: {len(run_exact_cache)} entries after the Stage-2 refinement; "
              f"stage2 parallel workers={stage2_parallel}")

        # --- R4B.2b REPAIR: stability FIRST, so the final ranking is known ------
        # The gate may run the ONE bounded escalation (the next supported search
        # breadth, at the Stage-2 draw budget, in the SAME worlds).  Its widened
        # result is captured through a sink because the stability REPORT is
        # serialized into the artifact and a world matrix must never be.  The
        # leader-change comparison is always Stage-2 versus Stage-1: the gate asks
        # whether widening the search changes the answer the narrow budget gave.
        leader_change = fr.analyze_leader_change(stage1_result, stage2_result)
        escalated_sink: dict = {}
        t_stability = time.time()
        stability = fr.assess_search_stability(
            refined_result=stage2_result,
            leader_change=leader_change,
            canonical_paired=refinement["canonical_paired_near_tie"],
            escalation=_escalation_runner(
                universe=universe, initial_state=source_state, scenario=scenario,
                player_meta=player_meta, bundles=bundles, conn=conn, base_config=optimizer_config,
                stage1_result=stage1_result, stage2_draws=int(args.stage2_draws),
                prebuilt_worlds=refinement.get("prebuilt_worlds"),
                certification=certification,
                finalist_partials=fr.finalist_partials(stage1_result, refinement["finalist_selection"]),
                exact_cache=run_exact_cache,
                cancel_probe=cancel_probe,
                parallel_workers=parallel_workers,
            ),
            config=fr.StabilityGateConfig(current_beam=int(args.beam)),
            escalated_result_sink=escalated_sink,
            cancel_probe=cancel_probe,
        )
        stability_seconds = time.time() - t_stability
        escalated_result = escalated_sink.get("result")

        # §4: after an escalation the FINAL ranking is the widened-search ranking,
        # evaluated on the same Stage-2 worlds.  Without an escalation it is the
        # Stage-2 finalist ranking.
        final = fr.final_ranking_after_escalation(
            stage2_result=stage2_result, escalated_result=escalated_result,
        )
        # From here on the DECISION is taken at the final supported evaluation
        # budget: `result` is that final result (widened when the escalation ran),
        # and both Stage 1 and the Stage-2-only ranking are reported separately.
        # This also keeps the documented
        # `result.get(dc.CANONICAL_PAIRED_DIAGNOSTIC_KEY)` consumption path exactly
        # as the confidence contract requires.
        result = final["result"]

        transfers_by_route = {}
        for route_id, record in (result.get("routes") or {}).items():
            transfers_by_route[route_id] = {
                int(action["event"]): [{"out": int(m["out"]), "in": int(m["in"])} for m in action.get("transfers") or []]
                for action in (record.get("actions") or [])
            }
        # R5-P0-01: `result` is a route_optimizer result, so it must cross the OPTIMIZER
        # boundary.  `routes_for_decision` is the comparator boundary and would adapt every
        # optimizer route to an empty per_event with null terminal accounting, which the
        # eligibility gate then (correctly) reported as an incomplete route - excluding every
        # route and emitting no recommendation at all.
        routes = fg.optimizer_routes_for_decision(
            result.get("routes") or {}, transfers_by_route=transfers_by_route)
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
        source_conn.close()

        # --- R4B.2b REPAIR §5: canonical paired record from the FINAL ranking ---
        # route_a is the FINAL preferred route and route_b is the ACTUAL next-ranked
        # route in the FINAL (post-escalation) ranking — not the old Stage-2
        # finalist runner-up.  The decision layer is authoritative for which route
        # is preferred (it may in principle exclude a route the optimizer ranked
        # first), so it is passed in explicitly and the comparator follows from the
        # FINAL ranking.
        decision_preferred = (decision.get("transfer_recommendation") or {}).get("preferred_route_id")
        final = fr.final_ranking_after_escalation(
            stage2_result=stage2_result,
            escalated_result=escalated_result,
            preferred_route_id=decision_preferred or final["final_ranking"]["preferred_route_id"],
        )
        final_leader = final["final_ranking"]["preferred_route_id"]
        comparator = final["final_ranking"]["runner_up_route_id"]
        canonical = final["canonical_paired_near_tie"]
        refinement["stage2_finalist_ranking"] = {
            "preferred_route_id": final["stage2_leader_route_id"],
            "runner_up_route_id": None,
        }
        refinement["optimizer_ranked_leader_route_id"] = final["stage2_leader_route_id"]
        refinement["final_rank_1_route_id"] = final["final_rank_1_route_id"]
        refinement["ranking_source"] = final["ranking_source"]
        refinement["comparator_source"] = final["comparator_source"]
        refinement["canonical_alignment"] = final["canonical_alignment"]
        refinement["canonical_paired_near_tie"] = canonical
        refinement["final_ranking"] = dict(final["final_ranking"])
        refinement["simulation_fidelity"]["final_ranking"] = dict(final["final_ranking"])
        refinement["simulation_fidelity"]["canonical_paired_near_tie"] = canonical
        refinement["leader_change"] = leader_change
        refinement["stability"] = stability
        refinement["simulation_fidelity"]["leader_change"] = leader_change
        refinement["simulation_fidelity"]["stability"] = stability
        refinement["timing_s"] = _stage_timings(
            search_seconds=search_seconds, refine_started=t_refine,
            refine_finished=t_refine_finished, stability_seconds=stability_seconds)
        result["canonical_paired_near_tie"] = canonical
        print(
            f"refinement: finalists={len(refinement['finalist_selection']['finalist_route_ids'])} "
            f"stage2_draws={int(args.stage2_draws)} prefix="
            f"{(refinement.get('prefix_invariance') or {}).get('status')} "
            f"leader_change={leader_change['changed']}/{leader_change['accepted']} "
            f"stability={stability['state']} escalation={stability['escalation_used']} "
            f"ranking_source={final['ranking_source']} "
            f"({stability_seconds:.1f}s)"
        )
        if escalated_result is not None:
            print(
                f"  escalation to beam {stability['escalation_beam']} re-ranked "
                f"{len(result.get('routes') or {})} routes on the Stage-2 worlds; "
                f"final leader {final_leader} (Stage-2 leader was {final['stage2_leader_route_id']})"
            )
        if canonical is None:
            print(
                f"confidence: {dc.DIAG_PAIRED_DIAGNOSTIC_REQUIRED}: no canonical paired record for "
                f"leader={final_leader} comparator={comparator}; confidence cannot be decisive and "
                "search stability cannot be claimed",
                file=sys.stderr,
            )

        # --- R4B.2c: decision confidence (computed AFTER ranking; never ranks) --
        # Paired CRN near-tie is CONSUMED from the route comparison when the result
        # exposes it; otherwise it is reported unavailable and the state cannot claim
        # a near tie.
        # --- ROLE-RELEVANT PLAYER SET (R4B.2c integration correction 1, REPAIR §6) --
        # NOT the certified squad.  The players whose role drives the
        # recommendation's thesis are: everyone transferred IN, everyone transferred
        # OUT, and the armband, **of the FINAL preferred route** — so a widened
        # search that changes the preferred route also moves the role-relevant set.
        # A transfer-IN player is usually NOT in the pre-transfer certified squad and
        # must nevertheless be evaluated.
        preferred_key = final_leader or preferred_route_id or lineup_route_id
        role_relevant_players = _role_relevant_ids(transfers_by_route, preferred_key, lineup_policy)

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

        # --- Suppression is applied BEFORE the artifact is written --------------
        # Every gate that removes decisiveness must be visible in the artifact that
        # is actually on disk.  Deferred suppression would leave the file asserting
        # a recommendation the runner no longer stands behind.
        suppression_reasons: list[str] = []
        if not fixture_horizon["complete"]:
            # R4B.2c decision gate: an unresolved fixture that could alter any team's
            # fixture set inside the four-GW window blocks the normal transfer
            # recommendation.  It is SUPPRESSED, never replaced by a "best H1 transfer".
            _suppress_transfer_recommendation(
                decision, reason=fg.DECISION_HORIZON_INCOMPLETE,
                extra={"fixture_horizon_blocking_reasons": fixture_horizon["blocking_reasons"]},
            )
            suppression_reasons.append(fg.DECISION_HORIZON_INCOMPLETE)
            print(
                "transfer recommendation SUPPRESSED: fixture horizon incomplete "
                f"({len(fixture_horizon['blocking_reasons'])} blocking reason(s))"
            )
        if stability["state"] != fr.SEARCH_STABLE:
            # R4B.2b search-stability gate: the bounded search cannot separate the
            # preferred route from its alternatives at the widest supported budget.
            # The route table stays available; decisiveness is removed, and there is
            # deliberately no best-current-GW-transfer fallback.
            _suppress_transfer_recommendation(
                decision, reason=fg.DECISION_SEARCH_NOT_STABLE,
                extra={
                    "search_stability_state": stability["state"],
                    "search_stability_basis": stability.get("stability_basis"),
                    "search_budget_sequence": stability["search_budget_sequence"],
                    "escalation_used": stability["escalation_used"],
                },
            )
            suppression_reasons.append(fg.DECISION_SEARCH_NOT_STABLE)
            print(
                f"transfer recommendation SUPPRESSED: {stability['state']} "
                f"(basis={stability.get('stability_basis')})"
            )
        if canonical is None:
            # §5: a decisive recommendation requires the canonical paired record.  It
            # could not be produced, so decisiveness is removed rather than inferred.
            _suppress_transfer_recommendation(
                decision, reason=dc.DIAG_PAIRED_DIAGNOSTIC_REQUIRED,
                extra={"paired_diagnostic_required": True,
                       "search_stability_state": stability["state"]},
            )
            suppression_reasons.append(dc.DIAG_PAIRED_DIAGNOSTIC_REQUIRED)
            print(
                f"transfer recommendation SUPPRESSED: {dc.DIAG_PAIRED_DIAGNOSTIC_REQUIRED} "
                f"(leader={final_leader} comparator={comparator})"
            )

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
                "world_info": stage1_result.get("world_info"),
                "search_stats": stage1_result.get("search_stats"),
                "exact_evaluations": stage1_result.get("exact_evaluations"),
                "promoted_route_count": stage1_result.get("promoted_route_count"),
                "flags": stage1_result.get("flags"),
                "no_stability_ladder_rerun": True,
                # Truthful draw provenance: Stage 1 screens EVERY route at this
                # count for EVERY event.  The higher Stage-2 count is a FINALIST-ONLY
                # refinement and is reported separately, never as route coverage.
                "stage1_search_draws": STAGE1_DRAWS,
                "draw_fidelity": {str(event): STAGE1_DRAWS for event in decision_events},
            },
            "finalist_refinement": {
                "stage2_draws": int(args.stage2_draws),
                "finalists": refinement["finalist_selection"],
                "final_ranking": refinement["final_ranking"],
                "optimizer_ranked_leader_route_id": refinement["optimizer_ranked_leader_route_id"],
                "canonical_alignment": refinement["canonical_alignment"],
                "leader_change": leader_change,
                "stability": stability,
                "route_table": {
                    "routes": result.get("routes"),
                    "flags": result.get("flags"),
                    "exact_evaluations": result.get("exact_evaluations"),
                },
                "timing_s": refinement["timing_s"],
                "no_recommendation": True,
            },
            "simulation_fidelity": refinement["simulation_fidelity"],
            "parallel_exact_scheduling": {
                "requested": parallel_workers,
                "stage2_workers": stage2_parallel,
                "escalation_workers": _scheduled_workers(escalated_result),
                "stage1_workers": None,
                "stage1_note": "Stage 1 stays sequential by design (see "
                               "PRODUCTION_PARALLEL_EXACT_WORKERS)",
                "semantics": "SCHEDULING_ONLY_BIT_IDENTICAL",
            },
            "canonical_paired_near_tie": canonical,
            "fixture_horizon": fixture_horizon,
            "decision_confidence": confidence,
            "suppression_reasons": suppression_reasons,
            "provenance": {
                **search_provenance,
                "search_artifact_cutoff": stage1_result.get("planning_cutoff"),
                "search_artifact_context_hash": stage1_result.get("planning_context_hash"),
                "search_supported_events": stage1_result.get("supported_events"),
                "refinement_cutoff": result.get("planning_cutoff"),
                "lineup_route_id": lineup_route_id,
                "lineup_basis": "CURRENT_GW_H1",
                # PE-9: the certification result is persisted WITH the decision, so the
                # authorisation this decision consumed is verifiable from the artifact
                # itself: the artifact identity, each event's canonical bundle identity,
                # the per-family run ids and model versions, the cutoff, the code and
                # data snapshot identities, the planning context hash, the certification
                # state per bundle, any calibration identity consulted, and the
                # unresolved disclosures.
                "certification": pe9_certification,
                "certification_result_identity": pe9_certification["certification_result_identity"],
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


def _escalation_runner(*, universe, initial_state, scenario, player_meta, bundles, conn,
                       base_config, stage1_result, stage2_draws, prebuilt_worlds,
                       finalist_partials, certification=None, non_production_worlds=None,
                       exact_cache=None, cancel_probe=None, parallel_workers=None):
    """The ONE bounded search-breadth escalation, as a closure over one beam width.

    Runs the next SUPPORTED search budget (the next beam width in
    ``route_stability.LADDER_BUDGETS``, never a draw-count change) at the Stage-2
    draw budget, in the SAME shared worlds, inheriting the Stage-1 nested
    survivors and forcing the refined finalists in so the two leaders are always
    comparable.  It never changes the objective, the pool, or the universe.

    The escalation re-scores in the SAME worlds Stage 2 used, which are the
    certified loader's own output, so it presents the same ``certification``
    artifact that authorised them.  A caller that supplies worlds from somewhere
    else must declare them through ``non_production_worlds``.
    """

    import dataclasses

    def run(beam: int):
        config = dataclasses.replace(
            rs.budget_config(int(beam), base_config), search_draws=int(stage2_draws)
        )
        return ro.optimize(
            universe=universe, initial_state=initial_state, scenario=scenario,
            player_meta=player_meta, bundles=bundles, conn=conn, config=config, cache_dir=None,
            non_production_worlds=non_production_worlds, certification=certification,
            prebuilt_worlds=prebuilt_worlds, required_routes=list(finalist_partials),
            nested_prior=ro.nested_budget_view(stage1_result),
            exact_cache=exact_cache, cancel_probe=cancel_probe,
            parallel_workers=parallel_workers,
        )

    return run


if __name__ == "__main__":
    raise SystemExit(main())
