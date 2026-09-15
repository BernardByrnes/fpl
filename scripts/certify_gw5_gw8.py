#!/usr/bin/env python3
"""R3: fresh GW5-GW8 predictive certification under ONE cutoff.

One execution run (R2A controller) owns the whole certification:
one run UUID, one writer lease, a sequential stage ledger, a hard wall-clock
stop, and cooperative cancellation checkpoints before every event.

Predictive certification only.  No route search, no transfer recommendation, no
Wildcard evaluation, no transfer or chip execution.

Usage:
    python scripts/certify_gw5_gw8.py --cutoff <UTC> --hard-stop-hours 4 \
        [--events 5,6,7,8] [--later-simulations 2000]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fpl_brain import (
    analytics,
    causality,
    certified_bundle,
    execution,
    execution_snapshot,
    four_gw_decision as fg,
    history_completeness as hc,
)
from fpl_brain.config import config_path, load_config
from fpl_brain.database import connect_database, connect_readonly_database
from fpl_brain.planning import get_planning_context
from fpl_brain.utils import utc_now

import freeze_predictions as freeze

OUT_DIR = Path("data/exports/four_gw/gw05")
SUBSTANTIVE_FAMILIES = (
    "baseline",
    "minutes",
    "minutes_coherent",
    "minutes_positional",
    "minutes_substitution",
    "minutes_joint",
    "team",
    "rates",
    "xpts",
    "monte_carlo",
)


class _EventFreezeFailed(RuntimeError):
    """A single event's freeze failed its readiness gate (recorded, not fatal)."""

    def __init__(self, message: str, code: int) -> None:
        super().__init__(message)
        self.code = int(code)


def _freeze_args(event: int, cutoff: str, simulations: int, out_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        gw=int(event),
        cutoff=str(cutoff),
        families="all",
        dry_run=False,
        mc_simulations=int(simulations),
        mc_calibration_states=20_000,
        mc_calibration_iterations=250,
        out_dir=str(out_dir),
        verbose=False,
        config=None,
        xpts_minutes_run=None,
        xpts_team_run=None,
        xpts_team_baseline_run=None,
        xpts_rate_run=None,
        xpts_run=None,
    )


def decide_search_permission(
    *,
    temporal_status: Any,
    dependency_validation: Any,
    horizon_status: Any,
    data_snapshot_sha256: Any,
    history_completeness: Mapping[str, Any],
    snapshot_error: str | None = None,
) -> tuple[bool, list[str]]:
    """Authorisation is COMPUTED from the conditions, never asserted.

    Extracted from ``main`` so the rule is executable from a test.  Its earlier
    inline form read an undefined local (``horizon_status``) and was only ever
    checked by a source-text assertion, so the defect survived the accepted suite
    while making the certification artifact impossible to write.

    ``history_completeness`` is REQUIRED -- it is the audit from
    ``history_completeness.audit_history_completeness`` evaluated against the SAME
    immutable snapshot -- so a caller cannot obtain permission merely by
    forgetting to evaluate the gate.  An incomplete audit WITHHOLDS permission:
    a bundle may never be certified as fresh while an officially completed
    event's required player history is missing.  The canonical token is always in
    the reason so an operator can grep the artifact.
    """

    reasons: list[str] = []
    if str(temporal_status).upper() != "CAUSAL":
        reasons.append("temporal_status is not CAUSAL")
    if str(dependency_validation).upper() != "COHERENT":
        reasons.append("dependency_validation is not COHERENT")
    if horizon_status != fg.DECISION_HORIZON_COMPLETE:
        reasons.append(f"horizon status is {horizon_status}")
    if not data_snapshot_sha256:
        reasons.append("no data snapshot identity")
    if snapshot_error:
        reasons.append(str(snapshot_error))
    if not history_completeness.get("complete"):
        blocker = hc.blocking_reason_token(history_completeness)
        detail = history_completeness.get("reasons") or []
        reasons.append(blocker if not detail else f"{blocker} ({', '.join(str(item) for item in detail)})")
    return (not reasons), reasons


def certified_horizon_status(conn, artifact, events, cutoff) -> str:
    """Four-GW horizon status from the SAME certified bundle identity the runner reads.

    ``event_support_from_certification`` refuses a missing bundle and proves the
    dependency DAG, so a status returned here is about the certified generation the
    decision engine will actually consume.  Callers fail closed on an exception.
    """

    support = fg.event_support_from_certification(conn, artifact, events=events, cutoff=cutoff)
    horizon = fg.evaluate_horizon(
        planning_event=int(events[0]),
        support_by_event=support,
        cutoff=str(cutoff),
        last_event=fg.season_last_event_from_db(conn),
    )
    return str(horizon["status"])


def build_certification_artifact(
    *,
    run_uuid: str,
    planning_cutoff: str,
    events: Sequence[int],
    snapshot: Any,
    certified: Mapping[str, Any],
    bundle_identity: Mapping[str, str],
    manager_state: Mapping[str, Any],
    model_versions: Sequence[tuple[str, str]],
    execution_started_at: Any,
) -> dict[str, Any]:
    """Construct the authoritative certification artifact payload.

    Extracted from ``main`` so the producer/consumer SEAM is testable: this is the
    exact construction the certification path performs, and its output must load
    through ``four_gw_decision.load_certification_artifact``.  ``decision_search_permitted``
    is deliberately seeded ``None`` and filled in only by the computed authorisation
    step, so it can never be a flag that is merely asserted.
    """

    import hashlib

    return {
        "schema": fg.CERTIFICATION_ARTIFACT_SCHEMA,
        "execution_run_uuid": run_uuid,
        "planning_cutoff": planning_cutoff,
        "events": list(events),
        "data_snapshot_sha256": snapshot.data_snapshot_sha256,
        "data_snapshot_created_at": snapshot.created_at,
        "data_snapshot_path": snapshot.path,
        "data_snapshot_source_db_identity": snapshot.source_db_identity,
        "code_snapshot_sha256": analytics.source_snapshot_sha256(),
        # Which code identity covered this certification, and the exact bytes of the
        # entry point whose wiring carries the history-completeness gate.  A v2
        # consumer refuses the artifact unless it declares the entry point covered,
        # so a certification minted without the gate cannot pass as current.
        "certification_wiring": fg.certification_wiring_identity(),
        "certified_bundles": certified,
        "certified_bundle_identity": bundle_identity,
        "four_gw_certification_identity": "sha256:" + hashlib.sha256(
            json.dumps(
                {"cutoff": planning_cutoff, "bundles": bundle_identity,
                 "data_snapshot_sha256": snapshot.data_snapshot_sha256},
                sort_keys=True, separators=(",", ":"),
            ).encode()
        ).hexdigest(),
        "manager_state_identity": manager_state,
        "model_versions": [{"model_family": m, "model_version": v} for m, v in model_versions],
        "dependency_validation": "COHERENT",
        "dependency_validation_detail": None,
        "temporal_status": "CAUSAL",
        "temporal_detail": {
            "rule": "planning_cutoff <= data_snapshot_created_at <= execution_started_at_utc (+/- skew)",
            "planning_cutoff": planning_cutoff,
            "data_snapshot_created_at": snapshot.created_at,
            "execution_started_at": execution_started_at,
        },
        "live_source_drift": execution_snapshot.live_source_drift(snapshot),
        # FACTUAL execution fields: what this certification did or did not do.
        "route_search_executed": False,
        "transfer_execution_performed": False,
        # AUTHORIZATION field the decision runner must check.  True only when the
        # four conditions below hold; it is deliberately separate from the factual
        # fields so it cannot be a flag that is ignored while search proceeds.
        "decision_search_permitted": None,
        "decision_search_permitted_reasons": [],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fresh GW5-GW8 predictive certification")
    parser.add_argument(
        "--cutoff",
        help="optional requested planning cutoff; must equal the snapshot consistency instant "
        "(otherwise HISTORICAL_SNAPSHOT_REQUIRED). Omit to mint it from the snapshot.",
    )
    parser.add_argument("--events", default="5,6,7,8")
    parser.add_argument("--hard-stop-hours", type=float, default=4.0)
    parser.add_argument("--gw5-simulations", type=int, default=10_000)
    parser.add_argument("--later-simulations", type=int, default=2_000,
                        help="horizon budget for GW6-8 (same gates, smaller draw budget)")
    parser.add_argument("--config")
    parser.add_argument("--out", default=str(OUT_DIR / "gw5_gw8_certification.json"))
    args = parser.parse_args(argv)

    events = [int(part) for part in str(args.events).split(",") if part.strip()]
    if events != [5, 6, 7, 8]:
        print(f"refusing: certification window must be exactly GW5-GW8, got {events}", file=sys.stderr)
        return 2

    # Preflight: a REQUESTED cutoff may not post-date this execution.  The effective
    # cutoff is minted from the snapshot consistency instant below.
    if args.cutoff:
        try:
            causality.assert_causal_cutoff(
                str(args.cutoff), utc_now(), label="gw5-gw8 certification preflight"
            )
        except causality.PlanningCutoffInFuture as failure:
            print(f"certification refused: {failure}", file=sys.stderr)
            return 2
        # Early historical-cutoff refusal: the consistency instant is always at or
        # after "now", so a requested cutoff older than that can never match it.
        # Refusing here avoids capturing a snapshot that is certain to be rejected.
        _now = causality._aware(utc_now(), label="now")
        _requested = causality._aware(str(args.cutoff), label="requested cutoff")
        if (_now - _requested).total_seconds() > execution_snapshot.CUTOFF_MATCH_TOLERANCE_SECONDS:
            print(
                "certification refused: HISTORICAL_SNAPSHOT_REQUIRED: requested cutoff "
                f"{args.cutoff} predates the current instant {utc_now()}; a fresh live snapshot "
                "cannot represent it. Supply an existing immutable snapshot for that cutoff.",
                file=sys.stderr,
            )
            return 2

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    artifact_dir = OUT_DIR / "predictions"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    conn = connect_database(config_path(config, "database"))
    document: dict = {
        "phase": "R3",
        "planning_cutoff": str(args.cutoff) if args.cutoff else None,
        "events": events,
        "predictive_certification_only": True,
        "route_search_run": False,
        "wildcard_evaluated": False,
        "transfers_or_chips_executed": 0,
        "started_at": utc_now(),
    }

    controller = execution.ExecutionController(conn)
    # --- clean production sequence ------------------------------------------
    # 1. generate the execution UUID WITHOUT beginning predictive execution
    provisional_uuid = str(__import__("uuid").uuid4())
    # 2. capture the immutable DB snapshot
    snapshot_dir = OUT_DIR / "snapshots" / provisional_uuid
    snapshot = execution_snapshot.capture_execution_snapshot(
        config_path(config, "database"),
        directory=snapshot_dir,
        execution_run_uuid=provisional_uuid,
        clock=controller.now_dt,
    )
    # 3/4. the LIVE cutoff rule: planning_cutoff == snapshot_consistency_at
    try:
        effective_cutoff = execution_snapshot.require_live_cutoff_matches_snapshot(
            snapshot.snapshot_consistency_at, args.cutoff
        )
    except execution_snapshot.SnapshotError as failure:
        print(f"certification refused: {failure}", file=sys.stderr)
        return 2
    print(
        f"execution snapshot: {snapshot.path} sha256={snapshot.data_snapshot_sha256[:16]}… "
        f"started={snapshot.snapshot_capture_started_at} consistency={snapshot.snapshot_consistency_at} "
        f"completed={snapshot.snapshot_capture_completed_at} window="
        f"{execution_snapshot.snapshot_consistency_window(snapshot):.3f}s bytes={snapshot.size_bytes}"
    )
    # 5/6. now start the controller and create the run row with the minted cutoff
    run_identity = controller.create_run(
        planning_event=events[0],
        planning_cutoff=effective_cutoff,
        hard_stop_at=execution.add_seconds(
            controller.now_dt(), float(args.hard_stop_hours) * 3600.0
        ),
        label="gw5_gw8_predictive_certification",
        families=["gw5_gw8_certification"],
    )
    controller.start()
    source_conn = execution_snapshot.open_snapshot(snapshot)
    document["execution_run_uuid"] = run_identity.run_uuid
    document.update(snapshot.as_dict())
    document["hard_stop_at"] = run_identity.hard_stop_at
    document["owner_pid"] = run_identity.owner_pid
    document["owner_host"] = run_identity.owner_host
    print(f"execution run_uuid={run_identity.run_uuid} pid={run_identity.owner_pid}")
    print(f"cutoff={effective_cutoff} hard_stop={run_identity.hard_stop_at}")

    per_event: dict[str, dict] = {}
    horizon: dict[str, dict] = {}
    try:
        run_lease = controller.acquire_run_lease()
        writer_lease = controller.acquire_writer_lease()
        document["run_lease_id"] = run_lease
        document["writer_lease_id"] = writer_lease
        print(f"leases held: run={run_lease} writer={writer_lease}")

        for event in events:
            simulations = args.gw5_simulations if event == 5 else args.later_simulations
            stage_name = f"FREEZE_GW{event}"
            started = time.time()
            try:
                with controller.stage(stage_name, detail={"simulations": simulations}):
                    controller.check_cancel()  # before each event
                    controller.heartbeat()
                    context = get_planning_context(
                        conn, int(config["fpl_entry_id"]), event,
                        as_of=effective_cutoff, season=config.get("season"),
                    )
                    horizon[str(event)] = {
                        "planning_health": context.health.get("status"),
                        "health_fail_reasons": context.health.get("fail_reasons"),
                        "health_warn_reasons": context.health.get("warn_reasons"),
                        "manager_state_source": (context.manager_state or {}).get("authoritative_source"),
                    }
                    if context.health.get("status") == "FAIL":
                        raise _EventFreezeFailed(
                            f"GW{event} planning health FAIL: {context.health.get('fail_reasons')}", 2
                        )
                    execution_snapshot.assert_snapshot_unchanged(snapshot)
                    code = freeze._freeze(
                        conn, config, _freeze_args(event, effective_cutoff, simulations, artifact_dir),
                        set(SUBSTANTIVE_FAMILIES), source_conn=source_conn, production=True,
                    )
                    if code != 0:
                        raise _EventFreezeFailed(f"GW{event} freeze returned {code}", code)
            except _EventFreezeFailed as failure:
                # Recorded and reported; every event is still attempted so the
                # horizon evaluation is complete rather than truncated.
                per_event[str(event)] = {
                    "stage": stage_name,
                    "simulations": simulations,
                    "seconds": round(time.time() - started, 1),
                    "freeze_return_code": failure.code,
                    "certified": False,
                    "failure": str(failure),
                }
                print(f"GW{event} NOT certified: {failure}", file=sys.stderr)
                continue
            per_event[str(event)] = {
                "stage": stage_name,
                "simulations": simulations,
                "seconds": round(time.time() - started, 1),
                "freeze_return_code": 0,
                "certified": True,
            }
            print(f"GW{event} certified fresh at cutoff {effective_cutoff}")
    except execution.RunCancelled:
        controller.acknowledge_cancel()
        document["status"] = "CANCELLED"
        raise
    except BaseException as exc:
        controller.finish(execution.RUN_FAILED, f"{type(exc).__name__}: {exc}")
        document["status"] = "FAILED"
        document["failure_reason"] = f"{type(exc).__name__}: {exc}"
        document["events_completed"] = sorted(per_event)
        Path(args.out).write_text(
            json.dumps(document, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
        )
        raise
    else:
        controller.finish(execution.RUN_COMPLETE)

    # Post-run: certify each event's families as ONE coherent predictive bundle and
    # record the certified bundle identity for the decision engine to consume.
    certified: dict[int, dict] = {}
    bundle_identity: dict[int, str] = {}
    try:
        for event in events:
            record = per_event.get(str(event)) or {}
            if not record.get("certified"):
                continue
            rows = conn.execute(
                "SELECT model_family, id FROM projection_runs WHERE planning_event=?"
                " AND data_cutoff=? AND model_family IN"
                " ('minutes_v1','team_strength_v1','player_rates_v1','xpts_v1','monte_carlo_v1')"
                " ORDER BY id DESC",
                (event, effective_cutoff),
            ).fetchall()
            # latest-per-family is only used to PROPOSE a bundle; validation then
            # proves the edges actually agree, which is what "latest" cannot do.
            runs: dict[str, int] = {}
            for row in rows:
                runs.setdefault(str(row["model_family"]), int(row["id"]))
            bundle = certified_bundle.certified_bundle_from_explicit_ids(
                conn, event=event, cutoff=effective_cutoff, runs=runs
            )
            certified[str(event)] = bundle.as_dict()
            bundle_identity[str(event)] = bundle.bundle_identity()
    except certified_bundle.BundleIncoherent as failure:
        document["status"] = "FAILED"
        document["predictive_bundle_status"] = certified_bundle.DIAG_PREDICTIVE_BUNDLE_INCOHERENT
        document["predictive_bundle_reasons"] = failure.reasons
        document["route_search_permitted"] = False
        Path(args.out).write_text(
            json.dumps(document, indent=2, sort_keys=True, default=str) + chr(10), encoding="utf-8"
        )
        print(f"certification refused: {failure}", file=sys.stderr)
        conn.close()
        return 2

    document["status"] = "COMPLETE"
    document["predictive_bundle_status"] = "PREDICTIVE_BUNDLE_COHERENT"
    document["certified_bundles"] = certified
    document["certified_bundle_identity"] = bundle_identity
    document["route_search_permitted"] = False

    # --- the authoritative certification artifact ---------------------------
    # This is the ONLY thing the decision runner consumes.  It names the exact run
    # ids, so the runner never rediscovers a run.
    execution_snapshot.assert_snapshot_unchanged(snapshot)
    manager_state = {}
    try:
        planning = get_planning_context(source_conn, int(config["fpl_entry_id"]), 5, as_of=effective_cutoff,
                                        season=config.get("season"))
        manager_state = {
            "event": 5,
            "free_transfers": (planning.manager_state or {}).get("free_transfers"),
            "event_start_free_transfers": (planning.manager_state or {}).get("event_start_free_transfers"),
            "bank_tenths": (planning.manager_state or {}).get("bank"),
            "authoritative_source": (planning.manager_state or {}).get("authoritative_source"),
            "health": planning.health.get("status"),
        }
    except Exception as exc:  # provenance must not be silently absent
        manager_state = {"error": f"{type(exc).__name__}: {exc}"}
    model_versions = sorted(
        {
            (row[0], row[1])
            for row in conn.execute(
                "SELECT model_family, model_version FROM projection_runs WHERE id>160"
                " GROUP BY model_family, model_version"
            )
        }
    )
    artifact = build_certification_artifact(
        run_uuid=run_identity.run_uuid,
        planning_cutoff=effective_cutoff,
        events=events,
        snapshot=snapshot,
        certified=certified,
        bundle_identity=bundle_identity,
        manager_state=manager_state,
        model_versions=model_versions,
        execution_started_at=run_identity.started_at,
    )
    # Authorisation is computed, never asserted.
    try:
        horizon_status = certified_horizon_status(conn, artifact, events, effective_cutoff)
    except Exception as failure:  # unresolvable horizon withholds, never crashes
        horizon_status = f"UNRESOLVED: {type(failure).__name__}: {failure}"
    snapshot_error = None
    try:
        execution_snapshot.assert_snapshot_unchanged(snapshot)
    except execution_snapshot.SnapshotError as failure:
        snapshot_error = str(failure)
    # Required completed-event history, audited on the SAME immutable snapshot the
    # predictions were generated from.  An unevaluable invariant WITHHOLDS rather
    # than crashes, and an incomplete one carries the canonical blocker token.
    try:
        history_audit: dict[str, Any] = hc.audit_history_completeness(
            source_conn, planning_event=int(events[0]), cutoff=effective_cutoff
        )
    except Exception as failure:  # noqa: BLE001 - withholding is the contract
        history_audit = {
            "schema": hc.HISTORY_COMPLETENESS_SCHEMA,
            "planning_event": int(events[0]),
            "cutoff": str(effective_cutoff),
            "complete": False,
            "blocker": hc.CERTIFIED_PREDICTION_INPUT_HISTORY_INCOMPLETE,
            "reasons": [f"UNRESOLVED: {type(failure).__name__}: {failure}"],
            "detail": "the history-completeness audit could not be evaluated",
        }
    permitted, permit_reasons = decide_search_permission(
        temporal_status=artifact["temporal_status"],
        dependency_validation=artifact["dependency_validation"],
        horizon_status=horizon_status,
        data_snapshot_sha256=artifact["data_snapshot_sha256"],
        snapshot_error=snapshot_error,
        history_completeness=history_audit,
    )
    artifact["decision_search_permitted"] = permitted
    artifact["decision_search_permitted_reasons"] = permit_reasons
    artifact["decision_search_horizon_status"] = horizon_status
    artifact["history_completeness"] = history_audit
    artifact_path = OUT_DIR / "certification_artifact.json"
    artifact_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True, default=str) + chr(10), encoding="utf-8"
    )
    document["certification_artifact"] = str(artifact_path)
    document["four_gw_certification_identity"] = artifact["four_gw_certification_identity"]
    document["live_source_drift"] = artifact["live_source_drift"]
    print(f"certification artifact: {artifact_path}")
    print(f"four-GW certification identity: {artifact['four_gw_certification_identity'][:24]}…")
    document["per_event"] = per_event
    document["planning"] = horizon
    document["finished_at"] = utc_now()
    document["stage_ledger"] = [
        {
            "stage": row["stage"],
            "status": row["status"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
            "detail": row["detail_json"],
        }
        for row in controller.stages()
    ]
    Path(args.out).write_text(
        json.dumps(document, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    print(f"certification run complete: {args.out}")
    try:
        source_conn.close()
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
