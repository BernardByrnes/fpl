#!/usr/bin/env python3
"""R3: fresh rolling four-GW predictive certification under ONE cutoff.

One execution run (R2A controller) owns the whole certification:
one run UUID, one writer lease, a sequential stage ledger, a hard wall-clock
stop, and cooperative cancellation checkpoints before every event.

Predictive certification only.  No route search, no transfer recommendation, no
Wildcard evaluation, no transfer or chip execution.

Usage:
    python scripts/certify_four_gw.py --cutoff <UTC> --hard-stop-hours 4 \
        [--planning-event <official next event>] [--events <canonical window>]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
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
    generation_store as gs,
    history_completeness as hc,
)
from fpl_brain.config import config_path, load_config
from fpl_brain.database import connect_database, connect_readonly_database
from fpl_brain.planning import get_planning_context
from fpl_brain.utils import utc_now

import freeze_predictions as freeze

DEFAULT_OUT_DIR = Path("data/exports/four_gw")
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


def parse_event_list(value: str | Sequence[int] | None) -> tuple[int, ...] | None:
    """Parse an optional explicit event list without normalizing invalid input away."""

    if value is None:
        return None
    if isinstance(value, str):
        parts = value.split(",")
        if not value.strip() or any(not part.strip() for part in parts):
            raise ValueError("--events must be a non-empty comma-separated list of event IDs")
        try:
            events = tuple(int(part.strip()) for part in parts)
        except ValueError as failure:
            raise ValueError("--events must contain only integer event IDs") from failure
    else:
        try:
            events = tuple(value)
        except TypeError as failure:
            raise ValueError("events must be a non-empty sequence of event IDs") from failure
        if any(isinstance(event, bool) or not isinstance(event, int) for event in events):
            raise ValueError("events must contain only integer event IDs")
    if not events:
        raise ValueError("events must not be empty")
    if any(event < 1 for event in events):
        raise ValueError("event IDs must be positive integers")
    if len(set(events)) != len(events):
        raise ValueError("events must not contain duplicates")
    return events


def canonical_certification_events(
    planning_event: int,
    *,
    last_event: int,
    supplied_events: str | Sequence[int] | None = None,
) -> tuple[int, ...]:
    """Resolve and validate the exact frozen transfer horizon.

    The existing decision contract owns both the normal four-event window and its
    legitimate season-end shortening. An explicit list can confirm that result but
    can never choose a different horizon.
    """

    expected = fg.decision_events(planning_event, last_event=last_event)
    supplied = parse_event_list(supplied_events)
    if supplied is not None and supplied != expected:
        raise ValueError(
            f"events {list(supplied)} do not match the canonical horizon "
            f"for planning event {planning_event}: {list(expected)}"
        )
    return expected


def resolve_planning_event(conn, requested: int | None = None) -> int:
    """Resolve the official next event or validate the caller's matching event."""

    try:
        next_rows = conn.execute(
            "SELECT id FROM events WHERE is_next=1 ORDER BY id"
        ).fetchall()
    except sqlite3.Error as failure:
        raise ValueError(f"official next-event state is unavailable: {failure}") from failure
    next_events = tuple(int(row[0]) for row in next_rows)
    if len(next_events) > 1:
        raise ValueError(f"official schedule has multiple next events: {list(next_events)}")

    if requested is None:
        if len(next_events) != 1:
            raise ValueError("cannot resolve planning event: official schedule has no unique is_next event")
        return next_events[0]
    if isinstance(requested, bool) or not isinstance(requested, int) or requested < 1:
        raise ValueError("planning_event must be a positive integer")
    exists = conn.execute("SELECT 1 FROM events WHERE id=?", (int(requested),)).fetchone()
    if exists is None:
        raise ValueError(f"planning event {requested} is absent from the official schedule")
    if next_events and int(requested) != next_events[0]:
        raise ValueError(
            f"planning event {requested} disagrees with official is_next event {next_events[0]}"
        )
    return int(requested)


def assert_minutes_authority_consistency() -> str:
    """Fail before snapshotting or model execution if certification has drifted."""

    required_version = certified_bundle.declared_required_versions().get("minutes_v1")
    production_version = str(freeze.joint_minutes.JOINT_MINUTES_MODEL_VERSION)
    if not required_version or str(required_version) != production_version:
        raise ValueError(
            "MINUTES_LINEAGE_AUTHORITY_MISMATCH: declared certification minutes version "
            f"{required_version!r} does not match the frozen production lineage "
            f"joint_minutes.JOINT_MINUTES_MODEL_VERSION={production_version!r}"
        )
    return production_version


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
    """Compatibility entry point delegating to the shared decision-layer rule."""

    return fg.decide_search_permission(
        temporal_status=temporal_status,
        dependency_validation=dependency_validation,
        horizon_status=horizon_status,
        data_snapshot_sha256=data_snapshot_sha256,
        history_completeness=history_completeness,
        snapshot_error=snapshot_error,
    )


def certified_bundle_runs(
    conn,
    *,
    event: int,
    cutoff: str,
    execution_run_uuid: str | None = None,
) -> dict[str, int]:
    """Propose the exact predictive closure consumed by one Monte Carlo run.

    The Monte Carlo execution row is scoped to the current PE-10 execution when
    its UUID is supplied. Its recorded xPts id leads to the exact xPts dependency
    rows; the canonical bundle validator then rechecks every row and every edge.
    No run is selected by creation order.

    A zero-fixture event has no downstream rows from which to observe dependency
    ids. Only in that case, use the unique same-execution family rows, resolving
    minutes by the separately declared production version. The regular nonblank
    path always requires a complete observed dependency closure.
    """

    event = int(event)
    cutoff = str(cutoff)
    return _certified_bundle_runs_for_execution(
        conn,
        event=event,
        cutoff=cutoff,
        execution_run_uuid=execution_run_uuid,
    )


def _certified_bundle_runs_for_execution(
    conn,
    *,
    event: int,
    cutoff: str,
    execution_run_uuid: str | None,
) -> dict[str, int]:
    """Implementation shared by the public selector and PE-10 execution path."""

    clauses = [
        "model_family='monte_carlo_v1'",
        "planning_event=?",
        "data_cutoff=?",
    ]
    params: list[object] = [int(event), str(cutoff)]
    if execution_run_uuid is not None:
        clauses.append("execution_run_uuid=?")
        params.append(str(execution_run_uuid))
    mc_rows = conn.execute(
        "SELECT id FROM projection_runs WHERE " + " AND ".join(clauses),
        tuple(params),
    ).fetchall()
    if len(mc_rows) != 1:
        qualifier = f" for execution {execution_run_uuid}" if execution_run_uuid else ""
        detail = "no" if not mc_rows else "multiple"
        raise certified_bundle.BundleIncoherent(
            [f"{detail} unique Monte Carlo run for GW{event} at cutoff {cutoff}{qualifier}"]
        )
    mc_run = int(mc_rows[0]["id"])
    blank_event = certified_bundle.event_fixture_count(conn, event) == 0
    mc_upstream = certified_bundle._upstream_run_ids(conn, "monte_carlo_v1", mc_run)

    def unique_run(family: str, *, model_version: str | None = None) -> int:
        family_clauses = [
            "model_family=?",
            "planning_event=?",
            "data_cutoff=?",
        ]
        family_params: list[object] = [family, event, cutoff]
        if execution_run_uuid is not None:
            family_clauses.append("execution_run_uuid=?")
            family_params.append(str(execution_run_uuid))
        if model_version is not None:
            family_clauses.append("model_version=?")
            family_params.append(str(model_version))
        rows = conn.execute(
            "SELECT id FROM projection_runs WHERE " + " AND ".join(family_clauses),
            tuple(family_params),
        ).fetchall()
        if len(rows) != 1:
            raise certified_bundle.BundleIncoherent(
                [f"GW{event} has {len(rows)} candidate {family} run(s) in the blank-event execution closure"]
            )
        return int(rows[0]["id"])

    if not mc_upstream and blank_event:
        # Blank-event projection tables correctly contain no rows. Preserve the
        # explicit MC run, and identify its sibling rows within the same execution.
        required_minutes = certified_bundle.declared_required_versions()["minutes_v1"]
        return {
            "minutes_v1": unique_run("minutes_v1", model_version=required_minutes),
            "team_strength_v1": unique_run("team_strength_v1"),
            "player_rates_v1": unique_run("player_rates_v1"),
            "xpts_v1": unique_run("xpts_v1"),
            "monte_carlo_v1": mc_run,
        }

    needed_mc = ("xpts_v1", "minutes_v1", "team_strength_v1", "player_rates_v1")
    missing_mc = [family for family in needed_mc if mc_upstream.get(family) is None]
    if missing_mc:
        raise certified_bundle.BundleIncoherent(
            [f"Monte Carlo run {mc_run} has an incomplete dependency closure: missing {missing_mc}"]
        )

    xpts_run = int(mc_upstream["xpts_v1"])
    xpts_upstream = certified_bundle._upstream_run_ids(conn, "xpts_v1", xpts_run)
    if not xpts_upstream and blank_event:
        required_minutes = certified_bundle.declared_required_versions()["minutes_v1"]
        minutes_run = unique_run("minutes_v1", model_version=required_minutes)
        team_run = unique_run("team_strength_v1")
        rate_run = unique_run("player_rates_v1")
    else:
        needed_xpts = ("minutes_v1", "team_strength_v1", "player_rates_v1")
        missing_xpts = [family for family in needed_xpts if xpts_upstream.get(family) is None]
        if missing_xpts:
            raise certified_bundle.BundleIncoherent(
                [f"xPts run {xpts_run} has an incomplete dependency closure: missing {missing_xpts}"]
            )
        minutes_run = int(xpts_upstream["minutes_v1"])
        team_run = int(xpts_upstream["team_strength_v1"])
        rate_run = int(xpts_upstream["player_rates_v1"])

    return {
        "minutes_v1": minutes_run,
        "team_strength_v1": team_run,
        "player_rates_v1": rate_run,
        "xpts_v1": xpts_run,
        "monte_carlo_v1": mc_run,
    }


def certify_generation_under_lease(
    conn,
    *,
    controller,
    events: Sequence[int],
    cutoff: str,
    snapshot: Any,
    per_event: Mapping[str, Any],
    calibration: Mapping[str, Any] | None = None,
    calibration_artifact_ref: str | Path | None = None,
    require_calibration: bool = False,
) -> tuple[dict[str, dict], dict[str, str], dict[str, str], Any]:
    """Certify the horizon and persist ONE generation, under the run's leases.

    Returns ``(certified_bundles, bundle_identities, required_versions, generation)``.
    This is the PE-9 §8 lifecycle driven by the EXISTING controller: the writer lease
    is held by this run (the store refuses otherwise), the pinned snapshot and the
    authoritative code identity are validated by the store rather than asserted here,
    and the generation/pointer transaction commits only after every gate passed.
    """

    required_versions = certified_bundle.declared_required_versions()
    certified: dict[str, dict] = {}
    bundle_identity: dict[str, str] = {}
    for event in events:
        record = per_event.get(str(event)) or {}
        if not record.get("certified"):
            continue
        runs = certified_bundle_runs(
            conn,
            event=int(event),
            cutoff=str(cutoff),
            execution_run_uuid=str(controller.run_uuid),
        )
        bundle = certified_bundle.certified_bundle_from_explicit_ids(
            conn, event=int(event), cutoff=str(cutoff), runs=runs,
            required_versions=required_versions,
            data_snapshot_sha256=snapshot.data_snapshot_sha256,
            # The code identity is recorded ON the bundle payload as well as derived
            # by the generation store, so the identity a consumer recomputes from the
            # stored payload is exactly the one minted here.
            code_snapshot_sha256=analytics.source_snapshot_sha256(),
        )
        certified[str(event)] = bundle.as_dict()
        bundle_identity[str(event)] = bundle.bundle_identity()

    runs_by_event = {
        int(event): {str(family): int(run) for family, run in (row.get("runs") or {}).items()}
        for event, row in certified.items()
    }
    generation = gs.certify_generation(
        conn,
        planning_event=int(events[0]),
        cutoff=str(cutoff),
        events=[int(event) for event in events],
        runs_by_event=runs_by_event,
        # The PINNED snapshot the run replaced: the store re-hashes the bytes and
        # refuses a digest that does not reproduce, so the generation commits to the
        # source it actually used.
        snapshot={
            "path": snapshot.path,
            "sha256": snapshot.data_snapshot_sha256,
            "size_bytes": snapshot.size_bytes,
            "source_db_identity": snapshot.source_db_identity,
            "execution_run_uuid": snapshot.execution_run_uuid,
        },
        calibration=calibration,
        calibration_artifact_ref=calibration_artifact_ref,
        require_calibration=bool(require_calibration),
        controller=controller,
    )
    return certified, bundle_identity, required_versions, generation


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
    required_versions: Mapping[str, str] | None = None,
    code_snapshot_sha256: str | None = None,
) -> dict[str, Any]:
    """Construct the authoritative certification artifact payload.

    Extracted from ``main`` so the producer/consumer SEAM is testable: this is the
    exact construction the certification path performs, and its output must load
    through ``four_gw_decision.load_certification_artifact``.  ``decision_search_permitted``
    is deliberately seeded ``None`` and filled in only by the computed authorisation
    step, so it can never be a flag that is merely asserted.

    ``required_model_versions`` is the ONE declared source of the model versions a
    certification requires (``certified_bundle.declared_required_versions``).  It is
    recorded on the artifact so the consumer pins the same versions the producer
    did, and a run from an unexpected version cannot certify.
    """

    import hashlib

    declared_versions = {
        str(family): str(version)
        for family, version in (required_versions or certified_bundle.declared_required_versions()).items()
    }
    # ONE code fingerprint for the artifact AND for every certified bundle payload,
    # computed once: the consumer recomputes each event's bundle identity from the
    # payload's OWN declared snapshots, so if the artifact and the payload disagreed
    # the certified identity could not round-trip.
    code_snapshot = code_snapshot_sha256 or analytics.source_snapshot_sha256()
    return {
        "schema": fg.CERTIFICATION_ARTIFACT_SCHEMA,
        "execution_run_uuid": run_uuid,
        "planning_cutoff": planning_cutoff,
        "events": list(events),
        "data_snapshot_sha256": snapshot.data_snapshot_sha256,
        "data_snapshot_created_at": snapshot.created_at,
        "data_snapshot_path": snapshot.path,
        "data_snapshot_source_db_identity": snapshot.source_db_identity,
        "code_snapshot_sha256": code_snapshot,
        # Which code identity covered this certification, and the exact bytes of the
        # entry point whose wiring carries the history-completeness gate.  A v2
        # consumer refuses the artifact unless it declares the entry point covered,
        # so a certification minted without the gate cannot pass as current.
        "certification_wiring": fg.certification_wiring_identity(),
        # PE-9: the model versions this certification REQUIRES.  Declared here, from
        # the one source, so the bundle can never be certified at a version nobody
        # pinned.
        "required_model_versions": declared_versions,
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
    parser = argparse.ArgumentParser(description="Fresh rolling four-GW predictive certification")
    parser.add_argument(
        "--planning-event",
        type=int,
        help="planning event; defaults to the unique official events.is_next row",
    )
    parser.add_argument(
        "--cutoff",
        help="optional requested planning cutoff; must equal the snapshot consistency instant "
        "(otherwise HISTORICAL_SNAPSHOT_REQUIRED). Omit to mint it from the snapshot.",
    )
    parser.add_argument(
        "--events",
        help="optional comma-separated horizon assertion; must exactly match the canonical window",
    )
    parser.add_argument("--hard-stop-hours", type=float, default=4.0)
    parser.add_argument(
        "--first-event-simulations",
        "--gw5-simulations",
        dest="first_event_simulations",
        type=int,
        default=10_000,
        help="Monte Carlo draw budget for the first event in the horizon (legacy alias: --gw5-simulations)",
    )
    parser.add_argument("--later-simulations", type=int, default=2_000,
                        help="draw budget for later events in the horizon (same gates, smaller budget)")
    parser.add_argument("--config")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR),
                        help="base directory for event-scoped snapshots and artifacts")
    parser.add_argument("--out", help="optional certification run summary JSON path")
    parser.add_argument(
        "--pe8-calibration",
        help="optional retained PE-8 calibration artifact to bind to this exact generation",
    )
    parser.add_argument(
        "--require-pe8-calibration",
        action="store_true",
        help="refuse certification unless a matching retained PE-8 artifact is supplied",
    )
    args = parser.parse_args(argv)

    try:
        assert_minutes_authority_consistency()
    except ValueError as failure:
        print(f"certification refused: {failure}", file=sys.stderr)
        return 2

    calibration = None
    calibration_artifact_ref = None
    if args.pe8_calibration:
        calibration_artifact_ref = Path(args.pe8_calibration)
        try:
            calibration = json.loads(calibration_artifact_ref.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as failure:
            print(f"certification refused: PE-8 artifact cannot be read: {failure}", file=sys.stderr)
            return 2
        if not isinstance(calibration, Mapping):
            print("certification refused: PE-8 artifact root must be an object", file=sys.stderr)
            return 2
    if args.require_pe8_calibration and calibration is None:
        print(
            "certification refused: --require-pe8-calibration needs --pe8-calibration",
            file=sys.stderr,
        )
        return 2

    try:
        supplied_events = parse_event_list(args.events)
    except ValueError as failure:
        print(f"certification refused: {failure}", file=sys.stderr)
        return 2

    # Preflight: a REQUESTED cutoff may not post-date this execution.  The effective
    # cutoff is minted from the snapshot consistency instant below.
    if args.cutoff:
        try:
            causality.assert_causal_cutoff(
                str(args.cutoff), utc_now(), label="four-GW certification preflight"
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

    config = load_config(args.config)
    conn = connect_database(config_path(config, "database"))
    try:
        planning_event = resolve_planning_event(conn, args.planning_event)
        events = canonical_certification_events(
            planning_event,
            last_event=fg.season_last_event_from_db(conn),
            supplied_events=supplied_events,
        )
    except ValueError as failure:
        print(f"certification refused: {failure}", file=sys.stderr)
        conn.close()
        return 2

    run_dir = Path(args.out_dir) / f"gw{planning_event:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = run_dir / "predictions"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.out) if args.out else run_dir / "certification_run.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    args.out = str(output_path)

    document: dict = {
        "phase": "R3",
        "planning_event": planning_event,
        "planning_cutoff": str(args.cutoff) if args.cutoff else None,
        "events": events,
        "predictive_certification_only": True,
        "route_search_run": False,
        "wildcard_evaluated": False,
        "transfers_or_chips_executed": 0,
        "started_at": utc_now(),
    }

    # --- clean production sequence ------------------------------------------
    # 1. generate the execution UUID WITHOUT beginning predictive execution
    provisional_uuid = str(__import__("uuid").uuid4())
    controller = execution.ExecutionController(conn, run_uuid=provisional_uuid)
    # 2. capture the immutable DB snapshot
    snapshot_dir = run_dir / "snapshots" / provisional_uuid
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
        label="four_gw_predictive_certification",
        families=["four_gw_certification"],
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
            simulations = args.first_event_simulations if event == events[0] else args.later_simulations
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
                        pinned_snapshot=snapshot,
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

        # --- PE-9: certify the horizon INTO the generation store ----------------
        # The generation is written UNDER the run's writer lease, inside the run,
        # because a certified generation is production evidence: it must belong to
        # the run that produced it, and validation plus the generation/pointer
        # transaction must not interleave with another writer.  A generation row
        # exists only when certification passed, so a failure here leaves the run
        # FAILED with no generation and the old pointer.
        with controller.stage("CERTIFY_GENERATION", detail={"events": list(events)}):
            controller.check_cancel()
            certified, bundle_identity, required_versions, generation = certify_generation_under_lease(
                conn,
                controller=controller,
                events=events,
                cutoff=effective_cutoff,
                snapshot=snapshot,
                per_event=per_event,
                calibration=calibration,
                calibration_artifact_ref=calibration_artifact_ref,
                require_calibration=args.require_pe8_calibration,
            )
        document["generation_id"] = generation.generation_id
        document["generation_manifest_sha256"] = generation.generation_id
        print(
            f"PE-9 certified generation: id={generation.generation_id[:24]}… "
            f"events={list(generation.events)} snapshot={str(generation.snapshot.get('sha256'))[:16]}…"
        )
    except execution.RunCancelled:
        controller.acknowledge_cancel()
        document["status"] = "CANCELLED"
        raise
    except certified_bundle.BundleIncoherent as failure:
        # A bundle that cannot be certified is RECORDED and reported, exactly as
        # before: the run is FAILED and no generation is written.
        controller.finish(execution.RUN_FAILED, f"BundleIncoherent: {'; '.join(failure.reasons)}")
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
    except (gs.GenerationRefused, certified_bundle.CertificationRefused) as failure:
        controller.finish(execution.RUN_FAILED, str(failure))
        document["status"] = "FAILED"
        document["certification_refusal"] = str(failure)
        Path(args.out).write_text(
            json.dumps(document, indent=2, sort_keys=True, default=str) + chr(10), encoding="utf-8"
        )
        print(f"certification refused: {failure}", file=sys.stderr)
        conn.close()
        return 2
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

    # ONE code fingerprint for the artifact and for every bundle payload (the same
    # value the generation store derives internally), so the identity a consumer
    # recomputes from the payload matches the one recorded here.
    certification_code_snapshot = analytics.source_snapshot_sha256()

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
        planning = get_planning_context(source_conn, int(config["fpl_entry_id"]), planning_event, as_of=effective_cutoff,
                                        season=config.get("season"))
        manager_state = {
            "event": planning_event,
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
        required_versions=required_versions,
        code_snapshot_sha256=certification_code_snapshot,
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
    # PE-9: certify the horizon AS A HORIZON, from THIS one artifact, and record the
    # per-bundle states and the phase terminal state.  No calibration evidence exists
    # at mint time (PE-8 consumes this artifact), so the calibration claim is not made
    # here -- it is made, or declined, at the decision boundary where the evidence is
    # available.  That is a state, never a refusal and never a pass.
    try:
        pe9 = certified_bundle.certify_decision_horizon(
            conn,
            certification=artifact,
            events=events,
            cutoff=effective_cutoff,
            required_versions=required_versions,
        )
    except certified_bundle.CertificationRefused as failure:
        document["status"] = "FAILED"
        document["certification_refusal"] = f"{failure.token}: {'; '.join(failure.reasons)}"
        Path(args.out).write_text(
            json.dumps(document, indent=2, sort_keys=True, default=str) + chr(10), encoding="utf-8"
        )
        print(f"certification refused: {failure}", file=sys.stderr)
        source_conn.close()
        conn.close()
        return 2
    pe9["certification_result_identity"] = certified_bundle.certification_result_identity(pe9)
    artifact["pe9_certification"] = pe9
    print(
        f"PE-9 certification: horizon={pe9['horizon_state']} phase={pe9['phase_terminal_state']} "
        f"identity={pe9['certification_result_identity'][:24]}…"
    )
    artifact_path = run_dir / "certification_artifact.json"
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
