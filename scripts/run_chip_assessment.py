#!/usr/bin/env python3
"""Run and immutably retain one verified four-chip production assessment.

The database is opened read-only. The entry point refuses before loading worlds
or optimizing routes unless the manager confirmation is present in the normal
generation's pinned snapshot and predates its cutoff.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fpl_brain import chip_assessment, chip_assessment_store, chip_decision
from fpl_brain import chip_reservation_calibration
from fpl_brain import chip_evaluator_readiness
from fpl_brain import generation_store, repositories


def _pinned_generation_position_resolver(conn: sqlite3.Connection):
    cache: dict[str, dict[str, str]] = {}
    cache_identity: dict[str, tuple[int, str, str]] = {}

    def resolve(outcome_record: dict) -> dict[str, str]:
        generation_id = str(outcome_record.get("generation_id") or "")
        if not generation_id:
            raise ValueError("outcome record has no source generation id")
        wanted_identity = (
            int(outcome_record.get("planning_event") or -1),
            str(outcome_record.get("cutoff") or ""),
            str(outcome_record.get("data_snapshot_sha256") or ""),
        )
        if generation_id in cache_identity and cache_identity[generation_id] != wanted_identity:
            raise ValueError("outcome source generation does not match its event, cutoff or snapshot")
        if generation_id not in cache:
            generation = generation_store.load_generation(conn, generation_id)
            if (
                generation.horizon_kind != generation_store.HORIZON_KIND_FOUR_GW
                or generation.planning_event != int(outcome_record.get("planning_event") or -1)
                or generation.cutoff != str(outcome_record.get("cutoff") or "")
                or str(generation.snapshot.get("sha256") or "")
                != str(outcome_record.get("data_snapshot_sha256") or "")
            ):
                raise ValueError("outcome source generation does not match its event, cutoff or snapshot")
            report = generation_store.verify_generation(conn, generation_id)
            if not report.get("verified"):
                raise ValueError("outcome source generation is not verified")
            snapshot_conn = generation_store._open_generation_snapshot(generation)
            try:
                rows = repositories.player_candidates(snapshot_conn)
            finally:
                snapshot_conn.close()
            all_positions = {
                str(int(row["id"])): str(row.get("position_short_name") or "")
                for row in rows
                if row.get("position_short_name")
            }
            cache[generation_id] = all_positions
            cache_identity[generation_id] = wanted_identity

        paired = outcome_record.get("paired_results") or {}
        if not isinstance(paired, dict) or set(paired) != {"play", "save"}:
            raise ValueError("outcome record has no complete canonical PLAY/SAVE arms")
        player_ids: set[int] = set()
        for arm_name in ("play", "save"):
            arm = paired.get(arm_name)
            if not isinstance(arm, dict):
                raise ValueError(f"outcome record has no canonical {arm_name.upper()} arm")
            try:
                squad = [int(value) for value in arm["proposed_squad_ids"]]
            except (KeyError, TypeError, ValueError) as failure:
                raise ValueError(
                    f"outcome record {arm_name.upper()} arm has no complete proposed squad"
                ) from failure
            if len(squad) != 15 or len(set(squad)) != 15:
                raise ValueError(
                    f"outcome record {arm_name.upper()} arm does not contain 15 unique players"
                )
            player_ids.update(squad)
        generation_positions = cache[generation_id]
        if any(str(player_id) not in generation_positions for player_id in player_ids):
            raise ValueError("pinned generation lacks positions for a complete PLAY/SAVE policy")
        return {
            str(player_id): generation_positions[str(player_id)] for player_id in sorted(player_ids)
        }

    return resolve


def _calibration_evidence_loader(
    evidence_root: Path,
    conn: sqlite3.Connection,
    *,
    source_position_resolver=None,
):
    root = evidence_root.resolve()
    resolve_positions = source_position_resolver or _pinned_generation_position_resolver(conn)

    def load(reference: str) -> dict:
        evidence_path = (root / str(reference)).resolve()
        if not evidence_path.is_relative_to(root):
            raise ValueError("causal evidence reference escapes its configured root")
        value = json.loads(evidence_path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("causal evidence artifact must be a JSON object")
        if value.get("schema") == chip_reservation_calibration.CAUSAL_OUTCOME_RECORD_SCHEMA:
            chip_reservation_calibration.verify_outcome_record_capture_sources(
                conn, value, source_position_resolver=resolve_positions,
            )
        return value

    return load


def _load_reservation_forecast_inputs(
    forecast_root: Path,
    conn: sqlite3.Connection,
) -> tuple[dict[str, dict], Any]:
    """Read explicit action files or one unambiguous retained content address."""

    root = forecast_root.resolve()
    verifier = _calibration_evidence_loader(root, conn)
    forecasts: dict[str, dict] = {}
    for action in chip_decision.PLAYABLE_CHIP_ACTIONS:
        explicit = root / f"{action}.json"
        if explicit.is_file():
            candidates = [explicit]
        else:
            candidates = []
            for path in sorted(root.glob("chip-reservation-forecast-*.json")):
                raw = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(raw, dict) and raw.get("action") == action:
                    candidates.append(path)
        identities: dict[str, tuple[Path, dict]] = {}
        for path in candidates:
            if not path.resolve().is_relative_to(root):
                raise ValueError("reservation forecast path escapes its configured root")
            raw = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError(f"reservation forecast must be a JSON object: {path}")
            identity = str(raw.get("artifact_sha256") or path.name)
            if identity in identities and identities[identity][1] != raw:
                raise ValueError(f"different reservation forecast bytes claim the same digest: {identity}")
            identities[identity] = (path, raw)
        if len(identities) > 1:
            raise ValueError(f"multiple retained reservation forecasts found for {action}; select one explicitly")
        if identities:
            _path, raw = next(iter(identities.values()))
            forecasts[action] = raw
    return forecasts, verifier


def _load_evaluator_readiness_inputs(
    readiness_root: Path,
    evidence_root: Path,
    conn: sqlite3.Connection,
) -> tuple[dict[str, dict], Any]:
    """Load one action-specific readiness record and its verified evidence reader."""

    root = readiness_root.resolve()
    verifier = _calibration_evidence_loader(evidence_root, conn)
    artifacts: dict[str, dict] = {}
    for action in (
        chip_decision.CHIP_ACTION_BB,
        chip_decision.CHIP_ACTION_FH,
        chip_decision.CHIP_ACTION_WC,
    ):
        path = root / f"{action}.json"
        if not path.is_file():
            continue
        if not path.resolve().is_relative_to(root):
            raise ValueError("evaluator readiness path escapes its configured root")
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"evaluator readiness artifact must be a JSON object: {path}")
        chip_evaluator_readiness.verify_retained_readiness_integrity(raw)
        if raw.get("action") != action:
            raise ValueError(f"evaluator readiness artifact action differs from its filename: {path}")
        artifacts[action] = raw
    return artifacts, verifier


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path, help="existing runtime database (read only)")
    parser.add_argument("--decision-id", required=True)
    parser.add_argument("--entry-id", required=True, type=int)
    parser.add_argument("--route-id", required=True,
                        help="explicit route from the verified normal four-event decision")
    parser.add_argument("--certification", help="canonical four-event certification artifact (required if FH eligible)")
    parser.add_argument("--wildcard-value-generation-id",
                        help="separate verified 6-10 event Wildcard value generation")
    parser.add_argument("--cache-dir", type=Path, help="disposable manager-world cache directory")
    parser.add_argument("--evidence-dir", required=True, type=Path,
                        help="directory for the immutable per-run assessment record")
    parser.add_argument("--season", default="2026/27")
    parser.add_argument("--reservation-calibration", type=Path,
                        help="content-addressed reservation calibration artifact")
    parser.add_argument("--calibration-evidence-dir", type=Path,
                        help="root directory for the calibration's retained causal evidence artifacts")
    parser.add_argument("--reservation-forecast-dir", type=Path,
                        help="directory with content-addressed opportunity inputs and optional BB/TC/FH/WC forecast JSON files")
    parser.add_argument("--evaluator-readiness-dir", type=Path,
                        help="directory with action-specific BB.json, FH.json and WC.json readiness artifacts")
    parser.add_argument("--evaluator-readiness-evidence-dir", type=Path,
                        help="root for retained causal observations cited by readiness artifacts")
    args = parser.parse_args(argv)

    if not args.db.is_file():
        parser.error(f"database does not exist: {args.db}")
    uri = f"file:{args.db.resolve().as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        calibration = None
        calibration_evidence_verifier = None
        reservation_forecasts = {}
        reservation_forecast_evidence_verifier = None
        evaluator_readiness_artifacts = {}
        evaluator_readiness_evidence_verifier = None
        if args.reservation_calibration is not None:
            if args.calibration_evidence_dir is None or not args.calibration_evidence_dir.is_dir():
                parser.error("--reservation-calibration requires an existing --calibration-evidence-dir")
            raw = json.loads(args.reservation_calibration.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                parser.error("reservation calibration artifact must be a JSON object")
            calibration = raw

            calibration_evidence_verifier = _calibration_evidence_loader(
                args.calibration_evidence_dir, conn,
            )
        if args.reservation_forecast_dir is not None:
            if not args.reservation_forecast_dir.is_dir():
                parser.error("--reservation-forecast-dir must be an existing directory")
            reservation_forecasts, reservation_forecast_evidence_verifier = (
                _load_reservation_forecast_inputs(args.reservation_forecast_dir, conn)
            )
        if args.evaluator_readiness_dir is not None:
            if not args.evaluator_readiness_dir.is_dir():
                parser.error("--evaluator-readiness-dir must be an existing directory")
            if (
                args.evaluator_readiness_evidence_dir is None
                or not args.evaluator_readiness_evidence_dir.is_dir()
            ):
                parser.error(
                    "--evaluator-readiness-dir requires an existing --evaluator-readiness-evidence-dir"
                )
            evaluator_readiness_artifacts, evaluator_readiness_evidence_verifier = (
                _load_evaluator_readiness_inputs(
                    args.evaluator_readiness_dir,
                    args.evaluator_readiness_evidence_dir,
                    conn,
                )
            )
        record = chip_assessment.run_production_chip_assessment(
            conn,
            decision_id=str(args.decision_id),
            entry_id=int(args.entry_id),
            route_id=str(args.route_id),
            rules=None,
            season=str(args.season),
            certification_path=args.certification,
            wildcard_value_generation_id=args.wildcard_value_generation_id,
            cache_dir=None if args.cache_dir is None else str(args.cache_dir),
            reservation_calibration_artifact=calibration,
            reservation_calibration_evidence_verifier=calibration_evidence_verifier,
            reservation_forecasts=reservation_forecasts,
            reservation_forecast_evidence_verifier=reservation_forecast_evidence_verifier,
            evaluator_readiness_artifacts=evaluator_readiness_artifacts,
            evaluator_readiness_evidence_verifier=evaluator_readiness_evidence_verifier,
        )
        receipt = chip_assessment_store.retain_assessment(record, args.evidence_dir)
        verified = chip_assessment_store.verify_assessment(receipt["path"])
        print(json.dumps({"receipt": receipt, "verification": verified}, sort_keys=True))
        return 0
    except chip_assessment.ChipAssessmentPreflightError as failure:
        print(json.dumps({
            "status": "PREFLIGHT_REFUSED",
            "detail": str(failure),
            "action_refusals": failure.action_refusals,
        }, sort_keys=True), file=sys.stderr)
        return 2
    except Exception as failure:
        print(json.dumps({
            "status": "ASSESSMENT_REFUSED",
            "detail": f"{type(failure).__name__}: {failure}",
        }, sort_keys=True), file=sys.stderr)
        return 3
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
