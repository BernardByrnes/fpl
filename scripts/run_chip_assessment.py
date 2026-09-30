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

from fpl_brain import chip_assessment, chip_assessment_store, chip_reservation_calibration
from fpl_brain import generation_store, repositories
from fpl_brain.season_rules import SeasonRules


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
            paired = outcome_record.get("paired_results") or {}
            play = paired.get("play") if isinstance(paired, dict) else None
            lineup = play.get("lineup") if isinstance(play, dict) else None
            if not isinstance(lineup, dict):
                raise ValueError("outcome record has no canonical source lineup")
            player_ids = {
                *(int(value) for value in lineup.get("starter_ids", ())),
                int(lineup["bench_gk_id"]),
                *(int(value) for value in lineup.get("bench_outfield_order", ())),
            }
            if len(player_ids) != 15 or any(str(player_id) not in all_positions for player_id in player_ids):
                raise ValueError("pinned generation lacks positions for the complete 15-player policy")
            cache[generation_id] = {
                str(player_id): all_positions[str(player_id)] for player_id in sorted(player_ids)
            }
            cache_identity[generation_id] = wanted_identity
        return dict(cache[generation_id])

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
        record = chip_assessment.run_production_chip_assessment(
            conn,
            decision_id=str(args.decision_id),
            entry_id=int(args.entry_id),
            route_id=str(args.route_id),
            rules=SeasonRules(season=str(args.season)),
            certification_path=args.certification,
            wildcard_value_generation_id=args.wildcard_value_generation_id,
            cache_dir=None if args.cache_dir is None else str(args.cache_dir),
            reservation_calibration_artifact=calibration,
            reservation_calibration_evidence_verifier=calibration_evidence_verifier,
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
