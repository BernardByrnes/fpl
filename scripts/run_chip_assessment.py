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

from fpl_brain import chip_assessment, chip_assessment_store
from fpl_brain.season_rules import SeasonRules


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

            evidence_root = args.calibration_evidence_dir.resolve()

            def calibration_evidence(reference: str) -> dict:
                evidence_path = (evidence_root / str(reference)).resolve()
                if not evidence_path.is_relative_to(evidence_root):
                    raise ValueError("causal evidence reference escapes its configured root")
                value = json.loads(evidence_path.read_text(encoding="utf-8"))
                if not isinstance(value, dict):
                    raise ValueError("causal evidence artifact must be a JSON object")
                return value

            calibration_evidence_verifier = calibration_evidence
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
