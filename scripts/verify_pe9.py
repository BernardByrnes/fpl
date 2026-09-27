#!/usr/bin/env python3
"""PE-9 — the canonical re-derivation audit commands.

    verify generation <generation_id>
    verify decision <decision_id>

ONE command pair, so an operator re-derives a certified generation or a production
decision from authoritative persisted evidence instead of trusting a stored label.
Both call the functions in :mod:`fpl_brain.generation_store`; this module only parses
the arguments, resolves the authoritative store and reports.

Exit codes
----------
0   verified
3   refused (the reason is printed; the token is what an operator greps for)
2   the command line itself is wrong
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fpl_brain import generation_store as gs
from fpl_brain.config import config_path, load_config
from fpl_brain.database import connect_readonly_database


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PE-9 re-derivation audit")
    parser.add_argument("--config")
    parser.add_argument("--database", help="override the authoritative store path")
    sub = parser.add_subparsers(dest="subject", required=True)
    generation = sub.add_parser("generation", help="verify one certified generation")
    generation.add_argument("generation_id")
    decision = sub.add_parser("decision", help="verify one production decision record")
    decision.add_argument("decision_id")
    parser.add_argument("--json", action="store_true", help="emit the report as JSON")
    args = parser.parse_args(argv)

    path = args.database
    if path is None:
        path = config_path(load_config(args.config), "database")
    if not Path(path).exists():
        print(f"verify refused: the authoritative store {path} does not exist", file=sys.stderr)
        return 3
    # The audit READS: a verification command must never be able to change the
    # evidence it is verifying.
    conn = connect_readonly_database(path)
    try:
        try:
            if args.subject == "generation":
                report = gs.verify_generation(conn, args.generation_id)
            else:
                report = gs.verify_decision(conn, args.decision_id)
        except gs.GenerationRefused as refusal:
            if args.json:
                print(json.dumps({"verified": False, "token": refusal.token,
                                  "reasons": refusal.reasons}))
            else:
                print(f"verify refused: {refusal}", file=sys.stderr)
            return 3
    finally:
        conn.close()
    if args.json:
        print(json.dumps(report, sort_keys=True, default=str))
    else:
        print(f"{args.subject} {report['verified'] and 'VERIFIED' or 'NOT VERIFIED'}: "
              f"{report.get('generation_id') or report.get('decision_id')}")
        for key in sorted(report):
            if key in {"verified", "schema"}:
                continue
            print(f"  {key}: {report[key]}")
    return 0 if report.get("verified") else 3


if __name__ == "__main__":
    raise SystemExit(main())
