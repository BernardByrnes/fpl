#!/usr/bin/env python3
"""Recompute the Phase-8B.1 certification from an existing ladder artifact.

No search, no world regeneration: the ladder records already contain the routes,
their canonical family signatures and the frontier lists.  The original artifact
is preserved; the corrected certification is written alongside it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fpl_brain import candidate_universe as cu, route_optimizer as ro, route_stability as rs


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    original = json.loads(Path(args.artifact).read_text(encoding="utf-8"))
    records = original["ladder_records"]
    budgets = original["budgets"]
    certification = rs.certify(records, budgets)
    monotonic = [ro.monotonic_check(records[i - 1], records[i]) for i in range(1, len(records))]
    identity = [ro.cross_budget_score_identity(records[i - 1], records[i]) for i in range(1, len(records))]

    corrected = {
        "phase_version": original.get("phase_version"),
        "planning_cutoff": original.get("planning_cutoff"),
        "supported_events": original.get("supported_events"),
        "budgets": budgets,
        "original_certification": original.get("certification"),
        "recertified": certification,
        "monotonic": monotonic,
        "cross_budget_score_identity": identity,
        "original_artifact": args.artifact,
        "note": ("Certification recomputed from the preserved ladder records after fixing the "
                 "frontier criterion to compare each objective's frontier separately (0.25 CORE "
                 "threshold unchanged). No search or world regeneration was performed."),
        "no_recommendation": True,
    }
    Path(args.out).write_text(json.dumps(cu.jsonable(corrected), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                              encoding="utf-8")
    print(f"recertified: original={original['certification']['overall']} -> corrected={certification['overall']} "
          f"{certification['status_flag']}")
    for name, item in certification["criteria"].items():
        print(f"  {name}: {item['status']}")
    g = certification["criteria"]["G_frontier"]["per_objective"]
    for objective, value in g.items():
        print(f"  G[{objective}]: {value['status']} smaller_best={value['smaller_best']} larger_best={value['larger_best']} "
              f"added={len(value['added_signatures'])} material_added={len(value['material_added'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
