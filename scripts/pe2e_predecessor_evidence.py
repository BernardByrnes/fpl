"""Measure the two PE-2E defects on whichever revision this runs against.

The same probe runs on the predecessor revision and on the repaired tree, and
prints the two numbers that separate them, so the comparison is a measurement
rather than an assertion about code that no longer exists:

  P2-01  the Monte Carlo clean-sheet Brier, whose target changed from the raw
         ``clean_sheets`` flag to the position-conditioned scoring event.
  P2-02  how many of four per-event xPts runs survive ``EvaluationIdentity``
         serialisation, when each event names its own run.

It uses only APIs common to both revisions, and introspects the identity's own
dataclass fields so it needs no branch on a version constant.
"""

from __future__ import annotations

import dataclasses
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from fpl_brain import calibration  # noqa: E402
from fpl_brain import walk_forward as wf  # noqa: E402
from fpl_brain.database import connect_database  # noqa: E402

#: Four target events, each with its OWN xPts run.  A certified bundle declares one
#: run set per event, so this is the normal multi-event shape, not a contrived one.
EVENT_RUNS = [(5, 380), (6, 393), (7, 406), (8, 419)]
EVENT_BASELINE = [(5, 369), (6, 382), (7, 395), (8, 408)]


def probe_clean_sheet(module) -> dict:
    from test_pe2e_evaluation_contracts import EVENT, _mc_world

    directory = Path(tempfile.mkdtemp(prefix="pe2e-probe-"))
    conn = connect_database(directory / "probe.db")
    try:
        run_id = _mc_world(conn)
        metrics = calibration.evaluate_monte_carlo_run(conn, run_id, EVENT)
        return {"brier_clean_sheet": metrics.get("brier_clean_sheet")}
    finally:
        conn.close()


def probe_identity() -> dict:
    names = {field.name for field in dataclasses.fields(wf.EvaluationIdentity)}
    common = dict(
        evaluation_version=wf.WALK_FORWARD_VERSION,
        grain=wf.GRAIN_PLAYER_EVENT,
        target_events=(5, 6, 7, 8),
        planning_cutoff="2026-09-10T12:00:00Z",
        baseline_kinds=("RECENT_POINTS_BASELINE",),
        eligible_population_digest="sha256:" + "d" * 64,
        model_versions=(("xpts_v1", "xpts_v1.4.1"),),
        code_snapshot_sha256="sha256:" + "a" * 64,
        code_revision="rev",
        missing_data_policy_version=wf.MISSING_DATA_POLICY_VERSION,
        outcome_state=tuple((event, "SCHEDULED") for event, _run in EVENT_RUNS),
    )
    if "per_event_runs" in names:
        triples = tuple(
            [(event, "xpts_v1", run) for event, run in EVENT_RUNS]
            + [(event, wf.BASELINE_FAMILY, run) for event, run in EVENT_BASELINE]
        )
        identity = wf.EvaluationIdentity(per_event_runs=triples, **common)
        serialised = identity.as_dict()["per_event_runs"]
    else:
        # The predecessor's representation cannot even express "event 7 used run 406":
        # its fields are (family, run_id) pairs, so four events' runs go in as four
        # pairs with the same family.
        identity = wf.EvaluationIdentity(
            projection_run_ids=tuple((("xpts_v1", run) for _event, run in EVENT_RUNS)),
            baseline_run_ids=tuple((("baseline", run) for _event, run in EVENT_BASELINE)),
            **common,
        )
        serialised = identity.as_dict()
        serialised = {
            "projection_run_ids": serialised.get("projection_run_ids"),
            "baseline_run_ids": serialised.get("baseline_run_ids"),
        }
    return {"fields": sorted(names), "serialised": serialised}


def main() -> int:
    print("=== P2-01: Monte Carlo clean-sheet Brier on five position-diverse rows ===")
    print("    DEF 90' clean sheet  p=0.6   target 1.0 either way")
    print("    MID 90' clean sheet  p=0.5   target 1.0 either way")
    print("    FWD 90' raw flag =1  p=0.0   target 0.0 (repaired) vs 1.0 (predecessor)")
    print("    GKP 45' no sheet     p=0.2   target 0.0 either way")
    print("    DEF 90' conceded     p=0.3   target 0.0 either way")
    print("    repaired expectation 0.54/5 = 0.108 ; predecessor expectation 1.54/5 = 0.308")
    cleaned = probe_clean_sheet(calibration)
    print("    measured brier_clean_sheet:", cleaned["brier_clean_sheet"])
    print()
    print("=== P2-02: four per-event xPts runs through EvaluationIdentity ===")
    identity = probe_identity()
    print("    identity fields:", identity["fields"])
    print("    serialised runs:", json.dumps(identity["serialised"], sort_keys=True))
    present = [run for _event, run in EVENT_RUNS if str(run) in json.dumps(identity["serialised"])]
    print("    xPts runs surviving:", present, f"({len(present)} of {len(EVENT_RUNS)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
