"""PE-8 calibration evaluation CLI — READ ONLY.  Writes no database row.

    python scripts/evaluate_pe8_calibration.py --certification <artifact.json> \
        [--config <config.json>] [--events 5 6 7] [--json-out <path>]

The certification artifact is loaded through the production loader
(``four_gw_decision.load_certification_artifact``), so the anchor this evaluation
resolves is the certified one and never a "newest row" query.  Target events
default to the artifact's own certified bundle events.

The artifact this prints is evidence for senior review.  It MEASURES: it does not
promote a transform, does not re-point an incumbent model or calibration version,
does not change the FPL assist-mapping constant, and does not touch the Monte
Carlo seed namespace or draw ordering.  ``OPEN`` is a legitimate terminal state.

The database is opened read-only and the canonical digest is printed, so a run
can be reproduced from the same persisted runs and outcomes.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fpl_brain import calibration_evaluation as ce
from fpl_brain import four_gw_decision as fg
from fpl_brain.config import config_path, load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="PE-8 calibration diagnosis and causal walk-forward transform evidence (read only)"
    )
    parser.add_argument("--certification", required=True, help="certification artifact JSON")
    parser.add_argument("--config")
    parser.add_argument("--events", type=int, nargs="+")
    parser.add_argument("--json-out")
    args = parser.parse_args(argv)

    certification = fg.load_certification_artifact(args.certification)
    events = args.events or sorted(
        int(event) for event in (certification.get("certified_bundles") or {})
    )
    if not events:
        print("the certification artifact declares no target event", file=sys.stderr)
        return 2

    config = load_config(args.config)
    conn = sqlite3.connect(f"file:{config_path(config, 'database')}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        artifact = ce.evaluate(conn, artifact=certification, events=events)
    finally:
        conn.close()

    return report(artifact, json_out=args.json_out)


def report(artifact: dict, *, json_out: str | None = None) -> int:
    """Print the artifact's evidence in the order a reviewer reads it."""

    identity = artifact["identity"]
    state = artifact["terminal_state"]
    print("schema           :", artifact["schema"])
    print("evaluation       :", artifact["evaluation_version"])
    print("certification    :", identity["certification_identity"])
    print("target events    :", identity["target_events"], "| evaluated:", identity["events_evaluated"])
    frozen = identity["frozen_incumbents"]
    print("frozen incumbents:", "unchanged" if frozen["unchanged"] else f"MISMATCH {frozen['mismatches']}")
    print("terminal state   :", state["state"])
    for reason in state["reasons"]:
        print("  -", reason)
    print("promotion        : performed =", state["promotion_performed"])

    for surface in artifact["probability_calibration"]["surfaces"]:
        figure = surface["incumbent"]["figure"]
        print(
            f"\n{surface['metric']:20} n={surface['population']['scored']:5} "
            f"grain={surface['grain']:14} brier={figure['brier']['value']} "
            f"reference={figure['brier_reference']['value']} rate={figure['observed_rate']}"
        )
        print(
            f"    digest={str(surface['population']['population_digest'])[:24]}… "
            f"absent={surface['population']['probability_absent']} "
            f"outcome_unavailable={surface['population']['outcome_unavailable']}"
        )
        print(
            f"    bins_over_floor={figure['reliability']['bins_over_floor']} "
            f"max_abs_gap={figure['reliability']['max_abs_gap_over_floor']} "
            f"-> {surface['diagnosis']['status']}"
        )
        challenger = surface["causal_challenger"]
        print(
            f"    sample={surface['sample']['sample_interpretation']} "
            f"causal challenger={challenger['status']} "
            f"({challenger['reason'] or ', '.join(challenger['exclusion_reasons'])})"
        )
        defcon = surface.get("defcon_calibration")
        if defcon:
            print(
                f"    defcon calibration={defcon['version']} identity={str(defcon['identity'])[:24]} "
                f"status={defcon['status']}"
            )

    headline = artifact["expected_value"]["headline"]
    print(
        f"\nheadline EV      : {headline['status']} grain={headline['grain']} n={headline['n']} "
        f"bias={headline.get('bias', {}).get('value')} tolerance={headline['tolerance']}"
    )
    for entry in artifact["expected_value"]["component_diagnostics"]["components"]:
        print(
            f"  {entry['component']:20} n={entry['n']:5} bias={entry['bias']['value']} "
            f"mae={entry['mae']['value']} grain={entry['grain']} "
            f"causal={entry['causally_interpretable']} flags={entry['non_causal_flags']}"
        )

    coverage = artifact["monte_carlo_coverage"]["coverage"]
    print("\nquantile coverage:", coverage["population"])
    for metric in coverage["metrics"]:
        print(
            f"  {metric['metric']:32} n={metric['value']['n']:5} "
            f"value={metric['value']['value']} nominal={metric['nominal_coverage']}"
        )
    print("  basis:", coverage["distribution_basis"])

    assist = artifact["assist_mapping"]
    print(
        f"\nassist mapping   : coefficient={assist['incumbent']['coefficient']} "
        f"calibrated={assist['incumbent']['assist_mapping_calibrated']} "
        f"flag={assist['incumbent']['production_flag']} "
        f"observed={assist['pooled_evidence']['observations']} observation(s) -> "
        f"{assist['decision']['outcome']}"
    )
    for reason in assist["decision"]["reasons"]:
        print("  -", reason)

    print("\nclaims           :", json.dumps(artifact["claims"], sort_keys=True)[:400], "…")
    print("artifact digest  :", ce.artifact_digest(artifact))

    if json_out:
        Path(json_out).write_text(
            json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8"
        )
        print("\nwrote", json_out)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
