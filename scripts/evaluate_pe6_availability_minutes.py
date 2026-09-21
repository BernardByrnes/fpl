"""PE-6 evaluation CLI — READ ONLY.  Never writes a projection run or a freeze.

    python scripts/evaluate_pe6_availability_minutes.py --config <config.json> \
        --events 2 3 4 --json-out <path>

The artifact this prints is evidence for senior review.  It does not promote the
challenger, does not re-point an incumbent version identifier, and does not
touch the Monte Carlo RNG.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fpl_brain import availability_minutes_evaluation as ev
from fpl_brain.config import config_path, load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="PE-6 availability / minutes incumbent-vs-challenger evaluation (read only)"
    )
    parser.add_argument("--config")
    parser.add_argument("--events", type=int, nargs="+", required=True)
    parser.add_argument("--json-out")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    conn = sqlite3.connect(f"file:{config_path(config, 'database')}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        artifact = ev.evaluate_events(conn, sorted(set(args.events)))
    finally:
        conn.close()

    identity = artifact["identity"]
    print("schema          :", artifact["schema"])
    print("challenger      :", identity["challenger"]["challenger_version"],
          "cfg", identity["challenger"]["challenger_config_hash"][:24] + "…")
    print("incumbent       :", identity["incumbent"]["minutes_model_version"],
          "cfg", identity["incumbent"]["incumbent_config_hash"][:24] + "…")
    print("identity check  :", identity["challenger"]["incumbent_identity"]["unchanged"])
    print("promotion       :", identity["challenger"]["promotion"])
    population = artifact["population"]
    print(f"\npopulation      : scored={population['scored']} excluded={population['excluded']} "
          f"coverage={population['coverage_share']}")
    print("                  exclusions:", json.dumps(population["excluded_by_status"]))
    sample = artifact["sample"]
    print(f"sample          : events={sample['target_events_with_observations']}/"
          f"{sample['target_events']} observations={sample['player_event_observations']} "
          f"-> {sample['sample_interpretation']}")
    print("\nincumbent metrics")
    for name in ("brier_p_start", "brier_p_60", "expected_minutes_mae",
                 "expected_minutes_bias", "brier_p_appearance"):
        metric = artifact["incumbent_metrics"].get(name) or {}
        print(f"  {name:22} {metric.get('value')} (n={metric.get('n')}, {metric.get('status')})")
    for arm, payload in sorted(artifact["arms"].items()):
        print(f"\narm: {arm}")
        print("  families:", payload["families"])
        for name, entry in sorted(payload["comparison"]["metrics"].items()):
            print(f"  {name:22} inc={entry['incumbent']} chal={entry['challenger']} "
                  f"delta={entry['delta']} -> {entry['preferred']}")
        print("  outcome:", payload["outcome"]["token"], payload["outcome"]["basis"])
    print("\nrecommendation  :", artifact["recommendation"]["token"],
          artifact["recommendation"]["basis"])
    if artifact["recommendation"]["accepted_refinement_families"]:
        print("  accepted families:", artifact["recommendation"]["accepted_refinement_families"])
    print("  review required:", artifact["recommendation"]["review_required"])

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
        print("\nwrote", args.json_out)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
