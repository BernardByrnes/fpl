"""PE-7 evaluation CLI — READ ONLY.  Never writes a projection run or a freeze.

    python scripts/evaluate_pe7_team_player_attack.py --config <config.json> \
        --events 2 3 4 --json-out <path>

Two families are measured against their frozen incumbents on one identical causal
walk-forward population each:

* TEAM — the stored official realised team-side xG per fixture side (primary), with
  actual goals as a secondary descriptive check only;
* PLAYER — xG/90 and xA/90 scored on the player's REALISED exposure for the target
  fixture, at player x fixture grain so a double gameweek keeps every fixture.

Player identity (membership, club, position) is resolved as of each cutoff from the
latest ACCEPTED official bootstrap generation at or before it; a cutoff with no
usable generation fails closed — its candidates are excluded at EVENT scope with no
candidate count — rather than being projected from the mutable ``players`` row.

The artifact this prints is evidence for senior review.  It MEASURES; it does not
accept or promote an arm, does not re-point an incumbent version identifier, and
does not touch the Monte Carlo RNG or any xPts / calibration path.  Repeated event
ids are normalized by the evaluation itself, so passing them is harmless.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fpl_brain import team_player_attack_evaluation as ev
from fpl_brain.config import config_path, load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="PE-7 team attack/defence + player attack incumbent-vs-challenger evaluation (read only)"
    )
    parser.add_argument("--config")
    parser.add_argument("--events", type=int, nargs="+", required=True)
    parser.add_argument("--json-out")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    conn = sqlite3.connect(f"file:{config_path(config, 'database')}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        artifact = ev.evaluate_events(conn, args.events)
    finally:
        conn.close()

    identity = artifact["identity"]
    frozen = identity["frozen_incumbents"]
    print("schema          :", artifact["schema"])
    print("version         :", artifact["evaluation_version"])
    print("cutoff policy   :", artifact["cutoff_policy"])
    print("frozen identity :", frozen["unchanged"], "| pe6 minutes trio:",
          ", ".join(frozen["pe6_minutes_trio"]))
    print("team challenger :", identity["team_challenger"]["challenger_version"],
          "cfg", identity["team_challenger"]["challenger_config_hash"][:24] + "…")
    print("player challenger:", identity["player_challenger"]["challenger_version"],
          "cfg", identity["player_challenger"]["challenger_config_hash"][:24] + "…")
    print("promotion       :", identity["promotion"])

    population = artifact["population"]
    for family in ("team", "player"):
        block = population[family]
        print(f"\npopulation {family:6}: grain={block['grain']} scored={block['scored']} "
              f"slots={block['candidate_slots']} digest={str(block['population_digest'])[:22]}…")
        print("                  exclusions:", json.dumps(block["excluded_by_status"]))
        accounting = block["accounting"]
        print("                  accounting:", json.dumps({
            "enumerated": accounting["enumerated_candidate_slots"],
            "scored": accounting["scored_rows"],
            "excluded": accounting["excluded_candidates"],
            "reconciles": accounting["reconciles"],
        }))
        if block["undecided_projection_counts"]:
            print("                  arm-undefined records:", json.dumps(block["undecided_projection_counts"]))
    print("                  same population:", population["same_population"])
    for entry in artifact["events"]:
        evidence = entry.get("team_evidence") or {}
        if evidence and not evidence.get("available"):
            print(
                f"                  event {entry['event']}: NO TEAM EVIDENCE at the cutoff "
                f"({evidence.get('matches', 0)} earlier completed match(es)) -> every side "
                "excluded with a status, none scored and none invented"
            )

    for family in ("team", "player"):
        section = artifact[family]
        sample = section["sample"]
        print(f"\n{family} sample     : events={sample['target_events_with_observations']}/"
              f"{sample['target_events']} observations={sample['observations']} "
              f"-> {sample['sample_interpretation']}")
        print(f"{family} incumbent  :", json.dumps({
            name: (section["incumbent_metrics"].get(name) or {}).get("value")
            for name in sorted(section["incumbent_metrics"])
            if isinstance(section["incumbent_metrics"].get(name), dict)
        }))
        for arm, payload in sorted(section["arms"].items()):
            print(f"  arm: {arm}")
            for name, entry in sorted(payload["comparison"]["metrics"].items()):
                print(f"    {name:20} inc={entry['incumbent']} chal={entry['challenger']} "
                      f"delta={entry['delta']} n={entry['n']} role={entry['role']} "
                      f"-> {entry['preferred']}")
            print("    measurement:", payload["measurement"]["token"], payload["measurement"]["basis"])
        strata = section["strata"]
        for dimension, block in sorted(strata.items()):
            sizes = {key: entry["n"] for key, entry in sorted(block["strata"].items())}
            print(f"    strata {dimension:22} {json.dumps(sizes)}")
        subsets = section["multi_family_subset_policy"]["declared_subsets"]
        print("    multi-family arms:", json.dumps({name: value["status"] for name, value in sorted(subsets.items())}))

    coherence = artifact["coherence"]
    print("\ncoherence       :", coherence["version"], coherence["grain"])
    for arm, entry in sorted(coherence["checks"]["arms"].items()):
        print(f"  {arm:50} events={entry['events']} exceeds={len(entry['exceeds_environment'])} "
              f"invented={len(entry['invented_fixture_attack'])} dirty={entry['dirty']}")

    print("causality audit :", artifact["causality"]["audited_surface_count"], "mutable surface(s);",
          "second predicate introduced:", artifact["causality"]["discipline"]["second_predicate_introduced"])
    for entry in artifact["causality"]["audit"]:
        print(f"  {entry['surface']:44} -> {entry['resolution'][:96]}…")
    print("xPts diagnostic :", artifact["xpts_diagnostic"]["status"])
    print("claims          :", artifact["claims"]["selection_claim"], artifact["claims"]["allowed"])
    print("artifact digest :", artifact["artifact_digest"])

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
        print("\nwrote", args.json_out)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
