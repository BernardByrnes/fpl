"""PE-6 evaluation CLI — READ ONLY.  Never writes a projection run or a freeze.

    python scripts/evaluate_pe6_availability_minutes.py --config <config.json> \
        --events 2 3 4 --json-out <path>

Minutes are scored at PE-2's player-fixture component grain, so every fixture of
a double gameweek is its own observation and is retained.

Player identity is resolved as of each cutoff from the latest ACCEPTED official
bootstrap generation at or before it; a cutoff with no usable generation fails
closed (its candidates are excluded at EVENT scope, with no candidate count) rather
than being projected or counted from the mutable ``players`` row.

The artifact this prints is evidence for senior review.  It MEASURES; it does not
accept or promote an arm, does not re-point an incumbent version identifier, and
does not touch the Monte Carlo RNG.  Repeated event ids are normalized by the
evaluation itself, so passing them is harmless.
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
        artifact = ev.evaluate_events(conn, args.events)
    finally:
        conn.close()

    identity = artifact["identity"]
    print("schema          :", artifact["schema"])
    print("grain           :", artifact["grain"], "|", artifact["grain_note"])
    print("challenger      :", identity["challenger"]["challenger_version"],
          "cfg", identity["challenger"]["challenger_config_hash"][:24] + "…")
    print("incumbent       :", identity["incumbent"]["minutes_model_version"],
          "cfg", identity["incumbent"]["incumbent_config_hash"][:24] + "…")
    print("identity check  :", identity["challenger"]["incumbent_identity"]["unchanged"])
    print("promotion       :", identity["challenger"]["promotion"])
    population = artifact["population"]
    print(f"\npopulation      : scored={population['scored']} excluded={population['excluded']} "
          f"candidates={population['candidates']} coverage={population['coverage_share']}")
    if not population["candidates_complete"]:
        print("                  candidate count UNAVAILABLE (no causal pool at the cutoff) for ",
              population["events_without_a_candidate_count"])
    print("                  exclusions:", json.dumps(population["excluded_by_status"]))
    accounting = population["accounting"]
    print("                  accounting:", json.dumps({
        "enumerated": accounting["enumerated_candidate_slots"],
        "scored": accounting["scored_rows"],
        "excluded": accounting["excluded_candidates"],
        "status_total": accounting["excluded_by_status_total"],
        "reconciles": accounting["reconciles"],
        "status_totals_reconcile": accounting["status_totals_reconcile"],
        "per_event_reconciles": accounting["per_event_reconciles"],
        "candidate_enumeration": accounting["candidate_enumeration"],
    }))
    identity_block = population["identity"]
    print("                  identity  :", json.dumps(identity_block["scored_rows_by_basis"]),
          f"cutoff-safe={identity_block['scored_rows_cutoff_safe']}",
          f"live-fallback-allowed={identity_block['live_fallback_allowed']}")
    for block in artifact["events"]:
        generation = ((block.get("identity") or {}).get("generation") or {}) if block.get("identity") else {}
        accepted = generation.get("accepted_generation") or {}
        print(f"                  event {block['event']}: generation {accepted.get('id')} "
              f"@ {accepted.get('captured_at')} pool={generation.get('element_ids_count')} "
              f"usable={generation.get('usable')} {generation.get('reasons') or ''}")
    priors = population["pool_priors"]
    print("                  priors    :",
          f"incumbent league_pools read used={priors['incumbent_league_pools_read_used']}")
    for event, block in sorted(priors["per_event"].items()):
        print(f"                  event {event}: pooled rows={block['pooled_rows']} "
              f"without cutoff identity={block['pooled_rows_without_cutoff_identity']} "
              f"players={block['pooled_players']}")
    print("                  store divergence (reported, never scored):", json.dumps({
        event: {
            "cutoff_pool_not_in_persisted_pool": block["in_cutoff_generation_not_in_persisted_pool"],
            "persisted_pool_not_in_cutoff_pool": block["in_persisted_pool_not_in_cutoff_generation"],
        }
        for event, block in sorted(artifact["store_divergence"]["per_event"].items())
        if block
    }))
    sample = artifact["sample"]
    print(f"sample          : events={sample['target_events_with_observations']}/"
          f"{sample['target_events']} observations={sample['player_fixture_observations']} "
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
        print("  measurement:", payload["measurement"]["token"], payload["measurement"]["basis"])
    subsets = artifact["multi_family_subset_policy"]
    print("\nmulti-family subsets:", json.dumps(
        {name: block["status"] for name, block in subsets["declared_subsets"].items()}
    ))
    summary = artifact["measurement_summary"]
    print("measurement     :", summary["token"], summary["basis"])
    print("  model selection:", summary["model_selection"])
    print("  review required:", summary["review_required"])

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
        print("\nwrote", args.json_out)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
