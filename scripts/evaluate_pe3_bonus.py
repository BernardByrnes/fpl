"""PE-3 evaluation CLI — READ ONLY.  Never writes a projection run or a generation row.

    python scripts/evaluate_pe3_bonus.py --config K:/FPL/config.json --json-out <path>
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fpl_brain import bonus_evaluation as be
from fpl_brain.config import config_path, load_config

GW4 = {
    "event": 4,
    "cutoff": "2026-09-12T10:40:04Z",
    "deadline": "2026-09-12T12:30:00Z",
    "chain": {"xpts": 133, "minutes": 127, "team": 128, "rate": 130},
    "fetch_run_id": 36,
    "xpts_run_id": 133,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PE-3 structural bonus evaluation (read only)")
    parser.add_argument("--config")
    parser.add_argument("--json-out")
    parser.add_argument("--raw-dir", default=str(Path("data/raw")))
    parser.add_argument("--skip-raw-sha-pin", action="store_true")
    args = parser.parse_args(argv)

    code_sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=str(Path(__file__).resolve().parents[1])).stdout.strip() or None
    config = load_config(args.config)
    conn = sqlite3.connect(f"file:{config_path(config, 'database')}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        artifact = be.evaluate_event(
            conn, event=GW4["event"], as_of=GW4["cutoff"], deadline=GW4["deadline"],
            chain=GW4["chain"], fetch_run_id=GW4["fetch_run_id"],
            raw_dir=args.raw_dir, target_event=GW4["event"],
            xpts_run_id=GW4["xpts_run_id"],
            expected_raw_sha256=None if args.skip_raw_sha_pin else be.GW4_LEGACY_RAW_SHA256,
            code_sha=code_sha,
        )
    finally:
        conn.close()

    hist = artifact["historical_input_identity"]
    rec = hist["legacy_reconstruction"]
    print("schema     :", artifact["schema"])
    print("mode       :", artifact["evaluation_mode"])
    print(f"historical : xpts={hist['xpts_run_id']} minutes={hist['minutes_run_id']} "
          f"team={hist['team_run_id']} rate={hist['rate_run_id']} fetch={hist['official_fetch_run_id']}")
    print(f"authority  : {hist['background_authority']}  formal_record={hist['formal_generation_record']}")
    print(f"legacy     : sha256={rec['raw_sha256'][:24]}… elements={rec['raw_elements']} "
          f"snapshots={rec['snapshots']} raw_json={rec['raw_json_matches']}/{rec['raw_json_compared']}")
    print(f"replay     : {artifact['current_replay_identity']['mc_version']} "
          f"sims={artifact['current_replay_identity']['simulations']} "
          f"cfg={artifact['current_replay_identity']['config_hash'][:24]}…")
    structural = artifact["structural"]
    print(f"\nstructural : n={structural.get('n')} digest={str(structural.get('population_digest'))[:24]}…")
    print(f"             mean_pred={structural.get('mean_predicted_bonus')} "
          f"mean_real={structural.get('mean_realised_bonus')}")
    for key in ("bias", "mae", "brier_p_any"):
        metric = structural.get(key)
        if isinstance(metric, dict):
            print(f"             {key:12} value={metric.get('value')} status={metric.get('status')} n={metric.get('n')}")
    for position, block in sorted((structural.get("by_position") or {}).items()):
        print(f"             {position}: {block.get('status')} n={block.get('n')} "
              f"pred={block.get('mean_predicted')} real={block.get('mean_realised')}")
    soft = artifact["soft_baseline"]
    print(f"\nsoft       : status={soft.get('status')} n={soft.get('n')} "
          f"same_population={artifact['same_population']}")
    if soft.get("status") == "OK":
        print(f"             mean_pred={soft.get('mean_predicted')} mean_real={soft.get('mean_realised')} "
              f"bias={soft.get('bias', {}).get('value')} mae={soft.get('mae', {}).get('value')}")
    diag = artifact["world_diagnostics"]
    print(f"\nworlds     : fixtures={diag['fixtures']} per_fixture={set(diag['worlds_per_fixture'].values())} "
          f"predictions={diag['player_fixture_predictions']}")
    print(f"             any_tie={diag['any_proxy_tie_worlds']} bonus_tie={diag['bonus_affecting_tie_worlds']} "
          f"over_six={diag['worlds_over_six_bonus']}")
    print(f"             rival_example={'yes' if diag['rival_dependence_examples'] else 'none'}")
    snap = artifact["snapshot_integrity_diagnostic"]
    print(f"\nsnapshot   : comparable={snap['comparable_players']} minute_match={snap['exact_minute_matches']} "
          f"bps_match={snap['exact_bps_matches']}")
    print(f"claim      : {artifact['claim']}")
    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(artifact, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
        print("artifact   :", args.json_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
