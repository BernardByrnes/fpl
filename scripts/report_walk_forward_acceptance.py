"""Print the PE-2D synthetic acceptance artifact's digest and headline numbers.

Builds the hand-computed acceptance world from the test module, renders the
scoreboard, and reports the canonical digest plus every metric the required report
asks for.  Read-only with respect to the live database: the world lives in a temp
directory.
"""

from __future__ import annotations

import hashlib
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from fpl_brain import walk_forward_scoreboard as sb  # noqa: E402
from fpl_brain.database import connect_database  # noqa: E402
from test_walk_forward_scoreboard import _world  # noqa: E402


def main() -> int:
    directory = Path(tempfile.mkdtemp(prefix="pe2d-artifact-"))
    conn = connect_database(directory / "acceptance.db")
    try:
        world = _world(conn)
        scoreboard = sb.build_scoreboard(conn, artifact=world["artifact"], events=[5], top_k=2)
        body = sb.canonical_bytes(scoreboard)
        digest = "sha256:" + hashlib.sha256(body).hexdigest()
        markdown = sb.render_markdown(scoreboard)

        print("SCOREBOARD DIGEST:", digest)
        print("CANONICAL BYTES:", len(body))
        print("STATUS:", scoreboard["status"])
        print("TOP_K:", scoreboard["identity"]["top_k"])
        print("N:", scoreboard["sample"]["player_event_observations"])
        print("POPULATION DIGEST:", scoreboard["population"]["shared_comparison_digest"])
        print("CANDIDATES:", scoreboard["population"]["candidates"],
              "EVALUATED:", scoreboard["population"]["evaluated"],
              "EXCLUDED:", scoreboard["population"]["excluded"])
        print("EXCLUDED BY STATUS:", scoreboard["population"]["excluded_by_status"])
        print("SAMPLE INTERPRETATION:", scoreboard["sample"]["sample_interpretation"])
        print()
        for arm in scoreboard["arms"]:
            print(f"ARM {arm['arm']}  version={arm['version']}  N={arm['N']}  status={arm['status']}")
            for field in ("mae", "rmse", "bias", "median_ae", "spearman", "top_k_hit_rate"):
                metric = arm[field]
                print(f"    {field:18} {metric['value']}  ({metric['status']})")
        print()
        for entry in scoreboard["probability_metrics"]:
            print(f"BRIER {entry['metric']:22} n={entry['population']['scored']:3} "
                  f"value={entry['brier']['value']} rate={entry['observed_rate']}")
        print()
        for entry in scoreboard["quantile_coverage"]["metrics"]:
            print(f"QUANTILE {entry['metric']:32} n={entry['value']['n']:3} "
                  f"value={entry['value']['value']} width={entry['mean_interval_width']['value']}")
        print("DISTRIBUTION BASIS:", scoreboard["quantile_coverage"]["distribution_basis"])
        print()
        for entry in scoreboard["component_diagnostics"]["metrics"]:
            print(f"COMPONENT {entry['metric']:14} value={entry['value']['value']} n={entry['value']['n']}")
        print()
        print("MARKDOWN FIRST LINE:", markdown.splitlines()[0])
        print("MARKDOWN LINES:", len(markdown.splitlines()))

        json_path, md_path = sb.write_scoreboard(scoreboard, directory=directory)[1:]
        print("WROTE:", json_path.name, md_path.name)
        print("FILE BYTES EQUAL CANONICAL:", json_path.read_bytes() == body)
        print("EMPTY DIR AT START:", sorted(p.name for p in directory.iterdir()) == ["acceptance.db"])
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
