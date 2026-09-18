"""PE-3 replay gate — official BPS totals -> OUR allocator -> official bonus.

READ-ONLY.  Opens the database read-only, reads official final BPS and bonus for every
eligible FINAL fixture, runs the canonical allocator, and reports exact-match counts.

Eligibility is decided by the repository's own finality authority, not by us:
  * ``events.finished = 1 AND events.data_checked = 1`` (the event is final and the data
    has been checked), and
  * ``fixtures.finished = 1`` (the fixture itself is final), and
  * every player row for that fixture carries a non-NULL ``bps`` AND ``bonus``.

2026/27 introduces post-match Opta review, so a provisional fixture or a provisional
event is EXCLUDED rather than scored — never evaluated as though it were final.

Usage:
    python scripts/replay_bonus_allocation.py --config K:/FPL/config.json
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fpl_brain import bonus_allocation as ba
from fpl_brain.config import config_path, load_config


def _eligible_fixtures(conn: sqlite3.Connection) -> list[int]:
    """Fixtures that are unambiguously FINAL and carry complete official BPS/bonus rows."""

    rows = conn.execute(
        """
        SELECT f.id AS fixture_id, f.event
        FROM fixtures f
        JOIN events e ON e.id = f.event
        WHERE f.finished = 1
          AND e.finished = 1
          AND e.data_checked = 1
          AND NOT EXISTS (
              SELECT 1 FROM player_gameweeks pg
              WHERE pg.fixture_id = f.id
                AND (pg.bps IS NULL OR pg.bonus IS NULL)
          )
          AND EXISTS (SELECT 1 FROM player_gameweeks pg WHERE pg.fixture_id = f.id)
        ORDER BY f.event, f.id
        """
    ).fetchall()
    return [int(row["fixture_id"]) for row in rows]


def replay(conn: sqlite3.Connection) -> dict:
    fixtures = _eligible_fixtures(conn)
    report: dict = {
        "allocator_version": ba.BONUS_ALLOCATOR_VERSION,
        "fixtures_tested": 0,
        "player_rows": 0,
        "exact_matches": 0,
        "mismatches": 0,
        "tie_fixtures": 0,
        "fixtures_exceeding_six_total_bonus": 0,
        "excluded": [],
        "mismatch_examples": [],
        "tie_examples": [],
    }
    for fixture_id in fixtures:
        rows = conn.execute(
            "SELECT player_id, bps, bonus FROM player_gameweeks WHERE fixture_id = ?",
            (fixture_id,),
        ).fetchall()
        bps_by_player = {int(r["player_id"]): int(r["bps"]) for r in rows}
        official = {int(r["player_id"]): int(r["bonus"]) for r in rows}

        ours = ba.allocate_fixture_bonus(bps_by_player)
        report["fixtures_tested"] += 1
        report["player_rows"] += len(ours)

        totals = ba.bonus_totals(official)
        if totals["exceeds_six"]:
            report["fixtures_exceeding_six_total_bonus"] += 1
        distinct = len({v for v in bps_by_player.values()})
        if distinct < len(bps_by_player):
            report["tie_fixtures"] += 1
            values = sorted(bps_by_player.values(), reverse=True)[:4]
            if len(report["tie_examples"]) < 5:
                report["tie_examples"].append(
                    {"fixture_id": fixture_id, "top_bps": values,
                     "official_total_bonus": totals["total_bonus"]}
                )

        for player_id, expected in ours.items():
            if official.get(player_id) == expected:
                report["exact_matches"] += 1
            else:
                report["mismatches"] += 1
                if len(report["mismatch_examples"]) < 10:
                    report["mismatch_examples"].append(
                        {"fixture_id": fixture_id, "player_id": player_id,
                         "bps": bps_by_player[player_id], "ours": expected,
                         "official": official.get(player_id)}
                    )
    # Anything final-looking that we deliberately did NOT score.
    provisional = conn.execute(
        """
        SELECT f.id AS fixture_id, f.event, e.finished AS event_finished,
               e.data_checked AS event_checked, f.finished AS fixture_finished
        FROM fixtures f JOIN events e ON e.id = f.event
        WHERE EXISTS (SELECT 1 FROM player_gameweeks pg
                      WHERE pg.fixture_id = f.id AND pg.bps IS NOT NULL)
          AND (f.finished = 0 OR e.finished = 0 OR e.data_checked = 0)
        ORDER BY f.id
        """
    ).fetchall()
    for row in provisional:
        report["excluded"].append(
            {"fixture_id": int(row["fixture_id"]), "event": int(row["event"]),
             "reason": "not final (fixture.finished=%d, event.finished=%d, data_checked=%d)"
                       % (row["fixture_finished"], row["event_finished"], row["event_checked"])}
        )
    # Fixtures with no BPS rows at all.
    missing = conn.execute(
        """
        SELECT f.id AS fixture_id, f.event FROM fixtures f JOIN events e ON e.id = f.event
        WHERE f.finished = 1 AND e.finished = 1 AND e.data_checked = 1
          AND NOT EXISTS (SELECT 1 FROM player_gameweeks pg WHERE pg.fixture_id = f.id)
        ORDER BY f.id
        """
    ).fetchall()
    for row in missing:
        report["excluded"].append(
            {"fixture_id": int(row["fixture_id"]), "event": int(row["event"]),
             "reason": "no player_gameweeks rows"}
        )
    report["status"] = "PASS" if report["mismatches"] == 0 else "FAIL"
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay official BPS -> bonus allocation")
    parser.add_argument("--config")
    parser.add_argument("--json-out")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    conn = sqlite3.connect(f"file:{config_path(config, 'database')}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        report = replay(conn)
    finally:
        conn.close()

    print(f"allocator     : {report['allocator_version']}")
    print(f"fixtures      : {report['fixtures_tested']}")
    print(f"player rows   : {report['player_rows']}")
    print(f"exact matches : {report['exact_matches']}")
    print(f"mismatches    : {report['mismatches']}")
    print(f"tie fixtures  : {report['tie_fixtures']}")
    print(f"fixtures whose OFFICIAL total bonus exceeds 6: "
          f"{report['fixtures_exceeding_six_total_bonus']}")
    for item in report["tie_examples"]:
        print(f"  tie example: {item}")
    for item in report["mismatch_examples"]:
        print(f"  MISMATCH: {item}")
    for item in report["excluded"]:
        print(f"  excluded: {item}")
    print(f"STATUS: {report['status']}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=1, sort_keys=True), encoding="utf-8")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
