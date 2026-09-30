#!/usr/bin/env python3
"""Certify Wildcard's separate 6–10 event value horizon.

The normal transfer product remains FOUR_GW. This command reuses projection
runs from one pinned execution and refuses to assemble a longer horizon from
later cutoffs, snapshots or predictive identities.

Example::

    python scripts/certify_wildcard_horizon.py --db <runtime-db> \
        --four-gw-generation <generation-id> --runs-json wildcard-runs.json \
        --events 6,7,8,9,10,11

``runs-json`` maps each event to the exact family/run ids produced under the
same cutoff as the verified four-event generation. It does not contain matrix
values or caller-asserted identities.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fpl_brain import execution, generation_store as gs
from fpl_brain.database import connect_database


def parse_events(value: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        parts = value.split(",")
        if not value.strip() or any(not part.strip() for part in parts):
            raise ValueError("--events must be a comma-separated list of event ids")
        events = tuple(int(part.strip()) for part in parts)
    else:
        events = tuple(int(event) for event in value)
    if len(set(events)) != len(events):
        raise ValueError("Wildcard value events must not contain duplicates")
    return events


def certify_wildcard_horizon(
    conn: sqlite3.Connection,
    *,
    four_gw_generation_id: str,
    events: Sequence[int],
    runs_by_event: Mapping[int, Mapping[str, int]],
    controller: Any,
) -> gs.CertifiedGeneration:
    """Publish the verified Wildcard generation from the normal generation's snapshot."""

    base = gs.load_generation(conn, str(four_gw_generation_id))
    report = gs.verify_generation(conn, base.generation_id)
    if not report.get("verified"):
        raise gs.GenerationRefused(
            gs.DIAG_GENERATION_NOT_CERTIFIED,
            ["the supplied four-event generation did not independently verify"],
        )
    return gs.certify_wildcard_value_generation(
        conn,
        chip_generation_id=base.generation_id,
        planning_event=int(base.planning_event),
        cutoff=str(base.cutoff),
        events=parse_events(events),
        runs_by_event={int(event): {str(key): int(value) for key, value in rows.items()}
                       for event, rows in runs_by_event.items()},
        snapshot=base.snapshot,
        controller=controller,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, help="the isolated writable runtime database")
    parser.add_argument("--four-gw-generation", required=True)
    parser.add_argument("--runs-json", required=True, type=Path)
    parser.add_argument("--events", required=True, type=parse_events)
    args = parser.parse_args(argv)

    raw = json.loads(args.runs_json.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        parser.error("runs-json must be an object mapping events to run-id maps")
    runs = {int(event): dict(rows) for event, rows in raw.items() if isinstance(rows, Mapping)}
    conn = connect_database(args.db)
    controller = execution.ExecutionController(conn)
    try:
        base = gs.load_generation(conn, args.four_gw_generation)
        controller.create_run(
            planning_event=int(base.planning_event),
            planning_cutoff=str(base.cutoff),
            hard_stop_at=datetime.now(timezone.utc) + timedelta(hours=1),
            label="wildcard_value_horizon_certification",
        )
        controller.start()
        controller.acquire_writer_lease()
        try:
            result = certify_wildcard_horizon(
                conn,
                four_gw_generation_id=base.generation_id,
                events=args.events,
                runs_by_event=runs,
                controller=controller,
            )
        except BaseException as failure:
            controller.finish(execution.RUN_FAILED, str(failure))
            raise
        controller.finish(execution.RUN_COMPLETE)
        print(json.dumps({
            "generation_id": result.generation_id,
            "horizon_kind": result.horizon_kind,
            "events": list(result.events),
            "cutoff": result.cutoff,
            "normal_generation_id": base.generation_id,
            "verified": bool(gs.verify_generation(conn, result.generation_id).get("verified")),
        }, sort_keys=True))
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
