"""Shared PE-9 fixtures: a synthetic world and the certified GENERATION on top of it.

Amendment 2 replaced the certification-artifact authority with a persisted,
content-addressed generation store.  These helpers build the smallest coherent
world the canonical validators accept, then certify it through the REAL
``generation_store.certify_generation`` path -- so a test that consumes a generation
is exercising the production certification lifecycle, not a mock of it.

Deliberately NOT here: any way to fabricate a "generation" object that never
passed the store.  The authority is the persisted row plus the pinned runs and
snapshot it names, so a fixture that skipped certification would be testing a code
path production does not have.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Mapping, Sequence

from fpl_brain import certified_bundle as cb
from fpl_brain import generation_store as gs
from fpl_brain.database import connect_database

CUTOFF = "2026-09-19T11:00:00Z"
OTHER_CUTOFF = "2026-09-18T11:00:00Z"
CODE_SNAPSHOT = "codehash"
DATA_SNAPSHOT = "sha256:" + "d" * 64
CONTEXT_HASH = "ctx"

#: The REAL declared versions, read from the one source rather than restated, so a
#: future version bump moves the fixtures with it instead of silently un-pinning.
VERSIONS = cb.declared_required_versions()

#: The four-event normal-transfer horizon the contract fixes.
HORIZON = (5, 6, 7, 8)


def base_world(conn: sqlite3.Connection) -> None:
    with conn:
        for position_id, short in ((1, "GKP"), (2, "DEF"), (3, "MID"), (4, "FWD")):
            conn.execute(
                "INSERT INTO positions(id, singular_name_short, raw_json, updated_at)"
                " VALUES (?,?, '{}','2026-09-01T00:00:00Z')",
                (position_id, short),
            )
        conn.execute(
            "INSERT INTO fetch_runs(id, started_at, status, trigger)"
            " VALUES (1,'2026-09-01T00:00:00Z','success','test')"
        )
        for team_id, name in ((1, "One"), (2, "Two")):
            conn.execute(
                "INSERT INTO teams(id, name, short_name, raw_json, updated_at)"
                " VALUES (?,?,?, '{}','2026-09-01T00:00:00Z')",
                (team_id, name, name[:3].upper()),
            )
        for player_id, name in ((1, "Alpha"), (2, "Beta")):
            conn.execute(
                "INSERT INTO players(id, web_name, team_id, element_type, is_active, first_seen_at,"
                " last_seen_at, raw_json, updated_at)"
                " VALUES (?,?,1,3,1,'2026-09-01T00:00:00Z','2026-09-01T00:00:00Z','{}',"
                "'2026-09-01T00:00:00Z')",
                (player_id, name),
            )


def add_event(conn: sqlite3.Connection, event: int) -> None:
    conn.execute(
        "INSERT INTO events(id, name, deadline_time, finished, raw_json, updated_at)"
        " VALUES (?,?,?,0,'{}','2026-09-01T00:00:00Z')",
        (event, f"GW{event}", f"2026-09-{20 + event}T11:30:00Z"),
    )


def add_fixture(conn: sqlite3.Connection, fixture_id: int, event: int, team_h: int, team_a: int) -> None:
    conn.execute(
        "INSERT INTO fixtures(id, event, team_h, team_a, kickoff_time, finished, started, raw_json,"
        " updated_at) VALUES (?,?,?,?,?,0,0,'{}','2026-09-01T00:00:00Z')",
        (fixture_id, event, team_h, team_a, f"2026-09-{20 + event}T14:00:00Z"),
    )


def add_run(
    conn: sqlite3.Connection,
    run_id: int,
    family: str,
    event: int,
    *,
    cutoff: str = CUTOFF,
    version: str | None = None,
    status: str = "complete",
    code_snapshot: str | None = CODE_SNAPSHOT,
    context_hash: str | None = CONTEXT_HASH,
) -> None:
    conn.execute(
        "INSERT INTO projection_runs(id, model_family, model_version, generated_at, planning_event,"
        " data_cutoff, status, source_snapshot_sha256, planning_context_hash)"
        " VALUES (?,?,?,'2026-09-19T11:01:00Z',?,?,?,?,?)",
        (
            run_id,
            family,
            version if version is not None else VERSIONS[family],
            event,
            cutoff,
            status,
            code_snapshot,
            context_hash,
        ),
    )


def add_xpts_row(
    conn: sqlite3.Connection,
    run_id: int,
    fixture_id: int,
    event: int,
    *,
    minutes_run_id: int,
    team_run_id: int,
    rate_run_id: int,
    player_id: int = 1,
) -> None:
    conn.execute(
        "INSERT INTO player_fixture_xpts_projections(projection_run_id, player_id, fixture_id, event,"
        " team_id, opponent_id, position, minutes_run_id, team_run_id, rate_run_id, payload_json,"
        " model_version, scoring_rules_version, generated_at)"
        " VALUES (?,?,?,?,1,2,'MID',?,?,?,'{}','xpts_v1','v1','2026-09-19T11:01:00Z')",
        (run_id, player_id, fixture_id, event, minutes_run_id, team_run_id, rate_run_id),
    )


def add_mc_row(
    conn: sqlite3.Connection,
    run_id: int,
    fixture_id: int,
    event: int,
    *,
    xpts_run_id: int,
    minutes_run_id: int,
    team_run_id: int,
    rate_run_id: int,
    player_id: int = 1,
) -> None:
    conn.execute(
        "INSERT INTO monte_carlo_distributions(projection_run_id, player_id, fixture_id, event,"
        " team_id, opponent_id, position, xpts_run_id, minutes_run_id, team_run_id, rate_run_id,"
        " payload_json, model_version, generated_at)"
        " VALUES (?,?,?,?,1,2,'MID',?,?,?,?,'{}','mc_v1','2026-09-19T11:01:00Z')",
        (run_id, player_id, fixture_id, event, xpts_run_id, minutes_run_id, team_run_id, rate_run_id),
    )


def synthetic_world(
    events: Sequence[int] = HORIZON,
    *,
    fixtures_per_event: int = 1,
    blanks: Sequence[int] = (),
    cutoff: str = CUTOFF,
) -> tuple[sqlite3.Connection, dict[int, dict[str, int]]]:
    """A coherent five-family world for every event.  Returns (conn, runs_by_event).

    ``blanks`` names events that are a BLANK Gameweek: they have their five runs but
    no fixture and therefore no player x fixture row, which is what PE-4 calls a
    valid zero-fixture world rather than missing data.
    """

    conn = connect_database(":memory:")
    base_world(conn)
    runs_by_event: dict[int, dict[str, int]] = {}
    next_run = 100
    next_fixture = 1000
    with conn:
        for event in events:
            add_event(conn, event)
            fixture_ids = []
            if int(event) not in {int(blank) for blank in blanks}:
                for _ in range(fixtures_per_event):
                    add_fixture(conn, next_fixture, event, 1, 2)
                    fixture_ids.append(next_fixture)
                    next_fixture += 1
            ids = {
                "minutes_v1": next_run,
                "team_strength_v1": next_run + 1,
                "player_rates_v1": next_run + 2,
                "xpts_v1": next_run + 3,
                "monte_carlo_v1": next_run + 4,
            }
            next_run += 5
            for family, run_id in ids.items():
                add_run(conn, run_id, family, event, cutoff=cutoff)
            for fixture_id in fixture_ids:
                add_xpts_row(
                    conn,
                    ids["xpts_v1"],
                    fixture_id,
                    event,
                    minutes_run_id=ids["minutes_v1"],
                    team_run_id=ids["team_strength_v1"],
                    rate_run_id=ids["player_rates_v1"],
                )
                add_mc_row(
                    conn,
                    ids["monte_carlo_v1"],
                    fixture_id,
                    event,
                    xpts_run_id=ids["xpts_v1"],
                    minutes_run_id=ids["minutes_v1"],
                    team_run_id=ids["team_strength_v1"],
                    rate_run_id=ids["player_rates_v1"],
                )
            runs_by_event[int(event)] = ids
    return conn, runs_by_event


def world_with_run_ids(
    runs_by_event: Mapping[int, Mapping[str, int]],
    *,
    cutoff: str = CUTOFF,
    code_snapshot: str | None = CODE_SNAPSHOT,
    context_hash: str | None = CONTEXT_HASH,
) -> tuple[sqlite3.Connection, dict[int, dict[str, int]]]:
    """A world built from EXPLICIT run ids, so a test's existing ids stay its own.

    Only the run rows are created: this is the shape a cache-path or transport test
    needs, where the certified run ids are the fixture's and no simulation happens.
    """

    conn = connect_database(":memory:")
    base_world(conn)
    with conn:
        for event, runs in runs_by_event.items():
            add_event(conn, int(event))
            for family, run_id in runs.items():
                add_run(
                    conn, int(run_id), str(family), int(event), cutoff=cutoff,
                    code_snapshot=code_snapshot, context_hash=context_hash,
                )
    return conn, {int(event): dict(runs) for event, runs in runs_by_event.items()}


def write_snapshot(path: Path) -> dict[str, Any]:
    """A real file the generation can pin, so retention checks have something to see."""

    from fpl_brain import execution_snapshot as es

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fpl-synthetic-snapshot")
    return {
        "path": str(path),
        "sha256": es.file_sha256(path),
        "size_bytes": int(path.stat().st_size),
        "source_db_identity": {"path": ":memory:", "schema_version": "18"},
        "execution_run_uuid": "00000000-0000-0000-0000-000000000001",
    }


def certify_world(
    conn: sqlite3.Connection,
    runs_by_event: Mapping[int, Mapping[str, int]],
    *,
    events: Sequence[int] | None = None,
    planning_event: int | None = None,
    cutoff: str = CUTOFF,
    horizon_kind: str = gs.HORIZON_KIND_FOUR_GW,
    snapshot_path: Path | None = None,
    calibration: Mapping[str, Any] | None = None,
) -> gs.CertifiedGeneration:
    """Certify the synthetic world through the REAL generation-store lifecycle."""

    resolved_events = [int(event) for event in (events if events is not None else runs_by_event)]
    snapshot = write_snapshot(snapshot_path) if snapshot_path is not None else {
        "path": None, "sha256": None, "size_bytes": None,
        "source_db_identity": None, "execution_run_uuid": "test-run",
    }
    return gs.certify_generation(
        conn,
        planning_event=int(planning_event if planning_event is not None else resolved_events[0]),
        horizon_kind=horizon_kind,
        cutoff=str(cutoff),
        runs_by_event={int(event): dict(runs) for event, runs in runs_by_event.items()},
        events=resolved_events,
        snapshot=snapshot,
        calibration=calibration,
        code_snapshot_sha256=CODE_SNAPSHOT,
        clock=lambda: "2026-09-19T11:02:00Z",
    )


__all__ = [
    "CODE_SNAPSHOT",
    "CONTEXT_HASH",
    "CUTOFF",
    "DATA_SNAPSHOT",
    "HORIZON",
    "OTHER_CUTOFF",
    "VERSIONS",
    "add_event",
    "add_fixture",
    "add_mc_row",
    "add_run",
    "add_xpts_row",
    "base_world",
    "certify_world",
    "synthetic_world",
    "world_with_run_ids",
    "write_snapshot",
]
