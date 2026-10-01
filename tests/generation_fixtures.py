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
import tempfile
import json
import shutil
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from fpl_brain import analytics
from fpl_brain import certified_bundle as cb
from fpl_brain import generation_store as gs
from fpl_brain.database import connect_database

CUTOFF = "2026-09-19T11:00:00Z"
OTHER_CUTOFF = "2026-09-18T11:00:00Z"
#: The AUTHORITATIVE code identity the generation store derives for itself.  A run
#: row records which code revision produced it, and certification now REQUIRES the
#: runs to record the running revision, so the fixtures stamp the same value the
#: store computes instead of a literal that no code identity could reproduce.
CODE_SNAPSHOT = analytics.source_snapshot_sha256()
DATA_SNAPSHOT = "sha256:" + "d" * 64
CONTEXT_HASH = "ctx"

#: The REAL declared versions, read from the one source rather than restated, so a
#: future version bump moves the fixtures with it instead of silently un-pinning.
VERSIONS = cb.declared_required_versions()

#: The four-event normal-transfer horizon the contract fixes.
HORIZON = (5, 6, 7, 8)
_FIXTURE_SNAPSHOTS: dict[int, dict[str, Any]] = {}
_FIXTURE_SNAPSHOTS_BY_DIGEST: dict[str, dict[str, Any]] = {}


def prepare_fixture_snapshot(
    conn: sqlite3.Connection, events: Sequence[int], *, cutoff: str = CUTOFF
) -> dict[str, Any]:
    """Capture the source state before prediction rows are inserted in a fixture."""

    from fpl_brain.planning import get_planning_context

    source_dir = Path(tempfile.mkdtemp(prefix="fpl-pe9-source-"))
    snapshot = write_snapshot(
        source_dir / "source.db", database=conn, planning_cutoff=str(cutoff)
    )
    inputs = {
        "entry_id": 1,
        "season": None,
        "as_of": str(cutoff),
        "scouting_stale_after_days": None,
        "official_price_stale_after_hours": None,
    }
    snapshot_conn = sqlite3.connect(f"file:{Path(snapshot['path'])}?mode=ro", uri=True)
    snapshot_conn.row_factory = sqlite3.Row
    try:
        context_hashes = {
            int(event): analytics.planning_context_reference(
                get_planning_context(
                    snapshot_conn, 1, int(event), as_of=str(cutoff), season=None,
                    scouting_stale_after_days=None, official_price_stale_after_hours=None,
                )
            )
            for event in events
        }
    finally:
        snapshot_conn.close()
    record = {"snapshot": snapshot, "inputs": inputs, "context_hashes": context_hashes}
    _FIXTURE_SNAPSHOTS[id(conn)] = record
    _FIXTURE_SNAPSHOTS_BY_DIGEST[str(snapshot["sha256"])] = record
    return record


def register_fixture_snapshot(conn: sqlite3.Connection, record: Mapping[str, Any]) -> None:
    """Attach an already captured canonical source snapshot to a copied test store."""

    copied = dict(record)
    _FIXTURE_SNAPSHOTS[id(conn)] = copied
    _FIXTURE_SNAPSHOTS_BY_DIGEST[str(copied["snapshot"]["sha256"])] = copied


def fixture_run_provenance(conn: sqlite3.Connection, event: int) -> dict[str, Any]:
    record = _FIXTURE_SNAPSHOTS.get(id(conn))
    if record is None:
        return {}
    snapshot = record["snapshot"]
    return {
        "data_snapshot_sha256": str(snapshot["sha256"]),
        "execution_run_uuid": str(snapshot["execution_run_uuid"]),
        "planning_context_inputs_json": json.dumps(record["inputs"], sort_keys=True),
        "planning_context_hash": record["context_hashes"].get(int(event)),
    }


def fixture_snapshot(conn: sqlite3.Connection) -> dict[str, Any]:
    record = _FIXTURE_SNAPSHOTS.get(id(conn))
    if record is None:
        raise ValueError("the fixture source snapshot was not prepared before projection rows")
    return dict(record["snapshot"])


def fixture_search_permission_evaluation(
    source_identity: Mapping[str, Any],
    *,
    events: Sequence[int] | None = None,
    horizon_kind: str = gs.HORIZON_KIND_FOUR_GW,
) -> dict[str, Any]:
    """Build explicit fixture-only permission evidence for tests outside the gate.

    Production code never imports this helper. The actual permission decision is
    exercised by ``test_production_search_permission``; legacy builder/lifecycle
    tests may use this fixture evidence to keep their intended test target isolated.
    """

    from fpl_brain import history_completeness as hc, search_permission as sp

    generation_id = str(source_identity["generation_id"])
    planning_event = int(source_identity.get("planning_event") or -1)
    cutoff = str(source_identity.get("origin_cutoff") or "")
    snapshot_sha256 = str(source_identity.get("data_snapshot_sha256") or "")
    selected_events = list(
        events if events is not None else range(planning_event, planning_event + 4)
    )
    history = {
        "schema": hc.HISTORY_COMPLETENESS_SCHEMA,
        "planning_event": planning_event,
        "cutoff": cutoff,
        "required_completed_events": [],
        "latest_required_completed_event": planning_event - 1,
        "complete": True,
        "blocker": None,
        "reasons": [],
        "detail": "explicit fixture-only complete history evidence",
    }
    body = {
        "schema": sp.SEARCH_PERMISSION_EVALUATION_SCHEMA,
        "permitted": True,
        "reasons": [],
        "origin": {
            "generation_id": generation_id,
            "generation_manifest_sha256": generation_id,
            "planning_event": planning_event,
            "cutoff": cutoff,
            "horizon_kind": horizon_kind,
            "events": selected_events,
            "data_snapshot_sha256": snapshot_sha256,
            "execution_run_uuid": "fixture-execution-run",
            "predictive_code_snapshot_sha256": source_identity.get(
                "predictive_code_snapshot_sha256"
            ),
        },
        "conditions": {
            "temporal_status": "CAUSAL",
            "causal_evidence_sha256": "sha256:" + "f" * 64,
            "dependency_validation": "COHERENT",
            "horizon_status": "DECISION_HORIZON_COMPLETE",
            "data_snapshot_sha256": snapshot_sha256,
            "snapshot_identity": "VERIFIED",
            "snapshot_error": None,
            "history_completeness": history,
        },
    }
    return {**body, "evaluation_sha256": sp.evaluation_digest(body)}


def _fixture_record_for_runs(
    conn: sqlite3.Connection, runs_by_event: Mapping[int, Mapping[str, int]]
) -> dict[str, Any] | None:
    run_ids = sorted({int(run_id) for rows in runs_by_event.values() for run_id in rows.values()})
    if not run_ids:
        return _FIXTURE_SNAPSHOTS.get(id(conn))
    placeholders = ",".join("?" for _ in run_ids)
    rows = conn.execute(
        f"SELECT DISTINCT data_snapshot_sha256 FROM projection_runs WHERE id IN ({placeholders})",
        run_ids,
    ).fetchall()
    digests = {str(row[0]) for row in rows if row[0]}
    if len(digests) == 1:
        digest = next(iter(digests))
        record = _FIXTURE_SNAPSHOTS.get(id(conn))
        if record is not None and str(record["snapshot"].get("sha256")) == digest:
            return record
        return _FIXTURE_SNAPSHOTS_BY_DIGEST.get(digest)
    return None


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
    provenance_override: Mapping[str, Any] | None = None,
) -> None:
    provenance = fixture_run_provenance(conn, event)
    if context_hash == CONTEXT_HASH and provenance:
        context_hash = provenance.pop("planning_context_hash")
    if provenance_override is not None:
        provenance.update(provenance_override)
    conn.execute(
        "INSERT INTO projection_runs(id, model_family, model_version, generated_at, planning_event,"
        " data_cutoff, status, source_snapshot_sha256, planning_context_hash, data_snapshot_sha256,"
        " execution_run_uuid, planning_context_inputs_json)"
        " VALUES (?,?,?,'2026-09-19T11:01:00Z',?,?,?,?,?,?,?,?)",
        (
            run_id,
            family,
            version if version is not None else VERSIONS[family],
            event,
            cutoff,
            status,
            code_snapshot,
            context_hash,
            provenance.get("data_snapshot_sha256"),
            provenance.get("execution_run_uuid"),
            provenance.get("planning_context_inputs_json"),
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
    fixture_rows: dict[int, list[int]] = {}
    with conn:
        for event in events:
            add_event(conn, event)
            fixture_ids = []
            if int(event) not in {int(blank) for blank in blanks}:
                for _ in range(fixtures_per_event):
                    add_fixture(conn, next_fixture, event, 1, 2)
                    fixture_ids.append(next_fixture)
                    next_fixture += 1
            fixture_rows[int(event)] = fixture_ids
    prepare_fixture_snapshot(conn, events, cutoff=cutoff)
    with conn:
        for event in events:
            fixture_ids = fixture_rows[int(event)]
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
    prepare_fixture_snapshot(conn, list(runs_by_event), cutoff=cutoff)
    with conn:
        for event, runs in runs_by_event.items():
            for family, run_id in runs.items():
                add_run(
                    conn, int(run_id), str(family), int(event), cutoff=cutoff,
                    code_snapshot=code_snapshot, context_hash=context_hash,
                )
    return conn, {int(event): dict(runs) for event, runs in runs_by_event.items()}


#: Where a fixture's snapshot lives when the caller does not name a file.  ONE path
#: per process, so two certifications of the SAME world resolve to the SAME
#: generation id (that idempotence is part of what the store promises), while a
#: caller-named path still produces a different world identity.
_DEFAULT_SNAPSHOT = Path(tempfile.mkdtemp(prefix="fpl-pe9-fixture-")) / "snapshot.db"


def _write_fixture_snapshot_manifest(
    snapshot: Mapping[str, Any], *, planning_cutoff: str = CUTOFF
) -> dict[str, Any]:
    """Write canonical timing evidence for explicitly synthetic PE-9 fixtures."""

    path = Path(str(snapshot["path"]))
    manifest_path = path.with_name(path.name + ".execution_source_snapshot.json")
    cutoff = str(planning_cutoff)
    captured = datetime.fromisoformat(cutoff.replace("Z", "+00:00"))
    completed_at = (captured + timedelta(seconds=1)).isoformat(timespec="seconds").replace("+00:00", "Z")
    execution_run_uuid = str(snapshot.get("execution_run_uuid") or "")
    payload = {
        "data_snapshot_path": str(path),
        "data_snapshot_sha256": str(snapshot["sha256"]),
        "data_snapshot_created_at": cutoff,
        "snapshot_lock_acquired_at": cutoff,
        "snapshot_capture_started_at": cutoff,
        "snapshot_consistency_at": cutoff,
        "snapshot_capture_completed_at": completed_at,
        "snapshot_capture_seconds": 1.0,
        "data_snapshot_source_db_identity": dict(snapshot["source_db_identity"]),
        "execution_run_uuid": execution_run_uuid,
        "planning_cutoff": cutoff,
        "data_snapshot_size_bytes": int(snapshot["size_bytes"]),
        "data_snapshot_manifest": str(manifest_path),
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {**dict(snapshot), "manifest_path": str(manifest_path)}


def ensure_fixture_execution_run(
    conn: sqlite3.Connection, *, planning_event: int, cutoff: str, snapshot: Mapping[str, Any]
) -> None:
    """Bind a synthetic snapshot to a run row after its recorded consistency time."""

    manifest_path = snapshot.get("manifest_path") or snapshot.get("data_snapshot_manifest")
    if not manifest_path or not Path(str(manifest_path)).is_file():
        return
    payload = json.loads(Path(str(manifest_path)).read_text(encoding="utf-8"))
    if (
        str(payload.get("planning_cutoff") or "") != str(cutoff)
        or str(payload.get("data_snapshot_path") or "") != str(snapshot.get("path") or "")
        or str(payload.get("execution_run_uuid") or "") != str(snapshot.get("execution_run_uuid") or "")
    ):
        return
    completed = datetime.fromisoformat(
        str(payload["snapshot_capture_completed_at"]).replace("Z", "+00:00")
    )
    started_at = (completed + timedelta(seconds=1)).isoformat(timespec="seconds").replace("+00:00", "Z")
    hard_stop_at = (completed + timedelta(hours=1)).isoformat(timespec="seconds").replace("+00:00", "Z")
    with conn:
        conn.execute(
            """INSERT OR IGNORE INTO execution_runs(
                   run_uuid, planning_event, planning_cutoff, semantic_run_key, status,
                   started_at, hard_stop_at, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                str(snapshot["execution_run_uuid"]), int(planning_event), str(cutoff),
                f"fixture-causal:{snapshot['execution_run_uuid']}", "COMPLETE", started_at,
                hard_stop_at, str(payload["snapshot_consistency_at"]), started_at,
            ),
        )


def write_snapshot(
    path: Path, *, database: str | Path | sqlite3.Connection, planning_cutoff: str = CUTOFF
) -> dict[str, Any]:
    """A real file the generation can pin, so retention checks have something to see.

    The bytes are a deterministic function of the file NAME, so a fixture that pins a
    different snapshot gets a different world identity -- which is what makes
    "a second generation over the same runs but another snapshot" a different
    generation rather than an accident of the clock.

    ``database`` names an existing store to copy: a DECISION reads causal source state
    from the pinned snapshot, so a fixture that drives one must pin a real database
    rather than a placeholder file.
    """

    from fpl_brain import execution_snapshot as es

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    _copy_database(database, path)
    snapshot_conn = sqlite3.connect(str(path))
    try:
        # Give two otherwise identical fixture snapshots distinct real database
        # bytes, without changing any FPL tables read by the decision pipeline.
        snapshot_conn.execute(
            "CREATE TABLE IF NOT EXISTS __pe9_fixture_snapshot_identity(value TEXT NOT NULL)"
        )
        snapshot_conn.execute("DELETE FROM __pe9_fixture_snapshot_identity")
        snapshot_conn.execute(
            "INSERT INTO __pe9_fixture_snapshot_identity(value) VALUES (?)", (path.name,)
        )
        snapshot_conn.commit()
    finally:
        snapshot_conn.close()
    if isinstance(database, sqlite3.Connection):
        schema = database.execute(
            "SELECT value FROM schema_meta WHERE key='schema_version'"
        ).fetchone()
        runs = database.execute("SELECT COUNT(*), MAX(id) FROM projection_runs").fetchone()
        identity = {
            "path": str(path),
            "schema_version": str(schema[0]) if schema else None,
            "projection_runs_count": int(runs[0]),
            "projection_runs_max_id": int(runs[1]) if runs[1] is not None else None,
        }
    else:
        identity = es.source_db_identity(database)
    snapshot = {
        "path": str(path),
        "sha256": es.file_sha256(path),
        "size_bytes": int(path.stat().st_size),
        "source_db_identity": identity,
        "execution_run_uuid": str(uuid.uuid5(uuid.NAMESPACE_URL, f"fpl-test-snapshot:{es.file_sha256(path)}")),
    }
    return _write_fixture_snapshot_manifest(snapshot, planning_cutoff=planning_cutoff)


def _write_default_snapshot(
    *, database: str | Path | sqlite3.Connection
) -> dict[str, Any]:
    """Retain one stable default snapshot per distinct fixture database.

    Many test modules certify different synthetic sources in one Python process. A
    single reusable path lets a later fixture overwrite a still-referenced snapshot,
    which correctly makes production verification refuse. Use the fixed default name
    only as a staging file, then keep each byte-distinct snapshot at a digest-named
    path. Re-certifying identical source bytes resolves to the same retained path.
    """

    from fpl_brain import execution_snapshot as es

    staged = write_snapshot(_DEFAULT_SNAPSHOT, database=database)
    digest = str(staged["sha256"])
    retained_path = _DEFAULT_SNAPSHOT.with_name(f"snapshot-{digest}.db")
    if retained_path.exists():
        if es.file_sha256(retained_path) != digest:
            raise RuntimeError(
                f"fixture snapshot path {retained_path} exists with bytes outside its digest name"
            )
        _DEFAULT_SNAPSHOT.unlink()
    else:
        _DEFAULT_SNAPSHOT.replace(retained_path)
    old_manifest = Path(str(staged["manifest_path"]))
    snapshot = {
        **staged,
        "path": str(retained_path),
        "source_db_identity": es.source_db_identity(retained_path),
        "manifest_path": str(retained_path.with_name(retained_path.name + ".execution_source_snapshot.json")),
    }
    if old_manifest.exists() and old_manifest != Path(snapshot["manifest_path"]):
        old_manifest.unlink()
    return _write_fixture_snapshot_manifest(snapshot)


def _copy_database(source: str | Path | sqlite3.Connection, target: Path) -> None:
    """Create a real fixture snapshot without post-certification PE-9 rows."""

    from fpl_brain.database import connect_database

    target_conn = connect_database(target)
    target_conn.execute("PRAGMA foreign_keys=OFF")
    close_source = False
    try:
        if isinstance(source, sqlite3.Connection):
            source_conn = source
        else:
            source_conn = sqlite3.connect(str(source))
            source_conn.row_factory = sqlite3.Row
            close_source = True
        source_tables = [
            str(row[0])
            for row in source_conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
            if not str(row[0]).startswith("sqlite_")
            and str(row[0]) not in {"generation", "current_generation", "engine_decision_records"}
            and not str(row[0]).startswith("execution_")
        ]
        target_tables = {
            str(row[0])
            for row in target_conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        for table in source_tables:
            if table not in target_tables:
                continue
            columns = [
                str(row[1])
                for row in source_conn.execute(f'PRAGMA table_info("{table}")')
            ]
            if not columns:
                continue
            source_rows = source_conn.execute(f'SELECT * FROM "{table}"').fetchall()
            if not source_rows:
                continue
            column_sql = ",".join(f'"{column}"' for column in columns)
            values_sql = ",".join("?" for _ in columns)
            target_conn.executemany(
                f'INSERT OR REPLACE INTO "{table}" ({column_sql}) VALUES ({values_sql})',
                [tuple(row[column] for column in columns) for row in source_rows],
            )
        target_conn.commit()
    finally:
        if close_source:
            source_conn.close()
        target_conn.close()


def certify_world(
    conn: sqlite3.Connection,
    runs_by_event: Mapping[int, Mapping[str, int]],
    *,
    events: Sequence[int] | None = None,
    planning_event: int | None = None,
    cutoff: str = CUTOFF,
    horizon_kind: str = gs.HORIZON_KIND_FOUR_GW,
    snapshot_path: Path | None = None,
    snapshot_database: str | Path | None = None,
    calibration_artifact_ref: str | Path | None = None,
    require_calibration: bool = False,
    calibration: Mapping[str, Any] | None = None,
) -> gs.CertifiedGeneration:
    """Certify the synthetic world through the REAL generation-store lifecycle.

    The snapshot is ALWAYS pinned: a generation commits to the immutable source it
    replaced, so a fixture that certified a world without one would be exercising a
    lifecycle production does not have.
    """

    from fpl_brain import execution_snapshot as es

    resolved_events = [int(event) for event in (events if events is not None else runs_by_event)]
    fixture_record = _fixture_record_for_runs(conn, runs_by_event)
    if snapshot_database is not None:
        source_database = snapshot_database
        target = snapshot_path or _DEFAULT_SNAPSHOT
        snapshot = write_snapshot(target, database=source_database, planning_cutoff=str(cutoff))
    elif fixture_record is not None:
        snapshot = dict(fixture_record["snapshot"])
        if snapshot_path is not None:
            target = Path(snapshot_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            if Path(snapshot["path"]).resolve() != target.resolve():
                shutil.copyfile(snapshot["path"], target)
            snapshot = _write_fixture_snapshot_manifest({
                **snapshot,
                "path": str(target),
                "size_bytes": int(target.stat().st_size),
                "sha256": es.file_sha256(target),
                "manifest_path": None,
            }, planning_cutoff=str(cutoff))
    else:
        source_database = conn
        snapshot = (
            write_snapshot(snapshot_path, database=source_database)
            if snapshot_path is not None
            else _write_default_snapshot(database=source_database)
        )
    if calibration is not None and calibration_artifact_ref is None:
        calibration_artifact_ref = Path(snapshot["path"]).with_name("pe8_calibration_evidence.json")
        Path(calibration_artifact_ref).write_text(
            json.dumps(calibration, sort_keys=True, separators=(",", ":"), default=str),
            encoding="utf-8",
        )
    return certify_under_writer_lease(
        conn,
        planning_event=int(planning_event if planning_event is not None else resolved_events[0]),
        horizon_kind=horizon_kind,
        cutoff=str(cutoff),
        runs_by_event={int(event): dict(runs) for event, runs in runs_by_event.items()},
        events=resolved_events,
        snapshot=snapshot,
        calibration=calibration,
        calibration_artifact_ref=calibration_artifact_ref,
        require_calibration=require_calibration,
        clock=lambda: "2026-09-19T11:02:00Z",
    )


def certify_under_writer_lease(conn: sqlite3.Connection, **kwargs) -> gs.CertifiedGeneration:
    """Exercise generation publication under a real active controller lease."""

    from fpl_brain import execution

    snapshot = kwargs.get("snapshot")
    if isinstance(snapshot, Mapping):
        ensure_fixture_execution_run(
            conn,
            planning_event=int(kwargs.get("planning_event") or 0),
            cutoff=str(kwargs.get("cutoff") or CUTOFF),
            snapshot=snapshot,
        )
    controller = execution.ExecutionController(conn)
    controller.create_run(
        planning_event=int(kwargs.get("planning_event") or 0),
        planning_cutoff=str(kwargs.get("cutoff") or CUTOFF),
        hard_stop_at=datetime.now(timezone.utc) + timedelta(hours=1),
        label="pe9_test_certification",
    )
    controller.start()
    controller.acquire_writer_lease()
    try:
        result = gs.certify_generation(conn, controller=controller, **kwargs)
    except BaseException:
        controller.finish(execution.RUN_FAILED, "fixture certification failed")
        raise
    controller.finish(execution.RUN_COMPLETE)
    return result


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
    "certify_under_writer_lease",
    "fixture_run_provenance",
    "fixture_snapshot",
    "register_fixture_snapshot",
    "prepare_fixture_snapshot",
    "synthetic_world",
    "world_with_run_ids",
    "write_snapshot",
]
