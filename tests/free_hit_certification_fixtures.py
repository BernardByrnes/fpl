"""A REAL v2 certification artifact fixture, built the way the certifier builds one.

Synthesised in shape only: every field is the one
``scripts/certify_gw5_gw8.py::build_certification_payload`` produces, and the
fixture must survive the REAL loader (``four_gw_decision.load_certification_artifact``).
Nothing here invents a Free-Hit-only field.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fpl_brain import certified_bundle as cb  # noqa: E402
from fpl_brain import four_gw_decision as fg  # noqa: E402
from fpl_brain.free_hit_decision_authority import (  # noqa: E402
    model_label,
    runs_label,
)

DATA_SNAPSHOT = "sha256:" + "d" * 64
CUTOFF = "2026-09-16T11:00:00Z"
#: The AUTHORITATIVE code identity: certification now requires the run rows to record
#: the running revision, and the identity-bearing bundle payload records the same
#: value, so a consumer recomputes exactly the identity the certifier minted.
from fpl_brain import analytics as _analytics  # noqa: E402

CODE_SNAPSHOT = _analytics.source_snapshot_sha256()
CONTEXT_HASH = "sha256:" + "p" * 64
#: The REAL run families the certifier records, so the canonical world-cache key
#: (which names them minutes/team/rate/xpts) can be derived from the bundle.
RUNS: dict[str, int] = {
    "minutes_v1": 101, "team_strength_v1": 102,
    "player_rates_v1": 103, "xpts_v1": 104,
}
#: How far apart two events' run ids sit.  Ids are globally unique in the real
#: table, so a per-event offset of ONE would make event 6's minutes run id equal
#: event 5's team run id -- two families claiming one row, which no real
#: certification can produce.
RUN_ID_STRIDE = 10
#: The REAL declared versions of the frozen families, read from the ONE declared
#: source rather than restated.  A certification that records a model version
#: nobody pins is exactly what ``UNSUPPORTED_MODEL_VERSION`` exists to refuse, so
#: a fixture that declared stale literals would be presenting a run no accepted
#: model produced.
MODEL_VERSIONS: dict[str, str] = {
    family: version
    for family, version in cb.declared_required_versions().items()
    if family in RUNS
}


def runs_for(event: int) -> dict[str, int]:
    """The certified run ids for ONE event.

    Real certifications commit DISTINCT runs per event -- the four Gameweeks are
    separate planning events with their own projection runs.  The previous fixture
    reused one mapping for all four, which masked a defect where the H1 predictive
    identity was stamped onto H2-H4.
    """

    offset = (int(event) - 5) * RUN_ID_STRIDE
    return {family: run_id + offset for family, run_id in RUNS.items()}


def seed_certified_runs(
    conn: Any, events: Sequence[int] = (5, 6, 7, 8), *, generated_at: str | None = None
) -> dict[int, dict[str, int]]:
    """Insert the ``projection_runs`` rows this fixture's certification DECLARES.

    A certified run id is not merely a number in an artifact: it names a row in
    ``projection_runs``, which is where a loader reads the model version the run was
    built from -- and an absent row is a refusal, never an empty legacy run.  A
    fixture database that declares run ids it does not carry is therefore not a
    certified generation, so the rows are inserted here with the SAME versions,
    cutoff and code identity the artifact declares.
    """

    import generation_fixtures as gf
    from fpl_brain.database import connect_database
    from fpl_brain import execution_snapshot as es

    stamped = str(generated_at or CUTOFF)
    events = tuple(int(event) for event in events)
    # This fixture models a stable predictive source independent of manager state.
    # Capture its canonical facts before inserting prediction runs or caller-specific
    # manager rows, then attach the retained snapshot to the live test store.
    source_conn = connect_database(":memory:")
    base_world(source_conn)
    source_conn.execute(
        "UPDATE schema_meta SET value=? WHERE key='created_at'", ("2026-09-16T11:00:00Z",)
    )
    with source_conn:
        for event in events:
            source_conn.execute(
                "INSERT OR IGNORE INTO fixtures(id, event, team_h, team_a, kickoff_time, finished,"
                " started, raw_json, updated_at) VALUES (?,?,?,?,?,0,0,'{}',?)",
                (5000 + event, event, 100, 101, f"2026-09-{event + 11:02d}T14:00:00Z", CUTOFF),
            )
    source_record = gf.prepare_fixture_snapshot(source_conn, events, cutoff=CUTOFF)
    stable_snapshot_path = _SNAPSHOT_PATH
    stable_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    if Path(source_record["snapshot"]["path"]).resolve() != stable_snapshot_path.resolve():
        shutil.copyfile(source_record["snapshot"]["path"], stable_snapshot_path)
    source_record = {
        **source_record,
        "snapshot": {
            **source_record["snapshot"],
            "path": str(stable_snapshot_path),
            "source_db_identity": es.source_db_identity(stable_snapshot_path),
        },
    }
    gf.register_fixture_snapshot(conn, source_record)
    source_conn.close()
    with conn:
        for event in events:
            conn.execute(
                "INSERT OR IGNORE INTO fixtures(id, event, team_h, team_a, kickoff_time, finished,"
                " started, raw_json, updated_at) VALUES (?,?,?,?,?,0,0,'{}',?)",
                (5000 + event, event, 100, 101, f"2026-09-{event + 11:02d}T14:00:00Z", CUTOFF),
            )
    runs_by_event: dict[int, dict[str, int]] = {}
    for event in events:
        runs = runs_for(int(event))
        provenance = gf.fixture_run_provenance(conn, int(event))
        context_hash = provenance.pop("planning_context_hash")
        for family, run_id in runs.items():
            conn.execute(
                "INSERT INTO projection_runs(id, model_family, model_version, generated_at,"
                " planning_event, planning_context_hash, data_cutoff, status, source_snapshot_sha256,"
                " data_snapshot_sha256, execution_run_uuid, planning_context_inputs_json)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    int(run_id), str(family), str(MODEL_VERSIONS[family]), stamped, int(event),
                    context_hash, CUTOFF, "complete", CODE_SNAPSHOT,
                    provenance["data_snapshot_sha256"], provenance["execution_run_uuid"],
                    provenance["planning_context_inputs_json"],
                ),
            )
        runs_by_event[int(event)] = {family: int(run_id) for family, run_id in runs.items()}
    return runs_by_event


def seed_certified_world(
    conn: Any,
    events: Sequence[int] = (5, 6, 7, 8),
    *,
    cutoff: str = CUTOFF,
    certify: bool = True,
):
    """Seed the rows a CERTIFIED GENERATION re-proves, then certify one.

    ``seed_certified_runs`` inserts the ``projection_runs`` rows the certification
    declares; a generation additionally re-proves the dependency closure from the
    ROWS, so the fixture world also needs the fixture, xPts and Monte Carlo rows.  A
    generation is then minted through the REAL ``generation_store.certify_generation``
    lifecycle, so what a test loads is what production would have persisted.

    The world is synchronised with a REAL pinned snapshot file: a generation commits
    to the immutable source it replaced, and certification refuses a world that pinned
    nothing.  Its content is fixed, so the generation this fixture mints is the same
    one ``world_cache_generation_id`` resolves in a throwaway store -- the cache key a
    test writes at import time is the key its own store derives.
    """

    from fpl_brain import generation_store as gs

    events = tuple(int(event) for event in events)
    runs_by_event: dict[int, dict[str, int]] = {}
    fixtures_by_event: dict[int, int] = {}
    with conn:
        for event in events:
            runs = runs_for(event)
            fixture_id = 5000 + event
            fixtures_by_event[event] = fixture_id
            conn.execute(
                "INSERT OR IGNORE INTO fixtures(id, event, team_h, team_a, kickoff_time, finished,"
                " started, raw_json, updated_at) VALUES (?,?,?,?,?,0,0,'{}',?)",
                (fixture_id, event, 100, 101, f"2026-09-{event + 11:02d}T14:00:00Z", CUTOFF),
            )
            conn.execute(
                "INSERT OR IGNORE INTO player_fixture_xpts_projections(projection_run_id, player_id,"
                " fixture_id, event, team_id, opponent_id, position, minutes_run_id, team_run_id,"
                " rate_run_id, payload_json, model_version, scoring_rules_version, generated_at)"
                " VALUES (?,?,?,?,100,101,'MID',?,?,?,'{}','xpts_v1','v1',?)",
                (int(runs["xpts_v1"]), 1, fixture_id, event, int(runs["minutes_v1"]),
                 int(runs["team_strength_v1"]), int(runs["player_rates_v1"]), CUTOFF),
            )
            runs_by_event[event] = {family: int(run_id) for family, run_id in runs.items()}
    if not certify:
        return None, runs_by_event

    import generation_fixtures as gf

    generation = gf.certify_world(
        conn,
        runs_by_event,
        planning_event=events[0],
        cutoff=str(cutoff),
        events=events,
        horizon_kind=gs.HORIZON_KIND_FOUR_GW,
    )
    return generation, runs_by_event


#: The pinned snapshot a certified generation commits to.  It contains only the
#: synthetic predictive source rows, so manager-state rows added by an individual
#: consumer test do not change the generation/cache identity.
_SNAPSHOT_DIR = Path(tempfile.mkdtemp(prefix="fpl-pe9-freehit-"))
_SNAPSHOT_PATH = _SNAPSHOT_DIR / "source.db"


def pinned_snapshot(database: Any) -> dict[str, Any]:
    """Create a real, deterministic SQLite snapshot of the fixture's predictive rows."""

    from fpl_brain import execution_snapshot as es
    from fpl_brain.database import connect_database

    _SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    if not _SNAPSHOT_PATH.exists():
        snapshot = connect_database(_SNAPSHOT_PATH)
        snapshot.execute("PRAGMA foreign_keys=OFF")
        # This consumer fixture certifies predictive runs. Keep unrelated manager,
        # acquisition and current-state rows out so the cache fixture and each test
        # certify the same synthetic predictive source.
        tables = (
            "schema_meta",
            "projection_runs",
            "fixtures",
            "player_fixture_xpts_projections",
            "monte_carlo_distributions",
        )
        try:
            for table in tables:
                columns = [
                    str(row[1])
                    for row in snapshot.execute(f'PRAGMA table_info("{table}")')
                ]
                if not columns:
                    continue
                rows = database.execute(f'SELECT * FROM "{table}"').fetchall()
                if not rows:
                    continue
                column_sql = ",".join(f'"{column}"' for column in columns)
                values_sql = ",".join("?" for _ in columns)
                snapshot.executemany(
                    f'INSERT OR REPLACE INTO "{table}" ({column_sql}) VALUES ({values_sql})',
                    [tuple(row) for row in rows],
                )
            snapshot.commit()
            snapshot.execute("VACUUM")
        finally:
            snapshot.close()
    return {
        "path": str(_SNAPSHOT_PATH),
        "sha256": es.file_sha256(_SNAPSHOT_PATH),
        "size_bytes": int(_SNAPSHOT_PATH.stat().st_size),
        "source_db_identity": es.source_db_identity(_SNAPSHOT_PATH),
        "execution_run_uuid": "00000000-0000-0000-0000-0000000000f1",
    }


def world_cache_generation_id(
    events: Sequence[int] = (5, 6, 7, 8), *, cutoff: str = CUTOFF
) -> str:
    """The generation id this fixture's synthetic world resolves to.

    Minted in a THROWAWAY in-memory store over exactly the same rows, so a cache
    written at import time (before any test database exists) is keyed on the SAME
    generation a test's own store will certify.  The certification is a pure function
    of the semantic evidence, which is what makes that possible.
    """

    from fpl_brain.database import connect_database

    conn = connect_database(":memory:")
    try:
        base_world(conn)
        events = tuple(int(event) for event in events)
        seed_certified_runs(conn, events)
        generation, _runs = seed_certified_world(conn, events, cutoff=cutoff)
        return generation.generation_id
    finally:
        conn.close()


def base_world(conn: Any) -> None:
    """The minimum world rows ``seed_certified_runs`` needs to be insertable."""

    with conn:
        for event in range(1, 39):
            conn.execute(
                "INSERT OR IGNORE INTO events(id, name, deadline_time, finished, raw_json, updated_at)"
                " VALUES (?,?,?,?, '{}', ?)",
                (event, f"GW{event}", f"2026-09-{event + 11:02d}T12:30:00Z", 1 if event < 5 else 0,
                 CUTOFF),
            )
        for team_id in (100, 101):
            conn.execute(
                "INSERT OR IGNORE INTO teams(id, name, short_name, raw_json, updated_at)"
                " VALUES (?,?,?, '{}', ?)",
                (team_id, f"Team {team_id}", f"T{team_id}", CUTOFF),
            )


def bundle(
    event: int, *, cutoff: str = CUTOFF, snapshot: str = DATA_SNAPSHOT,
    code: str = CODE_SNAPSHOT, context: str = CONTEXT_HASH,
    runs: Mapping[str, int] | None = None, models: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """One certified bundle in its persisted ``as_dict`` shape."""

    return {
        "event": int(event),
        "cutoff": cutoff,
        "runs": {k: int(v) for k, v in (runs if runs is not None else runs_for(int(event))).items()},
        "model_versions": dict(models or MODEL_VERSIONS),
        "code_snapshot_sha256": code,
        "data_snapshot_sha256": snapshot,
        "planning_context_hash": context,
    }


def certification_artifact(
    events: Sequence[int] = (5, 6, 7, 8),
    per_event: Mapping[int, Mapping[str, Any]] | None = None,
    **bundle_overrides: Any
) -> dict[str, Any]:
    """A complete, schema-valid v2 artifact, self-consistent by construction.

    ``per_event`` gives ONE event its own bundle dimensions, so a test can prove
    that a bundle is bound to its INTRINSIC event rather than to a container key.
    """

    per_event = per_event or {}
    bundles = {
        str(int(event)): bundle(int(event), **{**bundle_overrides, **per_event.get(int(event), {})})
        for event in events
    }
    declared = {event: cb.canonical_bundle_identity(row) for event, row in bundles.items()}
    first = bundles[str(int(events[0]))]
    payload = {
        "schema": fg.CERTIFICATION_ARTIFACT_SCHEMA,
        "execution_run_uuid": "fixture-run",
        "planning_cutoff": str(first["cutoff"]),
        "events": [int(event) for event in events],
        "data_snapshot_sha256": str(first["data_snapshot_sha256"]),
        "data_snapshot_created_at": CUTOFF,
        "data_snapshot_path": "data/snapshots/fixture.db",
        "data_snapshot_source_db_identity": "fixture",
        "code_snapshot_sha256": str(first["code_snapshot_sha256"]),
        "certification_wiring": fg.certification_wiring_identity(),
        "certified_bundles": bundles,
        "certified_bundle_identity": declared,
        # The certifier records the AUTHORITATIVE versions it required, from the one
        # declared source; the load boundary cross-checks them against that source.
        "required_model_versions": dict(cb.declared_required_versions()),
        "manager_state_identity": {"entry_id": 241392, "event": int(events[0])},
        "model_versions": [
            {"model_family": family, "model_version": version}
            for family, version in first["model_versions"].items()
        ],
        "dependency_validation": "COHERENT",
        "dependency_validation_detail": None,
        "temporal_status": "CAUSAL",
        "temporal_detail": {"rule": "fixture", "planning_cutoff": CUTOFF},
        "live_source_drift": "NONE",
        "history_completeness": {"schema": "fixture", "complete": True, "reasons": []},
        "route_search_executed": False,
        "transfer_execution_performed": False,
        "decision_search_permitted": True,
        "decision_search_permitted_reasons": [],
    }
    payload["four_gw_certification_identity"] = fg.certification_identity_of(payload)
    return payload


def write_artifact(tmp_path: Path, payload: Mapping[str, Any] | None = None, *, name="cert.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(dict(payload or certification_artifact())), encoding="utf-8")
    return path


def identity_for(bundle_row: Mapping[str, Any]) -> Any:
    """A world identity CONFORMING to a certified bundle, via the canonical labels."""

    from fpl_brain.chip_wildcard import WildcardPredictiveIdentity
    from fpl_brain.free_hit_decision_authority import model_label, runs_label

    return WildcardPredictiveIdentity(
        cutoff=str(bundle_row["cutoff"]),
        data_snapshot_sha256=str(bundle_row["data_snapshot_sha256"]),
        source_snapshot_sha256=str(bundle_row["code_snapshot_sha256"]),
        generation=runs_label(bundle_row["runs"]),
        model_config_identity=model_label(bundle_row["model_versions"]),
    )


def write_world_cache(
    cache_dir, *, generation_id: str, events: Sequence[int], union: Sequence[int],
    scores: Mapping[int, float] | None = None,
    worlds: int = 24, per_event_scores: Mapping[int, float] | None = None,
    bonus: Mapping[int, float] | None = None,
):
    """Populate the CANONICAL world cache so the loader can be exercised for real.

    Each matrix is written under the exact ``world_cache_key`` derived from the
    CERTIFIED GENERATION, its run ids, the config and the capture union -- the same
    key ``route_optimizer.build_event_worlds`` computes and stamps.  This is the only
    way to obtain a matrix the production adapter will accept.
    """

    import json
    from pathlib import Path as _Path

    from fpl_brain import route_optimizer as ro

    cache_dir = _Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    scores = scores or {}
    per_event_scores = per_event_scores or {}
    per_event_bonus = bonus or {}
    union = tuple(int(p) for p in union)
    keys = {}
    for event in events:
        event = int(event)
        from fpl_brain.free_hit_request_adapter import _CertifiedRunIds

        config = ro.OptimizerConfig(policy_selection_worlds=12)
        key = ro.world_cache_key(
            event=event, generation_id=str(generation_id), runs=runs_for(event),
            config=config, union_ids=union,
        )
        value = float(per_event_scores.get(event, scores.get("default", 1.0)))
        matrix = {
            "worlds": int(worlds),
            "player_ids": list(union),
            "core": {str(p): [value] * int(worlds) for p in union},
            "minutes": {str(p): [90.0] * int(worlds) for p in union},
            # The world-cache schema carries the per-player deterministic expected
            # bonus, because the armband objective consumes it.  A matrix without it
            # is a pre-v2 entry and is not a valid cache hit.
            "expected_bonus": {str(p): float(per_event_bonus.get(p, 0.0)) for p in union},
            "role_actionability": {str(p): False for p in union},
        }
        # PE-9 amendment 2 section 12: a cache HIT must be provable, so the entry
        # carries the content digest of the matrix it holds.  Without it the loader
        # (correctly) treats the file as a MISS and re-simulates the synthetic world.
        matrix[ro.CACHE_CONTENT_DIGEST_KEY] = ro.world_matrix_content_identity({
            "worlds": matrix["worlds"],
            "player_ids": [int(p) for p in matrix["player_ids"]],
            "core": {int(k): v for k, v in matrix["core"].items()},
            "minutes": {int(k): v for k, v in matrix["minutes"].items()},
            "expected_bonus": {int(k): v for k, v in matrix["expected_bonus"].items()},
            "role_actionability": {int(k): bool(v) for k, v in matrix["role_actionability"].items()},
        })
        (cache_dir / f"{key}.json").write_text(json.dumps(matrix), encoding="utf-8")
        keys[event] = key
    return keys


def identity_for_event(event: int, **overrides) -> Any:
    """A world identity CONFORMING to the certified bundle for ``event``."""

    return identity_for(bundle(int(event), **overrides))


def bundle_key(event: int, *, config, union, generation_id: str | None = None) -> str:
    """The canonical world-cache key for one event's certified GENERATION."""

    from fpl_brain import route_optimizer as ro

    return ro.world_cache_key(
        event=int(event), generation_id=str(generation_id or world_cache_generation_id()),
        runs=runs_for(int(event)), config=config,
        union_ids=tuple(int(p) for p in union),
    )
