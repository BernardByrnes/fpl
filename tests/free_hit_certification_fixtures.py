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
CODE_SNAPSHOT = "sha256:" + "c" * 64
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
MODEL_VERSIONS: dict[str, str] = {
    "xpts_v1": "xpts_v1.0.0", "minutes_v1": "minutes_v1.6.0",
    "team_strength_v1": "team_v1.0.0", "player_rates_v1": "rates_v6.1.0",
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

    stamped = str(generated_at or CUTOFF)
    runs_by_event: dict[int, dict[str, int]] = {}
    for event in events:
        runs = runs_for(int(event))
        for family, run_id in runs.items():
            conn.execute(
                "INSERT INTO projection_runs(id, model_family, model_version, generated_at,"
                " planning_event, planning_context_hash, data_cutoff, status, source_snapshot_sha256)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    int(run_id), str(family), str(MODEL_VERSIONS[family]), stamped, int(event),
                    CONTEXT_HASH, CUTOFF, "complete", CODE_SNAPSHOT,
                ),
            )
        runs_by_event[int(event)] = {family: int(run_id) for family, run_id in runs.items()}
    return runs_by_event


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
    cache_dir, *, events: Sequence[int], union: Sequence[int], scores: Mapping[int, float] | None = None,
    worlds: int = 24, per_event_scores: Mapping[int, float] | None = None,
    bonus: Mapping[int, float] | None = None,
):
    """Populate the CANONICAL world cache so the loader can be exercised for real.

    Each matrix is written under the exact ``world_cache_key`` derived from the
    certified run ids, the config and the capture union -- the same key
    ``route_optimizer.build_event_worlds`` computes and stamps.  This is the only
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

        bundle = _CertifiedRunIds(event, runs_for(event))
        config = ro.OptimizerConfig(policy_selection_worlds=12)
        key = ro.world_cache_key(event=event, bundle=bundle, config=config, union_ids=union)
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
        (cache_dir / f"{key}.json").write_text(json.dumps(matrix), encoding="utf-8")
        keys[event] = key
    return keys


def identity_for_event(event: int, **overrides) -> Any:
    """A world identity CONFORMING to the certified bundle for ``event``."""

    return identity_for(bundle(int(event), **overrides))


def bundle_key(event: int, *, config, union) -> str:
    """The canonical world-cache key for one event's certified bundle."""

    from fpl_brain import route_optimizer as ro
    from fpl_brain.free_hit_request_adapter import _CertifiedRunIds

    return ro.world_cache_key(event=int(event), bundle=_CertifiedRunIds(int(event), runs_for(int(event))),
                              config=config, union_ids=tuple(int(p) for p in union))
