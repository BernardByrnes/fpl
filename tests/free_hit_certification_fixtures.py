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

DATA_SNAPSHOT = "sha256:" + "d" * 64
CUTOFF = "2026-09-16T11:00:00Z"
CODE_SNAPSHOT = "sha256:" + "c" * 64
CONTEXT_HASH = "sha256:" + "p" * 64
RUNS: dict[str, int] = {"xpts": 101, "minutes": 102, "team": 103}
MODEL_VERSIONS: dict[str, str] = {"xpts": "xpts_v1.0.0", "minutes": "minutes_v1.6.0"}


def bundle(
    event: int, *, cutoff: str = CUTOFF, snapshot: str = DATA_SNAPSHOT,
    code: str = CODE_SNAPSHOT, context: str = CONTEXT_HASH,
    runs: Mapping[str, int] | None = None, models: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """One certified bundle in its persisted ``as_dict`` shape."""

    return {
        "event": int(event),
        "cutoff": cutoff,
        "runs": {k: int(v) for k, v in (runs or RUNS).items()},
        "model_versions": dict(models or MODEL_VERSIONS),
        "code_snapshot_sha256": code,
        "data_snapshot_sha256": snapshot,
        "planning_context_hash": context,
    }


def certification_artifact(
    events: Sequence[int] = (5, 6, 7, 8), **bundle_overrides: Any
) -> dict[str, Any]:
    """A complete, schema-valid v2 artifact, self-consistent by construction."""

    bundles = {str(int(event)): bundle(int(event), **bundle_overrides) for event in events}
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
