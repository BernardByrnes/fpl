"""PE-9 — the content-addressed certified generation store.

Authority amendment 2 replaced the isolated certified-service design with
**persisted, content-addressed evidence in the existing authoritative store**.  This
module is that store's contract.

The governing invariant:

    production decision logic may consume predictive data only through a persisted
    certified generation whose exact provenance can be independently re-derived and
    verified from authoritative persisted evidence.

A **generation** is an append-only row whose primary key IS the sha256 of its own
canonical semantic manifest.  The row is therefore self-verifying: recomputing the
digest from the persisted bytes must reproduce ``generation_id``.  A row may exist
ONLY when certification passed -- there is deliberately no mutable ``CERTIFIED``
flag to toggle -- so a crash before commit leaves no generation and the old pointer.

Three things are deliberately NOT here, because the threat model excludes them
(amendment 2 §2): no capability objects, no closure-captured secrets, no registry
that stands in for authority.  ``generation_id`` is a **selector**, not a
capability: an explicit historical ``generation_id`` selects that exact certified
generation, and the evidence that authorises the load is the persisted row plus the
pinned rows/snapshot it names.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from . import certified_bundle as cb

MANIFEST_SCHEMA = "fpl_brain.pe9_generation_manifest.v1"
LEGACY_DECISION_RECORD_SCHEMA = "fpl_brain.pe9_engine_decision_record.v1"
DECISION_RECORD_SCHEMA = "fpl_brain.pe9_engine_decision_record.v2"
DECISION_RECORD_SCHEMAS = frozenset({LEGACY_DECISION_RECORD_SCHEMA, DECISION_RECORD_SCHEMA})

#: The declared horizon kinds.  A generation is global predictive evidence for one
#: (planning_event, horizon_kind); manager-specific state never enters its identity.
HORIZON_KIND_FOUR_GW = "FOUR_GW"
HORIZON_KIND_MANAGER_WORLD = "MANAGER_WORLD"
HORIZON_KIND_WILDCARD_VALUE = "WILDCARD_VALUE"
HORIZON_KIND_CHIP_RESERVATION = "CHIP_RESERVATION"
HORIZON_KINDS: tuple[str, ...] = (
    HORIZON_KIND_FOUR_GW,
    HORIZON_KIND_MANAGER_WORLD,
    HORIZON_KIND_WILDCARD_VALUE,
    HORIZON_KIND_CHIP_RESERVATION,
)

#: The horizon LENGTH each kind is certified over.  The four-Gameweek normal-transfer
#: horizon is the frozen product rule; a manager-world generation covers exactly the
#: events it declares.  The length is part of the manifest, so the gate a generation
#: passed is reproducible rather than inferred from the caller's context.
HORIZON_LENGTHS: dict[str, int] = {HORIZON_KIND_FOUR_GW: 4}
WILDCARD_VALUE_MIN_EVENTS = 6
WILDCARD_VALUE_MAX_EVENTS = 10

# --- refusal tokens ---------------------------------------------------------
DIAG_GENERATION_UNKNOWN = "UNKNOWN_GENERATION_ID"
DIAG_GENERATION_MANIFEST_INVALID = "GENERATION_MANIFEST_INVALID"
DIAG_GENERATION_DIGEST_MISMATCH = "GENERATION_MANIFEST_MUTATED"
DIAG_GENERATION_NOT_CERTIFIED = "GENERATION_NOT_CERTIFIED"
DIAG_GENERATION_SNAPSHOT_UNVERIFIED = "GENERATION_SNAPSHOT_UNVERIFIED"
DIAG_GENERATION_POINTER_UNSET = "GENERATION_POINTER_UNSET"
DIAG_GENERATION_HORIZON_KIND_UNKNOWN = "GENERATION_HORIZON_KIND_UNKNOWN"
DIAG_DECISION_RECORD_UNKNOWN = "UNKNOWN_DECISION_ID"
DIAG_DECISION_RECORD_INVALID = "DECISION_RECORD_INVALID"
DIAG_DECISION_ARTIFACT_NOT_RETAINED = "DECISION_ARTIFACT_NOT_RETAINED"
DIAG_MANAGER_STATE_MISMATCH = "MANAGER_STATE_MISMATCH"
DIAG_MANAGER_STATE_EVIDENCE_MISSING = "MANAGER_STATE_EVIDENCE_MISSING"
DIAG_PRODUCTION_DESCRIPTOR_ONLY = "PRODUCTION_DESCRIPTOR_ONLY"
DIAG_PRODUCTION_PROFILE_UNKNOWN = "PRODUCTION_DECISION_PROFILE_UNKNOWN"
DIAG_DECISION_REPLAY_ONLY = "DECISION_REPLAY_ONLY"
DIAG_GENERATION_PLANNING_CONTEXT_REQUIRED = "GENERATION_PLANNING_CONTEXT_REQUIRED"
DIAG_PRODUCTION_SEARCH_PERMISSION_DENIED = "PRODUCTION_SEARCH_PERMISSION_DENIED"

#: How far the PE-8 evidence reference was actually REPRODUCED by the verifier.
#: ``NOT_CONSULTED`` is not a failure: no calibration claim was made.  There is no
#: third outcome: a reference that does not reproduce is a REFUSAL, so the verifier
#: never reports a reference as verified without recomputing it.
PE8_NOT_CONSULTED = "NOT_CONSULTED"
PE8_IDENTITY_AND_STATE_RECOMPUTED = "IDENTITY_AND_STATE_RECOMPUTED_FROM_MANIFEST"

#: A production caller must not supply any of these.  They are named so the refusal
#: says exactly which descriptor the caller tried to smuggle into a production call.
FORBIDDEN_PRODUCTION_DESCRIPTORS: tuple[str, ...] = (
    "matrix",
    "matrices",
    "bundles",
    "bundle",
    "worlds",
    "world_matrix",
    "prebuilt_worlds",
    "runs",
    "runs_by_event",
    "run_ids",
    "discovery_runs",
    "exact_runs",
    "certification",
    "certification_artifact",
    "artifact",
    "validation",
    "authorisation",
    "authorization",
    "manifest",
    "manifest_digest",
    "cache",
    "cache_dir",
    "cache_handle",
    "registry",
    "capability",
    "token",
    # The executor door itself: a callable must not be able to arrive as a descriptor.
    "decide",
    "executor",
    "decider",
    "callback",
    "callable",
    "function",
    "func",
    "pipeline",
    "entrypoint",
    "runner",
    "predictive_payload",
    "predictive_data",
    "dependency_mapping",
    "dependency_map",
    "dependencies",
    "run_mapping",
    "run_map",
    "source_conn",
    "source_connection",
    "snapshot",
    "snapshot_path",
    "data_snapshot",
    "data_snapshot_path",
    "artifact_path",
    "decision_artifact_path",
    "artifact_ref",
    "output_path",
    "cache_object",
    "cache_path",
)

_FOUR_GW_DECISION_PARAMETERS = frozenset(
    {
        "beam", "exact_evaluation_budget", "search_n_per_criterion", "singles_per_out",
        "max_transfers_per_event", "stage2_draws", "parallel_workers",
    }
)
_MANAGER_WORLD_DECISION_PARAMETERS = frozenset(
    {"simulations", "seed", "occupancy_audit", "top_k"}
)

#: Float values are refused in identity-bearing content: a float has no single
#: stable textual form, so a manifest carrying one could not be re-derived to the
#: same digest by an independent verifier.  Only stable strings, integers, booleans,
#: nulls and the fixed containers are accepted.
_IDENTITY_SCALARS = (str, int, bool, type(None))


class GenerationRefused(RuntimeError):
    """A PE-9 generation-store gate refused an operation.

    Every refusal carries a specific token, so "unknown", "mutated", "invalid" and
    "not certified" are never confused with one another.
    """

    def __init__(self, token: str, reasons: Sequence[str]) -> None:
        self.token = str(token)
        self.reasons = [str(reason) for reason in reasons]
        super().__init__(f"{self.token}: " + "; ".join(self.reasons))


class GenerationUnknown(GenerationRefused):
    def __init__(self, reasons: Sequence[str]) -> None:
        super().__init__(DIAG_GENERATION_UNKNOWN, reasons)


class GenerationManifestInvalid(GenerationRefused):
    def __init__(self, reasons: Sequence[str]) -> None:
        super().__init__(DIAG_GENERATION_MANIFEST_INVALID, reasons)


class GenerationManifestMutated(GenerationRefused):
    def __init__(self, reasons: Sequence[str]) -> None:
        super().__init__(DIAG_GENERATION_DIGEST_MISMATCH, reasons)


class GenerationNotCertified(GenerationRefused):
    def __init__(self, reasons: Sequence[str]) -> None:
        super().__init__(DIAG_GENERATION_NOT_CERTIFIED, reasons)


class GenerationSnapshotUnverified(GenerationRefused):
    def __init__(self, reasons: Sequence[str]) -> None:
        super().__init__(DIAG_GENERATION_SNAPSHOT_UNVERIFIED, reasons)


class DecisionRecordInvalid(GenerationRefused):
    def __init__(self, reasons: Sequence[str]) -> None:
        super().__init__(DIAG_DECISION_RECORD_INVALID, reasons)


class ProductionDescriptorOnly(GenerationRefused):
    """A production call carried a descriptor the production API must not accept."""

    def __init__(self, reasons: Sequence[str]) -> None:
        super().__init__(DIAG_PRODUCTION_DESCRIPTOR_ONLY, reasons)


def _wildcard_value_horizon_problems(
    *, planning_event: int, events: Sequence[int], horizon_length: int, last_event: int
) -> list[str]:
    """Validate Wildcard's separate 6–10 event certification contract.

    This product is intentionally distinct from ``FOUR_GW``. Its events must
    be one complete, contiguous sequence beginning at the planning event, and
    every declared event must exist in the pinned season snapshot.
    """

    resolved = tuple(int(event) for event in events)
    problems: list[str] = []
    if not (WILDCARD_VALUE_MIN_EVENTS <= len(resolved) <= WILDCARD_VALUE_MAX_EVENTS):
        problems.append(
            f"Wildcard value horizon needs {WILDCARD_VALUE_MIN_EVENTS}-{WILDCARD_VALUE_MAX_EVENTS} "
            f"events, got {len(resolved)}"
        )
    if int(horizon_length) != len(resolved):
        problems.append(
            f"Wildcard value horizon length {int(horizon_length)} does not match its "
            f"{len(resolved)} declared events"
        )
    if not resolved or resolved[0] != int(planning_event):
        problems.append("Wildcard value events must begin at the planning event")
    if len(set(resolved)) != len(resolved):
        problems.append("Wildcard value events contain duplicates")
    expected = tuple(range(int(planning_event), int(planning_event) + len(resolved)))
    if resolved != expected:
        problems.append(f"Wildcard value events {list(resolved)} are not contiguous from GW{planning_event}")
    if resolved and resolved[-1] > int(last_event):
        problems.append(
            f"Wildcard value horizon ends at GW{resolved[-1]}, beyond pinned season end GW{last_event}"
        )
    return problems


def _chip_reservation_horizon_problems(
    *, planning_event: int, events: Sequence[int], horizon_length: int, last_event: int,
) -> list[str]:
    """Validate the separate origin-pinned product used for chip forecasts.

    This product can extend beyond the normal four-event decision horizon. It
    is an input product only: it never widens a normal chip decision or creates
    a production decision profile. Wildcard consumers still bind a separate
    6-10 event value window within this product.
    """

    resolved = tuple(int(event) for event in events)
    problems: list[str] = []
    if len(resolved) < 4:
        problems.append("chip-reservation product must retain the exact four-event normal prefix")
    if int(horizon_length) != len(resolved):
        problems.append(
            f"chip-reservation horizon length {int(horizon_length)} does not match its {len(resolved)} events"
        )
    if not resolved or resolved[0] != int(planning_event):
        problems.append("chip-reservation events must begin at the origin planning event")
    if len(set(resolved)) != len(resolved):
        problems.append("chip-reservation events contain duplicates")
    expected = tuple(range(int(planning_event), int(planning_event) + len(resolved)))
    if resolved != expected:
        problems.append("chip-reservation events must be contiguous from the origin planning event")
    if resolved and resolved[-1] > int(last_event):
        problems.append(
            f"chip-reservation product ends at GW{resolved[-1]}, beyond pinned season end GW{last_event}"
        )
    return problems


# ---------------------------------------------------------------------------
# Canonical semantic manifest
# ---------------------------------------------------------------------------


def canonical_identity_value(value: Any, *, path: str = "manifest") -> Any:
    """The canonical, stable form of one identity-bearing value.

    ``None``, booleans, integers and strings pass through; mappings are keyed by
    string; sequences become lists.  A FLOAT is refused outright rather than
    stringified, because no single textual form of a float is guaranteed across
    implementations.  A set is refused too: it has no canonical order.
    """

    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        raise GenerationManifestInvalid(
            [
                f"{path}: a floating-point value ({value!r}) is not representable in identity-bearing "
                "content; a generation manifest carries stable strings, integers and fixed forms only"
            ]
        )
    if isinstance(value, Mapping):
        return {
            str(key): canonical_identity_value(item, path=f"{path}.{key}")
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [
            canonical_identity_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise GenerationManifestInvalid(
        [f"{path}: {type(value).__name__} is not representable in identity-bearing content"]
    )


def canonical_manifest_bytes(manifest: Mapping[str, Any]) -> bytes:
    """The ONE canonical byte form of a generation manifest.

    Producer and verifier both encode through here, so "these bytes digest to that
    identity" is a pure function of the manifest's content.
    """

    canonical = canonical_identity_value(dict(manifest))
    return json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def generation_id_of(manifest: Mapping[str, Any]) -> str:
    """``generation_id = sha256(canonical_semantic_manifest)``.

    Hashing the SEMANTIC PROJECTION makes the digest idempotent: a manifest read back
    from the store -- which necessarily carries row metadata such as ``created_at`` --
    resolves to the same identity as the manifest that was written.
    """

    return "sha256:" + hashlib.sha256(
        canonical_manifest_bytes(manifest_semantic_projection(manifest))
    ).hexdigest()


def manifest_semantic_projection(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """The IDENTITY-BEARING projection of a manifest.

    Everything the generation claims about the predictive world: the event, the
    horizon kind, the exact runs, the recorded versions, the planning context, the
    data-snapshot identity and the PE-8 evidence references.  Volatile fields --
    creation timestamps, lock instants, capture durations -- are deliberately
    absent, so re-certifying identical semantic evidence resolves to the SAME
    ``generation_id``.
    """

    manifest = dict(manifest)

    def _int_or_none(value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    projection = {
        "schema": manifest.get("schema"),
        "planning_event": _int_or_none(manifest.get("planning_event")),
        "horizon_kind": (
            None if manifest.get("horizon_kind") is None else str(manifest.get("horizon_kind"))
        ),
        "events": [int(event) for event in (manifest.get("events") or [])],
        "horizon_length": int(manifest.get("horizon_length") or 0),
        "cutoff": None if manifest.get("cutoff") is None else str(manifest.get("cutoff")),
        "required_model_versions": {
            str(k): str(v) for k, v in (manifest.get("required_model_versions") or {}).items()
        },
        "code_snapshot_sha256": manifest.get("code_snapshot_sha256"),
        "data_snapshot": {
            "sha256": (manifest.get("data_snapshot") or {}).get("sha256"),
            "size_bytes": (manifest.get("data_snapshot") or {}).get("size_bytes"),
            "source_db_identity": (manifest.get("data_snapshot") or {}).get("source_db_identity"),
            "execution_run_uuid": (manifest.get("data_snapshot") or {}).get("execution_run_uuid"),
        },
        "per_event": {
            str(event): dict(record)
            for event, record in sorted(
                (manifest.get("per_event") or {}).items(), key=lambda item: str(item[0])
            )
        },
        "horizon_state": manifest.get("horizon_state"),
        "last_event": manifest.get("last_event"),
        "pe8_evidence": manifest.get("pe8_evidence"),
        "disclosure": manifest.get("disclosure"),
    }
    # Legacy generation rows predate retained temporal evidence. Keep their
    # identity projection byte-for-byte compatible; new rows include the evidence
    # block (including an explicit UNRESOLVED result when it cannot be derived).
    if "search_permission_causality" in manifest:
        projection["search_permission_causality"] = manifest.get("search_permission_causality")
    return projection


@dataclass(frozen=True)
class CertifiedGeneration:
    """One persisted certified generation, loaded and digest-verified."""

    generation_id: str
    manifest: Mapping[str, Any]
    planning_event: int
    horizon_kind: str
    cutoff: str
    created_at: str

    @property
    def events(self) -> tuple[int, ...]:
        return tuple(int(event) for event in (self.manifest.get("events") or ()))

    @property
    def runs_by_event(self) -> dict[int, dict[str, int]]:
        return {
            int(event): {str(family): int(run) for family, run in (record.get("runs") or {}).items()}
            for event, record in (self.manifest.get("per_event") or {}).items()
        }

    @property
    def model_versions_by_event(self) -> dict[int, dict[str, str]]:
        return {
            int(event): {
                str(family): str(version)
                for family, version in (record.get("model_versions") or {}).items()
            }
            for event, record in (self.manifest.get("per_event") or {}).items()
        }

    @property
    def snapshot(self) -> Mapping[str, Any]:
        return self.manifest.get("data_snapshot") or {}

    def runs_for(self, event: int) -> dict[str, int]:
        return dict(self.runs_by_event.get(int(event)) or {})

    def as_dict(self) -> dict[str, Any]:
        return {
            "generation_id": self.generation_id,
            "planning_event": int(self.planning_event),
            "horizon_kind": self.horizon_kind,
            "cutoff": self.cutoff,
            "events": list(self.events),
            "created_at": self.created_at,
            "manifest": dict(self.manifest),
        }


# ---------------------------------------------------------------------------
# Manifest construction
# ---------------------------------------------------------------------------


def build_generation_manifest(
    *,
    planning_event: int,
    horizon_kind: str,
    cutoff: str,
    events: Sequence[int],
    per_event: Mapping[Any, Mapping[str, Any]],
    required_model_versions: Mapping[str, str],
    code_snapshot_sha256: str | None,
    snapshot: Mapping[str, Any],
    horizon_state: str,
    horizon_length: int,
    last_event: int,
    pe8_evidence: Mapping[str, Any],
    disclosure: Mapping[str, Any],
    search_permission_causality: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the canonical semantic manifest of ONE generation.

    The caller supplies only already-VALIDATED per-event records and already-READ
    identities: this function decides what is identity-bearing, not what is true.
    """

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "planning_event": int(planning_event),
        "horizon_kind": str(horizon_kind),
        "events": [int(event) for event in events],
        "horizon_length": int(horizon_length),
        "cutoff": str(cutoff),
        "required_model_versions": {
            str(family): str(version)
            for family, version in sorted(required_model_versions.items())
        },
        "code_snapshot_sha256": code_snapshot_sha256,
        "data_snapshot": {
            "path": snapshot.get("path"),
            "sha256": snapshot.get("sha256"),
            "size_bytes": snapshot.get("size_bytes"),
            "source_db_identity": snapshot.get("source_db_identity"),
            "execution_run_uuid": snapshot.get("execution_run_uuid"),
            **({"manifest_path": snapshot.get("manifest_path")}
               if snapshot.get("manifest_path") else {}),
        },
        "per_event": {
            str(int(event)): {
                "event": int(event),
                "bundle_identity": str(record.get("bundle_identity") or ""),
                "runs": {
                    str(family): int(run_id)
                    for family, run_id in sorted((record.get("runs") or {}).items())
                },
                "model_versions": {
                    str(family): str(version)
                    for family, version in sorted((record.get("model_versions") or {}).items())
                },
                "planning_context_hash": record.get("planning_context_hash"),
                "state": str(record.get("state") or ""),
                "structural_state": str(record.get("structural_state") or ""),
                "evidence_state": str(record.get("evidence_state") or ""),
                "zero_fixture_event": bool(record.get("zero_fixture_event")),
                "dependency_closure": {
                    str(family): {
                        str(dep): int(run_id)
                        for dep, run_id in sorted((deps or {}).items())
                    }
                    for family, deps in sorted((record.get("dependency_closure") or {}).items())
                },
                "reasons": [str(reason) for reason in (record.get("reasons") or [])],
            }
            for event, record in sorted(per_event.items(), key=lambda item: int(item[0]))
        },
        "horizon_state": str(horizon_state),
        "last_event": int(last_event),
        "pe8_evidence": canonical_identity_value(dict(pe8_evidence)),
        "disclosure": canonical_identity_value(dict(disclosure)),
        "search_permission_causality": (
            canonical_identity_value(dict(search_permission_causality))
            if isinstance(search_permission_causality, Mapping)
            else {
                "schema": "fpl_brain.generation_causality.v1",
                "temporal_status": "UNRESOLVED",
                "reasons": ["UNRESOLVED: causal timestamp evidence was not available at certification"],
            }
        ),
    }
    # Fails closed on any float the projection above let through.
    canonical_manifest_bytes(manifest)
    return manifest


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _row_to_generation(row: sqlite3.Row) -> CertifiedGeneration:
    manifest = json.loads(str(row["manifest_json"]))
    return CertifiedGeneration(
        generation_id=str(row["generation_id"]),
        manifest=manifest,
        planning_event=int(row["planning_event"]),
        horizon_kind=str(row["horizon_kind"]),
        cutoff=str(row["cutoff"]),
        created_at=str(row["created_at"]),
    )


def load_generation(conn: sqlite3.Connection, generation_id: str) -> CertifiedGeneration:
    """Load ONE generation and require its bytes to reproduce its own identity.

    ``UNKNOWN_GENERATION_ID`` and ``GENERATION_MANIFEST_MUTATED`` are deliberately
    different facts: an id nobody wrote, versus a row whose persisted bytes no
    longer digest to the key they were stored under.
    """

    wanted = str(generation_id or "").strip()
    if not wanted:
        raise GenerationUnknown(["no generation id was supplied"])
    row = conn.execute(
        "SELECT * FROM generation WHERE generation_id=?", (wanted,)
    ).fetchone()
    if row is None:
        raise GenerationUnknown([f"no generation {wanted!r} exists in the authoritative store"])
    manifest = json.loads(str(row["manifest_json"]))
    recorded = str(row["manifest_sha256"])
    recomputed = generation_id_of(manifest)
    if recomputed != wanted or recorded != wanted:
        raise GenerationManifestMutated(
            [
                f"generation {wanted} carries bytes that digest to {recomputed} "
                f"(recorded {recorded}); the manifest was changed after its id was calculated"
            ]
        )
    snapshot = manifest.get("data_snapshot") or {}
    try:
        row_identity = json.loads(str(row["snapshot_source_db_identity"] or "{}"))
    except (TypeError, json.JSONDecodeError):
        row_identity = None
    row_metadata = {
        "planning_event": int(row["planning_event"]),
        "horizon_kind": str(row["horizon_kind"]),
        "cutoff": str(row["cutoff"]),
        "snapshot_path": row["snapshot_path"],
        "snapshot_sha256": row["snapshot_sha256"],
        "snapshot_source_db_identity": row_identity,
    }
    expected_metadata = {
        "planning_event": int(manifest.get("planning_event")),
        "horizon_kind": str(manifest.get("horizon_kind")),
        "cutoff": str(manifest.get("cutoff")),
        "snapshot_path": snapshot.get("path"),
        "snapshot_sha256": snapshot.get("sha256"),
        "snapshot_source_db_identity": snapshot.get("source_db_identity"),
    }
    if row_metadata != expected_metadata:
        raise GenerationRefused(
            DIAG_GENERATION_MANIFEST_INVALID,
            [
                "the generation row metadata does not match its digest-verified manifest: "
                f"row={row_metadata!r}, manifest={expected_metadata!r}"
            ],
        )
    return _row_to_generation(row)


def current_generation_id(
    conn: sqlite3.Connection, planning_event: int, horizon_kind: str = HORIZON_KIND_FOUR_GW
) -> str | None:
    """The ``current_generation`` pointer's value, or ``None`` when it is unset."""

    row = conn.execute(
        "SELECT generation_id FROM current_generation WHERE planning_event=? AND horizon_kind=?",
        (int(planning_event), str(horizon_kind)),
    ).fetchone()
    return None if row is None else str(row["generation_id"])


def resolve_generation(
    conn: sqlite3.Connection,
    *,
    planning_event: int,
    horizon_kind: str = HORIZON_KIND_FOUR_GW,
    generation_id: str | None = None,
) -> CertifiedGeneration:
    """Resolve a generation by explicit SELECTOR or by the current pointer.

    ``generation_id=None`` resolves the ``current_generation`` pointer; an explicit
    id loads that exact certified historical generation and is unaffected by a
    concurrent pointer change.  An id that names a different event or horizon kind
    is a contradiction, not a selector: a generation is bound to its own event.

    NOTE: the pointer is a convenience SELECTOR, never evidence.  What authorises
    the load that follows is the persisted manifest, its re-verified digest, and the
    pinned runs/snapshot it names.
    """

    if horizon_kind not in HORIZON_KINDS:
        raise GenerationRefused(
            DIAG_GENERATION_HORIZON_KIND_UNKNOWN,
            [f"{horizon_kind!r} is not a declared horizon kind ({list(HORIZON_KINDS)})"],
        )
    if generation_id is None:
        pointer = current_generation_id(conn, int(planning_event), horizon_kind)
        if pointer is None:
            raise GenerationRefused(
                DIAG_GENERATION_POINTER_UNSET,
                [
                    f"no current certified generation is selected for GW{int(planning_event)} "
                    f"({horizon_kind}); a decision without a certified generation is not taken"
                ],
            )
        generation = load_generation(conn, pointer)
    else:
        generation = load_generation(conn, str(generation_id))
    if int(generation.planning_event) != int(planning_event):
        raise GenerationRefused(
            cb.STATE_PREDICTIVE_BUNDLE_INCOHERENT,
            [
                f"generation {generation.generation_id} was certified for GW{generation.planning_event}, "
                f"not the requested GW{int(planning_event)}"
            ],
        )
    if str(generation.horizon_kind) != str(horizon_kind):
        raise GenerationRefused(
            cb.STATE_PREDICTIVE_BUNDLE_INCOHERENT,
            [
                f"generation {generation.generation_id} is a {generation.horizon_kind} generation, not "
                f"{horizon_kind}"
            ],
        )
    return generation


def _declared_families(runs: Mapping[str, Any], record: Mapping[str, Any]) -> tuple[str, ...]:
    """The families one event's certification must cover.

    The families a world load READS (:data:`certified_bundle.LOAD_REQUIRED_FAMILIES`)
    plus every family this event's manifest actually declares.  A family that is
    declared is validated in full -- including its dependency edges -- while a family
    the world legitimately does not carry is a DECLARED absence rather than an
    invented requirement.
    """

    declared = set(str(family) for family in (record.get("model_versions") or {}))
    return tuple(sorted(set(cb.LOAD_REQUIRED_FAMILIES) | declared | set(runs)))


def _dependency_closure(
    conn: sqlite3.Connection, event: int, runs: Mapping[str, int]
) -> dict[str, dict[str, int]]:
    """The upstream run ids each family's rows record, for the manifest.

    Read from the ROWS, so a verifier can reproduce the closure rather than trust
    the record of it.
    """

    closure: dict[str, dict[str, int]] = {}
    for family in ("xpts_v1", "monte_carlo_v1"):
        run_id = runs.get(family)
        if run_id is None:
            continue
        upstream = cb._upstream_run_ids(conn, family, int(run_id))
        if upstream:
            closure[family] = {
                str(dep): int(value) for dep, value in sorted(upstream.items()) if value is not None
            }
    return closure


def authoritative_code_identity() -> str:
    """The ONE declared source of the code identity a certification requires.

    ``projection_runs.source_snapshot_sha256`` records which code revision produced a
    predictive run.  ``declared_required_versions`` is the declared source for the
    VERSIONS a certification requires; this is the same rule for the CODE: the
    expected identity is computed from the in-library source set, never accepted from
    a caller and never merely recorded, so a generation can only be certified over
    runs that the running code revision actually produced.
    """

    from . import analytics

    return str(analytics.source_snapshot_sha256())


def certify_generation(
    conn: sqlite3.Connection,
    *,
    planning_event: int,
    cutoff: str,
    runs_by_event: Mapping[int, Mapping[str, int]],
    snapshot: Mapping[str, Any] | None,
    horizon_kind: str = HORIZON_KIND_FOUR_GW,
    events: Sequence[int] | None = None,
    horizon_length: int | None = None,
    calibration: Mapping[str, Any] | None = None,
    calibration_artifact_ref: str | Path | None = None,
    require_calibration: bool = False,
    last_event: int | None = None,
    clock: Callable[[], str] | None = None,
    controller: Any = None,
) -> CertifiedGeneration:
    """Certify a horizon and persist ONE generation, atomically.

    The lifecycle (amendment 2 §8):

        acquire the existing writer lease  (``controller``, when supplied)
        -> validate each bundle with the existing canonical validators
        -> validate required model versions (authoritative in-library source only)
        -> validate the authoritative CODE identity against the run rows
        -> validate dependency closure, planning_context_hash and the cutoff
        -> validate the pinned data snapshot identity (hashed, not claimed)
        -> consult/link the applicable PE-8 evidence
        -> apply the four-event horizon gate
        -> construct the canonical semantic manifest and compute generation_id
        -> ONE TRANSACTION: insert generation idempotently + update current_generation

    Three things are deliberately NOT parameters, because a caller must not be able
    to pin them: ``required_versions`` comes from
    :func:`certified_bundle.declared_required_versions`, the CODE identity comes from
    :func:`authoritative_code_identity`, and the data snapshot is REQUIRED and its
    bytes are hashed here rather than accepted as a claim.

    A generation row exists only when certification passed.  The validation and the
    generation/pointer write run inside ONE ``BEGIN IMMEDIATE`` transaction, so a
    concurrent writer cannot interleave between the check and the persist; when an
    ``ExecutionController`` is supplied its writer lease must be ACTIVE (the caller
    drives this inside its own ``CERTIFY_GENERATION`` stage, so the run ledger names
    the certification that produced the row).  A crash before commit leaves no
    generation and the old pointer, and re-certifying identical semantic evidence
    resolves to the SAME ``generation_id``.
    """

    from . import four_gw_decision as fg

    if horizon_kind not in HORIZON_KINDS:
        raise GenerationRefused(
            DIAG_GENERATION_HORIZON_KIND_UNKNOWN,
            [f"{horizon_kind!r} is not a declared horizon kind ({list(HORIZON_KINDS)})"],
        )
    required_versions = cb.declared_required_versions()
    code_identity = authoritative_code_identity()
    snapshot_block = _snapshot_identity_block(snapshot)
    search_permission_causality = _generation_causality_for_manifest(
        conn,
        snapshot=snapshot_block,
        generation_planning_event=int(planning_event),
        generation_cutoff=str(cutoff),
    )
    calibration_digest, calibration_file_sha256 = _calibration_file_identity(
        calibration, calibration_artifact_ref, required=bool(require_calibration)
    )
    resolved_events = (
        [int(event) for event in events]
        if events is not None
        else sorted(int(event) for event in runs_by_event)
    )
    lease_run_uuid = assert_writer_lease_held(conn, controller)

    snapshot_conn = sqlite3.connect(
        f"file:{Path(snapshot_block['path'])}?mode=ro", uri=True
    )
    try:
        from . import four_gw_decision as fg

        snapshot_conn.row_factory = sqlite3.Row
        snapshot_last_event = int(fg.season_last_event_from_db(snapshot_conn))
    finally:
        snapshot_conn.close()
    if last_event is not None and int(last_event) != snapshot_last_event:
        raise GenerationRefused(
            DIAG_GENERATION_NOT_CERTIFIED,
            [
                f"the supplied season last event {int(last_event)} does not match the pinned "
                f"snapshot's last event {snapshot_last_event}"
            ],
        )
    resolved_last_event = snapshot_last_event
    resolved_length = int(
        horizon_length
        if horizon_length is not None
        else HORIZON_LENGTHS.get(str(horizon_kind), len(resolved_events))
    )
    if horizon_kind == HORIZON_KIND_WILDCARD_VALUE:
        problems = _wildcard_value_horizon_problems(
            planning_event=int(planning_event),
            events=resolved_events,
            horizon_length=resolved_length,
            last_event=resolved_last_event,
        )
        if problems:
            raise GenerationRefused(DIAG_GENERATION_NOT_CERTIFIED, problems)
    elif horizon_kind == HORIZON_KIND_CHIP_RESERVATION:
        problems = _chip_reservation_horizon_problems(
            planning_event=int(planning_event),
            events=resolved_events,
            horizon_length=resolved_length,
            last_event=resolved_last_event,
        )
        if problems:
            raise GenerationRefused(DIAG_GENERATION_NOT_CERTIFIED, problems)

    from .database import write_transaction

    # The validation and the persist are ONE write transaction: nothing can certify
    # between the checks below and the row that records their outcome.
    with write_transaction(conn):
        current_lease_run_uuid = assert_writer_lease_held(conn, controller)
        if current_lease_run_uuid != lease_run_uuid:
            raise GenerationRefused(
                DIAG_GENERATION_NOT_CERTIFIED,
                ["the active writer lease changed before generation publication"],
            )
        per_event: dict[int, dict[str, Any]] = {}
        support: dict[int, dict[str, Any]] = {}
        resolved_runs: dict[int, dict[str, int]] = {}
        base_bundle_identities: dict[str, str] = {}

        # PE-8 evidence is bound to the same certification identity produced by the
        # existing bundle identities.  Resolve those exact identities first; they
        # are revalidated below together with the evidence under this transaction.
        if calibration is not None:
            for event in resolved_events:
                runs = {
                    str(family): int(run)
                    for family, run in (runs_by_event.get(int(event)) or {}).items()
                }
                base = cb.certified_bundle_from_explicit_ids(
                    conn,
                    event=int(event),
                    cutoff=str(cutoff),
                    runs=runs,
                    required_versions=required_versions,
                    data_snapshot_sha256=snapshot_block["sha256"],
                    code_snapshot_sha256=code_identity,
                    expected_data_snapshot_sha256=snapshot_block["sha256"],
                    families=_declared_families(runs, {}),
                )
                resolved_runs[int(event)] = runs
                base_bundle_identities[str(int(event))] = base.bundle_identity()
            pe8_certification_identity = _certification_identity_for_bundles(
                cutoff=str(cutoff),
                bundle_identities=base_bundle_identities,
                snapshot_sha256=str(snapshot_block["sha256"]),
            )
        else:
            pe8_certification_identity = None

        for event in resolved_events:
            runs = resolved_runs.get(int(event)) or {
                str(family): int(run) for family, run in (runs_by_event.get(int(event)) or {}).items()
            }
            # The families this certification must cover: those a world load reads,
            # plus every family this event's world actually declares.  A world that
            # carries no Monte Carlo run declares none, so its absence is a DECLARED
            # absence rather than an invented requirement -- while a family that IS
            # declared is fully validated, dependency edges included.
            families = _declared_families(runs, {})
            certified = cb.certify_event_bundle(
                conn,
                event=int(event),
                cutoff=str(cutoff),
                runs=runs,
                required_versions=required_versions,
                data_snapshot_sha256=snapshot_block["sha256"],
                # The AUTHORITATIVE code identity: the runs must have been produced by
                # the running revision, so a certification cannot pin the code of a
                # world the current code did not build.
                code_snapshot_sha256=code_identity,
                expected_data_snapshot_sha256=snapshot_block["sha256"],
                calibration=calibration,
                require_calibration=bool(require_calibration),
                certification_identity=pe8_certification_identity,
                families=families,
            )
            if certified.structural_state not in cb.BLOCKING_BUNDLE_STATES:
                identity_failures = _run_identity_failures(
                    conn,
                    event=int(event),
                    runs=certified.runs,
                    code_identity=code_identity,
                    snapshot=snapshot_block,
                )
                if identity_failures:
                    context_failure = any("planning_context_hash" in reason for reason in identity_failures)
                    snapshot_failure = any(
                        "data snapshot" in reason or "execution UUID" in reason
                        for reason in identity_failures
                    )
                    raise GenerationRefused(
                        DIAG_GENERATION_PLANNING_CONTEXT_REQUIRED
                        if context_failure and not snapshot_failure
                        else cb.STATE_EVIDENCE_MISSING,
                        identity_failures,
                    )
            record = certified.as_dict()
            record["runs"] = {str(family): int(run) for family, run in certified.runs.items()}
            record["dependency_closure"] = _dependency_closure(conn, int(event), certified.runs)
            per_event[int(event)] = record
            support[int(event)] = {
                "supported": certified.state not in cb.BLOCKING_BUNDLE_STATES,
                "data_cutoff": str(cutoff) if certified.state not in cb.BLOCKING_BUNDLE_STATES else None,
                "missing_families": (
                    [] if certified.state not in cb.BLOCKING_BUNDLE_STATES else ["CERTIFICATION_BLOCKED"]
                ),
                "state": certified.state,
                "bundle_identity": certified.bundle_identity,
                "matched_runs": {str(k): int(v) for k, v in certified.runs.items()},
            }

        # The planning context is part of the predictive world: every family of an
        # event must have been predicted under ONE declared context, and it must be
        # DECLARED rather than absent.  A world whose context nobody recorded cannot
        # be re-derived to a planning state, so it is refused here rather than
        # recorded as a hole.
        contextless = [
            f"GW{event}: no planning_context_hash is recorded by the certified families"
            for event in resolved_events
            if str(per_event[int(event)].get("structural_state") or "") not in cb.BLOCKING_BUNDLE_STATES
            and not str(per_event[int(event)].get("planning_context_hash") or "").strip()
        ]
        if contextless:
            raise GenerationRefused(DIAG_GENERATION_PLANNING_CONTEXT_REQUIRED, contextless)

        horizon = fg.evaluate_horizon(
            planning_event=int(resolved_events[0]) if resolved_events else int(planning_event),
            support_by_event=support,
            cutoff=str(cutoff),
            length=resolved_length,
            last_event=resolved_last_event,
        )
        blocking_events = [int(event) for event in horizon.get("blocked_events") or []]
        if blocking_events or str(horizon["status"]) != fg.DECISION_HORIZON_COMPLETE:
            raise GenerationNotCertified(
                [
                    f"the horizon cannot be certified: {horizon['status']} (blocked events "
                    f"{blocking_events if blocking_events else 'none'})"
                ]
                + [
                    f"GW{int(event)}: "
                    + "; ".join(str(reason) for reason in per_event[int(event)]["reasons"])
                    for event in blocking_events
                    if per_event.get(int(event), {}).get("reasons")
                ]
            )

        pe8_evidence = _pe8_evidence_block(
            calibration,
            per_event,
            resolved_events,
            artifact_ref=str(calibration_artifact_ref) if calibration_artifact_ref else None,
            artifact_file_sha256=calibration_file_sha256,
            required=bool(require_calibration),
        )
        if calibration is not None and calibration_digest != pe8_evidence.get("artifact_digest"):
            raise GenerationRefused(
                cb.STATE_EVIDENCE_MISSING,
                ["the retained PE-8 artifact digest changed during certification"],
            )
        refreshed_snapshot = _snapshot_identity_block(snapshot_block)
        if refreshed_snapshot["sha256"] != snapshot_block["sha256"]:
            raise GenerationSnapshotUnverified(
                ["the pinned snapshot changed while generation certification was in progress"]
            )
        if calibration_artifact_ref is not None:
            from . import execution_snapshot as es

            if es.file_sha256(str(calibration_artifact_ref)) != calibration_file_sha256:
                raise GenerationRefused(
                    cb.STATE_EVIDENCE_MISSING,
                    ["the retained PE-8 artifact changed while generation certification was in progress"],
                )
        if authoritative_code_identity() != code_identity:
            raise GenerationRefused(
                cb.STATE_EVIDENCE_MISSING,
                ["the authoritative model code identity changed while generation certification was in progress"],
            )
        disclosure = cb.disclosure_block(calibration)
        manifest = build_generation_manifest(
            planning_event=int(planning_event),
            horizon_kind=str(horizon_kind),
            cutoff=str(cutoff),
            events=resolved_events,
            per_event=per_event,
            required_model_versions=required_versions,
            code_snapshot_sha256=code_identity,
            snapshot=snapshot_block,
            horizon_state=str(horizon["status"]),
            horizon_length=resolved_length,
            last_event=resolved_last_event,
            pe8_evidence=pe8_evidence,
            disclosure=disclosure,
            search_permission_causality=search_permission_causality,
        )
        generation_id = generation_id_of(manifest)
        created_at = (clock or _utc_now)()
        # The persisted bytes ARE the semantic manifest: row metadata (created_at, the
        # canonical snapshot column) lives in its own column and never inside the identity.
        manifest_json = canonical_manifest_bytes(manifest).decode("utf-8")

        conn.execute(
            "INSERT INTO generation(generation_id, manifest_json, manifest_sha256, planning_event,"
            " horizon_kind, cutoff, snapshot_path, snapshot_sha256, snapshot_source_db_identity,"
            " execution_run_uuid, created_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?)"
            " ON CONFLICT(generation_id) DO NOTHING",
            (
                generation_id,
                manifest_json,
                generation_id,
                int(planning_event),
                str(horizon_kind),
                str(cutoff),
                snapshot_block.get("path"),
                snapshot_block.get("sha256"),
                json.dumps(snapshot_block.get("source_db_identity"), sort_keys=True, default=str),
                snapshot_block.get("execution_run_uuid"),
                str(created_at),
            ),
        )
        conn.execute(
            "INSERT INTO current_generation(planning_event, horizon_kind, generation_id, updated_at)"
            " VALUES (?,?,?,?)"
            " ON CONFLICT(planning_event, horizon_kind) DO UPDATE SET"
            " generation_id=excluded.generation_id, updated_at=excluded.updated_at",
            (int(planning_event), str(horizon_kind), generation_id, str(created_at)),
        )
    return load_generation(conn, generation_id)


def certify_wildcard_value_generation(
    conn: sqlite3.Connection,
    *,
    chip_generation_id: str,
    planning_event: int,
    cutoff: str,
    events: Sequence[int],
    runs_by_event: Mapping[int, Mapping[str, int]],
    snapshot: Mapping[str, Any] | None,
    calibration: Mapping[str, Any] | None = None,
    calibration_artifact_ref: str | Path | None = None,
    require_calibration: bool = False,
    controller: Any = None,
    clock: Callable[[], str] | None = None,
) -> CertifiedGeneration:
    """Certify Wildcard's 6–10 event value product against a normal generation.

    The four-event normal-transfer generation remains a separate, unchanged
    product. This entry point requires a verified ``FOUR_GW`` generation for
    the same planning event and exact first four events, and binds the Wildcard
    value generation to that generation's cutoff, pinned snapshot and
    predictive code identity. Later planning runs cannot be stitched together
    by supplying a different cutoff, snapshot or code identity.
    """

    value_events = tuple(int(event) for event in events)
    base = load_generation(conn, str(chip_generation_id))
    base_report = verify_generation(conn, base.generation_id)
    if not base_report.get("verified"):
        raise GenerationRefused(
            DIAG_GENERATION_NOT_CERTIFIED,
            ["the normal four-event generation did not verify"],
        )
    if base.horizon_kind != HORIZON_KIND_FOUR_GW:
        raise GenerationRefused(
            DIAG_GENERATION_NOT_CERTIFIED,
            [f"the Wildcard value product requires a FOUR_GW base, got {base.horizon_kind!r}"],
        )
    if int(base.planning_event) != int(planning_event):
        raise GenerationRefused(
            DIAG_GENERATION_NOT_CERTIFIED,
            ["the Wildcard and normal products have different planning events"],
        )
    if tuple(base.events) != value_events[:4]:
        raise GenerationRefused(
            DIAG_GENERATION_NOT_CERTIFIED,
            ["the normal four-event product is not the exact first four Wildcard value events"],
        )
    normalized_runs = {
        int(event): {str(family): int(run_id) for family, run_id in families.items()}
        for event, families in runs_by_event.items()
    }
    prefix_run_disagreements = [
        f"GW{event}: Wildcard runs {normalized_runs.get(event)!r} do not exactly reuse "
        f"the normal certified runs {base.runs_for(event)!r}"
        for event in base.events
        if normalized_runs.get(int(event)) != base.runs_for(int(event))
    ]
    if prefix_run_disagreements:
        raise GenerationRefused(
            DIAG_GENERATION_NOT_CERTIFIED,
            [
                "the first four Wildcard value events must reuse the exact normal-generation "
                "run ids and dependency closure; separately generated later planning runs "
                "cannot be stitched into this assessment",
                *prefix_run_disagreements,
            ],
        )
    if str(base.cutoff) != str(cutoff):
        raise GenerationRefused(
            DIAG_GENERATION_NOT_CERTIFIED,
            ["the Wildcard and normal products have different cutoffs"],
        )
    snapshot_block = _snapshot_identity_block(snapshot)
    base_snapshot = base.manifest.get("data_snapshot") or {}
    if str(snapshot_block.get("sha256")) != str(base_snapshot.get("sha256")):
        raise GenerationRefused(
            DIAG_GENERATION_NOT_CERTIFIED,
            ["the Wildcard value product does not use the normal product's pinned data snapshot"],
        )
    base_code = str(base.manifest.get("code_snapshot_sha256") or "")
    if not base_code or base_code != authoritative_code_identity():
        raise GenerationRefused(
            DIAG_GENERATION_NOT_CERTIFIED,
            ["the normal product's predictive code identity differs from the current declared identity"],
        )

    result = certify_generation(
        conn,
        planning_event=int(planning_event),
        cutoff=str(cutoff),
        runs_by_event=runs_by_event,
        snapshot=snapshot_block,
        horizon_kind=HORIZON_KIND_WILDCARD_VALUE,
        events=value_events,
        horizon_length=len(value_events),
        calibration=calibration,
        calibration_artifact_ref=calibration_artifact_ref,
        require_calibration=require_calibration,
        controller=controller,
        clock=clock,
    )
    report = verify_generation(conn, result.generation_id)
    if not report.get("verified"):
        raise GenerationRefused(
            DIAG_GENERATION_NOT_CERTIFIED,
            ["the newly certified Wildcard value generation failed its independent verification"],
        )
    if (
        str(result.cutoff) != str(base.cutoff)
        or str(result.manifest.get("code_snapshot_sha256")) != base_code
        or str((result.manifest.get("data_snapshot") or {}).get("sha256"))
        != str(base_snapshot.get("sha256"))
    ):
        raise GenerationRefused(
            DIAG_GENERATION_NOT_CERTIFIED,
            ["the Wildcard and normal products do not reproduce one predictive identity"],
        )
    base_events = base.manifest.get("per_event") or {}
    result_events = result.manifest.get("per_event") or {}
    dependency_disagreements = [
        int(event)
        for event in base.events
        if (base_events.get(str(int(event))) or {}).get("dependency_closure")
        != (result_events.get(str(int(event))) or {}).get("dependency_closure")
    ]
    if dependency_disagreements:
        raise GenerationRefused(
            DIAG_GENERATION_NOT_CERTIFIED,
            [
                "the first four Wildcard events do not reproduce the normal generation's "
                f"dependency closure for {dependency_disagreements}"
            ],
        )
    return result


def certify_future_wildcard_value_generation(
    conn: sqlite3.Connection,
    *,
    coverage_product: Mapping[str, Any],
    planning_event: int,
    controller: Any = None,
    clock: Callable[[], str] | None = None,
) -> CertifiedGeneration:
    """Certify a future 6–10 event WC window from one origin-pinned product.

    This is the future-opportunity counterpart to
    :func:`certify_wildcard_value_generation`. The existing normal FOUR_GW
    generation is still the origin product's prefix. A future value window is
    derived only from the verified CHIP_RESERVATION product named by the
    retained coverage record; no later planning runs, free-standing run ids,
    snapshot or cutoff are caller supplied.
    """

    from . import chip_decision as cd, chip_reservation_forecast as crf

    source_identity = coverage_product.get("source_identity")
    if not isinstance(source_identity, Mapping):
        raise GenerationRefused(
            DIAG_GENERATION_NOT_CERTIFIED,
            ["the future WC coverage product omits its origin source identity"],
        )
    origin_event = int(source_identity.get("planning_event") or -1)
    expiry_event = coverage_product.get("expiry_event")
    try:
        verified_coverage = crf.verify_reservation_coverage_product(
            conn,
            coverage_product,
            expected={
                "action": cd.CHIP_ACTION_WC,
                "planning_event": origin_event,
                "expiry_event": expiry_event,
                "source_identity": dict(source_identity),
            },
        )
    except Exception as failure:
        raise GenerationRefused(
            DIAG_GENERATION_NOT_CERTIFIED,
            [f"the future WC coverage product did not verify: {failure}"],
        ) from failure
    if not verified_coverage.get("verified") or coverage_product.get("coverage_complete") is not True:
        raise GenerationRefused(
            DIAG_GENERATION_NOT_CERTIFIED,
            ["future WC certification requires complete, verified expiry coverage"],
        )

    event = int(planning_event)
    horizon_length = int(coverage_product.get("wildcard_value_horizon_length") or 0)
    value_events = tuple(range(event, event + horizon_length))
    forecast_events = {int(value) for value in coverage_product.get("forecast_events") or ()}
    if (
        expiry_event is None
        or event <= origin_event
        or event not in forecast_events
        or not WILDCARD_VALUE_MIN_EVENTS <= horizon_length <= WILDCARD_VALUE_MAX_EVENTS
    ):
        raise GenerationRefused(
            DIAG_GENERATION_NOT_CERTIFIED,
            ["future WC needs a known unexpired event and a declared 6–10 event value horizon"],
        )

    try:
        origin = load_generation(conn, str(coverage_product.get("root_generation_id") or ""))
        continuation = load_generation(conn, str(coverage_product.get("product_generation_id") or ""))
        origin_report = verify_generation(conn, origin.generation_id)
        continuation_report = verify_generation(conn, continuation.generation_id)
    except Exception as failure:
        raise GenerationRefused(
            DIAG_GENERATION_NOT_CERTIFIED,
            [f"future WC origin or continuation generation refused: {failure}"],
        ) from failure
    if (
        not origin_report.get("verified")
        or origin.horizon_kind != HORIZON_KIND_FOUR_GW
        or not continuation_report.get("verified")
        or continuation.horizon_kind != HORIZON_KIND_CHIP_RESERVATION
        or int(origin.planning_event) != origin_event
        or int(continuation.planning_event) != origin_event
        or tuple(continuation.events[:4]) != tuple(origin.events)
        or str(origin.cutoff) != str(continuation.cutoff)
        or str(origin.cutoff) != str(source_identity.get("origin_cutoff"))
        or str((origin.manifest.get("data_snapshot") or {}).get("sha256"))
        != str((continuation.manifest.get("data_snapshot") or {}).get("sha256"))
        or str(continuation.manifest.get("code_snapshot_sha256"))
        != str(origin.manifest.get("code_snapshot_sha256"))
        or str(continuation.manifest.get("code_snapshot_sha256"))
        != str(source_identity.get("predictive_code_snapshot_sha256"))
        or continuation.generation_id != str(coverage_product.get("product_generation_id"))
    ):
        raise GenerationRefused(
            DIAG_GENERATION_NOT_CERTIFIED,
            ["future WC source is not one origin-pinned normal/continuation identity"],
        )
    if not set(value_events).issubset({int(value) for value in continuation.events}):
        raise GenerationRefused(
            DIAG_GENERATION_NOT_CERTIFIED,
            ["the origin-pinned continuation product does not cover the complete future WC value window"],
        )
    source_runs = coverage_product.get("product_runs_by_event")
    if not isinstance(source_runs, Mapping) or any(
        {str(key): int(value) for key, value in (source_runs.get(str(value_event)) or {}).items()}
        != continuation.runs_for(value_event)
        for value_event in value_events
    ):
        raise GenerationRefused(
            DIAG_GENERATION_NOT_CERTIFIED,
            ["future WC runs differ from the verified CHIP_RESERVATION product"],
        )

    base_code = str(continuation.manifest.get("code_snapshot_sha256") or "")
    if not base_code or base_code != authoritative_code_identity():
        raise GenerationRefused(
            DIAG_GENERATION_NOT_CERTIFIED,
            ["origin-pinned predictive code identity differs from the current declared identity"],
        )
    runs_by_event = {value_event: continuation.runs_for(value_event) for value_event in value_events}
    snapshot = continuation.manifest.get("data_snapshot")
    if not isinstance(snapshot, Mapping):
        raise GenerationRefused(
            DIAG_GENERATION_NOT_CERTIFIED,
            ["origin-pinned continuation has no retained data snapshot"],
        )

    result = certify_generation(
        conn,
        planning_event=event,
        cutoff=str(continuation.cutoff),
        runs_by_event=runs_by_event,
        snapshot=snapshot,
        horizon_kind=HORIZON_KIND_WILDCARD_VALUE,
        events=value_events,
        horizon_length=horizon_length,
        controller=controller,
        clock=clock,
    )
    report = verify_generation(conn, result.generation_id)
    if not report.get("verified"):
        raise GenerationRefused(
            DIAG_GENERATION_NOT_CERTIFIED,
            ["the future Wildcard value generation failed independent verification"],
        )
    if (
        str(result.cutoff) != str(continuation.cutoff)
        or str(result.manifest.get("code_snapshot_sha256")) != base_code
        or str((result.manifest.get("data_snapshot") or {}).get("sha256"))
        != str((continuation.manifest.get("data_snapshot") or {}).get("sha256"))
    ):
        raise GenerationRefused(
            DIAG_GENERATION_NOT_CERTIFIED,
            ["future Wildcard generation differs from its origin predictive identity"],
        )
    source_events = continuation.manifest.get("per_event") or {}
    value_records = result.manifest.get("per_event") or {}
    closure_disagreements = [
        value_event for value_event in value_events
        if (source_events.get(str(value_event)) or {}).get("dependency_closure")
        != (value_records.get(str(value_event)) or {}).get("dependency_closure")
    ]
    if closure_disagreements:
        raise GenerationRefused(
            DIAG_GENERATION_NOT_CERTIFIED,
            [f"future Wildcard dependency closure differs from origin runs for {closure_disagreements}"],
        )
    return result


def certify_chip_reservation_generation(
    conn: sqlite3.Connection,
    *,
    chip_generation_id: str,
    planning_event: int,
    cutoff: str,
    events: Sequence[int],
    runs_by_event: Mapping[int, Mapping[str, int]],
    snapshot: Mapping[str, Any] | None,
    calibration: Mapping[str, Any] | None = None,
    calibration_artifact_ref: str | Path | None = None,
    require_calibration: bool = False,
    controller: Any = None,
    clock: Callable[[], str] | None = None,
) -> CertifiedGeneration:
    """Certify the origin-pinned event product used to cover chip expiry.

    This product is separate from normal decisions and does not relax their
    four-event contract. Its prefix must reuse the exact normal generation runs;
    its remaining events are certified from the same cutoff, snapshot and
    predictive code identity. Wildcard evaluators bind their own 6-10 event
    windows within this product.
    """

    product_events = tuple(int(event) for event in events)
    base = load_generation(conn, str(chip_generation_id))
    base_report = verify_generation(conn, base.generation_id)
    if not base_report.get("verified"):
        raise GenerationRefused(DIAG_GENERATION_NOT_CERTIFIED,
                                ["the normal four-event generation did not verify"])
    if base.horizon_kind != HORIZON_KIND_FOUR_GW:
        raise GenerationRefused(DIAG_GENERATION_NOT_CERTIFIED,
                                [f"chip-reservation product requires a FOUR_GW base, got {base.horizon_kind!r}"])
    if int(base.planning_event) != int(planning_event):
        raise GenerationRefused(DIAG_GENERATION_NOT_CERTIFIED,
                                ["chip-reservation and normal products have different planning events"])
    if product_events[:4] != tuple(int(event) for event in base.events):
        raise GenerationRefused(DIAG_GENERATION_NOT_CERTIFIED,
                                ["the normal four-event product is not the exact chip-reservation prefix"])
    if product_events != tuple(range(int(planning_event), int(planning_event) + len(product_events))):
        raise GenerationRefused(DIAG_GENERATION_NOT_CERTIFIED,
                                ["chip-reservation events are not contiguous from the origin event"])
    normalized_runs = {
        int(event): {str(family): int(run_id) for family, run_id in families.items()}
        for event, families in runs_by_event.items()
    }
    disagreements = [
        f"GW{event}: reservation runs {normalized_runs.get(int(event))!r} do not reuse "
        f"normal runs {base.runs_for(int(event))!r}"
        for event in base.events
        if normalized_runs.get(int(event)) != base.runs_for(int(event))
    ]
    if disagreements:
        raise GenerationRefused(
            DIAG_GENERATION_NOT_CERTIFIED,
            ["the first four reservation events must reuse the exact normal-generation run ids", *disagreements],
        )
    if str(cutoff) != str(base.cutoff):
        raise GenerationRefused(DIAG_GENERATION_NOT_CERTIFIED,
                                ["chip-reservation and normal products have different cutoffs"])
    snapshot_block = _snapshot_identity_block(snapshot)
    base_snapshot = base.manifest.get("data_snapshot") or {}
    base_code = str(base.manifest.get("code_snapshot_sha256") or "")
    if str(snapshot_block.get("sha256")) != str(base_snapshot.get("sha256")):
        raise GenerationRefused(DIAG_GENERATION_NOT_CERTIFIED,
                                ["chip-reservation product does not use the normal pinned snapshot"])
    if not base_code or base_code != authoritative_code_identity():
        raise GenerationRefused(DIAG_GENERATION_NOT_CERTIFIED,
                                ["normal predictive code identity differs from the current declared identity"])

    result = certify_generation(
        conn,
        planning_event=int(planning_event),
        cutoff=str(cutoff),
        runs_by_event=runs_by_event,
        snapshot=snapshot_block,
        horizon_kind=HORIZON_KIND_CHIP_RESERVATION,
        events=product_events,
        horizon_length=len(product_events),
        calibration=calibration,
        calibration_artifact_ref=calibration_artifact_ref,
        require_calibration=require_calibration,
        controller=controller,
        clock=clock,
    )
    report = verify_generation(conn, result.generation_id)
    if not report.get("verified"):
        raise GenerationRefused(DIAG_GENERATION_NOT_CERTIFIED,
                                ["the new chip-reservation product failed independent verification"])
    result_snapshot = result.manifest.get("data_snapshot") or {}
    if (
        str(result.cutoff) != str(base.cutoff)
        or str(result.manifest.get("code_snapshot_sha256")) != base_code
        or str(result_snapshot.get("sha256")) != str(base_snapshot.get("sha256"))
    ):
        raise GenerationRefused(DIAG_GENERATION_NOT_CERTIFIED,
                                ["chip-reservation and normal products do not share one predictive identity"])
    base_records = base.manifest.get("per_event") or {}
    product_records = result.manifest.get("per_event") or {}
    mismatched_prefix = [
        int(event) for event in base.events
        if (base_records.get(str(int(event))) or {}).get("dependency_closure")
        != (product_records.get(str(int(event))) or {}).get("dependency_closure")
    ]
    if mismatched_prefix:
        raise GenerationRefused(
            DIAG_GENERATION_NOT_CERTIFIED,
            [f"chip-reservation prefix dependency closure differs for {mismatched_prefix}"],
        )
    return result


def assert_writer_lease_held(conn: sqlite3.Connection, controller: Any) -> str:
    """Require the ACTIVE writer lease ``controller`` holds, or refuse with ``LeaseError``.

    Amendment 2 §8 opens the certification lifecycle with "acquire existing writer
    lease".  When a certification is driven by the execution controller it must happen
    UNDER that lease: a certification outside the single-writer boundary could
    interleave with another writer, and nothing would attribute it to a run.  The
    caller records its own ``CERTIFY_GENERATION`` stage, so the generation's
    provenance is the run ledger and the persisted row together.
    """

    if controller is None:
        from . import execution as ex

        raise ex.LeaseError(
            "CERTIFICATION_WITHOUT_WRITER_LEASE: generation publication requires an active writer "
            "lease held by a controller on the same database connection"
        )
    from . import execution as ex

    if not _same_sqlite_database(conn, controller.conn):
        raise ex.LeaseError(
            "CERTIFICATION_WITHOUT_WRITER_LEASE: the writer lease controller and generation store "
            "must use the same SQLite database"
        )

    lease = controller.active_lease(ex.LEASE_KIND_WRITER, "sqlite-writer")
    if lease is None:
        raise ex.LeaseError(
            "CERTIFICATION_WITHOUT_WRITER_LEASE: the certification controller holds no ACTIVE "
            "writer lease, so this certification would run outside the single-writer boundary"
        )
    if str(lease["run_uuid"]) != str(controller.run_uuid):
        raise ex.LeaseError(
            "CERTIFICATION_WITHOUT_WRITER_LEASE: the ACTIVE writer lease belongs to another run"
        )
    expires = ex.parse_utc_dt(lease["expires_at"])
    if expires is None or expires <= controller.now_dt():
        raise ex.LeaseError(
            "CERTIFICATION_WITHOUT_WRITER_LEASE: the writer lease is expired and is not ACTIVE"
        )
    if int(lease["owner_pid"]) != int(controller.pid) or str(lease["owner_host"]) != str(controller.host):
        raise ex.LeaseError(
            "CERTIFICATION_WITHOUT_WRITER_LEASE: the writer lease owner does not match its controller"
        )
    return str(controller.run_uuid)


def _same_sqlite_database(left: sqlite3.Connection, right: sqlite3.Connection) -> bool:
    """Compare the main database identity without trusting a caller-supplied path."""

    if left is right:
        return True

    def main_path(conn: sqlite3.Connection) -> str:
        row = next((item for item in conn.execute("PRAGMA database_list") if str(item[1]) == "main"), None)
        return str(row[2]) if row is not None else ""

    left_path, right_path = main_path(left), main_path(right)
    if not left_path or not right_path:
        return False
    return os.path.normcase(os.path.realpath(left_path)) == os.path.normcase(os.path.realpath(right_path))


def _utc_now() -> str:
    from .utils import utc_now

    return str(utc_now())


def _snapshot_identity_block(snapshot: Mapping[str, Any] | None) -> dict[str, Any]:
    """The pinned snapshot identity the manifest commits to -- REQUIRED, and VALIDATED.

    A generation is certified against ONE immutable source database, so the snapshot
    is not decoration: without it the certified runs cannot be re-derived from the
    causal source, and ``sha256`` is HASHED BY THE CERTIFIER from the file itself, so
    the manifest records a fact a verifier re-derives rather than a caller's claim.

    A snapshot that is ABSENT, that names no readable file, that carries no source
    database identity, or whose bytes do not hash to the recorded digest is REFUSED
    here (``GENERATION_SNAPSHOT_UNVERIFIED``) instead of being recorded as ``None``.
    A generation that pinned nothing could not be verified by anyone.
    """

    if snapshot is None:
        raise GenerationSnapshotUnverified(
            [
                "no data snapshot was supplied; a generation is certified against ONE immutable "
                "source database and commits to the bytes it actually used, so a certification "
                "without a pinned snapshot is not certified evidence"
            ]
        )
    snapshot = dict(snapshot)
    identity = snapshot.get("source_db_identity") or snapshot.get("data_snapshot_source_db_identity")
    path = snapshot.get("path") or snapshot.get("data_snapshot_path")
    claimed = snapshot.get("sha256") or snapshot.get("data_snapshot_sha256")
    if not path:
        raise GenerationSnapshotUnverified(
            [
                "the supplied snapshot names no file; a generation pins the immutable source it "
                "actually replaced"
            ]
        )
    from . import execution_snapshot as es

    snapshot_file = Path(str(path))
    if not snapshot_file.exists():
        raise GenerationSnapshotUnverified(
            [
                f"the data snapshot {path} does not exist; a generation pins the immutable source "
                "it actually replaced"
            ]
        )
    if not claimed:
        raise GenerationSnapshotUnverified(
            [
                f"the data snapshot {path} carries no recorded digest; the certifier commits to the "
                "bytes it actually used, and a claim nobody recorded cannot be re-derived"
            ]
        )
    live = es.file_sha256(snapshot_file)
    if str(claimed) != str(live):
        raise GenerationSnapshotUnverified(
            [
                f"the data snapshot {path} hashes to {live}, not the supplied {claimed}; a "
                "generation commits to the bytes it actually used"
            ]
        )
    if not isinstance(identity, Mapping) or not dict(identity):
        raise GenerationSnapshotUnverified(
            [
                f"the data snapshot {path} records no source database identity; a generation commits "
                "to WHICH source database it replaced, not merely to a file digest"
            ]
        )
    execution_run_uuid = str(snapshot.get("execution_run_uuid") or "").strip()
    if not execution_run_uuid:
        raise GenerationSnapshotUnverified(
            [f"the pinned snapshot {path} records no execution run UUID binding"]
        )
    identity = dict(identity)
    try:
        actual_identity = es.source_db_identity(snapshot_file)
    except (OSError, sqlite3.Error) as failure:
        raise GenerationSnapshotUnverified(
            [f"the data snapshot {path} cannot be opened as the source database it claims to be: {failure}"]
        ) from failure
    # ``source_db_identity`` is recorded from the live source at capture time.
    # Re-derive the database fields that VACUUM INTO preserves from the pinned
    # snapshot itself.  The source path and page count are intentionally excluded:
    # the snapshot has a different path and may be compacted while keeping identical
    # logical contents.  Its full bytes are already bound by ``live`` above.
    identity_fields = ("schema_version", "projection_runs_count", "projection_runs_max_id")
    missing_identity = [key for key in identity_fields if key not in identity]
    if missing_identity:
        raise GenerationSnapshotUnverified(
            [f"the source identity for snapshot {path} omits {missing_identity}"]
        )
    mismatches = [
        f"{key}: recorded {identity.get(key)!r}, snapshot contains {actual_identity.get(key)!r}"
        for key in identity_fields
        if identity.get(key) != actual_identity.get(key)
    ]
    if mismatches:
        raise GenerationSnapshotUnverified(
            ["the recorded source database identity does not reproduce from the pinned snapshot: " + "; ".join(mismatches)]
        )
    return {
        "path": str(path),
        "sha256": str(live),
        "size_bytes": int(snapshot_file.stat().st_size),
        "source_db_identity": identity,
        "execution_run_uuid": execution_run_uuid,
        **({
            "manifest_path": str(
                snapshot.get("manifest_path") or snapshot.get("data_snapshot_manifest")
            )
        } if (snapshot.get("manifest_path") or snapshot.get("data_snapshot_manifest")) else {}),
    }


def _derive_generation_causality(
    conn: sqlite3.Connection,
    *,
    snapshot: Mapping[str, Any],
    generation_planning_event: int,
    generation_cutoff: str,
) -> dict[str, Any]:
    """Reproduce temporal evidence from the snapshot manifest and execution row."""

    from . import execution_snapshot as es

    execution_run_uuid = str(snapshot.get("execution_run_uuid") or "")
    if not execution_run_uuid:
        raise GenerationRefused(
            DIAG_GENERATION_NOT_CERTIFIED,
            ["the origin snapshot has no execution UUID for causal timestamp derivation"],
        )
    try:
        run = conn.execute(
            "SELECT planning_event, planning_cutoff, started_at FROM execution_runs WHERE run_uuid=?",
            (execution_run_uuid,),
        ).fetchone()
    except sqlite3.Error as failure:
        raise GenerationRefused(
            DIAG_GENERATION_NOT_CERTIFIED,
            [f"the origin execution timestamp cannot be read: {failure}"],
        ) from failure
    if run is None:
        raise GenerationRefused(
            DIAG_GENERATION_NOT_CERTIFIED,
            [f"no retained execution run binds the origin snapshot UUID {execution_run_uuid}"],
        )
    if run["planning_event"] is None or not run["planning_cutoff"] or not run["started_at"]:
        raise GenerationRefused(
            DIAG_GENERATION_NOT_CERTIFIED,
            ["the retained origin execution omits planning event, cutoff, or started_at"],
        )
    if int(run["planning_event"]) != int(generation_planning_event):
        raise GenerationRefused(
            DIAG_GENERATION_NOT_CERTIFIED,
            [
                f"the origin execution planning event {int(run['planning_event'])} differs from "
                f"generation planning event {int(generation_planning_event)}"
            ],
        )
    origin_cutoff = str(run["planning_cutoff"])
    if origin_cutoff != str(generation_cutoff):
        raise GenerationRefused(
            DIAG_GENERATION_NOT_CERTIFIED,
            [
                f"the origin execution cutoff {origin_cutoff} differs from generation cutoff "
                f"{generation_cutoff}"
            ],
        )
    return es.derive_generation_causality(
        snapshot_path=str(snapshot.get("path") or ""),
        data_snapshot_sha256=str(snapshot.get("sha256") or ""),
        data_snapshot_size_bytes=int(snapshot.get("size_bytes") or 0),
        source_db_identity=dict(snapshot.get("source_db_identity") or {}),
        snapshot_manifest_path=(
            str(snapshot.get("manifest_path")) if snapshot.get("manifest_path") else None
        ),
        execution_run_uuid=execution_run_uuid,
        origin_planning_event=int(run["planning_event"]),
        origin_cutoff=origin_cutoff,
        execution_started_at=str(run["started_at"]),
    )


def _generation_causality_for_manifest(
    conn: sqlite3.Connection,
    *,
    snapshot: Mapping[str, Any],
    generation_planning_event: int,
    generation_cutoff: str,
) -> dict[str, Any]:
    """Persist a fail-closed temporal result even when required evidence is absent."""

    from . import execution_snapshot as es

    try:
        return _derive_generation_causality(
            conn,
            snapshot=snapshot,
            generation_planning_event=int(generation_planning_event),
            generation_cutoff=str(generation_cutoff),
        )
    except Exception as failure:  # evidence absence is a retained UNRESOLVED result
        detail = str(failure)
        unresolved_paths = {
            str(value)
            for value in (
                snapshot.get("path"),
                snapshot.get("manifest_path"),
                snapshot.get("data_snapshot_manifest"),
                (
                    Path(str(snapshot.get("path"))).parent / es.SNAPSHOT_MANIFEST
                    if snapshot.get("path")
                    else None
                ),
            )
            if value
        }
        for path in sorted(unresolved_paths, key=len, reverse=True):
            detail = detail.replace(path, "<retained-snapshot>")
        return {
            "schema": es.GENERATION_CAUSALITY_SCHEMA,
            "origin_planning_event": None,
            "origin_cutoff": str(generation_cutoff),
            "execution_run_uuid": str(snapshot.get("execution_run_uuid") or ""),
            "snapshot_sha256": str(snapshot.get("sha256") or ""),
            "temporal_status": "UNRESOLVED",
            "reasons": [f"UNRESOLVED: {type(failure).__name__}: {detail}"],
        }


def _run_identity_failures(
    conn: sqlite3.Connection,
    *,
    event: int,
    runs: Mapping[str, int],
    code_identity: str,
    snapshot: Mapping[str, Any],
) -> list[str]:
    """Bind every consumed run to the pinned snapshot and rederive its context."""

    failures: list[str] = []
    contexts: dict[str, list[str]] = {}
    from . import execution_snapshot as es

    try:
        snapshot_conn = sqlite3.connect(f"file:{Path(str(snapshot['path']))}?mode=ro", uri=True)
        snapshot_conn.row_factory = sqlite3.Row
        snapshot_conn.execute("PRAGMA query_only=ON")
    except (OSError, sqlite3.Error, KeyError) as failure:
        raise GenerationSnapshotUnverified(
            [f"GW{event}: the pinned snapshot cannot be opened to reproduce planning context: {failure}"]
        ) from failure
    try:
        current_digest = es.file_sha256(str(snapshot["path"]))
        if current_digest != str(snapshot["sha256"]):
            raise GenerationSnapshotUnverified(
                [f"the pinned snapshot changed before planning-context rederivation: {snapshot['path']}"]
            )
        for family in _declared_families(runs, {}):
            run_id = runs.get(family)
            row = cb._run(conn, int(run_id)) if run_id is not None else None
            if row is None:
                failures.append(f"GW{event}: {family} run {run_id!r} has no persisted identity row")
                continue
            recorded_code = str(row["source_snapshot_sha256"] or "").strip()
            if not recorded_code:
                failures.append(f"GW{event}: {family} run {run_id} records no authoritative code identity")
            elif recorded_code != str(code_identity):
                failures.append(
                    f"GW{event}: {family} run {run_id} code identity {recorded_code!r} "
                    f"!= authoritative identity {code_identity!r}"
                )

            run_snapshot = str(row["data_snapshot_sha256"] or "").strip()
            if not run_snapshot:
                failures.append(f"GW{event}: {family} run {run_id} records no data snapshot identity")
            elif run_snapshot != str(snapshot["sha256"]):
                failures.append(
                    f"GW{event}: {family} run {run_id} data snapshot {run_snapshot!r} "
                    f"!= pinned snapshot {snapshot['sha256']!r}"
                )
            run_uuid = str(row["execution_run_uuid"] or "").strip()
            if not run_uuid:
                failures.append(f"GW{event}: {family} run {run_id} records no execution run UUID")
            elif run_uuid != str(snapshot["execution_run_uuid"]):
                failures.append(
                    f"GW{event}: {family} run {run_id} execution UUID {run_uuid!r} "
                    f"!= pinned snapshot UUID {snapshot['execution_run_uuid']!r}"
                )

            context = str(row["planning_context_hash"] or "").strip()
            if not context:
                failures.append(f"GW{event}: {family} run {run_id} records no planning_context_hash")
                continue
            contexts.setdefault(context, []).append(family)
            raw_inputs = row["planning_context_inputs_json"]
            try:
                inputs = json.loads(str(raw_inputs)) if raw_inputs else None
                if not isinstance(inputs, Mapping):
                    raise ValueError("planning-context derivation inputs are absent")
                required = {
                    "entry_id", "season", "as_of", "scouting_stale_after_days",
                    "official_price_stale_after_hours",
                }
                if set(inputs) != required:
                    raise ValueError(f"planning-context input keys differ from {sorted(required)}")
                if str(inputs["as_of"]) != str(row["data_cutoff"]):
                    raise ValueError("planning-context as_of differs from the run data cutoff")
                from . import analytics
                from .planning import get_planning_context

                context_inputs = dict(inputs)
                entry_id = int(context_inputs.pop("entry_id"))
                context_as_of = str(context_inputs.pop("as_of"))
                reproduced = get_planning_context(
                    snapshot_conn,
                    entry_id,
                    int(event),
                    as_of=context_as_of,
                    season=context_inputs["season"],
                    scouting_stale_after_days=context_inputs["scouting_stale_after_days"],
                    official_price_stale_after_hours=context_inputs["official_price_stale_after_hours"],
                )
                reproduced_hash = analytics.planning_context_reference(reproduced)
                if reproduced_hash != context:
                    failures.append(
                        f"GW{event}: {family} run {run_id} planning_context_hash {context!r} "
                        f"does not reproduce from the pinned snapshot ({reproduced_hash!r})"
                    )
            except (TypeError, ValueError, KeyError, sqlite3.Error) as failure:
                failures.append(
                    f"GW{event}: {family} run {run_id} planning_context_hash cannot be reproduced "
                    f"from the pinned snapshot: {failure}"
                )
    finally:
        snapshot_conn.close()
    if len(contexts) > 1:
        failures.append(
            f"GW{event}: planning_context_hash differs across certified families "
            f"{ {value: sorted(families) for value, families in sorted(contexts.items())} }"
        )
    return failures


def _calibration_file_identity(
    calibration: Mapping[str, Any] | None,
    artifact_ref: str | Path | None,
    *,
    required: bool,
) -> tuple[str | None, str | None]:
    """Require consulted PE-8 evidence to have a retained, matching artifact."""

    if calibration is None:
        if required:
            raise GenerationRefused(
                cb.STATE_EVIDENCE_MISSING,
                ["a required PE-8 calibration evidence artifact is absent"],
            )
        if artifact_ref is not None:
            raise GenerationRefused(
                cb.STATE_EVIDENCE_MISSING,
                ["a PE-8 artifact path was supplied without its parsed evidence"],
            )
        return None, None
    if not artifact_ref:
        raise GenerationRefused(
            cb.STATE_EVIDENCE_MISSING,
            ["consulted PE-8 evidence has no retained artifact reference"],
        )
    path = Path(str(artifact_ref))
    if not path.is_file():
        raise GenerationRefused(
            cb.STATE_EVIDENCE_MISSING,
            [f"the consulted PE-8 evidence artifact {path} is not retained"],
        )
    from . import calibration_evaluation as ce
    from . import execution_snapshot as es

    try:
        retained = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as failure:
        raise GenerationRefused(
            cb.STATE_EVIDENCE_MISSING,
            [f"the consulted PE-8 evidence artifact {path} cannot be read: {failure}"],
        ) from failure
    if not isinstance(retained, Mapping) or ce.artifact_digest(retained) != ce.artifact_digest(calibration):
        raise GenerationRefused(
            cb.STATE_EVIDENCE_MISSING,
            [f"the retained PE-8 evidence artifact {path} does not match the supplied evidence"],
        )
    return ce.artifact_digest(calibration), es.file_sha256(path)


def _certification_identity_for_bundles(
    *, cutoff: str, bundle_identities: Mapping[str, str], snapshot_sha256: str
) -> str:
    from . import four_gw_decision as fg

    return fg.certification_identity_of(
        {
            "planning_cutoff": str(cutoff),
            "certified_bundle_identity": {
                str(event): str(identity) for event, identity in bundle_identities.items()
            },
            "data_snapshot_sha256": str(snapshot_sha256),
        }
    )


def _pe8_evidence_block(
    calibration: Mapping[str, Any] | None,
    per_event: Mapping[int, Mapping[str, Any]],
    events: Sequence[int],
    *,
    artifact_ref: str | None = None,
    artifact_file_sha256: str | None = None,
    required: bool = False,
) -> dict[str, Any]:
    """PE-8 evidence as it participates in the generation manifest.

    The frozen PE-8 semantics are preserved exactly: ``EVIDENCE_LIMITED`` and
    ``CALIBRATION_NOT_APPLICABLE`` are NOT failures, and nothing is promoted.  The
    block records the CALIBRATION IDENTITY, PE-8's own declared terminal state and
    each event's certification evidence state -- references a verifier reproduces
    rather than a re-derived judgement.
    """

    if calibration is None:
        return {
            "consulted": False,
            "required": bool(required),
            "identity": None,
            "reproduction": PE8_NOT_CONSULTED,
            "terminal_state": None,
            "state": cb.STATE_EVIDENCE_LIMITED,
            "per_event_state": {
                str(int(event)): str(per_event[int(event)].get("evidence_state")) for event in events
            },
            "refusals": [],
        }
    declared = calibration.get("identity") or {}
    from . import calibration_evaluation as ce

    surfaces = cb.calibration_surface_states(calibration)
    return {
        "consulted": True,
        "required": bool(required),
        "identity": cb.calibration_identity(calibration),
        "artifact_digest": ce.artifact_digest(calibration),
        "artifact_ref": str(artifact_ref) if artifact_ref else None,
        "artifact_file_sha256": str(artifact_file_sha256) if artifact_file_sha256 else None,
        "schema": calibration.get("schema"),
        "evaluation_version": calibration.get("evaluation_version"),
        # The identity's OWN declared fields travel with the reference, so an
        # independent verifier can RECOMPUTE the identity instead of trusting the
        # label.  Without them "the PE-8 evidence participated" would be a claim
        # nobody could reproduce.
        "certification_identity": declared.get("certification_identity"),
        "planning_cutoff": declared.get("planning_cutoff"),
        "terminal_state": cb.calibration_terminal_state(calibration)["state"],
        "state": cb.calibration_evidence_state(calibration, surfaces),
        "surfaces": [
            {
                "surface": reading.get("surface"),
                "state": reading.get("state"),
                "diagnosis": reading.get("diagnosis"),
            }
            for reading in surfaces
        ],
        "per_event_state": {
            str(int(event)): str(per_event[int(event)].get("evidence_state")) for event in events
        },
        "refusals": [],
    }


def reproduce_pe8_evidence(
    evidence: Mapping[str, Any], per_event: Mapping[str, Any]
) -> tuple[bool | None, list[str], str]:
    """Re-derive the PE-8 evidence reference from the generation's own record.

    Returns ``(reproduced, failures, boundary)``.  Two things are genuinely
    re-derived here:

    * the calibration IDENTITY, recomputed from the fields the manifest records and
      required to equal the recorded identity; and
    * the evidence STATE, recomputed by re-running PE-8's own roll-up over the
      recorded surfaces and terminal state.

    The boundary is stated rather than implied: PE-9 does not retain the PE-8
    artifact itself, so the surfaces are the recorded reading and the reproduction
    proves the reference and the roll-up, not the sample statistics behind them.
    """

    if not evidence.get("consulted"):
        # No calibration claim was made (``EVIDENCE_LIMITED`` is a state, never a
        # failure).  There is no reference to reproduce and none is claimed.
        return None, [], PE8_NOT_CONSULTED
    failures: list[str] = []
    recomputed_identity = cb.calibration_identity(
        {
            "schema": evidence.get("schema"),
            "evaluation_version": evidence.get("evaluation_version"),
            "identity": {
                "certification_identity": evidence.get("certification_identity"),
                "planning_cutoff": evidence.get("planning_cutoff"),
            },
        }
    )
    if recomputed_identity is None or str(recomputed_identity) != str(evidence.get("identity")):
        failures.append(
            "the recorded PE-8 calibration identity does not recompute from the fields the "
            f"manifest carries ({evidence.get('identity')} vs {recomputed_identity})"
        )
    recomputed_state = cb.calibration_evidence_state(
        {"terminal_state": {"state": evidence.get("terminal_state")}},
        [surface for surface in (evidence.get("surfaces") or []) if isinstance(surface, Mapping)],
    )
    if str(recomputed_state) != str(evidence.get("state")):
        failures.append(
            f"the recorded PE-8 evidence state {evidence.get('state')!r} does not reproduce from "
            f"its own surfaces and terminal state ({recomputed_state!r})"
        )
    for event, state in sorted((evidence.get("per_event_state") or {}).items()):
        record = (per_event or {}).get(str(event)) or {}
        if str(record.get("evidence_state")) != str(state):
            failures.append(
                f"GW{event}: the PE-8 evidence block records state {state!r} but the certified "
                f"record says {record.get('evidence_state')!r}"
            )
    return (not failures), failures, PE8_IDENTITY_AND_STATE_RECOMPUTED


# ---------------------------------------------------------------------------
# Verification (re-derivation audit)
# ---------------------------------------------------------------------------


def verify_generation(conn: sqlite3.Connection, generation_id: str) -> dict[str, Any]:
    """Re-derive a generation from retained runs, snapshot, and PE-8 evidence."""

    from . import execution_snapshot as es
    from . import four_gw_decision as fg

    generation = load_generation(conn, generation_id)
    manifest = dict(generation.manifest)
    failures: list[str] = []
    if generation.horizon_kind not in HORIZON_KINDS:
        failures.append(f"the generation declares unknown horizon kind {generation.horizon_kind!r}")
    elif generation.horizon_kind == HORIZON_KIND_WILDCARD_VALUE:
        try:
            failures.extend(
                _wildcard_value_horizon_problems(
                    planning_event=int(generation.planning_event),
                    events=generation.events,
                    horizon_length=int(manifest.get("horizon_length") or 0),
                    last_event=int(manifest.get("last_event") or 0),
                )
            )
        except (TypeError, ValueError) as failure:
            failures.append(f"the Wildcard value horizon declaration is malformed: {failure}")
    elif generation.horizon_kind == HORIZON_KIND_CHIP_RESERVATION:
        try:
            failures.extend(
                _chip_reservation_horizon_problems(
                    planning_event=int(generation.planning_event),
                    events=generation.events,
                    horizon_length=int(manifest.get("horizon_length") or 0),
                    last_event=int(manifest.get("last_event") or 0),
                )
            )
        except (TypeError, ValueError) as failure:
            failures.append(f"the chip-reservation horizon declaration is malformed: {failure}")
    authoritative_versions = {
        str(family): str(version) for family, version in cb.declared_required_versions().items()
    }
    recorded_versions = {
        str(family): str(version)
        for family, version in (manifest.get("required_model_versions") or {}).items()
    }
    if recorded_versions != authoritative_versions:
        failures.append(
            "the manifest's required model versions do not match the authoritative declarations "
            f"({recorded_versions} vs {authoritative_versions})"
        )
    code_identity = str(manifest.get("code_snapshot_sha256") or "").strip()
    if len(code_identity) != 64 or any(ch not in "0123456789abcdef" for ch in code_identity.lower()):
        failures.append("the generation records no valid authoritative code identity")

    snapshot = manifest.get("data_snapshot") or {}
    snapshot_status = "MISSING"
    try:
        snapshot = _snapshot_identity_block(snapshot)
        snapshot_status = "VERIFIED"
    except GenerationRefused as failure:
        raise failure

    causal_evidence_reproduced: bool | None = None
    if "search_permission_causality" in manifest:
        causal_block = manifest.get("search_permission_causality")
        if not isinstance(causal_block, Mapping):
            failures.append("the retained search-permission causality block is not an object")
        elif causal_block.get("schema") != "fpl_brain.generation_causality.v1":
            failures.append("the retained search-permission causality block has an unknown schema")
        elif causal_block.get("temporal_status") == "CAUSAL":
            try:
                reproduced_causality = _derive_generation_causality(
                    conn,
                    snapshot=snapshot,
                    generation_planning_event=int(generation.planning_event),
                    generation_cutoff=generation.cutoff,
                )
            except Exception as failure:
                failures.append(
                    "the generation's causal timestamp evidence cannot be reproduced: "
                    f"{type(failure).__name__}: {failure}"
                )
                causal_evidence_reproduced = False
            else:
                causal_evidence_reproduced = dict(causal_block) == reproduced_causality
                if not causal_evidence_reproduced:
                    failures.append(
                        "the retained search-permission causality block differs from the "
                        "canonical snapshot/execution timestamp derivation"
                    )
        elif causal_block.get("temporal_status") == "UNRESOLVED":
            causal_evidence_reproduced = False
        else:
            failures.append("the retained search-permission causality status is malformed")

    base_bundles: dict[int, cb.CertifiedBundle] = {}
    dependency_ok = True
    runs_complete = True
    versions_ok = recorded_versions == authoritative_versions
    for event in generation.events:
        event = int(event)
        record = (manifest.get("per_event") or {}).get(str(event)) or {}
        if int(record.get("event") or -1) != event:
            failures.append(f"GW{event}: the manifest's event record does not identify GW{event}")
        runs = {str(f): int(r) for f, r in (record.get("runs") or {}).items()}
        missing = [family for family in cb.LOAD_REQUIRED_FAMILIES if family not in runs]
        if missing:
            failures.append(f"GW{event}: the manifest names no run for {sorted(missing)}")
            runs_complete = False
            continue
        for family, run_id in sorted(runs.items()):
            row = cb._run(conn, run_id)
            if row is None:
                failures.append(f"GW{event}: {family} run {run_id} no longer exists in the store")
                runs_complete = False
                continue
            if str(row["status"]) != "complete":
                failures.append(f"GW{event}: {family} run {run_id} status is {row['status']!r}, not complete")
                runs_complete = False
            if int(row["planning_event"]) != event:
                failures.append(f"GW{event}: {family} run {run_id} is planning_event {row['planning_event']}")
            if str(row["data_cutoff"]) != str(generation.cutoff):
                failures.append(
                    f"GW{event}: {family} run {run_id} data_cutoff {row['data_cutoff']} "
                    f"!= generation cutoff {generation.cutoff}"
                )
            declared = (record.get("model_versions") or {}).get(family)
            expected = authoritative_versions.get(family)
            if declared is None:
                failures.append(f"GW{event}: {family} has no retained model-version declaration")
                versions_ok = False
            elif str(row["model_version"]) != str(declared):
                failures.append(
                    f"GW{event}: {family} run {run_id} records version {row['model_version']!r}, "
                    f"not the manifest's {declared!r}"
                )
                versions_ok = False
            if expected is not None and str(row["model_version"]) != str(expected):
                failures.append(
                    f"GW{event}: {family} run {run_id} version {row['model_version']!r} "
                    f"!= authoritative version {expected!r}"
                )
                versions_ok = False

        identity_failures = _run_identity_failures(
            conn, event=event, runs=runs, code_identity=code_identity, snapshot=snapshot
        )
        failures.extend(identity_failures)
        try:
            bundle = cb.certified_bundle_from_explicit_ids(
                conn,
                event=event,
                cutoff=generation.cutoff,
                runs=runs,
                required_versions=authoritative_versions,
                data_snapshot_sha256=snapshot.get("sha256"),
                code_snapshot_sha256=code_identity,
                expected_data_snapshot_sha256=snapshot.get("sha256"),
                families=_declared_families(runs, record),
            )
            base_bundles[event] = bundle
            if bundle.bundle_identity() != str(record.get("bundle_identity") or ""):
                failures.append(
                    f"GW{event}: the recorded bundle identity does not reproduce from the run rows "
                    f"({record.get('bundle_identity')} vs {bundle.bundle_identity()})"
                )
            if dict(bundle.model_versions) != {
                str(k): str(v) for k, v in (record.get("model_versions") or {}).items()
            }:
                failures.append(f"GW{event}: the recorded model versions do not reproduce from the run rows")
            if bundle.planning_context_hash != record.get("planning_context_hash"):
                failures.append(f"GW{event}: planning_context_hash does not reproduce from the run rows")
            if not bundle.planning_context_hash:
                failures.append(f"GW{event}: no planning_context_hash is present on the certified runs")
        except cb.BundleIncoherent as failure:
            failures.append(f"GW{event}: " + "; ".join(failure.reasons))

        try:
            reproduced_closure = _dependency_closure(conn, event, runs)
        except cb.BundleIncoherent as failure:
            failures.append(f"GW{event}: dependency closure is incoherent: {failure}")
            dependency_ok = False
        else:
            recorded_closure = {
                str(family): {str(dep): int(run) for dep, run in (deps or {}).items()}
                for family, deps in (record.get("dependency_closure") or {}).items()
            }
            if recorded_closure != reproduced_closure:
                failures.append(
                    f"GW{event}: dependency closure {recorded_closure} does not reproduce "
                    f"from run rows ({reproduced_closure})"
                )
                dependency_ok = False

    pe8 = manifest.get("pe8_evidence")
    if not isinstance(pe8, Mapping):
        pe8 = {}
        failures.append("the generation has no PE-8 evidence state")
    pe8_consulted = bool(pe8.get("consulted"))
    pe8_required = bool(pe8.get("required"))
    if pe8_required and not pe8_consulted:
        failures.append("the generation requires PE-8 evidence but records that it was not consulted")
    calibration: Mapping[str, Any] | None = None
    pe8_artifact_verified: bool | None = None
    if pe8_consulted:
        artifact_ref = pe8.get("artifact_ref")
        if not artifact_ref or not Path(str(artifact_ref)).is_file():
            raise GenerationRefused(
                cb.STATE_EVIDENCE_MISSING,
                [f"the consulted PE-8 artifact {artifact_ref!r} is not retained"],
            )
        else:
            try:
                calibration_payload = json.loads(Path(str(artifact_ref)).read_text(encoding="utf-8"))
                if not isinstance(calibration_payload, Mapping):
                    raise ValueError("artifact root is not an object")
                calibration = calibration_payload
                actual_file_sha = es.file_sha256(str(artifact_ref))
                from . import calibration_evaluation as ce

                actual_artifact_digest = ce.artifact_digest(calibration_payload)
                if actual_file_sha != str(pe8.get("artifact_file_sha256")):
                    failures.append("the retained PE-8 artifact file digest does not match the manifest")
                if actual_artifact_digest != str(pe8.get("artifact_digest")):
                    failures.append("the retained PE-8 artifact content digest does not match the manifest")
                pe8_artifact_verified = (
                    actual_file_sha == str(pe8.get("artifact_file_sha256"))
                    and actual_artifact_digest == str(pe8.get("artifact_digest"))
                )
            except (OSError, json.JSONDecodeError, ValueError) as failure:
                failures.append(f"the retained PE-8 artifact is unreadable: {failure}")
                pe8_artifact_verified = False
    else:
        pe8_artifact_verified = None

    reproduced_pe8_identity: str | None = None
    if base_bundles and len(base_bundles) == len(generation.events):
        reproduced_pe8_identity = _certification_identity_for_bundles(
            cutoff=generation.cutoff,
            bundle_identities={
                str(event): bundle.bundle_identity() for event, bundle in base_bundles.items()
            },
            snapshot_sha256=str(snapshot.get("sha256") or ""),
        )
    certified_records: dict[int, dict[str, Any]] = {}
    for event, bundle in base_bundles.items():
        try:
            certified = cb.certify_event_bundle(
                conn,
                event=int(event),
                cutoff=generation.cutoff,
                runs=bundle.runs,
                required_versions=authoritative_versions,
                data_snapshot_sha256=snapshot.get("sha256"),
                code_snapshot_sha256=code_identity,
                expected_data_snapshot_sha256=snapshot.get("sha256"),
                calibration=calibration,
                require_calibration=pe8_required,
                certification_identity=reproduced_pe8_identity,
                families=_declared_families(bundle.runs, (manifest.get("per_event") or {}).get(str(event)) or {}),
            )
            certified_records[int(event)] = certified.as_dict()
            original = (manifest.get("per_event") or {}).get(str(event)) or {}
            for field in ("state", "structural_state", "evidence_state", "bundle_identity"):
                if certified.as_dict().get(field) != original.get(field):
                    failures.append(
                        f"GW{event}: the recorded {field} {original.get(field)!r} does not reproduce "
                        f"from retained run and PE-8 evidence ({certified.as_dict().get(field)!r})"
                    )
        except cb.CertificationRefused as failure:
            failures.append(f"GW{event}: PE-8 evidence does not bind to this generation: {failure}")
        except cb.BundleIncoherent as failure:
            failures.append(f"GW{event}: " + "; ".join(failure.reasons))

    pe8_reproduced: bool | None
    pe8_boundary: str
    if not pe8_consulted:
        pe8_reproduced, pe8_boundary = None, PE8_NOT_CONSULTED
    elif calibration is None:
        pe8_reproduced, pe8_boundary = False, "RETAINED_ARTIFACT_UNAVAILABLE"
    else:
        expected_pe8 = _pe8_evidence_block(
            calibration,
            {event: (manifest.get("per_event") or {}).get(str(event)) or {} for event in generation.events},
            generation.events,
            artifact_ref=str(pe8.get("artifact_ref")),
            artifact_file_sha256=str(pe8.get("artifact_file_sha256")),
            required=pe8_required,
        )
        comparable_fields = (
            "consulted", "required", "identity", "artifact_digest", "artifact_ref",
            "artifact_file_sha256", "schema", "evaluation_version", "certification_identity",
            "planning_cutoff", "terminal_state", "state", "surfaces", "per_event_state",
        )
        summary_matches = all(pe8.get(key) == expected_pe8.get(key) for key in comparable_fields)
        if not summary_matches:
            failures.append("the PE-8 summary does not reproduce from its retained artifact")
        pe8_reproduced, pe8_failures, pe8_boundary = reproduce_pe8_evidence(
            expected_pe8, manifest.get("per_event") or {}
        )
        failures.extend(pe8_failures)
        if pe8_reproduced is not None:
            pe8_reproduced = bool(pe8_reproduced and summary_matches and pe8_artifact_verified)

    reproduced_horizon: dict[str, Any] = {"status": "UNVERIFIABLE"}
    last_event = manifest.get("last_event")
    if snapshot_status == "VERIFIED":
        snapshot_conn = sqlite3.connect(f"file:{Path(str(snapshot.get('path')))}?mode=ro", uri=True)
        try:
            snapshot_conn.row_factory = sqlite3.Row
            actual_last_event = int(fg.season_last_event_from_db(snapshot_conn))
            if last_event is None or int(last_event) != actual_last_event:
                failures.append(
                    f"the recorded horizon last event {last_event!r} does not match the pinned "
                    f"snapshot ({actual_last_event})"
                )
            support = {
                event: {
                    "supported": str((certified_records.get(event) or {}).get("state"))
                    not in cb.BLOCKING_BUNDLE_STATES,
                    "data_cutoff": generation.cutoff,
                    "missing_families": [],
                }
                for event in generation.events
            }
            reproduced_horizon = fg.evaluate_horizon(
                planning_event=int(generation.planning_event),
                support_by_event=support,
                cutoff=generation.cutoff,
                length=int(manifest.get("horizon_length") or 0),
                last_event=actual_last_event,
            )
        except (sqlite3.Error, TypeError, ValueError) as failure:
            failures.append(f"the pinned snapshot cannot reproduce the horizon: {failure}")
        finally:
            snapshot_conn.close()
    if str(reproduced_horizon["status"]) != str(manifest.get("horizon_state")):
        failures.append(
            f"the recorded horizon state {manifest.get('horizon_state')!r} does not reproduce "
            f"({reproduced_horizon['status']!r})"
        )

    if failures:
        raise GenerationRefused(
            cb.STATE_PREDICTIVE_BUNDLE_INCOHERENT
            if any("incoherent" in failure.lower() for failure in failures)
            else DIAG_GENERATION_NOT_CERTIFIED,
            failures,
        )
    return {
        "schema": MANIFEST_SCHEMA,
        "generation_id": generation.generation_id,
        "planning_event": generation.planning_event,
        "horizon_kind": generation.horizon_kind,
        "cutoff": generation.cutoff,
        "events": list(generation.events),
        "horizon_length": int(manifest.get("horizon_length") or 0),
        "manifest_digest_matches": True,
        "runs_exist": True,
        "runs_complete": runs_complete,
        "versions_valid": versions_ok,
        "code_identity_reproduced_from_runs": True,
        "planning_context_reproduced_from_snapshot": True,
        "bundle_identities_reproduced": True,
        "dependency_closure_reproduced": dependency_ok,
        "snapshot_identity": snapshot_status,
        "snapshot_source_identity_reproduced": True,
        "causal_evidence_reproduced": causal_evidence_reproduced,
        "pe8_evidence_consulted": pe8_consulted,
        "pe8_evidence_artifact_digest_verified": pe8_artifact_verified,
        "pe8_evidence_refs_reproduce": pe8_reproduced,
        "pe8_evidence_reproduction": pe8_boundary,
        "pe8_evidence_replay_boundary": (
            "the retained PE-8 artifact file and content digests are verified; its calibration identity, "
            "world binding, per-event states, and evidence roll-up are recomputed from that artifact"
            if pe8_consulted
            else "PE-8 evidence was not consulted; no PE-8 reference or reproduction is claimed"
        ),
        "horizon_state": str(reproduced_horizon["status"]),
        "verified": True,
    }


def require_snapshot_retained(generation: CertifiedGeneration) -> None:
    """Refuse when a generation's pinned snapshot is no longer retained.

    The minimum safe rule of amendment 2 §18: snapshots referenced by surviving
    generation/decision evidence must not be garbage-collected.  This is the check
    that makes the rule mechanical at the point where it matters.
    """

    snapshot = generation.snapshot
    path = snapshot.get("path")
    if not path:
        return
    if not Path(str(path)).exists():
        raise GenerationSnapshotUnverified(
            [
                f"generation {generation.generation_id} pins snapshot {path}, which no longer exists; "
                "a snapshot referenced by surviving generation evidence must be retained"
            ]
        )
    from . import execution_snapshot as es

    live = es.file_sha256(str(path))
    if snapshot.get("sha256") and live != str(snapshot.get("sha256")):
        raise GenerationSnapshotUnverified(
            [
                f"generation {generation.generation_id} pins snapshot {path} at "
                f"{snapshot.get('sha256')}, but the file hashes to {live}"
            ]
        )


def assert_generation_bundles_valid(conn: sqlite3.Connection, generation: CertifiedGeneration) -> None:
    """Re-run the EXISTING per-event bundle validation against the pinned runs.

    This is the production load's evidence check: the manifest is a claim about the
    run rows, never a substitute for them.  The required versions come from the
    declaration block the manifest carries, cross-checked against the authoritative
    in-library source, so a generation minted under a version nobody pins is refused.
    """

    authoritative = cb.declared_required_versions()
    code_identity = str(generation.manifest.get("code_snapshot_sha256") or "").strip()
    if not code_identity:
        raise GenerationRefused(
            cb.STATE_EVIDENCE_MISSING,
            [
                f"generation {generation.generation_id} records no code identity; a certified world "
                "that names no code revision cannot be re-derived by anyone"
            ],
        )
    recorded = {
        str(family): str(version)
        for family, version in (generation.manifest.get("required_model_versions") or {}).items()
    }
    for family, pinned in sorted(recorded.items()):
        wanted = authoritative.get(family)
        if wanted is not None and str(pinned) != str(wanted):
            raise GenerationRefused(
                cb.STATE_UNSUPPORTED_MODEL_VERSION,
                [
                    f"generation {generation.generation_id} pins {family} {pinned!r}, which is not the "
                    f"authoritative version {wanted!r}"
                ],
            )
    if conn is None:
        return
    for event in generation.events:
        runs = generation.runs_for(int(event))
        record = (generation.manifest.get("per_event") or {}).get(str(int(event))) or {}
        try:
            cb.validate_certified_bundle(
                conn,
                event=int(event),
                cutoff=generation.cutoff,
                runs=runs,
                required_versions=authoritative,
                data_snapshot_sha256=(generation.manifest.get("data_snapshot") or {}).get("sha256"),
                code_snapshot_sha256=generation.manifest.get("code_snapshot_sha256"),
                expected_data_snapshot_sha256=(
                    generation.manifest.get("data_snapshot") or {}
                ).get("sha256"),
                # Every family the manifest DECLARES, plus the families a load reads.
                # A world that carries no Monte Carlo run declares none: its absence is
                # a declared absence, not an invented requirement -- while a family
                # that IS declared is validated in full, dependency edges included.
                families=_declared_families(runs, record),
            )
        except cb.BundleIncoherent as failure:
            raise GenerationRefused(cb.bundle_state_from_reasons(failure.reasons), failure.reasons) from failure


def assert_cutoff_matches(conn: sqlite3.Connection, generation: CertifiedGeneration, *, cutoff: str) -> None:
    """The decision's cutoff must be exactly the generation's certified cutoff."""

    if str(generation.cutoff) != str(cutoff):
        raise GenerationRefused(
            cb.STATE_PREDICTIVE_BUNDLE_INCOHERENT,
            [
                f"generation {generation.generation_id} was certified at cutoff {generation.cutoff}, "
                f"not the requested {cutoff}"
            ],
        )


# ---------------------------------------------------------------------------
# Production decision records
# ---------------------------------------------------------------------------


def _canonical_decision_bytes(payload: Mapping[str, Any]) -> bytes:
    return canonical_manifest_bytes(payload)


def append_engine_decision_record(
    conn: sqlite3.Connection,
    *,
    generation: CertifiedGeneration,
    manager_packet_sha256: str,
    request_sha256: str,
    result_sha256: str,
    runner_identity: str,
    evidence: Mapping[str, Any],
    decision_artifact_ref: str | None = None,
    clock: Callable[[], str] | None = None,
) -> str:
    """Append ONE production decision attribution row and return its id.

    ``decision_id`` is the content digest of the record's own identity-bearing
    fields, so re-verifying a decision is recomputing that digest -- not trusting a
    label.  ``created_at`` is deliberately NOT part of the id.
    """

    evidence_block = canonical_identity_value(dict(evidence), path="evidence")
    identity = {
        "schema": DECISION_RECORD_SCHEMA,
        "generation_id": generation.generation_id,
        "planning_event": int(generation.planning_event),
        "horizon_kind": str(generation.horizon_kind),
        "manager_packet_sha256": str(manager_packet_sha256),
        "request_sha256": str(request_sha256),
        "result_sha256": str(result_sha256),
        "runner_identity": str(runner_identity),
        "evidence": evidence_block,
        "decision_artifact_ref": decision_artifact_ref,
    }
    decision_id = "sha256:" + hashlib.sha256(_canonical_decision_bytes(identity)).hexdigest()
    created_at = (clock or _utc_now)()
    conn.execute(
        "INSERT INTO engine_decision_records(decision_id, generation_id, planning_event, horizon_kind,"
        " manager_packet_sha256, request_sha256, result_sha256, runner_identity, evidence_json,"
        " decision_artifact_ref, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)"
        " ON CONFLICT(decision_id) DO NOTHING",
        (
            decision_id,
            generation.generation_id,
            int(generation.planning_event),
            str(generation.horizon_kind),
            str(manager_packet_sha256),
            str(request_sha256),
            str(result_sha256),
            str(runner_identity),
            json.dumps(evidence_block, sort_keys=True, separators=(",", ":")),
            decision_artifact_ref,
            str(created_at),
        ),
    )
    return decision_id


def load_engine_decision_record(conn: sqlite3.Connection, decision_id: str) -> dict[str, Any]:
    wanted = str(decision_id or "").strip()
    if not wanted:
        raise GenerationRefused(DIAG_DECISION_RECORD_UNKNOWN, ["no decision id was supplied"])
    row = conn.execute(
        "SELECT * FROM engine_decision_records WHERE decision_id=?", (wanted,)
    ).fetchone()
    if row is None:
        raise GenerationRefused(
            DIAG_DECISION_RECORD_UNKNOWN, [f"no decision record {wanted!r} exists"]
        )
    return dict(row)


def decision_identity_of(record: Mapping[str, Any]) -> str:
    """Recompute a decision record's id from its own persisted fields."""

    evidence = json.loads(str(record.get("evidence_json") or "{}"))
    record_schema = str(evidence.get("schema") or "") if isinstance(evidence, Mapping) else ""
    if record_schema not in DECISION_RECORD_SCHEMAS:
        raise ValueError(f"decision record declares unsupported evidence schema {record_schema!r}")
    identity = {
        "schema": record_schema,
        "generation_id": str(record.get("generation_id")),
        "planning_event": int(record.get("planning_event")),
        "horizon_kind": str(record.get("horizon_kind")),
        "manager_packet_sha256": str(record.get("manager_packet_sha256")),
        "request_sha256": str(record.get("request_sha256")),
        "result_sha256": str(record.get("result_sha256")),
        "runner_identity": str(record.get("runner_identity")),
        "evidence": evidence,
        "decision_artifact_ref": record.get("decision_artifact_ref"),
    }
    return "sha256:" + hashlib.sha256(_canonical_decision_bytes(identity)).hexdigest()


def verify_decision(conn: sqlite3.Connection, decision_id: str) -> dict[str, Any]:
    """Re-derive a production decision from the generation it attributes.

    Proves, or refuses with the specific failed step:

    * the referenced generation verifies;
    * the record's own id recomputes from its persisted fields;
    * the RETAINED decision artifact's result digest recomputes to the recorded one;
    * the manager packet and the request RECOMPUTE to the recorded digests.

    A record whose artifact is not retained is NOT reported as verified: the manager
    packet, the request and the result are exactly what that artifact holds, so
    without it nothing about the decision can be re-derived.  That refusal is the
    honest answer -- the alternative is the old behaviour of reporting digests
    "verified" when no source for them existed.

    Where the engine does not have byte-for-byte replayable decision execution, this
    verifies the STRONGEST frozen persisted artifact identity available and names the
    boundary in ``replay_boundary`` rather than inventing determinism.
    """

    record = load_engine_decision_record(conn, decision_id)
    failures: list[str] = []
    try:
        evidence_record = json.loads(str(record.get("evidence_json") or "{}"))
        if not isinstance(evidence_record, Mapping):
            raise ValueError("decision evidence is not an object")
        recomputed = decision_identity_of(record)
    except (TypeError, ValueError, json.JSONDecodeError) as failure:
        raise GenerationRefused(
            DIAG_DECISION_RECORD_INVALID,
            [f"the persisted decision record evidence cannot be re-derived: {failure}"],
        ) from failure
    if recomputed != str(decision_id):
        failures.append(
            f"the record's fields digest to {recomputed}, not the id it is stored under "
            f"({decision_id}); the record was mutated after it was written"
        )
    generation_report = verify_generation(conn, str(record["generation_id"]))
    generation = load_generation(conn, str(record["generation_id"]))
    artifact_ref = record.get("decision_artifact_ref")
    if not artifact_ref:
        raise GenerationRefused(
            DIAG_DECISION_ARTIFACT_NOT_RETAINED,
            [
                f"decision {decision_id} references no retained decision artifact, so its manager "
                "packet, request and result cannot be re-derived; a decision record without its "
                "artifact is not verifiable evidence"
            ],
        )
    path = Path(str(artifact_ref))
    if not path.exists():
        raise GenerationRefused(
            DIAG_DECISION_ARTIFACT_NOT_RETAINED,
            [
                f"the decision artifact {artifact_ref} referenced by decision {decision_id} is no "
                "longer retained; a decision whose evidence was discarded is not verifiable"
            ],
        )
    from . import execution_snapshot as es

    artifact_digest = es.file_sha256(path)
    recorded_artifact_digest = str(evidence_record.get("decision_artifact_file_sha256") or "")
    if not recorded_artifact_digest or artifact_digest != recorded_artifact_digest:
        failures.append(
            "the retained decision artifact file digest does not match the decision record "
            f"({artifact_digest} vs {recorded_artifact_digest or '<missing>'})"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as failure:
        raise GenerationRefused(
            DIAG_DECISION_RECORD_INVALID,
            [f"the decision artifact {artifact_ref} is unreadable: {failure}"],
        ) from failure
    if not isinstance(payload, Mapping):
        raise GenerationRefused(
            DIAG_DECISION_RECORD_INVALID,
            [f"the decision artifact {artifact_ref} does not hold a decision payload"],
        )

    permission_evidence_verified: bool | None = None
    evidence_schema = str(evidence_record.get("schema") or "")
    if evidence_schema == DECISION_RECORD_SCHEMA:
        provenance = payload.get("provenance") or {}
        permission = provenance.get("search_permission_evaluation") if isinstance(provenance, Mapping) else None
        binding = evidence_record.get("search_permission")
        if not isinstance(permission, Mapping) or not isinstance(binding, Mapping):
            failures.append(
                "the new-format decision record or artifact omits required search-permission evidence"
            )
            permission_evidence_verified = False
        else:
            from . import search_permission as sp

            permission_evidence_verified = sp.verify_recorded_evaluation(
                conn,
                generation_id=str(record["generation_id"]),
                evaluation=permission,
                record_binding=binding,
            )
            if not permission_evidence_verified:
                failures.append(
                    "the retained search-permission block, digest, record binding, or origin identities "
                    "do not reproduce from the verified generation and its pinned snapshot"
                )
    elif evidence_schema == LEGACY_DECISION_RECORD_SCHEMA:
        # Explicit v1 evidence keeps its historical verifier contract. Missing
        # permission evidence in a v2 record never falls back to this path.
        permission_evidence_verified = None
    else:
        failures.append(f"the decision record evidence schema {evidence_schema!r} is unsupported")

    result_digest = result_identity_of(payload)
    if result_digest != str(record["result_sha256"]):
        failures.append(
            "the decision artifact's result digest does not match the recorded result_sha256 "
            f"({result_digest} vs {record['result_sha256']})"
        )
    attribution = payload.get("attribution")
    manager_packet_verified = False
    request_verified = False
    if not isinstance(attribution, Mapping):
        failures.append(
            "the decision artifact carries no attribution block (the manager packet and request it "
            "consumed), so the recorded manager/request digests cannot be reproduced"
        )
    else:
        manager_packet = attribution.get("manager_packet")
        if not isinstance(manager_packet, Mapping):
            failures.append("the decision artifact records no manager packet to reproduce")
        elif packet_identity(manager_packet) != str(record["manager_packet_sha256"]):
            failures.append(
                "the artifact's manager packet does not digest to the recorded "
                f"manager_packet_sha256 ({packet_identity(manager_packet)} vs "
                f"{record['manager_packet_sha256']})"
            )
        else:
            manager_packet_verified = True
        request_block = attribution.get("request")
        if not isinstance(request_block, Mapping):
            failures.append("the decision artifact records no request object to reproduce")
        else:
            rebuilt = request_identity(
                planning_event=int(record["planning_event"]),
                horizon_kind=str(record["horizon_kind"]),
                cutoff=str(payload.get("planning_cutoff")),
                request=request_block,
            )
            if rebuilt != str(record["request_sha256"]):
                failures.append(
                    "the artifact's request does not digest to the recorded request_sha256 "
                    f"({rebuilt} vs {record['request_sha256']})"
                )
            else:
                request_verified = True

    runner_identity = str(payload.get("runner_identity") or "")
    if not runner_identity or runner_identity != str(record.get("runner_identity") or ""):
        failures.append("the retained runner identity is missing or differs from the decision record")
    runner_code_identity = str(payload.get("runner_code_identity") or "")
    if (
        len(runner_code_identity) != 71
        or not runner_code_identity.startswith("sha256:")
        or any(ch not in "0123456789abcdef" for ch in runner_code_identity[7:].lower())
        or runner_code_identity != str(evidence_record.get("runner_code_identity") or "")
    ):
        failures.append("the retained runner code identity is missing or differs from the decision record")

    manager_context_verified = False
    if isinstance(attribution, Mapping) and isinstance(attribution.get("manager_packet"), Mapping):
        manager_packet = attribution["manager_packet"]
        recorded_context = str(payload.get("manager_context_sha256") or "")
        reproduced_context = ""
        manager_state_rederived = False
        try:
            if payload.get("schema") == "fpl_brain.four_gw_decision.v1":
                reproduced_state = _rederive_four_gw_consumed_manager_state(
                    generation, manager_packet
                )
                if "consumed_manager_state" in attribution:
                    retained_state = attribution.get("consumed_manager_state")
                    if not isinstance(retained_state, Mapping):
                        failures.append(
                            "the retained attribution has no canonical consumed manager state"
                        )
                    elif _json_round_trip_form(dict(retained_state)) != reproduced_state:
                        failures.append(
                            "the manager state retained as consumed differs from the state "
                            "re-derived from the pinned snapshot"
                        )
                    else:
                        reproduced_context = four_gw_manager_context_identity(reproduced_state)
                        manager_state_rederived = True
                else:
                    # Backward compatibility for v1 artifacts written before the
                    # canonical consumed state was retained separately.  The helper
                    # still cross-checks every supplied assertion before reproducing
                    # the original digest meaning.
                    from . import analytics

                    reproduced_context = analytics.canonical_hash(
                        {
                            "entry_id": int(reproduced_state["entry_id"]),
                            "planning_event": int(reproduced_state["planning_event"]),
                            "cutoff": str(reproduced_state["cutoff"]),
                            "manager_state": reproduced_state["source_manager_context"],
                        }
                    )
                    manager_state_rederived = True
            else:
                reproduced_context = _fallback_manager_context_identity(
                    manager_packet,
                    planning_event=int(record["planning_event"]),
                    cutoff=generation.cutoff,
                )
                manager_state_rederived = True
            if manager_state_rederived and (
                not recorded_context or reproduced_context != recorded_context
            ):
                failures.append(
                    "the manager context digest does not reproduce from the canonical consumed "
                    "manager state and pinned source snapshot "
                    f"({reproduced_context} vs {recorded_context or '<missing>'})"
                )
            elif manager_state_rederived:
                manager_context_verified = True
        except Exception as failure:  # evidence refusal, never an unverified success
            failures.append(f"the manager context cannot be re-derived: {type(failure).__name__}: {failure}")

    if str(evidence_record.get("runner_identity") or "") != runner_identity:
        failures.append("the decision record evidence does not bind the retained runner identity")
    if str(evidence_record.get("decision_artifact_file_sha256") or "") != artifact_digest:
        failures.append("the decision record evidence does not bind the retained artifact file digest")
    if failures:
        raise GenerationRefused(DIAG_DECISION_RECORD_INVALID, failures)
    return {
        "schema": evidence_schema,
        "decision_id": str(decision_id),
        "generation_id": str(record["generation_id"]),
        "planning_event": int(record["planning_event"]),
        "generation_verified": bool(generation_report["verified"]),
        "record_digest_recomputed": True,
        "manager_packet_digest_verified": bool(manager_packet_verified),
        "manager_context_digest_verified": bool(manager_context_verified),
        "request_digest_verified": bool(request_verified),
        "result_digest_verified": True,
        "search_permission_evidence_verified": permission_evidence_verified,
        "runner_identity_verified": True,
        "runner_code_identity_bound": True,
        "runner_code_identity_replay_boundary": (
            "the code identity digest is bound by the retained decision artifact and append-only record; "
            "the original source files are not retained as part of this decision artifact"
        ),
        "decision_artifact": "VERIFIED",
        "decision_artifact_sha256": str(artifact_digest),
        "replay_boundary": "DECISION_REPLAY_NOT_PERFORMED",
        "replay_boundary_note": (
            "the generation and manager context re-derive from retained evidence; the decision artifact "
            "file, runner identity, code identity, manager packet, request, and result digests are verified; "
            "the decision itself is not re-executed"
        ),
        "verified": True,
    }


def _json_round_trip_form(value: Any) -> Any:
    """The form a value takes after ``json.dumps`` -> ``json.loads``.

    Keys become STRINGS (that is what JSON does), tuples become lists, and a value
    JSON cannot encode is stringified.  Normalising BEFORE the encode is what makes
    the digest of an in-memory artifact equal the digest of the same artifact read
    back from disk -- which is exactly what the verifier does, so without this the
    "digest" would depend on whether the artifact had been persisted yet.
    """

    if isinstance(value, Mapping):
        return {_json_key(key): _json_round_trip_form(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_round_trip_form(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return str(value)


def _json_key(key: Any) -> str:
    if isinstance(key, bool):
        return "true" if key else "false"
    if key is None:
        return "null"
    return str(key)


def decision_result_bytes(payload: Mapping[str, Any]) -> bytes:
    """The canonical byte form of a decision RESULT.

    Deliberately different from the generation manifest's rule, and for a stated
    reason: a decision result carries measured numbers (route scores, confidence
    readings), and Python's JSON encoding of a float is stable under a
    ``dumps``/``loads`` round trip, which is exactly what the verifier performs -- it
    RELOADS the retained artifact.  The manifest rule stays float-free because a
    manifest must be re-derivable by an independent implementation, not merely
    re-readable by the same one.
    """

    return json.dumps(
        _json_round_trip_form(decision_result_projection(payload)),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def result_identity_of(payload: Mapping[str, Any]) -> str:
    """The recorded result digest of a persisted decision artifact."""

    return "sha256:" + hashlib.sha256(decision_result_bytes(payload)).hexdigest()


def decision_result_projection(payload: Mapping[str, Any]) -> dict[str, Any]:
    """The identity-bearing projection of a persisted decision artifact.

    Only the decision CONTENT is covered: the route table, the recommendation, the
    suppression reasons, the confidence block, the fixture horizon and the
    generation it consumed.  Volatile telemetry (wall times, worker counts) is
    deliberately excluded, so re-running the same frozen decision over the same
    certified generation reproduces the same digest.
    """

    payload = dict(payload)
    return {
        "schema": payload.get("schema"),
        "planning_event": payload.get("planning_event"),
        "planning_cutoff": payload.get("planning_cutoff"),
        "decision_events": [int(event) for event in (payload.get("decision_events") or [])],
        "generation_id": (payload.get("provenance") or {}).get("generation_id"),
        "runner_identity": payload.get("runner_identity"),
        "runner_code_identity": payload.get("runner_code_identity"),
        "manager_context_sha256": payload.get("manager_context_sha256"),
        "suppression_reasons": list(payload.get("suppression_reasons") or []),
        "decision_confidence": payload.get("decision_confidence"),
        "fixture_horizon": payload.get("fixture_horizon"),
        "decision": payload.get("decision"),
        "route_table": ((payload.get("finalist_refinement") or {}).get("route_table") or {}).get("routes"),
    }


# ---------------------------------------------------------------------------
# Descriptor-only production API
# ---------------------------------------------------------------------------


def packet_identity(manager_packet: Mapping[str, Any]) -> str:
    """The manager packet's content digest (manager state is not predictive identity)."""

    return "sha256:" + hashlib.sha256(
        canonical_manifest_bytes({"manager_packet": dict(manager_packet)})
    ).hexdigest()


def request_identity(*, planning_event: int, horizon_kind: str, cutoff: str, request: Mapping[str, Any] | None) -> str:
    """The decision REQUEST's content digest: the ordinary, non-predictive parameters."""

    return "sha256:" + hashlib.sha256(
        canonical_manifest_bytes(
            {
                "planning_event": int(planning_event),
                "horizon_kind": str(horizon_kind),
                "cutoff": str(cutoff),
                "request": dict(request or {}),
            }
        )
    ).hexdigest()


def _decision_runner_code_identity(profile: DecisionProfile) -> str:
    """Fingerprint the retained production entrypoint and the engine code it calls."""

    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "fpl_brain").glob("*.py"), key=lambda path: path.name)
    module_name = PRODUCTION_DECISION_MODULES.get(str(profile.kind))
    if module_name:
        paths.append(root / module_name)
    digest = hashlib.sha256()
    for path in sorted(set(paths), key=lambda item: item.relative_to(root).as_posix()):
        if not path.is_file():
            raise GenerationRefused(
                DIAG_PRODUCTION_PROFILE_UNKNOWN,
                [f"the declared decision code file {path} is not available to fingerprint"],
            )
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return "sha256:" + digest.hexdigest()


def _fallback_manager_context_identity(
    manager_packet: Mapping[str, Any], *, planning_event: int, cutoff: str
) -> str:
    from . import analytics

    return analytics.canonical_hash(
        {
            "entry_id": int(manager_packet.get("entry_id") or 0),
            "planning_event": int(planning_event),
            "cutoff": str(cutoff),
            "manager_state": manager_state_from_packet(manager_packet),
        }
    )


def _reproduce_four_gw_manager_context(
    generation: CertifiedGeneration, manager_packet: Mapping[str, Any]
) -> str:
    """Reproduce the legacy four-GW source-context hash from its pinned snapshot.

    New decision artifacts bind the full consumed state separately.  This remains
    for already-retained v1 decision artifacts, and now checks every manager-state
    assertion they retained before reporting the legacy context digest reproduced.
    """

    from . import analytics
    state = _rederive_four_gw_consumed_manager_state(generation, manager_packet)
    return analytics.canonical_hash(
        {
            "entry_id": int(state["entry_id"]),
            "planning_event": int(state["planning_event"]),
            "cutoff": str(state["cutoff"]),
            "manager_state": state["source_manager_context"],
        }
    )


def _forbidden_descriptors_in(value: Any, *, path: str) -> list[str]:
    """Every forbidden descriptor KEY found anywhere inside a production input.

    The production API is descriptor-only at EVERY boundary, not just at the top
    level: a caller that cannot pass ``cache_dir=`` as a keyword could otherwise
    hide it inside the request mapping, and one that cannot pass ``runs=`` could
    hide a run mapping inside the manager packet.  The check is by KEY NAME at any
    depth, and a forbidden key refuses its entire subtree.
    """

    found: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            name = str(key)
            here = f"{path}.{name}"
            normalized = name.strip().casefold().replace("-", "_").replace(" ", "_")
            if normalized in FORBIDDEN_PRODUCTION_DESCRIPTORS:
                found.append(here)
                continue
            found.extend(_forbidden_descriptors_in(item, path=here))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found.extend(_forbidden_descriptors_in(item, path=f"{path}[{index}]"))
    return found


def assert_descriptor_only(kwargs: Mapping[str, Any]) -> None:
    """Refuse a production call that carries a predictive descriptor.

    The production caller may provide manager state, a planning event, a generation
    selector, ordinary decision parameters and a request tracing ID.  It must NOT
    provide matrices, bundles, predictive run mappings, dependency mappings,
    certification objects, caller-computed manifest digests, cache handles, registry
    entries or authority tokens -- so if any is present and not ``None``, the call is
    refused before anything is read.
    """

    smuggled = sorted(str(name) for name in kwargs)
    if smuggled:
        raise ProductionDescriptorOnly(
            [
                "a production decision accepts a manager packet, a planning event, a generation "
                f"selector and ordinary decision parameters only; it was also given {smuggled}. "
                "Predictive evidence is resolved from the certified generation, never from the caller"
            ]
        )


def assert_no_nested_descriptors(payload: Mapping[str, Any], *, boundary: str) -> None:
    """Refuse a prohibited descriptor hidden anywhere inside one production input."""

    found = sorted(set(_forbidden_descriptors_in(payload, path=boundary)))
    if found:
        raise ProductionDescriptorOnly(
            [
                "a production decision accepts ordinary decision parameters only; "
                f"{boundary} carries {found}. Predictive evidence, cache handles and certification "
                "objects are resolved from the certified generation, never supplied by the caller"
            ]
        )


#: The DECLARED production decision pipelines, keyed by horizon kind.  A caller
#: selects a profile BY NAME; it cannot supply an implementation.  The four-Gameweek
#: pipeline is the existing production runner module -- the same declared entry point
#: whose wiring the certification identity covers -- located here by name, never by a
#: caller-supplied path or callable.
PRODUCTION_DECISION_MODULES: dict[str, str] = {
    HORIZON_KIND_FOUR_GW: "scripts/run_four_gw_decision.py",
}
PRODUCTION_DECISION_ENTRYPOINT = "run_certified_four_gw_decision"


@dataclass(frozen=True)
class DecisionProfile:
    """The ORDINARY, non-predictive parameters one production decision runs with.

    ``kind`` names which declared production pipeline executes the decision.  It is
    a NAME from :data:`PRODUCTION_DECISION_MODULES`, not an implementation: an
    unknown kind is refused, and there is no parameter anywhere on the production
    API that accepts a callable.  ``parameters`` carries scheduling and search
    breadth (draw counts, beam width, worker count) -- never predictive evidence.
    """

    kind: str = HORIZON_KIND_FOUR_GW
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"kind": str(self.kind), "parameters": dict(self.parameters)}


def resolve_decision_profile(profile: Any) -> DecisionProfile:
    """Normalise and validate a decision profile, refusing an unknown kind."""

    if profile is None:
        return DecisionProfile()
    if isinstance(profile, DecisionProfile):
        resolved = profile
    elif isinstance(profile, Mapping):
        resolved = DecisionProfile(
            kind=str(profile.get("kind") or HORIZON_KIND_FOUR_GW),
            parameters=dict(profile.get("parameters") or {}),
        )
    else:
        raise ProductionDescriptorOnly(
            [
                "a decision profile is a declared kind plus ordinary decision parameters; "
                f"{type(profile).__name__} is not a profile"
            ]
        )
    assert_no_nested_descriptors(resolved.as_dict(), boundary="profile")
    if resolved.kind not in PRODUCTION_DECISION_MODULES and resolved.kind != HORIZON_KIND_MANAGER_WORLD:
        raise GenerationRefused(
            DIAG_PRODUCTION_PROFILE_UNKNOWN,
            [
                f"{resolved.kind!r} is not a declared production decision pipeline "
                f"({sorted([*PRODUCTION_DECISION_MODULES, HORIZON_KIND_MANAGER_WORLD])})"
            ],
        )
    allowed_parameters = (
        _FOUR_GW_DECISION_PARAMETERS
        if resolved.kind == HORIZON_KIND_FOUR_GW
        else _MANAGER_WORLD_DECISION_PARAMETERS
    )
    unknown_parameters = sorted(set(str(key) for key in resolved.parameters) - allowed_parameters)
    if unknown_parameters:
        raise ProductionDescriptorOnly(
            [
                f"profile {resolved.kind!r} accepts only declared ordinary parameters; "
                f"unknown parameter(s): {unknown_parameters}"
            ]
        )
    return resolved


def _declared_production_entrypoint(module_name: str) -> Callable[..., Mapping[str, Any]]:
    """Load the DECLARED production pipeline module by its repository-relative path.

    The library resolves it itself: no caller supplies the module, the path or the
    callable, so an implementation cannot be injected through the production API.
    """

    import importlib.util

    path = Path(__file__).resolve().parents[1] / module_name
    spec = importlib.util.spec_from_file_location(Path(module_name).stem, path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise GenerationRefused(
            DIAG_PRODUCTION_PROFILE_UNKNOWN, [f"the declared production module {module_name} is unreadable"]
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    entrypoint = getattr(module, PRODUCTION_DECISION_ENTRYPOINT, None)
    if not callable(entrypoint):  # pragma: no cover - defensive
        raise GenerationRefused(
            DIAG_PRODUCTION_PROFILE_UNKNOWN,
            [f"{module_name} does not provide {PRODUCTION_DECISION_ENTRYPOINT}()"],
        )
    return entrypoint


def manager_state_from_packet(manager_packet: Mapping[str, Any]) -> dict[str, Any]:
    """The manager state a production caller supplied, under either spelling.

    Manager state is a PERMITTED production input (amendment 2 §10), and both
    ``manager_packet["manager_state"]`` and a flat ``manager_packet["squad_ids"]``
    say the same thing about ONE manager.  Reading them through one helper keeps the
    pipelines consistent -- and keeps everything PREDICTIVE out of both spellings,
    because the descriptor check runs over the whole packet first.
    """

    nested = manager_packet.get("manager_state")
    state = dict(nested) if isinstance(nested, Mapping) else {}
    if manager_packet.get("squad_ids") is not None:
        state.setdefault("squad_ids", manager_packet.get("squad_ids"))
    return state


FOUR_GW_CONSUMED_MANAGER_STATE_SCHEMA = "fpl_brain.four_gw_consumed_manager_state.v1"


def canonical_four_gw_manager_state(
    *,
    entry_id: int,
    planning_event: int,
    cutoff: str,
    season: str | None,
    context: Any,
    squad: Mapping[str, Any],
    route_state: Any,
) -> dict[str, Any]:
    """Serialize the exact snapshot-derived manager state consumed by four-GW search.

    ``RouteState`` is the decision engine's actual initial state: it contains the
    squad and acquisition basis, bank in integer tenths, current free transfers,
    chip state, and the explicitly recorded event-start free-transfer value.  The
    source manager context is retained alongside it so the authority/provenance for
    those values can be independently re-derived later.
    """

    manager_context = dict(context.manager_state or {})
    bank = _required_manager_integer(manager_context.get("bank"), "bank")
    free_transfers = _required_manager_integer(
        manager_context.get("free_transfers"), "free_transfers"
    )
    event_start_free_transfers = _required_manager_integer(
        manager_context.get("event_start_free_transfers"),
        "event_start_free_transfers",
        allow_none=True,
    )
    if int(route_state.bank_tenths) != bank:
        raise GenerationRefused(
            DIAG_MANAGER_STATE_MISMATCH,
            ["the route state's bank_tenths differs from the pinned manager context bank"],
        )
    if int(route_state.free_transfers) != free_transfers:
        raise GenerationRefused(
            DIAG_MANAGER_STATE_MISMATCH,
            ["the route state's free_transfers differs from the pinned manager context"],
        )
    actual_event_start = route_state.event_start_free_transfers
    if actual_event_start is not None:
        actual_event_start = _required_manager_integer(
            actual_event_start, "route_state.event_start_free_transfers"
        )
    if actual_event_start != event_start_free_transfers:
        raise GenerationRefused(
            DIAG_MANAGER_STATE_MISMATCH,
            ["the route state's event-start free transfers differ from the pinned manager context"],
        )

    players = [player.as_dict() for player in route_state.players]
    player_ids = [int(player["player_id"]) for player in players]
    snapshot_squad_ids = sorted(int(pid) for pid in (squad.get("squad_ids") or []))
    if player_ids != snapshot_squad_ids:
        raise GenerationRefused(
            DIAG_MANAGER_STATE_MISMATCH,
            ["the route state's players differ from the squad resolved from the pinned snapshot"],
        )
    return {
        "schema": FOUR_GW_CONSUMED_MANAGER_STATE_SCHEMA,
        "entry_id": int(entry_id),
        "planning_event": int(planning_event),
        "cutoff": str(cutoff),
        "season": season,
        "source_manager_context": _json_round_trip_form(manager_context),
        "route_state": {
            "event": int(route_state.event),
            "players": players,
            "bank_tenths": bank,
            "free_transfers": free_transfers,
            "chip_state": _json_round_trip_form(list(route_state.chip_state or ())),
            "event_start_free_transfers": event_start_free_transfers,
        },
    }


def four_gw_manager_context_identity(consumed_manager_state: Mapping[str, Any]) -> str:
    """Digest the canonical state the four-GW production decision actually consumes."""

    from . import analytics

    state = _json_round_trip_form(dict(consumed_manager_state))
    return analytics.canonical_hash(
        {
            "schema": FOUR_GW_CONSUMED_MANAGER_STATE_SCHEMA,
            "entry_id": int(state["entry_id"]),
            "planning_event": int(state["planning_event"]),
            "cutoff": str(state["cutoff"]),
            "consumed_manager_state": state,
        }
    )


def assert_manager_packet_matches_canonical_state(
    manager_packet: Mapping[str, Any], consumed_manager_state: Mapping[str, Any]
) -> None:
    """Validate every caller-supplied manager-state assertion against snapshot state.

    Omitted assertions are allowed because the production boundary derives and
    retains the canonical state itself.  A present field, including a present
    ``None`` or numeric zero, is an assertion and must match semantically.
    """

    route_state = consumed_manager_state.get("route_state")
    if not isinstance(route_state, Mapping):
        raise GenerationRefused(
            DIAG_MANAGER_STATE_MISMATCH,
            ["the canonical four-GW manager state has no route_state"],
        )
    players = route_state.get("players") or []
    canonical = {
        "squad_ids": [int(player["player_id"]) for player in players],
        "bank_tenths": int(route_state["bank_tenths"]),
        "free_transfers": int(route_state["free_transfers"]),
        "event_start_free_transfers": route_state.get("event_start_free_transfers"),
        "authoritative_source": (
            consumed_manager_state.get("source_manager_context") or {}
        ).get("authoritative_source"),
    }
    assertions: list[tuple[str, Any, str]] = []
    nested = manager_packet.get("manager_state")
    if "manager_state" in manager_packet and not isinstance(nested, Mapping):
        raise GenerationRefused(
            DIAG_MANAGER_STATE_MISMATCH,
            ["manager_packet.manager_state must be an object of assertions when supplied"],
        )
    if isinstance(nested, Mapping):
        assertions.extend(
            (str(key), value, f"manager_packet.manager_state.{key}")
            for key, value in nested.items()
        )
    if "squad_ids" in manager_packet:
        assertions.append(("squad_ids", manager_packet.get("squad_ids"), "manager_packet.squad_ids"))

    failures: list[str] = []
    for field_name, supplied, label in assertions:
        if field_name not in canonical:
            failures.append(f"{label} is not a declared manager-state assertion")
            continue
        try:
            normalized = _normalize_manager_assertion(field_name, supplied)
        except (TypeError, ValueError) as failure:
            failures.append(f"{label} is invalid: {failure}")
            continue
        expected = canonical[field_name]
        if normalized != expected:
            failures.append(
                f"{label} does not match the pinned snapshot state ({normalized!r} vs {expected!r})"
            )
    if failures:
        raise GenerationRefused(DIAG_MANAGER_STATE_MISMATCH, failures)


def _required_manager_integer(value: Any, field_name: str, *, allow_none: bool = False) -> int | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise GenerationRefused(
            DIAG_MANAGER_STATE_EVIDENCE_MISSING,
            [f"the pinned manager context has no valid integer {field_name} value"],
        )
    if value < 0:
        raise GenerationRefused(
            DIAG_MANAGER_STATE_EVIDENCE_MISSING,
            [f"the pinned manager context {field_name} value is negative"],
        )
    return int(value)


def _normalize_manager_assertion(field_name: str, value: Any) -> Any:
    if field_name == "squad_ids":
        if not isinstance(value, (list, tuple)):
            raise TypeError("must be a list of player ids")
        if any(isinstance(pid, bool) or not isinstance(pid, int) for pid in value):
            raise TypeError("player ids must be integers")
        return sorted(int(pid) for pid in value)
    if field_name in {"bank_tenths", "free_transfers", "event_start_free_transfers"}:
        if value is None and field_name == "event_start_free_transfers":
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("must be an integer (zero is a value, not absence)")
        if value < 0:
            raise ValueError("must not be negative")
        return int(value)
    if field_name == "authoritative_source":
        if value is not None and not isinstance(value, str):
            raise TypeError("must be a string or null")
        return value
    return value


def _derive_four_gw_consumed_manager_state(
    source_conn: sqlite3.Connection,
    *,
    generation: CertifiedGeneration,
    manager_packet: Mapping[str, Any],
) -> dict[str, Any]:
    from . import manager_worlds, route_comparator
    from .planning import get_planning_context

    entry_id = int(manager_packet.get("entry_id") or 0)
    if entry_id <= 0:
        raise GenerationRefused(
            DIAG_MANAGER_STATE_EVIDENCE_MISSING,
            ["the manager packet has no usable entry_id for snapshot state resolution"],
        )
    context = get_planning_context(
        source_conn,
        entry_id,
        int(generation.planning_event),
        as_of=str(generation.cutoff),
        season=manager_packet.get("season"),
    )
    squad = manager_worlds.resolve_squad(context, source_conn)
    route_state = route_comparator.build_route_state(source_conn, context, squad)
    return canonical_four_gw_manager_state(
        entry_id=entry_id,
        planning_event=int(generation.planning_event),
        cutoff=str(generation.cutoff),
        season=manager_packet.get("season"),
        context=context,
        squad=squad,
        route_state=route_state,
    )


def _rederive_four_gw_consumed_manager_state(
    generation: CertifiedGeneration, manager_packet: Mapping[str, Any]
) -> dict[str, Any]:
    source_conn = _open_generation_snapshot(generation)
    try:
        state = _derive_four_gw_consumed_manager_state(
            source_conn, generation=generation, manager_packet=manager_packet
        )
        assert_manager_packet_matches_canonical_state(manager_packet, state)
        return state
    finally:
        source_conn.close()


def _manager_world_decision_executor(
    *,
    conn: sqlite3.Connection,
    generation: CertifiedGeneration,
    manager_packet: Mapping[str, Any],
    parameters: Mapping[str, Any],
    source_conn: sqlite3.Connection,
    controller: Any = None,
) -> dict[str, Any]:
    """The manager-world decision: the certified fix-15 lineup, from one generation.

    This pipeline lives in the library because it is assembled entirely from
    existing frozen functions: :func:`manager_worlds.build_manager_worlds` (which
    reads the certified run ids from the generation and refuses without one) and the
    Phase-6A lineup ranking.  No predictive input is taken from the caller.
    """

    from . import manager_lineup as ml
    from . import manager_worlds as mw

    squad_ids = [int(pid) for pid in (manager_state_from_packet(manager_packet).get("squad_ids") or [])]
    if len(squad_ids) != 15:
        raise DecisionRecordInvalid(
            [
                "a manager-world decision needs the manager's 15 player ids in the manager packet "
                f"(the squad is manager state); the packet names {len(squad_ids)}"
            ]
        )
    built = mw.build_manager_worlds(
        conn,
        generation=generation,
        planning_event=int(generation.planning_event),
        squad_ids=squad_ids,
        simulations=int(parameters.get("simulations", 10_000)),
        seed=int(parameters.get("seed", 20260911)),
        occupancy_audit=bool(parameters.get("occupancy_audit", True)),
    )
    positions, names = _squad_identity(source_conn, squad_ids)
    ranked = ml.rank_policies(squad_ids, positions, built["world_matrix"], top_k=int(parameters.get("top_k", 50)))
    top_rows = [
        {
            **policy.as_dict(names),
            **{
                key: value
                for key, value in ml.evaluate_policy(policy, built["world_matrix"], positions).items()
            },
        }
        for policy in ranked["top_policies"]
    ]
    return {
        "decision": {
            "status": "MANAGER_WORLD_EVALUATED",
            "planning_event": int(generation.planning_event),
            "top_policies": top_rows,
            "skeleton_count": ranked["skeletons"],
            "evaluated_policy_count": ranked["evaluated_policies"],
            "no_execution": True,
        },
        "world_info": built.get("simulation"),
        "manager_world": {
            "generation_id": generation.generation_id,
            "input_run_ids": built.get("input_run_ids"),
            "squad_ids": squad_ids,
            "simulation": built.get("simulation"),
        },
        "runner_identity": f"fpl_brain.generation_store:{mw.MANAGER_WORLDS_VERSION}",
    }


def _squad_identity(
    source_conn: sqlite3.Connection, squad_ids: Sequence[int]
) -> tuple[dict[int, str], dict[int, str]]:
    """The squad's positions and names, read from the PINNED snapshot.

    The caller supplied the squad (manager state); what each player IS -- position,
    club, display name -- comes from the snapshot the generation pins, so a packet
    cannot assert a position the certified source disagrees with.
    """

    from .manager_worlds import POSITION_IDS

    rows = source_conn.execute(
        "SELECT id, element_type, web_name, full_name FROM players WHERE id IN (%s)"
        % ",".join("?" for _ in squad_ids),
        [int(pid) for pid in squad_ids],
    ).fetchall()
    found = {int(row["id"]): dict(row) for row in rows}
    positions: dict[int, str] = {}
    names: dict[int, str] = {}
    missing: list[int] = []
    for pid in squad_ids:
        row = found.get(int(pid))
        position = POSITION_IDS.get(int(row["element_type"])) if row and row["element_type"] is not None else None
        if not position:
            missing.append(int(pid))
            continue
        positions[int(pid)] = str(position)
        names[int(pid)] = str(
            (row.get("full_name") or row.get("web_name") or f"Player {int(pid)}")
        )
    if missing:
        raise DecisionRecordInvalid(
            [
                f"the pinned snapshot does not resolve a position for {missing}; a manager-world "
                "decision is taken against the snapshot the generation pins"
            ]
        )
    return positions, names


def _decision_executor(profile: DecisionProfile) -> Callable[..., Mapping[str, Any]]:
    """The declared executor for a profile kind.  Never a caller-supplied callable."""

    if profile.kind == HORIZON_KIND_MANAGER_WORLD:
        return _manager_world_decision_executor
    return _declared_production_entrypoint(PRODUCTION_DECISION_MODULES[profile.kind])


def _decision_artifact_dir(conn: sqlite3.Connection, planning_event: int) -> Path:
    """Where a decision's artifact is RETAINED: beside the authoritative store.

    The path is derived from the store itself, so a decision record's artifact
    reference points at evidence the store owns; no caller-supplied directory can
    move it, and no caller-supplied path can stand in for it.
    """

    location = ":memory:"
    try:
        for row in conn.execute("PRAGMA database_list"):
            if str(row[1]) == "main":
                location = str(row[2]) or location
    except sqlite3.Error:  # pragma: no cover - defensive
        pass
    if location and location != ":memory:":
        base = Path(location).resolve().parent / "pe9_decisions"
    else:
        import tempfile

        base = Path(tempfile.gettempdir()) / "fpl_pe9_decisions"
    return base / f"gw{int(planning_event):02d}"


def _decision_artifact_bytes(artifact: Mapping[str, Any]) -> bytes:
    """The EXACT bytes a decision artifact is hashed as and retained as."""

    return (json.dumps(artifact, indent=2, sort_keys=True, default=str) + "\n").encode("utf-8")


def _confirm_retained_artifact(path: Path, payload: bytes) -> str:
    """Confirm the artifact already retained at ``path`` IS these bytes, or refuse."""

    retained = path.read_bytes()
    if retained != payload:
        raise DecisionRecordInvalid(
            [
                f"a decision artifact is already retained at {str(path)!r} with different bytes; "
                "a retained decision artifact is never replaced, so this decision cannot be "
                "recorded over it"
            ]
        )
    return str(path)


def _retain_decision_artifact(path: Path, payload: bytes) -> str:
    """RETAIN a decision artifact: publish it once, atomically, and never replace one.

    The name is content-addressed by ``payload`` itself, so two executions that produce
    the SAME bytes share one retained file (a repeated identical decision stays
    idempotent) while two executions that produce DIFFERENT bytes -- the reproducible
    decision whose volatile telemetry differs while its result digest does not -- each
    keep their own file instead of overwriting the evidence an earlier record still
    binds.  Publication is ATOMIC and NO-REPLACE, with no state exposed that is not
    complete: the payload is staged (written, flushed, fsynced) in a private temporary
    inside this function's cleanup protection, then hard-linked into place, which
    either creates the final name or fails without touching it.  A filesystem that
    cannot hard-link is therefore a REFUSAL, never a streamed write into the final
    path, and an occupied path is CONFIRMED or refused, never replaced.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex[:8]}")
    try:
        try:
            with open(temporary, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as failure:
            # A staging failure must never surface as a bare OSError: the caller refused
            # this decision, and the finally below discards the staged temporary.
            raise DecisionRecordInvalid(
                [
                    f"the store could not stage a decision artifact before publication "
                    f"({failure}); the staged temporary is discarded and nothing is retained"
                ]
            ) from failure
        try:
            os.link(temporary, path)
        except FileExistsError:
            # Some other execution got there first: reuse it only if it IS these bytes.
            return _confirm_retained_artifact(path, payload)
        except OSError as failure:
            # No atomic no-replace publication is available, so there is nothing safe
            # to do: refuse, leaving the final path untouched, rather than stream into
            # it and expose an incomplete artifact under its retained name.
            raise DecisionRecordInvalid(
                [
                    "the store cannot publish a decision artifact atomically on this filesystem "
                    f"(hard links are unavailable: {failure}); the staged artifact is discarded "
                    "and the final path is left untouched -- a decision artifact is never "
                    "written in place piece by piece"
                ]
            ) from failure
    finally:
        temporary.unlink(missing_ok=True)
    return str(path)


def make_decision(
    conn: sqlite3.Connection,
    manager_packet: Mapping[str, Any],
    planning_event: int,
    *,
    horizon_kind: str = HORIZON_KIND_FOUR_GW,
    generation_id: str | None = None,
    profile: Any = None,
    request: Mapping[str, Any] | None = None,
    controller: Any = None,
    **descriptors: Any,
) -> dict[str, Any]:
    """THE canonical descriptor-only production entrypoint.

        make_decision(manager_packet, planning_event, generation_id=None)

    ``generation_id=None`` resolves the ``current_generation`` pointer; an explicit
    ``generation_id`` loads that exact certified historical generation.
    ``generation_id`` is a SELECTOR, not a capability.

    The lifecycle (amendment 2 §11): resolve the generation id; load the canonical
    manifest; recompute the digest and require it to equal the id; verify
    event/horizon compatibility and the exact cutoff and the pinned snapshot
    identity; open the snapshot through the existing read-only/snapshot validation;
    re-run the existing per-event bundle validation against the pinned runs; verify
    the required versions and dependencies; build the worlds internally; execute the
    frozen decision logic; persist the decision artifact and append the
    ``engine_decision_record``; and return the decision with its record id.

    There is deliberately NO executor parameter: the pipeline is selected by the
    profile's declared KIND, so a caller cannot inject decision logic.  The manager
    packet carries manager identity and optional state assertions; for four-GW the
    canonical economic state is derived from the pinned snapshot.  The request
    carries ordinary decision parameters -- predictive evidence, cache handles and
    certification objects are refused wherever they appear, at any depth.

    The decision stays pinned to the generation selected at the START even if
    ``current_generation`` changes concurrently: the id is resolved once, and every
    later read uses that value.
    """

    assert_descriptor_only(descriptors)
    packet = dict(manager_packet or {})
    assert_no_nested_descriptors(packet, boundary="manager_packet")
    allowed_packet_fields = {
        "entry_id", "planning_event", "cutoff", "season", "manager_state", "squad_ids"
    }
    unknown_packet_fields = sorted(set(str(key) for key in packet) - allowed_packet_fields)
    if unknown_packet_fields:
        raise ProductionDescriptorOnly(
            [f"manager_packet has undeclared field(s): {unknown_packet_fields}"]
        )
    manager_state = packet.get("manager_state")
    if "manager_state" in packet and not isinstance(manager_state, Mapping):
        raise ProductionDescriptorOnly(
            ["manager_packet.manager_state must be an object when supplied"]
        )
    if isinstance(manager_state, Mapping):
        allowed_manager_fields = {
            "squad_ids", "bank_tenths", "free_transfers", "event_start_free_transfers",
            "authoritative_source",
        }
        unknown_manager_fields = sorted(
            set(str(key) for key in manager_state) - allowed_manager_fields
        )
        if unknown_manager_fields:
            raise ProductionDescriptorOnly(
                [f"manager_packet.manager_state has undeclared field(s): {unknown_manager_fields}"]
            )
    requested = dict(request or {})
    assert_no_nested_descriptors(requested, boundary="request")
    unknown_request_fields = sorted(set(str(key) for key in requested) - {"tracing_id"})
    if unknown_request_fields:
        raise ProductionDescriptorOnly(
            [f"request accepts only a tracing_id; received {unknown_request_fields}"]
        )
    resolved_profile = resolve_decision_profile(profile)
    if str(resolved_profile.kind) != str(horizon_kind):
        raise ProductionDescriptorOnly(
            [
                f"the decision profile kind {resolved_profile.kind!r} does not describe a "
                f"{horizon_kind} generation; one decision consumes one horizon kind"
            ]
        )

    generation = resolve_generation(
        conn,
        planning_event=int(planning_event),
        horizon_kind=horizon_kind,
        generation_id=generation_id,
    )
    # Digest re-verified (load_generation) and event/horizon compatibility checked
    # (resolve_generation).  The decision's cutoff must be EXACTLY the certified one;
    # a caller may restate it, but it may not move it.
    declared_cutoff = packet.get("cutoff")
    if declared_cutoff is not None:
        assert_cutoff_matches(conn, generation, cutoff=str(declared_cutoff))
    generation_report = verify_generation(conn, generation.generation_id)
    if not generation_report.get("verified"):
        raise GenerationRefused(
            DIAG_GENERATION_NOT_CERTIFIED,
            [f"generation {generation.generation_id} did not pass independent verification"],
        )
    require_snapshot_retained(generation)

    from . import search_permission as sp

    try:
        search_permission_evaluation = sp.require_search_permission(
            conn, generation.generation_id
        )
    except sp.SearchPermissionRefused as failure:
        raise GenerationRefused(
            DIAG_PRODUCTION_SEARCH_PERMISSION_DENIED, failure.reasons
        ) from failure

    runner_code_identity = _decision_runner_code_identity(resolved_profile)
    executor = _decision_executor(resolved_profile)
    source_conn = _open_generation_snapshot(generation)
    try:
        executor_arguments = {
            "conn": conn,
            "generation": generation,
            "manager_packet": packet,
            "parameters": dict(resolved_profile.parameters),
            "source_conn": source_conn,
            "controller": controller,
        }
        canonical_manager_state = None
        if resolved_profile.kind == HORIZON_KIND_FOUR_GW:
            canonical_manager_state = _derive_four_gw_consumed_manager_state(
                source_conn, generation=generation, manager_packet=packet
            )
            assert_manager_packet_matches_canonical_state(packet, canonical_manager_state)
            executor_arguments["canonical_manager_state"] = canonical_manager_state
        decision_result = executor(**executor_arguments)
    finally:
        source_conn.close()
    if _decision_runner_code_identity(resolved_profile) != runner_code_identity:
        raise DecisionRecordInvalid(
            ["the declared decision code changed while the production runner was executing"]
        )
    if not isinstance(decision_result, Mapping) or "decision" not in decision_result:
        raise DecisionRecordInvalid(
            [
                "the decision pipeline did not return a decision payload; a production decision "
                "that cannot be attributed is never recorded"
            ]
        )

    packet_digest = packet_identity(packet)
    request_digest = request_identity(
        planning_event=int(planning_event),
        horizon_kind=horizon_kind,
        cutoff=generation.cutoff,
        request=requested,
    )
    runner_identity = str(decision_result.get("runner_identity") or _default_runner_identity())
    result_provenance = dict(decision_result.get("provenance") or {})
    if resolved_profile.kind == HORIZON_KIND_FOUR_GW:
        returned_state = decision_result.get("consumed_manager_state")
        if not isinstance(returned_state, Mapping) or _json_round_trip_form(dict(returned_state)) != canonical_manager_state:
            raise DecisionRecordInvalid(
                [
                    "the four-GW runner did not return the exact manager state the production "
                    "boundary derived from the pinned snapshot"
                ]
            )
        manager_context_sha256 = four_gw_manager_context_identity(canonical_manager_state)
        reported_context_sha256 = str(result_provenance.get("planning_context_hash") or "")
        if reported_context_sha256 != manager_context_sha256:
            raise DecisionRecordInvalid(
                [
                    "the four-GW runner's manager-context digest does not bind the canonical "
                    "manager state derived from the pinned snapshot"
                ]
            )
    else:
        manager_context_sha256 = _fallback_manager_context_identity(
            packet, planning_event=int(planning_event), cutoff=generation.cutoff
        )
    # The pipeline's own artwork is PUBLISHED beside the canonical identity fields:
    # the operator-facing artifact keeps every block the production pipeline produced
    # (search configuration, refinement, scheduling, screening) while the
    # identity-bearing fields below are set HERE from the certified generation, so an
    # artifact cannot describe a predictive world the generation does not.
    artifact: dict[str, Any] = {
        key: value
        for key, value in dict(decision_result.get("artifact_blocks") or {}).items()
        if key not in {"attribution", "provenance"}
    }
    artifact.update({
        "schema": (
            "fpl_brain.four_gw_decision.v1"
            if str(generation.horizon_kind) == HORIZON_KIND_FOUR_GW
            else "fpl_brain.manager_world_decision.v1"
        ),
        "planning_event": int(generation.planning_event),
        "planning_cutoff": generation.cutoff,
        "decision_events": list(generation.events),
        # The attribution block keeps the caller's manager packet and request as
        # assertions/identity.  The canonical four-GW state actually consumed is
        # stored separately below and re-derived from the pinned snapshot by
        # ``verify_decision`` before its digest is reported as verified.
        "attribution": {
            "manager_packet": packet,
            "request": requested,
            "manager_packet_sha256": packet_digest,
            "request_sha256": request_digest,
            "manager_context_sha256": manager_context_sha256,
            "decision_profile": resolved_profile.as_dict(),
        },
        "runner_identity": runner_identity,
        "runner_code_identity": runner_code_identity,
        "manager_context_sha256": manager_context_sha256,
        "provenance": {
            **dict(decision_result.get("provenance") or {}),
            "generation_id": generation.generation_id,
            "generation_horizon_kind": generation.horizon_kind,
            "manager_packet_sha256": packet_digest,
            "request_sha256": request_digest,
            "snapshot": {
                "path": generation.snapshot.get("path"),
                "sha256": generation.snapshot.get("sha256"),
                "source_db_identity": generation.snapshot.get("source_db_identity"),
                "execution_run_uuid": generation.snapshot.get("execution_run_uuid"),
            },
            "certified_runs_by_event": {
                str(event): generation.runs_for(int(event)) for event in generation.events
            },
            "generation_manifest": dict(generation.manifest),
            "pe8_evidence": generation.manifest.get("pe8_evidence"),
            "disclosure": generation.manifest.get("disclosure"),
            "search_permission_evaluation": search_permission_evaluation,
        },
        "decision": decision_result.get("decision"),
        "screened_actions": decision_result.get("screened_actions"),
        "finalist_refinement": decision_result.get("finalist_refinement"),
        "decision_confidence": decision_result.get("decision_confidence"),
        "fixture_horizon": decision_result.get("fixture_horizon"),
        "suppression_reasons": decision_result.get("suppression_reasons") or [],
        "world_info": decision_result.get("world_info"),
        "no_execution": True,
    })
    if canonical_manager_state is not None:
        artifact["attribution"]["consumed_manager_state"] = canonical_manager_state
    result_digest = result_identity_of(artifact)
    artifact_bytes = _decision_artifact_bytes(artifact)
    # The RETAINED NAME carries both identities: the decision this artifact IS, and the
    # exact bytes retained for it.  The bytes are hashed from the same buffer that is
    # written, so the name cannot describe content other than what was retained, and a
    # repeat that reproduces the decision but not its volatile telemetry keeps a SEPARATE
    # artifact rather than overwriting the one an earlier record binds.
    artifact_path = _decision_artifact_dir(conn, int(generation.planning_event)) / (
        f"{result_digest.split(':', 1)[1][:32]}-{hashlib.sha256(artifact_bytes).hexdigest()}.json"
    )
    artifact_ref = _retain_decision_artifact(artifact_path, artifact_bytes)
    from . import execution_snapshot as es

    artifact_file_sha256 = es.file_sha256(artifact_ref)
    decision_id = append_engine_decision_record(
        conn,
        generation=generation,
        manager_packet_sha256=packet_digest,
        request_sha256=request_digest,
        result_sha256=result_digest,
        runner_identity=runner_identity,
        evidence={
            "schema": DECISION_RECORD_SCHEMA,
            "generation_id": generation.generation_id,
            "generation_manifest_verified": True,
            "runner_identity": runner_identity,
            "runner_code_identity": runner_code_identity,
            "manager_context_sha256": manager_context_sha256,
            "decision_artifact_file_sha256": artifact_file_sha256,
            "snapshot_identity_verified": True,
            "bundle_validation_rerun": True,
            "required_versions_verified": True,
            "dependency_closure_verified": True,
            "worlds_built_internally": True,
            "decision_profile": resolved_profile.as_dict(),
            "calibration_consulted": bool(
                (generation.manifest.get("pe8_evidence") or {}).get("consulted")
            ),
            "search_permission": sp.decision_record_binding(search_permission_evaluation),
        },
        decision_artifact_ref=artifact_ref,
    )
    conn.commit()
    return {
        "schema": "fpl_brain.pe9_decision_result.v1",
        "decision_record_id": decision_id,
        "generation_id": generation.generation_id,
        "planning_event": int(generation.planning_event),
        "planning_cutoff": generation.cutoff,
        "decision_events": list(generation.events),
        "decision": decision_result.get("decision"),
        "screened_actions": decision_result.get("screened_actions"),
        "finalist_refinement": decision_result.get("finalist_refinement"),
        "decision_confidence": decision_result.get("decision_confidence"),
        "fixture_horizon": decision_result.get("fixture_horizon"),
        "result_sha256": result_digest,
        "runner_identity": runner_identity,
        "runner_code_identity": runner_code_identity,
        "manager_context_sha256": manager_context_sha256,
        "decision_artifact_ref": artifact_ref,
        "artifact": artifact,
        "provenance": artifact["provenance"],
        "no_execution": True,
    }


def _default_runner_identity() -> str:
    import sys

    return f"{Path(sys.argv[0]).name or 'python'}:{MANIFEST_SCHEMA}"


def support_by_event(generation: CertifiedGeneration, *, cutoff: str | None = None) -> dict[int, dict[str, Any]]:
    """The per-event predictive support a certified generation PROVES.

    Reads the digest-verified manifest, so the run ids and the bundle identities are the
    certified ones rather than a rediscovery.  Shaped for
    :func:`four_gw_decision.evaluate_horizon`.
    """

    resolved_cutoff = str(cutoff if cutoff is not None else generation.cutoff)
    support: dict[int, dict[str, Any]] = {}
    for event in generation.events:
        record = (generation.manifest.get("per_event") or {}).get(str(int(event))) or {}
        state = str(record.get("state") or "")
        support[int(event)] = {
            "supported": state not in cb.BLOCKING_BUNDLE_STATES,
            "matched_runs": generation.runs_for(int(event)),
            "missing_families": [],
            "stale_families": [],
            "data_cutoff": resolved_cutoff,
            "run_cutoffs": [resolved_cutoff],
            "bundle_identity": record.get("bundle_identity"),
            "state": state,
            "source": "certified_generation",
            "generation_id": generation.generation_id,
        }
    return support


def open_generation_snapshot(generation: CertifiedGeneration) -> sqlite3.Connection:
    """Open the generation's pinned snapshot READ-ONLY, through the existing validator."""

    return _open_generation_snapshot(generation)


def _open_generation_snapshot(generation: CertifiedGeneration) -> sqlite3.Connection:
    """Open the generation's pinned snapshot READ-ONLY, through the existing validator."""

    from . import execution_snapshot as es

    require_snapshot_retained(generation)
    snapshot = generation.snapshot
    path = snapshot.get("path")
    if not path:
        raise GenerationSnapshotUnverified(
            [
                f"generation {generation.generation_id} pins no snapshot; a production decision "
                "requires the immutable causal source"
            ]
        )
    try:
        installed = es.ExecutionSnapshot(
            path=str(path),
            data_snapshot_sha256=str(snapshot.get("sha256")),
            snapshot_lock_acquired_at="",
            snapshot_capture_started_at="",
            snapshot_consistency_at="",
            snapshot_capture_completed_at="",
            snapshot_capture_seconds=0.0,
            source_db_identity=dict(snapshot.get("source_db_identity") or {}),
            execution_run_uuid=snapshot.get("execution_run_uuid"),
            planning_cutoff=generation.cutoff,
            size_bytes=int(snapshot.get("size_bytes") or Path(str(path)).stat().st_size),
        )
    except TypeError:  # pragma: no cover - defensive
        installed = None
    if installed is not None:
        return es.open_snapshot(installed)
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    return conn


__all__ = [
    "CertifiedGeneration",
    "DECISION_RECORD_SCHEMA",
    "DecisionProfile",
    "DIAG_DECISION_ARTIFACT_NOT_RETAINED",
    "DIAG_DECISION_RECORD_INVALID",
    "DIAG_DECISION_RECORD_UNKNOWN",
    "DIAG_GENERATION_PLANNING_CONTEXT_REQUIRED",
    "DIAG_PRODUCTION_PROFILE_UNKNOWN",
    "DIAG_GENERATION_DIGEST_MISMATCH",
    "DIAG_GENERATION_HORIZON_KIND_UNKNOWN",
    "DIAG_GENERATION_MANIFEST_INVALID",
    "DIAG_GENERATION_NOT_CERTIFIED",
    "DIAG_GENERATION_POINTER_UNSET",
    "DIAG_GENERATION_SNAPSHOT_UNVERIFIED",
    "DIAG_GENERATION_UNKNOWN",
    "DIAG_PRODUCTION_DESCRIPTOR_ONLY",
    "DecisionRecordInvalid",
    "FORBIDDEN_PRODUCTION_DESCRIPTORS",
    "GenerationManifestInvalid",
    "GenerationManifestMutated",
    "GenerationNotCertified",
    "GenerationRefused",
    "GenerationSnapshotUnverified",
    "GenerationUnknown",
    "HORIZON_KINDS",
    "HORIZON_LENGTHS",
    "HORIZON_KIND_FOUR_GW",
    "HORIZON_KIND_MANAGER_WORLD",
    "MANIFEST_SCHEMA",
    "PE8_IDENTITY_AND_STATE_RECOMPUTED",
    "PE8_NOT_CONSULTED",
    "PRODUCTION_DECISION_ENTRYPOINT",
    "PRODUCTION_DECISION_MODULES",
    "ProductionDescriptorOnly",
    "append_engine_decision_record",
    "assert_cutoff_matches",
    "assert_descriptor_only",
    "assert_generation_bundles_valid",
    "assert_no_nested_descriptors",
    "assert_writer_lease_held",
    "authoritative_code_identity",
    "build_generation_manifest",
    "canonical_identity_value",
    "canonical_manifest_bytes",
    "certify_generation",
    "current_generation_id",
    "decision_identity_of",
    "decision_result_projection",
    "generation_id_of",
    "load_engine_decision_record",
    "load_generation",
    "make_decision",
    "manager_state_from_packet",
    "manifest_semantic_projection",
    "packet_identity",
    "request_identity",
    "reproduce_pe8_evidence",
    "require_snapshot_retained",
    "resolve_decision_profile",
    "resolve_generation",
    "result_identity_of",
    "verify_decision",
    "verify_generation",
]
