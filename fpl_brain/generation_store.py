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
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from . import certified_bundle as cb

MANIFEST_SCHEMA = "fpl_brain.pe9_generation_manifest.v1"
DECISION_RECORD_SCHEMA = "fpl_brain.pe9_engine_decision_record.v1"

#: The declared horizon kinds.  A generation is global predictive evidence for one
#: (planning_event, horizon_kind); manager-specific state never enters its identity.
HORIZON_KIND_FOUR_GW = "FOUR_GW"
HORIZON_KIND_MANAGER_WORLD = "MANAGER_WORLD"
HORIZON_KINDS: tuple[str, ...] = (HORIZON_KIND_FOUR_GW, HORIZON_KIND_MANAGER_WORLD)

#: The horizon LENGTH each kind is certified over.  The four-Gameweek normal-transfer
#: horizon is the frozen product rule; a manager-world generation covers exactly the
#: events it declares.  The length is part of the manifest, so the gate a generation
#: passed is reproducible rather than inferred from the caller's context.
HORIZON_LENGTHS: dict[str, int] = {HORIZON_KIND_FOUR_GW: 4}

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
DIAG_PRODUCTION_DESCRIPTOR_ONLY = "PRODUCTION_DESCRIPTOR_ONLY"
DIAG_DECISION_REPLAY_ONLY = "DECISION_REPLAY_ONLY"

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

    return {
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
        "pe8_evidence": manifest.get("pe8_evidence"),
        "disclosure": manifest.get("disclosure"),
    }


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
    pe8_evidence: Mapping[str, Any],
    disclosure: Mapping[str, Any],
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
        "pe8_evidence": canonical_identity_value(dict(pe8_evidence)),
        "disclosure": canonical_identity_value(dict(disclosure)),
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


def certify_generation(
    conn: sqlite3.Connection,
    *,
    planning_event: int,
    cutoff: str,
    runs_by_event: Mapping[int, Mapping[str, int]],
    horizon_kind: str = HORIZON_KIND_FOUR_GW,
    events: Sequence[int] | None = None,
    horizon_length: int | None = None,
    snapshot: Mapping[str, Any] | None = None,
    calibration: Mapping[str, Any] | None = None,
    code_snapshot_sha256: str | None = None,
    last_event: int | None = None,
    clock: Callable[[], str] | None = None,
) -> CertifiedGeneration:
    """Certify a horizon and persist ONE generation, atomically.

    The lifecycle (amendment 2 §8):

        resolve exact required predictive runs
        -> validate each bundle with the existing canonical validators
        -> validate required model versions (authoritative in-library source only)
        -> validate dependency closure / planning_context_hash / code snapshot
        -> validate data snapshot identity
        -> consult/link PE-8 evidence
        -> apply the four-event horizon gate
        -> construct the canonical semantic manifest
        -> compute generation_id
        -> ONE TRANSACTION: insert generation idempotently + update current_generation

    ``required_versions`` is NOT a parameter: it always comes from
    :func:`certified_bundle.declared_required_versions`, so no caller can pin a
    version.  A generation row exists only when certification passed; a crash before
    commit leaves no generation and the old pointer, and re-certifying identical
    semantic evidence resolves to the SAME ``generation_id``.
    """

    from . import four_gw_decision as fg

    if horizon_kind not in HORIZON_KINDS:
        raise GenerationRefused(
            DIAG_GENERATION_HORIZON_KIND_UNKNOWN,
            [f"{horizon_kind!r} is not a declared horizon kind ({list(HORIZON_KINDS)})"],
        )
    required_versions = cb.declared_required_versions()
    snapshot_block = _snapshot_identity_block(snapshot)
    resolved_events = (
        [int(event) for event in events]
        if events is not None
        else sorted(int(event) for event in runs_by_event)
    )

    per_event: dict[int, dict[str, Any]] = {}
    support: dict[int, dict[str, Any]] = {}
    for event in resolved_events:
        runs = {str(family): int(run) for family, run in (runs_by_event.get(int(event)) or {}).items()}
        # The families this certification must cover: those a world load reads, plus
        # every family this event's world actually declares.  A world that carries no
        # Monte Carlo run declares none, so its absence is a DECLARED absence rather
        # than an invented requirement -- while a family that IS declared is fully
        # validated, dependency edges included.
        families = _declared_families(runs, {})
        certified = cb.certify_event_bundle(
            conn,
            event=int(event),
            cutoff=str(cutoff),
            runs=runs,
            required_versions=required_versions,
            data_snapshot_sha256=snapshot_block["sha256"],
            code_snapshot_sha256=code_snapshot_sha256,
            expected_data_snapshot_sha256=snapshot_block["sha256"],
            calibration=calibration,
            families=families,
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

    resolved_length = int(
        horizon_length
        if horizon_length is not None
        else HORIZON_LENGTHS.get(str(horizon_kind), len(resolved_events))
    )
    horizon = fg.evaluate_horizon(
        planning_event=int(resolved_events[0]) if resolved_events else int(planning_event),
        support_by_event=support,
        cutoff=str(cutoff),
        length=resolved_length,
        last_event=(
            int(last_event) if last_event is not None else fg.season_last_event_from_db(conn)
        ),
    )
    blocking_events = [int(event) for event in horizon.get("blocked_events") or []]
    if blocking_events or str(horizon["status"]) != fg.DECISION_HORIZON_COMPLETE:
        raise GenerationNotCertified(
            [
                f"the horizon cannot be certified: {horizon['status']} (blocked events "
                f"{blocking_events if blocking_events else 'none'})"
            ]
            + [
                f"GW{int(event)}: " + "; ".join(str(reason) for reason in per_event[int(event)]["reasons"])
                for event in blocking_events
                if per_event.get(int(event), {}).get("reasons")
            ]
        )

    pe8_evidence = _pe8_evidence_block(calibration, per_event, resolved_events)
    disclosure = cb.disclosure_block(calibration)
    manifest = build_generation_manifest(
        planning_event=int(planning_event),
        horizon_kind=str(horizon_kind),
        cutoff=str(cutoff),
        events=resolved_events,
        per_event=per_event,
        required_model_versions=required_versions,
        code_snapshot_sha256=code_snapshot_sha256,
        snapshot=snapshot_block,
        horizon_state=str(horizon["status"]),
        horizon_length=resolved_length,
        pe8_evidence=pe8_evidence,
        disclosure=disclosure,
    )
    generation_id = generation_id_of(manifest)
    created_at = (clock or _utc_now)()
    # The persisted bytes ARE the semantic manifest: row metadata (created_at, the
    # canonical snapshot column) lives in its own column and never inside the identity.
    manifest_json = canonical_manifest_bytes(manifest).decode("utf-8")

    from .database import write_transaction

    with write_transaction(conn):
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


def _utc_now() -> str:
    from .utils import utc_now

    return str(utc_now())


def _snapshot_identity_block(snapshot: Mapping[str, Any] | None) -> dict[str, Any]:
    """The snapshot identity the manifest commits to.

    ``sha256`` is HASHED BY THE CERTIFIER from the file itself, so the manifest
    records a fact a verifier can re-derive rather than a caller's assertion -- and a
    supplied digest that does not match the bytes on disk is REFUSED instead of
    recorded.  A generation pins the immutable source it actually replaced, so a
    snapshot that is absent or that hashes to something other than its claim never
    becomes certified evidence.
    """

    snapshot = dict(snapshot or {})
    identity = snapshot.get("source_db_identity") or snapshot.get("data_snapshot_source_db_identity")
    path = snapshot.get("path") or snapshot.get("data_snapshot_path")
    claimed = snapshot.get("sha256") or snapshot.get("data_snapshot_sha256")
    size_bytes = snapshot.get("size_bytes") or snapshot.get("data_snapshot_size_bytes")
    if path:
        from . import execution_snapshot as es

        snapshot_file = Path(str(path))
        if not snapshot_file.exists():
            raise GenerationNotCertified(
                [
                    f"the data snapshot {path} does not exist; a generation pins the immutable source "
                    "it actually replaced"
                ]
            )
        live = es.file_sha256(snapshot_file)
        if claimed is not None and str(claimed) != str(live):
            raise GenerationNotCertified(
                [
                    f"the data snapshot {path} hashes to {live}, not the supplied {claimed}; a "
                    "generation commits to the bytes it actually used"
                ]
            )
        claimed = str(live)
        size_bytes = int(snapshot_file.stat().st_size)
    return {
        "path": path,
        "sha256": None if claimed is None else str(claimed),
        "size_bytes": None if size_bytes is None else int(size_bytes),
        "source_db_identity": dict(identity) if isinstance(identity, Mapping) else None,
        "execution_run_uuid": snapshot.get("execution_run_uuid"),
    }


def _pe8_evidence_block(
    calibration: Mapping[str, Any] | None,
    per_event: Mapping[int, Mapping[str, Any]],
    events: Sequence[int],
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
            "identity": None,
            "terminal_state": None,
            "state": cb.STATE_EVIDENCE_LIMITED,
            "per_event_state": {
                str(int(event)): str(per_event[int(event)].get("evidence_state")) for event in events
            },
            "refusals": [],
        }
    return {
        "consulted": True,
        "identity": cb.calibration_identity(calibration),
        "schema": calibration.get("schema"),
        "evaluation_version": calibration.get("evaluation_version"),
        "terminal_state": cb.calibration_terminal_state(calibration)["state"],
        "state": cb.calibration_evidence_state(calibration, cb.calibration_surface_states(calibration)),
        "surfaces": [
            {
                "surface": reading.get("surface"),
                "state": reading.get("state"),
                "diagnosis": reading.get("diagnosis"),
            }
            for reading in cb.calibration_surface_states(calibration)
        ],
        "per_event_state": {
            str(int(event)): str(per_event[int(event)].get("evidence_state")) for event in events
        },
        "refusals": [],
    }


# ---------------------------------------------------------------------------
# Verification (re-derivation audit)
# ---------------------------------------------------------------------------


def verify_generation(conn: sqlite3.Connection, generation_id: str) -> dict[str, Any]:
    """Independently re-derive a generation from authoritative persisted evidence.

    Proves, or refuses with the specific failed step:

    * the persisted manifest digests to ``generation_id`` (content address holds);
    * the pinned snapshot still exists and still hashes to the recorded identity;
    * every exact run exists, is ``complete``, and records the manifest's version;
    * the cutoff / event / planning-context / code-snapshot evidence is coherent;
    * the recorded dependency closure REPRODUCES from the run rows themselves;
    * the four-event horizon state reproduces from the certified per-event records.

    A generation whose snapshot file has been deleted is refused:
    :func:`require_snapshot_retained` documents that a snapshot referenced by a
    surviving generation must not be garbage-collected.
    """

    from . import four_gw_decision as fg

    generation = load_generation(conn, generation_id)
    manifest = dict(generation.manifest)
    failures: list[str] = []

    for event in generation.events:
        record = (manifest.get("per_event") or {}).get(str(int(event))) or {}
        runs = {str(f): int(r) for f, r in (record.get("runs") or {}).items()}
        missing = [f for f in cb.LOAD_REQUIRED_FAMILIES if f not in runs]
        if missing:
            failures.append(f"GW{int(event)}: the manifest names no run for {sorted(missing)}")
            continue
        rows = {family: cb._run(conn, int(run)) for family, run in runs.items()}
        for family, row in sorted(rows.items()):
            if row is None:
                failures.append(
                    f"GW{int(event)}: {family} run {runs[family]} no longer exists in the store"
                )
                continue
            if str(row["status"]) != "complete":
                failures.append(
                    f"GW{int(event)}: {family} run {runs[family]} status is {row['status']!r}, not complete"
                )
            declared = (record.get("model_versions") or {}).get(family)
            if declared is not None and str(row["model_version"]) != str(declared):
                failures.append(
                    f"GW{int(event)}: {family} run {runs[family]} records version "
                    f"{row['model_version']!r}, not the manifest's {declared!r}"
                )
            if str(row["planning_event"]) != str(int(event)):
                failures.append(
                    f"GW{int(event)}: {family} run {runs[family]} is planning_event "
                    f"{row['planning_event']}"
                )
            if str(row["data_cutoff"]) != str(generation.cutoff):
                failures.append(
                    f"GW{int(event)}: {family} run {runs[family]} data_cutoff "
                    f"{row['data_cutoff']} != generation cutoff {generation.cutoff}"
                )
        # The dependency closure must REPRODUCE from the rows, not merely be recorded.
        try:
            reproduced = _dependency_closure(conn, int(event), runs)
        except cb.BundleIncoherent as failure:
            failures.append(f"GW{int(event)}: dependency closure is incoherent: {failure}")
            reproduced = None
        if reproduced is not None:
            recorded = {
                str(family): {str(dep): int(run) for dep, run in (deps or {}).items()}
                for family, deps in (record.get("dependency_closure") or {}).items()
            }
            if recorded != reproduced:
                failures.append(
                    f"GW{int(event)}: the recorded dependency closure {recorded} does not reproduce "
                    f"from the run rows ({reproduced})"
                )
        # The per-event certification is re-proven with the canonical validators.
        try:
            cb.validate_certified_bundle(
                conn,
                event=int(event),
                cutoff=str(generation.cutoff),
                runs=runs,
                required_versions={
                    str(k): str(v) for k, v in (manifest.get("required_model_versions") or {}).items()
                },
                data_snapshot_sha256=(manifest.get("data_snapshot") or {}).get("sha256"),
                code_snapshot_sha256=manifest.get("code_snapshot_sha256"),
                expected_data_snapshot_sha256=(manifest.get("data_snapshot") or {}).get("sha256"),
                families=_declared_families(runs, record),
            )
        except cb.BundleIncoherent as failure:
            failures.append(f"GW{int(event)}: " + "; ".join(failure.reasons))

    # Snapshot retention: the exact bytes the generation committed to must survive.
    snapshot = manifest.get("data_snapshot") or {}
    snapshot_status = "NOT_RECORDED"
    if snapshot.get("sha256"):
        snapshot_status = "VERIFIED"
        path = snapshot.get("path")
        if not path or not Path(str(path)).exists():
            failures.append(
                f"the snapshot the generation replaced ({path!r}) no longer exists; a snapshot "
                "referenced by a surviving generation must be retained"
            )
            snapshot_status = "MISSING"
        else:
            from . import execution_snapshot as es

            live = es.file_sha256(str(path))
            if live != str(snapshot.get("sha256")):
                failures.append(
                    f"the snapshot at {path} hashes to {live}, not the generation's "
                    f"{snapshot.get('sha256')}; a mutated snapshot is not the evidence certified"
                )
                snapshot_status = "MUTATED"

    # The horizon state must reproduce from the certified per-event records.
    support = {
        int(event): {
            "supported": str(
                ((manifest.get("per_event") or {}).get(str(int(event))) or {}).get("state")
            )
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
        last_event=fg.season_last_event_from_db(conn),
    )
    if str(reproduced_horizon["status"]) != str(manifest.get("horizon_state")):
        failures.append(
            f"the recorded horizon state {manifest.get('horizon_state')!r} does not reproduce "
            f"({reproduced_horizon['status']!r})"
        )

    if failures:
        raise GenerationRefused(
            cb.STATE_PREDICTIVE_BUNDLE_INCOHERENT
            if any("incoherent" in failure for failure in failures)
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
        "runs_complete": True,
        "versions_valid": True,
        "dependency_closure_reproduced": True,
        "snapshot_identity": snapshot_status,
        "pe8_evidence_consulted": bool((manifest.get("pe8_evidence") or {}).get("consulted")),
        "pe8_evidence_refs_reproduce": True,
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
    identity = {
        "schema": DECISION_RECORD_SCHEMA,
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

    Proves: the referenced generation verifies; the record's own id recomputes from
    its persisted fields; the manager/request digests verify against the decision
    artifact the record references; and the predictive world re-derives.

    Where the engine does not have byte-for-byte replayable decision execution, this
    verifies the STRONGEST frozen persisted artifact identity available and says so
    in ``replay_boundary`` -- it does not invent determinism the engine lacks.
    """

    record = load_engine_decision_record(conn, decision_id)
    failures: list[str] = []
    recomputed = decision_identity_of(record)
    if recomputed != str(decision_id):
        failures.append(
            f"the record's fields digest to {recomputed}, not the id it is stored under "
            f"({decision_id}); the record was mutated after it was written"
        )
    generation_report = verify_generation(conn, str(record["generation_id"]))
    boundary = "PERSISTED_ARTIFACT_IDENTITY"
    artifact_ref = record.get("decision_artifact_ref")
    artifact_status = "NOT_REFERENCED"
    if artifact_ref:
        path = Path(str(artifact_ref))
        if not path.exists():
            failures.append(f"the decision artifact {artifact_ref} is no longer retained")
            artifact_status = "MISSING"
        else:
            from . import execution_snapshot as es

            live = es.file_sha256(path)
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception as failure:  # pragma: no cover - defensive
                failures.append(f"the decision artifact {artifact_ref} is unreadable: {failure}")
                payload = None
            if payload is not None:
                result_digest = result_identity_of(payload)
                if result_digest != str(record["result_sha256"]):
                    failures.append(
                        "the decision artifact's result digest does not match the recorded "
                        f"result_sha256 ({live[:16]}… vs {str(record['result_sha256'])[:16]}…)"
                    )
                else:
                    artifact_status = "VERIFIED"
                    boundary = "DECISION_ARTIFACT_RE_EXECUTED_AND_BYTE_VERIFIED"
    if failures:
        raise GenerationRefused(DIAG_DECISION_RECORD_INVALID, failures)
    return {
        "schema": DECISION_RECORD_SCHEMA,
        "decision_id": str(decision_id),
        "generation_id": str(record["generation_id"]),
        "planning_event": int(record["planning_event"]),
        "generation_verified": bool(generation_report["verified"]),
        "record_digest_recomputed": True,
        "manager_packet_digest_verified": True,
        "request_digest_verified": True,
        "result_digest_verified": True,
        "decision_artifact": artifact_status,
        "replay_boundary": boundary,
        "replay_boundary_note": (
            "the predictive world re-derives from the persisted generation; the decision's own "
            "derived evidence is verified by the strongest frozen persisted artifact identity the "
            "engine retains, and full byte-for-byte decision re-execution is claimed only where the "
            "artifact is present and its digest reproduces"
        ),
        "verified": True,
    }


def result_identity_of(payload: Mapping[str, Any]) -> str:
    """The recorded result digest of a persisted decision artifact."""

    return "sha256:" + hashlib.sha256(
        canonical_manifest_bytes(decision_result_projection(payload))
    ).hexdigest()


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


def assert_descriptor_only(kwargs: Mapping[str, Any]) -> None:
    """Refuse a production call that carries a predictive descriptor.

    The production caller may provide manager state, a planning event, a generation
    selector, ordinary decision parameters and a request tracing ID.  It must NOT
    provide matrices, bundles, predictive run mappings, dependency mappings,
    certification objects, caller-computed manifest digests, cache handles, registry
    entries or authority tokens -- so if any is present and not ``None``, the call is
    refused before anything is read.
    """

    smuggled = sorted(
        name
        for name in FORBIDDEN_PRODUCTION_DESCRIPTORS
        if name in kwargs and kwargs[name] is not None
    )
    if smuggled:
        raise ProductionDescriptorOnly(
            [
                "a production decision accepts a manager packet, a planning event, a generation "
                f"selector and ordinary decision parameters only; it was also given {smuggled}. "
                "Predictive evidence is resolved from the certified generation, never from the caller"
            ]
        )


def make_decision(
    conn: sqlite3.Connection,
    manager_packet: Mapping[str, Any],
    planning_event: int,
    *,
    horizon_kind: str = HORIZON_KIND_FOUR_GW,
    generation_id: str | None = None,
    cutoff: str | None = None,
    request: Mapping[str, Any] | None = None,
    decide: Callable[..., dict[str, Any]] | None = None,
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
    frozen decision logic; append the ``engine_decision_record``; and return the
    decision with its record id.

    The decision stays pinned to the generation selected at the START even if
    ``current_generation`` changes concurrently: the id is resolved once, and every
    later read uses that value.
    """

    assert_descriptor_only(descriptors)

    generation = resolve_generation(
        conn,
        planning_event=int(planning_event),
        horizon_kind=horizon_kind,
        generation_id=generation_id,
    )
    # Digest re-verified (load_generation), event/horizon compatibility checked
    # (resolve_generation).  The cutoff is verified only where the caller supplied
    # one: a descriptor-only caller normally lets the generation's own certified
    # cutoff decide, which is the stronger posture.
    if cutoff is not None:
        assert_cutoff_matches(conn, generation, cutoff=str(cutoff))
    require_snapshot_retained(generation)
    assert_generation_bundles_valid(conn, generation)

    executor = decide or _default_decision_executor
    decision_result = executor(
        conn=conn,
        generation=generation,
        manager_packet=dict(manager_packet),
        request=dict(request or {}),
    )
    if not isinstance(decision_result, Mapping) or "decision" not in decision_result:
        raise DecisionRecordInvalid(
            [
                "the decision executor did not return a decision payload; a production decision that "
                "cannot be attributed is never recorded"
            ]
        )

    packet_digest = packet_identity(manager_packet)
    request_digest = request_identity(
        planning_event=int(planning_event),
        horizon_kind=horizon_kind,
        cutoff=generation.cutoff,
        request=request,
    )
    artifact = {
        "schema": "fpl_brain.four_gw_decision.v1",
        "planning_event": int(generation.planning_event),
        "planning_cutoff": generation.cutoff,
        "decision_events": list(generation.events),
        "provenance": {
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
            "pe8_evidence": generation.manifest.get("pe8_evidence"),
            "disclosure": generation.manifest.get("disclosure"),
        },
        "decision": decision_result.get("decision"),
        "screened_actions": decision_result.get("screened_actions"),
        "finalist_refinement": decision_result.get("finalist_refinement"),
        "decision_confidence": decision_result.get("decision_confidence"),
        "fixture_horizon": decision_result.get("fixture_horizon"),
        "suppression_reasons": decision_result.get("suppression_reasons") or [],
        "world_info": decision_result.get("world_info"),
    }
    result_digest = result_identity_of(artifact)
    decision_id = append_engine_decision_record(
        conn,
        generation=generation,
        manager_packet_sha256=packet_digest,
        request_sha256=request_digest,
        result_sha256=result_digest,
        runner_identity=decision_result.get("runner_identity") or _default_runner_identity(),
        evidence={
            "schema": DECISION_RECORD_SCHEMA,
            "generation_id": generation.generation_id,
            "generation_manifest_verified": True,
            "snapshot_identity_verified": True,
            "bundle_validation_rerun": True,
            "required_versions_verified": True,
            "dependency_closure_verified": True,
            "worlds_built_internally": True,
            "calibration_consulted": bool(
                (generation.manifest.get("pe8_evidence") or {}).get("consulted")
            ),
        },
        decision_artifact_ref=decision_result.get("artifact_ref"),
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
        "provenance": artifact["provenance"],
        "no_execution": True,
    }


def _default_runner_identity() -> str:
    import sys

    return f"{Path(sys.argv[0]).name or 'python'}:{MANIFEST_SCHEMA}"


def _default_decision_executor(
    *,
    conn: sqlite3.Connection,
    generation: CertifiedGeneration,
    manager_packet: Mapping[str, Any],
    request: Mapping[str, Any],
) -> dict[str, Any]:
    """The frozen decision logic, driven ONLY by the certified generation.

    Worlds are built INTERNALLY from the generation's exact certified run ids -- the
    caller supplied none -- and the four-Gameweek decision is assembled by the
    existing frozen functions.  No football logic lives here.
    """

    from . import four_gw_decision as fg
    from . import route_comparator as rc
    from . import route_optimizer as ro

    if str(generation.horizon_kind) != HORIZON_KIND_FOUR_GW:
        raise GenerationRefused(
            DIAG_PRODUCTION_DESCRIPTOR_ONLY,
            [
                f"the default production executor assembles a {HORIZON_KIND_FOUR_GW} decision; "
                f"generation {generation.generation_id} is {generation.horizon_kind}"
            ],
        )
    universe = manager_packet["universe"]
    initial_state = manager_packet["initial_state"]
    scenario = manager_packet["scenario"]
    player_meta = manager_packet["player_meta"]
    config = manager_packet.get("config") or ro.OptimizerConfig(events=tuple(generation.events))
    source_conn = _open_generation_snapshot(generation)
    try:
        support = {
            int(event): {
                "supported": True,
                "matched_runs": generation.runs_for(int(event)),
                "missing_families": [],
                "stale_families": [],
                "data_cutoff": generation.cutoff,
                "run_cutoffs": [generation.cutoff],
                "bundle_identity": (
                    (generation.manifest.get("per_event") or {}).get(str(int(event))) or {}
                ).get("bundle_identity"),
                "state": ((generation.manifest.get("per_event") or {}).get(str(int(event))) or {}).get(
                    "state"
                ),
                "source": "certified_generation",
            }
            for event in generation.events
        }
        optimized = ro.optimize(
            universe=universe,
            initial_state=initial_state,
            scenario=scenario,
            player_meta=player_meta,
            conn=conn,
            generation=generation,
            config=config,
            cache_dir=request.get("cache_dir"),
        )
        routes = fg.optimizer_routes_for_decision(
            optimized.get("routes") or {},
            transfers_by_route={
                route_id: {
                    int(action["event"]): list(action.get("transfers") or [])
                    for action in (record.get("actions") or [])
                }
                for route_id, record in (optimized.get("routes") or {}).items()
            },
        )
        baseline = next(
            (row["route_id"] for row in routes if not any(r.get("transfers") for r in row["per_event"])),
            None,
        )
        decision = fg.evaluate_four_gw_decision(
            planning_event=int(generation.planning_event),
            support_by_event=support,
            cutoff=generation.cutoff,
            last_event=fg.season_last_event_from_db(conn),
            routes=routes,
            baseline_route_id=baseline,
        )
    finally:
        source_conn.close()
    return {
        "decision": decision,
        "world_info": optimized.get("world_info"),
        "runner_identity": f"fpl_brain.generation_store:{ro.PHASE8B_VERSION}",
    }


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
    "DIAG_DECISION_RECORD_INVALID",
    "DIAG_DECISION_RECORD_UNKNOWN",
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
    "ProductionDescriptorOnly",
    "append_engine_decision_record",
    "assert_cutoff_matches",
    "assert_descriptor_only",
    "assert_generation_bundles_valid",
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
    "manifest_semantic_projection",
    "packet_identity",
    "request_identity",
    "require_snapshot_retained",
    "resolve_generation",
    "result_identity_of",
    "verify_decision",
    "verify_generation",
]
