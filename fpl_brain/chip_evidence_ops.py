"""Bounded operator workflow for prospective BB/TC evidence retention.

This module is an orchestration layer over the accepted generation, route,
permission, forecast, outcome-ledger and calibration validators.  It does not
create predictions, certify generations, run normal decisions, or produce a
production chip assessment.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sqlite3
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from . import chip_reservation_calibration as calibration
from . import chip_reservation_forecast as forecast


ORIGIN_SCHEMA = "fpl_brain.chip_evidence_origin.v1"
FORECAST_RECEIPT_SCHEMA = "fpl_brain.chip_evidence_forecast_receipt.v1"
CAPTURE_RECEIPT_SCHEMA = "fpl_brain.chip_evidence_capture_receipt.v1"
MATURATION_RECEIPT_SCHEMA = "fpl_brain.chip_evidence_maturation_receipt.v1"
HANDOFF_SCHEMA = "fpl_brain.chip_evidence_calibration_handoff.v1"
FORECAST_INTENT_SCHEMA = "fpl_brain.chip_evidence_forecast_intent.v1"
FORECAST_PUBLICATION_SCHEMA = "fpl_brain.chip_evidence_forecast_publication.v1"
CAPTURE_INTENT_SCHEMA = "fpl_brain.chip_evidence_capture_intent.v1"
RECEIPT_DIRNAME = ".chip-evidence"
_LOCK_TIMEOUT_SECONDS = 10.0
_thread_lock = threading.Lock()
_operation_locks: dict[str, threading.Lock] = {}


class ChipEvidenceError(RuntimeError):
    """An explicit collector refusal; no incomplete result is promoted."""


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _parse_utc(value: Any, *, name: str) -> datetime:
    text = str(value or "").strip()
    if not text:
        raise ChipEvidenceError(f"{name} is required")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as failure:
        raise ChipEvidenceError(f"{name} must be an ISO-8601 timestamp") from failure
    if parsed.tzinfo is None:
        raise ChipEvidenceError(f"{name} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _root(path: str | Path) -> Path:
    value = Path(path).expanduser().resolve()
    if value.exists() and not value.is_dir():
        raise ChipEvidenceError(f"evidence root is not a directory: {value}")
    return value


def _metadata_root(evidence_root: str | Path) -> Path:
    return _root(evidence_root) / RECEIPT_DIRNAME


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as failure:
        raise ChipEvidenceError(f"cannot read retained JSON {path}: {failure}") from failure
    if not isinstance(value, dict):
        raise ChipEvidenceError(f"retained JSON is not an object: {path}")
    return value


def _write_new_json(path: Path, value: Mapping[str, Any]) -> str:
    """Atomically publish immutable JSON; an existing name must have equal bytes."""

    path.parent.mkdir(parents=True, exist_ok=True)
    body = _json_bytes(dict(value))
    if path.exists():
        if path.read_bytes() != body:
            raise ChipEvidenceError(f"immutable evidence path already contains different bytes: {path}")
        return _file_sha256(path)
    temporary: Path | None = None
    try:
        fd, name = tempfile.mkstemp(prefix=path.name + ".tmp-", dir=str(path.parent))
        temporary = Path(name)
        with os.fdopen(fd, "wb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != body:
                raise ChipEvidenceError(f"immutable evidence path collision: {path}")
        return _file_sha256(path)
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def _sealed(schema: str, body: Mapping[str, Any]) -> dict[str, Any]:
    value = {"schema": schema, **dict(body)}
    value["receipt_sha256"] = _sha256(value)
    return value


def _verify_seal(value: Mapping[str, Any], *, schema: str) -> None:
    if value.get("schema") != schema:
        raise ChipEvidenceError(f"receipt schema is not {schema}")
    body = dict(value)
    recorded = str(body.pop("receipt_sha256", ""))
    if len(recorded) != 64 or _sha256(body) != recorded:
        raise ChipEvidenceError("receipt content digest does not verify")


@contextlib.contextmanager
def _operation_lock(evidence_root: str | Path, key: str) -> Iterator[None]:
    """Serialize one evidence operation across threads and processes."""

    directory = _metadata_root(evidence_root)
    directory.mkdir(parents=True, exist_ok=True)
    lock_path = directory / f"{key}.lock"
    with _thread_lock:
        local = _operation_locks.setdefault(str(lock_path), threading.Lock())
    if not local.acquire(timeout=_LOCK_TIMEOUT_SECONDS):
        raise ChipEvidenceError(f"operation is already running: {key}")
    handle = None
    try:
        handle = lock_path.open("a+b")
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        deadline = time.monotonic() + _LOCK_TIMEOUT_SECONDS
        while True:
            try:
                if os.name == "nt":
                    import msvcrt

                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except (BlockingIOError, OSError):
                if time.monotonic() >= deadline:
                    raise ChipEvidenceError(f"operation is already running in another process: {key}")
                time.sleep(0.025)
        yield
    finally:
        if handle is not None:
            try:
                if os.name == "nt":
                    import msvcrt

                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
            handle.close()
        local.release()


def _required_identity(
    *, db_path: str | Path, entry_id: int, generation_id: str, decision_id: str,
    route_id: str, planning_event: int, cutoff: str, evidence_root: str | Path,
) -> dict[str, Any]:
    if int(entry_id) <= 0 or int(planning_event) <= 0:
        raise ChipEvidenceError("entry id and planning event must be positive")
    if not all(str(value or "").strip() for value in (generation_id, decision_id, route_id)):
        raise ChipEvidenceError("generation id, decision id and route id are all required")
    _parse_utc(cutoff, name="cutoff")
    database_path = Path(db_path).expanduser().resolve()
    if not database_path.is_file():
        raise ChipEvidenceError(f"database does not exist: {database_path}")
    root = _root(evidence_root)
    return {
        "database_path": str(database_path),
        "entry_id": int(entry_id),
        "generation_id": str(generation_id),
        "decision_id": str(decision_id),
        "route_id": str(route_id),
        "planning_event": int(planning_event),
        "cutoff": str(cutoff),
        "evidence_root": str(root),
    }


def _origin_key(identity: Mapping[str, Any]) -> dict[str, Any]:
    return {key: identity[key] for key in (
        "entry_id", "generation_id", "decision_id", "route_id", "planning_event", "cutoff",
    )}


def _verify_origin_binding(
    receipt: Mapping[str, Any],
    artifact: Mapping[str, Any],
    *,
    origin_id: str,
    expected_identity: Mapping[str, Any],
) -> None:
    _verify_seal(receipt, schema=ORIGIN_SCHEMA)
    identity = artifact.get("identity")
    if artifact.get("schema") != ORIGIN_SCHEMA:
        raise ChipEvidenceError("origin manifest schema is unsupported")
    if not isinstance(identity, Mapping):
        raise ChipEvidenceError("origin manifest omits its explicit source identity")
    if _sha256(_origin_key(identity)) != str(origin_id):
        raise ChipEvidenceError("origin id does not reproduce from the manifest's explicit identity")
    if _origin_key(identity) != _origin_key(expected_identity):
        raise ChipEvidenceError("origin manifest differs from the explicit operation identity")
    if dict(identity) != dict(expected_identity):
        raise ChipEvidenceError("origin manifest database/evidence binding differs from this request")
    if str(artifact.get("origin_id") or "") != str(origin_id):
        raise ChipEvidenceError("origin manifest identity does not match its receipt")
    if str(receipt.get("origin_id") or "") != str(origin_id):
        raise ChipEvidenceError("origin receipt identity differs from the explicit operation identity")
    if str(receipt.get("origin_content_sha256") or "") != _sha256(artifact):
        raise ChipEvidenceError("origin manifest digest does not verify")
    identity_digest = _sha256(identity)
    if str(receipt.get("identity_sha256") or "") != identity_digest:
        raise ChipEvidenceError("origin receipt identity_sha256 does not verify")
    if str(receipt.get("registered_at") or "") != str(artifact.get("registered_at") or ""):
        raise ChipEvidenceError("origin receipt registration time differs from its manifest")
    _parse_utc(artifact.get("registered_at"), name="origin registered_at")


def _read_origin_receipt(
    evidence_root: str | Path,
    origin_id: str,
    *,
    expected_identity: Mapping[str, Any],
) -> dict[str, Any]:
    receipt_path = _metadata_root(evidence_root) / f"origin-{origin_id}.json"
    receipt = _read_json(receipt_path)
    reference = str(receipt.get("origin_ref") or "")
    if Path(reference).name != reference or not reference:
        raise ChipEvidenceError("origin receipt has an unsafe origin reference")
    artifact = _read_json(_root(evidence_root) / reference)
    _verify_origin_binding(
        receipt, artifact, origin_id=origin_id, expected_identity=expected_identity,
    )
    return {"receipt": receipt, "origin": artifact}


def _origin_verification(
    conn: sqlite3.Connection,
    *,
    entry_id: int,
    generation_id: str,
    decision_id: str,
    route_id: str,
    planning_event: int,
    cutoff: str,
) -> tuple[Any, Any, Any, dict[str, Any], dict[str, Any]]:
    from . import generation_store as gs
    from . import planning
    from . import search_permission as sp
    from .chip_route_assembly import load_verified_normal_route

    generation = gs.load_generation(conn, generation_id)
    generation_report = gs.verify_generation(conn, generation_id)
    if generation_report.get("verified") is not True:
        raise ChipEvidenceError("the explicitly named normal generation did not verify")
    if generation.horizon_kind != gs.HORIZON_KIND_FOUR_GW or len(generation.events) != 4:
        raise ChipEvidenceError("origin generation is not the canonical FOUR_GW product")
    if generation.planning_event != int(planning_event) or str(generation.cutoff) != str(cutoff):
        raise ChipEvidenceError("planning event/cutoff do not match the named generation")
    decision_report = gs.verify_decision(conn, decision_id)
    if decision_report.get("verified") is not True:
        raise ChipEvidenceError("the explicitly named decision did not verify")
    decision = gs.load_engine_decision_record(conn, decision_id)
    if str(decision.get("generation_id")) != str(generation_id):
        raise ChipEvidenceError("decision is not bound to the explicitly named generation")
    artifact_path = Path(str(decision.get("decision_artifact_ref") or ""))
    artifact = _read_json(artifact_path)
    manager_packet = ((artifact.get("attribution") or {}).get("manager_packet") or {})
    if int(manager_packet.get("entry_id") or 0) != int(entry_id):
        raise ChipEvidenceError("explicit entry id differs from the verified decision manager packet")
    if artifact.get("planning_event") not in (None, int(planning_event)):
        raise ChipEvidenceError("decision artifact names another planning event")
    route = load_verified_normal_route(conn, decision_id, route_id=route_id)
    if (
        route.generation_id != str(generation_id)
        or route.planning_event != int(planning_event)
        or str(route.cutoff) != str(cutoff)
        or route.route_id != str(route_id)
    ):
        raise ChipEvidenceError("reconstructed route differs from an explicit origin identity")
    permission = sp.require_search_permission(
        conn,
        generation_id,
        expected_origin_planning_event=int(planning_event),
        expected_origin_cutoff=str(cutoff),
        expected_snapshot_sha256=str(route.data_snapshot_sha256),
    )
    if permission.get("permitted") is not True or permission.get("reasons") != []:
        raise ChipEvidenceError("canonical origin search permission is not permitted")

    snapshot_conn = gs._open_generation_snapshot(generation)
    try:
        context = planning.get_planning_context(
            snapshot_conn,
            int(entry_id),
            int(planning_event),
            as_of=str(cutoff),
            season=(artifact.get("attribution") or {}).get("manager_packet", {}).get("season"),
        )
        consumed = dict((artifact.get("attribution") or {}).get("consumed_manager_state") or {})
        if int(consumed.get("entry_id") or 0) != int(entry_id):
            raise ChipEvidenceError("consumed manager state is not bound to the explicit entry")
        if (
            int(consumed.get("planning_event") or -1) != int(planning_event)
            or str(consumed.get("cutoff") or "") != str(cutoff)
        ):
            raise ChipEvidenceError("consumed manager state differs from the origin event/cutoff")
        chip_rows = planning.chips_state(snapshot_conn, int(entry_id), int(planning_event))
    finally:
        snapshot_conn.close()
    manager_state = dict(consumed.get("route_state") or {})
    if not manager_state:
        raise ChipEvidenceError("verified manager state has no canonical route state")
    if manager_state.get("event_start_free_transfers") is not None:
        manager_state["event_start_free_transfers"] = int(manager_state["event_start_free_transfers"])
    return generation, decision, route, {
        "manager_context": dict(consumed.get("source_manager_context") or {}),
        "route_state": manager_state,
        "chip_rows": [dict(row) for row in chip_rows],
        "planning_context_health": dict(getattr(context, "health", {}) or {}),
    }, permission


def _chip_eligibility(chip_rows: Sequence[Mapping[str, Any]], *, action: str) -> dict[str, Any]:
    from . import chip_decision as cd

    if action not in {cd.CHIP_ACTION_BB, cd.CHIP_ACTION_TC}:
        raise ChipEvidenceError("collector supports BB and TC only")
    name = "bboost" if action == cd.CHIP_ACTION_BB else "3xc"
    matching = [dict(row) for row in chip_rows if str(row.get("name") or "") == name]
    eligible = [
        row for row in matching
        if row.get("available_for_event") is True and row.get("used") is False
        and row.get("window_stop_event") is not None
    ]
    if len(eligible) != 1:
        raise ChipEvidenceError(
            f"pinned official chip state does not resolve exactly one eligible {action} chip/expiry "
            f"(found {len(eligible)})"
        )
    row = eligible[0]
    return {
        "action": action,
        "name": name,
        "eligible": True,
        "expiry_event": int(row["window_stop_event"]),
        "window_start_event": int(row["window_start_event"]),
        "window_stop_event": int(row["window_stop_event"]),
        "used": bool(row["used"]),
        "used_event": row.get("used_event"),
        "pinned_row": row,
    }


def _origin_manifest(
    *,
    identity: Mapping[str, Any],
    origin_id: str,
    registered_at: str,
    generation: Any,
    decision: Mapping[str, Any],
    route: Any,
    state: Mapping[str, Any],
    permission: Mapping[str, Any],
) -> dict[str, Any]:
    from . import chip_decision as cd

    source_manager = dict(state.get("manager_context") or {})
    manager_state = dict(state.get("route_state") or {})
    action_states: dict[str, Any] = {}
    for action, action_code in (("BB", cd.CHIP_ACTION_BB), ("TC", cd.CHIP_ACTION_TC)):
        try:
            action_states[action] = _chip_eligibility(state.get("chip_rows") or (), action=action_code)
        except ChipEvidenceError as failure:
            action_states[action] = {"eligible": False, "refusal": str(failure)}
    return {
        "schema": ORIGIN_SCHEMA,
        "origin_id": origin_id,
        "registered_at": registered_at,
        "identity": dict(identity),
        "generation": {
            "horizon_kind": generation.horizon_kind,
            "events": [int(value) for value in generation.events],
            "snapshot_sha256": str(generation.snapshot.get("sha256") or ""),
            "predictive_code_snapshot_sha256": str(
                generation.manifest.get("code_snapshot_sha256") or ""
            ),
            "certification_identity": str(route.certification_identity),
        },
        "decision": {
            "decision_id": str(identity["decision_id"]),
            "result_sha256": str(decision.get("result_sha256") or ""),
            "artifact_sha256": str(route.source_artifact_sha256),
            "verification": "VERIFIED",
        },
        "route": {
            "route_id": str(identity["route_id"]),
            "route_input_sha256": str(route.route_input_sha256),
            "actual_owned_ids": list(route.actual_owned_ids),
            "proposed_owned_ids": list(route.proposed_owned_ids),
            "post_h1_free_transfers": int(route.post_h1_free_transfers),
            "post_h1_bank_tenths": int(route.post_h1_bank_tenths),
            "policy": route.policy.as_dict(),
        },
        "manager_state": {
            "current_free_transfers": source_manager.get("free_transfers"),
            "bank_tenths": source_manager.get("bank"),
            "event_start_free_transfers": manager_state.get("event_start_free_transfers"),
            "route_state": manager_state,
            "source": "VERIFIED_NORMAL_GENERATION_PINNED_MANAGER_CONTEXT",
        },
        "chip_eligibility": action_states,
        "search_permission": dict(permission),
        "hypothetical_assessment_reuse": False,
    }


def _verify_origin_manifest_authority(
    artifact: Mapping[str, Any],
    *,
    expected_identity: Mapping[str, Any],
    origin_id: str,
    generation: Any,
    decision: Mapping[str, Any],
    route: Any,
    state: Mapping[str, Any],
    permission: Mapping[str, Any],
) -> None:
    expected = _origin_manifest(
        identity=expected_identity,
        origin_id=origin_id,
        registered_at=str(artifact.get("registered_at") or ""),
        generation=generation,
        decision=decision,
        route=route,
        state=state,
        permission=permission,
    )
    # Compare their persisted JSON representation: canonical route policies
    # contain tuples in memory, while a retained JSON manifest necessarily
    # reloads them as arrays.  Those are the same serialized evidence value.
    if _json_bytes(dict(artifact)) != _json_bytes(expected):
        differing = sorted(
            key for key in set(artifact) | set(expected)
            if _json_bytes(artifact.get(key)) != _json_bytes(expected.get(key))
        )
        detail = (
            {key: {"retained": artifact.get("route", {}).get(key), "verified": expected.get("route", {}).get(key)}
             for key in set(artifact.get("route", {})) | set(expected.get("route", {}))
             if _json_bytes(artifact.get("route", {}).get(key)) != _json_bytes(expected.get("route", {}).get(key))}
            if "route" in differing else {}
        )
        raise ChipEvidenceError(
            "retained origin manifest differs from the canonically reverified generation/decision/route/permission "
            f"on {differing}: {detail}"
        )


def _origin_receipt(origin_id: str, origin_ref: str, artifact: Mapping[str, Any]) -> dict[str, Any]:
    return _sealed(ORIGIN_SCHEMA, {
        "origin_id": origin_id,
        "origin_ref": origin_ref,
        "origin_content_sha256": _sha256(artifact),
        "identity_sha256": _sha256(artifact["identity"]),
        "registered_at": artifact["registered_at"],
    })


def register_origin(
    conn: sqlite3.Connection,
    *,
    db_path: str | Path,
    entry_id: int,
    generation_id: str,
    decision_id: str,
    route_id: str,
    planning_event: int,
    cutoff: str,
    evidence_root: str | Path,
) -> dict[str, Any]:
    """Verify and immutably register one explicit normal origin."""

    identity = _required_identity(
        db_path=db_path, entry_id=entry_id, generation_id=generation_id,
        decision_id=decision_id, route_id=route_id, planning_event=planning_event,
        cutoff=cutoff, evidence_root=evidence_root,
    )
    origin_id = _sha256(_origin_key(identity))
    with _operation_lock(evidence_root, f"origin-{origin_id}"):
        metadata = _metadata_root(evidence_root)
        receipt_path = metadata / f"origin-{origin_id}.json"
        generation, decision, route, state, permission = _origin_verification(
            conn,
            entry_id=entry_id,
            generation_id=generation_id,
            decision_id=decision_id,
            route_id=route_id,
            planning_event=planning_event,
            cutoff=cutoff,
        )
        if receipt_path.exists():
            retained = _read_origin_receipt(
                evidence_root, origin_id, expected_identity=identity,
            )
            _verify_origin_manifest_authority(
                retained["origin"], expected_identity=identity, origin_id=origin_id,
                generation=generation, decision=decision, route=route, state=state,
                permission=permission,
            )
            return retained

        body = _origin_manifest(
            identity=identity,
            origin_id=origin_id,
            registered_at=_utc_now(),
            generation=generation,
            decision=decision,
            route=route,
            state=state,
            permission=permission,
        )

        candidates: list[tuple[Path, dict[str, Any]]] = []
        for path in _root(evidence_root).glob("chip-origin-*.json"):
            candidate = _read_json(path)
            candidate_identity = candidate.get("identity")
            if (
                str(candidate.get("origin_id") or "") == origin_id
                or (isinstance(candidate_identity, Mapping)
                    and _origin_key(candidate_identity) == _origin_key(identity))
            ):
                candidates.append((path, candidate))
        if len(candidates) > 1:
            raise ChipEvidenceError("multiple origin manifests match the requested origin; refusing recovery")
        if candidates:
            path, candidate = candidates[0]
            if path.name != f"chip-origin-{_sha256(candidate)}.json":
                raise ChipEvidenceError("orphan origin manifest filename is not content-addressed")
            _verify_origin_manifest_authority(
                candidate, expected_identity=identity, origin_id=origin_id,
                generation=generation, decision=decision, route=route, state=state,
                permission=permission,
            )
            receipt = _origin_receipt(origin_id, path.name, candidate)
            _write_new_json(receipt_path, receipt)
            return _read_origin_receipt(
                evidence_root, origin_id, expected_identity=identity,
            )

        origin_digest = _sha256(body)
        origin_ref = f"chip-origin-{origin_digest}.json"
        _write_new_json(_root(evidence_root) / origin_ref, body)
        receipt = _origin_receipt(origin_id, origin_ref, body)
        _write_new_json(receipt_path, receipt)
        return _read_origin_receipt(
            evidence_root, origin_id, expected_identity=identity,
        )


def _flat_artifact(root: str | Path, reference: Any) -> tuple[Path, dict[str, Any]]:
    name = str(reference or "")
    if not name or Path(name).name != name:
        raise ChipEvidenceError("artifact reference must be a flat evidence-root filename")
    path = _root(root) / name
    value = _read_json(path)
    return path, value


def _source_identity(route: Any, generation: Any) -> dict[str, Any]:
    return {
        "source_decision_id": str(route.source_decision_id),
        "source_result_sha256": str(route.source_result_sha256),
        "source_artifact_sha256": str(route.source_artifact_sha256),
        "generation_id": str(route.generation_id),
        "planning_event": int(route.planning_event),
        "origin_cutoff": str(route.cutoff),
        "data_snapshot_sha256": str(route.data_snapshot_sha256),
        "predictive_code_snapshot_sha256": str(generation.manifest.get("code_snapshot_sha256") or ""),
        "certification_identity": str(route.certification_identity),
    }


def _revalidate_registered_origin(
    conn: sqlite3.Connection,
    *,
    identity: Mapping[str, Any],
) -> tuple[dict[str, Any], Any, Any, Any, dict[str, Any], dict[str, Any]]:
    origin_id = _sha256(_origin_key(identity))
    retained = _read_origin_receipt(
        identity["evidence_root"], origin_id, expected_identity=identity,
    )
    generation, decision, route, state, permission = _origin_verification(
        conn,
        entry_id=int(identity["entry_id"]),
        generation_id=str(identity["generation_id"]),
        decision_id=str(identity["decision_id"]),
        route_id=str(identity["route_id"]),
        planning_event=int(identity["planning_event"]),
        cutoff=str(identity["cutoff"]),
    )
    _verify_origin_manifest_authority(
        retained["origin"], expected_identity=identity, origin_id=origin_id,
        generation=generation, decision=decision, route=route, state=state,
        permission=permission,
    )
    return retained, generation, decision, route, state, permission


def _world_cache_requirements(
    conn: sqlite3.Connection,
    *,
    route: Any,
    generation: Any,
    coverage_product: Mapping[str, Any] | None,
    expiry_event: int,
) -> list[dict[str, Any]]:
    """Derive cache identity exactly as the accepted loaders, without building worlds."""

    from dataclasses import fields

    from . import candidate_universe as cu
    from . import generation_store as gs
    from . import route_optimizer as ro
    from .chip_wildcard import pool_binding_from_store

    decision_record = gs.load_engine_decision_record(conn, route.source_decision_id)
    artifact = _read_json(Path(str(decision_record["decision_artifact_ref"])))
    raw_config = dict((artifact.get("search") or {}).get("config") or {})
    allowed = {field.name for field in fields(ro.OptimizerConfig)}
    raw_config["events"] = route.events
    config = ro.OptimizerConfig(**{key: value for key, value in raw_config.items() if key in allowed})
    continuation = None
    if coverage_product is not None:
        continuation = gs.load_generation(conn, str(coverage_product["product_generation_id"]))
    rows: list[dict[str, Any]] = []
    for event in range(int(route.planning_event) + 1, int(expiry_event) + 1):
        source_generation = generation if event in route.events else continuation
        if source_generation is None:
            raise ChipEvidenceError(f"GW{event} has no explicitly certified world source")
        if event not in source_generation.events:
            raise ChipEvidenceError(f"GW{event} is absent from its certified world source")
        snapshot_conn = gs._open_generation_snapshot(source_generation)
        try:
            pool = pool_binding_from_store(snapshot_conn)
            player_ids = tuple(sorted(int(value) for value in pool.eligible_ids))
        finally:
            snapshot_conn.close()
        if not player_ids:
            raise ChipEvidenceError(f"GW{event} pinned official player pool is empty")
        runs = source_generation.runs_for(event)
        key = ro.world_cache_key(
            event=event,
            generation_id=source_generation.generation_id,
            runs=runs,
            config=config,
            union_ids=player_ids,
        )
        rows.append({
            "event": event,
            "generation_id": source_generation.generation_id,
            "runs": dict(runs),
            "cache_key": key,
            "player_ids": list(player_ids),
            "worlds": int(config.search_draws),
            "seed": int(config.seed),
        })
    return rows

def _preflight_world_cache(
    requirements: Sequence[Mapping[str, Any]],
    *,
    cache_dir: str | Path,
) -> list[dict[str, Any]]:
    from . import route_optimizer as ro

    directory = Path(cache_dir).expanduser().resolve()
    report: list[dict[str, Any]] = []
    misses: list[str] = []
    for requirement in requirements:
        path = directory / f"{requirement['cache_key']}.json"
        matrix, evidence = ro.read_world_cache_entry(path)
        hit = matrix is not None
        if hit:
            if (
                sorted(int(value) for value in matrix.get("player_ids", ()))
                != sorted(int(value) for value in requirement["player_ids"])
                or int(matrix.get("worlds", -1)) != int(requirement["worlds"])
            ):
                hit = False
                evidence = {"cache": {"status": "MISS_MATRIX_BINDING_MISMATCH", "cache_key": requirement["cache_key"]}}
        if not hit:
            misses.append(f"GW{requirement['event']}:{requirement['cache_key']}")
        report.append({
            "event": int(requirement["event"]),
            "cache_key": str(requirement["cache_key"]),
            "path": str(path),
            "status": evidence.get("cache", {}).get("status"),
            "hit": bool(hit),
        })
    return report


def _forecast_operation_id(origin_id: str, action: str, observation_id: str) -> str:
    return _sha256([str(origin_id), str(action), str(observation_id)])


def _read_forecast_receipt(evidence_root: str | Path, operation_id: str) -> dict[str, Any]:
    receipt = _read_json(_metadata_root(evidence_root) / f"forecast-{operation_id}.json")
    _verify_seal(receipt, schema=FORECAST_RECEIPT_SCHEMA)
    if str(receipt.get("operation_id") or "") != operation_id:
        raise ChipEvidenceError("forecast receipt filename and operation identity disagree")
    for field in ("forecast_ref", "causal_evidence_ref"):
        path, value = _flat_artifact(evidence_root, receipt.get(field))
        expected = str(receipt.get(field.removesuffix("_ref") + "_sha256") or "")
        digest = (
            calibration._canonical_sha256(value)
            if field == "causal_evidence_ref"
            else forecast.canonical_sha256(value)
        )
        if not expected or expected != digest:
            raise ChipEvidenceError(f"forecast receipt {field} digest does not verify")
        if path.name != str(receipt[field]):
            raise ChipEvidenceError("forecast artifact reference is not a flat filename")
    return receipt


def _issued_forecast_artifacts(
    conn: sqlite3.Connection,
    *,
    evidence_root: str | Path,
    source_forecast_ref: str,
    calculation_started_at: str,
    issued_at: str,
) -> tuple[dict[str, Any], str, str]:
    """Rebuild the canonical forecast records with the actual completed issue time.

    The accepted BB/TC producer accepts ``made_at`` before evaluation begins.
    Its result is therefore treated as a calculation artifact until the collector
    records completion.  This function uses the accepted forecast builders and
    verifiers to retain equivalent scored opportunities at the actual issue time;
    no evaluator is called and no world is built here.
    """

    from . import chip_reservation_forecast as crf

    started = _parse_utc(calculation_started_at, name="calculation_started_at")
    issued = _parse_utc(issued_at, name="issued_at")
    if issued < started:
        raise ChipEvidenceError("forecast issuance precedes calculation start")
    source_path, original = _flat_artifact(evidence_root, source_forecast_ref)
    original_body = dict(original)
    original_identity = str(original_body.pop("artifact_sha256", ""))
    if (
        source_path.name != f"chip-reservation-forecast-{original_identity}.json"
        or original_identity != crf.canonical_sha256(original_body)
        or str(original.get("made_at") or "") != calculation_started_at
    ):
        raise ChipEvidenceError("completed calculation forecast does not match its immutable start-time identity")

    original_records: dict[str, dict[str, Any]] = {}
    for item in original.get("opportunities") or ():
        if not isinstance(item, Mapping):
            raise ChipEvidenceError("completed calculation forecast contains a malformed opportunity")
        reference = str(item.get("artifact_ref") or "")
        path, payload = _flat_artifact(evidence_root, reference)
        payload_body = dict(payload)
        payload_identity = str(payload_body.pop("artifact_sha256", ""))
        if (
            path.name != f"chip-event-opportunity-{payload_identity}.json"
            or payload_identity != crf.canonical_sha256(payload_body)
            or dict(item.get("artifact_payload") or {}) != payload
            or str(item.get("artifact_sha256") or "") != payload_identity
        ):
            raise ChipEvidenceError("completed opportunity artifact does not match its retained forecast")
        original_records[reference] = payload
    crf.verify_reservation_forecast(
        original, store_conn=conn, expected={"made_at": calculation_started_at},
        evidence_verifier=original_records.__getitem__,
    )

    issued_records: dict[str, dict[str, Any]] = {}
    issued_refs: list[str] = []
    for old_ref, payload in sorted(
        original_records.items(), key=lambda pair: int(pair[1].get("event") or -1),
    ):
        rebuilt = crf.build_event_opportunity_record(
            action=str(payload["action"]),
            planning_event=int(payload["planning_event"]),
            event=int(payload["event"]),
            origin_cutoff=str(payload["origin_cutoff"]),
            made_at=issued_at,
            expected_incremental_points=float(payload["expected_incremental_points"]),
            opportunity_model=str(payload["opportunity_model"]),
            source_identity=payload["source_identity"],
            reservation_state=payload["reservation_state"],
            world_identity=str(payload["world_identity"]),
            outcome_arms=payload.get("outcome_arms"),
            input_as_of=str(payload["input_as_of"]),
            forecast_mode=str(payload["forecast_mode"]),
            value_units=str(payload["value_units"]),
            coverage_product_sha256=payload.get("coverage_product_sha256"),
            evaluator_identity=payload.get("evaluator_identity"),
            continuation_context=payload.get("continuation_context"),
        )
        permission = payload.get("search_permission_evaluation")
        if not isinstance(permission, Mapping):
            raise ChipEvidenceError("completed opportunity omits its canonical permission evaluation")
        rebuilt = crf._attach_search_permission_evaluation(rebuilt, permission)
        old_compare, new_compare = dict(payload), dict(rebuilt)
        old_compare.pop("made_at", None)
        old_compare.pop("artifact_sha256", None)
        new_compare.pop("made_at", None)
        new_compare.pop("artifact_sha256", None)
        if _json_bytes(old_compare) != _json_bytes(new_compare):
            raise ChipEvidenceError("reissued opportunity changed evaluated values, policy, or source identity")
        retained = crf.retain_event_opportunity_record(rebuilt, evidence_root, store_conn=conn)
        ref = Path(retained["path"]).name
        issued_refs.append(ref)
        issued_records[ref] = rebuilt

    forecast = crf.build_reservation_forecast(
        store_conn=conn,
        action=str(original["action"]),
        planning_event=int(original["planning_event"]),
        origin_cutoff=str(original["origin_cutoff"]),
        made_at=issued_at,
        expiry_event=original.get("expiry_event"),
        source_identity=original["source_identity"],
        reservation_state=original["reservation_state"],
        opportunity_refs=issued_refs,
        evidence_verifier=issued_records.__getitem__,
        input_as_of=str(original["input_as_of"]),
        forecast_mode=str(original["forecast_mode"]),
        coverage_product=original.get("coverage_product"),
        value_units=str(original["value_units"]),
    )
    stable_fields = (
        "schema", "version", "model", "action", "planning_event", "origin_cutoff",
        "input_as_of", "forecast_mode", "value_units", "expiry_event", "source_identity",
        "reservation_state", "reservation_state_sha256", "coverage_product",
        "coverage_product_sha256", "coverage_complete", "coverage_status", "covered_events",
        "raw_value", "selected_event", "reason_code", "selection_policy",
    )
    if any(forecast.get(key) != original.get(key) for key in stable_fields):
        raise ChipEvidenceError("reissued forecast changed coverage, selected event, value, or source")
    crf.verify_reservation_forecast(
        forecast,
        store_conn=conn,
        expected={
            "action": str(original["action"]),
            "planning_event": int(original["planning_event"]),
            "origin_cutoff": str(original["origin_cutoff"]),
            "made_at": issued_at,
            "input_as_of": str(original["input_as_of"]),
            "forecast_mode": str(original["forecast_mode"]),
            "expiry_event": original.get("expiry_event"),
            "source_identity": original["source_identity"],
        },
        evidence_verifier=issued_records.__getitem__,
    )
    retained_forecast = crf.retain_reservation_forecast(
        forecast, evidence_root, store_conn=conn,
    )
    return forecast, Path(retained_forecast["path"]).name, crf.canonical_sha256(forecast)


def _load_causal_origin(
    evidence_root: str | Path,
    *,
    observation_id: str,
    action: str,
    origin_id: str,
) -> tuple[Path, dict[str, Any], Path, dict[str, Any]]:
    candidates: list[tuple[Path, dict[str, Any]]] = []
    for path in _root(evidence_root).glob("chip-causal-evidence-*.json"):
        value = _read_json(path)
        if (
            str(value.get("observation_id") or "") == str(observation_id)
            and str(value.get("action") or "") == str(action)
            and value.get("label") is None
        ):
            candidates.append((path, value))
    if len(candidates) != 1:
        raise ChipEvidenceError(
            f"expected one unlabelled causal origin for {observation_id}/{action}; found {len(candidates)}"
        )
    causal_path, causal = candidates[0]
    causal_digest = calibration._canonical_sha256(causal)
    if causal_path.name != f"chip-causal-evidence-{causal_digest}.json":
        raise ChipEvidenceError("causal-origin filename is not content-addressed")
    source = causal.get("source")
    if not isinstance(source, Mapping):
        raise ChipEvidenceError("causal origin has no source identity")
    if str(source.get("generation_id") or "") == "" or int(source.get("planning_event") or -1) <= 0:
        raise ChipEvidenceError("causal origin source identity is incomplete")
    if str(causal.get("policy") or "") != calibration.CAUSAL_EVIDENCE_POLICY:
        raise ChipEvidenceError("causal origin policy is unsupported")
    if str(causal.get("forecast", {}).get("artifact_ref") or "") == "":
        raise ChipEvidenceError("causal origin has no forecast reference")
    forecast_path, forecast_value = _flat_artifact(
        evidence_root, causal["forecast"]["artifact_ref"],
    )
    if str(causal["forecast"].get("retained_content_sha256") or "") != calibration._canonical_sha256(
        forecast_value
    ):
        raise ChipEvidenceError("causal origin raw forecast digest does not verify")
    if str(causal["forecast"].get("artifact_sha256") or "") != str(
        forecast_value.get("artifact_sha256") or ""
    ):
        raise ChipEvidenceError("causal origin forecast artifact identity differs")
    if str(causal["forecast"].get("artifact_ref") or "") != forecast_path.name:
        raise ChipEvidenceError("causal origin raw forecast reference differs from its retained filename")
    if str(source.get("generation_id") or "") == "" or str(
        causal.get("action") or ""
    ) not in {"BB", "TC"}:
        raise ChipEvidenceError("causal origin is not a BB/TC source")
    return causal_path, causal, forecast_path, forecast_value


def forecast_origin(
    conn: sqlite3.Connection,
    *,
    db_path: str | Path,
    entry_id: int,
    generation_id: str,
    decision_id: str,
    route_id: str,
    planning_event: int,
    cutoff: str,
    action: str,
    expiry_event: int,
    observation_id: str,
    continuation_generation_id: str | None,
    evidence_root: str | Path,
    cache_dir: str | Path,
    materialize_worlds: bool = False,
) -> dict[str, Any]:
    """Retain one prospective BB or TC forecast on fully verified inputs."""

    from . import chip_decision as cd
    from . import chip_reservation_forecast as crf
    from . import generation_store as gs
    from . import planning
    from . import season_rules as sr
    from .chip_route_assembly import (
        _post_h1_save_reservation_state,
        build_bb_tc_reservation_forecast,
        load_verified_normal_route,
    )

    action_codes = {"BB": cd.CHIP_ACTION_BB, "TC": cd.CHIP_ACTION_TC,
                    cd.CHIP_ACTION_BB: cd.CHIP_ACTION_BB, cd.CHIP_ACTION_TC: cd.CHIP_ACTION_TC}
    action = action_codes.get(str(action).upper(), action_codes.get(str(action)))
    if action not in {cd.CHIP_ACTION_BB, cd.CHIP_ACTION_TC}:
        raise ChipEvidenceError("forecast action must be BB or TC")
    if not str(observation_id or "").strip():
        raise ChipEvidenceError("an explicit observation id is required")
    identity = _required_identity(
        db_path=db_path, entry_id=entry_id, generation_id=generation_id,
        decision_id=decision_id, route_id=route_id, planning_event=planning_event,
        cutoff=cutoff, evidence_root=evidence_root,
    )
    origin_id = _sha256(_origin_key(identity))
    operation_id = _forecast_operation_id(origin_id, action, observation_id)
    cache_path = Path(cache_dir).expanduser().resolve()
    expected_separate_cache = (_metadata_root(evidence_root) / "world-cache" / origin_id).resolve()
    if cache_path == _root(evidence_root) or cache_path == _metadata_root(evidence_root):
        raise ChipEvidenceError("world cache must be separate from the flat evidence root")
    if materialize_worlds and cache_path != expected_separate_cache:
        raise ChipEvidenceError(
            "--materialize-worlds requires the explicit origin-specific cache path "
            f"{expected_separate_cache}"
        )

    with _operation_lock(evidence_root, f"forecast-{operation_id}"):
        retained, generation, _decision, route, state, _permission = _revalidate_registered_origin(
            conn, identity=identity,
        )
        existing_receipt_path = _metadata_root(evidence_root) / f"forecast-{operation_id}.json"
        if existing_receipt_path.exists():
            _retained, _generation, _decision, _route, _state, parent = _load_forecast_origin(
                conn, identity=identity, action=action, observation_id=observation_id,
            )
            return parent["receipt"]

        origin = retained["origin"]
        action_key = "BB" if action == cd.CHIP_ACTION_BB else "TC"
        chip_state = origin.get("chip_eligibility", {}).get(action_key) or {}
        if chip_state.get("eligible") is not True:
            raise ChipEvidenceError(f"pinned origin has no eligible {action_key} chip")
        pinned_expiry = int(chip_state.get("expiry_event") or -1)
        if int(expiry_event) != pinned_expiry or pinned_expiry <= int(planning_event):
            raise ChipEvidenceError("requested expiry differs from pinned chip state or has no future opportunity")

        source_identity = _source_identity(route, generation)
        coverage_product = None
        # The accepted reservation-forecast builder only marks a future value
        # complete when it is bound to a same-origin CHIP_RESERVATION coverage
        # product. That product is required even when expiry lies inside the
        # normal route's four-event window; without it the producer can score
        # rows but can only return UNKNOWN/incomplete coverage.
        if pinned_expiry > int(planning_event) and continuation_generation_id:
            coverage_product = crf.build_reservation_coverage_product(
                conn,
                action=action,
                source_identity=source_identity,
                expiry_event=pinned_expiry,
                product_generation_id=str(continuation_generation_id),
            )
            if (
                coverage_product.get("coverage_complete") is not True
                or coverage_product.get("missing_events")
                or [int(value) for value in coverage_product.get("forecast_events") or ()]
                != list(range(int(planning_event) + 1, pinned_expiry + 1))
            ):
                raise ChipEvidenceError("same-origin reservation product does not cover every future event through expiry")

        forecast_events = (
            [int(value) for value in coverage_product["forecast_events"]]
            if coverage_product is not None
            else [event for event in route.events[1:] if event <= pinned_expiry]
        )
        if forecast_events != list(range(int(planning_event) + 1, pinned_expiry + 1)):
            raise ChipEvidenceError("verified forecast products do not provide contiguous future-event coverage")
        intent_path = _metadata_root(evidence_root) / f"forecast-intent-{operation_id}.json"
        publication_path = _metadata_root(evidence_root) / f"forecast-publication-{operation_id}.json"
        expected_intent = {
            "operation_id": operation_id,
            "identity_sha256": _sha256(identity),
            "origin_id": origin_id,
            "observation_id": str(observation_id),
            "action": action,
            "expiry_event": pinned_expiry,
            "continuation_generation_id": continuation_generation_id,
            "cache_dir": str(cache_path),
            "materialize_worlds": bool(materialize_worlds),
        }
        for event in forecast_events:
            state_name, _basis = planning.event_data_state(conn, event)
            if state_name == planning.EVENT_STATE_FINAL:
                raise ChipEvidenceError(
                    f"GW{event} is already officially final; refusing to issue a retrospective prospective forecast"
                )

        intent_exists = intent_path.exists()
        if (
            not intent_exists
            and pinned_expiry > int(planning_event)
            and not continuation_generation_id
        ):
            raise ChipEvidenceError(
                "complete future-chip coverage requires an explicit same-origin CHIP_RESERVATION generation"
            )
        if intent_exists:
            intent = _read_json(intent_path)
            _verify_seal(intent, schema=FORECAST_INTENT_SCHEMA)
            if any(intent.get(key) != value for key, value in expected_intent.items()):
                raise ChipEvidenceError("forecast retry differs from the immutable operation intent")
            calculation_started_at = str(intent.get("calculation_started_at") or "")
            _parse_utc(calculation_started_at, name="forecast calculation_started_at")
            if not publication_path.exists():
                raise ChipEvidenceError(
                    "forecast attempt has only an intent and no verified completion/publication receipt; "
                    "do not reuse its start time—retry with a new observation id"
                )
            publication = _read_json(publication_path)
            _verify_seal(publication, schema=FORECAST_PUBLICATION_SCHEMA)
            expected_publication = {
                **expected_intent,
                "intent_sha256": _sha256(intent),
            }
            if any(publication.get(key) != value for key, value in expected_publication.items()):
                raise ChipEvidenceError("forecast publication receipt differs from its immutable intent")
            issued_at = str(publication.get("issued_at") or "")
            _parse_utc(issued_at, name="forecast issued_at")
            cache_report = publication.get("cache")
            if not isinstance(cache_report, list):
                raise ChipEvidenceError("forecast publication receipt omits its verified cache report")
            original_forecast_ref = str(publication.get("source_forecast_ref") or "")
            source_path, source_forecast = _flat_artifact(evidence_root, original_forecast_ref)
            from . import chip_reservation_forecast as crf

            if (
                source_path.name != original_forecast_ref
                or str(source_forecast.get("artifact_sha256") or "")
                != str(publication.get("source_forecast_artifact_identity") or "")
                or crf.canonical_sha256({
                    key: value for key, value in source_forecast.items() if key != "artifact_sha256"
                }) != str(publication.get("source_forecast_artifact_identity") or "")
                or crf.canonical_sha256(source_forecast)
                != str(publication.get("source_forecast_content_sha256") or "")
            ):
                raise ChipEvidenceError("completed source forecast differs from its publication receipt")
            materialized_worlds = bool(publication.get("materialized_worlds"))
        else:
            prior_evidence = [
                path for path in _root(evidence_root).glob("chip-causal-evidence-*.json")
                if str(_read_json(path).get("observation_id") or "") == str(observation_id)
            ]
            if prior_evidence:
                raise ChipEvidenceError(
                    "forecast artifacts exist without their immutable operation intent; refusing to reissue"
                )
            cache_requirements = _world_cache_requirements(
                conn, route=route, generation=generation,
                coverage_product=coverage_product, expiry_event=pinned_expiry,
            )
            cache_report = _preflight_world_cache(cache_requirements, cache_dir=cache_path)
            misses = [row for row in cache_report if not row["hit"]]
            if misses and not materialize_worlds:
                raise ChipEvidenceError(
                    "canonical world cache is incomplete; refusing before scoring/world construction: "
                    + ", ".join(f"GW{row['event']}={row['status']}" for row in misses)
                )
            calculation_started_at = _utc_now()
            if _parse_utc(calculation_started_at, name="forecast calculation_started_at") < _parse_utc(
                cutoff, name="origin cutoff",
            ):
                raise ChipEvidenceError("forecast calculation starts before the pinned input cutoff")

            season_artifact = _read_json(Path(str(
                gs.load_engine_decision_record(conn, decision_id)["decision_artifact_ref"]
            )))
            manager_packet = (season_artifact.get("attribution") or {}).get("manager_packet") or {}
            season_code = str(manager_packet.get("season") or "")
            rules = None
            if any(event > int(route.events[-1]) for event in forecast_events):
                pinned_snapshot_conn = gs._open_generation_snapshot(generation)
                try:
                    pinned_rules = sr.resolve_origin_pinned_season_rules(
                        pinned_snapshot_conn,
                        season=season_code,
                        cutoff=str(route.cutoff),
                        data_snapshot_sha256=str(route.data_snapshot_sha256),
                    )
                finally:
                    pinned_snapshot_conn.close()
                if pinned_rules is None:
                    raise ChipEvidenceError("origin-pinned season rules are unavailable for continuation scoring")
                rules = pinned_rules.rules

            intent = _sealed(FORECAST_INTENT_SCHEMA, {
                **expected_intent,
                "calculation_started_at": calculation_started_at,
            })
            _write_new_json(intent_path, intent)
            generated = build_bb_tc_reservation_forecast(
                conn,
                route,
                action=action,
                expiry_event=pinned_expiry,
                reservation_state=_post_h1_save_reservation_state(route),
                made_at=calculation_started_at,
                evidence_root=evidence_root,
                cache_dir=cache_path,
                allow_materialization=materialize_worlds,
                continuation_generation_id=continuation_generation_id,
                rules=rules,
            )
            if generated.get("coverage_status") != forecast.FORECAST_READY:
                raise ChipEvidenceError("canonical forecast producer did not return complete through-expiry coverage")
            for event in forecast_events:
                state_name, _basis = planning.event_data_state(conn, event)
                if state_name == planning.EVENT_STATE_FINAL:
                    raise ChipEvidenceError(
                        f"GW{event} became officially final during forecast calculation; "
                        "the completed attempt is not a prospective origin"
                    )
            calculation_path = Path(str(generated.get("path") or "")).expanduser().resolve()
            if calculation_path.parent != _root(evidence_root):
                raise ChipEvidenceError("canonical producer forecast was not retained in the explicit flat evidence root")
            issued_at = _utc_now()
            _parse_utc(issued_at, name="forecast issued_at")
            if _parse_utc(issued_at, name="forecast issued_at") < _parse_utc(
                calculation_started_at, name="forecast calculation_started_at",
            ):
                raise ChipEvidenceError("forecast issuance precedes calculation completion")
            from . import chip_reservation_forecast as crf

            calculation_artifact = dict(generated.get("artifact") or {})
            calculation_identity = str(calculation_artifact.get("artifact_sha256") or "")
            calculation_body = {
                key: value for key, value in calculation_artifact.items()
                if key != "artifact_sha256"
            }
            if (
                not calculation_identity
                or calculation_path.name != f"chip-reservation-forecast-{calculation_identity}.json"
                or crf.canonical_sha256(calculation_body) != calculation_identity
                or _json_bytes(_read_json(calculation_path)) != _json_bytes(calculation_artifact)
            ):
                raise ChipEvidenceError("canonical producer returned an unverified completed forecast artifact")
            crf.verify_reservation_forecast(
                calculation_artifact,
                store_conn=conn,
                expected={
                    "action": action,
                    "planning_event": int(planning_event),
                    "origin_cutoff": str(cutoff),
                    "made_at": calculation_started_at,
                    "input_as_of": str(cutoff),
                    "forecast_mode": crf.FORECAST_MODE_PROSPECTIVE,
                    "expiry_event": pinned_expiry,
                    "source_identity": source_identity,
                },
                evidence_verifier=lambda reference: _flat_artifact(evidence_root, reference)[1],
            )
            publication = _sealed(FORECAST_PUBLICATION_SCHEMA, {
                **expected_intent,
                "intent_sha256": _sha256(intent),
                "calculation_started_at": calculation_started_at,
                "issued_at": issued_at,
                "source_forecast_ref": calculation_path.name,
                "source_forecast_artifact_identity": str(calculation_artifact.get("artifact_sha256") or ""),
                "source_forecast_content_sha256": crf.canonical_sha256(calculation_artifact),
                "cache": cache_report,
                "materialized_worlds": bool(misses and materialize_worlds),
            })
            _write_new_json(publication_path, publication)
            original_forecast_ref = calculation_path.name
            materialized_worlds = bool(misses and materialize_worlds)

        if str(publication.get("calculation_started_at") or "") != calculation_started_at:
            raise ChipEvidenceError("forecast publication and intent disagree on calculation start")
        issued_forecast, forecast_ref, _forecast_content_sha256 = _issued_forecast_artifacts(
            conn,
            evidence_root=evidence_root,
            source_forecast_ref=original_forecast_ref,
            calculation_started_at=calculation_started_at,
            issued_at=issued_at,
        )
        retained_origin = calibration.retain_causal_origin_observation(
            observation_id=str(observation_id),
            forecast_artifact=issued_forecast,
            evidence_root=evidence_root,
            store_conn=conn,
            evidence_verifier=lambda reference: _flat_artifact(evidence_root, reference)[1],
        )
        causal = retained_origin["causal_evidence"]
        causal_ref = str(retained_origin["causal_evidence_ref"])
        forecast_ref = str(retained_origin["forecast_ref"])
        snapshot_causality = generation.manifest.get("search_permission_causality") or {}
        if not isinstance(snapshot_causality, Mapping) or snapshot_causality.get("temporal_status") != "CAUSAL":
            raise ChipEvidenceError("pinned generation has no verified causal snapshot-timing evidence")
        # The snapshot's consistency instant identifies the data state being
        # observed. Capture completion is retained separately as the later time
        # when its immutable file became available to the certified execution.
        observed_at = str(snapshot_causality.get("snapshot_consistency_at") or "")
        snapshot_available_at = str(snapshot_causality.get("snapshot_capture_completed_at") or "")
        if not observed_at or _parse_utc(observed_at, name="pinned snapshot observed_at") > _parse_utc(
            cutoff, name="origin cutoff",
        ):
            raise ChipEvidenceError("pinned snapshot observation time is absent or later than the origin cutoff")
        if not snapshot_available_at:
            raise ChipEvidenceError("pinned snapshot availability time is absent from verified causal evidence")
        _parse_utc(snapshot_available_at, name="pinned snapshot available_at")
        selected_event = int(issued_forecast.get("selected_event") or -1)
        selected_opportunity = next((
            item.get("artifact_payload") for item in issued_forecast.get("opportunities") or ()
            if int((item.get("artifact_payload") or {}).get("event") or -1) == selected_event
        ), {})
        receipt = _sealed(FORECAST_RECEIPT_SCHEMA, {
            "operation_id": operation_id,
            "identity_sha256": _sha256(identity),
            "origin_id": origin_id,
            "observation_id": str(observation_id),
            "action": action,
            "planning_event": int(planning_event),
            "origin_cutoff": str(cutoff),
            "input_as_of": str(issued_forecast.get("input_as_of") or cutoff),
            "observed_at": observed_at,
            "snapshot_available_at": snapshot_available_at,
            "made_at": issued_at,
            "expiry_event": pinned_expiry,
            "selected_event": selected_event,
            "forecast_ref": forecast_ref,
            "forecast_sha256": calibration._canonical_sha256(
                _flat_artifact(evidence_root, forecast_ref)[1]
            ),
            "causal_evidence_ref": causal_ref,
            "causal_evidence_sha256": calibration._canonical_sha256(causal),
            "world_identity": str(causal.get("source", {}).get("world_identity") or ""),
            "cache": cache_report,
            "materialized_worlds": materialized_worlds,
            "coverage_product_sha256": (
                None if coverage_product is None else str(coverage_product.get("product_sha256") or "")
            ),
            "evaluator_identity": dict(selected_opportunity.get("evaluator_identity") or {}),
            "publication_ref": publication_path.name,
            "publication_sha256": _sha256(publication),
            "calculation_started_at": calculation_started_at,
            "issued_at": issued_at,
        })
        receipt_path = _metadata_root(evidence_root) / f"forecast-{operation_id}.json"
        _write_new_json(receipt_path, receipt)
        return _read_forecast_receipt(evidence_root, operation_id)


def _load_forecast_origin(
    conn: sqlite3.Connection,
    *,
    identity: Mapping[str, Any],
    action: str,
    observation_id: str,
) -> tuple[dict[str, Any], Any, Any, dict[str, Any], dict[str, Any], dict[str, Any]]:
    from . import chip_decision as cd
    from . import chip_reservation_forecast as crf

    retained, generation, decision, route, state, permission = _revalidate_registered_origin(
        conn, identity=identity,
    )
    origin_id = _sha256(_origin_key(identity))
    operation_id = _forecast_operation_id(origin_id, action, observation_id)
    receipt_path = _metadata_root(identity["evidence_root"]) / f"forecast-{operation_id}.json"
    if not receipt_path.exists():
        raise ChipEvidenceError("no verified prospective forecast receipt exists for this explicit observation")
    receipt = _read_forecast_receipt(identity["evidence_root"], operation_id)
    if (
        str(receipt.get("origin_id") or "") != origin_id
        or str(receipt.get("observation_id") or "") != str(observation_id)
        or str(receipt.get("identity_sha256") or "") != _sha256(identity)
        or str(receipt.get("action") or "") != action
        or int(receipt.get("planning_event") or -1) != int(identity["planning_event"])
        or str(receipt.get("origin_cutoff") or "") != str(identity["cutoff"])
    ):
        raise ChipEvidenceError("forecast receipt differs from the explicit origin/action/observation")
    intent_path = _metadata_root(identity["evidence_root"]) / f"forecast-intent-{operation_id}.json"
    publication_path = _metadata_root(identity["evidence_root"]) / f"forecast-publication-{operation_id}.json"
    intent = _read_json(intent_path)
    publication = _read_json(publication_path)
    _verify_seal(intent, schema=FORECAST_INTENT_SCHEMA)
    _verify_seal(publication, schema=FORECAST_PUBLICATION_SCHEMA)
    for field, expected_value in {
        "operation_id": operation_id,
        "identity_sha256": _sha256(identity),
        "origin_id": origin_id,
        "observation_id": str(observation_id),
        "action": action,
        "expiry_event": int(receipt.get("expiry_event") or -1),
    }.items():
        if intent.get(field) != expected_value or publication.get(field) != expected_value:
            raise ChipEvidenceError("forecast intent/publication is not bound to the explicit operation")
    if (
        str(publication.get("intent_sha256") or "") != _sha256(intent)
        or str(receipt.get("publication_ref") or "") != publication_path.name
        or str(receipt.get("publication_sha256") or "") != _sha256(publication)
        or str(publication.get("operation_id") or "") != operation_id
        or str(publication.get("identity_sha256") or "") != _sha256(identity)
        or str(publication.get("origin_id") or "") != origin_id
        or str(publication.get("observation_id") or "") != str(observation_id)
        or str(publication.get("action") or "") != action
        or str(intent.get("calculation_started_at") or "")
        != str(publication.get("calculation_started_at") or "")
        or str(receipt.get("calculation_started_at") or "")
        != str(publication.get("calculation_started_at") or "")
        or str(receipt.get("issued_at") or "") != str(publication.get("issued_at") or "")
        or str(receipt.get("made_at") or "") != str(publication.get("issued_at") or "")
    ):
        raise ChipEvidenceError("forecast receipt does not reproduce its immutable completion/publication marker")
    calculation_started_at = str(publication.get("calculation_started_at") or "")
    issued_at = str(publication.get("issued_at") or "")
    if _parse_utc(issued_at, name="forecast issued_at") < _parse_utc(
        calculation_started_at, name="forecast calculation_started_at",
    ):
        raise ChipEvidenceError("forecast publication precedes calculation start")
    causal_path, causal, _forecast_path, raw_forecast = _load_causal_origin(
        identity["evidence_root"], observation_id=observation_id,
        action=action, origin_id=origin_id,
    )
    expected_source = _source_identity(route, generation)
    source = dict(causal.get("source") or {})
    for name, value in expected_source.items():
        if source.get(name) != value:
            raise ChipEvidenceError(f"causal-origin source identity differs from verified origin on {name}")
    if (
        int(causal.get("planning_event") or -1) != int(identity["planning_event"])
        or str(causal.get("origin_cutoff") or "") != str(identity["cutoff"])
        or str(raw_forecast.get("action") or "") != action
        or int(raw_forecast.get("selected_event") or -1) != int(receipt.get("selected_event") or -2)
    ):
        raise ChipEvidenceError("causal origin and forecast receipt disagree on timing/action/event")
    crf.verify_reservation_forecast(
        raw_forecast,
        store_conn=conn,
        expected={
            "action": action,
            "planning_event": int(identity["planning_event"]),
            "origin_cutoff": str(identity["cutoff"]),
            "made_at": str(receipt.get("issued_at") or ""),
            "input_as_of": str(receipt.get("input_as_of") or ""),
            "forecast_mode": crf.FORECAST_MODE_PROSPECTIVE,
            "expiry_event": int(receipt.get("expiry_event") or -1),
            "source_identity": expected_source,
        },
        evidence_verifier=lambda reference: _flat_artifact(identity["evidence_root"], reference)[1],
    )
    snapshot_causality = generation.manifest.get("search_permission_causality") or {}
    if (
        str(receipt.get("input_as_of") or "") != str(identity["cutoff"])
        or not isinstance(snapshot_causality, Mapping)
        or snapshot_causality.get("temporal_status") != "CAUSAL"
        or str(receipt.get("observed_at") or "")
        != str(snapshot_causality.get("snapshot_consistency_at") or "")
        or str(receipt.get("snapshot_available_at") or "")
        != str(snapshot_causality.get("snapshot_capture_completed_at") or "")
        or _parse_utc(receipt.get("observed_at"), name="forecast observed_at")
        > _parse_utc(identity["cutoff"], name="origin cutoff")
    ):
        raise ChipEvidenceError("forecast receipt has missing or post-cutoff observation provenance")
    _parse_utc(receipt.get("snapshot_available_at"), name="forecast snapshot_available_at")
    return retained, generation, decision, route, state, {
        "receipt": receipt,
        "causal_path": causal_path,
        "causal": causal,
        "forecast": raw_forecast,
        "origin_id": origin_id,
        "operation_id": operation_id,
        "permission": permission,
    }


def _required_players_for_event(causal: Mapping[str, Any], event: int) -> list[int]:
    pair = causal.get("counterfactual_pair")
    if not isinstance(pair, Mapping):
        raise ChipEvidenceError("causal origin omits the PLAY/SAVE policy pair")
    required: set[int] = set()
    for role in ("play", "save"):
        arm = pair.get(role)
        if not isinstance(arm, Mapping):
            raise ChipEvidenceError(f"causal origin omits its {role.upper()} policy")
        schedule = arm.get("valuation_schedule")
        if isinstance(schedule, Mapping):
            rows = schedule.get("events") or ()
            for row in rows:
                if int(row.get("event") or -1) == int(event):
                    required.update(int(pid) for pid in row.get("proposed_squad_ids") or ())
        elif int(arm.get("event") or -1) == int(event):
            # The accepted BB/TC causal arm schema binds the selected event
            # directly on each PLAY/SAVE arm; it does not duplicate that
            # event under causal.forecast.selected_event.
            required.update(int(pid) for pid in arm.get("proposed_squad_ids") or ())
    if not required:
        raise ChipEvidenceError(f"the retained policies name no players for GW{int(event)}")
    return sorted(required)


def _read_verified_archive_payload(raw_root: str | Path, record: Mapping[str, Any]) -> dict[str, Any]:
    from . import raw_archive

    relative = str(record.get("relative_path") or "")
    if not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ChipEvidenceError("official archive record has an unsafe blob path")
    if not raw_archive.verify_archived_blob(raw_root, record):
        raise ChipEvidenceError(f"official raw archive blob does not verify: {record.get('capture_id')}")
    archive_path = raw_archive.archive_root(raw_root) / relative
    try:
        value = json.loads(archive_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as failure:
        raise ChipEvidenceError(f"official raw archive is not readable JSON: {failure}") from failure
    if not isinstance(value, dict) and not isinstance(value, list):
        raise ChipEvidenceError("official raw archive payload has an unsupported top-level shape")
    return value


def _official_event_archive(
    conn: sqlite3.Connection,
    *,
    raw_root: str | Path,
    fetch_run_id: int,
    event: int,
) -> dict[str, Any]:
    from . import parsers, planning, raw_archive

    run_row = conn.execute("SELECT * FROM fetch_runs WHERE id=?", (int(fetch_run_id),)).fetchone()
    if run_row is None:
        raise ChipEvidenceError(f"fetch run {int(fetch_run_id)} does not exist")
    run = dict(run_row)
    if str(run.get("status") or "") != "success":
        raise ChipEvidenceError("outcome capture requires the exact successful official fetch run")
    started_at = _parse_utc(run.get("started_at"), name="fetch run started_at")
    finished_at = _parse_utc(run.get("finished_at"), name="successful fetch run finished_at")
    if finished_at < started_at:
        raise ChipEvidenceError("successful fetch run has a reversed started_at/finished_at interval")
    if not run.get("raw_dir") or Path(str(run["raw_dir"])).expanduser().resolve() != Path(raw_root).expanduser().resolve():
        raise ChipEvidenceError("explicit raw archive root differs from the fetch run's retained raw_dir")
    try:
        ok = json.loads(run.get("endpoints_ok") or "[]")
        failed = json.loads(run.get("endpoints_failed") or "[]")
    except (TypeError, ValueError) as failure:
        raise ChipEvidenceError("fetch-run endpoint provenance is malformed") from failure
    required_endpoints = {"bootstrap-static", "fixtures", f"event/{int(event)}/live"}
    if not isinstance(ok, list) or not required_endpoints.issubset({str(item) for item in ok}):
        raise ChipEvidenceError("fetch run did not successfully collect bootstrap, fixtures and target event live")
    if failed not in ([], None):
        raise ChipEvidenceError("fetch run records failed endpoints; refusing to treat it as complete provenance")

    records = [
        dict(item) for item in raw_archive.load_manifest(raw_root)
        if str(item.get("run_id") or "") == str(fetch_run_id)
    ]
    bootstrap_rows = [item for item in records if item.get("source") == "bootstrap_static"]
    fixture_rows = [item for item in records if item.get("source") == "fixtures"]
    live_rows = [item for item in records if (
        str(item.get("source") or "") in {"event_live", f"event_live_{int(event)}"}
        and item.get("event") is not None
        and int(item.get("event")) == int(event)
    )]
    if len(bootstrap_rows) != 1 or len(fixture_rows) != 1 or len(live_rows) != 1:
        raise ChipEvidenceError("fetch run must have exactly one archived bootstrap, fixture and event-live response")
    bootstrap_record, fixture_record, live_record = bootstrap_rows[0], fixture_rows[0], live_rows[0]
    if any(str(item.get("run_id") or "") != str(fetch_run_id) for item in (bootstrap_record, fixture_record, live_record)):
        raise ChipEvidenceError("raw archive records are not bound to the explicit fetch run")
    timestamps = [str(item.get("observed_at") or "") for item in (bootstrap_record, fixture_record, live_record)]
    for timestamp in timestamps:
        _parse_utc(timestamp, name="official archive observed_at")
    if len(set(timestamps)) != 1:
        raise ChipEvidenceError(
            "bootstrap, fixtures and event-live archives do not share the fetch run's observation identity"
        )
    if _parse_utc(timestamps[0], name="shared official archive observed_at") > finished_at:
        raise ChipEvidenceError("official archive observation occurs after successful fetch completion")
    bootstrap_payload = _read_verified_archive_payload(raw_root, bootstrap_record)
    fixture_payload = _read_verified_archive_payload(raw_root, fixture_record)
    live_payload = _read_verified_archive_payload(raw_root, live_record)
    bootstrap_records = parsers.parse_bootstrap(bootstrap_payload, timestamps[0], int(event))
    parsers.validate_bootstrap_payload(bootstrap_payload, bootstrap_records)
    event_raw_rows = [row for row in bootstrap_payload.get("events", ())
                      if isinstance(row, Mapping) and int(row.get("id") or -1) == int(event)]
    if len(event_raw_rows) != 1:
        raise ChipEvidenceError("archived bootstrap does not uniquely identify the target event")
    event_row = event_raw_rows[0]
    if event_row.get("finished") not in (True, 1) or event_row.get("data_checked") not in (True, 1):
        raise ChipEvidenceError("archived official bootstrap does not mark the target event finished and data-checked")
    parsed_fixtures = parsers.parse_fixtures(fixture_payload if isinstance(fixture_payload, list) else [])
    event_fixtures = [row for row in parsed_fixtures if row.event == int(event)]
    if not event_fixtures:
        raise ChipEvidenceError("archived official fixtures contain no fixtures for the target event")
    fixture_ids = [int(row.id) for row in event_fixtures]
    if len(fixture_ids) != len(set(fixture_ids)):
        raise ChipEvidenceError("archived official fixtures repeat a target-event fixture id")
    unfinished = [row.id for row in event_fixtures if row.finished != 1]
    started_unfinished = [row.id for row in event_fixtures if row.started == 1 and row.finished != 1]
    if unfinished or started_unfinished:
        raise ChipEvidenceError(
            f"target event is not final in archived fixtures; unfinished={unfinished}, started_unfinished={started_unfinished}"
        )
    # finished_provisional is intentionally not an independent veto after the
    # official finished + data_checked event finality rule has passed.
    live_rows_parsed = parsers.parse_event_live(live_payload, int(event))
    raw_elements = live_payload.get("elements")
    if not isinstance(raw_elements, list) or not raw_elements or not live_rows_parsed:
        raise ChipEvidenceError("archived event-live payload has no parseable player element rows")
    database_state, state_basis = planning.event_data_state(conn, int(event))
    if database_state != planning.EVENT_STATE_FINAL:
        raise ChipEvidenceError("local official event state is not final/data-checked: " + "; ".join(state_basis))
    return {
        "run": run,
        "run_id": int(fetch_run_id),
        "event": int(event),
        "raw_root": str(Path(raw_root).expanduser().resolve()),
        "records": {"bootstrap": bootstrap_record, "fixtures": fixture_record, "event_live": live_record},
        "payloads": {"bootstrap": bootstrap_payload, "fixtures": fixture_payload, "event_live": live_payload},
        "parsed": {"bootstrap": bootstrap_records, "fixtures": event_fixtures, "event_live": live_rows_parsed},
        "official_final_at": timestamps[0],
        "observed_at": timestamps[0],
        # The complete official fetch was not available to the operator until
        # its successful run finished; raw source observation time stays separate.
        "available_at": str(run["finished_at"]),
    }


def _capture_operation_id(
    origin_id: str,
    observation_id: str,
    event: int,
    fetch_run_id: int,
    payload_sha256: str,
    captured_at: str,
) -> str:
    return _sha256([
        str(origin_id), str(observation_id), int(event), int(fetch_run_id),
        str(payload_sha256), str(captured_at),
    ])


def _verify_capture_receipt(
    conn: sqlite3.Connection,
    receipt: Mapping[str, Any],
    *,
    expected_identity_sha256: str,
    evidence_root: str | Path,
) -> dict[str, Any]:
    from . import outcome_ledger as ol

    _verify_seal(receipt, schema=CAPTURE_RECEIPT_SCHEMA)
    if str(receipt.get("identity_sha256") or "") != expected_identity_sha256:
        raise ChipEvidenceError("capture receipt is bound to another explicit origin identity")
    archive = _official_event_archive(
        conn,
        raw_root=str(receipt.get("raw_root") or ""),
        fetch_run_id=int(receipt.get("fetch_run_id") or -1),
        event=int(receipt.get("realization_event") or -1),
    )
    live_record = archive["records"]["event_live"]
    operation_id = _capture_operation_id(
        str(receipt.get("origin_id") or ""),
        str(receipt.get("observation_id") or ""),
        int(receipt.get("realization_event") or -1),
        int(receipt.get("fetch_run_id") or -1),
        str(live_record.get("payload_sha256") or ""),
        str(archive.get("available_at") or ""),
    )
    if str(receipt.get("operation_id") or "") != operation_id:
        raise ChipEvidenceError("capture receipt operation id does not reproduce from its source identity")
    if (
        str(receipt.get("raw_observed_at") or "") != str(archive.get("observed_at") or "")
        or str(receipt.get("available_at") or "") != str(archive.get("available_at") or "")
        or str(receipt.get("captured_at") or "") != str(archive.get("available_at") or "")
        or str(receipt.get("official_final_at") or "") != str(archive.get("official_final_at") or "")
    ):
        raise ChipEvidenceError("capture receipt conflates raw observation time and completed-fetch availability")
    for name, record in archive["records"].items():
        expected = receipt.get("raw_sources", {}).get(name) or {}
        if any(expected.get(key) != record.get(key) for key in (
            "capture_id", "source", "observed_at", "payload_sha256", "relative_path", "run_id", "event",
        )):
            raise ChipEvidenceError(f"capture receipt {name} archive binding differs from verified raw evidence")
    rows = receipt.get("captures")
    if not isinstance(rows, list):
        raise ChipEvidenceError("capture receipt has no capture list")
    from . import parsers

    event_live = archive["payloads"]["event_live"]
    raw_by_player: dict[int, Mapping[str, Any]] = {}
    for item in event_live.get("elements", ()):
        if not isinstance(item, Mapping):
            continue
        raw_id = item.get("id", item.get("element"))
        if raw_id is None:
            continue
        try:
            player_id = int(raw_id)
        except (TypeError, ValueError):
            continue
        if player_id in raw_by_player:
            raise ChipEvidenceError(f"archived event-live payload repeats player id {player_id}")
        raw_by_player[player_id] = item
    required_ids = [int(value) for value in receipt.get("required_player_ids") or ()]
    expected_captures: dict[int, tuple[str, dict[str, Any]]] = {}
    for player_id in required_ids:
        raw_item = raw_by_player.get(player_id)
        if raw_item is None:
            continue
        fields = parsers.parse_live_event_totals(dict(raw_item))
        if fields and not all(value is None for value in fields.values()):
            payload = ol._capture_fields(fields)
            digest = ol.capture_digest_for(
                grain=ol.GRAIN_PLAYER_EVENT,
                event=int(receipt["realization_event"]),
                player_id=player_id,
                fixture_id=None,
                captured_at=str(archive["available_at"]),
                observation_state=ol.OBSERVATION_FINAL,
                source_name="player_gameweeks_final",
                source_identity=f"player_gameweeks:{int(receipt['realization_event'])}",
                source_payload_sha256=str(live_record["payload_sha256"]),
                archive_capture_id=str(live_record["capture_id"]),
                payload=payload,
            )
            expected_captures[player_id] = (digest, payload)
    if set(expected_captures) != {int(row.get("player_id") or -1) for row in rows}:
        raise ChipEvidenceError("capture receipt player set differs from the archived event-live payload")
    expected_missing = sorted(set(required_ids) - set(expected_captures))
    if [int(value) for value in receipt.get("missing_player_ids") or ()] != expected_missing:
        raise ChipEvidenceError("capture receipt missing-player list differs from archived evidence")
    for item in rows:
        digest = str(item.get("capture_digest") or "")
        player_id = int(item.get("player_id") or -1)
        expected_capture = expected_captures.get(player_id)
        if expected_capture is None or expected_capture[0] != digest:
            raise ChipEvidenceError("capture receipt digest does not reproduce from the archived player totals")
        selected = conn.execute(
            "SELECT * FROM outcome_observation_captures WHERE capture_digest=?", (digest,),
        ).fetchone()
        if selected is None:
            raise ChipEvidenceError(f"capture receipt ledger row is missing: {digest}")
        stored = dict(selected)
        payload = json.loads(stored.get("payload_json") or "{}")
        computed = ol.capture_digest_for(
            grain=str(stored["grain"]), event=int(stored["event"]),
            player_id=int(stored["player_id"]),
            fixture_id=None if stored.get("fixture_id") is None else int(stored["fixture_id"]),
            captured_at=str(stored["captured_at"]),
            observation_state=str(stored["observation_state"]),
            source_name=str(stored["source_name"]),
            source_identity=stored.get("source_identity"),
            source_payload_sha256=stored.get("source_payload_sha256"),
            archive_capture_id=stored.get("archive_capture_id"),
            payload=payload,
        )
        if (
            computed != digest or stored.get("grain") != ol.GRAIN_PLAYER_EVENT
            or stored.get("fixture_id") is not None
            or stored.get("observation_state") != ol.OBSERVATION_FINAL
            or stored.get("source_name") != "player_gameweeks_final"
            or stored.get("source_identity") != f"player_gameweeks:{int(receipt['realization_event'])}"
            or int(stored.get("event") or -1) != int(receipt["realization_event"])
            or int(stored.get("player_id") or -1) != player_id
            or payload != expected_capture[1]
            or str(stored.get("captured_at")) != str(item.get("captured_at"))
            or str(stored.get("source_payload_sha256")) != str(archive["records"]["event_live"]["payload_sha256"])
            or str(stored.get("archive_capture_id")) != str(archive["records"]["event_live"]["capture_id"])
        ):
            raise ChipEvidenceError("retained outcome row is not bound to the verified final event-live archive")
    return archive


def capture_outcome(
    conn: sqlite3.Connection,
    *,
    db_path: str | Path,
    entry_id: int,
    generation_id: str,
    decision_id: str,
    route_id: str,
    planning_event: int,
    cutoff: str,
    action: str,
    observation_id: str,
    realization_event: int,
    fetch_run_id: int,
    raw_root: str | Path,
    evidence_root: str | Path,
) -> dict[str, Any]:
    """Append official FINAL player-event captures from one verified fetch run."""

    from . import database, outcome_ledger as ol, parsers

    action_map = {"BB": "BB", "TC": "TC",
                  "BENCH_BOOST": "BB", "TRIPLE_CAPTAIN": "TC"}
    action = action_map.get(str(action).upper(), str(action))
    if action not in {"BB", "TC"}:
        raise ChipEvidenceError("outcome capture action must be BB or TC")
    identity = _required_identity(
        db_path=db_path, entry_id=entry_id, generation_id=generation_id,
        decision_id=decision_id, route_id=route_id, planning_event=planning_event,
        cutoff=cutoff, evidence_root=evidence_root,
    )
    _retained, _generation, _decision, _route, _state, parent = _load_forecast_origin(
        conn, identity=identity, action=action, observation_id=observation_id,
    )
    if int(realization_event) != int(parent["receipt"].get("selected_event") or -1):
        raise ChipEvidenceError("realization event must equal the event selected by the retained forecast")
    required_players = _required_players_for_event(parent["causal"], int(realization_event))
    archive = _official_event_archive(
        conn, raw_root=raw_root, fetch_run_id=int(fetch_run_id), event=int(realization_event),
    )
    live_payload = archive["payloads"]["event_live"]
    raw_by_player: dict[int, Mapping[str, Any]] = {}
    for item in live_payload.get("elements") or ():
        if not isinstance(item, Mapping):
            continue
        raw_id = item.get("id", item.get("element"))
        try:
            player_id = int(raw_id)
        except (TypeError, ValueError):
            continue
        if player_id in raw_by_player:
            raise ChipEvidenceError(f"event-live payload repeats player id {player_id}")
        raw_by_player[player_id] = item
    totals: dict[int, dict[str, Any]] = {}
    missing_fields: dict[str, list[str]] = {}
    for player_id in required_players:
        raw_item = raw_by_player.get(player_id)
        if raw_item is None:
            continue
        values = parsers.parse_live_event_totals(dict(raw_item))
        if not values:
            continue
        totals[player_id] = values
        missing_fields[str(player_id)] = [
            field for field in ("minutes", "total_points") if field not in values or values[field] is None
        ]

    live_record = archive["records"]["event_live"]
    operation_id = _capture_operation_id(
        parent["origin_id"], observation_id, int(realization_event), int(fetch_run_id),
        str(live_record["payload_sha256"]), archive["available_at"],
    )
    receipt_path = _metadata_root(evidence_root) / f"capture-{operation_id}.json"
    with _operation_lock(evidence_root, f"capture-{operation_id}"):
        if receipt_path.exists():
            existing = _read_json(receipt_path)
            _verify_capture_receipt(
                conn, existing, expected_identity_sha256=_sha256(identity),
                evidence_root=evidence_root,
            )
            return existing

        source_records = archive["records"]
        source_fields = (
            "capture_id", "source", "observed_at", "payload_sha256", "relative_path", "run_id", "event",
        )
        raw_sources = {
            name: {field: record.get(field) for field in source_fields}
            for name, record in source_records.items()
        }
        intent_path = _metadata_root(evidence_root) / f"capture-intent-{operation_id}.json"
        intent = _sealed(CAPTURE_INTENT_SCHEMA, {
            "operation_id": operation_id,
            "identity_sha256": _sha256(identity),
            "origin_id": parent["origin_id"],
            "observation_id": str(observation_id),
            "action": action,
            "realization_event": int(realization_event),
            "fetch_run_id": int(fetch_run_id),
            "payload_sha256": str(live_record["payload_sha256"]),
            "captured_at": archive["available_at"],
        })
        if intent_path.exists():
            previous = _read_json(intent_path)
            _verify_seal(previous, schema=CAPTURE_INTENT_SCHEMA)
            if previous != intent:
                raise ChipEvidenceError("capture retry differs from its immutable exact-source intent")
        else:
            _write_new_json(intent_path, intent)

        rows_to_capture: list[dict[str, Any]] = []
        for player_id, fields in sorted(totals.items()):
            if all(value is None for value in fields.values()):
                continue
            payload = ol._capture_fields(fields)
            digest = ol.capture_digest_for(
                grain=ol.GRAIN_PLAYER_EVENT,
                event=int(realization_event),
                player_id=player_id,
                fixture_id=None,
                captured_at=archive["available_at"],
                observation_state=ol.OBSERVATION_FINAL,
                source_name="player_gameweeks_final",
                source_identity=f"player_gameweeks:{int(realization_event)}",
                source_payload_sha256=str(live_record["payload_sha256"]),
                archive_capture_id=str(live_record["capture_id"]),
                payload=payload,
            )
            rows_to_capture.append({"player_id": player_id, "fields": fields, "payload": payload,
                                    "capture_digest": digest})
        expected_digests = {str(row["capture_digest"]) for row in rows_to_capture}
        present = {
            str(row["capture_digest"])
            for row in ol.observation_captures(
                conn, grain=ol.GRAIN_PLAYER_EVENT, event=int(realization_event),
            )
            if str(row.get("capture_digest") or "") in expected_digests
        }
        if present and present != expected_digests:
            raise ChipEvidenceError("partial committed event capture exists without a verifiable complete receipt")
        inserted_rows: list[dict[str, Any]] = []
        if not present and rows_to_capture:
            with database.write_transaction(conn):
                for row in rows_to_capture:
                    result = ol.capture_observation(
                        conn,
                        grain=ol.GRAIN_PLAYER_EVENT,
                        event=int(realization_event),
                        player_id=int(row["player_id"]),
                        fields=row["fields"],
                        source_name="player_gameweeks_final",
                        official_final_at=archive["official_final_at"],
                        observation_state=ol.OBSERVATION_FINAL,
                        source_identity=f"player_gameweeks:{int(realization_event)}",
                        source_payload_sha256=str(live_record["payload_sha256"]),
                        fetch_run_id=int(fetch_run_id),
                        archive_capture_id=str(live_record["capture_id"]),
                        captured_at=archive["available_at"],
                        backfill=False,
                    )
                    if result.capture_digest != row["capture_digest"] or result.observation_state != ol.OBSERVATION_FINAL:
                        raise ChipEvidenceError("outcome ledger did not retain the expected official FINAL event row")
                    inserted_rows.append({
                        "player_id": int(row["player_id"]),
                        "capture_digest": str(result.capture_digest),
                        "captured_at": archive["available_at"],
                    })
        capture_rows = [
            {"player_id": int(row["player_id"]), "capture_digest": str(row["capture_digest"]),
             "captured_at": archive["available_at"]}
            for row in rows_to_capture
        ]
        receipt = _sealed(CAPTURE_RECEIPT_SCHEMA, {
            "operation_id": operation_id,
            "identity_sha256": _sha256(identity),
            "origin_id": parent["origin_id"],
            "observation_id": str(observation_id),
            "action": action,
            "realization_event": int(realization_event),
            "source_name": "player_gameweeks_final",
            "source_identity": f"player_gameweeks:{int(realization_event)}",
            "fetch_run_id": int(fetch_run_id),
            "raw_root": str(Path(raw_root).expanduser().resolve()),
            "raw_sources": raw_sources,
            "raw_observed_at": archive["observed_at"],
            "available_at": archive["available_at"],
            "captured_at": archive["available_at"],
            "official_final_at": archive["official_final_at"],
            "required_player_ids": required_players,
            "missing_player_ids": sorted(set(required_players) - {int(row["player_id"]) for row in rows_to_capture}),
            "missing_required_fields": missing_fields,
            "captures": capture_rows,
            "unavailable_players_remain_missing": True,
        })
        _write_new_json(receipt_path, receipt)
        _verify_capture_receipt(
            conn, receipt, expected_identity_sha256=_sha256(identity), evidence_root=evidence_root,
        )
        return receipt


def _maturation_key(observation_id: str, realization_event: int) -> str:
    return _sha256([str(observation_id), int(realization_event)])


def _calibration_row_from_artifacts(
    causal_evidence: Mapping[str, Any],
    outcome_record: Mapping[str, Any],
    *,
    evidence_ref: str,
    evidence_sha256: str,
    expiry_event: int,
) -> dict[str, Any]:
    forecast_meta = causal_evidence.get("forecast") or {}
    source = causal_evidence.get("source") or {}
    raw_value = forecast_meta.get("value")
    planning_event = int(causal_evidence.get("planning_event") or -1)
    expiry = int(expiry_event)
    if expiry < planning_event:
        raise ChipEvidenceError("matured artifacts have no valid expiry event")
    label = causal_evidence.get("label") or {}
    return {
        "observation_id": str(causal_evidence.get("observation_id") or ""),
        "action": str(causal_evidence.get("action") or ""),
        "planning_event": planning_event,
        "weeks_to_expiry": int(expiry) - planning_event,
        "origin_cutoff": str(causal_evidence.get("origin_cutoff") or ""),
        "forecast_made_at": str(forecast_meta.get("made_at") or ""),
        "forecast_input_as_of": str(forecast_meta.get("input_as_of") or ""),
        "forecast_mode": str(forecast_meta.get("forecast_mode") or ""),
        "forecast_value": float(raw_value),
        "label_available_at": str(label.get("available_at") or ""),
        "realized_value": float(outcome_record.get("observed_points")),
        "causal_evidence_ref": str(evidence_ref),
        "causal_evidence_sha256": str(evidence_sha256),
    }


def _verified_capture_receipts_for_observation(
    conn: sqlite3.Connection,
    *,
    evidence_root: str | Path,
    identity_sha256: str,
    origin_id: str,
    observation_id: str,
    action: str,
    event: int,
) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    for path in _metadata_root(evidence_root).glob("capture-*.json"):
        value = _read_json(path)
        if value.get("schema") == CAPTURE_INTENT_SCHEMA:
            continue
        if (
            str(value.get("observation_id") or "") != str(observation_id)
            or str(value.get("action") or "") != str(action)
            or int(value.get("realization_event") or -1) != int(event)
            or str(value.get("origin_id") or "") != str(origin_id)
        ):
            continue
        _verify_capture_receipt(
            conn, value, expected_identity_sha256=identity_sha256, evidence_root=evidence_root,
        )
        receipts.append(value)
    if not receipts:
        raise ChipEvidenceError("no verified official outcome-capture receipt exists for this observation/event")
    return receipts


def _validate_maturation_artifact_pair(
    conn: sqlite3.Connection,
    *,
    evidence_root: str | Path,
    evidence_path: Path,
    evidence_value: Mapping[str, Any],
    outcome_path: Path,
    outcome_value: Mapping[str, Any],
    causal_origin: Mapping[str, Any],
    observation_id: str,
    action: str,
    event: int,
    expiry_event: int,
    source_positions: Mapping[int, str],
    capture_receipts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Verify a complete retained label/outcome pair before receipt publication."""

    from . import chip_reservation_calibration as cal

    evidence_digest = cal._canonical_sha256(evidence_value)
    outcome_digest = cal._canonical_sha256(outcome_value)
    if (
        evidence_path.name != f"chip-causal-evidence-{evidence_digest}.json"
        or outcome_path.name != f"chip-outcome-{outcome_digest}.json"
        or str(evidence_value.get("observation_id") or "") != str(observation_id)
        or str(evidence_value.get("action") or "") != str(action)
        or str(outcome_value.get("observation_id") or "") != str(observation_id)
        or str(outcome_value.get("action") or "") != str(action)
        or int(outcome_value.get("realization_event") or -1) != int(event)
    ):
        raise ChipEvidenceError("maturation artifacts do not bind the requested observation/action/event")

    label = evidence_value.get("label")
    outcome_source = outcome_value.get("source")
    if (
        not isinstance(label, Mapping)
        or not isinstance(outcome_source, Mapping)
        or str(label.get("outcome_record_ref") or "") != outcome_path.name
        or str(label.get("outcome_record_sha256") or "") != outcome_digest
        or not isinstance(label.get("outcome_record"), Mapping)
        or _json_bytes(dict(label["outcome_record"])) != _json_bytes(dict(outcome_value))
        or str(label.get("available_at") or "") != str(evidence_value.get("label_available_at") or "")
        or outcome_source.get("available_at") != label.get("available_at")
    ):
        raise ChipEvidenceError("maturation evidence and outcome record are not mutually bound")

    # The label must preserve the canonical point-in-time origin apart from the
    # two fields the finalizer adds. Serialized comparison tolerates JSON tuple/list
    # normalization without weakening the content identity check.
    for key, value in causal_origin.items():
        if key not in {"label", "label_available_at"} and _json_bytes(evidence_value.get(key)) != _json_bytes(value):
            raise ChipEvidenceError(f"matured evidence changed point-in-time origin field {key}")

    calibration_row = _calibration_row_from_artifacts(
        evidence_value, outcome_value,
        evidence_ref=evidence_path.name, evidence_sha256=evidence_digest,
        expiry_event=int(expiry_event),
    )
    verifier = lambda reference: _flat_artifact(evidence_root, reference)[1]
    cal.validate_observation(
        calibration_row, evidence_verifier=verifier, store_conn=conn,
    )
    cal.verify_outcome_record_capture_sources(
        conn, outcome_value,
        source_position_resolver=lambda _record: {str(key): value for key, value in source_positions.items()},
    )
    outcome_digests = {
        str(item.get("capture_digest") or "")
        for item in (outcome_value.get("source") or {}).get("captures") or ()
    }
    retained_digests = {
        str(item.get("capture_digest") or "")
        for capture_receipt in capture_receipts for item in capture_receipt.get("captures") or ()
    }
    if not outcome_digests or not outcome_digests.issubset(retained_digests):
        raise ChipEvidenceError("matured outcome includes captures without verified exact-run archive receipts")
    return {
        "evidence_digest": evidence_digest,
        "outcome_digest": outcome_digest,
        "calibration_row": calibration_row,
    }


def _verify_maturation_receipt(
    conn: sqlite3.Connection,
    receipt: Mapping[str, Any],
    *,
    evidence_root: str | Path,
    identity: Mapping[str, Any],
    source_positions: Mapping[int, str],
) -> dict[str, Any]:
    _verify_seal(receipt, schema=MATURATION_RECEIPT_SCHEMA)
    observation_id = str(receipt.get("observation_id") or "")
    event = int(receipt.get("realization_event") or -1)
    if (
        str(receipt.get("receipt_key") or "") != _maturation_key(observation_id, event)
        or str(receipt.get("identity_sha256") or "") != _sha256(identity)
        or dict(receipt.get("identity") or {}) != dict(identity)
    ):
        raise ChipEvidenceError("maturation receipt key or explicit source identity does not verify")
    _retained, _generation, _decision, _route, _state, parent = _load_forecast_origin(
        conn, identity=identity, action=str(receipt.get("action") or ""),
        observation_id=observation_id,
    )
    if int(parent["receipt"].get("selected_event") or -1) != event:
        raise ChipEvidenceError("maturation receipt event differs from its selected forecast event")
    evidence_path, evidence_value = _flat_artifact(evidence_root, receipt.get("causal_evidence_ref"))
    outcome_path, outcome_value = _flat_artifact(evidence_root, receipt.get("outcome_ref"))
    capture_receipts = _verified_capture_receipts_for_observation(
        conn, evidence_root=evidence_root, identity_sha256=_sha256(identity),
        origin_id=str(parent["origin_id"]), observation_id=observation_id,
        action=str(receipt.get("action") or ""), event=event,
    )
    pair = _validate_maturation_artifact_pair(
        conn,
        evidence_root=evidence_root,
        evidence_path=evidence_path,
        evidence_value=evidence_value,
        outcome_path=outcome_path,
        outcome_value=outcome_value,
        causal_origin=parent["causal"],
        observation_id=observation_id,
        action=str(receipt.get("action") or ""),
        event=event,
        expiry_event=int(parent["receipt"].get("expiry_event") or -1),
        source_positions=source_positions,
        capture_receipts=capture_receipts,
    )
    if (
        str(receipt.get("causal_evidence_sha256") or "") != pair["evidence_digest"]
        or str(receipt.get("outcome_sha256") or "") != pair["outcome_digest"]
        or dict(receipt.get("calibration_row") or {}) != pair["calibration_row"]
    ):
        raise ChipEvidenceError("maturation receipt digests/calibration row differ from verified artifacts")
    return {"receipt": dict(receipt), "calibration_row": pair["calibration_row"]}


def mature_observation(
    conn: sqlite3.Connection,
    *,
    db_path: str | Path,
    entry_id: int,
    generation_id: str,
    decision_id: str,
    route_id: str,
    planning_event: int,
    cutoff: str,
    action: str,
    observation_id: str,
    realization_event: int,
    evidence_root: str | Path,
) -> dict[str, Any]:
    """Finalize one selected future BB/TC outcome at event grain, idempotently."""

    from . import candidate_universe as cu
    from . import generation_store as gs
    from . import chip_reservation_calibration as cal

    action_map = {"BB": "BB", "TC": "TC",
                  "BENCH_BOOST": "BB", "TRIPLE_CAPTAIN": "TC"}
    action = action_map.get(str(action).upper(), str(action))
    if action not in {"BB", "TC"}:
        raise ChipEvidenceError("maturation action must be BB or TC")
    identity = _required_identity(
        db_path=db_path, entry_id=entry_id, generation_id=generation_id,
        decision_id=decision_id, route_id=route_id, planning_event=planning_event,
        cutoff=cutoff, evidence_root=evidence_root,
    )
    receipt_key = _maturation_key(observation_id, int(realization_event))
    receipt_path = _metadata_root(evidence_root) / f"maturation-{receipt_key}.json"
    with _operation_lock(evidence_root, f"maturation-{receipt_key}"):
        retained, generation, _decision, route, _state, parent = _load_forecast_origin(
            conn, identity=identity, action=action, observation_id=observation_id,
        )
        if int(parent["receipt"].get("selected_event") or -1) != int(realization_event):
            raise ChipEvidenceError("realization event is not the event selected by the retained forecast")
        capture_receipts = _verified_capture_receipts_for_observation(
            conn, evidence_root=evidence_root, identity_sha256=_sha256(identity),
            origin_id=parent["origin_id"], observation_id=observation_id,
            action=action, event=int(realization_event),
        )
        snapshot_conn = gs._open_generation_snapshot(generation)
        try:
            pinned_pool = cu.load_pool(snapshot_conn)
            source_positions = {
                int(player_id): str(row["position"])
                for player_id, row in pinned_pool.items()
                if row.get("position") is not None
            }
        finally:
            snapshot_conn.close()
        if receipt_path.exists():
            receipt = _read_json(receipt_path)
            verified = _verify_maturation_receipt(
                conn, receipt, evidence_root=evidence_root, identity=identity,
                source_positions=source_positions,
            )
            return {"receipt": receipt, "calibration_row": verified["calibration_row"]}

        base_causal_path, causal, _forecast_path, _forecast_value = _load_causal_origin(
            evidence_root, observation_id=observation_id, action=action,
            origin_id=parent["origin_id"],
        )
        outcomes: list[tuple[Path, dict[str, Any]]] = []
        labelled_evidence: list[tuple[Path, dict[str, Any]]] = []
        for path in _root(evidence_root).glob("chip-outcome-*.json"):
            value = _read_json(path)
            if str(value.get("observation_id") or "") == str(observation_id):
                outcomes.append((path, value))
        for path in _root(evidence_root).glob("chip-causal-evidence-*.json"):
            value = _read_json(path)
            label = value.get("label")
            if (
                str(value.get("observation_id") or "") == str(observation_id)
                and isinstance(label, Mapping)
            ):
                labelled_evidence.append((path, value))
        if not outcomes and not labelled_evidence:
            outcome_artifact_names = {
                path.name for path in _root(evidence_root).glob("chip-outcome-*.json")
                if path.exists()
            }
            # The canonical finalizer may have crashed after publishing the
            # outcome record but before its labelled evidence.  Any such partial
            # record is detected by observation/event above and fails closed.
            try:
                finalized = cal.finalize_causal_observation(
                    conn,
                    causal,
                    store_conn=conn,
                    realization_event=int(realization_event),
                    evidence_root=evidence_root,
                    evidence_verifier=lambda reference: _flat_artifact(evidence_root, reference)[1],
                    source_position_resolver=lambda _record: {
                        str(key): value for key, value in source_positions.items()
                    },
                )
            except BaseException:
                # The random-ID finalizer is never called again if it left any
                # candidate artifact. A clean pre-publication failure is retryable.
                new_names = {
                    path.name for path in _root(evidence_root).glob("chip-outcome-*.json")
                } - outcome_artifact_names
                if new_names:
                    raise ChipEvidenceError(
                        "canonical finalizer failed after publishing outcome evidence; recovery inspection required"
                    )
                raise
            causal_path = Path(finalized["causal_evidence_path"])
            outcome_path = Path(finalized["outcome_path"])
            causal_value = _read_json(causal_path)
            outcome_value = _read_json(outcome_path)
        elif len(outcomes) == 1 and len(labelled_evidence) == 1:
            # A receipt may have failed to publish after the canonical finalizer
            # wrote both immutable artifacts. Recover only this complete pair;
            # validation below rebinds it to the original forecast and captures.
            outcome_path, outcome_value = outcomes[0]
            causal_path, causal_value = labelled_evidence[0]
        else:
            raise ChipEvidenceError(
                "maturation artifacts are partial or ambiguous without their deterministic receipt; "
                "refusing to finalize or reconstruct a receipt"
            )

        pair = _validate_maturation_artifact_pair(
            conn,
            evidence_root=evidence_root,
            evidence_path=causal_path,
            evidence_value=causal_value,
            outcome_path=outcome_path,
            outcome_value=outcome_value,
            causal_origin=causal,
            observation_id=observation_id,
            action=action,
            event=int(realization_event),
            expiry_event=int(parent["receipt"].get("expiry_event") or -1),
            source_positions=source_positions,
            capture_receipts=capture_receipts,
        )
        causal_digest = pair["evidence_digest"]
        outcome_digest = pair["outcome_digest"]
        calibration_row = pair["calibration_row"]
        receipt = _sealed(MATURATION_RECEIPT_SCHEMA, {
            "receipt_key": receipt_key,
            "identity": identity,
            "identity_sha256": _sha256(identity),
            "origin_id": parent["origin_id"],
            "observation_id": str(observation_id),
            "action": action,
            "realization_event": int(realization_event),
            "causal_evidence_ref": causal_path.name,
            "causal_evidence_sha256": causal_digest,
            "outcome_ref": outcome_path.name,
            "outcome_sha256": outcome_digest,
            "calibration_row": calibration_row,
            "capture_operation_ids": sorted(str(row.get("operation_id") or "") for row in capture_receipts),
            "matured_at": _utc_now(),
        })
        _write_new_json(receipt_path, receipt)
        _verify_maturation_receipt(
            conn, receipt, evidence_root=evidence_root, identity=identity,
            source_positions=source_positions,
        )
        return {"receipt": receipt, "calibration_row": calibration_row}


def load_verified_calibration_handoff(
    conn: sqlite3.Connection,
    *,
    db_path: str | Path,
    entry_id: int,
    generation_id: str,
    decision_id: str,
    route_id: str,
    planning_event: int,
    cutoff: str,
    evidence_root: str | Path,
) -> list[dict[str, Any]]:
    """Return only unique, receipt-verified rows suitable for later calibration intake."""

    from . import candidate_universe as cu
    from . import generation_store as gs

    identity = _required_identity(
        db_path=db_path, entry_id=entry_id, generation_id=generation_id,
        decision_id=decision_id, route_id=route_id, planning_event=planning_event,
        cutoff=cutoff, evidence_root=evidence_root,
    )
    retained, generation, _decision, _route, _state, _permission = _revalidate_registered_origin(
        conn, identity=identity,
    )
    origin_id = _sha256(_origin_key(identity))
    snapshot_conn = gs._open_generation_snapshot(generation)
    try:
        source_positions = {
            int(player_id): str(row["position"])
            for player_id, row in cu.load_pool(snapshot_conn).items()
            if row.get("position") is not None
        }
    finally:
        snapshot_conn.close()
    rows: list[dict[str, Any]] = []
    unique_observation_ids: set[str] = set()
    unique_origin_actions: set[tuple[str, int, str, str]] = set()
    for path in _metadata_root(evidence_root).glob("maturation-*.json"):
        receipt = _read_json(path)
        _verify_seal(receipt, schema=MATURATION_RECEIPT_SCHEMA)
        receipt_identity = receipt.get("identity")
        if not isinstance(receipt_identity, Mapping):
            raise ChipEvidenceError("maturation receipt omits its explicit source identity")
        receipt_origin_id = _sha256(_origin_key(receipt_identity))
        if receipt_origin_id != str(receipt.get("origin_id") or ""):
            raise ChipEvidenceError("maturation receipt origin identity does not reproduce")
        if receipt_origin_id != origin_id:
            continue
        if dict(receipt_identity) != identity:
            raise ChipEvidenceError("maturation receipt for this origin is bound to a different database/evidence root")
        verified = _verify_maturation_receipt(
            conn, receipt, evidence_root=evidence_root, identity=identity,
            source_positions=source_positions,
        )
        row = verified["calibration_row"]
        observation_id = str(row.get("observation_id") or "")
        key = (
            str(generation_id), int(planning_event), str(cutoff), str(row.get("action") or ""),
        )
        if observation_id in unique_observation_ids or key in unique_origin_actions:
            raise ChipEvidenceError("duplicate receipt/observation cannot enter the calibration handoff")
        unique_observation_ids.add(observation_id)
        unique_origin_actions.add(key)
        rows.append(row)
    rows.sort(key=lambda row: (str(row.get("action")), str(row.get("observation_id"))))
    return rows
