"""One production search-permission gate over persisted generation evidence.

PE-9 Amendment 2 keeps the certified generation as the production authority.
This module does not accept a capability, certificate object, permission flag, or
caller-provided history audit. It revalidates the selected generation and audits
its exact pinned snapshot before a production search begins.
"""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Mapping
from typing import Any

from . import execution_snapshot as es, history_completeness as hc

SEARCH_PERMISSION_EVALUATION_SCHEMA = "fpl_brain.production_search_permission.v1"
SEARCH_PERMISSION_BINDING_SCHEMA = "fpl_brain.production_search_permission_binding.v1"
PRODUCTION_SEARCH_PERMISSION_DENIED = "PRODUCTION_SEARCH_PERMISSION_DENIED"


class SearchPermissionRefused(RuntimeError):
    """A production search failed the derived permission conditions."""

    token = PRODUCTION_SEARCH_PERMISSION_DENIED

    def __init__(self, reasons: list[str], *, evaluation: Mapping[str, Any] | None = None) -> None:
        self.reasons = [str(reason) for reason in reasons]
        self.evaluation = dict(evaluation or {})
        super().__init__(f"{self.token}: " + "; ".join(self.reasons))


def _canonical_digest(payload: Mapping[str, Any]) -> str:
    from . import generation_store as gs

    return "sha256:" + hashlib.sha256(gs.canonical_manifest_bytes(payload)).hexdigest()


def evaluation_digest(evaluation: Mapping[str, Any]) -> str:
    """Recompute the digest of a retained evaluation block."""

    body = {str(key): value for key, value in evaluation.items() if key != "evaluation_sha256"}
    return _canonical_digest(body)


def _unresolved_history_audit(planning_event: int, cutoff: str, failure: Exception) -> dict[str, Any]:
    return {
        "schema": hc.HISTORY_COMPLETENESS_SCHEMA,
        "planning_event": int(planning_event),
        "cutoff": str(cutoff),
        "complete": False,
        "blocker": hc.CERTIFIED_PREDICTION_INPUT_HISTORY_INCOMPLETE,
        "reasons": [f"UNRESOLVED: {type(failure).__name__}: {failure}"],
        "detail": "the origin snapshot history-completeness audit could not be evaluated",
    }


def require_search_permission(
    conn: sqlite3.Connection,
    generation_id: str,
    *,
    expected_origin_planning_event: int | None = None,
    expected_origin_cutoff: str | None = None,
    expected_snapshot_sha256: str | None = None,
) -> dict[str, Any]:
    """Derive and require permission from one verified generation and its snapshot.

    ``expected_*`` values only assert that a chip/opportunity caller is still using
    the origin it claims to use. They can make the gate stricter, never permissive.
    No permission-relevant object is accepted from the caller.
    """

    from . import generation_store as gs

    reasons: list[str] = []
    try:
        generation = gs.load_generation(conn, str(generation_id))
        report = gs.verify_generation(conn, generation.generation_id)
    except Exception as failure:
        raise SearchPermissionRefused(
            [f"origin generation verification failed: {type(failure).__name__}: {failure}"]
        ) from failure
    if report.get("verified") is not True:
        reasons.append("the persisted origin generation did not pass independent integrity verification")

    origin_event = int(generation.planning_event)
    origin_cutoff = str(generation.cutoff)
    snapshot = dict(generation.snapshot)
    snapshot_sha256 = str(snapshot.get("sha256") or "")
    causal = generation.manifest.get("search_permission_causality")
    causal_is_valid = (
        isinstance(causal, Mapping)
        and causal.get("schema") == es.GENERATION_CAUSALITY_SCHEMA
        and causal.get("temporal_status") == "CAUSAL"
        and report.get("causal_evidence_reproduced") is True
    )

    if expected_origin_planning_event is not None and int(expected_origin_planning_event) != origin_event:
        reasons.append(
            f"origin planning event {origin_event} differs from asserted source event "
            f"{int(expected_origin_planning_event)}"
        )
    if expected_origin_cutoff is not None and str(expected_origin_cutoff) != origin_cutoff:
        reasons.append(
            f"origin cutoff {origin_cutoff} differs from asserted source cutoff {expected_origin_cutoff}"
        )
    if expected_snapshot_sha256 is not None and str(expected_snapshot_sha256) != snapshot_sha256:
        reasons.append("origin snapshot digest differs from the asserted source snapshot")

    snapshot_error: str | None = None
    try:
        source_conn = gs._open_generation_snapshot(generation)
    except Exception as failure:
        source_conn = None
        snapshot_error = f"{type(failure).__name__}: {failure}"
    if source_conn is not None:
        try:
            try:
                history_audit = hc.audit_history_completeness(
                    source_conn, planning_event=origin_event, cutoff=origin_cutoff
                )
                if (
                    int(history_audit.get("planning_event") or -1) != origin_event
                    or str(history_audit.get("cutoff") or "") != origin_cutoff
                ):
                    history_audit = _unresolved_history_audit(
                        origin_event,
                        origin_cutoff,
                        ValueError("history audit returned a different origin event or cutoff"),
                    )
            except Exception as failure:  # audit failure is always a refusal
                history_audit = _unresolved_history_audit(origin_event, origin_cutoff, failure)
        finally:
            source_conn.close()
    else:
        history_audit = _unresolved_history_audit(
            origin_event,
            origin_cutoff,
            RuntimeError(snapshot_error or "the pinned snapshot could not be opened"),
        )

    if not causal_is_valid:
        if isinstance(causal, Mapping):
            causal_reasons = causal.get("reasons") or []
            reasons.extend(str(item) for item in causal_reasons)
            if not causal_reasons:
                reasons.append(
                    "origin snapshot/execution timestamps are missing, unresolved, or did not reproduce"
                )
        else:
            reasons.append("the generation carries no retained causal timestamp evidence")

    dependency_validation = (
        "COHERENT"
        if report.get("dependency_closure_reproduced") is True
        and report.get("bundle_identities_reproduced") is True
        else "UNRESOLVED"
    )
    if snapshot_error:
        reasons.append(f"snapshot could not be opened: {snapshot_error}")

    # Reuse the certification rule itself so production does not grow a second
    # interpretation of temporal, dependency, horizon, snapshot and history policy.
    try:
        from .four_gw_decision import decide_search_permission

        permitted, predicate_reasons = decide_search_permission(
            temporal_status="CAUSAL" if causal_is_valid else "UNRESOLVED",
            dependency_validation=dependency_validation,
            horizon_status=report.get("horizon_state"),
            data_snapshot_sha256=snapshot_sha256,
            history_completeness=history_audit,
            snapshot_error=snapshot_error,
        )
    except Exception as failure:  # canonical predicate failure is never permission
        permitted = False
        predicate_reasons = [
            f"canonical search-permission predicate could not be evaluated: "
            f"{type(failure).__name__}: {failure}"
        ]
    reasons.extend(str(reason) for reason in predicate_reasons)
    # Keep ordered reasons stable while avoiding duplicate explanations that the
    # canonical predicate and the evidence-specific checks may both identify.
    reasons = list(dict.fromkeys(reasons))
    permitted = bool(permitted and not reasons)

    causal_digest = (
        _canonical_digest(dict(causal)) if isinstance(causal, Mapping) else None
    )
    body: dict[str, Any] = {
        "schema": SEARCH_PERMISSION_EVALUATION_SCHEMA,
        "permitted": permitted,
        "reasons": reasons,
        "origin": {
            "generation_id": generation.generation_id,
            "generation_manifest_sha256": generation.generation_id,
            "planning_event": origin_event,
            "cutoff": origin_cutoff,
            "horizon_kind": generation.horizon_kind,
            "events": list(generation.events),
            "data_snapshot_sha256": snapshot_sha256,
            "execution_run_uuid": snapshot.get("execution_run_uuid"),
            "predictive_code_snapshot_sha256": generation.manifest.get("code_snapshot_sha256"),
        },
        "conditions": {
            "temporal_status": "CAUSAL" if causal_is_valid else "UNRESOLVED",
            "causal_evidence_sha256": causal_digest,
            "dependency_validation": dependency_validation,
            "horizon_status": report.get("horizon_state"),
            "data_snapshot_sha256": snapshot_sha256,
            "snapshot_identity": report.get("snapshot_identity"),
            "snapshot_error": snapshot_error,
            "history_completeness": dict(history_audit),
        },
    }
    evaluation = {**body, "evaluation_sha256": _canonical_digest(body)}
    if not permitted:
        raise SearchPermissionRefused(reasons, evaluation=evaluation)
    return evaluation


def decision_record_binding(evaluation: Mapping[str, Any]) -> dict[str, Any]:
    """Minimal record-side binding for the artifact's full permission block."""

    origin = evaluation.get("origin") or {}
    return {
        "schema": SEARCH_PERMISSION_BINDING_SCHEMA,
        "evaluation_sha256": evaluation.get("evaluation_sha256"),
        "generation_id": origin.get("generation_id"),
        "generation_manifest_sha256": origin.get("generation_manifest_sha256"),
        "origin_planning_event": origin.get("planning_event"),
        "origin_cutoff": origin.get("cutoff"),
        "origin_horizon_kind": origin.get("horizon_kind"),
        "data_snapshot_sha256": origin.get("data_snapshot_sha256"),
    }


def verify_recorded_evaluation(
    conn: sqlite3.Connection,
    *,
    generation_id: str,
    evaluation: Mapping[str, Any],
    record_binding: Mapping[str, Any],
) -> bool:
    """Recompute and compare a new-format record's permission evidence."""

    if evaluation.get("schema") != SEARCH_PERMISSION_EVALUATION_SCHEMA:
        return False
    if evaluation.get("evaluation_sha256") != evaluation_digest(evaluation):
        return False
    try:
        expected = require_search_permission(conn, generation_id)
    except Exception:
        return False
    if dict(evaluation) != expected:
        return False
    return dict(record_binding) == decision_record_binding(expected)


__all__ = [
    "PRODUCTION_SEARCH_PERMISSION_DENIED",
    "SEARCH_PERMISSION_BINDING_SCHEMA",
    "SEARCH_PERMISSION_EVALUATION_SCHEMA",
    "SearchPermissionRefused",
    "decision_record_binding",
    "evaluation_digest",
    "require_search_permission",
    "verify_recorded_evaluation",
]
