"""Evidence-backed execution readiness, separate from reservation calibration.

Reservation calibration values the option to save a chip. This module evaluates
whether each immediate chip-value model has demonstrated prospective accuracy.
It consumes only independently revalidated, matured causal observations.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Sequence

from . import chip_decision as cd
from . import chip_reservation_calibration as crc

READINESS_SCHEMA = "fpl_brain.chip_evaluator_readiness.v1"
READINESS_VERSION = "chip_evaluator_readiness_v1.0.0"
READINESS_POLICY_VERSION = "prospective_action_specific_accuracy_v1"
READINESS_READY = "READY"
READINESS_INSUFFICIENT = "INSUFFICIENT_EVIDENCE"
READINESS_FAILED = "CRITERIA_NOT_MET"
READINESS_INVALID = "CHIP_EVALUATOR_READINESS_INVALID"
READINESS_CURRENT_EVALUATION_REFUSED = "CHIP_EVALUATOR_CURRENT_EVALUATION_REFUSED"

# The minimum is the locked calibration validation sample size. Error tolerances
# differ by the actual comparison: BB is one event, FH is the four-event route
# delta, and WC is its independently certified 6-10 event weighted value.
ACTION_CRITERIA: dict[str, dict[str, float | int]] = {
    cd.CHIP_ACTION_BB: {
        "minimum_matured_origins": 30,
        "minimum_distinct_decisions": 20,
        "maximum_mae_points": 2.0,
        "maximum_absolute_bias_points": 1.5,
        "minimum_interval_coverage": 0.80,
    },
    cd.CHIP_ACTION_FH: {
        "minimum_matured_origins": 30,
        "minimum_distinct_decisions": 20,
        "maximum_mae_points": 4.0,
        "maximum_absolute_bias_points": 1.5,
        "minimum_interval_coverage": 0.80,
    },
    cd.CHIP_ACTION_WC: {
        "minimum_matured_origins": 30,
        "minimum_distinct_decisions": 20,
        "maximum_mae_points": 5.0,
        "maximum_absolute_bias_points": 1.5,
        "minimum_interval_coverage": 0.80,
    },
}


class EvaluatorReadinessError(ValueError):
    """The readiness evidence is invalid, incompatible or insufficient."""


def _canonical(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False, default=str,
    ).encode("utf-8")


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical(payload)).hexdigest()


def _time(value: Any, *, name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError) as failure:
        raise EvaluatorReadinessError(f"{name} is not an ISO timestamp") from failure
    if parsed.tzinfo is None:
        raise EvaluatorReadinessError(f"{name} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _sample_rows(
    observations: Sequence[Mapping[str, Any]],
    *,
    action: str,
    evaluator_version: str,
    evidence_verifier: Callable[[str], Mapping[str, Any]] | None,
    evaluation_cutoff: str,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    if action not in ACTION_CRITERIA:
        raise EvaluatorReadinessError(f"{action} has no evaluator execution-readiness policy")
    cutoff = _time(evaluation_cutoff, name="evaluation_cutoff")
    if evidence_verifier is None:
        raise EvaluatorReadinessError("a retained causal-evidence verifier is required")
    rows: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    seen: set[str] = set()
    origins: set[tuple[int, str]] = set()
    decisions: set[str] = set()
    for raw in observations:
        observation_id = str(raw.get("observation_id") or "")
        try:
            row = crc.validate_observation(raw, evidence_verifier=evidence_verifier)
            if observation_id in seen:
                raise EvaluatorReadinessError("duplicate observation id")
            seen.add(observation_id)
            if row.action != action:
                raise EvaluatorReadinessError("observation action differs from readiness action")
            if row.forecast_mode != "PROSPECTIVE":
                raise EvaluatorReadinessError("historical replay cannot establish evaluator readiness")
            if row.label_available_at > cutoff:
                continue
            if row.evaluator_version != evaluator_version:
                raise EvaluatorReadinessError(
                    f"evaluator version {row.evaluator_version!r} differs from {evaluator_version!r}"
                )
            if (
                row.evaluator_forecast_value is None
                or row.evaluator_interval_low is None
                or row.evaluator_interval_high is None
                or not row.source_decision_id
            ):
                raise EvaluatorReadinessError("observation lacks evaluator forecast, interval or source decision")
            origin_key = (row.planning_event, row.origin_cutoff.isoformat())
            if origin_key in origins:
                raise EvaluatorReadinessError("duplicate planning origin is not independent readiness evidence")
            origins.add(origin_key)
            if row.source_decision_id in decisions:
                raise EvaluatorReadinessError("duplicate source decision is not an independent readiness origin")
            decisions.add(row.source_decision_id)
            rows.append({
                "observation_id": row.observation_id,
                "action": row.action,
                "source_decision_id": row.source_decision_id,
                "origin_cutoff": row.origin_cutoff.isoformat().replace("+00:00", "Z"),
                "label_available_at": row.label_available_at.isoformat().replace("+00:00", "Z"),
                "evaluator_version": row.evaluator_version,
                "evaluator_forecast_value": float(row.evaluator_forecast_value),
                "realized_value": float(row.realized_value),
                "interval_low": float(row.evaluator_interval_low),
                "interval_high": float(row.evaluator_interval_high),
                "causal_evidence_ref": row.evidence_ref,
                "causal_evidence_sha256": row.evidence_sha256,
            })
        except Exception as failure:
            rejected.append({
                "observation_id": observation_id,
                "reason": f"{type(failure).__name__}: {failure}",
            })
    return sorted(rows, key=lambda item: (item["origin_cutoff"], item["observation_id"])), rejected


def _evaluate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    action: str,
    evaluator_version: str,
    evaluation_cutoff: str,
    rejected: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    criteria = dict(ACTION_CRITERIA[action])
    residuals = [
        float(row["realized_value"]) - float(row["evaluator_forecast_value"])
        for row in rows
    ]
    mae = sum(abs(value) for value in residuals) / len(residuals) if residuals else None
    bias = sum(residuals) / len(residuals) if residuals else None
    coverage = (
        sum(
            1 for row in rows
            if float(row["evaluator_forecast_value"]) + float(row["interval_low"])
            <= float(row["realized_value"])
            <= float(row["evaluator_forecast_value"]) + float(row["interval_high"])
        ) / len(rows)
        if rows else None
    )
    distinct_decisions = len({str(row["source_decision_id"]) for row in rows})
    enough = (
        len(rows) >= int(criteria["minimum_matured_origins"])
        and distinct_decisions >= int(criteria["minimum_distinct_decisions"])
    )
    passes = bool(
        enough
        and not rejected
        and mae is not None and mae <= float(criteria["maximum_mae_points"])
        and bias is not None and abs(bias) <= float(criteria["maximum_absolute_bias_points"])
        and coverage is not None and coverage >= float(criteria["minimum_interval_coverage"])
    )
    if rejected:
        status = READINESS_INVALID
    elif not enough:
        status = READINESS_INSUFFICIENT
    else:
        status = READINESS_READY if passes else READINESS_FAILED
    body = {
        "schema": READINESS_SCHEMA,
        "version": READINESS_VERSION,
        "policy_version": READINESS_POLICY_VERSION,
        "evidence_class": "PRODUCTION",
        "action": action,
        "evaluator_version": str(evaluator_version),
        "evaluation_cutoff": _time(evaluation_cutoff, name="evaluation_cutoff").isoformat().replace("+00:00", "Z"),
        "criteria": criteria,
        "status": status,
        "execution_permitted": passes,
        "metrics": {
            "matured_origins": len(rows),
            "distinct_source_decisions": distinct_decisions,
            "mae_points": mae,
            "bias_points": bias,
            "interval_coverage": coverage,
        },
        "evidence_manifest": [dict(row) for row in rows],
        "rejected_observations": [dict(row) for row in rejected],
    }
    body["artifact_sha256"] = canonical_sha256(body)
    return body


def build_evaluator_readiness(
    observations: Sequence[Mapping[str, Any]],
    *,
    action: str,
    evaluator_version: str,
    evidence_verifier: Callable[[str], Mapping[str, Any]] | None,
    evaluation_cutoff: str,
) -> dict[str, Any]:
    """Validate the causal observations and produce an action-specific gate."""

    rows, rejected = _sample_rows(
        observations,
        action=action,
        evaluator_version=evaluator_version,
        evidence_verifier=evidence_verifier,
        evaluation_cutoff=evaluation_cutoff,
    )
    artifact = _evaluate_rows(
        rows,
        action=action,
        evaluator_version=evaluator_version,
        evaluation_cutoff=evaluation_cutoff,
        rejected=rejected,
    )
    artifact.pop("artifact_sha256", None)
    artifact["causal_observations"] = [dict(row) for row in observations]
    artifact["artifact_sha256"] = canonical_sha256(artifact)
    return artifact


def verify_evaluator_readiness(
    artifact: Mapping[str, Any],
    observations: Sequence[Mapping[str, Any]] | None = None,
    *,
    action: str,
    evaluator_version: str,
    evidence_verifier: Callable[[str], Mapping[str, Any]] | None,
) -> dict[str, Any]:
    """Rebuild the readiness claim from verified evidence and locked criteria."""

    body = dict(artifact)
    identity = str(body.pop("artifact_sha256", ""))
    if (
        body.get("schema") != READINESS_SCHEMA
        or body.get("version") != READINESS_VERSION
        or identity != canonical_sha256(body)
    ):
        raise EvaluatorReadinessError("readiness artifact schema or digest does not verify")
    if body.get("action") != action or body.get("evaluator_version") != evaluator_version:
        raise EvaluatorReadinessError("readiness artifact action/evaluator version is incompatible")
    if body.get("evidence_class") != "PRODUCTION":
        raise EvaluatorReadinessError("fixture readiness evidence cannot grant production permission")
    retained_observations = (
        [dict(row) for row in observations]
        if observations is not None
        else artifact.get("causal_observations")
    )
    if not isinstance(retained_observations, list):
        raise EvaluatorReadinessError("readiness artifact omits its causal-observation manifest")
    if observations is not None and [dict(row) for row in observations] != artifact.get("causal_observations"):
        raise EvaluatorReadinessError("readiness observations differ from the retained manifest")
    reproduced = build_evaluator_readiness(
        retained_observations,
        action=action,
        evaluator_version=evaluator_version,
        evidence_verifier=evidence_verifier,
        evaluation_cutoff=str(body.get("evaluation_cutoff") or ""),
    )
    if reproduced != dict(artifact):
        raise EvaluatorReadinessError("readiness artifact does not reproduce from retained causal evidence")
    return {
        "verified": True,
        "execution_permitted": bool(reproduced.get("execution_permitted")),
        "artifact_sha256": identity,
    }


def apply_verified_readiness(
    evaluation: cd.ChipEvaluation,
    artifact: Mapping[str, Any],
    observations: Sequence[Mapping[str, Any]] | None,
    *,
    current_cutoff: str,
    evidence_verifier: Callable[[str], Mapping[str, Any]] | None,
) -> tuple[cd.ChipEvaluation, dict[str, Any]]:
    """Apply independently verified readiness without touching evaluator math."""

    verification = verify_evaluator_readiness(
        artifact,
        observations,
        action=evaluation.action,
        evaluator_version=evaluation.evaluator_version,
        evidence_verifier=evidence_verifier,
    )
    if not verification["execution_permitted"]:
        return evaluation, {
            "status": artifact.get("status"),
            "execution_permitted": False,
            "artifact_sha256": verification["artifact_sha256"],
            "artifact": dict(artifact),
            "reason_code": READINESS_INSUFFICIENT
            if artifact.get("status") == READINESS_INSUFFICIENT else READINESS_FAILED,
        }
    from dataclasses import replace

    from . import chip_bench_boost as bb, chip_free_hit as fh, chip_wildcard as wc

    accepted_reason_codes = {
        cd.CHIP_ACTION_BB: {
            bb.DIAG_BB_EVALUATED,
            bb.DIAG_BB_REVIEW_ONLY,
            bb.DIAG_BB_BENCH_ALREADY_RECOVERED,
            bb.DIAG_BB_NO_BENCH_VALUE,
            bb.DIAG_BB_CAPTAIN_APPEARANCE_UNCERTAIN,
            bb.DIAG_BB_VICE_FALLBACK_MATERIAL,
            bb.DIAG_BB_NO_ARMBAND_POSSIBLE,
            bb.DIAG_BB_INPUT_UNCERTAINTY,
        },
        cd.CHIP_ACTION_FH: {
            fh.DIAG_FH_EVALUATED,
            fh.DIAG_FH_REVIEW_ONLY,
            fh.DIAG_FH_NO_TEMPORARY_GAIN,
            fh.DIAG_FH_BOUNDED_SEARCH,
            fh.DIAG_FH_INPUT_UNCERTAINTY,
        },
        cd.CHIP_ACTION_WC: {
            wc.WC_REASON_PLAY_DESCRIPTION,
            wc.WC_REASON_REVIEW_ONLY,
            wc.WC_REASON_POSITIVE,
            wc.WC_REASON_NOT_COMPETITIVE,
        },
    }
    evaluator_version_evidence_key = {
        cd.CHIP_ACTION_BB: "evaluator_version",
        cd.CHIP_ACTION_FH: "free_hit_evaluator_version",
        cd.CHIP_ACTION_WC: "wildcard_evaluator_version",
    }
    allowed = accepted_reason_codes.get(evaluation.action, set())
    reasons = set(evaluation.reason_codes)
    evidence = evaluation.evidence if isinstance(evaluation.evidence, Mapping) else {}
    try:
        mean_uplift = evaluation.mean_uplift
        finite_uplift = mean_uplift is not None and math.isfinite(mean_uplift)
    except (TypeError, ValueError, OverflowError):
        mean_uplift = None
        finite_uplift = False
    try:
        planning_event = int(evidence.get("planning_event"))
        horizon_events = tuple(int(event) for event in evidence.get("horizon_events", ()))
    except (TypeError, ValueError):
        planning_event = 0
        horizon_events = ()
    try:
        readiness_cutoff = _time(artifact.get("evaluation_cutoff"), name="readiness evaluation_cutoff")
        assessment_cutoff = _time(current_cutoff, name="current assessment cutoff")
        readiness_is_origin_safe = readiness_cutoff <= assessment_cutoff
    except EvaluatorReadinessError:
        readiness_is_origin_safe = False
    current_evaluation_is_valid = (
        bool(allowed)
        and not evaluation.execution_permitted
        and bool(reasons & allowed)
        and reasons <= allowed
        and finite_uplift
        and evaluation.data_snapshot_bound
        and isinstance(evidence, Mapping)
        and bool(str(evidence.get("certification_identity") or "").strip())
        and bool(str(evidence.get("data_snapshot_sha256") or "").strip())
        and planning_event > 0
        and horizon_events == cd.canonical_chip_horizon(planning_event)
        and readiness_is_origin_safe
        and evidence.get(evaluator_version_evidence_key.get(evaluation.action, ""))
        == evaluation.evaluator_version
    )
    if not current_evaluation_is_valid:
        return evaluation, {
            "status": READINESS_CURRENT_EVALUATION_REFUSED,
            "execution_permitted": False,
            "artifact_sha256": verification["artifact_sha256"],
            "artifact": dict(artifact),
            "reason_code": READINESS_CURRENT_EVALUATION_REFUSED,
        }

    review_only_reasons = {
        cd.CHIP_ACTION_BB: {bb.DIAG_BB_REVIEW_ONLY},
        cd.CHIP_ACTION_FH: {fh.DIAG_FH_REVIEW_ONLY},
        cd.CHIP_ACTION_WC: {wc.WC_REASON_REVIEW_ONLY},
    }[evaluation.action]
    evidence = {
        **dict(evidence),
        "evaluator_readiness": {
            "artifact_sha256": verification["artifact_sha256"],
            "policy_version": READINESS_POLICY_VERSION,
            "action": evaluation.action,
            "evaluator_version": evaluation.evaluator_version,
        },
        "execution_permission_source": "VERIFIED_ACTION_SPECIFIC_READINESS",
    }
    ready = replace(
        evaluation,
        reason_codes=tuple(sorted({
            *(reason for reason in evaluation.reason_codes if reason not in review_only_reasons),
            "CHIP_EVALUATOR_READINESS_VERIFIED",
        })),
        execution_permitted=True,
        calibration_status=cd.CALIBRATION_CALIBRATED,
        evidence=evidence,
    )
    return ready, {
        "status": READINESS_READY,
        "execution_permitted": True,
        "artifact_sha256": verification["artifact_sha256"],
        "artifact": dict(artifact),
    }


def build_fixture_readiness(
    rows: Sequence[Mapping[str, Any]],
    *,
    action: str,
    evaluator_version: str,
    evaluation_cutoff: str = "2026-10-01T00:00:00Z",
) -> dict[str, Any]:
    """Explicit synthetic-only helper for metric-path tests; production rejects it."""

    normalized = [dict(row) for row in rows]
    fixture = _evaluate_rows(
        normalized,
        action=action,
        evaluator_version=evaluator_version,
        evaluation_cutoff=evaluation_cutoff,
    )
    fixture["evidence_class"] = "FIXTURE_ONLY"
    fixture["production_usable"] = False
    fixture.pop("artifact_sha256", None)
    fixture["artifact_sha256"] = canonical_sha256(fixture)
    return fixture


def verify_retained_readiness_integrity(artifact: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute the immutable metric claim without an external ledger handle."""

    body = dict(artifact)
    identity = str(body.pop("artifact_sha256", ""))
    if (
        body.get("schema") != READINESS_SCHEMA
        or body.get("version") != READINESS_VERSION
        or body.get("policy_version") != READINESS_POLICY_VERSION
        or identity != canonical_sha256(body)
    ):
        raise EvaluatorReadinessError("retained readiness schema, policy or digest does not verify")
    action = str(body.get("action") or "")
    evaluator_version = str(body.get("evaluator_version") or "")
    if action not in ACTION_CRITERIA or body.get("criteria") != ACTION_CRITERIA[action]:
        raise EvaluatorReadinessError("retained readiness criteria do not match the locked action policy")
    if body.get("evidence_class") != "PRODUCTION":
        raise EvaluatorReadinessError("fixture readiness cannot be retained as production permission")
    rows = body.get("evidence_manifest")
    rejected = body.get("rejected_observations")
    if not isinstance(rows, list) or not isinstance(rejected, list):
        raise EvaluatorReadinessError("retained readiness evidence manifest is malformed")
    reproduced = _evaluate_rows(
        rows,
        action=action,
        evaluator_version=evaluator_version,
        evaluation_cutoff=str(body.get("evaluation_cutoff") or ""),
        rejected=rejected,
    )
    reproduced["causal_observations"] = list(body.get("causal_observations") or ())
    reproduced["artifact_sha256"] = canonical_sha256(reproduced)
    if reproduced != dict(artifact):
        raise EvaluatorReadinessError("retained readiness metrics do not reproduce from their evidence manifest")
    return {
        "verified": True,
        "execution_permitted": bool(reproduced.get("execution_permitted")),
        "artifact_sha256": identity,
    }
