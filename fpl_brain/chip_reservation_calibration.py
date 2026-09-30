"""Validated, walk-forward calibration for chip reservation values.

This module only calibrates the future-option estimate consumed by the chip
arbiter.  It does not change an evaluator's independent ``execution_permitted``
gate.  Evidence is accepted only through a caller-supplied artifact verifier;
without one, or without matured paired causal records, the result stays
UNCALIBRATED.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Sequence

from . import chip_decision as cd

CALIBRATION_SCHEMA = "fpl_brain.chip_reservation_calibration.v1"
CAUSAL_EVIDENCE_SCHEMA = "fpl_brain.chip_reservation_causal_evidence.v1"
CALIBRATION_VERSION = "chip_reservation_walkforward_v1.0.0"
CAUSAL_EVIDENCE_POLICY = "SAME_SCENARIO_PAIRED_PLAY_SAVE_TEMPORAL_LABELS_v1"

# These criteria are fixed in code before labels are evaluated. A caller cannot
# loosen them in a record after seeing the results.
MIN_TRAINING_ORIGINS = 20
MIN_VALIDATION_ORIGINS = 30
MIN_MAE_IMPROVEMENT = 0.05
MAX_ABSOLUTE_BIAS = 1.5
MIN_90_INTERVAL_COVERAGE = 0.80

DIAG_CALIBRATION_EVIDENCE_INVALID = "CHIP_RESERVATION_CAUSAL_EVIDENCE_INVALID"
DIAG_CALIBRATION_INSUFFICIENT = "CHIP_RESERVATION_CALIBRATION_INSUFFICIENT_CAUSAL_EVIDENCE"
DIAG_CALIBRATION_NOT_VALIDATED = "CHIP_RESERVATION_CALIBRATION_NOT_VALIDATED"


class ReservationCalibrationError(ValueError):
    pass


def _utc(value: Any, *, name: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ReservationCalibrationError(f"{name} is missing")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as failure:
        raise ReservationCalibrationError(f"{name} is not an ISO timestamp") from failure
    if parsed.tzinfo is None:
        raise ReservationCalibrationError(f"{name} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _number(value: Any, *, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as failure:
        raise ReservationCalibrationError(f"{name} is not numeric") from failure
    if not math.isfinite(parsed):
        raise ReservationCalibrationError(f"{name} is not finite")
    return parsed


def _canonical_sha256(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False, default=str).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _is_sha256(value: Any) -> bool:
    text = str(value or "")
    if text.startswith("sha256:"):
        text = text[7:]
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text.lower())


def expiry_bucket(weeks_to_expiry: int | None) -> str:
    if weeks_to_expiry is None:
        return "UNKNOWN"
    weeks = max(0, int(weeks_to_expiry))
    if weeks == 0:
        return "0"
    if weeks <= 4:
        return "1-4"
    return "5+"


@dataclass(frozen=True)
class VerifiedOpportunity:
    observation_id: str
    action: str
    planning_event: int
    weeks_to_expiry: int
    origin_cutoff: datetime
    forecast_made_at: datetime
    label_available_at: datetime
    forecast_value: float
    realized_value: float
    evidence_ref: str
    evidence_sha256: str

    @property
    def bucket(self) -> str:
        return expiry_bucket(self.weeks_to_expiry)


def _verify_pair(evidence: Mapping[str, Any], row: Mapping[str, Any]) -> None:
    if evidence.get("schema") != CAUSAL_EVIDENCE_SCHEMA:
        raise ReservationCalibrationError("causal evidence has an unsupported schema")
    if evidence.get("policy") != CAUSAL_EVIDENCE_POLICY:
        raise ReservationCalibrationError("causal evidence does not use the declared paired policy")
    for name in ("observation_id", "action", "planning_event", "origin_cutoff", "label_available_at"):
        if str(evidence.get(name)) != str(row.get(name)):
            raise ReservationCalibrationError(f"causal evidence disagrees on {name}")
    source = evidence.get("source")
    pair = evidence.get("counterfactual_pair")
    label = evidence.get("label")
    if not isinstance(source, Mapping) or not isinstance(pair, Mapping) or not isinstance(label, Mapping):
        raise ReservationCalibrationError("causal evidence lacks source, paired arms or outcome label")
    required_source = (
        "source_decision_id", "source_result_sha256", "source_artifact_sha256",
        "generation_id", "planning_event", "cutoff", "data_snapshot_sha256",
        "certification_identity", "world_identity",
    )
    if any(not str(source.get(name) or "").strip() for name in required_source):
        raise ReservationCalibrationError("causal evidence has incomplete source decision/generation identity")
    for name in ("source_result_sha256", "source_artifact_sha256", "data_snapshot_sha256",
                 "certification_identity", "world_identity"):
        if not _is_sha256(source.get(name)):
            raise ReservationCalibrationError(f"causal evidence source {name} is not a SHA-256 identity")
    if (
        str(source.get("cutoff")) != str(row.get("origin_cutoff"))
        or int(source.get("planning_event") or -1) != int(row.get("planning_event") or -2)
    ):
        raise ReservationCalibrationError("causal evidence source is not from the forecast origin")
    play, save = pair.get("play"), pair.get("save")
    if not isinstance(play, Mapping) or not isinstance(save, Mapping):
        raise ReservationCalibrationError("causal evidence must retain both PLAY and SAVE arms")
    paired_fields = ("scenario_identity", "world_identity", "proposed_squad_ids", "lineup")
    for name in paired_fields:
        if play.get(name) != save.get(name) or play.get(name) in (None, "", [], {}):
            raise ReservationCalibrationError(f"PLAY and SAVE do not share the same {name}")
    source_arm_fields = (
        "source_decision_id", "source_result_sha256", "source_artifact_sha256",
        "generation_id", "cutoff", "data_snapshot_sha256", "certification_identity",
    )
    for arm_name, arm in (("PLAY", play), ("SAVE", save)):
        for name in source_arm_fields:
            if str(arm.get(name) or "") != str(source.get(name) or ""):
                raise ReservationCalibrationError(f"{arm_name} arm is not bound to source {name}")
    if not str(source.get("world_identity") or "").strip() or str(
        play.get("world_identity")
    ) != str(source.get("world_identity")):
        raise ReservationCalibrationError("paired arms are not bound to the source world identity")
    for arm_name, arm in (("PLAY", play), ("SAVE", save)):
        if any(not str(arm.get(name) or "").strip()
               for name in ("arm_id", "artifact_ref", "artifact_sha256")):
            raise ReservationCalibrationError(f"{arm_name} arm is missing retained artifact identity")
        artifact_payload = arm.get("artifact_payload")
        if not isinstance(artifact_payload, Mapping) or _canonical_sha256(artifact_payload) != str(
            arm.get("artifact_sha256")
        ):
            raise ReservationCalibrationError(f"{arm_name} arm artifact content digest does not verify")
        for name in (*paired_fields, *source_arm_fields, "arm_id"):
            if artifact_payload.get(name) != arm.get(name):
                raise ReservationCalibrationError(f"{arm_name} arm artifact disagrees on {name}")
        _number(artifact_payload.get("paired_value"), name=f"{arm_name} paired value")
    if str(play.get("arm_id")) == str(save.get("arm_id")):
        raise ReservationCalibrationError("PLAY and SAVE must be separately retained counterfactual arms")
    if label.get("kind") != "OBSERVED_FUTURE_OUTCOME" or not str(label.get("outcome_record_ref") or "").strip():
        raise ReservationCalibrationError("causal label is not tied to a retained future outcome")
    outcome_record = label.get("outcome_record")
    outcome_digest = str(label.get("outcome_record_sha256") or "")
    if not isinstance(outcome_record, Mapping) or len(outcome_digest) != 64:
        raise ReservationCalibrationError("causal label has no content-addressed outcome record")
    if _canonical_sha256(outcome_record) != outcome_digest:
        raise ReservationCalibrationError("causal label outcome-record digest does not verify")
    if _number(label.get("realized_reservation_value"), name="realized reservation value") != _number(
        row.get("realized_value"), name="realized reservation value"
    ):
        raise ReservationCalibrationError("causal evidence label differs from the supplied outcome")
    if str(label.get("available_at")) != str(row.get("label_available_at")):
        raise ReservationCalibrationError("causal evidence outcome availability differs from the row")


def validate_observation(
    row: Mapping[str, Any], *, evidence_verifier: Callable[[str], Mapping[str, Any]] | None,
) -> VerifiedOpportunity:
    """Load and verify one point-in-time forecast/outcome pair.

    The verifier must read the referenced retained evidence and independently
    validate its byte/content digest. Calibration never trusts caller-provided
    ``verified`` flags or free-form calibration statuses.
    """

    if evidence_verifier is None:
        raise ReservationCalibrationError("a retained causal-evidence verifier is required")
    observation_id = str(row.get("observation_id") or "")
    action = str(row.get("action") or "")
    if not observation_id or action not in cd.PLAYABLE_CHIP_ACTIONS:
        raise ReservationCalibrationError("observation id or chip action is invalid")
    origin = _utc(row.get("origin_cutoff"), name="origin_cutoff")
    label_time = _utc(row.get("label_available_at"), name="label_available_at")
    forecast_time = _utc(row.get("forecast_made_at"), name="forecast_made_at")
    if forecast_time > origin:
        raise ReservationCalibrationError("reservation forecast was made after its origin cutoff")
    if label_time <= origin:
        raise ReservationCalibrationError("future outcome label was already available at the forecast cutoff")
    reference = str(row.get("causal_evidence_ref") or "")
    expected_digest = str(row.get("causal_evidence_sha256") or "")
    if not reference or not _is_sha256(expected_digest):
        raise ReservationCalibrationError("retained causal-evidence reference/digest is missing")
    try:
        evidence = evidence_verifier(reference)
    except Exception as failure:
        raise ReservationCalibrationError(f"retained causal evidence did not verify: {failure}") from failure
    if not isinstance(evidence, Mapping) or _canonical_sha256(evidence) != expected_digest:
        raise ReservationCalibrationError("retained causal-evidence content digest does not match")
    _verify_pair(evidence, row)
    forecast = evidence.get("forecast")
    if not isinstance(forecast, Mapping):
        raise ReservationCalibrationError("causal evidence has no retained origin forecast")
    if (
        _number(forecast.get("value"), name="evidence forecast")
        != _number(row.get("forecast_value"), name="forecast value")
        or str(forecast.get("made_at")) != str(row.get("forecast_made_at"))
    ):
        raise ReservationCalibrationError("causal evidence forecast differs from the supplied row")
    weeks = row.get("weeks_to_expiry")
    if weeks is None or int(weeks) < 0:
        raise ReservationCalibrationError("weeks_to_expiry must be an explicit non-negative integer")
    return VerifiedOpportunity(
        observation_id=observation_id,
        action=action,
        planning_event=int(row.get("planning_event")),
        weeks_to_expiry=int(weeks),
        origin_cutoff=origin,
        forecast_made_at=forecast_time,
        label_available_at=label_time,
        forecast_value=_number(row.get("forecast_value"), name="forecast value"),
        realized_value=_number(row.get("realized_value"), name="realized value"),
        evidence_ref=reference,
        evidence_sha256=expected_digest,
    )


def _quantile(values: Sequence[float], q: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, int(round(float(q) * (len(ordered) - 1)))))
    return ordered[index]


def _metrics(rows: Sequence[tuple[float, float]]) -> dict[str, float]:
    residuals = [actual - prediction for prediction, actual in rows]
    raw_mae = sum(abs(actual - prediction) for prediction, actual in rows) / len(rows)
    bias = sum(residuals) / len(residuals)
    return {"mae": raw_mae, "bias": bias}


def _walk_forward_group(rows: Sequence[VerifiedOpportunity]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: (row.origin_cutoff, row.observation_id))
    predictions: list[tuple[float, float, float, float, float]] = []
    for test in ordered:
        training = [
            row for row in ordered
            if row.observation_id != test.observation_id
            and row.origin_cutoff < test.origin_cutoff
            and row.label_available_at <= test.origin_cutoff
        ]
        if len(training) < MIN_TRAINING_ORIGINS:
            continue
        # Locked intercept-only correction. It calibrates a known action/expiry
        # band without fitting a flexible model to a small seasonal sample.
        bias_correction = sum(row.realized_value - row.forecast_value for row in training) / len(training)
        train_residuals = [row.realized_value - (row.forecast_value + bias_correction) for row in training]
        low = _quantile(train_residuals, 0.05)
        high = _quantile(train_residuals, 0.95)
        prediction = test.forecast_value + bias_correction
        predictions.append((test.forecast_value, prediction, test.realized_value, low, high))

    if len(predictions) < MIN_VALIDATION_ORIGINS:
        return {
            "status": cd.CALIBRATION_UNCALIBRATED,
            "reason": DIAG_CALIBRATION_INSUFFICIENT,
            "training_origins": len(ordered),
            "validation_origins": len(predictions),
        }
    raw_pairs = [(raw, actual) for raw, _adjusted, actual, _low, _high in predictions]
    adj_pairs = [(adjusted, actual) for _raw, adjusted, actual, _low, _high in predictions]
    raw = _metrics(raw_pairs)
    adjusted = _metrics(adj_pairs)
    coverage = sum(
        1 for _raw, predicted, actual, low, high in predictions
        if predicted + low <= actual <= predicted + high
    ) / len(predictions)
    improvement = 0.0 if raw["mae"] == 0.0 else (raw["mae"] - adjusted["mae"]) / raw["mae"]
    accepted = (
        improvement >= MIN_MAE_IMPROVEMENT
        and abs(adjusted["bias"]) <= MAX_ABSOLUTE_BIAS
        and coverage >= MIN_90_INTERVAL_COVERAGE
    )
    all_residuals = [row.realized_value - row.forecast_value for row in ordered]
    final_correction = sum(all_residuals) / len(all_residuals)
    calibrated_residuals = [residual - final_correction for residual in all_residuals]
    return {
        "status": cd.CALIBRATION_CALIBRATED if accepted else cd.CALIBRATION_UNCALIBRATED,
        "reason": None if accepted else DIAG_CALIBRATION_NOT_VALIDATED,
        "training_origins": len(ordered),
        "validation_origins": len(predictions),
        "raw_metrics": raw,
        "calibrated_metrics": adjusted,
        "mae_improvement_fraction": improvement,
        "central_90_interval_coverage": coverage,
        "final_bias_correction": final_correction,
        "final_residual_q05": _quantile(calibrated_residuals, 0.05),
        "final_residual_q95": _quantile(calibrated_residuals, 0.95),
    }


def evaluate_reservation_calibration(
    observations: Sequence[Mapping[str, Any]],
    *,
    evidence_verifier: Callable[[str], Mapping[str, Any]] | None,
    evaluation_cutoff: str,
) -> dict[str, Any]:
    """Evaluate a pre-registered, expanding-window reservation calibration.

    All supplied evidence is validated before fitting. Labels unavailable at the
    declared evaluation cutoff are excluded; labels unavailable at each
    walk-forward origin cannot enter that fold's training set.
    """

    cutoff = _utc(evaluation_cutoff, name="evaluation_cutoff")
    verified: list[VerifiedOpportunity] = []
    rejected: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw in observations:
        try:
            row = validate_observation(raw, evidence_verifier=evidence_verifier)
            if row.observation_id in seen:
                raise ReservationCalibrationError("duplicate observation_id")
            seen.add(row.observation_id)
            if row.label_available_at <= cutoff:
                verified.append(row)
            # Unmatured labels are intentionally not counted as failures or as
            # training data; they remain unavailable at this evaluation cutoff.
        except Exception as failure:
            rejected.append({"observation_id": str(raw.get("observation_id") or ""),
                             "reason": f"{type(failure).__name__}: {failure}"})
    groups: dict[tuple[str, str], list[VerifiedOpportunity]] = {}
    for row in verified:
        groups.setdefault((row.action, row.bucket), []).append(row)
    evaluated: dict[str, Any] = {}
    models: dict[str, Any] = {}
    for (action, bucket), rows in sorted(groups.items()):
        key = f"{action}:{bucket}"
        unique_origins: set[tuple[int, datetime]] = set()
        duplicates: list[str] = []
        for row in rows:
            origin_key = (row.planning_event, row.origin_cutoff)
            if origin_key in unique_origins:
                duplicates.append(row.observation_id)
            unique_origins.add(origin_key)
        report = (
            {
                "status": cd.CALIBRATION_UNCALIBRATED,
                "reason": DIAG_CALIBRATION_EVIDENCE_INVALID,
                "training_origins": len(rows),
                "validation_origins": 0,
                "duplicate_origin_observation_ids": duplicates,
            }
            if duplicates else _walk_forward_group(rows)
        )
        evaluated[key] = report
        if report.get("status") == cd.CALIBRATION_CALIBRATED:
            models[key] = {
                "bias_correction": float(report["final_bias_correction"]),
                "residual_q05": float(report["final_residual_q05"]),
                "residual_q95": float(report["final_residual_q95"]),
                "training_origins": len(rows),
            }
    observation_manifest = [
        {
            "observation_id": row.observation_id,
            "action": row.action,
            "planning_event": row.planning_event,
            "weeks_to_expiry": row.weeks_to_expiry,
            "origin_cutoff": row.origin_cutoff.isoformat().replace("+00:00", "Z"),
            "forecast_made_at": row.forecast_made_at.isoformat().replace("+00:00", "Z"),
            "label_available_at": row.label_available_at.isoformat().replace("+00:00", "Z"),
            "forecast_value": row.forecast_value,
            "realized_value": row.realized_value,
            "causal_evidence_ref": row.evidence_ref,
            "causal_evidence_sha256": row.evidence_sha256,
        }
        for row in sorted(verified, key=lambda item: (item.origin_cutoff, item.action, item.observation_id))
    ]
    body = {
        "schema": CALIBRATION_SCHEMA,
        "version": CALIBRATION_VERSION,
        "policy": {
            "minimum_training_origins": MIN_TRAINING_ORIGINS,
            "minimum_validation_origins": MIN_VALIDATION_ORIGINS,
            "minimum_mae_improvement_fraction": MIN_MAE_IMPROVEMENT,
            "maximum_absolute_bias": MAX_ABSOLUTE_BIAS,
            "central_90_interval_coverage_minimum": MIN_90_INTERVAL_COVERAGE,
            "walk_forward": "EXPANDING_ORIGIN_LABELS_MATURED_BY_TEST_CUTOFF",
            "correction": "ACTION_AND_EXPIRY_BUCKET_INTERCEPT_ONLY",
            "causal_evidence_policy": CAUSAL_EVIDENCE_POLICY,
        },
        "evaluation_cutoff": cutoff.isoformat().replace("+00:00", "Z"),
        "evaluation_period": {
            "first_origin_cutoff": observation_manifest[0]["origin_cutoff"] if observation_manifest else None,
            "last_label_available_at": max(
                (item["label_available_at"] for item in observation_manifest), default=None
            ),
        },
        "verified_observations": len(verified),
        "verified_observation_manifest": observation_manifest,
        "dataset_identity_sha256": _canonical_sha256(observation_manifest),
        "rejected_observations": rejected,
        "action_bucket_reports": evaluated,
        "models": models,
        "status": (
            cd.CALIBRATION_CALIBRATED
            if models and not rejected else cd.CALIBRATION_UNCALIBRATED
        ),
    }
    # One contradictory/tampered record invalidates the calibration attempt;
    # callers must resolve the evidence set and rerun against the same locked
    # criteria rather than obtaining a fit by silently excluding it.
    if rejected:
        body["models"] = {}
        for report in body["action_bucket_reports"].values():
            report["status"] = cd.CALIBRATION_UNCALIBRATED
            report["reason"] = DIAG_CALIBRATION_EVIDENCE_INVALID
    body["model_identity_sha256"] = _canonical_sha256({
        "version": body["version"], "policy": body["policy"], "models": body["models"],
    })
    body["identity_sha256"] = _canonical_sha256(body)
    return body


@dataclass(frozen=True)
class VerifiedReservationCalibration:
    """ReservationValue backed by the module's verified immutable artifact."""

    artifact: Mapping[str, Any]

    @classmethod
    def from_artifact(
        cls,
        artifact: Mapping[str, Any],
        *,
        evidence_verifier: Callable[[str], Mapping[str, Any]] | None,
    ) -> "VerifiedReservationCalibration":
        body = dict(artifact)
        identity = str(body.pop("identity_sha256", ""))
        if body.get("schema") != CALIBRATION_SCHEMA or body.get("version") != CALIBRATION_VERSION:
            raise ReservationCalibrationError("reservation calibration artifact schema/version is unknown")
        if identity != _canonical_sha256(body):
            raise ReservationCalibrationError("reservation calibration artifact digest does not verify")
        manifest = body.get("verified_observation_manifest")
        if not isinstance(manifest, list) or body.get("dataset_identity_sha256") != _canonical_sha256(manifest):
            raise ReservationCalibrationError("reservation calibration dataset identity does not verify")
        expected_model_identity = _canonical_sha256({
            "version": body.get("version"), "policy": body.get("policy"), "models": body.get("models"),
        })
        if body.get("model_identity_sha256") != expected_model_identity:
            raise ReservationCalibrationError("reservation model identity does not verify")
        if body.get("status") != cd.CALIBRATION_CALIBRATED or not body.get("models"):
            raise ReservationCalibrationError("reservation calibration artifact is not calibrated")
        if evidence_verifier is None:
            raise ReservationCalibrationError(
                "retained causal-evidence verifier is required to load a calibration artifact"
            )
        if body.get("rejected_observations"):
            raise ReservationCalibrationError("calibration artifact contains rejected causal observations")
        if int(body.get("verified_observations") or -1) != len(manifest):
            raise ReservationCalibrationError("calibration observation count does not match its manifest")
        required_row_fields = (
            "observation_id", "action", "planning_event", "weeks_to_expiry", "origin_cutoff",
            "forecast_made_at", "label_available_at", "forecast_value", "realized_value",
            "causal_evidence_ref", "causal_evidence_sha256",
        )
        try:
            rows = [
                {key: item[key] for key in required_row_fields}
                for item in manifest
            ]
            reproduced = evaluate_reservation_calibration(
                rows,
                evidence_verifier=evidence_verifier,
                evaluation_cutoff=str(body["evaluation_cutoff"]),
            )
        except Exception as failure:
            raise ReservationCalibrationError(
                f"calibration evidence could not be independently revalidated: {failure}"
            ) from failure
        if reproduced.get("status") != cd.CALIBRATION_CALIBRATED or _canonical_sha256(
            reproduced
        ) != _canonical_sha256(dict(artifact)):
            raise ReservationCalibrationError(
                "calibration status or model does not reproduce from retained causal evidence and locked criteria"
            )
        return cls(artifact=dict(artifact))

    def estimate(self, *, action: str, planning_event: int, expiry_event: int | None,
                 state: Mapping[str, Any]) -> cd.ReservationEstimate:
        weeks = None if expiry_event is None else max(0, int(expiry_event) - int(planning_event))
        model = (self.artifact.get("models") or {}).get(f"{action}:{expiry_bucket(weeks)}")
        if not isinstance(model, Mapping):
            return cd.ReservationEstimate(
                value=None, calibration_status=cd.CALIBRATION_UNCALIBRATED,
                terminal_value=0.0, weeks_to_expiry=weeks,
                reason_codes=(DIAG_CALIBRATION_INSUFFICIENT,),
                conditional_on=("action", "weeks_to_expiry_bucket"),
            )
        raw = state.get("raw_reservation_value")
        if raw is None:
            return cd.ReservationEstimate(
                value=None, calibration_status=cd.CALIBRATION_UNCALIBRATED,
                terminal_value=0.0, weeks_to_expiry=weeks,
                reason_codes=(DIAG_CALIBRATION_INSUFFICIENT,),
                conditional_on=("action", "weeks_to_expiry_bucket"),
            )
        value = max(0.0, _number(raw, name="raw reservation value") + float(model["bias_correction"]))
        return cd.ReservationEstimate(
            value=value,
            calibration_status=cd.CALIBRATION_CALIBRATED,
            terminal_value=0.0,
            weeks_to_expiry=weeks,
            reason_codes=(),
            conditional_on=("action", "weeks_to_expiry_bucket"),
        )
