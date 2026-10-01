"""Immutable per-run retention and verification for chip assessments."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import uuid
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from . import chip_decision as cd
from . import chip_reservation_forecast as crf

ASSESSMENT_SCHEMA = "fpl_brain.chip_operational_assessment.v1"
ASSESSMENT_VERSION = "chip_assessment_store_v1.2.0"
ASSESSMENT_ACTIONS = (cd.CHIP_ACTION_BB, cd.CHIP_ACTION_TC, cd.CHIP_ACTION_FH, cd.CHIP_ACTION_WC)

ASSESSMENT_INVALID = "CHIP_ASSESSMENT_RECORD_INVALID"
ASSESSMENT_PUBLICATION_UNAVAILABLE = "CHIP_ASSESSMENT_ATOMIC_PUBLICATION_UNAVAILABLE"


class ChipAssessmentStoreError(ValueError):
    pass


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False, default=str,
    ).encode("utf-8")


def _digest(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def free_hit_arm_identity(arms: Mapping[str, Any]) -> str:
    """Hash the serialized PLAY/SAVE/restoration payload consistently."""

    from . import analytics

    normalized = _plain({
        "play": arms.get("play"),
        "save": arms.get("save"),
        "restore_at_h2": arms.get("restore_at_h2"),
    })
    return analytics.canonical_hash(normalized)


def manager_state_identity(manager_state: Mapping[str, Any]) -> str:
    """Hash the full point-in-time manager basis in its serialized form."""

    from . import analytics

    payload = {
        "entry_id": int(manager_state["entry_id"]),
        "planning_event": int(manager_state["planning_event"]),
        "cutoff": str(manager_state["cutoff"]),
        "owned_ids": sorted(int(pid) for pid in manager_state["squad_ids"]),
        "purchase_price_tenths": {
            int(pid): int(price)
            for pid, price in dict(manager_state["purchase_price_tenths"]).items()
        },
        "bank_tenths": int(manager_state["bank_tenths"]),
        "free_transfers": int(manager_state["free_transfers"]),
        "event_start_free_transfers": int(manager_state["event_start_free_transfers"]),
    }
    return analytics.canonical_hash(_plain(payload))


def _plain(value: Any) -> Any:
    if hasattr(value, "as_dict") and callable(value.as_dict):
        return _plain(value.as_dict())
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _validate_context(record: Mapping[str, Any]) -> None:
    context = record.get("context")
    if not isinstance(context, Mapping):
        raise ChipAssessmentStoreError("assessment context is missing")
    for name in (
        "planning_event", "horizon_events", "cutoff", "data_snapshot_sha256",
        "certification_identity", "generation_id", "source_decision_id",
        "source_decision_result_sha256", "source_decision_artifact_sha256",
    ):
        value = context.get(name)
        if value is None or value == "" or value == []:
            raise ChipAssessmentStoreError(f"assessment context omits {name}")
    actions = record.get("chip_results")
    manager_state = record.get("manager_state")
    if not isinstance(manager_state, Mapping):
        raise ChipAssessmentStoreError("canonical manager state is missing")
    if context.get("manager_state_identity") is not None:
        try:
            manager_identity = manager_state_identity(manager_state)
        except (KeyError, TypeError, ValueError) as failure:
            raise ChipAssessmentStoreError(f"canonical manager state is incomplete: {failure}") from failure
        if str(context.get("manager_state_identity")) != manager_identity:
            raise ChipAssessmentStoreError("assessment manager-state identity does not reproduce")
    if not isinstance(actions, Mapping) or set(actions) != set(ASSESSMENT_ACTIONS):
        raise ChipAssessmentStoreError("assessment must retain a result or refusal for BB, TC, FH and WC")
    event = int(context["planning_event"])
    horizon = tuple(int(item) for item in context["horizon_events"])
    certification = str(context["certification_identity"])
    snapshot = str(context["data_snapshot_sha256"])
    evaluated: dict[str, Mapping[str, Any]] = {}
    for action in ASSESSMENT_ACTIONS:
        item = actions[action]
        if not isinstance(item, Mapping) or item.get("status") not in {"EVALUATED", "BLOCKED", "UNAVAILABLE"}:
            raise ChipAssessmentStoreError(f"{action} has no structured disposition")
        evaluation = item.get("evaluation")
        if item.get("status") == "EVALUATED":
            if not isinstance(evaluation, Mapping):
                raise ChipAssessmentStoreError(f"{action} is marked evaluated without an evaluation")
            evidence = evaluation.get("evidence") or {}
            if (
                str(evidence.get("certification_identity") or "") != certification
                or str(evidence.get("data_snapshot_sha256") or "") != snapshot
                or int(evidence.get("planning_event") or -1) != event
                or tuple(int(v) for v in (evidence.get("horizon_events") or ())) != horizon
            ):
                raise ChipAssessmentStoreError(f"{action} evaluation is not bound to assessment context")
            evaluated[action] = evaluation
            if action in {cd.CHIP_ACTION_BB, cd.CHIP_ACTION_FH, cd.CHIP_ACTION_WC}:
                readiness = item.get("evaluator_readiness")
                readiness_artifact = (
                    readiness.get("artifact") if isinstance(readiness, Mapping) else None
                )
                if bool(evaluation.get("execution_permitted")):
                    if not isinstance(readiness_artifact, Mapping):
                        raise ChipAssessmentStoreError(
                            f"{action} execution permission has no retained readiness artifact"
                        )
                    from . import chip_evaluator_readiness as cer

                    try:
                        report = cer.verify_retained_readiness_integrity(readiness_artifact)
                    except Exception as failure:
                        raise ChipAssessmentStoreError(
                            f"{action} readiness evidence does not verify: {failure}"
                        ) from failure
                    evaluator_identity = evidence.get("evaluator_readiness")
                    if (
                        readiness_artifact.get("action") != action
                        or readiness_artifact.get("evaluator_version") != evaluation.get("evaluator_version")
                        or readiness_artifact.get("status") != cer.READINESS_READY
                        or not report.get("execution_permitted")
                        or readiness.get("execution_permitted") is not True
                        or not isinstance(evaluator_identity, Mapping)
                        or evaluator_identity.get("artifact_sha256") != report.get("artifact_sha256")
                    ):
                        raise ChipAssessmentStoreError(
                            f"{action} execution permission disagrees with its action-specific readiness evidence"
                        )
                elif isinstance(readiness_artifact, Mapping):
                    from . import chip_evaluator_readiness as cer

                    try:
                        report = cer.verify_retained_readiness_integrity(readiness_artifact)
                    except Exception as failure:
                        raise ChipAssessmentStoreError(
                            f"{action} readiness evidence does not verify: {failure}"
                        ) from failure
                    if (
                        readiness_artifact.get("action") != action
                        or readiness_artifact.get("evaluator_version") != evaluation.get("evaluator_version")
                        or bool(report.get("execution_permitted")) != bool(readiness.get("execution_permitted"))
                    ):
                        raise ChipAssessmentStoreError(
                            f"{action} readiness report differs from its retained artifact"
                        )
            raw_forecast = item.get("raw_reservation_forecast")
            if raw_forecast is not None:
                source_identity = {
                    "source_decision_id": context["source_decision_id"],
                    "source_result_sha256": context["source_decision_result_sha256"],
                    "source_artifact_sha256": context["source_decision_artifact_sha256"],
                    "generation_id": context["generation_id"],
                    "planning_event": event,
                    "origin_cutoff": str(context["cutoff"]),
                    "data_snapshot_sha256": context["data_snapshot_sha256"],
                    "predictive_code_snapshot_sha256": context.get("predictive_code_snapshot_sha256"),
                    "certification_identity": certification,
                }
                try:
                    expiry_map = context.get("chip_expiry_events")
                    crf.verify_reservation_forecast(
                        raw_forecast,
                        expected={
                            "action": action,
                            "planning_event": event,
                            "origin_cutoff": str(context["cutoff"]),
                            **({"expiry_event": expiry_map[action]}
                               if isinstance(expiry_map, Mapping) and action in expiry_map else {}),
                            "source_identity": source_identity,
                        },
                    )
                except Exception as failure:
                    raise ChipAssessmentStoreError(
                        f"{action} raw reservation forecast is invalid: {failure}"
                    ) from failure
                expected_state: dict[str, Any] = {
                    "squad_ids": list(manager_state.get("squad_ids") or ())
                }
                save_state = (evidence.get("save_policy") or {}).get(
                    "post_save_state_for_reservation"
                )
                if isinstance(save_state, Mapping):
                    expected_state.update(dict(save_state))
                if dict(raw_forecast.get("reservation_state") or {}) != expected_state:
                    raise ChipAssessmentStoreError(
                        f"{action} raw reservation forecast is not bound to the retained SAVE state"
                    )
        elif not item.get("reason_codes"):
            raise ChipAssessmentStoreError(f"{action} refusal omits its structured reason code")

    if cd.CHIP_ACTION_BB in evaluated and cd.CHIP_ACTION_TC in evaluated:
        bb_evidence = evaluated[cd.CHIP_ACTION_BB].get("evidence") or {}
        tc_evidence = evaluated[cd.CHIP_ACTION_TC].get("evidence") or {}
        for name in ("scenario_identity", "world_identity", "proposed_owned_ids", "lineup"):
            if bb_evidence.get(name) != tc_evidence.get(name) or bb_evidence.get(name) in (None, "", [], {}):
                raise ChipAssessmentStoreError(f"BB and TC were not evaluated on one {name}")

    if cd.CHIP_ACTION_FH in evaluated:
        arms = actions[cd.CHIP_ACTION_FH].get("arm_evidence")
        if not isinstance(arms, Mapping) or not isinstance(arms.get("play"), Mapping) or not isinstance(
            arms.get("save"), Mapping
        ):
            raise ChipAssessmentStoreError("evaluated FH is missing retained PLAY and SAVE arm evidence")
        play, save = arms["play"], arms["save"]
        expected_sources = {
            "source_decision_id": context["source_decision_id"],
            "source_result_sha256": context["source_decision_result_sha256"],
            "source_artifact_sha256": context["source_decision_artifact_sha256"],
        }
        for name, expected in expected_sources.items():
            if arms.get(name) != expected:
                raise ChipAssessmentStoreError(f"FH arm assembly does not match assessment {name}")
        for name in ("generation_id", "cutoff", "data_snapshot_sha256", "certification_identity",
                     "manager_state_identity", "world_identity"):
            if play.get(name) != save.get(name):
                raise ChipAssessmentStoreError(f"FH PLAY and SAVE differ on {name}")
            if context.get(name) is not None and play.get(name) != context.get(name):
                raise ChipAssessmentStoreError(f"FH arms do not match assessment context on {name}")
        try:
            canonical_manager_identity = manager_state_identity(manager_state)
        except (KeyError, TypeError, ValueError) as failure:
            raise ChipAssessmentStoreError(f"FH canonical manager state is incomplete: {failure}") from failure
        if (
            context.get("manager_state_identity") != canonical_manager_identity
            or play.get("manager_state_identity") != canonical_manager_identity
        ):
            raise ChipAssessmentStoreError("FH arms are not bound to the retained canonical manager state")
        if actions[cd.CHIP_ACTION_FH].get("arm_identity") is None:
            raise ChipAssessmentStoreError("evaluated FH arms have no shared arm identity")
        restore = arms.get("restore_at_h2")
        if not isinstance(restore, Mapping):
            raise ChipAssessmentStoreError("evaluated FH does not retain its H2 restoration state")
        if (
            restore.get("permanent_squad_ids") != restore.get("restored_squad_ids")
            or restore.get("permanent_purchase_price_tenths") != restore.get("restored_purchase_price_tenths")
            or restore.get("permanent_bank_tenths") != restore.get("restored_bank_tenths")
            or int(restore.get("restored_h2_free_transfers", -1))
            != int(restore.get("event_start_h1_free_transfers", -2))
            or restore.get("restoration_problems")
        ):
            raise ChipAssessmentStoreError("evaluated FH does not preserve its permanent squad/bank/FT basis")
        manager_squad = sorted(int(pid) for pid in manager_state["squad_ids"])
        manager_basis = {
            str(int(pid)): int(price)
            for pid, price in dict(manager_state["purchase_price_tenths"]).items()
        }
        expected_restoration = {
            "permanent_squad_ids": manager_squad,
            "restored_squad_ids": manager_squad,
            "permanent_purchase_price_tenths": manager_basis,
            "restored_purchase_price_tenths": manager_basis,
            "permanent_bank_tenths": int(manager_state["bank_tenths"]),
            "restored_bank_tenths": int(manager_state["bank_tenths"]),
            "current_h1_free_transfers": int(manager_state["free_transfers"]),
            "event_start_h1_free_transfers": int(manager_state["event_start_free_transfers"]),
            "restored_h2_free_transfers": int(manager_state["event_start_free_transfers"]),
        }
        if any(restore.get(name) != value for name, value in expected_restoration.items()):
            raise ChipAssessmentStoreError(
                "evaluated FH restoration evidence does not reproduce from the canonical manager state"
            )
        expected_events = tuple(int(value) for value in horizon)
        play_actions = play.get("actions") or ()
        save_actions = save.get("actions") or ()
        if (
            tuple(int(row.get("event", -1)) for row in play_actions) != expected_events[1:]
            or tuple(int(row.get("event", -1)) for row in save_actions) != expected_events
            or play.get("source") != "OPTIMIZED_RESTORED_PERMANENT_H2_STATE"
            or save.get("source") != "VERIFIED_NORMAL_DECISION_ROUTE"
        ):
            raise ChipAssessmentStoreError("evaluated FH does not retain the required PLAY and SAVE route arms")
        restore_ids = tuple(sorted(int(pid) for pid in manager_state["squad_ids"]))
        restore_basis = {str(int(pid)): int(price) for pid, price in
                         dict(manager_state["purchase_price_tenths"]).items()}
        expected_save_start = {
            "event": expected_events[0],
            "squad_ids": list(restore_ids),
            "bank_tenths": int(manager_state["bank_tenths"]),
            "free_transfers": int(manager_state["free_transfers"]),
            "purchase_price_tenths": restore_basis,
        }
        expected_play_start = {
            "event": expected_events[1],
            "squad_ids": list(restore_ids),
            "bank_tenths": int(manager_state["bank_tenths"]),
            "free_transfers": int(manager_state["event_start_free_transfers"]),
            "purchase_price_tenths": restore_basis,
        }
        for name, expected in (("SAVE", expected_save_start), ("PLAY", expected_play_start)):
            actual = (save if name == "SAVE" else play).get("start_state")
            if not isinstance(actual, Mapping) or any(actual.get(key) != value for key, value in expected.items()):
                raise ChipAssessmentStoreError(
                    f"FH {name} start state does not reproduce from the canonical manager/restoration basis"
                )
        expected_save_config_events = list(expected_events)
        expected_play_config_events = list(expected_events[1:])
        save_config = save.get("route_config")
        play_config = play.get("route_config")
        if (
            not isinstance(save_config, Mapping)
            or not isinstance(play_config, Mapping)
            or save_config.get("events") != expected_save_config_events
            or play_config.get("events") != expected_play_config_events
        ):
            raise ChipAssessmentStoreError("FH PLAY/SAVE route configs do not bind the retained event horizon")
        expected_arm_identity = free_hit_arm_identity(arms)
        if str(actions[cd.CHIP_ACTION_FH].get("arm_identity")) != expected_arm_identity:
            raise ChipAssessmentStoreError("evaluated FH arm identity digest does not verify")

    if cd.CHIP_ACTION_WC in evaluated:
        value_events = tuple(int(v) for v in (actions[cd.CHIP_ACTION_WC].get("value_horizon_events") or ()))
        if not 6 <= len(value_events) <= 10:
            raise ChipAssessmentStoreError("evaluated WC does not retain a certified 6-10 event value horizon")
        if value_events[:4] != horizon:
            raise ChipAssessmentStoreError("WC value horizon does not preserve the normal four-event prefix")
        if not actions[cd.CHIP_ACTION_WC].get("value_generation_id"):
            raise ChipAssessmentStoreError("evaluated WC does not name its separate value generation")
        production = actions[cd.CHIP_ACTION_WC].get("production_evidence") or {}
        if (
            str(production.get("chip_generation_id") or "") != str(context["generation_id"])
            or str(production.get("value_generation_id") or "")
            != str(actions[cd.CHIP_ACTION_WC].get("value_generation_id"))
            or str(production.get("cutoff") or "") != str(context["cutoff"])
            or str(production.get("data_snapshot_sha256") or "")
            != str(context["data_snapshot_sha256"])
            or str(production.get("source_snapshot_sha256") or "")
            != str(context.get("predictive_code_snapshot_sha256") or "")
        ):
            raise ChipAssessmentStoreError("WC value generation does not match the normal certified identity")

    decision = record.get("decision")
    if not isinstance(decision, Mapping):
        raise ChipAssessmentStoreError("assessment has no arbiter decision")
    decision_metrics = decision.get("candidate_metrics") or {}
    selected_action = str(decision_metrics.get("selected_chip_action") or "")
    selected_forecast = actions.get(selected_action, {}).get("raw_reservation_forecast")
    selected_forecast_sha = (
        str(selected_forecast.get("artifact_sha256") or "")
        if isinstance(selected_forecast, Mapping) else None
    )
    if decision_metrics.get("raw_reservation_forecast_sha256") != selected_forecast_sha:
        raise ChipAssessmentStoreError("arbiter forecast identity differs from the retained selected-action forecast")
    if isinstance(selected_forecast, Mapping) and decision_metrics.get("raw_reservation_value") != selected_forecast.get(
        "raw_value"
    ):
        raise ChipAssessmentStoreError("arbiter raw reservation value differs from its retained forecast")
    if decision.get("status") == cd.STATUS_PLAY_CHIP:
        action = str(decision.get("recommended_action") or "")
        evaluation = evaluated.get(action)
        if evaluation is None or not bool(evaluation.get("execution_permitted")):
            raise ChipAssessmentStoreError("PLAY_CHIP bypasses an evaluator execution-permission gate")
        if str(decision.get("calibration_status")) != cd.CALIBRATION_CALIBRATED:
            raise ChipAssessmentStoreError("PLAY_CHIP has no validated reservation calibration")


def build_assessment_record(
    *,
    context: Mapping[str, Any],
    manager_state: Mapping[str, Any],
    chip_results: Mapping[str, Mapping[str, Any]],
    decision: Any,
    created_at: str | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Create one complete, self-digesting assessment record."""

    body = {
        "schema": ASSESSMENT_SCHEMA,
        "version": ASSESSMENT_VERSION,
        "run_id": str(run_id or uuid.uuid4()),
        "created_at": str(created_at or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")),
        "context": _plain(context),
        "manager_state": _plain(manager_state),
        "chip_results": _plain(chip_results),
        "decision": _plain(decision),
        "retention": {"immutable": True, "per_run": True},
    }
    _validate_context(body)
    body["record_sha256"] = _digest(body)
    return body


def retain_assessment(record: Mapping[str, Any], root: str | Path) -> dict[str, str]:
    """Publish one assessment with atomic no-replace semantics.

    Temporary staging is protected by ``finally``. If the filesystem cannot
    atomically hard-link the staged file into place, the operation fails closed;
    the final name is never streamed into or replaced.
    """

    value = _plain(record)
    if not isinstance(value, dict) or value.get("schema") != ASSESSMENT_SCHEMA:
        raise ChipAssessmentStoreError("refusing to retain an unknown assessment record")
    _validate_context(value)
    expected = str(value.pop("record_sha256", ""))
    if expected != _digest(value):
        raise ChipAssessmentStoreError("assessment record digest does not verify before retention")
    value["record_sha256"] = expected
    payload = _canonical_bytes(value)
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    target = root_path / f"chip-assessment-{value['run_id']}.json"
    temporary: Path | None = None
    try:
        fd, name = tempfile.mkstemp(prefix=target.name + ".tmp-", dir=str(root_path))
        temporary = Path(name)
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError as failure:
            raise ChipAssessmentStoreError(f"assessment path already exists: {target}") from failure
        except OSError as failure:
            raise ChipAssessmentStoreError(
                f"{ASSESSMENT_PUBLICATION_UNAVAILABLE}: atomic no-replace publication failed: {failure}"
            ) from failure
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
    return {
        "path": str(target),
        "record_sha256": expected,
        "file_sha256": hashlib.sha256(payload).hexdigest(),
    }


def verify_assessment(path: str | Path) -> dict[str, Any]:
    """Verify bytes, internal digest, context bindings and per-chip contracts."""

    target = Path(path)
    try:
        payload = target.read_bytes()
        record = json.loads(payload.decode("utf-8"))
    except Exception as failure:
        raise ChipAssessmentStoreError(f"{ASSESSMENT_INVALID}: unreadable assessment: {failure}") from failure
    if not isinstance(record, Mapping) or record.get("schema") != ASSESSMENT_SCHEMA:
        raise ChipAssessmentStoreError(f"{ASSESSMENT_INVALID}: assessment schema is invalid")
    body = dict(record)
    identity = str(body.pop("record_sha256", ""))
    if len(identity) != 64 or identity != _digest(body):
        raise ChipAssessmentStoreError(f"{ASSESSMENT_INVALID}: internal record digest differs")
    try:
        _validate_context(record)
    except Exception as failure:
        raise ChipAssessmentStoreError(f"{ASSESSMENT_INVALID}: {failure}") from failure
    return {
        "verified": True,
        "path": str(target),
        "run_id": str(record.get("run_id") or ""),
        "record_sha256": identity,
        "file_sha256": hashlib.sha256(payload).hexdigest(),
        "planning_event": int(record["context"]["planning_event"]),
        "generation_id": str(record["context"]["generation_id"]),
        "chip_actions": sorted(record["chip_results"]),
    }
