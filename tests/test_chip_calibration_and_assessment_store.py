"""Focused evidence and retention tests for the chip operational remediation."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from fpl_brain import chip_assessment_store as store
from fpl_brain import chip_assessment as assessment
from fpl_brain import chip_decision as cd
from fpl_brain import chip_reservation_calibration as calibration
from fpl_brain.chip_assessment import assemble_assessment_record


def _sha(payload):
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False, default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _causal_row(index: int, *, noise: float = 0.0):
    origin = datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(days=7 * index)
    label_time = origin + timedelta(days=2)
    origin_text = origin.isoformat().replace("+00:00", "Z")
    label_text = label_time.isoformat().replace("+00:00", "Z")
    forecast_time = (origin - timedelta(minutes=1)).isoformat().replace("+00:00", "Z")
    event = index + 1
    outcome_record = {
        "record_id": f"outcome-{index}",
        "observed_points": 15.0 + noise,
        "source": "verified_outcome_ledger_fixture",
    }
    source = {
        "source_decision_id": f"decision-{index}",
        "source_result_sha256": _sha({"result": index}),
        "source_artifact_sha256": _sha({"artifact": index}),
        "generation_id": f"generation-{index}",
        "planning_event": event,
        "cutoff": origin_text,
        "data_snapshot_sha256": _sha({"snapshot": index}),
        "certification_identity": "sha256:" + _sha({"certification": index}),
        "world_identity": "sha256:" + _sha({"world": index}),
    }
    common = {
        "scenario_identity": f"scenario-{index}",
        "world_identity": "sha256:" + _sha({"world": index}),
        "proposed_squad_ids": list(range(1, 16)),
        "lineup": {"starters": list(range(1, 12)), "captain": 1, "vice": 2},
        **{key: source[key] for key in (
            "source_decision_id", "source_result_sha256", "source_artifact_sha256",
            "generation_id", "cutoff", "data_snapshot_sha256", "certification_identity",
        )},
    }
    play_arm_payload = {"arm_id": f"play-{index}", "paired_value": 3.0, **common}
    save_arm_payload = {"arm_id": f"save-{index}", "paired_value": 1.0, **common}
    evidence = {
        "schema": calibration.CAUSAL_EVIDENCE_SCHEMA,
        "policy": calibration.CAUSAL_EVIDENCE_POLICY,
        "observation_id": f"obs-{index}",
        "action": cd.CHIP_ACTION_BB,
        "planning_event": event,
        "origin_cutoff": origin_text,
        "label_available_at": label_text,
        "source": source,
        "forecast": {"value": 10.0, "made_at": forecast_time},
        "counterfactual_pair": {
            "play": {"arm_id": f"play-{index}", "artifact_ref": f"play-{index}.json",
                     "artifact_sha256": _sha(play_arm_payload),
                     "artifact_payload": play_arm_payload, **common},
            "save": {"arm_id": f"save-{index}", "artifact_ref": f"save-{index}.json",
                     "artifact_sha256": _sha(save_arm_payload),
                     "artifact_payload": save_arm_payload, **common},
        },
        "label": {
            "kind": "OBSERVED_FUTURE_OUTCOME",
            "outcome_record_ref": f"outcome-{index}.json",
            "outcome_record": outcome_record,
            "outcome_record_sha256": _sha(outcome_record),
            "realized_reservation_value": 15.0 + noise,
            "available_at": label_text,
        },
    }
    row = {
        "observation_id": f"obs-{index}",
        "action": cd.CHIP_ACTION_BB,
        "planning_event": event,
        "weeks_to_expiry": 3,
        "origin_cutoff": origin_text,
        "forecast_made_at": forecast_time,
        "forecast_value": 10.0,
        "label_available_at": label_text,
        "realized_value": 15.0 + noise,
        "causal_evidence_ref": f"evidence-{index}",
        "causal_evidence_sha256": _sha(evidence),
    }
    return row, evidence


def _context():
    events = cd.canonical_chip_horizon(5)
    return {
        "planning_event": 5,
        "horizon_events": list(events),
        "cutoff": "2026-09-29T20:27:01Z",
        "data_snapshot_sha256": "snapshot-test",
        "certification_identity": "cert-test",
        "generation_id": "generation-test",
        "source_decision_id": "decision-test",
        "source_decision_result_sha256": "result-test",
        "source_decision_artifact_sha256": "artifact-test",
    }


def _blocked_results():
    return {
        action: {"status": "BLOCKED", "reason_codes": [f"{action}_FIXTURE_BLOCK"]}
        for action in cd.PLAYABLE_CHIP_ACTIONS
    }


def test_calibration_requires_verified_paired_temporal_causal_evidence():
    stored = {}
    rows = []
    for index in range(55):
        noise = (-2.0, -1.0, 0.0, 1.0, 2.0, 0.0, 1.0, -1.0)[index % 8]
        row, evidence = _causal_row(index, noise=noise)
        rows.append(row)
        stored[row["causal_evidence_ref"]] = evidence

    last_label = datetime.fromisoformat(rows[-1]["label_available_at"].replace("Z", "+00:00"))
    artifact = calibration.evaluate_reservation_calibration(
        rows,
        evidence_verifier=stored.__getitem__,
        evaluation_cutoff=(last_label + timedelta(days=1)).isoformat().replace("+00:00", "Z"),
    )
    assert artifact["status"] == cd.CALIBRATION_CALIBRATED
    model = artifact["models"]["BB:1-4"]
    assert model["training_origins"] == 55
    reservation = calibration.VerifiedReservationCalibration.from_artifact(
        artifact, evidence_verifier=stored.__getitem__,
    )
    estimate = reservation.estimate(
        action=cd.CHIP_ACTION_BB,
        planning_event=5,
        expiry_event=8,
        state={"raw_reservation_value": 10.0},
    )
    assert estimate.calibration_status == cd.CALIBRATION_CALIBRATED
    assert estimate.value == pytest.approx(15.0, abs=0.6)


def test_calibration_loader_rejects_self_asserted_model_even_with_recomputed_hashes():
    stored = {}
    rows = []
    for index in range(55):
        noise = (-2.0, -1.0, 0.0, 1.0, 2.0, 0.0, 1.0, -1.0)[index % 8]
        row, evidence = _causal_row(index, noise=noise)
        rows.append(row)
        stored[row["causal_evidence_ref"]] = evidence
    last_label = datetime.fromisoformat(rows[-1]["label_available_at"].replace("Z", "+00:00"))
    artifact = calibration.evaluate_reservation_calibration(
        rows,
        evidence_verifier=stored.__getitem__,
        evaluation_cutoff=(last_label + timedelta(days=1)).isoformat().replace("+00:00", "Z"),
    )
    forged = json.loads(json.dumps(artifact))
    forged["models"]["BB:1-4"]["bias_correction"] += 100.0
    forged["model_identity_sha256"] = _sha({
        "version": forged["version"], "policy": forged["policy"], "models": forged["models"],
    })
    forged["identity_sha256"] = _sha({key: value for key, value in forged.items()
                                       if key != "identity_sha256"})
    with pytest.raises(calibration.ReservationCalibrationError, match="does not reproduce"):
        calibration.VerifiedReservationCalibration.from_artifact(
            forged, evidence_verifier=stored.__getitem__,
        )


def test_calibration_is_uncalibrated_without_rows_and_rejects_future_leakage():
    empty = calibration.evaluate_reservation_calibration(
        [], evidence_verifier=lambda _ref: {}, evaluation_cutoff="2026-09-30T00:00:00Z",
    )
    assert empty["status"] == cd.CALIBRATION_UNCALIBRATED
    assert empty["models"] == {}

    row, evidence = _causal_row(1)
    row["forecast_made_at"] = "2025-01-09T00:00:00Z"
    with pytest.raises(calibration.ReservationCalibrationError, match="after its origin cutoff"):
        calibration.validate_observation(row, evidence_verifier=lambda _ref: evidence)


def test_calibration_rejects_mixed_scenario_arms_and_any_tampered_causal_row():
    row, evidence = _causal_row(0)
    evidence["counterfactual_pair"]["save"]["scenario_identity"] = "different-scenario"
    row["causal_evidence_sha256"] = _sha(evidence)
    with pytest.raises(calibration.ReservationCalibrationError, match="same scenario_identity"):
        calibration.validate_observation(row, evidence_verifier=lambda _ref: evidence)

    stored = {}
    rows = []
    for index in range(55):
        good_row, good_evidence = _causal_row(index)
        rows.append(good_row)
        stored[good_row["causal_evidence_ref"]] = good_evidence
    bad_row, bad_evidence = _causal_row(56)
    bad_row["causal_evidence_ref"] = "tampered-causal-evidence"
    bad_evidence["label"]["realized_reservation_value"] += 9.0
    stored[bad_row["causal_evidence_ref"]] = bad_evidence
    bad_row["causal_evidence_sha256"] = _sha(bad_evidence)
    rows.append(bad_row)
    last_label = datetime.fromisoformat(rows[-1]["label_available_at"].replace("Z", "+00:00"))
    result = calibration.evaluate_reservation_calibration(
        rows,
        evidence_verifier=stored.__getitem__,
        evaluation_cutoff=(last_label + timedelta(days=1)).isoformat().replace("+00:00", "Z"),
    )
    assert result["status"] == cd.CALIBRATION_UNCALIBRATED
    assert result["models"] == {}
    assert result["rejected_observations"]


def test_calibration_does_not_override_an_evaluators_execution_permission():
    stored = {}
    rows = []
    for index in range(55):
        noise = (-2.0, -1.0, 0.0, 1.0, 2.0, 0.0, 1.0, -1.0)[index % 8]
        row, evidence = _causal_row(index, noise=noise)
        rows.append(row)
        stored[row["causal_evidence_ref"]] = evidence
    last_label = datetime.fromisoformat(rows[-1]["label_available_at"].replace("Z", "+00:00"))
    artifact = calibration.evaluate_reservation_calibration(
        rows, evidence_verifier=stored.__getitem__,
        evaluation_cutoff=(last_label + timedelta(days=1)).isoformat().replace("+00:00", "Z"),
    )
    reservation = calibration.VerifiedReservationCalibration.from_artifact(
        artifact, evidence_verifier=stored.__getitem__,
    )
    binding = cd.ChipHorizonBinding(
        planning_event=5,
        horizon_events=cd.canonical_chip_horizon(5),
        certification_identity="cert-test",
        data_snapshot_sha256="snapshot-test",
    )
    evaluation = cd.ChipEvaluation(
        action=cd.CHIP_ACTION_BB,
        evaluator_version="fixture",
        candidate_metrics={"mean_paired_uplift": 100.0, "raw_reservation_value": 10.0},
        uncertainty={"paired_interval_low": 20.0, "paired_interval_high": 120.0},
        evidence={
            "certification_identity": "cert-test",
            "data_snapshot_sha256": "snapshot-test",
            "horizon_events": list(binding.horizon_events),
            "planning_event": 5,
        },
        execution_permitted=False,
        data_snapshot_bound=True,
    )
    decision = cd.decide_chip_action(
        horizon_binding=binding,
        chip_availability=[{
            "name": "bboost", "available_for_event": True,
            "window_start_event": 2, "window_stop_event": 8,
            "used": False, "expired": False,
        }],
        evaluations={cd.CHIP_ACTION_BB: evaluation},
        reservation=reservation,
        certification_valid=True,
        manager_state={"squad_ids": list(range(1, 16))},
    )
    assert decision.calibration_status == cd.CALIBRATION_CALIBRATED
    assert decision.status != cd.STATUS_PLAY_CHIP
    assert cd.DIAG_CHIP_EVALUATOR_UNCALIBRATED in decision.reason_codes


def test_assessment_store_retains_each_run_and_detects_tampering(tmp_path):
    context = _context()
    manager_state = {
        "squad_ids": list(range(1, 16)), "bank_tenths": 70,
        "free_transfers": 3, "event_start_free_transfers": None,
    }
    decision = {
        "recommended_action": cd.CHIP_ACTION_NO_CHIP,
        "status": cd.STATUS_INSUFFICIENT_EVIDENCE,
        "calibration_status": cd.CALIBRATION_UNCALIBRATED,
    }
    first = store.build_assessment_record(
        context=context, manager_state=manager_state, chip_results=_blocked_results(),
        decision=decision, run_id="run-a", created_at="2026-09-30T00:00:00Z",
    )
    second = store.build_assessment_record(
        context=context, manager_state=manager_state, chip_results=_blocked_results(),
        decision=decision, run_id="run-b", created_at="2026-09-30T00:00:01Z",
    )
    first_receipt = store.retain_assessment(first, tmp_path)
    second_receipt = store.retain_assessment(second, tmp_path)
    assert first_receipt["path"] != second_receipt["path"]
    assert store.verify_assessment(first_receipt["path"])["verified"] is True
    assert store.verify_assessment(second_receipt["path"])["verified"] is True

    target = __import__("pathlib").Path(first_receipt["path"])
    changed = json.loads(target.read_text(encoding="utf-8"))
    changed["chip_results"][cd.CHIP_ACTION_BB]["detail"] = "altered"
    target.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(store.ChipAssessmentStoreError, match="digest differs"):
        store.verify_assessment(target)


@pytest.mark.parametrize(
    "event_start,captured_at",
    [(None, "2026-09-29T20:27:01Z"), (3, "2026-09-29T20:27:02Z")],
)
def test_production_preflight_rejects_missing_or_late_event_start_confirmation(
    monkeypatch, event_start, captured_at,
):
    cutoff = "2026-09-29T20:27:01Z"
    squad = list(range(1, 16))
    context = SimpleNamespace(squad={"players": [{"player_id": pid} for pid in squad]})
    manager_state = {
        "event_start_free_transfers": event_start,
        "free_transfers": 3,
        "bank": 7,
        "manual": {
            "event_start_free_transfers": event_start,
            "captured_at": captured_at,
        },
    }
    monkeypatch.setattr(assessment.planning, "get_planning_context", lambda *_args: context)
    monkeypatch.setattr(assessment.repo, "manager_planning_state", lambda *_args: manager_state)
    route = SimpleNamespace(
        planning_event=6, cutoff=cutoff, actual_owned_ids=tuple(squad),
    )
    with pytest.raises(assessment.ChipAssessmentPreflightError) as caught:
        assessment._manager_snapshot(object(), entry_id=241392, route=route)
    assert caught.value.action_refusals[cd.CHIP_ACTION_FH][0] == \
        "FREE_HIT_PRODUCTION_MANAGER_STATE_MISSING"
    assert caught.value.action_refusals[cd.CHIP_ACTION_WC][0] == \
        "WILDCARD_PRODUCTION_MANAGER_STATE_MISSING"
    assert "no prediction worlds or route optimizer were loaded" in str(caught.value) or \
        "not captured by the retained generation cutoff" in str(caught.value)


def test_production_preflight_reports_missing_pinned_squad_without_attribute_error(monkeypatch):
    cutoff = "2026-09-29T20:27:01Z"
    context = SimpleNamespace(squad=None)
    manager_state = {
        "event_start_free_transfers": 3,
        "free_transfers": 3,
        "bank": 7,
        "manual": {"event_start_free_transfers": 3, "captured_at": cutoff},
    }
    monkeypatch.setattr(assessment.planning, "get_planning_context", lambda *_args: context)
    monkeypatch.setattr(assessment.repo, "manager_planning_state", lambda *_args: manager_state)
    route = SimpleNamespace(planning_event=6, cutoff=cutoff, actual_owned_ids=tuple(range(1, 16)))
    with pytest.raises(assessment.ChipAssessmentPreflightError, match="pinned canonical squad is missing"):
        assessment._manager_snapshot(object(), entry_id=241392, route=route)


def test_bb_tc_assessment_refuses_mixed_hypothetical_scenarios():
    context = _context()
    availability = [
        {"name": "bboost", "available_for_event": True, "window_start_event": 2,
         "window_stop_event": 19, "used": False, "expired": False},
        {"name": "3xc", "available_for_event": True, "window_start_event": 2,
         "window_stop_event": 19, "used": False, "expired": False},
    ]
    base_evidence = {
        "certification_identity": context["certification_identity"],
        "data_snapshot_sha256": context["data_snapshot_sha256"],
        "planning_event": context["planning_event"],
        "horizon_events": context["horizon_events"],
        "scenario_identity": "route-001-scenario",
        "world_identity": "same-h1-worlds",
        "proposed_owned_ids": list(range(1, 16)),
        "lineup": {"starter_ids": list(range(1, 12)), "captain_id": 1, "vice_captain_id": 2},
    }
    bb = cd.ChipEvaluation(
        action=cd.CHIP_ACTION_BB, evaluator_version="fixture",
        candidate_metrics={"mean_paired_uplift": 2.0},
        uncertainty={"paired_interval_low": 1.0, "paired_interval_high": 3.0},
        evidence=dict(base_evidence), data_snapshot_bound=True,
    )
    tc_evidence = {**base_evidence, "scenario_identity": "captured-lineup-scenario"}
    tc = cd.ChipEvaluation(
        action=cd.CHIP_ACTION_TC, evaluator_version="fixture",
        candidate_metrics={"mean_paired_uplift": 3.0},
        uncertainty={"paired_interval_low": 1.0, "paired_interval_high": 5.0},
        evidence=tc_evidence, data_snapshot_bound=True,
    )
    with pytest.raises(store.ChipAssessmentStoreError, match="one scenario_identity"):
        assemble_assessment_record(
            context=context,
            manager_state={"squad_ids": list(range(1, 16))},
            availability=availability,
            evaluations={cd.CHIP_ACTION_BB: bb, cd.CHIP_ACTION_TC: tc},
        )


def test_assessment_store_verifies_both_retained_free_hit_arms_and_restoration():
    context = _context()
    manager_state = {
        "entry_id": 241392,
        "planning_event": 5,
        "cutoff": context["cutoff"],
        "squad_ids": list(range(1, 16)),
        "purchase_price_tenths": {pid: 50 for pid in range(1, 16)},
        "bank_tenths": 70,
        "free_transfers": 2,
        "event_start_free_transfers": 3,
    }
    manager_state["manager_state_identity"] = store.manager_state_identity(manager_state)
    context.update({"manager_state_identity": manager_state["manager_state_identity"]})
    shared = {
        "source_decision_id": context["source_decision_id"],
        "source_result_sha256": context["source_decision_result_sha256"],
        "source_artifact_sha256": context["source_decision_artifact_sha256"],
        "generation_id": context["generation_id"],
        "cutoff": context["cutoff"],
        "data_snapshot_sha256": context["data_snapshot_sha256"],
        "certification_identity": context["certification_identity"],
        "manager_state_identity": context["manager_state_identity"],
        "world_identity": "same-certified-h1-worlds",
    }
    play = {
        "arm": "PLAY", "source": "OPTIMIZED_RESTORED_PERMANENT_H2_STATE",
        "actions": [{"event": 6}, {"event": 7}, {"event": 8}], **shared,
        "start_state": {
            "event": 6, "squad_ids": list(range(1, 16)), "bank_tenths": 70,
            "free_transfers": 3, "purchase_price_tenths": {pid: 50 for pid in range(1, 16)},
        },
        "route_config": {"events": [6, 7, 8]},
    }
    save = {
        "arm": "SAVE", "source": "VERIFIED_NORMAL_DECISION_ROUTE",
        "actions": [{"event": 5}, {"event": 6}, {"event": 7}, {"event": 8}], **shared,
        "start_state": {
            "event": 5, "squad_ids": list(range(1, 16)), "bank_tenths": 70,
            "free_transfers": 2, "purchase_price_tenths": {pid: 50 for pid in range(1, 16)},
        },
        "route_config": {"events": [5, 6, 7, 8]},
    }
    restoration = {
        "permanent_squad_ids": list(range(1, 16)),
        "restored_squad_ids": list(range(1, 16)),
        "permanent_purchase_price_tenths": {pid: 50 for pid in range(1, 16)},
        "restored_purchase_price_tenths": {pid: 50 for pid in range(1, 16)},
        "permanent_bank_tenths": 70,
        "restored_bank_tenths": 70,
        "current_h1_free_transfers": 2,
        "event_start_h1_free_transfers": 3,
        "restored_h2_free_transfers": 3,
        "free_transfers_rule": "preserve event-start FT",
        "restoration_problems": [],
    }
    arm_identity = store.free_hit_arm_identity({
        "play": play, "save": save, "restore_at_h2": restoration,
    })
    arms = {
        "play": play,
        "save": save,
        "restore_at_h2": restoration,
        "arm_identity": arm_identity,
        "source_decision_id": context["source_decision_id"],
        "source_result_sha256": context["source_decision_result_sha256"],
        "source_artifact_sha256": context["source_decision_artifact_sha256"],
    }
    evaluation = cd.ChipEvaluation(
        action=cd.CHIP_ACTION_FH,
        evaluator_version="fixture-fh",
        candidate_metrics={"mean_paired_uplift": 2.0},
        uncertainty={"paired_interval_low": 1.0, "paired_interval_high": 3.0},
        evidence={
            "certification_identity": context["certification_identity"],
            "data_snapshot_sha256": context["data_snapshot_sha256"],
            "planning_event": context["planning_event"],
            "horizon_events": context["horizon_events"],
        },
        execution_permitted=False,
        data_snapshot_bound=True,
    )
    results = _blocked_results()
    results[cd.CHIP_ACTION_FH] = {
        "status": "EVALUATED", "evaluation": {
            "action": evaluation.action,
            "evaluator_version": evaluation.evaluator_version,
            "candidate_metrics": dict(evaluation.candidate_metrics),
            "uncertainty": dict(evaluation.uncertainty),
            "reason_codes": [], "calibration_status": evaluation.calibration_status,
            "evidence": dict(evaluation.evidence), "execution_permitted": False,
        },
        "reason_codes": [], "arm_evidence": arms, "arm_identity": arm_identity,
    }
    record = store.build_assessment_record(
        context=context,
        manager_state=manager_state,
        chip_results=results,
        decision={"recommended_action": cd.CHIP_ACTION_NO_CHIP,
                  "status": cd.STATUS_CHIP_REVIEW_REQUIRED,
                  "calibration_status": cd.CALIBRATION_UNCALIBRATED},
    )
    assert record["chip_results"][cd.CHIP_ACTION_FH]["arm_evidence"]["play"]["arm"] == "PLAY"
    assert record["chip_results"][cd.CHIP_ACTION_FH]["arm_evidence"]["save"]["arm"] == "SAVE"
    results[cd.CHIP_ACTION_FH]["arm_evidence"]["play"]["start_state"]["free_transfers"] = 99
    forged_arm_identity = store.free_hit_arm_identity(results[cd.CHIP_ACTION_FH]["arm_evidence"])
    results[cd.CHIP_ACTION_FH]["arm_evidence"]["arm_identity"] = forged_arm_identity
    results[cd.CHIP_ACTION_FH]["arm_identity"] = forged_arm_identity
    with pytest.raises(store.ChipAssessmentStoreError, match="PLAY start state"):
        store.build_assessment_record(
            context=context,
            manager_state=manager_state,
            chip_results=results,
            decision={"recommended_action": cd.CHIP_ACTION_NO_CHIP,
                      "status": cd.STATUS_CHIP_REVIEW_REQUIRED,
                      "calibration_status": cd.CALIBRATION_UNCALIBRATED},
        )
