"""Action-specific readiness metrics and production/fixture separation."""

from __future__ import annotations

import sqlite3
from datetime import datetime
from dataclasses import replace
from types import SimpleNamespace

import pytest

from fpl_brain import chip_decision as cd
from fpl_brain import chip_bench_boost as bb
from fpl_brain import chip_evaluator_readiness as readiness
from fpl_brain import chip_free_hit as fh
from fpl_brain import chip_reservation_calibration as crc
from fpl_brain import chip_wildcard as wc


# This file tests action-specific readiness criteria using simulated causal
# observations. It does not grant production authority; the real origin-store
# permission checks are tested separately.
_SIMULATED_STORE_CONN = sqlite3.connect(":memory:")


@pytest.fixture(autouse=True)
def _supply_store_context_for_simulated_readiness(monkeypatch):
    for name in ("build_evaluator_readiness", "verify_evaluator_readiness", "apply_verified_readiness"):
        original = getattr(readiness, name)

        def wrapped(*args, __original=original, **kwargs):
            kwargs.setdefault("store_conn", _SIMULATED_STORE_CONN)
            return __original(*args, **kwargs)

        monkeypatch.setattr(readiness, name, wrapped)


def _fixture_rows(action: str = cd.CHIP_ACTION_BB, evaluator_version: str = "bb-fixture-v1"):
    return [
        {
            "observation_id": f"fixture-{index}",
            "action": action,
            "source_decision_id": f"decision-{index}",
            "origin_cutoff": f"2026-01-{(index % 28) + 1:02d}T00:00:00Z",
            "label_available_at": f"2026-02-{(index % 28) + 1:02d}T00:00:00Z",
            "evaluator_version": evaluator_version,
            "evaluator_forecast_value": 5.0,
            "realized_value": 5.0,
            "interval_low": 4.0,
            "interval_high": 6.0,
            "causal_evidence_ref": f"fixture-evidence-{index}",
            "causal_evidence_sha256": f"{index:064x}",
        }
        for index in range(30)
    ]


def test_fixture_readiness_metrics_can_pass_but_cannot_grant_production_permission():
    artifact = readiness.build_fixture_readiness(
        _fixture_rows(),
        action=cd.CHIP_ACTION_BB,
        evaluator_version="bb-fixture-v1",
    )
    assert artifact["status"] == readiness.READINESS_READY
    assert artifact["execution_permitted"] is True
    assert artifact["evidence_class"] == "FIXTURE_ONLY"
    assert artifact["production_usable"] is False
    with pytest.raises(readiness.EvaluatorReadinessError, match="fixture readiness"):
        readiness.verify_retained_readiness_integrity(artifact)


def test_action_specific_readiness_refuses_incompatible_versions_and_small_samples():
    artifact = readiness.build_fixture_readiness(
        _fixture_rows(),
        action=cd.CHIP_ACTION_BB,
        evaluator_version="bb-fixture-v1",
    )
    assert artifact["action"] == cd.CHIP_ACTION_BB
    assert artifact["evaluator_version"] == "bb-fixture-v1"

    too_small = readiness.build_fixture_readiness(
        _fixture_rows()[:12],
        action=cd.CHIP_ACTION_FH,
        evaluator_version="fh-fixture-v1",
    )
    assert too_small["status"] == readiness.READINESS_INSUFFICIENT
    assert too_small["execution_permitted"] is False


def test_failed_evaluator_readiness_is_independent_of_calibrated_reservation():
    binding = cd.ChipHorizonBinding(
        planning_event=5,
        horizon_events=(5, 6, 7, 8),
        certification_identity="cert",
        data_snapshot_sha256="snapshot",
    )

    class CalibratedReservation:
        def estimate(self, *, action, planning_event, expiry_event, state):
            return cd.ReservationEstimate(
                value=0.0,
                calibration_status=cd.CALIBRATION_CALIBRATED,
                terminal_value=0.0,
                weeks_to_expiry=10,
            )

    evaluation = cd.ChipEvaluation(
        action=cd.CHIP_ACTION_BB,
        evaluator_version="bb-fixture-v1",
        candidate_metrics={"mean_paired_uplift": 8.0},
        evidence={
            "certification_identity": "cert",
            "data_snapshot_sha256": "snapshot",
            "planning_event": 5,
            "horizon_events": [5, 6, 7, 8],
        },
        calibration_status=cd.CALIBRATION_UNCALIBRATED,
        execution_permitted=False,
        data_snapshot_bound=True,
    )
    decision = cd.decide_chip_action(
        horizon_binding=binding,
        chip_availability=[{
            "name": "bboost", "available_for_event": True, "used": False,
            "expired": False, "window_start_event": 1, "window_stop_event": 19,
        }],
        evaluations={cd.CHIP_ACTION_BB: evaluation},
        reservation=CalibratedReservation(),
        certification_valid=True,
        manager_state={"squad_ids": list(range(1, 16))},
    )
    comparison = decision.candidate_metrics["candidate_comparisons"][cd.CHIP_ACTION_BB]
    assert comparison["reservation_calibration_status"] == cd.CALIBRATION_CALIBRATED
    assert comparison["execution_permitted"] is False
    assert comparison["status"] == "UNRANKABLE"
    assert decision.status == cd.STATUS_CHIP_REVIEW_REQUIRED
    assert decision.recommended_action == cd.CHIP_ACTION_BB


def _install_fixture_causal_validator(monkeypatch):
    """Simulate the verified causal-ledger boundary; never production evidence."""

    def validate(raw, *, evidence_verifier, store_conn=None):
        evidence_verifier(str(raw["causal_evidence_ref"]))
        parse = lambda name: datetime.fromisoformat(str(raw[name]).replace("Z", "+00:00"))
        return SimpleNamespace(
            observation_id=str(raw["observation_id"]),
            action=str(raw["action"]),
            forecast_mode="PROSPECTIVE",
            label_available_at=parse("label_available_at"),
            evaluator_version=str(raw["evaluator_version"]),
            evaluator_forecast_value=float(raw["evaluator_forecast_value"]),
            evaluator_interval_low=float(raw["interval_low"]),
            evaluator_interval_high=float(raw["interval_high"]),
            evaluator_value_definition=raw.get("opportunity_value_definition"),
            evaluator_interval_value_definition=raw.get("uncertainty_value_definition"),
            source_decision_id=str(raw["source_decision_id"]),
            planning_event=int(raw["planning_event"]),
            origin_cutoff=parse("origin_cutoff"),
            realized_value=float(raw["realized_value"]),
            evidence_ref=str(raw["causal_evidence_ref"]),
            evidence_sha256=str(raw["causal_evidence_sha256"]),
        )

    monkeypatch.setattr(crc, "validate_observation", validate)


def _production_shaped_fixture_rows(action: str, evaluator_version: str, *, count: int = 30):
    return [
        {
            "observation_id": f"simulated-ledger-{index}",
            "action": action,
            "planning_event": 1 + index,
            "source_decision_id": f"simulated-decision-{index}",
            "origin_cutoff": f"2026-01-01T00:{index:02d}:00Z",
            "label_available_at": f"2026-02-01T00:{index:02d}:00Z",
            "evaluator_version": evaluator_version,
            "evaluator_forecast_value": 5.0,
            "realized_value": 5.0,
            "interval_low": 4.0,
            "interval_high": 6.0,
            "opportunity_value_definition": readiness.ACTION_VALUE_DEFINITIONS[action],
            "uncertainty_value_definition": readiness.ACTION_VALUE_DEFINITIONS[action],
            "causal_evidence_ref": f"simulated-causal-pair-{index}",
            "causal_evidence_sha256": f"{index + 1:064x}",
        }
        for index in range(count)
    ]


def _ready_review_evaluation(action: str, version: str) -> cd.ChipEvaluation:
    version_key = {
        cd.CHIP_ACTION_BB: "evaluator_version",
        cd.CHIP_ACTION_FH: "free_hit_evaluator_version",
        cd.CHIP_ACTION_WC: "wildcard_evaluator_version",
    }[action]
    reasons = {
        cd.CHIP_ACTION_BB: (bb.DIAG_BB_EVALUATED, bb.DIAG_BB_REVIEW_ONLY),
        cd.CHIP_ACTION_FH: (
            fh.DIAG_FH_EVALUATED,
            fh.DIAG_FH_REVIEW_ONLY,
            fh.DIAG_FH_BOUNDED_SEARCH,
        ),
        cd.CHIP_ACTION_WC: (
            wc.WC_REASON_PLAY_DESCRIPTION,
            wc.WC_REASON_REVIEW_ONLY,
            wc.WC_REASON_POSITIVE,
        ),
    }[action]
    return cd.ChipEvaluation(
        action=action,
        evaluator_version=version,
        candidate_metrics={"mean_paired_uplift": 8.0},
        reason_codes=reasons,
        execution_permitted=False,
        data_snapshot_bound=True,
        evidence={
            "certification_identity": f"cert-{action}",
            "data_snapshot_sha256": "a" * 64,
            "planning_event": 5,
            "horizon_events": list(cd.canonical_chip_horizon(5)),
            version_key: version,
        },
    )


@pytest.mark.parametrize("action", [cd.CHIP_ACTION_BB, cd.CHIP_ACTION_FH, cd.CHIP_ACTION_WC])
def test_fixture_causal_readiness_permits_only_its_matching_action_gate(monkeypatch, action):
    _install_fixture_causal_validator(monkeypatch)
    version = f"{action.lower()}-fixture-evaluator-v1"
    observations = _production_shaped_fixture_rows(action, version)
    verifier = lambda _reference: {"verified": True}
    artifact = readiness.build_evaluator_readiness(
        observations,
        action=action,
        evaluator_version=version,
        evidence_verifier=verifier,
        evaluation_cutoff="2026-03-01T00:00:00Z",
    )
    assert artifact["status"] == readiness.READINESS_READY
    assert artifact["execution_permitted"] is True
    assert artifact["evidence_class"] == "PRODUCTION"  # simulated validator only

    evaluation = _ready_review_evaluation(action, version)
    permitted, gate = readiness.apply_verified_readiness(
        evaluation,
        artifact,
        observations,
        current_cutoff="2026-03-02T00:00:00Z",
        evidence_verifier=verifier,
    )
    assert gate["execution_permitted"] is True
    assert permitted.execution_permitted is True
    assert permitted.calibration_status == cd.CALIBRATION_CALIBRATED
    assert "CHIP_EVALUATOR_READINESS_VERIFIED" in permitted.reason_codes


def test_interval_coverage_uses_absolute_production_bounds_and_refuses_old_fh_units(monkeypatch):
    rows = _production_shaped_fixture_rows(cd.CHIP_ACTION_BB, "bb-fixture-v1")
    for row in rows:
        row["evaluator_forecast_value"] = 1.0
        row["realized_value"] = 1.9
        row["interval_low"] = 0.8
        row["interval_high"] = 1.2
    outside = readiness.build_fixture_readiness(
        rows,
        action=cd.CHIP_ACTION_BB,
        evaluator_version="bb-fixture-v1",
    )
    assert outside["metrics"]["interval_coverage"] == 0.0
    assert outside["status"] == readiness.READINESS_FAILED

    _install_fixture_causal_validator(monkeypatch)
    fh_version = "fh-fixture-v1"
    old_unit_rows = _production_shaped_fixture_rows(cd.CHIP_ACTION_FH, fh_version)
    for row in old_unit_rows:
        row.pop("uncertainty_value_definition")
    old_units = readiness.build_evaluator_readiness(
        old_unit_rows,
        action=cd.CHIP_ACTION_FH,
        evaluator_version=fh_version,
        evidence_verifier=lambda _reference: {"verified": True},
        evaluation_cutoff="2026-03-01T00:00:00Z",
    )
    assert old_units["status"] == readiness.READINESS_INVALID
    assert old_units["execution_permitted"] is False


@pytest.mark.parametrize("action", [cd.CHIP_ACTION_BB, cd.CHIP_ACTION_FH, cd.CHIP_ACTION_WC])
def test_ready_artifact_cannot_upgrade_a_refused_current_evaluation(monkeypatch, action):
    _install_fixture_causal_validator(monkeypatch)
    version = f"{action.lower()}-fixture-evaluator-v1"
    observations = _production_shaped_fixture_rows(action, version)
    verifier = lambda _reference: {"verified": True}
    artifact = readiness.build_evaluator_readiness(
        observations,
        action=action,
        evaluator_version=version,
        evidence_verifier=verifier,
        evaluation_cutoff="2026-03-01T00:00:00Z",
    )
    review_only_reason = {
        cd.CHIP_ACTION_BB: bb.DIAG_BB_REVIEW_ONLY,
        cd.CHIP_ACTION_FH: fh.DIAG_FH_REVIEW_ONLY,
        cd.CHIP_ACTION_WC: wc.WC_REASON_REVIEW_ONLY,
    }[action]
    refused_evaluation = replace(
        _ready_review_evaluation(action, version),
        candidate_metrics={"mean_paired_uplift": None},
        reason_codes=(review_only_reason, "CURRENT_ACTION_INPUT_REFUSED"),
    )

    unchanged, gate = readiness.apply_verified_readiness(
        refused_evaluation,
        artifact,
        observations,
        current_cutoff="2026-03-02T00:00:00Z",
        evidence_verifier=verifier,
    )

    assert unchanged is refused_evaluation
    assert unchanged.execution_permitted is False
    assert gate["status"] == readiness.READINESS_CURRENT_EVALUATION_REFUSED
    assert gate["execution_permitted"] is False
    assert gate["reason_code"] == readiness.READINESS_CURRENT_EVALUATION_REFUSED


def test_fixture_causal_readiness_refuses_insufficient_or_version_mismatched_evidence(monkeypatch):
    _install_fixture_causal_validator(monkeypatch)
    action = cd.CHIP_ACTION_WC
    version = "wc-fixture-evaluator-v1"
    verifier = lambda _reference: {"verified": True}
    insufficient = _production_shaped_fixture_rows(action, version, count=12)
    blocked_evaluation = cd.ChipEvaluation(
        action=action,
        evaluator_version=version,
        candidate_metrics={"mean_paired_uplift": 8.0},
        execution_permitted=False,
    )
    insufficient_artifact = readiness.build_evaluator_readiness(
        insufficient,
        action=action,
        evaluator_version=version,
        evidence_verifier=verifier,
        evaluation_cutoff="2026-03-01T00:00:00Z",
    )
    unchanged, insufficient_gate = readiness.apply_verified_readiness(
        blocked_evaluation,
        insufficient_artifact,
        insufficient,
        current_cutoff="2026-03-02T00:00:00Z",
        evidence_verifier=verifier,
    )
    assert insufficient_gate["execution_permitted"] is False
    assert unchanged is blocked_evaluation

    incompatible = _production_shaped_fixture_rows(action, "wc-old-evaluator-v0")
    incompatible_artifact = readiness.build_evaluator_readiness(
        incompatible,
        action=action,
        evaluator_version=version,
        evidence_verifier=verifier,
        evaluation_cutoff="2026-03-01T00:00:00Z",
    )
    unchanged, incompatible_gate = readiness.apply_verified_readiness(
        blocked_evaluation,
        incompatible_artifact,
        incompatible,
        current_cutoff="2026-03-02T00:00:00Z",
        evidence_verifier=verifier,
    )
    assert incompatible_artifact["status"] == readiness.READINESS_INVALID
    assert incompatible_gate["execution_permitted"] is False
    assert unchanged is blocked_evaluation


@pytest.mark.parametrize("action", [cd.CHIP_ACTION_BB, cd.CHIP_ACTION_FH, cd.CHIP_ACTION_WC])
def test_future_readiness_artifact_cannot_authorize_earlier_assessment(monkeypatch, action):
    _install_fixture_causal_validator(monkeypatch)
    version = f"{action.lower()}-fixture-evaluator-v1"
    observations = _production_shaped_fixture_rows(action, version)
    verifier = lambda _reference: {"verified": True}
    artifact = readiness.build_evaluator_readiness(
        observations,
        action=action,
        evaluator_version=version,
        evidence_verifier=verifier,
        evaluation_cutoff="2026-03-01T00:00:00Z",
    )
    evaluation = _ready_review_evaluation(action, version)

    unchanged, gate = readiness.apply_verified_readiness(
        evaluation,
        artifact,
        observations,
        current_cutoff="2026-02-28T23:59:59Z",
        evidence_verifier=verifier,
    )

    assert unchanged is evaluation
    assert unchanged.execution_permitted is False
    assert gate["status"] == readiness.READINESS_CURRENT_EVALUATION_REFUSED
    assert gate["execution_permitted"] is False
    assert gate["reason_code"] == readiness.READINESS_CURRENT_EVALUATION_REFUSED


def test_historical_replay_cannot_establish_evaluator_readiness(monkeypatch):
    _install_fixture_causal_validator(monkeypatch)
    prospective_validator = crc.validate_observation

    def historical_validator(raw, *, evidence_verifier, store_conn=None):
        row = prospective_validator(
            raw, evidence_verifier=evidence_verifier, store_conn=store_conn,
        )
        return SimpleNamespace(**{**vars(row), "forecast_mode": "HISTORICAL_REPLAY"})

    monkeypatch.setattr(crc, "validate_observation", historical_validator)
    action = cd.CHIP_ACTION_BB
    version = "bb-fixture-evaluator-v1"
    artifact = readiness.build_evaluator_readiness(
        _production_shaped_fixture_rows(action, version),
        action=action,
        evaluator_version=version,
        evidence_verifier=lambda _reference: {"verified": True},
        evaluation_cutoff="2026-03-01T00:00:00Z",
    )
    assert artifact["status"] == readiness.READINESS_INVALID
    assert artifact["execution_permitted"] is False
    assert "historical replay cannot establish evaluator readiness" in artifact["rejected_observations"][0]["reason"]
