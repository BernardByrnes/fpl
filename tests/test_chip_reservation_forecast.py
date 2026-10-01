from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from fpl_brain import chip_decision as cd
from fpl_brain import chip_assessment as assessment
from fpl_brain import chip_assessment_store as assessment_store
from fpl_brain import chip_reservation_forecast as forecast
from fpl_brain import chip_reservation_calibration as calibration
from fpl_brain import chip_triple_captain as tc
from fpl_brain import manager_lineup
from fpl_brain import search_permission as sp
from scripts import run_chip_assessment
import generation_fixtures as gf


def _sha(value):
    return forecast.canonical_sha256(value)


def _source():
    return {
        "source_decision_id": "decision-1",
        "source_result_sha256": _sha({"result": 1}),
        "source_artifact_sha256": _sha({"artifact": 1}),
        "generation_id": "generation-1",
        "planning_event": 6,
        "origin_cutoff": "2026-10-01T08:00:00Z",
        "data_snapshot_sha256": _sha({"snapshot": 1}),
        "predictive_code_snapshot_sha256": _sha({"code": 1}),
        "certification_identity": _sha({"certification": 1}),
    }


def _inputs(*, coverage_product_sha256=None):
    source = _source()
    state = {
        "squad_ids": list(range(1, 16)),
    }
    records = {}
    for event, value in ((7, 4.0), (8, 6.0)):
        artifact = forecast.build_event_opportunity_record(
            action=cd.CHIP_ACTION_TC,
            planning_event=6,
            event=event,
            origin_cutoff="2026-10-01T08:00:00Z",
            made_at="2026-10-01T08:00:05Z",
            input_as_of="2026-10-01T08:00:00Z",
            expected_incremental_points=value,
            opportunity_model="fixture_tc_expected_captain_uplift_v1",
            source_identity=source,
            reservation_state=state,
            world_identity=_sha({"world": event}),
            coverage_product_sha256=coverage_product_sha256,
        )
        records[f"event-{event}"] = artifact
    return source, state, records


def _fixture_coverage_product(source, *, action, expiry_event, product_last=None, horizon_length=1):
    origin = int(source["planning_event"])
    required_last = origin - 1 if expiry_event is None else int(expiry_event) + int(horizon_length) - 1
    if product_last is None:
        product_last = max(origin + 3, required_last)
    events = list(range(origin, int(product_last) + 1))
    required = [] if expiry_event is None else list(range(origin, required_last + 1))
    missing = [event for event in required if event not in events]
    coverage_known = expiry_event is not None
    coverage_complete = bool(coverage_known and not missing)
    body = {
        "schema": forecast.RESERVATION_COVERAGE_PRODUCT_SCHEMA,
        "action": action,
        "planning_event": origin,
        "expiry_event": expiry_event,
        "forecast_events": [] if expiry_event is None else list(range(origin + 1, int(expiry_event) + 1)),
        "opportunity_horizon_length": int(horizon_length),
        "required_product_events": required,
        "product_events": events,
        "missing_events": missing,
        "coverage_known": coverage_known,
        "coverage_complete": coverage_complete,
        "coverage_status": forecast.FORECAST_READY if coverage_complete else forecast.FORECAST_INCOMPLETE,
        "coverage_reason": (
            None if coverage_complete
            else "CHIP_EXPIRY_UNKNOWN" if not coverage_known
            else "REQUIRED_PRODUCT_EVENTS_MISSING"
        ),
        "source_identity": source,
        "root_generation_id": source["generation_id"],
        "product_generation_id": "fixture-continuation-product",
        "product_generation_manifest_sha256": "fixture-continuation-product",
        "origin_cutoff": source["origin_cutoff"],
        "input_as_of": source["origin_cutoff"],
        "data_snapshot_sha256": source["data_snapshot_sha256"],
        "predictive_code_snapshot_sha256": source["predictive_code_snapshot_sha256"],
        "product_runs_by_event": {str(event): {"fixture": event} for event in events},
        "wildcard_value_horizon_length": horizon_length if action == cd.CHIP_ACTION_WC else None,
    }
    body["product_sha256"] = _sha(body)
    return body


def _build(records, *, expiry_event, product_last=None):
    source, state, _ = _inputs()
    coverage_product = (
        None if expiry_event is None else _fixture_coverage_product(
            source, action=cd.CHIP_ACTION_TC, expiry_event=expiry_event,
            product_last=product_last,
        )
    )
    return forecast.build_reservation_forecast(
        action=cd.CHIP_ACTION_TC,
        planning_event=6,
        origin_cutoff="2026-10-01T08:00:00Z",
        made_at="2026-10-01T08:00:30Z",
        input_as_of="2026-10-01T08:00:00Z",
        expiry_event=expiry_event,
        source_identity=source,
        reservation_state=state,
        opportunity_refs=tuple(records),
        evidence_verifier=records.__getitem__,
        coverage_product=coverage_product,
    )


def test_raw_forecast_is_content_addressed_and_uses_only_origin_expected_values():
    source, state, records = _inputs()
    artifact = _build(records, expiry_event=8)
    assert artifact["coverage_status"] == forecast.FORECAST_READY
    assert artifact["raw_value"] == 6.0
    assert artifact["selected_event"] == 8
    assert artifact["artifact_sha256"] == _sha({key: value for key, value in artifact.items()
                                                if key != "artifact_sha256"})
    result = forecast.verify_reservation_forecast(
        artifact,
        expected={
            "action": cd.CHIP_ACTION_TC,
            "planning_event": 6,
            "origin_cutoff": "2026-10-01T08:00:00Z",
            "expiry_event": 8,
            "source_identity": source,
        },
        evidence_verifier=records.__getitem__,
    )
    assert result["verified"] is True
    verified = forecast.VerifiedReservationForecast(artifact)
    verified.validate_for_reservation(
        action=cd.CHIP_ACTION_TC, planning_event=6, expiry_event=8, state=state,
    )


def test_complete_nonpositive_forecast_still_binds_earliest_best_event_for_maturation():
    source, state, _ = _inputs()
    records = {}
    for event, value in ((7, -4.0), (8, -2.0)):
        records[f"event-{event}"] = forecast.build_event_opportunity_record(
            action=cd.CHIP_ACTION_TC,
            planning_event=6,
            event=event,
            origin_cutoff="2026-10-01T08:00:00Z",
            made_at="2026-10-01T08:00:05Z",
            input_as_of="2026-10-01T08:00:00Z",
            expected_incremental_points=value,
            opportunity_model="fixture_tc_expected_captain_uplift_v1",
            source_identity=source,
            reservation_state=state,
            world_identity=_sha({"world": event}),
        )
    artifact = _build(records, expiry_event=8)
    assert artifact["raw_value"] == 0.0
    assert artifact["selected_event"] == 8
    forecast.verify_reservation_forecast(artifact, expected={})


@pytest.mark.parametrize("action", [cd.CHIP_ACTION_BB, cd.CHIP_ACTION_TC])
def test_certified_evaluator_builds_and_retains_future_event_causal_origin(action, tmp_path, monkeypatch):
    source = _source()
    monkeypatch.setattr(
        sp,
        "require_search_permission",
        lambda _conn, _generation_id, **_kwargs: gf.fixture_search_permission_evaluation(
            source, events=tuple(range(6, 10))
        ),
    )
    positions = {
        1: "GKP", 2: "GKP", 3: "DEF", 4: "DEF", 5: "DEF",
        6: "DEF", 7: "DEF", 8: "MID", 9: "MID", 10: "MID",
        11: "MID", 12: "MID", 13: "FWD", 14: "FWD", 15: "FWD",
    }
    policy = manager_lineup.ManagerPolicy(
        starter_ids=(1, 3, 4, 5, 8, 9, 10, 11, 13, 14, 15),
        bench_gk_id=2,
        bench_outfield_order=(6, 12, 7),
        captain_id=8,
        vice_captain_id=9,
    )
    world_identity = _sha({"certified event": 7})
    worlds = cd.ChipWorldInputs(
        worlds=4,
        player_ids=tuple(range(1, 16)),
        minutes={pid: (90.0, 90.0, 0.0, 90.0) for pid in range(1, 16)},
        core={pid: (float(pid), float(pid + 1), 0.0, float(pid + 2)) for pid in range(1, 16)},
        planning_event=6,
        horizon_events=cd.canonical_chip_horizon(6),
        certification_identity=source["certification_identity"],
        data_snapshot_sha256=source["data_snapshot_sha256"],
        world_seed=104,
        world_identity=world_identity,
        code_snapshot_sha256=source["predictive_code_snapshot_sha256"],
        source_event=7,
    )
    binding = cd.ChipHorizonBinding(
        planning_event=6,
        horizon_events=cd.canonical_chip_horizon(6),
        certification_identity=source["certification_identity"],
        data_snapshot_sha256=source["data_snapshot_sha256"],
    )
    reservation_state = {"squad_ids": list(range(1, 16))}
    conn = sqlite3.connect(":memory:")
    try:
        event_record = forecast.build_evaluated_event_opportunity_record(
            action=action,
            event=7,
            worlds=worlds,
            horizon_binding=binding,
            policy=policy,
            positions=positions,
            source_identity=source,
            reservation_state=reservation_state,
            made_at="2026-10-01T08:00:05Z",
            input_as_of="2026-10-01T08:00:00Z",
            conn=conn,
        )
    finally:
        conn.close()
    play_policy = event_record["outcome_arms"]["play"]["lineup"]
    save_policy = event_record["outcome_arms"]["save"]["lineup"]
    assert play_policy == save_policy
    if action == cd.CHIP_ACTION_TC:
        expected = tc.evaluate_triple_captain(tc.TripleCaptainRequest(
            worlds=worlds,
            horizon_binding=binding,
            policy=policy,
            positions=positions,
            chip_available=True,
        ))
        assert play_policy["captain_id"] == expected.candidate_metrics["captain_id"]
        assert play_policy["vice_captain_id"] == expected.candidate_metrics["vice_captain_id"]
    else:
        assert play_policy == policy.as_dict()
    records = {"opportunity-7.json": event_record}
    artifact = forecast.build_reservation_forecast(
        action=action,
        planning_event=6,
        origin_cutoff=source["origin_cutoff"],
        made_at="2026-10-01T08:00:30Z",
        input_as_of="2026-10-01T08:00:00Z",
        expiry_event=7,
        source_identity=source,
        reservation_state=reservation_state,
        opportunity_refs=tuple(records),
        evidence_verifier=records.__getitem__,
        coverage_product=_fixture_coverage_product(
            source, action=action, expiry_event=7, product_last=7,
        ),
    )
    retained = calibration.retain_causal_origin_observation(
        observation_id=f"{action.lower()}-origin-1",
        forecast_artifact=artifact,
        evidence_root=tmp_path,
        evidence_verifier=records.__getitem__,
    )
    causal = retained["causal_evidence"]
    assert retained["selected_event"] == 7
    assert causal["label"] is None
    assert causal["source"]["world_identity"] == world_identity
    assert _sha(causal["counterfactual_pair"]["play"]["lineup"]) == _sha(play_policy)
    assert _sha(causal["counterfactual_pair"]["save"]["lineup"]) == _sha(save_policy)
    assert (tmp_path / retained["forecast_ref"]).is_file()
    assert (tmp_path / retained["causal_evidence_ref"]).is_file()


@pytest.mark.parametrize("expiry_event", [9, None])
def test_short_or_unknown_coverage_remains_unknown_not_zero(expiry_event):
    _, _, records = _inputs()
    partial = {"event-7": records["event-7"]}
    artifact = _build(
        partial, expiry_event=expiry_event,
        product_last=8 if expiry_event is not None else None,
    )
    assert artifact["coverage_complete"] is False
    assert artifact["coverage_status"] == forecast.FORECAST_INCOMPLETE
    assert artifact["raw_value"] is None
    assert artifact["reason_code"] == "CHIP_RESERVATION_FORECAST_COVERAGE_INCOMPLETE"
    forecast.verify_reservation_forecast(artifact, expected={})


def test_unknown_expiry_coverage_product_is_explicitly_incomplete_even_with_future_runs():
    source, state, _ = _inputs()
    product = _fixture_coverage_product(
        source,
        action=cd.CHIP_ACTION_TC,
        expiry_event=None,
        product_last=12,
    )
    artifact = forecast.build_reservation_forecast(
        action=cd.CHIP_ACTION_TC,
        planning_event=6,
        origin_cutoff="2026-10-01T08:00:00Z",
        made_at="2026-10-01T08:00:30Z",
        input_as_of="2026-10-01T08:00:00Z",
        expiry_event=None,
        source_identity=source,
        reservation_state=state,
        opportunity_refs=(),
        evidence_verifier={}.__getitem__,
        coverage_product=product,
    )
    assert product["coverage_known"] is False
    assert product["coverage_complete"] is False
    assert product["coverage_reason"] == "CHIP_EXPIRY_UNKNOWN"
    assert artifact["coverage_complete"] is False
    assert artifact["raw_value"] is None
    assert artifact["reason_code"] == "CHIP_RESERVATION_FORECAST_COVERAGE_INCOMPLETE"
    forecast.verify_reservation_forecast(artifact, expected={})


def test_complete_prediction_product_with_missing_event_forecast_stays_unknown():
    source, state, _ = _inputs()
    product = _fixture_coverage_product(
        source,
        action=cd.CHIP_ACTION_TC,
        expiry_event=8,
        product_last=8,
    )
    source, state, records = _inputs(coverage_product_sha256=product["product_sha256"])
    artifact = forecast.build_reservation_forecast(
        action=cd.CHIP_ACTION_TC,
        planning_event=6,
        origin_cutoff="2026-10-01T08:00:00Z",
        made_at="2026-10-01T08:00:30Z",
        input_as_of="2026-10-01T08:00:00Z",
        expiry_event=8,
        source_identity=source,
        reservation_state=state,
        opportunity_refs=["event-7"],
        evidence_verifier=records.__getitem__,
        coverage_product=product,
    )
    assert product["coverage_complete"] is True
    assert artifact["coverage_complete"] is False
    assert artifact["coverage_status"] == forecast.FORECAST_INCOMPLETE
    assert artifact["raw_value"] is None
    forecast.verify_reservation_forecast(artifact, expected={})


def test_prospective_forecast_may_issue_after_cutoff_but_not_use_later_inputs():
    source, state, records = _inputs()
    prospective = forecast.build_event_opportunity_record(
        action=cd.CHIP_ACTION_TC,
        planning_event=6,
        event=7,
        origin_cutoff="2026-10-01T08:00:00Z",
        input_as_of="2026-10-01T08:00:00Z",
        made_at="2026-10-01T08:00:01Z",
        expected_incremental_points=4.0,
        opportunity_model="fixture_tc_expected_captain_uplift_v1",
        source_identity=source,
        reservation_state=state,
        world_identity=_sha({"world": 7}),
    )
    assert prospective["input_as_of"] == "2026-10-01T08:00:00Z"
    assert prospective["made_at"] == "2026-10-01T08:00:01Z"
    forecast._verify_event_opportunity(
        prospective, action=cd.CHIP_ACTION_TC, planning_event=6,
        origin_cutoff=source["origin_cutoff"], source_identity=source,
        reservation_state=state,
    )
    with pytest.raises(forecast.ReservationForecastError, match="later than the origin cutoff"):
        forecast.build_event_opportunity_record(
            action=cd.CHIP_ACTION_TC,
            planning_event=6,
            event=7,
            origin_cutoff="2026-10-01T08:00:00Z",
            input_as_of="2026-10-01T08:00:01Z",
            made_at="2026-10-01T08:00:01Z",
            expected_incremental_points=4.0,
            opportunity_model="fixture_tc_expected_captain_uplift_v1",
            source_identity=source,
            reservation_state=state,
            world_identity=_sha({"world": 7}),
        )

    mismatch = json.loads(json.dumps(records["event-7"]))
    mismatch["reservation_state"]["free_transfers"] = 0
    mismatch["artifact_sha256"] = _sha({key: value for key, value in mismatch.items()
                                         if key != "artifact_sha256"})
    with pytest.raises(forecast.ReservationForecastError, match="SAVE-state digest does not verify"):
        _build({"mismatch": mismatch}, expiry_event=7)


def test_forecast_verification_detects_tampering_and_assessment_substitution():
    source, _, records = _inputs()
    artifact = _build(records, expiry_event=8)
    forged = json.loads(json.dumps(artifact))
    forged["raw_value"] = 0.0
    with pytest.raises(forecast.ReservationForecastError, match="content digest"):
        forecast.verify_reservation_forecast(forged, expected={})
    with pytest.raises(forecast.ReservationForecastError, match="assessment context"):
        forecast.verify_reservation_forecast(
            artifact,
            expected={"source_identity": {**source, "generation_id": "other-generation"}},
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("model", "unregistered_model", "model is unsupported"),
        ("selected_event", 7, "value does not reproduce"),
        ("selection_policy", "select the largest realized outcome", "selection policy is unsupported"),
    ],
)
def test_forecast_verification_rejects_rehashed_semantic_substitution(field, value, message):
    _, _, records = _inputs()
    artifact = _build(records, expiry_event=8)
    forged = json.loads(json.dumps(artifact))
    forged[field] = value
    forged["artifact_sha256"] = _sha({key: item for key, item in forged.items()
                                      if key != "artifact_sha256"})
    with pytest.raises(forecast.ReservationForecastError, match=message):
        forecast.verify_reservation_forecast(forged, expected={})


def test_forecast_retention_uses_immutable_digest_path(tmp_path):
    _, _, records = _inputs()
    artifact = _build(records, expiry_event=8)
    receipt = forecast.retain_reservation_forecast(artifact, tmp_path)
    assert receipt["artifact_sha256"] == artifact["artifact_sha256"]
    retained = json.loads((tmp_path / Path(receipt["path"]).name).read_text(encoding="utf-8"))
    assert retained == artifact


def test_assessment_cli_discovers_one_content_addressed_action_forecast(tmp_path):
    _, _, records = _inputs()
    artifact = _build(records, expiry_event=8)
    forecast.retain_reservation_forecast(artifact, tmp_path)
    conn = sqlite3.connect(":memory:")
    try:
        loaded, verifier = run_chip_assessment._load_reservation_forecast_inputs(tmp_path, conn)
        assert loaded == {cd.CHIP_ACTION_TC: artifact}
        assert callable(verifier)
    finally:
        conn.close()


def test_assessment_cli_refuses_ambiguous_content_addressed_forecasts(tmp_path):
    _, _, records = _inputs()
    first = _build(records, expiry_event=8)
    partial = {"event-7": records["event-7"]}
    second = _build(partial, expiry_event=9, product_last=8)
    forecast.retain_reservation_forecast(first, tmp_path)
    forecast.retain_reservation_forecast(second, tmp_path)
    conn = sqlite3.connect(":memory:")
    try:
        with pytest.raises(ValueError, match="multiple retained reservation forecasts found for TC"):
            run_chip_assessment._load_reservation_forecast_inputs(tmp_path, conn)
    finally:
        conn.close()


def test_production_assessment_retains_the_verified_raw_forecast_and_arbiter_binding(tmp_path):
    source, state, opportunities = _inputs()
    artifact = _build(opportunities, expiry_event=8)
    context = {
        "planning_event": 6,
        "horizon_events": list(cd.canonical_chip_horizon(6)),
        "cutoff": source["origin_cutoff"],
        "data_snapshot_sha256": source["data_snapshot_sha256"],
        "certification_identity": source["certification_identity"],
        "generation_id": source["generation_id"],
        "predictive_code_snapshot_sha256": source["predictive_code_snapshot_sha256"],
        "source_decision_id": source["source_decision_id"],
        "source_decision_result_sha256": source["source_result_sha256"],
        "source_decision_artifact_sha256": source["source_artifact_sha256"],
    }
    evaluation = cd.ChipEvaluation(
        action=cd.CHIP_ACTION_TC,
        evaluator_version="fixture-tc",
        candidate_metrics={"mean_paired_uplift": 8.0},
        uncertainty={"paired_interval_low": 3.0, "paired_interval_high": 10.0},
        evidence={
            "planning_event": 6,
            "horizon_events": list(context["horizon_events"]),
            "certification_identity": context["certification_identity"],
            "data_snapshot_sha256": context["data_snapshot_sha256"],
        },
    )
    availability = [{
        "name": cd.CHIP_ACTION_TO_OFFICIAL_NAME[cd.CHIP_ACTION_TC],
        "available_for_event": True,
        "window_start_event": 2,
        "window_stop_event": 8,
        "used": False,
        "expired": False,
    }]
    retained = assessment.assemble_assessment_record(
        context=context,
        manager_state=state,
        availability=availability,
        evaluations={cd.CHIP_ACTION_TC: evaluation},
        reservation_forecasts={cd.CHIP_ACTION_TC: artifact},
        reservation_forecast_evidence_verifier=opportunities.__getitem__,
    )
    tc = retained["chip_results"][cd.CHIP_ACTION_TC]
    assert tc["raw_reservation_forecast"]["artifact_sha256"] == artifact["artifact_sha256"]
    metrics = retained["decision"]["candidate_metrics"]
    assert metrics["selected_chip_action"] == cd.CHIP_ACTION_TC
    assert metrics["raw_reservation_forecast_sha256"] == artifact["artifact_sha256"]
    assert metrics["raw_reservation_value"] == 6.0
    receipt = assessment_store.retain_assessment(retained, tmp_path)
    assert assessment_store.verify_assessment(receipt["path"])["verified"] is True
