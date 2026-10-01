"""Focused evidence and retention tests for the chip operational remediation."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from fpl_brain import chip_assessment_store as store
from fpl_brain import chip_assessment as assessment
from fpl_brain import chip_decision as cd
from fpl_brain import chip_reservation_calibration as calibration
from fpl_brain import chip_reservation_forecast as forecast
from fpl_brain import database
from fpl_brain import free_hit_production as fhp
from fpl_brain import outcome_ledger as ol
from fpl_brain.chip_assessment import assemble_assessment_record
from scripts import run_chip_assessment


def _sha(payload):
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False, default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


_RAW_FORECAST_FIXTURES = {}


def _causal_row(
    index: int, *, noise: float = 0.0, action: str = cd.CHIP_ACTION_BB,
    reservation_forecast_state: dict | None = None,
):
    origin = datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(days=7 * index)
    label_time = origin + timedelta(days=2)
    origin_text = origin.isoformat().replace("+00:00", "Z")
    label_text = label_time.isoformat().replace("+00:00", "Z")
    forecast_time = (origin - timedelta(minutes=1)).isoformat().replace("+00:00", "Z")
    event = index + 1
    realized_value = 15.0 + noise
    source = {
        "source_decision_id": f"decision-{index}",
        "source_result_sha256": _sha({"result": index}),
        "source_artifact_sha256": _sha({"artifact": index}),
        "generation_id": f"generation-{index}",
        "planning_event": event,
        "cutoff": origin_text,
        "data_snapshot_sha256": _sha({"snapshot": index}),
        "predictive_code_snapshot_sha256": _sha({"predictive_code": index}),
        "certification_identity": "sha256:" + _sha({"certification": index}),
        "world_identity": "sha256:" + _sha({"world": index}),
    }
    lineup = {
        "starter_ids": [1, 3, 4, 5, 8, 9, 10, 11, 13, 14, 15],
        "bench_gk_id": 2,
        "bench_outfield_order": [6, 12, 7],
        "captain_id": 8,
        "vice_captain_id": 9,
    }
    player_positions = {
        "1": "GKP", "2": "GKP", "3": "DEF", "4": "DEF", "5": "DEF",
        "6": "DEF", "7": "DEF", "8": "MID", "9": "MID", "10": "MID",
        "11": "MID", "12": "MID", "13": "FWD", "14": "FWD", "15": "FWD",
    }
    save_squad = list(range(1, 16))
    play_squad = list(range(1, 16))
    play_lineup = lineup
    play_positions = player_positions
    if action in {cd.CHIP_ACTION_FH, cd.CHIP_ACTION_WC}:
        # FH/WC change the played squad.  Their causal pair is intentionally
        # not forced through the same-squad BB/TC comparison contract.
        play_squad[-1] = 16
        play_lineup = {
            **lineup,
            "starter_ids": [1, 3, 4, 5, 8, 9, 10, 11, 13, 14, 16],
        }
        play_positions = {key: value for key, value in player_positions.items() if key != "15"}
        play_positions["16"] = "FWD"

    def arm_common(squad, arm_lineup, positions, role):
        values = {
            "action": action,
            "event": event + 1,
            "scenario_identity": f"scenario-{index}",
            "world_identity": "sha256:" + _sha({"world": index}),
            "proposed_squad_ids": squad,
            "lineup": arm_lineup,
            "player_positions": positions,
            "counterfactual_role": role,
            "scoring_rule_version": calibration.OUTCOME_SCORING_RULE_VERSION,
            **{key: source[key] for key in (
                "source_decision_id", "source_result_sha256", "source_artifact_sha256",
                "generation_id", "cutoff", "data_snapshot_sha256",
                "predictive_code_snapshot_sha256", "certification_identity",
            )},
        }
        if action in {cd.CHIP_ACTION_FH, cd.CHIP_ACTION_WC}:
            prices = {str(player_id): 50 for player_id in squad}
            values["action_semantics"] = {
                "role": role,
                "transfer_state": {
                    "squad_ids": squad,
                    "bank_tenths": 70,
                    "free_transfers": 3,
                    "purchase_price_tenths": prices,
                },
            }
            if action == cd.CHIP_ACTION_FH:
                values["action_semantics"]["transfer_state"]["event_start_free_transfers"] = 3
            if action == cd.CHIP_ACTION_FH and role == "PLAY":
                values["action_semantics"]["restore_at_h2"] = {
                    "restore_event": event + 2,
                    "permanent_squad_ids": save_squad,
                    "restored_squad_ids": save_squad,
                    "permanent_purchase_price_tenths": {str(pid): 50 for pid in save_squad},
                    "restored_purchase_price_tenths": {str(pid): 50 for pid in save_squad},
                    "permanent_bank_tenths": 70,
                    "restored_bank_tenths": 70,
                    "event_start_h1_free_transfers": 3,
                    "restored_h2_free_transfers": 3,
                }
            elif action == cd.CHIP_ACTION_FH:
                values["action_semantics"]["retains_chip_option"] = True
            elif action == cd.CHIP_ACTION_WC and role == "PLAY":
                values["action_semantics"].update({
                    "wildcard_applied": True,
                    "event_start_free_transfers": 3,
                    "free_transfers_after": 3,
                    "max_free_transfers": 5,
                })
            else:
                values["action_semantics"].update({
                    "wildcard_applied": False,
                    "retains_chip_option": True,
                })
        return values

    play_common = arm_common(play_squad, play_lineup, play_positions, "PLAY")
    save_common = arm_common(save_squad, lineup, player_positions, "SAVE")
    save_reservation_state = (
        dict(save_common["action_semantics"]["transfer_state"])
        if action in {cd.CHIP_ACTION_FH, cd.CHIP_ACTION_WC}
        else {"squad_ids": list(save_squad)}
    )
    forecast_state = dict(reservation_forecast_state or save_reservation_state)
    save_common["reservation_state"] = save_reservation_state
    play_common["reservation_state"] = (
        dict(play_common["action_semantics"]["transfer_state"])
        if action in {cd.CHIP_ACTION_FH, cd.CHIP_ACTION_WC}
        else dict(save_reservation_state)
    )
    starters = set(lineup["starter_ids"])
    save_weights = {str(player_id): 1.0 for player_id in starters}
    save_weights[str(lineup["captain_id"])] += 1.0
    if action == cd.CHIP_ACTION_BB:
        play_weights = {str(player_id): 1.0 for player_id in range(1, 16)}
        play_weights[str(lineup["captain_id"])] += 1.0
    elif action == cd.CHIP_ACTION_TC:
        play_weights = dict(save_weights)
        play_weights[str(lineup["captain_id"])] += 1.0
    else:
        play_weights = {str(player_id): 1.0 for player_id in play_lineup["starter_ids"]}
        play_weights[str(play_lineup["captain_id"])] += 1.0
    play_arm_payload = {
        "arm_id": f"play-{index}", "paired_value": 3.0,
        **play_common,
    }
    save_arm_payload = {
        "arm_id": f"save-{index}", "paired_value": 1.0,
        **save_common,
    }
    forecast_source = {
        key: source[key] for key in (
            "source_decision_id", "source_result_sha256", "source_artifact_sha256",
            "generation_id", "planning_event", "data_snapshot_sha256",
            "predictive_code_snapshot_sha256", "certification_identity",
        )
    }
    forecast_source["origin_cutoff"] = origin_text
    forecast_refs = []
    forecast_records = {}
    for future_event in range(event + 1, event + 4):
        reference = f"forecast-opportunity-{index}-{future_event}.json"
        outcome_arms = None
        if action in {cd.CHIP_ACTION_BB, cd.CHIP_ACTION_TC}:
            outcome_arms = {}
            for role, arm_payload in (("play", play_arm_payload), ("save", save_arm_payload)):
                frozen_arm = dict(arm_payload)
                frozen_arm["event"] = future_event
                frozen_arm["reservation_state"] = forecast_state
                if future_event != event + 1:
                    frozen_arm["scenario_identity"] = f"scenario-{index}-{future_event}"
                    frozen_arm["arm_id"] = f"{role}-{index}-{future_event}"
                outcome_arms[role] = frozen_arm
        opportunity = forecast.build_event_opportunity_record(
            action=action,
            planning_event=event,
            event=future_event,
            origin_cutoff=origin_text,
            made_at=forecast_time,
            expected_incremental_points=10.0 if future_event == event + 1 else 0.0,
            opportunity_model=f"fixture_{action.lower()}_future_opportunity_v1",
            source_identity=forecast_source,
            reservation_state=forecast_state,
            world_identity=source["world_identity"],
            outcome_arms=outcome_arms,
        )
        forecast_refs.append(reference)
        forecast_records[reference] = opportunity
    forecast_artifact = forecast.build_reservation_forecast(
        action=action,
        planning_event=event,
        origin_cutoff=origin_text,
        made_at=forecast_time,
        expiry_event=event + 3,
        source_identity=forecast_source,
        reservation_state=forecast_state,
        opportunity_refs=forecast_refs,
        evidence_verifier=forecast_records.__getitem__,
    )
    forecast_reference = f"forecast-{index}.json"
    _RAW_FORECAST_FIXTURES[forecast_reference] = forecast_artifact
    evidence = {
        "schema": calibration.CAUSAL_EVIDENCE_SCHEMA,
        "policy": calibration.CAUSAL_EVIDENCE_POLICY,
        "observation_id": f"obs-{index}",
        "action": action,
        "planning_event": event,
        "origin_cutoff": origin_text,
        "label_available_at": label_text,
        "source": source,
        "forecast": {
            "artifact_ref": forecast_reference,
            "artifact_sha256": forecast_artifact["artifact_sha256"],
            "retained_content_sha256": _sha(forecast_artifact),
            "value": forecast_artifact["raw_value"],
            "made_at": forecast_artifact["made_at"],
        },
        "counterfactual_pair": {
            "play": {"arm_id": f"play-{index}", "artifact_ref": f"play-{index}.json",
                     "artifact_sha256": _sha(play_arm_payload),
                     "artifact_payload": play_arm_payload, **play_arm_payload},
            "save": {"arm_id": f"save-{index}", "artifact_ref": f"save-{index}.json",
                     "artifact_sha256": _sha(save_arm_payload),
                     "artifact_payload": save_arm_payload, **save_arm_payload},
        },
    }
    captures = []
    for player_id in range(1, 17):
        total_points = (
            realized_value
            if (action == cd.CHIP_ACTION_BB and player_id == 6)
            or (action == cd.CHIP_ACTION_TC and player_id == lineup["captain_id"])
            or (action in {cd.CHIP_ACTION_FH, cd.CHIP_ACTION_WC} and player_id == 16)
            else 0.0
        )
        minutes = 90.0
        capture_payload = {"total_points": total_points, "minutes": minutes}
        capture_digest = ol.capture_digest_for(
            grain=ol.GRAIN_PLAYER_EVENT,
            event=event + 1,
            player_id=player_id,
            fixture_id=None,
            captured_at=label_text,
            observation_state=ol.OBSERVATION_FINAL,
            source_name="player_gameweeks_final",
            source_identity=f"player_gameweeks:{event + 1}",
            source_payload_sha256=None,
            archive_capture_id=None,
            payload=capture_payload,
        )
        captures.append({
            "capture_digest": capture_digest,
            "grain": ol.GRAIN_PLAYER_EVENT,
            "event": event + 1,
            "player_id": player_id,
            "fixture_id": None,
            "observation_state": ol.OBSERVATION_FINAL,
            "official_final_at": label_text,
            "captured_at": label_text,
            "source_name": "player_gameweeks_final",
            "source_identity": f"player_gameweeks:{event + 1}",
            "total_points": total_points,
            "minutes": minutes,
        })
    outcome_record = {
        "schema": calibration.CAUSAL_OUTCOME_RECORD_SCHEMA,
        "record_id": f"outcome-{index}",
        "observation_id": evidence["observation_id"],
        "action": evidence["action"],
        "planning_event": event,
        "origin_cutoff": origin_text,
        "scenario_identity": play_common["scenario_identity"],
        "world_identity": play_common["world_identity"],
        "play_arm_id": f"play-{index}",
        "play_artifact_sha256": _sha(play_arm_payload),
        "save_arm_id": f"save-{index}",
        "save_artifact_sha256": _sha(save_arm_payload),
        "source_decision_id": source["source_decision_id"],
        "source_result_sha256": source["source_result_sha256"],
        "source_artifact_sha256": source["source_artifact_sha256"],
        "generation_id": source["generation_id"],
        "cutoff": source["cutoff"],
        "data_snapshot_sha256": source["data_snapshot_sha256"],
        "predictive_code_snapshot_sha256": source["predictive_code_snapshot_sha256"],
        "certification_identity": source["certification_identity"],
        "label_definition_version": calibration.OUTCOME_LABEL_DEFINITION,
        "scoring_rule_version": calibration.OUTCOME_SCORING_RULE_VERSION,
        "realization_event": event + 1,
        "available_at": label_text,
        "source": {
            "schema": calibration.OUTCOME_CAPTURE_SET_SCHEMA,
            "available_at": label_text,
            "captures": captures,
        },
        "paired_results": {
            "play": {
                "proposed_squad_ids": play_squad,
                "lineup": play_lineup,
                "player_positions": play_positions,
                "scoring_weights": play_weights,
                "observed_points": sum(
                    item["total_points"] * play_weights.get(str(item["player_id"]), 0.0)
                    for item in captures
                ),
            },
            "save": {
                "proposed_squad_ids": save_squad,
                "lineup": lineup,
                "player_positions": player_positions,
                "scoring_weights": save_weights,
                "observed_points": sum(
                    item["total_points"] * save_weights.get(str(item["player_id"]), 0.0)
                    for item in captures
                ),
            },
        },
        "observed_points": realized_value,
    }
    evidence["label"] = {
        "kind": "OBSERVED_FUTURE_OUTCOME",
        "outcome_record_ref": f"outcome-{index}.json",
        "outcome_record": outcome_record,
        "outcome_record_sha256": _sha(outcome_record),
        "realized_reservation_value": realized_value,
        "available_at": label_text,
    }
    row = {
        "observation_id": f"obs-{index}",
        "action": action,
        "planning_event": event,
        "weeks_to_expiry": 3,
        "origin_cutoff": origin_text,
        "forecast_made_at": forecast_time,
        "forecast_value": 10.0,
        "label_available_at": label_text,
        "realized_value": realized_value,
        "causal_evidence_ref": f"evidence-{index}",
        "causal_evidence_sha256": _sha(evidence),
    }
    return row, evidence


def _store_causal_evidence(store_map, row, evidence):
    store_map[row["causal_evidence_ref"]] = evidence
    for arm in evidence["counterfactual_pair"].values():
        store_map[arm["artifact_ref"]] = arm["artifact_payload"]
    forecast_meta = evidence["forecast"]
    forecast_artifact = _RAW_FORECAST_FIXTURES[forecast_meta["artifact_ref"]]
    store_map[forecast_meta["artifact_ref"]] = forecast_artifact
    for opportunity in forecast_artifact["opportunities"]:
        store_map[opportunity["artifact_ref"]] = opportunity["artifact_payload"]
    label = evidence["label"]
    store_map[label["outcome_record_ref"]] = label["outcome_record"]


def _causal_evidence_resolver(evidence, *, retained_outcome=None):
    label = evidence["label"]
    retained = label["outcome_record"] if retained_outcome is None else retained_outcome

    def resolve(reference):
        if str(reference) == str(label["outcome_record_ref"]):
            return retained
        for arm in evidence["counterfactual_pair"].values():
            if str(reference) == str(arm["artifact_ref"]):
                return arm["artifact_payload"]
        forecast_meta = evidence["forecast"]
        if str(reference) == str(forecast_meta["artifact_ref"]):
            return _RAW_FORECAST_FIXTURES[forecast_meta["artifact_ref"]]
        for artifact in _RAW_FORECAST_FIXTURES.values():
            for opportunity in artifact["opportunities"]:
                if str(reference) == str(opportunity["artifact_ref"]):
                    return opportunity["artifact_payload"]
        return evidence

    return resolve


def _official_outcome_connection(event, *, player_ids=range(1, 16), final=True, final_at="2025-01-03T00:00:00Z"):
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    database.initialize_database(conn)
    from datetime import datetime, timedelta

    captured_at = (datetime.fromisoformat(final_at.replace("Z", "+00:00"))
                   + timedelta(minutes=5)).isoformat().replace("+00:00", "Z")
    conn.execute(
        "INSERT INTO events(id, name, finished, data_checked, raw_json, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (int(event), f"GW{event}", 1 if final else 0, 1 if final else 0, "{}", final_at),
    )
    for player_id in player_ids:
        ol.capture_observation(
            conn,
            grain=ol.GRAIN_PLAYER_EVENT,
            event=int(event),
            player_id=int(player_id),
            fields={"minutes": 90, "total_points": 1},
            source_name="player_gameweeks_final",
            official_final_at=final_at,
            observation_state=ol.OBSERVATION_FINAL if final else ol.OBSERVATION_PROVISIONAL,
            source_identity=f"player_gameweeks:{event}",
            captured_at=captured_at,
        )
    return conn


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
        _store_causal_evidence(stored, row, evidence)

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
    runtime_forecast = _RAW_FORECAST_FIXTURES["forecast-4.json"]
    estimate = reservation.estimate(
        action=cd.CHIP_ACTION_BB,
        planning_event=5,
        expiry_event=8,
        state={
            "squad_ids": list(range(1, 16)),
            "raw_reservation_value": 10.0,
            "raw_reservation_forecast": runtime_forecast,
        },
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
        _store_causal_evidence(stored, row, evidence)
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
        calibration.validate_observation(row, evidence_verifier=_causal_evidence_resolver(evidence))


def test_calibration_rejects_mixed_scenario_arms_and_any_tampered_causal_row():
    row, evidence = _causal_row(0)
    evidence["counterfactual_pair"]["save"]["scenario_identity"] = "different-scenario"
    row["causal_evidence_sha256"] = _sha(evidence)
    with pytest.raises(calibration.ReservationCalibrationError, match="selected forecast opportunity"):
        calibration.validate_observation(row, evidence_verifier=_causal_evidence_resolver(evidence))

    stored = {}
    rows = []
    for index in range(55):
        good_row, good_evidence = _causal_row(index)
        rows.append(good_row)
        _store_causal_evidence(stored, good_row, good_evidence)
    bad_row, bad_evidence = _causal_row(56)
    bad_row["causal_evidence_ref"] = "tampered-causal-evidence"
    bad_evidence["label"]["realized_reservation_value"] += 9.0
    _store_causal_evidence(stored, bad_row, bad_evidence)
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


def test_calibration_rejects_label_not_reproduced_by_retained_outcome_record():
    row, evidence = _causal_row(0)
    retained_outcome = dict(evidence["label"]["outcome_record"])
    # A coordinated edit to the row and label must still fail when the
    # content-addressed outcome record says the observed value was 15.
    row["realized_value"] = 999.0
    evidence["label"]["realized_reservation_value"] = 999.0
    row["causal_evidence_sha256"] = _sha(evidence)
    with pytest.raises(calibration.ReservationCalibrationError,
                       match="not reproduced by the retained paired outcome"):
        calibration.validate_observation(
            row,
            evidence_verifier=_causal_evidence_resolver(evidence, retained_outcome=retained_outcome),
        )


def test_rehashed_arbitrary_chip_weights_cannot_manufacture_calibration_labels():
    row, evidence = _causal_row(0)
    outcome = evidence["label"]["outcome_record"]
    result = outcome["paired_results"]["play"]
    forged_weights = dict(result["scoring_weights"])
    forged_weights["6"] = 100.0
    result["scoring_weights"] = forged_weights
    result["observed_points"] = sum(
        capture["total_points"] * forged_weights.get(str(capture["player_id"]), 0.0)
        for capture in outcome["source"]["captures"]
    )
    save_score = outcome["paired_results"]["save"]["observed_points"]
    forged_label = result["observed_points"] - save_score
    outcome["observed_points"] = forged_label
    evidence["label"]["realized_reservation_value"] = forged_label
    evidence["label"]["outcome_record_sha256"] = _sha(outcome)
    row["realized_value"] = forged_label
    row["causal_evidence_sha256"] = _sha(evidence)

    with pytest.raises(calibration.ReservationCalibrationError, match="canonical BB scorer"):
        calibration.validate_observation(row, evidence_verifier=_causal_evidence_resolver(evidence))


@pytest.mark.parametrize("action", [cd.CHIP_ACTION_FH, cd.CHIP_ACTION_WC])
def test_calibration_scores_action_specific_fh_wc_arms_with_distinct_squads(action):
    row, evidence = _causal_row(0, action=action)
    play = evidence["counterfactual_pair"]["play"]
    save = evidence["counterfactual_pair"]["save"]
    assert play["proposed_squad_ids"] != save["proposed_squad_ids"]
    assert play["world_identity"] == save["world_identity"]
    assert calibration.validate_observation(
        row, evidence_verifier=_causal_evidence_resolver(evidence),
    ).realized_value == pytest.approx(row["realized_value"])


def test_fh_reservation_scorer_rejects_broken_permanent_squad_restoration():
    row, evidence = _causal_row(0, action=cd.CHIP_ACTION_FH)
    play = evidence["counterfactual_pair"]["play"]
    play["action_semantics"]["restore_at_h2"]["restored_squad_ids"] = [9000]
    payload = dict(play["artifact_payload"])
    payload["action_semantics"] = play["action_semantics"]
    play["artifact_payload"] = payload
    play["artifact_sha256"] = _sha(payload)
    evidence["label"]["outcome_record"]["play_artifact_sha256"] = play["artifact_sha256"]
    outcome = evidence["label"]["outcome_record"]
    evidence["label"]["outcome_record_sha256"] = _sha(outcome)
    evidence["label"]["outcome_record"] = outcome
    row["causal_evidence_sha256"] = _sha(evidence)
    with pytest.raises(calibration.ReservationCalibrationError, match="does not preserve permanent squad"):
        calibration.validate_observation(row, evidence_verifier=_causal_evidence_resolver(evidence))


def test_fh_reservation_scorer_rejects_incomplete_restoration_evidence():
    row, evidence = _causal_row(0, action=cd.CHIP_ACTION_FH)
    play = evidence["counterfactual_pair"]["play"]
    play["action_semantics"]["restore_at_h2"] = {"restore_event": 3}
    row["causal_evidence_sha256"] = _sha(evidence)

    with pytest.raises(calibration.ReservationCalibrationError, match="restoration record is incomplete"):
        calibration.validate_observation(row, evidence_verifier=_causal_evidence_resolver(evidence))


def test_tc_calibration_rejects_causal_arms_substituted_for_selected_forecast_policies():
    row, evidence = _causal_row(0, action=cd.CHIP_ACTION_TC)
    for arm in evidence["counterfactual_pair"].values():
        lineup = dict(arm["lineup"])
        lineup["captain_id"] = 9
        lineup["vice_captain_id"] = 8
        arm["lineup"] = lineup
        payload = dict(arm["artifact_payload"])
        payload["lineup"] = lineup
        arm["artifact_payload"] = payload
        arm["artifact_sha256"] = _sha(payload)
    row["causal_evidence_sha256"] = _sha(evidence)

    with pytest.raises(
        calibration.ReservationCalibrationError,
        match="does not match the selected forecast opportunity",
    ):
        calibration.validate_observation(row, evidence_verifier=_causal_evidence_resolver(evidence))


def test_tc_finalizer_rejects_arm_policy_changed_after_origin_forecast(tmp_path):
    row, evidence = _causal_row(0, action=cd.CHIP_ACTION_TC)
    retained = {}
    _store_causal_evidence(retained, row, evidence)
    evidence.pop("label")
    for arm in evidence["counterfactual_pair"].values():
        lineup = dict(arm["lineup"])
        lineup["captain_id"] = 9
        lineup["vice_captain_id"] = 8
        arm["lineup"] = lineup

    conn = _official_outcome_connection(2)
    try:
        with pytest.raises(
            calibration.ReservationCalibrationError,
            match="differs from the selected forecast opportunity",
        ):
            calibration.finalize_causal_observation(
                conn,
                evidence,
                realization_event=2,
                evidence_root=tmp_path,
                evidence_verifier=retained.__getitem__,
            )
        assert list(tmp_path.iterdir()) == []
    finally:
        conn.close()


def test_wc_reservation_scorer_rejects_inconsistent_event_start_ft_retention():
    row, evidence = _causal_row(0, action=cd.CHIP_ACTION_WC)
    play = evidence["counterfactual_pair"]["play"]
    play["action_semantics"]["free_transfers_after"] = 1
    payload = dict(play["artifact_payload"])
    payload["action_semantics"] = play["action_semantics"]
    play["artifact_payload"] = payload
    play["artifact_sha256"] = _sha(payload)
    evidence["label"]["outcome_record"]["play_artifact_sha256"] = play["artifact_sha256"]
    outcome = evidence["label"]["outcome_record"]
    evidence["label"]["outcome_record_sha256"] = _sha(outcome)
    evidence["label"]["outcome_record"] = outcome
    row["causal_evidence_sha256"] = _sha(evidence)
    with pytest.raises(calibration.ReservationCalibrationError, match="free-transfer retention"):
        calibration.validate_observation(row, evidence_verifier=_causal_evidence_resolver(evidence))


def test_calibration_rejects_cross_action_scenario_reuse_across_training_rows():
    row, evidence = _causal_row(0)
    outcome = dict(evidence["label"]["outcome_record"])
    outcome["action"] = cd.CHIP_ACTION_TC
    evidence["label"]["outcome_record"] = outcome
    evidence["label"]["outcome_record_sha256"] = _sha(outcome)
    row["causal_evidence_sha256"] = _sha(evidence)
    with pytest.raises(calibration.ReservationCalibrationError, match="does not match its observation"):
        calibration.validate_observation(row, evidence_verifier=_causal_evidence_resolver(evidence))

    tc_row, tc_evidence = _causal_row(2, action=cd.CHIP_ACTION_TC)
    tc_outcome = tc_evidence["label"]["outcome_record"]
    calibration.validate_observation(
        tc_row, evidence_verifier=_causal_evidence_resolver(tc_evidence),
    )

    stored = {}
    rows = []
    for index in range(60, 115):
        bb_row, bb_evidence = _causal_row(index)
        bb_evidence["label"]["outcome_record_ref"] = "shared-tc-outcome.json"
        bb_evidence["label"]["outcome_record"] = tc_outcome
        bb_evidence["label"]["outcome_record_sha256"] = _sha(tc_outcome)
        bb_row["causal_evidence_sha256"] = _sha(bb_evidence)
        rows.append(bb_row)
        _store_causal_evidence(stored, bb_row, bb_evidence)
        stored["shared-tc-outcome.json"] = tc_outcome
    latest_label = datetime.fromisoformat(rows[-1]["label_available_at"].replace("Z", "+00:00"))
    result = calibration.evaluate_reservation_calibration(
        rows,
        evidence_verifier=stored.__getitem__,
        evaluation_cutoff=(latest_label + timedelta(days=1)).isoformat().replace("+00:00", "Z"),
    )
    assert result["status"] == cd.CALIBRATION_UNCALIBRATED
    assert result["models"] == {}
    assert len(result["rejected_observations"]) == len(rows)


def test_calibration_excludes_a_consistent_but_unmatured_outcome():
    row, evidence = _causal_row(1)
    outcome = evidence["label"]["outcome_record"]
    future_time = "2099-01-01T00:00:00Z"
    outcome["available_at"] = future_time
    outcome["source"]["available_at"] = future_time
    for capture in outcome["source"]["captures"]:
        capture["official_final_at"] = future_time
        capture["captured_at"] = future_time
        capture["capture_digest"] = ol.capture_digest_for(
            grain=capture["grain"],
            event=capture["event"],
            player_id=capture["player_id"],
            fixture_id=capture["fixture_id"],
            captured_at=future_time,
            observation_state=capture["observation_state"],
            source_name=capture["source_name"],
            source_identity=capture["source_identity"],
            source_payload_sha256=None,
            archive_capture_id=None,
            payload={"total_points": capture["total_points"], "minutes": capture["minutes"]},
        )
    evidence["label"]["available_at"] = future_time
    evidence["label_available_at"] = future_time
    row["label_available_at"] = future_time
    evidence["label"]["outcome_record_sha256"] = _sha(outcome)
    row["causal_evidence_sha256"] = _sha(evidence)
    calibration.validate_observation(row, evidence_verifier=_causal_evidence_resolver(evidence))
    result = calibration.evaluate_reservation_calibration(
        [row],
        evidence_verifier=_causal_evidence_resolver(evidence),
        evaluation_cutoff="2026-09-30T00:00:00Z",
    )
    assert result["verified_observations"] == 0
    assert result["rejected_observations"] == []
    assert result["models"] == {}
    assert result["status"] == cd.CALIBRATION_UNCALIBRATED


def test_production_outcome_loader_verifies_append_only_official_captures(tmp_path):
    _row, evidence = _causal_row(0)
    outcome = evidence["label"]["outcome_record"]
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        """CREATE TABLE outcome_observation_captures (
           capture_digest TEXT, grain TEXT, event INTEGER, player_id INTEGER,
           fixture_id INTEGER, official_final_at TEXT, captured_at TEXT,
           observation_state TEXT, source_name TEXT, source_identity TEXT,
           source_payload_sha256 TEXT, archive_capture_id TEXT, payload_json TEXT
        )"""
    )
    for capture in outcome["source"]["captures"]:
        payload = {"total_points": capture["total_points"], "minutes": capture["minutes"]}
        conn.execute(
            """INSERT INTO outcome_observation_captures VALUES
               (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                capture["capture_digest"], capture["grain"], capture["event"],
                capture["player_id"], capture["fixture_id"], capture["official_final_at"],
                capture["captured_at"], capture["observation_state"], capture["source_name"],
                capture["source_identity"], None, None, json.dumps(payload),
            ),
        )
    (tmp_path / "outcome.json").write_text(json.dumps(outcome), encoding="utf-8")
    source_positions = evidence["counterfactual_pair"]["play"]["player_positions"]
    loader = run_chip_assessment._calibration_evidence_loader(
        tmp_path, conn, source_position_resolver=lambda _record: source_positions,
    )
    assert loader("outcome.json") == outcome
    incorrect_source_positions = dict(source_positions)
    incorrect_source_positions["6"] = "MID"
    wrong_position_loader = run_chip_assessment._calibration_evidence_loader(
        tmp_path, conn, source_position_resolver=lambda _record: incorrect_source_positions,
    )
    with pytest.raises(calibration.ReservationCalibrationError, match="pinned generation snapshot"):
        wrong_position_loader("outcome.json")
    forged = json.loads(json.dumps(outcome))
    forged["source"]["captures"][0]["total_points"] += 1
    (tmp_path / "forged.json").write_text(json.dumps(forged), encoding="utf-8")
    with pytest.raises(calibration.ReservationCalibrationError, match="manifest differs"):
        loader("forged.json")

    missing_capture = json.loads(json.dumps(outcome))
    missing_capture["source"]["captures"][0]["capture_digest"] = "a" * 64
    (tmp_path / "missing.json").write_text(json.dumps(missing_capture), encoding="utf-8")
    with pytest.raises(calibration.ReservationCalibrationError, match="not retained in the official ledger"):
        loader("missing.json")
    conn.close()


def test_causal_finalizer_derives_and_retains_matured_outcome_from_official_ledger(tmp_path):
    _row, evidence = _causal_row(0, action=cd.CHIP_ACTION_BB)
    retained = {}
    _store_causal_evidence(retained, _row, evidence)
    evidence.pop("label")
    positions = evidence["counterfactual_pair"]["play"]["player_positions"]
    conn = _official_outcome_connection(2)
    try:
        result = calibration.finalize_causal_observation(
            conn,
            evidence,
            realization_event=2,
            evidence_root=tmp_path,
            evidence_verifier=retained.__getitem__,
            source_position_resolver=lambda _record: positions,
        )
        row = result["calibration_row"]
        assert row["realized_value"] == 4.0
        assert row["forecast_value"] == 10.0
        assert result["outcome_record"]["available_at"] == "2025-01-03T00:05:00Z"
        assert result["outcome_record"]["paired_results"]["play"]["scoring_weights"] != {}
        assert result["outcome_record"]["paired_results"]["save"]["scoring_weights"] != {}
        assert Path(result["outcome_path"]).is_file()
        assert Path(result["causal_evidence_path"]).is_file()
        calibration.verify_outcome_record_capture_sources(
            conn,
            result["outcome_record"],
            source_position_resolver=lambda _record: positions,
        )
    finally:
        conn.close()


@pytest.mark.parametrize(
    ("final", "player_ids"),
    [(True, range(1, 15)), (False, range(1, 16))],
    ids=["missing-player", "nonfinal-captures"],
)
def test_causal_finalizer_refuses_missing_or_nonfinal_official_captures(tmp_path, final, player_ids):
    _row, evidence = _causal_row(0, action=cd.CHIP_ACTION_BB)
    retained = {}
    _store_causal_evidence(retained, _row, evidence)
    evidence.pop("label")
    positions = evidence["counterfactual_pair"]["play"]["player_positions"]
    conn = _official_outcome_connection(2, player_ids=player_ids, final=final)
    try:
        with pytest.raises(calibration.ReservationCalibrationError, match="official final player-gameweek capture is missing"):
            calibration.finalize_causal_observation(
                conn,
                evidence,
                realization_event=2,
                evidence_root=tmp_path,
                evidence_verifier=retained.__getitem__,
                source_position_resolver=lambda _record: positions,
            )
        assert list(tmp_path.iterdir()) == []
    finally:
        conn.close()


def test_causal_finalizer_matures_the_origin_selected_event_only(tmp_path):
    _row, evidence = _causal_row(0, action=cd.CHIP_ACTION_BB)
    retained = {}
    _store_causal_evidence(retained, _row, evidence)
    evidence.pop("label")
    positions = evidence["counterfactual_pair"]["play"]["player_positions"]
    conn = _official_outcome_connection(3)
    try:
        with pytest.raises(calibration.ReservationCalibrationError, match="differs from the origin forecast's selected event"):
            calibration.finalize_causal_observation(
                conn,
                evidence,
                realization_event=3,
                evidence_root=tmp_path,
                evidence_verifier=retained.__getitem__,
                source_position_resolver=lambda _record: positions,
            )
        assert list(tmp_path.iterdir()) == []
    finally:
        conn.close()


def test_causal_finalizer_refuses_forecast_bound_to_a_different_save_state(tmp_path):
    _row, evidence = _causal_row(
        0,
        action=cd.CHIP_ACTION_BB,
        reservation_forecast_state={"squad_ids": list(range(1, 16)), "bank_tenths": 999},
    )
    retained = {}
    _store_causal_evidence(retained, _row, evidence)
    evidence.pop("label")
    positions = evidence["counterfactual_pair"]["play"]["player_positions"]
    conn = _official_outcome_connection(2)
    try:
        with pytest.raises(calibration.ReservationCalibrationError, match="not bound to the retained SAVE arm state"):
            calibration.finalize_causal_observation(
                conn,
                evidence,
                realization_event=2,
                evidence_root=tmp_path,
                evidence_verifier=retained.__getitem__,
                source_position_resolver=lambda _record: positions,
            )
        assert list(tmp_path.iterdir()) == []
    finally:
        conn.close()


def test_production_position_resolver_reads_the_bound_generation_snapshot(monkeypatch):
    _row, evidence = _causal_row(0)
    outcome = evidence["label"]["outcome_record"]
    position_map = evidence["counterfactual_pair"]["play"]["player_positions"]
    generation = SimpleNamespace(
        generation_id=outcome["generation_id"],
        planning_event=outcome["planning_event"],
        horizon_kind=run_chip_assessment.generation_store.HORIZON_KIND_FOUR_GW,
        cutoff=outcome["cutoff"],
        snapshot={"sha256": outcome["data_snapshot_sha256"]},
    )
    monkeypatch.setattr(
        run_chip_assessment.generation_store, "load_generation", lambda _conn, _id: generation,
    )
    monkeypatch.setattr(
        run_chip_assessment.generation_store, "verify_generation",
        lambda _conn, _id: {"verified": True},
    )
    monkeypatch.setattr(
        run_chip_assessment.generation_store, "_open_generation_snapshot",
        lambda _generation: SimpleNamespace(close=lambda: None),
    )
    monkeypatch.setattr(
        run_chip_assessment.repositories, "player_candidates",
        lambda _conn: [
            {"id": int(player_id), "position_short_name": position}
            for player_id, position in {**position_map, "16": "DEF"}.items()
        ],
    )
    resolver = run_chip_assessment._pinned_generation_position_resolver(sqlite3.connect(":memory:"))
    assert resolver(outcome) == position_map

    second_lineup_outcome = json.loads(json.dumps(outcome))
    second_lineup_outcome["paired_results"]["play"]["lineup"]["bench_outfield_order"][0] = 16
    second_lineup_outcome["paired_results"]["play"]["proposed_squad_ids"].remove(6)
    second_lineup_outcome["paired_results"]["play"]["proposed_squad_ids"].append(16)
    assert resolver(second_lineup_outcome) == {**position_map, "16": "DEF"}

    mismatched = dict(outcome)
    mismatched["data_snapshot_sha256"] = _sha({"different snapshot": 1})
    with pytest.raises(ValueError, match="event, cutoff or snapshot"):
        resolver(mismatched)


def test_production_position_resolver_covers_both_free_hit_squads(monkeypatch):
    _row, evidence = _causal_row(0, action=cd.CHIP_ACTION_FH)
    outcome = evidence["label"]["outcome_record"]
    play_positions = evidence["counterfactual_pair"]["play"]["player_positions"]
    save_positions = evidence["counterfactual_pair"]["save"]["player_positions"]
    expected_positions = {**save_positions, **play_positions}
    generation = SimpleNamespace(
        generation_id=outcome["generation_id"],
        planning_event=outcome["planning_event"],
        horizon_kind=run_chip_assessment.generation_store.HORIZON_KIND_FOUR_GW,
        cutoff=outcome["cutoff"],
        snapshot={"sha256": outcome["data_snapshot_sha256"]},
    )
    monkeypatch.setattr(
        run_chip_assessment.generation_store, "load_generation", lambda _conn, _id: generation,
    )
    monkeypatch.setattr(
        run_chip_assessment.generation_store, "verify_generation",
        lambda _conn, _id: {"verified": True},
    )
    monkeypatch.setattr(
        run_chip_assessment.generation_store, "_open_generation_snapshot",
        lambda _generation: SimpleNamespace(close=lambda: None),
    )
    monkeypatch.setattr(
        run_chip_assessment.repositories, "player_candidates",
        lambda _conn: [
            {"id": int(player_id), "position_short_name": position}
            for player_id, position in expected_positions.items()
        ],
    )
    resolver = run_chip_assessment._pinned_generation_position_resolver(sqlite3.connect(":memory:"))

    assert set(resolver(outcome)) == set(expected_positions)
    assert "15" in resolver(outcome) and "16" in resolver(outcome)


def test_calibration_does_not_override_an_evaluators_execution_permission():
    stored = {}
    rows = []
    for index in range(55):
        noise = (-2.0, -1.0, 0.0, 1.0, 2.0, 0.0, 1.0, -1.0)[index % 8]
        row, evidence = _causal_row(index, noise=noise)
        rows.append(row)
        _store_causal_evidence(stored, row, evidence)
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
        reservation_forecasts={cd.CHIP_ACTION_BB: _RAW_FORECAST_FIXTURES["forecast-4.json"]},
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
    source_identity = fhp._source_decision_identity(SimpleNamespace(
        source_decision_id=context["source_decision_id"],
        source_result_sha256=context["source_decision_result_sha256"],
        source_artifact_sha256=context["source_decision_artifact_sha256"],
    ))
    assert source_identity == {
        "source_decision_id": context["source_decision_id"],
        "source_result_sha256": context["source_decision_result_sha256"],
        "source_artifact_sha256": context["source_decision_artifact_sha256"],
    }
    shared = {
        **source_identity,
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
        **source_identity,
        "play": play,
        "save": save,
        "restore_at_h2": restoration,
        "arm_identity": arm_identity,
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

    results[cd.CHIP_ACTION_FH]["arm_evidence"]["play"]["start_state"]["free_transfers"] = 3
    restoration = results[cd.CHIP_ACTION_FH]["arm_evidence"]["restore_at_h2"]
    restoration.update({
        "permanent_squad_ids": [9000], "restored_squad_ids": [9000],
        "permanent_purchase_price_tenths": {"9000": 50},
        "restored_purchase_price_tenths": {"9000": 50},
        "permanent_bank_tenths": 999, "restored_bank_tenths": 999,
        "current_h1_free_transfers": 99, "event_start_h1_free_transfers": 99,
        "restored_h2_free_transfers": 99, "restoration_problems": [],
    })
    forged_arm_identity = store.free_hit_arm_identity(results[cd.CHIP_ACTION_FH]["arm_evidence"])
    results[cd.CHIP_ACTION_FH]["arm_evidence"]["arm_identity"] = forged_arm_identity
    results[cd.CHIP_ACTION_FH]["arm_identity"] = forged_arm_identity
    with pytest.raises(store.ChipAssessmentStoreError, match="restoration evidence does not reproduce"):
        store.build_assessment_record(
            context=context,
            manager_state=manager_state,
            chip_results=results,
            decision={"recommended_action": cd.CHIP_ACTION_NO_CHIP,
                      "status": cd.STATUS_CHIP_REVIEW_REQUIRED,
                      "calibration_status": cd.CALIBRATION_UNCALIBRATED},
        )
