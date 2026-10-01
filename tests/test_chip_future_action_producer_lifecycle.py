from __future__ import annotations

import json
import sqlite3
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

TESTS = Path(__file__).resolve().parent
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

import free_hit_certification_fixtures as cf
import test_chip_free_hit as fhfx
import test_chip_wildcard as wcfx
import generation_fixtures as gf
from fpl_brain import chip_decision as cd
from fpl_brain import chip_free_hit as fh
from fpl_brain import chip_reservation_calibration as calibration
from fpl_brain import chip_reservation_forecast as forecast
from fpl_brain import chip_wildcard as wc
from fpl_brain import free_hit_route as free_hit_route
from fpl_brain import free_hit_production as fhp
from fpl_brain import generation_store as gs
from fpl_brain import search_permission as sp
from fpl_brain import outcome_ledger as ol
from fpl_brain import season_rules as sr
from fpl_brain import transfer_state as ts


ORIGIN_EVENT = 6
FUTURE_EVENT = 7
CUTOFF = "2026-09-29T20:27:01Z"
MADE_AT = "2026-09-29T20:27:09Z"
SOURCE = {
    "source_decision_id": "fixture-origin-decision",
    "source_result_sha256": "1" * 64,
    "source_artifact_sha256": "2" * 64,
    "generation_id": "fixture-root-generation",
    "planning_event": ORIGIN_EVENT,
    "origin_cutoff": CUTOFF,
    "data_snapshot_sha256": cf.DATA_SNAPSHOT,
    "predictive_code_snapshot_sha256": cf.CODE_SNAPSHOT,
    "certification_identity": "3" * 64,
}


def _coverage_product(
    action: str, horizon_length: int, *, expiry_event: int = FUTURE_EVENT,
) -> dict:
    expiry = int(expiry_event)
    product_events = list(range(ORIGIN_EVENT, expiry + horizon_length))
    body = {
        "schema": forecast.RESERVATION_COVERAGE_PRODUCT_SCHEMA,
        "action": action,
        "planning_event": ORIGIN_EVENT,
        "expiry_event": expiry,
        "forecast_events": list(range(ORIGIN_EVENT + 1, expiry + 1)),
        "opportunity_horizon_length": horizon_length,
        "required_product_events": product_events,
        "product_events": product_events,
        "missing_events": [],
        "coverage_known": True,
        "coverage_complete": True,
        "coverage_status": forecast.FORECAST_READY,
        "coverage_reason": None,
        "source_identity": SOURCE,
        "root_generation_id": SOURCE["generation_id"],
        "product_generation_id": "fixture-origin-pinned-continuation",
        "product_generation_manifest_sha256": "fixture-origin-pinned-continuation",
        "origin_cutoff": CUTOFF,
        "input_as_of": CUTOFF,
        "data_snapshot_sha256": SOURCE["data_snapshot_sha256"],
        "predictive_code_snapshot_sha256": SOURCE["predictive_code_snapshot_sha256"],
        "product_runs_by_event": {
            str(event): cf.runs_for(event) for event in product_events
        },
        "wildcard_value_horizon_length": horizon_length if action == cd.CHIP_ACTION_WC else None,
    }
    body["product_sha256"] = forecast.canonical_sha256(body)
    return body


def _free_hit_request(
    event: int = FUTURE_EVENT, *, current_ft: int = 3, event_start_ft: int = 3,
):
    event = int(event)
    events = tuple(range(event, event + cd.CHIP_HORIZON_LENGTH))
    certificate = cf.certification_artifact(
        events=events,
        cutoff=CUTOFF,
        snapshot=cf.DATA_SNAPSHOT,
        code=cf.CODE_SNAPSHOT,
    )
    authority = fh.FreeHitDecisionAuthority.from_certification(
        certificate, loaded_from="<fixture future continuation certification>"
    )
    identity = cf.identity_for(certificate["certified_bundles"][str(event)])
    owned = fhfx.LEGAL_OWNED
    permanent = fh.FreeHitPermanentState(
        event=event,
        owned_ids=owned,
        purchase_price_tenths={pid: 50 for pid in owned},
        bank_tenths=70,
        free_transfers=current_ft,
        event_start_free_transfers=event_start_ft,
        positions=fhfx.POSITION,
        clubs=fhfx.CLUB,
    )
    core = {
        pid: tuple((5.0 + 8.0 * (event - FUTURE_EVENT)) if pid not in owned else 1.0
                   for _ in range(fhfx.WORLDS))
        for pid in fhfx.UNIVERSE
    }
    minutes = {pid: tuple(90.0 for _ in range(fhfx.WORLDS)) for pid in fhfx.UNIVERSE}
    worlds = fhfx._worlds(
        core=core, minutes=minutes, event=event, identity=identity,
    )
    binding = cd.ChipHorizonBinding(
        planning_event=event,
        horizon_events=events,
        certification_identity=authority.certification_identity,
        data_snapshot_sha256=cf.DATA_SNAPSHOT,
    )
    play_start_ft = fh.post_free_hit_ft_state(
        fhfx.RULES, event_start_free_transfers=permanent.event_start_free_transfers,
    )
    play = fhfx._canonical_route(
        arm=free_hit_route.ARM_PLAY,
        events=events[1:],
        permanent=permanent,
        bank_tenths=permanent.bank_tenths,
        free_transfers=play_start_ft,
        values=(10.0, 10.0, 10.0),
    )
    save = fhfx._canonical_route(
        arm=free_hit_route.ARM_SAVE,
        events=events,
        permanent=permanent,
        bank_tenths=permanent.bank_tenths,
        free_transfers=permanent.free_transfers,
        values=(12.0, 10.0, 10.0, 10.0),
    )
    request = fh.FreeHitRequest(
        permanent=permanent,
        horizon_binding=binding,
        h1_worlds=worlds,
        world_identity=identity,
        positions=fhfx.POSITION,
        clubs=fhfx.CLUB,
        market_price_tenths=fhfx._market(),
        pool_binding=fhfx._pool(),
        play_route=play,
        save_route=save,
        decision_authority=authority,
        chip_available=True,
        rules=fhfx.RULES,
    )
    assert fh.contract_problems(request) == []
    save_state = {
        "squad_ids": list(owned),
        "bank_tenths": permanent.bank_tenths,
        "purchase_price_tenths": {str(pid): 50 for pid in owned},
        "free_transfers": permanent.free_transfers,
        "event_start_free_transfers": permanent.event_start_free_transfers,
    }
    return request, save_state


def _wildcard_request(
    monkeypatch, future_event: int = FUTURE_EVENT, *, expiry_event: int | None = None,
):
    future_event = int(future_event)
    events = tuple(range(future_event, future_event + 6))
    owned = tuple(range(500, 515))
    positions = {
        **{pid: "GKP" for pid in (500, 501)},
        **{pid: "DEF" for pid in range(502, 507)},
        **{pid: "MID" for pid in range(507, 512)},
        **{pid: "FWD" for pid in range(512, 515)},
        515: "DEF", 516: "MID", 517: "MID", 518: "MID", 519: "FWD",
    }
    pool_ids = tuple(sorted(positions))
    source_generation = f"fixture-wildcard-value-generation-{future_event}"
    model_config = "4" * 64
    identity = wc.WildcardPredictiveIdentity(
        cutoff=CUTOFF,
        data_snapshot_sha256=cf.DATA_SNAPSHOT,
        source_snapshot_sha256=cf.CODE_SNAPSHOT,
        generation=source_generation,
        model_config_identity=model_config,
    )
    horizon = wc.wildcard_horizon(future_event, length=6, last_event=16)
    value_binding = wc.value_horizon_binding(
        future_event,
        horizon,
        decision_cutoff=CUTOFF,
        data_snapshot_sha256=cf.DATA_SNAPSHOT,
        source_snapshot_sha256=cf.CODE_SNAPSHOT,
        prediction_generation=source_generation,
        model_config_identity=model_config,
    )
    certification_identity = f"fixture-future-chip-certification-{future_event}"
    chip_binding = cd.ChipHorizonBinding(
        planning_event=future_event,
        horizon_events=tuple(range(future_event, future_event + 4)),
        certification_identity=certification_identity,
        data_snapshot_sha256=cf.DATA_SNAPSHOT,
    )
    players = {
        pid: wc.WildcardPlayer(
            player_id=pid,
            position=positions[pid],
            club_id=(20 + (pid - 500) % 10) if pid in owned else 30 + (pid - 515),
            market_price_tenths=50,
            events={
                event: wc.WildcardPlayerEvent(
                    event=event,
                    expected_points=(10.0 if pid not in owned and event >= FUTURE_EVENT + 1 else 3.0),
                    expected_minutes=90.0,
                    p_start=1.0,
                    availability=1.0,
                    fixture_count=1,
                    identity=identity,
                )
                for event in events
            },
            web_name=f"P{pid}",
        )
        for pid in pool_ids
    }
    worlds = {
        event: wc.WildcardWorldInputs(
            event=event,
            worlds=1,
            player_ids=pool_ids,
            minutes={pid: (90.0,) for pid in pool_ids},
            core={
                pid: ((10.0 if pid not in owned and event >= FUTURE_EVENT + 1 else 3.0),)
                for pid in pool_ids
            },
            identity=identity,
        )
        for event in events
    }
    policy = {
        "starter_ids": [500, 502, 503, 504, 507, 508, 509, 510, 512, 513, 514],
        "bench_gk_id": 501,
        "bench_outfield_order": [505, 506, 511],
        "captain_id": 512,
        "vice_captain_id": 510,
    }
    save_events = tuple(
        wc.WildcardSaveRouteEvent(
            event=event,
            squad_ids=owned,
            bank_tenths=70,
            purchase_price_tenths={pid: 50 for pid in owned},
            free_transfers=3,
            mean_net_core=36.0,
            hit_points=0,
            policy=policy,
        )
        for event in events[:4]
    )
    save_route = wc.WildcardSaveRoute(
        events=save_events,
        terminal_squad_ids=owned,
        terminal_bank_tenths=70,
        terminal_purchase_price_tenths={pid: 50 for pid in owned},
        terminal_free_transfers=3,
        cumulative_hits=0,
        wildcard_available=True,
    )
    element_digest = __import__("fpl_brain.ingest_provenance", fromlist=["element_id_sha256"]).element_id_sha256(
        sorted(pool_ids)
    )
    pool = wc.WildcardPoolBinding(
        generation_identity="fixture-accepted-player-pool",
        generation_id_sha256=element_digest,
        official_count=len(pool_ids),
        eligible_ids=pool_ids,
    )
    request = wc.WildcardRequest(
        planning_event=future_event,
        horizon=horizon,
        players=players,
        positions=positions,
        owned_ids=owned,
        purchase_price_tenths={pid: 50 for pid in owned},
        selling_price_tenths={pid: 50 for pid in owned},
        bank_tenths=70,
        rules=sr.SeasonRules(season="2026/27"),
        horizon_binding=chip_binding,
        certification_identity=certification_identity,
        data_snapshot_sha256=cf.DATA_SNAPSHOT,
        event_start_free_transfers=3,
        save_route=save_route,
        chip_availability=wcfx._chip_rows(future_event),
        value_horizon_binding=value_binding,
        worlds_by_event=worlds,
        pool_binding=pool,
    )

    coverage_expiry = int(expiry_event if expiry_event is not None else future_event + 1)
    product_runs = {
        covered: cf.runs_for(covered)
        for covered in range(ORIGIN_EVENT, coverage_expiry + 6)
    }
    chip_events = tuple(range(future_event, future_event + 4))
    chip_generation_id = "fixture-origin-pinned-continuation"

    def generation(generation_id, horizon_kind, gen_events, *, planning_event=future_event):
        return SimpleNamespace(
            generation_id=generation_id,
            horizon_kind=horizon_kind,
            planning_event=planning_event,
            events=gen_events,
            cutoff=CUTOFF,
            snapshot={"sha256": cf.DATA_SNAPSHOT},
            manifest={
                "code_snapshot_sha256": cf.CODE_SNAPSHOT,
                "per_event": {str(item): {"bundle_identity": f"bundle-{item}"} for item in gen_events},
            },
            runs_for=lambda item: product_runs[int(item)],
        )

    chip_generation = generation(
        chip_generation_id,
        gs.HORIZON_KIND_CHIP_RESERVATION,
        tuple(product_runs),
        planning_event=ORIGIN_EVENT,
    )
    value_generation = generation(
        source_generation, gs.HORIZON_KIND_WILDCARD_VALUE, events,
    )
    by_id = {chip_generation.generation_id: chip_generation, value_generation.generation_id: value_generation}
    monkeypatch.setattr(gs, "load_generation", lambda _conn, identity: by_id[str(identity)])
    monkeypatch.setattr(gs, "verify_generation", lambda *_args, **_kwargs: {"verified": True})
    monkeypatch.setattr(gs, "_certification_identity_for_bundles", lambda **_kwargs: certification_identity)
    save_state = {
        "squad_ids": list(owned),
        "bank_tenths": 70,
        "purchase_price_tenths": {str(pid): 50 for pid in owned},
        "free_transfers": 3,
        "event_start_free_transfers": 3,
        "wildcard_available": True,
    }
    return request, save_state, chip_generation_id, source_generation


def _insert_final_captures(conn, outcome_arms, positions):
    conn.execute(
        """CREATE TABLE outcome_observation_captures (
           capture_digest TEXT, grain TEXT, event INTEGER, player_id INTEGER,
           fixture_id INTEGER, official_final_at TEXT, captured_at TEXT,
           observation_state TEXT, source_name TEXT, source_identity TEXT,
           source_payload_sha256 TEXT, archive_capture_id TEXT, payload_json TEXT
        )"""
    )
    by_event: dict[int, set[int]] = {}
    for arm in outcome_arms.values():
        for row in arm["valuation_schedule"]["events"]:
            by_event.setdefault(int(row["event"]), set()).update(
                int(pid) for pid in row["proposed_squad_ids"]
            )
    for event, player_ids in sorted(by_event.items()):
        final_at = f"2026-10-{event + 10:02d}T12:00:00Z"
        for player_id in sorted(player_ids):
            payload = {"total_points": 4.0 if player_id % 2 else 2.0, "minutes": 90.0}
            digest = ol.capture_digest_for(
                grain=ol.GRAIN_PLAYER_EVENT,
                event=event,
                player_id=player_id,
                fixture_id=None,
                captured_at=final_at,
                observation_state=ol.OBSERVATION_FINAL,
                source_name="player_gameweeks_final",
                source_identity=f"player_gameweeks:{event}",
                source_payload_sha256=None,
                archive_capture_id=None,
                payload=payload,
            )
            conn.execute(
                "INSERT INTO outcome_observation_captures VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    digest, ol.GRAIN_PLAYER_EVENT, event, player_id, None,
                    final_at, final_at, ol.OBSERVATION_FINAL,
                    "player_gameweeks_final", f"player_gameweeks:{event}", None, None,
                    json.dumps(payload),
                ),
            )
    return {str(pid): position for pid, position in positions.items()}


@pytest.mark.parametrize("action", [cd.CHIP_ACTION_FH, cd.CHIP_ACTION_WC])
def test_future_action_producer_retention_maturation_and_calibration_path(
    action, monkeypatch, tmp_path,
):
    """Fixture inputs exercise each canonical evaluator through the full causal lifecycle."""

    # This lifecycle test uses deliberately synthetic source identities and focuses
    # on producer → retention → maturation → calibration. Permission refusal/admission
    # against a real persisted generation is covered by the dedicated gate tests.
    monkeypatch.setattr(
        sp,
        "require_search_permission",
        lambda _conn, _generation_id, **_kwargs: gf.fixture_search_permission_evaluation(SOURCE),
    )

    expiry = FUTURE_EVENT + 1
    product = _coverage_product(
        action, 4 if action == cd.CHIP_ACTION_FH else 6, expiry_event=expiry,
    )
    monkeypatch.setattr(
        forecast, "verify_reservation_coverage_product",
        lambda _conn, _product, *, expected: {"verified": True, "product_sha256": _product["product_sha256"]},
    )
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    try:
        evidence_root = tmp_path / action.lower()
        retained: dict[str, dict] = {}
        opportunities: dict[int, dict] = {}
        reservation_state = None
        positions = None
        for event in (FUTURE_EVENT, expiry):
            if action == cd.CHIP_ACTION_FH:
                request, save_state = _free_hit_request(event, current_ft=2, event_start_ft=3)
                verified_authority = request.decision_authority
                monkeypatch.setattr(
                    fh.FreeHitDecisionAuthority,
                    "from_verified_continuation_generation",
                    classmethod(
                        lambda cls, _conn, _generation_id, *, events: verified_authority
                    ),
                )
                positions = fhfx.POSITION
                opportunity = forecast.build_evaluated_event_opportunity_record(
                    action=action,
                    event=event,
                    action_request=request,
                    source_identity=SOURCE,
                    reservation_state=save_state,
                    made_at=MADE_AT,
                    conn=conn,
                    expiry_event=expiry,
                    coverage_product=product,
                )
            else:
                request, save_state, chip_generation_id, value_generation_id = _wildcard_request(
                    monkeypatch, event, expiry_event=expiry,
                )
                positions = request.positions
                opportunity = forecast.build_evaluated_event_opportunity_record(
                    action=action,
                    event=event,
                    action_request=request,
                    source_identity=SOURCE,
                    reservation_state=save_state,
                    made_at=MADE_AT,
                    conn=conn,
                    expiry_event=expiry,
                    coverage_product=product,
                    chip_generation_id=chip_generation_id,
                    value_generation_id=value_generation_id,
                )
            assert opportunity["made_at"] == MADE_AT
            assert opportunity["input_as_of"] == CUTOFF
            assert opportunity["coverage_product_sha256"] == product["product_sha256"]
            assert opportunity["action"] == action
            assert opportunity["event"] == event
            assert opportunity["source_identity"] == SOURCE
            if action == cd.CHIP_ACTION_FH:
                evaluator_identity = opportunity["evaluator_identity"]
                assert evaluator_identity["opportunity_value_definition"] == "FOUR_EVENT_NORMALIZED_MEAN_POINTS"
                assert evaluator_identity["uncertainty_value_definition"] == "FOUR_EVENT_NORMALIZED_MEAN_POINTS"
                assert evaluator_identity["expected_incremental_points"] == pytest.approx(
                    evaluator_identity["mean_four_event_uplift"] / cd.CHIP_HORIZON_LENGTH,
                )
                raw_uncertainty = evaluator_identity["four_event_uncertainty"]
                scaled_uncertainty = evaluator_identity["uncertainty"]
                for field in (
                    "paired_interval_low", "paired_interval_high",
                    "paired_quantile_05", "paired_quantile_50", "paired_quantile_95",
                ):
                    assert scaled_uncertainty[field] == pytest.approx(
                        raw_uncertainty[field] / cd.CHIP_HORIZON_LENGTH,
                    )
            elif action == cd.CHIP_ACTION_WC:
                assert opportunity["evaluator_identity"]["opportunity_value_definition"] == (
                    "WEIGHTED_WC_HORIZON_MEAN_POINTS"
                )
            assert set(opportunity["outcome_arms"]) == {"play", "save"}
            play_arm = opportunity["outcome_arms"]["play"]
            save_arm = opportunity["outcome_arms"]["save"]
            assert opportunity["world_identity"]
            assert play_arm["action"] == save_arm["action"] == action
            assert play_arm["event"] == save_arm["event"] == event
            assert play_arm["counterfactual_role"] == "PLAY"
            assert save_arm["counterfactual_role"] == "SAVE"
            assert play_arm["scenario_identity"] == save_arm["scenario_identity"]
            assert play_arm["world_identity"] == save_arm["world_identity"] == opportunity["world_identity"]
            assert play_arm["coverage_product_sha256"] == save_arm["coverage_product_sha256"] == product["product_sha256"]
            for role, arm in (("play", play_arm), ("save", save_arm)):
                assert arm["source_decision_id"] == SOURCE["source_decision_id"]
                assert arm["source_result_sha256"] == SOURCE["source_result_sha256"]
                assert arm["source_artifact_sha256"] == SOURCE["source_artifact_sha256"]
                assert arm["generation_id"] == SOURCE["generation_id"]
                assert arm["cutoff"] == SOURCE["origin_cutoff"]
                assert arm["data_snapshot_sha256"] == SOURCE["data_snapshot_sha256"]
                assert arm["predictive_code_snapshot_sha256"] == SOURCE["predictive_code_snapshot_sha256"]
                assert arm["certification_identity"] == SOURCE["certification_identity"]
                schedule = arm["valuation_schedule"]["events"]
                assert schedule[0]["event"] == event
                assert arm["lineup"] == schedule[0]["lineup"]
                assert arm["action_semantics"]["role"] == role.upper()
                semantic_state = arm["action_semantics"]["transfer_state"]
                schedule_state = schedule[0]["transfer_state"]
                assert all(schedule_state[key] == value for key, value in semantic_state.items() if key in schedule_state)
                assert all(key in semantic_state for key in schedule_state if key != "event")
                if action == cd.CHIP_ACTION_FH and role == "play":
                    assert semantic_state["event_start_free_transfers"] == save_state["event_start_free_transfers"]
            if action == cd.CHIP_ACTION_FH:
                assert len(play_arm["valuation_schedule"]["events"]) == cd.CHIP_HORIZON_LENGTH
                assert len(save_arm["valuation_schedule"]["events"]) == cd.CHIP_HORIZON_LENGTH
                restoration = play_arm["action_semantics"]["restore_at_h2"]
                assert save_state["free_transfers"] == 2
                assert save_state["event_start_free_transfers"] == 3
                origin_manager_state = play_arm["action_semantics"]["origin_manager_state"]
                assert origin_manager_state["free_transfers"] == 2
                assert origin_manager_state["event_start_free_transfers"] == 3
                assert restoration["event_start_h1_free_transfers"] == 3
                assert restoration["restored_h2_free_transfers"] == 3
                assert restoration["restore_event"] == event + 1
                assert restoration["permanent_squad_ids"] == restoration["restored_squad_ids"]
                assert restoration["permanent_squad_ids"] == save_state["squad_ids"]
                assert restoration["permanent_purchase_price_tenths"] == restoration["restored_purchase_price_tenths"]
                assert restoration["permanent_bank_tenths"] == restoration["restored_bank_tenths"]
            else:
                assert len(play_arm["valuation_schedule"]["events"]) == len(request.horizon.events)
                assert len(save_arm["valuation_schedule"]["events"]) == len(request.horizon.events)
                assert play_arm["action_semantics"]["wildcard_applied"] is True
                assert save_arm["action_semantics"]["wildcard_applied"] is False
                assert save_arm["action_semantics"]["retains_chip_option"] is True
            receipt = forecast.retain_event_opportunity_record(opportunity, evidence_root)
            reference = Path(receipt["path"]).name
            retained[reference] = opportunity
            opportunities[event] = opportunity
            reservation_state = save_state

        forecast_artifact = forecast.build_reservation_forecast(
            action=action,
            planning_event=ORIGIN_EVENT,
            origin_cutoff=CUTOFF,
            input_as_of=CUTOFF,
            made_at=MADE_AT,
            expiry_event=expiry,
            source_identity=SOURCE,
            reservation_state=reservation_state,
            opportunity_refs=tuple(retained),
            evidence_verifier=retained.__getitem__,
            coverage_product=product,
        )
        assert forecast_artifact["covered_events"] == [FUTURE_EVENT, expiry]
        assert forecast_artifact["selected_event"] == expiry
        origin = calibration.retain_causal_origin_observation(
            observation_id=f"fixture-{action.lower()}-origin",
            forecast_artifact=forecast_artifact,
            evidence_root=evidence_root,
            evidence_verifier=retained.__getitem__,
        )

        def evidence_verifier(reference):
            if reference in retained:
                return retained[reference]
            return json.loads((evidence_root / reference).read_text(encoding="utf-8"))

        outcome_arms = opportunities[expiry]["outcome_arms"]
        source_positions = _insert_final_captures(conn, outcome_arms, positions)
        with pytest.raises(
            calibration.ReservationCalibrationError,
            match="differs from the origin forecast's selected event",
        ):
            calibration.finalize_causal_observation(
                conn,
                origin["causal_evidence"],
                realization_event=FUTURE_EVENT,
                evidence_root=evidence_root,
                evidence_verifier=evidence_verifier,
                source_position_resolver=lambda _record: source_positions,
            )
        matured = calibration.finalize_causal_observation(
            conn,
            origin["causal_evidence"],
            realization_event=expiry,
            evidence_root=evidence_root,
            evidence_verifier=evidence_verifier,
            source_position_resolver=lambda _record: source_positions,
        )
        row = matured["calibration_row"]
        verified = calibration.validate_observation(row, evidence_verifier=evidence_verifier)
        assert verified.action == action
        result = calibration.evaluate_reservation_calibration(
            [row],
            evidence_verifier=evidence_verifier,
            evaluation_cutoff="2026-12-31T00:00:00Z",
        )
        assert result["status"] == cd.CALIBRATION_UNCALIBRATED
        assert result["models"] == {}
    finally:
        conn.close()


def test_future_free_hit_producer_refuses_altered_continuation_bundle_context(monkeypatch):
    monkeypatch.setattr(
        sp,
        "require_search_permission",
        lambda _conn, _generation_id, **_kwargs: gf.fixture_search_permission_evaluation(SOURCE),
    )
    request, save_state = _free_hit_request()
    product = _coverage_product(
        cd.CHIP_ACTION_FH, 4, expiry_event=FUTURE_EVENT,
    )
    altered_bundles = {
        int(event): dict(bundle)
        for event, bundle in request.decision_authority.bundle_map.items()
    }
    altered_bundles[FUTURE_EVENT + 1]["planning_context_hash"] = "sha256:" + "f" * 64
    altered_authority = replace(
        request.decision_authority,
        bundle_map=altered_bundles,
    )
    # This context field is outside the H1 world's direct identity projection;
    # the future producer must still compare every selected bundle with the
    # independently verified continuation authority.
    bad_request = replace(request, decision_authority=altered_authority)
    verified_authority = request.decision_authority
    monkeypatch.setattr(
        forecast,
        "verify_reservation_coverage_product",
        lambda _conn, _product, *, expected: {"verified": True},
    )
    monkeypatch.setattr(
        fh.FreeHitDecisionAuthority,
        "from_verified_continuation_generation",
        classmethod(
            lambda cls, _conn, _generation_id, *, events: verified_authority
        ),
    )
    conn = sqlite3.connect(":memory:")
    try:
        with pytest.raises(fhp.FreeHitProductionError, match="authority differs"):
            fhp.build_future_free_hit_event_opportunity(
                conn,
                request=bad_request,
                source_identity=SOURCE,
                event=FUTURE_EVENT,
                expiry_event=FUTURE_EVENT,
                coverage_product=product,
                reservation_state=save_state,
                made_at=MADE_AT,
            )
    finally:
        conn.close()
