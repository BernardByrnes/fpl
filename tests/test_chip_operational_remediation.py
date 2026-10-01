"""Regression coverage for the CHIP operational remediation boundaries."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from fpl_brain import candidate_universe as cu, execution, generation_store as gs
from fpl_brain.database import connect_database

import generation_fixtures as gf
import test_pe9_certification_integration as pe9
import free_hit_route_fixtures as fhfx

from fpl_brain import chip_assessment_store as chip_store
from fpl_brain import chip_decision as cd, chip_free_hit as fh, chip_route_assembly as cra, free_hit_production as fhp
from fpl_brain import chip_wildcard as cw, wildcard_request_adapter as wa
from fpl_brain import free_hit_request_adapter as fha, manager_lineup as ml
from fpl_brain import route_comparator as rc, route_optimizer as ro, transfer_state as ts
from fpl_brain import season_rules as sr
from fpl_brain import wildcard_production as wcp


def _certify_with_lease(conn, **kwargs):
    controller = execution.ExecutionController(conn)
    controller.create_run(
        planning_event=int(kwargs["planning_event"]),
        planning_cutoff=str(kwargs["cutoff"]),
        hard_stop_at=datetime.now(timezone.utc) + timedelta(hours=1),
        label="chip_remediation_test_certification",
    )
    controller.start()
    controller.acquire_writer_lease()
    try:
        result = gs.certify_generation(conn, controller=controller, **kwargs)
    except BaseException as failure:
        controller.finish(execution.RUN_FAILED, str(failure))
        raise
    controller.finish(execution.RUN_COMPLETE)
    return result


@pytest.mark.parametrize("length", (6, 10))
def test_wildcard_value_product_certifies_6_to_10_events_without_widening_normal_contract(length):
    events = tuple(range(1, 11))
    conn, runs = pe9._world(events=events)
    try:
        snapshot = gf.fixture_snapshot(conn)
        four_gw = _certify_with_lease(
            conn,
            planning_event=1,
            cutoff=pe9.CUTOFF,
            runs_by_event={event: runs[event] for event in range(1, 5)},
            snapshot=snapshot,
            horizon_kind=gs.HORIZON_KIND_FOUR_GW,
            events=tuple(range(1, 5)),
        )
        assert four_gw.horizon_kind == gs.HORIZON_KIND_FOUR_GW
        assert four_gw.events == (1, 2, 3, 4)

        controller = execution.ExecutionController(conn)
        controller.create_run(
            planning_event=1,
            planning_cutoff=pe9.CUTOFF,
            hard_stop_at=datetime.now(timezone.utc) + timedelta(hours=1),
            label="wildcard_value_generation_test",
        )
        controller.start()
        controller.acquire_writer_lease()
        try:
            wildcard = gs.certify_wildcard_value_generation(
                conn,
                chip_generation_id=four_gw.generation_id,
                planning_event=1,
                cutoff=pe9.CUTOFF,
                events=events[:length],
                runs_by_event={event: runs[event] for event in events[:length]},
                snapshot=snapshot,
                controller=controller,
            )
        except BaseException as failure:
            controller.finish(execution.RUN_FAILED, str(failure))
            raise
        controller.finish(execution.RUN_COMPLETE)

        report = gs.verify_generation(conn, wildcard.generation_id)
        assert report["verified"] is True
        assert wildcard.horizon_kind == gs.HORIZON_KIND_WILDCARD_VALUE
        assert wildcard.events == events[:length]
        assert wildcard.cutoff == four_gw.cutoff
        assert wildcard.snapshot["sha256"] == four_gw.snapshot["sha256"]
        assert wildcard.manifest["code_snapshot_sha256"] == four_gw.manifest["code_snapshot_sha256"]
    finally:
        conn.close()


@pytest.mark.parametrize(
    "events,length,fragment",
    [
        ((1, 2, 3, 4, 5), 5, "needs 6-10 events"),
        ((1, 2, 3, 4, 5, 7), 6, "not contiguous"),
        ((1, 2, 3, 4, 5, 6), 7, "does not match its 6 declared events"),
    ],
)
def test_wildcard_value_product_refuses_short_holes_and_length_disagreement(events, length, fragment):
    reasons = gs._wildcard_value_horizon_problems(
        planning_event=1, events=events, horizon_length=length, last_event=38
    )
    assert any(fragment in reason for reason in reasons)


def test_wildcard_value_product_refuses_horizon_past_pinned_season_end():
    reasons = gs._wildcard_value_horizon_problems(
        planning_event=1, events=tuple(range(1, 7)), horizon_length=6, last_event=5
    )
    assert any("beyond pinned season end GW5" in reason for reason in reasons)


def test_wildcard_value_product_refuses_newly_generated_prefix_runs():
    """A later run cannot be stitched onto the retained four-event decision."""

    events = tuple(range(1, 11))
    conn, runs = pe9._world(events=events)
    try:
        snapshot = gf.fixture_snapshot(conn)
        four_gw = _certify_with_lease(
            conn,
            planning_event=1,
            cutoff=pe9.CUTOFF,
            runs_by_event={event: runs[event] for event in range(1, 5)},
            snapshot=snapshot,
            horizon_kind=gs.HORIZON_KIND_FOUR_GW,
            events=tuple(range(1, 5)),
        )
        stitched = {event: dict(runs[event]) for event in events[:6]}
        stitched[1]["minutes_v1"] += 100_000
        controller = execution.ExecutionController(conn)
        controller.create_run(
            planning_event=1,
            planning_cutoff=pe9.CUTOFF,
            hard_stop_at=datetime.now(timezone.utc) + timedelta(hours=1),
            label="wildcard_prefix_stitching_refusal_test",
        )
        controller.start()
        controller.acquire_writer_lease()
        try:
            with pytest.raises(gs.GenerationRefused, match="exact normal-generation run ids"):
                gs.certify_wildcard_value_generation(
                    conn,
                    chip_generation_id=four_gw.generation_id,
                    planning_event=1,
                    cutoff=pe9.CUTOFF,
                    events=tuple(range(1, 7)),
                    runs_by_event=stitched,
                    snapshot=snapshot,
                    controller=controller,
                )
        finally:
            controller.finish(execution.RUN_COMPLETE)
    finally:
        conn.close()


def test_verified_route_replay_rebuilds_canonical_state_and_refuses_mutation():
    events = (5, 6, 7, 8)
    start = fhfx.start_state(event=5, bank_tenths=20, free_transfers=2)
    partial, terminal = fhfx.build_canonical_route(start=start, events=events)
    actions = ro._serialize_route(partial)
    snapshot = ts.PriceSnapshot(event=5, prices={pid: 50 for pid in fhfx.UNIVERSE})
    scenario = rc.flat_current_price_scenario(snapshot, events)
    terminal_payload = {
        "squad_ids": list(sorted(pid for pid in terminal.by_id())),
        "bank_tenths": terminal.bank_tenths,
        "free_transfers": terminal.free_transfers,
    }

    replayed = cra.replay_serialized_route(
        actions,
        start_state=start,
        expected_events=events,
        price_scenario=scenario,
        player_meta=fhfx.player_meta(),
        recorded_terminal=terminal_payload,
    )
    assert [row["event"] for row in replayed.actions] == list(events)
    assert tuple(sorted(pid for pid in replayed.state.by_id())) == tuple(sorted(pid for pid in terminal.by_id()))
    assert replayed.state.bank_tenths == terminal.bank_tenths
    assert replayed.state.free_transfers == terminal.free_transfers

    tampered = [dict(row) for row in actions]
    tampered[0]["bank_after"] += 1
    with pytest.raises(cra.ChipRouteAssemblyError, match="replay differs from the retained route"):
        cra.replay_serialized_route(
            tampered,
            start_state=start,
            expected_events=events,
            price_scenario=scenario,
            player_meta=fhfx.player_meta(),
        )


def test_bb_tc_evaluations_share_one_proposed_route_scenario_and_worlds():
    events = (5, 6, 7, 8)
    start = fhfx.start_state(event=5, bank_tenths=20, free_transfers=2)
    partial, terminal = fhfx.build_canonical_route(
        start=start, events=events, transfers={5: ((3, 18),)},
    )
    proposed = tuple(sorted(pid for pid in partial.actions[0]["transition"].next_event_state.by_id()))
    policy = ml.ManagerPolicy(
        starter_ids=(1, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14),
        bench_gk_id=2,
        bench_outfield_order=(18, 7, 15),
        captain_id=8,
        vice_captain_id=9,
    )
    positions = fhfx.positions_of(fhfx.UNIVERSE)
    assert set(ml.policy_player_ids(policy)) == set(proposed)
    assert ml.policy_legality_errors(policy, positions) == []
    snapshot_sha = "sha256:" + "d" * 64
    certification = "sha256:" + "c" * 64
    world_identity = "sha256:" + "w" * 64
    route = cra.VerifiedNormalRoute(
        source_decision_id="decision-fixture",
        source_result_sha256="sha256:" + "1" * 64,
        source_artifact_sha256="2" * 64,
        source_artifact_ref="<fixture>",
        generation_id="generation-fixture",
        planning_event=5,
        events=events,
        cutoff="2026-09-29T20:27:01Z",
        data_snapshot_sha256=snapshot_sha,
        certification_identity=certification,
        route_id="route_001",
        actual_owned_ids=tuple(sorted(pid for pid in start.by_id())),
        proposed_owned_ids=proposed,
        post_h1_bank_tenths=int(partial.actions[0]["transition"].next_event_state.bank_tenths),
        post_h1_free_transfers=int(partial.actions[0]["transition"].next_event_state.free_transfers),
        policy=policy,
        positions=positions,
        clubs={**fhfx.CLUB, **fhfx.POOL_CLUB},
        partial=partial,
        route_input_sha256="sha256:" + "3" * 64,
        runner_code_identity="sha256:" + "4" * 64,
    )
    player_ids = tuple(sorted(proposed))
    worlds = cd.ChipWorldInputs(
        worlds=8,
        player_ids=player_ids,
        minutes={pid: tuple(90.0 for _ in range(8)) for pid in player_ids},
        core={pid: tuple(2.0 for _ in range(8)) for pid in player_ids},
        planning_event=5,
        horizon_events=events,
        certification_identity=certification,
        data_snapshot_sha256=snapshot_sha,
        world_seed=20260911,
        world_identity=world_identity,
        code_snapshot_sha256="sha256:" + "5" * 64,
    )

    evaluations = cra.build_bb_tc_evaluations(route, worlds)
    bb_eval = evaluations[cd.CHIP_ACTION_BB]
    tc_eval = evaluations[cd.CHIP_ACTION_TC]
    assert bb_eval.evidence["scenario_kind"] == "PROPOSED_TRANSFER_ROUTE"
    assert tc_eval.evidence["scenario_kind"] == "PROPOSED_TRANSFER_ROUTE"
    assert bb_eval.evidence["scenario_identity"] == tc_eval.evidence["scenario_identity"]
    assert bb_eval.evidence["world_identity"] == tc_eval.evidence["world_identity"] == world_identity
    assert bb_eval.evidence["actual_owned_ids"] == list(route.actual_owned_ids)
    assert bb_eval.evidence["proposed_owned_ids"] == list(proposed)
    assert 3 in route.actual_owned_ids and 3 not in proposed
    assert 18 in proposed and 18 not in route.actual_owned_ids
    expected_save_state = {
        "event": 6,
        "squad_ids": list(proposed),
        "purchase_price_tenths": {
            str(pid): int(player.purchase_price_tenths)
            for pid, player in partial.actions[0]["transition"].next_event_state.by_id().items()
        },
        "bank_tenths": route.post_h1_bank_tenths,
        "free_transfers": route.post_h1_free_transfers,
        "event_start_free_transfers": route.post_h1_free_transfers,
        "chip_state": [],
        "source_decision_id": route.source_decision_id,
        "generation_id": route.generation_id,
        "route_id": route.route_id,
        "route_input_sha256": route.route_input_sha256,
    }
    bb_save_state = bb_eval.evidence["save_policy"]["post_save_state_for_reservation"]
    tc_save_state = tc_eval.evidence["save_policy"]["post_save_state_for_reservation"]
    assert bb_save_state == tc_save_state == expected_save_state
    assert bb_eval.execution_permitted is False
    assert tc_eval.execution_permitted is True


def test_free_hit_h2_state_uses_event_start_ft_not_current_remaining_ft():
    ids = tuple(fhfx.UNIVERSE)
    state = fha.FreeHitManagerState(
        entry_id=241392,
        planning_event=5,
        cutoff="2026-09-29T20:27:01Z",
        owned_ids=fhfx.SQUAD_IDS,
        purchase_price_tenths={pid: 50 for pid in fhfx.SQUAD_IDS},
        bank_tenths=70,
        free_transfers=2,
        event_start_free_transfers=3,
        positions={pid: fhfx.player_meta()[pid].position for pid in ids},
        clubs={pid: fhfx.player_meta()[pid].club_id for pid in ids},
        market_price_tenths={pid: 50 for pid in ids},
        eligible_ids=ids,
        pool_generation_identity="fixture-generation",
        pool_generation_id_sha256="sha256:" + "a" * 64,
    )
    rules = sr.SeasonRules(season="2026/27")
    restored, play_start = fhp._h2_restored_route_state(
        state, rules=rules, chip_state=({"name": "freehit", "number": 1},),
    )
    assert restored.owned_ids == state.owned_ids
    assert dict(restored.purchase_price_tenths) == dict(state.purchase_price_tenths)
    assert restored.bank_tenths == state.bank_tenths
    assert state.free_transfers == 2
    assert restored.free_transfers == 3
    assert play_start.event == 6
    assert play_start.event_start_free_transfers == 3
    assert play_start.chip_state == ({"name": "freehit", "number": 1},)


def test_free_hit_production_builder_emits_retained_play_save_arms(monkeypatch, tmp_path):
    """Exercise the production arm builder and retain its exact emitted evidence."""

    events = (5, 6, 7, 8)
    cutoff = "2026-09-29T20:27:01Z"
    snapshot_sha = "sha256:" + "d" * 64
    code_sha = "sha256:" + "c" * 64
    source_result_sha = "sha256:" + "1" * 64
    source_artifact_sha = "2" * 64
    meta = fhfx.player_meta()
    eligible_ids = tuple(sorted(meta))
    positions = {pid: row.position for pid, row in meta.items()}
    clubs = {pid: row.club_id for pid, row in meta.items()}
    basis = {pid: 50 for pid in fhfx.SQUAD_IDS}
    manager = fha.FreeHitManagerState(
        entry_id=241392,
        planning_event=events[0],
        cutoff=cutoff,
        owned_ids=fhfx.SQUAD_IDS,
        purchase_price_tenths=basis,
        bank_tenths=70,
        free_transfers=2,
        event_start_free_transfers=3,
        positions=positions,
        clubs=clubs,
        market_price_tenths={pid: 50 for pid in eligible_ids},
        eligible_ids=eligible_ids,
        pool_generation_identity="accepted-pool-fixture",
        pool_generation_id_sha256="sha256:" + "a" * 64,
    )
    start = fhfx.start_state(event=events[0], bank_tenths=70, free_transfers=2, basis=basis)
    save_partial, _ = fhfx.build_canonical_route(start=start, events=events)
    normal_config = ro.OptimizerConfig(events=events, search_draws=1, seed=20260911)
    route = cra.VerifiedNormalRoute(
        source_decision_id="decision-fixture",
        source_result_sha256=source_result_sha,
        source_artifact_sha256=source_artifact_sha,
        source_artifact_ref="<fixture-artifact>",
        generation_id="generation-fixture",
        planning_event=events[0],
        events=events,
        cutoff=cutoff,
        data_snapshot_sha256=snapshot_sha,
        certification_identity="sha256:" + "e" * 64,
        route_id="route-save-fixture",
        actual_owned_ids=fhfx.SQUAD_IDS,
        proposed_owned_ids=fhfx.SQUAD_IDS,
        post_h1_bank_tenths=70,
        post_h1_free_transfers=2,
        policy=None,
        positions=positions,
        clubs=clubs,
        partial=save_partial,
        route_input_sha256="sha256:" + "3" * 64,
        runner_code_identity="sha256:" + "4" * 64,
    )

    runs = {event: {"minutes_v1": event, "xpts_v1": event + 100} for event in events}
    generation = SimpleNamespace(
        generation_id="generation-fixture",
        horizon_kind=gs.HORIZON_KIND_FOUR_GW,
        planning_event=events[0],
        events=events,
        cutoff=cutoff,
        snapshot={"path": str(tmp_path / "snapshot.sqlite"), "sha256": snapshot_sha},
        manifest={"code_snapshot_sha256": code_sha},
        runs_for=lambda event: runs[int(event)],
    )
    pool_binding = SimpleNamespace(
        eligible_ids=eligible_ids,
        as_dict=lambda: {
            "generation_identity": manager.pool_generation_identity,
            "generation_id_sha256": manager.pool_generation_id_sha256,
            "eligible_ids": list(eligible_ids),
        },
    )
    captured = {}

    monkeypatch.setattr(fhp, "load_verified_normal_route", lambda *_args, **_kwargs: route)
    monkeypatch.setattr(gs, "load_generation", lambda *_args, **_kwargs: generation)
    monkeypatch.setattr(gs, "verify_generation", lambda *_args, **_kwargs: {"verified": True})
    monkeypatch.setattr(gs, "_open_generation_snapshot", lambda _generation: sqlite3.connect(":memory:"))
    monkeypatch.setattr(fha, "free_hit_manager_state", lambda *_args, **_kwargs: manager)
    monkeypatch.setattr(fha, "resolve_pool_binding", lambda _conn: pool_binding)
    monkeypatch.setattr(fhp, "_config_from_source_decision", lambda *_args: normal_config)
    monkeypatch.setattr(
        cu, "price_snapshot_as_of",
        lambda *_args, **_kwargs: ts.PriceSnapshot(event=events[0], prices={pid: 50 for pid in eligible_ids}),
    )
    monkeypatch.setattr(cu, "load_pool", lambda _conn: {pid: {} for pid in eligible_ids})
    monkeypatch.setattr(cu, "load_fixtures_by_team", lambda *_args: {})
    monkeypatch.setattr(cu, "load_projection_rows", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        cu, "build_universe",
        lambda **_kwargs: {"universe": [{"player_id": pid} for pid in eligible_ids]},
    )
    monkeypatch.setattr(cra.rc, "load_player_meta", lambda *_args: meta)
    monkeypatch.setattr(cu, "build_replacement_edges", lambda **_kwargs: [])
    monkeypatch.setattr(fhp, "_free_hit_route_cache_key", lambda *_args, **_kwargs: "h1-world-key")

    def optimize(*_args, **kwargs):
        play_partial, _terminal = fhfx.build_canonical_route(
            start=kwargs["initial_state"], events=tuple(kwargs["config"].events),
        )
        serialized = ro._serialize_route(play_partial)
        captured["play_partial"] = play_partial
        return {"routes": {"play-fixture": {
            "actions": serialized,
            "terminal_bank_tenths": play_partial.state.bank_tenths,
            "terminal_ft": play_partial.state.free_transfers,
        }}}

    monkeypatch.setattr(ro, "optimize", optimize)
    monkeypatch.setattr(ro, "best_route_family", lambda *_args, **_kwargs: {"route_id": "play-fixture"})
    monkeypatch.setattr(ro, "build_event_worlds", lambda *_args, **_kwargs: ({
        ro.MANAGER_MATRIX_IDENTITY_KEY: "h1-world-key",
        "worlds": 1,
        "player_ids": list(eligible_ids),
        "minutes": {pid: [90.0] for pid in eligible_ids},
        "core": {pid: [1.0] for pid in eligible_ids},
    }, {}))

    def accept_canonical_inputs(supplied_manager, certified, **kwargs):
        captured["manager"] = supplied_manager
        captured["certified"] = certified
        assert kwargs["manager_source_conn"] is not None
        assert certified.save.partial is save_partial
        assert tuple(action["event"] for action in certified.play.partial.actions) == events[1:]
        assert fh.state_fingerprint(certified.play.partial.state) == fh.state_fingerprint(
            captured["play_partial"].state
        )
        assert certified.horizon_binding.horizon_events == events
        assert certified.h1_worlds.identity == certified.world_identity
        return object()

    monkeypatch.setattr(fha, "build_free_hit_request", accept_canonical_inputs)
    monkeypatch.setattr(fhp.fh, "contract_problems", lambda _request: [])

    _request, arms = fhp.build_free_hit_production_request(
        sqlite3.connect(":memory:"),
        decision_id="decision-fixture",
        entry_id=manager.entry_id,
        certification_path="<fixture-certification>",
        rules=sr.SeasonRules(season="2026/27"),
        route_id=route.route_id,
        cache_dir=tmp_path,
    )

    assert captured["manager"] is manager
    assert arms["source_result_sha256"] == source_result_sha
    assert arms["source_artifact_sha256"] == source_artifact_sha
    assert [row["event"] for row in arms["save"]["actions"]] == list(events)
    assert [row["event"] for row in arms["play"]["actions"]] == list(events[1:])
    assert arms["restore_at_h2"]["permanent_squad_ids"] == arms["restore_at_h2"]["restored_squad_ids"]
    assert arms["restore_at_h2"]["restored_h2_free_transfers"] == 3

    manager_record = {
        "entry_id": manager.entry_id,
        "planning_event": manager.planning_event,
        "cutoff": manager.cutoff,
        "squad_ids": list(manager.owned_ids),
        "purchase_price_tenths": dict(manager.purchase_price_tenths),
        "bank_tenths": manager.bank_tenths,
        "free_transfers": manager.free_transfers,
        "event_start_free_transfers": manager.event_start_free_transfers,
    }
    manager_identity = chip_store.manager_state_identity(manager_record)
    context = {
        "planning_event": route.planning_event,
        "horizon_events": list(events),
        "cutoff": route.cutoff,
        "data_snapshot_sha256": route.data_snapshot_sha256,
        "certification_identity": route.certification_identity,
        "generation_id": route.generation_id,
        "source_decision_id": route.source_decision_id,
        "source_decision_result_sha256": route.source_result_sha256,
        "source_decision_artifact_sha256": route.source_artifact_sha256,
        "manager_state_identity": manager_identity,
    }
    shared = {
        "source_decision_id": route.source_decision_id,
        "source_result_sha256": route.source_result_sha256,
        "source_artifact_sha256": route.source_artifact_sha256,
        "generation_id": route.generation_id,
        "cutoff": route.cutoff,
        "data_snapshot_sha256": route.data_snapshot_sha256,
        "certification_identity": route.certification_identity,
        "manager_state_identity": manager_identity,
        "world_identity": arms["world_identity"],
    }
    arms["play"].update(shared)
    arms["save"].update(shared)
    arms["arm_identity"] = chip_store.free_hit_arm_identity(arms)
    results = {
        action: {"status": "BLOCKED", "reason_codes": [f"{action}_FIXTURE_BLOCK"]}
        for action in cd.PLAYABLE_CHIP_ACTIONS
    }
    evaluation = cd.ChipEvaluation(
        action=cd.CHIP_ACTION_FH,
        evaluator_version="fixture-fh",
        candidate_metrics={},
        uncertainty={},
        evidence={
            "certification_identity": route.certification_identity,
            "data_snapshot_sha256": route.data_snapshot_sha256,
            "planning_event": route.planning_event,
            "horizon_events": list(events),
        },
        execution_permitted=False,
        data_snapshot_bound=True,
    )
    results[cd.CHIP_ACTION_FH] = {
        "status": "EVALUATED",
        "evaluation": {
            "action": evaluation.action,
            "evaluator_version": evaluation.evaluator_version,
            "candidate_metrics": {},
            "uncertainty": {},
            "reason_codes": [],
            "calibration_status": evaluation.calibration_status,
            "evidence": dict(evaluation.evidence),
            "execution_permitted": False,
        },
        "reason_codes": [],
        "arm_evidence": arms,
        "arm_identity": arms["arm_identity"],
    }
    record = chip_store.build_assessment_record(
        context=context,
        manager_state=manager_record,
        chip_results=results,
        decision={
            "recommended_action": cd.CHIP_ACTION_NO_CHIP,
            "status": cd.STATUS_CHIP_REVIEW_REQUIRED,
            "calibration_status": cd.CALIBRATION_UNCALIBRATED,
        },
    )
    receipt = chip_store.retain_assessment(record, tmp_path / "assessments")
    assert chip_store.verify_assessment(receipt["path"])["verified"] is True


def test_wildcard_certified_rows_require_every_fixture_and_mark_real_blank_as_zero():
    identity = wcp.WildcardPredictiveIdentity(
        cutoff="2026-09-29T20:27:01Z",
        data_snapshot_sha256="sha256:" + "d" * 64,
        source_snapshot_sha256="sha256:" + "c" * 64,
        generation="generation-fixture",
        model_config_identity="sha256:" + "m" * 64,
    )
    common = dict(
        pool={1: {"position": "MID", "club_id": 100, "web_name": "P1"}},
        prices={1: 50},
        events=(5, 6, 7, 8, 9, 10),
        fixtures={(5, 100): [500]},
        xpts_by_event={5: {(1, 500): {
            "core_xpts": 3.2, "expected_minutes": 70.0, "p_start": 0.75,
        }}},
        minutes_by_event={5: {(1, 500): {"joint_availability": 0.9}}},
        identity=identity,
    )
    players = wcp._players_from_certified_rows(**common)
    assert players[1].at(5).expected_points == pytest.approx(3.2)
    assert players[1].at(6).fixture_count == 0
    assert players[1].at(6).expected_points == 0.0
    assert players[1].at(6).identity == identity

    broken = dict(common)
    broken["events"] = (5, 6, 7, 8, 9, 10)
    broken["xpts_by_event"] = {5: {}}
    with pytest.raises(wcp.WildcardProductionError, match="fixture coverage gaps"):
        wcp._players_from_certified_rows(**broken)


def test_wildcard_world_loader_rejects_wrong_cached_world_identity(monkeypatch):
    class Generation:
        generation_id = "generation-fixture"

        def runs_for(self, event):
            return {"minutes_v1": 1, "team_strength_v1": 2,
                    "player_rates_v1": 3, "xpts_v1": 4}

    def wrong_worlds(*_args, **_kwargs):
        return ({
            ro.MANAGER_MATRIX_IDENTITY_KEY: "wrong-world",
            "worlds": 1,
            "player_ids": [1],
            "minutes": {1: [90.0]},
            "core": {1: [1.0]},
        }, {})

    monkeypatch.setattr(ro, "build_event_worlds", wrong_worlds)
    identity = wcp.WildcardPredictiveIdentity(
        cutoff="2026-09-29T20:27:01Z",
        data_snapshot_sha256="sha256:" + "d" * 64,
        source_snapshot_sha256="sha256:" + "c" * 64,
        generation="generation-fixture",
        model_config_identity="sha256:" + "m" * 64,
    )
    with pytest.raises(wcp.WildcardProductionError, match="not from the canonical certified loader"):
        wcp._canonical_worlds(
            object(), generation=Generation(), events=(5,), player_ids=(1,),
            config=ro.OptimizerConfig(events=(5,), search_draws=1, seed=1),
            cache_dir=None, identity=identity,
        )


def test_wildcard_production_builder_binds_six_event_product_to_four_event_prefix(monkeypatch, tmp_path):
    """Exercise the production builder's distinct, same-cutoff value product."""

    from fpl_brain import repositories
    from fpl_brain.ingest_provenance import element_id_sha256

    chip_events = (5, 6, 7, 8)
    value_events = (5, 6, 7, 8, 9, 10)
    cutoff = "2026-09-29T20:27:01Z"
    snapshot_sha = "sha256:" + "d" * 64
    code_sha = "sha256:" + "c" * 64
    meta = fhfx.player_meta()
    eligible_ids = tuple(sorted(meta))
    official_digest = element_id_sha256(list(eligible_ids))
    pool_binding = cw.WildcardPoolBinding(
        generation_identity="accepted-pool-fixture",
        generation_id_sha256=official_digest,
        official_count=len(eligible_ids),
        eligible_ids=eligible_ids,
    )
    pool = {
        pid: {"position": player.position, "club_id": player.club_id, "web_name": f"p{pid}"}
        for pid, player in meta.items()
    }
    owned = fhfx.SQUAD_IDS
    manager = wa.WildcardManagerState(
        entry_id=241392,
        planning_event=chip_events[0],
        squad_ids=owned,
        bank_tenths=70,
        purchase_price_tenths={pid: 50 for pid in owned},
        event_start_free_transfers=3,
        chip_availability=({
            "name": "wildcard", "available_for_event": True,
            "window_start_event": 2, "window_stop_event": 19,
            "used": False, "expired": False,
        },),
        market_price_tenths={pid: 50 for pid in eligible_ids},
    )
    chip_runs = {
        event: {
            "minutes_v1": event,
            "team_strength_v1": event + 10,
            "player_rates_v1": event + 20,
            "xpts_v1": event + 30,
        }
        for event in chip_events
    }
    value_runs = {
        **chip_runs,
        9: {"minutes_v1": 9, "team_strength_v1": 19, "player_rates_v1": 29, "xpts_v1": 39},
        10: {"minutes_v1": 10, "team_strength_v1": 20, "player_rates_v1": 30, "xpts_v1": 40},
    }

    def generation(generation_id, horizon_kind, events, runs):
        return SimpleNamespace(
            generation_id=generation_id,
            horizon_kind=horizon_kind,
            planning_event=chip_events[0],
            events=events,
            cutoff=cutoff,
            snapshot={"path": str(tmp_path / "snapshot.sqlite"), "sha256": snapshot_sha},
            manifest={"code_snapshot_sha256": code_sha, "last_event": 38},
            runs_for=lambda event: runs[int(event)],
        )

    chip_generation = generation(
        "chip-generation-fixture", gs.HORIZON_KIND_FOUR_GW, chip_events, chip_runs,
    )
    value_generation = generation(
        "wildcard-value-generation-fixture", gs.HORIZON_KIND_WILDCARD_VALUE,
        value_events, value_runs,
    )
    route = SimpleNamespace(
        source_decision_id="decision-fixture",
        source_result_sha256="sha256:" + "1" * 64,
        source_artifact_sha256="2" * 64,
        generation_id=chip_generation.generation_id,
        planning_event=chip_events[0],
        events=chip_events,
        cutoff=cutoff,
        data_snapshot_sha256=snapshot_sha,
        certification_identity="sha256:" + "e" * 64,
        route_id="route-001",
        route_input_sha256="sha256:" + "3" * 64,
        partial=object(),
    )
    config = ro.OptimizerConfig(events=chip_events, search_draws=1, seed=20260911)
    accepted_pool = {
        "accepted": True,
        "captured_at": "accepted-pool-fixture",
        "official_element_count": len(eligible_ids),
        "element_ids": list(eligible_ids),
        "element_id_sha256": official_digest,
    }
    captured = {}

    monkeypatch.setattr(wcp, "load_verified_normal_route", lambda *_args, **_kwargs: route)
    monkeypatch.setattr(
        gs, "load_generation",
        lambda _conn, generation_id: {
            chip_generation.generation_id: chip_generation,
            value_generation.generation_id: value_generation,
        }[str(generation_id)],
    )
    monkeypatch.setattr(gs, "verify_generation", lambda *_args, **_kwargs: {"verified": True})
    monkeypatch.setattr(gs, "_open_generation_snapshot", lambda _generation: sqlite3.connect(":memory:"))
    monkeypatch.setattr(wcp, "pool_binding_from_store", lambda _conn: pool_binding)
    monkeypatch.setattr(wa, "wildcard_manager_state", lambda *_args, **_kwargs: manager)
    monkeypatch.setattr(cu, "load_pool", lambda _conn: pool)
    monkeypatch.setattr(wcp, "_route_config_from_decision", lambda *_args: config)
    monkeypatch.setattr(cu, "load_fixtures_by_team", lambda *_args: {})
    monkeypatch.setattr(cu, "load_projection_rows", lambda *_args, **_kwargs: {})

    def players_from_rows(**kwargs):
        captured["player_events"] = tuple(kwargs["events"])
        return {}

    monkeypatch.setattr(wcp, "_players_from_certified_rows", players_from_rows)

    def canonical_worlds(_conn, *, generation, events, player_ids, config, cache_dir, identity):
        captured.setdefault("world_sources", []).append((generation.generation_id, tuple(events)))
        return {
            int(event): cw.WildcardWorldInputs(
                event=int(event), worlds=1, player_ids=tuple(player_ids),
                minutes={int(pid): (90.0,) for pid in player_ids},
                core={int(pid): (3.0,) for pid in player_ids}, identity=identity,
            )
            for event in events
        }

    monkeypatch.setattr(wcp, "_canonical_worlds", canonical_worlds)
    monkeypatch.setattr(repositories, "latest_accepted_bootstrap_generation", lambda _conn: accepted_pool)

    def accept_production_inputs(supplied_manager, certified, route_arg, **kwargs):
        captured["manager"] = supplied_manager
        captured["certified"] = certified
        captured["canonical_route"] = kwargs["canonical_route"]
        assert route_arg is None
        assert kwargs["conn"] is not None
        return SimpleNamespace(accepted=True)

    monkeypatch.setattr(wa, "build_wildcard_request", accept_production_inputs)
    request, evidence = wcp.build_wildcard_production_request(
        sqlite3.connect(":memory:"),
        decision_id=route.source_decision_id,
        value_generation_id=value_generation.generation_id,
        entry_id=manager.entry_id,
        length=len(value_events),
        rules=sr.SeasonRules(season="2026/27"),
        route_id=route.route_id,
    )

    certified = captured["certified"]
    assert request.accepted is True
    assert captured["manager"] is manager
    assert certified.chip_generation_id == chip_generation.generation_id
    assert certified.value_generation_id == value_generation.generation_id
    assert certified.horizon.events == value_events
    assert certified.horizon.events[:4] == chip_generation.events
    assert certified.value_horizon_binding.decision_cutoff == chip_generation.cutoff
    assert certified.value_horizon_binding.data_snapshot_sha256 == snapshot_sha
    assert certified.value_horizon_binding.source_snapshot_sha256 == code_sha
    assert certified.value_horizon_binding.prediction_generation == value_generation.generation_id
    assert evidence["events"] == list(value_events)
    assert evidence["projection_run_ids_by_event"]["5"] == chip_generation.runs_for(5)
    assert captured["world_sources"] == [
        (value_generation.generation_id, value_events),
        (chip_generation.generation_id, chip_events),
    ]
