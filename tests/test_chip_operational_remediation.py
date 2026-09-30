"""Regression coverage for the CHIP operational remediation boundaries."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from fpl_brain import execution, generation_store as gs
from fpl_brain.database import connect_database

import generation_fixtures as gf
import test_pe9_certification_integration as pe9
import free_hit_route_fixtures as fhfx

from fpl_brain import chip_decision as cd, chip_route_assembly as cra, free_hit_production as fhp
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
