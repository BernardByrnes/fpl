from fpl_brain import chip_horizon_evidence_assembly as assembly
from fpl_brain import chip_route_assembly as cra, route_comparator as rc, route_optimizer as ro
from fpl_brain import transfer_state as ts

import pytest


def _one_hit_route():
    positions = ["GKP"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
    players = tuple(
        ts.RoutePlayer(pid, position, (pid - 1) // 3 + 1, 50)
        for pid, position in enumerate(positions, start=1)
    )
    meta = {p.player_id: ts.PlayerMeta(p.player_id, p.position, p.club_id) for p in players}
    meta[16] = ts.PlayerMeta(16, "MID", 6)
    prices = {pid: 55 for pid in meta}
    start = ts.RouteState(
        event=9, players=players, bank_tenths=20, free_transfers=0,
        event_start_free_transfers=0,
    )
    batch = ts.TransferBatch((ts.TransferAction(8, 16),))
    transition = ts.apply_transfer_batch(
        start, batch, ts.PriceSnapshot(event=9, prices=prices), meta,
    )
    assert transition.ok and transition.hit_points == 4
    partial = ro.PartialRoute(
        state=transition.next_event_state,
        actions=({
            "event": 9, "kind": "HIT", "batch": batch, "transition": transition,
            "hit_points": transition.hit_points,
            "squad_ids": tuple(sorted(transition.next_event_state.by_id())),
            "ft_after": transition.next_event_state.free_transfers,
            "bank_after": transition.next_event_state.bank_tenths,
        },),
        h1_proxy=0.0, window_proxy=0.0, hits=4,
    )
    return start, meta, prices, partial


def test_canonical_replay_rejects_a_resealed_false_zero_hit_claim():
    start, meta, prices, partial = _one_hit_route()
    rows = ro._serialize_route(partial)
    scenario = rc.flat_current_price_scenario(ts.PriceSnapshot(event=9, prices=prices), (9,))
    replay = cra.replay_serialized_route(
        rows, start_state=start, expected_events=(9,), price_scenario=scenario, player_meta=meta,
    )
    assert replay.hits == 4
    forged = [dict(rows[0], hit_points=0)]
    with pytest.raises(cra.ChipRouteAssemblyError, match="hit_points"):
        cra.replay_serialized_route(
            forged, start_state=start, expected_events=(9,),
            price_scenario=scenario, player_meta=meta,
        )
    with pytest.raises(assembly.FutureStateAssemblyError, match="source-replayed transfer hit cost 4"):
        assembly._require_zero_hit_arms(
            "FH", expected_events=(9,),
            arms={"PLAY": [{"event": 9, "hit_points": replay.actions[0]["hit_points"]}],
                  "SAVE": [{"event": 9, "hit_points": 0}]},
        )


def test_zero_hit_gate_requires_both_arms_every_scored_event_and_exact_integers():
    good = {
        "PLAY": [{"event": 5, "hit_points": 0}, {"event": 6, "hit_points": 0}],
        "SAVE": [{"event": 5, "hit_points": 0}, {"event": 6, "hit_points": 0}],
    }
    assembly._require_zero_hit_arms("WC", expected_events=(5, 6), arms=good)
    later_nonselected_hit = {
        "PLAY": good["PLAY"],
        "SAVE": [{"event": 5, "hit_points": 0}, {"event": 6, "hit_points": 4}],
    }
    with pytest.raises(assembly.FutureStateAssemblyError, match="GW6"):
        assembly._require_zero_hit_arms("WC", expected_events=(5, 6), arms=later_nonselected_hit)
    equal_paid_arms = {
        "PLAY": [{"event": 5, "hit_points": 4}, {"event": 6, "hit_points": 0}],
        "SAVE": [{"event": 5, "hit_points": 4}, {"event": 6, "hit_points": 0}],
    }
    with pytest.raises(assembly.FutureStateAssemblyError, match="GW5"):
        assembly._require_zero_hit_arms("WC", expected_events=(5, 6), arms=equal_paid_arms)
    equal_paid_arms = {
        "PLAY": [{"event": 5, "hit_points": 4}, {"event": 6, "hit_points": 0}],
        "SAVE": [{"event": 5, "hit_points": 4}, {"event": 6, "hit_points": 0}],
    }
    with pytest.raises(assembly.FutureStateAssemblyError, match="GW5"):
        assembly._require_zero_hit_arms("WC", expected_events=(5, 6), arms=equal_paid_arms)
    bool_is_not_an_integer = {
        "PLAY": [{"event": 5, "hit_points": False}, {"event": 6, "hit_points": 0}],
        "SAVE": good["SAVE"],
    }
    with pytest.raises(assembly.FutureStateAssemblyError, match="exact integer"):
        assembly._require_zero_hit_arms("WC", expected_events=(5, 6), arms=bool_is_not_an_integer)
    with pytest.raises(assembly.FutureStateAssemblyError, match="both PLAY and SAVE"):
        assembly._require_zero_hit_arms("FH", expected_events=(5,), arms={"PLAY": [{"event": 5, "hit_points": 0}]})
