"""Free Hit route authority — canonical SAVE, canonical PLAY tail, value authority.

Built on REAL canonical routes (``apply_transfer_batch`` transitions, not fake
DTOs) so these tests speak the same language as production route authority.

The decisive case is Sol's P1 counterexample: the alternative to playing Free Hit
is the accepted normal H1-H4 route, which may make a beneficial H1 transfer and
so enter H2 with a different squad, bank, basis and free-transfer count.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import free_hit_route_fixtures as fx  # noqa: E402
from fpl_brain import chip_free_hit as fh  # noqa: E402
from fpl_brain import free_hit_route as fr  # noqa: E402
from fpl_brain import season_rules as sr  # noqa: E402
from fpl_brain import transfer_state as ts  # noqa: E402

RULES = sr.SeasonRules(season="2026/27")
H1, H2, H3, H4 = 5, 6, 7, 8
SAVE_EVENTS = (H1, H2, H3, H4)
PLAY_EVENTS = (H2, H3, H4)
UNIVERSE = fx.UNIVERSE
SQUAD = fx.SQUAD_IDS


def _start(*, event, ids=SQUAD, bank=0, ft=2, basis=None):
    return fx.start_state(event=event, squad_ids=ids, bank_tenths=bank, free_transfers=ft, basis=basis)


def _expected(ids=SQUAD, bank=0, ft=2, event=H1, basis=None):
    return {
        "event": int(event), "squad_ids": list(ids), "bank_tenths": int(bank),
        "free_transfers": int(ft),
        "purchase_price_tenths": {int(p): int((basis or {}).get(int(p), fx.BASE_PURCHASE)) for p in ids},
    }


def _worlds(events, *, scores=None):
    return {int(e): fx.world_matrix(event=int(e), scores=scores) for e in events}


def _save_route(*, transfers=None, prices=None, start=None, worlds=None, expected=None, event=H1):
    partial, terminal = fx.build_canonical_route(
        start=start if start is not None else _start(event=event),
        events=SAVE_EVENTS, transfers=transfers, prices=prices,
    )
    return fr.free_hit_route_from_canonical_route(
        partial, arm=fr.ARM_SAVE, expected_events=SAVE_EVENTS, rules=RULES,
        expected_start_state=expected if expected is not None else _expected(event=event),
        worlds_by_event=worlds if worlds is not None else _worlds(SAVE_EVENTS),
        positions_of=fx.positions_of, route_config=fx.route_config(),
    ), partial, terminal


def _play_route(*, transfers=None, start=None, worlds=None, expected=None):
    partial, terminal = fx.build_canonical_route(
        start=start if start is not None else _start(event=H2, bank=7, ft=2),
        events=PLAY_EVENTS, transfers=transfers,
    )
    return fr.free_hit_route_from_canonical_route(
        partial, arm=fr.ARM_PLAY, expected_events=PLAY_EVENTS, rules=RULES,
        expected_start_state=expected if expected is not None else _expected(event=H2, bank=7, ft=2),
        worlds_by_event=worlds if worlds is not None else _worlds(PLAY_EVENTS),
        positions_of=fx.positions_of, route_config=fx.route_config(),
    ), partial, terminal


# ---------------------------------------------------------------------------
# P1 — THE SAVE COUNTEREXAMPLE
# ---------------------------------------------------------------------------


def test_P1_save_is_the_normal_route_and_a_beneficial_h1_transfer_changes_h2():
    """The alternative to the chip is NOT 'do nothing in H1'."""

    route, partial, _terminal = _save_route(transfers={H1: ((12, 22),)}, prices={H1: {22: 40}})
    first = partial.actions[0]
    assert first["kind"] == "NORMAL_TRANSFER", "the canonical SAVE route uses the transfer"
    assert int(first["hit_points"]) == 0, "2 FT pays for one transfer"

    h2 = route.event_for(H2)
    # H2 ARISES FROM THE ROUTE TRANSITION, never from a synthesised state.
    assert 12 not in h2.squad_ids and 22 in h2.squad_ids
    assert int(h2.bank_tenths) == 10, "50 in, 40 out"
    assert int(h2.purchase_price_tenths[22]) == 40, "the bought player's canonical basis"
    assert int(h2.free_transfers) == 3, "one of two FT spent, then the weekly +1"


def test_P1_save_h2_is_NOT_the_unchanged_permanent_squad():
    """The exact failure Sol identified: forcing SAVE back to H1's squad."""

    route, _partial, _terminal = _save_route(transfers={H1: ((12, 22),)}, prices={H1: {22: 40}})
    h2 = route.event_for(H2)
    assert tuple(sorted(h2.squad_ids)) != tuple(sorted(SQUAD)), "SAVE must not restore H1's squad"
    assert route.start_state["squad_ids"] == list(SQUAD), "it DID start from the permanent squad"


def test_P1_the_save_route_value_is_one_evaluation_of_one_route():
    route, _partial, _terminal = _save_route(transfers={H1: ((12, 22),)}, prices={H1: {22: 40}})
    assert len(route.events) == 4
    assert route.value() == pytest.approx(sum(entry.mean_net_core for entry in route.events))
    assert route.converter_version == fr.FREE_HIT_ROUTE_CONVERTER_VERSION
    h1_entry = route.event_for(H1)
    assert h1_entry is not None and h1_entry.mean_gross_core > 0.0


def test_P1_a_no_transfer_save_route_also_works():
    """SAVE is not hard-coded to one shape: a rolling route is canonical too."""

    route, partial, _terminal = _save_route()
    assert [a["kind"] for a in partial.actions] == ["ROLL"] * 4
    assert tuple(sorted(route.event_for(H2).squad_ids)) == tuple(sorted(SQUAD))
    assert route.value() > 0.0


def test_P1_a_hit_on_the_save_route_is_netted_exactly_once():
    route, _partial, _terminal = _save_route(
        transfers={H1: ((12, 22), (11, 21))},
        start=_start(event=H1, ft=1),
        expected=_expected(event=H1, ft=1),
    )
    h1_entry = route.event_for(H1)
    assert int(h1_entry.hit_points) == 4
    assert h1_entry.mean_net_core == pytest.approx(h1_entry.mean_gross_core - 4.0)
    assert route.cumulative_hits == 4
    assert route.value() == pytest.approx(sum(e.mean_gross_core for e in route.events) - 4.0)


def test_P1_save_handles_legitimately_negative_event_values():
    worlds = _worlds(SAVE_EVENTS, scores={pid: -0.5 for pid in UNIVERSE})
    route, _partial, _terminal = _save_route(worlds=worlds)
    assert all(entry.mean_gross_core < 0.0 for entry in route.events)
    assert route.value() == pytest.approx(sum(e.mean_net_core for e in route.events))


# ---------------------------------------------------------------------------
# PLAY — the H2-H4 tail from the RESTORED state
# ---------------------------------------------------------------------------


def test_PLAY_the_tail_starts_from_the_restored_permanent_state():
    route, _partial, _terminal = _play_route(transfers={H3: ((12, 22),)})
    assert route.arm == fr.ARM_PLAY
    assert [entry.event for entry in route.events] == list(PLAY_EVENTS)
    assert route.start_state["squad_ids"] == list(SQUAD)
    assert route.start_state["free_transfers"] == 2


def test_PLAY_no_temporary_free_hit_player_appears_in_the_tail():
    temporary = (16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 12, 13, 14)
    route, partial, _terminal = _play_route()
    assert set(route.start_state["squad_ids"]) == set(SQUAD)
    assert not (set(route.start_state["squad_ids"]) & (set(temporary) - set(SQUAD)))
    assert set(int(p.player_id) for p in partial.state.players) == set(SQUAD)


# ---------------------------------------------------------------------------
# VALUE AUTHORITY — the three-pass converter
# ---------------------------------------------------------------------------


def test_the_converter_has_no_route_value_parameter():
    """The 1,000,000 attack is unrepresentable, not merely defended."""

    import inspect

    parameters = set(inspect.signature(fr.free_hit_route_from_canonical_route).parameters)
    assert not parameters & {"mean_net_core", "values", "evaluation", "per_event", "value",
                             "route_value", "tail_value"}
    assert "expected_start_state" in parameters and "worlds_by_event" in parameters


def test_a_caller_cannot_inject_a_route_value_into_the_evaluation():
    """No production field carries a caller-supplied route point value."""

    route, _partial, _terminal = _save_route()
    for field in ("mean_net_core", "mean_gross_core", "hit_points"):
        assert field in fr.FreeHitRouteEvent.__dataclass_fields__
    # The DTO is frozen and every numeric field is populated from the evaluator.
    with pytest.raises(Exception):
        route.events[0].mean_net_core = 1_000_000        # type: ignore[misc]
    assert route.value() < 1_000_000


def test_the_converter_refuses_without_canonical_evaluation_inputs():
    partial, _terminal = fx.build_canonical_route(start=_start(event=H1), events=SAVE_EVENTS)
    with pytest.raises(fr.FreeHitRouteError) as caught:
        fr.free_hit_route_from_canonical_route(
            partial, arm=fr.ARM_SAVE, expected_events=SAVE_EVENTS, rules=RULES,
        )
    assert caught.value.reasons[0] == fr.FH_ROUTE_EVALUATION_INPUTS_REQUIRED


# ---------------------------------------------------------------------------
# PASS 1 — STATE AUTHORITY AND THE ATTACK MATRIX
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mutation,needle,token", [
    ("squad", "squad differs", fr.FH_ROUTE_START_STATE_MISMATCH),
    ("bank", "bank differs", fr.FH_ROUTE_START_STATE_MISMATCH),
    ("ft", "free transfers differ", fr.FH_ROUTE_START_STATE_MISMATCH),
    ("basis", "acquisition basis differs", fr.FH_ROUTE_BASIS_MISMATCH),
])
def test_a_wrong_start_state_refuses(mutation, needle, token):
    expected = _expected()
    if mutation == "squad":
        expected["squad_ids"] = [*SQUAD[:-1], 22]
    elif mutation == "bank":
        expected["bank_tenths"] = 99
    elif mutation == "ft":
        expected["free_transfers"] = 5
    else:
        expected["purchase_price_tenths"] = {pid: fx.BASE_PURCHASE + 1 for pid in SQUAD}
    with pytest.raises(fr.FreeHitRouteError) as caught:
        _save_route(expected=expected)
    assert caught.value.reasons[0] == token
    assert needle in str(caught.value)


def test_the_acquisition_basis_attack_is_the_one_sol_found_missing():
    """Same ids, same bank, same FT — ONLY a basis altered."""

    partial, _terminal = fx.build_canonical_route(start=_start(event=H1), events=SAVE_EVENTS)
    altered = {pid: fx.BASE_PURCHASE for pid in SQUAD}
    altered[12] = fx.BASE_PURCHASE + 3
    with pytest.raises(fr.FreeHitRouteError) as caught:
        fr.free_hit_route_from_canonical_route(
            partial, arm=fr.ARM_SAVE, expected_events=SAVE_EVENTS, rules=RULES,
            expected_start_state={**_expected(), "purchase_price_tenths": altered},
            worlds_by_event=_worlds(SAVE_EVENTS), positions_of=fx.positions_of,
            route_config=fx.route_config(),
        )
    assert caught.value.reasons[0] == fr.FH_ROUTE_BASIS_MISMATCH


def test_a_route_that_is_not_a_successful_canonical_transition_refuses():
    partial, _terminal = fx.build_canonical_route(start=_start(event=H1), events=SAVE_EVENTS)
    broken = ro_partial_with_bad_transition(partial)
    with pytest.raises(fr.FreeHitRouteError) as caught:
        fr.free_hit_route_from_canonical_route(
            broken, arm=fr.ARM_SAVE, expected_events=SAVE_EVENTS, rules=RULES,
            expected_start_state=_expected(), worlds_by_event=_worlds(SAVE_EVENTS),
            positions_of=fx.positions_of, route_config=fx.route_config(),
        )
    assert caught.value.reasons[0] == fr.FH_ROUTE_INVALID


def ro_partial_with_bad_transition(partial):
    """Wrap the route with a transition the engine never produced."""

    from dataclasses import replace

    actions = list(partial.actions)
    transition = replace(actions[1]["transition"], ok=False, errors=("FABRICATED",))
    actions[1] = {**actions[1], "transition": transition}
    return replace(partial, actions=tuple(actions))


def test_a_wrong_event_set_refuses():
    partial, _terminal = fx.build_canonical_route(start=_start(event=H1), events=SAVE_EVENTS)
    with pytest.raises(fr.FreeHitRouteError) as caught:
        fr.free_hit_route_from_canonical_route(
            partial, arm=fr.ARM_SAVE, expected_events=(H2, H3, H4), rules=RULES,
            expected_start_state=_expected(), worlds_by_event=_worlds(SAVE_EVENTS),
            positions_of=fx.positions_of, route_config=fx.route_config(),
        )
    assert caught.value.reasons[0] == fr.FH_ROUTE_INVALID


def test_an_unknown_arm_refuses():
    partial, _terminal = fx.build_canonical_route(start=_start(event=H1), events=SAVE_EVENTS)
    with pytest.raises(fr.FreeHitRouteError) as caught:
        fr.free_hit_route_from_canonical_route(
            partial, arm="SOMETHING_ELSE", expected_events=SAVE_EVENTS, rules=RULES,
            expected_start_state=_expected(), worlds_by_event=_worlds(SAVE_EVENTS),
            positions_of=fx.positions_of, route_config=fx.route_config(),
        )
    assert caught.value.reasons[0] == fr.FH_ROUTE_INVALID


# ---------------------------------------------------------------------------
# THE EVALUATOR CONSUMES THE ROUTES
# ---------------------------------------------------------------------------


def _permanent(*, bank=20, ft=2, event=H1):
    return fh.FreeHitPermanentState(
        event=int(event), owned_ids=tuple(sorted(SQUAD)),
        purchase_price_tenths={pid: fx.BASE_PURCHASE for pid in SQUAD},
        bank_tenths=int(bank), free_transfers=int(ft), event_start_free_transfers=int(ft),
        positions={pid: fx.POSITION[pid] for pid in SQUAD},
        clubs={pid: fx.CLUB[pid] for pid in SQUAD},
    )


def test_the_evaluator_uses_the_two_canonical_routes():
    """PLAY and SAVE are both route-derived; SAVE is NOT an H1-only baseline."""

    permanent = _permanent()
    save_route, _p, _t = _save_route(transfers={H1: ((12, 22),)}, prices={H1: {22: 40}})
    play_route, _p2, _t2 = _play_route()
    arms = fh.four_gw_arm_values(
        type("R", (), {"play_route": play_route, "save_route": save_route,
                       "planning_event": H1})(),
        h1_temporary_value=20.0,
    )
    assert arms["save_route_value"] == pytest.approx(save_route.value())
    assert arms["play_tail_value"] == pytest.approx(play_route.value())
    assert arms["four_gw_play_value"] == pytest.approx(20.0 + play_route.value())
    assert arms["four_gw_save_value"] == pytest.approx(save_route.value())


def test_the_play_h2_start_state_is_the_restored_permanent_state():
    """The two arms branch from the same H1 state and then evolve differently."""

    permanent = _permanent(bank=20, ft=2)
    from fpl_brain import chip_decision as cd

    binding = cd.ChipHorizonBinding(
        planning_event=H1, horizon_events=cd.canonical_chip_horizon(H1),
        certification_identity="sha256:" + "c" * 64, data_snapshot_sha256="sha256:" + "d" * 64,
    )
    request = fh.FreeHitRequest(
        permanent=permanent, horizon_binding=binding, h1_worlds=None, world_identity=None,
        positions={}, clubs={}, market_price_tenths={}, pool_binding=None, rules=RULES,
    )
    play_h2 = fh.play_h2_start_state(request)
    save_h1 = fh.save_h1_start_state(request)
    assert play_h2["squad_ids"] == save_h1["squad_ids"] == list(SQUAD)
    assert play_h2["bank_tenths"] == save_h1["bank_tenths"] == 20
    # The DIVERGENCE: preserved vs available.
    assert play_h2["free_transfers"] == fh.post_free_hit_ft_state(
        RULES, event_start_free_transfers=2
    )
    assert save_h1["free_transfers"] == 2


def test_building_a_play_route_from_the_save_state_refuses():
    """A PLAY tail that starts H2 from the wrong FT bank is refused."""

    with pytest.raises(fr.FreeHitRouteError) as caught:
        _play_route(expected=_expected(event=H2, bank=7, ft=99))
    assert caught.value.reasons[0] == fr.FH_ROUTE_START_STATE_MISMATCH
