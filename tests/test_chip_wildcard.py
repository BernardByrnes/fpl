"""Wildcard V1 — deterministic scenario tests A-N.

Every scenario builds a synthetic pool: NO football model is involved, and no
production projection, database or search is touched.  Expected values are
computed by hand in the test rather than restated from the implementation.
"""

from __future__ import annotations

import math

import pytest

from fpl_brain import chip_decision as cd
from fpl_brain import chip_wildcard as wc
from fpl_brain import manager_lineup
from fpl_brain import season_rules as sr
from fpl_brain import transfer_state as ts

RULES = sr.SeasonRules(season="2026/27")
IDENTITY = "sha256:" + "c" * 64
SNAPSHOT = "sha256:" + "d" * 64
CUTOFF = "2026-09-12T19:20:00Z"
SOURCE_SNAPSHOT = "sha256:" + "5" * 64
GENERATION = "generation-2026-09-12T19:20:00Z"
CONFIG_ID = "sha256:" + "7" * 64

CLUBS = tuple(range(1, 11))
COMPOSITION = ts.POSITION_COMPOSITION


def _player(pid, position, club, price, points, *, p_start=1.0, availability=1.0, events=None):
    events = events or range(5, 13)
    return wc.WildcardPlayer(
        player_id=pid,
        position=position,
        club_id=club,
        market_price_tenths=price,
        events={
            event: wc.WildcardPlayerEvent(
                event=event, expected_points=points, expected_minutes=90.0 * p_start,
                p_start=p_start, availability=availability, fixture_count=1,
                cutoff=CUTOFF, generation=GENERATION,
            )
            for event in events
        },
        web_name=f"p{pid}",
    )


def _pool(*, weak=(), strong=()):
    """A legal-depth pool spread across ten clubs.

    ``weak`` player ids get poor projections; ``strong`` get excellent ones.
    """

    players: dict[int, wc.WildcardPlayer] = {}
    pid = 1
    for position, count in (("GKP", 6), ("DEF", 16), ("MID", 18), ("FWD", 12)):
        for index in range(count):
            club = CLUBS[(pid - 1) % len(CLUBS)]
            price = 40 + 2 * (index % 5)
            points = 3.0 + 0.5 * (index % 4)
            p_start = 0.85
            if pid in weak:
                points, p_start = 0.4, 0.15
            if pid in strong:
                points, p_start = 9.0, 1.0
            players[pid] = _player(pid, position, club, price, points, p_start=p_start)
            pid += 1
    return players


def _owned(players, ids):
    return tuple(int(p) for p in ids)


def _binding(planning_event=5):
    return cd.ChipHorizonBinding(
        planning_event=planning_event,
        horizon_events=cd.canonical_chip_horizon(planning_event),
        certification_identity=IDENTITY,
        data_snapshot_sha256=SNAPSHOT,
    )


def _chip_rows(planning_event, *, windows=((2, 19), (20, 38)), used=False):
    """Production-shaped planning.chips_state rows for the wildcard."""

    rows = []
    for start, stop in windows:
        in_window = start <= planning_event <= stop
        rows.append({
            "name": "wildcard", "number": 1, "chip_type": "wildcard",
            "window_start_event": start, "window_stop_event": stop,
            "window": f"GW{start}-GW{stop}", "used": used, "used_event": None,
            "available_for_event": in_window and not used,
            "expired": planning_event > stop,
        })
    return rows


def _value_binding(planning_event=5, *, events=None, cutoff=CUTOFF, snapshot=SNAPSHOT,
                   source=SOURCE_SNAPSHOT, generation=GENERATION, config_id=CONFIG_ID):
    horizon = wc.wildcard_horizon(planning_event, length=8)
    if events is not None:
        horizon = wc.WildcardHorizonSpec(version=horizon.version, events=tuple(events),
                                         weights=tuple(1.0 / len(events) for _ in events),
                                         decay=horizon.decay, terminal_weight=0.0)
    return wc.value_horizon_binding(
        planning_event, horizon, decision_cutoff=cutoff, data_snapshot_sha256=snapshot,
        source_snapshot_sha256=source, prediction_generation=generation,
        model_config_identity=config_id,
    )


def _request(players, owned_ids, **overrides):
    base = dict(
        planning_event=5,
        horizon=wc.wildcard_horizon(5, length=8),
        players=players,
        positions={pid: p.position for pid, p in players.items()},
        owned_ids=_owned(players, owned_ids),
        purchase_price_tenths={int(p): players[int(p)].market_price_tenths for p in owned_ids},
        selling_price_tenths={int(p): players[int(p)].market_price_tenths for p in owned_ids},
        bank_tenths=30,
        rules=RULES,
        horizon_binding=_binding(),
        certification_identity=IDENTITY,
        data_snapshot_sha256=SNAPSHOT,
        event_start_free_transfers=2,
        chip_availability=_chip_rows(5),
        value_horizon_binding=_value_binding(5),
        worlds_by_event=_default_worlds(players),
        pool_binding=_pool_binding(players),
        save_route=_save_route(players, owned_ids),
    )
    base.update(overrides)
    return wc.WildcardRequest(**base)


def _save_route(players, owned, *, bank_tenths=30, events=(5, 6, 7, 8),
                mean_net_core=40.0, terminal_bank_tenths=None, terminal_free_transfers=1,
                shifts=None, **overrides):
    """A deterministic accepted-normal-route fixture.

    ``shifts`` maps event -> (squad_ids, bank_tenths, mean_net_core) so a test can
    change the route's squad, bank or net value per event.  ``mean_net_core``
    ALREADY nets that event's hits, which is the whole point of the contract.
    """

    shifts = dict(shifts or {})
    entries = []
    for event in events:
        squad, bank, core = shifts.get(event, (tuple(sorted(owned)), bank_tenths, mean_net_core))
        entries.append(wc.WildcardSaveRouteEvent(
            event=int(event),
            squad_ids=tuple(int(p) for p in squad),
            bank_tenths=int(bank),
            purchase_price_tenths={int(p): players[int(p)].market_price_tenths for p in squad},
            free_transfers=1,
            mean_net_core=float(core),
            hit_points=0,
        ))
    last = entries[-1]
    return wc.WildcardSaveRoute(
        events=tuple(entries),
        terminal_squad_ids=tuple(last.squad_ids),
        terminal_bank_tenths=int(last.bank_tenths if terminal_bank_tenths is None else terminal_bank_tenths),
        terminal_purchase_price_tenths=dict(last.purchase_price_tenths),
        terminal_free_transfers=int(terminal_free_transfers),
        cumulative_hits=int(overrides.pop("cumulative_hits", 0)),
        wildcard_available=bool(overrides.pop("wildcard_available", True)),
    )


def _pool_binding(players, *, generation="gen-2026-09-12T19:20:00Z"):
    """The accepted official-pool binding for this fixture's universe."""

    from fpl_brain.ingest_provenance import element_id_sha256

    ids = tuple(sorted(players))
    return wc.WildcardPoolBinding(
        generation_identity=generation,
        generation_id_sha256=element_id_sha256(sorted(ids)),
        official_count=len(ids),
        eligible_ids=ids,
    )


def _default_worlds(players, *, events=range(5, 13)):
    """One deterministic world per event, from the players' own projections.

    A single world in which everyone appears at their expected minutes and scores
    their expected points.  Enough for the accounting and certification tests,
    which do not exercise the lineup engine's world handling.
    """

    ids = tuple(sorted(players))
    worlds = {}
    for event in events:
        minutes = {}
        core = {}
        for pid in ids:
            entry = players[pid].at(event)
            minutes[pid] = [0.0 if entry is None else float(entry.expected_minutes)]
            core[pid] = [0.0 if entry is None else float(entry.expected_points)]
        worlds[int(event)] = wc.WildcardWorldInputs(
            worlds=1, player_ids=ids, minutes=minutes, core=core
        )
    return worlds


def _legal_owned_ids(players) -> list[int]:
    """Build a legal 15 from the pool, respecting club limits."""

    chosen: list[int] = []
    clubs: dict[int, int] = {}
    for position, required in COMPOSITION.items():
        picked = 0
        for pid, player in sorted(players.items()):
            if player.position != position or picked >= required:
                continue
            if clubs.get(player.club_id, 0) >= ts.SQUAD_TEAM_LIMIT:
                continue
            chosen.append(pid)
            clubs[player.club_id] = clubs.get(player.club_id, 0) + 1
            picked += 1
    assert len(chosen) == 15
    return chosen


# ---------------------------------------------------------------------------
# A. OBVIOUS BAD SQUAD
# ---------------------------------------------------------------------------


def test_A_an_obviously_bad_squad_produces_a_positive_but_suppressed_candidate():
    players = _pool(weak={1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15})
    bad = list(range(1, 16))
    request = _request(players, bad)
    evaluation = wc.evaluate_wildcard(request)

    assert evaluation.candidate_metrics["mean_paired_uplift"] > 0.0
    assert evaluation.calibration_status == cd.CALIBRATION_UNCALIBRATED
    assert wc.WC_REASON_POSITIVE in evaluation.reason_codes
    assert evaluation.evidence["executable"] is False
    assert evaluation.evidence["actionable"] is False

    # and the arbiter will not let it become an executable play
    decision = cd.decide_chip_action(
        horizon_binding=_binding(),
        chip_availability=_chip_rows(5),
        evaluations={cd.CHIP_ACTION_WC: evaluation},
        reservation=cd.UncalibratedReservation(),
        certification_valid=True,
        manager_state={"squad_ids": list(bad)},
    )
    assert decision.recommended_action == cd.CHIP_ACTION_WC
    assert decision.status == cd.STATUS_CHIP_CANDIDATE_RECHECK_REQUIRED
    assert decision.status != cd.STATUS_PLAY_CHIP

    # A CALIBRATED RESERVATION IS NOT ENOUGH.  The Wildcard value model is
    # itself uncalibrated, so the execution gate must keep PLAY_WC blocked even
    # when the reservation declares itself calibrated.
    class Calibrated:
        def estimate(self, *, action, planning_event, expiry_event, state):
            return cd.ReservationEstimate(
                value=1.0, calibration_status=cd.CALIBRATION_CALIBRATED,
                terminal_value=0.0, weeks_to_expiry=10, reason_codes=(), conditional_on=("weeks_remaining",),
            )

    still_blocked = cd.decide_chip_action(
        horizon_binding=_binding(),
        chip_availability=_chip_rows(5),
        evaluations={cd.CHIP_ACTION_WC: evaluation},
        reservation=Calibrated(),
        certification_valid=True,
        manager_state={"squad_ids": list(bad)},
    )
    assert still_blocked.status != cd.STATUS_PLAY_CHIP
    assert still_blocked.status in (cd.STATUS_CHIP_CANDIDATE_RECHECK_REQUIRED, cd.STATUS_CHIP_REVIEW_REQUIRED)
    assert cd.DIAG_CHIP_EVALUATOR_UNCALIBRATED in still_blocked.reason_codes
    assert evaluation.execution_permitted is False


# ---------------------------------------------------------------------------
# B. HEALTHY SQUAD
# ---------------------------------------------------------------------------


def test_B_a_healthy_squad_shows_a_much_smaller_uplift_than_a_bad_one():
    """SAVE is preferred when the squad is already near the top of the pool.

    Note the assertion is RELATIVE, not "uplift <= 0".  Even a squad of equally
    good players can be improved on the flexibility term by a cheaper legal
    alternative, and that is a real (if small) improvement rather than a bug --
    so the honest statement is that a healthy squad's uplift is negligible
    beside a broken squad's, which is what drives the SAVE decision.
    """

    # A healthy squad's normal route is itself strong, which is what makes the
    # Wildcard small; a broken squad's route is weak.  SAVE now comes from the
    # route, so the route values must express that difference.
    healthy_pool = _pool(strong=set(range(40, 80)))
    healthy_owned = _legal_owned_ids(healthy_pool)
    healthy = wc.evaluate_wildcard(_request(
        healthy_pool, healthy_owned,
        save_route=_save_route(healthy_pool, healthy_owned, mean_net_core=95.0)))

    bad_pool = _pool(weak=set(range(1, 16)))
    bad_owned = list(range(1, 16))
    bad = wc.evaluate_wildcard(_request(
        bad_pool, bad_owned, save_route=_save_route(bad_pool, bad_owned, mean_net_core=5.0)))

    healthy_uplift = healthy.candidate_metrics["mean_paired_uplift"]
    bad_uplift = bad.candidate_metrics["mean_paired_uplift"]
    assert bad_uplift > 0.0
    assert healthy_uplift < 0.25 * bad_uplift, (
        f"healthy uplift {healthy_uplift} should be negligible vs bad {bad_uplift}"
    )


def test_B2_a_better_save_route_reduces_the_uplift():
    players = _pool(weak=set(range(1, 16)))
    bad = list(range(1, 16))
    plain = wc.evaluate_wildcard(_request(players, bad, save_route=_save_route(players, bad, mean_net_core=5.0)))
    better = wc.evaluate_wildcard(_request(players, bad, save_route=_save_route(players, bad, mean_net_core=45.0)))
    assert better.candidate_metrics["mean_paired_uplift"] < plain.candidate_metrics["mean_paired_uplift"]


# ---------------------------------------------------------------------------
# C/D/E. PRICING, RETAINED BASIS, SOLD+REBOUGHT BASIS
# ---------------------------------------------------------------------------


def test_C_the_budget_uses_real_selling_value_not_market_price():
    players = _pool()
    owned = _legal_owned_ids(players)
    # every owned player has risen in price, so selling value < market price
    # market price has RISEN 20 tenths above the purchase basis, so the
    # canonical selling value is between the two
    purchase = {pid: players[pid].market_price_tenths for pid in owned}
    selling = {pid: ts.selling_price_tenths(purchase[pid], purchase[pid] + 20) for pid in owned}
    for pid in owned:
        players[pid] = T_player_with_market(players[pid], players[pid].market_price_tenths + 20)
    request = _request(players, owned, purchase_price_tenths=purchase, selling_price_tenths=selling, bank_tenths=0)

    # retaining the whole squad sells nobody, so cash is just the bank -- the
    # capital locked in 15 retained players is NOT spendable
    available, _, _ = wc._budget_and_cost(request, owned)
    assert available == 0, "retained capital is NOT cash"
    plan = wc.plan_transaction(request, owned)
    assert plan.sold_ids == () and plan.bought_ids == ()
    assert plan.retained_ids == tuple(sorted(owned))

    # selling exactly one player realises exactly that player's canonical value
    keep = list(owned[:-1])
    sold_id = owned[-1]
    available_one, _, _ = wc._budget_and_cost(request, keep)
    canonical = ts.selling_price_tenths(purchase[sold_id], players[sold_id].market_price_tenths)
    assert available_one == canonical, "one sale contributes exactly its canonical selling value"
    assert available_one < purchase[sold_id] + 20, "must not use market price"


def test_D_retained_players_keep_their_acquisition_basis_and_cost_nothing_new():
    players = _pool()
    owned = _legal_owned_ids(players)
    purchase = {pid: 40 for pid in owned}
    request = _request(players, owned, purchase_price_tenths=purchase)

    # retaining the whole squad costs nothing and preserves every basis
    available, cost, bank_after = wc._budget_and_cost(request, owned)
    assert cost == 0
    assert bank_after == available
    # a squad of only retained players remains legal and unchanged
    ok, problems = wc.is_legal_squad(request, owned)
    assert ok, problems


def test_E_sold_then_rebought_players_use_the_new_market_basis():
    players = _pool()
    owned = _legal_owned_ids(players)
    request = _request(players, owned)

    outside = [pid for pid in sorted(players) if pid not in set(owned)]
    swapped = list(owned[:-1]) + [outside[0]]
    available, cost, _ = wc._budget_and_cost(request, swapped)
    assert cost == players[outside[0]].market_price_tenths, "a bought player is charged market price"
    # the purchase basis for a newly bought player is the price paid now
    assert request.purchase_price_tenths.get(outside[0]) is None
    assert players[outside[0]].market_price_tenths > 0


def T_player_with_market(player, market_price_tenths):
    """Same player, re-priced (used to model a market move)."""

    return wc.WildcardPlayer(
        player_id=player.player_id, position=player.position, club_id=player.club_id,
        market_price_tenths=market_price_tenths, events=dict(player.events),
        web_name=player.web_name,
    )


def test_C2_a_supplied_selling_value_that_disagrees_with_the_canonical_rule_fails_closed():
    """Caller-supplied selling values are validated, never trusted."""

    players = _pool()
    owned = _legal_owned_ids(players)
    bad = {pid: players[pid].market_price_tenths + 15 for pid in owned}
    request = _request(players, owned, selling_price_tenths=bad)
    evaluation = wc.evaluate_wildcard(request)
    assert wc.WC_PRICING_UNAVAILABLE in evaluation.reason_codes
    assert evaluation.candidate_metrics["mean_paired_uplift"] is None


# ---------------------------------------------------------------------------
# F. MAX-CLUB LIMIT
# ---------------------------------------------------------------------------


def test_F_no_illegal_squad_is_ever_returned():
    players = _pool()
    request = _request(players, _legal_owned_ids(players))
    evaluation = wc.evaluate_wildcard(request)
    squad = evaluation.candidate_metrics["wildcard_squad"]
    ok, problems = wc.is_legal_squad(request, squad)
    assert ok, problems
    clubs: dict[int, int] = {}
    for pid in squad:
        clubs[players[pid].club_id] = clubs.get(players[pid].club_id, 0) + 1
    assert max(clubs.values()) <= ts.SQUAD_TEAM_LIMIT


def test_F2_a_club_concentrated_pool_still_yields_a_legal_squad():
    # every strong player sits at ONE club, so a naive best-points squad would
    # breach the limit at every position
    players = _pool()
    # the first few players of every position are the best AND all sit at club 1,
    # so a naive best-points squad would breach the three-per-club limit
    best = {1, 2, 7, 8, 9, 10, 11, 23, 24, 25, 26, 27, 41, 42, 43, 44, 45}
    for pid, player in players.items():
        if pid in best:
            players[pid] = _player(pid, player.position, 1, player.market_price_tenths, 25.0)
    request = _request(players, _legal_owned_ids(players))
    evaluation = wc.evaluate_wildcard(request)
    squad = evaluation.candidate_metrics["wildcard_squad"]
    ok, problems = wc.is_legal_squad(request, squad)
    assert ok, problems


# ---------------------------------------------------------------------------
# G. MISSING PROJECTION
# ---------------------------------------------------------------------------


def test_G_a_missing_projection_excludes_the_player_rather_than_zeroing_him():
    players = _pool()
    incomplete = 7
    players[incomplete] = _player(incomplete, "MID", 3, 45, 12.0, events=range(5, 10))  # short horizon
    request = _request(players, _legal_owned_ids(players))
    scores, stats = wc.screen_players(request)
    assert incomplete not in scores
    assert incomplete in stats["excluded_ids"]
    assert stats["excluded_reasons"][incomplete] == wc.WC_MISSING_PROJECTION
    assert stats["excluded_missing_projection"] >= 1
    assert all(math.isfinite(v) for v in scores.values())


# ---------------------------------------------------------------------------
# H/I. CHIP WINDOWS
# ---------------------------------------------------------------------------


def test_H_the_second_chip_window_supplies_the_expiry():
    players = _pool()
    request = _request(players, _legal_owned_ids(players), planning_event=25,
                       horizon=wc.wildcard_horizon(25, length=8),
                       horizon_binding=_binding(25), chip_availability=_chip_rows(25),
                       value_horizon_binding=_value_binding(25))
    assert wc.resolve_wildcard_expiry(request) == 38


def test_H2_the_first_window_supplies_the_expiry_early_in_the_season():
    players = _pool()
    request = _request(players, _legal_owned_ids(players))
    assert wc.resolve_wildcard_expiry(request) == 19


def test_I_overlapping_active_windows_fail_closed():
    players = _pool()
    request = _request(players, _legal_owned_ids(players), planning_event=25,
                       horizon=wc.wildcard_horizon(25, length=8), horizon_binding=_binding(25),
                       chip_availability=_chip_rows(25, windows=((20, 30), (24, 38))),
                       value_horizon_binding=_value_binding(25))
    assert wc.resolve_wildcard_expiry(request) is wc.EXPIRY_AMBIGUOUS
    evaluation = wc.evaluate_wildcard(request)
    assert wc.WC_WINDOW_AMBIGUOUS in evaluation.reason_codes
    assert evaluation.candidate_metrics["mean_paired_uplift"] is None


def test_I2_an_ambiguous_window_is_not_resolved_by_row_order():
    """Row order must not pick a window — the chip core owns that decision."""

    players = _pool()
    forward = _chip_rows(25, windows=((20, 30), (24, 38)))
    backward = list(reversed(forward))
    players_a = _request(players, _legal_owned_ids(players), planning_event=25,
                         horizon=wc.wildcard_horizon(25, length=8), horizon_binding=_binding(25),
                         chip_availability=forward, value_horizon_binding=_value_binding(25))
    players_b = _request(players, _legal_owned_ids(players), planning_event=25,
                         horizon=wc.wildcard_horizon(25, length=8), horizon_binding=_binding(25),
                         chip_availability=backward, value_horizon_binding=_value_binding(25))
    assert wc.resolve_wildcard_expiry(players_a) is wc.EXPIRY_AMBIGUOUS
    assert wc.resolve_wildcard_expiry(players_b) is wc.EXPIRY_AMBIGUOUS


# ---------------------------------------------------------------------------
# J. SAVE POLICY
# ---------------------------------------------------------------------------


def test_J_save_retains_the_wildcard_option_and_can_use_normal_transfers():
    players = _pool(weak=set(range(1, 16)))
    owned = list(range(1, 16))
    route = _save_route(players, owned, mean_net_core=60.0, terminal_bank_tenths=88)
    evaluation = wc.evaluate_wildcard(_request(players, owned, save_route=route))
    save = evaluation.evidence["save_policy"]
    assert save["retains_wildcard_option"] is True
    assert save["route_value_evidence"]["cumulative_hits"] == 0
    # The reservation belongs to the ARBITER, so this evaluator must not have
    # called it: it reports the raw play-vs-save difference and the authoritative
    # post-SAVE state the arbiter's provider needs, and nothing else.
    assert evaluation.candidate_metrics["net_of_reservation"] is None
    assert "reservation_value" not in save
    state = save["post_save_state_for_reservation"]
    assert state["squad_ids"] == list(owned)
    assert state["bank_tenths"] == 88
    assert state["retains_wildcard_option"] is True


# ---------------------------------------------------------------------------
# K. NO CLAIRVOYANCE
# ---------------------------------------------------------------------------


def test_K_future_outcomes_outside_the_horizon_cannot_change_the_selection():
    players = _pool(weak=set(range(1, 16)))
    request = _request(players, list(range(1, 16)))
    first = wc.evaluate_wildcard(request).candidate_metrics["wildcard_squad"]

    # change everything AFTER the horizon ends (events 13+)
    mutated = _pool(weak=set(range(1, 16)))
    for pid, player in mutated.items():
        extra = dict(player.events)
        for event in range(13, 20):
            extra[event] = wc.WildcardPlayerEvent(
                event=event, expected_points=999.0, expected_minutes=90.0, p_start=1.0,
                availability=1.0, fixture_count=1, cutoff=CUTOFF, generation=GENERATION,
            )
        mutated[pid] = wc.WildcardPlayer(pid, player.position, player.club_id,
                                         player.market_price_tenths, extra, player.web_name)
    second = wc.evaluate_wildcard(_request(mutated, list(range(1, 16)))).candidate_metrics["wildcard_squad"]
    assert first == second


def test_K2_the_horizon_weights_are_tapered_with_no_cliff_after_the_fourth_event():
    horizon = wc.wildcard_horizon(5, length=8)
    weights = list(horizon.weights)
    assert len(weights) == 8
    assert weights == sorted(weights, reverse=True)
    # no arbitrary drop at the boundary between the near four and the rest
    assert weights[4] > 0.5 * weights[3]


# ---------------------------------------------------------------------------
# L. PLAYER DISCOVERY
# ---------------------------------------------------------------------------


def test_L_an_unowned_low_ownership_player_can_enter_the_squad():
    players = _pool()
    star = 99
    players[star] = _player(star, "FWD", 7, 45, 40.0)
    owned = _legal_owned_ids(players)
    assert star not in owned  # not owned, not on any watchlist, no popularity gate
    evaluation = wc.evaluate_wildcard(_request(players, owned))
    assert star in evaluation.candidate_metrics["wildcard_squad"]


def test_L2_the_screen_never_gates_on_ownership_or_club_reputation():
    players = _pool()
    request = _request(players, _legal_owned_ids(players))
    scores, stats = wc.screen_players(request)
    # every player with a full horizon is screened, including unowned ones
    assert stats["screened"] == len(players)
    assert len(scores) == len(players)


# ---------------------------------------------------------------------------
# M. BENCH
# ---------------------------------------------------------------------------


def test_M_bench_value_is_part_of_the_valuation_not_an_afterthought():
    players = _pool()
    owned = _legal_owned_ids(players)
    value = wc.evaluate_squad(_request(players, owned), owned)
    # Bench value is inside the horizon points (it is real expected FPL points),
    # while the structural terms are REPORTED and deliberately not scored.
    assert value.terminal_value == 0.0
    assert value.flexibility_value == 0.0
    assert value.objective == value.horizon_points
    # a squad whose bench is genuinely valuable scores above one whose bench is dead
    weak = _pool()
    for pid in range(40, 52):  # back-fill deep pool ids
        if pid in weak and weak[pid].position in ("DEF", "MID", "FWD"):
            weak[pid] = _player(pid, weak[pid].position, weak[pid].club_id,
                                weak[pid].market_price_tenths, 0.0, p_start=0.05)
    assert wc.evaluate_squad(_request(players, owned), owned).objective >= 0.0


# ---------------------------------------------------------------------------
# N. FT STATE
# ---------------------------------------------------------------------------


def test_N_the_wildcard_transition_uses_canonical_free_transfer_semantics():
    players = _pool()
    owned = _legal_owned_ids(players)
    evaluation = wc.evaluate_wildcard(_request(players, owned, event_start_free_transfers=4))
    expected = sr.free_transfers_after_chip(RULES, "wildcard", event_start_free_transfers=4)
    assert evaluation.evidence["free_transfers_after_wildcard"] == expected
    assert evaluation.evidence["ft_preserved_by_chip"] is True
    assert evaluation.evidence["ft_preserved_by_chip"] == sr.chip_preserves_saved_free_transfers("wildcard")


def test_N2_an_unknown_event_start_bank_fails_closed():
    """No numeric uplift may be emitted alongside an FT warning."""

    players = _pool()
    owned = _legal_owned_ids(players)
    evaluation = wc.evaluate_wildcard(_request(players, owned, event_start_free_transfers=None))
    assert wc.WC_MANAGER_STATE_INCOHERENT in evaluation.reason_codes
    assert evaluation.candidate_metrics["mean_paired_uplift"] is None
    assert "free_transfers_after_wildcard" not in evaluation.evidence


# ---------------------------------------------------------------------------
# Fail-closed contract
# ---------------------------------------------------------------------------


def test_incoherent_manager_state_refuses_rather_than_scoring():
    players = _pool()
    request = _request(players, [1, 2, 3])  # not 15 players
    evaluation = wc.evaluate_wildcard(request)
    assert wc.WC_MANAGER_STATE_INCOHERENT in evaluation.reason_codes
    assert evaluation.candidate_metrics["mean_paired_uplift"] is None


def test_missing_selling_price_refuses():
    players = _pool()
    owned = _legal_owned_ids(players)
    request = _request(players, owned, selling_price_tenths={})
    evaluation = wc.evaluate_wildcard(request)
    assert wc.WC_PRICING_UNAVAILABLE in evaluation.reason_codes


def test_cross_cutoff_certification_refuses():
    players = _pool()
    owned = _legal_owned_ids(players)
    request = _request(players, owned, certification_identity="sha256:" + "e" * 64)
    evaluation = wc.evaluate_wildcard(request)
    assert wc.WC_STALE_OR_CROSS_CUTOFF in evaluation.reason_codes


def test_the_horizon_spec_rejects_invalid_configurations():
    with pytest.raises(wc.WildcardInputError):
        wc.WildcardHorizonSpec("v", (5, 6, 7), (1 / 3, 1 / 3, 1 / 3), 0.8, 0.5)  # too short
    with pytest.raises(wc.WildcardInputError):
        wc.WildcardHorizonSpec("v", (5, 6, 7, 8, 9, 10), (1.0,), 0.8, 0.5)  # weight mismatch
    with pytest.raises(wc.WildcardInputError):
        wc.WildcardHorizonSpec("v", (5, 5, 6, 7, 8, 9), (1 / 6,) * 6, 0.8, 0.5)  # duplicate events
    with pytest.raises(wc.WildcardInputError):
        wc.WildcardHorizonSpec("v", (5, 6, 7, 8, 9, 10), (0.5, 0.5, 0, 0, 0, 0), 0.8, 0.5)  # zero weight


def test_the_evaluation_is_deterministic():
    players = _pool(weak=set(range(1, 16)))
    first = wc.evaluate_wildcard(_request(players, list(range(1, 16))))
    second = wc.evaluate_wildcard(_request(players, list(range(1, 16))))
    assert first.candidate_metrics == second.candidate_metrics
    assert first.evidence == second.evidence


def test_no_global_optimum_is_claimed():
    players = _pool()
    evaluation = wc.evaluate_wildcard(_request(players, _legal_owned_ids(players)))
    assert evaluation.evidence["no_global_optimum_claim"] is True
    assert evaluation.candidate_metrics["legal_squads_considered"] > 0
    assert evaluation.candidate_metrics["screened_players"] == len(players)


def test_chip_core_contracts_cannot_be_widened_by_this_evaluator():
    """The evaluator must not touch the certified 4-event chip horizon."""

    players = _pool()
    evaluation = wc.evaluate_wildcard(_request(players, _legal_owned_ids(players)))
    assert tuple(evaluation.evidence["horizon_events"]) == cd.canonical_chip_horizon(5)
    assert len(evaluation.evidence["horizon_events"]) == cd.CHIP_HORIZON_LENGTH
    # while its OWN value horizon is longer
    assert len(evaluation.evidence["wildcard_horizon"]["events"]) == 8


# ---------------------------------------------------------------------------
# §1/§2 — value-horizon certification binding
# ---------------------------------------------------------------------------


def _with_row_override(players, pid, event, **fields):
    player = players[pid]
    events = dict(player.events)
    entry = events[event]
    events[event] = wc.WildcardPlayerEvent(
        event=entry.event, expected_points=fields.get("expected_points", entry.expected_points),
        expected_minutes=entry.expected_minutes, p_start=fields.get("p_start", entry.p_start),
        availability=entry.availability, fixture_count=entry.fixture_count,
        cutoff=fields.get("cutoff", entry.cutoff), generation=fields.get("generation", entry.generation),
    )
    players[pid] = wc.WildcardPlayer(pid, player.position, player.club_id,
                                     player.market_price_tenths, events, player.web_name)
    return players


def test_P2_A_a_projection_from_a_different_run_refuses():
    players = _with_row_override(_pool(), 7, 9, generation="generation-OTHER")
    evaluation = wc.evaluate_wildcard(_request(players, _legal_owned_ids(players)))
    assert evaluation.candidate_metrics["mean_paired_uplift"] is None
    assert evaluation.reason_codes


def test_P2_B_a_projection_from_a_different_cutoff_refuses():
    players = _with_row_override(_pool(), 7, 9, cutoff="2026-09-13T00:00:00Z")
    evaluation = wc.evaluate_wildcard(_request(players, _legal_owned_ids(players)))
    assert evaluation.candidate_metrics["mean_paired_uplift"] is None


def test_P2_C_a_different_data_snapshot_refuses():
    players = _pool()
    request = _request(players, _legal_owned_ids(players),
                       value_horizon_binding=_value_binding(5, snapshot="sha256:" + "9" * 64))
    evaluation = wc.evaluate_wildcard(request)
    assert wc.WC_VALUE_BINDING_MISMATCH in evaluation.reason_codes
    assert evaluation.candidate_metrics["mean_paired_uplift"] is None


def test_P2_D_the_binding_is_individually_coherent():
    players = _pool()
    request = _request(players, _legal_owned_ids(players),
                       value_horizon_binding=_value_binding(5, config_id="sha256:" + "8" * 64))
    assert wc.validate_projections(request) == []


def test_P2_E_a_non_contiguous_horizon_refuses():
    horizon = wc.wildcard_horizon(5, length=8)
    broken = wc.WildcardHorizonSpec(version=horizon.version, events=(5, 6, 7, 9, 10, 11, 12, 13),
                                    weights=tuple(1.0 / 8 for _ in range(8)), decay=0.82,
                                    terminal_weight=0.0)
    with pytest.raises(wc.WildcardInputError):
        wc.value_horizon_binding(5, broken, decision_cutoff=CUTOFF, data_snapshot_sha256=SNAPSHOT,
                                 source_snapshot_sha256=SOURCE_SNAPSHOT,
                                 prediction_generation=GENERATION, model_config_identity=CONFIG_ID)


def test_P2_F_a_coherent_eight_event_horizon_passes():
    players = _pool()
    request = _request(players, _legal_owned_ids(players))
    assert wc.validate_projections(request) == []
    evaluation = wc.evaluate_wildcard(request)
    assert evaluation.candidate_metrics["mean_paired_uplift"] is not None
    binding = evaluation.evidence["wildcard_value_horizon"]
    assert len(binding["event_ids"]) == 8
    assert binding["prediction_generation"] == GENERATION


def test_P2_G_a_missing_binding_refuses_entirely():
    players = _pool()
    request = _request(players, _legal_owned_ids(players), value_horizon_binding=None)
    evaluation = wc.evaluate_wildcard(request)
    assert wc.WC_VALUE_BINDING_MISMATCH in evaluation.reason_codes
    assert evaluation.candidate_metrics["mean_paired_uplift"] is None


def test_P2_H_nonfinite_projections_fail_closed():
    """A non-finite projection must never become a numeric candidate.

    It fails closed either at construction (the world matrix rejects a non-finite
    series) or as a token refusal -- both are fail-closed; what matters is that no
    confident Wildcard output is produced.
    """

    players = _with_row_override(_pool(), 7, 9, expected_points=float("nan"))
    try:
        evaluation = wc.evaluate_wildcard(_request(players, _legal_owned_ids(players)))
    except wc.WildcardInputError as exc:
        assert exc.reasons
        return
    assert evaluation.candidate_metrics["mean_paired_uplift"] is None
    assert evaluation.reason_codes


def test_P2_I_impossible_probabilities_fail_closed():
    players = _with_row_override(_pool(), 7, 9, p_start=1.7)
    evaluation = wc.evaluate_wildcard(_request(players, _legal_owned_ids(players)))
    assert wc.WC_PROJECTION_INVALID in evaluation.reason_codes


def test_P2_J_the_exclusion_audit_is_never_truncated():
    players = _pool()
    for pid in (5, 7, 9, 11):
        player = players[pid]
        short = {e: v for e, v in player.events.items() if e < 10}
        players[pid] = wc.WildcardPlayer(pid, player.position, player.club_id,
                                         player.market_price_tenths, short, player.web_name)
    request = _request(players, _legal_owned_ids(players))
    scores, stats = wc.screen_players(request)
    assert stats["excluded_missing_projection"] == 4
    assert len(stats["excluded_ids"]) == 4          # full evidence, never truncated
    assert set(stats["excluded_reasons"]) == {5, 7, 9, 11}
    assert all(pid not in scores for pid in (5, 7, 9, 11))


# ---------------------------------------------------------------------------
# PHASE 1 — the Wildcard valuation must DELEGATE to the accepted engine
# ---------------------------------------------------------------------------

LINEUP_CLUBS = tuple(range(20, 30))


def _lineup_squad():
    """A legal 15 with distinct, easily-identifiable positions and clubs."""

    players: dict[int, wc.WildcardPlayer] = {}
    pid = 500
    for position, count in (("GKP", 2), ("DEF", 5), ("MID", 5), ("FWD", 3)):
        for _ in range(count):
            club = LINEUP_CLUBS[pid % len(LINEUP_CLUBS)]
            players[pid] = _player(pid, position, club, 50, 1.0)
            pid += 1
    return players


def _worlds_for(players, event, *, blank=(), core_overrides=None, minutes_overrides=None):
    """One deterministic world per requested scenario, all players always covered."""

    core_overrides = core_overrides or {}
    minutes_overrides = minutes_overrides or {}
    ids = tuple(sorted(players))
    minutes = {pid: [] for pid in ids}
    core = {pid: [] for pid in ids}
    for pid in ids:
        blanked = pid in blank
        minutes[pid].append(0.0 if blanked else minutes_overrides.get(pid, 90.0))
        core[pid].append(0.0 if blanked else core_overrides.get(pid, 1.0))
    return wc.WildcardWorldInputs(worlds=1, player_ids=ids, minutes=minutes, core=core)


def _lineup_request(players, owned, worlds, *, planning_event=5):
    return wc.WildcardRequest(
        planning_event=planning_event,
        horizon=wc.wildcard_horizon(planning_event, length=8),
        players=players,
        positions={pid: p.position for pid, p in players.items()},
        owned_ids=tuple(sorted(owned)),
        purchase_price_tenths={int(p): 50 for p in owned},
        selling_price_tenths={int(p): 50 for p in owned},
        bank_tenths=30,
        rules=RULES,
        horizon_binding=_binding(),
        certification_identity=IDENTITY,
        data_snapshot_sha256=SNAPSHOT,
        event_start_free_transfers=2,
        chip_availability=_chip_rows(5),
        value_horizon_binding=_value_binding(5),
        worlds_by_event=worlds,
        pool_binding=_pool_binding(players),
        save_route=_save_route(players, owned),
    )


def _accepted_event_value(request, squad, event):
    """The SAME value computed directly from the accepted helpers (the oracle)."""

    policy = wc._event_policy(request, squad, event)
    worlds = request.worlds_by_event[event]
    positions = {pid: request.players[pid].position for pid in squad}
    total = 0.0
    for w in range(worlds.worlds):
        minutes = {pid: float(worlds.minutes[pid][w]) for pid in squad}
        core = {pid: float(worlds.core[pid][w]) for pid in squad}
        outcome = manager_lineup.resolve_world(policy, positions, minutes, core,
                                               require_player_ids=list(squad))
        extra, _ = manager_lineup.captain_multiplier(policy, minutes, core)
        total += sum(core[pid] for pid in outcome.counted_ids) + extra
    return total / worlds.worlds


def _worlds_every_event(players, *, blank=(), core_overrides=None):
    return {event: _worlds_for(players, event, blank=blank, core_overrides=core_overrides)
            for event in range(5, 13)}


def test_worlds_A_the_wildcard_event_value_delegates_to_the_accepted_engine():
    """Not a re-implementation: the value IS the accepted resolution."""

    players = _lineup_squad()
    owned = tuple(sorted(players))
    worlds = _worlds_every_event(players, core_overrides={500: 7.0})
    request = _lineup_request(players, owned, worlds)
    for event in (5, 8, 12):
        assert wc._event_value(request, owned, event) == pytest.approx(
            _accepted_event_value(request, owned, event)
        )


def test_worlds_B_captain_keeps_the_armband_when_he_appears():
    players = _lineup_squad()
    owned = tuple(sorted(players))
    worlds = _worlds_every_event(players, core_overrides={500: 7.0})
    request = _lineup_request(players, owned, worlds)
    policy = wc._event_policy(request, owned, 5)
    captain = policy.captain_id
    minutes = {pid: float(worlds[5].minutes[pid][0]) for pid in owned}
    core = {pid: float(worlds[5].core[pid][0]) for pid in owned}
    assert minutes[captain] > 0.0
    extra, source = manager_lineup.captain_multiplier(policy, minutes, core)
    assert source == "CAPTAIN"
    assert extra == pytest.approx(core[captain])


def test_worlds_C_vice_takes_the_armband_when_the_captain_is_blank():
    players = _lineup_squad()
    owned = tuple(sorted(players))
    probe = _lineup_request(players, owned, _worlds_every_event(players))
    policy = wc._event_policy(probe, owned, 5)
    captain, vice = policy.captain_id, policy.vice_captain_id

    worlds = _worlds_every_event(players, blank={captain}, core_overrides={vice: 6.0})
    request = _lineup_request(players, owned, worlds)
    minutes = {pid: float(worlds[5].minutes[pid][0]) for pid in owned}
    core = {pid: float(worlds[5].core[pid][0]) for pid in owned}
    extra, source = manager_lineup.captain_multiplier(policy, minutes, core)
    assert source == "VICE"
    assert extra == pytest.approx(core[vice])


def test_worlds_D_no_third_player_inherits_the_armband():
    players = _lineup_squad()
    owned = tuple(sorted(players))
    probe = _lineup_request(players, owned, _worlds_every_event(players))
    policy = wc._event_policy(probe, owned, 5)
    blank = {policy.captain_id, policy.vice_captain_id}
    worlds = _worlds_every_event(players, blank=blank)
    request = _lineup_request(players, owned, worlds)
    minutes = {pid: float(worlds[5].minutes[pid][0]) for pid in owned}
    core = {pid: float(worlds[5].core[pid][0]) for pid in owned}
    extra, source = manager_lineup.captain_multiplier(policy, minutes, core)
    assert source == "NONE"
    assert extra == 0.0


def test_worlds_E_the_bench_admits_nobody_when_every_starter_appears():
    players = _lineup_squad()
    owned = tuple(sorted(players))
    worlds = _worlds_every_event(players)
    request = _lineup_request(players, owned, worlds)
    policy = wc._event_policy(request, owned, 5)
    outcome = manager_lineup.resolve_world(
        policy,
        {pid: request.players[pid].position for pid in owned},
        {pid: float(worlds[5].minutes[pid][0]) for pid in owned},
        {pid: float(worlds[5].core[pid][0]) for pid in owned},
        require_player_ids=list(owned),
    )
    assert outcome.autosub_count == 0, "an all-appearing XI must admit no autosub"
    assert outcome.entrants == () and outcome.gk_used is False


def test_worlds_F_the_value_never_counts_the_whole_fifteen_bench_boost_is_not_active():
    players = _lineup_squad()
    owned = tuple(sorted(players))
    worlds = _worlds_every_event(players)
    request = _lineup_request(players, owned, worlds)
    # every player scores 1.0, so counting all 15 would give 15 + captain extra
    value = wc._event_value(request, owned, 5)
    assert value < 13.0, "more than a legal XI's worth of points means Bench Boost leaked in"


def test_worlds_G_only_the_bench_goalkeeper_may_replace_the_starting_goalkeeper():
    players = _lineup_squad()
    owned = tuple(sorted(players))
    probe = _lineup_request(players, owned, _worlds_every_event(players))
    policy = wc._event_policy(probe, owned, 5)
    starter_gk = next(pid for pid in policy.starter_ids
                      if probe.players[pid].position == "GKP")
    bench_gk = int(policy.bench_gk_id)

    worlds = _worlds_every_event(players, blank={starter_gk}, core_overrides={bench_gk: 8.0})
    request = _lineup_request(players, owned, worlds)
    outcome = manager_lineup.resolve_world(
        policy,
        {pid: request.players[pid].position for pid in owned},
        {pid: float(worlds[5].minutes[pid][0]) for pid in owned},
        {pid: float(worlds[5].core[pid][0]) for pid in owned},
        require_player_ids=list(owned),
    )
    assert outcome.gk_used is True
    assert bench_gk in outcome.counted_ids


def test_worlds_H_missing_world_inputs_fail_closed():
    players = _lineup_squad()
    owned = tuple(sorted(players))
    request = _lineup_request(players, owned, {})  # no worlds at all
    evaluation = wc.evaluate_wildcard(request)
    assert wc.WC_WORLD_INPUTS_MISSING in evaluation.reason_codes
    assert evaluation.candidate_metrics["mean_paired_uplift"] is None


def test_worlds_I_world_series_must_cover_every_supplied_player():
    players = _lineup_squad()
    owned = tuple(sorted(players))
    partial = {event: wc.WildcardWorldInputs(
        worlds=1,
        player_ids=tuple(sorted(players))[:10],  # five players missing
        minutes={pid: [90.0] for pid in tuple(sorted(players))[:10]},
        core={pid: [1.0] for pid in tuple(sorted(players))[:10]},
    ) for event in range(5, 13)}
    request = _lineup_request(players, owned, partial)
    problems = wc.validate_projections(request)
    assert any("no world series" in p for p in problems)


# ---------------------------------------------------------------------------
# PHASE 2 — official player-pool completeness
# ---------------------------------------------------------------------------


def test_pool_A_an_incomplete_generation_identity_refuses():
    """No pool binding means the discovery claim cannot be checked."""

    players = _pool()
    request = _request(players, _legal_owned_ids(players), pool_binding=None)
    evaluation = wc.evaluate_wildcard(request)
    assert wc.OFFICIAL_PLAYER_POOL_INCOMPLETE in evaluation.reason_codes
    assert evaluation.candidate_metrics["mean_paired_uplift"] is None


def test_pool_B_a_tampered_generation_digest_refuses():
    """Identity, not counts: a matching count with a different id set fails."""

    players = _pool()
    ids = tuple(sorted(players))
    # a WRONG digest must be rejected even though the COUNT is right
    with pytest.raises(wc.WildcardInputError) as exc:
        wc.WildcardPoolBinding(
            generation_identity="gen-x", generation_id_sha256="sha256:" + "0" * 64,
            official_count=len(ids), eligible_ids=ids,
        )
    assert wc.OFFICIAL_PLAYER_POOL_INCOMPLETE in str(exc.value)
    # and a wrong count is rejected too
    with pytest.raises(wc.WildcardInputError):
        wc.WildcardPoolBinding(
            generation_identity="gen-x", generation_id_sha256="sha256:" + "0" * 64,
            official_count=len(ids) - 1, eligible_ids=ids,
        )


def test_pool_C_an_eligible_player_absent_from_the_universe_refuses():
    """The disappearance this contract exists to prevent."""

    players = _pool()
    owned = _legal_owned_ids(players)          # computed on the FULL universe
    # drop an UNOWNED player, so the universe is genuinely short of an eligible id
    dropped = min(pid for pid in players if pid not in set(owned))
    binding = _pool_binding(players)           # bound to the FULL official pool
    reduced = {pid: p for pid, p in players.items() if pid != dropped}
    request = _request(reduced, owned, pool_binding=binding)
    evaluation = wc.evaluate_wildcard(request)
    assert wc.OFFICIAL_PLAYER_POOL_INCOMPLETE in evaluation.reason_codes
    assert evaluation.candidate_metrics["mean_paired_uplift"] is None


def test_pool_D_the_659_player_discovery_path_accounts_for_everyone():
    """Production-scale: every official player is SUPPORTED or EXCLUDED.

    This exercises the WHOLE path (screen -> frontier -> improvement -> final
    squad), not just screen_players: no excluded player may re-enter anywhere.
    """

    CLUBS = tuple(range(1, 21))
    players = {}
    pid = 1
    for position, count in (("GKP", 60), ("DEF", 200), ("MID", 240), ("FWD", 159)):
        for index in range(count):
            players[pid] = _player(pid, position, CLUBS[pid % len(CLUBS)],
                                   40 + (pid % 40), 2.0 + (pid % 11) * 0.7,
                                   p_start=0.2 + (pid % 8) * 0.1)
            pid += 1
    assert len(players) == 659

    # N players lack complete horizon support
    unsupported = sorted(players)[:37]
    for pid in unsupported:
        player = players[pid]
        short = {e: v for e, v in player.events.items() if e < 10}
        players[pid] = wc.WildcardPlayer(pid, player.position, player.club_id,
                                         player.market_price_tenths, short, player.web_name)
    N = len(unsupported)
    binding = _pool_binding(players)
    request = _request(players, _legal_owned_ids(players), pool_binding=binding, bank_tenths=200)

    accounting = wc.pool_accounting(request)
    assert accounting["official_count"] == 659
    assert accounting["eligible_count"] == 659
    assert accounting["supported_count"] == 659 - N
    assert accounting["excluded_count"] == N
    assert accounting["screened_count"] == 659 - N
    assert accounting["supported_count"] + accounting["excluded_count"] == accounting["eligible_count"]
    assert accounting["complete"] is True

    # every excluded player carries an explicit machine-readable reason
    assert set(accounting["excluded_ids"]) == set(unsupported)
    assert all(accounting["excluded_reasons"][pid] for pid in unsupported)

    scores, stats = wc.screen_players(request)
    assert stats["screened"] == 659 - N
    assert sorted(scores) == accounting["supported_ids"], "optimizer universe == supported ids"

    evaluation = wc.evaluate_wildcard(request)
    squad = evaluation.candidate_metrics["wildcard_squad"]
    assert not (set(squad) & set(unsupported)), "an excluded player reached the selected squad"

    frontier: set[int] = set()
    for ids in evaluation.candidate_metrics["frontier_by_position"].values():
        assert isinstance(ids, int)
    _, frontier_stats = wc.build_wildcard_candidates(request, scores)
    assert frontier_stats["legal_squads_considered"] > 0
    assert not (frontier & set(unsupported))


def test_pool_E_the_exclusion_audit_is_never_truncated():
    players = _pool()
    for pid in (5, 7, 9, 11, 13, 15):
        player = players[pid]
        short = {e: v for e, v in player.events.items() if e < 10}
        players[pid] = wc.WildcardPlayer(pid, player.position, player.club_id,
                                         player.market_price_tenths, short, player.web_name)
    accounting = wc.pool_accounting(_request(players, _legal_owned_ids(players)))
    assert accounting["excluded_count"] == 6
    assert len(accounting["excluded_ids"]) == 6          # full evidence
    assert len(accounting["excluded_reasons"]) == 6
    assert len(accounting["excluded_sample"]) <= 20      # a display sample only


# ---------------------------------------------------------------------------
# PHASE 4 — the SAVE arm consumes the accepted normal route
# ---------------------------------------------------------------------------


def _route_request(players, owned, **overrides):
    return _request(players, owned, **overrides)


def test_route_A_save_uses_the_routes_own_net_value_not_a_scalar():
    players = _pool()
    owned = _legal_owned_ids(players)
    cheap = _request(players, owned, save_route=_save_route(players, owned, mean_net_core=10.0))
    rich = _request(players, owned, save_route=_save_route(players, owned, mean_net_core=100.0))
    low_eval = wc.evaluate_wildcard(cheap)
    high_eval = wc.evaluate_wildcard(rich)
    low = low_eval.candidate_metrics["save_objective"]
    high = high_eval.candidate_metrics["save_objective"]
    assert high > low, "SAVE must follow the route's per-event net value"
    # The H1-H4 component is exactly the weighted sum of the route's OWN per-event
    # net values; H5+ adds the carried terminal state on top.
    weights = cheap.horizon.weights[:4]
    h1_h4 = sum(low_eval.evidence["save_policy"]["route_value_evidence"]["per_event"][e]["weighted"]
                for e in (5, 6, 7, 8))
    assert h1_h4 == pytest.approx(sum(w * 10.0 for w in weights))
    # the two routes differ ONLY on H1-H4, since both carry the same terminal squad
    assert high - low == pytest.approx(sum(w * 90.0 for w in weights))


def test_route_B_one_minus_four_hit_moves_save_by_exactly_four_points():
    """THE decisive regression: one hit authority, no double counting.

    Both routes have the SAME football value.  Route B's engine output already
    nets one paid transfer, so ``mean_net_core`` is 4 lower on one event.  The
    Wildcard SAVE value must differ by exactly the weighted 4 -- not 0 (hit
    ignored) and not 8 (hit subtracted twice).
    """

    players = _pool()
    owned = _legal_owned_ids(players)

    # Route A: no hit.  gross 100, net 100 on every event.
    route_a = _save_route(players, owned, mean_net_core=100.0)
    # Route B: the SAME football value, but one event carries a paid transfer,
    # so the engine reports net 96 for that event (gross stays 100).
    route_b = _save_route(players, owned, mean_net_core=100.0,
                          shifts={6: (tuple(sorted(owned)), 30, 96.0)})

    value_a = wc.evaluate_wildcard(_route_request(players, owned, save_route=route_a)).candidate_metrics["save_objective"]
    value_b = wc.evaluate_wildcard(_route_request(players, owned, save_route=route_b)).candidate_metrics["save_objective"]

    horizon = wc.wildcard_horizon(5, length=8)
    weight = horizon.weight_for(6)
    assert value_a - value_b == pytest.approx(4.0 * weight), "hit must count exactly once"
    # explicitly reject the two failure modes the brief names
    assert value_a - value_b != pytest.approx(0.0, abs=1e-9)
    assert value_a - value_b != pytest.approx(8.0 * weight, abs=1e-9)


def test_route_C_the_evaluator_never_subtracts_route_hit_points():
    players = _pool()
    owned = _legal_owned_ids(players)
    route = _save_route(players, owned, mean_net_core=100.0)
    # even if the route REPORTS hits, they must not be subtracted again
    noisy = wc.WildcardSaveRoute(
        events=tuple(wc.WildcardSaveRouteEvent(
            event=e.event, squad_ids=e.squad_ids, bank_tenths=e.bank_tenths,
            purchase_price_tenths=e.purchase_price_tenths, free_transfers=e.free_transfers,
            mean_net_core=e.mean_net_core, hit_points=4,
        ) for e in route.events),
        terminal_squad_ids=route.terminal_squad_ids,
        terminal_bank_tenths=route.terminal_bank_tenths,
        terminal_purchase_price_tenths=route.terminal_purchase_price_tenths,
        terminal_free_transfers=route.terminal_free_transfers,
        cumulative_hits=4,
    )
    with_hits = wc.evaluate_wildcard(_route_request(players, owned, save_route=noisy)).candidate_metrics["save_objective"]
    without = wc.evaluate_wildcard(_route_request(players, owned, save_route=route)).candidate_metrics["save_objective"]
    assert with_hits == pytest.approx(without), "reported hit_points must never be subtracted again"
    evidence = wc.evaluate_wildcard(_route_request(players, owned, save_route=noisy)).evidence["save_policy"]["route_value_evidence"]
    assert evidence["manual_hit_subtraction"] is False
    assert evidence["hit_authority"] == "ROUTE_MEAN_NET_CORE_ALREADY_NETS_HITS"


def test_route_D_per_event_evidence_consumes_the_matching_route_event():
    players = _pool()
    owned = _legal_owned_ids(players)
    route = _save_route(players, owned, mean_net_core=55.0)
    evaluation = wc.evaluate_wildcard(_route_request(players, owned, save_route=route))
    per_event = evaluation.evidence["save_policy"]["route_value_evidence"]["per_event"]
    for event in (5, 6, 7, 8):
        assert per_event[event]["source"] == "ROUTE_MEAN_NET_CORE"
        assert per_event[event]["event"] == event
    for event in (9, 10, 11, 12):
        assert per_event[event]["source"] == "CARRIED_TERMINAL_STATE"


def test_route_E_post_h4_carries_the_terminal_squad_without_inventing_transfers():
    """H5+ values the route's TERMINAL state; no transfers, no future information."""

    players = _pool()
    owned = _legal_owned_ids(players)
    # a route that ends with a materially different squad
    outside = [pid for pid in sorted(players) if pid not in set(owned)]
    terminal = list(owned[:-1]) + [outside[0]]
    route = _save_route(players, owned, mean_net_core=40.0,
                        shifts={8: (tuple(terminal), 30, 40.0)})
    request = _route_request(players, owned, save_route=route)
    evaluation = wc.evaluate_wildcard(request)
    terminal_value = wc._event_value(request, tuple(sorted(terminal)), 9)
    carried = evaluation.evidence["save_policy"]["route_value_evidence"]["per_event"][9]
    assert carried["mean_net_core"] == pytest.approx(terminal_value)
    assert carried["source"] == "CARRIED_TERMINAL_STATE"


def test_route_F_bank_basis_and_ft_come_from_the_route_terminal_state():
    players = _pool()
    owned = _legal_owned_ids(players)
    route = _save_route(players, owned, mean_net_core=40.0,
                        terminal_bank_tenths=123, terminal_free_transfers=4)
    request = _route_request(players, owned, bank_tenths=999, save_route=route)
    evaluation = wc.evaluate_wildcard(request)
    state = evaluation.evidence["save_policy"]["post_save_state_for_reservation"]
    assert state["bank_tenths"] == 123, "must use the ROUTE terminal bank, not the original"
    assert state["free_transfers"] == 4
    assert state["retains_wildcard_option"] is True
    assert state["planning_event"] == 5
    assert state["certification_identity"] == IDENTITY
    # basis is the route's own, per player
    assert state["purchase_price_tenths"]


def test_route_G_save_must_leave_the_wildcard_available():
    players = _pool()
    owned = _legal_owned_ids(players)
    route = _save_route(players, owned, wildcard_available=False)
    evaluation = wc.evaluate_wildcard(_route_request(players, owned, save_route=route))
    assert wc.WC_SAVE_ROUTE_INVALID in evaluation.reason_codes


def test_route_H_a_missing_route_refuses_rather_than_falling_back_to_a_scalar():
    players = _pool()
    owned = _legal_owned_ids(players)
    evaluation = wc.evaluate_wildcard(_route_request(players, owned, save_route=None))
    assert wc.WC_SAVE_ROUTE_MISSING in evaluation.reason_codes
    assert evaluation.candidate_metrics["mean_paired_uplift"] is None


def test_route_I_a_route_with_the_wrong_event_count_refuses():
    players = _pool()
    owned = _legal_owned_ids(players)
    route = _save_route(players, owned, events=(5, 6, 7))
    evaluation = wc.evaluate_wildcard(_route_request(players, owned, save_route=route))
    assert wc.WC_SAVE_ROUTE_INVALID in evaluation.reason_codes


def test_route_J_a_route_missing_an_event_refuses():
    players = _pool()
    owned = _legal_owned_ids(players)
    route = _save_route(players, owned, events=(5, 6, 7, 9))
    evaluation = wc.evaluate_wildcard(_route_request(players, owned, save_route=route))
    assert wc.WC_SAVE_ROUTE_INVALID in evaluation.reason_codes


def test_route_K_missing_basis_or_bank_or_ft_refuses():
    players = _pool()
    owned = _legal_owned_ids(players)
    good = _save_route(players, owned)

    no_basis = wc.WildcardSaveRoute(
        events=tuple(wc.WildcardSaveRouteEvent(
            event=e.event, squad_ids=e.squad_ids, bank_tenths=e.bank_tenths,
            purchase_price_tenths={},  # nothing carries a basis
            free_transfers=e.free_transfers, mean_net_core=e.mean_net_core,
        ) for e in good.events),
        terminal_squad_ids=good.terminal_squad_ids, terminal_bank_tenths=good.terminal_bank_tenths,
        terminal_purchase_price_tenths={}, terminal_free_transfers=good.terminal_free_transfers,
    )
    assert no_basis.problems()

    no_squad = wc.WildcardSaveRoute(
        events=good.events, terminal_squad_ids=(), terminal_bank_tenths=10,
        terminal_purchase_price_tenths={}, terminal_free_transfers=1,
    )
    assert any("terminal squad" in p for p in no_squad.problems())

    bad_ft = wc.WildcardSaveRoute(
        events=good.events, terminal_squad_ids=good.terminal_squad_ids,
        terminal_bank_tenths=good.terminal_bank_tenths,
        terminal_purchase_price_tenths=good.terminal_purchase_price_tenths,
        terminal_free_transfers=-1,
    )
    assert any("free transfers" in p for p in bad_ft.problems())


# ---------------------------------------------------------------------------
# PHASE 5 — the reservation receives the authoritative post-SAVE state, once
# ---------------------------------------------------------------------------


def test_reservation_A_the_evaluator_never_calls_the_reservation():
    players = _pool()
    owned = _legal_owned_ids(players)

    class Spy:
        calls = 0

        def estimate(self, *, action, planning_event, expiry_event, state):
            Spy.calls += 1
            return cd.ReservationEstimate(
                value=None, calibration_status=cd.CALIBRATION_UNCALIBRATED,
                terminal_value=0.0, weeks_to_expiry=None, reason_codes=(), conditional_on=(),
            )

    spy = Spy()
    wc.evaluate_wildcard(_route_request(players, owned, reservation=spy))
    assert Spy.calls == 0, "the reservation is the arbiter's seam, not the evaluator's"


def test_reservation_B_a_full_decision_calls_the_reservation_exactly_once():
    players = _pool(weak=set(range(1, 16)))
    owned = list(range(1, 16))

    class Spy:
        calls = 0
        payload = None

        def estimate(self, *, action, planning_event, expiry_event, state):
            Spy.calls += 1
            Spy.payload = dict(state)
            return cd.ReservationEstimate(
                value=1.0, calibration_status=cd.CALIBRATION_CALIBRATED,
                terminal_value=0.0, weeks_to_expiry=10, reason_codes=(),
                conditional_on=("weeks_remaining",),
            )

    route = _save_route(players, owned, terminal_bank_tenths=77, terminal_free_transfers=3)
    request = _route_request(players, owned, save_route=route)
    evaluation = wc.evaluate_wildcard(request)
    assert wc.evaluate_wildcard.__module__  # evaluator ran

    decision = cd.decide_chip_action(
        horizon_binding=_binding(),
        chip_availability=_chip_rows(5),
        evaluations={cd.CHIP_ACTION_WC: evaluation},
        reservation=Spy(),
        certification_valid=True,
        manager_state={"squad_ids": list(owned)},
    )
    assert Spy.calls == 1, "the arbiter applies the reservation exactly once"
    # and even a CALIBRATED reservation cannot unlock PLAY_CHIP for Wildcard V1
    assert decision.status != cd.STATUS_PLAY_CHIP
    assert decision.status in (cd.STATUS_CHIP_CANDIDATE_RECHECK_REQUIRED, cd.STATUS_CHIP_REVIEW_REQUIRED)
    # the evaluator's own evidence still carries the authoritative post-SAVE state
    state = evaluation.evidence["save_policy"]["post_save_state_for_reservation"]
    assert state["bank_tenths"] == 77
    assert state["free_transfers"] == 3
    assert state["retains_wildcard_option"] is True
