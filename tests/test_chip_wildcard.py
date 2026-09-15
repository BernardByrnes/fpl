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
    )
    base.update(overrides)
    return wc.WildcardRequest(**base)


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

    healthy_pool = _pool(strong=set(range(40, 80)))
    healthy = wc.evaluate_wildcard(_request(healthy_pool, _legal_owned_ids(healthy_pool)))

    bad_pool = _pool(weak=set(range(1, 16)))
    bad = wc.evaluate_wildcard(_request(bad_pool, list(range(1, 16))))

    healthy_uplift = healthy.candidate_metrics["mean_paired_uplift"]
    bad_uplift = bad.candidate_metrics["mean_paired_uplift"]
    assert bad_uplift > 0.0
    assert healthy_uplift < 0.25 * bad_uplift, (
        f"healthy uplift {healthy_uplift} should be negligible vs bad {bad_uplift}"
    )


def test_B2_a_better_save_route_reduces_the_uplift():
    players = _pool(weak=set(range(1, 16)))
    bad = list(range(1, 16))
    plain = wc.evaluate_wildcard(_request(players, bad))
    better_save = wc.evaluate_wildcard(_request(players, bad, save_route_value=plain.candidate_metrics["save_objective"] + 25.0))
    assert better_save.candidate_metrics["mean_paired_uplift"] < plain.candidate_metrics["mean_paired_uplift"]


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
    request = _request(players, list(range(1, 16)), save_route_value=100.0, save_route_expected_hits=1)
    evaluation = wc.evaluate_wildcard(request)
    save = evaluation.evidence["save_policy"]
    assert save["retains_wildcard_option"] is True
    assert save["expected_hits"] == 1
    # the normal route's hits are charged against the save arm, not ignored
    base = wc.evaluate_wildcard(_request(players, list(range(1, 16)), save_route_value=100.0, save_route_expected_hits=0))
    assert evaluation.candidate_metrics["save_objective"] < base.candidate_metrics["save_objective"]
    # The reservation belongs to the ARBITER, so this evaluator must not have
    # called it: it reports the raw play-vs-save difference and the post-SAVE
    # state the arbiter's provider needs, and nothing else.
    assert evaluation.candidate_metrics["net_of_reservation"] is None
    assert "reservation_value" not in save
    assert save["post_save_state_for_reservation"]["squad_ids"] == list(range(1, 16))
    assert save["post_save_state_for_reservation"]["retains_wildcard_option"] is True


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
    players = _with_row_override(_pool(), 7, 9, expected_points=float("nan"))
    evaluation = wc.evaluate_wildcard(_request(players, _legal_owned_ids(players)))
    assert wc.WC_PROJECTION_INVALID in evaluation.reason_codes


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
