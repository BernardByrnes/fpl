"""Phase 6A — fixed-15 manager lineup acceptance tests.

All tests are pure: they build synthetic joint-world matrices and never touch the
database or the frozen predictive models.
"""

from __future__ import annotations

import random

import pytest

from fpl_brain import manager_lineup as ml
from fpl_brain import manager_worlds as mw


def _squad(ndef, nmid, nfwd):
    """15 players: 2 GK + ndef DEF + nmid MID + nfwd FWD (must total 13)."""
    assert ndef + nmid + nfwd == 13
    gks = [1, 2]
    defs = list(range(10, 10 + ndef))
    mids = list(range(20, 20 + nmid))
    fwds = list(range(30, 30 + nfwd))
    positions = {}
    for pid in gks:
        positions[pid] = "GKP"
    for pid in defs:
        positions[pid] = "DEF"
    for pid in mids:
        positions[pid] = "MID"
    for pid in fwds:
        positions[pid] = "FWD"
    return gks, defs, mids, fwds, positions, sorted(positions)


def _policy(starter_ids, bench_gk, order, captain, vice):
    return ml.ManagerPolicy(tuple(starter_ids), bench_gk, tuple(order), captain, vice)


def _worlds(ids, per_world):
    """per_world: list of dicts {pid: (minutes, core)} → world matrix."""
    worlds = len(per_world)
    minutes = {pid: [0.0] * worlds for pid in ids}
    core = {pid: [0.0] * worlds for pid in ids}
    for index, row in enumerate(per_world):
        for pid in ids:
            value = row.get(pid)
            if value is not None:
                minutes[pid][index] = float(value[0])
                # An absent player cannot score: minutes == 0 implies core 0.
                core[pid][index] = float(value[1]) if float(value[0]) > 0 else 0.0
    return {"worlds": worlds, "player_ids": list(ids), "core": core, "minutes": minutes}


def _single(ids, row):
    return _worlds(ids, [row])


# ---------------------------------------------------------------------------
# Legality (1-8)
# ---------------------------------------------------------------------------


def _legal_case_a():
    gks, defs, mids, fwds, positions, ids = _squad(4, 6, 3)
    xi = defs[:3] + mids[:5] + fwds[:2] + [gks[0]]
    bench_gk = gks[1]
    order = (mids[5], defs[3], fwds[2])
    return ids, positions, xi, bench_gk, order


def test_legal_xi_accepted():
    ids, positions, xi, bench_gk, order = _legal_case_a()
    policy = _policy(xi, bench_gk, order, xi[3], xi[4])
    assert ml.policy_legality_errors(policy, positions) == []


def test_illegal_two_def_xi_rejected():
    gks, defs, mids, fwds, positions, ids = _squad(4, 6, 3)
    xi = defs[:2] + mids[:6] + fwds[:2] + [gks[0]]  # 2 DEF
    policy = _policy(xi, gks[1], (defs[2], defs[3], fwds[2]), xi[0], xi[1])
    errors = ml.policy_legality_errors(policy, positions)
    assert any("XI_FORMATION_ILLEGAL" in error for error in errors)


def test_illegal_six_def_xi_rejected():
    gks, defs, mids, fwds, positions, ids = _squad(6, 4, 3)
    xi = defs[:6] + mids[:3] + fwds[:1] + [gks[0]]  # 6 DEF
    policy = _policy(xi, gks[1], (mids[3], fwds[1], fwds[2]), xi[0], xi[1])
    errors = ml.policy_legality_errors(policy, positions)
    assert any("XI_FORMATION_ILLEGAL" in error for error in errors)


def test_captain_outside_xi_rejected():
    ids, positions, xi, bench_gk, order = _legal_case_a()
    bench_player = order[0]
    policy = _policy(xi, bench_gk, order, bench_player, xi[1])
    assert any("CAPTAIN_NOT_IN_XI" in error for error in ml.policy_legality_errors(policy, positions))


def test_captain_equals_vice_rejected():
    ids, positions, xi, bench_gk, order = _legal_case_a()
    policy = _policy(xi, bench_gk, order, xi[0], xi[0])
    assert any("CAPTAIN_EQUALS_VICE" in error for error in ml.policy_legality_errors(policy, positions))


def test_duplicate_player_rejected():
    ids, positions, xi, bench_gk, order = _legal_case_a()
    xi_dup = list(xi)
    xi_dup[1] = xi_dup[0]
    policy = _policy(xi_dup, bench_gk, order, xi[0], xi[2])
    errors = ml.policy_legality_errors(policy, positions)
    assert any("DUPLICATE_STARTER" in error or "SQUAD_NOT_15" in error for error in errors)


def test_exactly_one_starting_gk_enforced():
    gks, defs, mids, fwds, positions, ids = _squad(4, 6, 3)
    xi = [gks[0], gks[1]] + defs[:3] + mids[:4] + fwds[:2]  # 11 with two GK
    policy = _policy(xi, gks[1], (defs[3], mids[4], fwds[2]), xi[0], xi[2])
    errors = ml.policy_legality_errors(policy, positions)
    assert any("XI_NOT_EXACTLY_ONE_GK" in error for error in errors)


def test_exactly_one_bench_gk_enforced():
    ids, positions, xi, bench_gk, order = _legal_case_a()
    # Use a DEF as the "bench GK": must be rejected.
    policy = _policy(xi, order[0], (bench_gk, order[1], order[2]), xi[0], xi[1])
    errors = ml.policy_legality_errors(policy, positions)
    assert any("BENCH_GK_IS_NOT_GKP" in error or "OUTFIELD_BENCH_CONTAINS_GK" in error for error in errors)


# ---------------------------------------------------------------------------
# Autosubs (9-19)
# ---------------------------------------------------------------------------


def test_one_minute_starter_is_not_replaced():
    ids, positions, xi, bench_gk, order = _legal_case_a()
    policy = _policy(xi, bench_gk, order, xi[0], xi[1])
    row = {pid: (90.0, 1.0) for pid in xi}
    row[xi[0]] = (1.0, -3.0)  # plays one minute, negative
    for pid in order:
        row[pid] = (60.0, 5.0)
    outcome = ml.resolve_world(policy, positions, {k: v[0] for k, v in row.items()},
                               {k: v[1] for k, v in row.items()})
    assert xi[0] in outcome.counted_ids
    assert outcome.autosub_count == 0


def test_negative_points_starter_is_not_replaced():
    ids, positions, xi, bench_gk, order = _legal_case_a()
    policy = _policy(xi, bench_gk, order, xi[0], xi[1])
    row = {pid: (90.0, 2.0) for pid in xi}
    row[xi[5]] = (70.0, -5.0)
    for pid in order:
        row[pid] = (90.0, 9.0)
    outcome = ml.resolve_world(policy, positions, {k: v[0] for k, v in row.items()},
                               {k: v[1] for k, v in row.items()})
    assert xi[5] in outcome.counted_ids and outcome.autosub_count == 0


def test_zero_minute_gk_replaced_only_by_bench_gk():
    ids, positions, xi, bench_gk, order = _legal_case_a()
    policy = _policy(xi, bench_gk, order, xi[0], xi[1])
    gk = next(pid for pid in xi if positions[pid] == "GKP")
    row = {pid: (90.0, 2.0) for pid in xi}
    row[gk] = (0.0, 0.0)
    row[bench_gk] = (90.0, 6.0)
    outcome = ml.resolve_world(policy, positions, {k: v[0] for k, v in row.items()},
                               {k: v[1] for k, v in row.items()})
    assert outcome.gk_used and bench_gk in outcome.counted_ids and gk not in outcome.counted_ids


def test_gk_never_replaced_by_outfielder_and_vice_versa():
    ids, positions, xi, bench_gk, order = _legal_case_a()
    policy = _policy(xi, bench_gk, order, xi[0], xi[1])
    gk = next(pid for pid in xi if positions[pid] == "GKP")
    row = {pid: (90.0, 2.0) for pid in xi}
    row[gk] = (0.0, 0.0)
    row[bench_gk] = (0.0, 0.0)
    for pid in order:
        row[pid] = (90.0, 4.0)
    outcome = ml.resolve_world(policy, positions, {k: v[0] for k, v in row.items()},
                               {k: v[1] for k, v in row.items()})
    assert not outcome.gk_used
    assert gk in outcome.counted_ids
    assert all(positions[pid] != "GKP" for pid in outcome.entrants)


def test_case_a_first_legal_outfield_bench_enters():
    # Start 3-5-2, one DEF absent; bench [MID, DEF, FWD] → DEF enters (MID blocked).
    ids, positions, xi, bench_gk, order = _legal_case_a()
    policy = _policy(xi, bench_gk, order, xi[0], xi[1])
    missing_def = xi[0]
    row = {pid: (90.0, 2.0) for pid in xi}
    row[missing_def] = (0.0, 0.0)
    for pid in order:
        row[pid] = (90.0, 3.0)
    outcome = ml.resolve_world(policy, positions, {k: v[0] for k, v in row.items()},
                               {k: v[1] for k, v in row.items()})
    assert outcome.entrants == (order[1],)  # bench 2 DEF, not bench 1 MID


def test_case_b_formation_blocked_first_bench_skipped():
    # Start 3-4-3 with two DEF absent; bench [MID, DEF, DEF] → both DEF used, MID skipped.
    gks, defs, mids, fwds, positions, ids = _squad(5, 5, 3)
    xi = defs[:3] + mids[:4] + fwds[:3] + [gks[0]]
    order = (mids[4], defs[3], defs[4])
    policy = _policy(xi, gks[1], order, xi[0], xi[1])
    row = {pid: (90.0, 2.0) for pid in xi}
    row[defs[0]] = (0.0, 0.0)
    row[defs[1]] = (0.0, 0.0)
    for pid in order:
        row[pid] = (90.0, 3.0)
    outcome = ml.resolve_world(policy, positions, {k: v[0] for k, v in row.items()},
                               {k: v[1] for k, v in row.items()})
    assert set(outcome.entrants) == {defs[3], defs[4]}
    assert mids[4] not in outcome.entrants


def test_case_c_second_bench_enters_when_first_blocked():
    ids, positions, xi, bench_gk, order = _legal_case_a()
    policy = _policy(xi, bench_gk, order, xi[0], xi[1])
    row = {pid: (90.0, 2.0) for pid in xi}
    row[xi[0]] = (0.0, 0.0)  # one DEF absent
    for pid in order:
        row[pid] = (90.0, 3.0)
    # bench 1 is MID (blocked), bench 2 is DEF (legal)
    outcome = ml.resolve_world(policy, positions, {k: v[0] for k, v in row.items()},
                               {k: v[1] for k, v in row.items()})
    assert outcome.entrants == (order[1],)


def test_bench_priority_wins_between_multiple_legal_solutions():
    # One MID absent, bench [DEF, MID, FWD] both DEF and MID legal → bench order wins (DEF).
    gks, defs, mids, fwds, positions, ids = _squad(4, 6, 3)
    xi = defs[:3] + mids[:5] + fwds[:2] + [gks[0]]
    order = (defs[3], mids[5], fwds[2])
    policy = _policy(xi, gks[1], order, xi[0], xi[1])
    row = {pid: (90.0, 2.0) for pid in xi}
    row[mids[0]] = (0.0, 0.0)
    for pid in order:
        row[pid] = (90.0, 3.0)
    outcome = ml.resolve_world(policy, positions, {k: v[0] for k, v in row.items()},
                               {k: v[1] for k, v in row.items()})
    assert outcome.entrants == (order[0],)


def test_fewer_than_eleven_allowed_when_no_legal_substitute():
    ids, positions, xi, bench_gk, order = _legal_case_a()
    policy = _policy(xi, bench_gk, order, xi[0], xi[1])
    row = {pid: (90.0, 2.0) for pid in xi}
    missing_defs = [pid for pid in xi if positions[pid] == "DEF"][:2]
    for pid in missing_defs:
        row[pid] = (0.0, 0.0)
    for pid in order:
        row[pid] = (0.0, 0.0)  # no bench appeared
    outcome = ml.resolve_world(policy, positions, {k: v[0] for k, v in row.items()},
                               {k: v[1] for k, v in row.items()})
    assert outcome.autosub_count == 0
    assert len(outcome.counted_ids) == 9  # 11 - 2 absent


def test_negative_point_appearing_bench_still_enters_if_required():
    ids, positions, xi, bench_gk, order = _legal_case_a()
    policy = _policy(xi, bench_gk, order, xi[0], xi[1])
    missing_def = [pid for pid in xi if positions[pid] == "DEF"][0]
    row = {pid: (90.0, 2.0) for pid in xi}
    row[missing_def] = (0.0, 0.0)
    for pid in order:
        row[pid] = (0.0, 0.0)
    row[order[1]] = (90.0, -4.0)  # the only legal appearing bench, negative
    outcome = ml.resolve_world(policy, positions, {k: v[0] for k, v in row.items()},
                               {k: v[1] for k, v in row.items()})
    assert outcome.entrants == (order[1],) and outcome.autosub_points == pytest.approx(-4.0)


# ---------------------------------------------------------------------------
# Captain / vice (20-28)
# ---------------------------------------------------------------------------


def _captain_case():
    ids, positions, xi, bench_gk, order = _legal_case_a()
    policy = _policy(xi, bench_gk, order, xi[0], xi[1])
    return ids, positions, xi, order, policy


def test_captain_appears_doubled():
    ids, positions, xi, order, policy = _captain_case()
    minutes = {pid: 90.0 for pid in ids}
    core = {pid: 2.0 for pid in ids}
    core[policy.captain_id] = 7.0
    extra, armband = ml.captain_multiplier(policy, minutes, core)
    assert armband == "CAPTAIN" and extra == 7.0


def test_captain_one_minute_doubled():
    ids, positions, xi, order, policy = _captain_case()
    minutes = {pid: 90.0 for pid in ids}
    core = {pid: 2.0 for pid in ids}
    minutes[policy.captain_id] = 1.0
    core[policy.captain_id] = 5.0
    extra, armband = ml.captain_multiplier(policy, minutes, core)
    assert armband == "CAPTAIN" and extra == 5.0


def test_captain_negative_doubled():
    ids, positions, xi, order, policy = _captain_case()
    minutes = {pid: 90.0 for pid in ids}
    core = {pid: 2.0 for pid in ids}
    core[policy.captain_id] = -3.0
    extra, _ = ml.captain_multiplier(policy, minutes, core)
    assert extra == -3.0


def test_vice_doubled_when_captain_absent():
    ids, positions, xi, order, policy = _captain_case()
    minutes = {pid: 90.0 for pid in ids}
    core = {pid: 2.0 for pid in ids}
    minutes[policy.captain_id] = 0.0
    core[policy.vice_captain_id] = 6.0
    extra, armband = ml.captain_multiplier(policy, minutes, core)
    assert armband == "VICE" and extra == 6.0


def test_no_multiplier_when_both_absent():
    ids, positions, xi, order, policy = _captain_case()
    minutes = {pid: 90.0 for pid in ids}
    core = {pid: 2.0 for pid in ids}
    minutes[policy.captain_id] = 0.0
    minutes[policy.vice_captain_id] = 0.0
    extra, armband = ml.captain_multiplier(policy, minutes, core)
    assert armband == "NONE" and extra == 0.0


def test_autosub_does_not_inherit_captaincy():
    ids, positions, xi, order, _unused = _captain_case()
    policy = _policy(xi, _unused.bench_gk_id, order, xi[3], xi[4])
    world = _single(ids, {pid: (90.0, 2.0) for pid in xi})
    world["minutes"][policy.captain_id][0] = 0.0
    world["core"][policy.captain_id][0] = 0.0
    world["minutes"][policy.vice_captain_id][0] = 70.0
    world["core"][policy.vice_captain_id][0] = 8.0
    # captain is an outfield starter; replace by the first appearing bench
    starter = policy.captain_id
    replacement = order[0]
    world["minutes"][replacement][0] = 90.0
    world["core"][replacement][0] = 1.0
    metrics = ml.evaluate_policy(policy, world, positions)
    assert metrics["p_vice_takes_captaincy"] == pytest.approx(1.0)
    # Base: 8 other appeared outfield at 2 (=16), vice 8, GK 2, entrant 1 = 27.
    # Captaincy falls back to the vice, adding ONE extra copy of the vice's 8.
    assert metrics["mean_core"] == pytest.approx(27.0 + 8.0, abs=1e-9)


def test_dgw_player_appearing_in_one_fixture_not_autosubbed():
    ids, positions, xi, order, policy = _captain_case()
    # GW matrix is already aggregated across fixtures: 1 minute in one fixture.
    world = _single(ids, {pid: (90.0, 2.0) for pid in ids})
    world["minutes"][xi[0]][0] = 1.0
    outcome = ml.resolve_world(policy, positions,
                               {pid: world["minutes"][pid][0] for pid in ids},
                               {pid: world["core"][pid][0] for pid in ids})
    assert xi[0] in outcome.counted_ids and outcome.autosub_count == 0


def test_dgw_captain_doubles_entire_gw_score():
    ids, positions, xi, order, policy = _captain_case()
    # GW core = 4 (fixture 1) + 3 (fixture 2) = 7, appearance in one fixture.
    minutes = {pid: 90.0 for pid in ids}
    core = {pid: 0.0 for pid in ids}
    minutes[policy.captain_id] = 1.0
    core[policy.captain_id] = 7.0
    extra, armband = ml.captain_multiplier(policy, minutes, core)
    assert armband == "CAPTAIN" and extra == 7.0


def test_bgw_zero_minutes_can_be_autosubbed():
    ids, positions, xi, order, policy = _captain_case()
    world = _single(ids, {pid: (90.0, 2.0) for pid in ids})
    blank = [pid for pid in xi if positions[pid] == "DEF"][0]
    world["minutes"][blank][0] = 0.0
    outcome = ml.resolve_world(policy, positions,
                               {pid: world["minutes"][pid][0] for pid in ids},
                               {pid: world["core"][pid][0] for pid in ids})
    assert blank not in outcome.counted_ids


# ---------------------------------------------------------------------------
# Joint-world invariants (29-35)
# ---------------------------------------------------------------------------


def _fixture_for_worlds():
    from test_monte_carlo import _fixture, _finish
    return _finish(_fixture(cameo=0.2))


def test_capture_does_not_alter_football_and_is_consistent_across_subsets():
    from fpl_brain import monte_carlo as mc
    from fpl_brain.scoring_rules import DEFAULT_SCORING_RULES as RULES
    import copy
    fixture = _fixture_for_worlds()
    ids = [int(p["player_id"]) for side in fixture["sides"] for p in side["players"]]
    side1 = [int(p["player_id"]) for p in fixture["sides"][0]["players"]]
    side2 = [int(p["player_id"]) for p in fixture["sides"][1]["players"]]
    config = mc.MonteCarloConfig(simulations=300, seed=2026, calibration_states=500)
    plain = mc.simulate(copy.deepcopy({100: fixture}), config, RULES)
    only1 = mc.simulate(copy.deepcopy({100: fixture}), config, RULES, capture_player_ids=side1)
    both = mc.simulate(copy.deepcopy({100: fixture}), config, RULES, capture_player_ids=ids)
    assert plain["summaries"] == only1["summaries"] == both["summaries"]
    wm1, wmb = only1["world_matrix"], both["world_matrix"]
    for pid in side1:
        assert wm1["core"][pid] == wmb["core"][pid]
        assert wm1["minutes"][pid] == wmb["minutes"][pid]
    # Same seed + same frozen inputs → byte-identical matrix.
    again = mc.simulate(copy.deepcopy({100: fixture}), config, RULES, capture_player_ids=ids)
    assert again["world_matrix"] == both["world_matrix"]


def test_policy_evaluation_does_not_mutate_worlds():
    import copy
    ids, positions, xi, bench_gk, order = _legal_case_a()
    worlds = _worlds(ids, [{**{pid: (90.0, 2.0) for pid in ids}} for _ in range(20)])
    snapshot = copy.deepcopy(worlds)
    p1 = _policy(xi, bench_gk, order, xi[0], xi[1])
    p2 = _policy(xi, bench_gk, order, xi[1], xi[2])
    ml.evaluate_policy(p1, worlds, positions)
    ml.evaluate_policy(p2, worlds, positions)
    assert worlds == snapshot


def test_no_route_identity_in_football_rng():
    import inspect
    from fpl_brain import monte_carlo as mc
    source = inspect.getsource(mc)
    assert "route_id" not in source.lower()
    parameters = set(inspect.signature(mc.simulate).parameters)
    assert not any("route" in name for name in parameters)
    assert "capture_player_ids" in parameters


# ---------------------------------------------------------------------------
# Enumeration (36-40)
# ---------------------------------------------------------------------------


def test_enumeration_legal_unique_and_deterministic():
    ids, positions, xi, bench_gk, order = _legal_case_a()
    first = list(ml.enumerate_skeletons(ids, positions))
    second = list(ml.enumerate_skeletons(ids, positions))
    assert first == second  # deterministic
    assert len(first) == len(set(first))  # no duplicates
    for starter, gk, bench_order in first:
        policy = _policy(starter, gk, bench_order, starter[0], starter[1])
        assert ml.policy_legality_errors(policy, positions) == []


def test_enumeration_matches_hand_count():
    ids, positions, xi, bench_gk, order = _legal_case_a()
    skeletons = list(ml.enumerate_skeletons(ids, positions))
    # Hand count: 2 GK choices x (legal 10-of-13 outfield combos) x 3! bench orders.
    from itertools import combinations, permutations
    outfield = [pid for pid in ids if positions[pid] != "GKP"]
    legal = 0
    for combo in combinations(outfield, 10):
        counts = {p: 0 for p in ("DEF", "MID", "FWD")}
        for pid in combo:
            counts[positions[pid]] += 1
        if 3 <= counts["DEF"] <= 5 and 2 <= counts["MID"] <= 5 and 1 <= counts["FWD"] <= 3:
            legal += 1
    assert len(skeletons) == 2 * legal * 6


def test_rank_policies_best_mean_matches_brute_force():
    ids, positions, xi, bench_gk, order = _legal_case_a()
    random.seed(5)
    W = 6
    worlds = _worlds(ids, [
        {pid: (float(random.choice([0, 1, 65, 90])), round(random.uniform(-2, 10), 3)) for pid in ids}
        for _ in range(W)
    ])
    ranked = ml.rank_policies(ids, positions, worlds, top_k=1)
    best_ranked = ranked["top_policies"][0]
    # Independent brute force over skeletons and captain/vice pairs.
    best = -1e18
    for starter, gk, ben in ml.enumerate_skeletons(ids, positions):
        total = 0.0
        for w in range(W):
            minutes = {pid: worlds["minutes"][pid][w] for pid in ids}
            core = {pid: worlds["core"][pid][w] for pid in ids}
            outcome = ml.resolve_world(_policy(starter, gk, ben, starter[0], starter[1]), positions, minutes, core)
            total += sum(core[pid] for pid in outcome.counted_ids)
        base = total / W
        for cap in starter:
            for vice in starter:
                if vice == cap:
                    continue
                extra = sum(
                    (worlds["core"][cap][w] if worlds["minutes"][cap][w] > 0 else (
                        worlds["core"][vice][w] if worlds["minutes"][vice][w] > 0 else 0.0))
                    for w in range(W)
                ) / W
                best = max(best, base + extra)
    assert ml.evaluate_policy(best_ranked, worlds, positions)["mean_core"] == pytest.approx(best, abs=1e-9)
    assert ranked["evaluated_policies"] == ranked["skeletons"] * 11 * 10


def test_base_stats_match_brute_force():
    ids, positions, xi, bench_gk, order = _legal_case_a()
    random.seed(9)
    worlds = _worlds(ids, [
        {pid: (float(random.choice([0, 1, 65, 90])), round(random.uniform(-2, 10), 3)) for pid in ids}
        for _ in range(50)
    ])
    skeletons = list(ml.enumerate_skeletons(ids, positions))[:12]
    stats = mw.base_skeleton_stats(skeletons, positions, worlds)
    for index, (starter, gk, bench_order) in enumerate(skeletons):
        policy = _policy(starter, gk, bench_order, starter[0], starter[1])
        total = 0.0
        for world in range(worlds["worlds"]):
            minutes = {pid: worlds["minutes"][pid][world] for pid in ids}
            core = {pid: worlds["core"][pid][world] for pid in ids}
            outcome = ml.resolve_world(policy, positions, minutes, core)
            total += sum(core[pid] for pid in outcome.counted_ids)
        assert stats[index]["mean_core_base"] == pytest.approx(total / worlds["worlds"], abs=1e-9)
