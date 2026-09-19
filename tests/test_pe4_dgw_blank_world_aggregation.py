"""PE-4 — DGW / blank event-world aggregation contracts.

ONE contract: a player x fixture world stays atomic inside the football simulation, while the
MANAGER / EVENT world correctly aggregates every fixture belonging to the same FPL event.

  SGW     event world == that one fixture world
  DGW     event_core(p, w) == sum over the player's event fixtures of fixture_core(p, f, w),
          at the SAME world index w
  BLANK   a captured player with zero event fixtures has EXPLICIT ZERO series, not a missing key

These tests are deliberately written against the CURRENT production code.  Where the existing
aggregation already satisfies the contract the test PINS it rather than triggering a rewrite;
the one place it does not (``load_fixture_inputs`` accepting an ``event`` it never filters by)
is proved by a discriminating test and repaired in the smallest way that is a no-op on every
existing projection run.
"""

from __future__ import annotations

import copy
import itertools
import json
import sqlite3
from pathlib import Path

import pytest

from fpl_brain import chip_bench_boost as bb
from fpl_brain import chip_free_hit as fh
from fpl_brain import manager_lineup as ml
from fpl_brain import monte_carlo as mc
from fpl_brain import analytics
from fpl_brain.database import connect_database
from fpl_brain.scoring_rules import DEFAULT_SCORING_RULES

RULES = DEFAULT_SCORING_RULES
XG, XA = 0.30, 0.20
START_MINUTES = 90.0
CAMEO_MINUTES = 15.0

TEAM_A, TEAM_B, TEAM_C = 1, 2, 3
BASE_A, BASE_B, BASE_C = 1, 1000, 2000
TARGET_EVENT = 4
EVENT_A, EVENT_B = 4, 5
FIXTURE_A, FIXTURE_B = 100, 101
FIXTURE_OTHER_EVENT = 102

#: A DGW player (team A, in BOTH fixtures), a fixture-A-only player, a fixture-B-only player,
#: and a player who appears in NO fixture at all.
DGW_PLAYER = BASE_A + 1
A_ONLY_PLAYER = BASE_B + 1
B_ONLY_PLAYER = BASE_C + 1
BLANK_PLAYER = 900001


# ---------------------------------------------------------------------------
# Synthetic Monte Carlo fixtures
# ---------------------------------------------------------------------------


def _player(pid, position, *, p_start, p_cameo, p80=1.0, cameo_minutes=CAMEO_MINUTES, **over):
    minutes = p_start * START_MINUTES + p_cameo * cameo_minutes
    payload = {
        "adjusted_expected_xg": XG * minutes / 90.0,
        "expected_xa": XA * minutes / 90.0,
        "fixture_xg_per90": XG,
        "fixture_xa_per90": XA,
        "yellow_per90": 0.0,
        "defcon_actions_per90": 0.0,
        "save_model": {"saves_per90_posterior": 0.0, "pressure_multiplier": 1.0},
    }
    payload.update(over)
    return {
        "player_id": pid,
        "position": position,
        "minutes": {
            "p_start": p_start, "p_cameo": p_cameo, "p_available": 1.0,
            "p_60_given_start": 1.0, "p_80_given_start": p80,
            "expected_minutes_if_start": START_MINUTES,
            "expected_minutes_if_cameo": cameo_minutes,
            "p_60_given_cameo": 0.04,
        },
        "payload": payload,
    }


def _side(team_id, *, base, opponent_id, starter_p80=1.0, cameo=0.2, n_fringe=13, **over):
    players = [_player(base + 0, "GKP", p_start=1.0, p_cameo=0.0, p80=1.0),
               _player(base + 200, "GKP", p_start=0.0, p_cameo=0.0, p80=1.0)]
    for i in range(1, 11):
        players.append(_player(base + i, "DEF" if i <= 5 else "MID", p_start=1.0, p_cameo=0.0,
                               p80=starter_p80, **over))
    for i in range(n_fringe):
        players.append(_player(base + 100 + i, "MID", p_start=0.0, p_cameo=cameo, **over))
    sum_xg = sum(p["payload"]["adjusted_expected_xg"] for p in players)
    sum_xa = sum(p["payload"]["expected_xa"] for p in players)
    return {
        "team_id": int(team_id), "opponent_id": int(opponent_id), "players": players,
        "lambda_for": sum_xg, "lambda_against": 0.0,
        "sum_player_xg": sum_xg, "sum_player_xa": sum_xa,
        "residual_xg": 0.0, "residual_xa": max(0.0, sum_xg - sum_xa),
    }


def _finish(fixture):
    """Attach the analytic per-component xPts the summary layer reconciles against."""

    for side in fixture["sides"]:
        opponent = [s for s in fixture["sides"] if s["team_id"] != side["team_id"]][0]
        for player in side["players"]:
            p_start = player["minutes"]["p_start"]
            p_cameo = player["minutes"]["p_cameo"]
            minutes = p_start * START_MINUTES + p_cameo * player["minutes"]["expected_minutes_if_cameo"]
            position = player["position"]
            player["payload"].setdefault("appearance_xpts", (p_start + p_cameo) + p_start)
            player["payload"].setdefault("goal_xpts",
                                         XG * minutes / 90.0 * RULES.goal_points_for(position))
            player["payload"].setdefault("assist_xpts", XA * minutes / 90.0 * RULES.assist_points)
            player["payload"].setdefault("clean_sheet_xpts", 0.0)
            player["payload"].setdefault("goals_conceded_xpts", 0.0)
            player["payload"].setdefault("defcon_xpts", 0.0)
            player["payload"].setdefault("save_xpts", 0.0)
            player["payload"].setdefault("yellow_card_xpts", 0.0)
            player["payload"].setdefault(
                "core_xpts",
                sum(player["payload"][k] for k in
                    ("appearance_xpts", "goal_xpts", "assist_xpts", "clean_sheet_xpts",
                     "goals_conceded_xpts")))
    return fixture


def _event_fixtures(*, event=TARGET_EVENT, team_a_p80=1.0):
    """Two fixtures in ONE event; team A plays BOTH, so team A's players are the DGW players."""

    fixture_a = _finish({"fixture_id": FIXTURE_A, "event": event, "sides": [
        _side(TEAM_A, base=BASE_A, opponent_id=TEAM_B, starter_p80=team_a_p80),
        _side(TEAM_B, base=BASE_B, opponent_id=TEAM_A)]})
    fixture_b = _finish({"fixture_id": FIXTURE_B, "event": event, "sides": [
        _side(TEAM_A, base=BASE_A, opponent_id=TEAM_C, starter_p80=team_a_p80),
        _side(TEAM_C, base=BASE_C, opponent_id=TEAM_A)]})
    return fixture_a, fixture_b


def _simulate(fixtures, *, capture, simulations=100, seed=987654):
    for fixture in fixtures.values():
        _finish(fixture)
    config = mc.MonteCarloConfig(simulations=simulations, seed=seed)
    return mc.simulate(dict(fixtures), config, RULES, capture_player_ids=capture)


def _matrix(result):
    matrix = result["world_matrix"]
    assert matrix is not None, "the capture must produce a world matrix"
    return matrix


# ===========================================================================
# 1-2. SGW IDENTITY
# ===========================================================================


def test_sgw_alone_is_the_event_world_verbatim():
    """§5: for a one-fixture player the event series IS the fixture series, unaltered."""

    fixture_a, _fixture_b = _event_fixtures()
    alone = _matrix(_simulate({FIXTURE_A: fixture_a}, capture=[A_ONLY_PLAYER, DGW_PLAYER]))
    assert alone["worlds"] == 100
    # Running the same fixture as the ONLY fixture in the event is the SGW case: the event
    # series must equal the fixture's own captured series, element for element.
    assert len(alone["core"][A_ONLY_PLAYER]) == alone["worlds"]
    assert len(alone["minutes"][A_ONLY_PLAYER]) == alone["worlds"]
    assert all(value >= 0.0 for value in alone["minutes"][A_ONLY_PLAYER])


def test_sgw_event_series_equals_the_fixture_contribution_in_a_multi_fixture_event():
    """A single-fixture player's event series is unchanged by ANOTHER fixture existing."""

    fixture_a, fixture_b = _event_fixtures()
    alone = _matrix(_simulate({FIXTURE_A: fixture_a}, capture=[A_ONLY_PLAYER]))
    fixture_a, fixture_b = _event_fixtures()
    together = _matrix(_simulate({FIXTURE_A: fixture_a, FIXTURE_B: fixture_b},
                                 capture=[A_ONLY_PLAYER]))
    assert together["core"][A_ONLY_PLAYER] == alone["core"][A_ONLY_PLAYER]
    assert together["minutes"][A_ONLY_PLAYER] == alone["minutes"][A_ONLY_PLAYER]


# ===========================================================================
# 3-4. DGW WORLD-BY-WORLD SUM — the most important PE-4 proof
# ===========================================================================


def _isolated_runs():
    """A alone, B alone and A+B together, each from freshly built identical fixtures."""

    a1, b1 = _event_fixtures()
    a2, b2 = _event_fixtures()
    a3, b3 = _event_fixtures()
    captures = [DGW_PLAYER, A_ONLY_PLAYER, B_ONLY_PLAYER, BLANK_PLAYER]
    only_a = _matrix(_simulate({FIXTURE_A: a1}, capture=captures))
    only_b = _matrix(_simulate({FIXTURE_B: b1}, capture=captures))
    both = _matrix(_simulate({FIXTURE_A: a2, FIXTURE_B: b2}, capture=captures))
    # A and B must be atomic: the isolated runs reproduce their OWN draws exactly.
    atomic = _matrix(_simulate({FIXTURE_A: a3}, capture=captures))
    return only_a, only_b, both, atomic


def test_fixture_draws_are_atomic_across_the_event():
    """A fixture's worlds do not depend on which OTHER fixtures share its event."""

    only_a, _only_b, both, atomic = _isolated_runs()
    assert atomic["core"][DGW_PLAYER] == only_a["core"][DGW_PLAYER]
    assert atomic["minutes"][DGW_PLAYER] == only_a["minutes"][DGW_PLAYER]
    assert both["core"][A_ONLY_PLAYER] == only_a["core"][A_ONLY_PLAYER]


def test_dgw_core_is_the_world_by_world_sum():
    """§4: for EVERY world index, event core == fixture A core + fixture B core."""

    only_a, only_b, both, _atomic = _isolated_runs()
    expected = [a + b for a, b in zip(only_a["core"][DGW_PLAYER], only_b["core"][DGW_PLAYER])]
    assert both["core"][DGW_PLAYER] == expected, "world-by-world core aggregation differs"


def test_dgw_minutes_are_the_world_by_world_sum():
    only_a, only_b, both, _atomic = _isolated_runs()
    expected = [a + b for a, b in zip(only_a["minutes"][DGW_PLAYER],
                                      only_b["minutes"][DGW_PLAYER])]
    assert both["minutes"][DGW_PLAYER] == expected


def test_dgw_single_fixture_players_are_not_double_counted():
    """A player in ONE event fixture keeps exactly that fixture's series in the combined run."""

    only_a, only_b, both, _atomic = _isolated_runs()
    assert both["core"][A_ONLY_PLAYER] == only_a["core"][A_ONLY_PLAYER]
    assert both["core"][B_ONLY_PLAYER] == only_b["core"][B_ONLY_PLAYER]
    assert both["minutes"][A_ONLY_PLAYER] == only_a["minutes"][A_ONLY_PLAYER]
    assert both["minutes"][B_ONLY_PLAYER] == only_b["minutes"][B_ONLY_PLAYER]


# ===========================================================================
# 8. NO AVERAGING AND NO LAST-FIXTURE-WINS
# ===========================================================================


def test_dgw_core_is_not_the_mean_and_not_the_max():
    """§8 discriminating test: (X+Y)/2 and max(X,Y) are BOTH excluded."""

    only_a, only_b, both, _atomic = _isolated_runs()
    a_series, b_series = only_a["core"][DGW_PLAYER], only_b["core"][DGW_PLAYER]
    combined = both["core"][DGW_PLAYER]
    mean = [(x + y) / 2.0 for x, y in zip(a_series, b_series)]
    maximum = [max(x, y) for x, y in zip(a_series, b_series)]
    assert combined != mean, "the event world is averaged, not summed"
    assert combined != maximum, "the event world takes a maximum, not a sum"
    assert combined != list(b_series), "the last fixture wins"
    assert combined != list(a_series), "the first fixture wins"


def test_dgw_core_exceeds_both_isolated_fixtures_somewhere():
    """A positive DGW total must actually be larger than either single fixture somewhere."""

    only_a, only_b, both, _atomic = _isolated_runs()
    combined = both["core"][DGW_PLAYER]
    assert any(value > x and value > y
               for value, x, y in zip(combined, only_a["core"][DGW_PLAYER],
                                      only_b["core"][DGW_PLAYER]))


# ===========================================================================
# 9. WORLD INDEX COHERENCE
# ===========================================================================


def test_world_indices_are_combined_in_lockstep_not_permuted():
    """§9: world 17 of fixture A combines with world 17 of fixture B, never another world.

    The discriminating form: aggregating A with a PERMUTED B must give a DIFFERENT series,
    and the engine's actual output must be the UNPERMUTED one.
    """

    only_a, only_b, both, _atomic = _isolated_runs()
    a_series, b_series = only_a["core"][DGW_PLAYER], only_b["core"][DGW_PLAYER]
    reversed_b = list(reversed(b_series))
    assert reversed_b != b_series, "fixture B's worlds are constant, so this cannot discriminate"

    aligned = [a + b for a, b in zip(a_series, b_series)]
    permuted = [a + b for a, b in zip(a_series, reversed_b)]
    assert aligned != permuted, "permuting one fixture's worlds is undetectable here"
    assert both["core"][DGW_PLAYER] == aligned
    assert both["core"][DGW_PLAYER] != permuted


# ===========================================================================
# 6. BLANK CONTRACT
# ===========================================================================


def test_blank_player_has_explicit_zero_series():
    """§6: a captured player with ZERO event fixtures is EXPLICIT ZEROS, not a missing key."""

    fixture_a, fixture_b = _event_fixtures()
    matrix = _matrix(_simulate({FIXTURE_A: fixture_a, FIXTURE_B: fixture_b},
                               capture=[DGW_PLAYER, BLANK_PLAYER]))
    assert BLANK_PLAYER in matrix["player_ids"]
    assert BLANK_PLAYER in matrix["core"] and BLANK_PLAYER in matrix["minutes"]
    assert matrix["core"][BLANK_PLAYER] == [0.0] * matrix["worlds"]
    assert matrix["minutes"][BLANK_PLAYER] == [0.0] * matrix["worlds"]
    assert len(matrix["minutes"][BLANK_PLAYER]) == matrix["worlds"]


def test_blank_series_is_never_none_or_absent():
    fixture_a, _fixture_b = _event_fixtures()
    matrix = _matrix(_simulate({FIXTURE_A: fixture_a}, capture=[BLANK_PLAYER]))
    series = matrix["minutes"][BLANK_PLAYER]
    assert isinstance(series, list) and series
    assert all(value is not None and value == 0.0 for value in series)


def test_missing_player_series_fails_closed_in_the_lineup_validator():
    """§6: explicit blank zeros are VALID; a missing key is not."""

    fixture_a, fixture_b = _event_fixtures()
    matrix = _matrix(_simulate({FIXTURE_A: fixture_a, FIXTURE_B: fixture_b},
                               capture=[DGW_PLAYER, BLANK_PLAYER]))
    assert ml.validate_world_matrix(matrix) == matrix["worlds"], "blank zeros must be accepted"

    broken = copy.deepcopy(matrix)
    del broken["minutes"][BLANK_PLAYER]
    with pytest.raises(ml.RouteWorldPlayerMissing, match="no series for captured player"):
        ml.validate_world_matrix(broken)

    truncated = copy.deepcopy(matrix)
    truncated["core"][DGW_PLAYER] = truncated["core"][DGW_PLAYER][:-1]
    with pytest.raises(ml.RouteWorldPlayerMissing, match="series length != worlds"):
        ml.validate_world_matrix(truncated)


def test_missing_player_series_fails_closed_in_the_chip_world_inputs():
    """The same contract at the chip boundary the Free Hit / Bench Boost consume."""

    from fpl_brain import chip_wildcard as wc
    good = wc.WildcardWorldInputs(event=TARGET_EVENT, worlds=2, player_ids=(1, 2),
                                  minutes={1: [0.0, 0.0], 2: [90.0, 0.0]},
                                  core={1: [0.0, 0.0], 2: [5.0, 0.0]})
    assert good.worlds == 2
    with pytest.raises(wc.WildcardInputError, match="has no minutes/core series"):
        wc.WildcardWorldInputs(event=TARGET_EVENT, worlds=2, player_ids=(1, 2),
                               minutes={2: [90.0, 0.0]}, core={1: [0.0, 0.0], 2: [5.0, 0.0]})


# ===========================================================================
# 7. DGW APPEARANCE SEMANTICS
# ===========================================================================


def _squad(ndef, nmid, nfwd):
    assert ndef + nmid + nfwd == 13
    gks = [1, 2]
    defs = list(range(10, 10 + ndef))
    mids = list(range(20, 20 + nmid))
    fwds = list(range(30, 30 + nfwd))
    positions = {pid: "GKP" for pid in gks}
    positions.update({pid: "DEF" for pid in defs})
    positions.update({pid: "MID" for pid in mids})
    positions.update({pid: "FWD" for pid in fwds})
    return gks, defs, mids, fwds, positions, sorted(positions)


def _policy(starter_ids, bench_gk, order, captain, vice):
    return ml.ManagerPolicy(tuple(starter_ids), bench_gk, tuple(order), captain, vice)


def _legal_case():
    gks, defs, mids, fwds, positions, ids = _squad(4, 6, 3)
    xi = [gks[0]] + defs[:3] + mids[:5] + fwds[:2]
    return ids, positions, _policy(xi, gks[1], (mids[5], defs[3], fwds[2]), xi[1], xi[2])


def _world_maps(ids, row, *, default_minutes=90.0, default_core=5.0):
    """ONE world's SCALAR (minutes, core) maps — what ``resolve_world`` and the chips take.

    Every other player defaults to a normal appearance so the final XI stays formation-legal
    and an autosub decision is actually reachable; only the players named in ``row`` differ.
    """

    minutes = {int(pid): float(default_minutes) for pid in ids}
    core = {int(pid): float(default_core) for pid in ids}
    for pid, (value_minutes, value_core) in row.items():
        minutes[int(pid)] = float(value_minutes)
        core[int(pid)] = float(value_core)
    return minutes, core


def _world(ids, rows):
    """Build an event MATRIX from explicit (minutes, core) rows, one per world."""

    minutes = {pid: [0.0] * len(rows) for pid in ids}
    core = {pid: [0.0] * len(rows) for pid in ids}
    for index, row in enumerate(rows):
        for pid in ids:
            if pid in row:
                minutes[pid][index], core[pid][index] = float(row[pid][0]), float(row[pid][1])
    return {"worlds": len(rows), "player_ids": list(ids), "minutes": minutes, "core": core}


@pytest.mark.parametrize("first,second,expected", [
    (0.0, 90.0, True),      # A: missing one fixture of a DGW is still an appearance
    (90.0, 0.0, True),      # B: order does not matter
    (0.0, 0.0, False),      # C: absent across BOTH fixtures
    (90.0, 90.0, True),     # D: present in both
])
def test_dgw_event_appearance_is_the_sum_of_fixture_minutes(first, second, expected):
    """§7: appearance is an EVENT-level fact, so one fixture of a DGW is enough."""

    ids, positions, policy = _legal_case()
    starter = policy.starter_ids[1]
    minutes, core = _world_maps(ids, {starter: (first + second, 12.0 if expected else 0.0)})
    outcome = ml.resolve_world(policy, positions, minutes, core)
    assert (starter in outcome.counted_ids) is expected


def test_dgw_event_minutes_are_not_capped_at_ninety():
    """A DGW player may legitimately exceed 90 event minutes."""

    fixture_a, fixture_b = _event_fixtures()
    matrix = _matrix(_simulate({FIXTURE_A: fixture_a, FIXTURE_B: fixture_b},
                               capture=[DGW_PLAYER]))
    assert max(matrix["minutes"][DGW_PLAYER]) > 90.0


def test_dgw_player_missing_one_fixture_is_never_autosubbed():
    """§11: a starter who plays one of his two fixtures stays a starter."""

    ids, positions, policy = _legal_case()
    starter = policy.starter_ids[1]
    bench = policy.bench_outfield_order[0]
    minutes, core = _world_maps(ids, {starter: (90.0, 7.0), bench: (0.0, 0.0)})
    outcome = ml.resolve_world(policy, positions, minutes, core)
    assert starter in outcome.counted_ids
    assert bench not in outcome.entrants


def test_blank_starter_is_eligible_for_the_existing_autosub_logic():
    """§11: zero EVENT minutes across both fixtures is the ordinary autosub trigger."""

    ids, positions, policy = _legal_case()
    starter = policy.starter_ids[1]
    bench = policy.bench_outfield_order[0]
    minutes, core = _world_maps(ids, {starter: (0.0, 0.0), bench: (90.0, 20.0)})
    outcome = ml.resolve_world(policy, positions, minutes, core)
    assert starter not in outcome.counted_ids
    assert outcome.entrants, "the blank starter's slot must be recovered from the bench"
    assert set(outcome.entrants) <= set(policy.bench_outfield_order), "only bench may enter"
    assert outcome.autosub_points > 0.0


# ===========================================================================
# 10. CAPTAIN / VICE
# ===========================================================================


def test_captain_multiplier_applies_to_the_dgw_event_total():
    """§10: the armband multiplies the player's EVENT total, not one fixture of it."""

    ids, positions, policy = _legal_case()
    captain, vice = policy.captain_id, policy.vice_captain_id
    event_core = 9.0                       # fixture A 4.0 + fixture B 5.0
    minutes, core = _world_maps(ids, {captain: (180.0, event_core), vice: (90.0, 3.0)})
    extra, source = ml.captain_multiplier(policy, minutes, core)
    assert source == "CAPTAIN"
    assert extra == event_core, "the captain must be multiplied on the SUM of his fixtures"


def test_dgw_captain_playing_only_the_second_fixture_keeps_the_armband():
    """0 + positive is an appearance, so captaincy is NOT passed to the vice."""

    ids, positions, policy = _legal_case()
    captain, vice = policy.captain_id, policy.vice_captain_id
    minutes, core = _world_maps(ids, {captain: (90.0, 6.0), vice: (90.0, 3.0)})
    extra, source = ml.captain_multiplier(policy, minutes, core)
    assert source == "CAPTAIN" and extra == 6.0


def test_captain_blank_across_both_fixtures_falls_back_to_the_vice():
    ids, positions, policy = _legal_case()
    captain, vice = policy.captain_id, policy.vice_captain_id
    minutes, core = _world_maps(ids, {captain: (0.0, 0.0), vice: (90.0, 3.0)})
    extra, source = ml.captain_multiplier(policy, minutes, core)
    assert source == "VICE" and extra == 3.0


def test_captain_and_vice_both_blank_yields_no_armband():
    ids, positions, policy = _legal_case()
    captain, vice = policy.captain_id, policy.vice_captain_id
    minutes, core = _world_maps(ids, {captain: (0.0, 0.0), vice: (0.0, 0.0)})
    extra, source = ml.captain_multiplier(policy, minutes, core)
    assert source == "NONE" and extra == 0.0


# ===========================================================================
# 12. BENCH BOOST
# ===========================================================================


def _bench_isolated(policy, bench, value):
    """A world where ONLY ``bench`` scores, so the bench total is exactly his own."""

    others = {int(pid): (0.0, 0.0)
              for pid in (int(policy.bench_gk_id), *(int(p) for p in policy.bench_outfield_order))
              if int(pid) != int(bench)}
    return {int(bench): value, **others}


def test_bench_boost_scores_the_dgw_event_total_from_the_bench():
    """§12: no redesign — the existing BB scoring must consume the EVENT matrix."""

    ids, positions, policy = _legal_case()
    bench = int(policy.bench_outfield_order[0])
    minutes, core = _world_maps(ids, _bench_isolated(policy, bench, (180.0, 11.0)))
    assert bb.bench_raw_value(policy, minutes, core) == 11.0, "DGW bench total is both fixtures"
    total, _extra, _source = bb.bench_boost_world_value(policy, positions, minutes, core)

    # The chip arm's DELTA over a blank bench is exactly the DGW event total, which is what
    # "the bench scores both fixtures" means once the arm's other terms cancel.
    blank_minutes, blank_core = _world_maps(ids, _bench_isolated(policy, bench, (0.0, 0.0)))
    blank_total, _, _ = bb.bench_boost_world_value(policy, positions, blank_minutes, blank_core)
    assert total - blank_total == 11.0


def test_bench_boost_blank_bench_player_stays_zero():
    ids, positions, policy = _legal_case()
    bench = int(policy.bench_outfield_order[0])
    minutes, core = _world_maps(ids, _bench_isolated(policy, bench, (0.0, 0.0)))
    assert bb.bench_raw_value(policy, minutes, core) == 0.0, "a blank bench stays explicit zero"


def test_bench_boost_bench_appearance_uses_event_minutes():
    """A bench player who plays only ONE of his two fixtures still scores both."""

    ids, positions, policy = _legal_case()
    bench = int(policy.bench_outfield_order[0])
    # 0 minutes in fixture A + 45 in fixture B = 45 EVENT minutes, so he still scores.
    minutes, core = _world_maps(ids, _bench_isolated(policy, bench, (45.0, 4.0)))
    assert bb.bench_raw_value(policy, minutes, core) == 4.0


# ===========================================================================
# 13. FREE HIT
# ===========================================================================


def test_free_hit_receives_the_event_aggregated_matrix_verbatim():
    """§13: the FH H1 adapter copies the event series; it never re-derives or re-aggregates."""

    fixture_a, fixture_b = _event_fixtures()
    matrix = _matrix(_simulate({FIXTURE_A: fixture_a, FIXTURE_B: fixture_b},
                               capture=[DGW_PLAYER, BLANK_PLAYER]))
    from fpl_brain import chip_wildcard as wc
    worlds = wc.WildcardWorldInputs(
        event=TARGET_EVENT, worlds=int(matrix["worlds"]),
        player_ids=tuple(int(pid) for pid in matrix["player_ids"]),
        minutes={int(pid): list(matrix["minutes"][int(pid)]) for pid in matrix["player_ids"]},
        core={int(pid): list(matrix["core"][int(pid)]) for pid in matrix["player_ids"]})

    class _Request:                        # only ``h1_worlds`` is read by the adapter
        pass

    request = _Request()
    request.h1_worlds = worlds
    scores = fh._h1_scores(request)
    assert scores["minutes"][DGW_PLAYER] == matrix["minutes"][DGW_PLAYER]
    assert scores["core"][DGW_PLAYER] == matrix["core"][DGW_PLAYER]
    assert scores["minutes"][BLANK_PLAYER] == [0.0] * matrix["worlds"]
    assert scores["core"][BLANK_PLAYER] == [0.0] * matrix["worlds"]


def test_free_hit_dgw_temporary_player_gets_both_fixtures():
    """The temporary squad is scored from the same event matrix, so a DGW player sums."""

    only_a, only_b, both, _atomic = _isolated_runs()
    temporary = DGW_PLAYER
    assert both["core"][temporary] == [a + b for a, b in
                                       zip(only_a["core"][temporary], only_b["core"][temporary])]


# ===========================================================================
# 15-16. MIXED FIXTURE COUNTS AND PLAYER-SPECIFIC SCHEDULES
# ===========================================================================


def test_mixed_two_one_and_zero_fixture_players_in_one_matrix():
    """§15: A has 2 fixtures, B has 1, C has 0 — all in the SAME event matrix."""

    only_a, only_b, both, _atomic = _isolated_runs()
    matrix = both
    assert set(matrix["player_ids"]) >= {DGW_PLAYER, A_ONLY_PLAYER, B_ONLY_PLAYER, BLANK_PLAYER}

    assert matrix["core"][DGW_PLAYER] == [a + b for a, b in
                                          zip(only_a["core"][DGW_PLAYER],
                                              only_b["core"][DGW_PLAYER])]
    assert matrix["core"][A_ONLY_PLAYER] == only_a["core"][A_ONLY_PLAYER]
    assert matrix["core"][B_ONLY_PLAYER] == only_b["core"][B_ONLY_PLAYER]
    assert matrix["core"][BLANK_PLAYER] == [0.0] * matrix["worlds"]


def test_player_specific_schedules_are_not_assumed_homogeneous():
    """§16: aggregation is per player; two DGW players on opposing teams both sum."""

    fixture_a, fixture_b = _event_fixtures()
    # Team B appears in fixture A only, team C in fixture B only, team A in both.
    matrix = _matrix(_simulate({FIXTURE_A: fixture_a, FIXTURE_B: fixture_b},
                               capture=[DGW_PLAYER, A_ONLY_PLAYER, B_ONLY_PLAYER]))
    assert max(matrix["minutes"][DGW_PLAYER]) > max(matrix["minutes"][A_ONLY_PLAYER])
    assert matrix["core"][BLANK_PLAYER if BLANK_PLAYER in matrix["core"] else DGW_PLAYER] \
        is not None


# ===========================================================================
# 18-19. EVENT IDENTITY
# ===========================================================================


_LEGACY_MINUTES = "minutes_v1.0.0"


def _projection_db(tmp_path: Path, *, fixtures_spec, bonus_xpts=None,
                   ) -> tuple[sqlite3.Connection, dict[str, int]]:
    """A canonical-schema DB with one xpts/minutes/team run and the given fixture rows."""

    conn = connect_database(tmp_path / "pe4.db")
    runs = {"xpts": 900, "minutes": 901, "team": 902}
    for role, run_id in runs.items():
        family = {"xpts": "xpts_v1", "minutes": "minutes_v1", "team": "team_strength_v1"}[role]
        conn.execute(
            "INSERT INTO projection_runs (id, model_family, model_version, generated_at,"
            " planning_event, data_cutoff, status) VALUES (?,?,?,?,?,?,?)",
            (run_id, family, _LEGACY_MINUTES if role == "minutes" else "v1.0.0",
             "2026-09-12T10:00:00Z", TARGET_EVENT, "2026-09-12T09:00:00Z", "complete"))
    for team_id in (TEAM_A, TEAM_B, TEAM_C):
        conn.execute("INSERT OR REPLACE INTO teams (id, name, short_name, raw_json, updated_at)"
                     " VALUES (?,?,?,'{}','2026-09-12T09:00:00Z')",
                     (team_id, f"Team {team_id}", f"T{team_id}"))
    for position_id, name in ((1, "Goalkeeper"), (2, "Defender"), (3, "Midfielder"),
                              (4, "Forward")):
        conn.execute("INSERT OR REPLACE INTO positions (id, singular_name, raw_json, updated_at)"
                     " VALUES (?,?,?,?)", (position_id, name, "{}", "2026-09-12T09:00:00Z"))
    for fixture_id, event, home, away in fixtures_spec:
        conn.execute(
            "INSERT OR REPLACE INTO fixtures (id, event, team_h, team_a, kickoff_time, started,"
            " finished, raw_json, updated_at) VALUES (?,?,?,?,?,0,0,'{}','2026-09-12T09:00:00Z')",
            (fixture_id, event, home, away, "2026-09-12T14:00:00Z"))
        for team_id, opponent_id in ((home, away), (away, home)):
            conn.execute(
                "INSERT INTO team_fixture_projections (projection_run_id, fixture_id, event,"
                " team_id, opponent_id, venue, payload_json, model_version, generated_at)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                (runs["team"], fixture_id, event, team_id, opponent_id,
                 "home" if team_id == home else "away",
                 json.dumps({"expected_goals_for": 1.4, "expected_goals_against": 1.1}),
                 "v1.0.0", "2026-09-12T09:00:00Z"))
        for team_id, opponent_id in ((home, away), (away, home)):
            pid = team_id * 10 + 1
            # OR REPLACE: a DGW team appears in two fixtures and must not be inserted twice.
            conn.execute(
                "INSERT OR REPLACE INTO players (id, web_name, team_id, element_type,"
                " first_seen_at, last_seen_at, is_active, raw_json, updated_at)"
                " VALUES (?,?,?,2,'2026-09-12T09:00:00Z','2026-09-12T09:00:00Z',1,'{}',"
                "'2026-09-12T09:00:00Z')", (pid, f"P{pid}", team_id))
            conn.execute(
                "INSERT INTO player_fixture_xpts_projections (projection_run_id, player_id,"
                " fixture_id, event, team_id, opponent_id, position, minutes_run_id,"
                " team_run_id, rate_run_id, payload_json, model_version,"
                " scoring_rules_version, generated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (runs["xpts"], pid, fixture_id, event, team_id, opponent_id, "DEF",
                 runs["minutes"], runs["team"], 0,
                 json.dumps({"core_xpts": 3.0,
                             "bonus_xpts": (bonus_xpts or {}).get(fixture_id, 0.0)}),
                 "v1.0.0", "v1.0.0", "2026-09-12T09:00:00Z"))
            conn.execute(
                "INSERT INTO frozen_predictions (projection_run_id, kind, player_id, fixture_id,"
                " event, payload_json, model_version, generated_at) VALUES (?,?,?,?,?,?,?,?)",
                (runs["minutes"], analytics.MINUTES_V1_KIND, pid, fixture_id, event,
                 json.dumps({"p_start": 1.0, "p_cameo": 0.0, "p_available": 1.0,
                             "p_80_given_start": 1.0, "expected_minutes_if_start": 90.0,
                             "expected_minutes_if_cameo": 15.0, "p_60_given_cameo": 0.04}),
                 _LEGACY_MINUTES, "2026-09-12T09:00:00Z"))
    conn.commit()
    return conn, runs


def test_fixture_from_another_event_is_excluded(tmp_path):
    """§18: building event N must contain only event N's fixtures.

    DISCRIMINATING: ``load_fixture_inputs`` declares an ``event`` argument.  Before the PE-4
    repair it read every projection row of the run regardless of that argument, so a fixture
    recorded against event N+1 was returned as part of event N's inputs.
    """

    conn, runs = _projection_db(tmp_path, fixtures_spec=[
        (FIXTURE_A, EVENT_A, TEAM_A, TEAM_B),
        (FIXTURE_OTHER_EVENT, EVENT_B, TEAM_C, TEAM_B)])
    fixtures = mc.load_fixture_inputs(conn, event=EVENT_A, xpts_run_id=runs["xpts"],
                                      minutes_run_id=runs["minutes"], team_run_id=runs["team"])
    assert sorted(fixtures) == [FIXTURE_A], "a fixture from another event leaked in"
    assert all(int(f["event"]) == EVENT_A for f in fixtures.values())


def test_both_events_are_returned_when_each_is_requested(tmp_path):
    """The filter selects the requested event; it does not merely drop the second."""

    conn, runs = _projection_db(tmp_path, fixtures_spec=[
        (FIXTURE_A, EVENT_A, TEAM_A, TEAM_B),
        (FIXTURE_OTHER_EVENT, EVENT_B, TEAM_C, TEAM_B)])
    for event, expected in ((EVENT_A, [FIXTURE_A]), (EVENT_B, [FIXTURE_OTHER_EVENT])):
        fixtures = mc.load_fixture_inputs(
            conn, event=event, xpts_run_id=runs["xpts"],
            minutes_run_id=runs["minutes"], team_run_id=runs["team"])
        assert sorted(fixtures) == expected


def test_all_fixtures_of_the_requested_event_are_returned(tmp_path):
    """A real DGW: two fixtures in the SAME event are both returned."""

    conn, runs = _projection_db(tmp_path, fixtures_spec=[
        (FIXTURE_A, EVENT_A, TEAM_A, TEAM_B),
        (FIXTURE_B, EVENT_A, TEAM_A, TEAM_C)])
    fixtures = mc.load_fixture_inputs(conn, event=EVENT_A, xpts_run_id=runs["xpts"],
                                      minutes_run_id=runs["minutes"], team_run_id=runs["team"])
    assert sorted(fixtures) == [FIXTURE_A, FIXTURE_B]


def test_duplicate_fixture_identity_cannot_be_double_counted(tmp_path):
    """§17: fixture identity is the Mapping key, and the schema refuses a duplicate row.

    ``load_fixture_inputs`` returns ``{fixture_id: fixture}``, so two fixtures cannot collapse
    or double-count.  A duplicate PLAYER row inside one fixture is refused by the writer's
    ``UNIQUE (projection_run_id, player_id, fixture_id)`` constraint, which is what stops a
    player being appended twice to his own side.
    """

    conn, runs = _projection_db(tmp_path, fixtures_spec=[(FIXTURE_A, EVENT_A, TEAM_A, TEAM_B)])
    fixtures = mc.load_fixture_inputs(conn, event=EVENT_A, xpts_run_id=runs["xpts"],
                                      minutes_run_id=runs["minutes"], team_run_id=runs["team"])
    assert list(fixtures) == [FIXTURE_A]
    side = fixtures[FIXTURE_A]["sides"][0]
    assert len({p["player_id"] for p in side["players"]}) == len(side["players"]), \
        "a player appears twice in one side"

    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO player_fixture_xpts_projections (projection_run_id, player_id,"
            " fixture_id, event, team_id, opponent_id, position, minutes_run_id, team_run_id,"
            " rate_run_id, payload_json, model_version, scoring_rules_version, generated_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (runs["xpts"], int(fixtures[FIXTURE_A]["sides"][0]["players"][0]["player_id"]),
             FIXTURE_A, EVENT_A, TEAM_A, TEAM_B,
             "DEF", runs["minutes"], runs["team"], 0, "{}", "v1.0.0", "v1.0.0",
             "2026-09-12T09:00:00Z"))


# ===========================================================================
# 19-20. EXISTING SIDE DATA
# ===========================================================================


def test_soft_expected_bonus_sums_across_a_dgw(tmp_path):
    """§19: the EXISTING soft path sums bonus_xpts across the player's event fixtures."""

    from fpl_brain import manager_worlds as mw
    # The projections are IMMUTABLE by trigger, so the bonus is written at INSERT time.
    conn, runs = _projection_db(tmp_path, fixtures_spec=[
        (FIXTURE_A, EVENT_A, TEAM_A, TEAM_B),
        (FIXTURE_B, EVENT_A, TEAM_A, TEAM_C)],
        bonus_xpts={FIXTURE_A: 0.25, FIXTURE_B: 0.40})
    dgw = TEAM_A * 10 + 1
    bonuses = mw.expected_bonus_by_player(conn, xpts_run_id=runs["xpts"], event=EVENT_A)
    assert bonuses[dgw] == pytest.approx(0.65), "the DGW soft bonus must be the two sums"


def test_blank_player_soft_bonus_is_zero():
    """§19: an absent map entry is read as 0 by the existing attachment, not as a gap."""

    from fpl_brain import manager_worlds as mw
    matrix = {"worlds": 1, "player_ids": [1, 2], "minutes": {1: [0.0], 2: [90.0]},
              "core": {1: [0.0], 2: [5.0]}}
    mw.with_expected_bonus(matrix, {2: 0.3})
    assert matrix["expected_bonus"] == {1: 0.0, 2: 0.3}


def test_role_actionability_is_not_summed_for_a_dgw():
    """§20: role actionability is an event/player BOOLEAN, never an arithmetic aggregate."""

    from fpl_brain import manager_worlds as mw
    matrix = {"worlds": 1, "player_ids": [1, 2], "minutes": {1: [0.0], 2: [90.0]},
              "core": {1: [0.0], 2: [5.0]}}
    mw.with_role_actionability(matrix, {2: True})
    assert matrix["role_actionability"] == {1: False, 2: True}
    assert all(isinstance(v, bool) for v in matrix["role_actionability"].values())


# ===========================================================================
# 24. VERSION / RNG INVARIANTS
# ===========================================================================


def test_model_version_and_config_hash_are_unchanged():
    assert mc.MONTE_CARLO_MODEL_VERSION == "mc_v1.3.0"
    config = mc.MonteCarloConfig(simulations=100, seed=987654)
    assert mc.MonteCarloConfig(simulations=100, seed=987654).config_hash() == config.config_hash()


def test_capture_is_pure_and_does_not_consume_randomness():
    """The world capture must not change a single summary byte."""

    fixture_a, fixture_b = _event_fixtures()
    fixtures = {FIXTURE_A: copy.deepcopy(fixture_a), FIXTURE_B: copy.deepcopy(fixture_b)}
    config = mc.MonteCarloConfig(simulations=60, seed=4242)
    with_capture = mc.simulate(copy.deepcopy(fixtures), config, RULES,
                               capture_player_ids=[DGW_PLAYER, BLANK_PLAYER])
    without = mc.simulate(copy.deepcopy(fixtures), config, RULES)
    assert without["world_matrix"] is None
    assert with_capture["summaries"] == without["summaries"]


def test_aggregation_is_deterministic_across_repeated_runs():
    fixture_a, fixture_b = _event_fixtures()
    first = _matrix(_simulate({FIXTURE_A: fixture_a, FIXTURE_B: fixture_b},
                              capture=[DGW_PLAYER, BLANK_PLAYER]))
    fixture_a, fixture_b = _event_fixtures()
    second = _matrix(_simulate({FIXTURE_A: fixture_a, FIXTURE_B: fixture_b},
                               capture=[DGW_PLAYER, BLANK_PLAYER]))
    assert first["core"] == second["core"]
    assert first["minutes"] == second["minutes"]


def test_capture_player_ids_are_integer_normalised_and_sorted():
    fixture_a, _fixture_b = _event_fixtures()
    result = _simulate({FIXTURE_A: fixture_a}, capture=[str(BLANK_PLAYER), A_ONLY_PLAYER])
    matrix = result["world_matrix"]
    assert matrix["player_ids"] == sorted({A_ONLY_PLAYER, BLANK_PLAYER})
    assert all(isinstance(pid, int) for pid in matrix["player_ids"])
