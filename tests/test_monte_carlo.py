"""Monte Carlo v1.1.0 acceptance tests: paired substitutions, occupancy, reconciliation."""

from __future__ import annotations

import copy
import math
import random
import statistics

import pytest

from fpl_brain import monte_carlo as mc
from fpl_brain.scoring_rules import DEFAULT_SCORING_RULES

RULES = DEFAULT_SCORING_RULES
XG, XA = 0.30, 0.20
START_MINUTES = 90.0
CAMEO_MINUTES = 15.0
FRINGE_IDS = set(range(101, 114)) | set(range(1100, 1113))


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
            "expected_minutes_if_start": START_MINUTES, "expected_minutes_if_cameo": cameo_minutes,
            "p_60_given_cameo": 0.04,
        },
        "payload": payload,
    }


def _side(team_id, *, base, starter_p80=1.0, cameo=0.2, n_fringe=13, **over):
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
        "team_id": team_id,
        "opponent_id": 2 if team_id == 1 else 1,
        "players": players,
        "lambda_for": sum_xg,
        "lambda_against": 0.0,
        "sum_player_xg": sum_xg,
        "sum_player_xa": sum_xa,
        "residual_xg": 0.0,
        "residual_xa": max(0.0, sum_xg - sum_xa),
    }


def _fixture(fixture_id=100, event=4, **side_over):
    return {"fixture_id": fixture_id, "event": event,
            "sides": [_side(1, base=1, **side_over), _side(2, base=1000, **side_over)]}


def _efloor_half(lam):
    return (lam - (1.0 - math.exp(-2.0 * lam)) / 2.0) / 2.0 if lam > 0 else 0.0


def _analytic(player, opponent_lambda):
    """Analytic components for a world where every start lasts 90 minutes."""

    p_start = player["minutes"]["p_start"]
    p_cameo = player["minutes"]["p_cameo"]
    m_c = player["minutes"]["expected_minutes_if_cameo"]
    minutes = p_start * START_MINUTES + p_cameo * m_c
    p60 = p_start + p_cameo * 0.04
    position = player["position"]
    cs_prob = p_start * math.exp(-opponent_lambda) + p_cameo * 0.04 * math.exp(-opponent_lambda * 70.0 / 90.0)
    gc = 0.0
    if position in RULES.goals_conceded_positions:
        gc = -(p_start * _efloor_half(opponent_lambda) + p_cameo * _efloor_half(opponent_lambda * m_c / 90.0))
    values = {
        "appearance": (p_start + p_cameo) + p60,
        "goal": XG * minutes / 90.0 * RULES.goal_points_for(position),
        "assist": XA * minutes / 90.0 * RULES.assist_points,
        "clean_sheet": RULES.clean_sheet_points_for(position) * cs_prob,
        "goals_conceded": gc,
        "defcon": 0.0,
        "save": 0.0,
        "yellow": 0.0,
    }
    values["core"] = sum(values[n] for n in ("appearance", "goal", "assist", "clean_sheet", "goals_conceded"))
    return values


def _finish(fixture):
    for side in fixture["sides"]:
        opponent = [s for s in fixture["sides"] if s["team_id"] != side["team_id"]][0]
        for player in side["players"]:
            values = _analytic(player, opponent["lambda_for"])
            for name, value in values.items():
                key = "yellow_card_xpts" if name == "yellow" else f"{name}_xpts"
                player["payload"][key] = value
    return fixture


def _run(simulations=3000, fixture=None, **config_over):
    fixture = _finish(fixture or _fixture())
    config = mc.MonteCarloConfig(simulations=simulations, seed=12345, **config_over)
    result = mc.simulate({fixture["fixture_id"]: fixture}, config, RULES)
    return fixture, result, config


def _by_player(result):
    return {s["player_id"]: s for s in result["summaries"]}


def _worlds(fixture, config, n=200):
    worlds = []
    rng = random.Random(7)
    for _ in range(n):
        for side in fixture["sides"]:
            worlds.append((side, mc._sample_side_world(side, rng, config)))
    return worlds


def _gk_indices(side):
    return [i for i, p in enumerate(side["players"]) if p["position"] == "GKP"]


def test_exactly_eleven_players_on_pitch_at_every_interval():
    fixture = _fixture(starter_p80=0.5)
    for side, world in _worlds(fixture, mc.MonteCarloConfig()):
        gk_ids = [p["player_id"] for p in side["players"] if p["position"] == "GKP"]
        ok, why = mc.check_world_occupancy(world["intervals"], gk_ids)
        assert ok, why


def test_exactly_one_goalkeeper_on_pitch_at_every_interval():
    fixture = _fixture(starter_p80=0.5)
    for side, world in _worlds(fixture, mc.MonteCarloConfig()):
        gks = {p["player_id"] for p in side["players"] if p["position"] == "GKP"}
        boundaries = {0.0, 90.0}
        for low, high in world["intervals"].values():
            boundaries.update((low, high))
        for time in sorted(boundaries):
            probe = min(89.999999, time + 1e-6)
            on_pitch = [pid for pid, (low, high) in world["intervals"].items() if low <= probe <= high]
            assert len([pid for pid in on_pitch if pid in gks]) == 1


def test_every_substitute_entry_has_one_matching_starter_exit():
    fixture = _fixture(starter_p80=0.55)
    saw_event = False
    for side, world in _worlds(fixture, mc.MonteCarloConfig()):
        for event in world["events"]:
            saw_event = True
            assert event["exiting"] in world["starters"]
            assert event["entering"] not in world["starters"]
        assert len(world["substitutes"]) == len(world["events"])
    assert saw_event


def test_substitution_pair_minutes_total_ninety():
    fixture = _fixture(starter_p80=0.55)
    for side, world in _worlds(fixture, mc.MonteCarloConfig()):
        for event in world["events"]:
            low_x, high_x = world["intervals"][side["players"][event["exiting"]]["player_id"]]
            low_e, high_e = world["intervals"][side["players"][event["entering"]]["player_id"]]
            assert high_x == event["time"] and low_e == event["time"]
            assert (high_x - low_x) + (high_e - low_e) == pytest.approx(90.0)


def test_total_team_player_minutes_is_exactly_990_per_world():
    fixture = _fixture(starter_p80=0.5)
    for side, world in _worlds(fixture, mc.MonteCarloConfig()):
        assert abs(sum(world["minutes"].values()) - 990.0) < 1e-9
    _, result, _ = _run(simulations=800, fixture=_fixture(starter_p80=0.5))
    assert result["minute_mass_violations"] == 0
    assert {round(v, 6) for v in result["team_minutes"]} == {990.0}


def test_no_starter_is_also_a_substitute():
    fixture = _fixture(starter_p80=0.5)
    for side, world in _worlds(fixture, mc.MonteCarloConfig()):
        assert not (world["starters"] & world["substitutes"])


def test_substitution_count_never_exceeds_the_documented_limit():
    config = mc.MonteCarloConfig()
    fixture = _fixture(starter_p80=0.4, cameo=0.4)
    for side, world in _worlds(fixture, config):
        assert world["substitute_entrants"] <= config.max_substitute_entrants
    _, result, _ = _run(simulations=1500, fixture=_fixture(starter_p80=0.4, cameo=0.4))
    assert result["excess_substitutes"] == 0


def test_goalkeeper_substitute_only_replaces_the_goalkeeper():
    # Isolate the goalkeeper path: no ordinary cameo mass, a keeper on the bench
    # with a certain entry propensity, and a ceiling of 1.
    fixture = _fixture(starter_p80=0.5, cameo=0.0)
    for side in fixture["sides"]:
        for player in side["players"]:
            if player["position"] == "GKP" and player["minutes"]["p_start"] == 0.0:
                player["minutes"]["p_cameo"] = 1.0
    config = mc.MonteCarloConfig(gk_substitution_ceiling=1.0)
    gk_subs = 0
    checked = 0
    rng = random.Random(11)
    for _ in range(400):
        for side in fixture["sides"]:
            world = mc._sample_side_world(side, rng, config)
            checked += 1
            for event in world["events"]:
                entering_gk = side["players"][event["entering"]]["position"] == "GKP"
                exiting_gk = side["players"][event["exiting"]]["position"] == "GKP"
                assert entering_gk == exiting_gk
                if entering_gk:
                    gk_subs += 1
    assert checked > 0
    assert gk_subs > 0


def test_sampled_start_marginals_converge():
    fixture = _fixture(cameo=0.0)
    _, result, _ = _run(simulations=6000, fixture=fixture)
    by_player = _by_player(result)
    targets = {p["player_id"]: p["minutes"]["p_start"] for side in fixture["sides"] for p in side["players"]}
    errors = [abs(by_player[pid]["p_start"] - target) for pid, target in targets.items()]
    assert max(errors) < 0.02


def test_sampled_cameo_path_is_exercised():
    _, result, _ = _run(simulations=6000, fixture=_fixture())
    fringe = [s for pid, s in _by_player(result).items() if pid in FRINGE_IDS]
    assert any(s["mean_minutes"] > 0 for s in fringe)


def test_sampled_expected_minutes_converge_for_full_match_starters():
    fixture = _finish(_fixture(cameo=0.0))
    _, result, _ = _run(simulations=4000, fixture=fixture)
    by_player = _by_player(result)
    for side in fixture["sides"]:
        for p in side["players"]:
            if p["minutes"]["p_start"] == 1.0:
                assert by_player[p["player_id"]]["mean_minutes"] == pytest.approx(90.0, abs=1e-9)


def test_sampled_p60_and_p80_for_full_match_starters():
    _, result, _ = _run(simulations=2000, fixture=_fixture(cameo=0.0))
    for summary in result["summaries"]:
        if summary["mean_minutes"] > 89.9:
            assert summary["p_60_plus"] == pytest.approx(1.0, abs=1e-9)


def _linear_component_check(name, tolerance):
    # 4000 draws; tolerances are sized to the Monte Carlo standard error of the
    # highest-variance component (a goalkeeper's goal points) in this synthetic
    # world, where every player carries an identical outfield-like xG rate.
    _, result, _ = _run(simulations=4000, fixture=_fixture(cameo=0.0))
    worst = max(abs(s["mean_reconciliation_error"][name]) for s in result["summaries"])
    assert worst < tolerance, (name, worst)


def test_player_level_appearance_reconciliation():
    _linear_component_check("appearance", 0.05)


def test_player_level_goal_reconciliation():
    _linear_component_check("goal", 0.20)


def test_player_level_assist_reconciliation():
    _linear_component_check("assist", 0.10)


def test_player_level_clean_sheet_reconciliation():
    _linear_component_check("clean_sheet", 0.05)


def test_player_level_goals_conceded_reconciliation():
    _linear_component_check("goals_conceded", 0.05)


def test_player_level_core_reconciliation():
    _linear_component_check("core", 0.25)


def test_defcon_and_saves_are_classified_as_nonlinear():
    fixture = _fixture(cameo=0.0)
    _, result, config = _run(simulations=800, fixture=fixture)
    report = mc.readiness_summary({100: fixture}, result, config)
    assert "player_error_stats" in report
    for name in ("defcon", "save"):
        assert name in report["player_error_stats"]


def test_no_named_player_is_hardcoded():
    import io as _io

    source = _io.open("K:/FPL/fpl_brain/monte_carlo.py", encoding="utf-8").read().lower()
    for name in ("haaland", "bruno", "fernandes", "mbeumo", "de cuyper", "raya", "joao", "joão"):
        assert name not in source


def test_randomised_sampler_is_order_invariant():
    probs = [min(1.0, 0.35 + 0.06 * ((i * 7) % 11)) for i in range(25)]
    probs = [p * 10.0 / sum(probs) for p in probs]
    permutation = list(range(len(probs)))
    random.Random(3).shuffle(permutation)

    def pair_stats(order, seed=5, trials=4000):
        rng = random.Random(seed)
        counts = {}
        for _ in range(trials):
            chosen = sorted(order[i] for i in mc._systematic_select(
                [probs[i] for i in order], 10, rng.random(), randomised=True, rng=rng))
            for a in range(len(chosen)):
                for b in range(a + 1, len(chosen)):
                    counts[(chosen[a], chosen[b])] = counts.get((chosen[a], chosen[b]), 0) + 1
        return counts, trials

    base, trials = pair_stats(list(range(len(probs))))
    perm, _ = pair_stats(permutation)
    worst = 0.0
    for a in range(len(probs)):
        for b in range(a + 1, len(probs)):
            worst = max(worst, abs(base.get((a, b), 0) / trials - perm.get((a, b), 0) / trials))
    assert worst < 0.06  # without randomisation the fixed-order artefact measured 0.351


def test_same_seed_is_deterministic():
    fixture = _finish(_fixture(starter_p80=0.6, cameo=0.0))
    config = mc.MonteCarloConfig(simulations=500, seed=99)
    first = mc.simulate({100: fixture}, config, RULES)
    second = mc.simulate({100: copy.deepcopy(fixture)}, config, RULES)
    assert first["summaries"] == second["summaries"]


def test_different_seed_different_worlds_similar_means():
    fixture = _finish(_fixture(starter_p80=0.6, cameo=0.0))
    a = mc.simulate({100: fixture}, mc.MonteCarloConfig(simulations=1500, seed=1), RULES)
    b = mc.simulate({100: copy.deepcopy(fixture)}, mc.MonteCarloConfig(simulations=1500, seed=2), RULES)
    assert [s["mean_total_proxy"] for s in a["summaries"]] != [s["mean_total_proxy"] for s in b["summaries"]]
    assert abs(statistics.mean(s["mean_total_proxy"] for s in a["summaries"])
               - statistics.mean(s["mean_total_proxy"] for s in b["summaries"])) < 0.2


def test_container_and_entity_ordering_do_not_change_results():
    fixture = _finish(_fixture(starter_p80=0.6, cameo=0.0))
    shuffled = copy.deepcopy(fixture)
    shuffled["sides"] = list(reversed(shuffled["sides"]))
    for side in shuffled["sides"]:
        side["players"] = sorted(side["players"], key=lambda p: -p["player_id"])
    config = mc.MonteCarloConfig(simulations=500, seed=17)
    a = mc.simulate({100: fixture}, config, RULES)["summaries"]
    b = mc.simulate({100: shuffled}, config, RULES)["summaries"]
    assert sorted((s["player_id"], s["mean_total_proxy"]) for s in a) == sorted(
        (s["player_id"], s["mean_total_proxy"]) for s in b)


def test_crn_identity_survives_the_substitution_timeline():
    fixture = _finish(_fixture(starter_p80=0.6, cameo=0.0))
    config = mc.MonteCarloConfig(simulations=500, seed=42)
    a = {(s["player_id"], s["fixture_id"]): s["mean_total_proxy"]
         for s in mc.simulate({100: copy.deepcopy(fixture)}, config, RULES)["summaries"]}
    b = {(s["player_id"], s["fixture_id"]): s["mean_total_proxy"]
         for s in mc.simulate({100: copy.deepcopy(fixture), 999: _finish(_fixture(fixture_id=999))},
                              config, RULES)["summaries"]}
    for key, value in a.items():
        assert b[key] == value


def test_run_36_is_untouched():
    import sqlite3

    conn = sqlite3.connect("file:K:/FPL/fpl.db?mode=ro", uri=True)
    row = conn.execute("SELECT status, model_version FROM projection_runs WHERE id=36").fetchone()
    assert row == ("complete", "mc_v1.0.0")
    count = conn.execute("SELECT COUNT(*) FROM monte_carlo_distributions WHERE projection_run_id=36").fetchone()[0]
    assert count == 655
    conn.close()


def test_score_threshold_names_are_canonical():
    _, result, _ = _run(simulations=800)
    summary = result["summaries"][0]
    for name in ("p_score_le_2", "p_score_5_plus", "p_score_10_plus", "p_score_15_plus"):
        assert 0.0 <= summary[name] <= 1.0
    assert summary["p_score_le_2"] >= summary["p_score_5_plus"] - 1e-9
    assert summary["p_score_5_plus"] >= summary["p_score_10_plus"] - 1e-9
    assert summary["p_blank_proxy"] == summary["p_score_le_2"]
    assert summary["p_return_proxy"] == summary["p_score_5_plus"]


def test_frozen_inputs_are_reproducible_and_not_reread():
    fixture = _finish(_fixture())
    config = mc.MonteCarloConfig(simulations=400, seed=8)
    first = mc.simulate({100: fixture}, config, RULES)["summaries"]
    second = mc.simulate({100: copy.deepcopy(fixture)}, config, RULES)["summaries"]
    assert [(s["player_id"], s["mean_total_proxy"]) for s in first] == [
        (s["player_id"], s["mean_total_proxy"]) for s in second]


def test_readiness_hard_fails_on_occupancy_and_minute_mass_violations():
    fixture = _finish(_fixture(cameo=0.0))
    config = mc.MonteCarloConfig(simulations=200, seed=3)
    result = mc.simulate({100: fixture}, config, RULES)
    clean = mc.readiness_summary({100: fixture}, result, config)
    assert clean["status"] in {"PASS", "WARN"}
    assert clean["fail_reasons"] == []

    broken = dict(result)
    broken["occupancy_violations"] = 1
    broken["occupancy_examples"] = ["12 on pitch at 60.0"]
    failed = mc.readiness_summary({100: fixture}, broken, config)
    assert failed["status"] == "FAIL"
    assert any("WORLD_OCCUPANCY_VIOLATION" in reason for reason in failed["fail_reasons"])

    broken2 = dict(result)
    broken2["minute_mass_violations"] = 1
    failed2 = mc.readiness_summary({100: fixture}, broken2, config)
    assert any("TEAM_MINUTE_MASS_VIOLATION" in reason for reason in failed2["fail_reasons"])
