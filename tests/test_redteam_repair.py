"""Red-team P0 regression tests: instrument integrity and one parameter vector."""

from __future__ import annotations

import copy
import random
import statistics

import pytest

from fpl_brain import joint_minutes as jm
from fpl_brain import monte_carlo as mc
from test_monte_carlo import RULES, _fixture, _finish, _run


def test_core_linear_has_real_per_draw_variance():
    """The core_linear instrument must carry the variance of its own draws.

    The red-team defect was that core_linear had no accumulated variance, so its
    standard error was 0 and its standardised error was silently 0.00.
    """

    _, result, _ = _run(simulations=2000, fixture=_fixture(cameo=0.0))
    stds = [s["mc_component_std"]["core_linear"] for s in result["summaries"]]
    assert any(value > 0.0 for value in stds), "core_linear must have real per-draw variance"
    # appearance + goal + assist + CS + GC + yellow, all able to vary
    for summary in result["summaries"]:
        if summary["mean_minutes"] > 0:
            assert summary["mc_component_std"]["core_linear"] >= 0.0
    # core_linear must not simply equal the derived (core - defcon - save) with
    # zero spread: at least one player must show a non-zero standard error.
    assert max(stds) > 0.1


def test_zero_variance_with_nonzero_error_is_a_hard_fail():
    """A non-zero mismatch with zero simulated variance must FAIL, never map to z=0."""

    fixture = _finish(_fixture(cameo=0.0))
    config = mc.MonteCarloConfig(simulations=200, seed=5)
    result = mc.simulate({100: fixture}, config, RULES)
    broken = copy.deepcopy(result)
    # Force the defect: a non-zero error with zero variance.
    for summary in broken["summaries"]:
        summary["mc_component_std"]["goal"] = 0.0
        summary["mean_reconciliation_error"]["goal"] = 0.5
        summary["standardised_error"]["goal"] = float("inf")
        summary["zero_variance_mismatch"] = ["goal"]
    report = mc.readiness_summary({100: fixture}, broken, config)
    assert report["status"] == "FAIL"
    assert any("ZERO_VARIANCE_MISMATCH" in reason for reason in report["fail_reasons"])


def test_zero_variance_with_zero_error_is_allowed():
    fixture = _finish(_fixture(cameo=0.0))
    config = mc.MonteCarloConfig(simulations=200, seed=5)
    result = mc.simulate({100: fixture}, config, RULES)
    clean = copy.deepcopy(result)
    for summary in clean["summaries"]:
        summary["mc_component_std"]["goal"] = 0.0
        summary["mean_reconciliation_error"]["goal"] = 0.0
        summary["standardised_error"]["goal"] = 0.0
        summary["zero_variance_mismatch"] = []
    report = mc.readiness_summary({100: fixture}, clean, config)
    assert not any("ZERO_VARIANCE_MISMATCH" in reason for reason in report["fail_reasons"])


def test_minutes_and_mc_share_one_kernel_parameter_vector():
    """The frozen joint primitives must be the ones the MC actually samples."""

    players = [
        {"player_id": i, "position": "GKP" if i == 1 else "MID", "p_start": 1.0 if i <= 11 else 0.0,
         "p_available": 1.0, "exit_propensity": 0.3 + 0.01 * i, "entry_propensity": 0.0 if i <= 11 else 0.2,
         "expected_minutes_if_cameo": 15.0}
        for i in range(1, 26)
    ]
    profile = {"p_sub_count": [0.0, 0.0, 0.0, 0.0, 0.2, 0.8],
               "event_time_bands": [[0, 29], [30, 44], [45, 59], [60, 74], [75, 89]],
               "event_time_masses": [0.0, 0.0, 0.2, 1.9, 2.0],
               "gk_event_mass": 0.0}
    config = jm.JointMinutesConfig(integration_draws=400)

    # Integration uses the primitives directly.
    marginals = jm.integrate_side_marginals(players, profile, config, fixture_id=1, team_id=1)

    # A frozen row carrying the same primitives, as minutes_v1.5.1 stores them.
    frozen = [
        {
            "player_id": p["player_id"],
            "position": p["position"],
            "minutes": {
                "p_start": 0.0, "p_cameo": 0.0, "p_available": 0.0, "p_60_given_start": 0.0,
                "p_80_given_start": 0.0, "expected_minutes_if_start": 0.0,
                "expected_minutes_if_cameo": 0.0,
                "joint_start_target": p["p_start"],
                "joint_availability": p["p_available"],
                "joint_exit_propensity": p["exit_propensity"],
                "joint_entry_propensity": p["entry_propensity"],
                "joint_position": p["position"],
                "joint_expected_minutes_if_cameo": p["expected_minutes_if_cameo"],
            },
            "payload": {"fixture_xg_per90": 0.2, "fixture_xa_per90": 0.2, "yellow_per90": 0.0,
                        "defcon_actions_per90": 0.0,
                        "save_model": {"saves_per90_posterior": 0.0, "pressure_multiplier": 1.0}},
        }
        for p in players
    ]
    side = {"team_id": 1, "players": frozen, "substitution_profile": profile}
    rng = random.Random(9)
    # The MC adapter must read the frozen primitives verbatim: with p_start=0 and
    # p_available=0 in the legacy fields, reconstruction would give zero start and
    # zero entry propensity, so any minutes at all prove the primitives were used.
    worlds = [mc._sample_side_world(side, rng, mc.MonteCarloConfig()) for _ in range(50)]
    assert all(w["gk_starters"] == 1 and len(w["starters"]) == 11 for w in worlds)
    assert any(sum(w["minutes"].values()) > 0 for w in worlds)
    assert abs(sum(worlds[0]["minutes"].values()) - 990.0) < 1e-9
    # And the two consumers agree on the start marginals they imply.
    integrated = {pid: m["p_start"] for pid, m in marginals.items()}
    assert integrated[1] > 0.99 and integrated[25] == 0.0
