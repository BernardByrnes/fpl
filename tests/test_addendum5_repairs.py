"""Addendum 5 red-team remediation acceptance tests.

Covers the narrow zero-variance materiality rule, the frozen kernel primitive
vector, expectation-preserving scorer/assist calibration, per-draw component
variances, the DefCon/save minute mixture, the every-row probability gate, the
repaired calibration evaluator, full-band substitution timing and CRN category
substream invariance.
"""

from __future__ import annotations

import math

import pytest

from fpl_brain import analytics, calibration, joint_minutes, monte_carlo as mc, xpts
from fpl_brain.database import connect_database
from fpl_brain.models import FixtureRecord, PlayerGameweekRecord
from fpl_brain import repositories as repo
from fpl_brain.scoring_rules import DEFAULT_SCORING_RULES as RULES

from test_minutes_model import _world
from test_monte_carlo import _fixture, _finish


# ---------------------------------------------------------------------------
# Section 1 — zero-variance materiality boundaries.
# ---------------------------------------------------------------------------


def test_zero_variance_materiality_boundaries():
    materiality = 0.01
    assert mc._classify_standardised(0.0, 0.0, 1000, materiality) == (0.0, "ok")
    assert mc._classify_standardised(0.5 * materiality, 0.0, 1000, materiality)[1] == "below_materiality"
    z_equal, status_equal = mc._classify_standardised(materiality, 0.0, 1000, materiality)
    assert status_equal == "below_materiality" and z_equal == 0.0
    z_above, status_above = mc._classify_standardised(1.0000001 * materiality, 0.0, 1000, materiality)
    assert status_above == "mismatch" and z_above == float("inf")
    # Any real variance is classified normally, whatever the size.
    assert mc._classify_standardised(7.0, 2.0, 100, materiality)[1] == "ok"


def _fake_summary(pid=1, **over):
    base = {key: 0.5 for key in mc.PROBABILITY_KEYS}
    base.update({
        "player_id": pid,
        "fixture_id": 1,
        "simulations": 1000,
        "mean_reconciliation_error": {
            name: 0.0
            for name in ("appearance", "goal", "assist", "clean_sheet", "goals_conceded",
                         "defcon", "save", "yellow", "core_linear", "core")
        },
        "standardised_error": {},
        "zero_variance_mismatch": [],
        "zero_variance_below_materiality": [],
        "mc_component_std": {},
        "analytic_components": {},
    })
    base.update(over)
    return base


def test_material_zero_variance_mismatch_hard_fails():
    result = {
        "summaries": [_fake_summary(zero_variance_mismatch=["goal"])],
        "team_minutes": [],
        "calibration": [],
    }
    gate = mc.readiness_summary({}, result, mc.MonteCarloConfig())
    assert gate["status"] == "FAIL"
    assert any("ZERO_VARIANCE_MISMATCH" in reason for reason in gate["fail_reasons"])


def test_microscopic_zero_variance_is_recorded_not_failed():
    result = {
        "summaries": [_fake_summary(zero_variance_below_materiality=["assist"])],
        "team_minutes": [],
        "calibration": [],
    }
    gate = mc.readiness_summary({}, result, mc.MonteCarloConfig())
    assert gate["status"] != "FAIL"
    assert gate["zero_variance_below_materiality_components"] == ["assist"]
    assert any("ZERO_VARIANCE_BELOW_MATERIALITY" in reason for reason in gate["warn_reasons"])


# ---------------------------------------------------------------------------
# Section 2 — one exact kernel primitive vector, no reconstruction for v1.5.1.
# ---------------------------------------------------------------------------


def _primitive_minutes(p_start, p_cameo, p80=1.0, cameo=15.0, position="MID"):
    return {
        "joint_start_target": p_start,
        "joint_availability": 1.0,
        "joint_exit_propensity": 1.0 - p80,
        "joint_entry_propensity": (p_cameo / (1.0 - p_start)) if p_start < 1.0 else 0.0,
        "joint_position": position,
        "joint_expected_minutes_if_cameo": cameo,
        "primitive_source_version": joint_minutes.JOINT_MINUTES_MODEL_VERSION,
        # Deliberately contradictory legacy fields: the primitives must win.
        "p_start": 0.123456,
        "p_cameo": 0.0,
        "p_80_given_start": 0.0,
    }


def test_kernel_reads_primitive_vector_verbatim():
    side = {
        "team_id": 1,
        "players": [
            {"player_id": 7, "position": "MID",
             "payload": {"adjusted_expected_xg": 0.5}, "minutes": _primitive_minutes(0.8, 0.1)},
        ],
        "substitution_profile": {},
    }
    kernel = mc._kernel_players(side, mc.MonteCarloConfig())
    assert kernel[0]["p_start"] == pytest.approx(0.8)
    assert kernel[0]["p_available"] == pytest.approx(1.0)
    assert kernel[0]["exit_propensity"] == pytest.approx(0.0)
    assert kernel[0]["entry_propensity"] == pytest.approx(0.5)
    assert kernel[0]["expected_minutes_if_cameo"] == pytest.approx(15.0)


def test_readiness_fails_on_primitive_reconstruction():
    result = {"summaries": [], "team_minutes": [], "calibration": [], "primitive_reconstruction": 2}
    gate = mc.readiness_summary({}, result, mc.MonteCarloConfig())
    assert gate["status"] == "FAIL"
    assert gate["mc_primitives_verbatim"] is False
    assert any("KERNEL_PRIMITIVE_RECONSTRUCTION" in reason for reason in gate["fail_reasons"])


# ---------------------------------------------------------------------------
# Sections 3-7 — expectation-preserving scorer and assist calibration.
# ---------------------------------------------------------------------------


def _hetero_player(pid, position, p_start, p_cameo, xg90, xa90, *, p80=1.0, team=1):
    minutes = p_start * 90.0 + p_cameo * 15.0
    payload = {
        "adjusted_expected_xg": xg90 * minutes / 90.0,
        "expected_xa": xa90 * minutes / 90.0,
        "expected_fpl_assists": xa90 * minutes / 90.0,
        "goal_xpts": xg90 * minutes / 90.0 * RULES.goal_points_for(position),
        "assist_xpts": xa90 * minutes / 90.0 * RULES.assist_points,
        "fixture_xg_per90": xg90,
        "fixture_xa_per90": xa90,
        "yellow_per90": 0.0,
        "defcon_actions_per90": 0.0,
        "save_model": {"saves_per90_posterior": 0.0, "pressure_multiplier": 1.0},
        "core_xpts": 0.0,
    }
    for name in ("appearance", "clean_sheet", "goals_conceded", "defcon", "save", "yellow"):
        payload[f"{name}_xpts"] = 0.0
    payload["yellow_card_xpts"] = 0.0
    payload["bonus_xpts"] = 0.0
    minutes_payload = {
        **_primitive_minutes(p_start, p_cameo, p80=p80, position=position),
        "expected_minutes": minutes,
        "minute_state_distribution": [
            {"state": "0", "probability": 1.0 - p_start - p_cameo, "conditional_mean_minutes": 0.0},
            {"state": "1-59", "probability": p_cameo, "conditional_mean_minutes": 15.0},
            {"state": "60-79", "probability": 0.0, "conditional_mean_minutes": 0.0},
            {"state": "80-90", "probability": p_start, "conditional_mean_minutes": 90.0},
        ],
    }
    return {"player_id": pid, "position": position, "payload": payload, "minutes": minutes_payload}


def _hetero_side(team_id, base, *, star=True):
    players = [
        _hetero_player(base + 0, "GKP", 1.0, 0.0, 0.0, 0.0, team=team_id),
        _hetero_player(base + 100, "GKP", 0.0, 0.0, 0.0, 0.0, team=team_id),
    ]
    for index in range(1, 11):
        position = "DEF" if index <= 5 else "MID"
        if star and index == 1:
            players.append(_hetero_player(base + index, position, 1.0, 0.0, 1.5, 0.1, team=team_id))
        elif index == 2:
            players.append(_hetero_player(base + index, position, 1.0, 0.0, 0.35, 0.45, team=team_id))
        else:
            players.append(_hetero_player(base + index, position, 1.0, 0.0, 0.10, 0.08, team=team_id))
    # Rotation and substitute-heavy players plus a residual bucket.
    players.append(_hetero_player(base + 200, "MID", 0.4, 0.3, 0.4, 0.3, team=team_id))
    players.append(_hetero_player(base + 201, "FWD", 0.0, 0.35, 0.3, 0.2, team=team_id))
    sum_xg = sum(p["payload"]["adjusted_expected_xg"] for p in players)
    lambda_for = sum_xg * 1.15
    return {
        "team_id": team_id,
        "opponent_id": 2 if team_id == 1 else 1,
        "players": players,
        "substitution_profile": {},
        "lambda_for": lambda_for,
        "lambda_against": 1.3,
        "sum_player_xg": sum_xg,
        "residual_xg": lambda_for - sum_xg,
    }


def _hetero_fixture():
    return {"fixture_id": 500, "event": 4,
            "sides": [_hetero_side(1, base=10), _hetero_side(2, base=1000)]}


def test_heterogeneous_scorer_calibration_is_expectation_preserving():
    fixture = _hetero_fixture()
    config = mc.MonteCarloConfig(simulations=8000, seed=99, calibration_states=6000)
    result = mc.simulate({500: fixture}, config, RULES)
    by_id = {s["player_id"]: s for s in result["summaries"]}
    assert any(abs(s["scorer_target_share"] - s["scorer_achieved_share"]) > 1e-9 for s in result["summaries"])
    for summary in result["summaries"]:
        assert abs(summary["scorer_achieved_share"] - summary["scorer_target_share"]) < 2e-3
        analytic = summary["analytic_components"]["goal"]
        error = summary["mean_reconciliation_error"]["goal"]
        std = summary["mc_component_std"]["goal"]
        tolerance = max(0.03, 4.0 * std / math.sqrt(summary["simulations"]))
        assert abs(error) < tolerance, (summary["player_id"], error, tolerance)
    # The high-rate player must not be under-allocated, and a low-rate player
    # must not be inflated.
    star = by_id[11]
    assert star["analytic_components"]["goal"] > 1.0
    assert star["mean_reconciliation_error"]["goal"] > -0.05
    low = by_id[15]
    assert low["analytic_components"]["goal"] < star["analytic_components"]["goal"] / 5.0
    assert low["mean_reconciliation_error"]["goal"] < 0.05


def test_heterogeneous_assist_calibration_and_self_assist_prohibition():
    fixture = _hetero_fixture()
    config = mc.MonteCarloConfig(simulations=8000, seed=7, calibration_states=6000)
    result = mc.simulate({500: fixture}, config, RULES)
    for summary in result["summaries"]:
        assert abs(summary["assist_achieved_share"] - summary["assist_target_share"]) < 2e-3
        error = summary["mean_reconciliation_error"]["assist"]
        std = summary["mc_component_std"]["assist"]
        tolerance = max(0.05, 4.0 * std / math.sqrt(summary["simulations"]))
        assert abs(error) < tolerance, (summary["player_id"], error, tolerance)
    # Creator with meaningful xA but low xG must keep its assist expectation.
    creator = next(s for s in result["summaries"] if s["player_id"] == 12)
    assert creator["analytic_components"]["assist"] > 0.0
    assert creator["mean_reconciliation_error"]["assist"] > -0.05


def test_scorer_never_assists_own_goal():
    players = [
        {"player_id": 1, "position": "MID", "payload": {}},
        {"player_id": 2, "position": "MID", "payload": {}},
    ]
    entry = {
        "side": {"team_id": 1, "players": players},
        "intervals": {1: (0.0, 90.0), 2: (0.0, 90.0)},
        "goal_times": [45.0],
        "calibration": {
            "scorer": {"weights": [1.0, 0.0], "residual_weight": 0.0},
            "assist": {"weights": [1.0, 1.0], "residual_weight": 0.0},
        },
        "rng_scorer": mc._FastRng(11),
        "rng_assist": mc._FastRng(22),
    }
    draw: dict[int, dict[str, float]] = {}
    mc._allocate_and_score(entry, RULES, mc.MonteCarloConfig(), 1, draw)
    assert draw[1]["goal_count"] == 1.0
    assert draw[1]["assist_count"] == 0.0  # the scorer can never assist their own goal
    assert draw[2]["assist_count"] == 1.0
    assert draw[2]["goal_count"] == 0.0


# ---------------------------------------------------------------------------
# Section 8 — per-draw component variances.
# ---------------------------------------------------------------------------


def test_component_variance_uses_per_draw_totals():
    accumulators = {(1, 1): mc._empty_accumulator()}
    goal_points = RULES.goal_points_for("MID")
    # Draw 1: two goals.  Draw 2: two goals.  Per-draw total is a constant, so
    # the variance must be zero; per-goal squaring would wrongly give 2*gp^2.
    for _ in range(2):
        draw: dict[int, dict[str, float]] = {}
        bucket = mc._draw_bucket(draw, 1)
        bucket["goal"] = 2 * goal_points
        bucket["goal_count"] = 2.0
        bucket["goal_flag"] = 1.0
        mc._commit_draw(draw, accumulators, 1)
    sums = accumulators[(1, 1)]["sum"]
    assert sums["goal"] == 4 * goal_points
    assert sums["goal_sq"] == 2 * (2 * goal_points) ** 2
    variance = sums["goal_sq"] / 2 - (sums["goal"] / 2) ** 2
    assert variance == pytest.approx(0.0)


def test_multi_assist_component_variance():
    accumulators = {(1, 1): mc._empty_accumulator()}
    points = RULES.assist_points
    draw: dict[int, dict[str, float]] = {}
    bucket = mc._draw_bucket(draw, 1)
    bucket["assist"] = 3 * points
    bucket["assist_count"] = 3.0
    bucket["assist_flag"] = 1.0
    mc._commit_draw(draw, accumulators, 1)
    sums = accumulators[(1, 1)]["sum"]
    assert sums["assist_sq"] == (3 * points) ** 2


# ---------------------------------------------------------------------------
# Sections 9-10 — DefCon / save minute mixture.
# ---------------------------------------------------------------------------


def test_defcon_and_save_minute_mixture_match_manual_integration():
    payload = {
        "expected_minutes": 50.0,
        "minute_state_distribution": [
            {"state": "0", "probability": 0.5, "conditional_mean_minutes": 0.0},
            {"state": "1-59", "probability": 0.2, "conditional_mean_minutes": 30.0},
            {"state": "60-79", "probability": 0.2, "conditional_mean_minutes": 70.0},
            {"state": "80-90", "probability": 0.1, "conditional_mean_minutes": 85.0},
        ],
    }
    threshold = RULES.defcon_threshold_for("MID")
    manual = (
        0.2 * xpts.poisson_tail_probability(12.0 * 30.0 / 90.0, threshold)
        + 0.2 * xpts.poisson_tail_probability(12.0 * 70.0 / 90.0, threshold)
        + 0.1 * xpts.poisson_tail_probability(12.0 * 85.0 / 90.0, threshold)
    )
    value, hit, used = xpts.defcon_xpts_with_mixture("MID", 12.0, payload, RULES)
    assert used is True
    assert value == pytest.approx(RULES.defcon_points * manual)
    # Poisson threshold-hit probability is NOT globally convex, so the mixture
    # value need not sit on one side of the expected-minutes value.
    save_value, save_used = xpts.save_xpts_with_mixture(4.0, 1.0, payload, RULES)
    manual_save = (
        0.2 * xpts.expected_floor_poisson_ratio(4.0 * 30.0 / 90.0, RULES.saves_per_point)
        + 0.2 * xpts.expected_floor_poisson_ratio(4.0 * 70.0 / 90.0, RULES.saves_per_point)
        + 0.1 * xpts.expected_floor_poisson_ratio(4.0 * 85.0 / 90.0, RULES.saves_per_point)
    )
    assert save_used is True
    assert save_value == pytest.approx(manual_save)


def test_legacy_minutes_falls_back_when_no_mixture():
    legacy = {"expected_minutes": 50.0}
    _, _, used = xpts.defcon_xpts_with_mixture("MID", 12.0, legacy, RULES)
    assert used is False
    _, save_used = xpts.save_xpts_with_mixture(4.0, 1.0, legacy, RULES)
    assert save_used is False


# ---------------------------------------------------------------------------
# Section 12 — every-row probability gate.
# ---------------------------------------------------------------------------


def test_probability_gate_checks_each_row_independently():
    valid = _fake_summary(pid=1)
    invalid = _fake_summary(pid=2, p_goal=1.5)
    assert mc.probability_range_violations([valid]) == []
    first = mc.probability_range_violations([invalid, valid])
    assert len(first) == 1 and "player 2" in first[0]
    second = mc.probability_range_violations([valid, invalid])
    assert len(second) == 1 and "player 2" in second[0]
    # A later valid row must not mask an earlier invalid one.
    assert any("p_goal=1.5" in reason for reason in first)


# ---------------------------------------------------------------------------
# Sections 13-14 — repaired calibration evaluator.
# ---------------------------------------------------------------------------


def test_canonical_forecast_never_defaults_to_zero():
    assert calibration.canonical_forecast({}, "p_score_le_2") is None
    assert calibration.canonical_forecast({"p_score_le_2": 0.3}, "p_score_le_2") == 0.3
    # Legacy aliases are readable ONLY through the explicit adapter.
    assert calibration.canonical_forecast({"p_blank": 0.4}, "p_score_le_2") is None
    assert calibration.canonical_forecast({"p_blank": 0.4}, "p_score_le_2", legacy_adapter=True) == 0.4


def test_synthetic_final_event_calibration_end_to_end(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    try:
        _world(conn)
        with conn:
            repo.upsert_fixtures(conn, [FixtureRecord(
                id=100, event=4, team_h=1, team_a=2, team_h_score=1, team_a_score=0,
                started=1, finished=1, finished_provisional=0, minutes=90,
            )])
            repo.upsert_player_gameweeks(conn, [PlayerGameweekRecord(
                player_id=10, event=4, fixture_id=100, opponent_team=2, was_home=1,
                minutes=90, starts=1, total_points=14, goals_scored=1, assists=0,
                clean_sheets=1, goals_conceded=0, saves=0, bonus=0, yellow_cards=0,
                defensive_contribution=12, source="element_summary",
            )])
            conn.execute("UPDATE events SET finished=1, data_checked=1 WHERE id=4")
        run_id = analytics.create_projection_run(
            conn, model_family=mc.MONTE_CARLO_MODEL_FAMILY,
            model_version=mc.MONTE_CARLO_MODEL_VERSION, planning_event=4,
            planning_context_hash="hash", data_cutoff="2026-09-10T12:00:00Z",
            scouting_cutoff=None, official_run_ids={}, deadline_status="PRE_DEADLINE",
        )
        payload = {
            "mean_core": 13.5, "mean_total_proxy": 15.0,
            "q10": 2.0, "q25": 6.0, "q50": 10.0, "q75": 14.0, "q90": 18.0,
            "p_score_le_2": 0.1, "p_score_5_plus": 0.7,
            "p_score_10_plus": 0.4, "p_score_15_plus": 0.1,
            "p_goal": 0.3, "p_assist": 0.1, "p_clean_sheet_eligible": 0.4, "p_defcon_hit": 0.3,
        }
        analytics.freeze_monte_carlo_distribution(
            conn, run_id, player_id=10, fixture_id=100, event=4, team_id=1, opponent_id=2,
            position="DEF", xpts_run_id=1, minutes_run_id=1, team_run_id=1, rate_run_id=1,
            payload=payload, model_version=mc.MONTE_CARLO_MODEL_VERSION,
        )
        analytics.finish_projection_run(conn, run_id, "complete")

        metrics = calibration.evaluate_monte_carlo_run(conn, run_id, 4)
        assert metrics["actual_modelled_core_sample_count"] == 1.0
        assert metrics["no_total_proxy_coverage"] == 1.0
        assert "mc_core_mean_bias" in metrics
        assert "core_coverage_80" in metrics
        assert "brier_defcon" in metrics
        assert "coverage_50" not in metrics and "coverage_80" not in metrics
        # ACTUAL_MODELLED_CORE for this row: 2 (appearance) + 6 (goal) + 4 (CS) + 2 (DefCon) = 14.
        assert metrics["mc_core_mean_bias"] == pytest.approx(13.5 - 14.0)
    finally:
        conn.close()


def test_module_has_no_named_player_hardcoding():
    import inspect
    source = inspect.getsource(mc)
    assert "Salah" not in source and "Haaland" not in source
    # Calibration targets come from payload shares, never a stored player table.
    assert "defcon_actions_per90" in source


# ---------------------------------------------------------------------------
# Section 15 — full-band substitution timing.
# ---------------------------------------------------------------------------


def test_substitution_timing_covers_full_band():
    config = joint_minutes.JointMinutesConfig()
    rng = joint_minutes.random.Random(4242)
    times = [joint_minutes._event(0, 1, 15.0, rng, config, [1.0], [(75.0, 89.0)])["time"] for _ in range(4000)]
    assert min(times) <= 76.0 and max(times) >= 88.0
    # The old midpoint-with-jitter model could never reach the band edges.
    assert any(time <= 76 for time in times) and any(time >= 88 for time in times)


# ---------------------------------------------------------------------------
# Section 16 — CRN category substreams.
# ---------------------------------------------------------------------------


def _crn_fixture():
    fixture = _finish(_fixture(cameo=0.2))
    return fixture


def test_crn_invariance_to_irrelevant_container_row():
    fixture_a = _crn_fixture()
    fixture_b = _crn_fixture()
    import copy

    fixture_b = copy.deepcopy(fixture_b)
    # Add a zero-probability container player to side 1 only.
    side = fixture_b["sides"][0]
    side["players"].append({
        "player_id": 999999,
        "position": "MID",
        "minutes": {"p_start": 0.0, "p_cameo": 0.0, "p_available": 0.0,
                    "p_60_given_start": 0.0, "p_80_given_start": 0.0,
                    "expected_minutes_if_start": 0.0, "expected_minutes_if_cameo": 0.0},
        "payload": {"adjusted_expected_xg": 0.0, "expected_xa": 0.0,
                    "fixture_xg_per90": 0.0, "fixture_xa_per90": 0.0, "yellow_per90": 0.0,
                    "defcon_actions_per90": 0.0,
                    "save_model": {"saves_per90_posterior": 0.0, "pressure_multiplier": 1.0}},
    })
    config = mc.MonteCarloConfig(simulations=400, seed=2026, calibration_states=2000)
    result_a = mc.simulate({100: fixture_a}, config, RULES)
    result_b = mc.simulate({100: fixture_b}, config, RULES)
    map_a = {(s["player_id"], s["fixture_id"]): s for s in result_a["summaries"]}
    map_b = {(s["player_id"], s["fixture_id"]): s for s in result_b["summaries"]}
    shared = [key for key in map_a if key in map_b]
    assert shared
    for key in shared:
        assert map_a[key]["mean_minutes"] == pytest.approx(map_b[key]["mean_minutes"], abs=1e-9)
        assert map_a[key]["mean_reconciliation_error"]["goal"] == pytest.approx(
            map_b[key]["mean_reconciliation_error"]["goal"], abs=1e-9
        )
        assert map_a[key]["mc_components"]["assist"] == pytest.approx(
            map_b[key]["mc_components"]["assist"], abs=1e-9
        )
    # The opponent side is untouched by the other side's roster change.
    opponent_key = (1001, 100)
    assert map_a[opponent_key]["mean_reconciliation_error"]["goal"] == pytest.approx(
        map_b[opponent_key]["mean_reconciliation_error"]["goal"], abs=1e-9
    )


def test_price_and_ownership_do_not_affect_simulation():
    import copy

    fixture_base = _crn_fixture()
    fixture_priced = copy.deepcopy(fixture_base)
    for side in fixture_priced["sides"]:
        for player in side["players"]:
            player["payload"]["now_cost"] = 999
            player["payload"]["selected"] = 12345678
            player["payload"]["ownership"] = 99.9
    config = mc.MonteCarloConfig(simulations=300, seed=5, calibration_states=2000)
    base = mc.simulate({100: fixture_base}, config, RULES)
    priced = mc.simulate({100: fixture_priced}, config, RULES)
    for left, right in zip(base["summaries"], priced["summaries"]):
        assert left["player_id"] == right["player_id"]
        assert left["mean_minutes"] == pytest.approx(right["mean_minutes"])
        assert left["mean_core"] == pytest.approx(right["mean_core"])
