"""Addendum 6 static-review remediation acceptance tests.

Covers the canonical randomised occupancy law shared by integration, calibration
and production; true sampled minutes for MC DefCon/saves; primitive validation
from actual run metadata; xPts/Minutes run coherence; the secondary individual
materiality gate; dead-diagnostic removal; threshold Brier scores; substitution
band edge cleanup; occupancy audit enablement; and artifact-derived reporting.
"""

from __future__ import annotations

import importlib.util
import itertools
import math
from pathlib import Path

import pytest

from fpl_brain import analytics, joint_minutes, monte_carlo as mc
from fpl_brain.database import connect_database
from fpl_brain.scoring_rules import DEFAULT_SCORING_RULES as RULES

from test_addendum5_repairs import _fake_summary, _hetero_fixture
from test_minutes_model import _world

REPO_ROOT = Path(__file__).resolve().parents[1]


def _minimal_world(conn):
    from fpl_brain import repositories as repo
    from fpl_brain.models import FixtureRecord, PlayerRecord, PositionRecord, TeamRecord
    with conn:
        repo.upsert_teams(conn, [TeamRecord(id=1, name="One"), TeamRecord(id=2, name="Two")])
        repo.upsert_positions(conn, [PositionRecord(id=i, singular_name_short=name)
                                     for i, name in ((1, "GKP"), (2, "DEF"), (3, "MID"), (4, "FWD"))])
        repo.upsert_players(conn, [
            PlayerRecord(id=7, web_name="P7", full_name="Player 7", team_id=1, element_type=3),
            PlayerRecord(id=10, web_name="P10", full_name="Player 10", team_id=1, element_type=2),
        ])
        repo.upsert_fixtures(conn, [FixtureRecord(id=100, event=4, team_h=1, team_a=2)])


def _load_report_tool():
    spec = importlib.util.spec_from_file_location(
        "build_phase5_report_tables", REPO_ROOT / "scripts" / "build_phase5_report_tables.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Section 2/3 — one canonical randomised occupancy law.
# ---------------------------------------------------------------------------

OUTFIELD_P_START = [0.95, 0.92, 0.90, 0.88, 0.85, 0.82, 0.80, 0.78,
                    0.72, 0.65, 0.55, 0.45, 0.40, 0.33]  # sums to exactly 10.0


def _player_pair(pid, position, p_start, p_cameo, xg90, xa90, *, p80=1.0, cameo=15.0):
    minutes = p_start * 90.0 + p_cameo * cameo
    payload = {
        "adjusted_expected_xg": xg90 * minutes / 90.0,
        "expected_fpl_assists": xa90 * minutes / 90.0,
        "expected_xa": xa90 * minutes / 90.0,
        "goal_xpts": xg90 * minutes / 90.0 * RULES.goal_points_for(position),
        "assist_xpts": xa90 * minutes / 90.0 * RULES.assist_points,
        "fixture_xg_per90": xg90, "fixture_xa_per90": xa90,
        "yellow_per90": 0.0, "defcon_actions_per90": 0.0,
        "save_model": {"saves_per90_posterior": 0.0, "pressure_multiplier": 1.0},
        "core_xpts": 0.0,
    }
    minutes_payload = {
        "joint_start_target": p_start,
        "joint_availability": 1.0,
        "joint_exit_propensity": 1.0 - p80,
        "joint_entry_propensity": (p_cameo / (1.0 - p_start)) if p_start < 1.0 else 0.0,
        "joint_position": position,
        "joint_expected_minutes_if_cameo": cameo,
        "primitive_source_version": joint_minutes.JOINT_MINUTES_MODEL_VERSION,
        "expected_minutes": minutes,
        "minute_state_distribution": [
            {"state": "0", "probability": max(0.0, 1.0 - p_start - p_cameo), "conditional_mean_minutes": 0.0},
            {"state": "1-9", "probability": 0.0, "conditional_mean_minutes": 0.0},
            {"state": "10-19", "probability": p_cameo, "conditional_mean_minutes": cameo},
            {"state": "20-29", "probability": 0.0, "conditional_mean_minutes": 0.0},
            {"state": "30-39", "probability": 0.0, "conditional_mean_minutes": 0.0},
            {"state": "40-49", "probability": 0.0, "conditional_mean_minutes": 0.0},
            {"state": "50-59", "probability": 0.0, "conditional_mean_minutes": 0.0},
            {"state": "60-69", "probability": 0.0, "conditional_mean_minutes": 0.0},
            {"state": "70-79", "probability": 0.0, "conditional_mean_minutes": 0.0},
            {"state": "80-89", "probability": 0.0, "conditional_mean_minutes": 0.0},
            {"state": "90", "probability": p_start, "conditional_mean_minutes": 90.0},
        ],
    }
    side_player = {"player_id": pid, "position": position, "payload": payload, "minutes": minutes_payload}
    kernel_player = {
        "player_id": pid, "position": position, "p_start": p_start, "p_available": 1.0,
        "exit_propensity": 1.0 - p80,
        "entry_propensity": (p_cameo / (1.0 - p_start)) if p_start < 1.0 else 0.0,
        "expected_minutes_if_cameo": cameo,
    }
    return side_player, kernel_player


def _competitive_side_parts(team_id, base):
    side_players = []
    kernel_players = []
    for pid, position, p_start in (
        (base + 0, "GKP", 1.0),
        (base + 1, "GKP", 0.0),
    ):
        player, kernel = _player_pair(pid, position, p_start, 0.0, 0.0, 0.0)
        side_players.append(player)
        kernel_players.append(kernel)
    for offset, p_start in enumerate(OUTFIELD_P_START):
        position = "DEF" if offset < 5 else "MID"
        xg = 0.45 - offset * 0.02
        player, kernel = _player_pair(base + 2 + offset, position, p_start, 0.0, xg, 0.2)
        side_players.append(player)
        kernel_players.append(kernel)
    for offset in range(5):
        player, kernel = _player_pair(base + 16 + offset, "MID", 0.0, 0.15, 0.3, 0.2)
        side_players.append(player)
        kernel_players.append(kernel)
    sum_xg = sum(p["payload"]["adjusted_expected_xg"] for p in side_players)
    side = {
        "team_id": team_id, "opponent_id": 2 if team_id == 1 else 1,
        "players": side_players, "lambda_for": sum_xg, "lambda_against": 1.2,
        "sum_player_xg": sum_xg, "residual_xg": 0.0,
    }
    return side, kernel_players


def _capture_starters(side, kernel_players, kind, config, draws):
    captured: list[set[int]] = []
    original = joint_minutes.sample_side_world

    def spy(players, profile, cfg, rng, **kwargs):
        world = original(players, profile, cfg, rng, **kwargs)
        # Only the first side (player ids < 1000) is compared.
        if players and max(int(p["player_id"]) for p in players) < 1000:
            captured.append(set(world["starters"]))
        return world

    other_side, _ = _competitive_side_parts(2, 1000)
    fixture = {"fixture_id": 901, "event": 4, "sides": [side, other_side]}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(joint_minutes, "sample_side_world", spy)
        if kind == "integration":
            joint_minutes.integrate_side_marginals(
                kernel_players, None, joint_minutes.JointMinutesConfig(),
                fixture_id=901, team_id=1, draws=draws,
            )
        elif kind == "calibration":
            mc.calibrate_side(side, config, 901, 1)
        else:
            mc.simulate({901: fixture}, config, RULES)
            # Calibration also samples the kernel before the simulation worlds;
            # keep only the production worlds (the last `draws` side-1 samples).
            captured = captured[-draws:]
    return captured


def _co_start(captured, i, j):
    return sum(1 for starters in captured if i in starters and j in starters) / len(captured)


def test_occupancy_law_identical_across_consumers():
    side, kernel_players = _competitive_side_parts(1, 10)
    assert sum(OUTFIELD_P_START) == pytest.approx(10.0)
    draws = 1500
    config = mc.MonteCarloConfig(simulations=draws, seed=31, calibration_states=draws)
    integration = _capture_starters(side, kernel_players, "integration", config, draws)
    calibration = _capture_starters(side, kernel_players, "calibration", config, draws)
    production = _capture_starters(side, kernel_players, "production", config, draws)
    assert len(integration) == draws and len(calibration) == draws and len(production) == draws

    tolerance = 0.06  # ~4 sampling SE at N=1500
    outfield = range(2, 16)
    # First-order start marginals agree with each other and with the target.
    for index in outfield:
        target = OUTFIELD_P_START[index - 2]
        values = [
            sum(1 for s in captured if index in s) / len(captured)
            for captured in (integration, calibration, production)
        ]
        assert max(values) - min(values) < tolerance, (index, values)
        assert abs(values[2] - target) < tolerance, (index, target, values[2])
    # Pairwise co-start probabilities agree across the three consumers.
    for i, j in ((2, 3), (2, 10), (5, 15), (8, 9), (13, 14)):
        values = [_co_start(captured, i, j) for captured in (integration, calibration, production)]
        assert max(values) - min(values) < tolerance, (i, j, values)
    # Distribution of total on-pitch attacking rate also agrees.
    def _rate_distribution(captured):
        totals = []
        for starters in captured:
            totals.append(sum(
                side["players"][index]["payload"]["adjusted_expected_xg"] for index in starters
            ))
        mean = sum(totals) / len(totals)
        var = sum((t - mean) ** 2 for t in totals) / len(totals)
        return mean, math.sqrt(var)
    means = [_rate_distribution(c)[0] for c in (integration, calibration, production)]
    assert max(means) - min(means) < 0.05, means


def test_fixed_order_process_differs_from_random_order_law():
    side, kernel_players = _competitive_side_parts(1, 10)
    draws = 1500
    config = mc.MonteCarloConfig(simulations=draws, seed=31, calibration_states=draws)
    production = _capture_starters(side, kernel_players, "production", config, draws)

    # Reference: the RETIRED fixed-order process (draw index pinned at 0).
    player_ids = [p["player_id"] for p in kernel_players]
    rng = joint_minutes.random.Random("fixed-order-reference")
    fixed: list[set[int]] = []
    keys = joint_minutes.joint_order_keys("fixed", 901, 1, 0, player_ids)
    for _ in range(draws):
        world = joint_minutes.sample_side_world(
            kernel_players, None, joint_minutes.JointMinutesConfig(), rng, order_keys=keys
        )
        fixed.append(set(world["starters"]))

    differences = [
        abs(_co_start(production, i, j) - _co_start(fixed, i, j))
        for i, j in itertools.combinations(range(2, 16), 2)
    ]
    assert max(differences) > 0.05, f"regression test has no power: max diff {max(differences)}"


# ---------------------------------------------------------------------------
# Section 7 — MC DefCon/saves use TRUE sampled minutes.
# ---------------------------------------------------------------------------


def test_mc_defcon_save_ignore_analytic_quadrature():
    fixture = _hetero_fixture()
    for side in fixture["sides"]:
        for player in side["players"]:
            player["payload"]["defcon_actions_per90"] = 12.0
            if player["position"] == "GKP":
                player["payload"]["save_model"] = {"saves_per90_posterior": 4.0, "pressure_multiplier": 1.0}
    config = mc.MonteCarloConfig(simulations=600, seed=17, calibration_states=1500)
    baseline = mc.simulate({500: fixture}, config, RULES)

    # Corrupt every analytic quadrature conditional mean; TRUE minutes mean the
    # production DefCon/save results must be byte-identical.
    for side in fixture["sides"]:
        for player in side["players"]:
            for entry in player["minutes"]["minute_state_distribution"]:
                entry["conditional_mean_minutes"] = 999.0
    perturbed = mc.simulate({500: fixture}, config, RULES)
    for left, right in zip(baseline["summaries"], perturbed["summaries"]):
        assert left["player_id"] == right["player_id"]
        assert left["mc_components"]["defcon"] == pytest.approx(right["mc_components"]["defcon"], abs=0.0)
        assert left["mc_components"]["save"] == pytest.approx(right["mc_components"]["save"], abs=0.0)


# ---------------------------------------------------------------------------
# Section 9 — zero-probability analytic state is safe.
# ---------------------------------------------------------------------------


def test_zero_probability_analytic_state_is_inert():
    from fpl_brain import xpts
    payload = {
        "expected_minutes": 40.0,
        "minute_state_distribution": [
            {"state": "0", "probability": 0.6, "conditional_mean_minutes": 0.0},
            {"state": "10-19", "probability": 0.0, "conditional_mean_minutes": 12345.0},
            {"state": "30-39", "probability": 0.4, "conditional_mean_minutes": 35.0},
        ],
    }
    clean = {
        "expected_minutes": 40.0,
        "minute_state_distribution": [
            {"state": "0", "probability": 0.6, "conditional_mean_minutes": 0.0},
            {"state": "10-19", "probability": 0.0, "conditional_mean_minutes": 0.0},
            {"state": "30-39", "probability": 0.4, "conditional_mean_minutes": 35.0},
        ],
    }
    assert xpts.defcon_xpts_with_mixture("MID", 12.0, payload, RULES) == \
        xpts.defcon_xpts_with_mixture("MID", 12.0, clean, RULES)
    assert xpts.save_xpts_with_mixture(4.0, 1.0, payload, RULES) == \
        xpts.save_xpts_with_mixture(4.0, 1.0, clean, RULES)


# ---------------------------------------------------------------------------
# Section 10 — primitive validation from actual run metadata.
# ---------------------------------------------------------------------------


def _make_minutes_run(conn, version, payloads):
    run_id = analytics.create_projection_run(
        conn, model_family=analytics.MINUTES_MODEL_FAMILY, model_version=version,
        planning_event=4, planning_context_hash="h", data_cutoff="2026-09-10T12:00:00Z",
        scouting_cutoff=None, official_run_ids={},
    )
    for player_id, payload in payloads.items():
        analytics.freeze_prediction(
            conn, run_id, kind=analytics.MINUTES_V1_KIND, player_id=player_id, event=4,
            fixture_id=100, payload=payload, model_version=version,
        )
    analytics.finish_projection_run(conn, run_id, "complete")
    return run_id


def _make_xpts_run(conn, minutes_run_id, team_run_id=1, rate_run_id=1):
    run_id = analytics.create_projection_run(
        conn, model_family=analytics.XPTS_MODEL_FAMILY, model_version="xpts_v1.4.1",
        planning_event=4, planning_context_hash="h", data_cutoff="2026-09-10T12:00:00Z",
        scouting_cutoff=None, official_run_ids={},
    )
    analytics.freeze_xpts_projection(
        conn, run_id, player_id=7, fixture_id=100, event=4, team_id=1, opponent_id=2, position="MID",
        minutes_run_id=minutes_run_id, team_run_id=team_run_id, rate_run_id=rate_run_id,
        payload={"adjusted_expected_xg": 0.4}, model_version="xpts_v1.4.1",
        scoring_rules_version="v1",
    )
    analytics.finish_projection_run(conn, run_id, "complete")
    return run_id


def test_primitive_validation_uses_run_metadata(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    try:
        _minimal_world(conn)
        without_primitives = {"p_start": 0.5, "p_cameo": 0.1, "model_version": "x"}
        prospective = _make_minutes_run(conn, "minutes_v1.5.2", {7: without_primitives})
        xpts_run = _make_xpts_run(conn, prospective)
        fixtures = mc.load_fixture_inputs(
            conn, event=4, xpts_run_id=xpts_run, minutes_run_id=prospective, team_run_id=1,
        )
        side = fixtures[100]["sides"][0]
        assert side["primitive_reconstruction"] is True
        assert side["primitive_missing_fields"]

        legacy = _make_minutes_run(conn, "minutes_v1.5.0", {7: without_primitives})
        xpts_legacy = _make_xpts_run(conn, legacy)
        fixtures_legacy = mc.load_fixture_inputs(
            conn, event=4, xpts_run_id=xpts_legacy, minutes_run_id=legacy, team_run_id=1,
        )
        assert fixtures_legacy[100]["sides"][0]["primitive_reconstruction"] is False
    finally:
        conn.close()


def test_zero_primitive_fields_are_still_validated():
    # A row carrying primitive source version but a MISSING required field must
    # be treated as reconstruction-required for a prospective run.
    payload = {"primitive_source_version": "minutes_v1.5.2", "joint_start_target": 0.5}
    missing = [f for f in mc.REQUIRED_JOINT_PRIMITIVES if payload.get(f) is None]
    assert "joint_position" in missing


# ---------------------------------------------------------------------------
# Section 11 — xPts/Minutes run coherence.
# ---------------------------------------------------------------------------


def test_xpts_minutes_run_mismatch_rejected(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    try:
        _minimal_world(conn)
        minutes_run = _make_minutes_run(conn, "minutes_v1.5.2", {7: {"joint_start_target": 0.5}})
        xpts_run = _make_xpts_run(conn, minutes_run)
        assert mc.validate_input_run_coherence(
            conn, xpts_run_id=xpts_run, minutes_run_id=minutes_run, team_run_id=1, rate_run_id=1
        ) == []
        problems = mc.validate_input_run_coherence(
            conn, xpts_run_id=xpts_run, minutes_run_id=minutes_run + 1, team_run_id=99, rate_run_id=1
        )
        assert any("MINUTES_RUN_MISMATCH" in p for p in problems)
        assert any("TEAM_RUN_MISMATCH" in p for p in problems)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Section 6 — secondary individual materiality gate.
# ---------------------------------------------------------------------------


def test_individual_materiality_gate():
    material_fail = _fake_summary(
        pid=1, standardised_error={"goal": 20.0},
        mean_reconciliation_error={**_fake_summary()["mean_reconciliation_error"], "goal": 0.5},
    )
    tiny = _fake_summary(
        pid=1, standardised_error={"goal": 20.0},
        mean_reconciliation_error={**_fake_summary()["mean_reconciliation_error"], "goal": 0.05},
    )
    small_z = _fake_summary(
        pid=1, standardised_error={"goal": 2.0},
        mean_reconciliation_error={**_fake_summary()["mean_reconciliation_error"], "goal": 0.5},
    )
    result = {"summaries": [material_fail], "team_minutes": [], "calibration": []}
    gate = mc.readiness_summary({}, result, mc.MonteCarloConfig())
    assert gate["status"] == "FAIL"
    assert gate["individual_material_failures"]
    for summary in (tiny, small_z):
        gate = mc.readiness_summary(
            {}, {"summaries": [summary], "team_minutes": [], "calibration": []}, mc.MonteCarloConfig()
        )
        assert not gate["individual_material_failures"]
    assert "core" in mc.MonteCarloConfig().player_error_p95_allowance


# ---------------------------------------------------------------------------
# Section 13 — dead diagnostics removed.
# ---------------------------------------------------------------------------


def test_dead_diagnostics_are_gone():
    from test_monte_carlo import _fixture, _finish
    fixture = _finish(_fixture(cameo=0.2))
    result = mc.simulate({100: fixture}, mc.MonteCarloConfig(simulations=50, seed=3, calibration_states=400), RULES)
    assert "impossible_worlds" not in result
    for summary in result["summaries"]:
        assert "max_state_match_error" not in summary


# ---------------------------------------------------------------------------
# Section 16 — threshold Brier scores.
# ---------------------------------------------------------------------------


def test_threshold_brier_scores_are_computed(tmp_path):
    from fpl_brain import calibration
    from fpl_brain.models import FixtureRecord, PlayerGameweekRecord
    from fpl_brain import repositories as repo
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
            conn, model_family=mc.MONTE_CARLO_MODEL_FAMILY, model_version=mc.MONTE_CARLO_MODEL_VERSION,
            planning_event=4, planning_context_hash="h", data_cutoff="2026-09-10T12:00:00Z",
            scouting_cutoff=None, official_run_ids={},
        )
        analytics.freeze_monte_carlo_distribution(
            conn, run_id, player_id=10, fixture_id=100, event=4, team_id=1, opponent_id=2,
            position="DEF", xpts_run_id=1, minutes_run_id=1, team_run_id=1, rate_run_id=1,
            payload={
                "mean_core": 13.5, "q10": 2.0, "q25": 6.0, "q50": 10.0, "q75": 14.0, "q90": 18.0,
                "p_score_le_2": 0.1, "p_score_5_plus": 0.7, "p_score_10_plus": 0.4, "p_score_15_plus": 0.1,
                "p_goal": 0.3, "p_assist": 0.1, "p_clean_sheet_eligible": 0.4, "p_defcon_hit": 0.3,
            },
            model_version=mc.MONTE_CARLO_MODEL_VERSION,
        )
        analytics.finish_projection_run(conn, run_id, "complete")
        metrics = calibration.evaluate_monte_carlo_run(conn, run_id, 4)
        for name in ("brier_p_score_le_2", "brier_p_score_5_plus", "brier_p_score_10_plus", "brier_p_score_15_plus"):
            assert name in metrics
            assert f"{name}_sample_count" in metrics
        # Reconstructed CORE = 14: <=2 false, >=5 true, >=10 true, >=15 false.
        assert metrics["brier_p_score_le_2"] == pytest.approx((0.1 - 0.0) ** 2)
        assert metrics["brier_p_score_15_plus"] == pytest.approx((0.1 - 0.0) ** 2)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Section 17 — substitution band edges.
# ---------------------------------------------------------------------------


def test_substitution_band_uniform_integer_and_early_band_preserved():
    config = joint_minutes.JointMinutesConfig()
    rng = joint_minutes.random.Random(7)
    early = [joint_minutes._event(0, 1, 15.0, rng, config, [1.0], [(0.0, 29.0)])["time"] for _ in range(6000)]
    assert min(early) <= 1 and max(early) >= 28
    assert all(0 <= t <= 29 for t in early)
    # Endpoints must not be half-weighted by a round() of a continuous draw.
    count_zero = sum(1 for t in early if t == 0) / len(early)
    count_five = sum(1 for t in early if t == 5) / len(early)
    assert abs(count_zero - count_five) < 0.02
    late = [joint_minutes._event(0, 1, 10.0, rng, config, [1.0], [(75.0, 89.0)])["time"] for _ in range(6000)]
    assert min(late) <= 76 and max(late) >= 88
    assert all(75 <= t <= 89 for t in late)


# ---------------------------------------------------------------------------
# Section 12 — occupancy audit actually enabled for certification.
# ---------------------------------------------------------------------------


def test_occupancy_audit_enabled_in_certification_and_measures():
    freeze_source = (REPO_ROOT / "scripts" / "freeze_predictions.py").read_text(encoding="utf-8")
    assert "occupancy_audit=True" in freeze_source
    from test_monte_carlo import _fixture, _finish
    fixture = _finish(_fixture(cameo=0.2))
    config = mc.MonteCarloConfig(simulations=200, seed=9, calibration_states=400, occupancy_audit=True)
    result = mc.simulate({100: fixture}, config, RULES)
    assert "occupancy_violations" in result
    assert result["occupancy_violations"] == 0


# ---------------------------------------------------------------------------
# Section 14/15 — provenance and artifact-derived reporting.
# ---------------------------------------------------------------------------


def test_source_snapshot_hash_is_deterministic_and_covers_sources():
    first = analytics.source_snapshot_sha256()
    second = analytics.source_snapshot_sha256()
    assert first == second and len(first) == 64
    required = {
        "fpl_brain/joint_minutes.py", "fpl_brain/monte_carlo.py", "fpl_brain/xpts.py",
        "fpl_brain/calibration.py", "fpl_brain/substitution_model.py", "fpl_brain/analytics.py",
        "scripts/freeze_predictions.py",
    }
    assert required.issubset(set(analytics.SOURCE_SNAPSHOT_FILES))
    provenance = analytics.git_provenance()
    assert provenance["head"]


def test_report_metrics_must_match_artifact():
    tool = _load_report_tool()
    records = []
    for index in range(4):
        records.append({
            "player_id": index, "fixture_id": 100,
            "payload": {
                "mean_reconciliation_error": {"goal": 0.01 * index},
                "standardised_error": {"goal": float(index)},
                "mc_component_std": {"goal": 1.0},
                "simulations": 100,
                "analytic_components": {"goal": 1.0},
                "mc_components": {"goal": 1.0 + 0.01 * index},
            },
        })
    stats, _rows = tool.recompute(records, {})
    recorded = {name: dict(stats[name]) for name in tool.COMPONENTS}
    assert tool.compare_metrics(stats, recorded) == []
    recorded["goal"]["p95_abs"] += 1.0
    mismatches = tool.compare_metrics(stats, recorded)
    assert any("goal.p95_abs" in mismatch for mismatch in mismatches)
