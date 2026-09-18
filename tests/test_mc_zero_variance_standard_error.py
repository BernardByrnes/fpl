"""Zero-variance standard-error repair — the allowance is PRODUCTION-derived.

The GW7 certification refusal was caused by ONE row: a defender with a material analytic
goal expectation whose 2000-draw sample happened to contain zero goals, so the observed
sample variance was exactly 0 and the estimator reported an infinite standardised error.

The zero-variance branch now consults an allowance built from the PRODUCTION generator —
the per-world scoring intensity induced by the sampled play, the team's Poisson lambda
and the calibrated scorer weights — and NOT from the analytic comparator.  The analytic
target is only the quantity being tested, so it can never establish that the mechanism is
live.  These tests pin all three regimes:

  * LIVE LOW-RATE  — production exposes an opportunity; an unlucky zero sample may pass;
  * DEAD LOW-RATE  — production exposes NO opportunity; the row HARD FAILS even though
                     the analytic target is small and would have looked plausible;
  * DEAD MID/HIGH  — production expectation material to enormous; the row HARD FAILS.

Every row is built through the REAL accumulator (`_empty_accumulator` / `_commit_draw`)
and the REAL `_summarise` / `readiness_summary` path, so the tests drive the producer.
"""

from __future__ import annotations

import inspect
import math

import pytest

from fpl_brain import monte_carlo as mc

RULES = mc.DEFAULT_SCORING_RULES
FIXTURE_ID = 1000

#: The exact fresh GW7 values that tripped the gate (player 147, Tosin, DEF).
TOSIN_ANALYTIC_GOAL_POINTS = 0.013145
TOSIN_APPEARANCES = 924
TOSIN_WORLDS = 2000
TOSIN_APPEARANCE_POINTS = 2.0  # 1 for appearing + 1 for 60+

#: The production expectation the real GW7 freeze induces for that player.  It is very
#: close to the analytic-implied count because the scorer calibration is fitted to be
#: expectation-preserving -- which is exactly why the predecessor's analytic-derived
#: allowance looked correct on the happy path.  The repair does not ASSUME that identity;
#: the kill test below shows the two diverge decisively when the mechanism is dead.
TOSIN_PRODUCTION_EXPECTED_GOALS = TOSIN_WORLDS * TOSIN_ANALYTIC_GOAL_POINTS / 6.0

#: The predecessor revision accepted no production evidence at all.  Detecting that makes
#: a failure here BEHAVIOURAL (the row is blessed by the analytic target) instead of a
#: TypeError from a changed signature.
_SUMMARISE_ACCEPTS_PRODUCTION = (
    "production_goal_expectation" in inspect.signature(mc._summarise).parameters
)


def _drive(*, player_id: int, position: str, worlds: int, appearances: int,
           goal_counts, analytic: dict[str, float],
           production_expected_goals: float | None) -> tuple[dict, dict]:
    """Build one player-fixture row through the real accumulator and gate path."""

    key = (int(player_id), FIXTURE_ID)
    accumulators = {key: mc._empty_accumulator()}
    for index in range(worlds):
        draw_components: dict[int, dict[str, float]] = {}
        bucket = mc._draw_bucket(draw_components, int(player_id))
        if index < appearances:
            bucket["appearance"] = TOSIN_APPEARANCE_POINTS
            bucket["minutes"] = 90.0
        goals = goal_counts[index]
        if goals:
            bucket["goal"] = float(goals * RULES.goal_points_for(position))
            bucket["goal_count"] = float(goals)
            bucket["goal_flag"] = 1.0
        mc._commit_draw(draw_components, accumulators, FIXTURE_ID)

    config = mc.MonteCarloConfig(simulations=worlds)
    player_meta = {int(player_id): {"position": position, "team_id": 1}}
    production = {key: float(production_expected_goals)} if production_expected_goals is not None else {}
    if _SUMMARISE_ACCEPTS_PRODUCTION:
        summaries = mc._summarise(accumulators, {key: analytic}, config, player_meta, None,
                                  RULES, production)
    else:  # predecessor revision: no production evidence is accepted, which IS the defect
        summaries = mc._summarise(accumulators, {key: analytic}, config, player_meta, None)
    result = {"summaries": summaries, "team_minutes": []}
    return summaries[0], mc.readiness_summary({}, result, config)


def _matched_analytic(appearances: int, worlds: int, goal_points: float) -> dict[str, float]:
    """Analytic reference matching the simulated appearance so ONLY goal can mismatch."""

    appearance_mean = TOSIN_APPEARANCE_POINTS * appearances / worlds
    return {"appearance": appearance_mean, "core": appearance_mean, "goal": goal_points}


# ---------------------------------------------------------------------------
# THE P1-01 CLOSURE — production opportunity, not the analytic target, decides
# ---------------------------------------------------------------------------


def test_zero_production_opportunity_with_a_low_analytic_target_is_refused():
    """P1-01 KILL: RED on 161148c, which derived the allowance from the analytic target.

    The counterexample exactly: the comparator expects 0.013145 points, the sample caught
    zero goals, and the PRODUCTION mechanism exposed NO scoring opportunity in any world.
    That is a dead player-specific generator, and a small comparator expectation must not
    be able to bless it.  On 161148c this produced E = 4.3817 from the analytic target and
    passed; here it must HARD FAIL.
    """

    summary, gate = _drive(
        player_id=147, position="DEF", worlds=TOSIN_WORLDS, appearances=TOSIN_APPEARANCES,
        goal_counts=[0] * TOSIN_WORLDS,
        analytic=_matched_analytic(TOSIN_APPEARANCES, TOSIN_WORLDS, TOSIN_ANALYTIC_GOAL_POINTS),
        production_expected_goals=0.0,
    )

    assert summary["mc_component_std"]["goal"] == 0.0
    assert abs(summary["mean_reconciliation_error"]["goal"]) > 0.01
    assert math.isinf(summary["standardised_error"]["goal"])
    assert "goal" in summary["zero_variance_mismatch"]

    assert gate["status"] == "FAIL"
    assert any("ZERO_VARIANCE_MISMATCH" in reason for reason in gate["fail_reasons"])


def test_live_low_rate_zero_count_is_tolerated():
    """The SAME analytic target and the same zero sample, but production IS live.

    This is the distinction the repair has to preserve: an unlucky zero from a working
    low-rate mechanism may pass, a zero from a dead one may not.
    """

    summary, gate = _drive(
        player_id=147, position="DEF", worlds=TOSIN_WORLDS, appearances=TOSIN_APPEARANCES,
        goal_counts=[0] * TOSIN_WORLDS,
        analytic=_matched_analytic(TOSIN_APPEARANCES, TOSIN_WORLDS, TOSIN_ANALYTIC_GOAL_POINTS),
        production_expected_goals=TOSIN_PRODUCTION_EXPECTED_GOALS,
    )

    expected = TOSIN_PRODUCTION_EXPECTED_GOALS
    assert math.exp(-expected) >= mc.ZERO_EVENT_PLAUSIBILITY_FLOOR

    weight = RULES.goal_points_for("DEF")
    expected_se = weight * math.sqrt(expected) / TOSIN_WORLDS
    z = summary["standardised_error"]["goal"]
    assert math.isfinite(z)
    assert z == pytest.approx(-TOSIN_ANALYTIC_GOAL_POINTS / expected_se, rel=1e-12)

    assert "goal" not in summary["zero_variance_mismatch"]
    assert gate["status"] != "FAIL"


def test_the_analytic_target_cannot_tolerate_a_zero_production_row():
    """Pins that the comparator plays no part in the liveness decision.

    Analytic 2.0 points (a huge comparator expectation) with ZERO production opportunity
    is refused for the production reason, and analytic 0.013145 with the SAME zero
    opportunity is refused identically -- the target's magnitude changes nothing.
    """

    for analytic_points in (0.013145, 0.10, 2.0):
        summary, gate = _drive(
            player_id=9, position="DEF", worlds=TOSIN_WORLDS, appearances=TOSIN_WORLDS,
            goal_counts=[0] * TOSIN_WORLDS,
            analytic=_matched_analytic(TOSIN_WORLDS, TOSIN_WORLDS, analytic_points),
            production_expected_goals=0.0,
        )
        assert math.isinf(summary["standardised_error"]["goal"]), analytic_points
        assert gate["status"] == "FAIL", analytic_points


# ---------------------------------------------------------------------------
# MID-RATE DEAD CASES — production expectation is material
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("position,analytic_goal,production_goals", [
    ("DEF", 0.10, 33.333),   # the original P1-01 counterexample
    ("GKP", 0.10, 20.000),   # its 10-point-weight analogue
    ("DEF", 0.05, 16.667),
    ("MID", 0.09, 36.000),
    ("FWD", 0.07, 35.000),
])
def test_a_mid_rate_dead_goal_instrument_is_refused(position, analytic_goal, production_goals):
    """The production mechanism expected this many goals over the sample and delivered none."""

    # Read the boundary tolerantly so this remains runnable (and BEHAVIOURAL) on the
    # predecessors that predate the constant.
    floor = getattr(mc, "ZERO_EVENT_PLAUSIBILITY_FLOOR", 1e-3)
    assert math.exp(-production_goals) < floor
    summary, gate = _drive(
        player_id=2, position=position, worlds=TOSIN_WORLDS, appearances=TOSIN_WORLDS,
        goal_counts=[0] * TOSIN_WORLDS,
        analytic=_matched_analytic(TOSIN_WORLDS, TOSIN_WORLDS, analytic_goal),
        production_expected_goals=production_goals,
    )

    assert summary["mc_component_std"]["goal"] == 0.0
    assert abs(summary["mean_reconciliation_error"]["goal"]) > 0.01
    assert math.isinf(summary["standardised_error"]["goal"])
    assert "goal" in summary["zero_variance_mismatch"]
    assert gate["status"] == "FAIL"
    assert any("ZERO_VARIANCE_MISMATCH" in reason for reason in gate["fail_reasons"])


@pytest.mark.parametrize("position", ["FWD", "MID", "DEF", "GKP"])
def test_a_grossly_dead_goal_instrument_keeps_the_historical_classification(position):
    """At an enormous production expectation the historical ``inf`` rule still applies."""

    summary, gate = _drive(
        player_id=1, position=position, worlds=TOSIN_WORLDS, appearances=TOSIN_WORLDS,
        goal_counts=[0] * TOSIN_WORLDS,
        analytic=_matched_analytic(TOSIN_WORLDS, TOSIN_WORLDS, 2.0),
        production_expected_goals=700.0,
    )

    assert summary["mc_component_std"]["goal"] == 0.0
    assert math.isinf(summary["standardised_error"]["goal"])
    assert "goal" in summary["zero_variance_mismatch"]
    assert gate["status"] == "FAIL"
    assert any("ZERO_VARIANCE_MISMATCH" in reason for reason in gate["fail_reasons"])
    assert gate["individual_material_failures"]


# ---------------------------------------------------------------------------
# THE BOUNDARY — set by the PRODUCTION expected count
# ---------------------------------------------------------------------------


def test_the_tolerance_limit_is_the_production_expected_count():
    """A row is judged only while production expects fewer than ln(1/floor) goals."""

    limit = math.log(1.0 / mc.ZERO_EVENT_PLAUSIBILITY_FLOOR)
    for position in ("DEF", "GKP", "MID", "FWD"):
        weight = RULES.goal_points_for(position)
        inside = mc._production_goal_zero_variance_allowance("goal", position, RULES, TOSIN_WORLDS, limit * 0.99)
        outside = mc._production_goal_zero_variance_allowance("goal", position, RULES, TOSIN_WORLDS, limit * 1.01)
        assert inside.expected_count_in_sample == pytest.approx(limit * 0.99)
        assert outside.expected_count_in_sample == pytest.approx(limit * 1.01)
        assert inside.variance_per_world == pytest.approx(weight * weight * limit * 0.99 / TOSIN_WORLDS)
        assert mc._classify_standardised(-0.02, 0.0, TOSIN_WORLDS, 0.01, inside)[1] == "ok"
        assert math.isinf(mc._classify_standardised(-0.02, 0.0, TOSIN_WORLDS, 0.01, outside)[0])


def test_no_production_opportunity_offers_no_allowance():
    assert mc._production_goal_zero_variance_allowance("goal", "DEF", RULES, 2000, 0.0) is None
    assert mc._production_goal_zero_variance_allowance("goal", "DEF", RULES, 2000, -1.0) is None
    assert mc._production_goal_zero_variance_allowance("goal", None, RULES, 2000, 4.0) is None


def test_the_allowance_is_defined_only_for_goal():
    """assist/save/defcon/yellow keep the historical behaviour."""

    for component in ("assist", "save", "defcon", "yellow", "clean_sheet",
                      "goals_conceded", "appearance", "core", "core_linear"):
        assert mc._production_goal_zero_variance_allowance(
            component, "DEF", RULES, TOSIN_WORLDS, 10.0) is None, component


# ---------------------------------------------------------------------------
# UNCHANGED PATHS
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("analytic_goal", [0.002, 0.0099])
def test_microscopic_zero_variance_rows_stay_below_materiality(analytic_goal):
    """~245 low-rate keeper rows sit here; they must not be routed to a failure path."""

    summary, gate = _drive(
        player_id=250, position="GKP", worlds=TOSIN_WORLDS, appearances=TOSIN_WORLDS,
        goal_counts=[0] * TOSIN_WORLDS,
        analytic=_matched_analytic(TOSIN_WORLDS, TOSIN_WORLDS, analytic_goal),
        production_expected_goals=0.0,
    )
    assert "goal" in summary["zero_variance_below_materiality"]
    assert "goal" not in summary["zero_variance_mismatch"]
    assert summary["standardised_error"]["goal"] == 0.0
    assert gate["status"] != "FAIL"


def test_the_allowance_cannot_change_a_nonzero_variance_outcome():
    """Structural pin: with std > 0 the allowance is never consulted."""

    live = mc._ZeroVarianceAllowance(variance_per_world=1e9, expected_count_in_sample=1.0)
    dead = mc._ZeroVarianceAllowance(variance_per_world=1e9, expected_count_in_sample=0.0)
    for value, std, sims in ((0.5, 0.2, 2000), (-2.0, 0.9, 2000), (12.5, 0.05, 2000),
                             (0.013, 0.19, 2000), (1.0, 1e-12, 500)):
        without = mc._classify_standardised(value, std, sims, 0.01)
        assert without == mc._classify_standardised(value, std, sims, 0.01, live)
        assert without == mc._classify_standardised(value, std, sims, 0.01, dead)


def test_zero_variance_with_zero_error_is_still_allowed_with_an_allowance():
    assert mc._classify_standardised(0.0, 0.0, 2000, 0.01, None) == (0.0, "ok")
    live = mc._ZeroVarianceAllowance(variance_per_world=5.0, expected_count_in_sample=1.0)
    assert mc._classify_standardised(0.0, 0.0, 2000, 0.01, live) == (0.0, "ok")


def test_a_goal_row_that_does_sample_goals_keeps_its_exact_standardisation():
    """A real sampled goal standardises off the OBSERVED variance, not the allowance."""

    worlds, goals = 2000, [1 if i in (7, 401) else 0 for i in range(2000)]
    summary, _gate = _drive(
        player_id=147, position="DEF", worlds=worlds, appearances=TOSIN_APPEARANCES,
        goal_counts=goals,
        analytic=_matched_analytic(TOSIN_APPEARANCES, worlds, TOSIN_ANALYTIC_GOAL_POINTS),
        production_expected_goals=TOSIN_PRODUCTION_EXPECTED_GOALS,
    )
    observed_std = summary["mc_component_std"]["goal"]
    assert observed_std > 0.0
    error = summary["mean_reconciliation_error"]["goal"]
    assert summary["standardised_error"]["goal"] == pytest.approx(
        error / (observed_std / math.sqrt(worlds)), rel=1e-12)
    assert summary["zero_variance_mismatch"] == []


def test_a_component_without_a_provable_bound_keeps_the_infinite_classification():
    """The historical behaviour is intact where no production bound exists."""

    key = (7, FIXTURE_ID)
    accumulators = {key: mc._empty_accumulator()}
    for _ in range(2000):
        draw_components: dict[int, dict[str, float]] = {}
        bucket = mc._draw_bucket(draw_components, 7)
        bucket["appearance"] = 2.0
        mc._commit_draw(draw_components, accumulators, FIXTURE_ID)
    config = mc.MonteCarloConfig(simulations=2000)
    summaries = mc._summarise(accumulators, {key: {"appearance": 0.9, "core": 0.9, "assist": 0.5}},
                              config, {7: {"position": "MID", "team_id": 1}}, None)
    summary = summaries[0]
    assert summary["mc_component_std"]["assist"] == 0.0
    assert math.isinf(summary["standardised_error"]["assist"])
    assert "assist" in summary["zero_variance_mismatch"]


# ---------------------------------------------------------------------------
# THE DERIVATION ITSELF — conservation and liveness, through the real simulator
# ---------------------------------------------------------------------------


def test_the_production_expectation_conserves_the_team_goal_lambda(monkeypatch):
    """The per-player intensities must sum to exactly worlds x lambda_for per side.

    This is the structural proof that the integral over the piecewise-constant on-pitch
    sets is exact and that the scorer weights (with the residual bucket) form a proper
    categorical allocation.  It is a property of the DERIVATION, independent of any
    realised goal.
    """

    from test_monte_carlo import _fixture, _finish

    captured: dict = {}
    original = mc._summarise

    def spy(accumulators, analytic, config, player_meta, calibration=None, rules=None,
            production_goal_expectation=None):
        captured["production"] = dict(production_goal_expectation or {})
        return original(accumulators, analytic, config, player_meta, calibration, rules,
                        production_goal_expectation)

    monkeypatch.setattr(mc, "_summarise", spy)

    fixture = _finish(_fixture())
    worlds = 200
    config = mc.MonteCarloConfig(simulations=worlds, seed=12345)
    mc.simulate({fixture["fixture_id"]: fixture}, config, RULES)

    production = captured["production"]
    assert production, "the production expectation must be accumulated"
    assert all(value >= 0.0 for value in production.values())

    expected_total = 2 * worlds * float(fixture["sides"][0]["lambda_for"])
    assert sum(production.values()) == pytest.approx(expected_total, rel=1e-9)


def test_a_real_simulated_fixture_reconciles_with_no_zero_variance_mismatch():
    """The producer changes must not disturb a normal fixture's reconciliation.

    NOTE: the shared unit fixture already fails its gate on unrelated aggregate
    allowances at this size — identical fail reasons on the unmodified parent SHA — so
    this pins the properties the repair can affect, not the fixture's global verdict.
    """

    from test_monte_carlo import _fixture, _finish

    fixture = _finish(_fixture())
    config = mc.MonteCarloConfig(simulations=3000, seed=12345)
    result = mc.simulate({fixture["fixture_id"]: fixture}, config, RULES)
    gate = mc.readiness_summary({fixture["fixture_id"]: fixture}, result, config)

    assert not any("ZERO_VARIANCE_MISMATCH" in reason for reason in gate["fail_reasons"])
    assert sum(1 for s in result["summaries"] if s["zero_variance_mismatch"]) == 0

    for summary in result["summaries"]:
        if summary["mc_component_std"]["goal"] > 0.0:
            error = summary["mean_reconciliation_error"]["goal"]
            se = summary["mc_component_std"]["goal"] / math.sqrt(summary["simulations"])
            assert summary["standardised_error"]["goal"] == pytest.approx(error / se, rel=1e-12)
