"""Zero-variance standard-error repair — count components get a provable variance floor.

The GW7 certification refusal was caused by ONE row: a defender with a material analytic
goal expectation whose 2000-draw sample happened to contain zero goals, so the observed
sample variance was exactly 0 and the estimator reported an infinite standardised error.

`_classify_standardised` now accepts an ANALYTIC variance floor, used only when the
observed standard deviation is exactly zero, and only for components where the production
generator PROVABLY has Var >= mean.  These tests pin both sides:

  * CASE A — a legitimate low-rate zero-count must PASS;
  * CASE B — a genuinely dead instrument must still FAIL.

Every row here is built through the REAL accumulator (`_empty_accumulator` /
`_commit_draw`) and the REAL `_summarise` / `readiness_summary` path, so the tests drive
the producer, not a stand-in for it.
"""

from __future__ import annotations

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


def _drive(*, player_id: int, position: str, worlds: int, appearances: int,
           goal_counts, analytic: dict[str, float]) -> tuple[dict, dict]:
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
    # Called WITHOUT the new `rules` argument on purpose: the signature stays compatible
    # with the predecessor revision, so a failure here is a BEHAVIOURAL difference and
    # never a TypeError from a changed signature.
    summaries = mc._summarise(accumulators, {key: analytic}, config, player_meta, None)
    result = {"summaries": summaries, "team_minutes": []}
    return summaries[0], mc.readiness_summary({}, result, config)


def _matched_analytic(appearances: int, worlds: int, goal_points: float) -> dict[str, float]:
    """Analytic reference matching the simulated appearance so ONLY goal can mismatch."""

    appearance_mean = TOSIN_APPEARANCE_POINTS * appearances / worlds
    return {"appearance": appearance_mean, "core": appearance_mean, "goal": goal_points}


# ---------------------------------------------------------------------------
# PREDECESSOR KILL — the real GW7 shape
# ---------------------------------------------------------------------------


def test_low_rate_zero_count_is_not_an_infinite_standardised_error():
    """PREDECESSOR KILL: on b5b23bb this row yields inf and a hard ZERO_VARIANCE_MISMATCH.

    Shape taken from the real GW7 refusal: a DEF on the pitch in 924 of 2000 worlds,
    scoring in none, with a material analytic goal expectation.  Under a correct
    low-rate generator that is an ordinary Poisson tail (expected 4.38 goals per 2000
    draws, so P(zero) ~ 1.25%), not a broken instrument.
    """

    summary, gate = _drive(
        player_id=147, position="DEF", worlds=TOSIN_WORLDS, appearances=TOSIN_APPEARANCES,
        goal_counts=[0] * TOSIN_WORLDS,
        analytic=_matched_analytic(TOSIN_APPEARANCES, TOSIN_WORLDS, TOSIN_ANALYTIC_GOAL_POINTS),
    )

    # The zero variance and the material error are both still real.
    assert summary["mc_component_std"]["goal"] == 0.0
    assert summary["mean_reconciliation_error"]["goal"] == pytest.approx(-TOSIN_ANALYTIC_GOAL_POINTS)

    # But the standardised error is now finite, computed from the analytic variance floor.
    z = summary["standardised_error"]["goal"]
    assert math.isfinite(z), f"expected a finite standardised error, got {z!r}"
    expected_floor_se = math.sqrt(6 * TOSIN_ANALYTIC_GOAL_POINTS / TOSIN_WORLDS)
    assert z == pytest.approx(-TOSIN_ANALYTIC_GOAL_POINTS / expected_floor_se, rel=1e-12)
    assert abs(z) < mc.MonteCarloConfig().individual_max_z

    assert "goal" not in summary["zero_variance_mismatch"]
    assert not any("ZERO_VARIANCE_MISMATCH" in reason for reason in gate["fail_reasons"])
    assert gate["status"] != "FAIL"


# ---------------------------------------------------------------------------
# TWO-SIDED SAFETY — a dead instrument must still fail
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("position", ["FWD", "MID", "DEF", "GKP"])
def test_a_grossly_dead_goal_instrument_keeps_the_historical_classification(position):
    """CASE B at a large expectation: the original ``inf`` rule still applies, unchanged.

    With E in the hundreds or thousands, observing zero is impossible under the model, so
    the plausibility test refuses the row and the standardised error stays ``inf`` — the
    exact historical outcome, with the same `ZERO_VARIANCE_MISMATCH` reason.
    """

    analytic_goal = 2.0
    summary, gate = _drive(
        player_id=1, position=position, worlds=TOSIN_WORLDS, appearances=TOSIN_WORLDS,
        goal_counts=[0] * TOSIN_WORLDS,
        analytic=_matched_analytic(TOSIN_WORLDS, TOSIN_WORLDS, analytic_goal),
    )

    assert summary["mc_component_std"]["goal"] == 0.0
    assert math.isinf(summary["standardised_error"]["goal"])
    assert "goal" in summary["zero_variance_mismatch"]

    assert gate["status"] == "FAIL"
    assert any("ZERO_VARIANCE_MISMATCH" in reason for reason in gate["fail_reasons"])
    assert gate["individual_material_failures"], "a dead instrument must be a material failure"


# ---------------------------------------------------------------------------
# P1-01 — THE MID-RATE BAND.  The variance floor alone cannot catch these, because
# for a zero-event sample it saturates at |z| = sqrt(analytic_points * sims / weight),
# which only crosses individual_max_z (8) above ~0.19 points (DEF).  A dead generator
# expecting 0.10 points therefore produced z ~ 5.8 and PASSED.  The plausibility test
# closes that band: observing zero when 33.3 goals were expected has probability
# 3.3e-15 and must be refused.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("position,analytic_goal", [
    ("DEF", 0.10),   # the exact P1-01 counterexample
    ("GKP", 0.10),   # its 10-point-weight analogue: even more permissive before
    ("DEF", 0.05),   # between the two regimes
    ("MID", 0.09),
    ("FWD", 0.07),
])
def test_a_mid_rate_dead_goal_instrument_is_refused(position, analytic_goal):
    """P1-01 KILL: RED on 33fab651, which returned a finite z with status ``ok`` here.

    These rows sit ABOVE the 0.01 zero-variance materiality but BELOW the analytic
    expectation at which the variance-floor standardised error reaches 8, so the floor
    alone could not fail them and the historical 0.01 guard was effectively relaxed.
    """

    weight = RULES.goal_points_for(position)
    expected_goals = TOSIN_WORLDS * analytic_goal / weight
    # Read the boundary tolerantly: this test must be RUNNABLE against the reviewed
    # predecessor 33fab651 so that it fails there BEHAVIOURALLY (a finite z that should
    # have been inf), never with an AttributeError on a constant that did not exist yet.
    floor = getattr(mc, "ZERO_EVENT_PLAUSIBILITY_FLOOR", 1e-3)
    assert expected_goals > math.log(1.0 / floor), "in the refused regime"

    summary, gate = _drive(
        player_id=2, position=position, worlds=TOSIN_WORLDS, appearances=TOSIN_WORLDS,
        goal_counts=[0] * TOSIN_WORLDS,
        analytic=_matched_analytic(TOSIN_WORLDS, TOSIN_WORLDS, analytic_goal),
    )

    assert summary["mc_component_std"]["goal"] == 0.0
    assert abs(summary["mean_reconciliation_error"]["goal"]) > 0.01
    assert math.isinf(summary["standardised_error"]["goal"])
    assert "goal" in summary["zero_variance_mismatch"]

    assert gate["status"] == "FAIL"
    assert any("ZERO_VARIANCE_MISMATCH" in reason for reason in gate["fail_reasons"])


def test_the_tolerance_limit_is_the_expected_event_count_not_the_point_value():
    """The band is set by E (expected goals over the sample), so the point weight matters.

    Pinning the boundary keeps the rule from silently drifting: a row is judged only
    while its expected count stays under ``ln(1 / floor)``.
    """

    limit = math.log(1.0 / mc.ZERO_EVENT_PLAUSIBILITY_FLOOR)
    for position, allowed_points in (("DEF", 6 * limit / TOSIN_WORLDS),
                                     ("GKP", 10 * limit / TOSIN_WORLDS),
                                     ("MID", 5 * limit / TOSIN_WORLDS),
                                     ("FWD", 4 * limit / TOSIN_WORLDS)):
        weight = RULES.goal_points_for(position)
        just_inside = mc._analytic_variance_floor_per_world("goal", allowed_points * 0.99, position, RULES, TOSIN_WORLDS)
        just_outside = mc._analytic_variance_floor_per_world("goal", allowed_points * 1.01, position, RULES, TOSIN_WORLDS)
        assert just_inside.expected_count_in_sample < limit
        assert just_outside.expected_count_in_sample > limit
        assert mc._classify_standardised(-allowed_points * 0.99, 0.0, TOSIN_WORLDS, 0.01, just_inside)[1] == "ok"
        assert math.isinf(mc._classify_standardised(-allowed_points * 1.01, 0.0, TOSIN_WORLDS, 0.01, just_outside)[0])
        assert weight > 0


# ---------------------------------------------------------------------------
# BELOW-MATERIALITY BEHAVIOUR IS UNCHANGED
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("analytic_goal", [0.002, 0.0099])
def test_microscopic_zero_variance_rows_stay_below_materiality(analytic_goal):
    """~245 low-rate keeper rows sit here; they must not be routed to a failure path.

    An analytic expectation of exactly 0.0 is a different case: the reconciliation is
    then exactly zero and classifies as ``ok`` (unchanged by this repair).
    """

    summary, gate = _drive(
        player_id=250, position="GKP", worlds=TOSIN_WORLDS, appearances=TOSIN_WORLDS,
        goal_counts=[0] * TOSIN_WORLDS,
        analytic=_matched_analytic(TOSIN_WORLDS, TOSIN_WORLDS, analytic_goal),
    )
    assert "goal" in summary["zero_variance_below_materiality"]
    assert "goal" not in summary["zero_variance_mismatch"]
    assert summary["standardised_error"]["goal"] == 0.0
    assert gate["status"] != "FAIL"


# ---------------------------------------------------------------------------
# NON-ZERO-VARIANCE INVARIANCE — the ordinary path cannot be perturbed
# ---------------------------------------------------------------------------


def test_the_floor_cannot_change_a_nonzero_variance_outcome():
    """Structural pin: with std > 0 the allowance is never consulted."""

    huge = mc._ZeroVarianceAllowance(variance_per_world=1e9, expected_count_in_sample=0.0)
    plausible = mc._ZeroVarianceAllowance(variance_per_world=1e9, expected_count_in_sample=1.0)
    for value, std, sims in ((0.5, 0.2, 2000), (-2.0, 0.9, 2000), (12.5, 0.05, 2000),
                             (0.013, 0.19, 2000), (1.0, 1e-12, 500)):
        without = mc._classify_standardised(value, std, sims, 0.01)
        assert without == mc._classify_standardised(value, std, sims, 0.01, huge)
        assert without == mc._classify_standardised(value, std, sims, 0.01, plausible)


def test_zero_variance_with_zero_error_is_still_allowed_with_a_floor():
    assert mc._classify_standardised(0.0, 0.0, 2000, 0.01, None) == (0.0, "ok")
    huge = mc._ZeroVarianceAllowance(variance_per_world=5.0, expected_count_in_sample=1.0)
    assert mc._classify_standardised(0.0, 0.0, 2000, 0.01, huge) == (0.0, "ok")


def test_a_goal_row_that_does_sample_goals_keeps_its_exact_standardisation():
    """A real sampled goal must standardise off the OBSERVED variance, not the floor."""

    worlds, goals = 2000, [1 if i in (7, 401) else 0 for i in range(2000)]
    summary, _gate = _drive(
        player_id=147, position="DEF", worlds=worlds, appearances=TOSIN_APPEARANCES,
        goal_counts=goals,
        analytic=_matched_analytic(TOSIN_APPEARANCES, worlds, TOSIN_ANALYTIC_GOAL_POINTS),
    )
    observed_std = summary["mc_component_std"]["goal"]
    assert observed_std > 0.0
    error = summary["mean_reconciliation_error"]["goal"]
    se = observed_std / math.sqrt(worlds)
    assert summary["standardised_error"]["goal"] == pytest.approx(error / se, rel=1e-12)
    assert summary["zero_variance_mismatch"] == []


# ---------------------------------------------------------------------------
# FLOOR SCOPE — only components with a provable bound
# ---------------------------------------------------------------------------


def test_the_floor_is_defined_only_where_it_is_provable():
    """goal only.  assist/save/defcon/yellow keep the historical behaviour."""

    tosin = mc._analytic_variance_floor_per_world("goal", 0.013145, "DEF", RULES, TOSIN_WORLDS)
    assert tosin.variance_per_world == pytest.approx(6 * 0.013145)
    assert tosin.expected_count_in_sample == pytest.approx(TOSIN_WORLDS * 0.013145 / 6)

    big = mc._analytic_variance_floor_per_world("goal", 2.0, "FWD", RULES, TOSIN_WORLDS)
    assert big.variance_per_world == pytest.approx(8.0)
    assert big.expected_count_in_sample == pytest.approx(TOSIN_WORLDS * 2.0 / 4)

    assert mc._analytic_variance_floor_per_world("goal", 0.0, "DEF", RULES, TOSIN_WORLDS) is None
    assert mc._analytic_variance_floor_per_world("goal", 0.013, None, RULES, TOSIN_WORLDS) is None

    for component in ("assist", "save", "defcon", "yellow", "clean_sheet",
                      "goals_conceded", "appearance", "core", "core_linear"):
        assert mc._analytic_variance_floor_per_world(component, 2.0, "DEF", RULES, TOSIN_WORLDS) is None, component


def test_a_component_without_a_provable_bound_keeps_the_infinite_classification():
    """Defence in depth: the historical behaviour is intact where no bound exists."""

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
# END TO END — the real simulator still certifies
# ---------------------------------------------------------------------------


def test_a_real_simulated_fixture_reconciles_with_no_zero_variance_mismatch():
    """The producer changes must not disturb a normal fixture's reconciliation.

    NOTE: the shared unit fixture already fails its gate on unrelated aggregate
    allowances at this size — identical fail reasons on the UNMODIFIED parent SHA — so
    this test pins the properties the repair can affect, not the fixture's global verdict.
    """

    from test_monte_carlo import _fixture, _finish

    fixture = _finish(_fixture())
    config = mc.MonteCarloConfig(simulations=3000, seed=12345)
    result = mc.simulate({fixture["fixture_id"]: fixture}, config, RULES)
    gate = mc.readiness_summary({fixture["fixture_id"]: fixture}, result, config)

    # The repair must not manufacture any zero-variance mismatch.
    assert not any("ZERO_VARIANCE_MISMATCH" in reason for reason in gate["fail_reasons"])
    assert sum(1 for s in result["summaries"] if s["zero_variance_mismatch"]) == 0

    # every non-zero-variance goal row must standardise off its observed variance
    for summary in result["summaries"]:
        if summary["mc_component_std"]["goal"] > 0.0:
            error = summary["mean_reconciliation_error"]["goal"]
            se = summary["mc_component_std"]["goal"] / math.sqrt(summary["simulations"])
            assert summary["standardised_error"]["goal"] == pytest.approx(error / se, rel=1e-12)
