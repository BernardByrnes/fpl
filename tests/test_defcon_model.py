"""DEFCON model V1 — thresholds, shrinkage equivalence, tail behaviour, outputs.

The baseline is not restated: ``defcon_rate_posterior`` is pinned by test against
``xpts._shrink_rate`` (the function production actually calls), and the Poisson
tail is required to BE ``xpts.poisson_tail_probability``.  If either drifts, the
evaluation's "BASELINE" would stop being the Brain's model, which is the one
thing that would silently invalidate the whole comparison.
"""

from __future__ import annotations

import math
import random

import pytest

from fpl_brain import defcon_model as dm
from fpl_brain import scoring_rules as sr
from fpl_brain import xpts

RULES = sr.DEFAULT_SCORING_RULES


def test_canonical_thresholds_come_from_the_scoring_rules():
    assert RULES.defcon_threshold_for("DEF") == 10
    assert RULES.defcon_threshold_for("MID") == 12
    assert RULES.defcon_threshold_for("FWD") == 12
    for position in ("DEF", "MID", "FWD"):
        projection = dm.project_defcon(position=position, expected_minutes=90.0, rate_posterior=8.0)
        assert projection.threshold == RULES.defcon_threshold_for(position)
        assert projection.expected_points == pytest.approx(RULES.defcon_points * projection.p_threshold)


def test_goalkeepers_earn_no_defcon():
    projection = dm.project_defcon(position="GKP", expected_minutes=90.0, rate_posterior=20.0)
    assert projection.threshold is None
    assert projection.p_threshold == 0.0
    assert projection.expected_points == 0.0
    assert projection.variance == 0.0


def test_shrinkage_is_provably_the_engines_own_arithmetic():
    """The pinned equivalence — the whole comparison rests on this."""

    rng = random.Random(7)
    for _ in range(4000):
        minutes = rng.uniform(0.0, 2500.0)
        per90 = rng.uniform(0.0, 25.0)
        prior = rng.uniform(0.0, 20.0)
        ess = rng.choice((600.0, 300.0, 1200.0))
        mine = dm.defcon_rate_posterior(
            observed_actions=per90 * minutes / 90.0,
            observed_minutes=minutes,
            prior_rate_per90=prior,
            prior_ess_minutes=ess,
        )
        oracle = xpts._shrink_rate(minutes, per90, prior, ess)
        assert mine == pytest.approx(oracle, rel=1e-12, abs=1e-12)


def test_zero_minutes_returns_the_prior_unchanged():
    assert dm.defcon_rate_posterior(
        observed_actions=0.0, observed_minutes=0.0, prior_rate_per90=7.5, prior_ess_minutes=600.0
    ) == 7.5


def test_poisson_baseline_is_the_engine_function():
    assert dm.xpts.poisson_tail_probability is xpts.poisson_tail_probability
    projection = dm.project_defcon(
        position="MID", expected_minutes=90.0, rate_posterior=9.5, variant=dm.VARIANT_BASELINE_POISSON
    )
    assert projection.p_threshold == pytest.approx(xpts.poisson_tail_probability(9.5, 12), rel=1e-12)


def test_negative_binomial_degenerates_to_poisson_and_widens_the_tail():
    assert dm.negative_binomial_tail(9.5, dm.NB_POISSON_LIMIT_R, 12) == pytest.approx(
        xpts.poisson_tail_probability(9.5, 12), rel=1e-9
    )
    assert dm.negative_binomial_tail(9.5, 6.0, 12) > xpts.poisson_tail_probability(9.5, 12)


def test_count_variance_follows_the_model_family():
    assert dm.count_variance(dm.VARIANT_BASELINE_POISSON, 9.5, 6.0) == pytest.approx(9.5)
    assert dm.count_variance(dm.VARIANT_NEGATIVE_BINOMIAL, 9.5, 6.0) == pytest.approx(9.5 + 9.5**2 / 6.0)
    # a non-positive dispersion cannot produce a negative variance
    assert dm.count_variance(dm.VARIANT_NEGATIVE_BINOMIAL, 9.5, 0.0) == pytest.approx(9.5)


def test_level_scale_moves_the_probability_monotonically_and_only_when_named():
    scaled = dm.project_defcon(
        position="MID", expected_minutes=90.0, rate_posterior=8.0,
        variant=dm.VARIANT_LEVEL_CALIBRATED_POISSON, level_scale=0.6,
    )
    unscaled = dm.project_defcon(
        position="MID", expected_minutes=90.0, rate_posterior=8.0,
        variant=dm.VARIANT_LEVEL_CALIBRATED_POISSON, level_scale=1.0,
    )
    assert scaled.p_threshold < unscaled.p_threshold
    # a variant that does not carry the level scale must ignore it entirely
    ignored = dm.project_defcon(
        position="MID", expected_minutes=90.0, rate_posterior=8.0,
        variant=dm.VARIANT_BASELINE_POISSON, level_scale=0.1,
    )
    assert ignored.p_threshold == pytest.approx(unscaled.p_threshold, rel=1e-12)


def test_opponent_multiplier_only_applies_to_the_opponent_variants():
    plain = dm.project_defcon(position="MID", expected_minutes=90.0, rate_posterior=8.0,
                              variant=dm.VARIANT_BASELINE_POISSON, opponent_multiplier=1.5)
    assert plain.rate_posterior == pytest.approx(8.0)
    adjusted = dm.project_defcon(position="MID", expected_minutes=90.0, rate_posterior=8.0,
                                 variant=dm.VARIANT_OPPONENT_POISSON, opponent_multiplier=1.5)
    assert adjusted.rate_posterior == pytest.approx(12.0)
    assert adjusted.p_threshold > plain.p_threshold


def test_every_required_output_is_present_and_finite():
    projection = dm.project_defcon(position="DEF", expected_minutes=78.0, rate_posterior=7.25)
    payload = projection.as_dict()
    for key in dm.DEFCON_REQUIRED_OUTPUTS:
        assert key in payload, key
        assert math.isfinite(float(payload[key])), key
    assert 0.0 <= payload["P_DEFCON_THRESHOLD"] <= 1.0
    assert payload["DEFCON_VARIANCE"] >= 0.0


def test_projection_is_deterministic():
    args = dict(position="FWD", expected_minutes=64.0, rate_posterior=6.75, variant=dm.VARIANT_LEVEL_CALIBRATED_NEGATIVE_BINOMIAL,
                dispersion_r=5.0, opponent_multiplier=1.1, level_scale=0.95)
    first = dm.project_defcon(**args).as_dict()
    second = dm.project_defcon(**args).as_dict()
    assert first == second


def test_zero_and_low_minutes_collapse_the_probability():
    none = dm.project_defcon(position="MID", expected_minutes=0.0, rate_posterior=9.0)
    assert none.p_threshold == 0.0
    assert none.projected_defcon == 0.0
    low = dm.project_defcon(position="MID", expected_minutes=8.0, rate_posterior=9.0)
    full = dm.project_defcon(position="MID", expected_minutes=90.0, rate_posterior=9.0)
    assert low.p_threshold < full.p_threshold


def test_fit_count_statistics_recovers_overdispersion_and_handles_underdispersion():
    rng = random.Random(11)
    over = [{"position": "MID", "actions": float(max(0, int(round(rng.gauss(8.0, 6.0))))) } for _ in range(600)]
    stats = dm.fit_count_statistics(over)
    assert "MID" in stats
    assert stats["MID"].dispersion_r < dm.NB_POISSON_LIMIT_R  # genuine overdispersion

    tight = [{"position": "DEF", "actions": 6.0 + (i % 2)} for i in range(400)]
    tight_stats = dm.fit_count_statistics(tight)
    # variance below the mean cannot yield a negative r: it collapses to Poisson
    assert tight_stats["DEF"].dispersion_r == dm.NB_POISSON_LIMIT_R


def test_brier_log_loss_and_calibration_are_well_formed():
    predictions = [0.0, 0.25, 0.5, 0.75, 1.0]
    outcomes = [0, 0, 1, 1, 1]
    assert dm.brier_score(predictions, outcomes) == pytest.approx(
        sum((p - y) ** 2 for p, y in zip(predictions, outcomes)) / 5
    )
    assert dm.log_loss(predictions, outcomes) > 0.0
    # a perfectly calibrated constant predictor scores the entropy
    assert dm.log_loss([0.5] * 4, [0, 1, 0, 1]) == pytest.approx(math.log(2.0))
    table = dm.calibration_table(predictions, outcomes, bins=4)
    assert sum(row["n"] for row in table) == 5
    assert dm.brier_score([], []) != dm.brier_score([], [])  # NaN, not a silent 0


def test_expected_points_calibration_reports_signed_bias():
    calibration = dm.expected_points_calibration([0.5, 0.5], [1, 1], points=2)
    assert calibration["mean_predicted_points"] == pytest.approx(1.0)
    assert calibration["realised_points"] == pytest.approx(2.0)
    assert calibration["signed_bias_points"] == pytest.approx(-1.0)


def test_unknown_positions_and_empty_inputs_do_not_raise():
    assert dm.fit_count_statistics([]) == {}
    assert dm.dispersion_for({}, "MID") == dm.NB_POISSON_LIMIT_R
    unknown = dm.project_defcon(position="", expected_minutes=90.0, rate_posterior=5.0)
    assert unknown.threshold is None and unknown.p_threshold == 0.0
