"""Deterministic metric engine: hand-computed expectations and edge cases.

Every expected value in this file was calculated by hand from the definition, not
produced by the function under test.  Where a convention could reasonably differ
(median of an even sample, tie ordering, the bias sign) the convention is written
into the test so a changed convention shows up as a failure rather than a quiet
reinterpretation.
"""

from __future__ import annotations

import json
import math

import pytest

from fpl_brain import walk_forward_metrics as wm


# ---------------------------------------------------------------------------
# Points metrics
# ---------------------------------------------------------------------------


def test_mean_absolute_error_matches_the_hand_computation():
    # errors: 3-4=-1, 5-2=+3, 2-2=0  ->  |.| = 1,3,0  ->  mean = 4/3
    metric = wm.mean_absolute_error([3.0, 5.0, 2.0], [4.0, 2.0, 2.0])
    assert metric.ok
    assert metric.n == 3
    assert metric.value == pytest.approx(4.0 / 3.0, rel=0, abs=1e-12)


def test_root_mean_squared_error_matches_the_hand_computation():
    # squares: 1, 9, 0  ->  mean = 10/3  ->  sqrt = 1.8257418583505538
    metric = wm.root_mean_squared_error([3.0, 5.0, 2.0], [4.0, 2.0, 2.0])
    assert metric.ok
    assert metric.value == pytest.approx(math.sqrt(10.0 / 3.0), rel=0, abs=1e-12)


def test_bias_is_predicted_minus_actual_so_positive_means_overprediction():
    # (3-4) + (5-2) + (2-2) = 2  ->  +2/3.  Predicted exceeded actual overall.
    metric = wm.mean_bias([3.0, 5.0, 2.0], [4.0, 2.0, 2.0])
    assert metric.ok
    assert metric.value == pytest.approx(2.0 / 3.0, rel=0, abs=1e-12)
    assert metric.value > 0, "the model predicted more than was realised"

    under = wm.mean_bias([1.0, 1.0], [4.0, 4.0])
    assert under.value == pytest.approx(-3.0, rel=0, abs=1e-12)
    assert under.value < 0, "the model predicted less than was realised"

    # Exactly the same numbers, swapped arms: the sign must flip, never stay put.
    assert wm.mean_bias([4.0, 2.0, 2.0], [3.0, 5.0, 2.0]).value == pytest.approx(
        -2.0 / 3.0, rel=0, abs=1e-12
    )


def test_median_absolute_error_uses_the_mean_of_the_two_central_values_when_even():
    # errors: 1-0=+1, 1-1=0, 1-3=-2, 1-5=-4  ->  |.| = 1,0,2,4  ->  sorted 0,1,2,4
    # even count, so the median is (1+2)/2 = 1.5, not the lower of the two.
    metric = wm.median_absolute_error([1.0, 1.0, 1.0, 1.0], [0.0, 1.0, 3.0, 5.0])
    assert metric.ok
    assert metric.value == pytest.approx(1.5, rel=0, abs=1e-12)


def test_median_absolute_error_on_an_odd_sample_is_the_middle_value():
    # |.| of (3-4, 5-2, 2-2) = 1,3,0  ->  sorted 0,1,3  ->  1
    assert wm.median_absolute_error([3.0, 5.0, 2.0], [4.0, 2.0, 2.0]).value == pytest.approx(
        1.0, rel=0, abs=1e-12
    )


def test_negative_and_zero_realised_points_are_ordinary_values():
    # A defender can genuinely score negative FPL points; nothing here may clip.
    metric = wm.mean_bias([0.0, 0.0, 0.0], [-2.0, 0.0, 3.0])
    assert metric.ok
    assert metric.value == pytest.approx(-1.0 / 3.0, rel=0, abs=1e-12)  # (2+0-3)/3
    assert wm.mean_absolute_error([0.0, 0.0, 0.0], [-2.0, 0.0, 3.0]).value == pytest.approx(
        5.0 / 3.0, rel=0, abs=1e-12
    )


# ---------------------------------------------------------------------------
# Empty and single-observation samples
# ---------------------------------------------------------------------------


def test_an_empty_sample_is_explicitly_unavailable_never_zero():
    for fn in (
        wm.mean_absolute_error,
        wm.root_mean_squared_error,
        wm.mean_bias,
        wm.median_absolute_error,
    ):
        metric = fn([], [])
        assert metric.status == wm.METRIC_NO_SAMPLE
        assert metric.value is None, f"{fn.__name__} invented a value for an empty sample"
        assert metric.n == 0


def test_a_single_observation_defines_the_error_metrics_but_not_a_correlation():
    assert wm.mean_absolute_error([5.0], [3.0]).value == pytest.approx(2.0)
    assert wm.mean_bias([5.0], [3.0]).value == pytest.approx(2.0)
    rank = wm.spearman_rank_correlation([5.0], [3.0])
    assert rank.status == wm.METRIC_INSUFFICIENT_SAMPLE
    assert rank.value is None, "a correlation over one point must not become 0"


# ---------------------------------------------------------------------------
# Spearman
# ---------------------------------------------------------------------------


def test_spearman_is_plus_one_for_a_perfect_ordering_and_minus_one_for_the_reverse():
    assert wm.spearman_rank_correlation([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]).value == pytest.approx(1.0)
    assert wm.spearman_rank_correlation([1.0, 2.0, 3.0], [30.0, 20.0, 10.0]).value == pytest.approx(-1.0)


def test_spearman_with_a_tie_matches_the_hand_computation():
    # x = 1,2,2,3  -> average ranks 1, 2.5, 2.5, 4
    # y = 1,2,3,4  -> average ranks 1, 2, 3, 4
    # mean of each = 2.5
    # covariance   = (-1.5)(-1.5) + 0 + 0 + (1.5)(1.5) = 4.5
    # spread_x     = 2.25 + 0 + 0 + 2.25 = 4.5
    # spread_y     = 2.25 + 0.25 + 0.25 + 2.25 = 5.0
    # rho          = 4.5 / sqrt(4.5 * 5.0) = 0.9486832980505138
    metric = wm.spearman_rank_correlation([1.0, 2.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0])
    assert metric.ok and metric.n == 4
    assert metric.value == pytest.approx(0.9486832980505138, rel=0, abs=1e-12)


def test_average_ranks_averages_the_tied_positions():
    assert wm.average_ranks([10.0, 10.0, 10.0]) == [2.0, 2.0, 2.0]
    assert wm.average_ranks([5.0, 1.0, 5.0, 9.0]) == [2.5, 1.0, 2.5, 4.0]


def test_a_constant_side_makes_the_correlation_undefined_not_zero_or_one():
    """The whole point of the undefined state: no ordering information exists."""

    for first, second in (
        ([5.0, 5.0, 5.0], [1.0, 2.0, 3.0]),   # constant predictions
        ([1.0, 2.0, 3.0], [5.0, 5.0, 5.0]),   # constant actuals
        ([5.0, 5.0], [5.0, 5.0]),
    ):
        metric = wm.spearman_rank_correlation(first, second)
        assert metric.status == wm.METRIC_ZERO_VARIANCE
        assert metric.value is None, "an undefined denominator must not be replaced by 0"


# ---------------------------------------------------------------------------
# Top-k
# ---------------------------------------------------------------------------


def test_top_k_hit_rate_matches_the_hand_computation():
    # k=2.  Predicted order by (score desc, id asc): (5,1), (5,2) -> {1,2}
    #       Actual    order:                          (5,2), (5,3) -> {2,3}
    # intersection = {2}, size 1, divided by the declared k=2 -> 0.5
    metric = wm.top_k_hit_rate(
        [(5.0, 1), (5.0, 2), (1.0, 3)], [(5.0, 2), (5.0, 3), (1.0, 1)], k=2
    )
    assert metric.ok and metric.n == 3
    assert metric.value == pytest.approx(0.5)


def test_top_k_hit_rate_is_one_for_identical_rankings_and_zero_for_disjoint_ones():
    rows = [(9.0, 1), (7.0, 2), (5.0, 3), (1.0, 4)]
    assert wm.top_k_hit_rate(rows, rows, k=2).value == pytest.approx(1.0)
    disjoint = [(9.0, 4), (7.0, 3), (5.0, 2), (1.0, 1)]
    assert wm.top_k_hit_rate(rows, disjoint, k=2).value == pytest.approx(0.0)


def test_the_tie_policy_is_load_bearing_not_decorative():
    """Ascending ids break ties; a descending policy would report a different score.

    Predicted top-2 (score desc, id asc) = {9, 4}.  Actual top-2 = {4, 1}.
    Intersection {4} -> 1/2 = 0.5.  Ties broken by DESCENDING id would instead give
    predicted {4, 9} against actual {4, 9} -> 1.0, so this asserts the declared rule
    is the one actually in force.
    """

    predicted = [(3.0, 9), (3.0, 4), (2.0, 1)]
    actual = [(3.0, 4), (2.5, 1), (2.5, 9)]
    assert wm.ranked_ids(predicted, 2) == [4, 9]
    assert wm.ranked_ids(actual, 2) == [4, 1]
    assert wm.top_k_hit_rate(predicted, actual, k=2).value == pytest.approx(0.5)


def test_ranked_ids_orders_by_score_descending_then_player_id_ascending():
    assert wm.ranked_ids([(3.0, 9), (3.0, 4), (7.0, 1)], 2) == [1, 4]
    assert wm.ranked_ids([(1.0, 5)], 1) == [5]


def test_a_population_smaller_than_the_declared_k_is_unavailable_not_rescaled():
    """k is declared and fixed; a short population must not quietly change it."""

    metric = wm.top_k_hit_rate([(1.0, 1), (2.0, 2)], [(1.0, 1), (2.0, 2)], k=20)
    assert metric.status == wm.METRIC_INSUFFICIENT_SAMPLE
    assert metric.value is None
    assert metric.n == 2


def test_a_duplicate_player_in_one_ranked_set_is_refused():
    with pytest.raises(wm.MetricInputError):
        wm.ranked_ids([(1.0, 7), (2.0, 7)], 1)


def test_top_k_defaults_to_the_declared_policy_constant():
    assert wm.TOP_K == 20
    assert wm.top_k_hit_rate([(1.0, 1)] * 1, [(1.0, 1)] * 1).status == wm.METRIC_INSUFFICIENT_SAMPLE


# ---------------------------------------------------------------------------
# Input hygiene
# ---------------------------------------------------------------------------


def test_misaligned_arms_are_refused():
    with pytest.raises(wm.MetricInputError):
        wm.mean_absolute_error([1.0, 2.0], [1.0])


def test_a_non_finite_value_is_refused_rather_than_propagated():
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(wm.MetricInputError):
            wm.mean_absolute_error([bad], [1.0])
        with pytest.raises(wm.MetricInputError):
            wm.mean_bias([1.0], [bad])


def test_a_non_numeric_value_is_refused():
    with pytest.raises(wm.MetricInputError):
        wm.mean_bias(["not a number"], [1.0])


def test_no_nan_is_ever_serialised():
    """An unavailable metric serialises as an explicit null plus a reason."""

    payload = wm.spearman_rank_correlation([1.0], [1.0]).as_dict()
    text = json.dumps(payload, sort_keys=True)
    assert "NaN" not in text and "Infinity" not in text
    assert payload["value"] is None
    assert payload["status"] == wm.METRIC_INSUFFICIENT_SAMPLE


def test_values_are_rounded_once_at_the_serialisation_boundary():
    metric = wm.mean_absolute_error([3.0, 5.0, 2.0], [4.0, 2.0, 2.0])
    assert metric.value == pytest.approx(1.3333333333333333, rel=0, abs=1e-12)
    assert metric.as_dict()["value"] == 1.333333, "the reported value rounds to 6 places"
    assert metric.as_dict()["n"] == 3


# ---------------------------------------------------------------------------
# Brier
# ---------------------------------------------------------------------------


def test_brier_matches_the_hand_computation():
    # (0.9-1)^2 + (0.2-0)^2 + (0.5-1)^2 + (1.0-1)^2
    #   = 0.01 + 0.04 + 0.25 + 0.00 = 0.30  ->  /4 = 0.075
    metric = wm.brier_score([0.9, 0.2, 0.5, 1.0], [1.0, 0.0, 1.0, 1.0])
    assert metric.ok and metric.n == 4
    assert metric.value == pytest.approx(0.075, rel=0, abs=1e-12)


def test_brier_boundary_probabilities_are_accepted():
    # A confident and correct forecast is 0; a confident and wrong one is 1.
    assert wm.brier_score([1.0], [1.0]).value == pytest.approx(0.0)
    assert wm.brier_score([0.0], [1.0]).value == pytest.approx(1.0)
    assert wm.brier_score([0.0], [0.0]).value == pytest.approx(0.0)


def test_an_out_of_range_probability_fails_closed_instead_of_being_clipped():
    for bad in (-0.0001, 1.0001, -1.0, 2.0):
        with pytest.raises(wm.MetricInputError) as failure:
            wm.brier_score([bad], [1.0])
        assert "outside [0, 1]" in str(failure.value)


def test_a_non_binary_outcome_fails_closed():
    with pytest.raises(wm.MetricInputError):
        wm.brier_score([0.5], [2.0])


def test_a_missing_probability_is_the_callers_exclusion_not_a_zero_here():
    """This function cannot see a missing value; passing 0.0 would be a substitution.

    The empty-sample case is the only representation of "nothing to score", and it
    is explicitly unavailable rather than a perfect score.
    """

    metric = wm.brier_score([], [])
    assert metric.status == wm.METRIC_NO_SAMPLE and metric.value is None


def test_brier_reference_score_is_the_base_rate_forecast():
    # base rate = 3/4 = 0.75  ->  (0.0625 + 0.5625 + 0.0625 + 0.0625)/4 = 0.1875
    metric = wm.brier_reference_score([1.0, 0.0, 1.0, 1.0])
    assert metric.value == pytest.approx(0.1875, rel=0, abs=1e-12)


# ---------------------------------------------------------------------------
# Quantile coverage
# ---------------------------------------------------------------------------


def test_central_interval_coverage_matches_the_hand_computation():
    # (1,5,1) inside, (2,2,3) outside, (3,10,3) inside on the boundary -> 2/3
    metric = wm.central_interval_coverage([1.0, 2.0, 3.0], [5.0, 2.0, 10.0], [1.0, 3.0, 3.0])
    assert metric.ok and metric.n == 3
    assert metric.value == pytest.approx(2.0 / 3.0, rel=0, abs=1e-12)


def test_boundary_equality_counts_as_covered_on_both_ends():
    assert wm.central_interval_coverage([4.0], [8.0], [4.0]).value == pytest.approx(1.0)
    assert wm.central_interval_coverage([4.0], [8.0], [8.0]).value == pytest.approx(1.0)
    assert wm.central_interval_coverage([4.0], [8.0], [9.0]).value == pytest.approx(0.0)


def test_a_reversed_interval_fails_closed():
    with pytest.raises(wm.MetricInputError) as failure:
        wm.central_interval_coverage([9.0], [3.0], [5.0])
    assert "exceeds" in str(failure.value)


def test_mean_interval_width_rejects_a_reversed_pair():
    """The width has its own guard: a reversed pair cannot describe an interval."""

    with pytest.raises(wm.MetricInputError) as failure:
        wm.mean_interval_width([9.0], [3.0])
    assert "exceeds" in str(failure.value)


def test_mean_interval_width_matches_the_hand_computation():
    # ((5-1) + (2-2) + (10-3)) / 3 = (4 + 0 + 7)/3 = 11/3
    metric = wm.mean_interval_width([1.0, 2.0, 3.0], [5.0, 2.0, 10.0])
    assert metric.value == pytest.approx(11.0 / 3.0, rel=0, abs=1e-12)


def test_coverage_over_an_empty_sample_is_unavailable():
    assert wm.central_interval_coverage([], [], []).status == wm.METRIC_NO_SAMPLE
    assert wm.mean_interval_width([], []).status == wm.METRIC_NO_SAMPLE


def test_the_policy_version_is_published():
    assert wm.METRIC_POLICY_VERSION == "wf_metrics_v1.0.0"
