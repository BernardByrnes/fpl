"""Structural calibration of the DEFCON threshold probability.

These tests pin the SHIPPED mapping's semantics.  A calibrator that is not
strictly monotone would silently reorder players — a better raw DEFCON candidate
could come out worse — so monotonicity and rank preservation are asserted, not
assumed.  Clipping at the extremes must also be total: ``p_raw`` of exactly 0 or
1 must not produce ``inf`` or ``nan``.
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import validate_defcon_calibration as vc  # noqa: E402

GRID = [1e-6 + (1 - 2e-6) * i / 200 for i in range(201)]


def _synthetic(n: int = 4000, seed: int = 11):
    rng = random.Random(seed)
    pairs = []
    for _ in range(n):
        p = min(0.98, max(0.002, rng.betavariate(1.2, 4.0)))
        # a deliberately OVER-CONFIDENT generator: the truth is a deflated version
        true_p = min(1.0 - 1e-9, max(1e-9, p * 0.55))
        pairs.append((p, 1 if rng.random() < true_p else 0))
    return pairs


def test_clipping_is_total_at_the_extremes():
    for extreme in (0.0, 1.0, -0.5, 1.5):
        assert math.isfinite(vc._logit(extreme))
        assert 0.0 < vc.platt_apply(extreme, (0.1, 0.9)) < 1.0
        assert 0.0 < vc.beta_apply(extreme, (0.1, 0.5, -0.5)) < 1.0


def test_platt_recovers_a_known_monotone_mapping():
    pairs = _synthetic()
    a, b = vc.platt_fit(pairs)
    # fitted on an over-confident source, so the slope must shrink the logit
    assert b > 0.0
    assert b < 1.0
    assert a < 0.0
    assert vc._monotone(lambda p: vc.platt_apply(p, (a, b)), GRID)


def test_beta_is_monotone_on_the_fitted_parameters():
    a, b, c = vc.beta_fit(_synthetic())
    assert vc._monotone(lambda p: vc.beta_apply(p, (a, b, c)), GRID)


def test_the_frozen_shipped_constants_are_monotone_and_rank_preserving():
    """The actual constants proposed for production."""

    platt = (-0.668302, 0.635381)
    beta = (-1.451446, 0.373994, -1.315049)
    assert vc._monotone(lambda p: vc.platt_apply(p, platt), GRID)
    assert vc._monotone(lambda p: vc.beta_apply(p, beta), GRID)

    raw = [p for p in GRID[::4]]
    assert vc._spearman(raw, [vc.platt_apply(p, platt) for p in raw]) == pytest.approx(1.0, abs=1e-9)
    assert vc._spearman(raw, [vc.beta_apply(p, beta) for p in raw]) == pytest.approx(1.0, abs=1e-9)


def test_rank_correlation_penalises_a_non_monotone_mapping():
    raw = [i / 100 for i in range(1, 100)]
    monotone = [p * 0.5 for p in raw]
    scrambled = [p * 0.5 if i % 2 else 0.99 for i, p in enumerate(raw)]
    assert vc._spearman(raw, monotone) == pytest.approx(1.0, abs=1e-9)
    assert vc._spearman(raw, scrambled) < 0.9


def test_isotonic_is_monotone_by_construction():
    steps = vc.isotonic_fit(_synthetic(n=2000), min_bin=40)
    assert steps
    levels = [level for _, level in steps]
    assert all(b >= a for a, b in zip(levels, levels[1:]))
    assert vc._monotone(lambda p: vc.isotonic_apply(p, steps), GRID)


def test_calibration_improves_an_overconfident_source_out_of_sample():
    pairs = _synthetic()
    train, test = pairs[:2000], pairs[2000:]
    a, b = vc.platt_fit(train)
    base = vc.dm.brier_score([p for p, _ in test], [y for _, y in test])
    cal = vc.dm.brier_score([vc.platt_apply(p, (a, b)) for p, _ in test], [y for _, y in test])
    assert cal < base


def test_logistic_fit_handles_a_separable_design_without_diverging():
    # all-positive outcomes with a strong covariate: a naive Newton step would
    # blow up; the ridge keeps the weights finite.
    design = [[1.0, 6.0 + i] for i in range(30)]
    w = vc._logistic_fit(design, [1] * 30)
    assert all(math.isfinite(x) for x in w)


def test_logistic_fit_returns_weights_in_design_column_order():
    """Pin the convention that was silently transposed in the first report.

    ``[1.0, x]`` must yield ``[intercept, slope]``.  A regression test is the
    right place for this: the failure mode is not an exception, it is two
    plausible numbers describing the wrong model, and it produced an apparent
    "negative calibration slope" that survived into a written report.
    """

    # a design whose intercept is clearly ~0 and slope clearly ~1
    design = [[1.0, math.log(0.5 / 0.5)] for _ in range(3)]  # degenerate, guard only
    w = vc._logistic_fit(design, [1, 0, 1])
    assert all(math.isfinite(x) for x in w)

    # a well-conditioned case: outcome probability rises steeply with x
    rng = random.Random(3)
    pairs = []
    for _ in range(2000):
        x = rng.uniform(-3.0, 3.0)
        p = vc._sigmoid(0.2 + 2.0 * x)
        pairs.append(([1.0, x], 1 if rng.random() < p else 0))
    w = vc._logistic_fit([d for d, _ in pairs], [y for _, y in pairs])
    assert w[0] == pytest.approx(0.2, abs=0.15)   # w[0] is the INTERCEPT
    assert w[1] == pytest.approx(2.0, abs=0.15)   # w[1] is the SLOPE
    # and the sign convention matters: a positive slope must not be reported negative
    assert w[1] > 0.0


def test_platt_fit_agrees_with_the_standard_calibration_regression():
    """Platt IS the standard logistic calibration regression — same estimand.

    ``platt_fit`` and a hand-rolled ``y ~ 1 + logit(p)`` design must return the
    same coefficients, since they are the same call.  If they ever diverge, one
    of the two has started fitting a different model.
    """

    pairs = _synthetic()
    a, b = vc.platt_fit(pairs)
    w = vc._logistic_fit([[1.0, vc._logit(p)] for p, _ in pairs], [y for _, y in pairs])
    assert (a, b) == pytest.approx((w[0], w[1]), rel=0, abs=1e-12)


def test_rank_correlation_below_one_is_ties_not_reversal():
    """Clipping collapses distinct raw values; it must never reverse them."""

    a, b = vc.platt_fit(_synthetic())
    raw = [1e-9, 1e-8, 1e-7, 0.01, 0.2, 0.6, 0.9]  # several below the clip floor
    cal = [vc.platt_apply(p, (a, b)) for p in raw]
    for i in range(len(raw)):
        for j in range(len(raw)):
            if raw[i] < raw[j]:
                assert cal[i] <= cal[j] + 1e-18  # monotone: never a strict reversal
    assert vc._spearman(raw, cal) <= 1.0
