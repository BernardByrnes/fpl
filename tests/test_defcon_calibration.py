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
