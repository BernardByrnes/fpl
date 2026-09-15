"""Defensive-contribution (DEFCON) probability model — candidate V1.

TARGET AND AUTHORITY
--------------------
The target is the OFFICIAL FPL ``defensive_contribution`` value per player-match.
External process data NEVER replaces it: external action counts enter only as
explanatory context, never as the label or as a substitute for the official
number.

THE BASELINE IS THE BRAIN'S OWN MODEL
-------------------------------------
``xpts._defcon_p_hit`` already implements a DEFCON model:

    lam   = shrunk_actions_per90 * expected_minutes / 90
    P(hit) = P(Poisson(lam) >= threshold)

with the action rate shrunk toward a position-pooled prior by
``XPtsConfig.defcon_prior_strength_minutes``.  This module does not restate that
rule: the Poisson tail is taken from ``xpts.poisson_tail_probability`` (the very
function production calls) and the shrinkage is pinned by test against
``xpts._shrink_rate`` as the oracle, so the BASELINE variant is provably the
Brain's own arithmetic rather than a lookalike.  Any genuine improvement must
therefore beat production code, not a straw man.

WHY A COUNT MODEL AND NOT A NORMAL APPROXIMATION
------------------------------------------------
DEFCON is a threshold crossing on a small integer count with a hard cut at 10/12,
so the tail shape dominates the predicted probability.  Two candidate families
are provided:

* ``POISSON``      — the baseline (equidispersed).
* ``NEGATIVE_BINOMIAL`` — overdispersed: Var = lam + lam^2/r.  Defensive actions
  are bursty within a match, so the baseline's imposed Var = lam is a testable
  modelling assumption, not a fact.

The dispersion ``r`` is estimated from PRIOR matches only (method of moments per
position), never from the fold being scored.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from . import scoring_rules
from . import xpts

DEFCON_MODEL_VERSION = "defcon_v1.0.0"

#: Model variants, ordered from the production baseline outwards.
VARIANT_BASELINE_POISSON = "BASELINE_POISSON"
VARIANT_NEGATIVE_BINOMIAL = "NEGATIVE_BINOMIAL"
VARIANT_OPPONENT_POISSON = "OPPONENT_ADJUSTED_POISSON"
VARIANT_OPPONENT_NEGATIVE_BINOMIAL = "OPPONENT_ADJUSTED_NEGATIVE_BINOMIAL"
VARIANT_LEVEL_CALIBRATED_POISSON = "LEVEL_CALIBRATED_POISSON"
VARIANT_LEVEL_CALIBRATED_NEGATIVE_BINOMIAL = "LEVEL_CALIBRATED_NEGATIVE_BINOMIAL"
DEFCON_VARIANTS = (
    VARIANT_BASELINE_POISSON,
    VARIANT_NEGATIVE_BINOMIAL,
    VARIANT_OPPONENT_POISSON,
    VARIANT_OPPONENT_NEGATIVE_BINOMIAL,
    VARIANT_LEVEL_CALIBRATED_POISSON,
    VARIANT_LEVEL_CALIBRATED_NEGATIVE_BINOMIAL,
)

#: The required projection fields, named exactly as the phase brief specifies.
DEFCON_REQUIRED_OUTPUTS = (
    "DEFCON_RATE_POSTERIOR",
    "PROJECTED_DEFCON",
    "P_DEFCON_THRESHOLD",
    "EXPECTED_DEFCON_POINTS",
    "DEFCON_VARIANCE",
)

#: Dispersion used when a position shows no measurable overdispersion: large
#: ``r`` drives the negative binomial to the Poisson limit.
NB_POISSON_LIMIT_R = 1.0e6


@dataclass(frozen=True)
class DefconCountStatistics:
    """Per-position moment estimates used to fit overdispersion."""

    position: str
    matches: int
    mean_actions: float
    variance_actions: float

    @property
    def dispersion_r(self) -> float:
        """Method-of-moments ``r`` for Var = mu + mu^2/r."""

        excess = self.variance_actions - self.mean_actions
        if self.matches <= 1 or self.mean_actions <= 0.0 or excess <= 1e-9:
            return NB_POISSON_LIMIT_R
        return (self.mean_actions * self.mean_actions) / excess


def negative_binomial_tail(mean: float, dispersion_r: float, threshold: int) -> float:
    """P(N >= threshold) for N with mean ``mean`` and Var = mean + mean^2/r."""

    if threshold <= 0:
        return 1.0
    mean = max(0.0, float(mean))
    if mean <= 0.0:
        return 0.0
    r = float(dispersion_r)
    if not math.isfinite(r) or r <= 0.0 or r >= NB_POISSON_LIMIT_R:
        return xpts.poisson_tail_probability(mean, threshold)
    # P(N >= k) = 1 - I_{r/(r+mean)}(r, k) for integer k; evaluate the pmf sum
    # directly.  k is 10 or 12, so the sum is tiny and exact enough.
    p = r / (r + mean)
    log_p, log_q = math.log(p), math.log1p(-p)
    log_pmf = r * log_p
    cumulative = math.exp(log_pmf)
    for n in range(1, int(threshold)):
        log_pmf += math.log(n + r - 1) - math.log(n) + log_q
        cumulative += math.exp(log_pmf)
    return max(0.0, min(1.0, 1.0 - cumulative))


def count_variance(variant: str, mean: float, dispersion_r: float) -> float:
    """Model-implied variance of the action count at a fixed exposure."""

    mean = max(0.0, float(mean))
    if variant in (VARIANT_NEGATIVE_BINOMIAL, VARIANT_OPPONENT_NEGATIVE_BINOMIAL):
        r = float(dispersion_r)
        if r > 0.0 and r < NB_POISSON_LIMIT_R:
            return mean + (mean * mean) / r
    return mean


@dataclass(frozen=True)
class DefconProjection:
    """One player's DEFCON projection, carrying every required output."""

    player_id: int
    position: str
    threshold: int | None
    expected_minutes: float
    rate_posterior: float
    projected_defcon: float
    p_threshold: float
    expected_points: float
    variance: float
    variant: str
    dispersion_r: float
    prior_ess_minutes: float
    opponent_multiplier: float = 1.0
    level_scale: float = 1.0
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "player_id": int(self.player_id),
            "position": self.position,
            "threshold": self.threshold,
            "expected_minutes": round(float(self.expected_minutes), 6),
            "DEFCON_RATE_POSTERIOR": round(float(self.rate_posterior), 6),
            "PROJECTED_DEFCON": round(float(self.projected_defcon), 6),
            "P_DEFCON_THRESHOLD": round(float(self.p_threshold), 8),
            "EXPECTED_DEFCON_POINTS": round(float(self.expected_points), 8),
            "DEFCON_VARIANCE": round(float(self.variance), 6),
            "variant": self.variant,
            "dispersion_r": round(float(self.dispersion_r), 6),
            "prior_ess_minutes": float(self.prior_ess_minutes),
            "opponent_multiplier": round(float(self.opponent_multiplier), 6),
            "level_scale": round(float(self.level_scale), 6),
            "diagnostics": dict(self.diagnostics),
        }


def defcon_rate_posterior(
    *,
    observed_actions: float,
    observed_minutes: float,
    prior_rate_per90: float,
    prior_ess_minutes: float,
) -> float:
    """Shrunk actions-per-90 — the same shrinkage production applies.

    Pinned by test against ``xpts._shrink_rate`` so the baseline cannot drift
    from the engine.
    """

    if observed_minutes <= 0.0:
        return float(prior_rate_per90)
    observed_per90 = float(observed_actions) * 90.0 / float(observed_minutes)
    ess = float(prior_ess_minutes)
    return (float(observed_minutes) * observed_per90 + ess * float(prior_rate_per90)) / (
        float(observed_minutes) + ess
    )


def project_defcon(
    *,
    position: str,
    expected_minutes: float,
    rate_posterior: float,
    variant: str = VARIANT_BASELINE_POISSON,
    dispersion_r: float = NB_POISSON_LIMIT_R,
    opponent_multiplier: float = 1.0,
    level_scale: float = 1.0,
    prior_ess_minutes: float = 600.0,
    rules: scoring_rules.ScoringRules | None = None,
    diagnostics: Mapping[str, Any] | None = None,
    player_id: int = 0,
) -> DefconProjection:
    """Project one player-match's DEFCON outcome under a named variant."""

    resolved = rules or scoring_rules.DEFAULT_SCORING_RULES
    threshold = resolved.defcon_threshold_for(position)
    if threshold is None or position not in resolved.defcon_positions:
        return DefconProjection(
            player_id=player_id, position=position, threshold=None,
            expected_minutes=float(expected_minutes), rate_posterior=float(rate_posterior),
            projected_defcon=0.0, p_threshold=0.0, expected_points=0.0, variance=0.0,
            variant=variant, dispersion_r=float(dispersion_r), prior_ess_minutes=float(prior_ess_minutes),
            opponent_multiplier=float(opponent_multiplier), level_scale=float(level_scale),
            diagnostics=dict(diagnostics or {}),
        )

    multiplier = float(opponent_multiplier)
    effective_rate = float(rate_posterior)
    if variant in (VARIANT_OPPONENT_POISSON, VARIANT_OPPONENT_NEGATIVE_BINOMIAL):
        effective_rate = max(0.0, effective_rate * multiplier)
    # A level scale corrects a systematic rate bias; it is fitted on PRIOR
    # folds only, so it can never see the fold it scores.
    if variant in (VARIANT_LEVEL_CALIBRATED_POISSON, VARIANT_LEVEL_CALIBRATED_NEGATIVE_BINOMIAL):
        effective_rate = max(0.0, effective_rate * float(level_scale))
    mean_actions = max(0.0, effective_rate) * max(0.0, float(expected_minutes)) / 90.0

    if variant in (VARIANT_BASELINE_POISSON, VARIANT_OPPONENT_POISSON, VARIANT_LEVEL_CALIBRATED_POISSON):
        p_hit = xpts.poisson_tail_probability(mean_actions, threshold)
    else:
        p_hit = negative_binomial_tail(mean_actions, dispersion_r, threshold)

    return DefconProjection(
        player_id=player_id,
        position=position,
        threshold=int(threshold),
        expected_minutes=float(expected_minutes),
        rate_posterior=float(effective_rate),
        projected_defcon=mean_actions,
        p_threshold=float(p_hit),
        expected_points=float(resolved.defcon_points) * float(p_hit),
        variance=count_variance(variant, mean_actions, dispersion_r),
        variant=variant,
        dispersion_r=float(dispersion_r),
        prior_ess_minutes=float(prior_ess_minutes),
        opponent_multiplier=multiplier,
        level_scale=float(level_scale),
        diagnostics=dict(diagnostics or {}),
    )


def fit_count_statistics(
    observations: Sequence[Mapping[str, Any]],
) -> dict[str, DefconCountStatistics]:
    """Per-position moments of per-match action counts, from PRIOR data only.

    ``observations`` rows need ``position`` and ``actions``.  Callers must pass
    strictly earlier events than the fold being scored.
    """

    grouped: dict[str, list[float]] = {}
    for row in observations:
        position = str(row.get("position") or "")
        if not position:
            continue
        grouped.setdefault(position, []).append(float(row.get("actions") or 0.0))

    stats: dict[str, DefconCountStatistics] = {}
    for position, values in grouped.items():
        n = len(values)
        mean = sum(values) / n if n else 0.0
        variance = sum((v - mean) ** 2 for v in values) / (n - 1) if n > 1 else 0.0
        stats[position] = DefconCountStatistics(
            position=position, matches=n, mean_actions=mean, variance_actions=variance
        )
    return stats


def dispersion_for(
    statistics: Mapping[str, DefconCountStatistics], position: str
) -> float:
    entry = statistics.get(position)
    return NB_POISSON_LIMIT_R if entry is None else entry.dispersion_r


def brier_score(predictions: Sequence[float], outcomes: Sequence[int]) -> float:
    if not predictions:
        return float("nan")
    return sum((p - y) ** 2 for p, y in zip(predictions, outcomes)) / len(predictions)


def log_loss(predictions: Sequence[float], outcomes: Sequence[int], *, floor: float = 1e-12) -> float:
    if not predictions:
        return float("nan")
    total = 0.0
    for p, y in zip(predictions, outcomes):
        clamped = min(1.0 - floor, max(floor, float(p)))
        total += -(math.log(clamped) if y else math.log(1.0 - clamped))
    return total / len(predictions)


def calibration_table(
    predictions: Sequence[float], outcomes: Sequence[int], *, bins: int = 10
) -> list[dict[str, Any]]:
    """Mean predicted vs realised frequency per equal-width probability bin."""

    if not predictions:
        return []
    buckets: list[list[tuple[float, int]]] = [[] for _ in range(bins)]
    for p, y in zip(predictions, outcomes):
        index = min(bins - 1, max(0, int(float(p) * bins)))
        buckets[index].append((float(p), int(y)))
    table: list[dict[str, Any]] = []
    for index, bucket in enumerate(buckets):
        if not bucket:
            continue
        table.append(
            {
                "bin": index,
                "n": len(bucket),
                "mean_predicted": sum(p for p, _ in bucket) / len(bucket),
                "observed_rate": sum(y for _, y in bucket) / len(bucket),
            }
        )
    return table


def expected_points_calibration(
    predictions: Sequence[float], outcomes: Sequence[int], *, points: int = 2
) -> dict[str, float]:
    """Expected-vs-realised DEFCON points, the quantity a decision actually uses."""

    if not predictions:
        return {}
    mean_p = sum(predictions) / len(predictions)
    realised = sum(outcomes) / len(outcomes)
    return {
        "mean_predicted_points": points * mean_p,
        "realised_points": points * realised,
        "signed_bias_points": points * (mean_p - realised),
        "mean_predicted_probability": mean_p,
        "realised_hit_rate": realised,
    }
