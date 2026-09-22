"""PE-7 team attack/defence challenger — challenger-first, never promotion.

WHAT THIS IS
------------
PE-7 leaves the incumbent team model untouched: ``team_strength_v1.1.0`` and
``team_naive_v1.1.0`` are its identity, and this module neither renames nor
re-points them, nor writes through them.  A refinement must earn a promotion at
PE-7's terminal boundary; until then it lives here, under its own identity:

    ``TEAM_ATTACK_CHALLENGER_VERSION`` + ``TeamAttackChallengerConfig.config_hash()``

The challenger estimates the SAME quantities the incumbent publishes — a league
scoring level, a home advantage, one attack and one defence parameter per team —
and emits rows carrying the incumbent's own published field contract, so the two
arms are scored by identical code over identical populations.

WHAT IT CHANGES
---------------
Three declared refinement families, each inside PE-7's scope list, each ablated
on its own arm.

*Family 1 — ``league_level_natural_scale``.*  The incumbent's league level is the
weighted mean of ``log(xG + epsilon)``: a geometric-mean level whose value
depends on the declared epsilon and which sits below the arithmetic mean for any
spread of xG, because log is concave.  The challenger measures the level on the
NATURAL scale — the recency-weighted arithmetic mean xG, then logged — so the
level is the league's scoring level and carries no epsilon.

*Family 2 — ``venue_specific_level``.*  The incumbent measures home advantage as
a mean log-residual shrunk toward ZERO with a declared prior strength, and
estimates every team's attack against that single mixture level.  Two
consequences: a league-wide real home advantage is attenuated by construction,
and a team whose causal schedule is venue-unbalanced has part of its venue mix
folded into its attack.  The challenger measures the two venue levels directly
and shrinks the RATIO between them toward one with the incumbent's own declared
prior strength, so home advantage is an evidence-weighted ratio rather than a
residual pulled to null, and each team's parameters are estimated against the
venue levels.  The incumbent's ``lambda_for`` contract is preserved exactly: the
away level is published as ``league_log_baseline`` and the home/away log-ratio as
``home_advantage``, so the incumbent's own lambda function evaluates the
challenger's lambdas.

*Family 3 — ``data_derived_prior_strength``.*  The incumbent shrinks every team
toward the league level with a CONSTANT prior strength (12 match-equivalents by
declaration), whatever the league's own evidence says about how much teams
differ.  The challenger estimates the shrinkage strength from the evidence by
split-half reliability: each team's level-, venue- and opponent-adjusted residual
is split into two interleaved halves, the covariance of the two half-means
estimates the between-team signal, half their squared difference estimates the
noise, and Spearman-Brown gives the reliability of the full-window estimate.  The
implied strength is ``mean_team_weight * (1 - reliability) / reliability`` — small
where the league genuinely separates teams, larger where its spread is noise.
Where the window cannot support the estimate (too few teams, a half below the
declared weight floor, a non-positive between-team variance) the incumbent's
declared constant stands, and the fallback and its reason are recorded.

WHAT IT DELIBERATELY DOES NOT REFINE
------------------------------------
Recency weighting is IN PE-7's scope list and is a declared, inherited knob here
(``current_match_half_life=None`` means "the incumbent's own half-life"), but no
recency family is proposed.  Both arms apply the incumbent's weighting to the same
matches, so re-scaling the half-life would move a parameter PE-7 has no evidence
about rather than refine a quantity this phase can measure.  The knob exists so a
future review can exercise it under its own arm and its own measurement; it is not
used to make the challenger look different.

CAUSALITY IS UNCONDITIONAL, NOT A FAMILY
----------------------------------------
Every observation, every fixture identity and every parameter is read under the
PE-1 canonical boundary: fixture-side xG comes from
``team_model.team_match_rows`` -> ``team_model.completed_fixture_xg`` ->
``historical_observations.OBSERVATION_SQL_CLAUSES``, and fixture SIDE is decided
by ``player_gameweeks.was_home``, never ``players.team_id``, so a post-cutoff
transfer cannot re-attribute a historical fixture.  This module writes no SQL of
its own for evidence: it composes the incumbent's readers, so no second
historical-observation predicate exists here.

THE EQUIVALENCE PROPERTY
------------------------
With an EMPTY family set the challenger follows the incumbent's coordinate
descent step for step with the incumbent's own declared constants, so it
reproduces ``team_model.fit_team_strength`` bit for bit on the same evidence.
That is what makes the ablations readable: the delta between an arm and the
incumbent is the family, not a re-implementation.  A test pins it.
"""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass, field, fields
from typing import Any, Iterable, Mapping, Sequence

from . import analytics
from . import team_model as incumbent
from .utils import utc_now

# v0.1.0: initial PE-7 team challenger.  No promotion is performed here.
TEAM_ATTACK_CHALLENGER_VERSION = "team_attack_challenger_v0.1.0"
TEAM_ATTACK_CHALLENGER_FAMILY = "team_attack_challenger_pe7"

FAMILY_LEAGUE_LEVEL_NATURAL_SCALE = "league_level_natural_scale"
FAMILY_VENUE_SPECIFIC_LEVEL = "venue_specific_level"
FAMILY_DATA_DERIVED_PRIOR_STRENGTH = "data_derived_prior_strength"

#: Declared once, so every arm name, artifact and test derives from one list.
REFINEMENT_FAMILIES: tuple[str, ...] = (
    FAMILY_LEAGUE_LEVEL_NATURAL_SCALE,
    FAMILY_VENUE_SPECIFIC_LEVEL,
    FAMILY_DATA_DERIVED_PRIOR_STRENGTH,
)

ARM_ALL_REFINEMENTS = "challenger_all_refinements"

#: The incumbent team identities PE-7 must not silently rewrite, with the
#: literals the merged source at the PE-7 base carried.
FROZEN_TEAM_INCUMBENT_VERSIONS: dict[str, str] = {
    "TEAM_MODEL_VERSION": "team_strength_v1.1.0",
    "TEAM_BASELINE_MODEL_VERSION": "team_naive_v1.1.0",
}

# --- where a fitted value came from, published on the parameters AND the rows -
LEVEL_BASIS_NATURAL_SCALE = "LOG_OF_WEIGHTED_ARITHMETIC_MEAN_XG"
LEVEL_BASIS_WEIGHTED_MEAN_OF_LOGS = "WEIGHTED_MEAN_OF_LOG_XG_PLUS_EPSILON"
VENUE_BASIS_SHRUNK_LEVEL_RATIO = "SHRUNK_HOME_AWAY_LEVEL_RATIO"
VENUE_BASIS_RESIDUAL_MEAN = "WEIGHTED_MEAN_RESIDUAL_TOWARD_ZERO"
PRIOR_BASIS_SPLIT_HALF_RELIABILITY = "SPLIT_HALF_RELIABILITY"
PRIOR_BASIS_INCUMBENT_FALLBACK = "INCUMBENT_DECLARED_CONSTANT_FALLBACK"

# --- declared fail-closed reasons for the reliability estimate ---------------
RELIABILITY_REASON_TOO_FEW_TEAMS = "TOO_FEW_TEAMS_WITH_TWO_USABLE_HALVES"
RELIABILITY_REASON_NON_POSITIVE_BETWEEN_VARIANCE = "NON_POSITIVE_BETWEEN_TEAM_VARIANCE"
RELIABILITY_REASON_NO_HALF_NOISE = "NO_POSITIVE_HALF_SAMPLE_NOISE"

#: Tokens describing the derived strength's provenance for senior review.
STRENGTH_DERIVED = "DERIVED_FROM_THE_CUTOFFS_OWN_EVIDENCE"
STRENGTH_DERIVED_CLAMPED = "DERIVED_AND_CLAMPED_TO_THE_DECLARED_BOUND"
STRENGTH_FALLBACK = "DECLARED_CONSTANT_FALLBACK"

# --- challenger-only risk flags ---------------------------------------------
FLAG_LEVEL_NATURAL_SCALE_UNAVAILABLE = "LEVEL_NATURAL_SCALE_UNAVAILABLE"
FLAG_VENUE_LEVEL_SPLIT_UNAVAILABLE = "VENUE_LEVEL_SPLIT_UNAVAILABLE"
FLAG_DERIVED_PRIOR_STRENGTH = "DERIVED_PRIOR_STRENGTH"
FLAG_DERIVED_PRIOR_STRENGTH_UNAVAILABLE = "DERIVED_PRIOR_STRENGTH_UNAVAILABLE"
FLAG_DERIVED_PRIOR_STRENGTH_CLAMPED = "DERIVED_PRIOR_STRENGTH_CLAMPED"
FLAG_DERIVED_PRIOR_STRENGTH_ABOVE_DECLARED = "DERIVED_PRIOR_STRENGTH_ABOVE_DECLARED"
FLAG_DERIVED_PRIOR_STRENGTH_BELOW_DECLARED = "DERIVED_PRIOR_STRENGTH_BELOW_DECLARED"
FLAG_SPARSE_TEAM_EVIDENCE = "SPARSE_TEAM_EVIDENCE"

#: PE-7 compares the challenger with the INCUMBENT only.  The frozen naive
#: baseline is untouched and is deliberately not an arm here: a second comparator
#: would re-open a comparison that belongs to PE-2's scoreboard, not to PE-7.
COMPARATOR_NOTE = (
    "the incumbent is the comparator; the frozen naive team baseline is unchanged and is not "
    "an arm of this evaluation"
)


class ChallengerInconsistencyError(RuntimeError):
    """The challenger and the incumbent disagree about the evidence they share.

    Fail closed: a disagreement means one of them is describing a different
    world, and a silently reconciled number is exactly what PE-7 forbids.
    """


# ---------------------------------------------------------------------------
# Configuration and identity.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TeamAttackChallengerConfig:
    """Every challenger knob in one versioned structure; no magic numbers.

    ``None`` means "inherit the incumbent's declared value", so the challenger
    introduces no second knob for a quantity the incumbent already declares, and
    the empty-family equivalence is not a coincidence of copied defaults.
    """

    # --- inherited (None -> the incumbent's TeamStrengthConfig) -------------
    current_match_half_life: float | None = None
    attack_prior_strength: float | None = None
    defence_prior_strength: float | None = None
    home_advantage_prior_strength: float | None = None
    iterations: int | None = None
    xg_epsilon: float | None = None
    low_evidence_matches: int | None = None
    home_advantage_low_evidence_matches: int | None = None
    high_regularisation_share: float | None = None
    xg_xgc_tolerance: float | None = None

    # --- declared disclosure policy for the reliability estimate ------------
    #: Teams with two usable halves required before a derived strength is used at
    #: all.  A reliability estimated from a handful of teams is not evidence about
    #: the league, so below this the incumbent's declared constant stands.
    reliability_min_teams: int = 8
    #: Matches a team needs before it is split into two halves at all.
    reliability_min_matches_per_team: int = 4
    #: Recency weight each half must carry for the team to be counted.
    reliability_min_half_weight: float = 1.5
    #: A hard declared bound on the derived strength.  Beyond it the prior stops
    #: being evidence-weighted and becomes an assertion, so the value is clamped
    #: and the clamp is reported rather than hidden.
    reliability_max_prior_strength: float = 60.0

    def resolved(self, incumbent_config: "incumbent.TeamStrengthConfig") -> dict[str, Any]:
        """The effective values, with ``None`` inheriting from the incumbent."""

        def pick(value: Any, fallback: Any) -> Any:
            return fallback if value is None else value

        return {
            "current_match_half_life": float(
                pick(self.current_match_half_life, incumbent_config.current_match_half_life)
            ),
            "attack_prior_strength": float(
                pick(self.attack_prior_strength, incumbent_config.attack_prior_strength)
            ),
            "defence_prior_strength": float(
                pick(self.defence_prior_strength, incumbent_config.defence_prior_strength)
            ),
            "home_advantage_prior_strength": float(
                pick(self.home_advantage_prior_strength, incumbent_config.home_advantage_prior_strength)
            ),
            "iterations": int(pick(self.iterations, incumbent_config.iterations)),
            "xg_epsilon": float(pick(self.xg_epsilon, incumbent_config.xg_epsilon)),
            "low_evidence_matches": int(
                pick(self.low_evidence_matches, incumbent_config.low_evidence_matches)
            ),
            "home_advantage_low_evidence_matches": int(
                pick(
                    self.home_advantage_low_evidence_matches,
                    incumbent_config.home_advantage_low_evidence_matches,
                )
            ),
            "high_regularisation_share": float(
                pick(self.high_regularisation_share, incumbent_config.high_regularisation_share)
            ),
            "xg_xgc_tolerance": float(pick(self.xg_xgc_tolerance, incumbent_config.xg_xgc_tolerance)),
            "reliability_min_teams": int(self.reliability_min_teams),
            "reliability_min_matches_per_team": int(self.reliability_min_matches_per_team),
            "reliability_min_half_weight": float(self.reliability_min_half_weight),
            "reliability_max_prior_strength": float(self.reliability_max_prior_strength),
        }

    def config_hash(self) -> str:
        values = {item.name: getattr(self, item.name) for item in fields(self)}
        return analytics.canonical_hash({"model": TEAM_ATTACK_CHALLENGER_VERSION, **values})


def frozen_incumbent_identity() -> dict[str, Any]:
    """The incumbent team identities, verified against their declared literals."""

    actual = {name: str(getattr(incumbent, name)) for name in FROZEN_TEAM_INCUMBENT_VERSIONS}
    mismatches = {
        name: {"declared": declared, "actual": actual.get(name)}
        for name, declared in FROZEN_TEAM_INCUMBENT_VERSIONS.items()
        if actual.get(name) != declared
    }
    return {
        "values": actual,
        "declared": dict(FROZEN_TEAM_INCUMBENT_VERSIONS),
        "unchanged": not mismatches,
        "mismatches": mismatches,
    }


def challenger_identity(
    config: TeamAttackChallengerConfig | None = None,
    incumbent_config: "incumbent.TeamStrengthConfig | None" = None,
) -> dict[str, Any]:
    """The challenger's own identity, carried by every artifact it produces."""

    config = config or TeamAttackChallengerConfig()
    incumbent_config = incumbent_config or incumbent.TeamStrengthConfig()
    return {
        "family": TEAM_ATTACK_CHALLENGER_FAMILY,
        "challenger_version": TEAM_ATTACK_CHALLENGER_VERSION,
        "challenger_config_hash": config.config_hash(),
        "resolved_parameters": config.resolved(incumbent_config),
        "refinement_families": list(REFINEMENT_FAMILIES),
        "incumbent_identity": frozen_incumbent_identity(),
        "incumbent_config_hash": incumbent_config.config_hash(),
        "comparator": COMPARATOR_NOTE,
        "promotion": "NOT_PERFORMED_CHALLENGER_ONLY",
    }


def arm_name_for_family(family: str) -> str:
    return f"challenger_only::{family}"


def default_arm_definitions() -> dict[str, frozenset[str]]:
    """The full challenger plus one single-family ablation per refinement."""

    definitions: dict[str, frozenset[str]] = {
        ARM_ALL_REFINEMENTS: frozenset(REFINEMENT_FAMILIES)
    }
    for family in REFINEMENT_FAMILIES:
        definitions[arm_name_for_family(family)] = frozenset({family})
    return definitions


# ---------------------------------------------------------------------------
# Small math mirrors of the incumbent's own estimators.
# ---------------------------------------------------------------------------


def _shrink(numerator: float, denominator: float, prior_strength: float) -> float:
    """The incumbent's ridge/shrinkage estimator: ``num / (den + strength)``.

    Restated rather than reached for as a private incumbent symbol; a test pins
    the two to the same value on the same inputs.
    """

    total = denominator + float(prior_strength)
    return numerator / total if total > 0 else 0.0


def _weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float | None:
    total = sum(weights)
    if total <= 0:
        return None
    return sum(value * weight for value, weight in zip(values, weights)) / total


def _positive_log(value: float | None) -> float | None:
    if value is None or not math.isfinite(float(value)) or float(value) <= 0.0:
        return None
    return math.log(float(value))


# ---------------------------------------------------------------------------
# League scoring level and home advantage.
# ---------------------------------------------------------------------------


def _level_of(rows: Sequence[Mapping[str, Any]], resolved: Mapping[str, Any], natural_scale: bool) -> float | None:
    """One level for one row set, in the declared basis."""

    weights = [float(row["weight"]) for row in rows]
    if not rows:
        return None
    if natural_scale:
        mean_xg = _weighted_mean([float(row["xg_for"]) for row in rows], weights)
        return _positive_log(mean_xg)
    epsilon = float(resolved["xg_epsilon"])
    return _weighted_mean([math.log(float(row["xg_for"]) + epsilon) for row in rows], weights)


def league_level(
    rows: Sequence[Mapping[str, Any]],
    resolved: Mapping[str, Any],
    families: Iterable[str],
) -> dict[str, Any]:
    """The published level, the venue ratio when the family is active, and why.

    Family 1 decides HOW a level is measured (natural scale versus the
    epsilon form) and family 2 decides WHICH levels exist (one mixture level
    plus a residual home advantage, or two venue levels and their ratio).  With
    family 2 active the AWAY level is published as ``league_log_baseline`` and the
    home/away log-ratio as ``home_advantage``, so the incumbent's own
    ``lambda_for`` reproduces both venue levels exactly whatever the weight mix.

    ``home_advantage`` is ``None`` when the venue family is inactive or cannot be
    identified: the descent then estimates it exactly as the incumbent does.
    """

    families = frozenset(families)
    natural_scale = FAMILY_LEAGUE_LEVEL_NATURAL_SCALE in families
    flags: list[str] = []
    overall = _level_of(rows, resolved, natural_scale)
    level_basis = LEVEL_BASIS_NATURAL_SCALE if natural_scale else LEVEL_BASIS_WEIGHTED_MEAN_OF_LOGS
    if overall is None:
        # Every observed side scored exactly zero xG, so the natural-scale level
        # has no logarithm.  The incumbent's epsilon form is used and the reason
        # is recorded rather than a number being invented.
        overall = _level_of(rows, resolved, False)
        level_basis = LEVEL_BASIS_WEIGHTED_MEAN_OF_LOGS
        flags.append(FLAG_LEVEL_NATURAL_SCALE_UNAVAILABLE)

    venue_advantage: float | None = None
    venue_basis = VENUE_BASIS_RESIDUAL_MEAN
    baseline = overall
    if FAMILY_VENUE_SPECIFIC_LEVEL in families:
        home = [row for row in rows if row["venue"] == incumbent.VENUE_HOME]
        away = [row for row in rows if row["venue"] == incumbent.VENUE_AWAY]
        home_level = _level_of(home, resolved, natural_scale)
        away_level = _level_of(away, resolved, natural_scale)
        if home_level is None or away_level is None:
            # One venue carries no positive evidence, so the ratio is not
            # identified.  The residual-mean path stands and the gap is declared.
            flags.append(FLAG_VENUE_LEVEL_SPLIT_UNAVAILABLE)
        else:
            home_weight = sum(float(row["weight"]) for row in home)
            venue_advantage = _shrink(
                home_weight * (home_level - away_level),
                home_weight,
                float(resolved["home_advantage_prior_strength"]),
            )
            baseline = away_level
            venue_basis = VENUE_BASIS_SHRUNK_LEVEL_RATIO

    return {
        "league_log_baseline": baseline,
        "home_advantage": venue_advantage,
        "level_basis": level_basis,
        "venue_basis": venue_basis,
        "venue_split_identified": venue_advantage is not None,
        "flags": sorted(set(flags)),
    }


# ---------------------------------------------------------------------------
# Coordinate descent (the incumbent's own recursion, parameterised).
# ---------------------------------------------------------------------------


def _descent(
    rows: Sequence[Mapping[str, Any]],
    teams: Sequence[int],
    level: Mapping[str, Any],
    resolved: Mapping[str, Any],
    strengths: Mapping[str, float],
) -> dict[str, Any]:
    """The incumbent's coordinate descent, with the level and strengths supplied.

    Step for step this is ``team_model.fit_team_strength``'s loop: a home-advantage
    step over home rows (skipped when the venue ratio already fixes it), an attack
    step per team, a defence step per team, the same shrinkage estimator, the same
    iteration count, and the same weighted-centring of the two parameter vectors
    with the shift folded into the league baseline so every fitted lambda is
    unchanged.
    """

    epsilon = float(resolved["xg_epsilon"])
    baseline = float(level["league_log_baseline"])
    venue_fixed = level.get("home_advantage") is not None
    home_advantage = float(level["home_advantage"]) if venue_fixed else 0.0
    attack = {int(team): 0.0 for team in teams}
    defence = {int(team): 0.0 for team in teams}

    for _ in range(int(resolved["iterations"])):
        if not venue_fixed:
            num = den = 0.0
            for row in rows:
                if row["venue"] != incumbent.VENUE_HOME:
                    continue
                residual = (
                    math.log(float(row["xg_for"]) + epsilon)
                    - baseline
                    - attack[int(row["team_id"])]
                    - defence[int(row["opponent_id"])]
                )
                num += float(row["weight"]) * residual
                den += float(row["weight"])
            home_advantage = _shrink(num, den, float(strengths["home_advantage"]))

        for team in teams:
            num = den = 0.0
            for row in rows:
                if int(row["team_id"]) != int(team):
                    continue
                residual = (
                    math.log(float(row["xg_for"]) + epsilon)
                    - baseline
                    - (home_advantage if row["venue"] == incumbent.VENUE_HOME else 0.0)
                    - defence[int(row["opponent_id"])]
                )
                num += float(row["weight"]) * residual
                den += float(row["weight"])
            attack[int(team)] = _shrink(num, den, float(strengths["attack"]))

        for team in teams:
            num = den = 0.0
            for row in rows:
                if int(row["team_id"]) != int(team):
                    continue
                opponent_home = 0.0 if row["venue"] == incumbent.VENUE_HOME else 1.0
                residual = (
                    math.log(float(row["xg_against"]) + epsilon)
                    - baseline
                    - home_advantage * opponent_home
                    - attack[int(row["opponent_id"])]
                )
                num += float(row["weight"]) * residual
                den += float(row["weight"])
            defence[int(team)] = _shrink(num, den, float(strengths["defence"]))

    weight_totals = {
        int(team): sum(float(row["weight"]) for row in rows if int(row["team_id"]) == int(team))
        for team in teams
    }
    total_team_weight = sum(weight_totals.values()) or 1.0
    mean_attack = sum(weight_totals[t] * attack[t] for t in attack) / total_team_weight
    mean_defence = sum(weight_totals[t] * defence[t] for t in defence) / total_team_weight
    attack = {team: value - mean_attack for team, value in attack.items()}
    defence = {team: value - mean_defence for team, value in defence.items()}
    baseline += mean_attack + mean_defence
    return {
        "league_log_baseline": baseline,
        "home_advantage": home_advantage,
        "attack": attack,
        "defence": defence,
        "weight_totals": weight_totals,
        "total_team_weight": total_team_weight,
        "home_advantage_source": (
            VENUE_BASIS_SHRUNK_LEVEL_RATIO if venue_fixed else VENUE_BASIS_RESIDUAL_MEAN
        ),
    }


def adjusted_residuals(
    rows: Sequence[Mapping[str, Any]],
    params: Mapping[str, Any],
    resolved: Mapping[str, Any],
    component: str,
) -> dict[int, list[dict[str, Any]]]:
    """Each team's own deviation per match, venue- and opponent-adjusted.

    Attack: ``log(xG for + eps) - level - venue - defence[opponent]``.
    Defence: ``log(xG against + eps) - level - opponent venue - attack[opponent]``.

    These are the incumbent's OWN residual definitions, evaluated on one fitted
    parameter set, which is why the reliability below is measured on the model's
    residuals rather than on a hand-rolled proxy for them.
    """

    epsilon = float(resolved["xg_epsilon"])
    baseline = float(params["league_log_baseline"])
    home_advantage = float(params["home_advantage"])
    out: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        was_home = row["venue"] == incumbent.VENUE_HOME
        if component == "attack":
            signal = (
                math.log(float(row["xg_for"]) + epsilon)
                - baseline
                - (home_advantage if was_home else 0.0)
                - float(params["defence"].get(int(row["opponent_id"]), 0.0))
            )
        elif component == "defence":
            signal = (
                math.log(float(row["xg_against"]) + epsilon)
                - baseline
                - (0.0 if was_home else home_advantage)
                - float(params["attack"].get(int(row["opponent_id"]), 0.0))
            )
        else:  # pragma: no cover - guarded by the caller's declared component list
            raise ValueError(f"unsupported component: {component}")
        out.setdefault(int(row["team_id"]), []).append(
            {
                "age_matches": int(row["age_matches"]),
                "weight": float(row["weight"]),
                "signal": signal,
            }
        )
    for records in out.values():
        records.sort(key=lambda item: item["age_matches"])
    return out


def split_half_reliability(
    residuals: Mapping[int, Sequence[Mapping[str, Any]]],
    resolved: Mapping[str, Any],
    *,
    component: str,
    declared_fallback: float,
) -> dict[str, Any]:
    """Split-half reliability of the teams' own signals, and the implied strength.

    Matches are split INTERLEAVED by recency (oldest, third-oldest, ... in one
    half; second-oldest, fourth-oldest, ... in the other) rather than into a
    contiguous past and present block: an alternating split is not confounded
    with the recency weighting that produced the two halves' weights.

    Between-team signal: the weighted covariance of the two half-means (their
    noises are independent, so the covariance is all signal).  Noise: half the
    weighted mean squared difference of the two half-means.  Spearman-Brown turns
    the half-sample reliability into the full-window reliability, and the implied
    prior strength follows from it in the units the shrinkage estimator uses
    (match-equivalents of weight, against the league's own mean team weight).

    Fail closed: too few teams, a half below the declared weight floor, or a
    non-positive between-team covariance all mean the data cannot answer the
    question, and then the incumbent's declared constant stands with its reason.
    """

    min_teams = int(resolved["reliability_min_teams"])
    min_half_weight = float(resolved["reliability_min_half_weight"])
    max_strength = float(resolved["reliability_max_prior_strength"])
    usable: list[tuple[float, float, float, float]] = []  # own_weight, wA, meanA, meanB
    excluded = 0
    for team in sorted(residuals):
        records = list(residuals[team])
        if len(records) < 2:
            excluded += 1
            continue
        first = records[0::2]
        second = records[1::2]
        weight_first = sum(float(item["weight"]) for item in first)
        weight_second = sum(float(item["weight"]) for item in second)
        if not first or not second or min(weight_first, weight_second) < min_half_weight:
            excluded += 1
            continue
        mean_first = sum(float(item["weight"]) * float(item["signal"]) for item in first) / weight_first
        mean_second = sum(float(item["weight"]) * float(item["signal"]) for item in second) / weight_second
        usable.append((min(weight_first, weight_second), weight_first, mean_first, mean_second))

    base = {
        "component": component,
        "declared_fallback_prior_strength": round(float(declared_fallback), 6),
        "teams_examined": len(residuals),
        "teams_used": len(usable),
        "teams_excluded": excluded,
        "min_teams_required": min_teams,
        "min_half_weight_required": min_half_weight,
        "max_prior_strength": max_strength,
    }
    if len(usable) < min_teams:
        return {
            **base,
            "prior_strength": round(float(declared_fallback), 6),
            "basis": PRIOR_BASIS_INCUMBENT_FALLBACK,
            "status": STRENGTH_FALLBACK,
            "reason": RELIABILITY_REASON_TOO_FEW_TEAMS,
            "reliability_full_window": None,
            "between_team_covariance": None,
            "half_sample_noise_variance": None,
            "mean_team_weight": None,
        }

    own_weight_total = sum(item[0] for item in usable)
    weighted_first = sum(item[0] * item[2] for item in usable) / own_weight_total
    weighted_second = sum(item[0] * item[3] for item in usable) / own_weight_total
    between = (
        sum(item[0] * (item[2] - weighted_first) * (item[3] - weighted_second) for item in usable)
        / own_weight_total
    )
    noise = (
        sum(item[0] * (item[2] - item[3]) ** 2 for item in usable) / (2.0 * own_weight_total)
    )
    if between <= 0.0:
        return {
            **base,
            "prior_strength": round(float(declared_fallback), 6),
            "basis": PRIOR_BASIS_INCUMBENT_FALLBACK,
            "status": STRENGTH_FALLBACK,
            "reason": RELIABILITY_REASON_NON_POSITIVE_BETWEEN_VARIANCE,
            "reliability_full_window": None,
            "between_team_covariance": round(between, 12),
            "half_sample_noise_variance": round(noise, 12),
            "mean_team_weight": None,
        }
    if between + noise <= 0.0 or noise <= 0.0:
        # Two different refusals share this exit.  ``between + noise <= 0`` cannot
        # arise once between > 0, and is kept so the division below is guarded by a
        # stated condition rather than by the algebra above.  A ZERO noise term
        # would imply a perfectly reliable signal and therefore NO shrinkage at
        # all; on a store whose xG is rounded, two identical half-means are an
        # artefact of the stored precision, not evidence that team strength is
        # noise-free, so the declared constant stands instead.
        return {
            **base,
            "prior_strength": round(float(declared_fallback), 6),
            "basis": PRIOR_BASIS_INCUMBENT_FALLBACK,
            "status": STRENGTH_FALLBACK,
            "reason": RELIABILITY_REASON_NO_HALF_NOISE,
            "reliability_full_window": None,
            "between_team_covariance": round(between, 12),
            "half_sample_noise_variance": round(noise, 12),
            "mean_team_weight": None,
        }

    reliability_half = between / (between + noise)
    reliability_full = (2.0 * reliability_half) / (1.0 + reliability_half)
    mean_team_weight = sum(
        sum(float(item["weight"]) for item in records) for records in residuals.values()
    ) / len(residuals)
    raw_strength = mean_team_weight * (1.0 - reliability_full) / reliability_full if reliability_full > 0 else math.inf
    clamped = min(max(raw_strength, 0.0), max_strength)
    status = STRENGTH_DERIVED if clamped == raw_strength else STRENGTH_DERIVED_CLAMPED
    return {
        **base,
        "prior_strength": round(clamped, 6),
        "basis": PRIOR_BASIS_SPLIT_HALF_RELIABILITY,
        "status": status,
        "reason": None,
        "reliability_half_sample": round(reliability_half, 6),
        "reliability_full_window": round(reliability_full, 6),
        "between_team_covariance": round(between, 12),
        "half_sample_noise_variance": round(noise, 12),
        "mean_team_weight": round(mean_team_weight, 6),
        "unclamped_prior_strength": round(raw_strength, 6),
    }


# ---------------------------------------------------------------------------
# The fit.
# ---------------------------------------------------------------------------


def _empty_params(resolved: Mapping[str, Any], config: TeamAttackChallengerConfig, families: frozenset[str]) -> dict[str, Any]:
    return {
        "league_log_baseline": None,
        "home_advantage": 0.0,
        "attack": {},
        "defence": {},
        "teams": [],
        "team_match_counts": {},
        "team_weight_totals": {},
        "team_reconciliation_max": {},
        "home_match_weight": 0.0,
        "match_count": 0,
        "total_team_weight": 0.0,
        "config_hash": config.config_hash(),
        "data_gaps": ["no completed fixtures with official xG before the cutoff"],
        "challenger_version": TEAM_ATTACK_CHALLENGER_VERSION,
        "challenger_config_hash": config.config_hash(),
        "challenger_families": sorted(families),
        "level_basis": None,
        "venue_basis": None,
        "venue_split_identified": False,
        "attack_prior_strength_used": None,
        "defence_prior_strength_used": None,
        "home_advantage_prior_strength_used": None,
        "prior_strength_estimates": {},
        "level_flags": [],
    }


def fit_challenger_team_strength(
    conn: sqlite3.Connection,
    planning_event: int,
    cutoff: str,
    *,
    config: TeamAttackChallengerConfig | None = None,
    incumbent_config: "incumbent.TeamStrengthConfig | None" = None,
    families: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Fit the league level, the venue ratio and the team parameters.

    The evidence is the incumbent's own ``team_match_rows`` under the PE-1
    boundary, recency-weighted with the incumbent's own weighting.  Family 3, when
    active, runs the descent TWICE: once with the incumbent's declared constants
    to produce the residuals the reliability is measured on, and once with the
    derived strengths.  Every value published here is deterministic in the
    cutoff's evidence.
    """

    config = config or TeamAttackChallengerConfig()
    incumbent_config = incumbent_config or incumbent.TeamStrengthConfig()
    family_set = frozenset(REFINEMENT_FAMILIES if families is None else families)
    resolved = config.resolved(incumbent_config)

    rows = incumbent.attach_recency_weights(
        incumbent.team_match_rows(conn, int(planning_event), cutoff),
        float(resolved["current_match_half_life"]),
    )
    if not rows:
        return _empty_params(resolved, config, family_set)

    teams = sorted({int(row["team_id"]) for row in rows})
    level = league_level(rows, resolved, family_set)
    declared = {
        "attack": float(resolved["attack_prior_strength"]),
        "defence": float(resolved["defence_prior_strength"]),
        "home_advantage": float(resolved["home_advantage_prior_strength"]),
    }
    strengths = dict(declared)
    estimates: dict[str, Any] = {}
    if FAMILY_DATA_DERIVED_PRIOR_STRENGTH in family_set:
        first_pass = _descent(rows, teams, level, resolved, declared)
        for component in ("attack", "defence"):
            estimate = split_half_reliability(
                adjusted_residuals(rows, first_pass, resolved, component),
                resolved,
                component=component,
                declared_fallback=declared[component],
            )
            estimates[component] = estimate
            strengths[component] = float(estimate["prior_strength"])

    fitted = _descent(rows, teams, level, resolved, strengths)

    team_match_counts = {team: sum(1 for row in rows if int(row["team_id"]) == team) for team in teams}
    team_reconciliation_max = {
        team: max([float(row["reconciliation_diff"]) for row in rows if int(row["team_id"]) == team] or [0.0])
        for team in teams
    }
    home_match_weight = sum(
        float(row["weight"]) for row in rows if row["venue"] == incumbent.VENUE_HOME
    )
    flags = list(level["flags"])
    for component, estimate in sorted(estimates.items()):
        if estimate["status"] == STRENGTH_FALLBACK:
            flags.append(FLAG_DERIVED_PRIOR_STRENGTH_UNAVAILABLE)
        if estimate["status"] == STRENGTH_DERIVED_CLAMPED:
            flags.append(FLAG_DERIVED_PRIOR_STRENGTH_CLAMPED)
        if float(estimate["prior_strength"]) > declared[component]:
            flags.append(FLAG_DERIVED_PRIOR_STRENGTH_ABOVE_DECLARED)
        elif float(estimate["prior_strength"]) < declared[component]:
            flags.append(FLAG_DERIVED_PRIOR_STRENGTH_BELOW_DECLARED)
        if estimate["basis"] == PRIOR_BASIS_SPLIT_HALF_RELIABILITY:
            flags.append(FLAG_DERIVED_PRIOR_STRENGTH)

    return {
        "league_log_baseline": round(float(fitted["league_log_baseline"]), 6),
        "home_advantage": round(float(fitted["home_advantage"]), 6),
        "attack": {int(team): round(float(value), 6) for team, value in fitted["attack"].items()},
        "defence": {int(team): round(float(value), 6) for team, value in fitted["defence"].items()},
        "teams": teams,
        "team_match_counts": team_match_counts,
        "team_weight_totals": {
            int(team): round(float(value), 6) for team, value in fitted["weight_totals"].items()
        },
        "team_reconciliation_max": {
            int(team): round(float(value), 6) for team, value in team_reconciliation_max.items()
        },
        "home_match_weight": round(home_match_weight, 6),
        "match_count": len(rows) // 2,
        "total_team_weight": round(sum(float(row["weight"]) for row in rows), 6),
        "config_hash": config.config_hash(),
        "data_gaps": [],
        "challenger_version": TEAM_ATTACK_CHALLENGER_VERSION,
        "challenger_config_hash": config.config_hash(),
        "challenger_families": sorted(family_set),
        "level_basis": level["level_basis"],
        "venue_basis": fitted["home_advantage_source"],
        "venue_split_identified": bool(level["venue_split_identified"]),
        "attack_prior_strength_used": round(float(strengths["attack"]), 6),
        "defence_prior_strength_used": round(float(strengths["defence"]), 6),
        "home_advantage_prior_strength_used": round(float(strengths["home_advantage"]), 6),
        "prior_strength_estimates": estimates,
        "level_flags": sorted(set(flags)),
    }


# ---------------------------------------------------------------------------
# Projection build — the incumbent's published row contract.
# ---------------------------------------------------------------------------


def _project_side(
    fixture_id: int,
    event: int,
    team_id: int,
    opponent_id: int,
    venue: str,
    params: Mapping[str, Any],
    resolved: Mapping[str, Any],
    cutoff: str,
    generated_at: str,
    known_teams: set[int],
    families: frozenset[str],
) -> dict[str, Any]:
    """One fixture side, carrying the incumbent's published field contract."""

    lam_for = incumbent.lambda_for(team_id, opponent_id, venue, params)
    lam_against = incumbent.lambda_for(
        opponent_id, team_id, incumbent.VENUE_AWAY if venue == incumbent.VENUE_HOME else incumbent.VENUE_HOME, params
    )
    dist_for = incumbent.goals_distribution(lam_for)
    dist_against = incumbent.goals_distribution(lam_against)

    flags: list[str] = [incumbent.NO_HISTORICAL_TEAM_PRIOR]
    matches = int(params["team_match_counts"].get(int(team_id), 0))
    if matches < int(resolved["low_evidence_matches"]):
        flags.append("LOW_CURRENT_EVIDENCE")
    weight_total = float(params["team_weight_totals"].get(int(team_id), 0.0))
    evidence_share = weight_total / (weight_total + float(resolved["attack_prior_strength"]))
    if evidence_share < float(resolved["high_regularisation_share"]):
        flags.append("HIGH_REGULARISATION_DOMINANCE")
    if team_id not in known_teams or opponent_id not in known_teams or team_id == opponent_id:
        flags.append("CONTRADICTORY_TEAM_MAPPING")
    recon = float(params["team_reconciliation_max"].get(int(team_id), 0.0))
    if recon > float(resolved["xg_xgc_tolerance"]):
        flags.append("XG_XGC_RECONCILIATION_DISCREPANCY")
    if float(params.get("home_match_weight", 0.0)) < float(resolved["home_advantage_low_evidence_matches"]):
        flags.append("HOME_ADVANTAGE_WEAK_EVIDENCE")

    flags.extend(params.get("level_flags") or [])
    if FAMILY_DATA_DERIVED_PRIOR_STRENGTH in families and matches < int(
        resolved["reliability_min_matches_per_team"]
    ):
        # This team is too thin to have contributed to the reliability estimate
        # that decided its own shrinkage, so its weight is disclosed.
        flags.append(FLAG_SPARSE_TEAM_EVIDENCE)

    return {
        "fixture_id": int(fixture_id),
        "event": int(event),
        "team_id": int(team_id),
        "opponent_id": int(opponent_id),
        "venue": venue,
        "expected_goals_for": round(lam_for, 6),
        "expected_goals_against": round(lam_against, 6),
        "p_goals_0": round(dist_for["p_goals_0"], 6),
        "p_goals_1": round(dist_for["p_goals_1"], 6),
        "p_goals_2_plus": round(dist_for["p_goals_2_plus"], 6),
        "p_clean_sheet": round(dist_against["p_goals_0"], 6),
        "attack_rating": round(float(params["attack"].get(int(team_id), 0.0)), 6),
        "opponent_defence_rating": round(float(params["defence"].get(int(opponent_id), 0.0)), 6),
        "league_baseline": params["league_log_baseline"],
        "home_advantage": params["home_advantage"],
        "prior_strength_used": params["attack_prior_strength_used"],
        "current_evidence_weight": round(weight_total, 6),
        "current_evidence_share": round(evidence_share, 6),
        "model_version": TEAM_ATTACK_CHALLENGER_VERSION,
        "generated_at": generated_at,
        "input_cutoff": cutoff,
        "risk_flags": sorted(set(flags)),
        "provenance": {
            "xg_source": "sum_of_side_player_expected_goals_by_fixture",
            "side_attribution": "player_gameweeks.was_home",
            "reconciliation_source": "max_side_player_expected_goals_conceded",
            "team_matches_used": matches,
            "reconciliation_max_abs_diff": recon,
            "reconciliation_status": (
                "DISCREPANCY" if recon > float(resolved["xg_xgc_tolerance"]) else "OK"
            ),
            "prior_source": "league_average",
            "level_basis": params["level_basis"],
            "venue_basis": params["venue_basis"],
            "prior_strength_basis": {
                "attack": (params["prior_strength_estimates"].get("attack") or {}).get("basis"),
                "defence": (params["prior_strength_estimates"].get("defence") or {}).get("basis"),
            },
            "challenger_families": sorted(families),
            "promotion": "NOT_PERFORMED_CHALLENGER_ONLY",
        },
        "data_gaps": (
            ["team-specific historical prior unavailable; league-average prior used"]
            if incumbent.NO_HISTORICAL_TEAM_PRIOR in flags
            else []
        ),
    }


def build_challenger_team_projections(
    conn: sqlite3.Connection,
    planning_event: int,
    cutoff: str,
    *,
    config: TeamAttackChallengerConfig | None = None,
    incumbent_config: "incumbent.TeamStrengthConfig | None" = None,
    families: Iterable[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """One row per (fixture, side) for the planning event, in fixture order.

    The fixture list comes from the incumbent's own event fixture map, so a blank
    team simply has no fixture and therefore no projected attack: no fixture
    attack is invented for a blank, and each fixture of a double gameweek is its
    own atomic row.
    """

    config = config or TeamAttackChallengerConfig()
    incumbent_config = incumbent_config or incumbent.TeamStrengthConfig()
    family_set = frozenset(REFINEMENT_FAMILIES if families is None else families)
    resolved = config.resolved(incumbent_config)
    params = fit_challenger_team_strength(
        conn,
        int(planning_event),
        cutoff,
        config=config,
        incumbent_config=incumbent_config,
        families=family_set,
    )
    generated_at = utc_now()
    known_teams = {int(row["id"]) for row in conn.execute("SELECT id FROM teams").fetchall()}
    fixtures_by_team = analytics.event_fixture_map(conn, int(planning_event))
    fixtures = sorted(
        {int(fx["id"]): fx for fx in (f for team in fixtures_by_team.values() for f in team)}.values(),
        key=lambda fx: int(fx["id"]),
    )
    rows: list[dict[str, Any]] = []
    for fixture in fixtures:
        fixture_id = int(fixture["id"])
        home, away = int(fixture["team_h"]), int(fixture["team_a"])
        for team_id, opponent_id, venue in (
            (home, away, incumbent.VENUE_HOME),
            (away, home, incumbent.VENUE_AWAY),
        ):
            rows.append(
                _project_side(
                    fixture_id,
                    int(fixture["event"]),
                    team_id,
                    opponent_id,
                    venue,
                    params,
                    resolved,
                    cutoff,
                    generated_at,
                    known_teams,
                    family_set,
                )
            )
    meta = {
        "model_version": TEAM_ATTACK_CHALLENGER_VERSION,
        "config_hash": config.config_hash(),
        "params": params,
        "generated_at": generated_at,
        "fixture_team_pairs": len(rows),
        "challenger_families": sorted(family_set),
        "identity": challenger_identity(config, incumbent_config),
    }
    return rows, meta


# ---------------------------------------------------------------------------
# Arms.
# ---------------------------------------------------------------------------


@dataclass
class TeamChallengerArms:
    """One incumbent arm and the challenger arms, over the same key set."""

    planning_event: int
    cutoff: str
    incumbent_rows: dict[tuple[int, int], dict[str, Any]] = field(default_factory=dict)
    arms: dict[str, dict[tuple[int, int], dict[str, Any]]] = field(default_factory=dict)
    families: dict[str, frozenset[str]] = field(default_factory=dict)
    params: dict[str, dict[str, Any]] = field(default_factory=dict)
    identity: dict[str, Any] = field(default_factory=dict)

    def keys(self) -> list[tuple[int, int]]:
        return sorted(self.incumbent_rows)

    def arm_keys(self, arm: str) -> list[tuple[int, int]]:
        return sorted(self.arms[arm])

    def arm_names(self) -> tuple[str, ...]:
        return tuple(sorted(self.arms))

    def row(self, arm: str, key: tuple[int, int]) -> dict[str, Any]:
        return self.arms[arm][key]

    def verify_same_population(self) -> None:
        """Every arm covers exactly the incumbent's key set, or PE-7 stops."""

        from . import walk_forward as wf

        incumbent_keys = self.keys()
        for arm in self.arm_names():
            wf.assert_same_population(incumbent_keys, self.arm_keys(arm))


def build_challenger_arms(
    conn: sqlite3.Connection,
    planning_event: int,
    cutoff: str,
    *,
    incumbent_config: "incumbent.TeamStrengthConfig | None" = None,
    challenger_config: TeamAttackChallengerConfig | None = None,
    arm_definitions: Mapping[str, frozenset[str]] | None = None,
) -> TeamChallengerArms:
    """Project every fixture side of the event for the incumbent and the arms.

    The incumbent projection function is called unchanged, and the challenger is
    checked against it rather than in place of it.  Where the two descriptions of
    the SHARED evidence disagree — the same cutoff, the same completed fixtures —
    this stops with :class:`ChallengerInconsistencyError` instead of scoring two
    different worlds.
    """

    incumbent_config = incumbent_config or incumbent.TeamStrengthConfig()
    challenger_config = challenger_config or TeamAttackChallengerConfig()
    definitions = dict(arm_definitions or default_arm_definitions())

    incumbent_rows, incumbent_meta = incumbent.build_team_fixture_projections(
        conn, int(planning_event), cutoff, incumbent_config
    )
    incumbent_params = incumbent_meta["params"]
    arms: dict[str, dict[tuple[int, int], dict[str, Any]]] = {}
    params: dict[str, dict[str, Any]] = {}
    for arm in sorted(definitions):
        family_set = frozenset(definitions[arm])
        rows, meta = build_challenger_team_projections(
            conn,
            int(planning_event),
            cutoff,
            config=challenger_config,
            incumbent_config=incumbent_config,
            families=family_set,
        )
        challenger_counts = meta["params"]["team_match_counts"]
        if dict(challenger_counts) != dict(incumbent_params["team_match_counts"]):
            raise ChallengerInconsistencyError(
                "the challenger and the incumbent disagree about the shared team-match evidence at "
                f"event {planning_event}: the same cutoff must describe one evidence set"
            )
        arms[arm] = {(int(row["fixture_id"]), int(row["team_id"])): row for row in rows}
        params[arm] = meta["params"]
    return TeamChallengerArms(
        planning_event=int(planning_event),
        cutoff=str(cutoff),
        incumbent_rows={(int(r["fixture_id"]), int(r["team_id"])): r for r in incumbent_rows},
        arms=arms,
        families={arm: frozenset(families) for arm, families in definitions.items()},
        params=params,
        identity=challenger_identity(challenger_config, incumbent_config),
    )
