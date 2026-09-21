"""PE-6 availability / minutes challenger — challenger-first, never promotion.

WHAT THIS IS
------------
PE-6 improves the prediction of *whether and how long a player plays*.  The
incumbent is frozen: ``minutes_v1.8.0`` / ``minutes_v1.2.0`` / ``minutes_v1.5.2``
are its identity and this module neither renames nor re-points them, nor writes
through them.  A refinement must earn a promotion at PE-6's terminal boundary;
until then it lives here, under its own identity:

    ``AVAILABILITY_MINUTES_CHALLENGER_VERSION`` + ``AvailabilityMinutesChallengerConfig.config_hash()``

WHAT IT CHANGES, AND WHAT IT DELIBERATELY DOES NOT
--------------------------------------------------
The incumbent publishes four families of availability/minutes values as
**modelling assumptions rather than calibrated probabilities**.  PE-6 does not
preserve them blindly and does not replace them blindly; each family below is
either refined by *evidence that was already observable at the cutoff*, or left
declared-unchanged with that outcome stated:

* ``official_status_availability_defaults`` -> REFINED (family 1 below)
* ``bounded_scouting_modifiers`` -> UNCHANGED (see below)
* ``bounded_return_from_injury_modifiers`` -> REFINED (family 2 below)
* ``bounded_rotation_risk_modifiers`` -> REFINED (family 3 below)

*Family 1 — official status availability defaults.*  The declared table is a
statement about ``P(play | status)``.  Where a player's own causal window
contains completed rows that were played under the same **restricting** status
(``d``, ``i``, ``u``), that observed frequency replaces the declared number,
shrunk toward it with a declared prior strength.  Hard statuses (``s``, ``n``)
and the event-specific ``chance_of_playing_this_round`` signal keep their
precedence exactly as the incumbent defines them, and a scouting availability
override is never second-guessed.  Status ``a`` is deliberately NOT refined: a
0-minute row under status ``a`` is available-but-unused, which the incumbent
already models as *role* evidence (``AVAILABLE_NON_START_ZERO_MINUTES``);
treating it as unavailability would double-count non-selection.

*Family 2 — bounded return-from-injury modifiers.*  The incumbent discounts
every fixture of a returning player (``i``/``d``) by fixed factors.  The
challenger ATTENUATES those same factors by how much the player has already
demonstrated while carrying that status: the observed minutes of his own
returning-status appearances.  With no such evidence the attenuation is exactly
zero and the incumbent's modifiers stand untouched (bit-for-bit).

*Family 3 — bounded rotation-risk modifiers.*  The challenger does not re-tune
the scout-note bands.  Rotation and role uncertainty are refined through the
player's own **observed** non-selection record: the cameo rate is estimated with
the incumbent's own recency weighting (the start-rate path is recency-weighted;
the cameo path was not), and the cameo 60+ tail is estimated from the player's
own cameo appearances instead of being taken from the pool prior alone.

*``bounded_scouting_modifiers`` — UNCHANGED.*  PE-6 reports NO CHANGE for the
note bands themselves: the only available channel is the scout note, and
re-scaling the bands from the notes they are derived from would be tuning, not
evidence.  This is a PE-6 outcome, not an omission.

HOW IT IS BUILT (and why the numbers are comparable)
----------------------------------------------------
The incumbent's projection function is called UNCHANGED, once per player-fixture
with ``include_conditionals=True``, from the same PE-1 causal boundary the
incumbent uses (``analytics.completed_rows_as_of`` / ``snapshot_as_of`` /
``snapshot_history_as_of`` / ``repo.scouting_current_rows_as_of``).  No second
causal predicate exists in this module.  The challenger then applies its
declared refinements to those published marginals and re-derives the coherent
distribution with the incumbent's own coherence equations
(:func:`coherent_chain`), so ``P(available) -> P(start | available) ->
P(cameo | not start, available) -> P(0) -> P(1-59) -> P(60+) -> P(80+) ->
expected_minutes`` stays mathematically coherent in every row.

Status evidence is restricted to rows the incumbent attributes with an
EVENT-SPECIFIC status (``availability_basis == "event_context"``).  A row whose
status was only assumed from a trail that never varied proves nothing about that
row, so it is counted and excluded rather than used.

WHERE THE NUMBERS COME FROM
---------------------------
A row on which no refinement moves a chain input REPUBLISHES the base
projection's own derived values verbatim, so an unrefined row is bit-identical to
the incumbent's.  A row that IS refined re-derives the whole chain from the
published 6-decimal marginals (:func:`coherent_chain`), so its coherence
identities hold by construction.  Either way the challenger never re-fits, re-
anchors or re-pools anything it did not declare: the only quantities it may move
are ``P(available)``, ``P(cameo | not start)``, ``P(60+ | cameo)`` and the
return-ramp factors, and every one of those movements is recorded per family
with its evidence and sample size.  The suite pins both properties.

DETERMINISM
-----------
No randomness of any kind: no RNG, no Monte Carlo, no clock-dependent branch.
The same inputs and the same cutoff produce byte-identical rows apart from
``generated_at``.  The Monte Carlo draw order is untouched, and this module is
not wired into any production decision path.
"""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass, field, fields, replace
from typing import Any, Iterable, Mapping, Sequence

from . import analytics
from . import minutes_model as incumbent
from . import repositories as repo
from .utils import utc_now

AVAILABILITY_MINUTES_CHALLENGER_VERSION = "availability_minutes_challenger_v0.1.0"
# v0.1.0: the first PE-6 challenger.  A challenger identity, not an incumbent
# bump: nothing here is promoted, and the incumbent's own version strings are
# read-only in this module.

AVAILABILITY_MINUTES_CHALLENGER_FAMILY = "minutes_availability_challenger_pe6"

#: The incumbent identities this phase may not rename, re-point or overwrite.
FROZEN_INCUMBENT_VERSIONS: dict[str, str] = {
    "minutes_model": "minutes_v1.8.0",
    "minutes_coherent_model": "minutes_v1.2.0",
    "joint_minutes_model": "minutes_v1.5.2",
}


def frozen_incumbent_identity() -> dict[str, Any]:
    """The three frozen identities as the running code actually declares them.

    Read from the modules, never restated: the point of the check is to fail
    closed if a later change re-points one of them while claiming not to.
    """

    from . import joint_minutes

    observed = {
        "minutes_model": incumbent.MINUTES_MODEL_VERSION,
        "minutes_coherent_model": incumbent.MINUTES_COHERENT_MODEL_VERSION,
        "joint_minutes_model": joint_minutes.JOINT_MINUTES_MODEL_VERSION,
    }
    changed = sorted(
        name for name, declared in FROZEN_INCUMBENT_VERSIONS.items() if observed.get(name) != declared
    )
    return {
        "declared": dict(FROZEN_INCUMBENT_VERSIONS),
        "observed": observed,
        "unchanged": not changed,
        "changed": changed,
    }


# ---------------------------------------------------------------------------
# Refinement families.
# ---------------------------------------------------------------------------

FAMILY_AVAILABILITY_STATUS_EVIDENCE = "availability_status_evidence"
FAMILY_RETURN_FROM_INJURY_RAMP = "return_from_injury_ramp"
FAMILY_CAMEO_RATE_RECENCY = "cameo_rate_recency"
FAMILY_CAMEO_TAIL_EVIDENCE = "cameo_tail_evidence"

#: The families the challenger can turn on, in report order.
REFINEMENT_FAMILIES: tuple[str, ...] = (
    FAMILY_AVAILABILITY_STATUS_EVIDENCE,
    FAMILY_RETURN_FROM_INJURY_RAMP,
    FAMILY_CAMEO_RATE_RECENCY,
    FAMILY_CAMEO_TAIL_EVIDENCE,
)

#: The contract's four assumption families -> the PE-6 treatment of each.
ASSUMPTION_FAMILY_TREATMENT: dict[str, str] = {
    "official_status_availability_defaults": "REFINED",
    "bounded_scouting_modifiers": "UNCHANGED_BY_CHALLENGER",
    "bounded_return_from_injury_modifiers": "REFINED",
    "bounded_rotation_risk_modifiers": "REFINED",
}

#: ``contract assumption family`` -> the refinement family that addresses it.
ASSUMPTION_FAMILY_REFINEMENT: dict[str, str | None] = {
    "official_status_availability_defaults": FAMILY_AVAILABILITY_STATUS_EVIDENCE,
    "bounded_scouting_modifiers": None,
    "bounded_return_from_injury_modifiers": FAMILY_RETURN_FROM_INJURY_RAMP,
    "bounded_rotation_risk_modifiers": FAMILY_CAMEO_RATE_RECENCY,
}

#: The incumbent's status-at-event attribution basis for a row whose status is
#: genuinely event-specific.  Only such rows may be refined evidence: a status
#: attributed to a row merely because the trail never varied
#: (``assumed_constant_status``) is not proof that the status applied then.
EVENT_SPECIFIC_ATTRIBUTION = "event_context"

#: Only a *restricting* soft status measures ``P(play | status)``.  Status ``a``
#: is excluded on purpose: a 0-minute row under ``a`` is available-but-unused,
#: which is role evidence, not unavailability (see the module docstring).
SOFT_RESTRICTING_STATUSES: tuple[str, ...] = ("d", "i", "u")

#: Availability-source categories, derived structurally from the incumbent's own
#: published ``availability_source_summary``, never from its prose label.
BASIS_HARD_STATUS = "HARD_UNAVAILABLE_STATUS"
BASIS_EVENT_CHANCE = "EVENT_SPECIFIC_CHANCE_SIGNAL"
BASIS_DECLARED_STATUS_DEFAULT = "DECLARED_STATUS_AVAILABILITY_DEFAULT"
BASIS_NO_RESTRICTION = "NO_OFFICIAL_RESTRICTION"
BASIS_SCOUTING_OVERRIDE = "SCOUTING_AVAILABILITY_OVERRIDE"

AVAILABILITY_BASES: tuple[str, ...] = (
    BASIS_HARD_STATUS,
    BASIS_EVENT_CHANCE,
    BASIS_DECLARED_STATUS_DEFAULT,
    BASIS_NO_RESTRICTION,
    BASIS_SCOUTING_OVERRIDE,
)

#: The incumbent's scouting flag for a high injury-uncertainty note.  Mirrored
#: here as a named constant; the suite pins it against the incumbent's own fold.
SCOUT_RETURN_RAMP_FLAG = "RETURN_RAMP_SCOUT_HIGH"

#: Per-family outcome of one row, so an audit can tell "no evidence" from
#: "switched off" from "applied".
REFINEMENT_APPLIED = "APPLIED"
REFINEMENT_NOT_APPLICABLE = "NOT_APPLICABLE"
REFINEMENT_NO_EVIDENCE = "NO_EVIDENCE_FALLBACK_TO_INCUMBENT"
REFINEMENT_DISABLED = "DISABLED_IN_THIS_ARM"

#: Evidence classes that prove the player was available around the event.  Built
#: from the incumbent's public class constants; the suite pins it against the
#: incumbent's own (private) tuple so the two cannot silently drift.
START_OBSERVATION_CLASSES: tuple[str, ...] = (
    incumbent.EVIDENCE_STARTED,
    incumbent.EVIDENCE_BENCH_APPEARANCE,
    incumbent.EVIDENCE_AVAILABLE_NON_START_ZERO_MINUTES,
    incumbent.EVIDENCE_UNUSED_SUB_AVAILABLE,
    incumbent.EVIDENCE_NOT_IN_MATCHDAY_SQUAD,
)


class ChallengerInconsistencyError(RuntimeError):
    """The challenger and the incumbent disagree about the evidence they share.

    Fail closed: a disagreement means one of them is describing a different
    world, and a silently reconciled number is exactly what PE-6 forbids.
    """


# ---------------------------------------------------------------------------
# Configuration and identity.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AvailabilityMinutesChallengerConfig:
    """Every challenger tunable in one versioned structure; no magic numbers.

    ``None`` means "inherit the incumbent's declared value", so the challenger
    introduces no second knob for a quantity the incumbent already declares.
    """

    #: Prior weight (in rows) pulling a status estimate back to the declared
    #: assumption.  Four rows of prior per row of evidence is a deliberately
    #: cautious reading of a thin per-player status history.
    status_evidence_prior_strength: float = 4.0
    #: Fewer observed rows than this and the declared assumption stands: a
    #: single observation is not a calibration.
    status_evidence_min_rows: int = 2
    #: Observed minutes at which the return ramp is fully attenuated.
    ramp_recovery_reference_minutes: float = 90.0
    #: Appearances required before the ramp is attenuated at all.
    ramp_recovery_min_rows: int = 1
    #: ``None`` -> the incumbent's ``start_recency_half_life_matches``.
    cameo_recency_half_life_matches: float | None = None
    #: ``None`` -> the incumbent's ``cameo_p60_prior_strength``.
    cameo_tail_prior_strength: float | None = None

    def config_hash(self) -> str:
        values = {item.name: getattr(self, item.name) for item in fields(self)}
        return analytics.canonical_hash({"model": AVAILABILITY_MINUTES_CHALLENGER_VERSION, **values})

    def resolved(self, incumbent_config: "incumbent.MinutesModelConfig") -> dict[str, float]:
        """The effective values, with ``None`` inheriting from the incumbent."""

        return {
            "status_evidence_prior_strength": float(self.status_evidence_prior_strength),
            "status_evidence_min_rows": float(self.status_evidence_min_rows),
            "ramp_recovery_reference_minutes": float(self.ramp_recovery_reference_minutes),
            "ramp_recovery_min_rows": float(self.ramp_recovery_min_rows),
            "cameo_recency_half_life_matches": float(
                self.cameo_recency_half_life_matches
                if self.cameo_recency_half_life_matches is not None
                else incumbent_config.start_recency_half_life_matches
            ),
            "cameo_tail_prior_strength": float(
                self.cameo_tail_prior_strength
                if self.cameo_tail_prior_strength is not None
                else incumbent_config.cameo_p60_prior_strength
            ),
        }


def challenger_identity(
    config: AvailabilityMinutesChallengerConfig | None = None,
    incumbent_config: "incumbent.MinutesModelConfig | None" = None,
) -> dict[str, Any]:
    """The challenger's own identity, carried by every artifact it produces."""

    config = config or AvailabilityMinutesChallengerConfig()
    incumbent_config = incumbent_config or incumbent.MinutesModelConfig()
    return {
        "family": AVAILABILITY_MINUTES_CHALLENGER_FAMILY,
        "challenger_version": AVAILABILITY_MINUTES_CHALLENGER_VERSION,
        "challenger_config_hash": config.config_hash(),
        "resolved_parameters": config.resolved(incumbent_config),
        "refinement_families": list(REFINEMENT_FAMILIES),
        "assumption_family_treatment": dict(ASSUMPTION_FAMILY_TREATMENT),
        "incumbent_identity": frozen_incumbent_identity(),
        "incumbent_config_hash": incumbent_config.config_hash(),
        "promotion": "NOT_PERFORMED_CHALLENGER_ONLY",
    }


# ---------------------------------------------------------------------------
# Small math mirrors.
#
# These three formulas are the incumbent's, restated here rather than imported
# from its private namespace.  The suite pins each mirror against the
# incumbent's own function on a grid, so they cannot drift silently.
# ---------------------------------------------------------------------------


def _shrink(evidence_total: float, evidence_count: float, prior: float, strength: float) -> float:
    """``(evidence + strength * prior) / (count + strength)`` (incumbent's form)."""

    return (evidence_total + strength * prior) / (evidence_count + strength)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _round(value: float, digits: int = 6) -> float:
    return round(value, digits)


def _recency_weights(count: int, half_life: float) -> list[float]:
    """``2 ** -(index / half_life)``, newest first (incumbent's form)."""

    return [
        math.pow(2.0, -(index / max(float(half_life), 1e-6))) for index in range(int(count))
    ]


def _availability_basis(source_summary: Mapping[str, Any], config: "incumbent.MinutesModelConfig") -> str:
    """Which availability rule produced the incumbent's ``p_available``.

    Read from the incumbent's STRUCTURED summary fields.  The prose ``rule``
    string is never parsed: a display label must not be load-bearing.
    """

    if source_summary.get("scouting_override"):
        return BASIS_SCOUTING_OVERRIDE
    status = source_summary.get("status")
    chance = source_summary.get("chance_of_playing_this_round")
    if status in config.hard_unavailable_statuses:
        return BASIS_HARD_STATUS
    if isinstance(chance, (int, float)) and not isinstance(chance, bool):
        return BASIS_EVENT_CHANCE
    if status in config.official_availability_signal_defaults:
        return BASIS_DECLARED_STATUS_DEFAULT
    return BASIS_NO_RESTRICTION


# ---------------------------------------------------------------------------
# The coherent chain.
# ---------------------------------------------------------------------------


DERIVED_CHAIN_FIELDS: tuple[str, ...] = (
    "p_start",
    "p_cameo",
    "p_zero",
    "p_1_59",
    "p_60_plus",
    "p_80_plus",
    "expected_minutes",
)

#: How a challenger row's derived values were obtained.
DERIVED_BY_BASE_REPUBLICATION = "BASE_PROJECTION_REPUBLISHED"
DERIVED_BY_CHAIN_REDERIVATION = "CHAIN_REDERIVED_FROM_PUBLISHED_MARGINALS"


def coherent_chain(
    *,
    p_available: float,
    p_start_given_available: float,
    p_cameo_given_not_start: float,
    p_60_given_start: float,
    p_80_given_start: float,
    p_60_given_cameo: float,
    expected_minutes_if_start: float,
    expected_minutes_if_cameo: float,
) -> dict[str, float]:
    """Re-derive the coherent distribution, with the incumbent's own equations.

    Mirrors ``minutes_model.project_player_fixture``'s derivation exactly:

        P(start)  = P(available) * P(start | available)
        P(cameo)  = P(available) * (1 - P(start | available)) * P(cameo | not start)
        P(0)      = 1 - P(start) - P(cameo)
        P(60+)    = P(start) * P(60+ | start) + P(cameo) * P(60+ | cameo)
        P(80+)    = P(start) * P(80+ | start), never above P(60+)
        E[minutes] = P(start) * E[minutes | start] + P(cameo) * E[minutes | cameo]

    The cameo tail is folded into ``P(1-59)`` exactly as the incumbent declares
    (``p80_if_cameo == 0``), and the same range guards are applied, so an
    impossible distribution raises instead of being published.
    """

    p_start = _clamp(p_available * p_start_given_available)
    not_start_adj = (1.0 - p_start_given_available) if p_available > 0 else 1.0
    p_cameo = _clamp(p_available * not_start_adj * p_cameo_given_not_start)
    p_zero = _clamp(1.0 - p_start - p_cameo)
    p_60_plus = _clamp(p_start * p_60_given_start + p_cameo * p_60_given_cameo)
    p_80_plus = min(_clamp(p_start * p_80_given_start), p_60_plus)
    p_1_59 = _clamp(1.0 - p_zero - p_60_plus)
    expected_minutes = p_start * expected_minutes_if_start + p_cameo * expected_minutes_if_cameo

    for name, value in (
        ("p_available", p_available),
        ("p_start_given_available", p_start_given_available),
        ("p_cameo_given_not_start", p_cameo_given_not_start),
        ("p_60_given_start", p_60_given_start),
        ("p_80_given_start", p_80_given_start),
        ("p_60_given_cameo", p_60_given_cameo),
        ("p_start", p_start),
        ("p_cameo", p_cameo),
        ("p_zero", p_zero),
        ("p_1_59", p_1_59),
        ("p_60_plus", p_60_plus),
        ("p_80_plus", p_80_plus),
    ):
        if not 0.0 - 1e-9 <= value <= 1.0 + 1e-9:
            raise ValueError(f"challenger probability out of range: {name}={value}")
    if not 0.0 <= expected_minutes <= 90.0 + 1e-9:
        raise ValueError(f"challenger expected minutes out of range: {expected_minutes}")
    if abs((p_zero + p_1_59 + p_60_plus) - 1.0) > 1e-6:
        # Same repair the incumbent makes: the three body states must sum to one.
        p_zero = _clamp(1.0 - p_1_59 - p_60_plus)

    return {
        "p_start": _round(p_start),
        "p_cameo": _round(p_cameo),
        "p_zero": _round(p_zero),
        "p_1_59": _round(p_1_59),
        "p_60_plus": _round(p_60_plus),
        "p_80_plus": _round(p_80_plus),
        "expected_minutes": _round(expected_minutes),
    }


# ---------------------------------------------------------------------------
# Evidence.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChallengerEvidence:
    """Every causal quantity the refinements are allowed to read.

    Each block carries its own sample size, so a thin sample is visible as a
    thin sample rather than as a confident number.
    """

    availability_basis: str
    status_at_cutoff: Any
    chance_at_cutoff: Any
    incumbent_p_available: float
    row_attribution_bases: Mapping[str, int]
    event_specific_rows: int
    assumed_attribution_rows_excluded: int
    status_declared_default: float | None
    status_evidence_observations: int
    status_evidence_appearances: int
    status_estimate: float | None
    ramp_applies: bool
    ramp_attenuation: float
    ramp_recovery_observations: int
    ramp_recovery_mean_minutes: float | None
    ramp_effective_modifiers: Mapping[str, float]
    cameo_rate_pool_prior: float
    cameo_rate_observations: int
    cameo_rate_recent_observations: int
    cameo_rate_estimate: float | None
    cameo_tail_prior: float
    cameo_tail_observations: int
    cameo_tail_estimate: float | None
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        payload = {item.name: getattr(self, item.name) for item in fields(self)}
        payload["ramp_effective_modifiers"] = dict(self.ramp_effective_modifiers)
        payload["row_attribution_bases"] = dict(self.row_attribution_bases)
        payload["notes"] = list(self.notes)
        return payload


def build_player_evidence(
    evidence_rows: Sequence[Mapping[str, Any]],
    snapshot: Mapping[str, Any] | None,
    snapshot_history: Sequence[Mapping[str, Any]] | None,
    scout_notes: Sequence[Mapping[str, Any]],
    *,
    config: "incumbent.MinutesModelConfig",
    challenger_config: AvailabilityMinutesChallengerConfig,
    pools: "incumbent.LeaguePools",
    position: int,
    cutoff: str,
    incumbent_p_available: float,
    incumbent_source_summary: Mapping[str, Any],
) -> ChallengerEvidence:
    """Assemble the causal evidence for one player at one cutoff.

    Uses the incumbent's own row classification (``classify_evidence_rows``) and
    its own status-at-event attribution (``availability_status_at_event``, via
    that classification), so no second history predicate exists anywhere in
    PE-6.  The availability rule is read from the incumbent's published
    structured summary and cross-checked against the snapshot the incumbent read;
    a disagreement stops the build rather than being reconciled silently.
    """

    classification = incumbent.classify_evidence_rows(
        evidence_rows, snapshot_history=snapshot_history, config=config
    )
    # Only rows whose status is event-specific may be refined evidence.
    classified = [
        row for row in classification if row.get("availability_basis") == EVENT_SPECIFIC_ATTRIBUTION
    ]
    attribution_bases: dict[str, int] = {}
    for row in classification:
        basis_name = str(row.get("availability_basis"))
        attribution_bases[basis_name] = attribution_bases.get(basis_name, 0) + 1
    assumed_excluded = len(classification) - len(classified)
    status = (snapshot or {}).get("status")
    chance = (snapshot or {}).get("chance_of_playing_this_round")
    basis = _availability_basis(incumbent_source_summary, config)
    snapshot_basis = _availability_basis(
        {"status": status, "chance_of_playing_this_round": chance}, config
    )
    if basis != snapshot_basis and basis != BASIS_SCOUTING_OVERRIDE:
        raise ChallengerInconsistencyError(
            f"availability basis disagrees with the incumbent's own summary: "
            f"{basis!r} (published) vs {snapshot_basis!r} (snapshot at the cutoff)"
        )
    notes: list[str] = []

    # --- family 1: P(play | restricting status) from the player's own rows ----
    declared_default: float | None = None
    status_observations = 0
    status_appearances = 0
    status_estimate: float | None = None
    if basis != BASIS_DECLARED_STATUS_DEFAULT:
        notes.append(f"STATUS_ESTIMATE_NOT_APPLICABLE:{basis}")
    elif status not in SOFT_RESTRICTING_STATUSES:
        notes.append("STATUS_ESTIMATE_NOT_APPLICABLE:STATUS_IS_NOT_A_RESTRICTION")
    else:
        declared_default = float(config.official_availability_signal_defaults[str(status)])
        rows_for_status = [
            row for row in classified if row.get("availability_status_at_event") == status
        ]
        status_observations = len(rows_for_status)
        status_appearances = sum(
            1 for row in rows_for_status if float(row.get("minutes") or 0) > 0
        )
        if status_observations < challenger_config.status_evidence_min_rows:
            if status_observations == 0:
                notes.append(
                    "STATUS_ESTIMATE_NO_EVENT_SPECIFIC_ATTRIBUTION"
                    if assumed_excluded
                    else "STATUS_ESTIMATE_NO_ROWS_FOR_STATUS"
                )
            else:
                notes.append("STATUS_ESTIMATE_BELOW_MINIMUM_ROWS")
        else:
            status_estimate = _shrink(
                float(status_appearances),
                float(status_observations),
                declared_default,
                float(challenger_config.status_evidence_prior_strength),
            )

    # --- family 2: attenuate the return ramp by demonstrated recovery ---------
    returning_statuses = tuple(str(value) for value in config.returning_status_values)
    _adjustments, _modifier_ids, scout_flags = incumbent.fold_scout_modifiers(scout_notes, cutoff, config)
    ramp_applies = (snapshot is not None and status in returning_statuses) or (
        SCOUT_RETURN_RAMP_FLAG in scout_flags
    )
    ramp_rows = [
        row for row in classified if row.get("availability_status_at_event") in returning_statuses
    ]
    ramp_appearances = [
        float(row["minutes"]) for row in ramp_rows if float(row.get("minutes") or 0) > 0
    ]
    ramp_mean_minutes = (
        (sum(ramp_appearances) / len(ramp_appearances)) if ramp_appearances else None
    )
    ramp_attenuation = 0.0
    if ramp_applies and len(ramp_appearances) >= int(challenger_config.ramp_recovery_min_rows):
        if ramp_mean_minutes is not None:
            ramp_attenuation = _clamp(
                ramp_mean_minutes / float(challenger_config.ramp_recovery_reference_minutes)
            )
    effective_modifiers = {
        "returning_start_modifier": _attenuated(config.returning_start_modifier, ramp_attenuation),
        "returning_upper_minutes_modifier": _attenuated(
            config.returning_upper_minutes_modifier, ramp_attenuation
        ),
        "returning_minutes_if_start_modifier": _attenuated(
            config.returning_minutes_if_start_modifier, ramp_attenuation
        ),
    }

    # --- families 3 and 4: observed rotation / cameo evidence -----------------
    # These need only the row's own observed ``starts`` / ``minutes`` facts, not
    # its attributed status, so every classified row is usable here.
    observation_rows = [
        row for row in classification if row["evidence_class"] in START_OBSERVATION_CLASSES
    ]
    weights = _recency_weights(
        len(observation_rows), float(challenger_config.resolved(config)["cameo_recency_half_life_matches"])
    )
    not_start_rows = [
        weight for weight, row in zip(weights, observation_rows) if not row.get("starts")
    ]
    cameo_rows = [
        (weight, row)
        for weight, row in zip(weights, observation_rows)
        if not row.get("starts") and float(row.get("minutes") or 0) > 0
    ]
    not_start_weight = sum(not_start_rows)
    cameo_weight = sum(weight for weight, _row in cameo_rows)
    pool = pools.position.get(int(position)) or incumbent.structural_no_history_priors(config)
    pool_cameo_rate = float(pool.get("cameo_rate") or 0.5)
    cameo_rate_estimate = (
        _shrink(
            cameo_weight,
            not_start_weight,
            pool_cameo_rate,
            float(config.cameo_rate_prior_strength),
        )
        if not_start_weight > 0
        else None
    )
    cameo_tail_prior = max(float(pool.get("cameo_p60") or 0.0), float(config.cameo_p60_prior))
    cameo_tail_60_weight = sum(
        weight for weight, row in cameo_rows if float(row.get("minutes") or 0) >= 60
    )
    cameo_tail_estimate = _shrink(
        cameo_tail_60_weight,
        cameo_weight,
        cameo_tail_prior,
        float(
            challenger_config.cameo_tail_prior_strength
            if challenger_config.cameo_tail_prior_strength is not None
            else config.cameo_p60_prior_strength
        ),
    )

    return ChallengerEvidence(
        availability_basis=basis,
        status_at_cutoff=status,
        chance_at_cutoff=chance,
        incumbent_p_available=float(incumbent_p_available),
        row_attribution_bases=attribution_bases,
        event_specific_rows=len(classified),
        assumed_attribution_rows_excluded=assumed_excluded,
        status_declared_default=declared_default,
        status_evidence_observations=status_observations,
        status_evidence_appearances=status_appearances,
        status_estimate=(None if status_estimate is None else _round(status_estimate)),
        ramp_applies=bool(ramp_applies),
        ramp_attenuation=_round(ramp_attenuation),
        ramp_recovery_observations=len(ramp_appearances),
        ramp_recovery_mean_minutes=(
            None if ramp_mean_minutes is None else _round(ramp_mean_minutes)
        ),
        ramp_effective_modifiers={key: _round(value) for key, value in effective_modifiers.items()},
        cameo_rate_pool_prior=_round(pool_cameo_rate),
        cameo_rate_observations=len(not_start_rows),
        cameo_rate_recent_observations=len(cameo_rows),
        cameo_rate_estimate=(None if cameo_rate_estimate is None else _round(cameo_rate_estimate)),
        cameo_tail_prior=_round(cameo_tail_prior),
        cameo_tail_observations=len(cameo_rows),
        cameo_tail_estimate=_round(cameo_tail_estimate),
        notes=tuple(notes),
    )


def _attenuated(modifier: float, attenuation: float) -> float:
    """Move a declared discount toward 1.0 by ``attenuation`` in ``[0, 1]``."""

    return float(modifier) + (1.0 - float(modifier)) * _clamp(attenuation)


# ---------------------------------------------------------------------------
# Row refinement.
# ---------------------------------------------------------------------------


def refine_row(
    base_row: Mapping[str, Any],
    evidence: ChallengerEvidence,
    *,
    config: "incumbent.MinutesModelConfig",
    challenger_config: AvailabilityMinutesChallengerConfig,
    enabled_families: Iterable[str],
    position_id: int = 0,
) -> dict[str, Any]:
    """Apply the enabled refinements to one incumbent row and re-derive it.

    ``base_row`` must be the incumbent's row (optionally produced under the
    challenger's attenuated return-ramp configuration) with
    ``include_conditionals=True``.  Every family block records whether the
    refinement actually MOVED the published value (``changed``), so a no-op is
    never reported as an improvement.
    """

    enabled = frozenset(enabled_families)
    unknown = sorted(enabled - set(REFINEMENT_FAMILIES))
    if unknown:
        raise ValueError(f"unknown refinement family(ies): {unknown}")
    missing = [
        name
        for name in (
            "p_available",
            "p_start_given_available",
            "p_cameo_given_not_start",
            "p_60_given_start",
            "p_80_given_start",
            "p_60_given_cameo",
            "expected_minutes_if_start",
            "expected_minutes_if_cameo",
        )
        if name not in base_row
    ]
    if missing:
        raise ChallengerInconsistencyError(
            "the base row lacks the conditional quantities the coherent chain needs "
            f"({', '.join(missing)}); project it with include_conditionals=True"
        )

    p_available = float(base_row["p_available"])
    cameo_rate = float(base_row["p_cameo_given_not_start"])
    p_60_given_cameo = float(base_row["p_60_given_cameo"])
    refinements: dict[str, Any] = {}

    if FAMILY_AVAILABILITY_STATUS_EVIDENCE not in enabled:
        refinements[FAMILY_AVAILABILITY_STATUS_EVIDENCE] = {"status": REFINEMENT_DISABLED}
    elif evidence.status_estimate is None:
        refinements[FAMILY_AVAILABILITY_STATUS_EVIDENCE] = {
            "status": REFINEMENT_NOT_APPLICABLE
            if evidence.status_declared_default is None
            else REFINEMENT_NO_EVIDENCE,
            "basis": evidence.availability_basis,
            "observations": evidence.status_evidence_observations,
            "required_rows": int(challenger_config.status_evidence_min_rows),
            "declared_default": evidence.status_declared_default,
            "incumbent_p_available": evidence.incumbent_p_available,
            "assumed_attribution_rows_excluded": evidence.assumed_attribution_rows_excluded,
            "changed": False,
        }
    else:
        p_available = float(evidence.status_estimate)
        refinements[FAMILY_AVAILABILITY_STATUS_EVIDENCE] = {
            "status": REFINEMENT_APPLIED,
            "basis": evidence.availability_basis,
            "observations": evidence.status_evidence_observations,
            "appearances": evidence.status_evidence_appearances,
            "declared_default": evidence.status_declared_default,
            "estimate": evidence.status_estimate,
            "incumbent_p_available": evidence.incumbent_p_available,
            "prior_strength": float(challenger_config.status_evidence_prior_strength),
            "assumed_attribution_rows_excluded": evidence.assumed_attribution_rows_excluded,
            "changed": abs(p_available - float(base_row["p_available"])) > 1e-9,
        }

    if FAMILY_RETURN_FROM_INJURY_RAMP not in enabled:
        refinements[FAMILY_RETURN_FROM_INJURY_RAMP] = {"status": REFINEMENT_DISABLED}
    elif not evidence.ramp_applies:
        refinements[FAMILY_RETURN_FROM_INJURY_RAMP] = {
            "status": REFINEMENT_NOT_APPLICABLE,
            "reason": "no return-from-injury ramp applied to this row",
            "changed": False,
        }
    elif evidence.ramp_attenuation <= 0.0:
        refinements[FAMILY_RETURN_FROM_INJURY_RAMP] = {
            "status": REFINEMENT_NO_EVIDENCE,
            "recovery_observations": evidence.ramp_recovery_observations,
            "required_appearances": int(challenger_config.ramp_recovery_min_rows),
            "assumed_attribution_rows_excluded": evidence.assumed_attribution_rows_excluded,
            "changed": False,
        }
    else:
        refinements[FAMILY_RETURN_FROM_INJURY_RAMP] = {
            "status": REFINEMENT_APPLIED,
            "attenuation": evidence.ramp_attenuation,
            "recovery_observations": evidence.ramp_recovery_observations,
            "recovery_mean_minutes": evidence.ramp_recovery_mean_minutes,
            "reference_minutes": float(challenger_config.ramp_recovery_reference_minutes),
            "effective_modifiers": dict(evidence.ramp_effective_modifiers),
            "applied_in_projection_config": True,
            "assumed_attribution_rows_excluded": evidence.assumed_attribution_rows_excluded,
            "changed": True,
        }

    if FAMILY_CAMEO_RATE_RECENCY not in enabled:
        refinements[FAMILY_CAMEO_RATE_RECENCY] = {"status": REFINEMENT_DISABLED}
    elif evidence.cameo_rate_estimate is None:
        refinements[FAMILY_CAMEO_RATE_RECENCY] = {
            "status": REFINEMENT_NO_EVIDENCE,
            "observations": evidence.cameo_rate_observations,
            "incumbent_p_cameo_given_not_start": float(base_row["p_cameo_given_not_start"]),
            "changed": False,
        }
    else:
        cameo_rate = float(evidence.cameo_rate_estimate)
        refinements[FAMILY_CAMEO_RATE_RECENCY] = {
            "status": REFINEMENT_APPLIED,
            "observations": evidence.cameo_rate_observations,
            "cameo_observations": evidence.cameo_rate_recent_observations,
            "pool_prior": evidence.cameo_rate_pool_prior,
            "estimate": evidence.cameo_rate_estimate,
            "incumbent_p_cameo_given_not_start": float(base_row["p_cameo_given_not_start"]),
            "half_life_matches": float(
                challenger_config.resolved(config)["cameo_recency_half_life_matches"]
            ),
            "changed": abs(cameo_rate - float(base_row["p_cameo_given_not_start"])) > 1e-9,
        }

    if FAMILY_CAMEO_TAIL_EVIDENCE not in enabled:
        refinements[FAMILY_CAMEO_TAIL_EVIDENCE] = {"status": REFINEMENT_DISABLED}
    elif evidence.cameo_tail_observations == 0:
        # With no cameo appearance there is no observation to shrink, and the
        # estimate is exactly the incumbent's prior: report the fallback, not a
        # refinement that never happened.
        refinements[FAMILY_CAMEO_TAIL_EVIDENCE] = {
            "status": REFINEMENT_NO_EVIDENCE,
            "prior": evidence.cameo_tail_prior,
            "incumbent_p_60_given_cameo": float(base_row["p_60_given_cameo"]),
            "changed": False,
        }
    else:
        p_60_given_cameo = float(evidence.cameo_tail_estimate)
        refinements[FAMILY_CAMEO_TAIL_EVIDENCE] = {
            "status": REFINEMENT_APPLIED,
            "observations": evidence.cameo_tail_observations,
            "prior": evidence.cameo_tail_prior,
            "estimate": evidence.cameo_tail_estimate,
            "incumbent_p_60_given_cameo": float(base_row["p_60_given_cameo"]),
            "changed": abs(p_60_given_cameo - float(base_row["p_60_given_cameo"])) > 1e-9,
        }

    chain_inputs_changed = (
        abs(p_available - float(base_row["p_available"])) > 1e-9
        or abs(cameo_rate - float(base_row["p_cameo_given_not_start"])) > 1e-9
        or abs(p_60_given_cameo - float(base_row["p_60_given_cameo"])) > 1e-9
    )
    if chain_inputs_changed:
        chain = coherent_chain(
            p_available=p_available,
            p_start_given_available=float(base_row["p_start_given_available"]),
            p_cameo_given_not_start=cameo_rate,
            p_60_given_start=float(base_row["p_60_given_start"]),
            p_80_given_start=float(base_row["p_80_given_start"]),
            p_60_given_cameo=p_60_given_cameo,
            expected_minutes_if_start=float(base_row["expected_minutes_if_start"]),
            expected_minutes_if_cameo=float(base_row["expected_minutes_if_cameo"]),
        )
        derived_from = DERIVED_BY_CHAIN_REDERIVATION
    else:
        # No chain input moved: republish the base projection's OWN derived
        # values.  Re-deriving them from its published 6-decimal marginals would
        # shift them by the storage rounding (up to ~1e-5 minutes) and turn a
        # provable no-op into an approximation.
        chain = {name: float(base_row[name]) for name in DERIVED_CHAIN_FIELDS}
        derived_from = DERIVED_BY_BASE_REPUBLICATION

    row: dict[str, Any] = {
        "player_id": int(base_row["player_id"]),
        "fixture_id": int(base_row["fixture_id"]),
        "event": int(base_row["event"]),
        "model_version": AVAILABILITY_MINUTES_CHALLENGER_VERSION,
        "model_family": AVAILABILITY_MINUTES_CHALLENGER_FAMILY,
        "challenger_config_hash": challenger_config.config_hash(),
        "p_available": _round(p_available),
        "p_start_given_available": _round(float(base_row["p_start_given_available"])),
        "p_cameo_given_not_start": _round(cameo_rate),
        "p_60_given_start": _round(float(base_row["p_60_given_start"])),
        "p_80_given_start": _round(float(base_row["p_80_given_start"])),
        "p_60_given_cameo": _round(p_60_given_cameo),
        "expected_minutes_if_start": _round(float(base_row["expected_minutes_if_start"])),
        "expected_minutes_if_cameo": _round(float(base_row["expected_minutes_if_cameo"])),
        **chain,
        "derived_chain_source": derived_from,
        "enabled_families": sorted(enabled),
        "refinements": refinements,
        "evidence": evidence.as_dict(),
        "risk_flags": list(base_row.get("risk_flags") or []),
        "availability_basis": evidence.availability_basis,
        "status_at_cutoff": evidence.status_at_cutoff,
        "position_id": int(position_id),
        # Provenance: which incumbent artifact and which projection config this
        # row's marginals came from.  Never a claim that the incumbent's version
        # identifier is the challenger's.
        "base_row_model_version": str(base_row.get("model_version")),
        "base_row_availability_source_summary": dict(
            base_row.get("availability_source_summary") or {}
        ),
        "base_row": dict(base_row),
        "generated_at": utc_now(),
    }
    return row


# ---------------------------------------------------------------------------
# Arms.
# ---------------------------------------------------------------------------

ARM_ALL_REFINEMENTS = "challenger_all_refinements"


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


@dataclass
class ChallengerArms:
    """One incumbent arm and the challenger arms, over the same key set."""

    planning_event: int
    cutoff: str
    incumbent_rows: dict[tuple[int, int], dict[str, Any]] = field(default_factory=dict)
    arms: dict[str, dict[tuple[int, int], dict[str, Any]]] = field(default_factory=dict)
    evidence: dict[tuple[int, int], ChallengerEvidence] = field(default_factory=dict)
    identity: dict[str, Any] = field(default_factory=dict)

    def keys(self) -> list[tuple[int, int]]:
        return sorted(self.incumbent_rows)

    def arm_keys(self, arm: str) -> list[tuple[int, int]]:
        return sorted(self.arms[arm])

    def arm_names(self) -> tuple[str, ...]:
        return tuple(sorted(self.arms))

    def verify_same_population(self) -> None:
        """Every arm covers exactly the incumbent's key set, or PE-6 stops."""

        from . import walk_forward as wf

        incumbent_keys = self.keys()
        for arm in self.arm_names():
            wf.assert_same_population(incumbent_keys, self.arm_keys(arm))


def build_challenger_arms(
    conn: sqlite3.Connection,
    planning_event: int,
    cutoff: str,
    *,
    incumbent_config: "incumbent.MinutesModelConfig | None" = None,
    challenger_config: AvailabilityMinutesChallengerConfig | None = None,
    arm_definitions: Mapping[str, frozenset[str]] | None = None,
) -> ChallengerArms:
    """Project every player-fixture of the event for the incumbent and the arms.

    The incumbent projection function is called unchanged.  The only deviation
    is the ``include_conditionals=True`` flag (the coherent layer needs the same
    conditionals) and, where the return-ramp refinement is in play, a per-player
    ``MinutesModelConfig`` carrying the attenuated ramp factors -- both recorded
    in the row's provenance block.
    """

    config = incumbent_config or incumbent.MinutesModelConfig()
    challenger_config = challenger_config or AvailabilityMinutesChallengerConfig()
    definitions = dict(arm_definitions or default_arm_definitions())
    requested_families = frozenset().union(*definitions.values()) if definitions else frozenset()
    ramp_needed = FAMILY_RETURN_FROM_INJURY_RAMP in requested_families

    pools = incumbent.league_pools(conn, int(planning_event), cutoff)
    fixtures_by_team = analytics.event_fixture_map(conn, int(planning_event))
    result = ChallengerArms(
        planning_event=int(planning_event),
        cutoff=cutoff,
        identity=challenger_identity(challenger_config, config),
    )
    for arm in definitions:
        result.arms[arm] = {}

    for player in sorted(analytics.projectable_players(conn), key=lambda row: int(row["player_id"])):
        team_fixtures = fixtures_by_team.get(int(player["team_id"]))
        if not team_fixtures:
            continue
        player_id = int(player["player_id"])
        evidence_rows = analytics.completed_rows_as_of(conn, player_id, cutoff, int(planning_event))
        snapshot = analytics.snapshot_as_of(conn, player_id, cutoff)
        snapshot_history = analytics.snapshot_history_as_of(conn, player_id, cutoff)
        scout_notes = repo.scouting_current_rows_as_of(conn, cutoff, [player_id])
        position = int(player.get("element_type") or 0)
        for fixture in team_fixtures:
            base_row = incumbent.project_player_fixture(
                conn,
                pools,
                player,
                evidence_rows,
                fixture,
                snapshot,
                scout_notes,
                config,
                cutoff,
                include_conditionals=True,
                snapshot_history=snapshot_history,
            )
            key = (player_id, int(fixture["id"]))
            evidence = build_player_evidence(
                evidence_rows,
                snapshot,
                snapshot_history,
                scout_notes,
                config=config,
                challenger_config=challenger_config,
                pools=pools,
                position=position,
                cutoff=cutoff,
                incumbent_p_available=float(base_row["p_available"]),
                incumbent_source_summary=base_row.get("availability_source_summary") or {},
            )
            # The challenger must never disagree with the incumbent about whether
            # a return ramp is in play: the flag is the incumbent's own record.
            incumbent_ramp = "RETURN_RAMP" in set(base_row.get("risk_flags") or [])
            if evidence.ramp_applies != incumbent_ramp:
                raise ChallengerInconsistencyError(
                    f"player {player_id} fixture {key[1]}: challenger ramp state "
                    f"{evidence.ramp_applies} != incumbent RETURN_RAMP {incumbent_ramp}"
                )
            result.incumbent_rows[key] = base_row
            result.evidence[key] = evidence

            attenuated_row = None
            if ramp_needed and evidence.ramp_applies and evidence.ramp_attenuation > 0.0:
                attenuated_config = replace(
                    config,
                    returning_start_modifier=evidence.ramp_effective_modifiers["returning_start_modifier"],
                    returning_upper_minutes_modifier=evidence.ramp_effective_modifiers[
                        "returning_upper_minutes_modifier"
                    ],
                    returning_minutes_if_start_modifier=evidence.ramp_effective_modifiers[
                        "returning_minutes_if_start_modifier"
                    ],
                )
                attenuated_row = incumbent.project_player_fixture(
                    conn,
                    pools,
                    player,
                    evidence_rows,
                    fixture,
                    snapshot,
                    scout_notes,
                    attenuated_config,
                    cutoff,
                    include_conditionals=True,
                    snapshot_history=snapshot_history,
                )

            for arm, families in definitions.items():
                base = (
                    attenuated_row
                    if (attenuated_row is not None and FAMILY_RETURN_FROM_INJURY_RAMP in families)
                    else base_row
                )
                result.arms[arm][key] = refine_row(
                    base,
                    evidence,
                    config=config,
                    challenger_config=challenger_config,
                    enabled_families=families,
                    position_id=position,
                )

    result.verify_same_population()
    return result


def build_challenger_rows(
    conn: sqlite3.Connection,
    planning_event: int,
    cutoff: str,
    *,
    incumbent_config: "incumbent.MinutesModelConfig | None" = None,
    challenger_config: AvailabilityMinutesChallengerConfig | None = None,
    enabled_families: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """The challenger's rows for one event, sorted ``(player, fixture)``."""

    families = (
        frozenset(REFINEMENT_FAMILIES)
        if enabled_families is None
        else frozenset(enabled_families)
    )
    arms = build_challenger_arms(
        conn,
        planning_event,
        cutoff,
        incumbent_config=incumbent_config,
        challenger_config=challenger_config,
        arm_definitions={ARM_ALL_REFINEMENTS: families},
    )
    return [arms.arms[ARM_ALL_REFINEMENTS][key] for key in arms.keys()]
