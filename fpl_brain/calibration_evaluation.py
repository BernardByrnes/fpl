"""PE-8 calibration evaluation: do the engine's persisted numbers mean what they say?

WHAT THIS IS
------------
PE-8 is a calibration phase, not a model phase.  It answers one question: does a
stated probability match the realised frequency of the event it names, and is a
stated expectation unbiased -- and if not, can a causal, versioned, validated
transform be fitted that makes them mean what they say?

This module MEASURES.  It does not re-point an incumbent, does not bump a model
version, does not replace the DefCon calibration, and does not set a production
decision path.  ``NO CHANGE`` is a legitimate outcome and is recorded as one.

WHAT IT READS
-------------
Persisted runs and persisted outcomes only, through the frozen PE-1 / PE-2
contracts:

* the four declared probability surfaces, on the anchor ``xpts_v1`` population
  read from :func:`walk_forward_scoreboard.player_fixture_population` (so this
  module and the scoreboard cannot drift onto two different populations);
* each origin's FIT BASIS from the FROZEN prediction rows of every strictly
  earlier APPLICABLE certified target event -- a population decided by the
  certification, never by today's ``events`` row -- plus PE-5's append-only
  point-in-time observation captures, never from the current ``fixtures`` /
  ``player_gameweeks`` / ``players`` state;
* the persisted expected-points components at the grain each is persisted at;
* the Monte Carlo quantile grid, through the scoreboard's own coverage block;
* the uncalibrated FPL assist-mapping constant, as a DECISION it reports rather
  than a value it may change.

DIAGNOSE, THEN FIT
------------------
Every surface is measured and diagnosed first.  A challenger is constructed ONLY
where the diagnosis is MISCALIBRATED: a transform exists to correct a measured
defect, and fitting one speculatively -- then reporting its parameters "for
description" -- would put a candidate calibration in front of a reviewer with
nothing for it to correct.

NO IN-SAMPLE CALIBRATION
------------------------
A transform scored at target event ``E`` is fitted only on the FROZEN prediction
rows of strictly earlier events whose outcome was officially final AND captured
STRICTLY BEFORE ``E``'s own certified cutoff, as proven by PE-5's append-only
observation captures.  Membership comes from the persisted prediction, not from
the current tables: the projection rows are immutable at storage level and the
realised outcome is derived against the position PERSISTED with the prediction,
so a fixture that has since lost its played flag, a realised row that has since
been removed or blanked, a placeholder that has since appeared or a position that
has since been re-listed cannot add to or remove from a historical basis.  The
basis uses the outcome value THAT capture states, so a later official correction
is a later observation and cannot rewrite an earlier transform.  A row whose
timing cannot be proven -- no capture, a capture that arrived after the cutoff, a
provisional observation, a finality that is unstated or too late -- is EXCLUDED
AND COUNTED, never assumed known.  Every origin's fitted parameters and fit basis
are recorded, the basis digest is part of the fitted version string, and a basis
containing an outcome at or after its origin is REFUSED rather than approximated.
A figure produced by fitting and scoring on the same event set is not evidence
and is never reported as if it were.

SCORED ELIGIBILITY AND BASIS MEMBERSHIP ARE TWO QUESTIONS
---------------------------------------------------------
They are answered from different evidence and the two must not be conflated.  A
SCORED figure -- and the ORIGIN a transform is applied at, because a transform is
only ever applied where a figure is scored -- covers PE-2's declared
CURRENT-STATE population: the FINAL target events, read through
:func:`planning.event_data_state`.  A target event that is not officially FINAL
today leaves that population and is reported with its state.  A fit basis is a
different question: its CANDIDATES are the frozen prediction rows of every
APPLICABLE certified anchor target event, where "applicable" is decided by the
CERTIFICATION -- the anchor bundle's own ``xpts_v1`` run for that event -- and
never by the mutable current ``events`` row.  Each origin then admits the
strictly earlier candidates solely through the frozen prediction and the PE-5
capture / finality-at-cutoff rules.  A current ``events.finished`` edit can
therefore move a scored figure and the origin set; it can never add a row to or
remove a row from a later origin's basis, because no basis ever read it.

SAME POPULATION OR NO COMPARISON
--------------------------------
Incumbent and challenger are compared only on an identical key set, through
PE-2's own :func:`walk_forward.assert_same_population`.  An item that cannot
cover the comparison population is reported as UNREACHABLE carrying its reason,
never scored on the subset it happens to cover.  Every figure states its grain,
because a fixture-grain probability figure and an event-grain points figure
cover different populations in a double gameweek.

MISSING IS COUNTED, NEVER ZERO
------------------------------
A row whose probability is absent is excluded and counted; a scored probability
outside ``[0, 1]`` fails closed; a bin under the declared sample floor is
reported as insufficient; an empty count is rendered as null, never ``0.0``.  A
position outside a component's declared positions earns a REAL zero from it, so
that row is coverage rather than a gap, and the rule is stated by token.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from . import calibration as mc_calibration
from . import defcon_calibration as defcon_cal
from . import joint_minutes
from . import minutes_model
from . import monte_carlo
from . import outcome_ledger as ledger
from . import player_rates
from . import probability_calibration as pc
from . import team_model
from . import walk_forward as wf
from . import walk_forward_metrics as wm
from . import walk_forward_scoreboard as sb
from . import xpts as xpts_module
from .scoring_rules import DEFAULT_SCORING_RULES

PE8_SCHEMA_VERSION = "pe8_calibration_v1.0.0"
PE8_EVALUATION_VERSION = "pe8_calibration_eval_v1.0.0"

#: PE-8's terminal boundary.  ``OPEN`` is the correct state when the diagnosis is
#: supported but the evidence does not support a promotion, or when the sample
#: cannot support more than a descriptive report.
TERMINAL_READY_FOR_MERGE = "READY_FOR_MERGE"
TERMINAL_OPEN = "OPEN"

STATUS_OK = "OK"
STATUS_UNREACHABLE = "UNREACHABLE"
STATUS_POPULATION_MISMATCH = "POPULATION_MISMATCH"
#: No challenger was constructed at all, because nothing was diagnosed to correct.
STATUS_NOT_FITTED = "NOT_FITTED"

DIAGNOSIS_MISCALIBRATED = "MISCALIBRATED"
DIAGNOSIS_NO_MATERIAL_DEFECT = "NO_MATERIAL_DEFECT_DETECTED"
DIAGNOSIS_INSUFFICIENT = "INSUFFICIENT_FOR_DIAGNOSIS"

#: Why a row of the frozen prediction population did NOT enter an origin's fit
#: basis.  A basis row must be a persisted prediction row carrying the field the
#: surface is defined on, and must carry a PE-5 point-in-time observation whose
#: official finality AND capture both fall strictly before the origin's certified
#: cutoff; anything whose timing cannot be proven is excluded and counted, never
#: assumed known.
BASIS_CAPTURE_ABSENT = "POINT_IN_TIME_CAPTURE_ABSENT"
BASIS_CAPTURE_NOT_BEFORE_CUTOFF = "CAPTURE_NOT_BEFORE_CUTOFF"
BASIS_PROVISIONAL = "OBSERVATION_PROVISIONAL_AT_CUTOFF"
BASIS_FINALITY_UNPROVABLE = "OFFICIAL_FINALITY_UNPROVABLE"
BASIS_FINALITY_NOT_BEFORE_CUTOFF = "OFFICIAL_FINALITY_NOT_BEFORE_CUTOFF"
BASIS_OUTCOME_UNAVAILABLE = "POINT_IN_TIME_OUTCOME_UNAVAILABLE"
BASIS_PREDICTION_FIELD_ABSENT = "FROZEN_PREDICTION_FIELD_ABSENT"

BASIS_EXCLUSION_REASONS: tuple[str, ...] = (
    BASIS_CAPTURE_ABSENT,
    BASIS_CAPTURE_NOT_BEFORE_CUTOFF,
    BASIS_PROVISIONAL,
    BASIS_FINALITY_UNPROVABLE,
    BASIS_FINALITY_NOT_BEFORE_CUTOFF,
    BASIS_OUTCOME_UNAVAILABLE,
    BASIS_PREDICTION_FIELD_ABSENT,
)

#: The declared precedence of the exclusions above.  A row is counted under ONE
#: reason, and the order is a rule rather than an artefact: a frozen prediction
#: row that does not carry the field the surface is defined on was never a
#: candidate observation at all, so it is counted before its timing is examined.
BASIS_EXCLUSION_PRECEDENCE = (
    "FROZEN_PREDICTION_FIELD_ABSENT, then the point-in-time timing ladder "
    "(POINT_IN_TIME_CAPTURE_ABSENT / CAPTURE_NOT_BEFORE_CUTOFF / OBSERVATION_PROVISIONAL_AT_CUTOFF / "
    "OFFICIAL_FINALITY_UNPROVABLE / OFFICIAL_FINALITY_NOT_BEFORE_CUTOFF), then "
    "POINT_IN_TIME_OUTCOME_UNAVAILABLE"
)

#: The declared fit-basis CANDIDATE population, stated once and carried by every
#: block that builds a basis.  Candidacy is a different question from SCORED
#: eligibility, and the two read different evidence: a scored figure (and the
#: origin a transform is applied at) keeps the CURRENT-STATE FINAL filter, while a
#: candidate row is decided by the CERTIFICATION, so a target event that has since
#: lost its current FINAL state still contributes the rows it was predicted under
#: and cannot remove them from a later origin's basis.
FIT_BASIS_CANDIDATE_POLICY = (
    "candidates are the FROZEN prediction rows of every APPLICABLE certified anchor target event -- "
    "'applicable' is decided by the CERTIFICATION (the anchor bundle's own certified xpts_v1 run for "
    "that event), never by the mutable current events.finished / events.data_checked row -- so a target "
    "event that has since lost its current FINAL state still contributes the rows it was predicted "
    "under and cannot remove them from a later origin's basis.  SCORED eligibility and the ORIGIN set "
    "are a different question with a different filter, and are left exactly as PE-2 declares them: only "
    "a FINAL target event is scored, and only a FINAL target event is an origin a transform is applied "
    "at, so an event that is not final today is excluded and counted rather than fitted for."
)

#: The declared fit-basis policy, stated once and carried by every block that
#: builds a basis, so the rule a fitted transform obeyed is readable from the
#: artifact rather than inferred from the code that produced it.
BASIS_MEMBERSHIP_POLICY = (
    "membership is the FROZEN prediction rows of the strictly earlier CANDIDATES -- every applicable "
    "certified anchor target event of the anchor xpts_v1 runs -- read from the immutable projection "
    "rows, so a fixture that has since lost its played flag, a realised player_gameweeks row that has "
    "since been removed or blanked, a scheduled-placeholder flag that has since appeared and a players "
    "row that has since been re-listed cannot add to or remove from a historical basis -- combined with "
    "PE-5 append-only point-in-time observation captures whose official finality AND capture both fall "
    "strictly before the origin's OWN certified cutoff.  The realised outcome is derived against the "
    "position PERSISTED WITH THE PREDICTION, and the outcome VALUE is the capture's, so a later "
    "correction is a later observation and cannot rewrite an earlier transform.  A row whose timing "
    "cannot be proven is excluded and counted with its own reason."
)

#: Declared a priori.  A reliability bin whose observed frequency differs from its
#: mean stated probability by more than this, and which meets the declared bin
#: sample floor, is a diagnosed miscalibration.  A diagnosis is not a ranking and
#: no skill score is derived from it.
MATERIAL_CALIBRATION_GAP_TOLERANCE = 0.05
#: Declared a priori, in the units of the surface (points).
MATERIAL_EV_BIAS_TOLERANCE_POINTS = 0.25
MATERIAL_COMPONENT_BIAS_TOLERANCE_POINTS = 0.25

ASSIST_MAPPING_NO_CHANGE = "NO_CHANGE"
ASSIST_MAPPING_CANDIDATE = "CANDIDATE_FOR_REVIEW"
ASSIST_MAPPING_FLAG = "FPL_ASSIST_MAPPING_UNCALIBRATED"

#: The FPL assist mapping is the ONE expected-value constant PE-8 may fit.
ASSIST_MAPPING_SURFACE = "FPL_ASSIST_MAPPING"
ASSIST_MAPPING_GRAIN = wf.GRAIN_PLAYER_FIXTURE

PROBABILITY_SURFACE_CLAIM = (
    "Brier, its reference score and a binned reliability view at the declared grain; the reference "
    "score is carried beside the Brier so the number is interpretable rather than bare, and no skill "
    "score is derived and no arm is ranked"
)

#: Surfaces PE-8 deliberately does NOT widen onto, with the reason.  A future slice
#: that genuinely needs one of these arrives as its own evidenced artifact-contract
#: change, not as a silent broadening here.
EXCLUDED_SURFACES: tuple[dict[str, str], ...] = (
    {
        "surface": "p_goal",
        "reason": (
            "exists only in monte_carlo_distributions, a different run family with its own dependency "
            "closure; scoring it would widen the provenance surface with no V1 requirement behind it"
        ),
    },
    {
        "surface": "p_assist",
        "reason": (
            "exists only in monte_carlo_distributions, a different run family with its own dependency "
            "closure; scoring it would widen the provenance surface with no V1 requirement behind it"
        ),
    },
    {
        "surface": "*_xpts components",
        "reason": (
            "an expected-points value is not a probability; a Brier score computed from expected points "
            "is meaningless, so PE-8 computes none and reports these components as bias only"
        ),
    },
    {
        "surface": "monte_carlo draws / total-points distribution",
        "reason": (
            "the Monte Carlo artifact stores quantiles, never draws; no draw-level or full-distribution "
            "artifact is assumed, and no CRPS is computed or claimed"
        ),
    },
)

#: Production flags that make a component's bias describable but not causally
#: interpretable, DECLARED PER COMPONENT: the flag has to be about the component
#: in question.  ``PENALTIES_EMBEDDED`` concerns the goal and assist mappings, so
#: it does not disqualify an appearance figure; ``BONUS_SOFT`` concerns bonus, so
#: it does not disqualify a save figure.  A component figure carries whatever
#: flags were recorded on the rows that produced it, and this table decides
#: whether any of them makes its bias a description rather than a causal claim.
NON_CAUSAL_FLAGS_BY_COMPONENT: dict[str, frozenset[str]] = {
    "appearance_xpts": frozenset(),
    "goal_xpts": frozenset(
        {"PENALTIES_EMBEDDED", "MISSING_XG_RATE", "TEAM_XG_CAP_APPLIED", "TEAM_XG_EXCESS_WARN"}
    ),
    "assist_xpts": frozenset(
        {
            "PENALTIES_EMBEDDED",
            "FPL_ASSIST_MAPPING_UNCALIBRATED",
            "MISSING_XA_RATE",
            "NO_TEAM_SPECIFIC_HISTORICAL_PRIOR",
        }
    ),
    "clean_sheet_xpts": frozenset({"CS_MINUTES_APPROX_V1", "TEAM_MODEL_SENSITIVE"}),
    "goals_conceded_xpts": frozenset({"TEAM_MODEL_SENSITIVE"}),
    "defcon_xpts": frozenset(
        {"DEFCON_PRIOR_WEAK", "DEFCON_MINUTE_MIXTURE_UNAVAILABLE", "HISTORICAL_POSITION_UNKNOWN"}
    ),
    "save_xpts": frozenset({"SAVE_MODEL_LOW_CONFIDENCE", "SAVE_MINUTE_MIXTURE_UNAVAILABLE"}),
    "yellow_card_xpts": frozenset(),
    "bonus_xpts": frozenset({"BONUS_SOFT", "BONUS_LOW_CONFIDENCE", "BPS_RULE_DISCONTINUITY"}),
    "soft_xpts": frozenset({"BONUS_SOFT", "BONUS_LOW_CONFIDENCE", "BPS_RULE_DISCONTINUITY"}),
}

#: The CORE is a mixture of the eight core components, so any flag that
#: disqualifies one of them disqualifies the aggregate; the same is true of the
#: total, which additionally carries the unmodelled-scoring note.
_CORE_COMPONENTS = (
    "appearance_xpts",
    "goal_xpts",
    "assist_xpts",
    "clean_sheet_xpts",
    "goals_conceded_xpts",
    "defcon_xpts",
    "save_xpts",
    "yellow_card_xpts",
)
NON_CAUSAL_FLAGS_BY_COMPONENT["core_xpts"] = frozenset(
    flag for name in _CORE_COMPONENTS for flag in NON_CAUSAL_FLAGS_BY_COMPONENT[name]
)
NON_CAUSAL_FLAGS_BY_COMPONENT["total_xpts"] = frozenset(
    NON_CAUSAL_FLAGS_BY_COMPONENT["core_xpts"] | NON_CAUSAL_FLAGS_BY_COMPONENT["soft_xpts"]
)

#: A component with no declared entry is treated conservatively: any observed
#: non-causal-class flag disqualifies it.
NON_CAUSAL_COMPONENT_FLAGS = frozenset(
    {
        "BONUS_SOFT",
        "BONUS_LOW_CONFIDENCE",
        "BPS_RULE_DISCONTINUITY",
        "CS_MINUTES_APPROX_V1",
        "FPL_ASSIST_MAPPING_UNCALIBRATED",
        "PENALTIES_EMBEDDED",
        "SAVE_MODEL_LOW_CONFIDENCE",
        "SAVE_MINUTE_MIXTURE_UNAVAILABLE",
        "DEFCON_PRIOR_WEAK",
        "DEFCON_MINUTE_MIXTURE_UNAVAILABLE",
        "HISTORICAL_POSITION_UNKNOWN",
        "MISSING_XG_RATE",
        "MISSING_XA_RATE",
        "NO_TEAM_SPECIFIC_HISTORICAL_PRIOR",
        "TEAM_XG_CAP_APPLIED",
        "TEAM_XG_EXCESS_WARN",
        "TEAM_MODEL_SENSITIVE",
    }
)


class CalibrationEvaluationError(RuntimeError):
    """The evaluation could not be produced from the available evidence."""


# ---------------------------------------------------------------------------
# Declared surfaces
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ComponentDefinition:
    """One persisted expected-points component and its realised counterpart.

    ``grain`` is stated on every figure because the components are persisted PER
    PLAYER FIXTURE: reporting one at event grain would silently change which rows
    it covers, and a double gameweek would then be two observations instead of
    one.
    """

    component: str
    selector: str
    realised: str
    interpretation: str
    #: Set when a position outside the component's declared positions earns a real
    #: zero rather than a gap, so the coverage of those rows is stated.
    zero_rule: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "component": str(self.component),
            "persisted_field": str(self.component),
            "grain": wf.GRAIN_PLAYER_FIXTURE,
            "grain_note": (
                "persisted per player x fixture in the anchor xpts_v1 payload; NOT an event-grain "
                "figure and never summed into one unless the source does so"
            ),
            "realised_target": str(self.realised),
            "interpretation": str(self.interpretation),
            "structural_zero_rule": None if self.zero_rule is None else str(self.zero_rule),
        }


#: Every component the ``xpts_v1`` payload persists, with the realised counterpart
#: it is scored against -- both read from the versioned scoring-rules object, so
#: the producer's definition and this target cannot drift apart.
COMPONENT_DEFINITIONS: tuple[ComponentDefinition, ...] = (
    ComponentDefinition(
        component="appearance_xpts",
        selector="appearance_points",
        realised="appearance points actually earned from the realised minutes (1 under 60, 2 at 60+)",
        interpretation="direct: every input is the realised minutes themselves",
    ),
    ComponentDefinition(
        component="goal_xpts",
        selector="goal_points",
        realised="goals_scored x goal_points_for(position)",
        interpretation=(
            "approximation-flagged at production time: penalties remain embedded in xG and are not "
            "separated, and the team xG cap may have been applied"
        ),
    ),
    ComponentDefinition(
        component="assist_xpts",
        selector="assist_points",
        realised="assists x assist_points",
        interpretation=(
            "approximation-flagged at production time: the xA -> FPL assist mapping is uncalibrated "
            "(FPL_ASSIST_MAPPING_UNCALIBRATED) and penalties remain embedded in xA"
        ),
    ),
    ComponentDefinition(
        component="clean_sheet_xpts",
        selector="clean_sheet_points",
        realised=(
            "clean-sheet points actually earned under the canonical rule (60+ minutes, no goal "
            "conceded while on, and the position actually receives clean-sheet points)"
        ),
        interpretation=(
            "approximation-flagged at production time (CS_MINUTES_APPROX_V1): the goal hazard over "
            "the exposure branches is an approximation"
        ),
    ),
    ComponentDefinition(
        component="goals_conceded_xpts",
        selector="goals_conceded_points",
        realised="-(floor(conceded / goals_conceded_per_deduction)) x deduction points, for GKP/DEF only",
        interpretation="direct for GKP/DEF; identically zero elsewhere by the scoring rules",
        zero_rule=(
            "a position outside goals_conceded_positions earns a real zero deduction, so the row is "
            "covered rather than dropped"
        ),
    ),
    ComponentDefinition(
        component="defcon_xpts",
        selector="defcon_points",
        realised="defcon_points where defensive_contribution reaches the position's declared threshold",
        interpretation=(
            "direct on the calibrated probability it is built from; the inverse action rate it "
            "consumes may be weakly evidenced (DEFCON_PRIOR_WEAK)"
        ),
        zero_rule=(
            "a position outside the declared DefCon positions (GKP) earns no DefCon points, so its "
            "realised value is a real zero and the row is covered rather than dropped"
        ),
    ),
    ComponentDefinition(
        component="save_xpts",
        selector="save_points",
        realised="floor(saves / saves_per_point), for GKP only",
        interpretation=(
            "approximation-flagged at production time: the save rate is shrunk toward a prior "
            "(SAVE_MODEL_LOW_CONFIDENCE) and the pressure multiplier is bounded"
        ),
        zero_rule=(
            "an outfield position earns a real zero save points, so the row is covered rather than "
            "dropped"
        ),
    ),
    ComponentDefinition(
        component="yellow_card_xpts",
        selector="yellow_card_points",
        realised="yellow_cards x yellow_card_points",
        interpretation="direct: a shrunk per-90 rate times the declared card penalty",
    ),
    ComponentDefinition(
        component="bonus_xpts",
        selector="bonus_points",
        realised="the official bonus points column for the same player and fixture",
        interpretation=(
            "soft residual only (BONUS_SOFT, BONUS_LOW_CONFIDENCE, BPS_RULE_DISCONTINUITY): bonus is "
            "not structurally modelled, so its bias is descriptive, not causal"
        ),
    ),
    ComponentDefinition(
        component="soft_xpts",
        selector="soft_points",
        realised="the official bonus points column, because SOFT_COMPONENTS contains bonus only",
        interpretation="identical to bonus_xpts by construction: soft_xpts is the bonus residual alone",
    ),
    ComponentDefinition(
        component="core_xpts",
        selector="core_points",
        realised=(
            "the realised CORE reconstructed from proven official facts by "
            "calibration.actual_modelled_core (the same eight components the simulator models)"
        ),
        interpretation="direct: the realised CORE is reconstructed with the frozen scorer, not restated",
    ),
    ComponentDefinition(
        component="total_xpts",
        selector="total_points",
        realised="the official total_points column for the same player and fixture",
        interpretation=(
            "core plus soft, compared against a column that also carries unmodelled scoring (red "
            "cards, own goals, penalties), so a residual is expected and is counted, not hidden"
        ),
    ),
)

COMPONENT_BIAS_CONVENTION = sb.BIAS_CONVENTION


# ---------------------------------------------------------------------------
# Realised counterparts
# ---------------------------------------------------------------------------


def realised_component(
    selector: str, outcome: Mapping[str, Any], position: str, rules=DEFAULT_SCORING_RULES
) -> float | None:
    """The realised counterpart of one persisted component, or ``None`` for a gap.

    ``None`` means the realised quantity could not be established from the stored
    official facts.  It is counted as a data gap and never scored as zero -- with
    one declared exception.  Where the frozen scoring rules answer the question by
    POSITION alone, the real zero is returned BEFORE the component's own outcome
    column is consulted: a forward's save points and a forward's goals-conceded
    deduction do not depend on ``saves`` or ``goals_conceded`` any more than a
    GKP's DefCon points depend on ``defensive_contribution``, so an absent column
    there is not an unavailable outcome and must not drop a covered row out of the
    component population.  The order matters: consulting an irrelevant column
    first turns a structural zero into a gap.
    """

    if outcome is None:
        return None
    minutes = outcome.get("minutes")
    if minutes is None:
        # The row-level availability test every selector shares: without minutes
        # there is no official performance record to score at all (a scheduled
        # placeholder has a NULL minutes column), so this is a gap rather than a
        # structural zero.
        return None
    minutes = int(minutes)
    if selector == "appearance_points":
        if minutes >= rules.clean_sheet_minutes_required:
            return float(rules.appearance_long_points)
        if minutes > 0:
            return float(rules.appearance_short_points)
        return 0.0
    if selector == "goal_points":
        goals = outcome.get("goals_scored")
        if goals is None:
            return None
        return float(int(goals) * rules.goal_points_for(position))
    if selector == "assist_points":
        assists = outcome.get("assists")
        if assists is None:
            return None
        return float(int(assists) * rules.assist_points)
    if selector == "clean_sheet_points":
        conceded = outcome.get("goals_conceded")
        if conceded is None:
            return None
        earned = rules.earns_clean_sheet_points(position, minutes, int(conceded))
        return float(rules.clean_sheet_points_for(position) if earned else 0)
    if selector == "goals_conceded_points":
        # A position outside goals_conceded_positions suffers no deduction whatever
        # it conceded, so its realised value is a REAL zero and it is returned
        # BEFORE the conceded column is read.
        if position not in rules.goals_conceded_positions:
            return 0.0
        conceded = outcome.get("goals_conceded")
        if conceded is None:
            return None
        steps = int(conceded) // int(rules.goals_conceded_per_deduction)
        return -float(steps) * abs(float(rules.goals_conceded_points_for(position)))
    if selector == "defcon_points":
        # A position outside the declared DefCon positions earns no DefCon points
        # at all -- GKP is such a position, and the frozen rules say so explicitly
        # (``defcon_positions`` excludes it).  That is a REAL zero, so it is
        # returned BEFORE the threshold and the contribution are consulted: the
        # realised value does not depend on either, and treating a missing
        # contribution column as "unknown" here would drop a covered row out of
        # the component population for a question that was already answered.
        if position not in rules.defcon_positions:
            return 0.0
        threshold = rules.defcon_threshold_for(position)
        contribution = outcome.get("defensive_contribution")
        if threshold is None or contribution is None:
            return None
        return float(rules.defcon_points) if float(contribution) >= threshold else 0.0
    if selector == "save_points":
        # The same rule as DefCon, on the other side: only a GKP can earn save
        # points, so an outfield position's realised value is a REAL zero and it is
        # returned BEFORE the saves column is read.
        if position != "GKP":
            return 0.0
        saves = outcome.get("saves")
        if saves is None:
            return None
        return float(int(saves) // int(rules.saves_per_point))
    if selector == "yellow_card_points":
        cards = outcome.get("yellow_cards")
        if cards is None:
            return None
        return float(int(cards) * int(rules.yellow_card_points))
    if selector in ("bonus_points", "soft_points"):
        bonus = outcome.get("bonus")
        # The bonus column is a genuine zero far more often than it is absent, so
        # an absent column is a gap and a stored 0 is a realised 0.
        return None if bonus is None else float(int(bonus))
    if selector == "total_points":
        total = outcome.get("total_points")
        return None if total is None else float(total)
    if selector == "core_points":
        # The frozen scorer's own reconstruction, so the producer's event and this
        # target are the same event by construction.
        return mc_calibration.actual_modelled_core(outcome, position, rules)
    raise CalibrationEvaluationError(f"unknown component selector {selector!r}")


def structural_zero_rule(
    selector: str, position: str, *, rules=DEFAULT_SCORING_RULES
) -> str | None:
    """Why this position earns a REAL zero on this component, or ``None`` if it does not.

    A position outside the component's declared positions earns nothing from it by
    the frozen scoring rules -- a GKP earns no DefCon points, an outfield player
    earns no save points, a forward concedes no deductions -- so its realised value
    is a genuine zero, INDEPENDENT of the component's own outcome column, and the
    row BELONGS in the component population.  ``realised_component`` therefore
    returns that zero before consulting the column, and this function reports the
    rule by token.  Reporting the count makes the coverage visible instead of
    leaving a reader to infer it from an ``N``.
    """

    if selector == "defcon_points" and position not in rules.defcon_positions:
        return f"{position} is outside the declared DefCon positions, so it earns no DefCon points"
    if selector == "goals_conceded_points" and position not in rules.goals_conceded_positions:
        return f"{position} is outside goals_conceded_positions, so it suffers no deduction"
    if selector == "save_points" and position != "GKP":
        return f"{position} is not GKP, so it earns no save points"
    return None


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _final_events(conn: sqlite3.Connection, events: Sequence[int]) -> tuple[list[int], list[dict[str, Any]]]:
    """The target events whose stored official state is FINAL, and the rest.

    This answers ONE question -- SCORED eligibility -- and it is deliberately not
    the question a fit basis asks.  The declared probability population is
    restricted to FINAL target events, so a provisional or in-progress event is
    reported with its state instead of contributing rows, and the same set is the
    ORIGIN set, because a transform is only ever applied at an origin where a
    figure is scored.  A fit basis is a different role with a different filter: its
    candidates come from the certification
    (:func:`fit_basis_candidates`), so this current-state answer never adds a row
    to or removes a row from an earlier transform's basis.
    """

    final: list[int] = []
    excluded: list[dict[str, Any]] = []
    for event in sorted({int(event) for event in events}):
        state, reasons = _event_state(conn, event)
        if state == "FINAL":
            final.append(event)
        else:
            excluded.append({"event": event, "state": state, "basis": list(reasons)})
    return final, excluded


def fit_basis_candidates(
    anchor: wf.CertifiedAnchor,
) -> tuple[list[int], dict[int, int], list[dict[str, Any]]]:
    """Every APPLICABLE certified anchor target event, and its own ``xpts_v1`` run.

    "Applicable" is decided by the CERTIFICATION and not by the mutable current
    ``events`` row: the anchor bundle for the event names the certified
    ``xpts_v1`` run, so that event's frozen prediction rows are fit-basis
    candidates whether or not today's ``events.finished`` / ``events.data_checked``
    still call the event FINAL.  Whether a candidate row then enters an origin's
    basis is decided by the PE-5 point-in-time rules alone
    (:func:`point_in_time_basis`), never by this current-state flag.

    A target event whose certified bundle names no ``xpts_v1`` run has no frozen
    prediction rows to contribute; it is returned with its reason rather than
    silently dropped from the candidate pool or quietly treated as an empty one.
    """

    events = sorted({int(event) for event in anchor.event_ids})
    applicable: list[int] = []
    runs: dict[int, int] = {}
    inapplicable: list[dict[str, Any]] = []
    for event in events:
        run_id = anchor.for_event(event).run_id("xpts_v1")
        if run_id is None:
            inapplicable.append(
                {
                    "event": int(event),
                    "reason": (
                        "the certified bundle names no xpts_v1 run for this target event, so it has no "
                        "frozen prediction rows to contribute to a fit basis"
                    ),
                }
            )
            continue
        applicable.append(int(event))
        runs[int(event)] = int(run_id)
    return applicable, runs, inapplicable


def _event_state(conn: sqlite3.Connection, event: int) -> tuple[str, list[str]]:
    from . import planning

    state, reasons = planning.event_data_state(conn, int(event))
    return str(state), list(reasons)


def _xpts_runs(anchor: wf.CertifiedAnchor, events: Sequence[int]) -> dict[int, int]:
    runs: dict[int, int] = {}
    for event in events:
        run_id = anchor.for_event(int(event)).run_id("xpts_v1")
        if run_id is not None:
            runs[int(event)] = int(run_id)
    return runs


def _ancillary_runs(anchor: wf.CertifiedAnchor, events: Sequence[int], family: str) -> dict[int, int]:
    runs: dict[int, int] = {}
    for event in events:
        run_id = anchor.for_event(int(event)).run_id(family)
        if run_id is not None:
            runs[int(event)] = int(run_id)
    return runs


def _sample_block(*, target_events: int, events_with_observations: int, observations: int) -> dict[str, Any]:
    """PE-2's declared sample-size disclosure, carried by token rather than restated."""

    if (
        target_events >= sb.MIN_TARGET_EVENTS_FOR_DESCRIPTIVE
        and observations >= sb.MIN_OBSERVATIONS_FOR_DESCRIPTIVE
    ):
        interpretation = sb.SAMPLE_DESCRIPTIVE_ONLY
    else:
        interpretation = sb.SAMPLE_INSUFFICIENT
    return {
        "target_events": int(target_events),
        "target_events_with_observations": int(events_with_observations),
        "observations": int(observations),
        "sample_interpretation": interpretation,
        "policy": {
            "policy_version": sb.SAMPLE_POLICY_VERSION,
            "min_target_events_for_descriptive_reporting": sb.MIN_TARGET_EVENTS_FOR_DESCRIPTIVE,
            "min_observations_for_descriptive_reporting": sb.MIN_OBSERVATIONS_FOR_DESCRIPTIVE,
            "basis": (
                "the PE-2 disclosure floor, NOT a significance threshold; nothing in PE-8 tests a "
                "hypothesis, no arm is called better, and a promotion requires materially more "
                "evidence than a descriptive report"
            ),
        },
    }


def _figure(probabilities: Sequence[float], outcomes: Sequence[float]) -> dict[str, Any]:
    """Brier, its reference score and a reliability view over one aligned population."""

    observed_rate = (sum(outcomes) / len(outcomes)) if outcomes else None
    return {
        "n": len(probabilities),
        "brier": wm.brier_score(probabilities, outcomes).as_dict(),
        "brier_reference": wm.brier_reference_score(outcomes).as_dict(),
        "observed_rate": None if observed_rate is None else round(observed_rate, wm.METRIC_DECIMALS),
        "reliability": pc.reliability_view(probabilities, outcomes),
        "claim": PROBABILITY_SURFACE_CLAIM,
    }


def _diagnosis(bins_over_floor: int, max_abs_gap: float | None) -> dict[str, Any]:
    """The declared a priori diagnosis rule for one surface."""

    if bins_over_floor == 0:
        return {
            "status": DIAGNOSIS_INSUFFICIENT,
            "max_abs_gap_over_floor": None,
            "bins_over_floor": 0,
            "tolerance": MATERIAL_CALIBRATION_GAP_TOLERANCE,
            "reasons": [
                "no bin reached the declared bin sample floor, so no calibration claim is made about "
                "this surface"
            ],
        }
    if max_abs_gap is not None and max_abs_gap > MATERIAL_CALIBRATION_GAP_TOLERANCE:
        return {
            "status": DIAGNOSIS_MISCALIBRATED,
            "max_abs_gap_over_floor": round(float(max_abs_gap), wm.METRIC_DECIMALS),
            "bins_over_floor": int(bins_over_floor),
            "tolerance": MATERIAL_CALIBRATION_GAP_TOLERANCE,
            "reasons": [
                "at least one bin at or above the declared floor differs from its mean stated "
                f"probability by more than the declared tolerance "
                f"({MATERIAL_CALIBRATION_GAP_TOLERANCE}); this is a diagnosis, not a ranking"
            ],
        }
    return {
        "status": DIAGNOSIS_NO_MATERIAL_DEFECT,
        "max_abs_gap_over_floor": (
            None if max_abs_gap is None else round(float(max_abs_gap), wm.METRIC_DECIMALS)
        ),
        "bins_over_floor": int(bins_over_floor),
        "tolerance": MATERIAL_CALIBRATION_GAP_TOLERANCE,
        "reasons": [
            "every bin at or above the declared floor is within the declared tolerance of the stated "
            "probability; within the available sample no material defect is detected"
        ],
    }


def _metric_or_none(function, predicted: Sequence[float], actual: Sequence[float]) -> dict[str, Any]:
    return function(predicted, actual).as_dict()


# ---------------------------------------------------------------------------
# The same-population gate
# ---------------------------------------------------------------------------


def comparison_gate(
    incumbent_keys: Iterable[Sequence[Any]],
    arm_keys: Iterable[Sequence[Any]],
    *,
    grain: str = wf.GRAIN_PLAYER_FIXTURE,
) -> dict[str, Any]:
    """PE-2's same-population gate for one comparison, as a declared status.

    The gate is PE-2's own :func:`walk_forward.assert_same_population`; a
    mismatch is reported as UNREACHABLE with both sides named, never resolved by
    scoring whichever sample happened to be available.
    """

    try:
        wf.assert_same_population(incumbent_keys, arm_keys)
    except wf.PopulationMismatch as mismatch:
        return {
            "status": STATUS_POPULATION_MISMATCH,
            "same_population": False,
            "population_digest": None,
            "incumbent_only_n": len(mismatch.model_only),
            "arm_only_n": len(mismatch.baseline_only),
            "incumbent_only_example": [list(key) for key in mismatch.model_only[:5]],
            "arm_only_example": [list(key) for key in mismatch.baseline_only[:5]],
            "detail": (
                "the arms do not cover the identical eligible key set, so no comparison is reported "
                "rather than a comparison over two different samples"
            ),
        }
    keys = [tuple(int(part) for part in key) for key in arm_keys]
    return {
        "status": STATUS_OK,
        "same_population": True,
        "population_digest": wf.canonical_population_digest(keys, grain=grain),
        "incumbent_only_n": 0,
        "arm_only_n": 0,
        "incumbent_only_example": [],
        "arm_only_example": [],
        "detail": None,
    }


# ---------------------------------------------------------------------------
# Causal walk-forward transforms
# ---------------------------------------------------------------------------


def _count_exclusion(counters: dict[str, int], reason: str) -> None:
    """Count one refusal under exactly one reason."""

    key = str(reason)
    counters[key] = counters.get(key, 0) + 1


@dataclass(frozen=True)
class FrozenPredictionRow:
    """One persisted prediction row of an earlier event's own certified run.

    This is the FROZEN half of a fit basis.  ``player_fixture_xpts_projections``
    refuses UPDATE and DELETE at storage level, so the position a prediction was
    made at and the probability it stated cannot be re-pointed by a later refresh
    of ``fixtures``, ``player_gameweeks`` or ``players``.  Basis membership is
    therefore a property of what was PERSISTED at prediction time, never of what
    the current tables happen to say about a historical row.
    """

    event: int
    player_id: int
    fixture_id: int
    xpts_run_id: int
    position: str
    payload: Mapping[str, Any]

    @property
    def key(self) -> tuple[int, int, int]:
        return (int(self.event), int(self.player_id), int(self.fixture_id))

    def as_dict(self) -> dict[str, Any]:
        return {
            "event": int(self.event),
            "player_id": int(self.player_id),
            "fixture_id": int(self.fixture_id),
            "xpts_run_id": int(self.xpts_run_id),
            "position": str(self.position),
            "population": "frozen prediction row (immutable at storage level)",
        }


def frozen_prediction_rows(
    conn: sqlite3.Connection, *, runs: Mapping[int, int], events: Sequence[int]
) -> dict[int, list[FrozenPredictionRow]]:
    """Each event's frozen prediction rows, read from its own certified run.

    Read straight from the persisted, immutable projection rows rather than
    through the declared SCORING population, because the two answer different
    questions.  The scoring population is PE-2's declared CURRENT-STATE population
    -- played fixtures, a realised row that exists now and is not a placeholder, a
    position read from today's ``players`` table -- and it is the population every
    SCORED figure covers.  A fit basis is a different role: what was PREDICTED at
    the position it was predicted for, and what was THEN captured as final.  So a
    row that has since lost its played-fixture flag, its realised row, or its
    current position listing is still evidence of what an earlier transform was
    fitted on, and the two populations are named separately wherever both are
    reported.

    The ``events`` this reads are the CANDIDATES of a fit basis, and the caller
    supplies them from the CERTIFICATION (:func:`fit_basis_candidates`) rather than
    from the current ``events`` row: a target event that is not FINAL today was
    still PREDICTED, and nothing about that prediction becomes untrue because the
    current official-state flag changed afterwards.
    """

    wanted = sorted({int(event) for event in events})
    out: dict[int, list[FrozenPredictionRow]] = {event: [] for event in wanted}
    for event in wanted:
        run_id = runs.get(event)
        if run_id is None:
            continue
        for record in conn.execute(
            "SELECT player_id, fixture_id, position, payload_json "
            "FROM player_fixture_xpts_projections WHERE projection_run_id=? AND event=? "
            "ORDER BY player_id, fixture_id",
            (int(run_id), int(event)),
        ):
            out[event].append(
                FrozenPredictionRow(
                    event=int(event),
                    player_id=int(record["player_id"]),
                    fixture_id=int(record["fixture_id"]),
                    xpts_run_id=int(run_id),
                    position=str(record["position"] or ""),
                    payload=json.loads(record["payload_json"]) if record["payload_json"] else {},
                )
            )
    return out


def point_in_time_basis(
    *,
    conn: sqlite3.Connection,
    prediction_rows: Mapping[int, Sequence[FrozenPredictionRow]],
    cutoffs: Mapping[int, str],
    definition: sb.ProbabilityMetricDefinition,
    origins: Sequence[int] | None = None,
) -> tuple[dict[int, list[pc.CalibrationObservation]], dict[int, dict[str, int]]]:
    """Each origin's fit basis: frozen prediction rows PLUS PE-5 point-in-time evidence.

    For every origin event ``E`` the basis is the FROZEN prediction rows of strictly
    earlier CANDIDATES whose outcome is proven by an APPEND-ONLY PE-5 observation
    capture to have been officially final AND captured strictly before ``E``'s
    certified cutoff.  The two populations are separate on purpose:

    * ``prediction_rows`` is the CANDIDATE pool -- every applicable certified anchor
      target event (see :func:`fit_basis_candidates`), which is decided by the
      certification and NOT by today's ``events`` row;
    * ``origins`` is the ORIGIN set -- the SCORED target events, which do keep the
      current-state FINAL filter, because a transform is only ever applied where a
      figure is scored.  It defaults to the candidate pool only for callers that
      have no separate origin set.

    Four properties follow, and each is a rule rather than an implementation
    detail:

    * MEMBERSHIP COMES FROM THE PREDICTION, NOT FROM THE CURRENT TABLES.  The rows
      are the persisted projection rows of the earlier events' own certified runs,
      so neither a fixture that has since lost its played flag, nor a realised
      ``player_gameweeks`` row that has since been removed or blanked, nor a
      scheduled-placeholder flag that has since appeared can add to or remove from
      a historical basis.
    * CANDIDACY COMES FROM THE CERTIFICATION, NOT FROM CURRENT FINALITY.  A
      candidate event that is no longer FINAL today still contributes the rows it
      was predicted under, so a current-state finality edit cannot shrink a later
      origin's basis.
    * THE POSITION IS THE PREDICTION-TIME POSITION.  The realised outcome is
      derived against the position PERSISTED WITH THE PREDICTION rather than
      against today's ``players`` row, because the question a prediction asked was
      asked at the position it was made at.
    * THE OUTCOME VALUE IS THE CAPTURE'S.  A later official refresh is a later
      capture, so it cannot retroactively rewrite what an earlier transform was
      fitted on.

    Every refusal is named rather than assumed:

    * ``FROZEN_PREDICTION_FIELD_ABSENT`` -- the prediction row does not carry the
      field this surface is defined on, so it was never a candidate observation;
    * ``POINT_IN_TIME_CAPTURE_ABSENT`` -- the key has no observation at all;
    * ``CAPTURE_NOT_BEFORE_CUTOFF`` -- the result was official in time, but the
      repository captured it after the origin's cutoff;
    * ``OBSERVATION_PROVISIONAL_AT_CUTOFF`` -- the visible read was provisional;
    * ``OFFICIAL_FINALITY_UNPROVABLE`` -- the capture states no official finality;
    * ``OFFICIAL_FINALITY_NOT_BEFORE_CUTOFF`` -- the postponed-result case: the
      football event happened, but its result was not official when the origin's
      prediction was issued;
    * ``POINT_IN_TIME_OUTCOME_UNAVAILABLE`` -- the capture does not carry the field
      this surface's realised outcome is defined on.

    Every refusal is counted by reason, per origin, so the size of the excluded
    evidence is visible rather than silently missing, and the precedence between
    the reasons is declared by :data:`BASIS_EXCLUSION_PRECEDENCE` rather than left
    to the order of the branches below.
    """

    candidates = sorted({int(event) for event in prediction_rows})
    origin_events = (
        sorted({int(origin) for origin in origins}) if origins is not None else list(candidates)
    )
    basis: dict[int, list[pc.CalibrationObservation]] = {origin: [] for origin in origin_events}
    excluded: dict[int, dict[str, int]] = {origin: {} for origin in origin_events}
    if not origin_events:
        return basis, excluded

    captures: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    for capture in ledger.observation_captures(
        conn,
        grain=ledger.GRAIN_PLAYER_FIXTURE,
        events=sorted(set(candidates) | set(origin_events)),
    ):
        if capture.get("fixture_id") is None:
            continue
        captures.setdefault(
            (int(capture["event"]), int(capture["player_id"]), int(capture["fixture_id"])), []
        ).append(capture)

    for origin in origin_events:
        cutoff = _require_cutoff(cutoffs, origin)
        for event in (candidate for candidate in candidates if candidate < origin):
            for row in prediction_rows[event]:
                statement = row.payload.get(definition.field)
                if statement is None:
                    _count_exclusion(excluded[origin], BASIS_PREDICTION_FIELD_ABSENT)
                    continue
                outcome, reason = _point_in_time_outcome(
                    captures.get(row.key, ()),
                    cutoff=cutoff,
                    selector=definition.outcome_selector,
                    position=row.position,
                )
                if reason is not None:
                    _count_exclusion(excluded[origin], reason)
                    continue
                basis[origin].append(
                    pc.CalibrationObservation(
                        event=int(row.event),
                        key=(int(row.player_id), int(row.fixture_id)),
                        probability=float(statement),
                        outcome=float(outcome),
                    )
                )
    for origin, counters in excluded.items():
        excluded[origin] = {key: int(value) for key, value in sorted(counters.items())}
    return basis, excluded


def _point_in_time_outcome(
    captures: Sequence[Mapping[str, Any]],
    *,
    cutoff: str,
    selector: str,
    position: str,
) -> tuple[float | None, str | None]:
    """``(realised outcome, exclusion reason)`` for one frozen row at one cutoff.

    ``position`` is the position the prediction was MADE at, taken from the frozen
    prediction row: the realised outcome a probability names has to be judged at
    the position the probability was stated for, not at whatever position the
    current ``players`` table lists today.
    """

    visible = [
        capture
        for capture in captures
        if capture.get("captured_at") is not None and str(capture["captured_at"]) < str(cutoff)
    ]
    if not visible:
        return None, _unavailable_evidence_reason(captures, cutoff=cutoff)
    # The declared supersession policy from PE-5, reused rather than restated, so
    # a key with several visible captures resolves exactly as the ledger resolves it.
    chosen = ledger.select_capture(visible)
    if chosen is None:  # pragma: no cover - ``visible`` is non-empty
        return None, BASIS_CAPTURE_ABSENT
    if str(chosen.get("observation_state")) != ledger.OBSERVATION_FINAL:
        return None, BASIS_PROVISIONAL
    final_at = str(chosen.get("official_final_at") or "").strip()
    if not final_at:
        return None, BASIS_FINALITY_UNPROVABLE
    if final_at >= str(cutoff):
        return None, BASIS_FINALITY_NOT_BEFORE_CUTOFF
    realised = sb.realised_probability_outcome(
        selector, chosen.get("payload") or {}, str(position)
    )
    if realised is None:
        return None, BASIS_OUTCOME_UNAVAILABLE
    return float(realised), None


def _unavailable_evidence_reason(
    captures: Sequence[Mapping[str, Any]], *, cutoff: str
) -> str:
    """Why no capture could be read before ``cutoff``, as one countable reason.

    The two causes are different evidence problems and are reported separately: a
    result that was not OFFICIAL yet (the football side -- the postponed case) and
    a result that was official but had not been CAPTURED yet (the repository
    side).  Neither is assumed known; both are counted.
    """

    if not captures:
        return BASIS_CAPTURE_ABSENT
    late_finality = any(
        str(capture.get("observation_state")) == ledger.OBSERVATION_FINAL
        and str(capture.get("official_final_at") or "").strip()
        and str(capture["official_final_at"]) >= str(cutoff)
        for capture in captures
    )
    return BASIS_FINALITY_NOT_BEFORE_CUTOFF if late_finality else BASIS_CAPTURE_NOT_BEFORE_CUTOFF


def _require_cutoff(cutoffs: Mapping[int, str], origin: int) -> str:
    """The origin's certified cutoff, or refuse: an origin without one is not causal."""

    value = str(cutoffs.get(int(origin)) or "").strip()
    if not value:
        raise CalibrationEvaluationError(
            f"origin event {int(origin)} declares no certified cutoff, so no point-in-time basis can "
            "be built for it"
        )
    return value


def causal_origins(
    basis_by_origin: Mapping[int, Sequence[pc.CalibrationObservation]],
    *,
    surface: str,
    grain: str = wf.GRAIN_PLAYER_FIXTURE,
    clip_floor: float = pc.DEFAULT_CLIP_FLOOR,
) -> dict[int, pc.CausalFit]:
    """One fit per target event, over that origin's own strictly earlier basis.

    The bases are supplied already filtered -- in production by
    :func:`point_in_time_basis`, which is what makes them PE-5 evidence rather
    than "every row from an earlier event".  Nothing here reaches for a row the
    supplied basis does not contain: a fit that borrowed beyond it would describe
    a population no reported figure covers.
    """

    return {
        int(origin): pc.fit_platt_causal(
            list(observations),
            origin_event=int(origin),
            surface=str(surface),
            grain=str(grain),
            clip_floor=clip_floor,
        )
        for origin, observations in sorted(basis_by_origin.items())
    }


def _not_fitted_block(surface: str, grain: str, diagnosis: Mapping[str, Any]) -> dict[str, Any]:
    """The declared record for a surface that produced NO challenger.

    A transform exists to correct a MEASURED miscalibration.  Where nothing is
    diagnosed, nothing is fitted: reporting fitted parameters "for description
    only" would put a candidate calibration in front of a reviewer with no defect
    for it to correct, and the phase forbids fitting speculatively.
    """

    status = str(diagnosis.get("status"))
    if status == DIAGNOSIS_INSUFFICIENT:
        reason = (
            "the sample cannot support a calibration claim on this surface, so no defect is diagnosed "
            "and NO challenger was constructed"
        )
    else:
        reason = (
            "no material defect is diagnosed on this surface, so NO challenger was constructed: a "
            "transform exists only to correct a measured miscalibration"
        )
    return {
        "surface": str(surface),
        "grain": str(grain),
        "status": STATUS_NOT_FITTED,
        "constructed": False,
        "reason": reason,
        "diagnosis_status": status,
        "method": pc.PLATT_FIT_METHOD,
        "policy_version": pc.CAUSAL_FIT_POLICY_VERSION,
        "calibration_policy_version": pc.PROBABILITY_CALIBRATION_POLICY_VERSION,
        "basis_policy": (
            "no basis was built: a basis exists to fit a transform, and no transform is admissible "
            "without a diagnosed defect"
        ),
        "origins": [],
        "rows_without_origin_transform": None,
        "excluded_by_reason": {},
        "basis_excluded_by_reason": {},
        "exclusion_reasons": [],
        "population": None,
        "gate": None,
        "incumbent_on_comparison_population": None,
        "candidate": None,
        "versions_in_force": [],
    }


def _challenger_block(
    *,
    conn: sqlite3.Connection,
    rows: Sequence[sb.ProbabilityScoringRow],
    prediction_rows: Mapping[int, Sequence[FrozenPredictionRow]],
    cutoffs: Mapping[int, str],
    definition: sb.ProbabilityMetricDefinition,
    grain: str,
    origins: Sequence[int],
) -> tuple[dict[str, Any], list[sb.ProbabilityScoringRow]]:
    """The causal challenger arm: fitted transform per origin, scored at its own origin.

    Constructed only after a diagnosis, never before: the caller reaches this
    function only for a MISCALIBRATED surface, and the block says so.  Returns the
    reportable block and the rows the arm could actually cover.  A row whose
    origin had no admissible transform is EXCLUDED AND COUNTED, never scored under
    a transform fitted somewhere else.

    Three populations meet here and are named separately, because they answer
    different questions and only one of them may read current state:

    * the CANDIDATES, ``prediction_rows``: the frozen prediction rows of every
      applicable certified anchor target event, decided by the CERTIFICATION;
    * the ORIGINS, ``origins``: the SCORED target events, which keep the
      current-state FINAL filter because a transform is only applied where a figure
      is scored;
    * the SCORED rows, ``rows``: PE-2's declared current-state population, which is
      what the incumbent and challenger figures cover.
    """

    basis_by_origin, excluded_by_origin = point_in_time_basis(
        conn=conn,
        prediction_rows=prediction_rows,
        cutoffs=cutoffs,
        definition=definition,
        origins=origins,
    )
    fits = causal_origins(basis_by_origin, surface=definition.metric, grain=grain)
    origin_records: list[dict[str, Any]] = []
    covered: list[sb.ProbabilityScoringRow] = []
    excluded_by_reason: dict[str, int] = {}
    # The basis exclusions, summed over the origins: the per-origin counters below
    # say WHICH origin had to do without which rows, and this says how much
    # evidence the surface's fits collectively could not use.
    basis_excluded_total: dict[str, int] = {}
    for counters in excluded_by_origin.values():
        for reason, count in counters.items():
            basis_excluded_total[str(reason)] = basis_excluded_total.get(str(reason), 0) + int(count)
    for origin in sorted(fits):
        fit = fits[origin]
        origin_rows = [row for row in rows if int(row.event) == int(origin)]
        record: dict[str, Any] = {
            "origin_event": int(origin),
            "cutoff": _require_cutoff(cutoffs, origin),
            "fit": fit.as_dict(),
            "rows_at_origin": len(origin_rows),
            "rows_scored_at_origin": 0,
            "basis_excluded_by_reason": dict(excluded_by_origin.get(int(origin), {})),
        }
        if fit.available:
            # The fitted payload is read back through the fail-closed reader
            # before it is reported: a transform that cannot state its identity and
            # the policy it was fitted under is not a reportable calibration, and
            # an absent or mismatched provenance stops the evaluation here rather
            # than producing a number a consumer cannot attribute.
            spec = fit.spec
            if spec is None:  # pragma: no cover - guarded by ``available``
                raise CalibrationEvaluationError("an available fit carries no spec")
            pc.from_payload(spec.as_payload())
            record["provenance"] = spec.as_payload()
            covered.extend(origin_rows)
            record["rows_scored_at_origin"] = len(origin_rows)
        else:
            reason = str(fit.reason or pc.FIT_INSUFFICIENT_OBSERVATIONS)
            excluded_by_reason[reason] = excluded_by_reason.get(reason, 0) + len(origin_rows)
        origin_records.append(record)

    block: dict[str, Any] = {
        "surface": str(definition.metric),
        "grain": str(grain),
        "status": None,
        "constructed": True,
        "diagnosis_status": DIAGNOSIS_MISCALIBRATED,
        "method": pc.PLATT_FIT_METHOD,
        "policy_version": pc.CAUSAL_FIT_POLICY_VERSION,
        "calibration_policy_version": pc.PROBABILITY_CALIBRATION_POLICY_VERSION,
        "basis_policy": BASIS_MEMBERSHIP_POLICY,
        "basis_evidence": {
            "ledger_version": ledger.OUTCOME_LEDGER_VERSION,
            "capture_version": ledger.OBSERVATION_CAPTURE_VERSION,
            "supersession_policy_version": ledger.SUPERSESSION_POLICY_VERSION,
            "grain": ledger.GRAIN_PLAYER_FIXTURE,
            "membership": (
                "frozen prediction rows of the strictly earlier candidates -- every applicable "
                "certified anchor target event of the anchor xpts_v1 runs"
            ),
            "candidate_policy": FIT_BASIS_CANDIDATE_POLICY,
            "candidate_events": [int(event) for event in sorted(prediction_rows)],
            "origin_events": sorted(int(origin) for origin in fits),
            "position": "the position persisted with each prediction row, never the current players row",
            "strictly_before_cutoff": True,
            "exclusion_precedence": BASIS_EXCLUSION_PRECEDENCE,
            "exclusion_reasons": list(BASIS_EXCLUSION_REASONS),
        },
        "origins": origin_records,
        "basis_excluded_by_reason": {
            str(key): int(value) for key, value in sorted(basis_excluded_total.items())
        },
        "excluded_by_reason": {str(key): int(value) for key, value in sorted(excluded_by_reason.items())},
    }
    block["rows_without_origin_transform"] = sum(
        1 for row in rows if not (fits.get(int(row.event)) and fits[int(row.event)].available)
    )
    return _score_challenger(block, rows, covered, fits, excluded_by_reason, grain=grain), covered


def _score_challenger(
    block: dict[str, Any],
    rows: Sequence[sb.ProbabilityScoringRow],
    covered: Sequence[sb.ProbabilityScoringRow],
    fits: Mapping[int, pc.CausalFit],
    excluded_by_reason: Mapping[str, int],
    *,
    grain: str,
) -> dict[str, Any]:
    """Score the arm on the identical population, or report it unreachable."""

    if not covered:
        block.update(
            {
                "status": STATUS_UNREACHABLE,
                "reason": (
                    "no origin had an admissible causal transform, so the challenger cannot cover the "
                    "comparison population"
                ),
                "exclusion_reasons": sorted(excluded_by_reason),
                "population": {
                    "population_digest": None,
                    "n": 0,
                    "rows_without_origin_transform": int(block["rows_without_origin_transform"]),
                    "excluded_by_reason": block["excluded_by_reason"],
                },
                "gate": None,
                "incumbent_on_comparison_population": None,
                "candidate": None,
            }
        )
        return block

    covered_keys = [row.key for row in covered]
    # Two independent derivations of the same set: the per-origin loop above, and
    # a filter over the declared population below.  The gate proves they agree, so
    # the incumbent figure and the candidate figure can never describe two
    # different samples.
    incumbent_rows = [row for row in rows if fits[int(row.event)].available]
    gate = comparison_gate([row.key for row in incumbent_rows], covered_keys, grain=grain)
    block["gate"] = gate
    if gate["status"] != STATUS_OK:
        block.update(
            {
                "status": STATUS_UNREACHABLE,
                "reason": "the challenger arm does not cover the identical comparison population",
                "exclusion_reasons": sorted(excluded_by_reason),
                "population": {
                    "population_digest": None,
                    "n": 0,
                    "rows_without_origin_transform": int(block["rows_without_origin_transform"]),
                    "excluded_by_reason": block["excluded_by_reason"],
                },
                "incumbent_on_comparison_population": None,
                "candidate": None,
            }
        )
        return block

    incumbent_pairs = [(row.probability, row.outcome) for row in incumbent_rows]
    candidate_pairs: list[tuple[float, float]] = []
    for row in covered:
        spec = fits[int(row.event)].spec
        if spec is None:  # pragma: no cover - guarded by ``covered`` above
            raise CalibrationEvaluationError("a covered row has no transform in force")
        candidate_pairs.append((spec.apply(row.probability), row.outcome))
    block.update(
        {
            "status": STATUS_OK,
            "reason": None,
            "exclusion_reasons": sorted(excluded_by_reason),
            "population": {
                "population_digest": gate["population_digest"],
                "n": len(covered),
                "rows_without_origin_transform": int(block["rows_without_origin_transform"]),
                "excluded_by_reason": block["excluded_by_reason"],
            },
            "incumbent_on_comparison_population": _figure(
                [p for p, _y in incumbent_pairs], [y for _p, y in incumbent_pairs]
            ),
            "candidate": _figure([p for p, _y in candidate_pairs], [y for _p, y in candidate_pairs]),
            "versions_in_force": sorted(
                {
                    str(fits[int(row.event)].spec.version)
                    for row in covered
                    if fits[int(row.event)].spec is not None
                }
            ),
        }
    )
    return block


# ---------------------------------------------------------------------------
# DefCon single-definition check
# ---------------------------------------------------------------------------


def defcon_calibration_block(
    rows: Sequence[sb.ProbabilityScoringRow],
) -> dict[str, Any]:
    """The DefCon calibration identity carried on every DefCon calibration figure.

    Every producer must use the same declared calibration, and a second,
    independently derived DefCon probability is a defect rather than a
    refinement.  So this block requires the rows that produced the figure to agree
    on ONE identity, requires that identity to resolve in the frozen registry, and
    records the boundary behaviour that makes a stated impossibility stay
    impossible.
    """

    identities: dict[str, set[str]] = {}
    missing = 0
    for row in rows:
        payload = row.defcon_calibration_payload
        if not payload:
            missing += 1
            continue
        spec = defcon_cal.from_payload(payload)
        identities.setdefault(spec.version, set()).add(spec.identity())
    if not identities:
        return {
            "status": STATUS_UNREACHABLE,
            "version": None,
            "identity": None,
            "carried_on_every_row": False,
            "rows_without_a_calibration": int(missing),
            "single_definition": False,
            "reason": (
                "no scored DefCon row carried its calibration, so the figure cannot be tied to the spec "
                "that produced it and is not presented as a calibrated figure"
            ),
        }
    if len(identities) > 1:
        raise CalibrationEvaluationError(
            "the scored DefCon rows disagree about which calibration produced them: "
            f"{sorted(identities)}; the single-definition rule requires exactly one"
        )
    version = next(iter(identities))
    identity = next(iter(identities[version]))
    spec = defcon_cal.resolve(version)
    if spec.identity() != identity:
        raise CalibrationEvaluationError(
            f"the DefCon calibration persisted on the rows ({identity}) is not the registered "
            f"calibration for {version!r} ({spec.identity()})"
        )
    return {
        "status": STATUS_OK if missing == 0 else "PARTIAL",
        "version": str(version),
        "identity": str(identity),
        "spec": spec.as_dict(),
        "carried_on_every_row": missing == 0,
        "rows_without_a_calibration": int(missing),
        "single_definition": True,
        "reason": (
            None
            if missing == 0
            else f"{missing} scored row(s) carried no calibration, so the figure is tied to one "
            "calibration but not to every row it covers"
        ),
        "monotonicity": spec.method,
        "boundaries": {
            "raw_zero_stays_zero": spec.apply(0.0) == 0.0,
            "raw_one_stays_one": spec.apply(1.0) == 1.0,
        },
        "detail": (
            "one versioned calibration shared by every producer; PE-8 does not re-fit or replace it, "
            "and resolves it from the payload rather than from a global default"
        ),
    }


# ---------------------------------------------------------------------------
# Probability surfaces
# ---------------------------------------------------------------------------


def _transform_mandate(diagnosis: Mapping[str, Any], challenger: Mapping[str, Any]) -> dict[str, Any]:
    """Whether a transform has a MANDATE, which only a diagnosed defect gives it.

    A transform exists to correct a measured miscalibration.  So the DIAGNOSIS
    comes first and decides whether a challenger is constructed at all: where no
    defect is diagnosed, the challenger block records that none was built, rather
    than reporting fitted parameters a reviewer might read as a candidate.
    """

    status = str(diagnosis.get("status"))
    if status == DIAGNOSIS_MISCALIBRATED:
        mandate = "DIAGNOSED_DEFECT"
        note = (
            "a material calibration gap is diagnosed, so a transform has a mandate; the challenger is "
            "fitted per origin from strictly earlier point-in-time evidence, and it is still not "
            "promoted, so the transform's own status says whether an admissible one exists"
        )
    elif status == DIAGNOSIS_INSUFFICIENT:
        mandate = "INSUFFICIENT_FOR_DIAGNOSIS"
        note = (
            "the sample cannot support a calibration claim on this surface, so no defect is diagnosed "
            "and no challenger was constructed"
        )
    else:
        mandate = "NO_DEFECT_DIAGNOSED"
        note = (
            "no material defect is diagnosed on this surface, so no challenger was constructed: "
            "fitting one would be a speculative transform with nothing to correct"
        )
    return {
        "mandate": mandate,
        "note": note,
        "promotable_by_pe8": False,
        "transform_status": str(challenger.get("status")),
        "rule": (
            "a transform is admissible only to correct a measured miscalibration on a declared "
            "surface; the surface is diagnosed first and PE-8 fits no challenger unless the diagnosis "
            "is MISCALIBRATED, and promotes none at all"
        ),
    }


def probability_calibration_block(
    conn: sqlite3.Connection,
    *,
    events: Sequence[int],
    xpts_runs: Mapping[int, int],
    prediction_rows: Mapping[int, Sequence[FrozenPredictionRow]],
    cutoffs: Mapping[int, str],
    excluded_events: Sequence[Mapping[str, Any]] = (),
    inapplicable_candidates: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Every declared probability surface: diagnosis first, then any transform.

    The order is the contract's rule 1 made structural.  The incumbent figure and
    its diagnosis are computed for every surface; a challenger is built ONLY where
    the diagnosis is MISCALIBRATED, and its basis is the frozen prediction rows of
    the anchor's own runs, never the current-state tables.

    ``events`` is the SCORED target event set -- the FINAL events -- and it is also
    the ORIGIN set, because a transform is only applied where a figure is scored.
    ``prediction_rows`` is the CANDIDATE pool and is deliberately a different set:
    every applicable certified anchor target event, so a target event that is not
    FINAL today still contributes the rows it was predicted under.  The two are
    required rather than defaulted to each other, because defaulting them together
    is exactly the conflation the phase forbids.
    """

    wanted = sorted({int(event) for event in events})
    population = sb.player_fixture_population(conn, events=wanted, xpts_runs=xpts_runs)
    scored = sb.probability_scoring_rows(population["rows"])
    by_metric: dict[str, list[sb.ProbabilityScoringRow]] = {
        definition.metric: [] for definition in sb.PROBABILITY_METRICS
    }
    for row in scored["rows"]:
        by_metric[row.metric].append(row)

    declared_cutoffs = dict(cutoffs)
    frozen = {int(event): list(rows) for event, rows in prediction_rows.items()}
    surfaces: list[dict[str, Any]] = []
    for definition in sb.PROBABILITY_METRICS:
        rows = by_metric[definition.metric]
        probabilities = [row.probability for row in rows]
        outcomes = [row.outcome for row in rows]
        figure = _figure(probabilities, outcomes)
        events_with_rows = sorted({int(row.event) for row in rows})
        keys = [row.key for row in rows]
        diagnosis = _diagnosis(
            figure["reliability"]["bins_over_floor"], figure["reliability"]["max_abs_gap_over_floor"]
        )
        if diagnosis["status"] == DIAGNOSIS_MISCALIBRATED and rows:
            challenger, _covered = _challenger_block(
                conn=conn,
                rows=rows,
                prediction_rows=frozen,
                cutoffs=declared_cutoffs,
                definition=definition,
                grain=wf.GRAIN_PLAYER_FIXTURE,
                origins=wanted,
            )
        else:
            challenger = _not_fitted_block(
                definition.metric, wf.GRAIN_PLAYER_FIXTURE, diagnosis
            )
        block: dict[str, Any] = {
            "metric": definition.metric,
            "definition": definition.as_dict(),
            "grain": wf.GRAIN_PLAYER_FIXTURE,
            "grain_note": (
                "the probability surfaces are persisted per player x fixture; a double gameweek is "
                "therefore two rows here and one row at the headline event grain"
            ),
            "population": {
                "scored": len(rows),
                "probability_absent": int(scored["absent"][definition.metric]),
                "outcome_unavailable": int(scored["outcome_unavailable"][definition.metric]),
                "population_digest": wf.canonical_population_digest(
                    keys, grain=wf.GRAIN_PLAYER_FIXTURE
                ),
                "candidates": int(population["candidates"]),
                "excluded": {
                    "fixture_not_played": int(population["excluded"]["fixture_not_played"]),
                    "outcome_row_missing": int(population["excluded"]["outcome_row_missing"]),
                },
                "excluded_events": [dict(entry) for entry in excluded_events],
                "run_ids_by_event": int_keyed_runs(xpts_runs),
                "versions": list(population["versions"]),
                "policy": sb.PROBABILITY_POPULATION_POLICY,
            },
            "incumbent": {
                "calibration": pc.INCUMBENT_IDENTITY.as_payload(),
                "figure": figure,
                "note": (
                    "the incumbent is the persisted probability itself, declared as an identity "
                    "transform rather than reached by a fallback"
                ),
            },
            "diagnosis": diagnosis,
            "sample": _sample_block(
                target_events=len(wanted),
                events_with_observations=len(events_with_rows),
                observations=len(rows),
            ),
            "causal_challenger": challenger,
            "transform_mandate": _transform_mandate(diagnosis, challenger),
            "risk_flags": sorted({flag for row in rows for flag in row.risk_flags}),
            "claim": PROBABILITY_SURFACE_CLAIM,
        }
        if definition.metric == "BRIER_DEFCON":
            block["defcon_calibration"] = defcon_calibration_block(rows)
            block["population"]["gkp_excluded_reason"] = (
                "defcon_threshold_for returns None for GKP, so a GKP row is excluded from the DefCon "
                "population and counted (under outcome_unavailable) rather than scored against a "
                "fabricated threshold"
            )
        surfaces.append(block)
    return {
        "grain": wf.GRAIN_PLAYER_FIXTURE,
        "population_policy": sb.PROBABILITY_POPULATION_POLICY,
        "population": {
            "candidates": int(population["candidates"]),
            "rows": len(population["rows"]),
            "excluded": dict(population["excluded"]),
            "run_ids_by_event": int_keyed_runs(xpts_runs),
            "versions": list(population["versions"]),
        },
        # The two populations a fit basis separates, named here once for the whole
        # block: what a transform was ALLOWED TO LEARN FROM (candidates, decided by
        # the certification) and where it is APPLIED (origins = the scored events,
        # which keep the current-state FINAL filter).
        "fit_basis_candidates": {
            "policy": FIT_BASIS_CANDIDATE_POLICY,
            "candidate_events": [int(event) for event in sorted(frozen)],
            "candidate_rows_by_event": {
                int(event): len(rows) for event, rows in sorted(frozen.items())
            },
            "origin_events": list(wanted),
            "origin_rule": (
                "an origin is a SCORED target event; a transform is only ever applied where a figure "
                "is scored, so an event that is not officially FINAL today is excluded and counted "
                "rather than fitted for"
            ),
            "events_without_a_certified_xpts_v1_run": [
                dict(entry) for entry in inapplicable_candidates
            ],
        },
        "surfaces": surfaces,
    }


def int_keyed_runs(runs: Mapping[int, int]) -> dict[int, int]:
    """``{event: run_id}`` with int keys, so JSON keys and the identity always agree."""

    return {int(event): int(run_id) for event, run_id in sorted(runs.items())}


# ---------------------------------------------------------------------------
# Expected value
# ---------------------------------------------------------------------------


def expected_value_block(
    conn: sqlite3.Connection,
    *,
    anchor: wf.CertifiedAnchor,
    events: Sequence[int],
    xpts_runs: Mapping[int, int],
) -> dict[str, Any]:
    """The headline event-grain expectation, plus every persisted component."""

    return {
        "headline": _headline_block(conn, anchor=anchor, events=events),
        "component_diagnostics": _component_block(conn, events=events, xpts_runs=xpts_runs),
        "bias_convention": COMPONENT_BIAS_CONVENTION,
        "grain_policy": (
            "the headline is player x event, where the source itself sums a player's fixtures; every "
            "component is reported at player x fixture, the grain at which it is persisted.  A "
            "component is never summed into an event figure here, because the components the payload "
            "persists are not the event-grain claim the headline makes"
        ),
        "aggregate_warning": (
            "an aggregate bias can hide a compensating pair of component biases, so the aggregate "
            "figure alone is not accepted as a PE-8 result"
        ),
    }


def _headline_block(
    conn: sqlite3.Connection, *, anchor: wf.CertifiedAnchor, events: Sequence[int]
) -> dict[str, Any]:
    """The persisted expected points against realised total points, player x event."""

    block: dict[str, Any] = {
        "grain": wf.GRAIN_PLAYER_EVENT,
        "grain_note": sb.TARGET_GRAIN_NOTE,
        "target": "payload total_xpts summed over the player's fixtures, against realised total_points",
        "status": STATUS_UNREACHABLE,
        "reason": None,
        "population_digest": None,
        "n": 0,
        "metrics": [],
        "population": {"evaluated": 0, "excluded_by_status": {}},
        "tolerance": MATERIAL_EV_BIAS_TOLERANCE_POINTS,
    }
    if not events:
        block["reason"] = "no target event is final and evaluable"
        return block
    try:
        population = wf.build_event_population(conn, anchor=anchor, events=list(events), baseline_kind=None)
    except wf.WalkForwardError as error:
        block["reason"] = str(error)
        return block
    keys: list[tuple[int, int]] = []
    predicted: list[float] = []
    actual: list[float] = []
    for row in population.rows:
        value = row.get("model_xpts")
        realised = (row.get("outcome") or {}).get("total_points")
        if value is None or realised is None:
            continue
        keys.append((int(row["event"]), int(row["player_id"])))
        predicted.append(float(value))
        actual.append(float(realised))
    block["population"] = {
        "candidates": population.coverage()["candidates"],
        "evaluated": population.coverage()["evaluated"],
        "excluded_by_status": population.coverage()["by_status"],
        "scored": len(keys),
        "evaluated_without_recorded_points": len(population.rows) - len(keys),
    }
    block["sample"] = _sample_block(
        target_events=len(events),
        events_with_observations=len({event for event, _player in keys}),
        observations=len(keys),
    )
    block["population_digest"] = wf.canonical_population_digest(
        keys, grain=wf.GRAIN_PLAYER_EVENT
    )
    block["n"] = len(keys)
    if not keys:
        block["reason"] = (
            "no player x event observation has both a model projection and realised points, so this "
            "figure carries no value rather than a zero"
        )
        return block
    block["status"] = STATUS_OK
    block["metrics"] = [
        {"metric": "MAE", "value": _metric_or_none(wm.mean_absolute_error, predicted, actual)},
        {"metric": "RMSE", "value": _metric_or_none(wm.root_mean_squared_error, predicted, actual)},
        {"metric": "BIAS", "value": _metric_or_none(wm.mean_bias, predicted, actual)},
        {"metric": "MEDIAN_AE", "value": _metric_or_none(wm.median_absolute_error, predicted, actual)},
    ]
    block["bias"] = _metric_or_none(wm.mean_bias, predicted, actual)
    return block


def _component_block(
    conn: sqlite3.Connection,
    *,
    events: Sequence[int],
    xpts_runs: Mapping[int, int],
) -> dict[str, Any]:
    """Expected-value bias per persisted component, at the grain it is persisted at."""

    wanted = sorted({int(event) for event in events})
    block: dict[str, Any] = {
        "grain": wf.GRAIN_PLAYER_FIXTURE,
        "grain_note": (
            "every component is persisted per player x fixture in the anchor xpts_v1 payload and is "
            "reported there; a double gameweek is two component observations"
        ),
        "bias_convention": COMPONENT_BIAS_CONVENTION,
        "population": {"rows": 0, "candidates": 0, "excluded": {}},
        "components": [],
    }
    if not wanted or not xpts_runs:
        return block
    population = sb.player_fixture_population(conn, events=wanted, xpts_runs=xpts_runs)
    block["population"] = {
        "rows": len(population["rows"]),
        "candidates": int(population["candidates"]),
        "excluded": dict(population["excluded"]),
    }
    predictions: dict[str, list[float]] = {definition.component: [] for definition in COMPONENT_DEFINITIONS}
    realised_values: dict[str, list[float]] = {definition.component: [] for definition in COMPONENT_DEFINITIONS}
    gaps: dict[str, int] = {definition.component: 0 for definition in COMPONENT_DEFINITIONS}
    absent: dict[str, int] = {definition.component: 0 for definition in COMPONENT_DEFINITIONS}
    flags: dict[str, set[str]] = {definition.component: set() for definition in COMPONENT_DEFINITIONS}
    events_seen: dict[str, set[int]] = {definition.component: set() for definition in COMPONENT_DEFINITIONS}
    structural_zeros: dict[str, dict[str, int]] = {
        definition.component: {} for definition in COMPONENT_DEFINITIONS
    }
    for row in population["rows"]:
        for definition in COMPONENT_DEFINITIONS:
            value = row.payload.get(definition.component)
            if value is None:
                absent[definition.component] += 1
                continue
            target = realised_component(definition.selector, row.outcome, row.position)
            if target is None:
                gaps[definition.component] += 1
                continue
            if target == 0.0 and structural_zero_rule(definition.selector, row.position) is not None:
                # A real zero from the scoring rules, not a missing observation: the
                # row is covered, and the count says which positions supplied it.
                counters = structural_zeros[definition.component]
                counters[str(row.position)] = counters.get(str(row.position), 0) + 1
            predictions[definition.component].append(float(value))
            realised_values[definition.component].append(float(target))
            flags[definition.component].update(row.risk_flags())
            events_seen[definition.component].add(int(row.event))
    for definition in COMPONENT_DEFINITIONS:
        predicted = predictions[definition.component]
        actual = realised_values[definition.component]
        observed_flags = sorted(flags[definition.component])
        disqualifying = NON_CAUSAL_FLAGS_BY_COMPONENT.get(
            definition.component, NON_CAUSAL_COMPONENT_FLAGS
        )
        non_causal = sorted(set(observed_flags) & disqualifying)
        entry: dict[str, Any] = definition.as_dict()
        entry.update(
            {
                "n": len(predicted),
                "predicted_absent": absent[definition.component],
                "realised_unavailable": gaps[definition.component],
                "structural_zero_rows": sum(structural_zeros[definition.component].values()),
                "structural_zero_by_position": {
                    key: int(value) for key, value in sorted(structural_zeros[definition.component].items())
                },
                "mae": _metric_or_none(wm.mean_absolute_error, predicted, actual),
                "bias": _metric_or_none(wm.mean_bias, predicted, actual),
                "median_ae": _metric_or_none(wm.median_absolute_error, predicted, actual),
                "risk_flags": observed_flags,
                "non_causal_flags": non_causal,
                "causally_interpretable": not non_causal,
                "tolerance": MATERIAL_COMPONENT_BIAS_TOLERANCE_POINTS,
                "sample": _sample_block(
                    target_events=len(wanted),
                    events_with_observations=len(events_seen[definition.component]),
                    observations=len(predicted),
                ),
                "flag_note": (
                    "a component carrying any of these flags is an approximation, a proxy, or is "
                    "flagged as such at production time: its bias is a descriptive number and is not "
                    "presented as a causal claim"
                ),
            }
        )
        block["components"].append(entry)
    return block


# ---------------------------------------------------------------------------
# Monte Carlo coverage
# ---------------------------------------------------------------------------


def monte_carlo_block(
    conn: sqlite3.Connection, *, events: Sequence[int], monte_carlo_runs: Mapping[int, int]
) -> dict[str, Any]:
    """Quantile coverage at fixture grain, with the claims it does NOT make.

    The coverage computation itself is the scoreboard's declared block, reused
    rather than restated: same policy, same intervals, same population counters.
    """

    coverage = sb.quantile_coverage(
        conn, events=list(events), monte_carlo_runs=dict(monte_carlo_runs)
    )
    observations = int(coverage["population"]["scored"])
    sufficient = (
        len(events) >= sb.MIN_TARGET_EVENTS_FOR_DESCRIPTIVE
        and observations >= sb.MIN_OBSERVATIONS_FOR_DESCRIPTIVE
    )
    return {
        "coverage": coverage,
        "grain": wf.GRAIN_PLAYER_FIXTURE,
        "policy": sb.QUANTILE_POLICY,
        "sample": {
            "target_events": len(events),
            "observations": observations,
            "sample_interpretation": (
                sb.SAMPLE_DESCRIPTIVE_ONLY if sufficient else sb.SAMPLE_INSUFFICIENT
            ),
            "policy_version": sb.SAMPLE_POLICY_VERSION,
            "note": (
                "the same declared disclosure floor every other PE-8 figure carries; the observation "
                "count is the scored fixture-grain coverage population, and it is what the floor binds on"
            ),
        },
        "claims": {
            "coverage_label": "QUANTILE COVERAGE, never a calibrated predictive interval",
            "not_full_distribution_calibration": (
                "a band that contains the realised value at the stated rate does not certify the "
                "distribution's shape, its tails, or its dependence structure"
            ),
            "crps": (
                "NOT CLAIMED; the persisted artifact stores quantiles, never draws, so there is no "
                "total-points distribution to score"
            ),
            "saturation": "quantiles are never summed into an event figure",
            "interval_kind": "closed: a realised value exactly equal to a bound counts as covered",
            "reversed_pair": (
                "a reversed quantile pair fails closed through the metric engine rather than having "
                "its bounds silently swapped"
            ),
        },
    }


# ---------------------------------------------------------------------------
# FPL assist mapping
# ---------------------------------------------------------------------------


def assist_mapping_block(
    conn: sqlite3.Connection,
    *,
    events: Sequence[int],
    prediction_rows: Mapping[int, Sequence[FrozenPredictionRow]],
    cutoffs: Mapping[int, str],
) -> dict[str, Any]:
    """The uncalibrated assist-mapping constant, as a decision PE-8 reports.

    The coefficient stays ``1.0`` and ``assist_mapping_calibrated`` stays
    ``False`` unless a causal walk-forward fit on finalised outcomes supports a
    different value and a review promotes it.  A fit that IS supported is
    reported as a candidate; PE-8 does not re-point the constant, and it never
    silences the truthful ``FPL_ASSIST_MAPPING_UNCALIBRATED`` flag.

    Each origin's fit basis is built exactly as the probability surfaces' is, and
    the same separation applies: ``prediction_rows`` is the CANDIDATE pool (every
    applicable certified anchor target event, decided by the certification), while
    ``events`` is the SCORED/origin set (the FINAL target events, which keep the
    current-state filter).  So a target event that has since lost its current FINAL
    state still contributes the frozen ``expected_xa`` rows it was predicted under
    to every later origin's basis, and it leaves the SCORED population and the
    origin set -- which is a scored-event exclusion, not a basis edit.  The
    realised side is the official assists column OF THE PE-5 CAPTURE that was
    final and captured strictly before that origin's certified cutoff.  A row whose
    timing cannot be proven is excluded and counted.

    ``pooled_evidence`` is the ADMISSIBLE evidence, and nothing else: every row
    that entered at least one origin's basis, counted ONCE (at the value admitted
    at the latest origin that could see it), with the refusals summed by reason
    beside it.  The declared descriptive floors are tested against that pool, so a
    late capture, a provisional read, an absent observation or a correction that
    arrived after every cutoff cannot make the evidence look larger than what a
    fit was allowed to learn from, and a row's CURRENT table value is never
    substituted for the value the capture stated.
    """

    config = xpts_module.XPtsConfig()
    declared_cutoffs = dict(cutoffs)
    wanted = sorted({int(event) for event in events})
    block: dict[str, Any] = {
        "surface": ASSIST_MAPPING_SURFACE,
        "grain": ASSIST_MAPPING_GRAIN,
        "incumbent": {
            "field": "XPtsConfig.fpl_assist_mapping_coefficient",
            "coefficient": float(config.fpl_assist_mapping_coefficient),
            "assist_mapping_calibrated": bool(config.assist_mapping_calibrated),
            "production_flag": ASSIST_MAPPING_FLAG,
            "flag_is_emitted": not bool(config.assist_mapping_calibrated),
            "location": "fpl_brain/xpts.py",
        },
        "method": pc.ASSIST_MAPPING_FIT_METHOD,
        "policy_version": pc.CAUSAL_FIT_POLICY_VERSION,
        "basis_policy": BASIS_MEMBERSHIP_POLICY,
        "basis_evidence": {
            "ledger_version": ledger.OUTCOME_LEDGER_VERSION,
            "capture_version": ledger.OBSERVATION_CAPTURE_VERSION,
            "supersession_policy_version": ledger.SUPERSESSION_POLICY_VERSION,
            "grain": ledger.GRAIN_PLAYER_FIXTURE,
            "membership": (
                "frozen prediction rows of the strictly earlier candidates -- every applicable "
                "certified anchor target event of the anchor xpts_v1 runs"
            ),
            "candidate_policy": FIT_BASIS_CANDIDATE_POLICY,
            "candidate_events": [int(event) for event in sorted(prediction_rows)],
            "origin_events": list(wanted),
            "expected_side": "the persisted expected_xa of the frozen prediction row",
            "realised_side": "the official assists column OF THE CAPTURE, never the current table's",
            "strictly_before_cutoff": True,
            "exclusion_precedence": BASIS_EXCLUSION_PRECEDENCE,
            "exclusion_reasons": list(BASIS_EXCLUSION_REASONS),
        },
        "origins": [],
        "pooled_evidence": _pooled_evidence({}, {}, candidate_rows=0),
    }
    by_event: dict[int, list[FrozenPredictionRow]] = {
        int(event): list(rows) for event, rows in sorted(prediction_rows.items()) if rows
    }
    if not wanted or not by_event:
        block["decision"] = assist_mapping_decision(block["origins"], block["pooled_evidence"])
        return block

    captures: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    for capture in ledger.observation_captures(
        conn, grain=ledger.GRAIN_PLAYER_FIXTURE, events=sorted(by_event)
    ):
        if capture.get("fixture_id") is None:
            continue
        captures.setdefault(
            (int(capture["event"]), int(capture["player_id"]), int(capture["fixture_id"])), []
        ).append(capture)

    # One entry per admissible ROW, keyed by its own identity.  Origins are walked
    # in ascending order and a later admission overwrites an earlier one, so a row
    # that appears in several bases is counted once, at the value the latest origin
    # that could see it admitted.
    pooled: dict[tuple[int, int, int], tuple[float, float]] = {}
    excluded_total: dict[str, int] = {}
    candidate_rows = 0
    # The origins are the SCORED events, not the candidates: the candidate pool of
    # a target event that has since lost its current FINAL state is still read
    # here (it is a source of rows), but it is no longer an origin a transform is
    # applied at.
    for origin in wanted:
        cutoff = _require_cutoff(declared_cutoffs, origin)
        basis_rows: list[pc.AssistMappingObservation] = []
        excluded: dict[str, int] = {}
        for event in (candidate for candidate in sorted(by_event) if int(candidate) < int(origin)):
            for row in by_event[event]:
                candidate_rows += 1
                expected = row.payload.get("expected_xa")
                if expected is None:
                    _count_exclusion(excluded, BASIS_PREDICTION_FIELD_ABSENT)
                    continue
                realised, reason = _point_in_time_assists(
                    captures.get(row.key, ()), cutoff=cutoff
                )
                if reason is not None:
                    _count_exclusion(excluded, reason)
                    continue
                basis_rows.append(
                    pc.AssistMappingObservation(
                        event=int(event),
                        key=(int(row.player_id), int(row.fixture_id)),
                        expected_assists=float(expected),
                        realised_assists=float(realised),
                    )
                )
        for observation in basis_rows:
            pooled[(int(observation.event), int(observation.key[0]), int(observation.key[1]))] = (
                float(observation.expected_assists),
                float(observation.realised_assists),
            )
        for reason, count in excluded.items():
            excluded_total[reason] = excluded_total.get(reason, 0) + int(count)
        fit = pc.fit_assist_mapping_causal(
            basis_rows, origin_event=int(origin), surface=ASSIST_MAPPING_SURFACE, grain=ASSIST_MAPPING_GRAIN
        )
        entry: dict[str, Any] = {
            "origin_event": int(origin),
            "cutoff": cutoff,
            "rows_at_origin": len(by_event.get(int(origin), ())),
            "rows_admitted_at_origin": len(basis_rows),
            "basis_excluded_by_reason": {key: int(value) for key, value in sorted(excluded.items())},
            "fit": fit.as_dict(),
        }
        if fit.available:
            # Read the fitted payload back through the fail-closed reader before it
            # is reported: a coefficient that cannot state its identity and the
            # policy it was fitted under is not evidence of anything.
            entry["provenance"] = pc.assist_mapping_from_payload(fit.as_payload())
        block["origins"].append(entry)

    block["pooled_evidence"] = _pooled_evidence(pooled, excluded_total, candidate_rows=candidate_rows)
    block["decision"] = assist_mapping_decision(block["origins"], block["pooled_evidence"])
    return block


def _pooled_evidence(
    pooled: Mapping[tuple[int, int, int], tuple[float, float]],
    excluded: Mapping[str, int],
    *,
    candidate_rows: int,
) -> dict[str, Any]:
    """The pooled ADMISSIBLE evidence, with the refusals beside it.

    ``observations`` and ``events`` are what the declared descriptive floors bind
    on: rows that actually entered a fit basis, counted once each.  A row that no
    origin was allowed to learn from -- a late, provisional, absent or
    post-cutoff-corrected capture -- is not in the pool at all; it appears in
    ``excluded_by_reason`` instead, so the size of what could not be used is
    visible rather than missing.
    """

    rows = [pooled[key] for key in sorted(pooled)]
    observations = len(rows)
    events = len({int(key[0]) for key in pooled})
    sufficient = (
        events >= sb.MIN_TARGET_EVENTS_FOR_DESCRIPTIVE
        and observations >= sb.MIN_OBSERVATIONS_FOR_DESCRIPTIVE
    )
    return {
        "policy_version": pc.CAUSAL_FIT_POLICY_VERSION,
        "basis": (
            "the union of the origins' admissible fit bases: rows whose PE-5 point-in-time capture was "
            "FINAL and captured strictly before an origin's own certified cutoff, deduplicated by row "
            "and counted once, at the value admitted at the latest origin that could see it.  The "
            "current tables' outcome columns are never read, and a row no origin could learn from is "
            "excluded rather than pooled"
        ),
        "observations": int(observations),
        "events": int(events),
        "candidate_rows_examined": int(candidate_rows),
        "excluded_by_reason": {str(key): int(value) for key, value in sorted(excluded.items())},
        "expected_assists_total": round(sum(expected for expected, _realised in rows), 6),
        "realised_assists_total": round(sum(realised for _expected, realised in rows), 6),
        "sample_interpretation": sb.SAMPLE_DESCRIPTIVE_ONLY if sufficient else sb.SAMPLE_INSUFFICIENT,
        "floors": {
            "policy_version": sb.SAMPLE_POLICY_VERSION,
            "min_target_events_for_descriptive_reporting": sb.MIN_TARGET_EVENTS_FOR_DESCRIPTIVE,
            "min_observations_for_descriptive_reporting": sb.MIN_OBSERVATIONS_FOR_DESCRIPTIVE,
            "basis": (
                "the PE-2 disclosure floor, NOT a significance threshold; it is tested against the "
                "ADMISSIBLE evidence only, so evidence no transform could use cannot meet it"
            ),
        },
    }


def _point_in_time_assists(
    captures: Sequence[Mapping[str, Any]],
    *,
    cutoff: str,
) -> tuple[int | None, str | None]:
    """``(realised assists, exclusion reason)`` for one frozen prediction row.

    The realised side is the official assists column OF THE CAPTURE, so a later
    official refresh is a later capture and cannot rewrite what an earlier fit was
    fitted on.  The current ``player_gameweeks`` column is never consulted: it is
    the newest state of the row, not the state that was known at the cutoff.
    """

    visible = [
        capture
        for capture in captures
        if capture.get("captured_at") is not None and str(capture["captured_at"]) < str(cutoff)
    ]
    if not visible:
        return None, _unavailable_evidence_reason(captures, cutoff=cutoff)
    chosen = ledger.select_capture(visible)
    if chosen is None:  # pragma: no cover - ``visible`` is non-empty
        return None, BASIS_CAPTURE_ABSENT
    if str(chosen.get("observation_state")) != ledger.OBSERVATION_FINAL:
        return None, BASIS_PROVISIONAL
    final_at = str(chosen.get("official_final_at") or "").strip()
    if not final_at:
        return None, BASIS_FINALITY_UNPROVABLE
    if final_at >= str(cutoff):
        return None, BASIS_FINALITY_NOT_BEFORE_CUTOFF
    payload = chosen.get("payload") or {}
    assists = payload.get("assists")
    if assists is None:
        return None, BASIS_OUTCOME_UNAVAILABLE
    return int(assists), None


def assist_mapping_decision(
    origins: Sequence[Mapping[str, Any]], pooled: Mapping[str, Any]
) -> dict[str, Any]:
    """The declared a priori decision for the assist-mapping constant.

    ``NO_CHANGE`` is an expected outcome, not a failure: the coefficient stays
    ``1.0`` and the truthful flag keeps being emitted.  ``CANDIDATE_FOR_REVIEW``
    says a causal fit exists AND the admissible evidence meets the declared
    descriptive floors -- it still does not move the constant, because a promotion
    changes production behaviour and belongs to review.

    Both inputs are ADMISSIBLE evidence: ``pooled`` counts only rows that entered a
    real fit basis, so a late, provisional, absent or later-corrected capture can
    neither meet nor inflate a floor.  A fit counts as admissible only after its
    PROVENANCE is read back through
    :func:`probability_calibration.assist_mapping_from_payload`: a coefficient
    whose payload is missing its identity or the policy it was fitted under, or
    whose content disagrees with the identity it records, is not evidence and
    stops the decision rather than contributing a candidate.
    """

    observations = int(pooled.get("observations") or 0)
    events = int(pooled.get("events") or 0)
    sufficient = (
        events >= sb.MIN_TARGET_EVENTS_FOR_DESCRIPTIVE
        and observations >= sb.MIN_OBSERVATIONS_FOR_DESCRIPTIVE
    )
    admissible: list[dict[str, Any]] = []
    for entry in origins:
        fit = entry.get("fit") or {}
        if not fit.get("available"):
            continue
        provenance = pc.assist_mapping_from_payload(
            entry.get("provenance") or fit.get("provenance")
        )
        if bool(fit.get("in_bounds")):
            admissible.append(provenance)
    evidence = {
        "observations": int(observations),
        "events": int(events),
        "basis": (
            "admissible point-in-time evidence only: rows that entered an origin's fit basis, counted "
            "once; the current tables' outcome columns and any capture no cutoff admitted are not counted"
        ),
        "candidate_rows_examined": int(pooled.get("candidate_rows_examined") or 0),
        "excluded_by_reason": {
            str(key): int(value)
            for key, value in sorted((pooled.get("excluded_by_reason") or {}).items())
        },
        "sample_interpretation": str(pooled.get("sample_interpretation") or ""),
    }
    reasons: list[str] = []
    if not sufficient:
        reasons.append(
            "the available causal evidence is below the declared descriptive floors "
            f"({events} event(s), {observations} observation(s) that an origin's fit could actually "
            "learn from), so it cannot support a promotion"
        )
    if not admissible:
        reasons.append(
            "no origin produced an admissible causal fit of the assist mapping, so nothing supports a "
            "different value"
        )
    if sufficient and admissible:
        return {
            "outcome": ASSIST_MAPPING_CANDIDATE,
            "promotion_performed": False,
            "coefficient_after": 1.0,
            "assist_mapping_calibrated_after": False,
            "flag_after": ASSIST_MAPPING_FLAG,
            "reasons": [
                "a causal fit within the declared bounds exists and the evidence meets the declared "
                "floors; the production constant is NOT re-pointed by PE-8 and a promotion requires "
                "senior review"
            ],
            "candidate_coefficients": sorted(
                {float(provenance["coefficient"]) for provenance in admissible}
            ),
            "evidence": evidence,
            "floors": {
                "min_target_events": sb.MIN_TARGET_EVENTS_FOR_DESCRIPTIVE,
                "min_observations": sb.MIN_OBSERVATIONS_FOR_DESCRIPTIVE,
            },
        }
    return {
        "outcome": ASSIST_MAPPING_NO_CHANGE,
        "promotion_performed": False,
        "coefficient_after": 1.0,
        "assist_mapping_calibrated_after": False,
        "flag_after": ASSIST_MAPPING_FLAG,
        "reasons": reasons,
        "candidate_coefficients": [],
        "evidence": evidence,
        "floors": {
            "min_target_events": sb.MIN_TARGET_EVENTS_FOR_DESCRIPTIVE,
            "min_observations": sb.MIN_OBSERVATIONS_FOR_DESCRIPTIVE,
        },
    }


# ---------------------------------------------------------------------------
# Frozen identities
# ---------------------------------------------------------------------------


def frozen_incumbents_block() -> dict[str, Any]:
    """The incumbent identities PE-8 must not rewrite, read from the live modules."""

    declared = {
        "XPTS_MODEL_VERSION": ("xpts_v1.4.1", xpts_module.XPTS_MODEL_VERSION, "fpl_brain/xpts.py"),
        "MONTE_CARLO_MODEL_VERSION": ("mc_v1.3.0", monte_carlo.MONTE_CARLO_MODEL_VERSION, "fpl_brain/monte_carlo.py"),
        "MINUTES_MODEL_VERSION": ("minutes_v1.8.0", minutes_model.MINUTES_MODEL_VERSION, "fpl_brain/minutes_model.py"),
        "MINUTES_COHERENT_MODEL_VERSION": (
            "minutes_v1.2.0",
            minutes_model.MINUTES_COHERENT_MODEL_VERSION,
            "fpl_brain/minutes_model.py",
        ),
        "JOINT_MINUTES_MODEL_VERSION": (
            "minutes_v1.5.2",
            joint_minutes.JOINT_MINUTES_MODEL_VERSION,
            "fpl_brain/joint_minutes.py",
        ),
        "TEAM_MODEL_VERSION": ("team_strength_v1.1.0", team_model.TEAM_MODEL_VERSION, "fpl_brain/team_model.py"),
        "PLAYER_RATE_MODEL_VERSION": (
            "player_rates_v1.0.0",
            player_rates.PLAYER_RATE_MODEL_VERSION,
            "fpl_brain/player_rates.py",
        ),
        "DEFCON_CALIBRATION_VERSION": (
            "defcon_platt_v1.0.0",
            defcon_cal.DEFCON_CALIBRATION_VERSION,
            "fpl_brain/defcon_calibration.py",
        ),
    }
    entries = {
        name: {"expected": expected, "actual": str(actual), "location": location, "unchanged": str(actual) == expected}
        for name, (expected, actual, location) in declared.items()
    }
    mismatches = sorted(name for name, entry in entries.items() if not entry["unchanged"])
    return {
        "entries": entries,
        "unchanged": not mismatches,
        "mismatches": mismatches,
        "pe6_minutes_trio": ["MINUTES_MODEL_VERSION", "MINUTES_COHERENT_MODEL_VERSION", "JOINT_MINUTES_MODEL_VERSION"],
        "pe7_attack_models": ["TEAM_MODEL_VERSION", "PLAYER_RATE_MODEL_VERSION"],
        "promotion_target": (
            "PE-8 may fit a probability calibration and reports the assist-mapping decision; it may not "
            "bump an incumbent model version, replace the DefCon calibration, or wire a transform into "
            "a production decision path"
        ),
    }


# ---------------------------------------------------------------------------
# Terminal state
# ---------------------------------------------------------------------------


def terminal_state(artifact: Mapping[str, Any]) -> dict[str, Any]:
    """PE-8's declared terminal state, derived from the evidence it carries.

    ``OPEN`` is not a failure.  It is the correct state when the diagnosis is
    supported but the evidence does not support a promotion, or when the sample
    can support nothing beyond a descriptive report.
    """

    reasons: list[str] = []
    surfaces = artifact["probability_calibration"]["surfaces"]
    for surface in surfaces:
        interpretation = surface["sample"]["sample_interpretation"]
        if interpretation != sb.SAMPLE_DESCRIPTIVE_ONLY:
            reasons.append(
                f"{surface['metric']}: {interpretation} at {surface['sample']['observations']} "
                "observation(s) over "
                f"{surface['sample']['target_events_with_observations']} event(s)"
            )
        if surface["diagnosis"]["status"] == DIAGNOSIS_MISCALIBRATED:
            status = surface["causal_challenger"]["status"]
            reasons.append(
                f"{surface['metric']}: a material calibration gap of "
                f"{surface['diagnosis']['max_abs_gap_over_floor']} exceeds the declared tolerance "
                f"{MATERIAL_CALIBRATION_GAP_TOLERANCE}; the causal transform is {status}"
            )
        defcon = surface.get("defcon_calibration")
        if defcon and defcon.get("status") != STATUS_OK:
            reasons.append(
                f"{surface['metric']}: the DefCon calibration identity is {defcon.get('status')} "
                f"({defcon.get('reason')})"
            )
    if not artifact["probability_calibration"]["surfaces"]:
        reasons.append("no declared probability surface could be scored at all")

    headline = artifact["expected_value"]["headline"]
    if headline["status"] != STATUS_OK:
        reasons.append(f"the headline expected-value figure is {headline['status']}: {headline['reason']}")
    else:
        if headline["sample"]["sample_interpretation"] != sb.SAMPLE_DESCRIPTIVE_ONLY:
            reasons.append(
                "the headline expected-value figure is "
                f"{headline['sample']['sample_interpretation']} at "
                f"{headline['sample']['observations']} event-grain observation(s) over "
                f"{headline['sample']['target_events_with_observations']} event(s)"
            )
        if headline.get("bias", {}).get("value") is not None:
            if abs(float(headline["bias"]["value"])) > MATERIAL_EV_BIAS_TOLERANCE_POINTS:
                reasons.append(
                    f"the headline expected-value bias {headline['bias']['value']} exceeds the declared "
                    f"tolerance {MATERIAL_EV_BIAS_TOLERANCE_POINTS} points"
                )
    material_components = [
        entry["component"]
        for entry in artifact["expected_value"]["component_diagnostics"]["components"]
        if entry["bias"].get("value") is not None
        and abs(float(entry["bias"]["value"])) > MATERIAL_COMPONENT_BIAS_TOLERANCE_POINTS
    ]
    if material_components:
        reasons.append(
            "component bias beyond the declared tolerance "
            f"({MATERIAL_COMPONENT_BIAS_TOLERANCE_POINTS} points) on {sorted(material_components)}; no "
            "probability transform addresses an expected-value defect"
        )
    if not any(
        entry["n"] > 0
        for entry in artifact["expected_value"]["component_diagnostics"]["components"]
    ):
        reasons.append(
            "no expected-value component could be scored, so the per-component diagnosis the phase "
            "requires is unavailable"
        )

    decision = artifact["assist_mapping"]["decision"]
    if decision["outcome"] == ASSIST_MAPPING_CANDIDATE:
        reasons.append(
            "the assist mapping has a causal fit meeting the declared floors; the constant stays 1.0 "
            "and the promotion belongs to review"
        )

    state = TERMINAL_READY_FOR_MERGE if not reasons else TERMINAL_OPEN
    return {
        "state": state,
        "reasons": reasons,
        "promotion_performed": False,
        "incumbent_authoritative": True,
        "terminal_boundary": (
            "PE-8 ends at senior review; the state below is derived from the evidence this artifact "
            "carries and is one of READY_FOR_MERGE or OPEN"
        ),
        "open_rule": (
            "OPEN when the available evidence can support a diagnosis but not a promotion, or when the "
            "sample cannot support more than a descriptive report"
        ),
    }


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def evaluate(
    conn: sqlite3.Connection,
    *,
    artifact: Mapping[str, Any],
    events: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Build the PE-8 calibration evidence artifact for the certified anchor."""

    anchor = wf.discover_certified_anchor(conn, artifact, events=events)
    target_events = anchor.event_ids
    # SCORED eligibility, and nothing else: which target events are officially
    # FINAL TODAY.  This is PE-2's declared current-state population, and it
    # decides the scored figures and the origin set a transform may be applied at.
    final_events, excluded_events = _final_events(conn, target_events)
    xpts_runs = _xpts_runs(anchor, final_events)
    monte_carlo_runs = _ancillary_runs(anchor, final_events, "monte_carlo_v1")
    # Each origin's own certified cutoff, from its own certified bundle: a basis
    # that reached past its origin's cutoff would describe a world in which the
    # target was already known.
    cutoffs = {int(event): anchor.for_event(int(event)).cutoff for event in final_events}
    # FIT-BASIS CANDIDACY is a DIFFERENT question, with a different filter and a
    # different answer: every APPLICABLE certified anchor target event, decided by
    # the CERTIFICATION rather than by today's events row.  The FROZEN half of every
    # fit basis is therefore read from the immutable projection rows of every such
    # event -- even one that is not FINAL today -- so a current-state finality edit
    # can move a scored figure and an origin, and can never add a row to or remove a
    # row from an earlier transform's basis.  Membership of that basis is then
    # decided by the frozen prediction and the PE-5 rules alone.
    candidate_events, candidate_runs, inapplicable_events = fit_basis_candidates(anchor)
    prediction_rows = frozen_prediction_rows(
        conn, runs=candidate_runs, events=candidate_events
    )

    probability_block = probability_calibration_block(
        conn,
        events=final_events,
        xpts_runs=xpts_runs,
        prediction_rows=prediction_rows,
        cutoffs=cutoffs,
        excluded_events=excluded_events,
        inapplicable_candidates=inapplicable_events,
    )
    expected_value = expected_value_block(
        conn, anchor=anchor, events=final_events, xpts_runs=xpts_runs
    )
    coverage = monte_carlo_block(conn, events=final_events, monte_carlo_runs=monte_carlo_runs)
    assist = assist_mapping_block(
        conn,
        events=final_events,
        prediction_rows=prediction_rows,
        cutoffs=cutoffs,
    )

    body: dict[str, Any] = {
        "schema": PE8_SCHEMA_VERSION,
        "phase": "PE-8",
        "evaluation_version": PE8_EVALUATION_VERSION,
        "identity": {
            "evaluation_version": PE8_EVALUATION_VERSION,
            "schema_version": PE8_SCHEMA_VERSION,
            "walk_forward_identity": wf.WALK_FORWARD_VERSION,
            "missing_data_policy_version": wf.MISSING_DATA_POLICY_VERSION,
            "metric_policy_version": wm.METRIC_POLICY_VERSION,
            "sample_policy_version": sb.SAMPLE_POLICY_VERSION,
            "calibration_policy": pc.policy(),
            "certification_identity": anchor.certification_identity,
            "planning_cutoff": anchor.planning_cutoff,
            "target_events": [int(event) for event in target_events],
            "events_evaluated": [int(event) for event in final_events],
            "events_excluded": [dict(entry) for entry in excluded_events],
            "fit_basis_candidate_events": [int(event) for event in candidate_events],
            "fit_basis_events_without_a_certified_xpts_v1_run": [
                dict(entry) for entry in inapplicable_events
            ],
            "per_event_cutoffs": {int(event): cutoffs[int(event)] for event in final_events},
            "point_in_time_evidence": {
                "ledger_version": ledger.OUTCOME_LEDGER_VERSION,
                "capture_version": ledger.OBSERVATION_CAPTURE_VERSION,
                "supersession_policy_version": ledger.SUPERSESSION_POLICY_VERSION,
                "grain": ledger.GRAIN_PLAYER_FIXTURE,
                "basis_rule": BASIS_MEMBERSHIP_POLICY,
                "basis_candidate_rule": FIT_BASIS_CANDIDATE_POLICY,
                "basis_membership": (
                    "the frozen prediction rows of the anchor's own xpts_v1 runs read from "
                    "player_fixture_xpts_projections, which refuses UPDATE and DELETE at storage level"
                ),
                "basis_candidates": (
                    "every APPLICABLE certified anchor target event, decided by the CERTIFICATION (the "
                    "bundle's own xpts_v1 run) and never by today's events row; events_evaluated is the "
                    "SCORED and ORIGIN side of the same separation and keeps the current-state FINAL "
                    "filter, so a target event that is not final today leaves the scored figures and "
                    "the origin set without leaving -- or entering -- any earlier basis"
                ),
                "basis_position": (
                    "the position persisted with each prediction row; the current players row is not "
                    "consulted to decide what an earlier transform was fitted on"
                ),
                "scored_population": (
                    "a different role, unchanged: the SCORED figures cover PE-2's declared "
                    "current-state population (played fixtures, a realised row that exists now and is "
                    "not a scheduled placeholder, the current position), which is what makes them "
                    "reproducible against the scoreboard"
                ),
                "exclusion_precedence": BASIS_EXCLUSION_PRECEDENCE,
                "exclusion_reasons": list(BASIS_EXCLUSION_REASONS),
            },
            "per_event_runs": {
                str(event): {
                    "xpts_v1": xpts_runs.get(int(event)),
                    "monte_carlo_v1": monte_carlo_runs.get(int(event)),
                }
                for event in final_events
            },
            "grains": {
                "probability_surfaces": wf.GRAIN_PLAYER_FIXTURE,
                "expected_value_headline": wf.GRAIN_PLAYER_EVENT,
                "expected_value_components": wf.GRAIN_PLAYER_FIXTURE,
                "quantile_coverage": wf.GRAIN_PLAYER_FIXTURE,
                "grain_note": (
                    "a fixture-grain probability figure and an event-grain points figure are not "
                    "interchangeable: in a double gameweek they cover different populations"
                ),
            },
            "event_finality_predicate": "planning.event_data_state == FINAL (events.finished=1 and data_checked=1)",
            "bias_convention": COMPONENT_BIAS_CONVENTION,
            "tolerances": {
                "material_calibration_gap": MATERIAL_CALIBRATION_GAP_TOLERANCE,
                "material_ev_bias_points": MATERIAL_EV_BIAS_TOLERANCE_POINTS,
                "material_component_bias_points": MATERIAL_COMPONENT_BIAS_TOLERANCE_POINTS,
            },
            "frozen_incumbents": frozen_incumbents_block(),
        },
        "probability_calibration": probability_block,
        "expected_value": expected_value,
        "monte_carlo_coverage": coverage,
        "assist_mapping": assist,
        "excluded_surfaces": [dict(entry) for entry in EXCLUDED_SURFACES],
        "claims": claims_block(),
        "limitations": limitations_block(),
    }
    body["terminal_state"] = terminal_state(body)
    return body


def claims_block() -> dict[str, Any]:
    """What PE-8 claims, and the claims it explicitly does not make."""

    return {
        "ranking": "NONE; PE-8 measures, it never ranks arms, names a winner or derives a skill score",
        "selection": "NONE; nothing here selects a model, a transform or a policy",
        "calibrated_predictive_interval": (
            "NOT CLAIMED; the Monte Carlo figure is quantile COVERAGE, and coverage is not "
            "full-distribution calibration"
        ),
        "crps": (
            "NOT CLAIMED; the persisted artifact stores quantiles, never draws, and there is no "
            "total-points quantile distribution"
        ),
        "probability_outside_unit_interval": "fails closed; the evaluation stops rather than clipping",
        "missing_probability": "excluded and counted; never scored as 0.0",
        "empty_sample": "rendered as null; a sample of zero is not a score of zero",
        "promotion": (
            "NOT PERFORMED; every incumbent stays authoritative, no model version is bumped and no "
            "calibration version is replaced"
        ),
        "production_wiring": (
            "NONE; no transform fitted here is wired into a production decision path, because ranking "
            "behaviour and expected-value behaviour differ and unvalidated evidence cannot decide either"
        ),
        "continuous_proxy_tie_limitation": (
            "CARRIED, NOT RESOLVED; the PE-3 CONTINUOUS_PROXY_TIE_LIMITATION remains an open accepted "
            "limitation, and no figure here is presented as if it did not apply"
        ),
        "in_sample_calibration": (
            "NOT PERFORMED; every transform is scored at an origin strictly after its own fit basis"
        ),
        "fit_basis": (
            "PE-5 append-only point-in-time observation captures over the FROZEN prediction rows of the "
            "strictly earlier CANDIDATES -- every applicable certified anchor target event, decided by "
            "the certification and never by today's events row -- with the position persisted with each "
            "prediction; a row whose official finality or capture timing cannot be proven strictly "
            "before the origin's certified cutoff is excluded and counted, a later correction cannot "
            "rewrite an earlier transform, and no current events / fixtures / player_gameweeks / players "
            "state decides historical basis membership"
        ),
        "scored_population": (
            "PE-2's declared current-state population, unchanged: the scored figures cover the FINAL "
            "target events and their played fixtures with a realised row that exists now and is not a "
            "scheduled placeholder, which is what makes them reproducible against the scoreboard.  The "
            "fit basis is a different role and is named separately wherever both appear: a current "
            "finality edit moves the scored exclusion and the origin set, and moves no basis"
        ),
        "assist_evidence": (
            "the assist-mapping pooled evidence and its declared floors rest on ADMISSIBLE point-in-time "
            "evidence only -- rows that entered a real fit basis, counted once -- so a late, "
            "provisional, absent or later-corrected capture cannot meet or inflate a floor, and its "
            "refusals are reported by reason"
        ),
        "challenger_construction": (
            "a challenger is constructed ONLY where the surface's diagnosis is MISCALIBRATED; where no "
            "defect is diagnosed, no transform is fitted and the block says so"
        ),
        "fitted_provenance": (
            "every fitted payload carries its canonical identity and the applicable policy versions, and "
            "a payload whose identity or policy is absent or mismatched fails closed rather than "
            "producing an unattributable number"
        ),
    }


def limitations_block() -> list[str]:
    return [
        "Quantile coverage is reported as COVERAGE, never as a calibrated predictive interval, and "
        "CRPS is not claimed because no draw-level artifact is persisted.",
        "The probability surfaces are persisted per player x fixture; the headline expected-value "
        "figure is player x event.  They cover different populations in a double gameweek and are not "
        "interchangeable.",
        "An expected-value component whose rows carry a non-causal production flag is reported with "
        "that flag: its bias is a description, not a causal claim.",
        "A causally fitted transform is evidence for review, not a production change: PE-8 reports it "
        "and does not re-point anything.",
        "A fit basis is built from the FROZEN prediction rows of the strictly earlier candidates -- "
        "every applicable certified anchor target event, decided by the certification rather than by "
        "today's events row -- plus PE-5 append-only point-in-time captures that were official and "
        "recorded strictly before the origin's own certified cutoff, at the position the prediction was "
        "made at; a postponed result, a late capture or an unproven timing is excluded and counted "
        "rather than assumed known, and no change to the current events / fixtures / player_gameweeks / "
        "players tables can move an earlier transform.  Scored eligibility keeps the current-state FINAL "
        "filter, so a current finality edit moves the scored figures and the origin set and no basis.",
        "A position outside a component's declared positions earns a real zero from that component, "
        "returned before the component's own outcome column is consulted, and the row is covered (the "
        "count is reported); it is not a gap in the evidence.",
        "The assist-mapping pooled evidence and its declared descriptive floors count admissible "
        "point-in-time evidence only, deduplicated by row; evidence no transform could learn from is "
        "reported as an exclusion rather than pooled.",
        "CONTINUOUS_PROXY_TIE_LIMITATION from PE-3 remains an open, accepted limitation.",
        "Models, baselines and outcomes are read from persisted runs; nothing here regenerates a "
        "prediction, writes a database row, or changes any incumbent identity.",
    ]


# ---------------------------------------------------------------------------
# Canonical bytes
# ---------------------------------------------------------------------------


def canonical_bytes(artifact: Mapping[str, Any]) -> bytes:
    return json.dumps(
        artifact, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")


def artifact_digest(artifact: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(canonical_bytes(artifact)).hexdigest()
