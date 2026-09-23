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
* the persisted expected-points components at the grain each is persisted at;
* the Monte Carlo quantile grid, through the scoreboard's own coverage block;
* the uncalibrated FPL assist-mapping constant, as a DECISION it reports rather
  than a value it may change.

NO IN-SAMPLE CALIBRATION
------------------------
A transform scored at target event ``E`` is fitted only on outcomes finalised
STRICTLY BEFORE ``E``.  Every origin's fitted parameters and fit basis are
recorded, the basis digest is part of the fitted version string, and a basis
containing an outcome at or after its origin is REFUSED rather than
approximated.  A figure produced by fitting and scoring on the same event set is
not evidence and is never reported as if it were.

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
reported as insufficient; an empty count is rendered as null, never ``0.0``.
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

DIAGNOSIS_MISCALIBRATED = "MISCALIBRATED"
DIAGNOSIS_NO_MATERIAL_DEFECT = "NO_MATERIAL_DEFECT_DETECTED"
DIAGNOSIS_INSUFFICIENT = "INSUFFICIENT_FOR_DIAGNOSIS"

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


def _probability_definitions() -> dict[str, sb.ProbabilityMetricDefinition]:
    return {definition.metric: definition for definition in sb.PROBABILITY_METRICS}


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
    ),
    ComponentDefinition(
        component="defcon_xpts",
        selector="defcon_points",
        realised="defcon_points where defensive_contribution reaches the position's declared threshold",
        interpretation=(
            "direct on the calibrated probability it is built from; the inverse action rate it "
            "consumes may be weakly evidenced (DEFCON_PRIOR_WEAK)"
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
    official facts.  It is counted as a data gap and never scored as zero.
    """

    if outcome is None:
        return None
    minutes = outcome.get("minutes")
    if minutes is None:
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
        conceded = outcome.get("goals_conceded")
        if conceded is None:
            return None
        if position not in rules.goals_conceded_positions:
            return 0.0
        steps = int(conceded) // int(rules.goals_conceded_per_deduction)
        return -float(steps) * abs(float(rules.goals_conceded_points_for(position)))
    if selector == "defcon_points":
        threshold = rules.defcon_threshold_for(position)
        contribution = outcome.get("defensive_contribution")
        if threshold is None or contribution is None:
            return None
        if position not in rules.defcon_positions:
            return 0.0
        return float(rules.defcon_points) if float(contribution) >= threshold else 0.0
    if selector == "save_points":
        saves = outcome.get("saves")
        if saves is None:
            return None
        if position != "GKP":
            return 0.0
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


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _final_events(conn: sqlite3.Connection, events: Sequence[int]) -> tuple[list[int], list[dict[str, Any]]]:
    """The target events whose stored official state is FINAL, and the rest.

    The declared probability population is restricted to FINAL target events, so
    a provisional or in-progress event is reported with its state instead of
    contributing rows.
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


def causal_origins(
    rows: Sequence[sb.ProbabilityScoringRow],
    *,
    surface: str,
    grain: str = wf.GRAIN_PLAYER_FIXTURE,
    clip_floor: float = pc.DEFAULT_CLIP_FLOOR,
) -> dict[int, pc.CausalFit]:
    """One fit per target event, over the same surface's strictly earlier rows.

    The basis is drawn from THIS surface's rows inside the declared population and
    from strictly earlier target events of the same anchor.  Nothing outside the
    declared population is borrowed: a fit that reached beyond it would describe a
    population no reported figure covers.
    """

    by_event: dict[int, list[sb.ProbabilityScoringRow]] = {}
    for row in rows:
        by_event.setdefault(int(row.event), []).append(row)
    fits: dict[int, pc.CausalFit] = {}
    for origin in sorted(by_event):
        basis = [
            pc.CalibrationObservation(
                event=int(candidate.event),
                key=(int(candidate.player_id), int(candidate.fixture_id)),
                probability=float(candidate.probability),
                outcome=float(candidate.outcome),
            )
            for event, candidates in by_event.items()
            if int(event) < int(origin)
            for candidate in candidates
        ]
        fits[int(origin)] = pc.fit_platt_causal(
            basis, origin_event=int(origin), surface=str(surface), grain=str(grain),
            clip_floor=clip_floor,
        )
    return fits


def _challenger_block(
    rows: Sequence[sb.ProbabilityScoringRow],
    *,
    surface: str,
    grain: str,
) -> tuple[dict[str, Any], list[sb.ProbabilityScoringRow]]:
    """The causal challenger arm: fitted transform per origin, scored at its own origin.

    Returns the reportable block and the rows the arm could actually cover.  A row
    whose origin had no admissible transform is EXCLUDED AND COUNTED, never scored
    under a transform fitted somewhere else.
    """

    fits = causal_origins(rows, surface=surface, grain=grain)
    origins: list[dict[str, Any]] = []
    covered: list[sb.ProbabilityScoringRow] = []
    excluded_by_reason: dict[str, int] = {}
    for origin in sorted(fits):
        fit = fits[origin]
        origin_rows = [row for row in rows if int(row.event) == int(origin)]
        record: dict[str, Any] = {
            "origin_event": int(origin),
            "fit": fit.as_dict(),
            "rows_at_origin": len(origin_rows),
            "rows_scored_at_origin": 0,
        }
        if fit.available:
            covered.extend(origin_rows)
            record["rows_scored_at_origin"] = len(origin_rows)
        else:
            reason = str(fit.reason or pc.FIT_INSUFFICIENT_OBSERVATIONS)
            excluded_by_reason[reason] = excluded_by_reason.get(reason, 0) + len(origin_rows)
        origins.append(record)

    block: dict[str, Any] = {
        "surface": str(surface),
        "grain": str(grain),
        "method": pc.PLATT_FIT_METHOD,
        "policy_version": pc.CAUSAL_FIT_POLICY_VERSION,
        "calibration_policy_version": pc.PROBABILITY_CALIBRATION_POLICY_VERSION,
        "basis_policy": (
            "the same surface's rows from strictly earlier target events of this anchor, restricted to "
            "the declared population; an origin with an insufficient basis is reported rather than "
            "fitted from rows the figures do not cover"
        ),
        "origins": origins,
        "rows_without_origin_transform": sum(
            1 for row in rows if not fits[int(row.event)].available
        ),
        "excluded_by_reason": {str(key): int(value) for key, value in sorted(excluded_by_reason.items())},
    }
    if not covered:
        reasons = sorted(excluded_by_reason)
        block.update(
            {
                "status": STATUS_UNREACHABLE,
                "reason": (
                    "no origin had an admissible causal transform, so the challenger cannot cover the "
                    "comparison population"
                ),
                "exclusion_reasons": reasons,
                "population": {
                    "population_digest": None,
                    "n": 0,
                    "rows_without_origin_transform": len(rows),
                    "excluded_by_reason": block["excluded_by_reason"],
                },
                "gate": None,
                "incumbent_on_comparison_population": None,
                "candidate": None,
            }
        )
        return block, covered

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
                    "rows_without_origin_transform": block["rows_without_origin_transform"],
                    "excluded_by_reason": block["excluded_by_reason"],
                },
                "incumbent_on_comparison_population": None,
                "candidate": None,
            }
        )
        return block, covered

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
                "rows_without_origin_transform": block["rows_without_origin_transform"],
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
    return block, covered


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
    """Whether the fitted transform has a MANDATE, which only a diagnosed defect gives it.

    A transform exists to correct a measured miscalibration.  Where no defect is
    diagnosed the fitted parameters are still reported -- they are the sensitivity
    of the stated probabilities to a logit-linear map, which is exactly what a
    reader needs to see when the answer is "no defect" -- but they are not a
    candidate calibration, and nothing here proposes one.
    """

    status = str(diagnosis.get("status"))
    if status == DIAGNOSIS_MISCALIBRATED:
        mandate = "DIAGNOSED_DEFECT"
        note = (
            "a material calibration gap is diagnosed, so a transform has a mandate; it is still not "
            "promoted, and the transform's own status says whether an admissible one exists"
        )
    elif status == DIAGNOSIS_INSUFFICIENT:
        mandate = "INSUFFICIENT_FOR_DIAGNOSIS"
        note = (
            "the sample cannot support a calibration claim on this surface, so no defect is diagnosed "
            "and a transform is fitted for description only"
        )
    else:
        mandate = "NO_DEFECT_DIAGNOSED"
        note = (
            "no material defect is diagnosed on this surface, so the fitted parameters are a "
            "descriptive sensitivity rather than a candidate calibration"
        )
    return {
        "mandate": mandate,
        "note": note,
        "promotable_by_pe8": False,
        "transform_status": str(challenger.get("status")),
        "rule": (
            "a transform is admissible only to correct a measured miscalibration on a declared "
            "surface; PE-8 fits none speculatively and promotes none at all"
        ),
    }


def probability_calibration_block(
    conn: sqlite3.Connection,
    *,
    events: Sequence[int],
    xpts_runs: Mapping[int, int],
    excluded_events: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Every declared probability surface: diagnosis first, then any transform."""

    wanted = sorted({int(event) for event in events})
    population = sb.player_fixture_population(conn, events=wanted, xpts_runs=xpts_runs)
    scored = sb.probability_scoring_rows(population["rows"])
    by_metric: dict[str, list[sb.ProbabilityScoringRow]] = {
        definition.metric: [] for definition in sb.PROBABILITY_METRICS
    }
    for row in scored["rows"]:
        by_metric[row.metric].append(row)

    definitions = _probability_definitions()
    surfaces: list[dict[str, Any]] = []
    for definition in sb.PROBABILITY_METRICS:
        rows = by_metric[definition.metric]
        probabilities = [row.probability for row in rows]
        outcomes = [row.outcome for row in rows]
        figure = _figure(probabilities, outcomes)
        events_with_rows = sorted({int(row.event) for row in rows})
        keys = [row.key for row in rows]
        challenger, _covered = _challenger_block(
            rows, surface=definition.metric, grain=wf.GRAIN_PLAYER_FIXTURE
        )
        diagnosis = _diagnosis(
            figure["reliability"]["bins_over_floor"], figure["reliability"]["max_abs_gap_over_floor"]
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
    xpts_runs: Mapping[int, int],
) -> dict[str, Any]:
    """The uncalibrated assist-mapping constant, as a decision PE-8 reports.

    The coefficient stays ``1.0`` and ``assist_mapping_calibrated`` stays
    ``False`` unless a causal walk-forward fit on finalised outcomes supports a
    different value and a review promotes it.  A fit that IS supported is
    reported as a candidate; PE-8 does not re-point the constant, and it never
    silences the truthful ``FPL_ASSIST_MAPPING_UNCALIBRATED`` flag.
    """

    config = xpts_module.XPtsConfig()
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
        "basis_policy": (
            "the same population's strictly earlier target events; expected_xa is the persisted "
            "expected xA the production formula multiplies, and the realised side is the official "
            "assists column for the same player and fixture"
        ),
        "origins": [],
        "pooled_evidence": {
            "observations": 0,
            "events": 0,
            "expected_assists_total": 0.0,
            "realised_assists_total": 0.0,
        },
    }
    if not wanted or not xpts_runs:
        block["decision"] = assist_mapping_decision(block["origins"], block["pooled_evidence"])
        return block

    population = sb.player_fixture_population(conn, events=wanted, xpts_runs=xpts_runs)
    observations: list[tuple[int, sb.PlayerFixtureScoringRow, float]] = []
    for row in population["rows"]:
        expected = row.payload.get("expected_xa")
        if expected is None:
            continue
        observations.append((int(row.event), row, float(expected)))
    by_event: dict[int, list[tuple[sb.PlayerFixtureScoringRow, float]]] = {}
    for event, row, expected in observations:
        by_event.setdefault(int(event), []).append((row, expected))

    pooled_expected = 0.0
    pooled_realised = 0.0
    pooled_observations = 0
    for origin in sorted(by_event):
        basis_rows: list[pc.AssistMappingObservation] = []
        for event in sorted(by_event):
            if int(event) >= int(origin):
                continue
            for row, expected in by_event[event]:
                realised = row.outcome.get("assists")
                if realised is None:
                    # The realised side is the official assists column for the same
                    # player and fixture; an absent column is a gap, not a zero.
                    continue
                basis_rows.append(
                    pc.AssistMappingObservation(
                        event=int(event),
                        key=(int(row.player_id), int(row.fixture_id)),
                        expected_assists=float(expected),
                        realised_assists=float(int(realised)),
                    )
                )
        fit = pc.fit_assist_mapping_causal(
            basis_rows, origin_event=int(origin), surface=ASSIST_MAPPING_SURFACE, grain=ASSIST_MAPPING_GRAIN
        )
        block["origins"].append(
            {
                "origin_event": int(origin),
                "rows_at_origin": len(by_event[int(origin)]),
                "fit": fit.as_dict(),
            }
        )
    for _event, row, expected in observations:
        assists = row.outcome.get("assists")
        if assists is None:
            continue
        pooled_expected += float(expected)
        pooled_realised += float(int(assists))
        pooled_observations += 1
    block["pooled_evidence"] = {
        "observations": int(pooled_observations),
        "events": len(by_event),
        "expected_assists_total": round(pooled_expected, 6),
        "realised_assists_total": round(pooled_realised, 6),
    }
    block["decision"] = assist_mapping_decision(block["origins"], block["pooled_evidence"])
    return block


def assist_mapping_decision(
    origins: Sequence[Mapping[str, Any]], pooled: Mapping[str, Any]
) -> dict[str, Any]:
    """The declared a priori decision for the assist-mapping constant.

    ``NO_CHANGE`` is an expected outcome, not a failure: the coefficient stays
    ``1.0`` and the truthful flag keeps being emitted.  ``CANDIDATE_FOR_REVIEW``
    says a causal fit exists AND the evidence meets the declared descriptive
    floors -- it still does not move the constant, because a promotion changes
    production behaviour and belongs to review.
    """

    observations = int(pooled.get("observations") or 0)
    events = int(pooled.get("events") or 0)
    sufficient = (
        events >= sb.MIN_TARGET_EVENTS_FOR_DESCRIPTIVE
        and observations >= sb.MIN_OBSERVATIONS_FOR_DESCRIPTIVE
    )
    admissible = [
        entry
        for entry in origins
        if (entry.get("fit") or {}).get("available") and (entry.get("fit") or {}).get("in_bounds")
    ]
    reasons: list[str] = []
    if not sufficient:
        reasons.append(
            "the available causal evidence is below the declared descriptive floors "
            f"({events} event(s), {observations} observation(s)), so it cannot support a promotion"
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
                {float((entry["fit"])["coefficient"]) for entry in admissible}
            ),
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
    final_events, excluded_events = _final_events(conn, target_events)
    xpts_runs = _xpts_runs(anchor, final_events)
    monte_carlo_runs = _ancillary_runs(anchor, final_events, "monte_carlo_v1")

    probability_block = probability_calibration_block(
        conn, events=final_events, xpts_runs=xpts_runs, excluded_events=excluded_events
    )
    expected_value = expected_value_block(
        conn, anchor=anchor, events=final_events, xpts_runs=xpts_runs
    )
    coverage = monte_carlo_block(conn, events=final_events, monte_carlo_runs=monte_carlo_runs)
    assist = assist_mapping_block(conn, events=final_events, xpts_runs=xpts_runs)

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
