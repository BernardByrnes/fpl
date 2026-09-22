"""PE-7 evaluation — incumbent versus challenger on identical causal populations.

WHAT THIS IS
------------
One artifact for senior review, and nothing more: it MEASURES.  It accepts no
arm, promotes no model, re-points no incumbent version identifier, writes no
projection run and touches no RNG.  ``NO CHANGE`` is a perfectly good outcome of
the numbers below.

WHAT IS COMPARED
----------------
Two families, each with its own frozen incumbent and its own challenger arms:

* the TEAM family, whose primary target is the stored official realised TEAM-SIDE
  xG per fixture side.  xG is the primary target because the model estimates a
  scoring environment, not match-result luck; actual goals are reported as a
  SECONDARY descriptive check and cannot replace xG as the refinement target.
  Reported at minimum: MAE, RMSE, bias, home/away strata, team and evidence-volume
  strata, predicted versus realised mean xG, and the sample size of every block.
* the PLAYER family, whose primary target is player xG/xA scored against the
  player's own REALISED exposure for the target fixture — ``rate * minutes / 90``
  — so a rate comparison cannot be dominated by PE-6 predicted-minutes error.
  Reported at minimum: xG MAE and bias, xA MAE and bias, position strata,
  prior-evidence and low-history strata, role-change strata where the sample
  permits, and the sample size of every stratum.

Integrated xPts is NOT computed here.  PE-8 owns calibration, and PE-7 may not
promote a model because one downstream xPts sample happened to improve; the
artifact says so explicitly instead of quietly reporting a number it must not use.

IDENTICAL POPULATIONS, SAMPLE-SIZE HONESTY
------------------------------------------
Every arm is scored on ONE record set: a record is scorable only when every arm
produced a value for it, and the count of records dropped because some arm was
undefined is published per arm.  Incumbent and challenger therefore cover an
identical key set, which is checked with PE-2's own ``assert_same_population`` and
recorded as a canonical population digest.  PE-2's declared disclosure floors
(``walk_forward_scoreboard``) decide whether anything may be said at all: below
them the artifact reports INSUFFICIENT_FOR_ANY_CLAIM and draws no comparison.

CUTOFF AND CAUSALITY
--------------------
The cutoff for a target event is its official deadline — the last moment a manager
could have acted — and the predictor's inputs are read under the PE-1 canonical
boundary by the models themselves.  Candidate identity (membership, club,
position) comes from the cutoff's accepted official bootstrap generation.  This
applies to BOTH arms: the frozen incumbent player-rate arm is built by the
cutoff-safe adapter, so the baseline a challenger is scored against reads the same
observable evidence rather than the mutable persisted rows.  Every outcome this
module reads is used ONLY as evaluation evidence: it is never fed back to a model,
and a fixture that has not been played, or that carries no official xG, is excluded
with its own status rather than scored as zero.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Callable, Iterable, Mapping, Sequence

from . import analytics
from . import availability_minutes_challenger as identity_resolution
from . import joint_minutes
from . import minutes_model
from . import monte_carlo
from . import player_attack_challenger as player_challenger
from . import player_rates
from . import repositories as repo
from . import team_attack_challenger as team_challenger
from . import team_model
from . import team_player_attack_coherence as coherence
from . import walk_forward as wf
from . import walk_forward_metrics as wm
from . import walk_forward_scoreboard as wfs
from . import xpts
from .utils import utc_now

EVALUATION_VERSION = "team_player_attack_evaluation_v1.0.0"
ARTIFACT_SCHEMA = "pe7_team_player_attack_evaluation_v1"

#: The cutoff a walk-forward prediction is taken at: the target event's official
#: deadline, the last moment a manager could have acted.  A deadline that is not
#: before the event's first kickoff is refused rather than used -- the same policy
#: PE-6 evaluates under, restated here because each phase publishes its own
#: artifact and neither reaches into the other's module.
CUTOFF_POLICY = "TARGET_EVENT_DEADLINE_TIME"
MISSING_DATA_POLICY_VERSION = wf.MISSING_DATA_POLICY_VERSION

GRAIN_TEAM_FIXTURE_SIDE = coherence.GRAIN_FIXTURE_SIDE
GRAIN_PLAYER_FIXTURE = wf.GRAIN_PLAYER_FIXTURE
GRAIN_TEAM_NOTE = (
    "one observation per (target event, fixture, side): a double gameweek is two observations "
    "for the same team and is never collapsed into one"
)
GRAIN_PLAYER_NOTE = (
    "one observation per (target event, candidate player, fixture): a player's second fixture in "
    "a double gameweek is its own observation, scored on THAT fixture's realised exposure"
)

# --- exclusion statuses (an exclusion is counted, never scored) --------------
STATUS_BLANK_NO_FIXTURE = "TARGET_BLANK_NO_FIXTURE"
STATUS_FIXTURE_NOT_PLAYED = "TARGET_FIXTURE_NOT_PLAYED"
STATUS_FIXTURE_MISSING = "TARGET_FIXTURE_MISSING"
STATUS_TEAM_XG_MISSING = "TEAM_SIDE_XG_EVIDENCE_MISSING"
STATUS_OUTCOME_NOT_FINALISED = wf.OUTCOME_NOT_FINALISED
STATUS_OUTCOME_MISSING = "OUTCOME_EVIDENCE_MISSING"
STATUS_OUTCOME_PLACEHOLDER = wf.OUTCOME_PLACEHOLDER_EXCLUDED
STATUS_OUTCOME_XG_MISSING = "OUTCOME_XG_OR_XA_EVIDENCE_MISSING"
STATUS_NO_REALISED_EXPOSURE = "NO_REALISED_EXPOSURE_FOR_THE_TARGET_FIXTURE"
STATUS_MODEL_PROJECTION_MISSING = wf.MODEL_PROJECTION_MISSING
#: The cutoff holds no completed-fixture xG at all, so the team model has no
#: input evidence and no level to fit.  PE-2's own token for exactly that, reused
#: rather than re-worded: the frozen incumbent's ``lambda_for`` is undefined on an
#: empty evidence set, so a walk-forward that opens on the season's first event
#: must exclude its sides with a status instead of raising from inside the model.
STATUS_TEAM_EVIDENCE_UNAVAILABLE = wf.INPUT_EVIDENCE_UNAVAILABLE
STATUS_EVENT_CUTOFF_UNAVAILABLE = "TARGET_EVENT_CUTOFF_UNAVAILABLE"
STATUS_EVENT_NOT_FINAL = "TARGET_EVENT_OUTCOME_NOT_FINALISED"
STATUS_IDENTITY_UNRESOLVED_AT_CUTOFF = "CANDIDATE_IDENTITY_UNRESOLVED_AT_CUTOFF"
STATUS_IDENTITY_UNAVAILABLE_AT_CUTOFF = "CANDIDATE_IDENTITY_UNAVAILABLE_AT_CUTOFF"

EXCLUSION_STATUSES: tuple[str, ...] = (
    STATUS_BLANK_NO_FIXTURE,
    STATUS_FIXTURE_NOT_PLAYED,
    STATUS_FIXTURE_MISSING,
    STATUS_TEAM_XG_MISSING,
    STATUS_OUTCOME_NOT_FINALISED,
    STATUS_OUTCOME_MISSING,
    STATUS_OUTCOME_PLACEHOLDER,
    STATUS_OUTCOME_XG_MISSING,
    STATUS_NO_REALISED_EXPOSURE,
    STATUS_MODEL_PROJECTION_MISSING,
    STATUS_TEAM_EVIDENCE_UNAVAILABLE,
    STATUS_EVENT_CUTOFF_UNAVAILABLE,
    STATUS_EVENT_NOT_FINAL,
    STATUS_IDENTITY_UNRESOLVED_AT_CUTOFF,
    STATUS_IDENTITY_UNAVAILABLE_AT_CUTOFF,
)

SCOPE_PLAYER = "PLAYER"
SCOPE_EVENT = "EVENT"
SCOPE_TEAM_SIDE = "TEAM_SIDE"

# --- declared comparison rule (a measurement, never a verdict) ---------------
PREFERRED_CHALLENGER = "CHALLENGER"
PREFERRED_INCUMBENT = "INCUMBENT"
WITHIN_NOISE_FLOOR = "WITHIN_DECLARED_NOISE_FLOOR"

MEASUREMENT_UNDEFINED = "MEASUREMENT_UNDEFINED"
MEASUREMENT_INSUFFICIENT_SAMPLE = "INSUFFICIENT_FOR_ANY_CLAIM"
MEASUREMENT_CHALLENGER_PREFERRED = "CHALLENGER_PREFERRED_BY_THE_DECLARED_FLOORS"
MEASUREMENT_PARTIAL = "CHALLENGER_PREFERRED_ON_SOME_CRITERIA_ONLY"
MEASUREMENT_WITHIN_NOISE = "WITHIN_DECLARED_NOISE_FLOOR"
MEASUREMENT_INCUMBENT_PREFERRED = "INCUMBENT_PREFERRED_BY_THE_DECLARED_FLOORS"
MEASUREMENT_NO_ADVANTAGE = "NO_MEASURABLE_ADVANTAGE"

MEASUREMENT_BASIS_INSUFFICIENT_SAMPLE = "INSUFFICIENT_SAMPLE_UNDER_THE_PE2_POLICY"
MEASUREMENT_BASIS_UNDEFINED = "PRIMARY_CRITERION_UNDEFINED"
MEASUREMENT_BASIS_CHALLENGER_WORSE = "CHALLENGER_IS_WORSE_ON_A_PRIMARY_CRITERION"
MEASUREMENT_BASIS_WITHIN_NOISE = "EVERY_PRIMARY_CRITERION_INSIDE_ITS_DECLARED_FLOOR"
MEASUREMENT_BASIS_PARTIAL = "SOME_PRIMARY_CRITERIA_PREFER_THE_CHALLENGER"
MEASUREMENT_BASIS_ALL = "EVERY_PRIMARY_CRITERION_PREFERS_THE_CHALLENGER"
MEASUREMENT_BASIS_NO_ADVANTAGE = "NO_PRIMARY_CRITERION_PREFERS_THE_CHALLENGER"

MEASUREMENT_VOCABULARY: tuple[str, ...] = (
    MEASUREMENT_UNDEFINED,
    MEASUREMENT_INSUFFICIENT_SAMPLE,
    MEASUREMENT_CHALLENGER_PREFERRED,
    MEASUREMENT_PARTIAL,
    MEASUREMENT_WITHIN_NOISE,
    MEASUREMENT_NO_ADVANTAGE,
    MEASUREMENT_INCUMBENT_PREFERRED,
)

SELECTION_CLAIM_NOT_MADE = "NOT_CLAIMED_UNTIL_SENIOR_REVIEW"
CLAIMS_ALLOWED: tuple[str, ...] = ("DESCRIPTIVE_REPORTING_ONLY",)
CLAIMS_NOT_ALLOWED: tuple[str, ...] = (
    "MODEL_SELECTION",
    "PROMOTION",
    "CERTIFICATION",
    "PE8_CALIBRATION",
)

#: The criteria the measurement summary is computed on.  Declared a priori and
#: never selected from the data.
PRIMARY_CRITERIA_TEAM: tuple[str, ...] = ("team_xg_mae",)
PRIMARY_CRITERIA_PLAYER: tuple[str, ...] = ("player_xg_mae", "player_xa_mae")

TEAM_ERROR_METRICS: tuple[str, ...] = ("team_xg_mae", "team_xg_rmse", "team_goals_mae")
TEAM_BIAS_METRICS: tuple[str, ...] = ("team_xg_bias", "team_goals_bias")
PLAYER_ERROR_METRICS: tuple[str, ...] = ("player_xg_mae", "player_xa_mae")
PLAYER_BIAS_METRICS: tuple[str, ...] = ("player_xg_bias", "player_xa_bias")

#: Declared disclosure floors.  A change smaller than its floor is not an
#: improvement however it is signed.  These are NOT significance tests and NOT
#: acceptance thresholds.
DELTA_XG_MAE_IMPROVEMENT_MIN = 0.01
DELTA_GOALS_MAE_IMPROVEMENT_MIN = 0.01
DELTA_BIAS_IMPROVEMENT_MIN = 0.01

DELTA_FLOORS: dict[str, float] = {
    "team_xg_mae": DELTA_XG_MAE_IMPROVEMENT_MIN,
    "team_xg_rmse": DELTA_XG_MAE_IMPROVEMENT_MIN,
    "team_goals_mae": DELTA_GOALS_MAE_IMPROVEMENT_MIN,
    "team_xg_bias": DELTA_BIAS_IMPROVEMENT_MIN,
    "team_goals_bias": DELTA_BIAS_IMPROVEMENT_MIN,
    "player_xg_mae": DELTA_XG_MAE_IMPROVEMENT_MIN,
    "player_xa_mae": DELTA_XG_MAE_IMPROVEMENT_MIN,
    "player_xg_bias": DELTA_BIAS_IMPROVEMENT_MIN,
    "player_xa_bias": DELTA_BIAS_IMPROVEMENT_MIN,
}

DELTA_CONVENTION = (
    "delta = challenger - incumbent; for MAE/RMSE lower is better, for bias closer to zero is "
    "better, and |delta| below the declared floor is WITHIN_DECLARED_NOISE_FLOOR"
)
BIAS_CONVENTION = "mean(predicted - actual); positive means overprediction"

#: Declared disclosure floor for a stratum.  NOT a significance threshold.
MIN_STRATUM_OBSERVATIONS = 30

TEAM_STRATUM_DIMENSIONS: tuple[str, ...] = ("by_venue", "by_team", "by_team_evidence")
PLAYER_STRATUM_DIMENSIONS: tuple[str, ...] = (
    "by_position",
    "by_prior_evidence",
    "by_history_coverage",
    "by_role_change",
    "by_current_exposure",
)

#: The secondary criteria.  Reported so a reader can see them, excluded from the
#: measurement summary by construction.
SECONDARY_DESCRIPTIVE_CRITERIA: tuple[str, ...] = ("team_goals_mae", "team_goals_bias")
XPTS_DIAGNOSTIC_STATUS = "NOT_COMPUTED"
XPTS_DIAGNOSTIC_REASON = (
    "PE-8 owns probability/xPts calibration.  PE-7 may not promote a model because one downstream "
    "xPts sample happened to improve, so no integrated xPts diagnostic is computed here and none "
    "is claimed."
)

#: Prior-evidence bands, from the cutoff-observable prior-season rows only.
PRIOR_BAND_NONE = "NO_CUTOFF_OBSERVABLE_PRIOR_ROW"
PRIOR_BAND_LOW = "PRIOR_MINUTES_BELOW_450"
PRIOR_BAND_MEDIUM = "PRIOR_MINUTES_450_TO_899"
PRIOR_BAND_HIGH = "PRIOR_MINUTES_900_PLUS"
PRIOR_EVIDENCE_BANDS: tuple[str, ...] = (PRIOR_BAND_NONE, PRIOR_BAND_LOW, PRIOR_BAND_MEDIUM, PRIOR_BAND_HIGH)

EXPOSURE_BAND_NONE = "NO_CURRENT_EXPOSURE"
EXPOSURE_BAND_LOW = "CURRENT_MINUTES_BELOW_450"
EXPOSURE_BAND_HIGH = "CURRENT_MINUTES_450_PLUS"
EXPOSURE_BANDS: tuple[str, ...] = (EXPOSURE_BAND_NONE, EXPOSURE_BAND_LOW, EXPOSURE_BAND_HIGH)

COVERAGE_BAND_NONE = "NO_XG_BEARING_PRIOR_SEASON"
COVERAGE_BAND_PRESENT = "XG_BEARING_PRIOR_SEASON_PRESENT"
COVERAGE_BANDS: tuple[str, ...] = (COVERAGE_BAND_NONE, COVERAGE_BAND_PRESENT)

TEAM_EVIDENCE_LOW = "TEAM_MATCHES_BELOW_4"
TEAM_EVIDENCE_SUFFICIENT = "TEAM_MATCHES_4_PLUS"
TEAM_EVIDENCE_BANDS: tuple[str, ...] = (TEAM_EVIDENCE_LOW, TEAM_EVIDENCE_SUFFICIENT)

PRIOR_EVIDENCE_LOW_MINUTES = 450.0
PRIOR_EVIDENCE_HIGH_MINUTES = 900.0
CURRENT_EXPOSURE_MINUTES = 450.0

#: Volatile fields stripped before two artifacts are compared for determinism.
VOLATILE_ARTIFACT_KEYS: tuple[str, ...] = ("generated_at",)


#: Every mutable current-state surface PE-7 touches, with HOW the cutoff is kept
#: safe on that surface.  The contract requires an explicit audit of each of these
#: rather than an assurance that leakage "does not happen": a mutable table can
#: leak history until proven otherwise, so each use is named with its resolution.
CAUSALITY_AUDIT: tuple[dict[str, str], ...] = (
    {
        "surface": "players.team_id (candidate club)",
        "used_for": "which club a candidate plays for at the cutoff",
        "resolution": (
            "the club comes from the cutoff's accepted official bootstrap generation, never from "
            "the persisted column; the persisted pool is read only to REPORT divergence"
        ),
    },
    {
        "surface": "players.team_id (historical fixture side)",
        "used_for": "attributing a historical fixture's xG to a side",
        "resolution": (
            "fixture side comes from player_gameweeks.was_home; the club column is not read for it, "
            "so a later transfer cannot re-attribute a past fixture"
        ),
    },
    {
        "surface": "players.element_type",
        "used_for": "candidate position, position pools and position strata",
        "resolution": (
            "the position comes from the cutoff generation on BOTH arms: the challenger's pools and "
            "the incumbent comparison arm's pools are the same cutoff-stable pools, and neither the "
            "challenger nor the incumbent arm is built from the incumbent's persisted-row pooling"
        ),
    },
    {
        "surface": "players.is_active",
        "used_for": "candidate membership",
        "resolution": (
            "membership is the cutoff generation's element id set: an inactive player the cutoff "
            "placed in the pool stays a candidate, and the divergence is recorded"
        ),
    },
    {
        "surface": "current club identity (player/team joins)",
        "used_for": "pooled league and position priors",
        "resolution": (
            "cutoff_stable_rate_pools attributes every pooled prior-season row by the identity the "
            "cutoff resolves, over rows observable at the cutoff, and BOTH arms read those pools: "
            "the incumbent comparison arm is built by the cutoff-safe adapter, so the incumbent's "
            "own persisted-row pooling is not on either arm's path"
        ),
    },
    {
        "surface": "frozen incumbent player-rate arm (the comparison)",
        "used_for": "the arm every challenger is scored against",
        "resolution": (
            "incumbent_player_rate_rows rebuilds the incumbent arm from cutoff-observable season "
            "history and the cutoff-stable pools, with the incumbent's own arithmetic, field "
            "contract and model version; player_rates is not modified and no identity is "
            "re-pointed, so a post-cutoff season write or a later position change cannot move the "
            "ARM the delta is measured against"
        ),
    },
    {
        "surface": "current role / scouting evidence",
        "used_for": "the role-change discount and the role-segmented exposure",
        "resolution": (
            "repo.scouting_current_rows_as_of(cutoff) plus the incumbent's own eligibility rule "
            "(observed_at <= cutoff, not expired at it); a post-cutoff note proves nothing"
        ),
    },
    {
        "surface": "player/team joins for historical priors",
        "used_for": "team attack/defence parameters",
        "resolution": (
            "the team model reads team_model.team_match_rows -> completed_fixture_xg under "
            "historical_observations.OBSERVATION_SQL_CLAUSES; it joins no players row at all"
        ),
    },
    {
        "surface": "player_season_histories observed_at",
        "used_for": "prior-season evidence and the pooled priors built from it",
        "resolution": (
            "a row is evidence only when observed_at <= cutoff; a row with no observation time is "
            "excluded as unprovable, and both counts are published"
        ),
    },
)

#: How PE-7 keeps itself inside PE-1 rather than beside it.
CAUSALITY_DISCIPLINE: dict[str, Any] = {
    "boundary": "fpl_brain.historical_observations (PE-1), composed, never restated",
    "second_predicate_introduced": False,
    "evidence_readers": [
        "analytics.completed_rows_as_of -> historical_player_fixture_rows",
        "team_model.team_match_rows -> completed_fixture_xg -> fixture_side_xg",
        "repo.scouting_current_rows_as_of",
        "repo.player_season_histories (as-of filtered by this module's prior reader)",
    ],
    "incumbent_arm": (
        "the frozen incumbent player-rate comparison arm is built by "
        "player_attack_challenger.incumbent_player_rate_rows, which reads the same "
        "cutoff-observable season history and the same cutoff-stable pools as the challenger and "
        "keeps the incumbent's own arithmetic, field contract and model version; the incumbent's "
        "own prior and pool readers take no observation time and join the PERSISTED players row, so "
        "they are deliberately NOT the comparison path"
    ),
    "realised_outcomes": (
        "read for SCORING only, through team_model.realised_fixture_side_xg and a direct "
        "player_gameweeks lookup on the target fixture; no realised value is ever an input to a "
        "projection, and an unplayed fixture or a missing xG value is excluded rather than zeroed"
    ),
    "placeholders": "refused at the boundary by the canonical value signature, and counted",
    "missing_is_zero": False,
}


class EvaluationError(RuntimeError):
    """The evaluation cannot be produced honestly from what is available."""


# ---------------------------------------------------------------------------
# Frozen incumbent identities.
# ---------------------------------------------------------------------------


def frozen_incumbent_identity() -> dict[str, Any]:
    """Every incumbent identity PE-7 must not silently rewrite, verified.

    The PE-6 minutes trio is on this list because PE-7's promotion target excludes
    it, and the xPts / Monte Carlo identities are here because a PE-7 change that
    moved either would be a different work item, not a refinement.
    """

    declared = {
        "team_model.TEAM_MODEL_VERSION": (team_model.TEAM_MODEL_VERSION, "team_strength_v1.1.0"),
        "team_model.TEAM_BASELINE_MODEL_VERSION": (team_model.TEAM_BASELINE_MODEL_VERSION, "team_naive_v1.1.0"),
        "player_rates.PLAYER_RATE_MODEL_VERSION": (player_rates.PLAYER_RATE_MODEL_VERSION, "player_rates_v1.0.0"),
        "player_rates.PLAYER_RATE_BASELINE_MODEL_VERSION": (
            player_rates.PLAYER_RATE_BASELINE_MODEL_VERSION,
            "player_rate_baseline_v1.0.0",
        ),
        "minutes_model.MINUTES_MODEL_VERSION": (minutes_model.MINUTES_MODEL_VERSION, "minutes_v1.8.0"),
        "minutes_model.MINUTES_COHERENT_MODEL_VERSION": (
            minutes_model.MINUTES_COHERENT_MODEL_VERSION,
            "minutes_v1.2.0",
        ),
        "joint_minutes.JOINT_MINUTES_MODEL_VERSION": (
            joint_minutes.JOINT_MINUTES_MODEL_VERSION,
            "minutes_v1.5.2",
        ),
        "xpts.XPTS_MODEL_VERSION": (xpts.XPTS_MODEL_VERSION, "xpts_v1.4.1"),
        "monte_carlo.MONTE_CARLO_MODEL_VERSION": (monte_carlo.MONTE_CARLO_MODEL_VERSION, "mc_v1.3.0"),
    }
    mismatches = {
        name: {"declared": expected, "actual": actual}
        for name, (actual, expected) in declared.items()
        if actual != expected
    }
    return {
        "values": {name: actual for name, (actual, _expected) in declared.items()},
        "declared": {name: expected for name, (_actual, expected) in declared.items()},
        "unchanged": not mismatches,
        "mismatches": mismatches,
        "pe6_minutes_trio": [
            "minutes_model.MINUTES_MODEL_VERSION",
            "minutes_model.MINUTES_COHERENT_MODEL_VERSION",
            "joint_minutes.JOINT_MINUTES_MODEL_VERSION",
        ],
        "note": (
            "the PE-6 minutes trio is frozen and OUTSIDE PE-7's promotion target; the xPts and "
            "Monte Carlo identities are unchanged because PE-7 changes no downstream path"
        ),
    }


# ---------------------------------------------------------------------------
# Cutoff and outcome reads.
# ---------------------------------------------------------------------------


def event_cutoff(conn: sqlite3.Connection, event: int) -> tuple[str | None, list[str]]:
    """The cutoff a prediction for ``event`` is taken at, or why there is none.

    Fail closed: no deadline means no point in time anyone decided at, and a
    deadline at or after the first kickoff would score a prediction made with
    knowledge of the event.
    """

    row = conn.execute("SELECT deadline_time FROM events WHERE id=?", (int(event),)).fetchone()
    if row is None:
        return None, ["target event is not present in the events table"]
    deadline = str(row[0] or "").strip()
    if not deadline:
        return None, ["target event carries no official deadline_time"]
    kickoffs = [
        str(value)
        for (value,) in conn.execute(
            "SELECT kickoff_time FROM fixtures WHERE event=? AND kickoff_time IS NOT NULL",
            (int(event),),
        ).fetchall()
    ]
    if kickoffs and min(kickoffs) <= deadline:
        return None, [
            "the official deadline_time is not before the event's first kickoff, so it cannot "
            "serve as a causal cutoff"
        ]
    return deadline, []


def target_event_state(conn: sqlite3.Connection, event: int) -> tuple[str, list[str]]:
    from . import planning

    state, reasons = planning.event_data_state(conn, int(event))
    return str(state), list(reasons)


def event_fixtures(conn: sqlite3.Connection, event: int) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in conn.execute(
            "SELECT * FROM fixtures WHERE event=? ORDER BY id", (int(event),)
        ).fetchall()
    ]


def realised_team_side(
    conn: sqlite3.Connection, fixture_id: int, team_id: int
) -> tuple[dict[str, Any] | None, str]:
    """The realised team-side xG (primary) and goals (secondary) of one fixture side.

    A fixture that did not start and finish is not a played match -- a scheduled
    placeholder is not a result -- and a side whose official xG is missing is
    excluded rather than scored as zero.
    """

    row = conn.execute("SELECT * FROM fixtures WHERE id=?", (int(fixture_id),)).fetchone()
    if row is None:
        return None, STATUS_FIXTURE_MISSING
    fixture = dict(row)
    if not (fixture.get("finished") and fixture.get("started")):
        return None, STATUS_FIXTURE_NOT_PLAYED
    sides = team_model.realised_fixture_side_xg(conn, int(fixture_id))
    was_home = int(fixture["team_h"]) == int(team_id)
    xg = sides["home_xg"] if was_home else sides["away_xg"]
    if xg is None:
        return None, STATUS_TEAM_XG_MISSING
    goals = fixture.get("team_h_score") if was_home else fixture.get("team_a_score")
    return (
        {
            "xg": float(xg),
            "goals": None if goals is None else float(goals),
            "was_home": was_home,
            "venue": "home" if was_home else "away",
            "opponent_id": int(fixture["team_a"]) if was_home else int(fixture["team_h"]),
        },
        "",
    )


def realised_player_fixture(
    conn: sqlite3.Connection, player_id: int, fixture_id: int
) -> tuple[dict[str, Any] | None, str]:
    """The realised official facts for one player-fixture, or why there are none.

    A scheduled placeholder is not a did-not-play, and a missing row is not a
    zero.  A played row with no xG-family value is a MISSING value, excluded with
    its own status rather than counted as a zero.
    """

    row = conn.execute(
        "SELECT * FROM player_gameweeks WHERE player_id=? AND fixture_id=?",
        (int(player_id), int(fixture_id)),
    ).fetchone()
    if row is None:
        return None, STATUS_OUTCOME_MISSING
    values = dict(row)
    if repo.row_is_scheduled_placeholder(values):
        return None, STATUS_OUTCOME_PLACEHOLDER
    minutes = values.get("minutes")
    if minutes is None:
        return None, STATUS_OUTCOME_MISSING
    minutes_value = float(minutes)
    xg = values.get("expected_goals")
    xa = values.get("expected_assists")
    if xg is None or xa is None:
        return None, STATUS_OUTCOME_XG_MISSING
    return (
        {
            "minutes": minutes_value,
            "xg": float(xg),
            "xa": float(xa),
            "goals": None if values.get("goals_scored") is None else float(values["goals_scored"]),
            "assists": None if values.get("assists") is None else float(values["assists"]),
            "appeared": 1 if minutes_value > 0 else 0,
        },
        "",
    )


# ---------------------------------------------------------------------------
# Metrics.
# ---------------------------------------------------------------------------


def _metric(value: Any) -> Any:
    return value.as_dict()


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), wm.METRIC_DECIMALS)


def _pairs(records: Sequence[Mapping[str, Any]], arm: str, key: str) -> tuple[list[float], list[float]]:
    predicted: list[float] = []
    actual: list[float] = []
    for record in records:
        value = record["predicted"][arm][key]
        target = record["realised"].get(key)
        if value is None or target is None:
            continue
        predicted.append(float(value))
        actual.append(float(target))
    return predicted, actual


def team_arm_metrics(records: Sequence[Mapping[str, Any]], arm: str) -> dict[str, Any]:
    """The declared team metric set for one arm over one aligned population."""

    if not records:
        empty = wm.Metric(wm.METRIC_NO_SAMPLE).as_dict()
        return {
            "n": 0,
            "team_xg_mae": empty,
            "team_xg_rmse": empty,
            "team_xg_bias": empty,
            "team_goals_mae": empty,
            "team_goals_bias": empty,
            "team_xg_predicted_mean": None,
            "team_xg_realised_mean": None,
            "team_goals_predicted_mean": None,
            "team_goals_realised_mean": None,
        }
    xg_predicted, xg_realised = _pairs(records, arm, "xg")
    goals_predicted, goals_realised = _pairs(records, arm, "goals")
    return {
        "n": len(records),
        "team_xg_mae": _metric(wm.mean_absolute_error(xg_predicted, xg_realised)),
        "team_xg_rmse": _metric(wm.root_mean_squared_error(xg_predicted, xg_realised)),
        "team_xg_bias": _metric(wm.mean_bias(xg_predicted, xg_realised)),
        "team_goals_mae": _metric(wm.mean_absolute_error(goals_predicted, goals_realised)),
        "team_goals_bias": _metric(wm.mean_bias(goals_predicted, goals_realised)),
        "team_xg_predicted_mean": _mean(xg_predicted),
        "team_xg_realised_mean": _mean(xg_realised),
        "team_goals_predicted_mean": _mean(goals_predicted),
        "team_goals_realised_mean": _mean(goals_realised),
    }


def player_arm_metrics(records: Sequence[Mapping[str, Any]], arm: str) -> dict[str, Any]:
    """The declared player metric set for one arm over one aligned population."""

    if not records:
        empty = wm.Metric(wm.METRIC_NO_SAMPLE).as_dict()
        return {
            "n": 0,
            "player_xg_mae": empty,
            "player_xg_bias": empty,
            "player_xa_mae": empty,
            "player_xa_bias": empty,
            "player_xg_predicted_mean": None,
            "player_xg_realised_mean": None,
            "player_xa_predicted_mean": None,
            "player_xa_realised_mean": None,
            "realised_minutes_mean": None,
        }
    xg_predicted, xg_realised = _pairs(records, arm, "xg")
    xa_predicted, xa_realised = _pairs(records, arm, "xa")
    return {
        "n": len(records),
        "player_xg_mae": _metric(wm.mean_absolute_error(xg_predicted, xg_realised)),
        "player_xg_bias": _metric(wm.mean_bias(xg_predicted, xg_realised)),
        "player_xa_mae": _metric(wm.mean_absolute_error(xa_predicted, xa_realised)),
        "player_xa_bias": _metric(wm.mean_bias(xa_predicted, xa_realised)),
        "player_xg_predicted_mean": _mean(xg_predicted),
        "player_xg_realised_mean": _mean(xg_realised),
        "player_xa_predicted_mean": _mean(xa_predicted),
        "player_xa_realised_mean": _mean(xa_realised),
        "realised_minutes_mean": _mean([float(record["realised"]["minutes"]) for record in records]),
    }


def _preference(metric_name: str, incumbent_value: float | None, challenger_value: float | None) -> str | None:
    """Which arm the declared floor prefers on one metric, or None if undefined."""

    if incumbent_value is None or challenger_value is None:
        return None
    floor = float(DELTA_FLOORS.get(metric_name, 0.0))
    if metric_name in (*TEAM_ERROR_METRICS, *PLAYER_ERROR_METRICS):
        delta = float(challenger_value) - float(incumbent_value)
        if delta <= -floor:
            return PREFERRED_CHALLENGER
        if delta >= floor:
            return PREFERRED_INCUMBENT
        return WITHIN_NOISE_FLOOR
    improvement = abs(float(incumbent_value)) - abs(float(challenger_value))
    if improvement >= floor:
        return PREFERRED_CHALLENGER
    if improvement <= -floor:
        return PREFERRED_INCUMBENT
    return WITHIN_NOISE_FLOOR


def comparison_block(
    incumbent_metrics: Mapping[str, Any],
    challenger_metrics: Mapping[str, Any],
    *,
    error_metrics: Sequence[str],
    bias_metrics: Sequence[str],
    rate_metrics: Sequence[str],
) -> dict[str, Any]:
    """Per-metric incumbent/challenger comparison under the declared floors."""

    block: dict[str, Any] = {
        "delta_convention": DELTA_CONVENTION,
        "bias_convention": BIAS_CONVENTION,
        "floors": {name: float(DELTA_FLOORS.get(name, 0.0)) for name in (*error_metrics, *bias_metrics)},
        "floor_basis": "declared a priori disclosure floors, not significance tests",
        "metrics": {},
        "rates": {},
    }
    for name in (*error_metrics, *bias_metrics):
        incumbent_metric = incumbent_metrics.get(name) or {}
        challenger_metric = challenger_metrics.get(name) or {}
        incumbent_value = incumbent_metric.get("value")
        challenger_value = challenger_metric.get("value")
        delta = (
            None
            if incumbent_value is None or challenger_value is None
            else round(float(challenger_value) - float(incumbent_value), wm.METRIC_DECIMALS)
        )
        block["metrics"][name] = {
            "incumbent": incumbent_value,
            "challenger": challenger_value,
            "delta": delta,
            "n": min(int(incumbent_metric.get("n") or 0), int(challenger_metric.get("n") or 0)),
            "status": (
                "OK" if incumbent_value is not None and challenger_value is not None else "UNDEFINED"
            ),
            "preferred": _preference(name, incumbent_value, challenger_value),
            "role": "PRIMARY" if name in (*PRIMARY_CRITERIA_TEAM, *PRIMARY_CRITERIA_PLAYER) else (
                "SECONDARY_DESCRIPTIVE" if name in SECONDARY_DESCRIPTIVE_CRITERIA else "SUPPORTING"
            ),
        }
    for name in rate_metrics:
        block["rates"][name] = {
            "incumbent": incumbent_metrics.get(name),
            "challenger": challenger_metrics.get(name),
        }
    return block


def measurement_rule_block(primary_criteria: Sequence[str]) -> dict[str, Any]:
    """The declared, a-priori rule this artifact measures under."""

    return {
        "primary_criteria": list(primary_criteria),
        "secondary_descriptive_criteria": list(SECONDARY_DESCRIPTIVE_CRITERIA),
        "excluded_from_measurement": list(SECONDARY_DESCRIPTIVE_CRITERIA),
        "floors": {name: float(DELTA_FLOORS.get(name, 0.0)) for name in primary_criteria},
        "floor_basis": (
            "declared a priori disclosure floors, NOT significance tests and NOT acceptance "
            "thresholds"
        ),
        "delta_convention": DELTA_CONVENTION,
        "bias_convention": BIAS_CONVENTION,
        "measurement_vocabulary": list(MEASUREMENT_VOCABULARY),
        "criteria_declared_a_priori": True,
        "selected_from_the_data": False,
    }


def arm_measurement(
    comparison: Mapping[str, Any], *, primary_criteria: Sequence[str], sample_sufficient: bool
) -> dict[str, Any]:
    """What one arm's numbers show on this population.  Pure function.

    Deliberately not a verdict: the tokens describe measurements — which arm the
    declared floors prefer, and on which criteria — never an acceptance.  Order of
    refusal matters: an inadequate sample is reported as an inadequate sample,
    never as a failed challenger and never as a selection claim.
    """

    if not sample_sufficient:
        return {
            "token": MEASUREMENT_INSUFFICIENT_SAMPLE,
            "basis": [MEASUREMENT_BASIS_INSUFFICIENT_SAMPLE],
            "note": (
                "the PE-2 policy's descriptive floor is not met on this population, so no "
                "measurement of the difference is drawn and nothing is claimed"
            ),
        }
    metrics = comparison.get("metrics") or {}
    undefined = [name for name in primary_criteria if (metrics.get(name) or {}).get("status") != "OK"]
    if undefined:
        return {
            "token": MEASUREMENT_UNDEFINED,
            "basis": [MEASUREMENT_BASIS_UNDEFINED, *[f"{name}:UNDEFINED" for name in undefined]],
            "note": "a primary criterion could not be computed, so no measurement is drawn",
        }
    preferred = {name: (metrics.get(name) or {}).get("preferred") for name in primary_criteria}
    worse = sorted(name for name, value in preferred.items() if value == PREFERRED_INCUMBENT)
    better = sorted(name for name, value in preferred.items() if value == PREFERRED_CHALLENGER)
    if worse:
        return {
            "token": MEASUREMENT_INCUMBENT_PREFERRED,
            "basis": [
                MEASUREMENT_BASIS_CHALLENGER_WORSE,
                *[f"{name}:incumbent" for name in worse],
                *[f"{name}:challenger" for name in better],
            ],
            "note": (
                "the declared floors prefer the incumbent on a primary criterion, so there is no "
                "measurable advantage to report for this arm"
            ),
        }
    if not better:
        return {
            "token": MEASUREMENT_WITHIN_NOISE,
            "basis": [MEASUREMENT_BASIS_WITHIN_NOISE],
            "note": (
                "every primary criterion is inside its declared floor, so the two arms cannot be "
                "distinguished on this sample"
            ),
        }
    if len(better) == len(primary_criteria):
        return {
            "token": MEASUREMENT_CHALLENGER_PREFERRED,
            "basis": [MEASUREMENT_BASIS_ALL, *[f"{name}:challenger" for name in better]],
            "note": (
                "the declared floors prefer the challenger on every primary criterion of this "
                "sample; this is a measurement for senior review, not an acceptance"
            ),
        }
    return {
        "token": MEASUREMENT_PARTIAL,
        "basis": [
            MEASUREMENT_BASIS_PARTIAL,
            *[f"{name}:challenger" for name in better],
            *[
                f"{name}:within_noise"
                for name, value in preferred.items()
                if value == WITHIN_NOISE_FLOOR
            ],
        ],
        "note": (
            "the declared floors prefer the challenger on some primary criteria and neither on the "
            "rest; this is a measurement for senior review, not a partial acceptance"
        ),
    }


# ---------------------------------------------------------------------------
# Strata.
# ---------------------------------------------------------------------------


def _team_stratum_key(dimension: str) -> Callable[[Mapping[str, Any]], str]:
    if dimension == "by_venue":
        return lambda item: str(item["venue"])
    if dimension == "by_team":
        return lambda item: str(int(item["team_id"]))
    if dimension == "by_team_evidence":
        return lambda item: str(item["team_evidence_band"])
    raise ValueError(f"unknown team stratum dimension: {dimension}")


def _player_stratum_key(dimension: str) -> Callable[[Mapping[str, Any]], str]:
    if dimension == "by_position":
        return lambda item: str(int(item["position"]))
    if dimension == "by_prior_evidence":
        return lambda item: str(item["prior_evidence_band"])
    if dimension == "by_history_coverage":
        return lambda item: str(item["history_coverage_band"])
    if dimension == "by_role_change":
        return lambda item: "ROLE_CHANGE_SIGNAL" if item["role_change"] else "NO_ROLE_CHANGE_SIGNAL"
    if dimension == "by_current_exposure":
        return lambda item: str(item["current_exposure_band"])
    raise ValueError(f"unknown player stratum dimension: {dimension}")


def stratum_blocks(
    records: Sequence[Mapping[str, Any]],
    *,
    dimensions: Sequence[str],
    key_for: Callable[[str], Callable[[Mapping[str, Any]], str]],
    metrics_for: Callable[[Sequence[Mapping[str, Any]], str], dict[str, Any]],
    arms: Sequence[str],
    incumbent_arm: str,
    error_metrics: Sequence[str],
    bias_metrics: Sequence[str],
    rate_metrics: Sequence[str],
) -> dict[str, Any]:
    """Every stratum of every declared dimension, with its own sample size."""

    sections: dict[str, Any] = {}
    for dimension in dimensions:
        keyfn = key_for(dimension)
        grouped: dict[str, list[Mapping[str, Any]]] = {}
        for record in records:
            grouped.setdefault(keyfn(record), []).append(record)
        blocks: dict[str, Any] = {}
        for key in sorted(grouped):
            group = grouped[key]
            incumbent_metrics = metrics_for(group, incumbent_arm)
            n = len(group)
            arm_blocks: dict[str, Any] = {}
            for arm in arms:
                arm_metrics = metrics_for(group, arm)
                arm_blocks[arm] = {
                    "metrics": arm_metrics,
                    "comparison": comparison_block(
                        incumbent_metrics,
                        arm_metrics,
                        error_metrics=error_metrics,
                        bias_metrics=bias_metrics,
                        rate_metrics=rate_metrics,
                    ),
                }
            blocks[key] = {
                "n": n,
                "sample_interpretation": (
                    "SUFFICIENT_FOR_STRATUM_DESCRIPTIVE_REPORTING"
                    if n >= MIN_STRATUM_OBSERVATIONS
                    else "INSUFFICIENT_STRATUM_SAMPLE"
                ),
                "incumbent": incumbent_metrics,
                "arms": arm_blocks,
            }
        sections[dimension] = {
            "strata": blocks,
            "min_observations_for_stratum_reporting": MIN_STRATUM_OBSERVATIONS,
            "basis": (
                "a declared disclosure floor, NOT a significance test; a stratum below it is "
                "labelled insufficient with its n and no claim is drawn from it"
            ),
        }
    return sections


def multi_family_subset_policy(
    definitions: Mapping[str, frozenset[str]], arm_payloads: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Which multi-family subsets were evaluated, and the rule about them.

    A subset of families is a DIFFERENT model from each of its members, so its
    numbers cannot be assembled from single-family measurements.  Every declared
    multi-family arm is therefore listed with its own measurement, and a subset
    that was never evaluated is reported as ``NOT_EVALUATED``.
    """

    subsets: dict[str, Any] = {}
    for arm, families in sorted(definitions.items()):
        if len(families) < 2:
            continue
        payload = arm_payloads.get(arm) or {}
        subsets[arm] = {
            "arm": arm,
            "families": sorted(families),
            "headline": arm == team_challenger.ARM_ALL_REFINEMENTS,
            "status": "EVALUATED_AS_ITS_OWN_ARM" if payload else "NOT_EVALUATED",
            "measurement": (payload.get("measurement") if payload else None),
        }
    return {
        "rule": (
            "a multi-family subset is only reportable through an arm that evaluates it; no "
            "combination is assembled from single-family measurements, and no subset is "
            "recommended before it has its own arm"
        ),
        "declared_subsets": subsets,
        "single_family_arms_are_not_a_subset": True,
    }


# ---------------------------------------------------------------------------
# Population.
# ---------------------------------------------------------------------------


def normalize_target_events(events: Sequence[int]) -> dict[str, Any]:
    """The requested events, de-duplicated and sorted, with what changed."""

    requested = [int(value) for value in events]
    counts: dict[str, int] = {}
    for value in requested:
        counts[str(value)] = counts.get(str(value), 0) + 1
    normalized = sorted({int(value) for value in requested})
    repeated = {key: count for key, count in sorted(counts.items()) if count > 1}
    return {
        "requested": requested,
        "normalized": normalized,
        "duplicates_removed": len(requested) - len(normalized),
        "repeated_events": repeated,
        "rule": (
            "target events are de-duplicated and sorted before any projection, so a repeated id "
            "cannot add a second copy of an event's observations to any count"
        ),
    }


def team_population_digest(keys: Iterable[Sequence[Any]]) -> str:
    lines = sorted("|".join(str(int(part)) for part in key) for key in keys)
    return analytics.canonical_hash(
        {"grain": GRAIN_TEAM_FIXTURE_SIDE, "keys": lines, "version": EVALUATION_VERSION}
    )


def resolve_event(conn: sqlite3.Connection, event: int) -> dict[str, Any]:
    """The cutoff, the candidate pool and the identity provenance of one event.

    Fail closed on an event with no causal cutoff: nothing is projected, and the
    candidate count is reported as unavailable rather than enumerated from the
    mutable persisted pool.  The cutoff's TEAM evidence is resolved here too, so
    the team side can be excluded with a status when it holds none: the incumbent
    is fitted on completed fixtures of STRICTLY earlier events, so the season's
    opening event has a well-formed cutoff and no team evidence whatsoever.
    """

    cutoff, reasons = event_cutoff(conn, event)
    state, state_reasons = target_event_state(conn, event)
    block: dict[str, Any] = {
        "event": int(event),
        "cutoff": cutoff,
        "cutoff_policy": CUTOFF_POLICY,
        "cutoff_reasons": reasons,
        "outcome_state": state,
        "outcome_reasons": state_reasons,
        "fixtures": event_fixtures(conn, event),
        "identity_available": False,
        "identity_basis": None,
        "players": [],
        "unresolved": [],
        "identity_summary": None,
        "team_evidence": {
            "available": False,
            "matches": 0,
            "basis": (
                "team_model.team_match_rows: completed fixtures of strictly earlier events with "
                "official side xG, observed at or before the cutoff"
            ),
            "data_gaps": [],
        },
    }
    if cutoff is None:
        return block
    team_rows = team_model.team_match_rows(conn, int(event), cutoff)
    block["team_evidence"] = {
        "available": bool(team_rows),
        "matches": len(team_rows) // 2,
        "basis": (
            "team_model.team_match_rows: completed fixtures of strictly earlier events with "
            "official side xG, observed at or before the cutoff"
        ),
        "data_gaps": (
            []
            if team_rows
            else ["no completed fixture of an earlier event carries official side xG at this cutoff"]
        ),
    }
    resolution = identity_resolution.resolve_candidate_pool(conn, cutoff)
    block.update(
        {
            "identity_available": bool(resolution["identity_available"]),
            "identity_basis": "CUTOFF_ACCEPTED_BOOTSTRAP_GENERATION",
            "players": list(resolution["players"]),
            "unresolved": list(resolution["unresolved"]),
            "identity_summary": dict(resolution["summary"]),
            "candidate_count_available": bool(resolution["candidate_count_available"]),
        }
    )
    return block


def _player_strata(
    conn: sqlite3.Connection,
    player_id: int,
    cutoff: str,
    planning_event: int,
    incumbent_config: "player_rates.PlayerRatesConfig",
) -> dict[str, Any]:
    """The model-independent, cutoff-observable stratum keys of one candidate.

    Every key here is derived from the cutoff's own evidence — the visible
    prior-season rows, the player's own completed rows, the structured role signal
    and the cutoff identity — so the same record carries the same stratum for every
    arm, and no arm's own fitted value can decide which bucket it is measured in.
    """

    history = player_challenger.historical_season_rows_as_of(conn, int(player_id), cutoff)
    seasons = history.get("seasons") or {}
    prior_minutes = sum(
        float(row.get("minutes") or 0.0)
        for season, row in seasons.items()
        if season in player_rates.XG_BEARING_SEASONS
    )
    xg_bearing = any(
        season in player_rates.XG_BEARING_SEASONS
        and (row.get("expected_goals") is not None or row.get("expected_assists") is not None)
        for season, row in seasons.items()
    )
    if not xg_bearing:
        prior_band = PRIOR_BAND_NONE
    elif prior_minutes < PRIOR_EVIDENCE_LOW_MINUTES:
        prior_band = PRIOR_BAND_LOW
    elif prior_minutes < PRIOR_EVIDENCE_HIGH_MINUTES:
        prior_band = PRIOR_BAND_MEDIUM
    else:
        prior_band = PRIOR_BAND_HIGH
    exposure = player_challenger.current_exposure_rows(
        conn, int(player_id), player_rates.COMPONENT_XG, int(planning_event), cutoff
    )
    current_minutes = sum(float(row["minutes"]) for row in exposure["rows"])
    if current_minutes <= 0:
        exposure_band = EXPOSURE_BAND_NONE
    elif current_minutes < CURRENT_EXPOSURE_MINUTES:
        exposure_band = EXPOSURE_BAND_LOW
    else:
        exposure_band = EXPOSURE_BAND_HIGH
    notes = repo.scouting_current_rows_as_of(conn, cutoff, [int(player_id)])
    _factor, records, _flags = player_rates.role_change_modifiers(
        notes, cutoff, incumbent_config, 0.0
    )
    return {
        "prior_evidence_band": prior_band,
        "prior_minutes_visible": round(prior_minutes, 6),
        "history_coverage_band": COVERAGE_BAND_PRESENT if xg_bearing else COVERAGE_BAND_NONE,
        "current_exposure_band": exposure_band,
        "current_minutes": round(current_minutes, 6),
        "role_change": bool(records),
        "role_change_signals": sorted({str(record.get("signal")) for record in records}),
    }


def build_event_records(
    conn: sqlite3.Connection,
    event: int,
    resolution: Mapping[str, Any],
    *,
    team_config: team_challenger.TeamAttackChallengerConfig | None = None,
    team_incumbent_config: "team_model.TeamStrengthConfig | None" = None,
    player_config: player_challenger.PlayerAttackChallengerConfig | None = None,
    player_incumbent_config: "player_rates.PlayerRatesConfig | None" = None,
    team_arm_definitions: Mapping[str, frozenset[str]] | None = None,
    player_arm_definitions: Mapping[str, frozenset[str]] | None = None,
) -> dict[str, Any]:
    """One event's scorable records, exclusions, arms and coherence.

    A team-side record is scorable when the fixture was played and its official
    side xG exists.  A player record is scorable when the player's own row for
    that fixture is a real observation carrying xG and xA.  A record is dropped
    from BOTH arms when any arm leaves a component undefined, and how often that
    happened is published per arm, so the two arms always cover one identical
    population.
    """

    team_incumbent_config = team_incumbent_config or team_model.TeamStrengthConfig()
    player_incumbent_config = player_incumbent_config or player_rates.PlayerRatesConfig()
    team_config = team_config or team_challenger.TeamAttackChallengerConfig()
    player_config = player_config or player_challenger.PlayerAttackChallengerConfig()

    event = int(event)
    cutoff = resolution.get("cutoff")
    block: dict[str, Any] = {
        "event": event,
        "cutoff": cutoff,
        "team_records": [],
        "player_records": [],
        "team_excluded": [],
        "player_excluded": [],
        "team_keys": [],
        "player_keys": [],
        "team_arms": [],
        "player_arms": [],
        "team_undecided_projection_counts": {},
        "player_undecided_projection_counts": {},
        "coherence": {},
    }
    if cutoff is None:
        block["team_excluded"].append(
            {"status": STATUS_EVENT_CUTOFF_UNAVAILABLE, "scope": SCOPE_EVENT, "reasons": resolution.get("cutoff_reasons")}
        )
        block["player_excluded"].append(
            {"status": STATUS_EVENT_CUTOFF_UNAVAILABLE, "scope": SCOPE_EVENT, "reasons": resolution.get("cutoff_reasons")}
        )
        return block
    if not resolution.get("identity_available"):
        block["player_excluded"].append(
            {"status": STATUS_IDENTITY_UNAVAILABLE_AT_CUTOFF, "scope": SCOPE_EVENT}
        )
    for candidate in resolution.get("unresolved") or []:
        block["player_excluded"].append(
            {
                "status": STATUS_IDENTITY_UNRESOLVED_AT_CUTOFF,
                "scope": SCOPE_PLAYER,
                "player_id": candidate.get("player_id"),
                "reasons": candidate.get("notes"),
            }
        )

    fixtures = [dict(fixture) for fixture in resolution.get("fixtures") or []]
    fixture_sides = {
        int(fixture["id"]): (int(fixture["team_h"]), int(fixture["team_a"])) for fixture in fixtures
    }

    # --- team side ---------------------------------------------------------
    team_arms = None
    team_evidence = dict(resolution.get("team_evidence") or {})
    team_projectable = bool(team_evidence.get("available"))
    # Published on every event, projectable or not, so the reason a team population
    # is missing is visible where the population itself is reported.
    block["team_evidence"] = team_evidence
    if team_projectable:
        try:
            team_arms = team_challenger.build_challenger_arms(
                conn,
                event,
                cutoff,
                incumbent_config=team_incumbent_config,
                challenger_config=team_config,
                arm_definitions=team_arm_definitions,
            )
            team_arms.verify_same_population()
            block["team_arms"] = list(team_arms.arm_names())
            block["team_identity"] = team_arms.identity
        except team_challenger.ChallengerInconsistencyError as exc:
            raise EvaluationError(str(exc)) from exc
    # No completed-fixture xG at this cutoff means no team evidence at all, so there
    # is no level to fit and no honest fixture attack to publish.  The incumbent is
    # not asked to project (its ``lambda_for`` is undefined on an empty evidence
    # set), nothing is scored as a zero, and every fixture side below is excluded
    # with the reason.

    team_match_counts: dict[int, int] = {}
    for row in team_model.team_match_rows(conn, event, cutoff):
        team_match_counts[int(row["team_id"])] = team_match_counts.get(int(row["team_id"]), 0) + 1

    for fixture_id in sorted(fixture_sides):
        for team_id in fixture_sides[fixture_id]:
            key = (event, fixture_id, int(team_id))
            if not team_projectable:
                block["team_excluded"].append(
                    {
                        "status": STATUS_TEAM_EVIDENCE_UNAVAILABLE,
                        "scope": SCOPE_TEAM_SIDE,
                        "fixture_id": fixture_id,
                        "team_id": int(team_id),
                        "reasons": list(team_evidence.get("data_gaps") or []),
                    }
                )
                continue
            realised, status = realised_team_side(conn, fixture_id, int(team_id))
            if realised is None:
                block["team_excluded"].append(
                    {
                        "status": status,
                        "scope": SCOPE_TEAM_SIDE,
                        "fixture_id": fixture_id,
                        "team_id": int(team_id),
                    }
                )
                continue
            incumbent_row = team_arms.incumbent_rows.get((fixture_id, int(team_id)))
            predictions: dict[str, dict[str, Any]] = {}
            missing_arms: list[str] = []
            if incumbent_row is None:
                missing_arms.append("incumbent")
            else:
                predictions["incumbent"] = {
                    "xg": incumbent_row.get("expected_goals_for"),
                    "goals": incumbent_row.get("expected_goals_for"),
                }
            for arm in team_arms.arm_names():
                row = team_arms.row(arm, (fixture_id, int(team_id)))
                if row is None:
                    missing_arms.append(arm)
                    continue
                predictions[arm] = {
                    "xg": row.get("expected_goals_for"),
                    "goals": row.get("expected_goals_for"),
                }
            if missing_arms:
                block["team_excluded"].append(
                    {
                        "status": STATUS_MODEL_PROJECTION_MISSING,
                        "scope": SCOPE_TEAM_SIDE,
                        "fixture_id": fixture_id,
                        "team_id": int(team_id),
                        "arms": sorted(missing_arms),
                    }
                )
                for arm in missing_arms:
                    counts = block["team_undecided_projection_counts"]
                    counts[arm] = counts.get(arm, 0) + 1
                continue
            block["team_records"].append(
                {
                    "event": event,
                    "fixture_id": fixture_id,
                    "team_id": int(team_id),
                    "opponent_id": int(realised["opponent_id"]),
                    "venue": str(realised["venue"]),
                    "team_evidence_band": (
                        TEAM_EVIDENCE_LOW
                        if team_match_counts.get(int(team_id), 0) < team_incumbent_config.low_evidence_matches
                        else TEAM_EVIDENCE_SUFFICIENT
                    ),
                    "team_matches_used": int(team_match_counts.get(int(team_id), 0)),
                    "realised": {"xg": realised["xg"], "goals": realised["goals"]},
                    "predicted": predictions,
                }
            )
            block["team_keys"].append(key)

    # --- player side -------------------------------------------------------
    player_arms = None
    if resolution.get("identity_available") and resolution.get("players"):
        players = list(resolution["players"])
        identities = {
            int(player["player_id"]): (int(player["team_id"]), int(player["element_type"]))
            for player in players
        }
        player_arms = player_challenger.build_challenger_player_arms(
            conn,
            event,
            cutoff,
            players=players,
            identities=identities,
            incumbent_config=player_incumbent_config,
            challenger_config=player_config,
            arm_definitions=player_arm_definitions,
        )
        player_arms.verify_same_population()
        block["player_arms"] = list(player_arms.arm_names())
        block["player_identity"] = player_arms.identity
        block["pool_disclosure"] = dict(player_arms.pool_disclosure)
        block["incumbent_arm"] = dict(player_arms.incumbent_arm)
        block["ess_estimates"] = dict(player_arms.ess_estimates)
        block["candidate_players"] = len(players)

        for player in sorted(players, key=lambda row: int(row["player_id"])):
            player_id = int(player["player_id"])
            team_id = int(player["team_id"])
            strata = _player_strata(conn, player_id, cutoff, event, player_incumbent_config)
            played = [
                fixture_id
                for fixture_id, (home, away) in fixture_sides.items()
                if team_id in (home, away)
            ]
            if not played:
                block["player_excluded"].append(
                    {
                        "status": STATUS_BLANK_NO_FIXTURE,
                        "scope": SCOPE_PLAYER,
                        "player_id": player_id,
                    }
                )
                continue
            incumbent_rows = {
                component: player_arms.incumbent_rows.get((player_id, component))
                for component in player_rates.COMPONENTS
            }
            arm_rows = {
                arm: {
                    component: player_arms.arms[arm].get((player_id, component))
                    for component in player_rates.COMPONENTS
                }
                for arm in player_arms.arm_names()
            }
            for fixture_id in played:
                realised, status = realised_player_fixture(conn, player_id, fixture_id)
                if realised is None:
                    block["player_excluded"].append(
                        {
                            "status": status,
                            "scope": SCOPE_PLAYER,
                            "player_id": player_id,
                            "fixture_id": fixture_id,
                        }
                    )
                    continue
                if realised["minutes"] <= 0.0:
                    # A player who did not play has no realised exposure, so there
                    # is no rate observation to score; the row is counted, never
                    # scored as a zero.
                    block["player_excluded"].append(
                        {
                            "status": STATUS_NO_REALISED_EXPOSURE,
                            "scope": SCOPE_PLAYER,
                            "player_id": player_id,
                            "fixture_id": fixture_id,
                        }
                    )
                    continue
                predictions: dict[str, dict[str, Any]] = {}
                missing_arms: list[str] = []
                rows_by_arm: dict[str, Mapping[Any, Any]] = {
                    "incumbent": incumbent_rows,
                    **{arm: rows for arm, rows in arm_rows.items()},
                }
                for arm in sorted(rows_by_arm):
                    rows = rows_by_arm[arm]
                    undefined = [
                        component
                        for component in player_rates.COMPONENTS
                        if (rows.get(component) or {}).get("posterior_mean") is None
                    ]
                    if undefined:
                        missing_arms.append(f"{arm}:{'+'.join(sorted(undefined))}")
                        continue
                    predictions[arm] = {
                        component: (
                            float(rows[component]["posterior_mean"]) * realised["minutes"] / 90.0
                        )
                        for component in player_rates.COMPONENTS
                    }
                if missing_arms:
                    block["player_excluded"].append(
                        {
                            "status": STATUS_MODEL_PROJECTION_MISSING,
                            "scope": SCOPE_PLAYER,
                            "player_id": player_id,
                            "fixture_id": fixture_id,
                            "arms": sorted(missing_arms),
                        }
                    )
                    for arm in missing_arms:
                        counts = block["player_undecided_projection_counts"]
                        counts[arm] = counts.get(arm, 0) + 1
                    continue
                block["player_records"].append(
                    {
                        "event": event,
                        "fixture_id": fixture_id,
                        "player_id": player_id,
                        "team_id": team_id,
                        "position": int(player["element_type"]),
                        "venue": (
                            "home" if int(fixture_sides[fixture_id][0]) == team_id else "away"
                        ),
                        "prior_evidence_band": strata["prior_evidence_band"],
                        "history_coverage_band": strata["history_coverage_band"],
                        "current_exposure_band": strata["current_exposure_band"],
                        "role_change": bool(strata["role_change"]),
                        "realised": {
                            "xg": realised["xg"],
                            "xa": realised["xa"],
                            "minutes": realised["minutes"],
                            "goals": realised["goals"],
                            "assists": realised["assists"],
                        },
                        "predicted": {
                            "incumbent": {
                                "xg": predictions["incumbent"][player_rates.COMPONENT_XG],
                                "xa": predictions["incumbent"][player_rates.COMPONENT_XA],
                            },
                            **{
                                arm: {
                                    "xg": predictions[arm][player_rates.COMPONENT_XG],
                                    "xa": predictions[arm][player_rates.COMPONENT_XA],
                                }
                                for arm in player_arms.arm_names()
                            },
                        },
                        "strata": strata,
                    }
                )
                block["player_keys"].append((event, player_id, fixture_id))
    block["player_key_count"] = len(block["player_keys"])

    if team_arms is not None and player_arms is not None:
        block["coherence"] = coherence_for_event(conn, resolution, team_arms, player_arms)
    elif team_arms is not None:
        block["coherence"] = {
            "grain": coherence.GRAIN_FIXTURE_SIDE,
            "mass_rule": coherence.MASS_FORMULA,
            "redistribution": coherence.REDISTRIBUTION_NONE,
            "pairings": [],
            "arms": {},
            "summary": {},
            "note": (
                "no player projections exist for this event, so no team -> player attacking-mass "
                "comparison is made and none is invented"
            ),
        }
    else:
        block["coherence"] = {
            "grain": coherence.GRAIN_FIXTURE_SIDE,
            "mass_rule": coherence.MASS_FORMULA,
            "redistribution": coherence.REDISTRIBUTION_NONE,
            "pairings": [],
            "arms": {},
            "summary": {},
            "note": (
                "the team environment is unavailable at this cutoff, so there is no team attack "
                "expectation to check player mass against; the comparison is omitted explicitly "
                "rather than reported as coherent because it measured nothing"
            ),
            "team_evidence": team_evidence,
        }
    return block


def _arm_payload(
    records: Sequence[Mapping[str, Any]],
    arm: str,
    *,
    metrics_for: Callable[[Sequence[Mapping[str, Any]], str], dict[str, Any]],
    comparison_kwargs: Mapping[str, Any],
    primary_criteria: Sequence[str],
    sample_sufficient: bool,
) -> dict[str, Any]:
    """One arm's metrics, its comparison with the incumbent, and its measurement."""

    incumbent = metrics_for(records, "incumbent")
    challenger = metrics_for(records, arm)
    comparison = comparison_block(incumbent, challenger, **comparison_kwargs)
    return {
        "metrics": challenger,
        "incumbent_metrics": incumbent,
        "comparison": comparison,
        "measurement": arm_measurement(
            comparison, primary_criteria=primary_criteria, sample_sufficient=sample_sufficient
        ),
    }


# ---------------------------------------------------------------------------
# Team -> player coherence, per event and per arm.
# ---------------------------------------------------------------------------


def event_attack_mass(
    conn: sqlite3.Connection,
    resolution: Mapping[str, Any],
    player_rows: Sequence[Mapping[str, Any]],
    *,
    rate_field: str = "posterior_mean",
) -> list[dict[str, Any]]:
    """Player attacking mass for one event, from the arm's own rates.

    Exposure is the REALISED minutes of the target fixture, read through the same
    ``realised_player_fixture`` the metric path uses, so predicted-minutes error
    cannot move the coherence check and a player who did not play contributes
    nothing at all.

    The resolution MUST carry the event's fixture list.  One that does not is
    refused rather than read as "no fixtures": with no sides there would be no
    exposures and no mass, and the block would then report perfect coherence
    because it measured nothing.
    """

    if "fixtures" not in resolution:
        raise EvaluationError(
            "the coherence block requires a resolution carrying the event's FIXTURES; a "
            "resolution without one would report zero mass as coherence"
        )
    fixtures = [dict(fixture) for fixture in resolution.get("fixtures") or []]
    fixture_sides = {
        int(fixture["id"]): (int(fixture["team_h"]), int(fixture["team_a"])) for fixture in fixtures
    }
    exposures: dict[tuple[int, int], float] = {}
    for player in resolution.get("players") or []:
        player_id = int(player["player_id"])
        team_id = int(player["team_id"])
        for fixture_id, (home, away) in fixture_sides.items():
            if team_id not in (home, away):
                continue
            realised, _status = realised_player_fixture(conn, player_id, fixture_id)
            if realised is not None and realised["minutes"] > 0:
                exposures[(player_id, fixture_id)] = float(realised["minutes"])
    rows = []
    team_by_player = {
        int(player["player_id"]): int(player["team_id"])
        for player in (resolution.get("players") or [])
    }
    for row in player_rows:
        player_id = int(row["player_id"])
        team_id = team_by_player.get(player_id)
        if team_id is None:
            # No cutoff club means no side to attribute mass to, so no mass: the
            # club is never read from the row's own (mutable) context.
            continue
        rows.append(
            {
                "player_id": player_id,
                "component": str(row["component"]),
                "team_id": team_id,
                rate_field: row.get(rate_field),
            }
        )
    return coherence.player_attack_mass(rows, exposures, fixture_sides)


def _team_rows_for_arm(
    team_arms: "team_challenger.TeamChallengerArms", arm: str
) -> list[dict[str, Any]]:
    source = team_arms.incumbent_rows if arm == "incumbent" else team_arms.arms[arm]
    return [dict(row) for _key, row in sorted(source.items())]


def _player_rows_for_arm(
    player_arms: "player_challenger.PlayerChallengerArms", arm: str
) -> list[dict[str, Any]]:
    source = player_arms.incumbent_rows if arm == "incumbent" else player_arms.arms[arm]
    return [dict(row) for _key, row in sorted(source.items())]


def declared_coherence_pairings(
    team_arms: Sequence[str], player_arms: Sequence[str]
) -> list[tuple[str, str, str]]:
    """Which team arm is checked against which player arm, declared not inferred.

    Coherence crosses the two families, so a pairing is a claim about which team
    environment a player level is supposed to arise from.  Three rules, in order:

    * the frozen pair — the incumbent team model against the incumbent player model;
    * the headline pair — the full challenger of each family against the other;
    * every single-family ablation against the OTHER family's incumbent, labelled so
      a reader cannot mistake an ablation for the headline pair.

    The two families' arm names coincide ("challenger_all_refinements" and
    "challenger_only::<family>"), which is exactly why the pairing is published as
    data on every block instead of being left to the reader's assumption.
    """

    team_all = team_challenger.ARM_ALL_REFINEMENTS
    player_all = player_challenger.ARM_ALL_REFINEMENTS
    pairings: list[tuple[str, str, str]] = [("incumbent", "incumbent", "incumbent")]
    team_set = set(team_arms)
    player_set = set(player_arms)
    if team_all in team_set and player_all in player_set:
        pairings.append((team_all, player_all, team_all))
    for arm in sorted(team_set - {team_all}):
        pairings.append((arm, "incumbent", f"{arm}__with_incumbent_player_rates"))
    for arm in sorted(player_set - {player_all}):
        pairings.append(("incumbent", arm, f"{arm}__with_incumbent_team_environment"))
    return pairings


def coherence_for_event(
    conn: sqlite3.Connection,
    resolution: Mapping[str, Any],
    team_arms: "team_challenger.TeamChallengerArms",
    player_arms: "player_challenger.PlayerChallengerArms",
    pairings: Sequence[tuple[str, str, str]] | None = None,
) -> dict[str, Any]:
    """The coherence block of every declared pairing, on the event's fixtures."""

    fixtures = {
        int(fixture["id"]): dict(fixture) for fixture in (resolution.get("fixtures") or [])
    }
    pairs = list(
        pairings or declared_coherence_pairings(team_arms.arm_names(), player_arms.arm_names())
    )
    blocks: dict[str, Any] = {}
    for team_arm, player_arm, pairing_id in pairs:
        mass = event_attack_mass(conn, resolution, _player_rows_for_arm(player_arms, player_arm))
        block = coherence.attacking_mass_allocation(
            _team_rows_for_arm(team_arms, team_arm), mass, fixtures=fixtures
        )
        block["pairing"] = {
            "team_arm": team_arm,
            "player_arm": player_arm,
            "rule": (
                "the team environment this player level is checked against; a single-family arm is "
                "paired with the OTHER family's incumbent, never with another ablation"
            ),
        }
        blocks[pairing_id] = block
    return {
        "grain": coherence.GRAIN_FIXTURE_SIDE,
        "mass_rule": coherence.MASS_FORMULA,
        "redistribution": coherence.REDISTRIBUTION_NONE,
        "pairings": [
            {"id": pairing_id, "team_arm": team_arm, "player_arm": player_arm}
            for team_arm, player_arm, pairing_id in pairs
        ],
        "arms": blocks,
        "summary": {
            pairing_id: {
                "pairing": {"team_arm": block["pairing"]["team_arm"], "player_arm": block["pairing"]["player_arm"]},
                "status": block["status"],
                "digest": block["digest"],
                "checks": block["checks"],
                "exceeds_environment": block["exceeds_environment"],
                "invented_fixture_attack": block["invented_fixture_attack"],
            }
            for pairing_id, block in sorted(blocks.items())
        },
    }


# ---------------------------------------------------------------------------
# Population accounting and sample policy.
# ---------------------------------------------------------------------------


def _candidate_slots_for_event(
    resolution: Mapping[str, Any],
) -> tuple[int | None, int | None]:
    """The enumerated candidate slots for one event, or ``None`` when unavailable.

    A candidate COUNT exists only where a candidate POOL exists.  With no causal
    cutoff at all there is no pool on either side, so both counts are null: the
    fixture list alone is not a population, and scoring nothing while enumerating
    something would make the accounting describe a comparison that never happened.

    The player-side count additionally needs the cutoff's OFFICIAL pool identity;
    without it, the count is null rather than read off the mutable players table.
    The team-side count needs no identity at all, because a team side is a fixture
    side and is enumerated from the fixture list itself.
    """

    if resolution.get("cutoff") is None:
        return None, None
    fixtures = {
        int(fixture["id"]): (int(fixture["team_h"]), int(fixture["team_a"]))
        for fixture in (resolution.get("fixtures") or [])
    }
    team_slots = 2 * len(fixtures)
    if not resolution.get("identity_available"):
        return team_slots, None
    total = 0
    for player in resolution.get("players") or []:
        team_id = int(player["team_id"])
        played = sum(1 for pair in fixtures.values() if team_id in pair)
        total += played if played else 1
    return team_slots, total


def _status_counts(exclusions: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in exclusions:
        status = str(item.get("status"))
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


def _family_metrics_for(family: str) -> Callable[[Sequence[Mapping[str, Any]], str], dict[str, Any]]:
    return team_arm_metrics if family == "team" else player_arm_metrics


def _comparison_kwargs(family: str) -> dict[str, Any]:
    if family == "team":
        return {
            "error_metrics": TEAM_ERROR_METRICS,
            "bias_metrics": TEAM_BIAS_METRICS,
            "rate_metrics": (
                "team_xg_predicted_mean",
                "team_xg_realised_mean",
                "team_goals_predicted_mean",
                "team_goals_realised_mean",
            ),
        }
    return {
        "error_metrics": PLAYER_ERROR_METRICS,
        "bias_metrics": PLAYER_BIAS_METRICS,
        "rate_metrics": (
            "player_xg_predicted_mean",
            "player_xg_realised_mean",
            "player_xa_predicted_mean",
            "player_xa_realised_mean",
            "realised_minutes_mean",
        ),
    }


def _sample_block(
    target_events: Sequence[int],
    evaluated_events: set[int],
    observations: int,
) -> dict[str, Any]:
    events = len(target_events)
    sufficient = (
        events >= wfs.MIN_TARGET_EVENTS_FOR_DESCRIPTIVE
        and observations >= wfs.MIN_OBSERVATIONS_FOR_DESCRIPTIVE
    )
    return {
        "target_events": events,
        "target_events_with_observations": len(evaluated_events),
        "observations": observations,
        "sample_interpretation": (
            wfs.SAMPLE_DESCRIPTIVE_ONLY if sufficient else wfs.SAMPLE_INSUFFICIENT
        ),
        "sufficient_for_descriptive_reporting": sufficient,
        "policy": {
            "policy_version": wfs.SAMPLE_POLICY_VERSION,
            "min_target_events_for_descriptive_reporting": wfs.MIN_TARGET_EVENTS_FOR_DESCRIPTIVE,
            "min_observations_for_descriptive_reporting": wfs.MIN_OBSERVATIONS_FOR_DESCRIPTIVE,
            "basis": (
                "a declared disclosure floor, NOT a significance test; no arm is described as "
                "better, superior, proven or calibrated, and no skill score is derived"
            ),
        },
    }


def _accounting(
    enumerated: int | None, scored: int, exclusions: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    counts = _status_counts(exclusions)
    total = sum(counts.values())
    return {
        "enumerated_candidate_slots": enumerated,
        "scored_rows": scored,
        "excluded_candidates": total,
        "excluded_by_status": counts,
        "excluded_by_status_total": total,
        "reconciles": (enumerated is None) or (scored + total == enumerated),
        "rule": (
            "an excluded candidate is COUNTED with its status and is never scored as a zero; an "
            "unavailable enumeration is reported as null rather than as zero"
        ),
    }


# ---------------------------------------------------------------------------
# The artifact.
# ---------------------------------------------------------------------------


def artifact_digest(artifact: Mapping[str, Any]) -> str:
    """A digest over the artifact with its volatile fields stripped."""

    def strip(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {
                key: strip(item)
                for key, item in sorted(value.items())
                if key not in VOLATILE_ARTIFACT_KEYS
            }
        if isinstance(value, (list, tuple)):
            return [strip(item) for item in value]
        return value

    return analytics.canonical_hash(strip(artifact))


def evaluate_events(
    conn: sqlite3.Connection,
    events: Sequence[int],
    *,
    team_config: "team_challenger.TeamAttackChallengerConfig | None" = None,
    team_incumbent_config: "team_model.TeamStrengthConfig | None" = None,
    player_config: "player_challenger.PlayerAttackChallengerConfig | None" = None,
    player_incumbent_config: "player_rates.PlayerRatesConfig | None" = None,
    team_arm_definitions: Mapping[str, frozenset[str]] | None = None,
    player_arm_definitions: Mapping[str, frozenset[str]] | None = None,
) -> dict[str, Any]:
    """The PE-7 evaluation artifact for the requested target events.

    Read only: no projection run is written, no version identifier is re-pointed
    and no arm is accepted.  The artifact measures incumbent and challenger on one
    identical population per family and stops there.
    """

    team_incumbent_config = team_incumbent_config or team_model.TeamStrengthConfig()
    player_incumbent_config = player_incumbent_config or player_rates.PlayerRatesConfig()
    team_config = team_config or team_challenger.TeamAttackChallengerConfig()
    player_config = player_config or player_challenger.PlayerAttackChallengerConfig()
    team_definitions = dict(team_arm_definitions or team_challenger.default_arm_definitions())
    player_definitions = dict(player_arm_definitions or player_challenger.default_arm_definitions())
    normalized = normalize_target_events(events)

    event_blocks: list[dict[str, Any]] = []
    team_records: list[dict[str, Any]] = []
    player_records: list[dict[str, Any]] = []
    team_keys: list[tuple[int, int, int]] = []
    player_keys: list[tuple[int, int, int]] = []
    team_evaluated: set[int] = set()
    player_evaluated: set[int] = set()
    team_enumerated = 0
    team_unavailable_enumeration = 0
    player_enumerated = 0
    player_unavailable_enumeration = 0
    team_exclusions: list[dict[str, Any]] = []
    player_exclusions: list[dict[str, Any]] = []
    coherence_by_event: dict[str, Any] = {}
    team_arm_names: set[str] = set()
    player_arm_names: set[str] = set()

    for event in normalized["normalized"]:
        resolution = resolve_event(conn, event)
        block = build_event_records(
            conn,
            event,
            resolution,
            team_config=team_config,
            team_incumbent_config=team_incumbent_config,
            player_config=player_config,
            player_incumbent_config=player_incumbent_config,
            team_arm_definitions=team_definitions,
            player_arm_definitions=player_definitions,
        )
        block["event"] = int(event)
        block["cutoff"] = resolution.get("cutoff")
        block["fixtures"] = [int(fixture["id"]) for fixture in resolution.get("fixtures") or []]
        block["identity_summary"] = resolution.get("identity_summary")
        slots_team, slots_player = _candidate_slots_for_event(resolution)
        if slots_team is None:
            team_unavailable_enumeration += 1
        else:
            team_enumerated += slots_team
        if slots_player is None:
            player_unavailable_enumeration += 1
        else:
            player_enumerated += slots_player
        event_blocks.append(block)
        team_records.extend(block["team_records"])
        player_records.extend(block["player_records"])
        team_keys.extend(block["team_keys"])
        player_keys.extend(block["player_keys"])
        team_exclusions.extend(block["team_excluded"])
        player_exclusions.extend(block["player_excluded"])
        if block["team_records"]:
            team_evaluated.add(int(event))
        if block["player_records"]:
            player_evaluated.add(int(event))
        team_arm_names.update(block["team_arms"])
        player_arm_names.update(block["player_arms"])
        coherence_by_event[str(int(event))] = block.get("coherence") or {}

    team_arms = sorted(team_arm_names) or sorted(team_definitions)
    player_arms = sorted(player_arm_names) or sorted(player_definitions)
    for records, arms in ((team_records, team_arms), (player_records, player_arms)):
        expected = {"incumbent"} | set(arms)
        for record in records:
            if set(record["predicted"]) != expected:
                raise EvaluationError(
                    "an arm left a record undefined without the record being excluded: within a "
                    "family the arms must cover one identical population, or they cannot be compared"
                )

    team_sample = _sample_block(normalized["normalized"], team_evaluated, len(team_records))
    player_sample = _sample_block(normalized["normalized"], player_evaluated, len(player_records))

    team_payloads = {
        arm: _arm_payload(
            team_records,
            arm,
            metrics_for=team_arm_metrics,
            comparison_kwargs=_comparison_kwargs("team"),
            primary_criteria=PRIMARY_CRITERIA_TEAM,
            sample_sufficient=team_sample["sufficient_for_descriptive_reporting"],
        )
        for arm in team_arms
    }
    player_payloads = {
        arm: _arm_payload(
            player_records,
            arm,
            metrics_for=player_arm_metrics,
            comparison_kwargs=_comparison_kwargs("player"),
            primary_criteria=PRIMARY_CRITERIA_PLAYER,
            sample_sufficient=player_sample["sufficient_for_descriptive_reporting"],
        )
        for arm in player_arms
    }

    same_population = all(
        set(record["predicted"]) == expected
        for records, expected in (
            (team_records, {"incumbent"} | set(team_arms)),
            (player_records, {"incumbent"} | set(player_arms)),
        )
        for record in records
    )
    team_incumbent = team_arm_metrics(team_records, "incumbent")
    player_incumbent = player_arm_metrics(player_records, "incumbent")

    artifact = {
        "schema": ARTIFACT_SCHEMA,
        "evaluation_version": EVALUATION_VERSION,
        "missing_data_policy_version": MISSING_DATA_POLICY_VERSION,
        "cutoff_policy": CUTOFF_POLICY,
        "generated_at": utc_now(),
        "identity": {
            "frozen_incumbents": frozen_incumbent_identity(),
            "team_challenger": team_challenger.challenger_identity(team_config, team_incumbent_config),
            "player_challenger": player_challenger.challenger_identity(
                player_config, player_incumbent_config
            ),
            "coherence_version": coherence.COHERENCE_VERSION,
            "team_incumbent_config_hash": team_incumbent_config.config_hash(),
            "player_incumbent_config_hash": player_incumbent_config.config_hash(),
            "player_incumbent_arm_construction": player_challenger.INCUMBENT_ARM_CONSTRUCTION,
            "player_incumbent_arm_boundary": player_challenger.INCUMBENT_ARM_BOUNDARY,
            "promotion": "NOT_PERFORMED_THIS_ARTIFACT_MEASURES_ONLY",
            "comparator_note": team_challenger.COMPARATOR_NOTE,
        },
        "events": [
            {
                "event": block["event"],
                "cutoff": block["cutoff"],
                "fixtures": block["fixtures"],
                "identity_summary": block["identity_summary"],
                "team_evidence": block.get("team_evidence"),
                "pool_disclosure": block.get("pool_disclosure"),
                "ess_estimates": block.get("ess_estimates"),
                "team_arms": block["team_arms"],
                "player_arms": block["player_arms"],
                "incumbent_arm": block.get("incumbent_arm"),
                "team_undecided_projection_counts": block["team_undecided_projection_counts"],
                "player_undecided_projection_counts": block["player_undecided_projection_counts"],
                "excluded_by_status": {
                    "team": _status_counts(block["team_excluded"]),
                    "player": _status_counts(block["player_excluded"]),
                },
            }
            for block in event_blocks
        ],
        "population": {
            "target_events": normalized["normalized"],
            "normalization": normalized,
            "rule": (
                "a record is scorable only when EVERY arm produced a value for it, so the arms "
                "cover one identical key set; how often an arm was undefined is published rather "
                "than silently narrowing the comparison"
            ),
            "exclusion_statuses": list(EXCLUSION_STATUSES),
            "never_scored_as_zero": [
                "an event with no causal cutoff",
                "a blank: a team with no fixture in the target event",
                "a cutoff that holds no completed-fixture team xG, so the team model has no evidence",
                "a fixture that has not started and finished",
                "a target fixture that is missing from the store",
                "a fixture side whose official xG is missing",
                "a scheduled placeholder row",
                "a missing official outcome row",
                "an official outcome row with no xG-family value",
                "a player who did not play, so there is no realised exposure to score a rate on",
                "a record an arm left undefined",
                "a candidate whose club and position the cutoff cannot resolve",
                "a candidate the cutoff's official pool generation does not place in the pool",
            ],
            "team": {
                "grain": GRAIN_TEAM_FIXTURE_SIDE,
                "grain_note": GRAIN_TEAM_NOTE,
                "candidate_slots": None if team_unavailable_enumeration else team_enumerated,
                "candidate_count_available": team_unavailable_enumeration == 0,
                "events_without_a_candidate_count": team_unavailable_enumeration,
                "scored": len(team_records),
                "accounting": _accounting(
                    None if team_unavailable_enumeration else team_enumerated,
                    len(team_records),
                    team_exclusions,
                ),
                "excluded_by_status": _status_counts(team_exclusions),
                "population_digest": team_population_digest(team_keys),
                "arms_scored": team_arms,
                "undecided_projection_counts": _merge_undecided(event_blocks, "team"),
            },
            "player": {
                "grain": GRAIN_PLAYER_FIXTURE,
                "grain_note": GRAIN_PLAYER_NOTE,
                "candidate_slots": (
                    None if player_unavailable_enumeration else player_enumerated
                ),
                "candidate_count_available": player_unavailable_enumeration == 0,
                "events_without_a_candidate_count": player_unavailable_enumeration,
                "scored": len(player_records),
                "accounting": _accounting(
                    None if player_unavailable_enumeration else player_enumerated,
                    len(player_records),
                    player_exclusions,
                ),
                "excluded_by_status": _status_counts(player_exclusions),
                "population_digest": wf.canonical_population_digest(
                    player_keys, grain=GRAIN_PLAYER_FIXTURE
                ),
                "arms_scored": player_arms,
                "undecided_projection_counts": _merge_undecided(event_blocks, "player"),
            },
            "same_population": same_population,
            "same_population_rule": (
                "incumbent and every challenger arm are scored on the key set above; the assertion "
                "is PE-2's own, re-run per arm inside each event build: assert_same_population "
                "for the team arms and PopulationMismatch for the player arms, whose keys carry a "
                "component name as well as integer ids"
            ),
        },
        "team": {
            "primary_target": "stored official realised team-side xG",
            "secondary_target": "actual goals (descriptive only; it may not replace xG)",
            "secondary_rule": (
                "the goals check compares the SAME expectation the xG check uses: the model "
                "estimates a scoring environment, so the goals comparison is an outcome-level "
                "sanity check and never a second model"
            ),
            "incumbent_metrics": team_incumbent,
            "arms": team_payloads,
            "strata": stratum_blocks(
                team_records,
                dimensions=TEAM_STRATUM_DIMENSIONS,
                key_for=_team_stratum_key,
                metrics_for=team_arm_metrics,
                arms=team_arms,
                incumbent_arm="incumbent",
                **_comparison_kwargs("team"),
            ),
            "measurement_rule": measurement_rule_block(PRIMARY_CRITERIA_TEAM),
            "raw_realised_rule": (
                "realised side xG is read with team_model.realised_fixture_side_xg, which is "
                "deliberately OUTSIDE the historical boundary: a calibration read compares what "
                "was said before a match with what the match produced"
            ),
            "sample": team_sample,
            "multi_family_subset_policy": multi_family_subset_policy(team_definitions, team_payloads),
        },
        "player": {
            "primary_target": "player xG/90 and xA/90 scored on the player's REALISED exposure",
            "exposure_rule": (
                "expected xG/xA for the target fixture = posterior xG/90 * realised minutes / 90, "
                "so PE-6 minutes uncertainty cannot dominate the rate comparison"
            ),
            "penalties": "embedded in xG; no NPxG separation and none fabricated",
            "incumbent_arm": {
                "construction": player_challenger.INCUMBENT_ARM_CONSTRUCTION,
                "boundary": player_challenger.INCUMBENT_ARM_BOUNDARY,
                "equivalence": player_challenger.INCUMBENT_ARM_EQUIVALENCE,
                "note": (
                    "the arm every challenger is scored against is the frozen incumbent's own "
                    "arithmetic over the CUTOFF'S OWN evidence, so a post-cutoff season write or a "
                    "later identity change cannot move the baseline the delta is measured against"
                ),
            },
            "incumbent_metrics": player_incumbent,
            "arms": player_payloads,
            "strata": stratum_blocks(
                player_records,
                dimensions=PLAYER_STRATUM_DIMENSIONS,
                key_for=_player_stratum_key,
                metrics_for=player_arm_metrics,
                arms=player_arms,
                incumbent_arm="incumbent",
                **_comparison_kwargs("player"),
            ),
            "measurement_rule": measurement_rule_block(PRIMARY_CRITERIA_PLAYER),
            "sample": player_sample,
            "multi_family_subset_policy": multi_family_subset_policy(
                player_definitions, player_payloads
            ),
        },
        "coherence": {
            "version": coherence.COHERENCE_VERSION,
            "grain": coherence.GRAIN_FIXTURE_SIDE,
            "mass_rule": coherence.MASS_FORMULA,
            "redistribution": coherence.REDISTRIBUTION_NONE,
            "declared_tolerance": coherence.DECLARED_MASS_TOLERANCE,
            "per_event": coherence_by_event,
            "checks": _coherence_summary(coherence_by_event),
        },
        "xpts_diagnostic": {
            "status": XPTS_DIAGNOSTIC_STATUS,
            "reason": XPTS_DIAGNOSTIC_REASON,
            "may_not_be_used_for_promotion": True,
        },
        "sample": {
            "target_events": len(normalized["normalized"]),
            "team": team_sample,
            "player": player_sample,
            "headline_interpretation": (
                wfs.SAMPLE_DESCRIPTIVE_ONLY
                if team_sample["sufficient_for_descriptive_reporting"]
                and player_sample["sufficient_for_descriptive_reporting"]
                else wfs.SAMPLE_INSUFFICIENT
            ),
            "headline_basis": (
                "the weaker of the two families decides: a sample that supports descriptive "
                "reporting on one side does not license a claim on the other"
            ),
        },
        "causality": {
            "audit": [dict(entry) for entry in CAUSALITY_AUDIT],
            "discipline": dict(CAUSALITY_DISCIPLINE),
            "audited_surface_count": len(CAUSALITY_AUDIT),
        },
        "claims": {
            "selection_claim": SELECTION_CLAIM_NOT_MADE,
            "allowed": list(CLAIMS_ALLOWED),
            "not_allowed": list(CLAIMS_NOT_ALLOWED),
            "note": (
                "this artifact measures; senior review decides.  NO CHANGE is a legitimate PE-7 "
                "outcome and nothing here promotes, calibrates or certifies anything."
            ),
        },
    }
    artifact["artifact_digest"] = artifact_digest(artifact)
    return artifact


def _merge_undecided(event_blocks: Sequence[Mapping[str, Any]], family: str) -> dict[str, int]:
    """How often an arm left a record undefined, summed over the target events."""

    key = (
        "team_undecided_projection_counts"
        if family == "team"
        else "player_undecided_projection_counts"
    )
    merged: dict[str, int] = {}
    for block in event_blocks:
        for arm, count in (block.get(key) or {}).items():
            merged[str(arm)] = merged.get(str(arm), 0) + int(count)
    return dict(sorted(merged.items()))


def _coherence_summary(per_event: Mapping[str, Any]) -> dict[str, Any]:
    """Per-arm coherence over every event, with the aggregated pass/fail facts."""

    arms: dict[str, Any] = {}
    for event, block in per_event.items():
        for arm, summary in (block or {}).get("summary", {}).items():
            entry = arms.setdefault(
                arm,
                {
                    "events": [],
                    "exceeds_environment": [],
                    "invented_fixture_attack": [],
                    "dirty": [],
                },
            )
            entry["events"].append(int(event))
            checks = summary.get("checks") or {}
            if not checks.get("no_component_exceeds_the_team_environment", True):
                entry["exceeds_environment"].append(
                    {"event": int(event), "records": summary.get("exceeds_environment") or []}
                )
            if not checks.get("no_invented_fixture_attack", True):
                entry["invented_fixture_attack"].append(
                    {"event": int(event), "records": summary.get("invented_fixture_attack") or []}
                )
            for name, value in sorted(checks.items()):
                if isinstance(value, bool) and not value:
                    entry["dirty"].append(f"{name}@{event}")
    for entry in arms.values():
        entry["events"] = sorted(set(entry["events"]))
    return {
        "rule": (
            "the player level may not exceed the team environment, unallocated mass is explicit, a "
            "double gameweek stays fixture-atomic, a blank invents no fixture attack, and an "
            "excess is reported rather than clamped or redistributed"
        ),
        "arms": arms,
    }
