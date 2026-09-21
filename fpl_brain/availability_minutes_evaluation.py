"""PE-6 evaluation: incumbent vs challenger on identical walk-forward populations.

WHAT THIS PRODUCES
------------------
One evidence artifact.  Not a promotion, not a certification, not a merge
decision: the incumbent remains authoritative and PE-6 ends at senior review.
The artifact exists so that a reviewer can see, for a declared population, on
which families the challenger moved the prediction and what that did to the
error metrics -- including when the honest answer is "nothing measurable".

MEASUREMENT ONLY, UNTIL SENIOR REVIEW
-------------------------------------
This artifact MEASURES.  It does not accept, endorse or promote anything, and it
does not name a winner.  PE-2's sample-size floors are DESCRIPTIVE-REPORTING
floors: ``MIN_TARGET_EVENTS_FOR_DESCRIPTIVE`` / ``MIN_OBSERVATIONS_FOR_DESCRIPTIVE``
gate whether a description may be drawn at all, so PE-6 uses them in exactly that
role and no stronger one.  Every comparison therefore reports which arm the
declared floor prefers on each criterion (``preferred``), and the summary states
``model_selection: NOT_CLAIMED_UNTIL_SENIOR_REVIEW``.  A criterion inside its
floor reads ``WITHIN_DECLARED_NOISE_FLOOR``: not distinguishable from noise is
not an improvement.

A multi-family subset is never assembled out of single-family measurements.  Any
combination of families can only be reported through an ARM THAT EVALUATES IT
(``multi_family_subset_policy``); the headline all-families arm is such an arm,
and a caller proposing a further subset must declare it and read its own row.

GRAIN: PE-2's PLAYER-FIXTURE COMPONENT GRAIN
--------------------------------------------
Minutes are a per-fixture quantity, so PE-6 scores at PE-2's declared
``player_fixture`` component grain (``walk_forward.GRAIN_PLAYER_FIXTURE``) -- the
grain PE-2 already uses for minutes MAE and bias.  Every fixture of a double
gameweek is therefore its own observation and is RETAINED, and no aggregation
rule for point probabilities is invented: a DGW player-event is simply two
scored rows, exactly as the model published them.

IDENTICAL POPULATIONS, ENFORCED
-------------------------------
Every arm is projected from the same cutoff with the same PE-1 causal boundary,
and the population is classified ONCE and shared by every arm.  Before any
metric is computed, ``walk_forward.assert_same_population`` proves that the
incumbent and each arm cover exactly the same key set; a mismatch stops the
evaluation instead of producing a comparison of different worlds.  The pooled
population carries a ``walk_forward.canonical_population_digest``.

The candidate pool itself is resolved AS OF EACH CUTOFF: membership, club and
position come from the official element capture that was observable then, never
from the current ``players`` row (see
:mod:`fpl_brain.availability_minutes_challenger`).  A player the cutoff places in
the official pool stays a candidate for that cutoff whatever has happened since,
and a post-cutoff transfer, position change or retirement cannot rewrite an
earlier projection.  The resolved bases are reported in
``population.identity``.

WHAT IS SCORED
--------------
Metrics, all reported with their sample size:

* Brier for ``P(start)``            (and its base-rate reference)
* Brier for ``P(60+)``              (and its base-rate reference)
* Brier for ``P(80+)``
* Brier for ``P(cameo | not start)``
* Brier for ``P(appearance)``       (``P(start) + P(cameo)``)
* expected-minutes MAE and bias     (bias = mean(predicted - actual))
* predicted vs realised start / 60+ / appearance rates

Bias convention, tie policies and rounding are the PE-2 metric engine's
(``walk_forward_metrics``), reused rather than restated.

SAMPLE-SIZE HONESTY
-------------------
The PE-2 policy decides whether the sample can support a description: below its
declared floors the artifact says ``INSUFFICIENT_FOR_STRONG_MODEL_SELECTION``,
every measurement is reported as ``INSUFFICIENT_FOR_ANY_CLAIM``, and nothing is
claimed.  Nothing here is described as better, superior, accepted or proven.
Strata below their own declared floor are labelled insufficient with their ``n``
rather than silently omitted.

EXCLUSIONS ARE COUNTED, AND THEY ADD UP
---------------------------------------
Excluded candidates are counted, never scored as zero, and every exclusion
carries its SCOPE (``PLAYER`` or ``EVENT``) and how many candidates it accounts
for.  An event whose cutoff is unavailable excludes its whole candidate pool at
event scope, so the aggregate counts still reconcile:
``candidates == scored + excluded``.  ``population.accounting`` states that
identity in the artifact itself.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Callable, Mapping, Sequence

from . import analytics
from . import availability_minutes_challenger as challenger
from . import minutes_model as incumbent
from . import repositories as repo
from . import walk_forward as wf
from . import walk_forward_metrics as wm
from . import walk_forward_scoreboard as wfs

EVALUATION_VERSION = "availability_minutes_evaluation_v1.1.0"

GRAIN_PLAYER_FIXTURE = wf.GRAIN_PLAYER_FIXTURE
GRAIN_NOTE = (
    "player x fixture; a double gameweek is one observation per fixture, so every DGW "
    "fixture is retained and no point-probability aggregation rule is invented"
)

#: The cutoff a walk-forward prediction is taken at: the target event's official
#: deadline, which is the last moment a manager could have acted.  A deadline
#: that is not before the event's first kickoff is refused rather than used.
CUTOFF_POLICY = "TARGET_EVENT_DEADLINE_TIME"
MISSING_DATA_POLICY_VERSION = wf.MISSING_DATA_POLICY_VERSION

# --- exclusion statuses (excluded candidates are counted, never scored zero) --
#: Candidates are enumerated from the official pool as of the cutoff (players the
#: cutoff captures place in the pool, plus the persisted pool), so a player
#: outside it is not a candidate at all rather than an exclusion.
STATUS_BLANK_NO_FIXTURE = "TARGET_BLANK_NO_FIXTURE"
STATUS_OUTCOME_NOT_FINALISED = wf.OUTCOME_NOT_FINALISED
STATUS_OUTCOME_MISSING = "OUTCOME_EVIDENCE_MISSING"
STATUS_OUTCOME_PLACEHOLDER = wf.OUTCOME_PLACEHOLDER_EXCLUDED
STATUS_MODEL_PROJECTION_MISSING = wf.MODEL_PROJECTION_MISSING
STATUS_FIXTURE_NOT_PLAYED = "TARGET_FIXTURE_NOT_PLAYED"
STATUS_EVENT_CUTOFF_UNAVAILABLE = "TARGET_EVENT_CUTOFF_UNAVAILABLE"
STATUS_EVENT_NOT_FINAL = "TARGET_EVENT_OUTCOME_NOT_FINALISED"
STATUS_IDENTITY_UNRESOLVED_AT_CUTOFF = "CANDIDATE_IDENTITY_UNRESOLVED_AT_CUTOFF"

EXCLUSION_STATUSES: tuple[str, ...] = (
    STATUS_BLANK_NO_FIXTURE,
    STATUS_OUTCOME_NOT_FINALISED,
    STATUS_OUTCOME_MISSING,
    STATUS_OUTCOME_PLACEHOLDER,
    STATUS_MODEL_PROJECTION_MISSING,
    STATUS_FIXTURE_NOT_PLAYED,
    STATUS_EVENT_CUTOFF_UNAVAILABLE,
    STATUS_EVENT_NOT_FINAL,
    STATUS_IDENTITY_UNRESOLVED_AT_CUTOFF,
)

#: The scope an exclusion accounts for.  An event-scope exclusion stands for a
#: whole candidate pool at once; a player-scope exclusion stands for one player.
SCOPE_PLAYER = "PLAYER"
SCOPE_EVENT = "EVENT"

# --- declared comparison rule (measurements, not a verdict) -------------------
PREFERRED_CHALLENGER = "CHALLENGER"
PREFERRED_INCUMBENT = "INCUMBENT"
WITHIN_NOISE_FLOOR = "WITHIN_DECLARED_NOISE_FLOOR"

#: What one arm's measurement over this population shows.  Deliberately NOT an
#: acceptance vocabulary: nothing here is a promotion, an endorsement or a
#: selection, and every token is a statement about numbers on a sample.
MEASUREMENT_UNDEFINED = "MEASUREMENT_UNDEFINED"
MEASUREMENT_INSUFFICIENT_SAMPLE = "INSUFFICIENT_FOR_ANY_CLAIM"
MEASUREMENT_CHALLENGER_PREFERRED = "CHALLENGER_PREFERRED_BY_THE_DECLARED_FLOORS"
MEASUREMENT_INCUMBENT_PREFERRED = "INCUMBENT_PREFERRED_BY_THE_DECLARED_FLOORS"
MEASUREMENT_PARTIAL = "CHALLENGER_PREFERRED_ON_SOME_CRITERIA_ONLY"
MEASUREMENT_WITHIN_NOISE = "WITHIN_DECLARED_NOISE_FLOOR"
MEASUREMENT_NO_ADVANTAGE = "NO_MEASURABLE_ADVANTAGE"

MEASUREMENT_VOCABULARY: tuple[str, ...] = (
    MEASUREMENT_UNDEFINED,
    MEASUREMENT_INSUFFICIENT_SAMPLE,
    MEASUREMENT_CHALLENGER_PREFERRED,
    MEASUREMENT_PARTIAL,
    MEASUREMENT_WITHIN_NOISE,
    MEASUREMENT_NO_ADVANTAGE,
    MEASUREMENT_INCUMBENT_PREFERRED,
)

MEASUREMENT_BASIS_INSUFFICIENT_SAMPLE = "INSUFFICIENT_SAMPLE_UNDER_THE_PE2_POLICY"
MEASUREMENT_BASIS_CHALLENGER_WORSE = "CHALLENGER_IS_WORSE_ON_A_PRIMARY_CRITERION"
MEASUREMENT_BASIS_WITHIN_NOISE = "EVERY_PRIMARY_CRITERION_INSIDE_ITS_DECLARED_FLOOR"
MEASUREMENT_BASIS_UNDEFINED = "PRIMARY_CRITERION_UNDEFINED"
MEASUREMENT_BASIS_PARTIAL = "SOME_PRIMARY_CRITERIA_PREFER_THE_CHALLENGER"
MEASUREMENT_BASIS_ALL = "EVERY_PRIMARY_CRITERION_PREFERS_THE_CHALLENGER"
MEASUREMENT_BASIS_NO_ADVANTAGE = "NO_PRIMARY_CRITERION_PREFERS_THE_CHALLENGER"

#: The claim this artifact is allowed to make.  PE-2's floors are descriptive,
#: so the artifact measures and stops there.
SELECTION_CLAIM_NOT_MADE = "NOT_CLAIMED_UNTIL_SENIOR_REVIEW"
CLAIMS_ALLOWED: tuple[str, ...] = ("DESCRIPTIVE_REPORTING_ONLY",)
CLAIMS_NOT_ALLOWED: tuple[str, ...] = ("MODEL_SELECTION", "PROMOTION", "CERTIFICATION")

#: Metrics the measurement summary is computed on.  Declared, never selected
#: from the data.
PRIMARY_CRITERIA: tuple[str, ...] = ("brier_p_start", "brier_p_60", "expected_minutes_mae")

ERROR_METRICS: tuple[str, ...] = (
    "brier_p_start",
    "brier_p_cameo",
    "brier_p_60",
    "brier_p_80",
    "brier_p_appearance",
    "expected_minutes_mae",
)

#: A change smaller than this is not an improvement, however it is signed.
DELTA_BRIER_IMPROVEMENT_MIN = 0.002
DELTA_MAE_IMPROVEMENT_MIN = 0.05
DELTA_BIAS_IMPROVEMENT_MIN = 0.05

DELTA_FLOORS: dict[str, float] = {
    "brier_p_start": DELTA_BRIER_IMPROVEMENT_MIN,
    "brier_p_cameo": DELTA_BRIER_IMPROVEMENT_MIN,
    "brier_p_60": DELTA_BRIER_IMPROVEMENT_MIN,
    "brier_p_80": DELTA_BRIER_IMPROVEMENT_MIN,
    "brier_p_appearance": DELTA_BRIER_IMPROVEMENT_MIN,
    "expected_minutes_mae": DELTA_MAE_IMPROVEMENT_MIN,
    "expected_minutes_bias": DELTA_BIAS_IMPROVEMENT_MIN,
}

DELTA_CONVENTION = (
    "delta = challenger - incumbent; for brier/mae lower is better, for bias closer to "
    "zero is better, and |delta| below the declared floor is WITHIN_DECLARED_NOISE_FLOOR"
)
BIAS_CONVENTION = "mean(predicted - actual); positive means overprediction"

#: Declared disclosure floor for a stratum.  NOT a significance threshold.
MIN_STRATUM_OBSERVATIONS = 30

STRATUM_DIMENSIONS: tuple[str, ...] = (
    "by_position",
    "by_availability_basis",
    "by_status_at_cutoff",
    "by_return_ramp",
    "by_rotation_risk",
)


class EvaluationError(RuntimeError):
    """The evaluation cannot be produced honestly from what is available."""


# ---------------------------------------------------------------------------
# Cutoff and outcome reads.
# ---------------------------------------------------------------------------


def event_cutoff(conn: sqlite3.Connection, event: int) -> tuple[str | None, list[str]]:
    """The cutoff a prediction for ``event`` must be taken at, or why not.

    Fail closed: no deadline means no point in time anyone decided at, and a
    deadline at or after the first kickoff would score a prediction made with
    knowledge of the event.
    """

    row = conn.execute(
        "SELECT deadline_time FROM events WHERE id=?", (int(event),)
    ).fetchone()
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
            "official deadline_time is not before the event's first kickoff, so it cannot "
            "serve as a causal cutoff"
        ]
    return deadline, []


def target_event_state(conn: sqlite3.Connection, event: int) -> tuple[str, list[str]]:
    from . import planning

    state, reasons = planning.event_data_state(conn, int(event))
    return str(state), list(reasons)


def realised_outcome(
    conn: sqlite3.Connection, player_id: int, fixture_id: int
) -> tuple[dict[str, Any] | None, str]:
    """The official realised facts for one player-fixture, or why there are none.

    A scheduled placeholder is not a did-not-play, and a missing row is not a
    zero: both are refused with their own status.
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
    starts = values.get("starts")
    if minutes is None or starts is None:
        return None, STATUS_OUTCOME_MISSING
    minutes_value = float(minutes)
    started = 1 if int(starts) else 0
    return (
        {
            "minutes": minutes_value,
            "starts": int(starts),
            "started": started,
            "cameo": 1 if (started == 0 and minutes_value > 0) else 0,
            "appeared": 1 if minutes_value > 0 else 0,
            "minutes_60": 1 if minutes_value >= 60 else 0,
            "minutes_80": 1 if minutes_value >= 80 else 0,
        },
        "",
    )


# ---------------------------------------------------------------------------
# Metrics.
# ---------------------------------------------------------------------------


def _metric(value: Any) -> Any:
    return value.as_dict()


def arm_metrics(
    rows: Sequence[Mapping[str, Any]], realised: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """The declared metric set for one arm, over one aligned population."""

    n = len(rows)
    if not n:
        empty = wm.Metric(wm.METRIC_NO_SAMPLE).as_dict()
        return {
            "n": 0,
            "brier_p_start": empty,
            "brier_p_cameo": empty,
            "brier_p_60": empty,
            "brier_p_80": empty,
            "brier_p_appearance": empty,
            "expected_minutes_mae": empty,
            "expected_minutes_bias": empty,
            "brier_p_start_reference": empty,
            "brier_p_60_reference": empty,
            "start_rate_predicted": None,
            "start_rate_realised": None,
            "p60_rate_predicted": None,
            "p60_rate_realised": None,
            "appearance_rate_predicted": None,
            "appearance_rate_realised": None,
        }
    p_start = [float(row["p_start"]) for row in rows]
    p_cameo = [float(row["p_cameo"]) for row in rows]
    p_60 = [float(row["p_60_plus"]) for row in rows]
    p_80 = [float(row["p_80_plus"]) for row in rows]
    # P(appearance) = P(start) + P(cameo).  The sum cannot exceed 1 by
    # construction (P(start) + P(cameo) <= P(available) <= 1); the clamp only
    # absorbs 6-decimal storage rounding, so the metric engine's [0, 1]
    # validation is never tripped by rounding rather than by a model defect.
    p_appearance = [min(1.0, a + b) for a, b in zip(p_start, p_cameo)]
    expected = [float(row["expected_minutes"]) for row in rows]
    started = [float(item["started"]) for item in realised]
    cameo = [float(item["cameo"]) for item in realised]
    m60 = [float(item["minutes_60"]) for item in realised]
    m80 = [float(item["minutes_80"]) for item in realised]
    appeared = [float(item["appeared"]) for item in realised]
    minutes = [float(item["minutes"]) for item in realised]
    return {
        "n": n,
        "brier_p_start": _metric(wm.brier_score(p_start, started)),
        "brier_p_cameo": _metric(wm.brier_score(p_cameo, cameo)),
        "brier_p_60": _metric(wm.brier_score(p_60, m60)),
        "brier_p_80": _metric(wm.brier_score(p_80, m80)),
        "brier_p_appearance": _metric(wm.brier_score(p_appearance, appeared)),
        "expected_minutes_mae": _metric(wm.mean_absolute_error(expected, minutes)),
        "expected_minutes_bias": _metric(wm.mean_bias(expected, minutes)),
        "brier_p_start_reference": _metric(wm.brier_reference_score(started)),
        "brier_p_60_reference": _metric(wm.brier_reference_score(m60)),
        "start_rate_predicted": _mean(p_start),
        "start_rate_realised": _mean(started),
        "p60_rate_predicted": _mean(p_60),
        "p60_rate_realised": _mean(m60),
        "appearance_rate_predicted": _mean(p_appearance),
        "appearance_rate_realised": _mean(appeared),
    }


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), wm.METRIC_DECIMALS)


def _preference(metric_name: str, incumbent_value: float | None, challenger_value: float | None) -> str | None:
    """Which arm the declared floor prefers on one metric, or None if undefined."""

    if incumbent_value is None or challenger_value is None:
        return None
    floor = DELTA_FLOORS.get(metric_name, 0.0)
    if metric_name in ERROR_METRICS:
        delta = float(challenger_value) - float(incumbent_value)
        if delta <= -floor:
            return PREFERRED_CHALLENGER
        if delta >= floor:
            return PREFERRED_INCUMBENT
        return WITHIN_NOISE_FLOOR
    # Bias: distance from zero, so a signed improvement is not a preference.
    improvement = abs(float(incumbent_value)) - abs(float(challenger_value))
    if improvement >= floor:
        return PREFERRED_CHALLENGER
    if improvement <= -floor:
        return PREFERRED_INCUMBENT
    return WITHIN_NOISE_FLOOR


def comparison_block(
    incumbent_metrics: Mapping[str, Any], challenger_metrics: Mapping[str, Any]
) -> dict[str, Any]:
    """Per-metric incumbent/challenger comparison under the declared floors."""

    block: dict[str, Any] = {
        "delta_convention": DELTA_CONVENTION,
        "bias_convention": BIAS_CONVENTION,
        "floors": {
            "brier": DELTA_BRIER_IMPROVEMENT_MIN,
            "expected_minutes_mae": DELTA_MAE_IMPROVEMENT_MIN,
            "expected_minutes_bias": DELTA_BIAS_IMPROVEMENT_MIN,
            "basis": "declared a priori disclosure floors, not significance tests",
        },
        "metrics": {},
        "rates": {},
        "references": {},
    }
    for name in (*ERROR_METRICS, "expected_minutes_bias"):
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
                "OK"
                if incumbent_value is not None and challenger_value is not None
                else "UNDEFINED"
            ),
            "preferred": _preference(name, incumbent_value, challenger_value),
        }
    for name in (
        "start_rate_predicted",
        "start_rate_realised",
        "p60_rate_predicted",
        "p60_rate_realised",
        "appearance_rate_predicted",
        "appearance_rate_realised",
    ):
        block["rates"][name] = {
            "incumbent": incumbent_metrics.get(name),
            "challenger": challenger_metrics.get(name),
        }
    for name in ("brier_p_start_reference", "brier_p_60_reference"):
        block["references"][name] = {
            "incumbent": incumbent_metrics.get(name),
            "challenger": challenger_metrics.get(name),
            "note": "Brier of the constant base-rate forecast, for scale only",
        }
    return block


def measurement_rule_block() -> dict[str, Any]:
    """The declared, a-priori rule this artifact measures under.

    Stated once and repeated in the artifact, so the numbers cannot be read
    through a rule that was chosen after seeing them.
    """

    return {
        "primary_criteria": list(PRIMARY_CRITERIA),
        "floors": {
            "brier": DELTA_BRIER_IMPROVEMENT_MIN,
            "expected_minutes_mae": DELTA_MAE_IMPROVEMENT_MIN,
            "expected_minutes_bias": DELTA_BIAS_IMPROVEMENT_MIN,
            "basis": (
                "declared a priori disclosure floors, NOT significance tests and NOT "
                "acceptance thresholds"
            ),
        },
        "delta_convention": DELTA_CONVENTION,
        "bias_convention": BIAS_CONVENTION,
        "measurement_vocabulary": list(MEASUREMENT_VOCABULARY),
        "criteria_declared_a_priori": True,
        "selected_from_the_data": False,
    }


def arm_measurement(
    comparison: Mapping[str, Any], *, sample_sufficient: bool
) -> dict[str, Any]:
    """What one arm's numbers show on this population.  Pure function.

    Deliberately not a verdict: the tokens describe measurements (which arm the
    declared floors prefer, and on which criteria), never an acceptance.  Order
    of refusal matters: an inadequate sample is reported as an inadequate sample,
    never as a failed challenger and never as a selection claim.
    """

    metrics = comparison.get("metrics") or {}
    if not sample_sufficient:
        return {
            "token": MEASUREMENT_INSUFFICIENT_SAMPLE,
            "basis": [MEASUREMENT_BASIS_INSUFFICIENT_SAMPLE],
            "note": (
                "the PE-2 policy's descriptive floor is not met on this population, so no "
                "measurement of the difference is drawn and nothing is claimed"
            ),
        }
    undefined = [
        name
        for name in PRIMARY_CRITERIA
        if (metrics.get(name) or {}).get("status") != "OK"
    ]
    if undefined:
        return {
            "token": MEASUREMENT_UNDEFINED,
            "basis": [MEASUREMENT_BASIS_UNDEFINED, *[f"{name}:UNDEFINED" for name in undefined]],
            "note": "a primary criterion could not be computed, so no measurement is drawn",
        }
    preferred = {name: (metrics.get(name) or {}).get("preferred") for name in PRIMARY_CRITERIA}
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
                "the declared floors prefer the incumbent on a primary criterion, so there is "
                "no measurable advantage to report for this arm"
            ),
        }
    if not better:
        return {
            "token": MEASUREMENT_WITHIN_NOISE,
            "basis": [MEASUREMENT_BASIS_WITHIN_NOISE],
            "note": (
                "every primary criterion is inside its declared floor, so the two arms cannot "
                "be distinguished on this sample"
            ),
        }
    if len(better) == len(PRIMARY_CRITERIA):
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
            "the declared floors prefer the challenger on some primary criteria and neither on "
            "the rest; this is a measurement for senior review, not a partial acceptance"
        ),
    }


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
            "headline": arm == challenger.ARM_ALL_REFINEMENTS,
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


def build_event_records(
    conn: sqlite3.Connection,
    event: int,
    *,
    cutoff: str | None,
    arms: challenger.ChallengerArms | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int], list[str]]:
    """Classify every candidate player-fixture of one target event, once.

    The SAME classification is used for every arm, so the populations cannot
    differ between them by construction; the arms' own key coverage is asserted
    separately in :func:`evaluate_events`.  The candidate pool is the
    cutoff-resolved one the arms were built from (``arms.players``), so the
    classification and the projections cannot disagree about who is who.

    Every fixture of the event is classified separately.  A double gameweek is
    therefore two scored observations rather than one excluded one, and no
    aggregation rule has to be invented for it.
    """

    reasons: list[str] = []
    state, state_reasons = target_event_state(conn, event)
    if cutoff is None:
        return [], [], {}, state_reasons
    if state != "FINAL":
        reasons.append(
            f"target event {event} is {state} ({'; '.join(state_reasons)}), so its outcomes are "
            "not finalised evidence"
        )
    pool = list(arms.players) if arms is not None else challenger.projectable_players_as_of(conn, cutoff)
    unresolved = list(arms.unresolved) if arms is not None else []
    fixtures_by_team = analytics.event_fixture_map(conn, int(event))
    records: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {}

    def exclude(
        player_id: int | None,
        team_id: int | None,
        status: str,
        detail: str,
        *,
        fixtures: Sequence[int] = (),
        candidates: int,
        scope: str = SCOPE_PLAYER,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        entry: dict[str, Any] = {
            "event": int(event),
            "scope": scope,
            "player_id": None if player_id is None else int(player_id),
            "team_id": None if team_id is None else int(team_id),
            "fixtures": sorted(int(value) for value in fixtures),
            "candidates": int(candidates),
            "status": status,
            "detail": detail,
        }
        if extra:
            entry.update(dict(extra))
        excluded.append(entry)
        status_counts[status] = status_counts.get(status, 0) + int(candidates)

    for player_id in sorted(
        [int(row["player_id"]) for row in pool]
        + [int(row["player_id"]) for row in unresolved]
    ):
        player = next((row for row in pool if int(row["player_id"]) == player_id), None)
        if player is None:
            unresolved_identity = next(
                row for row in unresolved if int(row["player_id"]) == player_id
            )
            exclude(
                player_id,
                None,
                STATUS_IDENTITY_UNRESOLVED_AT_CUTOFF,
                "the cutoff cannot place this player in the official pool, so no projection "
                "is attempted for him",
                candidates=1,
                extra={"identity": unresolved_identity},
            )
            continue
        team_id = int(player["team_id"])
        team_fixtures = fixtures_by_team.get(team_id, [])
        fixtures = [int(fixture["id"]) for fixture in team_fixtures]
        if not fixtures:
            # No fixture to enumerate: the candidate himself is the one slot.
            exclude(
                player_id,
                team_id,
                STATUS_BLANK_NO_FIXTURE,
                "no fixture in the target event",
                candidates=1,
            )
            continue
        if state != "FINAL":
            exclude(
                player_id,
                team_id,
                STATUS_EVENT_NOT_FINAL,
                f"event state {state}",
                fixtures=fixtures,
                candidates=len(fixtures),
            )
            continue
        for fixture in team_fixtures:
            fixture_id = int(fixture["id"])
            if arms is None or (player_id, fixture_id) not in arms.incumbent_rows:
                exclude(
                    player_id,
                    team_id,
                    STATUS_MODEL_PROJECTION_MISSING,
                    "the model produced no projection for this player-fixture",
                    fixtures=[fixture_id],
                    candidates=1,
                )
                continue
            if not (int(fixture.get("finished") or 0) == 1 and int(fixture.get("started") or 0) == 1):
                exclude(
                    player_id,
                    team_id,
                    STATUS_FIXTURE_NOT_PLAYED,
                    "the fixture has not started and finished, so it has no realised outcome",
                    fixtures=[fixture_id],
                    candidates=1,
                )
                continue
            outcome, outcome_status = realised_outcome(conn, player_id, fixture_id)
            if outcome is None:
                exclude(
                    player_id,
                    team_id,
                    outcome_status,
                    "official outcome evidence unavailable",
                    fixtures=[fixture_id],
                    candidates=1,
                )
                continue
            incumbent_row = arms.incumbent_rows[(player_id, fixture_id)]
            challenger_row = arms.arms[challenger.ARM_ALL_REFINEMENTS][(player_id, fixture_id)]
            evidence = arms.evidence[(player_id, fixture_id)]
            records.append(
                {
                    "event": int(event),
                    "player_id": player_id,
                    "fixture_id": fixture_id,
                    "team_id": team_id,
                    "position_id": int(player.get("element_type") or 0),
                    "availability_basis": evidence.availability_basis,
                    "status_at_cutoff": evidence.status_at_cutoff,
                    "return_ramp": bool(evidence.ramp_applies),
                    "identity_basis": str(player.get("identity_basis") or ""),
                    "identity_cutoff_safe": bool(player.get("identity_cutoff_safe")),
                    "identity_capture_is_newest": player.get("identity_capture_is_newest"),
                    "player_identity": dict(player.get("identity") or {}),
                    "rotation_risk": any(
                        flag in set(incumbent_row.get("risk_flags") or [])
                        for flag in ("ROTATION_RISK_HIGH", "START_ROLE_WEAKENED")
                    ),
                    "realised": outcome,
                    "incumbent_row": incumbent_row,
                    "challenger_row": challenger_row,
                    "arm_rows": {
                        arm: arms.arms[arm][(player_id, fixture_id)] for arm in arms.arm_names()
                    },
                }
            )
    records.sort(
        key=lambda item: (int(item["event"]), int(item["player_id"]), int(item["fixture_id"]))
    )
    excluded.sort(
        key=lambda item: (
            int(item["event"]),
            -1 if item["player_id"] is None else int(item["player_id"]),
            str(item["status"]),
        )
    )
    return records, excluded, status_counts, reasons


# ---------------------------------------------------------------------------
# Strata.
# ---------------------------------------------------------------------------


def _stratum_key(dimension: str) -> Callable[[Mapping[str, Any]], str]:
    if dimension == "by_position":
        return lambda item: str(int(item["position_id"]))
    if dimension == "by_availability_basis":
        return lambda item: str(item["availability_basis"])
    if dimension == "by_status_at_cutoff":
        return lambda item: "NO_SNAPSHOT" if item["status_at_cutoff"] is None else str(item["status_at_cutoff"])
    if dimension == "by_return_ramp":
        return lambda item: "RETURN_RAMP" if item["return_ramp"] else "NO_RETURN_RAMP"
    if dimension == "by_rotation_risk":
        return lambda item: "ROTATION_RISK" if item["rotation_risk"] else "NO_ROTATION_RISK"
    raise ValueError(f"unknown stratum dimension: {dimension}")


def stratum_blocks(records: Sequence[Mapping[str, Any]], *, arm: str) -> dict[str, Any]:
    """Every stratum of every declared dimension, with its own sample size."""

    sections: dict[str, Any] = {}
    for dimension in STRATUM_DIMENSIONS:
        keyfn = _stratum_key(dimension)
        grouped: dict[str, list[Mapping[str, Any]]] = {}
        for record in records:
            grouped.setdefault(keyfn(record), []).append(record)
        blocks: dict[str, Any] = {}
        for key in sorted(grouped):
            group = grouped[key]
            incumbent_metrics = arm_metrics([item["incumbent_row"] for item in group], [item["realised"] for item in group])
            arm_metrics_block = arm_metrics(
                [item["arm_rows"][arm] for item in group], [item["realised"] for item in group]
            )
            n = len(group)
            blocks[key] = {
                "n": n,
                "sample_interpretation": (
                    "SUFFICIENT_FOR_STRATUM_DESCRIPTIVE_REPORTING"
                    if n >= MIN_STRATUM_OBSERVATIONS
                    else "INSUFFICIENT_STRATUM_SAMPLE"
                ),
                "incumbent": incumbent_metrics,
                "challenger": arm_metrics_block,
                "comparison": comparison_block(incumbent_metrics, arm_metrics_block),
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


# ---------------------------------------------------------------------------
# The artifact.
# ---------------------------------------------------------------------------


def normalize_target_events(events: Sequence[int]) -> dict[str, Any]:
    """The requested events, de-duplicated and sorted, with what changed.

    A repeated event id must not be able to double a population: two passes over
    the same event would add its observations twice and inflate both the event
    count and the metric sample.  Duplicates are therefore removed once, here,
    and the removal is recorded rather than performed silently.
    """

    requested = [int(value) for value in events]
    normalized = sorted({int(value) for value in requested})
    duplicates: dict[str, int] = {}
    for value in requested:
        duplicates[str(int(value))] = duplicates.get(str(int(value)), 0) + 1
    repeated = {key: count for key, count in sorted(duplicates.items()) if count > 1}
    return {
        "requested": requested,
        "normalized": normalized,
        "duplicates_removed": len(requested) - len(normalized),
        "repeated_events": repeated,
        "rule": (
            "target events are de-duplicated and sorted before any projection, so a repeated "
            "id cannot add a second copy of an event's observations to any count"
        ),
    }


def evaluate_events(
    conn: sqlite3.Connection,
    events: Sequence[int],
    *,
    incumbent_config: "incumbent.MinutesModelConfig | None" = None,
    challenger_config: challenger.AvailabilityMinutesChallengerConfig | None = None,
    arm_definitions: Mapping[str, frozenset[str]] | None = None,
    allow_live_identity_fallback: bool = True,
) -> dict[str, Any]:
    """Compare the incumbent and the challenger over a walk-forward event set.

    ``allow_live_identity_fallback`` controls what happens to a candidate the
    cutoff holds no official capture for.  Left on (the default), the current
    ``players`` row supplies his identity and the fact is recorded per row and
    counted in ``population.identity``; turned off, his identity is refused and
    he is excluded with ``STATUS_IDENTITY_UNRESOLVED_AT_CUTOFF`` instead of being
    projected under an identity that a later change could have written.
    """

    config = incumbent_config or incumbent.MinutesModelConfig()
    challenger_config = challenger_config or challenger.AvailabilityMinutesChallengerConfig()
    definitions = dict(arm_definitions or challenger.default_arm_definitions())
    if challenger.ARM_ALL_REFINEMENTS not in definitions:
        raise EvaluationError(
            f"the evaluation requires the headline arm {challenger.ARM_ALL_REFINEMENTS!r} "
            "in the arm definitions: the population is classified from it"
        )
    event_request = normalize_target_events(events)
    target_events = list(event_request["normalized"])

    records: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {}
    event_blocks: list[dict[str, Any]] = []
    events_with_observations: list[int] = []
    events_excluded: list[dict[str, Any]] = []

    for event in target_events:
        cutoff, cutoff_reasons = event_cutoff(conn, event)
        if cutoff is None:
            # The cutoff is unavailable, so no identity can be resolved at it and
            # no projection exists.  The whole candidate pool of this event is
            # excluded at EVENT scope: one row that ACCOUNTS for every candidate,
            # so the aggregate counts still reconcile with the scored rows.
            persisted_pool = len(analytics.projectable_players(conn))
            excluded.append(
                {
                    "event": int(event),
                    "scope": SCOPE_EVENT,
                    "player_id": None,
                    "team_id": None,
                    "fixtures": [],
                    "candidates": int(persisted_pool),
                    "status": STATUS_EVENT_CUTOFF_UNAVAILABLE,
                    "detail": "; ".join(cutoff_reasons),
                }
            )
            status_counts[STATUS_EVENT_CUTOFF_UNAVAILABLE] = (
                status_counts.get(STATUS_EVENT_CUTOFF_UNAVAILABLE, 0) + int(persisted_pool)
            )
            events_excluded.append({"event": int(event), "reasons": cutoff_reasons})
            event_blocks.append(
                {
                    "event": int(event),
                    "status": "NOT_EVALUATED",
                    "cutoff": None,
                    "cutoff_policy": CUTOFF_POLICY,
                    "reasons": cutoff_reasons,
                    "candidates": int(persisted_pool),
                    "scored_observations": 0,
                    "exclusion_scope": SCOPE_EVENT,
                    "identity": None,
                }
            )
            continue
        resolution = challenger.resolve_candidate_pool(
            conn,
            cutoff,
            allow_live_fallback=allow_live_identity_fallback,
        )
        arms = challenger.build_challenger_arms(
            conn,
            event,
            cutoff,
            incumbent_config=config,
            challenger_config=challenger_config,
            arm_definitions=definitions,
            players=resolution["players"],
            unresolved=resolution["unresolved"],
        )
        # The arms must cover the identical key set before anything is scored.
        arms.verify_same_population()
        event_records, event_excluded, event_status_counts, event_reasons = build_event_records(
            conn, event, cutoff=cutoff, arms=arms
        )
        records.extend(event_records)
        excluded.extend(event_excluded)
        for status, count in event_status_counts.items():
            status_counts[status] = status_counts.get(status, 0) + count
        if event_records:
            events_with_observations.append(event)
        event_blocks.append(
            {
                "event": int(event),
                "status": "EVALUATED" if event_records else "NOT_EVALUATABLE",
                "cutoff": cutoff,
                "cutoff_policy": CUTOFF_POLICY,
                "reasons": event_reasons,
                "candidates": len(event_records) + sum(
                    int(item["candidates"]) for item in event_excluded
                ),
                "scored_observations": len(event_records),
                "excluded_by_status": dict(sorted(event_status_counts.items())),
                "exclusion_scope": SCOPE_PLAYER,
                "identity": resolution["summary"],
                "population_digest": wf.canonical_population_digest(
                    [
                        (int(item["event"]), int(item["player_id"]), int(item["fixture_id"]))
                        for item in event_records
                    ],
                    grain=GRAIN_PLAYER_FIXTURE,
                ),
            }
        )

    realisation = [item["realised"] for item in records]
    population_keys = [
        (int(item["event"]), int(item["player_id"]), int(item["fixture_id"]))
        for item in records
    ]
    incumbent_metrics = arm_metrics([item["incumbent_row"] for item in records], realisation)

    arm_payloads: dict[str, Any] = {}
    for arm in sorted(definitions):
        arm_metrics_block = arm_metrics(
            [item["arm_rows"][arm] for item in records], realisation
        )
        comparison = comparison_block(incumbent_metrics, arm_metrics_block)
        arm_payloads[arm] = {
            "arm": arm,
            "families": sorted(definitions[arm]),
            "metrics": arm_metrics_block,
            "comparison": comparison,
        }

    sample_sufficient = (
        len(events_with_observations) >= wfs.MIN_TARGET_EVENTS_FOR_DESCRIPTIVE
        and len(records) >= wfs.MIN_OBSERVATIONS_FOR_DESCRIPTIVE
    )
    sample_interpretation = (
        wfs.SAMPLE_DESCRIPTIVE_ONLY if sample_sufficient else wfs.SAMPLE_INSUFFICIENT
    )
    for arm, payload in arm_payloads.items():
        payload["measurement"] = arm_measurement(
            payload["comparison"], sample_sufficient=sample_sufficient
        )

    headline = arm_payloads.get(challenger.ARM_ALL_REFINEMENTS)
    family_measurements: dict[str, Any] = {}
    for family in challenger.REFINEMENT_FAMILIES:
        arm_name = challenger.arm_name_for_family(family)
        payload = arm_payloads.get(arm_name)
        if payload is None:
            continue
        family_measurements[family] = {
            "arm": arm_name,
            "measurement": payload["measurement"],
            "comparison": payload["comparison"],
        }
    subset_policy = multi_family_subset_policy(definitions, arm_payloads)

    if headline is None:
        summary_token = MEASUREMENT_UNDEFINED
        summary_basis = ["the headline challenger arm was not evaluated"]
    else:
        summary_token = headline["measurement"]["token"]
        summary_basis = list(headline["measurement"]["basis"])

    scored_candidates = len(records)
    excluded_candidates = sum(int(item["candidates"]) for item in excluded)
    identity_bases: dict[str, int] = {}
    for item in records:
        basis = str(item.get("identity_basis") or challenger.IDENTITY_BASIS_UNRESOLVED)
        identity_bases[basis] = identity_bases.get(basis, 0) + 1
    cutoff_safe_rows = sum(1 for item in records if item.get("identity_cutoff_safe"))
    stale_capture_rows = sum(1 for item in records if item.get("identity_capture_is_newest") is False)
    exclusions_by_scope: dict[str, int] = {}
    for item in excluded:
        scope = str(item.get("scope") or SCOPE_PLAYER)
        exclusions_by_scope[scope] = exclusions_by_scope.get(scope, 0) + 1

    identity = challenger.challenger_identity(challenger_config, config)
    incumbent_identity = {
        "minutes_model_version": incumbent.MINUTES_MODEL_VERSION,
        "minutes_coherent_model_version": incumbent.MINUTES_COHERENT_MODEL_VERSION,
        "incumbent_config_hash": config.config_hash(),
        "incumbent_identity_check": identity["incumbent_identity"],
    }
    return {
        "schema": EVALUATION_VERSION,
        "grain": GRAIN_PLAYER_FIXTURE,
        "grain_note": GRAIN_NOTE,
        "cutoff_policy": CUTOFF_POLICY,
        "missing_data_policy_version": MISSING_DATA_POLICY_VERSION,
        "population_rule": {
            "candidates": (
                "one slot per candidate player-fixture of each target event, plus one slot for "
                "a candidate whose club has no fixture in the event (he has no fixture to "
                "enumerate).  The candidate pool is the official pool AS OF THE CUTOFF: the "
                "players the cutoff's official element captures place in the pool, plus the "
                "persisted pool with a recorded fallback where the cutoff holds no capture"
            ),
            "scored": (
                "a finalised event, a played fixture, a model projection from every arm, and a "
                "non-placeholder official outcome row for that exact player-fixture"
            ),
            "never_scored_as_zero": [
                "a blank event (no fixture)",
                "an unfinalised event",
                "a fixture that has not started and finished",
                "a scheduled placeholder row",
                "a missing official row",
                "a missing projection",
                "a candidate the cutoff cannot place in the official pool",
            ],
            "exclusion_statuses": list(EXCLUSION_STATUSES),
            "double_gameweek": (
                "scored: a double gameweek is two player-fixture observations, one per fixture"
            ),
            "accounting_unit": "candidate slot (see population_rule.candidates)",
        },
        "identity": {
            "evaluation_version": EVALUATION_VERSION,
            "challenger": identity,
            "incumbent": incumbent_identity,
            "code_revision": analytics.code_revision(),
            "push_or_promotion_performed": False,
            "promotion_note": (
                "this artifact is evidence for senior review; the incumbent stays authoritative "
                "and no version identifier is re-pointed by producing it"
            ),
        },
        "outcome_evidence": {
            "source": "the official player_gameweeks row for an officially FINAL event",
            "event_finality_predicate": "planning.event_data_state == FINAL (events.finished=1 and data_checked=1)",
            "pe5_ledger": "NOT_THE_READER_FOR_A_WALK_FORWARD_EVENT_SET",
            "pe5_ledger_basis": (
                "the PE-5 reality ledger relates ONE frozen prediction generation (identified by a "
                "freeze certificate) to reality; a walk-forward historical event set has no freeze "
                "certificate to relate, so the ledger is not the outcome reader here"
            ),
            "pe5_reuse": (
                "PE-1's frozen causal boundary supplies both arms' inputs; an outcome append through "
                "outcome_ledger.capture_observation is proven not to change an earlier prediction"
            ),
        },
        "population": {
            "target_events": target_events,
            "event_request": event_request,
            "events_with_observations": events_with_observations,
            "events_excluded": events_excluded,
            "candidates": scored_candidates + excluded_candidates,
            "scored": scored_candidates,
            "excluded": excluded_candidates,
            "excluded_by_status": dict(sorted(status_counts.items())),
            "exclusion_records": len(excluded),
            "exclusions_by_scope": dict(sorted(exclusions_by_scope.items())),
            "accounting": {
                "unit": "candidate slot",
                "identity": "candidates == scored + excluded",
                "candidates": scored_candidates + excluded_candidates,
                "scored": scored_candidates,
                "excluded": excluded_candidates,
                "reconciles": scored_candidates + excluded_candidates
                == scored_candidates + excluded_candidates,
                "event_scope_exclusions": exclusions_by_scope.get(SCOPE_EVENT, 0),
                "event_scope_candidates": sum(
                    int(item["candidates"])
                    for item in excluded
                    if str(item.get("scope")) == SCOPE_EVENT
                ),
                "note": (
                    "an event whose cutoff is unavailable excludes its whole candidate pool in one "
                    "EVENT-scope record; that record accounts for its candidates, so the status "
                    "counts and the totals still add up"
                ),
            },
            "identity": {
                "scored_rows_by_basis": dict(sorted(identity_bases.items())),
                "scored_rows_cutoff_safe": cutoff_safe_rows,
                "scored_rows": scored_candidates,
                "cutoff_safe_share": (
                    round(cutoff_safe_rows / scored_candidates, wm.METRIC_DECIMALS)
                    if scored_candidates
                    else None
                ),
                "basis_vocabulary": list(challenger.IDENTITY_BASES),
                "live_fallback_allowed": bool(allow_live_identity_fallback),
                "scored_rows_from_older_captures": stale_capture_rows,
                "stale_capture_rule": (
                    "a scored row whose capture was not the newest one visible at its cutoff is "
                    "counted here rather than dropped: the freshest official list may not have "
                    "named him, and that is doubt rather than evidence"
                ),
                "rule": (
                    "membership, club and position are resolved as of each cutoff from the "
                    "official element captures that were observable then"
                ),
                "incumbent_pool_identity": {
                    "read": (
                        "minutes_model.league_pools reads the persisted players row for its "
                        "aggregate position and team-position priors"
                    ),
                    "pe6_disclosure": (
                        "PE-6 resolves each CANDIDATE's membership, club and position as of the "
                        "cutoff and does not restate the incumbent's own pooling read.  A later "
                        "change to the persisted row can therefore still move the incumbent's "
                        "AGGREGATE priors; that is the incumbent's frozen behaviour and is not "
                        "PE-6's to rewrite"
                    ),
                },
            },
            "population_digest": wf.canonical_population_digest(
                population_keys, grain=GRAIN_PLAYER_FIXTURE
            ),
            "shared_population_across_arms": True,
            "shared_population_enforced_by": "walk_forward.assert_same_population",
            "coverage_share": (
                round(scored_candidates / (scored_candidates + excluded_candidates), wm.METRIC_DECIMALS)
                if (scored_candidates + excluded_candidates)
                else None
            ),
        },
        "sample": {
            "target_events": len(target_events),
            "target_events_with_observations": len(events_with_observations),
            "player_fixture_observations": len(records),
            "sample_interpretation": sample_interpretation,
            "policy": {
                "policy_version": wfs.SAMPLE_POLICY_VERSION,
                "min_target_events_for_descriptive_reporting": wfs.MIN_TARGET_EVENTS_FOR_DESCRIPTIVE,
                "min_observations_for_descriptive_reporting": wfs.MIN_OBSERVATIONS_FOR_DESCRIPTIVE,
                "basis": (
                    "PE-2's declared disclosure floor, reused verbatim; NOT a significance test, "
                    "and no arm is described as better, superior, proven or calibrated"
                ),
                "grain_of_the_observation_count": GRAIN_PLAYER_FIXTURE,
                "grain_note": (
                    "the floor is applied to the scored player-fixture observations; PE-2 declares "
                    "the floor at the event grain, and a fixture-grain count is larger, so this is "
                    "never a stricter gate than PE-2's own"
                ),
            },
            "stratum_policy": {
                "min_observations_for_stratum_reporting": MIN_STRATUM_OBSERVATIONS,
                "basis": "a declared disclosure floor, not a significance test",
            },
        },
        "incumbent_metrics": incumbent_metrics,
        "arms": arm_payloads,
        "headline_challenger_arm": challenger.ARM_ALL_REFINEMENTS,
        "refinement_families": family_measurements,
        "multi_family_subset_policy": subset_policy,
        "measurement_summary": {
            "token": summary_token,
            "basis": summary_basis,
            "headline_arm": challenger.ARM_ALL_REFINEMENTS,
            "model_selection": SELECTION_CLAIM_NOT_MADE,
            "claims_allowed_on_this_artifact": list(CLAIMS_ALLOWED),
            "claims_not_allowed_on_this_artifact": list(CLAIMS_NOT_ALLOWED),
            "review_required": True,
            "measurement_vocabulary": list(MEASUREMENT_VOCABULARY),
            "primary_criteria": list(PRIMARY_CRITERIA),
            "improvement_floors": {
                "brier": DELTA_BRIER_IMPROVEMENT_MIN,
                "expected_minutes_mae": DELTA_MAE_IMPROVEMENT_MIN,
                "expected_minutes_bias": DELTA_BIAS_IMPROVEMENT_MIN,
            },
            "sample_interpretation": sample_interpretation,
            "note": (
                "this artifact measures; it does not accept, endorse or promote an arm.  A family "
                "combination is only reportable through an arm that evaluates it, and the phase "
                "state is set at senior review"
            ),
        },
        "measurement_rule": measurement_rule_block(),
        "terminal_boundary": {
            "phase_state_before_review": "OPEN",
            "state_vocabulary": ["READY_FOR_MERGE", "OPEN"],
            "set_by": "senior review",
            "phase_state_note": (
                "the state is decided at PE-6's accepted terminal boundary; this artifact supplies "
                "the evidence and sets no phase state"
            ),
            "optional_hardening_after_ready_for_merge": False,
        },
        "strata": (
            stratum_blocks(records, arm=challenger.ARM_ALL_REFINEMENTS)
            if records
            else {dimension: {"strata": {}} for dimension in STRATUM_DIMENSIONS}
        ),
        "events": event_blocks,
        "exclusions": excluded,
    }
