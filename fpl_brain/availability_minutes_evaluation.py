"""PE-6 evaluation: incumbent vs challenger on identical walk-forward populations.

WHAT THIS PRODUCES
------------------
One evidence artifact.  Not a promotion, not a certification, not a merge
decision: the incumbent remains authoritative and PE-6 ends at senior review.
The artifact exists so that a reviewer can see, for a declared population, on
which families the challenger moved the prediction and what that did to the
error metrics -- including when the honest answer is "nothing measurable".

IDENTICAL POPULATIONS, ENFORCED
-------------------------------
Every arm is projected from the same cutoff with the same PE-1 causal boundary,
and the population is classified ONCE and shared by every arm.  Before any
metric is computed, ``walk_forward.assert_same_population`` proves that the
incumbent and each arm cover exactly the same key set; a mismatch stops the
evaluation instead of producing a comparison of different worlds.  The pooled
population carries a ``walk_forward.canonical_population_digest``.

WHAT IS SCORED
--------------
Grain ``player-event`` (PE-2's grain).  A double gameweek is excluded from the
point-metric population with an explicit status
(``MULTI_FIXTURE_EVENT_POINT_AGGREGATION_UNSPECIFIED``): the frozen incumbent
publishes per-fixture projections, and PE-4's aggregation contract is defined
per simulation world, so there is no frozen rule for aggregating two fixtures'
*point* probabilities.  Inventing one would be an out-of-scope modelling
assumption, so multi-fixture player-events are counted, labelled and left
unscored -- never scored as zero.

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
The PE-2 policy decides whether the sample can support anything: below its
declared floors the artifact says ``INSUFFICIENT_FOR_STRONG_MODEL_SELECTION``
and the challenger outcome is ``NO_CHANGE``.  Nothing here is described as
better, superior or proven.  Strata below their own declared floor are labelled
insufficient with their ``n`` rather than silently omitted.

THE DECISION RULE IS DECLARED, NOT DISCOVERED
---------------------------------------------
Criteria and floors are declared a priori in this module (see
``PRIMARY_CRITERIA`` / ``DELTA_BRIER_IMPROVEMENT_MIN`` /
``DELTA_MAE_IMPROVEMENT_MIN``) and repeated inside the artifact.  A change that
does not exceed its floor on an adequate sample is ``WITHIN_NOISE_FLOOR`` and
is not an improvement.  Every family is also evaluated alone, so a partially
accepted refinement is expressible as a per-family result rather than as a
compromise number.
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

EVALUATION_VERSION = "availability_minutes_evaluation_v1.0.0"

GRAIN_PLAYER_EVENT = wf.GRAIN_PLAYER_EVENT
GRAIN_NOTE = (
    "player x event; a double gameweek is one summed observation, never one per fixture"
)

#: The cutoff a walk-forward prediction is taken at: the target event's official
#: deadline, which is the last moment a manager could have acted.  A deadline
#: that is not before the event's first kickoff is refused rather than used.
CUTOFF_POLICY = "TARGET_EVENT_DEADLINE_TIME"
MISSING_DATA_POLICY_VERSION = wf.MISSING_DATA_POLICY_VERSION

# --- exclusion statuses (excluded candidates are counted, never scored zero) --
#: Candidates are enumerated from the official pool (active players with a club),
#: so a player outside it is not a candidate at all rather than an exclusion.
STATUS_BLANK_NO_FIXTURE = "TARGET_BLANK_NO_FIXTURE"
STATUS_MULTI_FIXTURE_UNSPECIFIED = "MULTI_FIXTURE_EVENT_POINT_AGGREGATION_UNSPECIFIED"
STATUS_OUTCOME_NOT_FINALISED = wf.OUTCOME_NOT_FINALISED
STATUS_OUTCOME_MISSING = "OUTCOME_EVIDENCE_MISSING"
STATUS_OUTCOME_PLACEHOLDER = wf.OUTCOME_PLACEHOLDER_EXCLUDED
STATUS_MODEL_PROJECTION_MISSING = wf.MODEL_PROJECTION_MISSING
STATUS_EVENT_CUTOFF_UNAVAILABLE = "TARGET_EVENT_CUTOFF_UNAVAILABLE"
STATUS_EVENT_NOT_FINAL = "TARGET_EVENT_OUTCOME_NOT_FINALISED"

EXCLUSION_STATUSES: tuple[str, ...] = (
    STATUS_BLANK_NO_FIXTURE,
    STATUS_MULTI_FIXTURE_UNSPECIFIED,
    STATUS_OUTCOME_NOT_FINALISED,
    STATUS_OUTCOME_MISSING,
    STATUS_OUTCOME_PLACEHOLDER,
    STATUS_MODEL_PROJECTION_MISSING,
    STATUS_EVENT_CUTOFF_UNAVAILABLE,
    STATUS_EVENT_NOT_FINAL,
)

# --- declared decision rule ---------------------------------------------------
PREFERRED_CHALLENGER = "CHALLENGER"
PREFERRED_INCUMBENT = "INCUMBENT"
WITHIN_NOISE_FLOOR = "WITHIN_DECLARED_NOISE_FLOOR"

OUTCOME_CHALLENGER_ACCEPTED = "CHALLENGER_ACCEPTED"
OUTCOME_PARTIALLY_ACCEPTED = "PARTIALLY_ACCEPTED"
OUTCOME_NO_CHANGE = "NO_CHANGE"

OUTCOME_VOCABULARY: tuple[str, ...] = (
    OUTCOME_CHALLENGER_ACCEPTED,
    OUTCOME_PARTIALLY_ACCEPTED,
    OUTCOME_NO_CHANGE,
)

NO_CHANGE_BASIS_INSUFFICIENT_SAMPLE = "INSUFFICIENT_SAMPLE_UNDER_THE_PE2_POLICY"
NO_CHANGE_BASIS_CHALLENGER_FAILS = "CHALLENGER_WORSENS_A_PRIMARY_CRITERION"
NO_CHANGE_BASIS_WITHIN_NOISE = "NOT_DISTINGUISHABLE_FROM_NOISE_ON_THIS_SAMPLE"
NO_CHANGE_BASIS_UNDEFINED = "PRIMARY_CRITERION_UNDEFINED"

#: Metrics the verdict is computed on.  Declared, never selected from the data.
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


def arm_outcome(
    comparison: Mapping[str, Any], *, sample_sufficient: bool
) -> dict[str, Any]:
    """The declared verdict for one arm.  Pure function of the comparison.

    Order of refusal matters: an inadequate sample is reported as an inadequate
    sample, never as a failed challenger, and never as a selection claim.
    """

    metrics = comparison.get("metrics") or {}
    if not sample_sufficient:
        return {
            "token": OUTCOME_NO_CHANGE,
            "basis": [NO_CHANGE_BASIS_INSUFFICIENT_SAMPLE],
            "note": (
                "the PE-2 sample policy does not support model selection on this population; "
                "the challenger is not accepted and the incumbent is not endorsed"
            ),
        }
    undefined = [
        name
        for name in PRIMARY_CRITERIA
        if (metrics.get(name) or {}).get("status") != "OK"
    ]
    if undefined:
        return {
            "token": OUTCOME_NO_CHANGE,
            "basis": [NO_CHANGE_BASIS_UNDEFINED, *[f"{name}:UNDEFINED" for name in undefined]],
            "note": "a primary criterion could not be computed, so no verdict is claimed",
        }
    preferred = {name: (metrics.get(name) or {}).get("preferred") for name in PRIMARY_CRITERIA}
    worsened = sorted(name for name, value in preferred.items() if value == PREFERRED_INCUMBENT)
    improved = sorted(name for name, value in preferred.items() if value == PREFERRED_CHALLENGER)
    if worsened:
        return {
            "token": OUTCOME_NO_CHANGE,
            "basis": [NO_CHANGE_BASIS_CHALLENGER_FAILS, *[f"{name}:worsened" for name in worsened]],
            "note": "the challenger worsens a primary criterion, so it is not an improvement",
        }
    if not improved:
        return {
            "token": OUTCOME_NO_CHANGE,
            "basis": [NO_CHANGE_BASIS_WITHIN_NOISE],
            "note": (
                "every primary criterion is inside its declared floor, so the challenger "
                "cannot be distinguished from the incumbent on this sample"
            ),
        }
    if len(improved) == len(PRIMARY_CRITERIA):
        return {
            "token": OUTCOME_CHALLENGER_ACCEPTED,
            "basis": [f"{name}:improved" for name in improved],
            "note": "every primary criterion improves beyond its declared floor",
        }
    return {
        "token": OUTCOME_PARTIALLY_ACCEPTED,
        "basis": [
            *[f"{name}:improved" for name in improved],
            *[
                f"{name}:within_noise"
                for name, value in preferred.items()
                if value == WITHIN_NOISE_FLOOR
            ],
        ],
        "note": "some primary criteria improve and none worsen",
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
    """Classify every official-pool candidate of one target event, once.

    The SAME classification is used for every arm, so the populations cannot
    differ between them by construction; the arms' own key coverage is asserted
    separately in :func:`evaluate_events`.
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
    pool = {int(row["player_id"]): row for row in analytics.projectable_players(conn)}
    fixtures_by_team = analytics.event_fixture_map(conn, int(event))
    records: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {}

    def exclude(player_id: int, team_id: int | None, status: str, detail: str, fixtures: Sequence[int]) -> None:
        excluded.append(
            {
                "event": int(event),
                "player_id": int(player_id),
                "team_id": None if team_id is None else int(team_id),
                "fixtures": sorted(int(value) for value in fixtures),
                "status": status,
                "detail": detail,
            }
        )
        status_counts[status] = status_counts.get(status, 0) + 1

    for player_id in sorted(pool):
        player = pool[player_id]
        team_id = int(player["team_id"])
        fixtures = [int(fixture["id"]) for fixture in fixtures_by_team.get(team_id, [])]
        if not fixtures:
            exclude(player_id, team_id, STATUS_BLANK_NO_FIXTURE, "no fixture in the target event", [])
            continue
        if state != "FINAL":
            exclude(player_id, team_id, STATUS_EVENT_NOT_FINAL, f"event state {state}", fixtures)
            continue
        if len(fixtures) != 1:
            exclude(
                player_id,
                team_id,
                STATUS_MULTI_FIXTURE_UNSPECIFIED,
                f"{len(fixtures)} fixtures in the event; point-probability aggregation is unspecified",
                fixtures,
            )
            continue
        fixture_id = fixtures[0]
        if arms is None or (player_id, fixture_id) not in arms.incumbent_rows:
            exclude(
                player_id,
                team_id,
                STATUS_MODEL_PROJECTION_MISSING,
                "the model produced no projection for this player-fixture",
                fixtures,
            )
            continue
        outcome, outcome_status = realised_outcome(conn, player_id, fixture_id)
        if outcome is None:
            exclude(player_id, team_id, outcome_status, "official outcome evidence unavailable", fixtures)
            continue
        incumbent_row = arms.incumbent_rows[(player_id, fixture_id)]
        challenger_row = arms.arms[challenger.ARM_ALL_REFINEMENTS][(player_id, fixture_id)]
        evidence = arms.evidence[(player_id, fixture_id)]
        records.append(
            {
                "event": int(event),
                "player_id": int(player_id),
                "fixture_id": int(fixture_id),
                "team_id": team_id,
                "position_id": int(player.get("element_type") or 0),
                "availability_basis": evidence.availability_basis,
                "status_at_cutoff": evidence.status_at_cutoff,
                "return_ramp": bool(evidence.ramp_applies),
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
    records.sort(key=lambda item: (int(item["event"]), int(item["player_id"])))
    excluded.sort(key=lambda item: (int(item["event"]), int(item["player_id"]), str(item["status"])))
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


def evaluate_events(
    conn: sqlite3.Connection,
    events: Sequence[int],
    *,
    incumbent_config: "incumbent.MinutesModelConfig | None" = None,
    challenger_config: challenger.AvailabilityMinutesChallengerConfig | None = None,
    arm_definitions: Mapping[str, frozenset[str]] | None = None,
) -> dict[str, Any]:
    """Compare the incumbent and the challenger over a walk-forward event set."""

    config = incumbent_config or incumbent.MinutesModelConfig()
    challenger_config = challenger_config or challenger.AvailabilityMinutesChallengerConfig()
    definitions = dict(arm_definitions or challenger.default_arm_definitions())
    if challenger.ARM_ALL_REFINEMENTS not in definitions:
        raise EvaluationError(
            f"the evaluation requires the headline arm {challenger.ARM_ALL_REFINEMENTS!r} "
            "in the arm definitions: the population is classified from it"
        )

    records: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {}
    event_blocks: list[dict[str, Any]] = []
    events_with_observations: list[int] = []
    events_excluded: list[dict[str, Any]] = []

    for event in sorted(int(value) for value in events):
        cutoff, cutoff_reasons = event_cutoff(conn, event)
        arms = (
            challenger.build_challenger_arms(
                conn,
                event,
                cutoff,
                incumbent_config=config,
                challenger_config=challenger_config,
                arm_definitions=definitions,
            )
            if cutoff is not None
            else None
        )
        event_records, event_excluded, event_status_counts, event_reasons = build_event_records(
            conn, event, cutoff=cutoff, arms=arms
        )
        if cutoff is None:
            candidates = len(analytics.projectable_players(conn))
            status_counts[STATUS_EVENT_CUTOFF_UNAVAILABLE] = (
                status_counts.get(STATUS_EVENT_CUTOFF_UNAVAILABLE, 0) + candidates
            )
            excluded.append(
                {
                    "event": event,
                    "player_id": None,
                    "team_id": None,
                    "fixtures": [],
                    "status": STATUS_EVENT_CUTOFF_UNAVAILABLE,
                    "detail": "; ".join(cutoff_reasons),
                    "candidates": candidates,
                }
            )
            events_excluded.append({"event": event, "reasons": cutoff_reasons})
            event_blocks.append(
                {
                    "event": event,
                    "status": "NOT_EVALUATED",
                    "reasons": cutoff_reasons,
                    "scored_observations": 0,
                }
            )
            continue
        # The arms must cover the identical key set before anything is scored.
        arms.verify_same_population()
        records.extend(event_records)
        excluded.extend(event_excluded)
        for status, count in event_status_counts.items():
            status_counts[status] = status_counts.get(status, 0) + count
        if event_records:
            events_with_observations.append(event)
        event_blocks.append(
            {
                "event": event,
                "status": "EVALUATED" if event_records else "NOT_EVALUATABLE",
                "cutoff": cutoff,
                "cutoff_policy": CUTOFF_POLICY,
                "reasons": event_reasons,
                "candidates": len(event_records) + len(event_excluded),
                "scored_observations": len(event_records),
                "excluded_by_status": dict(sorted(event_status_counts.items())),
                "population_digest": wf.canonical_population_digest(
                    [(int(item["event"]), int(item["player_id"])) for item in event_records],
                    grain=GRAIN_PLAYER_EVENT,
                ),
            }
        )

    realisation = [item["realised"] for item in records]
    population_keys = [(int(item["event"]), int(item["player_id"])) for item in records]
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
        payload["outcome"] = arm_outcome(payload["comparison"], sample_sufficient=sample_sufficient)

    headline = arm_payloads.get(challenger.ARM_ALL_REFINEMENTS)
    family_results: dict[str, Any] = {}
    for family in challenger.REFINEMENT_FAMILIES:
        arm_name = challenger.arm_name_for_family(family)
        payload = arm_payloads.get(arm_name)
        if payload is None:
            continue
        family_results[family] = {
            "arm": arm_name,
            "outcome": payload["outcome"],
            "comparison": payload["comparison"],
        }
    accepted_families = sorted(
        family
        for family, payload in family_results.items()
        if payload["outcome"]["token"] in {OUTCOME_CHALLENGER_ACCEPTED, OUTCOME_PARTIALLY_ACCEPTED}
    )
    if headline is None:
        recommendation = {
            "token": OUTCOME_NO_CHANGE,
            "basis": ["the headline challenger arm was not evaluated"],
            "accepted_refinement_families": [],
        }
    elif headline["outcome"]["token"] == OUTCOME_CHALLENGER_ACCEPTED:
        recommendation = {
            "token": OUTCOME_CHALLENGER_ACCEPTED,
            "basis": list(headline["outcome"]["basis"]),
            "accepted_refinement_families": sorted(challenger.REFINEMENT_FAMILIES),
        }
    elif accepted_families:
        recommendation = {
            "token": OUTCOME_PARTIALLY_ACCEPTED,
            "basis": [
                "no single arm improves every primary criterion, but "
                f"{len(accepted_families)} refinement family(ies) improve on this population"
            ],
            "accepted_refinement_families": accepted_families,
        }
    else:
        recommendation = {
            "token": OUTCOME_NO_CHANGE,
            "basis": list(headline["outcome"]["basis"]),
            "accepted_refinement_families": [],
        }

    identity = challenger.challenger_identity(challenger_config, config)
    incumbent_identity = {
        "minutes_model_version": incumbent.MINUTES_MODEL_VERSION,
        "minutes_coherent_model_version": incumbent.MINUTES_COHERENT_MODEL_VERSION,
        "incumbent_config_hash": config.config_hash(),
        "incumbent_identity_check": identity["incumbent_identity"],
    }
    return {
        "schema": EVALUATION_VERSION,
        "grain": GRAIN_PLAYER_EVENT,
        "grain_note": GRAIN_NOTE,
        "cutoff_policy": CUTOFF_POLICY,
        "missing_data_policy_version": MISSING_DATA_POLICY_VERSION,
        "population_rule": {
            "candidates": "every active player with an official club, per target event",
            "scored": (
                "exactly one fixture in the target event, a finalised event, a model projection "
                "from both arms, and a non-placeholder official outcome row"
            ),
            "never_scored_as_zero": [
                "a blank event (no fixture)",
                "a multi-fixture (double gameweek) player-event, whose point-probability "
                "aggregation is unspecified",
                "an unfinalised event",
                "a scheduled placeholder row",
                "a missing official row",
                "a missing projection",
            ],
            "exclusion_statuses": list(EXCLUSION_STATUSES),
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
            "target_events": sorted(int(value) for value in events),
            "events_with_observations": events_with_observations,
            "events_excluded": events_excluded,
            "candidates": len(records) + len(excluded),
            "scored": len(records),
            "excluded": len(excluded),
            "excluded_by_status": dict(sorted(status_counts.items())),
            "population_digest": wf.canonical_population_digest(
                population_keys, grain=GRAIN_PLAYER_EVENT
            ),
            "shared_population_across_arms": True,
            "shared_population_enforced_by": "walk_forward.assert_same_population",
            "coverage_share": (
                round(len(records) / (len(records) + len(excluded)), wm.METRIC_DECIMALS)
                if (len(records) + len(excluded))
                else None
            ),
        },
        "sample": {
            "target_events": len(set(int(value) for value in events)),
            "target_events_with_observations": len(events_with_observations),
            "player_event_observations": len(records),
            "sample_interpretation": sample_interpretation,
            "policy": {
                "policy_version": wfs.SAMPLE_POLICY_VERSION,
                "min_target_events_for_descriptive_reporting": wfs.MIN_TARGET_EVENTS_FOR_DESCRIPTIVE,
                "min_observations_for_descriptive_reporting": wfs.MIN_OBSERVATIONS_FOR_DESCRIPTIVE,
                "basis": (
                    "PE-2's declared disclosure floor, reused verbatim; NOT a significance test, "
                    "and no arm is described as better, superior, proven or calibrated"
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
        "refinement_families": family_results,
        "recommendation": {
            **recommendation,
            "review_required": True,
            "outcome_vocabulary": list(OUTCOME_VOCABULARY),
            "primary_criteria": list(PRIMARY_CRITERIA),
            "improvement_floors": {
                "brier": DELTA_BRIER_IMPROVEMENT_MIN,
                "expected_minutes_mae": DELTA_MAE_IMPROVEMENT_MIN,
                "expected_minutes_bias": DELTA_BIAS_IMPROVEMENT_MIN,
            },
            "sample_interpretation": sample_interpretation,
        },
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
