"""Deterministic walk-forward scoreboard: structured metrics and immutable artifact.

WHAT THIS IS
------------
PE-2C decided *which* observations are eligible and proved that two arms are only
comparable when they cover the identical eligible key set.  This module takes that
population and answers the reporting question:

    On the exact same eligible population, how accurate was the model compared
    with each valid naive points baseline?

It computes measurements.  It does not rank arms, does not name a winner, and does
not derive a skill score.  Choosing a model is a separate judgement needing far
more than one Gameweek of evidence, and the artifact says so in a machine-readable
field rather than leaving a reader to infer confidence from a tidy table.

WHAT IT READS
-------------
Persisted predictions and persisted outcomes, through the frozen PE-1 and PE-2C
contracts.  It writes no database row, adds no schema and never generates a
prediction.  Its only output is a content-addressed JSON artifact plus a Markdown
rendering of the same data.

SAME POPULATION, ENFORCED PER COMPARISON
----------------------------------------
PE-2C's :func:`walk_forward.assert_same_population` is applied to every arm against
the one shared comparison key set.  An arm that cannot cover that exact key set is
reported as unavailable and is given **no metric at all** -- never a score on a
smaller sample with a quietly different N.

This is deliberately per-comparison rather than one global abort.  The live P90
baseline genuinely has no value for players with no completed history, so a global
abort would make the artifact permanently unproducible -- and a gate that can never
pass gets disabled.  Failing the *comparison* keeps the guarantee that matters (no
number is ever reported on a mismatched population) while letting other arms be
measured.

NOTHING MISSING BECOMES ZERO
----------------------------
A blank Gameweek, an unplayed fixture, a scheduled placeholder, an absent baseline
value and a missing projection each keep their own name and are counted.  A metric
that is undefined comes back as ``value: null`` with a status token, never as 0.

PER-EVENT RUN IDENTITY
----------------------
Every run this artifact consumes is recorded per target event.  A certified bundle
declares one run set per event, and collapsing those into a single family->run map
would name only the last event's runs; the per-event map is the only record that is
complete for a multi-event target.

DETERMINISM
-----------
There is no wall-clock timestamp anywhere in the artifact.  The only times present
are supplied observations (the anchor's planning cutoff) and they are part of the
identity.  Same population + same predictions + same outcomes + same policy gives
byte-identical canonical JSON and therefore the same digest.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import analytics
from . import calibration
from . import repositories as repo
from . import walk_forward as wf
from . import walk_forward_metrics as wm
from .scoring_rules import DEFAULT_SCORING_RULES, POSITION_IDS

SCOREBOARD_SCHEMA_VERSION = "wf_scoreboard_v1.0.0"

#: Evaluation output gets its own narrow directory, so a scoreboard can never be
#: mistaken for a decision, a certification or a manager packet.
SCOREBOARD_RELATIVE_DIR = "data/evaluation"

STATUS_EVALUATED = "EVALUATED"
STATUS_PARTIALLY_EVALUATABLE = "PARTIALLY_EVALUATABLE"
STATUS_NOT_YET_EVALUATABLE = "NOT_YET_EVALUATABLE"

SCOREBOARD_STATUSES = (STATUS_EVALUATED, STATUS_PARTIALLY_EVALUATABLE, STATUS_NOT_YET_EVALUATABLE)

ARM_MODEL = "MODEL_XPTS"
ARM_POPULATION_MISMATCH = "POPULATION_MISMATCH"
ARM_OK = "OK"

#: Sample-size honesty.  These are DECLARED disclosure floors, not significance
#: thresholds: nothing here tests a hypothesis and no arm is ever called better.
#: Both labels are deliberately free of superiority claims.
SAMPLE_POLICY_VERSION = "wf_sample_policy_v1.0.0"
SAMPLE_INSUFFICIENT = "INSUFFICIENT_FOR_STRONG_MODEL_SELECTION"
SAMPLE_DESCRIPTIVE_ONLY = "SUFFICIENT_FOR_DESCRIPTIVE_REPORTING_ONLY"
MIN_TARGET_EVENTS_FOR_DESCRIPTIVE = 10
MIN_OBSERVATIONS_FOR_DESCRIPTIVE = 2000

#: Declared once, carried in the identity, never re-derived by a reader.
BIAS_CONVENTION = "mean(predicted - actual); positive means overprediction"
TOP_K_AGGREGATION = "MEAN_OVER_TARGET_EVENTS"
MEDIAN_CONVENTION = "statistics.median (mean of the two central values when the count is even)"
SPEARMAN_TIE_POLICY = "average ranks; undefined (never 0) when either side is constant"
TOP_K_TIE_POLICY = "score descending, then player_id ascending"
TARGET_GRAIN_NOTE = (
    "player x event; a double gameweek is one summed observation, never one per fixture"
)


class ScoreboardError(RuntimeError):
    """The scoreboard could not be produced from the available identity."""


class ScoreboardIdentityCollision(ScoreboardError):
    """An artifact already exists at this identity with different bytes."""


# ---------------------------------------------------------------------------
# Declared metric definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProbabilityMetricDefinition:
    """One Brier metric: the persisted field, and the outcome it is scored against."""

    metric: str
    field: str
    outcome: str
    outcome_selector: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "field": self.field,
            "grain": wf.GRAIN_PLAYER_FIXTURE,
            "realised_outcome": self.outcome,
            "range_check": (
                "every scored probability must lie in [0, 1] or the scoreboard fails closed"
            ),
            "missing_policy": (
                "a row whose probability is absent is excluded and counted; it is never scored as 0.0"
            ),
        }


#: The probabilities this slice scores.  Each is persisted in the anchor's own
#: ``xpts_v1`` payload, so a Brier score needs no second run family and no
#: cross-run provenance assumption.
#:
#: Deliberately NOT scored here:
#:   * ``p_goal`` / ``p_assist`` exist only in ``monte_carlo_distributions``, a
#:     different run family with its own dependency closure.  Scoring them would
#:     widen this slice's provenance surface with no V1 requirement behind it.
#:   * every ``*_xpts`` component is an EXPECTED POINTS value, not a probability.
#:     A Brier score computed from expected points would be meaningless, so this
#:     module does not compute one.
PROBABILITY_METRICS: tuple[ProbabilityMetricDefinition, ...] = (
    ProbabilityMetricDefinition(
        metric="BRIER_P_START",
        field="p_start",
        outcome="player_gameweeks.starts = 1 for this player and this fixture",
        outcome_selector="started",
    ),
    ProbabilityMetricDefinition(
        metric="BRIER_P_60_PLUS",
        field="p_60_plus",
        outcome="player_gameweeks.minutes >= 60 for this player and this fixture",
        outcome_selector="played_60",
    ),
    ProbabilityMetricDefinition(
        metric="BRIER_CLEAN_SHEET",
        field="clean_sheet_probability",
        outcome=(
            "the player earned clean-sheet points: minutes >= 60, zero goals conceded while on, "
            "and clean_sheet_points_for(position) > 0"
        ),
        outcome_selector="clean_sheet_points",
    ),
    ProbabilityMetricDefinition(
        metric="BRIER_DEFCON",
        field="defcon_p_hit",
        outcome="player_gameweeks.defensive_contribution >= defcon_threshold_for(position)",
        outcome_selector="defcon_hit",
    ),
)

PROBABILITY_POPULATION_POLICY = (
    "fixture grain, restricted to the anchor xpts_v1 run of each target event, for played fixtures of "
    "FINAL target events; a row is scored only when the realised player_gameweeks row exists for the "
    "same (player, fixture), is not a scheduled placeholder, has the required outcome column non-null, "
    "and the position defines the required threshold (DefCon excludes GKP, which has none)"
)

#: Monte Carlo quantiles are stored on a grid carrying the CORE components only.
#: Bonus is deterministic in that kernel, so there is no total-points distribution
#: and no CRPS.  The labels below say COVERAGE, never "calibrated predictive
#: interval".
QUANTILE_POLICY: dict[str, Any] = {
    "policy_version": "wf_quantile_policy_v1.0.0",
    "quantity": "ACTUAL_MODELLED_CORE (per player-fixture, reconstructed from proven official fixture facts)",
    "metrics": ["CENTRAL_50_QUANTILE_COVERAGE", "CENTRAL_80_QUANTILE_COVERAGE"],
    "intervals": {
        "CENTRAL_50_QUANTILE_COVERAGE": ["q25", "q75"],
        "CENTRAL_80_QUANTILE_COVERAGE": ["q10", "q90"],
    },
    "interval_kind": "closed; a realised value equal to a bound counts as covered",
    "crps": "NOT_IMPLEMENTED (the artifact stores quantiles, never draws)",
    "claim": "quantile coverage only; this is NOT a statement of complete distribution calibration",
    "aggregation": "fixture grain; quantiles are never summed into an event figure",
}


# ---------------------------------------------------------------------------
# Small read helpers
# ---------------------------------------------------------------------------


def _rows_by_key(population: wf.Population) -> dict[tuple[int, int], dict[str, Any]]:
    return {(int(row["event"]), int(row["player_id"])): row for row in population.rows}


def _model_value(row: Mapping[str, Any]) -> float | None:
    value = row.get("model_xpts")
    return None if value is None else float(value)


def _baseline_value(row: Mapping[str, Any]) -> float | None:
    value = row.get("baseline_value")
    return None if value is None else float(value)


def _realised_points(row: Mapping[str, Any]) -> float | None:
    value = (row.get("outcome") or {}).get("total_points")
    return None if value is None else float(value)


def _run_version(conn: sqlite3.Connection, run_id: int | None) -> str | None:
    if run_id is None:
        return None
    row = conn.execute(
        "SELECT model_version FROM projection_runs WHERE id=?", (int(run_id),)
    ).fetchone()
    return str(row[0]) if row else None


def _event_runs(anchor: wf.CertifiedAnchor, events: Sequence[int], family: str) -> dict[int, int]:
    """The run each target event declares for one family.  Complete, per event."""

    out: dict[int, int] = {}
    for event in events:
        run_id = anchor.for_event(int(event)).run_id(family)
        if run_id is not None:
            out[int(event)] = int(run_id)
    return out


def _baseline_runs(
    conn: sqlite3.Connection, anchor: wf.CertifiedAnchor, events: Sequence[int]
) -> dict[int, int]:
    """The baseline run belonging to each target event.

    Resolved by family + event + cutoff + code fingerprint, not by recency and not
    from the bundle's ``runs`` map: a four-GW certification declares no baseline
    member, so the anchor's own lookup is the authority for this family.
    """

    out: dict[int, int] = {}
    for event in events:
        run_id = wf.resolve_baseline_run(conn, anchor.for_event(int(event)))
        if run_id is not None:
            out[int(event)] = int(run_id)
    return out


def _played_fixture_ids(conn: sqlite3.Connection, events: Sequence[int]) -> set[int]:
    wanted = sorted({int(event) for event in events})
    if not wanted:
        return set()
    placeholders = ",".join("?" for _ in wanted)
    return {
        int(row[0])
        for row in conn.execute(
            f"SELECT id FROM fixtures WHERE finished=1 AND started=1 AND event IN ({placeholders})",
            tuple(wanted),
        )
    }


def _outcome_rows(
    conn: sqlite3.Connection, events: Sequence[int]
) -> dict[tuple[int, int], dict[str, Any]]:
    """Realised per-fixture outcome rows, scheduled placeholders removed.

    The placeholder predicate comes from the frozen PE-1 repository boundary, so
    this layer cannot drift from the causal-history contract.
    """

    wanted = sorted({int(event) for event in events})
    if not wanted:
        return {}
    placeholders = ",".join("?" for _ in wanted)
    predicate = repo.scheduled_placeholder_sql("player_gameweeks")
    excluded = {
        (int(row[0]), int(row[1]))
        for row in conn.execute(
            "SELECT player_id, fixture_id FROM player_gameweeks WHERE fixture_id > 0 "
            f"AND event IN ({placeholders}) AND ({predicate})",
            tuple(wanted),
        )
    }
    out: dict[tuple[int, int], dict[str, Any]] = {}
    for row in conn.execute(
        "SELECT player_id, fixture_id, event, minutes, starts, goals_scored, assists, clean_sheets, "
        "goals_conceded, saves, yellow_cards, defensive_contribution FROM player_gameweeks "
        f"WHERE fixture_id > 0 AND event IN ({placeholders})",
        tuple(wanted),
    ):
        key = (int(row["player_id"]), int(row["fixture_id"]))
        if key in excluded:
            continue
        out[key] = {name: row[name] for name in row.keys()}
    return out


def _positions(conn: sqlite3.Connection) -> dict[int, str]:
    return {
        int(row["id"]): POSITION_IDS[int(row["element_type"])]
        for row in conn.execute("SELECT id, element_type FROM players")
        if row["element_type"] is not None and int(row["element_type"]) in POSITION_IDS
    }


# ---------------------------------------------------------------------------
# Arm measurement
# ---------------------------------------------------------------------------


def _per_event_points(
    keys: Sequence[tuple[int, int]], predicted: Sequence[float], actual: Sequence[float]
) -> list[dict[str, Any]]:
    """N / MAE / RMSE / BIAS per target event, in ascending event order."""

    grouped: dict[int, list[int]] = {}
    for index, (event, _player_id) in enumerate(keys):
        grouped.setdefault(int(event), []).append(index)
    out: list[dict[str, Any]] = []
    for event in sorted(grouped):
        indexes = grouped[event]
        arm_predicted = [predicted[i] for i in indexes]
        arm_actual = [actual[i] for i in indexes]
        out.append(
            {
                "event": int(event),
                "N": len(indexes),
                "mae": wm.mean_absolute_error(arm_predicted, arm_actual).as_dict(),
                "rmse": wm.root_mean_squared_error(arm_predicted, arm_actual).as_dict(),
                "bias": wm.mean_bias(arm_predicted, arm_actual).as_dict(),
            }
        )
    return out


def _top_k_over_events(
    keys: Sequence[tuple[int, int]],
    predicted: Sequence[float],
    actual: Sequence[float],
    *,
    top_k: int,
) -> dict[str, Any]:
    """TOP_K_HIT_RATE, aggregated as the mean of the per-event hit rates.

    Each target event contributes one equally weighted hit rate.  With a single
    target event this is simply that event's value; the aggregation is declared in
    the identity so a reader never has to guess which of the two conventions was
    used.
    """

    grouped: dict[int, list[int]] = {}
    for index, (event, _player_id) in enumerate(keys):
        grouped.setdefault(int(event), []).append(index)
    per_event: list[wm.Metric] = []
    for event in sorted(grouped):
        indexes = grouped[event]
        per_event.append(
            wm.top_k_hit_rate(
                [(float(predicted[i]), int(keys[i][1])) for i in indexes],
                [(float(actual[i]), int(keys[i][1])) for i in indexes],
                k=top_k,
            )
        )
    available = [metric.value for metric in per_event if metric.ok]
    if not available:
        statuses = sorted({metric.status for metric in per_event}) or [wm.METRIC_NO_SAMPLE]
        return wm.Metric(
            statuses[0], n=len(keys), detail="no target event could support the declared k"
        ).as_dict()
    return wm.Metric(
        wm.METRIC_OK,
        value=sum(available) / len(available),
        n=len(keys),
        detail=f"{TOP_K_AGGREGATION} over {len(available)} target event(s)",
    ).as_dict()


def _arm_entry(
    conn: sqlite3.Connection,
    *,
    name: str,
    kind: str,
    run_id: int | None,
    keys: Sequence[tuple[int, int]],
    predicted: Sequence[float],
    actual: Sequence[float],
    top_k: int,
    population_digest: str,
) -> dict[str, Any]:
    return {
        "arm": name,
        "kind": kind,
        "version": _run_version(conn, run_id),
        "run_id": run_id,
        "N": len(keys),
        "mae": wm.mean_absolute_error(predicted, actual).as_dict(),
        "rmse": wm.root_mean_squared_error(predicted, actual).as_dict(),
        "bias": wm.mean_bias(predicted, actual).as_dict(),
        "median_ae": wm.median_absolute_error(predicted, actual).as_dict(),
        "spearman": wm.spearman_rank_correlation(predicted, actual).as_dict(),
        "top_k_hit_rate": _top_k_over_events(keys, predicted, actual, top_k=top_k),
        "coverage": {
            "status": ARM_OK,
            "shared_comparison_n": len(keys),
            "population_digest": population_digest,
            "model_only_n": 0,
            "arm_only_n": 0,
            "detail": None,
        },
        "population_digest": population_digest,
        "status": ARM_OK,
    }


def _unavailable_arm(
    conn: sqlite3.Connection, name: str, run_id: int | None, gate: Mapping[str, Any]
) -> dict[str, Any]:
    """An arm that cannot cover the shared population: no metric, and it says why."""

    unreachable = {
        "status": ARM_POPULATION_MISMATCH,
        "value": None,
        "n": 0,
        "detail": "the arm does not cover the shared comparison population",
    }
    return {
        "arm": name,
        "kind": "baseline",
        "version": _run_version(conn, run_id),
        "run_id": run_id,
        "N": None,
        "mae": dict(unreachable),
        "rmse": dict(unreachable),
        "bias": dict(unreachable),
        "median_ae": dict(unreachable),
        "spearman": dict(unreachable),
        "top_k_hit_rate": dict(unreachable),
        "coverage": dict(gate),
        "population_digest": gate.get("population_digest"),
        "status": ARM_POPULATION_MISMATCH,
    }


# ---------------------------------------------------------------------------
# Probability metrics
# ---------------------------------------------------------------------------


def _realised_probability_outcome(
    selector: str, outcome: Mapping[str, Any], position: str
) -> float | None:
    """The realised 0/1 value for one probability metric, or ``None`` for a gap."""

    minutes = outcome.get("minutes")
    if minutes is None:
        return None
    minutes = int(minutes)
    if selector == "started":
        if outcome.get("starts") is None:
            return None
        return 1.0 if int(outcome["starts"]) > 0 else 0.0
    if selector == "played_60":
        return 1.0 if minutes >= DEFAULT_SCORING_RULES.clean_sheet_minutes_required else 0.0
    if selector == "clean_sheet_points":
        conceded = outcome.get("goals_conceded")
        if conceded is None:
            return None
        achieved = (
            minutes >= DEFAULT_SCORING_RULES.clean_sheet_minutes_required
            and int(conceded) == 0
            and DEFAULT_SCORING_RULES.clean_sheet_points_for(position) > 0
        )
        return 1.0 if achieved else 0.0
    if selector == "defcon_hit":
        threshold = DEFAULT_SCORING_RULES.defcon_threshold_for(position)
        contribution = outcome.get("defensive_contribution")
        if threshold is None or contribution is None:
            # GKP has no DefCon threshold and is not a DefCon position at all.
            return None
        return 1.0 if float(contribution) >= threshold else 0.0
    raise ScoreboardError(f"unknown realised-outcome selector {selector!r}")


def probability_metrics(
    conn: sqlite3.Connection,
    *,
    events: Sequence[int],
    xpts_runs: Mapping[int, int],
) -> list[dict[str, Any]]:
    """Brier scores for the declared probabilities, on their declared populations."""

    wanted = sorted({int(event) for event in events})
    if not wanted or not xpts_runs:
        return []
    played = _played_fixture_ids(conn, wanted)
    outcomes = _outcome_rows(conn, wanted)
    positions = _positions(conn)
    collected: dict[str, list[tuple[float, float]]] = {d.metric: [] for d in PROBABILITY_METRICS}
    absent: dict[str, int] = {d.metric: 0 for d in PROBABILITY_METRICS}
    gaps: dict[str, int] = {d.metric: 0 for d in PROBABILITY_METRICS}
    versions: set[str] = set()
    for event in wanted:
        run_id = xpts_runs.get(event)
        if run_id is None:
            continue
        version = _run_version(conn, run_id)
        if version:
            versions.add(version)
        for record in conn.execute(
            "SELECT player_id, fixture_id, payload_json FROM player_fixture_xpts_projections "
            "WHERE projection_run_id=? AND event=? ORDER BY player_id, fixture_id",
            (int(run_id), int(event)),
        ):
            fixture_id = int(record["fixture_id"])
            if fixture_id not in played:
                continue
            outcome = outcomes.get((int(record["player_id"]), fixture_id))
            if outcome is None:
                continue
            payload = json.loads(record["payload_json"]) if record["payload_json"] else {}
            position = positions.get(int(record["player_id"])) or ""
            for definition in PROBABILITY_METRICS:
                raw = payload.get(definition.field)
                if raw is None:
                    absent[definition.metric] += 1
                    continue
                probability = float(raw)
                if not 0.0 <= probability <= 1.0:
                    # A stored probability outside [0, 1] is a trust failure in
                    # the producer.  Clipping would score a different question.
                    raise ScoreboardError(
                        f"{definition.field} for player {record['player_id']} fixture {fixture_id} "
                        f"is {probability!r}, outside [0, 1]"
                    )
                realised = _realised_probability_outcome(
                    definition.outcome_selector, outcome, position
                )
                if realised is None:
                    gaps[definition.metric] += 1
                    continue
                collected[definition.metric].append((probability, realised))

    out: list[dict[str, Any]] = []
    for definition in PROBABILITY_METRICS:
        pairs = collected[definition.metric]
        probabilities = [probability for probability, _y in pairs]
        realised_values = [value for _p, value in pairs]
        entry = definition.as_dict()
        entry["run_ids_by_event"] = {int(event): int(run_id) for event, run_id in sorted(xpts_runs.items())}
        entry["versions"] = sorted(versions)
        entry["brier"] = wm.brier_score(probabilities, realised_values).as_dict()
        entry["brier_reference"] = wm.brier_reference_score(realised_values).as_dict()
        entry["observed_rate"] = (
            round(sum(realised_values) / len(realised_values), wm.METRIC_DECIMALS)
            if realised_values
            else None
        )
        entry["population"] = {
            "scored": len(pairs),
            "probability_absent": absent[definition.metric],
            "outcome_unavailable": gaps[definition.metric],
        }
        out.append(entry)
    return out


# ---------------------------------------------------------------------------
# Quantile coverage
# ---------------------------------------------------------------------------


def quantile_coverage(
    conn: sqlite3.Connection,
    *,
    events: Sequence[int],
    monte_carlo_runs: Mapping[int, int],
) -> dict[str, Any]:
    """CENTRAL_50 / CENTRAL_80 coverage of the realised CORE, at fixture grain."""

    wanted = sorted({int(event) for event in events})
    block: dict[str, Any] = {
        "run_ids_by_event": {},
        "versions": [],
        "quantity": QUANTILE_POLICY["quantity"],
        "metrics": [],
        "distribution_basis": [],
        "population": {"scored": 0, "quantiles_absent": 0, "realised_core_unavailable": 0},
    }
    if not wanted or not monte_carlo_runs:
        return block
    played = _played_fixture_ids(conn, wanted)
    outcomes = _outcome_rows(conn, wanted)
    positions = _positions(conn)
    lows50: list[float] = []
    highs50: list[float] = []
    lows80: list[float] = []
    highs80: list[float] = []
    realised: list[float] = []
    bases: set[str] = set()
    versions: set[str] = set()
    data_gaps = 0
    absent = 0
    for event in wanted:
        run_id = monte_carlo_runs.get(event)
        if run_id is None:
            continue
        block["run_ids_by_event"][int(event)] = int(run_id)
        version = _run_version(conn, run_id)
        if version:
            versions.add(version)
        for record in conn.execute(
            "SELECT player_id, fixture_id, payload_json FROM monte_carlo_distributions "
            "WHERE projection_run_id=? AND event=? ORDER BY player_id, fixture_id",
            (int(run_id), int(event)),
        ):
            fixture_id = int(record["fixture_id"])
            if fixture_id not in played:
                continue
            outcome = outcomes.get((int(record["player_id"]), fixture_id))
            if outcome is None:
                continue
            payload = json.loads(record["payload_json"]) if record["payload_json"] else {}
            if payload.get("distribution_basis") is not None:
                bases.add(str(payload["distribution_basis"]))
            quantiles = {key: payload.get(key) for key in ("q10", "q25", "q75", "q90")}
            if any(value is None for value in quantiles.values()):
                absent += 1
                continue
            # The realised CORE comes from the frozen PE-1 contract rather than a
            # restatement of the scoring rules: two definitions of one quantity is
            # how two numbers that should agree stop agreeing.
            target = calibration.actual_modelled_core(
                outcome, positions.get(int(record["player_id"])), DEFAULT_SCORING_RULES
            )
            if target is None:
                data_gaps += 1
                continue
            lows50.append(float(quantiles["q25"]))
            highs50.append(float(quantiles["q75"]))
            lows80.append(float(quantiles["q10"]))
            highs80.append(float(quantiles["q90"]))
            realised.append(float(target))

    if realised:
        for metric, low, high, low_key, high_key, nominal in (
            ("CENTRAL_50_QUANTILE_COVERAGE", lows50, highs50, "q25", "q75", 0.50),
            ("CENTRAL_80_QUANTILE_COVERAGE", lows80, highs80, "q10", "q90", 0.80),
        ):
            block["metrics"].append(
                {
                    "metric": metric,
                    "interval": [low_key, high_key],
                    "value": wm.central_interval_coverage(low, high, realised).as_dict(),
                    "mean_interval_width": wm.mean_interval_width(low, high).as_dict(),
                    "nominal_coverage": nominal,
                }
            )
    block["versions"] = sorted(versions)
    block["distribution_basis"] = sorted(bases)
    block["population"] = {
        "scored": len(realised),
        "quantiles_absent": absent,
        "realised_core_unavailable": data_gaps,
    }
    return block


# ---------------------------------------------------------------------------
# Component diagnostics
# ---------------------------------------------------------------------------


def resolve_component_run(conn: sqlite3.Connection, xpts_run_id: int) -> int:
    """The minutes run the xPts engine ACTUALLY consumed.

    Read from the persisted dependency closure (``minutes_run_id`` on that run's
    xPts rows), never from the certified bundle's family label.  The bundle's
    ``minutes_v1`` member is a variant (the live bundle declares the joint
    ``minutes_v1.5.2``), and a family label does not say which run produced a given
    xPts row.  Following the closure is the only claim true by construction.
    """

    rows = conn.execute(
        "SELECT DISTINCT minutes_run_id FROM player_fixture_xpts_projections WHERE projection_run_id=?",
        (int(xpts_run_id),),
    ).fetchall()
    resolved = sorted(int(row[0]) for row in rows if row[0] is not None)
    if not resolved:
        raise ScoreboardError(f"xpts run {int(xpts_run_id)} records no minutes dependency")
    if len(resolved) > 1:
        raise ScoreboardError(
            f"xpts run {int(xpts_run_id)} consumed more than one minutes run: {resolved}"
        )
    return resolved[0]


def _has_minutes_dependency(conn: sqlite3.Connection, xpts_run_id: int) -> bool:
    row = conn.execute(
        "SELECT 1 FROM player_fixture_xpts_projections WHERE projection_run_id=? LIMIT 1",
        (int(xpts_run_id),),
    ).fetchone()
    return row is not None


def minutes_component_metrics(
    conn: sqlite3.Connection,
    *,
    events: Sequence[int],
    xpts_runs: Mapping[int, int],
) -> dict[str, Any]:
    """Minutes MAE and bias at the fixture grain, resolved through the closure."""

    block: dict[str, Any] = {
        "run_ids_by_event": {},
        "versions": [],
        "grain": wf.GRAIN_PLAYER_FIXTURE,
        "target": "player_gameweeks.minutes for the same player and fixture",
        "resolved_from": "player_fixture_xpts_projections.minutes_run_id (the consumed dependency closure)",
        "bundle_label_used": False,
        "metrics": [],
        "population": {
            "scored": 0,
            "expected_minutes_absent": 0,
            "realised_minutes_unavailable": 0,
        },
    }
    wanted = sorted({int(event) for event in events})
    if not wanted or not xpts_runs:
        return block
    played = _played_fixture_ids(conn, wanted)
    outcomes = _outcome_rows(conn, wanted)
    predicted: list[float] = []
    realised: list[float] = []
    versions: set[str] = set()
    absent = 0
    gaps = 0
    for event in wanted:
        xpts_run = xpts_runs.get(event)
        if xpts_run is None:
            continue
        minutes_run = resolve_component_run(conn, int(xpts_run))
        block["run_ids_by_event"][int(event)] = int(minutes_run)
        version = _run_version(conn, minutes_run)
        if version:
            versions.add(version)
        for record in conn.execute(
            "SELECT player_id, fixture_id, payload_json FROM frozen_predictions "
            "WHERE projection_run_id=? AND kind=? AND event=? "
            "ORDER BY player_id, fixture_id",
            (int(minutes_run), analytics.MINUTES_V1_KIND, int(event)),
        ):
            if record["fixture_id"] is None:
                continue
            fixture_id = int(record["fixture_id"])
            if fixture_id not in played:
                continue
            outcome = outcomes.get((int(record["player_id"]), fixture_id))
            if outcome is None:
                continue
            payload = json.loads(record["payload_json"]) if record["payload_json"] else {}
            expected = payload.get("expected_minutes")
            if expected is None:
                absent += 1
                continue
            if outcome.get("minutes") is None:
                gaps += 1
                continue
            predicted.append(float(expected))
            realised.append(float(outcome["minutes"]))
    if predicted:
        block["metrics"].append(
            {"metric": "MINUTES_MAE", "value": wm.mean_absolute_error(predicted, realised).as_dict()}
        )
        block["metrics"].append(
            {"metric": "MINUTES_BIAS", "value": wm.mean_bias(predicted, realised).as_dict()}
        )
    block["versions"] = sorted(versions)
    block["population"] = {
        "scored": len(predicted),
        "expected_minutes_absent": absent,
        "realised_minutes_unavailable": gaps,
    }
    return block


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def _shared_comparison(
    model_population: wf.Population, populations: Mapping[str, wf.Population]
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], dict[str, Any]]:
    """The one key set every arm must cover, plus each arm's gate outcome."""

    model_rows = _rows_by_key(model_population)
    scorable: list[tuple[int, int]] = []
    unscorable: list[tuple[int, int]] = []
    for key in model_population.eligible_keys:
        row = model_rows[key]
        if _model_value(row) is None or _realised_points(row) is None:
            unscorable.append(key)
        else:
            scorable.append(key)

    gate: dict[str, Any] = {"arms": {}, "shared_comparison_n": len(scorable)}
    for name, population in populations.items():
        rows = _rows_by_key(population)
        covered = [key for key in scorable if key in rows and _baseline_value(rows[key]) is not None]
        digest = wf.canonical_population_digest(covered, grain=wf.GRAIN_PLAYER_EVENT)
        try:
            wf.assert_same_population(scorable, covered)
        except wf.PopulationMismatch as mismatch:
            gate["arms"][name] = {
                "status": ARM_POPULATION_MISMATCH,
                "shared_comparison_n": len(scorable),
                "population_digest": digest,
                "model_only_n": len(mismatch.model_only),
                "arm_only_n": len(mismatch.baseline_only),
                "model_only_example": [list(key) for key in mismatch.model_only[:5]],
                "arm_only_example": [list(key) for key in mismatch.baseline_only[:5]],
                "detail": (
                    "the arm cannot cover the shared comparison population, so it is reported "
                    "without any metric rather than scored on a smaller sample"
                ),
            }
            continue
        gate["arms"][name] = {
            "status": ARM_OK,
            "shared_comparison_n": len(scorable),
            "population_digest": digest,
            "model_only_n": 0,
            "arm_only_n": 0,
            "model_only_example": [],
            "arm_only_example": [],
            "detail": None,
        }
    return scorable, unscorable, gate


def _population_block(
    model_population: wf.Population,
    scorable: Sequence[tuple[int, int]],
    unscorable: Sequence[tuple[int, int]],
    shared_digest: str,
) -> dict[str, Any]:
    coverage = model_population.coverage()
    return {
        "grain": wf.GRAIN_PLAYER_EVENT,
        "grain_note": TARGET_GRAIN_NOTE,
        "candidates": coverage["candidates"],
        "evaluated": coverage["evaluated"],
        "excluded": coverage["excluded"],
        "excluded_by_status": coverage["by_status"],
        "shared_comparison_n": len(scorable),
        "shared_comparison_digest": shared_digest,
        "shared_comparison_coverage": (
            round(len(scorable) / coverage["candidates"], wm.METRIC_DECIMALS)
            if coverage["candidates"]
            else None
        ),
        "evaluated_without_recorded_points": len(unscorable),
        "evaluated_without_recorded_points_example": [list(key) for key in unscorable[:5]],
        "note": (
            "excluded candidates are listed by status and are never scored as zero; "
            "shared_comparison_n is the population every arm must cover identically"
        ),
    }


def _sample_block(
    target_events: Sequence[int], scorable: Sequence[tuple[int, int]], evaluated_events: set[int]
) -> dict[str, Any]:
    observations = len(scorable)
    events = len(target_events)
    if events >= MIN_TARGET_EVENTS_FOR_DESCRIPTIVE and observations >= MIN_OBSERVATIONS_FOR_DESCRIPTIVE:
        interpretation = SAMPLE_DESCRIPTIVE_ONLY
    else:
        interpretation = SAMPLE_INSUFFICIENT
    return {
        "target_events": events,
        "target_events_with_observations": len(evaluated_events),
        "player_event_observations": observations,
        "sample_interpretation": interpretation,
        "policy": {
            "policy_version": SAMPLE_POLICY_VERSION,
            "min_target_events_for_descriptive_reporting": MIN_TARGET_EVENTS_FOR_DESCRIPTIVE,
            "min_observations_for_descriptive_reporting": MIN_OBSERVATIONS_FOR_DESCRIPTIVE,
            "basis": (
                "a declared disclosure floor, NOT a significance test; no arm is described as "
                "better, superior, proven or calibrated, and no skill score is derived"
            ),
        },
    }


def _status_for(target_events: Sequence[int], evaluated_events: set[int]) -> str:
    if not evaluated_events:
        return STATUS_NOT_YET_EVALUATABLE
    if len(evaluated_events) < len(target_events):
        return STATUS_PARTIALLY_EVALUATABLE
    return STATUS_EVALUATED


def _status_reasons(model_population: wf.Population, target_events: Sequence[int], scorable: Sequence[tuple[int, int]]) -> list[str]:
    if scorable:
        return []
    reasons = [
        f"{count} candidate(s) classified {status}"
        for status, count in sorted(model_population.status_counts.items())
        if status != wf.EVALUATED
    ]
    reasons.append(
        f"no player x event observation across target event(s) {[int(e) for e in target_events]} "
        "is scored, so this artifact carries no metric value"
    )
    return reasons


def build_scoreboard(
    conn: sqlite3.Connection,
    *,
    artifact: Mapping[str, Any],
    events: Sequence[int] | None = None,
    baseline_kinds: Sequence[str] | None = None,
    top_k: int = wm.TOP_K,
) -> dict[str, Any]:
    """Build the structured scoreboard for the certified anchor's target events."""

    kinds = (
        tuple(baseline_kinds) if baseline_kinds is not None else tuple(wf.HEADLINE_POINTS_BASELINE_KINDS)
    )
    anchor = wf.discover_certified_anchor(conn, artifact, events=events)
    target_events = anchor.event_ids

    model_population = wf.build_event_population(
        conn, anchor=anchor, events=target_events, baseline_kind=None
    )
    # ``require_same_population`` is left off here on purpose: PE-2C's gate is then
    # applied centrally in :func:`_shared_comparison`, once per arm, against the one
    # shared comparison key set.  Letting the build-level gate fire first would abort
    # the whole artifact at the first incomplete arm and hide the others; the central
    # gate reports every arm's outcome together, and it is the enforcement point.
    populations = {
        str(kind): wf.build_event_population(
            conn,
            anchor=anchor,
            events=target_events,
            baseline_kind=str(kind),
            require_same_population=False,
        )
        for kind in kinds
    }

    scorable, unscorable, gate = _shared_comparison(model_population, populations)
    shared_digest = wf.canonical_population_digest(scorable, grain=wf.GRAIN_PLAYER_EVENT)
    evaluated_events = {int(event) for event, _player in scorable}
    status = _status_for(target_events, evaluated_events)

    xpts_runs = _event_runs(anchor, target_events, "xpts_v1")
    baseline_runs = _baseline_runs(conn, anchor, target_events)
    arm_runs: dict[str, dict[int, int]] = {str(kind): dict(baseline_runs) for kind in kinds}

    model_rows = _rows_by_key(model_population)
    arms: list[dict[str, Any]] = []
    per_event: list[dict[str, Any]] = []
    model_actual: list[float] = []
    if scorable:
        model_actual = [float(_realised_points(model_rows[key])) for key in scorable]
        arms.append(
            _arm_entry(
                conn,
                name=ARM_MODEL,
                kind="xpts_v1",
                run_id=xpts_runs.get(target_events[0]),
                keys=scorable,
                predicted=[float(_model_value(model_rows[key])) for key in scorable],
                actual=model_actual,
                top_k=top_k,
                population_digest=shared_digest,
            )
        )
        per_event.extend(
            {"arm": ARM_MODEL, **row}
            for row in _per_event_points(
                scorable, [float(_model_value(model_rows[key])) for key in scorable], model_actual
            )
        )
        for kind in kinds:
            name = str(kind)
            arm_gate = gate["arms"][name]
            run_id = sorted(arm_runs[name].values())[0] if arm_runs[name] else None
            if arm_gate["status"] != ARM_OK:
                arms.append(_unavailable_arm(conn, name, run_id, arm_gate))
                continue
            rows = _rows_by_key(populations[name])
            arm_predicted = [float(_baseline_value(rows[key])) for key in scorable]
            arms.append(
                _arm_entry(
                    conn,
                    name=name,
                    kind="baseline",
                    run_id=run_id,
                    keys=scorable,
                    predicted=arm_predicted,
                    actual=model_actual,
                    top_k=top_k,
                    population_digest=shared_digest,
                )
            )
            per_event.extend(
                {"arm": name, **row} for row in _per_event_points(scorable, arm_predicted, model_actual)
            )

    probabilities: list[dict[str, Any]] = []
    quantiles: dict[str, Any] = {}
    components: dict[str, Any] = {}
    if status != STATUS_NOT_YET_EVALUATABLE:
        probabilities = probability_metrics(conn, events=target_events, xpts_runs=xpts_runs)
        monte_carlo_runs = _event_runs(anchor, target_events, "monte_carlo_v1")
        quantiles = quantile_coverage(conn, events=target_events, monte_carlo_runs=monte_carlo_runs)
        components = minutes_component_metrics(conn, events=target_events, xpts_runs=xpts_runs)

    return {
        "on_schema": SCOREBOARD_SCHEMA_VERSION,
        "status": status,
        "status_reasons": _status_reasons(model_population, target_events, scorable),
        "identity": _identity(
            conn, model_population, anchor, target_events, kinds, top_k, xpts_runs, arm_runs
        ),
        "target_events": [int(event) for event in target_events],
        "population": _population_block(model_population, scorable, unscorable, shared_digest),
        "sample": _sample_block(target_events, scorable, evaluated_events),
        "arms": arms,
        "per_event": sorted(per_event, key=lambda row: (str(row["arm"]), int(row["event"]))),
        "probability_metrics": probabilities,
        "quantile_coverage": quantiles,
        "component_diagnostics": components,
        "limitations": _limitations(status, scorable),
    }


def _identity(
    conn: sqlite3.Connection,
    model_population: wf.Population,
    anchor: wf.CertifiedAnchor,
    target_events: Sequence[int],
    kinds: Sequence[str],
    top_k: int,
    xpts_runs: Mapping[int, int],
    arm_runs: Mapping[str, Mapping[int, int]],
) -> dict[str, Any]:
    base = model_population.identity.as_dict()
    monte_carlo_runs = _event_runs(anchor, target_events, "monte_carlo_v1")
    minutes_runs: dict[int, int] = {}
    for event in target_events:
        run_id = xpts_runs.get(int(event))
        if run_id is None or not _has_minutes_dependency(conn, int(run_id)):
            # No projections for this event, so there is no closure to record.  An
            # AMBIGUOUS closure is still raised, by resolve_component_run itself.
            continue
        minutes_runs[int(event)] = resolve_component_run(conn, int(run_id))
    return {
        "walk_forward_identity": base,
        "certification_identity": anchor.certification_identity,
        "planning_cutoff": anchor.planning_cutoff,
        "scoreboard_schema_version": SCOREBOARD_SCHEMA_VERSION,
        "metric_policy_version": wm.METRIC_POLICY_VERSION,
        "sample_policy_version": SAMPLE_POLICY_VERSION,
        "quantile_policy": QUANTILE_POLICY,
        "top_k": int(top_k),
        "bias_convention": BIAS_CONVENTION,
        "median_convention": MEDIAN_CONVENTION,
        "spearman_tie_policy": SPEARMAN_TIE_POLICY,
        "top_k_tie_policy": TOP_K_TIE_POLICY,
        "top_k_aggregation": TOP_K_AGGREGATION,
        "baseline_kinds": [str(kind) for kind in kinds],
        "headline_grain": wf.GRAIN_PLAYER_EVENT,
        "headline_grain_note": TARGET_GRAIN_NOTE,
        "ranking_policy": "NONE; measurements only, no arm is ranked or named best",
        "probability_metric_definitions": [d.as_dict() for d in PROBABILITY_METRICS],
        "probability_population_policy": PROBABILITY_POPULATION_POLICY,
        "per_event_runs": {
            "xpts_v1": {int(k): int(v) for k, v in sorted(xpts_runs.items())},
            "monte_carlo_v1": {int(k): int(v) for k, v in sorted(monte_carlo_runs.items())},
            "minutes_v1": {int(k): int(v) for k, v in sorted(minutes_runs.items())},
            "baselines": {
                str(name): {int(k): int(v) for k, v in sorted(runs.items())}
                for name, runs in sorted(arm_runs.items())
            },
        },
        "note_on_per_event_runs": (
            "recorded per target event because a bundle declares one run set per event; the "
            "family-keyed map inside walk_forward_identity names only the last event's run"
        ),
    }


def _limitations(status: str, scorable: Sequence[tuple[int, int]]) -> list[str]:
    limitations = [
        "Bonus and other unmodelled point components are excluded from the Monte Carlo distribution: "
        "distribution_basis is CORE only, so there is no total-points distribution and no CRPS.",
        "Quantile coverage is reported as QUANTILE COVERAGE, not as a calibrated predictive interval.",
        "The headline grain is player x event; a double gameweek is one summed observation and an "
        "event-grain baseline is never expanded across its fixtures.",
        "No arm is ranked, and no skill score is derived from these measurements.",
        "Models, baselines and outcomes are read from persisted runs; nothing here regenerates a "
        "prediction or writes a database row.",
    ]
    if status == STATUS_NOT_YET_EVALUATABLE or not scorable:
        limitations.insert(
            0,
            "No target event is final and scored, so this artifact carries NO metric value at all. "
            "An unplayed Gameweek is not a zero-score Gameweek.",
        )
    if status == STATUS_PARTIALLY_EVALUATABLE:
        limitations.insert(
            0,
            "Only some target events are final and scored; every figure here covers the events that "
            "were evaluable, and the per-event rows show which.",
        )
    return limitations


# ---------------------------------------------------------------------------
# Canonical bytes, immutability, Markdown
# ---------------------------------------------------------------------------


def canonical_bytes(scoreboard: Mapping[str, Any]) -> bytes:
    """Canonical JSON over which the digest is defined: ``sha256`` of these bytes."""

    return json.dumps(
        scoreboard, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")


def scoreboard_digest(scoreboard: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(canonical_bytes(scoreboard)).hexdigest()


def artifact_path(digest: str, *, directory: Path | str | None = None, suffix: str = ".json") -> Path:
    root = Path(directory) if directory is not None else Path(SCOREBOARD_RELATIVE_DIR)
    return root / f"{digest.replace(':', '-')}{suffix}"


def write_artifact(body: bytes, target: Path, *, kind: str = "scoreboard") -> Path:
    """Write once at an identity path: identical bytes are idempotent, others fail.

    The bytes are staged in a sibling temp file, flushed and fsync-ed, then linked
    into place with ``os.link``, which is atomic and refuses to replace an existing
    file.  So a crash cannot leave a half-written artifact at an identity path, and
    two writers cannot silently disagree about what that identity means.
    """

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.read_bytes() == body:
            return target
        raise ScoreboardIdentityCollision(
            f"{kind} artifact {target.name} already exists with different bytes; "
            "one identity must always describe one content"
        )
    temp = target.with_name(f".{target.name}.tmp-{os.getpid()}-{os.urandom(4).hex()}")
    try:
        with open(temp, "wb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temp, target)
        except FileExistsError:
            if target.read_bytes() == body:
                return target
            raise ScoreboardIdentityCollision(
                f"{kind} artifact {target.name} was concurrently written with different bytes"
            ) from None
        except OSError:
            # A filesystem without hard links: fall back to create-exclusive, which
            # still refuses to overwrite but cannot guarantee completeness if the
            # process dies mid-write.
            descriptor = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(body)
                handle.flush()
                os.fsync(handle.fileno())
        return target
    finally:
        try:
            temp.unlink()
        except OSError:
            pass


def write_scoreboard(
    scoreboard: Mapping[str, Any], *, directory: Path | str | None = None
) -> tuple[str, Path, Path]:
    """Write the JSON and its Markdown rendering at their content-addressed paths."""

    digest = scoreboard_digest(scoreboard)
    root = Path(directory) if directory is not None else Path(SCOREBOARD_RELATIVE_DIR)
    json_path = write_artifact(
        canonical_bytes(scoreboard), artifact_path(digest, directory=root, suffix=".json")
    )
    markdown_path = write_artifact(
        render_markdown(scoreboard).encode("utf-8"),
        artifact_path(digest, directory=root, suffix=".md"),
        kind="markdown",
    )
    return digest, json_path, markdown_path


def _render_metric(metric: Mapping[str, Any]) -> str:
    value = metric.get("value")
    if value is None:
        return f"n/a ({metric.get('status')})"
    return str(value)


def render_markdown(scoreboard: Mapping[str, Any]) -> str:
    """A plain rendering of the structured scoreboard.  The JSON stays authoritative."""

    identity = scoreboard["identity"]
    population = scoreboard["population"]
    sample = scoreboard["sample"]
    lines: list[str] = [
        f"# Walk-forward scoreboard — {scoreboard['status']}",
        "",
        f"Schema: `{scoreboard['on_schema']}`  ",
        f"Metric policy: `{identity['metric_policy_version']}`  ",
        f"Certification identity: `{identity['certification_identity']}`  ",
        f"Planning cutoff: `{identity['planning_cutoff']}`  ",
        f"Target events: {', '.join(str(event) for event in scoreboard['target_events'])}",
        "",
    ]
    if scoreboard["status_reasons"]:
        lines.append("## Why there is no metric")
        lines.append("")
        lines.extend(f"- {reason}" for reason in scoreboard["status_reasons"])
        lines.append("")
    lines.extend(
        [
            "## Population",
            "",
            f"- Candidates: {population['candidates']}",
            f"- Evaluated: {population['evaluated']}",
            f"- Excluded: {population['excluded']}",
            f"- Shared comparison population: {population['shared_comparison_n']}",
            f"- Shared population digest: `{population['shared_comparison_digest']}`",
            f"- Observations (player x event): {sample['player_event_observations']}",
            f"- Sample interpretation: `{sample['sample_interpretation']}`",
            "",
        ]
    )
    if population["excluded_by_status"]:
        lines.extend(["### Excluded by status", "", "| status | count |", "| --- | --- |"])
        lines.extend(
            f"| {status} | {count} |"
            for status, count in sorted(population["excluded_by_status"].items())
            if status != wf.EVALUATED
        )
        lines.append("")
    if scoreboard["arms"]:
        lines.extend(
            [
                "## Arms (measurements, not a ranking)",
                "",
                "| arm | version | N | MAE | RMSE | BIAS | MEDIAN_AE | SPEARMAN | TOP_K_HIT_RATE |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for arm in scoreboard["arms"]:
            lines.append(
                "| {arm} | {version} | {n} | {mae} | {rmse} | {bias} | {median} | {rho} | {topk} |".format(
                    arm=arm["arm"],
                    version=arm["version"],
                    n=arm["N"] if arm["N"] is not None else "n/a",
                    mae=_render_metric(arm["mae"]),
                    rmse=_render_metric(arm["rmse"]),
                    bias=_render_metric(arm["bias"]),
                    median=_render_metric(arm["median_ae"]),
                    rho=_render_metric(arm["spearman"]),
                    topk=_render_metric(arm["top_k_hit_rate"]),
                )
            )
        lines.extend(
            [
                "",
                "## Per-event metrics",
                "",
                "| arm | event | N | MAE | RMSE | BIAS |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in scoreboard["per_event"]:
            lines.append(
                f"| {row['arm']} | {row['event']} | {row['N']} | {_render_metric(row['mae'])} | "
                f"{_render_metric(row['rmse'])} | {_render_metric(row['bias'])} |"
            )
        lines.append("")
    if scoreboard["probability_metrics"]:
        lines.extend(
            [
                "## Probability metrics (Brier)",
                "",
                "| metric | field | n | Brier | base rate | observed rate |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        for entry in scoreboard["probability_metrics"]:
            lines.append(
                f"| {entry['metric']} | {entry['field']} | {entry['population']['scored']} | "
                f"{_render_metric(entry['brier'])} | {_render_metric(entry['brier_reference'])} | "
                f"{entry['observed_rate']} |"
            )
        lines.append("")
    quantiles = scoreboard.get("quantile_coverage") or {}
    if quantiles.get("metrics"):
        lines.extend(
            [
                "## Quantile coverage",
                "",
                "| metric | interval | n | coverage | nominal | mean width |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        for entry in quantiles["metrics"]:
            low, high = entry["interval"]
            lines.append(
                f"| {entry['metric']} | {low}–{high} | {entry['value']['n']} | "
                f"{_render_metric(entry['value'])} | {entry['nominal_coverage']} | "
                f"{_render_metric(entry['mean_interval_width'])} |"
            )
        lines.append("")
        lines.extend(f"Distribution basis: `{basis}`" for basis in quantiles.get("distribution_basis") or [])
        lines.append("")
    components = scoreboard.get("component_diagnostics") or {}
    if components.get("metrics"):
        lines.extend(
            [
                "## Component diagnostics",
                "",
                f"Minutes runs (from the consumed dependency closure): {components['run_ids_by_event']}",
                "",
            ]
        )
        lines.extend(f"- {entry['metric']}: {_render_metric(entry['value'])}" for entry in components["metrics"])
        lines.append("")
    lines.extend(["## Limitations", ""])
    lines.extend(f"- {limitation}" for limitation in scoreboard["limitations"])
    lines.append("")
    return "\n".join(lines)
