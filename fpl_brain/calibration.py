"""Outcome observations and calibration evaluation for frozen predictions.

Outcomes attach ONLY after the target gameweek is officially final
(``events.finished=1`` and ``events.data_checked=1``) — never against
provisional data.  Calibration records are append-only rows computed from the
individual frozen prediction + outcome join, so every aggregate can be
recomputed and split later (by position, override, rotation risk, ...).
"""

from __future__ import annotations

import math
import sqlite3
from typing import Any, Mapping, Sequence

from . import analytics, minutes_model, repositories as repo, team_model
from .scoring_rules import DEFAULT_SCORING_RULES, POSITION_IDS
from .utils import utc_now

_LOG_CLAMP = 1e-12
_FINAL_STATE = "FINAL"

# Player-rate calibration is deliberately rate-level and prospective only: a
# pre-deadline rate belief is compared to the realised event exposure
# (minutes-weighted), never scored against a single fixture without exposure.
RATE_COMPONENT_ACTUAL_FIELD = {
    "xG_per90": "expected_goals",
    "xA_per90": "expected_assists",
}


def observe_event_outcomes(conn: sqlite3.Connection, event: int, *, allow_provisional: bool = False) -> int:
    """Append official outcomes for one gameweek (fixture grain, idempotent)."""

    state, _ = _event_data_state(conn, int(event))
    if state != _FINAL_STATE and not allow_provisional:
        raise ValueError(f"event {event} is {state}; outcomes may only be observed after official finalisation")
    rows = repo.completed_player_fixture_rows(conn, event=int(event))
    observed = 0
    with_idempotent_upsert = """INSERT INTO outcome_observations(
          event, player_id, fixture_id, actual_started, actual_minutes, actual_60_plus,
          actual_zero_minutes, actual_points, actual_xg, actual_xa, actual_defcon, observed_at, source
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(event, player_id, fixture_id) DO UPDATE SET
          actual_started=excluded.actual_started,
          actual_minutes=excluded.actual_minutes,
          actual_60_plus=excluded.actual_60_plus,
          actual_zero_minutes=excluded.actual_zero_minutes,
          actual_points=excluded.actual_points,
          actual_xg=excluded.actual_xg,
          actual_xa=excluded.actual_xa,
          actual_defcon=excluded.actual_defcon,
          observed_at=excluded.observed_at,
          source=excluded.source"""
    for row in rows:
        minutes = row.get("minutes")
        if minutes is None:
            continue
        minutes = int(minutes)
        conn.execute(
            with_idempotent_upsert,
            (
                int(event),
                int(row["player_id"]),
                int(row["fixture_id"]),
                1 if int(row.get("starts") or 0) == 1 else 0,
                minutes,
                1 if minutes >= 60 else 0,
                1 if minutes <= 0 else 0,
                row.get("total_points"),
                row.get("expected_goals"),
                row.get("expected_assists"),
                row.get("defensive_contribution"),
                utc_now(),
                "player_gameweeks_final",
            ),
        )
        observed += 1
    return observed


def _event_data_state(conn: sqlite3.Connection, event: int) -> tuple[str, str]:
    row = conn.execute("SELECT finished, data_checked FROM events WHERE id=?", (int(event),)).fetchone()
    if row is None:
        return "UNKNOWN", "event missing"
    if row["finished"] == 1 and row["data_checked"] == 1:
        return "FINAL", "officially final"
    return "PROVISIONAL_OTHER", "not officially final"


def _outcomes_for_event(conn: sqlite3.Connection, event: int) -> dict[tuple[int, int | None], dict[str, Any]]:
    outcomes: dict[tuple[int, int | None], dict[str, Any]] = {}
    for row in conn.execute("SELECT * FROM outcome_observations WHERE event=?", (int(event),)).fetchall():
        outcomes[(int(row["player_id"]), None if row["fixture_id"] is None else int(row["fixture_id"]))] = dict(row)
    return outcomes


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _log_value(p: float, actual: int) -> float:
    p = min(max(p, _LOG_CLAMP), 1.0 - _LOG_CLAMP)
    return -(actual * math.log(p) + (1 - actual) * math.log(1 - p))


def evaluate_frozen_predictions(
    conn: sqlite3.Connection,
    run_id: int,
    kinds: Sequence[str],
    *,
    payload_keys: Mapping[str, str] | None = None,
) -> dict[str, dict[str, float]]:
    """Aggregate calibration metrics per frozen kind over joined outcomes.

    ``payload_keys`` optionally remaps the minutes payload field names, so the
    same evaluation can score either the coherent marginals (default) or the
    retained raw independent marginals (``*_raw_independent``).
    """

    keys = payload_keys or {}
    key_start = keys.get("p_start", "p_start")
    key_p60 = keys.get("p_60_plus", "p_60_plus")
    key_p0 = keys.get("p_zero", "p_zero")
    key_p159 = keys.get("p_1_59", "p_1_59")
    key_minutes = keys.get("expected_minutes", "expected_minutes")
    aggregated: dict[str, dict[str, list[float]]] = {}
    for prediction in analytics.frozen_predictions(conn, int(run_id)):
        kind = prediction["kind"]
        if kind not in kinds:
            continue
        payload = prediction.get("payload") or {}
        event = int(prediction["event"])
        fixture_id = prediction.get("fixture_id")
        outcome = _outcomes_for_event(conn, event).get((int(prediction["player_id"]), fixture_id))
        if outcome is None:
            continue
        bucket = aggregated.setdefault(kind, {})
        blocks = bucket.setdefault("contrib", {})
        counts = bucket.setdefault("counts", {})

        if kind == analytics.MINUTES_V1_KIND:
            pairs = {
                "brier_start": (payload.get(key_start), outcome.get("actual_started")),
                "log_loss_start": (_log_value(payload.get(key_start) or 0.0, int(outcome.get("actual_started") or 0)), None),
                "brier_p60": (payload.get(key_p60), outcome.get("actual_60_plus")),
                "log_loss_p60": (_log_value(payload.get(key_p60) or 0.0, int(outcome.get("actual_60_plus") or 0)), None),
                "brier_p0": (payload.get(key_p0), outcome.get("actual_zero_minutes")),
            }
            expected = payload.get(key_minutes)
            actual = outcome.get("actual_minutes")
            signed_error = (float(expected) - float(actual)) if (expected is not None and actual is not None) else None
            if signed_error is not None:
                blocks.setdefault("signed_error_sum", []).append(signed_error)
                blocks.setdefault("mae_minutes", []).append(abs(signed_error))
            cls = [
                (_clamp01(payload.get(key_p0) or 0.0), int(outcome.get("actual_zero_minutes") or 0)),
                (_clamp01(payload.get(key_p159) or 0.0), 1 if 0 < int(outcome.get("actual_minutes") or 0) < 60 else 0),
                (_clamp01(payload.get(key_p60) or 0.0), 1 if int(outcome.get("actual_minutes") or 0) >= 60 else 0),
            ]
            mc_brier = sum(float(p - actual_val) ** 2 for p, actual_val in cls)
            actual_class = next(index for index, (_, actual_val) in enumerate(cls) if actual_val == 1)
            mc_log = _log_value(cls[actual_class][0], 1)
            for name, (predicted, actual_value) in pairs.items():
                if predicted is None or actual_value is None:
                    continue
                if name.startswith("brier"):
                    blocks.setdefault(name, []).append((float(predicted) - float(actual_value)) ** 2)
                else:
                    blocks.setdefault(name, []).append(float(predicted))
            blocks.setdefault("multiclass_brier", []).append(mc_brier)
            blocks.setdefault("multiclass_log_loss", []).append(mc_log)
        else:
            expected = payload.get("value")
            actual = outcome.get("actual_minutes")
            if expected is not None and actual is not None:
                blocks.setdefault("mae_minutes", []).append(abs(float(expected) - float(actual)))
                blocks.setdefault("signed_error_sum", []).append(float(expected) - float(actual))
            p_start = payload.get("p_start")
            started = outcome.get("actual_started")
            if p_start is not None and started is not None:
                blocks.setdefault("brier_start", []).append((float(p_start) - float(started)) ** 2)
                blocks.setdefault("log_loss_start", []).append(_log_value(float(p_start), int(started)))

    result: dict[str, dict[str, float]] = {}
    for kind, bucket in aggregated.items():
        metrics: dict[str, float] = {}
        for name, values in bucket["contrib"].items():
            if name == "signed_error_sum":
                if values:
                    metrics["signed_bias_minutes"] = sum(values) / len(values)
                    metrics["sample_count"] = len(values)
            else:
                if values:
                    metrics[name] = sum(values) / len(values)
        result[kind] = metrics
    return result


def record_calibration(
    conn: sqlite3.Connection,
    run_id: int,
    event: int,
    metrics: dict[str, float],
    model_version: str | None = None,
) -> int:
    """Append calibration records (append-only; aggregates recomputable)."""

    stamp = utc_now()
    written = 0
    for name, value in list(metrics.items()):
        if name == "sample_count":
            continue
        conn.execute(
            """INSERT INTO calibration_records(
                 projection_run_id, event, metric_name, sample_count, metric_value,
                 model_version, generated_at
               ) VALUES (?,?,?,?,?,?,?)""",
            (int(run_id), int(event), name, int(metrics.get("sample_count", 0)), float(value), model_version, stamp),
        )
        written += 1
    return written


def calibrate_event(
    conn: sqlite3.Connection,
    minutes_run_id: int,
    baseline_run_id: int | None,
    *,
    event: int | None = None,
) -> dict[str, Any]:
    """Evaluate the frozen minutes predictions and the naive minutes baseline."""

    run_row = analytics.get_projection_run(conn, int(minutes_run_id))
    event_value = int(event if event is not None else (run_row or {}).get("planning_event"))
    minute_metrics = evaluate_minutes_run(conn, int(minutes_run_id), event_value)
    with conn:
        record_calibration(
            conn,
            int(minutes_run_id),
            event_value,
            minute_metrics,
            (run_row or {}).get("model_version"),
        )
    baseline_metrics: dict[str, float] = {}
    if baseline_run_id is not None:
        baseline_metrics = evaluate_minutes_run(conn, int(baseline_run_id), event_value)
        with conn:
            record_calibration(
                conn,
                int(baseline_run_id),
                event_value,
                baseline_metrics,
                analytics.BASELINE_MODEL_VERSION,
            )
    return {
        "minutes_run_id": int(minutes_run_id),
        "baseline_run_id": baseline_run_id,
        "event": event_value,
        "minutes_v1": minute_metrics,
        "naive_baseline": baseline_metrics,
    }


def evaluate_minutes_run(conn: sqlite3.Connection, run_id: int, event: int) -> dict[str, float]:
    run = analytics.get_projection_run(conn, int(run_id))
    family = (run or {}).get("model_family")
    kinds = [analytics.NAIVE_MINUTES_KIND] if family == analytics.BASELINE_MODEL_FAMILY else [analytics.MINUTES_V1_KIND]
    per_kind = evaluate_frozen_predictions(conn, int(run_id), kinds)
    if not per_kind:
        return {}
    return next(iter(per_kind.values()))


# Payload fields for the retained raw independent minutes marginals, so the
# coherent challenger can be scored against the raw v1.1 marginals.
RAW_MINUTES_PAYLOAD_KEYS = {
    "p_start": "p_start_raw_independent",
    "p_60_plus": "p_60_plus_raw_independent",
    "p_zero": "p_zero_raw_independent",
    "p_1_59": "p_1_59_raw_independent",
    "expected_minutes": "expected_minutes_raw_independent",
}


def evaluate_minutes_variant(
    conn: sqlite3.Connection, run_id: int, event: int, *, raw: bool = False
) -> dict[str, float]:
    """Score a minutes run using the coherent or the raw independent marginals."""

    run = analytics.get_projection_run(conn, int(run_id))
    family = (run or {}).get("model_family")
    kinds = [analytics.NAIVE_MINUTES_KIND] if family == analytics.BASELINE_MODEL_FAMILY else [analytics.MINUTES_V1_KIND]
    per_kind = evaluate_frozen_predictions(
        conn, int(run_id), kinds, payload_keys=RAW_MINUTES_PAYLOAD_KEYS if raw else None
    )
    if not per_kind:
        return {}
    return next(iter(per_kind.values()))


def minutes_variant_comparison(
    conn: sqlite3.Connection, runs: dict[str, int], event: int
) -> dict[str, dict[str, float]]:
    """Score several minutes runs side by side.

    ``runs`` maps a label to a run id; a label ending in ``:raw`` is scored
    against the raw independent marginals, otherwise the coherent marginals.
    """

    comparison: dict[str, dict[str, float]] = {}
    for label, run_id in runs.items():
        comparison[label] = evaluate_minutes_variant(
            conn, int(run_id), int(event), raw=label.endswith(":raw")
        )
    return comparison


def baseline_comparison_summary(
    conn: sqlite3.Connection,
    event: int,
) -> dict[str, Any]:
    """Latest complete minutes_v1 run vs latest complete baseline run."""

    v1 = conn.execute(
        """SELECT id FROM projection_runs
            WHERE model_family='minutes_v1' AND status='complete' AND planning_event=?
            ORDER BY id DESC LIMIT 1""",
        (int(event),),
    ).fetchone()
    baseline = conn.execute(
        """SELECT id FROM projection_runs
            WHERE model_family='baseline' AND status='complete' AND planning_event=?
            ORDER BY id DESC LIMIT 1""",
        (int(event),),
    ).fetchone()
    if v1 is None or baseline is None:
        return {"available": False}
    summary = calibrate_event(conn, int(v1["id"]), int(baseline["id"]), event=int(event))
    return {"available": True, **summary}


# ---------------------------------------------------------------------------
# Team Attack/Defence Model calibration (Phase 2).
# ---------------------------------------------------------------------------


def _poisson_nll(actual_goals: int, lam: float) -> float:
    """-log Poisson(k; lambda) for a single team-fixture."""

    return lam - int(actual_goals) * math.log(lam) + math.lgamma(int(actual_goals) + 1)


def evaluate_team_run(conn: sqlite3.Connection, run_id: int, event: int) -> dict[str, float]:
    """Poisson NLL, lambda bias/MAE, clean-sheet and P(2+) Brier for a team run.

    Evaluated only against a finalised event (callers gate on FINAL).  Actual
    goals come from the official fixture score; actual team xG is the canonical
    side-sum, so the xG comparison is descriptive and never required.
    """

    rows = analytics.team_fixture_projections(conn, int(run_id))
    nll: list[float] = []
    bias: list[float] = []
    mae: list[float] = []
    cs_brier: list[float] = []
    p2_brier: list[float] = []
    xg_mae: list[float] = []
    for record in rows:
        if int(record["event"]) != int(event):
            continue
        payload = record.get("payload") or {}
        lam = payload.get("expected_goals_for")
        if lam is None or not math.isfinite(float(lam)) or float(lam) <= 0:
            continue
        fixture = conn.execute(
            "SELECT team_h, team_a, team_h_score, team_a_score FROM fixtures WHERE id=?",
            (int(record["fixture_id"]),),
        ).fetchone()
        if fixture is None:
            continue
        team_id = int(record["team_id"])
        if team_id == int(fixture["team_h"]):
            goals_for = fixture["team_h_score"]
            goals_against = fixture["team_a_score"]
            side = "home"
        elif team_id == int(fixture["team_a"]):
            goals_for = fixture["team_a_score"]
            goals_against = fixture["team_h_score"]
            side = "away"
        else:
            continue
        if goals_for is None or goals_against is None:
            continue
        actual_goals = int(goals_for)
        lam = float(lam)
        nll.append(_poisson_nll(actual_goals, lam))
        bias.append(lam - actual_goals)
        mae.append(abs(lam - actual_goals))
        actual_cs = 1.0 if int(goals_against) == 0 else 0.0
        cs_prob = payload.get("p_clean_sheet")
        if cs_prob is not None:
            cs_brier.append((float(cs_prob) - actual_cs) ** 2)
        p2 = payload.get("p_goals_2_plus")
        if p2 is not None:
            p2_brier.append((float(p2) - (1.0 if actual_goals >= 2 else 0.0)) ** 2)
        # Realised outcome, not a historical model input: this scores a past
        # prediction against what the match actually produced, so it must not be
        # windowed to an earlier cutoff.
        sides = team_model.realised_fixture_side_xg(conn, int(record["fixture_id"]))
        actual_xg = sides["home_xg"] if side == "home" else sides["away_xg"]
        if actual_xg is not None:
            xg_mae.append(abs(lam - float(actual_xg)))

    metrics: dict[str, float] = {}
    if nll:
        metrics["poisson_nll_goals"] = sum(nll) / len(nll)
        metrics["lambda_bias_goals"] = sum(bias) / len(bias)
        metrics["lambda_mae_goals"] = sum(mae) / len(mae)
        metrics["sample_count"] = float(len(nll))
    if cs_brier:
        metrics["clean_sheet_brier"] = sum(cs_brier) / len(cs_brier)
    if p2_brier:
        metrics["p_goals_2plus_brier"] = sum(p2_brier) / len(p2_brier)
    if xg_mae:
        metrics["lambda_mae_team_xg_descriptive"] = sum(xg_mae) / len(xg_mae)
    return metrics


def calibrate_team_event(
    conn: sqlite3.Connection,
    team_run_id: int,
    baseline_run_id: int | None,
    *,
    event: int | None = None,
) -> dict[str, Any]:
    """Evaluate the team model against a finalised event and record metrics."""

    run_row = analytics.get_projection_run(conn, int(team_run_id))
    event_value = int(event if event is not None else (run_row or {}).get("planning_event"))
    state, basis = _event_data_state(conn, event_value)
    if state != _FINAL_STATE:
        raise ValueError(f"event {event_value} is {state} ({basis}); team calibration requires official finalisation")
    team_metrics = evaluate_team_run(conn, int(team_run_id), event_value)
    with conn:
        record_calibration(conn, int(team_run_id), event_value, team_metrics, (run_row or {}).get("model_version"))
    baseline_metrics: dict[str, float] = {}
    if baseline_run_id is not None:
        baseline_metrics = evaluate_team_run(conn, int(baseline_run_id), event_value)
        baseline_run = analytics.get_projection_run(conn, int(baseline_run_id))
        with conn:
            record_calibration(
                conn, int(baseline_run_id), event_value, baseline_metrics, (baseline_run or {}).get("model_version")
            )
    return {
        "event": event_value,
        "team_run_id": int(team_run_id),
        "baseline_run_id": baseline_run_id,
        "team_strength_v1": team_metrics,
        "team_naive_baseline": baseline_metrics,
    }


# ---------------------------------------------------------------------------
# Player Attacking-Rate calibration (Phase 3) — rate-level, prospective only.
# ---------------------------------------------------------------------------


def evaluate_rate_run(conn: sqlite3.Connection, run_id: int, event: int) -> dict[str, float]:
    """Exposure-weighted, rate-level descriptive error for a player-rate run.

    The frozen posterior rate is compared to the realised event xG/xA per 90
    over the player's actual minutes.  Weighting by realised minutes keeps the
    comparison exposure-aware; it is deliberately NOT a fixture-level
    predictive score (a single fixture cannot identify a per-90 rate).  Kept
    descriptive until fixture-level expectation exists.
    """

    rows = analytics.player_rate_projections(conn, int(run_id))
    per_component: dict[str, dict[str, list[float]]] = {}
    for record in rows:
        if int(record["event"]) != int(event):
            continue
        payload = record.get("payload") or {}
        predicted = payload.get("posterior_mean")
        component = str(record["component"])
        field_name = RATE_COMPONENT_ACTUAL_FIELD.get(component)
        if predicted is None or field_name is None:
            continue
        actuals = conn.execute(
            """SELECT pg.minutes AS minutes, pg.%s AS value
                 FROM player_gameweeks pg JOIN fixtures f ON f.id=pg.fixture_id
                WHERE pg.player_id=? AND pg.event=? AND f.finished=1 AND pg.minutes>0"""
            % field_name,
            (int(record["player_id"]), int(event)),
        ).fetchall()
        minutes = sum(float(row["minutes"]) for row in actuals if row["minutes"] is not None)
        total = sum(float(row["value"]) for row in actuals if row["value"] is not None)
        if minutes <= 0:
            continue
        observed_rate = total / minutes * 90.0
        bucket = per_component.setdefault(component, {"w_abs": [], "w_signed": [], "weights": []})
        bucket["w_abs"].append(minutes * abs(float(predicted) - observed_rate))
        bucket["w_signed"].append(minutes * (float(predicted) - observed_rate))
        bucket["weights"].append(minutes)

    metrics: dict[str, float] = {}
    total_weight = 0.0
    for component, bucket in per_component.items():
        weight = sum(bucket["weights"])
        if weight <= 0:
            continue
        total_weight += weight
        key = component.replace("/", "_")
        metrics[f"{key}_exposure_weighted_mae"] = sum(bucket["w_abs"]) / weight
        metrics[f"{key}_exposure_weighted_bias"] = sum(bucket["w_signed"]) / weight
    if total_weight > 0:
        metrics["sample_minutes"] = total_weight
    return metrics


def rate_calibration_note() -> dict[str, Any]:
    """Explicit, honest statement of the player-rate calibration limits."""

    return {
        "stage": "rate_level_exposure_weighted",
        "fixture_level_scoring": False,
        "note": (
            "Player-rate calibration is prospective and exposure-weighted only. A per-90 rate cannot "
            "be scored against a single fixture without exposure, so full fixture-level rate "
            "calibration is deferred until the xPts layer supplies expected minutes and fixture scaling."
        ),
    }


# ---------------------------------------------------------------------------
# Monte Carlo calibration (Phase 5).
# ---------------------------------------------------------------------------


# Canonical threshold-forecast payload names (never the retired p_blank /
# p_return / p_haul_10_plus aliases, except through the legacy adapter below).
CANONICAL_THRESHOLD_FORECASTS = (
    "p_score_le_2", "p_score_5_plus", "p_score_10_plus", "p_score_15_plus",
)
LEGACY_THRESHOLD_FORECAST_ALIASES = {
    "p_score_le_2": "p_blank",
    "p_score_5_plus": "p_return",
    "p_score_10_plus": "p_haul_10_plus",
    "p_score_15_plus": "p_haul_15_plus",
}
# Runs at or below this Monte Carlo version stored the retired alias names.
LEGACY_MC_VERSIONS = ("mc_v1.0.0", "mc_v1.1.0")


def canonical_forecast(payload: Mapping[str, Any], key: str, *, legacy_adapter: bool = False) -> Any:
    """Read a canonical threshold forecast, never coercing a missing value to 0.

    Historical runs stored the retired ``p_blank`` / ``p_return`` /
    ``p_haul_10_plus`` names; those are read only through the explicit legacy
    adapter, and a genuinely absent forecast returns ``None`` (DATA_GAP).
    """

    value = payload.get(key)
    if value is None and legacy_adapter:
        alias = LEGACY_THRESHOLD_FORECAST_ALIASES.get(key)
        if alias:
            value = payload.get(alias)
    return value


def actual_modelled_core(actual: Mapping[str, Any], position: str | None, rules) -> float | None:
    """Reconstruct the realised CORE from proven official fixture facts.

    Uses exactly the components the simulator models (appearance, goals,
    assists, clean sheet, goals conceded, DefCon, saves, yellow) and excludes
    bonus, red cards, own goals and penalties.  Returns ``None`` when a required
    field is missing, so the caller records DATA_GAP instead of a wrong number.
    """

    if actual is None:
        return None
    minutes = actual.get("minutes")
    if minutes is None:
        return None
    required = ("goals_scored", "assists", "clean_sheets", "goals_conceded", "saves", "yellow_cards")
    if any(actual.get(field) is None for field in required):
        return None
    position = str(position or "")
    minutes = int(minutes)
    points = 0.0
    if minutes >= rules.clean_sheet_minutes_required:
        points += rules.appearance_long_points
    elif minutes > 0:
        points += rules.appearance_short_points
    points += int(actual.get("goals_scored") or 0) * rules.goal_points_for(position)
    points += int(actual.get("assists") or 0) * rules.assist_points
    conceded = int(actual.get("goals_conceded") or 0)
    if (
        minutes >= rules.clean_sheet_minutes_required
        and conceded == 0
        and rules.clean_sheet_points_for(position) > 0
    ):
        points += rules.clean_sheet_points_for(position)
    if position in rules.goals_conceded_positions and conceded > 0:
        points -= math.floor(conceded / rules.goals_conceded_per_deduction) * abs(
            rules.goals_conceded_points_for(position)
        )
    threshold = rules.defcon_threshold_for(position)
    if position in rules.defcon_positions and threshold is not None:
        contribution = actual.get("defensive_contribution")
        if contribution is None:
            return None
        if float(contribution) >= threshold:
            points += rules.defcon_points
    if position == "GKP":
        points += math.floor(int(actual.get("saves") or 0) / rules.saves_per_point)
    points += int(actual.get("yellow_cards") or 0) * rules.yellow_card_points
    return points


def evaluate_monte_carlo_run(conn: sqlite3.Connection, run_id: int, event: int) -> dict[str, float]:
    """Distribution-level calibration for a finalised event.

    CORE intervals and bias are compared to ``ACTUAL_MODELLED_CORE`` (the same
    components the simulator models), never to raw ``total_points``, which also
    contains bonus and other unmodelled components.  ``TOTAL_PROXY`` has no
    genuine stochastic distribution because bonus variance is not modelled, so
    no TOTAL coverage is published.  A missing required forecast is reported as
    a DATA_GAP, never silently predicted as zero.
    """

    # PE-9 gap 5: an ABSENT run is a refusal, never an empty legacy run.  Reading it
    # as ``{{}}`` parsed the model version as the empty string, which is not a legacy
    # version anyone declared, and silently selected the legacy adapter for a run
    # whose metadata was never read.
    from .monte_carlo import InputRunAbsent

    run_row = analytics.get_projection_run(conn, int(run_id))
    if run_row is None:
        raise InputRunAbsent(
            f"INPUT_RUN_ABSENT: no projection run {int(run_id)} for the Monte Carlo calibration of "
            f"event {int(event)}; a missing run is never defaulted to an empty legacy run"
        )
    model_version = str(run_row.get("model_version") or "")
    legacy_adapter = model_version in LEGACY_MC_VERSIONS or model_version.startswith(
        tuple(f"{prefix}" for prefix in LEGACY_MC_VERSIONS)
    )
    records = analytics.monte_carlo_distributions(conn, int(run_id))
    bias: list[float] = []
    sq: list[float] = []
    inside50: list[float] = []
    inside80: list[float] = []
    brier_goal: list[float] = []
    brier_assist: list[float] = []
    brier_cs: list[float] = []
    brier_defcon: list[float] = []
    # Threshold Brier scores against the reconstructed ACTUAL_MODELLED_CORE.
    brier_threshold: dict[str, list[float]] = {
        "brier_p_score_le_2": [], "brier_p_score_5_plus": [],
        "brier_p_score_10_plus": [], "brier_p_score_15_plus": [],
    }
    threshold_outcomes = (
        ("p_score_le_2", "le", 2, "brier_p_score_le_2"),
        ("p_score_5_plus", "ge", 5, "brier_p_score_5_plus"),
        ("p_score_10_plus", "ge", 10, "brier_p_score_10_plus"),
        ("p_score_15_plus", "ge", 15, "brier_p_score_15_plus"),
    )
    forecast_gaps: dict[str, int] = {key: 0 for key in CANONICAL_THRESHOLD_FORECASTS}
    core_sample = 0
    positions = {
        int(row["id"]): POSITION_IDS.get(int(row["element_type"])) if row["element_type"] is not None else None
        for row in conn.execute("SELECT id, element_type FROM players")
    }
    for record in records:
        if int(record["event"]) != int(event):
            continue
        payload = record.get("payload") or {}
        actual = conn.execute(
            """SELECT total_points, minutes, goals_scored, assists, clean_sheets, goals_conceded,
                      saves, yellow_cards, defensive_contribution
                 FROM player_gameweeks WHERE player_id=? AND fixture_id=?""",
            (int(record["player_id"]), int(record["fixture_id"])),
        ).fetchone()
        if actual is None:
            continue
        actual = dict(actual)
        position = positions.get(int(record["player_id"]))
        reconstructed = actual_modelled_core(actual, position, DEFAULT_SCORING_RULES)

        # Canonical threshold forecasts: read directly, or through the legacy
        # adapter for historical runs.  Never coerce a missing value to zero.
        for key in CANONICAL_THRESHOLD_FORECASTS:
            if canonical_forecast(payload, key, legacy_adapter=legacy_adapter) is None:
                forecast_gaps[key] += 1

        if reconstructed is None:
            continue  # DATA_GAP: no CORE coverage for this row
        core_sample += 1
        predicted = payload.get("mean_core")
        if predicted is None or not math.isfinite(float(predicted)):
            continue
        predicted = float(predicted)
        error = predicted - reconstructed
        bias.append(error)
        sq.append(error * error)
        if payload.get("q25") is not None and payload.get("q75") is not None:
            inside50.append(1.0 if float(payload["q25"]) <= reconstructed <= float(payload["q75"]) else 0.0)
        if payload.get("q10") is not None and payload.get("q90") is not None:
            inside80.append(1.0 if float(payload["q10"]) <= reconstructed <= float(payload["q90"]) else 0.0)

        # ``p_clean_sheet_eligible`` is the simulator's P(the player EARNED clean-sheet
        # points): its flag is set in the same block that awards them and requires
        # ``clean_sheet_points_for(position) > 0``.  The realised target must therefore
        # be the same scoring event, NOT the raw ``clean_sheets`` column, which the
        # official feed sets for any player on for 60+ minutes of a non-conceding
        # match -- forwards included, who score nothing for it.  Both clauses come
        # from the one versioned rule object so producer and target cannot drift.
        cs_scoring_event = (
            1.0
            if DEFAULT_SCORING_RULES.earns_clean_sheet_points(
                str(position or ""), int(actual["minutes"] or 0), int(actual["goals_conceded"] or 0)
            )
            else 0.0
        )
        for key, actual_value, bucket in (
            ("p_goal", 1.0 if int(actual["goals_scored"] or 0) > 0 else 0.0, brier_goal),
            ("p_assist", 1.0 if int(actual["assists"] or 0) > 0 else 0.0, brier_assist),
            ("p_clean_sheet_eligible", cs_scoring_event, brier_cs),
        ):
            value = payload.get(key)
            if value is not None and math.isfinite(float(value)):
                bucket.append((float(value) - actual_value) ** 2)
        threshold = DEFAULT_SCORING_RULES.defcon_threshold_for(str(position or ""))
        p_defcon = payload.get("p_defcon_hit")
        contribution = actual["defensive_contribution"]
        if (
            p_defcon is not None
            and math.isfinite(float(p_defcon))
            and threshold is not None
            and contribution is not None
        ):
            hit = 1.0 if float(contribution) >= threshold else 0.0
            brier_defcon.append((float(p_defcon) - hit) ** 2)

        for key, comparator, cut, bucket_key in threshold_outcomes:
            value = canonical_forecast(payload, key, legacy_adapter=legacy_adapter)
            if value is None or not math.isfinite(float(value)):
                continue
            hit = (reconstructed <= cut) if comparator == "le" else (reconstructed >= cut)
            brier_threshold[bucket_key].append((float(value) - (1.0 if hit else 0.0)) ** 2)

    metrics: dict[str, float] = {}
    if bias:
        metrics["mc_core_mean_bias"] = sum(bias) / len(bias)
        metrics["mc_core_rmse"] = math.sqrt(sum(sq) / len(sq))
        metrics["sample_count"] = float(len(bias))
    if inside50:
        metrics["core_coverage_50"] = sum(inside50) / len(inside50)
    if inside80:
        metrics["core_coverage_80"] = sum(inside80) / len(inside80)
    metrics["actual_modelled_core_sample_count"] = float(core_sample)
    for name, values in (
        ("brier_goal_any", brier_goal), ("brier_assist_any", brier_assist),
        ("brier_clean_sheet", brier_cs), ("brier_defcon", brier_defcon),
    ):
        if values:
            metrics[name] = sum(values) / len(values)
    for name, values in brier_threshold.items():
        if values:
            metrics[name] = sum(values) / len(values)
            metrics[f"{name}_sample_count"] = float(len(values))
    for key, count in forecast_gaps.items():
        if count:
            metrics[f"forecast_data_gap_{key}"] = float(count)
    if legacy_adapter:
        metrics["legacy_threshold_adapter_used"] = 1.0
    metrics["no_total_proxy_coverage"] = 1.0
    return metrics


def calibrate_monte_carlo_event(conn: sqlite3.Connection, run_id: int, *, event: int | None = None) -> dict[str, Any]:
    """Evaluate a Monte Carlo run against a finalised event."""

    run_row = analytics.get_projection_run(conn, int(run_id))
    event_value = int(event if event is not None else (run_row or {}).get("planning_event"))
    state, basis = _event_data_state(conn, event_value)
    if state != _FINAL_STATE:
        raise ValueError(f"event {event_value} is {state} ({basis}); Monte Carlo calibration requires finalisation")
    metrics = evaluate_monte_carlo_run(conn, int(run_id), event_value)
    with conn:
        record_calibration(conn, int(run_id), event_value, metrics, (run_row or {}).get("model_version"))
    return {"event": event_value, "monte_carlo_run_id": int(run_id), "mc_v1": metrics}


# ---------------------------------------------------------------------------
# xPts calibration (Phase 4).
# ---------------------------------------------------------------------------


def evaluate_xpts_run(conn: sqlite3.Connection, run_id: int, event: int) -> dict[str, float]:
    """Point-level bias/MAE/RMSE plus component hooks against a finalised event.

    Evaluated only after the event is officially FINAL.  No hyperparameters are
    tuned on these metrics; they are reported, not optimised.
    """

    rows = analytics.xpts_projections(conn, int(run_id))
    bias: list[float] = []
    mae: list[float] = []
    sq: list[float] = []
    ep_abs: list[float] = []
    ep_bias: list[float] = []
    comp: dict[str, list[float]] = {}

    def add(name: str, value: float) -> None:
        comp.setdefault(name, []).append(value)

    for record in rows:
        if int(record["event"]) != int(event):
            continue
        payload = record.get("payload") or {}
        actual = conn.execute(
            """SELECT total_points, expected_goals, expected_assists, minutes, clean_sheets,
                      saves, bonus, defensive_contribution
                 FROM player_gameweeks WHERE player_id=? AND fixture_id=?""",
            (int(record["player_id"]), int(record["fixture_id"])),
        ).fetchone()
        if actual is None or actual["total_points"] is None:
            continue
        predicted = payload.get("total_xpts")
        if predicted is None or not math.isfinite(float(predicted)):
            continue
        actual_points = float(actual["total_points"])
        error = float(predicted) - actual_points
        bias.append(error)
        mae.append(abs(error))
        sq.append(error * error)
        ep = payload.get("official_ep_next")
        if ep is not None:
            ep_abs.append(abs(float(ep) - actual_points))
            ep_bias.append(float(ep) - actual_points)

        # Component-level future calibration hooks.
        if payload.get("adjusted_expected_xg") is not None and actual["expected_goals"] is not None:
            add("xg_mae", abs(float(payload["adjusted_expected_xg"]) - float(actual["expected_goals"])))
        if payload.get("expected_xa") is not None and actual["expected_assists"] is not None:
            add("xa_mae", abs(float(payload["expected_xa"]) - float(actual["expected_assists"])))
        if payload.get("clean_sheet_probability") is not None:
            # DELIBERATELY the raw physical flag, and deliberately NOT the
            # position-conditioned scoring event used for ``p_clean_sheet_eligible``
            # above.  These are two different persisted probabilities with two
            # different meanings: ``xpts._clean_sheet_probability`` models
            # "60+ minutes of exposure with no opponent goal during it" and is
            # stored for every position (a forward's value is positive; only the
            # expected-POINTS layer multiplies it by ``clean_sheet_points_for``),
            # whereas the simulator's ``p_clean_sheet_eligible`` already excludes
            # positions that score nothing for a clean sheet.  Matching each target
            # to its own producer is what keeps them comparable; harmonising the two
            # targets would break one of them.
            add("clean_sheet_brier", (float(payload["clean_sheet_probability"]) - (1.0 if actual["clean_sheets"] else 0.0)) ** 2)
        if payload.get("defcon_p_hit") is not None and actual["defensive_contribution"] is not None:
            threshold = (DEFAULT_SCORING_RULES.defcon_threshold_for(str(record["position"])) or 10**9)
            hit = 1.0 if float(actual["defensive_contribution"]) >= threshold else 0.0
            add("defcon_brier", (float(payload["defcon_p_hit"]) - hit) ** 2)
        save_model = payload.get("save_model") or {}
        if save_model.get("lambda_saves") is not None and actual["saves"] is not None:
            add("saves_mae", abs(float(save_model["lambda_saves"]) - float(actual["saves"])))
        if payload.get("bonus_xpts") is not None and actual["bonus"] is not None:
            add("bonus_mae", abs(float(payload["bonus_xpts"]) - float(actual["bonus"])))

    metrics: dict[str, float] = {}
    if mae:
        metrics["xpts_bias"] = sum(bias) / len(bias)
        metrics["xpts_mae"] = sum(mae) / len(mae)
        metrics["xpts_rmse"] = math.sqrt(sum(sq) / len(sq))
        metrics["sample_count"] = float(len(mae))
    if ep_abs:
        metrics["ep_next_mae"] = sum(ep_abs) / len(ep_abs)
        metrics["ep_next_bias"] = sum(ep_bias) / len(ep_bias)
    for name, values in comp.items():
        if values:
            metrics[name] = sum(values) / len(values)
    return metrics


def calibrate_xpts_event(
    conn: sqlite3.Connection,
    xpts_run_id: int,
    *,
    event: int | None = None,
    baseline_run_id: int | None = None,
) -> dict[str, Any]:
    """Evaluate an xPts run against a finalised event and record the metrics."""

    run_row = analytics.get_projection_run(conn, int(xpts_run_id))
    event_value = int(event if event is not None else (run_row or {}).get("planning_event"))
    state, basis = _event_data_state(conn, event_value)
    if state != _FINAL_STATE:
        raise ValueError(f"event {event_value} is {state} ({basis}); xPts calibration requires official finalisation")
    metrics = evaluate_xpts_run(conn, int(xpts_run_id), event_value)
    with conn:
        record_calibration(conn, int(xpts_run_id), event_value, metrics, (run_row or {}).get("model_version"))
    return {
        "event": event_value,
        "xpts_run_id": int(xpts_run_id),
        "baseline_run_id": baseline_run_id,
        "xpts_v1": metrics,
    }
