"""Deterministic expected FPL points v1 (Analytics Phase 4).

Turns the three accepted inputs into one **analytic-mean** expected-points
projection per player-fixture:

    PlayerMinutesProjection + TeamFixtureProjection + PlayerRateProjection

No Monte Carlo: only expected means and closed-form Poisson expectations.
No transfer, captaincy, chip, or XI recommendation.

Anti-double-counting
--------------------
A player's historical/current xG already reflects the team environment, so the
fixture scaling applied to the player's attacking rate is an
**opponent/venue multiplier**, not the full team attacking rating:

    lambda_fixture   = exp(mu + home_term + attack_team + defence_opponent)
    lambda_reference = exp(mu + attack_team)
    fixture_multiplier = lambda_fixture / lambda_reference
                       = exp(home_term + defence_opponent)

Because penalties remain embedded in xG/xA, the multiplier also approximately
scales embedded penalty threat; this is documented and no correction is faked.

Poisson expectations are analytic
---------------------------------
* goals-conceded deduction uses ``E[floor(N/2)]`` in closed form, never
  ``floor(E[N]/2)``;
* save points use ``E[floor(S/3)]`` as a truncated analytic series, never
  ``E[S]/3``;
* DefCon uses ``P(Poisson >= threshold)`` scaled by expected exposure, with no
  hard ``P(60+)`` multiplication.
"""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass, fields
from typing import Any, Iterable, Mapping

from . import analytics, repositories as repo
from .scoring_rules import POSITION_IDS, SCORING_RULES_VERSION, ScoringRules, DEFAULT_SCORING_RULES
from .utils import parse_utc, utc_now

XPTS_MODEL_VERSION = "xpts_v1.4.1"

COMPONENT_XG = "xG_per90"
COMPONENT_XA = "xA_per90"

# Branch-placement of each component in the two-tier output.
CORE_COMPONENTS = (
    "appearance_xpts",
    "goal_xpts",
    "assist_xpts",
    "clean_sheet_xpts",
    "goals_conceded_xpts",
    "defcon_xpts",
    "save_xpts",
    "yellow_card_xpts",
)
SOFT_COMPONENTS = ("bonus_xpts",)


@dataclass(frozen=True)
class XPtsConfig:
    """Every tunable in one versioned structure; no magic numbers in code."""

    # FPL assist mapping: xA is not exactly an FPL assist.  Initial assumption,
    # explicitly uncalibrated; 1.0 until data supports another value.
    fpl_assist_mapping_coefficient: float = 1.0
    assist_mapping_calibrated: bool = False

    # Team xG sanity: player raw xG may not exceed this fraction of team lambda.
    # v1.2.0 sets this to 1.00 so ANY material excess above lambda is scaled
    # downward -- required so the Monte Carlo scorer probabilities can never
    # sum above the team's own goal expectation.
    team_xg_cap_fraction: float = 1.00
    team_xg_excess_warn_fraction: float = 1.02

    # Clean-sheet exposure approximation: on-pitch minutes for a long cameo.
    cameo_60_exposure_minutes: float = 70.0

    # Component prior strengths (minutes-equivalent).
    defcon_prior_strength_minutes: float = 600.0
    saves_prior_strength_minutes: float = 540.0
    yellow_prior_strength_minutes: float = 900.0
    bonus_prior_strength_minutes: float = 900.0
    prev_season_min_prior_minutes: float = 450.0

    # Goalkeeper save pressure multiplier bounds.
    save_pressure_multiplier_min: float = 0.6
    save_pressure_multiplier_max: float = 1.6

    # Team-minutes mass coherence tolerances (diagnostics only).
    # Measured: Minutes v1 sums to ~1,200-1,400 expected minutes and ~13.5
    # expected starters per side (vs 990 / 11) because each player is predicted
    # independently.  That is a structural property of the accepted, frozen
    # Minutes v1 (which this task must not normalise), not data corruption, so
    # it WARNs at >10% and only FAILs at >50% (a side more than half a
    # regulation match away from 990 minutes, or +/-7 starters -- i.e.
    # structurally broken rather than merely non-conserving).
    team_minutes_mass_warn_fraction: float = 0.10
    team_minutes_mass_fail_fraction: float = 0.50
    team_starters_warn_abs: float = 1.5
    team_starters_fail_abs: float = 7.0

    # Strict (decision-bearing) tolerances applied when the minutes input is a
    # team-coherent challenger: a material violation of the 11-starter or
    # 990-minute identity then FAILs rather than WARNs.
    coherent_start_tolerance: float = 1e-3
    coherent_minutes_tolerance: float = 0.05

    # Team-model sensitivity threshold on total xPts.
    team_model_sensitive_threshold: float = 0.5

    def config_hash(self) -> str:
        values = {item.name: getattr(self, item.name) for item in fields(self)}
        return analytics.canonical_hash({"model": XPTS_MODEL_VERSION, **values})


# ---------------------------------------------------------------------------
# Analytic Poisson helpers.
# ---------------------------------------------------------------------------


def poisson_pmf(k: int, lam: float) -> float:
    return math.exp(-lam) * lam**k / math.factorial(k)


def poisson_cdf(k: int, lam: float) -> float:
    if k < 0:
        return 0.0
    return sum(poisson_pmf(i, lam) for i in range(0, int(k) + 1))


def poisson_sf(lam: float, k: int) -> float:
    """P(N > k) for N ~ Poisson(lam)."""

    return max(0.0, 1.0 - poisson_cdf(k, lam))


def poisson_tail_probability(lam: float, threshold: int) -> float:
    """P(N >= threshold) = P(N > threshold-1)."""

    if threshold <= 0:
        return 1.0
    return poisson_sf(lam, int(threshold) - 1)


def expected_floor_half(lam: float) -> float:
    """E[floor(N/2)] for N ~ Poisson(lam), closed form.

    floor(N/2) = (N - (N mod 2))/2 and P(N odd) = (1 - e^{-2 lam})/2, so
    E[floor(N/2)] = (lam - (1 - e^{-2 lam})/2) / 2.  This is NOT floor(E[N]/2).
    """

    if lam <= 0.0:
        return 0.0
    return (lam - (1.0 - math.exp(-2.0 * lam)) / 2.0) / 2.0


def expected_floor_poisson_ratio(lam: float, divisor: int, tolerance: float = 1e-12, max_terms: int = 400) -> float:
    """E[floor(N/divisor)] for N ~ Poisson(lam) as a truncated analytic series.

    floor(N/d) = sum_{k>=1} 1[N >= k*d], so E[floor(N/d)] = sum_{k>=1} P(N >= k*d).
    Deterministic and analytic — not a simulation.
    """

    if lam <= 0.0 or divisor <= 1:
        return 0.0
    total = 0.0
    for k in range(1, max_terms + 1):
        probability = poisson_sf(lam, k * int(divisor) - 1)
        if probability < tolerance:
            break
        total += probability
    return total


# ---------------------------------------------------------------------------
# Evidence loaders.
# ---------------------------------------------------------------------------


def _history_value(raw_json: str | None, key: str) -> float | None:
    if not raw_json:
        return None
    try:
        data = json.loads(raw_json)
    except (TypeError, ValueError):
        return None
    if not isinstance(data, dict) or data.get(key) is None:
        return None
    try:
        return float(data[key])
    except (TypeError, ValueError):
        return None


def current_player_aggregates(
    conn: sqlite3.Connection, planning_event: int, cutoff: str
) -> dict[int, dict[str, float]]:
    """Completed-row per-player totals (minutes, saves, yellow, bonus, DefCon)."""

    rows = conn.execute(
        """SELECT pg.player_id, pg.minutes, pg.saves, pg.yellow_cards, pg.bonus, pg.defensive_contribution
             FROM player_gameweeks pg JOIN fixtures f ON f.id=pg.fixture_id
            WHERE f.finished=1 AND f.started=1 AND pg.event < ? AND f.event < ? AND f.event IS NOT NULL
              AND (f.kickoff_time IS NULL OR f.kickoff_time <= ?)""",
        (int(planning_event), int(planning_event), cutoff),
    ).fetchall()
    aggregates: dict[int, dict[str, float]] = {}
    for row in rows:
        minutes = row["minutes"]
        if minutes is None or float(minutes) <= 0:
            continue
        bucket = aggregates.setdefault(
            int(row["player_id"]),
            {"minutes": 0.0, "saves": 0.0, "yellow": 0.0, "bonus": 0.0, "defcon": 0.0, "rows": 0.0},
        )
        bucket["minutes"] += float(minutes)
        for key, column in (("saves", "saves"), ("yellow", "yellow_cards"), ("bonus", "bonus"), ("defcon", "defensive_contribution")):
            value = row[column]
            if value is not None:
                bucket[key] += float(value)
        bucket["rows"] += 1
    return aggregates


def position_pooled_current_rates(
    conn: sqlite3.Connection, planning_event: int, cutoff: str
) -> dict[str, dict[str, float]]:
    """Position-pooled current-season per-90 rates for saves/yellow/bonus/DefCon."""

    rows = conn.execute(
        """SELECT p.element_type, pg.minutes, pg.saves, pg.yellow_cards, pg.bonus, pg.defensive_contribution
             FROM player_gameweeks pg
             JOIN players p ON p.id=pg.player_id
             JOIN fixtures f ON f.id=pg.fixture_id
            WHERE f.finished=1 AND f.started=1 AND pg.event < ? AND f.event < ? AND f.event IS NOT NULL
              AND (f.kickoff_time IS NULL OR f.kickoff_time <= ?) AND pg.minutes > 0""",
        (int(planning_event), int(planning_event), cutoff),
    ).fetchall()
    totals: dict[str, dict[str, float]] = {}
    for row in rows:
        position = POSITION_IDS.get(int(row["element_type"])) if row["element_type"] is not None else None
        if position is None:
            continue
        bucket = totals.setdefault(position, {"minutes": 0.0, "saves": 0.0, "yellow": 0.0, "bonus": 0.0, "defcon": 0.0})
        minutes = float(row["minutes"])
        bucket["minutes"] += minutes
        for key, column in (("saves", "saves"), ("yellow", "yellow_cards"), ("bonus", "bonus"), ("defcon", "defensive_contribution")):
            value = row[column]
            if value is not None:
                bucket[key] += float(value)
    rates: dict[str, dict[str, float]] = {}
    for position, bucket in totals.items():
        minutes = bucket["minutes"] or 1.0
        rates[position] = {key: bucket[key] / minutes * 90.0 for key in ("saves", "yellow", "bonus", "defcon")}
    # League-wide fallbacks.
    league_minutes = sum(b["minutes"] for b in totals.values()) or 1.0
    league = {
        key: sum(b[key] for b in totals.values()) / league_minutes * 90.0
        for key in ("saves", "yellow", "bonus", "defcon")
    }
    rates["__league__"] = league
    return rates


def previous_season_saves_per90(conn: sqlite3.Connection, player_id: int, season: str = "2025/26") -> dict[str, Any]:
    """Same-player previous-season saves per 90 from official history_past."""

    for row in repo.player_season_histories(conn, int(player_id)):
        if str(row.get("season_name")) != season:
            continue
        minutes = row.get("minutes")
        saves = _history_value(row.get("raw_json"), "saves")
        if minutes and float(minutes) > 0 and saves is not None:
            return {"rate": saves / float(minutes) * 90.0, "minutes": float(minutes), "saves": saves, "season": season}
        return {"rate": None, "minutes": float(minutes or 0), "saves": saves, "season": season}
    return {"rate": None, "minutes": 0.0, "saves": None, "season": season}


def _shrink_rate(current_minutes: float, current_rate: float | None, prior_rate: float, ess: float) -> float:
    if current_rate is None or current_minutes <= 0:
        return prior_rate
    return (current_minutes * current_rate + ess * prior_rate) / (current_minutes + ess)


# ---------------------------------------------------------------------------
# Team-minutes mass coherence diagnostics.
# ---------------------------------------------------------------------------


def team_minutes_coherence(candidates: Iterable[Mapping[str, Any]], config: XPtsConfig) -> dict[str, Any]:
    """Per team-fixture sums of P(start), P(appearance) and expected minutes.

    Player minutes are predicted independently, so these identities are
    diagnostics only — Minutes v1 is never normalised here.
    """

    sides: dict[tuple[int, int], dict[str, float]] = {}
    for candidate in candidates:
        key = (int(candidate["fixture_id"]), int(candidate["team_id"]))
        bucket = sides.setdefault(key, {"p_start": 0.0, "p_appearance": 0.0, "expected_minutes": 0.0, "players": 0.0})
        payload = candidate["minutes"]
        p_start = float(payload.get("p_start") or 0.0)
        p_cameo = float(payload.get("p_cameo") or 0.0)
        bucket["p_start"] += p_start
        bucket["p_appearance"] += p_start + p_cameo
        bucket["expected_minutes"] += float(payload.get("expected_minutes") or 0.0)
        bucket["players"] += 1

    diagnostics: list[dict[str, Any]] = []
    for (fixture_id, team_id) in sorted(sides):
        bucket = sides[(fixture_id, team_id)]
        minutes_deviation = bucket["expected_minutes"] - 990.0
        starters_deviation = bucket["p_start"] - 11.0
        flags: list[str] = []
        if abs(starters_deviation) > config.team_starters_warn_abs:
            flags.append("TEAM_STARTER_MASS_WARN")
        if abs(minutes_deviation) > config.team_minutes_mass_warn_fraction * 990.0:
            flags.append("TEAM_MINUTES_MASS_WARN")
        diagnostics.append(
            {
                "fixture_id": fixture_id,
                "team_id": team_id,
                "players": int(bucket["players"]),
                "expected_starters_sum": round(bucket["p_start"], 6),
                "expected_appearances_sum": round(bucket["p_appearance"], 6),
                "expected_minutes_sum": round(bucket["expected_minutes"], 6),
                "starter_deviation_from_11": round(starters_deviation, 6),
                "minutes_deviation_from_990": round(minutes_deviation, 6),
                "risk_flags": flags,
            }
        )
    severe = [
        d for d in diagnostics
        if abs(d["minutes_deviation_from_990"]) > config.team_minutes_mass_fail_fraction * 990.0
        or abs(d["starter_deviation_from_11"]) > config.team_starters_fail_abs
    ]
    return {
        "sides": diagnostics,
        "severe_incoherence": severe,
        "max_abs_minutes_deviation": max((abs(d["minutes_deviation_from_990"]) for d in diagnostics), default=0.0),
        "max_abs_starter_deviation": max((abs(d["starter_deviation_from_11"]) for d in diagnostics), default=0.0),
    }


# ---------------------------------------------------------------------------
# Component models.
# ---------------------------------------------------------------------------


def _defcon_p_hit(position: str, actions_per90: float, expected_minutes: float, rules: ScoringRules) -> float:
    threshold = rules.defcon_threshold_for(position)
    if threshold is None or position not in rules.defcon_positions:
        return 0.0
    lam = max(0.0, actions_per90) * max(0.0, expected_minutes) / 90.0
    return poisson_tail_probability(lam, threshold)


def minute_state_mixture(minutes_payload: Mapping[str, Any]) -> list[tuple[float, float]] | None:
    """Frozen compact minute-state distribution (probability, mean minutes).

    Returns ``None`` for a legacy minutes run with no frozen states, so the
    caller can fall back to the old expected-minutes evaluation.
    """

    distribution = minutes_payload.get("minute_state_distribution")
    if not distribution:
        return None
    mixture = [
        (max(0.0, float(entry.get("probability") or 0.0)), max(0.0, float(entry.get("conditional_mean_minutes") or 0.0)))
        for entry in distribution
    ]
    if not any(probability > 0.0 for probability, _ in mixture):
        return None
    return mixture


def defcon_xpts_with_mixture(
    position: str, actions_per90: float, minutes_payload: Mapping[str, Any], rules: ScoringRules
) -> tuple[float, float, bool]:
    """DefCon expected points and hit probability through the minute mixture.

    ``P(Poisson(lambda(m)) >= threshold)`` is NOT globally convex in ``m``, so
    evaluating it once at ``E[minutes]`` carries no universal signed bias.  The
    mixture integrates the frozen states directly instead.
    """

    threshold = rules.defcon_threshold_for(position)
    if threshold is None or position not in rules.defcon_positions:
        return 0.0, 0.0, False
    mixture = minute_state_mixture(minutes_payload)
    if mixture is None:
        minutes = float(minutes_payload.get("expected_minutes") or 0.0)
        hit = _defcon_p_hit(position, actions_per90, minutes, rules)
        return rules.defcon_points * hit, hit, False
    # A zero-probability state contributes zero weight, so its conditional mean
    # is irrelevant and must never be evaluated.
    hit = sum(
        probability * poisson_tail_probability(max(0.0, actions_per90) * mean_minutes / 90.0, threshold)
        for probability, mean_minutes in mixture
        if probability > 0.0
    )
    return rules.defcon_points * hit, hit, True


def save_xpts_with_mixture(
    saves_per90: float, pressure: float, minutes_payload: Mapping[str, Any], rules: ScoringRules
) -> tuple[float, bool]:
    """Save expected points through the same frozen minute mixture."""

    mixture = minute_state_mixture(minutes_payload)
    if mixture is None:
        minutes = float(minutes_payload.get("expected_minutes") or 0.0)
        return expected_floor_poisson_ratio(
            max(0.0, saves_per90) * minutes / 90.0 * pressure, rules.saves_per_point
        ), False
    value = sum(
        probability * expected_floor_poisson_ratio(
            max(0.0, saves_per90) * mean_minutes / 90.0 * pressure, rules.saves_per_point
        )
        for probability, mean_minutes in mixture
        if probability > 0.0
    )
    return value, True


def _clean_sheet_probability(
    minutes_payload: Mapping[str, Any], lambda_against: float, config: XPtsConfig
) -> float:
    """P(CS eligibility and no opponent goal during personal exposure).

    Two transparent branches.  The model's own ``p_60_plus`` is conserved: the
    starting branch takes ``min(p_60_plus, p_start)`` and the rare long-cameo
    branch takes the remainder, so a projection with ``p_60_plus = 0`` can
    never earn clean-sheet points.  Each branch applies a homogeneous goal
    hazard over its own on-pitch exposure (``exp(-lambda * minutes / 90)``).
    Flagged ``CS_MINUTES_APPROX_V1`` by the caller.
    """

    p_start = float(minutes_payload.get("p_start") or 0.0)
    p_60_plus = float(minutes_payload.get("p_60_plus") or 0.0)
    m_start = float(minutes_payload.get("expected_minutes_if_start") or 0.0)
    p60_given_start = min(1.0, p_60_plus / p_start) if p_start > 1e-9 else 0.0
    start_60 = p_start * p60_given_start
    cameo_60 = max(0.0, p_60_plus - start_60)
    start_term = start_60 * math.exp(-max(0.0, lambda_against) * m_start / 90.0)
    cameo_term = cameo_60 * math.exp(
        -max(0.0, lambda_against) * config.cameo_60_exposure_minutes / 90.0
    )
    return max(0.0, min(1.0, start_term + cameo_term))


def _goals_conceded_deduction(
    minutes_payload: Mapping[str, Any], lambda_against: float, goals_per_deduction: int, points_per_deduction: int
) -> float:
    """-points_per_deduction · E[floor(N / goals_per_deduction)] (analytic).

    For the official rule (-1 point per 2 goals conceded) this is exactly
    ``-1 * E[floor(N/2)]``, never ``floor(E[N]/2)``.  Averaged over the start
    and cameo exposure branches with their own on-pitch minutes.
    """

    divisor = int(goals_per_deduction) if int(goals_per_deduction) > 0 else 2
    p_start = float(minutes_payload.get("p_start") or 0.0)
    p_cameo = float(minutes_payload.get("p_cameo") or 0.0)
    m_start = float(minutes_payload.get("expected_minutes_if_start") or 0.0)
    m_cameo = float(minutes_payload.get("expected_minutes_if_cameo") or 0.0)
    lambda_against = max(0.0, lambda_against)

    def expected_floor(lam: float) -> float:
        if divisor == 2:
            return expected_floor_half(lam)
        return expected_floor_poisson_ratio(lam, divisor)

    expected_steps = p_start * expected_floor(lambda_against * m_start / 90.0) + p_cameo * expected_floor(
        lambda_against * m_cameo / 90.0
    )
    return -float(points_per_deduction) * expected_steps


# ---------------------------------------------------------------------------
# Projection build.
# ---------------------------------------------------------------------------


def build_xpts_projections(
    conn: sqlite3.Connection,
    *,
    event: int,
    cutoff: str,
    minutes_run_id: int,
    team_run_id: int,
    team_baseline_run_id: int | None,
    rate_run_id: int,
    config: XPtsConfig | None = None,
    rules: ScoringRules | None = None,
) -> dict[str, Any]:
    """Build every player-fixture xPts projection from explicit frozen runs."""

    config = config or XPtsConfig()
    rules = rules or DEFAULT_SCORING_RULES
    generated_at = utc_now()

    # --- load frozen inputs -------------------------------------------------
    minutes_records = analytics.frozen_predictions(conn, int(minutes_run_id), [analytics.MINUTES_V1_KIND])
    team_records = analytics.team_fixture_projections(conn, int(team_run_id))
    baseline_records = (
        analytics.team_fixture_projections(conn, int(team_baseline_run_id)) if team_baseline_run_id else []
    )
    rate_records = analytics.player_rate_projections(conn, int(rate_run_id))

    team_by_side: dict[tuple[int, int], dict[str, Any]] = {
        (int(r["fixture_id"]), int(r["team_id"])): r for r in team_records
    }
    baseline_by_side: dict[tuple[int, int], dict[str, Any]] = {
        (int(r["fixture_id"]), int(r["team_id"])): r for r in baseline_records
    }
    rate_by_player: dict[tuple[int, str], dict[str, Any]] = {
        (int(r["player_id"]), str(r["component"])): r for r in rate_records
    }

    positions: dict[int, str] = {}
    player_team: dict[int, int] = {}
    for row in conn.execute("SELECT id, team_id, element_type FROM players WHERE is_active=1 AND team_id IS NOT NULL"):
        positions[int(row["id"])] = POSITION_IDS.get(int(row["element_type"])) if row["element_type"] is not None else None
        player_team[int(row["id"])] = int(row["team_id"])

    fixtures: dict[int, dict[str, Any]] = {
        int(row["id"]): dict(row)
        for row in conn.execute("SELECT id, event, team_h, team_a FROM fixtures WHERE event=?", (int(event),))
    }

    # Reference attacking environment (league baseline lambda) for save pressure.
    league_reference = None
    for record in team_records:
        baseline = (record.get("payload") or {}).get("league_baseline")
        if baseline is not None:
            league_reference = math.exp(float(baseline))
            break
    if league_reference is None:
        league_reference = sum(
            float((r.get("payload") or {}).get("expected_goals_for") or 0.0) for r in team_records
        ) / max(1, len(team_records)) or 1.0

    aggregates = current_player_aggregates(conn, int(event), cutoff)
    pooled_rates = position_pooled_current_rates(conn, int(event), cutoff)

    # --- first pass: raw expected xG/xA per player-fixture ------------------
    candidates: list[dict[str, Any]] = []
    for record in minutes_records:
        payload = record["payload"]
        player_id = int(record["player_id"])
        fixture_id = int(record["fixture_id"])
        fixture = fixtures.get(fixture_id)
        if fixture is None or player_id not in player_team:
            continue
        team_id = player_team[player_id]
        if team_id not in (int(fixture["team_h"]), int(fixture["team_a"])):
            continue
        opponent_id = int(fixture["team_a"]) if team_id == int(fixture["team_h"]) else int(fixture["team_h"])
        team_side = team_by_side.get((fixture_id, team_id))
        if team_side is None:
            continue
        candidates.append(
            {
                "player_id": player_id,
                "fixture_id": fixture_id,
                "event": int(event),
                "team_id": team_id,
                "opponent_id": opponent_id,
                "position": positions.get(player_id),
                "minutes_run_id": int(minutes_run_id),
                "team_run_id": int(team_run_id),
                "team_baseline_run_id": team_baseline_run_id,
                "rate_run_id": int(rate_run_id),
                "cutoff": cutoff,
                "minutes": payload,
                "team": team_side,
                "rate_xg": rate_by_player.get((player_id, COMPONENT_XG), {}).get("payload"),
                "rate_xa": rate_by_player.get((player_id, COMPONENT_XA), {}).get("payload"),
                "flags": [],
                "data_gaps": [],
            }
        )

    # --- team xG sanity pass: cap any impossible player mass -----------------
    residual: dict[tuple[int, int], dict[str, float]] = {}
    for candidate in candidates:
        payload = candidate["team"]["payload"]
        expected_minutes = float(candidate["minutes"].get("expected_minutes") or 0.0)
        rate_xg = (candidate["rate_xg"] or {}).get("posterior_mean")
        multiplier = _multiplier_from_row(candidate["team"])
        raw_xg = (float(rate_xg) * expected_minutes / 90.0 * multiplier) if rate_xg is not None else 0.0
        candidate["raw_expected_xg"] = raw_xg
        key = (candidate["fixture_id"], candidate["team_id"])
        bucket = residual.setdefault(key, {"sum_raw": 0.0, "lambda": float(payload.get("expected_goals_for") or 0.0)})
        bucket["sum_raw"] += raw_xg

    cap_scales: dict[tuple[int, int], float] = {}
    for key, bucket in residual.items():
        lambda_for = bucket["lambda"]
        sum_raw = bucket["sum_raw"]
        cap = config.team_xg_cap_fraction * lambda_for
        bucket["residual_bucket"] = max(0.0, lambda_for - sum_raw)
        bucket["excess_fraction"] = (sum_raw / lambda_for) if lambda_for > 0 else 0.0
        if lambda_for > 0 and sum_raw > cap:
            cap_scales[key] = cap / sum_raw
            bucket["cap_applied"] = True
        else:
            cap_scales[key] = 1.0
            bucket["cap_applied"] = False

    # --- team-minutes mass coherence diagnostics (never normalised) ----------
    coherence = team_minutes_coherence(candidates, config)
    input_team_coherence = any(
        (candidate["minutes"].get("team_coherence") or {}).get("applied") for candidate in candidates
    )

    # --- second pass: components -------------------------------------------
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        rows.append(
            _build_row(conn, candidate, config, rules, cap_scales, residual, baseline_by_side,
                       league_reference, aggregates, pooled_rates, generated_at)
        )

    meta = {
        "model_version": XPTS_MODEL_VERSION,
        "scoring_rules_version": SCORING_RULES_VERSION,
        "config_hash": config.config_hash(),
        "scoring_hash": rules.scoring_hash(),
        "generated_at": generated_at,
        "data_cutoff": cutoff,
        "event": int(event),
        "minutes_run_id": int(minutes_run_id),
        "team_run_id": int(team_run_id),
        "team_baseline_run_id": team_baseline_run_id,
        "rate_run_id": int(rate_run_id),
        "league_reference_lambda": round(float(league_reference), 6),
        "input_team_coherence": input_team_coherence,
        "residual_buckets": {
            f"{k[0]}:{k[1]}": {kk: (vv if isinstance(vv, bool) else round(vv, 6)) for kk, vv in v.items()}
            for k, v in sorted(residual.items())
        },
    }
    return {"rows": rows, "meta": meta, "coherence": coherence}


def _multiplier_from_row(team_row: Mapping[str, Any]) -> float:
    home_term = float((team_row.get("payload") or {}).get("home_advantage") or 0.0) if team_row.get("venue") == "home" else 0.0
    defence = float((team_row.get("payload") or {}).get("opponent_defence_rating") or 0.0)
    return math.exp(home_term + defence)


def _build_row(
    conn: sqlite3.Connection,
    candidate: Mapping[str, Any],
    config: XPtsConfig,
    rules: ScoringRules,
    cap_scales: Mapping[tuple[int, int], float],
    residual: Mapping[tuple[int, int], Mapping[str, float]],
    baseline_by_side: Mapping[tuple[int, int], Mapping[str, Any]],
    league_reference: float,
    aggregates: Mapping[int, Mapping[str, float]],
    pooled_rates: Mapping[str, Mapping[str, float]],
    generated_at: str,
) -> dict[str, Any]:
    player_id = int(candidate["player_id"])
    fixture_id = int(candidate["fixture_id"])
    team_id = int(candidate["team_id"])
    position = candidate["position"]
    minutes = candidate["minutes"]
    team_payload = candidate["team"]["payload"]
    key = (fixture_id, team_id)

    flags = list(candidate["flags"])
    data_gaps = list(candidate["data_gaps"])
    expected_minutes = float(minutes.get("expected_minutes") or 0.0)
    p_start = float(minutes.get("p_start") or 0.0)
    p_cameo = float(minutes.get("p_cameo") or 0.0)
    p_60_plus = float(minutes.get("p_60_plus") or 0.0)

    multiplier = _multiplier_from_row(candidate["team"])
    scale = float(cap_scales.get(key, 1.0))
    if scale < 1.0:
        flags.append("TEAM_XG_CAP_APPLIED")
    excess = float((residual.get(key) or {}).get("excess_fraction") or 0.0)
    if scale >= 1.0 and excess > config.team_xg_excess_warn_fraction:
        flags.append("TEAM_XG_EXCESS_WARN")

    raw_expected_xg = float(candidate.get("raw_expected_xg") or 0.0)
    adjusted_expected_xg = raw_expected_xg * scale

    rate_xg = candidate["rate_xg"] or {}
    rate_xa = candidate["rate_xa"] or {}
    if rate_xg.get("posterior_mean") is None:
        flags.append("MISSING_XG_RATE")
    if rate_xa.get("posterior_mean") is None:
        flags.append("MISSING_XA_RATE")
    if "PENALTIES_EMBEDDED" not in flags:
        flags.append("PENALTIES_EMBEDDED")

    # appearance: 1 point <60, 2 points 60+
    p_appearance = max(0.0, min(1.0, p_start + p_cameo))
    appearance_xpts = rules.appearance_points(p_appearance, p_60_plus)

    lambda_against = float(team_payload.get("expected_goals_against") or 0.0)

    goal_points = rules.goal_points_for(position) if position else 0
    goal_xpts = adjusted_expected_xg * goal_points

    xa_rate = rate_xa.get("posterior_mean")
    expected_xa = (float(xa_rate) * expected_minutes / 90.0 * multiplier) if xa_rate is not None else 0.0
    expected_fpl_assists = expected_xa * config.fpl_assist_mapping_coefficient
    assist_xpts = expected_fpl_assists * rules.assist_points
    if not config.assist_mapping_calibrated:
        flags.append("FPL_ASSIST_MAPPING_UNCALIBRATED")

    cs_position_points = rules.clean_sheet_points_for(position) if position else 0
    cs_prob = _clean_sheet_probability(minutes, lambda_against, config)
    clean_sheet_xpts = cs_position_points * cs_prob
    flags.append("CS_MINUTES_APPROX_V1")

    if position in rules.goals_conceded_positions:
        gc_points = abs(rules.goals_conceded_points_for(position)) or 1
        goals_conceded_xpts = _goals_conceded_deduction(
            minutes, lambda_against, rules.goals_conceded_per_deduction, gc_points
        )
    else:
        goals_conceded_xpts = 0.0

    # DefCon
    if position in rules.defcon_positions:
        aggregate = aggregates.get(player_id, {})
        current_minutes = float(aggregate.get("minutes") or 0.0)
        current_actions = float(aggregate.get("defcon") or 0.0)
        current_rate = (current_actions / current_minutes * 90.0) if current_minutes > 0 else None
        pooled = pooled_rates.get(position) or pooled_rates.get("__league__") or {"defcon": 0.0}
        actions_per90 = _shrink_rate(current_minutes, current_rate, float(pooled.get("defcon") or 0.0), config.defcon_prior_strength_minutes)
        if current_minutes <= 0:
            flags.append("DEFCON_PRIOR_WEAK")
        flags.append("HISTORICAL_POSITION_UNKNOWN")
        defcon_xpts, p_defcon_hit, defcon_mixture_used = defcon_xpts_with_mixture(
            position, actions_per90, minutes, rules
        )
        if not defcon_mixture_used:
            flags.append("DEFCON_MINUTE_MIXTURE_UNAVAILABLE")
    else:
        actions_per90 = None
        p_defcon_hit = 0.0
        defcon_xpts = 0.0

    # GK saves
    if position == "GKP":
        aggregate = aggregates.get(player_id, {})
        current_minutes = float(aggregate.get("minutes") or 0.0)
        current_saves = float(aggregate.get("saves") or 0.0)
        current_rate = (current_saves / current_minutes * 90.0) if current_minutes > 0 else None
        prev = previous_season_saves_per90(conn, player_id)
        pooled_gk = (pooled_rates.get("GKP") or {}).get("saves")
        if prev.get("rate") is not None and float(prev.get("minutes") or 0.0) >= config.prev_season_min_prior_minutes:
            prior_rate = float(prev["rate"])
            prior_source = "prev_season_same_player"
        elif pooled_gk is not None:
            prior_rate = float(pooled_gk)
            prior_source = "goalkeeper_pooled"
            flags.append("SAVE_MODEL_LOW_CONFIDENCE")
        else:
            prior_rate = 0.0
            prior_source = "no_prior"
            flags.append("SAVE_MODEL_LOW_CONFIDENCE")
        saves_per90 = _shrink_rate(current_minutes, current_rate, prior_rate, config.saves_prior_strength_minutes)
        pressure = float(lambda_against) / league_reference if league_reference > 0 else 1.0
        pressure = max(config.save_pressure_multiplier_min, min(config.save_pressure_multiplier_max, pressure))
        save_xpts, save_mixture_used = save_xpts_with_mixture(saves_per90, pressure, minutes, rules)
        if not save_mixture_used:
            flags.append("SAVE_MINUTE_MIXTURE_UNAVAILABLE")
        lambda_saves = saves_per90 * expected_minutes / 90.0 * pressure
        save_info = {"saves_per90_posterior": round(saves_per90, 6), "prior_source": prior_source,
                     "pressure_multiplier": round(pressure, 6), "lambda_saves": round(lambda_saves, 6),
                     "minute_mixture_used": save_mixture_used}
    else:
        lambda_saves = None
        save_info = None
        save_xpts = 0.0

    # Yellow cards (all positions) — heavily shrunk.
    aggregate = aggregates.get(player_id, {})
    current_minutes = float(aggregate.get("minutes") or 0.0)
    current_yellow_rate = (float(aggregate.get("yellow") or 0.0) / current_minutes * 90.0) if current_minutes > 0 else None
    pooled = pooled_rates.get(position) or pooled_rates.get("__league__") or {"yellow": 0.0}
    yellow_per90 = _shrink_rate(current_minutes, current_yellow_rate, float(pooled.get("yellow") or 0.0), config.yellow_prior_strength_minutes)
    expected_yellows = yellow_per90 * expected_minutes / 90.0
    yellow_card_xpts = expected_yellows * rules.yellow_card_points

    # Bonus (soft residual only).
    current_bonus_rate = (float(aggregate.get("bonus") or 0.0) / current_minutes * 90.0) if current_minutes > 0 else None
    pooled_bonus = pooled_rates.get(position) or pooled_rates.get("__league__") or {"bonus": 0.0}
    bonus_per90 = _shrink_rate(current_minutes, current_bonus_rate, float(pooled_bonus.get("bonus") or 0.0), config.bonus_prior_strength_minutes)
    expected_bonus = bonus_per90 * expected_minutes / 90.0
    bonus_xpts = expected_bonus
    flags.extend(["BONUS_SOFT", "BONUS_LOW_CONFIDENCE", "BPS_RULE_DISCONTINUITY"])

    core_xpts = (
        appearance_xpts + goal_xpts + assist_xpts + clean_sheet_xpts + goals_conceded_xpts
        + defcon_xpts + save_xpts + yellow_card_xpts
    )
    soft_xpts = bonus_xpts
    total_xpts = core_xpts + soft_xpts

    # Team-model sensitivity: recompute the football components with the naive
    # team baseline (no team-specific signal).
    baseline_row = baseline_by_side.get(key)
    if baseline_row is not None:
        baseline_payload = baseline_row["payload"]
        baseline_multiplier = 1.0  # naive venue-average baseline carries no team/opponent signal
        baseline_lambda_against = float(baseline_payload.get("expected_goals_against") or 0.0)
        b_goal = raw_expected_xg * scale * baseline_multiplier * goal_points
        b_assist = expected_xa * baseline_multiplier * config.fpl_assist_mapping_coefficient * rules.assist_points
        b_cs = cs_position_points * _clean_sheet_probability(minutes, baseline_lambda_against, config)
        b_gc = (
            _goals_conceded_deduction(
                minutes, baseline_lambda_against, rules.goals_conceded_per_deduction,
                abs(rules.goals_conceded_points_for(position)) or 1,
            )
            if position in rules.goals_conceded_positions
            else 0.0
        )
        b_save = 0.0
        if position == "GKP" and save_info is not None:
            b_pressure = max(
                config.save_pressure_multiplier_min,
                min(config.save_pressure_multiplier_max, baseline_lambda_against / league_reference if league_reference > 0 else 1.0),
            )
            b_save, _ = save_xpts_with_mixture(save_info["saves_per90_posterior"], b_pressure, minutes, rules)
        total_without_team_model = (
            appearance_xpts + b_goal + b_assist + b_cs + b_gc + defcon_xpts + b_save + yellow_card_xpts + soft_xpts
        )
        sensitivity = abs(total_xpts - total_without_team_model)
        if sensitivity > config.team_model_sensitive_threshold:
            flags.append("TEAM_MODEL_SENSITIVE")
    else:
        total_without_team_model = None
        sensitivity = None

    if "NO_TEAM_SPECIFIC_HISTORICAL_PRIOR" in (team_payload.get("risk_flags") or []):
        flags.append("NO_TEAM_SPECIFIC_HISTORICAL_PRIOR")

    official_ep_next = analytics.ep_next_as_of(conn, player_id, str(candidate.get("cutoff") or ""))

    return {
        "player_id": player_id,
        "fixture_id": fixture_id,
        "event": int(candidate["event"]),
        "team_id": team_id,
        "opponent_id": int(candidate["opponent_id"]),
        "position": position,
        "minutes_run_id": int(candidate["minutes_run_id"]),
        "team_run_id": int(candidate["team_run_id"]),
        "rate_run_id": int(candidate["rate_run_id"]),
        "expected_minutes": round(expected_minutes, 6),
        # MC-ready on-pitch rate inputs (whole-match expectation / expected
        # minutes).  Zero when the player has no expected exposure.
        "fixture_xg_per90": round(adjusted_expected_xg * 90.0 / expected_minutes, 6) if expected_minutes > 0 else 0.0,
        "fixture_xa_per90": round(expected_xa * 90.0 / expected_minutes, 6) if expected_minutes > 0 else 0.0,
        "yellow_per90": round(yellow_per90, 6),
        "p_start": round(p_start, 6),
        "p_appearance": round(p_appearance, 6),
        "p_60_plus": round(p_60_plus, 6),
        "fixture_multiplier": round(multiplier, 6),
        "raw_expected_xg": round(raw_expected_xg, 6),
        "adjusted_expected_xg": round(adjusted_expected_xg, 6),
        "xg_scale_factor": round(scale, 6),
        "expected_xa": round(expected_xa, 6),
        "expected_fpl_assists": round(expected_fpl_assists, 6),
        "appearance_xpts": round(appearance_xpts, 6),
        "goal_xpts": round(goal_xpts, 6),
        "assist_xpts": round(assist_xpts, 6),
        "clean_sheet_xpts": round(clean_sheet_xpts, 6),
        "clean_sheet_probability": round(cs_prob, 6),
        "goals_conceded_xpts": round(goals_conceded_xpts, 6),
        "defcon_xpts": round(defcon_xpts, 6),
        "defcon_p_hit": round(p_defcon_hit, 6),
        "defcon_actions_per90": round(actions_per90, 6) if actions_per90 is not None else None,
        "save_xpts": round(save_xpts, 6),
        "save_model": save_info,
        "yellow_card_xpts": round(yellow_card_xpts, 6),
        "bonus_xpts": round(bonus_xpts, 6),
        "core_xpts": round(core_xpts, 6),
        "soft_xpts": round(soft_xpts, 6),
        "total_xpts": round(total_xpts, 6),
        "total_xpts_without_team_model": round(total_without_team_model, 6) if total_without_team_model is not None else None,
        "team_model_sensitivity": round(sensitivity, 6) if sensitivity is not None else None,
        "official_ep_next": official_ep_next,
        "risk_flags": sorted(set(flags)),
        "data_gaps": sorted(set(data_gaps)),
        "provenance": {
            "inputs": {
                "minutes_run_id": int(candidate["minutes_run_id"]),
                "team_run_id": int(candidate["team_run_id"]),
                "rate_run_id": int(candidate["rate_run_id"]),
                "team_baseline_run_id": candidate.get("team_baseline_run_id"),
            },
            "fixture_multiplier_formula": "exp(home_term + opponent_defence_rating); opponent/venue only (no team-attack double count)",
            "penalties": "embedded_in_xG (no NPxG separation; no separate penalty expectation)",
            "assist_mapping": "xA x coefficient (UNCALIBRATED, coefficient config-driven and hashed)",
            "defcon_actions_measure": "player_gameweeks.defensive_contribution is an action COUNT compared to the position threshold",
            "components_core": list(CORE_COMPONENTS),
            "components_soft": list(SOFT_COMPONENTS),
        },
        "model_version": XPTS_MODEL_VERSION,
        "scoring_rules_version": SCORING_RULES_VERSION,
        "generated_at": generated_at,
        "data_cutoff": candidate.get("cutoff"),
    }


# ---------------------------------------------------------------------------
# Readiness gate.
# ---------------------------------------------------------------------------


def readiness_summary(
    context: Any,
    rows: list[dict[str, Any]],
    coherence: Mapping[str, Any] | None = None,
    *,
    deadline_status: str | None = None,
    data_cutoff: str | None = None,
    deadline: str | None = None,
    rules_verified: bool = True,
    strict_coherence: bool = False,
    config: XPtsConfig | None = None,
) -> dict[str, Any]:
    """xPts-specific gate; independent of the setup health gate.

    ``strict_coherence`` is set when the frozen minutes input is a
    team-coherent challenger: the 11-starter and 990-minute identities are then
    enforced to tight tolerances and a material violation FAILs, not WARNs.
    """

    config = config or XPtsConfig()
    fail_reasons: list[str] = []
    warn_reasons: list[str] = []
    flag_counts: dict[str, int] = {}
    if context is not None:
        if (getattr(context, "health", None) or {}).get("status") == "FAIL":
            fail_reasons.append("PLANNING_CONTEXT_FAILED: " + "; ".join((context.health or {}).get("fail_reasons") or []))
    if data_cutoff is not None and deadline is not None and deadline_status != "LATE_FREEZE":
        if parse_utc(data_cutoff) > parse_utc(deadline):
            fail_reasons.append("MODEL_INPUTS_AFTER_DEADLINE: a pre-deadline freeze cutoff may not exceed the deadline")
    if not rules_verified:
        fail_reasons.append("SCORING_RULES_UNAVAILABLE: scoring rules could not be verified against the official payload")
    if not rows:
        fail_reasons.append("NO_XPTS_ROWS: no player-fixture projections were produced")
    for row in rows:
        for flag in row.get("risk_flags", []):
            flag_counts[flag] = flag_counts.get(flag, 0) + 1
        for field_name in ("core_xpts", "soft_xpts", "total_xpts", "appearance_xpts"):
            value = row.get(field_name)
            if value is None or not math.isfinite(float(value)):
                fail_reasons.append(f"NON_FINITE_COMPONENT: player {row.get('player_id')} fixture {row.get('fixture_id')}")
        if "MISSING_XG_RATE" in row.get("risk_flags", []) or "MISSING_XA_RATE" in row.get("risk_flags", []):
            fail_reasons.append(f"MISSING_REQUIRED_RATE: player {row.get('player_id')}")
    if coherence:
        for side in coherence.get("severe_incoherence", []):
            fail_reasons.append(
                f"TEAM_MINUTES_MASS_INCOHERENT: fixture {side['fixture_id']} team {side['team_id']} "
                f"(minutes {side['minutes_deviation_from_990']}, starters {side['starter_deviation_from_11']})"
            )
        for side in coherence.get("sides", []):
            if strict_coherence:
                if abs(side["starter_deviation_from_11"]) > config.coherent_start_tolerance:
                    fail_reasons.append(
                        f"COHERENT_START_SUM_VIOLATION: fixture {side['fixture_id']} team {side['team_id']} "
                        f"deviation {side['starter_deviation_from_11']}"
                    )
                if abs(side["minutes_deviation_from_990"]) > config.coherent_minutes_tolerance:
                    fail_reasons.append(
                        f"COHERENT_MINUTES_SUM_VIOLATION: fixture {side['fixture_id']} team {side['team_id']} "
                        f"deviation {side['minutes_deviation_from_990']}"
                    )
            for flag in side.get("risk_flags", []):
                if flag in {"TEAM_MINUTES_MASS_WARN", "TEAM_STARTER_MASS_WARN"}:
                    warn_reasons.append(f"{flag}: fixture {side['fixture_id']} team {side['team_id']}")
    for flag in (
        "TEAM_MODEL_SENSITIVE",
        "NO_TEAM_SPECIFIC_HISTORICAL_PRIOR",
        "CS_MINUTES_APPROX_V1",
        "FPL_ASSIST_MAPPING_UNCALIBRATED",
        "PENALTIES_EMBEDDED",
        "DEFCON_PRIOR_WEAK",
        "HISTORICAL_POSITION_UNKNOWN",
        "SAVE_MODEL_LOW_CONFIDENCE",
        "BONUS_SOFT",
        "BONUS_LOW_CONFIDENCE",
        "BPS_RULE_DISCONTINUITY",
        "TEAM_XG_CAP_APPLIED",
        "TEAM_XG_EXCESS_WARN",
    ):
        if flag_counts.get(flag):
            warn_reasons.append(f"{flag}: {flag_counts[flag]} player-fixture projections affected")
    status = "FAIL" if fail_reasons else ("WARN" if warn_reasons else "PASS")
    return {
        "status": status,
        "strict_coherence": strict_coherence,
        "fail_reasons": sorted(set(fail_reasons)),
        "warn_reasons": warn_reasons,
        "counts": {f"rows_{flag.lower()}": count for flag, count in sorted(flag_counts.items())},
    }
