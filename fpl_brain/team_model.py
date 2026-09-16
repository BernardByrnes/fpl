"""Team Attack/Defence Model v1 — transparent per-fixture scoring environment.

This module produces football-event expectations only.  It deliberately stops
before Fantasy Premier League points: no clean-sheet points, no goal points,
no xPts.  The future xPts layer is expected to consume a TeamFixtureProjection
together with a PlayerRateProjection and a PlayerMinutesProjection.

Model form (log-linear Poisson, the simplest credible v1 family):

    log(lambda_home) = mu + home_advantage + attack_home + defence_weakness_away
    log(lambda_away) = mu +                    attack_away + defence_weakness_home

Fitting is a small, deterministic, strongly-shrunk coordinate descent — never
an unconstrained fit.  Team attack/defence parameters are pulled toward the
league average with a configured prior strength (in match-equivalents) and the
current-season evidence is recency-weighted.  Official FDR is never used
numerically; it stays display/context only.

xG source discipline
--------------------
The canonical team-fixture xG is the side-sum of the stored official
per-player ``expected_goals`` on that fixture.  Fixture side is taken from
``player_gameweeks.was_home`` (not ``players.team_id``) so a mid-season
transfer never re-attributes a past fixture.  A second, independent official
signal — the maximum ``expected_goals_conceded`` among that side's players —
is reconciled against the opponent's side-sum and any material disagreement is
surfaced, never averaged away.  ``expected_goals_conceded`` is pro-rated by
time on pitch, so it is used only as a reconciliation check, never summed.
"""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass, fields
from typing import Any, Iterable, Mapping

from . import analytics
from . import historical_observations as historical
from .utils import parse_utc, utc_now

# v1.1.0: the fixture-side xG evidence is read under the canonical causal
# historical boundary.  Previously the fixture was bounded (event/kickoff) but
# the ROWS were not, so a row written after the cutoff -- or a scheduled
# placeholder written after kickoff -- contributed to the side xG sum.  This
# changes the model's semantics for any cutoff where such a row exists, even
# though every currently certified cutoff is unaffected (no stored row both
# satisfies the old read and falls outside the boundary).
TEAM_MODEL_VERSION = "team_strength_v1.1.0"
TEAM_BASELINE_MODEL_VERSION = "team_naive_v1.0.0"

VENUE_HOME = "home"
VENUE_AWAY = "away"

# No previous-season *team-match* history exists locally, so every team starts
# from the league-average prior.  This flag is attached to every projection so
# the weak prior is never silently hidden (a better prior source can replace
# it later without changing the interface).
NO_HISTORICAL_TEAM_PRIOR = "NO_TEAM_SPECIFIC_HISTORICAL_PRIOR"


@dataclass(frozen=True)
class TeamStrengthConfig:
    """Every tunable in one versioned structure; no magic numbers in code."""

    # Recency: exponential decay over a team's own league matches.
    current_match_half_life: float = 11.0
    # Ridge/shrinkage strength, in match-equivalents of league-average prior.
    attack_prior_strength: float = 12.0
    defence_prior_strength: float = 12.0
    home_advantage_prior_strength: float = 12.0
    iterations: int = 6
    # Floor used inside the log so a zero-xG match stays finite.
    xg_epsilon: float = 0.05
    # Evidence thresholds used only for risk flags.
    low_evidence_matches: int = 4
    home_advantage_low_evidence_matches: int = 20
    high_regularisation_share: float = 0.25
    # Reconciliation tolerance between side-sum xG and opponent max xGC.
    xg_xgc_tolerance: float = 0.25

    def config_hash(self) -> str:
        values = {item.name: getattr(self, item.name) for item in fields(self)}
        return analytics.canonical_hash({"model": TEAM_MODEL_VERSION, **values})


# ---------------------------------------------------------------------------
# Official team-fixture xG derivation.
# ---------------------------------------------------------------------------


def _side_xg_from_rows(rows: Iterable[Any]) -> dict[str, Any]:
    """Side-summed official xG per side, from already-selected rows.

    ``was_home`` decides the side (never ``players.team_id``), so a mid-season
    transfer cannot re-attribute a past fixture.  ``expected_goals_conceded``
    is pro-rated by time on pitch, so it is reported as a per-side maximum for
    reconciliation only and never summed.
    """

    home_xg = away_xg = 0.0
    home_xgc_max: float | None = None
    away_xgc_max: float | None = None
    missing = 0
    saw_home = saw_away = False
    for row in rows:
        was_home = int(row["was_home"])
        xg = row["expected_goals"]
        xgc = row["expected_goals_conceded"]
        if xg is None:
            missing += 1
            continue
        if was_home:
            saw_home = True
            home_xg += float(xg)
            if xgc is not None:
                home_xgc_max = float(xgc) if home_xgc_max is None else max(home_xgc_max, float(xgc))
        else:
            saw_away = True
            away_xg += float(xg)
            if xgc is not None:
                away_xgc_max = float(xgc) if away_xgc_max is None else max(away_xgc_max, float(xgc))
    return {
        "home_xg": home_xg if saw_home else None,
        "away_xg": away_xg if saw_away else None,
        "home_xgc_max": home_xgc_max,
        "away_xgc_max": away_xgc_max,
        "missing_xg_rows": missing,
    }


#: The row-level evidence for one fixture side.  ``fixture_id`` and
#: ``was_home`` are this model's own identity/venue filters, applied AFTER the
#: canonical causal boundary; the boundary itself is never restated here.
_SIDE_XG_SELECT = (
    "SELECT was_home, expected_goals, expected_goals_conceded "
    "FROM player_gameweeks pg JOIN fixtures f ON f.id = pg.fixture_id WHERE "
)


def fixture_side_xg(
    conn: sqlite3.Connection, fixture_id: int, *, as_of: str, planning_event: int
) -> dict[str, Any]:
    """Canonical side xG for one fixture as of ``as_of``.

    The fixture identity is explicit, but identity is not causality: a row for
    this fixture that the league had not yet written at the cutoff, or a
    scheduled placeholder, is not evidence about the match.  ``as_of`` is
    required and there is no "now" default (see
    :mod:`fpl_brain.historical_observations`).
    """

    cutoff = historical.require_as_of(as_of)
    rows = conn.execute(
        f"{_SIDE_XG_SELECT}{historical.OBSERVATION_SQL_CLAUSES}"
        " AND pg.fixture_id = ? AND pg.was_home IS NOT NULL",
        (*historical.boundary_params(cutoff, planning_event=int(planning_event)), int(fixture_id)),
    ).fetchall()
    return _side_xg_from_rows(rows)


def realised_fixture_side_xg(conn: sqlite3.Connection, fixture_id: int) -> dict[str, Any]:
    """The FINAL side xG of a played fixture, for scoring a past prediction.

    This is deliberately outside the historical boundary: a calibration read
    compares what the model said before a match with what the match actually
    produced, so it must see the realised value rather than the value that was
    observable at some earlier cutoff.  It has no cutoff parameter for that
    reason, and it is not a historical model input.
    """

    rows = conn.execute(
        f"{_SIDE_XG_SELECT}pg.fixture_id = ? AND pg.was_home IS NOT NULL",
        (int(fixture_id),),
    ).fetchall()
    return _side_xg_from_rows(rows)


def completed_fixture_xg(
    conn: sqlite3.Connection, planning_event: int, cutoff: str
) -> list[dict[str, Any]]:
    """Completed fixtures before the cutoff with canonical side xG.

    No-lookahead: a fixture must be finished and started, belong to an event
    strictly before the planning event, and kick off at or before the cutoff.
    The fixture bound alone is not sufficient -- the row-level evidence for each
    fixture is read under the same causal boundary -- and ``cutoff`` is
    required, with no implicit "now".
    """

    cutoff = historical.require_as_of(cutoff)
    out: list[dict[str, Any]] = []
    rows = conn.execute(
        """SELECT id, event, kickoff_time, team_h, team_a
             FROM fixtures
            WHERE finished=1 AND started=1 AND event < ?
              AND event IS NOT NULL
              AND (kickoff_time IS NULL OR kickoff_time <= ?)
            ORDER BY event, kickoff_time, id""",
        (int(planning_event), cutoff),
    ).fetchall()
    for fixture in rows:
        sides = fixture_side_xg(
            conn, int(fixture["id"]), as_of=cutoff, planning_event=int(planning_event)
        )
        if sides["home_xg"] is None or sides["away_xg"] is None:
            # A fixture with missing official xG cannot contribute team totals.
            continue
        diff_home = abs(float(sides["home_xg"]) - float(sides["away_xgc_max"])) if sides["away_xgc_max"] is not None else None
        diff_away = abs(float(sides["away_xg"]) - float(sides["home_xgc_max"])) if sides["home_xgc_max"] is not None else None
        out.append(
            {
                "fixture_id": int(fixture["id"]),
                "event": int(fixture["event"]),
                "kickoff_time": fixture["kickoff_time"],
                "team_h": int(fixture["team_h"]),
                "team_a": int(fixture["team_a"]),
                "home_xg": round(float(sides["home_xg"]), 6),
                "away_xg": round(float(sides["away_xg"]), 6),
                "home_xgc_max": sides["home_xgc_max"],
                "away_xgc_max": sides["away_xgc_max"],
                "reconciliation_diff": max(
                    [value for value in (diff_home, diff_away) if value is not None] or [0.0]
                ),
                "missing_xg_rows": sides["missing_xg_rows"],
            }
        )
    return out


def team_match_rows(
    conn: sqlite3.Connection, planning_event: int, cutoff: str
) -> list[dict[str, Any]]:
    """One row per team per completed fixture, with per-team recency weights."""

    fixtures = completed_fixture_xg(conn, planning_event, cutoff)
    rows: list[dict[str, Any]] = []
    for fixture in fixtures:
        rows.append(
            {
                "team_id": fixture["team_h"],
                "opponent_id": fixture["team_a"],
                "venue": VENUE_HOME,
                "xg_for": fixture["home_xg"],
                "xg_against": fixture["away_xg"],
                "fixture_id": fixture["fixture_id"],
                "event": fixture["event"],
                "kickoff_time": fixture["kickoff_time"],
                "reconciliation_diff": fixture["reconciliation_diff"],
                "opponent_xgc_max": fixture["away_xgc_max"],
            }
        )
        rows.append(
            {
                "team_id": fixture["team_a"],
                "opponent_id": fixture["team_h"],
                "venue": VENUE_AWAY,
                "xg_for": fixture["away_xg"],
                "xg_against": fixture["home_xg"],
                "fixture_id": fixture["fixture_id"],
                "event": fixture["event"],
                "kickoff_time": fixture["kickoff_time"],
                "reconciliation_diff": fixture["reconciliation_diff"],
                "opponent_xgc_max": fixture["home_xgc_max"],
            }
        )
    return rows


def attach_recency_weights(rows: Iterable[Mapping[str, Any]], half_life: float) -> list[dict[str, Any]]:
    """Weight each team-match by how many of that team's matches ago it was."""

    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["team_id"]), []).append(dict(row))
    weighted: list[dict[str, Any]] = []
    for team_id in sorted(grouped):
        ordered = sorted(
            grouped[team_id],
            key=lambda r: (r["event"], str(r["kickoff_time"] or ""), r["fixture_id"]),
        )
        for age, row in enumerate(reversed(ordered)):
            row["age_matches"] = age
            row["weight"] = 0.5 ** (age / float(half_life)) if half_life > 0 else 1.0
            weighted.append(row)
    return weighted


def _shrink(numerator: float, denominator: float, prior_strength: float) -> float:
    """Ridge/shrinkage estimator toward a zero prior: num / (den + strength)."""

    return numerator / (denominator + float(prior_strength)) if (denominator + prior_strength) > 0 else 0.0


def fit_team_strength(
    conn: sqlite3.Connection,
    planning_event: int,
    cutoff: str,
    config: TeamStrengthConfig | None = None,
) -> dict[str, Any]:
    """Fit shrunk team attack/defence and home advantage by coordinate descent."""

    config = config or TeamStrengthConfig()
    rows = attach_recency_weights(team_match_rows(conn, planning_event, cutoff), config.current_match_half_life)
    epsilon = float(config.xg_epsilon)
    teams = sorted({int(row["team_id"]) for row in rows})
    if not rows:
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
            "config_hash": config.config_hash(),
            "data_gaps": ["no completed fixtures with official xG before the cutoff"],
        }

    total_weight = sum(row["weight"] for row in rows)
    league_log_baseline = sum(row["weight"] * math.log(float(row["xg_for"]) + epsilon) for row in rows) / total_weight

    attack = {team: 0.0 for team in teams}
    defence = {team: 0.0 for team in teams}
    home_advantage = 0.0

    for _ in range(int(config.iterations)):
        # Home advantage from home team-matches only (away rows carry no plus).
        num = den = 0.0
        for row in rows:
            if row["venue"] != VENUE_HOME:
                continue
            residual = (
                math.log(float(row["xg_for"]) + epsilon)
                - league_log_baseline
                - attack[int(row["team_id"])]
                - defence[int(row["opponent_id"])]
            )
            num += row["weight"] * residual
            den += row["weight"]
        home_advantage = _shrink(num, den, config.home_advantage_prior_strength)

        for team in teams:
            num = den = 0.0
            for row in rows:
                if int(row["team_id"]) != team:
                    continue
                residual = (
                    math.log(float(row["xg_for"]) + epsilon)
                    - league_log_baseline
                    - (home_advantage if row["venue"] == VENUE_HOME else 0.0)
                    - defence[int(row["opponent_id"])]
                )
                num += row["weight"] * residual
                den += row["weight"]
            attack[team] = _shrink(num, den, config.attack_prior_strength)

        for team in teams:
            num = den = 0.0
            for row in rows:
                if int(row["team_id"]) != team:
                    continue
                opponent_home = 0.0 if row["venue"] == VENUE_HOME else 1.0
                residual = (
                    math.log(float(row["xg_against"]) + epsilon)
                    - league_log_baseline
                    - home_advantage * opponent_home
                    - attack[int(row["opponent_id"])]
                )
                num += row["weight"] * residual
                den += row["weight"]
            defence[team] = _shrink(num, den, config.defence_prior_strength)

    # Centre attack and defence to weighted mean zero; fold the shift into the
    # league baseline so every fitted lambda is unchanged.
    weight_totals = {
        team: sum(row["weight"] for row in rows if int(row["team_id"]) == team) for team in teams
    }
    total_team_weight = sum(weight_totals.values()) or 1.0
    mean_attack = sum(weight_totals[t] * attack[t] for t in teams) / total_team_weight
    mean_defence = sum(weight_totals[t] * defence[t] for t in teams) / total_team_weight
    attack = {team: value - mean_attack for team, value in attack.items()}
    defence = {team: value - mean_defence for team, value in defence.items()}
    league_log_baseline += mean_attack + mean_defence

    team_match_counts = {team: sum(1 for row in rows if int(row["team_id"]) == team) for team in teams}
    team_reconciliation_max = {
        team: max([row["reconciliation_diff"] for row in rows if int(row["team_id"]) == team] or [0.0])
        for team in teams
    }
    home_match_weight = sum(row["weight"] for row in rows if row["venue"] == VENUE_HOME)

    return {
        "league_log_baseline": round(league_log_baseline, 6),
        "home_advantage": round(home_advantage, 6),
        "attack": {team: round(value, 6) for team, value in attack.items()},
        "defence": {team: round(value, 6) for team, value in defence.items()},
        "teams": teams,
        "team_match_counts": team_match_counts,
        "team_weight_totals": {team: round(value, 6) for team, value in weight_totals.items()},
        "team_reconciliation_max": {team: round(value, 6) for team, value in team_reconciliation_max.items()},
        "home_match_weight": round(home_match_weight, 6),
        "match_count": len(rows) // 2,
        "total_team_weight": round(total_weight, 6),
        "config_hash": config.config_hash(),
        "data_gaps": [],
    }


# ---------------------------------------------------------------------------
# Poisson outputs.
# ---------------------------------------------------------------------------


def poisson_pmf(k: int, lam: float) -> float:
    return math.exp(-lam) * lam**k / math.factorial(k)


def goals_distribution(lam: float) -> dict[str, float]:
    """P(0), P(1), P(2+) from one latent Poisson rate."""

    p0 = math.exp(-lam)
    p1 = lam * math.exp(-lam)
    return {"p_goals_0": p0, "p_goals_1": p1, "p_goals_2_plus": max(0.0, 1.0 - p0 - p1)}


def lambda_for(team_id: int, opponent_id: int, venue: str, params: Mapping[str, Any]) -> float:
    """Expected goals for ``team_id`` from the fitted parameters."""

    attack = params["attack"]
    defence = params["defence"]
    exponent = float(params["league_log_baseline"]) + float(attack.get(int(team_id), 0.0)) + float(
        defence.get(int(opponent_id), 0.0)
    )
    if venue == VENUE_HOME:
        exponent += float(params["home_advantage"])
    return math.exp(exponent)


# ---------------------------------------------------------------------------
# Projection build.
# ---------------------------------------------------------------------------


def build_team_fixture_projections(
    conn: sqlite3.Connection,
    planning_event: int,
    cutoff: str,
    config: TeamStrengthConfig | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """One TeamFixtureProjection per (fixture, side) for the planning event."""

    config = config or TeamStrengthConfig()
    params = fit_team_strength(conn, planning_event, cutoff, config)
    generated_at = utc_now()
    known_teams = {int(row["id"]) for row in conn.execute("SELECT id FROM teams").fetchall()}
    fixtures_by_team = analytics.event_fixture_map(conn, int(planning_event))
    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for fixture in sorted(
        {fx["id"]: fx for fx in (f for team in fixtures_by_team.values() for f in team)}.values(),
        key=lambda fx: int(fx["id"]),
    ):
        fixture_id = int(fixture["id"])
        home, away = int(fixture["team_h"]), int(fixture["team_a"])
        for team_id, opponent_id, venue in ((home, away, VENUE_HOME), (away, home, VENUE_AWAY)):
            rows.append(
                _project_side(
                    conn, fixture_id, int(fixture["event"]), team_id, opponent_id, venue,
                    params, config, cutoff, generated_at, known_teams,
                )
            )
            seen.add((fixture_id, team_id))
    meta = {
        "model_version": TEAM_MODEL_VERSION,
        "config_hash": config.config_hash(),
        "params": params,
        "generated_at": generated_at,
        "fixture_team_pairs": len(seen),
    }
    return rows, meta


def _project_side(
    conn: sqlite3.Connection,
    fixture_id: int,
    event: int,
    team_id: int,
    opponent_id: int,
    venue: str,
    params: Mapping[str, Any],
    config: TeamStrengthConfig,
    cutoff: str,
    generated_at: str,
    known_teams: set[int],
) -> dict[str, Any]:
    lam_for = lambda_for(team_id, opponent_id, venue, params)
    lam_against = lambda_for(opponent_id, team_id, VENUE_AWAY if venue == VENUE_HOME else VENUE_HOME, params)
    dist_for = goals_distribution(lam_for)
    dist_against = goals_distribution(lam_against)

    flags: list[str] = [NO_HISTORICAL_TEAM_PRIOR]
    matches = int(params["team_match_counts"].get(int(team_id), 0))
    if matches < config.low_evidence_matches:
        flags.append("LOW_CURRENT_EVIDENCE")
    weight_total = float(params["team_weight_totals"].get(int(team_id), 0.0))
    evidence_share = weight_total / (weight_total + config.attack_prior_strength)
    if evidence_share < config.high_regularisation_share:
        flags.append("HIGH_REGULARISATION_DOMINANCE")
    if team_id not in known_teams or opponent_id not in known_teams or team_id == opponent_id:
        flags.append("CONTRADICTORY_TEAM_MAPPING")
    recon = float(params["team_reconciliation_max"].get(int(team_id), 0.0))
    if recon > config.xg_xgc_tolerance:
        flags.append("XG_XGC_RECONCILIATION_DISCREPANCY")
    if float(params.get("home_match_weight", 0.0)) < config.home_advantage_low_evidence_matches:
        flags.append("HOME_ADVANTAGE_WEAK_EVIDENCE")

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
        "prior_strength_used": config.attack_prior_strength,
        "current_evidence_weight": round(weight_total, 6),
        "current_evidence_share": round(evidence_share, 6),
        "model_version": TEAM_MODEL_VERSION,
        "generated_at": generated_at,
        "input_cutoff": cutoff,
        "risk_flags": sorted(set(flags)),
        "provenance": {
            "xg_source": "sum_of_side_player_expected_goals_by_fixture",
            "side_attribution": "player_gameweeks.was_home",
            "reconciliation_source": "max_side_player_expected_goals_conceded",
            "team_matches_used": matches,
            "reconciliation_max_abs_diff": recon,
            "reconciliation_status": "DISCREPANCY" if recon > config.xg_xgc_tolerance else "OK",
            "prior_source": "league_average",
        },
        "data_gaps": (
            ["team-specific historical prior unavailable; league-average prior used"]
            if NO_HISTORICAL_TEAM_PRIOR in flags
            else []
        ),
    }


def build_naive_team_projections(
    conn: sqlite3.Connection,
    planning_event: int,
    cutoff: str,
    config: TeamStrengthConfig | None = None,
) -> list[dict[str, Any]]:
    """Deliberately simple comparator: league-average lambda by venue.

    No team-specific signal at all: every home side is predicted the league's
    recency-weighted mean home xG and every away side the mean away xG.  Frozen
    in its own run so future calibration can answer whether Team Model v1 beats
    it.
    """

    config = config or TeamStrengthConfig()
    rows = attach_recency_weights(team_match_rows(conn, planning_event, cutoff), config.current_match_half_life)
    home_weight = sum(row["weight"] for row in rows if row["venue"] == VENUE_HOME)
    away_weight = sum(row["weight"] for row in rows if row["venue"] == VENUE_AWAY)
    lam_home = (
        sum(row["weight"] * float(row["xg_for"]) for row in rows if row["venue"] == VENUE_HOME) / home_weight
        if home_weight
        else None
    )
    lam_away = (
        sum(row["weight"] * float(row["xg_for"]) for row in rows if row["venue"] == VENUE_AWAY) / away_weight
        if away_weight
        else None
    )
    generated_at = utc_now()
    fixtures_by_team = analytics.event_fixture_map(conn, int(planning_event))
    out: list[dict[str, Any]] = []
    fixtures = sorted(
        {fx["id"]: fx for fx in (f for team in fixtures_by_team.values() for f in team)}.values(),
        key=lambda fx: int(fx["id"]),
    )
    for fixture in fixtures:
        for team_id, opponent_id, venue in (
            (int(fixture["team_h"]), int(fixture["team_a"]), VENUE_HOME),
            (int(fixture["team_a"]), int(fixture["team_h"]), VENUE_AWAY),
        ):
            lam_for = lam_home if venue == VENUE_HOME else lam_away
            lam_against = lam_away if venue == VENUE_HOME else lam_home
            data_gaps = []
            if lam_for is None or lam_against is None:
                data_gaps.append("no completed league fixtures before the cutoff; baseline lambda unavailable")
            dist = goals_distribution(lam_for) if lam_for is not None else {}
            against = goals_distribution(lam_against) if lam_against is not None else {}
            out.append(
                {
                    "fixture_id": int(fixture["id"]),
                    "event": int(fixture["event"]),
                    "team_id": team_id,
                    "opponent_id": opponent_id,
                    "venue": venue,
                    "expected_goals_for": round(lam_for, 6) if lam_for is not None else None,
                    "expected_goals_against": round(lam_against, 6) if lam_against is not None else None,
                    "p_goals_0": round(dist.get("p_goals_0"), 6) if dist else None,
                    "p_goals_1": round(dist.get("p_goals_1"), 6) if dist else None,
                    "p_goals_2_plus": round(dist.get("p_goals_2_plus"), 6) if dist else None,
                    "p_clean_sheet": round(against.get("p_goals_0"), 6) if against else None,
                    "attack_rating": None,
                    "opponent_defence_rating": None,
                    "league_baseline": None,
                    "home_advantage": None,
                    "prior_strength_used": None,
                    "current_evidence_weight": None,
                    "model_version": TEAM_BASELINE_MODEL_VERSION,
                    "generated_at": generated_at,
                    "input_cutoff": cutoff,
                    "risk_flags": ["NO_HISTORICAL_TEAM_PRIOR", "LOW_CURRENT_EVIDENCE"],
                    "provenance": {
                        "baseline_kind": "LEAGUE_AVERAGE_VENUE",
                        "formula": "recency-weighted league mean xG by venue; no team-specific signal",
                    },
                    "data_gaps": data_gaps,
                }
            )
    return out


# ---------------------------------------------------------------------------
# Analytics readiness gate for the team model.
# ---------------------------------------------------------------------------

_PROBABILITY_FIELDS = ("p_goals_0", "p_goals_1", "p_goals_2_plus", "p_clean_sheet")


def readiness_summary(
    context: Any,
    rows: list[dict[str, Any]],
    *,
    deadline_status: str | None = None,
    data_cutoff: str | None = None,
    deadline: str | None = None,
) -> dict[str, Any]:
    """Team-model-specific gate; independent of the setup health gate."""

    fail_reasons: list[str] = []
    warn_reasons: list[str] = []
    counts: dict[str, int] = {}
    if context is not None:
        if (getattr(context, "health", None) or {}).get("status") == "FAIL":
            fail_reasons.append("PLANNING_CONTEXT_FAILED: " + "; ".join((context.health or {}).get("fail_reasons") or []))
    if data_cutoff is not None and deadline is not None and deadline_status != "LATE_FREEZE":
        if parse_utc(data_cutoff) > parse_utc(deadline):
            fail_reasons.append("MODEL_INPUTS_AFTER_DEADLINE: a pre-deadline freeze cutoff may not exceed the deadline")
    if not rows:
        fail_reasons.append("NO_VALID_FIXTURE: the planning event has no projectable fixture")
    flag_counts: dict[str, int] = {}
    for row in rows:
        for flag in row.get("risk_flags", []):
            flag_counts[flag] = flag_counts.get(flag, 0) + 1
        lam_for = row.get("expected_goals_for")
        lam_against = row.get("expected_goals_against")
        for lam in (lam_for, lam_against):
            if lam is None or not math.isfinite(float(lam)) or float(lam) <= 0.0:
                fail_reasons.append(f"INVALID_LAMBDA: fixture {row.get('fixture_id')} team {row.get('team_id')}")
        if row.get("team_id") == row.get("opponent_id"):
            fail_reasons.append(f"CONTRADICTORY_TEAM_MAPPING: fixture {row.get('fixture_id')}")
        probs = [row.get(field) for field in _PROBABILITY_FIELDS]
        if any(p is not None and (p < -1e-9 or p > 1.0 + 1e-9) for p in probs):
            fail_reasons.append(f"IMPOSSIBLE_PROBABILITY_MASS: fixture {row.get('fixture_id')} team {row.get('team_id')}")
        elif all(p is not None for p in probs[:3]):
            total = row["p_goals_0"] + row["p_goals_1"] + row["p_goals_2_plus"]
            # Triple is stored at 6 dp, so allow the accumulated rounding slack.
            if abs(total - 1.0) > 1e-5:
                fail_reasons.append(f"IMPOSSIBLE_PROBABILITY_MASS: fixture {row.get('fixture_id')} team {row.get('team_id')}")
    counts.update({f"rows_{flag.lower()}": count for flag, count in sorted(flag_counts.items())})
    for flag in (
        "LOW_CURRENT_EVIDENCE",
        "NO_TEAM_SPECIFIC_HISTORICAL_PRIOR",
        "XG_XGC_RECONCILIATION_DISCREPANCY",
        "HIGH_REGULARISATION_DOMINANCE",
        "HOME_ADVANTAGE_WEAK_EVIDENCE",
    ):
        if flag_counts.get(flag):
            warn_reasons.append(f"{flag}: {flag_counts[flag]} team-fixture projections affected")
    status = "FAIL" if fail_reasons else ("WARN" if warn_reasons else "PASS")
    return {
        "status": status,
        "fail_reasons": sorted(set(fail_reasons)),
        "warn_reasons": warn_reasons,
        "counts": counts,
    }
