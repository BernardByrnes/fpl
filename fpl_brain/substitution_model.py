"""Substitution-coherent Minutes (minutes_v1.4.0).

The accepted Minutes models estimated `P(60+|start)`, `P(80+|start)`,
`E[min|start]`, `P(cameo)`, `E[min|cameo]` **independently**.  Those marginals are
not jointly attainable by one physical substitution process (a starter who is not
withdrawn must play 90 minutes, so `P(60+|start)` is forced by how often starters
are withdrawn and when).  This module replaces that layer with **one coherent team
substitution process** that is the common parent of every conditional quantity --
and the Monte Carlo simulator samples from the same frozen profile, so the two
describe one football process.

Design
------
* The accepted availability/start model and the positional coherence
  (Σ GK P(start) = 1, Σ outfield P(start) = 10, Σ team P(start) = 11) are
  **preserved unchanged**; only the conditional-minutes/cameo layer is replaced.
* A league-pooled substitution prior is estimated from stored completed
  player-fixture evidence, using `entry_minute = 90 − minutes` for non-starters
  and starter under-90 rows **excluding red-card dismissals** (a dismissal is not
  a substitution).
* Per team-fixture a `TeamSubstitutionProfile` gives the substitution-count
  distribution and the expected event mass per time band.  Exactly `S_b` exit
  mass is allocated among starters and exactly `S_b` entry mass among
  non-starters, so exits and entries reconcile band by band and
  `Σ E[minutes] = 990` holds **by construction** (no separate cameo intercept).
* Every conditional quantity is then *derived* from that event mass.
"""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass, fields
from typing import Any, Iterable, Mapping, Sequence

from . import analytics

#: Bumped from ``minutes_v1.4.0`` by PE-1: this family's historical training
#: semantics changed.  Stale pre-round placeholders are no longer read as
#: zero-substitution team-fixtures, which corrected the fitted substitution rate.
#: A run under the new semantics must not present itself as the old version.
MINUTES_SUBSTITUTION_MODEL_VERSION = "minutes_v1.5.0"

GK_POSITION_ID = 1
MAX_ORDINARY_SUBSTITUTIONS = 5


@dataclass(frozen=True)
class SubstitutionConfig:
    """Every tunable in one versioned structure; no magic numbers in code."""

    # Time bands (lo, hi) inclusive over the substitution minute, with the
    # empirical representative event minute used for derived minutes.
    time_bands: tuple = ((0, 29), (30, 44), (45, 59), (60, 74), (75, 89))
    band_representative_minutes: tuple = (15.0, 37.0, 52.0, 67.0, 82.0)
    # Strong shrinkage of thin team evidence toward the league pool.
    team_evidence_prior_strength: float = 12.0
    # Bounded relative exit propensity: lower P(80+|start) evidence means a
    # higher chance of being withdrawn.
    exit_propensity_floor: float = 0.02
    exit_propensity_bounds: tuple = (0.02, 3.0)
    # Goalkeepers use a separate treatment; zero unless evidence exists.
    gk_substitution_prior: float = 0.0
    # Numerical tolerances for the allocation identities.
    mass_tolerance: float = 1e-9

    def config_hash(self) -> str:
        values = {item.name: getattr(self, item.name) for item in fields(self)}
        return analytics.canonical_hash({"model": MINUTES_SUBSTITUTION_MODEL_VERSION, **values})

    def band_for_minute(self, minute: float) -> int | None:
        for index, (low, high) in enumerate(self.time_bands):
            if low <= minute <= high:
                return index
        return None


# ---------------------------------------------------------------------------
# Evidence audit.
# ---------------------------------------------------------------------------


def audit_substitution_evidence(
    conn: sqlite3.Connection, planning_event: int, cutoff: str
) -> dict[str, Any]:
    """Reconstruct substitution evidence from stored completed facts.

    Only completed team-fixtures before the cutoff are used.  A starter leaving
    before 90 minutes with a red card is a **dismissal, not a substitution**, and
    is excluded (and counted) rather than treated as an exit.
    """

    from . import historical_observations as historical

    # The observation universe comes from the CANONICAL boundary, so a stale
    # pre-round placeholder can never be read as a real zero-substitution fixture.
    # The `minutes IS NOT NULL` filter that appeared to guard this was not history
    # authority at all: placeholders carry minutes=0, not NULL, so it never
    # excluded them.
    rows = conn.execute(
        """SELECT pg.player_id, pg.fixture_id, pg.was_home, pg.minutes, pg.starts, pg.red_cards,
                  f.team_h, f.team_a, p.element_type
             FROM player_gameweeks pg
             JOIN fixtures f ON f.id = pg.fixture_id
             JOIN players p ON p.id = pg.player_id
            WHERE {boundary}""".format(boundary=historical.OBSERVATION_SQL_CLAUSES),
        historical.boundary_params(cutoff, planning_event=int(planning_event)),
    ).fetchall()

    sides: dict[tuple[int, int], list[Any]] = {}
    for row in rows:
        team = int(row["team_h"]) if int(row["was_home"]) == 1 else int(row["team_a"])
        sides.setdefault((int(row["fixture_id"]), team), []).append(row)

    k_counts = [0] * (MAX_ORDINARY_SUBSTITUTIONS + 1)
    entry_minutes: list[float] = []
    exit_minutes: list[float] = []
    gk_substitutions = 0
    red_card_exits = 0
    stoppage_time_substitutions = 0
    starters_seen = 0
    for key in sorted(sides):
        plist = sides[key]
        starters = [r for r in plist if int(r["starts"] or 0) == 1]
        non_starters = [r for r in plist if int(r["starts"] or 0) != 1 and (r["minutes"] or 0) > 0]
        starters_seen += len(starters)
        ordinary = 0
        for row in non_starters:
            if int(row["element_type"]) == GK_POSITION_ID:
                gk_substitutions += 1
                continue
            ordinary += 1
            entry_minutes.append(90.0 - float(row["minutes"]))
            if float(row["minutes"]) <= 1.0:
                stoppage_time_substitutions += 1
        k_counts[min(ordinary, MAX_ORDINARY_SUBSTITUTIONS)] += 1
        for row in starters:
            if float(row["minutes"]) >= 90.0:
                continue
            if int(row["red_cards"] or 0) > 0:
                red_card_exits += 1
                continue
            if int(row["element_type"]) != GK_POSITION_ID:
                exit_minutes.append(float(row["minutes"]))

    team_fixtures = len(sides)
    return {
        "team_fixtures": team_fixtures,
        "starters_seen": starters_seen,
        "k_counts": k_counts,
        "k_probabilities": [count / team_fixtures for count in k_counts] if team_fixtures else None,
        "entry_minutes": entry_minutes,
        "exit_minutes": exit_minutes,
        "gk_substitutions": gk_substitutions,
        "red_card_exits": red_card_exits,
        "stoppage_time_substitutions": stoppage_time_substitutions,
        "expected_substitutions": (
            sum(index * count for index, count in enumerate(k_counts)) / team_fixtures if team_fixtures else None
        ),
        "evidence_source": "completed player_gameweeks (fixtures.finished=1, event<planning_event, kickoff<=cutoff)",
    }


def _band_probabilities(minutes: Sequence[float], config: SubstitutionConfig) -> list[float]:
    counts = [0.0] * len(config.time_bands)
    for minute in minutes:
        index = config.band_for_minute(float(minute))
        if index is not None:
            counts[index] += 1.0
    total = sum(counts)
    if total <= 0:
        return [0.0] * len(config.time_bands)
    return [value / total for value in counts]


# ---------------------------------------------------------------------------
# Allocation.
# ---------------------------------------------------------------------------


def allocate_capped(
    mass: float, weights: Sequence[float], capacities: Sequence[float], tolerance: float
) -> tuple[list[float], float]:
    """Allocate ``mass`` in proportion to ``weights`` under per-player caps.

    Deterministic iterative proportional fitting: scale, cap, redistribute the
    shortfall among the uncapped.  Returns ``(allocation, residual)``; the
    residual is non-zero only when the request is infeasible.
    """

    n = len(weights)
    allocation = [0.0] * n
    active = [i for i in range(n) if capacities[i] > 0.0 and weights[i] > 0.0]
    remaining = float(mass)
    for _ in range(64):
        if remaining <= tolerance or not active:
            break
        weight_total = sum(weights[i] for i in active)
        if weight_total <= 0:
            break
        scale = remaining / weight_total
        for i in active:
            allocation[i] = min(capacities[i], allocation[i] + weights[i] * scale)
        used = sum(allocation[i] for i in active)
        remaining = float(mass) - sum(allocation)
        if abs(remaining) <= tolerance:
            break
        active = [i for i in active if allocation[i] < capacities[i] - tolerance]
    return allocation, float(mass) - sum(allocation)


# ---------------------------------------------------------------------------
# Team substitution profile.
# ---------------------------------------------------------------------------


def build_team_substitution_profile(
    side_players,
    evidence,
    config,
    *,
    fixture_id,
    team_id,
):
    """One side's substitution process: count distribution and band event mass.

    Exit mass and entry mass are allocated over the **same** full side: a player
    can start in some worlds and come on in others, so the constraints are
    ``SUM_b exit_i,b <= P(start_i)`` and ``SUM_b entry_i,b <= P(available_i) -
    P(start_i)`` rather than a partition into disjoint groups.  ``side_players``
    entries carry ``p_start``, ``p_available``, ``p80_given_start`` (relative
    exit evidence) and ``cameo_propensity`` (relative entry evidence).
    """

    k_probabilities = evidence.get("k_probabilities")
    if not k_probabilities:
        raise ValueError(f"fixture {fixture_id} team {team_id}: no substitution evidence available")
    expected_substitutions = sum(index * probability for index, probability in enumerate(k_probabilities))

    band_probabilities = _band_probabilities(evidence.get("entry_minutes") or [], config)
    band_mass = [expected_substitutions * probability for probability in band_probabilities]
    total_band_mass = sum(band_mass)
    if total_band_mass > MAX_ORDINARY_SUBSTITUTIONS + config.mass_tolerance:
        raise ValueError(
            f"fixture {fixture_id} team {team_id}: expected substitution mass {total_band_mass:.4f} exceeds "
            f"the limit {MAX_ORDINARY_SUBSTITUTIONS}"
        )

    exit_capacities = [float(p.get("p_start") or 0.0) for p in side_players]
    exit_weights = [
        min(
            config.exit_propensity_bounds[1],
            max(config.exit_propensity_bounds[0], 1.0 - float(p.get("p80_given_start") or 0.0)),
        )
        for p in side_players
    ]
    entry_capacities = [
        max(0.0, float(p.get("p_available") or 0.0) - float(p.get("p_start") or 0.0)) for p in side_players
    ]
    entry_weights = [max(0.0, float(p.get("cameo_propensity") or 0.0)) for p in side_players]

    exit_mass = [[0.0] * len(config.time_bands) for _ in side_players]
    entry_mass = [[0.0] * len(config.time_bands) for _ in side_players]
    residuals = []
    for band_index, mass in enumerate(band_mass):
        if mass <= 0.0:
            continue
        # Capacity is the REMAINING budget: a player may only be withdrawn once
        # in total, so the per-band cap must net off earlier bands.
        exit_remaining = [
            max(0.0, exit_capacities[i] - sum(exit_mass[i])) for i in range(len(side_players))
        ]
        entry_remaining = [
            max(0.0, entry_capacities[i] - sum(entry_mass[i])) for i in range(len(side_players))
        ]
        exits, exit_residual = allocate_capped(mass, exit_weights, exit_remaining, config.mass_tolerance)
        entries, entry_residual = allocate_capped(mass, entry_weights, entry_remaining, config.mass_tolerance)
        residuals.extend([exit_residual, entry_residual])
        for index, value in enumerate(exits):
            exit_mass[index][band_index] = value
        for index, value in enumerate(entries):
            entry_mass[index][band_index] = value

    exit_totals = [sum(row) for row in exit_mass]
    entry_totals = [sum(row) for row in entry_mass]
    flags = []
    if abs(sum(exit_totals) - total_band_mass) > 1e-6 or abs(sum(entry_totals) - total_band_mass) > 1e-6:
        flags.append("SUBSTITUTION_MASS_INFEASIBLE")
    if evidence.get("gk_substitutions", 0) == 0:
        flags.append("GK_SUBSTITUTION_RARE_EVENT_EXCLUDED")
    if (evidence.get("team_fixtures") or 0) < 30:
        flags.append("SUBSTITUTION_PRIOR_THIN")

    players = []
    for index, player in enumerate(side_players):
        players.append(
            {
                "index": index,
                "p_start": round(float(player.get("p_start") or 0.0), 9),
                "p_available": round(float(player.get("p_available") or 0.0), 9),
                "exit_mass_by_band": [round(v, 9) for v in exit_mass[index]],
                "exit_total": round(exit_totals[index], 9),
                "entry_mass_by_band": [round(v, 9) for v in entry_mass[index]],
                "entry_total": round(entry_totals[index], 9),
            }
        )

    return {
        "fixture_id": int(fixture_id),
        "team_id": int(team_id),
        "expected_substitutions": round(expected_substitutions, 6),
        "p_sub_count": [round(p, 6) for p in k_probabilities],
        "event_time_bands": [list(band) for band in config.time_bands],
        "event_time_masses": [round(m, 6) for m in band_mass],
        "gk_event_mass": float(config.gk_substitution_prior),
        "evidence_matches": int(evidence.get("team_fixtures") or 0),
        "evidence_starters": int(evidence.get("starters_seen") or 0),
        "prior_source": evidence.get("evidence_source"),
        "expected_exit_mass": round(sum(exit_totals), 6),
        "expected_entry_mass": round(sum(entry_totals), 6),
        "max_abs_constraint_residual": max((abs(r) for r in residuals), default=0.0),
        "risk_flags": flags,
        "players": players,
    }


def derive_from_profile(side_players, profile, config):
    """Derive every conditional quantity from the shared substitution process."""

    representatives = list(config.band_representative_minutes)
    derived = []
    for row in profile["players"]:
        index = row["index"]
        p_start = float(row["p_start"])
        exit_bands = row["exit_mass_by_band"]
        entry_bands = row["entry_mass_by_band"]
        exit_total = float(row["exit_total"])
        entry_total = float(row["entry_total"])

        if p_start > 0.0:
            before_60 = sum(m for m, t in zip(exit_bands, representatives) if t < 60.0)
            before_80 = sum(m for m, t in zip(exit_bands, representatives) if t < 80.0)
            no_exit = max(0.0, p_start - exit_total)
            expected_start = (
                no_exit * 90.0 + sum(m * t for m, t in zip(exit_bands, representatives))
            ) / p_start
            p60_start = max(0.0, min(1.0, 1.0 - before_60 / p_start))
            p80_start = max(0.0, min(1.0, 1.0 - before_80 / p_start))
        else:
            expected_start = float(side_players[index].get("expected_minutes_if_start") or 0.0)
            p60_start = 0.0
            p80_start = 0.0

        if entry_total > 0.0:
            expected_cameo = sum(m * (90.0 - t) for m, t in zip(entry_bands, representatives)) / entry_total
            long_enough = sum(m for m, t in zip(entry_bands, representatives) if (90.0 - t) >= 60.0)
            p60_cameo = long_enough / entry_total
        else:
            expected_cameo = 0.0
            p60_cameo = 0.0

        derived.append(
            {
                "index": index,
                "p_start": p_start,
                "p_cameo": entry_total,
                "p_60_given_start": p60_start,
                "p_80_given_start": p80_start,
                "expected_minutes_if_start": expected_start,
                "p_60_given_cameo": max(0.0, min(1.0, p60_cameo)),
                "expected_minutes_if_cameo": expected_cameo,
                "exit_total": exit_total,
                "entry_total": entry_total,
            }
        )
    return derived


def derive_team_identities(derived: Sequence[Mapping[str, Any]], config: SubstitutionConfig) -> dict[str, float]:
    """Team-level identity check on the derived quantities."""

    p_start_total = sum(float(d["p_start"]) for d in derived)
    p_cameo_total = sum(float(d["p_cameo"]) for d in derived)
    expected_minutes = sum(
        float(d["p_start"]) * float(d["expected_minutes_if_start"])
        + float(d["p_cameo"]) * float(d["expected_minutes_if_cameo"])
        for d in derived
    )
    exit_mass = sum(float(d.get("exit_total") or 0.0) for d in derived)
    entry_mass = sum(float(d.get("entry_total") or 0.0) for d in derived)
    return {
        "p_start_total": p_start_total,
        "p_cameo_total": p_cameo_total,
        "expected_minutes": expected_minutes,
        "exit_mass": exit_mass,
        "entry_mass": entry_mass,
    }


# ---------------------------------------------------------------------------
# Minutes v1.4 builder.
# ---------------------------------------------------------------------------


def build_minutes_predictions_substitution_coherent(
    conn: sqlite3.Connection,
    planning_event: int,
    cutoff: str,
    model_config: Any | None = None,
    coherence_config: Any | None = None,
    sub_config: SubstitutionConfig | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Positional-coherent starts + a substitution-coherent conditional layer.

    Returns ``(rows, profiles)``.  The accepted start model and positional
    coherence are preserved; only the conditional-minutes/cameo layer is derived
    from the shared substitution process.  The v1.3 conditional values are kept
    on every row (``*_v13``) so prospective calibration can compare versions.
    """

    from . import minutes_coherence, minutes_model  # local import avoids a cycle

    model_config = model_config or minutes_model.MinutesModelConfig()
    coherence_config = coherence_config or minutes_coherence.MinutesCoherenceConfig()
    sub_config = sub_config or SubstitutionConfig()

    base_rows, _base_records = minutes_coherence.build_minutes_predictions_coherent(
        conn, planning_event, cutoff, model_config, coherence_config, positional=True
    )
    evidence = audit_substitution_evidence(conn, planning_event, cutoff)
    if not evidence.get("k_probabilities"):
        raise ValueError("no substitution evidence available before the cutoff")

    player_team = {
        int(row["id"]): int(row["team_id"])
        for row in conn.execute("SELECT id, team_id FROM players WHERE is_active=1 AND team_id IS NOT NULL")
    }
    sides: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in base_rows:
        sides.setdefault((int(row["fixture_id"]), player_team[int(row["player_id"])]), []).append(row)

    rows_out: list[dict[str, Any]] = []
    profiles_out: list[dict[str, Any]] = []
    for key in sorted(sides):
        payloads = sorted(sides[key], key=lambda r: int(r["player_id"]))
        side_players = [
            {
                "p_start": float(p.get("p_start") or 0.0),
                "p_available": float(p.get("p_available") or 0.0),
                "p80_given_start": float(p.get("p_80_given_start") or 0.0),
                "cameo_propensity": (
                    float(p.get("p_cameo") or 0.0) / max(1e-9, 1.0 - float(p.get("p_start") or 0.0))
                ),
                "expected_minutes_if_start": float(p.get("expected_minutes_if_start") or 0.0),
            }
            for p in payloads
        ]
        profile = build_team_substitution_profile(
            side_players, evidence, sub_config, fixture_id=key[0], team_id=key[1]
        )
        derived = derive_from_profile(side_players, profile, sub_config)
        identities = derive_team_identities(derived, sub_config)
        profile["identities"] = {name: round(value, 9) for name, value in identities.items()}
        profile["status"] = (
            "COHERENT"
            if abs(identities["p_start_total"] - 11.0) < 1e-4
            and abs(identities["expected_minutes"] - 990.0) < 1e-2
            and abs(identities["exit_mass"] - identities["entry_mass"]) < 1e-6
            else "INCOHERENT"
        )
        profiles_out.append(profile)

        side_profile_public = {
            "expected_substitutions": profile["expected_substitutions"],
            "p_sub_count": profile["p_sub_count"],
            "event_time_bands": profile["event_time_bands"],
            "event_time_masses": profile["event_time_masses"],
            "gk_event_mass": profile["gk_event_mass"],
            "risk_flags": profile["risk_flags"],
        }
        for derived_row in derived:
            position = derived_row["index"]
            payload = dict(payloads[position])
            payload.update(
                {
                    "p_60_given_start_v13": payload.get("p_60_given_start"),
                    "p_80_given_start_v13": payload.get("p_80_given_start"),
                    "expected_minutes_if_start_v13": payload.get("expected_minutes_if_start"),
                    "p_60_given_cameo_v13": payload.get("p_60_given_cameo"),
                    "expected_minutes_if_cameo_v13": payload.get("expected_minutes_if_cameo"),
                    "p_cameo_v13": payload.get("p_cameo"),
                    "p_60_plus_v13": payload.get("p_60_plus"),
                    "expected_minutes_v13": payload.get("expected_minutes"),
                    # v1.4 substitution-coherent values
                    "p_60_given_start": round(float(derived_row["p_60_given_start"]), 6),
                    "p_80_given_start": round(float(derived_row["p_80_given_start"]), 6),
                    "expected_minutes_if_start": round(float(derived_row["expected_minutes_if_start"]), 6),
                    "p_60_given_cameo": round(float(derived_row["p_60_given_cameo"]), 6),
                    "expected_minutes_if_cameo": round(float(derived_row["expected_minutes_if_cameo"]), 6),
                    "p_cameo": round(float(derived_row["p_cameo"]), 6),
                    "team_substitution_profile": side_profile_public,
                    "model_version": MINUTES_SUBSTITUTION_MODEL_VERSION,
                }
            )
            p_start = float(payload.get("p_start") or 0.0)
            p_cameo = float(payload.get("p_cameo") or 0.0)
            p_zero = max(0.0, 1.0 - p_start - p_cameo)
            p60 = min(1.0, p_start * float(derived_row["p_60_given_start"]) + p_cameo * float(derived_row["p_60_given_cameo"]))
            p80 = min(p_start * float(derived_row["p_80_given_start"]), p60)
            payload["p_zero"] = round(p_zero, 6)
            payload["p_60_plus"] = round(p60, 6)
            payload["p_80_plus"] = round(p80, 6)
            payload["p_1_59"] = round(max(0.0, 1.0 - p_zero - p60), 6)
            payload["expected_minutes"] = round(
                p_start * float(derived_row["expected_minutes_if_start"])
                + p_cameo * float(derived_row["expected_minutes_if_cameo"]),
                6,
            )
            rows_out.append(payload)
    return rows_out, profiles_out
