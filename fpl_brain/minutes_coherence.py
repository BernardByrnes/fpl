"""Team-coherence layer for Minutes v1 (minutes_v1.2.0 / v1.3.0).

The raw Minutes v1.1.0 model predicts every player independently, so the summed
marginals do not conserve a football side's lineup mass: on GW4 the raw model
summed to ~13.7 expected starters and ~1,224 expected minutes per side against
the true 11 / 990.  This module is a **deterministic post-processing layer**
over those raw marginals — it does not change the raw model, does not normalise
player skill/rate quantities, and does not simulate anything.

Two versions:

* **minutes_v1.2.0** — one shared team-level shift to the conditional start
  log-odds (Σ P(start) = 11) and one to the cameo probability (Σ E[minutes] =
  990).
* **minutes_v1.3.0** — the same principle applied **per position group**, so a
  side also satisfies Σ P(start) = 1 for goalkeepers and 10 for outfielders,
  which the team-wide solver does not guarantee (GW4 audit: GK mass 0.72–1.71).

Both preserve relative player evidence (a single monotone shift per group),
keep hard-out players at zero, retain the raw independent marginals in every
payload (``*_raw_independent``), and raise rather than invent minutes when a
side's identities are impossible.
"""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass, fields
from typing import Any, Sequence

from . import analytics, repositories as repo
from .minutes_model import (
    MINUTES_COHERENT_MODEL_VERSION,
    MinutesModelConfig,
    _clamp,
    _round,
    league_pools,
    project_player_fixture,
)

# Positional refinement challenger (Phase 5): GK = 1, outfield = 10.
MINUTES_POSITIONAL_MODEL_VERSION = "minutes_v1.3.0"

GK_POSITION_ID = 1


class TeamCoherenceError(ValueError):
    """A team-fixture whose lineup identity cannot be satisfied."""


@dataclass(frozen=True)
class MinutesCoherenceConfig:
    """Tunables for the deterministic team-coherence solver."""

    target_starters: float = 11.0
    target_goalkeepers: float = 1.0
    target_outfield: float = 10.0
    target_minutes: float = 990.0
    # Solver convergence targets (on unrounded quantities).
    start_tolerance: float = 1e-9
    minutes_tolerance: float = 1e-6
    # Verification tolerances, aware of the 6-dp per-row storage rounding.
    verify_start_tolerance: float = 1e-4
    verify_minutes_tolerance: float = 1e-2
    logit_epsilon: float = 1e-9
    solver_iterations: int = 200
    solver_bound: float = 60.0

    def config_hash(self) -> str:
        values = {item.name: getattr(self, item.name) for item in fields(self)}
        return analytics.canonical_hash({"model": MINUTES_COHERENT_MODEL_VERSION, **values})


def _logit(value: float, epsilon: float) -> float:
    clamped = min(max(float(value), epsilon), 1.0 - epsilon)
    return math.log(clamped / (1.0 - clamped))


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def _solve_shift(lo: float, hi: float, fn, target: float, iterations: int, tolerance: float) -> tuple[float, float]:
    """Deterministic bisection for a monotone increasing ``fn``."""

    for _ in range(int(iterations)):
        mid = (lo + hi) / 2.0
        residual = fn(mid) - target
        if abs(residual) <= tolerance:
            return mid, residual
        if residual < 0:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-12:
            break
    mid = (lo + hi) / 2.0
    return mid, fn(mid) - target


def _solve_group_start(
    available: Sequence[float],
    logit_sga: Sequence[float],
    indices: Sequence[int],
    target: float,
    config: MinutesCoherenceConfig,
    *,
    label: str,
    fixture_id: int,
    team_id: int,
) -> float:
    """One shared start-logit shift for a group of players."""

    avail_sum = sum(available[i] for i in indices)
    if avail_sum < target - config.start_tolerance:
        raise TeamCoherenceError(
            f"fixture {fixture_id} team {team_id} {label}: sum P(available)={avail_sum:.3f} < {target:g}"
        )
    if avail_sum <= target + config.start_tolerance:
        return config.solver_bound

    def total(shift: float) -> float:
        return sum(available[i] * _sigmoid(logit_sga[i] + shift) for i in indices)

    delta, _ = _solve_shift(
        -config.solver_bound, config.solver_bound, total, target,
        config.solver_iterations, config.start_tolerance,
    )
    return delta


def solve_team_coherence(
    payloads: list[dict[str, Any]],
    *,
    fixture_id: int,
    team_id: int,
    config: MinutesCoherenceConfig | None = None,
    positional: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply shared start shift(s) and one shared cameo shift to a side.

    ``positional=True`` solves the start shifts separately for goalkeepers
    (target 1) and outfielders (target 10) using the ``position_id`` field on
    each payload, so the side also satisfies the real XI composition.
    """

    config = config or MinutesCoherenceConfig()
    epsilon = config.logit_epsilon
    available = [float(p.get("p_available") or 0.0) for p in payloads]
    p_sga = [float(p.get("p_start_given_available") or 0.0) for p in payloads]
    cgr = [float(p.get("p_cameo_given_not_start") or 0.0) for p in payloads]
    m_start = [float(p.get("expected_minutes_if_start") or 0.0) for p in payloads]
    m_cameo = [float(p.get("expected_minutes_if_cameo") or 0.0) for p in payloads]

    raw_start_sum = sum(available[i] * p_sga[i] for i in range(len(payloads)))
    raw_minutes_sum = sum(
        available[i] * p_sga[i] * m_start[i]
        + available[i] * (1.0 - p_sga[i]) * cgr[i] * m_cameo[i]
        for i in range(len(payloads))
    )

    logit_sga = [_logit(value, epsilon) for value in p_sga]
    if positional:
        gk_indices = [i for i, p in enumerate(payloads) if int(p.get("position_id") or 0) == GK_POSITION_ID]
        out_indices = [i for i in range(len(payloads)) if i not in set(gk_indices)]
        delta_gk = _solve_group_start(
            available, logit_sga, gk_indices, config.target_goalkeepers, config,
            label="GKP", fixture_id=fixture_id, team_id=team_id,
        )
        delta_out = _solve_group_start(
            available, logit_sga, out_indices, config.target_outfield, config,
            label="outfield", fixture_id=fixture_id, team_id=team_id,
        )
        deltas = [
            delta_gk if int(p.get("position_id") or 0) == GK_POSITION_ID else delta_out for p in payloads
        ]
        start_intercept = delta_out  # headline intercept (outfield group)
        start_intercept_gk = delta_gk
    else:
        single = _solve_group_start(
            available, logit_sga, list(range(len(payloads))), config.target_starters, config,
            label="team", fixture_id=fixture_id, team_id=team_id,
        )
        deltas = [single] * len(payloads)
        start_intercept = single
        start_intercept_gk = single

    p_start = [available[i] * _sigmoid(logit_sga[i] + deltas[i]) for i in range(len(payloads))]
    for i, value in enumerate(p_start):
        if value > available[i] + 1e-9:
            raise TeamCoherenceError(f"fixture {fixture_id} team {team_id}: P(start) > P(available)")

    starter_minutes = sum(p_start[i] * m_start[i] for i in range(len(payloads)))
    required_cameo_minutes = config.target_minutes - starter_minutes
    if required_cameo_minutes < -config.minutes_tolerance:
        raise TeamCoherenceError(
            f"fixture {fixture_id} team {team_id}: starter minutes {starter_minutes:.3f} already exceed "
            f"{config.target_minutes:g}"
        )
    not_start_available = [max(0.0, available[i] - p_start[i]) for i in range(len(payloads))]
    max_cameo_minutes = sum(not_start_available[i] * m_cameo[i] for i in range(len(payloads)))
    if max_cameo_minutes < required_cameo_minutes - config.minutes_tolerance:
        raise TeamCoherenceError(
            f"fixture {fixture_id} team {team_id}: required cameo minutes {required_cameo_minutes:.3f} > "
            f"maximum achievable {max_cameo_minutes:.3f}"
        )

    logit_cgr = [_logit(value, epsilon) for value in cgr]
    if required_cameo_minutes <= config.minutes_tolerance:
        cameo_intercept = -config.solver_bound
    else:
        def cameo_total(shift: float) -> float:
            return sum(
                not_start_available[i] * _sigmoid(logit_cgr[i] + shift) * m_cameo[i]
                for i in range(len(payloads))
            )

        cameo_intercept, _ = _solve_shift(
            -config.solver_bound, config.solver_bound, cameo_total,
            required_cameo_minutes, config.solver_iterations, config.minutes_tolerance,
        )
    q_cameo = [_sigmoid(logit_cgr[i] + cameo_intercept) for i in range(len(payloads))]

    coherent: list[dict[str, Any]] = []
    unrounded_start_total = 0.0
    unrounded_minutes_total = 0.0
    for i, payload in enumerate(payloads):
        ps = min(p_start[i], available[i])
        pc = max(0.0, min(not_start_available[i] * q_cameo[i], available[i] - ps))
        pz = _clamp(1.0 - ps - pc)
        p60s = float(payload.get("p_60_given_start") or 0.0)
        p80s = float(payload.get("p_80_given_start") or 0.0)
        p60c = float(payload.get("p_60_given_cameo") or 0.0)
        p60 = _clamp(ps * p60s + pc * p60c)
        p80 = min(_clamp(ps * p80s), p60)
        p159 = _clamp(1.0 - pz - p60)
        expected = ps * m_start[i] + pc * m_cameo[i]
        unrounded_start_total += ps
        unrounded_minutes_total += expected
        row = dict(payload)
        row.update(
            {
                "p_start": _round(ps),
                "p_cameo": _round(pc),
                "p_zero": _round(pz),
                "p_1_59": _round(p159),
                "p_60_plus": _round(p60),
                "p_80_plus": _round(p80),
                "expected_minutes": _round(expected),
                "p_start_given_available": _round(min(1.0, ps / available[i]) if available[i] > 0 else 0.0),
                "p_cameo_given_not_start": _round(q_cameo[i]),
                "p_start_raw_independent": payload.get("p_start"),
                "p_cameo_raw_independent": payload.get("p_cameo"),
                "p_zero_raw_independent": payload.get("p_zero"),
                "p_1_59_raw_independent": payload.get("p_1_59"),
                "p_60_plus_raw_independent": payload.get("p_60_plus"),
                "p_80_plus_raw_independent": payload.get("p_80_plus"),
                "expected_minutes_raw_independent": payload.get("expected_minutes"),
                "model_version": (
                    MINUTES_POSITIONAL_MODEL_VERSION if positional else MINUTES_COHERENT_MODEL_VERSION
                ),
                "team_coherence": {
                    "applied": True,
                    "positional": bool(positional),
                    "fixture_id": int(fixture_id),
                    "team_id": int(team_id),
                    "start_intercept": _round(deltas[i]),
                    "cameo_intercept": _round(cameo_intercept),
                },
            }
        )
        coherent.append(row)

    adjusted_start_sum = sum(row["p_start"] for row in coherent)
    adjusted_minutes_sum = sum(row["expected_minutes"] for row in coherent)
    gk_start_sum = sum(
        row["p_start"] for row in coherent if int(row.get("position_id") or 0) == GK_POSITION_ID
    )
    record = {
        "fixture_id": int(fixture_id),
        "team_id": int(team_id),
        "players": len(payloads),
        "raw_start_sum": _round(raw_start_sum),
        "adjusted_start_sum": _round(adjusted_start_sum),
        "raw_minutes_sum": _round(raw_minutes_sum),
        "adjusted_minutes_sum": _round(adjusted_minutes_sum),
        "start_intercept": _round(start_intercept),
        "cameo_intercept": _round(cameo_intercept),
        "positional": bool(positional),
        "gk_start_intercept": _round(start_intercept_gk),
        "gk_start_sum": _round(gk_start_sum),
        # Constraint residuals are the true (unrounded) solver residuals; the
        # per-row 6-dp storage rounding adds a separate, tiny aggregate drift.
        "start_residual": _round(unrounded_start_total - config.target_starters),
        "minutes_residual": _round(unrounded_minutes_total - config.target_minutes, 6),
        "start_sum_rounded": _round(adjusted_start_sum),
        "minutes_sum_rounded": _round(adjusted_minutes_sum),
        "status": "COHERENT",
    }
    return coherent, record


def build_minutes_predictions_coherent(
    conn: sqlite3.Connection,
    planning_event: int,
    cutoff: str,
    config: MinutesModelConfig | None = None,
    coherence_config: MinutesCoherenceConfig | None = None,
    *,
    positional: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Raw v1.1 marginals plus a deterministic per-side coherence adjustment.

    ``positional=True`` additionally constrains each side to 1 GK + 10 outfield.
    """

    config = config or MinutesModelConfig()
    coherence_config = coherence_config or MinutesCoherenceConfig()
    pools = league_pools(conn, planning_event, cutoff)
    fixtures_by_team = analytics.event_fixture_map(conn, int(planning_event))
    by_side: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for player in sorted(analytics.projectable_players(conn), key=lambda row: int(row["player_id"])):
        team_id = int(player["team_id"])
        team_fixtures = fixtures_by_team.get(team_id)
        if not team_fixtures:
            continue
        player_id = int(player["player_id"])
        evidence_rows = analytics.completed_rows_as_of(conn, player_id, cutoff, int(planning_event))
        snapshot = analytics.snapshot_as_of(conn, player_id, cutoff)
        snapshot_history = analytics.snapshot_history_as_of(conn, player_id, cutoff)
        scout_notes = repo.scouting_current_rows_as_of(conn, cutoff, [player_id])
        for fixture in team_fixtures:
            payload = project_player_fixture(
                conn, pools, player, evidence_rows, fixture, snapshot, scout_notes, config, cutoff,
                include_conditionals=True, snapshot_history=snapshot_history,
            )
            payload["position_id"] = int(player.get("element_type") or 0)
            by_side.setdefault((int(fixture["id"]), team_id), []).append(payload)

    rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for (fixture_id, team_id) in sorted(by_side):
        coherent, record = solve_team_coherence(
            by_side[(fixture_id, team_id)], fixture_id=fixture_id, team_id=team_id,
            config=coherence_config, positional=positional,
        )
        rows.extend(coherent)
        records.append(record)
    return rows, records


def coherence_readiness(
    records: list[dict[str, Any]],
    coherence_config: MinutesCoherenceConfig | None = None,
) -> dict[str, Any]:
    """Verify every side satisfies the lineup identities within tolerance."""

    config = coherence_config or MinutesCoherenceConfig()
    fail_reasons: list[str] = []
    for record in records:
        if abs(record["start_residual"]) > config.verify_start_tolerance:
            fail_reasons.append(
                f"TEAM_START_SUM_VIOLATION: fixture {record['fixture_id']} team {record['team_id']} "
                f"residual {record['start_residual']}"
            )
        if abs(record["minutes_residual"]) > config.verify_minutes_tolerance:
            fail_reasons.append(
                f"TEAM_MINUTES_SUM_VIOLATION: fixture {record['fixture_id']} team {record['team_id']} "
                f"residual {record['minutes_residual']}"
            )
        if record.get("positional") and abs(record.get("gk_start_sum", 1.0) - config.target_goalkeepers) > config.verify_start_tolerance:
            fail_reasons.append(
                f"TEAM_GK_SUM_VIOLATION: fixture {record['fixture_id']} team {record['team_id']} "
                f"gk mass {record.get('gk_start_sum')}"
            )
    return {
        "status": "FAIL" if fail_reasons else "PASS",
        "fail_reasons": fail_reasons,
        "counts": {"sides": len(records)},
        "max_abs_start_residual": max((abs(r["start_residual"]) for r in records), default=0.0),
        "max_abs_minutes_residual": max((abs(r["minutes_residual"]) for r in records), default=0.0),
        "max_abs_start_rounded_drift": max(
            (abs(r.get("start_sum_rounded", 0.0) - config.target_starters) for r in records), default=0.0
        ),
        "max_abs_minutes_rounded_drift": max(
            (abs(r.get("minutes_sum_rounded", 0.0) - config.target_minutes) for r in records), default=0.0
        ),
    }
