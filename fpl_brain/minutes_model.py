"""Minutes Model v1 — hierarchical, coherent player-fixture minutes projections.

Per player-fixture (DGW-safe, blank weeks produce no projection):

    P(available)                          availability facts + official evidence
    P(start | available)                  recency-weighted shrinkage
    P(cameo | not start, available)       position-pooled
    P(0), P(1-59), P(60+), P(80+)         derived, mathematically coherent
    expected minutes (+ if start/cameo)

Strict no-lookahead: all evidence must be at or before the data cutoff, and
completed-gameweek evidence must belong to events strictly before the planning
event. No monetary input exists anywhere in this module — it is never an
expected-points model.
"""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass, field, fields
from typing import Any, Mapping, Sequence

from . import analytics
from . import historical_observations as historical, history_completeness as hc, repositories as repo
from .utils import parse_utc, utc_now

MINUTES_MODEL_VERSION = "minutes_v1.7.0"
# v1.7.0: the no-causal-history position prior now asserts the 990-minute /
# 11-starter identity (see ``structural_no_history_priors``), so an empty and
# legitimate causal window produces a coherent side instead of an impossible
# cameo budget.  This is a semantic change to the primary model's contract for
# the empty-history state; certified-cutoff outputs are unchanged because every
# position has causal observations there, so the fallback never activates.
# Team-coherence challenger (Phase 4 acceptance pass).  This is a deterministic
# post-processing layer over the raw v1.1 independent marginals: the raw model
# is preserved untouched in the same run payload (``*_raw_independent``).
MINUTES_COHERENT_MODEL_VERSION = "minutes_v1.2.0"

# INITIAL MODELLING ASSUMPTIONS for the official availability signal —
# explicitly not empirical/calibrated probabilities; every value is versioned
# in MinutesModelConfig and included in its config hash.  `s` (suspended)
# and `n` (registered out) are the ONLY hard facts; injured/doubtful are
# evidence refined by the event-specific chance when one exists.
# UPDATED in minutes_v1.1.0: these live in the config dataclass, not here.
OFFICIAL_STATUS_AVAILABILITY_DEFAULTS = {
    "a": 1.0,
    "d": 0.7,
    "i": 0.45,
    "s": 0.0,
    "u": 0.2,
    "n": 0.0,
}
HARD_UNAVAILABLE_STATUSES = ("s", "n")
_LOW_HISTORY_ROWS = 3
_INJURY_UNCERTAINTY_HIGH = 75.0

# Official scout-protocol probability bands -> bounded start deltas.
_START_BAND_DELTAS = {
    95.0: 0.10,
    85.0: 0.075,
    70.0: 0.025,
    50.0: 0.0,
    30.0: -0.025,
    15.0: -0.075,
    5.0: -0.10,
}


@dataclass(frozen=True)
class MinutesModelConfig:
    """Every tunable in one versioned structure; no magic numbers in code."""

    start_recency_half_life_matches: float = 4.0
    start_prior_strength: float = 5.0
    team_position_prior_backoff_strength: float = 6.0
    minutes_if_start_half_life_matches: float = 4.0
    minutes_if_start_prior_strength: float = 6.0
    p60_if_start_prior_strength: float = 5.0
    p80_if_start_prior_strength: float = 5.0
    cameo_rate_prior_strength: float = 8.0
    cameo_rate_prior: float = 0.5
    cameo_minutes_prior_strength: float = 10.0
    cameo_minutes_prior: float = 15.0
    cameo_p60_prior_strength: float = 20.0
    cameo_p60_prior: float = 0.04
    cameo_p80_prior_strength: float = 20.0

    midweek_rest_days_threshold: float = 4.0
    midweek_heavy_minutes_threshold: int = 75
    midweek_heavy_start_modifier: float = 0.90
    midweek_heavy_upper_minutes_modifier: float = 0.90
    midweek_rest_start_modifier: float = 1.05

    returning_start_modifier: float = 0.80
    returning_upper_minutes_modifier: float = 0.75
    returning_minutes_if_start_modifier: float = 0.85
    returning_status_values: tuple[str, ...] = ("i", "d")

    scout_p_start_delta_bound: float = 0.15
    scout_expected_minutes_blend_weight: float = 0.3
    scout_expected_minutes_blend_bound: float = 20.0

    # Official availability signal -> modelled availability multiplier.
    # INITIAL MODELLING ASSUMPTIONS (v1.1.0), never calibrated probabilities.
    official_availability_signal_defaults: dict[str, float] = field(
        default_factory=lambda: dict(OFFICIAL_STATUS_AVAILABILITY_DEFAULTS)
    )
    hard_unavailable_statuses: tuple[str, ...] = ("s", "n")

    # Previous-season player prior (v1.1.0), effective-sample-size semantics.
    prev_season_ess_cap: float = 12.0
    prev_season_ess_min_minutes: float = 450.0
    prev_season_sub_minutes_assumption: float = 18.0
    prev_season_prior_anchor_strength: float = 6.0
    prev_season_minutes_anchor_strength: float = 6.0
    prev_season_role_change_ess_discount: float = 0.5
    prev_season_returning_ess_discount: float = 0.6

    # --- role-evidence taxonomy (v1.6.0) -----------------------------------
    # Prior-club role provenance is not stored (player_season_histories has no
    # club column), so a strong prior role that has *not* reproduced at the
    # current club is treated as evidence of a role discontinuity rather than
    # as established current-team form.  The prior is downweighted, not
    # discarded: minutes-per-start (durability/position) keeps its own,
    # milder anchor strength.
    prev_season_role_discontinuity_ess_discount: float = 0.35
    role_discontinuity_min_available_observations: int = 2
    role_discontinuity_prior_start_rate: float = 0.5
    # A player with >= this many available zero-minute non-start events is reported
    # as strong current-season zero-minute non-start evidence.
    strong_zero_minute_non_start_evidence_threshold: int = 2

    def config_hash(self) -> str:
        values = {item.name: getattr(self, item.name) for item in fields(self)}
        return analytics.canonical_hash(
            {
                "model": MINUTES_MODEL_VERSION,
                **values,
                "start_band_deltas": _START_BAND_DELTAS,
            }
        )


# Regulation match length.  The coherent minutes layer already fixes a side at
# 11 starters sharing 990 player-minutes (``MinutesCoherenceConfig``), so 990/11
# is the only start-minutes anchor an uninformed side may assert.
REGULATION_MATCH_MINUTES = 90.0


def structural_no_history_priors(config: MinutesModelConfig) -> dict[str, float]:
    """The position prior used when the causal window holds NO observation.

    This is a statement of structural fact, not a fitted statistic.  A side is
    eleven players sharing ``990`` player-minutes, so the only start-minutes
    anchor consistent with that identity -- and therefore the only one that
    cannot manufacture an impossible cameo budget for an ordinary squad -- is
    that the XI plays the match.  Required cameo minutes are then exactly zero
    and any available bench satisfies them, for every squad size.

    The previous anchor of 65 minutes contradicted the coherence layer's own
    990-minute identity: it demanded ``990 - 11 x 65 = 275`` cameo minutes,
    while a normal 25-man squad supplies only ``(25 - 11) x 15 = 210``.  That
    is not an edge case; it held for every squad below 30 players, so an empty
    (and perfectly legitimate) causal window could never produce a coherent
    side.  ``p60``/``p80`` follow the same fact: a player who plays the match
    reaches both.

    Fields the config already owns are taken from it, so this cannot drift.
    """

    return {
        # Unknown role: the solver shifts this to the real XI size.
        "p_start": 0.35,
        "minutes_if_start": REGULATION_MATCH_MINUTES,
        "p60_if_start": 1.0,
        "p80_if_start": 1.0,
        "cameo_rate": float(config.cameo_rate_prior),
        "cameo_minutes": float(config.cameo_minutes_prior),
        "cameo_p60": float(config.cameo_p60_prior),
        "start_rows": 0,
    }


# ---------------------------------------------------------------------------
# Bounded scouting/manual modifier registry.
#
# Every rule maps a structured scouting note (key, band / protocol number) to
# one named bounded parameter effect. No free-form text becomes a number
# without an explicit rule; bounds clamp all effects deterministically.
# Available modifier types: START_ROLE_CONFIRMED / START_ROLE_WEAKENED /
# ROTATION_RISK_HIGH / RETURN_RAMP / OFFICIAL_OUT / OFFICIAL_AVAILABLE /
# MIDWEEK_HEAVY. Notes expire by their own timestamps; post-cutoff or expired
# notes never contribute.
# ---------------------------------------------------------------------------

SCOUT_MODIFIER_RULES: dict[tuple[str, str], tuple[str, float, str]] = {
    ("role_security_5gw", "very_high"): ("p_start_delta", 0.05, "START_ROLE_CONFIRMED"),
    ("role_security_5gw", "high"): ("p_start_delta", 0.025, "START_ROLE_CONFIRMED"),
    ("role_security_5gw", "low"): ("p_start_delta", -0.05, "START_ROLE_WEAKENED"),
    ("role_security_5gw", "very_low"): ("p_start_delta", -0.10, "START_ROLE_WEAKENED"),
    ("rotation_risk", "very_high"): ("p_start_delta", -0.10, "ROTATION_RISK_HIGH"),
    ("rotation_risk", "high"): ("p_start_delta", -0.05, "ROTATION_RISK_HIGH"),
    ("competition_for_position", "very_high"): ("p_start_delta", -0.10, "ROTATION_RISK_HIGH"),
    ("competition_for_position", "high"): ("p_start_delta", -0.05, "ROTATION_RISK_HIGH"),
    ("european_congestion", "high"): ("p_start_delta", -0.05, "ROTATION_RISK_HIGH"),
    ("european_congestion", "very_high"): ("p_start_delta", -0.075, "ROTATION_RISK_HIGH"),
    ("injury_uncertainty", "very_high"): ("p_start_delta", -0.05, "RETURN_RAMP"),
}
_TEXT_BANDS = {"very_low", "low", "high", "very_high"}


def fold_scout_modifiers(
    notes: list[dict[str, Any]],
    cutoff: str,
    config: MinutesModelConfig,
) -> tuple[dict[str, float], list[str], list[str]]:
    """Fold structured notes into transparent bounded parameter adjustments.

    Returns ``(adjustments, modifier_ids, risk_flags)`` where adjustments carry
    ``p_start_delta`` (bounded), ``expected_minutes_blend`` (value, weight) and
    availability overrides. Post-cutoff or expired notes are ignored; the same
    parameter family coming from the current-per-key view cannot self-conflict,
    so no value is silently chosen when evidence disagrees.
    """

    adjustments: dict[str, float] = {"p_start_delta_sum": 0.0}
    modifier_ids: list[str] = []
    flags: list[str] = []
    cutoff_dt = parse_utc(cutoff)
    for note in sorted(notes, key=lambda row: str(row.get("observed_at") or "")):
        observed = parse_utc(note.get("observed_at"))
        if observed is None or cutoff_dt is None or observed > cutoff_dt:
            continue
        expires = parse_utc(note.get("expires_at"))
        if expires is not None and cutoff_dt is not None and expires < cutoff_dt:
            flags.append("SCOUTING_NOTE_EXPIRED_IGNORED")
            continue
        key = str(note.get("key") or "")
        value_num = note.get("value_num")
        value_text = str(note.get("value_text") or "").strip()
        band = value_text if value_text in _TEXT_BANDS else None
        if key == "availability":
            band = value_text if value_text in {"available", "unavailable"} else None
        if key == "start_probability" and isinstance(value_num, (int, float)) and not isinstance(value_num, bool):
            delta = _START_BAND_DELTAS.get(float(value_num))
            if delta:
                adjustments["p_start_delta_sum"] += delta
                modifier_ids.append(f"scouting_note:{key}:band{float(value_num):g}:note{note.get('id')}")
                if delta > 0:
                    flags.append("START_ROLE_CONFIRMED" if delta >= 0.05 else "start_probability_uplift")
        elif key == "expected_minutes" and isinstance(value_num, (int, float)) and not isinstance(value_num, bool):
            # Bounded anchored blend of the model's own conditional minutes.
            weight = config.scout_expected_minutes_blend_weight
            adjustments["expected_minutes_blend_value"] = float(value_num)
            adjustments["expected_minutes_blend_weight"] = weight
            modifier_ids.append(f"scouting_note:{key}:blend:note{note.get('id')}")
        elif key == "injury_uncertainty" and isinstance(value_num, (int, float)) and not isinstance(value_num, bool):
            if float(value_num) >= _INJURY_UNCERTAINTY_HIGH:
                rule = SCOUT_MODIFIER_RULES.get(("injury_uncertainty", "very_high"))
                adjustments["p_start_delta_sum"] += float(rule[1])
                flags.append("RETURN_RAMP_SCOUT_HIGH")
                modifier_ids.append(f"scouting_note:{key}:high_uncertainty:note{note.get('id')}")
        elif key == "availability" and band:
            if band == "unavailable":
                adjustments["availability_override"] = 0.0
                modifier_ids.append(f"scouting_note:{key}:unavailable:note{note.get('id')}")
            else:
                adjustments["availability_floor_evidence"] = 1.0
                modifier_ids.append(f"scouting_note:{key}:available:note{note.get('id')}")
        elif band and (key, band) in SCOUT_MODIFIER_RULES:
            parameter, magnitude, modifier_type = SCOUT_MODIFIER_RULES[(key, band)]
            adjustments["p_start_delta_sum"] += float(magnitude)
            flags.append(modifier_type)
            modifier_ids.append(f"scouting_note:{key}:{band}:note{note.get('id')}")

    bound = config.scout_p_start_delta_bound
    canvas_sum = adjustments.get("p_start_delta_sum", 0.0)
    if abs(canvas_sum) > bound:
        flags.append("P_START_DELTA_CLAMPED")
    adjustments["p_start_delta"] = max(-bound, min(bound, canvas_sum))
    adjustments.pop("p_start_delta_sum", None)
    if adjustments.get("availability_override") is not None and adjustments.get("availability_floor_evidence") == 1.0:
        flags.append("CONFLICTING_AVAILABILITY_EVIDENCE")
        adjustments.pop("availability_override", None)
        adjustments.pop("availability_floor_evidence", None)
    return adjustments, modifier_ids, flags


# ---------------------------------------------------------------------------
# League-level pooling (transparent level-3/4 priors).
# ---------------------------------------------------------------------------


@dataclass
class LeaguePools:
    position: dict[int, dict[str, float]]
    team_position: dict[tuple[int, int], dict[str, float]]


def league_pools(conn: sqlite3.Connection, planning_event: int, cutoff: str) -> LeaguePools:
    position = analytics.positional_pooled_engagement(conn, planning_event, cutoff)
    rows = conn.execute(
        """SELECT p.team_id AS team_id, p.element_type AS pos, pg.starts AS started, pg.minutes AS minutes
           FROM player_gameweeks pg
           JOIN players p ON p.id=pg.player_id
           JOIN fixtures f ON f.id=pg.fixture_id
          WHERE {boundary}
            AND pg.starts IS NOT NULL AND pg.minutes IS NOT NULL
            AND p.team_id IS NOT NULL AND p.element_type IS NOT NULL""".format(
            boundary=historical.OBSERVATION_SQL_CLAUSES
        ),
        historical.boundary_params(cutoff, planning_event=int(planning_event)),
    ).fetchall()
    team_position: dict[tuple[int, int], dict[str, float]] = {}
    for row in rows:
        key = (int(row["team_id"]), int(row["pos"]))
        pool = team_position.setdefault(key, {"rows": 0.0, "starts": 0.0})
        pool["rows"] += 1
        pool["starts"] += 1 if row["started"] else 0
    return LeaguePools(position=position, team_position=team_position)


# ---------------------------------------------------------------------------
# Player-fixture model.
# ---------------------------------------------------------------------------


def season_prior(
    conn: sqlite3.Connection,
    player_id: int,
    *,
    returning: bool,
    role_weakened: bool,
    config: MinutesModelConfig,
) -> dict[str, Any] | None:
    """Player previous-season prior with explicit effective-sample-size semantics.

    Uses the official element-summary `history_past` store (proven fields:
    starts, minutes; no official appearances field).  Match count is derived
    with an explicit documented assumption (extra sub-app minutes ≈ 18'), then
    converted to an ESS with a cap; thin prior exposure, a role-change signal,
    and a returning-from-injury signal each reduce ESS by bounded, visible
    discounts — a one-appearance prior season can never look highly certain.
    """

    rows = repo.player_season_histories(conn, int(player_id))
    if not rows:
        return None
    row = rows[0]
    minutes = int(row.get("minutes") or 0)
    starts = int(row.get("starts") or 0)
    estimated_matches = round(
        starts + max(0.0, minutes - 90.0 * starts) / config.prev_season_sub_minutes_assumption
    )
    estimated_matches = min(max(estimated_matches, starts), 38)
    ess = min(float(estimated_matches), config.prev_season_ess_cap)
    discounts: list[str] = []
    if ess > 0 and minutes < config.prev_season_ess_min_minutes:
        thin_factor = minutes / config.prev_season_ess_min_minutes
        ess *= thin_factor
        discounts.append(f"thin_prev_exposure:{minutes}min<={int(config.prev_season_ess_min_minutes)}min")
    if role_weakened:
        ess *= config.prev_season_role_change_ess_discount
        discounts.append("role_change_prev_ess")
    if returning:
        ess *= config.prev_season_returning_ess_discount
        discounts.append("returning_prev_ess")
    ess = max(0.0, round(ess, 4))
    start_rate = round(starts / estimated_matches, 6) if estimated_matches > 0 else None
    minutes_per_start = round(minutes / starts, 4) if starts > 0 else None
    return {
        "season_name": row.get("season_name"),
        "official_starts": starts,
        "official_minutes": minutes,
        "estimated_previous_matches": estimated_matches,
        "effective_sample_size": ess,
        "start_rate": start_rate,
        "minutes_per_start": minutes_per_start,
        "ess_discounts": discounts,
        "matches_estimate_assumption": "extra sub-app minutes ~ PREV_SEASON_SUB_MINUTES_ASSUMPTION (18) per non-start appearance",
    }


def _shrink(evidence_total: float, evidence_count: float, prior: float, strength: float) -> float:
    return (evidence_total + strength * prior) / (evidence_count + strength)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _round(value: float, digits: int = 6) -> float:
    return round(value, digits)


def project_player_fixture(
    conn: sqlite3.Connection,
    pool: LeaguePools,
    player_row: Mapping[str, Any],
    evidence_rows: list[dict[str, Any]],
    fixture: Mapping[str, Any],
    snapshot: Mapping[str, Any] | None,
    scout_notes: list[dict[str, Any]],
    config: MinutesModelConfig,
    cutoff: str,
    *,
    include_conditionals: bool = False,
    snapshot_history: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """One coherent player-fixture projection (raw + adjusted values).

    ``include_conditionals`` additionally exposes the conditional quantities
    (P(60+|start), P(80+|start), P(60+|cameo)) that the team-coherence layer
    needs.  It defaults to False so the frozen v1.1.0 payload shape is
    unchanged.

    ``snapshot_history`` is the player's official status trail, used to tell an
    available non-selection apart from an unavailable player.  When it is omitted
    every 0-minute row is classed UNKNOWN and therefore contributes nothing,
    which is the conservative pre-v1.6.0 behaviour.
    """

    player_id = int(player_row["player_id"])
    team_id = int(player_row["team_id"])
    position = int(player_row["element_type"])
    flags: list[str] = []
    adjustments, modifier_ids, scout_flags = fold_scout_modifiers(scout_notes, cutoff, config)
    flags.extend(scout_flags)

    pos_pool = pool.position.get(position) or structural_no_history_priors(config)
    team_pos_pool = pool.team_position.get((team_id, position))

    # --- availability -------------------------------------------------------
    availability_sources: dict[str, Any] = {
        "official_snapshot": bool(snapshot),
        "status": (snapshot or {}).get("status"),
        "chance_of_playing_this_round": (snapshot or {}).get("chance_of_playing_this_round"),
    }
    status = (snapshot or {}).get("status")
    chance = (snapshot or {}).get("chance_of_playing_this_round")
    signal_defaults = config.official_availability_signal_defaults
    hard_statuses = config.hard_unavailable_statuses
    if status in hard_statuses:
        p_available = 0.0
        availability_sources["rule"] = f"hard official status '{status}' (config hard_unavailable_statuses)"
    elif isinstance(chance, (int, float)) and not isinstance(chance, bool):
        # Official event-specific availability signal; modelled multiplier.
        p_available = max(0.0, min(1.0, float(chance) / 100.0))
        availability_sources["rule"] = "official chance_of_playing_this_round signal (initial modelling assumption, not calibrated)"
    elif status in signal_defaults:
        p_available = float(signal_defaults[status])
        availability_sources["rule"] = f"official status '{status}' signal (initial modelling assumption, not calibrated)"
    else:
        p_available = 1.0
        availability_sources["rule"] = "no official restriction found (default available)"
    if "availability_override" in adjustments:
        p_available = float(adjustments["availability_override"])
        availability_sources["scouting_override"] = "OFFICIAL_OUT-type scouting note"
    if snapshot is None:
        flags.append("NO_OFFICIAL_STATUS_SNAPSHOT")

    official_return_ramp = (
        snapshot is not None and status in config.returning_status_values
    )
    returning = official_return_ramp or "RETURN_RAMP_SCOUT_HIGH" in flags
    if returning:
        flags.append("RETURN_RAMP")

    # --- P(start | available): recency-weighted shrinkage --------------------
    # v1.6.0 evidence taxonomy.  Every completed row is classified so that
    # non-selection can be distinguished from unavailability (see
    # ``classify_evidence_rows``).  Only rows that prove the player was
    # available and attributable rows enter the start-rate denominator;
    # unavailable and unknown rows contribute nothing at all.
    classified_rows = classify_evidence_rows(
        evidence_rows, snapshot_history=snapshot_history, config=config
    )
    placeholder_rows = [row for row in classified_rows if row.get("history_placeholder")]
    if placeholder_rows:
        # Report rather than absorb: a completed fixture whose row holds no
        # official observation must not look like a quiet non-selection.
        flags.append(hc.DIAG_COMPLETED_EVENT_PLACEHOLDER_ROW)
    observation_rows = [
        row for row in classified_rows if row["evidence_class"] in _START_OBSERVATION_CLASSES
    ]
    recency = _recency_weight_list(len(observation_rows), config.start_recency_half_life_matches)
    weighted_starts = 0.0
    weighted_obs = 0.0
    weighted_start_minutes = 0.0
    weighted_minutes_count = 0.0
    weighted_p60 = 0.0
    weighted_p80 = 0.0
    start_obs_count = 0
    not_start_count = 0
    available_non_start_zero_minute_count = 0
    cameo_count = 0
    cameo_minutes_total = 0.0
    evidence_class_counts: dict[str, int] = {name: 0 for name in EVIDENCE_CLASSES}
    for row in classified_rows:
        evidence_class_counts[row["evidence_class"]] = (
            evidence_class_counts.get(row["evidence_class"], 0) + 1
        )
    for index, row in enumerate(observation_rows):
        minutes = row.get("minutes")
        if minutes is None:
            continue
        weight = recency[index]
        started = row.get("starts")
        if started is None:
            continue
        weighted_obs += weight
        weighted_starts += weight * (1 if started else 0)
        if started:
            start_obs_count += 1
            weighted_minutes_count += weight
            weighted_start_minutes += weight * float(minutes)
            weighted_p60 += weight * (1 if float(minutes) >= 60 else 0)
            weighted_p80 += weight * (1 if float(minutes) >= 80 else 0)
        else:
            not_start_count += 1
            if row["evidence_class"] == EVIDENCE_AVAILABLE_NON_START_ZERO_MINUTES:
                available_non_start_zero_minute_count += 1
            if float(minutes) > 0:
                cameo_count += 1
                cameo_minutes_total += float(minutes)
    appearance_rows = [row for row in observation_rows if (row.get("minutes") or 0) > 0]

    level34_rate: float
    if team_pos_pool and team_pos_pool["rows"] > 0:
        backoff = config.team_position_prior_backoff_strength
        level34_rate = _shrink(team_pos_pool["starts"], team_pos_pool["rows"], pos_pool["p_start"], backoff)
        prior_source = "team+position pooled shrunk to position rate"
    else:
        level34_rate = pos_pool["p_start"]
        prior_source = "position pooled rate"
    level34_minutes_anchor = float(pos_pool.get("minutes_if_start") or REGULATION_MATCH_MINUTES)

    # --- previous-season player prior (v1.1.0, ESS semantics) ----------------
    role_weakened = "START_ROLE_WEAKENED" in flags or "ROTATION_RISK_HIGH" in flags
    prev_prior = season_prior(conn, player_id, returning=returning, role_weakened=role_weakened, config=config)
    prior_anchor_strength = config.prev_season_prior_anchor_strength
    minutes_anchor_strength = config.prev_season_minutes_anchor_strength

    # --- role discontinuity (v1.6.0) ----------------------------------------
    # A strong prior role that has NOT reproduced at the current club, while the
    # player was available often enough to have shown it, is a discontinuity.
    # The prior club cannot be read from storage, so the honest reading is
    # "prior role provenance is uncertain", not "the player is worse".
    prior_role_strength = "none"
    if prev_prior is not None and prev_prior["effective_sample_size"] > 0:
        rate = prev_prior.get("start_rate")
        if rate is not None:
            if rate >= config.role_discontinuity_prior_start_rate:
                prior_role_strength = "strong"
            elif rate > 0.0:
                prior_role_strength = "weak"
    available_observations = sum(
        evidence_class_counts.get(name, 0) for name in _START_OBSERVATION_CLASSES
    )
    available_non_start_zero_minute_observations = evidence_class_counts.get(
        EVIDENCE_AVAILABLE_NON_START_ZERO_MINUTES, 0
    )
    role_discontinuity = bool(
        prior_role_strength == "strong"
        and start_obs_count == 0
        and available_observations >= config.role_discontinuity_min_available_observations
    )

    if prev_prior is not None and prev_prior["effective_sample_size"] > 0:
        ess = float(prev_prior["effective_sample_size"])
        start_ess = ess
        if role_discontinuity:
            start_ess = ess * config.prev_season_role_discontinuity_ess_discount
            prev_prior = dict(prev_prior)
            prev_prior["ess_discounts"] = list(prev_prior.get("ess_discounts") or []) + [
                "role_discontinuity_prev_ess"
            ]
            prev_prior["role_discontinuity_ess_discount"] = (
                config.prev_season_role_discontinuity_ess_discount
            )
            prev_prior["start_ess_after_discontinuity"] = _round(start_ess)
            # A role discontinuity is observed.  Whether the *cause* is a club
            # change cannot be established from this evidence, so the discount
            # is labelled as a diagnostic, not as a proven transfer correction.
            flags.append(FLAG_ROLE_DISCONTINUITY_PRIOR_DISCOUNT)
            flags.append(FLAG_PRIOR_CLUB_UNVERIFIED)
        if prev_prior["start_rate"] is not None:
            level34_rate = (
                start_ess * prev_prior["start_rate"] + prior_anchor_strength * level34_rate
            ) / (start_ess + prior_anchor_strength)
            prior_source = (
                f"previous-season {prev_prior['season_name']} prior (ESS {start_ess:g}"
                + (", role-discontinuity discounted" if role_discontinuity else "")
                + ") shrunk to pooled rate"
            )
        if prev_prior.get("minutes_per_start"):
            # Durability/position information is retained at full ESS: a role
            # discontinuity is about *whether* he starts, not about whether he
            # can last 90 minutes.
            level34_minutes_anchor = (
                ess * float(prev_prior["minutes_per_start"]) + minutes_anchor_strength * level34_minutes_anchor
            ) / (ess + minutes_anchor_strength)
    prior_start = level34_rate

    if weighted_obs <= 0:
        p_start_given_available_raw = prior_start
        flags.append("NO_LEAGUE_START_EVIDENCE")
    else:
        p_start_given_available_raw = _shrink(
            weighted_starts, weighted_obs, prior_start, config.start_prior_strength
        )

    # --- role-evidence diagnostics (v1.6.0) ---------------------------------
    # SCOUTING_ROLE_CONFLICT: the manual role note and the observed current-season
    # role point in opposite directions.  Both directions are recorded, so the
    # flag means "these two evidence sources disagree", not "scouting is wrong".
    scouting_weakened = "START_ROLE_WEAKENED" in flags or "ROTATION_RISK_HIGH" in flags
    if scouting_weakened and start_obs_count > 0:
        flags.append(FLAG_SCOUTING_ROLE_CONFLICT)
    if "START_ROLE_CONFIRMED" in flags and start_obs_count == 0 and role_discontinuity:
        flags.append(FLAG_SCOUTING_ROLE_CONFLICT)
    if role_discontinuity:
        flags.append(FLAG_MODEL_PRIOR_CONFLICT)
    if available_non_start_zero_minute_observations >= config.strong_zero_minute_non_start_evidence_threshold:
        flags.append(FLAG_ZERO_MINUTE_NON_START_EVIDENCE_STRONG)
    if evidence_class_counts.get(EVIDENCE_NOT_IN_MATCHDAY_SQUAD, 0) == 0:
        # The official element-summary payload has no matchday-squad field, so
        # "out of the squad" can never be asserted from stored data.
        flags.append(FLAG_MATCHDAY_SQUAD_EVIDENCE_UNAVAILABLE)

    e_minutes_if_start_raw: float
    p60_if_start = float(pos_pool.get("p60_if_start") or 0.5)
    p80_if_start = float(pos_pool.get("p80_if_start") or 0.3)
    if weighted_minutes_count > 0:
        e_minutes_if_start_raw = _shrink(
            weighted_start_minutes, weighted_minutes_count, level34_minutes_anchor, config.minutes_if_start_prior_strength
        )
        p60_if_start = _shrink(weighted_p60, weighted_minutes_count, pos_pool.get("p60_if_start") or 0.55, config.p60_if_start_prior_strength)
        p80_if_start = _shrink(weighted_p80, weighted_minutes_count, pos_pool.get("p80_if_start") or 0.3, config.p80_if_start_prior_strength)
    else:
        e_minutes_if_start_raw = level34_minutes_anchor
        flags.append("NO_START_MINUTES_EVIDENCE")
    e_minutes_if_start_raw = min(max(e_minutes_if_start_raw, 0.0), 90.0)

    # --- cameo model (position-pooled) --------------------------------------
    not_start_available_rows = not_start_count
    if not_start_available_rows > 0 and pos_pool.get("not_start_rows", 0) >= 0:
        cameo_rate_raw = _shrink(cameo_count, not_start_available_rows, pos_pool.get("cameo_rate") or 0.5, config.cameo_rate_prior_strength)
    else:
        cameo_rate_raw = pos_pool.get("cameo_rate") or 0.5
    if cameo_count > 0:
        e_minutes_if_cameo_raw = _shrink(
            cameo_minutes_total, cameo_count, pos_pool.get("cameo_minutes") or 15.0, config.cameo_minutes_prior_strength
        )
    else:
        e_minutes_if_cameo_raw = float(pos_pool.get("cameo_minutes") or 15.0)
        flags.append("NO_CAMEO_MINUTES_EVIDENCE")
    e_minutes_if_cameo_raw = min(max(e_minutes_if_cameo_raw, 0.0), 90.0)
    p60_if_cameo_raw = _shrink(0.0, 0.0, max(float(pos_pool.get("cameo_p60") or 0.0), config.cameo_p60_prior), config.cameo_p60_prior_strength)
    p80_if_cameo_raw = 0.0  # cameo tail deliberately folded into 1-59 in v1
    cameo_rate_adj = max(0.0, min(1.0, cameo_rate_raw))

    # --- structural modifiers (midweek congestion, injury return) -----------
    p_sga_structural = p_start_given_available_raw
    upper_minutes_multiplier = 1.0
    minutes_if_start_multiplier = 1.0
    congestion_block = _midweek_block(
        conn, player_row, evidence_rows, fixture, config, cutoff
    )
    if congestion_block:
        flags.extend(congestion_block["flags"])
        if congestion_block["type"] == "MIDWEEK_HEAVY":
            p_sga_structural *= config.midweek_heavy_start_modifier
            upper_minutes_multiplier *= config.midweek_heavy_upper_minutes_modifier
        elif congestion_block["type"] == "MIDWEEK_REST":
            p_sga_structural *= config.midweek_rest_start_modifier
    if returning:
        p_sga_structural *= config.returning_start_modifier
        upper_minutes_multiplier *= config.returning_upper_minutes_modifier
        minutes_if_start_multiplier *= config.returning_minutes_if_start_modifier

    p60_if_start = min(max(_clamp(p60_if_start * upper_minutes_multiplier), 0.0), 1.0)
    p80_if_start = min(max(_clamp(p80_if_start * upper_minutes_multiplier), 0.0), p60_if_start)
    e_minutes_if_start = _clamp(
        e_minutes_if_start_raw * minutes_if_start_multiplier, 0.0, 90.0
    )

    # --- scouting/manual adjustments (bounded, recorded separately) ---------
    p_start_given_available_adj = _clamp(p_sga_structural + float(adjustments.get("p_start_delta", 0.0)))
    e_minutes_if_cameo = e_minutes_if_cameo_raw
    if "expected_minutes_blend_value" in adjustments:
        weight = float(adjustments.get("expected_minutes_blend_weight", 0.0))
        e_minutes_if_start = _clamp(
            (1.0 - weight) * e_minutes_if_start + weight * float(adjustments["expected_minutes_blend_value"]),
            0.0,
            90.0,
        )

    # --- derived, coherent distributions -------------------------------------
    # raw_* = the structural model only (recency shrinkage, congestion, return
    # ramp) before any scouting/manual modifier; adjusted = after modifiers.
    p_start_structural = _clamp(p_available * p_sga_structural)
    p_cameo_structural = _clamp(p_available * (1.0 - p_sga_structural) * cameo_rate_adj)
    raw_expected_minutes = p_start_structural * e_minutes_if_start_raw * minutes_if_start_multiplier + p_cameo_structural * e_minutes_if_cameo_raw

    p_start = _clamp(p_available * p_start_given_available_adj)
    not_start_adj = (1.0 - p_start_given_available_adj) if p_available > 0 else 1.0
    p_cameo = _clamp(p_available * not_start_adj * cameo_rate_adj)
    p_zero = _clamp(1.0 - p_start - p_cameo)
    p60_plus = _clamp(p_start * p60_if_start + p_cameo * p60_if_cameo_raw)
    p80_plus = _clamp(p_start * p80_if_start)
    p80_plus = min(p80_plus, p60_plus)
    p_1_59 = _clamp(1.0 - p_zero - p60_plus)
    adjusted_p_start = p_start
    raw_p_start = p_start_structural
    expected_minutes = p_start * e_minutes_if_start + p_cameo * e_minutes_if_cameo

    # --- summary coherence guards (model-level invariants) -------------------
    for name, value in (
        ("p_available", p_available), ("p_start", p_start), ("p_cameo", p_cameo),
        ("p_zero", p_zero), ("p_1_59", p_1_59), ("p_60_plus", p60_plus), ("p_80_plus", p80_plus),
        ("p_start_given_available", p_start_given_available_adj),
    ):
        if not 0.0 - 1e-9 <= value <= 1.0 + 1e-9:
            raise ValueError(f"probability out of range: {name}={value}")
    if abs((p_zero + p_1_59 + p60_plus) - 1.0) > 1e-6:
        p_zero = _clamp(1.0 - p_1_59 - p60_plus)

    if start_obs_count < _LOW_HISTORY_ROWS:
        flags.append("LOW_START_HISTORY")
    if not_start_available_rows < _LOW_HISTORY_ROWS:
        flags.append("LOW_CAMEO_EVIDENCE")

    return {
        "player_id": player_id,
        "fixture_id": int(fixture["id"]),
        "event": int(fixture.get("event")),
        "p_available": _round(p_available),
        "p_start_given_available": _round(p_start_given_available_adj),
        "p_start": _round(p_start),
        "p_cameo_given_not_start": _round(cameo_rate_adj),
        "p_cameo": _round(p_cameo),
        "p_zero": _round(p_zero),
        "p_1_59": _round(p_1_59),
        "p_60_plus": _round(p60_plus),
        "p_80_plus": _round(p80_plus),
        "expected_minutes": _round(expected_minutes),
        "expected_minutes_if_start": _round(e_minutes_if_start),
        "expected_minutes_if_cameo": _round(e_minutes_if_cameo),
        "raw_p_start": _round(raw_p_start),
        "adjusted_p_start": _round(p_start),
        "raw_expected_minutes": _round(expected_minutes),
        "adjusted_expected_minutes": _round(expected_minutes),
        "availability_source_summary": availability_sources,
        "modifier_ids": modifier_ids,
        "risk_flags": sorted(set(flags)),
        "start_evidence": {
            "observed_rows": len(evidence_rows),
            "known_available_rows": len(appearance_rows),
            "available_observation_rows": len(observation_rows),
            "starts_observed": start_obs_count,
            "not_start_rows": not_start_available_rows,
            "available_non_start_zero_minute_observations": available_non_start_zero_minute_count,
            "cameo_appearances": cameo_count,
            "prior_source": prior_source,
            "prior_start_rate": _round(prior_start),
        },
        "evidence_classes": evidence_class_counts,
        "history_completeness": {
            "placeholder_rows": len(placeholder_rows),
            "detail": (
                "no completed-fixture placeholder rows"
                if not placeholder_rows
                else f"{hc.CERTIFIED_PREDICTION_INPUT_HISTORY_INCOMPLETE}: completed fixture(s) "
                     "carry no official observation and are excluded from start evidence"
            ),
        },
        "role_evidence": {
            "prior_role_strength": prior_role_strength,
            "prior_role_discontinuity": role_discontinuity,
            "current_season_starts": start_obs_count,
            "available_observations": available_observations,
            "available_non_start_zero_minute_observations": available_non_start_zero_minute_count,
            "matchday_squad_evidence": (
                "AVAILABLE"
                if evidence_class_counts.get(EVIDENCE_NOT_IN_MATCHDAY_SQUAD, 0) > 0
                else "UNKNOWN_NO_OFFICIAL_SQUAD_FIELD"
            ),
            "prior_club_provenance": "UNVERIFIED_IN_STORED_EVIDENCE",
            "role_discontinuity_diagnostics": sorted(
                flag
                for flag in (
                    FLAG_ROLE_DISCONTINUITY_PRIOR_DISCOUNT,
                    FLAG_PRIOR_CLUB_UNVERIFIED,
                    FLAG_TRANSFER_ROLE_CHANGE,
                )
                if flag in flags
            ),
            "conflict_flags": sorted(
                flag
                for flag in (
                    *ROLE_CONFLICT_FLAGS,
                )
                if flag in flags
            ),
            "uncertainty_reasons": sorted(
                {
                    *(reason for reason in _role_uncertainty_reasons(
                        role_discontinuity=role_discontinuity,
                        available_non_start_zero_minute_observations=available_non_start_zero_minute_count,
                        evidence_class_counts=evidence_class_counts,
                        flags=flags,
                        config=config,
                    )),
                }
            ),
            "role_confidence": _role_confidence(
                role_discontinuity=role_discontinuity,
                evidence_class_counts=evidence_class_counts,
                flags=flags,
                config=config,
            ),
            "p_start_basis": "current_evidence" if weighted_obs > 0 else "prior_only",
        },
        "prev_season_prior": prev_prior,
        "model_version": MINUTES_MODEL_VERSION,
        "generated_at": utc_now(),
        **(
            {
                "p_60_given_start": _round(p60_if_start),
                "p_80_given_start": _round(p80_if_start),
                "p_60_given_cameo": _round(p60_if_cameo_raw),
            }
            if include_conditionals
            else {}
        ),
    }


# ---------------------------------------------------------------------------
# Evidence taxonomy (v1.6.0).
#
# The v1.1.0 rule was ``minutes > 0``: a 0-minute completed row was discarded
# entirely, so an unused substitute was indistinguishable from a player who had
# never been available.  R1 showed the consequence — a backup goalkeeper with
# three unused-substitute selections retained a prior-inflated start
# probability because non-selection generated no evidence at all.
#
# Every completed row is now assigned exactly one class, and each class has an
# explicit, documented contribution:
#
#   STARTED                positive start evidence (also start-minutes, P60+)
#   BENCH_APPEARANCE       appeared from the bench: positive cameo evidence and
#                          a non-start observation for the start rate
#   AVAILABLE_NON_START_ZERO_MINUTES
#                          known available around the event, did not start, and
#                          recorded 0 minutes: NEGATIVE start evidence, and a
#                          negative cameo observation.  Matchday-squad
#                          membership is UNKNOWN -- the official historical
#                          payload does not prove the player was named on the
#                          bench, so this class makes no bench claim.
#   UNUSED_SUB_AVAILABLE   named on the bench and did not play.  Reserved for a
#                          source that explicitly proves bench membership; never
#                          inferred from a 0-minute row.
#   NOT_IN_MATCHDAY_SQUAD  stronger negative role evidence than an available
#                          non-start, but only when the absence is not explained
#                          by unavailability AND squad membership is explicitly
#                          known.  Reserved for the same reason: the stored
#                          official payload carries no matchday-squad field
#                          (verified against player_gameweeks.raw_json).
#   UNAVAILABLE            injured/suspended/otherwise out.  NOT role evidence
#                          in either direction.
#   UNKNOWN                cannot be classified (missing official fields, or no
#                          availability trail).  Contributes NOTHING; it never
#                          silently becomes negative evidence.
# ---------------------------------------------------------------------------

EVIDENCE_STARTED = "STARTED"
EVIDENCE_BENCH_APPEARANCE = "BENCH_APPEARANCE"
# Inferred today: availability is known, bench membership is not.
EVIDENCE_AVAILABLE_NON_START_ZERO_MINUTES = "AVAILABLE_NON_START_ZERO_MINUTES"
# Reserved for explicit-source classifications.  Neither is ever inferred from a
# 0-minute row, because the official payload does not establish squad or bench
# membership.
EVIDENCE_UNUSED_SUB_AVAILABLE = "UNUSED_SUB_AVAILABLE"
EVIDENCE_NOT_IN_MATCHDAY_SQUAD = "NOT_IN_MATCHDAY_SQUAD"
EVIDENCE_UNAVAILABLE = "UNAVAILABLE"
EVIDENCE_UNKNOWN = "UNKNOWN"

EVIDENCE_CLASSES = (
    EVIDENCE_STARTED,
    EVIDENCE_BENCH_APPEARANCE,
    EVIDENCE_AVAILABLE_NON_START_ZERO_MINUTES,
    EVIDENCE_UNUSED_SUB_AVAILABLE,
    EVIDENCE_NOT_IN_MATCHDAY_SQUAD,
    EVIDENCE_UNAVAILABLE,
    EVIDENCE_UNKNOWN,
)

# Classes that are reserved for explicit-source evidence and must never be
# produced by inference alone.
EVIDENCE_EXPLICIT_SOURCE_ONLY = (
    EVIDENCE_UNUSED_SUB_AVAILABLE,
    EVIDENCE_NOT_IN_MATCHDAY_SQUAD,
)

# Role-discontinuity diagnostics.  ``ROLE_DISCONTINUITY_PRIOR_DISCOUNT`` states
# what was actually observed (a strong prior role that has not reproduced) and
# ``PRIOR_CLUB_UNVERIFIED`` states what could not be established.  A
# ``TRANSFER_ROLE_CHANGE`` diagnostic may only be added when a trusted table
# establishes the club transition; no such source exists in the current store
# (player_season_histories carries no club, and player_snapshots.news is free
# text, not a club-transition record), so it is defined but never emitted.
FLAG_ROLE_DISCONTINUITY_PRIOR_DISCOUNT = "ROLE_DISCONTINUITY_PRIOR_DISCOUNT"
FLAG_PRIOR_CLUB_UNVERIFIED = "PRIOR_CLUB_UNVERIFIED"
FLAG_TRANSFER_ROLE_CHANGE = "TRANSFER_ROLE_CHANGE"
FLAG_MODEL_PRIOR_CONFLICT = "MODEL_PRIOR_CONFLICT_WITH_CURRENT_ROLE"
# Stored evidence proves a non-start with zero minutes, NOT matchday-squad
# non-selection, so the label says exactly that.
FLAG_ZERO_MINUTE_NON_START_EVIDENCE_STRONG = "CURRENT_SEASON_ZERO_MINUTE_NON_START_EVIDENCE_STRONG"
FLAG_SCOUTING_ROLE_CONFLICT = "SCOUTING_ROLE_CONFLICT"
FLAG_MATCHDAY_SQUAD_EVIDENCE_UNAVAILABLE = "MATCHDAY_SQUAD_EVIDENCE_UNAVAILABLE"

ROLE_CONFLICT_FLAGS = (
    FLAG_MODEL_PRIOR_CONFLICT,
    FLAG_ZERO_MINUTE_NON_START_EVIDENCE_STRONG,
    FLAG_ROLE_DISCONTINUITY_PRIOR_DISCOUNT,
    FLAG_PRIOR_CLUB_UNVERIFIED,
    FLAG_TRANSFER_ROLE_CHANGE,
    FLAG_SCOUTING_ROLE_CONFLICT,
)

# Rows that prove the player was available around the event.  These form the
# denominator of the start rate.  Matchday-squad membership is NOT proven by the
# stored official payload and is never asserted.
_START_OBSERVATION_CLASSES = (
    EVIDENCE_STARTED,
    EVIDENCE_BENCH_APPEARANCE,
    EVIDENCE_AVAILABLE_NON_START_ZERO_MINUTES,
    EVIDENCE_UNUSED_SUB_AVAILABLE,
    EVIDENCE_NOT_IN_MATCHDAY_SQUAD,
)

# Official statuses that mean "not available for selection", so a 0-minute row
# is explained by unavailability rather than by losing the role.
_UNAVAILABLE_STATUSES = frozenset({"i", "u", "s", "n", "d"})


def availability_status_at_event(
    snapshot_history: Sequence[Mapping[str, Any]] | None,
    fixture_event: int | None,
) -> tuple[Any, Any, str]:
    """Official availability as known at a historical fixture.

    Returns ``(status, chance_of_playing, basis)``.

    Rule, in order:

    1. The newest snapshot whose ``event_context`` is at or before the fixture
       event — the status as it stood when the fixture was played.
    2. If no snapshot carries an event context (synthetic or legacy stores) but
       every snapshot agrees on one status, that constant status is used, with
       basis ``assumed_constant_status``.  A trail that never varied carries no
       event-specific information, so attributing it is safe.
    3. Otherwise ``(None, None, "unknown")`` — statuses varied and cannot be
       attributed to this fixture, so nothing is assumed.
    """

    rows = [row for row in (snapshot_history or []) if row is not None]
    if not rows:
        return None, None, "no_snapshot_history"

    dated = [
        row
        for row in rows
        if row.get("event_context") is not None
        and fixture_event is not None
        and int(row["event_context"]) <= int(fixture_event)
    ]
    if dated:
        chosen = max(
            dated,
            key=lambda row: (
                int(row["event_context"]),
                str(row.get("captured_at") or ""),
                int(row.get("id") or 0),
            ),
        )
        return chosen.get("status"), chosen.get("chance_of_playing_this_round"), "event_context"

    statuses = {row.get("status") for row in rows}
    if len(statuses) == 1:
        chosen = max(rows, key=lambda row: (str(row.get("captured_at") or ""), int(row.get("id") or 0)))
        return chosen.get("status"), chosen.get("chance_of_playing_this_round"), "assumed_constant_status"

    return None, None, "unknown"


def classify_evidence_rows(
    evidence_rows: Sequence[Mapping[str, Any]],
    *,
    snapshot_history: Sequence[Mapping[str, Any]] | None = None,
    config: "MinutesModelConfig | None" = None,
) -> list[dict[str, Any]]:
    """Assign one evidence class to each completed row.

    Returns new dicts (the input rows are not mutated) carrying
    ``evidence_class`` plus ``evidence_reason`` for auditability.
    """

    unavailable = set(_UNAVAILABLE_STATUSES)
    if config is not None:
        unavailable |= {str(value) for value in config.hard_unavailable_statuses}
    out: list[dict[str, Any]] = []
    for row in evidence_rows:
        row = dict(row)
        starts = row.get("starts")
        minutes = row.get("minutes")
        fixture_event = row.get("fixture_event", row.get("event"))
        status, chance, basis = availability_status_at_event(snapshot_history, fixture_event)
        row["availability_status_at_event"] = status
        row["availability_chance_at_event"] = chance
        row["availability_basis"] = basis

        if row.get("history_placeholder"):
            # The fixture is complete but the row carries no official
            # observation.  That is NOT a did-not-play -- a real DNP is all
            # explicit zeros -- so it must not enter the start-rate denominator
            # as negative selection evidence.  Keep the class UNKNOWN and say why,
            # so the gap is visible rather than silently absorbed.
            row["evidence_class"] = EVIDENCE_UNKNOWN
            row["evidence_reason"] = (
                "completed fixture carries a placeholder row with no official observation"
            )
            out.append(row)
            continue

        if starts is None or minutes is None:
            row["evidence_class"] = EVIDENCE_UNKNOWN
            row["evidence_reason"] = "missing_official_start_or_minutes_fields"
        elif starts:
            row["evidence_class"] = EVIDENCE_STARTED
            row["evidence_reason"] = "official start"
        elif float(minutes) > 0:
            row["evidence_class"] = EVIDENCE_BENCH_APPEARANCE
            # Playing without starting logically entails a substitute appearance,
            # so this class IS provable from the official record.
            row["evidence_reason"] = (
                f"substitute appearance: did not start and played {int(minutes)} minutes"
            )
        elif status in unavailable or chance == 0:
            row["evidence_class"] = EVIDENCE_UNAVAILABLE
            row["evidence_reason"] = (
                f"0 minutes explained by unavailability (status={status!r}, chance={chance!r})"
            )
        elif status is None:
            row["evidence_class"] = EVIDENCE_UNKNOWN
            row["evidence_reason"] = "0 minutes with no attributable availability evidence"
        else:
            row["evidence_class"] = EVIDENCE_AVAILABLE_NON_START_ZERO_MINUTES
            row["evidence_reason"] = (
                f"available (status={status!r}), did not start, 0 minutes: negative start "
                "evidence; matchday-squad membership unknown"
            )
        out.append(row)
    return out


def _role_uncertainty_reasons(
    *,
    role_discontinuity: bool,
    available_non_start_zero_minute_observations: int,
    evidence_class_counts: Mapping[str, int],
    flags: Sequence[str],
    config: MinutesModelConfig,
) -> list[str]:
    reasons: list[str] = []
    if role_discontinuity:
        reasons.append("prior_role_not_reproduced_at_current_club")
    if available_non_start_zero_minute_observations >= config.strong_zero_minute_non_start_evidence_threshold:
        reasons.append("repeated_available_zero_minute_non_start")
    if evidence_class_counts.get(EVIDENCE_UNKNOWN, 0) > 0:
        reasons.append("unclassified_completed_rows")
    if evidence_class_counts.get(EVIDENCE_STARTED, 0) == 0:
        reasons.append("no_current_season_starts")
    if "START_ROLE_WEAKENED" in flags or "ROTATION_RISK_HIGH" in flags:
        reasons.append("scouting_role_weakened")
    if "SCOUTING_NOTE_EXPIRED_IGNORED" in flags:
        reasons.append("scouting_evidence_expired")
    return reasons


def _role_confidence(
    *,
    role_discontinuity: bool,
    evidence_class_counts: Mapping[str, int],
    flags: Sequence[str],
    config: MinutesModelConfig,
) -> str:
    """Bounded confidence in the **role estimate** for downstream decision sanity.

    IMPORTANT SEMANTICS: this is confidence in the ROLE ESTIMATE -- how well the
    stored evidence supports the claim "this is the player's current starting
    role".  It is NOT the probability of starting, and it is not a confidence in
    ``p_start`` as a number.  A player whose role evidence is complete and
    consistent scores HIGH even when the evidence says he does not start (a
    well-evidenced backup), while a player with a thin or contradictory trail
    scores LOW even if ``p_start`` happens to be high.  ``p_start`` remains the
    probability; this field is the epistemic qualifier beside it.

    This never changes the optimizer objective.
    """

    if role_discontinuity or FLAG_SCOUTING_ROLE_CONFLICT in flags:
        return "LOW"
    unavailable = evidence_class_counts.get(EVIDENCE_UNAVAILABLE, 0)
    unknown = evidence_class_counts.get(EVIDENCE_UNKNOWN, 0)
    available = sum(evidence_class_counts.get(name, 0) for name in _START_OBSERVATION_CLASSES)
    if unavailable and available == 0:
        return "LOW"
    if unknown or available <= config.strong_zero_minute_non_start_evidence_threshold:
        return "MEDIUM"
    return "HIGH"


def _recency_weight_list(count: int, half_life: float) -> list[float]:
    return [math.pow(2.0, -(index / max(float(half_life), 1e-6))) for index in range(int(count))]


def _midweek_block(
    conn: sqlite3.Connection,
    player_row: Mapping[str, Any],
    evidence_rows: list[dict[str, Any]],
    fixture: Mapping[str, Any],
    config: MinutesModelConfig,
    cutoff: str,
) -> dict[str, Any] | None:
    """Transparent pooled congestion logic, never per-player coefficients."""

    kickoff = parse_utc(fixture.get("kickoff_time"))
    if kickoff is None:
        return {"type": "MIDWEEK_DATA_UNAVAILABLE", "flags": ["MIDWEEK_DATA_UNAVAILABLE"]}
    cutoff_dt = parse_utc(cutoff)
    previous: dict[str, Any] | None = None
    # Congestion evidence is any completed previous team fixture close enough
    # to this kickoff, regardless of which event number stores it (reschedules
    # legally live in older events).
    for row in conn.execute(
        """SELECT * FROM fixtures
           WHERE event IS NOT NULL AND event <= ? AND finished=1 AND started=1
             AND (team_h=? OR team_a=?)
           ORDER BY kickoff_time DESC, id DESC""",
        (int(fixture.get("event")), int(player_row["team_id"]), int(player_row["team_id"])),
    ).fetchall():
        record = dict(row)
        if int(record["id"]) == int(fixture["id"]):
            continue
        prev_kickoff = parse_utc(record.get("kickoff_time"))
        if prev_kickoff is None or prev_kickoff >= kickoff:
            continue
        if cutoff_dt is not None and prev_kickoff > cutoff_dt:
            continue
        previous = record
        break
    if previous is None:
        return None
    days_rest = (kickoff - parse_utc(previous["kickoff_time"])).total_seconds() / 86400.0
    if days_rest > config.midweek_rest_days_threshold:
        return None
    midweek_minutes = [
        float(row["minutes"]) for row in evidence_rows
        if int(row.get("fixture_id") or 0) == int(previous["id"]) and row.get("minutes") is not None
    ]
    if not midweek_minutes:
        return {"type": "MIDWEEK_DATA_UNAVAILABLE", "flags": ["MIDWEEK_DATA_UNAVAILABLE"], "days_rest": round(days_rest, 2)}
    minutes = float(midweek_minutes[0])
    if minutes >= config.midweek_heavy_minutes_threshold:
        return {"type": "MIDWEEK_HEAVY", "flags": ["MIDWEEK_HEAVY"], "days_rest": round(days_rest, 2), "midweek_minutes": minutes}
    if minutes == 0:
        started_recently = any(row.get("starts") for row in evidence_rows if int(row.get("fixture_id") or 0) != int(previous["id"]))
        if started_recently:
            return {"type": "MIDWEEK_REST", "flags": ["MIDWEEK_RESTED"], "days_rest": round(days_rest, 2), "midweek_minutes": 0}
    return None


def build_minutes_predictions(
    conn: sqlite3.Connection,
    planning_event: int,
    cutoff: str,
    config: MinutesModelConfig | None = None,
) -> list[dict[str, Any]]:
    """All player-fixture minute projections for the planning event."""

    config = config or MinutesModelConfig()
    pools = league_pools(conn, planning_event, cutoff)
    fixtures_by_team = analytics.event_fixture_map(conn, int(planning_event))
    rows_out: list[dict[str, Any]] = []
    for player in sorted(analytics.projectable_players(conn), key=lambda row: int(row["player_id"])):
        team_fixtures = fixtures_by_team.get(int(player["team_id"]))
        if not team_fixtures:
            continue
        player_id = int(player["player_id"])
        evidence_rows = analytics.completed_rows_as_of(conn, player_id, cutoff, int(planning_event))
        snapshot = analytics.snapshot_as_of(conn, player_id, cutoff)
        snapshot_history = analytics.snapshot_history_as_of(conn, player_id, cutoff)
        scout_notes = repo.scouting_current_rows_as_of(conn, cutoff, [player_id])
        for fixture in team_fixtures:
            rows_out.append(
                project_player_fixture(
                    conn,
                    pools,
                    player,
                    evidence_rows,
                    fixture,
                    snapshot,
                    scout_notes,
                    config,
                    cutoff,
                    snapshot_history=snapshot_history,
                )
            )
    return rows_out


# ---------------------------------------------------------------------------
# Analytics readiness gate (independent of the setup health gate).
# ---------------------------------------------------------------------------


def readiness_summary(
    context: Any,
    rows: list[dict[str, Any]],
    *,
    deadline_status: str | None = None,
    data_cutoff: str | None = None,
    deadline: str | None = None,
) -> dict[str, Any]:
    """Analytics-specific gate: never reuse the setup gate blindly."""

    fail_reasons: list[str] = []
    warn_reasons: list[str] = []
    counts: dict[str, int] = {}
    if context is not None:
        context_health = (getattr(context, "health", None) or {}).get("status")
        if context_health == "FAIL":
            fail_reasons.append(
                "PLANNING_CONTEXT_FAILED: " + "; ".join((context.health or {}).get("fail_reasons") or [])
            )
    if data_cutoff is not None and deadline is not None and deadline_status != "LATE_FREEZE":
        if parse_utc(data_cutoff) > parse_utc(deadline):
            fail_reasons.append("MODEL_INPUTS_AFTER_DEADLINE: a pre-deadline freeze cutoff may not exceed the deadline")
    flag_counts: dict[str, int] = {}
    for row in rows:
        for flag in row.get("risk_flags", []):
            flag_counts[flag] = flag_counts.get(flag, 0) + 1
    counts.update({f"players_{flag.lower()}": count for flag, count in sorted(flag_counts.items())})
    for flag in ("LOW_START_HISTORY", "RETURN_RAMP", "CONFLICTING_AVAILABILITY_EVIDENCE",
                 "SCOUTING_NOTE_EXPIRED_IGNORED", "MIDWEEK_DATA_UNAVAILABLE", "NO_LEAGUE_START_EVIDENCE"):
        if flag_counts.get(flag):
            warn_reasons.append(f"{flag}: {flag_counts[flag]} player-fixture projections affected")
    status = "FAIL" if fail_reasons else ("WARN" if warn_reasons else "PASS")
    return {"status": status, "fail_reasons": fail_reasons, "warn_reasons": warn_reasons, "counts": counts}
