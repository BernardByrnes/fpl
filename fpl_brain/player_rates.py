"""Player Attacking-Rate Shrinkage v1 — stable xG/90 and xA/90 beliefs.

This module produces attacking *rates* only.  It deliberately stops before
goals, FPL points, or any fixture scaling: the future xPts layer is expected to
combine a PlayerRateProjection with expected minutes and a TeamFixtureProjection.

Empirical-Bayes shrinking uses transparent minutes-equivalent exposure:

    posterior_rate = (current_minutes * current_rate + prior_ess * prior_rate)
                     / (current_minutes + prior_ess)

Penalties are NOT separable from the stored official data
------------------------------------------------------------------------------
The stored current-season and history_past payloads carry ``expected_goals``
only.  There is no per-penalty xG field, no NPxG field, and no penalty xG
component anywhere, so penalty and non-penalty xG cannot be reliably separated.
Penalties therefore stay EMBEDDED in the rate and the fields are named
``xG_per90`` / ``xA_per90`` — never ``NPxG_per90``.  No separate penalty
expectation is added, and ``penalties_order`` (present in the raw bootstrap) is
deliberately not used to move any number.

Prior history discipline
------------------------
``history_past`` xG/xA are season TOTALS, and the official endpoint only
carries them for the 2022/23–2025/26 seasons (older seasons return "0.00" for
the whole xG family, so they are treated as MISSING, not as genuine zeros).
No club identity is stored per season row, so a prior-season rate cannot be
shown to be club-portable; a conservative, configured ESS reduction is applied
instead of inventing a team-ratio adjustment.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, fields
from typing import Any, Iterable, Mapping

from . import analytics, history_completeness as hc, repositories as repo
from .utils import parse_utc, utc_now

PLAYER_RATE_MODEL_VERSION = "player_rates_v1.0.0"
PLAYER_RATE_BASELINE_MODEL_VERSION = "player_rate_baseline_v1.0.0"

COMPONENT_XG = "xG_per90"
COMPONENT_XA = "xA_per90"
COMPONENTS = (COMPONENT_XG, COMPONENT_XA)

# Seasons whose official history_past payloads carry a populated xG family.
# Earlier seasons return "0.00" for expected_goals/expected_assists and are
# treated as missing evidence, never as genuine zeros.
XG_BEARING_SEASONS = ("2025/26", "2024/25", "2023/24", "2022/23")

COMPONENT_HISTORY_FIELD = {
    COMPONENT_XG: "expected_goals",
    COMPONENT_XA: "expected_assists",
}

# Structured role-change signals only; no free-form text ever becomes a number.
# severity -> ESS discount factor.  The strongest (smallest) applicable factor
# is used, never a product, so correlated signals cannot compound.
ROLE_ESS_RULES: dict[tuple[str, str], tuple[float, str]] = {
    ("role_change", "suspected"): (0.5, "ROLE_CHANGE_SUSPECTED"),
    ("role_change", "confirmed"): (0.25, "ROLE_CHANGE_CONFIRMED"),
    # Existing structured proxy (same family the Minutes model treats as a
    # weakened start role): the established role is not secure.
    ("role_security_5gw", "low"): (0.5, "ROLE_CHANGE_SUSPECTED"),
    ("role_security_5gw", "very_low"): (0.5, "ROLE_CHANGE_SUSPECTED"),
    ("competition_for_position", "high"): (0.5, "ROLE_CHANGE_SUSPECTED"),
    ("competition_for_position", "very_high"): (0.5, "ROLE_CHANGE_SUSPECTED"),
}

_ROLE_BANDS = {"suspected", "confirmed", "low", "very_low", "high", "very_high"}


@dataclass(frozen=True)
class PlayerRatesConfig:
    """Every tunable in one versioned structure; no magic numbers in code."""

    # Minutes-equivalent prior strength.  Distinct per component on purpose.
    xg_prior_ess_minutes: float = 850.0
    xa_prior_ess_minutes: float = 1100.0

    prior_season: str = "2025/26"
    multi_season_decay: float = 0.6
    min_prior_minutes_for_same_player: float = 450.0
    low_prior_evidence_flag_minutes: float = 450.0

    # Role change alters TRUST in the prior (ESS), never the rate directly.
    role_change_ess_suspected: float = 0.5
    role_change_ess_confirmed: float = 0.25
    # Club identity is not stored per history season, so portability is
    # unverifiable; a conservative ESS reduction replaces a team-ratio guess.
    club_portability_ess_discount: float = 0.85

    def component_ess(self, component: str) -> float:
        if component == COMPONENT_XG:
            return float(self.xg_prior_ess_minutes)
        if component == COMPONENT_XA:
            return float(self.xa_prior_ess_minutes)
        raise ValueError(f"unsupported component: {component}")

    def config_hash(self) -> str:
        values = {item.name: getattr(self, item.name) for item in fields(self)}
        return analytics.canonical_hash({"model": PLAYER_RATE_MODEL_VERSION, **values})


# ---------------------------------------------------------------------------
# Historical prior evidence.
# ---------------------------------------------------------------------------


def _history_value(raw_json: str | None, field_name: str) -> float | None:
    if not raw_json:
        return None
    try:
        data = json.loads(raw_json)
    except (TypeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    value = data.get(field_name)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _historical_rows(conn: sqlite3.Connection, player_id: int) -> dict[str, dict[str, Any]]:
    """history_past rows by season with parsed xG-family totals."""

    out: dict[str, dict[str, Any]] = {}
    for row in repo.player_season_histories(conn, int(player_id)):
        season = str(row.get("season_name"))
        out[season] = {
            "season_name": season,
            "minutes": row.get("minutes"),
            "expected_goals": _history_value(row.get("raw_json"), "expected_goals"),
            "expected_assists": _history_value(row.get("raw_json"), "expected_assists"),
        }
    return out


def pooled_rates(conn: sqlite3.Connection, config: PlayerRatesConfig) -> dict[str, Any]:
    """Pooled per-90 rates from xG-bearing seasons, league-wide and by position.

    One pass over history rows for both components.  The player's CURRENT
    position is used (historical position is not stored), which the caller
    flags.  Computed once per build instead of once per player.
    """

    league = {component: {"total": 0.0, "minutes": 0.0} for component in COMPONENTS}
    by_position: dict[int, dict[str, dict[str, float]]] = {}
    rows = conn.execute(
        """SELECT h.minutes, h.raw_json, p.element_type
             FROM player_season_histories h JOIN players p ON p.id=h.player_id
            WHERE h.season_name IN (%s) AND h.minutes > 0"""
        % ",".join("?" for _ in XG_BEARING_SEASONS),
        tuple(XG_BEARING_SEASONS),
    ).fetchall()
    for row in rows:
        minutes = float(row["minutes"])
        element_type = row["element_type"]
        for component in COMPONENTS:
            value = _history_value(row["raw_json"], COMPONENT_HISTORY_FIELD[component])
            if value is None:
                continue
            league[component]["total"] += value
            league[component]["minutes"] += minutes
            if element_type is not None:
                bucket = by_position.setdefault(int(element_type), {c: {"total": 0.0, "minutes": 0.0} for c in COMPONENTS})
                bucket[component]["total"] += value
                bucket[component]["minutes"] += minutes

    def finalise(bucket: dict[str, float]) -> dict[str, Any]:
        rate = (bucket["total"] / bucket["minutes"] * 90.0) if bucket["minutes"] > 0 else None
        return {"rate": rate, "minutes": bucket["minutes"], "total": bucket["total"]}

    return {
        "league": {component: finalise(league[component]) for component in COMPONENTS},
        "position": {
            position: {component: finalise(values[component]) for component in COMPONENTS}
            for position, values in by_position.items()
        },
    }


def player_prior(
    conn: sqlite3.Connection,
    player_id: int,
    element_type: int | None,
    component: str,
    config: PlayerRatesConfig,
    pools: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Prior hierarchy: prev-season same-player, multi-season, position, league."""

    pools = pools if pools is not None else pooled_rates(conn, config)
    season_ess = config.component_ess(component)
    field_name = COMPONENT_HISTORY_FIELD[component]
    history = _historical_rows(conn, int(player_id))
    flags: list[str] = []
    ess_scale = 1.0

    # Level 1 — previous-season same-player.
    prev = history.get(config.prior_season)
    if prev and prev.get("minutes") and float(prev["minutes"]) > 0 and prev.get(field_name) is not None:
        minutes = float(prev["minutes"])
        value = float(prev[field_name])
        rate = value / minutes * 90.0
        if minutes < config.min_prior_minutes_for_same_player:
            ess_scale *= minutes / config.min_prior_minutes_for_same_player
            flags.append("TINY_HISTORICAL_SAMPLE")
        if minutes < config.low_prior_evidence_flag_minutes:
            flags.append("LOW_PRIOR_EVIDENCE")
        return {
            "prior_rate": rate,
            "prior_minutes": minutes,
            "prior_total": value,
            "prior_source": "prev_season_same_player",
            "prior_season": config.prior_season,
            "ess_scale": ess_scale,
            "same_player_history": True,
            "flags": flags,
        }

    # Level 2 — multi-season same-player across xG-bearing seasons.
    weighted_value = weighted_minutes = total_minutes = 0.0
    seasons_used: list[str] = []
    for index, season in enumerate(XG_BEARING_SEASONS):
        row = history.get(season)
        if not row or not row.get("minutes") or float(row["minutes"]) <= 0 or row.get(field_name) is None:
            continue
        weight = config.multi_season_decay**index
        weighted_value += weight * float(row[field_name])
        weighted_minutes += weight * float(row["minutes"])
        total_minutes += float(row["minutes"])
        seasons_used.append(season)
    if weighted_minutes > 0 and seasons_used:
        rate = weighted_value / weighted_minutes * 90.0
        ess_scale = min(1.0, total_minutes / config.min_prior_minutes_for_same_player)
        if total_minutes < config.low_prior_evidence_flag_minutes:
            flags.append("LOW_PRIOR_EVIDENCE")
        if ess_scale < 1.0:
            flags.append("TINY_HISTORICAL_SAMPLE")
        flags.append("MULTI_SEASON_PRIOR")
        return {
            "prior_rate": rate,
            "prior_minutes": total_minutes,
            "prior_total": weighted_value,
            "prior_source": "multi_season_same_player",
            "prior_season": None,
            "seasons_used": seasons_used,
            "ess_scale": ess_scale,
            "same_player_history": True,
            "flags": flags,
        }

    # Level 3 — position-pooled prior.
    pooled = (pools.get("position") or {}).get(int(element_type), {}).get(component) if element_type is not None else None
    if pooled and pooled.get("rate") is not None:
        flags.append("POOLED_PRIOR")
        flags.append("NO_HISTORICAL_PLAYER_PRIOR")
        flags.append("POSITION_FROM_CURRENT_SQUAD")
        return {
            "prior_rate": pooled["rate"],
            "prior_minutes": pooled["minutes"],
            "prior_total": pooled["total"],
            "prior_source": "position_pooled",
            "prior_season": None,
            "ess_scale": 1.0,
            "same_player_history": False,
            "flags": flags,
        }

    # Level 4 — league-pooled prior.
    league = (pools.get("league") or {}).get(component) or {"rate": None, "minutes": 0.0, "total": 0.0}
    flags.extend(["POOLED_PRIOR", "NO_HISTORICAL_PLAYER_PRIOR", "LEAGUE_POOLED_PRIOR"])
    return {
        "prior_rate": league["rate"],
        "prior_minutes": league["minutes"],
        "prior_total": league["total"],
        "prior_source": "league_pooled",
        "prior_season": None,
        "ess_scale": 1.0,
        "same_player_history": False,
        "flags": flags,
    }


# ---------------------------------------------------------------------------
# Current-season evidence.
# ---------------------------------------------------------------------------


def current_rate_evidence(
    conn: sqlite3.Connection,
    player_id: int,
    component: str,
    planning_event: int,
    cutoff: str,
) -> dict[str, Any]:
    """Completed player-fixture exposure and xG/xA totals before the cutoff.

    Only completed fixtures (shared ``fixtures.finished=1`` boundary, matching
    the Minutes model) with minutes > 0 contribute exposure.  A DGW contributes
    both fixtures independently.  A played row with missing xG/xA is excluded
    and flagged, never silently counted as zero.

    A completed-fixture row that carries no official observation at all is a
    stale schedule placeholder.  It contributes no exposure (it holds none) but
    it is REPORTED, never silently dropped: silence here would shrink the fitted
    exposure while the bundle still claimed to be certified history.
    """

    field_name = COMPONENT_HISTORY_FIELD[component]
    rows = analytics.completed_rows_as_of(conn, int(player_id), cutoff, int(planning_event))
    minutes = 0.0
    total = 0.0
    played_rows = 0
    missing = 0
    placeholders: list[dict[str, Any]] = []
    fixtures: list[int] = []
    for row in rows:
        if row.get("history_placeholder"):
            placeholders.append(
                {"event": row.get("event"), "fixture_id": row.get("fixture_id")}
            )
            continue
        row_minutes = row.get("minutes")
        if row_minutes is None or float(row_minutes) <= 0:
            continue
        played_rows += 1
        value = row.get(field_name)
        if value is None:
            missing += 1
            continue
        minutes += float(row_minutes)
        total += float(value)
        fixtures.append(int(row["fixture_id"]))
    flags: list[str] = []
    if missing:
        flags.append("CURRENT_XG_DATA_GAP")
    if placeholders:
        flags.append(hc.DIAG_COMPLETED_EVENT_PLACEHOLDER_ROW)
    rate = (total / minutes * 90.0) if minutes > 0 else None
    return {
        "current_minutes": minutes,
        "current_total": total,
        "current_rate": rate,
        "played_rows": played_rows,
        "fixtures": fixtures,
        "missing_xg_rows": missing,
        "placeholder_rows": sorted(
            placeholders, key=lambda item: (item["event"] or 0, item["fixture_id"] or 0)
        ),
        "flags": flags,
    }


# ---------------------------------------------------------------------------
# Role-change ESS handling.
# ---------------------------------------------------------------------------


def role_change_modifiers(
    notes: Iterable[Mapping[str, Any]],
    cutoff: str,
    config: PlayerRatesConfig,
    raw_prior_ess: float,
) -> tuple[float, list[dict[str, Any]], list[str]]:
    """Strongest structured role-change discount, with full provenance."""

    factor = 1.0
    records: list[dict[str, Any]] = []
    flags: list[str] = []
    cutoff_dt = parse_utc(cutoff)
    for note in sorted(notes, key=lambda row: str(row.get("observed_at") or "")):
        observed = parse_utc(note.get("observed_at"))
        if observed is None or cutoff_dt is None or observed > cutoff_dt:
            continue
        expires = parse_utc(note.get("expires_at"))
        if expires is not None and cutoff_dt is not None and expires < cutoff_dt:
            continue
        key = str(note.get("key") or "")
        band = str(note.get("value_text") or "").strip().lower()
        if band not in _ROLE_BANDS:
            continue
        rule = ROLE_ESS_RULES.get((key, band))
        if rule is None:
            continue
        discount, flag = rule
        if discount < factor:
            factor = discount
        records.append(
            {
                "id": f"scouting_note:{key}:{band}:note{note.get('id')}",
                "source": "scouting_note",
                "observed_at": note.get("observed_at"),
                "expires_at": note.get("expires_at"),
                "scope": "prior_ess",
                "signal": f"{key}={band}",
                "discount": discount,
                "raw_prior_ess": round(raw_prior_ess, 6),
                "adjusted_prior_ess": round(raw_prior_ess * discount, 6),
                "flag": flag,
            }
        )
        flags.append(flag)
    return factor, records, sorted(set(flags))


# ---------------------------------------------------------------------------
# Projection build.
# ---------------------------------------------------------------------------


def build_player_rate_projections(
    conn: sqlite3.Connection,
    planning_event: int,
    cutoff: str,
    config: PlayerRatesConfig | None = None,
) -> list[dict[str, Any]]:
    """One PlayerRateProjection per (player, component) for every active player."""

    config = config or PlayerRatesConfig()
    generated_at = utc_now()
    pools = pooled_rates(conn, config)
    out: list[dict[str, Any]] = []
    for player in sorted(analytics.projectable_players(conn), key=lambda row: int(row["player_id"])):
        player_id = int(player["player_id"])
        element_type = player.get("element_type")
        notes = repo.scouting_current_rows_as_of(conn, cutoff, [player_id])
        for component in COMPONENTS:
            out.append(
                _project_component(
                    conn, player_id, player.get("team_id"), element_type, component, planning_event, cutoff, config,
                    generated_at, notes, pools,
                )
            )
    return out


def _project_component(
    conn: sqlite3.Connection,
    player_id: int,
    team_id: int | None,
    element_type: int | None,
    component: str,
    planning_event: int,
    cutoff: str,
    config: PlayerRatesConfig,
    generated_at: str,
    notes: list[dict[str, Any]],
    pools: dict[str, Any],
) -> dict[str, Any]:
    prior = player_prior(conn, player_id, element_type, component, config, pools)
    evidence = current_rate_evidence(conn, player_id, component, planning_event, cutoff)
    raw_prior_ess = config.component_ess(component) * float(prior.get("ess_scale", 1.0))

    discount, modifier_records, role_flags = role_change_modifiers(notes, cutoff, config, raw_prior_ess)
    prior_ess = raw_prior_ess * discount
    portability_discount = 1.0
    flags = list(prior["flags"]) + list(evidence["flags"])

    if prior.get("same_player_history"):
        portability_discount = config.club_portability_ess_discount
        prior_ess *= portability_discount
        flags.append("CLUB_PORTABILITY_UNVERIFIED")
    if discount < 1.0:
        flags.extend(role_flags)

    prior_rate = prior["prior_rate"]
    current_rate = evidence["current_rate"]
    current_minutes = float(evidence["current_minutes"])
    data_gaps: list[str] = []

    if prior_rate is None and current_rate is None:
        posterior = None
        data_gaps.append("no prior and no current evidence; rate unavailable")
        flags.append("NO_PRIOR_AND_NO_CURRENT_EVIDENCE")
    elif prior_rate is None:
        posterior = current_rate
        flags.append("NO_PRIOR")
    elif current_minutes <= 0 or current_rate is None:
        posterior = prior_rate
        flags.append("NO_CURRENT_EVIDENCE")
    else:
        posterior = (current_minutes * current_rate + prior_ess * prior_rate) / (current_minutes + prior_ess)

    posterior_ess = current_minutes + prior_ess
    current_share = current_minutes / posterior_ess if posterior_ess > 0 else 0.0
    prior_share = prior_ess / posterior_ess if posterior_ess > 0 else 0.0

    if posterior is not None and posterior < 0.0:
        flags.append("NEGATIVE_RATE")
    if current_minutes < 0.0:
        flags.append("INVALID_MINUTES")
    if prior_rate is not None and prior_rate < 0.0:
        flags.append("CORRUPT_HISTORICAL_FIELD")
    if prior.get("prior_source") in {"position_pooled", "league_pooled"}:
        data_gaps.append("no same-player historical xG prior; pooled prior used")
    if evidence["placeholder_rows"]:
        # Report the gap instead of letting the exposure shrink in silence.
        data_gaps.append(
            f"{hc.CERTIFIED_PREDICTION_INPUT_HISTORY_INCOMPLETE}: "
            f"{len(evidence['placeholder_rows'])} completed-fixture row(s) carry no official "
            "observation and contribute no exposure"
        )

    return {
        "player_id": int(player_id),
        "component": component,
        "event": int(planning_event),
        "prior_mean": round(prior_rate, 6) if prior_rate is not None else None,
        "prior_ess": round(prior_ess, 6),
        "prior_ess_before_discounts": round(raw_prior_ess, 6),
        "prior_source": prior["prior_source"],
        "prior_season": prior.get("prior_season"),
        "prior_minutes": round(float(prior.get("prior_minutes") or 0.0), 6),
        "current_minutes": round(current_minutes, 6),
        "current_total": round(float(evidence["current_total"]), 6),
        "current_rate": round(current_rate, 6) if current_rate is not None else None,
        "current_played_rows": int(evidence["played_rows"]),
        "posterior_mean": round(posterior, 6) if posterior is not None else None,
        "posterior_ess": round(posterior_ess, 6),
        "posterior_uncertainty": {
            "effective_minutes": round(posterior_ess, 6),
            "prior_share": round(prior_share, 6),
            "current_share": round(current_share, 6),
        },
        "role_modifier_applied": bool(modifier_records),
        "role_modifier_ids": [record["id"] for record in modifier_records],
        "role_modifier_records": modifier_records,
        "club_portability_discount": round(portability_discount, 6),
        "team_context": {"team_id": int(team_id) if team_id is not None else None},
        "risk_flags": sorted(set(flags)),
        "model_version": PLAYER_RATE_MODEL_VERSION,
        "generated_at": generated_at,
        "data_cutoff": cutoff,
        "provenance": {
            "prior_hierarchy": "prev_season_same_player > multi_season_same_player > position_pooled > league_pooled",
            "history_xg_semantics": "season_total",
            "xg_bearing_seasons": list(XG_BEARING_SEASONS),
            "penalties": "embedded_in_xG",
            "npxg_separation": False,
            "penalty_note": (
                "No NPxG or penalty-xG field exists in the stored official data; penalties remain "
                "embedded and the rate is named xG_per90, never NPxG_per90."
            ),
            "club_identity_available": False,
            "role_change_semantics": "reduces prior ESS only; never adds attacking output",
            "evidence_boundary": "completed player-fixture rows (fixtures.finished=1) with minutes>0 before cutoff",
        },
        "data_gaps": data_gaps,
    }


def build_naive_rate_baselines(
    conn: sqlite3.Connection,
    planning_event: int,
    cutoff: str,
    config: PlayerRatesConfig | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """The two simple comparators: RAW_CURRENT_RATE and PRIOR_ONLY_RATE."""

    config = config or PlayerRatesConfig()
    generated_at = utc_now()
    pools = pooled_rates(conn, config)
    raw: list[dict[str, Any]] = []
    prior_only: list[dict[str, Any]] = []
    for player in sorted(analytics.projectable_players(conn), key=lambda row: int(row["player_id"])):
        player_id = int(player["player_id"])
        element_type = player.get("element_type")
        for component in COMPONENTS:
            prior = player_prior(conn, player_id, element_type, component, config, pools)
            evidence = current_rate_evidence(conn, player_id, component, planning_event, cutoff)
            raw.append(
                {
                    "player_id": player_id,
                    "component": component,
                    "event": int(planning_event),
                    "baseline_kind": "RAW_CURRENT_RATE",
                    "rate": evidence["current_rate"],
                    "current_minutes": evidence["current_minutes"],
                    "posterior_mean": evidence["current_rate"],
                    "model_version": PLAYER_RATE_BASELINE_MODEL_VERSION,
                    "generated_at": generated_at,
                    "data_cutoff": cutoff,
                    "data_gaps": (
                        [] if evidence["current_rate"] is not None
                        else ["no completed minutes before the cutoff; raw current rate unavailable"]
                    ),
                    "risk_flags": sorted(set(evidence["flags"])),
                }
            )
            prior_only.append(
                {
                    "player_id": player_id,
                    "component": component,
                    "event": int(planning_event),
                    "baseline_kind": "PRIOR_ONLY_RATE",
                    "rate": prior["prior_rate"],
                    "prior_minutes": prior["prior_minutes"],
                    "posterior_mean": prior["prior_rate"],
                    "prior_source": prior["prior_source"],
                    "model_version": PLAYER_RATE_BASELINE_MODEL_VERSION,
                    "generated_at": generated_at,
                    "data_cutoff": cutoff,
                    "data_gaps": (
                        [] if prior["prior_rate"] is not None
                        else ["no prior evidence available; prior-only rate unavailable"]
                    ),
                    "risk_flags": sorted(set(prior["flags"])),
                }
            )
    return {"RAW_CURRENT_RATE": raw, "PRIOR_ONLY_RATE": prior_only}


# ---------------------------------------------------------------------------
# Analytics readiness gate for player rates.
# ---------------------------------------------------------------------------


def readiness_summary(
    context: Any,
    rows: list[dict[str, Any]],
    *,
    deadline_status: str | None = None,
    data_cutoff: str | None = None,
    deadline: str | None = None,
) -> dict[str, Any]:
    """Player-rate-specific gate; independent of the setup health gate."""

    fail_reasons: list[str] = []
    warn_reasons: list[str] = []
    flag_counts: dict[str, int] = {}
    if context is not None:
        if (getattr(context, "health", None) or {}).get("status") == "FAIL":
            fail_reasons.append("PLANNING_CONTEXT_FAILED: " + "; ".join((context.health or {}).get("fail_reasons") or []))
    if data_cutoff is not None and deadline is not None and deadline_status != "LATE_FREEZE":
        if parse_utc(data_cutoff) > parse_utc(deadline):
            fail_reasons.append("MODEL_INPUTS_AFTER_DEADLINE: a pre-deadline freeze cutoff may not exceed the deadline")
    if not rows:
        fail_reasons.append("NO_PLAYER_RATE_ROWS: no projectable player rate projections")
    for row in rows:
        for flag in row.get("risk_flags", []):
            flag_counts[flag] = flag_counts.get(flag, 0) + 1
        posterior = row.get("posterior_mean")
        if posterior is not None and posterior < 0.0:
            fail_reasons.append(f"NEGATIVE_RATE: player {row.get('player_id')} {row.get('component')}")
        if float(row.get("current_minutes") or 0.0) < 0.0:
            fail_reasons.append(f"INVALID_MINUTES: player {row.get('player_id')}")
        if abs(float(row.get("current_total") or 0.0)) > 1e6:
            fail_reasons.append(f"CORRUPT_CURRENT_FIELD: player {row.get('player_id')}")
    for flag in (
        "NO_HISTORICAL_PLAYER_PRIOR",
        "TINY_HISTORICAL_SAMPLE",
        "ROLE_CHANGE_SUSPECTED",
        "ROLE_CHANGE_CONFIRMED",
        "CLUB_PORTABILITY_UNVERIFIED",
        "POOLED_PRIOR",
        "CURRENT_XG_DATA_GAP",
        "NO_CURRENT_EVIDENCE",
    ):
        if flag_counts.get(flag):
            warn_reasons.append(f"{flag}: {flag_counts[flag]} player-component projections affected")
    status = "FAIL" if fail_reasons else ("WARN" if warn_reasons else "PASS")
    return {
        "status": status,
        "fail_reasons": sorted(set(fail_reasons)),
        "warn_reasons": warn_reasons,
        "counts": {f"rows_{flag.lower()}": count for flag, count in sorted(flag_counts.items())},
    }
