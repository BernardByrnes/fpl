"""PE-7 player attacking-rate challenger — challenger-first, never promotion.

WHAT THIS IS
------------
PE-7 leaves the incumbent player-rate model untouched: ``player_rates_v1.0.0``
and ``player_rate_baseline_v1.0.0`` are its identity, and this module neither
renames nor re-points them, nor writes through them.  A refinement must earn a
promotion at PE-7's terminal boundary; until then it lives here, under its own
identity:

    ``PLAYER_ATTACK_CHALLENGER_VERSION`` + ``PlayerAttackChallengerConfig.config_hash()``

The challenger estimates the SAME quantities the incumbent publishes — an xG/90
and an xA/90 posterior per player per component — and emits rows carrying the
incumbent's own published field contract, so the two arms are scored by identical
code over identical populations.

Penalties stay EMBEDDED in xG.  No penalty or non-penalty xG component is
fabricated here, and the rate stays named ``xG_per90``, never ``NPxG_per90``.

WHAT IT CHANGES
---------------
Three declared refinement families, each inside PE-7's scope list, each ablated
on its own arm.

*Family 1 — ``data_derived_prior_strength``.*  The incumbent shrinks a player's
own xG/90 and xA/90 toward his prior with a CONSTANT, declared prior strength
(850 and 1100 minutes), whatever the cutoff's evidence says about how much
players differ or how noisy a per-90 rate is.  The challenger estimates the prior
strength from the evidence by method of moments: the per-minute variance of the
attacking value is estimated from the split-half difference of each player's own
matches, the between-player variance of true rates is what remains of the
observed spread once that sampling variance is removed, and the implied ESS is
``90 * sigma_per_minute / between_player_variance`` — the weight an
empirical-Bayes posterior puts on a prior.  A league whose players differ little
is shrunk hard; one whose players differ a lot is shrunk little.  Where the window
cannot support the estimate (too few players, a non-positive between-player
variance) the incumbent's declared constant stands, and the fallback and its
reason are recorded.

*Family 2 — ``role_segmented_exposure``.*  The incumbent sums a player's whole
current-season exposure into ONE current-rate estimate and discounts only the
PRIOR when a structured role signal exists.  A row completed before the signal
describes the OLD role, so pooling it with post-signal rows answers a question
about a role the player no longer has.  The challenger segments the player's own
rows at the latest cutoff-observable structured role signal: post-signal rows are
the current-role evidence, and pre-signal rows are folded into the player's own
prior as same-player evidence (their minutes and their value), so old-role scoring
informs the prior instead of being scored as if it were the new role.  The
boundary is a row's own observability (``updated_at`` under the PE-1 boundary), so
the segmentation is cutoff-safe by construction, and a player with no post-signal
rows is reported as role-changed with no current-role evidence rather than scored
on the old role.

*Family 3 — ``pooled_prior_role_alignment``.*  Under a CONFIRMED structured role
change — the same structured signal the incumbent already trusts as "confirmed" —
a previous-season same-player rate was earned in a role the player no longer
plays, so it is not a forecast of this one.  The challenger replaces the prior
MEAN for such a player with the position-pooled mean (the same pool, from the same
canonical boundary) and records the mean it replaced.  A merely SUSPECTED change
keeps the incumbent's treatment.  Precedence is declared: where family 2 also
applies, the confirmed-change alignment governs the prior mean and the segmented
pre-signal evidence is reported without entering it.

CUTOFF-STABLE IDENTITY AND CUTOFF-STABLE PRIORS ARE UNCONDITIONAL
-----------------------------------------------------------------
These are contract requirements, not families, and they are what let the
challenger answer "a later transfer or position change cannot alter an earlier
prior":

* the candidate pool, the club and the position come from the cutoff's accepted
  official bootstrap generation (the PE-6 identity resolution, reused rather than
  restated).  ``players.team_id`` / ``players.element_type`` / ``players.is_active``
  are never read for a candidate's identity;
* the league and position POOLS are built here from the same canonical PE-1
  boundary, attributing every pooled history row by the identity the CUTOFF
  resolves.  The incumbent's ``pooled_rates`` joins those rows to the PERSISTED
  players row — its frozen behaviour, not rewritten, and not on this path;
* a prior-season row is evidence only if it was OBSERVABLE at the cutoff
  (``observed_at <= cutoff``), and the league/position POOLS obey the same rule, so
  a post-cutoff season write cannot move a prior through the back door of a pooled
  mean.  A row with no observation time cannot be shown to have been observable, so
  it is excluded too, and every count is published rather than a newer value being
  silently substituted.  The direction is conservative and worth stating: on a store
  that keeps ONE row per (player, season), a post-cutoff refresh of an already
  observed row makes that row invisible to earlier cutoffs rather than admitting its
  post-cutoff content — the read can only lose evidence, never gain it;
* xG-family fields a season does not carry stay MISSING, never zero: an older
  season's "0.00" is not a genuine zero, and only the declared xG-bearing seasons
  can contribute a same-player prior at all.

THE EQUIVALENCE PROPERTY
------------------------
With an EMPTY family set, on a store whose prior-season rows were all observable
at the cutoff, the challenger reproduces the incumbent's prior choice, its ESS
scaling and its posterior mean from the same evidence.  A test pins it.
"""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass, field, fields
from typing import Any, Iterable, Mapping, Sequence

from . import analytics
from . import history_completeness as hc
from . import player_rates as incumbent
from . import repositories as repo
from . import walk_forward as wf
from .utils import utc_now

# v0.1.0: initial PE-7 player-rate challenger.  No promotion is performed here.
PLAYER_ATTACK_CHALLENGER_VERSION = "player_attack_challenger_v0.1.0"
PLAYER_ATTACK_CHALLENGER_FAMILY = "player_attack_challenger_pe7"

FAMILY_DATA_DERIVED_PRIOR_STRENGTH = "data_derived_prior_strength"
FAMILY_ROLE_SEGMENTED_EXPOSURE = "role_segmented_exposure"
FAMILY_POOLED_PRIOR_ROLE_ALIGNMENT = "pooled_prior_role_alignment"

REFINEMENT_FAMILIES: tuple[str, ...] = (
    FAMILY_DATA_DERIVED_PRIOR_STRENGTH,
    FAMILY_ROLE_SEGMENTED_EXPOSURE,
    FAMILY_POOLED_PRIOR_ROLE_ALIGNMENT,
)

ARM_ALL_REFINEMENTS = "challenger_all_refinements"

FROZEN_PLAYER_INCUMBENT_VERSIONS: dict[str, str] = {
    "PLAYER_RATE_MODEL_VERSION": "player_rates_v1.0.0",
    "PLAYER_RATE_BASELINE_MODEL_VERSION": "player_rate_baseline_v1.0.0",
}

ESS_BASIS_SPLIT_HALF_METHOD_OF_MOMENTS = "SPLIT_HALF_METHOD_OF_MOMENTS"
ESS_BASIS_INCUMBENT_FALLBACK = "INCUMBENT_DECLARED_CONSTANT_FALLBACK"

ESS_REASON_TOO_FEW_PLAYERS = "TOO_FEW_PLAYERS_WITH_A_RATE"
ESS_REASON_TOO_FEW_SPLIT_PLAYERS = "TOO_FEW_PLAYERS_WITH_TWO_HALVES"
ESS_REASON_NON_POSITIVE_BETWEEN_VARIANCE = "NON_POSITIVE_BETWEEN_PLAYER_VARIANCE"
ESS_REASON_NO_SAMPLING_VARIANCE = "NO_POSITIVE_SAMPLING_VARIANCE"

STRENGTH_DERIVED = "DERIVED_FROM_THE_CUTOFFS_OWN_EVIDENCE"
STRENGTH_DERIVED_CLAMPED = "DERIVED_AND_CLAMPED_TO_THE_DECLARED_BOUND"
STRENGTH_FALLBACK = "DECLARED_CONSTANT_FALLBACK"

#: The challenger reads position from the cutoff identity, never from the live row.
POSITION_BASIS = "POSITION_FROM_CUTOFF_IDENTITY"

FLAG_DERIVED_PRIOR_ESS = "DERIVED_PRIOR_ESS"
FLAG_DERIVED_PRIOR_ESS_UNAVAILABLE = "DERIVED_PRIOR_ESS_UNAVAILABLE"
FLAG_DERIVED_PRIOR_ESS_CLAMPED = "DERIVED_PRIOR_ESS_CLAMPED"
FLAG_DERIVED_PRIOR_ESS_ABOVE_DECLARED = "DERIVED_PRIOR_ESS_ABOVE_DECLARED"
FLAG_DERIVED_PRIOR_ESS_BELOW_DECLARED = "DERIVED_PRIOR_ESS_BELOW_DECLARED"
FLAG_ROLE_SEGMENTED_EXPOSURE = "ROLE_SEGMENTED_EXPOSURE"
FLAG_ROLE_CHANGE_NO_POST_SIGNAL_EVIDENCE = "ROLE_CHANGE_NO_POST_SIGNAL_EVIDENCE"
FLAG_PRIOR_MEAN_REPLACED = "PRIOR_MEAN_REPLACED_UNDER_CONFIRMED_ROLE_CHANGE"
FLAG_PRIOR_MEAN_REPLACEMENT_UNAVAILABLE = "PRIOR_MEAN_REPLACEMENT_UNAVAILABLE"
FLAG_POST_CUTOFF_HISTORY_EXCLUDED = "POST_CUTOFF_HISTORY_EXCLUDED"
FLAG_UNDATED_HISTORY_EXCLUDED = "UNDATED_HISTORY_ROW_EXCLUDED"
FLAG_NO_CUTOFF_PRIOR_SEASON = "NO_CUTOFF_OBSERVABLE_PRIOR_SEASON_ROW"

#: The incumbent's own flag names, reused so an ablation's delta stays
#: attributable to a family rather than to a re-worded vocabulary.
INCUMBENT_FLAGS_PRESERVED: tuple[str, ...] = (
    "NEGATIVE_RATE",
    "INVALID_MINUTES",
    "CORRUPT_HISTORICAL_FIELD",
    "NO_PRIOR",
    "NO_CURRENT_EVIDENCE",
    "NO_PRIOR_AND_NO_CURRENT_EVIDENCE",
    "LOW_PRIOR_EVIDENCE",
    "TINY_HISTORICAL_SAMPLE",
    "MULTI_SEASON_PRIOR",
    "POOLED_PRIOR",
    "NO_HISTORICAL_PLAYER_PRIOR",
    "CURRENT_XG_DATA_GAP",
    "CLUB_PORTABILITY_UNVERIFIED",
    hc.DIAG_COMPLETED_EVENT_PLACEHOLDER_ROW,
)


class ChallengerInconsistencyError(RuntimeError):
    """The challenger and the incumbent disagree about the evidence they share."""


# ---------------------------------------------------------------------------
# Configuration and identity.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlayerAttackChallengerConfig:
    """Every challenger knob in one versioned structure; no magic numbers.

    ``None`` means "inherit the incumbent's declared value", so the challenger
    introduces no second knob for a quantity the incumbent already declares.
    """

    # --- inherited (None -> the incumbent's PlayerRatesConfig) --------------
    xg_prior_ess_minutes: float | None = None
    xa_prior_ess_minutes: float | None = None
    prior_season: str | None = None
    multi_season_decay: float | None = None
    min_prior_minutes_for_same_player: float | None = None
    low_prior_evidence_flag_minutes: float | None = None
    role_change_ess_suspected: float | None = None
    role_change_ess_confirmed: float | None = None
    club_portability_ess_discount: float | None = None

    # --- declared disclosure policy for the derived prior strength ----------
    #: Players carrying at least one played row required before a derived ESS is
    #: used at all.  A variance estimated from a handful of players is not
    #: evidence about the league, so below this the incumbent's declared ESS
    #: stands and the fallback says so.
    ess_min_players: int = 25
    #: Players needed with TWO usable halves before the sampling variance, and
    #: with it the whole estimate, can be formed.
    ess_min_split_players: int = 10
    #: A hard declared bound on the derived prior strength: beyond it the prior
    #: stops being evidence-weighted and becomes an assertion, so the value is
    #: clamped and the clamp is reported rather than hidden.
    ess_max_minutes: float = 3000.0

    def resolved(self, incumbent_config: "incumbent.PlayerRatesConfig") -> dict[str, Any]:
        """The effective values, with ``None`` inheriting from the incumbent."""

        def pick(value: Any, fallback: Any) -> Any:
            return fallback if value is None else value

        return {
            "xg_prior_ess_minutes": float(
                pick(self.xg_prior_ess_minutes, incumbent_config.xg_prior_ess_minutes)
            ),
            "xa_prior_ess_minutes": float(
                pick(self.xa_prior_ess_minutes, incumbent_config.xa_prior_ess_minutes)
            ),
            "prior_season": str(pick(self.prior_season, incumbent_config.prior_season)),
            "multi_season_decay": float(pick(self.multi_season_decay, incumbent_config.multi_season_decay)),
            "min_prior_minutes_for_same_player": float(
                pick(self.min_prior_minutes_for_same_player, incumbent_config.min_prior_minutes_for_same_player)
            ),
            "low_prior_evidence_flag_minutes": float(
                pick(self.low_prior_evidence_flag_minutes, incumbent_config.low_prior_evidence_flag_minutes)
            ),
            "role_change_ess_suspected": float(
                pick(self.role_change_ess_suspected, incumbent_config.role_change_ess_suspected)
            ),
            "role_change_ess_confirmed": float(
                pick(self.role_change_ess_confirmed, incumbent_config.role_change_ess_confirmed)
            ),
            "club_portability_ess_discount": float(
                pick(self.club_portability_ess_discount, incumbent_config.club_portability_ess_discount)
            ),
            "ess_min_players": int(self.ess_min_players),
            "ess_min_split_players": int(self.ess_min_split_players),
            "ess_max_minutes": float(self.ess_max_minutes),
        }

    def declared_ess(self, component: str) -> float:
        """The incumbent's declared ESS for a component, as this config reads it."""

        fallback = incumbent.PlayerRatesConfig()
        if component == incumbent.COMPONENT_XG:
            value = self.xg_prior_ess_minutes
            return float(fallback.xg_prior_ess_minutes if value is None else value)
        if component == incumbent.COMPONENT_XA:
            value = self.xa_prior_ess_minutes
            return float(fallback.xa_prior_ess_minutes if value is None else value)
        raise ValueError(f"unsupported component: {component}")

    def config_hash(self) -> str:
        values = {item.name: getattr(self, item.name) for item in fields(self)}
        return analytics.canonical_hash({"model": PLAYER_ATTACK_CHALLENGER_VERSION, **values})


def resolved_ess(resolved: Mapping[str, Any], component: str) -> float:
    """The declared ESS inside a ``resolved`` block."""

    if component == incumbent.COMPONENT_XG:
        return float(resolved["xg_prior_ess_minutes"])
    if component == incumbent.COMPONENT_XA:
        return float(resolved["xa_prior_ess_minutes"])
    raise ValueError(f"unsupported component: {component}")


def frozen_incumbent_identity() -> dict[str, Any]:
    """The incumbent player-rate identities, verified against their literals."""

    actual = {name: str(getattr(incumbent, name)) for name in FROZEN_PLAYER_INCUMBENT_VERSIONS}
    mismatches = {
        name: {"declared": declared, "actual": actual.get(name)}
        for name, declared in FROZEN_PLAYER_INCUMBENT_VERSIONS.items()
        if actual.get(name) != declared
    }
    return {
        "values": actual,
        "declared": dict(FROZEN_PLAYER_INCUMBENT_VERSIONS),
        "unchanged": not mismatches,
        "mismatches": mismatches,
    }


def challenger_identity(
    config: PlayerAttackChallengerConfig | None = None,
    incumbent_config: "incumbent.PlayerRatesConfig | None" = None,
) -> dict[str, Any]:
    """The challenger's own identity, carried by every artifact it produces."""

    config = config or PlayerAttackChallengerConfig()
    incumbent_config = incumbent_config or incumbent.PlayerRatesConfig()
    return {
        "family": PLAYER_ATTACK_CHALLENGER_FAMILY,
        "challenger_version": PLAYER_ATTACK_CHALLENGER_VERSION,
        "challenger_config_hash": config.config_hash(),
        "resolved_parameters": config.resolved(incumbent_config),
        "refinement_families": list(REFINEMENT_FAMILIES),
        "incumbent_identity": frozen_incumbent_identity(),
        "incumbent_config_hash": incumbent_config.config_hash(),
        "penalties": "embedded_in_xG",
        "npxg_separation": False,
        "promotion": "NOT_PERFORMED_CHALLENGER_ONLY",
    }


def arm_name_for_family(family: str) -> str:
    return f"challenger_only::{family}"


def default_arm_definitions() -> dict[str, frozenset[str]]:
    """The full challenger plus one single-family ablation per refinement."""

    definitions: dict[str, frozenset[str]] = {
        ARM_ALL_REFINEMENTS: frozenset(REFINEMENT_FAMILIES)
    }
    for family in REFINEMENT_FAMILIES:
        definitions[arm_name_for_family(family)] = frozenset({family})
    return definitions


# ---------------------------------------------------------------------------
# Cutoff-observable prior-season history.
# ---------------------------------------------------------------------------


def _history_value(raw_json: str | None, field_name: str) -> float | None:
    """One xG-family value from a history payload; missing stays missing."""

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


def historical_season_rows_as_of(
    conn: sqlite3.Connection, player_id: int, cutoff: str
) -> dict[str, Any]:
    """Prior-season rows OBSERVABLE at ``cutoff``, plus what was excluded.

    ``player_season_histories`` carries an observation time per row, so a row
    written after the cutoff is not evidence at the cutoff.  A row with no
    observation time cannot be shown to have been observable either, so it is
    excluded and counted rather than assumed to predate the cutoff: silently
    preferring the older of two values is how a backdated rewrite would slip
    through.

    This is not a second historical-observation predicate.  PE-1's boundary is the
    player-fixture observation boundary; this is the same as-of discipline applied
    to a different table, keyed on that table's own observation time.
    """

    seasons: dict[str, dict[str, Any]] = {}
    post_cutoff = 0
    undated = 0
    for row in repo.player_season_histories(conn, int(player_id)):
        observed = row.get("observed_at")
        if observed is None or not str(observed).strip():
            undated += 1
            continue
        if str(observed) > str(cutoff):
            post_cutoff += 1
            continue
        season = str(row.get("season_name"))
        seasons[season] = {
            "season_name": season,
            "minutes": row.get("minutes"),
            "observed_at": str(observed),
            "expected_goals": _history_value(row.get("raw_json"), "expected_goals"),
            "expected_assists": _history_value(row.get("raw_json"), "expected_assists"),
        }
    return {
        "seasons": seasons,
        "excluded": {"post_cutoff_rows": post_cutoff, "undated_rows": undated},
        "rows_visible": sorted(seasons),
    }


# ---------------------------------------------------------------------------
# Cutoff-stable pools.
# ---------------------------------------------------------------------------


def cutoff_stable_rate_pools(
    conn: sqlite3.Connection,
    identities: Mapping[int, tuple[int, int]],
    components: Sequence[str] = incumbent.COMPONENTS,
    *,
    cutoff: str,
    seasons: Sequence[str] = incumbent.XG_BEARING_SEASONS,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """League and position pools attributed by the identity resolved at the cutoff.

    ``player_rates.pooled_rates`` joins every pooled history row to the PERSISTED
    ``players`` row for its position, so a later position change silently moves a
    historical pool.  That read is the incumbent's frozen behaviour and is not
    rewritten; this builder computes the same two tables — same seasons, same
    xG-bearing rule, same per-90 formula, same shape — over the cutoff-resolved
    identity, and hands the result to the incumbent's own prior machinery through
    its ``pools`` argument.

    ``cutoff`` is REQUIRED and there is no "now": a pool is prior evidence, so the
    rows it pools obey the same as-of rule as a player's own prior-season rows.  A
    row written after the cutoff, or one with no observation time at all, is not
    pooled and is COUNTED, and a row whose player has no cutoff identity is not
    pooled (the incumbent's own inner join drops such rows too) and is counted as
    well, so the share of the window each exclusion affects is visible rather than
    implied.
    """

    cutoff = str(cutoff)
    league = {component: {"total": 0.0, "minutes": 0.0} for component in components}
    by_position: dict[int, dict[str, dict[str, float]]] = {}
    rows = repo.player_season_histories(conn, None)
    excluded_horizon = 0
    excluded_post_cutoff = 0
    excluded_undated = 0
    unattributed_rows = 0
    pooled_players: set[int] = set()
    for row in rows:
        if str(row.get("season_name")) not in tuple(seasons):
            excluded_horizon += 1
            continue
        observed = row.get("observed_at")
        if observed is None or not str(observed).strip():
            excluded_undated += 1
            continue
        if str(observed) > cutoff:
            excluded_post_cutoff += 1
            continue
        minutes = row.get("minutes")
        if minutes is None or float(minutes) <= 0:
            continue
        identity = identities.get(int(row["player_id"]))
        if identity is None:
            unattributed_rows += 1
            continue
        position = int(identity[1])
        for component in components:
            value = _history_value(row.get("raw_json"), incumbent.COMPONENT_HISTORY_FIELD[component])
            if value is None:
                continue
            league[component]["total"] += value
            league[component]["minutes"] += float(minutes)
            bucket = by_position.setdefault(
                position, {c: {"total": 0.0, "minutes": 0.0} for c in components}
            )
            bucket[component]["total"] += value
            bucket[component]["minutes"] += float(minutes)
        pooled_players.add(int(row["player_id"]))

    def finalise(bucket: Mapping[str, float]) -> dict[str, Any]:
        rate = (bucket["total"] / bucket["minutes"] * 90.0) if bucket["minutes"] > 0 else None
        return {"rate": rate, "minutes": bucket["minutes"], "total": bucket["total"]}

    pools = {
        "league": {component: finalise(league[component]) for component in components},
        "position": {
            position: {component: finalise(values[component]) for component in components}
            for position, values in by_position.items()
        },
    }
    disclosure = {
        "construction": "PE-7 ADAPTER: cutoff_stable_rate_pools",
        "cutoff": cutoff,
        "identity_source": (
            "the accepted official bootstrap generation at or before the cutoff, per pooled row"
        ),
        "observability_source": "player_season_histories.observed_at <= cutoff",
        "incumbent_pooled_rates_read_used": False,
        "incumbent_read_note": (
            "player_rates.pooled_rates joins its pooled rows to the PERSISTED players row and "
            "applies no as-of filter; that frozen read is not on this path"
        ),
        "history_rows_considered": len(rows),
        "history_rows_outside_xg_bearing_seasons": excluded_horizon,
        "history_rows_written_after_the_cutoff": excluded_post_cutoff,
        "history_rows_without_an_observation_time": excluded_undated,
        "history_rows_without_cutoff_identity": unattributed_rows,
        "pooled_players": len(pooled_players),
        "position_keys": sorted(by_position),
        "xg_bearing_seasons": list(seasons),
    }
    return pools, disclosure


# ---------------------------------------------------------------------------
# Current-season exposure.
# ---------------------------------------------------------------------------


def current_exposure_rows(
    conn: sqlite3.Connection,
    player_id: int,
    component: str,
    planning_event: int,
    cutoff: str,
) -> dict[str, Any]:
    """The player's own completed pre-cutoff rows, one record each.

    The rows come from ``analytics.completed_rows_as_of``, which IS the PE-1
    canonical boundary; this function selects nothing beyond the component's field
    name and the played-minutes rule the incumbent uses.  A played row with a
    missing xG-family value is a DATA GAP: it is excluded and counted, never
    counted as zero.
    """

    field_name = incumbent.COMPONENT_HISTORY_FIELD[component]
    rows = analytics.completed_rows_as_of(conn, int(player_id), cutoff, int(planning_event))
    out: list[dict[str, Any]] = []
    missing = 0
    placeholders: list[dict[str, Any]] = []
    for row in rows:
        if row.get("history_placeholder"):
            placeholders.append({"event": row.get("event"), "fixture_id": row.get("fixture_id")})
            continue
        minutes = row.get("minutes")
        if minutes is None or float(minutes) <= 0:
            continue
        value = row.get(field_name)
        if value is None:
            missing += 1
            continue
        out.append(
            {
                "fixture_id": int(row["fixture_id"]),
                "event": int(row["event"]),
                "minutes": float(minutes),
                "value": float(value),
                "kickoff_time": row.get("fixture_kickoff"),
                "observed_at": row.get("updated_at"),
            }
        )
    out.sort(key=lambda item: (item["event"], str(item["kickoff_time"] or ""), item["fixture_id"]))
    return {"rows": out, "missing_value_rows": missing, "placeholder_rows": placeholders}


def _exposure_block(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    minutes = sum(float(row["minutes"]) for row in rows)
    total = sum(float(row["value"]) for row in rows)
    return {
        "minutes": minutes,
        "total": total,
        "rate": (total / minutes * 90.0) if minutes > 0 else None,
        "rows": len(rows),
        "fixtures": [int(row["fixture_id"]) for row in rows],
    }


# ---------------------------------------------------------------------------
# Prior hierarchy over cutoff-observable history.
# ---------------------------------------------------------------------------


def challenger_player_prior(
    player_id: int,
    element_type: int | None,
    component: str,
    resolved: Mapping[str, Any],
    pools: Mapping[str, Any],
    history: Mapping[str, Any],
) -> dict[str, Any]:
    """The incumbent's prior hierarchy over cutoff-observable history rows.

    Same four levels, same formulas, same flags and the same ESS scaling as
    ``player_rates.player_prior``; the only difference is WHICH rows are visible,
    which is the point: a prior-season row written after the cutoff is not evidence
    at the cutoff.
    """

    field_name = incumbent.COMPONENT_HISTORY_FIELD[component]
    seasons = history.get("seasons") or {}
    excluded = dict(history.get("excluded") or {})
    flags: list[str] = []
    if excluded.get("post_cutoff_rows"):
        flags.append(FLAG_POST_CUTOFF_HISTORY_EXCLUDED)
    if excluded.get("undated_rows"):
        flags.append(FLAG_UNDATED_HISTORY_EXCLUDED)
    prior_season = str(resolved["prior_season"])
    if prior_season not in seasons and (seasons or excluded.get("post_cutoff_rows") or excluded.get("undated_rows")):
        flags.append(FLAG_NO_CUTOFF_PRIOR_SEASON)
    ess_scale = 1.0

    prev = seasons.get(prior_season)
    if prev and prev.get("minutes") and float(prev["minutes"]) > 0 and prev.get(field_name) is not None:
        minutes = float(prev["minutes"])
        value = float(prev[field_name])
        if minutes < float(resolved["min_prior_minutes_for_same_player"]):
            ess_scale *= minutes / float(resolved["min_prior_minutes_for_same_player"])
            flags.append("TINY_HISTORICAL_SAMPLE")
        if minutes < float(resolved["low_prior_evidence_flag_minutes"]):
            flags.append("LOW_PRIOR_EVIDENCE")
        return {
            "prior_rate": value / minutes * 90.0,
            "prior_minutes": minutes,
            "prior_total": value,
            "prior_source": "prev_season_same_player",
            "prior_season": prior_season,
            "ess_scale": ess_scale,
            "same_player_history": True,
            "flags": flags,
            "history_rows_used": 1,
        }

    weighted_value = weighted_minutes = total_minutes = 0.0
    seasons_used: list[str] = []
    for index, season in enumerate(incumbent.XG_BEARING_SEASONS):
        row = seasons.get(season)
        if not row or not row.get("minutes") or float(row["minutes"]) <= 0 or row.get(field_name) is None:
            continue
        weight = float(resolved["multi_season_decay"]) ** index
        weighted_value += weight * float(row[field_name])
        weighted_minutes += weight * float(row["minutes"])
        total_minutes += float(row["minutes"])
        seasons_used.append(season)
    if weighted_minutes > 0 and seasons_used:
        ess_scale = min(1.0, total_minutes / float(resolved["min_prior_minutes_for_same_player"]))
        if total_minutes < float(resolved["low_prior_evidence_flag_minutes"]):
            flags.append("LOW_PRIOR_EVIDENCE")
        if ess_scale < 1.0:
            flags.append("TINY_HISTORICAL_SAMPLE")
        flags.append("MULTI_SEASON_PRIOR")
        return {
            "prior_rate": weighted_value / weighted_minutes * 90.0,
            "prior_minutes": total_minutes,
            "prior_total": weighted_value,
            "prior_source": "multi_season_same_player",
            "prior_season": None,
            "seasons_used": seasons_used,
            "ess_scale": ess_scale,
            "same_player_history": True,
            "flags": flags,
            "history_rows_used": len(seasons_used),
        }

    pooled = (
        (pools.get("position") or {}).get(int(element_type), {}).get(component)
        if element_type is not None
        else None
    )
    if pooled and pooled.get("rate") is not None:
        return {
            "prior_rate": pooled["rate"],
            "prior_minutes": pooled["minutes"],
            "prior_total": pooled["total"],
            "prior_source": "position_pooled",
            "prior_season": None,
            "ess_scale": 1.0,
            "same_player_history": False,
            "flags": [*flags, "POOLED_PRIOR", "NO_HISTORICAL_PLAYER_PRIOR", POSITION_BASIS],
            "history_rows_used": 0,
        }

    league = (pools.get("league") or {}).get(component) or {"rate": None, "minutes": 0.0, "total": 0.0}
    return {
        "prior_rate": league["rate"],
        "prior_minutes": league["minutes"],
        "prior_total": league["total"],
        "prior_source": "league_pooled",
        "prior_season": None,
        "ess_scale": 1.0,
        "same_player_history": False,
        "flags": [
            *flags,
            "POOLED_PRIOR",
            "NO_HISTORICAL_PLAYER_PRIOR",
            "LEAGUE_POOLED_PRIOR",
            POSITION_BASIS,
        ],
        "history_rows_used": 0,
    }


# ---------------------------------------------------------------------------
# Derived prior strength (family 1).
# ---------------------------------------------------------------------------


def _sampling_variance_per_minute(
    per_player_rows: Mapping[int, Sequence[Mapping[str, Any]]]
) -> tuple[float | None, int, int]:
    """Per-minute attacking-value variance from each player's own split halves.

    A player's rows are split interleaved by fixture order (first, third, ...
    against second, fourth, ...), so the split is not confounded with the recency
    of the season.  For a rate expressed per 90 minutes,
    ``Var(rate_half) = 8100 * sigma_per_minute / minutes_half``, so the squared
    difference of the two halves identifies ``sigma_per_minute`` directly.  Only
    players with TWO usable halves contribute, and the count is published.
    """

    estimates: list[float] = []
    for player_id in sorted(per_player_rows):
        rows = list(per_player_rows[player_id])
        first, second = rows[0::2], rows[1::2]
        if not first or not second:
            continue
        minutes_first = sum(float(row["minutes"]) for row in first)
        minutes_second = sum(float(row["minutes"]) for row in second)
        if minutes_first <= 0 or minutes_second <= 0:
            continue
        rate_first = sum(float(row["value"]) for row in first) / minutes_first * 90.0
        rate_second = sum(float(row["value"]) for row in second) / minutes_second * 90.0
        denominator = 8100.0 * (1.0 / minutes_first + 1.0 / minutes_second)
        if denominator <= 0:
            continue
        estimates.append((rate_first - rate_second) ** 2 / denominator)
    if not estimates:
        return None, 0, len(per_player_rows)
    return sum(estimates) / len(estimates), len(estimates), len(per_player_rows)


def derived_prior_strength(
    per_player_rows: Mapping[int, Sequence[Mapping[str, Any]]],
    resolved: Mapping[str, Any],
    *,
    component: str,
    declared_ess: float,
) -> dict[str, Any]:
    """Method-of-moments empirical-Bayes prior strength for one component.

    ``ESS = 90 * sigma_per_minute / between_player_variance``: the weight an
    empirical-Bayes posterior puts on a prior, in minutes of exposure, when the
    prior's mean is drawn from the population the observed players came from.

    Fail closed on a thin window, on a missing sampling variance or on a
    non-positive between-player variance: none of those can answer the question,
    so the incumbent's declared ESS stands and its reason is recorded.  A ZERO
    sampling variance is refused for the same reason rather than used: it would
    imply zero shrinkage, and in a store whose values are rounded to two decimals
    a zero difference between two halves is an artefact of the stored precision,
    not evidence that a per-90 rate is noise-free.
    """

    min_players = int(resolved["ess_min_players"])
    min_split_players = int(resolved["ess_min_split_players"])
    max_minutes = float(resolved["ess_max_minutes"])
    sampling_variance, split_designs, players_with_rows = _sampling_variance_per_minute(per_player_rows)
    rates = [
        (sum(float(row["value"]) for row in rows), sum(float(row["minutes"]) for row in rows))
        for _player_id, rows in sorted(per_player_rows.items())
        if rows
    ]
    rates = [(total, minutes) for total, minutes in rates if minutes > 0]
    base = {
        "component": component,
        "declared_fallback_prior_ess_minutes": round(float(declared_ess), 6),
        "players_with_rows": players_with_rows,
        "players_with_a_rate": len(rates),
        "players_with_two_halves": split_designs,
        "min_players_required": min_players,
        "min_split_players_required": min_split_players,
        "max_prior_ess_minutes": max_minutes,
    }
    if len(rates) < min_players or split_designs < min_split_players:
        return {
            **base,
            "prior_ess_minutes": round(float(declared_ess), 6),
            "basis": ESS_BASIS_INCUMBENT_FALLBACK,
            "status": STRENGTH_FALLBACK,
            "reason": (
                ESS_REASON_TOO_FEW_PLAYERS
                if len(rates) < min_players
                else ESS_REASON_TOO_FEW_SPLIT_PLAYERS
            ),
            "sampling_variance_per_minute": None,
            "between_player_variance": None,
        }
    if sampling_variance is None or sampling_variance <= 0.0 or not math.isfinite(sampling_variance):
        return {
            **base,
            "prior_ess_minutes": round(float(declared_ess), 6),
            "basis": ESS_BASIS_INCUMBENT_FALLBACK,
            "status": STRENGTH_FALLBACK,
            "reason": ESS_REASON_NO_SAMPLING_VARIANCE,
            "sampling_variance_per_minute": None,
            "between_player_variance": None,
        }

    # Exposure-weighted population mean rate, then each player's deviation from
    # it.  The mean squared deviation is the OBSERVED variance; removing the
    # sampling variance it contains leaves the spread of the true rates.
    total_minutes = sum(minutes for _total, minutes in rates)
    mean_rate = sum((total / minutes * 90.0) * (minutes / total_minutes) for total, minutes in rates)
    observed_variance = sum(
        ((total / minutes * 90.0 - mean_rate) ** 2) * (minutes / total_minutes)
        for total, minutes in rates
    )
    expected_sampling = sum(
        (8100.0 * sampling_variance / minutes) * (minutes / total_minutes) for _total, minutes in rates
    )
    between = observed_variance - expected_sampling
    if between <= 0.0:
        return {
            **base,
            "prior_ess_minutes": round(float(declared_ess), 6),
            "basis": ESS_BASIS_INCUMBENT_FALLBACK,
            "status": STRENGTH_FALLBACK,
            "reason": ESS_REASON_NON_POSITIVE_BETWEEN_VARIANCE,
            "sampling_variance_per_minute": round(sampling_variance, 12),
            "observed_rate_variance": round(observed_variance, 12),
            "between_player_variance": round(between, 12),
        }
    raw_ess = 90.0 * sampling_variance / between
    clamped = min(max(raw_ess, 0.0), max_minutes)
    return {
        **base,
        "prior_ess_minutes": round(clamped, 6),
        "basis": ESS_BASIS_SPLIT_HALF_METHOD_OF_MOMENTS,
        "status": STRENGTH_DERIVED if clamped == raw_ess else STRENGTH_DERIVED_CLAMPED,
        "reason": None,
        "sampling_variance_per_minute": round(sampling_variance, 12),
        "observed_rate_variance": round(observed_variance, 12),
        "between_player_variance": round(between, 12),
        "mean_rate": round(mean_rate, 12),
        "unclamped_prior_ess_minutes": round(raw_ess, 6),
    }


def population_exposure(
    conn: sqlite3.Connection,
    component: str,
    planning_event: int,
    cutoff: str,
    player_ids: Sequence[int],
) -> dict[int, list[dict[str, Any]]]:
    """Every candidate's played pre-cutoff rows for one component."""

    out: dict[int, list[dict[str, Any]]] = {}
    for player_id in player_ids:
        block = current_exposure_rows(conn, int(player_id), component, planning_event, cutoff)
        if block["rows"]:
            out[int(player_id)] = list(block["rows"])
    return out


def derived_ess_for(
    config: PlayerAttackChallengerConfig,
    resolved: Mapping[str, Any],
    families: Iterable[str],
    estimates: Mapping[str, Mapping[str, Any]],
    component: str,
) -> float:
    """The ESS the prior weight is computed against, under the active families."""

    if FAMILY_DATA_DERIVED_PRIOR_STRENGTH in frozenset(families) and component in estimates:
        return float(estimates[component]["prior_ess_minutes"])
    return resolved_ess(resolved, component)


# ---------------------------------------------------------------------------
# Projection.
# ---------------------------------------------------------------------------


def _project_component(
    *,
    player_id: int,
    component: str,
    team_id: int,
    element_type: int,
    planning_event: int,
    cutoff: str,
    generated_at: str,
    notes: Sequence[Mapping[str, Any]],
    exposure: Mapping[str, Any],
    prior: Mapping[str, Any],
    history: Mapping[str, Any],
    resolved: Mapping[str, Any],
    incumbent_config: "incumbent.PlayerRatesConfig",
    challenger_config: PlayerAttackChallengerConfig,
    families: frozenset[str],
    ess_estimate: Mapping[str, Any] | None,
    pools: Mapping[str, Any],
) -> dict[str, Any]:
    """One (player, component) row, carrying the incumbent's published contract."""

    flags = list(prior["flags"])
    data_gaps: list[str] = []
    if exposure["missing_value_rows"]:
        flags.append("CURRENT_XG_DATA_GAP")
    if exposure["placeholder_rows"]:
        flags.append(hc.DIAG_COMPLETED_EVENT_PLACEHOLDER_ROW)

    declared_ess = challenger_config.declared_ess(component)
    prior_ess_base = (
        derived_ess_for(challenger_config, resolved, families, {component: ess_estimate or {}}, component)
        if FAMILY_DATA_DERIVED_PRIOR_STRENGTH in families and ess_estimate
        else resolved_ess(resolved, component)
    )
    if FAMILY_DATA_DERIVED_PRIOR_STRENGTH in families and ess_estimate:
        if ess_estimate["status"] == STRENGTH_FALLBACK:
            flags.append(FLAG_DERIVED_PRIOR_ESS_UNAVAILABLE)
        if ess_estimate["status"] == STRENGTH_DERIVED_CLAMPED:
            flags.append(FLAG_DERIVED_PRIOR_ESS_CLAMPED)
        if prior_ess_base > declared_ess:
            flags.append(FLAG_DERIVED_PRIOR_ESS_ABOVE_DECLARED)
        elif prior_ess_base < declared_ess:
            flags.append(FLAG_DERIVED_PRIOR_ESS_BELOW_DECLARED)
        if ess_estimate["basis"] == ESS_BASIS_SPLIT_HALF_METHOD_OF_MOMENTS:
            flags.append(FLAG_DERIVED_PRIOR_ESS)

    prior_ess_before_segmentation = prior_ess_base * float(prior.get("ess_scale", 1.0))

    # The incumbent's own eligibility rule, called rather than restated, with the
    # incumbent's own config: structured keys and bands only, observed at or
    # before the cutoff, unexpired at it, strongest applicable discount.
    discount, role_records, role_flags = incumbent.role_change_modifiers(
        notes, cutoff, incumbent_config, prior_ess_before_segmentation
    )
    observed_times = [str(record["observed_at"]) for record in role_records if record.get("observed_at")]
    boundary = max(observed_times) if observed_times else None
    confirmed = any(record.get("flag") == "ROLE_CHANGE_CONFIRMED" for record in role_records)

    rows = list(exposure["rows"])
    superseded: list[dict[str, Any]] = []
    if FAMILY_ROLE_SEGMENTED_EXPOSURE in families and boundary is not None:
        kept: list[dict[str, Any]] = []
        for row in rows:
            observed = row.get("observed_at")
            if observed is not None and str(observed) <= boundary:
                superseded.append(row)
            else:
                kept.append(row)
        rows = kept
        if superseded:
            flags.append(FLAG_ROLE_SEGMENTED_EXPOSURE)
            if exposure["rows"] and not rows:
                flags.append(FLAG_ROLE_CHANGE_NO_POST_SIGNAL_EVIDENCE)

    current = _exposure_block(rows)
    superseded_block = _exposure_block(superseded)

    prior_rate = prior["prior_rate"]
    prior_ess = prior_ess_before_segmentation
    segmentation_block: dict[str, Any] = {
        "applied": False,
        "boundary_observed_at": boundary,
        "signal_records": list(role_records),
        "pre_segmentation_prior_ess": round(prior_ess_before_segmentation, 6),
    }
    if superseded_block["minutes"] > 0 and prior_rate is not None:
        # The pre-signal block is genuine same-player evidence about the PLAYER but
        # not about the role he now plays, so it enters the PRIOR at its own
        # minutes of weight instead of being scored as current-role evidence.
        denominator = superseded_block["minutes"] + prior_ess
        blended = (
            (superseded_block["total"] + prior_ess * prior_rate) / denominator
            if denominator > 0
            else prior_rate
        )
        segmentation_block.update(
            {
                "applied": True,
                "superseded_minutes": round(superseded_block["minutes"], 6),
                "superseded_rate": (
                    round(superseded_block["rate"], 6) if superseded_block["rate"] is not None else None
                ),
                "superseded_fixtures": list(superseded_block["fixtures"]),
                "prior_rate_before_segmentation": round(float(prior_rate), 6),
                "prior_rate_after_segmentation": round(float(blended), 6),
                "prior_ess_after_segmentation": round(denominator, 6),
            }
        )
        prior_rate = blended
        prior_ess = denominator

    alignment_block: dict[str, Any] = {
        "applied": False,
        "confirmed_role_change": bool(confirmed),
        "eligible_signals": len(role_records),
    }
    if (
        FAMILY_POOLED_PRIOR_ROLE_ALIGNMENT in families
        and confirmed
        and prior.get("same_player_history")
    ):
        pooled = (
            (pools.get("position") or {}).get(int(element_type), {}).get(component)
            if element_type is not None
            else None
        )
        if pooled and pooled.get("rate") is not None:
            alignment_block.update(
                {
                    "applied": True,
                    "prior_source_before": prior["prior_source"],
                    "prior_rate_before_alignment": (
                        round(float(prior_rate), 6) if prior_rate is not None else None
                    ),
                    "prior_rate_after_alignment": round(float(pooled["rate"]), 6),
                    "pooled_rate_source": "position_pooled_at_cutoff_identity",
                    "pooled_minutes": round(float(pooled["minutes"]), 6),
                }
            )
            flags.append(FLAG_PRIOR_MEAN_REPLACED)
            prior_rate = float(pooled["rate"])
        else:
            flags.append(FLAG_PRIOR_MEAN_REPLACEMENT_UNAVAILABLE)

    portability_discount = 1.0
    if prior.get("same_player_history"):
        portability_discount = float(resolved["club_portability_ess_discount"])
        flags.append("CLUB_PORTABILITY_UNVERIFIED")
    ess_before_discounts = prior_ess
    prior_ess = prior_ess * discount * portability_discount
    if discount < 1.0:
        flags.extend(role_flags)
    # The prior's weight, step by step, so the published number can be re-derived
    # from the published evidence rather than taken on trust.
    ess_breakdown = {
        "declared_or_derived_base_minutes": round(prior_ess_base, 6),
        "after_prior_hierarchy_scale": round(prior_ess_before_segmentation, 6),
        "after_role_segmentation": round(ess_before_discounts, 6),
        "role_change_discount": round(discount, 6),
        "club_portability_discount": round(portability_discount, 6),
        "final_prior_ess_minutes": round(prior_ess, 6),
    }

    current_rate = current["rate"]
    current_minutes = float(current["minutes"])
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
    if exposure["placeholder_rows"]:
        data_gaps.append(
            f"{hc.CERTIFIED_PREDICTION_INPUT_HISTORY_INCOMPLETE}: "
            f"{len(exposure['placeholder_rows'])} completed-fixture row(s) carry no official "
            "observation and contribute no exposure"
        )
    if segmentation_block["applied"]:
        data_gaps.append(
            "the role signal segmented this player's own exposure; pre-signal rows were folded into "
            "the prior as old-role evidence"
        )
    excluded_history = dict(history.get("excluded") or {})
    if excluded_history.get("post_cutoff_rows") or excluded_history.get("undated_rows"):
        data_gaps.append(
            "prior-season rows that were not observable at the cutoff were excluded: "
            f"{excluded_history.get('post_cutoff_rows', 0)} written after it, "
            f"{excluded_history.get('undated_rows', 0)} with no observation time"
        )

    return {
        "player_id": int(player_id),
        "component": component,
        "event": int(planning_event),
        "prior_mean": round(prior_rate, 6) if prior_rate is not None else None,
        "prior_ess": round(prior_ess, 6),
        "prior_ess_before_discounts": round(prior_ess_before_segmentation, 6),
        "prior_source": prior["prior_source"],
        "prior_season": prior.get("prior_season"),
        "prior_minutes": round(float(prior.get("prior_minutes") or 0.0), 6),
        "current_minutes": round(current_minutes, 6),
        "current_total": round(float(current["total"]), 6),
        "current_rate": round(current_rate, 6) if current_rate is not None else None,
        "current_played_rows": int(current["rows"]),
        "posterior_mean": round(posterior, 6) if posterior is not None else None,
        "posterior_ess": round(posterior_ess, 6),
        "posterior_uncertainty": {
            "effective_minutes": round(posterior_ess, 6),
            "prior_share": round(prior_share, 6),
            "current_share": round(current_share, 6),
        },
        "role_modifier_applied": bool(discount < 1.0),
        "role_modifier_ids": [record["id"] for record in role_records],
        "role_modifier_records": list(role_records),
        "club_portability_discount": round(portability_discount, 6),
        "team_context": {"team_id": int(team_id)},
        "risk_flags": sorted(set(flags)),
        "model_version": PLAYER_ATTACK_CHALLENGER_VERSION,
        "generated_at": generated_at,
        "data_cutoff": cutoff,
        "challenger_version": PLAYER_ATTACK_CHALLENGER_VERSION,
        "challenger_config_hash": challenger_config.config_hash(),
        "challenger_families": sorted(families),
        "prior_ess_basis": (ess_estimate or {}).get("basis", ESS_BASIS_INCUMBENT_FALLBACK),
        "prior_ess_breakdown": ess_breakdown,
        "prior_ess_estimate": dict(ess_estimate or {}),
        "role_segmentation": segmentation_block,
        "prior_mean_alignment": alignment_block,
        "exposure_detail": {
            "current_fixtures": list(current["fixtures"]),
            "superseded_fixtures": list(superseded_block["fixtures"]),
            "missing_value_rows": int(exposure["missing_value_rows"]),
            "placeholder_rows": list(exposure["placeholder_rows"]),
            "superseded_minutes": round(superseded_block["minutes"], 6),
            "superseded_rate": (
                round(superseded_block["rate"], 6) if superseded_block["rate"] is not None else None
            ),
            "history_rows_used": int(prior.get("history_rows_used") or 0),
            "history_rows_visible": list(history.get("rows_visible") or []),
            "history_rows_excluded_post_cutoff": int(excluded_history.get("post_cutoff_rows", 0)),
            "history_rows_excluded_undated": int(excluded_history.get("undated_rows", 0)),
        },
        "provenance": {
            "prior_hierarchy": (
                "prev_season_same_player > multi_season_same_player > position_pooled > league_pooled"
            ),
            "history_xg_semantics": "season_total",
            "history_read": "cutoff-observable rows only (observed_at <= cutoff)",
            "xg_bearing_seasons": list(incumbent.XG_BEARING_SEASONS),
            "penalties": "embedded_in_xG",
            "npxg_separation": False,
            "penalty_note": (
                "No NPxG or penalty-xG field exists in the stored official data; penalties remain "
                "embedded and the rate is named xG_per90, never NPxG_per90."
            ),
            "club_identity_available": False,
            "position_basis": POSITION_BASIS,
            "role_change_semantics": (
                "reduces prior ESS only; never adds attacking output; and when the segmentation "
                "family is active it moves pre-signal rows into the prior rather than the current "
                "evidence"
            ),
            "evidence_boundary": (
                "completed player-fixture rows (fixtures.finished=1) with minutes>0 before cutoff"
            ),
            "pool_identity": "the accepted official bootstrap generation at or before the cutoff",
            "promotion": "NOT_PERFORMED_CHALLENGER_ONLY",
        },
        "data_gaps": data_gaps,
    }


def build_challenger_player_rate_projections(
    conn: sqlite3.Connection,
    planning_event: int,
    cutoff: str,
    *,
    players: Sequence[Mapping[str, Any]],
    identities: Mapping[int, tuple[int, int]],
    config: PlayerAttackChallengerConfig | None = None,
    incumbent_config: "incumbent.PlayerRatesConfig | None" = None,
    families: Iterable[str] | None = None,
    pools: Mapping[str, Any] | None = None,
    ess_estimates: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """One row per (candidate player, component) for the planning event.

    ``players`` is the cutoff-resolved candidate pool — membership, club and
    position as the cutoff's accepted generation records them — and ``identities``
    maps every player the cutoff can place to ``(team_id, element_type)`` for the
    pool builder.  The persisted club, the persisted position and the persisted
    active flag are never read here.
    """

    config = config or PlayerAttackChallengerConfig()
    incumbent_config = incumbent_config or incumbent.PlayerRatesConfig()
    family_set = frozenset(REFINEMENT_FAMILIES if families is None else families)
    resolved = config.resolved(incumbent_config)
    generated_at = utc_now()
    if pools is None:
        pools, _disclosure = cutoff_stable_rate_pools(conn, identities, cutoff=cutoff)

    estimates: dict[str, Mapping[str, Any]] = dict(ess_estimates or {})
    if FAMILY_DATA_DERIVED_PRIOR_STRENGTH in family_set and not estimates:
        candidate_ids = sorted(
            {int(player["player_id"]) for player in players} | {int(pid) for pid in identities}
        )
        for component in incumbent.COMPONENTS:
            estimates[component] = derived_prior_strength(
                population_exposure(conn, component, int(planning_event), cutoff, candidate_ids),
                resolved,
                component=component,
                declared_ess=config.declared_ess(component),
            )

    out: list[dict[str, Any]] = []
    for player in sorted(players, key=lambda row: int(row["player_id"])):
        player_id = int(player["player_id"])
        team_id = int(player["team_id"])
        element_type = int(player["element_type"])
        notes = repo.scouting_current_rows_as_of(conn, cutoff, [player_id])
        history = historical_season_rows_as_of(conn, player_id, cutoff)
        for component in incumbent.COMPONENTS:
            out.append(
                _project_component(
                    player_id=player_id,
                    component=component,
                    team_id=team_id,
                    element_type=element_type,
                    planning_event=int(planning_event),
                    cutoff=cutoff,
                    generated_at=generated_at,
                    notes=notes,
                    exposure=current_exposure_rows(conn, player_id, component, int(planning_event), cutoff),
                    prior=challenger_player_prior(player_id, element_type, component, resolved, pools, history),
                    history=history,
                    resolved=resolved,
                    incumbent_config=incumbent_config,
                    challenger_config=config,
                    families=family_set,
                    ess_estimate=estimates.get(component),
                    pools=pools,
                )
            )
    return out


# ---------------------------------------------------------------------------
# Arms.
# ---------------------------------------------------------------------------


@dataclass
class PlayerChallengerArms:
    """One incumbent arm and the challenger arms, over the same key set."""

    planning_event: int
    cutoff: str
    incumbent_rows: dict[tuple[int, str], dict[str, Any]] = field(default_factory=dict)
    arms: dict[str, dict[tuple[int, str], dict[str, Any]]] = field(default_factory=dict)
    families: dict[str, frozenset[str]] = field(default_factory=dict)
    pool_disclosure: dict[str, Any] = field(default_factory=dict)
    ess_estimates: dict[str, Any] = field(default_factory=dict)
    identity: dict[str, Any] = field(default_factory=dict)
    players: list[dict[str, Any]] = field(default_factory=list)

    def keys(self) -> list[tuple[int, str]]:
        return sorted(self.incumbent_rows)

    def arm_keys(self, arm: str) -> list[tuple[int, str]]:
        return sorted(self.arms[arm])

    def arm_names(self) -> tuple[str, ...]:
        return tuple(sorted(self.arms))

    def row(self, arm: str, key: tuple[int, str]) -> dict[str, Any]:
        return self.arms[arm][(int(key[0]), str(key[1]))]

    def verify_same_population(self) -> None:
        """Every arm covers exactly the incumbent's key set, or PE-7 stops."""

        incumbent_keys = set(self.keys())
        for arm in self.arm_names():
            arm_keys = set(self.arm_keys(arm))
            if arm_keys != incumbent_keys:
                raise wf.PopulationMismatch(
                    f"arm {arm!r} covers a different population: "
                    f"{len(incumbent_keys - arm_keys)} incumbent-only key(s), "
                    f"{len(arm_keys - incumbent_keys)} arm-only key(s)"
                )


def build_challenger_player_arms(
    conn: sqlite3.Connection,
    planning_event: int,
    cutoff: str,
    *,
    players: Sequence[Mapping[str, Any]],
    identities: Mapping[int, tuple[int, int]],
    incumbent_config: "incumbent.PlayerRatesConfig | None" = None,
    challenger_config: PlayerAttackChallengerConfig | None = None,
    arm_definitions: Mapping[str, frozenset[str]] | None = None,
) -> PlayerChallengerArms:
    """Project every candidate/component for the incumbent and the challenger arms.

    The incumbent's own per-candidate projection function is called with the
    CUTOFF-RESOLVED identity and its own current pools, so the incumbent arm is
    incumbent code; where that function is reached for directly it is because the
    incumbent publishes no per-player entry point that accepts a cutoff pool, and
    the assertion below pins the two to the same construction.

    The cutoff-stable pools, the derived prior strengths and the role evidence are
    built ONCE per event and shared by every arm, so an ablation's delta is the
    family rather than a second draw of its own inputs.
    """

    incumbent_config = incumbent_config or incumbent.PlayerRatesConfig()
    challenger_config = challenger_config or PlayerAttackChallengerConfig()
    definitions = dict(arm_definitions or default_arm_definitions())

    incumbent_pools = incumbent.pooled_rates(conn, incumbent_config)
    incumbent_rows: list[dict[str, Any]] = []
    generated_at = utc_now()
    for player in sorted(players, key=lambda row: int(row["player_id"])):
        player_id = int(player["player_id"])
        element_type = int(player["element_type"])
        notes = repo.scouting_current_rows_as_of(conn, cutoff, [player_id])
        for component in incumbent.COMPONENTS:
            incumbent_rows.append(
                incumbent._project_component(
                    conn,
                    player_id,
                    int(player["team_id"]),
                    element_type,
                    component,
                    int(planning_event),
                    cutoff,
                    incumbent_config,
                    generated_at,
                    notes,
                    incumbent_pools,
                )
            )

    pools, disclosure = cutoff_stable_rate_pools(conn, identities, cutoff=cutoff)
    resolved = challenger_config.resolved(incumbent_config)
    candidate_ids = sorted(
        {int(player["player_id"]) for player in players} | {int(pid) for pid in identities}
    )
    estimates: dict[str, Any] = {}
    for component in incumbent.COMPONENTS:
        estimates[component] = derived_prior_strength(
            population_exposure(conn, component, int(planning_event), cutoff, candidate_ids),
            resolved,
            component=component,
            declared_ess=challenger_config.declared_ess(component),
        )

    arms: dict[str, dict[tuple[int, str], dict[str, Any]]] = {}
    for arm in sorted(definitions):
        rows = build_challenger_player_rate_projections(
            conn,
            int(planning_event),
            cutoff,
            players=players,
            identities=identities,
            config=challenger_config,
            incumbent_config=incumbent_config,
            families=frozenset(definitions[arm]),
            pools=pools,
            ess_estimates=estimates,
        )
        arms[arm] = {(int(row["player_id"]), str(row["component"])): row for row in rows}
    return PlayerChallengerArms(
        planning_event=int(planning_event),
        cutoff=str(cutoff),
        incumbent_rows={(int(row["player_id"]), str(row["component"])): row for row in incumbent_rows},
        arms=arms,
        families={arm: frozenset(families) for arm, families in definitions.items()},
        pool_disclosure=disclosure,
        ess_estimates=estimates,
        identity=challenger_identity(challenger_config, incumbent_config),
        players=[dict(player) for player in players],
    )
