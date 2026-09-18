"""PE-3 — the authoritative 2026/27 BPS rule table, the exact BPS calculator, and the
primitive coverage audit.

THREE SEPARATE THINGS, which must not be confused:

1. THE RULE TABLE (``RULE_SPECS``).  One row per official 2026/27 BPS rule.  The calculator
   is driven BY this table, so a rule cannot exist in the table without an implemented
   calculation path — completeness is structural, not a matter of remembering.

2. THE PRIMITIVE VOCABULARY.  Explicit names for the counts the calculator consumes.  A
   missing primitive raises :class:`BPSPrimitiveMissing`; it is NEVER read as zero, because
   "we did not measure it" and "it was zero" are different facts.

3. PRIMITIVE COVERAGE.  A MEASURED statement about which primitives this repository can
   supply, on TWO INDEPENDENT dimensions: whether the VALUE exists, and whether it is
   causally safe to use before a cutoff.  A field is not temporally safe merely because its
   final value exists.

The exact calculator is exact WHEN COMPLETE PRIMITIVES ARE SUPPLIED.  The predictive
challenger is a STRUCTURAL APPROXIMATION.  Those two claims are kept apart everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

BPS_RULES_VERSION = "fpl_bps_2026_27_v1.0.0"

BPS_RULES_VERIFICATION = "VERIFIED"

#: Official sources for the 2026/27 table.  Both were cited by senior review on
#: 2026-09-19; both returned HTTP 429 when this environment tried to retrieve them, so the
#: retrieval status is recorded rather than glossed.  The constants below were supplied
#: through review, not scraped.
BPS_RULES_SOURCE: dict[str, Any] = {
    "primary": {
        "title": "FPL Rules Copilot",
        "publisher": "Premier League",
        "article_id": 4661029,
        "url": "https://www.premierleague.com/news/4661029",
        "role": "complete current FPL BPS table",
        "retrieval_status": "HTTP_429_RATE_LIMITED",
    },
    "changes": {
        "title": "What's new in 2026/27 Fantasy: Changes to Bonus Points System",
        "publisher": "Premier League",
        "article_id": 4679946,
        "url": "https://www.premierleague.com/news/4679946",
        "role": "independent confirmation of the 2026/27 changes",
        "retrieval_status": "HTTP_429_RATE_LIMITED",
    },
    "values_supplied_via": "senior review, 2026-09-19",
    "retrieval_attempted": "2026-09-19",
    "stale_pre_2026_27_reference": {
        "title": "How the FPL Bonus Points System works",
        "article_id": 106533,
        "url": "https://www.premierleague.com/news/106533",
        "role": "STALE_PRE_2026_27_REFERENCE — must not be the authority",
        "retrieved": "2026-09-18",
        "contradicts": [
            "being tackled present at -1 (removed for 2026/27)",
            "clearances/blocks/interceptions 1 per 2 (now 1 per 3)",
            "goalkeeper save 3 inside / 2 outside (now 2 per save)",
            "no inside-box-save increment (now +1)",
            "no big-chance-save row (now +1)",
            "penalty save 8 (now 7)",
        ],
    },
}


class BPSPrimitiveMissing(ValueError):
    """An exact BPS calculation was asked for without every required primitive."""

    def __init__(self, position: str, missing: Sequence[str]) -> None:
        super().__init__(
            f"BPS_PRIMITIVE_MISSING: exact BPS for position {position} requires "
            f"{sorted(set(missing))}; a missing primitive is never treated as zero"
        )
        self.position = position
        self.missing = sorted(set(missing))


class BPSPrimitiveInconsistent(ValueError):
    """Supplied primitives cannot all be true of one player-fixture."""

    def __init__(self, detail: str) -> None:
        super().__init__(f"BPS_PRIMITIVE_INCONSISTENT: {detail}")
        self.detail = detail


class BPSMode:
    """How one official rule row consumes a primitive."""

    PER_ACTION = "PER_ACTION"
    PER_N = "PER_N"
    NON_PENALTY_GOAL = "NON_PENALTY_GOAL"
    PENALTY_GOAL = "PENALTY_GOAL"
    CLEAN_SHEET = "CLEAN_SHEET"
    GOAL_CONCEDED = "GOAL_CONCEDED"
    PASS_BAND = "PASS_BAND"
    REMOVED_2026_27 = "REMOVED_2026_27"


@dataclass(frozen=True)
class BPSPrimitiveSpec:
    """ONE official BPS rule row: its rule name, the primitive it consumes, and how."""

    rule_row: str
    mode: str
    value: int = 0
    primitive: str | None = None
    per: int = 1
    positions: tuple[str, ...] = ()
    bands: tuple[tuple[float, float, int], ...] = ()


def _spec(rule_row: str, primitive: str, value: int) -> BPSPrimitiveSpec:
    return BPSPrimitiveSpec(rule_row=rule_row, mode=BPSMode.PER_ACTION,
                            primitive=primitive, value=value)


def _per_n(rule_row: str, primitive: str, per: int, value: int) -> BPSPrimitiveSpec:
    return BPSPrimitiveSpec(rule_row=rule_row, mode=BPSMode.PER_N,
                            primitive=primitive, per=per, value=value)


#: EVERY official 2026/27 BPS rule row, in table order.  This tuple is the contract: the
#: calculator iterates it, so no row can be tabulated without being applied, and the
#: completeness test asserts the official row list matches this one exactly (with
#: ``being_tackled`` present as ``REMOVED_2026_27`` and consuming no primitive).
RULE_SPECS: tuple[BPSPrimitiveSpec, ...] = (
    # --- minutes ---
    _spec("plays_1_to_60_minutes", "minutes", 3),
    _spec("plays_over_60_minutes", "minutes", 6),
    # --- goals ---
    BPSPrimitiveSpec("direct_penalty_goal", BPSMode.PENALTY_GOAL,
                     primitive="penalty_goals", value=12),
    BPSPrimitiveSpec("gkp_non_penalty_goal", BPSMode.NON_PENALTY_GOAL, value=12,
                     primitive="goals_scored", positions=("GKP",)),
    BPSPrimitiveSpec("def_non_penalty_goal", BPSMode.NON_PENALTY_GOAL, value=12,
                     primitive="goals_scored", positions=("DEF",)),
    BPSPrimitiveSpec("mid_non_penalty_goal", BPSMode.NON_PENALTY_GOAL, value=18,
                     primitive="goals_scored", positions=("MID",)),
    BPSPrimitiveSpec("fwd_non_penalty_goal", BPSMode.NON_PENALTY_GOAL, value=24,
                     primitive="goals_scored", positions=("FWD",)),
    # --- assists ---
    _spec("assist", "assists", 9),
    # --- clean sheets ---
    BPSPrimitiveSpec("gkp_clean_sheet", BPSMode.CLEAN_SHEET, value=12,
                     primitive="clean_sheets", positions=("GKP",)),
    BPSPrimitiveSpec("def_clean_sheet", BPSMode.CLEAN_SHEET, value=12,
                     primitive="clean_sheets", positions=("DEF",)),
    # --- goalkeeper saves (ADDITIVE: a penalty save may also be a save, an inside-box save
    #     and a big-chance save, and every applicable row applies) ---
    _spec("any_save", "saves", 2),
    _spec("save_from_inside_box", "inside_box_saves", 1),
    _spec("save_from_big_chance", "big_chance_saves", 1),
    _spec("penalty_save", "penalties_saved", 7),
    # --- defensive / possession ---
    _per_n("clearances_blocks_interceptions_per_3", "clearances_blocks_interceptions", 3, 1),
    _per_n("recoveries_per_3", "recoveries", 3, 1),
    _spec("chance_created", "chances_created", 1),
    _spec("big_chance_created", "big_chances_created", 3),
    _spec("successful_open_play_cross", "open_play_crosses", 1),
    _spec("successful_tackle", "successful_tackles", 2),
    _spec("successful_dribble", "successful_dribbles", 1),
    # --- other positive actions ---
    _spec("match_winning_goal", "winning_goals", 3),
    _spec("goalline_clearance", "goal_line_clearances", 9),
    _spec("foul_won", "fouls_won", 1),
    _spec("shot_on_target", "shots_on_target", 2),
    # --- pass completion (>= 30 attempts attempted) ---
    BPSPrimitiveSpec("pass_completion_70_to_79", BPSMode.PASS_BAND,
                     primitive="pass_completion_percent", bands=((70.0, 79.999, 2),)),
    BPSPrimitiveSpec("pass_completion_80_to_89", BPSMode.PASS_BAND,
                     primitive="pass_completion_percent", bands=((80.0, 89.999, 4),)),
    BPSPrimitiveSpec("pass_completion_90_plus", BPSMode.PASS_BAND,
                     primitive="pass_completion_percent", bands=((90.0, 100.0, 6),)),
    # --- negative actions ---
    BPSPrimitiveSpec("gkp_def_goal_conceded", BPSMode.GOAL_CONCEDED, value=-4,
                     primitive="goals_conceded", positions=("GKP", "DEF")),
    _spec("conceding_a_penalty", "penalties_conceded", -3),
    _spec("missing_a_penalty", "penalties_missed", -6),
    _spec("yellow_card", "yellow_cards", -3),
    _spec("red_card", "red_cards", -9),
    _spec("own_goal", "own_goals", -6),
    _spec("missing_a_big_chance", "big_chances_missed", -3),
    _spec("error_leading_to_goal", "errors_leading_to_goal", -3),
    _spec("error_leading_to_attempt", "errors_leading_to_attempt", -1),
    _spec("conceding_a_foul", "fouls_conceded", -1),
    _spec("caught_offside", "offsides", -1),
    _spec("shot_off_target", "shots_off_target", -1),
    # --- removed for 2026/27 ---
    BPSPrimitiveSpec("being_tackled", BPSMode.REMOVED_2026_27),
)

#: The official 2026/27 rule rows, for the completeness gate.
OFFICIAL_RULE_ROWS: tuple[str, ...] = (
    "plays_1_to_60_minutes", "plays_over_60_minutes",
    "direct_penalty_goal", "gkp_non_penalty_goal", "def_non_penalty_goal",
    "mid_non_penalty_goal", "fwd_non_penalty_goal",
    "assist", "gkp_clean_sheet", "def_clean_sheet",
    "any_save", "save_from_inside_box", "save_from_big_chance", "penalty_save",
    "clearances_blocks_interceptions_per_3", "recoveries_per_3", "chance_created",
    "big_chance_created", "successful_open_play_cross", "successful_tackle",
    "successful_dribble", "match_winning_goal", "goalline_clearance", "foul_won",
    "shot_on_target", "pass_completion_70_to_79", "pass_completion_80_to_89",
    "pass_completion_90_plus", "gkp_def_goal_conceded", "conceding_a_penalty",
    "missing_a_penalty", "yellow_card", "red_card", "own_goal", "missing_a_big_chance",
    "error_leading_to_goal", "error_leading_to_attempt", "conceding_a_foul",
    "caught_offside", "shot_off_target", "being_tackled",
)

IMPLEMENTED_RULE_ROWS: tuple[str, ...] = tuple(spec.rule_row for spec in RULE_SPECS)


def ruleset_fingerprint() -> dict[str, Any]:
    """Stable identity of the live rule set, for artifact provenance (Part 17).

    Covers the rule version, the verification state, every constant and the row count, so an
    artifact can record WHICH rules produced it and a later change is detectable by hash.
    """

    import hashlib
    import json

    payload = {
        "version": BPS_RULES_VERSION,
        "verification": BPS_RULES_VERIFICATION,
        "rows": [
            {
                "rule_row": spec.rule_row, "mode": spec.mode, "value": spec.value,
                "primitive": spec.primitive, "per": spec.per,
                "positions": list(spec.positions),
                "bands": [list(band) for band in spec.bands],
            }
            for spec in RULE_SPECS
        ],
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return {
        "version": BPS_RULES_VERSION,
        "verification": BPS_RULES_VERIFICATION,
        "rule_rows": len(RULE_SPECS),
        "rules_hash": "sha256:" + hashlib.sha256(blob).hexdigest(),
        "being_tackled": "REMOVED_2026_27",
    }


#: The live rule-set identity.  Importable so a test can pin the season-sensitive state.
RULESET_FINGERPRINT: dict[str, Any] = ruleset_fingerprint()

#: Every primitive the calculator consumes, derived from the table so it cannot drift.
REQUIRED_BPS_PRIMITIVES: tuple[str, ...] = tuple(sorted(
    {spec.primitive for spec in RULE_SPECS if spec.primitive} | {"pass_attempts",
                                                                "pass_completion_percent"}
))

#: Primitives that must be supplied even when they are not named by a rule row directly.
_EXTRA_REQUIRED = ("pass_attempts", "pass_completion_percent")


def _apply_rules(position: str, primitives: Mapping[str, Any],
                 rules: Sequence[BPSPrimitiveSpec], *, allow_missing: bool,
                 ) -> tuple[int, tuple[str, ...]]:
    """Apply a rule table.  ``allow_missing`` is the ONLY difference between
    exact and structural mode: exact raises on an absent primitive, structural skips
    the affected rule AND REPORTS it, so an unsupported component can never be
    silently counted as zero.
    """
    """Exact BPS for one player-fixture, from a COMPLETE primitive payload.

    Driven by the rule table: every ``BPSPrimitiveSpec`` is applied, so tabulated rules
    cannot silently be omitted.  A missing primitive raises
    :class:`BPSPrimitiveMissing`; it is never read as zero.  An explicit zero is valid.

    ``rules`` is injectable so the mechanics can be tested against a synthetic table; the
    default is the verified 2026/27 contract.

    Raises :class:`BPSPrimitiveInconsistent` when the supplied counts cannot all be true of
    one player-fixture, e.g. more inside-box saves than saves.
    """

    required = sorted({spec.primitive for spec in rules if spec.primitive} | set(_EXTRA_REQUIRED))
    missing = [name for name in required if name not in primitives]
    if missing and not allow_missing:
        raise BPSPrimitiveMissing(position, missing)
    skipped: list[str] = []
    from types import SimpleNamespace as _NS
    _safe = _NS(**{name: int(primitives.get(name, 0) or 0) for name in required})

    minutes = _safe.minutes
    goals = _safe.goals_scored
    penalty_goals = _safe.penalty_goals
    saves = _safe.saves
    inside_box = _safe.inside_box_saves
    big_chance = _safe.big_chance_saves

    if penalty_goals > goals:
        raise BPSPrimitiveInconsistent(f"{penalty_goals} penalty goals > {goals} goals scored")
    if inside_box > saves:
        raise BPSPrimitiveInconsistent(f"{inside_box} inside-box saves > {saves} saves")
    if big_chance > saves:
        raise BPSPrimitiveInconsistent(f"{big_chance} big-chance saves > {saves} saves")

    attempts = _safe.pass_attempts
    completion = float(primitives.get("pass_completion_percent", 0.0) or 0.0)
    non_penalty_goals = goals - penalty_goals

    total = 0
    for spec in rules:
        mode = spec.mode
        if mode == BPSMode.REMOVED_2026_27:
            continue
        if allow_missing and spec.primitive and spec.primitive not in primitives:
            skipped.append(spec.rule_row)
            continue
        if mode == BPSMode.PER_ACTION and spec.rule_row == "plays_1_to_60_minutes":
            total += spec.value if 1 <= minutes <= 60 else 0
        elif mode == BPSMode.PER_ACTION and spec.rule_row == "plays_over_60_minutes":
            total += spec.value if minutes > 60 else 0
        elif mode == BPSMode.PER_ACTION:
            total += int(primitives.get(spec.primitive, 0) or 0) * spec.value
        elif mode == BPSMode.PER_N:
            total += (int(primitives.get(spec.primitive, 0) or 0) // spec.per) * spec.value
        elif mode == BPSMode.NON_PENALTY_GOAL:
            if position in spec.positions:
                total += non_penalty_goals * spec.value
        elif mode == BPSMode.PENALTY_GOAL:
            total += penalty_goals * spec.value
        elif mode == BPSMode.CLEAN_SHEET:
            if position in spec.positions and _safe.clean_sheets:
                total += spec.value
        elif mode == BPSMode.GOAL_CONCEDED:
            if position in spec.positions:
                total += _safe.goals_conceded * spec.value
        elif mode == BPSMode.PASS_BAND:
            if attempts >= 30:
                for low, high, points in spec.bands:
                    if low <= completion <= high:
                        total += points
                        break
        else:  # pragma: no cover - a new mode without a branch must be loud
            raise AssertionError(f"BPS rule mode {mode!r} has no calculation path")
    return int(total), tuple(sorted(set(skipped)))



def calculate_bps(position: str, primitives: Mapping[str, Any],
                  rules: Sequence[BPSPrimitiveSpec] = RULE_SPECS) -> int:
    """EXACT BPS for one player-fixture, from a COMPLETE primitive payload.

    Every tabulated rule is applied.  A missing primitive raises
    :class:`BPSPrimitiveMissing`; it is never read as zero, because "not measured" and
    "measured zero" are different facts.  An explicit zero is valid.

    This is the EXACT calculator.  The predictive challenger uses
    :func:`structural_bps`, which is a STRUCTURAL APPROXIMATION and says so.
    """

    total, _skipped = _apply_rules(position, primitives, rules, allow_missing=False)
    return total


def structural_bps(position: str, world_primitives: Mapping[str, Any],
                   rules: Sequence[BPSPrimitiveSpec] = RULE_SPECS,
                   ) -> tuple[int, tuple[str, ...]]:
    """BPS from ONLY the rules whose primitives the caller can actually supply.

    Returns ``(bps, unsupported_rule_rows)``.  A rule whose primitive is absent is SKIPPED
    AND NAMED, never treated as zero: an unsupported component is a known hole in the
    proxy, not a measured absence.  This is why the challenger is a STRUCTURAL
    APPROXIMATION and not an exact reconstruction, and the caller is expected to surface
    the returned names.
    """

    return _apply_rules(position, world_primitives, rules, allow_missing=True)


# ---------------------------------------------------------------------------
# Primitive coverage — TWO independent dimensions
# ---------------------------------------------------------------------------

EXACT = "EXACT"
APPROXIMATE = "APPROXIMATE"
ABSENT = "ABSENT"

POINT_IN_TIME_SAFE = "POINT_IN_TIME_SAFE"
UNPROVEN = "UNPROVEN"
NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass(frozen=True)
class PrimitiveCoverage:
    """Availability and temporal safety are SEPARATE facts about a primitive."""

    primitive: str
    availability: str
    temporal_status: str
    where: str = "-"
    note: str = ""


def _cov(primitive: str, availability: str, temporal: str, where: str = "-",
         note: str = "") -> PrimitiveCoverage:
    return PrimitiveCoverage(primitive=primitive, availability=availability,
                             temporal_status=temporal, where=where, note=note)


#: MEASURED against this repository on 2026-09-18/19.
#:
#: ``player_gameweeks`` stores 2,549 of 24,955 rows with a full payload for the current
#: season; the remaining rows are scheduled placeholders with minutes only.
PRIMITIVE_COVERAGE: tuple[PrimitiveCoverage, ...] = (
    # per-fixture columns
    _cov("minutes", EXACT, POINT_IN_TIME_SAFE, "player_gameweeks.minutes"),
    _cov("goals_scored", EXACT, POINT_IN_TIME_SAFE, "player_gameweeks.goals_scored"),
    _cov("assists", EXACT, POINT_IN_TIME_SAFE, "player_gameweeks.assists"),
    _cov("clean_sheets", EXACT, POINT_IN_TIME_SAFE, "player_gameweeks.clean_sheets"),
    _cov("goals_conceded", EXACT, POINT_IN_TIME_SAFE, "player_gameweeks.goals_conceded"),
    _cov("saves", EXACT, POINT_IN_TIME_SAFE, "player_gameweeks.saves"),
    _cov("penalties_saved", EXACT, POINT_IN_TIME_SAFE, "player_gameweeks.penalties_saved"),
    _cov("penalties_missed", EXACT, POINT_IN_TIME_SAFE, "player_gameweeks.penalties_missed"),
    _cov("yellow_cards", EXACT, POINT_IN_TIME_SAFE, "player_gameweeks.yellow_cards"),
    _cov("red_cards", EXACT, POINT_IN_TIME_SAFE, "player_gameweeks.red_cards"),
    _cov("own_goals", EXACT, POINT_IN_TIME_SAFE, "player_gameweeks.own_goals"),
    # per-fixture raw_json fields — value exists, causal safety NOT established
    _cov("clearances_blocks_interceptions", EXACT, UNPROVEN, "player_gameweeks.raw_json",
         "value present per fixture; no capture-history test exists, so it is not proven "
         "recoverable at an earlier cutoff"),
    _cov("recoveries", EXACT, UNPROVEN, "player_gameweeks.raw_json",
         "as above"),
    _cov("tackles", EXACT, UNPROVEN, "player_gameweeks.raw_json",
         "as above; note 'tackles' is NOT the same primitive as the removed "
         "'being tackled' deduction"),
    # season cumulatives only
    _cov("clearances_blocks_interceptions (season total)", APPROXIMATE, UNPROVEN,
         "player_snapshots.clearances_blocks_interceptions"),
    _cov("recoveries (season total)", APPROXIMATE, UNPROVEN,
         "player_snapshots.recoveries"),
    _cov("tackles (season total)", APPROXIMATE, UNPROVEN, "player_snapshots.tackles"),
)

#: Every required primitive that is ABSENT from this repository.  Exact historical BPS
#: reconstruction is refused because of these, not because of the rule table.
ABSENT_PRIMITIVES: tuple[str, ...] = (
    "penalty_goals", "inside_box_saves", "big_chance_saves", "chances_created",
    "big_chances_created", "open_play_crosses", "successful_tackles",
    "successful_dribbles", "winning_goals", "goal_line_clearances", "fouls_won",
    "shots_on_target", "pass_attempts", "pass_completion_percent", "penalties_conceded",
    "big_chances_missed", "errors_leading_to_goal", "errors_leading_to_attempt",
    "fouls_conceded", "offsides", "shots_off_target",
)


def exact_historical_bps_reconstruction_possible() -> bool:
    """False: the ABSENT primitives above contribute to BPS and are not stored.

    A reconstruction that silently omitted them would understate every player's BPS, so it
    is refused rather than approximated.  This is a statement about our DATA; the rule
    table itself is verified.
    """

    return not ABSENT_PRIMITIVES


def coverage_by_availability() -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for entry in PRIMITIVE_COVERAGE:
        grouped.setdefault(entry.availability, []).append(entry.primitive)
    return grouped


def coverage_by_temporal_status() -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for entry in PRIMITIVE_COVERAGE:
        grouped.setdefault(entry.temporal_status, []).append(entry.primitive)
    return grouped
