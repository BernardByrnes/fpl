"""PE-3 — BPS rule authority and primitive coverage.

TWO SEPARATE THINGS live here, and they must not be confused:

1. PRIMITIVE COVERAGE — a measured inventory of which BPS inputs this repository can
   actually supply, per player per fixture.  This is a statement about OUR DATA and it is
   complete and reliable.

2. THE BPS RULE TABLE — the official 2026/27 constants.  THIS IS NOT VERIFIED.  See
   ``BPS_RULES_VERIFICATION`` and ``CANDIDATE_RULE_TABLE_DISCREPANCIES`` below: the only
   official source this environment could retrieve publishes a table that contradicts six
   of the season-sensitive 2026/27 values, so no authoritative table could be established.
   Nothing in PE-3 may treat the candidate table as current.

Because of (2), :func:`calculate_bps` is deliberately exact against an INJECTED rule table
and is unit-tested against synthetic tables, never against the candidate table as though it
were the 2026/27 contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

BPS_RULES_VERSION = "fpl_bps_2026_27_v1.0.0"

#: Machine-readable verification state.  A consumer that needs the real 2026/27 table MUST
#: check this and refuse when it is not VERIFIED.
BPS_RULES_VERIFICATION = "UNVERIFIED_FOR_2026_27"

#: The single official source this environment could retrieve, and why it is not usable.
BPS_RULES_SOURCE = {
    "url": "https://www.premierleague.com/news/106533",
    "title": "How the FPL Bonus Points System works",
    "byline_date": "2026-04-01",
    "retrieved": "2026-09-18",
    "usable": False,
    "reason": (
        "The article is dated 2026-04-01 but publishes the PRE-2025/26 table. It "
        "contradicts six of the season-sensitive values this phase requires, so it cannot "
        "be encoded as the current contract. The FPL rules page "
        "(fantasy.premierleague.com/help/rules) is a JavaScript shell with no table, and no "
        "search engine was reachable from this environment to locate a newer article."
    ),
}

#: Every value where the retrieved source disagrees with the required 2026/27 rule.
CANDIDATE_RULE_TABLE_DISCREPANCIES: tuple[dict[str, Any], ...] = (
    {"rule": "being tackled", "required_2026_27": "REMOVED", "retrieved_source": "-1 BPS"},
    {"rule": "clearances/blocks/interceptions", "required_2026_27": "1 BPS per 3",
     "retrieved_source": "1 BPS per 2"},
    {"rule": "goalkeeper save", "required_2026_27": "2 BPS per save",
     "retrieved_source": "3 inside box / 2 outside box"},
    {"rule": "save inside the box", "required_2026_27": "+1 BPS",
     "retrieved_source": "no separate increment"},
    {"rule": "big-chance save", "required_2026_27": "+1 BPS",
     "retrieved_source": "absent from the table"},
    {"rule": "penalty save", "required_2026_27": "7 BPS", "retrieved_source": "8 BPS"},
)


class BPSPrimitiveMissing(ValueError):
    """An exact BPS calculation was asked for without every required primitive."""

    def __init__(self, position: str, missing: list[str]) -> None:
        super().__init__(
            f"BPS_PRIMITIVE_MISSING: exact BPS for position {position} requires "
            f"{missing}; a missing primitive is never treated as zero"
        )
        self.position = position
        self.missing = list(missing)


@dataclass(frozen=True)
class BPSRuleTable:
    """A complete BPS constant set.  Injected, never global.

    Only the constants that a supplied primitive can consume are represented.  A table is
    always paired with the identity of the rule set it came from, so a calculation can
    never be reported without saying which rules produced it.
    """

    version: str
    per_action: Mapping[str, int] = field(default_factory=dict)
    per_n_actions: Mapping[str, tuple[int, int]] = field(default_factory=dict)
    position_goal: Mapping[str, int] = field(default_factory=dict)
    position_clean_sheet: Mapping[str, int] = field(default_factory=dict)
    position_goal_conceded: Mapping[str, int] = field(default_factory=dict)
    pass_completion_bands: tuple[tuple[str, float, float, int], ...] = ()

    def constant(self, name: str) -> int:
        if name not in self.per_action:
            raise KeyError(f"BPS rule {name!r} is not in {self.version}")
        return int(self.per_action[name])


# ---------------------------------------------------------------------------
# Primitive coverage — MEASURED against this repository, 2026-09-18
# ---------------------------------------------------------------------------

#: Classification vocabulary.
AVAILABLE_EXACT = "AVAILABLE_EXACT"
AVAILABLE_APPROXIMATE = "AVAILABLE_APPROXIMATE"
ABSENT = "ABSENT"
NOT_POINT_IN_TIME_SAFE = "NOT_POINT_IN_TIME_SAFE"

#: What ``player_gameweeks`` actually stores, per player per fixture, for the current
#: season.  MEASURED: 2,549 of 24,955 rows carry the full payload; the remainder are
#: scheduled placeholders with minutes only.
PRIMITIVE_COVERAGE: dict[str, dict[str, str]] = {
    # --- stored as first-class columns on player_gameweeks ---
    "minutes": {"class": AVAILABLE_EXACT, "where": "player_gameweeks.minutes"},
    "goals_scored": {"class": AVAILABLE_EXACT, "where": "player_gameweeks.goals_scored"},
    "assists": {"class": AVAILABLE_EXACT, "where": "player_gameweeks.assists"},
    "clean_sheets": {"class": AVAILABLE_EXACT, "where": "player_gameweeks.clean_sheets"},
    "goals_conceded": {"class": AVAILABLE_EXACT, "where": "player_gameweeks.goals_conceded"},
    "saves": {"class": AVAILABLE_EXACT, "where": "player_gameweeks.saves"},
    "penalties_saved": {"class": AVAILABLE_EXACT, "where": "player_gameweeks.penalties_saved"},
    "penalties_missed": {"class": AVAILABLE_EXACT, "where": "player_gameweeks.penalties_missed"},
    "yellow_cards": {"class": AVAILABLE_EXACT, "where": "player_gameweeks.yellow_cards"},
    "red_cards": {"class": AVAILABLE_EXACT, "where": "player_gameweeks.red_cards"},
    "own_goals": {"class": AVAILABLE_EXACT, "where": "player_gameweeks.own_goals"},
    "bonus": {"class": AVAILABLE_EXACT, "where": "player_gameweeks.bonus"},
    "bps": {"class": AVAILABLE_EXACT, "where": "player_gameweeks.bps"},
    "defensive_contribution": {"class": AVAILABLE_EXACT,
                               "where": "player_gameweeks.defensive_contribution"},
    # --- present only inside player_gameweeks.raw_json ---
    "clearances_blocks_interceptions": {
        "class": AVAILABLE_EXACT, "where": "player_gameweeks.raw_json"},
    "recoveries": {"class": AVAILABLE_EXACT, "where": "player_gameweeks.raw_json"},
    "tackles": {"class": AVAILABLE_EXACT, "where": "player_gameweeks.raw_json"},
    # --- present, but only as SEASON CUMULATIVES on the bootstrap element ---
    "clearances_blocks_interceptions_season_total": {
        "class": AVAILABLE_APPROXIMATE, "where": "player_snapshots.clearances_blocks_interceptions"},
    "recoveries_season_total": {"class": AVAILABLE_APPROXIMATE,
                                "where": "player_snapshots.recoveries"},
    "tackles_season_total": {"class": AVAILABLE_APPROXIMATE,
                             "where": "player_snapshots.tackles"},
    # --- absent from every stored payload ---
    "penalty_vs_non_penalty_goal": {"class": ABSENT, "where": "-"},
    "save_inside_box": {"class": ABSENT, "where": "-"},
    "big_chance_save": {"class": ABSENT, "where": "-"},
    "big_chances_created": {"class": ABSENT, "where": "-"},
    "open_play_crosses": {"class": ABSENT, "where": "-"},
    "goal_line_clearances": {"class": ABSENT, "where": "-"},
    "fouls_won": {"class": ABSENT, "where": "-"},
    "fouls_conceded": {"class": ABSENT, "where": "-"},
    "shots_on_target": {"class": ABSENT, "where": "-"},
    "shots_off_target": {"class": ABSENT, "where": "-"},
    "big_chances_missed": {"class": ABSENT, "where": "-"},
    "offsides": {"class": ABSENT, "where": "-"},
    "errors_leading_to_shot": {"class": ABSENT, "where": "-"},
    "errors_leading_to_goal": {"class": ABSENT, "where": "-"},
    "pass_attempts": {"class": ABSENT, "where": "-"},
    "pass_completion": {"class": ABSENT, "where": "-"},
    "winning_goal_identity": {"class": ABSENT, "where": "-"},
    "successful_dribbles": {"class": ABSENT, "where": "-"},
    "key_passes": {"class": ABSENT, "where": "-"},
    "being_tackled": {"class": ABSENT, "where": "-"},
}


def coverage_by_class() -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for name, entry in PRIMITIVE_COVERAGE.items():
        grouped.setdefault(entry["class"], []).append(name)
    for names in grouped.values():
        names.sort()
    return grouped


def exact_historical_bps_reconstruction_possible() -> bool:
    """False: several BPS actions are not stored at all, so a full replay is impossible.

    MEASURED: of the required non-positive actions, ``fouls_conceded``, ``offside``,
    ``shot off target``, ``big chance missed``, ``errors leading to goal/attempt``,
    ``pass completion`` and ``goals conceded while on the pitch`` each contribute to BPS
    and none is stored per fixture.  A reconstruction that silently omitted them would
    understate every player's BPS, so it is refused rather than approximated.
    """

    grouped = coverage_by_class()
    return not grouped.get(ABSENT)


def missing_primitives_for_exact_bps() -> list[str]:
    """The ABSENT inputs that make exact historical BPS reconstruction impossible."""

    return sorted(coverage_by_class().get(ABSENT, []))


# ---------------------------------------------------------------------------
# The exact calculator — a SCORING function, not a predictive one
# ---------------------------------------------------------------------------


#: Every primitive an EXACT calculation must be given, independent of any rule table.
#: These are PRIMITIVE names; the rule table's ``per_action`` keys are RULE-constant names
#: ("played_over_60", "assist", "yellow_card"), which is a different vocabulary — deriving
#: the required set from the table would demand primitives that do not exist.
#:
#: ``penalty_goals`` is required ON PURPOSE.  Without it the calculator cannot separate a
#: goal scored direct from a penalty from an open-play goal, and that distinction changes
#: the score, so an "exact" mode that silently assumed zero penalty goals would not be
#: exact.  The cost is that a payload which genuinely cannot know is REFUSED rather than
#: guessed — which is the correct outcome for this repository, where penalty status is not
#: stored at all.
REQUIRED_BPS_PRIMITIVES = (
    "minutes", "goals_scored", "penalty_goals", "assists", "clean_sheets",
    "goals_conceded", "penalties_saved", "penalties_missed", "own_goals",
    "yellow_cards", "red_cards",
)


def calculate_bps(position: str, primitives: Mapping[str, int], rules: BPSRuleTable) -> int:
    """Exact BPS for one player-fixture, from a COMPLETE primitive payload.

    Every primitive that the rule table can consume must be supplied.  A missing key
    raises :class:`BPSPrimitiveMissing`: it is never read as zero, because a missing
    count and a genuine zero are different facts and only one of them is a measurement.

    Unknown keys are ignored so a caller may pass a superset (for example a whole
    ``player_gameweeks.raw_json``) without this function having to know about fields that
    carry no BPS weight.
    """

    required = set(REQUIRED_BPS_PRIMITIVES)
    required |= set(rules.per_n_actions)
    if rules.pass_completion_bands:
        required |= {"pass_attempts", "pass_completion_percent"}
    missing = sorted(name for name in required if name not in primitives)
    if missing:
        raise BPSPrimitiveMissing(position, missing)

    total = 0

    minutes = int(primitives["minutes"])
    if minutes > 0:
        total += rules.constant("played_1_to_60") if minutes <= 60 else rules.constant("played_over_60")

    goals = int(primitives.get("goals_scored", 0))
    if goals:
        total += goals * int(rules.position_goal.get(position, 0))

    penalty_goals = int(primitives.get("penalty_goals", 0))
    if penalty_goals:
        # A goal scored direct from a penalty carries its own value INSTEAD of the
        # position goal value.
        total -= penalty_goals * int(rules.position_goal.get(position, 0))
        total += penalty_goals * rules.constant("goal_from_penalty")

    total += int(primitives.get("assists", 0)) * rules.constant("assist")

    if int(primitives.get("clean_sheets", 0)) and position in rules.position_clean_sheet:
        total += int(rules.position_clean_sheet[position])

    if position in rules.position_goal_conceded:
        total += int(primitives.get("goals_conceded", 0)) * int(rules.position_goal_conceded[position])

    for name, (per, points) in rules.per_n_actions.items():
        count = int(primitives.get(name, 0))
        if per > 0 and count:
            total += (count // per) * points

    if int(primitives.get("penalties_saved", 0)):
        total += int(primitives["penalties_saved"]) * rules.constant("penalty_save")
    if int(primitives.get("penalties_missed", 0)):
        total += int(primitives["penalties_missed"]) * rules.constant("penalty_miss")
    if int(primitives.get("own_goals", 0)):
        total += int(primitives["own_goals"]) * rules.constant("own_goal")
    if int(primitives.get("yellow_cards", 0)):
        total += int(primitives["yellow_cards"]) * rules.constant("yellow_card")
    if int(primitives.get("red_cards", 0)):
        total += int(primitives["red_cards"]) * rules.constant("red_card")

    attempts = int(primitives.get("pass_attempts", 0))
    if attempts >= 30:
        completion = float(primitives.get("pass_completion_percent", 0.0))
        for _name, low, high, points in rules.pass_completion_bands:
            if low <= completion <= high:
                total += points
                break
    return int(total)
