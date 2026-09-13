"""Versioned FPL scoring rules for 2026/27 (Analytics Phase 4).

Scoring constants must never be scattered through the xPts code.  They live in
one frozen, hashable object with explicit provenance.

Evidence classes (kept visible, never blurred):

* **POINT VALUES — payload-verified.** The exact per-position point values are
  published by the official FPL API in ``bootstrap-static`` →
  ``game_config.scoring``.  They were re-fetched live from
  ``https://fantasy.premierleague.com/api/bootstrap-static/`` on 2026-09-11 and
  are also present in the locally stored official raw payload
  (``data/raw/33/bootstrap_static.json``), so they are re-derivable at every
  fetch.  ``verify_against_scoring_dict`` raises a drift alarm if the stored
  payload ever contradicts this object.
* **COMPLEMENTARY SEMANTICS — documentation-derived, not payload-encoded.**
  The payload carries the point *values* but not the mechanics: the 60-minute
  appearance/clean-sheet boundary, ``saves: 1`` meaning one point per three
  saves, ``goals_conceded: -1`` meaning one point per two conceded, the
  per-position DefCon action sets/thresholds, and bonus 1–3.  These come from
  the documented FPL rules and the project's already-validated DefCon config.
  Live web verification of these mechanics was only **partial**
  (``fantasy.premierleague.com/help/rules`` is a client-rendered app that
  returns no machine-readable content); this is surfaced rather than hidden.

Nothing here is an expected-points value; this module only describes the points
the game awards.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping

SCORING_RULES_VERSION = "fpl_scoring_2026_27_v1.0.0"

# Official element_type ids (payload element_types).
POSITION_IDS = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}

# Verification provenance.  Point values: payload-verified.  Mechanics:
# documentation-derived (see module docstring).
SCORING_RULES_PROVENANCE: dict[str, Any] = {
    "point_values_source": "official bootstrap-static game_config.scoring",
    "point_values_live_url": "https://fantasy.premierleague.com/api/bootstrap-static/",
    "point_values_verified_at": "2026-09-11",
    "point_values_stored_evidence": "data/raw/33/bootstrap_static.json (official fetch run 33)",
    "point_values_evidence_class": "PAYLOAD_VERIFIED",
    "semantics_evidence_class": "DOCUMENTATION_DERIVED",
    "semantics_sources": [
        "documented FPL rules (60-minute appearance/clean-sheet boundary, one point per three saves, one point per two conceded, bonus 1-3)",
        "project-validated DefCon thresholds/action sets (config/post_gw_market_report.default.json)",
        "online corroboration 2026-09-11 (one point per three saves; DEF 10 CBIT / MID-FWD 12 CBIRT)",
    ],
    "web_verification_status": "PARTIAL",
    "web_verification_note": (
        "The official FPL Help/Rules page is client-rendered and returned no machine-readable "
        "scoring content; the exact point values were instead verified from the official API "
        "payload (stronger, and re-derivable every fetch). Mechanic details could not be fully "
        "web-verified in this environment and are labelled DOCUMENTATION_DERIVED, not payload-verified."
    ),
}


@dataclass(frozen=True)
class ScoringRules:
    """One versioned scoring-rule set; every scoring number lives here."""

    appearance_short_points: int = 1  # played but under 60 minutes
    appearance_long_points: int = 2  # 60 minutes or more
    goal_points: dict[str, int] = field(
        default_factory=lambda: {"GKP": 10, "DEF": 6, "MID": 5, "FWD": 4}
    )
    assist_points: int = 3
    clean_sheet_points: dict[str, int] = field(
        default_factory=lambda: {"GKP": 4, "DEF": 4, "MID": 1, "FWD": 0}
    )
    clean_sheet_minutes_required: int = 60
    # A clean sheet needs no goal conceded while the player is on the pitch.
    clean_sheet_counts_conceded_while_on_pitch: bool = True
    saves_per_point: int = 3
    goals_conceded_per_deduction: int = 2  # one deduction step per this many goals
    goals_conceded_points: dict[str, int] = field(
        default_factory=lambda: {"GKP": -1, "DEF": -1, "MID": 0, "FWD": 0}
    )
    goals_conceded_positions: tuple[str, ...] = ("GKP", "DEF")
    defcon_points: int = 2
    defcon_never_stacks: bool = True
    defcon_minimum_minutes: int = 0  # current official rule is action-threshold based
    defcon_thresholds: dict[str, int] = field(
        default_factory=lambda: {"DEF": 10, "MID": 12, "FWD": 12}
    )
    defcon_action_sets: dict[str, str] = field(
        default_factory=lambda: {"DEF": "CBIT", "MID": "CBIRT", "FWD": "CBIRT"}
    )
    defcon_positions: tuple[str, ...] = ("DEF", "MID", "FWD")  # GKP earns 0
    yellow_card_points: int = -1
    red_card_points: int = -3
    own_goal_points: int = -2
    penalty_save_points: int = 5
    penalty_miss_points: int = -2
    bonus_min: int = 1
    bonus_max: int = 3

    def goal_points_for(self, position: str) -> int:
        return int(self.goal_points[position])

    def goals_conceded_points_for(self, position: str) -> int:
        """Points deducted per completed goals-conceded step (0 where n/a)."""

        return int(self.goals_conceded_points.get(position, 0))

    def appearance_points(self, p_appearance: float, p_60_plus: float) -> float:
        """Expected appearance points: 1 for playing, +1 more for 60+."""

        return p_appearance * self.appearance_short_points + p_60_plus * (
            self.appearance_long_points - self.appearance_short_points
        )

    def clean_sheet_points_for(self, position: str) -> int:
        return int(self.clean_sheet_points[position])

    def defcon_threshold_for(self, position: str) -> int | None:
        return self.defcon_thresholds.get(position)

    def defcon_action_set_for(self, position: str) -> str | None:
        return self.defcon_action_sets.get(position)

    def scoring_hash(self) -> str:
        """Hash of the scoring VALUES only (provenance timestamps excluded)."""

        from .analytics import canonical_hash

        values = {item.name: getattr(self, item.name) for item in fields(self)}
        return canonical_hash(
            {
                "rules_version": SCORING_RULES_VERSION,
                **{k: (list(v) if isinstance(v, tuple) else v) for k, v in values.items()},
            }
        )

    def to_dict(self) -> dict[str, Any]:
        values = {item.name: getattr(self, item.name) for item in fields(self)}
        values = {k: (list(v) if isinstance(v, tuple) else v) for k, v in values.items()}
        return {
            "scoring_rules_version": SCORING_RULES_VERSION,
            "rules": values,
            "provenance": SCORING_RULES_PROVENANCE,
            "scoring_hash": self.scoring_hash(),
            "unmodelled": list(UNMODELLED_SCORING),
        }


DEFAULT_SCORING_RULES = ScoringRules()

# Scoring dimensions intentionally NOT modelled in xPts v1, recorded so the
# omission is visible rather than silent.
UNMODELLED_SCORING: tuple[str, ...] = (
    "red cards (excluded from v1; red-card continuation after dismissal is therefore not represented)",
    "own goals (excluded from v1)",
    "penalty saves and misses (no player-specific expectation; penalties stay embedded in xG/xA)",
    "bonus BPS reconstruction (bonus is a soft residual only; the 2026/27 BPS rules changed)",
)


def scoring_rules_from_scoring_dict(scoring: Mapping[str, Any]) -> ScoringRules:
    """Build a rule set from an official ``game_config.scoring`` mapping."""

    def per_position(key: str) -> dict[str, int]:
        block = scoring.get(key) or {}
        return {position: int(block[position]) for position in POSITION_IDS.values() if position in block}

    return ScoringRules(
        appearance_short_points=int(scoring.get("short_play", 1)),
        appearance_long_points=int(scoring.get("long_play", 2)),
        goal_points=per_position("goals_scored"),
        assist_points=int(scoring.get("assists", 3)),
        clean_sheet_points=per_position("clean_sheets"),
        saves_per_point=DEFAULT_SCORING_RULES.saves_per_point,
        goals_conceded_per_deduction=DEFAULT_SCORING_RULES.goals_conceded_per_deduction,
        goals_conceded_points=per_position("goals_conceded"),
        defcon_points=int((scoring.get("defensive_contribution") or {}).get("DEF", 2)),
        yellow_card_points=int(scoring.get("yellow_cards", -1)),
        red_card_points=int(scoring.get("red_cards", -3)),
        own_goal_points=int(scoring.get("own_goals", -2)),
        penalty_save_points=int(scoring.get("penalties_saved", 5)),
        penalty_miss_points=int(scoring.get("penalties_missed", -2)),
    )


def load_stored_scoring(raw_dir: str | Path) -> tuple[dict[str, Any] | None, str | None]:
    """Newest stored ``bootstrap_static.json`` → its ``game_config.scoring``."""

    directory = Path(raw_dir)
    if not directory.exists():
        return None, None
    candidates = sorted(
        directory.glob("*/bootstrap_static.json"),
        key=lambda path: (path.parent.name.isdigit() and int(path.parent.name) or 0, path.stat().st_mtime),
    )
    for path in reversed(candidates):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        scoring = ((payload.get("game_config") or {}).get("scoring")) if isinstance(payload, dict) else None
        if scoring:
            return scoring, str(path)
    return None, None


def verify_against_scoring_dict(rules: ScoringRules, scoring: Mapping[str, Any]) -> list[str]:
    """Compare this rule set with an official scoring payload; list mismatches."""

    mismatches: list[str] = []
    rebuilt = scoring_rules_from_scoring_dict(scoring)
    for item in fields(rules):
        if item.name in {"saves_per_point", "goals_conceded_per_deduction", "clean_sheet_minutes_required",
                         "clean_sheet_counts_conceded_while_on_pitch", "defcon_never_stacks",
                         "defcon_minimum_minutes", "defcon_thresholds", "defcon_action_sets",
                         "defcon_positions", "goals_conceded_positions", "bonus_min", "bonus_max"}:
            continue  # semantics are not encoded in the payload
        expected = getattr(rules, item.name)
        actual = getattr(rebuilt, item.name)
        if expected != actual:
            mismatches.append(f"{item.name}: rules={expected!r} payload={actual!r}")
    return mismatches
