"""FPL 2026/27 scoring-rule tests: exact component arithmetic and semantics."""

from __future__ import annotations

from fpl_brain import xpts
from fpl_brain.scoring_rules import (
    DEFAULT_SCORING_RULES,
    SCORING_RULES_VERSION,
    ScoringRules,
    load_stored_scoring,
    scoring_rules_from_scoring_dict,
    verify_against_scoring_dict,
)

RULES = DEFAULT_SCORING_RULES


# 1 ------------------------------------------------------------------


def test_appearance_points_short_and_long():
    assert RULES.appearance_points(1.0, 0.0) == 1  # played under 60
    assert RULES.appearance_points(1.0, 1.0) == 2  # played 60+
    assert RULES.appearance_points(0.0, 0.0) == 0
    assert RULES.appearance_points(0.5, 0.25) == 0.75


# 2 ------------------------------------------------------------------


def test_goal_points_by_position():
    assert RULES.goal_points_for("GKP") == 10
    assert RULES.goal_points_for("DEF") == 6
    assert RULES.goal_points_for("MID") == 5
    assert RULES.goal_points_for("FWD") == 4


# 3 ------------------------------------------------------------------


def test_assist_points_are_three():
    assert RULES.assist_points == 3


# 4 ------------------------------------------------------------------


def test_clean_sheet_position_mapping():
    assert RULES.clean_sheet_points_for("GKP") == 4
    assert RULES.clean_sheet_points_for("DEF") == 4
    assert RULES.clean_sheet_points_for("MID") == 1
    assert RULES.clean_sheet_points_for("FWD") == 0


# 5 ------------------------------------------------------------------


def test_clean_sheet_requires_sixty_minutes():
    assert RULES.clean_sheet_minutes_required == 60
    assert RULES.clean_sheet_counts_conceded_while_on_pitch is True
    payload = {"p_start": 0.5, "p_cameo": 0.5, "p_60_plus": 0.0,
               "expected_minutes_if_start": 90.0, "expected_minutes_if_cameo": 15.0}
    assert xpts._clean_sheet_probability(payload, 0.0, xpts.XPtsConfig()) == 0.0


# 6 ------------------------------------------------------------------


def test_save_points_use_expected_floor_not_ratio():
    lam = 2.0
    floor_expected = xpts.expected_floor_poisson_ratio(lam, 3)
    assert floor_expected != lam / 3
    assert abs(floor_expected - 0.340126) < 1e-5  # hand-computed E[floor(S/3)] at lam=2


# 7 ------------------------------------------------------------------


def test_goals_conceded_uses_expected_floor_not_floor_of_mean():
    lam = 1.5
    assert xpts.expected_floor_half(lam) != int(lam // 2)  # 0.512 vs 0
    assert abs(xpts.expected_floor_half(lam) - 0.512447) < 1e-5


# 8 ------------------------------------------------------------------


def test_defcon_thresholds_sets_and_no_stacking():
    assert RULES.defcon_thresholds["DEF"] == 10
    assert RULES.defcon_thresholds["MID"] == 12
    assert RULES.defcon_thresholds["FWD"] == 12
    assert RULES.defcon_action_sets["DEF"] == "CBIT"
    assert RULES.defcon_action_sets["MID"] == "CBIRT"
    assert RULES.defcon_action_sets["FWD"] == "CBIRT"
    assert RULES.defcon_minimum_minutes == 0  # action-threshold based, no 60' rule
    assert RULES.defcon_never_stacks is True
    assert RULES.defcon_points == 2
    assert "GKP" not in RULES.defcon_positions


# 9 ------------------------------------------------------------------


def test_card_and_penalty_values():
    assert RULES.yellow_card_points == -1
    assert RULES.red_card_points == -3
    assert RULES.own_goal_points == -2
    assert RULES.penalty_save_points == 5
    assert RULES.penalty_miss_points == -2
    assert RULES.saves_per_point == 3
    assert RULES.goals_conceded_per_deduction == 2


# 10 -----------------------------------------------------------------


def test_scoring_hash_stable_and_payload_drift_detected():
    assert RULES.scoring_hash() == ScoringRules().scoring_hash()
    assert SCORING_RULES_VERSION.startswith("fpl_scoring_2026_27")
    payload = {
        "long_play": 2, "short_play": 1, "assists": 3,
        "goals_scored": {"DEF": 6, "FWD": 4, "GKP": 10, "MID": 5},
        "clean_sheets": {"DEF": 4, "FWD": 0, "GKP": 4, "MID": 1},
        "goals_conceded": {"DEF": -1, "FWD": 0, "GKP": -1, "MID": 0},
        "defensive_contribution": {"DEF": 2, "FWD": 2, "GKP": 0, "MID": 2},
        "yellow_cards": -1, "red_cards": -3, "own_goals": -2,
        "penalties_saved": 5, "penalties_missed": -2,
    }
    assert verify_against_scoring_dict(RULES, payload) == []
    drifted = dict(payload)
    drifted["goals_scored"] = {"DEF": 7, "FWD": 4, "GKP": 10, "MID": 5}
    mismatches = verify_against_scoring_dict(RULES, drifted)
    assert any("goal_points" in mismatch for mismatch in mismatches)
    rebuilt = scoring_rules_from_scoring_dict(payload)
    assert rebuilt.goal_points == RULES.goal_points


def test_load_stored_scoring_reads_official_payload():
    scoring, path = load_stored_scoring("K:/FPL/data/raw")
    if scoring is None:
        return  # payload not present in this environment; nothing to assert
    assert verify_against_scoring_dict(RULES, scoring) == [], path
