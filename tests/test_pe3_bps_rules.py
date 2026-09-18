"""PE-3 — BPS rule authority, primitive coverage, and the exact calculator.

The rule table is NOT verified for 2026/27 (see ``fpl_brain.bps_rules``), so these tests
prove the CALCULATOR is exact and fail-closed against SYNTHETIC rule tables.  They
deliberately do not assert that any candidate constant is the current official value —
that would launder an unverified table into a test.

The coverage tests DO assert measured facts about this repository's own data.
"""

from __future__ import annotations

import pytest

from fpl_brain import bps_rules as bps


def _table(**over) -> bps.BPSRuleTable:
    """A complete synthetic table, so a calculation never depends on the real constants."""

    base = dict(
        version="synthetic_test_table",
        per_action={
            "played_1_to_60": 3,
            "played_over_60": 6,
            "goal_from_penalty": 12,
            "assist": 9,
            "penalty_save": 8,
            "penalty_miss": -6,
            "own_goal": -6,
            "yellow_card": -3,
            "red_card": -9,
        },
        per_n_actions={"clearances_blocks_interceptions": (2, 1), "recoveries": (3, 1)},
        position_goal={"GKP": 12, "DEF": 12, "MID": 18, "FWD": 24},
        position_clean_sheet={"GKP": 12, "DEF": 12},
        position_goal_conceded={"GKP": -4, "DEF": -4},
        pass_completion_bands=(("low", 70.0, 79.999, 2), ("mid", 80.0, 89.999, 4),
                               ("high", 90.0, 100.0, 6)),
    )
    base.update(over)
    return bps.BPSRuleTable(**base)


_COMPLETE = {
    "minutes": 0, "goals_scored": 0, "penalty_goals": 0, "assists": 0, "clean_sheets": 0,
    "goals_conceded": 0,
    "penalties_saved": 0, "penalties_missed": 0, "own_goals": 0, "yellow_cards": 0,
    "red_cards": 0, "clearances_blocks_interceptions": 0, "recoveries": 0,
    "pass_attempts": 0, "pass_completion_percent": 0.0,
}


def _primitives(**over) -> dict:
    values = dict(_COMPLETE)
    values.update(over)
    return values


# ---------------------------------------------------------------------------
# Exact calculations, hand-worked against the synthetic table
# ---------------------------------------------------------------------------


def test_goalkeeper_full_house():
    """GKP: 90 min, 1 goal, 1 assist, clean sheet, 6 CBI, 9 recoveries, 1 penalty save."""
    points = bps.calculate_bps("GKP", _primitives(
        minutes=90, goals_scored=1, assists=1, clean_sheets=1,
        clearances_blocks_interceptions=6, recoveries=9, penalties_saved=1), _table())
    # 6 (over 60) + 12 (GKP goal) + 9 (assist) + 12 (GKP clean sheet) + 3 (6//2 CBI)
    # + 3 (9//3 recoveries) + 8 (penalty save) = 53
    assert points == 53


def test_defender_conceding():
    points = bps.calculate_bps("DEF", _primitives(minutes=90, goals_conceded=3), _table())
    assert points == 6 - 12  # over 60, three goals conceded at -4


def test_midfielder_scoring_and_assisting():
    points = bps.calculate_bps("MID", _primitives(minutes=75, goals_scored=2, assists=1), _table())
    assert points == 6 + 2 * 18 + 9


def test_forward_penalty_goal_replaces_the_position_value():
    """A penalty goal must not be added on top of the position goal value."""

    plain = bps.calculate_bps("FWD", _primitives(minutes=90, goals_scored=1), _table())
    assert plain == 6 + 24
    penalty = bps.calculate_bps("FWD", _primitives(minutes=90, goals_scored=1, penalty_goals=1), _table())
    assert penalty == 6 + 12  # 24 removed, 12 added


def test_cameo_minutes_band():
    assert bps.calculate_bps("MID", _primitives(minutes=1), _table()) == 3
    assert bps.calculate_bps("MID", _primitives(minutes=60), _table()) == 3
    assert bps.calculate_bps("MID", _primitives(minutes=61), _table()) == 6
    assert bps.calculate_bps("MID", _primitives(minutes=0), _table()) == 0


def test_per_n_actions_floor_division():
    table = _table()
    assert bps.calculate_bps("DEF", _primitives(minutes=90, clearances_blocks_interceptions=1), table) == 6
    assert bps.calculate_bps("DEF", _primitives(minutes=90, clearances_blocks_interceptions=2), table) == 7
    assert bps.calculate_bps("DEF", _primitives(minutes=90, clearances_blocks_interceptions=5), table) == 8
    assert bps.calculate_bps("DEF", _primitives(minutes=90, recoveries=8), table) == 8


def test_pass_completion_bands_require_thirty_attempts():
    table = _table()
    assert bps.calculate_bps("MID", _primitives(minutes=90, pass_attempts=29,
                                                pass_completion_percent=95.0), table) == 6
    assert bps.calculate_bps("MID", _primitives(minutes=90, pass_attempts=30,
                                                pass_completion_percent=95.0), table) == 12
    assert bps.calculate_bps("MID", _primitives(minutes=90, pass_attempts=30,
                                                pass_completion_percent=85.0), table) == 10
    assert bps.calculate_bps("MID", _primitives(minutes=90, pass_attempts=30,
                                                pass_completion_percent=75.0), table) == 8


def test_disciplinary_and_own_goal():
    points = bps.calculate_bps("DEF", _primitives(
        minutes=90, yellow_cards=1, own_goals=1, penalties_missed=1), _table())
    assert points == 6 - 3 - 6 - 6


def test_clean_sheet_only_counts_for_positions_in_the_table():
    """A forward's clean-sheet flag carries no BPS weight in this table."""

    assert bps.calculate_bps("FWD", _primitives(minutes=90, clean_sheets=1), _table()) == 6
    assert bps.calculate_bps("DEF", _primitives(minutes=90, clean_sheets=1), _table()) == 18


def test_unknown_primitives_are_ignored():
    """A caller may pass a superset, e.g. a whole player_gameweeks.raw_json."""

    extra = _primitives(minutes=90, influence=41.2, threat=9.0, some_unknown_field=7)
    assert bps.calculate_bps("DEF", extra, _table()) == 6


# ---------------------------------------------------------------------------
# Fail-closed on a missing primitive
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dropped", ["minutes", "assists", "clearances_blocks_interceptions",
                                     "recoveries", "pass_attempts",
                                     "pass_completion_percent"])
def test_missing_primitive_is_never_treated_as_zero(dropped):
    incomplete = _primitives()
    incomplete.pop(dropped)
    with pytest.raises(bps.BPSPrimitiveMissing) as failure:
        bps.calculate_bps("DEF", incomplete, _table())
    assert dropped in failure.value.missing


def test_missing_primitive_reports_every_absent_name():
    with pytest.raises(bps.BPSPrimitiveMissing) as failure:
        bps.calculate_bps("MID", {}, _table())
    missing = failure.value.missing
    assert "minutes" in missing and "assists" in missing and "recoveries" in missing
    assert "penalty_goals" in missing
    assert missing == sorted(missing)


def test_a_genuine_zero_is_not_the_same_as_a_missing_field():
    """Explicit zeros calculate; absent keys raise.  These are different facts."""

    assert bps.calculate_bps("MID", _primitives(minutes=90), _table()) == 6
    with pytest.raises(bps.BPSPrimitiveMissing):
        bps.calculate_bps("MID", {"minutes": 90}, _table())


# ---------------------------------------------------------------------------
# The rule table is NOT verified — this is a required property, not a bug
# ---------------------------------------------------------------------------


def test_rule_table_is_declared_unverified_and_the_reason_is_recorded():
    assert bps.BPS_RULES_VERSION == "fpl_bps_2026_27_v1.0.0"
    assert bps.BPS_RULES_VERIFICATION == "UNVERIFIED_FOR_2026_27"
    assert bps.BPS_RULES_SOURCE["usable"] is False
    assert bps.BPS_RULES_SOURCE["retrieved"]
    assert "JavaScript shell" in bps.BPS_RULES_SOURCE["reason"]


def test_the_six_season_sensitive_discrepancies_are_recorded():
    """Every value the retrieved source contradicts must stay visible."""

    rules = {item["rule"] for item in bps.CANDIDATE_RULE_TABLE_DISCREPANCIES}
    for required in ("being tackled", "clearances/blocks/interceptions",
                     "goalkeeper save", "save inside the box", "big-chance save",
                     "penalty save"):
        assert required in rules, required
    assert len(bps.CANDIDATE_RULE_TABLE_DISCREPANCIES) == 6
    for item in bps.CANDIDATE_RULE_TABLE_DISCREPANCIES:
        assert item["required_2026_27"] != item["retrieved_source"]


def test_no_candidate_rule_table_is_exported_for_use():
    """There must be no module-level BPSRuleTable a caller could mistake for the truth."""

    tables = [name for name, value in vars(bps).items()
              if isinstance(value, bps.BPSRuleTable)]
    assert tables == [], f"unexpected module-level rule tables: {tables}"


# ---------------------------------------------------------------------------
# Primitive coverage — measured facts about THIS repository
# ---------------------------------------------------------------------------


def test_coverage_classes_are_populated_and_disjoint():
    grouped = bps.coverage_by_class()
    assert set(grouped) <= {bps.AVAILABLE_EXACT, bps.AVAILABLE_APPROXIMATE,
                            bps.ABSENT, bps.NOT_POINT_IN_TIME_SAFE}
    seen: set[str] = set()
    for names in grouped.values():
        for name in names:
            assert name not in seen
            seen.add(name)
    assert seen == set(bps.PRIMITIVE_COVERAGE)


def test_the_primitives_that_are_stored_per_fixture_are_exact():
    grouped = bps.coverage_by_class()
    exact = set(grouped[bps.AVAILABLE_EXACT])
    for name in ("minutes", "goals_scored", "assists", "clean_sheets", "goals_conceded",
                 "saves", "penalties_saved", "penalties_missed", "yellow_cards",
                 "red_cards", "own_goals", "bps", "bonus",
                 "clearances_blocks_interceptions", "recoveries", "tackles"):
        assert name in exact, name


def test_exact_historical_bps_reconstruction_is_refused():
    """The phase's expected strong answer, asserted rather than assumed."""

    assert bps.exact_historical_bps_reconstruction_possible() is False
    missing = bps.missing_primitives_for_exact_bps()
    for name in ("fouls_conceded", "offsides", "shots_off_target", "pass_completion",
                 "big_chances_missed", "errors_leading_to_goal",
                 "winning_goal_identity", "penalty_vs_non_penalty_goal"):
        assert name in missing, name


def test_the_absent_primitives_carry_no_invented_source():
    for name in bps.missing_primitives_for_exact_bps():
        assert bps.PRIMITIVE_COVERAGE[name]["where"] == "-"
