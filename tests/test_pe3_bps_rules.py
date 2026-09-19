"""PE-3 — the verified 2026/27 BPS rule table and the exact calculator.

The rule table is VERIFIED against two official Premier League sources (see
``bps_rules.BPS_RULES_SOURCE``).  These tests pin EVERY official rule row, prove the
calculator applies the whole table, and prove the stale pre-2026/27 table would fail the
season-sensitive cases.

Vocabulary: the RULE TABLE's rows (``plays_over_60_minutes``, ``penalty_save``, …) are not
the PRIMITIVE names (``minutes``, ``penalties_saved``, …).  The mapping is explicit in
``BPSPrimitiveSpec`` so the two can never be conflated.
"""

from __future__ import annotations

import inspect

import pytest

from fpl_brain import bps_rules as b


def _zero() -> dict:
    return {name: 0 for name in b.REQUIRED_BPS_PRIMITIVES}


def _calc(position: str = "GKP", **over) -> int:
    primitives = _zero()
    primitives.update(over)
    return b.calculate_bps(position, primitives)


# ---------------------------------------------------------------------------
# RULE AUTHORITY
# ---------------------------------------------------------------------------


def test_rule_table_is_verified_and_both_official_sources_are_recorded():
    assert b.BPS_RULES_VERSION == "fpl_bps_2026_27_v1.0.0"
    assert b.BPS_RULES_VERIFICATION == "VERIFIED"
    primary = b.BPS_RULES_SOURCE["primary"]
    changes = b.BPS_RULES_SOURCE["changes"]
    assert primary["article_id"] == 4661029 and "Rules Copilot" in primary["title"]
    assert changes["article_id"] == 4679946 and "2026/27" in changes["title"]
    assert primary["publisher"] == b.BPS_RULES_SOURCE["changes"]["publisher"] == "Premier League"
    assert b.BPS_RULES_SOURCE["retrieval_attempted"]
    assert b.BPS_RULES_SOURCE["values_supplied_via"]


def test_the_stale_april_table_is_demoted_and_is_not_the_authority():
    stale = b.BPS_RULES_SOURCE["stale_pre_2026_27_reference"]
    assert stale["article_id"] == 106533
    assert stale["role"].startswith("STALE_PRE_2026_27_REFERENCE")
    assert len(stale["contradicts"]) == 6
    assert b.BPS_RULES_SOURCE["primary"]["article_id"] != stale["article_id"]


def test_ruleset_fingerprint_exposes_version_hash_and_removal():
    fingerprint = b.RULESET_FINGERPRINT
    assert fingerprint["version"] == b.BPS_RULES_VERSION
    assert fingerprint["verification"] == "VERIFIED"
    assert fingerprint["rule_rows"] == 41
    assert fingerprint["rules_hash"].startswith("sha256:")
    assert len(fingerprint["rules_hash"]) == len("sha256:") + 64
    assert fingerprint["being_tackled"] == "REMOVED_2026_27"


# ---------------------------------------------------------------------------
# COMPLETENESS — no rule row may exist without a calculation path
# ---------------------------------------------------------------------------


def test_every_official_rule_row_has_exactly_one_implemented_spec():
    assert set(b.INTERNAL_RULE_ROWS) == set(b.IMPLEMENTED_RULE_ROWS)
    assert len(b.INTERNAL_RULE_ROWS) == len(set(b.INTERNAL_RULE_ROWS)) == 41
    assert len(b.RULE_SPECS) == 41


def test_completeness_is_structural_and_the_calculator_applies_the_table():
    """The calculator must iterate the table, not merely agree with it by coincidence."""

    # `_apply_rules` is where the table is applied; `calculate_bps` and `structural_bps`
    # are thin wrappers that differ ONLY in the allow_missing flag.
    source = inspect.getsource(b._apply_rules)
    assert "for spec in rules" in source
    assert "REMOVED_2026_27" in source
    assert "has no calculation path" in source
    wrapper = inspect.getsource(b.calculate_bps)
    assert "allow_missing=False" in wrapper
    assert "allow_missing=True" in inspect.getsource(b.structural_bps)


def test_being_tackled_is_removed_and_consumes_no_primitive():
    spec = next(s for s in b.RULE_SPECS if s.rule_row == "being_tackled")
    assert spec.mode == b.BPSMode.REMOVED_2026_27
    assert spec.primitive is None
    assert "being_tackled" not in b.REQUIRED_BPS_PRIMITIVES
    assert not any("tackled" in name for name in b.REQUIRED_BPS_PRIMITIVES)


def test_every_spec_that_consumes_a_primitive_names_one_in_the_vocabulary():
    for spec in b.RULE_SPECS:
        if spec.mode != b.BPSMode.REMOVED_2026_27:
            assert spec.primitive, f"{spec.rule_row} declares no primitive"
            assert spec.primitive in b.REQUIRED_BPS_PRIMITIVES, spec.rule_row


# ---------------------------------------------------------------------------
# THE SEASON-SENSITIVE RULES
# ---------------------------------------------------------------------------


def test_being_tackled_contributes_exactly_zero():
    assert _calc(minutes=90) == 6
    assert _calc(minutes=90, successful_tackles=0) == 6


def test_cbi_is_one_point_per_three():
    assert _calc(minutes=90, clearances_blocks_interceptions=0) == 6
    assert _calc(minutes=90, clearances_blocks_interceptions=2) == 6
    assert _calc(minutes=90, clearances_blocks_interceptions=3) == 7
    assert _calc(minutes=90, clearances_blocks_interceptions=5) == 7
    assert _calc(minutes=90, clearances_blocks_interceptions=6) == 8


def test_recoveries_are_one_point_per_three():
    assert _calc(minutes=90, recoveries=2) == 6
    assert _calc(minutes=90, recoveries=3) == 7
    assert _calc(minutes=90, recoveries=9) == 9


def test_any_save_is_two_points():
    assert _calc(minutes=90, saves=0) == 6
    assert _calc(minutes=90, saves=1) == 8
    assert _calc(minutes=90, saves=5) == 16


def test_inside_box_save_adds_one_each():
    assert _calc(minutes=90, saves=4) == 14
    assert _calc(minutes=90, saves=4, inside_box_saves=2) == 16
    assert _calc(minutes=90, saves=4, inside_box_saves=4) == 18


def test_big_chance_save_adds_one_each():
    assert _calc(minutes=90, saves=4, big_chance_saves=2) == 16
    assert _calc(minutes=90, saves=4, inside_box_saves=1, big_chance_saves=1) == 16


def test_penalty_save_base_is_seven():
    """Isolating the penalty-save row: 6 minutes + 7 = 13, no other save primitive set."""

    assert _calc(minutes=90, penalties_saved=1) == 13
    assert _calc(minutes=90, penalties_saved=2) == 20


def test_penalty_save_semantics_are_ADDITIVE_not_a_replacement():
    """A penalty save may also be a save, an inside-box save and a big-chance save.

    6 (minutes) + 5 saves x 2 + 3 inside-box + 1 big-chance + 7 penalty save = 27.
    """

    points = _calc(minutes=90, saves=5, inside_box_saves=3, big_chance_saves=1,
                   penalties_saved=1)
    assert points == 27
    assert points > _calc(minutes=90, saves=5) + 7


def test_goal_values_by_position_and_penalty_replacement():
    for position, value in (("GKP", 12), ("DEF", 12), ("MID", 18), ("FWD", 24)):
        assert _calc(position, minutes=90, goals_scored=1) == 6 + value, position
    assert _calc("FWD", minutes=90, goals_scored=1, penalty_goals=1) == 6 + 12
    assert _calc("MID", minutes=90, goals_scored=2, penalty_goals=1) == 6 + 18 + 12


def test_remaining_positive_actions():
    assert _calc("MID", minutes=90, assists=1) == 15
    assert _calc("MID", minutes=90, chances_created=1) == 7
    assert _calc("MID", minutes=90, big_chances_created=1) == 9
    assert _calc("MID", minutes=90, open_play_crosses=2) == 8
    assert _calc("MID", minutes=90, successful_tackles=2) == 10
    assert _calc("MID", minutes=90, successful_dribbles=3) == 9
    assert _calc("MID", minutes=90, winning_goals=1) == 9
    assert _calc("MID", minutes=90, goal_line_clearances=1) == 15
    assert _calc("MID", minutes=90, fouls_won=2) == 8
    assert _calc("MID", minutes=90, shots_on_target=3) == 12


def test_all_negative_actions():
    assert _calc("DEF", minutes=90, goals_conceded=1) == 2
    assert _calc("DEF", minutes=90, penalties_conceded=1) == 3
    assert _calc("DEF", minutes=90, penalties_missed=1) == 0
    assert _calc("DEF", minutes=90, yellow_cards=1) == 3
    assert _calc("DEF", minutes=90, red_cards=1) == -3
    assert _calc("DEF", minutes=90, own_goals=1) == 0
    assert _calc("DEF", minutes=90, big_chances_missed=1) == 3
    assert _calc("DEF", minutes=90, errors_leading_to_goal=1) == 3
    assert _calc("DEF", minutes=90, errors_leading_to_attempt=1) == 5
    assert _calc("DEF", minutes=90, fouls_conceded=1) == 5
    assert _calc("DEF", minutes=90, offsides=1) == 5
    assert _calc("DEF", minutes=90, shots_off_target=1) == 5


def test_pass_completion_bands_need_thirty_attempts():
    assert _calc("MID", minutes=90, pass_attempts=29, pass_completion_percent=95.0) == 6
    assert _calc("MID", minutes=90, pass_attempts=30, pass_completion_percent=69.9) == 6
    assert _calc("MID", minutes=90, pass_attempts=30, pass_completion_percent=70.0) == 8
    assert _calc("MID", minutes=90, pass_attempts=30, pass_completion_percent=79.999) == 8
    assert _calc("MID", minutes=90, pass_attempts=30, pass_completion_percent=80.0) == 10
    assert _calc("MID", minutes=90, pass_attempts=30, pass_completion_percent=89.999) == 10
    assert _calc("MID", minutes=90, pass_attempts=30, pass_completion_percent=90.0) == 12


def test_minutes_bands():
    assert _calc(minutes=0) == 0
    assert _calc(minutes=1) == 3
    assert _calc(minutes=60) == 3
    assert _calc(minutes=61) == 6
    assert _calc(minutes=90) == 6


# ---------------------------------------------------------------------------
# POSITION / CONDITION SEMANTICS
# ---------------------------------------------------------------------------


def test_only_gkp_and_def_get_clean_sheet_bps():
    assert _calc("GKP", minutes=90, clean_sheets=1) == 18
    assert _calc("DEF", minutes=90, clean_sheets=1) == 18
    assert _calc("MID", minutes=90, clean_sheets=1) == 6
    assert _calc("FWD", minutes=90, clean_sheets=1) == 6


def test_only_gkp_and_def_are_deducted_for_goals_conceded():
    assert _calc("GKP", minutes=90, goals_conceded=2) == -2
    assert _calc("DEF", minutes=90, goals_conceded=2) == -2
    assert _calc("MID", minutes=90, goals_conceded=2) == 6
    assert _calc("FWD", minutes=90, goals_conceded=2) == 6


def test_a_full_hand_worked_goalkeeper_row():
    """Hand calculation: 90 min, 5 saves (3 inside box, 1 big chance), 1 penalty save,
    1 clean sheet, 4 CBI, 3 recoveries, 40 passes at 91%.

    6 + 10 + 3 + 1 + 7 + 12 + 1 + 1 + 6 = 47
    """

    assert _calc("GKP", minutes=90, saves=5, inside_box_saves=3, big_chance_saves=1,
                 penalties_saved=1, clean_sheets=1, clearances_blocks_interceptions=4,
                 recoveries=3, pass_attempts=40, pass_completion_percent=91.0) == 47


# ---------------------------------------------------------------------------
# FAIL-CLOSED INPUT HANDLING
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dropped", ["minutes", "goals_scored", "penalty_goals", "saves",
                                     "inside_box_saves", "big_chance_saves",
                                     "pass_attempts", "pass_completion_percent",
                                     "clearances_blocks_interceptions"])
def test_missing_primitive_is_never_treated_as_zero(dropped):
    incomplete = _zero()
    incomplete.pop(dropped)
    with pytest.raises(b.BPSPrimitiveMissing) as failure:
        b.calculate_bps("MID", incomplete)
    assert dropped in failure.value.missing


def test_an_explicit_zero_is_valid_but_an_absent_key_is_not():
    assert _calc(minutes=90) == 6
    with pytest.raises(b.BPSPrimitiveMissing):
        b.calculate_bps("MID", {"minutes": 90})


def test_inconsistent_counters_fail_closed():
    with pytest.raises(b.BPSPrimitiveInconsistent):
        _calc(minutes=90, goals_scored=1, penalty_goals=2)
    with pytest.raises(b.BPSPrimitiveInconsistent):
        _calc(minutes=90, saves=1, inside_box_saves=2)
    with pytest.raises(b.BPSPrimitiveInconsistent):
        _calc(minutes=90, saves=1, big_chance_saves=2)


def test_unknown_keys_are_ignored_so_a_superset_may_be_passed():
    primitives = _zero()
    primitives.update({"influence": 41.2, "threat": 9.0, "ict_index": 12.0,
                       "some_unknown_field": 7})
    assert b.calculate_bps("DEF", primitives) == 0


# ---------------------------------------------------------------------------
# THE STALE TABLE MUST FAIL THESE TESTS
# ---------------------------------------------------------------------------


def test_the_stale_pre_2026_27_table_would_fail_the_season_sensitive_cases():
    """The old constants are genuinely distinguishable from the current contract.

    Stale: CBI 1 per 2, penalty save 8, and a -1 "being tackled" deduction.  Each yields a
    DIFFERENT number from the current contract on the cases below, so these are a real
    discriminator rather than a restatement of the table.
    """

    assert _calc(minutes=90, clearances_blocks_interceptions=2) == 6 != 6 + 1
    assert _calc(minutes=90, penalties_saved=1) == 13 != 6 + 8
    assert b.RULESET_FINGERPRINT["being_tackled"] == "REMOVED_2026_27"
    stale_rows = b.BPS_RULES_SOURCE["stale_pre_2026_27_reference"]["contradicts"]
    assert any("being tackled" in row for row in stale_rows)
    assert any("penalty save 8" in row for row in stale_rows)
    assert any("1 per 2" in row for row in stale_rows)


# ---------------------------------------------------------------------------
# THE PRIMITIVE VOCABULARY
# ---------------------------------------------------------------------------


def test_the_complete_exact_primitive_vocabulary_is_available():
    required = set(b.REQUIRED_BPS_PRIMITIVES)
    for name in ("minutes", "goals_scored", "penalty_goals", "assists", "clean_sheets",
                 "goals_conceded", "saves", "inside_box_saves", "big_chance_saves",
                 "penalties_saved", "clearances_blocks_interceptions", "recoveries",
                 "chances_created", "big_chances_created", "open_play_crosses",
                 "successful_tackles", "successful_dribbles", "winning_goals",
                 "goal_line_clearances", "fouls_won", "shots_on_target", "pass_attempts",
                 "pass_completion_percent", "penalties_conceded", "penalties_missed",
                 "yellow_cards", "red_cards", "own_goals", "big_chances_missed",
                 "errors_leading_to_goal", "errors_leading_to_attempt", "fouls_conceded",
                 "offsides", "shots_off_target"):
        assert name in required, name
    assert "being_tackled" not in required


# ---------------------------------------------------------------------------
# COVERAGE — value availability and temporal safety are DIFFERENT facts
# ---------------------------------------------------------------------------


def test_coverage_reports_two_independent_dimensions():
    for entry in b.PRIMITIVE_COVERAGE:
        assert entry.availability in {b.EXACT, b.APPROXIMATE, b.ABSENT}
        assert entry.temporal_status in {b.POINT_IN_TIME_SAFE, b.UNPROVEN, b.NOT_APPLICABLE}
    by_avail = b.coverage_by_availability()
    by_time = b.coverage_by_temporal_status()
    assert by_avail[b.EXACT] and by_avail[b.APPROXIMATE]
    assert by_time[b.UNPROVEN] and by_time[b.POINT_IN_TIME_SAFE]


def test_raw_json_primitives_are_available_but_temporal_safety_is_unproven():
    """A field is not causally safe merely because its final value exists."""

    unproven = set(b.coverage_by_temporal_status()[b.UNPROVEN])
    for name in ("clearances_blocks_interceptions", "recoveries", "tackles"):
        entry = next(e for e in b.PRIMITIVE_COVERAGE if e.primitive == name)
        assert entry.availability == b.EXACT
        assert entry.temporal_status == b.UNPROVEN, name
        assert name in unproven
        assert entry.note, f"{name} must carry its limitation note"


def test_exact_historical_reconstruction_is_still_refused():
    assert b.exact_historical_bps_reconstruction_possible() is False
    for name in ("penalty_goals", "inside_box_saves", "big_chance_saves", "chances_created",
                 "pass_completion_percent", "errors_leading_to_goal", "winning_goals"):
        assert name in b.ABSENT_PRIMITIVES, name


def test_the_absent_primitives_are_exactly_required_ones_we_cannot_supply():
    required = set(b.REQUIRED_BPS_PRIMITIVES)
    for name in b.ABSENT_PRIMITIVES:
        assert name in required, name
