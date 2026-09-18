"""PE-3 — the structural BONUS/BPS challenger.

Pins the structural world model, the fixture COMPETITION (bonus is joint, never
per-player), the double-counting guard, and provenance.  The challenger is a STRUCTURAL
APPROXIMATION and every test that exercises it also asserts the unsupported components are
declared, so the approximation can never be mistaken for an exact reconstruction.
"""

from __future__ import annotations

import pytest

from fpl_brain import bonus_challenger as bc
from fpl_brain import bps_rules as bps


def _events(**over) -> dict:
    base = {"minutes": 0, "goals_scored": 0, "assists": 0, "clean_sheets": 0,
            "goals_conceded": 0, "saves": 0, "yellow_cards": 0}
    base.update(over)
    return base


def _bps(position: str, **over) -> int:
    return bc.structural_world_bps(1, position, _events(**over)).bps


# ---------------------------------------------------------------------------
# STRUCTURAL MODEL
# ---------------------------------------------------------------------------


def test_zero_minute_player_scores_nothing_structurally():
    assert _bps("MID") == 0


def test_cameo_is_the_lower_minutes_band():
    assert _bps("MID", minutes=12) == 3
    assert _bps("MID", minutes=60) == 3


def test_starter_over_sixty_gets_the_upper_band():
    assert _bps("MID", minutes=61) == 6
    assert _bps("MID", minutes=90) == 6


def test_scorer_is_positional_and_the_penalty_row_is_declared_unsupported():
    assert _bps("FWD", minutes=90, goals_scored=1) == 6 + 24
    assert _bps("MID", minutes=90, goals_scored=2) == 6 + 36
    flags = bc.structural_world_bps(1, "FWD", _events(minutes=90, goals_scored=1)).flags
    assert "PENALTY_GOAL_BPS_UNRESOLVED" in flags


def test_assister():
    assert _bps("MID", minutes=90, assists=1) == 6 + 9


def test_clean_sheet_only_for_gkp_and_def():
    assert _bps("GKP", minutes=90, clean_sheets=1) == 6 + 12
    assert _bps("DEF", minutes=90, clean_sheets=1) == 6 + 12
    assert _bps("MID", minutes=90, clean_sheets=1) == 6
    assert _bps("FWD", minutes=90, clean_sheets=1) == 6


def test_concession_deduction_only_for_gkp_and_def():
    assert _bps("GKP", minutes=90, goals_conceded=2) == 6 - 8
    assert _bps("DEF", minutes=90, goals_conceded=2) == 6 - 8
    assert _bps("MID", minutes=90, goals_conceded=2) == 6


def test_goalkeeper_saves_at_two_points_each_and_location_is_unsupported():
    assert _bps("GKP", minutes=90, saves=4) == 6 + 8
    result = bc.structural_world_bps(1, "GKP", _events(minutes=90, saves=4))
    assert "SAVE_LOCATION_BPS_UNRESOLVED" in result.flags
    assert "BIG_CHANCE_SAVE_BPS_UNRESOLVED" in result.flags
    assert "save_from_inside_box" in result.unsupported_rule_rows
    assert "save_from_big_chance" in result.unsupported_rule_rows


def test_yellow_card():
    assert _bps("DEF", minutes=90, yellow_cards=1) == 6 - 3


def test_red_cards_are_unsupported_rather_than_assumed_zero():
    """The simulator does not produce red cards, so the row must be NAMED, not applied."""

    result = bc.structural_world_bps(1, "DEF", _events(minutes=90))
    assert "red_card" in result.unsupported_rule_rows


def test_a_missing_STRUCTURAL_event_is_a_programming_error():
    with pytest.raises(bps.BPSPrimitiveMissing):
        bc.structural_world_bps(1, "MID", {"minutes": 90})


def test_every_unsupported_rule_row_is_declared_on_the_result():
    result = bc.structural_world_bps(1, "MID", _events(minutes=90))
    assert result.unsupported_rule_rows
    # every named hole must be a real table row
    table = {spec.rule_row for spec in bps.RULE_SPECS}
    for row in result.unsupported_rule_rows:
        assert row in table, row
    # ...and none of the structurally supported rows may be in it
    for row in ("plays_1_to_60_minutes", "plays_over_60_minutes", "assist",
                "any_save", "yellow_card", "fwd_non_penalty_goal"):
        assert row not in result.unsupported_rule_rows, row


# ---------------------------------------------------------------------------
# THE FIXTURE COMPETITION — bonus is joint
# ---------------------------------------------------------------------------


def test_another_players_event_changes_my_bonus():
    """The defining property: a player's bonus depends on everyone else in the fixture."""

    mine = {1: 30, 2: 20, 3: 10}
    before = bc.allocate_fixture_world(mine).bonus_by_player[1]
    # someone else has a huge game; my BPS is unchanged
    after = bc.allocate_fixture_world({**mine, 4: 90}).bonus_by_player[1]
    assert before == 3
    assert after == 2, "a rival's event must be able to change my bonus"


def test_bonus_is_never_determined_independently():
    """Same player BPS, different company, different bonus."""

    alone = bc.allocate_fixture_world({1: 40}).bonus_by_player[1]
    crowded = bc.allocate_fixture_world({1: 40, 2: 40, 3: 40, 4: 40}).bonus_by_player[1]
    assert alone == 3 and crowded == 3
    worse = bc.allocate_fixture_world({1: 40, 2: 50, 3: 60, 4: 70}).bonus_by_player[1]
    assert worse == 0


def test_a_fixture_world_can_award_more_than_six():
    world = {1: 50, 2: 50, 3: 30}
    assert bc.allocate_fixture_world(world).total_bonus == 7


def test_aggregate_over_worlds_reports_the_required_outputs():
    worlds = [
        {1: 60, 2: 55, 3: 55, 4: 40},
        {1: 40, 2: 55, 3: 55, 4: 60},
        {1: 50, 2: 50, 3: 50, 4: 50},
    ]
    summary = bc.aggregate_worlds(worlds)
    assert summary.worlds == 3
    for player_id in (1, 2, 3, 4):
        assert player_id in summary.expected_bonus
        assert player_id in summary.p_bonus_any
        assert player_id in summary.p_bonus_3
        assert player_id in summary.mean_bps_proxy
    # player 4 tops world 2 and ties the all-50 world 3, so P(bonus=3) = 2/3
    assert summary.p_bonus_3[4] == pytest.approx(2 / 3)
    assert summary.p_bonus_3[1] == pytest.approx(2 / 3)
    assert summary.mean_bps_proxy[1] == pytest.approx((60 + 40 + 50) / 3)


def test_ranking_changes_are_detected_across_worlds():
    changed = bc.aggregate_worlds([{1: 60, 2: 50}, {1: 50, 2: 60}])
    assert changed.ranking_changed_across_worlds is True
    stable = bc.aggregate_worlds([{1: 60, 2: 50}, {1: 61, 2: 50}])
    assert stable.ranking_changed_across_worlds is False


def test_declared_outputs_match_the_contract():
    summary = bc.aggregate_worlds([{1: 60, 2: 50, 3: 40}])
    payload = summary.as_dict()
    assert payload["grain"] == "player_x_fixture_x_world"
    assert payload["model_version"] == bc.BONUS_BPS_MODEL_VERSION
    assert set(payload) >= {"expected_bonus", "p_bonus_any", "p_bonus_3", "mean_bps_proxy",
                            "p_bonus_2plus", "unsupported_rule_rows", "flags"}


# ---------------------------------------------------------------------------
# DOUBLE COUNTING — the centred construction
# ---------------------------------------------------------------------------


def test_the_centred_construction_has_exactly_the_background_mean():
    """E[world_bps] == background.  This is the no-double-counting identity."""

    background = 42.0
    structural = [6.0, 24.0, 6.0, 42.0, 6.0]
    centred = bc.centred_world_bps(background, structural)
    assert sum(centred) / len(centred) == pytest.approx(background, abs=1e-12)


def test_the_centred_construction_preserves_deviations():
    """Only the MEAN is re-anchored; world-to-world movement is untouched."""

    background, structural = 30.0, [10.0, 20.0, 30.0]
    centred = bc.centred_world_bps(background, structural)
    assert centred[1] - centred[0] == pytest.approx(structural[1] - structural[0])
    assert centred[2] - centred[0] == pytest.approx(structural[2] - structural[0])
    assert sum(centred) / 3 == pytest.approx(background)


def test_an_uncentred_construction_would_double_count():
    """Show the failure mode this construction exists to prevent."""

    background, structural = 30.0, [10.0, 20.0, 30.0]
    uncentred_mean = background + sum(structural) / len(structural)
    assert uncentred_mean == 50.0 > background  # the events were counted twice
    assert sum(bc.centred_world_bps(background, structural)) / 3 == pytest.approx(background)


def test_empty_world_list_is_handled():
    assert bc.centred_world_bps(10.0, []) == []
    assert bc.aggregate_worlds([]).worlds == 0


# ---------------------------------------------------------------------------
# BACKGROUND PRIOR — causal, and no history predicate of its own
# ---------------------------------------------------------------------------


def test_background_bps_uses_only_positive_minute_rows():
    rows = [{"minutes": 90, "bps": 30}, {"minutes": 90, "bps": 24},
            {"minutes": 0, "bps": 0}]
    background = bc.background_bps_from_rows(rows, shrinkage_minutes=0)
    assert background.sample_minutes == 180
    assert background.source_rows == 2
    assert background.bps_per_90 == pytest.approx(27.0)


def test_background_bps_shrinks_toward_the_fallback():
    small = bc.background_bps_from_rows([{"minutes": 90, "bps": 90}], shrinkage_minutes=900)
    assert small.bps_per_90 < 90.0
    assert small.bps_per_90 == pytest.approx((90 / 990) * 90 + (900 / 990) * 0.0)


def test_background_bps_with_no_evidence_is_the_fallback():
    empty = bc.background_bps_from_rows([], fallback_per_90=5.0)
    assert empty.source_rows == 0 and empty.bps_per_90 == 5.0


def test_expected_bps_scales_with_minutes():
    background = bc.BackgroundBps(bps_per_90=18.0, sample_minutes=100, source_rows=1,
                                 shrinkage_minutes=0)
    assert background.expected_bps(90) == pytest.approx(18.0)
    assert background.expected_bps(45) == pytest.approx(9.0)
    assert background.expected_bps(0) == 0.0


# ---------------------------------------------------------------------------
# RNG INVARIANCE AND PROVENANCE
# ---------------------------------------------------------------------------


def test_the_challenger_consumes_no_randomness():
    """No RNG is imported or constructed anywhere in the challenger."""

    import inspect

    for module in (bc,):
        source = inspect.getsource(module)
        for banned in ("import random", "random.Random", "random.random",
                       "numpy", "default_rng", "seed("):
            assert banned not in source, f"{module.__name__} touches randomness: {banned}"


def test_the_challenger_imports_nothing_from_the_monte_carlo_simulator():
    """It must be a pure consumer of world events, not a participant in the draws."""

    import inspect

    source = inspect.getsource(bc)
    assert "import monte_carlo" not in source
    assert "from . import monte_carlo" not in source


def test_provenance_is_declared():
    assert bc.BONUS_BPS_MODEL_VERSION == "bonus_bps_v1.0.0"
    assert bps.BPS_RULES_VERSION == "fpl_bps_2026_27_v1.0.0"
    assert bps.RULESET_FINGERPRINT["rules_hash"].startswith("sha256:")
    assert bc.STRUCTURAL_EVENT_NAMES == ("minutes", "goals_scored", "assists",
                                         "clean_sheets", "goals_conceded", "saves",
                                         "yellow_cards")


def test_unsupported_components_are_always_declared_with_a_reason():
    names = bc.unsupported_component_names()
    assert set(names) == {"PENALTY_GOAL_BPS_UNRESOLVED", "SAVE_LOCATION_BPS_UNRESOLVED",
                          "BIG_CHANCE_SAVE_BPS_UNRESOLVED"}
    for flag, reason in bc.UNSUPPORTED_COMPONENT_FLAGS:
        assert flag in names
        assert reason and len(reason) > 20
