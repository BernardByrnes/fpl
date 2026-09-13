"""Season rules: official game-settings derivation, FT transitions, drift checks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fpl_brain.season_rules import (
    free_transfers_after_chip,
    SeasonRulesError,
    chip_keeps_saved_free_transfers,
    free_transfers_after_gameweek,
    price_engine_matches_rules,
    season_rules_from_bootstrap,
    season_rules_from_game_settings,
    transfer_hit_cost,
)

OFFICIAL_SETTINGS = {
    "squad_squadsize": 15,
    "squad_squadplay": 11,
    "squad_team_limit": 3,
    "squad_total_spend": 1000,
    "max_extra_free_transfers": 4,
    "transfers_sell_on_fee": 0.5,
    "element_sell_at_purchase_price": False,
}


def _official_bootstrap_payload():
    path = Path(__file__).parent / "fixtures" / "bootstrap_static_sample.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_rules_derive_from_official_bootstrap_payload():
    payload = _official_bootstrap_payload()
    rules = season_rules_from_bootstrap("2026/27", payload)
    # 1 base + 4 extra free transfers = official five-transfer bank cap.
    assert rules.max_free_transfers == 5
    assert rules.sell_on_fee == 0.5
    assert rules.sell_at_purchase_price_when_falling is False
    assert rules.squad_size == 15 and rules.squad_team_limit == 3
    assert rules.transfer_hit_cost == 4


def test_missing_settings_raise():
    with pytest.raises(SeasonRulesError):
        season_rules_from_bootstrap("2026/27", {})
    with pytest.raises(SeasonRulesError):
        season_rules_from_game_settings("2026/27", None)


def test_drift_on_rule_change_is_detected():
    drifted = {**OFFICIAL_SETTINGS, "max_extra_free_transfers": 3}
    rules = season_rules_from_game_settings("2026/27", drifted)
    assert rules.max_free_transfers == 4  # would visibly differ from assumptions


def test_ft_cap_of_five_and_rollover_property():
    rules = season_rules_from_game_settings("2026/27", OFFICIAL_SETTINGS)
    saved = 0
    for _ in range(10):
        saved = free_transfers_after_gameweek(rules, saved, 0)
    assert saved == rules.max_free_transfers == 5  # never drifts past the cap
    # Using FTs deducts from the bank before rolling over.
    assert free_transfers_after_gameweek(rules, 5, 5) == 1
    assert free_transfers_after_gameweek(rules, 5, 6) == 1  # no negative bank


def test_chip_weeks_preserve_saved_fts():
    rules = season_rules_from_game_settings("2026/27", OFFICIAL_SETTINGS)
    for chip_rule in (
        rules.wildcard_ft_rule,
        rules.free_hit_ft_rule,
        rules.bench_boost_ft_rule,
        rules.triple_captain_ft_rule,
    ):
        assert chip_keeps_saved_free_transfers(chip_rule)
    # Chip weeks RETAIN the saved bank: no weekly +1 across a Wildcard/Free Hit.
    for chip in ("wildcard", "freehit"):
        assert free_transfers_after_chip(rules, chip, event_start_free_transfers=3) == 3
    # Team chips do not change the transfer process, so the normal rollover applies.
    assert free_transfers_after_chip(rules, "bboost", event_start_free_transfers=3,
                                     free_transfers_available=3) == 4


def test_hit_cost_is_constant_per_extra_transfer():
    rules = season_rules_from_game_settings("2026/27", OFFICIAL_SETTINGS)
    assert transfer_hit_cost(rules, 1) == 4
    assert transfer_hit_cost(rules, 3) == 12
    assert transfer_hit_cost(rules, 0) == 0
    with pytest.raises(ValueError):
        transfer_hit_cost(rules, -1)


def test_price_engine_drift_check_flags_fee_change():
    rules = season_rules_from_game_settings("2026/27", OFFICIAL_SETTINGS)
    assert price_engine_matches_rules(rules) == (True, "selling-price engine matches official settings")
    drifted = season_rules_from_game_settings(
        "2026/27", {**OFFICIAL_SETTINGS, "element_sell_at_purchase_price": True}
    )
    ok, message = price_engine_matches_rules(drifted)
    assert not ok and "element_sell_at_purchase_price" in message
