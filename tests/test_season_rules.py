"""Season rules: official game-settings derivation, FT transitions, drift checks."""

from __future__ import annotations

import json
import hashlib
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
    resolve_origin_pinned_season_rules,
    verify_pinned_season_rules_evidence,
    transfer_hit_cost,
)
from fpl_brain import raw_archive, repositories as repo
from fpl_brain.database import connect_database

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


def test_continuation_rules_resolve_from_the_accepted_pre_cutoff_bootstrap_capture(tmp_path):
    conn = connect_database(tmp_path / "origin.db")
    raw_dir = tmp_path / "raw"

    def retain(captured_at: str, settings: dict[str, object]) -> None:
        run_id = repo.create_fetch_run(conn, "fetch_fpl", str(raw_dir), started_at=captured_at)
        payload = {"game_settings": settings}
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
        raw_archive.archive_raw_capture(
            raw_dir,
            source="bootstrap_static",
            observed_at=captured_at,
            body=body,
            run_id=run_id,
        )
        repo.record_bootstrap_generation(
            conn,
            captured_at=captured_at,
            accepted=True,
            official_element_count=1,
            parsed_count=1,
            persisted_count=1,
            element_ids=(1,),
            element_ids_sha256=hashlib.sha256(b"1").hexdigest(),
            acceptance_rule="fixture-complete",
            acceptance_rule_version="fixture-v1",
            fetch_run_id=run_id,
        )

    try:
        retain("2026-09-20T08:00:00Z", dict(OFFICIAL_SETTINGS))
        retain("2026-10-02T08:00:00Z", {**OFFICIAL_SETTINGS, "max_extra_free_transfers": 3})
        pinned = resolve_origin_pinned_season_rules(
            conn,
            season="2026/27",
            cutoff="2026-09-29T20:27:01Z",
            data_snapshot_sha256="d" * 64,
        )
        assert pinned.rules.max_free_transfers == 5
        assert pinned.evidence["captured_at"] == "2026-09-20T08:00:00Z"
        assert pinned.evidence["source"] == "ACCEPTED_BOOTSTRAP_CAPTURE_IN_ORIGIN_SNAPSHOT_ARCHIVE"
        assert verify_pinned_season_rules_evidence(
            pinned.evidence,
            pinned.rules,
            cutoff="2026-09-29T20:27:01Z",
            data_snapshot_sha256="d" * 64,
        ) is True
        with pytest.raises(SeasonRulesError, match="differs from the origin"):
            verify_pinned_season_rules_evidence(
                pinned.evidence,
                season_rules_from_game_settings("2026/27", {**OFFICIAL_SETTINGS, "max_extra_free_transfers": 3}),
                cutoff="2026-09-29T20:27:01Z",
                data_snapshot_sha256="d" * 64,
            )
        archived_payload = raw_dir / raw_archive.ARCHIVE_DIRNAME / pinned.evidence["archive_relative_path"]
        archived_payload.write_bytes(b"tampered")
        with pytest.raises(SeasonRulesError, match="payload digest does not verify"):
            resolve_origin_pinned_season_rules(
                conn,
                season="2026/27",
                cutoff="2026-09-29T20:27:01Z",
                data_snapshot_sha256="d" * 64,
            )
    finally:
        conn.close()
