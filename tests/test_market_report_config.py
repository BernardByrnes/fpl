from __future__ import annotations

import json

import pytest

from fpl_brain.market_report import MarketReportConfigError, load_market_config, market_config_hash


def test_market_config_merges_deterministically_independent_of_key_order(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(
        json.dumps({"output": {"base_dir": "custom"}, "sections": {"gw_stars": {"rows": 7}}}),
        encoding="utf-8",
    )
    second.write_text(
        '{"sections":{"gw_stars":{"rows":7}},"output":{"base_dir":"custom"}}',
        encoding="utf-8",
    )
    config_one = load_market_config(first)
    config_two = load_market_config(second)
    assert config_one == config_two
    assert market_config_hash(config_one) == market_config_hash(config_two)


def test_market_config_rejects_unknown_and_unsafe_values(tmp_path):
    unknown = tmp_path / "unknown.json"
    unknown.write_text('{"sections":{"budget_radar":{"enabled":true}}}', encoding="utf-8")
    with pytest.raises(MarketReportConfigError, match="Unknown"):
        load_market_config(unknown)

    unsafe = tmp_path / "unsafe.json"
    unsafe.write_text('{"output":{"include_wall_clock_generated_at":true}}', encoding="utf-8")
    with pytest.raises(MarketReportConfigError, match="deterministic"):
        load_market_config(unsafe)

    negative_rows = tmp_path / "negative_rows.json"
    negative_rows.write_text('{"sections":{"gw_stars":{"rows":-1}}}', encoding="utf-8")
    with pytest.raises(MarketReportConfigError, match="rows"):
        load_market_config(negative_rows)

    incoherent = tmp_path / "incoherent.json"
    incoherent.write_text(
        '{"sections":{"minutes_watch":{"lookback_completed_fixtures":1,"minimum_baseline_fixtures":2}}}',
        encoding="utf-8",
    )
    with pytest.raises(MarketReportConfigError, match="cannot exceed"):
        load_market_config(incoherent)

    unproven_defcon = tmp_path / "unproven_defcon.json"
    unproven_defcon.write_text(
        '{"sections":{"defensive_contributions":{"require_per_fixture_provenance":false}}}',
        encoding="utf-8",
    )
    with pytest.raises(MarketReportConfigError, match="per_fixture_provenance"):
        load_market_config(unproven_defcon)
