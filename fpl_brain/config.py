"""Configuration loading and validation."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Readable configuration or usage error."""


DEFAULT_CONFIG: dict[str, Any] = {
    "fpl_entry_id": None,
    "season": "2026/27",
    "request": {
        "timeout_connect_seconds": 5,
        "timeout_read_seconds": 20,
        "max_retries": 3,
        "backoff_base_seconds": 1.5,
        "user_agent": "fpl-brain/1.0 (personal local tool)",
        "polite_delay_seconds": 0.4,
    },
    "paths": {
        "database": "fpl.db",
        "raw_dir": "data/raw",
        "exports_dir": "data/exports",
    },
    "manual_overrides": {"free_transfers": None, "notes": ""},
    "strategy": {
        "risk_posture": "balanced",
        "wildcard_horizon": None,
        "bench_boost_plan": None,
        "free_hit_plan": None,
        "triple_captain_plan": None,
    },
    "report": {
        "fixture_horizons": [3, 5, 8],
        "scouting_stale_after_days": 14,
        "include_all_players": False,
        "player_data_selection": "squad_and_watchlist",
    },
}


def _merge(defaults: dict[str, Any], supplied: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(defaults)
    for key, value in supplied.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def _number(value: Any, name: str, integer: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be a number")
    if integer and int(value) != value:
        raise ConfigError(f"{name} must be an integer")
    if value < 0:
        raise ConfigError(f"{name} must not be negative")


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise ConfigError("Configuration root must be a JSON object")
    entry_id = config.get("fpl_entry_id")
    if entry_id is not None:
        if isinstance(entry_id, bool) or not isinstance(entry_id, int) or entry_id <= 0:
            raise ConfigError("fpl_entry_id must be a positive integer or null")
    for key in ("request", "paths", "manual_overrides", "strategy", "report"):
        if not isinstance(config.get(key), dict):
            raise ConfigError(f"{key} must be an object")
    request = config["request"]
    for key in ("timeout_connect_seconds", "timeout_read_seconds", "backoff_base_seconds", "polite_delay_seconds"):
        _number(request[key], f"request.{key}")
    _number(request["max_retries"], "request.max_retries", integer=True)
    if not isinstance(request["user_agent"], str) or not request["user_agent"].strip():
        raise ConfigError("request.user_agent must be a non-empty string")
    paths = config["paths"]
    for key in ("database", "raw_dir", "exports_dir"):
        if not isinstance(paths[key], str) or not paths[key].strip():
            raise ConfigError(f"paths.{key} must be a non-empty string")
    report = config["report"]
    if not isinstance(report["fixture_horizons"], list) or not report["fixture_horizons"]:
        raise ConfigError("report.fixture_horizons must be a non-empty list")
    for horizon in report["fixture_horizons"]:
        _number(horizon, "report.fixture_horizons", integer=True)
        if horizon == 0:
            raise ConfigError("report.fixture_horizons values must be positive")
    _number(report["scouting_stale_after_days"], "report.scouting_stale_after_days", integer=True)
    if not isinstance(report["include_all_players"], bool):
        raise ConfigError("report.include_all_players must be boolean")
    if report["player_data_selection"] not in {"squad_and_watchlist", "all"}:
        raise ConfigError("report.player_data_selection must be squad_and_watchlist or all")
    free_transfers = config["manual_overrides"].get("free_transfers")
    if free_transfers is not None:
        _number(free_transfers, "manual_overrides.free_transfers", integer=True)
    return config


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    configured_path = os.environ.get("FPL_BRAIN_CONFIG") if path is None else str(path)
    config_path = Path(configured_path or "config.json")
    if not config_path.exists():
        raise ConfigError(
            f"Configuration file not found: {config_path}. Copy config.example.json to config.json and edit it."
        )
    try:
        supplied = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid JSON in {config_path}: {exc}") from exc
    if not isinstance(supplied, dict):
        raise ConfigError(f"Configuration root in {config_path} must be a JSON object")
    return validate_config(_merge(DEFAULT_CONFIG, supplied))


def config_path(config: dict[str, Any], key: str) -> Path:
    return Path(config["paths"][key])
