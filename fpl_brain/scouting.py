"""Validated scouting imports kept strictly separate from official facts."""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

from . import repositories as repo
from .config import config_path
from .database import connect_database
from .utils import ensure_directory, json_text, normalise_name, utc_now

LOGGER = logging.getLogger(__name__)

NUMERIC_KEYS = {
    "start_probability",
    "penalty_probability",
    "freekick_probability",
    "corner_probability",
    "injury_uncertainty",
    "transfer_exit_risk",
    "expected_minutes",
}
ORDINAL_KEYS = {
    "rotation_risk",
    "role_security_5gw",
    "competition_for_position",
    "european_congestion",
    "attacking_threat",
    "defcon_potential",
}
TEXT_KEYS = {"likely_role", "set_piece_role", "tactical_note"}
UNIT_DEFAULTS = {
    "start_probability": "percent",
    "penalty_probability": "percent",
    "freekick_probability": "percent",
    "corner_probability": "percent",
    "injury_uncertainty": "percent",
    "transfer_exit_risk": "percent",
    "expected_minutes": "minutes",
}
CATEGORY_DEFAULTS = {
    **{key: "minutes" for key in ("start_probability", "expected_minutes", "rotation_risk", "role_security_5gw", "competition_for_position", "european_congestion")},
    **{key: "setpieces" for key in ("penalty_probability", "freekick_probability", "corner_probability", "set_piece_role")},
    **{key: "risk" for key in ("injury_uncertainty", "transfer_exit_risk")},
    **{key: "role" for key in ("likely_role", "attacking_threat", "defcon_potential")},
}
SHORTHAND_ALIASES = {"five_gameweek_role_security": "role_security_5gw"}
SUPPORTED_KEYS = NUMERIC_KEYS | ORDINAL_KEYS | TEXT_KEYS
VALID_CONFIDENCE = {"low", "medium", "high"}


class ScoutingError(ValueError):
    """Invalid scouting input or an import conflict."""


class DuplicateScoutingImport(ScoutingError):
    """The same source file was already imported."""


def validate_document(document: Any) -> None:
    if not isinstance(document, dict):
        raise ScoutingError("Scouting document must be a JSON object")
    if document.get("schema_version") != "1.0":
        raise ScoutingError("scouting schema_version must be '1.0'")
    if not isinstance(document.get("players"), list):
        raise ScoutingError("scouting players must be an array")
    for index, player in enumerate(document["players"]):
        if not isinstance(player, dict) or not isinstance(player.get("player_name"), str) or not player["player_name"].strip():
            raise ScoutingError(f"players[{index}].player_name must be a non-empty string")
        if player.get("confidence") is not None and player.get("confidence") not in VALID_CONFIDENCE:
            raise ScoutingError(f"players[{index}].confidence must be low, medium, or high")
        observations = player.get("observations")
        if observations is not None and not isinstance(observations, list):
            raise ScoutingError(f"players[{index}].observations must be an array")
        if isinstance(observations, list):
            has_canonical = any(isinstance(item, dict) for item in observations)
            metadata_keys = {"player_name", "player_id", "team_hint", "position_hint", "confidence", "summary", "observations", "evidence"}
            shorthand_keys = {key for key in player if key not in metadata_keys}
            if has_canonical and shorthand_keys:
                raise ScoutingError(f"players[{index}] cannot mix canonical observations with shorthand fields: {sorted(shorthand_keys)}")
            if not has_canonical:
                for key in shorthand_keys:
                    if player.get(key) is None:
                        raise ScoutingError(f"players[{index}].{key} must have a value")
            for item in observations:
                if not isinstance(item, (dict, str)):
                    raise ScoutingError(f"players[{index}].observations entries must be objects or strings")
                if isinstance(item, dict):
                    if not isinstance(item.get("key"), str) or not item["key"].strip():
                        raise ScoutingError(f"players[{index}].observations object needs a non-empty key")
                    if "value" not in item or item.get("value") is None:
                        raise ScoutingError(f"players[{index}].observations[{item.get('key')!r}] needs a value")
                    confidence = item.get("confidence")
                    if confidence is not None and confidence not in VALID_CONFIDENCE:
                        raise ScoutingError(f"players[{index}].observations[{item.get('key')!r}].confidence must be low, medium, or high")
        for key in set(player) & (SUPPORTED_KEYS | set(SHORTHAND_ALIASES)):
            if player.get(key) is None:
                raise ScoutingError(f"players[{index}].{key} must have a value")


def _read_document(path: str | Path) -> tuple[dict[str, Any], str, str]:
    source = Path(path)
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ScoutingError(f"Could not read scouting JSON {source}: {exc}") from exc
    validate_document(document)
    return document, digest, source.name


def _shorthand_observations(player: dict[str, Any], document: dict[str, Any]) -> list[dict[str, Any]]:
    top_observed = document.get("generated_at") or utc_now()
    top_expires = document.get("default_expires_at")
    player_confidence = player.get("confidence", "medium")
    common_evidence = player.get("evidence", [])
    common_note = player.get("summary")
    observations: list[dict[str, Any]] = []
    metadata_keys = {"player_name", "player_id", "team_hint", "position_hint", "confidence", "summary", "observations", "evidence"}
    for key, value in player.items():
        canonical_key = SHORTHAND_ALIASES.get(key, key)
        if key in metadata_keys:
            continue
        observations.append(
            {
                "key": canonical_key,
                "value": value,
                "unit": UNIT_DEFAULTS.get(canonical_key, "scale" if canonical_key in ORDINAL_KEYS else "none"),
                "category": CATEGORY_DEFAULTS.get(canonical_key, "other"),
                "confidence": player_confidence,
                "observed_at": top_observed,
                "expires_at": top_expires,
                "evidence": common_evidence,
                "note": common_note,
            }
        )
    narrative = player.get("observations", [])
    if isinstance(narrative, list):
        for text in narrative:
            if isinstance(text, str):
                observations.append(
                    {
                        "key": "tactical_note",
                        "value": text,
                        "unit": "none",
                        "category": "other",
                        "confidence": player_confidence,
                        "observed_at": top_observed,
                        "expires_at": top_expires,
                        "evidence": common_evidence,
                        "note": text,
                    }
                )
    return observations


def expand_observations(player: dict[str, Any], document: dict[str, Any]) -> list[dict[str, Any]]:
    observations = player.get("observations")
    if isinstance(observations, list) and any(isinstance(item, dict) for item in observations):
        expanded = []
        for item in observations:
            if isinstance(item, dict):
                expanded.append(dict(item))
            elif isinstance(item, str):
                expanded.append(
                    {
                        "key": "tactical_note",
                        "value": item,
                        "unit": "none",
                        "category": "other",
                        "confidence": player.get("confidence", "medium"),
                        "observed_at": document.get("generated_at") or utc_now(),
                        "expires_at": document.get("default_expires_at"),
                        "evidence": player.get("evidence", []),
                        "note": item,
                    }
                )
        return expanded
    return _shorthand_observations(player, document)


def _candidate_description(row: dict[str, Any]) -> dict[str, Any]:
    return {"id": row.get("id"), "full_name": row.get("full_name"), "web_name": row.get("web_name"), "team": row.get("team_name")}


def resolve_player(conn: sqlite3.Connection, player: dict[str, Any]) -> tuple[dict[str, Any] | None, str, list[dict[str, Any]]]:
    candidates = repo.player_candidates(conn)
    explicit_id = player.get("player_id")
    if explicit_id is not None:
        try:
            row = repo.get_player(conn, int(explicit_id))
        except (TypeError, ValueError):
            row = None
        if row is not None:
            return row, "explicit player_id", [_candidate_description(row)]
        return None, "explicit player_id does not exist", []

    target = normalise_name(player.get("player_name"))
    full_matches = [row for row in candidates if normalise_name(row.get("full_name")) == target]
    web_matches = [row for row in candidates if normalise_name(row.get("web_name")) == target]
    matches = full_matches or web_matches
    if not matches:
        matches = [
            row
            for row in candidates
            if target and (target in normalise_name(row.get("full_name")) or target in normalise_name(row.get("web_name")))
        ]
    if len(matches) > 1 and player.get("team_hint"):
        team_target = normalise_name(str(player["team_hint"]))
        narrowed = [
            row
            for row in matches
            if team_target in {normalise_name(row.get("team_name")), normalise_name(row.get("team_short_name"))}
        ]
        if narrowed:
            matches = narrowed
    if len(matches) > 1 and player.get("position_hint"):
        position_target = normalise_name(str(player["position_hint"]))
        narrowed = [row for row in matches if position_target == normalise_name(row.get("position_short_name"))]
        if narrowed:
            matches = narrowed
    descriptions = [_candidate_description(row) for row in matches]
    if len(matches) == 1:
        return matches[0], "name match", descriptions
    if not matches:
        return None, "no candidate", descriptions
    return None, "ambiguous candidate set", descriptions


def _value_fields(key: str, value: Any) -> tuple[str | None, float | None]:
    if isinstance(value, bool):
        return str(value).lower(), None
    if key in NUMERIC_KEYS:
        try:
            return None, float(value)
        except (TypeError, ValueError):
            LOGGER.warning("Scouting numeric key %s has non-numeric value %r", key, value)
            return str(value), None
    if isinstance(value, (int, float)):
        return None, float(value)
    return (None if value is None else str(value)), None


def _normalise_note(
    player: dict[str, Any],
    note: dict[str, Any],
    document: dict[str, Any],
) -> dict[str, Any]:
    key = str(note.get("key", "unknown"))
    key = SHORTHAND_ALIASES.get(key, key)
    if key not in NUMERIC_KEYS | ORDINAL_KEYS | TEXT_KEYS:
        LOGGER.warning("Unknown scouting key %s; importing as other", key)
        category = "other"
    else:
        category = note.get("category") or CATEGORY_DEFAULTS.get(key, "other")
    value = note.get("value")
    value_text, value_num = _value_fields(key, value)
    unit = note.get("unit") or UNIT_DEFAULTS.get(key, "scale" if key in ORDINAL_KEYS else "none")
    if key in NUMERIC_KEYS and value_num is not None:
        low, high = (0, 90) if key == "expected_minutes" else (0, 100)
        if not low <= value_num <= high:
            LOGGER.warning("Scouting value for %s is outside expected range: %s", key, value_num)
    confidence = note.get("confidence") or player.get("confidence") or "low"
    if confidence not in VALID_CONFIDENCE:
        raise ScoutingError(f"scouting confidence must be low, medium, or high, got {confidence!r}")
    return {
        "key": key,
        "category": category,
        "value_text": value_text,
        "value_num": value_num,
        "value_unit": unit,
        "confidence": confidence,
        "evidence": note.get("evidence", player.get("evidence", [])) or [],
        # The documented canonical contract (FPL_BRAIN_SCOUT_CONTEXT.md §5/§7 and
        # LUNA_SCOUT_PROTOCOL_V1.md section E) names the human-readable field
        # "observation"; "note" is retained first for backwards compatibility.
        "observation": note.get("note") or note.get("observation") or player.get("summary"),
        # Observation-level gameweek context wins over the root gameweek when supplied.
        "gameweek_context": note.get("gameweek_context") if note.get("gameweek_context") is not None else document.get("gameweek"),
        "observed_at": note.get("observed_at") or document.get("generated_at") or utc_now(),
        "expires_at": note.get("expires_at") or document.get("default_expires_at"),
        "agent": document.get("agent"),
    }


def _read_only_connection(path: Path) -> sqlite3.Connection | None:
    if not path.exists():
        return None
    conn = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def import_scouting(
    config: dict[str, Any],
    source_path: str | Path,
    dry_run: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    document, digest, file_name = _read_document(source_path)
    database_path = config_path(config, "database")
    conn = _read_only_connection(database_path) if dry_run else connect_database(database_path)
    if conn is None:
        if dry_run:
            return {"status": "dry-run", "players_total": len(document["players"]), "players_resolved": 0, "notes": 0, "unresolved": []}
        raise ScoutingError(f"Database does not exist yet: {database_path}. Run fetch_fpl.py first.")
    try:
        if repo.has_scouting_hash(conn, digest) and not force:
            raise DuplicateScoutingImport(f"{file_name} was already imported; use --force to append it again")
        resolved: list[tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]] = []
        unresolved: list[dict[str, Any]] = []
        for player in document["players"]:
            row, reason, candidates = resolve_player(conn, player)
            if row is None:
                unresolved.append({"player_name": player.get("player_name"), "reason": reason, "candidates": candidates})
            else:
                resolved.append((player, row, candidates))
        notes = [_normalise_note(player, note, document) | {"player_id": row["id"]} for player, row, _ in resolved for note in expand_observations(player, document)]
        result = {
            "status": "dry-run" if dry_run else "success",
            "players_total": len(document["players"]),
            "players_resolved": len(resolved),
            "notes": len(notes),
            "unresolved": unresolved,
        }
        if dry_run:
            return result
        with conn:
            import_id = repo.insert_scouting_import(
                conn,
                {
                    "source_file": str(source_path),
                    "file_sha256": digest,
                    "schema_version": document.get("schema_version"),
                    "agent": document.get("agent"),
                    "generated_at": document.get("generated_at"),
                    "gameweek": document.get("gameweek"),
                    "players_total": len(document["players"]),
                    "players_resolved": len(resolved),
                    "notes_inserted": len(notes),
                    "unresolved": unresolved,
                },
            )
            for note in notes:
                note["import_id"] = import_id
                repo.insert_scouting_note(conn, note)
        if unresolved:
            source_parent = Path(source_path).resolve().parent
            scouting_root = source_parent.parent if source_parent.name == "reports" else Path("scouting")
            rejected_dir = ensure_directory(scouting_root / "rejected")
            rejected_name = f"{utc_now().replace(':', '').replace('-', '')}_{file_name}"
            (rejected_dir / rejected_name).write_text(json.dumps(unresolved, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return result | {"import_id": import_id}
    finally:
        conn.close()
