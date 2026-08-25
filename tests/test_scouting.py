from __future__ import annotations

import copy
import json

import pytest

from fpl_brain import repositories as repo
from fpl_brain.config import DEFAULT_CONFIG
from fpl_brain.database import connect_database
from fpl_brain.models import PlayerRecord, PositionRecord, TeamRecord
from fpl_brain.scouting import ScoutingError, import_scouting, validate_document


def _config(tmp_path):
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["paths"]["database"] = str(tmp_path / "fpl.db")
    config["paths"]["raw_dir"] = str(tmp_path / "raw")
    config["paths"]["exports_dir"] = str(tmp_path / "exports")
    return config


def test_canonical_shorthand_resolution_staleness_inputs_and_dry_run(tmp_path):
    config = _config(tmp_path)
    conn = connect_database(config["paths"]["database"])
    with conn:
        repo.upsert_teams(conn, [TeamRecord(id=1, name="Arsenal", short_name="ARS")])
        repo.upsert_positions(conn, [PositionRecord(id=3, singular_name_short="MID")])
        repo.upsert_players(conn, [PlayerRecord(id=1, web_name="Raya", full_name="David Raya Martín", norm_name="david raya martin", team_id=1, element_type=3)])
    conn.close()
    source = tmp_path / "reports" / "gw1.json"
    source.parent.mkdir()
    source.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "generated_at": "2026-08-19T00:00:00Z",
                "players": [
                    {
                        "player_name": "david raya martin",
                        "team_hint": "ARS",
                        "observations": [
                            {"key": "start_probability", "value": 90, "confidence": "high", "observed_at": "2026-08-19T00:00:00Z"},
                            {"key": "rotation_risk", "value": "medium", "confidence": "medium", "observed_at": "2026-08-19T00:00:00Z"},
                            {"key": "future_unknown", "value": "watch", "confidence": "low", "observed_at": "2026-08-19T00:00:00Z"},
                        ],
                    },
                    {"player_name": "No Such Player", "start_probability": 50, "confidence": "low"},
                ],
            }
        ),
        encoding="utf-8",
    )
    result = import_scouting(config, source)
    assert result["players_resolved"] == 1
    assert len(result["unresolved"]) == 1
    assert (tmp_path / "rejected").exists()
    conn = connect_database(config["paths"]["database"])
    rows = conn.execute("SELECT key,value_text,value_num FROM scouting_notes ORDER BY id").fetchall()
    assert rows[0][0] == "start_probability"
    assert rows[0][2] == 90
    assert rows[1][0] == "rotation_risk"
    assert rows[1][1] == "medium"
    conn.close()

    dry_source = tmp_path / "reports" / "dry.json"
    dry_source.write_text(json.dumps({"schema_version": "1.0", "players": [{"player_name": "No Such Player", "start_probability": 10}]}), encoding="utf-8")
    dry = import_scouting(config, dry_source, dry_run=True)
    assert dry["status"] == "dry-run"
    conn = connect_database(config["paths"]["database"])
    assert conn.execute("SELECT COUNT(*) FROM scouting_imports").fetchone()[0] == 1
    conn.close()


@pytest.mark.parametrize(
    "document",
    [
        {"schema_version": "1.0", "players": [{"player_name": "Player", "observations": [{"key": "likely_role"}]}]},
        {"schema_version": "1.0", "players": [{"player_name": "Player", "confidence": "certain", "start_probability": 80}]},
        {"schema_version": "1.0", "players": [{"player_name": "Player", "observations": [{"key": "start_probability", "value": 80, "confidence": "certain"}]}]},
        {"schema_version": "1.0", "players": [{"player_name": "Player", "observations": [{"key": "start_probability", "value": 80}], "start_probability": 90}]},
    ],
)
def test_invalid_scouting_observation_confidence_and_mixed_schema_are_rejected(document):
    with pytest.raises(ScoutingError):
        validate_document(document)


def test_invalid_scouting_root_is_rejected():
    with pytest.raises(ScoutingError):
        validate_document([])


def test_canonical_observation_text_and_gameweek_context_are_stored(tmp_path):
    config = _config(tmp_path)
    conn = connect_database(config["paths"]["database"])
    with conn:
        repo.upsert_teams(conn, [TeamRecord(id=1, name="Arsenal", short_name="ARS")])
        repo.upsert_positions(conn, [PositionRecord(id=3, singular_name_short="MID")])
        repo.upsert_players(conn, [PlayerRecord(id=1, web_name="Raya", full_name="David Raya", norm_name="david raya", team_id=1, element_type=3)])
    conn.close()
    source = tmp_path / "canonical.json"
    source.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "generated_at": "2026-08-19T00:00:00Z",
                "gameweek": 2,
                "players": [
                    {
                        "player_id": 1,
                        "player_name": "Raya",
                        "observations": [
                            {
                                "key": "start_probability",
                                "value": 80,
                                "confidence": "medium",
                                "observation": "Expected to start if fit.",
                                "gameweek_context": 3,
                                "observed_at": "2026-08-19T00:00:00Z",
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    result = import_scouting(config, source)
    assert result["notes"] == 1
    conn = connect_database(config["paths"]["database"])
    row = conn.execute("SELECT observation, gameweek_context FROM scouting_notes").fetchone()
    conn.close()
    assert row[0] == "Expected to start if fit."
    assert row[1] == 3
