from __future__ import annotations

import copy

import pytest

import fpl_brain.ingest as ingest
from fpl_brain.config import DEFAULT_CONFIG
from fpl_brain.database import connect_database
from fpl_brain.models import PlayerRecord, TeamRecord
from fpl_brain.parsers import BootstrapValidationError


class FakeClient:
    def __init__(self, *args, **kwargs):
        return None

    def get_bootstrap_static(self):
        return getattr(self, "payload", {})

    def get_fixtures(self):
        return []

    def close(self):
        return None


def _config(tmp_path):
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["paths"]["database"] = str(tmp_path / "fpl.db")
    config["paths"]["raw_dir"] = str(tmp_path / "raw")
    config["paths"]["exports_dir"] = str(tmp_path / "exports")
    return config


def _seed_active(config):
    conn = connect_database(config["paths"]["database"])
    with conn:
        repo_records = [TeamRecord(id=1, name="One")]
        from fpl_brain import repositories as repo

        repo.upsert_teams(conn, repo_records)
        repo.upsert_players(conn, [PlayerRecord(id=10, web_name="Existing", full_name="Existing", team_id=1)])
    conn.close()


@pytest.mark.parametrize(
    "payload",
    [
        {"teams": [{"id": 1, "name": "One"}], "events": []},
        {"elements": [], "teams": [{"id": 1, "name": "One"}], "events": []},
        {"elements": [{}], "teams": [{"id": 1, "name": "One"}], "events": []},
        {"elements": [{"id": 1}], "teams": [{"id": 1, "name": "One"}], "events": []},
    ],
)
def test_invalid_bootstrap_never_deactivates_existing_players(monkeypatch, tmp_path, payload):
    config = _config(tmp_path)
    _seed_active(config)

    class PayloadClient(FakeClient):
        def get_bootstrap_static(self):
            return payload

    monkeypatch.setattr(ingest, "FplClient", PayloadClient)
    with pytest.raises(BootstrapValidationError):
        ingest.run_fetch(config)
    conn = connect_database(config["paths"]["database"])
    assert conn.execute("SELECT is_active FROM players WHERE id=10").fetchone()[0] == 1
    assert conn.execute("SELECT status FROM fetch_runs ORDER BY id DESC LIMIT 1").fetchone()[0] == "failed"
    conn.close()


def test_parser_failure_partway_preserves_existing_catalog(monkeypatch, tmp_path):
    config = _config(tmp_path)
    _seed_active(config)

    def fail_parse(*args, **kwargs):
        raise ValueError("parser failed partway")

    monkeypatch.setattr(ingest, "parse_bootstrap", fail_parse)
    monkeypatch.setattr(ingest, "FplClient", FakeClient)
    with pytest.raises(ValueError, match="parser failed"):
        ingest.run_fetch(config)
    conn = connect_database(config["paths"]["database"])
    assert conn.execute("SELECT is_active FROM players WHERE id=10").fetchone()[0] == 1
    conn.close()
