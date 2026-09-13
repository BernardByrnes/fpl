from __future__ import annotations

import copy
import json

import pytest
import requests

from fpl_brain.api import FplClient, FplTimeoutError
from fpl_brain.config import DEFAULT_CONFIG


class FakeResponse:
    def __init__(self, status_code: int, body: str = "{}", headers: dict[str, str] | None = None):
        self.status_code = status_code
        self.text = body
        self.headers = headers or {}


class FakeSession:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def get(self, url, timeout):
        self.calls.append((url, timeout))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def close(self):
        return None


def _config():
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["request"]["polite_delay_seconds"] = 0
    config["request"]["backoff_base_seconds"] = 2
    return config


def test_retry_timeout_retry_after_and_raw_persistence(monkeypatch, tmp_path):
    config = _config()
    config["request"]["max_retries"] = 1
    sleeps = []
    monkeypatch.setattr("fpl_brain.api.time.sleep", lambda delay: sleeps.append(delay))
    monkeypatch.setattr("fpl_brain.api.random.uniform", lambda *_: 0.0)
    body = json.dumps({"elements": [], "teams": [], "events": []})
    client = FplClient(config, raw_dir=tmp_path / "raw", run_id=7)
    session = FakeSession([FakeResponse(500), FakeResponse(200, body)])
    client.session = session
    assert client.get_bootstrap_static()["events"] == []
    assert len(session.calls) == 2
    assert (tmp_path / "raw" / "7" / "bootstrap_static.json").read_text(encoding="utf-8") == body

    client = FplClient(config)
    session = FakeSession([requests.Timeout(), requests.Timeout()])
    client.session = session
    with pytest.raises(FplTimeoutError):
        client.get_fixtures()
    assert len(session.calls) == 2

    client = FplClient(config)
    session = FakeSession([FakeResponse(429, headers={"Retry-After": "3"}), FakeResponse(200, "[]")])
    client.session = session
    client.get_fixtures()
    assert sleeps and sleeps[-1] >= 3


def test_polite_pause_applies_to_every_repeated_endpoint_without_first_request_sleep():
    client = FplClient(_config())
    client.session = FakeSession([FakeResponse(200, "[]"), FakeResponse(200, "{}"), FakeResponse(200, "{}")])
    pauses = []
    client._polite_pause = lambda: pauses.append(True)
    client.get_fixtures()
    client.get_event_live(1)
    client.get_entry(1)
    assert len(pauses) == 3


def test_picks_404_is_benign_no_data_yet():
    client = FplClient(_config())
    client.session = FakeSession([FakeResponse(404)])
    assert client.get_entry_picks(1, 1) is None


def test_transfer_history_returns_exact_rows_and_persists_raw_response(tmp_path):
    config = _config()
    body = json.dumps([{
        "entry": 241392,
        "element_in": 20,
        "element_out": 10,
        "event": 4,
        "time": "2026-09-07T10:00:00Z",
        "element_in_cost": 55,
        "element_out_cost": 50,
    }])
    client = FplClient(config, raw_dir=tmp_path / "raw", run_id=9)
    client.session = FakeSession([FakeResponse(200, body)])
    rows = client.get_entry_transfers(241392)
    assert rows == json.loads(body)
    assert (tmp_path / "raw" / "9" / "entry_241392_transfers.json").read_text(encoding="utf-8") == body


def test_transfer_history_404_is_unavailable():
    client = FplClient(_config())
    client.session = FakeSession([FakeResponse(404)])
    assert client.get_entry_transfers(241392) is None
