"""DH-01: the immutable archive admits only fully validated captures.

ARCHIVE ADMISSION occurs only AFTER the endpoint's full existing validation
contract has passed.  The senior counterexample -- HTTP 200, syntactically
valid JSON, top-level bootstrap keys present, but invalid under
``validate_bootstrap_payload()`` -- was archived and only later rejected by
ingest, so the archive could certify an observation ingest rejects.

These tests fail against the pre-repair code (which archived after only a
shallow top-level shape check) and pass once the endpoint's own full
validator gates archive publication.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fpl_brain import raw_archive
from fpl_brain.api import FplClient
from fpl_brain.parsers import BootstrapValidationError

CAPTURE = "2026-09-17T01:23:45Z"

# The senior counterexample verbatim: valid JSON, top-level keys present, but
# empty collections that validate_bootstrap_payload() rejects.
INVALID_EMPTY = b'{"elements": [], "teams": [], "events": []}'

# Keys present and non-empty, but an element without the required web_name.
INVALID_MALFORMED_ELEMENTS = (
    b'{"elements": [{"id": 1}], '
    b'"teams": [{"id": 1, "name": "Arsenal"}], "events": []}'
)

VALID = (
    b'{"elements": [{"id": 1, "web_name": "P1"}, {"id": 2, "web_name": "P2"}], '
    b'"teams": [{"id": 1, "name": "Arsenal"}], "events": []}'
)


class _FakeResponse:
    def __init__(self, body: bytes, status: int = 200) -> None:
        self.content = body
        self.text = body.decode("utf-8")
        self.status_code = status
        self.headers: dict[str, str] = {}


class _FakeSession:
    def __init__(self, responses):
        self._responses = list(responses)
        self.headers: dict[str, str] = {}

    def get(self, url, timeout=None):
        if not self._responses:
            raise AssertionError("no queued response for " + url)
        return self._responses.pop(0)

    def close(self):
        pass


def _client(tmp_path, responses, *, observed_at=CAPTURE, run_id="7"):
    client = FplClient(
        {"request": {"polite_delay_seconds": 0.0, "max_retries": 0}},
        raw_dir=tmp_path / "raw",
        run_id=run_id,
        observed_at=observed_at,
    )
    client.session = _FakeSession(responses)
    return client


def _archived_files(tmp_path) -> list[Path]:
    root = raw_archive.archive_root(tmp_path / "raw")
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*.json"))


def test_DH01_empty_but_well_shaped_bootstrap_is_rejected_before_archive(tmp_path):
    client = _client(tmp_path, [_FakeResponse(INVALID_EMPTY)])
    with pytest.raises(BootstrapValidationError):
        client.get_bootstrap_static()
    assert raw_archive.load_manifest(tmp_path / "raw") == [], "a rejected payload was admitted to the manifest"
    assert _archived_files(tmp_path) == [], "a rejected payload was admitted as a blob"


def test_DH01_malformed_elements_are_rejected_before_archive(tmp_path):
    client = _client(tmp_path, [_FakeResponse(INVALID_MALFORMED_ELEMENTS)])
    with pytest.raises(BootstrapValidationError):
        client.get_bootstrap_static()
    assert raw_archive.load_manifest(tmp_path / "raw") == []
    assert _archived_files(tmp_path) == []


def test_DH01_valid_bootstrap_archives_exactly_one_observation(tmp_path):
    client = _client(tmp_path, [_FakeResponse(VALID)])
    payload = client.get_bootstrap_static()
    assert payload["elements"] and payload["teams"] is not None

    records = raw_archive.load_manifest(tmp_path / "raw")
    assert len(records) == 1, "a valid bootstrap must yield exactly one accepted archive observation"
    record = records[0]
    assert record["source"] == "bootstrap_static"
    assert record["observed_at"] == CAPTURE
    blob = raw_archive.archive_root(tmp_path / "raw") / record["relative_path"]
    assert blob.read_bytes() == VALID, "the archive must hash and persist the ORIGINAL response.content"
    assert raw_archive.sha256_of(VALID) == record["payload_sha256"]
