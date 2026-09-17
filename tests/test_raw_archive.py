"""Raw archive contract: immutable point-in-time preservation of upstream bytes.

The fetch layer's ``raw_dir/<fetch_run_id>/`` files are an operational
convenience.  The archive is the evidence, and these tests hold it to that:
exact bytes, one observation identity per instant, atomic writes, and a refusal
to overwrite an identity with contradictory content.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from fpl_brain import raw_archive
from fpl_brain.api import FplClient, FplInvalidResponseError

CAPTURE_A = "2026-09-17T01:23:45Z"
CAPTURE_B = "2026-09-17T05:00:00Z"
BODY_ONE = b'{"elements": [{"id": 1}], "teams": [], "events": []}'
BODY_TWO = b'{"elements": [{"id": 1}, {"id": 2}], "teams": [], "events": []}'


class _FakeResponse:
    def __init__(self, body: bytes, status: int = 200) -> None:
        self.content = body
        self.text = body.decode("utf-8")
        self.status_code = status
        self.headers: dict[str, str] = {}


class _FakeSession:
    """Returns queued responses in order, so no network is touched."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.headers: dict[str, str] = {}
        self.requests: list[str] = []

    def get(self, url, timeout=None):
        self.requests.append(url)
        if not self._responses:
            raise AssertionError("no queued response for " + url)
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def close(self):
        pass


def _client(tmp_path, responses, *, observed_at=CAPTURE_A, run_id="7"):
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
    return sorted(path for path in root.rglob("*.json"))


# ---------------------------------------------------------------------------
# A / B / C / G / H
# ---------------------------------------------------------------------------


def test_A_a_successful_capture_is_archived_with_its_identity(tmp_path):
    client = _client(tmp_path, [_FakeResponse(BODY_ONE)])
    client.get_bootstrap_static()

    files = _archived_files(tmp_path)
    assert len(files) == 1
    record = raw_archive.load_manifest(tmp_path / "raw")[0]
    assert record["source"] == "bootstrap_static"
    assert record["observed_at"] == CAPTURE_A
    assert record["byte_count"] == len(BODY_ONE)
    assert record["capture_id"] == f"bootstrap_static@{CAPTURE_A}#{record['payload_sha256'][:12]}"


def test_G_the_recorded_hash_describes_the_archived_bytes(tmp_path):
    client = _client(tmp_path, [_FakeResponse(BODY_ONE)])
    client.get_bootstrap_static()

    record = raw_archive.load_manifest(tmp_path / "raw")[0]
    blob = raw_archive.archive_root(tmp_path / "raw") / record["relative_path"]
    assert blob.read_bytes() == BODY_ONE, "the archived file is not the received bytes"
    assert raw_archive.sha256_of(blob.read_bytes()) == record["payload_sha256"]
    assert raw_archive.verify_archived_blob(tmp_path / "raw", record) is True


def test_H_the_explicit_observation_time_is_preserved_exactly(tmp_path):
    client = _client(tmp_path, [_FakeResponse(BODY_ONE)], observed_at=CAPTURE_B)
    client.get_bootstrap_static()
    record = raw_archive.load_manifest(tmp_path / "raw")[0]
    assert record["observed_at"] == CAPTURE_B
    assert CAPTURE_B.replace(":", "").replace("-", "") in record["relative_path"]


def test_B_a_second_different_payload_does_not_overwrite_the_first(tmp_path):
    client = _client(tmp_path, [_FakeResponse(BODY_ONE), _FakeResponse(BODY_TWO)],
                     observed_at=CAPTURE_A)
    client.get_bootstrap_static()
    client.observed_at = CAPTURE_B
    client.get_bootstrap_static()

    records = raw_archive.load_manifest(tmp_path / "raw")
    assert len(records) == 2
    blobs = {record["relative_path"]: (raw_archive.archive_root(tmp_path / "raw") / record["relative_path"]).read_bytes()
             for record in records}
    assert sorted(blobs.values()) == sorted([BODY_ONE, BODY_TWO]), "an earlier capture was lost"


def test_C_the_latest_convenience_file_follows_the_newest_capture(tmp_path):
    """The legacy raw file is a convenience; it may change, the archive may not."""

    client = _client(tmp_path, [_FakeResponse(BODY_ONE), _FakeResponse(BODY_TWO)])
    client.get_bootstrap_static()
    client.observed_at = CAPTURE_B
    client.get_bootstrap_static()

    latest = tmp_path / "raw" / "7" / "bootstrap_static.json"
    assert latest.read_bytes() == BODY_TWO
    # ... while both observations remain archived.
    assert len(raw_archive.load_manifest(tmp_path / "raw")) == 2


# ---------------------------------------------------------------------------
# D — an identical payload at a later instant is a SECOND observation
# ---------------------------------------------------------------------------


def test_D_identical_bytes_at_a_later_instant_are_two_observations(tmp_path):
    client = _client(tmp_path, [_FakeResponse(BODY_ONE), _FakeResponse(BODY_ONE)])
    client.get_bootstrap_static()
    client.observed_at = CAPTURE_B
    client.get_bootstrap_static()

    records = raw_archive.load_manifest(tmp_path / "raw")
    assert len(records) == 2, "unchanged availability is still point-in-time evidence"
    assert {record["observed_at"] for record in records} == {CAPTURE_A, CAPTURE_B}
    assert {record["payload_sha256"] for record in records} == {raw_archive.sha256_of(BODY_ONE)}
    assert len(_archived_files(tmp_path)) == 2


def test_reoffering_the_same_observation_is_idempotent(tmp_path):
    first = raw_archive.archive_raw_capture(
        tmp_path / "raw", source="bootstrap_static", observed_at=CAPTURE_A, body=BODY_ONE)
    second = raw_archive.archive_raw_capture(
        tmp_path / "raw", source="bootstrap_static", observed_at=CAPTURE_A, body=BODY_ONE)
    assert first.as_dict() == second.as_dict()
    assert len(raw_archive.load_manifest(tmp_path / "raw")) == 1
    assert len(_archived_files(tmp_path)) == 1


# ---------------------------------------------------------------------------
# E — collision fails closed
# ---------------------------------------------------------------------------


def test_E_one_identity_with_different_bytes_fails_closed(tmp_path):
    raw_archive.archive_raw_capture(
        tmp_path / "raw", source="bootstrap_static", observed_at=CAPTURE_A, body=BODY_ONE)
    # Same source and same instant, different content: a contradiction.
    with pytest.raises(raw_archive.RawArchiveCollision):
        raw_archive.archive_raw_capture(
            tmp_path / "raw", source="bootstrap_static", observed_at=CAPTURE_A, body=BODY_TWO)

    records = raw_archive.load_manifest(tmp_path / "raw")
    assert len(records) == 1
    blob = raw_archive.archive_root(tmp_path / "raw") / records[0]["relative_path"]
    assert blob.read_bytes() == BODY_ONE, "the original bytes were overwritten"


# ---------------------------------------------------------------------------
# F — atomic write
# ---------------------------------------------------------------------------


def test_F_a_failed_write_leaves_no_archive_entry(tmp_path, monkeypatch):
    def explode(*_args, **_kwargs):
        raise OSError("simulated crash during rename")

    monkeypatch.setattr(raw_archive.os, "replace", explode)
    with pytest.raises(OSError):
        raw_archive.archive_raw_capture(
            tmp_path / "raw", source="bootstrap_static", observed_at=CAPTURE_A, body=BODY_ONE)

    assert _archived_files(tmp_path) == [], "a partial blob was left behind"
    assert raw_archive.load_manifest(tmp_path / "raw") == [], "a manifest line was left behind"
    leftovers = list(raw_archive.archive_root(tmp_path / "raw").rglob("*.tmp-*"))
    assert leftovers == [], f"temporary files leaked: {leftovers}"


def test_a_failed_manifest_append_does_not_hide_the_blob(tmp_path, monkeypatch):
    """The blob is written before its manifest line, so a crash between the two
    leaves a verifiable orphan rather than a claim with no bytes behind it."""

    def explode(*_args, **_kwargs):
        raise OSError("simulated crash while appending the manifest")

    monkeypatch.setattr(raw_archive, "_append_manifest", explode)
    with pytest.raises(OSError):
        raw_archive.archive_raw_capture(
            tmp_path / "raw", source="bootstrap_static", observed_at=CAPTURE_A, body=BODY_ONE)
    monkeypatch.undo()

    blobs = _archived_files(tmp_path)
    assert len(blobs) == 1 and blobs[0].read_bytes() == BODY_ONE
    assert raw_archive.load_manifest(tmp_path / "raw") == []


# ---------------------------------------------------------------------------
# I — event-scoped identity
# ---------------------------------------------------------------------------


def test_I_event_scoped_sources_do_not_collide_across_events(tmp_path):
    kwargs = {"source": "event_live", "observed_at": CAPTURE_A}
    five = raw_archive.archive_raw_capture(tmp_path / "raw", body=BODY_ONE, event=5, **kwargs)
    six = raw_archive.archive_raw_capture(tmp_path / "raw", body=BODY_TWO, event=6, **kwargs)

    assert five.relative_path != six.relative_path
    assert "event_5" in five.relative_path and "event_6" in six.relative_path
    assert len(_archived_files(tmp_path)) == 2
    assert {record["event"] for record in raw_archive.load_manifest(tmp_path / "raw")} == {5, 6}


# ---------------------------------------------------------------------------
# J — malformed / unsuccessful responses are never admitted
# ---------------------------------------------------------------------------


def test_J_a_non_json_200_is_not_archived(tmp_path):
    client = _client(tmp_path, [_FakeResponse(b"<html>not json</html>")])
    with pytest.raises(FplInvalidResponseError):
        client.get_bootstrap_static()
    assert raw_archive.load_manifest(tmp_path / "raw") == []
    assert _archived_files(tmp_path) == []


def test_J_a_wrongly_shaped_200_is_not_archived(tmp_path):
    client = _client(tmp_path, [_FakeResponse(b'{"teams": []}')])  # missing elements/events
    with pytest.raises(FplInvalidResponseError):
        client.get_bootstrap_static()
    assert raw_archive.load_manifest(tmp_path / "raw") == []


def test_J_an_error_status_is_not_archived(tmp_path):
    from fpl_brain.api import FplHttpError

    client = _client(tmp_path, [_FakeResponse(b'{"elements": []}', status=503)])
    with pytest.raises(FplHttpError):
        client.get_bootstrap_static()
    assert raw_archive.load_manifest(tmp_path / "raw") == []


# ---------------------------------------------------------------------------
# The archive refuses ambiguous input
# ---------------------------------------------------------------------------


def test_an_archive_capture_requires_source_time_and_bytes(tmp_path):
    with pytest.raises(raw_archive.RawArchiveError):
        raw_archive.archive_raw_capture(tmp_path / "raw", source="", observed_at=CAPTURE_A, body=BODY_ONE)
    with pytest.raises(raw_archive.RawArchiveError):
        raw_archive.archive_raw_capture(tmp_path / "raw", source="bootstrap_static", observed_at="", body=BODY_ONE)
    with pytest.raises(raw_archive.RawArchiveError):
        raw_archive.archive_raw_capture(
            tmp_path / "raw", source="bootstrap_static", observed_at=CAPTURE_A, body=BODY_ONE.decode("utf-8"))


def test_the_manifest_is_append_only_and_ordered(tmp_path):
    for index, stamp in enumerate((CAPTURE_B, CAPTURE_A)):
        raw_archive.archive_raw_capture(
            tmp_path / "raw", source="event_status", observed_at=stamp, body=BODY_ONE + str(index).encode())
    records = raw_archive.load_manifest(tmp_path / "raw")
    assert [record["observed_at"] for record in records] == [CAPTURE_A, CAPTURE_B]
    lines = (raw_archive.archive_root(tmp_path / "raw") / raw_archive.MANIFEST_FILENAME).read_text(
        encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert all(json.loads(line)["payload_sha256"] for line in lines)
