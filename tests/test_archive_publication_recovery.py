"""DH-02: crash-safe, retryable archive publication.

A persisted blob without a manifest record must be RECOVERABLE on retry; an
idempotent manifest return is only valid after verifying the referenced blob
exists and hashes to the recorded digest; manifest publication must never
expose a partially written record; and a valid blob is never deleted merely
because metadata publication failed.

Each case fails against the pre-repair code for the intended reason:
  A. the retry returned without repairing the missing manifest record;
  B./C. the idempotent return never verified the blob;
  D. the manifest was appended to in place rather than atomically replaced.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fpl_brain import raw_archive

CAPTURE_A = "2026-09-17T01:23:45Z"
CAPTURE_B = "2026-09-17T05:00:00Z"
BODY = b'{"elements": [{"id": 1, "web_name": "P1"}], "ok": true}'


def _publish(tmp_path, *, body=BODY, observed_at=CAPTURE_A, **kwargs):
    return raw_archive.archive_raw_capture(
        tmp_path / "raw", source="bootstrap_static",
        observed_at=observed_at, body=body, **kwargs,
    )


def _blob_for(tmp_path, record) -> Path:
    return raw_archive.archive_root(tmp_path / "raw") / record["relative_path"]


def _manifest_file(tmp_path) -> Path:
    return raw_archive.archive_root(tmp_path / "raw") / raw_archive.MANIFEST_FILENAME


def _tmp_leftovers(tmp_path) -> list[Path]:
    root = raw_archive.archive_root(tmp_path / "raw")
    if not root.exists():
        return []
    return sorted(root.rglob("*.tmp-*"))


def _record_for(body: bytes, observed_at: str) -> dict:
    digest = raw_archive.sha256_of(body)
    return {
        "relative_path": raw_archive.relative_blob_path("bootstrap_static", observed_at, digest, None),
    }


def test_DH02_A_manifest_failure_is_recoverable_on_retry(tmp_path, monkeypatch):
    real_append = raw_archive._append_manifest
    calls: list[str] = []

    def flaky(root, record):
        if not calls:
            calls.append("failed")
            raise OSError("simulated crash before manifest publication")
        return real_append(root, record)

    monkeypatch.setattr(raw_archive, "_append_manifest", flaky)
    with pytest.raises(OSError):
        _publish(tmp_path)
    assert calls == ["failed"]
    # The blob may remain, but there must be no false completed record.
    assert raw_archive.load_manifest(tmp_path / "raw") == []
    orphan = _blob_for(tmp_path, _record_for(BODY, CAPTURE_A))
    assert orphan.read_bytes() == BODY

    # Retry: the existing blob is verified and the missing record published.
    monkeypatch.undo()
    record = _publish(tmp_path)
    records = raw_archive.load_manifest(tmp_path / "raw")
    assert len(records) == 1, "retry left no manifest record: the observation is still hidden"
    assert records[0]["payload_sha256"] == record.payload_sha256
    assert _blob_for(tmp_path, records[0]).read_bytes() == BODY
    assert raw_archive.verify_archived_blob(tmp_path / "raw", records[0]) is True
    assert _tmp_leftovers(tmp_path) == []


def test_DH02_B_idempotent_return_with_missing_blob_fails_closed(tmp_path):
    record = _publish(tmp_path)
    _blob_for(tmp_path, record.as_dict()).unlink()
    with pytest.raises(raw_archive.RawArchiveError):
        _publish(tmp_path)
    records = raw_archive.load_manifest(tmp_path / "raw")
    assert len(records) == 1, "a fail-closed collision changed the manifest"


def test_DH02_C_idempotent_return_with_corrupt_blob_fails_closed(tmp_path):
    record = _publish(tmp_path)
    blob = _blob_for(tmp_path, record.as_dict())
    blob.write_bytes(b'{"tampered": true}')
    with pytest.raises(raw_archive.RawArchiveError):
        _publish(tmp_path)
    records = raw_archive.load_manifest(tmp_path / "raw")
    assert len(records) == 1
    assert records[0]["payload_sha256"] == record.payload_sha256


def test_DH02_D_interrupted_manifest_replace_leaves_manifest_intact(tmp_path, monkeypatch):
    import os as _os

    first = _publish(tmp_path, observed_at=CAPTURE_A)
    before = _manifest_file(tmp_path).read_bytes()
    assert raw_archive.load_manifest(tmp_path / "raw") != []

    real_replace = _os.replace

    def selective_replace(src, dst, *args, **kwargs):
        if Path(dst).name == raw_archive.MANIFEST_FILENAME:
            raise OSError("simulated crash before atomic manifest replace")
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(raw_archive.os, "replace", selective_replace)
    with pytest.raises(OSError):
        _publish(tmp_path, observed_at=CAPTURE_B)
    # The previous manifest is byte-identical and fully parseable: no
    # truncated live JSONL record was exposed.
    assert _manifest_file(tmp_path).read_bytes() == before
    records = raw_archive.load_manifest(tmp_path / "raw")
    assert len(records) == 1
    assert records[0]["payload_sha256"] == first.payload_sha256
    assert _tmp_leftovers(tmp_path) == []

    # The orphan blob from the interrupted attempt is recovered on retry, and
    # the manifest then holds both observations with no duplicates.
    monkeypatch.undo()
    _publish(tmp_path, observed_at=CAPTURE_B)
    records = raw_archive.load_manifest(tmp_path / "raw")
    assert len(records) == 2
    assert len({record["capture_id"] for record in records}) == 2
    for record in records:
        assert raw_archive.verify_archived_blob(tmp_path / "raw", record) is True
    assert json.loads(_manifest_file(tmp_path).read_text(encoding="utf-8").splitlines()[0])["source"]
