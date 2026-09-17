"""DH-03: archive identity is established under one process-safe lock.

Two overlapping publishers that both observe "identity absent" must not be
able to publish contradictory bytes for one (source, observed_at, event)
identity.  Same identity + same payload stays idempotent; same identity +
different payload lets exactly one publisher win while the other fails closed.

These are REAL cross-process regressions: each attempt runs in an independent
interpreter (``sys.executable`` + a worker script), synchronized through
ready/go files so both publishers contend.  A threads-only proof would not
exercise the cross-process ownership the contract requires.  The conflicting
case runs several rounds so the race is discriminating.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from fpl_brain import raw_archive

ROOT = Path(__file__).resolve().parents[1]
ROUNDS = 5

_WORKER = """\
import json
import sys
import time
from pathlib import Path

raw_dir, source, observed_at, body_hex, event_s, ready_path, go_path, result_path = sys.argv[1:9]
body = bytes.fromhex(body_hex)
event = None if event_s == "none" else int(event_s)
Path(ready_path).write_text("ready", encoding="utf-8")
deadline = time.monotonic() + 60.0
while not Path(go_path).exists():
    if time.monotonic() > deadline:
        Path(result_path).write_text(json.dumps({"status": "error", "type": "TimeoutError"}))
        raise SystemExit(1)
    time.sleep(0.002)
from fpl_brain import raw_archive
try:
    record = raw_archive.archive_raw_capture(
        raw_dir, source=source, observed_at=observed_at, body=body, event=event)
    Path(result_path).write_text(json.dumps({"status": "ok", "sha": record.payload_sha256}))
except Exception as exc:
    Path(result_path).write_text(json.dumps(
        {"status": "error", "type": type(exc).__name__, "message": str(exc)[:500]}))
"""


def _run_race(tmp_path, *, observed_at, body_a, body_b, source="bootstrap_static", event=None):
    workdir = tmp_path / "workers"
    workdir.mkdir(parents=True, exist_ok=True)
    worker = workdir / "race_worker.py"
    if not worker.exists():
        worker.write_text(_WORKER, encoding="utf-8")
    tag = observed_at.replace(":", "").replace("-", "")
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    procs = []
    result_paths = []
    for side, body in (("a", body_a), ("b", body_b)):
        ready = workdir / f"{tag}_{side}.ready"
        result = workdir / f"{tag}_{side}.result.json"
        result_paths.append(result)
        procs.append(subprocess.Popen(
            [sys.executable, str(worker), str(tmp_path / "raw"), source, observed_at,
             body.hex(), "none" if event is None else str(event),
             str(ready), str(workdir / f"{tag}.go"), str(result)],
            cwd=str(ROOT), env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ))
    try:
        go = workdir / f"{tag}.go"
        if go.exists():
            go.unlink()
        deadline = time.monotonic() + 60.0
        while True:
            ready_files = [workdir / f"{tag}_{side}.ready" for side in ("a", "b")]
            if all(path.exists() for path in ready_files):
                break
            if time.monotonic() > deadline:
                raise TimeoutError("race workers never became ready")
            time.sleep(0.005)
        go.write_text("go", encoding="utf-8")
        for proc in procs:
            proc.wait(timeout=120)
            assert proc.returncode == 0, f"race worker crashed with {proc.returncode}"
        results = []
        for path in result_paths:
            deadline = time.monotonic() + 60.0
            while not path.exists():
                if time.monotonic() > deadline:
                    raise TimeoutError(f"no result at {path}")
                time.sleep(0.005)
            results.append(json.loads(path.read_text(encoding="utf-8")))
        return results
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.kill()


def _identity_records(tmp_path, *, observed_at, source="bootstrap_static", event=None):
    return [
        item for item in raw_archive.load_manifest(tmp_path / "raw")
        if str(item.get("source")) == source
        and str(item.get("observed_at")) == observed_at
        and (None if item.get("event") is None else int(item["event"])) == event
    ]


def test_DH03_different_payloads_same_identity_exactly_one_winner(tmp_path):
    for round_index in range(ROUNDS):
        observed_at = f"2026-09-17T01:23:{45 + round_index:02d}Z"
        body_a = json.dumps({"round": round_index, "side": "a", "players": [1, 2, 3]}).encode()
        body_b = json.dumps({"round": round_index, "side": "b", "players": [4, 5, 6]}).encode()
        results = _run_race(tmp_path, observed_at=observed_at, body_a=body_a, body_b=body_b)

        kinds = sorted(result["status"] for result in results)
        assert kinds == ["error", "ok"], f"round {round_index}: conflicting identity must fail one side closed, got {results}"
        loser = next(result for result in results if result["status"] == "error")
        winner = next(result for result in results if result["status"] == "ok")
        assert loser["type"] == "RawArchiveCollision", f"round {round_index}: loser must fail closed, got {loser}"

        records = _identity_records(tmp_path, observed_at=observed_at)
        assert len(records) == 1, f"round {round_index}: conflicting identity left {len(records)} records"
        assert records[0]["payload_sha256"] == winner["sha"], "the manifest must name the winning bytes"
        assert raw_archive.verify_archived_blob(tmp_path / "raw", records[0]) is True


def test_DH03_same_payload_concurrent_publication_is_idempotent(tmp_path):
    observed_at = "2026-09-17T09:00:00Z"
    body = json.dumps({"round": "shared", "players": [1, 2, 3]}).encode()
    results = _run_race(tmp_path, observed_at=observed_at, body_a=body, body_b=body)

    assert [result["status"] for result in results] == ["ok", "ok"]
    assert results[0]["sha"] == results[1]["sha"] == raw_archive.sha256_of(body)
    records = _identity_records(tmp_path, observed_at=observed_at)
    assert len(records) == 1, "concurrent identical publication must not duplicate the identity record"
    assert raw_archive.verify_archived_blob(tmp_path / "raw", records[0]) is True
