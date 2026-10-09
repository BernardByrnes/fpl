from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from scripts import certify_wildcard_horizon as cli
from fpl_brain import execution


class _Conn:
    closed = False

    def close(self):
        self.closed = True


class _Controller:
    def __init__(self, *, start_error=None, finish_error=None):
        self.start_error = start_error
        self.finish_error = finish_error
        self.finished = []

    def create_run(self, **_kwargs):
        return None

    def start(self):
        if self.start_error is not None:
            raise self.start_error

    def acquire_writer_lease(self):
        return None

    def finish(self, status, message=None):
        self.finished.append((status, message))
        if self.finish_error is not None:
            raise self.finish_error


def _args_file(tmp_path, value):
    runs_path = tmp_path / "runs.json"
    runs_path.write_text(json.dumps(value), encoding="utf-8")
    return runs_path


def test_wildcard_certification_rejects_malformed_runs_json_before_connect(tmp_path, monkeypatch):
    connected = []
    monkeypatch.setattr(cli, "connect_database", lambda _path: connected.append(True))
    with pytest.raises(SystemExit):
        cli.main([
            "--db", "unused.sqlite", "--four-gw-generation", "g",
            "--runs-json", str(_args_file(tmp_path, {"5": [1, 2]})), "--events", "5,6,7,8,9,10",
        ])
    assert connected == []


def test_wildcard_certification_marks_early_lifecycle_failure_failed(tmp_path, monkeypatch):
    conn = _Conn()
    controller = _Controller(start_error=RuntimeError("start failed"))
    monkeypatch.setattr(cli, "connect_database", lambda _path: conn)
    monkeypatch.setattr(cli.execution, "ExecutionController", lambda _conn: controller)
    monkeypatch.setattr(cli.gs, "load_generation", lambda *_args: SimpleNamespace(
        generation_id="g", planning_event=5, cutoff="2026-09-01T00:00:00Z",
    ))
    with pytest.raises(RuntimeError, match="start failed"):
        cli.main([
            "--db", "unused.sqlite", "--four-gw-generation", "g",
            "--runs-json", str(_args_file(tmp_path, {"5": {"minutes_v1": 1}})),
            "--events", "5,6,7,8,9,10",
        ])
    assert controller.finished[0][0] == execution.RUN_FAILED
    assert conn.closed is True


def test_wildcard_certification_preserves_primary_error_if_failed_finalization_fails(
    tmp_path, monkeypatch,
):
    conn = _Conn()
    controller = _Controller(
        start_error=RuntimeError("primary start failure"),
        finish_error=OSError("bookkeeping failure"),
    )
    monkeypatch.setattr(cli, "connect_database", lambda _path: conn)
    monkeypatch.setattr(cli.execution, "ExecutionController", lambda _conn: controller)
    monkeypatch.setattr(cli.gs, "load_generation", lambda *_args: SimpleNamespace(
        generation_id="g", planning_event=5, cutoff="2026-09-01T00:00:00Z",
    ))
    with pytest.raises(RuntimeError, match="primary start failure") as caught:
        cli.main([
            "--db", "unused.sqlite", "--four-gw-generation", "g",
            "--runs-json", str(_args_file(tmp_path, {"5": {"minutes_v1": 1}})),
            "--events", "5,6,7,8,9,10",
        ])
    assert any("bookkeeping failure" in note for note in getattr(caught.value, "__notes__", ()))
    assert conn.closed is True
