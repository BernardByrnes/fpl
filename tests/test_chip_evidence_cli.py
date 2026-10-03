from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "chip_evidence.py"
_SPEC = importlib.util.spec_from_file_location("chip_evidence_cli", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
cli = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cli)


class _Connection:
    def __init__(self):
        self.closed = False
        self.executed = []
        self.row_factory = None

    def execute(self, sql):
        self.executed.append(sql)

    def close(self):
        self.closed = True


_COMMON = [
    "--db", "db.sqlite", "--entry-id", "77", "--generation-id", "g-1",
    "--decision-id", "d-1", "--route-id", "route-1", "--planning-event", "6",
    "--origin-cutoff", "2026-10-01T10:00:00Z", "--evidence-root", "evidence",
]


@pytest.mark.parametrize(
    ("command", "extra"),
    [
        ("register-origin", []),
        ("forecast", ["--action", "BB", "--expiry-event", "8", "--observation-id", "bb-1", "--cache-dir", "cache"]),
        ("capture-outcome", ["--action", "TC", "--observation-id", "tc-1", "--realization-event", "7", "--fetch-run-id", "12", "--raw-root", "raw"]),
        ("mature", ["--action", "BB", "--observation-id", "bb-1", "--realization-event", "7"]),
    ],
)
def test_cli_has_only_explicitly_bound_workflow_commands(command, extra):
    args = cli._parser().parse_args([command, *_COMMON, *extra])
    assert args.command == command
    for field in ("db", "entry_id", "generation_id", "decision_id", "route_id", "planning_event", "origin_cutoff", "evidence_root"):
        assert getattr(args, field) is not None


def test_cli_refuses_implicit_latest_identity():
    with pytest.raises(SystemExit):
        cli._parser().parse_args(["register-origin", "--db", "db.sqlite"])


@pytest.mark.parametrize(
    ("command", "extra", "handler_name", "connection_mode"),
    [
        ("register-origin", [], "register_origin", "readonly"),
        ("forecast", ["--action", "BB", "--expiry-event", "8", "--observation-id", "bb-1", "--cache-dir", "cache"], "forecast_origin", "readonly"),
        ("capture-outcome", ["--action", "TC", "--observation-id", "tc-1", "--realization-event", "7", "--fetch-run-id", "12", "--raw-root", "raw"], "capture_outcome", "writable"),
        ("mature", ["--action", "BB", "--observation-id", "bb-1", "--realization-event", "7"], "mature_observation", "readonly"),
    ],
)
def test_cli_uses_readonly_database_except_for_append_only_outcome_capture(
    tmp_path, monkeypatch, capsys, command, extra, handler_name, connection_mode,
):
    connection = _Connection()
    calls = []
    db_path = tmp_path / "db.sqlite"
    db_path.touch()
    common = [*_COMMON]
    common[1] = str(db_path)

    def readonly(path):
        calls.append(("readonly", path))
        return connection

    def writable(path, **kwargs):
        calls.append(("writable", path, kwargs))
        return connection

    monkeypatch.setattr(cli.database, "connect_readonly_database", readonly)
    monkeypatch.setattr(cli.sqlite3, "connect", writable)
    monkeypatch.setattr(cli, handler_name, lambda _conn, **kwargs: {"bound": kwargs})

    assert cli.main([command, *common, *extra]) == 0
    emitted = json.loads(capsys.readouterr().out)
    assert emitted["bound"]["generation_id"] == "g-1"
    assert [call[0] for call in calls] == [connection_mode]
    if connection_mode == "writable":
        assert connection.executed == ["PRAGMA busy_timeout=30000"]
        assert calls[0][2]["timeout"] == 30.0
    else:
        assert connection.executed == []
    assert connection.closed is True
