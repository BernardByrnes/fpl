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
        ("register-horizon-origin", ["--action", "FH"]),
        ("forecast", ["--action", "BB", "--expiry-event", "8", "--observation-id", "bb-1", "--cache-dir", "cache"]),
        (
            "horizon-forecast",
            ["--action", "WC", "--expiry-event", "8", "--observation-id", "wc-1",
             "--continuation-generation-id", "chip-gen", "--cache-dir", "cache",
             "--value-horizon-length", "6", "--value-generation", "7=value-gen-7"],
        ),
        ("capture-outcome", ["--action", "TC", "--observation-id", "tc-1", "--realization-event", "7", "--fetch-run-id", "12", "--raw-root", "raw"]),
        ("capture-outcome", ["--action", "FH", "--observation-id", "fh-1", "--realization-event", "7", "--fetch-run-id", "12", "--raw-root", "raw"]),
        ("mature", ["--action", "BB", "--observation-id", "bb-1", "--realization-event", "7"]),
        ("mature", ["--action", "WC", "--observation-id", "wc-1", "--realization-event", "7"]),
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


def test_cli_requires_explicit_horizon_value_generations_and_rejects_duplicate_event(
    tmp_path, monkeypatch, capsys,
):
    args = cli._parser().parse_args([
        "horizon-forecast", *_COMMON, "--action", "WC", "--expiry-event", "8",
        "--observation-id", "wc-1", "--continuation-generation-id", "chip-gen",
        "--cache-dir", "cache", "--value-horizon-length", "6",
        "--value-generation", "7=value-gen-7", "--value-generation", "8=value-gen-8",
    ])
    assert dict(args.value_generation) == {7: "value-gen-7", 8: "value-gen-8"}
    connection = _Connection()
    db_path = tmp_path / "existing.sqlite"
    db_path.touch()
    monkeypatch.setattr(cli.database, "connect_readonly_database", lambda _path: connection)
    common = [*_COMMON]
    common[1] = str(db_path)
    common[common.index("--planning-event") + 1] = "4"
    assert cli.main([
        "horizon-forecast", *common, "--action", "WC", "--expiry-event", "6",
        "--observation-id", "wc-1", "--continuation-generation-id", "chip-gen",
        "--cache-dir", "cache", "--value-horizon-length", "6",
        "--value-generation", "5=first", "--value-generation", "5=second",
    ]) == 2
    assert "WC value generation event 5 is repeated" in capsys.readouterr().err
    assert connection.closed is True


@pytest.mark.parametrize("action", ["FH", "WC"])
def test_cli_accepts_horizon_capture_and_maturation_actions(action):
    capture = cli._parser().parse_args([
        "capture-outcome", *_COMMON, "--action", action, "--observation-id", "obs-1",
        "--realization-event", "7", "--fetch-run-id", "12", "--raw-root", "raw",
    ])
    mature = cli._parser().parse_args([
        "mature", *_COMMON, "--action", action, "--observation-id", "obs-1",
        "--realization-event", "7",
    ])
    assert capture.action == mature.action == action


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


def test_horizon_forecast_cli_binds_explicit_products_and_uses_readonly_store(
    tmp_path, monkeypatch, capsys,
):
    from fpl_brain import chip_decision as cd
    from fpl_brain import chip_evidence_ops as evidence_ops
    from fpl_brain import chip_horizon_evidence_ops as horizon_ops
    from fpl_brain import chip_reservation_forecast as crf
    from fpl_brain import chip_route_assembly as cra
    from fpl_brain import generation_store as gs

    connection = _Connection()
    db_path = tmp_path / "db.sqlite"
    db_path.touch()
    identity = {"database_path": str(db_path.resolve()), "origin": "explicit"}
    generation = object()
    route = object()
    coverage = {"product_sha256": "coverage-digest"}
    calls = {}

    monkeypatch.setattr(cli.database, "connect_readonly_database", lambda _path: connection)
    monkeypatch.setattr(evidence_ops, "_required_identity", lambda **_kwargs: identity)
    monkeypatch.setattr(gs, "load_generation", lambda _conn, _generation_id: generation)
    monkeypatch.setattr(cra, "load_verified_normal_route", lambda *_args, **_kwargs: route)
    monkeypatch.setattr(evidence_ops, "_source_identity", lambda _route, _generation: {"source": "bound"})

    def build_coverage(_conn, **kwargs):
        calls["coverage"] = kwargs
        return coverage

    def build_forecast(_conn, **kwargs):
        calls["forecast"] = kwargs
        return {"receipt": {"action": kwargs["action"]}}

    monkeypatch.setattr(crf, "build_reservation_coverage_product", build_coverage)
    monkeypatch.setattr(horizon_ops, "build_future_action_forecast", build_forecast)
    common = [*_COMMON]
    common[1] = str(db_path)
    common[common.index("--planning-event") + 1] = "4"
    assert cli.main([
        "horizon-forecast", *common, "--action", "WC", "--expiry-event", "6",
        "--observation-id", "wc-obs", "--continuation-generation-id", "chip-gen",
        "--cache-dir", "world-cache", "--value-horizon-length", "6",
        "--value-generation", "5=wc5", "--value-generation", "6=wc6",
    ]) == 0
    emitted = json.loads(capsys.readouterr().out)
    assert emitted["coverage_product_sha256"] == "coverage-digest"
    assert calls["coverage"] == {
        "action": cd.CHIP_ACTION_WC,
        "source_identity": {"source": "bound"},
        "expiry_event": 6,
        "product_generation_id": "chip-gen",
        "wildcard_value_horizon_length": 6,
    }
    assert calls["forecast"]["identity"] == identity
    assert calls["forecast"]["value_generation_ids"] == {5: "wc5", 6: "wc6"}
    assert calls["forecast"]["coverage_product"] is coverage
    assert connection.closed is True
    assert connection.executed == []


def test_register_horizon_origin_cli_is_explicit_and_readonly(tmp_path, monkeypatch, capsys):
    from fpl_brain import chip_horizon_evidence_ops as horizon_ops

    connection = _Connection()
    db_path = tmp_path / "db.sqlite"
    db_path.touch()
    captured = {}
    monkeypatch.setattr(cli.database, "connect_readonly_database", lambda _path: connection)
    monkeypatch.setattr(
        horizon_ops, "register_horizon_origin",
        lambda _conn, **kwargs: captured.update(kwargs) or {"registered": True},
    )
    common = [*_COMMON]
    common[1] = str(db_path)
    assert cli.main(["register-horizon-origin", *common, "--action", "WC"]) == 0
    assert json.loads(capsys.readouterr().out) == {"registered": True}
    assert captured["action"] == "WC"
    assert captured["planning_event"] == 6
    assert connection.closed is True
    assert connection.executed == []
