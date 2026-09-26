"""PE-9 section 16 — the ``build_manager_packet`` CLI and its published provenance.

The repair kept the descriptive Phase-6A packet useful while moving the predictive
load onto the certified generation.  These are CLI/OUTPUT regressions in the literal
sense: the command is invoked with real arguments, its exit code is asserted, and the
files it writes are read back and checked.

The generation is certified through the REAL ``generation_store`` lifecycle over a
synthetic world, so what the CLI resolves is what production would have persisted.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from fpl_brain.database import connect_database

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_manager_packet.py"


def _module():
    spec = importlib.util.spec_from_file_location("build_manager_packet_cli", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _world_store(tmp_path):
    """A file-backed store holding one certified MANAGER_WORLD generation.

    The packet's world load REGENERATES the football worlds from the certified run
    rows, so the rows must contain real player x fixture evidence rather than just
    run headers.
    """

    import generation_fixtures as gf

    path = tmp_path / "fpl.db"
    conn = connect_database(path)
    world_conn, runs = gf.synthetic_world(events=(5,))
    for table in (
        "positions", "fetch_runs", "teams", "players", "events", "fixtures",
        "projection_runs", "player_fixture_xpts_projections", "monte_carlo_distributions",
    ):
        for row in world_conn.execute(f"SELECT * FROM {table}").fetchall():
            columns = list(row.keys())
            conn.execute(
                f"INSERT INTO {table}({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",
                tuple(row[column] for column in columns),
            )
    conn.commit()
    world_conn.close()
    generation = gf.certify_world(
        conn, runs, events=(5,), horizon_kind=__import__(
            "fpl_brain.generation_store", fromlist=["x"]
        ).HORIZON_KIND_MANAGER_WORLD,
        snapshot_path=tmp_path / "snapshot.db",
    )
    conn.close()
    return path, generation


def test_the_cli_requires_a_planning_event():
    """There is no GW4 default: omitting the event is refused before any work."""

    module = _module()
    with pytest.raises(SystemExit):
        module.main(["--out-dir", "unused"])


def test_the_generation_selector_is_resolved_and_a_mismatched_run_id_refuses(tmp_path):
    module = _module()
    path, generation = _world_store(tmp_path)
    conn = connect_database(path)
    try:
        module.__dict__["connect_database"] = lambda *_a, **_k: conn
        runs = module.certified_run_ids(generation, event=5)
        assert runs["minutes_v1"] == generation.runs_for(5)["minutes_v1"]
        # A supplied run id that is not the certified one is refused rather than
        # preferred, defaulted or silently ignored.
        with pytest.raises(module.CertificationMismatch):
            module.certified_run_ids(generation, event=5, xpts_run=999_999)
        # A family the generation does not record cannot be claimed either.
        with pytest.raises(module.CertificationMismatch):
            module.certified_run_ids(generation, event=5, monte_carlo_run=1)
        versions = module.certified_model_versions(generation, event=5)
        assert versions["xpts_v1"] == generation.model_versions_by_event[5]["xpts_v1"]
    finally:
        conn.close()


def _run_cli(module, path, tmp_path, capsys, *extra):
    """Invoke the REAL CLI against the fixture store and return (code, stdout, stderr)."""

    monkeypatch_database = connect_database(path)
    module.__dict__["connect_database"] = lambda *_a, **_k: monkeypatch_database
    module.__dict__["config_path"] = lambda config, key: (
        str(tmp_path / "out") if key == "exports_dir" else str(path)
    )
    module.__dict__["load_config"] = lambda _path=None: {"fpl_entry_id": 1, "season": "2026/27"}
    try:
        code = module.main([
            "--gw", "5", "--out-dir", str(tmp_path / "out"), "--simulations", "4", *extra,
        ])
    finally:
        pass
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_the_cli_publishes_the_generation_provenance_on_stdout(tmp_path, capsys):
    """The command RESOLVES the generation and prints what it consumed.

    The synthetic world has no 15-player squad, so the descriptive packet itself
    stops at the squad check -- but the generation it resolved, re-proved and pinned
    is printed FIRST, which is exactly the provenance an operator needs to see and
    the part that must never regress.
    """

    module = _module()
    path, generation = _world_store(tmp_path)
    code, out, err = _run_cli(module, path, tmp_path, capsys, "--generation", generation.generation_id)
    assert code == 3, (out, err)  # the fixture squad is not a real 15
    assert f"generation={generation.generation_id[:24]}" in out
    assert "generation provenance:" in out
    assert generation.generation_id in out
    assert "PlanningContext squad has" in err

    provenance = module.certified_snapshot_provenance(generation)
    assert provenance["generation_id"] == generation.generation_id
    assert provenance["snapshot_sha256"] == generation.snapshot["sha256"]
    assert provenance["pe8_evidence"]["consulted"] is False
    assert provenance["disclosure"]["resolution"] == "NOT_ATTEMPTED"


def test_the_cli_refuses_an_unknown_generation_selector(tmp_path, capsys):
    module = _module()
    path, _generation = _world_store(tmp_path)
    code, _out, err = _run_cli(module, path, tmp_path, capsys, "--generation", "sha256:" + "0" * 64)
    assert code == 3
    assert "UNKNOWN_GENERATION_ID" in err


def test_the_cli_refuses_a_run_id_that_is_not_the_certified_one(tmp_path, capsys):
    module = _module()
    path, generation = _world_store(tmp_path)
    code, _out, err = _run_cli(
        module, path, tmp_path, capsys, "--generation", generation.generation_id,
        "--xpts-run", "999999",
    )
    assert code == 3
    assert "not the run the certified generation records" in err
