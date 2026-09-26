"""PE-9 — the canonical re-derivation audit commands.

    verify generation <generation_id>
    verify decision <decision_id>

One command pair, so an operator re-derives a certified generation or a production
decision from authoritative persisted evidence instead of trusting a stored label.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import generation_fixtures as gf
from fpl_brain import generation_store as gs
from fpl_brain.database import connect_database

VERIFY = Path(__file__).resolve().parents[1] / "scripts" / "verify_pe9.py"


def _runner():
    import importlib.util

    spec = importlib.util.spec_from_file_location("verify_pe9", VERIFY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _store(tmp_path):
    """A file-backed authoritative store holding one certified generation."""

    path = tmp_path / "fpl.db"
    conn = connect_database(path)
    world_conn, runs = gf.synthetic_world()
    for table in (
        "positions", "fetch_runs", "teams", "players", "events", "fixtures",
        "projection_runs", "player_fixture_xpts_projections", "monte_carlo_distributions",
    ):
        rows = world_conn.execute(f"SELECT * FROM {table}").fetchall()
        for row in rows:
            columns = list(row.keys())
            conn.execute(
                f"INSERT INTO {table}({','.join(columns)})"
                f" VALUES ({','.join('?' for _ in columns)})",
                tuple(row[column] for column in columns),
            )
    conn.commit()
    world_conn.close()
    generation = gf.certify_world(conn, runs, snapshot_path=tmp_path / "snapshot.db")
    conn.close()
    return path, generation


def test_the_verify_generation_command_reports_verified(tmp_path, capsys):
    module = _runner()
    path, generation = _store(tmp_path)
    assert module.main(["--database", str(path), "generation", generation.generation_id]) == 0
    out = capsys.readouterr().out
    assert "VERIFIED" in out
    assert "manifest_digest_matches: True" in out


def test_the_verify_generation_command_json_mode(tmp_path, capsys):
    module = _runner()
    path, generation = _store(tmp_path)
    assert module.main(
        ["--database", str(path), "--json", "generation", generation.generation_id]
    ) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["verified"] is True
    assert report["generation_id"] == generation.generation_id
    assert report["dependency_closure_reproduced"] is True


def test_the_verify_generation_command_refuses_an_unknown_id(tmp_path, capsys):
    module = _runner()
    path, _generation = _store(tmp_path)
    code = module.main(["--database", str(path), "generation", "sha256:" + "0" * 64])
    assert code == 3
    assert "UNKNOWN_GENERATION_ID" in capsys.readouterr().err


def test_the_verify_generation_command_refuses_a_mutated_manifest(tmp_path, capsys):
    module = _runner()
    path, generation = _store(tmp_path)
    # Edit the persisted bytes underneath the id they were stored under.  The
    # append-only trigger refuses an UPDATE, so the store must be edited as a raw
    # file -- which is exactly the route the digest exists to catch.
    import sqlite3

    raw = sqlite3.connect(str(path))
    try:
        raw.execute("DROP TRIGGER IF EXISTS generation_no_update")
        manifest = json.loads(
            raw.execute(
                "SELECT manifest_json FROM generation WHERE generation_id=?",
                (generation.generation_id,),
            ).fetchone()[0]
        )
        manifest["per_event"]["5"]["runs"]["xpts_v1"] = 99999
        raw.execute(
            "UPDATE generation SET manifest_json=? WHERE generation_id=?",
            (json.dumps(manifest), generation.generation_id),
        )
        raw.commit()
    finally:
        raw.close()
    code = module.main(["--database", str(path), "generation", generation.generation_id])
    assert code == 3
    assert "GENERATION_MANIFEST_MUTATED" in capsys.readouterr().err


def test_the_verify_decision_command_reports_verified(tmp_path, capsys):
    module = _runner()
    path, generation = _store(tmp_path)
    conn = connect_database(path)
    artifact_path = tmp_path / "four_gw_decision.json"
    # The attribution block is what makes the recorded manager/request digests
    # REPRODUCIBLE from the retained artifact; without it the verifier refuses.
    packet = {"entry_id": 1, "planning_event": generation.planning_event}
    request = {"tracing_id": "verify-command"}
    runner_code_identity = "sha256:" + "b" * 64
    manager_context_sha256 = gs._fallback_manager_context_identity(
        packet, planning_event=generation.planning_event, cutoff=generation.cutoff
    )
    artifact = {
        "schema": "fpl_brain.manager_world_decision.v1",
        "planning_event": generation.planning_event,
        "planning_cutoff": generation.cutoff,
        "decision_events": list(generation.events),
        "runner_identity": "test",
        "runner_code_identity": runner_code_identity,
        "manager_context_sha256": manager_context_sha256,
        "attribution": {
            "manager_packet": packet,
            "request": request,
            "manager_packet_sha256": gs.packet_identity(packet),
            "request_sha256": gs.request_identity(
                planning_event=generation.planning_event,
                horizon_kind=gs.HORIZON_KIND_FOUR_GW,
                cutoff=generation.cutoff, request=request,
            ),
            "manager_context_sha256": manager_context_sha256,
        },
        "provenance": {"generation_id": generation.generation_id},
        "decision": {"k": "v"},
        "suppression_reasons": [],
        "decision_confidence": None,
        "fixture_horizon": None,
        "finalist_refinement": {"route_table": {"routes": {}}},
    }
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    from fpl_brain.execution_snapshot import file_sha256

    decision_id = gs.append_engine_decision_record(
        conn, generation=generation,
        manager_packet_sha256=gs.packet_identity(packet),
        request_sha256=gs.request_identity(
            planning_event=generation.planning_event, horizon_kind=gs.HORIZON_KIND_FOUR_GW,
            cutoff=generation.cutoff, request=request,
        ),
        result_sha256=gs.result_identity_of(artifact),
        runner_identity="test", evidence={
            "schema": gs.DECISION_RECORD_SCHEMA,
            "runner_identity": "test",
            "runner_code_identity": runner_code_identity,
            "decision_artifact_file_sha256": file_sha256(artifact_path),
        },
        decision_artifact_ref=str(artifact_path),
    )
    conn.commit()
    conn.close()
    assert module.main(["--database", str(path), "decision", decision_id]) == 0
    out = capsys.readouterr().out
    assert "VERIFIED" in out
    assert "generation_verified: True" in out

    # An unknown decision id refuses with its own token.
    code = module.main(["--database", str(path), "decision", "sha256:" + "9" * 64])
    assert code == 3
    assert "UNKNOWN_DECISION_ID" in capsys.readouterr().err


def test_the_verify_command_refuses_a_missing_store(tmp_path, capsys):
    module = _runner()
    code = module.main(["--database", str(tmp_path / "absent.db"), "generation", "sha256:" + "1" * 64])
    assert code == 3
    assert "does not exist" in capsys.readouterr().err


def test_the_verify_command_reads_the_store_read_only(tmp_path):
    """A verification command must never be able to change the evidence it verifies."""

    module = _runner()
    path, generation = _store(tmp_path)
    before = path.read_bytes()
    module.main(["--database", str(path), "generation", generation.generation_id])
    assert path.read_bytes() == before
    source = VERIFY.read_text(encoding="utf-8")
    assert "connect_readonly_database" in source
    assert "connect_database(" not in source
