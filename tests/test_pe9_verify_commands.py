"""PE-9 — the canonical re-derivation audit commands.

    verify generation <generation_id>
    verify decision <decision_id>

One command pair, so an operator re-derives a certified generation or a production
decision from authoritative persisted evidence instead of trusting a stored label.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import generation_fixtures as gf
import test_production_search_permission as permission_fixtures
from fpl_brain import generation_store as gs, search_permission as sp
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


def _insert_historical_v1_decision(
    conn, *, generation, artifact, artifact_path, packet, request,
    runner_identity, runner_code_identity,
):
    """Persist a pre-permission v1 row using its original identity contract."""

    from fpl_brain.execution_snapshot import file_sha256

    evidence = gs.canonical_identity_value({
        "schema": gs.LEGACY_DECISION_RECORD_SCHEMA,
        "runner_identity": runner_identity,
        "runner_code_identity": runner_code_identity,
        "decision_artifact_file_sha256": file_sha256(artifact_path),
    }, path="evidence")
    manager_packet_sha256 = gs.packet_identity(packet)
    request_sha256 = gs.request_identity(
        planning_event=generation.planning_event,
        horizon_kind=generation.horizon_kind,
        cutoff=generation.cutoff,
        request=request,
    )
    result_sha256 = gs.result_identity_of(artifact)
    identity = {
        "schema": gs.LEGACY_DECISION_RECORD_SCHEMA,
        "generation_id": generation.generation_id,
        "planning_event": generation.planning_event,
        "horizon_kind": generation.horizon_kind,
        "manager_packet_sha256": manager_packet_sha256,
        "request_sha256": request_sha256,
        "result_sha256": result_sha256,
        "runner_identity": runner_identity,
        "evidence": evidence,
        "decision_artifact_ref": str(artifact_path),
    }
    decision_id = "sha256:" + hashlib.sha256(gs._canonical_decision_bytes(identity)).hexdigest()
    conn.execute(
        "INSERT INTO engine_decision_records(decision_id, generation_id, planning_event, "
        "horizon_kind, manager_packet_sha256, request_sha256, result_sha256, runner_identity, "
        "evidence_json, decision_artifact_ref, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (
            decision_id,
            generation.generation_id,
            generation.planning_event,
            generation.horizon_kind,
            manager_packet_sha256,
            request_sha256,
            result_sha256,
            runner_identity,
            json.dumps(evidence, sort_keys=True, separators=(",", ":")),
            str(artifact_path),
            "2026-09-19T11:02:00Z",
        ),
    )
    conn.commit()
    return decision_id


def _make_canonical_v2_decision(tmp_path, monkeypatch):
    """Use real generation/permission paths and stub only expensive execution."""

    path = tmp_path / "permission-source.db"
    conn, generation = permission_fixtures._generation(tmp_path)
    monkeypatch.setattr(
        gs,
        "_decision_executor",
        lambda _profile: permission_fixtures._declared_runner,
    )
    try:
        outcome = gs.make_decision(
            conn,
            permission_fixtures._manager_packet(),
            generation.planning_event,
            horizon_kind=gs.HORIZON_KIND_FOUR_GW,
            generation_id=generation.generation_id,
            profile=gs.DecisionProfile(kind=gs.HORIZON_KIND_FOUR_GW),
        )
    finally:
        conn.close()
    return path, generation, outcome


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


def test_the_verify_decision_command_reports_historical_v1_verified(tmp_path, capsys, monkeypatch):
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
    decision_id = _insert_historical_v1_decision(
        conn,
        generation=generation,
        artifact=artifact,
        artifact_path=artifact_path,
        packet=packet,
        request=request,
        runner_identity="test",
        runner_code_identity=runner_code_identity,
    )
    conn.close()

    # Historical v1 verification must not enter the new production permission gate.
    monkeypatch.setattr(
        sp,
        "require_search_permission",
        lambda *_args, **_kwargs: pytest.fail("v1 verification invoked the v2 permission gate"),
    )
    before = path.read_bytes()
    assert module.main(["--database", str(path), "--json", "decision", decision_id]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["verified"] is True
    assert report["generation_verified"] is True
    assert report["search_permission_evidence_verified"] is None
    assert path.read_bytes() == before

    # An unknown decision id refuses with its own token.
    code = module.main(["--database", str(path), "decision", "sha256:" + "9" * 64])
    assert code == 3
    assert "UNKNOWN_DECISION_ID" in capsys.readouterr().err


def test_the_verify_decision_command_reports_canonical_v2_verified(
    tmp_path, monkeypatch, capsys
):
    module = _runner()
    path, _generation, outcome = _make_canonical_v2_decision(tmp_path, monkeypatch)
    before = path.read_bytes()

    assert module.main([
        "--database", str(path), "--json", "decision", outcome["decision_record_id"]
    ]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["verified"] is True
    assert report["search_permission_evidence_verified"] is True
    assert path.read_bytes() == before


def test_the_verify_decision_command_refuses_v2_without_permission_evidence(
    tmp_path, monkeypatch, capsys
):
    module = _runner()
    path, generation, outcome = _make_canonical_v2_decision(tmp_path, monkeypatch)
    conn = connect_database(path)
    try:
        original_record = gs.load_engine_decision_record(conn, outcome["decision_record_id"])
        original_artifact = json.loads(
            Path(outcome["decision_artifact_ref"]).read_text(encoding="utf-8")
        )
        original_artifact["provenance"].pop("search_permission_evaluation")
        forged_artifact_path = tmp_path / "v2-without-search-permission.json"
        forged_artifact_path.write_text(
            json.dumps(original_artifact, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )

        evidence = json.loads(original_record["evidence_json"])
        evidence.pop("search_permission", None)
        from fpl_brain.execution_snapshot import file_sha256

        evidence["decision_artifact_file_sha256"] = file_sha256(forged_artifact_path)
        assert evidence["schema"] == gs.DECISION_RECORD_SCHEMA
        decision_id = gs.append_engine_decision_record(
            conn,
            generation=generation,
            manager_packet_sha256=str(original_record["manager_packet_sha256"]),
            request_sha256=str(original_record["request_sha256"]),
            result_sha256=gs.result_identity_of(original_artifact),
            runner_identity=str(original_record["runner_identity"]),
            evidence=evidence,
            decision_artifact_ref=str(forged_artifact_path),
        )
        conn.commit()
    finally:
        conn.close()

    before = path.read_bytes()
    assert module.main([
        "--database", str(path), "--json", "decision", decision_id
    ]) == 3
    report = json.loads(capsys.readouterr().out)
    assert report["verified"] is False
    assert report["token"] == gs.DIAG_DECISION_RECORD_INVALID
    assert any("omits required search-permission evidence" in reason for reason in report["reasons"])
    assert path.read_bytes() == before


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
