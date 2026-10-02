"""Focused production search-permission regressions over retained PE-9 fixtures."""

from __future__ import annotations

import json
import importlib
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import generation_fixtures as gf
import test_pe9_production_decision as pe9
from fpl_brain import (
    analytics,
    chip_decision as cd,
    chip_free_hit as fh,
    chip_reservation_forecast as crf,
    chip_route_assembly as cra,
    free_hit_production as fhp,
    generation_store as gs,
    history_completeness as hc,
    manager_lineup,
    search_permission as sp,
    wildcard_production as wcp,
)
from fpl_brain import execution_snapshot as es
from fpl_brain import four_gw_decision as fg

_scripts_dir = str(Path(__file__).resolve().parents[1] / "scripts")
_inserted_scripts_dir = _scripts_dir not in sys.path
if _inserted_scripts_dir:
    sys.path.insert(0, _scripts_dir)
try:
    certifier = importlib.import_module("certify_four_gw")
finally:
    if _inserted_scripts_dir:
        sys.path.remove(_scripts_dir)


def _manager_packet() -> dict:
    return {
        "entry_id": 241392,
        "planning_event": 4,
        "cutoff": gf.CUTOFF,
        "season": "2026/27",
    }


def _declared_runner(**kwargs):
    state = kwargs["canonical_manager_state"]
    return {
        "decision": {"status": "SEARCH_PERMISSION_TEST"},
        "consumed_manager_state": state,
        "provenance": {"planning_context_hash": gs.four_gw_manager_context_identity(state)},
        "runner_identity": "search-permission-test-runner",
        "artifact_blocks": {},
    }


def _generation(tmp_path, *, completed_history_placeholder=False):
    path = tmp_path / "permission-source.db"
    return pe9._four_gw_certified_manager_world(
        path, bank=0, free_transfers=0, event_start_free_transfers=0,
        completed_history_placeholder=completed_history_placeholder,
    )


def _incomplete_audit(generation):
    return {
        "schema": hc.HISTORY_COMPLETENESS_SCHEMA,
        "planning_event": generation.planning_event,
        "cutoff": generation.cutoff,
        "required_completed_events": [generation.planning_event - 1],
        "latest_required_completed_event": generation.planning_event - 1,
        "complete": False,
        "blocker": hc.CERTIFIED_PREDICTION_INPUT_HISTORY_INCOMPLETE,
        "reasons": [hc.DIAG_COMPLETED_EVENT_PLACEHOLDER_ROW],
        "detail": "fixture completed event retains a placeholder row",
    }


def _origin_identity(generation) -> dict[str, str | int]:
    return {
        "generation_id": generation.generation_id,
        "planning_event": generation.planning_event,
        "origin_cutoff": generation.cutoff,
        "data_snapshot_sha256": generation.snapshot["sha256"],
    }


def test_certifier_compatibility_entrypoint_delegates_to_shared_predicate():
    conditions = {
        "temporal_status": "UNRESOLVED",
        "dependency_validation": "COHERENT",
        "horizon_status": fg.DECISION_HORIZON_COMPLETE,
        "data_snapshot_sha256": "fixture-snapshot",
        "history_completeness": _incomplete_audit(
            SimpleNamespace(planning_event=5, cutoff=gf.CUTOFF)
        ),
    }
    assert certifier.decide_search_permission(**conditions) == fg.decide_search_permission(**conditions)


def test_predictive_source_identity_covers_shared_permission_implementation_bytes(tmp_path):
    predicate_path = "fpl_brain/four_gw_decision.py"
    certifier_path = "scripts/certify_four_gw.py"
    assert predicate_path in analytics.SOURCE_SNAPSHOT_FILES

    source_root = Path(__file__).resolve().parents[1]
    isolated_predicate = tmp_path / predicate_path
    isolated_certifier = tmp_path / certifier_path
    isolated_predicate.parent.mkdir(parents=True)
    isolated_certifier.parent.mkdir(parents=True)
    isolated_predicate.write_bytes((source_root / predicate_path).read_bytes())
    isolated_certifier.write_bytes((source_root / certifier_path).read_bytes())

    before = analytics.source_snapshot_sha256(root=tmp_path)
    legacy_source_files = tuple(
        relative for relative in analytics.SOURCE_SNAPSHOT_FILES
        if relative != predicate_path
    )
    legacy_before = analytics.source_snapshot_sha256(
        paths=legacy_source_files, root=tmp_path,
    )
    certifier_bytes = isolated_certifier.read_bytes()
    predicate_bytes = isolated_predicate.read_bytes()
    old_rule = b'if str(temporal_status).upper() != "CAUSAL":'
    new_rule = b'if str(temporal_status).strip().upper() != "CAUSAL":'
    assert predicate_bytes.count(old_rule) == 1
    isolated_predicate.write_bytes(predicate_bytes.replace(old_rule, new_rule, 1))
    after = analytics.source_snapshot_sha256(root=tmp_path)
    legacy_after = analytics.source_snapshot_sha256(
        paths=legacy_source_files, root=tmp_path,
    )

    assert isolated_certifier.read_bytes() == certifier_bytes
    assert before != after
    assert legacy_before == legacy_after


def test_complete_causal_history_admits_canonical_decision_and_binds_evaluation(
    tmp_path, monkeypatch
):
    conn, generation = _generation(tmp_path)
    monkeypatch.setattr(gs, "_decision_executor", lambda _profile: _declared_runner)
    try:
        report = gs.verify_generation(conn, generation.generation_id)
        assert report["verified"] is True
        assert report["causal_evidence_reproduced"] is True

        evaluation = sp.require_search_permission(conn, generation.generation_id)
        assert evaluation["permitted"] is True
        assert evaluation["conditions"]["history_completeness"]["complete"] is True
        assert evaluation["origin"]["planning_event"] == generation.planning_event
        assert evaluation["origin"]["cutoff"] == generation.cutoff

        outcome = gs.make_decision(
            conn,
            _manager_packet(),
            generation.planning_event,
            horizon_kind=gs.HORIZON_KIND_FOUR_GW,
            generation_id=generation.generation_id,
            profile=gs.DecisionProfile(kind=gs.HORIZON_KIND_FOUR_GW),
        )
        decision_report = gs.verify_decision(conn, outcome["decision_record_id"])
        assert decision_report["verified"] is True
        assert decision_report["search_permission_evidence_verified"] is True
        retained = json.loads(Path(outcome["decision_artifact_ref"]).read_text(encoding="utf-8"))
        assert retained["provenance"]["search_permission_evaluation"]["evaluation_sha256"] == (
            evaluation["evaluation_sha256"]
        )
    finally:
        conn.close()


def test_new_decision_schema_without_permission_evidence_never_uses_legacy_path(
    tmp_path, monkeypatch
):
    conn, generation = _generation(tmp_path)
    monkeypatch.setattr(gs, "_decision_executor", lambda _profile: _declared_runner)
    try:
        outcome = gs.make_decision(
            conn,
            _manager_packet(),
            generation.planning_event,
            horizon_kind=gs.HORIZON_KIND_FOUR_GW,
            generation_id=generation.generation_id,
            profile=gs.DecisionProfile(kind=gs.HORIZON_KIND_FOUR_GW),
        )
        original = json.loads(Path(outcome["decision_artifact_ref"]).read_text(encoding="utf-8"))
        original["provenance"].pop("search_permission_evaluation")
        forged_artifact = tmp_path / "new-schema-without-permission.json"
        forged_artifact.write_text(
            json.dumps(original, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        historical_evidence = json.loads(
            gs.load_engine_decision_record(conn, outcome["decision_record_id"])["evidence_json"]
        )
        historical_evidence.pop("search_permission", None)
        historical_evidence["decision_artifact_file_sha256"] = es.file_sha256(forged_artifact)
        forged_id = gs.append_engine_decision_record(
            conn,
            generation=generation,
            manager_packet_sha256=gs.packet_identity(_manager_packet()),
            request_sha256=gs.request_identity(
                planning_event=generation.planning_event,
                horizon_kind=gs.HORIZON_KIND_FOUR_GW,
                cutoff=generation.cutoff,
                request={},
            ),
            result_sha256=gs.result_identity_of(original),
            runner_identity=str(original["runner_identity"]),
            evidence=historical_evidence,
            decision_artifact_ref=str(forged_artifact),
        )
        conn.commit()
        with pytest.raises(gs.GenerationRefused) as refused:
            gs.verify_decision(conn, forged_id)
        assert refused.value.token == gs.DIAG_DECISION_RECORD_INVALID
        assert "omits required search-permission evidence" in str(refused.value)
    finally:
        conn.close()


def test_incomplete_history_refuses_before_runner_or_decision_artifact(tmp_path, monkeypatch):
    conn, generation = _generation(tmp_path)
    monkeypatch.setattr(hc, "audit_history_completeness", lambda *_a, **_k: _incomplete_audit(generation))
    calls = []

    def runner_observer(**kwargs):
        calls.append("runner")
        return _declared_runner(**kwargs)

    monkeypatch.setattr(gs, "_decision_executor", lambda _profile: runner_observer)
    before_records = int(conn.execute("SELECT COUNT(*) FROM engine_decision_records").fetchone()[0])
    artifact_dir = gs._decision_artifact_dir(conn, generation.planning_event)
    before_artifacts = set(artifact_dir.glob("*.json")) if artifact_dir.exists() else set()
    try:
        with pytest.raises(gs.GenerationRefused) as refused:
            gs.make_decision(
                conn,
                _manager_packet(),
                generation.planning_event,
                horizon_kind=gs.HORIZON_KIND_FOUR_GW,
                generation_id=generation.generation_id,
                profile=gs.DecisionProfile(kind=gs.HORIZON_KIND_FOUR_GW),
            )
        assert refused.value.token == sp.PRODUCTION_SEARCH_PERMISSION_DENIED
        assert hc.CERTIFIED_PREDICTION_INPUT_HISTORY_INCOMPLETE in str(refused.value)
        assert calls == []
        assert int(conn.execute("SELECT COUNT(*) FROM engine_decision_records").fetchone()[0]) == before_records
        after_artifacts = set(artifact_dir.glob("*.json")) if artifact_dir.exists() else set()
        assert after_artifacts == before_artifacts
    finally:
        conn.close()


def test_warm_cache_and_positive_caller_claim_cannot_bypass_history_refusal(tmp_path, monkeypatch):
    conn, generation = _generation(tmp_path)
    monkeypatch.setattr(hc, "audit_history_completeness", lambda *_a, **_k: _incomplete_audit(generation))
    warm_cache = tmp_path / "preexisting-warm-world-cache.json"
    warm_cache.write_text('{"fixture": "warm cache present"}\n', encoding="utf-8")
    runner_calls = []
    monkeypatch.setattr(
        gs,
        "_decision_executor",
        lambda _profile: lambda **_kwargs: runner_calls.append(str(warm_cache))
        or _declared_runner(**_kwargs),
    )
    try:
        with pytest.raises(sp.SearchPermissionRefused) as refused:
            sp.require_search_permission(conn, generation.generation_id)
        assert hc.CERTIFIED_PREDICTION_INPUT_HISTORY_INCOMPLETE in str(refused.value)
        with pytest.raises(TypeError):
            sp.require_search_permission(conn, generation.generation_id, permission=True)
        with pytest.raises(gs.ProductionDescriptorOnly):
            gs.make_decision(
                conn,
                _manager_packet(),
                generation.planning_event,
                horizon_kind=gs.HORIZON_KIND_FOUR_GW,
                generation_id=generation.generation_id,
                profile=gs.DecisionProfile(kind=gs.HORIZON_KIND_FOUR_GW),
                request={"search_permission": {"permitted": True}},
            )
        assert warm_cache.is_file()
        assert runner_calls == []
        assert conn.execute("SELECT COUNT(*) FROM engine_decision_records").fetchone()[0] == 0
    finally:
        conn.close()


def test_missing_causal_timestamp_evidence_refuses_new_search(tmp_path, monkeypatch):
    original_manifest_writer = gf._write_fixture_snapshot_manifest

    def omit_manifest_binding(snapshot, *, planning_cutoff=gf.CUTOFF):
        record = original_manifest_writer(snapshot, planning_cutoff=planning_cutoff)
        record.pop("manifest_path", None)
        return record

    monkeypatch.setattr(gf, "_write_fixture_snapshot_manifest", omit_manifest_binding)
    conn, generation = _generation(tmp_path)
    monkeypatch.setattr(gs, "_decision_executor", lambda _profile: pytest.fail("uncausal origin reached search"))
    try:
        report = gs.verify_generation(conn, generation.generation_id)
        assert report["verified"] is True
        assert report["causal_evidence_reproduced"] is False
        with pytest.raises(sp.SearchPermissionRefused) as refused:
            sp.require_search_permission(conn, generation.generation_id)
        assert "no retained execution run binds the origin snapshot UUID" in str(refused.value)
        with pytest.raises(gs.GenerationRefused) as decision_refusal:
            gs.make_decision(
                conn,
                _manager_packet(),
                generation.planning_event,
                horizon_kind=gs.HORIZON_KIND_FOUR_GW,
                generation_id=generation.generation_id,
                profile=gs.DecisionProfile(kind=gs.HORIZON_KIND_FOUR_GW),
            )
        assert decision_refusal.value.token == sp.PRODUCTION_SEARCH_PERMISSION_DENIED
    finally:
        conn.close()


def test_snapshot_must_be_fully_captured_before_origin_execution(tmp_path):
    conn, generation = _generation(tmp_path)
    try:
        snapshot = dict(generation.snapshot)
        with pytest.raises(es.CausalityError, match="snapshot capture completion"):
            es.derive_generation_causality(
                snapshot_path=snapshot["path"],
                data_snapshot_sha256=snapshot["sha256"],
                data_snapshot_size_bytes=snapshot["size_bytes"],
                source_db_identity=snapshot["source_db_identity"],
                snapshot_manifest_path=snapshot["manifest_path"],
                execution_run_uuid=snapshot["execution_run_uuid"],
                origin_planning_event=5,
                origin_cutoff=gf.CUTOFF,
                # The fixture's canonical capture-completion instant is cutoff + 1s.
                # Matching cutoff strings alone must not make this causal.
                execution_started_at=gf.CUTOFF,
            )
    finally:
        conn.close()


def test_direct_evaluated_opportunity_without_authoritative_store_is_denied(tmp_path):
    conn, generation = _generation(tmp_path)
    conn.close()
    with pytest.raises(sp.SearchPermissionRefused) as refused:
        crf.build_evaluated_event_opportunity_record(
            action=cd.CHIP_ACTION_BB,
            event=generation.planning_event + 1,
            source_identity=_origin_identity(generation),
            reservation_state={},
            made_at="2026-09-19T11:00:03Z",
        )
    assert refused.value.token == sp.PRODUCTION_SEARCH_PERMISSION_DENIED
    assert "requires the authoritative generation store" in str(refused.value)


@pytest.mark.parametrize("action", ["fh_request", "wc_request"])
def test_chip_request_factories_gate_before_snapshot_or_cache_work(tmp_path, monkeypatch, action):
    conn, generation = _generation(tmp_path)
    monkeypatch.setattr(hc, "audit_history_completeness", lambda *_a, **_k: _incomplete_audit(generation))
    route = SimpleNamespace(generation_id=generation.generation_id)
    monkeypatch.setattr(fhp, "load_verified_normal_route", lambda *_a, **_k: route)
    monkeypatch.setattr(wcp, "load_verified_normal_route", lambda *_a, **_k: route)
    original_open = gs._open_generation_snapshot
    snapshot_opens = []

    def tracked_snapshot_open(value):
        snapshot_opens.append(value.generation_id)
        return original_open(value)

    monkeypatch.setattr(
        gs,
        "_open_generation_snapshot",
        tracked_snapshot_open,
    )
    monkeypatch.setattr(
        "fpl_brain.free_hit_production._snapshot_cache_dir",
        lambda *_a, **_k: pytest.fail("Free Hit cache consulted before the origin permission gate"),
    )
    monkeypatch.setattr(
        "fpl_brain.wildcard_production._snapshot_cache_dir",
        lambda *_a, **_k: pytest.fail("Wildcard cache consulted before the origin permission gate"),
        raising=False,
    )

    if action == "wc_request":
        value_events = tuple(range(generation.planning_event, generation.planning_event + 6))
        value_generation = SimpleNamespace(
            generation_id="fixture-wildcard-value-generation",
            planning_event=generation.planning_event,
            horizon_kind=gs.HORIZON_KIND_WILDCARD_VALUE,
            events=value_events,
            cutoff=generation.cutoff,
            snapshot=generation.snapshot,
            manifest=generation.manifest,
            runs_for=generation.runs_for,
        )
        monkeypatch.setattr(
            gs,
            "load_generation",
            lambda _conn, identity: generation if identity == generation.generation_id else value_generation,
        )
        monkeypatch.setattr(
            gs,
            "verify_generation",
            lambda _conn, identity: {
                "verified": True,
                "causal_evidence_reproduced": True,
                "dependency_closure_reproduced": True,
                "bundle_identities_reproduced": True,
                "horizon_state": "DECISION_HORIZON_COMPLETE",
                "snapshot_identity": "VERIFIED",
            },
        )

    try:
        with pytest.raises(sp.SearchPermissionRefused) as refused:
            if action == "fh_request":
                fhp.build_free_hit_production_request(
                    conn,
                    decision_id="fixture-decision",
                    entry_id=241392,
                    certification_path="unused.json",
                    rules=object(),
                )
            else:
                wcp.build_wildcard_production_request(
                    conn,
                    decision_id="fixture-decision",
                    value_generation_id="fixture-wildcard-value-generation",
                    entry_id=241392,
                    length=6,
                    rules=object(),
                )
        assert refused.value.token == sp.PRODUCTION_SEARCH_PERMISSION_DENIED
        assert hc.CERTIFIED_PREDICTION_INPUT_HISTORY_INCOMPLETE in str(refused.value)
        assert snapshot_opens == [generation.generation_id]
    finally:
        conn.close()


@pytest.mark.parametrize("producer", ["bb_tc", "evaluated", "fh", "wc"])
def test_direct_future_chip_producers_share_origin_gate_before_expensive_work(
    tmp_path, monkeypatch, producer
):
    conn, generation = _generation(tmp_path)
    audited_origins = []

    def audit_origin(_conn, *, planning_event, cutoff):
        audited_origins.append((planning_event, cutoff))
        return _incomplete_audit(generation)

    monkeypatch.setattr(hc, "audit_history_completeness", audit_origin)
    source = _origin_identity(generation)
    try:
        if producer == "bb_tc":
            route = SimpleNamespace(
                generation_id=generation.generation_id,
                planning_event=generation.planning_event,
                cutoff=generation.cutoff,
                data_snapshot_sha256=generation.snapshot["sha256"],
                events=generation.events,
            )
            monkeypatch.setattr(
                cra,
                "load_certified_event_chip_worlds",
                lambda *_a, **_k: pytest.fail("world cache was consumed before the permission gate"),
            )
            call = lambda: cra.build_future_event_chip_opportunity(
                conn,
                route,
                action=cd.CHIP_ACTION_BB,
                event=generation.planning_event + 1,
                reservation_state={},
                made_at="2026-09-19T11:00:03Z",
            )
        elif producer == "evaluated":
            call = lambda: crf.build_evaluated_event_opportunity_record(
                action=cd.CHIP_ACTION_BB,
                event=generation.planning_event + 1,
                source_identity=source,
                reservation_state={},
                made_at="2026-09-19T11:00:03Z",
                conn=conn,
            )
        elif producer == "fh":
            request = SimpleNamespace(planning_event=generation.planning_event + 1)
            call = lambda: fhp.build_future_free_hit_event_opportunity(
                conn,
                request=request,
                source_identity=source,
                event=generation.planning_event + 1,
                expiry_event=generation.planning_event + 1,
                coverage_product={},
                reservation_state={},
                made_at="2026-09-19T11:00:03Z",
            )
        else:
            request = SimpleNamespace(planning_event=generation.planning_event + 1)
            call = lambda: wcp.build_future_wildcard_event_opportunity(
                conn,
                request=request,
                source_identity=source,
                event=generation.planning_event + 1,
                expiry_event=generation.planning_event + 1,
                chip_generation_id="unused-before-gate",
                value_generation_id="unused-before-gate",
                coverage_product={},
                reservation_state={},
                made_at="2026-09-19T11:00:03Z",
            )

        with pytest.raises(Exception) as refused:
            call()
        assert refused.value.token == sp.PRODUCTION_SEARCH_PERMISSION_DENIED
        assert sp.PRODUCTION_SEARCH_PERMISSION_DENIED in str(refused.value)
        assert hc.CERTIFIED_PREDICTION_INPUT_HISTORY_INCOMPLETE in str(refused.value)
        assert audited_origins and all(
            origin == (generation.planning_event, generation.cutoff)
            for origin in audited_origins
        )
        assert audited_origins and all(
            origin == (generation.planning_event, generation.cutoff)
            for origin in audited_origins
        )
    finally:
        conn.close()


def test_structural_inspection_rejects_malformed_permission_claims_without_production_authority():
    source = {
        "source_decision_id": "decision-1",
        "source_result_sha256": "a" * 64,
        "source_artifact_sha256": "b" * 64,
        "generation_id": "sha256:" + "c" * 64,
        "planning_event": 5,
        "origin_cutoff": "2026-09-19T11:00:00Z",
        "data_snapshot_sha256": "d" * 64,
        "predictive_code_snapshot_sha256": "e" * 64,
        "certification_identity": "f" * 64,
    }
    record = crf.build_event_opportunity_record(
        action=cd.CHIP_ACTION_BB,
        planning_event=5,
        event=6,
        origin_cutoff="2026-09-19T11:00:00Z",
        made_at="2026-09-19T11:00:03Z",
        expected_incremental_points=0.0,
        opportunity_model="fixture",
        source_identity=source,
        reservation_state={"squad_ids": list(range(1, 16))},
        world_identity="fixture-world",
    )
    fake_eval = {
        "schema": sp.SEARCH_PERMISSION_EVALUATION_SCHEMA,
        "permitted": True,
        "reasons": [],
        "origin": {
            "generation_id": source["generation_id"],
            "generation_manifest_sha256": source["generation_id"],
            "planning_event": 5,
            "cutoff": source["origin_cutoff"],
            "data_snapshot_sha256": source["data_snapshot_sha256"],
            "predictive_code_snapshot_sha256": source["predictive_code_snapshot_sha256"],
        },
        "conditions": {
            "temporal_status": "CAUSAL",
            "dependency_validation": "COHERENT",
            "history_completeness": {"complete": True},
        },
    }
    fake_eval["evaluation_sha256"] = sp.evaluation_digest(fake_eval)
    retained = crf._attach_search_permission_evaluation(record, fake_eval)

    def rehash(value):
        value.pop("artifact_sha256", None)
        value["artifact_sha256"] = crf.canonical_sha256(value)
        return value

    tampered_records = []
    stale_digest = json.loads(json.dumps(retained))
    stale_digest["search_permission_evaluation"]["permitted"] = False
    tampered_records.append(rehash(stale_digest))

    substituted = json.loads(json.dumps(retained))
    substituted["search_permission_evaluation"]["conditions"]["history_completeness"]["complete"] = False
    substituted["search_permission_evaluation"]["evaluation_sha256"] = sp.evaluation_digest(
        substituted["search_permission_evaluation"]
    )
    tampered_records.append(rehash(substituted))

    wrong_origin = json.loads(json.dumps(retained))
    wrong_origin["search_permission_evaluation"]["origin"]["planning_event"] = 4
    wrong_origin["search_permission_evaluation"]["evaluation_sha256"] = sp.evaluation_digest(
        wrong_origin["search_permission_evaluation"]
    )
    tampered_records.append(rehash(wrong_origin))

    removed_block = json.loads(json.dumps(retained))
    removed_block.pop("search_permission_evaluation")
    tampered_records.append(rehash(removed_block))

    for tampered in tampered_records:
        with pytest.raises(crf.ReservationForecastError, match="search-permission evidence"):
            crf.inspect_event_opportunity_structure(
                tampered,
                action=cd.CHIP_ACTION_BB,
                planning_event=5,
                origin_cutoff="2026-09-19T11:00:00Z",
                source_identity=source,
                reservation_state=None,
            )


def test_structural_forecast_detects_nested_opportunity_byte_substitution():
    source = {
        "source_decision_id": "decision-1",
        "source_result_sha256": "a" * 64,
        "source_artifact_sha256": "b" * 64,
        "generation_id": "sha256:" + "c" * 64,
        "planning_event": 5,
        "origin_cutoff": "2026-09-19T11:00:00Z",
        "data_snapshot_sha256": "d" * 64,
        "predictive_code_snapshot_sha256": "e" * 64,
        "certification_identity": "f" * 64,
    }
    state = {"squad_ids": list(range(1, 16))}
    raw = crf.build_event_opportunity_record(
        action=cd.CHIP_ACTION_BB,
        planning_event=5,
        event=6,
        origin_cutoff=source["origin_cutoff"],
        made_at="2026-09-19T11:00:03Z",
        expected_incremental_points=0.0,
        opportunity_model="fixture",
        source_identity=source,
        reservation_state=state,
        world_identity="fixture-world",
    )
    evaluation = {
        "schema": sp.SEARCH_PERMISSION_EVALUATION_SCHEMA,
        "permitted": True,
        "reasons": [],
        "origin": {
            "generation_id": source["generation_id"],
            "generation_manifest_sha256": source["generation_id"],
            "planning_event": 5,
            "cutoff": source["origin_cutoff"],
            "horizon_kind": "FOUR_GW",
            "data_snapshot_sha256": source["data_snapshot_sha256"],
            "predictive_code_snapshot_sha256": source["predictive_code_snapshot_sha256"],
        },
        "conditions": {
            "temporal_status": "CAUSAL",
            "dependency_validation": "COHERENT",
            "history_completeness": {
                "complete": True,
                "planning_event": 5,
                "cutoff": source["origin_cutoff"],
            },
        },
    }
    evaluation["evaluation_sha256"] = sp.evaluation_digest(evaluation)
    retained = crf._attach_search_permission_evaluation(raw, evaluation)

    forecast = crf.build_structural_reservation_forecast(
        action=cd.CHIP_ACTION_BB,
        planning_event=5,
        origin_cutoff=source["origin_cutoff"],
        made_at="2026-09-19T11:00:04Z",
        expiry_event=6,
        source_identity=source,
        reservation_state=state,
        opportunity_refs=["event-6.json"],
        evidence_verifier=lambda _ref: retained,
    )
    stripped = dict(retained)
    stripped.pop("search_permission_evidence_schema")
    stripped.pop("search_permission_evaluation")
    stripped.pop("artifact_sha256")
    stripped["artifact_sha256"] = crf.canonical_sha256(stripped)

    with pytest.raises(crf.ReservationForecastError, match="differs from the forecast manifest"):
        crf.inspect_reservation_forecast_structure(
            forecast,
            expected={
                "action": cd.CHIP_ACTION_BB,
                "planning_event": 5,
                "origin_cutoff": source["origin_cutoff"],
                "source_identity": source,
                "reservation_state": state,
            },
            evidence_verifier=lambda _ref: stripped,
        )


def _permission_source_identity(conn, generation, monkeypatch, *, create_decision=True):
    per_event = generation.manifest.get("per_event") or {}
    bundle_identities = {
        str(event): str((per_event.get(str(event)) or {}).get("bundle_identity") or "")
        for event in generation.events
    }
    certification_identity = gs._certification_identity_for_bundles(
        cutoff=generation.cutoff,
        bundle_identities=bundle_identities,
        snapshot_sha256=generation.snapshot["sha256"],
    )
    decision_fields = {
        "source_decision_id": "fixture-decision-not-used-for-authority",
        "source_result_sha256": "sha256:" + "a" * 64,
        "source_artifact_sha256": "sha256:" + "b" * 64,
    }
    if create_decision:
        monkeypatch.setattr(gs, "_decision_executor", lambda _profile: _declared_runner)
        outcome = gs.make_decision(
            conn,
            _manager_packet(),
            generation.planning_event,
            horizon_kind=gs.HORIZON_KIND_FOUR_GW,
            generation_id=generation.generation_id,
            profile=gs.DecisionProfile(kind=gs.HORIZON_KIND_FOUR_GW),
        )
        decision_fields = {
            "source_decision_id": outcome["decision_record_id"],
            "source_result_sha256": outcome["result_sha256"],
            "source_artifact_sha256": es.file_sha256(outcome["decision_artifact_ref"]),
        }
    return {
        **decision_fields,
        "generation_id": generation.generation_id,
        "planning_event": generation.planning_event,
        "origin_cutoff": generation.cutoff,
        "data_snapshot_sha256": generation.snapshot["sha256"],
        "predictive_code_snapshot_sha256": generation.manifest["code_snapshot_sha256"],
        "certification_identity": certification_identity,
    }


def _real_evaluated_opportunity(conn, generation, source):
    positions = {
        1: "GKP", 2: "GKP", 3: "DEF", 4: "DEF", 5: "DEF",
        6: "DEF", 7: "DEF", 8: "MID", 9: "MID", 10: "MID",
        11: "MID", 12: "MID", 13: "FWD", 14: "FWD", 15: "FWD",
    }
    policy = manager_lineup.ManagerPolicy(
        starter_ids=(1, 3, 4, 5, 8, 9, 10, 11, 13, 14, 15),
        bench_gk_id=2,
        bench_outfield_order=(6, 12, 7),
        captain_id=8,
        vice_captain_id=9,
    )
    future_event = generation.planning_event + 1
    world_identity = crf.canonical_sha256({"certified event": future_event})
    worlds = cd.ChipWorldInputs(
        worlds=4,
        player_ids=tuple(range(1, 16)),
        minutes={pid: (90.0, 90.0, 0.0, 90.0) for pid in range(1, 16)},
        core={pid: (float(pid), float(pid + 1), 0.0, float(pid + 2)) for pid in range(1, 16)},
        planning_event=generation.planning_event,
        horizon_events=cd.canonical_chip_horizon(generation.planning_event),
        certification_identity=source["certification_identity"],
        data_snapshot_sha256=source["data_snapshot_sha256"],
        world_seed=104,
        world_identity=world_identity,
        code_snapshot_sha256=source["predictive_code_snapshot_sha256"],
        source_event=future_event,
    )
    binding = cd.ChipHorizonBinding(
        planning_event=generation.planning_event,
        horizon_events=cd.canonical_chip_horizon(generation.planning_event),
        certification_identity=source["certification_identity"],
        data_snapshot_sha256=source["data_snapshot_sha256"],
    )
    return crf.build_evaluated_event_opportunity_record(
        action=cd.CHIP_ACTION_TC,
        event=future_event,
        worlds=worlds,
        horizon_binding=binding,
        policy=policy,
        positions=positions,
        source_identity=source,
        reservation_state={"squad_ids": list(range(1, 16))},
        made_at="2026-09-19T11:00:05Z",
        input_as_of=generation.cutoff,
        conn=conn,
    )


def _rehash_opportunity(record):
    record.pop("artifact_sha256", None)
    record["artifact_sha256"] = crf.canonical_sha256(record)
    return record


def _minimal_opportunity(source, evaluation):
    record = crf.build_event_opportunity_record(
        action=cd.CHIP_ACTION_BB,
        planning_event=int(source["planning_event"]),
        event=int(source["planning_event"]) + 1,
        origin_cutoff=str(source["origin_cutoff"]),
        made_at="2026-09-19T11:00:05Z",
        expected_incremental_points=0.0,
        opportunity_model="permission-regression-fixture",
        source_identity=source,
        reservation_state={"squad_ids": list(range(1, 16))},
        world_identity="permission-regression-world",
    )
    return crf._attach_search_permission_evaluation(record, evaluation)


def test_real_certified_origin_permission_survives_forecast_retention_and_reverification(
    tmp_path, monkeypatch,
):
    conn, generation = _generation(tmp_path)
    try:
        source = _permission_source_identity(conn, generation, monkeypatch)
        opportunity = _real_evaluated_opportunity(conn, generation, source)
        evaluation = sp.require_search_permission(conn, generation.generation_id)
        assert opportunity["search_permission_evaluation"] == evaluation
        assert crf._verify_event_opportunity(
            opportunity,
            store_conn=conn,
            action=cd.CHIP_ACTION_TC,
            planning_event=generation.planning_event,
            origin_cutoff=generation.cutoff,
            source_identity=source,
            reservation_state=None,
        )["production_permission_verified"] is True

        root = tmp_path / "opportunity-evidence"
        opportunity_receipt = crf.retain_event_opportunity_record(
            opportunity, root, store_conn=conn,
        )
        retained = {Path(opportunity_receipt["path"]).name: opportunity}
        artifact = crf.build_reservation_forecast(
            store_conn=conn,
            action=cd.CHIP_ACTION_TC,
            planning_event=generation.planning_event,
            origin_cutoff=generation.cutoff,
            made_at="2026-09-19T11:00:30Z",
            expiry_event=generation.planning_event + 1,
            source_identity=source,
            reservation_state={"squad_ids": list(range(1, 16))},
            opportunity_refs=tuple(retained),
            evidence_verifier=retained.__getitem__,
        )
        assert artifact["coverage_complete"] is False
        assert artifact["raw_value"] is None
        report = crf.verify_reservation_forecast(
            artifact, expected={}, store_conn=conn,
        )
        assert report["production_permission_verified"] is True
        receipt = crf.retain_reservation_forecast(artifact, root, store_conn=conn)
        assert receipt["artifact_sha256"] == artifact["artifact_sha256"]
        again = crf.verify_reservation_forecast(
            json.loads(Path(receipt["path"]).read_text(encoding="utf-8")),
            expected={}, store_conn=conn,
        )
        assert again["production_permission_verified"] is True
    finally:
        conn.close()


def test_self_declared_positive_permission_cannot_authorize_unknown_generation():
    source = {
        "source_decision_id": "decision-1",
        "source_result_sha256": "a" * 64,
        "source_artifact_sha256": "b" * 64,
        "generation_id": "sha256:" + "c" * 64,
        "planning_event": 5,
        "origin_cutoff": "2026-09-19T11:00:00Z",
        "data_snapshot_sha256": "d" * 64,
        "predictive_code_snapshot_sha256": "e" * 64,
        "certification_identity": "f" * 64,
    }
    history = {
        "schema": hc.HISTORY_COMPLETENESS_SCHEMA,
        "planning_event": 5,
        "cutoff": source["origin_cutoff"],
        "complete": True,
        "required_completed_events": [],
        "reasons": [],
    }
    body = {
        "schema": sp.SEARCH_PERMISSION_EVALUATION_SCHEMA,
        "permitted": True,
        "reasons": [],
        "origin": {
            "generation_id": source["generation_id"],
            "generation_manifest_sha256": source["generation_id"],
            "planning_event": 5,
            "cutoff": source["origin_cutoff"],
            "horizon_kind": "FOUR_GW",
            "events": [5, 6, 7, 8],
            "data_snapshot_sha256": source["data_snapshot_sha256"],
            "predictive_code_snapshot_sha256": source["predictive_code_snapshot_sha256"],
        },
        "conditions": {
            "temporal_status": "CAUSAL",
            "causal_evidence_sha256": "sha256:" + "1" * 64,
            "dependency_validation": "COHERENT",
            "horizon_status": fg.DECISION_HORIZON_COMPLETE,
            "data_snapshot_sha256": source["data_snapshot_sha256"],
            "snapshot_identity": "VERIFIED",
            "snapshot_error": None,
            "history_completeness": history,
        },
    }
    evaluation = {**body, "evaluation_sha256": sp.evaluation_digest(body)}
    record = _minimal_opportunity(source, evaluation)
    conn = sqlite3.connect(":memory:")
    try:
        with pytest.raises(crf.ReservationForecastError, match="did not reproduce"):
            crf._verify_event_opportunity(
                record,
                store_conn=conn,
                action=cd.CHIP_ACTION_BB,
                planning_event=5,
                origin_cutoff=source["origin_cutoff"],
                source_identity=source,
                reservation_state=None,
            )
    finally:
        conn.close()


def test_rehashed_opportunity_permission_substitutions_are_rederived_and_refused(
    tmp_path, monkeypatch,
):
    conn, generation = _generation(tmp_path)
    try:
        source = _permission_source_identity(conn, generation, monkeypatch)
        authoritative = sp.require_search_permission(conn, generation.generation_id)

        cases = []
        unknown = json.loads(json.dumps(authoritative))
        unknown_id = "sha256:" + "9" * 64
        unknown["origin"]["generation_id"] = unknown_id
        unknown["origin"]["generation_manifest_sha256"] = unknown_id
        unknown["evaluation_sha256"] = sp.evaluation_digest(unknown)
        unknown_source = {**source, "generation_id": unknown_id}
        cases.append((unknown_source, unknown, "could not be independently verified"))

        wrong_event = json.loads(json.dumps(authoritative))
        wrong_event["origin"]["planning_event"] = generation.planning_event - 1
        wrong_event["conditions"]["history_completeness"]["planning_event"] = generation.planning_event - 1
        wrong_event["evaluation_sha256"] = sp.evaluation_digest(wrong_event)
        wrong_event_source = {**source, "planning_event": generation.planning_event - 1}
        cases.append((wrong_event_source, wrong_event, "differs from asserted source event"))

        wrong_snapshot = json.loads(json.dumps(authoritative))
        wrong_snapshot_sha = "sha256:" + "8" * 64
        wrong_snapshot["origin"]["data_snapshot_sha256"] = wrong_snapshot_sha
        wrong_snapshot["conditions"]["data_snapshot_sha256"] = wrong_snapshot_sha
        wrong_snapshot["evaluation_sha256"] = sp.evaluation_digest(wrong_snapshot)
        wrong_snapshot_source = {**source, "data_snapshot_sha256": wrong_snapshot_sha}
        cases.append((wrong_snapshot_source, wrong_snapshot, "snapshot digest differs"))

        wrong_horizon = json.loads(json.dumps(authoritative))
        wrong_horizon["conditions"]["horizon_status"] = fg.DECISION_HORIZON_INCOMPLETE
        wrong_horizon["evaluation_sha256"] = sp.evaluation_digest(wrong_horizon)
        cases.append((source, wrong_horizon, "differs from the authoritative derivation"))

        wrong_cause = json.loads(json.dumps(authoritative))
        wrong_cause["conditions"]["causal_evidence_sha256"] = "sha256:" + "7" * 64
        wrong_cause["evaluation_sha256"] = sp.evaluation_digest(wrong_cause)
        cases.append((source, wrong_cause, "differs from the authoritative derivation"))

        altered_history = json.loads(json.dumps(authoritative))
        altered_history["conditions"]["history_completeness"]["detail"] = "caller says complete"
        altered_history["evaluation_sha256"] = sp.evaluation_digest(altered_history)
        cases.append((source, altered_history, "differs from the authoritative derivation"))

        for index, (case_source, case_evaluation, message) in enumerate(cases):
            forged = _minimal_opportunity(case_source, case_evaluation)
            with pytest.raises(crf.ReservationForecastError, match=message):
                crf._verify_event_opportunity(
                    forged,
                    store_conn=conn,
                    action=cd.CHIP_ACTION_BB,
                    planning_event=int(case_source["planning_event"]),
                    origin_cutoff=str(case_source["origin_cutoff"]),
                    source_identity=case_source,
                    reservation_state=None,
                )
            with pytest.raises(crf.ReservationForecastError):
                crf.build_reservation_forecast(
                    store_conn=conn,
                    action=cd.CHIP_ACTION_BB,
                    planning_event=int(case_source["planning_event"]),
                    origin_cutoff=str(case_source["origin_cutoff"]),
                    made_at="2026-09-19T11:00:30Z",
                    expiry_event=int(case_source["planning_event"]) + 1,
                    source_identity=case_source,
                    reservation_state={"squad_ids": list(range(1, 16))},
                    opportunity_refs=(f"forged-{index}",),
                    evidence_verifier=lambda _reference, value=forged: value,
                )
            with pytest.raises(crf.ReservationForecastError):
                crf.retain_event_opportunity_record(
                    forged, tmp_path / f"rejected-{index}", store_conn=conn,
                )

        stripped = _real_evaluated_opportunity(conn, generation, source)
        stripped.pop("search_permission_evidence_schema")
        stripped.pop("search_permission_evaluation")
        _rehash_opportunity(stripped)
        with pytest.raises(crf.ReservationForecastError, match="missing its required"):
            crf._verify_event_opportunity(
                stripped,
                store_conn=conn,
                action=cd.CHIP_ACTION_TC,
                planning_event=generation.planning_event,
                origin_cutoff=generation.cutoff,
                source_identity=source,
                reservation_state=None,
            )
        with pytest.raises(crf.ReservationForecastError, match="missing its required"):
            crf.retain_event_opportunity_record(
                stripped,
                tmp_path / "stripped-permission",
                store_conn=conn,
            )
        with pytest.raises(crf.ReservationForecastError):
            crf.build_reservation_forecast(
                store_conn=conn,
                action=cd.CHIP_ACTION_TC,
                planning_event=generation.planning_event,
                origin_cutoff=generation.cutoff,
                made_at="2026-09-19T11:00:30Z",
                expiry_event=generation.planning_event + 1,
                source_identity=source,
                reservation_state={"squad_ids": list(range(1, 16))},
                opportunity_refs=("stripped",),
                evidence_verifier=lambda _reference: stripped,
            )
    finally:
        conn.close()


def test_incomplete_pinned_history_overrides_a_rehashed_positive_claim(tmp_path, monkeypatch):
    conn, generation = _generation(tmp_path, completed_history_placeholder=True)
    try:
        verified = gs.verify_generation(conn, generation.generation_id)
        assert verified["verified"] is True
        audit_conn = gs._open_generation_snapshot(generation)
        try:
            audit = hc.audit_history_completeness(
                audit_conn,
                planning_event=generation.planning_event,
                cutoff=generation.cutoff,
            )
        finally:
            audit_conn.close()
        assert audit["complete"] is False
        assert hc.DIAG_COMPLETED_EVENT_PLACEHOLDER_ROW in audit["reasons"]

        source = _permission_source_identity(conn, generation, monkeypatch, create_decision=False)
        claimed = sp.require_search_permission
        with pytest.raises(sp.SearchPermissionRefused, match=hc.CERTIFIED_PREDICTION_INPUT_HISTORY_INCOMPLETE):
            claimed(conn, generation.generation_id)
        # Reuse the actual generation/source identifiers but claim a complete audit.
        body = {
            "schema": sp.SEARCH_PERMISSION_EVALUATION_SCHEMA,
            "permitted": True,
            "reasons": [],
            "origin": {
                "generation_id": generation.generation_id,
                "generation_manifest_sha256": generation.generation_id,
                "planning_event": generation.planning_event,
                "cutoff": generation.cutoff,
                "horizon_kind": gs.HORIZON_KIND_FOUR_GW,
                "events": list(generation.events),
                "data_snapshot_sha256": generation.snapshot["sha256"],
                "predictive_code_snapshot_sha256": generation.manifest["code_snapshot_sha256"],
            },
            "conditions": {
                "temporal_status": "CAUSAL",
                "causal_evidence_sha256": "sha256:" + "1" * 64,
                "dependency_validation": "COHERENT",
                "horizon_status": fg.DECISION_HORIZON_COMPLETE,
                "data_snapshot_sha256": generation.snapshot["sha256"],
                "snapshot_identity": "VERIFIED",
                "snapshot_error": None,
                "history_completeness": {
                    "schema": hc.HISTORY_COMPLETENESS_SCHEMA,
                    "planning_event": generation.planning_event,
                    "cutoff": generation.cutoff,
                    "complete": True,
                    "required_completed_events": [generation.planning_event - 1],
                    "reasons": [],
                },
            },
        }
        claimed_evaluation = {**body, "evaluation_sha256": sp.evaluation_digest(body)}
        forged = _minimal_opportunity(source, claimed_evaluation)
        with pytest.raises(crf.ReservationForecastError, match="did not reproduce"):
            crf._verify_event_opportunity(
                forged,
                store_conn=conn,
                action=cd.CHIP_ACTION_BB,
                planning_event=generation.planning_event,
                origin_cutoff=generation.cutoff,
                source_identity=source,
                reservation_state=None,
            )
        with pytest.raises(crf.ReservationForecastError):
            crf.build_reservation_forecast(
                store_conn=conn,
                action=cd.CHIP_ACTION_BB,
                planning_event=generation.planning_event,
                origin_cutoff=generation.cutoff,
                made_at="2026-09-19T11:00:30Z",
                expiry_event=generation.planning_event + 1,
                source_identity=source,
                reservation_state={"squad_ids": list(range(1, 16))},
                opportunity_refs=("claimed-complete-history",),
                evidence_verifier=lambda _reference: forged,
            )
    finally:
        conn.close()
