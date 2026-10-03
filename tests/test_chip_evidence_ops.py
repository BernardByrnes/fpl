from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from fpl_brain import chip_evidence_ops as ops
from fpl_brain import database, outcome_ledger as ol, raw_archive, route_optimizer as ro


def _identity(tmp_path: Path, db_path: Path) -> dict:
    db_path.touch(exist_ok=True)
    return {
        "db_path": db_path,
        "entry_id": 77,
        "generation_id": "generation-fixture",
        "decision_id": "decision-fixture",
        "route_id": "route-fixture",
        "planning_event": 1,
        "cutoff": "2026-10-01T10:00:00Z",
        "evidence_root": tmp_path / "evidence",
    }


def _raw_capture_set(tmp_path: Path, *, finished=True, data_checked=True,
                     fixture_finished=True, players=(1,), run_id=None,
                     observed_base="2026-10-01T10:10:00Z", run_started_at=None,
                     run_finished_at=None, conflicting_archive_time=False):
    db_path = tmp_path / "fpl.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    database.initialize_database(conn)
    conn.execute(
        "INSERT INTO events(id,name,finished,data_checked,raw_json,updated_at) VALUES(2,'GW2',?,?,?,?)",
        (int(finished), int(data_checked), "{}", observed_base),
    )
    conn.execute(
        "INSERT INTO fixtures(id,event,started,finished,finished_provisional,raw_json,updated_at) "
        "VALUES(200,2,?,?,1,'{}',?)",
        (int(fixture_finished), int(fixture_finished), observed_base),
    )
    raw_root = tmp_path / "raw"
    import datetime as dt

    base_time = ops._parse_utc(observed_base, name="fixture observation time")
    started_at = run_started_at or (base_time + dt.timedelta(seconds=1)).isoformat().replace("+00:00", "Z")
    finished_at = run_finished_at or (base_time + dt.timedelta(seconds=3)).isoformat().replace("+00:00", "Z")
    if run_id is None:
        cursor = conn.execute(
            "INSERT INTO fetch_runs(started_at,finished_at,status,trigger,current_event,endpoints_ok,"
            "endpoints_failed,raw_dir) VALUES(?,?, 'success','test',2,?,?,?)",
            (started_at, finished_at, json.dumps([
                "bootstrap-static", "fixtures", "event/2/live",
            ]), "[]", str(raw_root.resolve())),
        )
        run_id = int(cursor.lastrowid)
    else:
        conn.execute(
            "INSERT INTO fetch_runs(id,started_at,finished_at,status,trigger,current_event,endpoints_ok,"
            "endpoints_failed,raw_dir) VALUES(?,?,?,'success','test',2,?,'[]',?)",
            (int(run_id), started_at, finished_at, json.dumps([
                "bootstrap-static", "fixtures", "event/2/live",
            ]), str(raw_root.resolve())),
        )
    conn.commit()
    event = {
        "id": 2, "name": "GW2", "finished": bool(finished),
        "data_checked": bool(data_checked),
    }
    bootstrap = {
        "events": [event],
        "teams": [{"id": 1, "name": "Team"}],
        "elements": [
            {"id": player, "web_name": f"P{player}", "team": 1, "element_type": 3}
            for player in sorted(set(players) | {2})
        ],
    }
    fixture_payload = [{
        "id": 200, "event": 2, "started": bool(fixture_finished),
        "finished": bool(fixture_finished), "finished_provisional": True,
        "team_h": 1, "team_a": 2,
    }]
    live = {"elements": [
        {"id": player, "stats": {"minutes": 0, "total_points": 0}}
        for player in players
    ]}
    conflicting_time = (base_time + dt.timedelta(seconds=1)).isoformat().replace("+00:00", "Z")
    observed_times = (
        observed_base,
        conflicting_time if conflicting_archive_time else observed_base,
        observed_base,
    )
    for source, observed_at, value, event_id in (
        ("bootstrap_static", observed_times[0], bootstrap, None),
        ("fixtures", observed_times[1], fixture_payload, 2),
        ("event_live_2", observed_times[2], live, 2),
    ):
        raw_archive.archive_raw_capture(
            raw_root,
            source=source,
            observed_at=observed_at,
            body=json.dumps(value, separators=(",", ":")).encode("utf-8"),
            event=event_id,
            run_id=run_id,
        )
    return conn, db_path, raw_root, int(run_id)


def _forecast_parent(event=2, required=(1, 2)):
    schedule = [{"event": event, "proposed_squad_ids": list(required)}]
    causal = {
        "forecast": {"selected_event": event},
        "counterfactual_pair": {
            role: {"valuation_schedule": {"events": schedule}}
            for role in ("play", "save")
        },
    }
    return {
        "origin_id": "origin-fixture",
        "receipt": {"selected_event": event},
        "causal": causal,
    }


def test_origin_manifest_and_receipt_bind_exact_explicit_source_identity(tmp_path):
    identity = {
        "entry_id": 77, "generation_id": "g", "decision_id": "d", "route_id": "r",
        "planning_event": 6, "cutoff": "2026-10-01T10:00:00Z",
        "database_path": str(tmp_path / "fpl.db"), "evidence_root": str(tmp_path / "evidence"),
    }
    origin_id = ops._sha256(ops._origin_key(identity))
    origin = {
        "schema": ops.ORIGIN_SCHEMA,
        "origin_id": origin_id,
        "registered_at": "2026-10-01T10:01:00Z",
        "identity": identity,
    }
    receipt = ops._origin_receipt(origin_id, "chip-origin-fixture.json", origin)
    ops._verify_origin_binding(receipt, origin, origin_id=origin_id, expected_identity=identity)

    substituted = {**identity, "route_id": "another-route"}
    forged_origin = {**origin, "identity": substituted}
    forged_receipt = ops._origin_receipt(origin_id, "chip-origin-fixture.json", forged_origin)
    with pytest.raises(ops.ChipEvidenceError, match="origin id does not reproduce"):
        ops._verify_origin_binding(
            forged_receipt, forged_origin, origin_id=origin_id, expected_identity=identity,
        )


def test_register_origin_recovers_manifest_published_before_receipt(tmp_path, monkeypatch):
    db_path = tmp_path / "fpl.db"
    db_path.touch()
    evidence_root = tmp_path / "evidence"
    kwargs = {
        "db_path": db_path, "entry_id": 77, "generation_id": "g",
        "decision_id": "d", "route_id": "r", "planning_event": 6,
        "cutoff": "2026-10-01T10:00:00Z", "evidence_root": evidence_root,
    }
    identity = ops._required_identity(**kwargs)
    origin_id = ops._sha256(ops._origin_key(identity))
    receipt_path = ops._metadata_root(evidence_root) / f"origin-{origin_id}.json"
    monkeypatch.setattr(ops, "_origin_verification", lambda *_a, **_k: (object(), {}, object(), {}, {}))
    monkeypatch.setattr(
        ops,
        "_origin_manifest",
        lambda *, identity, origin_id, registered_at, **_kwargs: {
            "schema": ops.ORIGIN_SCHEMA,
            "origin_id": origin_id,
            "registered_at": registered_at,
            "identity": dict(identity),
        },
    )
    monkeypatch.setattr(ops, "_verify_origin_manifest_authority", lambda *_a, **_k: None)
    original_write = ops._write_new_json
    failed = False

    def fail_receipt_once(path, value):
        nonlocal failed
        if Path(path) == receipt_path and not failed:
            failed = True
            raise OSError("injected origin-receipt publication failure")
        return original_write(path, value)

    monkeypatch.setattr(ops, "_write_new_json", fail_receipt_once)
    conn = sqlite3.connect(":memory:")
    try:
        with pytest.raises(OSError, match="injected origin-receipt"):
            ops.register_origin(conn, **kwargs)
        manifests = list(evidence_root.glob("chip-origin-*.json"))
        assert len(manifests) == 1
        retained_registered_at = ops._read_json(manifests[0])["registered_at"]

        monkeypatch.setattr(ops, "_write_new_json", original_write)
        recovered = ops.register_origin(conn, **kwargs)
        assert recovered["origin"]["registered_at"] == retained_registered_at
        assert recovered["receipt"]["origin_id"] == origin_id
        assert len(list(evidence_root.glob("chip-origin-*.json"))) == 1
        assert receipt_path.is_file()
    finally:
        conn.close()


def test_forecast_intent_without_completion_is_not_replayed_with_old_time(tmp_path, monkeypatch):
    db_path = tmp_path / "fpl.db"
    db_path.touch()
    evidence_root = tmp_path / "evidence"
    identity = {
        "database_path": str(db_path.resolve()), "entry_id": 77,
        "generation_id": "g", "decision_id": "d", "route_id": "r",
        "planning_event": 1, "cutoff": "2026-10-01T10:00:00Z",
        "evidence_root": str(evidence_root.resolve()),
    }
    origin_id = ops._sha256(ops._origin_key(identity))
    operation_id = ops._forecast_operation_id(origin_id, "BB", "bb-obs-1")
    intent_path = ops._metadata_root(evidence_root) / f"forecast-intent-{operation_id}.json"
    intent = ops._sealed(ops.FORECAST_INTENT_SCHEMA, {
        "operation_id": operation_id,
        "identity_sha256": ops._sha256(identity),
        "origin_id": origin_id,
        "observation_id": "bb-obs-1",
        "action": "BB",
        "expiry_event": 4,
        "continuation_generation_id": None,
        "cache_dir": str((ops._metadata_root(evidence_root) / "world-cache" / origin_id).resolve()),
        "materialize_worlds": False,
        "calculation_started_at": "2026-10-01T10:01:00Z",
    })
    ops._write_new_json(intent_path, intent)

    class FakeRoute:
        events = (1, 2, 3, 4)
        planning_event = 1

    retained = {"origin": {"chip_eligibility": {"BB": {"eligible": True, "expiry_event": 4}}}}
    monkeypatch.setattr(ops, "_required_identity", lambda **_kwargs: identity)
    monkeypatch.setattr(
        ops, "_revalidate_registered_origin",
        lambda *_args, **_kwargs: (retained, object(), {}, FakeRoute(), {}, {}),
    )
    monkeypatch.setattr(ops, "_source_identity", lambda *_args: {})
    from fpl_brain import planning
    from fpl_brain import chip_route_assembly as cra
    monkeypatch.setattr(planning, "event_data_state", lambda *_args: (planning.EVENT_STATE_SCHEDULED, []))
    monkeypatch.setattr(ops, "_world_cache_requirements", lambda *_args, **_kwargs: pytest.fail("must not re-score"))
    monkeypatch.setattr(cra, "build_bb_tc_reservation_forecast", lambda *_args, **_kwargs: pytest.fail("must not re-score"))

    conn = sqlite3.connect(":memory:")
    with pytest.raises(ops.ChipEvidenceError, match="only an intent"):
        ops.forecast_origin(
            conn,
            db_path=db_path,
            entry_id=77,
            generation_id="g",
            decision_id="d",
            route_id="r",
            planning_event=1,
            cutoff="2026-10-01T10:00:00Z",
            action="BB",
            expiry_event=4,
            observation_id="bb-obs-1",
            continuation_generation_id=None,
            evidence_root=evidence_root,
            cache_dir=Path(identity["evidence_root"]) / ".chip-evidence" / "world-cache" / origin_id,
        )
    conn.close()


def test_forecast_requires_same_origin_coverage_before_cache_or_scoring(tmp_path, monkeypatch):
    db_path = tmp_path / "fpl.db"
    db_path.touch()
    evidence_root = tmp_path / "evidence"
    identity = {
        "database_path": str(db_path.resolve()), "entry_id": 77,
        "generation_id": "g", "decision_id": "d", "route_id": "r",
        "planning_event": 1, "cutoff": "2026-10-01T10:00:00Z",
        "evidence_root": str(evidence_root.resolve()),
    }

    class FakeRoute:
        events = (1, 2, 3, 4)
        planning_event = 1

    retained = {"origin": {"chip_eligibility": {"BB": {"eligible": True, "expiry_event": 4}}}}
    monkeypatch.setattr(ops, "_required_identity", lambda **_kwargs: identity)
    monkeypatch.setattr(
        ops, "_revalidate_registered_origin",
        lambda *_args, **_kwargs: (retained, object(), {}, FakeRoute(), {}, {}),
    )
    monkeypatch.setattr(ops, "_source_identity", lambda *_args: {})
    from fpl_brain import planning
    from fpl_brain import chip_route_assembly as cra

    monkeypatch.setattr(planning, "event_data_state", lambda *_args: (planning.EVENT_STATE_SCHEDULED, []))
    monkeypatch.setattr(ops, "_world_cache_requirements", lambda *_a, **_k: pytest.fail("must refuse before cache work"))
    monkeypatch.setattr(cra, "build_bb_tc_reservation_forecast", lambda *_a, **_k: pytest.fail("must not score"))
    cache_dir = evidence_root / ".chip-evidence" / "world-cache" / "origin-fixture"
    conn = sqlite3.connect(":memory:")
    try:
        with pytest.raises(ops.ChipEvidenceError, match="complete future-chip coverage requires"):
            ops.forecast_origin(
                conn,
                db_path=db_path, entry_id=77, generation_id="g", decision_id="d", route_id="r",
                planning_event=1, cutoff="2026-10-01T10:00:00Z", action="BB", expiry_event=4,
                observation_id="bb-no-coverage", continuation_generation_id=None,
                evidence_root=evidence_root, cache_dir=cache_dir,
            )
        assert not list(evidence_root.glob("chip-reservation-forecast-*.json"))
        assert not list((evidence_root / ".chip-evidence").glob("forecast-intent-*.json"))
    finally:
        conn.close()


@pytest.mark.parametrize(
    ("finished", "data_checked", "fixture_finished", "accepted"),
    [(True, True, True, True), (False, False, False, False), (True, False, True, False),
     (True, True, False, False)],
)
def test_official_outcome_archive_requires_event_and_fixture_finality(
    tmp_path, finished, data_checked, fixture_finished, accepted,
):
    conn, _db_path, raw_root, run_id = _raw_capture_set(
        tmp_path, finished=finished, data_checked=data_checked,
        fixture_finished=fixture_finished,
    )
    try:
        if accepted:
            archive = ops._official_event_archive(conn, raw_root=raw_root, fetch_run_id=run_id, event=2)
            assert archive["event"] == 2
            assert archive["parsed"]["fixtures"][0].finished == 1
            # A finalized, data-checked event may retain finished_provisional=true.
            assert archive["payloads"]["fixtures"][0]["finished_provisional"] is True
        else:
            with pytest.raises(ops.ChipEvidenceError, match="finished|final|unfinished"):
                ops._official_event_archive(conn, raw_root=raw_root, fetch_run_id=run_id, event=2)
    finally:
        conn.close()


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"run_started_at": "2026-10-01T10:13:00Z", "run_finished_at": "2026-10-01T10:12:00Z"}, "reversed"),
        ({"run_started_at": "2026-10-01T10:08:00Z", "run_finished_at": "2026-10-01T10:09:00Z"}, "after successful fetch completion"),
        ({"conflicting_archive_time": True}, "share the fetch run's observation identity"),
    ],
)
def test_official_archive_requires_valid_completed_fetch_interval_and_shared_observation(
    tmp_path, options, message,
):
    conn, _db_path, raw_root, run_id = _raw_capture_set(tmp_path, **options)
    try:
        with pytest.raises(ops.ChipEvidenceError, match=message):
            ops._official_event_archive(conn, raw_root=raw_root, fetch_run_id=run_id, event=2)
    finally:
        conn.close()


def test_official_archive_requires_successful_fetch_completion_timestamp(tmp_path):
    conn, _db_path, raw_root, run_id = _raw_capture_set(tmp_path)
    try:
        conn.execute("UPDATE fetch_runs SET finished_at=NULL WHERE id=?", (run_id,))
        conn.commit()
        with pytest.raises(ops.ChipEvidenceError, match="finished_at is required"):
            ops._official_event_archive(conn, raw_root=raw_root, fetch_run_id=run_id, event=2)
    finally:
        conn.close()


def test_capture_retry_recovers_receipt_without_duplicate_rows_and_later_capture_is_distinct(
    tmp_path, monkeypatch,
):
    conn, db_path, raw_root, run_id = _raw_capture_set(tmp_path, players=(1,))
    evidence_root = tmp_path / "evidence"
    monkeypatch.setattr(
        ops, "_load_forecast_origin",
        lambda *_args, **_kwargs: ({}, None, None, None, {}, _forecast_parent()),
    )
    kwargs = {
        "db_path": db_path, "entry_id": 77, "generation_id": "g", "decision_id": "d",
        "route_id": "r", "planning_event": 1, "cutoff": "2026-10-01T10:00:00Z",
        "action": "BB", "observation_id": "bb-obs", "realization_event": 2,
        "fetch_run_id": run_id, "raw_root": raw_root, "evidence_root": evidence_root,
    }
    try:
        first = ops.capture_outcome(conn, **kwargs)
        count_after_first = conn.execute("SELECT COUNT(*) FROM outcome_observation_captures").fetchone()[0]
        assert count_after_first == 1
        assert first["missing_player_ids"] == [2]
        assert first["captures"][0]["player_id"] == 1

        receipt_path = ops._metadata_root(evidence_root) / f"capture-{first['operation_id']}.json"
        receipt_path.unlink()
        recovered = ops.capture_outcome(conn, **kwargs)
        assert recovered == first
        assert conn.execute("SELECT COUNT(*) FROM outcome_observation_captures").fetchone()[0] == 1

        later_root = raw_root
        later_bootstrap = {
            "events": [{"id": 2, "name": "GW2", "finished": True, "data_checked": True}],
            "teams": [{"id": 1, "name": "Team"}],
            "elements": [{"id": 1, "web_name": "P1", "team": 1, "element_type": 3}],
        }
        later_fixtures = [{
            "id": 200, "event": 2, "started": True, "finished": True,
            "finished_provisional": True, "team_h": 1, "team_a": 2,
        }]
        later_live = {"elements": [{"id": 1, "stats": {"minutes": 0, "total_points": 0}}]}
        later_at = ("2026-10-01T11:00:00Z", "2026-10-01T11:01:00Z", "2026-10-01T11:02:00Z")
        new_run = conn.execute(
            "INSERT INTO fetch_runs(started_at,finished_at,status,trigger,current_event,endpoints_ok,"
            "endpoints_failed,raw_dir) VALUES(?,?, 'success','test',2,?,?,?)",
            (later_at[0], later_at[2], json.dumps([
                "bootstrap-static", "fixtures", "event/2/live",
            ]), "[]", str(later_root.resolve())),
        ).lastrowid
        for source, observed_at, payload, event in (
            ("bootstrap_static", later_at[0], later_bootstrap, None),
            ("fixtures", later_at[0], later_fixtures, 2),
            ("event_live_2", later_at[0], later_live, 2),
        ):
            raw_archive.archive_raw_capture(
                later_root,
                source=source,
                observed_at=observed_at,
                body=json.dumps(payload, separators=(",", ":")).encode(),
                event=event,
                run_id=new_run,
            )
        conn.commit()
        later = ops.capture_outcome(conn, **{**kwargs, "fetch_run_id": int(new_run)})
        assert later["operation_id"] != first["operation_id"]
        assert later["captured_at"] > first["captured_at"]
        assert conn.execute("SELECT COUNT(*) FROM outcome_observation_captures").fetchone()[0] == 2
    finally:
        conn.close()


def test_partial_unreceipted_capture_refuses_instead_of_completing_from_a_subset(tmp_path, monkeypatch):
    conn, db_path, raw_root, run_id = _raw_capture_set(tmp_path, players=(1, 2))
    evidence_root = tmp_path / "evidence"
    monkeypatch.setattr(
        ops, "_load_forecast_origin",
        lambda *_args, **_kwargs: ({}, None, None, None, {}, _forecast_parent()),
    )
    archive = ops._official_event_archive(conn, raw_root=raw_root, fetch_run_id=run_id, event=2)
    live_row = archive["payloads"]["event_live"]["elements"][0]
    fields = __import__("fpl_brain.parsers", fromlist=["parse_live_event_totals"]).parse_live_event_totals(live_row)
    conn.commit()
    ol.capture_observation(
        conn,
        grain=ol.GRAIN_PLAYER_EVENT,
        event=2,
        player_id=1,
        fields=fields,
        source_name="player_gameweeks_final",
        official_final_at=archive["official_final_at"],
        observation_state=ol.OBSERVATION_FINAL,
        source_identity="player_gameweeks:2",
        source_payload_sha256=archive["records"]["event_live"]["payload_sha256"],
        fetch_run_id=run_id,
        archive_capture_id=archive["records"]["event_live"]["capture_id"],
        captured_at=archive["available_at"],
    )
    with pytest.raises(ops.ChipEvidenceError, match="partial committed event capture"):
        ops.capture_outcome(
            conn,
            db_path=db_path, entry_id=77, generation_id="g", decision_id="d", route_id="r",
            planning_event=1, cutoff="2026-10-01T10:00:00Z", action="BB",
            observation_id="bb-partial", realization_event=2, fetch_run_id=run_id,
            raw_root=raw_root, evidence_root=evidence_root,
        )
    conn.close()


def test_maturation_receipt_key_is_stable_for_observation_and_event():
    left = ops._maturation_key("tc-origin-7", 13)
    assert left == ops._maturation_key("tc-origin-7", 13)
    assert left != ops._maturation_key("tc-origin-8", 13)
    assert left != ops._maturation_key("tc-origin-7", 14)


@pytest.mark.parametrize(
    "orphan_artifacts",
    [
        [("chip-outcome-orphan.json", {"observation_id": "orphan-observation", "realization_event": 5})],
        [
            ("chip-outcome-orphan-a.json", {"observation_id": "orphan-observation", "realization_event": 5}),
            ("chip-outcome-orphan-b.json", {"observation_id": "orphan-observation", "realization_event": 5}),
        ],
    ],
    ids=["partial-pair", "ambiguous-pairs"],
)
def test_maturation_partial_or_ambiguous_artifacts_refuse_to_finalize_again(
    tmp_path, monkeypatch, orphan_artifacts,
):
    from types import SimpleNamespace

    from fpl_brain import candidate_universe as cu
    from fpl_brain import chip_reservation_calibration as cal
    from fpl_brain import generation_store as gs

    db_path = tmp_path / "fpl.db"
    db_path.touch()
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()
    identity = {"entry_id": 77, "generation_id": "g", "decision_id": "d", "route_id": "r"}
    generation = SimpleNamespace()
    retained_parent = {
        "origin_id": "origin-fixture",
        "receipt": {"selected_event": 5, "expiry_event": 5},
        "causal": {},
    }
    monkeypatch.setattr(ops, "_required_identity", lambda **_kwargs: identity)
    monkeypatch.setattr(
        ops, "_load_forecast_origin",
        lambda *_a, **_k: ({}, generation, {}, object(), {}, retained_parent),
    )
    monkeypatch.setattr(ops, "_verified_capture_receipts_for_observation", lambda *_a, **_k: [{"captures": []}])

    class Snapshot:
        def close(self):
            pass

    monkeypatch.setattr(gs, "_open_generation_snapshot", lambda _generation: Snapshot())
    monkeypatch.setattr(cu, "load_pool", lambda _snapshot: {1: {"position": "MID"}})
    monkeypatch.setattr(
        ops, "_load_causal_origin",
        lambda *_a, **_k: (evidence_root / "causal.json", {}, evidence_root / "forecast.json", {}),
    )
    monkeypatch.setattr(
        cal, "finalize_causal_observation",
        lambda *_a, **_k: pytest.fail("must not finalize when unreceipted artifacts already exist"),
    )
    for name, artifact in orphan_artifacts:
        (evidence_root / name).write_text(json.dumps(artifact), encoding="utf-8")
    conn = sqlite3.connect(":memory:")
    try:
        with pytest.raises(ops.ChipEvidenceError, match="partial or ambiguous"):
            ops.mature_observation(
                conn,
                db_path=db_path, entry_id=77, generation_id="g", decision_id="d", route_id="r",
                planning_event=1, cutoff="2026-10-01T10:00:00Z", action="BB",
                observation_id="orphan-observation", realization_event=5, evidence_root=evidence_root,
            )
    finally:
        conn.close()


@pytest.mark.parametrize("action", ["BB", "TC"])
def test_canonical_bb_tc_origin_forecast_final_capture_and_maturation_lifecycle(
    tmp_path, monkeypatch, action,
):
    """Exercise BB and TC through the complete retained workflow on temp fixtures."""
    import test_four_gw_decision as four_gw
    import test_pe9_production_decision as pe9
    from fpl_brain import (
        candidate_universe as cu,
        chip_route_assembly as cra,
        chip_reservation_calibration as cal,
        chip_reservation_forecast as crf,
        execution,
        generation_store as gs,
        manager_lineup as ml,
        models,
        repositories as repo,
        route_comparator as rc,
        transfer_state as ts,
    )

    original_seed = four_gw._seed_regression_manager

    def seed_with_chips(conn, entry_id=241392, event=4):
        result = original_seed(conn, entry_id=entry_id, event=event)
        repo.upsert_chips(conn, [
            models.ChipRecord(
                id=2, name="bboost", number=1, chip_type="boost", start_event=4, stop_event=5,
            ),
            models.ChipRecord(
                id=3, name="3xc", number=1, chip_type="captain", start_event=4, stop_event=5,
            ),
        ])
        from fpl_brain.ingest_provenance import element_id_sha256

        ids = repo.active_player_ids(conn)
        repo.record_bootstrap_generation(
            conn,
            captured_at="2026-09-11T20:00:00Z",
            accepted=True,
            official_element_count=len(ids),
            parsed_count=len(ids),
            persisted_count=len(ids),
            element_ids=ids,
            element_ids_sha256=element_id_sha256(ids),
            acceptance_rule="fixture-only exact player pool",
            acceptance_rule_version="test-v1",
        )
        return result

    monkeypatch.setattr(four_gw, "_seed_regression_manager", seed_with_chips)
    db_path = tmp_path / "fpl.db"
    conn, generation = pe9._four_gw_certified_manager_world(db_path)
    try:
        def canonical_route_runner(**kwargs):
            canonical = kwargs["canonical_manager_state"]
            source_conn = kwargs["source_conn"]
            gen = kwargs["generation"]
            initial = cra._state_from_payload(canonical["route_state"])
            player_ids = sorted(initial.by_id())
            price_snapshot = cu.price_snapshot_as_of(
                source_conn, gen.planning_event, gen.cutoff, required_player_ids=player_ids,
            )
            scenario = rc.flat_current_price_scenario(price_snapshot, gen.events)
            meta = rc.load_player_meta(source_conn, player_ids)
            positions = {pid: str(meta[pid].position) for pid in player_ids}
            policy = ml.ManagerPolicy(
                starter_ids=(1, 13, 14, 15, 21, 22, 23, 24, 31, 32, 33),
                bench_gk_id=2,
                bench_outfield_order=(41, 42, 25),
                captain_id=21,
                vice_captain_id=22,
            )
            actions = []
            state = initial
            for event in gen.events:
                transition = ts.apply_transfer_batch(
                    state, ts.TransferBatch.roll(), scenario.snapshot_for(event), meta,
                )
                assert transition.ok, transition.errors
                state = transition.next_event_state
                actions.append({
                    "event": int(event), "kind": "ROLL", "transfers": [],
                    "squad_ids": sorted(state.by_id()), "bank_after": state.bank_tenths,
                    "ft_after": state.free_transfers, "hit_points": transition.hit_points,
                })
            route_row = {
                "actions": actions,
                "terminal_bank_tenths": state.bank_tenths,
                "terminal_ft": state.free_transfers,
                "per_event": [
                    {"event": int(event), "policy": policy.as_dict()}
                    for event in gen.events
                ],
            }
            return {
                "decision": {"status": "FIXTURE_ROUTE_REPLAY"},
                "consumed_manager_state": canonical,
                "provenance": {"planning_context_hash": gs.four_gw_manager_context_identity(canonical)},
                "runner_identity": "chip-evidence-lifecycle-fixture",
                "artifact_blocks": {"search": {"config": {"search_draws": 2, "seed": 62026}}},
                "finalist_refinement": {"route_table": {"routes": {"route_fixture": route_row}}},
            }

        monkeypatch.setattr(gs, "_decision_executor", lambda _profile: canonical_route_runner)
        decision = gs.make_decision(
            conn,
            {"entry_id": 241392, "planning_event": 4, "cutoff": generation.cutoff, "season": "2026/27"},
            4,
            horizon_kind=gs.HORIZON_KIND_FOUR_GW,
            generation_id=generation.generation_id,
            profile=gs.DecisionProfile(kind=gs.HORIZON_KIND_FOUR_GW),
        )
        decision_id = str(decision["decision_record_id"])
        # The accepted forecast contract requires a verified same-origin
        # CHIP_RESERVATION product for any future expiry, including one inside
        # the normal decision's four-event window.
        from datetime import datetime, timedelta, timezone

        continuation_controller = execution.ExecutionController(conn)
        continuation_controller.create_run(
            planning_event=4,
            planning_cutoff=generation.cutoff,
            hard_stop_at=datetime.now(timezone.utc) + timedelta(hours=1),
            label="chip_evidence_lifecycle_fixture_continuation",
        )
        continuation_controller.start()
        continuation_controller.acquire_writer_lease()
        try:
            continuation = gs.certify_chip_reservation_generation(
                conn,
                chip_generation_id=generation.generation_id,
                planning_event=4,
                cutoff=generation.cutoff,
                events=generation.events,
                runs_by_event={event: generation.runs_for(event) for event in generation.events},
                snapshot=generation.snapshot,
                controller=continuation_controller,
            )
        except BaseException as failure:
            continuation_controller.finish(execution.RUN_FAILED, str(failure))
            raise
        continuation_controller.finish(execution.RUN_COMPLETE)
        identity = {
            "db_path": db_path,
            "entry_id": 241392,
            "generation_id": generation.generation_id,
            "decision_id": decision_id,
            "route_id": "route_fixture",
            "planning_event": 4,
            "cutoff": generation.cutoff,
            "evidence_root": tmp_path / "evidence",
        }
        origin = ops.register_origin(conn, **identity)
        origin_id = origin["receipt"]["origin_id"]
        route = cra.load_verified_normal_route(conn, decision_id, route_id="route_fixture")
        loaded_generation = gs.load_generation(conn, generation.generation_id)
        coverage_product = crf.build_reservation_coverage_product(
            conn,
            action=action,
            source_identity=ops._source_identity(route, loaded_generation),
            expiry_event=5,
            product_generation_id=continuation.generation_id,
        )
        requirements = ops._world_cache_requirements(
            conn, route=route, generation=loaded_generation,
            coverage_product=coverage_product, expiry_event=5,
        )
        assert [row["event"] for row in requirements] == [5]
        requirement = requirements[0]
        player_ids = requirement["player_ids"]
        matrix = {
            "worlds": 2,
            "player_ids": player_ids,
            "core": {pid: [5.0, 6.0] for pid in player_ids},
            "minutes": {pid: [90.0, 90.0] for pid in player_ids},
            "expected_bonus": {pid: 0.0 for pid in player_ids},
            "role_actionability": {pid: False for pid in player_ids},
        }
        cache_entry = {
            "worlds": matrix["worlds"],
            "player_ids": matrix["player_ids"],
            "core": {str(pid): values for pid, values in matrix["core"].items()},
            "minutes": {str(pid): values for pid, values in matrix["minutes"].items()},
            "expected_bonus": {str(pid): value for pid, value in matrix["expected_bonus"].items()},
            "role_actionability": {str(pid): value for pid, value in matrix["role_actionability"].items()},
            ro.CACHE_CONTENT_DIGEST_KEY: ro.world_matrix_content_identity(matrix),
        }
        # Short separate path avoids Windows MAX_PATH in pytest's long temp root.
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        assert cache_dir.is_dir()
        (cache_dir / f"{requirement['cache_key']}.json").write_text(
            json.dumps(cache_entry), encoding="utf-8",
        )

        forecast_receipt = ops.forecast_origin(
            conn,
            **identity,
            action=action,
            expiry_event=5,
            observation_id=f"{action.lower()}-fixture-observation-1",
            continuation_generation_id=continuation.generation_id,
            cache_dir=cache_dir,
        )
        assert forecast_receipt["selected_event"] == 5
        assert forecast_receipt["materialized_worlds"] is False
        assert forecast_receipt["issued_at"] >= forecast_receipt["calculation_started_at"]

        # Simulate a later official-final fetch only in this temporary test DB and raw archive.
        final_at = ops._utc_now()
        conn.execute("UPDATE events SET finished=1,data_checked=1,updated_at=? WHERE id=5", (final_at,))
        conn.execute(
            "UPDATE fixtures SET started=1,finished=1,finished_provisional=1,updated_at=? WHERE id=42",
            (final_at,),
        )
        conn.commit()
        raw_root = tmp_path / "raw"
        import datetime as dt

        base = ops._parse_utc(final_at, name="fixture final timestamp")
        raw_observed_at = base.isoformat(timespec="microseconds").replace("+00:00", "Z")
        run_started_at = (base + dt.timedelta(seconds=1)).isoformat(timespec="microseconds").replace("+00:00", "Z")
        run_finished_at = (base + dt.timedelta(seconds=3)).isoformat(timespec="microseconds").replace("+00:00", "Z")
        run_id = int(conn.execute(
            "INSERT INTO fetch_runs(started_at,finished_at,status,trigger,current_event,endpoints_ok,"
            "endpoints_failed,raw_dir) VALUES(?,?,'success','chip-evidence-test',5,?,?,?)",
            (run_started_at, run_finished_at, json.dumps([
                "bootstrap-static", "fixtures", "event/5/live",
            ]), "[]", str(raw_root.resolve())),
        ).lastrowid)
        conn.commit()
        bootstrap = {
            "events": [{"id": 5, "name": "GW5", "finished": True, "data_checked": True}],
            "teams": [{"id": 1, "name": "One"}, {"id": 2, "name": "Two"}],
            "elements": [{
                "id": pid, "web_name": f"P{pid}", "team": 1,
                "element_type": {"GKP": 1, "DEF": 2, "MID": 3, "FWD": 4}[route.positions[pid]],
            } for pid in route.proposed_owned_ids],
        }
        fixtures = [{
            "id": 42, "event": 5, "started": True, "finished": True,
            "finished_provisional": True, "team_h": 1, "team_a": 2,
        }]
        live = {"elements": [{"id": pid, "stats": {"minutes": 0, "total_points": 0}}
                              for pid in route.proposed_owned_ids]}
        for source, observed_at, payload, event_id in (
            ("bootstrap_static", raw_observed_at, bootstrap, None),
            ("fixtures", raw_observed_at, fixtures, 5),
            ("event_live_5", raw_observed_at, live, 5),
        ):
            raw_archive.archive_raw_capture(
                raw_root,
                source=source,
                observed_at=observed_at,
                body=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
                event=event_id,
                run_id=run_id,
            )

        capture = ops.capture_outcome(
            conn,
            **identity,
            action=action,
            observation_id=f"{action.lower()}-fixture-observation-1",
            realization_event=5,
            fetch_run_id=run_id,
            raw_root=raw_root,
        )
        assert capture["source_name"] == "player_gameweeks_final"
        assert capture["source_identity"] == "player_gameweeks:5"
        assert capture["missing_player_ids"] == []
        assert capture["raw_observed_at"] == raw_observed_at
        assert capture["available_at"] == run_finished_at
        assert capture["captured_at"] == run_finished_at

        finalize_calls = []
        original_finalize = cal.finalize_causal_observation

        def count_finalization(*args, **kwargs):
            finalize_calls.append(1)
            return original_finalize(*args, **kwargs)

        monkeypatch.setattr(cal, "finalize_causal_observation", count_finalization)
        key = ops._maturation_key(f"{action.lower()}-fixture-observation-1", 5)
        receipt_path = ops._metadata_root(identity["evidence_root"]) / f"maturation-{key}.json"
        if action == "BB":
            original_write = ops._write_new_json
            failed_receipt = False

            def fail_maturation_receipt_once(path, value):
                nonlocal failed_receipt
                if Path(path) == receipt_path and not failed_receipt:
                    failed_receipt = True
                    raise OSError("injected maturation-receipt publication failure")
                return original_write(path, value)

            monkeypatch.setattr(ops, "_write_new_json", fail_maturation_receipt_once)
            with pytest.raises(OSError, match="injected maturation-receipt"):
                ops.mature_observation(
                    conn,
                    **identity,
                    action=action,
                    observation_id=f"{action.lower()}-fixture-observation-1",
                    realization_event=5,
                )
            assert failed_receipt is True
            assert len(finalize_calls) == 1
            assert not receipt_path.exists()
            monkeypatch.setattr(ops, "_write_new_json", original_write)

        matured = ops.mature_observation(
            conn,
            **identity,
            action=action,
            observation_id=f"{action.lower()}-fixture-observation-1",
            realization_event=5,
        )
        assert len(finalize_calls) == 1
        assert receipt_path.is_file()
        assert len(list(Path(identity["evidence_root"]).glob("chip-outcome-*.json"))) == 1
        assert matured["receipt"]["receipt_key"] == key
        assert matured["calibration_row"]["realized_value"] == 0.0
        retried = ops.mature_observation(
            conn,
            **identity,
            action=action,
            observation_id=f"{action.lower()}-fixture-observation-1",
            realization_event=5,
        )
        assert retried["receipt"] == matured["receipt"]
        handoff = ops.load_verified_calibration_handoff(conn, **identity)
        assert len(handoff) == 1
        assert handoff[0]["observation_id"] == f"{action.lower()}-fixture-observation-1"
    finally:
        conn.close()
