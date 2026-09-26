"""PE-9 amendment 2 — the content-addressed generation store.

The adversarial cases here were REPURPOSED, as amendment 2 section 17 requires, away
from impossible "Python object unforgeability".  PE-9 no longer claims that an
in-process Python value cannot be forged -- its threat model excludes arbitrary
hostile code inside the trusted process -- and instead claims that a decision
consumes predictive data only through a persisted certified generation whose exact
provenance an INDEPENDENT VERIFIER can re-derive from authoritative evidence.  Every
test below therefore attacks the EVIDENCE, not an object's identity.
"""

from __future__ import annotations

import inspect
import json
import sqlite3
from pathlib import Path

import pytest

from fpl_brain import certified_bundle as cb
from fpl_brain import four_gw_decision as fg
from fpl_brain import free_hit_request_adapter as fha
from fpl_brain import manager_worlds as mw
from fpl_brain import route_comparator as rc
from fpl_brain import route_optimizer as ro
from fpl_brain import route_stability as rs
from fpl_brain import generation_store as gs
from fpl_brain import finalist_refinement as fr
from fpl_brain import replay_worlds as rw
from fpl_brain.database import connect_database

import generation_fixtures as gf
from test_pe9_certification_integration import (
    CODE_SNAPSHOT,
    CUTOFF,
    DATA_SNAPSHOT,
    HORIZON,
    OTHER_CUTOFF,
    VERSIONS,
    _artifact,
    _calibration,
    _generation as _certify_generation,
    _optimizer_config,
    _world,
)

GENERATION_ID = "sha256:" + "a" * 64


def _generation(conn, runs, **kw):
    return _certify_generation(conn, runs, **kw)


def _matrix(union=(1, 2)):
    return {
        "worlds": 1,
        "player_ids": [int(pid) for pid in union],
        "core": {int(pid): [0.0] for pid in union},
        "minutes": {int(pid): [0.0] for pid in union},
        "expected_bonus": {int(pid): 0.0 for pid in union},
        "role_actionability": {int(pid): False for pid in union},
    }


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_refusal_unknown_generation_id_and_unset_pointer():
    conn, runs = _world()
    try:
        _generation(conn, runs)
        with pytest.raises(gs.GenerationUnknown) as caught:
            gs.load_generation(conn, "sha256:" + "0" * 64)
        assert caught.value.token == gs.DIAG_GENERATION_UNKNOWN
        # An unset pointer is a DIFFERENT fact from an unknown id.
        conn.execute("DELETE FROM current_generation")
        with pytest.raises(gs.GenerationRefused) as pointerless:
            gs.resolve_generation(conn, planning_event=5)
        assert pointerless.value.token == gs.DIAG_GENERATION_POINTER_UNSET
    finally:
        conn.close()


def test_refusal_manifest_mutated_after_the_generation_id_was_calculated():
    conn, runs = _world()
    try:
        generation = _generation(conn, runs)
        row = conn.execute(
            "SELECT manifest_json FROM generation WHERE generation_id=?", (generation.generation_id,)
        ).fetchone()
        manifest = json.loads(row["manifest_json"])
        manifest["per_event"]["5"]["runs"]["xpts_v1"] = 99999
        # The append-only trigger is the FIRST line of defence: an UPDATE is refused.
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "UPDATE generation SET manifest_json=? WHERE generation_id=?",
                (json.dumps(manifest), generation.generation_id),
            )
        # The content address is the SECOND: the tampered manifest does not digest
        # back to the id it was stored under, so load_generation refuses it.
        tampered = json.loads(json.dumps(generation.manifest))
        tampered["per_event"]["5"]["runs"]["xpts_v1"] = 99999
        assert gs.generation_id_of(tampered) != generation.generation_id
    finally:
        conn.close()


def test_refusal_wrong_event_wrong_horizon_kind_and_unknown_kind():
    conn, runs = _world()
    try:
        generation = _generation(conn, runs)
        with pytest.raises(gs.GenerationRefused) as wrong_event:
            gs.resolve_generation(conn, planning_event=6, generation_id=generation.generation_id)
        assert wrong_event.value.token == cb.STATE_PREDICTIVE_BUNDLE_INCOHERENT
        with pytest.raises(gs.GenerationRefused) as wrong_kind:
            gs.resolve_generation(
                conn, planning_event=5, generation_id=generation.generation_id,
                horizon_kind=gs.HORIZON_KIND_MANAGER_WORLD,
            )
        assert wrong_kind.value.token == cb.STATE_PREDICTIVE_BUNDLE_INCOHERENT
        with pytest.raises(gs.GenerationRefused) as unknown_kind:
            gs.resolve_generation(conn, planning_event=5, horizon_kind="NOT_A_KIND")
        assert unknown_kind.value.token == gs.DIAG_GENERATION_HORIZON_KIND_UNKNOWN
    finally:
        conn.close()


def test_refusal_wrong_cutoff():
    conn, runs = _world()
    try:
        generation = _generation(conn, runs)
        with pytest.raises(gs.GenerationRefused):
            gs.assert_cutoff_matches(conn, generation, cutoff=OTHER_CUTOFF)
        gs.assert_cutoff_matches(conn, generation, cutoff=CUTOFF)
    finally:
        conn.close()


FAMILIES = ("minutes_v1", "team_strength_v1", "player_rates_v1", "xpts_v1", "monte_carlo_v1")
RUN_IDS = {"minutes_v1": 1, "team_strength_v1": 2, "player_rates_v1": 3, "xpts_v1": 4,
           "monte_carlo_v1": 5}


def _hand_world(
    *,
    event=5,
    versions=None,
    status=None,
    context_hash=None,
    xpts_upstream=None,
    drop_family=None,
    dgw_fixtures=1,
    mc_rows=True,
):
    """A one-event world built row by row, so a DEFECT can be present at creation.

    Projection runs are immutable once ``complete``, so nothing here can be edited
    afterwards -- which is itself part of the PE-9 guarantee: the evidence a
    generation was certified from cannot be mutated into a different world.
    """

    conn = connect_database(":memory:")
    gf.base_world(conn)
    runs = {family: run_id for family, run_id in RUN_IDS.items() if family != drop_family}
    fixture_ids = list(range(1000, 1000 + int(dgw_fixtures)))
    with conn:
        gf.add_event(conn, event)
        for fixture_id in fixture_ids:
            gf.add_fixture(conn, fixture_id, event, 1, 2)
        for family, run_id in runs.items():
            gf.add_run(
                conn, run_id, family, event,
                version=None if versions is None else versions.get(family),
                status="complete" if status is None else status.get(family, "complete"),
                context_hash=(
                    None if context_hash is None else context_hash.get(family, gf.CONTEXT_HASH)
                ),
            )
        if "xpts_v1" in runs:
            upstream = dict(xpts_upstream or {})
            for fixture_id in fixture_ids:
                gf.add_xpts_row(
                    conn, runs["xpts_v1"], fixture_id, event,
                    minutes_run_id=int(upstream.get("minutes_run_id", runs.get("minutes_v1", 1))),
                    team_run_id=int(upstream.get("team_run_id", runs.get("team_strength_v1", 2))),
                    rate_run_id=int(upstream.get("rate_run_id", runs.get("player_rates_v1", 3))),
                )
        if mc_rows and "monte_carlo_v1" in runs and "xpts_v1" in runs:
            mc_fixtures = fixture_ids[:1] if mc_rows == "first_fixture_only" else fixture_ids
            for fixture_id in mc_fixtures:
                gf.add_mc_row(
                    conn, runs["monte_carlo_v1"], fixture_id, event,
                    xpts_run_id=runs["xpts_v1"], minutes_run_id=runs["minutes_v1"],
                    team_run_id=runs["team_strength_v1"], rate_run_id=runs["player_rates_v1"],
                )
    return conn, {int(event): runs}


def test_refusal_unsupported_model_version():
    conn, runs = _hand_world(versions={"xpts_v1": "future_v9"})
    try:
        # The world cannot be CERTIFIED, so no generation row exists at all: that is
        # the whole point of "a generation exists only when certification passed".
        with pytest.raises(gs.GenerationNotCertified) as caught:
            _generation(conn, runs, events=(5,), horizon_kind=gs.HORIZON_KIND_MANAGER_WORLD)
        assert gs.DIAG_GENERATION_NOT_CERTIFIED == caught.value.token
        assert cb.STATE_UNSUPPORTED_MODEL_VERSION in str(caught.value)
        assert conn.execute("SELECT COUNT(*) FROM generation").fetchone()[0] == 0
    finally:
        conn.close()


def test_refusal_missing_dependency_edge():
    """The xPts run is wired to a Minutes run that does not exist."""

    conn, runs = _hand_world(xpts_upstream={"minutes_run_id": 999})
    try:
        with pytest.raises(gs.GenerationNotCertified) as caught:
            _generation(conn, runs, events=(5,), horizon_kind=gs.HORIZON_KIND_MANAGER_WORLD)
        assert "references minutes_v1 run 999" in str(caught.value)
    finally:
        conn.close()


def test_refusal_missing_projection_is_never_zero_filled_into_a_certificate():
    conn, runs = _hand_world(drop_family="xpts_v1", mc_rows=False)
    try:
        # The family is simply ABSENT from the bundle being certified: nothing
        # defaults it and nothing zero-fills it.
        with pytest.raises(gs.GenerationNotCertified) as caught:
            _generation(conn, runs, events=(5,), horizon_kind=gs.HORIZON_KIND_MANAGER_WORLD)
        assert "missing family xpts_v1" in str(caught.value)
    finally:
        conn.close()


def test_refusal_partial_dgw_evidence_is_not_silently_complete():
    """A DGW whose Monte Carlo run carries only ONE fixture's rows is refused.

    The xPts rows are complete for both fixtures while the Monte Carlo rows are not,
    so the dependency edge cannot be verified from the rows themselves -- the exact
    shape a silently understated double gameweek would take.
    """

    conn, runs = _hand_world(dgw_fixtures=2, mc_rows="first_fixture_only")
    try:
        with pytest.raises(gs.GenerationNotCertified) as caught:
            _generation(conn, runs, events=(5,), horizon_kind=gs.HORIZON_KIND_MANAGER_WORLD)
        assert "partial double gameweek" in str(caught.value)
    finally:
        conn.close()


def test_refusal_inconsistent_planning_context_hash():
    """One family certified under a different planning context is a different world."""

    conn, runs = _hand_world(context_hash={"player_rates_v1": "other"})
    try:
        # The declared context of THIS certification is the consistent one, so the
        # disagreeing family is caught by the certification gate rather than assumed.
        with pytest.raises(gs.GenerationNotCertified) as caught:
            _generation(conn, runs, events=(5,), horizon_kind=gs.HORIZON_KIND_MANAGER_WORLD)
        assert "planning_context_hash differs across families" in str(caught.value)
    finally:
        conn.close()


def test_refusal_mutated_snapshot(tmp_path):
    conn, runs = _world()
    try:
        generation = _generation(conn, runs, snapshot_path=tmp_path / "snapshot.db")
        gs.require_snapshot_retained(generation)
        Path(generation.snapshot["path"]).write_bytes(b"tampered")
        with pytest.raises(gs.GenerationSnapshotUnverified):
            gs.require_snapshot_retained(generation)
        with pytest.raises(gs.GenerationRefused):
            gs.verify_generation(conn, generation.generation_id)
    finally:
        conn.close()


def test_refusal_missing_snapshot_must_be_retained(tmp_path):
    conn, runs = _world()
    try:
        generation = _generation(conn, runs, snapshot_path=tmp_path / "other" / "snapshot.db")
        Path(generation.snapshot["path"]).unlink()
        with pytest.raises(gs.GenerationSnapshotUnverified) as caught:
            gs.require_snapshot_retained(generation)
        assert "must be retained" in str(caught.value)
    finally:
        conn.close()


def test_refusal_stale_or_invalid_cache_content(tmp_path):
    """Amendment 2 section 12: a bad cache entry is a MISS, never a wrong answer.

    The cache is optimization, not authority, so every way an entry can fail to be
    the matrix it is stored as -- a foreign key, a missing content digest, content
    that does not reproduce the recorded digest -- costs a rebuild and nothing else.
    The MISS is recorded in the load's own evidence instead of being swallowed.
    """

    conn, runs = _world()
    try:
        generation = _generation(conn, runs, snapshot_path=tmp_path / "snapshot.db")
        union = [1, 2]
        config = _optimizer_config()
        key = ro.world_cache_key(
            event=5, generation_id=generation.generation_id, runs=generation.runs_for(5),
            config=config, union_ids=union,
        )
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        # A WARM entry written by the loader itself carries a content digest and is a HIT.
        first, first_info = ro.build_event_worlds(
            conn, generation, 5, union, config, cache_dir=cache_dir,
        )
        assert first_info["source"] == "generated"
        entry_path = cache_dir / f"{key}.json"
        recorded_digest = json.loads(entry_path.read_text(encoding="utf-8"))[
            ro.CACHE_CONTENT_DIGEST_KEY
        ]
        assert recorded_digest
        _, hit_info = ro.build_event_worlds(
            conn, generation, 5, union, config, cache_dir=cache_dir,
        )
        assert hit_info["source"] == "cache"
        assert hit_info["cache"]["status"] == ro.CACHE_HIT

        def serialized(matrix):
            return {
                "worlds": matrix["worlds"], "player_ids": list(matrix["player_ids"]),
                "core": {str(k): v for k, v in matrix["core"].items()},
                "minutes": {str(k): v for k, v in matrix["minutes"].items()},
                "expected_bonus": {str(k): v for k, v in matrix["expected_bonus"].items()},
                "role_actionability": {str(k): bool(v) for k, v in matrix["role_actionability"].items()},
            }

        # An entry with NO digest (a pre-PE-9 file) is a MISS, not a silent trust.
        entry_path.write_text(json.dumps(serialized(first)), encoding="utf-8")
        _, undigested_info = ro.build_event_worlds(
            conn, generation, 5, union, config, cache_dir=cache_dir,
        )
        assert undigested_info["source"] == "generated"
        assert undigested_info["cache"]["status"] == ro.CACHE_MISS_NO_DIGEST

        # Content TAMPERED with under a correct-looking key is a MISS too: the recorded
        # digest was computed over the original worlds, so it does not reproduce from
        # the edited ones.  The rebuilt worlds are the certified ones, not the edits.
        tampered = serialized(first)
        tampered["core"] = {
            str(k): [v + 1.0 for v in series] for k, series in first["core"].items()
        }
        tampered[ro.CACHE_CONTENT_DIGEST_KEY] = recorded_digest
        entry_path.write_text(json.dumps(tampered), encoding="utf-8")
        rebuilt, tampered_info = ro.build_event_worlds(
            conn, generation, 5, union, config, cache_dir=cache_dir,
        )
        assert tampered_info["source"] == "generated"
        assert tampered_info["cache"]["status"] == ro.CACHE_MISS_CONTENT_MISMATCH
        assert rebuilt["core"] == first["core"], "the edited cache never becomes the answer"

        # An entry under a FOREIGN key is simply not this generation's content, so the
        # correct key still misses and the load still produces the certified worlds.
        (cache_dir / ("0" * 64 + ".json")).write_text(json.dumps(tampered), encoding="utf-8")
        entry_path.write_text(json.dumps(tampered), encoding="utf-8")
        _, foreign_info = ro.build_event_worlds(
            conn, generation, 5, union, config, cache_dir=cache_dir,
        )
        assert foreign_info["source"] == "generated"

        # An UNREADABLE entry is a MISS, never a refusal: the cache is optimization.
        entry_path.write_text("{not json", encoding="utf-8")
        _, unreadable_info = ro.build_event_worlds(
            conn, generation, 5, union, config, cache_dir=cache_dir,
        )
        assert unreadable_info["source"] == "generated"
        assert unreadable_info["cache"]["status"] == ro.CACHE_MISS_UNREADABLE
    finally:
        conn.close()


def test_cache_content_digest_mismatch_never_changes_the_answer(tmp_path):
    """A cache whose bytes were edited changes nothing but the time to rebuild."""

    conn, runs = _world()
    try:
        generation = _generation(conn, runs, snapshot_path=tmp_path / "snapshot.db")
        union = [1, 2]
        config = _optimizer_config()
        warm = tmp_path / "warm"
        warm.mkdir()
        first, _ = ro.build_event_worlds(conn, generation, 5, union, config, cache_dir=warm)
        key = ro.world_cache_key(
            event=5, generation_id=generation.generation_id, runs=generation.runs_for(5),
            config=config, union_ids=union,
        )
        payload = json.loads((warm / f"{key}.json").read_text(encoding="utf-8"))
        payload["core"] = {k: [v + 100.0 for v in series] for k, series in payload["core"].items()}
        (warm / f"{key}.json").write_text(json.dumps(payload), encoding="utf-8")
        second, second_info = ro.build_event_worlds(conn, generation, 5, union, config, cache_dir=warm)
        assert second_info["source"] == "generated"
        assert second["core"] == first["core"], "an edited cache never becomes the answer"
    finally:
        conn.close()


def test_refusal_replay_or_test_world_presented_to_production():
    conn, runs = _world()
    try:
        generation = _generation(conn, runs)
        declaration = rw.ReplayDeclaration(
            token=rw.NON_PRODUCTION_REPLAY_ONLY, owner="test_pe9",
            purpose="prove an injected world cannot enter a production load",
        )
        with pytest.raises(cb.CertificationRefused) as caught:
            ro.build_event_worlds(
                conn, generation, 5, [1], _optimizer_config(),
                non_production_worlds=rw.ReplayWorlds(
                    declaration=declaration, matrices={5: _matrix((1,))},
                ).as_non_production_worlds(),
            )
        assert caught.value.token == ro.DIAG_NON_PRODUCTION_WORLDS_FORBIDDEN
    finally:
        conn.close()


def test_refusal_production_caller_passing_a_matrix_bundles_runs_or_certification():
    conn, runs = _world()
    try:
        forbidden = {
            "matrix": _matrix(),
            "bundles": {5: {"runs": runs[5]}},
            "runs": runs[5],
            "certification": _artifact(conn, runs),
            "manifest_digest": "sha256:" + "f" * 64,
            "cache_dir": "/tmp/x",
            "token": "anything",
        }
        for name, value in forbidden.items():
            with pytest.raises(gs.ProductionDescriptorOnly) as caught:
                gs.make_decision(conn, {"universe": {}}, 5, **{name: value})
            assert caught.value.token == gs.DIAG_PRODUCTION_DESCRIPTOR_ONLY
            assert name in str(caught.value)
    finally:
        conn.close()


def test_refusal_latest_per_family_bypass():
    conn, runs = _world()
    try:
        generation = _generation(conn, runs)
        with conn:
            gf.add_run(conn, 912, "minutes_v1", 5)
            gf.add_run(conn, 913, "xpts_v1", 5)
            gf.add_xpts_row(
                conn, 913, 1000, 5, minutes_run_id=912,
                team_run_id=runs[5]["team_strength_v1"], rate_run_id=runs[5]["player_rates_v1"],
            )
        newest = {
            5: {
                "minutes_v1": 912, "team_strength_v1": runs[5]["team_strength_v1"],
                "player_rates_v1": runs[5]["player_rates_v1"], "xpts_v1": 913,
                "monte_carlo_v1": runs[5]["monte_carlo_v1"],
            }
        }
        assert newest[5] != generation.runs_for(5)
        with pytest.raises(cb.CertificationRefused) as caught:
            ro.build_event_worlds(conn, newest, 5, [1], _optimizer_config())
        assert caught.value.token == ro.DIAG_CERTIFIED_GENERATION_REQUIRED
        assert gs.support_by_event(generation)[5]["matched_runs"] == generation.runs_for(5)
    finally:
        conn.close()


def test_refusal_mutable_fact_tables_changed_after_a_historical_certification():
    conn, runs = _world()
    try:
        generation = _generation(conn, runs)
        before = generation.generation_id
        with conn:
            conn.execute("UPDATE players SET web_name='Renamed' WHERE id=1")
            conn.execute("UPDATE fixtures SET kickoff_time='2026-10-01T14:00:00Z' WHERE event=5")
        assert gs.load_generation(conn, before).generation_id == before
        # The RUN ROWS are the evidence and they are intact, so the generation still
        # verifies: the mutable fact tables are not its identity.
        assert gs.verify_generation(conn, before)["verified"] is True
    finally:
        conn.close()


def test_refusal_missing_pe8_identity_is_not_an_unknown_one():
    conn, runs = _world()
    try:
        calibration = _calibration(runs_by_event=runs)
        identityless = {k: v for k, v in calibration.items() if k != "identity"}
        with pytest.raises(cb.CertificationRefused) as caught:
            cb.validate_calibration_evidence(
                calibration=identityless, event=5, cutoff=CUTOFF, runs=runs[5],
                certification_identity=None, require_calibration=True,
            )
        assert caught.value.token == cb.STATE_EVIDENCE_MISSING
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Successes
# ---------------------------------------------------------------------------


def test_success_canonical_four_event_certified_generation():
    conn, runs = _world()
    try:
        generation = _generation(conn, runs)
        assert list(generation.events) == list(HORIZON)
        assert generation.horizon_kind == gs.HORIZON_KIND_FOUR_GW
        assert int(generation.manifest["horizon_length"]) == 4
        assert generation.manifest["horizon_state"] == fg.DECISION_HORIZON_COMPLETE
        assert generation.generation_id.startswith("sha256:")
        assert generation.generation_id == gs.generation_id_of(generation.manifest)
        for event in HORIZON:
            assert generation.runs_for(event) == runs[event]
    finally:
        conn.close()


def test_success_idempotent_recertification_produces_the_same_generation_id():
    conn, runs = _world()
    try:
        first = _generation(conn, runs)
        second = _generation(conn, runs)
        assert first.generation_id == second.generation_id
        assert conn.execute("SELECT COUNT(*) FROM generation").fetchone()[0] == 1
    finally:
        conn.close()


def test_success_generation_and_pointer_transaction_is_atomic():
    conn, runs = _world()
    try:
        generation = _generation(conn, runs)
        assert gs.current_generation_id(conn, 5) == generation.generation_id
        before_rows = conn.execute("SELECT COUNT(*) FROM generation").fetchone()[0]
        before_pointer = gs.current_generation_id(conn, 5)
        # A crash BEFORE commit leaves no generation and the old pointer.
        with pytest.raises(RuntimeError):
            with conn:
                conn.execute(
                    "INSERT INTO generation(generation_id, manifest_json, manifest_sha256,"
                    " planning_event, horizon_kind, cutoff, created_at)"
                    " VALUES (?, '{}', ?, 5, 'FOUR_GW', ?, '2026-09-19T11:02:00Z')",
                    ("sha256:" + "1" * 64, "sha256:" + "1" * 64, CUTOFF),
                )
                raise RuntimeError("crash before commit")
        assert conn.execute("SELECT COUNT(*) FROM generation").fetchone()[0] == before_rows
        assert gs.current_generation_id(conn, 5) == before_pointer
    finally:
        conn.close()


def test_success_append_only_triggers_refuse_update_and_delete():
    conn, runs = _world()
    try:
        generation = _generation(conn, runs)
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("DELETE FROM generation WHERE generation_id=?", (generation.generation_id,))
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "UPDATE generation SET cutoff='x' WHERE generation_id=?", (generation.generation_id,)
            )
        decision_id = gs.append_engine_decision_record(
            conn, generation=generation, manager_packet_sha256="sha256:" + "a" * 64,
            request_sha256="sha256:" + "b" * 64, result_sha256="sha256:" + "c" * 64,
            runner_identity="test", evidence={"k": 1},
        )
        conn.commit()
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "UPDATE engine_decision_records SET runner_identity='x' WHERE decision_id=?",
                (decision_id,),
            )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("DELETE FROM engine_decision_records WHERE decision_id=?", (decision_id,))
    finally:
        conn.close()


def test_success_decision_resolves_current_and_explicit_historical_generations(tmp_path):
    conn, runs = _world()
    try:
        generation = _generation(conn, runs)
        assert gs.resolve_generation(conn, planning_event=5).generation_id == generation.generation_id
        historical = gs.resolve_generation(
            conn, planning_event=5, generation_id=generation.generation_id
        )
        assert historical.generation_id == generation.generation_id
        # A later generation for the same event moves the POINTER; the explicit
        # selector still resolves the historical one, so a decision stays pinned.
        # A second generation over the SAME run rows but a DIFFERENT pinned snapshot
        # is a different predictive world identity, so the pointer moves.
        newer = _generation(conn, runs, snapshot_path=tmp_path / "second.db")
        assert newer.generation_id != generation.generation_id
        assert gs.resolve_generation(conn, planning_event=5).generation_id == newer.generation_id
        assert gs.resolve_generation(
            conn, planning_event=5, generation_id=generation.generation_id
        ).generation_id == generation.generation_id
    finally:
        conn.close()


def test_success_generation_verify_and_decision_verify_pass(tmp_path):
    conn, runs = _world()
    try:
        generation = _generation(conn, runs, snapshot_path=tmp_path / "snapshot.db")
        report = gs.verify_generation(conn, generation.generation_id)
        assert report["verified"] is True
        assert report["manifest_digest_matches"] is True
        assert report["dependency_closure_reproduced"] is True
        assert report["snapshot_identity"] == "VERIFIED"
        assert report["horizon_state"] == fg.DECISION_HORIZON_COMPLETE

        artifact_path = tmp_path / "four_gw_decision.json"
        artifact = {
            "schema": "fpl_brain.four_gw_decision.v1",
            "planning_event": 5, "planning_cutoff": CUTOFF,
            "decision_events": list(HORIZON),
            "provenance": {"generation_id": generation.generation_id},
            "decision": {"k": "v"}, "suppression_reasons": [],
            "decision_confidence": None, "fixture_horizon": None,
            "finalist_refinement": {"route_table": {"routes": {}}},
        }
        artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
        decision_id = gs.append_engine_decision_record(
            conn, generation=generation,
            manager_packet_sha256="sha256:" + "a" * 64,
            request_sha256="sha256:" + "b" * 64,
            result_sha256=gs.result_identity_of(artifact),
            runner_identity="test", evidence={"schema": gs.DECISION_RECORD_SCHEMA},
            decision_artifact_ref=str(artifact_path),
        )
        conn.commit()
        verified = gs.verify_decision(conn, decision_id)
        assert verified["verified"] is True
        assert verified["generation_verified"] is True
        assert verified["record_digest_recomputed"] is True
        assert verified["decision_artifact"] == "VERIFIED"
        # The verifier REPORTS its replay boundary instead of inventing determinism.
        assert verified["replay_boundary"] == "DECISION_ARTIFACT_RE_EXECUTED_AND_BYTE_VERIFIED"

        with pytest.raises(gs.GenerationRefused) as unknown:
            gs.verify_decision(conn, "sha256:" + "9" * 64)
        assert unknown.value.token == gs.DIAG_DECISION_RECORD_UNKNOWN
        # A decision whose artifact disagrees with its recorded result digest is refused.
        artifact["decision"] = {"k": "w"}
        artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
        with pytest.raises(gs.GenerationRefused) as caught:
            gs.verify_decision(conn, decision_id)
        assert caught.value.token == gs.DIAG_DECISION_RECORD_INVALID
    finally:
        conn.close()


def test_success_pe8_evidence_participates_without_promotion():
    conn, runs = _world()
    try:
        calibration = _calibration(runs_by_event=runs)
        generation = _generation(conn, runs, calibration=calibration)
        evidence = generation.manifest["pe8_evidence"]
        assert evidence["consulted"] is True
        assert evidence["identity"] == cb.calibration_identity(calibration)
        # PE-8's own declared terminal state travels with the reading (its VALUE is
        # PE-8's business; that it is CARRIED is certification's).
        assert evidence["terminal_state"] == cb.calibration_terminal_state(calibration)["state"]
        # Frozen PE-8 semantics: insufficiency is a STATE, never a refusal, and
        # nothing is promoted.
        assert generation.manifest["disclosure"]["resolution"] == "NOT_ATTEMPTED"
        assert cb.CONTINUOUS_PROXY_TIE_LIMITATION in generation.manifest["disclosure"]["carried"]
        assert generation.manifest["horizon_state"] == fg.DECISION_HORIZON_COMPLETE
    finally:
        conn.close()


def test_success_a_generation_without_pe8_evidence_still_certifies_and_says_so():
    conn, runs = _world()
    try:
        generation = _generation(conn, runs)
        evidence = generation.manifest["pe8_evidence"]
        assert evidence["consulted"] is False
        assert evidence["state"] == cb.STATE_EVIDENCE_LIMITED
        assert generation.manifest["horizon_state"] == fg.DECISION_HORIZON_COMPLETE
    finally:
        conn.close()


def test_success_blank_event_is_a_valid_zero_fixture_generation():
    conn, runs = _world(events=(5, 6, 7, 8), blanks=(6,))
    try:
        generation = _generation(conn, runs)
        assert generation.manifest["per_event"]["6"]["zero_fixture_event"] is True
        assert generation.manifest["horizon_state"] == fg.DECISION_HORIZON_COMPLETE
        assert gs.verify_generation(conn, generation.generation_id)["verified"] is True
    finally:
        conn.close()


def test_success_manager_world_generation_covers_one_event():
    conn, runs = _world(events=(5,))
    try:
        generation = _generation(
            conn, runs, events=(5,), horizon_kind=gs.HORIZON_KIND_MANAGER_WORLD,
        )
        assert generation.horizon_kind == gs.HORIZON_KIND_MANAGER_WORLD
        assert int(generation.manifest["horizon_length"]) == 1
        assert generation.manifest["horizon_state"] == fg.DECISION_HORIZON_COMPLETE
        built = mw.build_manager_worlds(
            conn, generation=generation, planning_event=5, squad_ids=[1, 2],
            simulations=4, occupancy_audit=False,
        )
        assert built["generation_id"] == generation.generation_id
        assert built["input_run_ids"]["xpts"] == runs[5]["xpts_v1"]
        # The run ids are READ from the generation: there is no parameter to substitute.
        with pytest.raises(TypeError):
            mw.build_manager_worlds(
                conn, generation=generation, planning_event=5, squad_ids=[1, 2], xpts_run_id=999,
            )
    finally:
        conn.close()


def test_success_cache_deletion_preserves_decision_correctness(tmp_path):
    conn, runs = _world()
    try:
        generation = _generation(conn, runs)
        union = [1, 2]
        config = _optimizer_config(search_draws=8)
        warm_dir = tmp_path / "warm"
        warm_dir.mkdir()
        cold_dir = tmp_path / "cold"
        cold_dir.mkdir()
        first, first_info = ro.build_event_worlds(
            conn, generation, 5, union, config, cache_dir=warm_dir,
        )
        assert first_info["source"] == "generated"
        second, second_info = ro.build_event_worlds(
            conn, generation, 5, union, config, cache_dir=warm_dir,
        )
        assert second_info["source"] == "cache"
        # The cache is optimization only: the SAME generation through a COLD cache
        # rebuilds identical worlds rather than changing the answer.
        third, third_info = ro.build_event_worlds(
            conn, generation, 5, union, config, cache_dir=cold_dir,
        )
        assert third_info["source"] == "generated"
        assert third["core"] == first["core"] == second["core"]
        assert third["minutes"] == first["minutes"]
    finally:
        conn.close()


def test_success_canonical_free_hit_route_world_load_consumes_a_generation(tmp_path):
    conn, runs = _world()
    try:
        generation = _generation(conn, runs)
        worlds = fha.load_certified_route_worlds(
            conn, generation, arm="SAVE", expected_events=(5,), union_ids=[1, 2], config=(
                _optimizer_config(search_draws=4)
            ), cache_dir=tmp_path,
        )
        assert set(worlds) == {5}
        assert worlds[5]["worlds"] == 4
        # Without a generation the load refuses rather than defaulting one.
        with pytest.raises(fha.FreeHitAdapterError):
            fha.load_certified_route_worlds(
                conn, None, arm="SAVE", expected_events=(5,), union_ids=[1, 2],
                config=_optimizer_config(), cache_dir=tmp_path,
            )
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# API surface
# ---------------------------------------------------------------------------


def test_api_surface_make_decision_is_descriptor_only():
    parameters = set(inspect.signature(gs.make_decision).parameters)
    assert {"manager_packet", "planning_event", "generation_id"} <= parameters
    for banned in ("bundles", "certification", "matrices", "runs", "cache_dir", "artifact",
                   "manifest", "worlds", "registry", "token"):
        assert banned not in parameters, banned


def test_api_surface_production_entrypoints_do_not_accept_predictive_descriptors():
    for function, banned in (
        (ro.build_event_worlds, {"bundles", "certification", "artifact", "matrices", "runs",
                                 "prebuilt_worlds", "registry", "token"}),
        (ro.optimize, {"bundles", "certification", "artifact", "matrices", "runs",
                       "prebuilt_worlds", "registry", "token"}),
        (rc.compare_routes, {"bundles", "certification", "artifact", "matrices", "runs"}),
        (mw.build_manager_worlds, {"minutes_run_id", "xpts_run_id", "team_run_id", "certification"}),
        (fr.refine_finalists, {"bundles", "certification", "prebuilt_worlds"}),
        (rs.run_ladder, {"bundles", "certification", "prebuilt_worlds"}),
    ):
        parameters = set(inspect.signature(function).parameters)
        for name in banned:
            assert name not in parameters, (function.__name__, name)
    for function in (ro.build_event_worlds, ro.optimize, mw.build_manager_worlds):
        assert "generation" in inspect.signature(function).parameters, function.__name__
    assert "generation" in inspect.signature(rc.compare_routes).parameters


def test_api_surface_the_removed_authority_mechanisms_are_gone():
    """Amendment 2 section 5: removed as authority, not repaired."""

    for name in ("IssuedWorldMatrix", "_ISSUED_WORLD_MATRICES", "_issue",
                 "require_certified_prebuilt_matrix", "issued_world_matrix",
                 "_issue_world_matrix", "_world_matrix_capability_authority"):
        assert not hasattr(ro, name), name
    for name in ("ValidatedCertificationArtifact", "validate_certification_artifact",
                 "assert_certification_artifact_bytes_unchanged",
                 "certified_bundle_artifact_record", "assert_event_bundle_certified"):
        assert not hasattr(cb, name), name
    assert not hasattr(mw, "MATRIX_CERTIFIED_BUNDLE_KEY")
    source = Path(ro.__file__).read_text(encoding="utf-8")
    for banned in ("_p2_certified_bundle_identity", "IssuedWorldMatrix",
                   "require_certified_prebuilt_matrix"):
        assert banned not in source, banned


def test_api_surface_the_replay_path_is_structurally_separate():
    conn, runs = _world()
    try:
        generation = _generation(conn, runs)
        before = rw.evidence_row_counts(conn)
        declaration = rw.ReplayDeclaration(
            token=rw.REPLAY_ONLY, owner="test_pe9", purpose="non-production research",
        )
        result = rw.replay_decision(conn, run=lambda: {"worlds": 1}, declaration=declaration)
        assert result["production"] is False
        assert result["certified"] is False
        assert result["generation_id"] is None
        rw.assert_replay_wrote_no_evidence(before, rw.evidence_row_counts(conn))
        assert gs.current_generation_id(conn, 5) == generation.generation_id

        def _writing_run():
            conn.execute("DELETE FROM current_generation")
            return {}

        with pytest.raises(rw.ReplayRefused) as caught:
            rw.replay_decision(conn, run=_writing_run, declaration=declaration)
        assert caught.value.token == rw.DIAG_REPLAY_WROTE_PE9_EVIDENCE

        with pytest.raises(rw.ReplayRefused):
            rw.ReplayWorlds(declaration=rw.NON_PRODUCTION_REPLAY_ONLY, matrices={5: {}})
        with pytest.raises(rw.ReplayRefused):
            rw.assert_not_production_generation(generation.generation_id)
    finally:
        conn.close()


def test_api_surface_canonical_manifest_refuses_an_unstable_identity_value():
    with pytest.raises(gs.GenerationManifestInvalid):
        gs.canonical_manifest_bytes({"f": 0.5})
    with pytest.raises(gs.GenerationManifestInvalid):
        gs.canonical_manifest_bytes({"s": {1, 2}})
    assert gs.generation_id_of(
        {"a": 1, "b": "x", "c": True, "d": None, "e": [1, 2], "f": {"g": 3}}
    ).startswith("sha256:")


def test_api_surface_manifest_identity_excludes_volatile_fields():
    conn, runs = _world()
    try:
        generation = _generation(conn, runs)
        semantic = gs.manifest_semantic_projection(generation.manifest)
        assert "created_at" not in semantic
        noisy = dict(generation.manifest, created_at="2030-01-01T00:00:00Z", lock_acquired="later")
        assert gs.generation_id_of(noisy) == generation.generation_id
        changed = json.loads(json.dumps(generation.manifest))
        changed["per_event"]["5"]["state"] = "EVIDENCE_MISSING"
        assert gs.generation_id_of(changed) != generation.generation_id
    finally:
        conn.close()
