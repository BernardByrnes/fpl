"""PE-9 — the production decision entrypoint and the certification gates.

These are REAL integration tests, not source assertions: a synthetic 15-player world
is certified through ``generation_store.certify_generation`` under the leased
``ExecutionController``, and the decision is taken through
``generation_store.make_decision`` against the PINNED snapshot that generation names.
What is asserted is what production does -- a persisted ``engine_decision_records``
row, a retained decision artifact beside the store, and a verifier that REPRODUCES the
manager packet, request and result digests instead of reporting them as verified.
"""

from __future__ import annotations

import inspect
import json
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import generation_fixtures as gf
from fpl_brain import certified_bundle as cb
from fpl_brain import execution
from fpl_brain import generation_store as gs
from fpl_brain import manager_worlds, route_comparator, repositories as repo
from fpl_brain.database import connect_database
from fpl_brain.planning import get_planning_context

#: A legal 15-player squad (2 GK, 5 DEF, 5 MID, 3 FWD), which is what a manager-world
#: decision consumes.  Manager state is a PERMITTED caller input; the predictive
#: evidence is not.
SQUAD = list(range(1, 16))
POSITIONS = (
    "GKP", "GKP", "DEF", "DEF", "DEF", "DEF", "DEF",
    "MID", "MID", "MID", "MID", "MID", "FWD", "FWD", "FWD",
)
_ELEMENT_TYPE = {"GKP": 1, "DEF": 2, "MID": 3, "FWD": 4}
FAMILIES = ("minutes_v1", "team_strength_v1", "player_rates_v1", "xpts_v1", "monte_carlo_v1")
RUN_IDS = {family: 100 + index for index, family in enumerate(FAMILIES)}
PROFILE = gs.DecisionProfile(
    kind=gs.HORIZON_KIND_MANAGER_WORLD,
    parameters={"simulations": 8, "top_k": 3, "occupancy_audit": False},
)


def _world(path: Path) -> gs.CertifiedGeneration:
    """A file-backed world carrying a certified MANAGER_WORLD generation."""

    conn = connect_database(path)
    gf.base_world(conn)
    with conn:
        for pid in range(3, 16):
            conn.execute(
                "INSERT INTO players(id, web_name, team_id, element_type, is_active, first_seen_at,"
                " last_seen_at, raw_json, updated_at)"
                " VALUES (?,?,1,3,1,'2026-09-01T00:00:00Z','2026-09-01T00:00:00Z','{}',"
                "'2026-09-01T00:00:00Z')",
                (pid, f"P{pid}"),
            )
        for pid, position in zip(SQUAD, POSITIONS):
            conn.execute(
                "UPDATE players SET element_type=? WHERE id=?", (_ELEMENT_TYPE[position], pid)
            )
    gf.add_event(conn, 5)
    with conn:
        gf.add_fixture(conn, 1000, 5, 1, 2)
    gf.prepare_fixture_snapshot(conn, (5,), cutoff=gf.CUTOFF)
    with conn:
        for family, run_id in RUN_IDS.items():
            gf.add_run(conn, run_id, family, 5)
        for pid in SQUAD:
            gf.add_xpts_row(
                conn, RUN_IDS["xpts_v1"], 1000, 5, minutes_run_id=RUN_IDS["minutes_v1"],
                team_run_id=RUN_IDS["team_strength_v1"], rate_run_id=RUN_IDS["player_rates_v1"],
                player_id=pid,
            )
            gf.add_mc_row(
                conn, RUN_IDS["monte_carlo_v1"], 1000, 5, xpts_run_id=RUN_IDS["xpts_v1"],
                minutes_run_id=RUN_IDS["minutes_v1"], team_run_id=RUN_IDS["team_strength_v1"],
                rate_run_id=RUN_IDS["player_rates_v1"], player_id=pid,
            )
    conn.commit()
    generation = gf.certify_world(
        conn, {5: dict(RUN_IDS)}, events=(5,),
        horizon_kind=gs.HORIZON_KIND_MANAGER_WORLD,
        # The decision READS the pinned snapshot, so the fixture pins a real copy of
        # the exact source state used to produce these runs.
        snapshot_path=path.parent / "snapshot.db",
    )
    conn.close()
    return generation


def _packet() -> dict:
    return {
        "entry_id": 1,
        "planning_event": 5,
        "cutoff": gf.CUTOFF,
        "squad_ids": list(SQUAD),
    }


def _decision(conn, generation_id=None, **over):
    arguments = {
        "horizon_kind": gs.HORIZON_KIND_MANAGER_WORLD,
        "profile": PROFILE,
        "request": {"tracing_id": "test-pe9"},
    }
    packet = over.pop("packet", None) or _packet()
    arguments.update(over)
    return gs.make_decision(conn, packet, 5, generation_id=generation_id, **arguments)


def _four_gw_manager_state_snapshot(
    tmp_path: Path, *, bank: int | None = 7, free_transfers: int | None = 1,
    event_start_free_transfers: int | None = 2,
):
    """Build a real manager snapshot and return its canonical four-GW route state."""

    import test_four_gw_decision as four_gw_fixtures

    path = tmp_path / "manager-source.db"
    conn = connect_database(path)
    four_gw_fixtures._seed_regression_manager(conn, entry_id=241392, event=4)
    repo.upsert_manual_manager_state(
        conn,
        241392,
        4,
        free_transfers,
        bank,
        captured_at=gf.CUTOFF,
        event_start_free_transfers=event_start_free_transfers,
    )
    conn.commit()
    snapshot = gf.write_snapshot(tmp_path / "manager-snapshot.db", database=conn)
    source = sqlite3.connect(f"file:{snapshot['path']}?mode=ro", uri=True)
    source.row_factory = sqlite3.Row
    try:
        context = get_planning_context(
            source, 241392, 4, as_of=gf.CUTOFF, season="2026/27"
        )
        squad = manager_worlds.resolve_squad(context, source)
        route_state = route_comparator.build_route_state(source, context, squad)
        canonical = gs.canonical_four_gw_manager_state(
            entry_id=241392,
            planning_event=4,
            cutoff=gf.CUTOFF,
            season="2026/27",
            context=context,
            squad=squad,
            route_state=route_state,
        )
    finally:
        source.close()
        conn.close()
    return snapshot, canonical


def _four_gw_manager_packet(canonical_state: dict, **state_overrides) -> dict:
    route = canonical_state["route_state"]
    state = {
        "squad_ids": [int(player["player_id"]) for player in route["players"]],
        "bank_tenths": int(route["bank_tenths"]),
        "free_transfers": int(route["free_transfers"]),
        "event_start_free_transfers": route["event_start_free_transfers"],
        "authoritative_source": canonical_state["source_manager_context"]["authoritative_source"],
    }
    state.update(state_overrides)
    return {
        "entry_id": int(canonical_state["entry_id"]),
        "planning_event": int(canonical_state["planning_event"]),
        "cutoff": str(canonical_state["cutoff"]),
        "season": canonical_state["season"],
        "manager_state": state,
    }


def _four_gw_certified_manager_world(
    path: Path, *, bank: int = 7, free_transfers: int = 0,
    event_start_free_transfers: int | None = None,
):
    """Create a four-event generation whose pinned snapshot has real manager state."""

    import test_four_gw_decision as four_gw_fixtures

    conn = connect_database(path)
    gf.base_world(conn)
    four_gw_fixtures._seed_regression_manager(conn, entry_id=241392, event=4)
    repo.upsert_manual_manager_state(
        conn,
        241392,
        4,
        free_transfers,
        bank,
        captured_at=gf.CUTOFF,
        event_start_free_transfers=event_start_free_transfers,
    )
    events = (4, 5, 6, 7)
    fixture_ids = {4: [41], 5: [42], 6: [43], 7: [44]}
    with conn:
        for event in events[1:]:
            gf.add_event(conn, event)
            gf.add_fixture(conn, fixture_ids[event][0], event, 1, 2)
        gf.prepare_fixture_snapshot(conn, events, cutoff=gf.CUTOFF)
        runs_by_event = {}
        next_run = 100
        for event in events:
            ids = {
                "minutes_v1": next_run,
                "team_strength_v1": next_run + 1,
                "player_rates_v1": next_run + 2,
                "xpts_v1": next_run + 3,
                "monte_carlo_v1": next_run + 4,
            }
            next_run += 5
            for family, run_id in ids.items():
                gf.add_run(conn, run_id, family, event, cutoff=gf.CUTOFF)
            for fixture_id in fixture_ids[event]:
                gf.add_xpts_row(
                    conn,
                    ids["xpts_v1"],
                    fixture_id,
                    event,
                    minutes_run_id=ids["minutes_v1"],
                    team_run_id=ids["team_strength_v1"],
                    rate_run_id=ids["player_rates_v1"],
                )
                gf.add_mc_row(
                    conn,
                    ids["monte_carlo_v1"],
                    fixture_id,
                    event,
                    xpts_run_id=ids["xpts_v1"],
                    minutes_run_id=ids["minutes_v1"],
                    team_run_id=ids["team_strength_v1"],
                    rate_run_id=ids["player_rates_v1"],
                )
            runs_by_event[event] = ids
    generation = gf.certify_world(
        conn,
        runs_by_event,
        events=events,
        planning_event=4,
        horizon_kind=gs.HORIZON_KIND_FOUR_GW,
        snapshot_path=path.parent / "four-gw-snapshot.db",
    )
    return conn, generation


@pytest.mark.parametrize("free_transfers", [0, 1, 2])
def test_canonical_four_gw_state_binds_zero_and_nonzero_bank_and_ft_values(
    tmp_path, free_transfers
):
    _snapshot, canonical = _four_gw_manager_state_snapshot(
        tmp_path,
        bank=0,
        free_transfers=free_transfers,
        event_start_free_transfers=free_transfers,
    )
    route = canonical["route_state"]
    assert route["bank_tenths"] == 0
    assert route["free_transfers"] == free_transfers
    assert route["event_start_free_transfers"] == free_transfers
    gs.assert_manager_packet_matches_canonical_state(
        _four_gw_manager_packet(canonical), canonical
    )
    assert gs.four_gw_manager_context_identity(canonical).startswith("sha256:")


def test_canonical_four_gw_state_refuses_packet_economic_mismatches_and_allows_omissions(
    tmp_path,
):
    _snapshot, canonical = _four_gw_manager_state_snapshot(
        tmp_path, bank=7, free_transfers=1, event_start_free_transfers=2
    )
    exact = _four_gw_manager_packet(canonical)
    gs.assert_manager_packet_matches_canonical_state(exact, canonical)
    # The caller may supply only the manager identity: canonical state is derived
    # from the pinned snapshot and retained separately by the production boundary.
    gs.assert_manager_packet_matches_canonical_state(
        {key: exact[key] for key in ("entry_id", "planning_event", "cutoff", "season")},
        canonical,
    )

    for mismatch in (
        {"bank_tenths": 8},
        {"free_transfers": 0},
        {"bank_tenths": 8, "free_transfers": 0},
        {"event_start_free_transfers": 1},
        {"squad_ids": [*range(1, 15), 99]},
    ):
        packet = _four_gw_manager_packet(canonical, **mismatch)
        with pytest.raises(gs.GenerationRefused) as refused:
            gs.assert_manager_packet_matches_canonical_state(packet, canonical)
        assert refused.value.token == gs.DIAG_MANAGER_STATE_MISMATCH

    # A present None is not treated as an omitted assertion or as numeric zero.
    packet = _four_gw_manager_packet(canonical, bank_tenths=None)
    with pytest.raises(gs.GenerationRefused) as missing_value:
        gs.assert_manager_packet_matches_canonical_state(packet, canonical)
    assert missing_value.value.token == gs.DIAG_MANAGER_STATE_MISMATCH

    # None is a legitimate event-start FT value when the authoritative snapshot
    # does not contain it; an omitted assertion remains safe because canonical
    # attribution is derived and retained independently.
    no_event_start_dir = tmp_path / "event-start-none"
    no_event_start_dir.mkdir()
    _snapshot, no_event_start = _four_gw_manager_state_snapshot(
        no_event_start_dir, bank=0, free_transfers=0, event_start_free_transfers=None
    )
    packet = _four_gw_manager_packet(no_event_start)
    gs.assert_manager_packet_matches_canonical_state(packet, no_event_start)
    packet["manager_state"].pop("event_start_free_transfers")
    gs.assert_manager_packet_matches_canonical_state(packet, no_event_start)
    packet["manager_state"]["event_start_free_transfers"] = 0
    with pytest.raises(gs.GenerationRefused) as unexpected_value:
        gs.assert_manager_packet_matches_canonical_state(packet, no_event_start)
    assert unexpected_value.value.token == gs.DIAG_MANAGER_STATE_MISMATCH


def test_canonical_four_gw_state_refuses_missing_snapshot_bank_or_free_transfers(tmp_path):
    with pytest.raises(gs.GenerationRefused) as missing:
        _four_gw_manager_state_snapshot(
            tmp_path, bank=None, free_transfers=None, event_start_free_transfers=None
        )
    assert missing.value.token == gs.DIAG_MANAGER_STATE_EVIDENCE_MISSING


def test_make_decision_executes_and_retains_the_snapshot_derived_manager_state(
    tmp_path, monkeypatch
):
    path = tmp_path / "four-gw.db"
    conn, generation = _four_gw_certified_manager_world(
        path, bank=0, free_transfers=0, event_start_free_transfers=0
    )
    captured = {}

    def declared_runner(**kwargs):
        state = kwargs["canonical_manager_state"]
        captured["state"] = state
        return {
            "decision": {"status": "CANONICAL_MANAGER_STATE_TEST"},
            "consumed_manager_state": state,
            "provenance": {
                "planning_context_hash": gs.four_gw_manager_context_identity(state)
            },
            "runner_identity": "test-declared-four-gw-runner",
            "artifact_blocks": {},
        }

    monkeypatch.setattr(gs, "_decision_executor", lambda _profile: declared_runner)
    try:
        packet = {
            "entry_id": 241392,
            "planning_event": 4,
            "cutoff": gf.CUTOFF,
            "season": "2026/27",
        }
        outcome = gs.make_decision(
            conn,
            packet,
            4,
            horizon_kind=gs.HORIZON_KIND_FOUR_GW,
            generation_id=generation.generation_id,
            profile=gs.DecisionProfile(kind=gs.HORIZON_KIND_FOUR_GW),
        )

        consumed = captured["state"]
        assert consumed["route_state"]["bank_tenths"] == 0
        assert consumed["route_state"]["free_transfers"] == 0
        assert consumed["route_state"]["event_start_free_transfers"] == 0
        assert outcome["artifact"]["attribution"]["manager_packet"] == packet
        assert outcome["artifact"]["attribution"]["consumed_manager_state"] == consumed
        assert outcome["manager_context_sha256"] == gs.four_gw_manager_context_identity(consumed)
        assert gs.verify_decision(conn, outcome["decision_record_id"])["verified"] is True
    finally:
        conn.close()


@pytest.mark.parametrize(
    "field_name", ["bank_tenths", "free_transfers", "event_start_free_transfers"]
)
def test_verify_decision_refuses_retained_manager_state_tampering(
    tmp_path, monkeypatch, field_name
):
    path = tmp_path / "four-gw.db"
    conn, generation = _four_gw_certified_manager_world(
        path, bank=0, free_transfers=0, event_start_free_transfers=0
    )

    def declared_runner(**kwargs):
        state = kwargs["canonical_manager_state"]
        return {
            "decision": {"status": "CANONICAL_MANAGER_STATE_TEST"},
            "consumed_manager_state": state,
            "provenance": {
                "planning_context_hash": gs.four_gw_manager_context_identity(state)
            },
            "runner_identity": "test-declared-four-gw-runner",
            "artifact_blocks": {},
        }

    monkeypatch.setattr(gs, "_decision_executor", lambda _profile: declared_runner)
    try:
        packet = {
            "entry_id": 241392,
            "planning_event": 4,
            "cutoff": gf.CUTOFF,
            "season": "2026/27",
        }
        outcome = gs.make_decision(
            conn,
            packet,
            4,
            horizon_kind=gs.HORIZON_KIND_FOUR_GW,
            generation_id=generation.generation_id,
            profile=gs.DecisionProfile(kind=gs.HORIZON_KIND_FOUR_GW),
        )
        artifact_path = Path(outcome["decision_artifact_ref"])
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        artifact["attribution"]["consumed_manager_state"]["route_state"][field_name] += 1
        artifact_path.write_text(
            json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )

        with pytest.raises(gs.GenerationRefused) as refused:
            gs.verify_decision(conn, outcome["decision_record_id"])
        assert refused.value.token == gs.DIAG_DECISION_RECORD_INVALID
        assert any(
            "retained as consumed differs from the state re-derived from the pinned snapshot" in item
            for item in refused.value.reasons
        )
    finally:
        conn.close()


def test_make_decision_refuses_manager_state_assertions_before_execution_or_persistence(
    tmp_path, monkeypatch
):
    path = tmp_path / "four-gw.db"
    conn, generation = _four_gw_certified_manager_world(
        path, bank=0, free_transfers=0, event_start_free_transfers=0
    )
    called = []

    def should_not_run(**_kwargs):
        called.append(True)
        raise AssertionError("decision executor ran for mismatched manager state")

    monkeypatch.setattr(gs, "_decision_executor", lambda _profile: should_not_run)
    try:
        base_packet = {
            "entry_id": 241392,
            "planning_event": 4,
            "cutoff": gf.CUTOFF,
            "season": "2026/27",
        }
        for assertions in (
            {"bank_tenths": 1},
            {"free_transfers": 1},
            {"bank_tenths": 1, "free_transfers": 1},
            {"event_start_free_transfers": 1},
            {"bank_tenths": None},
        ):
            packet = {**base_packet, "manager_state": assertions}
            with pytest.raises(gs.GenerationRefused) as refused:
                gs.make_decision(
                    conn,
                    packet,
                    4,
                    horizon_kind=gs.HORIZON_KIND_FOUR_GW,
                    generation_id=generation.generation_id,
                    profile=gs.DecisionProfile(kind=gs.HORIZON_KIND_FOUR_GW),
                )
            assert refused.value.token == gs.DIAG_MANAGER_STATE_MISMATCH
        with pytest.raises(gs.ProductionDescriptorOnly):
            gs.make_decision(
                conn,
                {**base_packet, "manager_state": None},
                4,
                horizon_kind=gs.HORIZON_KIND_FOUR_GW,
                generation_id=generation.generation_id,
                profile=gs.DecisionProfile(kind=gs.HORIZON_KIND_FOUR_GW),
            )
        assert called == []
        assert conn.execute("SELECT COUNT(*) FROM engine_decision_records").fetchone()[0] == 0
    finally:
        conn.close()


def test_four_gw_production_runner_passes_the_pinned_route_state_to_optimizer(
    tmp_path, monkeypatch
):
    import test_four_gw_decision as four_gw_fixtures
    import fpl_brain.route_comparator as route_comparator

    path = tmp_path / "four-gw.db"
    conn, generation = _four_gw_certified_manager_world(
        path, bank=0, free_transfers=0, event_start_free_transfers=0
    )
    runner = gs._declared_production_entrypoint(
        gs.PRODUCTION_DECISION_MODULES[gs.HORIZON_KIND_FOUR_GW]
    )
    runner_globals = runner.__globals__
    captured = {}

    def stop_at_optimizer(**kwargs):
        captured["initial_state"] = kwargs["initial_state"]
        raise RuntimeError("test reached the real optimizer boundary")

    monkeypatch.setattr(runner_globals["cu"], "price_snapshot_as_of", lambda *a, **k: object())
    monkeypatch.setattr(route_comparator, "flat_current_price_scenario", lambda *a, **k: object())
    monkeypatch.setattr(runner_globals["cu"], "load_pool", lambda *a, **k: {"players": []})
    monkeypatch.setattr(
        runner_globals["provenance"],
        "assert_official_pool_identity",
        lambda **_k: {
            "official_generation_id": "test-generation",
            "snapshot_pool_count": 15,
            "snapshot_pool_ids_sha256": "sha256:" + "a" * 64,
            "official_pool_identity_match": True,
        },
    )
    monkeypatch.setattr(runner_globals["cu"], "load_fixtures_by_team", lambda *a, **k: {})
    monkeypatch.setattr(runner_globals["cu"], "load_projection_rows", lambda *a, **k: [])
    monkeypatch.setattr(
        runner_globals["cu"],
        "build_universe",
        lambda **_k: {"universe": [{"player_id": 1}], "excluded": []},
    )
    monkeypatch.setattr(runner_globals["cu"], "build_replacement_edges", lambda **_k: [])
    monkeypatch.setattr(runner_globals["fg"], "screen_legal_actions", lambda **_k: {})
    monkeypatch.setattr(runner_globals["cu"], "discovery_completeness", lambda **_k: {})
    monkeypatch.setattr(runner_globals["ro"], "optimize", stop_at_optimizer)

    source_conn = gs._open_generation_snapshot(generation)
    packet = {
        "entry_id": 241392,
        "planning_event": 4,
        "cutoff": gf.CUTOFF,
        "season": "2026/27",
    }
    canonical = gs._derive_four_gw_consumed_manager_state(
        source_conn, generation=generation, manager_packet=packet
    )
    try:
        with pytest.raises(RuntimeError, match="real optimizer boundary"):
            runner(
                conn=conn,
                generation=generation,
                manager_packet=packet,
                parameters={"stage2_draws": runner_globals["STAGE1_DRAWS"] + 1},
                source_conn=source_conn,
                canonical_manager_state=canonical,
            )
        initial_state = captured["initial_state"]
        expected = canonical["route_state"]
        assert initial_state.bank_tenths == expected["bank_tenths"] == 0
        assert initial_state.free_transfers == expected["free_transfers"] == 0
        assert initial_state.event_start_free_transfers == expected["event_start_free_transfers"] == 0
        assert [player.as_dict() for player in initial_state.players] == expected["players"]
    finally:
        source_conn.close()
        conn.close()


# ---------------------------------------------------------------------------
# Real decisions, through the canonical entrypoint
# ---------------------------------------------------------------------------


def test_make_decision_resolves_the_current_generation_and_records_the_decision(tmp_path):
    path = tmp_path / "fpl.db"
    generation = _world(path)
    conn = connect_database(path)
    try:
        outcome = _decision(conn)
        assert outcome["generation_id"] == generation.generation_id
        assert outcome["decision"]["status"] == "MANAGER_WORLD_EVALUATED"

        # The record is PERSISTED, not narrated.
        row = conn.execute(
            "SELECT * FROM engine_decision_records WHERE decision_id=?",
            (outcome["decision_record_id"],),
        ).fetchone()
        assert row is not None
        assert str(row["generation_id"]) == generation.generation_id
        assert str(row["result_sha256"]) == outcome["result_sha256"]

        # The artifact is RETAINED beside the authoritative store, and it is named by
        # the decision's own content digest.
        artifact_path = Path(outcome["decision_artifact_ref"])
        assert artifact_path.exists()
        assert artifact_path.parent.name == "gw05"
        assert outcome["result_sha256"].split(":", 1)[1][:32] in artifact_path.name

        # The verifier REPRODUCES every digest from that artifact.
        report = gs.verify_decision(conn, outcome["decision_record_id"])
        assert report["verified"] is True
        assert report["generation_verified"] is True
        assert report["manager_packet_digest_verified"] is True
        assert report["request_digest_verified"] is True
        assert report["result_digest_verified"] is True
        assert report["decision_artifact_sha256"]

        # The predictive inputs came from the GENERATION and its pinned snapshot.
        provenance = outcome["provenance"]
        assert provenance["generation_id"] == generation.generation_id
        assert provenance["snapshot"]["sha256"] == generation.snapshot["sha256"]
        assert outcome["artifact"]["attribution"]["manager_packet"] == _packet()
    finally:
        conn.close()


def test_make_decision_is_idempotent_and_never_rewrites_history(tmp_path):
    path = tmp_path / "fpl.db"
    historical = _world(path)
    conn = connect_database(path)
    try:
        first = _decision(conn)
        second = _decision(conn)
        assert first["decision_record_id"] == second["decision_record_id"]
        assert conn.execute("SELECT COUNT(*) FROM engine_decision_records").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM generation").fetchone()[0] == 1
    finally:
        conn.close()


def test_repeat_decisions_retain_each_execution_s_artifact_and_both_still_verify(tmp_path):
    """A repeated decision keeps BOTH executions' evidence, and both records verify.

    This is the shape production actually has: re-running the same decision over the
    same certified generation reproduces the same RESULT but not the same artifact
    bytes, so a result-keyed retention path made the later run overwrite the evidence
    the earlier record still bound.  Each execution must keep its own immutable
    artifact, and the earlier record must still verify after the repeat.
    """

    path = tmp_path / "fpl.db"
    _world(path)
    conn = connect_database(path)
    try:
        first = _decision(conn, request={"tracing_id": "acceptance-primary"})
        first_path = Path(first["decision_artifact_ref"])
        first_bytes = first_path.read_bytes()

        second = _decision(conn, request={"tracing_id": "acceptance-repeat"})
        second_path = Path(second["decision_artifact_ref"])

        # Same decision, different execution: two records, two retained artifacts.
        assert second["result_sha256"] == first["result_sha256"]
        assert second["decision_record_id"] != first["decision_record_id"]
        assert second_path != first_path
        assert second_path.exists()
        assert second_path.read_bytes() != first_bytes
        assert first_path.read_bytes() == first_bytes, "the earlier artifact was overwritten"

        # The record the repeat used to destroy still verifies, and so does the repeat.
        assert gs.verify_decision(conn, first["decision_record_id"])["verified"] is True
        assert gs.verify_decision(conn, second["decision_record_id"])["verified"] is True

        # Neither retention left a publication temporary behind.
        assert not [item.name for item in first_path.parent.glob("*.tmp-*")]
    finally:
        conn.close()


def test_tampering_with_one_repeat_artifact_fails_only_that_record(tmp_path):
    """Tampering is detected per RECORD: its own artifact breaks, the sibling does not."""

    path = tmp_path / "fpl.db"
    _world(path)
    conn = connect_database(path)
    try:
        first = _decision(conn, request={"tracing_id": "acceptance-primary"})
        second = _decision(conn, request={"tracing_id": "acceptance-repeat"})
        second_path = Path(second["decision_artifact_ref"])
        retained = second_path.read_bytes()

        # Still valid JSON, different bytes: the digest binding is what must catch it.
        second_path.write_bytes(retained + b"  ")
        with pytest.raises(gs.GenerationRefused) as tampered:
            gs.verify_decision(conn, second["decision_record_id"])
        assert "does not match the decision record" in str(tampered.value)

        # The sibling record is untouched by tampering with the other artifact.
        assert gs.verify_decision(conn, first["decision_record_id"])["verified"] is True

        # Restoring the retained bytes restores verification: nothing else was relied on.
        second_path.write_bytes(retained)
        assert gs.verify_decision(conn, second["decision_record_id"])["verified"] is True

        # The same tampering against the FIRST record fails for that record in turn.
        first_path = Path(first["decision_artifact_ref"])
        first_bytes = first_path.read_bytes()
        first_path.write_bytes(first_bytes + b"  ")
        with pytest.raises(gs.GenerationRefused) as tampered_first:
            gs.verify_decision(conn, first["decision_record_id"])
        assert "does not match the decision record" in str(tampered_first.value)
        assert gs.verify_decision(conn, second["decision_record_id"])["verified"] is True
    finally:
        conn.close()


def test_a_retained_decision_artifact_is_never_replaced(tmp_path):
    """An occupied retention path is refused, not overwritten."""

    path = tmp_path / "fpl.db"
    _world(path)
    conn = connect_database(path)
    try:
        first = _decision(conn)
        artifact_path = Path(first["decision_artifact_ref"])
        foreign = b'{"schema": "not-this-decision"}\n'
        artifact_path.write_bytes(foreign)

        # The identical decision resolves to the SAME content-addressed path, where those
        # bytes are occupied by something else: it may not be replaced by a write.
        with pytest.raises(gs.DecisionRecordInvalid) as occupied:
            _decision(conn)
        assert "never replaced" in str(occupied.value)
        assert artifact_path.read_bytes() == foreign
        assert conn.execute("SELECT COUNT(*) FROM engine_decision_records").fetchone()[0] == 1
        assert not [item.name for item in artifact_path.parent.glob("*.tmp-*")]
    finally:
        conn.close()


def test_a_staging_failure_leaves_no_artifact_no_temporary_and_no_record(tmp_path, monkeypatch):
    """Staging runs under cleanup protection: an fsync failure leaves nothing behind."""

    import errno
    import os as os_module

    path = tmp_path / "fpl.db"
    _world(path)
    conn = connect_database(path)
    try:
        def no_space(fd):
            raise OSError(errno.ENOSPC, "simulated: no space left on device")

        monkeypatch.setattr(os_module, "fsync", no_space)
        with pytest.raises(gs.DecisionRecordInvalid) as staged:
            _decision(conn)
        assert "could not stage a decision artifact" in str(staged.value)

        artifact_dir = tmp_path / "pe9_decisions" / "gw05"
        assert artifact_dir.is_dir()
        assert list(artifact_dir.iterdir()) == [], "a partial artifact or temporary survived"
        assert conn.execute("SELECT COUNT(*) FROM engine_decision_records").fetchone()[0] == 0
    finally:
        conn.close()


def test_a_filesystem_without_hard_links_refuses_instead_of_streaming(tmp_path, monkeypatch):
    """No atomic no-replace publication => refuse; the final path is never written in place."""

    import errno
    import os as os_module

    path = tmp_path / "fpl.db"
    _world(path)
    conn = connect_database(path)
    try:
        def no_links(source, destination):
            raise OSError(errno.ENOSYS, "simulated: hard links unavailable")

        monkeypatch.setattr(os_module, "link", no_links)
        with pytest.raises(gs.DecisionRecordInvalid) as refused:
            _decision(conn)
        assert "atomically" in str(refused.value)

        artifact_dir = tmp_path / "pe9_decisions" / "gw05"
        assert artifact_dir.is_dir()
        assert list(artifact_dir.iterdir()) == [], "the final path or a temporary was created"
        assert conn.execute("SELECT COUNT(*) FROM engine_decision_records").fetchone()[0] == 0
    finally:
        conn.close()


def test_make_decision_pins_an_explicit_historical_generation(tmp_path):
    """``generation_id`` is a SELECTOR: a historical decision re-derives from ITS row."""

    path = tmp_path / "fpl.db"
    historical = _world(path)
    conn = connect_database(path)
    try:
        # Reusing these rows against a newly copied, different database must fail:
        # their immutable provenance names the original production snapshot.
        with pytest.raises(gs.GenerationRefused) as mismatched_snapshot:
            gf.certify_world(
                conn, {5: dict(RUN_IDS)}, events=(5,),
                horizon_kind=gs.HORIZON_KIND_MANAGER_WORLD,
                snapshot_path=tmp_path / "second.db", snapshot_database=path,
            )
        assert mismatched_snapshot.value.token == cb.STATE_EVIDENCE_MISSING
        assert gs.current_generation_id(conn, 5, gs.HORIZON_KIND_MANAGER_WORLD) == historical.generation_id

        pinned = _decision(conn, historical.generation_id)
        assert pinned["generation_id"] == historical.generation_id
        assert pinned["provenance"]["snapshot"]["sha256"] == historical.snapshot["sha256"]
        assert gs.verify_decision(conn, pinned["decision_record_id"])["generation_verified"] is True

        # The current selector still names the original, verified generation.
        current = _decision(conn)
        assert current["generation_id"] == historical.generation_id
        assert current["decision_record_id"] == pinned["decision_record_id"]
        assert conn.execute("SELECT COUNT(*) FROM engine_decision_records").fetchone()[0] == 1
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# The production API cannot be handed decision logic, evidence or handles
# ---------------------------------------------------------------------------


def test_make_decision_has_no_executor_parameter():
    parameters = inspect.signature(gs.make_decision).parameters
    assert "decide" not in parameters
    assert "executor" not in parameters
    assert "profile" in parameters
    # An executor cannot be smuggled in through the descriptor door either.
    conn = connect_database(":memory:")
    try:
        with pytest.raises(gs.ProductionDescriptorOnly):
            gs.make_decision(conn, _packet(), 5, decide=lambda **kwargs: {"decision": {}})
    finally:
        conn.close()


def test_a_declared_profile_kind_is_required_and_selects_the_pipeline(tmp_path):
    path = tmp_path / "fpl.db"
    _world(path)
    conn = connect_database(path)
    try:
        with pytest.raises(gs.GenerationRefused) as unknown:
            _decision(conn, profile=gs.DecisionProfile(kind="ARBITRARY_PIPELINE"))
        assert unknown.value.token == gs.DIAG_PRODUCTION_PROFILE_UNKNOWN
        # The declared production module really is the runner, resolved by NAME.
        assert gs.PRODUCTION_DECISION_MODULES[gs.HORIZON_KIND_FOUR_GW] == (
            "scripts/run_four_gw_decision.py"
        )
        entry = Path(gs.PRODUCTION_DECISION_MODULES[gs.HORIZON_KIND_FOUR_GW])
        assert gs.PRODUCTION_DECISION_ENTRYPOINT in (
            Path(__file__).resolve().parents[1] / entry
        ).read_text(encoding="utf-8")
        # And the library really resolves it, with exactly the declared interface:
        # the certified generation, the PINNED source, the manager packet and the
        # ordinary parameters -- no callable, and nothing predictive.
        resolved = gs._declared_production_entrypoint(
            gs.PRODUCTION_DECISION_MODULES[gs.HORIZON_KIND_FOUR_GW]
        )
        assert callable(resolved)
        assert set(inspect.signature(resolved).parameters) == {
            "conn", "generation", "manager_packet", "parameters", "source_conn",
            "canonical_manager_state", "controller",
        }
    finally:
        conn.close()


def test_make_decision_refuses_a_nested_cache_handle(tmp_path):
    path = tmp_path / "fpl.db"
    _world(path)
    conn = connect_database(path)
    try:
        with pytest.raises(gs.ProductionDescriptorOnly) as caught:
            _decision(conn, request={"tracing_id": "x", "cache_dir": str(tmp_path / "cache")})
        assert "request.cache_dir" in str(caught.value)
        with pytest.raises(gs.ProductionDescriptorOnly) as nested:
            _decision(conn, packet={**_packet(), "cache": {"dir": str(tmp_path)}})
        assert "manager_packet.cache" in str(nested.value)
        with pytest.raises(gs.ProductionDescriptorOnly) as profile:
            _decision(
                conn,
                profile={"kind": gs.HORIZON_KIND_MANAGER_WORLD,
                         "parameters": {"cache_dir": str(tmp_path)}},
            )
        assert "profile.parameters.cache_dir" in str(profile.value)
    finally:
        conn.close()


def test_make_decision_refuses_a_nested_run_mapping_or_matrix(tmp_path):
    path = tmp_path / "fpl.db"
    _world(path)
    conn = connect_database(path)
    try:
        for payload, where in (
            ({"runs": {5: dict(RUN_IDS)}}, "manager_packet.runs"),
            ({"matrix": {"worlds": 1}}, "manager_packet.matrix"),
            ({"bundles": {"5": {}}}, "manager_packet.bundles"),
            ({"certification": {"schema": "x"}}, "manager_packet.certification"),
            ({"manifest_digest": "sha256:" + "0" * 64}, "manager_packet.manifest_digest"),
        ):
            with pytest.raises(gs.ProductionDescriptorOnly) as caught:
                _decision(conn, packet={**_packet(), **payload})
            assert where in str(caught.value), where
        with pytest.raises(gs.ProductionDescriptorOnly) as nested:
            _decision(conn, request={"tracing_id": "x", "worlds": {5: {"worlds": 1}}})
        assert "request.worlds" in str(nested.value)
    finally:
        conn.close()


def test_make_decision_refuses_an_unusable_manager_packet_or_cutoff(tmp_path):
    path = tmp_path / "fpl.db"
    generation = _world(path)
    conn = connect_database(path)
    try:
        with pytest.raises(gs.DecisionRecordInvalid):
            _decision(conn, packet={"planning_event": 5, "cutoff": gf.CUTOFF})
        with pytest.raises(gs.GenerationRefused):
            _decision(
                conn,
                packet={**_packet(), "cutoff": gf.OTHER_CUTOFF},
            )
        # An unknown generation id, and an unset pointer, are different facts.
        with pytest.raises(gs.GenerationUnknown):
            _decision(conn, "sha256:" + "0" * 64)
        conn.execute("DELETE FROM current_generation")
        with pytest.raises(gs.GenerationRefused) as pointerless:
            _decision(conn)
        assert pointerless.value.token == gs.DIAG_GENERATION_POINTER_UNSET
        assert gs.current_generation_id(conn, 5, gs.HORIZON_KIND_MANAGER_WORLD) is None
        assert generation.generation_id
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Certification gates: the snapshot, the code identity, the planning context
# ---------------------------------------------------------------------------


def _certify(conn, *, snapshot=None, runs=None, **over):
    arguments = {
        "planning_event": 5,
        "cutoff": gf.CUTOFF,
        "runs_by_event": {5: dict(runs or RUN_IDS)},
        "events": (5,),
        "horizon_kind": gs.HORIZON_KIND_MANAGER_WORLD,
        "snapshot": snapshot,
    }
    arguments.update(over)
    if "controller" in arguments:
        return gs.certify_generation(conn, **arguments)
    return gf.certify_under_writer_lease(conn, **arguments)


def test_certification_requires_a_pinned_snapshot(tmp_path):
    path = tmp_path / "fpl.db"
    historical = _world(path)
    conn = connect_database(path)
    try:
        before = conn.execute("SELECT COUNT(*) FROM generation").fetchone()[0]
        with pytest.raises(gs.GenerationSnapshotUnverified):
            _certify(conn, snapshot=None)
        with pytest.raises(gs.GenerationSnapshotUnverified) as missing:
            _certify(conn, snapshot={"path": str(tmp_path / "absent.db"),
                                     "sha256": "sha256:" + "a" * 64,
                                     "source_db_identity": {"schema_version": "18"}})
        assert "does not exist" in str(missing.value)
        real = gf.write_snapshot(tmp_path / "live.db", database=path)
        with pytest.raises(gs.GenerationSnapshotUnverified) as mutated:
            _certify(conn, snapshot={**real, "sha256": "sha256:" + "b" * 64})
        assert "hashes to" in str(mutated.value)
        with pytest.raises(gs.GenerationSnapshotUnverified) as identityless:
            _certify(conn, snapshot={**real, "source_db_identity": None})
        assert "source database identity" in str(identityless.value)
        mismatched_identity = {
            **real,
            "source_db_identity": {**real["source_db_identity"], "projection_runs_count": 999},
        }
        with pytest.raises(gs.GenerationSnapshotUnverified) as mismatched:
            _certify(conn, snapshot=mismatched_identity)
        assert "does not reproduce from the pinned snapshot" in str(mismatched.value)
        # Nothing was written by any of those refusals.
        assert conn.execute("SELECT COUNT(*) FROM generation").fetchone()[0] == before
        # And the same world certifies once a real identity is pinned.
        assert _certify(conn, snapshot=historical.snapshot).generation_id.startswith("sha256:")
    finally:
        conn.close()


def test_certification_refuses_missing_required_pe8_evidence(tmp_path):
    path = tmp_path / "fpl.db"
    _world(path)
    conn = connect_database(path)
    try:
        before = conn.execute("SELECT COUNT(*) FROM generation").fetchone()[0]
        snapshot = gf.write_snapshot(tmp_path / "snapshot.db", database=path)
        with pytest.raises(gs.GenerationRefused) as missing:
            _certify(conn, snapshot=snapshot, require_calibration=True)
        assert missing.value.token == cb.STATE_EVIDENCE_MISSING
        assert conn.execute("SELECT COUNT(*) FROM generation").fetchone()[0] == before
    finally:
        conn.close()


def test_certification_requires_the_authoritative_code_identity(tmp_path):
    """The runs must have been produced by the RUNNING code revision."""

    conn = connect_database(":memory:")
    try:
        gf.base_world(conn)
        gf.add_event(conn, 5)
        with conn:
            for family, run_id in RUN_IDS.items():
                gf.add_run(conn, run_id, family, 5, code_snapshot="some-other-revision")
        # xPts rows are not needed: the structural gate refuses the code identity first.
        with pytest.raises((gs.GenerationNotCertified, gs.GenerationRefused)) as caught:
            _certify(conn, snapshot=gf.write_snapshot(tmp_path / "s.db", database=conn))
        assert "code snapshot" in str(caught.value)
        assert conn.execute("SELECT COUNT(*) FROM generation").fetchone()[0] == 0
    finally:
        conn.close()


def test_certification_requires_a_planning_context(tmp_path):
    conn = connect_database(":memory:")
    try:
        gf.base_world(conn)
        gf.add_event(conn, 5)
        gf.prepare_fixture_snapshot(conn, (5,), cutoff=gf.CUTOFF)
        with conn:
            for family, run_id in RUN_IDS.items():
                gf.add_run(conn, run_id, family, 5, context_hash=None)
        with pytest.raises(gs.GenerationRefused) as caught:
            _certify(conn, snapshot=gf.fixture_snapshot(conn))
        assert caught.value.token == gs.DIAG_GENERATION_PLANNING_CONTEXT_REQUIRED
        assert conn.execute("SELECT COUNT(*) FROM generation").fetchone()[0] == 0
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Leases and concurrency
# ---------------------------------------------------------------------------


def test_certification_runs_under_the_leased_controller(tmp_path):
    path = tmp_path / "fpl.db"
    _world(path)
    conn = connect_database(path)
    try:
        before = conn.execute("SELECT COUNT(*) FROM generation").fetchone()[0]
        controller = execution.ExecutionController(conn)
        controller.create_run(
            planning_event=5,
            planning_cutoff=gf.CUTOFF,
            hard_stop_at=execution.add_seconds(controller.now_dt(), 3600),
            label="pe9-test",
            families=["certification"],
        )
        controller.start()
        current_id = gs.current_generation_id(conn, 5, gs.HORIZON_KIND_MANAGER_WORLD)
        snapshot = gs.load_generation(conn, current_id).snapshot
        with pytest.raises(execution.LeaseError) as unleased:
            _certify(conn, snapshot=snapshot, controller=controller)
        assert "CERTIFICATION_WITHOUT_WRITER_LEASE" in str(unleased.value)
        assert conn.execute("SELECT COUNT(*) FROM generation").fetchone()[0] == before

        controller.acquire_run_lease()
        writer_lease = controller.acquire_writer_lease()
        with controller.stage("CERTIFY_GENERATION", detail={"events": [5]}):
            generation = _certify(conn, snapshot=snapshot, controller=controller)
        assert generation.generation_id.startswith("sha256:")
        # The run's own ledger records the certification it performed.
        stages = [dict(row)["stage"] if not isinstance(row, dict) else row["stage"]
                  for row in controller.stages()]
        assert "CERTIFY_GENERATION" in stages
        assert controller.active_lease(execution.LEASE_KIND_WRITER, "sqlite-writer")["id"] == writer_lease
        controller.finish(execution.RUN_COMPLETE)
    finally:
        conn.close()


def test_generation_publication_requires_a_live_lease_for_the_same_database():
    conn, runs = gf.synthetic_world()
    other = connect_database(":memory:")
    controllers = []
    try:
        snapshot = gf.fixture_snapshot(conn)
        kwargs = {
            "planning_event": 5, "cutoff": gf.CUTOFF, "runs_by_event": runs,
            "events": gf.HORIZON, "snapshot": snapshot,
        }
        with pytest.raises(execution.LeaseError, match="CERTIFICATION_WITHOUT_WRITER_LEASE"):
            gs.certify_generation(conn, **kwargs)
        assert conn.execute("SELECT COUNT(*) FROM generation").fetchone()[0] == 0
        assert gs.current_generation_id(conn, 5) is None

        foreign = execution.ExecutionController(other)
        foreign.create_run(
            planning_event=5, planning_cutoff=gf.CUTOFF,
            hard_stop_at=execution.add_seconds(foreign.now_dt(), 3600), label="foreign-store",
        )
        foreign.start()
        foreign.acquire_writer_lease()
        controllers.append(foreign)
        with pytest.raises(execution.LeaseError, match="same SQLite database"):
            gs.certify_generation(conn, controller=foreign, **kwargs)
        assert conn.execute("SELECT COUNT(*) FROM generation").fetchone()[0] == 0

        owner = execution.ExecutionController(conn)
        owner.create_run(
            planning_event=5, planning_cutoff=gf.CUTOFF,
            hard_stop_at=execution.add_seconds(owner.now_dt(), 3600), label="lease-owner",
        )
        owner.start()
        owner.acquire_writer_lease()
        controllers.append(owner)
        wrong_run = execution.ExecutionController(conn)
        wrong_run.create_run(
            planning_event=5, planning_cutoff=gf.CUTOFF,
            hard_stop_at=execution.add_seconds(wrong_run.now_dt(), 3600), label="wrong-run",
        )
        wrong_run.start()
        controllers.append(wrong_run)
        with pytest.raises(execution.LeaseError, match="belongs to another run"):
            gs.certify_generation(conn, controller=wrong_run, **kwargs)
        assert conn.execute("SELECT COUNT(*) FROM generation").fetchone()[0] == 0
        owner.finish(execution.RUN_COMPLETE)
        wrong_run.finish(execution.RUN_COMPLETE)
        controllers.remove(owner)
        controllers.remove(wrong_run)

        now = [datetime.now(timezone.utc)]
        expired = execution.ExecutionController(conn, wall_clock=lambda: now[0])
        expired.create_run(
            planning_event=5, planning_cutoff=gf.CUTOFF,
            hard_stop_at=now[0] + timedelta(hours=1), label="expired-lease",
        )
        expired.start()
        expired.acquire_writer_lease(ttl_seconds=1)
        now[0] += timedelta(seconds=2)
        controllers.append(expired)
        with pytest.raises(execution.LeaseError, match="expired"):
            gs.certify_generation(conn, controller=expired, **kwargs)
        assert conn.execute("SELECT COUNT(*) FROM generation").fetchone()[0] == 0
        assert gs.current_generation_id(conn, 5) is None
    finally:
        for controller in reversed(controllers):
            controller.finish(execution.RUN_COMPLETE)
        other.close()
        conn.close()


def test_concurrent_certification_serialises_on_one_generation(tmp_path):
    """Two certifiers of the SAME semantic world resolve to ONE generation row."""

    path = tmp_path / "fpl.db"
    generation = _world(path)
    snapshot = generation.snapshot
    baseline = connect_database(path)
    before = baseline.execute("SELECT COUNT(*) FROM generation").fetchone()[0]
    baseline.close()
    barrier = threading.Barrier(2)
    results: list[str] = []
    failures: list[BaseException] = []
    owner_conn = connect_database(path)
    owner = execution.ExecutionController(owner_conn)
    owner.create_run(
        planning_event=5, planning_cutoff=gf.CUTOFF,
        hard_stop_at=execution.add_seconds(owner.now_dt(), 3600), label="pe9-race",
    )
    owner.start()
    owner.acquire_writer_lease()

    def certify():
        conn = connect_database(path)
        try:
            controller = execution.ExecutionController(
                conn, run_uuid=owner.run_uuid, pid=owner.pid, host=owner.host,
            )
            barrier.wait(timeout=30)
            results.append(_certify(conn, snapshot=snapshot, controller=controller).generation_id)
        except BaseException as failure:  # noqa: BLE001 - reported below
            failures.append(failure)
        finally:
            conn.close()

    threads = [threading.Thread(target=certify) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)
    try:
        assert not failures, failures
        assert len(set(results)) == 1
        conn = connect_database(path)
        try:
            assert conn.execute("SELECT COUNT(*) FROM generation").fetchone()[0] == before
            # Both certifiers resolved the SAME semantic generation, and the pointer
            # names it: contention cannot produce two rows for one world.
            assert gs.current_generation_id(
                conn, 5, gs.HORIZON_KIND_MANAGER_WORLD
            ) == results[0]
        finally:
            conn.close()
    finally:
        owner.finish(execution.RUN_COMPLETE)
        owner_conn.close()


# ---------------------------------------------------------------------------
# Verification tells the truth, or refuses
# ---------------------------------------------------------------------------


def test_verify_decision_refuses_without_a_retained_artifact(tmp_path):
    path = tmp_path / "fpl.db"
    generation = _world(path)
    conn = connect_database(path)
    try:
        outcome = _decision(conn)
        record = conn.execute(
            "SELECT * FROM engine_decision_records WHERE decision_id=?",
            (outcome["decision_record_id"],),
        ).fetchone()
        assert record is not None

        # A record that names no artifact cannot have its manager packet, request or
        # result re-derived, and is refused rather than reported as verified.
        bare = gs.append_engine_decision_record(
            conn, generation=generation,
            manager_packet_sha256="sha256:" + "a" * 64,
            request_sha256="sha256:" + "b" * 64,
            result_sha256="sha256:" + "c" * 64,
            runner_identity="test", evidence={"schema": gs.DECISION_RECORD_SCHEMA},
        )
        conn.commit()
        with pytest.raises(gs.GenerationRefused) as unreferenced:
            gs.verify_decision(conn, bare)
        assert unreferenced.value.token == gs.DIAG_DECISION_ARTIFACT_NOT_RETAINED

        # A record whose artifact was DISCARDED is refused the same way.
        Path(outcome["decision_artifact_ref"]).unlink()
        with pytest.raises(gs.GenerationRefused) as discarded:
            gs.verify_decision(conn, outcome["decision_record_id"])
        assert discarded.value.token == gs.DIAG_DECISION_ARTIFACT_NOT_RETAINED
    finally:
        conn.close()


def test_verify_decision_refuses_an_artifact_without_attribution_or_with_a_mutated_one(tmp_path):
    path = tmp_path / "fpl.db"
    _world(path)
    conn = connect_database(path)
    try:
        outcome = _decision(conn)
        artifact_path = Path(outcome["decision_artifact_ref"])
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        assert payload["attribution"]["manager_packet"] == _packet()

        # Without the attribution block there is nothing to reproduce the recorded
        # manager/request digests FROM.
        stripped = json.loads(json.dumps(payload))
        stripped.pop("attribution")
        artifact_path.write_text(json.dumps(stripped), encoding="utf-8")
        with pytest.raises(gs.GenerationRefused) as unattributed:
            gs.verify_decision(conn, outcome["decision_record_id"])
        assert "no attribution block" in str(unattributed.value)

        # A manager packet edited underneath its recorded digest is caught by the
        # RECOMPUTATION, not by a label.
        tampered = json.loads(json.dumps(payload))
        tampered["attribution"]["manager_packet"]["entry_id"] = 999
        artifact_path.write_text(json.dumps(tampered), encoding="utf-8")
        with pytest.raises(gs.GenerationRefused) as mutated:
            gs.verify_decision(conn, outcome["decision_record_id"])
        assert "manager_packet_sha256" in str(mutated.value)
    finally:
        conn.close()


def test_verify_generation_reproduces_the_pe8_reference_or_says_it_did_not_consult_it(tmp_path):
    path = tmp_path / "fpl.db"
    _world(path)
    conn = connect_database(path)
    try:
        generation = gs.resolve_generation(
            conn, planning_event=5, horizon_kind=gs.HORIZON_KIND_MANAGER_WORLD
        )
        report = gs.verify_generation(conn, generation.generation_id)
        assert report["pe8_evidence_consulted"] is False
        assert report["pe8_evidence_reproduction"] == gs.PE8_NOT_CONSULTED
        assert report["pe8_evidence_refs_reproduce"] is None

        # A consulted artifact's identity and state recompute from what the manifest
        # records; a disagreeing state does not.
        calibration = {
            "schema": cb.known_calibration_versions()["schema"],
            "evaluation_version": cb.known_calibration_versions()["evaluation_version"],
            "identity": {
                "certification_identity": "sha256:" + "f" * 64,
                "planning_cutoff": gf.CUTOFF,
            },
            "terminal_state": {"state": "OPEN", "reasons": []},
            "surfaces": [],
        }
        surface = {"surface": "minutes_60min_probability", "state": cb.STATE_CERTIFIED_COHERENT}
        evidence = {
            "consulted": True,
            "identity": cb.calibration_identity(calibration),
            "schema": calibration["schema"],
            "evaluation_version": calibration["evaluation_version"],
            "certification_identity": "sha256:" + "f" * 64,
            "planning_cutoff": gf.CUTOFF,
            "terminal_state": "OPEN",
            "state": cb.calibration_evidence_state(calibration, [surface]),
            "surfaces": [surface],
            "per_event_state": {"5": cb.STATE_EVIDENCE_LIMITED},
        }
        reproduced, failures = gs.reproduce_pe8_evidence(
            evidence, {"5": {"evidence_state": cb.STATE_EVIDENCE_LIMITED}}
        )[:2]
        assert reproduced is True and failures == []
        broken = dict(evidence, terminal_state="READY_FOR_MERGE")
        reproduced, failures = gs.reproduce_pe8_evidence(
            broken, {"5": {"evidence_state": cb.STATE_EVIDENCE_LIMITED}}
        )[:2]
        assert reproduced is False
        assert any("does not reproduce" in failure for failure in failures)
        mismatched, failures = gs.reproduce_pe8_evidence(
            evidence, {"5": {"evidence_state": cb.STATE_CERTIFIED_COHERENT}}
        )[:2]
        assert mismatched is False and failures
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# The CERTIFIED production runner must reach decision-artifact assembly
# ---------------------------------------------------------------------------


class _PermissiveStub(dict):
    """A permissive mapping/callable standing in for the search internals.

    This regression's subject is the artifact ASSEMBLY performed by the certified
    production runner -- the site that crashed with an undefined name AFTER a completed
    search.  Every heavy collaborator is replaced by this stub so the real assembly code
    runs in milliseconds; the value the assertion depends on (the cutoff guard) is still
    produced by production code, never by the stub.
    """

    def __missing__(self, key):
        return _PermissiveStub()

    def __call__(self, *args, **kwargs):
        return _PermissiveStub()

    def get(self, key, default=None):
        return _PermissiveStub()


def _stub_certified_runner_search(runner_globals, monkeypatch, calls):
    """Replace the bounded search and its refinement with cheap permissive stubs."""

    def fake_optimize(**kwargs):
        calls["optimize"] += 1
        calls["cache_dir"] = kwargs.get("cache_dir")
        return _PermissiveStub()

    monkeypatch.setattr(runner_globals["ro"], "optimize", fake_optimize)
    for name in ("refine_finalists", "analyze_leader_change", "assess_search_stability",
                 "final_ranking_after_escalation", "finalist_partials"):
        if hasattr(runner_globals["fr"], name):
            monkeypatch.setattr(runner_globals["fr"], name, lambda *a, **k: _PermissiveStub())


def _stub_certified_runner_boundaries(runner_globals, monkeypatch, route_comparator):
    monkeypatch.setattr(runner_globals["cu"], "price_snapshot_as_of", lambda *a, **k: object())
    monkeypatch.setattr(route_comparator, "flat_current_price_scenario", lambda *a, **k: object())
    monkeypatch.setattr(runner_globals["cu"], "load_pool", lambda *a, **k: {"players": []})
    monkeypatch.setattr(
        runner_globals["provenance"],
        "assert_official_pool_identity",
        lambda **_k: {
            "official_generation_id": "test-generation",
            "snapshot_pool_count": 15,
            "snapshot_pool_ids_sha256": "sha256:" + "a" * 64,
            "official_pool_identity_match": True,
        },
    )
    monkeypatch.setattr(runner_globals["cu"], "load_fixtures_by_team", lambda *a, **k: {})
    monkeypatch.setattr(runner_globals["cu"], "load_projection_rows", lambda *a, **k: [])
    monkeypatch.setattr(
        runner_globals["cu"],
        "build_universe",
        lambda **_k: {"universe": [{"player_id": 1}], "excluded": []},
    )
    monkeypatch.setattr(runner_globals["cu"], "build_replacement_edges", lambda **_k: [])
    monkeypatch.setattr(runner_globals["fg"], "screen_legal_actions", lambda **_k: {})
    monkeypatch.setattr(runner_globals["cu"], "discovery_completeness", lambda **_k: {})
    monkeypatch.setattr(
        runner_globals["fg"], "optimizer_routes_for_decision",
        lambda *a, **k: [{"route_id": "route-0", "transfers": [], "per_event": []}],
    )
    monkeypatch.setattr(
        runner_globals["fg"], "evaluate_four_gw_decision",
        lambda *a, **k: {"transfer_recommendation": {"preferred_route_id": "route-0"}},
    )
    monkeypatch.setattr(
        runner_globals["fg"], "lineup_policy_for_route", lambda *a, **k: {"status": "LINEUP_ONLY"}
    )
    monkeypatch.setattr(
        runner_globals["fg"], "classify_fixture_horizon",
        lambda *a, **k: {"status": "OK", "complete": True, "blocking_reasons": []},
    )


def test_certified_executor_reaches_artifact_assembly_and_reports_the_override_guard(
    tmp_path, monkeypatch
):
    """The certified runner must ASSEMBLE its decision artifact, guard included.

    Regression for the defect that made the real production runner raise
    ``NameError: name 'override' is not defined`` while building
    ``artifact["cutoff_guard"]`` after a completed search.  The guard is now computed from
    the PINNED snapshot's manager context before any search, and the artifact carries that
    value.  The fixture's snapshot holds real user-confirmed manager state, so the guard
    must identify the override instant and pass.
    """

    import fpl_brain.route_comparator as route_comparator

    path = tmp_path / "four-gw.db"
    conn, generation = _four_gw_certified_manager_world(
        path, bank=7, free_transfers=3, event_start_free_transfers=None
    )
    runner = gs._declared_production_entrypoint(
        gs.PRODUCTION_DECISION_MODULES[gs.HORIZON_KIND_FOUR_GW]
    )
    g = runner.__globals__
    calls = {"optimize": 0, "cache_dir": "unset"}

    _stub_certified_runner_boundaries(g, monkeypatch, route_comparator)
    _stub_certified_runner_search(g, monkeypatch, calls)

    source_conn = gs._open_generation_snapshot(generation)
    packet = {"entry_id": 241392, "planning_event": 4, "cutoff": gf.CUTOFF, "season": "2026/27"}
    canonical = gs._derive_four_gw_consumed_manager_state(
        source_conn, generation=generation, manager_packet=packet
    )
    try:
        outcome = runner(
            conn=conn,
            generation=generation,
            manager_packet=packet,
            parameters={"stage2_draws": g["STAGE1_DRAWS"] + 1},
            source_conn=source_conn,
            canonical_manager_state=canonical,
        )
    finally:
        source_conn.close()

    blocks = outcome.get("artifact_blocks") or outcome
    assert "cutoff_guard" in blocks, "the certified decision artifact must carry the cutoff guard"
    guard = blocks["cutoff_guard"]
    # Real override evidence: the snapshot's user-confirmed state was captured at the
    # instant the certified cutoff covers, so the guard identifies it and passes.
    assert guard["override_captured_at"] == gf.CUTOFF
    assert guard["pass"] is True
    assert guard["status"] == "PASS"
    assert guard["planning_cutoff"] == generation.cutoff

    # The search ran exactly once, and its cache location was derived from the CERTIFIED
    # generation's snapshot -- never from the source checkout, never from the packet.
    assert calls["optimize"] == 1
    snapshot_row = conn.execute(
        "SELECT snapshot_path FROM generation WHERE generation_id=?", (generation.generation_id,)
    ).fetchone()
    assert snapshot_row is not None, "the certified generation must persist its snapshot path"
    assert calls["cache_dir"] == Path(snapshot_row["snapshot_path"]).parent / "world_cache" / "manager_worlds"
    assert "cache_dir" not in packet
    conn.close()


def test_certified_executor_checks_the_override_guard_before_any_search(tmp_path, monkeypatch):
    """A failing cutoff guard must refuse BEFORE the search.

    The guard is the one check that cannot wait: measured afterwards it would spend the
    whole search budget on a decision that must not be taken.  The guard's verdict is
    forced to fail here so the EXECUTOR's ordering is what is under test -- the search
    must never be reached, and the refusal must carry the specific token.
    """

    import fpl_brain.route_comparator as route_comparator

    path = tmp_path / "four-gw.db"
    conn, generation = _four_gw_certified_manager_world(
        path, bank=7, free_transfers=3, event_start_free_transfers=None
    )
    runner = gs._declared_production_entrypoint(
        gs.PRODUCTION_DECISION_MODULES[gs.HORIZON_KIND_FOUR_GW]
    )
    g = runner.__globals__
    calls = {"optimize": 0, "cache_dir": "unset"}

    _stub_certified_runner_boundaries(g, monkeypatch, route_comparator)
    _stub_certified_runner_search(g, monkeypatch, calls)
    monkeypatch.setattr(
        g["fg"],
        "verify_cutoff_covers_override",
        lambda **_k: {
            "status": g["fg"].CUTOFF_PRECEDES_OVERRIDE,
            "pass": False,
            "planning_cutoff": gf.CUTOFF,
            "override_captured_at": gf.CUTOFF,
            "detail": "forced for the ordering test",
        },
    )

    source_conn = gs._open_generation_snapshot(generation)
    try:
        with pytest.raises(gs.GenerationRefused) as refusal:
            runner(
                conn=conn,
                generation=generation,
                manager_packet={
                    "entry_id": 241392, "planning_event": 4, "cutoff": gf.CUTOFF,
                    "season": "2026/27",
                },
                parameters={"stage2_draws": g["STAGE1_DRAWS"] + 1},
                source_conn=source_conn,
                canonical_manager_state=None,
            )
        assert refusal.value.token == g["DIAG_CUTOFF_GUARD_FAILED"]
        assert "does not cover the confirmed manager override" in str(refusal.value)
        assert calls["optimize"] == 0, "the guard must refuse before the search"
    finally:
        source_conn.close()
        conn.close()
