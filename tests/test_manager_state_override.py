"""User-confirmed manager-state override and no-write guarantees.

Item 1: the production PlanningContext must resolve an explicit user-confirmed
override (post-transfer FT/bank/ownership) and must fail rather than silently
fall back to stale official state.

Item 3: decision-layer operations create no database rows (before/after proof).
"""

from __future__ import annotations

import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest

from fpl_brain import four_gw_decision as fg
from fpl_brain import manager_worlds
from fpl_brain import repositories as repo
from fpl_brain import route_comparator as rc
from fpl_brain import transfer_state as ts
from fpl_brain.database import connect_database
from fpl_brain.models import (
    EventRecord,
    FixtureRecord,
    PickRecord,
    PlayerRecord,
    PlayerSnapshotRecord,
    PositionRecord,
    TeamRecord,
)
from fpl_brain.planning import get_planning_context
from test_transfer_state import CLUB, POOL_CLUB, POOL_POSITION, POSITION

REPO_ROOT = Path(__file__).resolve().parents[1]

# 2 GKP / 5 DEF / 5 MID / 3 FWD.  Players 41 (already confirmed) and 42 (the
# move being confirmed now) are unowned DEF candidates; 51 is an unowned MID.
# Official (stale) picks at the planning event, and the ledger-active squad
# after the first already-confirmed move (12 OUT, 41 IN).
_BASE_SQUAD = [1, 2, 12, 13, 14, 15, 43, 21, 22, 23, 24, 25, 31, 32, 33]
_PURCHASE = 45
_MARKET = {13: 53, 42: 45}  # sell 13 at 49 (45 + (53-45)//2), buy 42 at 45 -> 3 + 49 - 45 = 7


def _seed_pre_override_world(conn, entry_id: int = 241392, event: int = 4):
    """Ledger ahead of official history; FT 1 / £0.3m; one confirmed move already in."""

    ids = sorted(set(_BASE_SQUAD) | {41, 42, 51, 55})
    team_ids = sorted({int(CLUB.get(pid) or POOL_CLUB[pid]) for pid in ids})
    market = {pid: int(_MARKET.get(pid, _PURCHASE)) for pid in ids}
    with conn:
        repo.upsert_teams(conn, [TeamRecord(id=team_id, name=f"Team {team_id}") for team_id in team_ids])
        repo.upsert_positions(conn, [
            PositionRecord(id=1, singular_name="Goalkeeper", singular_name_short="GKP"),
            PositionRecord(id=2, singular_name="Defender", singular_name_short="DEF"),
            PositionRecord(id=3, singular_name="Midfielder", singular_name_short="MID"),
            PositionRecord(id=4, singular_name="Forward", singular_name_short="FWD"),
        ])
        repo.upsert_events(conn, [
            EventRecord(id=1, finished=1, data_checked=1, deadline_time="2026-08-21T17:30:00Z", raw_json={}),
            EventRecord(id=event, finished=0, data_checked=0, deadline_time="2026-09-12T12:30:00Z", raw_json={}),
        ])
        repo.upsert_fixtures(conn, [FixtureRecord(id=41, event=event, team_h=1, team_a=2,
                                                  started=0, finished=0, raw_json={})])
        repo.upsert_players(conn, [
            PlayerRecord(id=pid, web_name=f"P{pid}", full_name=f"Player {pid}",
                         team_id=CLUB.get(pid) or POOL_CLUB[pid],
                         element_type={"GKP": 1, "DEF": 2, "MID": 3, "FWD": 4}[POSITION.get(pid) or POOL_POSITION[pid]])
            for pid in ids
        ])
        run = repo.create_fetch_run(conn, "fetch_fpl", started_at="2026-09-12T08:00:00Z")
        repo.insert_snapshots(conn, [
            PlayerSnapshotRecord(player_id=pid, captured_at="2026-09-12T08:00:00Z", now_cost=market[pid], raw_json={})
            for pid in ids
        ], run)
        repo.finish_fetch_run(conn, run, "success", current_event=event)
        # Official picks exist only for the initial Gameweek (as in the live DB),
        # so the confirmed acquisition ledger is the authoritative ownership view
        # for the planning event.
        repo.upsert_squad_picks(conn, entry_id, 1, [
            PickRecord(player_id=pid, position=index + 1, raw_json={}) for index, pid in enumerate(_BASE_SQUAD)
        ])
        for pid in _BASE_SQUAD:
            repo.insert_manager_acquisition(conn, entry_id, pid, 1, _PURCHASE,
                                            source="official_transfer_history", created_at="2026-08-30T10:00:00Z")
        # A previously confirmed move already in the ledger: 12 OUT, 41 IN (manual).
        row_twelve = next(row for row in repo.active_manager_acquisitions(conn, entry_id)
                          if int(row["player_id"]) == 12)
        repo.close_manager_acquisition(conn, int(row_twelve["id"]), event, sold_at="2026-09-11T22:46:18Z")
        repo.insert_manager_acquisition(conn, entry_id, 41, event, _PURCHASE, source="manual",
                                       acquired_at="2026-09-11T22:46:18Z")
        repo.upsert_manager_selling_prices(conn, entry_id, event, {41: 45},
                                           captured_at="2026-09-11T22:46:18Z", market_prices_at_capture={41: 45})
        # Official-but-stale sync: the public endpoint has not published the move.
        sync_run = repo.create_fetch_run(conn, "sync_manager", started_at="2026-09-11T20:00:00Z")
        repo.finish_fetch_run(conn, sync_run, "success", current_event=event)
        conn.execute(
            """INSERT INTO manager_state(entry_id, fetch_run_id, captured_at, event, bank, team_value,
               total_transfers, event_transfers, event_transfers_cost, points_on_bench, active_chip,
               free_transfers_manual, raw_json)
               VALUES (?, ?, '2026-09-11T20:00:00Z', ?, 3, 825, 0, 0, 0, 0, NULL, 1, ?)""",
            (entry_id, sync_run, event,
             '{"transfers_endpoint_available": true, "transfers": [], "history": {"current": []}}'),
        )
        repo.upsert_manual_manager_state(conn, entry_id, event, 1, 3,
                                         source="user_correction_1FT_first_move_executed",
                                         captured_at="2026-09-11T22:46:18Z")


def _temp_config(tmp_path) -> Path:
    config = {
        "fpl_entry_id": 241392,
        "season": "2026/27",
        "paths": {
            "database": str(tmp_path / "fpl.db"),
            "raw_dir": str(tmp_path / "raw"),
            "exports_dir": str(tmp_path / "exports"),
        },
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def _cli():
    path = REPO_ROOT / "scripts/confirm_manager_state.py"
    spec = importlib.util.spec_from_file_location("confirm_manager_state", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Item 1
# ---------------------------------------------------------------------------


def test_confirm_manager_state_override_is_authoritative_and_never_recharged(tmp_path):
    config_file = _temp_config(tmp_path)
    conn = connect_database(tmp_path / "fpl.db")
    try:
        _seed_pre_override_world(conn)
        before = get_planning_context(conn, 241392, 4, as_of="2026-09-12T09:00:00Z")
        assert before.manager_state["free_transfers"] == 1
        assert before.manager_state["bank"] == 3
        assert before.health["status"] == "PASS"

        cli = _cli()
        rc_code = cli.main([
            "--config", str(config_file), "--event", "4", "--ft", "0", "--bank", "7",
            "--event-start-ft", "2",
            "--transfer", "13:42", "--source-label", "user_confirmed_gw4_second_move",
            "--captured-at", "2026-09-12T09:00:00Z", "--quiet",
        ])
        assert rc_code == 0

        context = get_planning_context(conn, 241392, 4, as_of="2026-09-12T09:30:00Z")
        state = context.manager_state
        assert state["free_transfers"] == 0 and state["free_transfers_source"] == "manual"
        assert state["bank"] == 7 and state["bank_source"] == "manual"
        assert state["authoritative_source"] == "user_confirmed_override"
        assert state["override_authoritative"] is True
        assert state["field_provenance"] == {
            "free_transfers": "manual", "bank": "manual", "event_start_free_transfers": "manual",
        }
        assert state["event_start_free_transfers"] == 2
        assert state["event_start_free_transfers_source"] == "manual"
        assert state["stale_official_state_overridden"] is True
        assert state["official_api"]["free_transfers"] == 1  # official row still stale
        assert state["override"]["source"] == "user_confirmed_gw4_second_move"
        assert context.health["status"] == "PASS", context.health

        squad = manager_worlds.resolve_squad(context, conn)
        owned = set(int(pid) for pid in squad["squad_ids"])
        assert len(owned) == 15
        assert 42 in owned and 41 in owned and 13 not in owned and 12 not in owned

        # Already-executed moves are represented once and never charged again.
        route_state = rc.build_route_state(conn, context, squad)
        assert route_state.free_transfers == 0
        assert route_state.bank_tenths == 7
        assert route_state.cumulative_hit_points == 0
        assert route_state.event_start_free_transfers == 2  # explicit, not inferred from 0
        meta = rc.load_player_meta(conn, sorted(owned | {55}))
        extra = ts.apply_transfer_batch(route_state, ts.TransferBatch((ts.TransferAction(22, 55),)),
                                        ts.PriceSnapshot(event=4, prices={22: 45, 55: 45}), meta)
        assert extra.ok, extra.errors
        assert extra.hit_points == 4  # any ADDITIONAL transfer is paid while FT remaining is 0
        assert extra.paid_transfers == 1
        assert 41 in extra.squad_after.by_id()  # prior signing untouched, never recharged

        # Replaying the same override is rejected rather than double-charged.
        again = cli.main([
            "--config", str(config_file), "--event", "4", "--ft", "0", "--bank", "7",
            "--transfer", "13:42", "--quiet",
        ])
        assert again == 3
        assert conn.execute("SELECT COUNT(*) FROM manager_player_acquisitions WHERE sold_event IS NULL"
                            ).fetchone()[0] == 15
    finally:
        conn.close()


def test_missing_override_fails_rather_than_reverting_to_stale_official_state(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    try:
        _seed_pre_override_world(conn)
        with conn:
            # Remove the explicit override entirely (both the current row and the
            # append-only observations it resolves from).
            conn.execute("DELETE FROM manager_manual_state WHERE entry_id=241392 AND event=4")
            conn.execute("DELETE FROM manager_state_observations WHERE entry_id=241392 AND event=4")
        context = get_planning_context(conn, 241392, 4, as_of="2026-09-12T09:30:00Z")
        assert context.health["status"] == "FAIL"
        assert any("no explicit user-confirmed override" in reason for reason in context.health["fail_reasons"])
        assert any("ahead of official transfer history" in reason for reason in context.warnings)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Item 3
# ---------------------------------------------------------------------------


def _insert_run(conn, family, event, cutoff, status="complete"):
    conn.execute(
        "INSERT INTO projection_runs(model_family, model_version, generated_at, planning_event, "
        "data_cutoff, status) VALUES (?,?,?,?,?,?)",
        (family, "test_v1", "2026-09-12T00:00:00Z", int(event), cutoff, status),
    )


def test_decision_layer_operations_create_no_database_rows(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    try:
        with conn:
            for event in (4, 5, 6, 7):
                for family in fg.REQUIRED_HORIZON_FAMILIES:
                    _insert_run(conn, family, event, "CUT")

        def stats():
            return (
                conn.execute("SELECT COUNT(*) FROM projection_runs").fetchone()[0],
                conn.execute("SELECT COALESCE(MAX(id),0) FROM projection_runs").fetchone()[0],
                conn.execute("SELECT COUNT(*) FROM frozen_predictions").fetchone()[0],
                conn.execute("SELECT COUNT(*) FROM monte_carlo_distributions").fetchone()[0],
            )

        before = stats()
        support = fg.event_support_from_db(conn, (4, 5, 6, 7), "CUT")
        horizon = fg.evaluate_horizon(planning_event=4, support_by_event=support, cutoff="CUT")
        decision = fg.evaluate_four_gw_decision(
            planning_event=4, support_by_event=support, cutoff="CUT",
            screened_actions={"legal_single_transfers": 0},
            routes=[{"route_id": "ROLL", "valid": True,
                     "per_event": [{"event": event, "mean_gross_core": 10.0, "hit_points": 0}
                                   for event in (4, 5, 6, 7)],
                     "terminal_ft": 5, "terminal_bank_tenths": 3}],
        )
        fg.wildcard_trigger_screen(weak_slot_count=4, prior_paid_transfers=1)
        after = stats()
        assert horizon["status"] == fg.DECISION_HORIZON_COMPLETE
        assert decision["transfer_recommendation"]["preferred_route_id"] == "ROLL"
        assert before == after, f"decision layer wrote to the database: {before} -> {after}"
    finally:
        conn.close()
