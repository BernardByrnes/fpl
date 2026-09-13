"""PlanningContext coherence, event finality semantics, and the health gate."""

from __future__ import annotations

import pytest

from fpl_brain import repositories as repo
from fpl_brain.database import connect_database
from fpl_brain.models import (
    ChipRecord,
    EventRecord,
    FixtureRecord,
    ManagerChipRecord,
    PickRecord,
    PlayerRecord,
    PlayerSnapshotRecord,
    TeamRecord,
)
from fpl_brain.planning import (
    EVENT_STATE_FINAL,
    EVENT_STATE_PROVISIONAL,
    EVENT_STATE_SCHEDULED,
    HEALTH_FAIL,
    HEALTH_PASS,
    HEALTH_WARN,
    event_data_state,
    get_planning_context,
)


PLAYER_IDS = list(range(1, 16))


def _seed(conn, *, eventfinished: bool, data_checked: bool):
    with conn:
        repo.upsert_teams(conn, [TeamRecord(id=1, name="One"), TeamRecord(id=2, name="Two")])
        repo.upsert_chips(
            conn,
            [
                ChipRecord(id=1, name="wildcard", number=1, chip_type="transfer", start_event=2, stop_event=19),
                ChipRecord(id=2, name="wildcard", number=2, chip_type="transfer", start_event=20, stop_event=38),
                ChipRecord(id=3, name="freehit", number=1, chip_type="transfer", start_event=2, stop_event=19),
                ChipRecord(id=4, name="freehit", number=2, chip_type="transfer", start_event=20, stop_event=38),
            ],
        )
        repo.upsert_events(
            conn,
            [
                EventRecord(id=1, finished=1, data_checked=1, deadline_time="2026-08-21T17:30:00Z", raw_json={}),
                EventRecord(
                    id=2,
                    finished=1 if eventfinished else 0,
                    data_checked=1 if data_checked else 0,
                    deadline_time="2026-08-28T17:30:00Z",
                    raw_json={},
                ),
                EventRecord(id=3, finished=0, data_checked=0, deadline_time=None, raw_json={}),
            ],
        )
        repo.upsert_fixtures(
            conn,
            [  # GW2 fixtures all finished; GW3 future; GW1 one started-unfinished
                FixtureRecord(id=11, event=1, team_h=1, team_a=2, started=1, finished=1, raw_json={}),
                FixtureRecord(id=21, event=2, team_h=1, team_a=2, started=1, finished=1, raw_json={}),
                FixtureRecord(id=31, event=3, team_h=1, team_a=2, started=0, finished=0, raw_json={}),
            ],
        )
        repo.upsert_players(conn, [PlayerRecord(id=pid, web_name=f"P{pid}", full_name=f"Player {pid}") for pid in PLAYER_IDS])
        run1 = repo.create_fetch_run(conn, "fetch_fpl", started_at="2026-08-30T09:00:00Z")
        repo.insert_snapshots(
            conn,
            [PlayerSnapshotRecord(player_id=pid, captured_at="2026-08-30T09:00:00Z", now_cost=50, raw_json={}) for pid in PLAYER_IDS],
            run1,
        )
        repo.finish_fetch_run(conn, run1, "success", current_event=2)
        repo.upsert_squad_picks(
            conn,
            241392,
            2,
            [PickRecord(player_id=pid, position=index + 1, raw_json={}) for index, pid in enumerate(PLAYER_IDS)],
        )
        for pid in PLAYER_IDS:
            repo.insert_manager_acquisition(
                conn, 241392, pid, 2, 50, source="official_transfer_history", created_at="2026-08-30T10:00:00Z"
            )
        repo.upsert_manual_manager_state(conn, 241392, 2, 3, 0, captured_at="2026-08-29T12:00:00Z")
        # Wildcard set #2 used at GW21: set #1 (GW2-GW19) stays available.
        repo.upsert_manager_chips(conn, 241392, [ManagerChipRecord(name="wildcard", event=21, time="2026-09-20T00:00:00Z")])
        run2 = repo.create_fetch_run(conn, "sync_manager", started_at="2026-08-30T12:00:00Z")
        repo.finish_fetch_run(conn, run2, "success", current_event=2)
    return run1, run2


def _owner_records(conn):
    return repo.active_manager_acquisitions(conn, 241392)


def test_event_data_state_semantics(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    _seed(conn, eventfinished=True, data_checked=True)
    assert event_data_state(conn, 2)[0] == EVENT_STATE_FINAL
    second = connect_database(tmp_path / "second.db")
    _seed(second, eventfinished=False, data_checked=False)
    assert event_data_state(second, 2)[0] == EVENT_STATE_PROVISIONAL
    assert "all fixtures finished while event is not officially finalised" in event_data_state(second, 2)[1]
    assert event_data_state(second, 3)[0] == EVENT_STATE_SCHEDULED
    assert event_data_state(second, 99)[0] == "UNKNOWN"
    second.close()
    conn.close()


def test_planning_context_passes_on_coherent_state(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    _seed(conn, eventfinished=True, data_checked=True)
    context = get_planning_context(
        conn, 241392, 2, season="2026/27", scouting_stale_after_days=14, as_of="2026-08-30T00:00:00Z"
    )
    assert context.squad["squad_state"] == "exact_target_event"
    assert context.squad["player_count"] == 15
    assert context.squad["status"] == "OK"
    assert context.manager_state["free_transfers"] == 3
    assert context.manager_state["free_transfers_source"] == "manual"
    assert context.acquisitions["reconciliation_status"] in {"success", "data_gap"}
    # A context call must never mutate the acquisition ledger (dry-run only).
    assert conn.execute("SELECT COUNT(*) FROM manager_player_acquisitions").fetchone()[0] == 15
    assert len(context.selling_prices) == 15
    assert context.health["status"] == HEALTH_PASS
    assert context.official_runs["fetch"]["run_id"] is not None
    assert context.official_runs["manager_sync"]["run_id"] is not None
    wildcard = [chip for chip in context.chips if chip["name"] == "wildcard"]
    # Two chip sets stored (window 1-19 and window-2 definitions may differ):
    assert any(chip["used"] for chip in wildcard)
    assert context.event_data_state == EVENT_STATE_FINAL
    conn.close()


def test_provisional_event_raises_health_warn(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    _seed(conn, eventfinished=False, data_checked=False)
    context = get_planning_context(conn, 241392, 2)
    assert context.event_data_state == EVENT_STATE_PROVISIONAL
    assert context.health["status"] == HEALTH_WARN
    assert any("PROVISIONAL" in reason for reason in context.health["warn_reasons"])
    conn.close()


def test_missing_squad_fails_health_gate(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    _seed(conn, eventfinished=True, data_checked=True)
    with conn:
        # No verified squad of 15: neither official picks nor a verified ledger.
        conn.execute("DELETE FROM squad_picks WHERE entry_id=241392 AND event=2")
        conn.execute("DELETE FROM player_snapshots")
        conn.execute("DELETE FROM manager_player_acquisitions WHERE entry_id=241392")
    context = get_planning_context(conn, 241392, 2)
    assert context.health["status"] == HEALTH_FAIL
    assert any("verified 15-player squad" in reason for reason in context.health["fail_reasons"])
    conn.close()


def test_ledger_resolves_stale_official_picks_without_old_squad_pass(tmp_path):
    """A stale official picks/history view plus a verified manual transfer must
    resolve the real current squad, never the stale one.

    The verified manual observation (2 FT, £0.3m bank) supersedes the pre-
    transfer manual observation; the acquisition ledger — with the outgoing
    player closed and the incoming purchase recorded with explicit manual
    provenance — is current ownership while the official data lags.
    """
    conn = connect_database(tmp_path / "fpl.db")
    _seed(conn, eventfinished=True, data_checked=True)  # official squads end at GW2-equivalent
    with conn:
        # Official picks/history: unchanged (stale), Maguire-equivalent (player 1) owned.
        # GW planning event has no official picks. First the pre-transfer manual
        # view, then the later verified manual move: 16 IN (47), 1 OUT, 2 FT, £0.3m.
        repo.upsert_manual_manager_state(conn, 241392, 4, 3, 0, captured_at="2026-09-08T10:00:00Z")
        repo.upsert_manual_manager_state(conn, 241392, 4, 2, 3, captured_at="2026-09-10T16:00:00Z")
        # The incoming player exists officially at £4.7m with a market snapshot.
        repo.upsert_players(conn, [PlayerRecord(id=16, web_name="P16", full_name="Player 16")])
        fresh_run = repo.create_fetch_run(conn, "fetch_fpl", started_at="2026-09-10T09:00:00Z")
        repo.finish_fetch_run(conn, fresh_run, "success", current_event=4)
        repo.insert_snapshots(
            conn,
            [PlayerSnapshotRecord(player_id=pid, captured_at="2026-09-10T09:00:00Z", now_cost=50, raw_json={}) for pid in range(1, 16)],
            fresh_run,
        )
        repo.insert_snapshots(
            conn,
            [PlayerSnapshotRecord(player_id=16, captured_at="2026-09-10T09:00:00Z", now_cost=47, raw_json={})],
            fresh_run,
        )
    maguire = next(row for row in repo.active_manager_acquisitions(conn, 241392) if int(row["player_id"]) == 1)
    with conn:
        # Drop the seed's own stale GW3 fixture artifact (never finished, no
        # kickoff) so the horizon warning does not shadow the scenario.
        conn.execute("DELETE FROM fixtures WHERE id=31")
        repo.close_manager_acquisition(conn, int(maguire["id"]), 4, sold_at="2026-09-10T16:00:00Z")
        repo.insert_manager_acquisition(
            conn, 241392, 16, 4, 47, source="manual", created_at="2026-09-10T16:00:00Z"
        )
    context = get_planning_context(conn, 241392, 4, as_of="2026-09-10T16:00:00Z")
    assert context.squad["squad_state"] == "acquisition_ledger"
    squad_ids = [player["player_id"] for player in context.squad["players"]]
    assert 16 in squad_ids and 1 not in squad_ids
    assert len(squad_ids) == 15
    assert context.manager_state["free_transfers"] == 2
    assert context.manager_state["bank"] == 3
    assert context.manager_state["free_transfers_source"] == "manual"
    # Newest verified observation wins (the pre-transfer 3 FT state is history).
    assert context.manager_state["observation_count"] == 2
    assert context.manager_state["latest_observation"] == "2026-09-10T16:00:00Z"
    assert context.health["status"] == HEALTH_PASS
    # Sanity: the acquisitions context never mutated the ledger (dry-run only).
    assert conn.execute("SELECT COUNT(*) FROM manager_player_acquisitions WHERE sold_event IS NULL AND entry_id=241392").fetchone()[0] == 15
    assert conn.execute("SELECT COUNT(*) FROM manager_player_acquisitions").fetchone()[0] == 16
    conn.close()


def test_official_row_later_reconciles_manual_without_duplicates(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    micros = _seed_ledger_transfer_world(conn)
    official = dict(micros)
    official["transfer"] = {
        "entry": 99,
        "element_in": 16,
        "element_out": 1,
        "event": 4,
        "time": "2026-09-11T09:00:00Z",
        "element_in_cost": 47,
        "element_out_cost": 46,
    }
    result = repo.reconcile_manager_acquisitions(conn, 99, 4, [official["transfer"]], True, captured_at="2026-09-11T10:00:00Z")
    assert result["status"] == "success"
    assert result["inserted"] == 0 and result["closed"] == 0
    assert result.get("confirmed_manual") == 1
    stints = repo.list_manager_acquisitions(conn, 99, 16)
    assert [(row["source"], row["acquired_event"], row["purchase_price"], row["sold_event"], row["acquired_at"]) for row in stints] == [
        ("reconciled", 4, 47, None, "2026-09-11T09:00:00Z")
    ]
    assert conn.execute("SELECT COUNT(*) FROM manager_player_acquisitions").fetchone()[0] == micros["acquisition_count"]
    # Exact official provenance never claims verification it lacks: the manual
    # source label was explicit before confirmation and reconciled after.
    again = repo.reconcile_manager_acquisitions(conn, 99, 4, [official["transfer"]], True, captured_at="2026-09-11T11:00:00Z")
    assert again["status"] == "success"
    assert again.get("confirmed_manual", 0) == 0
    assert conn.execute("SELECT COUNT(*) FROM manager_player_acquisitions").fetchone()[0] == micros["acquisition_count"]
    conn.close()


def _seed_ledger_transfer_world(conn):
    """World with a stale official view and a verified pre-deadline manual move."""

    with conn:
        repo.upsert_teams(conn, [TeamRecord(id=1, name="One"), TeamRecord(id=2, name="Two")])
        repo.upsert_events(
            conn,
            [
                EventRecord(id=1, finished=1, data_checked=1, deadline_time="2026-08-21T17:30:00Z", raw_json={}),
                EventRecord(id=2, finished=1, data_checked=1, deadline_time="2026-08-28T17:30:00Z", raw_json={}),
            ],
        )
        repo.upsert_players(conn, [PlayerRecord(id=pid, web_name=f"P{pid}", full_name=f"Player {pid}") for pid in range(1, 17)])
        run = repo.create_fetch_run(conn, "fetch_fpl")
        repo.insert_snapshots(
            conn,
            [PlayerSnapshotRecord(player_id=pid, captured_at="2026-08-30T09:00:00Z", now_cost=47 if pid == 16 else 50, raw_json={}) for pid in range(1, 17)],
            run,
        )
        repo.upsert_squad_picks(conn, 99, 1, [PickRecord(player_id=pid, position=i + 1, raw_json={}) for i, pid in enumerate(range(1, 16))])
        for pid in range(1, 16):
            repo.insert_manager_acquisition(conn, 99, pid, 1, 50, source="official_transfer_history", created_at="2026-08-30T10:00:00Z")
        # Fresh official price run (prices current; only official TRANSFER
        # history is stale in this scenario).
        fresh_run = repo.create_fetch_run(conn, "fetch_fpl", started_at="2026-09-10T09:00:00Z")
        repo.finish_fetch_run(conn, fresh_run, "success", current_event=4)
        repo.insert_snapshots(
            conn,
            [PlayerSnapshotRecord(player_id=pid, captured_at="2026-09-10T09:00:00Z", now_cost=50, raw_json={}) for pid in range(1, 17)],
            fresh_run,
        )
        # A user-verified pre-deadline move: player 1 OUT (closed), 16 IN (manual £4.7m).
        row_one = next(row for row in repo.active_manager_acquisitions(conn, 99) if int(row["player_id"]) == 1)
        repo.close_manager_acquisition(conn, int(row_one["id"]), 4, sold_at="2026-09-10T16:00:00Z")
        repo.insert_manager_acquisition(conn, 99, 16, 4, 47, source="manual", created_at="2026-09-10T16:00:00Z")
        repo.upsert_manager_selling_prices(conn, 99, 4, {16: 47}, captured_at="2026-09-10T16:00:00Z", market_prices_at_capture={16: 47})
        # Official-but-stale transfer history: a successful sync whose stored
        # history does not yet know about the real move.
        run_id = repo.create_fetch_run(conn, "sync_manager", started_at="2026-09-07T14:34:56Z")
        repo.finish_fetch_run(conn, run_id, "success", current_event=4)
        conn.execute(
            """INSERT INTO manager_state(entry_id, fetch_run_id, captured_at, event, bank, team_value,
               total_transfers, event_transfers, event_transfers_cost, points_on_bench, active_chip,
               free_transfers_manual, raw_json)
               VALUES (99, ?, '2026-09-07T14:34:56Z', 4, 0, 825, 0, 0, 0, 0, NULL, NULL, ?)""",
            (run_id, '{"transfers_endpoint_available": true, "transfers": [], "history": {"current": []}}'),
        )
    return {
        "acquisition_count": conn.execute("SELECT COUNT(*) FROM manager_player_acquisitions").fetchone()[0]
    }


def test_official_lag_without_override_fails_instead_of_using_stale_state(tmp_path):
    """A ledger ahead of official history must not silently fall back to API state.

    The public endpoint has not published a verified manual move, so the official
    free-transfer/bank view is stale.  Without an explicit user-confirmed override
    the planning gate FAILS (rather than quietly planning from the stale numbers);
    recording the override restores PASS.
    """

    conn = connect_database(tmp_path / "fpl.db")
    micros = _seed_ledger_transfer_world(conn)
    before = conn.execute("SELECT COUNT(*) FROM manager_player_acquisitions").fetchone()[0]
    context = get_planning_context(conn, 99, 4, as_of="2026-09-10T16:30:00Z")
    assert context.squad["squad_state"] == "acquisition_ledger"
    assert context.health["status"] == HEALTH_FAIL
    assert any("no explicit user-confirmed override" in reason for reason in context.health["fail_reasons"])
    assert any("ahead of official transfer history" in reason for reason in context.warnings)
    # The manual-provenance acquisition is the explicit current state, and no
    # official evidence was fabricated for it.
    assert conn.execute("SELECT COUNT(*) FROM manager_player_acquisitions").fetchone()[0] == before

    # An explicit override makes the state authoritative and the gate PASS.
    with conn:
        repo.upsert_manual_manager_state(conn, 99, 4, 1, 3, source="user_confirmed_override",
                                         captured_at="2026-09-10T16:20:00Z")
    resolved = get_planning_context(conn, 99, 4, as_of="2026-09-10T16:30:00Z")
    assert resolved.manager_state["authoritative_source"] == "user_confirmed_override"
    assert resolved.manager_state["free_transfers_source"] == "manual"
    assert resolved.health["status"] == HEALTH_PASS
    conn.close()


def test_manual_state_as_of_lookup(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    _seed(conn, eventfinished=True, data_checked=True)
    with conn:
        repo.upsert_manual_manager_state(conn, 241392, 2, 2, 4, captured_at="2026-08-30T20:00:00Z")
    elders = get_planning_context(conn, 241392, 2, as_of="2026-08-30T00:00:00Z")
    assert elders.manager_state["free_transfers"] == 3
    assert elders.manager_state["observation_count"] == 1
    current = get_planning_context(conn, 241392, 2)
    assert current.manager_state["free_transfers"] == 2
    assert current.manager_state["observation_count"] == 2
    conn.close()


def test_selling_price_mismatch_fails_health_gate(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    _seed(conn, eventfinished=True, data_checked=True)
    with conn:
        # Manual snapshot from the same market price disagreeing on selling value.
        repo.upsert_manager_selling_prices(
            conn, 241392, 2, {1: 40}, captured_at="2026-08-29T12:00:00Z", market_prices_at_capture={1: 50}
        )
    context = get_planning_context(conn, 241392, 2)
    assert context.health["status"] == HEALTH_FAIL
    assert any("MISMATCH" in reason for reason in context.health["fail_reasons"])
    conn.close()


def test_manual_state_as_of_lookup(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    _seed(conn, eventfinished=True, data_checked=True)
    with conn:
        repo.upsert_manual_manager_state(conn, 241392, 2, 2, 4, captured_at="2026-08-30T20:00:00Z")
    elders = get_planning_context(conn, 241392, 2, as_of="2026-08-30T00:00:00Z")
    assert elders.manager_state["free_transfers"] == 3
    assert elders.manager_state["observation_count"] == 1
    current = get_planning_context(conn, 241392, 2)
    assert current.manager_state["free_transfers"] == 2
    assert current.manager_state["observation_count"] == 2
    conn.close()


def test_illegal_route_fails_health_gate(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    _seed(conn, eventfinished=True, data_checked=True)
    context = get_planning_context(conn, 241392, 2, route_out=[404], route_in=[2])
    assert context.health["status"] == HEALTH_FAIL
    assert any("illegal" in reason for reason in context.health["fail_reasons"])
    assert context.route_inputs["legal"] is False
    conn.close()


def test_chip_window_availability_and_expiry(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    _seed(conn, eventfinished=True, data_checked=True)
    early = get_planning_context(conn, 241392, 2, as_of="2026-08-30T00:00:00Z")
    wildcard_list = [chip for chip in early.chips if chip["name"] == "wildcard"]
    assert len(wildcard_list) == 2
    first_set, second_set = wildcard_list  # ordered by start_event: GW2-GW19, GW20-GW38
    assert first_set["used"] is False
    assert first_set["available_for_event"] is True
    assert second_set["available_for_event"] is False  # planning event outside its window
    assert second_set["expired"] is False

    # A used chip cannot be recommended later, and a window-expired chip
    # cannot be saved for any future gameweek.  The seed used wildcard set #2
    # at GW21, so set #1 stays available and set #2 is consumed.
    from fpl_brain.planning import chips_state

    late = chips_state(conn, 241392, 22)
    sets = [chip for chip in late if chip["name"] == "wildcard"]
    assert sets[0]["used"] is False and sets[1]["used"] is True
    gone = chips_state(conn, 241392, 40)
    expired_sets = [chip for chip in gone if chip["name"] == "wildcard"]
    assert all(chip["expired"] for chip in expired_sets)
    assert all(chip["available_for_event"] is False for chip in expired_sets)
    conn.close()
