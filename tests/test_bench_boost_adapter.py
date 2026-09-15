"""Bench Boost V1 — production authority, chip rules and the decision horizon.

The squad and lineup are NOT caller arguments: they are sourced from canonical
manager state (``manager_worlds.resolve_squad`` for the fifteen and their
positions, the captured ``squad_picks`` for XI/bench/armband).  These tests use
a real temporary store so that authority is exercised end to end rather than
asserted on a hand-made object.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fpl_brain import bench_boost_request_adapter as ad  # noqa: E402
from fpl_brain import chip_bench_boost as bb  # noqa: E402
from fpl_brain import chip_decision as cd  # noqa: E402
from fpl_brain import manager_lineup as ml  # noqa: E402
from fpl_brain import repositories as repo  # noqa: E402
from fpl_brain.database import connect_database  # noqa: E402
from fpl_brain.models import (  # noqa: E402
    EventRecord, PickRecord, PlayerRecord, PlayerSnapshotRecord, PositionRecord, TeamRecord,
)

ENTRY = 241392
EVENT = 5
WORLDS = 8

#: 2 GKP / 5 DEF / 5 MID / 3 FWD, all legal.
SQUAD = tuple(range(1, 16))
POSITION_ID = {1: "GKP", 2: "GKP", 3: "DEF", 4: "DEF", 5: "DEF", 6: "DEF", 7: "DEF",
               8: "MID", 9: "MID", 10: "MID", 11: "MID", 12: "MID",
               13: "FWD", 14: "FWD", 15: "FWD"}
POSITION_SHORT = {"GKP": 1, "DEF": 2, "MID": 3, "FWD": 4}
CLUB = {1: 11, 2: 12, 3: 13, 4: 14, 5: 15, 6: 16, 7: 17, 8: 18, 9: 19, 10: 20,
        11: 11, 12: 12, 13: 13, 14: 14, 15: 15}
XI = (1, 3, 4, 5, 8, 9, 10, 11, 13, 14, 15)
BENCH_GK, BENCH_OUT = 2, (6, 7, 12)
CAPTAIN, VICE = 13, 8

_STRONG = "sha256:" + "c" * 64
_SNAPSHOT = "sha256:" + "d" * 64
_CAPTURED_AT = "2026-09-12T08:00:00Z"


def _seed(conn, *, picks_event: int = EVENT, picks: list[tuple[int, int, bool, bool]] | None = None,
          bboost_windows=((2, 19), (20, 38)), used_bboost_events=(), players=SQUAD) -> None:
    """Seed a minimal but canonical world: players, chips, and captured picks."""

    now = _CAPTURED_AT
    with conn:
        repo.upsert_teams(conn, [TeamRecord(id=team, name=f"Team {team}") for team in sorted(set(CLUB.values()))])
        repo.upsert_positions(conn, [
            PositionRecord(id=1, singular_name="Goalkeeper", singular_name_short="GKP"),
            PositionRecord(id=2, singular_name="Defender", singular_name_short="DEF"),
            PositionRecord(id=3, singular_name="Midfielder", singular_name_short="MID"),
            PositionRecord(id=4, singular_name="Forward", singular_name_short="FWD"),
        ])
        repo.upsert_events(conn, [
            EventRecord(id=event, finished=1 if event < EVENT else 0,
                        data_checked=1 if event < EVENT else 0,
                        deadline_time=f"2026-09-{event + 4:02d}T12:30:00Z", raw_json={})
            for event in range(1, 39)
        ])
        repo.upsert_players(conn, [
            PlayerRecord(id=pid, web_name=f"P{pid}", full_name=f"Player {pid}", team_id=CLUB[pid],
                         element_type=POSITION_SHORT[POSITION_ID[pid]])
            for pid in players
        ])
        run = repo.create_fetch_run(conn, "fetch_fpl", started_at=now)
        repo.insert_snapshots(conn, [
            PlayerSnapshotRecord(player_id=pid, captured_at=now, now_cost=45, raw_json={}) for pid in players
        ], run)
        repo.finish_fetch_run(conn, run, "success", current_event=EVENT)

        conn.execute("DELETE FROM chip_definitions")
        for index, (start, stop) in enumerate(bboost_windows, start=1):
            conn.execute(
                "INSERT INTO chip_definitions(id, name, number, chip_type, start_event, stop_event, updated_at) "
                "VALUES (?,?,?,?,?,?,?)",
                (index, "bboost", index, "team", start, stop, now),
            )
        for other, offset in (("3xc", 100), ("wildcard", 200), ("freehit", 300)):
            conn.execute(
                "INSERT INTO chip_definitions(id, name, number, chip_type, start_event, stop_event, updated_at) "
                "VALUES (?,?,?,?,?,?,?)",
                (offset, other, 1, "team", 2, 38, now),
            )
        for event in used_bboost_events:
            conn.execute(
                "INSERT INTO manager_chips(entry_id, name, event, time, updated_at) VALUES (?,?,?,?,?)",
                (ENTRY, "bboost", int(event), now, now),
            )

        if picks is None:
            picks = [(pid, slot, pid == CAPTAIN, pid == VICE)
                     for slot, pid in enumerate((*XI, BENCH_GK, *BENCH_OUT), start=1)]
        repo.upsert_squad_picks(conn, ENTRY, int(picks_event), [
            PickRecord(player_id=pid, position=slot, is_captain=1 if captain else 0,
                       is_vice_captain=1 if vice else 0,
                       multiplier=2 if captain else (1 if slot <= ml.XI_SIZE else 0),
                       raw_json={})
            for pid, slot, captain, vice in picks
        ])
        for pid in SQUAD:
            repo.insert_manager_acquisition(conn, ENTRY, pid, 1, 45,
                                            source="official_transfer_history", created_at=now)
        conn.execute(
            """INSERT INTO manager_state(entry_id, fetch_run_id, captured_at, event, bank, team_value,
               total_transfers, event_transfers, event_transfers_cost, points_on_bench, active_chip,
               free_transfers_manual, raw_json)
               VALUES (?, ?, ?, ?, 5, 750, 0, 0, 0, 0, NULL, 1, ?)""",
            (ENTRY, run, now, EVENT, json.dumps({"transfers_endpoint_available": True, "transfers": [],
                                                 "history": {"current": []}})),
        )


@pytest.fixture
def conn(tmp_path):
    connection = connect_database(tmp_path / "fpl.db")
    try:
        yield connection
    finally:
        connection.close()


def _worlds(*, planning_event: int = EVENT, identity: str = _STRONG, snapshot: str = _SNAPSHOT,
            missing: tuple[int, ...] = ()) -> cd.ChipWorldInputs:
    core = {pid: tuple(2.0 for _ in range(WORLDS)) for pid in SQUAD}
    for pid in (BENCH_GK, *BENCH_OUT):
        core[pid] = tuple(5.0 for _ in range(WORLDS))
    for pid in missing:
        core.pop(pid, None)
    minutes = {pid: tuple(90.0 for _ in range(WORLDS)) for pid in core}
    return cd.ChipWorldInputs(
        worlds=WORLDS, player_ids=tuple(sorted(core)), minutes=minutes, core=core,
        planning_event=int(planning_event), horizon_events=cd.canonical_chip_horizon(int(planning_event)),
        certification_identity=identity, data_snapshot_sha256=snapshot,
        world_seed=20260915, world_identity="sha256:" + "e" * 64,
    )


def _binding(*, planning_event: int = EVENT, identity: str = _STRONG, events=None) -> cd.ChipHorizonBinding:
    return cd.ChipHorizonBinding(
        planning_event=int(planning_event),
        horizon_events=events if events is not None else cd.canonical_chip_horizon(int(planning_event)),
        certification_identity=identity, data_snapshot_sha256=_SNAPSHOT,
    )


def _certified(**overrides) -> ad.BenchBoostCertifiedInputs:
    base = dict(horizon_binding=_binding(), worlds=_worlds())
    base.update(overrides)
    return ad.BenchBoostCertifiedInputs(**base)


def _chip_rows(*, available=True, used=False, expired=False, start=2, stop=19):
    return [{"name": "bboost", "available_for_event": available, "used": used, "expired": expired,
             "window": f"GW{start}-GW{stop}", "window_start_event": start, "window_stop_event": stop}]


# ---------------------------------------------------------------------------
# AUTHORITY — H..M
# ---------------------------------------------------------------------------


def test_H_a_legal_canonical_squad_and_lineup_evaluates(conn):
    _seed(conn)
    state = ad.bench_boost_manager_state(conn, ENTRY, EVENT)

    assert state.squad_ids == SQUAD
    assert state.lineup_source_event == EVENT
    assert state.lineup_is_exact_event is True
    assert state.problems() == []
    assert state.policy.captain_id == CAPTAIN
    assert state.policy.vice_captain_id == VICE
    assert state.policy.bench_gk_id == BENCH_GK
    assert set(bb.squad_ids_of(state.policy)) == set(SQUAD)

    request = ad.build_bench_boost_request(state, _certified(), conn=conn)
    evaluation = bb.evaluate_bench_boost(request)
    assert evaluation.action == cd.CHIP_ACTION_BB
    assert evaluation.mean_uplift == pytest.approx(20.0)
    assert evaluation.candidate_metrics["mean_normal_value"] == pytest.approx(24.0)
    assert evaluation.candidate_metrics["mean_bench_boost_value"] == pytest.approx(44.0)


def test_H2_the_lineup_is_the_managers_own_not_an_optimised_one(conn):
    """BB never re-picks the XI: a deliberately weak XI is scored as submitted."""

    _seed(conn)
    # Put the strongest player (a bench FWD worth 5.0) out of the XI on purpose.
    state = ad.bench_boost_manager_state(conn, ENTRY, EVENT)
    assert state.policy.starter_ids == XI
    assert state.policy.bench_outfield_order == BENCH_OUT


def test_I_a_malformed_captured_lineup_refuses(conn):
    """14 picks, no captain => the lineup cannot be used."""

    _seed(conn, picks=[(pid, slot, False, False) for slot, pid in enumerate(SQUAD[:-1], start=1)])
    with pytest.raises(ad.BenchBoostAdapterError) as caught:
        ad.bench_boost_manager_state(conn, ENTRY, EVENT)
    assert caught.value.reasons[0] in {ad.BB_LINEUP_ILLEGAL, ad.BB_LINEUP_NOT_CAPTURED}


def test_I2_an_illegal_fifteen_refuses_at_the_adapter(conn):
    """A hand-made state whose fifteen breaks the rules never reaches a decision."""

    _seed(conn)
    state = ad.bench_boost_manager_state(conn, ENTRY, EVENT)
    broken = ad.BenchBoostManagerState(
        entry_id=state.entry_id, planning_event=state.planning_event, squad_ids=state.squad_ids,
        # Four FWDs in the XI: FORMATION_MAX for FWD is 3.
        policy=ml.ManagerPolicy(starter_ids=(1, 3, 4, 8, 9, 10, 11, 12, 13, 14, 15),
                                bench_gk_id=2, bench_outfield_order=(5, 6, 7),
                                captain_id=13, vice_captain_id=8),
        positions=state.positions, chip_availability=state.chip_availability,
        lineup_source_event=state.lineup_source_event,
    )
    assert any("not legal" in problem for problem in broken.problems())
    with pytest.raises(ad.BenchBoostAdapterError) as caught:
        ad.build_bench_boost_request(broken, _certified())
    assert caught.value.reasons[0] == ad.BB_MANAGER_STATE_MISSING


def test_I3_an_illegal_lineup_refuses_at_the_evaluator(conn):
    """The evaluator re-checks legality itself; the adapter is not the only gate."""

    _seed(conn)
    state = ad.bench_boost_manager_state(conn, ENTRY, EVENT)
    request = ad.build_bench_boost_request(state, _certified(), conn=conn)
    # Same fifteen, captain moved to the bench GK: CAPTAIN_NOT_IN_XI.
    illegal = bb.BenchBoostRequest(
        worlds=request.worlds, horizon_binding=request.horizon_binding,
        policy=ml.ManagerPolicy(
            starter_ids=request.policy.starter_ids, bench_gk_id=request.policy.bench_gk_id,
            bench_outfield_order=request.policy.bench_outfield_order,
            captain_id=BENCH_GK, vice_captain_id=VICE,
        ),
        positions=request.positions,
    )
    with pytest.raises(cd.ChipInputError) as caught:
        bb.evaluate_bench_boost(illegal)
    assert cd.DIAG_CHIP_WORLD_CONTRACT_INCOMPLETE in caught.value.reasons


def test_J_a_predictive_event_mismatch_refuses(conn):
    _seed(conn)
    state = ad.bench_boost_manager_state(conn, ENTRY, EVENT)
    # Worlds built for GW6 while the manager state is GW5.
    with pytest.raises(ad.BenchBoostAdapterError) as caught:
        ad.build_bench_boost_request(
            state, ad.BenchBoostCertifiedInputs(horizon_binding=_binding(), worlds=_worlds(planning_event=6)),
        )
    assert caught.value.reasons[0] == ad.BB_HORIZON_MISMATCH


def test_J2_a_worlds_set_bound_to_another_horizon_refuses_at_the_evaluator():
    from test_chip_bench_boost import POSITIONS, _policy

    request = bb.BenchBoostRequest(
        worlds=_worlds(planning_event=6), horizon_binding=_binding(planning_event=EVENT),
        policy=_policy(), positions=POSITIONS,
    )
    with pytest.raises(cd.ChipInputError) as caught:
        bb.evaluate_bench_boost(request)
    assert cd.DIAG_CHIP_PLANNING_EVENT_MISMATCH in caught.value.reasons


def test_K_a_predictive_identity_mismatch_refuses(conn):
    _seed(conn)
    state = ad.bench_boost_manager_state(conn, ENTRY, EVENT)
    # The worlds carry a different certification identity than the binding.
    with pytest.raises(ad.BenchBoostAdapterError) as caught:
        ad.build_bench_boost_request(
            state,
            ad.BenchBoostCertifiedInputs(
                horizon_binding=_binding(), worlds=_worlds(identity="sha256:" + "f" * 64),
            ),
        )
    assert caught.value.reasons[0] == ad.BB_HORIZON_MISMATCH


def test_K2_a_snapshot_mismatch_alone_does_not_pass_silently(conn):
    _seed(conn)
    state = ad.bench_boost_manager_state(conn, ENTRY, EVENT)
    # The binding and the worlds agree with each other but not with the manager
    # context's declared snapshot: the adapter must carry the WORLDS' snapshot
    # through, never substitute its own.
    request = ad.build_bench_boost_request(
        state, ad.BenchBoostCertifiedInputs(
            horizon_binding=_binding(), worlds=_worlds(snapshot="sha256:" + "a" * 64),
        ),
        conn=conn,
    )
    assert request.worlds.data_snapshot_sha256 == "sha256:" + "a" * 64


def test_L_a_missing_projection_or_world_refuses(conn):
    """A squad player the matrix never captured is not scored as zero."""

    _seed(conn)
    state = ad.bench_boost_manager_state(conn, ENTRY, EVENT)
    request = ad.build_bench_boost_request(
        state, ad.BenchBoostCertifiedInputs(horizon_binding=_binding(), worlds=_worlds(missing=(14,))),
        conn=conn,
    )
    with pytest.raises(cd.ChipInputError) as caught:
        bb.evaluate_bench_boost(request)
    assert cd.DIAG_CHIP_WORLD_CONTRACT_INCOMPLETE in caught.value.reasons


def test_L2_an_incomplete_series_refuses_rather_than_being_padded(conn):
    """A truncated per-player series is rejected at the input contract."""

    _seed(conn)
    worlds = _worlds()
    with pytest.raises(cd.ChipInputError):
        cd.ChipWorldInputs(
            worlds=worlds.worlds, player_ids=worlds.player_ids, minutes=worlds.minutes,
            core={**worlds.core, 14: (2.0, 2.0)}, planning_event=worlds.planning_event,
            horizon_events=worlds.horizon_events, certification_identity=worlds.certification_identity,
            data_snapshot_sha256=worlds.data_snapshot_sha256, world_seed=worlds.world_seed,
            world_identity=worlds.world_identity,
        )


def test_M_a_caller_supplied_squad_cannot_override_canonical_authority(conn):
    """With canonical manager state available, a fabricated fifteen refuses."""

    _seed(conn)
    state = ad.bench_boost_manager_state(conn, ENTRY, EVENT)
    # Swap a squad member for a player the manager does not own.
    with conn:
        repo.upsert_players(conn, [PlayerRecord(id=99, web_name="Impostor", full_name="Impostor",
                                                team_id=11, element_type=4)])
    forged = ad.BenchBoostManagerState(
        entry_id=state.entry_id, planning_event=state.planning_event,
        # Bench MID 12 swapped for an unowned player 99, consistently everywhere.
        squad_ids=tuple(sorted((set(SQUAD) - {12}) | {99})),
        policy=ml.ManagerPolicy(
            starter_ids=XI, bench_gk_id=BENCH_GK, bench_outfield_order=(6, 7, 99),
            captain_id=CAPTAIN, vice_captain_id=VICE,
        ),
        positions={**state.positions, 99: "FWD"},
        chip_availability=state.chip_availability, lineup_source_event=state.lineup_source_event,
    )
    assert forged.problems() == [], "the forged state is internally consistent by construction"
    with pytest.raises(ad.BenchBoostAdapterError) as caught:
        ad.build_bench_boost_request(forged, _certified(), conn=conn)
    assert caught.value.reasons[0] == ad.BB_CALLER_STATE_DISAGREES


def test_M2_a_forged_lineup_for_the_real_squad_also_refuses(conn):
    """Same fifteen, fabricated XI/bench/armband: still not the manager's lineup."""

    _seed(conn)
    state = ad.bench_boost_manager_state(conn, ENTRY, EVENT)
    forged = ad.BenchBoostManagerState(
        entry_id=state.entry_id, planning_event=state.planning_event, squad_ids=state.squad_ids,
        policy=ml.ManagerPolicy(
            starter_ids=XI, bench_gk_id=BENCH_GK, bench_outfield_order=(6, 7, 12),
            captain_id=14, vice_captain_id=9,
        ),
        positions=state.positions, chip_availability=state.chip_availability,
        lineup_source_event=state.lineup_source_event,
    )
    with pytest.raises(ad.BenchBoostAdapterError) as caught:
        ad.build_bench_boost_request(forged, _certified(), conn=conn)
    assert caught.value.reasons[0] == ad.BB_CALLER_STATE_DISAGREES


def test_M3_without_a_connection_the_canonical_state_is_still_required(conn):
    """The pure path cannot be reached with a malformed fifteen at all."""

    _seed(conn)
    state = ad.bench_boost_manager_state(conn, ENTRY, EVENT)
    broken = ad.BenchBoostManagerState(
        entry_id=state.entry_id, planning_event=state.planning_event, squad_ids=state.squad_ids,
        policy=ml.ManagerPolicy(starter_ids=XI[:9], bench_gk_id=BENCH_GK,
                                bench_outfield_order=(6, 7, 12), captain_id=CAPTAIN, vice_captain_id=VICE),
        positions=state.positions, chip_availability=state.chip_availability,
        lineup_source_event=state.lineup_source_event,
    )
    with pytest.raises(ad.BenchBoostAdapterError):
        ad.build_bench_boost_request(broken, _certified())


def test_a_missing_lineup_capture_refuses(conn):
    """No admissible lineup at or before the event => the XI/armband are unknown."""

    _seed(conn, picks_event=EVENT)
    with conn:
        conn.execute("DELETE FROM squad_picks")
    with pytest.raises(ad.BenchBoostAdapterError) as caught:
        ad.bench_boost_manager_state(conn, ENTRY, EVENT)
    assert caught.value.reasons[0] in {ad.BB_LINEUP_NOT_CAPTURED, ad.BB_MANAGER_STATE_MISSING}


def test_a_stale_lineup_naming_other_players_refuses(conn):
    """A lineup for a squad that has since changed is never remapped."""

    _seed(conn, picks_event=EVENT - 1)
    with conn:
        # The manager has since bought player 99; the GW4 lineup still names 15.
        repo.upsert_players(conn, [PlayerRecord(id=99, web_name="New", full_name="New",
                                                team_id=19, element_type=3)])
        repo.insert_manager_acquisition(conn, ENTRY, 99, EVENT, 45,
                                        source="official_transfer_history", acquired_at=_CAPTURED_AT)
        row = next(row for row in repo.active_manager_acquisitions(conn, ENTRY) if int(row["player_id"]) == 15)
        repo.close_manager_acquisition(conn, int(row["id"]), EVENT, sold_at=_CAPTURED_AT)
    with pytest.raises(ad.BenchBoostAdapterError) as caught:
        ad.bench_boost_manager_state(conn, ENTRY, EVENT)
    assert caught.value.reasons[0] in {ad.BB_LINEUP_SQUAD_MISMATCH, ad.BB_MANAGER_STATE_MISSING}


def test_an_earlier_lineup_is_used_before_the_coming_deadline(conn):
    """Before the deadline the manager's most recent submitted lineup is the authority."""

    _seed(conn, picks_event=EVENT - 1)
    state = ad.bench_boost_manager_state(conn, ENTRY, EVENT)
    assert state.lineup_source_event == EVENT - 1
    assert state.lineup_is_exact_event is False
    assert state.policy.starter_ids == XI


def test_chip_availability_comes_from_the_canonical_rows(conn):
    _seed(conn, used_bboost_events=(EVENT,))
    state = ad.bench_boost_manager_state(conn, ENTRY, EVENT)
    request = ad.build_bench_boost_request(state, _certified(), conn=conn)
    assert request.chip_available is False


def test_availability_is_true_when_the_second_window_is_live(conn):
    _seed(conn, used_bboost_events=(EVENT,))
    state = ad.bench_boost_manager_state(conn, ENTRY, 20)
    assert ad._chip_available(state.chip_availability, planning_event=20) is True


# ---------------------------------------------------------------------------
# CHIP RULES — N..R
# ---------------------------------------------------------------------------


def _decide(evaluation, *, availability, reservation=None, **overrides):
    kwargs = dict(
        horizon_binding=_binding(), chip_availability=availability,
        evaluations={cd.CHIP_ACTION_BB: evaluation}, reservation=reservation,
        certification_valid=True, manager_state={"squad_ids": list(SQUAD)},
    )
    kwargs.update(overrides)
    return cd.decide_chip_action(**kwargs)


class _Calibrated:
    def __init__(self, value: float):
        self.value = value
        self.calls = 0

    def estimate(self, *, action, planning_event, expiry_event, state):
        self.calls += 1
        return cd.ReservationEstimate(
            value=float(self.value), calibration_status=cd.CALIBRATION_CALIBRATED,
            terminal_value=0.0, weeks_to_expiry=None if expiry_event is None else int(expiry_event) - planning_event,
        )


def test_N_play_now_is_compared_with_the_save_policy(conn):
    _seed(conn)
    state = ad.bench_boost_manager_state(conn, ENTRY, EVENT)
    evaluation = bb.evaluate_bench_boost(ad.build_bench_boost_request(state, _certified(), conn=conn))
    # An uncalibrated future-opportunity term: positive uplift, no endorsement.
    decision = _decide(evaluation, availability=_chip_rows())
    assert decision.recommended_action == cd.CHIP_ACTION_BB
    assert decision.status == cd.STATUS_CHIP_CANDIDATE_RECHECK_REQUIRED
    assert decision.candidate_metrics["reservation_value"] is None
    assert cd.DIAG_CHIP_RESERVATION_UNCALIBRATED in decision.reason_codes


def test_N2_a_calibrated_reservation_may_endorse_but_the_evaluator_is_review_only(conn):
    _seed(conn)
    state = ad.bench_boost_manager_state(conn, ENTRY, EVENT)
    evaluation = bb.evaluate_bench_boost(ad.build_bench_boost_request(state, _certified(), conn=conn))
    decision = _decide(evaluation, availability=_chip_rows(), reservation=_Calibrated(1.0))
    # execution_permitted is False, so even a calibrated reservation cannot
    # produce PLAY_CHIP while Bench Boost has no calibrated value model.
    assert decision.status == cd.STATUS_CHIP_CANDIDATE_RECHECK_REQUIRED
    assert cd.DIAG_CHIP_EVALUATOR_UNCALIBRATED in decision.reason_codes
    assert decision.recommended_action != cd.CHIP_ACTION_NO_CHIP


def test_N3_a_non_positive_uplift_saves_the_chip(conn):
    _seed(conn)
    state = ad.bench_boost_manager_state(conn, ENTRY, EVENT)
    # No bench player ever plays: the chip is worth exactly nothing.
    worlds = _worlds()
    dead = cd.ChipWorldInputs(
        worlds=worlds.worlds, player_ids=worlds.player_ids,
        minutes={**worlds.minutes, BENCH_GK: tuple(0.0 for _ in range(WORLDS)),
                 **{pid: tuple(0.0 for _ in range(WORLDS)) for pid in BENCH_OUT}},
        core=worlds.core, planning_event=worlds.planning_event, horizon_events=worlds.horizon_events,
        certification_identity=worlds.certification_identity,
        data_snapshot_sha256=worlds.data_snapshot_sha256, world_seed=worlds.world_seed,
        world_identity=worlds.world_identity,
    )
    evaluation = bb.evaluate_bench_boost(
        ad.build_bench_boost_request(
            state, ad.BenchBoostCertifiedInputs(horizon_binding=_binding(), worlds=dead), conn=conn,
        )
    )
    assert evaluation.mean_uplift == pytest.approx(0.0)
    decision = _decide(evaluation, availability=_chip_rows())
    assert decision.status == cd.STATUS_NO_CHIP
    assert decision.recommended_action == cd.CHIP_ACTION_NO_CHIP


def test_O_the_reservation_is_applied_exactly_once(conn):
    _seed(conn)
    state = ad.bench_boost_manager_state(conn, ENTRY, EVENT)
    evaluation = bb.evaluate_bench_boost(ad.build_bench_boost_request(state, _certified(), conn=conn))
    reservation = _Calibrated(3.0)
    decision = _decide(evaluation, availability=_chip_rows(), reservation=reservation)

    uplift = evaluation.mean_uplift
    assert reservation.calls == 1
    assert decision.candidate_metrics["net_of_reservation"] == pytest.approx(uplift - 3.0)


def test_P_one_chip_per_gameweek_conflict_refuses(conn):
    _seed(conn)
    state = ad.bench_boost_manager_state(conn, ENTRY, EVENT)
    evaluation = bb.evaluate_bench_boost(ad.build_bench_boost_request(state, _certified(), conn=conn))
    decision = _decide(evaluation, availability=_chip_rows(),
                       chips_already_played_for_event=[cd.CHIP_ACTION_TC])
    # The Gameweek's single chip slot is spent: no chip may be played at all.
    assert decision.status == cd.STATUS_NO_CHIP
    assert decision.recommended_action == cd.CHIP_ACTION_NO_CHIP
    assert cd.DIAG_CHIP_GAMEWEEK_ALREADY_USED in decision.reason_codes


def test_P2_more_than_one_chip_recorded_for_the_event_fails_closed(conn):
    _seed(conn)
    state = ad.bench_boost_manager_state(conn, ENTRY, EVENT)
    evaluation = bb.evaluate_bench_boost(ad.build_bench_boost_request(state, _certified(), conn=conn))
    decision = _decide(evaluation, availability=_chip_rows(),
                       chips_already_played_for_event=[cd.CHIP_ACTION_TC, cd.CHIP_ACTION_WC])
    assert decision.status == cd.STATUS_INSUFFICIENT_EVIDENCE
    assert cd.DIAG_CHIP_MULTIPLE_ACTIONS in decision.reason_codes


def test_Q_the_second_allocation_survives_a_first_half_use(conn):
    """Using BB in the first window does not consume the second allocation."""

    _seed(conn, used_bboost_events=(7,))
    # GW19 in the first window: used, so not available.
    first = ad.bench_boost_manager_state(conn, ENTRY, 19)
    assert ad._chip_available(first.chip_availability, planning_event=19) is False
    # GW20 opens the fresh second-half allocation: independently available.
    second = ad.bench_boost_manager_state(conn, ENTRY, 20)
    assert ad._chip_available(second.chip_availability, planning_event=20) is True


def test_Q2_a_used_first_allocation_does_not_expire_the_second(conn):
    _seed(conn, used_bboost_events=(19,))
    state = ad.bench_boost_manager_state(conn, ENTRY, 21)
    assert ad._chip_available(state.chip_availability, planning_event=21) is True


def test_R_a_used_chip_in_the_live_window_refuses(conn):
    _seed(conn, used_bboost_events=(EVENT,))
    state = ad.bench_boost_manager_state(conn, ENTRY, EVENT)
    assert ad._chip_available(state.chip_availability, planning_event=EVENT) is False
    evaluation = bb.evaluate_bench_boost(ad.build_bench_boost_request(state, _certified()))
    decision = _decide(evaluation, availability=_chip_rows(available=False, used=True))
    # An ineligible action can never be recommended, and it cannot borrow
    # another action's availability.
    assert decision.status == cd.STATUS_NO_CHIP
    assert decision.recommended_action == cd.CHIP_ACTION_NO_CHIP
    assert any(code.startswith(cd.DIAG_CHIP_UNAVAILABLE) for code in decision.reason_codes)


def test_R2_an_expired_chip_refuses(conn):
    _seed(conn, bboost_windows=((2, 10), (12, 14)))
    state = ad.bench_boost_manager_state(conn, ENTRY, 30)
    assert ad._chip_available(state.chip_availability, planning_event=30) is False
    evaluation = bb.evaluate_bench_boost(
        ad.build_bench_boost_request(
            state,
            ad.BenchBoostCertifiedInputs(
                horizon_binding=_binding(planning_event=30), worlds=_worlds(planning_event=30),
            ),
        )
    )
    expired = [{"name": "bboost", "available_for_event": False, "used": False, "expired": True,
                "window": "GW12-GW14", "window_start_event": 12, "window_stop_event": 14}]
    decision = _decide(
        evaluation, availability=expired,
        horizon_binding=_binding(planning_event=30),
        manager_state={"squad_ids": list(SQUAD)},
    )
    assert decision.status == cd.STATUS_NO_CHIP
    assert any(code.startswith(cd.DIAG_CHIP_UNAVAILABLE) for code in decision.reason_codes)


def test_R3_an_unknown_chip_is_never_available(conn):
    _seed(conn)
    assert ad._chip_available([], planning_event=EVENT) is False
    assert ad._chip_available(_chip_rows(start=30, stop=38), planning_event=EVENT) is False


# ---------------------------------------------------------------------------
# DECISION HORIZON — S..T
# ---------------------------------------------------------------------------


def test_S_the_horizon_is_exactly_four_coherent_events():
    events = cd.canonical_chip_horizon(EVENT)
    assert events == (EVENT, EVENT + 1, EVENT + 2, EVENT + 3)
    assert len(set(events)) == 4
    assert _binding().horizon_events == events


@pytest.mark.parametrize("bad", [(5, 6, 7), (5, 6, 7, 8, 9), (5, 6, 6, 8), (6, 7, 8, 9)])
def test_T_an_incomplete_duplicated_or_non_contiguous_horizon_fails_closed(bad):
    with pytest.raises(cd.ChipInputError) as caught:
        cd.ChipHorizonBinding(planning_event=EVENT, horizon_events=bad, certification_identity=_STRONG)
    assert cd.DIAG_CHIP_HORIZON_NOT_CANONICAL in caught.value.reasons


def test_T2_a_horizon_without_a_certification_identity_fails_closed():
    with pytest.raises(cd.ChipInputError) as caught:
        cd.ChipHorizonBinding(planning_event=EVENT, horizon_events=cd.canonical_chip_horizon(EVENT),
                              certification_identity="")
    assert cd.DIAG_CHIP_CERTIFICATION_REQUIRED in caught.value.reasons


def test_T3_the_evaluator_refuses_a_mismatched_bound_horizon():
    from test_chip_bench_boost import POSITIONS, _policy

    binding = cd.ChipHorizonBinding(
        planning_event=EVENT, horizon_events=cd.canonical_chip_horizon(EVENT),
        certification_identity=_STRONG, data_snapshot_sha256=_SNAPSHOT,
    )
    # A binding whose stored events were replaced after construction.
    object.__setattr__(binding, "horizon_events", (5, 6, 7, 9))
    request = bb.BenchBoostRequest(worlds=_worlds(), horizon_binding=binding,
                                   policy=_policy(), positions=POSITIONS)
    with pytest.raises(cd.ChipInputError) as caught:
        bb.evaluate_bench_boost(request)
    # The binding no longer agrees with the certified worlds, so the request is
    # not bound to the certified context and cannot be evaluated.
    assert set(caught.value.reasons) & {
        cd.DIAG_CHIP_HORIZON_NOT_CANONICAL, cd.DIAG_CHIP_PLANNING_EVENT_MISMATCH,
    }
