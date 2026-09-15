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
          bboost_windows=((2, 19), (20, 38)), used_bboost_events=(), players=SQUAD,
          positions=None, clubs=None) -> None:
    """Seed a minimal but canonical world: players, chips, and captured picks.

    ``positions`` and ``clubs`` override the canonical player labels so a
    deliberately illegal canonical squad can be constructed; ``is_starting`` is
    not settable through ``PickRecord`` because the canonical writer derives it
    from the slot, which is exactly the rule under test.
    """

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
            PlayerRecord(id=pid, web_name=f"P{pid}", full_name=f"Player {pid}",
                         team_id=(clubs if clubs is not None else CLUB)[pid],
                         element_type=POSITION_SHORT[(positions if positions is not None else POSITION_ID)[pid]])
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


def _binding(*, planning_event: int = EVENT, identity: str = _STRONG, events=None,
             snapshot: str = _SNAPSHOT) -> cd.ChipHorizonBinding:
    return cd.ChipHorizonBinding(
        planning_event=int(planning_event),
        horizon_events=events if events is not None else cd.canonical_chip_horizon(int(planning_event)),
        certification_identity=identity, data_snapshot_sha256=snapshot,
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
        positions=state.positions, clubs=state.clubs,
        chip_availability=state.chip_availability, lineup_source_event=state.lineup_source_event,
    )
    assert any("not legal" in problem for problem in broken.problems())
    with pytest.raises(ad.BenchBoostAdapterError) as caught:
        ad.build_bench_boost_request(broken, _certified(), allow_unverified_manager_state=True)
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
            allow_unverified_manager_state=True,
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
            allow_unverified_manager_state=True,
        )
    assert caught.value.reasons[0] == ad.BB_HORIZON_MISMATCH


def test_K2_a_snapshot_mismatch_alone_refuses(conn):
    """The binding snapshot and the world snapshot must be the SAME identity.

    The event window and the certification identity can agree while the data
    snapshot differs, and that used to pass: a world set built from snapshot A
    could be evaluated against a binding authorised for snapshot D.  The mismatch
    is now a refusal, and no numeric evaluation is produced.
    """

    _seed(conn)
    state = ad.bench_boost_manager_state(conn, ENTRY, EVENT)
    other = "sha256:" + "a" * 64
    assert other != _SNAPSHOT
    with pytest.raises(ad.BenchBoostAdapterError) as caught:
        ad.build_bench_boost_request(
            state,
            ad.BenchBoostCertifiedInputs(horizon_binding=_binding(), worlds=_worlds(snapshot=other)),
            allow_unverified_manager_state=True,
        )
    assert caught.value.reasons[0] == ad.BB_HORIZON_MISMATCH
    assert "data snapshot" in str(caught.value)


def test_K3_the_snapshot_mismatch_is_visible_in_the_binding_diagnostic():
    """``matches_worlds`` itself names the snapshot, not just the horizon."""

    problems = _binding().matches_worlds(_worlds(snapshot="sha256:" + "a" * 64))
    assert any("data snapshot" in problem for problem in problems), problems


def test_K4_matching_snapshots_pass():
    assert _binding().matches_worlds(_worlds(snapshot=_SNAPSHOT)) == []


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
    # Swap a bench MID for a MID the manager does not own, so the forged fifteen
    # satisfies the FULL composition and club rules and only the canonical
    # manager state can reveal it.
    with conn:
        repo.upsert_players(conn, [PlayerRecord(id=99, web_name="Impostor", full_name="Impostor",
                                                team_id=11, element_type=3)])
    forged = ad.BenchBoostManagerState(
        entry_id=state.entry_id, planning_event=state.planning_event,
        squad_ids=tuple(sorted((set(SQUAD) - {12}) | {99})),
        policy=ml.ManagerPolicy(
            starter_ids=XI, bench_gk_id=BENCH_GK, bench_outfield_order=(6, 7, 99),
            captain_id=CAPTAIN, vice_captain_id=VICE,
        ),
        positions={**state.positions, 99: "MID"},
        clubs={**state.clubs, 99: 11},
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
        positions=state.positions, clubs=state.clubs,
        chip_availability=state.chip_availability, lineup_source_event=state.lineup_source_event,
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
        positions=state.positions, clubs=state.clubs,
        chip_availability=state.chip_availability, lineup_source_event=state.lineup_source_event,
    )
    with pytest.raises(ad.BenchBoostAdapterError):
        ad.build_bench_boost_request(broken, _certified(), allow_unverified_manager_state=True)


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
    evaluation = bb.evaluate_bench_boost(ad.build_bench_boost_request(state, _certified(), allow_unverified_manager_state=True))
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
            conn=conn,
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


# ---------------------------------------------------------------------------
# P2-A — CANONICAL POSITION / SQUAD-LEGALITY AUTHORITY (Sol repair)
# ---------------------------------------------------------------------------


def _state_with(conn, *, positions, clubs):
    """A state carrying the given labels, so one rule can be isolated.

    Assumes the store is already seeded; it deliberately does NOT re-derive the
    labels, which is the whole point of the forged-state tests below.
    """

    base = ad.bench_boost_manager_state(conn, ENTRY, EVENT)
    return ad.BenchBoostManagerState(
        entry_id=base.entry_id, planning_event=base.planning_event, squad_ids=base.squad_ids,
        policy=base.policy, positions=positions, clubs=clubs,
        chip_availability=base.chip_availability, lineup_source_event=base.lineup_source_event,
    )


def test_P2A_canonical_position_map_passes(conn):
    """The canonical map is accepted and IS the map the evaluator receives."""

    _seed(conn)
    state = ad.bench_boost_manager_state(conn, ENTRY, EVENT)
    assert state.positions == POSITION_ID
    assert state.clubs == CLUB
    assert state.problems() == []
    request = ad.build_bench_boost_request(state, _certified(), conn=conn)
    assert request.positions == POSITION_ID


def test_P2A_a_def_mid_label_swap_on_the_same_fifteen_refuses(conn):
    """Sol's counterexample: identical ids and a superficially legal squad.

    Swapping DEF/MID between bench players 6 and 12 keeps the fifteen, the
    composition (2 GKP / 5 DEF / 5 MID / 3 FWD), the club counts and the starting
    XI legal -- yet it changes which autosubs are legal, and therefore the chip's
    value.  Only canonical authority can reveal it, so it must refuse.
    """

    _seed(conn)
    state = ad.bench_boost_manager_state(conn, ENTRY, EVENT)
    swapped = {**state.positions, 6: "MID", 12: "DEF"}
    assert swapped != state.positions
    forged = _state_with(conn, positions=swapped, clubs=state.clubs)
    # Locally the forged state is perfectly legal -- that is the whole point.
    assert forged.problems() == [], forged.problems()
    assert ml.policy_legality_errors(forged.policy, swapped) == []
    with pytest.raises(ad.BenchBoostAdapterError) as caught:
        ad.build_bench_boost_request(forged, _certified(), conn=conn)
    assert caught.value.reasons[0] == ad.BB_CALLER_STATE_DISAGREES
    assert "position" in str(caught.value)


def test_P2A_a_club_relabel_refuses(conn):
    """The club map is canonical too, not a caller's label."""

    _seed(conn)
    state = ad.bench_boost_manager_state(conn, ENTRY, EVENT)
    forged = _state_with(conn, positions=state.positions, clubs={**state.clubs, 6: 99})
    with pytest.raises(ad.BenchBoostAdapterError) as caught:
        ad.build_bench_boost_request(forged, _certified(), conn=conn)
    assert caught.value.reasons[0] == ad.BB_CALLER_STATE_DISAGREES
    assert "club" in str(caught.value)


def test_P2A_a_legal_2_5_5_3_squad_passes(conn):
    _seed(conn)
    assert _state_with(conn, positions=POSITION_ID, clubs=CLUB).problems() == []


#: Compositions that keep a LEGAL ELEVEN fieldable -- which is exactly why the
#: lineup engine alone never caught them, and why the squad check exists.
_ILLEGAL_BUT_FIELDABLE = {
    "2/4/6/3": {6: "MID", 7: "MID", 12: "DEF"},   # DEF 4, MID 6
    "2/5/4/4": {12: "FWD"},                        # MID 4, FWD 4
}


@pytest.mark.parametrize("label,composition", sorted(_ILLEGAL_BUT_FIELDABLE.items()))
def test_P2A_an_illegal_composition_refuses_even_when_an_xi_is_fieldable(conn, label, composition):
    """Sol's second finding: the FIFTEEN must be legal, not merely an eleven.

    A surplus player simply sits on the bench, so a 2/4/6/3 squad fields a legal
    XI and passed before.  It must refuse now, and the lineup engine must NOT be
    the thing that catches it -- otherwise this test would prove nothing.
    """

    _seed(conn)
    positions = {**POSITION_ID, **composition}
    counts = {}
    for pid in SQUAD:
        counts[positions[pid]] = counts.get(positions[pid], 0) + 1
    assert {k: counts.get(k, 0) for k in ("GKP", "DEF", "MID", "FWD")} != ad.ts.POSITION_COMPOSITION

    state = _state_with(conn, positions=positions, clubs=CLUB)
    assert ml.policy_legality_errors(state.policy, positions) == [], (
        f"{label}: the fixture must keep a legal eleven, or the squad check is untested"
    )
    problems = state.problems()
    assert any("POSITION_INVALID" in problem for problem in problems), (label, problems)
    with pytest.raises(ad.BenchBoostAdapterError) as caught:
        ad.build_bench_boost_request(state, _certified(), allow_unverified_manager_state=True)
    assert caught.value.reasons[0] == ad.BB_MANAGER_STATE_MISSING


def test_P2A_the_composition_matrix_from_the_review(conn):
    """The exact matrix the review asked for, at the squad-legality level."""

    _seed(conn)
    legal = _state_with(conn, positions=POSITION_ID, clubs=CLUB)
    assert legal.problems() == [], "2/5/5/3 must PASS"

    for label, composition in (
        ("2/4/6/3", {6: "MID", 7: "MID"}),
        ("1/6/5/3", {2: "DEF", 6: "DEF", 7: "DEF"}),
    ):
        positions = {**POSITION_ID, **composition}
        counts = {}
        for pid in SQUAD:
            counts[positions[pid]] = counts.get(positions[pid], 0) + 1
        assert len(counts) and sum(counts.values()) == 15
        state = _state_with(conn, positions=positions, clubs=CLUB)
        problems = state.problems()
        assert any("POSITION_INVALID" in problem for problem in problems), (label, problems)
        with pytest.raises(ad.BenchBoostAdapterError):
            ad.build_bench_boost_request(state, _certified(), allow_unverified_manager_state=True)


def test_P2A_the_composition_error_names_the_position(conn):
    _seed(conn)
    state = _state_with(conn, positions={**POSITION_ID, 7: "MID"}, clubs=CLUB)
    problems = state.problems()
    assert any("DEF=4 != 5" in problem for problem in problems), problems
    assert any("MID=6 != 5" in problem for problem in problems), problems


def test_P2A_a_duplicate_player_refuses(conn):
    _seed(conn)
    state = ad.bench_boost_manager_state(conn, ENTRY, EVENT)
    duped = ad.BenchBoostManagerState(
        entry_id=state.entry_id, planning_event=state.planning_event,
        squad_ids=tuple((*state.squad_ids[:-1], state.squad_ids[0])),
        policy=state.policy, positions=state.positions, clubs=state.clubs,
        chip_availability=state.chip_availability, lineup_source_event=state.lineup_source_event,
    )
    assert any("SQUAD_NOT_15_UNIQUE" in problem for problem in duped.problems())
    with pytest.raises(ad.BenchBoostAdapterError):
        ad.build_bench_boost_request(duped, _certified(), allow_unverified_manager_state=True)


def test_P2A_a_wrong_total_count_refuses(conn):
    _seed(conn)
    state = ad.bench_boost_manager_state(conn, ENTRY, EVENT)
    short = ad.BenchBoostManagerState(
        entry_id=state.entry_id, planning_event=state.planning_event, squad_ids=state.squad_ids[:-1],
        policy=state.policy, positions=state.positions, clubs=state.clubs,
        chip_availability=state.chip_availability, lineup_source_event=state.lineup_source_event,
    )
    assert any("SQUAD_NOT_15_UNIQUE" in problem for problem in short.problems())
    with pytest.raises(ad.BenchBoostAdapterError):
        ad.build_bench_boost_request(short, _certified(), allow_unverified_manager_state=True)


def test_P2A_four_players_from_one_club_refuse(conn):
    """Sol's third finding: the club limit needs canonical club ids.

    ``manager_worlds.resolve_squad`` discards them, so before the repair the
    three-per-club rule was uncheckable and four teammates passed.  The four span
    the XI and the bench, so the refusal cannot depend on the arrangement.
    """

    clubs = {**CLUB, 3: 11, 4: 11}   # players 1, 11, 3, 4 -> four at club 11
    _seed(conn, clubs=clubs)
    with pytest.raises(ad.BenchBoostAdapterError) as caught:
        ad.bench_boost_manager_state(conn, ENTRY, EVENT)
    assert caught.value.reasons[0] == ad.BB_MANAGER_STATE_MISSING
    assert "CLUB_LIMIT_EXCEEDED" in str(caught.value)
    assert "CLUB_LIMIT_EXCEEDED: {11: 4}" in str(caught.value)


def test_P2A_exactly_three_from_one_club_passes(conn):
    """The limit is <= 3, so three is legal and must not be refused."""

    clubs = {**CLUB, 3: 11}          # players 1, 11, 3 -> exactly three at club 11
    _seed(conn, clubs=clubs)
    state = ad.bench_boost_manager_state(conn, ENTRY, EVENT)
    assert state.problems() == []
    request = ad.build_bench_boost_request(state, _certified(), conn=conn)
    assert bb.evaluate_bench_boost(request).action == cd.CHIP_ACTION_BB


def test_P2A_the_club_ids_survive_the_canonical_read(conn):
    """The club map is retrieved, not defaulted away."""

    _seed(conn)
    state = ad.bench_boost_manager_state(conn, ENTRY, EVENT)
    assert set(state.clubs) == set(SQUAD)
    assert set(state.clubs.values()) == set(CLUB.values())


def test_P2A_an_unknown_position_or_club_refuses(conn):
    """An unknown label must never satisfy a composition count."""

    _seed(conn)
    state = ad.bench_boost_manager_state(conn, ENTRY, EVENT)
    positions = {pid: value for pid, value in state.positions.items() if pid != 7}
    assert any("UNKNOWN_POSITION" in problem for problem in
               _state_with(conn, positions=positions, clubs=state.clubs).problems())

    clubs = {pid: value for pid, value in state.clubs.items() if pid != 7}
    assert any("UNKNOWN_CLUB" in problem for problem in
               _state_with(conn, positions=state.positions, clubs=clubs).problems())


def test_P2A_the_player_authority_describes_every_squad_member(conn):
    """A squad member the player authority cannot describe refuses, never defaults."""

    _seed(conn)
    with pytest.raises(ad.BenchBoostAdapterError) as caught:
        ad._canonical_positions_and_clubs(conn, [*SQUAD, 987654])
    assert caught.value.reasons[0] == ad.BB_MANAGER_STATE_MISSING
    assert "987654" in str(caught.value)


def test_P2A_the_shared_helper_matches_the_constant_authority():
    """One rule set: the helper consumes the module the transition path owns."""

    assert ad.ts.POSITION_COMPOSITION == {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
    assert ad.ts.SQUAD_TEAM_LIMIT == 3
    assert ad.ts.SQUAD_SIZE == 15
    assert ad.ts.squad_composition_errors(SQUAD, POSITION_ID, CLUB) == []


def test_P2A_the_shared_helper_rejects_every_illegal_shape():
    """The helper itself, exercised directly, so its own rule is pinned."""

    assert ad.ts.squad_composition_errors(SQUAD, {**POSITION_ID, 7: "MID"}, CLUB) != []
    assert ad.ts.squad_composition_errors(SQUAD[:14], POSITION_ID, CLUB) != []
    assert ad.ts.squad_composition_errors((*SQUAD[:14], SQUAD[0]), POSITION_ID, CLUB) != []
    assert ad.ts.squad_composition_errors(SQUAD, POSITION_ID, {**CLUB, 2: 11, 3: 11}) != []
    # Unknown labels never count towards a required position or club.
    assert ad.ts.squad_composition_errors(SQUAD, {**POSITION_ID, 7: "COACH"}, CLUB) != []
    assert ad.ts.squad_composition_errors(SQUAD, POSITION_ID, {k: v for k, v in CLUB.items() if k != 7}) != []


# ---------------------------------------------------------------------------
# P2-B — SNAPSHOT IDENTITY COHERENCE (Sol repair)
# ---------------------------------------------------------------------------

_SNAP_A = "sha256:" + "a" * 64
_SNAP_B = "sha256:" + "b" * 64


def _evaluation_with_snapshot(evaluation: cd.ChipEvaluation, snapshot: str) -> cd.ChipEvaluation:
    return cd.ChipEvaluation(
        action=evaluation.action, evaluator_version=evaluation.evaluator_version,
        candidate_metrics=dict(evaluation.candidate_metrics),
        uncertainty=dict(evaluation.uncertainty), reason_codes=tuple(evaluation.reason_codes),
        calibration_status=evaluation.calibration_status,
        evidence={**evaluation.evidence, "data_snapshot_sha256": snapshot},
        execution_permitted=evaluation.execution_permitted,
    )


def _binding_held_at(snapshot: str) -> cd.ChipHorizonBinding:
    return cd.ChipHorizonBinding(
        planning_event=EVENT, horizon_events=cd.canonical_chip_horizon(EVENT),
        certification_identity=_STRONG, data_snapshot_sha256=snapshot,
    )


def _bb_evaluation(conn, snapshot: str = _SNAP_A):
    """A real BB evaluation whose binding, worlds and evidence all use ``snapshot``."""

    _seed(conn)
    state = ad.bench_boost_manager_state(conn, ENTRY, EVENT)
    request = ad.build_bench_boost_request(
        state,
        ad.BenchBoostCertifiedInputs(
            horizon_binding=_binding_held_at(snapshot), worlds=_worlds(snapshot=snapshot),
        ),
        conn=conn,
    )
    return bb.evaluate_bench_boost(request)


def test_P2B_a_coherent_snapshot_family_passes(conn):
    """binding = world = evaluation = arbiter context, all snapshot A."""

    evaluation = _bb_evaluation(conn, _SNAP_A)
    assert evaluation.evidence["data_snapshot_sha256"] == _SNAP_A
    decision = _decide(evaluation, availability=_chip_rows(), horizon_binding=_binding_held_at(_SNAP_A))
    assert decision.recommended_action == cd.CHIP_ACTION_BB
    assert decision.evidence["data_snapshot_sha256"] == _SNAP_A
    assert cd.DIAG_CHIP_EVALUATION_CONTEXT_MISMATCH not in decision.reason_codes
    assert decision.candidate_metrics["mean_paired_uplift"] == pytest.approx(20.0)


def test_P2B_a_binding_world_snapshot_mismatch_refuses(conn):
    """binding = A, world = B: no numeric evaluation may be produced."""

    _seed(conn)
    state = ad.bench_boost_manager_state(conn, ENTRY, EVENT)
    with pytest.raises(ad.BenchBoostAdapterError) as caught:
        ad.build_bench_boost_request(
            state,
            ad.BenchBoostCertifiedInputs(
                horizon_binding=_binding_held_at(_SNAP_A), worlds=_worlds(snapshot=_SNAP_B),
            ),
            allow_unverified_manager_state=True,
        )
    assert caught.value.reasons[0] == ad.BB_HORIZON_MISMATCH
    assert "data snapshot" in str(caught.value)


def test_P2B_an_evaluation_from_another_snapshot_refuses_before_arbitration(conn):
    """binding = A, evaluation = B: the arbiter must not admit it."""

    evaluation = _bb_evaluation(conn, _SNAP_A)
    assert evaluation.evidence["data_snapshot_sha256"] == _SNAP_A
    foreign = _evaluation_with_snapshot(evaluation, _SNAP_B)
    decision = _decide(foreign, availability=_chip_rows(), horizon_binding=_binding_held_at(_SNAP_A))
    assert decision.status == cd.STATUS_INSUFFICIENT_EVIDENCE
    assert decision.recommended_action == cd.CHIP_ACTION_NO_CHIP
    assert cd.DIAG_CHIP_EVALUATION_CONTEXT_MISMATCH in decision.reason_codes
    assert cd.DIAG_CHIP_DATA_SNAPSHOT_MISMATCH in decision.reason_codes
    assert decision.candidate_metrics == {}, "no numeric evaluation may reach arbitration"


def test_P2B_the_same_evaluation_is_admitted_when_the_snapshot_coheres(conn):
    """The control: the identical evaluation passes when its snapshot matches.

    Without this, the refusal above could be caused by anything else about the
    evaluation rather than by the snapshot it declares.
    """

    evaluation = _bb_evaluation(conn, _SNAP_A)
    coherent = _evaluation_with_snapshot(evaluation, _SNAP_A)
    decision = _decide(coherent, availability=_chip_rows(), horizon_binding=_binding_held_at(_SNAP_A))
    assert cd.DIAG_CHIP_EVALUATION_CONTEXT_MISMATCH not in decision.reason_codes
    assert decision.candidate_metrics["mean_paired_uplift"] == pytest.approx(20.0)


def test_P2B_a_snapshot_disagreement_is_reported_as_a_snapshot_problem():
    """The cause is machine-readable, not merely a horizon message."""

    binding = _binding_held_at(_SNAP_A)
    problems = binding.matches_worlds(_worlds(snapshot=_SNAP_B))
    assert any("data snapshot" in problem for problem in problems)
    assert not any("horizon" in problem for problem in problems)
    assert binding.matches_worlds(_worlds(snapshot=_SNAP_A)) == []


def test_P2B_a_binding_without_a_snapshot_makes_no_claim():
    """A binding that declares no snapshot is not a claim about any snapshot."""

    binding = cd.ChipHorizonBinding(
        planning_event=EVENT, horizon_events=cd.canonical_chip_horizon(EVENT),
        certification_identity=_STRONG, data_snapshot_sha256=None,
    )
    assert binding.matches_worlds(_worlds(snapshot=_SNAP_B)) == []


def test_P2B_the_four_event_horizon_has_one_snapshot_authority(conn):
    """No partial snapshot family exists for BB, and the one it has is checked.

    Bench Boost's world input is a single certified matrix bound to the four-event
    horizon (Triple Captain's shape), so a "three events on A, one on B" family is
    not representable: there is exactly one snapshot for the horizon, and the
    binding must agree with it.  A caller cannot smuggle a per-event variant in,
    because there is no per-event channel to smuggle it through -- the only
    degrees of freedom are binding vs worlds, and that pair is now compared.
    """

    evaluation = _bb_evaluation(conn, _SNAP_A)
    assert len(evaluation.evidence["horizon_events"]) == 4
    assert evaluation.evidence["horizon_events"] == list(cd.canonical_chip_horizon(EVENT))
    assert evaluation.evidence["data_snapshot_sha256"] == _SNAP_A

    # And the mismatching variant refuses rather than evaluating three events.
    state = ad.bench_boost_manager_state(conn, ENTRY, EVENT)
    with pytest.raises(ad.BenchBoostAdapterError):
        ad.build_bench_boost_request(
            state,
            ad.BenchBoostCertifiedInputs(
                horizon_binding=_binding_held_at(_SNAP_A), worlds=_worlds(snapshot=_SNAP_B),
            ),
            allow_unverified_manager_state=True,
        )


def test_P2A_a_state_without_canonical_authority_refuses_by_default(conn):
    """The production API fails closed: no connection, no authority, no request.

    Positions, clubs and the lineup are canonical facts.  A caller holding only a
    hand-made state has nothing to assert them from, so building a request from
    one is refused unless the caller explicitly declares a simulation.
    """

    _seed(conn)
    state = ad.bench_boost_manager_state(conn, ENTRY, EVENT)
    with pytest.raises(ad.BenchBoostAdapterError) as caught:
        ad.build_bench_boost_request(state, _certified())
    assert caught.value.reasons[0] == ad.BB_CANONICAL_AUTHORITY_REQUIRED
    assert "canonical manager state" in str(caught.value)

    # The explicit opt-out is the only way through the pure path.
    request = ad.build_bench_boost_request(
        state, _certified(), allow_unverified_manager_state=True,
    )
    assert request.positions == POSITION_ID


def test_P2A_the_relabel_cannot_pass_while_canonical_authority_exists(conn):
    """No combination of flags lets a relabel through when the store is present."""

    _seed(conn)
    state = ad.bench_boost_manager_state(conn, ENTRY, EVENT)
    forged = _state_with(conn, positions={**state.positions, 6: "MID", 12: "DEF"}, clubs=state.clubs)
    for kwargs in ({}, {"allow_unverified_manager_state": True}):
        with pytest.raises(ad.BenchBoostAdapterError) as caught:
            ad.build_bench_boost_request(forged, _certified(), conn=conn, **kwargs)
        assert caught.value.reasons[0] == ad.BB_CALLER_STATE_DISAGREES
