"""Rolling four-Gameweek decision-layer tests (regression tests A-J).

Fast and synthetic: no Monte Carlo, no production optimizer, no predictive runs.
The rules under test are the decision-layer product rules, not the models.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from fpl_brain import candidate_universe as cu
from fpl_brain import four_gw_decision as fg
from fpl_brain import manager_worlds
from fpl_brain import repositories as repo
from fpl_brain import route_comparator as rc
from fpl_brain import transfer_state as ts
from fpl_brain.database import connect_database
from fpl_brain.models import (
    ChipRecord,
    EventRecord,
    FixtureRecord,
    PickRecord,
    PlayerRecord,
    PlayerSnapshotRecord,
    PositionRecord,
    TeamRecord,
)
from fpl_brain.planning import get_planning_context
from test_transfer_state import CLUB, POOL_CLUB, POOL_POSITION, POSITION, SQUAD_IDS, _meta, _state

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_UNIVERSE = REPO_ROOT / "data/exports/operational_refresh/corrected_gw04/fresh_candidate_universe_gw04.json"


def _support(events, cutoff="C1", supported=True):
    return {
        int(event): {"event": int(event), "supported": bool(supported), "data_cutoff": cutoff}
        for event in events
    }


# ---------------------------------------------------------------------------
# A. Rolling window
# ---------------------------------------------------------------------------


def test_a_rolling_window_is_current_plus_three():
    assert fg.DECISION_HORIZON_LENGTH == 4
    assert fg.decision_events(4) == (4, 5, 6, 7)
    assert fg.decision_events(9) == (9, 10, 11, 12)
    # Never pad with non-existent Gameweeks.
    assert fg.decision_events(35) == (35, 36, 37, 38)
    assert fg.decision_events(36) == (36, 37, 38)
    assert fg.decision_events(37) == (37, 38)
    assert fg.decision_events(38) == (38,)
    assert fg.lineup_events(4) == (4,)
    assert fg.lineup_events(9) == (9,)
    assert fg.lineup_events(38) == (38,)
    with pytest.raises(fg.HorizonError):
        fg.decision_events(0)
    with pytest.raises(fg.HorizonError):
        fg.lineup_events(-1)


# ---------------------------------------------------------------------------
# B. Readiness block: GW4-6 supported, GW7 missing
# ---------------------------------------------------------------------------


def test_b_missing_final_event_blocks_normal_transfer_recommendation():
    support = _support([4, 5, 6])
    horizon = fg.evaluate_horizon(planning_event=4, support_by_event=support, cutoff="C1")
    assert horizon["status"] == fg.DECISION_HORIZON_INCOMPLETE
    assert horizon["blocked_events"] == [7]
    assert horizon["supported_events"] == [4, 5, 6]
    assert horizon["missing_prediction_events"] == [7]
    assert not fg.transfer_recommendation_allowed(horizon)

    routes = [{
        "route_id": "ROUTE_X", "valid": True,
        "per_event": [{"event": e, "mean_gross_core": 10.0, "hit_points": 0} for e in (4, 5, 6, 7)],
        "terminal_ft": 1, "terminal_bank_tenths": 5,
    }]
    decision = fg.evaluate_four_gw_decision(
        planning_event=4, support_by_event=support, cutoff="C1", routes=routes,
    )
    assert decision["transfer_recommendation"]["status"] == fg.RECOMMENDATION_SUPPRESSED
    assert decision["transfer_recommendation"]["preferred_route_id"] is None
    assert decision["transfer_recommendation"]["ranking"] == []
    assert decision["decision_board"] == []
    assert decision["no_recommendation"] is True
    assert decision["transfer_decision"]["decision_events"] == [4, 5, 6, 7]
    assert "incomplete" in decision["transfer_recommendation"]["message"]


def test_b_complete_horizon_emits_four_gw_recommendation():
    support = _support([4, 5, 6, 7])
    horizon = fg.evaluate_horizon(planning_event=4, support_by_event=support, cutoff="C1")
    assert horizon["status"] == fg.DECISION_HORIZON_COMPLETE
    assert fg.transfer_recommendation_allowed(horizon)
    routes = [
        {"route_id": "ROLL", "valid": True,
         "per_event": [{"event": e, "mean_gross_core": 10.0, "hit_points": 0} for e in (4, 5, 6, 7)],
         "terminal_ft": 5, "terminal_bank_tenths": 3},
        {"route_id": "AGGRESSIVE", "valid": True,
         "per_event": [{"event": e, "mean_gross_core": 12.0, "hit_points": 0} for e in (4, 5, 6, 7)],
         "terminal_ft": 1, "terminal_bank_tenths": 0},
    ]
    decision = fg.evaluate_four_gw_decision(planning_event=4, support_by_event=support, cutoff="C1", routes=routes)
    assert decision["transfer_recommendation"]["status"] == fg.RECOMMENDATION_AVAILABLE
    assert decision["transfer_recommendation"]["preferred_route_id"] == "AGGRESSIVE"
    assert decision["transfer_recommendation"]["objective"] == "FOUR_GW_NET_CORE"
    assert len(decision["decision_board"]) == 2


# ---------------------------------------------------------------------------
# C. Same-cutoff requirement
# ---------------------------------------------------------------------------


def _insert_run(conn, family, event, cutoff, status="complete"):
    conn.execute(
        "INSERT INTO projection_runs(model_family, model_version, generated_at, planning_event, "
        "data_cutoff, status) VALUES (?,?,?,?,?,?)",
        (family, "test_v1", "2026-09-11T00:00:00Z", int(event), cutoff, status),
    )


def test_c_stale_future_cutoffs_never_count_as_support():
    support = {
        4: {"supported": True, "data_cutoff": "FRESH"},
        5: {"supported": True, "data_cutoff": "OLD"},
        6: {"supported": True, "data_cutoff": "OLD"},
        7: {"supported": True, "data_cutoff": "OLD"},
    }
    horizon = fg.evaluate_horizon(planning_event=4, support_by_event=support, cutoff="FRESH")
    assert horizon["status"] == fg.DECISION_HORIZON_INCOMPLETE
    assert horizon["stale_cutoff_events"] == [5, 6, 7]
    assert horizon["events"]["5"]["reason"] == "STALE_CUTOFF_MISMATCH"
    assert horizon["supported_events"] == [4]


def test_c_event_support_from_db_requires_same_cutoff_and_all_families(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    try:
        with conn:
            # GW4 is fresh; GW5/GW6 exist only at an older cutoff (stale); GW7
            # has no Monte Carlo run at all (a missing predictive family).
            for event in (4, 5, 6):
                for family in fg.REQUIRED_HORIZON_FAMILIES:
                    _insert_run(conn, family, event, "FRESH" if event == 4 else "OLD")
            for family in fg.REQUIRED_HORIZON_FAMILIES:
                if family == "monte_carlo_v1":
                    continue
                _insert_run(conn, family, 7, "FRESH")
        support = fg.event_support_from_db(conn, (4, 5, 6, 7), "FRESH")
        assert support[4]["supported"] is True
        assert support[5]["supported"] is False
        assert "minutes_v1" in support[5]["stale_families"]
        assert support[7]["supported"] is False
        assert "monte_carlo_v1" in support[7]["missing_families"]
        horizon = fg.evaluate_horizon(planning_event=4, support_by_event=support, cutoff="FRESH")
        assert horizon["status"] == fg.DECISION_HORIZON_INCOMPLETE
        assert horizon["blocked_events"] == [5, 6, 7]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# D. Full candidate eligibility
# ---------------------------------------------------------------------------


def _synthetic_universe(events=(4, 5, 6, 7)):
    pool = {
        pid: {"player_id": pid, "position": POSITION.get(pid) or POOL_POSITION[pid],
              "club_id": CLUB.get(pid) or POOL_CLUB[pid], "web_name": f"W{pid}", "full_name": f"P{pid}"}
        for pid in sorted(set(SQUAD_IDS) | set(POOL_POSITION))
    }
    clubs: dict[int, list[int]] = {}
    for pid, meta in pool.items():
        clubs.setdefault(int(meta["club_id"]), []).append(pid)
    fixtures = {(event, club): [1000 + club] for event in events for club in clubs}
    core = {pid: 5.0 for pid in pool}
    core.update({11: 0.0, 41: 14.0, 42: 15.0, 51: 13.0})  # 41/42 are unowned, strong, never in a finalist set
    xpts = {event: {(pid, 1000 + int(pool[pid]["club_id"])): {"core_xpts": core[pid], "expected_minutes": 90.0,
                                                              "p_start": 1.0, "p_60_plus": 1.0, "risk_flags": []}
                    for pid in pool} for event in events}
    minutes = {event: {(pid, 1000 + int(pool[pid]["club_id"])): {"joint_availability": 1.0} for pid in pool}
               for event in events}
    prices = {pid: 50 for pid in pool}
    prices.update({41: 45, 42: 45, 51: 45})
    universe = cu.build_universe(
        pool=pool, events_fixtures=fixtures, xpts_rows_by_event=xpts, minutes_rows_by_event=minutes,
        events=list(events), owned_ids=list(SQUAD_IDS), price_snapshot=ts.PriceSnapshot(event=4, prices=prices),
        config=cu.CandidateConfig(top_n_per_criterion=3), planning_cutoff="TEST",
    )
    state = _state(ft=2, bank=0)
    meta = _meta()
    edges = cu.build_replacement_edges(
        universe_rows=universe["universe"], owned_ids=SQUAD_IDS, state=state,
        price_snapshot=ts.PriceSnapshot(event=4, prices=prices), player_meta=meta,
    )
    return universe, edges


def test_d_every_legal_universe_player_is_screened_without_being_named():
    universe, edges = _synthetic_universe()
    screen = fg.screen_legal_actions(
        universe_rows=universe["universe"], replacement_edges=edges, owned_ids=SQUAD_IDS,
        decision_events_window=fg.decision_events(4),
    )
    assert screen["enumeration_exhaustive"] is True
    assert screen["coverage"]["every_legal_edge_screened"] is True
    assert screen["coverage"]["every_legal_in_player_eligible"] is True
    # Player 41 (an unowned pool defender, never part of any historical finalist
    # list) is discovered automatically from the fresh universe.
    assert 41 in screen["discoverable_in_player_ids"]
    assert 41 in set(screen["eligible_in_player_ids"])
    pool_ids = {int(edge["in_player_id"]) for edge in screen["promotion_pool"]}
    assert 41 in pool_ids
    assert screen["legal_single_transfers"] > 0


@pytest.mark.skipif(not REAL_UNIVERSE.exists(), reason="fresh candidate universe artifact not present")
def test_d_real_example_players_are_discoverable_from_the_fresh_universe():
    artifact = json.loads(REAL_UNIVERSE.read_text(encoding="utf-8"))
    rows = artifact["universe"]
    edges = artifact.get("replacement_edges") or []
    owned = [int(row["player_id"]) for row in rows if row.get("owned")]
    screen = fg.screen_legal_actions(
        universe_rows=rows, replacement_edges=edges, owned_ids=owned,
        decision_events_window=fg.decision_events(4),
    )
    # Examples named by the operator; production logic does not reference them.
    for player_id in (154, 40, 379):  # Cole Palmer, Morgan Rogers, Alexander Isak
        assert player_id in screen["discoverable_in_player_ids"], f"{player_id} not discoverable"
        assert player_id in set(screen["eligible_in_player_ids"])
    assert screen["enumeration_exhaustive"] is True


# ---------------------------------------------------------------------------
# E. Four-GW ranking
# ---------------------------------------------------------------------------


def test_e_four_gw_net_objective_outranks_better_current_gw():
    route_a = {  # best current GW, nothing after
        "route_id": "A_GW4_ONLY",
        "per_event": [{"event": 4, "mean_gross_core": 20.0, "hit_points": 0},
                      {"event": 5, "mean_gross_core": 0.0, "hit_points": 0},
                      {"event": 6, "mean_gross_core": 0.0, "hit_points": 0},
                      {"event": 7, "mean_gross_core": 0.0, "hit_points": 0}],
        "terminal_ft": 1, "terminal_bank_tenths": 0,
    }
    route_b = {  # worse current GW, better four-GW structure
        "route_id": "B_FOUR_GW",
        "per_event": [{"event": 4, "mean_gross_core": 15.0, "hit_points": 0},
                      {"event": 5, "mean_gross_core": 6.0, "hit_points": 0},
                      {"event": 6, "mean_gross_core": 6.0, "hit_points": 0},
                      {"event": 7, "mean_gross_core": 6.0, "hit_points": 0}],
        "terminal_ft": 1, "terminal_bank_tenths": 0,
    }
    values = {route["route_id"]: fg.four_gw_window_value(route["per_event"]) for route in (route_a, route_b)}
    assert values["A_GW4_ONLY"]["four_gw_net_core"] == pytest.approx(20.0)
    assert values["B_FOUR_GW"]["four_gw_net_core"] == pytest.approx(33.0)
    ranked = fg.rank_routes_by_four_gw(
        [{**route, **values[route["route_id"]]} for route in (route_a, route_b)]
    )
    assert [row["route_id"] for row in ranked] == ["B_FOUR_GW", "A_GW4_ONLY"]


def test_e_hits_are_subtracted_from_the_four_gw_objective():
    per_event = [{"event": 4, "mean_gross_core": 40.0, "hit_points": 4},
                 {"event": 5, "mean_gross_core": 5.0, "hit_points": 0},
                 {"event": 6, "mean_gross_core": 5.0, "hit_points": 0},
                 {"event": 7, "mean_gross_core": 5.0, "hit_points": 0}]
    value = fg.four_gw_window_value(per_event)
    assert value["four_gw_gross_core"] == pytest.approx(55.0)
    assert value["total_hit_points"] == 4
    assert value["four_gw_net_core"] == pytest.approx(51.0)
    # An out-of-window event is ignored.
    assert fg.four_gw_window_value(per_event + [{"event": 8, "mean_gross_core": 99.0, "hit_points": 0}],
                                   events=[4, 5, 6, 7])["four_gw_net_core"] == pytest.approx(51.0)


# ---------------------------------------------------------------------------
# F. Hit accounting with zero free transfers
# ---------------------------------------------------------------------------


def test_f_zero_free_transfers_costs_four_exactly_once():
    state = _state(ft=0, bank=0)
    batch = ts.TransferBatch((ts.TransferAction(11, 41),))
    result = ts.apply_transfer_batch(state, batch, ts.PriceSnapshot(event=4, prices={41: 45, 11: 50}), _meta())
    assert result.ok, result.errors
    assert result.paid_transfers == 1
    assert result.hit_points == 4
    assert result.ft_used == 0
    # The hit is charged once into the cumulative total, and next GW grants one FT.
    assert result.squad_after.cumulative_hit_points == 4
    assert result.next_event_state.cumulative_hit_points == 4
    assert result.next_event_state.free_transfers == 1


# ---------------------------------------------------------------------------
# G. Already-executed moves are not charged again
# ---------------------------------------------------------------------------


def test_g_preexecuted_transfer_is_represented_not_recharged():
    # A prior GW4 move (11 -> 41) is already reflected in the starting squad,
    # bank and remaining FT.  The decision-layer state must not charge it again.
    players = tuple(
        ts.RoutePlayer(pid, POSITION.get(pid) or POOL_POSITION[pid], CLUB.get(pid) or POOL_CLUB[pid], 45)
        for pid in [p for p in SQUAD_IDS if p != 11] + [41]
    )
    state = ts.RouteState(event=4, players=players, bank_tenths=7, free_transfers=0)
    assert 41 in state.by_id() and 11 not in state.by_id()
    assert state.position_counts()["DEF"] == 5
    assert max(state.club_counts().values()) <= 3
    assert state.cumulative_hit_points == 0

    # One further current-GW transfer costs exactly one hit, not two.
    batch = ts.TransferBatch((ts.TransferAction(12, 42),))
    result = ts.apply_transfer_batch(state, batch, ts.PriceSnapshot(event=4, prices={12: 50, 42: 45}), _meta())
    assert result.ok, result.errors
    assert result.hit_points == 4
    assert result.squad_after.cumulative_hit_points == 4
    assert 41 in result.squad_after.by_id()  # the earlier signing survives untouched


# ---------------------------------------------------------------------------
# H. Real regression manager state (0 FT / £0.7m / Davis owned / Rodon absent)
# ---------------------------------------------------------------------------

_REGRESSION_SQUAD = [1, 2, 13, 14, 15, 21, 22, 23, 24, 25, 31, 32, 33, 41, 42]
_REGRESSION_SELL_OUT = [11, 12]  # already-executed outgoing players
_REGRESSION_EXTRA = [43]  # an unowned DEF candidate used by the "next transfer" check


def _seed_regression_manager(conn, entry_id: int = 241392, event: int = 4):
    ids = sorted(set(_REGRESSION_SQUAD) | set(_REGRESSION_SELL_OUT) | set(_REGRESSION_EXTRA))
    team_ids = sorted({int(CLUB.get(pid) or POOL_CLUB[pid]) for pid in ids})
    with conn:
        repo.upsert_teams(conn, [TeamRecord(id=team_id, name=f"Team {team_id}") for team_id in team_ids])
        repo.upsert_positions(conn, [
            PositionRecord(id=1, singular_name="Goalkeeper", singular_name_short="GKP"),
            PositionRecord(id=2, singular_name="Defender", singular_name_short="DEF"),
            PositionRecord(id=3, singular_name="Midfielder", singular_name_short="MID"),
            PositionRecord(id=4, singular_name="Forward", singular_name_short="FWD"),
        ])
        repo.upsert_chips(conn, [ChipRecord(id=1, name="wildcard", number=1, chip_type="transfer",
                                            start_event=2, stop_event=19)])
        repo.upsert_events(conn, [EventRecord(id=1, finished=1, data_checked=1,
                                              deadline_time="2026-08-21T17:30:00Z", raw_json={}),
                                  EventRecord(id=event, finished=0, data_checked=0,
                                              deadline_time="2026-09-12T12:30:00Z", raw_json={})])
        repo.upsert_fixtures(conn, [FixtureRecord(id=41, event=event, team_h=1, team_a=2,
                                                  started=0, finished=0, raw_json={})])
        repo.upsert_players(conn, [
            PlayerRecord(id=pid, web_name=f"P{pid}", full_name=f"Player {pid}",
                         team_id=CLUB.get(pid) or POOL_CLUB[pid],
                         element_type={"GKP": 1, "DEF": 2, "MID": 3, "FWD": 4}[POSITION.get(pid) or POOL_POSITION[pid]])
            for pid in ids
        ])
        run = repo.create_fetch_run(conn, "fetch_fpl", started_at="2026-09-11T20:00:00Z")
        repo.insert_snapshots(conn, [
            PlayerSnapshotRecord(player_id=pid, captured_at="2026-09-11T20:00:00Z", now_cost=50, raw_json={})
            for pid in ids
        ], run)
        repo.finish_fetch_run(conn, run, "success", current_event=event)
        repo.upsert_squad_picks(conn, entry_id, event, [
            PickRecord(player_id=pid, position=index + 1, raw_json={})
            for index, pid in enumerate(_REGRESSION_SQUAD)
        ])
        for pid in _REGRESSION_SQUAD:
            repo.insert_manager_acquisition(conn, entry_id, pid, event, 45, source="manual",
                                            created_at="2026-09-11T22:00:00Z")
        # 0 free transfers remaining in GW4, £0.7m bank — the state after two
        # already-executed GW4 moves (the outgoing players are already gone).
        repo.upsert_manual_manager_state(conn, entry_id, event, 0, 7, captured_at="2026-09-11T22:46:18Z")
    return run


def test_h_regression_manager_state_reconciles_and_is_not_recharged(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    try:
        _seed_regression_manager(conn)
        context = get_planning_context(conn, 241392, 4, as_of="2026-09-11T22:46:18Z", season="2026/27")
        assert context.manager_state["free_transfers"] == 0
        assert context.manager_state["bank"] == 7
        squad = manager_worlds.resolve_squad(context, conn)
        squad_ids = sorted(int(pid) for pid in squad["squad_ids"])
        assert 41 in squad_ids and 42 in squad_ids
        assert 11 not in squad_ids and 12 not in squad_ids

        state = rc.build_route_state(conn, context, squad)
        assert state.free_transfers == 0
        assert state.bank_tenths == 7
        assert state.cumulative_hit_points == 0
        assert max(state.club_counts().values()) <= 3
        assert state.position_counts() == {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}

        meta = rc.load_player_meta(conn, sorted(set(squad_ids) | {43}))
        result = ts.apply_transfer_batch(state, ts.TransferBatch((ts.TransferAction(13, 43),)),
                                         ts.PriceSnapshot(event=4, prices={13: 50, 43: 45}), meta)
        assert result.ok, result.errors
        assert result.hit_points == 4  # exactly one extra transfer, charged once
        assert result.next_event_state.free_transfers == 1
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# I. Lineup separation
# ---------------------------------------------------------------------------


def test_i_lineup_remains_current_gw_when_transfer_horizon_is_incomplete():
    support = _support([4])  # only GW4 supported
    lineup = {"status": fg.LINEUP_ONLY, "policy": {"captain_id": 426, "vice_captain_id": 427}}
    decision = fg.evaluate_four_gw_decision(
        planning_event=4, support_by_event=support, cutoff="C1", lineup=lineup,
    )
    assert decision["horizon"]["status"] == fg.DECISION_HORIZON_INCOMPLETE
    assert decision["transfer_recommendation"]["status"] == fg.RECOMMENDATION_SUPPRESSED
    assert decision["transfer_recommendation"]["preferred_route_id"] is None
    assert decision["lineup_decision"]["horizon_length"] == 1
    assert decision["lineup_decision"]["decision_events"] == [4]
    assert decision["lineup_recommendation"] is not None
    assert decision["lineup_recommendation"]["lineup_only"] is True
    assert decision["lineup_recommendation"]["basis"] == "CURRENT_GW_H1"
    assert decision["lineup_recommendation"]["basis_events"] == [4]
    assert "GW4 lineup is supported" in decision["operator_summary"]
    assert "horizon" in decision["operator_summary"]


def test_i_lineup_can_be_unsupported_without_touching_the_transfer_rule():
    decision = fg.evaluate_four_gw_decision(planning_event=4, support_by_event={}, cutoff="C1", lineup=None)
    assert decision["horizon"]["status"] == fg.DECISION_HORIZON_INCOMPLETE
    assert decision["transfer_recommendation"]["status"] == fg.RECOMMENDATION_SUPPRESSED
    assert decision["lineup_recommendation"] is None


# ---------------------------------------------------------------------------
# J. Frozen predictive models untouched
# ---------------------------------------------------------------------------


def test_j_predictive_model_versions_unchanged():
    from fpl_brain import joint_minutes, monte_carlo, xpts

    # Re-pinned for R2C (scorer reconciliation numerical correctness), which
    # deliberately bumped the Monte Carlo version from mc_v1.2.1 to mc_v1.3.0.  The
    # guard's intent is unchanged: a DECISION-layer change must not move a predictive
    # model version on its own.
    assert joint_minutes.JOINT_MINUTES_MODEL_VERSION == "minutes_v1.5.2"
    assert monte_carlo.MONTE_CARLO_MODEL_VERSION == "mc_v1.3.0"
    assert xpts.XPTS_MODEL_VERSION == "xpts_v1.4.1"
    assert ts.HIT_POINTS_PER_EXTRA_TRANSFER == 4
    assert ts.MAX_STORED_FREE_TRANSFERS == 5
    assert ts.SQUAD_TEAM_LIMIT == 3


def test_j_decision_layer_has_no_predictive_coupling():
    source = (REPO_ROOT / "fpl_brain/four_gw_decision.py").read_text(encoding="utf-8")
    for forbidden in ("monte_carlo", "joint_minutes", "xpts", "calibration", "numpy", "random"):
        assert f"import {forbidden}" not in source
        assert f"from .{forbidden}" not in source
    # The only import is the sqlite3 stdlib module used for the readiness query.
    assert "import sqlite3" in source


# ---------------------------------------------------------------------------
# Wildcard trigger screen (Task 7)
# ---------------------------------------------------------------------------


def test_wildcard_screen_never_fabricates_points():
    quiet = fg.wildcard_trigger_screen()
    assert quiet["status"] == fg.WILDCARD_NOT_COMPETITIVE
    assert quiet["provisional_points_estimate"] is None
    assert quiet["actionable"] is False

    busy = fg.wildcard_trigger_screen(weak_slot_count=5, availability_problems=2, desired_transfer_count=9)
    assert busy["status"] == fg.WILDCARD_REVIEW_REQUIRED
    assert busy["provisional_points_estimate"] is None
    assert busy["actionable"] is False
    assert fg.WILDCARD_HIT_INTERACTION_FLAG in busy["flags"]
    assert busy["hit_interaction_verified"] is True
    assert "UNVERIFIED" not in fg.WILDCARD_HIT_INTERACTION_FLAG

    # An UNVERIFIED mapping is not evidence: quantitative Wildcard optimisation
    # does not exist, so a bare dict can never make the chip actionable.
    injected = fg.wildcard_trigger_screen(
        weak_slot_count=5, supported_evaluation={"four_gw_net_core": 12.0},
    )
    assert injected["status"] == fg.WILDCARD_REVIEW_REQUIRED
    assert injected["actionable"] is False
    assert injected["recommendation"] == "NONE"
    assert injected["evaluation_verified"] is False
    assert injected["overstatement_prevented"] is True
    assert injected["wildcard_quantitative_capability"] == fg.WILDCARD_QUANTITATIVE_CAPABILITY
    assert "PLAY_WILDCARD" not in json.dumps(injected)

    # A VERIFIED separate four-GW evaluation is real evidence, but the play rule on
    # top of it is UNCALIBRATED, so it is surfaced as a non-executable signal and
    # the chip stays review-only.  Nothing here may be executable until a
    # calibrated four-GW Wildcard evaluator exists.
    verified = fg.wildcard_trigger_screen(
        weak_slot_count=5,
        supported_evaluation={
            "schema": fg.SUPPORTED_WILDCARD_EVALUATION_SCHEMA,
            "four_gw_net_core": 12.0,
            "four_gw_certification_identity": "sha256:abc",
            "data_snapshot_sha256": "deadbeef",
            "distinct_squad_from_current": True,
            "wildcard_squad_player_ids": [1, 2, 3],
        },
    )
    assert verified["status"] == fg.WILDCARD_EVALUATION_SUPPORTED
    assert verified["evaluation_verified"] is True
    assert verified["actionable"] is False
    assert verified["executable"] is False
    assert verified["recommendation"] == "NONE"
    assert verified["verified_evaluation_signal"] == "POSITIVE"
    assert verified["calibration_status"] == fg.WILDCARD_CALIBRATION_STATUS
    assert "PLAY_WILDCARD" not in json.dumps(verified)


# ---------------------------------------------------------------------------
# Operational report contract: an H1 result may never be presented as the
# best normal transfer recommendation.
# ---------------------------------------------------------------------------


def _load_refresh_script():
    import importlib.util

    path = REPO_ROOT / "scripts/final_operational_refresh_gw04.py"
    spec = importlib.util.spec_from_file_location("final_operational_refresh_gw04", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _minimal_report(horizon_status: str) -> dict:
    supported = [4, 5, 6, 7] if horizon_status == fg.DECISION_HORIZON_COMPLETE else [4]
    blocked = [] if horizon_status == fg.DECISION_HORIZON_COMPLETE else [5, 6, 7]
    horizon = {
        "decision_events": [4, 5, 6, 7], "status": horizon_status,
        "supported_events": supported, "blocked_events": blocked,
        "events": {
            str(event): {"supported": event in supported, "reason": None if event in supported else "PREDICTION_READINESS_FAILED",
                         "data_cutoff": "CUT", "missing_families": [] if event in supported else ["monte_carlo_v1"],
                         "stale_families": []}
            for event in (4, 5, 6, 7)
        },
    }
    policy = {"starter_names": ["A"], "bench_gk_name": "B", "bench_outfield_names": ["C"],
              "captain_name": "A", "vice_captain_name": "C"}
    route_row = {
        "transfer_label": "ROLL", "gw4_net_core": 39.0, "valid": True,
        "transfer_accounting": {"sale_proceeds_tenths": 0, "purchase_cost_tenths": 0, "hit_points": 0,
                                "bank_after_tenths": 3, "next_free_transfers": 5},
        "frozen_comparison": {"gw4_net_core": 39.0},
    }
    return {
        "planning_cutoff": "CUT", "planning_context": {"deadline": "D"},
        "decision_layer": {
            "horizon": horizon,
            "screened_actions": {"enumerated_transfer_pairs": 1, "enumeration_exhaustive": True,
                                 "screened_legal_actions": 1, "illegal_single_transfers": 0,
                                 "promotion_pool_size": 1, "promotion_pool_limit": 240},
            "wildcard_screen": {"status": fg.WILDCARD_NOT_COMPETITIVE, "signals": []},
            "operator_summary": "GW4 lineup is supported, but the four-Gameweek transfer horizon is incomplete.",
        },
        "h1_descriptive_route": {"route_id": "ROLL", "policy": policy},
        "route_comparison": {"routes": {route_id: dict(route_row) for route_id in
                                        ("A_RODON_JUSTIN_MUHAREMOVIC_DAVIS", "B_RAYA_MARTINEZ_RODON_DAVIS",
                                         "RODON_ONLY_DAVIS", "ROLL")},
                             "ranking": [{"rank": 1, "route_id": "ROLL", "gw4_net_core": 39.0}]},
        "official_refresh": {"fetch_run": {}, "manager_sync": {}, "price_freshness": {"freshness": "CURRENT"},
                             "price_snapshot_id": "SNAP"},
        "corrected_manager_state": {"free_transfers_before": 0, "bank_before": "£0.7m"},
        "artifacts": {"operational_json": "j", "operational_markdown": "m", "candidate_universe": "c",
                      "manager_packet": "p", "world_cache": "w", "scouting_input": "s"},
    }


def test_report_suppresses_transfer_recommendation_when_horizon_incomplete():
    script = _load_refresh_script()
    markdown = script._markdown(_minimal_report(fg.DECISION_HORIZON_INCOMPLETE))
    assert "SUPPRESSED" in markdown
    assert "DECISION_HORIZON_INCOMPLETE" in markdown
    assert "NOT a transfer recommendation" in markdown
    assert "lineup-only" in markdown
    # The old H1-as-preference language must be gone.
    assert "Fresh GW4 preference" not in markdown
    assert "Best transfer" not in markdown


def test_report_states_the_decision_events_and_blocked_horizon_events():
    script = _load_refresh_script()
    report = _minimal_report(fg.DECISION_HORIZON_INCOMPLETE)
    markdown = script._markdown(report)
    assert "[4, 5, 6, 7]" in markdown
    for event in (5, 6, 7):
        assert f"GW{event}" in markdown
    assert "blocked" in markdown.lower()
    # With a complete horizon the same renderer would still label the objective
    # as the four-GW transfer decision, not a current-GW preference.
    complete = _minimal_report(fg.DECISION_HORIZON_COMPLETE)
    complete_markdown = script._markdown(complete)
    assert "DECISION_HORIZON_COMPLETE" in complete_markdown


# ---------------------------------------------------------------------------
# Item 2 — end-of-season horizon behaviour
# ---------------------------------------------------------------------------


def test_season_end_short_horizon_uses_every_remaining_event():
    for event, expected in ((35, [35, 36, 37, 38]), (36, [36, 37, 38]), (37, [37, 38]), (38, [38])):
        horizon = fg.evaluate_horizon(
            planning_event=event, support_by_event=_support(expected), cutoff="C1",
        )
        assert horizon["decision_events"] == expected, event
        assert horizon["effective_horizon_length"] == len(expected)
        assert horizon["requested_horizon_length"] == 4
        if len(expected) == 4:
            assert horizon["status"] == fg.DECISION_HORIZON_COMPLETE
            assert horizon["short_horizon"] is False
        else:
            # Fewer than four real events remain: this is NOT "incomplete".
            assert horizon["status"] == fg.SEASON_END_SHORT_HORIZON
            assert horizon["short_horizon"] is True
        assert horizon["complete"] is True
        assert fg.transfer_recommendation_allowed(horizon) is True
        assert 39 not in horizon["decision_events"]
        assert horizon["season_last_event"] == 38


def test_season_end_short_horizon_still_blocks_on_unready_event():
    support = _support([36, 37, 38])
    support[37] = {"supported": False, "data_cutoff": "C1"}
    horizon = fg.evaluate_horizon(planning_event=36, support_by_event=support, cutoff="C1")
    assert horizon["status"] == fg.DECISION_HORIZON_INCOMPLETE
    assert horizon["blocked_events"] == [37]
    assert fg.transfer_recommendation_allowed(horizon) is False


def test_season_end_short_horizon_emits_recommendation_over_remaining_events():
    support = _support([38])
    routes = [{
        "route_id": "ROLL", "valid": True,
        "per_event": [{"event": 38, "mean_gross_core": 30.0, "hit_points": 0}],
        "terminal_ft": 3, "terminal_bank_tenths": 5,
    }]
    decision = fg.evaluate_four_gw_decision(planning_event=38, support_by_event=support, cutoff="C1", routes=routes)
    assert decision["horizon"]["status"] == fg.SEASON_END_SHORT_HORIZON
    assert decision["transfer_recommendation"]["status"] == fg.RECOMMENDATION_AVAILABLE
    assert decision["transfer_recommendation"]["basis_events"] == [38]
    assert decision["transfer_recommendation"]["preferred_route_id"] == "ROLL"
    assert decision["transfer_recommendation"]["ranking"][0]["four_gw_net_core"] == 30.0
    assert "season-end" in decision["operator_summary"]


def test_planning_beyond_the_season_is_rejected():
    with pytest.raises(fg.HorizonError):
        fg.decision_events(39)
    with pytest.raises(fg.HorizonError):
        fg.evaluate_horizon(planning_event=39, support_by_event={}, cutoff="C1")


# ---------------------------------------------------------------------------
# Item 4 — verified Wildcard same-Gameweek rule
# ---------------------------------------------------------------------------


def test_wildcard_covers_transfers_already_made_in_the_same_gameweek():
    from fpl_brain.season_rules import SeasonRules

    rules = SeasonRules(season="2026/27")
    # Two transfers were already charged earlier in the GW, then the Wildcard is played.
    covered = fg.wildcard_same_gw_hit(prior_paid_transfers=2, wildcard_transfers=5, rules=rules)
    assert covered["gross_hit_without_chip"] == 28  # 7 extra transfers x 4
    assert covered["hit_covered_by_wildcard"] == 28
    assert covered["net_hit"] == 0
    assert covered["covers_prior_same_gw_transfers"] is True
    assert covered["saved_free_transfers_preserved"] is True


def test_no_transfer_deduction_remains_after_a_wildcard():
    rule = fg.wildcard_same_gw_hit(prior_paid_transfers=1, wildcard_transfers=0)
    assert rule["net_hit"] == 0
    # Without the chip the same already-charged transfer stands.
    from fpl_brain.season_rules import SeasonRules, wildcard_gameweek_hit

    inactive = wildcard_gameweek_hit(SeasonRules(season="2026/27"), prior_paid_transfers=1,
                                     wildcard_transfers=0, wildcard_active=False)
    assert inactive["net_hit"] == 4
    assert inactive["hit_covered_by_wildcard"] == 0


def test_wildcard_saved_ft_rule_follows_official_behaviour():
    from fpl_brain.season_rules import (
        SeasonRules,
        chip_keeps_saved_free_transfers,
        free_transfers_after_chip,
        free_transfers_after_gameweek,
    )

    rules = SeasonRules(season="2026/27")
    assert chip_keeps_saved_free_transfers(rules.wildcard_ft_rule)
    assert rules.wildcard_transfers_are_permanent is True
    assert rules.wildcard_cancellable_once_confirmed is False
    assert rules.one_chip_per_gameweek is True
    # Official rule: the Wildcard RETAINS the saved bank — no weekly +1 across it.
    for saved in (1, 2, 3, 5):
        assert free_transfers_after_chip(rules, "wildcard", event_start_free_transfers=saved) == saved
        assert free_transfers_after_chip(rules, "freehit", event_start_free_transfers=saved) == saved
    # The normal rollover (no chip) still accrues.
    assert free_transfers_after_gameweek(rules, 3, 0) == 4


def test_wildcard_prior_transfers_stay_part_of_the_squad_and_are_not_recharged():
    """Previously executed GW moves remain owned; the chip only removes the hit."""
    players = tuple(
        ts.RoutePlayer(pid, POSITION.get(pid) or POOL_POSITION[pid], CLUB.get(pid) or POOL_CLUB[pid], 45)
        for pid in [p for p in SQUAD_IDS if p != 11] + [41]
    )
    state = ts.RouteState(event=4, players=players, bank_tenths=7, free_transfers=0)
    # The earlier signing is still owned...
    assert 41 in state.by_id() and 11 not in state.by_id()
    # ...and a further GW4 transfer would cost one hit without the chip...
    batch = ts.TransferBatch((ts.TransferAction(12, 42),))
    without = ts.apply_transfer_batch(state, batch, ts.PriceSnapshot(event=4, prices={12: 50, 42: 45}), _meta())
    assert without.hit_points == 4
    # ...which the Wildcard covers entirely (no deduction remains for the GW).
    covered = fg.wildcard_same_gw_hit(prior_paid_transfers=without.paid_transfers, wildcard_transfers=0)
    assert covered["net_hit"] == 0
    assert covered["gross_hit_without_chip"] == 4

