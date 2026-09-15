"""Pre-run correctness patch tests: chip FT transition, trigger metric,
route eligibility, and cutoff/override safety.
"""

from __future__ import annotations

import pytest

from fpl_brain import four_gw_decision as fg
from fpl_brain import transfer_state as ts
from fpl_brain.season_rules import SeasonRules, free_transfers_after_chip


def _support(events, cutoff="C1"):
    return {int(event): {"supported": True, "data_cutoff": cutoff} for event in events}


def _route(route_id, *, valid=True, events=(4, 5, 6, 7), core=10.0, hit=0, terminal=True, **over):
    route = {
        "route_id": route_id,
        "valid": valid,
        "per_event": [{"event": event, "mean_gross_core": core, "hit_points": hit} for event in events],
    }
    if terminal:
        route["terminal_ft"] = 1
        route["terminal_bank_tenths"] = 5
    route.update(over)
    return route


# ---------------------------------------------------------------------------
# Item 1 — Wildcard / Free Hit FT transition
# ---------------------------------------------------------------------------


def test_wildcard_retains_saved_ft_without_weekly_accrual():
    rules = SeasonRules(season="2026/27")
    for saved in (1, 2, 3, 5):
        assert free_transfers_after_chip(rules, "wildcard", event_start_free_transfers=saved) == saved
    # Not the normal rollover (which would give 4 from 3).
    assert free_transfers_after_chip(rules, "wildcard", event_start_free_transfers=3) != 4


def test_wildcard_activated_after_same_gw_transfers_uses_event_start_bank():
    # The real GW4 shape: entered with 2 FT, both already used, 0 remaining.
    assert free_transfers_after_chip(SeasonRules(season="2026/27"), "wildcard",
                                     event_start_free_transfers=2) == 2
    # The state engine carries the explicit event-start bank, NOT the remaining count.
    state = ts.RouteState(event=4, players=(), bank_tenths=7, free_transfers=0,
                          event_start_free_transfers=2)
    assert state.free_transfers == 0 and state.event_start_free_transfers == 2
    assert ts.chip_next_event_free_transfers("wildcard", event_start_free_transfers=2) == 2
    assert ts.chip_next_event_free_transfers(
        "wildcard", event_start_free_transfers=2, ft_before=0
    ) == 2


def test_free_hit_retains_saved_ft_and_team_chips_roll_normally():
    rules = SeasonRules(season="2026/27")
    assert free_transfers_after_chip(rules, "freehit", event_start_free_transfers=5) == 5
    # Bench Boost / Triple Captain do not change the transfer process.
    assert free_transfers_after_chip(rules, "bboost", event_start_free_transfers=2,
                                     free_transfers_available=2) == 3
    for chip in ("wildcard", "freehit", "bboost", "3xc"):
        assert free_transfers_after_chip(rules, chip, event_start_free_transfers=3,
                                         free_transfers_available=3) <= rules.max_free_transfers


def test_unrecorded_event_start_ft_is_refused_not_inferred():
    with pytest.raises(ValueError):
        free_transfers_after_chip(SeasonRules(season="2026/27"), "wildcard",
                                  event_start_free_transfers=None)
    with pytest.raises(ts.ChipFTTransitionError):
        ts.chip_next_event_free_transfers("wildcard", event_start_free_transfers=None)
    # A depleted remaining count must not be used as the preserved bank.
    with pytest.raises(ValueError):
        fg.chip_free_transfers_after("wildcard", event_start_free_transfers=None,
                                     free_transfers_available=0)


def test_four_gw_chip_free_transfers_after_delegates_to_season_rules():
    for saved in (1, 2, 3, 5):
        assert fg.chip_free_transfers_after("wildcard", event_start_free_transfers=saved) == saved


# ---------------------------------------------------------------------------
# Item 2 — Wildcard trigger input metric
# ---------------------------------------------------------------------------


def test_desired_squad_changes_is_bounded_by_squad_size_not_action_space():
    # 2,000 legal options spread across 15 slots, none materially better.
    legal = [
        {"out_player_id": slot, "in_player_id": 1000 + index, "four_gw_proxy_delta": 0.2}
        for index in range(2000)
        for slot in [1 + (index % 15)]
    ]
    result = fg.desired_squad_changes(legal_actions=legal, materiality_core=0.5)
    assert result["desired_transfer_count"] == 0
    assert result["units"] == "SQUAD_CHANGES"
    assert result["max_possible"] == 15
    assert result["legal_actions_considered"] == 2000


def test_desired_squad_changes_counts_distinct_slots_with_material_gains():
    legal = []
    # Slots 3, 7 and 11 have a material replacement; the rest are marginal.
    for slot in range(1, 16):
        delta = 2.0 if slot in (3, 7, 11) else 0.1
        for _ in range(50):
            legal.append({"out_player_id": slot, "in_player_id": 500 + slot, "four_gw_proxy_delta": delta})
    result = fg.desired_squad_changes(legal_actions=legal, materiality_core=0.5)
    assert result["desired_transfer_count"] == 3
    assert result["slots_with_material_replacement"] == [3, 7, 11]
    assert result["desired_transfer_count"] <= 15


def test_many_legal_options_do_not_force_wildcard_review():
    legal = [
        {"out_player_id": 1 + (index % 15), "in_player_id": 900 + index, "four_gw_proxy_delta": 0.05}
        for index in range(2000)
    ]
    desired = fg.desired_squad_changes(legal_actions=legal, materiality_core=0.5)
    screen = fg.wildcard_trigger_screen(
        weak_slot_count=0, availability_problems=0, desired_transfer_count=desired["desired_transfer_count"],
    )
    assert desired["desired_transfer_count"] == 0
    assert screen["status"] == fg.WILDCARD_NOT_COMPETITIVE
    assert screen["evidence"]["desired_transfer_count"] == 0


def test_wildcard_screen_reports_the_ft_transition_it_would_apply():
    screen = fg.wildcard_trigger_screen(weak_slot_count=5, event_start_free_transfers=2,
                                        free_transfers_available=0, wildcard_transfers=6,
                                        prior_paid_transfers=1)
    assert screen["free_transfer_transition"]["status"] == "SUPPORTED"
    assert screen["free_transfer_transition"]["next_event_free_transfers"] == 2
    assert screen["same_gameweek_hit_rule"]["net_hit"] == 0
    assert screen["hit_interaction_verified"] is True
    unrecorded = fg.wildcard_trigger_screen(weak_slot_count=5, event_start_free_transfers=None)
    assert unrecorded["free_transfer_transition"]["status"].startswith("EVENT_START_FT_NOT_RECORDED")
    assert unrecorded["free_transfer_transition"]["next_event_free_transfers"] is None


# ---------------------------------------------------------------------------
# Item 3 — invalid / incomplete routes must never rank
# ---------------------------------------------------------------------------


def test_route_eligibility_requires_validity_completeness_and_terminal_accounting():
    good = fg.route_eligibility(_route("GOOD"), decision_events_window=(4, 5, 6, 7))
    assert good["eligible"] is True and good["reasons"] == []

    invalid = fg.route_eligibility(_route("BAD", valid=False), decision_events_window=(4, 5, 6, 7))
    assert invalid["eligible"] is False and "ROUTE_INVALID" in invalid["reasons"]

    missing = fg.route_eligibility(_route("MISSING", events=(4, 5, 7)), decision_events_window=(4, 5, 6, 7))
    assert missing["eligible"] is False
    assert any("MISSING_EVENT_EVALUATION:GW6" == reason for reason in missing["reasons"])

    duplicated = fg.route_eligibility(
        _route("DUP", events=(4, 5, 5, 6, 7)), decision_events_window=(4, 5, 6, 7)
    )
    assert duplicated["eligible"] is False
    assert any("DUPLICATE_EVENT_EVALUATION:GW5" == reason for reason in duplicated["reasons"])

    no_terminal = fg.route_eligibility(
        _route("NOTERM", terminal=False), decision_events_window=(4, 5, 6, 7)
    )
    assert no_terminal["eligible"] is False
    assert any(reason.startswith("TERMINAL_ACCOUNTING_MISSING") for reason in no_terminal["reasons"])


def test_invalid_route_with_highest_four_gw_core_is_excluded():
    support = _support([4, 5, 6, 7])
    routes = [
        _route("CHEAT", valid=False, core=99.0),           # highest apparent CORE, invalid
        _route("HONEST", valid=True, core=40.0),
    ]
    decision = fg.evaluate_four_gw_decision(planning_event=4, support_by_event=support, cutoff="C1", routes=routes)
    block = decision["transfer_recommendation"]
    assert block["status"] == fg.RECOMMENDATION_AVAILABLE
    assert block["preferred_route_id"] == "HONEST"
    assert [row["route_id"] for row in block["ranking"]] == ["HONEST"]
    assert block["excluded_routes"] == [{"route_id": "CHEAT", "reason": "ROUTE_INVALID"}]
    # The board still shows the excluded route, explicitly marked.
    board = {row["route_id"]: row for row in decision["decision_board"]}
    assert board["CHEAT"]["eligible"] is False
    assert board["CHEAT"]["exclusion_reason"] == "ROUTE_INVALID"


def test_route_missing_a_decision_event_is_excluded_with_reason():
    support = _support([4, 5, 6, 7])
    routes = [
        _route("MISSING_GW6", events=(4, 5, 7), core=80.0),
        _route("COMPLETE", valid=True, core=30.0),
    ]
    decision = fg.evaluate_four_gw_decision(planning_event=4, support_by_event=support, cutoff="C1", routes=routes)
    block = decision["transfer_recommendation"]
    assert block["preferred_route_id"] == "COMPLETE"
    excluded = {row["route_id"]: row["reason"] for row in block["excluded_routes"]}
    assert "MISSING_EVENT_EVALUATION:GW6" in excluded["MISSING_GW6"]


def test_all_invalid_routes_suppress_with_no_valid_routes_status():
    support = _support([4, 5, 6, 7])
    routes = [_route("A", valid=False, core=99.0), _route("B", events=(4, 5, 6), core=98.0)]
    decision = fg.evaluate_four_gw_decision(planning_event=4, support_by_event=support, cutoff="C1", routes=routes)
    block = decision["transfer_recommendation"]
    assert block["status"] == fg.RECOMMENDATION_SUPPRESSED_NO_VALID_ROUTES
    assert block["preferred_route_id"] is None
    assert block["ranking"] == []
    assert block["reason"] == "NO_ELIGIBLE_COMPLETE_ROUTE"
    assert len(block["excluded_routes"]) == 2
    assert decision["no_recommendation"] is True
    assert "no valid complete route" in decision["operator_summary"]


def test_supported_horizon_with_no_routes_at_all_is_not_a_recommendation():
    support = _support([4, 5, 6, 7])
    decision = fg.evaluate_four_gw_decision(planning_event=4, support_by_event=support, cutoff="C1", routes=None)
    assert decision["transfer_recommendation"]["status"] == fg.RECOMMENDATION_SUPPRESSED_NO_VALID_ROUTES
    assert decision["transfer_recommendation"]["preferred_route_id"] is None
    assert decision["no_recommendation"] is True


# ---------------------------------------------------------------------------
# Item 5 — fresh cutoff must cover the manager-state override
# ---------------------------------------------------------------------------


def test_cutoff_must_cover_the_manager_state_override():
    stale = fg.verify_cutoff_covers_override(
        planning_cutoff="2026-09-11T22:46:18Z", override_captured_at="2026-09-12T08:40:38Z",
    )
    assert stale["pass"] is False
    assert stale["status"] == fg.CUTOFF_PRECEDES_OVERRIDE
    assert stale["delta_seconds"] < 0

    fresh = fg.verify_cutoff_covers_override(
        planning_cutoff="2026-09-12T09:00:00Z", override_captured_at="2026-09-12T08:40:38Z",
    )
    assert fresh["pass"] is True and fresh["status"] == "PASS"

    none = fg.verify_cutoff_covers_override(planning_cutoff="2026-09-12T09:00:00Z", override_captured_at=None)
    assert none["pass"] is True and none["status"] == "NO_OVERRIDE_RECORDED"

    assert fg.verify_cutoff_covers_override(planning_cutoff=None,
                                            override_captured_at="2026-09-12T08:40:38Z")["pass"] is False


def test_assert_cutoff_covers_override_raises_on_stale_cutoff():
    with pytest.raises(fg.CutoffOverrideError):
        fg.assert_cutoff_covers_override(
            planning_cutoff="2026-09-11T22:46:18Z", override_captured_at="2026-09-12T08:40:38Z",
        )
    assert fg.assert_cutoff_covers_override(
        planning_cutoff="2026-09-12T09:00:00Z", override_captured_at="2026-09-12T08:40:38Z",
    )["status"] == "PASS"


# ---------------------------------------------------------------------------
# Item 5 (script-level) — the archival GW4 runner refuses a pre-override cutoff
# ---------------------------------------------------------------------------


def _load_refresh_script():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts/final_operational_refresh_gw04.py"
    spec = importlib.util.spec_from_file_location("final_operational_refresh_gw04_pre_run", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_archival_runner_refuses_a_cutoff_that_predates_the_override(tmp_path):
    from fpl_brain.database import connect_database
    from test_manager_state_override import _seed_pre_override_world, _temp_config

    config_file = _temp_config(tmp_path)
    conn = connect_database(tmp_path / "fpl.db")
    try:
        _seed_pre_override_world(conn)
        # Record an override AFTER the cutoff the runner would use.
        with conn:
            from fpl_brain import repositories as repo

            repo.upsert_manual_manager_state(conn, 241392, 4, 0, 7, source="user_confirmed_override",
                                             captured_at="2026-09-12T09:00:00Z",
                                             event_start_free_transfers=2)
    finally:
        conn.close()

    script = _load_refresh_script()
    # An older cutoff must fail fast (rc 5) before any simulation or artifact write.
    rc_old = script.main(["--config", str(config_file), "--cutoff", "2026-09-11T22:46:18Z"])
    assert rc_old == 5
    assert not (tmp_path / "exports").exists() or not any((tmp_path / "exports").rglob("*.json"))


# ---------------------------------------------------------------------------
# Addendum micro-fix — next-event start FT, route adapter, unusable scores,
# Wildcard capability limit
# ---------------------------------------------------------------------------


def _micro_state(**over):
    from test_transfer_state import CLUB, POSITION, SQUAD_IDS
    import fpl_brain.transfer_state as _ts

    players = tuple(_ts.RoutePlayer(pid, POSITION[pid], CLUB[pid], 45) for pid in SQUAD_IDS)
    state = _ts.RouteState(event=4, players=players, bank_tenths=7, free_transfers=0,
                           event_start_free_transfers=2)
    for key, value in over.items():
        state = _ts.RouteState(**{**state.__dict__, key: value})
    return state


def test_next_event_event_start_ft_is_the_rolled_over_bank():
    from test_transfer_state import _meta

    state = _micro_state()  # GW4: event_start_ft 2, remaining 0
    batch = ts.TransferBatch((ts.TransferAction(11, 41),))
    result = ts.apply_transfer_batch(state, batch, ts.PriceSnapshot(event=4, prices={41: 45, 11: 45}), _meta())
    assert result.ok, result.errors
    # Same-event squad keeps the CURRENT event-start value (2)...
    assert result.squad_after.event_start_free_transfers == 2
    # ...but the NEXT event starts from the rolled-over bank, not the old 2.
    assert result.next_event_state.free_transfers == 1
    assert result.next_event_state.event_start_free_transfers == 1

    # GW5 ROLL -> GW6 carries 2 in both fields.
    rolled = result.next_event_state
    roll = ts.apply_transfer_batch(rolled, ts.TransferBatch.roll(),
                                   ts.PriceSnapshot(event=5, prices={}), _meta())
    assert roll.ok, roll.errors
    assert roll.next_event_state.free_transfers == 2
    assert roll.next_event_state.event_start_free_transfers == 2


def test_route_for_decision_adapts_comparator_output():
    comparison = {
        "route_id": "R1", "valid": True, "errors": [],
        "events": [
            {"event": 4, "mean_gross_core": 40.0, "hit_points": 0, "bank_after_tenths": 12,
             "next_bank_tenths": 12, "next_free_transfers": 1, "selected_policy": {"captain_id": 1},
             "transfers": [{"out": 10, "in": 11}]},
            {"event": 5, "mean_gross_core": 35.0, "hit_points": 4, "bank_after_tenths": 8,
             "next_bank_tenths": 8, "next_free_transfers": 1, "selected_policy": None, "transfers": []},
            {"event": 6, "mean_gross_core": 30.0, "hit_points": 0, "bank_after_tenths": 6,
             "next_bank_tenths": 6, "next_free_transfers": 1, "selected_policy": None, "transfers": []},
            {"event": 7, "mean_gross_core": 28.0, "hit_points": 0, "bank_after_tenths": 5,
             "next_bank_tenths": 5, "next_free_transfers": 1, "selected_policy": None, "transfers": []},
        ],
        "terminal_state": {"free_transfers": 1, "bank_tenths": 5, "squad_hash": "x", "event": 7},
    }
    adapted = fg.route_for_decision(comparison)
    assert adapted["terminal_ft"] == 1
    assert adapted["terminal_bank_tenths"] == 5
    assert [record["event"] for record in adapted["per_event"]] == [4, 5, 6, 7]
    assert adapted["per_event"][0]["mean_gross_core"] == 40.0
    eligible = fg.route_eligibility(adapted, decision_events_window=(4, 5, 6, 7))
    assert eligible["eligible"] is True

    support = _support([4, 5, 6, 7])
    decision = fg.evaluate_four_gw_decision(planning_event=4, support_by_event=support, cutoff="C1",
                                            routes=[adapted])
    assert decision["transfer_recommendation"]["status"] == fg.RECOMMENDATION_AVAILABLE
    # gross 133 - hit 4 = 129
    assert decision["transfer_recommendation"]["ranking"][0]["four_gw_net_core"] == pytest.approx(129.0)


def test_unusable_event_core_is_rejected_not_zeroed():
    for bad in (None, float("nan"), float("inf"), float("-inf"), "n/a"):
        route = _route("BAD", core=10.0)
        route["per_event"][2]["mean_gross_core"] = bad
        result = fg.route_eligibility(route, decision_events_window=(4, 5, 6, 7))
        assert result["eligible"] is False, bad
        assert any(reason.startswith("UNUSABLE_EVENT_CORE:GW6") for reason in result["reasons"]), (bad, result)
    # A well-formed route is unaffected.
    assert fg.route_eligibility(_route("OK"), decision_events_window=(4, 5, 6, 7))["eligible"] is True


def test_unusable_event_is_not_summed_as_zero():
    per_event = [
        {"event": 4, "mean_gross_core": 10.0, "hit_points": 0},
        {"event": 5, "mean_gross_core": None, "hit_points": 0},
        {"event": 6, "mean_gross_core": 10.0, "hit_points": 0},
        {"event": 7, "mean_gross_core": 10.0, "hit_points": 0},
    ]
    value = fg.four_gw_window_value(per_event)
    assert value["unusable_events"] == [5]
    assert value["events"] == [4, 6, 7]
    assert value["four_gw_gross_core"] == pytest.approx(30.0)


def test_wildcard_is_never_recommended_without_a_separate_evaluation():
    review = fg.wildcard_trigger_screen(weak_slot_count=6, availability_problems=3, desired_transfer_count=5)
    assert review["status"] == fg.WILDCARD_REVIEW_REQUIRED
    assert review["recommendation"] == "NONE"
    assert review["actionable"] is False
    assert fg.WILDCARD_REQUIRES_SEPARATE_EVALUATION in review["flags"]
    assert "never recommended from the trigger alone" in review["recommendation_basis"]

    # A bare mapping is an unverifiable claim: it can NOT produce PLAY_WILDCARD.
    injected = fg.wildcard_trigger_screen(weak_slot_count=6,
                                          supported_evaluation={"four_gw_net_core": 12.0})
    assert injected["status"] == fg.WILDCARD_REVIEW_REQUIRED
    assert injected["recommendation"] == "NONE"
    assert injected["actionable"] is False
    assert injected["overstatement_prevented"] is True
    assert fg.WILDCARD_QUANTITATIVE_CAPABILITY == "NOT_SUPPORTED"

    # A VERIFIED evaluation may still advise against the chip.
    verified_negative = fg.wildcard_trigger_screen(
        weak_slot_count=6,
        supported_evaluation={
            "schema": fg.SUPPORTED_WILDCARD_EVALUATION_SCHEMA,
            "four_gw_net_core": -8.0,
            "four_gw_certification_identity": "sha256:abc",
            "data_snapshot_sha256": "deadbeef",
            "distinct_squad_from_current": True,
            "wildcard_squad_player_ids": [1, 2, 3],
        },
    )
    # The verified negative sign is reported as a signal, never as an executable
    # recommendation: the Wildcard play rule is uncalibrated.
    assert verified_negative["recommendation"] == "NONE"
    assert verified_negative["verified_evaluation_signal"] == "NEGATIVE"
    assert verified_negative["executable"] is False
    assert verified_negative["calibration_status"] == fg.WILDCARD_CALIBRATION_STATUS
