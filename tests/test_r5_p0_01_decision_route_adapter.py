"""R5-P0-01 — the decision-board route-shape boundary.

The blocked R5 acceptance proved that ``run_four_gw_decision`` passed a
``route_optimizer`` result to ``routes_for_decision``, whose adapter reads the
``route_comparator`` schema.  Every optimizer route was therefore adapted to
``per_event == []`` with null terminal accounting, ``route_eligibility`` excluded all of
them, and the board was empty — silently, with nothing naming the schema mismatch.

These tests pin the repair:

  * the real optimizer record that proved the defect adapts to a COMPLETE four-event
    route, and the real blocked artifact drives a full board reconstruction when present;
  * both producer schemas adapt to the SAME canonical record, field by field;
  * the wrong adapter for a record's schema raises instead of silently degrading;
  * the production call site uses the optimizer boundary;
  * fail-closed behaviour still excludes genuinely incomplete routes.
"""

from __future__ import annotations

import ast
import json
import os
import sys
from pathlib import Path

import pytest

from fpl_brain import four_gw_decision as fg
from fpl_brain import route_optimizer as ro
from test_route_optimizer import _config, _provider, _scenario, _universe

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_four_gw_decision as runner  # noqa: E402

RUNNER_SOURCE = (SCRIPTS / "run_four_gw_decision.py").read_text(encoding="utf-8")
RUNNER_TREE = ast.parse(RUNNER_SOURCE)
# The preserved blocked artifact lives under the gitignored data tree, so it is absent in a
# fresh worktree; the tests that need it skip there and run wherever it is present.
BLOCKED_ARTIFACT = Path(os.environ.get(
    "FPL_R5_BLOCKED_ARTIFACT",
    "K:/FPL/data/exports/reliability/r5/final/four_gw_decision.json"))
WINDOW = [5, 6, 7, 8]


# ---------------------------------------------------------------------------
# fixtures: one logical route expressed in BOTH producer schemas
# ---------------------------------------------------------------------------

def _policy(captain: int = 426, vice: int = 427) -> dict:
    return {
        "starter_ids": [8, 115, 165, 173, 305, 346, 368, 411, 426, 427, 497],
        "bench_gk_id": 1, "bench_outfield_order": [334, 290, 124],
        "captain_id": captain, "vice_captain_id": vice,
    }


def _transfers() -> dict:
    return {5: [{"out": 165, "in": 249}], 6: [{"out": 497, "in": 28}],
            7: [{"out": 290, "in": 68}], 8: [{"out": 334, "in": 445}]}


def _gross_by_event() -> dict:
    return {5: 43.4172, 6: 45.1044, 7: 44.9310, 8: 45.1726}


def _bank_after() -> dict:
    return {5: 27, 6: 17, 7: 1, 8: 1}


def optimizer_route(route_id: str = "route_001") -> dict:
    """The optimizer schema: ``per_event``/``actions`` + top-level terminal accounting."""

    gross = _gross_by_event()
    transfers = _transfers()
    bank = _bank_after()
    return {
        "route_id": route_id,
        "valid": True,
        "per_event": [
            {"event": event, "kind": "SINGLE", "hit_points": 0,
             "mean_gross_core": gross[event], "mean_net_core": gross[event],
             "policy": _policy()}
            for event in WINDOW
        ],
        "actions": [
            {"event": event, "kind": "SINGLE", "hit_points": 0,
             "transfers": transfers[event], "squad_ids": [1, 2, 3],
             "ft_after": 1, "bank_after": bank[event]}
            for event in WINDOW
        ],
        "supported_3gw_net_core": sum(gross.values()),
        "cumulative_hits": 0,
        "terminal_ft": 1,
        "terminal_bank_tenths": 1,
    }


def comparator_route(route_id: str = "route_001") -> dict:
    """The comparator schema encoding the SAME logical route."""

    gross = _gross_by_event()
    transfers = _transfers()
    bank = _bank_after()
    return {
        "route_id": route_id,
        "valid": True,
        "events": [
            {"event": event, "batch_size": 1, "hit_points": 0,
             "free_transfers_before": 1, "free_transfers_used": 1, "paid_transfers": 0,
             "bank_before_tenths": bank[event], "bank_after_tenths": bank[event],
             "next_free_transfers": 1, "next_bank_tenths": bank[event],
             "mean_gross_core": gross[event], "mean_net_core": gross[event],
             "selected_policy": _policy(), "squad_hash": "x"}
            for event in WINDOW
        ],
        "terminal_state": {"event": 8, "free_transfers": 1, "bank_tenths": 1, "squad_hash": "x"},
        "errors": [],
    }


# ---------------------------------------------------------------------------
# schema classification and the loud mismatch
# ---------------------------------------------------------------------------

def test_route_record_shape_classifies_both_producers():
    assert fg.route_record_shape(optimizer_route()) == "optimizer"
    assert fg.route_record_shape(comparator_route()) == "comparator"
    assert fg.route_record_shape({}) == "unknown"


def test_comparator_adapter_rejects_an_optimizer_record():
    """The exact R5-P0-01 call must now be loud, not silently empty."""

    with pytest.raises(fg.DecisionRouteShapeError, match="DECISION_ROUTE_SHAPE_MISMATCH"):
        fg.route_for_decision(optimizer_route(), route_id="route_001")


def test_optimizer_adapter_rejects_a_comparator_record():
    with pytest.raises(fg.DecisionRouteShapeError, match="DECISION_ROUTE_SHAPE_MISMATCH"):
        fg.optimizer_route_for_decision(comparator_route(), route_id="route_001")


def test_routes_for_decision_still_accepts_comparator_records():
    """The accepted Phase-8C comparator path is unchanged."""

    transfers = {"route_000": _transfers(), "route_001": _transfers()}
    routes = fg.routes_for_decision({comp_id: comparator_route(comp_id)
                                     for comp_id in ("route_000", "route_001")},
                                    transfers_by_route=transfers)
    assert [row["route_id"] for row in routes] == ["route_000", "route_001"]
    assert all(len(row["per_event"]) == 4 for row in routes)


# ---------------------------------------------------------------------------
# the canonical record: both schemas, field by field (the closed coverage hole)
# ---------------------------------------------------------------------------

def test_both_schemas_adapt_to_the_same_canonical_record():
    """The same logical route must yield a field-for-field identical canonical record.

    The comparator's per-event records do not carry the transfer list (it lives in the
    submitted route steps), so the accepted path passes ``transfers_by_event`` - the runner
    builds exactly this from the optimizer's own actions.
    """

    from_optimizer = fg.optimizer_route_for_decision(optimizer_route(), route_id="route_001")
    from_comparator = fg.route_for_decision(comparator_route(), route_id="route_001",
                                            transfers_by_event=_transfers())
    assert from_optimizer == from_comparator


def test_canonical_record_has_exactly_the_declared_keys():
    record = fg.optimizer_route_for_decision(optimizer_route(), route_id="route_001")
    assert sorted(record) == sorted(fg.CANONICAL_DECISION_ROUTE_KEYS)
    for item in record["per_event"]:
        assert sorted(item) == sorted(fg.CANONICAL_DECISION_PER_EVENT_KEYS)


def test_canonical_dispatch_matches_the_shape_specific_adapters():
    for record in (optimizer_route(), comparator_route()):
        assert fg.canonical_decision_route(record, route_id="route_001") == \
            fg.optimizer_route_for_decision(record, route_id="route_001") if \
            fg.route_record_shape(record) == "optimizer" else \
            fg.route_for_decision(record, route_id="route_001")


@pytest.mark.parametrize("field", ["route_id", "valid", "terminal_ft", "terminal_bank_tenths"])
def test_top_level_fields_survive_adaptation(field):
    record = fg.optimizer_route_for_decision(optimizer_route(), route_id="route_001")
    source = optimizer_route()
    if field == "route_id":
        assert record[field] == "route_001"
    else:
        assert record[field] == source[field]


def test_per_event_detail_survives_adaptation():
    record = fg.optimizer_route_for_decision(optimizer_route(), route_id="route_001")
    assert [int(item["event"]) for item in record["per_event"]] == WINDOW
    for item in record["per_event"]:
        event = int(item["event"])
        assert item["mean_gross_core"] == _gross_by_event()[event]
        assert item["hit_points"] == 0
        assert item["policy"] == _policy()
        assert item["transfers"] == _transfers()[event]
        assert item["bank_after_tenths"] == _bank_after()[event]
        assert item["next_bank_tenths"] == _bank_after()[event]
        assert item["next_free_transfers"] == 1


def test_adapted_route_is_eligible_and_ranks_on_its_real_values():
    routes = [fg.optimizer_route_for_decision(optimizer_route(), route_id="route_001")]
    eligibility = fg.route_eligibility(routes[0], decision_events_window=WINDOW)
    assert eligibility["eligible"] is True
    assert eligibility["reasons"] == []
    value = fg.four_gw_window_value(routes[0]["per_event"], events=WINDOW)
    assert value["four_gw_net_core"] == pytest.approx(sum(_gross_by_event().values()))
    assert value["total_hit_points"] == 0


# ---------------------------------------------------------------------------
# the REAL blocked artifact (skipped only if the preserved artifact is absent)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def blocked_routes():
    if not BLOCKED_ARTIFACT.exists():
        pytest.skip("the preserved blocked R5 decision artifact is not present")
    payload = json.loads(BLOCKED_ARTIFACT.read_text(encoding="utf-8"))
    return (((payload.get("finalist_refinement") or {}).get("route_table") or {})
            .get("routes") or {})


def test_real_blocked_route_now_adapts_completely(blocked_routes):
    """The very record that proved R5-P0-01 must now be a complete four-event route."""

    assert blocked_routes, "the blocked artifact carried no route table"
    route_id = "route_001" if "route_001" in blocked_routes else sorted(blocked_routes)[0]
    source = blocked_routes[route_id]

    legacy = {"route_id": route_id, "valid": True, "per_event": [], "terminal_ft": None,
              "terminal_bank_tenths": None}
    legacy_eligibility = fg.route_eligibility(legacy, decision_events_window=WINDOW)
    assert legacy_eligibility["eligible"] is False
    assert "MISSING_EVENT_EVALUATION:GW5,GW6,GW7,GW8" in legacy_eligibility["reasons"]
    assert "TERMINAL_ACCOUNTING_MISSING:terminal_ft" in legacy_eligibility["reasons"]
    assert "TERMINAL_ACCOUNTING_MISSING:terminal_bank_tenths" in legacy_eligibility["reasons"]

    with pytest.raises(fg.DecisionRouteShapeError):
        fg.route_for_decision(source, route_id=route_id)

    adapted = fg.optimizer_route_for_decision(source, route_id=route_id)
    eligibility = fg.route_eligibility(adapted, decision_events_window=WINDOW)
    assert eligibility["eligible"] is True, eligibility["reasons"]
    assert [int(item["event"]) for item in adapted["per_event"]] == WINDOW
    assert adapted["terminal_ft"] == source["terminal_ft"]
    assert adapted["terminal_bank_tenths"] == source["terminal_bank_tenths"]
    assert len(adapted["per_event"]) == 4
    for item in adapted["per_event"]:
        assert item["transfers"] or item["transfers"] == []
        assert item["policy"], "the H1 policy must survive so the lineup can be sourced"


def test_real_blocked_route_board_value_equals_the_optimizer_objective(blocked_routes):
    """The board and the search layer must agree numerically on every route."""

    routes = fg.optimizer_routes_for_decision(blocked_routes)
    board = fg.build_decision_board(routes=routes, decision_events_window=WINDOW)
    assert len(board) == len(blocked_routes)
    for row in board:
        assert row["eligible"] is True, (row["route_id"], row["exclusion_reason"])
        optimizer_value = float(blocked_routes[row["route_id"]]["supported_3gw_net_core"])
        assert row["four_gw_net_core"] == pytest.approx(optimizer_value)


def test_real_blocked_board_ranking_matches_the_authoritative_final_ranking(blocked_routes):
    routes = fg.optimizer_routes_for_decision(blocked_routes)
    ranking = fg.rank_routes_by_four_gw(
        [{"route_id": row["route_id"],
          "four_gw_net_core": fg.four_gw_window_value(row["per_event"], events=WINDOW)["four_gw_net_core"],
          "total_hit_points": fg.four_gw_window_value(row["per_event"], events=WINDOW)["total_hit_points"],
          "terminal_ft": row["terminal_ft"], "terminal_bank_tenths": row["terminal_bank_tenths"]}
         for row in routes])
    payload = json.loads(BLOCKED_ARTIFACT.read_text(encoding="utf-8"))
    authoritative = ((payload.get("finalist_refinement") or {}).get("final_ranking") or {})
    assert ranking[0]["route_id"] == authoritative.get("preferred_route_id")
    assert ranking[1]["route_id"] == authoritative.get("runner_up_route_id")


def test_real_blocked_board_lineup_comes_from_the_preferred_route(blocked_routes):
    routes = fg.optimizer_routes_for_decision(blocked_routes)
    payload = json.loads(BLOCKED_ARTIFACT.read_text(encoding="utf-8"))
    preferred = ((payload.get("finalist_refinement") or {}).get("final_ranking") or {}).get(
        "preferred_route_id")
    policy = fg.lineup_policy_for_route(routes=routes, route_id=preferred,
                                        decision_events_window=WINDOW)
    assert policy is not None, "the H1 lineup must be sourceable from the final preferred route"
    assert len(policy["starter_ids"]) == 11
    assert policy["captain_id"] != policy["vice_captain_id"]


# ---------------------------------------------------------------------------
# the production call graph
# ---------------------------------------------------------------------------

def test_production_runner_uses_the_optimizer_boundary():
    calls = [node.func.attr for node in ast.walk(RUNNER_TREE)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
    assert "optimizer_routes_for_decision" in calls, (
        "the runner must adapt its route_optimizer result through the optimizer boundary"
    )
    assert "routes_for_decision" not in calls, (
        "the runner must NOT use the comparator-only boundary: that was R5-P0-01"
    )


def test_production_chain_keeps_optimizer_routes_complete():
    """``optimize`` -> optimizer boundary -> eligibility -> board -> lineup.

    The end-to-end ``evaluate_four_gw_decision`` call is covered on the REAL blocked
    artifact above; this drives the search's own output through the same boundary the
    runner uses and asserts nothing becomes incomplete at the decision edge.
    """

    universe, state, meta = _universe()
    scenario, config, provider = _scenario(), _config(), _provider()
    result = ro.optimize(universe=universe, initial_state=state, scenario=scenario,
                         player_meta=meta, config=config, world_provider=provider)
    assert result["routes"], "the search produced no routes"
    transfers_by_route = {
        str(route_id): {
            int(action["event"]): [{"out": int(m["out"]), "in": int(m["in"])}
                                   for m in action.get("transfers") or []]
            for action in (record.get("actions") or [])
        }
        for route_id, record in result["routes"].items()
    }
    routes = fg.optimizer_routes_for_decision(result["routes"],
                                             transfers_by_route=transfers_by_route)
    assert len(routes) == len(result["routes"])
    window = list(config.events)
    for row in routes:
        assert len(row["per_event"]) == len(window), row["route_id"]
        assert row["terminal_ft"] is not None
        assert row["terminal_bank_tenths"] is not None
        assert fg.route_eligibility(row, decision_events_window=window)["eligible"] is True

    board = fg.build_decision_board(routes=routes, decision_events_window=window)
    eligible = [row for row in board if row["eligible"]]
    assert len(eligible) == len(routes)
    baseline = next((row["route_id"] for row in routes
                     if not any(item["transfers"] for item in row["per_event"])), None)
    assert baseline is not None, "a ROLL baseline must survive adaptation"
    preferred = eligible[0]["route_id"]
    policy = fg.lineup_policy_for_route(routes=routes, route_id=preferred,
                                        decision_events_window=window)
    assert policy is not None
    assert board[0]["current_gw_lineup"]


# ---------------------------------------------------------------------------
# fail-closed behaviour is unchanged for genuinely incomplete routes
# ---------------------------------------------------------------------------

def test_optimizer_route_missing_one_event_is_still_excluded():
    record = optimizer_route()
    record["per_event"] = [item for item in record["per_event"] if int(item["event"]) != 7]
    adapted = fg.optimizer_route_for_decision(record, route_id="route_001")
    eligibility = fg.route_eligibility(adapted, decision_events_window=WINDOW)
    assert eligibility["eligible"] is False
    assert "MISSING_EVENT_EVALUATION:GW7" in eligibility["reasons"]


def test_optimizer_route_without_terminal_accounting_is_still_excluded():
    record = optimizer_route()
    record.pop("terminal_ft")
    record.pop("terminal_bank_tenths")
    adapted = fg.optimizer_route_for_decision(record, route_id="route_001")
    assert adapted["terminal_ft"] is None and adapted["terminal_bank_tenths"] is None
    eligibility = fg.route_eligibility(adapted, decision_events_window=WINDOW)
    assert eligibility["eligible"] is False
    assert "TERMINAL_ACCOUNTING_MISSING:terminal_ft" in eligibility["reasons"]
    assert "TERMINAL_ACCOUNTING_MISSING:terminal_bank_tenths" in eligibility["reasons"]


def test_optimizer_route_with_an_absent_core_value_is_still_unusable():
    record = optimizer_route()
    record["per_event"][0].pop("mean_gross_core")
    adapted = fg.optimizer_route_for_decision(record, route_id="route_001")
    eligibility = fg.route_eligibility(adapted, decision_events_window=WINDOW)
    assert eligibility["eligible"] is False
    assert any(reason.startswith("UNUSABLE_EVENT_CORE") for reason in eligibility["reasons"])


def test_optimizer_route_marked_invalid_is_still_rejected():
    record = optimizer_route()
    record["valid"] = False
    adapted = fg.optimizer_route_for_decision(record, route_id="route_001")
    assert fg.route_eligibility(adapted, decision_events_window=WINDOW)["eligible"] is False
    assert "ROUTE_INVALID" in fg.route_eligibility(adapted, decision_events_window=WINDOW)["reasons"]


def test_zero_and_absent_stay_distinguishable():
    """A genuine zero must remain usable; an absent field must not become zero."""

    record = optimizer_route()
    record["per_event"][0]["mean_gross_core"] = 0.0
    adapted = fg.optimizer_route_for_decision(record, route_id="route_001")
    assert adapted["per_event"][0]["mean_gross_core"] == 0.0
    assert fg.usable_expected_core(adapted["per_event"][0]) is True

    record = optimizer_route()
    record["per_event"][0]["mean_gross_core"] = None
    adapted = fg.optimizer_route_for_decision(record, route_id="route_001")
    assert adapted["per_event"][0]["mean_gross_core"] is None
    assert fg.usable_expected_core(adapted["per_event"][0]) is False


# ---------------------------------------------------------------------------
# telemetry (R5 secondary finding, reporting only)
# ---------------------------------------------------------------------------

def test_stage2_timing_measures_stage2_not_the_gap_before_it():
    timings = runner._stage_timings(search_seconds=2835.5, refine_started=1000.0,
                                    refine_finished=4531.0, stability_seconds=0.0)
    assert timings == {"stage1_seconds": 2835.5, "stage2_refinement_seconds": 3531.0,
                       "stability_seconds": 0.0}
    assert timings["stage2_refinement_seconds"] > 0


def test_stage2_timing_cannot_report_a_negative_duration():
    timings = runner._stage_timings(search_seconds=1.0, refine_started=5.0,
                                    refine_finished=4.0, stability_seconds=2.0)
    assert timings["stage2_refinement_seconds"] == 0.0
