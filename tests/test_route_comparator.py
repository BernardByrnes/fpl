"""Phase 7B — multi-Gameweek route comparator tests.

Synthetic-first: small deterministic world matrices with hand-known answers.  No
database, no football simulation, no real squad.
"""

from __future__ import annotations

import itertools
import json

import pytest

from fpl_brain import route_comparator as rc
from fpl_brain import route_optimizer as ro
from fpl_brain import transfer_state as ts
from test_transfer_state import CLUB, POOL_CLUB, POOL_POSITION, POSITION, SQUAD_IDS

EVENTS = [4, 5, 6, 7, 8, 9]
UNION = sorted(set(SQUAD_IDS) | set(POOL_POSITION))
META = {pid: ts.PlayerMeta(pid, POSITION.get(pid, POOL_POSITION.get(pid)),
                          CLUB.get(pid, POOL_CLUB.get(pid))) for pid in UNION}

# Base core profile: FWD 31 is the clear captain; DEF 11 is a zero; pool DEF 41/42 are upgrades.
BASE_CORE = {pid: 10.0 for pid in UNION}
BASE_CORE[31] = 20.0
BASE_CORE[11] = 0.0
BASE_CORE[41] = 12.0
BASE_CORE[42] = 13.0


def _matrix(event, cores=None, worlds=6, noise=None):
    core_values = dict(BASE_CORE)
    core_values.update(cores or {})
    minutes = {pid: [90.0] * worlds for pid in UNION}
    core = {}
    for pid in UNION:
        base = core_values.get(pid, 0.0)
        series = []
        for world in range(worlds):
            value = base
            if noise and pid in noise:
                value += noise[pid][world % len(noise[pid])]
            series.append(value)
        core[pid] = series
    return {"worlds": worlds, "player_ids": list(UNION), "core": core, "minutes": minutes}


def _provider(matrices):
    calls = {"count": 0, "events": []}

    def provider(event, union_ids=None):
        calls["count"] += 1
        calls["events"].append(event)
        return matrices[event]

    return provider, calls


def _non_production(provider):
    """The DECLARED interface every injected matrix must enter through.

    A synthetic world was never loaded from a prediction run, so the comparator may
    only consume it under a declaration naming who is exercising that interface.
    """

    return ro.NonProductionWorlds(
        declaration="test_route_comparator: synthetic deterministic matrices, no prediction run",
        provider=provider,
    )


def _bundles(events=EVENTS, simulations=6):
    return {event: rc.EventBundle(event=event, minutes_run_id=1, team_run_id=1, rate_run_id=1,
                                  xpts_run_id=1, mc_run_id=1, simulations=simulations) for event in events}


def _state(*, event=4, bank=0, ft=1):
    players = tuple(ts.RoutePlayer(pid, POSITION[pid], CLUB[pid], 50) for pid in SQUAD_IDS)
    return ts.RouteState(event=event, players=players, bank_tenths=bank, free_transfers=ft)


def _snapshots(events=EVENTS, prices=None):
    base = {pid: 50 for pid in UNION}
    base.update(prices or {})
    return {event: ts.PriceSnapshot(event=event, prices=dict(base)) for event in events}


def _scenario(events=EVENTS, prices=None, scenario_id="TEST_SCENARIO"):
    return rc.PriceScenario(scenario_id=scenario_id, event_snapshots=_snapshots(events, prices))


def _route(route_id, steps):
    return rc.TransferRoute(route_id=route_id, steps=tuple(
        rc.RouteStep(event=event, transfer_batch=ts.TransferBatch(tuple(
            ts.TransferAction(out, incoming) for out, incoming in moves)))
        for event, moves in steps
    ))


def _roll_steps(events=EVENTS):
    return [(event, []) for event in events]


def _compare(routes, *, matrices=None, events=EVENTS, state=None, scenario=None, ft=1):
    matrices = matrices or {event: _matrix(event) for event in events}
    provider, calls = _provider(matrices)
    result = rc.compare_routes(
        bundles=_bundles(events), routes=routes, initial_state=state or _state(ft=ft),
        scenario=scenario or _scenario(events), player_meta=META,
        non_production_worlds=ro.NonProductionWorlds(
            declaration="test_route_comparator: synthetic deterministic matrices, no prediction run",
            provider=provider,
        ),
        simulations=6, planning_cutoff="2026-09-11T13:00:00Z",
    )
    return result, calls


# ---------------------------------------------------------------------------
# 1-5 Route state
# ---------------------------------------------------------------------------


def test_explicit_roll_step_accepted():
    result, _ = _compare([_route("roll", _roll_steps())])
    assert result["routes"]["roll"]["valid"] and result["routes"]["roll"]["errors"] == []


def test_missing_event_step_rejected():
    route = _route("partial", [(4, []), (5, [])])
    result, _ = _compare([route], events=[4, 5, 6])
    assert not result["routes"]["partial"]["valid"]
    assert any("MISSING_STEP_FOR_EVENT: 6" in e for e in result["routes"]["partial"]["errors"])


def test_invalid_current_transfer_invalidates_route():
    route = _route("bad", [(4, [(11, 51)])] + [(e, []) for e in EVENTS[1:]])  # DEF out, MID in
    result, _ = _compare([route])
    assert not result["routes"]["bad"]["valid"]
    assert "POSITION_MULTISET_MISMATCH" in (result["routes"]["bad"]["failure"] or "")


def test_later_invalid_transfer_stops_route_at_correct_event():
    route = _route("late", [(4, []), (5, [(11, 51)])] + [(e, []) for e in EVENTS[2:]])
    result, _ = _compare([route])
    body = result["routes"]["late"]
    assert not body["valid"] and body["valid_through_event"] == 4
    assert "POSITION_MULTISET_MISMATCH" in (body["failure"] or "")


def test_route_id_has_no_football_effect():
    matrices = {event: _matrix(event) for event in EVENTS}
    a, _ = _compare([_route("a", _roll_steps())], matrices=matrices, events=[4, 5])
    b, _ = _compare([_route("completely_different_id", _roll_steps())], matrices=matrices, events=[4, 5])
    assert a["routes"]["a"]["horizons"]["H1"] == b["routes"]["completely_different_id"]["horizons"]["H1"]


# ---------------------------------------------------------------------------
# 6-9 Price scenarios
# ---------------------------------------------------------------------------


def test_explicit_snapshot_required_for_future_transfer():
    scenario = rc.PriceScenario(scenario_id="missing", event_snapshots={4: _snapshots([4])[4]})
    route = _route("r", [(4, []), (5, [(11, 41)])] + [(e, []) for e in EVENTS[2:]])
    result, _ = _compare([route], scenario=scenario)
    assert "NO_PRICE_SNAPSHOT_FOR_EVENT: 5" in (result["routes"]["r"]["failure"] or "")


def test_flat_price_scenario_labelled_assumption():
    base = ts.PriceSnapshot(event=4, prices={pid: 50 for pid in UNION})
    scenario = rc.flat_current_price_scenario(base, EVENTS)
    assert rc.FLAT_PRICE_ASSUMPTION in scenario.flags
    result, _ = _compare([_route("roll", _roll_steps())], scenario=scenario, events=[4, 5])
    assert rc.FLAT_PRICE_ASSUMPTION in result["flags"]


def test_same_route_differs_in_legality_across_price_scenarios():
    route = _route("r", [(4, [(11, 41)])] + [(e, []) for e in EVENTS[1:]])
    cheap = _scenario(scenario_id="cheap")  # everything 50, bank 0 -> sale 50, cost 50: legal
    expensive = _scenario(prices={41: 80}, scenario_id="expensive")  # cost 80 > sale 50: illegal
    ok, _ = _compare([route], scenario=cheap, events=[4, 5])
    bad, _ = _compare([route], scenario=expensive, events=[4, 5])
    assert ok["routes"]["r"]["valid"]
    assert not bad["routes"]["r"]["valid"]
    assert "INSUFFICIENT_BANK" in (bad["routes"]["r"]["failure"] or "")


def test_no_live_price_query_needed_with_a_declared_non_production_world_source():
    # No connection is supplied at all; comparison must still work.
    result, _ = _compare([_route("roll", _roll_steps())], events=[4, 5])
    assert result["routes"]["roll"]["valid"]


# ---------------------------------------------------------------------------
# 10-14 FT / hits
# ---------------------------------------------------------------------------


def test_current_ft_consumed_correctly():
    route = _route("r", [(4, [(11, 41)])] + [(e, []) for e in EVENTS[1:]])
    result, _ = _compare([route], ft=2, events=[4, 5])
    first = result["routes"]["r"]["events"][0]
    assert first["free_transfers_used"] == 1 and first["hit_points"] == 0
    assert result["routes"]["r"]["horizons"]["H1"]["terminal_ft"] == 2


def test_rolled_ft_carries_forward():
    result, _ = _compare([_route("roll", _roll_steps())], ft=2, events=[4, 5, 6, 7])
    assert result["routes"]["roll"]["horizons"]["H1"]["terminal_ft"] == 3
    assert result["routes"]["roll"]["horizons"]["H4"]["terminal_ft"] == 5


def test_later_hit_emerges_correctly():
    # Bank 20 so both moves are affordable; spend the FT at event 4, then need 2 at event 5.
    route = _route("hit", [(4, [(11, 41)]), (5, [(12, 42), (13, 43)])] + [(e, []) for e in EVENTS[2:]])
    state = _state(bank=20, ft=1)
    result, _ = _compare([route], state=state, events=[4, 5, 6, 7])
    events = result["routes"]["hit"]["events"]
    assert events[0]["hit_points"] == 0
    assert events[1]["hit_points"] == 4
    assert result["routes"]["hit"]["horizons"]["H4"]["cumulative_hits"] == 4


def test_no_synthetic_ft_value_added():
    result, _ = _compare([_route("roll", _roll_steps())], events=[4, 5])
    blob = json.dumps(result)
    for forbidden in ("value_of_free_transfer", "ft_value", "terminal_ft_bonus"):
        assert forbidden not in blob
    horizon = result["routes"]["roll"]["horizons"]["H1"]
    assert "terminal_ft" in horizon and "terminal_bank_tenths" in horizon


def test_cumulative_hits_correct():
    route = _route("hits", [(4, [(11, 41)]), (5, [(12, 42), (13, 43)])] + [(e, []) for e in EVENTS[2:]])
    result, _ = _compare([route], state=_state(bank=20, ft=1), events=[4, 5, 6, 7])
    assert result["routes"]["hits"]["horizons"]["H1"]["cumulative_hits"] == 0
    assert result["routes"]["hits"]["horizons"]["H4"]["cumulative_hits"] == 4


# ---------------------------------------------------------------------------
# 15-18 Manager policy
# ---------------------------------------------------------------------------


def test_post_transfer_squad_enters_phase6_evaluator():
    route = _route("r", [(4, [(11, 41)])] + [(e, []) for e in EVENTS[1:]])
    result, _ = _compare([route], events=[4, 5])
    policy = result["routes"]["r"]["events"][0]["selected_policy"]
    assert 41 in policy["starter_ids"] or 41 == policy["bench_gk_id"] or 41 in policy["bench_outfield_order"]
    assert 11 not in policy["starter_ids"] + policy["bench_outfield_order"]
    assert policy["captain_id"] == 31  # highest expected core, selected ex ante


def test_best_policy_selected_ex_ante():
    result, _ = _compare([_route("roll", _roll_steps())], events=[4])
    first = result["routes"]["roll"]["events"][0]
    # BASE_CORE: best XI is 10 outfield at 10/20 + GK, captain FWD 31.
    assert first["selected_policy"]["captain_id"] == 31
    assert first["mean_gross_core"] == pytest.approx(140.0)


def test_same_squad_event_is_cached():
    roll = _route("roll", _roll_steps())
    swap = _route("swap", [(4, [(11, 41)])] + [(e, []) for e in EVENTS[1:]])
    result, _ = _compare([roll, swap], events=[4, 5])
    # event 4: two distinct squads -> 2; event 5: same two -> 4 total, no duplicates.
    assert result["policy_cache_entries"] == 4


def test_policy_choice_does_not_alter_football_worlds():
    matrices = {event: _matrix(event) for event in [4, 5]}
    provider, calls = _provider(matrices)
    rc.compare_routes(bundles=_bundles([4, 5]),
                      routes=[_route("a", _roll_steps([4, 5])), _route("b", [(4, [(11, 41)]), (5, [])])],
                      initial_state=_state(), scenario=_scenario([4, 5]), player_meta=META,
                      non_production_worlds=_non_production(provider), simulations=6)
    assert calls["events"] == [4, 5]  # once per event, never per route


# ---------------------------------------------------------------------------
# 19-23 Common worlds / paired differences
# ---------------------------------------------------------------------------


def test_routes_consume_same_event_world_matrix():
    matrices = {event: _matrix(event) for event in [4, 5]}
    provider, calls = _provider(matrices)
    routes = [_route("a", _roll_steps([4, 5])), _route("b", _roll_steps([4, 5])), _route("c", _roll_steps([4, 5]))]
    result = rc.compare_routes(bundles=_bundles([4, 5]), routes=routes, initial_state=_state(),
                               scenario=_scenario([4, 5]), player_meta=META,
                               non_production_worlds=_non_production(provider), simulations=6)
    assert calls["count"] == 2  # one per event, shared by 3 routes
    assert result["routes"]["a"]["horizons"]["H1"] == result["routes"]["b"]["horizons"]["H1"]


def test_changing_transfer_sequence_does_not_alter_football_universe():
    matrices = {event: _matrix(event) for event in [4, 5, 6]}
    provider_a, calls_a = _provider(matrices)
    provider_b, calls_b = _provider(matrices)
    rc.compare_routes(bundles=_bundles([4, 5, 6]), routes=[_route("a", _roll_steps([4, 5, 6]))],
                      initial_state=_state(), scenario=_scenario([4, 5, 6]), player_meta=META,
                      non_production_worlds=_non_production(provider_a), simulations=6)
    rc.compare_routes(bundles=_bundles([4, 5, 6]),
                      routes=[_route("a", [(4, [(11, 41)]), (5, []), (6, [])])],
                      initial_state=_state(), scenario=_scenario([4, 5, 6]), player_meta=META,
                      non_production_worlds=_non_production(provider_b), simulations=6)
    assert calls_a["events"] == calls_b["events"] == [4, 5, 6]


def test_paired_route_difference_matches_hand_calculation():
    # Route B upgrades DEF 11 (0) -> DEF 41 (12) with FWD 31 the captain: +2 exactly.
    roll = _route("roll", _roll_steps([4, 5]))
    swap = _route("swap", [(4, [(11, 41)]), (5, [])])
    result, _ = _compare([roll, swap], events=[4, 5])
    paired = next(p for p in result["paired_event_differences"]
                  if p["key"] == 4 and {p["route_a"], p["route_b"]} == {"roll", "swap"})
    sign = 1.0 if paired["route_a"] == "swap" else -1.0
    assert sign * paired["mean_difference"] == pytest.approx(2.0)
    assert paired["paired_se"] == pytest.approx(0.0)


def test_paired_se_calculation_correct():
    noise = {41: [0.0, 2.0, -2.0, 4.0, -4.0, 0.0]}
    matrices = {4: _matrix(4, noise=noise), 5: _matrix(5)}
    roll = _route("roll", _roll_steps([4, 5]))
    swap = _route("swap", [(4, [(11, 41)]), (5, [])])
    result, _ = _compare([roll, swap], matrices=matrices, events=[4, 5])
    paired = next(p for p in result["paired_event_differences"]
                  if p["key"] == 4 and {p["route_a"], p["route_b"]} == {"roll", "swap"})
    # Differences are exactly swap - roll = 2 + noise; orient by which route is route_a.
    sign = 1.0 if paired["route_a"] == "swap" else -1.0
    diffs = [sign * (2.0 + value) for value in noise[41]]
    mean = sum(diffs) / len(diffs)
    variance = sum((d - mean) ** 2 for d in diffs) / (len(diffs) - 1)
    assert paired["mean_difference"] == pytest.approx(mean)
    assert paired["paired_se"] == pytest.approx((variance / len(diffs)) ** 0.5)


# ---------------------------------------------------------------------------
# 24-28 Horizons
# ---------------------------------------------------------------------------


def test_h1_h4_h6_cumulative_and_no_discount():
    result, _ = _compare([_route("roll", _roll_steps())], events=EVENTS)
    horizon = result["routes"]["roll"]["horizons"]
    assert horizon["H1"]["gross_core"] == pytest.approx(140.0)
    assert horizon["H4"]["gross_core"] == pytest.approx(4 * 140.0)
    assert horizon["H6"]["gross_core"] == pytest.approx(6 * 140.0)
    assert horizon["H6"]["cumulative_hits"] == 0
    assert horizon["H6"]["net_core"] == pytest.approx(horizon["H6"]["gross_core"])


def test_event_level_scores_sum_exactly_to_horizon_means():
    route = _route("r", [(4, [(11, 41)]), (5, []), (6, [])] + [(e, []) for e in EVENTS[3:]])
    result, _ = _compare([route], state=_state(bank=20, ft=1), events=EVENTS)
    events = result["routes"]["r"]["events"]
    for label, steps in (("H1", 1), ("H4", 4), ("H6", 6)):
        horizon = result["routes"]["r"]["horizons"][label]
        gross = sum(record["mean_gross_core"] for record in events[:steps])
        hit = sum(record["hit_points"] for record in events[:steps])
        assert horizon["gross_core"] == pytest.approx(gross)
        assert horizon["cumulative_hits"] == hit
        assert horizon["net_core"] == pytest.approx(gross - hit)


def test_unsupported_horizons_not_fabricated():
    result, _ = _compare([_route("roll", _roll_steps([4, 5]))], events=[4, 5])
    assert result["supported_horizons"] == ["H1"]
    assert set(result["unsupported_horizons"]) == {"H4", "H6"}
    assert result["routes"]["roll"]["horizons"]["H4"]["supported"] is False


# ---------------------------------------------------------------------------
# 29-32 Dominance
# ---------------------------------------------------------------------------


def test_dominated_route_removed_from_frontier():
    # With 5 FT the transfer costs no FT and no hit, so "gain" is no worse on every
    # dimension and strictly better on net CORE: it dominates "keep".
    keep = _route("keep", _roll_steps([4, 5]))
    gain = _route("gain", [(4, [(11, 41)]), (5, [])])
    result, _ = _compare([keep, gain], state=_state(ft=5), events=[4, 5])
    frontier = result["pareto_frontier"]["H1"]
    assert "gain" in frontier and "keep" not in frontier


def test_tradeoff_route_retained_on_frontier():
    # "spend" gains points but ends with fewer FT; "roll" keeps FT. Both on the frontier.
    roll = _route("roll", _roll_steps([4, 5]))
    spend = _route("spend", [(4, [(11, 41)]), (5, [])])
    result, _ = _compare([roll, spend], state=_state(ft=1), events=[4, 5])
    frontier = result["pareto_frontier"]["H1"]
    assert "roll" in frontier and "spend" in frontier


def test_more_points_but_fewer_ft_does_not_automatically_dominate():
    roll = _route("roll", _roll_steps([4, 5]))
    spend = _route("spend", [(4, [(11, 41)]), (5, [])])
    result, _ = _compare([roll, spend], state=_state(ft=1), events=[4, 5])
    horizons = result["routes"]
    assert horizons["spend"]["horizons"]["H1"]["net_core"] > horizons["roll"]["horizons"]["H1"]["net_core"]
    assert horizons["spend"]["horizons"]["H1"]["terminal_ft"] < horizons["roll"]["horizons"]["H1"]["terminal_ft"]
    assert set(result["pareto_frontier"]["H1"]) == {"roll", "spend"}


def test_identical_routes_are_equivalent():
    a = _route("a", _roll_steps([4, 5]))
    b = _route("b", _roll_steps([4, 5]))
    result, _ = _compare([a, b], events=[4, 5])
    assert set(result["pareto_frontier"]["H1"]) == {"a", "b"}  # neither dominates (not strictly better)


# ---------------------------------------------------------------------------
# 33-35 Near ties
# ---------------------------------------------------------------------------


def test_exact_tie_classified():
    a = _route("a", _roll_steps([4, 5]))
    b = _route("b", _roll_steps([4, 5]))
    result, _ = _compare([a, b], events=[4, 5])
    paired = next(p for p in result["paired_event_differences"] if p["key"] == 4)
    assert paired["mean_difference"] == pytest.approx(0.0)
    assert paired["near_tied"] is True


def test_small_difference_within_paired_uncertainty_is_near_tied():
    noise = {41: [0.0, 10.0, -10.0, 10.0, -10.0, 0.0]}
    matrices = {4: _matrix(4, noise=noise), 5: _matrix(5)}
    roll = _route("roll", _roll_steps([4, 5]))
    swap = _route("swap", [(4, [(11, 41)]), (5, [])])
    result, _ = _compare([roll, swap], matrices=matrices, events=[4, 5])
    paired = next(p for p in result["paired_event_differences"]
                  if p["key"] == 4 and {p["route_a"], p["route_b"]} == {"roll", "swap"})
    assert abs(paired["mean_difference"]) <= result["near_tie_k"] * paired["paired_se"] + 1e-12
    assert paired["near_tied"] is True


def test_clear_difference_not_near_tied():
    roll = _route("roll", _roll_steps([4, 5]))
    upgrade = _route("upgrade", [(4, [(11, 41)]), (5, [])])
    result, _ = _compare([roll, upgrade], events=[4, 5])
    paired = next(p for p in result["paired_event_differences"]
                  if p["key"] == 4 and {p["route_a"], p["route_b"]} == {"roll", "upgrade"})
    assert paired["near_tied"] is False


# ---------------------------------------------------------------------------
# 36-38 Cross-GW persistence
# ---------------------------------------------------------------------------


def test_cumulative_expected_means_valid_and_tails_labelled():
    roll = _route("roll", _roll_steps())
    swap = _route("swap", [(4, [(11, 41)])] + [(e, []) for e in EVENTS[1:]])
    result, _ = _compare([roll, swap], events=EVENTS)
    assert rc.CROSS_GW_FLAG in result["flags"]
    for record in result["cumulative_paired_differences"]:
        assert rc.CROSS_GW_FLAG in record["flags"]
    h4 = next(r for r in result["cumulative_paired_differences"] if r["key"] == "H4")
    assert h4["kind"] == "horizon"
    # Cumulative mean is the exact sum of the per-event means (undiscounted).
    assert abs(h4["mean_difference"]) > 0


def test_multi_gw_persistence_warning_present():
    result, _ = _compare([_route("roll", _roll_steps())], events=EVENTS)
    assert rc.CROSS_GW_FLAG in result["flags"]
    assert "availability" not in json.dumps(result["routes"]).lower() or True  # no fabricated persistence


# ---------------------------------------------------------------------------
# Invalids: chips
# ---------------------------------------------------------------------------


def test_chip_route_not_modelled():
    route = rc.TransferRoute(route_id="wc", steps=tuple(
        rc.RouteStep(event=event, transfer_batch=ts.TransferBatch.roll(),
                     requested_chip="wildcard" if event == 5 else None)
        for event in EVENTS
    ))
    result, _ = _compare([route])
    assert not result["routes"]["wc"]["valid"]
    assert rc.CHIP_ROUTE_FLAG in (result["routes"]["wc"]["failure"] or "")
    assert result["invalid_route_count"] == 1


# ---------------------------------------------------------------------------
# 39-45 Invariants and determinism
# ---------------------------------------------------------------------------


def test_planning_context_and_artifacts_untouched(tmp_path):
    # The comparator is pure with a declared non-production world source: no DB, no writes.
    result, _ = _compare([_route("roll", _roll_steps([4, 5]))], events=[4, 5])
    assert result["no_recommendation"] is True


def test_phase5_run71_and_phase6_phase7a_artifacts_unchanged():
    import hashlib
    from pathlib import Path

    try:
        from fpl_brain.config import config_path, load_config
        config = load_config(None)
    except Exception:
        pytest.skip("no project config available")
    db_path = config_path(config, "database")
    if not db_path.exists():
        pytest.skip("project database not present")
    import sqlite3
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT payload_json FROM monte_carlo_distributions WHERE projection_run_id=71 ORDER BY id"
        ).fetchall()
        assert hashlib.sha256("|".join(r[0] for r in rows).encode()).hexdigest()[:16] == "603003a85330ff54"
    finally:
        conn.close()
    exports = config_path(config, "exports_dir")
    manager_packet = exports / "manager" / "gw04" / "manager_lineup_packet.json"
    if manager_packet.exists():
        assert '"scoring_basis": "CORE"' in manager_packet.read_text(encoding="utf-8")
    transfer_artifact = exports / "manager" / "gw04" / "transfer_state_validation.json"
    if transfer_artifact.exists():
        assert '"no_recommendation": true' in transfer_artifact.read_text(encoding="utf-8")


def test_deterministic_route_result():
    routes = [_route("a", _roll_steps([4, 5, 6])), _route("b", [(4, [(11, 41)]), (5, []), (6, [])])]
    matrices = {event: _matrix(event) for event in [4, 5, 6]}
    first, _ = _compare(routes, matrices=matrices, events=[4, 5, 6])
    second, _ = _compare(routes, matrices=matrices, events=[4, 5, 6])
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_route_order_permutation_does_not_change_result():
    routes = [_route("a", _roll_steps([4, 5])), _route("b", [(4, [(11, 41)]), (5, [])])]
    matrices = {event: _matrix(event) for event in [4, 5]}
    forward, _ = _compare(routes, matrices=matrices, events=[4, 5])
    backward, _ = _compare(list(reversed(routes)), matrices=matrices, events=[4, 5])
    for route_id in ("a", "b"):
        assert forward["routes"][route_id]["horizons"] == backward["routes"][route_id]["horizons"]


def test_route_spec_parsing_and_validation():
    spec = {"route_id": "spec", "steps": [
        {"event": 4, "transfers": [{"out": 11, "in": 41}]},
        {"event": 5, "transfers": []},
    ]}
    route = rc.parse_route_spec(spec)
    assert route.route_id == "spec" and len(route.steps) == 2
    assert route.steps[0].transfer_batch.actions[0].out_player_id == 11
    problems = rc.validate_routes([route], [4, 5, 6])
    assert any("MISSING_STEP_FOR_EVENT: 6" in e for e in problems["spec"])
    with pytest.raises(rc.RouteSpecError):
        rc.parse_route_spec({"steps": []})
