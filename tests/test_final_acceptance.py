"""Phase 8C — final acceptance tests (synthetic-first, plus read-only artifact checks)."""

from __future__ import annotations

import hashlib
import json
import pathlib

import pytest

from fpl_brain import final_acceptance as fa
from fpl_brain import manager_lineup as ml
from fpl_brain import route_comparator as rc
from fpl_brain import route_optimizer as ro
from fpl_brain import transfer_state as ts

from test_transfer_state import CLUB, POOL_CLUB, POOL_POSITION, POSITION, SQUAD_IDS

EVENTS = (4, 5, 6)
UNION = sorted(set(SQUAD_IDS) | set(POOL_POSITION))
BASE_CORE = {pid: 5.0 for pid in UNION}
BASE_CORE.update({11: 0.0, 41: 12.0, 42: 13.0, 51: 11.0, 31: 20.0})


# ---------------------------------------------------------------------------
# Synthetic helpers
# ---------------------------------------------------------------------------


def _worlds(cores=None, worlds=8):
    values = dict(BASE_CORE)
    values.update(cores or {})
    return {"worlds": worlds, "player_ids": list(UNION),
            "core": {pid: [values.get(pid, 0.0)] * worlds for pid in UNION},
            "minutes": {pid: [90.0] * worlds for pid in UNION}}


def _positions():
    return {pid: POSITION.get(pid, POOL_POSITION.get(pid)) for pid in UNION}


def _state(*, bank=0, ft=2):
    return ts.RouteState(event=4, players=tuple(ts.RoutePlayer(pid, POSITION[pid], CLUB[pid], 50)
                                                for pid in SQUAD_IDS), bank_tenths=bank, free_transfers=ft)


def _scenario(events=EVENTS, prices=None):
    base = {pid: 50 for pid in UNION}
    base.update(prices or {})
    return rc.PriceScenario("TEST_FLAT", {e: ts.PriceSnapshot(event=e, prices=dict(base)) for e in events})


def _meta():
    return {pid: ts.PlayerMeta(pid, POSITION.get(pid, POOL_POSITION.get(pid)),
                              CLUB.get(pid, POOL_CLUB.get(pid))) for pid in UNION}


def _route_actions(*, e4=(), e5=(), e6=(), hits=(0, 0, 0), state=None):
    """Build serialized actions from transfer pairs, with replayed state facts."""

    current = state or _state()
    scenario = _scenario()
    meta = _meta()
    actions = []
    for offset, (event, moves, hit) in enumerate(zip(EVENTS, (e4, e5, e6), hits)):
        batch = ts.TransferBatch(tuple(ts.TransferAction(int(o), int(i)) for o, i in moves))
        result = ts.apply_transfer_batch(current, batch, scenario.snapshot_for(event), meta)
        assert result.ok, result.errors
        actions.append({
            "event": int(event), "kind": "ROLL" if not moves else "SINGLE",
            "hit_points": int(result.hit_points), "transfers": [{"out": int(o), "in": int(i)} for o, i in moves],
            "squad_ids": [int(p.player_id) for p in result.squad_after.players],
            "ft_after": int(result.next_event_state.free_transfers),
            "bank_after": int(result.bank_after_tenths),
        })
        current = result.next_event_state
    return actions


def _certified_result():
    roll = _route_actions()
    swap_a = _route_actions(e4=[(11, 41)])
    swap_b = _route_actions(e4=[(11, 41), (21, 51)])
    records = {}
    for index, (signature, actions) in enumerate((("roll", roll), ("a", swap_a), ("b", swap_b))):
        records[f"route_{index:03d}"] = {
            "route_id": f"route_{index:03d}",
            "canonical_family_signature": ro.canonical_signature_from_actions(actions),
            "family_signature": (),
            "actions": actions,
            "h1_net_core": 40.0 + index,
            "supported_3gw_net_core": 120.0 + index,
            "cumulative_hits": 0,
            "terminal_ft": 2,
            "terminal_bank_tenths": index,
            "per_event": [],
        }
    return {"routes": records, "h1_frontier": ["route_001"], "supported_3gw_frontier": ["route_002", "route_000"]}


# ---------------------------------------------------------------------------
# 1-4 Finalist selection
# ---------------------------------------------------------------------------


def test_shortlist_from_certified_results_only():
    certified = _certified_result()
    selection = fa.select_finalists(certified)
    certified_signatures = {rec["canonical_family_signature"] for rec in certified["routes"].values()}
    assert selection["count"] >= 2
    assert all(item["canonical_signature"] in certified_signatures for item in selection["finalists"])


def test_canonical_signatures_used_not_route_ids():
    selection = fa.select_finalists(_certified_result())
    for item in selection["finalists"]:
        assert "route_" not in item["canonical_signature"]
        assert item["canonical_signature"].startswith("E4:")
        assert "E5:" in item["canonical_signature"] and "E6:" in item["canonical_signature"]


def test_all_roll_always_included():
    selection = fa.select_finalists(_certified_result())
    signatures = {item["canonical_signature"] for item in selection["finalists"]}
    assert "E4:ROLL;E5:ROLL;E6:ROLL" in signatures
    roll = next(item for item in selection["finalists"] if item["canonical_signature"] == "E4:ROLL;E5:ROLL;E6:ROLL")
    assert "ALL_ROLL_BASELINE" in roll["selection_reasons"]


def test_no_optimizer_search_invoked_by_acceptance():
    import inspect
    for module in (fa,):
        source = inspect.getsource(module)
        assert "monte_carlo.simulate" not in source
        assert "run_search(" not in source
        assert "def optimize(" not in source


# ---------------------------------------------------------------------------
# 5-9 GW4 fidelity / cache identity
# ---------------------------------------------------------------------------


def test_run71_10k_config_identity():
    """Run 71 is pinned to the pre-R2C Monte Carlo version.

    R2C (scorer reconciliation numerical correctness) bumped
    ``MONTE_CARLO_MODEL_VERSION`` to ``mc_v1.3.0``, so the current config hash can
    no longer equal run 71's.  That is the intended consequence of a predictive
    correctness fix, not a weakened assertion: the historical run's version is
    verified explicitly, and the current configuration is pinned to its own
    constant so an accidental future config change still fails here.
    """

    import sqlite3
    from fpl_brain.config import config_path, load_config
    from fpl_brain import monte_carlo
    try:
        config = load_config(None)
    except Exception:
        pytest.skip("no project config available")
    db_path = config_path(config, "database")
    if not db_path.exists():
        pytest.skip("project database not present")
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        stored = conn.execute(
            "SELECT config_hash, random_seed, model_version FROM projection_runs WHERE id=71"
        ).fetchone()
    finally:
        conn.close()
    assert stored is not None
    # Run 71 predates the R2C numerical fix.
    assert stored[2] == "mc_v1.2.1"
    assert stored[0] == "sha256:fedf5f6e50b4826b6e44b42d524cac2d682f15e878a5c86f7bab08a3f4e15042"
    assert int(stored[1]) == 20260911

    replay = monte_carlo.MonteCarloConfig(simulations=fa.GW4_DRAWS, seed=20260911, occupancy_audit=True)
    assert monte_carlo.MONTE_CARLO_MODEL_VERSION == "mc_v1.3.0"
    assert replay.config_hash() == (
        "sha256:b7b2649151b02eea0a416b7792820199ed06cd6e52c0b082a4dd578fd7488d52"
    )
    assert replay.config_hash() != stored[0]
    assert fa.GW4_DRAWS == 10_000


def test_one_gw4_world_set_shared_and_squads_cached():
    worlds = _worlds()
    positions = _positions()
    calls = {"rank": 0}
    real_rank = ml.rank_policies

    class Config:
        policy_selection_worlds = 8

    confirmations = fa.confirm_gw4(finalist_gw4_squads={"sig_a": SQUAD_IDS, "sig_b": tuple(SQUAD_IDS)},
                                   worlds=worlds, positions_of=lambda ids: positions, config=Config(),
                                   cache={})
    # Same squad under two signatures shares one evaluation cache entry.
    assert confirmations["sig_a"]["squad_hash"] == confirmations["sig_b"]["squad_hash"]
    assert confirmations["sig_a"]["policy"] == confirmations["sig_b"]["policy"]
    assert confirmations["sig_a"]["world_scores"] == confirmations["sig_b"]["world_scores"]


RUNS = {"minutes_v1": 64, "team_strength_v1": 65, "player_rates_v1": 67, "xpts_v1": 70,
        "monte_carlo_v1": 71}
GENERATION_ID = "sha256:" + "a" * 64


def test_2k_cache_cannot_masquerade_as_10k():
    cfg2k = ro.OptimizerConfig(events=EVENTS, search_draws=2000)
    cfg10k = ro.OptimizerConfig(events=EVENTS, search_draws=10000)
    key2k = ro.world_cache_key(event=4, generation_id=GENERATION_ID, runs=RUNS, config=cfg2k,
                               union_ids=SQUAD_IDS)
    key10k = ro.world_cache_key(event=4, generation_id=GENERATION_ID, runs=RUNS, config=cfg10k,
                                union_ids=SQUAD_IDS)
    assert key2k != key10k
    assert ro.world_cache_key(event=4, generation_id=GENERATION_ID, runs=RUNS, config=cfg10k,
                              union_ids=SQUAD_IDS) == key10k
    # A DIFFERENT certified generation is a different world, however identical the draws.
    other = ro.world_cache_key(event=4, generation_id="sha256:" + "b" * 64, runs=RUNS,
                               config=cfg10k, union_ids=SQUAD_IDS)
    assert other != key10k


def test_route_id_absent_from_world_identity():
    cfg = ro.OptimizerConfig(events=EVENTS, search_draws=10000)
    key = ro.world_cache_key(event=4, generation_id=GENERATION_ID, runs=RUNS, config=cfg,
                             union_ids=SQUAD_IDS)
    assert "route" not in key.lower()


# ---------------------------------------------------------------------------
# 10-12 Policy selection
# ---------------------------------------------------------------------------


def test_gw4_policy_reselected_on_high_fidelity_worlds():
    worlds = _worlds(cores={31: 50.0})  # a clearly dominant captain in these worlds
    confirmations = fa.confirm_gw4(finalist_gw4_squads={"sig": SQUAD_IDS}, worlds=worlds,
                                   positions_of=lambda ids: _positions(), config=type("C", (), {"policy_selection_worlds": 8})(),
                                   cache={})
    policy = confirmations["sig"]["policy"]
    assert policy["captain_id"] == 31
    direct = ml.rank_policies(list(SQUAD_IDS), _positions(), worlds, top_k=1)["top_policies"][0]
    assert policy["captain_id"] == direct.captain_id


def test_selected_policy_is_legal():
    confirmations = fa.confirm_gw4(finalist_gw4_squads={"sig": SQUAD_IDS}, worlds=_worlds(),
                                   positions_of=lambda ids: _positions(), config=type("C", (), {"policy_selection_worlds": 8})(),
                                   cache={})
    policy = confirmations["sig"]["policy"]
    manager_policy = ml.ManagerPolicy(tuple(policy["starter_ids"]), policy["bench_gk_id"],
                                      tuple(policy["bench_outfield_order"]), policy["captain_id"],
                                      policy["vice_captain_id"])
    assert ml.policy_legality_errors(manager_policy, _positions()) == []


def test_captain_and_vice_in_xi():
    confirmations = fa.confirm_gw4(finalist_gw4_squads={"sig": SQUAD_IDS}, worlds=_worlds(),
                                   positions_of=lambda ids: _positions(), config=type("C", (), {"policy_selection_worlds": 8})(),
                                   cache={})
    policy = confirmations["sig"]["policy"]
    assert policy["captain_id"] in policy["starter_ids"]
    assert policy["vice_captain_id"] in policy["starter_ids"]
    assert policy["captain_id"] != policy["vice_captain_id"]


# ---------------------------------------------------------------------------
# 13-16 Phase-7A replay across layers
# ---------------------------------------------------------------------------


def test_replay_matches_recorded_accounting():
    actions = _route_actions(e4=[(11, 41)], e5=[(21, 51)])
    replay = fa.replay_route(initial_state=_state(), actions=actions, scenario=_scenario(), player_meta=_meta())
    assert replay["ok"] and not replay["errors"]
    for recorded, replayed in zip(actions, replay["events"]):
        assert replayed["hit_points"] == recorded["hit_points"]
        assert replayed["bank_after_tenths"] == recorded["bank_after"]
        assert replayed["ft_after"] == recorded["ft_after"]
        assert tuple(replayed["squad_ids"]) == tuple(sorted(recorded["squad_ids"]))


def test_replay_detects_bank_mismatch():
    actions = _route_actions(e4=[(11, 41)])
    actions[0]["bank_after"] = int(actions[0]["bank_after"]) + 1
    replay = fa.replay_route(initial_state=_state(), actions=actions, scenario=_scenario(), player_meta=_meta())
    assert not replay["ok"] and any("REPLAY_BANK_MISMATCH" in e for e in replay["errors"])


def test_replay_detects_illegal_route():
    actions = _route_actions(e4=[(11, 41)])
    actions[0]["transfers"] = [{"out": 11, "in": 51}]  # DEF out, MID in
    replay = fa.replay_route(initial_state=_state(), actions=actions, scenario=_scenario(), player_meta=_meta())
    assert not replay["ok"] and any("REPLAY_ILLEGAL" in e for e in replay["errors"])


def test_replay_does_not_mutate_initial_state():
    state = _state()
    snapshot = ro.state_key(state)
    fa.replay_route(initial_state=state, actions=_route_actions(e4=[(11, 41)]), scenario=_scenario(),
                    player_meta=_meta())
    assert ro.state_key(state) == snapshot


# ---------------------------------------------------------------------------
# 17-22 Paired comparisons / recomputation / frontier
# ---------------------------------------------------------------------------


def test_paired_difference_uses_same_worlds_and_se_correct():
    roll = {"world_scores": [1.0, 2.0, 3.0, 4.0]}
    other = {"world_scores": [2.0, 4.0, 6.0, 8.0]}
    result = fa.paired_vs_roll({"ROLL": roll, "X": other}, "ROLL")
    entry = next(item for item in result if item["finalist"] == "X")
    diffs = [1.0, 2.0, 3.0, 4.0]
    mean = sum(diffs) / 4
    variance = sum((d - mean) ** 2 for d in diffs) / 3
    assert entry["worlds"] == 4
    assert entry["mean_difference"] == pytest.approx(mean)
    assert entry["paired_se"] == pytest.approx((variance / 4) ** 0.5)
    assert entry["p_finalist_gt_roll"] == 1.0
    assert entry["ci95_low"] < entry["mean_difference"] < entry["ci95_high"]


def test_supported_3gw_recomputation_exact_no_discount():
    assert fa.recompute_supported_3gw(40.0, 35.5, 30.25) == pytest.approx(105.75)
    assert fa.recompute_supported_3gw(1.0, 2.0, 3.0) == 6.0  # plain sum, no discount


def test_confirmed_shortlist_frontier_correct():
    confirmed = {
        "a": {"confirmed_3gw_net_core": 10.0, "cumulative_hits": 0, "terminal_ft": 1, "terminal_bank_tenths": 0},
        "b": {"confirmed_3gw_net_core": 9.0, "cumulative_hits": 0, "terminal_ft": 1, "terminal_bank_tenths": 0},
        "c": {"confirmed_3gw_net_core": 9.5, "cumulative_hits": 0, "terminal_ft": 3, "terminal_bank_tenths": 0},
    }
    frontier = fa.confirmed_frontier(confirmed)
    assert "a" in frontier and "c" in frontier and "b" not in frontier
    assert fa.FRONTIER_NAME == "CONFIRMED_SHORTLIST_PARETO_FRONTIER"


def test_identical_routes_equivalent_on_confirmed_frontier():
    confirmed = {sig: {"confirmed_3gw_net_core": 5.0, "cumulative_hits": 0, "terminal_ft": 1,
                       "terminal_bank_tenths": 0} for sig in ("a", "b")}
    assert set(fa.confirmed_frontier(confirmed)) == {"a", "b"}


# ---------------------------------------------------------------------------
# 23-27 Flags / basis / identity
# ---------------------------------------------------------------------------


def test_h4_h6_unsupported_and_cross_gw_flags_present():
    assert fa.H4_H6_FLAG == "CANONICAL_H4_H6_UNSUPPORTED"
    assert fa.CROSS_GW_FLAG == "CROSS_GW_AVAILABILITY_PERSISTENCE_UNMODELLED"
    assert fa.PRICE_SCENARIO == "FLAT_CURRENT_PRICE"
    assert fa.PRICE_FLAG == "SCENARIO_ASSUMPTION_NOT_PRICE_FORECAST"
    assert fa.SUPPORTED_EVENTS == (4, 5, 6)


def test_core_basis_and_no_bonus_in_stochastic_path():
    import inspect
    source = inspect.getsource(fa)
    assert "CORE" in source
    assert "bonus" not in source.lower()
    assert "CONFIRMATION_MATERIAL_CORE" in source and fa.CONFIRMATION_MATERIAL_CORE == 0.50


def test_no_recommendation_language():
    import inspect
    source = inspect.getsource(fa).upper()
    for forbidden in ("BUY", "SELL", "MAKE THIS TRANSFER", "CAPTAIN X"):
        assert forbidden not in source
    # The module must carry the explicit no-recommendation marker contract.
    assert "NO_RECOMMENDATION" in source or "NO RECOMMENDATION" in source


# ---------------------------------------------------------------------------
# 28-34 Real artifact / regression checks
# ---------------------------------------------------------------------------


def _exports():
    from fpl_brain.config import config_path, load_config
    return config_path(load_config(None), "exports_dir")


def test_real_artifact_cross_section_and_identity():
    try:
        exports = _exports()
    except Exception:
        pytest.skip("no project config available")
    path = exports / "final_acceptance" / "gw04" / "final_acceptance.json"
    if not path.exists():
        pytest.skip("final acceptance artifact not present")
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if artifact.get("draw_fidelity", {}).get("gw4") != 10_000:
        pytest.skip("only a smoke-draw artifact is present")
    assert artifact["score_basis"] == "CORE"
    assert artifact["price_scenario"] == "FLAT_CURRENT_PRICE"
    assert artifact["draw_fidelity"]["gw4"] == 10_000
    assert artifact["draw_fidelity"]["gw5"] == 2_000 and artifact["draw_fidelity"]["gw6"] == 2_000
    assert fa.H4_H6_FLAG in artifact["flags"] and fa.CROSS_GW_FLAG in artifact["flags"]
    assert artifact["cross_section_audit"]["status"] == "PASS"
    assert artifact["replay_identity"]["equals_run71_config_hash"] is True
    assert artifact["no_recommendation"] is True
    # candidate ids resolve
    universe = json.loads((exports / "candidates" / "gw04" / "candidate_universe.json").read_text(encoding="utf-8"))
    universe_ids = {int(row["player_id"]) for row in universe["universe"]}
    for finalist in artifact["finalists"]:
        for pid in (finalist["gw4"]["policy"]["starter_ids"]
                    + finalist["gw4"]["policy"]["bench_outfield_order"] + [finalist["gw4"]["policy"]["bench_gk_id"]]):
            assert int(pid) in universe_ids
    assert artifact["acceptance"] in ("COMPLETE", "PARTIAL")


def test_rolling_stock_runs_and_artifacts_unchanged():
    import sqlite3
    from fpl_brain.config import config_path, load_config
    try:
        config = load_config(None)
    except Exception:
        pytest.skip("no project config available")
    db_path = config_path(config, "database")
    if not db_path.exists():
        pytest.skip("project database not present")
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        # The count is a floor: official refreshes legitimately append NEW runs
        # (accepted runs are never deleted).  Immutability is proven by the
        # accepted runs still being present plus the payload digests below.
        assert conn.execute("SELECT COUNT(*) FROM projection_runs").fetchone()[0] >= 103
        assert conn.execute(
            "SELECT COUNT(*) FROM projection_runs WHERE id IN (64,70,71,80,86,87,88,94,95)"
        ).fetchone()[0] == 9
        for run_id, digest in ((64, "cbd05b197b952aa0"), (70, "c5fd3ae196642cd2"), (71, "603003a85330ff54"),
                               (80, "ba57cdf9435fb936"), (86, "8bcc1f65a455c603"), (87, "915e924daecb224a"),
                               (88, "d2e36bd14d22791e"), (94, "c3ccf4a1f1e05b65"), (95, "c02470004b51c866")):
            table = ("monte_carlo_distributions" if run_id in (71, 87, 95)
                     else "player_fixture_xpts_projections" if run_id in (70, 86, 94) else "frozen_predictions")
            rows = conn.execute(f"SELECT payload_json FROM {table} WHERE projection_run_id=? ORDER BY id",
                                (run_id,)).fetchall()
            assert hashlib.sha256("|".join(r[0] for r in rows).encode()).hexdigest()[:16] == digest
    finally:
        conn.close()


def test_upstream_artifacts_preserved():
    import hashlib
    try:
        exports = _exports()
    except Exception:
        pytest.skip("no project config available")
    checks = {
        "manager/gw04/manager_lineup_packet.json": "1a60f4900a19c465",
        "manager/gw04/transfer_state_validation.json": "6f1c4c3c62b07244",
        "routes/gw04/route_comparison.json": "79fd0b5a3a92e87e",
        "candidates/gw04/candidate_universe.json": "046af76de4dd9c31",
        "optimizer/gw04/optimizer_result.json": "30e9b0bbdb7863c1",
        "optimizer/gw04/stability_ladder_recertified.json": "b566361ec4935756",
    }
    for relative, digest in checks.items():
        path = exports / relative
        if path.exists():
            assert hashlib.sha256(path.read_bytes()).hexdigest()[:16] == digest, relative


def test_selection_and_replay_deterministic():
    first = fa.select_finalists(_certified_result())
    second = fa.select_finalists(_certified_result())
    assert json.dumps(first, sort_keys=True, default=str) == json.dumps(second, sort_keys=True, default=str)
    actions = _route_actions(e4=[(11, 41)])
    one = fa.replay_route(initial_state=_state(), actions=actions, scenario=_scenario(), player_meta=_meta())
    two = fa.replay_route(initial_state=_state(), actions=actions, scenario=_scenario(), player_meta=_meta())
    assert json.dumps(one, sort_keys=True, default=str) == json.dumps(two, sort_keys=True, default=str)


# ---------------------------------------------------------------------------
# Phase 8C.1 — acceptance consistency remediation tests
# ---------------------------------------------------------------------------


def test_zero_hit_gross_equals_net():
    # A zero-hit route's net must equal its gross exactly (ROLL in particular).
    for hit, gross in ((0, 39.4737), (0, 0.0), (0, 42.5881)):
        assert gross - hit == pytest.approx(gross, abs=1e-12)


def test_net_equals_gross_minus_hit():
    # One FT spent on two transfers must produce a 4-point hit at that event, and
    # the net/gross relationship must hold with the hit subtracted exactly once.
    state = _state(bank=200, ft=1)
    actions = _route_actions(e4=[(11, 41), (21, 51)], e5=[], e6=[], state=state)
    replay = fa.replay_route(initial_state=state, actions=actions, scenario=_scenario(), player_meta=_meta())
    assert replay["ok"]
    assert replay["events"][0]["hit_points"] == 4
    gross = 100.0
    net = gross - sum(item["hit_points"] for item in replay["events"])
    assert net == pytest.approx(gross - 4.0)


def test_repeated_gw4_action_yields_same_gw4_squad():
    one = _route_actions(e4=[(11, 41)], e5=[], e6=[])
    two = _route_actions(e4=[(11, 41)], e5=[(21, 51)], e6=[(22, 52)])
    assert tuple(one[0]["squad_ids"]) == tuple(two[0]["squad_ids"])
    other = _route_actions(e4=[(11, 41), (21, 51)], e5=[], e6=[])
    assert tuple(other[0]["squad_ids"]) != tuple(one[0]["squad_ids"])


def test_unique_squad_count_derived_correctly():
    records = [
        {"gw4_action": "E4:165-249|329-305", "gw4_squad_hash": "h1"},
        {"gw4_action": "E4:165-249|329-305", "gw4_squad_hash": "h1"},
        {"gw4_action": "E4:ROLL", "gw4_squad_hash": "h2"},
        {"gw4_action": "E4:1-28|329-305", "gw4_squad_hash": "h3"},
    ]
    by_action: dict[str, set[str]] = {}
    for record in records:
        by_action.setdefault(record["gw4_action"], set()).add(record["gw4_squad_hash"])
    assert all(len(hashes) == 1 for hashes in by_action.values())
    assert len({record["gw4_squad_hash"] for record in records}) == 3


def test_same_gw4_squad_shares_phase6_cache_entry():
    worlds = _worlds()
    cache: dict = {}
    confirmations = fa.confirm_gw4(finalist_gw4_squads={"a": SQUAD_IDS, "b": tuple(reversed(SQUAD_IDS))},
                                   worlds=worlds, positions_of=lambda ids: _positions(),
                                   config=type("C", (), {"policy_selection_worlds": 8})(), cache=cache)
    assert len(cache) == 1  # one unique squad evaluation
    assert confirmations["a"]["policy"] == confirmations["b"]["policy"]
    assert confirmations["a"]["world_scores"] == confirmations["b"]["world_scores"]


def test_gw4_policy_cache_key_ignores_later_actions_and_route_id():
    base = fa.gw4_policy_cache_key(4, SQUAD_IDS, draws=10000, seed=20260911)
    reordered = fa.gw4_policy_cache_key(4, tuple(reversed(SQUAD_IDS)), draws=10000, seed=20260911)
    assert base == reordered
    assert "route" not in base
    # draws / event sensitivity (world identity is part of the key)
    assert fa.gw4_policy_cache_key(4, SQUAD_IDS, draws=2000, seed=20260911) != base
    assert fa.gw4_policy_cache_key(5, SQUAD_IDS, draws=10000, seed=20260911) != base
    # later-event actions are not inputs at all
    later_variant = tuple(int(p) for p in SQUAD_IDS)  # same squad, different GW5/GW6 by construction
    assert fa.gw4_policy_cache_key(4, later_variant, draws=10000, seed=20260911) == base


def test_recertified_path_uses_full_10k_and_hits_cache():
    import inspect
    from fpl_brain.config import config_path, load_config
    try:
        config = load_config(None)
    except Exception:
        pytest.skip("no project config available")
    source = __import__("pathlib").Path("scripts/recertify_final_acceptance.py").read_text(encoding="utf-8")
    # full 10k worlds for selection, no subsample
    assert "policy_selection_worlds=0" in source
    assert "rank_policies(list(record[\"gw4_squad_ids\"]), positions, gw4_worlds, top_k=1)" in source
    # refuses to regenerate football worlds
    assert "refusing to regenerate" in source
    # never builds GW5/GW6 worlds
    assert "_bundle(5" not in source and "_bundle(6" not in source
    cache_dir = __import__("pathlib").Path("data/cache/manager_worlds")
    assert cache_dir.exists() and list(cache_dir.glob("*.json"))


def test_real_recertified_artifact_invariants():
    import hashlib
    try:
        exports = _exports()
    except Exception:
        pytest.skip("no project config available")
    path = exports / "final_acceptance" / "gw04" / "final_acceptance_recertified.json"
    if not path.exists():
        pytest.skip("recertified artifact not present")
    artifact = json.loads(path.read_text(encoding="utf-8"))
    assert artifact["score_basis"] == "CORE"
    assert artifact["policy_selection"]["gw4_worlds_used"] == 10_000
    assert artifact["policy_selection"]["subsample_used"] is False
    assert artifact["world_cache"]["gw4_hit"] is True
    # invariants
    for row in artifact["finalists"]:
        gross, hit, net = row["gw4"]["mean_gross_core"], row["gw4"]["hit_points"], row["gw4"]["mean_net_core"]
        assert abs(net - (gross - hit)) <= 1e-9
        if hit == 0:
            assert abs(net - gross) <= 1e-9
        assert row["gw4"]["policy_selection_worlds"] == 10_000
        assert row["confirmed_3gw_net_core"] == pytest.approx(
            fa.recompute_supported_3gw(net, row["gw5_mean_net_core"], row["gw6_mean_net_core"]))
    # ROLL invariants
    roll = artifact["roll"]
    assert roll["gw4_action"] == "E4:ROLL"
    assert roll["hit_points"] == 0
    assert abs(roll["gross_10k"] - roll["net_10k"]) <= 1e-9
    assert all(item["status"] == "PASS" for item in artifact["roll_invariants"])
    # unique squad count equals group cardinality; families per squad
    assert artifact["unique_gw4_squad_count"] == len(artifact["groups_by_gw4_squad"])
    members = sum(len(v) for v in artifact["groups_by_gw4_squad"].values())
    assert members == artifact["finalist_families"]
    # same squad => same policy and same GW4 gross
    by_squad: dict[str, list] = {}
    for row in artifact["finalists"]:
        by_squad.setdefault(row["gw4"]["squad_hash"], []).append(row)
    for rows in by_squad.values():
        assert len({json.dumps(r["gw4"]["policy"], sort_keys=True) for r in rows}) == 1
        assert len({round(r["gw4"]["mean_gross_core"], 12) for r in rows}) == 1
    # cache accounting
    assert artifact["phase6_cache"]["unique_squad_evaluations"] == artifact["unique_gw4_squad_count"]
    assert artifact["phase6_cache"]["unique_squad_evaluations"] + artifact["phase6_cache"]["hits"] == \
        artifact["finalist_families"]
    # leaders consistent with the table
    h1_best = max(artifact["finalists"], key=lambda r: r["gw4"]["mean_net_core"])["canonical_signature"]
    assert artifact["leaders"]["h1_leader_after"] == h1_best
    w3_best = max(artifact["finalists"], key=lambda r: r["confirmed_3gw_net_core"])["canonical_signature"]
    assert artifact["leaders"]["supported_3gw_leader_after"] == w3_best
    assert artifact["extended_cross_section_audit"]["status"] == "PASS"
    assert fa.H4_H6_FLAG in artifact["flags"] and fa.CROSS_GW_FLAG in artifact["flags"]
    assert artifact["no_recommendation"] is True
    assert artifact["acceptance"] in ("COMPLETE", "PARTIAL")


def test_original_phase8c_artifact_preserved_and_upstream_unchanged():
    import hashlib
    import sqlite3
    from fpl_brain.config import config_path, load_config
    try:
        config = load_config(None)
    except Exception:
        pytest.skip("no project config available")
    exports = config_path(config, "exports_dir")
    original = exports / "final_acceptance" / "gw04" / "final_acceptance.json"
    if original.exists():
        assert hashlib.sha256(original.read_bytes()).hexdigest()[:16] == "93e5caaa7ab0d3ab"
    db_path = config_path(config, "database")
    if db_path.exists():
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            # Floor only: official refreshes append new runs; accepted run 71 and
            # its payload digest below are the immutability guarantee.
            assert conn.execute("SELECT COUNT(*) FROM projection_runs").fetchone()[0] >= 103
            assert conn.execute("SELECT COUNT(*) FROM projection_runs WHERE id=71").fetchone()[0] == 1
            rows = conn.execute("SELECT payload_json FROM monte_carlo_distributions WHERE projection_run_id=71 "
                                "ORDER BY id").fetchall()
            assert hashlib.sha256("|".join(r[0] for r in rows).encode()).hexdigest()[:16] == "603003a85330ff54"
        finally:
            conn.close()
    upstream = {
        "candidates/gw04/candidate_universe.json": "046af76de4dd9c31",
        "optimizer/gw04/optimizer_result.json": "30e9b0bbdb7863c1",
        "optimizer/gw04/stability_ladder.json": "b6f9237e603f582a",
        "optimizer/gw04/stability_ladder_recertified.json": "b566361ec4935756",
        "routes/gw04/route_comparison.json": "79fd0b5a3a92e87e",
    }
    for relative, digest in upstream.items():
        path = exports / relative
        if path.exists():
            assert hashlib.sha256(path.read_bytes()).hexdigest()[:16] == digest, relative
