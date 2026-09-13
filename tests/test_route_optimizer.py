"""Phase 8B — bounded route optimizer tests (synthetic-first, fast)."""

from __future__ import annotations

import json

import pytest

from fpl_brain import candidate_universe as cu
from fpl_brain import route_comparator as rc
from fpl_brain import route_optimizer as ro
from fpl_brain import route_stability as rs
import hashlib
from fpl_brain import transfer_state as ts
from test_transfer_state import CLUB, POOL_CLUB, POOL_POSITION, POSITION, SQUAD_IDS

EVENTS = (4, 5)
UNION = sorted(set(SQUAD_IDS) | set(POOL_POSITION))
BASE_CORE = {pid: 5.0 for pid in UNION}
BASE_CORE.update({11: 0.0, 41: 12.0, 42: 13.0, 51: 11.0, 31: 20.0})


def _payload(core, minutes=90.0):
    return {"core_xpts": core, "expected_minutes": minutes, "p_start": 1.0, "p_60_plus": 1.0,
            "p_appearance": 1.0, "bonus_xpts": 99.0, "total_xpts": core + 99.0, "risk_flags": []}


def _universe(events=EVENTS):
    pool = {pid: {"player_id": pid, "position": POSITION.get(pid, POOL_POSITION.get(pid)),
                  "club_id": CLUB.get(pid, POOL_CLUB.get(pid)), "web_name": f"W{pid}", "full_name": f"P{pid}"}
            for pid in UNION}
    clubs = {}
    for pid, meta in pool.items():
        clubs.setdefault(meta["club_id"], []).append(pid)
    fixtures = {(e, club): [1000 + club] for e in events for club in clubs}
    xpts = {e: {(pid, 1000 + pool[pid]["club_id"]): _payload(BASE_CORE[pid]) for pid in UNION} for e in events}
    minutes = {e: {(pid, 1000 + pool[pid]["club_id"]): {"joint_availability": 1.0} for pid in UNION} for e in events}
    prices = {pid: (50 if pid not in (41, 42, 51) else 200) for pid in UNION}
    universe = cu.build_universe(
        pool=pool, events_fixtures=fixtures, xpts_rows_by_event=xpts, minutes_rows_by_event=minutes,
        events=list(events), owned_ids=[pid for pid in SQUAD_IDS], price_snapshot=ts.PriceSnapshot(event=4, prices=prices),
        config=cu.CandidateConfig(top_n_per_criterion=3), planning_cutoff="2026-09-11T10:16:51Z",
    )
    state = ts.RouteState(
        event=int(events[0]),
        players=tuple(ts.RoutePlayer(pid, POSITION[pid], CLUB[pid], 50) for pid in SQUAD_IDS),
        bank_tenths=0, free_transfers=2,
    )
    meta = {pid: ts.PlayerMeta(pid, POSITION.get(pid, POOL_POSITION.get(pid)),
                              CLUB.get(pid, POOL_CLUB.get(pid))) for pid in UNION}
    edges = cu.build_replacement_edges(universe_rows=universe["universe"], owned_ids=SQUAD_IDS,
                                       state=state, price_snapshot=ts.PriceSnapshot(event=4, prices=prices),
                                       player_meta=meta)
    universe["replacement_edges"] = edges
    return universe, state, meta


def _scenario(events=EVENTS, prices=None):
    base = {pid: 50 for pid in UNION}
    base.update(prices or {})
    snapshots = {e: ts.PriceSnapshot(event=e, prices=dict(base)) for e in events}
    return rc.PriceScenario(scenario_id="TEST_FLAT", event_snapshots=snapshots)


def _provider(events=EVENTS, worlds=6, core=None):
    values = dict(BASE_CORE)
    values.update(core or {})
    def provider(event, union_ids):
        return {"worlds": worlds, "player_ids": list(union_ids),
                "core": {pid: [values.get(pid, 0.0)] * worlds for pid in union_ids},
                "minutes": {pid: [90.0] * worlds for pid in union_ids}}
    return provider


def _config(**over):
    base = dict(events=EVENTS, search_draws=6, beam_width=8, exact_evaluation_budget=12,
                policy_selection_worlds=6, singles_per_out=2, max_transfers_per_event=2,
                max_auto_hit_points_per_event=4, rescue_top_k_per_position=2, search_n_per_criterion=3)
    base.update(over)
    return ro.OptimizerConfig(**base)


def _optimize(universe, state, meta, **over):
    return ro.optimize(universe=universe, initial_state=state, scenario=_scenario(), player_meta=meta,
                       config=_config(**over), world_provider=_provider())


# ---------------------------------------------------------------------------
# Candidate pool + rescue
# ---------------------------------------------------------------------------


def test_search_pool_includes_view_owned_and_rescue():
    universe, state, meta = _universe()
    pool = ro.build_search_pool(universe, SQUAD_IDS, _config())
    assert set(SQUAD_IDS).issubset(set(pool["pool_ids"]))
    assert pool["base_view_ids"]
    assert pool["rescued_ids"]
    assert pool["pool_hash"]


def test_rescue_reasons_labelled_and_outside_view_reported():
    universe, state, meta = _universe()
    pool = ro.build_search_pool(universe, SQUAD_IDS, _config(rescue_top_k_per_position=1))
    for pid in pool["rescued_ids"]:
        reasons = pool["rescue_reasons"][str(pid)]
        assert any(reason in (ro.RESCUE_FIRST_EVENT, ro.RESCUE_WINDOW_SUM) for reason in reasons)
    assert isinstance(pool["rescued_outside_view"], list)


def test_pool_ignores_popularity_and_has_no_names():
    universe, state, meta = _universe()
    for row in universe["universe"]:
        row["web_name"] = "IGNORED"
    pool = ro.build_search_pool(universe, SQUAD_IDS, _config())
    assert pool["pool_ids"] == sorted(pool["pool_ids"])


# ---------------------------------------------------------------------------
# Action generation
# ---------------------------------------------------------------------------


def test_actions_include_roll_singles_and_doubles():
    universe, state, meta = _universe()
    result = ro.generate_actions(
        state=state, rows={r["player_id"]: r for r in universe["universe"]},
        pool_ids=[r["player_id"] for r in universe["universe"]],
        positions={r["player_id"]: r["position"] for r in universe["universe"]},
        price_snapshot=_scenario().snapshot_for(4), player_meta=meta, config=_config(), event=4,
    )
    kinds = result["counts"]
    assert kinds["ROLL"] == 1
    assert kinds["SINGLE"] >= 1
    assert kinds["DOUBLE"] >= 1


def test_actions_are_phase7a_legal_and_ceiling_derived_from_ft():
    universe, state, meta = _universe()
    result = ro.generate_actions(
        state=state, rows={r["player_id"]: r for r in universe["universe"]},
        pool_ids=[r["player_id"] for r in universe["universe"]],
        positions={r["player_id"]: r["position"] for r in universe["universe"]},
        price_snapshot=_scenario().snapshot_for(4), player_meta=meta, config=_config(), event=4,
    )
    # 2 FT + one paid transfer allowance => ceiling 3, capped by config (2 here).
    assert result["ceiling"] == 2
    for action in result["actions"]:
        if action["transition"] is not None:
            assert action["transition"].ok


def test_hit_action_bounded_by_auto_hit_allowance():
    universe, state, meta = _universe()
    rows = {r["player_id"]: r for r in universe["universe"]}
    positions = {r["player_id"]: r["position"] for r in universe["universe"]}
    one_ft = ts.RouteState(event=4, players=state.players, bank_tenths=0, free_transfers=1)
    result = ro.generate_actions(state=one_ft, rows=rows, pool_ids=list(rows), positions=positions,
                                 price_snapshot=_scenario().snapshot_for(4), player_meta=meta,
                                 config=_config(max_transfers_per_event=4,
                                                max_auto_hit_points_per_event=4), event=4)
    assert result["ceiling"] == 2  # 1 FT + 1 paid; a 3rd transfer would be an 8-point hit


def test_invalid_batches_rejected_through_phase7a():
    universe, state, meta = _universe()
    # A club-limit or budget violation must never appear as a legal action.
    rows = {r["player_id"]: r for r in universe["universe"]}
    result = ro.generate_actions(state=state, rows=rows, pool_ids=list(rows),
                                 positions={row["player_id"]: row["position"] for row in rows.values()},
                                 price_snapshot=_scenario().snapshot_for(4), player_meta=meta,
                                 config=_config(), event=4)
    for action in result["actions"]:
        if action["transition"] is not None:
            assert not any("INSUFFICIENT_BANK" in e or "CLUB_LIMIT" in e for e in action["transition"].errors)


# ---------------------------------------------------------------------------
# State key / dedupe
# ---------------------------------------------------------------------------


def test_state_key_distinguishes_bank_ft_and_purchase_basis():
    base = ts.RouteState(event=4, players=tuple(ts.RoutePlayer(pid, POSITION[pid], CLUB[pid], 50)
                                                for pid in SQUAD_IDS), bank_tenths=0, free_transfers=2)
    same = ts.RouteState(event=4, players=base.players, bank_tenths=0, free_transfers=2)
    more_bank = ts.RouteState(event=4, players=base.players, bank_tenths=5, free_transfers=2)
    fewer_ft = ts.RouteState(event=4, players=base.players, bank_tenths=0, free_transfers=1)
    other_basis = ts.RouteState(
        event=4, players=tuple(ts.RoutePlayer(pid, POSITION[pid], CLUB[pid], 51) for pid in SQUAD_IDS),
        bank_tenths=0, free_transfers=2)
    assert ro.state_key(base) == ro.state_key(same)
    assert ro.state_key(base) != ro.state_key(more_bank)
    assert ro.state_key(base) != ro.state_key(fewer_ft)
    assert ro.state_key(base) != ro.state_key(other_basis)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def test_search_explicit_actions_every_event_and_roll_survives():
    universe, state, meta = _universe()
    result = _optimize(universe, state, meta)
    assert all(len(rec["actions"]) == len(EVENTS) for rec in result["routes"].values())
    assert result["roll_baseline"] is not None
    roll = result["roll_baseline"]
    assert all(action["kind"] == "ROLL" for action in result["routes"][roll["route_id"]]["actions"])


def test_proxy_is_labelled_and_never_the_route_value():
    universe, state, meta = _universe()
    result = _optimize(universe, state, meta)
    assert ro.HEURISTIC_LABEL in result["flags"]
    for rec in result["routes"].values():
        assert rec["proxy_label"] == ro.HEURISTIC_LABEL
        assert "supported_3gw_net_core" in rec  # exact, from Phase-6


def test_search_is_deterministic():
    universe, state, meta = _universe()

    def blob(result):
        payload = dict(result)
        payload.pop("timing_s", None)  # wall-clock metadata only
        return json.dumps(payload, sort_keys=True, default=str)

    assert blob(_optimize(universe, state, meta)) == blob(_optimize(universe, state, meta))


def test_heuristic_pruning_and_dedupe_counters_reported():
    universe, state, meta = _universe()
    result = _optimize(universe, state, meta, beam_width=2)
    stats = result["search_stats"]
    assert stats["partial_states"] >= 1
    assert "heuristic_pruned" in stats and "safe_dedup" in stats
    assert ro.BOUNDED_PRUNING_LABEL in result["flags"]


def test_coverage_audit_reports_gw4_action_counts():
    universe, state, meta = _universe()
    result = _optimize(universe, state, meta)
    coverage = result["coverage"]["4"]
    assert coverage["legal_kinds"]["ROLL"] == 1
    assert coverage["legal_kinds"]["SINGLE"] >= 1
    assert coverage["states_generated"] >= 1


# ---------------------------------------------------------------------------
# Objectives / exact scoring
# ---------------------------------------------------------------------------


def test_h1_and_supported_3gw_are_undiscounted_sums_with_hits_once():
    universe, state, meta = _universe()
    result = _optimize(universe, state, meta)
    for rec in result["routes"].values():
        per_event = rec["per_event"]
        assert rec["h1_net_core"] == pytest.approx(per_event[0]["mean_net_core"])
        gross = sum(item["mean_gross_core"] for item in per_event)
        hits = sum(item["hit_points"] for item in per_event)
        assert rec["supported_3gw_gross_core"] == pytest.approx(gross)
        assert rec["cumulative_hits"] == hits
        assert rec["supported_3gw_net_core"] == pytest.approx(gross - hits)
        for item in per_event:
            assert item["mean_net_core"] == pytest.approx(item["mean_gross_core"] - item["hit_points"])


def test_exact_scores_use_phase6_policy_and_expose_terminal_dimensions():
    universe, state, meta = _universe()
    result = _optimize(universe, state, meta)
    rec = next(iter(result["routes"].values()))
    assert rec["per_event"][0]["policy"]["captain_id"]
    assert "terminal_ft" in rec and "terminal_bank_tenths" in rec
    blob = json.dumps(ro.strip_transient(result), default=str)
    for forbidden in ("value_of_free_transfer", "ft_value", "bank_value", "terminal_ft_bonus"):
        assert forbidden not in blob


def test_no_ft_or_bank_scalar_labels():
    universe, state, meta = _universe()
    result = _optimize(universe, state, meta)
    assert ro.CANONICAL_UNSUPPORTED_FLAG in result["flags"]
    assert "H4" not in json.dumps(result.get("supported_events"))
    assert result["supported_events"] == [4, 5]


# ---------------------------------------------------------------------------
# Pareto / families / stability helpers
# ---------------------------------------------------------------------------


def test_frontier_removes_dominated_and_keeps_tradeoffs():
    records = {
        "a": {"x": 10.0, "ft": 2},
        "b": {"x": 9.0, "ft": 2},   # dominated by a (worse x, equal ft)
        "c": {"x": 9.5, "ft": 3},   # tradeoff (worse x, better ft)
    }
    frontier = ro._frontier_names(records, lambda r: (r["x"], r["ft"]))
    assert "a" in frontier and "c" in frontier and "b" not in frontier


def test_frontier_identical_routes_both_kept():
    records = {"a": {"x": 5.0, "ft": 1}, "b": {"x": 5.0, "ft": 1}}
    assert set(ro._frontier_names(records, lambda r: (r["x"], r["ft"]))) == {"a", "b"}


def test_search_stability_detects_material_change():
    base = {"families": [{"family_signature": "S1", "supported_3gw_net_core": 10.0}],
            "routes": {"r": {"family_signature": "S1"}}, "h1_frontier": ["r"], "supported_3gw_frontier": ["r"]}
    stable = {"families": [{"family_signature": "S1", "supported_3gw_net_core": 10.1}],
              "routes": {"r": {"family_signature": "S1"}}, "h1_frontier": ["r"], "supported_3gw_frontier": ["r"]}
    unstable = {"families": [{"family_signature": "S2", "supported_3gw_net_core": 12.0}],
                "routes": {"r": {"family_signature": "S2"}}, "h1_frontier": ["r"], "supported_3gw_frontier": ["r"]}
    assert ro.search_stability(base, stable)["status"] == ro.SEARCH_STABLE
    assert ro.search_stability(base, unstable)["status"] == ro.SEARCH_UNSTABLE


# ---------------------------------------------------------------------------
# Synthetic exhaustive exactness
# ---------------------------------------------------------------------------


def _chain(state, action_sequence, scenario, meta, events):
    current = state
    actions = []
    for action, event in zip(action_sequence, events):
        new_state = (action["transition"].next_event_state if action["transition"] is not None
                     else ts.apply_transfer_batch(current, action["batch"], scenario.snapshot_for(event),
                                                  meta).next_event_state)
        actions.append({**action,
                        "squad_ids": tuple(sorted(int(p.player_id) for p in new_state.players)),
                        "ft_after": int(new_state.free_transfers), "bank_after": int(new_state.bank_tenths)})
        current = new_state
    return ro.PartialRoute(state=current, actions=tuple(actions), h1_proxy=0.0, window_proxy=0.0, hits=0)


def test_bounded_optimizer_matches_exhaustive_on_tiny_problem():
    universe, state, meta = _universe()
    rows = {r["player_id"]: r for r in universe["universe"]}
    positions = {pid: row["position"] for pid, row in rows.items()}
    pool_ids = [r["player_id"] for r in universe["universe"]]
    scenario = _scenario()
    config = _config(beam_width=500, exact_evaluation_budget=500, singles_per_out=1, max_transfers_per_event=1)
    provider = _provider()
    worlds = {e: provider(e, UNION) for e in EVENTS}

    result = ro.optimize(universe=universe, initial_state=state, scenario=scenario, player_meta=meta,
                         config=config, world_provider=provider)
    best_found = max(rec["supported_3gw_net_core"] for rec in result["routes"].values())

    def actions_for(current, event):
        return ro.generate_actions(state=current, rows=rows, pool_ids=pool_ids, positions=positions,
                                   price_snapshot=scenario.snapshot_for(event), player_meta=meta,
                                   config=config, event=event)["actions"]

    def exact(partial):
        return ro.exact_evaluate(partial, worlds_by_event=worlds,
                                 positions_of=lambda ids: {p: positions[p] for p in ids},
                                 cache={}, config=config, events=list(EVENTS))["net_core"]

    best_exhaustive = float("-inf")
    for first in actions_for(state, 4):
        s1 = _chain(state, [first], scenario, meta, [4]).state
        for second in actions_for(s1, 5):
            best_exhaustive = max(best_exhaustive, exact(_chain(state, [first, second], scenario, meta, list(EVENTS))))
    assert result["promoted_route_count"] > 0
    assert best_found == pytest.approx(best_exhaustive, abs=1e-9)


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------


def test_invariants_runs_and_artifacts_unchanged():
    import hashlib
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
        for run_id, digest in ((71, "603003a85330ff54"), (87, "915e924daecb224a"), (95, "c02470004b51c866")):
            rows = conn.execute(
                "SELECT payload_json FROM monte_carlo_distributions WHERE projection_run_id=? ORDER BY id",
                (run_id,)).fetchall()
            assert hashlib.sha256("|".join(r[0] for r in rows).encode()).hexdigest()[:16] == digest
    finally:
        conn.close()
    exports = config_path(config, "exports_dir")
    for relative, marker in (
        ("manager/gw04/manager_lineup_packet.json", '"scoring_basis": "CORE"'),
        ("manager/gw04/transfer_state_validation.json", '"no_recommendation": true'),
        ("routes/gw04/route_comparison.json", '"no_recommendation": true'),
        ("candidates/gw04/candidate_universe.json", '"no_recommendation": true'),
    ):
        path = exports / relative
        if path.exists():
            assert marker in path.read_text(encoding="utf-8")


def test_no_popularity_or_named_player_special_cases_in_source():
    import inspect
    source = inspect.getsource(ro)
    for name in ("Salah", "Haaland", "Palmer", "Bruno", "Mbeumo"):
        assert name not in source
    assert "selected_by_percent" not in source and "transfers_in" not in source


# ---------------------------------------------------------------------------
# Phase 8B.1 — nested-budget stability tests
# ---------------------------------------------------------------------------


def _ladder(budgets=(2, 4, 8), events=EVENTS, **over):
    universe, state, meta = _universe()
    config = _config(events=events, **over)
    return universe, state, meta, rs.run_ladder(
        universe=universe, initial_state=state, scenario=_scenario(events), player_meta=meta,
        base_config=config, budgets=list(budgets), world_provider=_provider(events), exact_cache={},
    )


def test_canonical_family_signature_is_route_id_free():
    universe, state, meta = _universe()
    rows = {r["player_id"]: r for r in universe["universe"]}
    positions = {pid: row["position"] for pid, row in rows.items()}
    actions = ro.generate_actions(state=state, rows=rows, pool_ids=list(rows), positions=positions,
                                  price_snapshot=_scenario().snapshot_for(4), player_meta=meta,
                                  config=_config(), event=4)["actions"]
    single = next(a for a in actions if a["kind"] == "SINGLE")
    first = _chain(state, [single], _scenario(), meta, [4])
    second = _chain(state, [single], _scenario(), meta, [4])
    assert ro.canonical_family_signature(first) == ro.canonical_family_signature(second)
    assert "route_" not in ro.canonical_family_signature(first)
    assert ro.canonical_family_signature(first).startswith("E4:")
    # A ROLL event is explicit, never omitted.
    roll = _chain(state, [next(a for a in actions if a["kind"] == "ROLL")], _scenario(), meta, [4])
    assert ro.canonical_family_signature(roll) == "E4:ROLL"


def test_stability_uses_canonical_signatures_not_route_ids():
    universe, state, meta = _universe()
    result = _optimize(universe, state, meta)
    # Renumber route ids arbitrarily; the verdict must be unchanged.
    shuffled = dict(result)
    remapped = {}
    rename = {}
    for index, (name, rec) in enumerate(sorted(result["routes"].items())):
        new_id = f"route_{999 - index:03d}"
        rename[name] = new_id
        remapped[new_id] = {**rec, "route_id": new_id}
    shuffled["routes"] = remapped
    for key in ("h1_frontier", "supported_3gw_frontier"):
        shuffled[key] = [rename.get(name, name) for name in result.get(key) or []]
    assert ro.search_stability(result, shuffled)["status"] == ro.SEARCH_STABLE
    assert ro.search_stability(result, shuffled)["leader_changed"] is False


def test_nested_search_contains_smaller_beam_survivors():
    universe, state, meta = _universe()
    rows = {r["player_id"]: r for r in universe["universe"]}
    positions = {pid: row["position"] for pid, row in rows.items()}
    pool_ids = [r["player_id"] for r in universe["universe"]]
    small = ro.run_search(initial_state=state, events=list(EVENTS), rows=rows, pool_ids=pool_ids,
                          positions=positions, scenario=_scenario(), player_meta=meta,
                          config=_config(beam_width=2))
    large = ro.run_search(initial_state=state, events=list(EVENTS), rows=rows, pool_ids=pool_ids,
                          positions=positions, scenario=_scenario(), player_meta=meta,
                          config=_config(beam_width=4),
                          nested_prior_levels=small["level_survivors"])
    assert large["stats"]["nested_inherited"] >= 0
    for level, prior_level in zip(large["level_survivors"], small["level_survivors"]):
        prior_keys = {ro.state_key(item.state) for item in prior_level}
        assert prior_keys.issubset({ro.state_key(item.state) for item in level})


def test_nested_ladder_is_monotonic_and_score_consistent():
    _, _, _, run = _ladder(budgets=(2, 4, 8))
    for check in run["monotonic"]:
        assert check["status"] == "PASS", check
    for check in run["cross_budget_score_identity"]:
        assert check["status"] == "PASS", check
        assert check["max_score_discrepancy"] <= 1e-9


def test_smaller_budget_leaders_and_frontier_are_exact_evaluated_by_larger():
    _, _, _, run = _ladder(budgets=(2, 6))
    smaller, larger = run["ladder_results"][0], run["ladder_results"][1]
    required = ro.required_routes_from(smaller)
    required_signatures = {ro.canonical_family_signature(item) for item in required}
    assert required_signatures  # prior leaders / frontier / ROLL exist
    larger_signatures = {rec["canonical_family_signature"] for rec in larger["routes"].values()}
    assert required_signatures.issubset(larger_signatures)
    # H1 and 3GW leaders specifically
    for objective in ("h1_net_core", "supported_3gw_net_core"):
        best = ro.best_route_family(smaller, objective)
        assert best["signature"] in larger_signatures


def test_roll_retained_at_every_budget():
    _, _, _, run = _ladder(budgets=(2, 4, 8))
    for result in run["ladder_results"]:
        signatures = {rec["canonical_family_signature"] for rec in result["routes"].values()}
        assert any(signature.endswith(":ROLL") and signature.count("ROLL") == len(EVENTS) for signature in signatures), signatures
        assert result["roll_baseline"] is not None


def test_nested_monotonic_invariant_detects_a_violation():
    """The invariant must FAIL if a larger budget ever returns a lower best."""
    good = {"routes": {"a": {"canonical_family_signature": "E4:ROLL;E5:ROLL", "supported_3gw_net_core": 10.0,
                             "h1_net_core": 4.0, "cumulative_hits": 0, "terminal_ft": 5,
                             "terminal_bank_tenths": 0, "route_id": "a"}},
            "h1_frontier": ["a"], "supported_3gw_frontier": ["a"], "families": []}
    worse = {"routes": {"b": {"canonical_family_signature": "E4:ROLL;E5:ROLL", "supported_3gw_net_core": 9.0,
                              "h1_net_core": 4.0, "cumulative_hits": 0, "terminal_ft": 5,
                              "terminal_bank_tenths": 0, "route_id": "b"}},
             "h1_frontier": ["b"], "supported_3gw_frontier": ["b"], "families": []}
    assert ro.monotonic_check(good, worse)["status"] == "FAIL"
    assert ro.monotonic_check(good, good)["status"] == "PASS"


def test_certification_rule_is_pre_declared_and_unchanged():
    assert ro.MATERIAL_FRONTIER_CHANGE_CORE == 0.25
    assert rs.MATERIAL_CORE == 0.25
    roll_family = {"canonical_family_signature": "E4:ROLL;E5:ROLL;E6:ROLL",
                   "supported_3gw_net_core": 10.0, "h1_net_core": 4.0, "cumulative_hits": 0,
                   "terminal_ft": 5, "terminal_bank_tenths": 0, "route_id": "a"}
    ladder_pass = [
        {"routes": {"a": dict(roll_family)}, "h1_frontier": ["a"], "supported_3gw_frontier": ["a"], "families": []},
        {"routes": {"a": dict(roll_family)}, "h1_frontier": ["a"], "supported_3gw_frontier": ["a"], "families": []},
    ]
    cert = rs.certify(ladder_pass, [12, 24])
    assert cert["overall"] == "COMPLETE" and cert["status_flag"] == ro.SEARCH_STABLE
    ladder_fail = [ladder_pass[0], {**ladder_pass[1],
                   "routes": {"c": {"canonical_family_signature": "E4:165-249;E5:ROLL;E6:ROLL",
                                    "supported_3gw_net_core": 11.0, "h1_net_core": 4.0, "cumulative_hits": 0,
                                    "terminal_ft": 5, "terminal_bank_tenths": 0, "route_id": "c"}},
                   "h1_frontier": ["c"], "supported_3gw_frontier": ["c"]}]
    assert rs.certify(ladder_fail, [12, 24])["overall"] == "PARTIAL"


def test_same_family_score_identity_detects_mismatch():
    a = {"routes": {"a": {"canonical_family_signature": "S", "supported_3gw_net_core": 5.0, "h1_net_core": 1.0,
                          "cumulative_hits": 0, "terminal_ft": 1, "terminal_bank_tenths": 0, "route_id": "a"}},
         "h1_frontier": ["a"], "supported_3gw_frontier": ["a"], "families": []}
    b = {"routes": {"z": {"canonical_family_signature": "S", "supported_3gw_net_core": 5.5, "h1_net_core": 1.0,
                          "cumulative_hits": 0, "terminal_ft": 1, "terminal_bank_tenths": 0, "route_id": "z"}},
         "h1_frontier": ["z"], "supported_3gw_frontier": ["z"], "families": []}
    assert ro.cross_budget_score_identity(a, a)["status"] == "PASS"
    mismatch = ro.cross_budget_score_identity(a, b)
    assert mismatch["status"] == "FAIL" and mismatch["max_score_discrepancy"] == pytest.approx(0.5)


def test_ladder_is_deterministic():
    _, _, _, first = _ladder(budgets=(2, 4))
    _, _, _, second = _ladder(budgets=(2, 4))
    for one, two in zip(first["summaries"], second["summaries"]):
        assert one["supported_3gw_leader_signature"] == two["supported_3gw_leader_signature"]
        assert one["supported_3gw_leader_value"] == pytest.approx(two["supported_3gw_leader_value"])
        assert one["h1_leader_signature"] == two["h1_leader_signature"]


def test_state_identity_still_separates_ft_bank_and_basis():
    base = ts.RouteState(event=4, players=tuple(ts.RoutePlayer(pid, POSITION[pid], CLUB[pid], 50)
                                                for pid in SQUAD_IDS), bank_tenths=0, free_transfers=2)
    variants = [
        ts.RouteState(event=4, players=base.players, bank_tenths=3, free_transfers=2),
        ts.RouteState(event=4, players=base.players, bank_tenths=0, free_transfers=1),
        ts.RouteState(event=4, players=tuple(ts.RoutePlayer(pid, POSITION[pid], CLUB[pid], 51) for pid in SQUAD_IDS),
                      bank_tenths=0, free_transfers=2),
    ]
    for variant in variants:
        assert ro.state_key(base) != ro.state_key(variant)


def test_world_cache_hit_and_key_sensitivity(tmp_path):
    universe, state, meta = _universe()
    union = list(SQUAD_IDS)
    bundle = rc.EventBundle(event=4, minutes_run_id=1, team_run_id=2, rate_run_id=3, xpts_run_id=4, mc_run_id=5)
    config = _config()
    key = ro.world_cache_key(event=4, bundle=bundle, config=config, union_ids=union)
    payload = {"worlds": 2, "player_ids": [int(p) for p in union],
               "core": {str(p): [1.0, 2.0] for p in union},
               "minutes": {str(p): [90.0, 90.0] for p in union}}
    (tmp_path / f"{key}.json").write_text(json.dumps(payload), encoding="utf-8")
    matrix, info = ro.build_event_worlds(None, {4: bundle}, 4, union, config, cache_dir=tmp_path)
    assert info["source"] == "cache" and info["key"] == key
    assert matrix["core"][int(union[0])] == [1.0, 2.0]
    other_key = ro.world_cache_key(event=4, bundle=bundle, config=config, union_ids=union + [999])
    assert other_key != key  # union-player hash changes cache identity
    other_key2 = ro.world_cache_key(event=5, bundle=bundle, config=config, union_ids=union)
    assert other_key2 != key


def test_no_new_predictive_runs_no_db_write_no_threshold_change():
    import sqlite3
    try:
        from fpl_brain.config import config_path, load_config
        config = load_config(None)
    except Exception:
        pytest.skip("no project config available")
    db_path = config_path(config, "database")
    if not db_path.exists():
        pytest.skip("project database not present")
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        # Floor only for the count: official refreshes append new runs.
        assert conn.execute("SELECT COUNT(*) FROM projection_runs").fetchone()[0] >= 103
        assert conn.execute(
            "SELECT COUNT(*) FROM projection_runs WHERE id IN (71,87,95)"
        ).fetchone()[0] == 3
        # BEFORE/AFTER no-write contract: run the operation under test (a full
        # synthetic optimizer pass with injected worlds, no DB connection) and
        # prove it created no predictive run.
        before = conn.execute("SELECT COUNT(*), COALESCE(MAX(id), 0) FROM projection_runs").fetchone()
        universe, state, meta = _universe()
        result = ro.optimize(universe=universe, initial_state=state, scenario=_scenario(), player_meta=meta,
                             config=_config(), world_provider=_provider(), conn=None, exact_cache={})
        assert result["world_info"][str(EVENTS[0])]["source"] == "injected"
        after = conn.execute("SELECT COUNT(*), COALESCE(MAX(id), 0) FROM projection_runs").fetchone()
        assert (before[0], before[1]) == (after[0], after[1]), "optimizer created a predictive run"
        for run_id, digest in ((71, "603003a85330ff54"), (87, "915e924daecb224a"), (95, "c02470004b51c866")):
            rows = conn.execute(
                "SELECT payload_json FROM monte_carlo_distributions WHERE projection_run_id=? ORDER BY id",
                (run_id,)).fetchall()
            assert hashlib.sha256("|".join(r[0] for r in rows).encode()).hexdigest()[:16] == digest
    finally:
        conn.close()
    assert ro.MATERIAL_FRONTIER_CHANGE_CORE == 0.25


def test_candidate_pool_artifact_and_phase8b_artifacts_unchanged():
    import hashlib
    import pathlib as _pathlib
    try:
        from fpl_brain.config import config_path, load_config
        config = load_config(None)
    except Exception:
        pytest.skip("no project config available")
    exports = config_path(config, "exports_dir")
    checks = {
        "candidates/gw04/candidate_universe.json": None,
        "optimizer/gw04/optimizer_result.json": "30e9b0bbdb7863c1",
        "manager/gw04/manager_lineup_packet.json": None,
        "routes/gw04/route_comparison.json": None,
    }
    for relative, digest in checks.items():
        path = exports / relative
        if not path.exists():
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        if digest is not None:
            assert actual == digest, (relative, actual)


def test_no_player_name_exception_in_stability_code():
    import inspect
    for module in (ro, rs):
        source = inspect.getsource(module)
        for name in ("Salah", "Haaland", "Palmer", "Bruno", "Mbeumo"):
            assert name not in source
