"""Phase 8A — candidate universe tests (pure, synthetic-first)."""

from __future__ import annotations

import json

import pytest

from fpl_brain import candidate_universe as cu
from fpl_brain import transfer_state as ts
from fpl_brain.season_rules import window_flag

EVENTS = (4, 5, 6)


def _pool(spec):
    """spec: {player_id: (position, club_id)} → canonical pool mapping."""
    return {
        int(pid): {"player_id": int(pid), "position": position, "club_id": int(club),
                   "web_name": f"W{pid}", "full_name": f"Player {pid}"}
        for pid, (position, club) in spec.items()
    }


def _fixtures(mapping):
    """mapping: {event: {club_id: [fixture_ids]}} → {(event, club): [ids]}"""
    out = {}
    for event, clubs in mapping.items():
        for club, fids in clubs.items():
            out[(int(event), int(club))] = [int(f) for f in fids]
    return out


def _rows(entries, per_player=1):
    """entries: {player_id: {(event, fixture_id): {...}}} → {event: {(pid, fid): payload}}"""
    out: dict[int, dict[tuple[int, int], dict]] = {event: {} for event in EVENTS}
    for pid, by_key in entries.items():
        for (event, fid), payload in by_key.items():
            out.setdefault(int(event), {})[(int(pid), int(fid))] = dict(payload)
    return out


def _payload(core=5.0, minutes=90.0, p_start=1.0, p60=1.0, **over):
    base = {
        "core_xpts": core, "expected_minutes": minutes, "p_start": p_start, "p_60_plus": p60,
        "p_appearance": 1.0, "bonus_xpts": 99.0, "soft_xpts": 99.0, "total_xpts": float(core) + 99.0,
    }
    base.update(over)
    return base


def _minutes(availability=1.0, **over):
    base = {"joint_availability": availability, "p_available": availability}
    base.update(over)
    return base


def _snapshot(prices):
    return ts.PriceSnapshot(event=4, prices={int(pid): int(price) for pid, price in prices.items()})


def _build(*, spec, fixtures, xpts, minutes, prices, owned=(), config=None):
    return cu.build_universe(
        pool=_pool(spec), events_fixtures=_fixtures(fixtures), xpts_rows_by_event=_rows(xpts),
        minutes_rows_by_event=_rows(minutes), events=list(EVENTS), owned_ids=owned,
        price_snapshot=_snapshot(prices), config=config, planning_cutoff="2026-09-11T10:16:51Z",
    )


# ---------------------------------------------------------------------------
# Source / identity
# ---------------------------------------------------------------------------


def test_player_identity_is_canonical_id_not_name():
    universe = _build(
        spec={1: ("MID", 10), 2: ("MID", 11)},
        fixtures={e: {10: [100], 11: [101]} for e in EVENTS},
        xpts={1: {(e, 100): _payload(5.0) for e in EVENTS}, 2: {(e, 101): _payload(9.0) for e in EVENTS}},
        minutes={1: {(e, 100): _minutes() for e in EVENTS}, 2: {(e, 101): _minutes() for e in EVENTS}},
        prices={1: 50, 2: 50},
    )
    ids = [row["player_id"] for row in universe["universe"]]
    assert ids == [1, 2]
    # Names are display metadata only and never used for identity.
    renamed = _build(
        spec={1: ("MID", 10), 2: ("MID", 11)},
        fixtures={e: {10: [100], 11: [101]} for e in EVENTS},
        xpts={1: {(e, 100): _payload(5.0) for e in EVENTS}, 2: {(e, 101): _payload(9.0) for e in EVENTS}},
        minutes={1: {(e, 100): _minutes() for e in EVENTS}, 2: {(e, 101): _minutes() for e in EVENTS}},
        prices={1: 50, 2: 50},
    )
    for row in renamed["universe"]:
        row["web_name"] = "SOMETHING ELSE"
    assert renaming_order_is_stable(universe, renamed)


def renaming_order_is_stable(a, b):
    return [r["player_id"] for r in a["universe"]] == [r["player_id"] for r in b["universe"]]


def test_owned_flags_from_caller_not_reconstructed():
    universe = _build(
        spec={1: ("MID", 10), 2: ("MID", 11)},
        fixtures={e: {10: [100], 11: [101]} for e in EVENTS},
        xpts={1: {(e, 100): _payload() for e in EVENTS}, 2: {(e, 101): _payload() for e in EVENTS}},
        minutes={1: {(e, 100): _minutes() for e in EVENTS}, 2: {(e, 101): _minutes() for e in EVENTS}},
        prices={1: 50, 2: 50}, owned=[2],
    )
    by_id = {row["player_id"]: row for row in universe["universe"]}
    assert by_id[2]["owned"] is True and by_id[1]["owned"] is False


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------


def _single_player_universe(**over):
    payloads = over.get("payloads") or {e: _payload() for e in EVENTS}
    return _build(
        spec={1: ("MID", 10)},
        fixtures=over.get("fixtures") or {e: {10: [100]} for e in EVENTS},
        xpts=over.get("xpts") or {1: {(e, 100): payloads[e] for e in EVENTS}},
        minutes=over.get("minutes") or {1: {(e, 100): _minutes() for e in EVENTS}},
        prices=over.get("prices") or {1: 50},
    )


def test_gw4_gw5_gw6_expected_core_resolved():
    universe = _single_player_universe(payloads={4: _payload(7.0), 5: _payload(3.0), 6: _payload(1.0)})
    features = {f.event: f for f in universe["universe"][0]["events"]}
    assert features[4].expected_core == 7.0
    assert features[5].expected_core == 3.0
    assert features[6].expected_core == 1.0


def test_three_gw_sum_exact_and_no_discount():
    universe = _single_player_universe(payloads={4: _payload(7.0), 5: _payload(3.0), 6: _payload(1.0)})
    row = universe["universe"][0]
    assert row["supported_3gw_expected_core"] == pytest.approx(11.0)
    assert row["supported_3gw_expected_minutes"] == pytest.approx(270.0)
    assert row["mean_expected_core_per_event"] == pytest.approx(11.0 / 3)


def test_dgw_aggregates_across_fixtures():
    universe = _single_player_universe(
        fixtures={e: {10: [100, 101]} for e in EVENTS},
        xpts={1: {**{(e, 100): _payload(4.0) for e in EVENTS}, **{(e, 101): _payload(6.0) for e in EVENTS}}},
        minutes={1: {**{(e, 100): _minutes() for e in EVENTS}, **{(e, 101): _minutes() for e in EVENTS}}},
    )
    feature = universe["universe"][0]["events"][0]
    assert feature.fixture_count == 2
    assert feature.expected_core == pytest.approx(10.0)
    assert feature.expected_minutes == pytest.approx(180.0)


def test_bgw_is_explicit_no_fixture_zero():
    universe = _single_player_universe(fixtures={4: {10: [100]}, 5: {99: [200]}, 6: {10: [100]}})
    features = {f.event: f for f in universe["universe"][0]["events"]}
    assert features[5].status == cu.NO_FIXTURE
    assert features[5].expected_core == 0.0 and features[5].fixture_count == 0
    assert universe["completeness_audit"]["by_event"]["5"][cu.NO_FIXTURE] == 1


def test_missing_projection_is_not_coerced_to_zero():
    universe = _single_player_universe(
        fixtures={e: {10: [100]} for e in EVENTS},
        xpts={1: {(4, 100): _payload(5.0), (5, 100): _payload(5.0)}},  # event 6 row absent
        minutes={1: {(e, 100): _minutes() for e in EVENTS}},
    )
    features = {f.event: f for f in universe["universe"][0]["events"]}
    assert features[6].status == cu.MISSING_PROJECTION
    audit = universe["completeness_audit"]
    assert audit["by_event"]["6"][cu.MISSING_PROJECTION] == 1
    assert {"player_id": 1, "event": 6} in audit["missing_projection_players"]


def test_predicted_partial_flagged_when_one_dgw_fixture_missing():
    universe = _single_player_universe(
        fixtures={e: {10: [100, 101]} for e in EVENTS},
        xpts={1: {(e, 100): _payload(4.0) for e in EVENTS}},  # fixture 101 missing
        minutes={1: {**{(e, 100): _minutes() for e in EVENTS}, **{(e, 101): _minutes() for e in EVENTS}}},
    )
    feature = universe["universe"][0]["events"][0]
    assert feature.status == cu.PREDICTED and feature.partial is True
    assert universe["completeness_audit"]["by_event"]["4"]["PREDICTED_PARTIAL"] == 1


# ---------------------------------------------------------------------------
# Scoring basis
# ---------------------------------------------------------------------------


def test_core_used_and_deterministic_bonus_excluded():
    universe = _single_player_universe(
        payloads={e: _payload(core=6.0, bonus_xpts=50.0, soft_xpts=50.0, total_xpts=56.0) for e in EVENTS}
    )
    row = universe["universe"][0]
    assert universe["score_basis"] == "CORE"
    assert row["supported_3gw_expected_core"] == pytest.approx(18.0)  # 3 × 6, bonus never added


def test_total_proxy_not_used_for_search_value():
    universe = _single_player_universe(
        payloads={e: _payload(core=6.0, bonus_xpts=1000.0, total_xpts=1006.0) for e in EVENTS}
    )
    row = universe["universe"][0]
    assert row["descriptive_value"] == pytest.approx(18.0 / 5.0)  # core/£m, not total


# ---------------------------------------------------------------------------
# Price
# ---------------------------------------------------------------------------


def test_prices_integer_tenths_and_no_sell_price_for_unowned():
    universe = _single_player_universe(prices={1: 47})
    row = universe["universe"][0]
    assert row["current_market_price_tenths"] == 47 and isinstance(row["current_market_price_tenths"], int)
    assert "selling_price_tenths" not in row and "effective_selling_price" not in row


def test_missing_price_excludes_with_reason():
    universe = _build(
        spec={1: ("MID", 10)},
        fixtures={e: {10: [100]} for e in EVENTS},
        xpts={1: {(e, 100): _payload() for e in EVENTS}},
        minutes={1: {(e, 100): _minutes() for e in EVENTS}},
        prices={},
    )
    assert universe["universe_count"] == 0
    assert universe["excluded"] == [{"player_id": 1, "reason": "MISSING_CURRENT_PRICE"}]


# ---------------------------------------------------------------------------
# Universe retention / popularity
# ---------------------------------------------------------------------------


def test_low_minute_and_hard_out_players_retained():
    universe = _build(
        spec={1: ("MID", 10), 2: ("MID", 11), 3: ("DEF", 12)},
        fixtures={e: {10: [100], 11: [101], 12: [102]} for e in EVENTS},
        xpts={1: {(e, 100): _payload(core=0.01, minutes=2.0, p_start=0.01) for e in EVENTS},
              2: {(e, 101): _payload(core=0.0, minutes=0.0, p_start=0.0) for e in EVENTS},
              3: {(e, 102): _payload(core=0.0, minutes=0.0, p_start=0.0) for e in EVENTS}},
        minutes={1: {(e, 100): _minutes(0.01) for e in EVENTS},
                 2: {(e, 101): _minutes(0.0) for e in EVENTS},
                 3: {(e, 102): _minutes(0.0) for e in EVENTS}},
        prices={1: 45, 2: 40, 3: 40},
    )
    assert universe["universe_count"] == 3


def test_invalid_position_excluded_with_reason():
    pool = cu.load_pool  # untouched; build with a bad position directly
    universe = cu.build_universe(
        pool={1: {"player_id": 1, "position": None, "club_id": 10, "web_name": "x", "full_name": "x"}},
        events_fixtures=_fixtures({e: {10: [100]} for e in EVENTS}), xpts_rows_by_event={e: {} for e in EVENTS},
        minutes_rows_by_event={e: {} for e in EVENTS}, events=list(EVENTS), owned_ids=[],
        price_snapshot=_snapshot({1: 50}),
    )
    assert universe["universe"] == []
    assert universe["excluded"] == [{"player_id": 1, "reason": "INVALID_POSITION"}]


def test_popularity_does_not_alter_inclusion_or_metrics():
    base = {1: {(e, 100): _payload(5.0) for e in EVENTS}}
    loud = {1: {(e, 100): _payload(5.0, selected_by_percent=99.9, transfers_in=10_000_000) for e in EVENTS}}
    a = _build(spec={1: ("MID", 10)}, fixtures={e: {10: [100]} for e in EVENTS}, xpts=base,
               minutes={1: {(e, 100): _minutes() for e in EVENTS}}, prices={1: 50})
    b = _build(spec={1: ("MID", 10)}, fixtures={e: {10: [100]} for e in EVENTS}, xpts=loud,
               minutes={1: {(e, 100): _minutes() for e in EVENTS}}, prices={1: 50})
    assert a["universe"][0]["supported_3gw_expected_core"] == b["universe"][0]["supported_3gw_expected_core"]
    assert a["universe"][0]["inclusion_reasons"] == b["universe"][0]["inclusion_reasons"]


def test_deterministic_ordering_position_then_player_id():
    universe = _build(
        spec={5: ("FWD", 10), 1: ("GKP", 11), 3: ("MID", 12), 2: ("DEF", 13)},
        fixtures={e: {10: [100], 11: [101], 12: [102], 13: [103]} for e in EVENTS},
        xpts={pid: {(e, fid): _payload() for e in EVENTS}
              for pid, fid in ((5, 100), (1, 101), (3, 102), (2, 103))},
        minutes={pid: {(e, fid): _minutes() for e in EVENTS}
                 for pid, fid in ((5, 100), (1, 101), (3, 102), (2, 103))},
        prices={5: 50, 1: 50, 3: 50, 2: 50},
    )
    assert [row["player_id"] for row in universe["universe"]] == [1, 2, 3, 5]


# ---------------------------------------------------------------------------
# Search view
# ---------------------------------------------------------------------------


def _five_player_universe():
    spec = {1: ("MID", 10), 2: ("MID", 11), 3: ("MID", 12), 4: ("MID", 13), 5: ("MID", 14)}
    cores = {1: 9.0, 2: 7.0, 3: 5.0, 4: 3.0, 5: 1.0}
    fixtures = {e: {10 + i: [100 + i] for i in range(5)} for e in EVENTS}
    xpts = {pid: {(e, 100 + pid - 1): _payload(cores[pid], minutes=float(90 - pid)) for e in EVENTS}
            for pid in spec}
    minutes = {pid: {(e, 100 + pid - 1): _minutes() for e in EVENTS} for pid in spec}
    prices = {1: 100, 2: 80, 3: 60, 4: 40, 5: 20}
    return _build(spec=spec, fixtures=fixtures, xpts=xpts, minutes=minutes, prices=prices,
                  config=cu.CandidateConfig(top_n_per_criterion=2))


def test_search_view_criteria_inclusion_and_reasons():
    universe = _five_player_universe()
    reasons = {row["player_id"]: row["inclusion_reasons"] for row in universe["search_view"]}
    assert "TOP_FIRST_EVENT_CORE_MID" in reasons[1] and "TOP_WINDOW_CORE_MID" in reasons[1]
    assert "TOP_MINUTES_MID" in reasons[1]
    assert "TOP_VALUE_MID" in reasons[1]   # highest window-sum core per £m
    assert "TOP_CHEAP_MID" in reasons[5]   # cheapest two


def test_search_view_does_not_shrink_universe():
    universe = _five_player_universe()
    assert universe["universe_count"] == 5
    assert universe["search_view_count"] <= universe["universe_count"]
    assert universe["search_view_count"] >= 1


def test_search_view_rankings_deterministic_tiebreak():
    universe = _build(
        spec={1: ("MID", 10), 2: ("MID", 11)},
        fixtures={e: {10: [100], 11: [101]} for e in EVENTS},
        xpts={1: {(e, 100): _payload(5.0) for e in EVENTS}, 2: {(e, 101): _payload(5.0) for e in EVENTS}},
        minutes={1: {(e, 100): _minutes() for e in EVENTS}, 2: {(e, 101): _minutes() for e in EVENTS}},
        prices={1: 50, 2: 50}, config=cu.CandidateConfig(top_n_per_criterion=1),
    )
    ranked = [row["player_id"] for row in universe["search_view"] if "TOP_FIRST_EVENT_CORE_MID" in row["inclusion_reasons"]]
    assert ranked == [1]  # equal score/price → lower player_id wins


# ---------------------------------------------------------------------------
# Dominance (safe marking only)
# ---------------------------------------------------------------------------


def test_safe_dominance_marked_but_player_retained():
    universe = _build(
        spec={1: ("MID", 10), 2: ("MID", 10)},
        fixtures={e: {10: [100]} for e in EVENTS},
        xpts={1: {(e, 100): _payload(9.0, minutes=90.0) for e in EVENTS},
              2: {(e, 100): _payload(4.0, minutes=60.0) for e in EVENTS}},
        minutes={1: {(e, 100): _minutes() for e in EVENTS}, 2: {(e, 100): _minutes() for e in EVENTS}},
        prices={1: 50, 2: 60},
    )
    by_id = {row["player_id"]: row for row in universe["universe"]}
    assert by_id[2]["search_view_dominated_by"] == 1
    assert universe["universe_count"] == 2  # never erased


def test_dominance_not_applied_across_clubs():
    universe = _build(
        spec={1: ("MID", 10), 2: ("MID", 11)},
        fixtures={e: {10: [100], 11: [101]} for e in EVENTS},
        xpts={1: {(e, 100): _payload(9.0) for e in EVENTS}, 2: {(e, 101): _payload(4.0) for e in EVENTS}},
        minutes={1: {(e, 100): _minutes() for e in EVENTS}, 2: {(e, 101): _minutes() for e in EVENTS}},
        prices={1: 50, 2: 60},
    )
    by_id = {row["player_id"]: row for row in universe["universe"]}
    assert by_id[2]["search_view_dominated_by"] is None


# ---------------------------------------------------------------------------
# Replacement edges (Phase-7A)
# ---------------------------------------------------------------------------


def _edge_universe(prices=None, owned=(1,)):
    spec = {1: ("MID", 10), 2: ("MID", 11), 3: ("DEF", 12)}
    fixtures = {e: {10: [100], 11: [101], 12: [102]} for e in EVENTS}
    xpts = {pid: {(e, fid): _payload() for e in EVENTS} for pid, fid in ((1, 100), (2, 101), (3, 102))}
    minutes = {pid: {(e, fid): _minutes() for e in EVENTS} for pid, fid in ((1, 100), (2, 101), (3, 102))}
    return _build(spec=spec, fixtures=fixtures, xpts=xpts, minutes=minutes,
                  prices=prices or {1: 50, 2: 50, 3: 50}, owned=owned)


def _state_for(universe, owned=(1,)):
    players = tuple(ts.RoutePlayer(row["player_id"], row["position"], row["club_id"],
                                   row["current_market_price_tenths"])
                    for row in universe["universe"] if row["owned"])
    return ts.RouteState(event=4, players=players, bank_tenths=0, free_transfers=1)


def _meta_for(universe):
    return {row["player_id"]: ts.PlayerMeta(row["player_id"], row["position"], row["club_id"])
            for row in universe["universe"]}


def test_replacement_edges_same_position_only_and_not_owned():
    universe = _edge_universe(owned=(1,))
    edges = cu.build_replacement_edges(
        universe_rows=universe["universe"], owned_ids=[1], state=_state_for(universe),
        price_snapshot=_snapshot({1: 50, 2: 50, 3: 50}),
        player_meta=_meta_for(universe),
    )
    assert edges and all(edge["in_player_id"] != 1 for edge in edges)
    assert all(edge["position"] == "MID" for edge in edges)  # DEF target 3 excluded


def test_replacement_edge_legality_matches_phase7a():
    universe = _edge_universe(owned=(1,))
    snapshot = _snapshot({1: 50, 2: 50, 3: 50})
    state = _state_for(universe)
    meta = _meta_for(universe)
    edges = cu.build_replacement_edges(universe_rows=universe["universe"], owned_ids=[1],
                                       state=state, price_snapshot=snapshot, player_meta=meta)
    direct = ts.apply_transfer_batch(state, ts.TransferBatch((ts.TransferAction(1, 2),)), snapshot, meta)
    edge = next(e for e in edges if e["in_player_id"] == 2)
    assert edge["currently_legal_single_transfer"] is direct.ok
    assert edge["bank_after_tenths"] == direct.bank_after_tenths


def test_illegal_single_edge_keeps_candidate_and_expensiveness_not_filtered():
    universe = _edge_universe(prices={1: 50, 2: 200, 3: 50}, owned=(1,))
    snapshot = _snapshot({1: 50, 2: 200, 3: 50})
    edges = cu.build_replacement_edges(universe_rows=universe["universe"], owned_ids=[1],
                                       state=_state_for(universe), price_snapshot=snapshot,
                                       player_meta=_meta_for(universe))
    expensive = next(e for e in edges if e["in_player_id"] == 2)
    assert expensive["currently_legal_single_transfer"] is False
    assert "INSUFFICIENT_BANK" in (expensive["failure_reason"] or "")
    # The pair-enabled expensive candidate stays in the full universe.
    assert 2 in {row["player_id"] for row in universe["universe"]}


# ---------------------------------------------------------------------------
# Artifact / invariants
# ---------------------------------------------------------------------------


def test_universe_build_is_deterministic_and_jsonable():
    first = _five_player_universe()
    second = _five_player_universe()
    assert json.dumps(cu.jsonable(first), sort_keys=True) == json.dumps(cu.jsonable(second), sort_keys=True)
    assert cu.jsonable(first)["universe"][0]["events"][0]["event"] == 4


def test_no_recommendation_marker_and_flags():
    universe = _five_player_universe()
    assert universe["no_recommendation"] is True
    # The window flag is derived from the REAL decision events, not hard-coded.
    assert window_flag(universe["supported_events"]) in universe["flags"]
    assert cu.CANONICAL_UNSUPPORTED_FLAG in universe["flags"]


def test_real_project_artifacts_and_runs_unchanged():
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
        for run_id, digest, table in (
            (71, "603003a85330ff54", "monte_carlo_distributions"),
            (87, "915e924daecb224a", "monte_carlo_distributions"),
            (95, "c02470004b51c866", "monte_carlo_distributions"),
        ):
            rows = conn.execute(
                f"SELECT payload_json FROM {table} WHERE projection_run_id=? ORDER BY id", (run_id,)
            ).fetchall()
            assert hashlib.sha256("|".join(r[0] for r in rows).encode()).hexdigest()[:16] == digest
    finally:
        conn.close()
    exports = config_path(config, "exports_dir")
    for relative, marker in (
        ("manager/gw04/manager_lineup_packet.json", '"scoring_basis": "CORE"'),
        ("manager/gw04/transfer_state_validation.json", '"no_recommendation": true'),
        ("routes/gw04/route_comparison.json", '"no_recommendation": true'),
    ):
        path = exports / relative
        if path.exists():
            assert marker in path.read_text(encoding="utf-8")
