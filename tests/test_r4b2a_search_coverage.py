"""R4B.2a — bounded-search coverage + rescue lens tests.

Synthetic and deterministic.  Nothing here runs a production prediction, a
production Monte Carlo, or a live route optimisation, and nothing writes to the
project database.
"""

from __future__ import annotations

import hashlib

import sqlite3
from pathlib import Path

import pytest

from test_route_optimizer import _np  # noqa: E402

from fpl_brain import candidate_universe as cu
from fpl_brain import route_comparator as rc
from fpl_brain import route_optimizer as ro
from fpl_brain import transfer_state as ts
from test_transfer_state import CLUB, POOL_CLUB, POOL_POSITION, POSITION, SQUAD_IDS

REPO_ROOT = Path(__file__).resolve().parents[1]
EVENTS = (5, 6, 7, 8)
WIDTH = 8


# ---------------------------------------------------------------------------
# Synthetic state builders
# ---------------------------------------------------------------------------
def _state(*, bank=0, ft=2, event=5, purchase=50):
    return ts.RouteState(
        event=event,
        players=tuple(ts.RoutePlayer(pid, POSITION[pid], CLUB[pid], purchase) for pid in SQUAD_IDS),
        bank_tenths=bank, free_transfers=ft,
    )


def _route(*, net=0.0, h1=0.0, hits=0, bank=0, ft=2, tag="R", event_proxies=()):
    """A distinguishable PartialRoute.

    ``state_key`` includes each player's purchase price, so a stable digest of
    ``tag`` is folded into one player's purchase price.  That keeps every route's
    STATE distinct (so the cap's dedup cannot collapse the synthetic fixtures)
    while leaving bank, FT, hits and proxies fully under the caller's control.
    """

    uniq = int(hashlib.sha256(str(tag).encode()).hexdigest()[:6], 16) % 1_000_000
    players = tuple(
        ts.RoutePlayer(pid, POSITION[pid], CLUB[pid], 50 + (uniq if pid == SQUAD_IDS[0] else 0))
        for pid in SQUAD_IDS
    )
    route_state = ts.RouteState(event=5, players=players, bank_tenths=int(bank),
                               free_transfers=int(ft))
    batch = ts.TransferBatch((ts.TransferAction(int(SQUAD_IDS[0]), 41),))
    actions = ({"event": 5, "kind": tag, "batch": batch, "transition": None,
                "hit_points": int(hits), "ft_after": int(ft), "bank_after": int(bank),
                "squad_ids": tuple(sorted(SQUAD_IDS))},)
    return ro.PartialRoute(state=route_state, actions=actions,
                           h1_proxy=float(h1), window_proxy=float(net + hits), hits=int(hits),
                           event_proxies=tuple(event_proxies))


def _config(**over):
    base = dict(events=EVENTS, beam_width=WIDTH, search_draws=6)
    base.update(over)
    return ro.OptimizerConfig(**base)


def _keys(routes):
    return [ro.state_key(r.state) for r in routes]


# ---------------------------------------------------------------------------
# F — TOP_NET_PROXY preserves today's primary top set
# ---------------------------------------------------------------------------
def test_F_top_net_proxy_preserves_the_primary_top_set():
    states = [_route(net=float(i), h1=float(i) / 2, hits=i % 3, bank=i * 10, ft=i % 4, tag=f"S{i}")
              for i in range(60)]
    budgets = ro.retention_budgets(_config())
    expected = sorted(states, key=ro._primary_key)[: budgets["KN"]]
    picks = ro._lens_picks(states, ro.TOP_NET_PROXY_LENS, events=EVENTS,
                           initial_bank_tenths=0, kn=budgets["KN"], kd=budgets["KD"])
    assert _keys(picks) == _keys(expected)
    # And the primary ordering itself is unchanged.
    assert ro._primary_key is not None
    assert budgets["KN"] == max(4 * WIDTH, 24) == 32


# ---------------------------------------------------------------------------
# A — BANK_DELTA rescues the highest-bank route
# ---------------------------------------------------------------------------
def test_A_highest_bank_route_below_primary_prefilter_survives_bank_delta():
    # 60 strong routes (high net) + one weak-but-rich route that the OLD
    # net-proxy prefilter (top 32) would have dropped entirely.
    strong = [_route(net=100.0 + i, h1=50.0, hits=0, bank=0, ft=2, tag=f"STRONG{i}")
              for i in range(60)]
    enabler = _route(net=-500.0, h1=-100.0, hits=0, bank=9_000, ft=1, tag="ENABLER")
    states = strong + [enabler]

    budgets = ro.retention_budgets(_config())
    old_prefilter = sorted(states, key=ro._primary_key)[: budgets["KN"]]
    assert ro.state_key(enabler.state) not in _keys(old_prefilter), "precondition: enabler is outside the old prefilter"

    picks = ro._lens_picks(states, "BANK_DELTA", events=EVENTS, initial_bank_tenths=0,
                           kn=budgets["KN"], kd=budgets["KD"])
    assert ro.state_key(enabler.state) in _keys(picks)

    coverage: dict = {}
    retained = ro._retain(states, _config(), events=EVENTS, initial_bank_tenths=0, coverage=coverage)
    assert ro.state_key(enabler.state) in _keys(retained)
    rescued = {item["signature"] for item in coverage["rescued_routes"]}
    assert str(enabler.signature()) in rescued
    entry = next(i for i in coverage["rescued_routes"] if i["signature"] == str(enabler.signature()))
    assert "BANK_DELTA" in entry["rescued_by"]
    assert entry["primary_rank"] is None


def test_A_bank_delta_is_relative_to_the_initial_bank():
    """BANK_DELTA = terminal - initial, so a rich-but-unchanged route is not 'rich'."""

    spent = _route(net=1.0, bank=0, tag="SPENT")          # 500 -> 0
    kept = _route(net=-1.0, bank=500, tag="KEPT")          # 500 -> 500
    picks = ro._lens_picks([spent, kept], "BANK_DELTA", events=EVENTS,
                           initial_bank_tenths=500, kn=32, kd=4)
    assert ro.state_key(kept.state) == ro.state_key(picks[0].state)


# ---------------------------------------------------------------------------
# B — FT_PRESERVATION rescues the highest-FT route
# ---------------------------------------------------------------------------
def test_B_highest_ft_route_below_primary_prefilter_survives_ft_preservation():
    strong = [_route(net=100.0 + i, hits=0, bank=0, ft=0, tag=f"S{i}") for i in range(60)]
    flexible = _route(net=-900.0, hits=0, bank=0, ft=5, tag="FLEX")
    states = strong + [flexible]
    budgets = ro.retention_budgets(_config())
    assert ro.state_key(flexible.state) not in _keys(
        sorted(states, key=ro._primary_key)[: budgets["KN"]])

    coverage: dict = {}
    retained = ro._retain(states, _config(), events=EVENTS, initial_bank_tenths=0, coverage=coverage)
    assert ro.state_key(flexible.state) in _keys(retained)
    entry = next(i for i in coverage["rescued_routes"] if i["signature"] == str(flexible.signature()))
    assert "FT_PRESERVATION" in entry["rescued_by"]


def test_B_ft_preservation_ordering_is_ft_then_hits_then_bank():
    a = _route(net=0.0, hits=0, bank=100, ft=3, tag="A")
    b = _route(net=0.0, hits=4, bank=900, ft=3, tag="B")   # same FT, more hits -> behind
    c = _route(net=0.0, hits=0, bank=0, ft=2, tag="C")     # fewer FT -> last
    picks = ro._lens_picks([a, b, c], "FT_PRESERVATION", events=EVENTS,
                           initial_bank_tenths=0, kn=32, kd=4)
    assert [ro.state_key(p.state) for p in picks] == [ro.state_key(a.state),
                                                      ro.state_key(b.state),
                                                      ro.state_key(c.state)]


# ---------------------------------------------------------------------------
# C — LOW_HITS rescues the lowest-hit route
# ---------------------------------------------------------------------------
def test_C_lowest_hit_route_below_primary_prefilter_survives_low_hits():
    # Heavy-hit routes dominate net proxy; one zero-hit route is buried below.
    hitters = [_route(net=100.0 + i, hits=4, bank=0, ft=1, tag=f"H{i}") for i in range(60)]
    clean = _route(net=-50.0, hits=0, bank=0, ft=1, tag="CLEAN")
    states = hitters + [clean]
    budgets = ro.retention_budgets(_config())
    assert ro.state_key(clean.state) not in _keys(
        sorted(states, key=ro._primary_key)[: budgets["KN"]])

    coverage: dict = {}
    retained = ro._retain(states, _config(), events=EVENTS, initial_bank_tenths=0, coverage=coverage)
    assert ro.state_key(clean.state) in _keys(retained)
    entry = next(i for i in coverage["rescued_routes"] if i["signature"] == str(clean.signature()))
    assert "LOW_HITS" in entry["rescued_by"]


# ---------------------------------------------------------------------------
# D — PARETO over the FULL set rescues a trade-off route
# ---------------------------------------------------------------------------
def test_D_full_set_pareto_route_below_prefilter_survives():
    # Many routes that dominate on net proxy only.
    greedy = [_route(net=200.0 - i * 0.1, h1=0.0, hits=0, bank=0, ft=0, tag=f"G{i}")
              for i in range(60)]
    # A trade-off route: poor net proxy but the best bank AND best FT.  It is on
    # the full frontier but far outside the old top-32 net-proxy prefilter.
    tradeoff = _route(net=-300.0, h1=0.0, hits=0, bank=5_000, ft=5, tag="TRADEOFF")
    states = greedy + [tradeoff]
    budgets = ro.retention_budgets(_config())
    assert ro.state_key(tradeoff.state) not in _keys(
        sorted(states, key=ro._primary_key)[: budgets["KN"]])

    dims = lambda s: (s.net_proxy(), -s.hits, s.state.bank_tenths, s.state.free_transfers)
    frontier = ro._pareto(states, dims)
    assert ro.state_key(tradeoff.state) in _keys(frontier)

    coverage: dict = {}
    retained = ro._retain(states, _config(), events=EVENTS, initial_bank_tenths=0, coverage=coverage)
    assert ro.state_key(tradeoff.state) in _keys(retained)
    entry = next(i for i in coverage["rescued_routes"] if i["signature"] == str(tradeoff.signature()))
    assert "PARETO" in entry["rescued_by"]


def test_D_pareto_is_computed_over_the_full_set_not_a_prefilter():
    """A route can only be on the frontier if the frontier saw it at all."""

    # Two incomparable routes where the second has the worst net proxy in the set.
    a = _route(net=1000.0, bank=0, ft=0, tag="A")
    b = _route(net=-1000.0, bank=100, ft=5, tag="B")
    filler = [_route(net=float(i), bank=10, ft=1, tag=f"F{i}") for i in range(50)]
    frontier = ro._pareto([a, b] + filler, lambda s: (s.net_proxy(), -s.hits,
                                                      s.state.bank_tenths, s.state.free_transfers))
    keys = _keys(frontier)
    assert ro.state_key(a.state) in keys and ro.state_key(b.state) in keys


# ---------------------------------------------------------------------------
# E — PER_EVENT_MARGINAL rescues a deferred (weak-now, strong-later) route
# ---------------------------------------------------------------------------
def test_E_deferred_route_survives_via_a_later_window_suffix():
    # event_proxies = (e0, e1, e2, e3): [0.1, 0.1, 9.0, 9.0]
    deferred = _route(net=18.2, h1=0.1, hits=0, bank=0, ft=1, tag="DEFERRED",
                      event_proxies=(0.1, 0.1, 9.0, 9.0))
    early = [_route(net=30.0 + i * 0.5, h1=20.0, hits=0, bank=0, ft=1, tag=f"EARLY{i}",
                    event_proxies=(20.0, 2.0, 2.0, 2.0))
             for i in range(60)]
    states = early + [deferred]
    budgets = ro.retention_budgets(_config())
    assert ro.state_key(deferred.state) not in _keys(
        sorted(states, key=ro._primary_key)[: budgets["KN"]]), "precondition: aggregate rank is outside the old prefilter"

    picks = ro._lens_picks(states, ro.PER_EVENT_MARGINAL_LENS, events=EVENTS,
                           initial_bank_tenths=0, kn=budgets["KN"], kd=budgets["KD"])
    assert ro.state_key(deferred.state) in _keys(picks)

    coverage: dict = {}
    retained = ro._retain(states, _config(), events=EVENTS, initial_bank_tenths=0, coverage=coverage)
    assert ro.state_key(deferred.state) in _keys(retained)
    entry = next(i for i in coverage["rescued_routes"] if i["signature"] == str(deferred.signature()))
    assert ro.PER_EVENT_MARGINAL_LENS in entry["rescued_by"]


def test_E_suffix_proxy_is_a_true_remaining_window_suffix():
    route = _route(tag="S", event_proxies=(1.0, 2.0, 3.0, 4.0))
    assert route.suffix_proxy(1) == pytest.approx(9.0)   # e1:e3
    assert route.suffix_proxy(2) == pytest.approx(7.0)   # e2:e3
    assert route.suffix_proxy(3) == pytest.approx(4.0)   # e3
    assert route.suffix_proxy(0) == pytest.approx(10.0)  # whole window


# ---------------------------------------------------------------------------
# G / H — dedup and determinism
# ---------------------------------------------------------------------------
def test_G_duplicate_selection_by_multiple_lenses_counts_one_route_once():
    # One route is simultaneously the best bank, best FT and lowest hits.
    star = _route(net=1.0, h1=1.0, hits=0, bank=9_000, ft=5, tag="STAR")
    filler = [_route(net=float(i), hits=2, bank=0, ft=0, tag=f"F{i}") for i in range(40)]
    states = filler + [star]
    coverage: dict = {}
    retained = ro._retain(states, _config(), events=EVENTS, initial_bank_tenths=0, coverage=coverage)
    keys = _keys(retained)
    assert len(keys) == len(set(keys)), "a state must appear at most once"
    # Multiple lenses claimed it, and its unique_added contribution is counted once
    # across whichever lens reached it first.
    claims = [lens for lens, values in coverage["lenses"].items() if values["unique_added"] > 0]
    assert len(claims) >= 2
    entry = next(i for i in coverage["rescued_routes"] if i["signature"] == str(star.signature()))
    assert len(entry["rescued_by"]) >= 2


def test_H_result_is_deterministic_across_repeated_executions():
    states = [_route(net=float(i % 7), h1=float(i % 5), hits=i % 3, bank=i * 7, ft=i % 4,
                     tag=f"T{i}") for i in range(120)]
    reference = _keys(ro._retain(states, _config(), events=EVENTS, initial_bank_tenths=50))
    for _ in range(3):
        assert _keys(ro._retain(states, _config(), events=EVENTS,
                                initial_bank_tenths=50)) == reference
    # Shuffling the input must not change the outcome either.
    import random
    shuffled = list(states)
    random.Random(11).shuffle(shuffled)
    assert _keys(ro._retain(shuffled, _config(), events=EVENTS,
                            initial_bank_tenths=50)) == reference


# ---------------------------------------------------------------------------
# I / J — bounds and REQUIRED pass-through
# ---------------------------------------------------------------------------
def test_I_non_required_retained_never_exceeds_k_max():
    budget = ro.retention_budgets(_config())["K_MAX"]
    states = [_route(net=float(i), h1=float(i // 2), hits=i % 5, bank=i * 3, ft=i % 6,
                     tag=f"B{i}") for i in range(2000)]
    retained = ro._retain(states, _config(), events=EVENTS, initial_bank_tenths=0)
    assert len(retained) <= budget
    assert budget == 2 * max(4 * WIDTH, 24) == 64


def test_I_large_pareto_frontier_is_still_bounded():
    # Deliberately incomparable: each route is uniquely best on one dimension.
    states = [_route(net=float(i), h1=0.0, hits=i, bank=i * 2, ft=i % 7, tag=f"P{i}")
              for i in range(400)]
    retained = ro._retain(states, _config(), events=EVENTS, initial_bank_tenths=0)
    assert len(retained) <= ro.retention_budgets(_config())["K_MAX"]


def test_J_required_route_survives_even_when_k_max_is_saturated():
    budget = ro.retention_budgets(_config())["K_MAX"]
    # Enough states to saturate the cap on their own.
    states = [_route(net=float(i), hits=0, bank=i, ft=i % 5, tag=f"S{i}") for i in range(budget * 4)]
    required = _route(net=-10_000.0, hits=99, bank=0, ft=0, tag="REQUIRED_ROUTE")
    assert ro.state_key(required.state) not in _keys(states)

    retained = ro._retain(states, _config(), events=EVENTS, initial_bank_tenths=0,
                          required=(required,))
    keys = _keys(retained)
    assert ro.state_key(required.state) in keys
    assert len(keys) <= budget + 1
    # And the non-required set still respects the cap on its own.
    assert len([k for k in keys if k != ro.state_key(required.state)]) <= budget


def test_J_required_is_unioned_after_the_cap_not_competing_for_slots():
    budget = ro.retention_budgets(_config())["K_MAX"]
    states = [_route(net=float(i), hits=0, bank=i, ft=i % 5, tag=f"S{i}") for i in range(budget * 3)]
    required = tuple(_route(net=-1_000.0 - i, hits=1, bank=0, ft=0, tag=f"REQ{i}") for i in range(5))
    coverage: dict = {}
    retained = ro._retain(states, _config(), events=EVENTS, initial_bank_tenths=0,
                          required=required, coverage=coverage)
    keys = _keys(retained)
    for route in required:
        assert ro.state_key(route.state) in keys
    assert len(keys) <= budget + len(required)
    assert coverage["required_count"] == len(required)
    assert coverage["required_added"] == len(required)


# ---------------------------------------------------------------------------
# K / L — no bias, no scoring bonus
# ---------------------------------------------------------------------------
def test_K_ownership_and_popularity_inputs_do_not_affect_retention():
    """No ownership/popularity/club/watchlist field is read by retention."""

    states = [_route(net=float(i), hits=0, bank=i, ft=i % 3, tag=f"S{i}") for i in range(50)]
    baseline = _keys(ro._retain(states, _config(), events=EVENTS, initial_bank_tenths=0))
    for _ in range(2):
        assert _keys(ro._retain(states, _config(), events=EVENTS,
                                initial_bank_tenths=0)) == baseline

    source = (REPO_ROOT / "fpl_brain" / "route_optimizer.py").read_text(encoding="utf-8")
    retain_block = source[source.index("def _retain("):source.index("def _event_proxy_parts(")]
    for forbidden in ("selected_by_percent", "ownership", "ownership_", "popularity", "watchlist",
                      "differential", "reputation", "web_name"):
        assert forbidden not in retain_block, f"retention must not read {forbidden}"


def test_L_no_position_scoring_bonus_is_introduced():
    source = (REPO_ROOT / "fpl_brain" / "route_optimizer.py").read_text(encoding="utf-8")
    retain_block = source[source.index("def _retain("):source.index("def _event_proxy_parts(")]
    assert "POSITIONS" not in retain_block
    # The objective itself is untouched: net_proxy is still window_proxy - hits.
    route = _route(net=10.0, hits=3)
    assert route.net_proxy() == pytest.approx(route.window_proxy - route.hits)


# ---------------------------------------------------------------------------
# M / N — screening stays ahead of retention; the objective is unchanged
# ---------------------------------------------------------------------------
def test_M_all_player_screening_still_occurs_before_retention():
    """Wiring assertion on the production runner: screen -> promote -> optimise."""

    source = (REPO_ROOT / "scripts" / "run_four_gw_decision.py").read_text(encoding="utf-8")
    screen_at = source.index("fg.screen_legal_actions(")
    discovery_at = source.index("cu.discovery_completeness(")
    optimize_at = source.index("ro.optimize(")
    assert screen_at < discovery_at < optimize_at, (
        "numerical screening and discovery completeness must run before the bounded search"
    )
    # And nothing in the retention path filters players by ownership/price tier.
    module = (REPO_ROOT / "fpl_brain" / "route_optimizer.py").read_text(encoding="utf-8")
    assert "search_view_ids" in module  # the view that builds the search pool


def test_M_screening_exhaustiveness_is_still_asserted():
    from test_r4b1_decision_correctness import _universe  # reuse the synthetic universe
    universe, _, _, _ = _universe(EVENTS)
    screen = fg_screen(universe, EVENTS)
    assert screen["enumeration_exhaustive"] is True
    assert screen["coverage"]["every_legal_edge_screened"] is True


def fg_screen(universe, events):
    from fpl_brain import four_gw_decision as fg
    return fg.screen_legal_actions(
        universe_rows=universe["universe"], replacement_edges=universe["replacement_edges"],
        owned_ids=SQUAD_IDS, decision_events_window=events,
    )


def test_N_objective_score_and_ranking_function_are_unchanged():
    # net_proxy, h1 ordering and the primary key must be exactly as before.
    r = _route(net=7.5, h1=2.25, hits=2, bank=100, ft=3)
    assert r.net_proxy() == pytest.approx(7.5)
    assert ro._primary_key(r)[:3] == (-7.5, -2.25, 2)
    assert ro._primary_key(r)[3] == r.signature()  # deterministic trailing tie-break only
    # The four-GW objective constant and the promotion ranking are untouched.
    source = (REPO_ROOT / "fpl_brain" / "route_optimizer.py").read_text(encoding="utf-8")
    assert "MATERIAL_FRONTIER_CHANGE_CORE = 0.25" in source


# ---------------------------------------------------------------------------
# search_coverage artifact
# ---------------------------------------------------------------------------
def test_search_coverage_artifact_shape():
    states = [_route(net=float(i), hits=i % 3, bank=i * 5, ft=i % 4, tag=f"S{i}")
              for i in range(80)]
    coverage_levels = []
    level: dict = {}
    ro._retain(states, _config(), events=EVENTS, initial_bank_tenths=0, coverage=level)
    coverage_levels.append({"event": 5, **level})
    summary = ro._search_coverage_summary(coverage_levels, _config(), {"heuristic_retained": 10})
    assert summary["global_optimality_claimed"] is False
    assert summary["coverage_is_not_quality"] is True
    for key in ("states_generated", "states_retained", "heuristic_pruned",
                "non_required_retained_total", "k_max_per_level",
                "counts_are_aggregated_across_levels",
                "required_count", "k_max", "k_n", "k_d", "lens_order", "cap_policy",
                "levels", "lenses", "rescued_routes"):
        assert key in summary, key
    for lens in ro.RETENTION_LENS_ORDER:
        assert lens in summary["lenses"]
        for field in ("considered", "selected", "unique_added"):
            assert field in summary["lenses"][lens]
    assert ro.REQUIRED_PASS_THROUGH in summary["lenses"]
    # The aggregate must not expose a per-level cap name: it is a SUM.
    assert "non_required_cap" not in summary
    assert summary["k_max_per_level"] == summary["k_max"] == 64
    assert summary["counts_are_aggregated_across_levels"] is True
    # The artifact must not claim rescued routes are better.
    assert not any("better" in str(item.get("rescued_by")) for item in summary["rescued_routes"])


def test_optimize_exposes_search_coverage():
    from test_r4b1_decision_correctness import _provider, _scenario, _universe
    universe, state, meta, _ = _universe(EVENTS)
    result = ro.optimize(universe=universe, initial_state=state, scenario=_scenario(EVENTS),
                         player_meta=meta, config=_config(search_draws=6, exact_evaluation_budget=4,
                                                          policy_selection_worlds=6, singles_per_out=1,
                                                          max_transfers_per_event=1,
                                                          rescue_top_k_per_position=1,
                                                          search_n_per_criterion=2),
                         non_production_worlds=_np(_provider(universe)))
    coverage = result["search_coverage"]
    assert coverage["global_optimality_claimed"] is False
    assert coverage["k_max"] == 64
    assert coverage["states_generated"] >= coverage["states_retained"] >= 0
    assert set(coverage["lenses"]) >= set(ro.RETENTION_LENS_ORDER)


# ---------------------------------------------------------------------------
# O — projection_runs is untouched
# ---------------------------------------------------------------------------
def test_O_projection_runs_remains_212():
    db_path = REPO_ROOT / "fpl.db"
    if not db_path.exists():
        pytest.skip("no local database")
    conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    try:
        count, max_id = conn.execute("SELECT COUNT(*), MAX(id) FROM projection_runs").fetchone()
    finally:
        conn.close()
    assert count == 212
    assert max_id == 212


# ---------------------------------------------------------------------------
# §2 arithmetic regression (pre-R4B.2b checkpoint)
# ---------------------------------------------------------------------------
def test_suffix_proxy_arithmetic_for_the_deferred_route():
    """Direct arithmetic regression for the deferred route's OWN proxies.

    The R4B.2a report originally printed the (1,2,3,4) route's suffix values next
    to the (0.1,0.1,9.0,9.0) route.  These are the CORRECT values for the deferred
    route, and the invariant ``suffix_proxy(i) == sum(p[i:])`` is asserted
    directly so no discount, hidden weighting or event-number-specific adjustment
    can creep in.
    """

    proxies = (0.1, 0.1, 9.0, 9.0)
    route = _route(tag="DEFERRED-ARITH", event_proxies=proxies)
    assert route.suffix_proxy(1) == pytest.approx(18.1)   # e1:e3
    assert route.suffix_proxy(2) == pytest.approx(18.0)   # e2:e3
    assert route.suffix_proxy(3) == pytest.approx(9.0)    # e3
    assert route.suffix_proxy(0) == pytest.approx(18.2)   # whole window

    for start in range(len(proxies)):
        assert route.suffix_proxy(start) == pytest.approx(sum(proxies[start:]))

    # No hidden weighting: consecutive suffixes differ by exactly the dropped term.
    for start in range(len(proxies) - 1):
        assert (route.suffix_proxy(start) - route.suffix_proxy(start + 1)) == pytest.approx(
            proxies[start])

    # No discount factor: a large late value still dominates a tiny early one.
    assert route.suffix_proxy(1) > route.suffix_proxy(0) / 2


# ---------------------------------------------------------------------------
# §3 boundary ties on the primary lens
# ---------------------------------------------------------------------------
def test_primary_lens_boundary_ties_are_deterministic():
    """> KN routes with exact (net,h1,hits) ties straddling the KN boundary.

    Membership must be identical however the input is ordered: the appended
    ``signature()`` tie-break decides, and no objective value changes.
    """

    import random

    budgets = ro.retention_budgets(_config())
    kn = budgets["KN"]

    states = [_route(net=1000.0 + i, h1=500.0, hits=0, bank=i, ft=1, tag=f"HIGH{i}")
              for i in range(kn)]
    tied = [_route(net=1.0, h1=1.0, hits=0, bank=0, ft=1, tag=f"TIE{i:03d}") for i in range(20)]
    states += tied

    def membership(seq):
        picks = ro._lens_picks(seq, ro.TOP_NET_PROXY_LENS, events=EVENTS,
                               initial_bank_tenths=0, kn=kn, kd=budgets["KD"])
        assert len(picks) == kn
        return sorted(ro.state_key(p.state) for p in picks)

    reference = membership(states)
    for seed in (1, 2, 3, 7, 11):
        shuffled = list(states)
        random.Random(seed).shuffle(shuffled)
        assert membership(shuffled) == reference, f"membership changed under shuffle seed {seed}"

    # The football ordering is untouched: -net_proxy, -h1_proxy, hits still lead.
    first = ro._lens_picks(states, ro.TOP_NET_PROXY_LENS, events=EVENTS,
                           initial_bank_tenths=0, kn=kn, kd=budgets["KD"])[0]
    assert first.net_proxy() == pytest.approx(1000.0 + kn - 1)
    # ...and signature is only a TRAILING tie-break.
    a, b = tied[0], tied[1]
    assert ro._primary_key(a)[:3] == ro._primary_key(b)[:3]
    assert ro._primary_key(a) != ro._primary_key(b)
    assert ro._primary_key(a)[3] == a.signature()


# ---------------------------------------------------------------------------
# §4 position-starvation MEASUREMENT ONLY (no position lens, no bonus)
# ---------------------------------------------------------------------------
def test_position_starvation_measurement_cheap_def_gk_route_survives():
    """Measure whether a cheap DEF/GK-shaped route survives on structure alone.

    The lens layer is deliberately position-blind, so a "cheap DEF/GK" route is
    modelled by its SHAPE: modest immediate CORE, a large bank delta and a strong
    remaining-window suffix, competing against attacker-like routes with much
    larger raw CORE that dominate the primary ranking.

    NO position bonus and NO position lens exist; this only MEASURES survival.
    """

    budgets = ro.retention_budgets(_config())
    kn, kd = budgets["KN"], budgets["KD"]

    # Attacker-like: heavy raw CORE now, no bank, no late swing.
    attackers = [_route(net=500.0 + i, h1=400.0, hits=0, bank=0, ft=1, tag=f"ATT{i}",
                        event_proxies=(125.0, 125.0, 125.0, 125.0)) for i in range(60)]
    # Cheap DEF/GK-like: weak now, cheap, creates bank, strong in the last two events.
    cheap_def = _route(net=40.0, h1=2.0, hits=0, bank=2_500, ft=1, tag="CHEAP_DEF_GK",
                       event_proxies=(2.0, 2.0, 18.0, 18.0))
    states = attackers + [cheap_def]

    key = ro.state_key(cheap_def.state)
    assert key not in _keys(sorted(states, key=ro._primary_key)[:kn]), (
        "measurement precondition: the cheap DEF/GK route must be outside the primary top set"
    )

    lens_credit = {
        lens: key in _keys(ro._lens_picks(states, lens, events=EVENTS, initial_bank_tenths=0,
                                          kn=kn, kd=kd))
        for lens in ro.RETENTION_LENS_ORDER
    }
    structural = ["BANK_DELTA", "PARETO", "PER_EVENT_MARGINAL"]
    survived_by = [lens for lens in structural if lens_credit[lens]]

    coverage: dict = {}
    retained = ro._retain(states, _config(), events=EVENTS, initial_bank_tenths=0, coverage=coverage)
    assert key in _keys(retained), (
        f"POSITION STARVATION: the cheap DEF/GK route was dropped. lens credit={lens_credit}"
    )
    assert survived_by, f"survived but no structural lens claimed it: {lens_credit}"

    entry = next(i for i in coverage["rescued_routes"]
                 if i["signature"] == str(cheap_def.signature()))
    assert set(entry["rescued_by"]) & set(structural), entry["rescued_by"]
    assert entry["primary_rank"] is None

    # The measurement must not have introduced any positional weighting.
    source = (REPO_ROOT / "fpl_brain" / "route_optimizer.py").read_text(encoding="utf-8")
    retain_block = source[source.index("def _retain("):source.index("def _event_proxy_parts(")]
    assert "POSITIONS" not in retain_block
    assert "position" not in retain_block.lower()
