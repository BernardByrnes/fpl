"""R4B.2b STABILITY REPAIR — escalation breadth re-pin + final-ranking propagation.

Two independent things are proven here.

**§1 re-pin (the described defect).** The claim under test is that the single
escalation "seeds the wider search only from Stage-1 nested survivors" and
"cannot detect routes whose earlier prefixes were pruned by beam 8".  These tests
run the REAL production primitives (`run_search`, `_retain`, `generate_actions`)
and show that:

* ``run_search`` always begins from the SAME INITIAL ROUTE STATE and the nested
  prior is merged in AFTER each level's retention, so it can only ADD;
* therefore the escalated tree is a strict SUPERSET of the narrow tree;
* and a level-0 prefix that the narrow retention PRUNED is retained by the
  escalation, whose successors are reachable from the wider frontier and NOT from
  the narrow frontier (the control: expanding only the narrow frontier — the
  shape the claim assumes — would miss them).

**§4–§6 final-ranking propagation.** After an escalation the FINAL ranking, the
canonical paired record and the role-relevant set must follow the widened result,
not the pre-escalation Stage-2 finalists.

Synthetic and deterministic; no production prediction, no live DB.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fpl_brain import candidate_universe as cu
from fpl_brain import finalist_refinement as fr
from fpl_brain import route_optimizer as ro
from fpl_brain import route_stability as rs
from fpl_brain import transfer_state as ts
from test_transfer_state import CLUB, POSITION, SQUAD_IDS

REPO_ROOT = Path(__file__).resolve().parents[1]
EVENTS = (5, 6)
NARROW_BEAM = 6      # KN=24, KD=4, K_MAX=48
WIDE_BEAM = 12       # KN=48, KD=6, K_MAX=96

# A pool big enough that the narrow retention cap BINDS, with distinct clubs so
# the 3-per-club rule never interferes.
POOL_POSITION: dict[int, str] = {}
POOL_CLUB: dict[int, int] = {}
for offset, (position, count) in enumerate((("GKP", 2), ("DEF", 8), ("MID", 8), ("FWD", 6))):
    for index in range(count):
        pid = 200 + offset * 20 + index
        POOL_POSITION[pid] = position
        POOL_CLUB[pid] = 400 + offset * 20 + index
POOL_IDS = sorted(POOL_POSITION)
UNION = sorted(set(SQUAD_IDS) | set(POOL_IDS))
# Distinct per-player core so the level-0 net-proxy ordering is strictly varied.
CORE = {pid: 1.0 + (index % 17) * 0.25 for index, pid in enumerate(UNION)}
OWNED_PRICE, POOL_PRICE = 100, 12


def _payload(core: float) -> dict:
    return {"core_xpts": core, "expected_minutes": 90.0, "p_start": 1.0, "p_60_plus": 1.0,
            "p_appearance": 1.0, "bonus_xpts": 0.0, "total_xpts": core, "risk_flags": []}


def _position_of(pid: int) -> str:
    return POOL_POSITION[pid] if pid in POOL_POSITION else POSITION[pid]


def _club_of(pid: int) -> int:
    return POOL_CLUB[pid] if pid in POOL_CLUB else CLUB[pid]


def _universe(events=EVENTS):
    pool = {
        pid: {"player_id": pid, "position": _position_of(pid), "club_id": _club_of(pid),
              "web_name": f"W{pid}", "full_name": f"P{pid}"}
        for pid in UNION
    }
    clubs: dict[int, list[int]] = {}
    for pid, meta in pool.items():
        clubs.setdefault(meta["club_id"], []).append(pid)
    fixtures = {(event, club): [1000 + club] for event in events for club in clubs}
    xpts = {event: {(pid, 1000 + pool[pid]["club_id"]): _payload(CORE[pid]) for pid in UNION}
            for event in events}
    minutes = {event: {(pid, 1000 + pool[pid]["club_id"]): {"joint_availability": 1.0} for pid in UNION}
               for event in events}
    prices = {pid: (POOL_PRICE if pid in POOL_POSITION else OWNED_PRICE) for pid in UNION}
    snapshot = ts.PriceSnapshot(event=int(events[0]), prices=prices)
    universe = cu.build_universe(
        pool=pool, events_fixtures=fixtures, xpts_rows_by_event=xpts, minutes_rows_by_event=minutes,
        events=list(events), owned_ids=list(SQUAD_IDS), price_snapshot=snapshot,
        config=cu.CandidateConfig(top_n_per_criterion=3), planning_cutoff="2026-09-13T00:00:00Z",
    )
    state = ts.RouteState(
        event=int(events[0]),
        players=tuple(ts.RoutePlayer(pid, POSITION[pid], CLUB[pid], 50) for pid in SQUAD_IDS),
        bank_tenths=0, free_transfers=2,
    )
    meta = {pid: ts.PlayerMeta(pid, _position_of(pid), _club_of(pid)) for pid in UNION}
    universe["replacement_edges"] = cu.build_replacement_edges(
        universe_rows=universe["universe"], owned_ids=list(SQUAD_IDS), state=state,
        price_snapshot=snapshot, player_meta=meta,
    )
    return universe, state, meta, prices


def _scenario(prices, events=EVENTS):
    from fpl_brain import route_comparator as rc

    snapshots = {int(event): ts.PriceSnapshot(event=int(event), prices=dict(prices)) for event in events}
    return rc.PriceScenario(scenario_id="REPAIR_TEST_FLAT", event_snapshots=snapshots)


def _config(beam, **over):
    base = dict(events=EVENTS, search_draws=6, seed=20260913, beam_width=beam,
                exact_evaluation_budget=20, policy_selection_worlds=6,
                singles_per_out=6, max_transfers_per_event=1,
                max_auto_hit_points_per_event=0, rescue_top_k_per_position=2,
                search_n_per_criterion=3)
    base.update(over)
    return ro.OptimizerConfig(**base)


def _provider(worlds=6):
    def provider(event, union_ids):
        return {"worlds": worlds, "player_ids": list(union_ids),
                "core": {pid: [CORE.get(pid, 0.0)] * worlds for pid in union_ids},
                "minutes": {pid: [90.0] * worlds for pid in union_ids}}
    return provider


def _keys(routes):
    return {ro.state_key(route.state) for route in routes}


def _search(beam, *, nested_prior_levels=None):
    universe, state, meta, prices = _universe()
    rows = {int(row["player_id"]): row for row in universe["universe"]}
    positions = {pid: row["position"] for pid, row in rows.items()}
    return ro.run_search(
        initial_state=state, events=list(EVENTS), rows=rows,
        pool_ids=[pid for pid in POOL_IDS if pid in rows],
        positions=positions, scenario=_scenario(prices), player_meta=meta,
        config=_config(beam), nested_prior_levels=nested_prior_levels,
    )


def _expand(frontier, *, event, prices, universe, state, meta):
    """One real expansion step, mirroring run_search's generation exactly."""

    rows = {int(row["player_id"]): row for row in universe["universe"]}
    positions = {pid: row["position"] for pid, row in rows.items()}
    successors = set()
    for partial in frontier:
        snapshot = _scenario(prices).snapshot_for(event)
        action_set = ro.generate_actions(
            state=partial.state, rows=rows, pool_ids=[pid for pid in POOL_IDS if pid in rows],
            positions=positions, price_snapshot=snapshot, player_meta=meta,
            config=_config(WIDE_BEAM), event=event,
        )
        for action in action_set["actions"]:
            transition = ts.apply_transfer_batch(partial.state, action["batch"], snapshot, meta)
            if not transition.ok:
                continue
            successors.add(ro.state_key(transition.next_event_state))
    return successors


# ---------------------------------------------------------------------------
# §1 RE-PIN — the described defect is NOT reproduced
# ---------------------------------------------------------------------------


def test_repin_run_search_always_starts_from_the_same_initial_route_state():
    """Source proof: the search begins from the initial state, never from a prior frontier."""

    source = (REPO_ROOT / "fpl_brain" / "route_optimizer.py").read_text(encoding="utf-8")
    assert "states = [PartialRoute(state=initial_state, actions=(), h1_proxy=0.0, window_proxy=0.0, hits=0)]" in source
    # the prior is merged only AFTER the level's generated set has been retained
    body = source[source.index("def run_search"):source.index("def build_event_worlds")]
    assert body.index("retained = _retain(generated") < body.index("if level_index < len(nested_prior_levels):")
    assert "merged = {state_key(item.state): item for item in retained}" in body


def test_repin_nested_prior_cannot_restrict_generation_at_any_level():
    """Generation depends on the frontier set, and the prior is a UNION onto it."""

    narrow = _search(NARROW_BEAM)
    wider_with_prior = _search(WIDE_BEAM, nested_prior_levels=narrow["level_survivors"])
    wider_without_prior = _search(WIDE_BEAM)
    # level 0 generation is identical with and without the prior -> the prior is
    # not a seed and cannot remove a single generated state
    assert narrow["stats"]["levels"][0]["states_generated"] == \
        wider_without_prior["stats"]["levels"][0]["states_generated"]
    assert wider_with_prior["stats"]["levels"][0]["states_generated"] == \
        wider_without_prior["stats"]["levels"][0]["states_generated"]
    # and the prior is genuinely add-only
    assert wider_with_prior["stats"]["nested_inherited"] > 0
    for level in wider_with_prior["stats"]["levels"]:
        assert level["nested_inherited"] >= 0


def test_repin_escalated_tree_is_a_superset_of_the_narrow_tree():
    narrow = _search(NARROW_BEAM)
    wider = _search(WIDE_BEAM, nested_prior_levels=narrow["level_survivors"])
    assert len(narrow["level_survivors"]) == len(wider["level_survivors"]) == len(EVENTS)
    for index in range(len(EVENTS)):
        assert _keys(narrow["level_survivors"][index]) <= _keys(wider["level_survivors"][index]), index
        assert wider["stats"]["levels"][index]["states_generated"] >= \
            narrow["stats"]["levels"][index]["states_generated"], index
    # a genuinely wider search, not a re-labelled narrow one
    assert wider["stats"]["levels"][0]["states_generated"] > 0
    assert wider["stats"]["heuristic_retained"] > narrow["stats"]["heuristic_retained"]


def test_repin_a_prefix_pruned_by_the_narrow_beam_is_recovered_by_the_escalation():
    """§7 A/B: narrow prunes X; the escalation keeps X and can expand it."""

    narrow = _search(NARROW_BEAM)
    wider = _search(WIDE_BEAM, nested_prior_levels=narrow["level_survivors"])

    narrow_frontier = _keys(narrow["level_survivors"][0])
    wider_frontier = _keys(wider["level_survivors"][0])
    # precondition: the narrow retention cap really did bind
    assert narrow["coverage_levels"][0]["heuristic_pruned"] > 0, "the narrow cap must bind"
    recovered = wider_frontier - narrow_frontier
    assert recovered, "A: the narrow beam pruned prefixes the escalation retains"
    assert not (wider_frontier < narrow_frontier), "B: the escalation is not a subset"

    universe, state, meta, prices = _universe()
    pruned = next(route for route in wider["level_survivors"][0]
                  if ro.state_key(route.state) in recovered)
    own = _expand([pruned], event=EVENTS[1], prices=prices, universe=universe, state=state, meta=meta)
    assert own, "X must have legal successors at the next event"
    assert not own <= _expand(narrow["level_survivors"][0], event=EVENTS[1], prices=prices,
                              universe=universe, state=state, meta=meta), (
        "the narrow frontier cannot reach every successor of the recovered prefix"
    )


def test_control_a_survivor_frontier_seeded_search_would_miss_the_pruned_prefix():
    """§7 C: the shape the claim assumes — expand only the narrow frontier — misses X.

    This is the CONTROL: it shows the described defect WOULD be real for a search
    that began from the beam-8 frontier, and therefore that the current
    implementation (which does not) is the thing that avoids it.
    """

    narrow = _search(NARROW_BEAM)
    wider = _search(WIDE_BEAM, nested_prior_levels=narrow["level_survivors"])
    recovered = _keys(wider["level_survivors"][0]) - _keys(narrow["level_survivors"][0])
    assert recovered, "precondition: the narrow beam pruned at least one prefix"

    universe, state, meta, prices = _universe()
    restricted = _expand(narrow["level_survivors"][0], event=EVENTS[1], prices=prices,
                         universe=universe, state=state, meta=meta)
    for route in wider["level_survivors"][0]:
        if ro.state_key(route.state) not in recovered:
            continue
        own = _expand([route], event=EVENTS[1], prices=prices, universe=universe,
                      state=state, meta=meta)
        assert not own <= restricted, (
            "a search restricted to the narrow frontier would have missed this prefix's successors"
        )
        break
    else:  # pragma: no cover - guarded by the recovery assertion above
        raise AssertionError("no recovered prefix found")


def test_repin_escalation_expands_the_search_tree_and_loses_nothing():
    """The wider search explores strictly more and never loses a narrow survivor.

    Note the honest scope: with a fixed ``exact_evaluation_budget`` the promoted
    top-N can coincide, because the extra breadth lives in the retained tail.  What
    is guaranteed — and what the breadth escalation relies on — is that the WIDER
    TREE contains strictly more retained routes at every level, so a route whose
    prefix the narrow retention pruned is present to be evaluated at all.
    """

    narrow = _search(NARROW_BEAM)
    wider = _search(WIDE_BEAM, nested_prior_levels=narrow["level_survivors"])
    narrow_promoted = {
        ro.canonical_family_signature(route)
        for route in ro._select_promoted(narrow["final_states"], _config(NARROW_BEAM))
    }
    wider_promoted = {
        ro.canonical_family_signature(route)
        for route in ro._select_promoted(wider["final_states"], _config(WIDE_BEAM))
    }
    assert narrow_promoted
    assert narrow_promoted <= wider_promoted, "the wider run must not lose a narrow promoted route"
    # strictly more is retained at the final level, so it is available to evaluate
    assert len(wider["final_states"]) > len(narrow["final_states"])
    assert _keys(narrow["final_states"]) < _keys(wider["final_states"])


def test_repin_the_real_escalation_shape_restarts_from_the_initial_state_and_reuses_worlds():
    """The production escalation closure: initial state, next ladder beam, Stage-2 worlds."""

    source = (REPO_ROOT / "scripts" / "run_four_gw_decision.py").read_text(encoding="utf-8")
    runner = source[source.index("def _escalation_runner"):]
    runner = runner[: runner.index("\nif __name__")]
    # the next SUPPORTED search budget, the Stage-2 draws, the SAME worlds, the
    # Stage-1 nested view, and the finalists forced in — nothing else changes
    assert "rs.budget_config(int(beam), base_config)" in runner
    assert "search_draws=int(stage2_draws)" in runner
    assert "generation=generation" in runner
    # NO cross-stage prebuilt-matrix reuse: the escalation reaches the SAME certified
    # generation through the content-addressed cache, which is optimization rather than
    # authority, so a cold cache rebuilds the same worlds instead of changing the answer.
    assert "prebuilt_worlds" not in runner
    assert "certified_bundle" not in runner
    assert "nested_prior=ro.nested_budget_view(stage1_result)" in runner
    assert "required_routes=list(finalist_partials)" in runner
    assert "initial_state=initial_state" in runner
    # it does not rebuild the universe, re-screen actions, or regenerate worlds
    for banned in ("build_universe", "screen_legal_actions", "build_event_worlds", "monte_carlo"):
        assert banned not in runner, banned


# ---------------------------------------------------------------------------
# §4–§6 — the FINAL ranking, canonical record and role-relevant set
# ---------------------------------------------------------------------------

A = "route_a"
B = "route_b"
C = "route_c"


def _record(route_id, value):
    return {"route_id": route_id, "supported_3gw_net_core": float(value),
            "canonical_family_signature": f"SIG_{route_id}", "family_signature": route_id,
            "terminal_ft": 2, "terminal_bank_tenths": 0, "valid": True, "actions": [],
            "h1_net_core": float(value) / 4.0}


def _pair(a, b, mean, se, worlds=10_000):
    near = (abs(mean) <= fr.NEAR_TIE_K * se) if se > 0 else mean == 0.0
    return {"route_a": a, "route_b": b, "worlds": worlds, "mean_difference": float(mean),
            "paired_se": float(se), "near_tied": bool(near)}


def _result(specs, pairs):
    return {"routes": {route_id: _record(route_id, value) for route_id, value in specs},
            "paired_supported_3gw": list(pairs)}


def test_final_ranking_uses_the_escalated_result_when_an_escalation_ran():
    stage2 = _result([(A, 30.0), (B, 29.9)], [_pair(A, B, 0.10, 0.10)])
    escalated = _result([(A, 30.0), (B, 29.9), (C, 29.95)],
                        [_pair(A, C, 0.05, 0.02), _pair(A, B, 0.10, 0.10)])
    final = fr.final_ranking_after_escalation(stage2_result=stage2, escalated_result=escalated)
    assert final["escalated"] is True
    assert final["ranking_source"] == "ESCALATED_FINAL_RANKING"
    assert final["comparator_source"] == "FINAL_WIDENED_RANKING"
    assert final["result"] is escalated
    assert final["stage2_leader_route_id"] == A
    # A stays the leader, but the FINAL runner-up is C (29.95), not B (29.9)
    assert final["final_ranking"] == {"preferred_route_id": A, "runner_up_route_id": C}
    assert not fr.final_ranking_after_escalation(
        stage2_result=stage2, escalated_result=None)["escalated"]


def test_canonical_route_b_comes_from_the_final_widened_ranking_not_the_stage2_runner_up():
    """§5/§8 case 1: A > C > B — the pair must be A-vs-C, never A-vs-B."""

    stage2 = _result([(A, 30.0), (B, 29.9)], [_pair(A, B, 0.10, 0.10)])
    escalated = _result([(A, 30.0), (B, 29.9), (C, 29.95)],
                        [_pair(A, C, 0.05, 0.02), _pair(A, B, 0.10, 0.10)])
    final = fr.final_ranking_after_escalation(stage2_result=stage2, escalated_result=escalated)
    canonical = final["canonical_paired_near_tie"]
    assert canonical["route_a"] == A
    assert canonical["route_b"] == C, "route_b must be the FINAL ranking's next route"
    assert canonical["route_b"] != B
    assert canonical["mean_difference"] == pytest.approx(0.05), "the pair is A-vs-C, not A-vs-B"
    assert canonical["worlds"] == 10_000
    assert canonical["near_tied"] is False   # 0.05 <= 1.96 * 0.02 is False
    # and the Stage-2-only ranking still reports B, so the two are distinguishable
    stage2_only = fr.final_ranking_after_escalation(stage2_result=stage2)
    assert stage2_only["canonical_paired_near_tie"]["route_b"] == B


def test_widened_search_that_changes_the_leader_drives_canonical_and_role_relevant():
    """§8 case 2: C > A > B — C is the final route and confidence consumes C-vs-A."""

    from fpl_brain import decision_confidence as dc

    stage2 = _result([(A, 30.0), (B, 29.9)], [_pair(A, B, 0.10, 0.10)])
    escalated = _result([(C, 31.0), (A, 30.0), (B, 29.9)],
                        [_pair(C, A, 0.60, 0.05), _pair(A, B, 0.10, 0.10)])
    final = fr.final_ranking_after_escalation(stage2_result=stage2, escalated_result=escalated)
    assert final["final_ranking"] == {"preferred_route_id": C, "runner_up_route_id": A}
    canonical = final["canonical_paired_near_tie"]
    assert canonical["route_a"] == C and canonical["route_b"] == A
    assert canonical["mean_difference"] == pytest.approx(0.60)

    # the stability gate classifies the material leader change rather than
    # silently declaring the narrow answer stable
    stability = fr.assess_search_stability(
        refined_result=stage2,
        leader_change=fr.analyze_leader_change(stage2, escalated),
        canonical_paired=fr.canonical_paired_record(stage2, leader_route_id=A, runner_up_route_id=B),
        escalation=lambda beam: escalated,
        config=fr.StabilityGateConfig(current_beam=8),
    )
    assert stability["state"] == fr.SEARCH_NOT_STABLE
    assert stability["escalation_used"] is True
    assert stability["stability_basis"] == "ESCALATED_LEADER_MATERIALLY_DIFFERENT"

    # confidence is computed from the FINAL canonical record (C vs A)
    confidence = dc.classify_decision_confidence(
        paired=canonical, role_evidence={}, focus_player_ids=[],
    )
    assert confidence["paired_delta_mean"] == pytest.approx(0.60)
    assert confidence["near_tie"] is False
    assert confidence["state"] == dc.CONFIDENCE_STRONG


def test_role_relevant_players_follow_the_final_preferred_route():
    """§6/§8: the role-relevant set is derived from the FINAL route, not the old one."""

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import run_four_gw_decision as runner

    transfers = {
        A: {5: [{"out": 11, "in": 41}]},
        C: {5: [{"out": 21, "in": 51}], 6: [{"out": 22, "in": 52}]},
    }
    lineup = {"captain_id": 51, "vice_captain_id": 31}
    assert runner._role_relevant_ids(transfers, A, lineup) == [11, 31, 41, 51]
    assert runner._role_relevant_ids(transfers, C, lineup) == [21, 22, 31, 51, 52]
    # the runner derives it from the FINAL leader and passes it to the classifier
    source = (REPO_ROOT / "scripts" / "run_four_gw_decision.py").read_text(encoding="utf-8")
    assert "preferred_key = final_leader or preferred_route_id or lineup_route_id" in source
    assert "focus_player_ids=role_relevant_players" in source


def test_runner_order_adopts_the_final_result_before_the_decision_and_confidence():
    """§4/§6 wiring: widening happens before the board, canonical and confidence.

    The route-adaptation literal is matched over the KNOWN adapter names rather than one
    hard-coded spelling: R5-P0-01 intentionally moved the runner onto
    ``optimizer_routes_for_decision``, and a single ``source.index`` would silently stop
    guarding the ordering the moment the call is renamed - the same brittleness that let
    R5-P0-01 survive a green suite.
    """

    source = (REPO_ROOT / "scripts" / "run_four_gw_decision.py").read_text(encoding="utf-8")
    stability_at = source.index("stability = fr.assess_search_stability(")
    adopt_at = source.index("final = fr.final_ranking_after_escalation(")
    board_at = min(
        (at for at in (source.find("fg.routes_for_decision("),
                       source.find("fg.optimizer_routes_for_decision("),
                       source.find("fg.canonical_decision_route("))
         if at != -1),
        default=-1,
    )
    assert board_at != -1, "the runner must adapt its routes for the decision layer"
    evaluate_at = source.index("decision = fg.evaluate_four_gw_decision(")
    canonical_at = source.rindex("fr.final_ranking_after_escalation(")
    confidence_at = source.index("confidence = dc.classify_decision_confidence(")
    assert stability_at < adopt_at < board_at < evaluate_at < canonical_at < confidence_at
    # the canonical record is published against the DECISION's preferred route
    assert "preferred_route_id=decision_preferred or final[" in source
    # and it is the FINAL result that carries it
    assert 'result["canonical_paired_near_tie"] = canonical' in source
    assert "escalated_result_sink=escalated_sink" in source


def test_escalated_result_sink_is_always_written():
    a, b = _result([(A, 30.0), (B, 29.9)], [_pair(A, B, 0.10, 0.10)]), None
    canonical = fr.canonical_paired_record(a, leader_route_id=A, runner_up_route_id=B)
    sink: dict = {}
    fr.assess_search_stability(
        refined_result=a, leader_change={"statistically_material": False},
        canonical_paired=canonical, escalation=lambda beam: a,
        config=fr.StabilityGateConfig(current_beam=8), escalated_result_sink=sink,
    )
    assert sink["result"] is a
    sink2: dict = {"result": "stale"}
    fr.assess_search_stability(
        refined_result=a, leader_change={"statistically_material": False},
        canonical_paired=canonical, escalation=None,
        config=fr.StabilityGateConfig(current_beam=48), escalated_result_sink=sink2,
    )
    assert sink2["result"] is None, "a stale value must never survive"


# ---------------------------------------------------------------------------
# §9 — preserved properties
# ---------------------------------------------------------------------------


def test_preserved_properties_are_intact():
    source = (REPO_ROOT / "fpl_brain" / "finalist_refinement.py").read_text(encoding="utf-8")
    runner = (REPO_ROOT / "scripts" / "run_four_gw_decision.py").read_text(encoding="utf-8")
    assert fr.STAGE1_DRAWS == 2_000 and fr.STAGE2_DRAWS == 10_000
    assert rs.LADDER_BUDGET_SEMANTICS == "BEAM_WIDTH_SEARCH_BREADTH_NOT_MONTE_CARLO_DRAWS"
    assert rs.LADDER_BUDGETS == (12, 24, 48)
    assert fr.SEARCH_STABLE == ro.SEARCH_STABLE
    # one escalation maximum, still
    assert runner.count("escalation=_escalation_runner(") == 1
    assert runner.count("assess_search_stability(") == 1
    assert runner.count("escalated_result_sink=escalated_sink") == 1
    # no best-H1 fallback, no chip capability, no ownership/popularity input
    for banned in ("best_h1", "BEST_H1", "PLAY_WILDCARD"):
        assert banned not in runner, banned
    for banned in ("ownership", "selected_by_percent", "popularity"):
        assert banned not in source, banned
