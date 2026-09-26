"""R4B.2b — finalist refinement + paired search-stability gate tests.

Synthetic and deterministic wherever a real Monte Carlo run is not REQUIRED.
Nothing here runs a production prediction, a production route search, or writes
to the project database.  The one live-kernel test (#16-N) exercises the real
Monte Carlo kernel on a synthetic fixture, because the world-prefix property is a
kernel property and must be proven, not assumed.

Test map (task section -> test name):
  15 A-J  test_finalist_*  / test_canonical_*  / test_missing_paired_*
  16 K-R  test_stage1_*  / test_stage2_*  / test_prefix_*  / test_leader_change_*
  17 S-AD test_ladder_*  / test_escalation_*  / test_stable_*  / test_unstable_*
          test_no_best_h1_*  / test_confidence_*  / test_r4b2a_*  / test_r4b2c_*
          test_wildcard_*  / test_projection_runs_*
"""

from __future__ import annotations

import copy
import hashlib
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fpl_brain import decision_confidence as dc
from fpl_brain import finalist_refinement as fr
from fpl_brain import four_gw_decision as fg
from fpl_brain import monte_carlo as mc
from fpl_brain import route_optimizer as ro
from fpl_brain import route_stability as rs
from fpl_brain import transfer_state as ts
from fpl_brain.scoring_rules import DEFAULT_SCORING_RULES
from test_transfer_state import CLUB, POSITION, SQUAD_IDS

EVENTS = (5, 6, 7, 8)
REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Synthetic optimizer-result builders
# ---------------------------------------------------------------------------


def _partial(tag: str, *, hits: int = 0, bank: int = 0, ft: int = 2, in_id: int = 41):
    """A distinguishable PartialRoute with a DISTINCT canonical family signature."""

    uniq = int(hashlib.sha256(str(tag).encode()).hexdigest()[:6], 16) % 100_000
    players = tuple(
        ts.RoutePlayer(pid, POSITION[pid], CLUB[pid], 50 + (uniq if pid == SQUAD_IDS[0] else 0))
        for pid in SQUAD_IDS
    )
    state = ts.RouteState(event=EVENTS[0], players=players, bank_tenths=int(bank),
                          free_transfers=int(ft))
    batch = ts.TransferBatch((ts.TransferAction(int(SQUAD_IDS[0]), int(in_id)),))
    actions = ({
        "event": EVENTS[0], "kind": tag, "batch": batch, "transition": None,
        "hit_points": int(hits), "ft_after": int(ft), "bank_after": int(bank),
        "squad_ids": tuple(sorted(SQUAD_IDS)),
    },)
    return ro.PartialRoute(state=state, actions=actions, h1_proxy=0.0, window_proxy=0.0, hits=int(hits))


def _roll_partial(bank: int = 0, ft: int = 2):
    players = tuple(ts.RoutePlayer(pid, POSITION[pid], CLUB[pid], 50) for pid in SQUAD_IDS)
    state = ts.RouteState(event=EVENTS[0], players=players, bank_tenths=int(bank), free_transfers=int(ft))
    actions = ({
        "event": event, "kind": "ROLL", "batch": ts.TransferBatch(()), "transition": None,
        "hit_points": 0, "ft_after": int(ft), "bank_after": int(bank),
        "squad_ids": tuple(sorted(SQUAD_IDS)),
    } for event in EVENTS)
    return ro.PartialRoute(state=state, actions=tuple(actions), h1_proxy=0.0, window_proxy=0.0, hits=0)


def _record(partial, value, *, ft=None, bank=None, rescued=False):
    return {
        "route_id": None,  # filled by the caller
        "canonical_family_signature": ro.canonical_family_signature(partial),
        "family_signature": ro.family_signature(partial),
        "actions": ro._serialize_route(partial),
        "supported_3gw_net_core": float(value),
        "h1_net_core": float(value) / 4.0,
        "cumulative_hits": int(partial.hits),
        "terminal_ft": int(partial.state.free_transfers if ft is None else ft),
        "terminal_bank_tenths": int(partial.state.bank_tenths if bank is None else bank),
        "valid": True,
        "uses_rescued_player": bool(rescued),
        "selected_by_percent": 0.0,          # must never influence selection
        "transfers_in_event": 0,             # must never influence selection
    }


def _pair(a, b, mean, se, *, worlds=2000):
    near = (abs(mean) <= fr.NEAR_TIE_K * se) if se > 0 else mean == 0.0
    return {"route_a": a, "route_b": b, "worlds": int(worlds), "mean_difference": float(mean),
            "paired_se": float(se), "near_tied": bool(near)}


def _result(specs, *, roll_id=None, pairs=(), promoted=None, extra_routes=()):
    """``specs``: list of (route_id, PartialRoute, value[, rescued])."""

    records: dict[str, dict] = {}
    partials: dict[str, ro.PartialRoute] = {}
    for spec in specs:
        route_id, partial, value = spec[0], spec[1], spec[2]
        rescued = bool(spec[3]) if len(spec) > 3 else False
        record = _record(partial, value, rescued=rescued)
        record["route_id"] = route_id
        records[route_id] = record
        partials[route_id] = partial
    for route_id, record in extra_routes:
        records[route_id] = record
    if promoted is None:
        promoted = [partials[rid] for rid in records if rid in partials]
    return {
        "routes": records,
        "paired_supported_3gw": list(pairs),
        "roll_baseline": ({"route_id": roll_id, "supported_3gw_net_core": records[roll_id]["supported_3gw_net_core"]}
                          if roll_id and roll_id in records else None),
        "promoted_routes": list(promoted),
    }


# ---------------------------------------------------------------------------
# 15 A-J — finalists + canonical paired record
# ---------------------------------------------------------------------------


def test_finalist_stage1_leader_is_always_a_finalist():
    a, b, c = _partial("A", in_id=41), _partial("B", in_id=42), _partial("C", in_id=43)
    result = _result([("r_a", a, 30.0), ("r_b", b, 29.0), ("r_c", c, 10.0)])
    selection = fr.select_finalists(result)
    assert selection["leader_route_id"] == "r_a"
    assert "r_a" in selection["finalist_route_ids"]
    reasons = {item["route_id"]: item["reasons"] for item in selection["finalists"]}
    assert fr.REASON_STAGE1_LEADER in reasons["r_a"]


def test_finalist_roll_is_always_a_finalist():
    roll = _roll_partial()
    a = _partial("A", in_id=41)
    result = _result([("r_roll", roll, 5.0), ("r_a", a, 30.0)], roll_id="r_roll")
    selection = fr.select_finalists(result)
    assert "r_roll" in selection["finalist_route_ids"]
    reasons = {item["route_id"]: item["reasons"] for item in selection["finalists"]}
    assert fr.REASON_ROLL_BASELINE in reasons["r_roll"]


def test_finalist_every_paired_near_tied_route_becomes_a_finalist():
    a, b, c, d = (_partial("A", in_id=41), _partial("B", in_id=42),
                  _partial("C", in_id=43), _partial("D", in_id=44))
    result = _result(
        [("r_a", a, 30.0), ("r_b", b, 29.9), ("r_c", c, 29.5), ("r_d", d, 10.0)],
        pairs=[_pair("r_a", "r_b", 0.10, 0.10),    # near tie: 0.10 <= 1.96*0.10
               _pair("r_a", "r_c", 0.50, 0.10),    # NOT near tied
               _pair("r_a", "r_d", 20.0, 1.0)],    # NOT near tied
    )
    selection = fr.select_finalists(result)
    assert selection["finalist_route_ids"] == ["r_a", "r_b"]
    assert selection["near_tied_route_ids"] == ["r_b"]


def test_finalist_rescue_lens_route_is_not_promoted_by_the_lens():
    """17-D / 15-D: a lens-rescued route is a finalist only on the near-tie rule."""

    a, rescued, near = _partial("A", in_id=41), _partial("R", in_id=42), _partial("N", in_id=43)
    result = _result(
        [("r_a", a, 30.0), ("r_rescued", rescued, 5.0, True), ("r_near", near, 29.9, True)],
        pairs=[_pair("r_a", "r_rescued", 25.0, 1.0),   # clearly separated
               _pair("r_a", "r_near", 0.05, 0.10)],    # near tied
    )
    selection = fr.select_finalists(result)
    assert "r_rescued" not in selection["finalist_route_ids"]
    assert "r_near" in selection["finalist_route_ids"]
    audit = {item["route_id"]: item for item in selection["rescue_lens_routes_not_promoted"]}
    assert audit["r_rescued"]["uses_rescued_player"] is True
    assert audit["r_rescued"]["near_tied_with_leader"] is False
    # no objective bonus for being rescued
    assert selection["leader_route_id"] == "r_a"


def test_finalist_no_ownership_or_popularity_input_affects_selection():
    source = (REPO_ROOT / "fpl_brain" / "finalist_refinement.py").read_text(encoding="utf-8")
    for banned in ("selected_by_percent", "ownership", "popularity", "transfers_in_event",
                   "Salah", "Haaland", "Palmer"):
        assert banned not in source, banned
    # functional: a hugely popular, non-near-tied route is not promoted
    a, popular = _partial("A", in_id=41), _partial("P", in_id=42)
    record = _record(popular, 1.0)
    record["route_id"] = "r_popular"
    record["selected_by_percent"] = 99.9
    result = _result([("r_a", a, 30.0)], extra_routes=[("r_popular", record)],
                     promoted=[a, popular])
    result["paired_supported_3gw"] = [_pair("r_a", "r_popular", 29.0, 0.5)]
    assert fr.select_finalists(result)["finalist_route_ids"] == ["r_a"]


def test_canonical_paired_key_published_for_a_decisive_final_leader():
    a, b = _partial("A", in_id=41), _partial("B", in_id=42)
    pairs = [_pair("r_a", "r_b", 3.0, 0.5)]
    refined = _result([("r_a", a, 30.0), ("r_b", b, 27.0)], pairs=pairs)
    canonical = fr.canonical_paired_record(refined, leader_route_id="r_a", runner_up_route_id="r_b")
    assert canonical is not None
    assert canonical["route_a"] == "r_a" and canonical["route_b"] == "r_b"
    assert canonical["near_tied"] is False
    assert canonical["role"] == "FINAL_PREFERRED_LEADER_VS_FINAL_RUNNER_UP"
    assert fr.canonical_paired_record(
        refined, leader_route_id="r_a", runner_up_route_id="r_absent") is None


def test_canonical_route_a_is_the_final_preferred_leader_and_b_the_runner_up():
    a, b = _partial("A", in_id=41), _partial("B", in_id=42)
    # reversed orientation in the published pair must be normalised
    refined = _result([("r_a", a, 30.0), ("r_b", b, 29.9)],
                      pairs=[_pair("r_b", "r_a", 0.10, 0.10)])
    assert fr.ranked_route_ids(refined) == ["r_a", "r_b"]
    canonical = fr.canonical_paired_record(refined, leader_route_id="r_a", runner_up_route_id="r_b")
    assert canonical["route_a"] == "r_a" and canonical["route_b"] == "r_b"
    assert canonical["orientation_normalised"] is True
    assert canonical["mean_difference"] == pytest.approx(-0.10)


def test_canonical_near_tied_matches_the_paired_1_96_se_criterion_exactly():
    a, b = _partial("A", in_id=41), _partial("B", in_id=42)
    for mean, se in ((0.19, 0.10), (0.20, 0.10), (0.21, 0.10), (0.0, 0.0), (0.5, 1.0)):
        refined = _result([("r_a", a, 30.0), ("r_b", b, 29.9)],
                          pairs=[_pair("r_a", "r_b", mean, se)])
        canonical = fr.canonical_paired_record(refined, leader_route_id="r_a", runner_up_route_id="r_b")
        expected = (abs(mean) <= 1.96 * se) if se > 0 else mean == 0.0
        assert canonical["near_tied"] is expected, (mean, se)
        assert canonical["near_tie_k"] == 1.96
        # the 0.25 materiality threshold is never the near-tie rule
        assert canonical["near_tied"] == fr.near_tie_verdict(
            {"mean_difference": mean, "paired_se": se})["near_tied"]


def test_missing_paired_evidence_cannot_produce_strong_recommendation():
    a, b = _partial("A", in_id=41), _partial("B", in_id=42)
    refined = _result([("r_a", a, 30.0), ("r_b", b, 27.0)], pairs=[])  # no paired evidence
    assert fr.canonical_paired_record(refined, leader_route_id="r_a", runner_up_route_id="r_b") is None
    confidence = dc.classify_decision_confidence(
        paired=None, role_evidence={}, focus_player_ids=[],
    )
    assert confidence["state"] == dc.CONFIDENCE_INCOMPLETE
    assert confidence["state"] != dc.CONFIDENCE_STRONG
    assert confidence["confidence_diagnostic"] == dc.DIAG_PAIRED_DIAGNOSTIC_REQUIRED
    assert confidence["decisive"] is False


# ---------------------------------------------------------------------------
# 16 K-R — higher-draw refinement
# ---------------------------------------------------------------------------


def test_stage1_draws_are_two_thousand_and_stage2_is_explicitly_higher():
    assert fr.STAGE1_DRAWS == 2_000
    assert ro.OptimizerConfig(events=EVENTS).search_draws == fr.STAGE1_DRAWS
    assert fr.STAGE2_DRAWS > fr.STAGE1_DRAWS
    assert fr.STAGE2_DRAWS == 10_000


class _RecordingOptimizer:
    def __init__(self, refined_result):
        self.calls = []
        self.refined_result = refined_result

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return self.refined_result


def _refinement_fixture():
    a, b, roll = _partial("A", in_id=41), _partial("B", in_id=42), _roll_partial()
    stage1 = _result([("r_a", a, 30.0), ("r_b", b, 29.9), ("r_roll", roll, 5.0)], roll_id="r_roll",
                     pairs=[_pair("r_a", "r_b", 0.10, 0.10)])
    refined = _result([("s_a", a, 30.0), ("s_b", b, 29.8), ("s_roll", roll, 5.0)], roll_id="s_roll",
                      pairs=[_pair("s_a", "s_b", 0.22, 0.10)])
    return stage1, refined


def test_stage2_uses_the_declared_higher_draw_budget_and_the_same_inputs():
    stage1, refined = _refinement_fixture()
    optimizer = _RecordingOptimizer(refined)
    base_config = ro.OptimizerConfig(events=EVENTS, search_draws=fr.STAGE1_DRAWS, seed=20260911,
                                     beam_width=8, exact_evaluation_budget=20)
    sentinel_universe = {"universe": [], "replacement_edges": []}
    report = fr.refine_finalists(
        universe=sentinel_universe, initial_state=_partial("X").state,
        scenario="SCENARIO", player_meta={41: "META"}, base_config=base_config,
        stage1_result=stage1,
        # The declared NON-PRODUCTION door: this fixture supplies worlds that were never
        # loaded from a certified generation, and it says so out loud.
        non_production_worlds=_np({5: {"worlds": 10_000, "player_ids": []}}),
        verify_prefix=False, optimizer=optimizer,
    )
    assert len(optimizer.calls) == 1
    call = optimizer.calls[0]
    assert call["config"].search_draws == fr.STAGE2_DRAWS
    assert call["config"].seed == base_config.seed
    assert call["config"].events == base_config.events
    assert call["universe"] is sentinel_universe
    assert call["scenario"] == "SCENARIO"
    assert call["player_meta"] == {41: "META"}
    assert call["config"].exact_evaluation_budget == 0, "only finalists may be exact-evaluated"
    # the refined result ranks ONLY finalists
    assert report["final_ranking"]["preferred_route_id"] == "s_a"
    assert report["final_ranking"]["runner_up_route_id"] == "s_b"
    assert report["canonical_paired_near_tie"]["route_a"] == "s_a"
    assert report["canonical_paired_near_tie"]["route_b"] == "s_b"


def test_stage2_refinement_creates_no_predictive_run():
    """Read-only: the refinement never touches a repository or writes a run."""

    stage1, refined = _refinement_fixture()

    class _WriteForbidden:
        def __getattr__(self, name):
            raise AssertionError(f"refinement touched the database via {name!r}")

    optimizer = _RecordingOptimizer(refined)
    fr.refine_finalists(
        universe={"universe": [], "replacement_edges": []}, initial_state=_partial("X").state,
        scenario=None, player_meta={}, base_config=ro.OptimizerConfig(events=EVENTS, seed=1),
        stage1_result=stage1, conn=_WriteForbidden(),
        non_production_worlds=_np({5: {"worlds": 10_000, "player_ids": []}}),
        verify_prefix=False, optimizer=optimizer,
    )
    source = (REPO_ROOT / "fpl_brain" / "finalist_refinement.py").read_text(encoding="utf-8")
    for banned in ("INSERT", "UPDATE ", "DELETE ", "projection_runs", "monte_carlo_distributions",
                   "repositories", "session_post", "execute("):
        assert banned not in source, banned


def test_prefix_invariance_first_2000_worlds_of_10000_match_stage1_real_kernel():
    """16-N: proven on the REAL Monte Carlo kernel, captured surfaces only."""

    from test_monte_carlo import _fixture

    fixture = _fixture(fixture_id=100, event=4)
    capture = [1, 2, 11, 12, 13, 21, 22, 23, 31, 32]
    low = mc.simulate({100: copy.deepcopy(fixture)},
                      mc.MonteCarloConfig(simulations=2_000, seed=20260911), DEFAULT_SCORING_RULES,
                      capture_player_ids=capture)["world_matrix"]
    high = mc.simulate({100: copy.deepcopy(fixture)},
                       mc.MonteCarloConfig(simulations=10_000, seed=20260911), DEFAULT_SCORING_RULES,
                       capture_player_ids=capture)["world_matrix"]
    report = fr.compare_world_prefix(low, high)
    assert report["status"] == fr.PREFIX_STATUS_PASS, report["first_mismatches"]
    assert report["scope"] == fr.PREFIX_INVARIANCE_SCOPE
    assert report["worlds_compared"] == 2_000
    assert report["shared_captured_players"] == len(capture)
    # the claim is deliberately narrow
    assert "NOT every hidden RNG stream" in report["claim"]
    # a different seed must NOT pass, proving the check can fail
    other = mc.simulate({100: copy.deepcopy(fixture)},
                        mc.MonteCarloConfig(simulations=2_000, seed=777), DEFAULT_SCORING_RULES,
                        capture_player_ids=capture)["world_matrix"]
    assert fr.compare_world_prefix(other, high)["status"] == fr.PREFIX_STATUS_FAIL


def test_leader_change_statistical_and_practical_materiality_are_both_required():
    a, b = _partial("A", in_id=41), _partial("B", in_id=42)
    stage1 = _result([("x_a", a, 30.0), ("x_b", b, 29.0)])
    # 16-P: tiny numerical reorder inside the near-tie band
    p = _result([("y_b", b, 30.01), ("y_a", a, 30.00)], pairs=[_pair("y_b", "y_a", 0.01, 0.10)])
    change = fr.analyze_leader_change(stage1, p)
    assert change["changed"] is True
    assert change["statistically_material"] is False
    assert change["accepted"] is False
    # 16-Q: statistically material but not practically material (<= 0.25 CORE)
    q = _result([("y_b", b, 30.30), ("y_a", a, 30.10)], pairs=[_pair("y_b", "y_a", 0.20, 0.02)])
    change = fr.analyze_leader_change(stage1, q)
    assert change["statistically_material"] is True
    assert change["practically_material"] is False
    assert change["accepted"] is False
    # 16-R: both -> material, direction recorded
    r = _result([("y_b", b, 31.0), ("y_a", a, 30.5)], pairs=[_pair("y_b", "y_a", 0.50, 0.02)])
    change = fr.analyze_leader_change(stage1, r)
    assert change["statistically_material"] is True and change["practically_material"] is True
    assert change["accepted"] is True
    assert change["direction"] == fr.DIRECTION_REFINED_LEADER_BETTER
    # unchanged leader
    same = _result([("z_a", a, 30.0), ("z_b", b, 29.0)])
    assert fr.analyze_leader_change(stage1, same)["changed"] is False


# ---------------------------------------------------------------------------
# 17 S-AD — search stability
# ---------------------------------------------------------------------------


def test_ladder_budgets_are_beam_widths_not_monte_carlo_draws():
    assert rs.LADDER_BUDGETS == (12, 24, 48)
    assert rs.LADDER_BUDGET_SEMANTICS == "BEAM_WIDTH_SEARCH_BREADTH_NOT_MONTE_CARLO_DRAWS"
    base = ro.OptimizerConfig(events=EVENTS, search_draws=2_000, seed=11, beam_width=8)
    escalated = rs.budget_config(24, base)
    assert escalated.beam_width == 24
    assert escalated.search_draws == base.search_draws, "a search budget must not change the draw count"
    assert escalated.seed == base.seed
    # retention budgets derive from the BEAM WIDTH
    assert ro.retention_budgets(escalated)["KN"] == max(4 * 24, 24)
    assert rs.next_ladder_budget(8) == 12
    assert rs.next_ladder_budget(12) == 24
    assert rs.next_ladder_budget(48) is None
    assert rs.next_ladder_budget(8, (12, 24, 48)) == 12


def _escalation_counter(escalated_result):
    calls = []

    def escalation(beam):
        calls.append(int(beam))
        return escalated_result

    return calls, escalation


def test_escalation_is_bounded_to_exactly_one():
    a, b = _partial("A", in_id=41), _partial("B", in_id=42)
    refined = _result([("r_a", a, 30.0), ("r_b", b, 29.9)], pairs=[_pair("r_a", "r_b", 0.05, 0.10)])
    canonical = fr.canonical_paired_record(refined, leader_route_id="r_a", runner_up_route_id="r_b")
    escalated = _result([("e_c", _partial("C", in_id=43), 40.0), ("e_a", a, 30.0)],
                        pairs=[_pair("e_c", "e_a", 5.0, 0.2)])
    calls, escalation = _escalation_counter(escalated)
    report = fr.assess_search_stability(
        refined_result=refined, leader_change={"statistically_material": False},
        canonical_paired=canonical, escalation=escalation,
        config=fr.StabilityGateConfig(current_beam=8),
    )
    assert len(calls) == 1, "at most one escalation"
    assert report["escalation_bounded_to_one"] is True
    assert report["search_budget_sequence"] == [8, 12]
    assert report["state"] == fr.SEARCH_NOT_STABLE


def test_escalation_reports_instability_when_no_larger_budget_exists():
    a, b = _partial("A", in_id=41), _partial("B", in_id=42)
    refined = _result([("r_a", a, 30.0), ("r_b", b, 29.9)], pairs=[_pair("r_a", "r_b", 0.05, 0.10)])
    canonical = fr.canonical_paired_record(refined, leader_route_id="r_a", runner_up_route_id="r_b")
    calls, escalation = _escalation_counter(refined)
    report = fr.assess_search_stability(
        refined_result=refined, leader_change={"statistically_material": False},
        canonical_paired=canonical, escalation=escalation,
        config=fr.StabilityGateConfig(current_beam=48),
    )
    assert calls == []
    assert report["escalation_used"] is False
    assert report["state"] == fr.SEARCH_NOT_STABLE
    assert report["stability_basis"] == "NO_LARGER_SUPPORTED_SEARCH_BUDGET"


def test_stable_same_leader_after_escalation_is_search_stable():
    """17-U."""

    a, b = _partial("A", in_id=41), _partial("B", in_id=42)
    refined = _result([("r_a", a, 30.0), ("r_b", b, 29.9)], pairs=[_pair("r_a", "r_b", 0.05, 0.10)])
    canonical = fr.canonical_paired_record(refined, leader_route_id="r_a", runner_up_route_id="r_b")
    escalated = _result([("e_a", a, 31.0), ("e_b", b, 30.0)], pairs=[_pair("e_a", "e_b", 1.0, 0.1)])
    calls, escalation = _escalation_counter(escalated)
    report = fr.assess_search_stability(
        refined_result=refined, leader_change={"statistically_material": False},
        canonical_paired=canonical, escalation=escalation,
        config=fr.StabilityGateConfig(current_beam=8),
    )
    assert len(calls) == 1
    assert report["leaders_agree"] is True
    assert report["state"] == fr.SEARCH_STABLE


def test_materially_unstable_preferred_route_after_escalation_is_not_stable():
    """17-V."""

    a, b, c = _partial("A", in_id=41), _partial("B", in_id=42), _partial("C", in_id=43)
    refined = _result([("r_a", a, 30.0), ("r_b", b, 29.9)], pairs=[_pair("r_a", "r_b", 0.05, 0.10)])
    canonical = fr.canonical_paired_record(refined, leader_route_id="r_a", runner_up_route_id="r_b")
    escalated = _result([("e_c", c, 40.0), ("e_a", a, 30.0)], pairs=[_pair("e_c", "e_a", 5.0, 0.2)])
    calls, escalation = _escalation_counter(escalated)
    report = fr.assess_search_stability(
        refined_result=refined, leader_change={"statistically_material": False},
        canonical_paired=canonical, escalation=escalation,
        config=fr.StabilityGateConfig(current_beam=8),
    )
    assert report["leaders_agree"] is False
    assert report["escalated_leader_statistically_material"] is True
    assert report["escalated_leader_practically_material"] is True
    assert report["state"] == fr.SEARCH_NOT_STABLE


def test_escalated_non_material_reorder_remains_stable():
    """A numerical reorder at the wider budget is not instability."""

    a, b, c = _partial("A", in_id=41), _partial("B", in_id=42), _partial("C", in_id=43)
    refined = _result([("r_a", a, 30.0), ("r_b", b, 29.9)], pairs=[_pair("r_a", "r_b", 0.05, 0.10)])
    canonical = fr.canonical_paired_record(refined, leader_route_id="r_a", runner_up_route_id="r_b")
    escalated = _result([("e_c", c, 30.05), ("e_a", a, 30.00)], pairs=[_pair("e_c", "e_a", 0.05, 0.10)])
    calls, escalation = _escalation_counter(escalated)
    report = fr.assess_search_stability(
        refined_result=refined, leader_change={"statistically_material": False},
        canonical_paired=canonical, escalation=escalation,
        config=fr.StabilityGateConfig(current_beam=8),
    )
    assert report["leaders_agree"] is False
    assert report["state"] == fr.SEARCH_STABLE
    assert report["stability_basis"] == "ESCALATED_REORDER_NOT_MATERIAL"


def test_decisive_refined_win_needs_no_escalation_and_never_claims_one_beam():
    a, b = _partial("A", in_id=41), _partial("B", in_id=42)
    refined = _result([("r_a", a, 30.0), ("r_b", b, 26.0)], pairs=[_pair("r_a", "r_b", 4.0, 0.2)])
    canonical = fr.canonical_paired_record(refined, leader_route_id="r_a", runner_up_route_id="r_b")
    calls, escalation = _escalation_counter(refined)
    report = fr.assess_search_stability(
        refined_result=refined, leader_change={"statistically_material": False},
        canonical_paired=canonical, escalation=escalation,
        config=fr.StabilityGateConfig(current_beam=8),
    )
    assert calls == []
    assert report["escalation_used"] is False
    assert report["state"] == fr.SEARCH_STABLE
    assert report["stability_basis"] == "DECISIVE_REFINED_PAIRED_WIN"


def test_small_but_resolved_margin_still_gets_one_breadth_check():
    """A statistically resolved but practically immaterial win is not left untested."""

    a, b = _partial("A", in_id=41), _partial("B", in_id=42)
    refined = _result([("r_a", a, 30.00), ("r_b", b, 29.90)], pairs=[_pair("r_a", "r_b", 0.10, 0.02)])
    canonical = fr.canonical_paired_record(refined, leader_route_id="r_a", runner_up_route_id="r_b")
    assert canonical["near_tied"] is False, "0.10 > 1.96 * 0.02"
    assert abs(canonical["mean_difference"]) <= fr.MATERIAL_FRONTIER_CHANGE_CORE
    escalated = _result([("e_a", a, 30.0), ("e_b", b, 29.9)], pairs=[_pair("e_a", "e_b", 0.1, 0.02)])
    calls, escalation = _escalation_counter(escalated)
    report = fr.assess_search_stability(
        refined_result=refined, leader_change={"statistically_material": False},
        canonical_paired=canonical, escalation=escalation,
        config=fr.StabilityGateConfig(current_beam=8),
    )
    assert report["escalation_trigger"] == "IMMATERIAL_PAIRED_MARGIN"
    assert len(calls) == 1
    assert report["state"] == fr.SEARCH_STABLE, "the wider search reproduces the same leader"
    assert report["stability_basis"] == "ESCALATED_SEARCH_REPRODUCES_LEADER"


def test_resolved_immaterial_margin_without_a_larger_budget_fails_closed():
    a, b = _partial("A", in_id=41), _partial("B", in_id=42)
    refined = _result([("r_a", a, 30.00), ("r_b", b, 29.90)], pairs=[_pair("r_a", "r_b", 0.10, 0.02)])
    canonical = fr.canonical_paired_record(refined, leader_route_id="r_a", runner_up_route_id="r_b")
    calls, escalation = _escalation_counter(refined)
    report = fr.assess_search_stability(
        refined_result=refined, leader_change={"statistically_material": False},
        canonical_paired=canonical, escalation=escalation,
        config=fr.StabilityGateConfig(current_beam=48),
    )
    assert calls == []
    assert report["state"] == fr.SEARCH_NOT_STABLE
    assert report["stability_basis"] == "NO_LARGER_SUPPORTED_SEARCH_BUDGET"


def test_missing_canonical_record_suppresses_the_recommendation():
    runner = _runner_module()
    decision = {
        "transfer_recommendation": {
            "status": fg.RECOMMENDATION_AVAILABLE,
            "preferred_route_id": "route_001",
            "ranking": [{"rank": 1, "route_id": "route_001"}],
        }
    }
    suppressed = runner._suppress_transfer_recommendation(
        decision, reason=dc.DIAG_PAIRED_DIAGNOSTIC_REQUIRED,
        extra={"paired_diagnostic_required": True},
    )
    assert suppressed["status"] == fg.RECOMMENDATION_SUPPRESSED
    assert suppressed["reason"] == dc.DIAG_PAIRED_DIAGNOSTIC_REQUIRED
    assert suppressed["preferred_route_id"] is None
    assert suppressed["ranking"], "the route table stays available"
    source = (REPO_ROOT / "scripts" / "run_four_gw_decision.py").read_text(encoding="utf-8")
    assert "if canonical is None:" in source
    assert source.rindex('reason=dc.DIAG_PAIRED_DIAGNOSTIC_REQUIRED') < source.index(
        '(out_dir / "four_gw_decision.json").write_text(')


def test_a_failed_escalation_fails_closed():
    a, b = _partial("A", in_id=41), _partial("B", in_id=42)
    refined = _result([("r_a", a, 30.0), ("r_b", b, 29.9)], pairs=[_pair("r_a", "r_b", 0.05, 0.10)])
    canonical = fr.canonical_paired_record(refined, leader_route_id="r_a", runner_up_route_id="r_b")

    def boom(beam):
        raise RuntimeError("escalation unavailable")

    report = fr.assess_search_stability(
        refined_result=refined, leader_change={"statistically_material": False},
        canonical_paired=canonical, escalation=boom,
        config=fr.StabilityGateConfig(current_beam=8),
    )
    assert report["state"] == fr.SEARCH_NOT_STABLE
    assert report["diagnostic"] == fr.DIAG_ESCALATION_FAILED


def test_stability_never_re_ranks_and_adds_no_objective_points():
    source = (REPO_ROOT / "fpl_brain" / "finalist_refinement.py").read_text(encoding="utf-8")
    # stability is a gate: it must not build a bonus term
    for banned in ("+ stability", "stability_bonus", "+ confidence", "confidence_bonus",
                   "ownership_bonus", "popularity_bonus", "bank_bonus", "ft_bonus"):
        assert banned not in source, banned
    a, b = _partial("A", in_id=41), _partial("B", in_id=42)
    refined = _result([("r_a", a, 30.0), ("r_b", b, 29.9)], pairs=[_pair("r_a", "r_b", 0.05, 0.10)])
    canonical = fr.canonical_paired_record(refined, leader_route_id="r_a", runner_up_route_id="r_b")
    before = fr.ranked_route_ids(refined)
    escalated = _result([("e_a", a, 31.0), ("e_b", b, 30.0)])
    calls, escalation = _escalation_counter(escalated)
    fr.assess_search_stability(
        refined_result=refined, leader_change={"statistically_material": False},
        canonical_paired=canonical, escalation=escalation,
        config=fr.StabilityGateConfig(current_beam=8),
    )
    assert fr.ranked_route_ids(refined) == before


def test_no_best_h1_fallback_and_confidence_never_changes_the_preferred_route():
    """17-W / 17-X / 17-Y."""

    source = (REPO_ROOT / "scripts" / "run_four_gw_decision.py").read_text(encoding="utf-8")
    assert "DECISION_SEARCH_NOT_STABLE" in source
    assert "best_h1" not in source and "BEST_H1" not in source
    # the suppression CALLS are applied before the artifact is written (rindex:
    # the helper's own definition must not satisfy this)
    write_index = source.index('(out_dir / "four_gw_decision.json").write_text(')
    assert source.rindex("_suppress_transfer_recommendation(") < write_index
    assert source.rindex("assess_search_stability(") < write_index
    # confidence is computed after the ranked decision, and the preferred route
    # comes from the decision (confidence never supplies or changes a route)
    assert source.rindex("fg.evaluate_four_gw_decision(") < source.index("classify_decision_confidence(")
    assert dc.CONFIDENCE_STATES  # and the classifier cannot rank
    assert "affects_ranking" in (REPO_ROOT / "fpl_brain" / "decision_confidence.py").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# 17 Z-AD — no-regression invariants
# ---------------------------------------------------------------------------


def test_r4b2a_search_coverage_invariants_remain_true():
    assert ro.DEFAULT_RETENTION_LENSES == (
        ro.TOP_NET_PROXY_LENS, "TOP_H1", "LOW_HITS", "BANK_DELTA", "FT_PRESERVATION",
        ro.PARETO_LENS, ro.PER_EVENT_MARGINAL_LENS,
    )
    assert ro.RETENTION_LENS_ORDER == ro.DEFAULT_RETENTION_LENSES
    assert ro.REQUIRED_PASS_THROUGH == "REQUIRED"
    # no position-specific lens or weight
    for name in ro.RETENTION_LENS_ORDER:
        assert not any(position in name for position in ro.POSITIONS)
    # K_MAX is derived from the beam width alone and bounded
    for beam in (8, 12, 24, 48):
        budgets = ro.retention_budgets(ro.OptimizerConfig(events=EVENTS, beam_width=beam))
        assert budgets["K_MAX"] == 2 * max(4 * beam, 24)
        assert budgets["KD"] == max(4, beam // 2)
    source = (REPO_ROOT / "fpl_brain" / "route_optimizer.py").read_text(encoding="utf-8")
    assert '"global_optimality_claimed": False' in source
    # the refinement module never defines or re-bounds the lens set: it propagates
    # the caller's declared lenses unchanged and never mutates the universe
    fr_source = (REPO_ROOT / "fpl_brain" / "finalist_refinement.py").read_text(encoding="utf-8")
    assert "retention_lenses=base_config.retention_lenses" in fr_source
    for banned in ("DEFAULT_RETENTION_LENSES =", "retention_lenses=(", "universe[", "pool_ids ="):
        assert banned not in fr_source, banned


def test_r4b2c_gates_and_role_evidence_wiring_remain_true():
    fg_source = (REPO_ROOT / "fpl_brain" / "four_gw_decision.py").read_text(encoding="utf-8")
    assert "DECISION_HORIZON_INCOMPLETE" in fg_source
    assert "def classify_fixture_horizon" in fg_source
    assert "NOT_SUPPORTED" in fg_source
    runner = (REPO_ROOT / "scripts" / "run_four_gw_decision.py").read_text(encoding="utf-8")
    # fixture horizon still classified from the certification snapshot and still gates
    assert "fg.classify_fixture_horizon(" in runner
    assert "fixture_horizon_blocking_reasons" in runner
    # transfer-IN role evidence unchanged
    assert 'int(move["in"])' in runner and 'int(move["out"])' in runner
    assert '"captain_id", "vice_captain_id"' in runner
    assert "certified_minutes_run_only" in runner
    # role_evidence rides the new decision-confidence payload
    assert "role_evidence_source" in runner


def test_wildcard_remains_not_supported_quantitatively():
    fr_source = (REPO_ROOT / "fpl_brain" / "finalist_refinement.py").read_text(encoding="utf-8")
    for banned in ("PLAY_WILDCARD", "wildcard", "bench_boost", "triple_captain", "free_hit"):
        assert banned not in fr_source.lower(), banned
    fg_source = (REPO_ROOT / "fpl_brain" / "four_gw_decision.py").read_text(encoding="utf-8")
    assert "NOT_SUPPORTED" in fg_source


def test_projection_runs_remains_212():
    pytest.importorskip("fpl_brain.config")
    try:
        from fpl_brain.config import config_path, load_config

        config = load_config(None)
    except Exception:
        pytest.skip("no project config available in this worktree")
    db_path = config_path(config, "database")
    if not db_path.exists():
        pytest.skip("project database not present")
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        total, max_id = conn.execute("SELECT COUNT(*), MAX(id) FROM projection_runs").fetchone()
    finally:
        conn.close()
    assert total == 212 and max_id == 212


# ---------------------------------------------------------------------------
# Runner wiring (17 W/X) — suppression and the single escalation
# ---------------------------------------------------------------------------


def _runner_module():
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import run_four_gw_decision as runner

    return runner


def test_suppression_helper_clears_decisiveness_but_keeps_the_route_table():
    runner = _runner_module()
    decision = {
        "transfer_recommendation": {
            "status": fg.RECOMMENDATION_AVAILABLE,
            "preferred_route_id": "route_007",
            "ranking": [{"rank": 1, "route_id": "route_007", "four_gw_net_core": 31.0},
                        {"rank": 2, "route_id": "route_003", "four_gw_net_core": 30.9}],
            "eligible_route_count": 2,
        }
    }
    suppressed = runner._suppress_transfer_recommendation(
        decision, reason=fg.DECISION_SEARCH_NOT_STABLE,
        extra={"search_stability_state": fr.SEARCH_NOT_STABLE},
    )
    assert suppressed["status"] == fg.RECOMMENDATION_SUPPRESSED
    assert suppressed["reason"] == fg.DECISION_SEARCH_NOT_STABLE
    assert suppressed["preferred_route_id"] is None, "no route may look recommended"
    assert suppressed["search_stability_state"] == fr.SEARCH_NOT_STABLE
    # the route table and the ranking remain available
    assert len(suppressed["ranking"]) == 2
    assert suppressed["eligible_route_count"] == 2
    assert decision["transfer_recommendation"] is suppressed


def test_escalation_runner_uses_the_next_beam_width_at_the_stage2_draws(monkeypatch):
    runner = _runner_module()
    captured = {}

    def fake_optimize(**kwargs):
        captured.update(kwargs)
        return {"routes": {}, "planning_cutoff": None}

    monkeypatch.setattr(ro, "optimize", fake_optimize)
    a, b = _partial("A", in_id=41), _partial("B", in_id=42)
    base_config = ro.OptimizerConfig(events=EVENTS, search_draws=fr.STAGE1_DRAWS, seed=SEED_SENTINEL,
                                     beam_width=8, exact_evaluation_budget=20)
    stage1 = _result([("r_a", a, 30.0), ("r_b", b, 29.9)])
    stage1["level_survivors"] = ["sentinel-levels"]
    run = runner._escalation_runner(
        universe={"universe": []}, initial_state=_partial("X").state, scenario=None,
        player_meta={}, generation=None, conn=None, base_config=base_config,
        stage1_result=stage1, stage2_draws=fr.STAGE2_DRAWS,
        finalist_partials=[a, b],
    )
    run(12)
    config = captured["config"]
    assert config.beam_width == 12, "the next supported SEARCH budget is a beam width"
    assert config.search_draws == fr.STAGE2_DRAWS, "and it runs at the Stage-2 draw budget"
    assert config.seed == SEED_SENTINEL
    assert config.events == EVENTS
    assert captured["required_routes"] == [a, b]
    assert captured["nested_prior"]["level_survivors"] == ["sentinel-levels"]
    # and the ladder is bounded: there is no way to ask for a second escalation
    assert rs.next_ladder_budget(12) == 24


def _np(matrices):
    """The DECLARED non-production door: these worlds were never loaded from a
    certified generation, and the declaration says who is exercising the interface."""

    from fpl_brain import route_optimizer as ro

    return ro.NonProductionWorlds(
        declaration="test_r4b2b_finalist_stability: synthetic worlds, no certified generation",
        matrices=matrices,
    )


SEED_SENTINEL = 424242


def test_runner_artifact_asserts_the_refinement_and_never_advertises_a_ladder():
    source = (REPO_ROOT / "scripts" / "run_four_gw_decision.py").read_text(encoding="utf-8")
    # Stage 1 draw fidelity is still reported per event
    assert '"draw_fidelity": {str(event): STAGE1_DRAWS for event in decision_events}' in source
    # the higher count is a finalist-only refinement, reported separately
    assert '"finalist_refinement"' in source
    assert '"simulation_fidelity"' in source
    assert "refinement[\"simulation_fidelity\"]" in source
    # the canonical record is published on the result the confidence code reads
    assert 'result["canonical_paired_near_tie"] = canonical' in source
    # suppression CALLS happen before the artifact hits disk (rindex so the
    # helper's definition cannot satisfy the check)
    write_index = source.index('(out_dir / "four_gw_decision.json").write_text(')
    assert source.rindex("_suppress_transfer_recommendation(") < write_index
    assert source.rindex("assess_search_stability(") < write_index
    # no open-ended ladder: exactly one escalation call site
    assert source.count("escalation=_escalation_runner(") == 1
    assert source.count("rs.budget_config(") == 1
    assert source.count("escalation_beam_override") == 0
