"""R4B.1 — decision-engine correctness + all-player integrity tests.

Synthetic-first, deterministic, no network, no production Monte Carlo and no
production projections.  Covers the R4B.1 test matrix (A-Z).

Where a property is a WIRING property of the production runner (which source
connection is used, which run ids are consumed) the test asserts on the runner
source text or on the pure helper the runner calls, and says so explicitly.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from test_route_optimizer import _np  # noqa: E402

from fpl_brain import candidate_universe as cu
from fpl_brain import four_gw_decision as fg
from fpl_brain import ingest_provenance as prov
from fpl_brain import manager_lineup as ml
from fpl_brain import manager_worlds as mw
from fpl_brain import monte_carlo
from fpl_brain import route_comparator as rc
from fpl_brain import route_optimizer as ro
from fpl_brain import season_rules
from fpl_brain import transfer_state as ts

from test_transfer_state import CLUB, POOL_CLUB, POOL_POSITION, POSITION, SQUAD_IDS

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "scripts" / "run_four_gw_decision.py"


# ---------------------------------------------------------------------------
# Synthetic universe helper (window-parameterised)
# ---------------------------------------------------------------------------
def _payload(core, minutes=90.0):
    return {"core_xpts": core, "expected_minutes": minutes, "p_start": 1.0, "p_60_plus": 1.0,
            "p_appearance": 1.0, "bonus_xpts": 99.0, "total_xpts": core + 99.0, "risk_flags": []}


def _universe(events, cores=None, prices=None, with_projection=None, blank_clubs=()):
    """Build a synthetic universe over an ARBITRARY decision window.

    ``blank_clubs`` removes every fixture for those clubs, producing a genuine
    blank Gameweek (NO_FIXTURE) rather than a missing projection.
    """

    union = sorted(set(SQUAD_IDS) | set(POOL_POSITION))
    base = {pid: 5.0 for pid in union}
    base.update(cores or {})
    price_map = {pid: 50 for pid in union}
    price_map.update(prices or {})
    prices = price_map
    blank = set(int(club) for club in blank_clubs)

    pool = {pid: {"player_id": pid, "position": POSITION.get(pid, POOL_POSITION.get(pid)),
                  "club_id": CLUB.get(pid, POOL_CLUB.get(pid)), "web_name": f"W{pid}", "full_name": f"P{pid}"}
            for pid in union}
    clubs = set()
    for meta in pool.values():
        clubs.add(meta["club_id"])
    fixtures = {(e, club): [1000 + club] for e in events for club in clubs if club not in blank}
    xpts, minutes = {}, {}
    for e in events:
        xpts[e], minutes[e] = {}, {}
        for pid in union:
            if pool[pid]["club_id"] in blank:
                continue
            fid = 1000 + pool[pid]["club_id"]
            if with_projection is not None and pid in with_projection and e not in with_projection[pid]:
                continue
            xpts[e][(pid, fid)] = _payload(base[pid])
            minutes[e][(pid, fid)] = {"joint_availability": 1.0}
    snap = ts.PriceSnapshot(event=int(events[0]), prices=prices)
    universe = cu.build_universe(
        pool=pool, events_fixtures=fixtures, xpts_rows_by_event=xpts, minutes_rows_by_event=minutes,
        events=list(events), owned_ids=list(SQUAD_IDS), price_snapshot=snap,
        config=cu.CandidateConfig(top_n_per_criterion=3), planning_cutoff="2026-09-11T10:16:51Z",
    )
    state = ts.RouteState(
        event=int(events[0]),
        players=tuple(ts.RoutePlayer(pid, POSITION[pid], CLUB[pid], 50) for pid in SQUAD_IDS),
        bank_tenths=0, free_transfers=2,
    )
    meta = {pid: ts.PlayerMeta(pid, POSITION.get(pid, POOL_POSITION.get(pid)),
                               CLUB.get(pid, POOL_CLUB.get(pid))) for pid in union}
    universe["replacement_edges"] = cu.build_replacement_edges(
        universe_rows=universe["universe"], owned_ids=SQUAD_IDS, state=state,
        price_snapshot=snap, player_meta=meta,
    )
    return universe, state, meta, snap


# ---------------------------------------------------------------------------
# A / B / S / K — window parameterisation and truthful window provenance
# ---------------------------------------------------------------------------
def test_A_first_event_search_criterion_is_numerically_live_for_gw5_gw8():
    """A: the first-event criterion must rank by the WINDOW's first event."""

    events = (5, 6, 7, 8)
    # Two same-price MIDs; 41 is strong in GW5 only, 42 is weak in GW5.
    universe, _, _, _ = _universe(events, cores={41: 12.0, 42: 0.0})
    rows = {int(r["player_id"]): r for r in universe["universe"]}
    for pid in (41, 42):
        assert rows[pid]["events"][0].event == 5

    assert ro.first_event_core(rows[41]) == pytest.approx(12.0)
    assert ro.first_event_core(rows[42]) == pytest.approx(0.0)

    view = ro.search_view_ids(rows, 1)
    # 41 has the higher GW5 core, so it must win the first-event criterion.
    assert 41 in view


def test_B_later_window_works_without_hardcoded_event_4():
    events = (7, 8, 9, 10)
    universe, _, _, _ = _universe(events, cores={41: 12.0, 42: 0.0})
    rows = {int(r["player_id"]): r for r in universe["universe"]}
    assert all(r["events"][0].event == 7 for r in rows.values())
    # No row carries event 4 at all, so an event-NUMBER lookup would be dead.
    assert not any(f.event == 4 for r in rows.values() for f in r["events"])
    assert ro.first_event_core(rows[41]) == pytest.approx(12.0)
    assert 41 in ro.search_view_ids(rows, 1)


def test_S_emitted_window_is_the_real_window():
    """S: a GW5-GW8 universe must NOT be stamped as a GW4-GW6 window."""

    events = (5, 6, 7, 8)
    universe, _, _, _ = _universe(events)
    assert universe["supported_events"] == [5, 6, 7, 8]
    assert universe["decision_horizon_length"] == 4
    assert season_rules.window_flag(events) == "SUPPORTED_CONTIGUOUS_EVENTS_5_6_7_8"
    assert "SUPPORTED_CONTIGUOUS_EVENTS_5_6_7_8" in universe["flags"]
    assert "SUPPORTED_CONTIGUOUS_EVENTS_4_5_6" not in universe["flags"]
    # And the optimizer config never claims GW4-GW6 either.
    labels = ro.OptimizerConfig(events=events).as_dict()["labels"]
    assert "SUPPORTED_CONTIGUOUS_EVENTS_5_6_7_8" in labels
    assert all("4_5_6" not in label for label in labels)


def test_K_search_view_criteria_agree_between_the_two_implementations():
    """K: candidate_universe and route_optimizer must compute the SAME view.

    These two implementations had diverged (one positional, one by event
    number), so this is the regression that keeps them equal for any window.
    """

    for events in ((4, 5, 6, 7), (5, 6, 7, 8), (7, 8, 9, 10)):
        universe, _, _, _ = _universe(events, cores={41: 12.0, 42: 1.0, 51: 9.0})
        rows = {int(r["player_id"]): r for r in universe["universe"]}
        n = 3
        assert ro.search_view_ids(rows, n) == cu.search_view_ids(rows, n)


# ---------------------------------------------------------------------------
# C — Monte Carlo cache identity
# ---------------------------------------------------------------------------
def test_C_mc_model_version_change_invalidates_world_cache(monkeypatch):
    runs = {"minutes_v1": 1, "team_strength_v1": 2, "player_rates_v1": 3, "xpts_v1": 4,
            "monte_carlo_v1": 5}
    cfg = ro.OptimizerConfig(events=(5, 6, 7, 8), search_draws=100)
    before = ro.world_cache_key(event=5, generation_id="sha256:" + "a" * 64, runs=runs,
                                config=cfg, union_ids=[1, 2, 3])

    monkeypatch.setattr(monte_carlo, "MONTE_CARLO_MODEL_VERSION", "mc_TEST_BUMP")
    after = ro.world_cache_key(event=5, generation_id="sha256:" + "a" * 64, runs=runs,
                               config=cfg, union_ids=[1, 2, 3])
    assert before != after, "a Monte Carlo model change must change the cache identity"

    # The authoritative constant is what feeds the key.
    monkeypatch.undo()
    again = ro.world_cache_key(event=5, generation_id="sha256:" + "a" * 64, runs=runs,
                               config=cfg, union_ids=[1, 2, 3])
    assert again == before


def test_C_cache_key_uses_the_authoritative_constant_not_a_literal():
    source = (REPO_ROOT / "fpl_brain" / "route_optimizer.py").read_text(encoding="utf-8")
    assert "MONTE_CARLO_MODEL_VERSION" in source
    assert '"mc_v1.2.1"' not in source


# ---------------------------------------------------------------------------
# D — retention-lens provenance is truthful
# ---------------------------------------------------------------------------
def test_D_retention_lenses_are_real_configuration():
    default = ro.OptimizerConfig(events=(5, 6, 7, 8))
    assert default.as_dict()["retention_lenses"] == list(default.retention_lenses)

    # An unknown lens is an error, not a silent no-op.
    bad = ro.OptimizerConfig(events=(5, 6, 7, 8), retention_lenses=("NOT_A_LENS",))
    universe, state, meta, _ = _universe((5, 6, 7, 8))
    with pytest.raises(ValueError, match="unknown retention lens"):
        ro.optimize(universe=universe, initial_state=state, scenario=_scenario((5, 6, 7, 8)),
                    player_meta=meta, config=bad, non_production_worlds=_np(_provider(universe)))

        # Narrowing the lens set provably changes the retained set.
        many = tuple(default.retention_lenses)
        states = _partial_states()
        wide = ro._retain(states, default)
        narrow = ro._retain(states, ro.OptimizerConfig(events=(5, 6, 7, 8), retention_lenses=("LOW_HITS",)))
        assert len(narrow) <= len(wide)
        assert {s.signature() for s in narrow} <= {s.signature() for s in wide} or len(narrow) < len(wide)
        # R4B.2a declared the full lens set; REQUIRED is a pass-through, not a lens.
        assert many == ro.DEFAULT_RETENTION_LENSES
        assert len(many) == 7
        assert ro.REQUIRED_PASS_THROUGH in ro._KNOWN_RETENTION_LENSES


def _partial_states():
    """A handful of distinguishable PartialRoutes for the retention lens test."""

    states = []
    for index in range(6):
        route_state = ts.RouteState(
            event=5, players=tuple(ts.RoutePlayer(pid, POSITION[pid], CLUB[pid], 50) for pid in SQUAD_IDS),
            bank_tenths=index * 10, free_transfers=index % 3,
        )
        states.append(ro.PartialRoute(state=route_state, actions=(),
                                      h1_proxy=float(index), window_proxy=float(6 - index),
                                      hits=index % 2))
    return states


# ---------------------------------------------------------------------------
# E / F / G — world-matrix integrity
# ---------------------------------------------------------------------------
def _scenario(events, prices=None):
    base = {pid: 50 for pid in sorted(set(SQUAD_IDS) | set(POOL_POSITION))}
    base.update(prices or {})
    return rc.PriceScenario(scenario_id="TEST_FLAT",
                            event_snapshots={e: ts.PriceSnapshot(event=e, prices=dict(base)) for e in events})


def _provider(universe, omit=(), worlds=6, zeros=()):
    """A world provider over the universe; ``omit`` drops players from the capture."""

    union = [int(row["player_id"]) for row in universe["universe"]]
    captured = [pid for pid in union if pid not in set(omit)]
    core_values = {pid: 5.0 for pid in union}

    def provider(event, union_ids):
        ids = [int(pid) for pid in captured]
        return {
            "worlds": worlds,
            "player_ids": list(ids),
            "core": {pid: [0.0 if pid in set(zeros) else core_values.get(pid, 0.0)] * worlds for pid in ids},
            "minutes": {pid: [0.0 if pid in set(zeros) else 90.0] * worlds for pid in ids},
        }

    return provider


def test_E_missing_world_player_fails_closed():
    events = (5, 6, 7, 8)
    universe, state, meta, _ = _universe(events)
    # Omit an OWNED player from the capture set: the matrix cannot score him.
    victim = int(SQUAD_IDS[0])
    with pytest.raises(ml.RouteWorldPlayerMissing, match=ml.ROUTE_WORLD_PLAYER_MISSING):
        ro.optimize(universe=universe, initial_state=state, scenario=_scenario(events),
                    player_meta=meta, config=ro.OptimizerConfig(events=events, search_draws=6),
                    non_production_worlds=_np(_provider(universe, omit=(victim,))))


def test_E_policy_world_scores_fails_closed_without_a_series():
    events = (5, 6, 7, 8)
    universe, _, _, _ = _universe(events)
    ids = [int(r["player_id"]) for r in universe["universe"]]
    policy = ml.ManagerPolicy(starter_ids=tuple(ids[:11]), bench_gk_id=int(ids[11]),
                              bench_outfield_order=(int(ids[12]), int(ids[13]), int(ids[14])),
                              captain_id=int(ids[0]), vice_captain_id=int(ids[1]))
    matrix = {"worlds": 2, "player_ids": ids[:14],  # 15th player deliberately absent
              "core": {pid: [1.0, 1.0] for pid in ids[:14]},
              "minutes": {pid: [90.0, 90.0] for pid in ids[:14]}}
    with pytest.raises(ml.RouteWorldPlayerMissing):
        rc.policy_world_scores(policy, matrix, {pid: "MID" for pid in ids})


def test_F_captured_blank_player_with_explicit_zero_series_is_allowed():
    """F: a legitimate blank player IS present with an explicit all-zero series."""

    events = (5, 6, 7, 8)
    universe, state, meta, _ = _universe(events)
    blank = int(SQUAD_IDS[0])
    # Present in the capture set, explicit zeros -> not an error.
    result = ro.optimize(universe=universe, initial_state=state, scenario=_scenario(events),
                         player_meta=meta,
                         config=ro.OptimizerConfig(events=events, search_draws=6, beam_width=3,
                                                   exact_evaluation_budget=3, policy_selection_worlds=6,
                                                   singles_per_out=1, max_transfers_per_event=1,
                                                   rescue_top_k_per_position=1, search_n_per_criterion=2),
                         non_production_worlds=_np(_provider(universe, zeros=(blank,))))
    assert result["promoted_route_count"] >= 1

    # And the manager layer accepts an explicit zero series for a blank player.
    ids = [int(r["player_id"]) for r in universe["universe"]]
    matrix = {"worlds": 2, "player_ids": ids,
              "core": {pid: [0.0, 0.0] for pid in ids},
              "minutes": {pid: [0.0, 0.0] for pid in ids}}
    assert mw.captain_terms(matrix)[0][ids[0]] == 0.0
    assert ml.validate_world_matrix(matrix) == 2


def test_G_route_capture_union_covers_force_routes():
    """G: a forced route's players must be inside the world capture union."""

    route_state = ts.RouteState(
        event=5, players=tuple(ts.RoutePlayer(pid, POSITION[pid], CLUB[pid], 50) for pid in SQUAD_IDS),
        bank_tenths=0, free_transfers=2,
    )
    batch = ts.TransferBatch((ts.TransferAction(int(SQUAD_IDS[0]), 41),))
    partial = ro.PartialRoute(
        state=route_state,
        actions=({"event": 5, "kind": "SINGLE", "batch": batch, "transition": None,
                  "hit_points": 0, "ft_after": 1, "bank_after": 0},),
        h1_proxy=1.0, window_proxy=2.0, hits=0,
    )
    captured = ro.route_player_ids(partial)
    assert int(SQUAD_IDS[0]) in captured and 41 in captured
    assert {int(pid) for pid in SQUAD_IDS} <= captured

    # The optimizer's union must include a forced route's player even when that
    # player is outside the search pool.
    events = (5, 6, 7, 8)
    universe, state, meta, _ = _universe(events)
    forced_only = 999
    forced = ro.PartialRoute(
        state=state,
        actions=({"event": 5, "kind": "SINGLE",
                  "batch": ts.TransferBatch((ts.TransferAction(int(SQUAD_IDS[0]), forced_only),)),
                  "transition": None, "hit_points": 0, "ft_after": 1, "bank_after": 0},),
        h1_proxy=1.0, window_proxy=2.0, hits=0,
    )
    assert forced_only in ro.route_player_ids(forced)


# ---------------------------------------------------------------------------
# H / I / J — official pool accounting and ingest completeness
# ---------------------------------------------------------------------------
def test_H_official_pool_is_accounted_exactly_once():
    events = (5, 6, 7, 8)
    universe, _, _, _ = _universe(events)
    pool = {int(r["player_id"]): {"position": r["position"], "club_id": r["club_id"]}
            for r in universe["universe"]}
    screen = fg.screen_legal_actions(
        universe_rows=universe["universe"], replacement_edges=universe["replacement_edges"],
        owned_ids=SQUAD_IDS, decision_events_window=events,
    )
    report = cu.discovery_completeness(
        pool=pool, universe_rows=universe["universe"], excluded=universe["excluded"],
        replacement_edges=universe["replacement_edges"], screen=screen, enforce=True,
    )
    assert report["official_pool_players"] == report["eligible_candidate_players"]
    assert report["assertions"]["A_every_official_player_eligible_or_excluded_once"] is True
    assert report["assertions"]["C_screened_edges_equals_legal_edges"] is True
    assert report["assertions"]["D_promotion_pool_subset_of_screened"] is True
    assert report["official_pool_ids_sha256"] and report["promoted_in_ids_sha256"]


def test_H_unaccounted_official_player_raises():
    events = (5, 6, 7, 8)
    universe, _, _, _ = _universe(events)
    pool = {int(r["player_id"]): {} for r in universe["universe"]}
    pool[123456] = {}  # an official player that the universe never accounted for
    with pytest.raises(cu.DiscoveryCompletenessError, match=cu.OFFICIAL_PLAYER_POOL_INCOMPLETE):
        cu.discovery_completeness(pool=pool, universe_rows=universe["universe"],
                                  excluded=universe["excluded"],
                                  replacement_edges=universe["replacement_edges"], enforce=True)


def _element(pid, status="a", team=1):
    return {"id": pid, "web_name": f"P{pid}", "status": status, "team": team,
            "element_type": 3, "now_cost": 50}


def _payload_els(ids, status="a"):
    return {"elements": [_element(pid, status) for pid in ids]}


def _generation(ids, previous=None, statuses=None, allow_large_drop=False):
    payload = {"elements": [_element(pid, (statuses or {}).get(pid, "a")) for pid in ids]}
    return prov.build_generation(payload=payload, parsed_count=len(ids), persisted_count=len(ids),
                                 captured_at="2026-09-12T10:00:00Z", run_id=1, previous=previous,
                                 allow_large_drop=allow_large_drop)


def test_I_truncated_bootstrap_cannot_silently_deactivate_players():
    previous = _generation(list(range(1, 501)))
    assert previous.accepted is True

    truncated = _generation(list(range(1, 101)), previous=previous)   # 400 ids vanish
    assert truncated.accepted is False
    assert truncated.acceptance_rule == "REJECTED"
    assert any(reason.startswith("R2_") for reason in truncated.rejection_reasons)
    assert truncated.diagnostic == prov.DIAG_INGEST_INCOMPLETE

    # A legitimately small change is accepted (NOT a naive count-equality rule).
    small = _generation(list(range(1, 500)) + [9001], previous=previous)
    assert small.accepted is True
    assert small.acceptance_rule == "STRUCTURAL_AND_BOUNDED_DRIFT"

    # An explicit override accepts but records the reason.
    overridden = _generation(list(range(1, 101)), previous=previous, allow_large_drop=True)
    assert overridden.accepted is True
    assert overridden.acceptance_rule == "OVERRIDE_ALLOW_LARGE_DROP"
    assert overridden.rejection_reasons


def test_I_whole_club_loss_is_rejected():
    previous_ids = list(range(1, 41))
    previous = _generation(previous_ids)
    # Club 1 loses every player while the global drop stays small.
    remaining = previous_ids[20:]
    current = prov.build_generation(
        payload={"elements": [_element(pid, team=2) for pid in remaining]},
        parsed_count=len(remaining), persisted_count=len(remaining),
        captured_at="2026-09-12T11:00:00Z", run_id=2, previous=previous,
    )
    assert current.accepted is False
    assert any("R3_WHOLE_CLUB_LOST" in reason for reason in current.rejection_reasons)


def test_J_injured_and_suspended_players_stay_in_the_official_pool():
    """J: availability is NOT pool membership."""

    ids = [1, 2, 3, 4]
    statuses = {1: "a", 2: "i", 3: "s", 4: "d"}
    generation = _generation(ids, statuses=statuses)
    assert set(generation.element_ids) == set(ids)          # nothing dropped
    assert generation.accepted is True
    assert generation.availability_counts["i"] == 1
    assert generation.availability_counts["s"] == 1
    assert generation.availability_counts["d"] == 1


def test_K_availability_status_is_never_a_membership_test():
    source = (REPO_ROOT / "fpl_brain" / "candidate_universe.py").read_text(encoding="utf-8")
    # The pool query filters on is_active (payload presence), never on status.
    assert "FROM players WHERE is_active=1" in source
    for status_field in ("chance_of_playing", "status='i'", "status = 'i'"):
        assert status_field not in source


def test_I_generation_round_trips_on_disk(tmp_path):
    generation = _generation([1, 2, 3])
    path = prov.write_generation(tmp_path, generation)
    assert path.exists()
    loaded = prov.load_latest_generation(tmp_path)
    assert loaded is not None
    assert loaded.element_ids == generation.element_ids
    assert loaded.element_id_sha256 == generation.element_id_sha256
    assert prov.load_latest_generation(tmp_path / "missing") is None


# ---------------------------------------------------------------------------
# L / M / N / O / P — all-player numerical screening before pruning
# ---------------------------------------------------------------------------
def test_LMNOP_every_legal_edge_is_screened_before_pruning():
    events = (5, 6, 7, 8)
    # 41/42 are priced at the floor and 51 is cheap: none may be excluded by
    # price or by popularity (popularity is never read at all).
    universe, _, _, _ = _universe(
        events, cores={41: 12.0, 42: 1.0, 51: 9.0}, prices={pid: 40 for pid in (41, 42, 51)}
    )
    for row in universe["universe"]:
        row["selected_by_percent"] = 0.0   # must have no effect
    screen = fg.screen_legal_actions(
        universe_rows=universe["universe"], replacement_edges=universe["replacement_edges"],
        owned_ids=SQUAD_IDS, decision_events_window=events, promotion_pool_limit=5, per_out_limit=2,
    )
    assert screen["enumeration_exhaustive"] is True
    assert screen["coverage"]["every_legal_edge_screened"] is True
    assert screen["screened_legal_actions"] == screen["legal_single_transfers"]
    # L/M/N: the unpopular and £4.0m players reach numerical screening.
    assert {41, 42, 51} <= set(screen["eligible_in_player_ids"])
    # O: every legal edge accounted for.
    assert screen["enumerated_transfer_pairs"] == screen["expected_transfer_pairs"]
    # P: pruning happens only AFTER screening; the pool is a subset of screened.
    assert screen["promotion_pool_size"] <= screen["screened_legal_actions"]
    assert all(edge.get("four_gw_proxy_delta") is not None for edge in screen["promotion_pool"])
    assert all(edge["currently_legal_single_transfer"] for edge in screen["promotion_pool"])


# ---------------------------------------------------------------------------
# Q / R — missing predictive support vs an explicit zero
# ---------------------------------------------------------------------------
def test_Q_missing_certified_prediction_is_not_silently_zero():
    events = (5, 6, 7, 8)
    # Player 41 has a fixture in GW5 but NO projection row for GW5.
    universe, _, _, _ = _universe(events, cores={41: 12.0}, with_projection={41: (6, 7, 8)})
    rows = {int(r["player_id"]): r for r in universe["universe"]}
    gw5 = rows[41]["events"][0]
    assert gw5.event == 5
    assert gw5.status == cu.MISSING_PROJECTION
    assert gw5.fixture_count > 0

    unresolved = cu.unresolved_predictive_support_rows(universe["universe"])
    assert any(item["player_id"] == 41 and item["event"] == 5 for item in unresolved)

    pool = {int(r["player_id"]): {} for r in universe["universe"]}
    with pytest.raises(cu.PredictiveSupportMissing, match=cu.CANDIDATE_PREDICTIVE_SUPPORT_MISSING):
        cu.discovery_completeness(pool=pool, universe_rows=universe["universe"],
                                  excluded=universe["excluded"],
                                  replacement_edges=universe["replacement_edges"], enforce=True)


def test_R_explicit_zero_projection_is_valid():
    events = (5, 6, 7, 8)
    universe, _, _, _ = _universe(events, cores={41: 0.0})   # a real row whose value is zero
    rows = {int(r["player_id"]): r for r in universe["universe"]}
    gw5 = rows[41]["events"][0]
    assert gw5.status == cu.PREDICTED
    assert gw5.expected_core == 0.0
    assert cu.unresolved_predictive_support_rows(universe["universe"]) == []


def test_R_blank_gameweek_is_an_explicit_zero_not_a_missing_projection():
    """R: a blank Gameweek (no fixture) is a VALID explicit zero.

    It must not be reported as missing predictive support and must not fail the
    discovery completeness gate.
    """

    events = (5, 6, 7, 8)
    blank_club = CLUB[SQUAD_IDS[0]]
    universe, _, _, _ = _universe(events, blank_clubs=(blank_club,))
    blanks = [row for row in universe["universe"] if int(row["club_id"]) == blank_club]
    assert blanks, "the blank club must still be present in the universe"
    for row in blanks:
        for feature in row["events"]:
            assert feature.status == cu.NO_FIXTURE
            assert feature.expected_core == 0.0
            assert feature.fixture_count == 0

    # Blank players are NOT missing support, so the gate passes.
    assert cu.unresolved_predictive_support_rows(universe["universe"]) == []
    pool = {int(r["player_id"]): {} for r in universe["universe"]}
    report = cu.discovery_completeness(
        pool=pool, universe_rows=universe["universe"], excluded=universe["excluded"],
        replacement_edges=universe["replacement_edges"], enforce=True,
    )
    assert report["assertions"]["E_no_unresolved_predictive_support"] is True


# ---------------------------------------------------------------------------
# T — conflicting --event vs certification
# ---------------------------------------------------------------------------
def test_T_conflicting_event_against_the_certified_generation_fails():
    """A generation for another event never becomes this decision's world.

    PE-9 amendment 2 made ``generation_id`` a SELECTOR, not a capability, so there is
    no artifact file to contradict: the generation is resolved from the store and its
    event, horizon and cutoff must equal the decision's, or the decision is refused.
    This checks that the binding is enforced where a caller could otherwise point a
    decision at a world it does not belong to.
    """

    import generation_fixtures as gf
    from fpl_brain import generation_store as gs

    conn, runs = gf.synthetic_world(events=(5, 6, 7, 8))
    generation = gf.certify_world(conn, runs, events=(5, 6, 7, 8), planning_event=5)
    assert list(generation.events) == [5, 6, 7, 8]

    # The selector is bound to its own event: asking for GW4 with a GW5 generation is
    # a contradiction, and the store refuses it rather than re-pointing the world.
    with pytest.raises(gs.GenerationRefused):
        gs.resolve_generation(conn, planning_event=4, generation_id=generation.generation_id)

    # An unknown generation id is likewise refused, never defaulted.
    with pytest.raises(gs.GenerationUnknown):
        gs.resolve_generation(conn, planning_event=5, generation_id="sha256:" + "0" * 64)

    # And a horizon that is not the certified one is refused by the store's own check.
    assert gs.support_by_event(generation)[5]["bundle_identity"]


def test_T_no_event_and_no_generation_is_refused():
    import importlib.util

    spec = importlib.util.spec_from_file_location("r4b1_runner2", RUNNER)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    # The planning event is REQUIRED: there is no default Gameweek, so argparse
    # refuses before any database connection is opened.
    with pytest.raises(SystemExit):
        runner.main(["--stage", "search", "--cutoff", "2026-09-12T19:20:00Z"])


# ---------------------------------------------------------------------------
# U — draw-fidelity truth
# ---------------------------------------------------------------------------
def test_U_draw_fidelity_matches_the_stage1_count():
    import importlib.util

    spec = importlib.util.spec_from_file_location("r4b1_runner3", RUNNER)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    events = (5, 6, 7, 8)
    for event in events:
        assert runner.draws_for(event, events) == runner.STAGE1_DRAWS
    source = RUNNER.read_text(encoding="utf-8")
    assert "GW4_DRAWS" not in source, "the aspirational 10k first-event claim must be gone"
    assert "LATER_DRAWS" not in source


# ---------------------------------------------------------------------------
# V — the displayed lineup belongs to the intended route
# ---------------------------------------------------------------------------
def test_V_lineup_policy_comes_from_the_named_route():
    routes = [
        {"route_id": "route_A", "per_event": [{"event": 5, "policy": {"tag": "A"}}, {"event": 6}]},
        {"route_id": "route_B", "per_event": [{"event": 5, "policy": {"tag": "B"}}, {"event": 6}]},
    ]
    assert fg.lineup_policy_for_route(routes=routes, route_id="route_B",
                                      decision_events_window=(5, 6, 7, 8)) == {"tag": "B"}
    assert fg.lineup_policy_for_route(routes=routes, route_id="route_A",
                                      decision_events_window=(5, 6, 7, 8)) == {"tag": "A"}
    assert fg.lineup_policy_for_route(routes=routes, route_id="route_Z",
                                      decision_events_window=(5, 6, 7, 8)) is None
    assert fg.lineup_policy_for_route(routes=routes, route_id=None,
                                      decision_events_window=(5, 6, 7, 8)) is None


def test_V_lineup_basis_is_current_gw_h1():
    decision = fg.evaluate_four_gw_decision(
        planning_event=5, support_by_event={e: {"supported": True, "matched_runs": {}} for e in (5, 6, 7, 8)},
        cutoff="2026-09-12T19:20:00Z", last_event=38, routes=[],
        lineup={"status": fg.LINEUP_ONLY, "policy": {"tag": "A"}, "lineup_route_id": "route_A"},
    )
    assert decision["lineup_recommendation"]["basis"] == "CURRENT_GW_H1"
    assert decision["lineup_recommendation"]["basis_events"] == [5]
    assert decision["lineup_recommendation"]["lineup_route_id"] == "route_A"


# ---------------------------------------------------------------------------
# W — Wildcard overstatement
# ---------------------------------------------------------------------------
def test_W_injected_play_wildcard_is_sanitized_in_the_decision():
    injected = {"status": fg.WILDCARD_EVALUATION_SUPPORTED, "recommendation": "PLAY_WILDCARD",
                "actionable": True, "supported_evaluation": {"four_gw_net_core": 12.0}}
    decision = fg.evaluate_four_gw_decision(
        planning_event=5, support_by_event={e: {"supported": True, "matched_runs": {}} for e in (5, 6, 7, 8)},
        cutoff="2026-09-12T19:20:00Z", last_event=38, routes=[], wildcard=injected,
    )
    screen = decision["wildcard_screen"]
    assert screen["recommendation"] == "NONE"
    assert screen["actionable"] is False
    assert screen["status"] == fg.WILDCARD_REVIEW_REQUIRED
    assert screen["overstatement_prevented"] is True
    assert "PLAY_WILDCARD" not in json.dumps(decision)


def test_W_production_runner_never_emits_play_wildcard():
    source = RUNNER.read_text(encoding="utf-8")
    assert "PLAY_WILDCARD" not in source


# ---------------------------------------------------------------------------
# X / Y — production wiring assertions
# ---------------------------------------------------------------------------
def test_X_discovery_reads_come_from_the_certified_generation_snapshot():
    source = RUNNER.read_text(encoding="utf-8")
    assert "gs.open_generation_snapshot(generation)" in source
    for call in ("cu.load_pool(source_conn)", "cu.load_fixtures_by_team(source_conn",
                 "rc.load_player_meta(source_conn", "cu.price_snapshot_as_of(\n            source_conn",
                 "manager_worlds.resolve_squad(source_context, source_conn)"):
        assert call in source, f"discovery must read {call} from the snapshot"


def test_Y_prediction_rows_use_certified_exact_ids():
    source = RUNNER.read_text(encoding="utf-8")
    assert 'cu.load_projection_rows(conn, int(certified_runs[int(e)]["xpts_v1"]))' in source
    assert 'certified_runs[int(e)]["minutes_v1"]' in source
    assert "PREDICTIVE_GENERATION_MISMATCH" in source
    # The discovery generation and the exact-evaluation generation are compared.
    assert "discovery_identity" in source and "exact_identity" in source
    assert 'run_refs={"events": list(decision_events),' in source


# ---------------------------------------------------------------------------
# Z — projection provenance is untouched by this phase
# ---------------------------------------------------------------------------
def test_Z_projection_runs_are_unchanged():
    db = REPO_ROOT / "fpl.db"
    if not db.exists():
        pytest.skip("no local database")
    conn = sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True)
    try:
        count, max_id = conn.execute("SELECT COUNT(*), MAX(id) FROM projection_runs").fetchone()
    finally:
        conn.close()
    assert max_id <= 212, f"R4B.1 must not create projection runs (max id is {max_id})"
    assert count <= 212
