"""P2 — exact-equivalence tests for the performance changes.

The fast path does not ship because it is faster; it ships because it is provably
identical.  Every test here asserts EXACT equality (``==``), never a tolerance.

Covered:
  * ``manager_worlds.base_skeleton_stats`` vs the verbatim reference on synthetic
    worlds covering the phase's edge-case list, and on deterministic stochastic
    worlds;
  * the full ``manager_lineup.rank_policies`` ranking under either implementation;
  * the matrix-keyed memo (identical values, hit accounting);
  * the price-snapshot identity (unchanged string, stamped and unstamped);
  * the tagged player-meta mapping;
  * the shared run-scoped exact cache (identical routes, and hits only on an
    identical evaluation identity);
  * the search hot-path primitives (``player_event_core``, ``_feature_core``).
"""

from __future__ import annotations

import random

import pytest

from test_route_optimizer import _np  # noqa: E402

from fpl_brain import manager_lineup as ml
from fpl_brain import manager_worlds as mw
from fpl_brain import route_comparator as rc
from fpl_brain import route_optimizer as ro
from fpl_brain import transfer_state as ts
from test_transfer_state import POSITION, SQUAD_IDS

POSITIONS = {pid: POSITION[pid] for pid in SQUAD_IDS}
STAT_FIELDS = ("mean_core_base", "p_any_autosub", "expected_autosub_points_added",
               "expected_number_of_autosubs", "bench_gk_used_probability",
               "bench_slot_1_used_probability", "bench_slot_2_used_probability",
               "bench_slot_3_used_probability")


def matrix_from_worlds(worlds: list[dict[int, tuple[float, float]]],
                       player_ids: list[int] | None = None) -> dict:
    """One dict per world mapping player_id -> (minutes, core)."""

    ids = list(player_ids or SQUAD_IDS)
    core = {pid: [float(row.get(pid, (0.0, 0.0))[1]) for row in worlds] for pid in ids}
    minutes = {pid: [float(row.get(pid, (0.0, 0.0))[0]) for row in worlds] for pid in ids}
    return {"worlds": len(worlds), "player_ids": ids, "core": core, "minutes": minutes}


def base_rows(count: int, **overrides):
    row = {pid: (72.0, 3.0 + 0.25 * pid) for pid in SQUAD_IDS}
    row.update(overrides)
    return [dict(row) for _ in range(count)]


# ---------------------------------------------------------------------------
# §9 synthetic edge cases
# ---------------------------------------------------------------------------

def _absent(rows, world, *pids):
    for pid in pids:
        rows[world][pid] = (0.0, 0.0)
    return rows


EDGE_CASES = {
    "all_players_appear": lambda: base_rows(8),
    "one_starter_absent": lambda: _absent(base_rows(8), 0, 11),
    "multiple_starters_absent": lambda: _absent(base_rows(8), 0, 11, 12, 21),
    "goalkeeper_absent": lambda: _absent(base_rows(8), 0, 1),
    "both_goalkeepers_absent": lambda: _absent(base_rows(8), 0, 1, 2),
    "multiple_bench_substitutions": lambda: _absent(base_rows(8), 0, 11, 12, 13),
    "zero_minute_player": lambda: _absent(base_rows(8), 0, 33),
    "all_players_absent": lambda: [{pid: (0.0, 0.0) for pid in SQUAD_IDS} for _ in range(4)],
    "five_defenders_absent": lambda: _absent(base_rows(6), 2, 11, 12, 13, 14, 15),
    "duplicate_appearance_patterns": lambda: (
        base_rows(3) + _absent(base_rows(3), 1, 11) + _absent(base_rows(3), 0, 11, 31)),
    "defender_heavy_bench": lambda: _absent(base_rows(6), 1, 11, 12),
}


@pytest.mark.parametrize("name", sorted(EDGE_CASES))
def test_base_skeleton_stats_matches_reference_on_edge_cases(name):
    matrix = matrix_from_worlds(EDGE_CASES[name]())
    skeletons = list(ml.enumerate_skeletons(SQUAD_IDS, POSITIONS))
    reference = mw.base_skeleton_stats_reference(skeletons, POSITIONS, matrix)
    new = mw.base_skeleton_stats(skeletons, POSITIONS, matrix)
    assert len(reference) == len(new)
    for ref_row, new_row in zip(reference, new):
        for field in STAT_FIELDS:
            assert ref_row[field] == new_row[field], (name, field, ref_row[field], new_row[field])


def test_base_skeleton_stats_matches_reference_on_stochastic_worlds():
    rng = random.Random(20260913)
    rows = []
    for _ in range(240):
        row = {}
        for pid in SQUAD_IDS:
            if rng.random() < 0.18:
                row[pid] = (0.0, 0.0)
            else:
                row[pid] = (float(rng.choice([0.0, 23.0, 61.0, 90.0])),
                            round(rng.uniform(-1.0, 14.0), 3))
        rows.append(row)
    matrix = matrix_from_worlds(rows)
    skeletons = list(ml.enumerate_skeletons(SQUAD_IDS, POSITIONS))
    reference = mw.base_skeleton_stats_reference(skeletons, POSITIONS, matrix)
    new = mw.base_skeleton_stats(skeletons, POSITIONS, matrix)
    for ref_row, new_row in zip(reference, new):
        for field in STAT_FIELDS:
            assert ref_row[field] == new_row[field], (field, ref_row[field], new_row[field])


def test_rank_policies_ranking_identical_under_either_implementation():
    matrix = matrix_from_worlds(EDGE_CASES["duplicate_appearance_patterns"]())
    skeletons = list(ml.enumerate_skeletons(SQUAD_IDS, POSITIONS))
    reference = mw.base_skeleton_stats_reference(skeletons, POSITIONS, matrix)
    new = mw.base_skeleton_stats(skeletons, POSITIONS, matrix)
    ref_rank = ml.rank_policies(SQUAD_IDS, POSITIONS, matrix, top_k=5, skeleton_stats=reference)
    new_rank = ml.rank_policies(SQUAD_IDS, POSITIONS, matrix, top_k=5, skeleton_stats=new)
    assert ref_rank["evaluated_policies"] == new_rank["evaluated_policies"]
    assert ref_rank["top_mean_by_key"] == new_rank["top_mean_by_key"]
    assert ([p.ordering_key() for p in ref_rank["top_policies"]]
            == [p.ordering_key() for p in new_rank["top_policies"]])


def test_rank_policies_selects_the_same_policy_through_the_fast_path():
    """No ``skeleton_stats`` injection: the production path is compared end to end."""

    matrix = matrix_from_worlds(EDGE_CASES["goalkeeper_absent"]())
    direct = ml.rank_policies(SQUAD_IDS, POSITIONS, matrix, top_k=3)
    skeletons = list(ml.enumerate_skeletons(SQUAD_IDS, POSITIONS))
    injected = ml.rank_policies(
        SQUAD_IDS, POSITIONS, matrix, top_k=3,
        skeleton_stats=mw.base_skeleton_stats_reference(skeletons, POSITIONS, matrix))
    assert ([p.ordering_key() for p in direct["top_policies"]]
            == [p.ordering_key() for p in injected["top_policies"]])


# ---------------------------------------------------------------------------
# §4 matrix-keyed memo
# ---------------------------------------------------------------------------

def _stamped(matrix: dict, identity: str) -> dict:
    matrix[mw.MATRIX_IDENTITY_KEY] = identity
    return matrix


def test_captain_terms_memo_returns_identical_terms_and_counts_hits():
    mw.clear_matrix_memos()
    mw.reset_matrix_memo_stats()
    matrix = _stamped(matrix_from_worlds(base_rows(30)), "test:captain_terms")
    first_a, first_c = mw.captain_terms(matrix)
    second_a, second_c = mw.captain_terms(matrix)
    assert first_a == second_a
    assert first_c == second_c
    # The outer dicts are copies, so a caller cannot corrupt the cached keying.
    second_a.clear()
    third_a, _ = mw.captain_terms(matrix)
    assert third_a == first_a
    stats = mw.matrix_memo_stats()
    assert stats["captain_terms_calls"] == 3
    assert stats["captain_terms_computes"] == 1
    assert stats["captain_terms_hits"] == 2


def test_captain_terms_is_not_memoised_across_different_matrix_identities():
    mw.clear_matrix_memos()
    mw.reset_matrix_memo_stats()
    first = _stamped(matrix_from_worlds(base_rows(20)), "test:identity-A")
    second = _stamped(matrix_from_worlds(base_rows(20)), "test:identity-B")
    second["core"][SQUAD_IDS[0]] = [99.0] * 20
    mw.captain_terms(first)
    mw.captain_terms(second)
    assert mw.matrix_memo_stats()["captain_terms_computes"] == 2


def test_unstamped_matrix_is_never_memoised():
    mw.clear_matrix_memos()
    mw.reset_matrix_memo_stats()
    matrix = matrix_from_worlds(base_rows(12))
    assert mw.matrix_identity(matrix) is None
    mw.captain_terms(matrix)
    mw.captain_terms(matrix)
    assert mw.matrix_memo_stats()["captain_terms_computes"] == 2
    assert mw.matrix_memo_stats()["captain_terms_hits"] == 0


def test_appearance_groups_memo_is_used_and_identical():
    mw.clear_matrix_memos()
    mw.reset_matrix_memo_stats()
    matrix = _stamped(matrix_from_worlds(base_rows(40)), "test:groups")
    first = mw._appearance_groups(matrix)
    second = mw._appearance_groups(matrix)
    assert first is second
    assert mw.matrix_memo_stats()["appearance_groups_computes"] == 1


def test_subsample_gets_a_distinct_provenance_identity():
    matrix = _stamped(matrix_from_worlds(base_rows(20)), "test:parent")
    sliced = ro._subsample(matrix, 5)
    assert int(sliced["worlds"]) == 5
    assert sliced[ro.MANAGER_MATRIX_IDENTITY_KEY] != matrix[ro.MANAGER_MATRIX_IDENTITY_KEY]
    assert sliced[ro.MANAGER_MATRIX_IDENTITY_KEY].startswith("test:parent|subsample|")
    # A no-op subsample is the same world set, so it must keep the parent identity.
    assert ro._subsample(matrix, 0) is matrix


# ---------------------------------------------------------------------------
# §5 search wins
# ---------------------------------------------------------------------------

def test_price_snapshot_identity_string_is_unchanged_by_stamping():
    prices = {pid: 50 + pid for pid in range(1, 658)}
    unstamped = ts.PriceSnapshot(event=6, prices=dict(prices))
    stamped = ts.PriceSnapshot(event=6, prices=dict(prices),
                               snapshot_id=ts.price_snapshot_identity(6, dict(prices)))
    assert unstamped.identity() == stamped.identity()
    assert stamped.identity() == stamped.snapshot_id
    assert ts.price_snapshot_identity(6, dict(prices)) == unstamped.identity()


def test_flat_current_price_scenario_stamps_the_canonical_identity():
    prices = {pid: 45 + pid for pid in range(1, 300)}
    base = ts.PriceSnapshot(event=5, prices=prices)
    scenario = rc.flat_current_price_scenario(base, (5, 6))
    for event in (5, 6):
        snapshot = scenario.snapshot_for(event)
        assert snapshot.snapshot_id is not None
        assert snapshot.identity() == ts.PriceSnapshot(event=event, prices=prices).identity()


def test_normalise_meta_is_content_identical_and_tagged():
    meta = {pid: ts.PlayerMeta(pid, POSITION[pid], pid) for pid in SQUAD_IDS}
    from typing import Mapping as TypingMapping
    plain = {int(pid): value for pid, value in meta.items()}
    normalised = ts.normalise_meta(plain)
    assert isinstance(normalised, ts.NormalisedMeta)
    assert dict(normalised) == plain
    assert isinstance(normalised, dict)
    assert isinstance(normalised, TypingMapping)
    # Re-normalising the tagged mapping is a no-op and gives the same object back.
    assert ts.normalise_meta(normalised) is normalised
    # The iterable form still works.
    assert dict(ts.normalise_meta(list(plain.values()))) == plain


def test_normalise_meta_accepts_a_mapping_of_plain_dicts():
    meta = {pid: {"position": POSITION[pid], "club_id": pid} for pid in SQUAD_IDS}
    normalised = ts.normalise_meta(meta)
    assert set(normalised) == set(SQUAD_IDS)
    assert all(isinstance(value, ts.PlayerMeta) for value in normalised.values())


def test_apply_transfer_batch_is_unchanged_by_the_tagged_mapping():
    state = ts.RouteState(
        event=4,
        players=tuple(ts.RoutePlayer(pid, POSITION[pid], pid, 50) for pid in SQUAD_IDS),
        bank_tenths=10, free_transfers=2)
    prices = {pid: 50 for pid in list(SQUAD_IDS) + [41, 51, 61]}
    snapshot = ts.PriceSnapshot(event=4, prices=prices,
                                snapshot_id=ts.price_snapshot_identity(4, prices))
    plain = {pid: ts.PlayerMeta(pid, POSITION.get(pid, "DEF"), pid) for pid in prices}
    tagged = ts.normalise_meta(plain)
    batch = ts.TransferBatch((ts.TransferAction(SQUAD_IDS[0], 41),))
    from_plain = ts.apply_transfer_batch(state, batch, snapshot, plain)
    from_tagged = ts.apply_transfer_batch(state, batch, snapshot, tagged)
    assert from_plain.ok == from_tagged.ok
    assert from_plain.hit_points == from_tagged.hit_points
    assert from_plain.next_event_state == from_tagged.next_event_state
    assert from_plain.price_snapshot_id == from_tagged.price_snapshot_id


def test_player_event_core_matches_a_first_match_scan():
    row = {"events": [{"event": 5, "expected_core": 4.5}, {"event": 6, "expected_core": 7.25},
                      {"event": 7, "expected_core": 0.0}]}

    def naive(target):
        for feature in row["events"]:
            if int(feature["event"]) == int(target):
                return float(feature.get("expected_core") or 0.0)
        return 0.0

    for event in (5, 6, 7, 8, 99):
        assert ro.player_event_core(row, event) == naive(event)
    # The index is cached on the row and a second call must agree.
    assert ro.player_event_core(row, 6) == 7.25


def test_player_event_core_keeps_first_match_when_events_are_duplicated():
    row = {"events": [{"event": 5, "expected_core": 1.0}, {"event": 5, "expected_core": 2.0}]}
    assert ro.player_event_core(row, 5) == 1.0


def test_feature_core_agrees_for_mapping_and_object_features():
    class Feature:
        def __init__(self, core):
            self.expected_core = core

    assert ro._feature_core({"expected_core": 3.5}) == 3.5
    assert ro._feature_core(Feature(3.5)) == 3.5
    assert ro._feature_core({"expected_core": None}) == 0.0


def test_window_proxy_is_unchanged_and_order_preserving():
    rows = {pid: {"events": [{"event": 4, "expected_core": 1.0 * pid},
                             {"event": 5, "expected_core": 2.0 * pid}]} for pid in SQUAD_IDS}
    squad = list(SQUAD_IDS)
    assert ro.window_proxy(rows, squad, [4, 5]) == sum(
        1.0 * pid + 2.0 * pid for pid in squad)
    assert ro.window_proxy(rows, squad, [4]) == sum(1.0 * pid for pid in squad)


# ---------------------------------------------------------------------------
# §3 shared exact cache
# ---------------------------------------------------------------------------

def _search_fixture():
    from test_route_optimizer import _config, _np, _provider, _scenario, _universe
    universe, state, meta = _universe()
    return universe, state, meta, _scenario(), _config(), _provider()


def test_exact_evaluation_key_covers_the_world_identity():
    config = ro.OptimizerConfig(events=(4, 5), search_draws=1000, seed=7)
    squad = (1, 2, 3)
    same = ro.exact_evaluation_key(4, squad, config, "worlds:AAA")
    assert same == ro.exact_evaluation_key(4, squad, config, "worlds:AAA")
    assert same != ro.exact_evaluation_key(4, squad, config, "worlds:BBB")
    assert same != ro.exact_evaluation_key(5, squad, config, "worlds:AAA")
    assert same != ro.exact_evaluation_key(4, (1, 2, 4), config, "worlds:AAA")
    assert same != ro.exact_evaluation_key(
        4, squad, ro.OptimizerConfig(events=(4, 5), search_draws=2000, seed=7), "worlds:AAA")
    assert same != ro.exact_evaluation_key(
        4, squad, ro.OptimizerConfig(events=(4, 5), search_draws=1000, seed=8), "worlds:AAA")
    # No identity at all is still a valid (weaker) key, and never equals a stamped one.
    assert ro.exact_evaluation_key(4, squad, config, None) == \
        ro.exact_evaluation_key(4, squad, config, None)
    assert ro.exact_evaluation_key(4, squad, config, None) != same


def test_shared_exact_cache_reuses_identical_evaluations_and_keeps_routes_identical():
    universe, state, meta, scenario, config, provider = _search_fixture()
    # Fresh caches: the second call recomputes everything.
    fresh_first = ro.optimize(universe=universe, initial_state=state, scenario=scenario,
                              player_meta=meta, config=config, non_production_worlds=_np(provider),
                              exact_cache={})
    fresh_second = ro.optimize(universe=universe, initial_state=state, scenario=scenario,
                               player_meta=meta, config=config, non_production_worlds=_np(provider),
                               exact_cache={})
    # One shared cache: the second call must hit.
    shared: dict = {}
    shared_first = ro.optimize(universe=universe, initial_state=state, scenario=scenario,
                               player_meta=meta, config=config, non_production_worlds=_np(provider),
                               exact_cache=shared)
    shared_second = ro.optimize(universe=universe, initial_state=state, scenario=scenario,
                                player_meta=meta, config=config, non_production_worlds=_np(provider),
                                exact_cache=shared)

    assert fresh_first["routes"] == shared_first["routes"]
    assert fresh_second["routes"] == shared_second["routes"]
    # The shared second call must have re-used every evaluation it needed: one hit for
    # every (route, event) exact evaluation.
    evaluations = sum(len(record["actions"]) for record in shared_second["routes"].values())
    assert shared_second["exact_cache_hits"] == evaluations
    # A fresh-cache run still de-duplicates the pairs that two promoted routes share WITHIN
    # the call, but it must never reach the shared run's total.
    assert fresh_second["exact_cache_hits"] < shared_second["exact_cache_hits"]
    # Both runs must evaluate exactly the same distinct set of (event, squad) pairs.
    assert fresh_second["exact_cache_entries"] == shared_second["exact_cache_entries"]
    assert fresh_second["exact_evaluations"] == shared_second["exact_evaluations"]
    # And both must agree on the ranking and the frontier.
    assert fresh_second["supported_3gw_frontier"] == shared_second["supported_3gw_frontier"]
    assert (fresh_second["roll_baseline"] or {}) == (shared_second["roll_baseline"] or {})
    for name, record in fresh_second["routes"].items():
        other = shared_second["routes"][name]
        assert record["supported_3gw_net_core"] == other["supported_3gw_net_core"]
        assert record["h1_net_core"] == other["h1_net_core"]
        assert record["event_scores"].keys() == other["event_scores"].keys()
        for event in record["event_scores"]:
            assert record["event_scores"][event] == other["event_scores"][event]


def test_world_identity_prefers_the_stamped_matrix_identity():
    from test_route_optimizer import _config
    universe, state, meta, scenario, config, provider = _search_fixture()
    matrix = {"worlds": 3, "player_ids": [1, 2], "core": {1: [0.0] * 3, 2: [0.0] * 3},
              "minutes": {1: [0.0] * 3, 2: [0.0] * 3}}
    matrix[ro.MANAGER_MATRIX_IDENTITY_KEY] = "stamped-identity"
    identity = ro._worlds_identity({4: matrix}, bundles=None, config=config, union=[1, 2])
    assert identity == {4: "stamped-identity"}


def test_world_identity_falls_back_to_the_certified_cache_key():
    from test_route_optimizer import _config
    bundle = rc.EventBundle(event=4, minutes_run_id=1, team_run_id=2, rate_run_id=3, xpts_run_id=4)
    config = _config()
    matrix = {"worlds": 3, "player_ids": [1, 2], "core": {1: [0.0] * 3, 2: [0.0] * 3},
              "minutes": {1: [0.0] * 3, 2: [0.0] * 3}}
    identity = ro._worlds_identity({4: matrix}, bundles={4: bundle}, config=config, union=[1, 2])
    assert identity == {4: ro.world_cache_key(event=4, bundle=bundle, config=config,
                                             union_ids=[1, 2])}
    # An unidentifiable matrix yields None rather than a guess.
    assert ro._worlds_identity({4: {"worlds": 1, "player_ids": []}}, bundles=None,
                               config=config, union=[]) is None


def test_generate_actions_counts_and_order_are_stable():
    """The de-duplicated delta computation must not change the generated action set."""

    universe, state, meta, scenario, config, provider = _search_fixture()
    rows = {int(row["player_id"]): row for row in universe["universe"]}
    pool = ro.build_search_pool(universe, SQUAD_IDS, config)
    positions = {pid: str(row["position"]) for pid, row in rows.items()}
    snapshot = scenario.snapshot_for(4)
    first = ro.generate_actions(state=state, rows=rows, pool_ids=pool["pool_ids"],
                                positions=positions, price_snapshot=snapshot, player_meta=meta,
                                config=config, event=4)
    second = ro.generate_actions(state=state, rows=rows, pool_ids=pool["pool_ids"],
                                 positions=positions, price_snapshot=snapshot, player_meta=meta,
                                 config=config, event=4)
    assert first["generated"] == second["generated"]
    assert first["counts"] == second["counts"]
    assert [a["kind"] for a in first["actions"]] == [a["kind"] for a in second["actions"]]
    assert ([(a["kind"], a["batch"].out_ids(), a["batch"].in_ids()) for a in first["actions"]]
            == [(a["kind"], a["batch"].out_ids(), a["batch"].in_ids()) for a in second["actions"]])
    # The pre-computed delta must equal a fresh _delta for every SINGLE action (a DOUBLE
    # records the sum of its two singles, which is asserted separately).
    for action in first["actions"]:
        if action["kind"] == "SINGLE":
            move = action["batch"].actions[0]
            expected = ro._delta(rows, int(move.out_player_id), int(move.in_player_id))
            assert action["delta_3gw"] == expected[1]
    for action in first["actions"]:
        expected = 0.0
        for move in action["batch"].actions:
            expected += ro._delta(rows, int(move.out_player_id), int(move.in_player_id))[1]
        assert action["delta_3gw"] == expected, action["kind"]
