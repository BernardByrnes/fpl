"""P0 — the parallel exact FILE transport must carry the FULL semantic matrix.

The exact evaluation consumes two policy-level blocks beyond the raw world series:

* ``expected_bonus``  — the deterministic armband objective; absent, the captain is
  chosen on CORE alone (``manager_lineup.CAPTAIN_VALUE_BASIS_CORE_ONLY``);
* ``role_actionability`` — the legal-policy restriction; absent, a role-conflicted
  goalkeeper is startable again (``manager_lineup.enumerate_skeletons``).

The parent builds both.  A worker that loads a matrix from the certified cache FILE
must therefore receive both, and must never be the place where a missing block is
silently defaulted.  The BLOB transport pickles the whole dict and so never exposed
the loss, which is exactly why the defect survived the P3 and Stage-1 equivalence
gates: those fixtures use ``world_provider``, which has no path stamp and therefore
travels as a blob.

These tests drive the REAL transport functions — ``build_event_worlds`` reading a real
cache file, ``describe_matrix_source``, ``_worker_init`` and ``worker_evaluate`` — with
a fixture in which BOTH blocks are independently load-bearing.
"""

from __future__ import annotations

import json

import pytest

from fpl_brain import manager_lineup as ml
from fpl_brain import manager_worlds as mw
from fpl_brain import parallel_exact as px
from fpl_brain import route_comparator as rc
from fpl_brain import route_optimizer as ro
from test_route_optimizer import _certified_bundle, _config, _scenario, _universe
from test_transfer_state import POSITION, SQUAD_IDS

#: A bundle that DECLARES its certified provenance.  ``build_event_worlds`` refuses a
#: bundle that cannot prove which certified run ids it came from, so the fixture is
#: built the way a producer builds one, naming the model version of every family.
BUNDLE = _certified_bundle(4)
WORLDS = 6

#: Every unlisted player scores the fixture's flat default.
FLAT_CORE = 5.0
#: GK 1 is the STRONGER keeper and is role-unresolved, so the restriction must bench him.
#: MID 25 carries a large deterministic bonus, so the bonus-inclusive armband must make
#: him captain where a CORE-only objective makes FWD 31 captain.
CORE = {1: 10.0, 2: 1.0, 25: 15.0, 31: 20.0}
BONUS = {25: 30.0}
UNRESOLVED = (1,)

#: Independently measured on this fixture, so a later edit cannot quietly neuter it.
EXPECTED_CAPTAIN_WITH_BLOCKS = 25
EXPECTED_CAPTAIN_WITHOUT_BLOCKS = 31
EXPECTED_BENCH_GK_WITH_BLOCKS = 1
EXPECTED_BENCH_GK_WITHOUT_BLOCKS = 2


@pytest.fixture(autouse=True)
def _restore_monte_carlo_guard():
    """``_worker_init`` installs an enforced no-Monte-Carlo guard; undo it after a test.

    That patch is correct in a worker PROCESS, where it is the process's whole life.  These
    tests drive the worker in-process, so leaving it installed would replace
    ``monte_carlo.simulate`` with a ``*args/**kwargs`` stub for every later test in the
    session and break anything that reads its signature.
    """

    from fpl_brain import monte_carlo

    original = monte_carlo.simulate
    try:
        yield
    finally:
        monte_carlo.simulate = original


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def _semantic_matrix(union, *, drop=()):
    """The canonical six-block matrix, or a deliberately incomplete one."""

    blocks = {
        "worlds": WORLDS,
        "player_ids": [int(p) for p in union],
        "core": {int(p): [float(CORE.get(int(p), FLAT_CORE))] * WORLDS for p in union},
        "minutes": {int(p): [90.0] * WORLDS for p in union},
        "expected_bonus": {int(p): float(BONUS.get(int(p), 0.0)) for p in union},
        "role_actionability": {int(p): (int(p) in UNRESOLVED) for p in union},
    }
    return {name: value for name, value in blocks.items() if name not in drop}


def _union_for(universe, state, config):
    """Exactly the union ``optimize`` captures worlds for."""

    pool = ro.build_search_pool(universe, [int(p.player_id) for p in state.players], config)
    return ro.union_player_ids(state, sorted(set(pool["pool_ids"])))


def _write_cache_file(tmp_path, union, config, *, drop=(), event=4):
    key = ro.world_cache_key(event=event, bundle=BUNDLE, config=config, union_ids=union)
    payload = {name: ({str(k): v for k, v in value.items()} if isinstance(value, dict) else value)
               for name, value in _semantic_matrix(union, drop=drop).items()}
    (tmp_path / f"{key}.json").write_text(json.dumps(payload), encoding="utf-8")
    return key


def _parent_matrix(tmp_path, universe, state, config, *, drop=()):
    """The parent-side matrix, produced by the REAL cache reader."""

    union = _union_for(universe, state, config)
    _write_cache_file(tmp_path, union, config, drop=drop)
    matrix, info = ro.build_event_worlds(None, {4: BUNDLE}, 4, union, config, cache_dir=tmp_path)
    assert info["source"] == "cache", info
    return matrix, union


def _worker_result(matrix, squad, config, *, event=4):
    """Run the REAL worker entry point over whichever transport this matrix implies.

    Matrix-keyed memos are dropped first: in production the parent and its workers live in
    separate processes, so an in-process replay must clear them or one arm would serve the
    other's memoised ``captain_terms``/``_appearance_groups`` under the shared identity.
    """

    mw.clear_matrix_memos()
    identity = str(matrix[mw.MATRIX_IDENTITY_KEY])
    kind, payload, digest = px.describe_matrix_source(matrix)
    px._worker_init({"config": config,
                     "positions_by_id": {int(p): POSITION[int(p)] for p in squad},
                     "matrix_sources": {event: (kind, payload)},
                     "matrix_identity": {event: identity},
                     "matrix_cache": 1})
    task = px.ExactTask(key=ro.exact_evaluation_key(event, tuple(squad), config, identity),
                        event=event, squad_ids=tuple(squad), world_identity=identity,
                        matrix_digest=digest)
    return kind, px.worker_evaluate(task)


def _sequential_result(matrix, squad, config):
    """The parent's own in-process evaluation — the same code ``_sequential_task`` runs."""

    mw.clear_matrix_memos()
    squad = tuple(int(p) for p in squad)
    positions = {pid: POSITION[pid] for pid in squad}
    selection = ro._subsample(matrix, int(config.policy_selection_worlds))
    policy = ml.rank_policies(list(squad), positions, selection, top_k=1)["top_policies"][0]
    scores = rc.policy_world_scores(policy, matrix, positions)
    return policy, scores


def _assert_same_decision(left_policy, left_scores, right_policy, right_scores, *, label):
    assert left_policy.starter_ids == right_policy.starter_ids, label
    assert left_policy.bench_gk_id == right_policy.bench_gk_id, label
    assert left_policy.bench_outfield_order == right_policy.bench_outfield_order, label
    assert left_policy.captain_id == right_policy.captain_id, label
    assert left_policy.vice_captain_id == right_policy.vice_captain_id, label
    assert left_scores == right_scores, label
    assert sum(left_scores) / len(left_scores) == sum(right_scores) / len(right_scores), label


# ---------------------------------------------------------------------------
# A — FILE transport must equal BLOB transport and the sequential path
# ---------------------------------------------------------------------------


def test_file_transport_hands_the_worker_the_same_matrix_as_blob(tmp_path):
    """PREDECESSOR KILL: on 04cd1e6 the FILE loader rebuilt only four blocks.

    The same certified content is delivered twice — once as the cache file a production
    run writes, once as the blob a provider-injected matrix produces — and both must
    yield the same decision.  On the predecessor the file arm loses the armband objective
    and the keeper restriction, so captain, vice and bench GK all differ.
    """

    universe, state, meta = _universe()
    config = _config()
    squad = tuple(int(p) for p in SQUAD_IDS)

    parent, union = _parent_matrix(tmp_path, universe, state, config)
    identity = str(parent[mw.MATRIX_IDENTITY_KEY])

    # The SAME content as a blob: carry the identity, drop every internal stamp so the
    # matrix has no path and therefore travels as a blob.
    blob_matrix = {name: value for name, value in parent.items() if not name.startswith("_")}
    blob_matrix[mw.MATRIX_IDENTITY_KEY] = identity
    blob_kind, blob_result = _worker_result(blob_matrix, squad, config)

    file_kind, file_result = _worker_result(parent, squad, config)
    assert file_kind == "file" and blob_kind == "blob"

    seq_policy, seq_scores = _sequential_result(parent, squad, config)

    for label, result in (("blob", blob_result), ("file", file_result)):
        _assert_same_decision(result.entry["policy"], result.entry["scores"],
                              seq_policy, seq_scores, label=f"{label} vs sequential")
    _assert_same_decision(file_result.entry["policy"], file_result.entry["scores"],
                          blob_result.entry["policy"], blob_result.entry["scores"],
                          label="file vs blob")
    assert file_result.entry["mean_gross"] == blob_result.entry["mean_gross"]


def test_the_fixture_is_load_bearing_for_the_armband(tmp_path):
    """Guard the guard: a fixture that stopped discriminating would void the test above."""

    universe, state, meta = _universe()
    config = _config()
    squad = tuple(int(p) for p in SQUAD_IDS)
    union = _union_for(universe, state, config)

    with_blocks = _semantic_matrix(union)
    without = _semantic_matrix(union, drop=("expected_bonus",))
    p_with, _ = _sequential_result(with_blocks, squad, config)
    p_without, _ = _sequential_result(without, squad, config)

    assert p_with.captain_id == EXPECTED_CAPTAIN_WITH_BLOCKS
    assert p_without.captain_id == EXPECTED_CAPTAIN_WITHOUT_BLOCKS
    assert ml.captain_value_basis(with_blocks) != ml.captain_value_basis(without)


def test_the_fixture_is_load_bearing_for_role_actionability(tmp_path):
    """The historical visible defect: the restricted keeper must be benched."""

    universe, state, meta = _universe()
    config = _config()
    squad = tuple(int(p) for p in SQUAD_IDS)
    union = _union_for(universe, state, config)

    with_blocks = _semantic_matrix(union)
    without = _semantic_matrix(union, drop=("role_actionability",))
    p_with, _ = _sequential_result(with_blocks, squad, config)
    p_without, _ = _sequential_result(without, squad, config)

    assert p_with.bench_gk_id == EXPECTED_BENCH_GK_WITH_BLOCKS
    assert p_without.bench_gk_id == EXPECTED_BENCH_GK_WITHOUT_BLOCKS
    assert 1 not in p_with.starter_ids and 1 in p_without.starter_ids


# ---------------------------------------------------------------------------
# B / C — each block survives FILE transport on its own
# ---------------------------------------------------------------------------


def test_expected_bonus_survives_file_transport(tmp_path):
    universe, state, meta = _universe()
    config = _config()
    squad = tuple(int(p) for p in SQUAD_IDS)
    parent, _union = _parent_matrix(tmp_path, universe, state, config)

    kind, result = _worker_result(parent, squad, config)
    assert kind == "file"
    assert result.entry["policy"].captain_id == EXPECTED_CAPTAIN_WITH_BLOCKS


def test_role_actionability_survives_file_transport(tmp_path):
    universe, state, meta = _universe()
    config = _config()
    squad = tuple(int(p) for p in SQUAD_IDS)
    parent, _union = _parent_matrix(tmp_path, universe, state, config)

    kind, result = _worker_result(parent, squad, config)
    assert kind == "file"
    assert result.entry["policy"].bench_gk_id == EXPECTED_BENCH_GK_WITH_BLOCKS
    assert 1 not in result.entry["policy"].starter_ids


# ---------------------------------------------------------------------------
# D / E — a current-schema matrix missing a semantic block fails CLOSED
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dropped", ["expected_bonus", "role_actionability"])
def test_a_matrix_missing_a_semantic_block_fails_closed_in_the_worker(tmp_path, dropped):
    """A worker must never default a missing block to "no bonus" / "no restriction".

    The file is written WITHOUT the block and the matrix is stamped with its path, so the
    worker really loads it from disk.  A lenient loader is the failure mode under test: on
    the predecessor the worker evaluates it happily, having dropped the block anyway.
    """

    universe, state, meta = _universe()
    config = _config()
    squad = tuple(int(p) for p in SQUAD_IDS)

    union = _union_for(universe, state, config)
    key = _write_cache_file(tmp_path, union, config, drop=(dropped,))
    path = tmp_path / f"{key}.json"
    matrix = _semantic_matrix(union, drop=(dropped,))
    matrix[mw.MATRIX_IDENTITY_KEY] = key
    ro._stamp_matrix_path(matrix, path)

    kind, payload, digest = px.describe_matrix_source(matrix)
    assert kind == "file", "this arm must exercise the file transport"
    px._worker_init({"config": config,
                     "positions_by_id": {int(p): POSITION[int(p)] for p in squad},
                     "matrix_sources": {4: (kind, payload)},
                     "matrix_identity": {4: key}, "matrix_cache": 1})
    task = px.ExactTask(key=ro.exact_evaluation_key(4, tuple(squad), config, key),
                        event=4, squad_ids=tuple(squad), world_identity=key,
                        matrix_digest=digest)
    with pytest.raises(px.ParallelExactError) as failure:
        px.worker_evaluate(task)
    assert dropped in str(failure.value)
    assert "SEMANTIC" in str(failure.value).upper()


@pytest.mark.parametrize("dropped", ["expected_bonus", "role_actionability"])
def test_a_blob_missing_a_semantic_block_fails_closed_in_the_worker(dropped):
    """The blob transport is not a validation bypass: it enforces the same contract."""

    universe, state, meta = _universe()
    config = _config()
    squad = tuple(int(p) for p in SQUAD_IDS)
    union = _union_for(universe, state, config)

    matrix = _semantic_matrix(union, drop=(dropped,))
    kind, payload, digest = px.describe_matrix_source(matrix)
    assert kind == "blob"
    px._worker_init({"config": config,
                     "positions_by_id": {int(p): POSITION[int(p)] for p in squad},
                     "matrix_sources": {4: (kind, payload)},
                     "matrix_identity": {}, "matrix_cache": 1})
    task = px.ExactTask(key=ro.exact_evaluation_key(4, tuple(squad), config, None),
                        event=4, squad_ids=tuple(squad), world_identity=None,
                        matrix_digest=digest)
    with pytest.raises(px.ParallelExactError) as failure:
        px.worker_evaluate(task)
    assert dropped in str(failure.value)


# ---------------------------------------------------------------------------
# F — ONE shared semantic contract, applied by both transports
# ---------------------------------------------------------------------------


def test_both_transports_run_the_same_semantic_validation(tmp_path, monkeypatch):
    """Neither transport may reconstruct a matrix on its own terms."""

    seen: list[str] = []
    original = px.normalise_semantic_matrix

    def spy(raw, **kwargs):
        seen.append(str(kwargs.get("source", "")))
        return original(raw, **kwargs)

    monkeypatch.setattr(px, "normalise_semantic_matrix", spy)

    universe, state, meta = _universe()
    config = _config()
    squad = tuple(int(p) for p in SQUAD_IDS)
    parent, _union = _parent_matrix(tmp_path, universe, state, config)

    file_kind, _ = _worker_result(parent, squad, config)
    blob_matrix = {name: value for name, value in parent.items() if not name.startswith("_")}
    blob_matrix[mw.MATRIX_IDENTITY_KEY] = str(parent[mw.MATRIX_IDENTITY_KEY])
    blob_kind, _ = _worker_result(blob_matrix, squad, config)

    assert (file_kind, blob_kind) == ("file", "blob")
    assert any("file" in source for source in seen), seen
    assert any("blob" in source for source in seen), seen


# ---------------------------------------------------------------------------
# End-to-end: a real parallel pool over real file transport
# ---------------------------------------------------------------------------


def test_optimize_parallel_equals_sequential_over_real_file_transport(tmp_path):
    """The production wiring: cache_dir (=> FILE transport) and a real spawn pool.

    This is the end-to-end form of the same claim.  It is deliberately *not* run through
    ``world_provider``, because an injected matrix has no path stamp and would travel as
    a blob — the very reason the predecessor's gates could not see the defect.
    """

    universe, state, meta = _universe()
    config = _config()
    union = _union_for(universe, state, config)
    for event in config.events:
        _write_cache_file(tmp_path, union, config, event=int(event))

    bundles = {int(event): _certified_bundle(int(event)) for event in config.events}
    scenario = _scenario()

    def _run(workers):
        return ro.optimize(universe=universe, initial_state=state, scenario=scenario,
                           player_meta=meta, bundles=bundles, conn=None, config=config,
                           cache_dir=tmp_path, exact_cache={}, parallel_workers=workers)

    sequential = _run(None)
    parallel = _run(2)

    reported = parallel["parallel_exact"]
    assert reported["path"] == "parallel"
    assert reported["worker_jobs_completed"] == reported["unique_exact_keys"]

    assert parallel["routes"] == sequential["routes"]
    assert parallel["h1_frontier"] == sequential["h1_frontier"]
    assert parallel["supported_3gw_frontier"] == sequential["supported_3gw_frontier"]
    assert parallel["families"] == sequential["families"]
    assert parallel["paired_h1"] == sequential["paired_h1"]
    assert parallel["paired_supported_3gw"] == sequential["paired_supported_3gw"]
    assert parallel["roll_baseline"] == sequential["roll_baseline"]
    assert parallel["exact_cache_entries"] == sequential["exact_cache_entries"]

    # The transport actually used was the FILE one.
    assert all(info["source"] == "cache" for info in parallel["world_info"].values())

    # Every per-event decision and every per-world score vector must match exactly.
    assert sorted(parallel["routes"]) == sorted(sequential["routes"])
    for route_id, record in sequential["routes"].items():
        other = parallel["routes"][route_id]
        assert other["per_event"] == record["per_event"], route_id
        assert other["event_scores"] == record["event_scores"], route_id
        for mine, theirs in zip(other["per_event"], record["per_event"]):
            assert mine["policy"] == theirs["policy"], route_id
            assert mine["mean_gross_core"] == theirs["mean_gross_core"], route_id
