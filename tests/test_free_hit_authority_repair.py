"""Free Hit V1 authority repair — the four finding clusters from the Sol review.

P1  the four-GW comparison values BOTH arms on their own actual H2-H4 route
P2a canonical cutoff / keyset / pool authority
P2b authoritative player-pool completeness
P2c predictive identity anchored to the CERTIFIED authority, not to a second
    caller-supplied object
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fpl_brain import chip_free_hit as fh  # noqa: E402
from fpl_brain import season_rules as sr  # noqa: E402
from test_chip_free_hit import (  # noqa: E402
    DATA, EVENT, GENERATION, LEGAL_OWNED, SOURCE, UNIVERSE, _authority, _core_map, _identity,
    _permanent, _pool, _request, _tails, _worlds, CONFIG, CUTOFF,
)

RULES = sr.SeasonRules(season="2026/27")


def _improving(**overrides):
    values = {pid: 1.0 for pid in UNIVERSE}
    for pid in (20, 21, 22, 23, 24, 28, 29, 30):
        values[pid] = 6.0
    return _request(core=_core_map(values), bank=200, **overrides)


# ---------------------------------------------------------------------------
# P1 — THE FOUR-GW COMPARISON
#
# The decisive four-GW route tests live in ``test_free_hit_route_authority.py``,
# built on REAL canonical routes.  They replaced the tests that stood here: those
# were written against a Free Hit tail DTO whose numeric fields were supplied by
# the caller, which is precisely the authority this repair removed.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# P2c — PREDICTIVE AUTHORITY
# ---------------------------------------------------------------------------


def test_P2c_the_self_certification_attack_refuses():
    """Both caller identities mutated TOGETHER still cannot be authoritative.

    This is Sol's decisive case: the world matrix and the request's companion
    identity agree with each other on cutoff, source snapshot, generation and
    model/config, while the data snapshot still matches the binding.  Agreement
    between two caller objects is not authority.
    """

    coherent = _improving()
    certified = _authority()
    mutations = {
        "cutoff": {"cutoff": "2026-09-16T12:00:00Z"},
        "source_snapshot_sha256": {"source_snapshot_sha256": "sha256:" + "8" * 64},
        "generation": {"generation": "2026-09-16T09:00:00Z"},
        "model_config_identity": {"model_config_identity": "sha256:" + "7" * 64},
    }
    for dimension, mutation in mutations.items():
        forged_identity = _identity(**mutation)
        request = _improving(
            identity=forged_identity,
            worlds=_worlds(identity=forged_identity),   # BOTH mutated together
            authority=certified,
        )
        assert request.world_identity.disagreements_with(
            request.h1_worlds.identity
        ) == [], f"{dimension}: the two caller objects must agree for this attack to be real"
        problems = fh.contract_problems(request)
        assert any(
            fh.FH_DECISION_AUTHORITY_MISMATCH in problem and dimension in problem
            for problem in problems
        ), (dimension, problems)
        # And a coherent control passes.
    assert not any(
        fh.FH_DECISION_AUTHORITY_MISMATCH in problem
        for problem in fh.contract_problems(coherent)
    )


@pytest.mark.parametrize("dimension", ["cutoff", "data_snapshot_sha256", "source_snapshot_sha256",
                                       "generation", "model_config_identity"])
def test_P2c_the_bound_identity_is_anchored_to_the_authority(dimension):
    """Mutating ONLY the bound identity (the matrix staying certified) also refuses."""

    coherent = _improving()
    field = {"cutoff": "cutoff", "data_snapshot_sha256": "data_snapshot_sha256",
             "source_snapshot_sha256": "source_snapshot_sha256", "generation": "generation",
             "model_config_identity": "model_config_identity"}[dimension]
    forged = _identity(**{field: "sha256:" + "9" * 64 if field != "cutoff" else "2026-09-16T23:00:00Z"})
    request = _improving(identity=forged)
    problems = fh.contract_problems(request)
    assert any(dimension in problem for problem in problems), (dimension, problems)
    assert not any(
        fh.FH_DECISION_AUTHORITY_MISMATCH in problem for problem in fh.contract_problems(coherent)
    )


def test_P2c_an_authority_whose_identity_does_not_recompute_refuses():
    """A copied certification identity on a different artifact fails rather than certifying."""

    from fpl_brain import four_gw_decision as fg

    other = _authority()
    artifact = {
        "planning_cutoff": "2026-09-16T12:00:00Z",       # different cutoff
        "data_snapshot_sha256": DATA,
        "certified_bundle_identity": {"source_snapshot_sha256": SOURCE, "generation": GENERATION,
                                      "model_config_identity": CONFIG},
        "four_gw_certification_identity": other.certification_identity,   # the COPIED identity
    }
    assert fg.certification_identity_of(artifact) != other.certification_identity
    with pytest.raises(fh.FreeHitInputError) as caught:
        fh.FreeHitDecisionAuthority.from_certification(artifact)
    assert caught.value.reasons[0] == fh.FH_DECISION_AUTHORITY_REQUIRED


def test_P2c_a_missing_authority_refuses():
    request = _improving(authority=_authority())     # coherent control
    assert not any(fh.FH_DECISION_AUTHORITY_REQUIRED in p for p in fh.contract_problems(request))
    bare = fh.FreeHitRequest(
        permanent=request.permanent, horizon_binding=request.horizon_binding,
        h1_worlds=request.h1_worlds, world_identity=request.world_identity,
        positions=request.positions, clubs=request.clubs,
        market_price_tenths=request.market_price_tenths, pool_binding=request.pool_binding,
        play_route=request.play_route, save_route=request.save_route, rules=RULES,
    )
    problems = fh.contract_problems(bare)
    assert any(fh.FH_DECISION_AUTHORITY_REQUIRED in problem for problem in problems), problems


# ---------------------------------------------------------------------------
# P2b — PLAYER POOL COMPLETENESS
# ---------------------------------------------------------------------------


def test_P2b_an_extra_non_official_player_is_a_contradiction_not_an_exclusion():
    """Sol's rule: an extra player is refused outright, never 'excluded but continue'."""

    ids = (*UNIVERSE, 999)
    values = {pid: 1.0 for pid in UNIVERSE}
    values[999] = 500.0
    core = _core_map(values)
    minutes = {pid: tuple(90.0 for _ in range(8)) for pid in ids}
    request = _request(
        worlds=_worlds(core, minutes, ids=ids), prices=_market_all() | {999: 1}, pool=_pool(UNIVERSE),
    )
    scores, stats = fh.screen_players(request)
    assert 999 not in scores
    assert stats["excluded_not_officially_eligible"] == 1
    # And it cannot re-enter anywhere downstream.
    temporary, search = fh.optimize_free_hit_squad(request)
    assert 999 not in temporary.squad_ids
    assert 999 not in search["seed_squad"] and 999 not in search["improved_squad"]
    assert all(999 not in row["squad"] for row in search["exact_scored"])


def _market_all() -> dict[int, int]:
    return {pid: 50 for pid in UNIVERSE}


# ---------------------------------------------------------------------------
# PASSED CORE SEMANTICS — regression guards
# ---------------------------------------------------------------------------


def test_the_core_free_hit_semantics_are_unchanged():
    """The four repairs must not reopen anything Sol passed."""

    play_tail, save_tail = _tails(_permanent(LEGAL_OWNED, bank=200))
    evaluation = fh.evaluate_free_hit(_improving(play_tail=play_tail, save_tail=save_tail))
    metrics = evaluation.candidate_metrics
    assert metrics["restoration_leaks"] == []
    assert metrics["permanent_squad_unchanged_by_chip"] is True
    assert metrics["normal_transfer_hits_charged"] == 0
    assert metrics["temporary_basis_is_temporary"] is True
    assert metrics["no_global_optimum_claim"] is True
    assert evaluation.execution_permitted is False
    assert evaluation.data_snapshot_bound is True
    assert evaluation.evidence["free_hit_quantitative_capability"] == "SUPPORTED_REVIEW_ONLY"


# ---------------------------------------------------------------------------
# P2 — WORLD KEYSET COMPLETENESS
# ---------------------------------------------------------------------------


def _world_keyset_problems(*, ids, values=None):
    values = values or {pid: 1.0 for pid in ids}
    core = _core_map(values)
    minutes = {pid: tuple(90.0 for _ in range(8)) for pid in ids}
    request = _request(worlds=_worlds(core, minutes, ids=ids), pool=_pool(UNIVERSE))
    return fh.contract_problems(request)


def test_P2_the_world_keyset_must_be_the_official_pool_exactly():
    assert not any(fh.FH_WORLD_KEYSET_MISMATCH in p for p in _world_keyset_problems(ids=UNIVERSE))


def test_P2_a_missing_official_player_in_the_world_matrix_refuses():
    """No projection is a contradiction, not a screening exclusion."""

    ids = tuple(pid for pid in UNIVERSE if pid != 7)
    problems = _world_keyset_problems(ids=ids)
    assert any(fh.FH_WORLD_KEYSET_MISMATCH in p and "omits" in p for p in problems), problems


def test_P2_an_extra_world_only_player_refuses_before_any_search():
    """999 in the MATRIX is refused even when every metadata map is canonical."""

    ids = (*UNIVERSE, 999)
    values = {pid: 1.0 for pid in UNIVERSE}
    values[999] = 500.0
    problems = _world_keyset_problems(ids=ids, values=values)
    assert any(fh.FH_WORLD_KEYSET_MISMATCH in p and "non-official" in p for p in problems), problems


def test_P2_the_same_count_with_wrong_ids_refuses():
    ids = tuple(sorted(set(UNIVERSE) - {30} | {999}))
    problems = _world_keyset_problems(ids=ids)
    assert any(fh.FH_WORLD_KEYSET_MISMATCH in p for p in problems), problems


def test_P2_a_contradictory_universe_never_becomes_a_smaller_search():
    """The refusal happens BEFORE screening, so no search work begins."""

    ids = (*UNIVERSE, 999)
    request = _request(worlds=_worlds(ids=ids), pool=_pool(UNIVERSE))
    evaluation = fh.evaluate_free_hit(request)
    assert evaluation.mean_uplift is None
    assert evaluation.candidate_metrics == {"mean_paired_uplift": None}
    assert fh.FH_WORLD_KEYSET_MISMATCH in evaluation.reason_codes
