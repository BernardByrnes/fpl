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
# ---------------------------------------------------------------------------


def test_P1_the_arms_enter_H2_with_genuinely_different_states():
    """The counterexample: a played chip preserves the bank; an ordinary week accrues."""

    permanent = _permanent(LEGAL_OWNED, bank=200, ft=2)
    request = fh.FreeHitRequest(
        permanent=permanent, horizon_binding=_request().horizon_binding,
        h1_worlds=_worlds(), world_identity=_identity(),
        positions={}, clubs={}, market_price_tenths={}, pool_binding=_pool(), rules=RULES,
    )
    restored = fh.restore_permanent_state(permanent, rules=RULES, restored_event=EVENT + 1)
    save_state = fh.save_arm_h2_state(request)
    assert restored.free_transfers == 2            # preserved by the chip
    assert save_state.free_transfers == 3          # ordinary progression
    assert restored.free_transfers != save_state.free_transfers
    # Squad, bank and basis are identical: ONLY the free-transfer count diverges.
    assert restored.owned_ids == save_state.owned_ids
    assert restored.bank_tenths == save_state.bank_tenths
    assert restored.equals_permanent(permanent) == []


def test_P1_the_four_gw_value_is_the_h1_difference_PLUS_the_tail_difference():
    """The uplift is a real four-GW number, not an H1-only one wearing a wrapper."""

    permanent = _permanent(LEGAL_OWNED, bank=200, ft=2)
    play_tail, save_tail = _tails(permanent, play_values=(9.0, 9.0, 9.0), save_values=(6.0, 6.0, 6.0))
    request = _improving(play_tail=play_tail, save_tail=save_tail)
    evaluation = fh.evaluate_free_hit(request)
    metrics = evaluation.candidate_metrics

    assert metrics["h1_paired_uplift"] != 0.0
    assert metrics["play_tail_value"] == pytest.approx(27.0)
    assert metrics["save_tail_value"] == pytest.approx(18.0)
    assert metrics["tail_delta"] == pytest.approx(9.0)
    assert metrics["mean_paired_uplift"] == pytest.approx(
        metrics["h1_paired_uplift"] + metrics["tail_delta"], abs=1e-5
    )
    assert metrics["four_gw_play_value"] == pytest.approx(
        metrics["temporary_h1_value"] + metrics["play_tail_value"], abs=1e-5
    )
    assert metrics["four_gw_save_value"] == pytest.approx(
        metrics["permanent_h1_baseline"] + metrics["save_tail_value"], abs=1e-5
    )
    assert metrics["mean_paired_uplift"] == pytest.approx(
        metrics["four_gw_play_value"] - metrics["four_gw_save_value"], abs=1e-5
    )
    assert evaluation.evidence["planning_event"] == EVENT
    assert metrics["arms_valued_separately"] is True


def test_P1_the_same_H1_uplift_with_a_different_downstream_route_decides_differently():
    """Sol's decisive case: identical H1 gain, different H2-H4 consequence."""

    permanent = _permanent(LEGAL_OWNED, bank=200, ft=2)
    flat_play, flat_save = _tails(permanent, play_values=(5.0,) * 3, save_values=(5.0,) * 3)
    better_play, better_save = _tails(permanent, play_values=(20.0,) * 3, save_values=(5.0,) * 3)
    worse_play, worse_save = _tails(permanent, play_values=(1.0,) * 3, save_values=(5.0,) * 3)

    baseline = fh.evaluate_free_hit(_improving(play_tail=flat_play, save_tail=flat_save))
    better = fh.evaluate_free_hit(_improving(play_tail=better_play, save_tail=better_save))
    worse = fh.evaluate_free_hit(_improving(play_tail=worse_play, save_tail=worse_save))

    # The H1 component is IDENTICAL across all three: only the tail differs.
    assert better.candidate_metrics["h1_paired_uplift"] == pytest.approx(
        baseline.candidate_metrics["h1_paired_uplift"]
    )
    assert worse.candidate_metrics["h1_paired_uplift"] == pytest.approx(
        baseline.candidate_metrics["h1_paired_uplift"]
    )
    # And the DECISION value moves with it.
    assert better.mean_uplift == pytest.approx(baseline.mean_uplift + 45.0, abs=1e-5)
    assert worse.mean_uplift == pytest.approx(baseline.mean_uplift - 12.0, abs=1e-5)
    assert worse.mean_uplift < baseline.mean_uplift < better.mean_uplift


def test_P1_the_play_tail_must_start_from_the_restored_permanent_squad():
    """A temporary Free Hit squad leaking into H2-H4 is refused, not valued."""

    permanent = _permanent(LEGAL_OWNED)
    play_tail, save_tail = _tails(permanent)
    temporary_squad = tuple(sorted(set(UNIVERSE) - set(LEGAL_OWNED)))[:15]
    leaked = fh.FreeHitTailRoute(
        arm=play_tail.arm, events=play_tail.events, h2_squad_ids=temporary_squad,
        h2_bank_tenths=play_tail.h2_bank_tenths,
        h2_purchase_price_tenths=play_tail.h2_purchase_price_tenths,
        h2_free_transfers=play_tail.h2_free_transfers,
    )
    request = _improving(play_tail=leaked, save_tail=save_tail)
    problems = fh.contract_problems(request)
    assert any(fh.FH_TAIL_ROUTE_INVALID in problem for problem in problems), problems
    assert any("does not start H2 from the expected permanent squad" in problem for problem in problems)
    assert fh.evaluate_free_hit(request).mean_uplift is None


@pytest.mark.parametrize("mutation,needle", [
    ("ft", "free transfers"),
    ("bank", "bank"),
    ("label", "labelled"),
])
def test_P1_a_tail_that_describes_the_wrong_arm_state_refuses(mutation, needle):
    permanent = _permanent(LEGAL_OWNED, bank=200, ft=2)
    play_tail, save_tail = _tails(permanent)
    if mutation == "ft":
        forged = fh.FreeHitTailRoute(
            arm=play_tail.arm, events=play_tail.events, h2_squad_ids=play_tail.h2_squad_ids,
            h2_bank_tenths=play_tail.h2_bank_tenths,
            h2_purchase_price_tenths=play_tail.h2_purchase_price_tenths, h2_free_transfers=3,
        )
        request = _improving(play_tail=forged, save_tail=save_tail)
    elif mutation == "bank":
        forged = fh.FreeHitTailRoute(
            arm=play_tail.arm, events=play_tail.events, h2_squad_ids=play_tail.h2_squad_ids,
            h2_bank_tenths=999, h2_purchase_price_tenths=play_tail.h2_purchase_price_tenths,
            h2_free_transfers=play_tail.h2_free_transfers,
        )
        request = _improving(play_tail=forged, save_tail=save_tail)
    else:
        forged = fh.FreeHitTailRoute(
            arm=fh.FreeHitTailRoute.FREE_HIT_ARM_SAVE, events=play_tail.events,
            h2_squad_ids=play_tail.h2_squad_ids, h2_bank_tenths=play_tail.h2_bank_tenths,
            h2_purchase_price_tenths=play_tail.h2_purchase_price_tenths,
            h2_free_transfers=play_tail.h2_free_transfers,
        )
        request = _improving(play_tail=forged, save_tail=save_tail)
    problems = fh.contract_problems(request)
    assert any(fh.FH_TAIL_ROUTE_INVALID in problem for problem in problems), (mutation, problems)
    assert any(needle in problem for problem in problems), (mutation, problems)


def test_P1_a_missing_tail_refuses():
    """A request without both arms' routes is not an exact four-GW decision."""

    coherent = _improving()
    bare = fh.FreeHitRequest(
        permanent=coherent.permanent, horizon_binding=coherent.horizon_binding,
        h1_worlds=coherent.h1_worlds, world_identity=coherent.world_identity,
        positions=coherent.positions, clubs=coherent.clubs,
        market_price_tenths=coherent.market_price_tenths, pool_binding=coherent.pool_binding,
        decision_authority=coherent.decision_authority, rules=RULES,
    )
    problems = fh.contract_problems(bare)
    assert any(fh.FH_TAIL_ROUTE_MISSING in problem for problem in problems), problems
    assert fh.evaluate_free_hit(bare).mean_uplift is None


def test_P1_the_save_arm_is_not_a_clairvoyant_future_chip():
    """SAVE is ordinary progression; no best-future-Free-Hit is searched for."""

    source = Path(fh.__file__).read_text(encoding="utf-8")
    assert "clairvoyant" not in source.lower()
    permanent = _permanent(LEGAL_OWNED, bank=200, ft=2)
    _play, save_tail = _tails(permanent)
    # The SAVE state's free transfers are the CANONICAL progression, not a chip.
    assert save_tail.h2_free_transfers == sr.free_transfers_after_gameweek(RULES, 2, 0)
    assert save_tail.h2_free_transfers != fh.post_free_hit_ft_state(
        RULES, event_start_free_transfers=2
    )


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
        play_tail=request.play_tail, save_tail=request.save_tail, rules=RULES,
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
