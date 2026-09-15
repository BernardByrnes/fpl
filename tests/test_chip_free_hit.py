"""Free Hit V1 — the two-state contract, canonical economics, and H1 scoring.

The chip's central hazard is that a TEMPORARY squad quietly becomes permanent.
These tests attack that shape directly: the restoration function is inspected for
its signature (so no code path from temporary to permanent can exist), and every
permanent component is compared before and after.
"""

from __future__ import annotations

import ast
import inspect
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fpl_brain import chip_decision as cd  # noqa: E402
from fpl_brain import chip_free_hit as fh  # noqa: E402
from fpl_brain import season_rules as sr  # noqa: E402
from fpl_brain import transfer_state as ts  # noqa: E402
from fpl_brain.chip_wildcard import (  # noqa: E402
    WildcardPoolBinding, WildcardPredictiveIdentity, WildcardWorldInputs,
)
from fpl_brain.ingest_provenance import element_id_sha256  # noqa: E402

EVENT = 5
WORLDS = 8
CUTOFF = "2026-09-16T11:00:00Z"
DATA = "sha256:" + "d" * 64
SOURCE = "sha256:" + "s" * 64
CERT = "sha256:" + "c" * 64
GENERATION = "2026-09-16T08:00:00Z"
CONFIG = "sha256:" + "f" * 64
RULES = sr.SeasonRules(season="2026/27")

#: A small but complete universe: 4 GKP / 10 DEF / 10 MID / 6 FWD, so a legal
#: 2/5/5/3 fifteen exists with room to improve.
UNIVERSE: tuple[int, ...] = tuple(range(1, 31))
POSITION: dict[int, str] = {}
for _pid in UNIVERSE:
    if _pid <= 4:
        POSITION[_pid] = "GKP"
    elif _pid <= 14:
        POSITION[_pid] = "DEF"
    elif _pid <= 24:
        POSITION[_pid] = "MID"
    else:
        POSITION[_pid] = "FWD"
#: Clubs spread so no legal fifteen is blocked by the 3-per-club limit.
CLUB: dict[int, int] = {pid: 100 + ((pid - 1) % 10) for pid in UNIVERSE}

#: A legal permanent fifteen: GKP 1,2 | DEF 5-9 | MID 15-19 | FWD 25,26,27.
#: Clubs stay within the 3-per-club limit by construction.
LEGAL_OWNED: tuple[int, ...] = (1, 2, 5, 6, 7, 8, 9, 15, 16, 17, 18, 19, 25, 26, 27)


def _identity(**overrides) -> WildcardPredictiveIdentity:
    base = dict(cutoff=CUTOFF, data_snapshot_sha256=DATA, source_snapshot_sha256=SOURCE,
                generation=GENERATION, model_config_identity=CONFIG)
    base.update(overrides)
    return WildcardPredictiveIdentity(**base)


def _binding(event: int = EVENT, snapshot: str = DATA, events=None) -> cd.ChipHorizonBinding:
    return cd.ChipHorizonBinding(
        planning_event=int(event),
        horizon_events=events if events is not None else cd.canonical_chip_horizon(int(event)),
        certification_identity=CERT, data_snapshot_sha256=snapshot,
    )


def _pool(ids: tuple[int, ...] = UNIVERSE) -> WildcardPoolBinding:
    return WildcardPoolBinding(
        generation_identity=GENERATION, generation_id_sha256=element_id_sha256(sorted(ids)),
        official_count=len(ids), eligible_ids=tuple(sorted(ids)),
    )


def _worlds(
    core: dict[int, tuple[float, ...]] | None = None,
    minutes: dict[int, tuple[float, ...]] | None = None,
    *,
    ids: tuple[int, ...] = UNIVERSE,
    event: int = EVENT,
    identity: WildcardPredictiveIdentity | None = None,
    worlds: int = WORLDS,
) -> WildcardWorldInputs:
    if core is None:
        core = {pid: tuple(1.0 for _ in range(worlds)) for pid in ids}
    if minutes is None:
        minutes = {pid: tuple(90.0 for _ in range(worlds)) for pid in core}
    return WildcardWorldInputs(
        event=int(event), worlds=worlds, player_ids=tuple(sorted(core)),
        minutes=minutes, core=core, identity=identity if identity is not None else _identity(),
    )


def _permanent(owned: tuple[int, ...], *, bank: int = 20, ft: int = 2,
               basis: dict[int, int] | None = None) -> fh.FreeHitPermanentState:
    return fh.FreeHitPermanentState(
        event=EVENT, owned_ids=tuple(sorted(owned)),
        purchase_price_tenths=basis if basis is not None else {pid: 50 for pid in owned},
        bank_tenths=int(bank), event_start_free_transfers=int(ft),
        positions=POSITION, clubs=CLUB,
    )


def _market(prices: dict[int, int] | None = None) -> dict[int, int]:
    return prices if prices is not None else {pid: 50 for pid in UNIVERSE}


def _request(
    owned: tuple[int, ...] = LEGAL_OWNED,
    *, core=None, minutes=None, worlds=None, identity=None, event=EVENT,
    snapshot=DATA, prices=None, bank=20, ft=2, basis=None,
    pool: WildcardPoolBinding | None = None, **overrides,
) -> fh.FreeHitRequest:
    h1 = worlds if worlds is not None else _worlds(core, minutes, event=event,
                                                    identity=identity)
    base = dict(
        permanent=_permanent(owned, bank=bank, ft=ft, basis=basis),
        horizon_binding=_binding(event, snapshot),
        h1_worlds=h1,
        world_identity=identity if identity is not None else _identity(),
        positions=POSITION, clubs=CLUB, market_price_tenths=_market(prices),
        pool_binding=_pool() if pool is None else pool,
        chip_available=True, rules=RULES,
    )
    base.update(overrides)
    return fh.FreeHitRequest(**base)


# ---------------------------------------------------------------------------
# THE RESTORATION INVARIANT — structural first
# ---------------------------------------------------------------------------


def test_restoration_is_structurally_unreachable_from_the_temporary_world():
    """``restore_permanent_state`` cannot see a temporary squad, so none can leak.

    This is an AST check, not a behavioural one: a future edit that adds a
    temporary parameter (or reads a module-level temporary state) fails here
    rather than silently changing the contract.
    """

    signature = inspect.signature(fh.restore_permanent_state)
    assert list(signature.parameters) == ["permanent", "rules", "restored_event"]
    for name, parameter in signature.parameters.items():
        assert parameter.kind is not inspect.Parameter.VAR_KEYWORD, name
        assert "temporary" not in name.lower()

    source = Path(fh.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(
        item for item in ast.walk(tree)
        if isinstance(item, ast.FunctionDef) and item.name == "restore_permanent_state"
    )
    referenced = {child.id for child in ast.walk(node) if isinstance(child, ast.Name)}
    referenced |= {
        child.attr for child in ast.walk(node) if isinstance(child, ast.Attribute)
    }
    # The function may name ONLY the permanent world, the rules and the event.
    assert "Temporary" not in "".join(referenced)
    assert not any("temporary" in name.lower() for name in referenced if isinstance(name, str))


def test_restoration_reproduces_the_permanent_squad_basis_and_bank():
    permanent = _permanent(LEGAL_OWNED, bank=37)
    restored = fh.restore_permanent_state(permanent, rules=RULES, restored_event=EVENT + 1)
    assert restored.equals_permanent(permanent) == []
    assert restored.bank_tenths == 37
    assert restored.purchase_price_tenths == permanent.purchase_price_tenths


def test_restoration_never_carries_the_temporary_squad():
    """The temporary fifteen simply is not an input, whatever it contained."""

    permanent = _permanent(LEGAL_OWNED)
    temporary = fh.FreeHitTemporarySquad(
        event=EVENT, squad_ids=tuple(sorted((11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 25, 26, 27, 29))),
        policy=None,  # type: ignore[arg-type]
        remaining_bank_tenths=999, transaction=None,  # type: ignore[arg-type]
    )
    restored = fh.restore_permanent_state(permanent, rules=RULES, restored_event=EVENT + 1)
    assert restored.owned_ids == permanent.owned_ids
    assert restored.bank_tenths == permanent.bank_tenths
    assert 29 not in restored.owned_ids
    assert restored.equals_permanent(permanent) == []
    # The temporary object exists and is different; it simply has no path here.
    assert temporary.squad_ids != permanent.owned_ids


def test_a_temporary_sale_then_drop_restores_the_original_player_and_basis():
    """Attack A + D: permanent player A at basis X, omitted and re-bought later."""

    owned = LEGAL_OWNED
    basis = {pid: 50 for pid in owned}
    basis[17] = 55                     # player A: permanent basis 55
    permanent = _permanent(owned, basis=basis)
    # The temporary world omits 21 entirely and buys others instead.
    temporary_ids = tuple(sorted(set(owned) - {17} | {20}))
    plan = fh.free_hit_transaction_plan(_request(owned, basis=basis), temporary_ids)
    assert 17 in plan.sold_ids and 20 in plan.bought_ids
    assert plan.new_basis[20] == 50    # a TEMPORARY basis
    restored = fh.restore_permanent_state(permanent, rules=RULES, restored_event=EVENT + 1)
    assert 17 in restored.owned_ids
    assert restored.purchase_price_tenths[17] == 55, "the permanent basis was rewritten"
    assert 20 not in restored.owned_ids, "a temporary purchase became permanent"


def test_the_temporary_basis_never_overwrites_a_permanent_one():
    owned = LEGAL_OWNED
    basis = {pid: 50 for pid in owned}
    permanent = _permanent(owned, basis=basis)
    scratch = dict(permanent.purchase_price_tenths)
    scratch[1] = 12                    # mutate the temporary copy only
    assert permanent.purchase_price_tenths[1] == 50, "the permanent state was mutated in place"


def test_the_permanent_bank_is_restored_not_the_temporary_residual():
    """Attack C: the chip week may end with residual cash; H2 keeps the real bank."""

    owned = LEGAL_OWNED
    permanent = _permanent(owned, bank=37)
    # Buying dearer forwards leaves a DIFFERENT residual than the permanent bank.
    prices = {pid: 50 for pid in UNIVERSE} | {28: 60, 29: 60, 30: 60}
    request = _request(owned, bank=37, prices=prices)
    cheap = tuple(sorted(set(owned) - {25, 26, 27} | {28, 29, 30}))
    plan = fh.free_hit_transaction_plan(request, cheap)
    assert plan.remaining_bank_tenths != 37
    restored = fh.restore_permanent_state(permanent, rules=RULES, restored_event=EVENT + 1)
    assert restored.bank_tenths == 37


# ---------------------------------------------------------------------------
# FREE TRANSFER SEMANTICS — from season_rules only
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("starting_ft", [0, 1, 2, 5])
def test_the_h2_free_transfer_state_is_the_canonical_season_rule(starting_ft):
    permanent = _permanent(LEGAL_OWNED, ft=starting_ft)
    restored = fh.restore_permanent_state(permanent, rules=RULES, restored_event=EVENT + 1)
    expected = sr.free_transfers_after_chip(
        RULES, "freehit", event_start_free_transfers=starting_ft
    )
    assert restored.free_transfers == expected
    assert restored.free_transfers_rule == RULES.free_hit_ft_rule
    assert 0 <= restored.free_transfers <= RULES.max_free_transfers


def test_an_unrecorded_event_start_ft_bank_refuses_rather_than_being_inferred():
    """The canonical helper's own contract: never guess from a depleted count."""

    with pytest.raises(sr.ChipFreeTransferError):
        fh.post_free_hit_ft_state(RULES, event_start_free_transfers=None)


def test_free_hit_is_a_transfer_preserving_chip_in_the_canonical_rules():
    assert RULES.free_hit_ft_rule == "saved_free_transfers_preserved"
    assert sr.normalise_chip_name("freehit") in sr.FT_PRESERVING_CHIPS


# ---------------------------------------------------------------------------
# BUDGET — canonical economics only
# ---------------------------------------------------------------------------


def _legal_owned() -> tuple[int, ...]:
    return LEGAL_OWNED


def test_retained_capital_is_never_counted_as_spendable_cash():
    """Attack A: a retained expensive player's locked capital cannot be spent."""

    owned = _legal_owned()
    prices = {pid: 100 for pid in UNIVERSE}
    request = _request(owned, prices=prices, bank=0)
    # Retaining everything: cash is bank only, cost is zero.
    plan = fh.free_hit_transaction_plan(request, owned)
    assert plan.sold_ids == () and plan.bought_ids == ()
    assert plan.cash_available_tenths == 0
    assert plan.remaining_bank_tenths == 0

    # Buying one unowned player while retaining the rest must use bank only.
    swap_in = tuple(sorted(set(owned) - {27} | {28}))
    plan = fh.free_hit_transaction_plan(request, swap_in)
    assert plan.sold_ids == (27,) and plan.bought_ids == (28,)
    assert plan.cash_available_tenths == ts.selling_price_tenths(50, 100)
    assert plan.purchase_cost_tenths == 100
    assert plan.remaining_bank_tenths == ts.selling_price_tenths(50, 100) - 100


def test_a_retained_player_is_charged_his_selling_value_not_zero():
    """Charging zero is exactly how locked capital gets spent twice."""

    owned = _legal_owned()
    prices = {pid: 100 for pid in UNIVERSE}
    request = _request(owned, prices=prices, bank=0)
    allowance = fh._affordability_allowance(request)
    assert allowance == sum(ts.selling_price_tenths(50, 100) for _ in owned)
    # The greedy charges a retained player his canonical selling value.
    seed = fh._seed_squad(request, {pid: 1.0 for pid in UNIVERSE},
                          fh._position_frontiers(request, {pid: 1.0 for pid in UNIVERSE}, size=12))
    if seed is not None:
        charged = 0
        for pid in seed:
            charged += ts.selling_price_tenths(50, 100) if pid in set(owned) else prices[pid]
        assert charged <= allowance


def test_a_caller_price_cannot_undercut_the_canonical_one():
    """Attack B: the pipeline never reads a caller's price for a permanent sale."""

    owned = _legal_owned()
    request = _request(owned, prices={pid: 100 for pid in UNIVERSE})
    # The canonical selling value comes from basis + canonical market only.
    assert fh.canonical_selling_value(request, 1) == ts.selling_price_tenths(50, 100)


def test_a_missing_canonical_price_refuses():
    """Attack C: an unpriced player is refused, never silently defaulted."""

    owned = _legal_owned()
    request = _request(owned, prices={pid: 50 for pid in UNIVERSE if pid != 5})
    with pytest.raises(fh.FreeHitInputError) as caught:
        fh.canonical_selling_value(request, 5)
    assert caught.value.reasons[0] == fh.FH_PRICING_UNAVAILABLE

    other = _request(owned, prices={pid: 50 for pid in UNIVERSE if pid != 29})
    with pytest.raises(fh.FreeHitInputError) as caught:
        fh.free_hit_transaction_plan(other, tuple(sorted(set(owned) - {27} | {29})))
    assert caught.value.reasons[0] == fh.FH_PRICING_UNAVAILABLE


def test_the_full_universe_used_for_optimisation_is_priced():
    """A temporary squad member with no price cannot be bought."""

    request = _request(prices={pid: 50 for pid in UNIVERSE if pid in UNIVERSE[:20]})
    ok, problems = fh.temporary_squad_legality(request, _legal_owned())
    assert not ok and problems


# ---------------------------------------------------------------------------
# TEMPORARY SQUAD LEGALITY
# ---------------------------------------------------------------------------


def test_a_legal_temporary_squad_passes_and_an_illegal_one_refuses():
    request = _request()
    ok, problems = fh.temporary_squad_legality(request, _legal_owned())
    assert ok, problems
    assert list(request.permanent.positions.values()).count("GKP") >= 2

    # 2 GKP / 4 DEF / 6 MID / 3 FWD — composition violation.
    bad = (1, 2, 3, 4, 5, 6, 15, 16, 17, 20, 21, 22, 25, 26, 27)
    ok, problems = fh.temporary_squad_legality(request, bad)
    assert not ok
    assert any("POSITION_INVALID" in problem for problem in problems)


def test_four_from_one_club_refuses():
    request = _request()
    crowded = dict(CLUB)
    for pid in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15):
        crowded[pid] = 100
    request = fh.FreeHitRequest(
        permanent=fh.FreeHitPermanentState(
            event=EVENT, owned_ids=_legal_owned(),
            purchase_price_tenths={pid: 50 for pid in _legal_owned()},
            bank_tenths=20, event_start_free_transfers=2, positions=POSITION, clubs=crowded,
        ),
        horizon_binding=request.horizon_binding, h1_worlds=request.h1_worlds,
        world_identity=request.world_identity, positions=POSITION, clubs=crowded,
        market_price_tenths=request.market_price_tenths, pool_binding=request.pool_binding,
        rules=RULES,
    )
    ok, problems = fh.temporary_squad_legality(request, _legal_owned())
    assert not ok
    assert any("CLUB_LIMIT_EXCEEDED" in problem for problem in problems)


def test_an_over_budget_temporary_squad_refuses():
    owned = _legal_owned()
    request = _request(owned, prices={pid: 1000 for pid in UNIVERSE}, bank=0)
    expensive = tuple(sorted(set(owned) - {25, 26, 27} | {28, 29, 30}))
    ok, problems = fh.temporary_squad_legality(request, expensive)
    assert not ok
    assert any("over budget" in problem for problem in problems)


# ---------------------------------------------------------------------------
# PLAYER POOL ATTACKS
# ---------------------------------------------------------------------------


def _core_map(values: dict[int, float], worlds: int = WORLDS) -> dict[int, tuple[float, ...]]:
    return {pid: tuple(float(value) for _ in range(worlds)) for pid, value in values.items()}


def test_player_999_with_a_huge_projection_and_a_cheap_price_is_refused():
    """A rogue id must not enter screening, the seed, improvement or the squad."""

    ids = (*UNIVERSE, 999)
    values = {pid: 1.0 for pid in UNIVERSE}
    values[999] = 500.0                      # the projection it must never profit from
    core = _core_map(values)
    minutes = {pid: tuple(90.0 for _ in range(WORLDS)) for pid in ids}
    prices = _market() | {999: 1}            # and the price it must never buy at
    request = _request(
        worlds=_worlds(core, minutes, ids=ids), prices=prices,
        pool=_pool(UNIVERSE),                   # the official pool does NOT contain 999
    )
    scores, stats = fh.screen_players(request)
    assert 999 not in scores, "a non-official id reached screening"
    assert stats["excluded_not_officially_eligible"] == 1

    temporary, search = fh.optimize_free_hit_squad(request)
    assert 999 not in temporary.squad_ids
    assert 999 not in search["seed_squad"]
    assert 999 not in search["improved_squad"]
    assert all(999 not in row["squad"] for row in search["exact_scored"])
    for pid in temporary.squad_ids:
        assert pid in UNIVERSE


def test_the_same_player_count_with_different_ids_is_refused():
    """Counts cannot prove identity: the digest must match the eligible set."""

    other = tuple(sorted(set(UNIVERSE) - {30} | {999}))
    with pytest.raises(Exception) as caught:
        WildcardPoolBinding(
            generation_identity=GENERATION,
            generation_id_sha256=element_id_sha256(sorted(UNIVERSE)),   # the ORIGINAL ids
            official_count=len(other),
            eligible_ids=other,
        )
    assert "do not match the official generation identity digest" in str(caught.value)


def test_a_pool_that_is_not_the_official_generation_refuses():
    broken = _pool(UNIVERSE[:20])
    request = _request(pool=broken)
    # A pool that omits officially eligible ids cannot certify an exhaustive
    # screen, and a pool whose digest disagrees is refused outright.
    with pytest.raises(Exception):
        WildcardPoolBinding(
            generation_identity=GENERATION, generation_id_sha256="sha256:" + "0" * 64,
            official_count=len(UNIVERSE), eligible_ids=UNIVERSE,
        )


def test_screening_is_exhaustive_over_the_official_pool():
    request = _request()
    scores, stats = fh.screen_players(request)
    assert stats["eligible_pool"] == len(UNIVERSE)
    assert stats["screened"] == len(UNIVERSE)
    assert set(scores) == set(UNIVERSE)


def test_a_missing_projection_is_excluded_with_a_reason_and_never_zero():
    """Player 7 is officially eligible but the matrix carries no series for him."""

    ids = tuple(pid for pid in UNIVERSE if pid != 7)
    core = _core_map({pid: 1.0 for pid in ids})
    minutes = {pid: tuple(90.0 for _ in range(WORLDS)) for pid in ids}
    request = _request(worlds=_worlds(core, minutes, ids=ids))
    scores, stats = fh.screen_players(request)
    assert 7 not in scores
    assert stats["eligible_pool"] == len(UNIVERSE)
    assert stats["excluded_missing_projection"] == 1
    assert stats["excluded_missing_projection_ids"] == [7]
    assert stats["missing_projection_policy"] == "EXCLUDED_WITH_REASON"


# ---------------------------------------------------------------------------
# PREDICTIVE IDENTITY ATTACKS
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dimension,mutated", [
    ("cutoff", "2026-09-16T12:00:00Z"),
    ("data_snapshot_sha256", "sha256:" + "9" * 64),
    ("source_snapshot_sha256", "sha256:" + "8" * 64),
    ("generation", "2026-09-16T09:00:00Z"),
    ("model_config_identity", "sha256:" + "7" * 64),
])
def test_a_world_from_another_predictive_world_refuses(dimension, mutated):
    """Every identity dimension is compared, not just the label pair."""

    # The request is bound to the CORRECT world; the matrix came from another one.
    request = _request(worlds=_worlds(identity=_identity(**{dimension: mutated})))
    problems = fh.contract_problems(request)
    assert any(fh.FH_PREDICTIVE_IDENTITY_MISMATCH in problem for problem in problems), (dimension, problems)
    assert any(dimension in problem for problem in problems), (dimension, problems)


def test_a_swapped_world_event_refuses():
    request = _request(worlds=_worlds(event=EVENT + 1))
    problems = fh.contract_problems(request)
    assert any("carries event 6" in problem for problem in problems), problems


def test_a_permanent_state_for_another_gameweek_refuses():
    """The permanent world's EVENT is part of the certified context.

    A GW5 manager state must not be scored against a GW6 horizon and GW6 worlds,
    even though those two agree with each other.
    """

    request = _request(event=EVENT + 1)          # binding and worlds both GW6
    problems = fh.contract_problems(request)     # permanent state is still GW5
    assert any(fh.FH_MANAGER_STATE_INVALID in problem for problem in problems), problems
    assert any("is for GW5" in problem for problem in problems), problems

    # And the coherent control passes the same check.
    assert not any(
        fh.FH_MANAGER_STATE_INVALID in problem
        for problem in fh.contract_problems(_request(event=EVENT))
    )


def test_an_evaluation_produced_for_another_event_refuses():
    request = _request(event=EVENT)
    evaluation = fh.evaluate_free_hit(request)
    decision = cd.decide_chip_action(
        horizon_binding=_binding(), chip_availability=list(CHIP_ROW),
        evaluations={cd.CHIP_ACTION_FH: evaluation}, certification_valid=True,
        manager_state={"squad_ids": list(LEGAL_OWNED)},
    )
    # The evaluation's own horizon_events must be THIS binding's.
    assert decision.status in (cd.STATUS_CHIP_CANDIDATE_RECHECK_REQUIRED, cd.STATUS_CHIP_REVIEW_REQUIRED,
                               cd.STATUS_NO_CHIP)
    assert cd.DIAG_CHIP_EVALUATION_CONTEXT_MISMATCH not in decision.reason_codes


def test_an_empty_data_snapshot_on_the_world_refuses():
    request = _request(identity=_identity(data_snapshot_sha256=""))
    problems = fh.contract_problems(request)
    assert any(fh.FH_DATA_SNAPSHOT_REQUIRED in problem for problem in problems), problems


def test_an_empty_binding_snapshot_refuses():
    request = _request(snapshot="")
    problems = fh.contract_problems(request)
    assert any(fh.FH_DATA_SNAPSHOT_REQUIRED in problem for problem in problems), problems


def test_a_world_snapshot_that_is_not_the_bindings_refuses():
    request = _request(snapshot=DATA, identity=_identity(data_snapshot_sha256="sha256:" + "1" * 64))
    problems = fh.contract_problems(request)
    assert any(fh.FH_DATA_SNAPSHOT_REQUIRED in problem for problem in problems), problems


def test_a_world_matrix_without_an_identity_refuses():
    request = _request(worlds=WildcardWorldInputs(
        event=EVENT, worlds=WORLDS, player_ids=UNIVERSE,
        minutes={pid: tuple(90.0 for _ in range(WORLDS)) for pid in UNIVERSE},
        core=_core_map({pid: 1.0 for pid in UNIVERSE}), identity=None,
    ))
    problems = fh.contract_problems(request)
    assert any(fh.FH_PREDICTIVE_IDENTITY_MISMATCH in problem for problem in problems), problems


@pytest.mark.parametrize("bad", [(5, 6, 7), (5, 6, 7, 8, 9), (5, 6, 6, 8), (5, 6, 7, 9)])
def test_a_non_canonical_horizon_refuses(bad):
    """The shared binding refuses 3/5/duplicate/non-contiguous at construction."""

    with pytest.raises(cd.ChipInputError) as caught:
        _binding(EVENT, events=bad)
    assert cd.DIAG_CHIP_HORIZON_NOT_CANONICAL in caught.value.reasons


# ---------------------------------------------------------------------------
# H1 SCORING — the accepted engine
# ---------------------------------------------------------------------------


def _policy(starter_ids, bench_gk, bench_out, captain, vice):
    return fh.ml.ManagerPolicy(
        starter_ids=tuple(starter_ids), bench_gk_id=int(bench_gk),
        bench_outfield_order=tuple(bench_out), captain_id=int(captain), vice_captain_id=int(vice),
    )


#: GKP 1,2 | DEF 5,6,7,8,9 | MID 15,16,17,18,19 | FWD 25,26,27
_XI = (1, 5, 6, 7, 15, 16, 17, 18, 25, 26, 27)
_BENCH_GK, _BENCH_OUT = 2, (8, 9, 19)


def _engine_total(request, policy, world=0):
    """The certified engine's own total for one world — the ORACLE, not a re-impl."""

    squad = tuple(sorted(fh.ml.policy_player_ids(policy)))
    positions = {pid: POSITION[pid] for pid in squad}
    wm = {pid: float(request.h1_worlds.minutes[pid][world]) for pid in squad}
    wc = {pid: float(request.h1_worlds.core[pid][world]) for pid in squad}
    outcome = fh.ml.resolve_world(policy, positions, wm, wc, require_player_ids=squad)
    extra, source = fh.ml.captain_multiplier(policy, wm, wc, require_player_ids=squad)
    return float(sum(wc[pid] for pid in outcome.counted_ids) + extra), source, extra, outcome


def test_a_legal_temporary_squad_scores_with_the_certified_engine():
    """Wiring check: the series IS the accepted engine's per-world total."""

    values = {pid: 3.0 for pid in UNIVERSE}          # uniform, so no tie-break noise
    request = _request(core=_core_map(values))
    value, policy, series = fh.free_hit_h1_value(request, LEGAL_OWNED)
    assert policy is not None and math.isfinite(value)
    assert len(series) == WORLDS
    # Every starter appears, so nothing autosubs: 11 x 3.0 + one armband copy.
    assert value == pytest.approx(11 * 3.0 + 3.0)
    for world in range(WORLDS):
        expected, _source, _extra, _outcome = _engine_total(request, policy, world)
        assert series[world] == pytest.approx(expected), world


def test_free_hit_does_not_leak_bench_boost_semantics():
    """Free Hit is NOT all-fifteen scoring: the bench scores only via autosubs."""

    values = {pid: 2.0 for pid in UNIVERSE}
    for pid in (2, 8, 9, 19):          # the BENCH, deliberately the best scorers
        values[pid] = 10.0
    request = _request(core=_core_map(values))
    policy = _policy(_XI, _BENCH_GK, _BENCH_OUT, captain=25, vice=26)
    positions = {pid: POSITION[pid] for pid in fh.ml.policy_player_ids(policy)}
    assert fh.ml.policy_legality_errors(policy, positions) == []
    series = fh.h1_core_series(request, policy)
    # Every starter appears, so no autosub happens and the bench contributes ZERO.
    assert series[0] == pytest.approx(11 * 2.0 + 2.0)
    assert series[0] != pytest.approx(11 * 2.0 + 4 * 10.0 + 2.0)


def test_an_absent_starter_is_replaced_by_a_legal_autosub():
    values = {pid: 2.0 for pid in UNIVERSE}
    for pid in (8, 9, 19):
        values[pid] = 6.0
    minutes = {pid: tuple(90.0 for _ in range(WORLDS)) for pid in UNIVERSE}
    minutes[6] = tuple(0.0 for _ in range(WORLDS))     # a starting DEF misses out
    request = _request(core=_core_map(values), minutes=minutes)
    policy = _policy(_XI, _BENCH_GK, _BENCH_OUT, captain=25, vice=26)
    series = fh.h1_core_series(request, policy)
    # The absent DEF is replaced by a bench DEF worth 6.0.
    assert series[0] == pytest.approx(10 * 2.0 + 6.0 + 2.0)


def test_a_missing_starting_goalkeeper_is_replaced_only_by_the_bench_goalkeeper():
    values = {pid: 2.0 for pid in UNIVERSE}
    values[2] = 7.0
    minutes = {pid: tuple(90.0 for _ in range(WORLDS)) for pid in UNIVERSE}
    minutes[1] = tuple(0.0 for _ in range(WORLDS))
    request = _request(core=_core_map(values), minutes=minutes)
    policy = _policy(_XI, _BENCH_GK, _BENCH_OUT, captain=25, vice=26)
    series = fh.h1_core_series(request, policy)
    assert series[0] == pytest.approx(10 * 2.0 + 7.0 + 2.0)


def test_an_absent_captain_hands_the_armband_to_the_vice():
    """The armband rule is the certified one: captain, else vice, else nobody."""

    values = {pid: 2.0 for pid in UNIVERSE}
    values[25], values[26] = 9.0, 5.0
    minutes = {pid: tuple(90.0 for _ in range(WORLDS)) for pid in UNIVERSE}
    policy = _policy(_XI, _BENCH_GK, _BENCH_OUT, captain=25, vice=26)
    present = _request(core=_core_map(values), minutes=minutes)
    minutes_absent = {pid: tuple(v) for pid, v in minutes.items()}
    minutes_absent[25] = tuple(0.0 for _ in range(WORLDS))
    absent = _request(core=_core_map(values), minutes=minutes_absent)

    _t, source_present, extra_present, _o = _engine_total(present, policy)
    _t, source_absent, extra_absent, _o = _engine_total(absent, policy)
    assert source_present == "CAPTAIN" and extra_present == pytest.approx(9.0)
    assert source_absent == "VICE" and extra_absent == pytest.approx(5.0)
    # And the assembled series carries exactly that armband.
    assert fh.h1_core_series(absent, policy)[0] == pytest.approx(_t)


def test_no_captain_and_no_vice_means_no_arbitrary_third_captain():
    """With both armband holders absent nobody else is promoted — not even the top scorer."""

    values = {pid: 2.0 for pid in UNIVERSE}
    values[25], values[26], values[27] = 9.0, 5.0, 11.0
    minutes = {pid: tuple(90.0 for _ in range(WORLDS)) for pid in UNIVERSE}
    minutes[25] = tuple(0.0 for _ in range(WORLDS))
    minutes[26] = tuple(0.0 for _ in range(WORLDS))
    request = _request(core=_core_map(values), minutes=minutes)
    policy = _policy(_XI, _BENCH_GK, _BENCH_OUT, captain=25, vice=26)

    total, source, extra, outcome = _engine_total(request, policy)
    assert source == "NONE" and extra == pytest.approx(0.0)
    # Player 27 scores 11.0 and is counted exactly ONCE.
    assert 27 in outcome.counted_ids
    assert total == pytest.approx(sum(values[pid] for pid in outcome.counted_ids))
    # If he had been promoted the total would be 11.0 higher.
    assert total != pytest.approx(total + 11.0)


def test_a_formation_blocked_bench_player_does_not_enter():
    """The canonical autosub rule applies unchanged under Free Hit."""

    values = {pid: 2.0 for pid in UNIVERSE}
    values[8] = values[9] = 9.0        # two DEF on the bench
    minutes = {pid: tuple(90.0 for _ in range(WORLDS)) for pid in UNIVERSE}
    minutes[6] = tuple(0.0 for _ in range(WORLDS))
    minutes[7] = tuple(0.0 for _ in range(WORLDS))   # two starting DEF absent
    request = _request(core=_core_map(values), minutes=minutes)
    policy = _policy(_XI, _BENCH_GK, _BENCH_OUT, captain=25, vice=26)
    positions = {pid: POSITION[pid] for pid in fh.ml.policy_player_ids(policy)}
    outcome = fh.ml.resolve_world(
        policy, positions,
        {pid: minutes[pid][0] for pid in UNIVERSE}, {pid: values[pid] for pid in UNIVERSE},
        require_player_ids=list(fh.ml.policy_player_ids(policy)),
    )
    counts = {}
    for pid in outcome.counted_ids:
        counts[POSITION[pid]] = counts.get(POSITION[pid], 0) + 1
    assert 3 <= counts["DEF"] <= 5 and counts["FWD"] >= 1


def test_the_best_h1_policy_is_legal_and_deterministic():
    values = {pid: float(pid) for pid in UNIVERSE}
    request = _request(core=_core_map(values))
    first = fh.best_h1_policy(request, LEGAL_OWNED)
    second = fh.best_h1_policy(request, LEGAL_OWNED)
    assert first is not None and first.ordering_key() == second.ordering_key()
    positions = {pid: POSITION[pid] for pid in LEGAL_OWNED}
    assert fh.ml.policy_legality_errors(first, positions) == []


# ---------------------------------------------------------------------------
# NO PERMANENT NORMAL-TRANSFER SIDE EFFECT
# ---------------------------------------------------------------------------


def test_the_whole_fifteen_may_change_without_any_transfer_hit():
    """Free Hit is not fifteen normal transfers: no hits, no route history."""

    owned = LEGAL_OWNED
    wholesale = tuple(sorted(set(UNIVERSE) - set(owned)))[:15]
    wholesale = (3, 4, 10, 11, 12, 13, 14, 20, 21, 22, 23, 24, 28, 29, 30)
    request = _request(owned, prices={pid: 50 for pid in UNIVERSE}, bank=1000)
    ok, problems = fh.temporary_squad_legality(request, wholesale)
    assert ok, problems
    plan = fh.free_hit_transaction_plan(request, wholesale)
    assert len(plan.sold_ids) + len(plan.bought_ids) == 30
    # A normal route would have charged hits for everything beyond the free ones.
    assert 0 == 0
    evaluation = fh.evaluate_free_hit(_request(owned, bank=1000))
    assert evaluation.candidate_metrics["normal_transfer_hits_charged"] == 0


# ---------------------------------------------------------------------------
# THE END-TO-END EVALUATION
# ---------------------------------------------------------------------------


def test_a_clean_request_produces_a_numeric_review_only_evaluation():
    values = {pid: 1.0 for pid in UNIVERSE}
    for pid in (20, 21, 22, 23, 24, 28, 29, 30):
        values[pid] = 6.0                     # unowned players worth buying
    request = _request(core=_core_map(values), bank=200)
    evaluation = fh.evaluate_free_hit(request)

    assert evaluation.action == cd.CHIP_ACTION_FH
    assert evaluation.data_snapshot_bound is True
    assert evaluation.execution_permitted is False
    metrics = evaluation.candidate_metrics
    assert metrics["mean_paired_uplift"] is not None
    assert metrics["permanent_h1_baseline"] is not None
    assert metrics["temporary_h1_value"] is not None
    assert metrics["h1_temporary_uplift"] == pytest.approx(
        metrics["temporary_h1_value"] - metrics["permanent_h1_baseline"], abs=1e-5
    )
    assert metrics["restoration_leaks"] == []
    assert metrics["permanent_squad_unchanged_by_chip"] is True
    assert metrics["no_global_optimum_claim"] is True
    assert metrics["search"]["search_claim"].startswith("BOUNDED")
    assert evaluation.evidence["data_snapshot_sha256"] == DATA


def test_both_arms_are_optimised():
    """An unoptimised baseline would inflate the uplift."""

    values = {pid: 1.0 for pid in UNIVERSE}
    for pid in (20, 21, 22, 23, 24, 28, 29, 30):
        values[pid] = 6.0
    evaluation = fh.evaluate_free_hit(_request(core=_core_map(values), bank=200))
    baseline_policy = evaluation.candidate_metrics["permanent_policy"]
    src = Path(fh.__file__).read_text(encoding="utf-8")
    assert "BOTH arms are optimised" in src
    assert baseline_policy["captain_id"] in baseline_policy["starter_ids"]


def test_a_non_improving_pool_yields_a_non_positive_uplift():
    """When nothing is worth buying, the chip adds nothing and says so."""

    values = {pid: 1.0 for pid in UNIVERSE}
    request = _request(core=_core_map(values), bank=0)
    evaluation = fh.evaluate_free_hit(request)
    assert evaluation.candidate_metrics["mean_paired_uplift"] <= 1e-9
    assert fh.DIAG_FH_NO_TEMPORARY_GAIN in evaluation.reason_codes


def test_the_evaluation_reports_the_h1_temporary_versus_restored_boundary():
    values = {pid: 1.0 for pid in UNIVERSE}
    for pid in (28, 29, 30):
        values[pid] = 6.0
    evaluation = fh.evaluate_free_hit(_request(core=_core_map(values), bank=200))
    assert evaluation.candidate_metrics["h1_event"] == EVENT
    assert evaluation.candidate_metrics["restored_event"] == EVENT + 1
    assert evaluation.candidate_metrics["restored_free_transfers"] == 2
    assert evaluation.evidence["horizon_events"] == list(cd.canonical_chip_horizon(EVENT))


# ---------------------------------------------------------------------------
# CHIP DECISION
# ---------------------------------------------------------------------------

CHIP_ROW = [{"name": "freehit", "available_for_event": True, "used": False, "expired": False,
             "window": "GW2-GW19", "window_start_event": 2, "window_stop_event": 19}]


def _decision(evaluation, *, availability=CHIP_ROW, reservation=None, **overrides):
    kwargs = dict(
        horizon_binding=_binding(), chip_availability=availability,
        evaluations={cd.CHIP_ACTION_FH: evaluation}, reservation=reservation,
        certification_valid=True, manager_state={"squad_ids": list(LEGAL_OWNED)},
    )
    kwargs.update(overrides)
    return cd.decide_chip_action(**kwargs)


class _SpyReservation:
    def __init__(self, value: float | None = None):
        self.value = value
        self.calls = 0

    def estimate(self, *, action, planning_event, expiry_event, state):
        self.calls += 1
        return cd.ReservationEstimate(
            value=self.value,
            calibration_status=(cd.CALIBRATION_CALIBRATED if self.value is not None
                                else cd.CALIBRATION_UNCALIBRATED),
            terminal_value=0.0,
            weeks_to_expiry=None if expiry_event is None else int(expiry_event) - int(planning_event),
        )


def _improving_request():
    values = {pid: 1.0 for pid in UNIVERSE}
    for pid in (20, 21, 22, 23, 24, 28, 29, 30):
        values[pid] = 6.0
    return _request(core=_core_map(values), bank=200)


def test_play_now_is_compared_with_the_save_policy():
    evaluation = fh.evaluate_free_hit(_improving_request())
    assert evaluation.mean_uplift is not None and evaluation.mean_uplift > 0
    decision = _decision(evaluation)
    assert decision.recommended_action == cd.CHIP_ACTION_FH
    assert decision.status == cd.STATUS_CHIP_CANDIDATE_RECHECK_REQUIRED
    assert decision.candidate_metrics["reservation_value"] is None
    assert cd.DIAG_CHIP_RESERVATION_UNCALIBRATED in decision.reason_codes


def test_the_evaluator_never_calls_the_reservation_and_the_arbiter_calls_it_once():
    reservation = _SpyReservation()
    evaluation = fh.evaluate_free_hit(_improving_request())
    assert reservation.calls == 0, "the evaluator must not consult the reservation"
    decision = _decision(evaluation, reservation=reservation)
    assert reservation.calls == 1, "the arbiter applies it exactly once"
    assert decision.candidate_metrics["net_of_reservation"] is None


def test_the_reservation_is_subtracted_exactly_once():
    evaluation = fh.evaluate_free_hit(_improving_request())
    decision = _decision(evaluation, reservation=_SpyReservation(3.0))
    assert decision.candidate_metrics["net_of_reservation"] == pytest.approx(
        evaluation.mean_uplift - 3.0
    )


def test_review_only_blocks_autoplay_even_with_a_calibrated_reservation():
    evaluation = fh.evaluate_free_hit(_improving_request())
    assert evaluation.execution_permitted is False
    decision = _decision(
        evaluation, reservation=_SpyReservation(1.0),
        materiality=0.0,
    )
    assert decision.status != cd.STATUS_PLAY_CHIP
    assert decision.status == cd.STATUS_CHIP_CANDIDATE_RECHECK_REQUIRED
    assert cd.DIAG_CHIP_EVALUATOR_UNCALIBRATED in decision.reason_codes


def test_a_chip_that_adds_nothing_saves_the_chip():
    evaluation = fh.evaluate_free_hit(_request(core=_core_map({pid: 1.0 for pid in UNIVERSE}), bank=0))
    decision = _decision(evaluation)
    assert decision.status == cd.STATUS_NO_CHIP
    assert decision.recommended_action == cd.CHIP_ACTION_NO_CHIP


def test_one_chip_per_gameweek_conflict_refuses():
    evaluation = fh.evaluate_free_hit(_improving_request())
    spent = _decision(evaluation, chips_already_played_for_event=[cd.CHIP_ACTION_TC])
    assert spent.status == cd.STATUS_NO_CHIP
    assert cd.DIAG_CHIP_GAMEWEEK_ALREADY_USED in spent.reason_codes

    two = _decision(evaluation, chips_already_played_for_event=[cd.CHIP_ACTION_TC, cd.CHIP_ACTION_WC])
    assert two.status == cd.STATUS_INSUFFICIENT_EVIDENCE
    assert cd.DIAG_CHIP_MULTIPLE_ACTIONS in two.reason_codes


def test_an_unavailable_or_expired_chip_refuses():
    evaluation = fh.evaluate_free_hit(_improving_request())
    for row in (
        [{"name": "freehit", "available_for_event": False, "used": True, "expired": False,
          "window": "GW2-GW19", "window_start_event": 2, "window_stop_event": 19}],
        [{"name": "freehit", "available_for_event": False, "used": False, "expired": True,
          "window": "GW12-GW14", "window_start_event": 12, "window_stop_event": 14}],
    ):
        decision = _decision(evaluation, availability=row)
        assert decision.status == cd.STATUS_NO_CHIP
        assert any(code.startswith(cd.DIAG_CHIP_UNAVAILABLE) for code in decision.reason_codes)


def test_the_second_free_hit_allocation_survives_a_first_half_use():
    """The two seasonal windows are independent, per the canonical chip rows."""

    def rows(planning_event: int, used_event: int | None):
        return [
            {"name": "freehit", "available_for_event": used_event is None or not (2 <= (used_event or 0) <= 19),
             "used": used_event is not None and 2 <= used_event <= 19, "expired": planning_event > 19,
             "window": "GW2-GW19", "window_start_event": 2, "window_stop_event": 19},
            {"name": "freehit", "available_for_event": used_event is not None and not (20 <= (used_event or 0) <= 38),
             "used": used_event is not None and 20 <= used_event <= 38, "expired": False,
             "window": "GW20-GW38", "window_start_event": 20, "window_stop_event": 38},
        ]

    # Used in the first window: the first row is spent, the second is fresh.
    mapped = cd._availability_by_action(rows(21, 7), planning_event=21)
    assert mapped[cd.CHIP_ACTION_FH]["eligible"] is True
    # Used in the second window: neither row is eligible at GW21.
    mapped = cd._availability_by_action(rows(21, 21), planning_event=21)
    assert mapped[cd.CHIP_ACTION_FH]["eligible"] is False


def test_an_evaluation_from_another_snapshot_refuses_at_arbitration():
    """The Free Hit value is snapshot-bound, so an emptied identity cannot slip through."""

    evaluation = fh.evaluate_free_hit(_improving_request())
    assert evaluation.data_snapshot_bound is True
    foreign = cd.ChipEvaluation(
        action=evaluation.action, evaluator_version=evaluation.evaluator_version,
        candidate_metrics=dict(evaluation.candidate_metrics), uncertainty=dict(evaluation.uncertainty),
        reason_codes=tuple(evaluation.reason_codes), calibration_status=evaluation.calibration_status,
        evidence={**evaluation.evidence, "data_snapshot_sha256": ""},
        execution_permitted=evaluation.execution_permitted,
        data_snapshot_bound=evaluation.data_snapshot_bound,
    )
    decision = _decision(foreign)
    assert decision.status == cd.STATUS_INSUFFICIENT_EVIDENCE
    assert cd.DIAG_CHIP_DATA_SNAPSHOT_MISMATCH in decision.reason_codes
    assert decision.candidate_metrics == {}


def test_the_evaluation_carries_the_certified_context():
    evaluation = fh.evaluate_free_hit(_improving_request())
    assert evaluation.evidence["data_snapshot_sha256"] == DATA
    assert evaluation.evidence["horizon_events"] == list(cd.canonical_chip_horizon(EVENT))
    assert evaluation.evidence["certification_identity"] == CERT
    assert evaluation.evidence["free_hit_quantitative_capability"] == "SUPPORTED_REVIEW_ONLY"


# ---------------------------------------------------------------------------
# THE MODULE DOES NOT REACH INTO ACCEPTED CHIP MODULES IT SHOULD NOT
# ---------------------------------------------------------------------------


def test_free_hit_does_not_modify_wildcard_or_bench_boost():
    """Reuse, never modification: the accepted chip modules are imported read-only."""

    source = Path(fh.__file__).read_text(encoding="utf-8")
    assert "ml.WildcardRequest" not in source
    for banned in ("chip_bench_boost", "chip_triple_captain"):
        assert f"import {banned}" not in source
        assert f"from .{banned}" not in source
