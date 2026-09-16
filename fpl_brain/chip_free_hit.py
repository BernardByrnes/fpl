"""Free Hit evaluator — an H1 TEMPORARY squad over a permanently restored state.

WHAT THE CHIP DOES
------------------
Free Hit lets the manager field a completely different fifteen for ONE Gameweek.
The temporary squad is a WORLD OF ITS OWN:

    PERMANENT world  : the canonical squad, basis, bank and free transfers
                       entering H1
    TEMPORARY world  : a legal fifteen, its H1 lineup/captain, and the residual
                       bank the chip week leaves — H1 ONLY
    RESTORED world   : the permanent world again, at H2

The chip is worth the difference between the temporary H1 score and the score
the PERMANENT squad would have produced in H1.

THE RESTORATION INVARIANT, STRUCTURALLY
---------------------------------------
Temporary Free Hit activity must never rewrite the permanent squad, acquisition
basis or bank.  That is not enforced by remembering to undo something later —
it is enforced by there being NO CODE PATH from a temporary squad to a permanent
one:

    ``restore_permanent_state(permanent, rules=..., restored_event=...)``

takes the permanent state and the season rules and NOTHING ELSE.  It cannot see
the temporary squad, so it cannot leak it; the restored squad, basis and bank
are the permanent ones by construction, and only the free-transfer count moves,
and only because ``season_rules`` says it does.  A test asserts that signature,
so a future edit that reaches for the temporary world fails rather than silently
changing the contract.

Free Hit is therefore NEVER modelled as "unlimited transfers, then reverse
transfers".  That would rewrite bases, erase ownership history and produce
incorrect accounting; the temporary state is simply kept apart.

BUDGET — CANONICAL ECONOMICS ONLY
---------------------------------
The temporary world's cash is

    bank + SUM(canonical selling value of permanently owned players NOT retained)

and its cost is

    SUM(canonical market price of temporarily bought players)

A RETAINED player contributes to neither side, which is the whole point: locked
capital in a retained expensive player cannot be spent twice.  The selling rule
is ``transfer_state.selling_price_tenths`` — never a caller's figure — and the
result is the accepted ``WildcardTransactionPlan`` shape, so the two chips cannot
drift apart on economics.  A temporary purchase basis lives only in the temporary
world and never becomes a permanent basis.

SEARCH — BOUNDED, AND SAID SO
-----------------------------
Screening is EXHAUSTIVE over the official eligible pool: every officially
eligible player with an H1 projection is scored, and a player with no projection
is excluded with a reason rather than treated as zero.  The search then takes a
deterministic greedy seed plus bounded local improvement on a MEAN-BASED
surrogate, and scores the resulting frontier EXACTLY with the certified engine.
The claim is deliberately "best candidate found under bounded Free Hit search",
never "globally optimal squad": the objective is not proven convex and no
exhaustive enumeration over all legal fifteens is performed.

SCORING — THE ACCEPTED ENGINE, NOT A SHORTCUT
---------------------------------------------
H1 is scored with ``manager_lineup.resolve_world`` (legal autosubs, formation
constraints, bench priority, goalkeeper-only goalkeeper substitution) and
``manager_lineup.captain_multiplier`` (captain -> vice -> nobody).  Free Hit does
NOT imply all fifteen score — that is Bench Boost semantics — so the bench
contributes only through legal autosubs, exactly as in a normal Gameweek.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from . import chip_decision as cd
from . import manager_lineup as ml
from . import season_rules as sr
from . import transfer_state as ts
from .candidate_universe import OFFICIAL_PLAYER_POOL_INCOMPLETE
from .chip_wildcard import (
    WildcardPoolBinding,
    WildcardPredictiveIdentity,
    WildcardTransactionPlan,
    WildcardWorldInputs,
)

FREE_HIT_EVALUATOR_VERSION = "chip_fh_v1.0.0"

#: The chip's own official name, for the season-rule helpers.
FREE_HIT_CHIP_NAME = "freehit"

#: Interval half-width constant, matching the engine's near-tie convention.
PAIRED_INTERVAL_K = 1.96

#: Search bounds.  Deterministic and reported; the result is the best candidate
#: found under them, never a claimed global optimum.
FH_POSITION_FRONTIER_SIZE = 12
FH_IMPROVEMENT_PASSES = 6
FH_SQUAD_FRONTIER_SIZE = 8

DIAG_FH_EVALUATED = "CHIP_FH_EVALUATED"
DIAG_FH_REVIEW_ONLY = "CHIP_FH_REVIEW_ONLY_UNCALIBRATED"
DIAG_FH_NO_TEMPORARY_GAIN = "CHIP_FH_TEMPORARY_SQUAD_ADDS_NOTHING"
DIAG_FH_POOL_INCOMPLETE = "CHIP_FH_OFFICIAL_POOL_INCOMPLETE"
DIAG_FH_EXCLUDED_MISSING_PROJECTION = "CHIP_FH_PLAYER_EXCLUDED_MISSING_PROJECTION"
DIAG_FH_BOUNDED_SEARCH = "CHIP_FH_BOUNDED_SEARCH_NO_GLOBAL_OPTIMUM"
DIAG_FH_INPUT_UNCERTAINTY = "CHIP_FH_INPUT_UNCERTAINTY_PROPAGATED"

#: The refusal token set.  Each names ONE broken contract.
FH_MANAGER_STATE_INVALID = "FREE_HIT_MANAGER_STATE_INVALID"
FH_WORLD_INPUTS_MISSING = "FREE_HIT_WORLD_INPUTS_MISSING"
FH_WORLD_INPUTS_MALFORMED = "FREE_HIT_WORLD_INPUTS_MALFORMED"
FH_PROJECTION_INVALID = "FREE_HIT_PROJECTION_INVALID"
FH_PRICING_UNAVAILABLE = "FREE_HIT_PRICING_BASIS_UNAVAILABLE"
FH_ILLEGAL_SQUAD = "FREE_HIT_ILLEGAL_TEMPORARY_SQUAD"
FH_FRONTIER_EMPTY = "FREE_HIT_OPTIMIZER_FRONTIER_EMPTY"
FH_DATA_SNAPSHOT_REQUIRED = "FREE_HIT_DATA_SNAPSHOT_REQUIRED"
FH_PREDICTIVE_IDENTITY_MISMATCH = "FREE_HIT_PREDICTIVE_IDENTITY_MISMATCH"
FH_HORIZON_NOT_CANONICAL = "FREE_HIT_HORIZON_NOT_CANONICAL"
#: The four-GW comparison needs BOTH arms' H2-H4 routes.
FH_TAIL_ROUTE_MISSING = "FREE_HIT_FOUR_GW_TAIL_ROUTE_MISSING"
FH_TAIL_ROUTE_INVALID = "FREE_HIT_FOUR_GW_TAIL_ROUTE_INVALID"
FH_DECISION_AUTHORITY_REQUIRED = "FREE_HIT_CANONICAL_DECISION_AUTHORITY_REQUIRED"
FH_DECISION_AUTHORITY_MISMATCH = "FREE_HIT_PREDICTIVE_EVIDENCE_NOT_THE_CERTIFIED_ONE"
#: The world matrix keyset must BE the authoritative eligible universe.
FH_WORLD_KEYSET_MISMATCH = "FREE_HIT_WORLD_KEYSET_IS_NOT_THE_OFFICIAL_POOL"


class FreeHitError(ValueError):
    """Base class for Free Hit contract violations."""


class FreeHitInputError(FreeHitError):
    """A supplied input does not satisfy the Free Hit contract."""

    def __init__(self, message: str, *, reasons: Sequence[str] = ()) -> None:
        super().__init__(message)
        self.reasons = tuple(str(reason) for reason in reasons)


# ---------------------------------------------------------------------------
# The permanent world — canonical, and the ONLY input to restoration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FreeHitPermanentState:
    """The manager's REAL state entering H1.

    Every field is canonical: the squad, the acquisition basis, the bank, the
    event-start free-transfer bank, and the position/club authority.  A caller's
    own numbers are consistency evidence at best (see the production adapter).
    """

    event: int
    owned_ids: tuple[int, ...]
    purchase_price_tenths: Mapping[int, int]
    bank_tenths: int
    event_start_free_transfers: int
    positions: Mapping[int, str]
    clubs: Mapping[int, int]

    def problems(self) -> list[str]:
        found: list[str] = []
        found.extend(ts.squad_composition_errors(self.owned_ids, self.positions, self.clubs))
        missing_basis = [int(p) for p in self.owned_ids if int(p) not in self.purchase_price_tenths]
        if missing_basis:
            found.append(f"{len(missing_basis)} owned players have no acquisition basis: {missing_basis[:6]}")
        if int(self.bank_tenths) < 0:
            found.append(f"negative bank {self.bank_tenths}")
        if int(self.event_start_free_transfers) < 0:
            found.append(f"negative event-start free transfers {self.event_start_free_transfers}")
        return found


# ---------------------------------------------------------------------------
# The temporary world — H1 only, and never a source of permanent truth
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FreeHitTemporarySquad:
    """A legal fifteen that exists for the chip Gameweek ONLY.

    It carries its own residual bank and its own (temporary) purchase bases, and
    it is deliberately a SEPARATE object from ``FreeHitPermanentState``: nothing
    in this module reads a permanent fact out of it.
    """

    event: int
    squad_ids: tuple[int, ...]
    policy: ml.ManagerPolicy
    remaining_bank_tenths: int
    transaction: WildcardTransactionPlan
    expected_h1_core: float = float("nan")

    def as_dict(self) -> dict[str, Any]:
        return {
            "event": int(self.event),
            "squad_ids": list(self.squad_ids),
            "starter_ids": list(self.policy.sorted_starter_ids()),
            "bench_gk_id": int(self.policy.bench_gk_id),
            "bench_outfield_order": list(self.policy.bench_outfield_order),
            "captain_id": int(self.policy.captain_id),
            "vice_captain_id": int(self.policy.vice_captain_id),
            "remaining_bank_tenths": int(self.remaining_bank_tenths),
            "expected_h1_core": None if math.isnan(self.expected_h1_core) else round(self.expected_h1_core, 6),
            "permanent": False,
        }


@dataclass(frozen=True)
class RestoredPermanentState:
    """The manager's state again at H2 — derived from the PERMANENT world alone."""

    event: int
    owned_ids: tuple[int, ...]
    purchase_price_tenths: Mapping[int, int]
    bank_tenths: int
    free_transfers: int
    free_transfers_rule: str

    def equals_permanent(self, permanent: FreeHitPermanentState) -> list[str]:
        """Every way this restoration differs from the pre-Free-Hit state.

        Only the free-transfer count may differ, and only because
        ``season_rules`` defines the Free Hit transition.  An empty list is the
        restoration invariant holding.
        """

        found: list[str] = []
        if tuple(int(p) for p in self.owned_ids) != tuple(int(p) for p in permanent.owned_ids):
            found.append("squad")
        if {int(k): int(v) for k, v in self.purchase_price_tenths.items()} != {
            int(k): int(v) for k, v in permanent.purchase_price_tenths.items()
        }:
            found.append("acquisition basis")
        if int(self.bank_tenths) != int(permanent.bank_tenths):
            found.append("bank")
        return found


def post_free_hit_ft_state(
    rules: sr.SeasonRules, *, event_start_free_transfers: int | None
) -> int:
    """The H2 free-transfer count, from the ONE canonical season-rule helper.

    Free Hit is a transfer-preserving chip in ``season_rules``; the rule — not
    this module — decides what happens to a saved bank, and it refuses when the
    event-start bank was never recorded rather than inferring one from a
    possibly depleted current count.
    """

    return int(
        sr.free_transfers_after_chip(
            rules, FREE_HIT_CHIP_NAME, event_start_free_transfers=event_start_free_transfers
        )
    )


def restore_permanent_state(
    permanent: FreeHitPermanentState, *, rules: sr.SeasonRules, restored_event: int
) -> RestoredPermanentState:
    """The H2 state.  Takes the PERMANENT world and the rules — nothing else.

    The temporary world is not a parameter, cannot be reached from here, and
    therefore cannot leak: the squad, the acquisition basis and the bank are the
    permanent ones, and the free-transfer count is the canonical season-rule
    outcome.  This signature IS the restoration invariant.
    """

    return RestoredPermanentState(
        event=int(restored_event),
        owned_ids=tuple(int(p) for p in permanent.owned_ids),
        purchase_price_tenths={int(k): int(v) for k, v in permanent.purchase_price_tenths.items()},
        bank_tenths=int(permanent.bank_tenths),
        free_transfers=post_free_hit_ft_state(
            rules, event_start_free_transfers=permanent.event_start_free_transfers
        ),
        free_transfers_rule=str(rules.free_hit_ft_rule),
    )


# ---------------------------------------------------------------------------
# The request
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FreeHitRequest:
    """Everything the Free Hit evaluator may consume — nothing from the model."""

    permanent: FreeHitPermanentState
    horizon_binding: cd.ChipHorizonBinding
    #: H1 predictive worlds.  The artifact carries its OWN event and its own
    #: complete predictive identity; neither is taken from the request.
    h1_worlds: WildcardWorldInputs
    #: The predictive world this decision is authorised against.  The H1 worlds
    #: must BE that world, in every dimension.
    world_identity: WildcardPredictiveIdentity
    #: Canonical position and club per eligible player, and the canonical market
    #: price.  Never a caller's label map.
    positions: Mapping[int, str]
    clubs: Mapping[int, int]
    market_price_tenths: Mapping[int, int]
    #: The official eligible universe this evaluation is exhaustive over.
    pool_binding: WildcardPoolBinding | None = None
    #: The H2-H4 route each arm actually walks, evaluated by the canonical route
    #: engine.  Both are REQUIRED: an H1-only comparison cannot be an exact
    #: four-GW chip decision, because a played Free Hit and an ordinary Gameweek
    #: enter H2 with different free-transfer banks.
    play_tail: FreeHitTailRoute | None = None
    save_tail: FreeHitTailRoute | None = None
    #: The CANONICAL certified decision context every predictive dimension is
    #: anchored to.  Comparing two request-owned identities against each other
    #: proves nothing, so the authority -- not a second supplied object -- is
    #: what the evidence must equal.
    decision_authority: "FreeHitDecisionAuthority | None" = None
    chip_available: bool = True
    rules: sr.SeasonRules = field(default_factory=lambda: sr.SeasonRules(season="2026/27"))
    calibration_status: str = cd.CALIBRATION_UNCALIBRATED
    input_uncertainty_flags: Sequence[str] = field(default_factory=tuple)

    @property
    def planning_event(self) -> int:
        return int(self.horizon_binding.planning_event)

    @property
    def horizon_events(self) -> tuple[int, ...]:
        return tuple(int(event) for event in self.horizon_binding.horizon_events)


# ---------------------------------------------------------------------------
# Canonical economics
# ---------------------------------------------------------------------------


def free_hit_transaction_plan(
    request: FreeHitRequest, squad: Sequence[int]
) -> WildcardTransactionPlan:
    """Temporary-world accounting, in the accepted ``WildcardTransactionPlan`` shape.

    ``cash_available = bank + SUM(canonical selling value of owned players NOT
    retained)`` and ``purchase_cost = SUM(canonical market price of temporarily
    bought players)``.  A retained player is on NEITHER side, so retained capital
    is never counted as spendable cash.  Selling values come from
    ``transfer_state.selling_price_tenths``; a player with no canonical price
    refuses rather than being defaulted.
    """

    target = {int(p) for p in squad}
    owned = {int(p) for p in request.permanent.owned_ids}
    unknown = sorted(pid for pid in target if pid not in request.market_price_tenths)
    if unknown:
        raise FreeHitInputError(
            f"{FH_PRICING_UNAVAILABLE}: no canonical market price for temporary squad member(s) "
            f"{unknown[:8]}",
            reasons=(FH_PRICING_UNAVAILABLE,),
        )
    sold = tuple(sorted(owned - target))
    bought = tuple(sorted(target - owned))

    cash = int(request.permanent.bank_tenths)
    for pid in sold:
        cash += canonical_selling_value(request, pid)
    cost = sum(int(request.market_price_tenths[pid]) for pid in bought)
    return WildcardTransactionPlan(
        retained_ids=tuple(sorted(owned & target)),
        sold_ids=sold,
        bought_ids=bought,
        rebought_ids=(),
        cash_available_tenths=int(cash),
        purchase_cost_tenths=int(cost),
        remaining_bank_tenths=int(cash) - int(cost),
        # A TEMPORARY basis: it belongs to the chip week and is never written
        # back over a permanent acquisition basis.
        new_basis={pid: int(request.market_price_tenths[pid]) for pid in bought},
    )


def canonical_selling_value(request: FreeHitRequest, player_id: int) -> int:
    """The canonical selling value of a permanently owned player.

    The rule is ``transfer_state.selling_price_tenths``; this function never
    invents one and refuses when the acquisition basis or the market price is
    unknown.
    """

    pid = int(player_id)
    purchase = request.permanent.purchase_price_tenths.get(pid)
    if purchase is None:
        raise FreeHitInputError(
            f"{FH_PRICING_UNAVAILABLE}: no acquisition basis for permanently owned player {pid}",
            reasons=(FH_PRICING_UNAVAILABLE,),
        )
    market = request.market_price_tenths.get(pid)
    if market is None:
        raise FreeHitInputError(
            f"{FH_PRICING_UNAVAILABLE}: no canonical market price for permanently owned player {pid}",
            reasons=(FH_PRICING_UNAVAILABLE,),
        )
    return int(ts.selling_price_tenths(int(purchase), int(market)))


def temporary_squad_legality(request: FreeHitRequest, squad: Sequence[int]) -> tuple[bool, list[str]]:
    """Canonical legality of a temporary fifteen: composition, clubs, budget.

    Composition and the club limit come from ``transfer_state``'s single shared
    rule; affordability comes from the canonical transaction plan.  Positions and
    clubs are the canonical maps, never a caller's labels.
    """

    problems: list[str] = []
    ids = [int(p) for p in squad]
    positions = {pid: str(request.positions.get(pid) or "") for pid in ids}
    clubs = {pid: int(request.clubs.get(pid) or 0) for pid in ids}
    problems.extend(ts.squad_composition_errors(ids, positions, clubs))
    try:
        plan = free_hit_transaction_plan(request, ids)
    except FreeHitInputError as exc:
        problems.append(str(exc))
        return False, problems
    if plan.remaining_bank_tenths < 0:
        problems.append(
            f"over budget by {-plan.remaining_bank_tenths} tenths "
            f"(cash {plan.cash_available_tenths}, cost {plan.purchase_cost_tenths})"
        )
    return (not problems), problems


# ---------------------------------------------------------------------------
# H1 scoring — the accepted engine
# ---------------------------------------------------------------------------


def _h1_scores(request: FreeHitRequest) -> dict[int, dict[int, float]]:
    """Per-player per-world H1 minutes and core, for the eligible universe."""

    worlds = request.h1_worlds
    return {
        "minutes": {int(pid): [float(v) for v in worlds.minutes[int(pid)]] for pid in worlds.player_ids},
        "core": {int(pid): [float(v) for v in worlds.core[int(pid)]] for pid in worlds.player_ids},
    }


def mean_h1_core(request: FreeHitRequest, player_id: int) -> float:
    series = request.h1_worlds.core.get(int(player_id))
    if not series:
        return float("nan")
    return float(sum(float(v) for v in series) / len(series))


def best_h1_policy(
    request: FreeHitRequest, squad: Sequence[int]
) -> ml.ManagerPolicy | None:
    """The best legal H1 policy for a squad, by mean core — deterministic.

    Mirrors the accepted Wildcard ``_best_xi`` construction: the strongest legal
    eleven by mean core (one goalkeeper, then the best legal outfield
    complement), the bench ordered by mean core, and the armband on the strongest
    starter with the vice on the next.  Selecting an XI is a management decision;
    its VALUE is then resolved by the certified engine over the worlds rather
    than approximated here.
    """

    ids = [int(p) for p in squad]
    ranks = sorted(ids, key=lambda pid: (-mean_h1_core(request, pid), pid))
    positions = {pid: str(request.positions.get(pid) or "") for pid in ids}
    gks = [pid for pid in ranks if positions[pid] == "GKP"]
    if len(gks) != 2:
        return None
    starters_gk = gks[0]
    bench_gk = gks[1]
    outfield = [pid for pid in ranks if positions[pid] != "GKP"]
    best: tuple[float, list[int]] | None = None
    counts = [
        (defenders, midfielders, forwards)
        for defenders in range(ml.FORMATION_MIN["DEF"], ml.FORMATION_MAX["DEF"] + 1)
        for midfielders in range(ml.FORMATION_MIN["MID"], ml.FORMATION_MAX["MID"] + 1)
        for forwards in range(ml.FORMATION_MIN["FWD"], ml.FORMATION_MAX["FWD"] + 1)
        if defenders + midfielders + forwards == ml.XI_SIZE - 1
    ]
    for defenders, midfielders, forwards in sorted(counts):
        selection = [starters_gk]
        for position, needed in (("DEF", defenders), ("MID", midfielders), ("FWD", forwards)):
            pool = [pid for pid in outfield if positions[pid] == position][:needed]
            if len(pool) != needed:
                break
            selection += pool
        if len(selection) != ml.XI_SIZE:
            continue
        total = sum(mean_h1_core(request, pid) for pid in selection)
        key = sorted(selection)
        if best is None or total > best[0] + 1e-12 or (abs(total - best[0]) <= 1e-12 and key < best[1]):
            best = (total, key)
    if best is None:
        return None
    starter_ids = tuple(best[1])
    bench = [pid for pid in outfield if pid not in set(starter_ids)]
    ordered = tuple(sorted(bench, key=lambda pid: (-mean_h1_core(request, pid), pid)))
    armband = sorted(starter_ids, key=lambda pid: (-mean_h1_core(request, pid), pid))
    policy = ml.ManagerPolicy(
        starter_ids=starter_ids,
        bench_gk_id=int(bench_gk),
        bench_outfield_order=ordered,
        captain_id=int(armband[0]),
        vice_captain_id=int(armband[1]),
    )
    if ml.policy_legality_errors(policy, positions):
        return None
    return policy


def h1_core_series(request: FreeHitRequest, policy: ml.ManagerPolicy) -> list[float]:
    """Per-world H1 total under the CERTIFIED engine.

    Normal (non-Bench-Boost) scoring: only the resolved counted XI scores, so the
    four bench slots contribute solely through legal autosubs.  Free Hit does not
    imply all fifteen score.
    """

    squad = tuple(sorted(ml.policy_player_ids(policy)))
    positions = {pid: str(request.positions.get(pid) or "") for pid in squad}
    scores = _h1_scores(request)
    worlds = int(request.h1_worlds.worlds)
    series: list[float] = []
    for world in range(worlds):
        w_minutes = {pid: scores["minutes"][pid][world] for pid in squad}
        w_core = {pid: scores["core"][pid][world] for pid in squad}
        outcome = ml.resolve_world(
            policy, positions, w_minutes, w_core, require_player_ids=squad
        )
        extra, _ = ml.captain_multiplier(policy, w_minutes, w_core, require_player_ids=squad)
        series.append(float(sum(w_core[pid] for pid in outcome.counted_ids) + float(extra)))
    return series


def free_hit_h1_value(request: FreeHitRequest, squad: Sequence[int]) -> tuple[float, ml.ManagerPolicy | None, list[float]]:
    """The exact certified H1 expected core of a temporary squad, and its series."""

    policy = best_h1_policy(request, squad)
    if policy is None:
        return float("nan"), None, []
    series = h1_core_series(request, policy)
    return (float(sum(series) / len(series)) if series else float("nan")), policy, series


# ---------------------------------------------------------------------------
# The bounded search
# ---------------------------------------------------------------------------


def screen_players(request: FreeHitRequest) -> tuple[dict[int, float], dict[str, Any]]:
    """Exhaustive screening over the official pool; no filtering by popularity.

    A player with no H1 projection is EXCLUDED WITH A REASON, never scored as
    zero.  Only officially eligible ids are screened, so a rogue id cannot enter
    the search at all.
    """

    eligible = sorted(int(p) for p in (request.pool_binding.eligible_ids if request.pool_binding else ()))
    if not eligible:
        eligible = sorted(int(p) for p in request.h1_worlds.player_ids)
    eligible_set = set(eligible)
    # A player in the matrix who is NOT officially eligible is excluded; a player
    # in the official pool with NO projection is excluded with a reason.  Both are
    # counted so the screen is auditable, and neither is ever scored as zero.
    excluded_not_eligible = sum(
        1 for pid in request.h1_worlds.player_ids if int(pid) not in eligible_set
    )
    scores: dict[int, float] = {}
    missing: list[int] = []
    for pid in eligible:
        value = mean_h1_core(request, pid)
        if not math.isfinite(value):
            missing.append(int(pid))
            continue
        scores[int(pid)] = value
    return scores, {
        "eligible_pool": len(eligible),
        "screened": len(scores),
        "excluded_missing_projection": len(missing),
        "excluded_missing_projection_ids": missing[:16],
        "excluded_not_officially_eligible": int(excluded_not_eligible),
        "missing_projection_policy": "EXCLUDED_WITH_REASON",
    }


def _position_frontiers(
    request: FreeHitRequest, scores: Mapping[int, float], *, size: int
) -> dict[str, list[int]]:
    by_position: dict[str, list[int]] = defaultdict(list)
    for pid, _score in scores.items():
        by_position[str(request.positions.get(int(pid)) or "")].append(int(pid))
    frontiers: dict[str, list[int]] = {}
    for position in ("GKP", *ts.OUTFIELD_POSITIONS):
        candidates = sorted(by_position.get(position, []), key=lambda pid: (-scores[pid], pid))
        frontiers[position] = candidates[: max(size, ts.POSITION_COMPOSITION.get(position, 0))]
    return frontiers


def _affordability_allowance(request: FreeHitRequest) -> int:
    """Cash available if EVERY permanently owned player were sold.

    The greedy seed charges a retained player his CANONICAL SELLING VALUE, not
    zero: retaining a player forfeits exactly that cash, so charging zero is how
    locked capital gets spent twice.  With that charging rule this allowance is
    exactly equivalent to the canonical transaction plan.
    """

    total = int(request.permanent.bank_tenths)
    for pid in request.permanent.owned_ids:
        total += canonical_selling_value(request, int(pid))
    return total


def _seed_squad(
    request: FreeHitRequest, scores: Mapping[int, float], frontiers: Mapping[str, Sequence[int]]
) -> list[int] | None:
    """Deterministic greedy seed filling the canonical composition within budget."""

    allowance = _affordability_allowance(request)
    owned = {int(p) for p in request.permanent.owned_ids}
    squad: list[int] = []
    clubs: dict[int, int] = defaultdict(int)
    spent = 0
    for position, required in ts.POSITION_COMPOSITION.items():
        candidates = sorted(
            (pid for pid in frontiers.get(position, ()) if pid in scores),
            key=lambda pid: (-scores[pid], pid),
        )
        picked = 0
        for pid in candidates:
            if picked >= required:
                break
            club = int(request.clubs.get(pid) or 0)
            if clubs[club] >= ts.SQUAD_TEAM_LIMIT:
                continue
            price = (
                canonical_selling_value(request, pid)
                if pid in owned
                else int(request.market_price_tenths.get(pid) or 0)
            )
            if pid not in owned and pid not in request.market_price_tenths:
                continue
            if spent + price > allowance:
                continue
            squad.append(pid)
            clubs[club] += 1
            spent += price
            picked += 1
        if picked < required:
            return None
    return squad if len(squad) == ts.SQUAD_SIZE else None


def _surrogate(request: FreeHitRequest, squad: Sequence[int]) -> float:
    """A cheap MEAN-BASED objective for the search: no world loop.

    It ranks candidate fifteens; it is never reported as the Free Hit value.  The
    reported value is always the exact certified H1 expected core.
    """

    policy = best_h1_policy(request, squad)
    if policy is None:
        return float("-inf")
    xi = sum(mean_h1_core(request, pid) for pid in policy.starter_ids)
    armband = sorted(policy.starter_ids, key=lambda pid: (-mean_h1_core(request, pid), pid))
    return float(xi + mean_h1_core(request, armband[0]))


def _improve(
    request: FreeHitRequest, squad: list[int], *, passes: int, universe: Sequence[int]
) -> list[int]:
    """Bounded local improvement: only strictly-improving, still-legal swaps."""

    current = list(squad)
    current_value = _surrogate(request, current)
    for _ in range(max(0, passes)):
        best_gain = 0.0
        best_swap: tuple[int, int] | None = None
        ordered_universe = sorted(int(p) for p in universe)
        for index in range(len(current)):
            for candidate in ordered_universe:
                if candidate in current:
                    continue
                trial = list(current)
                trial[index] = candidate
                ok, _ = temporary_squad_legality(request, trial)
                if not ok:
                    continue
                gain = _surrogate(request, trial) - current_value
                if gain > best_gain + 1e-12 or (
                    abs(gain - best_gain) <= 1e-12 and best_swap is not None and candidate < best_swap[1]
                ):
                    best_gain = gain
                    best_swap = (index, candidate)
        if best_swap is None or best_gain <= 1e-12:
            break
        current[best_swap[0]] = best_swap[1]
        current_value += best_gain
    return current


def optimize_free_hit_squad(
    request: FreeHitRequest, *, frontier_size: int = FH_SQUAD_FRONTIER_SIZE
) -> tuple[FreeHitTemporarySquad, dict[str, Any]]:
    """The best temporary fifteen found under the BOUNDED Free Hit search.

    Not a global optimum: screening is exhaustive over the official pool, the
    seed is greedy, improvement is bounded, and the frontier is a fixed size.
    """

    scores, screen_stats = screen_players(request)
    if not scores:
        raise FreeHitInputError(
            f"{FH_FRONTIER_EMPTY}: no officially eligible player has an H1 projection",
            reasons=(FH_FRONTIER_EMPTY,),
        )
    frontiers = _position_frontiers(request, scores, size=FH_POSITION_FRONTIER_SIZE)
    seed = _seed_squad(request, scores, frontiers)
    if seed is None:
        raise FreeHitInputError(
            f"{FH_FRONTIER_EMPTY}: no affordable legal fifteen could be seeded from the official pool",
            reasons=(FH_FRONTIER_EMPTY,),
        )
    universe = tuple(sorted(scores))
    improved = _improve(request, seed, passes=FH_IMPROVEMENT_PASSES, universe=universe)

    # A small deterministic frontier: the improved squad plus its best
    # single-position-restricted neighbours, scored EXACTLY.
    candidates: list[tuple[float, list[int]]] = []
    seen: set[tuple[int, ...]] = set()
    for candidate in (improved, seed):
        key = tuple(sorted(candidate))
        if key in seen:
            continue
        seen.add(key)
        candidates.append((_surrogate(request, candidate), list(candidate)))
    for position in ("GKP", *ts.OUTFIELD_POSITIONS):
        for pid in frontiers.get(position, [])[:frontier_size]:
            if pid in improved:
                continue
            trial = list(improved)
            replaceable = [i for i, other in enumerate(trial) if request.positions.get(other) == position]
            if not replaceable:
                continue
            trial[replaceable[-1]] = pid
            ok, _ = temporary_squad_legality(request, trial)
            if not ok:
                continue
            key = tuple(sorted(trial))
            if key in seen:
                continue
            seen.add(key)
            candidates.append((_surrogate(request, trial), trial))

    candidates.sort(key=lambda row: (-row[0], tuple(sorted(row[1]))))
    best: FreeHitTemporarySquad | None = None
    exact_scored: list[dict[str, Any]] = []
    for _score, squad in candidates[: max(1, int(frontier_size))]:
        value, policy, series = free_hit_h1_value(request, squad)
        if policy is None or not math.isfinite(value):
            continue
        plan = free_hit_transaction_plan(request, squad)
        exact_scored.append(
            {"squad": list(sorted(squad)), "expected_h1_core": round(value, 6), "worlds": len(series)}
        )
        temporary = FreeHitTemporarySquad(
            event=int(request.planning_event),
            squad_ids=tuple(sorted(int(p) for p in squad)),
            policy=policy,
            remaining_bank_tenths=int(plan.remaining_bank_tenths),
            transaction=plan,
            expected_h1_core=float(value),
        )
        if best is None or value > best.expected_h1_core + 1e-12 or (
            abs(value - best.expected_h1_core) <= 1e-12
            and temporary.squad_ids < best.squad_ids
        ):
            best = temporary
    if best is None:
        raise FreeHitInputError(
            f"{FH_FRONTIER_EMPTY}: no candidate fifteen produced a finite H1 value",
            reasons=(FH_FRONTIER_EMPTY,),
        )
    return best, {
        **screen_stats,
        "universe_players": len(universe),
        "seed_squad": sorted(int(p) for p in seed),
        "improved_squad": sorted(int(p) for p in improved),
        "frontier_candidates": len(candidates),
        "exact_scored": exact_scored[: int(frontier_size)],
        "position_frontier_size": int(FH_POSITION_FRONTIER_SIZE),
        "improvement_passes": int(FH_IMPROVEMENT_PASSES),
        "search_claim": "BOUNDED_SINGLE_GAMEWEEK_SEARCH_NO_GLOBAL_OPTIMUM_CLAIM",
    }




# ---------------------------------------------------------------------------
# Contract validation
# ---------------------------------------------------------------------------


def worlds_identity_of(request: "FreeHitRequest"):
    """The world matrix's OWN identity -- never the request's companion object."""

    return getattr(request.h1_worlds, "identity", None)


def _refuse(token: str, detail: str) -> cd.ChipEvaluation:
    """A refusal shaped like the arbiter's ``ChipEvaluation``, with no number."""

    return cd.ChipEvaluation(
        action=cd.CHIP_ACTION_FH,
        evaluator_version=FREE_HIT_EVALUATOR_VERSION,
        candidate_metrics={"mean_paired_uplift": None},
        uncertainty={},
        reason_codes=(token,),
        calibration_status=cd.CALIBRATION_UNCALIBRATED,
        evidence={
            "refusal_detail": str(detail),
            "free_hit_evaluator_version": FREE_HIT_EVALUATOR_VERSION,
            "executable": False,
            "actionable": False,
        },
        execution_permitted=False,
        data_snapshot_bound=True,
    )


def contract_problems(request: FreeHitRequest) -> list[str]:
    """Every way the request breaks the certified Free Hit contract.

    The predictive identity is checked in EVERY dimension the accepted Wildcard
    authority carries, and an EMPTY identity is a refusal rather than a skipped
    comparison: the H1 number must be attributable to the capture that produced
    it, and the artifact's identity is its own evidence -- it is never stamped on
    from the request or inferred from the binding.
    """

    problems: list[str] = []
    binding = request.horizon_binding
    events = tuple(int(event) for event in binding.horizon_events)
    canonical = cd.canonical_chip_horizon(int(binding.planning_event))
    if events != canonical:
        problems.append(f"{FH_HORIZON_NOT_CANONICAL}: horizon {list(events)} is not {list(canonical)}")

    authority = request.decision_authority
    if authority is None:
        problems.append(
            f"{FH_DECISION_AUTHORITY_REQUIRED}: no canonical certified decision authority is bound, so "
            "the predictive evidence has nothing to be anchored to"
        )
    else:
        for problem in authority.problems():
            problems.append(f"{FH_DECISION_AUTHORITY_REQUIRED}: {problem}")
        # BOTH the bound identity and the world's identity must be the CERTIFIED
        # one.  Requiring only that they agree with each other is exactly the
        # self-certification this closes.
        for label, supplied in (("bound identity", identity := request.world_identity),
                                ("world matrix identity", worlds_identity_of(request))):
            for dimension in authority.disagreements_with(supplied, label=label):
                problems.append(f"{FH_DECISION_AUTHORITY_MISMATCH}: {dimension}")
        if str(binding.certification_identity) != str(authority.certification_identity):
            problems.append(
                f"{FH_DECISION_AUTHORITY_MISMATCH}: the horizon binding's certification identity is not "
                "the certified one"
            )

    binding_snapshot = str(binding.data_snapshot_sha256 or "").strip()
    identity = request.world_identity
    if not binding_snapshot:
        problems.append(f"{FH_DATA_SNAPSHOT_REQUIRED}: the horizon binding carries no data snapshot identity")
    if identity is None:
        problems.append(f"{FH_PREDICTIVE_IDENTITY_MISMATCH}: the request declares no predictive identity")
    else:
        for problem in identity.problems():
            problems.append(f"{FH_PREDICTIVE_IDENTITY_MISMATCH}: bound identity {problem}")
        if binding_snapshot and str(identity.data_snapshot_sha256 or "").strip():
            if str(identity.data_snapshot_sha256) != binding_snapshot:
                problems.append(
                    f"{FH_PREDICTIVE_IDENTITY_MISMATCH}: the bound predictive identity is not the "
                    "binding's data snapshot"
                )

    worlds = request.h1_worlds
    if int(worlds.event) != int(binding.planning_event):
        problems.append(
            f"{FH_WORLD_INPUTS_MALFORMED}: the H1 world matrix carries event {int(worlds.event)}, "
            f"not the planning event {int(binding.planning_event)}"
        )
    # A MISSING snapshot is decided first: it is not "disagrees with the binding",
    # it is absent, and it must carry the snapshot token rather than the generic
    # identity-completeness one.
    world_snapshot = ""
    if worlds.identity is not None:
        world_snapshot = str(worlds.identity.data_snapshot_sha256 or "").strip()
    if worlds.identity is None:
        problems.append(f"{FH_PREDICTIVE_IDENTITY_MISMATCH}: the world matrix carries no predictive identity")
    elif not world_snapshot:
        problems.append(f"{FH_DATA_SNAPSHOT_REQUIRED}: the H1 world matrix carries no data snapshot identity")
    elif binding_snapshot and world_snapshot != binding_snapshot:
        problems.append(
            f"{FH_DATA_SNAPSHOT_REQUIRED}: the H1 world matrix was produced from a different capture "
            "than the certified binding authorises"
        )

    if worlds.identity is not None:
        for problem in worlds.identity.problems():
            if problem.strip() == "data_snapshot_sha256 is empty":
                continue          # already reported as a snapshot problem above
            problems.append(f"{FH_PREDICTIVE_IDENTITY_MISMATCH}: world identity {problem}")

    if worlds.identity is not None and identity is not None and world_snapshot:
        for dimension in worlds.identity.disagreements_with(identity):
            problems.append(
                f"{FH_PREDICTIVE_IDENTITY_MISMATCH}: the world matrix disagrees with the bound "
                f"predictive world on {dimension}"
            )

    for problem in worlds.problems():
        problems.append(f"{FH_WORLD_INPUTS_MALFORMED}: {problem}")

    # The WORLD KEYSET must BE the authoritative eligible universe, exactly.  A
    # missing official player has no predictive support and an extra id does not
    # officially exist; neither may be turned into a screening exclusion, because
    # that is how a contradictory universe silently becomes a smaller search.
    if request.pool_binding is not None:
        eligible = {int(p) for p in request.pool_binding.eligible_ids}
        world_ids = {int(p) for p in worlds.player_ids}
        missing = sorted(eligible - world_ids)
        extra = sorted(world_ids - eligible)
        if missing:
            problems.append(
                f"{FH_WORLD_KEYSET_MISMATCH}: the world matrix omits {len(missing)} officially "
                f"eligible player(s): {missing[:8]}"
            )
        if extra:
            problems.append(
                f"{FH_WORLD_KEYSET_MISMATCH}: the world matrix carries {len(extra)} non-official "
                f"player(s): {extra[:8]}"
            )

    play_tail = request.play_tail
    save_tail = request.save_tail
    if play_tail is None or save_tail is None:
        problems.append(
            f"{FH_TAIL_ROUTE_MISSING}: both arms need their canonical H2-H4 route; an H1-only "
            "comparison is not an exact four-GW chip decision"
        )
    else:
        tail_events = list(events)[1:]
        restored = restore_permanent_state(
            request.permanent, rules=request.rules, restored_event=int(binding.planning_event) + 1
        )
        save_state = save_arm_h2_state(request)
        problems.extend(
            f"{FH_TAIL_ROUTE_INVALID}: {problem}"
            for problem in play_tail.problems(
                expected_arm=FreeHitTailRoute.FREE_HIT_ARM_PLAY, expected_events=tail_events,
                expected_squad=request.permanent.owned_ids, expected_bank=restored.bank_tenths,
                expected_free_transfers=restored.free_transfers,
            )
        )
        problems.extend(
            f"{FH_TAIL_ROUTE_INVALID}: {problem}"
            for problem in save_tail.problems(
                expected_arm=FreeHitTailRoute.FREE_HIT_ARM_SAVE, expected_events=tail_events,
                expected_squad=save_state.owned_ids, expected_bank=save_state.bank_tenths,
                expected_free_transfers=save_state.free_transfers,
            )
        )

    if int(request.permanent.event) != int(binding.planning_event):
        problems.append(
            f"{FH_MANAGER_STATE_INVALID}: the permanent state is for GW{int(request.permanent.event)}, "
            f"not the planning event GW{int(binding.planning_event)}"
        )
    for problem in request.permanent.problems():
        problems.append(f"{FH_MANAGER_STATE_INVALID}: {problem}")

    if request.pool_binding is None:
        problems.append(f"{DIAG_FH_POOL_INCOMPLETE}: no official eligible-player pool is bound")
    else:
        for problem in request.pool_binding.problems():
            problems.append(f"{DIAG_FH_POOL_INCOMPLETE}: {problem}")

    # An owned player with no point-in-time price has no canonical selling value,
    # so the temporary budget cannot be computed at all.
    missing_owned = sorted(
        int(p) for p in request.permanent.owned_ids if int(p) not in request.market_price_tenths
    )
    if missing_owned:
        problems.append(
            f"{FH_PRICING_UNAVAILABLE}: no canonical price for owned player(s) {missing_owned[:8]}"
        )
    return problems


# ---------------------------------------------------------------------------
# The evaluation
# ---------------------------------------------------------------------------


def evaluate_free_hit(request: FreeHitRequest) -> cd.ChipEvaluation:
    """PLAY_FH_NOW vs SAVE_FH -> the arbiter's ``ChipEvaluation`` shape.

    The chip's own effect is H1 only: the temporary squad exists for one
    Gameweek, so H2-H4 are the RESTORED PERMANENT state in BOTH arms and that
    consequence is identical across them.  The horizon must therefore still be
    the exact canonical four events, and the decision is the paired H1
    difference.

    BOTH arms are optimised: the temporary fifteen by the bounded Free Hit
    search, and the permanent fifteen by the same best-legal-policy rule.  Using
    an unoptimised baseline would inflate the reported uplift.

    The reservation is the chip arbiter's seam.  This evaluator calls it ZERO
    times and never subtracts a future-opportunity term itself.
    """

    problems = contract_problems(request)
    if problems:
        first = problems[0]
        token = FH_PROJECTION_INVALID
        for candidate in (
            FH_HORIZON_NOT_CANONICAL, FH_DATA_SNAPSHOT_REQUIRED, FH_PREDICTIVE_IDENTITY_MISMATCH,
            FH_WORLD_KEYSET_MISMATCH, FH_WORLD_INPUTS_MALFORMED, FH_MANAGER_STATE_INVALID,
            FH_PRICING_UNAVAILABLE, DIAG_FH_POOL_INCOMPLETE, FH_DECISION_AUTHORITY_REQUIRED,
            FH_DECISION_AUTHORITY_MISMATCH, FH_TAIL_ROUTE_MISSING, FH_TAIL_ROUTE_INVALID,
        ):
            if first.startswith(candidate):
                token = candidate
                break
        return _refuse(token, "; ".join(problems[:8]))

    try:
        temporary, search_stats = optimize_free_hit_squad(request)
    except FreeHitInputError as exc:
        return _refuse((exc.reasons[0] if exc.reasons else FH_FRONTIER_EMPTY), str(exc))

    permanent_ids = tuple(int(p) for p in request.permanent.owned_ids)
    baseline_value, baseline_policy, baseline_series = free_hit_h1_value(request, permanent_ids)
    if baseline_policy is None or not math.isfinite(baseline_value):
        return _refuse(
            FH_ILLEGAL_SQUAD,
            "the permanent fifteen has no legal H1 policy under the certified worlds",
        )

    temporary_series = h1_core_series(request, temporary.policy)
    paired = [
        float(temporary_series[index]) - float(baseline_series[index])
        for index in range(min(len(temporary_series), len(baseline_series)))
    ]
    h1_uplift = float(sum(paired) / len(paired)) if paired else float("nan")
    # The H2-H4 arms are valued on their OWN actual states -- the Free Hit arm
    # from the restored permanent state, the SAVE arm from ordinary progression --
    # so the free-transfer divergence between them enters the decision instead of
    # being assumed away.
    arms = four_gw_arm_values(
        request, h1_temporary_value=float(temporary.expected_h1_core), h1_permanent_value=float(baseline_value)
    )
    mean_uplift = float(h1_uplift) + float(arms["tail_delta"])
    paired_se = _std(paired) / math.sqrt(len(paired)) if len(paired) > 1 else 0.0
    ordered = sorted(paired)

    restored = restore_permanent_state(
        request.permanent,
        rules=request.rules,
        restored_event=int(request.planning_event) + 1,
    )
    leaks = restored.equals_permanent(request.permanent)

    flags: list[str] = [DIAG_FH_EVALUATED, DIAG_FH_REVIEW_ONLY, DIAG_FH_BOUNDED_SEARCH]
    if search_stats.get("excluded_missing_projection"):
        flags.append(DIAG_FH_EXCLUDED_MISSING_PROJECTION)
    if mean_uplift <= 0.0:
        flags.append(DIAG_FH_NO_TEMPORARY_GAIN)
    propagated = tuple(str(flag) for flag in request.input_uncertainty_flags)
    if propagated:
        flags.append(DIAG_FH_INPUT_UNCERTAINTY)

    return cd.ChipEvaluation(
        action=cd.CHIP_ACTION_FH,
        evaluator_version=FREE_HIT_EVALUATOR_VERSION,
        candidate_metrics={
            "mean_paired_uplift": round(mean_uplift, 6),
            "h1_paired_uplift": round(h1_uplift, 6),
            "permanent_h1_baseline": round(baseline_value, 6),
            "temporary_h1_value": round(temporary.expected_h1_core, 6),
            "h1_temporary_uplift": round(
                float(temporary.expected_h1_core) - float(baseline_value), 6
            ),
            # The four-GW decomposition: H1 temporary vs permanent, then the two
            # ACTUAL H2-H4 tails, then the totals the chip is decided on.
            **{key: round(value, 6) for key, value in arms.items()},
            "play_tail_events": (request.play_tail.event_values() if request.play_tail else []),
            "save_tail_events": (request.save_tail.event_values() if request.save_tail else []),
            "play_h2_state": (request.play_tail.h2_state() if request.play_tail else {}),
            "save_h2_state": (request.save_tail.h2_state() if request.save_tail else {}),
            "paired_se": round(paired_se, 6),
            "value_basis": "CORE_POINTS",
            "temporary_squad": list(temporary.squad_ids),
            "temporary_policy": temporary.policy.as_dict(),
            "permanent_policy": baseline_policy.as_dict(),
            "temporary_remaining_bank_tenths": int(temporary.remaining_bank_tenths),
            "temporary_sold_ids": list(temporary.transaction.sold_ids),
            "temporary_bought_ids": list(temporary.transaction.bought_ids),
            "temporary_basis_is_temporary": True,
            "permanent_squad_unchanged_by_chip": True,
            "normal_transfer_hits_charged": 0,
            "four_gw_basis": (
                "Free Hit alters the H1 squad AND the free-transfer bank the manager carries into H2: "
                "a played chip preserves the saved bank while an ordinary Gameweek accrues one, so the "
                "two arms are valued on their own restored/ordinarily-progressed H2 states through H4 "
                "rather than assumed equal"
            ),
            "arms_valued_separately": True,
            "horizon_events": list(request.horizon_events),
            "h1_event": int(request.planning_event),
            "restored_event": int(restored.event),
            "restored_free_transfers": int(restored.free_transfers),
            "restored_free_transfers_rule": str(restored.free_transfers_rule),
            "restoration_leaks": list(leaks),
            "chip_available": bool(request.chip_available),
            "search": dict(search_stats),
            "no_global_optimum_claim": True,
        },
        uncertainty={
            "paired_interval_low": round(mean_uplift - PAIRED_INTERVAL_K * paired_se, 6),
            "paired_interval_high": round(mean_uplift + PAIRED_INTERVAL_K * paired_se, 6),
            "paired_quantile_05": round(_quantile(ordered, 0.05), 6),
            "paired_quantile_50": round(_quantile(ordered, 0.50), 6),
            "paired_quantile_95": round(_quantile(ordered, 0.95), 6),
            "interval_k": PAIRED_INTERVAL_K,
            "worlds": int(request.h1_worlds.worlds),
            "input_uncertainty_flags": list(propagated),
        },
        reason_codes=tuple(sorted(flags)),
        calibration_status=request.calibration_status,
        evidence={
            "certification_identity": str(request.horizon_binding.certification_identity),
            "data_snapshot_sha256": str(request.horizon_binding.data_snapshot_sha256 or ""),
            "horizon_events": list(request.horizon_events),
            "planning_event": int(request.planning_event),
            "predictive_identity": request.world_identity.as_dict(),
            "official_player_pool": request.pool_binding.as_dict() if request.pool_binding else {},
            "free_hit_evaluator_version": FREE_HIT_EVALUATOR_VERSION,
            "free_hit_quantitative_capability": "SUPPORTED_REVIEW_ONLY",
            "executable": False,
            "actionable": False,
        },
        # REVIEW-ONLY: no calibrated Free Hit execution model exists, so a
        # positive uplift may be reported but can never execute on.
        execution_permitted=False,
        data_snapshot_bound=True,
    )


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def _std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    index = min(len(sorted_values) - 1, max(0, int(round(q * (len(sorted_values) - 1)))))
    return float(sorted_values[index])

# ---------------------------------------------------------------------------
# THE H2-H4 TAIL — the four-GW comparison is not an H1-only one
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FreeHitTailRoute:
    """One ARM's H2-H4 route, evaluated by the canonical route engine.

    Free Hit's effect is not confined to H1.  A played Free Hit PRESERVES the
    saved free-transfer bank while an ordinary Gameweek ACCRUES one, so the two
    arms enter H2 with genuinely different states even when neither squad
    changes -- and that difference propagates through H2-H4 route value.  An
    H1-only comparison that asserts "H2-H4 are identical across arms" is
    therefore wrong, and this type is how the real difference is carried.

    ``events`` are the route engine's OWN per-event results.  Each
    ``mean_net_core`` ALREADY nets that event's hit deduction, so no hit term is
    subtracted again anywhere in this module -- double-counting a -4 hit as an
    8-point swing is the one authority this contract exists to protect.

    ``h2_*`` is the state this arm actually starts H2 from, and it is VALIDATED
    against the arm it claims to be: the restored permanent squad with the
    canonical post-Free-Hit free-transfer count for PLAY, and the permanent squad
    with ordinary progression for SAVE.  A route that describes some other state
    is refused rather than silently valued.
    """

    arm: str
    events: tuple[Any, ...]
    h2_squad_ids: tuple[int, ...]
    h2_bank_tenths: int
    h2_purchase_price_tenths: Mapping[int, int]
    h2_free_transfers: int

    FREE_HIT_ARM_PLAY = "PLAY"
    FREE_HIT_ARM_SAVE = "SAVE"

    def value(self) -> float:
        """The arm's H2-H4 core: the sum of the route engine's per-event net core."""

        return float(sum(float(getattr(entry, "mean_net_core")) for entry in self.events))

    def event_values(self) -> list[dict[str, Any]]:
        return [
            {
                "event": int(getattr(entry, "event")),
                "mean_net_core": round(float(getattr(entry, "mean_net_core")), 6),
                "hit_points": int(getattr(entry, "hit_points", 0)),
                "free_transfers": int(getattr(entry, "free_transfers", 0)),
            }
            for entry in self.events
        ]

    def h2_state(self) -> dict[str, Any]:
        return {
            "arm": str(self.arm),
            "squad_ids": [int(p) for p in self.h2_squad_ids],
            "bank_tenths": int(self.h2_bank_tenths),
            "purchase_price_tenths": {int(k): int(v) for k, v in sorted(self.h2_purchase_price_tenths.items())},
            "free_transfers": int(self.h2_free_transfers),
        }

    def problems(
        self,
        *,
        expected_arm: str,
        expected_events: Sequence[int],
        expected_squad: Sequence[int],
        expected_bank: int,
        expected_free_transfers: int,
    ) -> list[str]:
        found: list[str] = []
        if str(self.arm) != str(expected_arm):
            found.append(f"the route is labelled {self.arm!r}, expected {expected_arm!r}")
        events = tuple(int(getattr(entry, "event", -1)) for entry in self.events)
        if len(events) != len(expected_events):
            found.append(f"{expected_arm} tail has {len(events)} events, expected {len(expected_events)}")
        if events and events != tuple(int(e) for e in expected_events):
            found.append(f"{expected_arm} tail events {list(events)} != {list(expected_events)}")
        if tuple(int(p) for p in self.h2_squad_ids) != tuple(int(p) for p in expected_squad):
            # The arm must start H2 from the RESTORED permanent squad: a Free Hit
            # temporary squad leaking into H2-H4 would be exactly the failure this
            # check exists to catch.
            found.append(f"{expected_arm} tail does not start H2 from the expected permanent squad")
        if int(self.h2_bank_tenths) != int(expected_bank):
            found.append(f"{expected_arm} tail H2 bank != the expected permanent bank")
        if int(self.h2_free_transfers) != int(expected_free_transfers):
            found.append(
                f"{expected_arm} tail H2 free transfers {int(self.h2_free_transfers)} != the "
                f"canonical {int(expected_free_transfers)}"
            )
        for entry in self.events:
            for problem in (getattr(entry, "problems", lambda: [])() or []):
                found.append(f"{expected_arm}: {problem}")
        return found


def save_arm_h2_state(request: "FreeHitRequest") -> RestoredPermanentState:
    """The state the SAVE arm enters H2 with: ordinary Gameweek progression."""

    permanent = request.permanent
    normal_ft = int(
        sr.free_transfers_after_gameweek(request.rules, int(permanent.event_start_free_transfers), 0)
    )
    return RestoredPermanentState(
        event=int(request.planning_event) + 1,
        owned_ids=tuple(int(p) for p in permanent.owned_ids),
        purchase_price_tenths={int(k): int(v) for k, v in permanent.purchase_price_tenths.items()},
        bank_tenths=int(permanent.bank_tenths),
        free_transfers=normal_ft,
        free_transfers_rule="ordinary_gameweek_progression",
    )


def four_gw_arm_values(
    request: "FreeHitRequest",
    *,
    h1_temporary_value: float,
    h1_permanent_value: float,
) -> dict[str, Any]:
    """The TRUE four-GW values of both arms, from their ACTUAL H2 states.

    PLAY : H1 temporary Free Hit value, then the restored state's H2-H4 route
    SAVE : H1 permanent value, then the ordinarily-progressed state's H2-H4 route

    Nothing here assumes the tails are equal, and nothing invents a future
    transfer: both tails come from the canonical route engine.
    """

    play_tail = request.play_tail
    save_tail = request.save_tail
    play_tail_value = play_tail.value() if play_tail is not None else float("nan")
    save_tail_value = save_tail.value() if save_tail is not None else float("nan")
    return {
        "h1_temporary_value": float(h1_temporary_value),
        "h1_permanent_value": float(h1_permanent_value),
        "play_tail_value": float(play_tail_value),
        "save_tail_value": float(save_tail_value),
        "four_gw_play_value": float(h1_temporary_value) + float(play_tail_value),
        "four_gw_save_value": float(h1_permanent_value) + float(save_tail_value),
        "tail_delta": float(play_tail_value) - float(save_tail_value),
    }

# ---------------------------------------------------------------------------
# THE CANONICAL DECISION AUTHORITY
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FreeHitDecisionAuthority:
    """The CANONICAL certified decision context a Free Hit request is anchored to.

    Comparing the request's world identity against ANOTHER request-owned identity
    proves nothing: a caller can alter both together on the cutoff, the source
    snapshot, the prediction run or the model/config while leaving the data
    snapshot intact, and the two artifacts will happily agree with each other.
    Agreement between two caller objects is not authority.

    This type is built from the CERTIFICATION ARTIFACT -- the accepted canonical
    object -- and its identity is RECOMPUTED from that artifact's own fields by
    ``four_gw_decision.certification_identity_of``, so a copied identity on a
    different artifact fails.  Every predictive dimension is then anchored here
    rather than to another supplied object.
    """

    planning_cutoff: str
    data_snapshot_sha256: str
    certification_identity: str
    certified_bundle_identity: Mapping[str, Any] = field(default_factory=dict)
    source_snapshot_sha256: str = ""
    prediction_generation: str = ""
    model_config_identity: str = ""

    def problems(self) -> list[str]:
        found: list[str] = []
        for name in ("planning_cutoff", "data_snapshot_sha256", "certification_identity",
                     "source_snapshot_sha256", "prediction_generation", "model_config_identity"):
            if not str(getattr(self, name) or "").strip():
                found.append(f"the certified decision authority carries no {name}")
        return found

    def disagreements_with(
        self, identity: WildcardPredictiveIdentity | None, *, event: int | None = None,
        label: str = "identity",
    ) -> list[str]:
        """Every predictive dimension on which ``identity`` is not the certified one."""

        found: list[str] = []
        if identity is None:
            return [f"{label} carries no predictive identity"]
        if event is not None and int(event) != int(event):
            found.append(f"{label} event")
        pairs = (
            ("cutoff", str(identity.cutoff), str(self.planning_cutoff)),
            ("data_snapshot_sha256", str(identity.data_snapshot_sha256), str(self.data_snapshot_sha256)),
            ("source_snapshot_sha256", str(identity.source_snapshot_sha256), str(self.source_snapshot_sha256)),
            ("generation", str(identity.generation), str(self.prediction_generation)),
            ("model_config_identity", str(identity.model_config_identity), str(self.model_config_identity)),
        )
        for name, supplied, certified in pairs:
            if supplied != certified:
                found.append(f"{label} {name}")
        return found

    def as_dict(self) -> dict[str, Any]:
        return {
            "planning_cutoff": str(self.planning_cutoff),
            "data_snapshot_sha256": str(self.data_snapshot_sha256),
            "certification_identity": str(self.certification_identity),
            "source_snapshot_sha256": str(self.source_snapshot_sha256),
            "prediction_generation": str(self.prediction_generation),
            "model_config_identity": str(self.model_config_identity),
        }

    @classmethod
    def from_certification(cls, artifact: Mapping[str, Any]) -> "FreeHitDecisionAuthority":
        """Build the authority from a certification artifact, or refuse.

        The artifact's own ``four_gw_certification_identity`` must equal the value
        RECOMPUTED from its cutoff, bundle identities and data snapshot, so a
        recognised identity copied onto a different artifact is refused.  Any
        dimension the artifact does not carry refuses: this object is the anchor,
        and an anchor with a missing dimension would silently stop anchoring it.
        """

        from . import four_gw_decision as fg

        if not artifact:
            raise FreeHitInputError(
                f"{FH_DECISION_AUTHORITY_REQUIRED}: no certification artifact supplied",
                reasons=(FH_DECISION_AUTHORITY_REQUIRED,),
            )
        declared = str(
            artifact.get("four_gw_certification_identity") or artifact.get("certification_identity") or ""
        )
        recomputed = fg.certification_identity_of(artifact)
        if not declared:
            raise FreeHitInputError(
                f"{FH_DECISION_AUTHORITY_REQUIRED}: the certification artifact declares no identity",
                reasons=(FH_DECISION_AUTHORITY_REQUIRED,),
            )
        if str(declared) != str(recomputed):
            raise FreeHitInputError(
                f"{FH_DECISION_AUTHORITY_REQUIRED}: the certification artifact's declared identity is not "
                "the one its own fields produce",
                reasons=(FH_DECISION_AUTHORITY_REQUIRED,),
            )
        bundles = artifact.get("certified_bundle_identity") or {}
        authority = cls(
            planning_cutoff=str(artifact.get("planning_cutoff") or ""),
            data_snapshot_sha256=str(artifact.get("data_snapshot_sha256") or ""),
            certification_identity=str(declared),
            certified_bundle_identity=dict(bundles) if isinstance(bundles, Mapping) else {},
            source_snapshot_sha256=str(
                artifact.get("source_snapshot_sha256")
                or (bundles.get("source_snapshot_sha256") if isinstance(bundles, Mapping) else "")
                or ""
            ),
            prediction_generation=str(
                artifact.get("prediction_generation")
                or artifact.get("projection_run_identity")
                or (bundles.get("generation") if isinstance(bundles, Mapping) else "")
                or ""
            ),
            model_config_identity=str(
                artifact.get("model_config_identity")
                or artifact.get("config_hash")
                or (bundles.get("model_config_identity") if isinstance(bundles, Mapping) else "")
                or ""
            ),
        )
        problems = authority.problems()
        if problems:
            raise FreeHitInputError(
                f"{FH_DECISION_AUTHORITY_REQUIRED}: " + "; ".join(problems[:6]),
                reasons=(FH_DECISION_AUTHORITY_REQUIRED,),
            )
        return authority
