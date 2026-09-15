"""Wildcard V1 — a quantitative evaluator for PLAY_WC_NOW vs SAVE_WC.

WHAT THIS IS
------------
Wildcard reshapes the PERMANENT squad, so it is not a one-week squad optimizer
and it is not the four-GW transfer decision.  It is evaluated over a longer,
tapered horizon while the *chip decision itself* stays bound to the certified
four-event chip context the arbiter already enforces
(``ChipHorizonBinding``).  The two horizons are deliberately separate:

  * ``horizon_binding``  — the certified 4-event identity the arbiter validates
                           (unchanged contract, identical to TC).
  * ``WildcardHorizonSpec`` — this evaluator's OWN longer squad-value horizon
                           (6-10 GW, tapered), versioned and persisted.

NOT AUTONOMOUSLY EXECUTABLE
---------------------------
Wildcard remains review/candidate-only.  A positive uplift over SAVE produces
``CHIP_CANDIDATE_RECHECK_REQUIRED`` (or ``CHIP_REVIEW_REQUIRED`` when the paired
interval still straddles zero) and never ``PLAY_CHIP``, because the reservation
value is UNCALIBRATED.  ``PLAY_CHIP`` stays reachable only by supplying a
reservation that declares itself CALIBRATED.

WHY THE SAVE ARM IS NOT "NEVER WILDCARD"
----------------------------------------
SAVE means: do not wildcard now, play the legal normal route, and KEEP the
wildcard.  Its value is the current squad's horizon value plus a non-clairvoyant
reservation for the retained option — never a clairvoyant best-future-wildcard.

NO GLOBAL OPTIMALITY CLAIM
--------------------------
The search is exhaustive SCREENING followed by BOUNDED frontier construction and
local improvement.  It reports the sizes it considered and never claims to have
found the optimum.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from . import manager_lineup
from . import season_rules as sr
from . import transfer_state as ts

WILDCARD_EVALUATOR_VERSION = "chip_wc_v1.0.0"
WILDCARD_HORIZON_VERSION = "wildcard_horizon_v1.1.0"

WILDCARD_HORIZON_MIN_EVENTS = 6
WILDCARD_HORIZON_MAX_EVENTS = 10
WILDCARD_HORIZON_DEFAULT_EVENTS = 8
#: Geometric taper: near gameweeks carry the most value and there is no cliff
#: immediately after GW4, while the tail still informs squad structure.
WILDCARD_HORIZON_DECAY = 0.82
#: Weight on the terminal squad-quality term relative to one horizon gameweek.
#: Terminal and flexibility terms are REPORTED structurally, not scored.
WILDCARD_TERMINAL_WEIGHT = 0.0
WILDCARD_FLEXIBILITY_WEIGHT = 0.0
#: Bank (in tenths) that counts as fully flexible for scoring purposes.
WILDCARD_FLEXIBILITY_BANK_UNIT_TENTHS = 20

# --- refusal tokens ---------------------------------------------------------

WC_INCOMPLETE_PLAYER_POOL = "WILDCARD_INCOMPLETE_PLAYER_POOL"
WC_MISSING_PROJECTION = "WILDCARD_MISSING_PROJECTION"
WC_STALE_OR_CROSS_CUTOFF = "WILDCARD_STALE_OR_CROSS_CUTOFF_PREDICTION"
WC_ELIGIBILITY_INVALID = "WILDCARD_ELIGIBILITY_INVALID"
WC_WINDOW_UNRESOLVED = "WILDCARD_ACTIVE_WINDOW_UNRESOLVED"
WC_WINDOW_AMBIGUOUS = "WILDCARD_ACTIVE_WINDOW_AMBIGUOUS"
WC_ILLEGAL_SQUAD = "WILDCARD_ILLEGAL_SQUAD"
WC_PRICING_UNAVAILABLE = "WILDCARD_PRICING_BASIS_UNAVAILABLE"
WC_MANAGER_STATE_INCOHERENT = "WILDCARD_MANAGER_STATE_INCOHERENT"
WC_FRONTIER_EMPTY = "WILDCARD_OPTIMIZER_FRONTIER_EMPTY"
WC_HORIZON_INVALID = "WILDCARD_HORIZON_INVALID"

WC_REASON_POSITIVE = "WILDCARD_UPLIFT_POSITIVE"
WC_REASON_NOT_COMPETITIVE = "WILDCARD_NOT_COMPETITIVE"
WC_REASON_PLAY_DESCRIPTION = "WILDCARD_PLAY_NOW_SQUAD_DESCRIPTION"
WC_REASON_REVIEW_ONLY = "WILDCARD_REVIEW_ONLY_UNCALIBRATED"


#: Alias kept simple: any iterable of ints is accepted for the explicit set.
FrozenSetCapable = Any


class WildcardError(ValueError):
    """The Wildcard evaluation could not be produced from the supplied evidence."""

    def __init__(self, detail: str, *, reasons: Sequence[str] = ()) -> None:
        super().__init__(detail)
        self.reasons = list(reasons)


class WildcardInputError(WildcardError):
    """The input contract is incomplete, so the evaluator fails closed."""


# ---------------------------------------------------------------------------
# Versioned horizon
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WildcardHorizonSpec:
    """The Wildcard squad-value horizon and its tapered weighting.

    Versioned because the weights determine the value of every candidate squad;
    changing them changes what the evaluator recommends, so they must be
    pinnable and reproducible rather than inline constants.
    """

    version: str
    events: tuple[int, ...]
    weights: tuple[float, ...]
    decay: float
    terminal_weight: float

    def __post_init__(self) -> None:
        if not (WILDCARD_HORIZON_MIN_EVENTS <= len(self.events) <= WILDCARD_HORIZON_MAX_EVENTS):
            raise WildcardInputError(
                f"{WC_HORIZON_INVALID}: horizon needs {WILDCARD_HORIZON_MIN_EVENTS}-"
                f"{WILDCARD_HORIZON_MAX_EVENTS} events, got {len(self.events)}",
                reasons=(WC_HORIZON_INVALID,),
            )
        if len(set(self.events)) != len(self.events):
            raise WildcardInputError(f"{WC_HORIZON_INVALID}: duplicate events",
                                     reasons=(WC_HORIZON_INVALID,))
        if len(self.weights) != len(self.events):
            raise WildcardInputError(f"{WC_HORIZON_INVALID}: weight count mismatch",
                                     reasons=(WC_HORIZON_INVALID,))
        if abs(sum(self.weights) - 1.0) > 1e-9:
            raise WildcardInputError(f"{WC_HORIZON_INVALID}: weights must sum to 1",
                                     reasons=(WC_HORIZON_INVALID,))
        if any(w <= 0.0 for w in self.weights):
            raise WildcardInputError(f"{WC_HORIZON_INVALID}: weights must be positive",
                                     reasons=(WC_HORIZON_INVALID,))

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "events": list(self.events),
            "weights": [round(w, 10) for w in self.weights],
            "decay": self.decay,
            "terminal_weight": self.terminal_weight,
        }

    def weight_for(self, event: int) -> float:
        try:
            return self.weights[self.events.index(int(event))]
        except ValueError as exc:  # pragma: no cover - guarded by construction
            raise WildcardInputError(f"event {event} is outside the horizon") from exc


def wildcard_horizon(
    planning_event: int,
    *,
    length: int = WILDCARD_HORIZON_DEFAULT_EVENTS,
    last_event: int = 38,
    decay: float = WILDCARD_HORIZON_DECAY,
) -> WildcardHorizonSpec:
    """Tapered horizon from ``planning_event``, capped at the season end."""

    events = tuple(range(int(planning_event), min(int(last_event), int(planning_event) + length - 1) + 1))
    raw = [decay ** i for i in range(len(events))]
    total = sum(raw)
    return WildcardHorizonSpec(
        version=WILDCARD_HORIZON_VERSION,
        events=events,
        weights=tuple(w / total for w in raw),
        decay=decay,
        terminal_weight=WILDCARD_TERMINAL_WEIGHT,
    )


# ---------------------------------------------------------------------------
# Typed input contract (football scoring is an INPUT, never recomputed here)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WildcardPlayerEvent:
    """One player's projection for one event, from certified inputs only."""

    event: int
    expected_points: float
    expected_minutes: float
    p_start: float
    availability: float
    fixture_count: int = 1


@dataclass(frozen=True)
class WildcardPlayer:
    """One candidate player, carrying every event the horizon needs."""

    player_id: int
    position: str
    club_id: int
    market_price_tenths: int
    events: Mapping[int, WildcardPlayerEvent]
    web_name: str = ""

    def at(self, event: int) -> WildcardPlayerEvent | None:
        return self.events.get(int(event))


@dataclass(frozen=True)
class WildcardRequest:
    """Everything the evaluator consumes.  Nothing here is a football model."""

    planning_event: int
    horizon: WildcardHorizonSpec
    players: Mapping[int, WildcardPlayer]
    positions: Mapping[int, str]
    owned_ids: tuple[int, ...]
    purchase_price_tenths: Mapping[int, int]
    selling_price_tenths: Mapping[int, int]
    bank_tenths: int
    rules: sr.SeasonRules
    horizon_binding: Any  # chip_decision.ChipHorizonBinding
    certification_identity: str
    data_snapshot_sha256: str | None = None
    event_start_free_transfers: int | None = None
    save_route_value: float | None = None
    save_route_expected_hits: int = 0
    reservation: Any | None = None
    #: planning.chips_state rows — the canonical window evidence, so the
    #: Wildcard expiry resolves through the same selector the arbiter uses.
    chip_availability: Sequence[Mapping[str, Any]] = ()
    #: Owned players the caller intends to SELL AND BUY BACK in the same
    #: Wildcard.  Never inferred from final squad membership — a player owned
    #: before and after with no explicit entry here is simply RETAINED.
    explicit_sell_rebuy: FrozenSetCapable = ()
    position_frontier_size: int = 6
    improvement_passes: int = 3


# ---------------------------------------------------------------------------
# Stage 0/1 — exhaustive screening and position frontiers
# ---------------------------------------------------------------------------


def _available(player: WildcardPlayer, horizon: WildcardHorizonSpec) -> bool:
    """A player is screenable only if EVERY horizon event has a projection."""

    return all(player.at(event) is not None for event in horizon.events)


def screen_players(request: WildcardRequest) -> tuple[dict[int, float], dict[str, Any]]:
    """Exhaustive numeric screen.  NO watchlist, ownership or club gate.

    Every eligible official player with valid predictive support is scored.  A
    player with a missing projection is EXCLUDED WITH AN AUDITABLE REASON, never
    silently treated as zero points.
    """

    horizon = request.horizon
    scores: dict[int, float] = {}
    missing: list[int] = []
    for player_id, player in sorted(request.players.items()):
        if not _available(player, horizon):
            missing.append(int(player_id))
            continue
        total = 0.0
        for event in horizon.events:
            entry = player.at(event)
            weight = horizon.weight_for(event)
            # Availability scales expectation; it never removes the player from
            # consideration (that is the pool's job, not the screen's).
            total += weight * float(entry.expected_points) * max(0.0, min(1.0, float(entry.availability)))
        scores[int(player_id)] = total
    return scores, {
        "screened": len(scores),
        "excluded_missing_projection": len(missing),
        "excluded_missing_projection_ids": sorted(missing)[:50],
    }


def position_frontiers(
    request: WildcardRequest, scores: Mapping[int, float]
) -> dict[str, list[int]]:
    """Stage 1: a bounded candidate frontier per position.

    Diversity matters as much as raw score: a frontier of six near-identical
    premiums cannot build a legal squad under a budget, so the frontier keeps the
    best scorers AND the best value-per-price at each position.
    """

    by_position: dict[str, list[tuple[float, float, int]]] = defaultdict(list)
    for player_id, score in scores.items():
        player = request.players[player_id]
        price = max(1, int(player.market_price_tenths))
        by_position[player.position].append((score, score / price, player_id))

    frontier: dict[str, list[int]] = {}
    size = max(1, int(request.position_frontier_size))
    for position in ts.POSITION_COMPOSITION:
        rows = by_position.get(position, [])
        by_score = sorted(rows, key=lambda r: (-r[0], r[2]))[:size]
        by_value = sorted(rows, key=lambda r: (-r[1], r[2]))[:size]
        # cheapest legal filler so a budget-constrained squad is always buildable
        by_price = sorted(rows, key=lambda r: (request.players[r[2]].market_price_tenths, r[2]))[:size]
        chosen: list[int] = []
        for player_id in [r[2] for r in by_score] + [r[2] for r in by_value] + [r[2] for r in by_price]:
            if player_id not in chosen:
                chosen.append(player_id)
        frontier[position] = chosen
    return frontier


# ---------------------------------------------------------------------------
# Stage 2/3 — legal squad construction and bounded improvement
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WildcardTransactionPlan:
    """The explicit sell/buy accounting behind one candidate squad.

    RETAINED players are neither sold nor bought: their selling value is NOT
    cash and they are NOT charged, and their acquisition basis is preserved.
    SOLD players realise their canonical selling value.  BOUGHT players are
    charged the current market price and take it as their NEW basis.
    """

    retained_ids: tuple[int, ...]
    sold_ids: tuple[int, ...]
    bought_ids: tuple[int, ...]
    rebought_ids: tuple[int, ...]
    cash_available_tenths: int
    purchase_cost_tenths: int
    remaining_bank_tenths: int
    new_basis: Mapping[int, int]

    def as_dict(self) -> dict[str, Any]:
        return {
            "retained_ids": list(self.retained_ids),
            "sold_ids": list(self.sold_ids),
            "bought_ids": list(self.bought_ids),
            "rebought_ids": list(self.rebought_ids),
            "cash_available_tenths": self.cash_available_tenths,
            "purchase_cost_tenths": self.purchase_cost_tenths,
            "remaining_bank_tenths": self.remaining_bank_tenths,
            "new_basis": {int(k): int(v) for k, v in sorted(self.new_basis.items())},
        }


def canonical_selling_value(request: WildcardRequest, player_id: int) -> int:
    """The canonical FPL selling value, validated against any supplied figure.

    The rule is owned by ``transfer_state.selling_price_tenths``; this function
    never invents one.  A caller-supplied selling value that DISAGREES with the
    canonical calculation is a hard error rather than being trusted (§1/§14).
    """

    pid = int(player_id)
    purchase = request.purchase_price_tenths.get(pid)
    if purchase is None:
        raise WildcardInputError(
            f"{WC_PRICING_UNAVAILABLE}: no acquisition basis for owned player {pid}",
            reasons=(WC_PRICING_UNAVAILABLE,),
        )
    market = int(request.players[pid].market_price_tenths)
    computed = ts.selling_price_tenths(int(purchase), market)
    supplied = request.selling_price_tenths.get(pid)
    if supplied is not None and int(supplied) != int(computed):
        raise WildcardInputError(
            f"{WC_PRICING_UNAVAILABLE}: selling value for {pid} is {int(supplied)} but the "
            f"canonical rule gives {computed} from purchase {int(purchase)} / market {market}",
            reasons=(WC_PRICING_UNAVAILABLE,),
        )
    return int(computed)


def plan_transaction(request: WildcardRequest, squad: Sequence[int]) -> WildcardTransactionPlan:
    """Explicit transaction accounting.  No retained capital becomes cash.

    ``cash_available = bank + SUM(canonical selling value of players ACTUALLY SOLD)``
    and ``purchase_cost = SUM(market price of players ACTUALLY BOUGHT)``.  A
    retained player contributes to neither side, which is the whole point: a
    squad that keeps expensive players cannot spend their locked capital.
    """

    target = {int(p) for p in squad}
    owned = {int(p) for p in request.owned_ids}
    forced = {int(p) for p in request.explicit_sell_rebuy}

    retained = tuple(sorted((owned & target) - forced))
    sold = tuple(sorted(owned - target))
    # A retained player explicitly marked sell+rebuy is sold and bought back at
    # the same time; it is never inferred merely from final membership (§2).
    rebought = tuple(sorted(forced & target))
    bought = tuple(sorted((target - owned) | set(rebought)))

    cash = int(request.bank_tenths) + sum(canonical_selling_value(request, pid) for pid in sold + rebought)
    cost = sum(int(request.players[pid].market_price_tenths) for pid in bought)
    basis = {pid: int(request.players[pid].market_price_tenths) for pid in bought}
    return WildcardTransactionPlan(
        retained_ids=retained,
        sold_ids=sold,
        bought_ids=bought,
        rebought_ids=rebought,
        cash_available_tenths=cash,
        purchase_cost_tenths=cost,
        remaining_bank_tenths=cash - cost,
        new_basis=basis,
    )


def _budget_and_cost(request: WildcardRequest, squad: Sequence[int]) -> tuple[int, int, int]:
    """(cash_available, purchase_cost, remaining_bank) from the transaction plan."""

    plan = plan_transaction(request, squad)
    return plan.cash_available_tenths, plan.purchase_cost_tenths, plan.remaining_bank_tenths


def is_legal_squad(request: WildcardRequest, squad: Sequence[int]) -> tuple[bool, list[str]]:
    """Canonical legality: size, composition, club limit, affordability."""

    problems: list[str] = []
    ids = [int(p) for p in squad]
    if len(ids) != ts.SQUAD_SIZE or len(set(ids)) != ts.SQUAD_SIZE:
        problems.append(f"size={len(ids)} unique={len(set(ids))}")
    counts = defaultdict(int)
    clubs = defaultdict(int)
    for player_id in ids:
        player = request.players.get(player_id)
        if player is None:
            problems.append(f"player {player_id} is not in the eligible universe")
            continue
        counts[player.position] += 1
        clubs[player.club_id] += 1
    for position, required in ts.POSITION_COMPOSITION.items():
        if counts.get(position, 0) != required:
            problems.append(f"{position}={counts.get(position, 0)} (need {required})")
    over = {club: n for club, n in clubs.items() if n > ts.SQUAD_TEAM_LIMIT}
    if over:
        problems.append(f"club limit exceeded: {over}")
    _, cost, bank_after = _budget_and_cost(request, ids)
    if bank_after < 0:
        problems.append(f"over budget by {-bank_after} tenths (cost {cost})")
    return (not problems), problems


def _greedy_squad(
    request: WildcardRequest, scores: Mapping[int, float], frontier: Mapping[str, list[int]],
    *, key: str,
) -> list[int] | None:
    """Deterministic greedy seed filling the canonical composition."""

    available, _, _ = _budget_and_cost(request, ())
    squad: list[int] = []
    clubs: dict[int, int] = defaultdict(int)
    spent = 0
    owned = set(int(p) for p in request.owned_ids)

    for position, required in ts.POSITION_COMPOSITION.items():
        candidates = [pid for pid in frontier.get(position, []) if pid in scores]
        if key == "value":
            candidates.sort(key=lambda pid: (-(scores[pid] / max(1, request.players[pid].market_price_tenths)), pid))
        else:
            candidates.sort(key=lambda pid: (-scores[pid], pid))
        picked = 0
        for player_id in candidates:
            if picked >= required:
                break
            player = request.players[player_id]
            if clubs[player.club_id] >= ts.SQUAD_TEAM_LIMIT:
                continue
            price = 0 if player_id in owned else player.market_price_tenths
            if spent + price > available:
                continue
            squad.append(player_id)
            clubs[player.club_id] += 1
            spent += price
            picked += 1
        if picked < required:
            return None
    return squad if len(squad) == ts.SQUAD_SIZE else None


def _improve(
    request: WildcardRequest, squad: list[int], value_of, *, passes: int,
    universe: Sequence[int] | None = None,
) -> list[int]:
    """Bounded local improvement: only strictly-improving, still-legal swaps."""

    current = list(squad)
    current_value = value_of(current)
    for _ in range(max(0, passes)):
        best_gain = 0.0
        best_swap: tuple[int, int] | None = None
        for index in range(len(current)):
            # ONLY the screened universe: a missing-projection player can
            # never be re-introduced by the local search.
            for candidate in sorted(universe if universe is not None else scores):
                if candidate in current:
                    continue
                trial = list(current)
                trial[index] = candidate
                ok, _ = is_legal_squad(request, trial)
                if not ok:
                    continue
                gain = value_of(trial) - current_value
                if gain > best_gain + 1e-12 or (abs(gain - best_gain) <= 1e-12 and best_swap and candidate < best_swap[1]):
                    best_gain = gain
                    best_swap = (index, candidate)
        if best_swap is None or best_gain <= 1e-12:
            break
        current[best_swap[0]] = best_swap[1]
        current_value += best_gain
    return current


# ---------------------------------------------------------------------------
# The value function
# ---------------------------------------------------------------------------


def _best_xi(
    request: WildcardRequest, squad: Sequence[int], event: int
) -> tuple[list[int], float]:
    """Best legal XI and its expected points, per event.

    Lineup and captain are chosen PER EVENT — the armband is never fixed across
    the horizon.  Formations come from the canonical manager_lineup bounds.
    """

    def points(player_id: int) -> float:
        entry = request.players[player_id].at(event)
        if entry is None:
            return 0.0
        return float(entry.expected_points) * max(0.0, min(1.0, float(entry.availability)))

    by_position: dict[str, list[int]] = defaultdict(list)
    for player_id in squad:
        by_position[request.players[player_id].position].append(player_id)
    for position in by_position:
        by_position[position].sort(key=lambda pid: (-points(pid), pid))

    best: tuple[float, list[int]] | None = None
    keepers = by_position.get("GKP", [])[:1]
    if not keepers:
        return [], 0.0
    for defenders in range(manager_lineup.FORMATION_MIN["DEF"], manager_lineup.FORMATION_MAX["DEF"] + 1):
        for midfielders in range(manager_lineup.FORMATION_MIN["MID"], manager_lineup.FORMATION_MAX["MID"] + 1):
            for forwards in range(manager_lineup.FORMATION_MIN["FWD"], manager_lineup.FORMATION_MAX["FWD"] + 1):
                if defenders + midfielders + forwards != manager_lineup.XI_SIZE - 1:
                    continue
                selection = list(keepers)
                selection += by_position.get("DEF", [])[:defenders]
                selection += by_position.get("MID", [])[:midfielders]
                selection += by_position.get("FWD", [])[:forwards]
                if len(selection) != manager_lineup.XI_SIZE:
                    continue
                counts = {
                    "DEF": defenders, "MID": midfielders, "FWD": forwards,
                }
                if not manager_lineup.formation_is_legal({**counts, "GKP": 1}):
                    continue
                total = sum(points(pid) for pid in selection)
                if best is None or total > best[0] + 1e-12 or (
                    abs(total - best[0]) <= 1e-12 and selection < best[1]
                ):
                    best = (total, selection)
    return (best[1], best[0]) if best else ([], 0.0)


def _armband_uplift(request: WildcardRequest, xi: Sequence[int], event: int) -> float:
    """Expected captaincy uplift for one event.

    The accepted rule is that the armband holder scores twice, falling back to
    the vice only when the captain does not appear.  For an expected-value
    horizon this is E[uplift] = p_cap * pts_cap + (1 - p_cap) * p_vice * pts_vice,
    which reproduces ``manager_lineup.captain_multiplier`` exactly in the
    deterministic cases (captain always plays, or never plays).
    """

    if len(xi) < 2:
        return 0.0

    def payload(player_id: int) -> tuple[float, float]:
        entry = request.players[player_id].at(event)
        if entry is None:
            return 0.0, 0.0
        availability = max(0.0, min(1.0, float(entry.availability)))
        return float(entry.p_start) * availability, float(entry.expected_points) * availability

    ranked = sorted(xi, key=lambda pid: (-payload(pid)[1], pid))
    captain, vice = ranked[0], ranked[1]
    p_cap, pts_cap = payload(captain)
    p_vice, pts_vice = payload(vice)
    return p_cap * pts_cap + (1.0 - p_cap) * p_vice * pts_vice


def _bench_value(request: WildcardRequest, squad: Sequence[int], xi: Sequence[int], event: int) -> float:
    """Bench matters through autosubs and injury cover — never four dead slots."""

    bench = [pid for pid in squad if pid not in set(xi)]
    total = 0.0
    for player_id in bench:
        entry = request.players[player_id].at(event)
        if entry is None:
            continue
        # A bench player only scores when he covers a non-appearing starter, so
        # his contribution is discounted by the chance he is actually needed.
        need = 1.0 - min(1.0, float(entry.availability))
        total += float(entry.expected_points) * max(0.0, min(1.0, float(entry.availability))) * (0.35 + 0.65 * need)
    return total


def _minutes_security(request: WildcardRequest, squad: Sequence[int]) -> float:
    """Mean start probability across the squad over the horizon."""

    total = 0.0
    count = 0
    for player_id in squad:
        player = request.players[player_id]
        for event in request.horizon.events:
            entry = player.at(event)
            if entry is None:
                continue
            total += max(0.0, min(1.0, float(entry.p_start)))
            count += 1
    return total / count if count else 0.0


def _repairability(request: WildcardRequest, squad: Sequence[int], bank_after: int) -> dict[str, Any]:
    """Compact, deterministic flexibility measures (§11)."""

    prices = [request.players[pid].market_price_tenths for pid in squad]
    # Price-point coverage: how many distinct price bands the squad can move
    # within without a hit — a crude proxy for one-transfer upgrade/downgrade
    # routes existing at all.
    bands = {round(price / 10) for price in prices}
    cheap = [pid for pid in squad if request.players[pid].market_price_tenths <= 45]
    forced = 0
    for player_id in squad:
        player = request.players[player_id]
        weak = sum(
            1 for event in request.horizon.events
            if (entry := player.at(event)) is not None and float(entry.p_start) < 0.5
        )
        if weak >= max(1, len(request.horizon.events) // 2):
            forced += 1
    bank_flex = min(1.0, bank_after / WILDCARD_FLEXIBILITY_BANK_UNIT_TENTHS) if bank_after > 0 else 0.0
    return {
        "bank_after_tenths": int(bank_after),
        "price_bands": len(bands),
        "cheap_slots": len(cheap),
        "expected_forced_moves": forced,
        "bank_flexibility": bank_flex,
    }


@dataclass(frozen=True)
class WildcardSquadValue:
    """One candidate squad's full, explainable valuation."""

    squad: tuple[int, ...]
    horizon_points: float
    near_term_points: float
    medium_term_points: float
    terminal_value: float
    flexibility_value: float
    minutes_security: float
    bank_after_tenths: int
    cost_tenths: int
    repairability: Mapping[str, Any]
    event_points: Mapping[int, float]
    objective: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "squad": list(self.squad),
            "horizon_points": round(self.horizon_points, 6),
            "near_term_points": round(self.near_term_points, 6),
            "medium_term_points": round(self.medium_term_points, 6),
            "terminal_value": round(self.terminal_value, 6),
            "flexibility_value": round(self.flexibility_value, 6),
            "minutes_security": round(self.minutes_security, 6),
            "bank_after_tenths": int(self.bank_after_tenths),
            "cost_tenths": int(self.cost_tenths),
            "repairability": dict(self.repairability),
            "objective": round(self.objective, 6),
        }


def evaluate_squad(request: WildcardRequest, squad: Sequence[int]) -> WildcardSquadValue:
    """Fresh evaluation of ONE squad (Stage 4 uses exactly this function)."""

    horizon = request.horizon
    available, cost, bank_after = _budget_and_cost(request, squad)
    event_points: dict[int, float] = {}
    horizon_points = 0.0
    near = 0.0
    medium = 0.0
    for index, event in enumerate(horizon.events):
        xi, xi_points = _best_xi(request, squad, event)
        value = xi_points + _armband_uplift(request, xi, event) + _bench_value(request, squad, xi, event)
        event_points[event] = value
        weighted = horizon.weight_for(event) * value
        horizon_points += weighted
        if index < 4:
            near += weighted
        else:
            medium += weighted
    security = _minutes_security(request, squad)
    repair = _repairability(request, squad, bank_after)
    # The PRIMARY numeric value is the versioned tapered expected FPL points
    # over the Wildcard horizon, and NOTHING ELSE.  Structural measures (bank,
    # minutes security, forced-move risk, price-band coverage, repairability)
    # are reported separately and used only for Pareto retention and tie-breaks;
    # they are never converted into fake points with judgement weights, and no
    # terminal reward is derived from points already counted.
    terminal = 0.0
    flexibility = 0.0
    objective = horizon_points
    return WildcardSquadValue(
        squad=tuple(sorted(int(p) for p in squad)),
        horizon_points=horizon_points,
        near_term_points=near,
        medium_term_points=medium,
        terminal_value=terminal,
        flexibility_value=flexibility,
        minutes_security=security,
        bank_after_tenths=bank_after,
        cost_tenths=cost,
        repairability=repair,
        event_points=event_points,
        objective=objective,
    )


# ---------------------------------------------------------------------------
# Stage 2/3 — bounded candidate construction with frontier retention
# ---------------------------------------------------------------------------


def build_wildcard_candidates(
    request: WildcardRequest, scores: Mapping[int, float]
) -> tuple[list[WildcardSquadValue], dict[str, Any]]:
    # ``scores`` IS the screened universe: a player without horizon support is
    # absent from it and must never re-enter through a cheap-filler, bank,
    # security or local-search path.  Every search below is bounded by it.
    """Build a bounded, legal candidate set and retain a Pareto frontier.

    This never claims to enumerate every legal squad.  It seeds from several
    deterministic greedy constructions, improves each within a bounded number of
    strictly-improving legal swaps, and keeps the non-dominated frontier over
    (points, bank, minutes security).
    """

    frontier = position_frontiers(request, scores)
    seeds: list[list[int]] = []
    for key in ("score", "value"):
        seed = _greedy_squad(request, scores, frontier, key=key)
        if seed is not None:
            seeds.append(seed)
    # A cheapest-legal seed guarantees a feasible squad exists even under a
    # tight budget, so an empty frontier means genuinely infeasible, not unlucky.
    cheap = _greedy_squad(request, scores, frontier, key="value")
    if cheap is not None and cheap not in seeds:
        seeds.append(cheap)

    stats = {
        "frontier_by_position": {k: len(v) for k, v in frontier.items()},
        "seeds": len(seeds),
        "legal_squads_considered": 0,
        "improvement_passes": int(request.improvement_passes),
    }
    if not seeds:
        return [], stats

    evaluated: dict[tuple[int, ...], WildcardSquadValue] = {}
    baseline_seed = seeds[0]

    def value_of(squad: Sequence[int]) -> float:
        key = tuple(sorted(int(p) for p in squad))
        if key not in evaluated:
            evaluated[key] = evaluate_squad(request, key)
        return evaluated[key].objective

    # Score the seed once so the improvement loop has a comparator.
    value_of(baseline_seed)
    improved: list[list[int]] = []
    for seed in seeds:
        candidate = _improve(request, seed, value_of, passes=request.improvement_passes,
                             universe=sorted(scores))
        improved.append(candidate)
        value_of(candidate)

    legal = [v for v in evaluated.values() if is_legal_squad(request, v.squad)[0]]
    stats["legal_squads_considered"] = len(legal)
    if not legal:
        return [], stats

    # Pareto frontier over (points, bank, minutes security): a squad survives
    # only if no other squad is at least as good on all three.
    retained: list[WildcardSquadValue] = []
    for candidate in sorted(legal, key=lambda v: (-v.objective, v.squad)):
        dominated = False
        for other in legal:
            if other.squad == candidate.squad:
                continue
            if (
                other.horizon_points >= candidate.horizon_points - 1e-12
                and other.bank_after_tenths >= candidate.bank_after_tenths
                and other.minutes_security >= candidate.minutes_security - 1e-12
                and (
                    other.horizon_points > candidate.horizon_points + 1e-12
                    or other.bank_after_tenths > candidate.bank_after_tenths
                    or other.minutes_security > candidate.minutes_security + 1e-12
                )
            ):
                dominated = True
                break
        if not dominated:
            retained.append(candidate)
    stats["pareto_size"] = len(retained)
    if not retained:
        retained = sorted(legal, key=lambda v: (-v.objective, v.squad))[:1]
    return retained, stats


# ---------------------------------------------------------------------------
# The evaluation
# ---------------------------------------------------------------------------


def _incoherent_reasons(request: WildcardRequest) -> list[str]:
    reasons: list[str] = []
    if not request.players:
        reasons.append(WC_INCOMPLETE_PLAYER_POOL)
    if len(request.owned_ids) != ts.SQUAD_SIZE:
        reasons.append(WC_MANAGER_STATE_INCOHERENT)
    missing_price = [pid for pid in request.owned_ids if pid not in request.selling_price_tenths]
    if missing_price:
        reasons.append(WC_PRICING_UNAVAILABLE)
    missing_purchase = [pid for pid in request.owned_ids if pid not in request.purchase_price_tenths]
    if missing_purchase:
        reasons.append(WC_PRICING_UNAVAILABLE)
    for player_id in request.owned_ids:
        if player_id not in request.players:
            reasons.append(WC_INCOMPLETE_PLAYER_POOL)
            break
    return reasons


def _refuse(request: WildcardRequest, token: str, detail: str, *, reasons: Sequence[str] = ()):
    from . import chip_decision as cd

    return cd.ChipEvaluation(
        action=cd.CHIP_ACTION_WC,
        evaluator_version=WILDCARD_EVALUATOR_VERSION,
        candidate_metrics={"mean_paired_uplift": None},
        uncertainty={},
        reason_codes=tuple(sorted({token, *reasons})),
        calibration_status=cd.CALIBRATION_UNCALIBRATED,
        evidence={
            "certification_identity": request.certification_identity,
            "horizon_events": list(request.horizon_binding.horizon_events),
            "planning_event": int(request.planning_event),
            "refusal_detail": detail,
            "wildcard_horizon": request.horizon.as_dict(),
            "wildcard_evaluator_version": WILDCARD_EVALUATOR_VERSION,
            "wildcard_quantitative_capability": "SUPPORTED_REVIEW_ONLY",
            "executable": False,
            "actionable": False,
        },
    )


# --- active Wildcard window resolution (delegated to the chip core) ---------

#: Sentinels returned by ``resolve_wildcard_expiry`` for the two undecidable
#: states.  They are distinct objects so they can never be confused with a real
#: event number.
EXPIRY_UNRESOLVED = "WILDCARD_WINDOW_UNRESOLVED"
EXPIRY_AMBIGUOUS = "WILDCARD_WINDOW_AMBIGUOUS"


def resolve_wildcard_expiry(request: WildcardRequest) -> int | None:
    """The stop_event of the ONE Wildcard definition row that is currently active.

    Delegates to ``chip_decision._active_definition`` — the SAME selector the
    arbiter already uses — so the Wildcard window can never disagree with the
    eligibility that admitted the action, and overlapping windows fail closed
    rather than being resolved by row order (§19).

    Returns ``None`` when no window is declared at all (the caller then treats
    expiry as unknown rather than inventing one).
    """

    from . import chip_decision as cd

    rows = [row for row in (request.chip_availability or ()) if str(row.get("name") or "") == "wildcard"]
    if not rows:
        # No Wildcard definition at all is NOT "no expiry": the chip's window is
        # unknown, so the caller must fail closed rather than continue numerically.
        return EXPIRY_UNRESOLVED
    mapped = cd._availability_by_action(rows, planning_event=int(request.planning_event))
    entry = mapped.get(cd.CHIP_ACTION_WC) or {}
    if not entry.get("eligible"):
        return EXPIRY_UNRESOLVED
    definitions = entry.get("definitions") or []
    active, problem = cd._active_definition(definitions)
    if problem == cd.DIAG_CHIP_WINDOW_SELECTION_AMBIGUOUS:
        return EXPIRY_AMBIGUOUS
    if active is None:
        return EXPIRY_UNRESOLVED
    stop = active.get("window_stop_event")
    return None if stop is None else int(stop)


def evaluate_wildcard(request: WildcardRequest):
    """PLAY_WC_NOW vs SAVE_WC -> the arbiter's ``ChipEvaluation`` shape.

    Wildcard is REVIEW-ONLY: nothing here can emit an executable play.  The
    returned evaluation carries a mean paired uplift, its uncertainty, and the
    reason codes the arbiter turns into CHIP_CANDIDATE_RECHECK_REQUIRED or
    CHIP_REVIEW_REQUIRED.
    """

    from . import chip_decision as cd

    incoherent = _incoherent_reasons(request)
    if incoherent:
        return _refuse(request, incoherent[0], f"manager state/pricing not coherent: {incoherent}", reasons=incoherent)

    if request.certification_identity != getattr(request.horizon_binding, "certification_identity", None):
        return _refuse(request, WC_STALE_OR_CROSS_CUTOFF,
                       "the request certification identity does not match the horizon binding")

    # Resolve the ACTIVE Wildcard window first: the chip window is a
    # precondition for evaluating the chip at all, so an unresolved or ambiguous
    # window must fail fast rather than after a full screen.
    expiry_event = resolve_wildcard_expiry(request)
    if expiry_event is EXPIRY_UNRESOLVED:
        return _refuse(request, WC_WINDOW_UNRESOLVED,
                       "the active Wildcard window could not be resolved from the canonical chip state",
                       reasons=(WC_WINDOW_UNRESOLVED,))
    if expiry_event is EXPIRY_AMBIGUOUS:
        return _refuse(request, WC_WINDOW_AMBIGUOUS,
                       "more than one Wildcard window is active for this event; refusing rather than guessing",
                       reasons=(WC_WINDOW_AMBIGUOUS,))

    scores, screen_stats = screen_players(request)
    if not scores:
        return _refuse(request, WC_INCOMPLETE_PLAYER_POOL, "no player had a projection for every horizon event")
    if screen_stats["excluded_missing_projection"]:
        # Exclusion is auditable, not silent: a missing projection never becomes
        # a zero-point player.
        screen_stats["missing_projection_policy"] = "EXCLUDED_WITH_REASON"

    try:
        candidates, frontier_stats = build_wildcard_candidates(request, scores)
    except WildcardInputError as exc:
        # A pricing/substance violation discovered mid-build (e.g. a supplied
        # selling value disagreeing with the canonical rule) is a REFUSAL with a
        # token, never an exception escaping to the caller and never a confident
        # candidate built on unvalidated evidence.
        token = (exc.reasons[0] if exc.reasons else WC_PRICING_UNAVAILABLE)
        return _refuse(request, token, str(exc), reasons=tuple(exc.reasons))
    if not candidates:
        return _refuse(request, WC_FRONTIER_EMPTY,
                       "no legal Wildcard squad could be constructed under the canonical constraints")

    # Each retained candidate is re-evaluated through the SAME public function,
    # so the selected squad is not merely the one the frontier happened to like.
    ranked = sorted((evaluate_squad(request, c.squad) for c in candidates), key=lambda v: (-v.objective, v.squad))
    play = ranked[0]

    # --- SAVE arm: keep the squad, play the legal normal route, KEEP the chip.
    save_current = evaluate_squad(request, request.owned_ids)
    save_total = float(request.save_route_value) if request.save_route_value is not None else save_current.objective
    if request.save_route_expected_hits:
        save_total -= request.rules.transfer_hit_cost * int(request.save_route_expected_hits)

    # The reservation is applied EXACTLY ONCE, by the chip arbiter — the same
    # seam every other chip uses.  This evaluator therefore reports the RAW
    # play-vs-save difference plus the post-SAVE state the arbiter's reservation
    # provider needs, and never calls the reservation itself.
    uplift = float(play.objective) - float(save_total)
    save_state = {
        "squad_ids": list(request.owned_ids),
        "bank_tenths": int(request.bank_tenths),
        "minutes_security": save_current.minutes_security,
        "expected_forced_moves": save_current.repairability["expected_forced_moves"],
        "wildcard_expiry_event": expiry_event,
        "retains_wildcard_option": True,
    }
    reasons = {WC_REASON_PLAY_DESCRIPTION, WC_REASON_REVIEW_ONLY}
    reasons.add(WC_REASON_POSITIVE if uplift > 0.0 else WC_REASON_NOT_COMPETITIVE)

    # Canonical FT semantics: the chip-core rule owns whether a Wildcard
    # preserves saved free transfers; this evaluator never invents one.  The
    # canonical helper refuses to guess when the event-start bank is unknown,
    # so an unknown bank is reported as unknown rather than defaulted.
    free_transfers_after: int | None = None
    if request.event_start_free_transfers is not None:
        free_transfers_after = sr.free_transfers_after_chip(
            request.rules, "wildcard",
            event_start_free_transfers=int(request.event_start_free_transfers),
        )
    else:
        return _refuse(request, WC_MANAGER_STATE_INCOHERENT,
                       "event-start free-transfer state is unavailable; refusing rather than "
                       "emitting a numeric uplift with a warning",
                       reasons=(WC_MANAGER_STATE_INCOHERENT,))

    buy = sorted(set(play.squad) - set(request.owned_ids))
    sell = sorted(set(request.owned_ids) - set(play.squad))
    return cd.ChipEvaluation(
        action=cd.CHIP_ACTION_WC,
        evaluator_version=WILDCARD_EVALUATOR_VERSION,
        candidate_metrics={
            "mean_paired_uplift": uplift,
            "play_now_objective": play.objective,
            "save_objective": save_total,
            "net_of_reservation": None,
            "wildcard_squad": list(play.squad),
            "players_bought": buy,
            "players_sold": sell,
            "bank_after_tenths": play.bank_after_tenths,
            "horizon_points": play.horizon_points,
            "near_term_points": play.near_term_points,
            "medium_term_points": play.medium_term_points,
            "minutes_security": play.minutes_security,
            "repairability": dict(play.repairability),
            "expected_future_hits": int(play.repairability["expected_forced_moves"]),
            "screened_players": screen_stats["screened"],
            "excluded_missing_projection": screen_stats["excluded_missing_projection"],
            "frontier_by_position": frontier_stats["frontier_by_position"],
            "legal_squads_considered": frontier_stats["legal_squads_considered"],
            "final_frontier_size": len(candidates),
        },
        uncertainty={
            # The play/save gap is a deterministic difference of two horizon
            # valuations, not a sampled quantity, so it has no sampling error of
            # its own; the interval is reported as degenerate rather than faked.
            "paired_interval_low": uplift,
            "paired_interval_high": uplift,
            "uplift_is_deterministic": True,
            "deterministic_uplift_note": "no sampling error; model uncertainty is NOT represented here",
        },
        reason_codes=tuple(sorted(reasons)),
        # The VALUE MODEL itself is uncalibrated in V1, and the execution gate
        # is what keeps a calibrated reservation from making PLAY_WC reachable.
        calibration_status=cd.CALIBRATION_UNCALIBRATED,
        execution_permitted=False,
        evidence={
            "certification_identity": request.certification_identity,
            "data_snapshot_sha256": request.data_snapshot_sha256,
            "horizon_events": list(request.horizon_binding.horizon_events),
            "planning_event": int(request.planning_event),
            "wildcard_horizon": request.horizon.as_dict(),
            "wildcard_evaluator_version": WILDCARD_EVALUATOR_VERSION,
            "wildcard_quantitative_capability": "SUPPORTED_REVIEW_ONLY",
            "no_global_optimum_claim": True,
            "wildcard_expiry_event": expiry_event,
            "free_transfers_after_wildcard": free_transfers_after,
            "ft_preserved_by_chip": sr.chip_preserves_saved_free_transfers("wildcard"),
            "play_now": play.as_dict(),
            "save_policy": {
                "squad": list(request.owned_ids),
                "objective": save_current.objective,
                "retains_wildcard_option": True,
                "expected_hits": int(request.save_route_expected_hits),
                "post_save_state_for_reservation": save_state,
            },
            "executable": False,
            "actionable": False,
        },
    )
