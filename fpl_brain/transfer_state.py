"""Phase 7A — deterministic FPL transfer state engine (accounting + legality).

Pure, immutable, in-memory.  It takes a derived :class:`RouteState` (built from
the canonical PlanningContext) and applies an ATOMIC :class:`TransferBatch`
against a caller-supplied :class:`PriceSnapshot`, returning the post-transfer
state, the Gameweek scoring deduction, and the carried state for the next event.

It deliberately does NOT rank players, generate candidates, compare routes, model
chips, forecast prices, or make any recommendation.  All money is integer tenths
of £m — never binary floating point.
"""

from __future__ import annotations

from collections.abc import Mapping as AbcMapping
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Iterable, Mapping

from .season_rules import ChipFreeTransferError, FT_PRESERVING_CHIPS, NON_TRANSFER_CHIPS

TRANSFER_RULES_VERSION = "fpl_transfer_rules_2026_27_v1"

# Verified against the official FPL bootstrap API ``game_settings`` on
# 2026-09-11 (https://fantasy.premierleague.com/api/bootstrap-static/):
#   transfers_cap = 20, transfers_sell_on_fee = 0.5,
#   max_extra_free_transfers = 4, squad_team_limit = 3, squad_squadsize = 15,
#   element_sell_at_purchase_price = false, ui_currency_multiplier = 10.
# The 1-free-transfer-per-GW allowance and the -4 hit are not exposed by that
# machine-readable endpoint (the /help/rules page is client-rendered); they are
# recorded here as explicit, single-point constants with that provenance note.
MAX_TRANSFERS_PER_EVENT = 20
SELL_ON_FEE = 0.5
MAX_EXTRA_FREE_TRANSFERS = 4
MAX_STORED_FREE_TRANSFERS = 1 + MAX_EXTRA_FREE_TRANSFERS  # 5
FREE_TRANSFERS_PER_EVENT = 1
HIT_POINTS_PER_EXTRA_TRANSFER = 4
SQUAD_SIZE = 15
SQUAD_TEAM_LIMIT = 3
POSITION_COMPOSITION = {"GKP": 2, "DEF": 5, "MID": 5, "FWD": 3}
OUTFIELD_POSITIONS = ("DEF", "MID", "FWD")

TRANSFER_RULE_FLAGS = (
    "NORMAL_TRANSFERS_ONLY_NO_CHIP_TRANSITION",
    "ONE_ATOMIC_BATCH_PER_EVENT",
    "NO_PRICE_FORECAST_EXPLICIT_SNAPSHOT_REQUIRED",
)


def selling_price_tenths(purchase_price: int, current_market_price: int) -> int:
    """Canonical FPL selling price in integer tenths of £m.

    ``current <= purchase`` → current; otherwise purchase plus half the profit
    (floored).  Equivalent to ``purchase + floor((current - purchase) / 2)``.
    """

    purchase = int(purchase_price)
    current = int(current_market_price)
    if current <= purchase:
        return current
    return purchase + (current - purchase) // 2


@dataclass(frozen=True)
class PriceSnapshot:
    """Explicit per-event market prices; a transition never queries live data."""

    event: int
    prices: Mapping[int, int]
    snapshot_id: str | None = None

    def price(self, player_id: int) -> int | None:
        value = self.prices.get(int(player_id))
        return None if value is None else int(value)

    def identity(self) -> str:
        """Canonical identity of this snapshot's prices.

        ``snapshot_id`` short-circuits; production scenario builders set it so the
        identity costs nothing per candidate batch.  The fallback is memoised on the
        EXACT content ``(event, sorted prices)``, so it stays correct for a snapshot
        constructed without an id and never hashes the same content twice.
        """

        if self.snapshot_id:
            return self.snapshot_id
        items = tuple(sorted((int(pid), int(price)) for pid, price in self.prices.items()))
        return _canonical_prices_identity(int(self.event), items)


@dataclass(frozen=True)
class PlayerMeta:
    """Position and club for a player id (incoming players need this)."""

    player_id: int
    position: str
    club_id: int


@dataclass(frozen=True)
class RoutePlayer:
    player_id: int
    position: str
    club_id: int
    purchase_price_tenths: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "player_id": int(self.player_id),
            "position": self.position,
            "club_id": int(self.club_id),
            "purchase_price_tenths": int(self.purchase_price_tenths),
        }


@dataclass(frozen=True)
class RouteState:
    """Derived prospective manager state (NOT a persisted manager database).

    ``event=G`` means the manager state immediately BEFORE the Gameweek-G
    deadline.  Never mutate a PlanningContext; never mutate this object.

    ``free_transfers`` is the FT bank available for the CURRENT event.
    ``event_start_free_transfers`` is the FT bank the manager held at the START
    of the event, before any current-event transfer (official or user-confirmed).
    The two differ when transfers have already been made; the difference matters
    because a Wildcard/Free Hit played after those transfers must preserve the
    EVENT-START bank, not the remaining count.  ``None`` means "not explicitly
    recorded" and callers must refuse to infer it.
    """

    event: int
    players: tuple[RoutePlayer, ...]
    bank_tenths: int
    free_transfers: int
    chip_state: tuple = ()
    transfers_made_this_event: int = 0
    cumulative_hit_points: int = 0
    route_actions: tuple = ()
    event_start_free_transfers: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "players", tuple(sorted(self.players, key=lambda p: int(p.player_id))))

    def by_id(self) -> dict[int, RoutePlayer]:
        return {int(p.player_id): p for p in self.players}

    def position_counts(self) -> dict[str, int]:
        counts = {position: 0 for position in (*OUTFIELD_POSITIONS, "GKP")}
        for player in self.players:
            if player.position in counts:
                counts[player.position] += 1
        return counts

    def club_counts(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        for player in self.players:
            counts[int(player.club_id)] = counts.get(int(player.club_id), 0) + 1
        return counts

    def squad_hash(self) -> str:
        from . import analytics

        return analytics.canonical_hash([
            {
                "player_id": int(p.player_id), "position": p.position,
                "club_id": int(p.club_id), "purchase_price_tenths": int(p.purchase_price_tenths),
            }
            for p in self.players
        ])


@dataclass(frozen=True)
class TransferAction:
    out_player_id: int
    in_player_id: int


@dataclass(frozen=True)
class TransferBatch:
    """All transfers before one Gameweek deadline, evaluated atomically."""

    actions: tuple[TransferAction, ...] = ()

    @staticmethod
    def roll() -> "TransferBatch":
        return TransferBatch(())

    def __len__(self) -> int:
        return len(self.actions)

    def out_ids(self) -> tuple[int, ...]:
        return tuple(int(a.out_player_id) for a in self.actions)

    def in_ids(self) -> tuple[int, ...]:
        return tuple(int(a.in_player_id) for a in self.actions)


@dataclass(frozen=True)
class TransferTransitionResult:
    ok: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    before_state: RouteState
    batch: TransferBatch
    price_snapshot_id: str
    sale_proceeds_tenths: int
    purchase_cost_tenths: int
    bank_before_tenths: int
    bank_after_tenths: int
    ft_before: int
    ft_used: int
    paid_transfers: int
    hit_points: int
    squad_after: RouteState | None
    next_event_state: RouteState | None
    flags: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "event": int(self.before_state.event),
            "batch": [{"out": int(a.out_player_id), "in": int(a.in_player_id)} for a in self.batch.actions],
            "batch_size": len(self.batch),
            "price_snapshot_id": self.price_snapshot_id,
            "sale_proceeds_tenths": int(self.sale_proceeds_tenths),
            "purchase_cost_tenths": int(self.purchase_cost_tenths),
            "bank_before_tenths": int(self.bank_before_tenths),
            "bank_after_tenths": int(self.bank_after_tenths),
            "ft_before": int(self.ft_before),
            "ft_used": int(self.ft_used),
            "paid_transfers": int(self.paid_transfers),
            "hit_points": int(self.hit_points),
            "squad_after": None if self.squad_after is None else [p.as_dict() for p in self.squad_after.players],
            "next_event_state": None if self.next_event_state is None else {
                "event": int(self.next_event_state.event),
                "bank_tenths": int(self.next_event_state.bank_tenths),
                "free_transfers": int(self.next_event_state.free_transfers),
                "squad_hash": self.next_event_state.squad_hash(),
            },
            "flags": list(self.flags),
        }


def next_event_free_transfers(ft_before: int, transfer_count: int) -> int:
    """Normal (no-chip) next-Gameweek FT: bank what is left, then add one, cap 5."""

    remaining_bankable = max(0, int(ft_before) - int(transfer_count))
    return min(MAX_STORED_FREE_TRANSFERS, remaining_bankable + FREE_TRANSFERS_PER_EVENT)


#: Chips that make a whole Gameweek's transfers free and retain the saved FT
#: state (Wildcard / Free Hit), and the team chips that leave normal weekly
#: accrual in place.  Both are the CANONICAL tuples from ``season_rules`` -- one
#: definition, so the state layer cannot drift from the rule layer.  The rule
#: itself lives in ``fpl_brain.season_rules.free_transfers_after_chip``.
CHIP_FT_PRESERVING = FT_PRESERVING_CHIPS
CHIP_BOOSTING = NON_TRANSFER_CHIPS

#: Raised when a chip FT transition lacks the explicitly recorded event-start bank.
ChipFTTransitionError = ChipFreeTransferError


def chip_next_event_free_transfers(
    chip: str,
    *,
    event_start_free_transfers: int | None,
    ft_before: int | None = None,
    transfer_count: int = 0,
) -> int:
    """Free transfers entering the next Gameweek after a chip is played.

    Thin state-engine wrapper around the canonical rule in
    :func:`fpl_brain.season_rules.free_transfers_after_chip` (Wildcard/Free Hit
    retain the saved FT state; no weekly +1 accrual; the retained value is the
    explicitly recorded EVENT-START bank).
    """

    from .season_rules import SeasonRules, free_transfers_after_chip

    return free_transfers_after_chip(
        SeasonRules(season="2026/27"),
        chip,
        event_start_free_transfers=event_start_free_transfers,
        free_transfers_available=ft_before,
        free_transfers_used=int(transfer_count),
    )


def hit_points_for(ft_before: int, transfer_count: int) -> tuple[int, int, int]:
    """Return (free_used, paid_transfers, hit_points)."""

    free_used = min(int(ft_before), int(transfer_count))
    paid = max(0, int(transfer_count) - int(ft_before))
    return free_used, paid, HIT_POINTS_PER_EXTRA_TRANSFER * paid


def apply_transfer_batch(
    state: RouteState,
    batch: TransferBatch,
    price_snapshot: PriceSnapshot,
    player_meta: Mapping[int, PlayerMeta] | Iterable[PlayerMeta],
    *,
    active_chips: Iterable[str] = (),
) -> TransferTransitionResult:
    """Apply one atomic batch; on ANY error the original state is returned unchanged."""

    meta = normalise_meta(player_meta)
    errors: list[str] = []
    warnings: list[str] = list(TRANSFER_RULE_FLAGS)
    actions = tuple(batch.actions)

    if state.transfers_made_this_event:
        errors.append(
            "MULTIPLE_BATCHES_PER_EVENT_UNSUPPORTED: this state already applied "
            f"{state.transfers_made_this_event} transfer(s) in event {state.event}; use ONE atomic batch per event"
        )
    chips = tuple(str(chip) for chip in active_chips)
    if chips:
        errors.append(
            f"CHIP_TRANSITION_NOT_MODELLED: active chip(s) {sorted(chips)} for event {state.event} are out of scope"
        )

    event_transfer_count = int(state.transfers_made_this_event) + len(actions)
    if event_transfer_count > MAX_TRANSFERS_PER_EVENT:
        errors.append(f"TRANSFER_CAP_EXCEEDED: {event_transfer_count} > {MAX_TRANSFERS_PER_EVENT}")

    out_ids = batch.out_ids()
    in_ids = batch.in_ids()
    if len(set(out_ids)) != len(out_ids):
        errors.append("DUPLICATE_OUTGOING")
    if len(set(in_ids)) != len(in_ids):
        errors.append("DUPLICATE_INCOMING")
    same = set(out_ids) & set(in_ids)
    if same:
        errors.append(f"SAME_PLAYER_OUT_AND_IN: {sorted(same)}")

    squad = state.by_id()
    owned = set(squad)
    for pid in out_ids:
        if pid not in owned:
            errors.append(f"OUTGOING_NOT_OWNED: {pid}")
    for pid in in_ids:
        if pid in owned or pid in set(out_ids):
            errors.append(f"INCOMING_ALREADY_OWNED: {pid}")

    out_players = [squad[pid] for pid in out_ids if pid in squad]
    in_players: list[PlayerMeta] = []
    for pid in in_ids:
        if pid not in meta:
            errors.append(f"MISSING_PLAYER_META: {pid}")
            continue
        in_players.append(meta[pid])

    out_counts: dict[str, int] = {}
    for player in out_players:
        out_counts[player.position] = out_counts.get(player.position, 0) + 1
    in_counts: dict[str, int] = {}
    for player in in_players:
        in_counts[player.position] = in_counts.get(player.position, 0) + 1
    for position in (*OUTFIELD_POSITIONS, "GKP"):
        if out_counts.get(position, 0) != in_counts.get(position, 0):
            errors.append(
                f"POSITION_MULTISET_MISMATCH: {position} out={out_counts.get(position, 0)} "
                f"in={in_counts.get(position, 0)}"
            )

    sale_proceeds = 0
    for player in out_players:
        current = price_snapshot.price(player.player_id)
        if current is None:
            errors.append(f"MISSING_OUTGOING_PRICE: {player.player_id}")
            continue
        sale_proceeds += selling_price_tenths(player.purchase_price_tenths, current)
    purchase_cost = 0
    for player in in_players:
        current = price_snapshot.price(player.player_id)
        if current is None:
            errors.append(f"MISSING_INCOMING_PRICE: {player.player_id}")
            continue
        purchase_cost += current

    bank_after = int(state.bank_tenths) + sale_proceeds - purchase_cost
    if bank_after < 0:
        errors.append(f"INSUFFICIENT_BANK: bank_after={bank_after}")

    # Final squad legality (atomic: intermediate states are irrelevant).
    final_players = [p for p in state.players if int(p.player_id) not in set(out_ids)]
    final_players.extend(
        RoutePlayer(meta.player_id, meta.position, meta.club_id,
                    price_snapshot.price(meta.player_id) or 0)
        for meta in in_players
    )
    final_ids = [int(p.player_id) for p in final_players]
    if len(final_ids) != SQUAD_SIZE or len(set(final_ids)) != SQUAD_SIZE:
        errors.append(f"FINAL_SQUAD_NOT_15_UNIQUE: {len(set(final_ids))}")
    final_counts = {position: 0 for position in (*OUTFIELD_POSITIONS, "GKP")}
    for player in final_players:
        if player.position in final_counts:
            final_counts[player.position] += 1
    for position, required in POSITION_COMPOSITION.items():
        if final_counts.get(position, 0) != required:
            errors.append(f"FINAL_POSITION_INVALID: {position}={final_counts.get(position, 0)} != {required}")
    club_counts: dict[int, int] = {}
    for player in final_players:
        club_counts[int(player.club_id)] = club_counts.get(int(player.club_id), 0) + 1
    over = {club: count for club, count in club_counts.items() if count > SQUAD_TEAM_LIMIT}
    if over:
        errors.append(f"CLUB_LIMIT_EXCEEDED: {over}")

    ft_before = int(state.free_transfers)
    free_used, paid, hit_points = hit_points_for(ft_before, event_transfer_count)

    if errors:
        return TransferTransitionResult(
            ok=False, errors=tuple(sorted(set(errors))), warnings=tuple(warnings),
            before_state=state, batch=batch, price_snapshot_id=price_snapshot.identity(),
            sale_proceeds_tenths=sale_proceeds, purchase_cost_tenths=purchase_cost,
            bank_before_tenths=int(state.bank_tenths), bank_after_tenths=bank_after,
            ft_before=ft_before, ft_used=free_used, paid_transfers=paid, hit_points=hit_points,
            squad_after=None, next_event_state=None, flags=tuple(warnings),
        )

    squad_after = RouteState(
        event=int(state.event), players=tuple(final_players), bank_tenths=bank_after,
        free_transfers=ft_before, chip_state=state.chip_state,
        transfers_made_this_event=event_transfer_count,
        cumulative_hit_points=int(state.cumulative_hit_points) + hit_points,
        route_actions=tuple(state.route_actions) + actions,
        event_start_free_transfers=state.event_start_free_transfers,
    )
    next_ft = next_event_free_transfers(ft_before, event_transfer_count)
    next_event_state = RouteState(
        event=int(state.event) + 1, players=tuple(final_players), bank_tenths=bank_after,
        free_transfers=next_ft, chip_state=state.chip_state,
        transfers_made_this_event=0,
        cumulative_hit_points=int(state.cumulative_hit_points) + hit_points,
        route_actions=tuple(state.route_actions) + actions,
        # The NEXT event's start bank is what it carries in, i.e. the rolled-over
        # FT — NOT the previous event's start value.  (squad_after above keeps the
        # CURRENT event's start value, which is the one a same-event Wildcard
        # preservation must use.)
        event_start_free_transfers=next_ft,
    )
    return TransferTransitionResult(
        ok=True, errors=(), warnings=tuple(warnings), before_state=state, batch=batch,
        price_snapshot_id=price_snapshot.identity(), sale_proceeds_tenths=sale_proceeds,
        purchase_cost_tenths=purchase_cost, bank_before_tenths=int(state.bank_tenths),
        bank_after_tenths=bank_after, ft_before=ft_before, ft_used=free_used,
        paid_transfers=paid, hit_points=hit_points, squad_after=squad_after,
        next_event_state=next_event_state, flags=tuple(warnings),
    )


@lru_cache(maxsize=32)
def _canonical_prices_identity(event: int, items: tuple) -> str:
    """``analytics.canonical_hash`` over one price map, memoised on its exact content.

    The payload is constructed exactly as ``PriceSnapshot.identity`` always built it,
    so the returned string is unchanged; only the number of times it is computed
    changes.
    """

    from . import analytics

    return analytics.canonical_hash({
        "event": int(event),
        "prices": {str(pid): int(price) for pid, price in items},
    })


def price_snapshot_identity(event: int, prices: Mapping[int, Any]) -> str:
    """The canonical identity ``PriceSnapshot(event, prices).identity()`` would return.

    Lets a scenario builder stamp ``snapshot_id`` once so the per-batch identity is
    free.  Delegates to the same memoised content hash, so the string is identical
    to the un-stamped path.
    """

    return _canonical_prices_identity(
        int(event), tuple(sorted((int(pid), int(price)) for pid, price in prices.items()))
    )


class NormalisedMeta(dict):
    """The exact dict ``normalise_meta`` returns, tagged so it is never rebuilt.

    A search applies one ``player_meta`` mapping to ~135,000 candidate batches.
    Rebuilding the ~657-entry dict on every call is pure repeated work: the value
    is a pure function of the input mapping.  Tagging the RESULT (a dict subclass,
    so every existing ``dict``/``Mapping`` consumer is unaffected) lets a hot
    caller hand the already-normalised mapping back in and makes the normalisation
    a no-op, without a process-global memo whose key would have to be the input's
    identity — which is not sound.

    The contents are identical to what ``normalise_meta`` produces, so no reader
    can observe a difference beyond ``type(result)``.
    """


def normalise_meta(
    player_meta: Mapping[int, PlayerMeta] | Iterable[PlayerMeta],
) -> NormalisedMeta:
    """Canonical ``{int player_id: PlayerMeta}`` mapping for one search.

    Call this ONCE per search and pass the result to every ``apply_transfer_batch``
    call.  ``apply_transfer_batch`` detects the tagged result and skips the rebuild.
    """

    if isinstance(player_meta, NormalisedMeta):
        return player_meta
    if isinstance(player_meta, AbcMapping):
        return NormalisedMeta(
            (int(pid), meta if isinstance(meta, PlayerMeta)
             else PlayerMeta(int(pid), meta["position"], meta["club_id"]))
            for pid, meta in player_meta.items()
        )
    return NormalisedMeta((int(meta.player_id), meta) for meta in player_meta)


def _normalise_meta(player_meta: Mapping[int, PlayerMeta] | Iterable[PlayerMeta]) -> dict[int, PlayerMeta]:
    return normalise_meta(player_meta)
