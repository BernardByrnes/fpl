"""Production construction path for ``FreeHitRequest``.

This module is the SEAM between accepted production state and the Free Hit
evaluator.  It assembles already-accepted contracts; it creates no Free Hit
architecture of its own.

    canonical PlanningContext squad  ─┐
    canonical acquisition basis        │
    canonical bank / free transfers    ├─► build_free_hit_request ─► FreeHitRequest
    canonical positions and clubs      │                                    │
    canonical PIT market prices        │                                    ▼
    official accepted player pool      │                            evaluate_free_hit
    certified H1 worlds + identity   ──┘                                    │
                                                                            ▼
                                                                     chip arbiter

Two deliberate boundaries:

  * ``free_hit_manager_state(conn, ...)`` is the ONLY part that touches the
    database, and it reads canonical accessors read-only.
  * ``build_free_hit_request(...)`` is assembly plus verification.  It never
    substitutes a value it could not source: a missing basis, a missing price, a
    missing pool or a missing snapshot refuses rather than being defaulted.

THE PERMANENT SQUAD IS NOT A CALLER ARGUMENT
--------------------------------------------
The fifteen, their basis, the bank and the free-transfer bank come from the
accepted planning/repository authorities, and positions and clubs come from the
accepted player authority.  A caller cannot relabel a position, reprice a player
or assert a bank: when the connection the state was derived from is supplied,
``build_free_hit_request`` RE-DERIVES the canonical state and refuses on any
disagreement -- the same discipline the Bench Boost adapter uses.

PRICES ARE POINT-IN-TIME
------------------------
Every price is read through ``analytics.snapshot_as_of`` -- "the freshest
official snapshot captured at or before the cutoff" -- rather than MAX(captured_at),
so a historical replay cannot see a later price.  Price evidence and predictive
evidence are bound to the same cutoff, so a post-cutoff price can never be
paired with a historical prediction world.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from . import chip_decision as cd
from . import chip_free_hit as fh
from . import manager_lineup as ml
from . import planning as planning_module
from . import repositories as repo
from . import season_rules as sr
from .chip_wildcard import (
    WildcardPoolBinding,
    WildcardPredictiveIdentity,
    WildcardWorldInputs,
    pool_binding_from_store,
)

FREE_HIT_ADAPTER_VERSION = "free_hit_adapter_v1.0.0"

FH_MANAGER_STATE_MISSING = "FREE_HIT_PRODUCTION_MANAGER_STATE_MISSING"
FH_CALLER_STATE_DISAGREES = "FREE_HIT_CALLER_STATE_DISAGREES_WITH_CANONICAL"
FH_CANONICAL_AUTHORITY_REQUIRED = "FREE_HIT_CANONICAL_AUTHORITY_REQUIRED"
FH_CERTIFIED_INPUTS_INVALID = "FREE_HIT_CERTIFIED_INPUTS_INVALID"
FH_CUTOFF_MISMATCH = "FREE_HIT_PRICE_PREDICTION_CUTOFF_MISMATCH"


class FreeHitAdapterError(fh.FreeHitInputError):
    """The production adapter could not assemble an authoritative request."""


# ---------------------------------------------------------------------------
# Authoritative manager state (the ONLY database-touching part)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FreeHitManagerState:
    """The manager facts the adapter needs, sourced canonically."""

    entry_id: int
    planning_event: int
    cutoff: str
    owned_ids: tuple[int, ...]
    purchase_price_tenths: Mapping[int, int]
    bank_tenths: int
    event_start_free_transfers: int
    positions: Mapping[int, str]
    clubs: Mapping[int, int]
    market_price_tenths: Mapping[int, int]
    market_price_basis: str = "UNKNOWN"
    cached_selling_price_tenths: Mapping[int, int] = field(default_factory=dict)

    def permanent_state(self) -> fh.FreeHitPermanentState:
        return fh.FreeHitPermanentState(
            event=int(self.planning_event),
            owned_ids=tuple(sorted(int(p) for p in self.owned_ids)),
            purchase_price_tenths={int(k): int(v) for k, v in self.purchase_price_tenths.items()},
            bank_tenths=int(self.bank_tenths),
            event_start_free_transfers=int(self.event_start_free_transfers),
            positions={int(k): str(v) for k, v in self.positions.items()},
            clubs={int(k): int(v) for k, v in self.clubs.items()},
        )

    def problems(self) -> list[str]:
        found: list[str] = list(self.permanent_state().problems())
        if not str(self.cutoff or "").strip():
            found.append("no decision cutoff")
        # Prices are a PRICING concern, not a state-consistency one: an unpriced
        # player is reported as a pricing failure by the contract check so the
        # refusal token names the real cause.
        return found


def free_hit_manager_state(
    conn: sqlite3.Connection,
    entry_id: int,
    planning_event: int,
    *,
    cutoff: str,
    as_of: str | None = None,
    player_ids: Sequence[int] | None = None,
) -> FreeHitManagerState:
    """Source the manager facts from canonical accessors, read-only.

    The squad, the acquisition basis, the bank and the free-transfer bank come
    from the accepted planning/repository authorities; positions and clubs come
    from ``route_comparator.load_player_meta``; prices come from the canonical
    point-in-time accessor at ``cutoff``.
    """

    from . import route_comparator as rc

    context = planning_module.get_planning_context(conn, int(entry_id), int(planning_event), as_of)
    state = repo.manager_planning_state(conn, int(entry_id), int(planning_event), as_of)
    squad = tuple(sorted(int(row["player_id"]) for row in (context.squad.get("players") or ())))

    bank = state.get("bank")
    event_start_ft = state.get("event_start_free_transfers")
    if bank is None or event_start_ft is None:
        raise FreeHitAdapterError(
            f"{FH_MANAGER_STATE_MISSING}: bank or event-start free transfers unavailable from "
            f"canonical manager state (bank={bank!r}, ft={event_start_ft!r})",
            reasons=(FH_MANAGER_STATE_MISSING,),
        )

    basis: dict[int, int] = {}
    cached: dict[int, int] = {}
    for row in (context.selling_prices or ()):
        pid = int(row["player_id"])
        if row.get("purchase_price") is not None:
            basis[pid] = int(row["purchase_price"])
        if row.get("effective_selling_price") is not None:
            cached[pid] = int(row["effective_selling_price"])

    # Positions and clubs are canonical facts, and the club is the one fact the
    # manager-world resolver discards entirely.
    universe = tuple(sorted({int(p) for p in (player_ids or ())} | set(squad)))
    meta = rc.load_player_meta(conn, universe)
    missing_meta = sorted(pid for pid in universe if int(pid) not in meta)
    if missing_meta:
        raise FreeHitAdapterError(
            f"{FH_MANAGER_STATE_MISSING}: the player authority does not describe player(s) "
            f"{missing_meta[:8]}",
            reasons=(FH_MANAGER_STATE_MISSING,),
        )
    positions = {pid: str(meta[pid].position) for pid in universe}
    clubs = {pid: int(meta[pid].club_id) for pid in universe}

    # Point-in-time prices for the WHOLE eligible universe, not merely the owned
    # squad: an incoming Free Hit player's cost must never rest on a caller's
    # figure, and a historical replay must not see a later price.
    prices = _canonical_market_prices(conn, universe, cutoff)

    return FreeHitManagerState(
        entry_id=int(entry_id),
        planning_event=int(planning_event),
        cutoff=str(cutoff),
        owned_ids=squad,
        purchase_price_tenths=basis,
        bank_tenths=int(bank),
        event_start_free_transfers=int(event_start_ft),
        positions=positions,
        clubs=clubs,
        market_price_tenths=prices,
        market_price_basis=f"analytics.snapshot_as_of@{cutoff}",
        cached_selling_price_tenths=cached,
    )


def _canonical_market_prices(
    conn: sqlite3.Connection, player_ids: Sequence[int], cutoff: str
) -> dict[int, int]:
    """Current official price per player AS OF the decision cutoff."""

    from . import analytics as analytics_module

    prices: dict[int, int] = {}
    for pid in player_ids:
        snapshot = analytics_module.snapshot_as_of(conn, int(pid), str(cutoff))
        if snapshot is None:
            continue
        cost = snapshot.get("now_cost")
        if cost is None:
            continue
        try:
            prices[int(pid)] = int(cost)
        except (TypeError, ValueError):
            continue
    return prices


# ---------------------------------------------------------------------------
# Certified predictive evidence (supplied, then validated)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FreeHitCertifiedInputs:
    """The certified predictive evidence a caller must supply.

    The adapter cannot MANUFACTURE these -- producing certified worlds requires
    running the prediction pipeline, which this task must not do.  It therefore
    accepts them and validates them exhaustively: the H1 matrix must BE the bound
    predictive world in every dimension, the snapshot must be present on both
    sides, and the pool must be the accepted official generation.
    """

    horizon_binding: cd.ChipHorizonBinding
    h1_worlds: WildcardWorldInputs
    world_identity: WildcardPredictiveIdentity
    pool_binding: WildcardPoolBinding


# ---------------------------------------------------------------------------
# The assembly
# ---------------------------------------------------------------------------


def build_free_hit_request(
    manager: FreeHitManagerState,
    certified: FreeHitCertifiedInputs,
    *,
    conn: sqlite3.Connection | None = None,
    as_of: str | None = None,
    allow_unverified_manager_state: bool = False,
    player_ids: Sequence[int] | None = None,
    rules: sr.SeasonRules | None = None,
    chip_available: bool = True,
    calibration_status: str = cd.CALIBRATION_UNCALIBRATED,
    input_uncertainty_flags: Sequence[str] = (),
) -> fh.FreeHitRequest:
    """Assemble an authoritative ``FreeHitRequest``, or refuse.

    Nothing here is defaulted.  In particular:

      * the permanent squad, basis, bank and FT are the ones the canonical
        accessors supplied;
      * when ``conn`` is supplied the canonical state is RE-DERIVED and any
        disagreement refuses, so a caller cannot self-certify a permanent squad,
        a bank, a basis or a position/club label;
      * the certified horizon must be the canonical four-event window, and the
        H1 matrix must carry a NON-EMPTY identity that equals the binding's.
    """

    if conn is None and not allow_unverified_manager_state:
        raise FreeHitAdapterError(
            f"{FH_CANONICAL_AUTHORITY_REQUIRED}: a request must be built against the canonical "
            "manager state; pass the connection, or set allow_unverified_manager_state=True to "
            "declare that this is a simulation with no database",
            reasons=(FH_CANONICAL_AUTHORITY_REQUIRED,),
        )

    problems = manager.problems()
    if problems:
        raise FreeHitAdapterError(
            f"{FH_MANAGER_STATE_MISSING}: the supplied manager state is not usable: "
            + "; ".join(problems),
            reasons=(FH_MANAGER_STATE_MISSING,),
        )

    if conn is not None:
        canonical = free_hit_manager_state(
            conn, int(manager.entry_id), int(manager.planning_event),
            cutoff=str(manager.cutoff), as_of=as_of, player_ids=player_ids,
        )
        disagreements = _state_disagreements(canonical, manager)
        if disagreements:
            raise FreeHitAdapterError(
                f"{FH_CALLER_STATE_DISAGREES}: the supplied manager state is not the canonical one: "
                + "; ".join(disagreements),
                reasons=(FH_CALLER_STATE_DISAGREES,),
            )
        if str(canonical.cutoff) != str(manager.cutoff):
            raise FreeHitAdapterError(
                f"{FH_CUTOFF_MISMATCH}: the supplied cutoff is not the canonical decision cutoff",
                reasons=(FH_CUTOFF_MISMATCH,),
            )

    binding = certified.horizon_binding
    if int(binding.planning_event) != int(manager.planning_event):
        raise FreeHitAdapterError(
            f"{fh.FH_HORIZON_NOT_CANONICAL}: the certified horizon is bound to planning event "
            f"{int(binding.planning_event)} but the manager state is for GW{int(manager.planning_event)}",
            reasons=(fh.FH_HORIZON_NOT_CANONICAL,),
        )

    request = fh.FreeHitRequest(
        permanent=manager.permanent_state(),
        horizon_binding=binding,
        h1_worlds=certified.h1_worlds,
        world_identity=certified.world_identity,
        positions=dict(manager.positions),
        clubs=dict(manager.clubs),
        market_price_tenths=dict(manager.market_price_tenths),
        pool_binding=certified.pool_binding,
        chip_available=bool(chip_available),
        rules=rules if rules is not None else sr.SeasonRules(season="2026/27"),
        calibration_status=str(calibration_status),
        input_uncertainty_flags=tuple(str(flag) for flag in input_uncertainty_flags),
    )
    # Fail closed BEFORE any numeric work, on the shared contract.
    contract = fh.contract_problems(request)
    if contract:
        first = contract[0]
        token = fh.FH_PROJECTION_INVALID
        for candidate in (
            fh.FH_HORIZON_NOT_CANONICAL, fh.FH_DATA_SNAPSHOT_REQUIRED,
            fh.FH_PREDICTIVE_IDENTITY_MISMATCH, fh.FH_WORLD_INPUTS_MALFORMED,
            fh.FH_MANAGER_STATE_INVALID, fh.FH_PRICING_UNAVAILABLE,
        ):
            if first.startswith(candidate):
                token = candidate
                break
        raise FreeHitAdapterError(
            f"{token}: the certified inputs are not a coherent Free Hit context: "
            + "; ".join(contract[:6]),
            reasons=(token,),
        )
    return request


def _state_disagreements(canonical: FreeHitManagerState, supplied: FreeHitManagerState) -> list[str]:
    """Every way a caller's manager state differs from the canonical one."""

    found: list[str] = []
    if tuple(sorted(int(p) for p in supplied.owned_ids)) != tuple(sorted(int(p) for p in canonical.owned_ids)):
        found.append(
            f"permanent squad differs ({len(supplied.owned_ids)} supplied vs "
            f"{len(canonical.owned_ids)} canonical)"
        )
    if int(supplied.bank_tenths) != int(canonical.bank_tenths):
        found.append(f"bank {int(supplied.bank_tenths)} != canonical {int(canonical.bank_tenths)}")
    if int(supplied.event_start_free_transfers) != int(canonical.event_start_free_transfers):
        found.append(
            f"event-start free transfers {int(supplied.event_start_free_transfers)} != canonical "
            f"{int(canonical.event_start_free_transfers)}"
        )
    basis_diff = sorted(
        pid for pid in supplied.owned_ids
        if int(supplied.purchase_price_tenths.get(int(pid)) or -1)
        != int(canonical.purchase_price_tenths.get(int(pid)) or -1)
    )
    if basis_diff:
        found.append(f"acquisition basis of {basis_diff[:8]} is not the canonical one")
    relabelled = sorted(
        pid for pid in canonical.positions
        if str(supplied.positions.get(int(pid))) != str(canonical.positions.get(int(pid)))
    )
    if relabelled:
        found.append(f"position of {relabelled[:8]} is not the canonical one")
    reclubbed = sorted(
        pid for pid in canonical.clubs
        if int(supplied.clubs.get(int(pid)) or 0) != int(canonical.clubs.get(int(pid)) or 0)
    )
    if reclubbed:
        found.append(f"club of {reclubbed[:8]} is not the canonical one")
    repriced = sorted(
        pid for pid in canonical.market_price_tenths
        if int(supplied.market_price_tenths.get(int(pid)) or -1)
        != int(canonical.market_price_tenths.get(int(pid)) or -1)
    )
    if repriced:
        found.append(f"market price of {repriced[:8]} is not the canonical one")
    return found


def resolve_pool_binding(conn: sqlite3.Connection) -> WildcardPoolBinding:
    """The official eligible pool, straight from the accepted-generation store."""

    return pool_binding_from_store(conn)
