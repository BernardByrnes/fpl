"""Production construction path for ``WildcardRequest``.

This module is the SEAM between accepted production state and the Wildcard
evaluator.  It assembles already-accepted contracts; it creates no Wildcard
architecture of its own.

    canonical manager state  ─┐
    official player pool      │
    canonical prices          ├─► build_wildcard_request ─► WildcardRequest
    canonical selling values  │                              │
    certified projections     │                              ▼
    real four-GW SAVE route  ─┘                        evaluate_wildcard
                                                                 │
                                                                 ▼
                                                          chip arbiter

Two deliberate boundaries:

  * ``wildcard_manager_state(conn, ...)`` is the ONLY part that touches the
    database, and it reads canonical accessors read-only.
  * ``build_wildcard_request(...)`` is pure assembly plus validation.  It never
    substitutes a value it could not source: missing authoritative evidence
    refuses rather than being defaulted, and a caller cannot establish
    production truth (player universe, selling values, FT, bank, route) by
    passing a list or a scalar.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from . import chip_decision as cd
from . import chip_wildcard as wc
from . import planning as planning_module
from . import repositories as repo
from . import season_rules as sr
from . import transfer_state as ts
from . import wildcard_save_route
from .candidate_universe import OFFICIAL_PLAYER_POOL_INCOMPLETE

WILDCARD_ADAPTER_VERSION = "wildcard_adapter_v1.0.0"

WC_MANAGER_STATE_MISSING = "WILDCARD_PRODUCTION_MANAGER_STATE_MISSING"
WC_MANAGER_STATE_MISMATCH = "WILDCARD_PRODUCTION_MANAGER_STATE_MISMATCH"


class WildcardAdapterError(wc.WildcardInputError):
    """The production adapter could not assemble an authoritative request."""


# ---------------------------------------------------------------------------
# Authoritative manager state (the ONLY database-touching part)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WildcardManagerState:
    """The manager facts the adapter needs, sourced canonically."""

    entry_id: int
    planning_event: int
    squad_ids: tuple[int, ...]
    bank_tenths: int
    purchase_price_tenths: Mapping[int, int]
    event_start_free_transfers: int
    chip_availability: tuple[Mapping[str, Any], ...]
    market_price_tenths: Mapping[int, int]
    #: Any cached/official selling value present in the store.  EVIDENCE ONLY:
    #: the adapter recomputes canonically and refuses on disagreement.
    cached_selling_price_tenths: Mapping[int, int] = field(default_factory=dict)

    def problems(self) -> list[str]:
        found: list[str] = []
        if not self.squad_ids:
            found.append("no owned squad")
        if len(self.squad_ids) != ts.SQUAD_SIZE:
            found.append(f"squad has {len(self.squad_ids)} players, expected {ts.SQUAD_SIZE}")
        for pid, price in self.market_price_tenths.items():
            if not isinstance(price, int) or price <= 0:
                found.append(f"player {pid} has a non-positive or non-integer price {price!r}")
        if int(self.bank_tenths) < 0:
            found.append("negative bank")
        if int(self.event_start_free_transfers) < 0:
            found.append("negative event-start free transfers")
        if not self.chip_availability:
            found.append("no chip availability rows")
        missing_basis = [pid for pid in self.squad_ids if int(pid) not in self.purchase_price_tenths]
        if missing_basis:
            found.append(f"{len(missing_basis)} owned players have no acquisition basis")
        missing_price = [pid for pid in self.squad_ids if int(pid) not in self.market_price_tenths]
        if missing_price:
            found.append(f"{len(missing_price)} owned players have no canonical market price")
        return found


def wildcard_manager_state(
    conn: sqlite3.Connection, entry_id: int, planning_event: int, *,
    cutoff: str, eligible_ids: Sequence[int] | None = None, as_of: str | None = None
) -> WildcardManagerState:
    """Source the manager facts from canonical accessors, read-only.

    Uses ``planning.get_planning_context`` (squad, acquisitions, selling prices,
    chip definitions) and ``repositories.manager_planning_state`` (bank, FT).
    Anything the canonical state does not supply refuses: the adapter never
    invents a bank, a basis or an FT count.
    """

    context = planning_module.get_planning_context(conn, int(entry_id), int(planning_event), as_of)
    squad = tuple(int(p["player_id"]) for p in (context.squad.get("players") or ()))
    chips = tuple(dict(row) for row in (context.chips or ()))
    state = repo.manager_planning_state(conn, int(entry_id), int(planning_event), as_of)

    bank = state.get("bank")
    event_start_ft = state.get("event_start_free_transfers")
    if bank is None or event_start_ft is None:
        raise WildcardAdapterError(
            f"{WC_MANAGER_STATE_MISSING}: bank or event-start free transfers unavailable "
            f"from canonical manager state (bank={bank!r}, ft={event_start_ft!r})",
            reasons=(WC_MANAGER_STATE_MISSING,),
        )

    basis: dict[int, int] = {}
    for row in (context.selling_prices or ()):
        pid = int(row["player_id"])
        price = row.get("purchase_price")
        if price is not None:
            basis[pid] = int(price)
    cached: dict[int, int] = {}
    for row in (context.selling_prices or ()):
        pid = int(row["player_id"])
        effective = row.get("effective_selling_price")
        if effective is not None:
            cached[pid] = int(effective)

    # Canonicalize prices for the WHOLE eligible universe, not just the owned
    # squad: an incoming Wildcard candidate's affordability must never rest on
    # a caller-supplied price.
    universe = tuple(int(p) for p in (eligible_ids if eligible_ids is not None else squad))
    market = _canonical_market_prices(conn, tuple(sorted(set(universe) | set(squad))), cutoff)
    return WildcardManagerState(
        entry_id=int(entry_id),
        planning_event=int(planning_event),
        squad_ids=squad,
        bank_tenths=int(bank),
        purchase_price_tenths=basis,
        event_start_free_transfers=int(event_start_ft),
        chip_availability=chips,
        market_price_tenths=market,
        cached_selling_price_tenths=cached,
    )


def _canonical_market_prices(
    conn: sqlite3.Connection, player_ids: Sequence[int], cutoff: str
) -> dict[int, int]:
    """Current official price per player AS OF the decision cutoff.

    Uses the canonical point-in-time accessor ``analytics.snapshot_as_of`` --
    "freshest official snapshot captured at or before the cutoff" -- rather than
    MAX(captured_at), which would leak a post-cutoff price into a historical
    replay.  A player with no snapshot at or before the cutoff has NO canonical
    price and is reported as missing rather than being defaulted.
    """

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
            value = int(cost)
        except (TypeError, ValueError):
            continue
        prices[int(pid)] = value
    return prices


# ---------------------------------------------------------------------------
# Certified predictive evidence (supplied, then validated)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WildcardCertifiedInputs:
    """The certified predictive evidence a caller must supply.

    The adapter cannot MANUFACTURE these -- producing certified projections and
    worlds requires running the prediction pipeline, which this task must not do.
    It therefore accepts them and validates them exhaustively: the bindings must
    agree, every row must pass the per-row identity check, and every horizon
    event must carry a complete world matrix.
    """

    horizon: wc.WildcardHorizonSpec
    chip_horizon_binding: Any
    value_horizon_binding: wc.WildcardValueHorizonBinding
    players: Mapping[int, wc.WildcardPlayer]
    worlds_by_event: Mapping[int, wc.WildcardWorldInputs]
    generation: Mapping[str, Any]
    #: Separate persisted PE-9 products. ``generation`` above remains the
    #: accepted official player-pool identity for compatibility; these ids bind
    #: the normal four-event decision and the distinct 6–10 event value product.
    chip_generation_id: str | None = None
    value_generation_id: str | None = None


def _manager_fields_equal(name: str, supplied: Any, expected: Any) -> bool:
    """Compare canonical manager fields using their actual scalar/map shape."""

    if name in {"squad_ids", "chip_availability"}:
        return tuple(supplied) == tuple(expected)
    if name in {
        "purchase_price_tenths",
        "market_price_tenths",
        "cached_selling_price_tenths",
    }:
        return {int(key): int(value) for key, value in dict(supplied).items()} == {
            int(key): int(value) for key, value in dict(expected).items()
        }
    return supplied == expected


# ---------------------------------------------------------------------------
# The assembly
# ---------------------------------------------------------------------------


def build_wildcard_request(
    manager: WildcardManagerState,
    certified: WildcardCertifiedInputs,
    route: wc.WildcardSaveRoute | None = None,
    *,
    rules: sr.SeasonRules,
    data_snapshot_sha256: str,
    reservation: Any | None = None,
    conn: sqlite3.Connection | None = None,
    #: Read-only pinned snapshot for manager, pool and price authorities. ``conn``
    #: remains the certified prediction/evidence store.
    manager_source_conn: sqlite3.Connection | None = None,
    pool_binding: wc.WildcardPoolBinding | None = None,
    canonical_route: Any | None = None,
    #: Authoritative evaluation INPUTS for the canonical route: the world
    #: matrices the route engine used plus the position resolver and run
    #: configuration.  NOT an evaluation RESULT -- the converter calls
    #: ``route_optimizer.exact_evaluate`` itself, so there is no channel for a
    #: caller to supply a numeric SAVE value.
    route_worlds_by_event: Mapping[int, Any] | None = None,
    route_positions_of: Any | None = None,
    route_config: Any | None = None,
    route_events: Sequence[int] | None = None,
    club_of: Mapping[int, int] | None = None,
) -> wc.WildcardRequest:
    """Assemble an authoritative ``WildcardRequest``, or refuse.

    Nothing here is defaulted.  In particular:
      * selling values are RECOMPUTED from the acquisition basis and the
        canonical market price, and any cached value that disagrees refuses;
      * the player universe is validated against the official generation, not
        trusted because a caller supplied it;
      * the certified bindings must agree with each other.
    """

    manager_problems = manager.problems()
    if manager_problems:
        raise WildcardAdapterError(
            f"{WC_MANAGER_STATE_MISSING}: {'; '.join(manager_problems[:6])}",
            reasons=(WC_MANAGER_STATE_MISSING,),
        )
    if conn is not None and canonical_route is None:
        raise WildcardAdapterError(
            f"{wc.WC_SAVE_ROUTE_MISSING}: production SAVE requires the canonical normal route; "
            "a preconstructed WildcardSaveRoute is not authoritative",
            reasons=(wc.WC_SAVE_ROUTE_MISSING,),
        )

    # --- pool identity: the STORE is the authority, not the caller's list
    #
    # In production a connection is supplied and the binding is RESOLVED from
    # the canonical accepted-generation store
    # (repositories.latest_accepted_bootstrap_generation), so a caller cannot
    # choose the eligible ids, the count, the digest or the acceptance flag.
    # A supplied binding is at most evidence: if it disagrees with the store
    # that is a contradiction and it refuses.
    supplied_binding = pool_binding
    authority_conn = manager_source_conn or conn
    if conn is not None:
        resolved_binding = wc.pool_binding_from_store(authority_conn)
        if supplied_binding is not None and supplied_binding.as_dict() != resolved_binding.as_dict():
            raise WildcardAdapterError(
                f"{OFFICIAL_PLAYER_POOL_INCOMPLETE}: the supplied pool binding disagrees with "
                "the accepted official generation recorded in the store",
                reasons=(OFFICIAL_PLAYER_POOL_INCOMPLETE,),
            )
        pool_binding = resolved_binding
        canonical_manager = wildcard_manager_state(
            authority_conn,
            int(manager.entry_id),
            int(manager.planning_event),
            cutoff=str(certified.value_horizon_binding.decision_cutoff),
            eligible_ids=pool_binding.eligible_ids,
            as_of=str(certified.value_horizon_binding.decision_cutoff),
        )
        manager_disagreements: list[str] = []
        for name in ("entry_id", "planning_event", "squad_ids", "bank_tenths",
                     "event_start_free_transfers", "purchase_price_tenths",
                     "market_price_tenths", "cached_selling_price_tenths", "chip_availability"):
            expected = getattr(canonical_manager, name)
            supplied = getattr(manager, name)
            if not _manager_fields_equal(name, supplied, expected):
                manager_disagreements.append(name)
        if manager_disagreements:
            raise WildcardAdapterError(
                f"{WC_MANAGER_STATE_MISMATCH}: supplied manager fields disagree with the pinned "
                f"snapshot at {certified.value_horizon_binding.decision_cutoff}: "
                f"{manager_disagreements}",
                reasons=(WC_MANAGER_STATE_MISMATCH,),
            )
    else:
        # Synthetic/evaluator-level construction.  Still requires an EXPLICIT
        # accepted generation -- absence is never acceptance.
        pool_binding = supplied_binding or wc.pool_binding_from_generation(certified.generation)

    # --- certified evidence must be internally coherent
    value_binding = certified.value_horizon_binding
    binding_problems = value_binding.problems()
    binding_problems.extend(value_binding.disagreements_with(certified.chip_horizon_binding))
    if binding_problems:
        raise WildcardAdapterError(
            f"{wc.WC_VALUE_BINDING_MISMATCH}: {'; '.join(binding_problems[:6])}",
            reasons=(wc.WC_VALUE_BINDING_MISMATCH,),
        )

    if conn is not None:
        # A production Wildcard must bind both of its certified products from
        # the persisted store. The old ``generation`` field is the official
        # player-pool binding, not a prediction-generation authority.
        if not certified.chip_generation_id or not certified.value_generation_id:
            raise WildcardAdapterError(
                f"{wc.WC_VALUE_BINDING_MISMATCH}: production Wildcard requires persisted "
                "four-event and 6–10 event generation ids",
                reasons=(wc.WC_VALUE_BINDING_MISMATCH,),
            )
        from . import generation_store as gs

        try:
            chip_generation = gs.load_generation(conn, str(certified.chip_generation_id))
            value_generation = gs.load_generation(conn, str(certified.value_generation_id))
            chip_report = gs.verify_generation(conn, chip_generation.generation_id)
            value_report = gs.verify_generation(conn, value_generation.generation_id)
        except gs.GenerationRefused as failure:
            raise WildcardAdapterError(
                f"{wc.WC_VALUE_BINDING_MISMATCH}: a required certified generation refused: {failure}",
                reasons=(wc.WC_VALUE_BINDING_MISMATCH,),
            ) from failure
        if not chip_report.get("verified") or not value_report.get("verified"):
            raise WildcardAdapterError(
                f"{wc.WC_VALUE_BINDING_MISMATCH}: a required certified generation did not verify",
                reasons=(wc.WC_VALUE_BINDING_MISMATCH,),
            )
        chip_snapshot = (chip_generation.manifest.get("data_snapshot") or {}).get("sha256")
        value_snapshot = (value_generation.manifest.get("data_snapshot") or {}).get("sha256")
        chip_code = str(chip_generation.manifest.get("code_snapshot_sha256") or "")
        value_code = str(value_generation.manifest.get("code_snapshot_sha256") or "")
        disagreements = []
        if chip_generation.horizon_kind != gs.HORIZON_KIND_FOUR_GW:
            disagreements.append("the chip decision generation is not FOUR_GW")
        if value_generation.horizon_kind != gs.HORIZON_KIND_WILDCARD_VALUE:
            disagreements.append("the Wildcard value generation is not WILDCARD_VALUE")
        if tuple(chip_generation.events) != tuple(certified.chip_horizon_binding.horizon_events):
            disagreements.append("the chip generation events differ from the four-event binding")
        if tuple(value_generation.events) != tuple(certified.horizon.events):
            disagreements.append("the value generation events differ from the Wildcard horizon")
        if int(chip_generation.planning_event) != int(manager.planning_event) or int(
            value_generation.planning_event
        ) != int(manager.planning_event):
            disagreements.append("a certified generation has a different planning event")
        if str(chip_generation.cutoff) != str(value_binding.decision_cutoff) or str(
            value_generation.cutoff
        ) != str(value_binding.decision_cutoff):
            disagreements.append("the two certified generations differ from the Wildcard cutoff")
        if str(chip_snapshot) != str(value_snapshot) or str(chip_snapshot) != str(
            value_binding.data_snapshot_sha256
        ) or str(chip_snapshot) != str(data_snapshot_sha256):
            disagreements.append("the two certified generations differ from the Wildcard data snapshot")
        if not chip_code or chip_code != value_code or chip_code != str(
            value_binding.source_snapshot_sha256
        ):
            disagreements.append("the two certified generations differ from the predictive source identity")
        chip_bundle_identities = {
            str(int(event)): str(
                ((chip_generation.manifest.get("per_event") or {}).get(str(int(event)) or {})
                 or {}).get("bundle_identity") or ""
            )
            for event in chip_generation.events
        }
        expected_chip_certification_identity = gs._certification_identity_for_bundles(
            cutoff=str(chip_generation.cutoff),
            bundle_identities=chip_bundle_identities,
            snapshot_sha256=str(chip_snapshot or ""),
        )
        if str(certified.chip_horizon_binding.certification_identity) != str(
            expected_chip_certification_identity
        ):
            disagreements.append(
                "the chip binding does not name the persisted FOUR_GW bundle certification identity"
            )
        if str(value_binding.prediction_generation) != value_generation.generation_id:
            disagreements.append("the value binding does not name the persisted WILDCARD_VALUE generation")
        if tuple(value_generation.events[:4]) != tuple(chip_generation.events):
            disagreements.append("the normal four events are not the exact prefix of the Wildcard product")
        if disagreements:
            raise WildcardAdapterError(
                f"{wc.WC_VALUE_BINDING_MISMATCH}: " + "; ".join(disagreements[:8]),
                reasons=(wc.WC_VALUE_BINDING_MISMATCH,),
            )

    # --- the universe must cover the whole eligible pool
    missing = [pid for pid in pool_binding.eligible_ids if int(pid) not in certified.players]
    if missing:
        raise WildcardAdapterError(
            f"{OFFICIAL_PLAYER_POOL_INCOMPLETE}: {len(missing)} eligible players have no "
            f"projection row (e.g. {missing[:5]})",
            reasons=(OFFICIAL_PLAYER_POOL_INCOMPLETE,),
        )

    # --- canonical prices for EVERY candidate, cross-checked against the rows
    #
    # The screening universe is the eligible pool, so a row's price must be the
    # authoritative point-in-time price.  A caller-supplied price that disagrees
    # is a contradiction, not a hint: it refuses rather than being overwritten,
    # because silently substituting would hide evidence that the caller's
    # universe is not the certified one.
    missing_price: list[int] = []
    disagreeing: list[tuple[int, int, int]] = []
    for pid in sorted(int(p) for p in certified.players):
        canonical = manager.market_price_tenths.get(int(pid))
        supplied = certified.players[int(pid)].market_price_tenths
        if canonical is None:
            missing_price.append(int(pid))
            continue
        if int(canonical) != int(supplied):
            disagreeing.append((int(pid), int(supplied), int(canonical)))
    if missing_price:
        raise WildcardAdapterError(
            f"{wc.WC_PRICING_UNAVAILABLE}: {len(missing_price)} candidate players have no "
            f"canonical point-in-time price (e.g. {missing_price[:5]})",
            reasons=(wc.WC_PRICING_UNAVAILABLE,),
        )
    if disagreeing:
        pid, supplied, canonical = disagreeing[0]
        raise WildcardAdapterError(
            f"{wc.WC_PRICING_UNAVAILABLE}: player {pid} is priced {supplied} by the caller but "
            f"{canonical} by the canonical point-in-time snapshot "
            f"({len(disagreeing)} disagreement(s))",
            reasons=(wc.WC_PRICING_UNAVAILABLE,),
        )

    # --- canonical selling values, recomputed and cross-checked
    selling: dict[int, int] = {}
    for pid in manager.squad_ids:
        purchase = int(manager.purchase_price_tenths[pid])
        market = int(manager.market_price_tenths[pid])
        canonical = ts.selling_price_tenths(purchase, market)
        cached = manager.cached_selling_price_tenths.get(int(pid))
        if cached is not None and int(cached) != int(canonical):
            raise WildcardAdapterError(
                f"{wc.WC_PRICING_UNAVAILABLE}: player {pid} cached selling value {int(cached)} "
                f"disagrees with the canonical {canonical}",
                reasons=(wc.WC_PRICING_UNAVAILABLE,),
            )
        selling[int(pid)] = int(canonical)

    # --- SAVE authority: the CANONICAL route is the authority, and this adapter
    # constructs the Wildcard SAVE input from it.  A preconstructed
    # WildcardSaveRoute is synthetic evidence and is refused whenever a canonical
    # route (or a production connection) is in play, so a caller cannot
    # self-certify squads, bank, basis, FT or mean_net_core.
    if canonical_route is not None:
        if route is not None:
            raise WildcardAdapterError(
                f"{wc.WC_SAVE_ROUTE_INVALID}: supply either the canonical route or a synthetic "
                "WildcardSaveRoute, not both",
                reasons=(wc.WC_SAVE_ROUTE_INVALID,),
            )
        if route_worlds_by_event is None or route_positions_of is None:
            raise WildcardAdapterError(
                f"{wc.WC_SAVE_ROUTE_INVALID}: the canonical route must be accompanied by its "
                "authoritative evaluation INPUTS (world matrices and the position resolver); "
                "the SAVE value is evaluated internally and is never caller-supplied",
                reasons=(wc.WC_SAVE_ROUTE_INVALID,),
            )
        # The converter validates the route (PASS 1), evaluates that SAME route
        # canonically (PASS 2) and builds the DTO from the canonical values
        # (PASS 3).  Nothing numeric is forwarded from here.
        route = wildcard_save_route.wildcard_save_route_from_canonical_route(
            canonical_route,
            planning_event=int(manager.planning_event), horizon=certified.horizon,
            rules=rules, expected_horizon=certified.horizon.events[:4], club_of=club_of,
            worlds_by_event=route_worlds_by_event,
            positions_of=route_positions_of,
            config=route_config,
            events=route_events,
        )
    # --- build the request; the evaluator's own validation does the rest
    return wc.WildcardRequest(
        planning_event=int(manager.planning_event),
        horizon=certified.horizon,
        players=dict(certified.players),
        positions={int(pid): p.position for pid, p in certified.players.items()},
        owned_ids=tuple(int(p) for p in manager.squad_ids),
        purchase_price_tenths={int(k): int(v) for k, v in manager.purchase_price_tenths.items()},
        selling_price_tenths=selling,
        bank_tenths=int(manager.bank_tenths),
        rules=rules,
        horizon_binding=certified.chip_horizon_binding,
        certification_identity=str(getattr(certified.chip_horizon_binding, "certification_identity", "")),
        data_snapshot_sha256=str(data_snapshot_sha256),
        event_start_free_transfers=int(manager.event_start_free_transfers),
        reservation=reservation,
        chip_availability=tuple(manager.chip_availability),
        value_horizon_binding=value_binding,
        worlds_by_event=dict(certified.worlds_by_event),
        pool_binding=pool_binding,
        save_route=route,
    )
