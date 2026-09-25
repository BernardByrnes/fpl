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
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import certified_bundle as cb
from . import chip_decision as cd
from . import chip_free_hit as fh
from .free_hit_route import FreeHitRouteError
from . import manager_lineup as ml
from . import planning as planning_module
from . import repositories as repo
from . import season_rules as sr
from .chip_wildcard import (
    WildcardWorldInputs,
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
FH_KEYSET_MISMATCH = "FREE_HIT_CALLER_KEYSET_IS_NOT_THE_CANONICAL_UNIVERSE"
FH_POOL_MISMATCH = "FREE_HIT_POOL_IS_NOT_THE_ACCEPTED_GENERATION"
FH_UNIVERSE_INCOMPLETE = "FREE_HIT_CANONICAL_UNIVERSE_INCOMPLETE"
#: Production Free Hit must LOAD the canonical certification.  There is no
#: caller-supplied authority, and no fallback to request-owned values.
FH_CERTIFICATION_REQUIRED = "FREE_HIT_PRODUCTION_CERTIFICATION_REQUIRED"


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
    #: FT AVAILABLE entering H1 (the normal route's starting state).
    free_transfers: int
    event_start_free_transfers: int
    positions: Mapping[int, str]
    clubs: Mapping[int, int]
    market_price_tenths: Mapping[int, int]
    #: The AUTHORITATIVE eligible universe, from the accepted generation.  Caller
    #: maps never define it: a caller that omits a candidate from its own list
    #: must not thereby narrow which players are canonically re-derived.
    eligible_ids: tuple[int, ...] = ()
    pool_generation_identity: str = ""
    pool_generation_id_sha256: str = ""
    market_price_basis: str = "UNKNOWN"
    cached_selling_price_tenths: Mapping[int, int] = field(default_factory=dict)

    def permanent_state(self) -> fh.FreeHitPermanentState:
        return fh.FreeHitPermanentState(
            event=int(self.planning_event),
            owned_ids=tuple(sorted(int(p) for p in self.owned_ids)),
            purchase_price_tenths={int(k): int(v) for k, v in self.purchase_price_tenths.items()},
            bank_tenths=int(self.bank_tenths),
            free_transfers=int(self.free_transfers),
            event_start_free_transfers=int(self.event_start_free_transfers),
            positions={int(k): str(v) for k, v in self.positions.items()},
            clubs={int(k): int(v) for k, v in self.clubs.items()},
        )

    def problems(self) -> list[str]:
        found: list[str] = list(self.permanent_state().problems())
        # EXACT keyset equality against the authoritative universe.  A missing key
        # is a silently dropped player and an extra key is a player that does not
        # officially exist; both are contradictions, not judgement calls.
        expected = set(int(p) for p in self.eligible_ids)
        for label, mapping in (
            ("positions", self.positions), ("clubs", self.clubs),
            ("market prices", self.market_price_tenths),
        ):
            supplied = set(int(k) for k in mapping)
            extra = sorted(supplied - expected)
            missing = sorted(expected - supplied)
            if extra:
                found.append(f"{label} carry {len(extra)} non-official player(s): {extra[:6]}")
            if missing:
                found.append(f"{label} omit {len(missing)} official player(s): {missing[:6]}")
        if not expected:
            found.append("no authoritative eligible universe")
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
    decision_cutoff: str,
    as_of: str | None = None,
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
    available_ft = state.get("free_transfers")
    if bank is None or event_start_ft is None or available_ft is None:
        raise FreeHitAdapterError(
            f"{FH_MANAGER_STATE_MISSING}: bank or event-start free transfers unavailable from "
            f"canonical manager state (bank={bank!r}, ft={available_ft!r}, "
            f"event_start_ft={event_start_ft!r})",
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

    # THE AUTHORITATIVE UNIVERSE comes from the accepted generation, never from a
    # caller's list: narrowing the caller's list must not narrow what is verified.
    pool = resolve_pool_binding(conn)
    universe = tuple(sorted(int(p) for p in pool.eligible_ids))
    for pid in squad:
        if int(pid) not in set(universe):
            raise FreeHitAdapterError(
                f"{FH_UNIVERSE_INCOMPLETE}: the permanently owned player {pid} is not in the accepted "
                "official eligible universe",
                reasons=(FH_UNIVERSE_INCOMPLETE,),
            )
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
    # PIT prices at the CERTIFIED decision cutoff.  A caller's cutoff never
    # drives price retrieval, so a later cutoff cannot pull future prices back
    # into an earlier decision.
    prices = _canonical_market_prices(conn, universe, decision_cutoff)

    return FreeHitManagerState(
        entry_id=int(entry_id),
        planning_event=int(planning_event),
        cutoff=str(decision_cutoff),
        owned_ids=squad,
        purchase_price_tenths=basis,
        bank_tenths=int(bank),
        free_transfers=int(available_ft),
        event_start_free_transfers=int(event_start_ft),
        positions=positions,
        clubs=clubs,
        market_price_tenths=prices,
        eligible_ids=universe,
        pool_generation_identity=str(pool.generation_identity),
        pool_generation_id_sha256=str(pool.generation_id_sha256),
        market_price_basis=f"analytics.snapshot_as_of@{decision_cutoff}",
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
class FreeHitArmRoute:
    """ONE arm's canonical route ingredients, before conversion.

    Deliberately NOT a ``FreeHitRoute``: accepting a converted DTO would bypass
    structural validation, acquisition-basis validation, inter-event continuity,
    terminal-state validation and ``exact_evaluate`` all at once.

    ``certified_worlds_by_event`` holds the ACCEPTED provenance-carrying world
    artifact -- ``chip_wildcard.WildcardWorldInputs`` -- which carries its OWN
    intrinsic event, its OWN predictive identity AND the numeric matrix as ONE
    indivisible object.  There is deliberately no separate matrix map and no
    separate identity map: two parallel containers can be made to disagree, and
    validating one while evaluating the other is exactly the defect this closes.
    """

    partial: Any
    positions_of: Any
    route_config: Any
    #: There is deliberately NO world/matrix parameter.  The numeric matrices are
    #: LOADED by the adapter through the accepted ``route_optimizer.build_event_worlds``
    #: -- content-addressed by the CERTIFIED bundle's run ids -- so a caller cannot
    #: supply numbers at all, and the matrix handed to ``exact_evaluate`` is always
    #: the one the canonical loader produced for that certified event.


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
    #: The two arms' CANONICAL ROUTE INGREDIENTS -- never converted results.  The
    #: adapter calls ``free_hit_route_from_canonical_route`` itself, so a caller has
    #: no channel through which a route VALUE could reach the decision: there is no
    #: ``FreeHitRoute`` parameter anywhere on the production path.
    play: "FreeHitArmRoute"
    save: "FreeHitArmRoute"


# ---------------------------------------------------------------------------
# The assembly
# ---------------------------------------------------------------------------


def build_free_hit_request(
    manager: FreeHitManagerState,
    certified: FreeHitCertifiedInputs,
    *,
    conn: sqlite3.Connection | None = None,
    #: The canonical certification artifact the decision is authorised by.  This
    #: is the ONE production source of decision authority.
    certification_path: str | Path | None = None,
    #: The canonical manager-world cache the route matrices are loaded from.
    world_cache_dir: str | Path | None = None,
    as_of: str | None = None,
    allow_unverified_manager_state: bool = False,
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

    # ── the canonical decision context is LOADED, never supplied ──────────────
    # `certification_path` is the ONE production source of authority: the accepted
    # loader validates the real v2 artifact and `FreeHitDecisionAuthority`
    # recomputes the certification identity from it.  A caller cannot pass an
    # authority object at all, so "B agrees with B, therefore B is trusted" has no
    # production path.  Price retrieval below uses the CERTIFIED cutoff, so a later
    # caller cutoff can never pull post-cutoff prices into this decision.
    if not str(certification_path or "").strip():
        raise FreeHitAdapterError(
            f"{FH_CERTIFICATION_REQUIRED}: production Free Hit must load the canonical certification "
            "artifact; no caller-supplied decision authority is accepted",
            reasons=(FH_CERTIFICATION_REQUIRED,),
        )
    try:
        authority = fh.load_decision_authority(certification_path)
    except fh.FreeHitAuthorityError as exc:
        raise FreeHitAdapterError(
            f"{FH_CERTIFICATION_REQUIRED}: the canonical certification could not be loaded: {exc}",
            reasons=(FH_CERTIFICATION_REQUIRED,),
        ) from exc
    certified_cutoff = str(authority.planning_cutoff or "").strip()
    if not certified_cutoff:
        raise FreeHitAdapterError(
            f"{FH_CUTOFF_MISMATCH}: the certified decision authority carries no planning cutoff",
            reasons=(FH_CUTOFF_MISMATCH,),
        )
    if str(manager.cutoff or "").strip() and str(manager.cutoff) != certified_cutoff:
        raise FreeHitAdapterError(
            f"{FH_CUTOFF_MISMATCH}: the supplied cutoff {manager.cutoff!r} is not the certified decision "
            f"cutoff {certified_cutoff!r}",
            reasons=(FH_CUTOFF_MISMATCH,),
        )
    if str(certified.world_identity.cutoff or "").strip() != certified_cutoff:
        raise FreeHitAdapterError(
            f"{FH_CUTOFF_MISMATCH}: the predictive evidence was not produced at the certified cutoff",
            reasons=(FH_CUTOFF_MISMATCH,),
        )

    if conn is not None:
        # The pool is the accepted generation from the store, and the supplied
        # binding must BE it: identity, digest and exact ids.
        canonical_pool = resolve_pool_binding(conn)
        pool_problems = _pool_disagreements(canonical_pool, certified.pool_binding)
        if pool_problems:
            raise FreeHitAdapterError(
                f"{FH_POOL_MISMATCH}: the supplied pool is not the accepted generation: "
                + "; ".join(pool_problems),
                reasons=(FH_POOL_MISMATCH,),
            )
        canonical = free_hit_manager_state(
            conn, int(manager.entry_id), int(manager.planning_event),
            decision_cutoff=certified_cutoff, as_of=as_of,
        )
        disagreements = _state_disagreements(canonical, manager)
        if disagreements:
            raise FreeHitAdapterError(
                f"{FH_CALLER_STATE_DISAGREES}: the supplied manager state is not the canonical one: "
                + "; ".join(disagreements),
                reasons=(FH_CALLER_STATE_DISAGREES,),
            )
        if str(canonical.cutoff) != certified_cutoff:
            raise FreeHitAdapterError(
                f"{FH_CUTOFF_MISMATCH}: prices were not retrieved at the certified decision cutoff",
                reasons=(FH_CUTOFF_MISMATCH,),
            )

    binding = certified.horizon_binding
    if int(binding.planning_event) != int(manager.planning_event):
        raise FreeHitAdapterError(
            f"{fh.FH_HORIZON_NOT_CANONICAL}: the certified horizon is bound to planning event "
            f"{int(binding.planning_event)} but the manager state is for GW{int(manager.planning_event)}",
            reasons=(fh.FH_HORIZON_NOT_CANONICAL,),
        )

    # ── bind each arm's route worlds to the certified bundle, PER EVENT ───────
    # Validation and matrix extraction happen together, so the matrices handed to
    # exact_evaluate are the ones carried by the artifacts that were just proved
    # to be the certified event bundles.
    horizon = tuple(int(e) for e in binding.horizon_events)
    bundles = certified_event_bundles(authority)
    if world_cache_dir is None:
        raise FreeHitAdapterError(
            f"{fh.FH_DECISION_AUTHORITY_REQUIRED}: no world-cache directory is configured, so the "
            "certified numeric worlds cannot be loaded",
            reasons=(fh.FH_DECISION_AUTHORITY_REQUIRED,),
        )
    save_worlds = load_certified_route_worlds(
        conn, bundles, arm="SAVE", expected_events=horizon,
        union_ids=manager.eligible_ids, config=certified.save.route_config,
        cache_dir=world_cache_dir,
    )
    play_worlds = load_certified_route_worlds(
        conn, bundles, arm="PLAY", expected_events=horizon[1:],
        union_ids=manager.eligible_ids, config=certified.play.route_config,
        cache_dir=world_cache_dir,
    )

    # ── A2: derive the expected start states, then CONVERT both arms HERE ─────
    # Conversion happens in the adapter, on the canonical routes, so the only
    # routes that can enter the request are ones the accepted converter produced
    # and ``exact_evaluate`` valued.
    permanent = manager.permanent_state()
    probe = fh.FreeHitRequest(
        permanent=permanent, horizon_binding=binding, h1_worlds=certified.h1_worlds,
        world_identity=certified.world_identity, positions=dict(manager.positions),
        clubs=dict(manager.clubs), market_price_tenths=dict(manager.market_price_tenths),
        pool_binding=certified.pool_binding, chip_available=bool(chip_available),
        rules=rules if rules is not None else sr.SeasonRules(season="2026/27"),
    )
    play_expected = fh.play_h2_start_state(probe)
    save_expected = fh.save_h1_start_state(probe)
    try:
        save_route = fh.free_hit_route_from_canonical_route(
            certified.save.partial, arm=fh.ARM_SAVE,
            expected_events=horizon,
            rules=probe.rules, expected_start_state=save_expected,
            worlds_by_event=save_worlds,
            positions_of=certified.save.positions_of, route_config=certified.save.route_config,
        )
        play_route = fh.free_hit_route_from_canonical_route(
            certified.play.partial, arm=fh.ARM_PLAY,
            expected_events=horizon[1:],
            rules=probe.rules, expected_start_state=play_expected,
            worlds_by_event=play_worlds,
            positions_of=certified.play.positions_of, route_config=certified.play.route_config,
        )
    except FreeHitRouteError as exc:
        raise FreeHitAdapterError(
            f"{fh.FH_TAIL_ROUTE_INVALID}: the canonical route could not be converted: {exc}",
            reasons=(fh.FH_TAIL_ROUTE_INVALID,),
        ) from exc

    request = fh.FreeHitRequest(
        permanent=manager.permanent_state(),
        horizon_binding=binding,
        h1_worlds=certified.h1_worlds,
        world_identity=certified.world_identity,
        positions=dict(manager.positions),
        clubs=dict(manager.clubs),
        market_price_tenths=dict(manager.market_price_tenths),
        pool_binding=certified.pool_binding,
        play_route=play_route,
        save_route=save_route,
        # The LOADED authority: derived here from the canonical certification
        # artifact, never accepted from the caller.
        decision_authority=authority,
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
            fh.FH_DECISION_AUTHORITY_REQUIRED, fh.FH_DECISION_AUTHORITY_MISMATCH,
            fh.FH_TAIL_ROUTE_MISSING, fh.FH_TAIL_ROUTE_INVALID,
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


class _CertifiedRunIds:
    """The four upstream run ids a certified bundle commits to.

    ``route_optimizer.world_cache_key`` names them ``minutes/team/rate/xpts``; the
    certification schema stores the same runs under their model-family names.  The
    certified IDENTITY and the recorded model VERSIONS are carried too, because the
    canonical loader refuses any bundle that cannot declare which certified bundle
    its run ids came from and which version each family's run carries.
    """

    __slots__ = ("event", "minutes_run_id", "team_run_id", "rate_run_id", "xpts_run_id",
                 "mc_run_id", "planning_cutoff", "model_versions", "certified_bundle_identity",
                 "source_snapshot_sha256", "code_snapshot_sha256", "planning_context_hash")

    def __init__(self, event: int, row: Mapping[str, Any]) -> None:
        # Accepts a persisted bundle row (which carries the provenance) as well as a
        # bare family -> run-id mapping, which carries none: the loader refuses the
        # latter, and there is no default that would invent the missing provenance.
        if isinstance(row.get("runs"), Mapping):
            record: Mapping[str, Any] = row
            runs = dict(row["runs"])
        else:
            record = {}
            runs = dict(row)
        self.event = int(event)
        self.minutes_run_id = int(runs["minutes_v1"])
        self.team_run_id = int(runs["team_strength_v1"])
        self.rate_run_id = int(runs["player_rates_v1"])
        self.xpts_run_id = int(runs["xpts_v1"])
        self.mc_run_id = None if runs.get("monte_carlo_v1") is None else int(runs["monte_carlo_v1"])
        self.planning_cutoff = record.get("cutoff")
        self.code_snapshot_sha256 = record.get("code_snapshot_sha256")
        self.source_snapshot_sha256 = record.get("data_snapshot_sha256")
        self.planning_context_hash = record.get("planning_context_hash")
        self.model_versions = {
            str(family): str(version)
            for family, version in (record.get("model_versions") or {}).items()
        }
        self.certified_bundle_identity = str(record.get("certified_bundle_identity") or "") or (
            cb.certified_bundle_identity_for(
                event=int(event),
                cutoff=self.planning_cutoff,
                runs=self.certified_runs,
                model_versions=self.model_versions,
                code_snapshot_sha256=self.code_snapshot_sha256,
                data_snapshot_sha256=self.source_snapshot_sha256,
                planning_context_hash=self.planning_context_hash,
            )
        )

    @property
    def certified_runs(self) -> dict[str, int]:
        """The exact run ids under their model-family names.

        The Monte Carlo run is included when the bundle row declares one: the
        certified identity covers EVERY family the bundle names, so omitting a
        declared family here would compute a different identity from the one the
        certification minted.
        """

        runs = {
            "minutes_v1": int(self.minutes_run_id),
            "team_strength_v1": int(self.team_run_id),
            "player_rates_v1": int(self.rate_run_id),
            "xpts_v1": int(self.xpts_run_id),
        }
        if self.mc_run_id is not None:
            runs["monte_carlo_v1"] = int(self.mc_run_id)
        return runs

    def as_identity_payload(self) -> dict:
        return cb.bundle_identity_payload(
            event=int(self.event),
            cutoff=self.planning_cutoff,
            runs=self.certified_runs,
            model_versions=self.model_versions,
            code_snapshot_sha256=self.code_snapshot_sha256,
            data_snapshot_sha256=self.source_snapshot_sha256,
            planning_context_hash=self.planning_context_hash,
        )


def certified_event_bundles(authority: Any) -> dict[int, _CertifiedRunIds]:
    """The certified run ids per event, straight from the loaded artifact."""

    bundles: dict[int, _CertifiedRunIds] = {}
    for event, row in (authority.bundle_map or {}).items():
        runs = dict(row.get("runs") or {})
        missing = [k for k in ("minutes_v1", "team_strength_v1", "player_rates_v1", "xpts_v1")
                   if k not in runs]
        if missing:
            raise FreeHitAdapterError(
                f"{fh.FH_DECISION_AUTHORITY_REQUIRED}: the certified bundle for event {int(event)} "
                f"omits run(s) {missing}; the numeric matrix cannot be loaded",
                reasons=(fh.FH_DECISION_AUTHORITY_REQUIRED,),
            )
        bundles[int(event)] = _CertifiedRunIds(int(event), row)
    return bundles


def load_certified_route_worlds(
    conn: sqlite3.Connection,
    bundles: Mapping[int, _CertifiedRunIds],
    *,
    arm: str,
    expected_events: Sequence[int],
    union_ids: Sequence[int],
    config: Any,
    cache_dir: Any | None = None,
) -> dict[int, dict[str, Any]]:
    """Load ONE arm's route worlds from the CANONICAL matrix authority.

    ``route_optimizer.build_event_worlds`` is the accepted loader: it derives the
    world-cache key from the CERTIFIED run ids, the Monte Carlo model identity, the
    simulation count, the seed and the capture union, reads the matrix from the
    content-addressed cache under that key (or regenerates it from those certified
    runs), and STAMPS the result with that key.  The matrices returned here are the
    loader's own output -- there is no caller matrix anywhere on the path.
    """

    from . import route_optimizer as ro

    loaded: dict[int, dict[str, Any]] = {}
    union = tuple(int(p) for p in union_ids)
    for event in expected_events:
        event = int(event)
        bundle = bundles.get(event)
        if bundle is None:
            raise FreeHitAdapterError(
                f"{fh.FH_DECISION_AUTHORITY_REQUIRED}: the certification certifies no bundle for "
                f"{arm} route event {event}",
                reasons=(fh.FH_DECISION_AUTHORITY_REQUIRED,),
            )
        try:
            matrix, _info = ro.build_event_worlds(
                conn, bundles, event, union, config, cache_dir=cache_dir
            )
        except Exception as exc:  # the canonical loader's own refusals are authoritative
            raise FreeHitAdapterError(
                f"{fh.FH_DECISION_AUTHORITY_REQUIRED}: the canonical worlds for {arm} route event "
                f"{event} could not be loaded from the certified bundle: {exc}",
                reasons=(fh.FH_DECISION_AUTHORITY_REQUIRED,),
            ) from exc
        expected_key = ro.world_cache_key(event=event, bundle=bundle, config=config, union_ids=union)
        stamp = str(matrix.get(ro.MANAGER_MATRIX_IDENTITY_KEY) or "")
        if stamp != expected_key:
            raise FreeHitAdapterError(
                f"{fh.FH_DECISION_AUTHORITY_REQUIRED}: the {arm} world matrix for event {event} does "
                "not carry the canonical identity of its certified bundle",
                reasons=(fh.FH_DECISION_AUTHORITY_REQUIRED,),
            )
        if {int(p) for p in matrix["player_ids"]} != set(union):
            raise FreeHitAdapterError(
                f"{fh.FH_DECISION_AUTHORITY_REQUIRED}: the {arm} world matrix for event {event} does "
                "not cover the authoritative universe",
                reasons=(fh.FH_DECISION_AUTHORITY_REQUIRED,),
            )
        loaded[event] = {
            "worlds": int(matrix["worlds"]), "player_ids": tuple(matrix["player_ids"]),
            "minutes": matrix["minutes"], "core": matrix["core"],
        }
    return loaded


def _pool_disagreements(canonical: WildcardPoolBinding, supplied: WildcardPoolBinding) -> list[str]:
    """Every way a supplied pool differs from the accepted generation."""

    found: list[str] = []
    if str(supplied.generation_identity) != str(canonical.generation_identity):
        found.append("generation identity")
    if str(supplied.generation_id_sha256) != str(canonical.generation_id_sha256):
        found.append("generation id digest")
    if int(supplied.official_count) != int(canonical.official_count):
        found.append("official count")
    if tuple(sorted(int(p) for p in supplied.eligible_ids)) != tuple(
        sorted(int(p) for p in canonical.eligible_ids)
    ):
        found.append("eligible ids")
    return found


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
    # The two free-transfer concepts are INDEPENDENT and both are authority.  The
    # event-start bank is what a played chip preserves; the CURRENT bank is what a
    # normal route may spend at H1.  Comparing only one lets the other be forged,
    # and inferring either from the other is exactly the collapse to avoid.
    if int(supplied.event_start_free_transfers) != int(canonical.event_start_free_transfers):
        found.append(
            f"event-start free transfers {int(supplied.event_start_free_transfers)} != canonical "
            f"{int(canonical.event_start_free_transfers)}"
        )
    if int(supplied.free_transfers) != int(canonical.free_transfers):
        found.append(
            f"current free transfers {int(supplied.free_transfers)} != canonical "
            f"{int(canonical.free_transfers)}"
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
    # EXACT keyset equality: an extra key is a non-official player and a missing
    # key is a silently dropped one.  Neither is a matter of judgement.
    for label, canonical_map, supplied_map in (
        ("positions", canonical.positions, supplied.positions),
        ("clubs", canonical.clubs, supplied.clubs),
        ("market prices", canonical.market_price_tenths, supplied.market_price_tenths),
    ):
        extra = sorted(set(int(k) for k in supplied_map) - set(int(k) for k in canonical_map))
        missing = sorted(set(int(k) for k in canonical_map) - set(int(k) for k in supplied_map))
        if extra:
            found.append(f"{label} carry {len(extra)} non-official player(s): {extra[:6]}")
        if missing:
            found.append(f"{label} omit {len(missing)} official player(s): {missing[:6]}")
    return found


def resolve_pool_binding(conn: sqlite3.Connection) -> WildcardPoolBinding:
    """The official eligible pool, straight from the accepted-generation store."""

    return pool_binding_from_store(conn)
