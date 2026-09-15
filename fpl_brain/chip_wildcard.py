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
# The canonical token for an incomplete official player pool is owned by the
# candidate-universe contract; Wildcard reuses it rather than inventing a synonym.
from .candidate_universe import OFFICIAL_PLAYER_POOL_INCOMPLETE

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
WC_VALUE_BINDING_MISMATCH = "WILDCARD_VALUE_HORIZON_BINDING_MISMATCH"
WC_PROJECTION_IDENTITY_MISMATCH = "WILDCARD_PROJECTION_IDENTITY_MISMATCH"
WC_PROJECTION_INVALID = "WILDCARD_PROJECTION_INVALID"
WC_PROJECTION_DUPLICATE = "WILDCARD_PROJECTION_DUPLICATE"
WC_WORLD_INPUTS_MISSING = "WILDCARD_WORLD_INPUTS_MISSING"
WC_WORLD_INPUTS_MALFORMED = "WILDCARD_WORLD_INPUTS_MALFORMED"

WC_REASON_POSITIVE = "WILDCARD_UPLIFT_POSITIVE"
WC_REASON_NOT_COMPETITIVE = "WILDCARD_NOT_COMPETITIVE"
WC_REASON_PLAY_DESCRIPTION = "WILDCARD_PLAY_NOW_SQUAD_DESCRIPTION"
WC_REASON_REVIEW_ONLY = "WILDCARD_REVIEW_ONLY_UNCALIBRATED"

#: The SAVE arm consumes the accepted normal route.  These refuse when that
#: route is absent or does not expose the authoritative per-event state.
WC_SAVE_ROUTE_MISSING = "WILDCARD_SAVE_ROUTE_MISSING"
WC_SAVE_ROUTE_INVALID = "WILDCARD_SAVE_ROUTE_INVALID"


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


@dataclass(frozen=True)
class WildcardValueHorizonBinding:
    """The certified identity of the LONGER Wildcard valuation horizon.

    Distinct from ``chip_decision.ChipHorizonBinding``, which stays exactly four
    events for the chip-DECISION contract.  This one binds the 6-10 event value
    horizon, and every projection row the evaluator consumes must belong to it.
    A request-level string is NOT sufficient: arbitrary rows could otherwise be
    inserted under a matching label, so the rows carry their own identity and are
    checked individually (§2).
    """

    planning_event: int
    event_ids: tuple[int, ...]
    decision_cutoff: str
    data_snapshot_sha256: str
    source_snapshot_sha256: str
    prediction_generation: str
    model_config_identity: str
    horizon_version: str

    def __post_init__(self) -> None:
        problems = self.problems()
        if problems:
            raise WildcardInputError(
                f"{WC_VALUE_BINDING_MISMATCH}: {'; '.join(problems)}",
                reasons=(WC_VALUE_BINDING_MISMATCH,),
            )

    def problems(self) -> list[str]:
        found: list[str] = []
        events = tuple(int(e) for e in self.event_ids)
        if not (WILDCARD_HORIZON_MIN_EVENTS <= len(events) <= WILDCARD_HORIZON_MAX_EVENTS):
            found.append(f"horizon length {len(events)} outside "
                         f"{WILDCARD_HORIZON_MIN_EVENTS}-{WILDCARD_HORIZON_MAX_EVENTS}")
        if len(set(events)) != len(events):
            found.append("duplicate events")
        if events and events[0] != int(self.planning_event):
            found.append(f"first event {events[0]} != planning event {self.planning_event}")
        if events and events != tuple(range(events[0], events[0] + len(events))):
            found.append(f"events {list(events)} are not contiguous")
        for name in ("decision_cutoff", "data_snapshot_sha256", "source_snapshot_sha256",
                     "prediction_generation", "model_config_identity", "horizon_version"):
            if not str(getattr(self, name) or "").strip():
                found.append(f"{name} is empty")
        return found

    def as_dict(self) -> dict[str, Any]:
        return {
            "planning_event": int(self.planning_event),
            "event_ids": [int(e) for e in self.event_ids],
            "decision_cutoff": str(self.decision_cutoff),
            "data_snapshot_sha256": str(self.data_snapshot_sha256),
            "source_snapshot_sha256": str(self.source_snapshot_sha256),
            "prediction_generation": str(self.prediction_generation),
            "model_config_identity": str(self.model_config_identity),
            "horizon_version": str(self.horizon_version),
        }

    def identity(self) -> str:
        from .analytics import canonical_hash

        return canonical_hash(self.as_dict())

    def disagreements_with(self, chip_binding: Any) -> list[str]:
        """Where this value binding fails to agree with the chip DECISION binding.

        The two are different contracts and neither is widened, but they must
        describe the same planning event, cutoff and data snapshot, or the play
        and save arms would be valued against different worlds.
        """

        found: list[str] = []
        if int(getattr(chip_binding, "planning_event", -1)) != int(self.planning_event):
            found.append("planning event differs from the chip decision binding")
        chip_snapshot = getattr(chip_binding, "data_snapshot_sha256", None)
        if chip_snapshot and str(chip_snapshot) != str(self.data_snapshot_sha256):
            found.append("data snapshot differs from the chip decision binding")
        chip_cutoff = getattr(chip_binding, "decision_cutoff", None)
        if chip_cutoff is not None and str(chip_cutoff) != str(self.decision_cutoff):
            found.append("decision cutoff differs from the chip decision binding")
        chip_events = tuple(int(e) for e in getattr(chip_binding, "horizon_events", ()) or ())
        if chip_events and not set(chip_events).issubset(set(self.event_ids)):
            found.append("the chip decision events are not inside the value horizon")
        return found


def value_horizon_binding(
    planning_event: int,
    horizon: "WildcardHorizonSpec",
    *,
    decision_cutoff: str,
    data_snapshot_sha256: str,
    source_snapshot_sha256: str,
    prediction_generation: str,
    model_config_identity: str,
) -> WildcardValueHorizonBinding:
    return WildcardValueHorizonBinding(
        planning_event=int(planning_event),
        event_ids=tuple(int(e) for e in horizon.events),
        decision_cutoff=str(decision_cutoff),
        data_snapshot_sha256=str(data_snapshot_sha256),
        source_snapshot_sha256=str(source_snapshot_sha256),
        prediction_generation=str(prediction_generation),
        model_config_identity=str(model_config_identity),
        horizon_version=str(horizon.version),
    )


@dataclass(frozen=True)
class WildcardPoolBinding:
    """The official eligible-player pool this Wildcard evaluation is exhaustive over.

    The canonical authority is the ACCEPTED official bootstrap generation
    (``ingest_provenance``): this wrapper only carries its identity alongside the
    eligible id set, and validates by IDENTITY — the generation's
    ``element_id_sha256`` must equal the digest of the eligible ids.  A count
    match with a different id set is a failure, because counts alone cannot prove
    identity.  Without this a caller could hand over a short list and the
    "exhaustive discovery" claim would be unfalsifiable.
    """

    generation_identity: str
    generation_id_sha256: str
    official_count: int
    eligible_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        problems = self.problems()
        if problems:
            raise WildcardInputError(
                f"{OFFICIAL_PLAYER_POOL_INCOMPLETE}: {'; '.join(problems[:6])}",
                reasons=(OFFICIAL_PLAYER_POOL_INCOMPLETE,),
            )

    def problems(self) -> list[str]:
        from .ingest_provenance import element_id_sha256

        found: list[str] = []
        if not str(self.generation_identity or "").strip():
            found.append("no official pool generation identity")
        ids = tuple(int(p) for p in self.eligible_ids)
        if len(set(ids)) != len(ids):
            found.append("duplicate ids in the eligible pool")
        if int(self.official_count) != len(ids):
            found.append(f"official_count {int(self.official_count)} != {len(ids)} eligible ids")
        if not str(self.generation_id_sha256 or "").strip():
            found.append("no official generation id digest")
        elif ids and str(self.generation_id_sha256) != element_id_sha256(sorted(ids)):
            found.append("the eligible ids do not match the official generation identity digest")
        return found

    def as_dict(self) -> dict[str, Any]:
        return {
            "generation_identity": str(self.generation_identity),
            "generation_id_sha256": str(self.generation_id_sha256),
            "official_count": int(self.official_count),
            "eligible_count": len(self.eligible_ids),
        }


def pool_binding_from_generation(generation: Mapping[str, Any]) -> WildcardPoolBinding:
    """Build the binding from an ACCEPTED official bootstrap generation row.

    Requires an EXPLICIT accepted flag.  A missing or falsy ``accepted`` is NOT
    accepted: without a recorded acceptance there is nothing to certify the pool
    against, and treating absence as acceptance is precisely how a caller would
    self-certify a universe.
    """

    if not generation:
        raise WildcardInputError(
            f"{OFFICIAL_PLAYER_POOL_INCOMPLETE}: no accepted official generation supplied",
            reasons=(OFFICIAL_PLAYER_POOL_INCOMPLETE,),
        )
    if generation.get("accepted") is None:
        raise WildcardInputError(
            f"{OFFICIAL_PLAYER_POOL_INCOMPLETE}: the official generation carries no explicit "
            "accepted status",
            reasons=(OFFICIAL_PLAYER_POOL_INCOMPLETE,),
        )
    if not generation.get("accepted"):
        raise WildcardInputError(
            f"{OFFICIAL_PLAYER_POOL_INCOMPLETE}: the supplied official generation is not accepted "
            f"({generation.get('rejection_reasons') or 'rejected'})",
            reasons=(OFFICIAL_PLAYER_POOL_INCOMPLETE,),
        )
    return WildcardPoolBinding(
        generation_identity=str(generation.get("captured_at") or generation.get("id") or ""),
        # The in-memory record uses ``element_id_sha256``; the persisted
        # ``bootstrap_generations`` column is ``element_ids_sha256``.  Accept
        # both, or a store row would look like it had no digest at all.
        generation_id_sha256=str(
            generation.get("element_id_sha256") or generation.get("element_ids_sha256") or ""
        ),
        official_count=int(generation.get("official_element_count") or len(generation.get("element_ids") or ())),
        eligible_ids=tuple(sorted(int(p) for p in (generation.get("eligible_ids") or generation.get("element_ids") or ()))),
    )


def pool_binding_from_store(conn: Any) -> WildcardPoolBinding:
    """Resolve the binding from the CANONICAL accepted-generation store.

    The production authority is ``repositories.latest_accepted_bootstrap_generation``
    -- the newest row recorded with ``accepted=1``.  A caller cannot choose these
    ids, the count, the digest or the acceptance flag: they come from the store.
    """

    from . import repositories as repo

    generation = repo.latest_accepted_bootstrap_generation(conn)
    if not generation:
        raise WildcardInputError(
            f"{OFFICIAL_PLAYER_POOL_INCOMPLETE}: no accepted official generation is recorded",
            reasons=(OFFICIAL_PLAYER_POOL_INCOMPLETE,),
        )
    return pool_binding_from_generation(dict(generation))


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
    """One player's projection for one event, from certified inputs only.

    ``cutoff`` and ``generation`` are this ROW's own provenance.  They are
    checked against the value-horizon binding individually, because a binding
    that only covers the request cannot stop an arbitrary row being inserted
    under a matching label.
    """

    event: int
    expected_points: float
    expected_minutes: float
    p_start: float
    availability: float
    fixture_count: int = 1
    cutoff: str = ""
    generation: str = ""

    def problems(self, binding: "WildcardValueHorizonBinding | None" = None) -> list[str]:
        found: list[str] = []
        for name in ("expected_points", "expected_minutes", "p_start", "availability"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                found.append(f"{name}={value!r} is not finite")
        for name in ("expected_points", "expected_minutes"):
            if float(getattr(self, name)) < 0.0:
                found.append(f"{name} is negative")
        for name in ("p_start", "availability"):
            if not 0.0 <= float(getattr(self, name)) <= 1.0:
                found.append(f"{name}={getattr(self, name)!r} is not a probability")
        if int(self.fixture_count) < 0:
            found.append("fixture_count is negative")
        if binding is not None:
            if str(self.cutoff) != str(binding.decision_cutoff):
                found.append(f"cutoff {self.cutoff!r} != bound {binding.decision_cutoff!r}")
            if str(self.generation) != str(binding.prediction_generation):
                found.append(f"generation {self.generation!r} != bound {binding.prediction_generation!r}")
        return found


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
class WildcardWorldInputs:
    """Per-event, per-world minutes and core points — the ACCEPTED world shape.

    Mirrors the chip core's ``ChipWorldInputs`` deliberately: a plain matrix, not
    a model object, so a replacement projection/world generator feeds the
    Wildcard valuation without any Wildcard change.  The accepted lineup engine
    consumes these worlds directly, so appearance, captain fallback, bench order,
    formation-legal autosubs and the goalkeeper-only goalkeeper substitution are
    all resolved by ``manager_lineup`` rather than approximated here.
    """

    worlds: int
    player_ids: tuple[int, ...]
    minutes: Mapping[int, Sequence[float]]
    core: Mapping[int, Sequence[float]]

    def __post_init__(self) -> None:
        problems = self.problems()
        if problems:
            raise WildcardInputError(
                f"{WC_WORLD_INPUTS_MALFORMED}: {'; '.join(problems[:6])}",
                reasons=(WC_WORLD_INPUTS_MALFORMED,),
            )

    def problems(self) -> list[str]:
        found: list[str] = []
        if int(self.worlds) <= 0:
            found.append(f"worlds={self.worlds} must be positive")
            return found
        for player_id in self.player_ids:
            pid = int(player_id)
            minutes = self.minutes.get(pid)
            core = self.core.get(pid)
            if minutes is None or core is None:
                found.append(f"player {pid} has no minutes/core series")
                continue
            if len(minutes) != int(self.worlds) or len(core) != int(self.worlds):
                found.append(f"player {pid} series length != {int(self.worlds)}")
                continue
            for index, value in enumerate(minutes):
                if not math.isfinite(float(value)) or float(value) < 0.0:
                    found.append(f"player {pid} world {index} minutes={value!r} is invalid")
                    break
            for index, value in enumerate(core):
                if not math.isfinite(float(value)):
                    found.append(f"player {pid} world {index} core={value!r} is not finite")
                    break
        return found

    def as_dict(self) -> dict[str, Any]:
        return {"worlds": int(self.worlds), "players": len(self.player_ids)}


@dataclass(frozen=True)
class WildcardSaveRouteEvent:
    """One event of the accepted normal four-GW route, as the SAVE arm sees it.

    ``mean_net_core`` is the route engine's OWN per-event value and ALREADY nets
    that event's hit deduction (``route_optimizer.exact_evaluate`` returns
    ``mean_gross_core`` and ``mean_net_core`` side by side, with
    ``net_core = gross_core - cumulative_hits``).  ``hit_points`` is carried for
    EVIDENCE ONLY and must never be subtracted from ``mean_net_core`` -- doing so
    would double-count, turning a single -4 hit into an 8-point swing.  That is
    the one authority this contract exists to protect.
    """

    event: int
    squad_ids: tuple[int, ...]
    bank_tenths: int
    purchase_price_tenths: Mapping[int, int]
    free_transfers: int
    mean_net_core: float
    hit_points: int = 0
    actions: tuple = ()

    def problems(self) -> list[str]:
        found: list[str] = []
        if not math.isfinite(float(self.mean_net_core)):
            found.append(f"event {self.event}: mean_net_core is not finite")
        if not self.squad_ids:
            found.append(f"event {self.event}: no resulting squad")
        if int(self.bank_tenths) < 0:
            found.append(f"event {self.event}: negative bank")
        if int(self.free_transfers) < 0:
            found.append(f"event {self.event}: negative free transfers")
        missing = [pid for pid in self.squad_ids if int(pid) not in self.purchase_price_tenths]
        if missing:
            found.append(f"event {self.event}: {len(missing)} squad players have no acquisition basis")
        return found

    def as_dict(self) -> dict[str, Any]:
        return {
            "event": int(self.event),
            "squad_ids": [int(p) for p in self.squad_ids],
            "bank_tenths": int(self.bank_tenths),
            "free_transfers": int(self.free_transfers),
            "mean_net_core": round(float(self.mean_net_core), 6),
            "hit_points": int(self.hit_points),
        }


@dataclass(frozen=True)
class WildcardSaveRoute:
    """The accepted normal route, terminal state included.

    The SAVE_WC policy is: do not wildcard, play this route, and KEEP the chip.
    The terminal state is the route's final permanent state, carried unchanged
    through the remaining Wildcard horizon -- no H5+ transfers are invented.
    """

    events: tuple[WildcardSaveRouteEvent, ...]
    terminal_squad_ids: tuple[int, ...]
    terminal_bank_tenths: int
    terminal_purchase_price_tenths: Mapping[int, int]
    terminal_free_transfers: int
    cumulative_hits: int = 0
    wildcard_available: bool = True

    def problems(self) -> list[str]:
        found: list[str] = []
        if len(self.events) != 4:
            found.append(f"a normal route must have exactly 4 events, got {len(self.events)}")
        for entry in self.events:
            found.extend(entry.problems())
        if not self.terminal_squad_ids:
            found.append("no terminal squad")
        if int(self.terminal_bank_tenths) < 0:
            found.append("negative terminal bank")
        missing = [pid for pid in self.terminal_squad_ids if int(pid) not in self.terminal_purchase_price_tenths]
        if missing:
            found.append(f"{len(missing)} terminal squad players have no acquisition basis")
        if int(self.terminal_free_transfers) < 0:
            found.append("negative terminal free transfers")
        if not self.wildcard_available:
            found.append("SAVE_WC must leave the Wildcard available")
        return found

    def as_dict(self) -> dict[str, Any]:
        return {
            "events": [e.as_dict() for e in self.events],
            "terminal_squad_ids": [int(p) for p in self.terminal_squad_ids],
            "terminal_bank_tenths": int(self.terminal_bank_tenths),
            "terminal_free_transfers": int(self.terminal_free_transfers),
            "cumulative_hits": int(self.cumulative_hits),
            "wildcard_available": bool(self.wildcard_available),
        }

    def post_save_state(self) -> dict[str, Any]:
        """The authoritative post-SAVE state handed to the arbiter's reservation."""

        return {
            "squad_ids": [int(p) for p in self.terminal_squad_ids],
            "bank_tenths": int(self.terminal_bank_tenths),
            "purchase_price_tenths": {int(k): int(v) for k, v in sorted(self.terminal_purchase_price_tenths.items())},
            "free_transfers": int(self.terminal_free_transfers),
            "retains_wildcard_option": bool(self.wildcard_available),
            "cumulative_hits": int(self.cumulative_hits),
        }


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
    #: The ACTUAL accepted normal four-GW route.  This is the ONLY authoritative
    #: SAVE input: the scalar ``save_route_value`` / ``save_route_expected_hits``
    #: fields it replaces could be invented by a caller, which is what the SAVE
    #: wiring exists to prevent.
    save_route: WildcardSaveRoute | None = None
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
    #: The longer VALUE horizon, bound separately.  Optional only so that a
    #: caller who supplies nothing cannot accidentally look certified: the
    #: evaluator refuses when it is absent.
    value_horizon_binding: WildcardValueHorizonBinding | None = None
    #: Per-event world matrices the ACCEPTED lineup engine resolves over.  An
    #: event with no world inputs is a hard failure: the event value is never
    #: approximated from start probabilities.
    worlds_by_event: Mapping[int, WildcardWorldInputs] = field(default_factory=dict)
    #: The accepted official eligible-player pool.  Absent means the discovery
    #: claim cannot be checked, so the evaluator refuses rather than assuming
    #: the supplied universe is exhaustive.
    pool_binding: WildcardPoolBinding | None = None


# ---------------------------------------------------------------------------
# Stage 0/1 — exhaustive screening and position frontiers
# ---------------------------------------------------------------------------


def _available(player: WildcardPlayer, horizon: WildcardHorizonSpec) -> bool:
    """A player is screenable only if EVERY horizon event has a projection."""

    return all(player.at(event) is not None for event in horizon.events)


def pool_accounting(request: WildcardRequest) -> dict[str, Any]:
    """Account for EVERY eligible official player as SUPPORTED or EXCLUDED.

    The invariant is ``supported_count + excluded_count == eligible_count`` and
    ``screened_ids == supported_ids``.  A player who is neither supported nor
    excluded is reported in ``unaccounted_ids``, which is a hard failure: no
    eligible player may simply disappear from the universe.  The audit is
    machine-readable and never truncated.
    """

    binding = request.pool_binding
    if binding is None:
        return {
            "official_count": None, "eligible_count": None, "supported_count": None,
            "excluded_count": None, "screened_count": None,
            "generation_identity": None, "generation_id_sha256": None,
            "supported_ids": [], "excluded_ids": [], "excluded_reasons": {},
            "unaccounted_ids": [], "complete": False,
            "reason": "no official pool binding supplied",
        }

    horizon = request.horizon
    eligible = tuple(int(p) for p in binding.eligible_ids)
    supported: list[int] = []
    excluded: list[int] = []
    absent: list[int] = []
    for pid in eligible:
        player = request.players.get(pid)
        if player is None:
            # ABSENT from the universe entirely is NOT an exclusion with a
            # reason -- it is the silent disappearance this contract exists to
            # catch, and it must fail rather than be filed as "excluded".
            absent.append(pid)
        elif _available(player, horizon):
            supported.append(pid)
        else:
            # PRESENT but without complete horizon support: a legitimate,
            # auditable exclusion.
            excluded.append(pid)

    unaccounted = sorted(absent)
    # A player present in the SUPPLIED universe but absent from the canonical
    # eligible pool is contradictory evidence: the caller's universe is not the
    # certified one.  It is refused rather than silently ignored, because
    # ignoring it is exactly how a nonofficial player reaches the optimizer.
    supplied = {int(pid) for pid in request.players}
    extra = sorted(supplied - set(eligible))
    complete = (
        not unaccounted
        and not extra
        and len(supported) + len(excluded) == len(eligible)
    )
    return {
        "official_count": len(eligible),
        "eligible_count": len(eligible),
        "supported_count": len(supported),
        "excluded_count": len(excluded),
        "screened_count": len(supported),
        "generation_identity": str(binding.generation_identity),
        "generation_id_sha256": str(binding.generation_id_sha256),
        "supported_ids": sorted(supported),
        "excluded_ids": sorted(excluded),
        "excluded_reasons": {pid: WC_MISSING_PROJECTION for pid in sorted(excluded)},
        "excluded_sample": sorted(excluded)[:20],
        "unaccounted_ids": unaccounted,
        "extra_ids": extra,
        "extra_sample": extra[:20],
        "complete": complete,
        "reason": (
            None if complete
            else (f"{len(extra)} supplied players are not in the canonical eligible pool "
                  f"(e.g. {extra[:5]})" if extra else f"{len(unaccounted)} eligible players are unaccounted for")
        ),
    }


def validate_projections(request: WildcardRequest) -> list[str]:
    """Every projection row must belong to the bound value-horizon generation.

    Checked for EVERY row the evaluator can consume — screening, squad
    evaluation, captaincy and horizon valuation all read the same rows — so a
    current/latest row cannot be injected into a historical replay.
    """

    binding = request.value_horizon_binding
    problems: list[str] = []
    if binding is None:
        return [f"{WC_VALUE_BINDING_MISMATCH}: no value-horizon binding supplied"]
    problems.extend(binding.problems())
    # Every binding-level problem carries the binding token as its prefix, so
    # the caller always reports the correct refusal token rather than a
    # projection one.
    for detail in binding.disagreements_with(request.horizon_binding):
        problems.append(f"{WC_VALUE_BINDING_MISMATCH}: {detail}")
    if tuple(int(e) for e in binding.event_ids) != tuple(int(e) for e in request.horizon.events):
        problems.append(
            f"{WC_VALUE_BINDING_MISMATCH}: the horizon spec and the value binding "
            "describe different events"
        )
    if str(binding.horizon_version) != str(request.horizon.version):
        problems.append(
            f"{WC_VALUE_BINDING_MISMATCH}: the horizon spec and the value binding "
            "disagree on the horizon version"
        )
    # Every horizon event must carry world inputs, and those worlds must cover
    # EVERY supplied player: a partially covered world matrix would let the
    # accepted resolver treat a missing player as an implicit non-appearance.
    if not request.worlds_by_event:
        problems.append(f"{WC_WORLD_INPUTS_MISSING}: no world inputs supplied for any horizon event")
    else:
        supplied = {int(pid) for pid in request.players}
        for event in request.horizon.events:
            world_inputs = request.worlds_by_event.get(int(event))
            if world_inputs is None:
                problems.append(f"{WC_WORLD_INPUTS_MISSING}: no world inputs for event {event}")
                continue
            covered = {int(pid) for pid in world_inputs.player_ids}
            uncovered = sorted(supplied - covered)
            if uncovered:
                problems.append(
                    f"event {event}: {len(uncovered)} supplied players have no world series "
                    f"(e.g. {uncovered[:5]})"
                )
    for player_id, player in sorted(request.players.items()):
        if int(player_id) != int(player.player_id):
            problems.append(f"player key {player_id} disagrees with the row id {player.player_id}")
        for event, entry in sorted(player.events.items()):
            if int(event) != int(entry.event):
                problems.append(f"player {player_id}: key {event} disagrees with row event {entry.event}")
            for detail in entry.problems(binding):
                problems.append(f"player {player_id} event {event}: {detail}")
    return problems


def screen_players(request: WildcardRequest) -> tuple[dict[int, float], dict[str, Any]]:
    """Exhaustive numeric screen.  NO watchlist, ownership or club gate.

    Every eligible official player with valid predictive support is scored.  A
    player with a missing projection is EXCLUDED WITH AN AUDITABLE REASON, never
    silently treated as zero points.
    """

    horizon = request.horizon
    # The screening universe is the CANONICAL eligible pool, never an arbitrary
    # request.players.  Without this a caller could add a player outside the
    # certified pool (the "999" counterexample) and have him screened and
    # optimised while the pool accounting still looked complete.
    canonical = (
        tuple(int(p) for p in request.pool_binding.eligible_ids)
        if request.pool_binding is not None
        else tuple(sorted(int(p) for p in request.players))
    )
    scores: dict[int, float] = {}
    missing: list[int] = []
    for player_id in canonical:
        player = request.players.get(int(player_id))
        if player is None or not _available(player, horizon):
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
    reasons = {int(pid): WC_MISSING_PROJECTION for pid in missing}
    return scores, {
        "screened": len(scores),
        "excluded_missing_projection": len(missing),
        # FULL machine-readable exclusion evidence — never truncated, so the
        # audit can account for every official player.
        "excluded_ids": sorted(reasons),
        "excluded_reasons": {int(k): v for k, v in sorted(reasons.items())},
        # a short human-readable sample is ALSO provided, but it never replaces
        # the full evidence above
        "excluded_sample": sorted(reasons)[:20],
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


def _mean_core(request: WildcardRequest, player_id: int, event: int) -> float:
    """Mean core points for one player at one event, from the bound worlds."""

    worlds = request.worlds_by_event.get(int(event))
    if worlds is None:
        return 0.0
    series = worlds.core.get(int(player_id)) or ()
    return (sum(float(v) for v in series) / len(series)) if series else 0.0


def _event_policy(
    request: WildcardRequest, squad: Sequence[int], event: int
) -> manager_lineup.ManagerPolicy | None:
    """The fifteen the manager would field: best legal XI, armband on its best.

    Choosing WHICH eleven to start, and who wears the armband, is the management
    decision this evaluator makes.  The VALUE of that decision is then resolved
    by the accepted engine over worlds — the armband is never approximated with a
    start-probability shortcut, and the bench is never scored directly.
    """

    xi, _ = _best_xi(request, squad, event)
    if len(xi) != manager_lineup.XI_SIZE:
        return None
    bench = [pid for pid in squad if pid not in set(xi)]
    bench_gk = [pid for pid in bench if request.players[pid].position == "GKP"]
    bench_out = [pid for pid in bench if request.players[pid].position != "GKP"]
    if len(bench_gk) != 1 or len(bench_out) != manager_lineup.OUTFIELD_BENCH_SIZE:
        return None
    ranked = sorted(xi, key=lambda pid: (-_mean_core(request, pid, event), pid))
    return manager_lineup.ManagerPolicy(
        starter_ids=tuple(sorted(xi)),
        bench_gk_id=int(bench_gk[0]),
        bench_outfield_order=tuple(sorted(bench_out, key=lambda pid: (-_mean_core(request, pid, event), pid))),
        captain_id=int(ranked[0]),
        vice_captain_id=int(ranked[1]),
    )


def _event_value(request: WildcardRequest, squad: Sequence[int], event: int) -> float:
    """Expected event points for a squad, resolved by the ACCEPTED engine.

    For every supplied world the legal lineup, captain fallback and outfield /
    goalkeeper autosubs are resolved by ``manager_lineup.resolve_world`` and the
    armband by ``manager_lineup.captain_multiplier`` — the same semantics the
    accepted manager-world engine uses.  The event value is the mean world total.
    Bench Boost is never active: only the resolved counted XI scores, so the four
    bench slots contribute solely through legal autosubs.
    """

    worlds = request.worlds_by_event.get(int(event))
    if worlds is None:
        raise WildcardInputError(
            f"{WC_WORLD_INPUTS_MISSING}: no world inputs for event {event}",
            reasons=(WC_WORLD_INPUTS_MISSING,),
        )
    policy = _event_policy(request, squad, event)
    if policy is None:
        # A squad that cannot field a legal XI/bench has no event value; it is
        # reported as zero rather than being silently scored on a partial lineup.
        return 0.0
    positions = {int(pid): request.players[int(pid)].position for pid in squad}
    total = 0.0
    for world in range(worlds.worlds):
        minutes = {int(pid): float(worlds.minutes[int(pid)][world]) for pid in squad}
        core = {int(pid): float(worlds.core[int(pid)][world]) for pid in squad}
        outcome = manager_lineup.resolve_world(
            policy, positions, minutes, core, require_player_ids=list(squad)
        )
        extra, _ = manager_lineup.captain_multiplier(policy, minutes, core)
        total += sum(core[pid] for pid in outcome.counted_ids) + extra
    return total / worlds.worlds


def save_route_value(request: WildcardRequest, route: WildcardSaveRoute) -> tuple[float, dict[str, Any]]:
    """SAVE_WC value from the ACCEPTED normal route plus the carried terminal state.

    H1-H4 use the route engine's OWN per-event ``mean_net_core``, which already
    nets that route's transfer hits, so NO hit term is subtracted here.  H5+ value
    the route's TERMINAL permanent state (squad/basis/bank/FT) carried forward
    unchanged -- no further transfers are invented and no future information is
    used, because V1 does not construct a route beyond H4.
    """

    horizon = request.horizon
    first_four = tuple(int(e) for e in horizon.events[:4])
    route_events = {int(entry.event): entry for entry in route.events}

    total = 0.0
    per_event: dict[int, dict[str, Any]] = {}
    for event in first_four:
        entry = route_events.get(int(event))
        if entry is None:
            raise WildcardInputError(
                f"{WC_SAVE_ROUTE_INVALID}: the route has no state for horizon event {event}",
                reasons=(WC_SAVE_ROUTE_INVALID,),
            )
        # mean_net_core ALREADY includes this event's hit.  Subtracting
        # entry.hit_points here would be the double-count the contract forbids.
        weighted = horizon.weight_for(event) * float(entry.mean_net_core)
        total += weighted
        per_event[int(event)] = {**entry.as_dict(), "weighted": weighted, "source": "ROUTE_MEAN_NET_CORE"}

    # H5+ : the route's terminal permanent state, valued with the accepted
    # Wildcard event/world valuation.  No transfers, no new hits.
    carried = tuple(int(p) for p in route.terminal_squad_ids)
    for event in [int(e) for e in horizon.events][4:]:
        value = _event_value(request, carried, event)
        weighted = horizon.weight_for(event) * value
        total += weighted
        per_event[int(event)] = {"event": int(event), "weighted": weighted,
                                 "source": "CARRIED_TERMINAL_STATE", "mean_net_core": value}

    evidence = {
        "per_event": per_event,
        "terminal_bank_tenths": int(route.terminal_bank_tenths),
        "terminal_free_transfers": int(route.terminal_free_transfers),
        "cumulative_hits": int(route.cumulative_hits),
        "hit_authority": "ROUTE_MEAN_NET_CORE_ALREADY_NETS_HITS",
        "manual_hit_subtraction": False,
    }
    return total, evidence


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
        value = _event_value(request, squad, event)
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
            "wildcard_value_horizon": (request.value_horizon_binding.as_dict()
                                        if request.value_horizon_binding else None),
            "official_player_pool": pool_accounting(request),
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

    expiry_event = resolve_wildcard_expiry(request)
    if expiry_event is EXPIRY_UNRESOLVED:
        return _refuse(request, WC_WINDOW_UNRESOLVED,
                       "the active Wildcard window could not be resolved from the canonical chip state",
                       reasons=(WC_WINDOW_UNRESOLVED,))
    if expiry_event is EXPIRY_AMBIGUOUS:
        return _refuse(request, WC_WINDOW_AMBIGUOUS,
                       "more than one Wildcard window is active for this event; refusing rather than guessing",
                       reasons=(WC_WINDOW_AMBIGUOUS,))

    accounting = pool_accounting(request)
    if not accounting["complete"]:
        return _refuse(request, OFFICIAL_PLAYER_POOL_INCOMPLETE,
                       str(accounting["reason"] or "the official player pool is not fully accounted for"),
                       reasons=(OFFICIAL_PLAYER_POOL_INCOMPLETE,))

    projection_problems = validate_projections(request)
    if projection_problems:
        # The first problem carries its own token prefix; never guess it from a
        # substring, or a world-inputs failure would be reported as a binding one.
        first = str(projection_problems[0])
        token = WC_PROJECTION_INVALID
        for candidate in (WC_VALUE_BINDING_MISMATCH, WC_WORLD_INPUTS_MISSING,
                          WC_WORLD_INPUTS_MALFORMED, WC_PROJECTION_INVALID):
            if first.startswith(candidate):
                token = candidate
                break
        return _refuse(request, token, "; ".join(projection_problems[:8]),
                       reasons=(token,))

    incoherent = _incoherent_reasons(request)
    if incoherent:
        return _refuse(request, incoherent[0], f"manager state/pricing not coherent: {incoherent}", reasons=incoherent)

    if request.certification_identity != getattr(request.horizon_binding, "certification_identity", None):
        return _refuse(request, WC_STALE_OR_CROSS_CUTOFF,
                       "the request certification identity does not match the horizon binding")

    # Resolve the ACTIVE Wildcard window first: the chip window is a
    # precondition for evaluating the chip at all, so an unresolved or ambiguous
    # window must fail fast rather than after a full screen.
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
    try:
        ranked = sorted((evaluate_squad(request, c.squad) for c in candidates),
                        key=lambda v: (-v.objective, v.squad))
    except WildcardInputError as exc:
        token = (exc.reasons[0] if exc.reasons else WC_WORLD_INPUTS_MISSING)
        return _refuse(request, token, str(exc), reasons=tuple(exc.reasons))
    play = ranked[0]

    # --- SAVE arm: do not wildcard, play the ACCEPTED normal route, KEEP the chip.
    route = request.save_route
    if route is None:
        return _refuse(request, WC_SAVE_ROUTE_MISSING,
                       "SAVE requires the accepted normal four-GW route; scalar SAVE "
                       "inputs are not authoritative",
                       reasons=(WC_SAVE_ROUTE_MISSING,))
    route_problems = route.problems()
    if route_problems:
        return _refuse(request, WC_SAVE_ROUTE_INVALID,
                       "; ".join(route_problems[:6]), reasons=(WC_SAVE_ROUTE_INVALID,))
    if tuple(int(e.event) for e in route.events) != tuple(int(e) for e in request.horizon.events[:4]):
        return _refuse(request, WC_SAVE_ROUTE_INVALID,
                       "the route's events are not the first four horizon events",
                       reasons=(WC_SAVE_ROUTE_INVALID,))
    try:
        save_total, save_evidence = save_route_value(request, route)
    except WildcardInputError as exc:
        return _refuse(request, WC_SAVE_ROUTE_INVALID, str(exc), reasons=tuple(exc.reasons))

    # The reservation is applied EXACTLY ONCE, by the chip arbiter — the same
    # seam every other chip uses.  This evaluator therefore reports the RAW
    # play-vs-save difference plus the post-SAVE state the arbiter's reservation
    # provider needs, and never calls the reservation itself.
    uplift = float(play.objective) - float(save_total)
    carried = evaluate_squad(request, route.terminal_squad_ids)
    save_state = {
        **route.post_save_state(),
        "wildcard_expiry_event": expiry_event,
        "minutes_security": carried.minutes_security,
        "expected_forced_moves": carried.repairability["expected_forced_moves"],
        "planning_event": int(request.planning_event),
        "certification_identity": request.certification_identity,
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
            "wildcard_value_horizon": request.value_horizon_binding.as_dict(),
            "wildcard_value_horizon_identity": request.value_horizon_binding.identity(),
            "official_player_pool": pool_accounting(request),
            "wildcard_evaluator_version": WILDCARD_EVALUATOR_VERSION,
            "wildcard_quantitative_capability": "SUPPORTED_REVIEW_ONLY",
            "no_global_optimum_claim": True,
            "wildcard_expiry_event": expiry_event,
            "free_transfers_after_wildcard": free_transfers_after,
            "ft_preserved_by_chip": sr.chip_preserves_saved_free_transfers("wildcard"),
            "play_now": play.as_dict(),
            "save_policy": {
                "objective": save_total,
                "route": route.as_dict(),
                "route_value_evidence": save_evidence,
                "retains_wildcard_option": bool(route.wildcard_available),
                "post_save_state_for_reservation": save_state,
            },
            "executable": False,
            "actionable": False,
        },
    )
