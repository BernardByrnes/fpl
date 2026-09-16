"""Canonical Free Hit route authority — one converter, three passes.

WHY THIS MODULE EXISTS
----------------------
Free Hit's decision compares two LEGAL POLICIES over the exact four-Gameweek
horizon:

    PLAY  H1   a temporary Free Hit fifteen
          H2-4 a canonical normal route from the RESTORED permanent state
    SAVE  H1-4 a canonical normal route from the CURRENT permanent state

The alternative to playing the chip is NOT "do nothing in H1".  It is the
accepted normal-transfer route, which may make a beneficial H1 transfer and
therefore enter H2 with a DIFFERENT squad, bank, acquisition basis and
free-transfer count.  Reconstructing a fake "SAVE = unchanged squad + zero
transfers" arm is wrong, and so is accepting a caller's per-event point values:
a route is state authority, and its VALUE must be computed from the route.

THE THREE PASSES, IN THIS ORDER ON PURPOSE
------------------------------------------
  PASS 1  validate the route's STRUCTURE and STATE AUTHORITY -- exact events,
          the exact starting state (squad, bank, free transfers AND acquisition
          basis), per-event continuity, squad legality, FT bounds, terminal
          equality.
  PASS 2  evaluate THAT SAME route canonically with
          ``route_optimizer.exact_evaluate``.
  PASS 3  build the evidence DTO from the canonical per-event values.

A structure failure refuses in PASS 1, before anything numeric happens.  There is
deliberately NO ``mean_net_core``, ``evaluation``, ``values`` or ``per_event``
parameter anywhere in this module: a caller has no channel through which to
inject route points, so the 1,000,000-per-event attack is not defended against,
it is unrepresentable.

HITS
----
``exact_evaluate`` returns ``mean_gross_core`` and ``mean_net_core`` side by
side, with ``net_core = gross_core - cumulative_hits``.  This module copies
``mean_net_core`` and never subtracts a hit again -- double-counting a -4 hit as
an 8-point swing is the one authority this contract exists to protect.
``hit_points`` is carried for EVIDENCE only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from . import season_rules as sr
from . import transfer_state as ts

FREE_HIT_ROUTE_CONVERTER_VERSION = "free_hit_route_converter_v1.0.0"

#: The arm labels.  One definition, so a route cannot be mislabelled by accident.
ARM_PLAY = "PLAY"
ARM_SAVE = "SAVE"

FH_ROUTE_INVALID = "FREE_HIT_ROUTE_INVALID"
FH_ROUTE_START_STATE_MISMATCH = "FREE_HIT_ROUTE_START_STATE_MISMATCH"
FH_ROUTE_BASIS_MISMATCH = "FREE_HIT_ROUTE_ACQUISITION_BASIS_MISMATCH"
FH_ROUTE_CONTINUITY_MISMATCH = "FREE_HIT_ROUTE_STATE_CONTINUITY_MISMATCH"
FH_ROUTE_EVALUATION_INPUTS_REQUIRED = "FREE_HIT_ROUTE_EVALUATION_INPUTS_REQUIRED"


class FreeHitRouteError(ValueError):
    """A supplied route is not a canonical Free Hit route."""

    def __init__(self, message: str, *, reasons: Sequence[str] = ()) -> None:
        super().__init__(message)
        self.reasons = tuple(str(reason) for reason in reasons)


def _squad_of(state: Any) -> tuple[int, ...]:
    return tuple(sorted(int(player.player_id) for player in state.players))


def _basis_of(state: Any) -> dict[int, int]:
    return {int(player.player_id): int(player.purchase_price_tenths) for player in state.players}


def _positions_of(state: Any) -> dict[int, str]:
    return {int(player.player_id): str(player.position) for player in state.players}


def _clubs_of(state: Any) -> dict[int, int]:
    counts: dict[int, int] = {}
    for player in state.players:
        counts[int(player.club_id)] = counts.get(int(player.club_id), 0) + 1
    return counts


def state_fingerprint(state: Any) -> dict[str, Any]:
    """The canonical identity of one ``RouteState``.

    Squad, bank, free transfers AND acquisition basis.  The basis is included
    deliberately: two states can agree on every id and the bank while a retained
    player's basis differs, which silently changes what a later sale is worth.
    """

    return {
        "event": int(getattr(state, "event")),
        "squad_ids": list(_squad_of(state)),
        "bank_tenths": int(state.bank_tenths),
        "free_transfers": int(state.free_transfers),
        "purchase_price_tenths": {int(k): int(v) for k, v in sorted(_basis_of(state).items())},
    }


@dataclass(frozen=True)
class FreeHitRouteEvent:
    """One canonically evaluated event of a Free Hit route.

    Every numeric field is COPIED from ``route_optimizer.exact_evaluate`` on the
    validated route.  Nothing here is accepted from a caller.
    """

    event: int
    squad_ids: tuple[int, ...]
    bank_tenths: int
    purchase_price_tenths: Mapping[int, int]
    free_transfers: int
    mean_net_core: float
    mean_gross_core: float
    hit_points: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "event": int(self.event),
            "squad_ids": [int(p) for p in self.squad_ids],
            "bank_tenths": int(self.bank_tenths),
            "free_transfers": int(self.free_transfers),
            "mean_net_core": round(float(self.mean_net_core), 6),
            "mean_gross_core": round(float(self.mean_gross_core), 6),
            "hit_points": int(self.hit_points),
        }


@dataclass(frozen=True)
class FreeHitRoute:
    """A canonically evaluated route for ONE arm, start state included."""

    arm: str
    start_state: Mapping[str, Any]
    events: tuple[FreeHitRouteEvent, ...]
    terminal_state: Mapping[str, Any]
    cumulative_hits: int = 0
    converter_version: str = FREE_HIT_ROUTE_CONVERTER_VERSION

    def value(self) -> float:
        return float(sum(float(entry.mean_net_core) for entry in self.events))

    def event_for(self, event: int) -> FreeHitRouteEvent | None:
        for entry in self.events:
            if int(entry.event) == int(event):
                return entry
        return None

    def as_dict(self) -> dict[str, Any]:
        return {
            "arm": str(self.arm),
            "converter_version": str(self.converter_version),
            "start_state": dict(self.start_state),
            "events": [entry.as_dict() for entry in self.events],
            "terminal_state": dict(self.terminal_state),
            "value": round(self.value(), 6),
            "cumulative_hits": int(self.cumulative_hits),
        }


def _expected_state_problems(actual: Any, expected: Mapping[str, Any], *, label: str) -> list[str]:
    """Every way ``actual`` differs from the expected canonical start state."""

    found: list[str] = []
    if _squad_of(actual) != tuple(int(p) for p in expected.get("squad_ids") or ()):
        found.append(f"{label}: squad differs")
    if int(actual.bank_tenths) != int(expected.get("bank_tenths", -(10**9))):
        found.append(f"{label}: bank differs")
    if int(actual.free_transfers) != int(expected.get("free_transfers", -(10**9))):
        found.append(f"{label}: free transfers differ")
    expected_basis = {int(k): int(v) for k, v in (expected.get("purchase_price_tenths") or {}).items()}
    actual_basis = _basis_of(actual)
    if actual_basis != expected_basis:
        differing = sorted(
            pid for pid in set(actual_basis) | set(expected_basis)
            if int(actual_basis.get(pid, -1)) != int(expected_basis.get(pid, -1))
        )
        found.append(f"{label}: acquisition basis differs for {differing[:8]}")
    if int(getattr(actual, "event")) != int(expected.get("event", getattr(actual, "event"))):
        found.append(f"{label}: starting event differs")
    return found


def free_hit_route_from_canonical_route(
    partial: Any,
    *,
    arm: str,
    expected_events: Sequence[int],
    rules: sr.SeasonRules,
    expected_start_state: Mapping[str, Any] | None = None,
    worlds_by_event: Mapping[int, Any] | None = None,
    positions_of: Any | None = None,
    route_config: Any | None = None,
    events: Sequence[int] | None = None,
    cache: Mapping[Any, Any] | None = None,
) -> FreeHitRoute:
    """Validate and canonically evaluate ONE arm's route, or refuse.

    There is NO parameter through which a caller can supply a route VALUE.
    """

    expected = tuple(int(event) for event in expected_events)
    if str(arm) not in (ARM_PLAY, ARM_SAVE):
        raise FreeHitRouteError(
            f"{FH_ROUTE_INVALID}: unknown arm {arm!r}", reasons=(FH_ROUTE_INVALID,)
        )

    # ------------------------------------------------------------------
    # PASS 1 — structure and state authority
    # ------------------------------------------------------------------
    actions = tuple(getattr(partial, "actions", ()) or ())
    if len(actions) != len(expected):
        raise FreeHitRouteError(
            f"{FH_ROUTE_INVALID}: a {arm} route must have exactly {len(expected)} events, "
            f"got {len(actions)}",
            reasons=(FH_ROUTE_INVALID,),
        )
    route_events = tuple(int(action.get("event", -1)) for action in actions)
    if route_events != expected:
        raise FreeHitRouteError(
            f"{FH_ROUTE_INVALID}: {arm} route events {list(route_events)} != {list(expected)}",
            reasons=(FH_ROUTE_INVALID,),
        )
    terminal = getattr(partial, "state", None)
    if terminal is None:
        raise FreeHitRouteError(
            f"{FH_ROUTE_INVALID}: the {arm} route has no canonical terminal state",
            reasons=(FH_ROUTE_INVALID,),
        )

    if expected_start_state is not None:
        first_transition = actions[0].get("transition")
        first_before = getattr(first_transition, "before_state", None) if first_transition else None
        if first_before is None:
            raise FreeHitRouteError(
                f"{FH_ROUTE_INVALID}: the {arm} route's first event carries no before-state",
                reasons=(FH_ROUTE_INVALID,),
            )
        start_problems = _expected_state_problems(
            first_before, expected_start_state, label=f"{arm} route start"
        )
        if start_problems:
            token = (
                FH_ROUTE_BASIS_MISMATCH
                if any("acquisition basis" in problem for problem in start_problems)
                else FH_ROUTE_START_STATE_MISMATCH
            )
            raise FreeHitRouteError(f"{token}: " + "; ".join(start_problems[:4]), reasons=(token,))

    previous_state = None
    entries: list[dict[str, Any]] = []
    previous_hits = 0
    for action in actions:
        event = int(action["event"])
        transition = action.get("transition")
        if transition is None:
            raise FreeHitRouteError(
                f"{FH_ROUTE_INVALID}: {arm} event {event} carries no canonical transition",
                reasons=(FH_ROUTE_INVALID,),
            )
        if not bool(getattr(transition, "ok", False)):
            raise FreeHitRouteError(
                f"{FH_ROUTE_INVALID}: {arm} event {event} transition is not a successful canonical "
                f"transition: {getattr(transition, 'errors', ())}",
                reasons=(FH_ROUTE_INVALID,),
            )
        next_state = getattr(transition, "next_event_state", None)
        squad_after = getattr(transition, "squad_after", None)
        if next_state is None or squad_after is None:
            raise FreeHitRouteError(
                f"{FH_ROUTE_INVALID}: {arm} event {event} transition exposes no resulting state",
                reasons=(FH_ROUTE_INVALID,),
            )
        if (
            _squad_of(squad_after) != _squad_of(next_state)
            or _basis_of(squad_after) != _basis_of(next_state)
            or int(squad_after.bank_tenths) != int(next_state.bank_tenths)
        ):
            raise FreeHitRouteError(
                f"{FH_ROUTE_CONTINUITY_MISMATCH}: {arm} event {event} squad_after disagrees with "
                "next_event_state on squad, bank or acquisition basis",
                reasons=(FH_ROUTE_CONTINUITY_MISMATCH,),
            )
        before = getattr(transition, "before_state", None)
        if previous_state is not None and before is not None:
            if (
                _squad_of(before) != _squad_of(previous_state)
                or int(before.bank_tenths) != int(previous_state.bank_tenths)
                or int(before.free_transfers) != int(previous_state.free_transfers)
                or _basis_of(before) != _basis_of(previous_state)
            ):
                raise FreeHitRouteError(
                    f"{FH_ROUTE_CONTINUITY_MISMATCH}: {arm} event {event} does not continue from "
                    "the previous event's resulting state (squad, bank, free transfers and "
                    "acquisition basis must all carry over)",
                    reasons=(FH_ROUTE_CONTINUITY_MISMATCH,),
                )
        previous_state = next_state

        squad = _squad_of(next_state)
        positions = _positions_of(next_state)
        clubs = {int(p.player_id): int(p.club_id) for p in next_state.players}
        composition = ts.squad_composition_errors(squad, positions, clubs)
        if composition:
            raise FreeHitRouteError(
                f"{FH_ROUTE_INVALID}: {arm} event {event}: " + "; ".join(composition[:4]),
                reasons=(FH_ROUTE_INVALID,),
            )
        over = {
            club: count for club, count in _clubs_of(next_state).items() if count > ts.SQUAD_TEAM_LIMIT
        }
        if over:
            raise FreeHitRouteError(
                f"{FH_ROUTE_INVALID}: {arm} event {event} exceeds the club limit: {over}",
                reasons=(FH_ROUTE_INVALID,),
            )
        basis = _basis_of(next_state)
        missing_basis = sorted(pid for pid in squad if pid not in basis)
        if missing_basis:
            raise FreeHitRouteError(
                f"{FH_ROUTE_BASIS_MISMATCH}: {arm} event {event} has no acquisition basis for "
                f"{missing_basis[:8]}",
                reasons=(FH_ROUTE_BASIS_MISMATCH,),
            )
        ft_after = int(next_state.free_transfers)
        if ft_after < 0 or ft_after > int(rules.max_free_transfers):
            raise FreeHitRouteError(
                f"{FH_ROUTE_INVALID}: {arm} event {event} free transfers {ft_after} is outside the "
                f"canonical range 0..{int(rules.max_free_transfers)}",
                reasons=(FH_ROUTE_INVALID,),
            )
        hit = int(action.get("hit_points") or 0)
        if hit < 0 or (hit % 4) != 0:
            raise FreeHitRouteError(
                f"{FH_ROUTE_INVALID}: {arm} event {event} hit_points {hit} is not a canonical hit "
                "multiple of 4",
                reasons=(FH_ROUTE_INVALID,),
            )
        previous_hits = hit
        entries.append({"event": event, "state": next_state, "hit_points": hit})

    if (
        _squad_of(terminal) != _squad_of(previous_state)
        or int(terminal.bank_tenths) != int(previous_state.bank_tenths)
        or int(terminal.free_transfers) != int(previous_state.free_transfers)
        or _basis_of(terminal) != _basis_of(previous_state)
    ):
        raise FreeHitRouteError(
            f"{FH_ROUTE_INVALID}: the {arm} route's terminal state is not its final event state "
            "(squad, bank, free transfers and acquisition basis must all match)",
            reasons=(FH_ROUTE_INVALID,),
        )

    # ------------------------------------------------------------------
    # PASS 2 — canonical exact evaluation of THIS route
    # ------------------------------------------------------------------
    if worlds_by_event is None or positions_of is None:
        raise FreeHitRouteError(
            f"{FH_ROUTE_EVALUATION_INPUTS_REQUIRED}: the canonical exact-evaluation inputs (worlds "
            "and positions) are required; route values are never accepted from the caller",
            reasons=(FH_ROUTE_EVALUATION_INPUTS_REQUIRED,),
        )
    from . import route_optimizer as _ro

    evaluation = _ro.exact_evaluate(
        partial,
        worlds_by_event={int(e): matrix for e, matrix in worlds_by_event.items()},
        positions_of=positions_of,
        cache=dict(cache) if cache is not None else {},
        config=route_config,
        events=tuple(int(e) for e in (events or expected)),
    )
    per_event = {int(row["event"]): row for row in (evaluation.get("per_event") or [])}
    missing_values = [event for event in expected if event not in per_event]
    if missing_values:
        raise FreeHitRouteError(
            f"{FH_ROUTE_INVALID}: the canonical evaluation returned no value for event(s) "
            f"{missing_values}",
            reasons=(FH_ROUTE_INVALID,),
        )

    # ------------------------------------------------------------------
    # PASS 3 — build the DTO from the CANONICAL values
    # ------------------------------------------------------------------
    first_transition = actions[0].get("transition")
    first_before = getattr(first_transition, "before_state", None)
    start_state = state_fingerprint(first_before) if first_before is not None else (
        dict(expected_start_state) if expected_start_state is not None else state_fingerprint(entries[0]["state"])
    )
    built = tuple(
        FreeHitRouteEvent(
            event=int(entry["event"]),
            squad_ids=_squad_of(entry["state"]),
            bank_tenths=int(entry["state"].bank_tenths),
            purchase_price_tenths=_basis_of(entry["state"]),
            free_transfers=int(entry["state"].free_transfers),
            mean_net_core=float(per_event[int(entry["event"])]["mean_net_core"]),
            mean_gross_core=float(per_event[int(entry["event"])]["mean_gross_core"]),
            hit_points=int(per_event[int(entry["event"])].get("hit_points") or 0),
        )
        for entry in entries
    )
    return FreeHitRoute(
        arm=str(arm),
        start_state=start_state,
        events=built,
        terminal_state=state_fingerprint(terminal),
        cumulative_hits=int(evaluation.get("cumulative_hits") or 0),
    )
