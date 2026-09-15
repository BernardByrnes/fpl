"""One-way converter: canonical normal route -> Wildcard SAVE input.

Production authority must be:

    canonical accepted normal route result
        -> this converter
        -> WildcardSaveRoute
        -> Wildcard evaluator

and NOT:

    caller constructs WildcardSaveRoute -> the adapter trusts it

Everything this module copies comes from the canonical route objects traced in
``route_optimizer`` / ``transfer_state``:

  * ``PartialRoute.state``      -- the terminal canonical state (squad, bank, FT)
  * ``PartialRoute.actions``    -- one record per event, each carrying its own
                                   canonical ``TransferTransitionResult``
                                   (``before_state`` / ``squad_after`` /
                                   ``next_event_state`` / ``bank_after_tenths`` /
                                   ``free_transfers`` / ``hit_points``)
  * ``RoutePlayer.purchase_price_tenths`` -- the acquisition basis, owned by the
                                   canonical state rather than by the caller
  * ``exact_evaluate(...)["per_event"][i]["mean_net_core"]`` -- the route engine's
                                   OWN per-event value, which ALREADY nets that
                                   event's hit

Nothing is defaulted.  A route that cannot be proven canonical refuses.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from . import chip_wildcard as wc
from . import season_rules as sr
from . import transfer_state as ts

WILDCARD_SAVE_ROUTE_CONVERTER_VERSION = "wildcard_save_route_converter_v1.0.0"

#: A normal SAVE route is exactly the accepted four-GW normal-transfer horizon.
NORMAL_ROUTE_LENGTH = 4


class WildcardSaveRouteError(wc.WildcardInputError):
    """The canonical route could not be proven, so SAVE refuses."""


def _squad_of(state: Any) -> tuple[int, ...]:
    return tuple(sorted(int(p.player_id) for p in state.players))


def _basis_of(state: Any) -> dict[int, int]:
    return {int(p.player_id): int(p.purchase_price_tenths) for p in state.players}


def _positions_of(state: Any) -> dict[int, str]:
    return {int(p.player_id): str(p.position) for p in state.players}


def _check_squad_legal(
    squad: Sequence[int], positions: Mapping[int, str], rules: sr.SeasonRules, *, label: str
) -> list[str]:
    """Canonical squad legality: size, composition, club limit.

    Reuses ``transfer_state``'s own constants rather than restating the rules, so
    a divergence is impossible.
    """

    problems: list[str] = []
    ids = [int(p) for p in squad]
    if len(ids) != ts.SQUAD_SIZE:
        problems.append(f"{label}: squad has {len(ids)} players, expected {ts.SQUAD_SIZE}")
    if len(set(ids)) != len(ids):
        problems.append(f"{label}: squad contains duplicate player ids")
    counts: dict[str, int] = {}
    clubs: dict[int, int] = {}
    for pid in ids:
        position = positions.get(pid)
        if position is None:
            problems.append(f"{label}: player {pid} has no canonical position")
            continue
        counts[position] = counts.get(position, 0) + 1
    for position, required in ts.POSITION_COMPOSITION.items():
        if counts.get(position, 0) != required:
            problems.append(f"{label}: {position}={counts.get(position, 0)} (need {required})")
    # the club limit needs club ids, which the canonical action carries
    return problems


def wildcard_save_route_from_canonical_route(
    partial: Any,
    evaluation: Mapping[str, Any],
    *,
    planning_event: int,
    horizon: wc.WildcardHorizonSpec,
    rules: sr.SeasonRules,
    expected_horizon: Sequence[int] | None = None,
    club_of: Mapping[int, int] | None = None,
) -> wc.WildcardSaveRoute:
    """Convert ONE canonical route result into the Wildcard SAVE input, or refuse.

    ``partial`` is the canonical ``PartialRoute`` produced by the accepted route
    engine and ``evaluation`` is its ``exact_evaluate`` result (the source of
    ``mean_net_core``).  Both are required: a route without its own evaluation
    cannot supply a value, and a value without a route cannot supply a state.
    """

    problems: list[str] = []
    actions = tuple(getattr(partial, "actions", ()) or ())
    expected_events = tuple(int(e) for e in (expected_horizon or horizon.events[:NORMAL_ROUTE_LENGTH]))

    if len(actions) != NORMAL_ROUTE_LENGTH:
        problems.append(
            f"a normal SAVE route must have exactly {NORMAL_ROUTE_LENGTH} events, got {len(actions)}"
        )
    events = tuple(int(a.get("event", -1)) for a in actions)
    if events and events != expected_events:
        problems.append(f"route events {list(events)} != the normal horizon {list(expected_events)}")

    per_event = {int(row["event"]): row for row in (evaluation.get("per_event") or ())}
    terminal = getattr(partial, "state", None)
    if terminal is None:
        problems.append("the route has no canonical terminal state")
    if problems:
        raise WildcardSaveRouteError(
            f"{wc.WC_SAVE_ROUTE_INVALID}: {'; '.join(problems[:6])}",
            reasons=(wc.WC_SAVE_ROUTE_INVALID,),
        )

    entries: list[wc.WildcardSaveRouteEvent] = []
    previous_state = None
    for action in actions:
        event = int(action["event"])
        transition = action.get("transition")
        if transition is None:
            raise WildcardSaveRouteError(
                f"{wc.WC_SAVE_ROUTE_INVALID}: event {event} carries no canonical transition",
                reasons=(wc.WC_SAVE_ROUTE_INVALID,),
            )
        next_state = getattr(transition, "next_event_state", None)
        squad_after = getattr(transition, "squad_after", None)
        if next_state is None or squad_after is None:
            raise WildcardSaveRouteError(
                f"{wc.WC_SAVE_ROUTE_INVALID}: event {event} transition exposes no resulting state",
                reasons=(wc.WC_SAVE_ROUTE_INVALID,),
            )

        # --- CONTINUITY is proven from the CANONICAL TRANSITIONS, not by
        # inspecting whether each squad merely "looks legal":
        #   previous RouteState + canonical transition == next_event_state
        # and squad_after must agree with the resulting state.
        if _squad_of(squad_after) != _squad_of(next_state):
            raise WildcardSaveRouteError(
                f"{wc.WC_SAVE_ROUTE_INVALID}: event {event} squad_after disagrees with "
                "next_event_state",
                reasons=(wc.WC_SAVE_ROUTE_INVALID,),
            )
        if (_basis_of(squad_after) != _basis_of(next_state)
                or int(squad_after.bank_tenths) != int(next_state.bank_tenths)):
            raise WildcardSaveRouteError(
                f"{wc.WC_SAVE_ROUTE_INVALID}: event {event} squad_after disagrees with "
                "next_event_state on bank or acquisition basis",
                reasons=(wc.WC_SAVE_ROUTE_INVALID,),
            )
        before = getattr(transition, "before_state", None)
        if previous_state is not None and before is not None:
            if (_squad_of(before) != _squad_of(previous_state)
                    or int(before.bank_tenths) != int(previous_state.bank_tenths)
                    or _basis_of(before) != _basis_of(previous_state)):
                raise WildcardSaveRouteError(
                    f"{wc.WC_SAVE_ROUTE_INVALID}: event {event} does not continue from the "
                    "previous event's resulting state",
                    reasons=(wc.WC_SAVE_ROUTE_INVALID,),
                )
        previous_state = next_state

        squad = _squad_of(next_state)
        positions = _positions_of(next_state)
        squad_problems = _check_squad_legal(squad, positions, rules, label=f"event {event}")
        if squad_problems:
            raise WildcardSaveRouteError(
                f"{wc.WC_SAVE_ROUTE_INVALID}: {'; '.join(squad_problems[:4])}",
                reasons=(wc.WC_SAVE_ROUTE_INVALID,),
            )
        # The club ids come from the canonical state itself (RoutePlayer.club_id),
        # so an optional caller map can only ADD evidence, never replace it.
        clubs: dict[int, int] = {}
        for player in next_state.players:
            clubs[int(player.club_id)] = clubs.get(int(player.club_id), 0) + 1
        if club_of is not None:
            for pid in squad:
                club = club_of.get(int(pid))
                if club is not None:
                    clubs[int(club)] = clubs.get(int(club), 0) + 1
        over = {c: n for c, n in clubs.items() if n > ts.SQUAD_TEAM_LIMIT}
        if over:
            raise WildcardSaveRouteError(
                f"{wc.WC_SAVE_ROUTE_INVALID}: event {event} exceeds the club limit: {over}",
                reasons=(wc.WC_SAVE_ROUTE_INVALID,),
            )

        # FT bounds from the canonical season rule, not a guessed maximum.
        ft_after = int(next_state.free_transfers)
        if ft_after < 0 or ft_after > int(rules.max_free_transfers):
            raise WildcardSaveRouteError(
                f"{wc.WC_SAVE_ROUTE_INVALID}: event {event} free transfers {ft_after} is outside "
                f"the canonical range 0..{int(rules.max_free_transfers)}",
                reasons=(wc.WC_SAVE_ROUTE_INVALID,),
            )

        # --- mean_net_core: the route engine's OWN value for this event, which
        # already nets the hit.  A route event with no evaluated value refuses
        # rather than being given a number.
        row = per_event.get(event)
        if row is None or row.get("mean_net_core") is None:
            raise WildcardSaveRouteError(
                f"{wc.WC_SAVE_ROUTE_INVALID}: event {event} has no canonical exact-evaluation value",
                reasons=(wc.WC_SAVE_ROUTE_INVALID,),
            )
        net = float(row["mean_net_core"])
        if not math.isfinite(net):
            raise WildcardSaveRouteError(
                f"{wc.WC_SAVE_ROUTE_INVALID}: event {event} mean_net_core is not finite",
                reasons=(wc.WC_SAVE_ROUTE_INVALID,),
            )

        entries.append(wc.WildcardSaveRouteEvent(
            event=event,
            squad_ids=squad,
            bank_tenths=int(next_state.bank_tenths),
            purchase_price_tenths=_basis_of(next_state),
            free_transfers=ft_after,
            mean_net_core=net,
            hit_points=int(action.get("hit_points") or 0),
            actions=(str(action.get("kind") or ""),)
        ))

    # --- TERMINAL EQUALITY: the SAVE terminal state IS the canonical final state,
    # never a separately supplied lookalike.
    # The comparison includes the per-player ACQUISITION BASIS: a terminal state
    # that matches on squad/bank/FT but carries different purchase prices is a
    # different state, and a tampered basis must not survive.
    if (_squad_of(terminal) != _squad_of(previous_state)
            or int(terminal.bank_tenths) != int(previous_state.bank_tenths)
            or int(terminal.free_transfers) != int(previous_state.free_transfers)
            or _basis_of(terminal) != _basis_of(previous_state)):
        raise WildcardSaveRouteError(
            f"{wc.WC_SAVE_ROUTE_INVALID}: the route's terminal state is not its final event state "
            "(squad, bank, free transfers and acquisition basis must all match)",
            reasons=(wc.WC_SAVE_ROUTE_INVALID,),
        )

    terminal_ft = int(terminal.free_transfers)
    if terminal_ft < 0 or terminal_ft > int(rules.max_free_transfers):
        raise WildcardSaveRouteError(
            f"{wc.WC_SAVE_ROUTE_INVALID}: terminal free transfers {terminal_ft} is outside the "
            f"canonical range 0..{int(rules.max_free_transfers)}",
            reasons=(wc.WC_SAVE_ROUTE_INVALID,),
        )

    route = wc.WildcardSaveRoute(
        events=tuple(entries),
        terminal_squad_ids=_squad_of(terminal),
        terminal_bank_tenths=int(terminal.bank_tenths),
        terminal_purchase_price_tenths=_basis_of(terminal),
        terminal_free_transfers=terminal_ft,
        cumulative_hits=int(getattr(partial, "hits", 0) or 0),
        wildcard_available=True,
    )
    problems = route.problems()
    if problems:
        raise WildcardSaveRouteError(
            f"{wc.WC_SAVE_ROUTE_INVALID}: {'; '.join(problems[:6])}",
            reasons=(wc.WC_SAVE_ROUTE_INVALID,),
        )
    return route
