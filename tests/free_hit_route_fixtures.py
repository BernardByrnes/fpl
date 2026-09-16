"""Canonical Free Hit route fixtures — real route objects, no fake DTOs.

Every route here is built the way production builds one: ``RouteState`` +
``RoutePlayer``, a real ``TransferBatch`` per event, and
``transfer_state.apply_transfer_batch`` producing the canonical
``TransferTransitionResult`` (before_state / squad_after / next_event_state,
bank and free-transfer progression, hits).  Nothing hand-writes an H2 state.

The helper deliberately accepts TRANSFERS, never point values.  Route values are
produced only by ``route_optimizer.exact_evaluate`` on the route it builds.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fpl_brain import manager_lineup as ml  # noqa: E402
from fpl_brain import route_optimizer as ro  # noqa: E402
from fpl_brain import season_rules as sr  # noqa: E402
from fpl_brain import transfer_state as ts  # noqa: E402

RULES = sr.SeasonRules(season="2026/27")

#: GKP 1,2 | DEF 3-7 | MID 8-12 | FWD 13-15, with clubs spread so a legal
#: fifteen is constructible under the 3-per-club limit.
SQUAD_IDS: tuple[int, ...] = tuple(range(1, 16))
POSITION: dict[int, str] = {
    1: "GKP", 2: "GKP",
    3: "DEF", 4: "DEF", 5: "DEF", 6: "DEF", 7: "DEF",
    8: "MID", 9: "MID", 10: "MID", 11: "MID", 12: "MID",
    13: "FWD", 14: "FWD", 15: "FWD",
}
#: Five clubs, three players each — a legal fifteen under the 3-per-club limit.
CLUB: dict[int, int] = {
    1: 100, 2: 101, 3: 102, 4: 103, 5: 104,
    6: 100, 7: 101, 8: 102, 9: 103, 10: 104,
    11: 100, 12: 101, 13: 102, 14: 103, 15: 104,
}
#: Unowned replacements, one per position, on clubs that keep the limit legal.
BENCH_POOL: tuple[int, ...] = (16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27)
POOL_POSITION: dict[int, str] = {
    16: "GKP", 17: "GKP", 18: "DEF", 19: "DEF", 20: "DEF",
    21: "MID", 22: "MID", 23: "MID", 24: "FWD", 25: "FWD", 26: "FWD", 27: "DEF",
}
POOL_CLUB: dict[int, int] = {pid: 200 + (index % 8) for index, pid in enumerate(BENCH_POOL)}

UNIVERSE: tuple[int, ...] = tuple(sorted((*SQUAD_IDS, *BENCH_POOL)))
BASE_PURCHASE = 50


def player_meta() -> dict[int, ts.PlayerMeta]:
    return {
        pid: ts.PlayerMeta(pid, POSITION.get(pid) or POOL_POSITION[pid], CLUB.get(pid) or POOL_CLUB[pid])
        for pid in UNIVERSE
    }


def positions_of(squad_ids: Sequence[int]) -> dict[int, str]:
    return {int(pid): (POSITION.get(int(pid)) or POOL_POSITION[int(pid)]) for pid in squad_ids}


def start_state(
    *, event: int, squad_ids: Sequence[int] = SQUAD_IDS, bank_tenths: int = 0,
    free_transfers: int = 2, purchase: int = BASE_PURCHASE,
    basis: Mapping[int, int] | None = None,
) -> ts.RouteState:
    return ts.RouteState(
        event=int(event),
        players=tuple(
            ts.RoutePlayer(int(pid), POSITION.get(int(pid)) or POOL_POSITION[int(pid)],
                           CLUB.get(int(pid)) or POOL_CLUB[int(pid)],
                           int((basis or {}).get(int(pid), purchase)))
            for pid in squad_ids
        ),
        bank_tenths=int(bank_tenths),
        free_transfers=int(free_transfers),
    )


def world_matrix(
    *, event: int, worlds: int = 24, scores: Mapping[int, float] | None = None,
    minutes: Mapping[int, float] | None = None,
) -> dict[str, Any]:
    """A per-event world matrix in the shape the accepted evaluator consumes."""

    scores = scores or {}
    minutes = minutes or {}
    player_ids = list(UNIVERSE)
    return {
        "worlds": int(worlds),
        "player_ids": player_ids,
        "minutes": {
            pid: tuple(float(minutes.get(pid, 90.0)) for _ in range(int(worlds))) for pid in player_ids
        },
        "core": {
            pid: tuple(float(scores.get(pid, 1.0)) for _ in range(int(worlds))) for pid in player_ids
        },
    }


def route_config(*, selection_worlds: int = 12) -> ro.OptimizerConfig:
    return ro.OptimizerConfig(policy_selection_worlds=int(selection_worlds))


def build_canonical_route(
    *,
    start: ts.RouteState,
    universe: Sequence[int] | None = None,
    price_default: int = BASE_PURCHASE,
    events: Sequence[int],
    transfers: Mapping[int, Sequence[tuple[int, int]]] | None = None,
    prices: Mapping[int, Mapping[int, int]] | None = None,
    meta: Mapping[int, ts.PlayerMeta] | None = None,
    rules: sr.SeasonRules = RULES,
) -> tuple[ro.PartialRoute, ts.RouteState]:
    """Build a genuine ``PartialRoute`` by applying real transfer batches.

    ``transfers`` maps an event to ``(out_player_id, in_player_id)`` pairs.  An
    event with no entry ROLLS (an empty batch), which the canonical transition
    still records — so a "no transfer" event is a real canonical action rather
    than a missing one.
    """

    transfers = transfers or {}
    prices = prices or {}
    universe = tuple(int(p) for p in (universe if universe is not None else UNIVERSE))
    meta = meta if meta is not None else player_meta()
    current = start
    actions: list[dict[str, Any]] = []
    for event in events:
        event = int(event)
        pairs = tuple(transfers.get(event) or ())
        batch = ts.TransferBatch(tuple(ts.TransferAction(int(out), int(in_)) for out, in_ in pairs))
        # The canonical snapshot prices the WHOLE universe; an override shifts the
        # named players only, so a transition always has the prices it needs.
        snapshot_prices = {pid: int(price_default) for pid in universe}
        snapshot_prices.update(
            {int(pid): int(value) for pid, value in (prices.get(event) or {}).items()}
        )
        snapshot = ts.PriceSnapshot(event=event, prices=snapshot_prices)
        result = ts.apply_transfer_batch(current, batch, snapshot, meta)
        if not result.ok:
            raise AssertionError(f"event {event} transition rejected: {result.errors}")
        next_state = result.next_event_state
        actions.append({
            "event": event,
            "kind": ("ROLL" if not batch.actions else "NORMAL_TRANSFER"),
            "batch": batch,
            "transition": result,
            "hit_points": int(result.hit_points),
            "squad_ids": tuple(sorted(int(p.player_id) for p in next_state.players)),
            "ft_after": int(next_state.free_transfers),
            "bank_after": int(next_state.bank_tenths),
        })
        current = next_state
    return (
        ro.PartialRoute(state=current, actions=tuple(actions), h1_proxy=0.0, window_proxy=0.0, hits=0),
        current,
    )


def evaluate_route(
    partial: ro.PartialRoute, *, events: Sequence[int], worlds: Mapping[int, Mapping[str, Any]] | None = None,
    worlds_default: Mapping[str, Any] | None = None, selection_worlds: int = 12,
) -> dict[str, Any]:
    """Run the accepted exact evaluator over a fixture route."""

    bodies = {int(event): dict(worlds_default or world_matrix(event=int(event))) for event in events}
    if worlds:
        bodies.update({int(event): dict(matrix) for event, matrix in worlds.items()})
    return ro.exact_evaluate(
        partial,
        worlds_by_event=bodies,
        positions_of=positions_of,
        cache={},
        config=route_config(selection_worlds=selection_worlds),
        events=tuple(int(event) for event in events),
    )
