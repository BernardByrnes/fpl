"""Verified manager-state projection for future FH/WC evidence assembly.

This module binds future states to explicitly named verified decisions/routes
and certified continuation products. It never creates predictions or makes a
normal decision. A canonical world cache is read-only by default; materializing
a missing matrix requires an explicit flag and an isolated cache directory.
Lineup and route values are computed by the accepted route/lineup evaluators.
Returned manager state is a projection, never an official future manager fact.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from . import generation_store as gs
from . import route_optimizer as ro, transfer_state as ts
from .chip_route_assembly import (
    ChipRouteAssemblyError,
    VerifiedNormalRoute,
    _no_transfer_continuation_state,
    load_verified_normal_route,
)

PROJECTED_MANAGER_STATE_SCHEMA = "fpl_brain.projected_future_manager_state.v1"
NO_TRANSFER_CONTINUATION_MODEL = "CARRY_TERMINAL_ROUTE_STATE_NO_TRANSFERS_PER_EVENT_LINEUP_V1"


class FutureStateAssemblyError(ValueError):
    """A future state cannot be bound to the requested verified origin."""


@dataclass(frozen=True)
class ProjectedFutureManagerState:
    """A route-bound projection with explicit non-factual status."""

    source_decision_id: str
    source_generation_id: str
    source_route_id: str
    source_route_input_sha256: str
    origin_event: int
    origin_cutoff: str
    data_snapshot_sha256: str
    target_event: int
    projection_model: str
    season_rules_evidence: Mapping[str, Any]
    season_rules: Any
    state: Any

    def as_dict(self) -> dict[str, Any]:
        state = self.state
        return {
            "schema": PROJECTED_MANAGER_STATE_SCHEMA,
            "classification": "PROJECTED_ASSUMED_NOT_OFFICIAL_MANAGER_FACT",
            "source": {
                "decision_id": self.source_decision_id,
                "generation_id": self.source_generation_id,
                "route_id": self.source_route_id,
                "route_input_sha256": self.source_route_input_sha256,
                "origin_event": int(self.origin_event),
                "origin_cutoff": self.origin_cutoff,
                "data_snapshot_sha256": self.data_snapshot_sha256,
            },
            "target_event": int(self.target_event),
            "projection_model": self.projection_model,
            "season_rules_evidence": dict(self.season_rules_evidence),
            "manager_state": {
                "squad_ids": sorted(int(player.player_id) for player in state.players),
                "purchase_price_tenths": {
                    str(int(player.player_id)): int(player.purchase_price_tenths)
                    for player in state.players
                },
                "bank_tenths": int(state.bank_tenths),
                "free_transfers": int(state.free_transfers),
                "event_start_free_transfers": (
                    None if state.event_start_free_transfers is None
                    else int(state.event_start_free_transfers)
                ),
                "chip_state": list(state.chip_state),
            },
        }


@dataclass(frozen=True)
class ProjectedFutureRoute:
    """Future SAVE policy retaining verified route actions before continuation."""

    source_decision_id: str
    source_generation_id: str
    source_route_id: str
    target_event: int
    events: tuple[int, ...]
    retained_route_suffix_events: tuple[int, ...]
    no_transfer_continuation_events: tuple[int, ...]
    continuation_model: str | None
    season_rules_evidence: Mapping[str, Any]
    partial: Any

    def as_dict(self) -> dict[str, Any]:
        return {
            "classification": "PROJECTED_ROUTE_POLICY_NOT_OFFICIAL_MANAGER_FACT",
            "arm": "SAVE",
            "source": {
                "decision_id": self.source_decision_id,
                "generation_id": self.source_generation_id,
                "route_id": self.source_route_id,
            },
            "events": list(self.events),
            "retained_route_suffix_events": list(self.retained_route_suffix_events),
            "no_transfer_continuation_events": list(self.no_transfer_continuation_events),
            "continuation_model": self.continuation_model,
            "season_rules_evidence": dict(self.season_rules_evidence),
        }


def _origin_start_state(route: VerifiedNormalRoute) -> Any:
    actions = tuple(route.partial.actions or ())
    if not actions:
        raise FutureStateAssemblyError("verified normal route has no replayed actions")
    transition = actions[0].get("transition")
    state = getattr(transition, "before_state", None)
    if state is None or int(getattr(state, "event", -1)) != int(route.planning_event):
        raise FutureStateAssemblyError("verified normal route has no canonical origin start state")
    if getattr(state, "event_start_free_transfers", None) is None:
        raise FutureStateAssemblyError(
            "origin event-start free transfers are unknown; future FH/WC projection is refused"
        )
    return state


def _assert_transfer_rules_compatible(rules: Any) -> None:
    """Refuse pinned official rules that differ from transfer_state semantics."""

    expected = {
        "base_free_transfers": ts.FREE_TRANSFERS_PER_EVENT,
        "max_free_transfers": ts.MAX_STORED_FREE_TRANSFERS,
        "max_extra_free_transfers": ts.MAX_EXTRA_FREE_TRANSFERS,
        "squad_size": ts.SQUAD_SIZE,
        "squad_team_limit": ts.SQUAD_TEAM_LIMIT,
        "transfer_hit_cost": ts.HIT_POINTS_PER_EXTRA_TRANSFER,
        "sell_on_fee": ts.SELL_ON_FEE,
    }
    mismatches = [
        f"{name}={getattr(rules, name, None)!r} (transfer_state={value!r})"
        for name, value in expected.items()
        if getattr(rules, name, None) != value
    ]
    if bool(getattr(rules, "sell_at_purchase_price_when_falling", True)):
        mismatches.append("sell_at_purchase_price_when_falling=True (unsupported by transfer_state)")
    if mismatches:
        raise FutureStateAssemblyError(
            "pinned season rules are incompatible with accepted transfer_state semantics: "
            + "; ".join(mismatches)
        )


def _project_future_manager_state(
    route: VerifiedNormalRoute,
    *,
    target_event: int,
    rules: Any,
) -> tuple[Any, str]:
    """Project state through retained route actions, then explicit no-transfer continuation.

    Within the retained normal horizon, the preceding event's canonical
    ``next_event_state`` supplies the target state. Beyond it, the existing
    ``_no_transfer_continuation_state`` applies official free-transfer rules to
    the verified terminal route state. No kickoff, latest squad, or future
    manager endpoint is consulted.
    """

    _assert_transfer_rules_compatible(rules)
    start = _origin_start_state(route)
    del start  # Its purpose above is the fail-closed origin FT gate.
    target = int(target_event)
    if target <= int(route.planning_event):
        raise FutureStateAssemblyError("target event must follow the verified origin event")
    events = tuple(int(event) for event in route.events)
    if not events or events != tuple(range(events[0], events[0] + len(events))):
        raise FutureStateAssemblyError("verified route events are not a contiguous horizon")

    if target <= events[-1]:
        previous = target - 1
        matching = [row for row in route.partial.actions if int(row.get("event", -1)) == previous]
        if len(matching) != 1:
            raise FutureStateAssemblyError(
                f"verified route does not contain exactly one transition for GW{previous}"
            )
        transition = matching[0].get("transition")
        if transition is None or not bool(getattr(transition, "ok", False)):
            raise FutureStateAssemblyError(f"GW{previous} route transition is not proven successful")
        state = getattr(transition, "next_event_state", None)
        model = "VERIFIED_NORMAL_ROUTE_TRANSITIONS_V1"
    else:
        try:
            state = _no_transfer_continuation_state(route, event=target, rules=rules)
        except ChipRouteAssemblyError as failure:
            raise FutureStateAssemblyError(str(failure)) from failure
        model = NO_TRANSFER_CONTINUATION_MODEL

    if state is None or int(getattr(state, "event", -1)) != target:
        raise FutureStateAssemblyError("canonical projection did not produce the requested event state")
    if (
        getattr(state, "event_start_free_transfers", None) is None
        or int(getattr(state, "free_transfers", -1)) < 0
        or int(getattr(state, "bank_tenths", -1)) < 0
    ):
        raise FutureStateAssemblyError("projected state omits a required FT or bank value")
    return state, model


def _build_canonical_roll_route(
    start_state: Any,
    *,
    events: tuple[int, ...] | list[int],
    player_meta: Mapping[int, Any],
    rules: Any,
) -> Any:
    """Build a fixed no-transfer route using canonical transfer transitions.

    This is the explicit SAVE/continuation policy, not an optimizer result.
    ``apply_transfer_batch`` is used for every event. Since every batch is a
    roll, no price is consumed or forecast; the transition carries the exact
    squad basis and bank forward and applies the canonical FT rule.
    """

    _assert_transfer_rules_compatible(rules)
    target = tuple(int(event) for event in events)
    if not target or target != tuple(range(int(start_state.event), int(start_state.event) + len(target))):
        raise FutureStateAssemblyError("roll-route events must be contiguous from the start state")
    if getattr(start_state, "event_start_free_transfers", None) is None:
        raise FutureStateAssemblyError("roll-route start event has unknown event-start free transfers")
    state = start_state
    actions: list[dict[str, Any]] = []
    for event in target:
        batch = ts.TransferBatch.roll()
        price_snapshot = ts.PriceSnapshot(event=event, prices={})
        transition = ts.apply_transfer_batch(state, batch, price_snapshot, player_meta)
        if not transition.ok or transition.next_event_state is None:
            raise FutureStateAssemblyError(
                f"canonical no-transfer transition refused at GW{event}: {transition.errors[:6]}"
            )
        action = ro._action(event, "ROLL", batch, transition, transition.hit_points, 0.0)
        action["squad_ids"] = tuple(sorted(int(pid) for pid in transition.next_event_state.by_id()))
        action["ft_after"] = int(transition.next_event_state.free_transfers)
        action["bank_after"] = int(transition.next_event_state.bank_tenths)
        actions.append(action)
        state = transition.next_event_state
    return ro.PartialRoute(
        state=state,
        actions=tuple(actions),
        h1_proxy=0.0,
        window_proxy=0.0,
        hits=sum(int(row["hit_points"]) for row in actions),
    )


def _build_future_save_route(
    route: VerifiedNormalRoute,
    *,
    target_event: int,
    horizon_length: int,
    rules: Any,
    player_meta: Mapping[int, Any],
    season_rules_evidence: Mapping[str, Any],
) -> ProjectedFutureRoute:
    """Preserve verified normal-route suffix actions; roll only beyond that suffix."""

    _assert_transfer_rules_compatible(rules)
    start, _projection = _project_future_manager_state(
        route, target_event=int(target_event), rules=rules,
    )
    if int(horizon_length) <= 0:
        raise FutureStateAssemblyError("future route horizon must be positive")
    events = tuple(range(int(target_event), int(target_event) + int(horizon_length)))
    route_events = tuple(int(value) for value in route.events)
    suffix_events = tuple(event for event in events if event in route_events)
    expected_suffix = tuple(
        range(int(target_event), min(events[-1], route_events[-1]) + 1)
    ) if int(target_event) <= route_events[-1] else ()
    if suffix_events != expected_suffix:
        raise FutureStateAssemblyError("verified route suffix does not cover a contiguous future prefix")

    suffix_actions = tuple(
        row for row in route.partial.actions
        if int(target_event) <= int(row.get("event", -1)) <= (expected_suffix[-1] if expected_suffix else -1)
    )
    if suffix_actions:
        first_before = getattr(suffix_actions[0].get("transition"), "before_state", None)
        if first_before != start:
            raise FutureStateAssemblyError("verified route suffix does not begin at projected target state")
        for index, action in enumerate(suffix_actions):
            transition = action.get("transition")
            if transition is None or not bool(getattr(transition, "ok", False)):
                raise FutureStateAssemblyError("verified route suffix contains an unproven transition")
            if index and getattr(transition, "before_state", None) != getattr(
                suffix_actions[index - 1].get("transition"), "next_event_state", None,
            ):
                raise FutureStateAssemblyError("verified route suffix transitions are discontinuous")
        last_transition = suffix_actions[-1]["transition"]
        state = last_transition.next_event_state
        retained = suffix_actions
    else:
        state = start
        retained = ()

    continuation_events = tuple(event for event in events if event not in set(suffix_events))
    continuation = None
    if continuation_events:
        if continuation_events != tuple(range(int(state.event), int(state.event) + len(continuation_events))):
            raise FutureStateAssemblyError("continuation events do not begin at verified route terminal state")
        continuation = _build_canonical_roll_route(
            state, events=continuation_events, player_meta=player_meta, rules=rules,
        )

    all_actions = tuple(retained) + (() if continuation is None else tuple(continuation.actions))
    terminal = state if continuation is None else continuation.state
    partial = ro.PartialRoute(
        state=terminal,
        actions=all_actions,
        h1_proxy=0.0,
        window_proxy=0.0,
        hits=sum(int(row.get("hit_points") or 0) for row in all_actions),
    )
    if tuple(int(row.get("event", -1)) for row in all_actions) != events:
        raise FutureStateAssemblyError("assembled SAVE route does not cover its exact event horizon")
    return ProjectedFutureRoute(
        source_decision_id=str(route.source_decision_id),
        source_generation_id=str(route.generation_id),
        source_route_id=str(route.route_id),
        target_event=int(target_event),
        events=events,
        retained_route_suffix_events=tuple(int(row["event"]) for row in retained),
        no_transfer_continuation_events=continuation_events,
        continuation_model=NO_TRANSFER_CONTINUATION_MODEL if continuation_events else None,
        season_rules_evidence=dict(season_rules_evidence),
        partial=partial,
    )


def load_projected_future_manager_state(
    conn: Any,
    *,
    decision_id: str,
    generation_id: str,
    route_id: str,
    entry_id: int,
    origin_cutoff: str,
    data_snapshot_sha256: str,
    target_event: int,
    season: str,
) -> ProjectedFutureManagerState:
    """Reverify an explicit normal origin and return its future state projection.

    The caller must name the exact decision, generation, route, manager entry,
    cutoff, snapshot digest, target event, and season. This function only reads
    the evidence database and pinned snapshot. It never resolves a latest
    pointer or writes a manager fact.
    """

    string_fields = (
        decision_id, generation_id, route_id, origin_cutoff, data_snapshot_sha256, season,
    )
    if any(type(value) is not str or not value.strip() for value in string_fields):
        raise FutureStateAssemblyError("explicit origin identity is incomplete")
    if type(entry_id) is not int or entry_id <= 0:
        raise FutureStateAssemblyError("explicit manager entry identity must be a positive integer")
    if type(target_event) is not int or target_event <= 0:
        raise FutureStateAssemblyError("target event identity must be a positive integer")
    try:
        parsed_cutoff = datetime.fromisoformat(origin_cutoff.replace("Z", "+00:00"))
    except ValueError as failure:
        raise FutureStateAssemblyError("origin cutoff is not an ISO timestamp") from failure
    if parsed_cutoff.tzinfo is None:
        raise FutureStateAssemblyError("origin cutoff must include a timezone")
    digest = data_snapshot_sha256.removeprefix("sha256:")
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest.lower()):
        raise FutureStateAssemblyError("origin snapshot identity must be a SHA-256 digest")
    try:
        route = load_verified_normal_route(
            conn, str(decision_id), route_id=str(route_id), require_recommended=False,
        )
        generation = gs.load_generation(conn, str(generation_id))
        verification = gs.verify_generation(conn, str(generation_id))
    except Exception as failure:
        raise FutureStateAssemblyError(
            f"explicit normal decision/generation/route did not verify: {type(failure).__name__}: {failure}"
        ) from failure
    if not verification.get("verified"):
        raise FutureStateAssemblyError("explicit source generation is not verified")
    if (
        route.source_decision_id != str(decision_id)
        or route.generation_id != str(generation_id)
        or route.route_id != str(route_id)
        or str(route.cutoff) != str(origin_cutoff)
        or str(route.data_snapshot_sha256).removeprefix("sha256:")
        != str(data_snapshot_sha256).removeprefix("sha256:")
        or generation.horizon_kind != gs.HORIZON_KIND_FOUR_GW
        or int(generation.planning_event) != int(route.planning_event)
        or str(generation.cutoff) != str(origin_cutoff)
        or str(generation.snapshot.get("sha256") or "").removeprefix("sha256:")
        != str(data_snapshot_sha256).removeprefix("sha256:")
    ):
        raise FutureStateAssemblyError("verified route/generation differs from the explicit origin identity")

    from . import search_permission as sp

    try:
        permission = sp.require_search_permission(
            conn,
            str(generation_id),
            expected_origin_planning_event=int(route.planning_event),
            expected_origin_cutoff=str(origin_cutoff),
        expected_snapshot_sha256=str(data_snapshot_sha256),
        )
    except Exception as failure:
        raise FutureStateAssemblyError(f"origin search permission refused: {failure}") from failure
    if permission.get("permitted") is not True or permission.get("reasons") != []:
        raise FutureStateAssemblyError("origin search permission is not permitted")

    artifact_path = Path(str(route.source_artifact_ref))
    try:
        artifact_bytes = artifact_path.read_bytes()
        if hashlib.sha256(artifact_bytes).hexdigest() != str(route.source_artifact_sha256):
            raise ValueError("decision artifact digest changed")
        artifact = json.loads(artifact_bytes.decode("utf-8"))
        packet = ((artifact.get("attribution") or {}).get("manager_packet") or {})
    except Exception as failure:
        raise FutureStateAssemblyError(f"verified manager entry binding is unavailable: {failure}") from failure
    if int(packet.get("entry_id") or 0) != int(entry_id):
        raise FutureStateAssemblyError("explicit manager entry differs from the verified decision packet")

    from . import season_rules as sr

    snapshot_conn = gs._open_generation_snapshot(generation)
    try:
        pinned = sr.resolve_origin_pinned_season_rules(
            snapshot_conn,
            season=str(season),
            cutoff=str(origin_cutoff),
            data_snapshot_sha256=str(data_snapshot_sha256),
        )
    except Exception as failure:
        raise FutureStateAssemblyError(f"origin-pinned season rules are unavailable: {failure}") from failure
    finally:
        snapshot_conn.close()

    state, model = _project_future_manager_state(
        route, target_event=int(target_event), rules=pinned.rules,
    )
    return ProjectedFutureManagerState(
        source_decision_id=str(decision_id),
        source_generation_id=str(generation_id),
        source_route_id=str(route_id),
        source_route_input_sha256=str(route.route_input_sha256),
        origin_event=int(route.planning_event),
        origin_cutoff=str(origin_cutoff),
        data_snapshot_sha256=str(route.data_snapshot_sha256),
        target_event=int(target_event),
        projection_model=model,
        season_rules_evidence=dict(pinned.evidence),
        season_rules=pinned.rules,
        state=state,
    )


def _load_future_request_context(
    conn: Any,
    *,
    decision_id: str,
    generation_id: str,
    route_id: str,
    entry_id: int,
    origin_cutoff: str,
    data_snapshot_sha256: str,
    target_event: int,
    season: str,
    continuation_generation_id: str,
    cache_dir: str | Path,
) -> tuple[Any, Any, Any, Any, Any, dict[int, Any], dict[int, Any], dict[int, int], Any, Any, Any]:
    """Load exact-origin facts and certified continuation worlds, never latest."""

    from dataclasses import fields

    from . import candidate_universe as cu, chip_wildcard as cw
    from . import free_hit_request_adapter as fa, route_optimizer as optimizer
    from . import route_comparator as rc

    projected = load_projected_future_manager_state(
        conn,
        decision_id=decision_id,
        generation_id=generation_id,
        route_id=route_id,
        entry_id=entry_id,
        origin_cutoff=origin_cutoff,
        data_snapshot_sha256=data_snapshot_sha256,
        target_event=target_event,
        season=season,
    )
    route = load_verified_normal_route(
        conn, decision_id, route_id=route_id, require_recommended=False,
    )
    generation = gs.load_generation(conn, generation_id)
    continuation = gs.load_generation(conn, continuation_generation_id)
    report = gs.verify_generation(conn, continuation_generation_id)
    window = tuple(range(int(target_event), int(target_event) + 4))
    if (
        not report.get("verified")
        or continuation.horizon_kind != gs.HORIZON_KIND_CHIP_RESERVATION
        or int(continuation.planning_event) != int(route.planning_event)
        or tuple(int(event) for event in continuation.events[:4]) != tuple(route.events)
        or str(continuation.cutoff) != str(route.cutoff)
        or str(continuation.snapshot.get("sha256") or "") != str(route.data_snapshot_sha256)
        or str(continuation.manifest.get("code_snapshot_sha256") or "")
        != str(generation.manifest.get("code_snapshot_sha256") or "")
        or not set(window).issubset(set(continuation.events))
        or any(continuation.runs_for(event) != generation.runs_for(event) for event in route.events)
    ):
        raise FutureStateAssemblyError(
            "explicit CHIP_RESERVATION continuation does not bind the verified origin and exact future four-event window"
        )

    snapshot_conn = gs._open_generation_snapshot(generation)
    try:
        pool = cu.load_pool(snapshot_conn)
        pool_binding = cw.pool_binding_from_store(snapshot_conn)
        if tuple(sorted(pool)) != tuple(sorted(pool_binding.eligible_ids)):
            raise FutureStateAssemblyError("origin-pinned player rows differ from the accepted official pool")
        player_meta = rc.load_player_meta(snapshot_conn, pool_binding.eligible_ids)
        if set(player_meta) != set(pool_binding.eligible_ids):
            raise FutureStateAssemblyError("origin-pinned player position/club metadata is incomplete")
        prices = fa._canonical_market_prices(snapshot_conn, pool_binding.eligible_ids, str(route.cutoff))
        if set(prices) != set(pool_binding.eligible_ids):
            raise FutureStateAssemblyError("origin-pinned official market prices are incomplete for the player pool")
        artifact_record = gs.load_engine_decision_record(conn, decision_id)
        artifact = json.loads(Path(str(artifact_record["decision_artifact_ref"])).read_text(encoding="utf-8"))
        raw_config = dict((artifact.get("search") or {}).get("config") or {})
        allowed = {field.name for field in fields(ro.OptimizerConfig)}
        raw_config["events"] = window
        config = ro.OptimizerConfig(**{key: value for key, value in raw_config.items() if key in allowed})
        rules = projected.season_rules
        sr = __import__("fpl_brain.season_rules", fromlist=["verify_pinned_season_rules_evidence"])
        sr.verify_pinned_season_rules_evidence(
            projected.season_rules_evidence,
            rules,
            cutoff=str(route.cutoff),
            data_snapshot_sha256=str(route.data_snapshot_sha256),
            allow_fixture=True,
        )
    finally:
        snapshot_conn.close()

    matrix_by_event: dict[int, Any] = {}
    cache_report: list[dict[str, Any]] = []
    if not str(cache_dir or "").strip():
        raise FutureStateAssemblyError("an explicit cache directory is required")
    cache_root = Path(cache_dir).expanduser().resolve()
    for event in window:
        try:
            matrix, evidence = optimizer.build_event_worlds(
                conn,
                continuation,
                event,
                tuple(sorted(pool_binding.eligible_ids)),
                config,
                cache_dir=cache_root,
                allow_materialization=False,
            )
        except Exception as failure:
            raise FutureStateAssemblyError(
                f"certified continuation worlds for GW{event} are unavailable: {failure}"
            ) from failure
        expected_key = optimizer.world_cache_key(
            event=event,
            generation_id=continuation.generation_id,
            runs=continuation.runs_for(event),
            config=config,
            union_ids=tuple(sorted(pool_binding.eligible_ids)),
        )
        if (
            str(matrix.get(optimizer.MANAGER_MATRIX_IDENTITY_KEY) or "") != expected_key
            or tuple(sorted(int(pid) for pid in matrix.get("player_ids") or ()))
            != tuple(sorted(pool_binding.eligible_ids))
        ):
            raise FutureStateAssemblyError(f"GW{event} world matrix is not the exact canonical continuation matrix")
        matrix_by_event[event] = matrix
        cache_report.append({
            "event": event,
            "generation_id": continuation.generation_id,
            "cache_key": expected_key,
            **dict(evidence.get("cache") or {}),
        })
    return projected, route, generation, continuation, pool_binding, pool, player_meta, prices, config, rules, matrix_by_event, cache_report


def _route_record(route: VerifiedNormalRoute) -> Mapping[str, Any]:
    try:
        artifact = json.loads(Path(str(route.source_artifact_ref)).read_text(encoding="utf-8"))
        row = (((artifact.get("finalist_refinement") or {}).get("route_table") or {}).get("routes") or {}).get(
            str(route.route_id)
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as failure:
        raise FutureStateAssemblyError(f"verified normal route policy artifact is unavailable: {failure}") from failure
    if not isinstance(row, Mapping):
        raise FutureStateAssemblyError("verified route policy record is absent from its source decision")
    return row


def _source_identity(route: VerifiedNormalRoute, generation: Any) -> dict[str, Any]:
    return {
        "source_decision_id": str(route.source_decision_id),
        "source_result_sha256": str(route.source_result_sha256),
        "source_artifact_sha256": str(route.source_artifact_sha256),
        "generation_id": str(route.generation_id),
        "planning_event": int(route.planning_event),
        "origin_cutoff": str(route.cutoff),
        "data_snapshot_sha256": str(route.data_snapshot_sha256),
        "predictive_code_snapshot_sha256": str(generation.manifest.get("code_snapshot_sha256") or ""),
        "certification_identity": str(route.certification_identity),
    }


def _verify_chip_window(
    rows: list[Mapping[str, Any]], *, action: str, event: int, expiry_event: int,
) -> Mapping[str, Any]:
    """Use the accepted per-definition selector; never trust caller booleans."""

    from . import chip_decision as cd

    mapped = cd._availability_by_action(rows, planning_event=int(event))
    entry = mapped.get(str(action)) or {}
    if entry.get("eligible") is not True:
        raise FutureStateAssemblyError(f"pinned official chip state does not establish {action} eligibility at GW{event}")
    active, problem = cd._active_definition(entry.get("definitions") or ())
    if problem or not isinstance(active, Mapping):
        raise FutureStateAssemblyError(f"pinned official {action} window is absent or ambiguous: {problem}")
    start, stop = active.get("window_start_event"), active.get("window_stop_event")
    if (
        start is None or stop is None
        or int(start) > int(event) or int(event) > int(stop)
        or int(stop) != int(expiry_event)
        or int(expiry_event) < int(event)
    ):
        raise FutureStateAssemblyError(f"pinned official {action} window does not bind its declared expiry")
    return dict(active)


def _verify_future_coverage(
    conn: Any, *, route: VerifiedNormalRoute, generation: Any, action: str,
    event: int, expiry_event: int, continuation_generation_id: str,
    coverage_product: Mapping[str, Any],
) -> dict[str, Any]:
    from . import chip_reservation_forecast as crf

    source_identity = _source_identity(route, generation)
    try:
        verification = crf.verify_reservation_coverage_product(
            conn, coverage_product,
            expected={
                "action": str(action), "planning_event": int(route.planning_event),
                "expiry_event": int(expiry_event), "source_identity": source_identity,
            },
        )
    except Exception as failure:
        raise FutureStateAssemblyError(f"persisted {action} expiry-coverage product did not verify: {failure}") from failure
    if verification.get("verified") is not True:
        raise FutureStateAssemblyError(f"persisted {action} expiry-coverage product did not verify")
    # The verifier intentionally returns only a compact verification receipt. Keep
    # using the caller's payload for its already-digest-verified contract fields.
    verified = dict(coverage_product)
    if (
        verified.get("coverage_complete") is not True
        or str(verified.get("product_generation_id") or "") != str(continuation_generation_id)
        or int(event) not in {int(value) for value in verified.get("forecast_events") or ()}
        or int(event) > int(expiry_event)
    ):
        raise FutureStateAssemblyError(f"persisted {action} coverage does not authorize GW{event} through expiry")
    return verified


def _load_minutes_projection_rows(conn: Any, run_id: int) -> dict[tuple[int, int], dict[str, Any]]:
    """Load only the certified MINUTES_V1 frozen-prediction family."""

    from . import analytics, candidate_universe as cu

    return cu.load_projection_rows(conn, int(run_id), analytics.MINUTES_V1_KIND)


def _add_future_lineup_policies(
    partial: Any,
    *,
    route: VerifiedNormalRoute,
    source_route_record: Mapping[str, Any],
    worlds_by_event: Mapping[int, Any],
    positions: Mapping[int, str],
    preserve_source_suffix: bool = True,
) -> Any:
    """Keep retained route lineups; rank only explicit no-transfer continuation events."""

    from . import manager_lineup as ml
    from .chip_route_assembly import _policy_from_route

    actions = []
    for action in partial.actions:
        event = int(action["event"])
        transition = action.get("transition")
        state = getattr(transition, "next_event_state", None)
        if state is None:
            raise FutureStateAssemblyError(f"GW{event} projected route has no canonical next state")
        squad = tuple(sorted(int(pid) for pid in state.by_id()))
        if preserve_source_suffix and event in route.events:
            policy = _policy_from_route(source_route_record, event=event)
        else:
            matrix = worlds_by_event[event]
            ranked = ml.rank_policies(
                list(squad), positions, matrix, top_k=1,
            )
            rows = list(ranked.get("top_policies") or ())
            policy = rows[0] if rows else None
        if not isinstance(policy, ml.ManagerPolicy):
            raise FutureStateAssemblyError(f"GW{event} has no canonical retained/ranked lineup policy")
        if set(ml.policy_player_ids(policy)) != set(squad):
            raise FutureStateAssemblyError(f"GW{event} lineup policy differs from its projected squad")
        actions.append({**dict(action), "policy": policy.as_dict()})
    return ro.PartialRoute(
        state=partial.state,
        actions=tuple(actions),
        h1_proxy=partial.h1_proxy,
        window_proxy=partial.window_proxy,
        hits=partial.hits,
    )


def build_future_free_hit_request(
    conn: Any,
    *,
    decision_id: str,
    generation_id: str,
    route_id: str,
    entry_id: int,
    origin_cutoff: str,
    data_snapshot_sha256: str,
    target_event: int,
    season: str,
    continuation_generation_id: str,
    cache_dir: str | Path,
    expiry_event: int,
    coverage_product: Mapping[str, Any],
    input_uncertainty_flags: tuple[str, ...] = (),
) -> tuple[Any, dict[str, Any]]:
    """Compose one typed FH request from explicit origin plus certified future worlds."""

    from dataclasses import replace

    from . import analytics, chip_decision as cd, chip_free_hit as fh
    from . import free_hit_decision_authority as fda
    from . import free_hit_request_adapter as fa
    from . import free_hit_route as fhr
    from . import chip_wildcard as cw
    from . import route_optimizer as optimizer, candidate_universe as cu
    from . import route_comparator as comparator, planning
    from .chip_route_assembly import replay_serialized_route

    target = int(target_event)
    try:
        (
            projected, route, _generation, continuation, pool_binding, pool,
            player_meta, prices, config, rules, matrices, cache_report,
        ) = _load_future_request_context(
            conn,
            decision_id=decision_id,
            generation_id=generation_id,
            route_id=route_id,
            entry_id=entry_id,
            origin_cutoff=origin_cutoff,
            data_snapshot_sha256=data_snapshot_sha256,
            target_event=target,
            season=season,
            continuation_generation_id=continuation_generation_id,
            cache_dir=cache_dir,
        )
        window = tuple(range(target, target + 4))
        verified_coverage = _verify_future_coverage(
            conn, route=route, generation=_generation, action=cd.CHIP_ACTION_FH,
            event=target, expiry_event=int(expiry_event),
            continuation_generation_id=continuation_generation_id,
            coverage_product=coverage_product,
        )
        source_snapshot = gs._open_generation_snapshot(continuation)
        try:
            chip_rows = planning.chips_state(source_snapshot, int(entry_id), target)
        finally:
            source_snapshot.close()
        chip_window = _verify_chip_window(
            chip_rows, action=cd.CHIP_ACTION_FH, event=target, expiry_event=int(expiry_event),
        )
        authority = fh.FreeHitDecisionAuthority.from_verified_continuation_generation(
            conn, continuation_generation_id, events=window,
        )
        bundle = authority.bundle_for(target) or {}
        identity = cw.WildcardPredictiveIdentity(
            cutoff=str(bundle.get("cutoff") or ""),
            data_snapshot_sha256=str(bundle.get("data_snapshot_sha256") or ""),
            source_snapshot_sha256=str(bundle.get("code_snapshot_sha256") or ""),
            generation=fda.runs_label(bundle.get("runs") or {}),
            model_config_identity=fda.model_label(bundle.get("model_versions") or {}),
        )
        h1_matrix = matrices[target]
        h1_worlds = cw.WildcardWorldInputs(
            event=target,
            worlds=int(h1_matrix["worlds"]),
            player_ids=tuple(sorted(pool_binding.eligible_ids)),
            minutes={pid: tuple(float(value) for value in h1_matrix["minutes"][pid]) for pid in pool_binding.eligible_ids},
            core={pid: tuple(float(value) for value in h1_matrix["core"][pid]) for pid in pool_binding.eligible_ids},
            identity=identity,
        )
        binding = cd.ChipHorizonBinding(
            planning_event=target,
            horizon_events=window,
            certification_identity=str(authority.certification_identity),
            data_snapshot_sha256=str(authority.data_snapshot_sha256),
        )
        position_map = {pid: str(row.position) for pid, row in player_meta.items()}
        club_map = {pid: int(row.club_id) for pid, row in player_meta.items()}
        save_partial = _build_future_save_route(
            route,
            target_event=target,
            horizon_length=4,
            rules=rules,
            player_meta=player_meta,
            season_rules_evidence=projected.season_rules_evidence,
        ).partial
        source_record = _route_record(route)
        save_partial = _add_future_lineup_policies(
            save_partial, route=route, source_route_record=source_record,
            worlds_by_event=matrices, positions=position_map,
        )
        permanent = fh.FreeHitPermanentState(
            event=target,
            owned_ids=tuple(sorted(int(player.player_id) for player in projected.state.players)),
            purchase_price_tenths={
                int(player.player_id): int(player.purchase_price_tenths)
                for player in projected.state.players
            },
            bank_tenths=int(projected.state.bank_tenths),
            free_transfers=int(projected.state.free_transfers),
            event_start_free_transfers=int(projected.state.event_start_free_transfers),
            positions=position_map,
            clubs=club_map,
        )
        restored = fh.restore_permanent_state(
            permanent, rules=rules, restored_event=target + 1,
        )
        restore_problems = restored.equals_permanent(permanent)
        if restore_problems:
            raise FutureStateAssemblyError(f"FH permanent-state restoration changed {restore_problems}")
        play_start = ts.RouteState(
            event=int(restored.event),
            players=tuple(
                ts.RoutePlayer(
                    player_id=int(pid), position=position_map[int(pid)], club_id=club_map[int(pid)],
                    purchase_price_tenths=int(restored.purchase_price_tenths[int(pid)]),
                ) for pid in restored.owned_ids
            ),
            bank_tenths=int(restored.bank_tenths),
            free_transfers=int(restored.free_transfers),
            chip_state=tuple(save_partial.actions[0]["transition"].before_state.chip_state),
            event_start_free_transfers=int(restored.free_transfers),
        )
        route_config = replace(config, events=window)
        play_config = replace(config, events=window[1:])
        tail_events = window[1:]
        source_snapshot = gs._open_generation_snapshot(continuation)
        try:
            tail_fixtures = cu.load_fixtures_by_team(source_snapshot, tail_events)
            price_snapshot = cu.price_snapshot_as_of(
                source_snapshot, target, str(route.cutoff),
                required_player_ids=tuple(sorted(pool_binding.eligible_ids)),
            )
        finally:
            source_snapshot.close()
        tail_xpts: dict[int, Any] = {}
        tail_minutes: dict[int, Any] = {}
        for event in tail_events:
            runs = continuation.runs_for(event)
            if "xpts_v1" not in runs or "minutes_v1" not in runs:
                raise FutureStateAssemblyError(f"FH PLAY tail GW{event} omits certified xPts/minutes runs")
            tail_xpts[event] = cu.load_projection_rows(conn, int(runs["xpts_v1"]))
            tail_minutes[event] = _load_minutes_projection_rows(conn, int(runs["minutes_v1"]))
        price_scenario = comparator.flat_current_price_scenario(price_snapshot, tail_events)
        universe = cu.build_universe(
            pool=pool, events_fixtures=tail_fixtures,
            xpts_rows_by_event=tail_xpts, minutes_rows_by_event=tail_minutes,
            events=tail_events, owned_ids=restored.owned_ids,
            price_snapshot=price_snapshot, planning_cutoff=str(route.cutoff),
            run_refs={"events": list(tail_events),
                      "runs": {str(event): continuation.runs_for(event) for event in tail_events}},
        )
        universe["replacement_edges"] = cu.build_replacement_edges(
            universe_rows=universe["universe"], owned_ids=restored.owned_ids,
            state=play_start, price_snapshot=price_snapshot, player_meta=player_meta,
        )
        play_result = optimizer.optimize(
            universe=universe, initial_state=play_start, scenario=price_scenario,
            player_meta=player_meta, generation=continuation, conn=conn,
            config=play_config, cache_dir=Path(cache_dir).expanduser().resolve(),
            allow_world_materialization=False,
            provenance={
                "source_decision_id": route.source_decision_id,
                "source_decision_result_sha256": route.source_result_sha256,
                "source_route_id": route.route_id,
                "free_hit_arm": "PLAY_RESTORED_PERMANENT_H2_H4",
                "generation_id": continuation.generation_id,
                "planning_cutoff": continuation.cutoff,
                "data_snapshot_sha256": continuation.snapshot.get("sha256"),
            },
        )
        best = optimizer.best_route_family(play_result, objective="supported_3gw_net_core")
        if not best or not best.get("route_id"):
            raise FutureStateAssemblyError("canonical FH PLAY tail optimizer retained no route")
        play_route_id = str(best["route_id"])
        play_record = (play_result.get("routes") or {}).get(play_route_id)
        if not isinstance(play_record, Mapping):
            raise FutureStateAssemblyError("canonical FH PLAY tail route is absent from optimizer output")
        play_partial = replay_serialized_route(
            play_record.get("actions") or (), start_state=play_start,
            expected_events=tail_events, price_scenario=price_scenario,
            player_meta=player_meta,
            recorded_terminal={
                "squad_ids": (play_record.get("actions") or [{}])[-1].get("squad_ids") or (),
                "bank_tenths": play_record.get("terminal_bank_tenths"),
                "free_transfers": play_record.get("terminal_ft"),
            },
        )
        play_matrices: dict[int, Any] = {}
        play_union = tuple(sorted(int(pid) for pid in play_result.get("union_players") or ()))
        if not play_union or not set(int(pid) for pid in play_start.by_id()).issubset(set(play_union)):
            raise FutureStateAssemblyError("FH PLAY optimizer returned an incomplete certified capture union")
        for event in tail_events:
            matrix, _evidence = optimizer.build_event_worlds(
                conn, continuation, event, play_union, play_config,
                cache_dir=Path(cache_dir).expanduser().resolve(), allow_materialization=False,
            )
            expected_key = optimizer.world_cache_key(
                event=event, generation_id=continuation.generation_id,
                runs=continuation.runs_for(event), config=play_config, union_ids=play_union,
            )
            if (
                str(matrix.get(optimizer.MANAGER_MATRIX_IDENTITY_KEY) or "") != expected_key
                or tuple(sorted(int(pid) for pid in matrix.get("player_ids") or ())) != play_union
            ):
                raise FutureStateAssemblyError(f"FH PLAY GW{event} is not its exact optimizer cache matrix")
            play_matrices[event] = matrix
        save_route = fhr.free_hit_route_from_canonical_route(
            save_partial,
            arm=fhr.ARM_SAVE,
            expected_events=window,
            rules=rules,
            expected_start_state={
                "event": target,
                "squad_ids": sorted(permanent.owned_ids),
                "bank_tenths": permanent.bank_tenths,
                "free_transfers": permanent.free_transfers,
                "purchase_price_tenths": {str(k): v for k, v in permanent.purchase_price_tenths.items()},
            },
            worlds_by_event={event: matrices[event] for event in window},
            positions_of=lambda squad_ids: {int(pid): position_map[int(pid)] for pid in squad_ids},
            route_config=route_config,
            events=window,
        )
        play_route = fhr.free_hit_route_from_canonical_route(
            play_partial,
            arm=fhr.ARM_PLAY,
            expected_events=window[1:],
            rules=rules,
            expected_start_state={
                "event": target + 1,
                "squad_ids": sorted(int(pid) for pid in restored.owned_ids),
                "bank_tenths": int(restored.bank_tenths),
                "free_transfers": int(restored.free_transfers),
                "purchase_price_tenths": {str(k): int(v) for k, v in restored.purchase_price_tenths.items()},
            },
            worlds_by_event=play_matrices,
            positions_of=lambda squad_ids: {int(pid): position_map[int(pid)] for pid in squad_ids},
            route_config=play_config,
            events=window[1:],
        )
        request = fh.FreeHitRequest(
            permanent=permanent,
            horizon_binding=binding,
            h1_worlds=h1_worlds,
            world_identity=identity,
            positions=position_map,
            clubs=club_map,
            market_price_tenths=prices,
            pool_binding=pool_binding,
            play_route=play_route,
            save_route=save_route,
            decision_authority=authority,
            chip_available=bool(chip_window.get("eligible")),
            rules=rules,
            calibration_status=cd.CALIBRATION_UNCALIBRATED,
            input_uncertainty_flags=tuple(input_uncertainty_flags),
        )
        problems = fh.contract_problems(request)
        if problems:
            raise FutureStateAssemblyError("future FH typed request failed canonical validation: " + "; ".join(problems[:8]))
        return request, {
            "manager_projection": projected.as_dict(),
            "authority": authority.as_dict(),
            "horizon_events": list(window),
            "continuation_generation_id": continuation.generation_id,
            "world_cache_keys": {
                str(event): optimizer.world_cache_key(
                    event=event,
                    generation_id=continuation.generation_id,
                    runs=continuation.runs_for(event),
                    config=config,
                    union_ids=tuple(sorted(pool_binding.eligible_ids)),
                ) for event in window
            },
            "world_cache_report": cache_report,
            "coverage_product_sha256": verified_coverage.get("product_sha256"),
            "chip_window": dict(chip_window),
            "save_route_model": "VERIFIED_NORMAL_ROUTE_SUFFIX_THEN_CANONICAL_NO_TRANSFER_CONTINUATION",
            "play_route_model": "FH_RESTORED_PERMANENT_STATE_THEN_CANONICAL_BOUNDED_OPTIMIZER_H2_H4",
            "play_route_id": play_route_id,
            "play_route_signature": str(best["signature"]),
            "hit_cost_witness": _seal_hit_cost_witness({
                "action": "FH", "event": target,
                "source_identity": _source_identity(route, _generation),
                "continuation_generation_id": str(continuation.generation_id),
                "save_route_actions": optimizer._serialize_route(save_partial),
                "play_route_id": play_route_id,
                "play_route_signature": str(best["signature"]),
                "play_route_record": dict(play_record),
                "play_route_record_sha256": __import__(
                    "fpl_brain.chip_reservation_forecast", fromlist=["canonical_sha256"],
                ).canonical_sha256(play_record),
                "play_start_state": _hit_route_state_payload(play_start),
            }),
            "play_optimizer_world_info": dict(play_result.get("world_info") or {}),
            "evaluator_version": fh.FREE_HIT_EVALUATOR_VERSION,
        }
    except FutureStateAssemblyError:
        raise
    except Exception as failure:
        raise FutureStateAssemblyError(
            f"future FH request composition refused: {type(failure).__name__}: {failure}"
        ) from failure


def build_future_wildcard_request(
    conn: Any,
    *,
    decision_id: str,
    generation_id: str,
    route_id: str,
    entry_id: int,
    origin_cutoff: str,
    data_snapshot_sha256: str,
    target_event: int,
    season: str,
    continuation_generation_id: str,
    value_generation_id: str,
    cache_dir: str | Path,
    expiry_event: int,
    coverage_product: Mapping[str, Any],
    input_uncertainty_flags: tuple[str, ...] = (),
) -> tuple[Any, dict[str, Any]]:
    """Compose one typed WC request with separate four-event and long-value products."""

    from dataclasses import replace
    from . import analytics, candidate_universe as cu, chip_decision as cd, chip_wildcard as wc
    from . import planning, repositories, route_optimizer as optimizer
    from . import wildcard_production as wcp
    from . import wildcard_save_route as wsr

    target = int(target_event)
    try:
        (
            projected, route, _generation, chip_generation, pool_binding, pool,
            player_meta, prices, config, rules, chip_matrices, cache_report,
        ) = _load_future_request_context(
            conn,
            decision_id=decision_id,
            generation_id=generation_id,
            route_id=route_id,
            entry_id=entry_id,
            origin_cutoff=origin_cutoff,
            data_snapshot_sha256=data_snapshot_sha256,
            target_event=target,
            season=season,
            continuation_generation_id=continuation_generation_id,
            cache_dir=cache_dir,
        )
        verified_coverage = _verify_future_coverage(
            conn, route=route, generation=_generation, action=cd.CHIP_ACTION_WC,
            event=target, expiry_event=int(expiry_event),
            continuation_generation_id=continuation_generation_id,
            coverage_product=coverage_product,
        )
        value_generation = gs.load_generation(conn, value_generation_id)
        value_report = gs.verify_generation(conn, value_generation_id)
        if (
            not value_report.get("verified")
            or value_generation.horizon_kind != gs.HORIZON_KIND_WILDCARD_VALUE
            or int(value_generation.planning_event) != target
            or str(value_generation.cutoff) != str(chip_generation.cutoff)
            or str(value_generation.snapshot.get("sha256") or "") != str(chip_generation.snapshot.get("sha256") or "")
            or str(value_generation.manifest.get("code_snapshot_sha256") or "")
            != str(chip_generation.manifest.get("code_snapshot_sha256") or "")
            or tuple(value_generation.events[:4]) != tuple(range(target, target + 4))
            or any(value_generation.runs_for(event) != chip_generation.runs_for(event) for event in value_generation.events[:4])
            or not set(value_generation.events).issubset(set(chip_generation.events))
            or any(value_generation.runs_for(event) != chip_generation.runs_for(event) for event in value_generation.events)
        ):
            raise FutureStateAssemblyError("explicit WILDCARD_VALUE product is not the compatible same-origin continuation")
        value_events = tuple(int(event) for event in value_generation.events)
        if not 6 <= len(value_events) <= 10 or value_events != tuple(range(target, target + len(value_events))):
            raise FutureStateAssemblyError("WILDCARD_VALUE product must provide its exact contiguous 6–10 event horizon")
        if not set(value_events).issubset({int(value) for value in verified_coverage.get("product_events") or ()}):
            raise FutureStateAssemblyError("verified Wildcard coverage omits part of the full 6–10 event value horizon")
        if str(verified_coverage.get("product_generation_id") or "") != str(continuation_generation_id):
            raise FutureStateAssemblyError("verified Wildcard coverage and explicit CHIP_RESERVATION generation differ")
        chip_per_event_identity = chip_generation.manifest.get("per_event") or {}
        value_per_event_identity = value_generation.manifest.get("per_event") or {}
        if any(
            (value_per_event_identity.get(str(event)) or {}).get("dependency_closure")
            != (chip_per_event_identity.get(str(event)) or {}).get("dependency_closure")
            for event in value_events
        ):
            raise FutureStateAssemblyError("WILDCARD_VALUE per-event dependency closure differs from CHIP_RESERVATION")

        snapshot_conn = gs._open_generation_snapshot(chip_generation)
        try:
            from . import season_rules as sr
            official = repositories.latest_accepted_bootstrap_generation(snapshot_conn)
            chip_rows = planning.chips_state(snapshot_conn, int(entry_id), target)
            fixtures = cu.load_fixtures_by_team(snapshot_conn, value_events)
            value_xpts: dict[int, Any] = {}
            value_minutes: dict[int, Any] = {}
            for event in value_events:
                runs = value_generation.runs_for(event)
                if "xpts_v1" not in runs or "minutes_v1" not in runs:
                    raise FutureStateAssemblyError(f"WILDCARD_VALUE GW{event} omits xPts/minutes runs")
                value_xpts[event] = cu.load_projection_rows(conn, int(runs["xpts_v1"]))
                value_minutes[event] = _load_minutes_projection_rows(conn, int(runs["minutes_v1"]))
        finally:
            snapshot_conn.close()
        if not official:
            raise FutureStateAssemblyError("pinned snapshot has no accepted official player pool")
        chip_window = _verify_chip_window(
            chip_rows, action=cd.CHIP_ACTION_WC, event=target, expiry_event=int(expiry_event),
        )

        raw_config = dict((json.loads(Path(str(gs.load_engine_decision_record(conn, decision_id)["decision_artifact_ref"])).read_text(encoding="utf-8")).get("search") or {}).get("config") or {})
        allowed = {field.name for field in __import__("dataclasses").fields(ro.OptimizerConfig)}
        raw_config["events"] = value_events
        value_config = ro.OptimizerConfig(**{key: value for key, value in raw_config.items() if key in allowed})
        chip_config = replace(value_config, events=tuple(range(target, target + 4)))
        identity_config = {
            "generation_id": value_generation.generation_id,
            "events": list(value_events),
            "runs_by_event": {str(event): value_generation.runs_for(event) for event in value_events},
            "world_config": value_config.as_dict(),
        }
        model_config_identity = analytics.canonical_hash(identity_config)
        identity = wc.WildcardPredictiveIdentity(
            cutoff=str(value_generation.cutoff),
            data_snapshot_sha256=str(value_generation.snapshot.get("sha256") or ""),
            source_snapshot_sha256=str(value_generation.manifest.get("code_snapshot_sha256") or ""),
            generation=str(value_generation.generation_id),
            model_config_identity=model_config_identity,
        )
        players = wcp._players_from_certified_rows(
            pool=pool,
            prices=prices,
            events=value_events,
            fixtures=fixtures,
            xpts_by_event=value_xpts,
            minutes_by_event=value_minutes,
            identity=identity,
        )
        horizon = wcp.wildcard_horizon(
            target,
            length=len(value_events),
            last_event=int(value_generation.manifest.get("last_event") or value_events[-1]),
        )
        if tuple(horizon.events) != value_events:
            raise FutureStateAssemblyError("canonical Wildcard horizon differs from certified value events")
        value_binding = wcp.value_horizon_binding(
            target,
            horizon,
            decision_cutoff=str(value_generation.cutoff),
            data_snapshot_sha256=str(value_generation.snapshot.get("sha256") or ""),
            source_snapshot_sha256=str(value_generation.manifest.get("code_snapshot_sha256") or ""),
            prediction_generation=str(value_generation.generation_id),
            model_config_identity=model_config_identity,
        )
        chip_per_event = chip_per_event_identity
        chip_events = tuple(range(target, target + 4))
        missing_chip_bundles = [
            event for event in chip_events
            if not str((chip_per_event.get(str(event)) or {}).get("bundle_identity") or "")
        ]
        if missing_chip_bundles:
            raise FutureStateAssemblyError(
                "CHIP_RESERVATION continuation omits certified bundles for the declared "
                f"four-event WC window: {missing_chip_bundles}"
            )
        certification_identity = gs._certification_identity_for_bundles(
            cutoff=str(chip_generation.cutoff),
            bundle_identities={
                str(event): str((chip_per_event.get(str(event)) or {}).get("bundle_identity") or "")
                for event in chip_events
            },
            snapshot_sha256=str(chip_generation.snapshot.get("sha256") or ""),
        )
        chip_binding = cd.ChipHorizonBinding(
            planning_event=target,
            horizon_events=chip_events,
            certification_identity=certification_identity,
            data_snapshot_sha256=str(chip_generation.snapshot.get("sha256") or ""),
        )
        value_matrices: dict[int, Any] = {}
        value_worlds: dict[int, Any] = {}
        value_cache_report: list[dict[str, Any]] = []
        for event in value_events:
            matrix, cache_evidence = optimizer.build_event_worlds(
                conn, value_generation, event, tuple(sorted(pool_binding.eligible_ids)), value_config,
                cache_dir=Path(cache_dir).expanduser().resolve(),
                allow_materialization=False,
            )
            expected_key = optimizer.world_cache_key(
                event=event, generation_id=value_generation.generation_id,
                runs=value_generation.runs_for(event), config=value_config,
                union_ids=tuple(sorted(pool_binding.eligible_ids)),
            )
            if str(matrix.get(optimizer.MANAGER_MATRIX_IDENTITY_KEY) or "") != expected_key:
                raise FutureStateAssemblyError(f"WILDCARD_VALUE GW{event} is not its canonical certified world")
            if tuple(sorted(int(pid) for pid in matrix.get("player_ids") or ())) != tuple(sorted(pool_binding.eligible_ids)):
                raise FutureStateAssemblyError(f"WILDCARD_VALUE GW{event} world has a different official player pool")
            value_matrices[event] = matrix
            value_cache_report.append({
                "event": event,
                "generation_id": value_generation.generation_id,
                "cache_key": expected_key,
                **dict(cache_evidence.get("cache") or {}),
            })
            value_worlds[event] = wc.WildcardWorldInputs(
                event=event,
                worlds=int(matrix["worlds"]),
                player_ids=tuple(sorted(pool_binding.eligible_ids)),
                minutes={pid: tuple(float(value) for value in matrix["minutes"][pid]) for pid in pool_binding.eligible_ids},
                core={pid: tuple(float(value) for value in matrix["core"][pid]) for pid in pool_binding.eligible_ids},
                identity=identity,
            )
        save_route_projection = _build_future_save_route(
            route,
            target_event=target,
            horizon_length=4,
            rules=rules,
            player_meta=player_meta,
            season_rules_evidence=projected.season_rules_evidence,
        ).partial
        save_actions = tuple(save_route_projection.actions or ())
        if not save_actions:
            raise FutureStateAssemblyError("canonical WC SAVE route has no target-event transition")
        save_first_transition = save_actions[0].get("transition")
        save_start_state = getattr(save_first_transition, "before_state", None)
        if save_start_state is None or save_start_state != projected.state:
            raise FutureStateAssemblyError("canonical WC SAVE route does not start from the verified target state")
        save_start_ft = getattr(save_start_state, "event_start_free_transfers", None)
        if type(save_start_ft) is not int:
            raise FutureStateAssemblyError("canonical WC SAVE route start omits proven event-start free transfers")
        save_route_projection = _add_future_lineup_policies(
            save_route_projection, route=route, source_route_record=_route_record(route),
            worlds_by_event=chip_matrices,
            positions={pid: str(row.position) for pid, row in player_meta.items()},
        )
        save_route = wsr.wildcard_save_route_from_canonical_route(
            save_route_projection,
            planning_event=target,
            horizon=horizon,
            rules=rules,
            expected_horizon=tuple(range(target, target + 4)),
            worlds_by_event={event: chip_matrices[event] for event in range(target, target + 4)},
            positions_of=lambda squad_ids: {
                int(pid): str(player_meta[int(pid)].position) for pid in squad_ids
            },
            config=chip_config,
            events=tuple(range(target, target + 4)),
        )
        state = projected.state
        basis = {int(player.player_id): int(player.purchase_price_tenths) for player in state.players}
        selling = {
            int(pid): int(ts.selling_price_tenths(int(basis[int(pid)]), int(prices[int(pid)])))
            for pid in basis
        }
        request = wc.WildcardRequest(
            planning_event=target,
            horizon=horizon,
            players=dict(players),
            positions={int(pid): str(player.position) for pid, player in players.items()},
            owned_ids=tuple(sorted(basis)),
            purchase_price_tenths=basis,
            selling_price_tenths=selling,
            bank_tenths=int(state.bank_tenths),
            rules=rules,
            horizon_binding=chip_binding,
            certification_identity=str(chip_binding.certification_identity),
            data_snapshot_sha256=str(chip_generation.snapshot.get("sha256") or ""),
            event_start_free_transfers=int(save_start_ft),
            save_route=save_route,
            chip_availability=tuple(dict(row) for row in chip_rows),
            value_horizon_binding=value_binding,
            worlds_by_event=dict(value_worlds),
            pool_binding=pool_binding,
        )
        problems = wc.validate_projections(request)
        if problems:
            raise FutureStateAssemblyError("future WC typed request failed canonical validation: " + "; ".join(problems[:8]))
        return request, {
            "manager_projection": projected.as_dict(),
            "chip_generation_id": chip_generation.generation_id,
            "value_generation_id": value_generation.generation_id,
            "horizon_events": list(value_events),
            "chip_events": list(chip_binding.horizon_events),
            "world_cache_keys": {
                str(event): optimizer.world_cache_key(
                    event=event, generation_id=value_generation.generation_id,
                    runs=value_generation.runs_for(event), config=value_config,
                    union_ids=tuple(sorted(pool_binding.eligible_ids)),
                ) for event in value_events
            },
            "world_cache_report": [*cache_report, *value_cache_report],
            "coverage_product_sha256": verified_coverage.get("product_sha256"),
            "chip_window": dict(chip_window),
            "save_route_model": "VERIFIED_NORMAL_ROUTE_SUFFIX_THEN_CANONICAL_NO_TRANSFER_CONTINUATION",
            "hit_cost_witness": _seal_hit_cost_witness({
                "action": "WC", "event": target,
                "source_identity": _source_identity(route, _generation),
                "continuation_generation_id": str(chip_generation.generation_id),
                "value_generation_id": str(value_generation.generation_id),
                "save_route_actions": optimizer._serialize_route(save_route_projection),
            }),
            "save_route_start_event_state": {
                "event": int(save_start_state.event),
                "free_transfers": int(save_start_state.free_transfers),
                "event_start_free_transfers": int(save_start_ft),
            },
            "save_route_target_event_dto_free_transfers": int(
                next(row for row in save_route.events if int(row.event) == target).free_transfers
            ),
            "evaluator_version": wc.WILDCARD_EVALUATOR_VERSION,
            "input_uncertainty_flags": list(input_uncertainty_flags),
        }
    except FutureStateAssemblyError:
        raise
    except Exception as failure:
        raise FutureStateAssemblyError(
            f"future WC request composition refused: {type(failure).__name__}: {failure}"
        ) from failure


HIT_COST_WITNESS_SCHEMA = "fpl_brain.chip_horizon_hit_cost_witness.v1"


def _seal_hit_cost_witness(value: Mapping[str, Any]) -> dict[str, Any]:
    from . import chip_reservation_forecast as crf

    body = dict(value)
    body["schema"] = HIT_COST_WITNESS_SCHEMA
    body["witness_sha256"] = crf.canonical_sha256(body)
    return body


def _hit_route_state_payload(state: Any) -> dict[str, Any]:
    return {
        "event": int(state.event),
        "squad_ids": sorted(int(pid) for pid in state.by_id()),
        "bank_tenths": int(state.bank_tenths),
        "purchase_price_tenths": {
            str(int(player.player_id)): int(player.purchase_price_tenths)
            for player in state.players
        },
        "free_transfers": int(state.free_transfers),
        "event_start_free_transfers": (
            None if state.event_start_free_transfers is None
            else int(state.event_start_free_transfers)
        ),
    }


def _hit_int(value: Any, *, where: str) -> int:
    if type(value) is not int:
        raise FutureStateAssemblyError(f"{where} must be an exact integer")
    return value


def _require_zero_hit_arms(
    action: str, *, expected_events: tuple[int, ...], arms: Mapping[str, Any],
    defer_nonzero_hits: bool = False,
) -> tuple[str, ...]:
    if set(arms) != {"PLAY", "SAVE"}:
        raise FutureStateAssemblyError(f"{action} witness must verify both PLAY and SAVE")
    nonzero_hits = []
    for role in ("PLAY", "SAVE"):
        rows = arms[role]
        events = tuple(_hit_int(row.get("event"), where=f"{action} {role} event") for row in rows)
        if events != expected_events:
            raise FutureStateAssemblyError(
                f"{action} {role} hit witness does not cover the exact scored event horizon"
            )
        for row in rows:
            points = _hit_int(row.get("hit_points"), where=f"{action} {role} GW{row['event']} hit_points")
            if points != 0:
                message = (
                    f"{action} {role} GW{row['event']} has source-replayed transfer hit cost {points}; "
                    "gross-only realized scoring is unsupported"
                )
                if not defer_nonzero_hits:
                    raise FutureStateAssemblyError(message)
                nonzero_hits.append(message)
    return tuple(nonzero_hits)


def _hit_schedule(
    action: str, role: str, arm: Mapping[str, Any], *,
    expected_events: tuple[int, ...], expected_weights: tuple[float, ...],
) -> list[Mapping[str, Any]]:
    import math

    schedule = arm.get("valuation_schedule")
    rows = schedule.get("events") if isinstance(schedule, Mapping) else None
    if not isinstance(rows, list) or len(rows) != len(expected_events):
        raise FutureStateAssemblyError(f"{action} {role} omits its complete valuation schedule")
    for index, (row, event, weight) in enumerate(zip(rows, expected_events, expected_weights)):
        if not isinstance(row, Mapping):
            raise FutureStateAssemblyError(f"{action} {role} schedule row {index} is malformed")
        if _hit_int(row.get("event"), where=f"{action} {role} schedule event") != event:
            raise FutureStateAssemblyError(f"{action} {role} schedule event set/order differs from its certified horizon")
        actual_weight = row.get("weight")
        if type(actual_weight) not in (int, float) or not math.isfinite(float(actual_weight)):
            raise FutureStateAssemblyError(f"{action} {role} schedule weight is not finite numeric data")
        if abs(float(actual_weight) - float(weight)) > 1e-9:
            raise FutureStateAssemblyError(f"{action} {role} schedule weight differs from its pinned horizon")
    return rows


def _check_hit_schedule_state(
    action: str, role: str, row: Mapping[str, Any], state: Any, *,
    free_transfers: int, event_start_free_transfers: int | None = None,
) -> None:
    supplied = row.get("transfer_state")
    if not isinstance(supplied, Mapping):
        raise FutureStateAssemblyError(f"{action} {role} GW{row.get('event')} omits transfer state")
    expected = _hit_route_state_payload(state)
    for field in ("squad_ids", "bank_tenths", "purchase_price_tenths"):
        if supplied.get(field) != expected[field]:
            raise FutureStateAssemblyError(
                f"{action} {role} GW{row.get('event')} transfer state differs on {field}"
            )
    if _hit_int(supplied.get("free_transfers"), where=f"{action} {role} transfer-state FT") != int(free_transfers):
        raise FutureStateAssemblyError(f"{action} {role} GW{row.get('event')} FT boundary differs")
    if event_start_free_transfers is not None:
        if _hit_int(
            supplied.get("event_start_free_transfers"),
            where=f"{action} {role} event-start FT",
        ) != int(event_start_free_transfers):
            raise FutureStateAssemblyError(f"{action} {role} GW{row.get('event')} event-start FT differs")


def _strict_serialized_route(actions: Any, *, action: str, role: str) -> None:
    if not isinstance(actions, list) or not actions:
        raise FutureStateAssemblyError(f"{action} {role} source route actions are absent")
    for row in actions:
        if not isinstance(row, Mapping):
            raise FutureStateAssemblyError(f"{action} {role} source route action is malformed")
        for field in ("event", "hit_points", "bank_after", "ft_after"):
            _hit_int(row.get(field), where=f"{action} {role} route {field}")
        if not isinstance(row.get("kind"), str):
            raise FutureStateAssemblyError(f"{action} {role} route kind is absent")
        squad = row.get("squad_ids")
        if not isinstance(squad, list) or any(type(pid) is not int for pid in squad):
            raise FutureStateAssemblyError(f"{action} {role} route squad IDs are not exact integers")
        transfers = row.get("transfers")
        if not isinstance(transfers, list):
            raise FutureStateAssemblyError(f"{action} {role} route transfer list is malformed")
        for move in transfers:
            if not isinstance(move, Mapping):
                raise FutureStateAssemblyError(f"{action} {role} route transfer is malformed")
            _hit_int(move.get("out"), where=f"{action} {role} transfer out")
            _hit_int(move.get("in"), where=f"{action} {role} transfer in")


def verify_future_action_hit_cost_witnesses(
    conn: Any, *,
    identity: Mapping[str, Any],
    action: str,
    continuation_generation_id: str,
    value_generation_ids: Mapping[int, str],
    expected_events: tuple[int, ...],
    assembly_reports: Any,
    opportunity_records: Any,
) -> str:
    """Source-replay every FH/WC schedule and return its content digest.

    This verifier reads only pinned origin/continuation material. It never
    loads world matrices, scores an evaluator, runs an optimizer or writes.
    """
    from types import SimpleNamespace

    from . import candidate_universe as cu, chip_decision as cd
    from . import chip_free_hit as fh, chip_reservation_forecast as crf
    from . import chip_wildcard as wc, free_hit_request_adapter as fa
    from . import generation_store as gs, route_comparator as rc
    from . import season_rules as sr, wildcard_production as wcp
    from .chip_route_assembly import load_verified_normal_route, replay_serialized_route

    if action not in {cd.CHIP_ACTION_FH, cd.CHIP_ACTION_WC}:
        raise FutureStateAssemblyError("zero-hit source replay is restricted to FH/WC")
    if not expected_events or any(type(event) is not int for event in expected_events):
        raise FutureStateAssemblyError("FH/WC opportunity event set is empty or not exact integers")
    reports = {}
    for report in assembly_reports or ():
        event = _hit_int(report.get("event"), where="assembly report event")
        if event in reports:
            raise FutureStateAssemblyError(f"duplicate FH/WC assembly report for GW{event}")
        reports[event] = report
    records = {}
    for record in opportunity_records or ():
        event = _hit_int(record.get("event"), where="opportunity event")
        if event in records:
            raise FutureStateAssemblyError(f"duplicate FH/WC opportunity record for GW{event}")
        records[event] = record
    if tuple(sorted(reports)) != expected_events or tuple(sorted(records)) != expected_events:
        raise FutureStateAssemblyError("FH/WC zero-hit proof does not cover every opportunity through expiry")

    proof_rows = []
    source_hit_failures = []
    for event in expected_events:
        report, record = reports[event], records[event]
        witness = report.get("hit_cost_witness")
        if not isinstance(witness, Mapping) or witness.get("schema") != HIT_COST_WITNESS_SCHEMA:
            raise FutureStateAssemblyError(f"GW{event} FH/WC hit-cost witness is missing or unsupported")
        witness_body = dict(witness)
        witness_digest = str(witness_body.pop("witness_sha256", ""))
        if witness_digest != crf.canonical_sha256(witness_body):
            raise FutureStateAssemblyError(f"GW{event} FH/WC hit-cost witness digest does not verify")
        if (
            witness.get("action") != action
            or _hit_int(witness.get("event"), where="witness event") != event
            or str(witness.get("continuation_generation_id") or "") != str(continuation_generation_id)
        ):
            raise FutureStateAssemblyError(f"GW{event} hit-cost witness differs from its explicit request")

        decision_id = str(identity["decision_id"])
        generation_id = str(identity["generation_id"])
        route_id = str(identity["route_id"])
        generation = gs.load_generation(conn, generation_id)
        generation_check = gs.verify_generation(conn, generation_id)
        if generation_check.get("verified") is not True:
            raise FutureStateAssemblyError("pinned normal generation no longer verifies")
        route = load_verified_normal_route(
            conn, decision_id, route_id=route_id, require_recommended=False,
        )
        source_identity = _source_identity(route, generation)
        if record.get("source_identity") != source_identity or witness.get("source_identity") != source_identity:
            raise FutureStateAssemblyError(f"GW{event} witness is not bound to the verified source route")
        if str(route.generation_id) != generation_id or str(route.cutoff) != str(identity["cutoff"]):
            raise FutureStateAssemblyError("source route differs from the explicit generation/cutoff")

        decision = gs.load_engine_decision_record(conn, decision_id)
        artifact = json.loads(Path(str(decision["decision_artifact_ref"])).read_text(encoding="utf-8"))
        packet = (artifact.get("attribution") or {}).get("manager_packet") or {}
        season = str(packet.get("season") or "")
        if not season:
            raise FutureStateAssemblyError("verified source decision does not retain its season")
        projected = load_projected_future_manager_state(
            conn, decision_id=decision_id, generation_id=generation_id, route_id=route_id,
            entry_id=int(identity["entry_id"]), origin_cutoff=str(identity["cutoff"]),
            data_snapshot_sha256=str(route.data_snapshot_sha256), target_event=event, season=season,
        )
        rules = projected.season_rules
        _assert_transfer_rules_compatible(rules)
        continuation = gs.load_generation(conn, str(continuation_generation_id))
        continuation_check = gs.verify_generation(conn, str(continuation_generation_id))
        needed_route_events = tuple(range(event, event + 4))
        if (
            continuation_check.get("verified") is not True
            or continuation.horizon_kind != gs.HORIZON_KIND_CHIP_RESERVATION
            or int(continuation.planning_event) != int(route.planning_event)
            or str(continuation.cutoff) != str(route.cutoff)
            or str(continuation.snapshot.get("sha256") or "") != str(route.data_snapshot_sha256)
            or tuple(continuation.events[:4]) != tuple(route.events)
            or any(continuation.runs_for(e) != generation.runs_for(e) for e in route.events)
            or not set(needed_route_events).issubset(set(continuation.events))
        ):
            raise FutureStateAssemblyError(f"GW{event} continuation is not the pinned same-origin route product")

        origin_snapshot = gs._open_generation_snapshot(generation)
        try:
            pool_binding = wc.pool_binding_from_store(origin_snapshot)
            eligible_ids = tuple(sorted(int(pid) for pid in pool_binding.eligible_ids))
            player_meta = rc.load_player_meta(origin_snapshot, eligible_ids)
            prices = fa._canonical_market_prices(origin_snapshot, eligible_ids, str(route.cutoff))
        finally:
            origin_snapshot.close()
        if set(player_meta) != set(eligible_ids) or set(prices) != set(eligible_ids):
            raise FutureStateAssemblyError("pinned official player metadata/prices are incomplete")
        save_projection = _build_future_save_route(
            route, target_event=event, horizon_length=4, rules=rules,
            player_meta=player_meta, season_rules_evidence=projected.season_rules_evidence,
        ).partial
        save_rows = ro._serialize_route(save_projection)
        if witness.get("save_route_actions") != save_rows:
            raise FutureStateAssemblyError(f"GW{event} SAVE route witness differs from canonical source replay")
        if len(save_projection.actions) != 4:
            raise FutureStateAssemblyError(f"GW{event} SAVE route does not cover its exact four-event prefix")
        save_hits = [
            {"event": _hit_int(row.get("event"), where="SAVE route event"),
             "hit_points": _hit_int(row["transition"].hit_points, where="SAVE replay hit_points")}
            for row in save_projection.actions
        ]
        arms = record.get("outcome_arms")
        if not isinstance(arms, Mapping) or set(arms) != {"play", "save"}:
            raise FutureStateAssemblyError(f"GW{event} FH/WC opportunity omits paired outcome arms")

        if action == cd.CHIP_ACTION_FH:
            scored_events = tuple(range(event, event + 4))
            weights = tuple(1.0 / 4.0 for _ in scored_events)
            save_schedule = _hit_schedule(action, "SAVE", arms["save"],
                                          expected_events=scored_events, expected_weights=weights)
            play_schedule = _hit_schedule(action, "PLAY", arms["play"],
                                          expected_events=scored_events, expected_weights=weights)
            for row, replayed in zip(save_schedule, save_projection.actions):
                _check_hit_schedule_state(action, "SAVE", row, replayed["transition"].next_event_state,
                                          free_transfers=replayed["transition"].next_event_state.free_transfers)
            permanent = fh.FreeHitPermanentState(
                event=event, owned_ids=tuple(sorted(projected.state.by_id())),
                purchase_price_tenths={p.player_id: p.purchase_price_tenths for p in projected.state.players},
                bank_tenths=projected.state.bank_tenths, free_transfers=projected.state.free_transfers,
                event_start_free_transfers=projected.state.event_start_free_transfers,
                positions={pid: route.positions[pid] for pid in projected.state.by_id()},
                clubs={pid: route.clubs[pid] for pid in projected.state.by_id()},
            )
            if permanent.event_start_free_transfers is None:
                raise FutureStateAssemblyError("FH source state has unknown event-start free transfers")
            temporary_ids = tuple(sorted(_hit_int(pid, where="FH temporary squad ID")
                                          for pid in play_schedule[0].get("proposed_squad_ids") or ()))
            transaction = fh.free_hit_transaction_plan(
                SimpleNamespace(permanent=permanent, market_price_tenths=prices), temporary_ids,
            )
            play_semantics = arms["play"].get("action_semantics") or {}
            save_semantics = arms["save"].get("action_semantics") or {}
            restore = play_semantics.get("restore_at_h2") or {}
            if (
                play_semantics.get("role") != "PLAY"
                or _hit_int(restore.get("restore_event"), where="FH restore event") != event + 1
                or restore.get("origin_manager_state_sha256")
                != save_semantics.get("origin_manager_state_sha256")
                or _hit_int(
                    restore.get("event_start_h1_free_transfers"),
                    where="FH event-start free transfers",
                ) != int(permanent.event_start_free_transfers)
                or play_schedule[0].get("state_source") != "TEMPORARY_FREE_HIT_SQUAD_FOR_ONE_EVENT"
                or tuple(temporary_ids) != tuple(sorted(_hit_int(
                    pid, where="FH temporary schedule squad ID"
                ) for pid in play_schedule[0].get("proposed_squad_ids") or ()))
            ):
                raise FutureStateAssemblyError(
                    "FH H1 is not bound to accepted active-chip/restoration semantics"
                )
            temporary_basis = {
                str(pid): int(permanent.purchase_price_tenths[pid])
                for pid in transaction.retained_ids
            }
            temporary_basis.update({str(pid): int(price) for pid, price in transaction.new_basis.items()})
            temporary_state = SimpleNamespace(
                event=event,
                players=tuple(ts.RoutePlayer(
                    pid, player_meta[pid].position, player_meta[pid].club_id,
                    int(temporary_basis[str(pid)]),
                ) for pid in temporary_ids),
                by_id=lambda: {pid: None for pid in temporary_ids},
                bank_tenths=transaction.remaining_bank_tenths,
                free_transfers=permanent.free_transfers,
                event_start_free_transfers=permanent.event_start_free_transfers,
            )
            temporary_market_state = SimpleNamespace(
                event=event,
                players=tuple(ts.RoutePlayer(
                    pid, player_meta[pid].position, player_meta[pid].club_id,
                    int(prices[pid]),
                ) for pid in temporary_ids),
                by_id=lambda: {pid: None for pid in temporary_ids},
                bank_tenths=transaction.remaining_bank_tenths,
                free_transfers=permanent.free_transfers,
                event_start_free_transfers=permanent.event_start_free_transfers,
            )
            _check_hit_schedule_state(
                action, "PLAY", play_schedule[0], temporary_state,
                free_transfers=permanent.free_transfers,
            )
            _check_hit_schedule_state(
                action, "PLAY",
                {"event": event, "transfer_state": play_semantics.get("transfer_state")},
                temporary_market_state, free_transfers=permanent.free_transfers,
                event_start_free_transfers=permanent.event_start_free_transfers,
            )
            restored = fh.restore_permanent_state(permanent, rules=rules, restored_event=event + 1)
            first_save_before = save_projection.actions[0]["transition"].before_state
            play_start = ts.RouteState(
                event=restored.event,
                players=tuple(ts.RoutePlayer(
                    player_id=pid, position=player_meta[pid].position, club_id=player_meta[pid].club_id,
                    purchase_price_tenths=int(restored.purchase_price_tenths[pid]),
                ) for pid in restored.owned_ids),
                bank_tenths=restored.bank_tenths, free_transfers=restored.free_transfers,
                chip_state=tuple(first_save_before.chip_state),
                event_start_free_transfers=restored.free_transfers,
            )
            if witness.get("play_start_state") != _hit_route_state_payload(play_start):
                raise FutureStateAssemblyError("FH selected PLAY route start differs from canonical H2 restoration")
            play_record = witness.get("play_route_record")
            if (
                not isinstance(play_record, Mapping)
                or str(witness.get("play_route_id") or "") != str(report.get("play_route_id") or "")
                or crf.canonical_sha256(play_record) != str(witness.get("play_route_record_sha256") or "")
            ):
                raise FutureStateAssemblyError("FH witness substituted or changed the originally selected optimizer route")
            raw_actions = play_record.get("actions")
            _strict_serialized_route(raw_actions, action=action, role="PLAY")
            selected_signature = str(report.get("play_route_signature") or "")
            if (
                not selected_signature
                or str(witness.get("play_route_signature") or "") != selected_signature
                or str(play_record.get("route_id") or "") != str(witness.get("play_route_id") or "")
                or str(play_record.get("canonical_family_signature") or "") != selected_signature
                or ro.canonical_signature_from_actions(raw_actions) != selected_signature
            ):
                raise FutureStateAssemblyError(
                    "FH PLAY serialized actions differ from the originally selected optimizer family identity"
                )
            tail_events = tuple(range(event + 1, event + 4))
            source_snapshot = gs._open_generation_snapshot(continuation)
            try:
                base_prices = cu.price_snapshot_as_of(
                    source_snapshot, event, str(route.cutoff), required_player_ids=eligible_ids,
                )
            finally:
                source_snapshot.close()
            scenario = rc.flat_current_price_scenario(base_prices, tail_events)
            play_partial = replay_serialized_route(
                raw_actions, start_state=play_start, expected_events=tail_events,
                price_scenario=scenario, player_meta=player_meta,
                recorded_terminal={
                    "squad_ids": raw_actions[-1].get("squad_ids") or (),
                    "bank_tenths": play_record.get("terminal_bank_tenths"),
                    "free_transfers": play_record.get("terminal_ft"),
                },
            )
            for row, replayed in zip(play_schedule[1:], play_partial.actions):
                _check_hit_schedule_state(action, "PLAY", row, replayed["transition"].next_event_state,
                                          free_transfers=replayed["transition"].next_event_state.free_transfers)
            play_hits = [{"event": event, "hit_points": 0}] + [
                {"event": _hit_int(row.get("event"), where="FH PLAY event"),
                 "hit_points": _hit_int(row["transition"].hit_points, where="FH PLAY replay hit_points")}
                for row in play_partial.actions
            ]
        else:
            value_id = str(value_generation_ids.get(event) or "")
            if value_id != str(witness.get("value_generation_id") or ""):
                raise FutureStateAssemblyError(f"GW{event} WC witness differs from its explicit value generation")
            value_generation = gs.load_generation(conn, value_id)
            value_check = gs.verify_generation(conn, value_id)
            scored_events = tuple(int(v) for v in value_generation.events)
            if (
                value_check.get("verified") is not True
                or value_generation.horizon_kind != gs.HORIZON_KIND_WILDCARD_VALUE
                or int(value_generation.planning_event) != event
                or not 6 <= len(scored_events) <= 10
                or scored_events != tuple(range(event, event + len(scored_events)))
                or not set(scored_events).issubset(set(continuation.events))
                or str(value_generation.cutoff) != str(route.cutoff)
                or str(value_generation.snapshot.get("sha256") or "") != str(route.data_snapshot_sha256)
                or any(value_generation.runs_for(e) != continuation.runs_for(e) for e in scored_events)
            ):
                raise FutureStateAssemblyError(f"GW{event} WC value horizon is not the exact certified continuation")
            horizon = wcp.wildcard_horizon(
                event, length=len(scored_events),
                last_event=int(value_generation.manifest.get("last_event") or scored_events[-1]),
            )
            weights = tuple(float(horizon.weight_for(e)) for e in scored_events)
            play_schedule = _hit_schedule(action, "PLAY", arms["play"],
                                          expected_events=scored_events, expected_weights=weights)
            save_schedule = _hit_schedule(action, "SAVE", arms["save"],
                                          expected_events=scored_events, expected_weights=weights)
            event_start_ft = projected.state.event_start_free_transfers
            if type(event_start_ft) is not int:
                raise FutureStateAssemblyError("WC verified target event-start FT is unknown")
            play_squad = tuple(sorted(_hit_int(pid, where="WC PLAY squad ID")
                                      for pid in play_schedule[0].get("proposed_squad_ids") or ()))
            basis = {int(pid): int(player.purchase_price_tenths)
                     for pid, player in projected.state.by_id().items()}
            tx_request = SimpleNamespace(
                owned_ids=tuple(sorted(basis)), purchase_price_tenths=basis,
                selling_price_tenths={}, bank_tenths=int(projected.state.bank_tenths),
                explicit_sell_rebuy=(),
                players={pid: SimpleNamespace(market_price_tenths=int(price)) for pid, price in prices.items()},
            )
            transaction = wc.plan_transaction(tx_request, play_squad)
            transaction_fields = {
                "retained_ids": list(transaction.retained_ids), "sold_ids": list(transaction.sold_ids),
                "bought_ids": list(transaction.bought_ids), "rebought_ids": list(transaction.rebought_ids),
                "new_basis": {str(pid): int(price) for pid, price in sorted(transaction.new_basis.items())},
                "remaining_bank_tenths": int(transaction.remaining_bank_tenths),
            }
            play_semantics = arms["play"].get("action_semantics") or {}
            if play_semantics.get("transaction") != transaction_fields:
                raise FutureStateAssemblyError("WC PLAY transaction differs from canonical plan_transaction")
            play_ft = sr.free_transfers_after_chip(
                rules, "wildcard", event_start_free_transfers=event_start_ft,
            )
            if (
                _hit_int(play_semantics.get("event_start_free_transfers"), where="WC event-start FT")
                != event_start_ft
                or _hit_int(play_semantics.get("free_transfers_after"), where="WC post-chip FT") != play_ft
                or _hit_int(play_semantics.get("max_free_transfers"), where="WC pinned FT cap")
                != int(rules.max_free_transfers)
            ):
                raise FutureStateAssemblyError("WC PLAY FT transition differs from origin-pinned season rules")
            play_basis = {str(pid): int(basis[pid]) for pid in transaction.retained_ids}
            play_basis.update({str(pid): int(value) for pid, value in transaction.new_basis.items()})
            if any(
                type(row.get("wildcard_active")) is not bool
                or row.get("wildcard_active") is not (index == 0)
                for index, row in enumerate(play_schedule)
            ):
                raise FutureStateAssemblyError("WC PLAY active-chip schedule does not match H1")
            # The complete canonical transaction is the WC H1 transfer set; no separate
            # pre-chip paid batch exists in this supported PLAY schedule.
            wildcard_hit = sr.wildcard_gameweek_hit(
                rules, prior_paid_transfers=0,
                wildcard_transfers=len(transaction.bought_ids), wildcard_active=True,
            )
            if wildcard_hit.get("wildcard_active") is not True:
                raise FutureStateAssemblyError("WC H1 hit calculation did not use the active chip")
            h1_hit_points = _hit_int(wildcard_hit.get("net_hit"), where="WC H1 wildcard net hit")
            h1_state = ts.RouteState(
                event=event,
                players=tuple(ts.RoutePlayer(
                    pid, player_meta[pid].position, player_meta[pid].club_id,
                    int(play_basis[str(pid)]),
                ) for pid in play_squad),
                bank_tenths=int(transaction.remaining_bank_tenths),
                free_transfers=int(play_ft), event_start_free_transfers=int(play_ft),
                chip_state=tuple(projected.state.chip_state),
                transfers_made_this_event=0,
                cumulative_hit_points=(
                    int(projected.state.cumulative_hit_points) + int(h1_hit_points)
                ),
                route_actions=tuple(projected.state.route_actions),
            )
            _check_hit_schedule_state(
                action, "PLAY", play_schedule[0], h1_state,
                free_transfers=int(play_ft), event_start_free_transfers=int(play_ft),
            )
            play_hits = [{
                "event": _hit_int(play_schedule[0].get("event"), where="WC PLAY H1 event"),
                "hit_points": h1_hit_points, "free_transfers": int(play_ft),
            }]
            play_tail_events = scored_events[1:]
            if play_tail_events:
                h2_ft = sr.free_transfers_after_gameweek(rules, play_ft, 0)
                play_tail_start = ts.RouteState(
                    event=play_tail_events[0],
                    players=h1_state.players,
                    chip_state=h1_state.chip_state,
                    bank_tenths=h1_state.bank_tenths,
                    free_transfers=int(h2_ft), event_start_free_transfers=int(h2_ft),
                    transfers_made_this_event=0,
                    cumulative_hit_points=int(h1_state.cumulative_hit_points),
                    route_actions=tuple(h1_state.route_actions),
                )
                play_tail = _build_canonical_roll_route(
                    play_tail_start, events=play_tail_events, player_meta=player_meta, rules=rules,
                )
                for row, replayed in zip(play_schedule[1:], play_tail.actions):
                    state = replayed["transition"].before_state
                    _check_hit_schedule_state(
                        action, "PLAY", row, state,
                        free_transfers=state.free_transfers,
                        event_start_free_transfers=state.event_start_free_transfers,
                    )
                    play_hits.append({
                        "event": _hit_int(row.get("event"), where="WC PLAY event"),
                        "hit_points": _hit_int(
                            replayed["transition"].hit_points, where="WC PLAY canonical roll hit_points",
                        ),
                        "free_transfers": int(state.free_transfers),
                    })

            tail_events = scored_events[4:]
            tail = (
                _build_canonical_roll_route(
                    save_projection.state, events=tail_events, player_meta=player_meta, rules=rules,
                ) if tail_events else None
            )
            canonical_save = list(save_projection.actions) + ([] if tail is None else list(tail.actions))
            save_hits = list(save_hits)
            for index, (row, replayed) in enumerate(zip(save_schedule, canonical_save)):
                transition = replayed.get("transition")
                is_prefix = index < 4
                expected_state = transition.next_event_state if is_prefix else transition.before_state
                expected_ft = expected_state.free_transfers
                if row.get("state_source") != (
                    "CANONICAL_NORMAL_ROUTE" if is_prefix else "CARRIED_CANONICAL_TERMINAL_STATE"
                ):
                    raise FutureStateAssemblyError("WC SAVE route/tail source label differs from canonical policy")
                _check_hit_schedule_state(
                    action, "SAVE", row, expected_state, free_transfers=expected_ft,
                    event_start_free_transfers=expected_ft,
                )
                if row.get("transfer_state", {}).get("wildcard_available") is not True:
                    raise FutureStateAssemblyError("WC SAVE schedule does not preserve the chip option")
                if not is_prefix:
                    save_hits.append({
                        "event": _hit_int(replayed.get("event"), where="WC SAVE carried event"),
                        "hit_points": _hit_int(transition.hit_points, where="WC SAVE roll hit_points"),
                    })
        source_hit_failures.extend(_require_zero_hit_arms(
            action, expected_events=scored_events,
            arms={"PLAY": play_hits, "SAVE": save_hits},
            defer_nonzero_hits=True,
        ))
        proof_rows.append({
            "event": event, "witness_sha256": witness_digest,
            "source_identity": source_identity,
            "play_hits": play_hits, "save_hits": save_hits,
            "scored_events": list(scored_events),
        })
    if source_hit_failures:
        raise FutureStateAssemblyError(
            "FH/WC opportunity schedules include paid hits: " + "; ".join(source_hit_failures)
        )
    return crf.canonical_sha256(proof_rows)
