"""Verified normal-route reconstruction shared by chip production paths.

This module reads a retained PE-9 decision, re-verifies its decision record,
replays the recorded transfer actions against the decision's pinned manager
state and point-in-time price scenario, and exposes the proposed H1 squad and
lineup.  Route scores in the retained JSON are never accepted as authority.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import analytics, generation_store as gs, manager_lineup as ml
from . import route_comparator as rc, route_optimizer as ro, transfer_state as ts
from . import search_permission as sp

ROUTE_REPLAY_VERSION = "chip_normal_route_replay_v1"
ROUTE_SOURCE_INVALID = "CHIP_SOURCE_DECISION_INVALID"
ROUTE_RECONSTRUCTION_INVALID = "CHIP_ROUTE_RECONSTRUCTION_INVALID"
PROPOSED_ROUTE_UNAVAILABLE = "CHIP_PROPOSED_ROUTE_UNAVAILABLE"


class ChipRouteAssemblyError(ValueError):
    """A retained route cannot be reconstructed from canonical evidence."""


def _state_from_payload(payload: Mapping[str, Any]) -> ts.RouteState:
    players = tuple(
        ts.RoutePlayer(
            player_id=int(row["player_id"]),
            position=str(row["position"]),
            club_id=int(row["club_id"]),
            purchase_price_tenths=int(row["purchase_price_tenths"]),
        )
        for row in (payload.get("players") or ())
    )
    return ts.RouteState(
        event=int(payload["event"]),
        players=players,
        bank_tenths=int(payload["bank_tenths"]),
        free_transfers=int(payload["free_transfers"]),
        chip_state=tuple(payload.get("chip_state") or ()),
        event_start_free_transfers=(
            None if payload.get("event_start_free_transfers") is None
            else int(payload["event_start_free_transfers"])
        ),
    )


def replay_serialized_route(
    actions: Sequence[Mapping[str, Any]],
    *,
    start_state: ts.RouteState,
    expected_events: Sequence[int],
    price_scenario: rc.PriceScenario,
    player_meta: Mapping[int, Any],
    recorded_terminal: Mapping[str, Any] | None = None,
) -> ro.PartialRoute:
    """Rebuild transitions and reject any state that differs from the record.

    Only transfer ids are consumed from serialized action rows. The resulting
    bank, FT, squad, hit cost, transition state and terminal state are all
    recomputed with ``apply_transfer_batch``.
    """

    expected = tuple(int(event) for event in expected_events)
    if not expected or tuple(int(row.get("event", -1)) for row in actions) != expected:
        raise ChipRouteAssemblyError(
            f"{ROUTE_RECONSTRUCTION_INVALID}: route events do not equal {list(expected)}"
        )
    state = start_state
    rebuilt: list[dict[str, Any]] = []
    for row, event in zip(actions, expected):
        transfers = row.get("transfers")
        if not isinstance(transfers, Sequence) or isinstance(transfers, (str, bytes)):
            raise ChipRouteAssemblyError(
                f"{ROUTE_RECONSTRUCTION_INVALID}: GW{event} transfers are not a list"
            )
        try:
            batch = ts.TransferBatch(tuple(
                ts.TransferAction(int(move["out"]), int(move["in"]))
                for move in transfers
            ))
        except (KeyError, TypeError, ValueError) as failure:
            raise ChipRouteAssemblyError(
                f"{ROUTE_RECONSTRUCTION_INVALID}: GW{event} transfer ids are malformed: {failure}"
            ) from failure
        kind = str(row.get("kind") or "")
        if kind not in {"ROLL", "SINGLE", "DOUBLE", "HIT"}:
            raise ChipRouteAssemblyError(
                f"{ROUTE_RECONSTRUCTION_INVALID}: GW{event} has unsupported action kind {kind!r}"
            )
        if (kind == "ROLL") != (len(batch.actions) == 0):
            raise ChipRouteAssemblyError(
                f"{ROUTE_RECONSTRUCTION_INVALID}: GW{event} action kind and transfer count disagree"
            )
        snapshot = price_scenario.snapshot_for(event)
        if snapshot is None:
            raise ChipRouteAssemblyError(
                f"{ROUTE_RECONSTRUCTION_INVALID}: no point-in-time price snapshot for GW{event}"
            )
        transition = ts.apply_transfer_batch(state, batch, snapshot, player_meta)
        if not transition.ok:
            raise ChipRouteAssemblyError(
                f"{ROUTE_RECONSTRUCTION_INVALID}: GW{event} transfer replay refused: "
                + "; ".join(str(problem) for problem in transition.errors[:5])
            )
        next_state = transition.next_event_state
        expected_squad = tuple(sorted(int(pid) for pid in (row.get("squad_ids") or ())))
        actual_squad = tuple(sorted(int(player.player_id) for player in next_state.players))
        comparisons = {
            "squad_ids": (actual_squad, expected_squad),
            "bank_after": (int(next_state.bank_tenths), int(row.get("bank_after", -1))),
            "ft_after": (int(next_state.free_transfers), int(row.get("ft_after", -1))),
            "hit_points": (int(transition.hit_points), int(row.get("hit_points", -1))),
        }
        differing = [name for name, (actual, supplied) in comparisons.items() if actual != supplied]
        if differing:
            raise ChipRouteAssemblyError(
                f"{ROUTE_RECONSTRUCTION_INVALID}: GW{event} replay differs from the retained route on "
                f"{differing}"
            )
        rebuilt.append({
            "event": event,
            "kind": kind,
            "batch": batch,
            "transition": transition,
            "hit_points": int(transition.hit_points),
            "delta_3gw": 0.0,
            "squad_ids": actual_squad,
            "ft_after": int(next_state.free_transfers),
            "bank_after": int(next_state.bank_tenths),
        })
        state = next_state

    if recorded_terminal is not None:
        terminal_squad = tuple(sorted(int(pid) for pid in state.by_id()))
        expected_squad = tuple(sorted(int(pid) for pid in (recorded_terminal.get("squad_ids") or ())))
        if (
            terminal_squad != expected_squad
            or int(state.bank_tenths) != int(recorded_terminal.get("bank_tenths", -1))
            or int(state.free_transfers) != int(recorded_terminal.get("free_transfers", -1))
        ):
            raise ChipRouteAssemblyError(
                f"{ROUTE_RECONSTRUCTION_INVALID}: replayed terminal state differs from the retained decision"
            )
    return ro.PartialRoute(
        state=state,
        actions=tuple(rebuilt),
        h1_proxy=0.0,
        window_proxy=0.0,
        hits=sum(int(action["hit_points"]) for action in rebuilt),
    )


@dataclass(frozen=True)
class VerifiedNormalRoute:
    source_decision_id: str
    source_result_sha256: str
    source_artifact_sha256: str
    source_artifact_ref: str
    generation_id: str
    planning_event: int
    events: tuple[int, ...]
    cutoff: str
    data_snapshot_sha256: str
    certification_identity: str
    route_id: str
    actual_owned_ids: tuple[int, ...]
    proposed_owned_ids: tuple[int, ...]
    post_h1_bank_tenths: int
    post_h1_free_transfers: int
    policy: ml.ManagerPolicy
    positions: Mapping[int, str]
    clubs: Mapping[int, int]
    partial: ro.PartialRoute
    route_input_sha256: str
    runner_code_identity: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_decision_id": self.source_decision_id,
            "source_result_sha256": self.source_result_sha256,
            "source_artifact_sha256": self.source_artifact_sha256,
            "generation_id": self.generation_id,
            "planning_event": self.planning_event,
            "events": list(self.events),
            "cutoff": self.cutoff,
            "data_snapshot_sha256": self.data_snapshot_sha256,
            "certification_identity": self.certification_identity,
            "route_id": self.route_id,
            "actual_owned_ids": list(self.actual_owned_ids),
            "proposed_owned_ids": list(self.proposed_owned_ids),
            "post_h1_bank_tenths": self.post_h1_bank_tenths,
            "post_h1_free_transfers": self.post_h1_free_transfers,
            "policy": self.policy.as_dict(),
            "route_input_sha256": self.route_input_sha256,
            "route_replay_version": ROUTE_REPLAY_VERSION,
            "runner_code_identity": self.runner_code_identity,
        }


def _certification_identity(generation: Any) -> str:
    per_event = generation.manifest.get("per_event") or {}
    identities = {
        str(int(event)): str((per_event.get(str(int(event))) or {}).get("bundle_identity") or "")
        for event in generation.events
    }
    if not all(identities.values()):
        raise ChipRouteAssemblyError(
            f"{ROUTE_SOURCE_INVALID}: the normal generation omits a certified bundle identity"
        )
    return gs._certification_identity_for_bundles(
        cutoff=str(generation.cutoff),
        bundle_identities=identities,
        snapshot_sha256=str(generation.snapshot.get("sha256") or ""),
    )


def _policy_from_route(record: Mapping[str, Any], *, event: int) -> ml.ManagerPolicy:
    row = next(
        (item for item in (record.get("per_event") or ()) if int(item.get("event", -1)) == int(event)),
        None,
    )
    payload = None if not isinstance(row, Mapping) else row.get("policy")
    if not isinstance(payload, Mapping):
        raise ChipRouteAssemblyError(
            f"{ROUTE_RECONSTRUCTION_INVALID}: the retained route has no H1 lineup policy"
        )
    try:
        return ml.ManagerPolicy(
            starter_ids=tuple(int(pid) for pid in payload["starter_ids"]),
            bench_gk_id=int(payload["bench_gk_id"]),
            bench_outfield_order=tuple(int(pid) for pid in payload["bench_outfield_order"]),
            captain_id=int(payload["captain_id"]),
            vice_captain_id=int(payload["vice_captain_id"]),
        )
    except (KeyError, TypeError, ValueError) as failure:
        raise ChipRouteAssemblyError(
            f"{ROUTE_RECONSTRUCTION_INVALID}: the H1 lineup policy is malformed: {failure}"
        ) from failure


def load_verified_normal_route(
    conn: sqlite3.Connection,
    decision_id: str,
    *,
    route_id: str | None = None,
    require_recommended: bool = False,
) -> VerifiedNormalRoute:
    """Verify and replay one normal decision route from its pinned snapshot."""

    try:
        verification = gs.verify_decision(conn, str(decision_id))
        if not verification.get("verified"):
            raise ValueError("source decision did not verify")
        record = gs.load_engine_decision_record(conn, str(decision_id))
        if str(record.get("horizon_kind")) != gs.HORIZON_KIND_FOUR_GW:
            raise ValueError("source decision is not a FOUR_GW decision")
        generation = gs.load_generation(conn, str(record["generation_id"]))
        if generation.horizon_kind != gs.HORIZON_KIND_FOUR_GW or len(generation.events) != 4:
            raise ValueError("source decision does not bind the normal four-event product")
        artifact_path = Path(str(record.get("decision_artifact_ref") or ""))
        artifact_bytes = artifact_path.read_bytes()
        artifact = json.loads(artifact_bytes.decode("utf-8"))
        evidence = json.loads(str(record.get("evidence_json") or "{}"))
        artifact_digest = hashlib.sha256(artifact_bytes).hexdigest()
        if artifact_digest != str(evidence.get("decision_artifact_file_sha256") or ""):
            raise ValueError("source decision artifact digest changed after verification")
    except Exception as failure:
        raise ChipRouteAssemblyError(
            f"{ROUTE_SOURCE_INVALID}: {type(failure).__name__}: {failure}"
        ) from failure

    if not isinstance(artifact, Mapping) or artifact.get("schema") != "fpl_brain.four_gw_decision.v1":
        raise ChipRouteAssemblyError(f"{ROUTE_SOURCE_INVALID}: retained artifact is not a four-GW decision")
    decision = artifact.get("decision") or {}
    refinement = artifact.get("finalist_refinement") or {}
    route_table = refinement.get("route_table") or {}
    routes = route_table.get("routes") or {}
    recommendation = (decision.get("transfer_recommendation") or {})
    if route_id is None:
        route_id = recommendation.get("preferred_route_id")
        if not route_id and not require_recommended:
            route_id = (decision.get("lineup") or {}).get("lineup_route_id")
    if not route_id or str(route_id) not in routes:
        raise ChipRouteAssemblyError(
            f"{PROPOSED_ROUTE_UNAVAILABLE}: no retained route matches the preferred route id"
        )
    if require_recommended and (
        str(recommendation.get("status") or "") != "TRANSFER_RECOMMENDATION_AVAILABLE"
        or str(recommendation.get("preferred_route_id") or "") != str(route_id)
    ):
        raise ChipRouteAssemblyError(
            f"{PROPOSED_ROUTE_UNAVAILABLE}: the selected route is not the decision's available preferred route"
        )
    route_record = routes[str(route_id)]
    actions = tuple(route_record.get("actions") or ())
    if not actions:
        raise ChipRouteAssemblyError(f"{ROUTE_RECONSTRUCTION_INVALID}: the selected route has no actions")

    attribution = artifact.get("attribution") or {}
    manager_packet = attribution.get("manager_packet") or {}
    retained_state = attribution.get("consumed_manager_state")
    if not isinstance(retained_state, Mapping):
        raise ChipRouteAssemblyError(f"{ROUTE_SOURCE_INVALID}: canonical consumed manager state is absent")

    source_conn = gs._open_generation_snapshot(generation)
    try:
        canonical_state = gs._derive_four_gw_consumed_manager_state(
            source_conn, generation=generation, manager_packet=manager_packet,
        )
        if gs._json_round_trip_form(dict(retained_state)) != canonical_state:
            raise ChipRouteAssemblyError(
                f"{ROUTE_SOURCE_INVALID}: retained manager state differs from the pinned snapshot"
            )
        initial_payload = canonical_state["route_state"]
        initial_state = _state_from_payload(initial_payload)
        relevant_ids = set(int(player.player_id) for player in initial_state.players)
        for row in actions:
            for move in row.get("transfers") or ():
                relevant_ids.update((int(move["out"]), int(move["in"])))
        base_prices = __import__("fpl_brain.candidate_universe", fromlist=["price_snapshot_as_of"]).price_snapshot_as_of(
            source_conn,
            int(generation.planning_event),
            str(generation.cutoff),
            required_player_ids=sorted(relevant_ids),
        )
        scenario = rc.flat_current_price_scenario(base_prices, generation.events)
        meta = rc.load_player_meta(source_conn, sorted(relevant_ids))
        missing_meta = sorted(relevant_ids - set(int(pid) for pid in meta))
        if missing_meta:
            raise ChipRouteAssemblyError(
                f"{ROUTE_RECONSTRUCTION_INVALID}: pinned player metadata is missing {missing_meta[:8]}"
            )
        partial = replay_serialized_route(
            actions,
            start_state=initial_state,
            expected_events=generation.events,
            price_scenario=scenario,
            player_meta=meta,
            recorded_terminal={
                "squad_ids": list((route_record.get("actions") or [{}])[-1].get("squad_ids") or ()),
                "bank_tenths": route_record.get("terminal_bank_tenths"),
                "free_transfers": route_record.get("terminal_ft"),
            },
        )
    finally:
        source_conn.close()

    policy = _policy_from_route(route_record, event=int(generation.planning_event))
    positions = {int(pid): str(player.position) for pid, player in meta.items()}
    clubs = {int(pid): int(player.club_id) for pid, player in meta.items()}
    proposed_ids = tuple(sorted(int(player.player_id) for player in partial.actions[0]["transition"].next_event_state.players))
    if set(ml.policy_player_ids(policy)) != set(proposed_ids):
        raise ChipRouteAssemblyError(
            f"{ROUTE_RECONSTRUCTION_INVALID}: H1 policy squad differs from the replayed proposed squad"
        )
    legality = ml.policy_legality_errors(policy, positions)
    if legality:
        raise ChipRouteAssemblyError(
            f"{ROUTE_RECONSTRUCTION_INVALID}: H1 policy is illegal: {legality[:6]}"
        )
    stated_lineup = decision.get("lineup") or {}
    if stated_lineup and str(stated_lineup.get("lineup_route_id") or "") == str(route_id):
        stated_policy = stated_lineup.get("policy") or {}
        stated_projection = {key: stated_policy.get(key) for key in (
            "starter_ids", "bench_gk_id", "bench_outfield_order", "captain_id", "vice_captain_id"
        )}
        route_projection = policy.as_dict()
        if any(stated_projection[key] != route_projection[key] for key in stated_projection):
            raise ChipRouteAssemblyError(
                f"{ROUTE_RECONSTRUCTION_INVALID}: decision lineup differs from the retained route policy"
            )

    route_input_sha = analytics.canonical_hash(list(actions))
    snapshot_sha = str(generation.snapshot.get("sha256") or "")
    return VerifiedNormalRoute(
        source_decision_id=str(decision_id),
        source_result_sha256=str(record.get("result_sha256") or ""),
        source_artifact_sha256=artifact_digest,
        source_artifact_ref=str(artifact_path),
        generation_id=str(generation.generation_id),
        planning_event=int(generation.planning_event),
        events=tuple(int(event) for event in generation.events),
        cutoff=str(generation.cutoff),
        data_snapshot_sha256=snapshot_sha,
        certification_identity=_certification_identity(generation),
        route_id=str(route_id),
        actual_owned_ids=tuple(sorted(int(player.player_id) for player in initial_state.players)),
        proposed_owned_ids=proposed_ids,
        post_h1_bank_tenths=int(partial.actions[0]["transition"].next_event_state.bank_tenths),
        post_h1_free_transfers=int(partial.actions[0]["transition"].next_event_state.free_transfers),
        policy=policy,
        positions=positions,
        clubs=clubs,
        partial=partial,
        route_input_sha256=route_input_sha,
        runner_code_identity=str(artifact.get("runner_code_identity") or ""),
    )


def build_bb_tc_evaluations(
    route: VerifiedNormalRoute,
    worlds: Any,
    *,
    chip_available: bool = True,
) -> dict[str, Any]:
    """Evaluate BB and TC on exactly one proposed route, lineup and world set."""

    from . import chip_bench_boost as bb, chip_decision as cd, chip_triple_captain as tc

    if (
        int(worlds.planning_event) != route.planning_event
        or tuple(int(event) for event in worlds.horizon_events) != route.events
        or str(worlds.certification_identity) != route.certification_identity
        or str(worlds.data_snapshot_sha256) != route.data_snapshot_sha256
    ):
        raise ChipRouteAssemblyError(
            f"{ROUTE_RECONSTRUCTION_INVALID}: BB/TC worlds do not match the verified route identity"
        )
    if set(ml.policy_player_ids(route.policy)) != set(route.proposed_owned_ids):
        raise ChipRouteAssemblyError(f"{ROUTE_RECONSTRUCTION_INVALID}: proposed lineup is not the proposed squad")
    binding = cd.ChipHorizonBinding(
        planning_event=route.planning_event,
        horizon_events=route.events,
        certification_identity=route.certification_identity,
        data_snapshot_sha256=route.data_snapshot_sha256,
    )
    first_transition = route.partial.actions[0]["transition"]
    post_save_state = getattr(first_transition, "next_event_state", None)
    if post_save_state is None:
        raise ChipRouteAssemblyError(
            f"{ROUTE_RECONSTRUCTION_INVALID}: verified route has no canonical post-H1 SAVE state"
        )
    post_save_ids = tuple(sorted(int(pid) for pid in post_save_state.by_id()))
    if (
        post_save_ids != tuple(sorted(int(pid) for pid in route.proposed_owned_ids))
        or int(post_save_state.bank_tenths) != int(route.post_h1_bank_tenths)
        or int(post_save_state.free_transfers) != int(route.post_h1_free_transfers)
        or int(post_save_state.event) != int(route.planning_event) + 1
    ):
        raise ChipRouteAssemblyError(
            f"{ROUTE_RECONSTRUCTION_INVALID}: post-H1 SAVE state differs from the verified proposed route"
        )
    post_save_reservation_state = {
        "event": int(post_save_state.event),
        "squad_ids": list(post_save_ids),
        "purchase_price_tenths": {
            str(int(player.player_id)): int(player.purchase_price_tenths)
            for player in post_save_state.players
        },
        "bank_tenths": int(post_save_state.bank_tenths),
        "free_transfers": int(post_save_state.free_transfers),
        "event_start_free_transfers": (
            None if post_save_state.event_start_free_transfers is None
            else int(post_save_state.event_start_free_transfers)
        ),
        "chip_state": list(post_save_state.chip_state),
        "source_decision_id": route.source_decision_id,
        "generation_id": route.generation_id,
        "route_id": route.route_id,
        "route_input_sha256": route.route_input_sha256,
    }
    bb_evaluation = bb.evaluate_bench_boost(bb.BenchBoostRequest(
        worlds=worlds,
        horizon_binding=binding,
        policy=route.policy,
        positions=dict(route.positions),
        chip_available=bool(chip_available),
    ))
    tc_evaluation = tc.evaluate_triple_captain(tc.TripleCaptainRequest(
        worlds=worlds,
        horizon_binding=binding,
        policy=route.policy,
        positions=dict(route.positions),
        chip_available=bool(chip_available),
        route_consequence_delta=0.0,
    ))
    common = {
        "source_decision_id": route.source_decision_id,
        "source_decision_result_sha256": route.source_result_sha256,
        "source_decision_artifact_sha256": route.source_artifact_sha256,
        "generation_id": route.generation_id,
        "route_id": route.route_id,
        "scenario_kind": "PROPOSED_TRANSFER_ROUTE",
        "actual_owned_ids": list(route.actual_owned_ids),
        "proposed_owned_ids": list(route.proposed_owned_ids),
        "lineup": route.policy.as_dict(),
        "scenario_identity": analytics.canonical_hash({
            "source_decision_id": route.source_decision_id,
            "source_result_sha256": route.source_result_sha256,
            "source_artifact_sha256": route.source_artifact_sha256,
            "generation_id": route.generation_id,
            "route_id": route.route_id,
            "actual_owned_ids": list(route.actual_owned_ids),
            "proposed_owned_ids": list(route.proposed_owned_ids),
            "post_h1_bank_tenths": route.post_h1_bank_tenths,
            "post_h1_free_transfers": route.post_h1_free_transfers,
            "lineup": route.policy.as_dict(),
            "cutoff": route.cutoff,
            "data_snapshot_sha256": route.data_snapshot_sha256,
            "certification_identity": route.certification_identity,
            "world_identity": worlds.world_identity,
        }),
        "world_identity": worlds.world_identity,
        "data_snapshot_sha256": route.data_snapshot_sha256,
        "certification_identity": route.certification_identity,
        "planning_event": route.planning_event,
        "horizon_events": list(route.events),
        "save_policy": {
            "objective": "RETAIN_CHIP_AFTER_THE_VERIFIED_NORMAL_H1_ROUTE",
            "post_save_state_for_reservation": post_save_reservation_state,
        },
    }
    from dataclasses import replace

    return {
        cd.CHIP_ACTION_BB: replace(
            bb_evaluation, evidence={**dict(bb_evaluation.evidence), **common}
        ),
        cd.CHIP_ACTION_TC: replace(
            tc_evaluation, evidence={**dict(tc_evaluation.evidence), **common}
        ),
    }


def chip_worlds_from_event_matrix(
    matrix: Mapping[str, Any],
    *,
    route: VerifiedNormalRoute,
    generation: Any,
    config: ro.OptimizerConfig,
    official_player_ids: Sequence[int],
    event: int,
) -> Any:
    """Bind one certified event matrix to the normal four-event source identity."""

    from . import chip_decision as cd

    player_ids = tuple(sorted(int(pid) for pid in official_player_ids))
    event = int(event)
    if event not in route.events:
        raise ChipRouteAssemblyError(
            f"{ROUTE_RECONSTRUCTION_INVALID}: event GW{event} is outside the certified normal horizon"
        )
    expected_key = ro.world_cache_key(
        event=event, generation_id=str(generation.generation_id),
        runs=generation.runs_for(event), config=config, union_ids=player_ids,
    )
    if str(matrix.get(ro.MANAGER_MATRIX_IDENTITY_KEY) or "") != expected_key:
        raise ChipRouteAssemblyError(
            f"{ROUTE_RECONSTRUCTION_INVALID}: H1 chip worlds are not from the canonical certified loader"
        )
    actual_ids = tuple(sorted(int(pid) for pid in (matrix.get("player_ids") or ())))
    if actual_ids != player_ids:
        raise ChipRouteAssemblyError(
            f"{ROUTE_RECONSTRUCTION_INVALID}: H1 chip worlds do not cover the accepted official player pool"
        )
    source_code = str(generation.manifest.get("code_snapshot_sha256") or "")
    snapshot = str(generation.snapshot.get("sha256") or "")
    if not source_code or not snapshot or str(route.data_snapshot_sha256) != snapshot:
        raise ChipRouteAssemblyError(
            f"{ROUTE_RECONSTRUCTION_INVALID}: H1 chip worlds have incomplete generation identity"
        )
    world_identity = analytics.canonical_hash({
        "schema": "fpl_brain.chip_event_world_identity.v1",
        "cache_key": expected_key,
        "generation_id": generation.generation_id,
        "source_event": event,
        "planning_event": int(route.planning_event),
        "horizon_events": list(route.events),
        "cutoff": str(generation.cutoff),
        "data_snapshot_sha256": snapshot,
        "source_snapshot_sha256": source_code,
        "certification_identity": route.certification_identity,
        "optimizer_config": config.as_dict(),
    })
    try:
        worlds = cd.ChipWorldInputs.from_world_matrix(
            matrix,
            planning_event=int(route.planning_event),
            horizon_events=route.events,
            certification_identity=route.certification_identity,
            data_snapshot_sha256=snapshot,
            world_seed=int(config.seed),
            world_identity=world_identity,
            code_snapshot_sha256=source_code,
            source_event=event,
        )
    except Exception as failure:
        raise ChipRouteAssemblyError(
            f"{ROUTE_RECONSTRUCTION_INVALID}: certified H1 chip matrix is malformed: {failure}"
        ) from failure
    worlds.validate()
    return worlds


def chip_worlds_from_h1_matrix(
    matrix: Mapping[str, Any],
    *,
    route: VerifiedNormalRoute,
    generation: Any,
    config: ro.OptimizerConfig,
    official_player_ids: Sequence[int],
) -> Any:
    """Bind the H1 matrix to the unchanged four-event chip contract."""

    return chip_worlds_from_event_matrix(
        matrix,
        route=route,
        generation=generation,
        config=config,
        official_player_ids=official_player_ids,
        event=int(route.planning_event),
    )


def load_certified_h1_chip_worlds(
    conn: sqlite3.Connection,
    route: VerifiedNormalRoute,
    *,
    cache_dir: str | Path | None = None,
) -> Any:
    """Load only the canonical H1 score/minute worlds for a verified route."""

    try:
        generation = gs.load_generation(conn, route.generation_id)
        report = gs.verify_generation(conn, generation.generation_id)
        if not report.get("verified") or generation.horizon_kind != gs.HORIZON_KIND_FOUR_GW:
            raise ValueError("normal four-event generation did not verify")
        if tuple(int(event) for event in generation.events) != route.events:
            raise ValueError("route events differ from the certified normal generation")
        record = gs.load_engine_decision_record(conn, route.source_decision_id)
        artifact = json.loads(Path(str(record["decision_artifact_ref"])).read_text(encoding="utf-8"))
        raw_config = dict((artifact.get("search") or {}).get("config") or {})
        allowed = {field.name for field in fields(ro.OptimizerConfig)}
        raw_config["events"] = route.events
        config = ro.OptimizerConfig(**{key: value for key, value in raw_config.items() if key in allowed})
        source_conn = gs._open_generation_snapshot(generation)
        try:
            from .chip_wildcard import pool_binding_from_store

            pool = pool_binding_from_store(source_conn)
            official_ids = tuple(sorted(int(pid) for pid in pool.eligible_ids))
            if not official_ids:
                raise ValueError("pinned official player pool is empty")
        finally:
            source_conn.close()
        matrix, _details = ro.build_event_worlds(
            conn, generation, int(route.planning_event), official_ids, config,
            cache_dir=None if cache_dir is None else Path(cache_dir),
        )
        return chip_worlds_from_h1_matrix(
            matrix,
            route=route,
            generation=generation,
            config=config,
            official_player_ids=official_ids,
        )
    except ChipRouteAssemblyError:
        raise
    except Exception as failure:
        raise ChipRouteAssemblyError(
            f"{ROUTE_RECONSTRUCTION_INVALID}: canonical H1 world assembly refused: "
            f"{type(failure).__name__}: {failure}"
        ) from failure


def load_certified_event_chip_worlds(
    conn: sqlite3.Connection,
    route: VerifiedNormalRoute,
    *,
    event: int,
    cache_dir: str | Path | None = None,
) -> Any:
    """Load one event matrix from the exact verified normal generation."""

    event = int(event)
    try:
        generation = gs.load_generation(conn, route.generation_id)
        report = gs.verify_generation(conn, generation.generation_id)
        if not report.get("verified") or generation.horizon_kind != gs.HORIZON_KIND_FOUR_GW:
            raise ValueError("normal four-event generation did not verify")
        if tuple(int(value) for value in generation.events) != route.events or event not in route.events:
            raise ValueError("event or route does not match the certified normal generation")
        record = gs.load_engine_decision_record(conn, route.source_decision_id)
        artifact = json.loads(Path(str(record["decision_artifact_ref"])).read_text(encoding="utf-8"))
        raw_config = dict((artifact.get("search") or {}).get("config") or {})
        allowed = {field.name for field in fields(ro.OptimizerConfig)}
        raw_config["events"] = route.events
        config = ro.OptimizerConfig(**{key: value for key, value in raw_config.items() if key in allowed})
        source_conn = gs._open_generation_snapshot(generation)
        try:
            from .chip_wildcard import pool_binding_from_store

            pool = pool_binding_from_store(source_conn)
            official_ids = tuple(sorted(int(pid) for pid in pool.eligible_ids))
            if not official_ids:
                raise ValueError("pinned official player pool is empty")
        finally:
            source_conn.close()
        matrix, _details = ro.build_event_worlds(
            conn,
            generation,
            event,
            official_ids,
            config,
            cache_dir=None if cache_dir is None else Path(cache_dir),
        )
        return chip_worlds_from_event_matrix(
            matrix,
            route=route,
            generation=generation,
            config=config,
            official_player_ids=official_ids,
            event=event,
        )
    except ChipRouteAssemblyError:
        raise
    except Exception as failure:
        raise ChipRouteAssemblyError(
            f"{ROUTE_RECONSTRUCTION_INVALID}: certified event world assembly refused: "
            f"{type(failure).__name__}: {failure}"
        ) from failure


def load_certified_continuation_event_chip_worlds(
    conn: sqlite3.Connection,
    route: VerifiedNormalRoute,
    *,
    event: int,
    action: str,
    expiry_event: int,
    coverage_product: Mapping[str, Any],
    cache_dir: str | Path | None = None,
) -> tuple[Any, Mapping[int, Any], Any]:
    """Load one post-route event from the verified origin-pinned continuation.

    The evaluator's normal four-event binding remains unchanged. The event-world
    identity additionally commits to the continuation generation, event bundle,
    exact runs and expiry product.
    """

    from . import chip_decision as cd, chip_reservation_forecast as crf
    from . import candidate_universe as cu

    event = int(event)
    if action not in {cd.CHIP_ACTION_BB, cd.CHIP_ACTION_TC}:
        raise ChipRouteAssemblyError(
            f"{ROUTE_RECONSTRUCTION_INVALID}: continuation world loader supports only BB/TC"
        )
    if event <= int(route.events[-1]):
        raise ChipRouteAssemblyError(
            f"{ROUTE_RECONSTRUCTION_INVALID}: GW{event} is not beyond the normal route horizon"
        )
    try:
        root_generation = gs.load_generation(conn, route.generation_id)
        continuation_id = str(coverage_product.get("product_generation_id") or "")
        continuation = gs.load_generation(conn, continuation_id)
        root_report = gs.verify_generation(conn, root_generation.generation_id)
        continuation_report = gs.verify_generation(conn, continuation.generation_id)
        source_identity = {
            "source_decision_id": route.source_decision_id,
            "source_result_sha256": route.source_result_sha256,
            "source_artifact_sha256": route.source_artifact_sha256,
            "generation_id": route.generation_id,
            "planning_event": int(route.planning_event),
            "origin_cutoff": str(route.cutoff),
            "data_snapshot_sha256": str(route.data_snapshot_sha256),
            "predictive_code_snapshot_sha256": str(
                root_generation.manifest.get("code_snapshot_sha256") or ""
            ),
            "certification_identity": str(route.certification_identity),
        }
        crf.verify_reservation_coverage_product(
            conn,
            coverage_product,
            expected={
                "action": action,
                "planning_event": int(route.planning_event),
                "expiry_event": int(expiry_event),
                "source_identity": source_identity,
            },
        )
        if (
            not root_report.get("verified")
            or root_generation.horizon_kind != gs.HORIZON_KIND_FOUR_GW
            or tuple(int(value) for value in root_generation.events) != route.events
            or not continuation_report.get("verified")
            or continuation.horizon_kind != gs.HORIZON_KIND_CHIP_RESERVATION
            or int(continuation.planning_event) != int(route.planning_event)
            or str(continuation.cutoff) != str(root_generation.cutoff)
            or str(continuation.snapshot.get("sha256")) != str(root_generation.snapshot.get("sha256"))
            or str(continuation.manifest.get("code_snapshot_sha256"))
            != str(root_generation.manifest.get("code_snapshot_sha256"))
            or tuple(int(value) for value in continuation.events[:4]) != route.events
            or any(
                continuation.runs_for(int(root_event)) != root_generation.runs_for(int(root_event))
                for root_event in route.events
            )
            or event not in tuple(int(value) for value in continuation.events)
            or event not in tuple(int(value) for value in coverage_product.get("product_events") or ())
            or event in tuple(int(value) for value in coverage_product.get("missing_events") or ())
        ):
            raise ValueError("continuation generation does not verify the required root and event identities")
        declared_runs = (coverage_product.get("product_runs_by_event") or {}).get(str(event))
        if not isinstance(declared_runs, Mapping) or {
            str(key): int(value) for key, value in declared_runs.items()
        } != continuation.runs_for(event):
            raise ValueError(f"GW{event} continuation runs differ from the retained coverage product")

        record = gs.load_engine_decision_record(conn, route.source_decision_id)
        artifact = json.loads(Path(str(record["decision_artifact_ref"])).read_text(encoding="utf-8"))
        raw_config = dict((artifact.get("search") or {}).get("config") or {})
        allowed = {field.name for field in fields(ro.OptimizerConfig)}
        raw_config["events"] = route.events
        config = ro.OptimizerConfig(**{key: value for key, value in raw_config.items() if key in allowed})
        source_conn = gs._open_generation_snapshot(continuation)
        try:
            from .chip_wildcard import pool_binding_from_store

            pool = pool_binding_from_store(source_conn)
            official_ids = tuple(sorted(int(pid) for pid in pool.eligible_ids))
            if not official_ids:
                raise ValueError("continuation snapshot has an empty official player pool")
        finally:
            source_conn.close()
        matrix, _details = ro.build_event_worlds(
            conn,
            continuation,
            event,
            official_ids,
            config,
            cache_dir=None if cache_dir is None else Path(cache_dir),
        )
        bundle_identity = str(
            ((continuation.manifest.get("per_event") or {}).get(str(event)) or {}).get("bundle_identity") or ""
        )
        if not bundle_identity:
            raise ValueError(f"GW{event} continuation bundle has no certified identity")
        world_identity = analytics.canonical_hash({
            "schema": "fpl_brain.chip_reservation_continuation_world.v1",
            "root_generation_id": route.generation_id,
            "root_certification_identity": route.certification_identity,
            "continuation_generation_id": continuation.generation_id,
            "continuation_bundle_identity": bundle_identity,
            "continuation_runs": continuation.runs_for(event),
            "coverage_product_sha256": coverage_product.get("product_sha256"),
            "source_event": event,
            "origin_cutoff": str(route.cutoff),
            "data_snapshot_sha256": str(route.data_snapshot_sha256),
            "predictive_code_snapshot_sha256": source_identity["predictive_code_snapshot_sha256"],
            "optimizer_config": config.as_dict(),
        })
        worlds = cd.ChipWorldInputs.from_world_matrix(
            matrix,
            planning_event=int(route.planning_event),
            horizon_events=route.events,
            certification_identity=route.certification_identity,
            data_snapshot_sha256=route.data_snapshot_sha256,
            world_seed=int(config.seed),
            world_identity=world_identity,
            code_snapshot_sha256=source_identity["predictive_code_snapshot_sha256"],
            source_event=event,
        )
        worlds.validate()
        source_conn = gs._open_generation_snapshot(continuation)
        try:
            pool_rows = cu.load_pool(source_conn)
        finally:
            source_conn.close()
        squad_ids = {int(pid) for pid in route.partial.state.by_id()}
        positions = {
            int(pid): str(pool_rows[int(pid)]["position"])
            for pid in squad_ids
            if int(pid) in pool_rows and pool_rows[int(pid)].get("position")
        }
        if set(positions) != squad_ids:
            raise ValueError("terminal route squad has missing pinned positions in continuation snapshot")
        return worlds, positions, matrix
    except ChipRouteAssemblyError:
        raise
    except Exception as failure:
        raise ChipRouteAssemblyError(
            f"{ROUTE_RECONSTRUCTION_INVALID}: verified continuation event assembly refused: "
            f"{type(failure).__name__}: {failure}"
        ) from failure


def build_future_event_chip_opportunity(
    conn: sqlite3.Connection,
    route: VerifiedNormalRoute,
    *,
    action: str,
    event: int,
    reservation_state: Mapping[str, Any],
    made_at: str,
    cache_dir: str | Path | None = None,
    coverage_product: Mapping[str, Any] | None = None,
    rules: Any | None = None,
    rules_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Produce one future BB/TC opportunity from certified event worlds.

    Events inside the normal generation use that generation's retained route
    policy. Later events require a verified CHIP_RESERVATION product and use
    the explicitly declared no-transfer continuation from the route terminal
    state, with a separately ranked legal lineup for each event. The normal
    decision horizon remains four events.
    """

    from . import chip_decision as cd
    from . import chip_reservation_forecast as crf
    from . import candidate_universe as cu

    event = int(event)
    if action not in {cd.CHIP_ACTION_BB, cd.CHIP_ACTION_TC}:
        raise ChipRouteAssemblyError(
            f"{ROUTE_RECONSTRUCTION_INVALID}: {action} has no future-event production evaluator"
        )
    if event <= int(route.planning_event):
        raise ChipRouteAssemblyError(
            f"{ROUTE_RECONSTRUCTION_INVALID}: forecast event {event} is not after the planning event"
        )
    try:
        generation = gs.load_generation(conn, route.generation_id)
        report = gs.verify_generation(conn, generation.generation_id)
        if not report.get("verified") or generation.horizon_kind != gs.HORIZON_KIND_FOUR_GW:
            raise ValueError("normal four-event generation did not verify")
        try:
            sp.require_search_permission(
                conn,
                generation.generation_id,
                expected_origin_planning_event=int(route.planning_event),
                expected_origin_cutoff=str(route.cutoff),
                expected_snapshot_sha256=str(route.data_snapshot_sha256),
            )
        except sp.SearchPermissionRefused:
            raise
        except Exception as failure:
            raise sp.SearchPermissionRefused(
                [
                    "future chip-opportunity origin permission could not be evaluated: "
                    f"{type(failure).__name__}: {failure}"
                ]
            ) from failure
        continuation_context = None
        coverage_product_sha256 = None
        if coverage_product is not None:
            product_identity = str(coverage_product.get("product_sha256") or "")
            if len(product_identity) != 64:
                raise ValueError("expiry coverage product has no SHA-256 identity")
            coverage_product_sha256 = product_identity

        if event in route.events:
            decision_record = gs.load_engine_decision_record(conn, route.source_decision_id)
            artifact = json.loads(Path(str(decision_record["decision_artifact_ref"])).read_text(encoding="utf-8"))
            route_record = (((artifact.get("finalist_refinement") or {}).get("route_table") or {})
                            .get("routes") or {}).get(str(route.route_id))
            if not isinstance(route_record, Mapping):
                raise ValueError("selected normal route is absent from the verified decision artifact")
            policy = _policy_from_route(route_record, event=event)
            route_action = next((
                row for row in route.partial.actions if int(row.get("event", -1)) == event
            ), None)
            if not isinstance(route_action, Mapping):
                raise ValueError("verified normal route has no replayed state for the future event")
            transition = route_action.get("transition")
            event_state = getattr(transition, "next_event_state", None)
            if event_state is None:
                raise ValueError("verified normal route has no canonical post-transfer event state")
            squad_ids = set(int(pid) for pid in event_state.by_id())
            if squad_ids != set(ml.policy_player_ids(policy)):
                raise ValueError("future route lineup does not match its replayed 15-player squad")
            source_conn = gs._open_generation_snapshot(generation)
            try:
                pool = cu.load_pool(source_conn)
            finally:
                source_conn.close()
            positions = {
                int(pid): str(pool[int(pid)]["position"])
                for pid in squad_ids
                if int(pid) in pool and pool[int(pid)].get("position")
            }
            if set(positions) != squad_ids:
                raise ValueError("future route squad has missing pinned player positions")
            worlds = load_certified_event_chip_worlds(
                conn, route, event=event, cache_dir=cache_dir,
            )
        else:
            if coverage_product is None:
                raise ValueError("post-route BB/TC forecasts require a verified CHIP_RESERVATION coverage product")
            from . import season_rules as sr

            if not isinstance(rules, sr.SeasonRules):
                raise ValueError("post-route BB/TC forecasts require season rules resolved from the pinned official snapshot")
            if not isinstance(rules_evidence, Mapping):
                raise ValueError("post-route BB/TC forecasts require verified origin season-rule evidence")
            sr.verify_pinned_season_rules_evidence(
                rules_evidence,
                rules,
                cutoff=route.cutoff,
                data_snapshot_sha256=route.data_snapshot_sha256,
                allow_fixture=True,
            )
            expiry_event = coverage_product.get("expiry_event")
            if expiry_event is None or event > int(expiry_event):
                raise ValueError("post-route event is outside the declared chip-expiry window")
            worlds, positions, matrix = load_certified_continuation_event_chip_worlds(
                conn,
                route,
                event=event,
                action=action,
                expiry_event=int(expiry_event),
                coverage_product=coverage_product,
                cache_dir=cache_dir,
            )
            event_state = _no_transfer_continuation_state(route, event=event, rules=rules)
            squad_ids = {int(pid) for pid in event_state.by_id()}
            if set(positions) != squad_ids:
                raise ValueError("continuation event squad differs from its certified player positions")
            ranked = ml.rank_policies(sorted(squad_ids), positions, matrix, top_k=1)
            top_policies = list(ranked.get("top_policies") or ())
            if not top_policies or not isinstance(top_policies[0], ml.ManagerPolicy):
                raise ValueError("continuation event did not produce a canonical legal lineup")
            policy = top_policies[0]
            state_payload = {
                "event": int(event_state.event),
                "squad_ids": sorted(squad_ids),
                "purchase_price_tenths": {
                    str(int(player.player_id)): int(player.purchase_price_tenths)
                    for player in event_state.players
                },
                "bank_tenths": int(event_state.bank_tenths),
                "free_transfers": int(event_state.free_transfers),
                "event_start_free_transfers": (
                    None if event_state.event_start_free_transfers is None
                    else int(event_state.event_start_free_transfers)
                ),
                "chip_state": list(event_state.chip_state),
                "continuation_model": "CARRY_TERMINAL_ROUTE_STATE_NO_TRANSFERS_PER_EVENT_LINEUP_V1",
            }
            context = {
                "schema": "fpl_brain.chip_reservation_continuation_event.v1",
                "model": "CARRY_TERMINAL_ROUTE_STATE_NO_TRANSFERS_PER_EVENT_LINEUP_V1",
                "action": action,
                "event": int(event),
                "coverage_product_sha256": coverage_product_sha256,
                "world_identity": str(worlds.world_identity),
                "event_manager_state": state_payload,
                "event_policy": policy.as_dict(),
                "player_positions": {str(int(pid)): str(value) for pid, value in sorted(positions.items())},
                "season_rules": asdict(rules),
                "season_rules_sha256": crf.canonical_sha256(asdict(rules)),
                "season_rules_source": str(rules_evidence["source"]),
                "season_rules_evidence": dict(rules_evidence),
                "advanced_no_transfer_events": list(range(int(route.events[-1]) + 1, int(event))),
                "origin_cutoff": str(route.cutoff),
            }
            context["context_sha256"] = crf.canonical_sha256(context)
            continuation_context = context
        binding = cd.ChipHorizonBinding(
            planning_event=int(route.planning_event),
            horizon_events=tuple(route.events),
            certification_identity=str(route.certification_identity),
            data_snapshot_sha256=str(route.data_snapshot_sha256),
        )
        source_identity = {
            "source_decision_id": route.source_decision_id,
            "source_result_sha256": route.source_result_sha256,
            "source_artifact_sha256": route.source_artifact_sha256,
            "generation_id": route.generation_id,
            "planning_event": int(route.planning_event),
            "origin_cutoff": str(route.cutoff),
            "data_snapshot_sha256": str(route.data_snapshot_sha256),
            "predictive_code_snapshot_sha256": str(generation.manifest.get("code_snapshot_sha256") or ""),
            "certification_identity": str(route.certification_identity),
        }
        return crf.build_evaluated_event_opportunity_record(
            action=action,
            event=event,
            worlds=worlds,
            horizon_binding=binding,
            policy=policy,
            positions=positions,
            source_identity=source_identity,
            reservation_state=reservation_state,
            made_at=made_at,
            conn=conn,
            input_as_of=str(route.cutoff),
            coverage_product_sha256=coverage_product_sha256,
            continuation_context=continuation_context,
        )
    except ChipRouteAssemblyError:
        raise
    except sp.SearchPermissionRefused:
        raise
    except Exception as failure:
        raise ChipRouteAssemblyError(
            f"{ROUTE_RECONSTRUCTION_INVALID}: future {action} opportunity refused: "
            f"{type(failure).__name__}: {failure}"
        ) from failure


def _no_transfer_continuation_state(
    route: VerifiedNormalRoute,
    *,
    event: int,
    rules: Any,
) -> ts.RouteState:
    """Carry the verified route's terminal squad/bank through no-transfer GWs.

    This is an explicit forecast model, not a claim that a manager will make no
    transfers. Every intervening no-transfer GW advances FT under the supplied
    official season rules; the proposed permanent squad, purchase basis, bank,
    and chip state remain exactly those of the verified terminal route.
    """

    from . import season_rules as sr

    state = route.partial.state
    target = int(event)
    if state.event != int(route.events[-1]) + 1 or target < int(state.event):
        raise ChipRouteAssemblyError(
            f"{ROUTE_RECONSTRUCTION_INVALID}: terminal route state cannot seed GW{target} continuation"
        )
    while int(state.event) < target:
        next_ft = sr.free_transfers_after_gameweek(rules, int(state.free_transfers), 0)
        state = ts.RouteState(
            event=int(state.event) + 1,
            players=tuple(state.players),
            bank_tenths=int(state.bank_tenths),
            free_transfers=int(next_ft),
            chip_state=tuple(state.chip_state),
            event_start_free_transfers=int(next_ft),
        )
    return state


def _post_h1_save_reservation_state(route: VerifiedNormalRoute) -> dict[str, Any]:
    """Derive the sole SAVE state that future BB/TC opportunities may carry."""

    if not route.partial.actions:
        raise ChipRouteAssemblyError(
            f"{ROUTE_RECONSTRUCTION_INVALID}: verified normal route has no H1 SAVE transition"
        )
    transition = route.partial.actions[0].get("transition")
    state = getattr(transition, "next_event_state", None)
    if state is None or int(state.event) != int(route.planning_event) + 1:
        raise ChipRouteAssemblyError(
            f"{ROUTE_RECONSTRUCTION_INVALID}: verified normal route has no canonical post-H1 SAVE state"
        )
    squad_ids = tuple(sorted(int(pid) for pid in state.by_id()))
    return {
        "event": int(state.event),
        "squad_ids": list(squad_ids),
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
        "source_decision_id": route.source_decision_id,
        "generation_id": route.generation_id,
        "route_id": route.route_id,
        "route_input_sha256": route.route_input_sha256,
    }


def build_bb_tc_reservation_forecast(
    conn: sqlite3.Connection,
    route: VerifiedNormalRoute,
    *,
    action: str,
    expiry_event: int | None,
    reservation_state: Mapping[str, Any],
    made_at: str,
    evidence_root: str | Path,
    cache_dir: str | Path | None = None,
    continuation_generation_id: str | None = None,
    rules: Any | None = None,
) -> dict[str, Any]:
    """Retain the route-backed BB/TC future forecast through known coverage.

    The normal generation contributes the exact future-route events in its
    four-event window. A separately certified CHIP_RESERVATION generation can
    extend one-event BB/TC forecasts through expiry; the terminal route state is
    carried with no transfers and receives a newly ranked legal lineup per
    continuation event. If either the product or the continuation model inputs
    are missing, coverage stays explicitly incomplete and numerically unknown.
    """

    from . import chip_decision as cd
    from . import chip_reservation_forecast as crf

    if action not in {cd.CHIP_ACTION_BB, cd.CHIP_ACTION_TC}:
        raise ChipRouteAssemblyError(
            f"{ROUTE_RECONSTRUCTION_INVALID}: {action} has no production future-reservation route"
        )
    expected_reservation_state = _post_h1_save_reservation_state(route)
    if dict(reservation_state) != expected_reservation_state:
        raise ChipRouteAssemblyError(
            f"{ROUTE_RECONSTRUCTION_INVALID}: reservation SAVE state differs from the verified post-H1 route"
        )
    root = Path(evidence_root)
    root.mkdir(parents=True, exist_ok=True)
    opportunity_refs: list[str] = []
    local_records: dict[str, Mapping[str, Any]] = {}
    source_generation = gs.load_generation(conn, route.generation_id)
    source_identity = {
        "source_decision_id": route.source_decision_id,
        "source_result_sha256": route.source_result_sha256,
        "source_artifact_sha256": route.source_artifact_sha256,
        "generation_id": route.generation_id,
        "planning_event": int(route.planning_event),
        "origin_cutoff": str(route.cutoff),
        "data_snapshot_sha256": str(route.data_snapshot_sha256),
        "predictive_code_snapshot_sha256": str(source_generation.manifest.get("code_snapshot_sha256") or ""),
        "certification_identity": str(route.certification_identity),
    }
    coverage_product = None
    if continuation_generation_id is not None:
        coverage_product = crf.build_reservation_coverage_product(
            conn,
            action=action,
            source_identity=source_identity,
            expiry_event=None if expiry_event is None else int(expiry_event),
            product_generation_id=str(continuation_generation_id),
        )
    product_events = (
        {int(value) for value in coverage_product.get("product_events") or ()}
        if coverage_product is not None else set(route.events)
    )
    forecast_events = (
        [int(value) for value in coverage_product.get("forecast_events") or ()]
        if coverage_product is not None
        else [int(event) for event in route.events[1:]
              if expiry_event is None or int(event) <= int(expiry_event)]
    )
    from . import season_rules as sr

    continuation_rules: Any | None = None
    continuation_rules_evidence: Mapping[str, Any] | None = None
    if (
        coverage_product is not None
        and isinstance(rules, sr.SeasonRules)
        and any(int(event) > int(route.events[-1]) for event in forecast_events)
    ):
        pinned_snapshot_conn = None
        try:
            pinned_snapshot_conn = gs._open_generation_snapshot(source_generation)
            pinned = sr.resolve_origin_pinned_season_rules(
                pinned_snapshot_conn,
                season=str(getattr(rules, "season", "")),
                cutoff=str(route.cutoff),
                data_snapshot_sha256=str(route.data_snapshot_sha256),
            )
        except (sr.SeasonRulesError, sqlite3.Error, OSError, ValueError):
            pinned = None
        finally:
            if pinned_snapshot_conn is not None:
                pinned_snapshot_conn.close()
        if pinned is not None:
            if asdict(rules) != asdict(pinned.rules):
                raise ChipRouteAssemblyError(
                    f"{ROUTE_RECONSTRUCTION_INVALID}: caller season rules differ from the origin-pinned official settings"
                )
            continuation_rules = pinned.rules
            continuation_rules_evidence = dict(pinned.evidence)
    for event in forecast_events:
        if int(event) not in product_events:
            continue
        if int(event) > int(route.events[-1]) and (
            coverage_product is None or continuation_rules is None or continuation_rules_evidence is None
        ):
            # A verified product alone provides worlds, but the explicitly
            # versioned state-continuation rules are also required to score the
            # correct carried manager state. Keep this forecast UNKNOWN.
            continue
        opportunity = build_future_event_chip_opportunity(
            conn,
            route,
            action=action,
            event=int(event),
            reservation_state=reservation_state,
            made_at=made_at,
            cache_dir=cache_dir,
            coverage_product=coverage_product,
            rules=continuation_rules if int(event) > int(route.events[-1]) else None,
            rules_evidence=(
                continuation_rules_evidence if int(event) > int(route.events[-1]) else None
            ),
        )
        receipt = crf.retain_event_opportunity_record(opportunity, root, store_conn=conn)
        reference = Path(receipt["path"]).name
        opportunity_refs.append(reference)
        local_records[reference] = opportunity
    forecast = crf.build_reservation_forecast(
        store_conn=conn,
        action=action,
        planning_event=int(route.planning_event),
        origin_cutoff=str(route.cutoff),
        made_at=made_at,
        expiry_event=None if expiry_event is None else int(expiry_event),
        source_identity=source_identity,
        reservation_state=reservation_state,
        opportunity_refs=opportunity_refs,
        evidence_verifier=local_records.__getitem__,
        input_as_of=str(route.cutoff),
        coverage_product=coverage_product,
    )
    receipt = crf.retain_reservation_forecast(forecast, root, store_conn=conn)
    return {
        "artifact": forecast,
        "path": receipt["path"],
        "artifact_sha256": receipt["artifact_sha256"],
        "opportunity_refs": opportunity_refs,
        "coverage_status": forecast["coverage_status"],
        "raw_value": forecast["raw_value"],
    }
