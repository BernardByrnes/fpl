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
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import analytics, generation_store as gs, manager_lineup as ml
from . import route_comparator as rc, route_optimizer as ro, transfer_state as ts

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


def chip_worlds_from_h1_matrix(
    matrix: Mapping[str, Any],
    *,
    route: VerifiedNormalRoute,
    generation: Any,
    config: ro.OptimizerConfig,
    official_player_ids: Sequence[int],
) -> Any:
    """Bind the certified H1 matrix to the unchanged four-event chip contract."""

    from . import chip_decision as cd

    player_ids = tuple(sorted(int(pid) for pid in official_player_ids))
    expected_key = ro.world_cache_key(
        event=int(route.planning_event), generation_id=str(generation.generation_id),
        runs=generation.runs_for(int(route.planning_event)), config=config, union_ids=player_ids,
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
        "schema": "fpl_brain.chip_h1_world_identity.v1",
        "cache_key": expected_key,
        "generation_id": generation.generation_id,
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
        )
    except Exception as failure:
        raise ChipRouteAssemblyError(
            f"{ROUTE_RECONSTRUCTION_INVALID}: certified H1 chip matrix is malformed: {failure}"
        ) from failure
    worlds.validate()
    return worlds


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
