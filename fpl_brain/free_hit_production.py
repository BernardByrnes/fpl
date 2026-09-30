"""Canonical production assembly for Free Hit PLAY and SAVE route arms."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import fields, replace
from pathlib import Path
from typing import Any, Mapping

from . import analytics, candidate_universe as cu, chip_free_hit as fh, execution_snapshot as es
from . import free_hit_request_adapter as fa, generation_store as gs, route_comparator as rc
from . import route_optimizer as ro, transfer_state as ts
from .chip_route_assembly import (
    ChipRouteAssemblyError,
    VerifiedNormalRoute,
    _state_from_payload,
    load_verified_normal_route,
    replay_serialized_route,
)
from .chip_assessment_store import manager_state_identity
from .chip_wildcard import (
    WildcardPredictiveIdentity,
    WildcardWorldInputs,
)

FH_PRODUCTION_ARMS_INVALID = "FREE_HIT_PRODUCTION_ARMS_INVALID"


class FreeHitProductionError(ValueError):
    """A production Free Hit arm cannot be built from the verified route world."""


def _config_from_source_decision(
    conn: sqlite3.Connection, route: VerifiedNormalRoute, generation: Any
) -> ro.OptimizerConfig:
    record = gs.load_engine_decision_record(conn, route.source_decision_id)
    try:
        artifact = json.loads(Path(str(record["decision_artifact_ref"])).read_text(encoding="utf-8"))
        raw = dict((artifact.get("search") or {}).get("config") or {})
        raw["events"] = tuple(int(event) for event in generation.events)
        allowed = {field.name for field in fields(ro.OptimizerConfig)}
        return ro.OptimizerConfig(**{key: value for key, value in raw.items() if key in allowed})
    except (OSError, KeyError, ValueError, TypeError, json.JSONDecodeError) as failure:
        raise FreeHitProductionError(
            f"{FH_PRODUCTION_ARMS_INVALID}: source route config could not be loaded: {failure}"
        ) from failure


def _snapshot_cache_dir(generation: Any, explicit: str | Path | None) -> Path:
    if explicit is not None:
        return Path(explicit)
    path = Path(str(generation.snapshot.get("path") or ""))
    if not path.is_absolute():
        raise FreeHitProductionError(
            f"{FH_PRODUCTION_ARMS_INVALID}: pinned snapshot has no absolute path for the certified world cache"
        )
    return path.parent / "world_cache" / "manager_worlds"


def _h2_restored_route_state(
    manager: fa.FreeHitManagerState,
    *,
    rules: Any,
    chip_state: tuple,
) -> tuple[fh.RestoredPermanentState, ts.RouteState]:
    permanent = manager.permanent_state()
    restored = fh.restore_permanent_state(
        permanent, rules=rules, restored_event=int(manager.planning_event) + 1,
    )
    restoration_problems = restored.equals_permanent(permanent)
    if restoration_problems:
        raise FreeHitProductionError(
            f"{FH_PRODUCTION_ARMS_INVALID}: Free Hit permanent-state restoration changed "
            f"{restoration_problems}"
        )
    state = ts.RouteState(
        event=int(restored.event),
        players=tuple(
            ts.RoutePlayer(
                player_id=int(pid),
                position=str(manager.positions[int(pid)]),
                club_id=int(manager.clubs[int(pid)]),
                purchase_price_tenths=int(restored.purchase_price_tenths[int(pid)]),
            )
            for pid in restored.owned_ids
        ),
        bank_tenths=int(restored.bank_tenths),
        free_transfers=int(restored.free_transfers),
        chip_state=tuple(chip_state),
        event_start_free_transfers=int(restored.free_transfers),
    )
    return restored, state


def _route_start_disagreements(manager: fa.FreeHitManagerState, route: VerifiedNormalRoute) -> list[str]:
    state = route.partial.actions[0]["transition"].before_state
    found: list[str] = []
    if tuple(sorted(int(player.player_id) for player in state.players)) != tuple(manager.owned_ids):
        found.append("owned squad")
    if int(state.bank_tenths) != int(manager.bank_tenths):
        found.append("bank")
    if int(state.free_transfers) != int(manager.free_transfers):
        found.append("current free transfers")
    basis = {int(player.player_id): int(player.purchase_price_tenths) for player in state.players}
    if basis != {int(k): int(v) for k, v in manager.purchase_price_tenths.items()}:
        found.append("acquisition basis")
    return found


def _free_hit_route_cache_key(generation: Any, event: int, config: ro.OptimizerConfig,
                              player_ids: tuple[int, ...]) -> str:
    return ro.world_cache_key(
        event=int(event), generation_id=generation.generation_id,
        runs=generation.runs_for(int(event)), config=config, union_ids=player_ids,
    )


def build_free_hit_production_request(
    conn: sqlite3.Connection,
    *,
    decision_id: str,
    entry_id: int,
    certification_path: str | Path,
    rules: Any,
    route_id: str | None = None,
    cache_dir: str | Path | None = None,
) -> tuple[fh.FreeHitRequest, dict[str, Any]]:
    """Build and verify both Free Hit arms from one canonical input world.

    SAVE replays the verified normal route and is never optimized a second time.
    PLAY necessarily needs a distinct H2–H4 route because Free Hit restores the
    permanent squad and transfer state before H2; only that tail is optimized.
    No world or route values are accepted from the caller.
    """

    try:
        save_route = load_verified_normal_route(
            conn, decision_id, route_id=route_id, require_recommended=False,
        )
        generation = gs.load_generation(conn, save_route.generation_id)
        report = gs.verify_generation(conn, generation.generation_id)
    except (ChipRouteAssemblyError, gs.GenerationRefused) as failure:
        raise FreeHitProductionError(
            f"{FH_PRODUCTION_ARMS_INVALID}: verified normal decision refused: {failure}"
        ) from failure
    if not report.get("verified") or generation.horizon_kind != gs.HORIZON_KIND_FOUR_GW:
        raise FreeHitProductionError(
            f"{FH_PRODUCTION_ARMS_INVALID}: Free Hit requires a verified FOUR_GW generation"
        )
    events = tuple(int(event) for event in generation.events)
    if len(events) != 4 or events != tuple(range(events[0], events[0] + 4)):
        raise FreeHitProductionError(
            f"{FH_PRODUCTION_ARMS_INVALID}: the normal decision does not provide four contiguous events"
        )
    cache_path = _snapshot_cache_dir(generation, cache_dir)
    source_conn = gs._open_generation_snapshot(generation)
    try:
        manager = fa.free_hit_manager_state(
            source_conn,
            int(entry_id),
            int(generation.planning_event),
            decision_cutoff=str(generation.cutoff),
            as_of=str(generation.cutoff),
        )
        manager_problems = manager.problems()
        if manager_problems:
            raise fa.FreeHitAdapterError(
                f"{fa.FH_MANAGER_STATE_MISSING}: canonical manager state is incomplete: "
                + "; ".join(manager_problems[:6]),
                reasons=(fa.FH_MANAGER_STATE_MISSING,),
            )
        route_state_disagreements = _route_start_disagreements(manager, save_route)
        if route_state_disagreements:
            raise FreeHitProductionError(
                f"{FH_PRODUCTION_ARMS_INVALID}: verified decision route and pinned manager state "
                f"differ on {route_state_disagreements}"
            )
        pool_binding = fa.resolve_pool_binding(source_conn)
        player_ids = tuple(sorted(int(pid) for pid in pool_binding.eligible_ids))
        config = _config_from_source_decision(conn, save_route, generation)
        tail_events = events[1:]
        play_config = replace(config, events=tail_events)
        normal_route_config = replace(config, events=events)
        price_snapshot = cu.price_snapshot_as_of(
            source_conn,
            int(generation.planning_event),
            str(generation.cutoff),
            required_player_ids=player_ids,
        )
        price_scenario = rc.flat_current_price_scenario(price_snapshot, tail_events)
        restored, play_start = _h2_restored_route_state(
            manager,
            rules=rules,
            chip_state=save_route.partial.actions[0]["transition"].before_state.chip_state,
        )

        pool = cu.load_pool(source_conn)
        fixtures = cu.load_fixtures_by_team(source_conn, tail_events)
        xpts_by_event: dict[int, dict[tuple[int, int], dict[str, Any]]] = {}
        minutes_by_event: dict[int, dict[tuple[int, int], dict[str, Any]]] = {}
        for event in tail_events:
            runs = generation.runs_for(event)
            xpts_by_event[event] = cu.load_projection_rows(conn, int(runs["xpts_v1"]))
            minutes_by_event[event] = cu.load_projection_rows(
                conn, int(runs["minutes_v1"]), analytics.MINUTES_V1_KIND,
            )
        universe = cu.build_universe(
            pool=pool,
            events_fixtures=fixtures,
            xpts_rows_by_event=xpts_by_event,
            minutes_rows_by_event=minutes_by_event,
            events=tail_events,
            owned_ids=restored.owned_ids,
            price_snapshot=price_snapshot,
            planning_cutoff=str(generation.cutoff),
            run_refs={"events": list(tail_events),
                      "runs": {str(event): generation.runs_for(event) for event in tail_events}},
        )
        meta_ids = sorted(int(row["player_id"]) for row in universe["universe"])
        player_meta = rc.load_player_meta(source_conn, meta_ids)
        missing_owned = sorted(set(restored.owned_ids) - set(player_meta))
        if missing_owned:
            raise FreeHitProductionError(
                f"{FH_PRODUCTION_ARMS_INVALID}: restored permanent squad has no route metadata "
                f"for {missing_owned[:8]}"
            )
        universe["replacement_edges"] = cu.build_replacement_edges(
            universe_rows=universe["universe"],
            owned_ids=restored.owned_ids,
            state=play_start,
            price_snapshot=price_snapshot,
            player_meta=player_meta,
        )
        play_result = ro.optimize(
            universe=universe,
            initial_state=play_start,
            scenario=price_scenario,
            player_meta=player_meta,
            generation=generation,
            conn=conn,
            config=play_config,
            cache_dir=cache_path,
            provenance={
                "source_decision_id": save_route.source_decision_id,
                "source_decision_result_sha256": save_route.source_result_sha256,
                "source_route_id": save_route.route_id,
                "free_hit_arm": "PLAY_H2_H4",
                "generation_id": generation.generation_id,
                "planning_cutoff": generation.cutoff,
                "data_snapshot_sha256": generation.snapshot.get("sha256"),
            },
        )
        best = ro.best_route_family(play_result, objective="supported_3gw_net_core")
        if not best or not best.get("route_id"):
            raise FreeHitProductionError(
                f"{FH_PRODUCTION_ARMS_INVALID}: PLAY tail optimizer retained no route"
            )
        play_route_id = str(best["route_id"])
        play_record = (play_result.get("routes") or {}).get(play_route_id)
        if not isinstance(play_record, Mapping):
            raise FreeHitProductionError(
                f"{FH_PRODUCTION_ARMS_INVALID}: selected PLAY route was not retained in the optimizer output"
            )
        play_partial = replay_serialized_route(
            play_record.get("actions") or (),
            start_state=play_start,
            expected_events=tail_events,
            price_scenario=price_scenario,
            player_meta=player_meta,
            recorded_terminal={
                "squad_ids": (play_record.get("actions") or [{}])[-1].get("squad_ids") or (),
                "bank_tenths": play_record.get("terminal_bank_tenths"),
                "free_transfers": play_record.get("terminal_ft"),
            },
        )
        position_map = {int(pid): str(meta.position) for pid, meta in player_meta.items()}
        play_input = fa.FreeHitArmRoute(
            partial=play_partial,
            positions_of=lambda ids: {int(pid): position_map[int(pid)] for pid in ids},
            route_config=play_config,
        )
        save_input = fa.FreeHitArmRoute(
            partial=save_route.partial,
            positions_of=lambda ids: {int(pid): position_map[int(pid)] for pid in ids},
            route_config=normal_route_config,
        )
        code_identity = str(generation.manifest.get("code_snapshot_sha256") or "")
        model_config_identity = analytics.canonical_hash({
            "generation_id": generation.generation_id,
            "world_search_draws": int(config.search_draws),
            "world_seed": int(config.seed),
            "monte_carlo_model_version": __import__(
                "fpl_brain.monte_carlo", fromlist=["MONTE_CARLO_MODEL_VERSION"]
            ).MONTE_CARLO_MODEL_VERSION,
            "official_union_ids": list(player_ids),
        })
        world_identity = WildcardPredictiveIdentity(
            cutoff=str(generation.cutoff),
            data_snapshot_sha256=str(generation.snapshot.get("sha256") or ""),
            source_snapshot_sha256=code_identity,
            generation=str(generation.generation_id),
            model_config_identity=model_config_identity,
        )
        h1_matrix, _info = ro.build_event_worlds(
            conn,
            generation,
            int(generation.planning_event),
            player_ids,
            normal_route_config,
            cache_dir=cache_path,
        )
        expected_h1_key = _free_hit_route_cache_key(
            generation, int(generation.planning_event), normal_route_config, player_ids,
        )
        if str(h1_matrix.get(ro.MANAGER_MATRIX_IDENTITY_KEY) or "") != expected_h1_key:
            raise FreeHitProductionError(
                f"{FH_PRODUCTION_ARMS_INVALID}: H1 matrix does not carry the certified cache identity"
            )
        if tuple(sorted(int(pid) for pid in h1_matrix.get("player_ids") or ())) != player_ids:
            raise FreeHitProductionError(
                f"{FH_PRODUCTION_ARMS_INVALID}: H1 matrix does not cover the accepted official pool"
            )
        h1_worlds = WildcardWorldInputs(
            event=int(generation.planning_event),
            worlds=int(h1_matrix["worlds"]),
            player_ids=player_ids,
            minutes={int(pid): tuple(float(v) for v in h1_matrix["minutes"][int(pid)])
                     for pid in player_ids},
            core={int(pid): tuple(float(v) for v in h1_matrix["core"][int(pid)])
                  for pid in player_ids},
            identity=world_identity,
        )
        chip_cert_identity = save_route.certification_identity
        horizon_binding = fa.cd.ChipHorizonBinding(
            planning_event=int(generation.planning_event),
            horizon_events=events,
            certification_identity=chip_cert_identity,
            data_snapshot_sha256=str(generation.snapshot.get("sha256") or ""),
        )
        certified = fa.FreeHitCertifiedInputs(
            horizon_binding=horizon_binding,
            h1_worlds=h1_worlds,
            world_identity=world_identity,
            pool_binding=pool_binding,
            play=play_input,
            save=save_input,
        )
        request = fa.build_free_hit_request(
            manager,
            certified,
            conn=conn,
            manager_source_conn=source_conn,
            certification_path=certification_path,
            world_cache_dir=cache_path,
            as_of=str(generation.cutoff),
            rules=rules,
            generation=generation,
        )
        contract = fh.contract_problems(request)
        if contract:
            raise FreeHitProductionError(
                f"{FH_PRODUCTION_ARMS_INVALID}: the assembled Free Hit request failed contract "
                + "; ".join(contract[:8])
            )
        manager_identity = manager_state_identity({
            "entry_id": int(manager.entry_id),
            "planning_event": int(manager.planning_event),
            "cutoff": str(manager.cutoff),
            "squad_ids": list(manager.owned_ids),
            "purchase_price_tenths": dict(manager.purchase_price_tenths),
            "bank_tenths": int(manager.bank_tenths),
            "free_transfers": int(manager.free_transfers),
            "event_start_free_transfers": int(manager.event_start_free_transfers),
        })
        arm_evidence = {
            "schema": "fpl_brain.free_hit_arm_assembly.v1",
            "source_decision_id": save_route.source_decision_id,
            "source_decision_result_sha256": save_route.source_result_sha256,
            "source_decision_artifact_sha256": save_route.source_artifact_sha256,
            "source_save_route_id": save_route.route_id,
            "source_save_route_input_sha256": save_route.route_input_sha256,
            "generation_id": generation.generation_id,
            "cutoff": generation.cutoff,
            "data_snapshot_sha256": generation.snapshot.get("sha256"),
            "predictive_code_snapshot_sha256": code_identity,
            "model_config_identity": model_config_identity,
            "certification_identity": chip_cert_identity,
            "manager_state_identity": manager_identity,
            "official_pool": pool_binding.as_dict(),
            "world_identity": expected_h1_key,
            "world_config": config.as_dict(),
            "restore_at_h2": {
                "permanent_squad_ids": list(manager.owned_ids),
                "restored_squad_ids": list(restored.owned_ids),
                "permanent_purchase_price_tenths": dict(manager.purchase_price_tenths),
                "restored_purchase_price_tenths": dict(restored.purchase_price_tenths),
                "permanent_bank_tenths": manager.bank_tenths,
                "restored_bank_tenths": restored.bank_tenths,
                "current_h1_free_transfers": manager.free_transfers,
                "event_start_h1_free_transfers": manager.event_start_free_transfers,
                "restored_h2_free_transfers": restored.free_transfers,
                "free_transfers_rule": restored.free_transfers_rule,
                "restoration_problems": restored.equals_permanent(manager.permanent_state()),
            },
            "save": {
                "arm": "SAVE",
                "start_state": fh.state_fingerprint(save_route.partial.actions[0]["transition"].before_state),
                "actions": ro._serialize_route(save_route.partial),
                "terminal_state": fh.state_fingerprint(save_route.partial.state),
                "route_config": normal_route_config.as_dict(),
                "source": "VERIFIED_NORMAL_DECISION_ROUTE",
            },
            "play": {
                "arm": "PLAY",
                "start_state": fh.state_fingerprint(play_partial.actions[0]["transition"].before_state),
                "actions": ro._serialize_route(play_partial),
                "terminal_state": fh.state_fingerprint(play_partial.state),
                "route_config": play_config.as_dict(),
                "source": "OPTIMIZED_RESTORED_PERMANENT_H2_STATE",
                "optimizer_route_id": play_route_id,
                "optimizer_result_sha256": analytics.canonical_hash(play_result),
            },
            "both_arms_verified_by": "free_hit_request_adapter.free_hit_route_from_canonical_route",
        }
        return request, arm_evidence
    except (FreeHitProductionError, fa.FreeHitAdapterError):
        raise
    except Exception as failure:
        raise FreeHitProductionError(
            f"{FH_PRODUCTION_ARMS_INVALID}: production arm assembly refused: "
            f"{type(failure).__name__}: {failure}"
        ) from failure
    finally:
        source_conn.close()
