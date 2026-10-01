"""Canonical production assembly for Free Hit PLAY and SAVE route arms."""

from __future__ import annotations

import json
import math
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


def _source_decision_identity(route: VerifiedNormalRoute) -> dict[str, str]:
    """Return the shared source-decision fields consumed by arm retention."""

    return {
        "source_decision_id": str(route.source_decision_id),
        "source_result_sha256": str(route.source_result_sha256),
        "source_artifact_sha256": str(route.source_artifact_sha256),
    }


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
            **_source_decision_identity(save_route),
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


def build_future_free_hit_event_opportunity(
    conn: sqlite3.Connection,
    *,
    request: fh.FreeHitRequest,
    source_identity: Mapping[str, Any],
    event: int,
    expiry_event: int,
    coverage_product: Mapping[str, Any],
    reservation_state: Mapping[str, Any],
    made_at: str,
) -> dict[str, Any]:
    """Run the canonical FH evaluator for one future event and freeze both arms.

    ``request`` is a typed future-event request assembled from the projected
    permanent state and canonical route/world machinery. Its predictive
    authority is independently derived again from the origin-pinned
    CHIP_RESERVATION generation here. This producer accepts the typed request,
    never caller-supplied numeric values or finished arm artifacts.
    """

    from . import chip_decision as cd, chip_reservation_forecast as crf
    from . import chip_reservation_calibration as crc
    from . import manager_lineup as ml

    event = int(event)
    origin_event = int(source_identity.get("planning_event") or -1)
    if event <= origin_event or int(request.planning_event) != event:
        raise FreeHitProductionError(
            f"{FH_PRODUCTION_ARMS_INVALID}: future FH request does not bind the requested event/origin"
        )
    try:
        crf.verify_reservation_coverage_product(
            conn,
            coverage_product,
            expected={
                "action": cd.CHIP_ACTION_FH,
                "planning_event": origin_event,
                "expiry_event": int(expiry_event),
                "source_identity": dict(source_identity),
            },
        )
    except Exception as failure:
        raise FreeHitProductionError(
            f"{FH_PRODUCTION_ARMS_INVALID}: FH expiry coverage did not verify: {failure}"
        ) from failure
    product_events = {int(value) for value in coverage_product.get("product_events") or ()}
    required_window = tuple(range(event, event + fh.cd.CHIP_HORIZON_LENGTH))
    if not set(required_window).issubset(product_events):
        raise FreeHitProductionError(
            f"{FH_PRODUCTION_ARMS_INVALID}: certified FH opportunity product omits H1-H4 events "
            f"{sorted(set(required_window) - product_events)}"
        )
    if coverage_product.get("coverage_complete") is not True or event not in {
        int(value) for value in coverage_product.get("forecast_events") or ()
    }:
        raise FreeHitProductionError(
            f"{FH_PRODUCTION_ARMS_INVALID}: FH expiry coverage is explicitly incomplete"
        )
    if tuple(int(value) for value in request.horizon_events) != required_window:
        raise FreeHitProductionError(
            f"{FH_PRODUCTION_ARMS_INVALID}: FH request does not preserve its four-event contract"
        )
    problems = fh.contract_problems(request)
    if problems:
        raise FreeHitProductionError(
            f"{FH_PRODUCTION_ARMS_INVALID}: canonical future FH request refused: {'; '.join(problems[:8])}"
        )
    identity = request.world_identity
    product_generation_id = str(coverage_product.get("product_generation_id") or "")
    from .free_hit_decision_authority import model_label, runs_label

    authority = request.decision_authority
    try:
        if not product_generation_id:
            raise ValueError("coverage product has no continuation generation identity")
        verified_authority = fh.FreeHitDecisionAuthority.from_verified_continuation_generation(
            conn,
            product_generation_id,
            events=required_window,
        )
    except Exception as failure:
        raise FreeHitProductionError(
            f"{FH_PRODUCTION_ARMS_INVALID}: future FH continuation authority did not verify: {failure}"
        ) from failure

    def authority_payload(value: Any) -> dict[str, Any]:
        payload = value.as_dict()
        # The loaded-from description identifies the proof path, not the
        # predictive authority represented by the remaining fields.
        payload.pop("loaded_from", None)
        return payload

    authority_disagrees = authority is None
    if authority is not None:
        authority_disagrees = (
            authority_payload(authority) != authority_payload(verified_authority)
            or any(
                dict(authority.bundle_for(covered_event) or {})
                != dict(verified_authority.bundle_for(covered_event) or {})
                for covered_event in required_window
            )
        )
    if authority_disagrees:
        raise FreeHitProductionError(
            f"{FH_PRODUCTION_ARMS_INVALID}: future FH request authority differs from the "
            "verified origin continuation generation"
        )
    authority_h1 = authority.bundle_for(event) if authority is not None else None
    expected_generation_label = (
        runs_label(authority_h1.get("runs")) if isinstance(authority_h1, Mapping) else ""
    )
    product_run_map = coverage_product.get("product_runs_by_event") or {}
    event_run_disagreements = []
    for covered_event in request.horizon_events:
        bundle = authority.bundle_for(int(covered_event)) if authority is not None else None
        expected_runs = bundle.get("runs") if isinstance(bundle, Mapping) else None
        if not isinstance(expected_runs, Mapping) or {
            str(key): int(value) for key, value in expected_runs.items()
        } != {
            str(key): int(value)
            for key, value in (product_run_map.get(str(int(covered_event))) or {}).items()
        }:
            event_run_disagreements.append(int(covered_event))
    if (
        int(request.h1_worlds.event) != event
        or str(identity.generation) != expected_generation_label
        or not authority_h1
        or str(identity.model_config_identity) != model_label(authority_h1.get("model_versions") or {})
        or event_run_disagreements
        or str(identity.cutoff) != str(source_identity.get("origin_cutoff"))
        or str(identity.data_snapshot_sha256) != str(source_identity.get("data_snapshot_sha256"))
        or str(identity.source_snapshot_sha256) != str(source_identity.get("predictive_code_snapshot_sha256"))
    ):
        raise FreeHitProductionError(
            f"{FH_PRODUCTION_ARMS_INVALID}: future FH worlds differ from verified origin/product identity"
        )
    evaluation = fh.evaluate_free_hit(request)
    metrics = evaluation.candidate_metrics
    if evaluation.mean_uplift is None or not isinstance(metrics.get("temporary_policy"), Mapping):
        detail = "; ".join(evaluation.reason_codes)
        raise FreeHitProductionError(
            f"{FH_PRODUCTION_ARMS_INVALID}: canonical future FH evaluator refused: {detail}"
        )
    try:
        play_squad = sorted(int(value) for value in metrics["temporary_squad"])
        save_squad = sorted(int(value) for value in request.permanent.owned_ids)
        play_policy = dict(metrics["temporary_policy"])
        play_bank = int(metrics["temporary_remaining_bank_tenths"])
        save_state = {
            "squad_ids": save_squad,
            "bank_tenths": int(request.permanent.bank_tenths),
            "purchase_price_tenths": {
                str(int(pid)): int(value)
                for pid, value in request.permanent.purchase_price_tenths.items()
            },
            "free_transfers": int(request.permanent.free_transfers),
            "event_start_free_transfers": int(request.permanent.event_start_free_transfers),
        }
        restored = fh.restore_permanent_state(
            request.permanent, rules=request.rules, restored_event=event + 1,
        )
    except (KeyError, TypeError, ValueError, AttributeError) as failure:
        raise FreeHitProductionError(
            f"{FH_PRODUCTION_ARMS_INVALID}: FH evaluator omitted a canonical policy/state: {failure}"
        ) from failure
    if dict(reservation_state) != save_state:
        raise FreeHitProductionError(
            f"{FH_PRODUCTION_ARMS_INVALID}: FH forecast SAVE state differs from projected manager state"
        )
    positions = {str(int(pid)): str(request.positions[int(pid)]) for pid in set(play_squad) | set(save_squad)}
    horizon_events = tuple(int(value) for value in request.horizon_events)
    route_play = {int(row.event): row for row in request.play_route.events}
    route_save = {int(row.event): row for row in request.save_route.events}
    save_policy = dict(route_save[event].policy or {})
    if not save_policy:
        raise FreeHitProductionError(
            f"{FH_PRODUCTION_ARMS_INVALID}: canonical future FH SAVE route omitted its H1 policy"
        )

    def policy_positions(squad: list[int]) -> dict[str, str]:
        return {str(pid): str(request.positions[pid]) for pid in squad}

    def route_row(row: Any, *, weight: float) -> dict[str, Any]:
        lineup = dict(row.policy or {})
        if not lineup:
            raise KeyError(f"canonical FH route omitted GW{row.event} lineup policy")
        squad = sorted(int(value) for value in row.squad_ids)
        return {
            "event": int(row.event), "weight": float(weight),
            "proposed_squad_ids": squad, "lineup": lineup,
            "player_positions": policy_positions(squad),
            "transfer_state": {
                "event": int(row.event), "squad_ids": squad,
                "bank_tenths": int(row.bank_tenths),
                "purchase_price_tenths": {
                    str(int(pid)): int(value)
                    for pid, value in sorted(row.purchase_price_tenths.items())
                },
                "free_transfers": int(row.free_transfers),
            },
        }

    if tuple(sorted(route_save)) != horizon_events or tuple(sorted(route_play)) != horizon_events[1:]:
        raise FreeHitProductionError(
            f"{FH_PRODUCTION_ARMS_INVALID}: canonical FH PLAY/SAVE routes do not cover the same four-event decision"
        )
    play_purchase_basis = {
        str(pid): int(request.permanent.purchase_price_tenths[pid])
        if pid in request.permanent.purchase_price_tenths
        else int(request.market_price_tenths[pid])
        for pid in play_squad
    }
    play_schedule = [{
        "event": event, "weight": 1.0 / fh.cd.CHIP_HORIZON_LENGTH,
        "proposed_squad_ids": play_squad, "lineup": play_policy,
        "player_positions": policy_positions(play_squad),
        "transfer_state": {
            "event": event, "squad_ids": play_squad, "bank_tenths": play_bank,
            "purchase_price_tenths": play_purchase_basis,
            "free_transfers": int(request.permanent.free_transfers),
        },
        "expected_points": float(metrics["temporary_h1_value"]),
        "state_source": "TEMPORARY_FREE_HIT_SQUAD_FOR_ONE_EVENT",
    }, *[
        {
            **route_row(route_play[tail_event], weight=1.0 / fh.cd.CHIP_HORIZON_LENGTH),
            "expected_points": float(route_play[tail_event].mean_net_core),
            "state_source": "CANONICAL_ROUTE_FROM_RESTORED_PERMANENT_STATE",
        }
        for tail_event in horizon_events[1:]
    ]]
    save_schedule = [
        {
            **route_row(route_save[scheduled_event], weight=1.0 / fh.cd.CHIP_HORIZON_LENGTH),
            "expected_points": float(route_save[scheduled_event].mean_net_core),
            "state_source": "CANONICAL_NORMAL_SAVE_ROUTE",
        }
        for scheduled_event in horizon_events
    ]
    play_value = sum(row["weight"] * row["expected_points"] for row in play_schedule)
    save_value = sum(row["weight"] * row["expected_points"] for row in save_schedule)
    if (
        abs(play_value - float(metrics["four_gw_play_value"]) / fh.cd.CHIP_HORIZON_LENGTH) > 1e-6
        or abs(save_value - float(metrics["four_gw_save_value"]) / fh.cd.CHIP_HORIZON_LENGTH) > 1e-6
        or abs(play_value - save_value - float(evaluation.mean_uplift) / fh.cd.CHIP_HORIZON_LENGTH) > 1e-6
    ):
        raise FreeHitProductionError(
            f"{FH_PRODUCTION_ARMS_INVALID}: retained four-event schedules do not reproduce canonical FH value"
        )
    origin_manager_state = {
        "event": int(request.permanent.event),
        "squad_ids": save_squad,
        "bank_tenths": int(request.permanent.bank_tenths),
        "purchase_price_tenths": {
            str(int(pid)): int(value)
            for pid, value in sorted(request.permanent.purchase_price_tenths.items())
        },
        "free_transfers": int(request.permanent.free_transfers),
        "event_start_free_transfers": int(request.permanent.event_start_free_transfers),
    }
    restored_state = {
        "event": int(restored.event),
        "squad_ids": list(restored.owned_ids),
        "bank_tenths": int(restored.bank_tenths),
        "purchase_price_tenths": {
            str(int(pid)): int(value)
            for pid, value in sorted(restored.purchase_price_tenths.items())
        },
        "free_transfers": int(restored.free_transfers),
        "free_transfers_rule": str(restored.free_transfers_rule),
    }
    restored_h2_start_state = dict(request.play_route.start_state)
    try:
        normalized_h2_start = {
            "event": int(restored_h2_start_state["event"]),
            "squad_ids": sorted(int(value) for value in restored_h2_start_state["squad_ids"]),
            "bank_tenths": int(restored_h2_start_state["bank_tenths"]),
            "free_transfers": int(restored_h2_start_state["free_transfers"]),
            "purchase_price_tenths": {
                str(int(key)): int(value)
                for key, value in restored_h2_start_state["purchase_price_tenths"].items()
            },
        }
        expected_h2_start = {
            "event": int(restored.event),
            "squad_ids": sorted(int(value) for value in restored.owned_ids),
            "bank_tenths": int(restored.bank_tenths),
            "free_transfers": int(restored.free_transfers),
            "purchase_price_tenths": {
                str(int(pid)): int(value)
                for pid, value in sorted(restored.purchase_price_tenths.items())
            },
        }
    except (KeyError, TypeError, ValueError, AttributeError) as failure:
        raise FreeHitProductionError(
            f"{FH_PRODUCTION_ARMS_INVALID}: canonical PLAY route omits its restored H2 start state: {failure}"
        ) from failure
    if normalized_h2_start != expected_h2_start:
        raise FreeHitProductionError(
            f"{FH_PRODUCTION_ARMS_INVALID}: canonical PLAY route does not start from the restored permanent H2 state"
        )
    play_state = {
        "squad_ids": play_squad,
        "bank_tenths": play_bank,
        "purchase_price_tenths": {
            str(pid): int(request.market_price_tenths[pid]) for pid in play_squad
        },
        "free_transfers": int(request.permanent.free_transfers),
        "event_start_free_transfers": int(request.permanent.event_start_free_transfers),
    }
    world_identity = crf.canonical_sha256({
        "event": event,
        "generation_id": product_generation_id,
        "world_identity": identity.as_dict(),
        "world_count": int(request.h1_worlds.worlds),
    })
    scenario_identity = crf.canonical_sha256({
        "action": cd.CHIP_ACTION_FH,
        "event": event,
        "source_identity": dict(source_identity),
        "world_identity": world_identity,
        "play_squad": play_squad,
        "save_squad": save_squad,
        "play_policy": play_policy,
        "save_policy": save_policy,
        "restored_state": restored_state,
    })
    source_fields = {
        "source_decision_id": str(source_identity["source_decision_id"]),
        "source_result_sha256": str(source_identity["source_result_sha256"]),
        "source_artifact_sha256": str(source_identity["source_artifact_sha256"]),
        "generation_id": str(source_identity["generation_id"]),
        "cutoff": str(source_identity["origin_cutoff"]),
        "data_snapshot_sha256": str(source_identity["data_snapshot_sha256"]),
        "predictive_code_snapshot_sha256": str(source_identity["predictive_code_snapshot_sha256"]),
        "certification_identity": str(source_identity["certification_identity"]),
        "product_generation_id": product_generation_id,
        "coverage_product_sha256": str(coverage_product["product_sha256"]),
    }
    expected = float(evaluation.mean_uplift) / fh.cd.CHIP_HORIZON_LENGTH
    four_event_uncertainty = dict(evaluation.uncertainty)
    normalized_uncertainty = dict(four_event_uncertainty)
    for field in (
        "paired_interval_low", "paired_interval_high",
        "paired_quantile_05", "paired_quantile_50", "paired_quantile_95",
    ):
        try:
            four_event_value = float(four_event_uncertainty[field])
        except (KeyError, TypeError, ValueError, OverflowError) as failure:
            raise FreeHitProductionError(
                f"{FH_PRODUCTION_ARMS_INVALID}: FH evaluator uncertainty omits a finite {field}"
            ) from failure
        if not math.isfinite(four_event_value):
            raise FreeHitProductionError(
                f"{FH_PRODUCTION_ARMS_INVALID}: FH evaluator uncertainty has a non-finite {field}"
            )
        normalized_uncertainty[field] = round(
            four_event_value / fh.cd.CHIP_HORIZON_LENGTH, 6,
        )
    arms: dict[str, dict[str, Any]] = {
        "play": {
            "arm_id": f"fh-play-{crf.canonical_sha256([scenario_identity, 'PLAY'])[:16]}",
            "action": cd.CHIP_ACTION_FH, "event": event,
            "scenario_identity": scenario_identity, "world_identity": world_identity,
            "counterfactual_role": "PLAY", "proposed_squad_ids": play_squad,
            "lineup": play_policy, "player_positions": policy_positions(play_squad),
            "reservation_state": save_state,
            "scoring_rule_version": crc.OUTCOME_SCORING_RULE_VERSION,
            "paired_value": play_value,
            "valuation_schedule": {
                "value_definition": "FOUR_EVENT_NORMALIZED_MEAN_POINTS",
                "events": play_schedule,
            },
            "evaluator_version": evaluation.evaluator_version,
            "action_semantics": {
                "role": "PLAY", "transfer_state": play_state,
                "origin_manager_state": origin_manager_state,
                "origin_manager_state_sha256": crf.canonical_sha256(origin_manager_state),
                "restore_at_h2": {
                    "restore_event": event + 1,
                    "restored_h2_start_state": normalized_h2_start,
                    "origin_manager_state_sha256": crf.canonical_sha256(origin_manager_state),
                    "permanent_squad_ids": save_squad,
                    "restored_squad_ids": list(restored.owned_ids),
                    "permanent_purchase_price_tenths": save_state["purchase_price_tenths"],
                    "restored_purchase_price_tenths": {
                        str(int(pid)): int(value) for pid, value in restored.purchase_price_tenths.items()
                    },
                    "permanent_bank_tenths": int(request.permanent.bank_tenths),
                    "restored_bank_tenths": int(restored.bank_tenths),
                    "event_start_h1_free_transfers": int(request.permanent.event_start_free_transfers),
                    "restored_h2_free_transfers": int(restored.free_transfers),
                },
                "valuation_schedule_required": True,
            },
            **source_fields,
        },
        "save": {
            "arm_id": f"fh-save-{crf.canonical_sha256([scenario_identity, 'SAVE'])[:16]}",
            "action": cd.CHIP_ACTION_FH, "event": event,
            "scenario_identity": scenario_identity, "world_identity": world_identity,
            "counterfactual_role": "SAVE",
            "proposed_squad_ids": save_schedule[0]["proposed_squad_ids"],
            "lineup": save_schedule[0]["lineup"],
            "player_positions": save_schedule[0]["player_positions"],
            "reservation_state": save_state,
            "scoring_rule_version": crc.OUTCOME_SCORING_RULE_VERSION,
            "paired_value": save_value,
            "valuation_schedule": {
                "value_definition": "FOUR_EVENT_NORMALIZED_MEAN_POINTS",
                "events": save_schedule,
            },
            "evaluator_version": evaluation.evaluator_version,
            "action_semantics": {
                "role": "SAVE", "transfer_state": save_schedule[0]["transfer_state"],
                "origin_manager_state": origin_manager_state,
                "origin_manager_state_sha256": crf.canonical_sha256(origin_manager_state),
                "retains_chip_option": True,
                "valuation_schedule_required": True,
            },
            **source_fields,
        },
    }
    return crf.build_event_opportunity_record(
        action=cd.CHIP_ACTION_FH,
        planning_event=origin_event,
        event=event,
        origin_cutoff=str(source_identity["origin_cutoff"]),
        input_as_of=str(source_identity["origin_cutoff"]),
        made_at=made_at,
        expected_incremental_points=expected,
        opportunity_model=f"{evaluation.evaluator_version}:H1_EVENT_UPLIFT_V1",
        source_identity=source_identity,
        reservation_state=save_state,
        world_identity=world_identity,
        outcome_arms=arms,
        coverage_product_sha256=str(coverage_product["product_sha256"]),
        evaluator_identity={
            "evaluator_version": str(evaluation.evaluator_version),
            "action": cd.CHIP_ACTION_FH,
            "event": event,
            "expected_incremental_points": expected,
            "mean_four_event_uplift": float(evaluation.mean_uplift),
            "uncertainty": normalized_uncertainty,
            "four_event_uncertainty": four_event_uncertainty,
            "decision_horizon_events": list(request.horizon_events),
            "world_identity": world_identity,
            "source_identity": dict(source_identity),
            "opportunity_value_definition": "FOUR_EVENT_NORMALIZED_MEAN_POINTS",
            "uncertainty_value_definition": "FOUR_EVENT_NORMALIZED_MEAN_POINTS",
        },
    )
