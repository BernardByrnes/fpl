"""Production assembly for Wildcard's distinct 6–10 event value product."""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import fields
from pathlib import Path
from typing import Any, Mapping

from . import analytics, candidate_universe as cu, generation_store as gs
from . import chip_decision as cd, chip_reservation_calibration as crc
from . import chip_reservation_forecast as crf, manager_lineup as ml, season_rules as sr
from . import route_optimizer as ro, wildcard_request_adapter as wa
from .chip_wildcard import (
    WildcardPlayer,
    WildcardPlayerEvent,
    WildcardPredictiveIdentity,
    WildcardWorldInputs,
    pool_binding_from_store,
    value_horizon_binding,
    wildcard_horizon,
)
from .chip_route_assembly import ChipRouteAssemblyError, load_verified_normal_route

WC_PRODUCTION_INPUTS_MISSING = "WILDCARD_PRODUCTION_CERTIFIED_INPUTS_MISSING"


class WildcardProductionError(ValueError):
    """Certified Wildcard evidence cannot be assembled from one pinned world."""


def _route_config_from_decision(conn: sqlite3.Connection, decision_id: str, generation: Any) -> ro.OptimizerConfig:
    record = gs.load_engine_decision_record(conn, str(decision_id))
    artifact_path = Path(str(record.get("decision_artifact_ref") or ""))
    try:
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        raw = dict((artifact.get("search") or {}).get("config") or {})
        raw["events"] = tuple(int(event) for event in generation.events)
        allowed = {field.name for field in fields(ro.OptimizerConfig)}
        return ro.OptimizerConfig(**{key: value for key, value in raw.items() if key in allowed})
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as failure:
        raise WildcardProductionError(
            f"{WC_PRODUCTION_INPUTS_MISSING}: source decision has no reusable route-world config: {failure}"
        ) from failure


def _finite_number(row: Mapping[str, Any], key: str, *, context: str) -> float:
    value = row.get(key)
    if value is None:
        raise WildcardProductionError(
            f"{WC_PRODUCTION_INPUTS_MISSING}: {context} omits required projection field {key}"
        )
    try:
        number = float(value)
    except (TypeError, ValueError) as failure:
        raise WildcardProductionError(
            f"{WC_PRODUCTION_INPUTS_MISSING}: {context} field {key} is not numeric"
        ) from failure
    if not math.isfinite(number):
        raise WildcardProductionError(
            f"{WC_PRODUCTION_INPUTS_MISSING}: {context} field {key} is not finite"
        )
    return number


def _players_from_certified_rows(
    *,
    pool: Mapping[int, Mapping[str, Any]],
    prices: Mapping[int, int],
    events: tuple[int, ...],
    fixtures: Mapping[tuple[int, int], list[int]],
    xpts_by_event: Mapping[int, Mapping[tuple[int, int], Mapping[str, Any]]],
    minutes_by_event: Mapping[int, Mapping[tuple[int, int], Mapping[str, Any]]],
    identity: WildcardPredictiveIdentity,
) -> dict[int, WildcardPlayer]:
    players: dict[int, WildcardPlayer] = {}
    for raw_pid, meta in sorted(pool.items()):
        pid = int(raw_pid)
        position = str(meta.get("position") or "")
        club_id = meta.get("club_id")
        price = prices.get(pid)
        if position not in {"GKP", "DEF", "MID", "FWD"} or club_id is None or price is None:
            raise WildcardProductionError(
                f"{WC_PRODUCTION_INPUTS_MISSING}: official player {pid} lacks pinned position, club or price"
            )
        per_event: dict[int, WildcardPlayerEvent] = {}
        for event in events:
            fixture_ids = tuple(int(fid) for fid in fixtures.get((event, int(club_id)), ()))
            if not fixture_ids:
                per_event[event] = WildcardPlayerEvent(
                    event=event,
                    expected_points=0.0,
                    expected_minutes=0.0,
                    p_start=0.0,
                    availability=0.0,
                    fixture_count=0,
                    identity=identity,
                )
                continue
            xpts_rows = xpts_by_event.get(event, {})
            minutes_rows = minutes_by_event.get(event, {})
            missing_xpts = [fid for fid in fixture_ids if (pid, fid) not in xpts_rows]
            missing_minutes = [fid for fid in fixture_ids if (pid, fid) not in minutes_rows]
            if missing_xpts or missing_minutes:
                raise WildcardProductionError(
                    f"{WC_PRODUCTION_INPUTS_MISSING}: GW{event} player {pid} has fixture coverage gaps "
                    f"(xPts={missing_xpts[:4]}, minutes={missing_minutes[:4]})"
                )
            point_total = minute_total = 0.0
            starts: list[float] = []
            availability_values: list[float] = []
            for fixture_id in fixture_ids:
                xrow = xpts_rows[(pid, fixture_id)]
                mrow = minutes_rows[(pid, fixture_id)]
                point_total += _finite_number(xrow, "core_xpts", context=f"GW{event} player {pid} xPts")
                minute_total += _finite_number(
                    xrow, "expected_minutes", context=f"GW{event} player {pid} xPts"
                )
                starts.append(_finite_number(xrow, "p_start", context=f"GW{event} player {pid} xPts"))
                availability_key = next(
                    (name for name in ("joint_availability", "p_available", "p_appearance")
                     if mrow.get(name) is not None),
                    None,
                )
                if availability_key is None:
                    raise WildcardProductionError(
                        f"{WC_PRODUCTION_INPUTS_MISSING}: GW{event} player {pid} minutes row has no "
                        "joint availability field"
                    )
                availability_values.append(_finite_number(
                    mrow, availability_key, context=f"GW{event} player {pid} minutes"
                ))
            if any(not 0.0 <= value <= 1.0 for value in (*starts, *availability_values)):
                raise WildcardProductionError(
                    f"{WC_PRODUCTION_INPUTS_MISSING}: GW{event} player {pid} has an invalid probability"
                )
            per_event[event] = WildcardPlayerEvent(
                event=event,
                expected_points=point_total,
                expected_minutes=minute_total,
                p_start=max(starts),
                availability=max(availability_values),
                fixture_count=len(fixture_ids),
                identity=identity,
            )
        players[pid] = WildcardPlayer(
            player_id=pid,
            position=position,
            club_id=int(club_id),
            market_price_tenths=int(price),
            events=per_event,
            web_name=str(meta.get("web_name") or ""),
        )
    return players


def _canonical_worlds(
    conn: sqlite3.Connection,
    *,
    generation: Any,
    events: tuple[int, ...],
    player_ids: tuple[int, ...],
    config: ro.OptimizerConfig,
    cache_dir: Path | None,
    identity: WildcardPredictiveIdentity,
) -> dict[int, WildcardWorldInputs]:
    worlds_by_event: dict[int, WildcardWorldInputs] = {}
    for event in events:
        try:
            matrix, _info = ro.build_event_worlds(
                conn, generation, int(event), player_ids, config, cache_dir=cache_dir,
            )
        except Exception as failure:
            raise WildcardProductionError(
                f"{WC_PRODUCTION_INPUTS_MISSING}: certified world load refused for GW{event}: {failure}"
            ) from failure
        expected_key = ro.world_cache_key(
            event=int(event), generation_id=generation.generation_id,
            runs=generation.runs_for(int(event)), config=config, union_ids=player_ids,
        )
        if str(matrix.get(ro.MANAGER_MATRIX_IDENTITY_KEY) or "") != expected_key:
            raise WildcardProductionError(
                f"{WC_PRODUCTION_INPUTS_MISSING}: GW{event} matrix is not from the canonical certified loader"
            )
        actual_ids = tuple(sorted(int(pid) for pid in matrix.get("player_ids") or ()))
        if actual_ids != player_ids:
            raise WildcardProductionError(
                f"{WC_PRODUCTION_INPUTS_MISSING}: GW{event} matrix player set differs from the official pool"
            )
        worlds_by_event[int(event)] = WildcardWorldInputs(
            event=int(event),
            worlds=int(matrix["worlds"]),
            player_ids=player_ids,
            minutes={int(pid): tuple(float(value) for value in matrix["minutes"][int(pid)])
                     for pid in player_ids},
            core={int(pid): tuple(float(value) for value in matrix["core"][int(pid)])
                  for pid in player_ids},
            identity=identity,
        )
    return worlds_by_event


def build_wildcard_production_request(
    conn: sqlite3.Connection,
    *,
    decision_id: str,
    value_generation_id: str,
    entry_id: int,
    length: int,
    rules: Any,
    cache_dir: str | Path | None = None,
    reservation: Any | None = None,
    route_id: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Build a Wildcard request from two verified, same-world generations.

    ``conn`` is the certified live evidence store. Manager, price, fixture,
    player and chip state is read from the normal generation's immutable pinned
    source snapshot. No projection or prediction is regenerated here.
    """

    try:
        route = load_verified_normal_route(
            conn, decision_id, route_id=route_id, require_recommended=False,
        )
        chip_generation = gs.load_generation(conn, route.generation_id)
        value_generation = gs.load_generation(conn, str(value_generation_id))
        chip_report = gs.verify_generation(conn, chip_generation.generation_id)
        value_report = gs.verify_generation(conn, value_generation.generation_id)
    except (ChipRouteAssemblyError, gs.GenerationRefused) as failure:
        raise WildcardProductionError(
            f"{WC_PRODUCTION_INPUTS_MISSING}: source generation or decision refused: {failure}"
        ) from failure
    if not chip_report.get("verified") or not value_report.get("verified"):
        raise WildcardProductionError(f"{WC_PRODUCTION_INPUTS_MISSING}: a required generation did not verify")
    if chip_generation.horizon_kind != gs.HORIZON_KIND_FOUR_GW:
        raise WildcardProductionError(f"{WC_PRODUCTION_INPUTS_MISSING}: normal chip generation is not FOUR_GW")
    if value_generation.horizon_kind != gs.HORIZON_KIND_WILDCARD_VALUE:
        raise WildcardProductionError(
            f"{WC_PRODUCTION_INPUTS_MISSING}: Wildcard value generation is not WILDCARD_VALUE"
        )
    if (
        value_generation.planning_event != chip_generation.planning_event
        or tuple(value_generation.events[:4]) != tuple(chip_generation.events)
        or str(value_generation.cutoff) != str(chip_generation.cutoff)
        or str(value_generation.snapshot.get("sha256")) != str(chip_generation.snapshot.get("sha256"))
        or str(value_generation.manifest.get("code_snapshot_sha256"))
        != str(chip_generation.manifest.get("code_snapshot_sha256"))
        or any(value_generation.runs_for(event) != chip_generation.runs_for(event)
               for event in chip_generation.events)
    ):
        raise WildcardProductionError(
            f"{WC_PRODUCTION_INPUTS_MISSING}: Wildcard value generation does not share the normal "
            "planning event, exact four-run prefix, cutoff, snapshot and predictive code identity"
        )

    value_events = tuple(int(event) for event in value_generation.events)
    if not (gs.WILDCARD_VALUE_MIN_EVENTS <= len(value_events) <= gs.WILDCARD_VALUE_MAX_EVENTS):
        raise WildcardProductionError(
            f"{WC_PRODUCTION_INPUTS_MISSING}: persisted Wildcard generation has {len(value_events)} events"
        )
    if int(length) != len(value_events):
        raise WildcardProductionError(
            f"{WC_PRODUCTION_INPUTS_MISSING}: requested Wildcard length {int(length)} does not match "
            f"the separately certified {len(value_events)} event generation"
        )
    planning_event = int(chip_generation.planning_event)
    if value_events != tuple(range(planning_event, planning_event + len(value_events))):
        raise WildcardProductionError(f"{WC_PRODUCTION_INPUTS_MISSING}: Wildcard value events are not contiguous")

    from . import search_permission as sp

    try:
        search_permission_evaluation = sp.require_search_permission(
            conn,
            chip_generation.generation_id,
            expected_origin_planning_event=planning_event,
            expected_origin_cutoff=str(chip_generation.cutoff),
            expected_snapshot_sha256=str(chip_generation.snapshot.get("sha256") or ""),
        )
    except sp.SearchPermissionRefused:
        raise
    except Exception as failure:
        raise sp.SearchPermissionRefused(
            [f"Wildcard production origin permission could not be evaluated: {type(failure).__name__}: {failure}"]
        ) from failure

    source_conn = gs._open_generation_snapshot(chip_generation)
    try:
        pool_binding = pool_binding_from_store(source_conn)
        manager = wa.wildcard_manager_state(
            source_conn,
            int(entry_id),
            planning_event,
            cutoff=str(chip_generation.cutoff),
            eligible_ids=pool_binding.eligible_ids,
            as_of=str(chip_generation.cutoff),
        )
        pool = cu.load_pool(source_conn)
        if tuple(sorted(pool)) != tuple(sorted(pool_binding.eligible_ids)):
            raise WildcardProductionError(
                f"{WC_PRODUCTION_INPUTS_MISSING}: pinned active-player set differs from accepted official pool"
            )
        config = _route_config_from_decision(conn, decision_id, chip_generation)
        identity_config = {
            "generation_id": value_generation.generation_id,
            "events": list(value_events),
            "runs_by_event": {
                str(event): value_generation.runs_for(event) for event in value_events
            },
            "world_config": config.as_dict(),
        }
        model_config_identity = analytics.canonical_hash(identity_config)
        predictive_identity = WildcardPredictiveIdentity(
            cutoff=str(value_generation.cutoff),
            data_snapshot_sha256=str(value_generation.snapshot.get("sha256") or ""),
            source_snapshot_sha256=str(value_generation.manifest.get("code_snapshot_sha256") or ""),
            generation=str(value_generation.generation_id),
            model_config_identity=model_config_identity,
        )
        fixtures = cu.load_fixtures_by_team(source_conn, value_events)
        xpts_by_event: dict[int, dict[tuple[int, int], dict[str, Any]]] = {}
        minutes_by_event: dict[int, dict[tuple[int, int], dict[str, Any]]] = {}
        for event in value_events:
            runs = value_generation.runs_for(event)
            if "xpts_v1" not in runs or "minutes_v1" not in runs:
                raise WildcardProductionError(
                    f"{WC_PRODUCTION_INPUTS_MISSING}: GW{event} certified generation omits xPts/minutes runs"
                )
            xpts_by_event[event] = cu.load_projection_rows(conn, int(runs["xpts_v1"]))
            minutes_by_event[event] = cu.load_projection_rows(
                conn, int(runs["minutes_v1"]), analytics.MINUTES_V1_KIND,
            )
        players = _players_from_certified_rows(
            pool=pool,
            prices=manager.market_price_tenths,
            events=value_events,
            fixtures=fixtures,
            xpts_by_event=xpts_by_event,
            minutes_by_event=minutes_by_event,
            identity=predictive_identity,
        )
        horizon = wildcard_horizon(
            planning_event,
            length=len(value_events),
            last_event=int(value_generation.manifest.get("last_event") or value_events[-1]),
        )
        if tuple(horizon.events) != value_events:
            raise WildcardProductionError(
                f"{WC_PRODUCTION_INPUTS_MISSING}: supported Wildcard horizon differs from persisted events"
            )
        value_binding = value_horizon_binding(
            planning_event,
            horizon,
            decision_cutoff=str(value_generation.cutoff),
            data_snapshot_sha256=str(value_generation.snapshot.get("sha256") or ""),
            source_snapshot_sha256=str(value_generation.manifest.get("code_snapshot_sha256") or ""),
            prediction_generation=str(value_generation.generation_id),
            model_config_identity=model_config_identity,
        )
        chip_binding = wa.cd.ChipHorizonBinding(
            planning_event=planning_event,
            horizon_events=tuple(chip_generation.events),
            certification_identity=route.certification_identity,
            data_snapshot_sha256=str(chip_generation.snapshot.get("sha256") or ""),
        )
        value_worlds = _canonical_worlds(
            conn,
            generation=value_generation,
            events=value_events,
            player_ids=tuple(sorted(pool_binding.eligible_ids)),
            config=config,
            cache_dir=None if cache_dir is None else Path(cache_dir),
            identity=predictive_identity,
        )
        save_worlds = _canonical_worlds(
            conn,
            generation=chip_generation,
            events=tuple(chip_generation.events),
            player_ids=tuple(sorted(pool_binding.eligible_ids)),
            config=config,
            cache_dir=None if cache_dir is None else Path(cache_dir),
            identity=predictive_identity,
        )
        official = __import__("fpl_brain.repositories", fromlist=["latest_accepted_bootstrap_generation"]).latest_accepted_bootstrap_generation(source_conn)
        if not official:
            raise WildcardProductionError(f"{WC_PRODUCTION_INPUTS_MISSING}: pinned snapshot has no accepted pool")
        certified = wa.WildcardCertifiedInputs(
            horizon=horizon,
            chip_horizon_binding=chip_binding,
            value_horizon_binding=value_binding,
            players=players,
            worlds_by_event=value_worlds,
            generation=dict(official),
            chip_generation_id=chip_generation.generation_id,
            value_generation_id=value_generation.generation_id,
        )
        positions_of = lambda squad_ids: {
            int(pid): str(pool[int(pid)]["position"]) for pid in squad_ids if int(pid) in pool
        }
        request = wa.build_wildcard_request(
            manager,
            certified,
            None,
            rules=rules,
            data_snapshot_sha256=str(chip_generation.snapshot.get("sha256") or ""),
            reservation=reservation,
            conn=conn,
            manager_source_conn=source_conn,
            pool_binding=pool_binding,
            canonical_route=route.partial,
            route_worlds_by_event={event: {
                "worlds": body.worlds,
                "player_ids": list(body.player_ids),
                "minutes": body.minutes,
                "core": body.core,
            } for event, body in save_worlds.items()},
            route_positions_of=positions_of,
            route_config=config,
            route_events=tuple(chip_generation.events),
            club_of={int(pid): int(meta["club_id"]) for pid, meta in pool.items()},
        )
        return request, {
            "chip_generation_id": chip_generation.generation_id,
            "value_generation_id": value_generation.generation_id,
            "planning_event": planning_event,
            "events": list(value_events),
            "cutoff": str(value_generation.cutoff),
            "data_snapshot_sha256": str(value_generation.snapshot.get("sha256") or ""),
            "search_permission_evaluation": search_permission_evaluation,
            "source_snapshot_sha256": str(value_generation.manifest.get("code_snapshot_sha256") or ""),
            "model_config_identity": model_config_identity,
            "source_decision_id": route.source_decision_id,
            "route_id": route.route_id,
            "official_pool": pool_binding.as_dict(),
            "projection_run_ids_by_event": {
                str(event): value_generation.runs_for(event) for event in value_events
            },
            "route_input_sha256": route.route_input_sha256,
            "world_cache_keys": {
                str(event): ro.world_cache_key(
                    event=event, generation_id=value_generation.generation_id,
                    runs=value_generation.runs_for(event), config=config,
                    union_ids=tuple(sorted(pool_binding.eligible_ids)),
                ) for event in value_events
            },
        }
    except (wa.WildcardAdapterError, WildcardProductionError):
        raise
    except Exception as failure:
        raise WildcardProductionError(
            f"{WC_PRODUCTION_INPUTS_MISSING}: Wildcard production assembly refused: "
            f"{type(failure).__name__}: {failure}"
        ) from failure
    finally:
        source_conn.close()


def build_future_wildcard_event_opportunity(
    conn: sqlite3.Connection,
    *,
    request: Any,
    source_identity: Mapping[str, Any],
    event: int,
    expiry_event: int,
    chip_generation_id: str,
    value_generation_id: str,
    coverage_product: Mapping[str, Any],
    reservation_state: Mapping[str, Any],
    made_at: str,
) -> dict[str, Any]:
    """Evaluate and retain a future WC opportunity from its canonical request.

    The request is a typed ``WildcardRequest`` built by the production adapter.
    Its four-event SAVE route and 6–10-event value inputs are independently
    re-bound here to verified future-window generations inside the origin-pinned
    continuation product. No event score or arm artifact is accepted from the
    caller.
    """

    from . import chip_wildcard as wc, search_permission as sp

    event = int(event)
    origin_event = int(source_identity.get("planning_event") or -1)
    try:
        permission_evaluation = sp.require_search_permission(
            conn,
            str(source_identity.get("generation_id") or ""),
            expected_origin_planning_event=origin_event,
            expected_origin_cutoff=str(source_identity.get("origin_cutoff") or ""),
            expected_snapshot_sha256=str(source_identity.get("data_snapshot_sha256") or ""),
        )
    except sp.SearchPermissionRefused:
        raise
    except Exception as failure:
        raise sp.SearchPermissionRefused(
            [f"future WC origin permission could not be evaluated: {type(failure).__name__}: {failure}"]
        ) from failure
    value_binding = getattr(request, "value_horizon_binding", None)
    chip_binding = getattr(request, "horizon_binding", None)
    try:
        crf.verify_reservation_coverage_product(
            conn,
            coverage_product,
            expected={
                "action": cd.CHIP_ACTION_WC,
                "planning_event": origin_event,
                "expiry_event": int(expiry_event),
                "source_identity": dict(source_identity),
            },
        )
        chip_generation = gs.load_generation(conn, str(chip_generation_id))
        value_generation = gs.load_generation(conn, str(value_generation_id))
        chip_report = gs.verify_generation(conn, chip_generation.generation_id)
        value_report = gs.verify_generation(conn, value_generation.generation_id)
    except Exception as failure:
        raise WildcardProductionError(
            f"{WC_PRODUCTION_INPUTS_MISSING}: future WC certified products refused: {failure}"
        ) from failure
    if not chip_report.get("verified") or not value_report.get("verified"):
        raise WildcardProductionError(f"{WC_PRODUCTION_INPUTS_MISSING}: a future WC product did not verify")
    product_events = tuple(int(value) for value in coverage_product.get("product_events") or ())
    value_events = tuple(int(value) for value in getattr(getattr(request, "horizon", None), "events", ()))
    chip_events = tuple(int(value) for value in getattr(chip_binding, "horizon_events", ()))
    if (
        event <= origin_event
        or int(getattr(request, "planning_event", -1)) != event
        or chip_generation.horizon_kind != gs.HORIZON_KIND_CHIP_RESERVATION
        or value_generation.horizon_kind != gs.HORIZON_KIND_WILDCARD_VALUE
        or int(chip_generation.planning_event) != origin_event
        or str(chip_generation.generation_id) != str(coverage_product.get("product_generation_id"))
        or tuple(chip_generation.events) != product_events
        or int(value_generation.planning_event) != event
        or chip_events != tuple(range(event, event + 4))
        or not set(chip_events).issubset(set(product_events))
        or tuple(value_generation.events) != value_events
        or not 6 <= len(value_events) <= 10
        or value_events != tuple(range(event, event + len(value_events)))
        or tuple(value_events[:4]) != chip_events
        or not set(value_events).issubset(set(product_events))
        or coverage_product.get("coverage_complete") is not True
        or event not in {int(value) for value in coverage_product.get("forecast_events") or ()}
        or str(value_generation.generation_id) != str(getattr(value_binding, "prediction_generation", ""))
        or str(value_binding.planning_event if value_binding else -1) != str(event)
        or tuple(int(v) for v in (value_binding.event_ids if value_binding else ())) != value_events
        or str(chip_binding.certification_identity if chip_binding else "")
        != gs._certification_identity_for_bundles(
            cutoff=str(chip_generation.cutoff),
            bundle_identities={
                str(e): str(((chip_generation.manifest.get("per_event") or {}).get(str(e)) or {}).get("bundle_identity") or "")
                for e in chip_events
            },
            snapshot_sha256=str(chip_generation.snapshot.get("sha256") or ""),
        )
    ):
        raise WildcardProductionError(
            f"{WC_PRODUCTION_INPUTS_MISSING}: future WC request does not preserve its four-event decision and separate 6–10-event value contracts"
        )
    expected_cutoff = str(source_identity.get("origin_cutoff") or "")
    expected_snapshot = str(source_identity.get("data_snapshot_sha256") or "")
    expected_code = str(source_identity.get("predictive_code_snapshot_sha256") or "")
    continuation_runs = coverage_product.get("product_runs_by_event") or {}
    identity_disagreements = []
    for generation, label in ((chip_generation, "CHIP_RESERVATION"), (value_generation, "WILDCARD_VALUE")):
        if (
            str(generation.cutoff) != expected_cutoff
            or str(generation.snapshot.get("sha256") or "") != expected_snapshot
            or str(generation.manifest.get("code_snapshot_sha256") or "") != expected_code
        ):
            identity_disagreements.append(f"{label} cutoff/snapshot/predictive identity differs from origin")
        for covered_event in generation.events:
            if generation.runs_for(covered_event) != {
                str(k): int(v) for k, v in (continuation_runs.get(str(int(covered_event))) or {}).items()
            }:
                identity_disagreements.append(f"{label} GW{covered_event} runs differ from continuation product")
    chip_per_event = chip_generation.manifest.get("per_event") or {}
    expected_window_identity = gs._certification_identity_for_bundles(
        cutoff=str(chip_generation.cutoff),
        bundle_identities={
            str(covered_event): str((chip_per_event.get(str(covered_event)) or {}).get("bundle_identity") or "")
            for covered_event in chip_events
        },
        snapshot_sha256=str(chip_generation.snapshot.get("sha256") or ""),
    )
    if str(chip_binding.certification_identity if chip_binding else "") != expected_window_identity:
        identity_disagreements.append("future four-event binding differs from the verified continuation slice")
    value_per_event = value_generation.manifest.get("per_event") or {}
    if any(
        value_generation.runs_for(value_event) != chip_generation.runs_for(value_event)
        or (value_per_event.get(str(value_event)) or {}).get("dependency_closure")
        != (chip_per_event.get(str(value_event)) or {}).get("dependency_closure")
        for value_event in value_events
    ):
        identity_disagreements.append("future WILDCARD_VALUE window differs from the origin continuation runs")
    if identity_disagreements:
        raise WildcardProductionError(
            f"{WC_PRODUCTION_INPUTS_MISSING}: " + "; ".join(identity_disagreements[:6])
        )

    evaluation = wc.evaluate_wildcard(request)
    if evaluation.mean_uplift is None or evaluation.candidate_metrics.get("wildcard_squad") is None:
        raise WildcardProductionError(
            f"{WC_PRODUCTION_INPUTS_MISSING}: canonical future Wildcard evaluator refused: "
            + "; ".join(evaluation.reason_codes[:8])
        )
    play_squad = tuple(sorted(int(value) for value in evaluation.candidate_metrics["wildcard_squad"]))
    transaction = wc.plan_transaction(request, play_squad)
    play_prices = {int(value): int(request.purchase_price_tenths[value]) for value in transaction.retained_ids}
    play_prices.update({int(key): int(value) for key, value in transaction.new_basis.items()})
    if set(play_prices) != set(play_squad):
        raise WildcardProductionError(f"{WC_PRODUCTION_INPUTS_MISSING}: WC PLAY acquisition basis is incomplete")
    event_start_ft = request.event_start_free_transfers
    free_transfers_after = evaluation.evidence.get("free_transfers_after_wildcard")
    if event_start_ft is None or free_transfers_after is None:
        raise WildcardProductionError(f"{WC_PRODUCTION_INPUTS_MISSING}: WC event-start FT transition is unknown")
    max_free_transfers = int(request.rules.max_free_transfers)
    play_policy_map = (evaluation.evidence.get("play_now") or {}).get("event_policies") or {}
    if set(str(event_value) for event_value in value_events) - set(play_policy_map):
        raise WildcardProductionError(f"{WC_PRODUCTION_INPUTS_MISSING}: WC PLAY omits a value-horizon event policy")
    play_positions = {str(pid): str(request.players[pid].position) for pid in play_squad}
    play_bank = int(transaction.remaining_bank_tenths)
    play_ft = int(free_transfers_after)
    play_event_points = {
        int(key): float(value)
        for key, value in ((evaluation.evidence.get("play_now") or {}).get("event_points") or {}).items()
    }
    play_schedule: list[dict[str, Any]] = []
    for horizon_event in value_events:
        policy = dict(play_policy_map[str(horizon_event)])
        if horizon_event not in play_event_points:
            raise WildcardProductionError(
                f"{WC_PRODUCTION_INPUTS_MISSING}: WC PLAY omitted expected points for GW{horizon_event}"
            )
        play_state = {
            "squad_ids": list(play_squad), "bank_tenths": play_bank,
            "purchase_price_tenths": {str(k): int(v) for k, v in sorted(play_prices.items())},
            "free_transfers": play_ft, "event_start_free_transfers": play_ft,
            "wildcard_available": False,
        }
        play_schedule.append({
            "event": horizon_event, "weight": float(request.horizon.weight_for(horizon_event)),
            "proposed_squad_ids": list(play_squad), "lineup": policy,
            "player_positions": play_positions, "transfer_state": play_state,
            "expected_points": play_event_points[horizon_event],
            "wildcard_active": horizon_event == event,
        })
        if horizon_event != value_events[-1]:
            play_ft = sr.free_transfers_after_gameweek(request.rules, play_ft, 0)

    save_route = request.save_route
    if save_route is None:
        raise WildcardProductionError(f"{WC_PRODUCTION_INPUTS_MISSING}: WC SAVE route is missing")
    save_route_events = {int(row.event): row for row in save_route.events}
    terminal_squad = tuple(int(value) for value in save_route.terminal_squad_ids)
    terminal_prices = {int(k): int(v) for k, v in save_route.terminal_purchase_price_tenths.items()}
    terminal_bank = int(save_route.terminal_bank_tenths)
    save_ft = int(save_route.terminal_free_transfers)
    save_schedule: list[dict[str, Any]] = []
    save_positions_by_event: dict[int, dict[str, str]] = {}
    for horizon_event in value_events:
        if horizon_event in save_route_events:
            route_row = save_route_events[horizon_event]
            squad = tuple(int(value) for value in route_row.squad_ids)
            bank = int(route_row.bank_tenths)
            prices = {int(k): int(v) for k, v in route_row.purchase_price_tenths.items()}
            free_transfers = int(route_row.free_transfers)
            policy = dict(route_row.policy or {})
            if not policy:
                raise WildcardProductionError(
                    f"{WC_PRODUCTION_INPUTS_MISSING}: canonical WC SAVE route omitted its lineup at GW{horizon_event}"
                )
            source_label = "CANONICAL_NORMAL_ROUTE"
            expected_points = float(route_row.mean_net_core)
        else:
            squad = terminal_squad
            bank = terminal_bank
            prices = terminal_prices
            free_transfers = save_ft
            policy_obj = wc._event_policy(request, squad, horizon_event)
            if policy_obj is None:
                raise WildcardProductionError(
                    f"{WC_PRODUCTION_INPUTS_MISSING}: carried WC SAVE terminal state has no legal lineup at GW{horizon_event}"
                )
            policy = {
                "starter_ids": list(policy_obj.starter_ids),
                "bench_gk_id": int(policy_obj.bench_gk_id),
                "bench_outfield_order": list(policy_obj.bench_outfield_order),
                "captain_id": int(policy_obj.captain_id),
                "vice_captain_id": int(policy_obj.vice_captain_id),
            }
            source_label = "CARRIED_CANONICAL_TERMINAL_STATE"
            expected_points = float(wc._event_value(
                request, squad, horizon_event,
                policy=wc.manager_lineup.ManagerPolicy(
                    starter_ids=tuple(int(value) for value in policy["starter_ids"]),
                    bench_gk_id=int(policy["bench_gk_id"]),
                    bench_outfield_order=tuple(int(value) for value in policy["bench_outfield_order"]),
                    captain_id=int(policy["captain_id"]),
                    vice_captain_id=int(policy["vice_captain_id"]),
                ),
            ))
            save_ft = sr.free_transfers_after_gameweek(request.rules, save_ft, 0)
        positions = {str(pid): str(request.players[pid].position) for pid in squad}
        save_positions_by_event[horizon_event] = positions
        transfer_state = {
            "squad_ids": list(squad), "bank_tenths": bank,
            "purchase_price_tenths": {str(k): int(v) for k, v in sorted(prices.items())},
            "free_transfers": free_transfers,
            "event_start_free_transfers": free_transfers,
            "wildcard_available": True,
        }
        save_schedule.append({
            "event": horizon_event, "weight": float(request.horizon.weight_for(horizon_event)),
            "proposed_squad_ids": list(squad), "lineup": policy,
            "player_positions": positions, "transfer_state": transfer_state,
            "expected_points": expected_points,
            "state_source": source_label,
        })

    first_play, first_save = play_schedule[0], save_schedule[0]
    play_state = dict(first_play["transfer_state"])
    save_state = dict(first_save["transfer_state"])
    play_value = sum(float(row["weight"]) * float(row["expected_points"]) for row in play_schedule)
    save_value = sum(float(row["weight"]) * float(row["expected_points"]) for row in save_schedule)
    if (
        abs(play_value - float(evaluation.candidate_metrics.get("play_now_objective"))) > 1e-6
        or abs(save_value - float(evaluation.candidate_metrics.get("save_objective"))) > 1e-6
        or abs(play_value - save_value - float(evaluation.mean_uplift)) > 1e-6
    ):
        raise WildcardProductionError(
            f"{WC_PRODUCTION_INPUTS_MISSING}: retained WC schedules do not reproduce canonical weighted-horizon value"
        )
    origin_manager_state = {
        "event": event,
        "squad_ids": sorted(int(value) for value in request.owned_ids),
        "bank_tenths": int(request.bank_tenths),
        "free_transfers": int(event_start_ft),
        "purchase_price_tenths": {
            str(int(key)): int(value)
            for key, value in sorted(request.purchase_price_tenths.items())
        },
    }
    if dict(reservation_state) != save_state:
        raise WildcardProductionError(
            f"{WC_PRODUCTION_INPUTS_MISSING}: supplied future SAVE state differs from the canonical WC route state"
        )
    world_identity = crf.canonical_sha256({
        "source_identity": dict(source_identity),
        "coverage_product_sha256": coverage_product.get("product_sha256"),
        "chip_generation_id": chip_generation.generation_id,
        "value_generation_id": value_generation.generation_id,
        "value_binding": value_binding.as_dict(),
        "event_worlds": {
            str(e): request.worlds_by_event[e].identity.as_dict() for e in value_events
        },
    })
    scenario_identity = crf.canonical_sha256({
        "action": cd.CHIP_ACTION_WC, "event": event,
        "source_identity": dict(source_identity), "world_identity": world_identity,
        "play_schedule": play_schedule, "save_schedule": save_schedule,
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
        "product_generation_id": str(value_generation.generation_id),
        "coverage_product_sha256": str(coverage_product["product_sha256"]),
    }
    arms = {
        "play": {
            "arm_id": f"wc-play-{crf.canonical_sha256([scenario_identity, 'PLAY'])[:16]}",
            "action": cd.CHIP_ACTION_WC, "event": event,
            "scenario_identity": scenario_identity, "world_identity": world_identity,
            "counterfactual_role": "PLAY", "proposed_squad_ids": list(play_squad),
            "lineup": first_play["lineup"], "player_positions": play_positions,
            "reservation_state": save_state, "scoring_rule_version": crc.OUTCOME_SCORING_RULE_VERSION,
            "paired_value": play_value, "valuation_schedule": {
                "value_definition": "WEIGHTED_WILDCARD_HORIZON_MEAN_POINTS",
                "events": play_schedule,
            },
            "action_semantics": {
                "role": "PLAY", "transfer_state": play_state,
                "origin_manager_state": origin_manager_state,
                "origin_manager_state_sha256": crf.canonical_sha256(origin_manager_state),
                "wildcard_applied": True,
                "transaction": {
                    "retained_ids": list(transaction.retained_ids),
                    "sold_ids": list(transaction.sold_ids),
                    "bought_ids": list(transaction.bought_ids),
                    "rebought_ids": list(transaction.rebought_ids),
                    "new_basis": {
                        str(int(key)): int(value)
                        for key, value in sorted(transaction.new_basis.items())
                    },
                    "remaining_bank_tenths": int(transaction.remaining_bank_tenths),
                },
                "event_start_free_transfers": int(event_start_ft),
                "free_transfers_after": int(free_transfers_after),
                "max_free_transfers": max_free_transfers,
                "valuation_schedule_required": True,
            },
            **source_fields,
        },
        "save": {
            "arm_id": f"wc-save-{crf.canonical_sha256([scenario_identity, 'SAVE'])[:16]}",
            "action": cd.CHIP_ACTION_WC, "event": event,
            "scenario_identity": scenario_identity, "world_identity": world_identity,
            "counterfactual_role": "SAVE", "proposed_squad_ids": first_save["proposed_squad_ids"],
            "lineup": first_save["lineup"], "player_positions": first_save["player_positions"],
            "reservation_state": save_state, "scoring_rule_version": crc.OUTCOME_SCORING_RULE_VERSION,
            "paired_value": save_value, "valuation_schedule": {
                "value_definition": "WEIGHTED_WILDCARD_HORIZON_MEAN_POINTS",
                "events": save_schedule,
            },
            "action_semantics": {
                "role": "SAVE", "transfer_state": save_state,
                "origin_manager_state": origin_manager_state,
                "origin_manager_state_sha256": crf.canonical_sha256(origin_manager_state),
                "wildcard_applied": False, "retains_chip_option": True,
                "valuation_schedule_required": True,
            },
            **source_fields,
        },
    }
    record = crf.build_event_opportunity_record(
        action=cd.CHIP_ACTION_WC, planning_event=origin_event, event=event,
        origin_cutoff=str(source_identity["origin_cutoff"]),
        input_as_of=str(source_identity["origin_cutoff"]), made_at=made_at,
        expected_incremental_points=float(evaluation.mean_uplift),
        opportunity_model=f"{evaluation.evaluator_version}:WEIGHTED_HORIZON_UPLIFT_V1",
        source_identity=source_identity, reservation_state=save_state,
        world_identity=world_identity, outcome_arms=arms,
        coverage_product_sha256=str(coverage_product["product_sha256"]),
        evaluator_identity={
            "evaluator_version": str(evaluation.evaluator_version),
            "action": cd.CHIP_ACTION_WC, "event": event,
            "expected_incremental_points": float(evaluation.mean_uplift),
            "uncertainty": dict(evaluation.uncertainty),
            "opportunity_value_definition": "WEIGHTED_WC_HORIZON_MEAN_POINTS",
            "uncertainty_value_definition": "WEIGHTED_WC_HORIZON_MEAN_POINTS",
            "decision_horizon_events": list(chip_events),
            "value_horizon_events": list(value_events),
            "world_identity": world_identity,
            "source_identity": dict(source_identity),
            "opportunity_value_definition": "WEIGHTED_WC_HORIZON_MEAN_POINTS",
        },
    )
    return crf._attach_search_permission_evaluation(record, permission_evaluation)
