"""Production assembly for Wildcard's distinct 6–10 event value product."""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import fields
from pathlib import Path
from typing import Any, Mapping

from . import analytics, candidate_universe as cu, generation_store as gs
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
