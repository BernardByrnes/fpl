"""Content-addressed, point-in-time forecasts for future chip opportunity.

This layer produces the raw opportunity forecast consumed by reservation
calibration. It is intentionally separate from fitting or applying calibration.
Per-event opportunity payloads must come from a chip evaluator bound to the
verified generation and the exact SAVE state. Incomplete coverage through the
known chip expiry remains unknown; it is never filled with zero.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from . import chip_decision as cd

EVENT_OPPORTUNITY_SCHEMA = "fpl_brain.chip_event_opportunity_forecast.v2"
RESERVATION_FORECAST_SCHEMA = "fpl_brain.chip_reservation_forecast.v2"
RESERVATION_COVERAGE_PRODUCT_SCHEMA = "fpl_brain.chip_reservation_coverage_product.v2"
RESERVATION_FORECAST_VERSION = "chip_reservation_forecast_v2.0.0"
RESERVATION_FORECAST_MODEL = "max_origin_expected_opportunity_through_expiry_v3"
FORECAST_MODE_PROSPECTIVE = "PROSPECTIVE"
FORECAST_MODE_HISTORICAL_REPLAY = "HISTORICAL_REPLAY"
FORECAST_VALUE_UNITS = cd.CHIP_COMPARISON_BASIS
FORECAST_READY = "COMPLETE_THROUGH_EXPIRY"
FORECAST_INCOMPLETE = "INCOMPLETE_COVERAGE"
FORECAST_IDENTITY_INVALID = "CHIP_RESERVATION_FORECAST_IDENTITY_INVALID"
FORECAST_PUBLICATION_UNAVAILABLE = "CHIP_RESERVATION_FORECAST_ATOMIC_PUBLICATION_UNAVAILABLE"

SOURCE_IDENTITY_FIELDS = (
    "source_decision_id",
    "source_result_sha256",
    "source_artifact_sha256",
    "generation_id",
    "planning_event",
    "origin_cutoff",
    "data_snapshot_sha256",
    "predictive_code_snapshot_sha256",
    "certification_identity",
)


class ReservationForecastError(ValueError):
    """A raw reservation forecast is incomplete, inconsistent or tampered."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False, default=str,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _is_digest(value: Any) -> bool:
    text = str(value or "").removeprefix("sha256:")
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text.lower())


def _time(value: Any, *, name: str) -> str:
    from datetime import datetime, timezone

    text = str(value or "")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as failure:
        raise ReservationForecastError(f"{name} is not an ISO timestamp") from failure
    if parsed.tzinfo is None:
        raise ReservationForecastError(f"{name} must include a timezone")
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _number(value: Any, *, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as failure:
        raise ReservationForecastError(f"{name} is not numeric") from failure
    if not math.isfinite(number):
        raise ReservationForecastError(f"{name} must be finite")
    return number


def build_reservation_coverage_product(
    conn: Any,
    *,
    action: str,
    source_identity: Mapping[str, Any],
    expiry_event: int | None,
    product_generation_id: str,
    wildcard_value_horizon_length: int | None = None,
) -> dict[str, Any]:
    """Bind expiry coverage to a separately verified origin-pinned generation.

    The normal decision generation remains FOUR_GW. BB/TC need one-event
    matrices through expiry; FH needs a four-event window for every future play;
    WC needs its independent 6-10 event value window for each future play. The
    separate CHIP_RESERVATION generation must reuse the normal four-event run
    prefix and the exact cutoff, snapshot and predictive code identity.
    """

    if action not in cd.PLAYABLE_CHIP_ACTIONS:
        raise ReservationForecastError("coverage product has an unknown chip action")
    if action == cd.CHIP_ACTION_FH:
        horizon_length = cd.CHIP_HORIZON_LENGTH
    elif action == cd.CHIP_ACTION_WC:
        from . import chip_wildcard as wc

        horizon_length = int(wildcard_value_horizon_length or 0)
        if not wc.WILDCARD_HORIZON_MIN_EVENTS <= horizon_length <= wc.WILDCARD_HORIZON_MAX_EVENTS:
            raise ReservationForecastError(
                "WC coverage product must retain its separate 6-10 event value horizon"
            )
    else:
        horizon_length = 1
    origin_event = int(source_identity.get("planning_event") or 0)
    if expiry_event is None:
        required_last = origin_event - 1
        forecast_events: tuple[int, ...] = ()
    else:
        expiry_event = int(expiry_event)
        if expiry_event < origin_event:
            raise ReservationForecastError("chip expiry cannot precede its planning event")
        required_last = (
            origin_event if expiry_event == origin_event
            else expiry_event + horizon_length - 1
        )
        forecast_events = (
            () if expiry_event == origin_event
            else tuple(range(origin_event + 1, expiry_event + 1))
        )

    from . import generation_store as gs

    try:
        base = gs.load_generation(conn, str(source_identity.get("generation_id") or ""))
        product = gs.load_generation(conn, str(product_generation_id))
        base_report = gs.verify_generation(conn, base.generation_id)
        product_report = gs.verify_generation(conn, product.generation_id)
    except Exception as failure:
        raise ReservationForecastError(f"verified reservation product could not be loaded: {failure}") from failure
    if not base_report.get("verified") or base.horizon_kind != gs.HORIZON_KIND_FOUR_GW:
        raise ReservationForecastError("reservation coverage root is not a verified FOUR_GW generation")
    if not product_report.get("verified") or product.horizon_kind != gs.HORIZON_KIND_CHIP_RESERVATION:
        raise ReservationForecastError("expiry coverage requires a verified CHIP_RESERVATION generation")
    origin = int(source_identity.get("planning_event") or -1)
    if (
        int(base.planning_event) != origin
        or int(product.planning_event) != origin
        or tuple(int(event) for event in base.events) != tuple(int(event) for event in product.events[:4])
        or any(str(value) != str(base.cutoff) for value in (
            product.cutoff, source_identity.get("origin_cutoff"),
        ))
        or str(base.snapshot.get("sha256")) != str(product.snapshot.get("sha256"))
        or str(base.snapshot.get("sha256")) != str(source_identity.get("data_snapshot_sha256"))
        or str(base.manifest.get("code_snapshot_sha256")) != str(product.manifest.get("code_snapshot_sha256"))
        or str(base.manifest.get("code_snapshot_sha256"))
        != str(source_identity.get("predictive_code_snapshot_sha256"))
    ):
        raise ReservationForecastError(
            "reservation product differs from the origin in event prefix, cutoff, snapshot or predictive identity"
        )
    prefix_disagreements = [
        int(event) for event in base.events
        if base.runs_for(int(event)) != product.runs_for(int(event))
    ]
    if prefix_disagreements:
        raise ReservationForecastError(
            f"reservation product does not reuse the exact normal run prefix for {prefix_disagreements}"
        )
    product_events = tuple(int(event) for event in product.events)
    required_events = (
        [] if expiry_event is None
        else list(range(origin, required_last + 1))
    )
    missing = [event for event in required_events if event not in product_events]
    coverage_known = expiry_event is not None
    coverage_complete = bool(coverage_known and not missing)
    body = {
        "schema": RESERVATION_COVERAGE_PRODUCT_SCHEMA,
        "action": action,
        "planning_event": origin,
        "expiry_event": None if expiry_event is None else int(expiry_event),
        "forecast_events": list(forecast_events),
        "opportunity_horizon_length": int(horizon_length),
        "required_product_events": required_events,
        "product_events": list(product_events),
        "missing_events": missing,
        "coverage_known": coverage_known,
        "coverage_complete": coverage_complete,
        "coverage_status": FORECAST_READY if coverage_complete else FORECAST_INCOMPLETE,
        "coverage_reason": (
            None if coverage_complete
            else "CHIP_EXPIRY_UNKNOWN" if not coverage_known
            else "REQUIRED_PRODUCT_EVENTS_MISSING"
        ),
        "source_identity": dict(source_identity),
        "root_generation_id": str(base.generation_id),
        "product_generation_id": str(product.generation_id),
        "product_generation_manifest_sha256": str(product.generation_id),
        "origin_cutoff": str(base.cutoff),
        "input_as_of": str(base.cutoff),
        "data_snapshot_sha256": str(base.snapshot.get("sha256") or ""),
        "predictive_code_snapshot_sha256": str(base.manifest.get("code_snapshot_sha256") or ""),
        "product_runs_by_event": {
            str(event): product.runs_for(event) for event in product_events
        },
        "wildcard_value_horizon_length": (
            int(horizon_length) if action == cd.CHIP_ACTION_WC else None
        ),
    }
    body["product_sha256"] = canonical_sha256(body)
    return body


def verify_reservation_coverage_product(
    conn: Any,
    product: Mapping[str, Any],
    *,
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    """Re-load both generations and reproduce an expiry coverage product."""

    if product.get("schema") != RESERVATION_COVERAGE_PRODUCT_SCHEMA:
        raise ReservationForecastError("reservation coverage product schema is unsupported")
    body = dict(product)
    identity = str(body.pop("product_sha256", ""))
    if len(identity) != 64 or identity != canonical_sha256(body):
        raise ReservationForecastError("reservation coverage product digest does not verify")
    for key, value in expected.items():
        if product.get(key) != value:
            raise ReservationForecastError(f"coverage product disagrees with requested {key}")
    reproduced = build_reservation_coverage_product(
        conn,
        action=str(product.get("action") or ""),
        source_identity=product.get("source_identity") or {},
        expiry_event=product.get("expiry_event"),
        product_generation_id=str(product.get("product_generation_id") or ""),
        wildcard_value_horizon_length=product.get("wildcard_value_horizon_length"),
    )
    if reproduced != dict(product):
        raise ReservationForecastError("reservation coverage product does not reproduce from verified generations")
    return {"verified": True, "product_sha256": identity}


def _verify_coverage_product_payload(
    product: Mapping[str, Any],
    *,
    action: str,
    planning_event: int,
    expiry_event: int | None,
    source_identity: Mapping[str, Any],
    input_as_of: str,
) -> tuple[str, bool]:
    """Check the self-contained coverage contract; DB verification is separate."""

    if product.get("schema") != RESERVATION_COVERAGE_PRODUCT_SCHEMA:
        raise ReservationForecastError("reservation coverage product schema is unsupported")
    body = dict(product)
    identity = str(body.pop("product_sha256", ""))
    if not _is_digest(identity) or identity != canonical_sha256(body):
        raise ReservationForecastError("reservation coverage product digest does not verify")
    if action == cd.CHIP_ACTION_FH:
        horizon_length = cd.CHIP_HORIZON_LENGTH
    elif action == cd.CHIP_ACTION_WC:
        from . import chip_wildcard as wc

        horizon_length = int(product.get("wildcard_value_horizon_length") or 0)
        if not wc.WILDCARD_HORIZON_MIN_EVENTS <= horizon_length <= wc.WILDCARD_HORIZON_MAX_EVENTS:
            raise ReservationForecastError("WC coverage product omits its supported 6–10 event horizon")
    else:
        horizon_length = 1
    expected_forecast_events = (
        [] if expiry_event is None or int(expiry_event) <= int(planning_event)
        else list(range(int(planning_event) + 1, int(expiry_event) + 1))
    )
    if expiry_event is None:
        required_product_events: list[int] = []
    elif int(expiry_event) == int(planning_event):
        required_product_events = [int(planning_event)]
    else:
        required_last = int(expiry_event) + horizon_length - 1
        required_product_events = list(range(int(planning_event), required_last + 1))
    try:
        product_events = [int(value) for value in product.get("product_events") or ()]
        forecast_events = [int(value) for value in product.get("forecast_events") or ()]
        missing = [int(value) for value in product.get("missing_events") or ()]
    except (TypeError, ValueError) as failure:
        raise ReservationForecastError("reservation coverage product event lists are malformed") from failure
    if (
        product.get("action") != action
        or int(product.get("planning_event") or -1) != int(planning_event)
        or product.get("expiry_event") != (None if expiry_event is None else int(expiry_event))
        or dict(product.get("source_identity") or {}) != dict(source_identity)
        or product.get("origin_cutoff") != str(source_identity.get("origin_cutoff"))
        or product.get("input_as_of") != str(input_as_of)
        or int(product.get("opportunity_horizon_length") or 0) != horizon_length
        or product.get("root_generation_id") != str(source_identity.get("generation_id"))
        or product.get("product_generation_id") != product.get("product_generation_manifest_sha256")
        or not str(product.get("product_generation_id") or "")
        or product.get("data_snapshot_sha256") != str(source_identity.get("data_snapshot_sha256"))
        or product.get("predictive_code_snapshot_sha256")
        != str(source_identity.get("predictive_code_snapshot_sha256"))
        or forecast_events != expected_forecast_events
        or product_events != sorted(set(product_events))
        or (product_events and product_events != list(range(product_events[0], product_events[-1] + 1)))
        or (not missing and not set(required_product_events).issubset(set(product_events)))
        or missing != [event for event in required_product_events if event not in product_events]
        or bool(product.get("coverage_known")) != (expiry_event is not None)
        or bool(product.get("coverage_complete")) != (expiry_event is not None and not missing)
        or product.get("coverage_status") != (
            FORECAST_READY if expiry_event is not None and not missing else FORECAST_INCOMPLETE
        )
        or product.get("coverage_reason") != (
            None if expiry_event is not None and not missing
            else "CHIP_EXPIRY_UNKNOWN" if expiry_event is None
            else "REQUIRED_PRODUCT_EVENTS_MISSING"
        )
    ):
        raise ReservationForecastError("reservation coverage product does not match its origin/expiry contract")
    run_map = product.get("product_runs_by_event")
    if not isinstance(run_map, Mapping) or {int(key) for key in run_map} != set(product_events):
        raise ReservationForecastError("reservation coverage product run manifest does not cover its certified events")
    if action == cd.CHIP_ACTION_WC:
        if int(product.get("wildcard_value_horizon_length") or 0) != horizon_length:
            raise ReservationForecastError("WC coverage product value horizon does not reproduce")
    elif product.get("wildcard_value_horizon_length") is not None:
        raise ReservationForecastError("non-WC coverage product declares a Wildcard value horizon")
    return identity, bool(product.get("coverage_complete"))


def build_event_opportunity_record(
    *,
    action: str,
    planning_event: int,
    event: int,
    origin_cutoff: str,
    made_at: str,
    expected_incremental_points: float,
    opportunity_model: str,
    source_identity: Mapping[str, Any],
    reservation_state: Mapping[str, Any],
    world_identity: str,
    outcome_arms: Mapping[str, Mapping[str, Any]] | None = None,
    input_as_of: str | None = None,
    forecast_mode: str = FORECAST_MODE_PROSPECTIVE,
    value_units: str = FORECAST_VALUE_UNITS,
    coverage_product_sha256: str | None = None,
    evaluator_identity: Mapping[str, Any] | None = None,
    continuation_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Retain one evaluator-produced future event opportunity at the origin.

    ``expected_incremental_points`` must be a model forecast computed from the
    named certified world input, not a later realized score. The source and
    exact SAVE state are embedded so that consumers can reproduce the binding.
    """

    if action not in cd.PLAYABLE_CHIP_ACTIONS:
        raise ReservationForecastError("event opportunity has an unknown chip action")
    planning_event, event = int(planning_event), int(event)
    if event <= planning_event:
        raise ReservationForecastError("reservation opportunity must be for a future event")
    cutoff = _time(origin_cutoff, name="origin_cutoff")
    made = _time(made_at, name="made_at")
    as_of = _time(input_as_of or cutoff, name="input_as_of")
    if as_of > cutoff:
        raise ReservationForecastError("event opportunity inputs are later than the origin cutoff")
    if made < as_of:
        raise ReservationForecastError("event opportunity issuance precedes its input-as-of time")
    if forecast_mode not in {FORECAST_MODE_PROSPECTIVE, FORECAST_MODE_HISTORICAL_REPLAY}:
        raise ReservationForecastError("event opportunity forecast mode is unsupported")
    if value_units != FORECAST_VALUE_UNITS:
        raise ReservationForecastError("event opportunity uses incompatible comparison value units")
    source = dict(source_identity)
    for name in SOURCE_IDENTITY_FIELDS:
        if not str(source.get(name) or "").strip():
            raise ReservationForecastError(f"event opportunity omits source identity {name}")
    if int(source["planning_event"]) != planning_event or _time(
        source["origin_cutoff"], name="source origin_cutoff"
    ) != cutoff:
        raise ReservationForecastError("event opportunity source does not match its planning origin")
    for name in ("source_result_sha256", "source_artifact_sha256", "data_snapshot_sha256",
                 "predictive_code_snapshot_sha256", "certification_identity"):
        digest = str(source[name]).removeprefix("sha256:")
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest.lower()):
            raise ReservationForecastError(f"event opportunity source {name} is not a SHA-256 identity")
    world = str(world_identity or "")
    if not world:
        raise ReservationForecastError("event opportunity has no certified world identity")
    state = dict(reservation_state)
    if not state:
        raise ReservationForecastError("event opportunity has no explicit SAVE state")
    if not str(opportunity_model or "").strip():
        raise ReservationForecastError("event opportunity has no model identity")
    body = {
        "schema": EVENT_OPPORTUNITY_SCHEMA,
        "action": action,
        "planning_event": planning_event,
        "event": event,
        "origin_cutoff": cutoff,
        "input_as_of": as_of,
        "made_at": made,
        "forecast_mode": forecast_mode,
        "value_units": value_units,
        "expected_incremental_points": _number(
            expected_incremental_points, name="expected_incremental_points",
        ),
        "opportunity_model": str(opportunity_model),
        "source_identity": source,
        "reservation_state": state,
        "reservation_state_sha256": canonical_sha256(state),
        "world_identity": world,
        "forecast_kind": "POINT_IN_TIME_EXPECTATION",
    }
    if coverage_product_sha256 is not None:
        digest = str(coverage_product_sha256).removeprefix("sha256:")
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest.lower()):
            raise ReservationForecastError("event opportunity coverage product identity is not a SHA-256")
        body["coverage_product_sha256"] = digest
    if evaluator_identity is not None:
        identity = dict(evaluator_identity)
        evaluator_version = str(identity.get("evaluator_version") or "").strip()
        if not evaluator_version:
            raise ReservationForecastError("event opportunity evaluator identity omits evaluator_version")
        if identity.get("action") not in (None, action) or identity.get("event") not in (None, event):
            raise ReservationForecastError("event opportunity evaluator identity has another action/event")
        if "expected_incremental_points" in identity and _number(
            identity["expected_incremental_points"], name="evaluator expected_incremental_points",
        ) != _number(expected_incremental_points, name="expected_incremental_points"):
            raise ReservationForecastError("event opportunity evaluator identity disagrees with its value")
        identity.update({
            "action": action,
            "event": event,
            "expected_incremental_points": _number(
                expected_incremental_points, name="expected_incremental_points",
            ),
        })
        body["evaluator_identity"] = identity
    if continuation_context is not None:
        context = dict(continuation_context)
        context_digest = str(context.pop("context_sha256", ""))
        if not _is_digest(context_digest) or context_digest != canonical_sha256(context):
            raise ReservationForecastError("continuation event context digest does not verify")
        if (
            context.get("schema") != "fpl_brain.chip_reservation_continuation_event.v1"
            or context.get("model") != "CARRY_TERMINAL_ROUTE_STATE_NO_TRANSFERS_PER_EVENT_LINEUP_V1"
            or context.get("action") != action
            or int(context.get("event") or -1) != event
            or str(context.get("coverage_product_sha256") or "")
            != str(coverage_product_sha256 or "")
            or str(context.get("world_identity") or "") != world
        ):
            raise ReservationForecastError("continuation event context differs from the opportunity identity")
        event_state = context.get("event_manager_state")
        event_policy = context.get("event_policy")
        event_positions = context.get("player_positions")
        season_rules = context.get("season_rules")
        season_rules_evidence = context.get("season_rules_evidence")
        try:
            from . import season_rules as sr

            if not isinstance(season_rules, Mapping) or not isinstance(season_rules_evidence, Mapping):
                raise ValueError("season rules or their provenance are absent")
            resolved_rules = sr.SeasonRules(**dict(season_rules))
            sr.verify_pinned_season_rules_evidence(
                season_rules_evidence,
                resolved_rules,
                cutoff=str(origin_cutoff),
                data_snapshot_sha256=str(source_identity.get("data_snapshot_sha256") or ""),
                allow_fixture=True,
            )
        except Exception as failure:
            raise ReservationForecastError(
                "continuation event season-rule identity is absent or invalid"
            ) from failure
        if (
            context.get("season_rules_source") != season_rules_evidence.get("source")
            or str(context.get("season_rules_sha256") or "") != canonical_sha256(season_rules)
        ):
            raise ReservationForecastError("continuation event season-rule identity is absent or invalid")
        if not isinstance(event_state, Mapping) or not isinstance(event_policy, Mapping) or not isinstance(event_positions, Mapping):
            raise ReservationForecastError("continuation event context omits its manager state or policy")
        squad = {int(value) for value in event_state.get("squad_ids") or ()}
        basis = {int(key): int(value) for key, value in (event_state.get("purchase_price_tenths") or {}).items()}
        if (
            int(event_state.get("event") or -1) != event
            or len(squad) != 15
            or set(basis) != squad
            or int(event_state.get("bank_tenths", -1)) < 0
            or int(event_state.get("free_transfers", -1)) < 0
            or {int(key) for key in event_positions} != squad
        ):
            raise ReservationForecastError("continuation event manager state is incomplete or malformed")
        for arm in (dict(outcome_arms or {}).get("play"), dict(outcome_arms or {}).get("save")):
            if not isinstance(arm, Mapping):
                raise ReservationForecastError("continuation event omits a paired outcome arm")
            if (
                arm.get("continuation_context_sha256") != context_digest
                or arm.get("event_manager_state") != dict(event_state)
                or arm.get("lineup") != dict(event_policy)
                or arm.get("player_positions") != dict(event_positions)
                or {int(value) for value in arm.get("proposed_squad_ids") or ()} != squad
            ):
                raise ReservationForecastError("continuation event arm differs from its retained state/policy")
        context["context_sha256"] = context_digest
        body["continuation_context"] = context
    if outcome_arms is not None:
        if set(outcome_arms) != {"play", "save"}:
            raise ReservationForecastError("event opportunity outcome arms must contain PLAY and SAVE")
        if any(not isinstance(arm, Mapping) for arm in outcome_arms.values()):
            raise ReservationForecastError("event opportunity outcome arm is malformed")
        arms = {name: dict(arm) for name, arm in outcome_arms.items()}
        from . import manager_lineup as ml
        from .chip_reservation_calibration import OUTCOME_SCORING_RULE_VERSION

        for role, arm in (("PLAY", arms["play"]), ("SAVE", arms["save"])):
            if (
                arm.get("action") != action
                or arm.get("event") != event
                or arm.get("counterfactual_role") != role
                or arm.get("world_identity") != world
                or arm.get("scenario_identity") != arms["play"].get("scenario_identity")
                or arm.get("reservation_state") != state
                or arm.get("scoring_rule_version") != OUTCOME_SCORING_RULE_VERSION
            ):
                raise ReservationForecastError(
                    f"event opportunity {role} outcome arm is not bound to its action/event/world/SAVE state"
                )
            for arm_key, source_key in (
                ("source_decision_id", "source_decision_id"),
                ("source_result_sha256", "source_result_sha256"),
                ("source_artifact_sha256", "source_artifact_sha256"),
                ("generation_id", "generation_id"),
                ("cutoff", "origin_cutoff"),
                ("data_snapshot_sha256", "data_snapshot_sha256"),
                ("predictive_code_snapshot_sha256", "predictive_code_snapshot_sha256"),
                ("certification_identity", "certification_identity"),
            ):
                if str(arm.get(arm_key) or "") != str(source[source_key] or ""):
                    raise ReservationForecastError(
                        f"event opportunity {role} outcome arm differs from source identity on {arm_key}"
                    )
            try:
                squad = {int(value) for value in arm["proposed_squad_ids"]}
                lineup = arm["lineup"]
                positions = {int(key): str(value) for key, value in arm["player_positions"].items()}
                policy = ml.ManagerPolicy(
                    starter_ids=tuple(int(value) for value in lineup["starter_ids"]),
                    bench_gk_id=int(lineup["bench_gk_id"]),
                    bench_outfield_order=tuple(int(value) for value in lineup["bench_outfield_order"]),
                    captain_id=int(lineup["captain_id"]),
                    vice_captain_id=int(lineup["vice_captain_id"]),
                )
            except (AttributeError, KeyError, TypeError, ValueError) as failure:
                raise ReservationForecastError(
                    f"event opportunity {role} outcome policy is malformed: {failure}"
                ) from failure
            if squad != set(ml.policy_player_ids(policy)) or positions.keys() != squad:
                raise ReservationForecastError(
                    f"event opportunity {role} outcome policy does not cover its proposed squad"
                )
            legality = ml.policy_legality_errors(policy, positions)
            if legality:
                raise ReservationForecastError(
                    f"event opportunity {role} outcome policy is illegal: {legality}"
                )
            _number(arm.get("paired_value"), name=f"{role} paired value")
        if (
            arms["play"].get("scenario_identity") in (None, "")
        ):
            raise ReservationForecastError("event outcome arms have no shared scenario identity")
        if action in {cd.CHIP_ACTION_BB, cd.CHIP_ACTION_TC} and (
            arms["play"].get("proposed_squad_ids") != arms["save"].get("proposed_squad_ids")
            or arms["play"].get("lineup") != arms["save"].get("lineup")
            or arms["play"].get("player_positions") != arms["save"].get("player_positions")
        ):
            raise ReservationForecastError("BB/TC event outcome arms do not share one scenario policy")
        if action in {cd.CHIP_ACTION_FH, cd.CHIP_ACTION_WC}:
            from .chip_reservation_calibration import _verify_action_pair_policies

            try:
                _verify_action_pair_policies(
                    action, arms["play"], arms["save"], planning_event=planning_event,
                )
            except Exception as failure:
                raise ReservationForecastError(
                    f"{action} event outcome arms violate their action-specific state semantics: {failure}"
                ) from failure
            scheduled = any(
                isinstance(arm.get("valuation_schedule"), Mapping)
                or (arm.get("action_semantics") or {}).get("valuation_schedule_required") is True
                for arm in arms.values()
            )
            if scheduled:
                product_digest = str(coverage_product_sha256 or "")
                if not _is_digest(product_digest) or any(
                    str(arm.get("coverage_product_sha256") or "") != product_digest
                    for arm in arms.values()
                ):
                    raise ReservationForecastError(
                        f"{action} production arms must bind the verified expiry-coverage product"
                    )
        body["outcome_arms"] = arms
    body["artifact_sha256"] = canonical_sha256(body)
    return body


def build_evaluated_event_opportunity_record(
    *,
    action: str,
    event: int,
    worlds: cd.ChipWorldInputs | None = None,
    horizon_binding: cd.ChipHorizonBinding | None = None,
    policy: Any | None = None,
    positions: Mapping[int, str] | None = None,
    source_identity: Mapping[str, Any],
    reservation_state: Mapping[str, Any],
    made_at: str,
    input_as_of: str | None = None,
    coverage_product_sha256: str | None = None,
    conn: Any | None = None,
    expiry_event: int | None = None,
    coverage_product: Mapping[str, Any] | None = None,
    action_request: Any | None = None,
    chip_generation_id: str | None = None,
    value_generation_id: str | None = None,
    continuation_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Produce a canonical event opportunity and its maturation arms.

    The normal-route SAVE policy and chip PLAY policy are scored by the existing
    production evaluator on a certified matrix explicitly tagged for ``event``.
    FH/WC dispatch to their dedicated production producers, which consume typed
    requests already assembled from verified chip-specific routes/generations.
    """

    from . import search_permission as sp

    if conn is None:
        raise sp.SearchPermissionRefused(
            ["a production event-opportunity evaluation requires the authoritative generation store"]
        )
    try:
        origin_event = int(source_identity.get("planning_event") or -1)
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
            [
                "event-opportunity origin permission could not be evaluated: "
                f"{type(failure).__name__}: {failure}"
            ]
        ) from failure

    if action in {cd.CHIP_ACTION_FH, cd.CHIP_ACTION_WC}:
        if expiry_event is None or not isinstance(coverage_product, Mapping):
            raise ReservationForecastError(
                f"{action} future opportunity requires a verified connection and expiry-coverage product"
            )
        typed_request = action_request if action_request is not None else policy
        if typed_request is None:
            raise ReservationForecastError(f"{action} future opportunity requires its canonical typed request")
        try:
            if action == cd.CHIP_ACTION_FH:
                from . import free_hit_production as fhp

                return fhp.build_future_free_hit_event_opportunity(
                    conn,
                    request=typed_request,
                    source_identity=source_identity,
                    event=event,
                    expiry_event=int(expiry_event),
                    coverage_product=coverage_product,
                    reservation_state=reservation_state,
                    made_at=made_at,
                )
            from . import wildcard_production as wcp

            if not chip_generation_id or not value_generation_id:
                raise ReservationForecastError(
                    "future WC opportunity requires the verified four-event and 6–10-event generation identities"
                )
            return wcp.build_future_wildcard_event_opportunity(
                conn,
                request=typed_request,
                source_identity=source_identity,
                event=event,
                expiry_event=int(expiry_event),
                chip_generation_id=chip_generation_id,
                value_generation_id=value_generation_id,
                coverage_product=coverage_product,
                reservation_state=reservation_state,
                made_at=made_at,
            )
        except ReservationForecastError:
            raise
        except Exception as failure:
            raise ReservationForecastError(
                f"{action} canonical future opportunity producer refused: {failure}"
            ) from failure
    if action not in {cd.CHIP_ACTION_BB, cd.CHIP_ACTION_TC}:
        raise ReservationForecastError(
            f"{action} has no production future-event opportunity evaluator"
        )
    if worlds is None or horizon_binding is None or policy is None or positions is None:
        raise ReservationForecastError(f"{action} opportunity is missing its certified event/policy inputs")
    event = int(event)
    planning_event = int(source_identity.get("planning_event") or -1)
    if event <= planning_event or worlds.source_event != event:
        raise ReservationForecastError("certified world matrix is not tagged for the forecast event")
    if (
        int(worlds.planning_event) != planning_event
        or str(worlds.data_snapshot_sha256) != str(source_identity.get("data_snapshot_sha256") or "")
        or str(worlds.certification_identity) != str(source_identity.get("certification_identity") or "")
        or str(worlds.code_snapshot_sha256) != str(source_identity.get("predictive_code_snapshot_sha256") or "")
    ):
        raise ReservationForecastError("certified event worlds do not match the source snapshot identity")
    problems = horizon_binding.matches_worlds(worlds)
    if problems:
        raise ReservationForecastError("certified event worlds do not match the normal horizon: " + "; ".join(problems))
    if int(horizon_binding.planning_event) != planning_event:
        raise ReservationForecastError("forecast horizon binding uses another planning event")
    cutoff = _time(source_identity.get("origin_cutoff"), name="source origin_cutoff")

    from . import chip_bench_boost as bb
    from . import chip_triple_captain as tc
    from . import manager_lineup as ml
    from .chip_reservation_calibration import OUTCOME_SCORING_RULE_VERSION

    if action == cd.CHIP_ACTION_BB:
        evaluation = bb.evaluate_bench_boost(bb.BenchBoostRequest(
            worlds=worlds,
            horizon_binding=horizon_binding,
            policy=policy,
            positions=positions,
            chip_available=True,
        ))
        play_policy = save_policy = policy
    else:
        evaluation = tc.evaluate_triple_captain(tc.TripleCaptainRequest(
            worlds=worlds,
            horizon_binding=horizon_binding,
            policy=policy,
            positions=positions,
            chip_available=True,
        ))
        from dataclasses import replace

        metrics = evaluation.candidate_metrics
        try:
            play_pair = (int(metrics["captain_id"]), int(metrics["vice_captain_id"]))
            save_pair = (int(metrics["save_captain_id"]), int(metrics["save_vice_captain_id"]))
        except (KeyError, TypeError, ValueError) as failure:
            raise ReservationForecastError(
                "Triple Captain evaluator omitted its selected PLAY/SAVE captaincy policies"
            ) from failure
        if play_pair != save_pair:
            raise ReservationForecastError(
                "Triple Captain evaluator returned different PLAY and SAVE captaincy policies"
            )
        starters = {int(player_id) for player_id in policy.starter_ids}
        if (
            play_pair[0] == play_pair[1]
            or play_pair[0] not in starters
            or play_pair[1] not in starters
        ):
            raise ReservationForecastError(
                "Triple Captain evaluator selected an illegal captain/vice policy"
            )
        play_policy = save_policy = replace(
            policy, captain_id=play_pair[0], vice_captain_id=play_pair[1],
        )
    expected_value = _number(
        evaluation.candidate_metrics.get("mean_paired_uplift"),
        name="evaluator expected incremental points",
    )
    squad_ids = sorted(int(value) for value in ml.policy_player_ids(policy))
    position_map = {str(int(pid)): str(positions[int(pid)]) for pid in squad_ids}
    play_policy_payload = play_policy.as_dict()
    save_policy_payload = save_policy.as_dict()
    scenario = canonical_sha256({
        "action": action,
        "event": event,
        "source_identity": dict(source_identity),
        "world_identity": str(worlds.world_identity),
        "squad_ids": squad_ids,
        "starting_lineup": {
            "starter_ids": list(policy.starter_ids),
            "bench_gk_id": int(policy.bench_gk_id),
            "bench_outfield_order": list(policy.bench_outfield_order),
        },
        "play_policy": play_policy_payload,
        "save_policy": save_policy_payload,
        "positions": position_map,
        "continuation_context_sha256": (
            None if continuation_context is None
            else str(continuation_context.get("context_sha256") or "")
        ),
    })
    source_fields = {
        "source_decision_id": str(source_identity["source_decision_id"]),
        "source_result_sha256": str(source_identity["source_result_sha256"]),
        "source_artifact_sha256": str(source_identity["source_artifact_sha256"]),
        "generation_id": str(source_identity["generation_id"]),
        "cutoff": cutoff,
        "data_snapshot_sha256": str(source_identity["data_snapshot_sha256"]),
        "predictive_code_snapshot_sha256": str(source_identity["predictive_code_snapshot_sha256"]),
        "certification_identity": str(source_identity["certification_identity"]),
    }
    state = dict(reservation_state)
    arms = {
        role: {
            "arm_id": f"{action.lower()}-{role}-{canonical_sha256([scenario, role])[:16]}",
            "action": action,
            "event": event,
            "scenario_identity": scenario,
            "world_identity": str(worlds.world_identity),
            "counterfactual_role": role.upper(),
            "proposed_squad_ids": squad_ids,
            "lineup": play_policy_payload if role == "play" else save_policy_payload,
            "player_positions": position_map,
            "reservation_state": state,
            "scoring_rule_version": OUTCOME_SCORING_RULE_VERSION,
            "paired_value": expected_value if role == "play" else 0.0,
            **({
                "continuation_context_sha256": str(continuation_context["context_sha256"]),
                "event_manager_state": dict(continuation_context["event_manager_state"]),
            } if continuation_context is not None else {}),
            **source_fields,
        }
        for role in ("play", "save")
    }
    model = f"{evaluation.evaluator_version}:mean_paired_uplift_v1"
    record = build_event_opportunity_record(
        action=action,
        planning_event=planning_event,
        event=event,
        origin_cutoff=cutoff,
        made_at=made_at,
        expected_incremental_points=expected_value,
        opportunity_model=model,
        source_identity=source_identity,
        reservation_state=state,
        world_identity=str(worlds.world_identity),
        outcome_arms=arms,
        input_as_of=input_as_of or cutoff,
        forecast_mode=FORECAST_MODE_PROSPECTIVE,
        value_units=FORECAST_VALUE_UNITS,
        coverage_product_sha256=coverage_product_sha256,
        evaluator_identity={
            "evaluator_version": str(evaluation.evaluator_version),
            "action": action,
            "event": event,
            "expected_incremental_points": expected_value,
            "uncertainty": dict(evaluation.uncertainty),
            "opportunity_value_definition": "ONE_EVENT_MEAN_POINTS",
            "uncertainty_value_definition": "ONE_EVENT_MEAN_POINTS",
            "decision_horizon_events": list(horizon_binding.horizon_events),
            "world_identity": str(worlds.world_identity),
            "source_identity": dict(source_identity),
            **({"continuation_context_sha256": str(continuation_context["context_sha256"])}
               if continuation_context is not None else {}),
        },
        continuation_context=continuation_context,
    )
    return _attach_search_permission_evaluation(record, permission_evaluation)


def _attach_search_permission_evaluation(
    record: Mapping[str, Any], evaluation: Mapping[str, Any]
) -> dict[str, Any]:
    """Retain the derived permission block in a content-addressed opportunity."""

    body = dict(record)
    body.pop("artifact_sha256", None)
    body["search_permission_evidence_schema"] = "fpl_brain.chip_opportunity_permission.v1"
    body["search_permission_evaluation"] = dict(evaluation)
    body["artifact_sha256"] = canonical_sha256(body)
    return body


def inspect_event_opportunity_structure(
    record: Mapping[str, Any],
    *,
    action: str,
    planning_event: int,
    origin_cutoff: str,
    source_identity: Mapping[str, Any],
    reservation_state: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Check only an opportunity's self-contained shape and byte identities.

    This is for diagnostics and historical inspection. It does not load the
    origin generation, reproduce causal evidence, audit history, or establish
    permission for a production search.
    """
    if record.get("schema") != EVENT_OPPORTUNITY_SCHEMA:
        raise ReservationForecastError("retained event opportunity has an unsupported schema")
    body = dict(record)
    identity = str(body.pop("artifact_sha256", ""))
    if len(identity) != 64 or identity != canonical_sha256(body):
        raise ReservationForecastError("event opportunity content digest does not verify")
    if (
        record.get("action") != action
        or int(record.get("planning_event") or -1) != int(planning_event)
        or int(record.get("event") or -1) <= int(planning_event)
        or _time(record.get("origin_cutoff"), name="event opportunity origin_cutoff")
        != _time(origin_cutoff, name="forecast origin_cutoff")
        or _time(record.get("input_as_of"), name="event opportunity input_as_of")
        > _time(origin_cutoff, name="forecast origin_cutoff")
        or _time(record.get("made_at"), name="event opportunity made_at")
        < _time(record.get("input_as_of"), name="event opportunity input_as_of")
        or record.get("forecast_kind") != "POINT_IN_TIME_EXPECTATION"
        or record.get("forecast_mode") not in {FORECAST_MODE_PROSPECTIVE, FORECAST_MODE_HISTORICAL_REPLAY}
        or record.get("value_units") != FORECAST_VALUE_UNITS
    ):
        raise ReservationForecastError("event opportunity action, time or forecast kind does not match the origin")
    if dict(record.get("source_identity") or {}) != dict(source_identity):
        raise ReservationForecastError("event opportunity is not bound to the forecast source identity")
    state = record.get("reservation_state")
    if not isinstance(state, Mapping) or not state:
        raise ReservationForecastError("event opportunity has no explicit event-specific SAVE state")
    if reservation_state is not None and dict(state) != dict(reservation_state):
        raise ReservationForecastError("event opportunity is not bound to the exact SAVE state")
    if record.get("reservation_state_sha256") != canonical_sha256(state):
        raise ReservationForecastError("event opportunity SAVE-state digest does not verify")
    _number(record.get("expected_incremental_points"), name="event opportunity value")
    evaluator_identity = record.get("evaluator_identity")
    if evaluator_identity is not None:
        if not isinstance(evaluator_identity, Mapping):
            raise ReservationForecastError("event opportunity evaluator identity is malformed")
        if (
            not str(evaluator_identity.get("evaluator_version") or "").strip()
            or evaluator_identity.get("action") != action
            or evaluator_identity.get("event") != int(record["event"])
            or _number(
                evaluator_identity.get("expected_incremental_points"),
                name="evaluator expected_incremental_points",
            ) != _number(record.get("expected_incremental_points"), name="event opportunity value")
        ):
            raise ReservationForecastError("event opportunity evaluator identity does not bind its action/value")
    if not str(record.get("opportunity_model") or "").strip() or not str(
        record.get("world_identity") or ""
    ).strip():
        raise ReservationForecastError("event opportunity lacks model or certified-world identity")
    source = record.get("source_identity")
    if not isinstance(source, Mapping):
        raise ReservationForecastError("event opportunity has no source identity")
    for name in SOURCE_IDENTITY_FIELDS:
        if not str(source.get(name) or "").strip():
            raise ReservationForecastError(f"event opportunity omits source identity {name}")
    for name in (
        "source_result_sha256", "source_artifact_sha256", "data_snapshot_sha256",
        "predictive_code_snapshot_sha256", "certification_identity",
    ):
        value = str(source[name]).removeprefix("sha256:")
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value.lower()):
            raise ReservationForecastError(f"event opportunity source {name} is not a SHA-256 identity")
    if int(source["planning_event"]) != int(planning_event) or _time(
        source["origin_cutoff"], name="event opportunity source origin_cutoff"
    ) != _time(origin_cutoff, name="forecast origin_cutoff"):
        raise ReservationForecastError("event opportunity source does not match its planning origin")
    permission_marker = record.get("search_permission_evidence_schema")
    permission_evaluation = record.get("search_permission_evaluation")
    if permission_marker is None:
        if permission_evaluation is not None:
            raise ReservationForecastError("event opportunity has unversioned search-permission evidence")
    else:
        from . import search_permission as sp

        if (
            permission_marker != "fpl_brain.chip_opportunity_permission.v1"
            or not isinstance(permission_evaluation, Mapping)
            or permission_evaluation.get("schema") != sp.SEARCH_PERMISSION_EVALUATION_SCHEMA
            or permission_evaluation.get("evaluation_sha256") != sp.evaluation_digest(permission_evaluation)
            or permission_evaluation.get("permitted") is not True
            or permission_evaluation.get("reasons") != []
        ):
            raise ReservationForecastError("event opportunity search-permission evidence is malformed or denied")
        permission_origin = permission_evaluation.get("origin")
        permission_conditions = permission_evaluation.get("conditions")
        if not isinstance(permission_origin, Mapping) or not isinstance(permission_conditions, Mapping):
            raise ReservationForecastError("event opportunity search-permission evidence omits its origin conditions")
        if (
            permission_origin.get("generation_id") != source.get("generation_id")
            or permission_origin.get("generation_manifest_sha256") != source.get("generation_id")
            or permission_origin.get("planning_event") != int(planning_event)
            or permission_origin.get("horizon_kind") != "FOUR_GW"
            or _time(permission_origin.get("cutoff"), name="permission origin cutoff")
            != _time(origin_cutoff, name="forecast origin_cutoff")
            or permission_origin.get("data_snapshot_sha256") != source.get("data_snapshot_sha256")
            or permission_origin.get("predictive_code_snapshot_sha256")
            != source.get("predictive_code_snapshot_sha256")
            or permission_conditions.get("temporal_status") != "CAUSAL"
            or permission_conditions.get("dependency_validation") != "COHERENT"
            or not isinstance(permission_conditions.get("history_completeness"), Mapping)
            or permission_conditions["history_completeness"].get("complete") is not True
            or permission_conditions["history_completeness"].get("planning_event") != int(planning_event)
            or _time(
                permission_conditions["history_completeness"].get("cutoff"),
                name="permission history audit cutoff",
            ) != _time(origin_cutoff, name="forecast origin_cutoff")
        ):
            raise ReservationForecastError(
                "event opportunity search-permission evidence does not bind a causal, complete origin"
            )
    outcome_arms = record.get("outcome_arms")
    if outcome_arms is not None:
        rebuilt = build_event_opportunity_record(
            action=action,
            planning_event=planning_event,
            event=int(record["event"]),
            origin_cutoff=str(record["origin_cutoff"]),
            made_at=str(record["made_at"]),
            expected_incremental_points=float(record["expected_incremental_points"]),
            opportunity_model=str(record["opportunity_model"]),
            source_identity=source,
            reservation_state=state,
            world_identity=str(record["world_identity"]),
            outcome_arms=outcome_arms,
            input_as_of=str(record["input_as_of"]),
            forecast_mode=str(record["forecast_mode"]),
            value_units=str(record["value_units"]),
            coverage_product_sha256=record.get("coverage_product_sha256"),
            evaluator_identity=record.get("evaluator_identity"),
            continuation_context=record.get("continuation_context"),
        )
        if permission_marker is not None:
            rebuilt = _attach_search_permission_evaluation(rebuilt, permission_evaluation)
        if dict(rebuilt) != dict(record):
            raise ReservationForecastError("event opportunity outcome-arm manifest does not reproduce")
    return {
        "structural_integrity_verified": True,
        "production_permission_verified": False,
        "verification_scope": "STRUCTURAL_ONLY",
    }


def _verify_event_opportunity(
    record: Mapping[str, Any],
    *,
    store_conn: sqlite3.Connection,
    action: str,
    planning_event: int,
    origin_cutoff: str,
    source_identity: Mapping[str, Any],
    reservation_state: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Verify shape and independently reproduce an opportunity's permission."""

    inspect_event_opportunity_structure(
        record,
        action=action,
        planning_event=planning_event,
        origin_cutoff=origin_cutoff,
        source_identity=source_identity,
        reservation_state=reservation_state,
    )
    permission_marker = record.get("search_permission_evidence_schema")
    permission_evaluation = record.get("search_permission_evaluation")
    if (
        permission_marker != "fpl_brain.chip_opportunity_permission.v1"
        or not isinstance(permission_evaluation, Mapping)
    ):
        raise ReservationForecastError(
            "production event opportunity is missing its required search-permission evidence"
        )
    from . import search_permission as sp

    try:
        expected = sp.verify_recorded_opportunity_evaluation(
            store_conn,
            source_identity=source_identity,
            evaluation=permission_evaluation,
        )
    except Exception as failure:
        raise ReservationForecastError(
            f"event opportunity search-permission evidence did not reproduce: {failure}"
        ) from failure
    if dict(permission_evaluation) != expected:
        raise ReservationForecastError(
            "event opportunity search-permission evidence differs from authoritative origin evidence"
        )
    return {
        "structural_integrity_verified": True,
        "production_permission_verified": True,
        "verification_scope": "AUTHORITATIVE_ORIGIN_PERMISSION",
        "evaluation_sha256": expected.get("evaluation_sha256"),
    }


def _build_reservation_forecast(
    *,
    action: str,
    planning_event: int,
    origin_cutoff: str,
    made_at: str,
    expiry_event: int | None,
    source_identity: Mapping[str, Any],
    reservation_state: Mapping[str, Any],
    opportunity_refs: Sequence[str],
    evidence_verifier: Callable[[str], Mapping[str, Any]],
    input_as_of: str | None = None,
    forecast_mode: str = FORECAST_MODE_PROSPECTIVE,
    coverage_product: Mapping[str, Any] | None = None,
    value_units: str = FORECAST_VALUE_UNITS,
    _opportunity_verifier: Callable[..., Any],
) -> dict[str, Any]:
    """Create a raw forecast from separately retained, verified event forecasts.

    The point estimate is the best expected future event value already visible
    at the origin. It does not select using future realized outcomes. A value of
    zero is emitted only when every event through a known expiry has been
    forecast and every opportunity estimate is non-positive.
    """

    if action not in cd.PLAYABLE_CHIP_ACTIONS:
        raise ReservationForecastError("reservation forecast has an unknown chip action")
    planning_event = int(planning_event)
    cutoff, forecast_time = _time(origin_cutoff, name="origin_cutoff"), _time(made_at, name="made_at")
    as_of = _time(input_as_of or cutoff, name="input_as_of")
    if as_of > cutoff:
        raise ReservationForecastError("reservation forecast inputs are later than its origin cutoff")
    if forecast_time < as_of:
        raise ReservationForecastError("reservation forecast issuance precedes its input-as-of time")
    if forecast_mode not in {FORECAST_MODE_PROSPECTIVE, FORECAST_MODE_HISTORICAL_REPLAY}:
        raise ReservationForecastError("reservation forecast mode is unsupported")
    if value_units != FORECAST_VALUE_UNITS:
        raise ReservationForecastError("reservation forecast uses incompatible comparison value units")
    if expiry_event is not None and int(expiry_event) < planning_event:
        raise ReservationForecastError("chip expiry cannot precede its planning event")
    if evidence_verifier is None:
        raise ReservationForecastError("a retained event-opportunity verifier is required")
    if not opportunity_refs and not (
        expiry_event is None
        or int(expiry_event) == planning_event
        or coverage_product is not None
    ):
        raise ReservationForecastError("reservation forecast has no retained event opportunities")
    source = dict(source_identity)
    state = dict(reservation_state)
    opportunities: list[dict[str, Any]] = []
    seen_events: set[int] = set()
    for reference in opportunity_refs:
        try:
            record = evidence_verifier(str(reference))
        except Exception as failure:
            raise ReservationForecastError(f"retained event opportunity did not verify: {failure}") from failure
        if not isinstance(record, Mapping):
            raise ReservationForecastError("retained event opportunity is not an object")
        _opportunity_verifier(
            record,
            action=action,
            planning_event=planning_event,
            origin_cutoff=cutoff,
            source_identity=source,
            reservation_state=None,
        )
        if str(record.get("input_as_of")) > as_of:
            raise ReservationForecastError("event opportunity inputs are later than forecast input_as_of")
        if _time(record.get("made_at"), name="event opportunity made_at") > forecast_time:
            raise ReservationForecastError("reservation forecast predates one of its event opportunities")
        if record.get("value_units") != value_units:
            raise ReservationForecastError("event opportunity comparison value units differ from forecast")
        event = int(record["event"])
        if event in seen_events:
            raise ReservationForecastError("reservation forecast repeats a future event")
        seen_events.add(event)
        opportunities.append({
            "artifact_ref": str(reference),
            "artifact_sha256": str(record["artifact_sha256"]),
            "artifact_payload": dict(record),
        })
    opportunities.sort(key=lambda row: int(row["artifact_payload"]["event"]))
    events = [int(row["artifact_payload"]["event"]) for row in opportunities]
    observed_event_set_complete = (
        expiry_event is not None
        and events == list(range(planning_event + 1, int(expiry_event) + 1))
    )
    complete = bool(expiry_event is not None and int(expiry_event) == planning_event)
    coverage_product_identity: str | None = None
    if coverage_product is not None:
        product = dict(coverage_product)
        coverage_product_identity, product_complete = _verify_coverage_product_payload(
            product,
            action=action,
            planning_event=planning_event,
            expiry_event=expiry_event,
            source_identity=source,
            input_as_of=as_of,
        )
        product_forecast_events = [int(value) for value in product.get("forecast_events") or ()]
        if not set(events).issubset(set(product_forecast_events)):
            raise ReservationForecastError("opportunity events differ from the retained expiry-coverage product")
        if any(
            (
                row["artifact_payload"].get("coverage_product_sha256") is not None
                and str(row["artifact_payload"].get("coverage_product_sha256")) != coverage_product_identity
            )
            or (
                row["artifact_payload"].get("continuation_context") is not None
                and str(row["artifact_payload"].get("coverage_product_sha256") or "")
                != coverage_product_identity
            )
            for row in opportunities
        ):
            raise ReservationForecastError("event opportunity is not bound to the retained expiry-coverage product")
        complete = bool(
            observed_event_set_complete
            and product_complete
            and events == product_forecast_events
        )
    elif observed_event_set_complete and int(expiry_event) == planning_event:
        # Once the chip has expired there is no remaining opportunity to forecast.
        complete = True
    if complete:
        best = max(
            (float(row["artifact_payload"]["expected_incremental_points"]) for row in opportunities),
            default=0.0,
        )
        raw_value: float | None = max(0.0, best)
        selected_event = next((
            int(row["artifact_payload"]["event"])
            for row in opportunities
            if float(row["artifact_payload"]["expected_incremental_points"]) == best
        ), None)
        status, reason = FORECAST_READY, None
    else:
        raw_value = None
        selected_event = None
        status, reason = FORECAST_INCOMPLETE, "CHIP_RESERVATION_FORECAST_COVERAGE_INCOMPLETE"
    body: dict[str, Any] = {
        "schema": RESERVATION_FORECAST_SCHEMA,
        "version": RESERVATION_FORECAST_VERSION,
        "model": RESERVATION_FORECAST_MODEL,
        "action": action,
        "planning_event": planning_event,
        "origin_cutoff": cutoff,
        "input_as_of": as_of,
        "made_at": forecast_time,
        "forecast_mode": forecast_mode,
        "value_units": value_units,
        "expiry_event": None if expiry_event is None else int(expiry_event),
        "source_identity": source,
        "reservation_state": state,
        "reservation_state_sha256": canonical_sha256(state),
        "opportunities": opportunities,
        "coverage_product": None if coverage_product is None else dict(coverage_product),
        "coverage_product_sha256": coverage_product_identity,
        "covered_events": events,
        "coverage_complete": bool(complete),
        "coverage_status": status,
        "raw_value": raw_value,
        "selected_event": selected_event,
        "reason_code": reason,
        "selection_policy": "BEST_EXPECTED_VALUE_VISIBLE_AT_ORIGIN; NO_REALIZED_FUTURE_SELECTION",
    }
    body["artifact_sha256"] = canonical_sha256(body)
    return body


def build_reservation_forecast(
    *,
    store_conn: sqlite3.Connection,
    action: str,
    planning_event: int,
    origin_cutoff: str,
    made_at: str,
    expiry_event: int | None,
    source_identity: Mapping[str, Any],
    reservation_state: Mapping[str, Any],
    opportunity_refs: Sequence[str],
    evidence_verifier: Callable[[str], Mapping[str, Any]],
    input_as_of: str | None = None,
    forecast_mode: str = FORECAST_MODE_PROSPECTIVE,
    coverage_product: Mapping[str, Any] | None = None,
    value_units: str = FORECAST_VALUE_UNITS,
) -> dict[str, Any]:
    """Build a production forecast after re-verifying each origin in the store."""

    if not isinstance(store_conn, sqlite3.Connection):
        raise ReservationForecastError("an authoritative generation-store connection is required")
    return _build_reservation_forecast(
        action=action,
        planning_event=planning_event,
        origin_cutoff=origin_cutoff,
        made_at=made_at,
        expiry_event=expiry_event,
        source_identity=source_identity,
        reservation_state=reservation_state,
        opportunity_refs=opportunity_refs,
        evidence_verifier=evidence_verifier,
        input_as_of=input_as_of,
        forecast_mode=forecast_mode,
        coverage_product=coverage_product,
        value_units=value_units,
        _opportunity_verifier=lambda record, **kwargs: _verify_event_opportunity(
            record, store_conn=store_conn, **kwargs
        ),
    )


def build_structural_reservation_forecast(
    *,
    action: str,
    planning_event: int,
    origin_cutoff: str,
    made_at: str,
    expiry_event: int | None,
    source_identity: Mapping[str, Any],
    reservation_state: Mapping[str, Any],
    opportunity_refs: Sequence[str],
    evidence_verifier: Callable[[str], Mapping[str, Any]],
    input_as_of: str | None = None,
    forecast_mode: str = FORECAST_MODE_PROSPECTIVE,
    coverage_product: Mapping[str, Any] | None = None,
    value_units: str = FORECAST_VALUE_UNITS,
) -> dict[str, Any]:
    """Build a structurally checked diagnostic forecast without production authority."""

    return _build_reservation_forecast(
        action=action,
        planning_event=planning_event,
        origin_cutoff=origin_cutoff,
        made_at=made_at,
        expiry_event=expiry_event,
        source_identity=source_identity,
        reservation_state=reservation_state,
        opportunity_refs=opportunity_refs,
        evidence_verifier=evidence_verifier,
        input_as_of=input_as_of,
        forecast_mode=forecast_mode,
        coverage_product=coverage_product,
        value_units=value_units,
        _opportunity_verifier=inspect_event_opportunity_structure,
    )


def _verify_reservation_forecast(
    forecast: Mapping[str, Any],
    *,
    expected: Mapping[str, Any],
    evidence_verifier: Callable[[str], Mapping[str, Any]] | None = None,
    _opportunity_verifier: Callable[..., Any],
) -> dict[str, Any]:
    """Verify content, nested sources, forecast timing and assessment binding."""

    if forecast.get("schema") != RESERVATION_FORECAST_SCHEMA or forecast.get("version") != RESERVATION_FORECAST_VERSION:
        raise ReservationForecastError("reservation forecast schema/version is unsupported")
    if forecast.get("model") != RESERVATION_FORECAST_MODEL:
        raise ReservationForecastError("reservation forecast model is unsupported")
    body = dict(forecast)
    identity = str(body.pop("artifact_sha256", ""))
    if len(identity) != 64 or identity != canonical_sha256(body):
        raise ReservationForecastError("reservation forecast content digest does not verify")
    for name, value in expected.items():
        if forecast.get(name) != value:
            raise ReservationForecastError(f"reservation forecast disagrees with assessment context on {name}")
    origin_time = _time(forecast.get("origin_cutoff"), name="forecast origin_cutoff")
    input_as_of = _time(forecast.get("input_as_of"), name="forecast input_as_of")
    forecast_time = _time(forecast.get("made_at"), name="forecast made_at")
    if input_as_of > origin_time:
        raise ReservationForecastError("reservation forecast inputs are later than its origin cutoff")
    if forecast_time < input_as_of:
        raise ReservationForecastError("reservation forecast issuance precedes its input-as-of time")
    if forecast.get("forecast_mode") not in {FORECAST_MODE_PROSPECTIVE, FORECAST_MODE_HISTORICAL_REPLAY}:
        raise ReservationForecastError("reservation forecast mode is unsupported")
    if forecast.get("value_units") != FORECAST_VALUE_UNITS:
        raise ReservationForecastError("reservation forecast comparison value units are unsupported")
    action = str(forecast.get("action") or "")
    try:
        planning_event = int(forecast.get("planning_event"))
    except (TypeError, ValueError) as failure:
        raise ReservationForecastError("reservation forecast planning event is invalid") from failure
    if action not in cd.PLAYABLE_CHIP_ACTIONS:
        raise ReservationForecastError("reservation forecast has an unknown chip action")
    reservation_state = forecast.get("reservation_state")
    source_identity = forecast.get("source_identity")
    if not isinstance(reservation_state, Mapping) or not reservation_state:
        raise ReservationForecastError("reservation forecast has no exact SAVE state")
    if forecast.get("reservation_state_sha256") != canonical_sha256(reservation_state):
        raise ReservationForecastError("reservation forecast SAVE-state digest does not verify")
    if not isinstance(source_identity, Mapping):
        raise ReservationForecastError("reservation forecast has no source identity")
    opportunities = forecast.get("opportunities")
    if not isinstance(opportunities, list):
        raise ReservationForecastError("reservation forecast has no event opportunity manifest")
    if not opportunities and forecast.get("expiry_event") not in (
        None, forecast.get("planning_event"),
    ):
        raise ReservationForecastError("reservation forecast has no event opportunity manifest")
    checked_events: list[int] = []
    for item in opportunities:
        if (
            not isinstance(item, Mapping)
            or not str(item.get("artifact_ref") or "").strip()
            or not isinstance(item.get("artifact_payload"), Mapping)
        ):
            raise ReservationForecastError("reservation forecast event manifest is malformed")
        payload = item["artifact_payload"]
        digest = str(item.get("artifact_sha256") or "")
        if digest != payload.get("artifact_sha256"):
            raise ReservationForecastError("reservation forecast event digest differs from its payload")
        if evidence_verifier is not None:
            try:
                retained = evidence_verifier(str(item.get("artifact_ref") or ""))
            except Exception as failure:
                raise ReservationForecastError(f"retained event opportunity did not re-verify: {failure}") from failure
            if not isinstance(retained, Mapping) or _canonical_bytes(retained) != _canonical_bytes(payload):
                raise ReservationForecastError("retained event opportunity differs from the forecast manifest")
        _opportunity_verifier(
            payload,
            action=action,
            planning_event=planning_event,
            origin_cutoff=str(forecast.get("origin_cutoff") or ""),
            source_identity=source_identity,
            reservation_state=None,
        )
        if str(payload.get("input_as_of")) > input_as_of:
            raise ReservationForecastError("event opportunity inputs are later than forecast input_as_of")
        if _time(payload.get("made_at"), name="event opportunity made_at") > forecast_time:
            raise ReservationForecastError("reservation forecast predates one of its event opportunities")
        if payload.get("value_units") != forecast.get("value_units"):
            raise ReservationForecastError("event opportunity comparison value units differ from forecast")
        checked_events.append(int(payload["event"]))
    if checked_events != sorted(set(checked_events)):
        raise ReservationForecastError("reservation forecast event manifest is not strictly increasing")
    try:
        declared_events = [int(value) for value in forecast.get("covered_events") or ()]
    except (TypeError, ValueError) as failure:
        raise ReservationForecastError("reservation forecast event coverage is malformed") from failure
    if checked_events != declared_events:
        raise ReservationForecastError("reservation forecast event coverage does not reproduce")
    expiry = forecast.get("expiry_event")
    try:
        expiry_event = None if expiry is None else int(expiry)
    except (TypeError, ValueError) as failure:
        raise ReservationForecastError("reservation forecast expiry event is invalid") from failure
    if expiry_event is not None and expiry_event < planning_event:
        raise ReservationForecastError("reservation forecast expiry precedes its planning event")
    observed_event_set_complete = expiry_event is not None and checked_events == list(range(
        planning_event + 1, expiry_event + 1,
    ))
    complete = bool(expiry_event is not None and expiry_event == planning_event)
    coverage_product = forecast.get("coverage_product")
    coverage_product_sha256 = forecast.get("coverage_product_sha256")
    if coverage_product is None:
        if coverage_product_sha256 is not None:
            raise ReservationForecastError("reservation forecast declares a missing coverage product")
    else:
        if not isinstance(coverage_product, Mapping):
            raise ReservationForecastError("reservation forecast coverage product is malformed")
        product_identity, product_complete = _verify_coverage_product_payload(
            coverage_product,
            action=action,
            planning_event=planning_event,
            expiry_event=expiry_event,
            source_identity=source_identity,
            input_as_of=input_as_of,
        )
        if product_identity != str(coverage_product_sha256):
            raise ReservationForecastError("reservation forecast coverage-product identity does not verify")
        product_forecast_events = [int(value) for value in coverage_product.get("forecast_events") or ()]
        if not set(checked_events).issubset(set(product_forecast_events)):
            raise ReservationForecastError("reservation coverage product does not reproduce forecast opportunity events")
        if any(
            (
                row["artifact_payload"].get("coverage_product_sha256") is not None
                and str(row["artifact_payload"].get("coverage_product_sha256")) != product_identity
            )
            or (
                row["artifact_payload"].get("continuation_context") is not None
                and str(row["artifact_payload"].get("coverage_product_sha256") or "") != product_identity
            )
            for row in opportunities
        ):
            raise ReservationForecastError("retained event opportunity differs from the coverage product")
        complete = bool(
            observed_event_set_complete
            and product_complete
            and checked_events == product_forecast_events
        )
        if action in {cd.CHIP_ACTION_FH, cd.CHIP_ACTION_WC} and any(
            any(
                isinstance(arm.get("valuation_schedule"), Mapping)
                or (arm.get("action_semantics") or {}).get("valuation_schedule_required") is True
                for arm in (row["artifact_payload"].get("outcome_arms") or {}).values()
                if isinstance(arm, Mapping)
            )
            and any(
                arm.get("coverage_product_sha256") != product_identity
                for arm in (row["artifact_payload"].get("outcome_arms") or {}).values()
                if isinstance(arm, Mapping)
            )
            for row in opportunities
        ):
            raise ReservationForecastError("future FH/WC event producer is bound to another expiry product")
    if expiry_event is not None and expiry_event == planning_event:
        complete = True
    if bool(forecast.get("coverage_complete")) != complete:
        raise ReservationForecastError("reservation forecast coverage-complete claim does not reproduce")
    if forecast.get("selection_policy") != "BEST_EXPECTED_VALUE_VISIBLE_AT_ORIGIN; NO_REALIZED_FUTURE_SELECTION":
        raise ReservationForecastError("reservation forecast selection policy is unsupported")
    if complete:
        best = max(
            (float(row["artifact_payload"]["expected_incremental_points"]) for row in opportunities),
            default=0.0,
        )
        expected_value = max(0.0, best)
        expected_event = next((
            int(row["artifact_payload"]["event"])
            for row in opportunities
            if float(row["artifact_payload"]["expected_incremental_points"]) == best
        ), None)
        if forecast.get("coverage_status") != FORECAST_READY or _number(
            forecast.get("raw_value"), name="raw reservation value",
        ) != expected_value or forecast.get("selected_event") != expected_event or forecast.get("reason_code") is not None:
            raise ReservationForecastError("complete reservation forecast value does not reproduce")
    elif (
        forecast.get("raw_value") is not None
        or forecast.get("selected_event") is not None
        or forecast.get("coverage_status") != FORECAST_INCOMPLETE
        or forecast.get("reason_code") != "CHIP_RESERVATION_FORECAST_COVERAGE_INCOMPLETE"
    ):
        raise ReservationForecastError("incomplete reservation forecast was silently treated as numeric")
    return {"verified": True, "artifact_sha256": identity, "raw_value": forecast.get("raw_value")}


def verify_reservation_forecast(
    forecast: Mapping[str, Any],
    *,
    expected: Mapping[str, Any],
    store_conn: sqlite3.Connection,
    evidence_verifier: Callable[[str], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Verify a production forecast and every opportunity against its origin store."""

    if not isinstance(store_conn, sqlite3.Connection):
        raise ReservationForecastError("an authoritative generation-store connection is required")
    report = _verify_reservation_forecast(
        forecast,
        expected=expected,
        evidence_verifier=evidence_verifier,
        _opportunity_verifier=lambda record, **kwargs: _verify_event_opportunity(
            record, store_conn=store_conn, **kwargs
        ),
    )
    return {**report, "production_permission_verified": True}


def inspect_reservation_forecast_structure(
    forecast: Mapping[str, Any],
    *,
    expected: Mapping[str, Any],
    evidence_verifier: Callable[[str], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Inspect forecast bytes and semantics without granting production authority."""

    report = _verify_reservation_forecast(
        forecast,
        expected=expected,
        evidence_verifier=evidence_verifier,
        _opportunity_verifier=inspect_event_opportunity_structure,
    )
    return {
        **report,
        "production_permission_verified": False,
        "verification_scope": "STRUCTURAL_ONLY",
    }


@dataclass(frozen=True)
class VerifiedReservationForecast:
    artifact: Mapping[str, Any]

    @property
    def raw_value(self) -> float | None:
        value = self.artifact.get("raw_value")
        return None if value is None else float(value)

    @property
    def artifact_sha256(self) -> str:
        return str(self.artifact.get("artifact_sha256") or "")

    def validate_for_reservation(
        self, *, action: str, planning_event: int, expiry_event: int | None, state: Mapping[str, Any],
        store_conn: sqlite3.Connection,
        evidence_verifier: Callable[[str], Mapping[str, Any]] | None = None,
    ) -> None:
        verify_reservation_forecast(
            self.artifact,
            expected={
                "action": action,
                "planning_event": int(planning_event),
                "expiry_event": None if expiry_event is None else int(expiry_event),
            },
            store_conn=store_conn,
            evidence_verifier=evidence_verifier,
        )
        forecast_state = self.artifact.get("reservation_state")
        decision_state = {
            key: value for key, value in state.items()
            if key not in {"raw_reservation_value", "raw_reservation_forecast"}
        }
        if not isinstance(forecast_state, Mapping) or dict(forecast_state) != decision_state:
            raise ReservationForecastError("reservation forecast does not match the arbiter SAVE state")


def retain_reservation_forecast(
    forecast: Mapping[str, Any],
    root: str | Path,
    *,
    store_conn: sqlite3.Connection,
) -> dict[str, str]:
    """Atomically retain one forecast with no replacement or final-path stream."""

    verify_reservation_forecast(forecast, expected={}, store_conn=store_conn)
    payload = _canonical_bytes(forecast)
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    target = root_path / f"chip-reservation-forecast-{forecast['artifact_sha256']}.json"
    temporary: Path | None = None
    try:
        fd, name = tempfile.mkstemp(prefix=target.name + ".tmp-", dir=str(root_path))
        temporary = Path(name)
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError as failure:
            if target.read_bytes() != payload:
                raise ReservationForecastError(f"reservation forecast path already exists with different bytes: {target}") from failure
        except OSError as failure:
            raise ReservationForecastError(
                f"{FORECAST_PUBLICATION_UNAVAILABLE}: atomic no-replace publication failed: {failure}"
            ) from failure
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
    return {"path": str(target), "artifact_sha256": str(forecast["artifact_sha256"])}


def retain_event_opportunity_record(
    record: Mapping[str, Any],
    root: str | Path,
    *,
    store_conn: sqlite3.Connection,
) -> dict[str, str]:
    """Atomically retain one certified event opportunity without replacement."""

    _verify_event_opportunity(
        record,
        store_conn=store_conn,
        action=str(record.get("action") or ""),
        planning_event=int(record.get("planning_event") or -1),
        origin_cutoff=str(record.get("origin_cutoff") or ""),
        source_identity=record.get("source_identity") or {},
        reservation_state=record.get("reservation_state") or {},
    )
    payload = _canonical_bytes(record)
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    target = root_path / f"chip-event-opportunity-{record['artifact_sha256']}.json"
    temporary: Path | None = None
    try:
        fd, name = tempfile.mkstemp(prefix=target.name + ".tmp-", dir=str(root_path))
        temporary = Path(name)
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError as failure:
            if target.read_bytes() != payload:
                raise ReservationForecastError(
                    f"event opportunity path already exists with different bytes: {target}"
                ) from failure
        except OSError as failure:
            raise ReservationForecastError(
                f"{FORECAST_PUBLICATION_UNAVAILABLE}: atomic no-replace publication failed: {failure}"
            ) from failure
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
    return {"path": str(target), "artifact_sha256": str(record["artifact_sha256"])}
