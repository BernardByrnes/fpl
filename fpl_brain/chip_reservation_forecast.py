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
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from . import chip_decision as cd

EVENT_OPPORTUNITY_SCHEMA = "fpl_brain.chip_event_opportunity_forecast.v1"
RESERVATION_FORECAST_SCHEMA = "fpl_brain.chip_reservation_forecast.v1"
RESERVATION_FORECAST_VERSION = "chip_reservation_forecast_v1.1.0"
RESERVATION_FORECAST_MODEL = "max_point_in_time_expected_opportunity_through_expiry_v2"
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
    cutoff, made = _time(origin_cutoff, name="origin_cutoff"), _time(made_at, name="made_at")
    if made > cutoff:
        raise ReservationForecastError("event opportunity was generated after its origin cutoff")
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
        "made_at": made,
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
    if outcome_arms is not None:
        if action not in {cd.CHIP_ACTION_BB, cd.CHIP_ACTION_TC}:
            raise ReservationForecastError(
                f"{action} has no production future-event outcome-arm producer"
            )
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
            or arms["play"].get("proposed_squad_ids") != arms["save"].get("proposed_squad_ids")
            or arms["play"].get("lineup") != arms["save"].get("lineup")
            or arms["play"].get("player_positions") != arms["save"].get("player_positions")
        ):
            raise ReservationForecastError("BB/TC event outcome arms do not share one scenario policy")
        body["outcome_arms"] = arms
    body["artifact_sha256"] = canonical_sha256(body)
    return body


def build_evaluated_event_opportunity_record(
    *,
    action: str,
    event: int,
    worlds: cd.ChipWorldInputs,
    horizon_binding: cd.ChipHorizonBinding,
    policy: Any,
    positions: Mapping[int, str],
    source_identity: Mapping[str, Any],
    reservation_state: Mapping[str, Any],
    made_at: str,
) -> dict[str, Any]:
    """Produce a BB/TC opportunity and maturation arms from one certified event.

    The normal-route SAVE policy and chip PLAY policy are scored by the existing
    production evaluator on a certified matrix explicitly tagged for ``event``.
    FH/WC stay fail-closed until their event-future route producers are present.
    """

    if action not in {cd.CHIP_ACTION_BB, cd.CHIP_ACTION_TC}:
        raise ReservationForecastError(
            f"{action} has no production future-event opportunity evaluator"
        )
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
    if _time(made_at, name="made_at") > cutoff:
        raise ReservationForecastError("event opportunity was generated after its origin cutoff")

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
            **source_fields,
        }
        for role in ("play", "save")
    }
    model = f"{evaluation.evaluator_version}:mean_paired_uplift_v1"
    return build_event_opportunity_record(
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
    )


def _verify_event_opportunity(
    record: Mapping[str, Any],
    *,
    action: str,
    planning_event: int,
    origin_cutoff: str,
    source_identity: Mapping[str, Any],
    reservation_state: Mapping[str, Any],
) -> None:
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
        or _time(record.get("made_at"), name="event opportunity made_at")
        > _time(origin_cutoff, name="forecast origin_cutoff")
        or record.get("forecast_kind") != "POINT_IN_TIME_EXPECTATION"
    ):
        raise ReservationForecastError("event opportunity action, time or forecast kind does not match the origin")
    if dict(record.get("source_identity") or {}) != dict(source_identity):
        raise ReservationForecastError("event opportunity is not bound to the forecast source identity")
    state = record.get("reservation_state")
    if not isinstance(state, Mapping) or dict(state) != dict(reservation_state):
        raise ReservationForecastError("event opportunity is not bound to the exact SAVE state")
    if record.get("reservation_state_sha256") != canonical_sha256(state):
        raise ReservationForecastError("event opportunity SAVE-state digest does not verify")
    _number(record.get("expected_incremental_points"), name="event opportunity value")
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
        )
        if dict(rebuilt) != dict(record):
            raise ReservationForecastError("event opportunity outcome-arm manifest does not reproduce")


def build_reservation_forecast(
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
    if forecast_time > cutoff:
        raise ReservationForecastError("reservation forecast was made after its origin cutoff")
    if expiry_event is not None and int(expiry_event) < planning_event:
        raise ReservationForecastError("chip expiry cannot precede its planning event")
    if evidence_verifier is None:
        raise ReservationForecastError("a retained event-opportunity verifier is required")
    if not opportunity_refs and not (
        expiry_event is not None and int(expiry_event) == planning_event
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
        _verify_event_opportunity(
            record,
            action=action,
            planning_event=planning_event,
            origin_cutoff=cutoff,
            source_identity=source,
            reservation_state=state,
        )
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
    complete = expiry_event is not None and events == list(range(planning_event + 1, int(expiry_event) + 1))
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
        "made_at": forecast_time,
        "expiry_event": None if expiry_event is None else int(expiry_event),
        "source_identity": source,
        "reservation_state": state,
        "reservation_state_sha256": canonical_sha256(state),
        "opportunities": opportunities,
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


def verify_reservation_forecast(
    forecast: Mapping[str, Any],
    *,
    expected: Mapping[str, Any],
    evidence_verifier: Callable[[str], Mapping[str, Any]] | None = None,
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
    if _time(forecast.get("made_at"), name="forecast made_at") > _time(
        forecast.get("origin_cutoff"), name="forecast origin_cutoff"
    ):
        raise ReservationForecastError("reservation forecast was made after its origin cutoff")
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
    if not opportunities and forecast.get("expiry_event") != forecast.get("planning_event"):
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
        _verify_event_opportunity(
            payload,
            action=action,
            planning_event=planning_event,
            origin_cutoff=str(forecast.get("origin_cutoff") or ""),
            source_identity=source_identity,
            reservation_state=reservation_state,
        )
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
    complete = expiry_event is not None and checked_events == list(range(
        planning_event + 1, expiry_event + 1,
    ))
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
        evidence_verifier: Callable[[str], Mapping[str, Any]] | None = None,
    ) -> None:
        verify_reservation_forecast(
            self.artifact,
            expected={
                "action": action,
                "planning_event": int(planning_event),
                "expiry_event": None if expiry_event is None else int(expiry_event),
            },
            evidence_verifier=evidence_verifier,
        )
        forecast_state = self.artifact.get("reservation_state")
        decision_state = {
            key: value for key, value in state.items()
            if key not in {"raw_reservation_value", "raw_reservation_forecast"}
        }
        if not isinstance(forecast_state, Mapping) or dict(forecast_state) != decision_state:
            raise ReservationForecastError("reservation forecast does not match the arbiter SAVE state")


def retain_reservation_forecast(forecast: Mapping[str, Any], root: str | Path) -> dict[str, str]:
    """Atomically retain one forecast with no replacement or final-path stream."""

    verify_reservation_forecast(forecast, expected={})
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


def retain_event_opportunity_record(record: Mapping[str, Any], root: str | Path) -> dict[str, str]:
    """Atomically retain one certified event opportunity without replacement."""

    _verify_event_opportunity(
        record,
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
