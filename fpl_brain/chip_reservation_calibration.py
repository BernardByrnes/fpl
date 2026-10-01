"""Validated, walk-forward calibration for chip reservation values.

This module only calibrates the future-option estimate consumed by the chip
arbiter.  It does not change an evaluator's independent ``execution_permitted``
gate.  Evidence is accepted only through a caller-supplied artifact verifier;
without one, or without matured paired causal records, the result stays
UNCALIBRATED.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import tempfile
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from . import chip_decision as cd
from . import chip_reservation_forecast as crf
from . import manager_lineup as ml
from . import outcome_ledger as ol

CALIBRATION_SCHEMA = "fpl_brain.chip_reservation_calibration.v1"
CAUSAL_EVIDENCE_SCHEMA = "fpl_brain.chip_reservation_causal_evidence.v3"
CAUSAL_OUTCOME_RECORD_SCHEMA = "fpl_brain.chip_reservation_outcome_record.v3"
OUTCOME_CAPTURE_SET_SCHEMA = "fpl_brain.outcome_ledger_capture_set.v1"
OUTCOME_LABEL_DEFINITION = "CANONICAL_CHIP_PLAY_MINUS_SAVE_EVENT_POINTS_V3"
OUTCOME_SCORING_RULE_VERSION = "manager_lineup_and_action_specific_chip_scores_v2"
CALIBRATION_VERSION = "chip_reservation_walkforward_v4.0.0"
CAUSAL_EVIDENCE_POLICY = "SAME_SOURCE_WORLD_ACTION_SPECIFIC_PLAY_SAVE_TEMPORAL_LABELS_v3"

# These criteria are fixed in code before labels are evaluated. A caller cannot
# loosen them in a record after seeing the results.
MIN_TRAINING_ORIGINS = 20
MIN_VALIDATION_ORIGINS = 30
MIN_MAE_IMPROVEMENT = 0.05
MAX_ABSOLUTE_BIAS = 1.5
MIN_90_INTERVAL_COVERAGE = 0.80

DIAG_CALIBRATION_EVIDENCE_INVALID = "CHIP_RESERVATION_CAUSAL_EVIDENCE_INVALID"
DIAG_CALIBRATION_INSUFFICIENT = "CHIP_RESERVATION_CALIBRATION_INSUFFICIENT_CAUSAL_EVIDENCE"
DIAG_CALIBRATION_NOT_VALIDATED = "CHIP_RESERVATION_CALIBRATION_NOT_VALIDATED"
DIAG_RESERVATION_FORECAST_REQUIRED = "CHIP_RESERVATION_RAW_FORECAST_REQUIRED"
DIAG_RESERVATION_FORECAST_INCOMPLETE = "CHIP_RESERVATION_FORECAST_COVERAGE_INCOMPLETE"


class ReservationCalibrationError(ValueError):
    pass


def _utc(value: Any, *, name: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ReservationCalibrationError(f"{name} is missing")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as failure:
        raise ReservationCalibrationError(f"{name} is not an ISO timestamp") from failure
    if parsed.tzinfo is None:
        raise ReservationCalibrationError(f"{name} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _number(value: Any, *, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as failure:
        raise ReservationCalibrationError(f"{name} is not numeric") from failure
    if not math.isfinite(parsed):
        raise ReservationCalibrationError(f"{name} is not finite")
    return parsed


def _canonical_sha256(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False, default=str).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False, default=str,
    ).encode("utf-8")


def _is_sha256(value: Any) -> bool:
    text = str(value or "")
    if text.startswith("sha256:"):
        text = text[7:]
    return len(text) == 64 and all(char in "0123456789abcdef" for char in text.lower())


def _canonical_scoring_weights(
    *,
    action: str,
    arm_name: str,
    arm: Mapping[str, Any],
    player_points: Mapping[int, float],
    player_minutes: Mapping[int, float],
) -> dict[str, float]:
    """Rebuild realized chip point weights from each retained FPL arm policy.

    Caller-provided weights are data to verify, never the scoring authority.
    """

    if action not in cd.PLAYABLE_CHIP_ACTIONS:
        raise ReservationCalibrationError(
            f"{action} has no canonical reservation outcome scorer"
        )
    if arm_name not in {"play", "save"}:
        raise ReservationCalibrationError("paired outcome arm name is invalid")
    try:
        squad_ids = tuple(int(value) for value in arm.get("proposed_squad_ids", ()))
        lineup = arm.get("lineup")
        if not isinstance(lineup, Mapping):
            raise ValueError("lineup is missing")
        policy = ml.ManagerPolicy(
            starter_ids=tuple(int(value) for value in lineup["starter_ids"]),
            bench_gk_id=int(lineup["bench_gk_id"]),
            bench_outfield_order=tuple(int(value) for value in lineup["bench_outfield_order"]),
            captain_id=int(lineup["captain_id"]),
            vice_captain_id=int(lineup["vice_captain_id"]),
        )
        raw_positions = arm.get("player_positions")
        if not isinstance(raw_positions, Mapping):
            raise ValueError("source player positions are missing")
        positions: dict[int, str] = {}
        for raw_player_id, value in raw_positions.items():
            player_id = int(raw_player_id)
            if player_id in positions:
                raise ValueError("duplicate player position")
            positions[player_id] = str(value)
    except (KeyError, TypeError, ValueError) as failure:
        raise ReservationCalibrationError(
            f"{arm_name.upper()} arm has an invalid canonical scoring policy: {failure}"
        ) from failure

    policy_ids = ml.policy_player_ids(policy)
    if (
        len(squad_ids) != ml.SQUAD_SIZE
        or len(set(squad_ids)) != ml.SQUAD_SIZE
        or set(squad_ids) != policy_ids
        or set(positions) != policy_ids
    ):
        raise ReservationCalibrationError(
            f"{arm_name.upper()} arm does not retain one complete 15-player scoring policy"
        )
    legality = ml.policy_legality_errors(policy, positions)
    if legality:
        raise ReservationCalibrationError(
            f"{arm_name.upper()} arm scoring policy is not a legal FPL lineup: {', '.join(legality)}"
        )
    if policy_ids - set(player_points) or policy_ids - set(player_minutes):
        raise ReservationCalibrationError(
            f"{arm_name.upper()} arm lacks official points or minutes for its full squad"
        )

    points = {player_id: float(player_points[player_id]) for player_id in policy_ids}
    minutes = {player_id: float(player_minutes[player_id]) for player_id in policy_ids}
    normal = ml.resolve_world(
        policy, positions, minutes, points, require_player_ids=policy_ids,
    )
    weights = {str(player_id): 1.0 for player_id in normal.counted_ids}
    armband_player: int | None = None
    if minutes[policy.captain_id] > 0.0:
        armband_player = policy.captain_id
    elif minutes[policy.vice_captain_id] > 0.0:
        armband_player = policy.vice_captain_id
    if armband_player is not None:
        weights[str(armband_player)] = weights.get(str(armband_player), 0.0) + 1.0

    if action == cd.CHIP_ACTION_BB and arm_name == "play":
        weights = {
            str(player_id): 1.0
            for player_id in policy_ids
            if minutes[player_id] > 0.0
        }
        if armband_player is not None:
            weights[str(armband_player)] += 1.0
    elif action == cd.CHIP_ACTION_TC and arm_name == "play" and armband_player is not None:
        # The normal arm already has the standard extra captain copy; TC adds
        # one more copy to the same captain/vice selected by appearance.
        weights[str(armband_player)] += 1.0
    return dict(sorted(weights.items(), key=lambda item: int(item[0])))


def _verify_forecast_matches_save_state(
    forecast: Mapping[str, Any], save_arm: Mapping[str, Any],
) -> None:
    """Validate origin and selected-event SAVE states at their own grain.

    The forecast-level state is the manager's state after the origin SAVE route.
    A selected opportunity can occur several events later, after normal FT/state
    transitions. Requiring those states to be byte-identical incorrectly
    rejects a valid continuation and erases the event-grain state contract.
    """

    forecast_state = forecast.get("reservation_state")
    if not isinstance(forecast_state, Mapping) or not forecast_state:
        raise ReservationCalibrationError("raw reservation forecast omits its origin SAVE state")
    if forecast.get("reservation_state_sha256") != _canonical_sha256(forecast_state):
        raise ReservationCalibrationError("raw reservation forecast origin SAVE-state digest does not verify")

    try:
        selected_event = int(forecast.get("selected_event"))
    except (TypeError, ValueError) as failure:
        raise ReservationCalibrationError("raw reservation forecast has no selected event") from failure
    opportunities = forecast.get("opportunities")
    selected = [
        item.get("artifact_payload")
        for item in opportunities or ()
        if isinstance(item, Mapping)
        and isinstance(item.get("artifact_payload"), Mapping)
        and int(item["artifact_payload"].get("event") or -1) == selected_event
    ]
    if len(selected) != 1:
        raise ReservationCalibrationError("forecast does not retain exactly one selected-event SAVE state")
    expected_arms = selected[0].get("outcome_arms")
    expected_save = expected_arms.get("save") if isinstance(expected_arms, Mapping) else None
    selected_state = selected[0].get("reservation_state")
    arm_state = expected_save.get("reservation_state") if isinstance(expected_save, Mapping) else None
    if (
        not isinstance(selected_state, Mapping)
        or not isinstance(arm_state, Mapping)
        or dict(selected_state) != dict(arm_state)
        or selected[0].get("reservation_state_sha256") != _canonical_sha256(selected_state)
    ):
        raise ReservationCalibrationError("selected-event SAVE state differs from its opportunity arms")
    # The retained causal SAVE arm is checked byte-for-byte against these
    # selected-event opportunity arms by _verify_pair_matches_selected_forecast.
    if not isinstance(save_arm, Mapping) or dict(save_arm.get("reservation_state") or {}) != dict(arm_state):
        raise ReservationCalibrationError("matured SAVE arm differs from the selected-event SAVE state")


def _verify_pair_matches_selected_forecast(
    action: str,
    forecast: Mapping[str, Any],
    pair: Mapping[str, Any],
) -> None:
    """Bind causal arms to the selected event and any frozen origin policies."""

    try:
        selected_event = int(forecast["selected_event"])
    except (KeyError, TypeError, ValueError) as failure:
        raise ReservationCalibrationError("raw reservation forecast has no selected event") from failure
    if set(pair) != {"play", "save"}:
        raise ReservationCalibrationError("causal evidence does not retain exactly one PLAY/SAVE pair")
    for role in ("play", "save"):
        retained_arm = pair.get(role)
        payload = retained_arm.get("artifact_payload") if isinstance(retained_arm, Mapping) else None
        if not isinstance(payload, Mapping):
            raise ReservationCalibrationError(f"retained {role.upper()} policy is malformed")
        expected_role = role.upper()
        for location, artifact in (("manifest", retained_arm), ("artifact", payload)):
            declared_event = artifact.get("event")
            if (
                isinstance(declared_event, bool)
                or not isinstance(declared_event, int)
                or declared_event != selected_event
            ):
                raise ReservationCalibrationError(
                    f"retained {role.upper()} {location} event differs from the selected forecast opportunity"
                )
            if artifact.get("action") != action:
                raise ReservationCalibrationError(
                    f"retained {role.upper()} {location} action differs from the selected forecast opportunity"
                )
            if artifact.get("counterfactual_role") != expected_role:
                raise ReservationCalibrationError(
                    f"retained {role.upper()} {location} role differs from the selected forecast opportunity"
                )

    opportunities = forecast.get("opportunities")
    if not isinstance(opportunities, list):
        raise ReservationCalibrationError("raw reservation forecast has no selected event manifest")
    selected_payloads = []
    for item in opportunities:
        payload = item.get("artifact_payload") if isinstance(item, Mapping) else None
        if isinstance(payload, Mapping):
            try:
                event = int(payload.get("event"))
            except (TypeError, ValueError):
                continue
            if event == selected_event:
                selected_payloads.append(payload)
    if len(selected_payloads) != 1:
        raise ReservationCalibrationError(
            "raw reservation forecast does not identify exactly one selected event opportunity"
        )
    expected_arms = selected_payloads[0].get("outcome_arms")
    if not isinstance(expected_arms, Mapping) or set(expected_arms) != {"play", "save"}:
        raise ReservationCalibrationError(
            "selected forecast opportunity omits its frozen PLAY/SAVE policies"
        )
    pair_payloads = {
        role: pair[role].get("artifact_payload") if isinstance(pair.get(role), Mapping) else None
        for role in ("play", "save")
    }
    if not all(isinstance(payload, Mapping) for payload in pair_payloads.values()):
        raise ReservationCalibrationError("retained selected-event causal arms are malformed")
    _verify_action_pair_policies(
        action,
        pair_payloads["play"],
        pair_payloads["save"],
        planning_event=int(forecast.get("planning_event") or -1),
    )
    for role in ("play", "save"):
        retained_arm = pair.get(role)
        expected_arm = expected_arms.get(role)
        if not isinstance(retained_arm, Mapping) or not isinstance(expected_arm, Mapping):
            raise ReservationCalibrationError("selected forecast opportunity has a malformed causal arm")
        payload = retained_arm.get("artifact_payload")
        if not isinstance(payload, Mapping) or dict(payload) != dict(expected_arm):
            raise ReservationCalibrationError(
                f"retained {role.upper()} policy does not match the selected forecast opportunity"
            )
        expected_digest = _canonical_sha256(expected_arm)
        if (
            str(retained_arm.get("artifact_sha256") or "") != expected_digest
            or str(retained_arm.get("arm_id") or "") != str(expected_arm.get("arm_id") or "")
        ):
            raise ReservationCalibrationError(
                f"retained {role.upper()} identity does not match the selected forecast opportunity"
            )
        for name, value in expected_arm.items():
            if retained_arm.get(name) != value:
                raise ReservationCalibrationError(
                    f"retained {role.upper()} {name} differs from the selected forecast opportunity"
                )


def _verify_action_pair_policies(
    action: str,
    play: Mapping[str, Any],
    save: Mapping[str, Any],
    *,
    planning_event: int,
) -> None:
    """Apply the comparison contract specific to each chip's counterfactual.

    BB/TC compare the same explicit proposed squad and lineup. FH/WC change
    squad and/or lineup by definition, so those pairs instead bind each legal
    policy to its own explicit transfer state and chip transition semantics.
    """

    if action in {cd.CHIP_ACTION_BB, cd.CHIP_ACTION_TC}:
        for name in ("proposed_squad_ids", "lineup", "player_positions"):
            if play.get(name) != save.get(name) or play.get(name) in (None, "", [], {}):
                raise ReservationCalibrationError(f"PLAY and SAVE do not share the same {name}")
        return

    transfer_states: dict[str, Mapping[str, Any]] = {}
    for arm_name, arm, expected_role in (
        ("PLAY", play, "PLAY"), ("SAVE", save, "SAVE"),
    ):
        if arm.get("counterfactual_role") != expected_role:
            raise ReservationCalibrationError(f"{action} {arm_name} arm has the wrong counterfactual role")
        semantics = arm.get("action_semantics")
        state = semantics.get("transfer_state") if isinstance(semantics, Mapping) else None
        if not isinstance(semantics, Mapping) or not isinstance(state, Mapping):
            raise ReservationCalibrationError(f"{action} {arm_name} arm omits explicit transfer-state semantics")
        transfer_states[expected_role] = state
        if semantics.get("role") != expected_role:
            raise ReservationCalibrationError(f"{action} {arm_name} action semantics disagree with its arm role")
        try:
            state_squad = tuple(int(value) for value in state["squad_ids"])
            prices = {int(key): int(value) for key, value in state["purchase_price_tenths"].items()}
            bank = int(state["bank_tenths"])
            free_transfers = int(state["free_transfers"])
        except (KeyError, TypeError, ValueError, AttributeError) as failure:
            raise ReservationCalibrationError(
                f"{action} {arm_name} transfer state is incomplete: {failure}"
            ) from failure
        if (
            len(state_squad) != ml.SQUAD_SIZE
            or len(set(state_squad)) != ml.SQUAD_SIZE
            or bank < 0
            or free_transfers < 0
            or set(prices) != set(state_squad)
            or tuple(sorted(state_squad)) != tuple(sorted(int(value) for value in arm.get("proposed_squad_ids", ())))
        ):
            raise ReservationCalibrationError(
                f"{action} {arm_name} transfer state does not bind its complete 15-player policy"
            )

    play_semantics = play["action_semantics"]
    save_semantics = save["action_semantics"]
    _verify_valuation_schedules(action, play, save, planning_event=planning_event)
    if action == cd.CHIP_ACTION_FH:
        restoration = play_semantics.get("restore_at_h2")
        if not isinstance(restoration, Mapping):
            raise ReservationCalibrationError("FH PLAY omits its permanent-state restoration record")
        required_fields = (
            "restore_event", "permanent_squad_ids", "restored_squad_ids",
            "permanent_purchase_price_tenths", "restored_purchase_price_tenths",
            "permanent_bank_tenths", "restored_bank_tenths",
            "event_start_h1_free_transfers", "restored_h2_free_transfers",
            "restored_h2_start_state",
        )
        missing = [name for name in required_fields if name not in restoration]
        if missing:
            raise ReservationCalibrationError(
                "FH PLAY permanent-state restoration record is incomplete: " + ", ".join(missing)
            )

        def strict_int(value: Any, name: str) -> int:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ReservationCalibrationError(f"FH PLAY restoration {name} must be a non-negative integer")
            return value

        def squad_ids(
            value: Any, name: str, *, require_squad_size: bool = True,
        ) -> tuple[int, ...]:
            if not isinstance(value, (list, tuple)):
                raise ReservationCalibrationError(f"FH PLAY restoration {name} must be a player-id list")
            values = tuple(strict_int(item, name) for item in value)
            if len(set(values)) != len(values) or (
                require_squad_size and len(values) != ml.SQUAD_SIZE
            ):
                raise ReservationCalibrationError(f"FH PLAY restoration {name} must contain 15 unique players")
            return values

        def prices(value: Any, name: str) -> dict[int, int]:
            if not isinstance(value, Mapping):
                raise ReservationCalibrationError(f"FH PLAY restoration {name} must be a purchase-price mapping")
            result: dict[int, int] = {}
            try:
                for key, item in value.items():
                    if isinstance(key, bool):
                        raise ValueError("boolean player id")
                    player_id = int(key)
                    if player_id in result:
                        raise ValueError("duplicate player id")
                    result[player_id] = strict_int(item, name)
            except (TypeError, ValueError) as failure:
                raise ReservationCalibrationError(f"FH PLAY restoration {name} is malformed") from failure
            return result

        permanent_squad = squad_ids(restoration["permanent_squad_ids"], "permanent_squad_ids")
        restored_squad = squad_ids(
            restoration["restored_squad_ids"], "restored_squad_ids", require_squad_size=False,
        )
        permanent_prices = prices(
            restoration["permanent_purchase_price_tenths"], "permanent_purchase_price_tenths",
        )
        restored_prices = prices(
            restoration["restored_purchase_price_tenths"], "restored_purchase_price_tenths",
        )
        permanent_bank = strict_int(restoration["permanent_bank_tenths"], "permanent_bank_tenths")
        restored_bank = strict_int(restoration["restored_bank_tenths"], "restored_bank_tenths")
        event_start_ft = strict_int(
            restoration["event_start_h1_free_transfers"], "event_start_h1_free_transfers",
        )
        restored_ft = strict_int(restoration["restored_h2_free_transfers"], "restored_h2_free_transfers")
        if (
            set(permanent_squad) != set(restored_squad)
            or permanent_prices != restored_prices
            or permanent_bank != restored_bank
            or event_start_ft != restored_ft
        ):
            raise ReservationCalibrationError("FH PLAY does not preserve permanent squad, bank, basis and event-start FT")
        h2_start = restoration["restored_h2_start_state"]
        if not isinstance(h2_start, Mapping):
            raise ReservationCalibrationError("FH PLAY restored H2 route start state is malformed")
        try:
            h2_start_event = strict_int(h2_start["event"], "restored_h2_start_state.event")
            h2_start_squad = squad_ids(
                h2_start["squad_ids"], "restored_h2_start_state.squad_ids",
            )
            h2_start_prices = prices(
                h2_start["purchase_price_tenths"],
                "restored_h2_start_state.purchase_price_tenths",
            )
            h2_start_bank = strict_int(
                h2_start["bank_tenths"], "restored_h2_start_state.bank_tenths",
            )
            h2_start_ft = strict_int(
                h2_start["free_transfers"], "restored_h2_start_state.free_transfers",
            )
        except (KeyError, TypeError, ValueError, AttributeError) as failure:
            raise ReservationCalibrationError(
                f"FH PLAY restored H2 route start state is incomplete: {failure}"
            ) from failure
        if (
            h2_start_event != strict_int(restoration["restore_event"], "restore_event")
            or set(h2_start_squad) != set(restored_squad)
            or h2_start_prices != restored_prices
            or h2_start_bank != restored_bank
            or h2_start_ft != restored_ft
        ):
            raise ReservationCalibrationError(
                "FH PLAY canonical H2 route does not start from the restored permanent manager state"
            )
        origin_state = play_semantics.get("origin_manager_state")
        save_origin_state = save_semantics.get("origin_manager_state")
        if origin_state is None and save_origin_state is None:
            legacy_save_state = transfer_states["SAVE"]
            origin_state = {
                "event": int(planning_event),
                "squad_ids": list(legacy_save_state.get("squad_ids") or ()),
                "bank_tenths": int(legacy_save_state.get("bank_tenths", -1)),
                "free_transfers": int(legacy_save_state.get("event_start_free_transfers", -1)),
                "purchase_price_tenths": dict(legacy_save_state.get("purchase_price_tenths") or {}),
            }
            save_origin_state = origin_state
        try:
            if not isinstance(origin_state, Mapping) or not isinstance(save_origin_state, Mapping):
                raise KeyError("original permanent manager state")
            save_squad = squad_ids(origin_state["squad_ids"], "origin manager squad_ids")
            save_prices = prices(origin_state["purchase_price_tenths"], "origin manager purchase_price_tenths")
            save_bank = strict_int(origin_state["bank_tenths"], "origin manager bank_tenths")
            save_event_start_ft = strict_int(
                origin_state["free_transfers"], "origin manager free_transfers",
            )
            play_event = strict_int(play["event"], "PLAY event")
            restore_event = strict_int(restoration["restore_event"], "restore_event")
        except (KeyError, TypeError, ValueError, AttributeError) as failure:
            raise ReservationCalibrationError(
                f"FH PLAY restoration cannot be bound to the canonical SAVE state/event: {failure}"
            ) from failure
        if (
            dict(origin_state) != dict(save_origin_state)
            or set(permanent_squad) != set(save_squad)
            or permanent_prices != save_prices
            or permanent_bank != save_bank
            or event_start_ft != save_event_start_ft
        ):
            raise ReservationCalibrationError(
                "FH PLAY restoration does not bind to the canonical permanent SAVE manager state"
            )
        if play_event <= int(planning_event) or restore_event != play_event + 1:
            raise ReservationCalibrationError(
                "FH permanent-state restoration must occur in H2 after the actual PLAY event"
            )
        if save_semantics.get("retains_chip_option") is not True:
            raise ReservationCalibrationError("FH SAVE arm does not retain the Free Hit option")
    elif action == cd.CHIP_ACTION_WC:
        if play_semantics.get("wildcard_applied") is not True:
            raise ReservationCalibrationError("WC PLAY arm does not declare the Wildcard transfer action")
        if save_semantics.get("wildcard_applied") is not False or save_semantics.get("retains_chip_option") is not True:
            raise ReservationCalibrationError("WC SAVE arm does not preserve the Wildcard option")
        try:
            event_start_ft = int(play_semantics["event_start_free_transfers"])
            free_transfers_after = int(play_semantics["free_transfers_after"])
            max_free_transfers = int(play_semantics["max_free_transfers"])
        except (KeyError, TypeError, ValueError) as failure:
            raise ReservationCalibrationError("WC PLAY omits explicit event-start FT transition semantics") from failure
        if (
            event_start_ft < 0 or max_free_transfers < 1
            or free_transfers_after != min(event_start_ft, max_free_transfers)
        ):
            raise ReservationCalibrationError("WC PLAY free-transfer retention does not follow the declared rule")
        if not isinstance(play.get("valuation_schedule"), Mapping):
            return
        origin_state = play_semantics.get("origin_manager_state")
        save_origin_state = save_semantics.get("origin_manager_state")
        origin_digest = _canonical_sha256(origin_state) if isinstance(origin_state, Mapping) else ""
        if (
            not isinstance(origin_state, Mapping)
            or dict(origin_state) != dict(save_origin_state or {})
            or play_semantics.get("origin_manager_state_sha256") != origin_digest
            or save_semantics.get("origin_manager_state_sha256") != origin_digest
            or int(origin_state.get("event", -1)) != int(play.get("event") or -2)
            or int(origin_state.get("free_transfers", -1)) != event_start_ft
        ):
            raise ReservationCalibrationError("WC arms do not bind the same future permanent manager state")
        play_transaction = play_semantics.get("transaction")
        if not isinstance(play_transaction, Mapping):
            raise ReservationCalibrationError("WC PLAY omits its canonical permanent-squad transaction record")
        origin_squad = {int(value) for value in origin_state.get("squad_ids") or ()}
        origin_prices = {
            int(key): int(value)
            for key, value in (origin_state.get("purchase_price_tenths") or {}).items()
        }
        play_rows = (play.get("valuation_schedule") or {}).get("events") or []
        save_rows = (save.get("valuation_schedule") or {}).get("events") or []
        play_squad = {int(value) for value in play_rows[0].get("proposed_squad_ids") or ()} if play_rows else set()
        play_state = play_rows[0].get("transfer_state") or {} if play_rows else {}
        play_prices = {
            int(key): int(value)
            for key, value in (play_state.get("purchase_price_tenths") or {}).items()
        }
        retained = {int(value) for value in play_transaction.get("retained_ids") or ()}
        sold = {int(value) for value in play_transaction.get("sold_ids") or ()}
        bought = {int(value) for value in play_transaction.get("bought_ids") or ()}
        rebought = {int(value) for value in play_transaction.get("rebought_ids") or ()}
        new_basis = {
            int(key): int(value) for key, value in (play_transaction.get("new_basis") or {}).items()
        }
        if (
            retained != (origin_squad & play_squad) - rebought
            or sold != (origin_squad - play_squad) | rebought
            or bought != (play_squad - origin_squad) | rebought
            or set(play_prices) != play_squad
            or any(play_prices.get(pid) != origin_prices.get(pid) for pid in retained)
            or any(play_prices.get(pid) != new_basis.get(pid) for pid in bought | rebought)
            or int(play_transaction.get("remaining_bank_tenths", -1)) != int(play_state.get("bank_tenths", -2))
            or int(play_state.get("free_transfers", -1)) != free_transfers_after
        ):
            raise ReservationCalibrationError("WC PLAY transaction does not reproduce squad, basis, bank or FT semantics")
        if not play_rows or not save_rows:
            raise ReservationCalibrationError("WC schedules are missing the declared value horizon")
        if play_rows[0]["event"] != int(play.get("event") or -1):
            raise ReservationCalibrationError("WC PLAY top-level event differs from its first scheduled event")
        if play_rows[0].get("wildcard_active") is not True or any(
            row.get("wildcard_active") is True for row in play_rows[1:]
        ):
            raise ReservationCalibrationError("WC PLAY schedule does not apply the chip only in its selected event")
        for previous, current in zip(play_rows, play_rows[1:]):
            prior_ft = int(previous["transfer_state"]["free_transfers"])
            current_ft = int(current["transfer_state"]["free_transfers"])
            if current_ft != min(prior_ft + 1, max_free_transfers):
                raise ReservationCalibrationError("WC PLAY schedule violates its declared inter-event FT progression")


def _verify_valuation_schedules(
    action: str,
    play: Mapping[str, Any],
    save: Mapping[str, Any],
    *,
    planning_event: int,
) -> None:
    """Validate explicit multi-event PLAY/SAVE schedules when a producer supplies them.

    Legacy one-event records remain readable. New FH/WC production records set
    ``valuation_schedule_required`` and therefore cannot train calibration from
    a single selected-event score.
    """

    play_semantics = play.get("action_semantics")
    save_semantics = save.get("action_semantics")
    play_required = bool(
        isinstance(play_semantics, Mapping)
        and play_semantics.get("valuation_schedule_required") is True
    )
    save_required = bool(
        isinstance(save_semantics, Mapping)
        and save_semantics.get("valuation_schedule_required") is True
    )
    play_schedule = play.get("valuation_schedule")
    save_schedule = save.get("valuation_schedule")
    if not (play_required or save_required or play_schedule is not None or save_schedule is not None):
        return
    if play_required != save_required or not isinstance(play_schedule, Mapping) or not isinstance(save_schedule, Mapping):
        raise ReservationCalibrationError(f"{action} requires matching retained PLAY/SAVE valuation schedules")

    def rows(schedule: Mapping[str, Any], role: str) -> list[dict[str, Any]]:
        raw = schedule.get("events")
        if not isinstance(raw, list) or not raw:
            raise ReservationCalibrationError(f"{action} {role} valuation schedule is empty")
        result: list[dict[str, Any]] = []
        seen: set[int] = set()
        for item in raw:
            if not isinstance(item, Mapping):
                raise ReservationCalibrationError(f"{action} {role} valuation schedule row is malformed")
            try:
                event = int(item["event"])
                weight = _number(item["weight"], name=f"{action} {role} event weight")
                expected_points = _number(
                    item["expected_points"], name=f"{action} {role} expected event points",
                )
                squad = tuple(int(value) for value in item["proposed_squad_ids"])
                lineup = item["lineup"]
                positions = item["player_positions"]
                state = item["transfer_state"]
            except (KeyError, TypeError, ValueError, AttributeError) as failure:
                raise ReservationCalibrationError(
                    f"{action} {role} valuation schedule omits event policy/state: {failure}"
                ) from failure
            if event in seen or weight <= 0.0 or len(squad) != ml.SQUAD_SIZE or len(set(squad)) != ml.SQUAD_SIZE:
                raise ReservationCalibrationError(f"{action} {role} valuation schedule has invalid event/squad/weight")
            if not isinstance(lineup, Mapping) or not isinstance(positions, Mapping) or not isinstance(state, Mapping):
                raise ReservationCalibrationError(f"{action} {role} valuation schedule policy/state is malformed")
            if {int(key) for key in positions} != set(squad):
                raise ReservationCalibrationError(f"{action} {role} event {event} positions do not cover its squad")
            if {int(value) for value in state.get("squad_ids", ())} != set(squad):
                raise ReservationCalibrationError(f"{action} {role} event {event} transfer state differs from its squad")
            try:
                policy = ml.ManagerPolicy(
                    starter_ids=tuple(int(value) for value in lineup["starter_ids"]),
                    bench_gk_id=int(lineup["bench_gk_id"]),
                    bench_outfield_order=tuple(int(value) for value in lineup["bench_outfield_order"]),
                    captain_id=int(lineup["captain_id"]),
                    vice_captain_id=int(lineup["vice_captain_id"]),
                )
                normalized_positions = {int(key): str(value) for key, value in positions.items()}
            except (KeyError, TypeError, ValueError, AttributeError) as failure:
                raise ReservationCalibrationError(
                    f"{action} {role} event {event} lineup is malformed: {failure}"
                ) from failure
            if set(ml.policy_player_ids(policy)) != set(squad):
                raise ReservationCalibrationError(f"{action} {role} event {event} lineup differs from its squad")
            legality = ml.policy_legality_errors(policy, normalized_positions)
            if legality:
                raise ReservationCalibrationError(
                    f"{action} {role} event {event} lineup is illegal: {legality}"
                )
            seen.add(event)
            result.append({
                **dict(item), "event": event, "weight": weight,
                "expected_points": expected_points,
            })
        events = [row["event"] for row in result]
        if events != list(range(events[0], events[0] + len(events))):
            raise ReservationCalibrationError(f"{action} {role} valuation schedule events are not contiguous")
        if abs(sum(row["weight"] for row in result) - 1.0) > 1e-9:
            raise ReservationCalibrationError(f"{action} {role} valuation schedule weights must sum to one")
        return result

    play_rows, save_rows = rows(play_schedule, "PLAY"), rows(save_schedule, "SAVE")
    if len(play_rows) != len(save_rows) or [r["event"] for r in play_rows] != [r["event"] for r in save_rows]:
        raise ReservationCalibrationError(f"{action} PLAY/SAVE valuation schedules cover different events")
    if [r["weight"] for r in play_rows] != [r["weight"] for r in save_rows]:
        raise ReservationCalibrationError(f"{action} PLAY/SAVE valuation schedules use different event weights")
    if play_rows[0]["event"] != int(play.get("event") or -1) or save_rows[0]["event"] != int(save.get("event") or -1):
        raise ReservationCalibrationError(f"{action} top-level arms differ from their first scheduled event")
    if action == cd.CHIP_ACTION_FH:
        if len(play_rows) != cd.CHIP_HORIZON_LENGTH:
            raise ReservationCalibrationError("FH reservation outcome must cover exactly the four-event horizon")
        if any(abs(row["weight"] - 1.0 / cd.CHIP_HORIZON_LENGTH) > 1e-9 for row in play_rows):
            raise ReservationCalibrationError("FH schedule must use the normalized four-event comparison basis")
        if play_rows[0]["event"] <= int(planning_event):
            raise ReservationCalibrationError("FH valuation schedule does not start after its origin")
        restoration = ((play.get("action_semantics") or {}).get("restore_at_h2") or {})
        if int(restoration.get("restore_event") or -1) != play_rows[1]["event"]:
            raise ReservationCalibrationError("FH valuation schedule does not restore permanent state in H2")
        restored_squad = {int(value) for value in restoration.get("restored_squad_ids") or ()}
        restored_prices = {int(key): int(value) for key, value in (restoration.get("restored_purchase_price_tenths") or {}).items()}
        restored_bank = int(restoration.get("restored_bank_tenths", -1))
        restored_ft = int(restoration.get("restored_h2_free_transfers", -1))
        second = play_rows[1]
        second_state = second["transfer_state"]
        if (
            int(second["event"]) != int(restoration["restore_event"])
            or int(second_state.get("event", -1)) != int(restoration["restore_event"])
        ):
            raise ReservationCalibrationError(
                "FH H2 PLAY schedule event differs from its permanent restoration event"
            )
    elif action == cd.CHIP_ACTION_WC:
        if not 6 <= len(play_rows) <= 10:
            raise ReservationCalibrationError("WC valuation schedule must preserve its separate 6–10 event horizon")
        play_squad = set(int(value) for value in play_rows[0]["proposed_squad_ids"])
        if any(set(int(value) for value in row["proposed_squad_ids"]) != play_squad for row in play_rows):
            raise ReservationCalibrationError("WC PLAY schedule changes the Wildcard squad within its value horizon")

    for role, rows_for_arm, arm in (("PLAY", play_rows, play), ("SAVE", save_rows, save)):
        value_definition = (arm.get("valuation_schedule") or {}).get("value_definition")
        expected_definition = (
            "FOUR_EVENT_NORMALIZED_MEAN_POINTS" if action == cd.CHIP_ACTION_FH
            else "WEIGHTED_WILDCARD_HORIZON_MEAN_POINTS"
        )
        if value_definition != expected_definition:
            raise ReservationCalibrationError(f"{action} {role} schedule uses an incompatible value definition")
        reproduced = sum(row["weight"] * row["expected_points"] for row in rows_for_arm)
        if abs(reproduced - _number(arm.get("paired_value"), name=f"{action} {role} paired value")) > 1e-6:
            raise ReservationCalibrationError(f"{action} {role} expected event values do not reproduce paired value")

    if action == cd.CHIP_ACTION_FH:
        play_semantics = play.get("action_semantics") or {}
        save_semantics = save.get("action_semantics") or {}
        original = play_semantics.get("origin_manager_state")
        save_original = save_semantics.get("origin_manager_state")
        original_digest = _canonical_sha256(original) if isinstance(original, Mapping) else ""
        restoration = play_semantics.get("restore_at_h2") or {}
        original_squad = {int(value) for value in (original or {}).get("squad_ids") or ()}
        original_prices = {
            int(key): int(value)
            for key, value in ((original or {}).get("purchase_price_tenths") or {}).items()
        }
        if (
            not isinstance(original, Mapping)
            or dict(original) != dict(save_original or {})
            or play_semantics.get("origin_manager_state_sha256") != original_digest
            or save_semantics.get("origin_manager_state_sha256") != original_digest
            or int(original.get("event", -1)) != int(play.get("event") or -2)
            or original_squad != {int(value) for value in restoration.get("permanent_squad_ids") or ()}
            or original_squad != {int(value) for value in restoration.get("restored_squad_ids") or ()}
            or original_prices != {
                int(key): int(value)
                for key, value in (restoration.get("permanent_purchase_price_tenths") or {}).items()
            }
            or original_prices != {
                int(key): int(value)
                for key, value in (restoration.get("restored_purchase_price_tenths") or {}).items()
            }
            or int(original.get("bank_tenths", -1)) != int(restoration.get("permanent_bank_tenths", -2))
            or int(original.get("bank_tenths", -1)) != int(restoration.get("restored_bank_tenths", -2))
            or int(original.get("free_transfers", -1)) != int(restoration.get("event_start_h1_free_transfers", -2))
            or int(original.get("free_transfers", -1)) != int(restoration.get("restored_h2_free_transfers", -2))
            or str(restoration.get("origin_manager_state_sha256") or "") != original_digest
        ):
            raise ReservationCalibrationError("FH PLAY restoration does not reproduce the original permanent state")


def _verify_outcome_record(
    outcome_record: Mapping[str, Any],
    *,
    evidence: Mapping[str, Any],
    row: Mapping[str, Any],
    label: Mapping[str, Any],
    play: Mapping[str, Any],
    save: Mapping[str, Any],
) -> float:
    """Bind a matured paired score to its origin, arms and official captures."""

    if outcome_record.get("schema") != CAUSAL_OUTCOME_RECORD_SCHEMA:
        raise ReservationCalibrationError("retained outcome record has an unsupported schema")
    identity = {
        "observation_id": row.get("observation_id"),
        "action": row.get("action"),
        "planning_event": int(row.get("planning_event")),
        "origin_cutoff": row.get("origin_cutoff"),
        "scenario_identity": play.get("scenario_identity"),
        "world_identity": play.get("world_identity"),
        "play_arm_id": play.get("arm_id"),
        "play_artifact_sha256": play.get("artifact_sha256"),
        "save_arm_id": save.get("arm_id"),
        "save_artifact_sha256": save.get("artifact_sha256"),
    }
    if not str(outcome_record.get("record_id") or "").strip():
        raise ReservationCalibrationError("retained outcome record has no record id")
    if any(outcome_record.get(name) != value for name, value in identity.items()):
        raise ReservationCalibrationError(
            "retained outcome record does not match its observation, action, scenario or paired arms"
        )
    source_identity = evidence.get("source")
    if not isinstance(source_identity, Mapping):
        raise ReservationCalibrationError("causal evidence has no source identity")
    for name in (
        "source_decision_id", "source_result_sha256", "source_artifact_sha256",
        "generation_id", "planning_event", "cutoff", "data_snapshot_sha256",
        "predictive_code_snapshot_sha256", "certification_identity", "world_identity",
    ):
        if str(outcome_record.get(name) or "") != str(source_identity.get(name) or ""):
            raise ReservationCalibrationError(f"retained outcome record is not bound to source {name}")
    if outcome_record.get("label_definition_version") != OUTCOME_LABEL_DEFINITION:
        raise ReservationCalibrationError("retained outcome record uses an unsupported paired-label definition")
    if outcome_record.get("scoring_rule_version") != OUTCOME_SCORING_RULE_VERSION:
        raise ReservationCalibrationError("retained outcome record uses an unsupported canonical scoring rule")

    source = outcome_record.get("source")
    captures = source.get("captures") if isinstance(source, Mapping) else None
    if (
        not isinstance(source, Mapping)
        or source.get("schema") != OUTCOME_CAPTURE_SET_SCHEMA
        or not isinstance(captures, list)
        or not captures
    ):
        raise ReservationCalibrationError("retained outcome record has no official capture set")
    realization_event = int(outcome_record.get("realization_event") or -1)
    if realization_event <= int(row.get("planning_event") or -1):
        raise ReservationCalibrationError("paired outcome is not from a future event")
    play_schedule = play.get("valuation_schedule")
    save_schedule = save.get("valuation_schedule")
    if isinstance(play_schedule, Mapping) or isinstance(save_schedule, Mapping):
        _verify_valuation_schedules(
            str(row.get("action") or ""), play, save,
            planning_event=int(row.get("planning_event") or -1),
        )
        scheduled_events = [int(item["event"]) for item in play_schedule["events"]]
    else:
        scheduled_events = [realization_event]
    if scheduled_events[0] != realization_event:
        raise ReservationCalibrationError("outcome schedule does not start at the selected realization event")
    if outcome_record.get("realization_events", scheduled_events) != scheduled_events:
        raise ReservationCalibrationError("outcome record realization event set differs from the retained schedules")
    player_points: dict[int, dict[str, float]] = {event: {} for event in scheduled_events}
    player_minutes: dict[int, dict[str, float]] = {event: {} for event in scheduled_events}
    capture_ids: set[str] = set()
    capture_times: list[datetime] = []
    capture_events: set[int] = set()
    for capture in captures:
        if not isinstance(capture, Mapping):
            raise ReservationCalibrationError("outcome capture manifest is malformed")
        digest = str(capture.get("capture_digest") or "")
        if not _is_sha256(digest) or digest in capture_ids:
            raise ReservationCalibrationError("outcome capture manifest has an invalid or duplicate digest")
        capture_ids.add(digest)
        capture_event = int(capture.get("event") or -1)
        if (
            capture.get("grain") != "player_event"
            or capture_event not in player_points
            or capture.get("fixture_id") is not None
            or capture.get("observation_state") != ol.OBSERVATION_FINAL
            or capture.get("source_name") != "player_gameweeks_final"
            or capture.get("source_identity") != f"player_gameweeks:{capture_event}"
        ):
            raise ReservationCalibrationError(
                "outcome capture is not a final event-grain official player-gameweek result"
            )
        player_id = str(int(capture.get("player_id")))
        if player_id in player_points[capture_event]:
            raise ReservationCalibrationError("outcome capture set repeats a player in a realization event")
        player_points[capture_event][player_id] = _number(capture.get("total_points"), name="captured player total_points")
        minutes = _number(capture.get("minutes"), name="captured player minutes")
        if minutes < 0.0:
            raise ReservationCalibrationError("captured player minutes cannot be negative")
        player_minutes[capture_event][player_id] = minutes
        capture_events.add(capture_event)
        final_at = _utc(capture.get("official_final_at"), name="outcome official_final_at")
        captured_at = _utc(capture.get("captured_at"), name="outcome captured_at")
        if captured_at < final_at:
            raise ReservationCalibrationError("outcome capture predates official finality")
        capture_times.append(captured_at)
    available_at = max(capture_times)
    if capture_events != set(scheduled_events):
        raise ReservationCalibrationError("outcome capture set does not cover every scheduled realization event")
    declared_available = _utc(outcome_record.get("available_at"), name="outcome available_at")
    source_available = _utc(source.get("available_at"), name="capture-set available_at")
    if declared_available != available_at or source_available != available_at:
        raise ReservationCalibrationError("outcome availability does not match its latest official capture")
    if (
        available_at != _utc(label.get("available_at"), name="causal label available_at")
        or available_at != _utc(row.get("label_available_at"), name="row label_available_at")
    ):
        raise ReservationCalibrationError("outcome capture availability differs from the causal label")
    if available_at <= _utc(row.get("origin_cutoff"), name="origin_cutoff"):
        raise ReservationCalibrationError("paired outcome was available at or before its origin")

    paired_results = outcome_record.get("paired_results")
    if not isinstance(paired_results, Mapping):
        raise ReservationCalibrationError("outcome record has no paired PLAY/SAVE results")
    scores: dict[str, float] = {}
    for arm_name, arm in (("play", play), ("save", save)):
        result = paired_results.get(arm_name)
        if not isinstance(result, Mapping):
            raise ReservationCalibrationError(f"{arm_name.upper()} outcome is malformed")
        schedule = arm.get("valuation_schedule")
        if isinstance(schedule, Mapping):
            schedule_rows = [dict(item) for item in schedule.get("events") or ()]
        else:
            schedule_rows = [{
                "event": realization_event, "weight": 1.0,
                "proposed_squad_ids": arm.get("proposed_squad_ids"),
                "lineup": arm.get("lineup"),
                "player_positions": arm.get("player_positions"),
            }]
        declared_event_results = result.get("events")
        if declared_event_results is None and not isinstance(schedule, Mapping):
            # Read legacy single-event outcome records without weakening the new
            # FH/WC schedule contract. New multi-event producers always emit an
            # explicit event result list.
            if (
                result.get("lineup") != arm.get("lineup")
                or result.get("player_positions") != arm.get("player_positions")
            ):
                raise ReservationCalibrationError(
                    f"{arm_name.upper()} outcome scorer differs from its retained lineup or source positions"
                )
            declared_event_results = [{
                "event": realization_event, "weight": 1.0,
                "lineup": result.get("lineup"),
                "player_positions": result.get("player_positions"),
                "scoring_weights": result.get("scoring_weights"),
                "observed_points": result.get("observed_points"),
            }]
        if not isinstance(declared_event_results, list) or len(declared_event_results) != len(schedule_rows):
            raise ReservationCalibrationError(f"{arm_name.upper()} outcome omits per-event scoring results")
        weighted_total = 0.0
        rebuilt_event_results: list[dict[str, Any]] = []
        for scheduled, declared in zip(schedule_rows, declared_event_results):
            if not isinstance(declared, Mapping):
                raise ReservationCalibrationError(f"{arm_name.upper()} per-event scoring result is malformed")
            event = int(scheduled["event"])
            event_arm = {
                **dict(arm),
                "proposed_squad_ids": scheduled["proposed_squad_ids"],
                "lineup": scheduled["lineup"],
                "player_positions": scheduled["player_positions"],
            }
            if (
                declared.get("event") != event
                or declared.get("lineup") != scheduled["lineup"]
                or declared.get("player_positions") != scheduled["player_positions"]
                or _number(declared.get("weight"), name=f"{arm_name} event weight")
                != _number(scheduled.get("weight", 1.0), name=f"{arm_name} schedule weight")
            ):
                raise ReservationCalibrationError(f"{arm_name.upper()} outcome differs from its retained event schedule")
            result_weights = declared.get("scoring_weights")
            if not isinstance(result_weights, Mapping):
                raise ReservationCalibrationError(f"{arm_name.upper()} outcome omits its reconstructed scoring weights")
            expected_weights = _canonical_scoring_weights(
                action=str(row.get("action") or ""), arm_name=arm_name, arm=event_arm,
                player_points={int(key): value for key, value in player_points[event].items()},
                player_minutes={int(key): value for key, value in player_minutes[event].items()},
            )
            normalized_weights: dict[str, float] = {}
            for raw_player_id, raw_weight in result_weights.items():
                try:
                    player_id = str(int(raw_player_id))
                except (TypeError, ValueError) as failure:
                    raise ReservationCalibrationError(f"{arm_name.upper()} scorer has an invalid player id") from failure
                if player_id in normalized_weights:
                    raise ReservationCalibrationError(f"{arm_name.upper()} scorer has a duplicate player id")
                normalized_weights[player_id] = _number(raw_weight, name=f"{arm_name} scoring weight")
            if normalized_weights != expected_weights:
                raise ReservationCalibrationError(
                    f"{arm_name.upper()} scoring weights do not reproduce from the canonical {row.get('action')} scorer"
                )
            score = sum(
                player_points[event][player_id] * weight for player_id, weight in expected_weights.items()
            )
            if _number(declared.get("observed_points"), name=f"{arm_name} event observed points") != score:
                raise ReservationCalibrationError(f"{arm_name.upper()} score does not reproduce from official captures")
            weighted_total += float(scheduled.get("weight", 1.0)) * score
            rebuilt_event_results.append({
                "event": event, "weight": float(scheduled.get("weight", 1.0)),
                "proposed_squad_ids": [int(value) for value in scheduled["proposed_squad_ids"]],
                "lineup": dict(scheduled["lineup"]),
                "player_positions": dict(scheduled["player_positions"]),
                "scoring_weights": expected_weights,
                "observed_points": score,
            })
        if result.get("events") is not None and result.get("events") != rebuilt_event_results:
            # Numeric dictionaries are canonical JSON-compatible objects, so a
            # full equality check catches caller-supplied event-score edits.
            raise ReservationCalibrationError(f"{arm_name.upper()} event results do not reproduce")
        if _number(result.get("observed_points"), name=f"{arm_name} observed points") != weighted_total:
            raise ReservationCalibrationError(f"{arm_name.upper()} weighted score does not reproduce")
        scores[arm_name] = weighted_total
    realized_value = scores["play"] - scores["save"]
    if _number(outcome_record.get("observed_points"), name="paired observed points") != realized_value:
        raise ReservationCalibrationError("paired observed reservation value does not reproduce from PLAY/SAVE scores")
    return realized_value


def verify_outcome_record_capture_sources(
    conn: sqlite3.Connection,
    outcome_record: Mapping[str, Any],
    *,
    source_position_resolver: Callable[[Mapping[str, Any]], Mapping[str, str]] | None = None,
) -> None:
    """Resolve retained outcome captures against the append-only official ledger.

    A caller-supplied digest or ``source`` string is not authoritative. The
    production reader calls this function before returning an outcome artifact
    to the calibration validator; every cited capture must exist in the local
    append-only ledger and reproduce its content-derived capture digest.
    """

    if outcome_record.get("schema") != CAUSAL_OUTCOME_RECORD_SCHEMA:
        raise ReservationCalibrationError("outcome record has an unsupported schema")
    source = outcome_record.get("source")
    captures = source.get("captures") if isinstance(source, Mapping) else None
    if (
        not isinstance(source, Mapping)
        or source.get("schema") != OUTCOME_CAPTURE_SET_SCHEMA
        or not isinstance(captures, list)
        or not captures
    ):
        raise ReservationCalibrationError("outcome record has no official capture set")

    for capture in captures:
        if not isinstance(capture, Mapping):
            raise ReservationCalibrationError("outcome capture reference is malformed")
        digest = str(capture.get("capture_digest") or "")
        if not _is_sha256(digest):
            raise ReservationCalibrationError("outcome capture reference has no SHA-256 digest")
        try:
            row = conn.execute(
                "SELECT * FROM outcome_observation_captures WHERE capture_digest=?",
                (digest,),
            ).fetchone()
        except sqlite3.Error as failure:
            raise ReservationCalibrationError(
                f"official outcome capture ledger is unavailable: {failure}"
            ) from failure
        if row is None:
            raise ReservationCalibrationError("outcome capture digest is not retained in the official ledger")
        stored = dict(row)
        try:
            payload = json.loads(stored.get("payload_json") or "{}")
        except (TypeError, ValueError) as failure:
            raise ReservationCalibrationError("retained outcome capture payload is invalid JSON") from failure
        if not isinstance(payload, Mapping):
            raise ReservationCalibrationError("retained outcome capture payload is not an object")
        computed = ol.capture_digest_for(
            grain=str(stored["grain"]),
            event=int(stored["event"]),
            player_id=int(stored["player_id"]),
            fixture_id=None if stored.get("fixture_id") is None else int(stored["fixture_id"]),
            captured_at=str(stored["captured_at"]),
            observation_state=str(stored["observation_state"]),
            source_name=str(stored["source_name"]),
            source_identity=stored.get("source_identity"),
            source_payload_sha256=stored.get("source_payload_sha256"),
            archive_capture_id=stored.get("archive_capture_id"),
            payload=payload,
        )
        if computed != digest or str(stored.get("observation_state")) != ol.OBSERVATION_FINAL:
            raise ReservationCalibrationError(
                "outcome capture digest or official-final state does not verify"
            )
        if (
            str(stored.get("source_name") or "") != "player_gameweeks_final"
            or str(stored.get("grain") or "") != ol.GRAIN_PLAYER_EVENT
            or stored.get("fixture_id") is not None
            or str(stored.get("source_identity") or "") != f"player_gameweeks:{int(stored['event'])}"
        ):
            raise ReservationCalibrationError("outcome capture is not from the official player-gameweek source")
        expected = {
            "capture_digest": digest,
            "grain": str(stored["grain"]),
            "event": int(stored["event"]),
            "player_id": int(stored["player_id"]),
            "fixture_id": None if stored.get("fixture_id") is None else int(stored["fixture_id"]),
            "observation_state": str(stored["observation_state"]),
            "official_final_at": stored.get("official_final_at"),
            "captured_at": str(stored["captured_at"]),
            "source_name": str(stored["source_name"]),
            "source_identity": stored.get("source_identity"),
            "total_points": payload.get("total_points"),
            "minutes": payload.get("minutes"),
        }
        if any(capture.get(name) != value for name, value in expected.items()):
            raise ReservationCalibrationError(
                "outcome capture manifest differs from the retained official capture"
            )

    paired_results = outcome_record.get("paired_results")
    if not isinstance(paired_results, Mapping):
        raise ReservationCalibrationError("outcome record has no paired PLAY/SAVE results")
    play_result, save_result = paired_results.get("play"), paired_results.get("save")
    if not isinstance(play_result, Mapping) or not isinstance(save_result, Mapping):
        raise ReservationCalibrationError("outcome record has malformed paired PLAY/SAVE results")
    play_positions = play_result.get("player_positions")
    save_positions = save_result.get("player_positions")
    if not isinstance(play_positions, Mapping) or not isinstance(save_positions, Mapping):
        raise ReservationCalibrationError("paired outcome scoring omits an arm's source position map")
    action = str(outcome_record.get("action") or "")
    if action in {cd.CHIP_ACTION_BB, cd.CHIP_ACTION_TC} and dict(play_positions) != dict(save_positions):
        raise ReservationCalibrationError("BB/TC paired scoring does not use one retained source position map")
    if source_position_resolver is not None:
        try:
            expected_positions = source_position_resolver(outcome_record)
        except Exception as failure:
            raise ReservationCalibrationError(
                f"source generation positions could not be verified: {failure}"
            ) from failure
        if not isinstance(expected_positions, Mapping):
            raise ReservationCalibrationError("source generation position resolver returned no mapping")
        try:
            normalized_expected = {str(int(key)): str(value) for key, value in expected_positions.items()}
            normalized_source = {str(int(key)): str(value) for key, value in expected_positions.items()}
            normalized_play = {str(int(key)): str(value) for key, value in play_positions.items()}
            normalized_save = {str(int(key)): str(value) for key, value in save_positions.items()}
        except (TypeError, ValueError) as failure:
            raise ReservationCalibrationError("paired outcome contains an invalid source position map") from failure
        for arm_name, result, normalized in (
            ("PLAY", play_result, normalized_play), ("SAVE", save_result, normalized_save),
        ):
            try:
                policy_ids = {str(int(value)) for value in result["proposed_squad_ids"]}
            except (KeyError, TypeError, ValueError) as failure:
                raise ReservationCalibrationError(
                    f"{arm_name} paired outcome omits its proposed squad ids"
                ) from failure
            if set(normalized) != policy_ids or normalized != {
                player_id: position for player_id, position in normalized_source.items()
                if player_id in policy_ids
            }:
                raise ReservationCalibrationError(
                    f"{arm_name} paired outcome source positions differ from the pinned generation snapshot"
                )
            schedule_rows = result.get("events") or [{
                "proposed_squad_ids": result.get("proposed_squad_ids"),
                "player_positions": result.get("player_positions"),
            }]
            if not isinstance(schedule_rows, list) or not schedule_rows:
                raise ReservationCalibrationError(f"{arm_name} outcome has no verifiable event positions")
            for schedule_row in schedule_rows:
                try:
                    event_ids = {str(int(value)) for value in schedule_row["proposed_squad_ids"]}
                    event_positions = {str(int(key)): str(value)
                                       for key, value in schedule_row["player_positions"].items()}
                except (KeyError, TypeError, ValueError, AttributeError) as failure:
                    raise ReservationCalibrationError(
                        f"{arm_name} event schedule has malformed position evidence"
                    ) from failure
                if set(event_positions) != event_ids or event_positions != {
                    player_id: position for player_id, position in normalized_source.items()
                    if player_id in event_ids
                }:
                    raise ReservationCalibrationError(
                        f"{arm_name} event schedule positions differ from the pinned generation snapshot"
                    )


def _retain_content_addressed_json(root: Path, filename: str, payload: Mapping[str, Any]) -> str:
    """Retain one canonical JSON artifact without replacing an existing name."""

    root.mkdir(parents=True, exist_ok=True)
    target = root / filename
    encoded = _canonical_bytes(payload)
    if target.exists():
        if target.read_bytes() != encoded:
            raise ReservationCalibrationError(f"content-addressed artifact path already has different bytes: {target}")
        return str(target)
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(prefix=target.name + ".tmp-", dir=str(root))
        temporary = Path(name)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            if target.read_bytes() != encoded:
                raise ReservationCalibrationError(
                    f"content-addressed artifact path already has different bytes: {target}"
                )
        except OSError as failure:
            raise ReservationCalibrationError(
                f"atomic no-replace artifact publication failed: {failure}"
            ) from failure
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
    return str(target)


def retain_causal_origin_observation(
    *,
    observation_id: str,
    forecast_artifact: Mapping[str, Any],
    evidence_root: str | Path,
    evidence_verifier: Callable[[str], Mapping[str, Any]],
) -> dict[str, Any]:
    """Retain the selected future event's evaluated PLAY/SAVE origin evidence.

    This freezes the action-specific paired policies chosen by the point-in-time
    forecast. It does not create realized labels; those are added later from
    official final outcome captures.
    """

    if not str(observation_id or "").strip() or evidence_verifier is None:
        raise ReservationCalibrationError("causal origin needs an observation id and retained-artifact verifier")
    forecast = json.loads(_canonical_bytes(forecast_artifact).decode("utf-8"))
    if not isinstance(forecast, dict):
        raise ReservationCalibrationError("origin reservation forecast is not an object")
    action = str(forecast.get("action") or "")
    planning_event = int(forecast.get("planning_event") or -1)
    origin = _utc(forecast.get("origin_cutoff"), name="origin_cutoff")
    source_identity = forecast.get("source_identity")
    if not isinstance(source_identity, Mapping):
        raise ReservationCalibrationError("origin reservation forecast omits source identity")
    try:
        crf.verify_reservation_forecast(
            forecast,
            expected={
                "action": action,
                "planning_event": planning_event,
                "origin_cutoff": origin.isoformat().replace("+00:00", "Z"),
            },
            evidence_verifier=evidence_verifier,
        )
    except Exception as failure:
        raise ReservationCalibrationError(f"origin reservation forecast did not verify: {failure}") from failure
    if (
        not forecast.get("coverage_complete")
        or forecast.get("raw_value") is None
        or forecast.get("selected_event") is None
    ):
        raise ReservationCalibrationError("origin reservation forecast is incomplete or has no selected event")

    root = Path(evidence_root)
    local_records: dict[str, Mapping[str, Any]] = {}
    manifest = forecast.get("opportunities")
    if not isinstance(manifest, list) or not manifest:
        raise ReservationCalibrationError("origin reservation forecast has no retained event records")
    for item in manifest:
        payload = item.get("artifact_payload") if isinstance(item, Mapping) else None
        if not isinstance(payload, Mapping):
            raise ReservationCalibrationError("origin forecast event record is malformed")
        receipt = crf.retain_event_opportunity_record(payload, root)
        reference = Path(receipt["path"]).name
        local_records[reference] = dict(payload)
        item["artifact_ref"] = reference
    body = dict(forecast)
    body.pop("artifact_sha256", None)
    forecast["artifact_sha256"] = crf.canonical_sha256(body)
    try:
        crf.verify_reservation_forecast(
            forecast,
            expected={},
            evidence_verifier=local_records.__getitem__,
        )
    except Exception as failure:
        raise ReservationCalibrationError(f"self-contained origin forecast did not verify: {failure}") from failure
    forecast_receipt = crf.retain_reservation_forecast(forecast, root)
    forecast_reference = Path(forecast_receipt["path"]).name

    selected_event = int(forecast["selected_event"])
    selected_row = next((
        item["artifact_payload"] for item in forecast["opportunities"]
        if int(item["artifact_payload"]["event"]) == selected_event
    ), None)
    if not isinstance(selected_row, Mapping):
        raise ReservationCalibrationError("selected future event is absent from the forecast manifest")
    outcome_arms = selected_row.get("outcome_arms")
    if not isinstance(outcome_arms, Mapping) or set(outcome_arms) != {"play", "save"}:
        raise ReservationCalibrationError("selected event has no retained evaluator PLAY/SAVE policies")
    play_payload, save_payload = outcome_arms.get("play"), outcome_arms.get("save")
    if not isinstance(play_payload, Mapping) or not isinstance(save_payload, Mapping):
        raise ReservationCalibrationError("selected event PLAY/SAVE policies are malformed")
    _verify_action_pair_policies(action, play_payload, save_payload, planning_event=planning_event)
    _verify_forecast_matches_save_state(forecast, save_payload)
    world_identity = str(selected_row.get("world_identity") or "")
    if not world_identity or str(play_payload.get("world_identity")) != world_identity:
        raise ReservationCalibrationError("selected future arms are not bound to the forecast's certified world")

    source = {
        **dict(source_identity),
        "cutoff": origin.isoformat().replace("+00:00", "Z"),
        "world_identity": world_identity,
    }
    pair: dict[str, dict[str, Any]] = {}
    for role, payload in (("play", play_payload), ("save", save_payload)):
        artifact_payload = dict(payload)
        digest = _canonical_sha256(artifact_payload)
        reference = f"chip-causal-arm-{digest}.json"
        _retain_content_addressed_json(root, reference, artifact_payload)
        pair[role] = {
            "arm_id": str(artifact_payload.get("arm_id") or ""),
            "artifact_ref": reference,
            "artifact_sha256": digest,
            "artifact_payload": artifact_payload,
            **artifact_payload,
        }
    evidence: dict[str, Any] = {
        "schema": CAUSAL_EVIDENCE_SCHEMA,
        "policy": CAUSAL_EVIDENCE_POLICY,
        "observation_id": str(observation_id),
        "action": action,
        "planning_event": planning_event,
        "origin_cutoff": origin.isoformat().replace("+00:00", "Z"),
        "source": source,
        "forecast": {
            "artifact_ref": forecast_reference,
            "artifact_sha256": str(forecast["artifact_sha256"]),
            "retained_content_sha256": _canonical_sha256(forecast),
            "value": forecast.get("raw_value"),
            "made_at": forecast.get("made_at"),
            "input_as_of": forecast.get("input_as_of"),
            "forecast_mode": forecast.get("forecast_mode"),
        },
        "counterfactual_pair": pair,
        "label": None,
    }
    evidence_digest = _canonical_sha256(evidence)
    evidence_reference = f"chip-causal-evidence-{evidence_digest}.json"
    evidence_path = _retain_content_addressed_json(root, evidence_reference, evidence)
    return {
        "observation_id": str(observation_id),
        "causal_evidence_ref": evidence_reference,
        "causal_evidence_sha256": evidence_digest,
        "causal_evidence_path": evidence_path,
        "forecast_ref": forecast_reference,
        "forecast_sha256": str(forecast["artifact_sha256"]),
        "selected_event": selected_event,
        "causal_evidence": evidence,
    }


def finalize_causal_observation(
    conn: sqlite3.Connection,
    origin_evidence: Mapping[str, Any],
    *,
    realization_event: int,
    evidence_root: str | Path,
    evidence_verifier: Callable[[str], Mapping[str, Any]],
    source_position_resolver: Callable[[Mapping[str, Any]], Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    """Build and retain a matured label from official final player-event captures.

    The input contains only the point-in-time source, raw forecast and retained
    PLAY/SAVE policies. Realized weights are created here from the later official
    minutes and scores, never required in the origin artifacts. The returned
    calibration row points to immutable content-addressed evidence and outcome
    records. Incomplete raw-forecast coverage cannot be finalized for fitting.
    """

    if evidence_verifier is None:
        raise ReservationCalibrationError("a retained origin-artifact verifier is required")
    evidence = json.loads(_canonical_bytes(origin_evidence).decode("utf-8"))
    if not isinstance(evidence, dict) or evidence.get("schema") != CAUSAL_EVIDENCE_SCHEMA:
        raise ReservationCalibrationError("origin evidence has an unsupported causal-evidence schema")
    if evidence.get("policy") != CAUSAL_EVIDENCE_POLICY:
        raise ReservationCalibrationError("origin evidence does not use the declared causal policy")
    if evidence.get("label") is not None:
        raise ReservationCalibrationError("origin evidence already has an outcome label")
    action = str(evidence.get("action") or "")
    planning_event = int(evidence.get("planning_event") or -1)
    realization_event = int(realization_event)
    if action not in cd.PLAYABLE_CHIP_ACTIONS or realization_event <= planning_event:
        raise ReservationCalibrationError("causal outcome action or realization event is invalid")
    origin = _utc(evidence.get("origin_cutoff"), name="origin_cutoff")
    source = evidence.get("source")
    pair = evidence.get("counterfactual_pair")
    if not isinstance(source, Mapping) or not isinstance(pair, Mapping):
        raise ReservationCalibrationError("origin evidence omits its source identity or paired arms")
    play, save = pair.get("play"), pair.get("save")
    if not isinstance(play, Mapping) or not isinstance(save, Mapping):
        raise ReservationCalibrationError("origin evidence must retain both PLAY and SAVE arms")
    _verify_action_pair_policies(action, play, save, planning_event=planning_event)
    for arm_name, arm in (("PLAY", play), ("SAVE", save)):
        payload = arm.get("artifact_payload")
        if not isinstance(payload, Mapping) or _canonical_sha256(payload) != str(arm.get("artifact_sha256") or ""):
            raise ReservationCalibrationError(f"{arm_name} origin arm artifact digest does not verify")
        try:
            retained_arm = evidence_verifier(str(arm.get("artifact_ref") or ""))
        except Exception as failure:
            raise ReservationCalibrationError(f"retained {arm_name} origin arm is unavailable: {failure}") from failure
        if not isinstance(retained_arm, Mapping) or dict(retained_arm) != dict(payload):
            raise ReservationCalibrationError(f"retained {arm_name} origin arm differs from its manifest")
    forecast_meta = evidence.get("forecast")
    if not isinstance(forecast_meta, Mapping):
        raise ReservationCalibrationError("origin evidence has no retained raw reservation forecast")
    try:
        raw_forecast = evidence_verifier(str(forecast_meta.get("artifact_ref") or ""))
    except Exception as failure:
        raise ReservationCalibrationError(f"retained raw reservation forecast is unavailable: {failure}") from failure
    if not isinstance(raw_forecast, Mapping):
        raise ReservationCalibrationError("retained raw reservation forecast is not an object")
    expected_source = {
        "source_decision_id": source.get("source_decision_id"),
        "source_result_sha256": source.get("source_result_sha256"),
        "source_artifact_sha256": source.get("source_artifact_sha256"),
        "generation_id": source.get("generation_id"),
        "planning_event": planning_event,
        "origin_cutoff": origin.isoformat().replace("+00:00", "Z"),
        "data_snapshot_sha256": source.get("data_snapshot_sha256"),
        "predictive_code_snapshot_sha256": source.get("predictive_code_snapshot_sha256"),
        "certification_identity": source.get("certification_identity"),
    }
    try:
        crf.verify_reservation_forecast(
            raw_forecast,
            expected={
                "action": action,
                "planning_event": planning_event,
                "origin_cutoff": origin.isoformat().replace("+00:00", "Z"),
                "source_identity": expected_source,
            },
            evidence_verifier=evidence_verifier,
        )
    except Exception as failure:
        raise ReservationCalibrationError(f"raw reservation forecast is not valid at its origin: {failure}") from failure
    _verify_forecast_matches_save_state(raw_forecast, save)
    _verify_pair_matches_selected_forecast(action, raw_forecast, pair)
    expiry_event = raw_forecast.get("expiry_event")
    raw_value = raw_forecast.get("raw_value")
    if (
        not raw_forecast.get("coverage_complete")
        or expiry_event is None
        or raw_value is None
        or int(expiry_event) <= planning_event
    ):
        raise ReservationCalibrationError(
            "raw reservation forecast has unknown expiry or incomplete event coverage; it cannot train calibration"
        )
    if int(raw_forecast.get("selected_event") or -1) != realization_event:
        raise ReservationCalibrationError(
            "realized outcome event differs from the origin forecast's selected event"
        )
    weeks_to_expiry = int(expiry_event) - planning_event
    play_schedule = play.get("valuation_schedule")
    save_schedule = save.get("valuation_schedule")
    if isinstance(play_schedule, Mapping) or isinstance(save_schedule, Mapping):
        _verify_valuation_schedules(action, play, save, planning_event=planning_event)
        scheduled_events = [int(item["event"]) for item in play_schedule["events"]]
    else:
        scheduled_events = [realization_event]
    if scheduled_events[0] != realization_event:
        raise ReservationCalibrationError("selected forecast event is not the first outcome schedule event")

    arm_rows: dict[str, list[dict[str, Any]]] = {}
    required_players_by_event: dict[int, set[int]] = {event: set() for event in scheduled_events}
    for arm_name, arm in (("play", play), ("save", save)):
        schedule = arm.get("valuation_schedule")
        if isinstance(schedule, Mapping):
            rows = [dict(item) for item in schedule.get("events") or ()]
        else:
            rows = [{
                "event": realization_event, "weight": 1.0,
                "proposed_squad_ids": list(arm.get("proposed_squad_ids") or ()),
                "lineup": dict(arm.get("lineup") or {}),
                "player_positions": dict(arm.get("player_positions") or {}),
                "transfer_state": dict((arm.get("action_semantics") or {}).get("transfer_state") or {}),
            }]
        arm_rows[arm_name] = rows
        for row in rows:
            required_players_by_event[int(row["event"])].update(
                int(value) for value in row["proposed_squad_ids"]
            )

    captures: list[dict[str, Any]] = []
    player_points: dict[int, dict[int, float]] = {}
    player_minutes: dict[int, dict[int, float]] = {}
    capture_times: list[datetime] = []
    for event in scheduled_events:
        required_players = required_players_by_event[event]
        by_player: dict[int, list[Mapping[str, Any]]] = {}
        for capture in ol.observation_captures(conn, grain=ol.GRAIN_PLAYER_EVENT, event=event):
            if int(capture.get("player_id") or -1) in required_players:
                by_player.setdefault(int(capture["player_id"]), []).append(capture)
        player_points[event] = {}
        player_minutes[event] = {}
        for player_id in sorted(required_players):
            official_final = [
                row for row in by_player.get(player_id, ())
                if row.get("grain") == ol.GRAIN_PLAYER_EVENT
                and row.get("fixture_id") is None
                and row.get("observation_state") == ol.OBSERVATION_FINAL
                and row.get("source_name") == "player_gameweeks_final"
                and row.get("source_identity") == f"player_gameweeks:{event}"
            ]
            selected = ol.select_capture(official_final)
            if selected is None:
                raise ReservationCalibrationError(
                    f"official final player-gameweek capture is missing for player {player_id} in GW{event}"
                )
            payload = selected.get("payload")
            if not isinstance(payload, Mapping):
                raise ReservationCalibrationError(f"official player capture payload is invalid for player {player_id}")
            points = _number(payload.get("total_points"), name=f"player {player_id} total_points")
            minutes = _number(payload.get("minutes"), name=f"player {player_id} minutes")
            if minutes < 0.0:
                raise ReservationCalibrationError(f"official minutes are negative for player {player_id}")
            official_final_at = _utc(selected.get("official_final_at"), name="official_final_at")
            captured_at = _utc(selected.get("captured_at"), name="captured_at")
            if captured_at < official_final_at:
                raise ReservationCalibrationError("official outcome capture predates official finality")
            player_points[event][player_id] = points
            player_minutes[event][player_id] = minutes
            capture_times.append(captured_at)
            captures.append({
                "capture_digest": str(selected.get("capture_digest") or ""),
                "grain": ol.GRAIN_PLAYER_EVENT,
                "event": event,
                "player_id": player_id,
                "fixture_id": None,
                "observation_state": ol.OBSERVATION_FINAL,
                "official_final_at": official_final_at.isoformat().replace("+00:00", "Z"),
                "captured_at": captured_at.isoformat().replace("+00:00", "Z"),
                "source_name": str(selected.get("source_name") or ""),
                "source_identity": str(selected.get("source_identity") or ""),
                "total_points": points,
                "minutes": minutes,
            })
    available_at = max(capture_times).isoformat().replace("+00:00", "Z")
    paired_results: dict[str, dict[str, Any]] = {}
    for arm_name, arm in (("play", play), ("save", save)):
        event_results: list[dict[str, Any]] = []
        weighted_observed = 0.0
        for scheduled in arm_rows[arm_name]:
            event = int(scheduled["event"])
            event_arm = {
                **dict(arm),
                "proposed_squad_ids": scheduled["proposed_squad_ids"],
                "lineup": scheduled["lineup"],
                "player_positions": scheduled["player_positions"],
            }
            weights = _canonical_scoring_weights(
                action=action, arm_name=arm_name, arm=event_arm,
                player_points=player_points[event], player_minutes=player_minutes[event],
            )
            observed = sum(player_points[event][int(player_id)] * weight
                           for player_id, weight in weights.items())
            event_weight = float(scheduled.get("weight", 1.0))
            weighted_observed += event_weight * observed
            event_results.append({
                "event": event, "weight": event_weight,
                "proposed_squad_ids": [int(value) for value in scheduled["proposed_squad_ids"]],
                "lineup": dict(scheduled["lineup"]),
                "player_positions": dict(scheduled["player_positions"]),
                "scoring_weights": weights,
                "observed_points": observed,
            })
        paired_results[arm_name] = {
            "proposed_squad_ids": [int(value) for value in arm["proposed_squad_ids"]],
            "lineup": dict(arm["lineup"]),
            "player_positions": dict(arm["player_positions"]),
            "events": event_results,
            "observed_points": weighted_observed,
        }
        if len(event_results) == 1:
            paired_results[arm_name]["scoring_weights"] = dict(event_results[0]["scoring_weights"])
    realized = paired_results["play"]["observed_points"] - paired_results["save"]["observed_points"]
    outcome_record: dict[str, Any] = {
        "schema": CAUSAL_OUTCOME_RECORD_SCHEMA,
        "record_id": str(uuid.uuid4()),
        "observation_id": str(evidence.get("observation_id") or ""),
        "action": action,
        "planning_event": planning_event,
        "origin_cutoff": origin.isoformat().replace("+00:00", "Z"),
        "scenario_identity": str(play.get("scenario_identity") or ""),
        "world_identity": str(play.get("world_identity") or ""),
        "play_arm_id": str(play.get("arm_id") or ""),
        "play_artifact_sha256": str(play.get("artifact_sha256") or ""),
        "save_arm_id": str(save.get("arm_id") or ""),
        "save_artifact_sha256": str(save.get("artifact_sha256") or ""),
        "source_decision_id": str(source.get("source_decision_id") or ""),
        "source_result_sha256": str(source.get("source_result_sha256") or ""),
        "source_artifact_sha256": str(source.get("source_artifact_sha256") or ""),
        "generation_id": str(source.get("generation_id") or ""),
        "cutoff": origin.isoformat().replace("+00:00", "Z"),
        "data_snapshot_sha256": str(source.get("data_snapshot_sha256") or ""),
        "predictive_code_snapshot_sha256": str(source.get("predictive_code_snapshot_sha256") or ""),
        "certification_identity": str(source.get("certification_identity") or ""),
        "label_definition_version": OUTCOME_LABEL_DEFINITION,
        "scoring_rule_version": OUTCOME_SCORING_RULE_VERSION,
        "realization_event": realization_event,
        "realization_events": scheduled_events,
        "available_at": available_at,
        "source": {
            "schema": OUTCOME_CAPTURE_SET_SCHEMA,
            "available_at": available_at,
            "captures": captures,
        },
        "paired_results": paired_results,
        "observed_points": realized,
    }
    verify_outcome_record_capture_sources(conn, outcome_record, source_position_resolver=source_position_resolver)
    outcome_digest = _canonical_sha256(outcome_record)
    outcome_reference = f"chip-outcome-{outcome_digest}.json"
    final_evidence = dict(evidence)
    final_evidence["label_available_at"] = available_at
    final_evidence["label"] = {
        "kind": "OBSERVED_FUTURE_OUTCOME",
        "outcome_record_ref": outcome_reference,
        "outcome_record": outcome_record,
        "outcome_record_sha256": outcome_digest,
        "realized_reservation_value": realized,
        "available_at": available_at,
    }
    evidence_digest = _canonical_sha256(final_evidence)
    evidence_reference = f"chip-causal-evidence-{evidence_digest}.json"
    calibration_row = {
        "observation_id": str(evidence.get("observation_id") or ""),
        "action": action,
        "planning_event": planning_event,
        "weeks_to_expiry": weeks_to_expiry,
        "origin_cutoff": origin.isoformat().replace("+00:00", "Z"),
        "forecast_made_at": str(raw_forecast.get("made_at") or ""),
        "forecast_input_as_of": str(raw_forecast.get("input_as_of") or ""),
        "forecast_mode": str(raw_forecast.get("forecast_mode") or ""),
        "forecast_value": _number(raw_value, name="raw reservation forecast"),
        "label_available_at": available_at,
        "realized_value": realized,
        "causal_evidence_ref": evidence_reference,
        "causal_evidence_sha256": evidence_digest,
    }
    local_artifacts = {
        evidence_reference: final_evidence,
        outcome_reference: outcome_record,
    }

    def resolve(reference: str) -> Mapping[str, Any]:
        if str(reference) in local_artifacts:
            return local_artifacts[str(reference)]
        return evidence_verifier(str(reference))

    validate_observation(calibration_row, evidence_verifier=resolve)
    root = Path(evidence_root)
    outcome_path = _retain_content_addressed_json(root, outcome_reference, outcome_record)
    evidence_path = _retain_content_addressed_json(root, evidence_reference, final_evidence)
    return {
        "calibration_row": calibration_row,
        "outcome_record": outcome_record,
        "causal_evidence": final_evidence,
        "outcome_path": outcome_path,
        "causal_evidence_path": evidence_path,
        "outcome_sha256": outcome_digest,
        "causal_evidence_sha256": evidence_digest,
    }


def expiry_bucket(weeks_to_expiry: int | None) -> str:
    if weeks_to_expiry is None:
        return "UNKNOWN"
    weeks = max(0, int(weeks_to_expiry))
    if weeks == 0:
        return "0"
    if weeks <= 4:
        return "1-4"
    return "5+"


@dataclass(frozen=True)
class VerifiedOpportunity:
    observation_id: str
    action: str
    planning_event: int
    weeks_to_expiry: int
    origin_cutoff: datetime
    forecast_input_as_of: datetime
    forecast_made_at: datetime
    forecast_mode: str
    label_available_at: datetime
    forecast_value: float
    realized_value: float
    evidence_ref: str
    evidence_sha256: str
    source_decision_id: str | None = None
    evaluator_version: str | None = None
    evaluator_value_definition: str | None = None
    evaluator_interval_value_definition: str | None = None
    evaluator_forecast_value: float | None = None
    evaluator_interval_low: float | None = None
    evaluator_interval_high: float | None = None

    @property
    def bucket(self) -> str:
        return expiry_bucket(self.weeks_to_expiry)


def _verify_pair(
    evidence: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    retained_outcome_record: Mapping[str, Any],
    evidence_verifier: Callable[[str], Mapping[str, Any]],
) -> None:
    if evidence.get("schema") != CAUSAL_EVIDENCE_SCHEMA:
        raise ReservationCalibrationError("causal evidence has an unsupported schema")
    if evidence.get("policy") != CAUSAL_EVIDENCE_POLICY:
        raise ReservationCalibrationError("causal evidence does not use the declared paired policy")
    for name in ("observation_id", "action", "planning_event", "origin_cutoff", "label_available_at"):
        if str(evidence.get(name)) != str(row.get(name)):
            raise ReservationCalibrationError(f"causal evidence disagrees on {name}")
    source = evidence.get("source")
    pair = evidence.get("counterfactual_pair")
    label = evidence.get("label")
    if not isinstance(source, Mapping) or not isinstance(pair, Mapping) or not isinstance(label, Mapping):
        raise ReservationCalibrationError("causal evidence lacks source, paired arms or outcome label")
    required_source = (
        "source_decision_id", "source_result_sha256", "source_artifact_sha256",
        "generation_id", "planning_event", "cutoff", "data_snapshot_sha256",
        "predictive_code_snapshot_sha256", "certification_identity", "world_identity",
    )
    if any(not str(source.get(name) or "").strip() for name in required_source):
        raise ReservationCalibrationError("causal evidence has incomplete source decision/generation identity")
    for name in ("source_result_sha256", "source_artifact_sha256", "data_snapshot_sha256",
                 "predictive_code_snapshot_sha256", "certification_identity", "world_identity"):
        if not _is_sha256(source.get(name)):
            raise ReservationCalibrationError(f"causal evidence source {name} is not a SHA-256 identity")
    if (
        str(source.get("cutoff")) != str(row.get("origin_cutoff"))
        or int(source.get("planning_event") or -1) != int(row.get("planning_event") or -2)
    ):
        raise ReservationCalibrationError("causal evidence source is not from the forecast origin")
    play, save = pair.get("play"), pair.get("save")
    if not isinstance(play, Mapping) or not isinstance(save, Mapping):
        raise ReservationCalibrationError("causal evidence must retain both PLAY and SAVE arms")
    action = str(row.get("action") or "")
    paired_fields = ("scenario_identity", "world_identity")
    for name in paired_fields:
        if play.get(name) != save.get(name) or play.get(name) in (None, "", [], {}):
            raise ReservationCalibrationError(f"PLAY and SAVE do not share the same {name}")
    _verify_action_pair_policies(
        action, play, save, planning_event=int(row.get("planning_event") or -1),
    )
    source_arm_fields = (
        "source_decision_id", "source_result_sha256", "source_artifact_sha256",
        "generation_id", "cutoff", "data_snapshot_sha256", "predictive_code_snapshot_sha256",
        "certification_identity",
    )
    for arm_name, arm in (("PLAY", play), ("SAVE", save)):
        for name in source_arm_fields:
            if str(arm.get(name) or "") != str(source.get(name) or ""):
                raise ReservationCalibrationError(f"{arm_name} arm is not bound to source {name}")
    if not str(source.get("world_identity") or "").strip() or str(
        play.get("world_identity")
    ) != str(source.get("world_identity")):
        raise ReservationCalibrationError("paired arms are not bound to the source world identity")
    for arm_name, arm in (("PLAY", play), ("SAVE", save)):
        if any(not str(arm.get(name) or "").strip()
               for name in ("arm_id", "artifact_ref", "artifact_sha256")):
            raise ReservationCalibrationError(f"{arm_name} arm is missing retained artifact identity")
        artifact_payload = arm.get("artifact_payload")
        if not isinstance(artifact_payload, Mapping) or _canonical_sha256(artifact_payload) != str(
            arm.get("artifact_sha256")
        ):
            raise ReservationCalibrationError(f"{arm_name} arm artifact content digest does not verify")
        try:
            retained_arm = evidence_verifier(str(arm.get("artifact_ref") or ""))
        except Exception as failure:
            raise ReservationCalibrationError(
                f"retained {arm_name} arm artifact did not verify: {failure}"
            ) from failure
        if not isinstance(retained_arm, Mapping) or dict(retained_arm) != dict(artifact_payload):
            raise ReservationCalibrationError(
                f"retained {arm_name} arm artifact differs from the causal evidence manifest"
            )
        for name in (
            *paired_fields, *source_arm_fields, "arm_id", "proposed_squad_ids", "lineup",
            "player_positions", "counterfactual_role", "action_semantics", "scoring_rule_version",
        ):
            if artifact_payload.get(name) != arm.get(name):
                raise ReservationCalibrationError(f"{arm_name} arm artifact disagrees on {name}")
        if (
            arm.get("scoring_rule_version") != OUTCOME_SCORING_RULE_VERSION
            or artifact_payload.get("scoring_rule_version") != OUTCOME_SCORING_RULE_VERSION
        ):
            raise ReservationCalibrationError(f"{arm_name} arm does not bind the canonical outcome scorer")
        _number(artifact_payload.get("paired_value"), name=f"{arm_name} paired value")
    if str(play.get("arm_id")) == str(save.get("arm_id")):
        raise ReservationCalibrationError("PLAY and SAVE must be separately retained counterfactual arms")
    if label.get("kind") != "OBSERVED_FUTURE_OUTCOME" or not str(label.get("outcome_record_ref") or "").strip():
        raise ReservationCalibrationError("causal label is not tied to a retained future outcome")
    outcome_record = label.get("outcome_record")
    outcome_digest = str(label.get("outcome_record_sha256") or "")
    if not isinstance(outcome_record, Mapping) or len(outcome_digest) != 64:
        raise ReservationCalibrationError("causal label has no content-addressed outcome record")
    if _canonical_sha256(outcome_record) != outcome_digest:
        raise ReservationCalibrationError("causal label outcome-record digest does not verify")
    if (
        _canonical_sha256(retained_outcome_record) != outcome_digest
        or dict(retained_outcome_record) != dict(outcome_record)
    ):
        raise ReservationCalibrationError(
            "causal outcome differs from the separately retained outcome record"
        )
    realized_value = _verify_outcome_record(
        retained_outcome_record,
        evidence=evidence,
        row=row,
        label=label,
        play=play,
        save=save,
    )
    if (
        realized_value != _number(label.get("realized_reservation_value"), name="realized reservation value")
        or realized_value != _number(row.get("realized_value"), name="realized reservation value")
    ):
        raise ReservationCalibrationError(
            "causal evidence label is not reproduced by the retained paired outcome"
        )


def validate_observation(
    row: Mapping[str, Any], *, evidence_verifier: Callable[[str], Mapping[str, Any]] | None,
) -> VerifiedOpportunity:
    """Load and verify one point-in-time forecast/outcome pair.

    The verifier must read the referenced retained evidence and independently
    validate its byte/content digest. Calibration never trusts caller-provided
    ``verified`` flags or free-form calibration statuses.
    """

    if evidence_verifier is None:
        raise ReservationCalibrationError("a retained causal-evidence verifier is required")
    observation_id = str(row.get("observation_id") or "")
    action = str(row.get("action") or "")
    if not observation_id or action not in cd.PLAYABLE_CHIP_ACTIONS:
        raise ReservationCalibrationError("observation id or chip action is invalid")
    origin = _utc(row.get("origin_cutoff"), name="origin_cutoff")
    label_time = _utc(row.get("label_available_at"), name="label_available_at")
    forecast_time = _utc(row.get("forecast_made_at"), name="forecast_made_at")
    input_as_of = _utc(row.get("forecast_input_as_of") or row.get("origin_cutoff"), name="forecast_input_as_of")
    forecast_mode = str(row.get("forecast_mode") or crf.FORECAST_MODE_PROSPECTIVE)
    if input_as_of > origin:
        raise ReservationCalibrationError("forecast inputs are later than the origin cutoff")
    if forecast_time < input_as_of:
        raise ReservationCalibrationError("forecast issuance precedes its input-as-of time")
    if forecast_mode != crf.FORECAST_MODE_PROSPECTIVE:
        raise ReservationCalibrationError("historical replay forecasts cannot enter causal calibration")
    if label_time <= forecast_time:
        raise ReservationCalibrationError("future outcome label was available before forecast issuance")
    reference = str(row.get("causal_evidence_ref") or "")
    expected_digest = str(row.get("causal_evidence_sha256") or "")
    if not reference or not _is_sha256(expected_digest):
        raise ReservationCalibrationError("retained causal-evidence reference/digest is missing")
    try:
        evidence = evidence_verifier(reference)
    except Exception as failure:
        raise ReservationCalibrationError(f"retained causal evidence did not verify: {failure}") from failure
    if not isinstance(evidence, Mapping) or _canonical_sha256(evidence) != expected_digest:
        raise ReservationCalibrationError("retained causal-evidence content digest does not match")
    label = evidence.get("label")
    outcome_reference = str(label.get("outcome_record_ref") or "") if isinstance(label, Mapping) else ""
    if not outcome_reference:
        raise ReservationCalibrationError("causal evidence has no retained outcome-record reference")
    try:
        retained_outcome_record = evidence_verifier(outcome_reference)
    except Exception as failure:
        raise ReservationCalibrationError(
            f"retained outcome record did not verify: {failure}"
        ) from failure
    if not isinstance(retained_outcome_record, Mapping):
        raise ReservationCalibrationError("retained outcome record is not a JSON object")
    weeks = row.get("weeks_to_expiry")
    if weeks is None or int(weeks) < 0:
        raise ReservationCalibrationError("weeks_to_expiry must be an explicit non-negative integer")
    forecast = evidence.get("forecast")
    if not isinstance(forecast, Mapping):
        raise ReservationCalibrationError("causal evidence has no retained raw reservation forecast")
    forecast_reference = str(forecast.get("artifact_ref") or "")
    if not forecast_reference:
        raise ReservationCalibrationError("causal evidence has no raw forecast artifact reference")
    try:
        forecast_artifact = evidence_verifier(forecast_reference)
    except Exception as failure:
        raise ReservationCalibrationError(f"retained raw reservation forecast did not verify: {failure}") from failure
    if not isinstance(forecast_artifact, Mapping):
        raise ReservationCalibrationError("retained raw reservation forecast is not an object")
    if (
        not _is_sha256(forecast.get("retained_content_sha256"))
        or _canonical_sha256(forecast_artifact) != str(forecast.get("retained_content_sha256"))
        or str(forecast_artifact.get("artifact_sha256") or "")
        != str(forecast.get("artifact_sha256") or "")
    ):
        raise ReservationCalibrationError("retained raw reservation forecast digest does not verify")
    source = evidence.get("source")
    expected_forecast_source = {
        "source_decision_id": source.get("source_decision_id"),
        "source_result_sha256": source.get("source_result_sha256"),
        "source_artifact_sha256": source.get("source_artifact_sha256"),
        "generation_id": source.get("generation_id"),
        "planning_event": int(row.get("planning_event")),
        "origin_cutoff": str(row.get("origin_cutoff")),
        "data_snapshot_sha256": source.get("data_snapshot_sha256"),
        "predictive_code_snapshot_sha256": source.get("predictive_code_snapshot_sha256"),
        "certification_identity": source.get("certification_identity"),
    }
    try:
        crf.verify_reservation_forecast(
            forecast_artifact,
            expected={
                "action": action,
                "planning_event": int(row.get("planning_event")),
                "origin_cutoff": str(row.get("origin_cutoff")),
                "made_at": str(row.get("forecast_made_at")),
                "input_as_of": input_as_of.isoformat().replace("+00:00", "Z"),
                "forecast_mode": forecast_mode,
                "expiry_event": int(row.get("planning_event")) + int(weeks),
                "source_identity": expected_forecast_source,
            },
            evidence_verifier=evidence_verifier,
        )
    except Exception as failure:
        raise ReservationCalibrationError(f"retained raw reservation forecast is invalid: {failure}") from failure
    pair = evidence.get("counterfactual_pair")
    save_arm = pair.get("save") if isinstance(pair, Mapping) else None
    if not isinstance(save_arm, Mapping):
        raise ReservationCalibrationError("causal evidence omits the retained SAVE arm")
    if not isinstance(pair, Mapping):
        raise ReservationCalibrationError("causal evidence omits its retained PLAY/SAVE pair")
    _verify_forecast_matches_save_state(forecast_artifact, save_arm)
    _verify_pair_matches_selected_forecast(action, forecast_artifact, pair)
    _verify_pair(
        evidence, row, retained_outcome_record=retained_outcome_record,
        evidence_verifier=evidence_verifier,
    )
    outcome_record = label.get("outcome_record") if isinstance(label, Mapping) else None
    selected_event = forecast_artifact.get("selected_event")
    if (
        not isinstance(outcome_record, Mapping)
        or int(outcome_record.get("realization_event") or -1) != int(selected_event or -2)
    ):
        raise ReservationCalibrationError(
            "matured outcome event differs from the origin forecast's selected event"
        )
    if (
        not forecast_artifact.get("coverage_complete")
        or forecast_artifact.get("raw_value") is None
        or _number(forecast_artifact.get("raw_value"), name="raw forecast value")
        != _number(row.get("forecast_value"), name="forecast value")
        or _number(forecast.get("value"), name="evidence forecast value")
        != _number(row.get("forecast_value"), name="forecast value")
        or str(forecast.get("made_at")) != str(row.get("forecast_made_at"))
        or str(forecast_artifact.get("input_as_of")) != input_as_of.isoformat().replace("+00:00", "Z")
        or str(forecast_artifact.get("forecast_mode")) != forecast_mode
    ):
        raise ReservationCalibrationError(
            "raw reservation forecast is incomplete or differs from the supplied calibration row"
        )
    selected_event = int(forecast_artifact.get("selected_event") or -1)
    selected_opportunities = [
        item.get("artifact_payload")
        for item in forecast_artifact.get("opportunities") or ()
        if isinstance(item, Mapping)
        and isinstance(item.get("artifact_payload"), Mapping)
        and int(item["artifact_payload"].get("event") or -1) == selected_event
    ]
    if len(selected_opportunities) != 1:
        raise ReservationCalibrationError("forecast does not retain one selected evaluator opportunity")
    evaluator_identity = selected_opportunities[0].get("evaluator_identity")
    evaluator_version = None
    evaluator_value_definition = None
    evaluator_interval_value_definition = None
    evaluator_value = evaluator_low = evaluator_high = None
    if evaluator_identity is not None:
        if not isinstance(evaluator_identity, Mapping):
            raise ReservationCalibrationError("selected evaluator identity is malformed")
        evaluator_version = str(evaluator_identity.get("evaluator_version") or "").strip() or None
        evaluator_value_definition = (
            str(evaluator_identity.get("opportunity_value_definition") or "").strip() or None
        )
        evaluator_interval_value_definition = (
            str(evaluator_identity.get("uncertainty_value_definition") or "").strip() or None
        )
        if evaluator_version is not None:
            evaluator_value = _number(
                evaluator_identity.get("expected_incremental_points"),
                name="selected evaluator forecast value",
            )
            uncertainty = evaluator_identity.get("uncertainty")
            if not isinstance(uncertainty, Mapping):
                raise ReservationCalibrationError("selected evaluator forecast omits its uncertainty interval")
            evaluator_low = _number(
                uncertainty.get("paired_interval_low"), name="evaluator paired_interval_low",
            )
            evaluator_high = _number(
                uncertainty.get("paired_interval_high"), name="evaluator paired_interval_high",
            )
            if evaluator_low > evaluator_high:
                raise ReservationCalibrationError("evaluator uncertainty interval is reversed")
    return VerifiedOpportunity(
        observation_id=observation_id,
        action=action,
        planning_event=int(row.get("planning_event")),
        weeks_to_expiry=int(weeks),
        origin_cutoff=origin,
        forecast_input_as_of=input_as_of,
        forecast_made_at=forecast_time,
        forecast_mode=forecast_mode,
        label_available_at=label_time,
        forecast_value=_number(row.get("forecast_value"), name="forecast value"),
        realized_value=_number(row.get("realized_value"), name="realized value"),
        evidence_ref=reference,
        evidence_sha256=expected_digest,
        source_decision_id=str(source.get("source_decision_id") or "") or None,
        evaluator_version=evaluator_version,
        evaluator_value_definition=evaluator_value_definition,
        evaluator_interval_value_definition=evaluator_interval_value_definition,
        evaluator_forecast_value=evaluator_value,
        evaluator_interval_low=evaluator_low,
        evaluator_interval_high=evaluator_high,
    )


def _quantile(values: Sequence[float], q: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, int(round(float(q) * (len(ordered) - 1)))))
    return ordered[index]


def _metrics(rows: Sequence[tuple[float, float]]) -> dict[str, float]:
    residuals = [actual - prediction for prediction, actual in rows]
    raw_mae = sum(abs(actual - prediction) for prediction, actual in rows) / len(rows)
    bias = sum(residuals) / len(residuals)
    return {"mae": raw_mae, "bias": bias}


def _walk_forward_group(rows: Sequence[VerifiedOpportunity]) -> dict[str, Any]:
    ordered = sorted(rows, key=lambda row: (row.origin_cutoff, row.observation_id))
    predictions: list[tuple[float, float, float, float, float]] = []
    for test in ordered:
        training = [
            row for row in ordered
            if row.observation_id != test.observation_id
            and row.origin_cutoff < test.origin_cutoff
            and row.label_available_at <= test.origin_cutoff
        ]
        if len(training) < MIN_TRAINING_ORIGINS:
            continue
        # Locked intercept-only correction. It calibrates a known action/expiry
        # band without fitting a flexible model to a small seasonal sample.
        bias_correction = sum(row.realized_value - row.forecast_value for row in training) / len(training)
        train_residuals = [row.realized_value - (row.forecast_value + bias_correction) for row in training]
        low = _quantile(train_residuals, 0.05)
        high = _quantile(train_residuals, 0.95)
        prediction = test.forecast_value + bias_correction
        predictions.append((test.forecast_value, prediction, test.realized_value, low, high))

    if len(predictions) < MIN_VALIDATION_ORIGINS:
        return {
            "status": cd.CALIBRATION_UNCALIBRATED,
            "reason": DIAG_CALIBRATION_INSUFFICIENT,
            "training_origins": len(ordered),
            "validation_origins": len(predictions),
        }
    raw_pairs = [(raw, actual) for raw, _adjusted, actual, _low, _high in predictions]
    adj_pairs = [(adjusted, actual) for _raw, adjusted, actual, _low, _high in predictions]
    raw = _metrics(raw_pairs)
    adjusted = _metrics(adj_pairs)
    coverage = sum(
        1 for _raw, predicted, actual, low, high in predictions
        if predicted + low <= actual <= predicted + high
    ) / len(predictions)
    improvement = 0.0 if raw["mae"] == 0.0 else (raw["mae"] - adjusted["mae"]) / raw["mae"]
    accepted = (
        improvement >= MIN_MAE_IMPROVEMENT
        and abs(adjusted["bias"]) <= MAX_ABSOLUTE_BIAS
        and coverage >= MIN_90_INTERVAL_COVERAGE
    )
    all_residuals = [row.realized_value - row.forecast_value for row in ordered]
    final_correction = sum(all_residuals) / len(all_residuals)
    calibrated_residuals = [residual - final_correction for residual in all_residuals]
    return {
        "status": cd.CALIBRATION_CALIBRATED if accepted else cd.CALIBRATION_UNCALIBRATED,
        "reason": None if accepted else DIAG_CALIBRATION_NOT_VALIDATED,
        "training_origins": len(ordered),
        "validation_origins": len(predictions),
        "raw_metrics": raw,
        "calibrated_metrics": adjusted,
        "mae_improvement_fraction": improvement,
        "central_90_interval_coverage": coverage,
        "final_bias_correction": final_correction,
        "final_residual_q05": _quantile(calibrated_residuals, 0.05),
        "final_residual_q95": _quantile(calibrated_residuals, 0.95),
    }


def evaluate_reservation_calibration(
    observations: Sequence[Mapping[str, Any]],
    *,
    evidence_verifier: Callable[[str], Mapping[str, Any]] | None,
    evaluation_cutoff: str,
) -> dict[str, Any]:
    """Evaluate a pre-registered, expanding-window reservation calibration.

    All supplied evidence is validated before fitting. Labels unavailable at the
    declared evaluation cutoff are excluded; labels unavailable at each
    walk-forward origin cannot enter that fold's training set.
    """

    cutoff = _utc(evaluation_cutoff, name="evaluation_cutoff")
    verified: list[VerifiedOpportunity] = []
    rejected: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw in observations:
        try:
            row = validate_observation(raw, evidence_verifier=evidence_verifier)
            if row.observation_id in seen:
                raise ReservationCalibrationError("duplicate observation_id")
            seen.add(row.observation_id)
            if row.label_available_at <= cutoff:
                verified.append(row)
            # Unmatured labels are intentionally not counted as failures or as
            # training data; they remain unavailable at this evaluation cutoff.
        except Exception as failure:
            rejected.append({"observation_id": str(raw.get("observation_id") or ""),
                             "reason": f"{type(failure).__name__}: {failure}"})
    groups: dict[tuple[str, str], list[VerifiedOpportunity]] = {}
    for row in verified:
        groups.setdefault((row.action, row.bucket), []).append(row)
    evaluated: dict[str, Any] = {}
    models: dict[str, Any] = {}
    for (action, bucket), rows in sorted(groups.items()):
        key = f"{action}:{bucket}"
        unique_origins: set[tuple[int, datetime]] = set()
        duplicates: list[str] = []
        for row in rows:
            origin_key = (row.planning_event, row.origin_cutoff)
            if origin_key in unique_origins:
                duplicates.append(row.observation_id)
            unique_origins.add(origin_key)
        report = (
            {
                "status": cd.CALIBRATION_UNCALIBRATED,
                "reason": DIAG_CALIBRATION_EVIDENCE_INVALID,
                "training_origins": len(rows),
                "validation_origins": 0,
                "duplicate_origin_observation_ids": duplicates,
            }
            if duplicates else _walk_forward_group(rows)
        )
        evaluated[key] = report
        if report.get("status") == cd.CALIBRATION_CALIBRATED:
            models[key] = {
                "bias_correction": float(report["final_bias_correction"]),
                "residual_q05": float(report["final_residual_q05"]),
                "residual_q95": float(report["final_residual_q95"]),
                "training_origins": len(rows),
            }
    observation_manifest = [
        {
            "observation_id": row.observation_id,
            "action": row.action,
            "planning_event": row.planning_event,
            "weeks_to_expiry": row.weeks_to_expiry,
            "origin_cutoff": row.origin_cutoff.isoformat().replace("+00:00", "Z"),
            "forecast_input_as_of": row.forecast_input_as_of.isoformat().replace("+00:00", "Z"),
            "forecast_made_at": row.forecast_made_at.isoformat().replace("+00:00", "Z"),
            "forecast_mode": row.forecast_mode,
            "label_available_at": row.label_available_at.isoformat().replace("+00:00", "Z"),
            "forecast_value": row.forecast_value,
            "realized_value": row.realized_value,
            "causal_evidence_ref": row.evidence_ref,
            "causal_evidence_sha256": row.evidence_sha256,
            "source_decision_id": row.source_decision_id,
            "evaluator_version": row.evaluator_version,
            "evaluator_forecast_value": row.evaluator_forecast_value,
            "evaluator_interval_low": row.evaluator_interval_low,
            "evaluator_interval_high": row.evaluator_interval_high,
        }
        for row in sorted(verified, key=lambda item: (item.origin_cutoff, item.action, item.observation_id))
    ]
    body = {
        "schema": CALIBRATION_SCHEMA,
        "version": CALIBRATION_VERSION,
        "policy": {
            "minimum_training_origins": MIN_TRAINING_ORIGINS,
            "minimum_validation_origins": MIN_VALIDATION_ORIGINS,
            "minimum_mae_improvement_fraction": MIN_MAE_IMPROVEMENT,
            "maximum_absolute_bias": MAX_ABSOLUTE_BIAS,
            "central_90_interval_coverage_minimum": MIN_90_INTERVAL_COVERAGE,
            "walk_forward": "EXPANDING_ORIGIN_LABELS_MATURED_BY_TEST_CUTOFF",
            "correction": "ACTION_AND_EXPIRY_BUCKET_INTERCEPT_ONLY",
            "causal_evidence_policy": CAUSAL_EVIDENCE_POLICY,
        },
        "evaluation_cutoff": cutoff.isoformat().replace("+00:00", "Z"),
        "evaluation_period": {
            "first_origin_cutoff": observation_manifest[0]["origin_cutoff"] if observation_manifest else None,
            "last_label_available_at": max(
                (item["label_available_at"] for item in observation_manifest), default=None
            ),
        },
        "verified_observations": len(verified),
        "verified_observation_manifest": observation_manifest,
        "dataset_identity_sha256": _canonical_sha256(observation_manifest),
        "rejected_observations": rejected,
        "action_bucket_reports": evaluated,
        "models": models,
        "status": (
            cd.CALIBRATION_CALIBRATED
            if models and not rejected else cd.CALIBRATION_UNCALIBRATED
        ),
    }
    # One contradictory/tampered record invalidates the calibration attempt;
    # callers must resolve the evidence set and rerun against the same locked
    # criteria rather than obtaining a fit by silently excluding it.
    if rejected:
        body["models"] = {}
        for report in body["action_bucket_reports"].values():
            report["status"] = cd.CALIBRATION_UNCALIBRATED
            report["reason"] = DIAG_CALIBRATION_EVIDENCE_INVALID
    body["model_identity_sha256"] = _canonical_sha256({
        "version": body["version"], "policy": body["policy"], "models": body["models"],
    })
    body["identity_sha256"] = _canonical_sha256(body)
    return body


@dataclass(frozen=True)
class VerifiedReservationCalibration:
    """ReservationValue backed by the module's verified immutable artifact."""

    artifact: Mapping[str, Any]
    evidence_verifier: Callable[[str], Mapping[str, Any]] | None = None
    forecast_evidence_verifier: Callable[[str], Mapping[str, Any]] | None = None

    @classmethod
    def from_artifact(
        cls,
        artifact: Mapping[str, Any],
        *,
        evidence_verifier: Callable[[str], Mapping[str, Any]] | None,
        forecast_evidence_verifier: Callable[[str], Mapping[str, Any]] | None = None,
    ) -> "VerifiedReservationCalibration":
        body = dict(artifact)
        identity = str(body.pop("identity_sha256", ""))
        if body.get("schema") != CALIBRATION_SCHEMA or body.get("version") != CALIBRATION_VERSION:
            raise ReservationCalibrationError("reservation calibration artifact schema/version is unknown")
        if identity != _canonical_sha256(body):
            raise ReservationCalibrationError("reservation calibration artifact digest does not verify")
        manifest = body.get("verified_observation_manifest")
        if not isinstance(manifest, list) or body.get("dataset_identity_sha256") != _canonical_sha256(manifest):
            raise ReservationCalibrationError("reservation calibration dataset identity does not verify")
        expected_model_identity = _canonical_sha256({
            "version": body.get("version"), "policy": body.get("policy"), "models": body.get("models"),
        })
        if body.get("model_identity_sha256") != expected_model_identity:
            raise ReservationCalibrationError("reservation model identity does not verify")
        if body.get("status") != cd.CALIBRATION_CALIBRATED or not body.get("models"):
            raise ReservationCalibrationError("reservation calibration artifact is not calibrated")
        if evidence_verifier is None:
            raise ReservationCalibrationError(
                "retained causal-evidence verifier is required to load a calibration artifact"
            )
        if body.get("rejected_observations"):
            raise ReservationCalibrationError("calibration artifact contains rejected causal observations")
        if int(body.get("verified_observations") or -1) != len(manifest):
            raise ReservationCalibrationError("calibration observation count does not match its manifest")
        required_row_fields = (
            "observation_id", "action", "planning_event", "weeks_to_expiry", "origin_cutoff",
            "forecast_input_as_of", "forecast_made_at", "forecast_mode", "label_available_at",
            "forecast_value", "realized_value",
            "causal_evidence_ref", "causal_evidence_sha256",
        )
        try:
            rows = [
                {key: item[key] for key in required_row_fields}
                for item in manifest
            ]
            reproduced = evaluate_reservation_calibration(
                rows,
                evidence_verifier=evidence_verifier,
                evaluation_cutoff=str(body["evaluation_cutoff"]),
            )
        except Exception as failure:
            raise ReservationCalibrationError(
                f"calibration evidence could not be independently revalidated: {failure}"
            ) from failure
        if reproduced.get("status") != cd.CALIBRATION_CALIBRATED or _canonical_sha256(
            reproduced
        ) != _canonical_sha256(dict(artifact)):
            raise ReservationCalibrationError(
                "calibration status or model does not reproduce from retained causal evidence and locked criteria"
            )
        return cls(
            artifact=dict(artifact),
            evidence_verifier=evidence_verifier,
            forecast_evidence_verifier=forecast_evidence_verifier or evidence_verifier,
        )

    def estimate(self, *, action: str, planning_event: int, expiry_event: int | None,
                 state: Mapping[str, Any]) -> cd.ReservationEstimate:
        weeks = None if expiry_event is None else max(0, int(expiry_event) - int(planning_event))
        model = (self.artifact.get("models") or {}).get(f"{action}:{expiry_bucket(weeks)}")
        if not isinstance(model, Mapping):
            return cd.ReservationEstimate(
                value=None, calibration_status=cd.CALIBRATION_UNCALIBRATED,
                terminal_value=0.0, weeks_to_expiry=weeks,
                reason_codes=(DIAG_CALIBRATION_INSUFFICIENT,),
                conditional_on=("action", "weeks_to_expiry_bucket"),
            )
        raw_forecast = state.get("raw_reservation_forecast")
        if not isinstance(raw_forecast, Mapping):
            return cd.ReservationEstimate(
                value=None, calibration_status=cd.CALIBRATION_UNCALIBRATED,
                terminal_value=0.0, weeks_to_expiry=weeks,
                reason_codes=(DIAG_RESERVATION_FORECAST_REQUIRED,),
                conditional_on=("action", "weeks_to_expiry_bucket"),
            )
        try:
            forecast_verifier = self.forecast_evidence_verifier or self.evidence_verifier
            if forecast_verifier is None:
                raise crf.ReservationForecastError("retained event-opportunity verifier is required")
            verified_forecast = crf.VerifiedReservationForecast(artifact=raw_forecast)
            verified_forecast.validate_for_reservation(
                action=action,
                planning_event=int(planning_event),
                expiry_event=expiry_event,
                state=state,
                evidence_verifier=forecast_verifier,
            )
        except Exception:
            return cd.ReservationEstimate(
                value=None, calibration_status=cd.CALIBRATION_UNCALIBRATED,
                terminal_value=0.0, weeks_to_expiry=weeks,
                reason_codes=(crf.FORECAST_IDENTITY_INVALID,),
                conditional_on=("action", "weeks_to_expiry_bucket", "verified_raw_forecast"),
            )
        raw = verified_forecast.raw_value
        if raw is None:
            return cd.ReservationEstimate(
                value=None, calibration_status=cd.CALIBRATION_UNCALIBRATED,
                terminal_value=0.0, weeks_to_expiry=weeks,
                reason_codes=(DIAG_RESERVATION_FORECAST_INCOMPLETE,),
                conditional_on=("action", "weeks_to_expiry_bucket", "verified_raw_forecast"),
            )
        declared_raw = state.get("raw_reservation_value")
        if declared_raw is not None and _number(declared_raw, name="raw reservation value") != raw:
            return cd.ReservationEstimate(
                value=None, calibration_status=cd.CALIBRATION_UNCALIBRATED,
                terminal_value=0.0, weeks_to_expiry=weeks,
                reason_codes=(crf.FORECAST_IDENTITY_INVALID,),
                conditional_on=("action", "weeks_to_expiry_bucket", "verified_raw_forecast"),
            )
        value = max(0.0, _number(raw, name="raw reservation value") + float(model["bias_correction"]))
        return cd.ReservationEstimate(
            value=value,
            calibration_status=cd.CALIBRATION_CALIBRATED,
            terminal_value=0.0,
            weeks_to_expiry=weeks,
            reason_codes=(),
            conditional_on=("action", "weeks_to_expiry_bucket"),
        )
