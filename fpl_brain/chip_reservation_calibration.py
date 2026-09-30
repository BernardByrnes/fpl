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
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Sequence

from . import chip_decision as cd
from . import manager_lineup as ml
from . import outcome_ledger as ol

CALIBRATION_SCHEMA = "fpl_brain.chip_reservation_calibration.v1"
CAUSAL_EVIDENCE_SCHEMA = "fpl_brain.chip_reservation_causal_evidence.v2"
CAUSAL_OUTCOME_RECORD_SCHEMA = "fpl_brain.chip_reservation_outcome_record.v2"
OUTCOME_CAPTURE_SET_SCHEMA = "fpl_brain.outcome_ledger_capture_set.v1"
OUTCOME_LABEL_DEFINITION = "CANONICAL_BB_TC_PLAY_MINUS_SAVE_EVENT_POINTS_V2"
OUTCOME_SCORING_RULE_VERSION = "manager_lineup_and_bb_tc_realized_scores_v1"
CALIBRATION_VERSION = "chip_reservation_walkforward_v3.0.0"
CAUSAL_EVIDENCE_POLICY = "SAME_SCENARIO_PAIRED_PLAY_SAVE_TEMPORAL_LABELS_v2"

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
    """Rebuild realized BB/TC point weights from the retained FPL policy.

    Caller-provided weights are data to verify, never the scoring authority.
    FH/WC need action-specific squad and transfer-state scorers; until those are
    implemented, they cannot contribute calibration labels.
    """

    if action not in {cd.CHIP_ACTION_BB, cd.CHIP_ACTION_TC}:
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
        "certification_identity", "world_identity",
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
    player_points: dict[str, float] = {}
    player_minutes: dict[str, float] = {}
    capture_ids: set[str] = set()
    capture_times: list[datetime] = []
    for capture in captures:
        if not isinstance(capture, Mapping):
            raise ReservationCalibrationError("outcome capture manifest is malformed")
        digest = str(capture.get("capture_digest") or "")
        if not _is_sha256(digest) or digest in capture_ids:
            raise ReservationCalibrationError("outcome capture manifest has an invalid or duplicate digest")
        capture_ids.add(digest)
        if (
            capture.get("grain") != "player_event"
            or int(capture.get("event") or -1) != realization_event
            or capture.get("fixture_id") is not None
            or capture.get("observation_state") != ol.OBSERVATION_FINAL
            or capture.get("source_name") != "player_gameweeks_final"
            or capture.get("source_identity") != f"player_gameweeks:{realization_event}"
        ):
            raise ReservationCalibrationError(
                "outcome capture is not a final event-grain official player-gameweek result"
            )
        player_id = str(int(capture.get("player_id")))
        if player_id in player_points:
            raise ReservationCalibrationError("outcome capture set repeats a player in the realization event")
        player_points[player_id] = _number(capture.get("total_points"), name="captured player total_points")
        minutes = _number(capture.get("minutes"), name="captured player minutes")
        if minutes < 0.0:
            raise ReservationCalibrationError("captured player minutes cannot be negative")
        player_minutes[player_id] = minutes
        final_at = _utc(capture.get("official_final_at"), name="outcome official_final_at")
        captured_at = _utc(capture.get("captured_at"), name="outcome captured_at")
        if captured_at < final_at:
            raise ReservationCalibrationError("outcome capture predates official finality")
        capture_times.append(captured_at)
    available_at = max(capture_times)
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
        weights = arm.get("scoring_weights")
        payload = arm.get("artifact_payload")
        payload_weights = payload.get("scoring_weights") if isinstance(payload, Mapping) else None
        result_weights = result.get("scoring_weights") if isinstance(result, Mapping) else None
        if (
            not isinstance(result, Mapping)
            or not isinstance(weights, Mapping)
            or not weights
            or not isinstance(payload, Mapping)
            or not isinstance(payload_weights, Mapping)
            or not isinstance(result_weights, Mapping)
            or dict(payload_weights) != dict(weights)
            or dict(result_weights) != dict(weights)
        ):
            raise ReservationCalibrationError(f"{arm_name.upper()} score is not bound to its retained arm scorer")
        if (
            result.get("lineup") != arm.get("lineup")
            or result.get("player_positions") != arm.get("player_positions")
        ):
            raise ReservationCalibrationError(
                f"{arm_name.upper()} outcome scorer differs from its retained lineup or source positions"
            )
        expected_weights = _canonical_scoring_weights(
            action=str(row.get("action") or ""),
            arm_name=arm_name,
            arm=arm,
            player_points={int(key): value for key, value in player_points.items()},
            player_minutes={int(key): value for key, value in player_minutes.items()},
        )
        normalized_weights: dict[str, float] = {}
        for raw_player_id, raw_weight in weights.items():
            try:
                player_id = str(int(raw_player_id))
            except (TypeError, ValueError) as failure:
                raise ReservationCalibrationError(
                    f"{arm_name.upper()} scorer has an invalid player id"
                ) from failure
            if player_id in normalized_weights:
                raise ReservationCalibrationError(
                    f"{arm_name.upper()} scorer has a duplicate player id"
                )
            normalized_weights[player_id] = _number(raw_weight, name=f"{arm_name} scoring weight")
        if normalized_weights != expected_weights:
            raise ReservationCalibrationError(
                f"{arm_name.upper()} scoring weights do not reproduce from the canonical "
                f"{row.get('action')} scorer"
            )
        score = sum(
            player_points[player_id] * weight
            for player_id, weight in expected_weights.items()
        )
        declared_score = _number(result.get("observed_points"), name=f"{arm_name} observed points")
        if declared_score != score:
            raise ReservationCalibrationError(f"{arm_name.upper()} score does not reproduce from official captures")
        scores[arm_name] = score
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
    if not isinstance(play_positions, Mapping) or dict(play_positions) != dict(save_positions or {}):
        raise ReservationCalibrationError("paired outcome scoring does not use one retained source position map")
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
            normalized_play = {str(int(key)): str(value) for key, value in play_positions.items()}
        except (TypeError, ValueError) as failure:
            raise ReservationCalibrationError("paired outcome contains an invalid source position map") from failure
        if normalized_play != normalized_expected:
            raise ReservationCalibrationError(
                "paired outcome source positions differ from the pinned generation snapshot"
            )


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
    forecast_made_at: datetime
    label_available_at: datetime
    forecast_value: float
    realized_value: float
    evidence_ref: str
    evidence_sha256: str

    @property
    def bucket(self) -> str:
        return expiry_bucket(self.weeks_to_expiry)


def _verify_pair(
    evidence: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    retained_outcome_record: Mapping[str, Any],
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
        "certification_identity", "world_identity",
    )
    if any(not str(source.get(name) or "").strip() for name in required_source):
        raise ReservationCalibrationError("causal evidence has incomplete source decision/generation identity")
    for name in ("source_result_sha256", "source_artifact_sha256", "data_snapshot_sha256",
                 "certification_identity", "world_identity"):
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
    paired_fields = (
        "scenario_identity", "world_identity", "proposed_squad_ids", "lineup", "player_positions",
    )
    for name in paired_fields:
        if play.get(name) != save.get(name) or play.get(name) in (None, "", [], {}):
            raise ReservationCalibrationError(f"PLAY and SAVE do not share the same {name}")
    source_arm_fields = (
        "source_decision_id", "source_result_sha256", "source_artifact_sha256",
        "generation_id", "cutoff", "data_snapshot_sha256", "certification_identity",
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
        for name in (*paired_fields, *source_arm_fields, "arm_id"):
            if artifact_payload.get(name) != arm.get(name):
                raise ReservationCalibrationError(f"{arm_name} arm artifact disagrees on {name}")
        if not isinstance(arm.get("scoring_weights"), Mapping):
            raise ReservationCalibrationError(f"{arm_name} arm has no retained outcome scoring weights")
        if dict(artifact_payload.get("scoring_weights") or {}) != dict(arm.get("scoring_weights") or {}):
            raise ReservationCalibrationError(f"{arm_name} arm scoring weights differ from its retained artifact")
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
    if forecast_time > origin:
        raise ReservationCalibrationError("reservation forecast was made after its origin cutoff")
    if label_time <= origin:
        raise ReservationCalibrationError("future outcome label was already available at the forecast cutoff")
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
    _verify_pair(evidence, row, retained_outcome_record=retained_outcome_record)
    forecast = evidence.get("forecast")
    if not isinstance(forecast, Mapping):
        raise ReservationCalibrationError("causal evidence has no retained origin forecast")
    if (
        _number(forecast.get("value"), name="evidence forecast")
        != _number(row.get("forecast_value"), name="forecast value")
        or str(forecast.get("made_at")) != str(row.get("forecast_made_at"))
    ):
        raise ReservationCalibrationError("causal evidence forecast differs from the supplied row")
    weeks = row.get("weeks_to_expiry")
    if weeks is None or int(weeks) < 0:
        raise ReservationCalibrationError("weeks_to_expiry must be an explicit non-negative integer")
    return VerifiedOpportunity(
        observation_id=observation_id,
        action=action,
        planning_event=int(row.get("planning_event")),
        weeks_to_expiry=int(weeks),
        origin_cutoff=origin,
        forecast_made_at=forecast_time,
        label_available_at=label_time,
        forecast_value=_number(row.get("forecast_value"), name="forecast value"),
        realized_value=_number(row.get("realized_value"), name="realized value"),
        evidence_ref=reference,
        evidence_sha256=expected_digest,
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
            "forecast_made_at": row.forecast_made_at.isoformat().replace("+00:00", "Z"),
            "label_available_at": row.label_available_at.isoformat().replace("+00:00", "Z"),
            "forecast_value": row.forecast_value,
            "realized_value": row.realized_value,
            "causal_evidence_ref": row.evidence_ref,
            "causal_evidence_sha256": row.evidence_sha256,
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

    @classmethod
    def from_artifact(
        cls,
        artifact: Mapping[str, Any],
        *,
        evidence_verifier: Callable[[str], Mapping[str, Any]] | None,
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
            "forecast_made_at", "label_available_at", "forecast_value", "realized_value",
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
        return cls(artifact=dict(artifact))

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
        raw = state.get("raw_reservation_value")
        if raw is None:
            return cd.ReservationEstimate(
                value=None, calibration_status=cd.CALIBRATION_UNCALIBRATED,
                terminal_value=0.0, weeks_to_expiry=weeks,
                reason_codes=(DIAG_CALIBRATION_INSUFFICIENT,),
                conditional_on=("action", "weeks_to_expiry_bucket"),
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
