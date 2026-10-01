"""Production assembly for one coherent, four-chip operational assessment.

The entry point requires an explicit route id and a manager confirmation that
is already present in the certified generation's pinned snapshot before it
loads any world matrices or runs either route optimizer.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Any, Callable, Mapping

from . import (
    chip_decision as cd,
    chip_route_assembly as cra,
    chip_reservation_forecast as crf,
    free_hit_production as fhp,
    free_hit_request_adapter as fha,
    generation_store as gs,
    planning,
    repositories as repo,
    wildcard_production as wcp,
    wildcard_request_adapter as wa,
)
from .chip_assessment_store import (
    build_assessment_record,
    free_hit_arm_identity,
    manager_state_identity,
)
from .chip_reservation_calibration import VerifiedReservationCalibration

CHIP_MANAGER_CONFIRMATION_REQUIRED = "CHIP_PRODUCTION_MANAGER_CONFIRMATION_REQUIRED"
CHIP_NORMAL_CONTEXT_INVALID = "CHIP_PRODUCTION_NORMAL_CONTEXT_INVALID"
CHIP_WORLD_INPUTS_UNAVAILABLE = "CHIP_PRODUCTION_CERTIFIED_WORLDS_UNAVAILABLE"
CHIP_ASSESSMENT_CALIBRATION_INVALID = "CHIP_RESERVATION_CALIBRATION_ARTIFACT_INVALID"


class ChipAssessmentPreflightError(ValueError):
    def __init__(self, detail: str, *, action_refusals: Mapping[str, str] | None = None) -> None:
        super().__init__(detail)
        self.action_refusals = dict(action_refusals or {})


def _iso_utc(value: Any, *, name: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ChipAssessmentPreflightError(f"{CHIP_MANAGER_CONFIRMATION_REQUIRED}: {name} is missing")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as failure:
        raise ChipAssessmentPreflightError(
            f"{CHIP_MANAGER_CONFIRMATION_REQUIRED}: {name} is not an ISO timestamp"
        ) from failure
    if parsed.tzinfo is None:
        raise ChipAssessmentPreflightError(
            f"{CHIP_MANAGER_CONFIRMATION_REQUIRED}: {name} must be timezone-aware"
        )
    return parsed.astimezone(timezone.utc)


def _evaluation_dict(evaluation: cd.ChipEvaluation) -> dict[str, Any]:
    return {
        "action": evaluation.action,
        "evaluator_version": evaluation.evaluator_version,
        "candidate_metrics": dict(evaluation.candidate_metrics),
        "uncertainty": dict(evaluation.uncertainty),
        "reason_codes": list(evaluation.reason_codes),
        "calibration_status": evaluation.calibration_status,
        "evidence": dict(evaluation.evidence),
        "execution_permitted": bool(evaluation.execution_permitted),
        "data_snapshot_bound": bool(evaluation.data_snapshot_bound),
    }


def assemble_assessment_record(
    *,
    context: Mapping[str, Any],
    manager_state: Mapping[str, Any],
    availability: list[Mapping[str, Any]],
    evaluations: Mapping[str, cd.ChipEvaluation],
    blocked: Mapping[str, tuple[str, str]] | None = None,
    extra_evidence: Mapping[str, Mapping[str, Any]] | None = None,
    reservation: Any | None = None,
    reservation_forecasts: Mapping[str, Mapping[str, Any]] | None = None,
    reservation_forecast_evidence_verifier: Callable[[str], Mapping[str, Any]] | None = None,
    chips_already_played_for_event: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Create the four-chip record and run the unchanged canonical arbiter."""

    by_action = cd._availability_by_action(availability, planning_event=int(context["planning_event"]))
    blocked = dict(blocked or {})
    extra_evidence = dict(extra_evidence or {})
    expiry_by_action: dict[str, int | None] = {}
    for action in cd.PLAYABLE_CHIP_ACTIONS:
        row = by_action.get(action) or {}
        definitions = row.get("definitions") or ()
        if definitions:
            active_definition, window_problem = cd._active_definition(definitions)
            expiry_by_action[action] = (
                None if window_problem is not None or active_definition is None
                else (None if active_definition.get("window_stop_event") is None
                      else int(active_definition["window_stop_event"]))
            )
        else:
            expiry_by_action[action] = (
                None if row.get("window_stop_event") is None else int(row["window_stop_event"])
            )
    verified_forecasts: dict[str, Mapping[str, Any]] = {}
    if reservation_forecasts and reservation_forecast_evidence_verifier is None:
        raise ChipAssessmentPreflightError(
            f"{crf.FORECAST_IDENTITY_INVALID}: a retained event-opportunity verifier is required"
        )
    for action, artifact in dict(reservation_forecasts or {}).items():
        row = by_action.get(str(action))
        if action not in cd.PLAYABLE_CHIP_ACTIONS or row is None or not bool(row.get("eligible")):
            raise ChipAssessmentPreflightError(
                f"{crf.FORECAST_IDENTITY_INVALID}: forecast supplied for an unavailable chip action {action}"
            )
        expiry_event = expiry_by_action[str(action)]
        source_identity = {
            "source_decision_id": context["source_decision_id"],
            "source_result_sha256": context["source_decision_result_sha256"],
            "source_artifact_sha256": context["source_decision_artifact_sha256"],
            "generation_id": context["generation_id"],
            "planning_event": int(context["planning_event"]),
            "origin_cutoff": str(context["cutoff"]),
            "data_snapshot_sha256": context["data_snapshot_sha256"],
            "predictive_code_snapshot_sha256": context.get("predictive_code_snapshot_sha256"),
            "certification_identity": context["certification_identity"],
        }
        try:
            crf.verify_reservation_forecast(
                artifact,
                expected={
                    "action": str(action),
                    "planning_event": int(context["planning_event"]),
                    "origin_cutoff": str(context["cutoff"]),
                    "expiry_event": None if expiry_event is None else int(expiry_event),
                    "source_identity": source_identity,
                },
                evidence_verifier=reservation_forecast_evidence_verifier,
            )
        except Exception as failure:
            raise ChipAssessmentPreflightError(
                f"{crf.FORECAST_IDENTITY_INVALID}: {failure}"
            ) from failure
        evaluation = evaluations.get(str(action))
        if evaluation is None:
            raise ChipAssessmentPreflightError(
                f"{crf.FORECAST_IDENTITY_INVALID}: forecast has no evaluated SAVE policy for {action}"
            )
        expected_state: dict[str, Any] = {"squad_ids": list(manager_state.get("squad_ids") or ())}
        save_policy = (evaluation.evidence.get("save_policy") or {}).get(
            "post_save_state_for_reservation"
        )
        if isinstance(save_policy, Mapping):
            expected_state.update(dict(save_policy))
        forecast_state = artifact.get("reservation_state")
        if not isinstance(forecast_state, Mapping) or dict(forecast_state) != expected_state:
            raise ChipAssessmentPreflightError(
                f"{crf.FORECAST_IDENTITY_INVALID}: {action} forecast does not match its verified SAVE state"
            )
        verified_forecasts[str(action)] = dict(artifact)
        extra_evidence[str(action)] = {
            **dict(extra_evidence.get(str(action)) or {}),
            "raw_reservation_forecast": dict(artifact),
        }
    chip_results: dict[str, dict[str, Any]] = {}
    for action in cd.PLAYABLE_CHIP_ACTIONS:
        evaluation = evaluations.get(action)
        row = by_action.get(action)
        if evaluation is not None and row is not None and bool(row.get("eligible")):
            item: dict[str, Any] = {
                "status": "EVALUATED",
                "evaluation": _evaluation_dict(evaluation),
                "reason_codes": list(evaluation.reason_codes),
            }
            item.update(dict(extra_evidence.get(action) or {}))
        elif action in blocked:
            token, detail = blocked[action]
            item = {"status": "BLOCKED", "reason_codes": [str(token)], "detail": str(detail)}
        elif row is None or not bool(row.get("eligible")):
            item = {
                "status": "UNAVAILABLE",
                "reason_codes": [cd.DIAG_CHIP_UNAVAILABLE],
                "detail": "canonical chip state does not make this chip eligible for the planning event",
            }
        else:
            item = {
                "status": "BLOCKED",
                "reason_codes": [f"{cd.DIAG_CHIP_EVALUATOR_NOT_IMPLEMENTED}:{action}"],
                "detail": "no verified production evaluator was assembled",
            }
        chip_results[action] = item

    binding = cd.ChipHorizonBinding(
        planning_event=int(context["planning_event"]),
        horizon_events=tuple(int(event) for event in context["horizon_events"]),
        certification_identity=str(context["certification_identity"]),
        data_snapshot_sha256=str(context["data_snapshot_sha256"]),
    )
    decision = cd.decide_chip_action(
        horizon_binding=binding,
        chip_availability=availability,
        evaluations=evaluations,
        reservation=reservation or cd.UncalibratedReservation(),
        reservation_forecasts=verified_forecasts,
        certification_valid=True,
        manager_state=manager_state,
        chips_already_played_for_event=chips_already_played_for_event,
    )
    retained_context = dict(context)
    retained_context["chip_expiry_events"] = dict(expiry_by_action)
    return build_assessment_record(
        context=retained_context,
        manager_state=manager_state,
        chip_results=chip_results,
        decision=decision,
    )


def _manager_snapshot(source_conn: sqlite3.Connection, *, entry_id: int, route: Any) -> tuple[Any, dict[str, Any]]:
    event = int(route.planning_event)
    cutoff = str(route.cutoff)
    context = planning.get_planning_context(source_conn, int(entry_id), event, cutoff)
    state = repo.manager_planning_state(source_conn, int(entry_id), event, cutoff)
    manual = state.get("manual")
    event_start = state.get("event_start_free_transfers")
    def action_refusals(detail: str) -> dict[str, tuple[str, str]]:
        return {
            cd.CHIP_ACTION_FH: (fha.FH_MANAGER_STATE_MISSING, detail),
            cd.CHIP_ACTION_WC: (wa.WC_MANAGER_STATE_MISSING, detail),
        }

    if event_start is None or not isinstance(manual, Mapping):
        detail = f"pinned manager state has event_start_free_transfers={event_start!r}"
        raise ChipAssessmentPreflightError(
            f"{CHIP_MANAGER_CONFIRMATION_REQUIRED}: explicit manager confirmation including "
            "event-start free transfers is absent from the pinned snapshot; no prediction worlds "
            "or route optimizer were loaded",
            action_refusals=action_refusals(detail),
        )
    captured_at = manual.get("captured_at")
    if captured_at is None or _iso_utc(captured_at, name="manager confirmation captured_at") > _iso_utc(
        cutoff, name="generation cutoff"
    ):
        raise ChipAssessmentPreflightError(
            f"{CHIP_MANAGER_CONFIRMATION_REQUIRED}: manager confirmation was not captured by the "
            "retained generation cutoff; no prediction worlds or route optimizer were loaded",
            action_refusals=action_refusals(
                "the confirmation timestamp is missing or later than the generation cutoff and is excluded"
            ),
        )
    if state.get("bank") is None or state.get("free_transfers") is None:
        raise ChipAssessmentPreflightError(
            f"{CHIP_MANAGER_CONFIRMATION_REQUIRED}: confirmed bank and current free transfers are "
            "required independently of event-start free transfers",
            action_refusals=action_refusals("bank or current free transfers are missing from the pinned manager state"),
        )
    squad = getattr(context, "squad", None)
    squad_rows = squad.get("players") if isinstance(squad, Mapping) else None
    if not isinstance(squad_rows, list) or not squad_rows:
        raise ChipAssessmentPreflightError(
            f"{CHIP_MANAGER_CONFIRMATION_REQUIRED}: pinned canonical squad is missing",
            action_refusals=action_refusals("pinned canonical squad is missing"),
        )
    actual = tuple(sorted(int(row["player_id"]) for row in squad_rows))
    if actual != tuple(route.actual_owned_ids):
        raise ChipAssessmentPreflightError(
            f"{CHIP_NORMAL_CONTEXT_INVALID}: pinned manager squad differs from the verified normal decision"
        )
    return context, state


def run_production_chip_assessment(
    conn: sqlite3.Connection,
    *,
    decision_id: str,
    entry_id: int,
    route_id: str,
    rules: Any,
    certification_path: str | None = None,
    wildcard_value_generation_id: str | None = None,
    cache_dir: str | None = None,
    reservation_calibration_artifact: Mapping[str, Any] | None = None,
    reservation_calibration_evidence_verifier: Callable[[str], Mapping[str, Any]] | None = None,
    reservation_forecasts: Mapping[str, Mapping[str, Any]] | None = None,
    reservation_forecast_evidence_verifier: Callable[[str], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run only after an explicit pre-cutoff manager confirmation is pinned.

    It deliberately requires the route id instead of choosing between a
    captured-lineup assessment and a hypothetical route. The BB/TC pair uses
    that same route, proposed squad, lineup, H1 matrix and world identity.
    """

    if not str(route_id or "").strip():
        raise ChipAssessmentPreflightError(f"{CHIP_NORMAL_CONTEXT_INVALID}: an explicit route id is required")
    route = cra.load_verified_normal_route(conn, decision_id, route_id=route_id)
    generation = gs.load_generation(conn, route.generation_id)
    report = gs.verify_generation(conn, generation.generation_id)
    if not report.get("verified") or generation.horizon_kind != gs.HORIZON_KIND_FOUR_GW:
        raise ChipAssessmentPreflightError(
            f"{CHIP_NORMAL_CONTEXT_INVALID}: normal four-event generation is not verified"
        )
    source_conn = gs._open_generation_snapshot(generation)
    try:
        manager_context, canonical = _manager_snapshot(source_conn, entry_id=entry_id, route=route)
        # Use the exact point-in-time state and its independent field provenance.
        manager = fha.free_hit_manager_state(
            source_conn,
            int(entry_id),
            route.planning_event,
            decision_cutoff=route.cutoff,
            as_of=route.cutoff,
        )
        if manager.problems():
            raise ChipAssessmentPreflightError(
                f"{CHIP_MANAGER_CONFIRMATION_REQUIRED}: canonical manager facts are incomplete: "
                + "; ".join(manager.problems()[:8])
            )
        availability = [dict(row) for row in (manager_context.chips or ())]
        by_action = cd._availability_by_action(availability, planning_event=route.planning_event)
        eligible = {action for action, row in by_action.items() if row.get("eligible")}

        # Validate every eligible long-running path before the first world load
        # or route search. Missing manager facts, a mismatched certificate, or
        # an absent Wildcard generation must not be discovered after expensive
        # operational work has started.
        if cd.CHIP_ACTION_FH in eligible:
            if not str(certification_path or "").strip():
                raise ChipAssessmentPreflightError(
                    f"{fha.FH_CERTIFICATION_REQUIRED}: eligible Free Hit requires the retained canonical "
                    "certification artifact",
                    action_refusals={cd.CHIP_ACTION_FH: (fha.FH_CERTIFICATION_REQUIRED,
                        "no certification artifact supplied")},
                )
            try:
                from .free_hit_decision_authority import load_decision_authority

                authority = load_decision_authority(certification_path)
                if (
                    str(authority.planning_cutoff) != route.cutoff
                    or str(authority.data_snapshot_sha256) != route.data_snapshot_sha256
                    or tuple(authority.certified_events) != route.events
                ):
                    raise ValueError("canonical Free Hit certification differs from normal decision identity")
            except Exception as failure:
                raise ChipAssessmentPreflightError(
                    f"{fha.FH_CERTIFICATION_REQUIRED}: {failure}",
                    action_refusals={cd.CHIP_ACTION_FH: (fha.FH_CERTIFICATION_REQUIRED, str(failure))},
                ) from failure

        if cd.CHIP_ACTION_WC in eligible:
            if not wildcard_value_generation_id:
                raise ChipAssessmentPreflightError(
                    f"{wcp.WC_PRODUCTION_INPUTS_MISSING}: eligible Wildcard requires a separate, verified "
                    "6-10 event value generation",
                    action_refusals={cd.CHIP_ACTION_WC: (
                        wcp.WC_PRODUCTION_INPUTS_MISSING,
                        "no separate Wildcard value generation supplied",
                    )},
                )
            try:
                value_generation = gs.load_generation(conn, str(wildcard_value_generation_id))
                value_report = gs.verify_generation(conn, value_generation.generation_id)
                if (
                    not value_report.get("verified")
                    or value_generation.horizon_kind != gs.HORIZON_KIND_WILDCARD_VALUE
                    or not 6 <= len(value_generation.events) <= 10
                    or value_generation.events[:4] != generation.events
                    or int(value_generation.planning_event) != int(generation.planning_event)
                    or str(value_generation.cutoff) != str(generation.cutoff)
                    or str(value_generation.snapshot.get("sha256")) != route.data_snapshot_sha256
                    or str(value_generation.manifest.get("code_snapshot_sha256"))
                    != str(generation.manifest.get("code_snapshot_sha256"))
                    or any(value_generation.runs_for(event) != generation.runs_for(event)
                           for event in generation.events)
                ):
                    raise ValueError("Wildcard generation does not match the verified normal product identity")
                if tuple(value_generation.events) != tuple(range(
                    route.planning_event, route.planning_event + len(value_generation.events)
                )):
                    raise ValueError("Wildcard generation events are not a contiguous horizon from planning event")
                wa.wildcard_manager_state(
                    source_conn,
                    int(entry_id),
                    route.planning_event,
                    cutoff=route.cutoff,
                    eligible_ids=manager.eligible_ids,
                    as_of=route.cutoff,
                )
            except Exception as failure:
                reasons = tuple(getattr(failure, "reasons", ()) or ())
                token = reasons[0] if reasons else wcp.WC_PRODUCTION_INPUTS_MISSING
                raise ChipAssessmentPreflightError(
                    f"{token}: {failure}",
                    action_refusals={cd.CHIP_ACTION_WC: (str(token), str(failure))},
                ) from failure

        evaluations: dict[str, cd.ChipEvaluation] = {}
        blocked: dict[str, tuple[str, str]] = {}
        extras: dict[str, dict[str, Any]] = {}

        if eligible.intersection({cd.CHIP_ACTION_BB, cd.CHIP_ACTION_TC}):
            try:
                worlds = cra.load_certified_h1_chip_worlds(conn, route, cache_dir=cache_dir)
                pair = cra.build_bb_tc_evaluations(route, worlds)
                evaluations.update({action: evaluation for action, evaluation in pair.items() if action in eligible})
            except Exception as failure:
                for action in eligible.intersection({cd.CHIP_ACTION_BB, cd.CHIP_ACTION_TC}):
                    blocked[action] = (CHIP_WORLD_INPUTS_UNAVAILABLE, str(failure))

        if cd.CHIP_ACTION_FH in eligible:
            try:
                request, arms = fhp.build_free_hit_production_request(
                    conn,
                    decision_id=decision_id,
                    entry_id=int(entry_id),
                    certification_path=certification_path,
                    rules=rules,
                    route_id=route_id,
                    cache_dir=cache_dir,
                )
                from . import chip_free_hit as fh

                evaluation = fh.evaluate_free_hit(request)
                evaluations[cd.CHIP_ACTION_FH] = evaluation
                shared_arm_context = {
                    "source_decision_id": route.source_decision_id,
                    "source_result_sha256": route.source_result_sha256,
                    "source_artifact_sha256": route.source_artifact_sha256,
                    "generation_id": route.generation_id,
                    "cutoff": route.cutoff,
                    "data_snapshot_sha256": route.data_snapshot_sha256,
                    "certification_identity": route.certification_identity,
                    "manager_state_identity": arms["manager_state_identity"],
                    "world_identity": arms["world_identity"],
                }
                arms["play"].update(shared_arm_context)
                arms["save"].update(shared_arm_context)
                arms["arm_identity"] = free_hit_arm_identity(arms)
                extras[cd.CHIP_ACTION_FH] = {"arm_evidence": arms, "arm_identity": arms["arm_identity"]}
            except Exception as failure:
                reasons = tuple(getattr(failure, "reasons", ()) or ())
                token = reasons[0] if reasons else fhp.FH_PRODUCTION_ARMS_INVALID
                blocked[cd.CHIP_ACTION_FH] = (str(token), str(failure))

        if cd.CHIP_ACTION_WC in eligible:
            if not wildcard_value_generation_id:
                blocked[cd.CHIP_ACTION_WC] = (
                    wcp.WC_PRODUCTION_INPUTS_MISSING,
                    "a separately certified 6-10 event Wildcard value generation is required",
                )
            else:
                try:
                    value_generation = gs.load_generation(conn, str(wildcard_value_generation_id))
                    request, evidence = wcp.build_wildcard_production_request(
                        conn,
                        decision_id=decision_id,
                        value_generation_id=wildcard_value_generation_id,
                        entry_id=int(entry_id),
                        length=len(value_generation.events),
                        rules=rules,
                        cache_dir=cache_dir,
                        route_id=route_id,
                    )
                    from . import chip_wildcard as wc

                    evaluations[cd.CHIP_ACTION_WC] = wc.evaluate_wildcard(request)
                    extras[cd.CHIP_ACTION_WC] = {
                        "value_generation_id": evidence["value_generation_id"],
                        "value_horizon_events": evidence["events"],
                        "production_evidence": evidence,
                    }
                except Exception as failure:
                    reasons = tuple(getattr(failure, "reasons", ()) or ())
                    token = reasons[0] if reasons else wcp.WC_PRODUCTION_INPUTS_MISSING
                    blocked[cd.CHIP_ACTION_WC] = (str(token), str(failure))

        reservation: Any = cd.UncalibratedReservation()
        if reservation_calibration_artifact is not None:
            try:
                reservation = VerifiedReservationCalibration.from_artifact(
                    reservation_calibration_artifact,
                    evidence_verifier=reservation_calibration_evidence_verifier,
                    forecast_evidence_verifier=reservation_forecast_evidence_verifier,
                )
            except Exception as failure:
                raise ChipAssessmentPreflightError(
                    f"{CHIP_ASSESSMENT_CALIBRATION_INVALID}: {failure}"
                ) from failure

        manager_for_arbiter = {
            "entry_id": int(manager.entry_id),
            "planning_event": int(manager.planning_event),
            "cutoff": str(manager.cutoff),
            "squad_ids": list(route.actual_owned_ids),
            "purchase_price_tenths": dict(manager.purchase_price_tenths),
            "bank_tenths": int(manager.bank_tenths),
            "free_transfers": int(manager.free_transfers),
            "event_start_free_transfers": int(manager.event_start_free_transfers),
        }
        manager_for_arbiter["manager_state_identity"] = manager_state_identity(manager_for_arbiter)
        context = {
            "planning_event": route.planning_event,
            "horizon_events": list(route.events),
            "cutoff": route.cutoff,
            "data_snapshot_sha256": route.data_snapshot_sha256,
            "certification_identity": route.certification_identity,
            "generation_id": route.generation_id,
            "predictive_code_snapshot_sha256": str(generation.manifest.get("code_snapshot_sha256") or ""),
            "source_decision_id": route.source_decision_id,
            "source_decision_result_sha256": route.source_result_sha256,
            "source_decision_artifact_sha256": route.source_artifact_sha256,
            "route_id": route.route_id,
            "scenario_identity": next((
                dict(evaluation.evidence).get("scenario_identity")
                for evaluation in evaluations.values()
                if evaluation.action in {cd.CHIP_ACTION_BB, cd.CHIP_ACTION_TC}
            ), None),
            "manager_state_identity": manager_for_arbiter["manager_state_identity"],
        }
        used_chips = tuple(
            str(row.get("name")) for row in availability
            if row.get("used") and int(row.get("used_event") or -1) == route.planning_event
        )
        return assemble_assessment_record(
            context=context,
            manager_state=manager_for_arbiter,
            availability=availability,
            evaluations=evaluations,
            blocked=blocked,
            extra_evidence=extras,
            reservation=reservation,
            reservation_forecasts=reservation_forecasts,
            reservation_forecast_evidence_verifier=reservation_forecast_evidence_verifier,
            chips_already_played_for_event=used_chips,
        )
    finally:
        source_conn.close()
