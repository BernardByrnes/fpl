"""Additive FH/WC operator orchestration over accepted forecast/evidence APIs.

The existing ``chip_evidence_ops`` BB/TC identities remain unchanged. This
module records action-specific FH/WC eligibility beside that verified origin
and composes future typed requests from explicit continuation products.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from . import chip_decision as cd

HORIZON_ORIGIN_SCHEMA = "fpl_brain.chip_horizon_evidence_origin.v1"
HORIZON_FORECAST_INTENT_SCHEMA = "fpl_brain.chip_horizon_forecast_intent.v1"
HORIZON_FORECAST_PUBLICATION_SCHEMA = "fpl_brain.chip_horizon_forecast_publication.v1"


def _action_name(action: str) -> tuple[str, str]:
    value = str(action).strip().upper()
    choices = {
        "FH": (cd.CHIP_ACTION_FH, "freehit"),
        "FREE_HIT": (cd.CHIP_ACTION_FH, "freehit"),
        cd.CHIP_ACTION_FH: (cd.CHIP_ACTION_FH, "freehit"),
        "WC": (cd.CHIP_ACTION_WC, "wildcard"),
        "WILDCARD": (cd.CHIP_ACTION_WC, "wildcard"),
        cd.CHIP_ACTION_WC: (cd.CHIP_ACTION_WC, "wildcard"),
    }
    if value not in choices:
        raise ValueError("horizon evidence action must be FH or WC")
    return choices[value]


def _pinned_chip_state(rows: list[Mapping[str, Any]], *, action: str, name: str) -> dict[str, Any]:
    matches = [dict(row) for row in rows if str(row.get("name") or "").lower() == name]
    eligible = [
        row for row in matches
        if row.get("available_for_event") is True
        and row.get("used") is False
        and row.get("window_start_event") is not None
        and row.get("window_stop_event") is not None
    ]
    if len(eligible) != 1:
        raise RuntimeError(
            f"pinned official chip state does not resolve one available {action} window "
            f"(found {len(eligible)})"
        )
    row = eligible[0]
    start, stop = int(row["window_start_event"]), int(row["window_stop_event"])
    if start <= 0 or stop < start:
        raise RuntimeError(f"pinned official {action} window is malformed")
    return {
        "action": action,
        "chip_name": name,
        "eligible_at_origin": True,
        "window_start_event": start,
        "expiry_event": stop,
        "used": False,
        "used_event": row.get("used_event"),
        "pinned_row": row,
    }


def register_horizon_origin(
    conn: Any,
    *,
    db_path: str | Path,
    entry_id: int,
    generation_id: str,
    decision_id: str,
    route_id: str,
    planning_event: int,
    cutoff: str,
    action: str,
    evidence_root: str | Path,
) -> dict[str, Any]:
    """Register the canonical normal origin plus immutable FH/WC eligibility."""

    from . import chip_evidence_ops as ops

    canonical_action, chip_name = _action_name(action)
    identity = ops._required_identity(
        db_path=db_path, entry_id=entry_id, generation_id=generation_id,
        decision_id=decision_id, route_id=route_id, planning_event=planning_event,
        cutoff=cutoff, evidence_root=evidence_root,
    )
    origin = ops.register_origin(
        conn,
        db_path=db_path,
        entry_id=entry_id,
        generation_id=generation_id,
        decision_id=decision_id,
        route_id=route_id,
        planning_event=planning_event,
        cutoff=cutoff,
        evidence_root=evidence_root,
    )
    origin_id = str(origin.get("origin", {}).get("origin_id") or "")
    if not origin_id:
        raise ops.ChipEvidenceError("verified base origin has no origin identity")
    key = f"horizon-origin-{origin_id}-{canonical_action}"
    receipt_path = ops._metadata_root(evidence_root) / f"{key}.json"
    with ops._operation_lock(evidence_root, key):
        _generation, _decision, route, state, permission = ops._origin_verification(
            conn,
            entry_id=entry_id,
            generation_id=generation_id,
            decision_id=decision_id,
            route_id=route_id,
            planning_event=planning_event,
            cutoff=cutoff,
        )
        if permission.get("permitted") is not True or permission.get("reasons") != []:
            raise ops.ChipEvidenceError("FH/WC origin search permission is not permitted")
        chip_state = _pinned_chip_state(
            list(state.get("chip_rows") or ()), action=canonical_action, name=chip_name,
        )
        expected_base = {
            "origin_id": origin_id,
            "identity_sha256": ops._sha256(identity),
            "action": canonical_action,
            "registered_at": None,
            "source_decision_id": route.source_decision_id,
            "generation_id": route.generation_id,
            "route_id": route.route_id,
            "route_input_sha256": route.route_input_sha256,
            "origin_cutoff": str(cutoff),
            "chip_eligibility": chip_state,
        }
        if receipt_path.exists():
            receipt = ops._read_json(receipt_path)
            ops._verify_seal(receipt, schema=HORIZON_ORIGIN_SCHEMA)
            ref = str(receipt.get("artifact_ref") or "")
            if Path(ref).name != ref:
                raise ops.ChipEvidenceError("FH/WC origin receipt has an unsafe artifact reference")
            artifact = ops._read_json(ops._root(evidence_root) / ref)
            body = dict(artifact)
            digest = str(body.pop("artifact_sha256", ""))
            if (
                digest != ops._sha256(body)
                or ref != f"chip-horizon-origin-{digest}.json"
                or any(artifact.get(field) != value for field, value in expected_base.items() if field != "registered_at")
                or str(artifact.get("registered_at") or "") != str(receipt.get("registered_at") or "")
                or str(receipt.get("artifact_sha256") or "") != digest
            ):
                raise ops.ChipEvidenceError("retained FH/WC origin evidence differs from the verified source")
            ops._parse_utc(receipt.get("registered_at"), name="FH/WC origin registered_at")
            return {"receipt": receipt, "origin": artifact}

        artifact = {
            "schema": HORIZON_ORIGIN_SCHEMA,
            **expected_base,
            "registered_at": ops._utc_now(),
        }
        artifact["artifact_sha256"] = ops._sha256(artifact)
        artifact_ref = f"chip-horizon-origin-{artifact['artifact_sha256']}.json"
        ops._write_new_json(ops._root(evidence_root) / artifact_ref, artifact)
        receipt = ops._sealed(HORIZON_ORIGIN_SCHEMA, {
            "origin_id": origin_id,
            "identity_sha256": ops._sha256(identity),
            "action": canonical_action,
            "artifact_ref": artifact_ref,
            "artifact_sha256": artifact["artifact_sha256"],
            "registered_at": artifact["registered_at"],
        })
        ops._write_new_json(receipt_path, receipt)
        return {"receipt": receipt, "origin": artifact}


def _verify_horizon_origin(
    evidence_root: str | Path,
    *,
    origin_id: str,
    identity: Mapping[str, Any],
    action: str,
) -> dict[str, Any]:
    from . import chip_evidence_ops as ops

    canonical_action, _name = _action_name(action)
    receipt_path = ops._metadata_root(evidence_root) / f"horizon-origin-{origin_id}-{canonical_action}.json"
    receipt = ops._read_json(receipt_path)
    ops._verify_seal(receipt, schema=HORIZON_ORIGIN_SCHEMA)
    ref = str(receipt.get("artifact_ref") or "")
    if Path(ref).name != ref or not ref:
        raise ops.ChipEvidenceError("FH/WC origin receipt has an unsafe artifact reference")
    artifact = ops._read_json(ops._root(evidence_root) / ref)
    body = dict(artifact)
    digest = str(body.pop("artifact_sha256", ""))
    if (
        digest != ops._sha256(body)
        or ref != f"chip-horizon-origin-{digest}.json"
        or receipt.get("origin_id") != origin_id
        or receipt.get("identity_sha256") != ops._sha256(identity)
        or artifact.get("origin_id") != origin_id
        or artifact.get("identity_sha256") != ops._sha256(identity)
        or artifact.get("action") != canonical_action
        or receipt.get("action") != canonical_action
        or receipt.get("artifact_sha256") != digest
    ):
        raise ops.ChipEvidenceError("FH/WC origin receipt/artifact binding does not verify")
    ops._parse_utc(artifact.get("registered_at"), name="FH/WC origin registered_at")
    return artifact


def _forecast_receipt_from_publication(
    conn: Any,
    *,
    identity: Mapping[str, Any],
    origin_id: str,
    operation_id: str,
    observation_id: str,
    action: str,
    expiry_event: int,
    publication: Mapping[str, Any],
    intent: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reverify a complete publication and derive its deterministic receipt."""
    from . import chip_evidence_ops as ops
    from . import chip_reservation_calibration as cal, chip_reservation_forecast as crf
    from . import generation_store as gs

    ops._verify_seal(intent, schema=ops.FORECAST_INTENT_SCHEMA)
    ops._verify_seal(publication, schema=ops.FORECAST_PUBLICATION_SCHEMA)
    calculation_started_at = str(intent.get("calculation_started_at") or "")
    issued_at = str(publication.get("issued_at") or "")
    available_at = str(publication.get("forecast_available_at") or "")
    calculation_started = ops._parse_utc(calculation_started_at, name="forecast calculation_started_at")
    issued = ops._parse_utc(issued_at, name="forecast issued_at")
    available = ops._parse_utc(available_at, name="forecast_available_at")
    if issued < calculation_started or available < issued:
        raise ops.ChipEvidenceError(
            "forecast_available_at must be a valid completed-availability time no earlier than issuance"
        )

    ops._verify_prospective_state_sequence(
        checks=(
            intent.get("prospective_preflight") or (),
            intent.get("prospective_calculation_start") or (),
            publication.get("prospective_issuance") or (),
        ),
        events=list(range(int(identity["planning_event"]) + 1, int(expiry_event) + 1)),
        cutoff=str(identity["cutoff"]),
        calculation_started_at=str(intent.get("calculation_started_at") or ""),
        issued_at=str(publication.get("issued_at") or ""),
    )
    # The retained forecast and causal artifacts are complete before this
    # second full-horizon witness is collected.  It binds the recoverable
    # availability time, rather than treating evaluator completion alone as
    # publication availability.
    ops._verify_prospective_state_sequence(
        checks=(
            intent.get("prospective_preflight") or (),
            intent.get("prospective_calculation_start") or (),
            publication.get("prospective_availability") or (),
        ),
        events=list(range(int(identity["planning_event"]) + 1, int(expiry_event) + 1)),
        cutoff=str(identity["cutoff"]),
        calculation_started_at=calculation_started_at,
        issued_at=available_at,
    )
    valuation_events = [int(value) for value in intent.get("valuation_state_events") or ()]
    if not valuation_events or valuation_events != sorted(set(valuation_events)):
        raise ops.ChipEvidenceError("FH/WC publication omits its exact valuation-state event set")
    ops._verify_prospective_state_sequence(
        checks=(
            intent.get("valuation_preflight") or (),
            intent.get("valuation_calculation_start") or (),
            publication.get("valuation_prospective_issuance") or (),
        ),
        events=valuation_events,
        cutoff=str(identity["cutoff"]),
        calculation_started_at=str(intent.get("calculation_started_at") or ""),
        issued_at=str(publication.get("issued_at") or ""),
    )
    ops._verify_prospective_state_sequence(
        checks=(
            intent.get("valuation_preflight") or (),
            intent.get("valuation_calculation_start") or (),
            publication.get("valuation_prospective_availability") or (),
        ),
        events=valuation_events,
        cutoff=str(identity["cutoff"]),
        calculation_started_at=calculation_started_at,
        issued_at=available_at,
    )
    forecast_ref = str(publication.get("forecast_ref") or "")
    forecast_path, forecast_value = ops._flat_artifact(identity["evidence_root"], forecast_ref)
    if (
        forecast_path.name != forecast_ref
        or crf.canonical_sha256(forecast_value) != str(publication.get("forecast_sha256") or "")
        or str(forecast_value.get("action") or "") != str(action)
        or int(forecast_value.get("expiry_event") or -1) != int(expiry_event)
        or str(forecast_value.get("made_at") or "") != str(publication.get("issued_at") or "")
    ):
        raise ops.ChipEvidenceError("published FH/WC forecast artifact does not match its sealed publication")
    causal_path, causal, causal_forecast_path, _raw_forecast = ops._load_causal_origin(
        identity["evidence_root"], observation_id=observation_id,
        action=action, origin_id=origin_id,
    )
    if (
        causal_path.name != str(publication.get("causal_evidence_ref") or "")
        or cal._canonical_sha256(causal) != str(publication.get("causal_evidence_sha256") or "")
        or causal_forecast_path.name != forecast_ref
        or str(causal.get("forecast", {}).get("artifact_sha256") or "")
        != str(forecast_value.get("artifact_sha256") or "")
    ):
        raise ops.ChipEvidenceError("published FH/WC causal artifact does not bind the issued forecast")
    retained, generation, _decision, route, _state, permission = ops._revalidate_registered_origin(
        conn, identity=identity,
    )
    if permission.get("permitted") is not True or permission.get("reasons") != []:
        raise ops.ChipEvidenceError("published FH/WC forecast origin permission no longer verifies")
    source_identity = ops._source_identity(route, generation)
    value_generation_ids = {
        int(event): str(value)
        for event, value in (intent.get("value_generation_ids") or {}).items()
    }
    coverage_product = crf.build_reservation_coverage_product(
        conn,
        action=action,
        source_identity=source_identity,
        expiry_event=int(expiry_event),
        product_generation_id=str(intent.get("continuation_generation_id") or ""),
        wildcard_value_horizon_length=(
            None if action != "WC" else int(intent.get("coverage_product_horizon_length") or 0)
        ),
    )
    if (
        str(coverage_product.get("product_sha256") or "")
        != str(intent.get("coverage_product_sha256") or "")
        or coverage_product.get("coverage_complete") is not True
        or coverage_product.get("missing_events")
        or [int(value) for value in coverage_product.get("forecast_events") or ()]
        != list(range(int(identity["planning_event"]) + 1, int(expiry_event) + 1))
    ):
        raise ops.ChipEvidenceError("published FH/WC coverage product does not reproduce from its certified generation")
    product_events = {int(value) for value in coverage_product.get("product_events") or ()}
    expected_valuation_set = set(range(int(identity["planning_event"]) + 1, int(expiry_event) + 1))
    for event in coverage_product["forecast_events"]:
        four_event_slice = set(range(int(event), int(event) + 4))
        if not four_event_slice.issubset(product_events):
            raise ops.ChipEvidenceError(f"GW{event} forecast valuation horizon is not covered by its certified continuation")
        expected_valuation_set.update(four_event_slice)
    forecast_events = [int(value) for value in coverage_product["forecast_events"]]
    if action == "WC":
        if set(value_generation_ids) != set(forecast_events) or any(not value_generation_ids[event] for event in forecast_events):
            raise ops.ChipEvidenceError("published WC forecast omits its per-event certified value generations")
        for event in forecast_events:
            value_generation = gs.load_generation(conn, value_generation_ids[event])
            value_report = gs.verify_generation(conn, value_generation.generation_id)
            value_events = tuple(int(value) for value in value_generation.events)
            if (
                value_report.get("verified") is not True
                or value_generation.horizon_kind != gs.HORIZON_KIND_WILDCARD_VALUE
                or int(value_generation.planning_event) != event
                or len(value_events) < 6 or len(value_events) > 10
                or value_events != tuple(range(event, event + len(value_events)))
                or not set(value_events).issubset(product_events)
                or str(value_generation.cutoff) != str(generation.cutoff)
                or str(value_generation.snapshot.get("sha256") or "")
                != str(generation.snapshot.get("sha256") or "")
                or str(value_generation.manifest.get("code_snapshot_sha256") or "")
                != str(generation.manifest.get("code_snapshot_sha256") or "")
            ):
                raise ops.ChipEvidenceError(f"GW{event} WC value generation binding does not verify")
            expected_valuation_set.update(value_events)
    elif value_generation_ids:
        raise ops.ChipEvidenceError("published FH forecast carries unrelated WC value generations")
    expected_valuation_events = sorted(expected_valuation_set)
    if valuation_events != expected_valuation_events:
        raise ops.ChipEvidenceError("published valuation-state event set differs from certified action horizons")
    crf.verify_reservation_forecast(
        forecast_value,
        store_conn=conn,
        expected={
            "action": action,
            "planning_event": int(identity["planning_event"]),
            "origin_cutoff": str(identity["cutoff"]),
            "made_at": str(publication.get("issued_at") or ""),
            "input_as_of": str(identity["cutoff"]),
            "forecast_mode": crf.FORECAST_MODE_PROSPECTIVE,
            "expiry_event": int(expiry_event),
            "source_identity": ops._source_identity(route, generation),
        },
        evidence_verifier=lambda ref: ops._flat_artifact(identity["evidence_root"], ref)[1],
    )
    calculation_ref = str(publication.get("source_forecast_ref") or "")
    calculation_path, calculation_forecast = ops._flat_artifact(identity["evidence_root"], calculation_ref)
    calculation_digest = crf.canonical_sha256(calculation_forecast)
    if (
        calculation_path.name != calculation_ref
        or str(calculation_forecast.get("artifact_sha256") or "")
        != str(publication.get("source_forecast_artifact_identity") or "")
        or calculation_digest != str(publication.get("source_forecast_content_sha256") or "")
        or str(calculation_forecast.get("made_at") or "")
        != str(intent.get("calculation_started_at") or "")
    ):
        raise ops.ChipEvidenceError("calculation forecast does not match its immutable publication source")
    reissued_forecast, reissued_ref, reissued_digest = ops._issued_forecast_artifacts(
        conn,
        evidence_root=identity["evidence_root"],
        source_forecast_ref=calculation_ref,
        calculation_started_at=str(intent.get("calculation_started_at") or ""),
        issued_at=str(publication.get("issued_at") or ""),
    )
    if reissued_ref != forecast_ref or reissued_digest != crf.canonical_sha256(forecast_value) or reissued_forecast != forecast_value:
        raise ops.ChipEvidenceError("published forecast changed policy or score during deterministic issue-time reconstruction")
    source_causality = generation.manifest.get("search_permission_causality") or {}
    observed_at = str(source_causality.get("snapshot_consistency_at") or "")
    snapshot_available_at = str(source_causality.get("snapshot_capture_completed_at") or "")
    if (
        source_causality.get("temporal_status") != "CAUSAL"
        or not observed_at or not snapshot_available_at
        or ops._parse_utc(observed_at, name="origin snapshot observed_at")
        > ops._parse_utc(identity["cutoff"], name="origin cutoff")
    ):
        raise ops.ChipEvidenceError("published FH/WC origin lacks verified snapshot observation/availability")
    ops._parse_utc(snapshot_available_at, name="origin snapshot available_at")
    selected_event = int(forecast_value.get("selected_event") or -1)
    selected = next((
        item.get("artifact_payload") for item in forecast_value.get("opportunities") or ()
        if int((item.get("artifact_payload") or {}).get("event") or -1) == selected_event
    ), {})
    receipt = ops._sealed(ops.FORECAST_RECEIPT_SCHEMA, {
        "operation_id": operation_id,
        "identity_sha256": ops._sha256(identity),
        "origin_id": origin_id,
        "observation_id": str(observation_id),
        "action": action,
        "planning_event": int(identity["planning_event"]),
        "origin_cutoff": str(identity["cutoff"]),
        "input_as_of": str(forecast_value.get("input_as_of") or identity["cutoff"]),
        "observed_at": observed_at,
        "snapshot_available_at": snapshot_available_at,
        "made_at": str(publication.get("issued_at") or ""),
        "expiry_event": int(expiry_event),
        "selected_event": selected_event,
        "forecast_ref": forecast_ref,
        "forecast_sha256": crf.canonical_sha256(forecast_value),
        "causal_evidence_ref": causal_path.name,
        "causal_evidence_sha256": cal._canonical_sha256(causal),
        "world_identity": str(causal.get("source", {}).get("world_identity") or ""),
        "cache": list(publication.get("cache") or []),
        "materialized_worlds": False,
        "coverage_product_sha256": str(intent.get("coverage_product_sha256") or ""),
        "valuation_state_events": valuation_events,
        "valuation_state_evidence_sha256": ops._sha256({
            "preflight": intent.get("valuation_preflight") or [],
            "calculation_start": intent.get("valuation_calculation_start") or [],
            "issuance": publication.get("valuation_prospective_issuance") or [],
            "availability": publication.get("valuation_prospective_availability") or [],
        }),
        "forecast_available_at": available_at,
        "evaluator_identity": dict(selected.get("evaluator_identity") or {}),
        "publication_ref": f"forecast-publication-{operation_id}.json",
        "publication_sha256": ops._sha256(publication),
        "calculation_started_at": str(intent.get("calculation_started_at") or ""),
        "issued_at": str(publication.get("issued_at") or ""),
    })
    return receipt, forecast_value


def _verified_target_chip_available(
    request: Any, *, action: str, event: int, expiry_event: int, window_start_event: int,
) -> bool:
    """Derive the per-event chip flag from the canonical typed request evidence."""
    from . import chip_decision as cd
    from . import chip_evidence_ops as ops

    if action == cd.CHIP_ACTION_FH:
        if getattr(request, "chip_available", None) is not True:
            raise ops.ChipEvidenceError(f"pinned FH is not available for GW{event}")
        return True
    rows = [
        dict(row) for row in (getattr(request, "chip_availability", ()) or ())
        if str(row.get("name") or "").strip().lower()
        == str(cd.CHIP_ACTION_TO_OFFICIAL_NAME[cd.CHIP_ACTION_WC]).lower()
    ]
    eligible = [
        row for row in rows
        if row.get("available_for_event") is True
        and row.get("used") is False
        and row.get("expired") is False
        and int(row.get("window_start_event") or -1) == int(window_start_event)
        and int(row.get("window_stop_event") or -1) == int(expiry_event)
    ]
    if len(rows) != 1 or len(eligible) != 1:
        raise ops.ChipEvidenceError(
            f"canonical target request does not prove one available, unexpired WC at GW{event}"
        )
    return True


def build_future_action_forecast(
    conn: Any,
    *,
    identity: Mapping[str, Any],
    action: str,
    observation_id: str,
    entry_id: int,
    continuation_generation_id: str,
    value_generation_ids: Mapping[int, str] | None,
    expiry_event: int,
    coverage_product: Mapping[str, Any],
    evidence_root: str | Path,
    cache_dir: str | Path,
) -> dict[str, Any]:
    """Score and retain a complete prospective FH/WC opportunity forecast.

    Public composition is always cache-only.  A caller cannot turn this
    operator path into a world builder by passing an opt-in flag.
    """

    from . import chip_reservation_calibration as cal
    from . import chip_reservation_forecast as crf
    from . import chip_route_assembly as cra
    from . import generation_store as gs
    from .chip_horizon_evidence_assembly import (
        build_future_free_hit_request,
        build_future_wildcard_request,
    )

    from . import chip_evidence_ops as ops

    canonical_action, _chip = _action_name(action)
    if not str(observation_id or "").strip():
        raise ops.ChipEvidenceError("an explicit observation id is required")
    expected_identity = ops._required_identity(
        db_path=identity.get("database_path"), entry_id=entry_id,
        generation_id=identity.get("generation_id"), decision_id=identity.get("decision_id"),
        route_id=identity.get("route_id"), planning_event=identity.get("planning_event"),
        cutoff=identity.get("cutoff"), evidence_root=evidence_root,
    )
    if expected_identity != dict(identity):
        raise ops.ChipEvidenceError("forecast identity differs from its explicit database/evidence-root binding")
    root = ops._root(evidence_root)
    cache_path = Path(cache_dir).expanduser().resolve()
    if cache_path in {root.resolve(), ops._metadata_root(evidence_root).resolve()}:
        raise ops.ChipEvidenceError("world cache must be separate from the flat evidence root")

    origin_id = ops._sha256(ops._origin_key(identity))
    operation_id = ops._forecast_operation_id(origin_id, canonical_action, observation_id)
    receipt_path = ops._metadata_root(evidence_root) / f"forecast-{operation_id}.json"
    intent_path = ops._metadata_root(evidence_root) / f"forecast-intent-{operation_id}.json"
    publication_path = ops._metadata_root(evidence_root) / f"forecast-publication-{operation_id}.json"

    with ops._operation_lock(evidence_root, f"forecast-{operation_id}"):
        retained, generation, _decision, route, _state, permission = ops._revalidate_registered_origin(
            conn, identity=identity,
        )
        horizon_origin = _verify_horizon_origin(
            evidence_root, origin_id=origin_id, identity=identity, action=canonical_action,
        )
        if permission.get("permitted") is not True or permission.get("reasons") != []:
            raise ops.ChipEvidenceError("FH/WC origin search permission is not permitted")
        chip_eligibility = dict(horizon_origin.get("chip_eligibility") or {})
        if chip_eligibility.get("eligible_at_origin") is not True:
            raise ops.ChipEvidenceError("pinned origin does not establish this FH/WC chip as eligible")
        pinned_expiry = int(chip_eligibility.get("expiry_event") or -1)
        pinned_window_start = int(chip_eligibility.get("window_start_event") or -1)
        if int(expiry_event) != pinned_expiry or pinned_expiry <= int(identity["planning_event"]):
            raise ops.ChipEvidenceError("requested expiry differs from pinned FH/WC state or has no future opportunity")
        if pinned_window_start <= 0 or pinned_window_start > int(identity["planning_event"]):
            raise ops.ChipEvidenceError("pinned FH/WC origin has an invalid chip-window start")

        source_identity = ops._source_identity(route, generation)
        crf.verify_reservation_coverage_product(
            conn,
            coverage_product,
            expected={
                "action": canonical_action,
                "planning_event": int(identity["planning_event"]),
                "expiry_event": pinned_expiry,
                "source_identity": source_identity,
            },
        )
        forecast_events = [int(value) for value in coverage_product.get("forecast_events") or ()]
        expected_events = list(range(int(identity["planning_event"]) + 1, pinned_expiry + 1))
        if (
            coverage_product.get("coverage_complete") is not True
            or coverage_product.get("missing_events")
            or forecast_events != expected_events
        ):
            raise ops.ChipEvidenceError(
                "same-origin FH/WC coverage must prove every contiguous opportunity through pinned expiry"
            )
        normalized_value_ids = {
            int(event): str(value) for event, value in (value_generation_ids or {}).items()
        }
        if canonical_action == cd.CHIP_ACTION_WC:
            if set(normalized_value_ids) != set(forecast_events) or any(
                not normalized_value_ids[event] for event in forecast_events
            ):
                raise ops.ChipEvidenceError("WC requires one explicit WILDCARD_VALUE generation per forecast event")
        elif normalized_value_ids:
            raise ops.ChipEvidenceError("FH does not accept unrelated WILDCARD_VALUE generations")

        product_events = {int(value) for value in coverage_product.get("product_events") or ()}
        valuation_event_set = set(forecast_events)
        for event in forecast_events:
            four_event_slice = set(range(event, event + 4))
            if not four_event_slice.issubset(product_events):
                raise ops.ChipEvidenceError(
                    f"GW{event} chip route valuation horizon is not wholly covered by its certified continuation"
                )
            valuation_event_set.update(four_event_slice)
            if canonical_action == cd.CHIP_ACTION_WC:
                value_generation = gs.load_generation(conn, normalized_value_ids[event])
                value_report = gs.verify_generation(conn, value_generation.generation_id)
                value_events = tuple(int(value) for value in value_generation.events)
                if (
                    value_report.get("verified") is not True
                    or value_generation.horizon_kind != gs.HORIZON_KIND_WILDCARD_VALUE
                    or int(value_generation.planning_event) != event
                    or len(value_events) < 6 or len(value_events) > 10
                    or value_events != tuple(range(event, event + len(value_events)))
                    or not set(value_events).issubset(product_events)
                    or str(value_generation.cutoff) != str(generation.cutoff)
                    or str(value_generation.snapshot.get("sha256") or "")
                    != str(generation.snapshot.get("sha256") or "")
                    or str(value_generation.manifest.get("code_snapshot_sha256") or "")
                    != str(generation.manifest.get("code_snapshot_sha256") or "")
                ):
                    raise ops.ChipEvidenceError(
                        f"GW{event} WC valuation generation is not a verified same-origin 6–10 event product"
                    )
                valuation_event_set.update(value_events)
        valuation_events = sorted(valuation_event_set)
        if not set(valuation_events).issubset(product_events):
            raise ops.ChipEvidenceError("certified continuation does not cover every FH/WC valuation event")

        expected_intent = {
            "operation_id": operation_id,
            "identity_sha256": ops._sha256(identity),
            "origin_id": origin_id,
            "observation_id": str(observation_id),
            "action": canonical_action,
            "expiry_event": pinned_expiry,
            "continuation_generation_id": str(continuation_generation_id),
            "value_generation_ids": {str(k): normalized_value_ids[k] for k in sorted(normalized_value_ids)},
            "coverage_product_sha256": str(coverage_product.get("product_sha256") or ""),
            "coverage_product_horizon_length": (
                None if canonical_action != cd.CHIP_ACTION_WC
                else int(coverage_product.get("wildcard_value_horizon_length") or 0)
            ),
            "valuation_state_events": valuation_events,
            "coverage_product_horizon_length": (
                None if canonical_action != cd.CHIP_ACTION_WC
                else int(coverage_product.get("wildcard_value_horizon_length") or 0)
            ),
            "cache_dir": str(cache_path),
            "materialized_worlds": False,
        }

        if receipt_path.exists():
            parent = ops._load_forecast_origin(
                conn, identity=identity, action=canonical_action, observation_id=observation_id,
            )[5]
            intent = ops._read_json(intent_path)
            publication = ops._read_json(publication_path)
            ops._verify_forecast_request_binding(intent, publication, expected_intent=expected_intent)
            verified_receipt, verified_forecast = _forecast_receipt_from_publication(
                conn, identity=identity, origin_id=origin_id, operation_id=operation_id,
                observation_id=observation_id, action=canonical_action, expiry_event=pinned_expiry,
                publication=publication, intent=intent,
            )
            if verified_receipt != parent["receipt"] or verified_forecast != parent["forecast"]:
                raise ops.ChipEvidenceError("forecast retry disagrees with its deterministic published receipt")
            return {
                "receipt": verified_receipt,
                "forecast": verified_forecast,
                "assembly_reports": publication.get("assembly_reports") or [],
                "cache_report": publication.get("cache") or [],
                "reused": True,
            }

        if intent_path.exists():
            intent = ops._read_json(intent_path)
            if not publication_path.exists():
                raise ops.ChipEvidenceError(
                    "forecast operation has an immutable intent but no complete publication; "
                    "refusing to replay or guess whether partial artifacts were written"
                )
            publication = ops._read_json(publication_path)
            ops._verify_forecast_request_binding(intent, publication, expected_intent=expected_intent)
            if str(publication.get("intent_sha256") or "") != ops._sha256(intent):
                raise ops.ChipEvidenceError("forecast publication does not bind its immutable intent")
            recovered_receipt, recovered_forecast = _forecast_receipt_from_publication(
                conn, identity=identity, origin_id=origin_id, operation_id=operation_id,
                observation_id=observation_id, action=canonical_action, expiry_event=pinned_expiry,
                publication=publication, intent=intent,
            )
            ops._write_new_json(receipt_path, recovered_receipt)
            return {
                "receipt": ops._read_forecast_receipt(evidence_root, operation_id),
                "forecast": recovered_forecast,
                "assembly_reports": publication.get("assembly_reports") or [],
                "cache_report": publication.get("cache") or [],
                "reused": True,
                "recovered_publication": True,
            }

        forecast_intent = ops._sealed(ops.FORECAST_INTENT_SCHEMA, {
            **expected_intent,
            "prospective_preflight": ops._collect_prospective_state_checks(
                conn, events=forecast_events, cutoff=str(identity["cutoff"]), stage="preflight",
            ),
            "prospective_calculation_start": ops._collect_prospective_state_checks(
                conn, events=forecast_events, cutoff=str(identity["cutoff"]), stage="calculation_start",
            ),
            "valuation_preflight": ops._collect_prospective_state_checks(
                conn, events=valuation_events, cutoff=str(identity["cutoff"]), stage="preflight",
            ),
            "valuation_calculation_start": ops._collect_prospective_state_checks(
                conn, events=valuation_events, cutoff=str(identity["cutoff"]), stage="calculation_start",
            ),
            "calculation_started_at": ops._utc_now(),
        })
        ops._write_new_json(intent_path, forecast_intent)
        calculation_started_at = str(forecast_intent["calculation_started_at"])
        reservation_state = cra._post_h1_save_reservation_state(route)
        decision_artifact = ops._read_json(Path(str(route.source_artifact_ref)))
        season = str((decision_artifact.get("attribution") or {}).get("manager_packet", {}).get("season") or "")
        if not season:
            raise ops.ChipEvidenceError("verified source decision does not retain its season")

        records: list[dict[str, Any]] = []
        assembly_reports: list[dict[str, Any]] = []
        cache_report_by_key: dict[tuple[str, str], dict[str, Any]] = {}
        for event in forecast_events:
            common = {
                "decision_id": str(identity["decision_id"]),
                "generation_id": str(identity["generation_id"]),
                "route_id": str(identity["route_id"]),
                "entry_id": int(entry_id),
                "origin_cutoff": str(identity["cutoff"]),
                "data_snapshot_sha256": str(route.data_snapshot_sha256),
                "target_event": event,
                "season": season,
                "continuation_generation_id": str(continuation_generation_id),
                "cache_dir": cache_path,
            }
            if canonical_action == cd.CHIP_ACTION_FH:
                request, report = build_future_free_hit_request(
                    conn, **common, expiry_event=pinned_expiry, coverage_product=coverage_product,
                )
                if request.permanent.event_start_free_transfers is None:
                    raise ops.ChipEvidenceError("FH permanent state has unknown event-start free transfers")
                event_reservation_state = {
                    "squad_ids": sorted(int(pid) for pid in request.permanent.owned_ids),
                    "bank_tenths": int(request.permanent.bank_tenths),
                    "purchase_price_tenths": {
                        str(int(pid)): int(value)
                        for pid, value in sorted(request.permanent.purchase_price_tenths.items())
                    },
                    "free_transfers": int(request.permanent.free_transfers),
                    "event_start_free_transfers": int(request.permanent.event_start_free_transfers),
                }
                value_generation_id = None
            else:
                value_generation_id = normalized_value_ids[event]
                request, report = build_future_wildcard_request(
                    conn, **common, value_generation_id=value_generation_id,
                    expiry_event=pinned_expiry, coverage_product=coverage_product,
                )
                first_save = next((row for row in request.save_route.events if int(row.event) == event), None)
                if first_save is None:
                    raise ops.ChipEvidenceError(f"WC SAVE route omits target GW{event}")
                save_start_state = report.get("save_route_start_event_state")
                if (
                    type(request.event_start_free_transfers) is not int
                    or not isinstance(save_start_state, Mapping)
                    or int(save_start_state.get("event") or -1) != int(event)
                    or type(save_start_state.get("event_start_free_transfers")) is not int
                    or int(save_start_state["event_start_free_transfers"])
                    != int(request.event_start_free_transfers)
                    or type(first_save.free_transfers) is not int
                ):
                    raise ops.ChipEvidenceError(
                        "WC event-start free transfers are not bound to the canonical SAVE route "
                        f"pre-event state (request={request.event_start_free_transfers!r}, "
                        f"SAVE_GW{event}={save_start_state!r})"
                    )
                event_reservation_state = {
                    "squad_ids": sorted(int(pid) for pid in first_save.squad_ids),
                    "bank_tenths": int(first_save.bank_tenths),
                    "purchase_price_tenths": {
                        str(int(pid)): int(value)
                        for pid, value in sorted(first_save.purchase_price_tenths.items())
                    },
                    "free_transfers": int(first_save.free_transfers),
                    "event_start_free_transfers": int(first_save.free_transfers),
                    "wildcard_available": _verified_target_chip_available(
                        request, action=canonical_action, event=event, expiry_event=pinned_expiry,
                        window_start_event=pinned_window_start,
                    ),
                }
            report = {
                **report,
                "verified_origin_event_start_free_transfers": (
                    request.permanent.event_start_free_transfers
                    if canonical_action == cd.CHIP_ACTION_FH
                    else request.event_start_free_transfers
                ),
            }
            records.append(crf.build_evaluated_event_opportunity_record(
                action=canonical_action,
                event=event,
                source_identity=source_identity,
                reservation_state=event_reservation_state,
                made_at=calculation_started_at,
                input_as_of=str(identity["cutoff"]),
                conn=conn,
                expiry_event=pinned_expiry,
                coverage_product=coverage_product,
                action_request=request,
                chip_generation_id=(str(continuation_generation_id)
                                    if canonical_action == cd.CHIP_ACTION_WC else None),
                value_generation_id=value_generation_id,
            ))
            assembly_reports.append({"event": event, **report})
            for cache_row in report.get("world_cache_report") or ():
                key = (str(cache_row.get("generation_id") or ""), str(cache_row.get("cache_key") or ""))
                cache_report_by_key[key] = dict(cache_row)

        prospective_issuance = ops._collect_prospective_state_checks(
            conn, events=forecast_events, cutoff=str(identity["cutoff"]), stage="issuance",
        )
        valuation_issuance = ops._collect_prospective_state_checks(
            conn, events=valuation_events, cutoff=str(identity["cutoff"]), stage="issuance",
        )
        issued_at = ops._utc_now()
        ops._verify_prospective_state_sequence(
            checks=(
                forecast_intent["prospective_preflight"],
                forecast_intent["prospective_calculation_start"],
                prospective_issuance,
            ),
            events=forecast_events,
            cutoff=str(identity["cutoff"]),
            calculation_started_at=calculation_started_at,
            issued_at=issued_at,
        )
        ops._verify_prospective_state_sequence(
            checks=(
                forecast_intent["valuation_preflight"],
                forecast_intent["valuation_calculation_start"],
                valuation_issuance,
            ),
            events=valuation_events,
            cutoff=str(identity["cutoff"]),
            calculation_started_at=calculation_started_at,
            issued_at=issued_at,
        )

        opportunity_refs: list[str] = []
        for record in records:
            retained_opportunity = crf.retain_event_opportunity_record(record, root, store_conn=conn)
            opportunity_refs.append(Path(str(retained_opportunity["path"])).name)
        calculation_forecast = crf.build_reservation_forecast(
            store_conn=conn,
            action=canonical_action,
            planning_event=int(identity["planning_event"]),
            origin_cutoff=str(identity["cutoff"]),
            made_at=calculation_started_at,
            expiry_event=pinned_expiry,
            source_identity=source_identity,
            reservation_state=reservation_state,
            opportunity_refs=opportunity_refs,
            evidence_verifier=lambda reference: ops._flat_artifact(root, reference)[1],
            input_as_of=str(identity["cutoff"]),
            forecast_mode=crf.FORECAST_MODE_PROSPECTIVE,
            coverage_product=coverage_product,
        )
        retained_calculation = crf.retain_reservation_forecast(
            calculation_forecast, root, store_conn=conn,
        )
        issued_forecast, forecast_ref, forecast_digest = ops._issued_forecast_artifacts(
            conn,
            evidence_root=evidence_root,
            source_forecast_ref=Path(str(retained_calculation["path"])).name,
            calculation_started_at=calculation_started_at,
            issued_at=issued_at,
        )
        causal_origin = cal.retain_causal_origin_observation(
            observation_id=str(observation_id),
            forecast_artifact=issued_forecast,
            evidence_root=evidence_root,
            store_conn=conn,
            evidence_verifier=lambda reference: ops._flat_artifact(evidence_root, reference)[1],
        )
        causal = causal_origin["causal_evidence"]
        causal_ref = str(causal_origin["causal_evidence_ref"])
        causal_digest = cal._canonical_sha256(causal)
        generation_causality = generation.manifest.get("search_permission_causality") or {}
        observed_at = str(generation_causality.get("snapshot_consistency_at") or "")
        snapshot_available_at = str(generation_causality.get("snapshot_capture_completed_at") or "")
        if (
            generation_causality.get("temporal_status") != "CAUSAL"
            or not observed_at or not snapshot_available_at
            or ops._parse_utc(observed_at, name="origin snapshot observed_at")
            > ops._parse_utc(identity["cutoff"], name="origin cutoff")
        ):
            raise ops.ChipEvidenceError("pinned generation lacks verified causal snapshot observation/availability times")
        ops._parse_utc(snapshot_available_at, name="origin snapshot available_at")
        prospective_availability = ops._collect_prospective_state_checks(
            conn, events=forecast_events, cutoff=str(identity["cutoff"]), stage="issuance",
        )
        valuation_prospective_availability = ops._collect_prospective_state_checks(
            conn, events=valuation_events, cutoff=str(identity["cutoff"]), stage="issuance",
        )
        selected_event = int(issued_forecast.get("selected_event") or -1)
        selected_opportunity = next((
            item.get("artifact_payload") for item in issued_forecast.get("opportunities") or ()
            if int((item.get("artifact_payload") or {}).get("event") or -1) == selected_event
        ), {})
        cache_report = [cache_report_by_key[key] for key in sorted(cache_report_by_key)]
        publication = ops._sealed(ops.FORECAST_PUBLICATION_SCHEMA, {
            **expected_intent,
            "intent_sha256": ops._sha256(forecast_intent),
            "calculation_started_at": calculation_started_at,
            "issued_at": issued_at,
            "forecast_available_at": ops._utc_now(),
            "prospective_issuance": prospective_issuance,
            "prospective_availability": prospective_availability,
            "valuation_prospective_issuance": valuation_issuance,
            "valuation_prospective_availability": valuation_prospective_availability,
            "source_forecast_ref": Path(str(retained_calculation["path"])).name,
            "source_forecast_artifact_identity": str(calculation_forecast.get("artifact_sha256") or ""),
            "source_forecast_content_sha256": crf.canonical_sha256(calculation_forecast),
            "forecast_ref": forecast_ref,
            "forecast_sha256": forecast_digest,
            "causal_evidence_ref": causal_ref,
            "causal_evidence_sha256": causal_digest,
            "cache": cache_report,
            "assembly_reports": assembly_reports,
            "materialized_worlds": False,
        })
        ops._write_new_json(publication_path, publication)
        receipt, verified_issued_forecast = _forecast_receipt_from_publication(
            conn, identity=identity, origin_id=origin_id, operation_id=operation_id,
            observation_id=observation_id, action=canonical_action, expiry_event=pinned_expiry,
            publication=publication, intent=forecast_intent,
        )
        if verified_issued_forecast != issued_forecast:
            raise ops.ChipEvidenceError("issued FH/WC forecast differs from the completed publication artifacts")
        ops._write_new_json(receipt_path, receipt)
        verified_receipt = ops._read_forecast_receipt(evidence_root, operation_id)
        return {
            "receipt": verified_receipt,
            "forecast": issued_forecast,
            "assembly_reports": assembly_reports,
            "cache_report": cache_report,
            "reused": False,
        }
