"""Canonical rolling four-Gameweek decision layer (normal transfer decisions).

PRODUCT RULE encoded here, once, so it cannot drift:

* A **normal transfer decision** is always planned over a rolling four-event
  horizon — the current Gameweek plus the next three: ``[N, N+1, N+2, N+3]``.
* A **lineup / bench / captain / vice** decision stays a current-GW (H1)
  decision.
* A current-GW-only result must NEVER be presented as the best normal transfer
  recommendation.  If any of the four decision events lacks coherent accepted
  predictive support **under the same planning cutoff**, the transfer
  recommendation is suppressed and the status is ``DECISION_HORIZON_INCOMPLETE``.

This module is decision/orchestration only.  It does not touch Minutes, Team
Strength, Player Rates, xPts, Monte Carlo, calibration, gates, RNG, or any
accepted Phase-5 predictive logic, and it never simulates or searches by
itself: callers pass in already-computed route records or prebuilt worlds.
"""

from __future__ import annotations

import math
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import certified_bundle as cb
from .season_rules import SeasonRules, free_transfers_after_chip, wildcard_gameweek_hit
from .utils import parse_utc

# ---------------------------------------------------------------------------
# Canonical horizon
# ---------------------------------------------------------------------------

DECISION_HORIZON_LENGTH = 4
"""A normal transfer decision spans the current GW plus the next three."""

LINEUP_HORIZON_LENGTH = 1
"""Lineup / bench / captain / vice remain current-GW (H1) decisions."""

SEASON_LAST_EVENT = 38
"""Official FPL season length.  The event schedule is the authority for which
Gameweeks exist; a horizon is never padded with non-existent Gameweeks."""

# --- four-GW fixture-horizon classification (R4B.2c) -----------------------
#
# The transfer horizon is exactly the FOUR consecutive certified decision events.
# A generic horizon (e.g. 8) is deliberately NOT used.  All fixture evidence comes
# from the immutable certification snapshot, never live fixture state.
OUTSIDE_HORIZON = "OUTSIDE_HORIZON"
PLAUSIBLY_INSIDE = "PLAUSIBLY_INSIDE"
UNKNOWN_HORIZON = "UNKNOWN_HORIZON"

CERTIFIED_BLANK = "CERTIFIED_BLANK"
CERTIFIED_SINGLE = "CERTIFIED_SINGLE"
CERTIFIED_DOUBLE_OR_MORE = "CERTIFIED_DOUBLE_OR_MORE"

FIXTURE_INTEGRITY_DUPLICATE_IDENTITY = "FIXTURE_DUPLICATE_IDENTITY"

#: Which classes block a normal four-GW transfer recommendation.
BLOCKING_FIXTURE_CLASSES = (PLAUSIBLY_INSIDE, UNKNOWN_HORIZON)

DECISION_HORIZON_COMPLETE = "DECISION_HORIZON_COMPLETE"
DECISION_HORIZON_INCOMPLETE = "DECISION_HORIZON_INCOMPLETE"
SEASON_END_SHORT_HORIZON = "SEASON_END_SHORT_HORIZON"
"""Fewer than four real Gameweeks remain.  The horizon uses every remaining
event and is NOT 'incomplete' merely because GW39+ do not exist."""

RECOMMENDATION_SUPPRESSED = "TRANSFER_RECOMMENDATION_SUPPRESSED_HORIZON_INCOMPLETE"
RECOMMENDATION_SUPPRESSED_NO_VALID_ROUTES = "TRANSFER_RECOMMENDATION_SUPPRESSED_NO_VALID_ROUTES"
RECOMMENDATION_AVAILABLE = "TRANSFER_RECOMMENDATION_AVAILABLE"
LINEUP_ONLY = "LINEUP_ONLY_CURRENT_GW"
LINEUP_UNSUPPORTED = "LINEUP_UNSUPPORTED"

#: R4B.2b: the bounded search could not separate the preferred route from its
#: alternatives at the widest supported budget.  The route table stays available,
#: but no decisive normal-transfer recommendation is emitted (and there is
#: deliberately NO best-current-GW-transfer fallback).
DECISION_SEARCH_NOT_STABLE = "DECISION_SEARCH_NOT_STABLE_AT_CURRENT_BUDGET"

#: Predictive families that must all be present and complete for one event to
#: count as supported.  Names are the accepted ``projection_runs.model_family``
#: values; nothing here changes what those models compute.
REQUIRED_HORIZON_FAMILIES: tuple[str, ...] = (
    "minutes_v1",
    "team_strength_v1",
    "player_rates_v1",
    "xpts_v1",
    "monte_carlo_v1",
)

HORIZON_FLAGS = (
    "SAME_CUTOFF_REQUIRED_ACROSS_HORIZON",
    "NO_STALE_FUTURE_PREDICTIONS",
    "FOUR_GW_NET_CORE_PRIMARY_TRANSFER_OBJECTIVE",
    "LINEUP_REMAINS_CURRENT_GW",
)

WILDCARD_EVALUATION_SUPPORTED = "WILDCARD_EVALUATION_SUPPORTED"


def sanitize_wildcard_screen(wildcard: Mapping[str, Any]) -> dict[str, Any]:
    """Strip any unverifiable actionable Wildcard claim from a passthrough block.

    ``evaluate_four_gw_decision`` must never echo a caller-supplied mapping
    straight into a production artifact: a hand-built
    ``{"recommendation": "PLAY_WILDCARD"}`` would otherwise appear to be a
    recommendation.  The block is preserved for auditability, but any claim that
    is not backed by a VERIFIED evaluation is demoted to review-only.
    """

    block = dict(wildcard)
    verified, reasons = wildcard_evaluation_is_supported(block.get("supported_evaluation"))
    if verified:
        return block
    stripped = False
    if str(block.get("recommendation") or "").upper() in {"PLAY_WILDCARD", "DO_NOT_PLAY_WILDCARD"}:
        block["recommendation"] = "NONE"
        stripped = True
    if block.get("actionable") is True:
        block["actionable"] = False
        stripped = True
    if block.get("status") == WILDCARD_EVALUATION_SUPPORTED:
        block["status"] = WILDCARD_REVIEW_REQUIRED
        stripped = True
    block["wildcard_quantitative_capability"] = WILDCARD_QUANTITATIVE_CAPABILITY
    block["overstatement_prevented"] = True
    block["overstatement_reasons"] = [
        WILDCARD_INJECTION_REJECTED,
        *[str(reason) for reason in reasons],
        *(["stripped prior actionable claim"] if stripped else []),
    ]
    return block

#: Quantitative four-GW Wildcard optimisation does NOT exist.  Until it does, no
#: production path may emit an actionable Wildcard recommendation, and a
#: caller-supplied mapping can never stand in for a real evaluator.
WILDCARD_QUANTITATIVE_CAPABILITY = "NOT_SUPPORTED"
SUPPORTED_WILDCARD_EVALUATION_SCHEMA = "fpl_brain.wildcard_squad_evaluation.v1"

#: The Wildcard play rule is NOT calibrated, so the screen never emits an
#: executable recommendation: a verified evaluation is reported as a signal and
#: the chip stays review-only until a point-in-time calibration exists.
WILDCARD_CALIBRATION_STATUS = "UNCALIBRATED_PROVISIONAL"
WILDCARD_INJECTION_REJECTED = "WILDCARD_UNVERIFIED_EVALUATION_REJECTED"


def wildcard_evaluation_is_supported(
    supported_evaluation: Mapping[str, Any] | None,
) -> tuple[bool, list[str]]:
    """Verify a caller-supplied Wildcard evaluation.  Returns (verified, reasons).

    A supported evaluation must be a real, separately-computed four-GW Wildcard
    squad evaluation: it must carry the recognised schema, be tied to the same
    certified generation (certification identity + data snapshot), and prove it
    evaluated a squad DISTINCT from the current one.  Anything else — including a
    bare ``{"four_gw_net_core": 12.0}`` — is rejected, so ``PLAY_WILDCARD``
    cannot be manufactured by injecting a mapping.
    """

    if supported_evaluation is None:
        return False, ["no supported_evaluation supplied"]
    if not isinstance(supported_evaluation, Mapping):
        return False, [f"supported_evaluation is {type(supported_evaluation).__name__}, not a mapping"]
    reasons: list[str] = []
    schema = supported_evaluation.get("schema")
    if schema != SUPPORTED_WILDCARD_EVALUATION_SCHEMA:
        reasons.append(f"schema {schema!r} != {SUPPORTED_WILDCARD_EVALUATION_SCHEMA!r}")
    if not supported_evaluation.get("four_gw_certification_identity"):
        reasons.append("no four_gw_certification_identity")
    if not supported_evaluation.get("data_snapshot_sha256"):
        reasons.append("no data_snapshot_sha256")
    if supported_evaluation.get("distinct_squad_from_current") is not True:
        reasons.append("not proven a squad distinct from the current squad")
    if not supported_evaluation.get("wildcard_squad_player_ids"):
        reasons.append("no evaluated wildcard squad")
    return (not reasons), reasons
WILDCARD_REVIEW_REQUIRED = "WILDCARD_REVIEW_REQUIRED"
WILDCARD_NOT_COMPETITIVE = "WILDCARD_NOT_COMPETITIVE"

#: Official 2026/27 rule (verified 2026-09-12): a Wildcard makes ALL transfers
#: in that Gameweek free — including transfers already made before activating
#: it — so no transfer deduction remains for the Gameweek.  The rules engine
#: now encodes this (`season_rules.wildcard_gameweek_hit`), so the interaction
#: is VERIFIED rather than flagged unverified.
WILDCARD_HIT_INTERACTION_FLAG = "WILDCARD_SAME_GW_HIT_INTERACTION_VERIFIED_2026_27"
WILDCARD_NOT_MODELLED_FLAG = "WILDCARD_FULL_OPTIMIZER_NOT_IMPLEMENTED"
WILDCARD_REQUIRES_SEPARATE_EVALUATION = "WILDCARD_RECOMMENDATION_REQUIRES_SEPARATE_FOUR_GW_EVALUATION"
WILDCARD_HIT_RULE_TEXT = (
    "Wildcard covers every transfer in the Gameweek, including transfers made "
    "before the Wildcard was activated; saved free transfers are retained; one "
    "chip per Gameweek; Wildcard transfers are permanent and cannot be cancelled "
    "once confirmed."
)


class HorizonError(ValueError):
    """The requested decision horizon is malformed."""


def decision_events(
    planning_event: int,
    *,
    length: int = DECISION_HORIZON_LENGTH,
    last_event: int = SEASON_LAST_EVENT,
) -> tuple[int, ...]:
    """The canonical rolling window ``[N, N+1, ...]`` over REAL Gameweeks only.

    The window is capped at ``last_event`` (the season's final Gameweek), so
    GW38 yields ``(38,)`` rather than the non-existent ``(38, 39, 40, 41)``.
    Every consumer derives the window from here; no caller should write a
    literal ``4``.
    """

    if isinstance(planning_event, bool) or not isinstance(planning_event, int) or planning_event < 1:
        raise HorizonError("planning_event must be a positive integer")
    if isinstance(last_event, bool) or not isinstance(last_event, int) or last_event < 1:
        raise HorizonError("last_event must be a positive integer")
    if isinstance(length, bool) or not isinstance(length, int) or length < 1:
        raise HorizonError("horizon length must be a positive integer")
    if int(planning_event) > int(last_event):
        raise HorizonError(
            f"planning_event {int(planning_event)} is beyond the season's last event {int(last_event)}"
        )
    stop = min(int(planning_event) + int(length) - 1, int(last_event))
    return tuple(range(int(planning_event), stop + 1))


def lineup_events(planning_event: int, *, last_event: int = SEASON_LAST_EVENT) -> tuple[int, ...]:
    """Current-GW window for lineup / bench / captain / vice."""

    return decision_events(planning_event, length=LINEUP_HORIZON_LENGTH, last_event=last_event)


def season_last_event_from_db(conn: sqlite3.Connection) -> int:
    """Last Gameweek defined by the stored official schedule.

    The schedule is the authority for which Gameweeks exist; the constant
    ``SEASON_LAST_EVENT`` is only the fallback when the schedule is not loaded.
    """

    try:
        row = conn.execute("SELECT MAX(id) FROM events").fetchone()
    except sqlite3.Error:
        return SEASON_LAST_EVENT
    value = int(row[0]) if row is not None and row[0] is not None else 0
    return value if value >= 1 else SEASON_LAST_EVENT


# ---------------------------------------------------------------------------
# Cutoff / override safety
# ---------------------------------------------------------------------------

CUTOFF_PRECEDES_OVERRIDE = "PLANNING_CUTOFF_PRECEDES_MANAGER_OVERRIDE"


class CutoffOverrideError(ValueError):
    """The planning cutoff does not cover the authoritative manager override."""


def verify_cutoff_covers_override(
    *,
    planning_cutoff: str | None,
    override_captured_at: str | None,
) -> dict[str, Any]:
    """A pre-deadline planning run must not use a cutoff older than the override.

    ``manager_state_as_of(old_cutoff)`` would correctly resolve the OLDER
    observation (e.g. 1 FT / £0.3m before the confirmed moves).  A production
    run therefore has to establish a new cutoff at or after the override.
    """

    if override_captured_at is None:
        return {
            "status": "NO_OVERRIDE_RECORDED",
            "pass": True,
            "planning_cutoff": planning_cutoff,
            "override_captured_at": None,
            "detail": "no explicit manager-state override to cover",
        }
    if planning_cutoff is None:
        return {
            "status": "PLANNING_CUTOFF_MISSING",
            "pass": False,
            "planning_cutoff": None,
            "override_captured_at": str(override_captured_at),
            "detail": "the planning cutoff must be explicit for a pre-deadline run",
        }
    planned = parse_utc(str(planning_cutoff))
    override = parse_utc(str(override_captured_at))
    if planned is None or override is None:
        return {
            "status": "CUTOFF_UNPARSEABLE",
            "pass": False,
            "planning_cutoff": str(planning_cutoff),
            "override_captured_at": str(override_captured_at),
            "detail": "cutoff timestamps must be ISO-8601 UTC values",
        }
    ok = planned >= override
    return {
        "status": "PASS" if ok else CUTOFF_PRECEDES_OVERRIDE,
        "pass": bool(ok),
        "planning_cutoff": str(planning_cutoff),
        "override_captured_at": str(override_captured_at),
        "delta_seconds": (planned - override).total_seconds(),
        "detail": (
            "planning cutoff covers the manager-state override"
            if ok else
            "the planning cutoff predates the authoritative manager-state override; "
            "re-run with a new cutoff at or after the override or the planning state will "
            "resolve the older observation"
        ),
    }


def assert_cutoff_covers_override(
    *,
    planning_cutoff: str | None,
    override_captured_at: str | None,
) -> dict[str, Any]:
    """Fail fast when the planning cutoff predates the manager-state override."""

    result = verify_cutoff_covers_override(
        planning_cutoff=planning_cutoff, override_captured_at=override_captured_at
    )
    if not result["pass"]:
        raise CutoffOverrideError(f"{result['status']}: {result['detail']}")
    return result


# ---------------------------------------------------------------------------
# Horizon readiness gate
# ---------------------------------------------------------------------------


DIAG_DECISION_CERTIFICATION_REQUIRED = "DECISION_CERTIFICATION_REQUIRED"
DIAG_DECISION_SOURCE_SNAPSHOT_REQUIRED = "DECISION_SOURCE_SNAPSHOT_REQUIRED"


def open_certification_source(certification: Mapping[str, Any]) -> sqlite3.Connection:
    """Open the certification's immutable snapshot READ-ONLY for causal source reads.

    A certification artifact alone does not authorise reading causal source data
    from the live database.  This performs the retention checks -- the snapshot must
    exist, hash to the certified identity, match the certified source-database
    identity, and open read-only -- and fails closed with
    ``DECISION_SOURCE_SNAPSHOT_REQUIRED`` otherwise.  The snapshot must therefore
    survive until the decision lifecycle completes.
    """

    from . import execution_snapshot as es

    path = certification.get("data_snapshot_path")
    expected_sha = certification.get("data_snapshot_sha256")
    if not path or not expected_sha:
        raise DecisionCertificationRequired(
            f"{DIAG_DECISION_SOURCE_SNAPSHOT_REQUIRED}: certification carries no usable data snapshot"
        )
    if not Path(path).exists():
        raise DecisionCertificationRequired(
            f"{es.DIAG_SNAPSHOT_MISSING}: certification snapshot {path} is missing; it must be "
            "retained until the decision lifecycle completes"
        )
    current = es.file_sha256(path)
    if current != str(expected_sha):
        raise DecisionCertificationRequired(
            f"{es.DIAG_SNAPSHOT_MUTATED}: certification snapshot hash {current} != certified "
            f"{expected_sha}"
        )
    expected_identity = certification.get("data_snapshot_source_db_identity") or {}
    if expected_identity:
        live = es.source_db_identity(expected_identity.get("path") or path)
        if expected_identity.get("schema_version") not in (None, live.get("schema_version")):
            raise DecisionCertificationRequired(
                "certification snapshot source identity does not match the expected certification"
            )
    try:
        installed = es.ExecutionSnapshot(
            path=str(path),
            data_snapshot_sha256=str(expected_sha),
            snapshot_lock_acquired_at=str(certification.get("snapshot_lock_acquired_at") or ""),
            snapshot_capture_started_at=str(certification.get("snapshot_capture_started_at") or ""),
            snapshot_consistency_at=str(certification.get("snapshot_consistency_at") or ""),
            snapshot_capture_completed_at=str(certification.get("snapshot_capture_completed_at") or ""),
            snapshot_capture_seconds=float(certification.get("snapshot_capture_seconds") or 0.0),
            source_db_identity=dict(expected_identity),
            execution_run_uuid=certification.get("execution_run_uuid"),
            planning_cutoff=certification.get("planning_cutoff"),
            size_bytes=int(Path(path).stat().st_size),
        )
    except TypeError:
        installed = None
    if installed is not None:
        return es.open_snapshot(installed)
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    return conn



class DecisionCertificationRequired(RuntimeError):
    """A production decision was attempted without a valid certification artifact."""

    def __init__(self, detail: str) -> None:
        super().__init__(f"{DIAG_DECISION_CERTIFICATION_REQUIRED}: {detail}")
        self.detail = detail


# --- certification artifact contract ---------------------------------------
# v1 is the LEGACY contract: it predates the certified-history repair and may
# legitimately omit ``history_completeness``.  It is grandfathered by an EXPLICIT,
# auditable condition (its declared schema), never by "the field is missing".
# v2 is the contract for every certification minted from the history-completeness
# repair onwards and REQUIRES the audit, so the new gate can never be omitted
# silently by an older entry point.
CERTIFICATION_ARTIFACT_SCHEMA_V1 = "fpl_brain.certification_artifact.v1"
CERTIFICATION_ARTIFACT_SCHEMA_V2 = "fpl_brain.certification_artifact.v2"
CERTIFICATION_ARTIFACT_SCHEMA = CERTIFICATION_ARTIFACT_SCHEMA_V2
LEGACY_CERTIFICATION_ARTIFACT_SCHEMAS = (CERTIFICATION_ARTIFACT_SCHEMA_V1,)
SUPPORTED_CERTIFICATION_ARTIFACT_SCHEMAS = (CERTIFICATION_ARTIFACT_SCHEMA_V1, CERTIFICATION_ARTIFACT_SCHEMA_V2)
# Schemas that must carry the history audit; anything newer inherits the rule.
HISTORY_COMPLETENESS_REQUIRED_SCHEMAS = (CERTIFICATION_ARTIFACT_SCHEMA_V2,)

# An OLD producer copy still declares v1, so the schema ALONE cannot be the legacy
# test -- accepting v1 by schema would let a stale certifier mint an audit-free
# artifact that the current consumer accepts.  Legacy compatibility is therefore
# granted only to explicitly recognised historical certification identities
# (four_gw_certification_identity), not to whatever merely looks old.
LEGACY_CERTIFICATION_IDENTITIES = (
    # Accepted R5 GW5-GW8 certification, planning cutoff 2026-09-14T08:14:22Z.
    # Minted before the history-completeness repair; audited read-only and still
    # valid at its own cutoff (GW3 required, GW4 correctly not required).
    "sha256:264cd90e56a3e85fc03ec10324688acba06b3d4903ed96dc90ca1d73685a8bb5",
)

# The certification entry point whose wiring contains the history-completeness
# gate.  It is part of SOURCE_SNAPSHOT_FILES, and every v2 artifact must declare
# it as covered, so a consumer can prove the gate was in the producing wiring.
CERTIFIER_ENTRY_POINT = "scripts/certify_gw5_gw8.py"

DIAG_CERTIFIED_HISTORY_COMPLETENESS_EVIDENCE_MISSING = "CERTIFIED_HISTORY_COMPLETENESS_EVIDENCE_MISSING"
DIAG_CERTIFICATION_WIRING_IDENTITY_MISSING = "CERTIFICATION_WIRING_IDENTITY_MISSING"
DIAG_LEGACY_CERTIFICATION_IDENTITY_UNRECOGNISED = "LEGACY_CERTIFICATION_IDENTITY_UNRECOGNISED"
DIAG_CERTIFICATION_BUNDLE_IDENTITY_MISMATCH = "CERTIFICATION_BUNDLE_IDENTITY_MISMATCH"
DIAG_CERTIFICATION_EVENT_SET_MISMATCH = "CERTIFICATION_EVENT_SET_MISMATCH"
# PE-9: an ABSENT artifact and a CONTRADICTORY one are different facts, and the
# tokens that say so are declared once, in the certification module.
DIAG_CERTIFICATION_ARTIFACT_ABSENT = cb.DIAG_CERTIFICATION_ARTIFACT_ABSENT
DIAG_CERTIFICATION_ARTIFACT_CONTRADICTORY = cb.DIAG_CERTIFICATION_ARTIFACT_CONTRADICTORY


def certification_artifact_requires_history_completeness(schema: Any) -> bool:
    """True when this schema must carry a complete history audit."""

    return str(schema) in set(HISTORY_COMPLETENESS_REQUIRED_SCHEMAS)


def certification_wiring_identity(root: str | Path | None = None) -> dict[str, Any]:
    """The code-identity block every v2 certification must carry.

    One definition, used by the certifier that writes the block and by the
    contracts that assert it: the entry point whose wiring contains the
    history-completeness gate, its exact bytes, and the file set the composite
    ``code_snapshot_sha256`` covered.
    """

    import hashlib
    from pathlib import Path as _Path

    from . import analytics

    base = _Path(root) if root is not None else _Path(__file__).resolve().parents[1]
    return {
        "entry_point": CERTIFIER_ENTRY_POINT,
        "entry_point_sha256": hashlib.sha256((base / CERTIFIER_ENTRY_POINT).read_bytes()).hexdigest(),
        "covered_source_files": list(analytics.SOURCE_SNAPSHOT_FILES),
    }


def _assert_bundle_identities_bind(payload: Mapping[str, Any]) -> None:
    """Prove the declared bundle identities describe the bundles actually consumed.

    ``event_support_from_certification`` reads ``certified_bundles``; the
    certification identity hashes the declared ``certified_bundle_identity``.
    Unless the two correspond, an artifact could keep a recognised label while
    swapping the bundles underneath it.  The canonical identity is recomputed with
    the SAME algorithm the producer uses (``certified_bundle.canonical_bundle_identity``),
    so this holds for legacy and current artifacts alike and no shape is special-cased.
    """

    from . import certified_bundle as cb

    declared = payload.get("certified_bundle_identity") or {}
    bundles = payload.get("certified_bundles") or {}
    mismatches: list[str] = []
    # The declared horizon must BE the certified bundle set, so an altered `events`
    # list cannot silently re-point the decision at a horizon the artifact does
    # not certify.  (This binds events without redefining the certification
    # identity, which would invalidate the recognised legacy value.)
    declared_events = sorted(int(event) for event in (payload.get("events") or []))
    bundle_events = sorted(int(event) for event in bundles)
    if declared_events != bundle_events:
        mismatches.append(
            f"declared events {declared_events} do not match the certified bundle events {bundle_events}"
        )
    for event, bundle in bundles.items():
        if not isinstance(bundle, Mapping):
            mismatches.append(f"event {event}: bundle is not an object")
            continue
        try:
            recomputed = cb.canonical_bundle_identity(bundle)
        except (KeyError, TypeError, ValueError) as failure:
            mismatches.append(f"event {event}: bundle is malformed ({failure})")
            continue
        if str(declared.get(str(event))) != recomputed:
            mismatches.append(
                f"event {event}: declared {str(declared.get(str(event)))[:23]}... does not bind the "
                f"bundles consumed (recomputed {recomputed[:23]}...)"
            )
    if mismatches:
        raise DecisionCertificationRequired(
            f"{DIAG_CERTIFICATION_BUNDLE_IDENTITY_MISMATCH}: the certification's declared bundle "
            "identities do not bind the certified bundles a decision would consume: "
            + "; ".join(mismatches)
        )


def certification_identity_of(payload: Mapping[str, Any]) -> str:
    """Recompute an artifact's own certification identity from its own fields.

    ``four_gw_certification_identity`` is a pure function of the cutoff, the
    per-event bundle identities and the data snapshot identity, so a consumer can
    verify it without any code-drift assumption.  That is what makes the legacy
    allowlist an identity test rather than a forgeable label: copying a recognised
    identity onto a different artifact fails this check.

    The fields are read through ``certified_bundle.plain_certification_value`` so the
    SAME bytes are hashed whether the caller holds the parsed JSON or the read-only
    snapshot a validated artifact exposes: the fingerprint is a function of the
    artifact's content, never of the container it arrived in.
    """

    import hashlib
    import json as _json

    from . import certified_bundle as cb

    fingerprint = cb.plain_certification_value(
        {
            "cutoff": payload.get("planning_cutoff"),
            "bundles": payload.get("certified_bundle_identity") or {},
            "data_snapshot_sha256": payload.get("data_snapshot_sha256"),
        }
    )
    return "sha256:" + hashlib.sha256(
        _json.dumps(fingerprint, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def validate_certification_artifact(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """The COMPLETE authorization contract, applied to an artifact mapping.

    Fails closed with ``DECISION_CERTIFICATION_REQUIRED`` when the mapping is
    malformed or does not carry the fields a decision needs.  There is no fallback
    to latest-per-family discovery anywhere in this path.

    This is the contract a LOAD consumes, so it is applied to the mapping form here
    as well as to the file form by :func:`load_certification_artifact`: a caller that
    presents an artifact-shaped mapping instead of a loaded artifact has exactly the
    same contract re-run over it, and one that cannot pass it is never an
    authorisation -- however self-consistent its own run ids and bundle identities
    are.

    The certification contract is versioned.  ``v2`` requires the
    history-completeness audit and the certification-wiring identity, so a
    certification minted without the gate cannot pass as current; ``v1`` is the
    legacy contract and may omit the audit.  Absence is grandfathered ONLY by the
    artifact's own declared legacy schema -- never by a missing field, and never by
    inferring PASS.
    """

    if not isinstance(payload, Mapping):
        raise DecisionCertificationRequired(
            f"the certification artifact is {type(payload).__name__}, not an object"
        )
    if payload.get("schema") not in SUPPORTED_CERTIFICATION_ARTIFACT_SCHEMAS:
        raise DecisionCertificationRequired(
            f"artifact schema {payload.get('schema')!r} is not one of "
            f"{list(SUPPORTED_CERTIFICATION_ARTIFACT_SCHEMAS)!r}"
        )
    if str(payload.get("temporal_status") or "").upper() != "CAUSAL":
        raise DecisionCertificationRequired(
            f"certification temporal_status is {payload.get('temporal_status')!r}, not 'CAUSAL'"
        )
    if str(payload.get("dependency_validation") or "").upper() != "COHERENT":
        from . import certified_bundle as cb

        raise cb.BundleIncoherent([str(payload.get("dependency_validation_detail") or
                                       "artifact is not marked COHERENT")])
    if not payload.get("certified_bundles"):
        raise DecisionCertificationRequired("certification artifact carries no certified bundles")
    if not payload.get("data_snapshot_sha256"):
        raise DecisionCertificationRequired("certification artifact carries no data snapshot identity")
    # The authorisation flag is the gate.  A factually-recorded "search not executed"
    # field must never be mistaken for permission to search.
    if payload.get("decision_search_permitted") is not True:
        reasons = payload.get("decision_search_permitted_reasons") or ["flag not set"]
        raise DecisionCertificationRequired(
            "decision_search_permitted is not true: " + "; ".join(str(r) for r in reasons)
        )
    # Defence in depth.  A certification minted after the historical-input repair
    # carries the completeness audit itself; when that audit says the required
    # completed-event history is not there, the artifact is not actionable even if
    # the permission flag were somehow stale.
    completeness = payload.get("history_completeness")
    if isinstance(completeness, Mapping) and completeness.get("complete") is not True:
        from . import history_completeness as hc

        raise DecisionCertificationRequired(
            f"{hc.blocking_reason_token(completeness)}: the certification's completed-event history "
            f"audit is not complete ({completeness.get('reasons')})"
        )
    # A NEW certification must not be able to omit the audit.  Absence is only
    # grandfathered for an EXPLICITLY RECOGNISED historical certification
    # identity -- never for a mere schema label, an old-looking cutoff, or the
    # simple fact that the field is missing, and never inferred as PASS.
    # The declared per-event bundle identities must BIND the bundles a decision
    # actually consumes.  ``certified_bundle_identity`` is the label the
    # certification identity hashes, while ``certified_bundles`` is what
    # ``event_support_from_certification`` reads, so without this check a payload
    # could keep the allowlisted label and swap the bundles underneath it.
    _assert_bundle_identities_bind(payload)
    schema = payload.get("schema")
    completeness = payload.get("history_completeness")
    if certification_artifact_requires_history_completeness(schema):
        # v2: the audit and the wiring identity are mandatory.
        if not isinstance(completeness, Mapping):
            raise DecisionCertificationRequired(
                f"{DIAG_CERTIFIED_HISTORY_COMPLETENESS_EVIDENCE_MISSING}: certification schema "
                f"{schema!r} requires a history_completeness audit and none is present; "
                "a missing audit is never inferred as PASS"
            )
        wiring = payload.get("certification_wiring")
        covered = list(wiring.get("covered_source_files") or []) if isinstance(wiring, Mapping) else []
        if not isinstance(wiring, Mapping) or CERTIFIER_ENTRY_POINT not in covered:
            raise DecisionCertificationRequired(
                f"{DIAG_CERTIFICATION_WIRING_IDENTITY_MISSING}: certification schema {schema!r} must "
                f"declare {CERTIFIER_ENTRY_POINT!r} among its covered source files, so the consumer can "
                "prove the producing wiring carried the history-completeness gate"
            )
        if not wiring.get("entry_point_sha256"):
            raise DecisionCertificationRequired(
                f"{DIAG_CERTIFICATION_WIRING_IDENTITY_MISSING}: certification schema {schema!r} carries "
                "no entry-point code identity"
            )
    else:
        # EVERY v1 artifact is a legacy artifact, so the recognised, self-consistent
        # identity is required UNCONDITIONALLY -- not merely when the audit happens
        # to be absent.  Otherwise a stale producer could satisfy the schema and
        # then become acceptable simply by adding {"complete": true}.
        identity = str(payload.get("four_gw_certification_identity") or "")
        recognised = identity in set(LEGACY_CERTIFICATION_IDENTITIES)
        self_consistent = identity == certification_identity_of(payload)
        if not (recognised and self_consistent):
            detail = (
                "not a recognised legacy identity" if not recognised
                else "the identity does not match this artifact's own cutoff/bundles/snapshot"
            )
            raise DecisionCertificationRequired(
                f"{DIAG_CERTIFIED_HISTORY_COMPLETENESS_EVIDENCE_MISSING}: "
                f"{DIAG_LEGACY_CERTIFICATION_IDENTITY_UNRECOGNISED}: schema {schema!r} is a legacy "
                f"artifact, which requires a recognised AND self-consistent historical certification "
                f"identity ({detail}); {identity or '<none>'!r} is not a grandfathered legacy certification"
            )
    # A PRESENT audit must be complete at every version.
    if isinstance(completeness, Mapping) and completeness.get("complete") is not True:
        from . import history_completeness as hc

        raise DecisionCertificationRequired(
            f"{hc.blocking_reason_token(completeness)}: the certification's completed-event history "
            f"audit is not complete ({completeness.get('reasons')})"
        )
    if payload.get("route_search_executed") is not False or payload.get(
        "transfer_execution_performed"
    ) is not False:
        raise DecisionCertificationRequired(
            "artifact records prior execution; it is not a clean certification"
        )
    return payload


def load_certification_artifact(path: str | Path) -> Any:
    """Load and validate a HISTORICAL certification artifact produced by the certifier.

    It reads the artifact's bytes and applies the complete artifact contract
    (:func:`validate_certification_artifact`).  The value returned is a PLAIN
    mapping: under PE-9 amendment 2 the authority for a predictive load is a
    persisted, content-addressed GENERATION (:mod:`fpl_brain.generation_store`), not
    a caller-carried object, so nothing here is a capability and no production
    predictive boundary is gated on the result.
    """

    import json
    from pathlib import Path as _Path

    artifact_path = _Path(path)
    if not artifact_path.exists():
        raise DecisionCertificationRequired(
            f"{DIAG_CERTIFICATION_ARTIFACT_ABSENT}: no certification artifact at {artifact_path}"
        )
    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    except Exception as exc:  # malformed artifact must not be silently ignored
        raise DecisionCertificationRequired(f"certification artifact unreadable: {exc}") from exc
    return validate_certification_artifact(payload)


def canonical_event_horizon(values: Iterable[Any] | None) -> tuple[int, ...]:
    """Canonical (sorted, int) horizon for an exact-identity comparison.

    Order is canonicalised, duplicates are NOT collapsed: a horizon that repeats
    an event therefore fails an equality test deterministically instead of being
    silently normalised into acceptance.
    """

    return tuple(sorted(int(event) for event in (values or ())))


def assert_certification_artifact_describes_decision(
    certification: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
    *,
    events: Iterable[int] | None = None,
    cutoff: str | None = None,
) -> Mapping[str, Any]:
    """Exactly one certification artifact per decision, and it must describe it.

    ``CERTIFICATION_ARTIFACT_ABSENT`` and ``CERTIFICATION_ARTIFACT_CONTRADICTORY``
    are deliberately different tokens: a missing authorisation and an artifact
    that contradicts the decision are different operational facts.  A horizon
    assembled from two artifacts is a contradiction and is refused.
    """

    try:
        resolved = cb.require_certification_artifact(certification)
    except cb.CertificationArtifactAbsent as failure:
        raise DecisionCertificationRequired(str(failure)) from failure
    except cb.CertificationArtifactContradictory as failure:
        raise DecisionCertificationRequired(str(failure)) from failure
    contradictions: list[str] = []
    if events is not None:
        requested = canonical_event_horizon(events)
        certified = canonical_event_horizon(resolved.get("events"))
        if requested != certified:
            contradictions.append(
                f"{DIAG_CERTIFICATION_EVENT_SET_MISMATCH}: the decision horizon {list(requested)} is "
                f"not the certified horizon {list(certified)}"
            )
    if cutoff is not None:
        declared = resolved.get("planning_cutoff")
        # The bundle-level cutoff is enforced exactly, by the dependency validator,
        # against this same decision cutoff.  Here only a DECLARED artifact cutoff
        # that contradicts the decision is refused: an artifact that declares none
        # is bounded by its bundles, which is the identity that protects freshness.
        if declared is not None and str(declared) != str(cutoff):
            contradictions.append(
                f"the certification artifact's planning cutoff {declared} is not the decision "
                f"cutoff {cutoff}"
            )
    if not resolved.get("data_snapshot_sha256"):
        contradictions.append("the certification artifact carries no data snapshot identity")
    if contradictions:
        raise DecisionCertificationRequired(
            f"{DIAG_CERTIFICATION_ARTIFACT_CONTRADICTORY}: " + "; ".join(contradictions)
        )
    return resolved


def event_support_from_certification(
    conn: sqlite3.Connection,
    certification: Mapping[str, Any],
    *,
    events: Iterable[int],
    cutoff: str,
) -> dict[int, dict[str, Any]]:
    """Support derived from a CERTIFICATION ARTIFACT's exact certified run ids.

    The horizon a decision CONSUMES must equal the horizon the artifact CERTIFIED.
    A caller-supplied subset is not authorisation: a certification for five events
    must never authorize a four-event consumer, nor the reverse.  The comparison is
    canonical (sorted ints) and exact, so subsets, supersets, duplicates and
    reorderings are all refused before any support is returned.
    """

    certification = assert_certification_artifact_describes_decision(
        certification, events=events, cutoff=cutoff
    )
    bundles = certification.get("certified_bundles") or {}
    selected = {
        int(event): {
            "runs": dict((bundles[str(event)] or {}).get("runs") or {}),
            "data_snapshot_sha256": (bundles[str(event)] or {}).get("data_snapshot_sha256")
            or certification.get("data_snapshot_sha256"),
            "code_snapshot_sha256": (bundles[str(event)] or {}).get("code_snapshot_sha256")
            or certification.get("code_snapshot_sha256"),
        }
        for event in events
        if str(event) in bundles
    }
    missing = [int(event) for event in events if int(event) not in selected]
    if missing:
        raise DecisionCertificationRequired(f"certification covers no bundle for event {missing}")
    return event_support_from_certified_bundles(
        conn,
        selected,
        cutoff=cutoff,
        # PE-9 gap 1: the required model versions come from the ONE declared
        # IN-LIBRARY source and NEVER from the artifact.  Reading the artifact's own
        # record made the caller authoritative: an artifact that declared a relaxed or
        # absent pin would have silently un-pinned the certification.  The artifact's
        # record is no longer consulted here at all.
        required_versions=cb.declared_required_versions(),
        data_snapshot_sha256=certification.get("data_snapshot_sha256"),
    )


def event_support_from_certified_bundles(
    conn: sqlite3.Connection,
    bundles_by_event: Mapping[int, Mapping[str, Any]],
    *,
    cutoff: str,
    required_versions: Mapping[str, str] | None = None,
    data_snapshot_sha256: str | None = None,
) -> dict[int, dict[str, Any]]:
    """Per-event support derived from CERTIFIED bundles' EXACT run ids.

    Unlike :func:`event_support_from_db`, this never rediscovers a "latest
    matching run" per family: the bundle names the run ids, they are validated as
    one coherent DAG via ``certified_bundle``, and only then do they count as
    support.  An incoherent bundle raises instead of silently degrading.

    ``data_snapshot_sha256`` is the certification artifact's own snapshot identity.
    A bundle that declares a DIFFERENT one describes a different predictive world
    and is refused, so the snapshot identity is validated rather than recorded.
    """

    from . import certified_bundle as cb

    support: dict[int, dict[str, Any]] = {}
    failures: dict[int, list[str]] = {}
    for event, bundle in bundles_by_event.items():
        runs = {
            str(family): int(run_id)
            for family, run_id in (bundle.get("runs") or {}).items()
        }
        try:
            certified = cb.certified_bundle_from_explicit_ids(
                conn,
                event=int(event),
                cutoff=cutoff,
                runs=runs,
                required_versions=required_versions,
                data_snapshot_sha256=bundle.get("data_snapshot_sha256"),
                code_snapshot_sha256=bundle.get("code_snapshot_sha256"),
                expected_data_snapshot_sha256=data_snapshot_sha256,
            )
        except cb.BundleIncoherent as failure:
            # Which EVENT failed is part of the refusal: one bad event makes the
            # horizon incomplete, and an operator must be able to see which one
            # without re-deriving it from the run ids.
            failures[int(event)] = list(failure.reasons)
            continue
        support[int(event)] = {
            "supported": True,
            "matched_runs": dict(certified.runs),
            "missing_families": [],
            "stale_families": [],
            # validate_certified_bundle() refuses the bundle unless EVERY family's
            # projection_runs.data_cutoff equals this cutoff, so the value is proven
            # rather than assumed.  Without it evaluate_horizon() sees data_cutoff=None
            # and marks every certified event STALE_CUTOFF_MISMATCH.
            "data_cutoff": str(cutoff),
            "run_cutoffs": [str(cutoff)],
            "bundle_identity": certified.bundle_identity(),
            "state": cb.STATE_CERTIFIED_COHERENT,
            "source": "certified_bundle",
        }
    if failures:
        raise cb.BundleIncoherent(
            [
                f"GW{event}: {reason}"
                for event, reasons in sorted(failures.items())
                for reason in reasons
            ]
        )
    return support



def event_support_from_db(
    conn: sqlite3.Connection,
    events: Iterable[int],
    cutoff: str,
    *,
    families: Sequence[str] = REQUIRED_HORIZON_FAMILIES,
) -> dict[int, dict[str, Any]]:
    """NON-PRODUCTION (diagnostics and tests only).

    Per-event predictive support derived from accepted projection runs, choosing
    the NEWEST same-cutoff run per family.  That silently lets a later rerun of
    one family replace the certified one, and it validates no dependency edges,
    so it must NOT be used for a production decision.  Production readiness
    resolves the certified GENERATION instead
    (:func:`generation_store.resolve_generation` / :func:`generation_store.support_by_event`)
    and refuses when no generation is selected.
    """

    support: dict[int, dict[str, Any]] = {}
    for event in events:
        rows = conn.execute(
            "SELECT id, model_family, model_version, data_cutoff, status "
            "FROM projection_runs WHERE planning_event=? ORDER BY id",
            (int(event),),
        ).fetchall()
        by_family: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            record = dict(row)
            by_family.setdefault(str(record["model_family"]), []).append(record)
        matched: dict[str, int] = {}
        missing: list[str] = []
        stale_families: list[str] = []
        for family in families:
            candidates = by_family.get(family) or []
            fresh = [
                record for record in candidates
                if str(record.get("data_cutoff")) == str(cutoff) and str(record.get("status")) == "complete"
            ]
            if fresh:
                matched[family] = int(max(record["id"] for record in fresh))
                continue
            if candidates:
                stale_families.append(family)
            missing.append(family)
        run_cutoffs = sorted({str(record.get("data_cutoff")) for records in by_family.values() for record in records})
        support[int(event)] = {
            "event": int(event),
            "supported": not missing,
            "data_cutoff": cutoff if not missing else (run_cutoffs[-1] if run_cutoffs else None),
            "matched_runs": matched,
            "missing_families": missing,
            "stale_families": stale_families,
            "observed_cutoffs": run_cutoffs,
            "has_any_run": bool(rows),
        }
    return support


def evaluate_horizon(
    *,
    planning_event: int,
    support_by_event: Mapping[int, Mapping[str, Any]],
    cutoff: str | None = None,
    length: int = DECISION_HORIZON_LENGTH,
    last_event: int = SEASON_LAST_EVENT,
) -> dict[str, Any]:
    """Contiguous rolling-horizon readiness over the season's real events.

    ``support_by_event[event]`` is any mapping with ``supported`` (bool),
    ``data_cutoff`` (str | None) and optionally ``missing_families`` /
    ``stale_families``.  When ``cutoff`` is given, an event whose support was
    produced under a different cutoff is NOT supported — fresh GW data may not
    be silently combined with stale future predictions.

    Status is one of:

    * ``DECISION_HORIZON_COMPLETE`` — all four real events are supported.
    * ``SEASON_END_SHORT_HORIZON`` — fewer than four real Gameweeks remain
      (e.g. GW36 -> 36/37/38) and every remaining event is supported.  The
      transfer rule still applies, over all remaining events.
    * ``DECISION_HORIZON_INCOMPLETE`` — at least one available event lacks
      coherent same-cutoff support; the transfer recommendation is suppressed.
    """

    events = decision_events(planning_event, length=length, last_event=last_event)
    requested = int(length)
    effective = len(events)
    short_horizon = effective < requested
    per_event: dict[str, dict[str, Any]] = {}
    supported: list[int] = []
    blocked: list[int] = []
    missing_events: list[int] = []
    stale_events: list[int] = []
    for event in events:
        raw = support_by_event.get(event)
        if raw is None:
            per_event[str(event)] = {
                "event": int(event), "supported": False, "reason": "NO_SUPPORT_RECORD",
                "data_cutoff": None, "missing_families": list(REQUIRED_HORIZON_FAMILIES),
                "stale_families": [], "cutoff_matches": False,
            }
            blocked.append(int(event))
            missing_events.append(int(event))
            continue
        declared = bool(raw.get("supported"))
        event_cutoff = None if raw.get("data_cutoff") is None else str(raw.get("data_cutoff"))
        cutoff_matches = cutoff is None or event_cutoff == str(cutoff)
        ok = declared and cutoff_matches
        reason = None
        if not declared:
            reason = "PREDICTION_READINESS_FAILED"
        elif not cutoff_matches:
            reason = "STALE_CUTOFF_MISMATCH"
        record = {
            "event": int(event),
            "supported": ok,
            "declared_supported": declared,
            "reason": reason,
            "data_cutoff": event_cutoff,
            "missing_families": list(raw.get("missing_families") or []),
            "stale_families": list(raw.get("stale_families") or []),
            "cutoff_matches": bool(cutoff_matches),
        }
        per_event[str(event)] = record
        if ok:
            supported.append(int(event))
        else:
            blocked.append(int(event))
            if reason == "STALE_CUTOFF_MISMATCH":
                stale_events.append(int(event))
            else:
                missing_events.append(int(event))
    if blocked:
        status = DECISION_HORIZON_INCOMPLETE
    elif short_horizon:
        status = SEASON_END_SHORT_HORIZON
    else:
        status = DECISION_HORIZON_COMPLETE
    return {
        "horizon_length": int(length),
        "requested_horizon_length": requested,
        "effective_horizon_length": effective,
        "short_horizon": bool(short_horizon),
        "season_last_event": int(last_event),
        "planning_event": int(planning_event),
        "decision_events": list(events),
        "planning_cutoff": None if cutoff is None else str(cutoff),
        "same_cutoff_required": True,
        "status": status,
        "complete": not blocked,
        "supported_events": supported,
        "blocked_events": blocked,
        "missing_prediction_events": missing_events,
        "stale_cutoff_events": stale_events,
        "events": per_event,
        "flags": list(HORIZON_FLAGS) + (["SEASON_END_SHORT_HORIZON"] if short_horizon else []),
    }


def transfer_recommendation_allowed(horizon: Mapping[str, Any]) -> bool:
    """A normal transfer recommendation requires every available event to be
    supported — four events normally, or all remaining events at season end."""

    return bool(horizon.get("complete")) and str(horizon.get("status")) in {
        DECISION_HORIZON_COMPLETE,
        SEASON_END_SHORT_HORIZON,
    }


# ---------------------------------------------------------------------------
# Four-GW net value (primary transfer objective)
# ---------------------------------------------------------------------------


def _feature_value(feature: Any, field: str) -> Any:
    """Read an event feature field from a mapping or an ``EventFeature`` dataclass."""

    if isinstance(feature, Mapping):
        return feature.get(field)
    return getattr(feature, field, None)


def event_expected_core(row: Mapping[str, Any], event: int) -> float:
    """Per-event expected CORE from a candidate-universe player row.

    Accepts the raw universe (``EventFeature`` dataclasses) as well as the
    JSON-safe form, because callers legitimately hold both.
    """

    for feature in row.get("events") or []:
        if int(_feature_value(feature, "event")) == int(event):
            return float(_feature_value(feature, "expected_core") or 0.0)
    return 0.0


def event_availability(row: Mapping[str, Any], event: int) -> float:
    """Per-event availability flag from a candidate-universe player row."""

    for feature in row.get("events") or []:
        if int(_feature_value(feature, "event")) == int(event):
            return float(_feature_value(feature, "availability") or 0.0)
    return 0.0


def four_gw_expected_core(row: Mapping[str, Any], events: Sequence[int]) -> float:
    """Sum of per-event expected CORE over the decision events for one player."""

    return sum(event_expected_core(row, int(event)) for event in events)


def four_gw_window_value(
    per_event: Sequence[Mapping[str, Any]],
    *,
    events: Sequence[int] | None = None,
) -> dict[str, Any]:
    """SUM expected CORE over the decision events minus hits incurred there.

    ``per_event`` entries provide ``event``, ``mean_gross_core`` (or
    ``gross_core``) and ``hit_points``.  This is the primary transfer-route
    objective; current-GW gross/net stay reported for transparency.
    """

    ordered = sorted(
        (record for record in per_event if events is None or int(record["event"]) in {int(e) for e in events}),
        key=lambda record: int(record["event"]),
    )
    gross = 0.0
    hits = 0
    usable: list[int] = []
    unusable: list[int] = []
    for record in ordered:
        if not usable_expected_core(record):
            unusable.append(int(record["event"]))
            continue
        usable.append(int(record["event"]))
        gross += float(record.get("mean_gross_core", record.get("gross_core")))
        hits += int(record.get("hit_points", 0) or 0)
    return {
        "events": usable,
        "unusable_events": unusable,
        "event_expected_core": [
            float(record.get("mean_gross_core", record.get("gross_core")))
            for record in ordered if usable_expected_core(record)
        ],
        "four_gw_gross_core": gross,
        "total_hit_points": hits,
        "four_gw_net_core": gross - hits,
    }


def route_eligibility(
    route: Mapping[str, Any],
    *,
    decision_events_window: Sequence[int],
) -> dict[str, Any]:
    """Whether a route may enter the transfer ranking, with explicit reasons.

    A route is eligible only when it is valid, carries EXACTLY ONE usable event
    evaluation for EVERY event in the decision horizon, and has coherent
    terminal accounting.  A missing event is never silently treated as zero
    expected CORE, and an event score that is None / NaN / infinite is
    UNUSABLE rather than zero.
    """

    window = [int(event) for event in decision_events_window]
    reasons: list[str] = []
    route_id = str(route.get("route_id"))
    if route.get("valid") is not True:
        reasons.append("ROUTE_INVALID")
    per_event_list = list(route.get("per_event") or [])
    counts: dict[int, int] = {}
    unusable: list[int] = []
    for record in per_event_list:
        event = int(record["event"])
        counts[event] = counts.get(event, 0) + 1
        if not usable_expected_core(record):
            unusable.append(event)
    missing = [event for event in window if counts.get(event, 0) == 0]
    duplicated = [event for event in window if counts.get(event, 0) > 1]
    extra = [event for event in sorted(counts) if event not in window]
    if missing:
        reasons.append("MISSING_EVENT_EVALUATION:" + ",".join(f"GW{event}" for event in missing))
    if duplicated:
        reasons.append("DUPLICATE_EVENT_EVALUATION:" + ",".join(f"GW{event}" for event in duplicated))
    if extra:
        reasons.append("EVENT_OUTSIDE_HORIZON:" + ",".join(f"GW{event}" for event in extra))
    if unusable:
        reasons.append("UNUSABLE_EVENT_CORE:" + ",".join(f"GW{event}" for event in sorted(set(unusable))))
    for field in ("terminal_ft", "terminal_bank_tenths"):
        if route.get(field) is None:
            reasons.append(f"TERMINAL_ACCOUNTING_MISSING:{field}")
    return {
        "route_id": route_id,
        "eligible": not reasons,
        "reasons": reasons,
        "events_present": sorted(counts),
        "decision_events": window,
    }


def usable_expected_core(record: Mapping[str, Any]) -> bool:
    """True only for a finite, present expected-CORE value on an event record."""

    value = record.get("mean_gross_core", record.get("gross_core"))
    if value is None:
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


def route_for_decision(
    comparison_route: Mapping[str, Any],
    *,
    route_id: str | None = None,
    transfers_by_event: Mapping[int, Sequence[Mapping[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Adapt a route_comparator route record into the decision-layer shape.

    ``route_comparator`` reports ``events`` (per-event records) and a
    ``terminal_state`` block; the decision layer needs ``per_event``,
    ``terminal_ft`` and ``terminal_bank_tenths``.  Doing that conversion in one
    place prevents a silently mis-shaped route from ever reaching the ranking.

    The comparator's per-event records do not carry the transfer list (that
    lives in the submitted route steps), so callers pass
    ``transfers_by_event`` explicitly when the board needs the sequence.

    This is the ``route_comparator`` boundary only: a ``route_optimizer`` record (which
    carries ``per_event``/``actions``) raises instead of being adapted into an incomplete
    route, because adapting it silently is exactly the R5-P0-01 defect.
    """

    if route_record_shape(comparison_route) == "optimizer":
        raise DecisionRouteShapeError(
            "DECISION_ROUTE_SHAPE_MISMATCH: route_for_decision received a route_optimizer "
            "record (it carries 'per_event'/'actions'); use optimizer_route_for_decision / "
            "optimizer_routes_for_decision for optimizer records"
        )
    transfers_by_event = dict(transfers_by_event or {})
    per_event = [
        {
            "event": int(record["event"]),
            "mean_gross_core": record.get("mean_gross_core"),
            "hit_points": int(record.get("hit_points") or 0),
            "policy": record.get("selected_policy"),
            "transfers": [
                {"out": int(move["out"]), "in": int(move["in"])}
                for move in (
                    (record.get("transfers") or transfers_by_event.get(int(record["event"])) or [])
                )
            ],
            "bank_after_tenths": record.get("bank_after_tenths"),
            "next_bank_tenths": record.get("next_bank_tenths"),
            "next_free_transfers": record.get("next_free_transfers"),
        }
        for record in (comparison_route.get("events") or [])
    ]
    terminal = comparison_route.get("terminal_state") or {}
    return {
        "route_id": str(route_id if route_id is not None else comparison_route.get("route_id")),
        "valid": bool(comparison_route.get("valid")),
        "per_event": per_event,
        "terminal_ft": terminal.get("free_transfers"),
        "terminal_bank_tenths": terminal.get("bank_tenths"),
        "chip_used": comparison_route.get("chip_used"),
        "errors": list(comparison_route.get("errors") or []),
    }


def routes_for_decision(
    routes: Mapping[str, Mapping[str, Any]],
    *,
    transfers_by_route: Mapping[str, Mapping[int, Sequence[Mapping[str, Any]]]] | None = None,
) -> list[dict[str, Any]]:
    """Adapt every route in a ``compare_routes`` result for the decision layer.

    This is the ``route_comparator`` boundary only.  A ``route_optimizer`` result must go
    through :func:`optimizer_routes_for_decision`; feeding an optimizer record to this
    function raises :class:`DecisionRouteShapeError` instead of silently producing an
    incomplete route (see ``R5-P0-01``).
    """

    transfers_by_route = dict(transfers_by_route or {})
    return [
        route_for_decision(record, route_id=str(route_id), transfers_by_event=transfers_by_route.get(str(route_id)))
        for route_id, record in sorted(routes.items())
    ]


# ---------------------------------------------------------------------------
# The decision layer's canonical route record — ONE declared boundary per source schema
# ---------------------------------------------------------------------------
#
# ``route_eligibility`` / ``build_decision_board`` consume exactly one shape.  Two upstream
# producers emit routes in DIFFERENT schemas, so there are exactly two adapters, and each
# reads only its own declared schema:
#
#   route_comparator  -> route_for_decision            (``events`` + ``terminal_state``)
#   route_optimizer   -> optimizer_route_for_decision  (``per_event`` + ``actions`` + top-level
#                                                       ``terminal_ft`` / ``terminal_bank_tenths``)
#
# THE CANONICAL RECORD (what both adapters return, and all the board reads):
#   route_id, valid, per_event[
#       event, mean_gross_core, hit_points, policy, transfers[{out,in}],
#       bank_after_tenths, next_bank_tenths, next_free_transfers
#   ], terminal_ft, terminal_bank_tenths, chip_used, errors
#
# Why this boundary is explicit rather than a set of ``.get()`` fallbacks: the R5-P0-01
# defect was SILENT.  Feeding an optimizer record to the comparator adapter produced
# ``per_event == []`` and null terminal accounting, which ``route_eligibility`` then
# correctly reported as an incomplete route — so every route was excluded and the board
# was empty, with nothing anywhere saying "wrong schema".  A single declared mapping per
# producer, plus a loud error when the WRONG producer's record arrives, is what makes that
# failure impossible to repeat quietly.
CANONICAL_DECISION_ROUTE_KEYS: tuple = (
    "route_id", "valid", "per_event", "terminal_ft", "terminal_bank_tenths", "chip_used",
    "errors",
)
CANONICAL_DECISION_PER_EVENT_KEYS: tuple = (
    "event", "mean_gross_core", "hit_points", "policy", "transfers", "bank_after_tenths",
    "next_bank_tenths", "next_free_transfers",
)
OPTIMIZER_ROUTE_DISCRIMINATORS: tuple = ("per_event", "actions")
COMPARATOR_ROUTE_DISCRIMINATORS: tuple = ("events", "terminal_state")


class DecisionRouteShapeError(ValueError):
    """A route record was handed to the adapter for a different producer's schema."""


def route_record_shape(record: Mapping[str, Any]) -> str:
    """Which declared producer schema a route record belongs to.

    ``"optimizer"`` / ``"comparator"`` by discriminator key, else ``"unknown"``.  The
    discriminators are the keys only that producer emits, so this cannot be satisfied by
    coincidence for a route that has one shape and not the other.
    """

    keys = set(record or ())
    if keys & set(OPTIMIZER_ROUTE_DISCRIMINATORS):
        return "optimizer"
    if keys & set(COMPARATOR_ROUTE_DISCRIMINATORS):
        return "comparator"
    return "unknown"


def _optimizer_transfers(record: Mapping[str, Any]) -> dict[int, list[dict[str, int]]]:
    """Per-event transfer list from a ``route_optimizer`` record's serialized ``actions``."""

    out: dict[int, list[dict[str, int]]] = {}
    for action in record.get("actions") or []:
        out[int(action["event"])] = [
            {"out": int(move["out"]), "in": int(move["in"])}
            for move in (action.get("transfers") or [])
        ]
    return out


def optimizer_route_for_decision(
    record: Mapping[str, Any],
    *,
    route_id: str | None = None,
) -> dict[str, Any]:
    """Adapt a ``route_optimizer`` route record into the canonical decision record.

    ``route_optimizer.optimize`` reports ``per_event`` (event, kind, hit points, gross/net
    CORE, selected policy), the serialized per-event ``actions`` (transfer list, bank after
    the event's batch, free transfers after it) and the route's TERMINAL ``ft``/bank at the
    top level.  This maps exactly those fields onto the canonical record — nothing is
    invented and nothing is defaulted: a genuinely absent event, transfer list or terminal
    value stays absent/None so :func:`route_eligibility` still fails it closed with
    ``MISSING_EVENT_EVALUATION`` / ``TERMINAL_ACCOUNTING_MISSING``.
    """

    shape = route_record_shape(record)
    if shape == "comparator":
        raise DecisionRouteShapeError(
            "DECISION_ROUTE_SHAPE_MISMATCH: optimizer_route_for_decision received a "
            "route_comparator record (it carries 'events'/'terminal_state'); use "
            "route_for_decision/routes_for_decision for comparator records"
        )

    transfers_by_event = _optimizer_transfers(record)
    per_event = [
        {
            "event": int(item["event"]),
            "mean_gross_core": item.get("mean_gross_core"),
            "hit_points": int(item.get("hit_points") or 0),
            "policy": item.get("policy"),
            # The optimizer's action for this event is where the transfer list and the
            # post-batch bank / free transfers live; absent means absent, not empty.
            "transfers": list(transfers_by_event.get(int(item["event"]), [])),
            "bank_after_tenths": _optimizer_event_accounting(record, int(item["event"]), "bank_after"),
            "next_bank_tenths": _optimizer_event_accounting(record, int(item["event"]), "bank_after"),
            "next_free_transfers": _optimizer_event_accounting(record, int(item["event"]), "ft_after"),
        }
        for item in (record.get("per_event") or [])
    ]
    return {
        "route_id": str(route_id if route_id is not None else record.get("route_id")),
        "valid": bool(record.get("valid")),
        "per_event": per_event,
        "terminal_ft": record.get("terminal_ft"),
        "terminal_bank_tenths": record.get("terminal_bank_tenths"),
        # A route_optimizer route never activates a chip (chip modelling is out of scope).
        "chip_used": None,
        "errors": [],
    }


def _optimizer_event_accounting(record: Mapping[str, Any], event: int, field: str):
    """One per-event accounting value from the optimizer's serialized actions.

    ``None`` when the event has no action or the action does not carry the field: the
    decision layer must be able to tell "not recorded" from "zero".
    """

    for action in record.get("actions") or []:
        if int(action["event"]) == int(event):
            value = action.get(field)
            return None if value is None else int(value)
    return None


def optimizer_routes_for_decision(
    routes: Mapping[str, Mapping[str, Any]],
    *,
    transfers_by_route: Mapping[str, Mapping[int, Sequence[Mapping[str, Any]]]] | None = None,
) -> list[dict[str, Any]]:
    """Adapt every route in a ``route_optimizer.optimize`` result for the decision layer.

    This is the boundary the production runner must use.  ``transfers_by_route`` is accepted
    for signature symmetry with :func:`routes_for_decision` and is NOT needed: an optimizer
    record already carries its transfers on the serialized actions.
    """

    return [
        optimizer_route_for_decision(record, route_id=str(route_id))
        for route_id, record in sorted(routes.items())
    ]


def canonical_decision_route(record: Mapping[str, Any], *, route_id: str | None = None) -> dict[str, Any]:
    """The canonical decision record for a route of EITHER declared schema.

    One entry point for callers that legitimately do not know which producer they hold
    (for example a report that reads a saved artifact).  It dispatches on the declared
    discriminator keys and never guesses field-by-field; an unrecognised record is adapted
    as a comparator record only when it has no optimizer discriminator at all.
    """

    if route_record_shape(record) == "optimizer":
        return optimizer_route_for_decision(record, route_id=route_id)
    return route_for_decision(record, route_id=route_id)


def lineup_policy_for_route(
    *,
    routes: Sequence[Mapping[str, Any]],
    route_id: str | None,
    decision_events_window: Sequence[int],
) -> dict[str, Any] | None:
    """The H1 policy of ONE named route, or None when that route has none.

    Used so the operator-visible current-GW lineup belongs to the route being
    recommended, instead of whichever route happened to appear first.
    """

    if route_id is None:
        return None
    window = [int(event) for event in decision_events_window]
    if not window:
        return None
    first_event = window[0]
    for row in routes:
        if str(row.get("route_id")) != str(route_id):
            continue
        for record in row.get("per_event") or []:
            if int(record.get("event", -1)) == first_event and record.get("policy"):
                return dict(record["policy"])
    return None


def rank_routes_by_four_gw(
    routes: Sequence[Mapping[str, Any]],
    *,
    value_key: str = "four_gw_net_core",
) -> list[dict[str, Any]]:
    """Rank transfer routes by the four-GW net objective (ties broken safely).

    Callers must pass only ELIGIBLE routes; this function does not re-check
    validity (use :func:`route_eligibility` first).
    """

    ranked = sorted(
        routes,
        key=lambda record: (
            -float(record.get(value_key, float("-inf"))),
            -int(record.get("terminal_ft", 0) or 0),
            -int(record.get("terminal_bank_tenths", 0) or 0),
            str(record.get("route_id", "")),
        ),
    )
    return [{**record, "rank": index} for index, record in enumerate(ranked, start=1)]


# ---------------------------------------------------------------------------
# Full legal action screen before route promotion
# ---------------------------------------------------------------------------


def screen_legal_actions(
    *,
    universe_rows: Sequence[Mapping[str, Any]],
    replacement_edges: Sequence[Mapping[str, Any]],
    owned_ids: Iterable[int],
    decision_events_window: Sequence[int],
    promotion_pool_limit: int = 240,
    per_out_limit: int = 12,
) -> dict[str, Any]:
    """Screen EVERY legal single transfer before any route is promoted.

    ``replacement_edges`` is the Phase-7A enumeration owned × same-position
    universe candidate.  The screen asserts that enumeration is exhaustive for
    the inputs it was given, so a player who is present in the fresh
    CandidateUniverse and forms a legal transfer can enter screening without
    anyone naming that player first.
    """

    owned = {int(pid) for pid in owned_ids}
    rows = {int(row["player_id"]): row for row in universe_rows}
    expected_pairs = sum(
        1
        for out_id in owned
        for in_id, row in rows.items()
        if in_id not in owned and out_id in rows and row["position"] == rows[out_id]["position"]
    )
    edges = list(replacement_edges)
    legal = [edge for edge in edges if edge.get("currently_legal_single_transfer")]
    illegal = [edge for edge in edges if not edge.get("currently_legal_single_transfer")]

    window = [int(event) for event in decision_events_window]
    scored: list[dict[str, Any]] = []
    for edge in legal:
        out_row = rows.get(int(edge["out_player_id"]))
        in_row = rows.get(int(edge["in_player_id"]))
        if out_row is None or in_row is None:
            continue
        proxy = sum(
            event_expected_core(in_row, event) - event_expected_core(out_row, event) for event in window
        )
        scored.append({**edge, "four_gw_proxy_delta": proxy})
    scored.sort(key=lambda edge: (-float(edge["four_gw_proxy_delta"]), int(edge["out_player_id"]), int(edge["in_player_id"])))

    pool: list[dict[str, Any]] = []
    per_out: dict[int, int] = {}
    for edge in scored:
        out_id = int(edge["out_player_id"])
        if per_out.get(out_id, 0) >= int(per_out_limit):
            continue
        pool.append(edge)
        per_out[out_id] = per_out.get(out_id, 0) + 1
        if len(pool) >= int(promotion_pool_limit):
            break

    discoverable = sorted({int(edge["in_player_id"]) for edge in legal})
    eligible = sorted({int(edge["in_player_id"]) for edge in scored})
    enumerated = len(edges)
    desired = desired_squad_changes(legal_actions=scored)
    return {
        "enumerated_transfer_pairs": int(enumerated),
        "expected_transfer_pairs": int(expected_pairs),
        "enumeration_exhaustive": bool(enumerated == expected_pairs),
        "legal_single_transfers": len(legal),
        "illegal_single_transfers": len(illegal),
        "screened_legal_actions": len(scored),
        "promotion_pool_size": len(pool),
        "promotion_pool_limit": int(promotion_pool_limit),
        "per_out_limit": int(per_out_limit),
        "eligible_in_player_ids": eligible,
        "discoverable_in_player_ids": discoverable,
        "promotion_pool": pool,
        "desired_squad_changes": desired,
        "coverage": {
            "every_legal_edge_screened": bool(len(scored) == len(legal)),
            "every_legal_in_player_eligible": bool(set(discoverable) == set(eligible)),
        },
        "reasons": {
            "out_player_id": "owned player in the active RouteState",
            "in_player_id": "present in the fresh CandidateUniverse",
            "legality": "Phase-7A apply_transfer_batch",
            "proxy": "sum of per-event expected CORE over the decision events",
        },
        "no_recommendation": True,
    }


# ---------------------------------------------------------------------------
# Wildcard trigger screen (bounded, auditable, no fabricated points)
# ---------------------------------------------------------------------------


def wildcard_same_gw_hit(
    *,
    prior_paid_transfers: int = 0,
    wildcard_transfers: int = 0,
    rules: SeasonRules | None = None,
) -> dict[str, Any]:
    """Gameweek transfer deduction when the Wildcard is played.

    Delegates to the season rules so the official rule lives in one place.
    With the chip active the net hit is 0 even when extra transfers were
    already charged earlier in the same Gameweek.
    """

    return wildcard_gameweek_hit(
        rules or SeasonRules(season="2026/27"),
        prior_paid_transfers=int(prior_paid_transfers),
        wildcard_transfers=int(wildcard_transfers),
        wildcard_active=True,
    )


def chip_free_transfers_after(
    chip: str,
    *,
    event_start_free_transfers: int | None,
    free_transfers_available: int | None = None,
    free_transfers_used: int = 0,
    rules: SeasonRules | None = None,
) -> int:
    """Free transfers entering the Gameweek after a chip week.

    Wildcard/Free Hit retain the saved FT state (no weekly +1 accrual) and the
    retained value is the EVENT-START bank, which must be explicitly recorded.
    """

    return free_transfers_after_chip(
        rules or SeasonRules(season="2026/27"),
        chip,
        event_start_free_transfers=event_start_free_transfers,
        free_transfers_available=free_transfers_available,
        free_transfers_used=int(free_transfers_used),
    )


def desired_squad_changes(
    *,
    legal_actions: Sequence[Mapping[str, Any]],
    materiality_core: float = 0.5,
) -> dict[str, Any]:
    """Squad-change units, NOT candidate-edge units.

    Counts the DISTINCT owned slots for which at least one legal replacement has
    a materially positive four-GW proxy delta.  The result is bounded by the
    squad size (<= 15), so a market with thousands of legal options does not
    imply thousands of desired moves.  Producers must supply the per-action
    ``out_player_id`` and ``four_gw_proxy_delta`` (see ``screen_legal_actions``).
    """

    best_by_slot: dict[int, float] = {}
    for edge in legal_actions:
        if edge.get("out_player_id") is None:
            continue
        out_id = int(edge["out_player_id"])
        delta = float(edge.get("four_gw_proxy_delta", 0.0) or 0.0)
        if delta > best_by_slot.get(out_id, float("-inf")):
            best_by_slot[out_id] = delta
    slots = sorted(pid for pid, delta in best_by_slot.items() if delta >= float(materiality_core))
    return {
        "desired_transfer_count": len(slots),
        "desired_transfer_source": "DISTINCT_OWNED_SLOTS_WITH_MATERIAL_4GW_REPLACEMENT",
        "materiality_core": float(materiality_core),
        "slots_with_material_replacement": slots,
        "legal_actions_considered": len(legal_actions),
        "max_possible": 15,
        "units": "SQUAD_CHANGES",
        "note": (
            "candidate-edge count is the size of the legal action space and is never used as the "
            "number of desired squad changes"
        ),
    }


def wildcard_trigger_screen(
    *,
    weak_slot_count: int = 0,
    desired_transfer_count: int = 0,
    expected_hit_burden: int = 0,
    weak_slot_threshold: int = 3,
    desired_transfer_threshold: int = 4,
    hit_burden_threshold: int = 8,
    structural_budget_constraint: bool = False,
    availability_problems: int = 0,
    prior_paid_transfers: int = 0,
    wildcard_transfers: int = 0,
    event_start_free_transfers: int | None = None,
    free_transfers_available: int | None = None,
    rules: SeasonRules | None = None,
    supported_evaluation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Surface whether a Wildcard deserves review — never invent its points.

    ``WILDCARD_EVALUATION_SUPPORTED`` is returned only when the caller supplies
    an evaluation that can be VERIFIED as a supported, separate four-GW Wildcard
    squad evaluation.  A bare mapping (for example
    ``{"four_gw_net_core": 12.0}``) is not evidence of anything and can never
    produce ``PLAY_WILDCARD``: quantitative Wildcard optimisation is NOT
    implemented (``wildcard_quantitative_capability = NOT_SUPPORTED``), so the
    chip can be surfaced for review but never recommended.

    The official same-Gameweek hit rule and the FT-retention transition are
    encoded by the season rules, so the trigger reports the verified rules and
    the deduction the chip would cover.
    """

    resolved_rules = rules or SeasonRules(season="2026/27")
    supported, unsupported_reasons = wildcard_evaluation_is_supported(supported_evaluation)
    hit_rule = wildcard_same_gw_hit(
        prior_paid_transfers=prior_paid_transfers,
        wildcard_transfers=wildcard_transfers,
        rules=resolved_rules,
    )
    try:
        next_ft = chip_free_transfers_after(
            "wildcard", event_start_free_transfers=event_start_free_transfers,
            free_transfers_available=free_transfers_available,
            free_transfers_used=int(wildcard_transfers), rules=resolved_rules,
        )
        ft_transition_status = "SUPPORTED"
    except ValueError as exc:
        next_ft = None
        ft_transition_status = f"EVENT_START_FT_NOT_RECORDED: {exc}"
    signals: list[str] = []
    if int(weak_slot_count) >= int(weak_slot_threshold):
        signals.append(f"WEAK_SQUAD_SLOTS:{int(weak_slot_count)}>={int(weak_slot_threshold)}")
    if int(desired_transfer_count) >= int(desired_transfer_threshold):
        signals.append(f"DESIRED_TRANSFERS:{int(desired_transfer_count)}>={int(desired_transfer_threshold)}")
    if int(expected_hit_burden) >= int(hit_burden_threshold):
        signals.append(f"HIT_BURDEN:{int(expected_hit_burden)}>={int(hit_burden_threshold)}")
    if structural_budget_constraint:
        signals.append("STRUCTURAL_BUDGET_CONSTRAINT")
    if int(availability_problems) > 0:
        signals.append(f"AVAILABILITY_PROBLEMS:{int(availability_problems)}")

    verified_evaluation_signal = None
    if supported:
        status = WILDCARD_EVALUATION_SUPPORTED
        try:
            net_gain = float(supported_evaluation.get("four_gw_net_core", supported_evaluation.get("net_gain", 0.0)))
        except (TypeError, ValueError):
            net_gain = 0.0
        # The verified evaluation IS evidence, but the play rule on top of it is
        # uncalibrated, so it is surfaced as a non-executable signal.  Nothing
        # here may become an executable recommendation until a calibrated
        # four-GW Wildcard evaluator exists.
        verified_evaluation_signal = "POSITIVE" if (math.isfinite(net_gain) and net_gain > 0) else "NEGATIVE"
        recommendation = "NONE"
    elif signals:
        status = WILDCARD_REVIEW_REQUIRED
        recommendation = "NONE"
    else:
        status = WILDCARD_NOT_COMPETITIVE
        recommendation = "NONE"

    return {
        "status": status,
        "recommendation": recommendation,
        "recommendation_basis": (
            "a VERIFIED separate four-GW Wildcard squad evaluation was supplied, so its sign is reported "
            "as a non-executable signal; the play rule is UNCALIBRATED, so the chip remains review-only"
            if supported else
            (
                f"{WILDCARD_INJECTION_REJECTED}: the supplied evaluation is not verifiable "
                f"({'; '.join(unsupported_reasons)}), so the Wildcard is surfaced for review only "
                "— it is never recommended from an unverified mapping"
            )
            if supported_evaluation is not None else
            "no supported separate four-GW Wildcard squad evaluation exists, so the Wildcard is "
            "surfaced for review only — it is never recommended from the trigger alone"
        ),
        "signals": signals,
        "signal_count": len(signals),
        "thresholds": {
            "weak_slot_count": int(weak_slot_threshold),
            "desired_transfer_count": int(desired_transfer_threshold),
            "hit_burden": int(hit_burden_threshold),
        },
        "evidence": {
            "weak_slot_count": int(weak_slot_count),
            "desired_transfer_count": int(desired_transfer_count),
            "expected_hit_burden": int(expected_hit_burden),
            "structural_budget_constraint": bool(structural_budget_constraint),
            "availability_problems": int(availability_problems),
            "prior_paid_transfers": int(prior_paid_transfers),
            "wildcard_transfers": int(wildcard_transfers),
        },
        "supported_evaluation": None if supported_evaluation is None else dict(supported_evaluation),
        "evaluation_verified": bool(supported),
        "evaluation_rejection_reasons": list(unsupported_reasons),
        # An actionable Wildcard is impossible while no quantitative four-GW
        # Wildcard evaluator exists.  A caller-supplied mapping that claimed one
        # is reported here rather than being surfaced as a recommendation.
        "wildcard_quantitative_capability": WILDCARD_QUANTITATIVE_CAPABILITY,
        "overstatement_prevented": bool(supported_evaluation is not None and not supported),
        "calibration_status": WILDCARD_CALIBRATION_STATUS,
        "verified_evaluation_signal": verified_evaluation_signal,
        "provisional_points_estimate": None,
        "hit_rule": WILDCARD_HIT_RULE_TEXT,
        "same_gameweek_hit_rule": hit_rule,
        "free_transfer_transition": {
            "status": ft_transition_status,
            "next_event_free_transfers": next_ft,
            "event_start_free_transfers": event_start_free_transfers,
            "note": "Wildcard/Free Hit retain the saved FT state (no weekly +1); the retained value is the event-start bank",
        },
        "flags": [WILDCARD_NOT_MODELLED_FLAG, WILDCARD_HIT_INTERACTION_FLAG,
                  WILDCARD_REQUIRES_SEPARATE_EVALUATION],
        "hit_interaction_verified": True,
        # Actions are impossible while the play rule is uncalibrated: even a
        # verified evaluation stays non-executable.
        "actionable": False,
        "executable": False,
        "no_recommendation": True,
    }


# ---------------------------------------------------------------------------
# Final decision board contract
# ---------------------------------------------------------------------------


def build_decision_board(
    *,
    routes: Sequence[Mapping[str, Any]],
    decision_events_window: Sequence[int],
    baseline_route_id: str | None = None,
    paired: Mapping[str, Mapping[str, Any]] | None = None,
    readiness: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Structured board rows (correctness over presentation).

    Each row carries the transfer sequence by GW, per-event expected CORE,
    four-GW gross/hits/net, bank and FT after each event, terminal state,
    chip usage, readiness/risk flags and any paired difference vs baseline.
    """

    window = [int(event) for event in decision_events_window]
    board: list[dict[str, Any]] = []
    readiness = dict(readiness or {})
    paired = dict(paired or {})
    for record in routes:
        per_event = {int(item["event"]): item for item in record.get("per_event") or []}
        value = four_gw_window_value(list(per_event.values()), events=window)
        transfers_by_event = {
            str(event): [
                {"out": int(move["out"]), "in": int(move["in"])}
                for move in (per_event.get(event, {}).get("transfers") or [])
            ]
            for event in window
        }
        route_id = str(record.get("route_id"))
        eligibility = route_eligibility(record, decision_events_window=window)
        entry: dict[str, Any] = {
            "route_id": route_id,
            "eligible": eligibility["eligible"],
            "exclusion_reason": None if eligibility["eligible"] else "; ".join(eligibility["reasons"]),
            "events_present": eligibility["events_present"],
            "transfer_sequence_by_gw": transfers_by_event,
            "gw_expected_core": value["event_expected_core"],
            "four_gw_gross_core": value["four_gw_gross_core"],
            "total_hit_points": value["total_hit_points"],
            "four_gw_net_core": value["four_gw_net_core"],
            "bank_after_by_event": [per_event.get(event, {}).get("next_bank_tenths",
                                   per_event.get(event, {}).get("bank_after_tenths")) for event in window],
            "ft_after_by_event": [per_event.get(event, {}).get("next_free_transfers",
                                per_event.get(event, {}).get("ft_after")) for event in window],
            "terminal_bank_tenths": record.get("terminal_bank_tenths"),
            "terminal_ft": record.get("terminal_ft"),
            "chip_used": record.get("chip_used"),
            "readiness": {
                "status": readiness.get("status"),
                "supported_events": readiness.get("supported_events"),
                "blocked_events": readiness.get("blocked_events"),
            },
            "risk_flags": list(record.get("risk_flags") or []),
            "valid": bool(record.get("valid", True)),
            "is_baseline": route_id == baseline_route_id,
            "paired_difference_vs_baseline": paired.get(route_id),
        }
        entry["current_gw_lineup"] = per_event.get(window[0], {}).get("policy")
        board.append(entry)
    return board


# ---------------------------------------------------------------------------
# Top-level decision evaluation
# ---------------------------------------------------------------------------


def evaluate_four_gw_decision(
    *,
    planning_event: int,
    support_by_event: Mapping[int, Mapping[str, Any]],
    cutoff: str | None = None,
    length: int = DECISION_HORIZON_LENGTH,
    last_event: int = SEASON_LAST_EVENT,
    screened_actions: Mapping[str, Any] | None = None,
    routes: Sequence[Mapping[str, Any]] | None = None,
    baseline_route_id: str | None = None,
    paired: Mapping[str, Mapping[str, Any]] | None = None,
    lineup: Mapping[str, Any] | None = None,
    wildcard: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the operational decision under the four-GW product rule.

    When the horizon is incomplete the transfer recommendation is suppressed;
    a valid current-GW lineup/captain block may still be returned, explicitly
    labelled lineup-only.  At season end (fewer than four real events remain)
    the same rule applies over all remaining events.
    """

    horizon = evaluate_horizon(
        planning_event=planning_event, support_by_event=support_by_event, cutoff=cutoff,
        length=length, last_event=last_event,
    )
    events = list(horizon["decision_events"])
    horizon_ok = transfer_recommendation_allowed(horizon)

    board: list[dict[str, Any]] = []
    if routes and horizon_ok:
        board = build_decision_board(
            routes=routes, decision_events_window=events, baseline_route_id=baseline_route_id,
            paired=paired, readiness=horizon,
        )
    eligible_rows = [row for row in board if row.get("eligible")]
    excluded_rows = [
        {"route_id": row["route_id"], "reason": row.get("exclusion_reason")}
        for row in board if not row.get("eligible")
    ]
    ranked: list[dict[str, Any]] = []
    if horizon_ok and eligible_rows:
        ranked = rank_routes_by_four_gw([
            {
                "route_id": row["route_id"],
                "four_gw_net_core": row["four_gw_net_core"],
                "total_hit_points": row["total_hit_points"],
                "terminal_ft": row["terminal_ft"] or 0,
                "terminal_bank_tenths": row["terminal_bank_tenths"] or 0,
            }
            for row in eligible_rows
        ])

    if not horizon_ok:
        transfer_block = {
            "status": RECOMMENDATION_SUPPRESSED,
            "objective": "FOUR_GW_NET_CORE",
            "basis_events": events,
            "ranking": [],
            "preferred_route_id": None,
            "reason": DECISION_HORIZON_INCOMPLETE,
            "blocked_events": horizon["blocked_events"],
            "missing_prediction_events": horizon["missing_prediction_events"],
            "stale_cutoff_events": horizon["stale_cutoff_events"],
            "message": (
                "the four-gameweek transfer horizon is incomplete; a normal transfer "
                "recommendation is not emitted from current-GW-only data"
            ),
        }
    elif not eligible_rows:
        transfer_block = {
            "status": RECOMMENDATION_SUPPRESSED_NO_VALID_ROUTES,
            "objective": "FOUR_GW_NET_CORE",
            "basis_events": events,
            "ranking": [],
            "preferred_route_id": None,
            "reason": "NO_ELIGIBLE_COMPLETE_ROUTE",
            "excluded_routes": excluded_rows,
            "routes_considered": len(board),
            "message": (
                "the predictive horizon is supported, but no route is eligible for transfer ranking: "
                "every candidate is invalid, incomplete across the decision events, or missing terminal accounting"
            ),
        }
    else:
        transfer_block = {
            "status": RECOMMENDATION_AVAILABLE,
            "objective": "FOUR_GW_NET_CORE",
            "basis_events": events,
            "ranking": ranked,
            "preferred_route_id": ranked[0]["route_id"] if ranked else None,
            "eligible_route_count": len(eligible_rows),
            "excluded_routes": excluded_rows,
        }

    lineup_block: dict[str, Any] | None = None
    if lineup is not None:
        lineup_block = {**dict(lineup), "basis": "CURRENT_GW_H1", "basis_events": [events[0]]}
        lineup_block["status"] = lineup_block.get("status") or (LINEUP_ONLY if horizon_ok else LINEUP_ONLY)
        lineup_block["lineup_only"] = True

    if horizon_ok and eligible_rows:
        scope = "season-end short horizon" if horizon["short_horizon"] else "four-Gameweek horizon"
        operator_summary = (
            f"Transfer decision over GW{events[0]}-GW{events[-1]} ({scope}) is supported; "
            f"lineup/captain remain current-GW (GW{events[0]}) decisions."
        )
    elif horizon_ok:
        operator_summary = (
            f"GW{events[0]}-GW{events[-1]} predictive support is complete, but no valid complete route "
            "exists, so no normal transfer recommendation is emitted."
        )
    elif lineup_block is not None:
        operator_summary = (
            f"GW{events[0]} lineup is supported, but the four-Gameweek transfer horizon "
            f"({', '.join(f'GW{event}' for event in events)}) is incomplete; "
            "no normal transfer recommendation is emitted."
        )
    else:
        operator_summary = (
            f"The four-Gameweek transfer horizon ({', '.join(f'GW{event}' for event in events)}) "
            "is incomplete; no normal transfer recommendation is emitted."
        )

    return {
        "planning_event": int(planning_event),
        "planning_cutoff": horizon["planning_cutoff"],
        "operator_summary": operator_summary,
        "horizon": horizon,
        "transfer_decision": {
            "horizon_length": int(length),
            "decision_events": events,
            "current_gw": events[0],
            "primary_objective": "FOUR_GW_NET_CORE",
            "status": transfer_block["status"],
        },
        "transfer_recommendation": transfer_block,
        "lineup_decision": {
            "horizon_length": LINEUP_HORIZON_LENGTH,
            "decision_events": [events[0]],
            "current_gw": events[0],
            "primary_objective": "CURRENT_GW_NET_CORE",
            "status": LINEUP_ONLY,
        },
        "lineup_recommendation": lineup_block,
        "decision_board": board,
        "screened_actions": None if screened_actions is None else dict(screened_actions),
        "wildcard_screen": None if wildcard is None else sanitize_wildcard_screen(wildcard),
        "no_execution": True,
        "no_recommendation": transfer_block["status"] != RECOMMENDATION_AVAILABLE,
    }


def horizon_temporal_boundaries(
    conn: sqlite3.Connection,
    decision_events: Sequence[int],
    *,
    last_event: int | None = None,
) -> dict[str, Any]:
    """Temporal boundaries of the decision window.

    IMPORTANT TIMING RULE: an FPL deadline is at the START of the Gameweek, so the
    final decision event's deadline is NOT the end of that event.  A fixture after
    the GW8 deadline can still belong to GW8.  The upper boundary is therefore
    derived from the FIRST EVENT AFTER THE WINDOW (for GW5-GW8 that is the GW9
    deadline), and kickoff >= that boundary is what proves a fixture outside the
    window.  At season end there is no next event, so no boundary is invented.
    """

    events = [int(e) for e in decision_events]
    if not events:
        return {"start_boundary": None, "end_boundary": None, "boundary_basis": "no_decision_events"}
    start_row = conn.execute(
        "SELECT deadline_time FROM events WHERE id=?", (min(events),)
    ).fetchone()
    start_boundary = str(start_row[0]) if start_row and start_row[0] else None
    next_event = max(events) + 1
    if last_event is not None and next_event > int(last_event):
        return {
            "start_boundary": start_boundary,
            "end_boundary": None,
            "boundary_basis": "SEASON_END_NO_NEXT_EVENT",
            "next_event": None,
        }
    end_row = conn.execute(
        "SELECT deadline_time FROM events WHERE id=?", (next_event,)
    ).fetchone()
    end_boundary = str(end_row[0]) if end_row and end_row[0] else None
    return {
        "start_boundary": start_boundary,
        "end_boundary": end_boundary,
        "boundary_basis": "NEXT_EVENT_DEADLINE_AFTER_WINDOW" if end_boundary else "NEXT_EVENT_MISSING",
        "next_event": next_event,
    }


def classify_fixture_horizon(
    conn: sqlite3.Connection,
    decision_events: Sequence[int],
    *,
    last_event: int | None = None,
) -> dict[str, Any]:
    """Classify every fixture against the exact four-GW decision window.

    ``conn`` must be the immutable certification snapshot.  Blank/DGW semantics are
    TEAM-SPECIFIC: a blank for team T in event E is certified only when the snapshot
    has zero assigned fixtures for T in E AND no unresolved fixture involving T
    could plausibly land in E.  A double Gameweek is two or more DISTINCT assigned
    fixtures for the same team and event; an exact duplicate of the same fixture
    identity is a data-integrity failure, not a DGW.
    """

    events = [int(e) for e in decision_events]
    event_set = set(events)
    bounds = horizon_temporal_boundaries(conn, events, last_event=last_event)
    start_boundary = bounds["start_boundary"]
    end_boundary = bounds["end_boundary"]

    rows = [
        dict(row)
        for row in conn.execute(
            "SELECT id, event, team_h, team_a, kickoff_time, finished, started FROM fixtures ORDER BY id"
        )
    ]

    outside: list[dict[str, Any]] = []
    plausibly_inside: list[dict[str, Any]] = []
    unknown: list[dict[str, Any]] = []
    integrity: list[dict[str, Any]] = []

    seen_rows: dict[tuple, int] = {}
    for row in rows:
        identity = (
            row["team_h"] if row["team_h"] is not None else None,
            row["team_a"] if row["team_a"] is not None else None,
            str(row["kickoff_time"]) if row["kickoff_time"] else None,
        )
        if identity in seen_rows and identity[0] is not None and identity[2] is not None:
            integrity.append(
                {
                    "fixture_id": int(row["id"]),
                    "duplicate_of": seen_rows[identity],
                    "evidence": "identical (team_h, team_a, kickoff_time) fixture identity",
                    "diagnostic": FIXTURE_INTEGRITY_DUPLICATE_IDENTITY,
                }
            )
        seen_rows.setdefault(identity, int(row["id"]))

    # possible_events[team] = events an unresolved fixture involving that team could
    # still be assigned to.  A team blank is NOT certified while this is non-empty.
    possible_events: dict[int, set[int]] = {}
    for row in rows:
        teams = [int(x) for x in (row["team_h"], row["team_a"]) if x is not None]
        event = row["event"]
        kickoff = str(row["kickoff_time"]) if row["kickoff_time"] else None
        record = {
            "fixture_id": int(row["id"]),
            "teams": teams,
            "event": int(event) if event is not None else None,
            "kickoff_time": kickoff,
            "finished": int(row["finished"] or 0),
        }
        if event is not None:
            if int(event) in event_set:
                continue  # assigned inside the window; counted by team_event_states
            record["evidence"] = f"assigned to event {int(event)}, outside {events[0]}-{events[-1]}"
            outside.append(record)
            continue
        if kickoff is None:
            record["evidence"] = "event NULL and kickoff unknown"
            unknown.append(record)
            for team in teams:
                possible_events.setdefault(team, set()).update(events)
            continue
        if start_boundary is not None and kickoff < start_boundary:
            # An FPL deadline is at the START of the Gameweek, so a fixture belonging
            # to a decision event kicks off at or after that event's deadline.  A
            # kickoff strictly before the window's start boundary therefore cannot
            # belong to any decision event.
            record["evidence"] = (
                f"kickoff {kickoff} precedes the window start boundary {start_boundary}; "
                "cannot belong to a decision event"
            )
            outside.append(record)
            continue
        if end_boundary is not None and kickoff >= end_boundary:
            record["evidence"] = (
                f"kickoff {kickoff} is at/after the next-event boundary {end_boundary} "
                f"({bounds['boundary_basis']})"
            )
            outside.append(record)
            continue
        if end_boundary is None:
            record["evidence"] = (
                "no next-event boundary available (season end); cannot prove the fixture outside"
            )
            unknown.append(record)
            for team in teams:
                possible_events.setdefault(team, set()).update(events)
            continue
        record["evidence"] = (
            f"kickoff {kickoff} lies inside the window ({start_boundary} .. {end_boundary})"
        )
        plausibly_inside.append(record)
        for team in teams:
            possible_events.setdefault(team, set()).update(events)

    team_event_counts: dict[tuple[int, int], set[int]] = {}
    for row in rows:
        event = row["event"]
        if event is None or int(event) not in event_set:
            continue
        for team in (row["team_h"], row["team_a"]):
            if team is None:
                continue
            team_event_counts.setdefault((int(team), int(event)), set()).add(int(row["id"]))

    teams = {pair[0] for pair in team_event_counts} | set(possible_events)
    team_event_states: dict[str, dict[str, Any]] = {}
    for team in sorted(teams):
        for event in events:
            distinct = team_event_counts.get((team, event), set())
            unresolved_possible = event in possible_events.get(team, set())
            if len(distinct) >= 2:
                state = CERTIFIED_DOUBLE_OR_MORE  # valid DGW; never blocked for count > 1
            elif len(distinct) == 1:
                state = CERTIFIED_SINGLE
            elif unresolved_possible:
                # zero assigned fixtures is NOT a certified blank while an unresolved
                # fixture involving this team could still land here
                state = PLAUSIBLY_INSIDE
            else:
                state = CERTIFIED_BLANK
            team_event_states[f"{team}:{event}"] = {
                "fixture_count": len(distinct),
                "state": state,
                "unresolved_possible": unresolved_possible,
            }

    contiguous = (
        len(events) == DECISION_HORIZON_LENGTH
        and max(events) - min(events) + 1 == len(events)
    )
    blocking_reasons: list[str] = []
    if plausibly_inside:
        blocking_reasons.append(
            f"{len(plausibly_inside)} unresolved fixture(s) could plausibly fall inside the "
            f"{events[0]}-{events[-1]} window"
        )
    if unknown:
        blocking_reasons.append(f"{len(unknown)} fixture(s) have unknown horizon placement")
    if integrity:
        blocking_reasons.append(f"{len(integrity)} fixture-data integrity failure(s)")
    if not contiguous:
        blocking_reasons.append(f"decision events {events} are not four contiguous events")

    return {
        "decision_events": events,
        "horizon_start_boundary": start_boundary,
        "horizon_end_boundary": end_boundary,
        "boundary_basis": bounds["boundary_basis"],
        "next_event": bounds.get("next_event"),
        "outside": outside,
        "plausibly_inside": plausibly_inside,
        "unknown": unknown,
        "integrity_failures": integrity,
        "team_event_states": team_event_states,
        "blocking_reasons": blocking_reasons,
        "complete": not blocking_reasons,
    }
