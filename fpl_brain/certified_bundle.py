"""Certified predictive bundle / DAG contract (R4A).

Horizon support as previously implemented proved only: same planning event, same
family, same cutoff, ``complete``.  That is insufficient, because predictive
families form a dependency graph:

    minutes ─┐
    team    ─┼─► xpts ─┐
    rates   ─┘         ├─► monte_carlo
                       ┘

A same-cutoff mix of families that do not actually reference one another is not a
certifiable bundle, and a newer same-cutoff rerun of one family must not silently
replace the certified one.

This module makes the bundle explicit: exact run ids per family per event, plus
validation of the dependency edges, the data/code snapshot identity, the model
versions and the planning context.  The decision engine consumes the bundle's
EXACT run ids; it never rediscoveres "latest matching run" per family.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

DIAG_PREDICTIVE_BUNDLE_INCOHERENT = "PREDICTIVE_BUNDLE_INCOHERENT"

#: Reason strings are prefixed with their own token when they are NOT a plain
#: coherence failure, so every refusal names its cause in the text an operator
#: greps, while ``BundleIncoherent`` keeps reporting every incoherence at once.
DIAG_UNSUPPORTED_MODEL_VERSION = "UNSUPPORTED_MODEL_VERSION"
DIAG_EVIDENCE_MISSING = "EVIDENCE_MISSING"
DIAG_CERTIFICATION_BLOCKER_UNRESOLVED = "CERTIFICATION_BLOCKER_UNRESOLVED"

# family -> upstream families its run must reference
BUNDLE_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "minutes_v1": (),
    "team_strength_v1": (),
    "player_rates_v1": (),
    "xpts_v1": ("minutes_v1", "team_strength_v1", "player_rates_v1"),
    "monte_carlo_v1": ("minutes_v1", "team_strength_v1", "player_rates_v1", "xpts_v1"),
}

#: The families a WORLD LOAD actually consumes: the Monte Carlo run is not read by
#: ``route_optimizer.build_event_worlds`` (the simulation is regenerated from these
#: four), so a load requires exactly these to be recorded.  Every family the
#: artifact DOES record is still enforced where it is a dependency of one of these,
#: and the bundle identity covers every family either side declares.
LOAD_REQUIRED_FAMILIES: tuple[str, ...] = (
    "minutes_v1",
    "team_strength_v1",
    "player_rates_v1",
    "xpts_v1",
)

# family -> (columns on its own table carrying the upstream run ids, source table)
_XPTS_UPSTREAM = {
    "minutes_v1": "minutes_run_id",
    "team_strength_v1": "team_run_id",
    "player_rates_v1": "rate_run_id",
}

MC_UPSTREAM = {
    "minutes_v1": "minutes_run_id",
    "team_strength_v1": "team_run_id",
    "player_rates_v1": "rate_run_id",
    "xpts_v1": "xpts_run_id",
}


class BundleIncoherent(RuntimeError):
    """The candidate families do not form one coherent predictive bundle."""

    def __init__(self, reasons: Sequence[str], tokens: Sequence[str] | None = None) -> None:
        reasons = [str(reason) for reason in reasons]
        self.reasons = list(reasons)
        self.tokens = list(tokens) if tokens else reason_tokens(reasons)
        detail = "; ".join(reasons)
        if self.tokens == [DIAG_PREDICTIVE_BUNDLE_INCOHERENT]:
            super().__init__(f"{DIAG_PREDICTIVE_BUNDLE_INCOHERENT}: {detail}")
        else:
            # The specific token is named BESIDE the existing one, so a refusal can
            # never lose the token an established consumer already greps for.
            super().__init__(
                f"{DIAG_PREDICTIVE_BUNDLE_INCOHERENT}: {detail} [{', '.join(self.tokens)}]"
            )


def reason_tokens(reasons: Sequence[str]) -> list[str]:
    """The declared tokens a set of incoherence reasons is reported under.

    A reason carries its own token as a prefix when it is not a plain coherence
    failure; ``PREDICTIVE_BUNDLE_INCOHERENT`` is the token for the rest.  The
    default is therefore always present, and the specific tokens follow it.
    """

    found: list[str] = [DIAG_PREDICTIVE_BUNDLE_INCOHERENT]
    for reason in reasons:
        text = str(reason)
        for token in (
            DIAG_UNSUPPORTED_MODEL_VERSION,
            DIAG_EVIDENCE_MISSING,
            DIAG_CERTIFICATION_BLOCKER_UNRESOLVED,
        ):
            if text.startswith(token) and token not in found:
                found.append(token)
    return found


@dataclass(frozen=True)
class CertifiedBundle:
    """Exact run ids and identities for one event's certified predictions."""

    event: int
    cutoff: str
    runs: Mapping[str, int]
    model_versions: Mapping[str, str]
    code_snapshot_sha256: str | None = None
    data_snapshot_sha256: str | None = None
    planning_context_hash: str | None = None
    required_versions: Mapping[str, str] = field(default_factory=dict)
    #: PE-4's zero-fixture (BLANK) event world.  Reported, and deliberately NOT part
    #: of the identity: it is a reading of the event's grain, not a predictive value.
    zero_fixture_event: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "event": int(self.event),
            "cutoff": self.cutoff,
            "runs": {k: int(v) for k, v in self.runs.items()},
            "model_versions": dict(self.model_versions),
            "code_snapshot_sha256": self.code_snapshot_sha256,
            "data_snapshot_sha256": self.data_snapshot_sha256,
            "planning_context_hash": self.planning_context_hash,
        }

    def bundle_identity(self) -> str:
        """Deterministic identity of this bundle (exact ids + identities)."""

        return canonical_bundle_identity(self.as_dict())


def canonical_bundle_identity(bundle: Mapping[str, Any]) -> str:
    """Identity of a bundle in its persisted ``as_dict`` shape.

    ONE algorithm, shared by the producer (``CertifiedBundle.bundle_identity``) and
    by any consumer that must recompute an identity from stored bytes, so a
    declared ``certified_bundle_identity`` label can be checked against the
    ``certified_bundles`` a decision actually consumes.  Keep in step with
    ``CertifiedBundle.as_dict``: those seven fields ARE the identity.
    """

    canonical = {
        "event": int(bundle["event"]),
        "cutoff": bundle.get("cutoff"),
        "runs": {key: int(value) for key, value in (bundle.get("runs") or {}).items()},
        "model_versions": dict(bundle.get("model_versions") or {}),
        "code_snapshot_sha256": bundle.get("code_snapshot_sha256"),
        "data_snapshot_sha256": bundle.get("data_snapshot_sha256"),
        "planning_context_hash": bundle.get("planning_context_hash"),
    }
    return "sha256:" + hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _run(conn: sqlite3.Connection, run_id: int) -> Mapping[str, Any] | None:
    return conn.execute("SELECT * FROM projection_runs WHERE id=?", (int(run_id),)).fetchone()


def event_fixture_count(conn: sqlite3.Connection, event: int) -> int:
    """How many fixtures this event has at all.

    PE-4's frozen rule: a BLANK event is a valid zero-fixture world, so an empty
    projection run for it is the CORRECT evidence rather than absent evidence.  The
    count decides only which of those two facts a missing dependency edge is, and
    is never part of any certified identity.
    """

    row = conn.execute(
        "SELECT COUNT(*) AS n FROM fixtures WHERE event=?", (int(event),)
    ).fetchone()
    return 0 if row is None else int(row["n"])


def _upstream_run_ids(conn: sqlite3.Connection, family: str, run_id: int) -> dict[str, int | None]:
    """Upstream run ids a family's rows actually reference, if recorded.

    EVERY row of the run is read, not just the first: one run carries the whole
    event's player x fixture evidence, and in a double gameweek that is more than
    one fixture.  A run whose rows disagree about which upstream run they were built
    from is not one predictive world, so a fixture is certified or refused as a unit
    rather than on whichever row happened to be read first.
    """

    if family == "xpts_v1":
        rows = conn.execute(
            "SELECT DISTINCT minutes_run_id, team_run_id, rate_run_id"
            " FROM player_fixture_xpts_projections WHERE projection_run_id=?",
            (int(run_id),),
        ).fetchall()
        if not rows:
            return {}
        if len(rows) > 1:
            raise BundleIncoherent(
                [
                    f"xpts_v1 run {run_id} was built from {len(rows)} different upstream combination(s) "
                    f"{sorted(str(tuple(row)) for row in rows)}; a fixture is certified or refused as a unit"
                ]
            )
        row = rows[0]
        return {
            "minutes_v1": row["minutes_run_id"],
            "team_strength_v1": row["team_run_id"],
            "player_rates_v1": row["rate_run_id"],
        }
    if family == "monte_carlo_v1":
        rows = conn.execute(
            "SELECT DISTINCT minutes_run_id, team_run_id, rate_run_id, xpts_run_id"
            " FROM monte_carlo_distributions WHERE projection_run_id=?",
            (int(run_id),),
        ).fetchall()
        if not rows:
            return {}
        if len(rows) > 1:
            raise BundleIncoherent(
                [
                    f"monte_carlo_v1 run {run_id} was built from {len(rows)} different upstream "
                    f"combination(s) {sorted(str(tuple(row)) for row in rows)}; a fixture is certified or "
                    "refused as a unit"
                ]
            )
        row = rows[0]
        return {
            "minutes_v1": row["minutes_run_id"],
            "team_strength_v1": row["team_run_id"],
            "player_rates_v1": row["rate_run_id"],
            "xpts_v1": row["xpts_run_id"],
        }
    return {}


def validate_certified_bundle(
    conn: sqlite3.Connection,
    *,
    event: int,
    cutoff: str,
    runs: Mapping[str, int],
    required_versions: Mapping[str, str] | None = None,
    data_snapshot_sha256: str | None = None,
    code_snapshot_sha256: str | None = None,
    expected_data_snapshot_sha256: str | None = None,
    families: Sequence[str] | None = None,
) -> CertifiedBundle:
    """Validate one event's candidate families as a coherent predictive bundle.

    Raises :class:`BundleIncoherent` with every reason found, so a caller can
    report all incoherences at once instead of the first.

    ``expected_data_snapshot_sha256`` is the certification artifact's own data
    snapshot identity.  A bundle that DECLARES a different one describes a
    different predictive world and is incoherent; the identity is validated here
    rather than merely recorded.

    ``families`` is the set of families this validation must cover.  The default
    is the whole declared graph.  The consumer boundary passes the families it
    actually LOADS together with the families the certification RECORDED for the
    event, so the recorded closure is re-proven without inventing a requirement
    the load does not have -- a load that reads four families must not be
    authorised by an artifact that recorded three, and an artifact that records a
    family whose rows are incoherent is refused wherever it is named.
    """

    families = tuple(families) if families else tuple(BUNDLE_DEPENDENCIES)

    reasons: list[str] = []
    versions: dict[str, str] = {}
    context_hashes: set[str] = set()
    code_snapshots: set[str] = set()
    cutoffs: set[str] = set()

    for family in families:
        run_id = runs.get(family)
        if run_id is None:
            reasons.append(f"missing family {family}")
            continue
        row = _run(conn, int(run_id))
        if row is None:
            reasons.append(f"run {run_id} for {family} does not exist")
            continue
        if str(row["model_family"]) != family:
            reasons.append(
                f"run {run_id} is family {row['model_family']!r}, expected {family!r}"
            )
        if int(row["planning_event"]) != int(event):
            reasons.append(
                f"run {run_id} ({family}) is planning_event {row['planning_event']}, expected {event}"
            )
        if str(row["status"]) != "complete":
            reasons.append(f"run {run_id} ({family}) status is {row['status']!r}, not complete")
        if str(row["data_cutoff"]) != str(cutoff):
            reasons.append(
                f"run {run_id} ({family}) data_cutoff {row['data_cutoff']} != bundle cutoff {cutoff}"
            )
        cutoffs.add(str(row["data_cutoff"]))
        versions[family] = str(row["model_version"])
        if row["planning_context_hash"]:
            context_hashes.add(str(row["planning_context_hash"]))
        # projection_runs.source_snapshot_sha256 is the CODE fingerprint, not a data
        # identity; it is checked for cross-family consistency and against an
        # explicitly supplied code_snapshot_sha256, never against a DATA snapshot.
        if row["source_snapshot_sha256"]:
            code_snapshots.add(str(row["source_snapshot_sha256"]))

    # Dependency edges: xPts and Monte Carlo must reference THIS bundle's runs.
    blank_event = event_fixture_count(conn, int(event)) == 0
    for family in ("xpts_v1", "monte_carlo_v1"):
        run_id = runs.get(family)
        if run_id is None:
            continue
        upstream = _upstream_run_ids(conn, family, int(run_id))
        if not upstream:
            if blank_event:
                # A BLANK event has no fixtures, so its xPts / Monte Carlo runs
                # correctly carry no rows and therefore record no dependency edge.
                # PE-4: a blank is not missing data and must not be refused as if a
                # projection were absent, nor fabricated into rows.
                continue
            reasons.append(
                f"{family} run {run_id} records no upstream run ids; its dependency edges cannot be verified"
            )
            continue
        for dep_family, column in (
            _XPTS_UPSTREAM.items() if family == "xpts_v1" else MC_UPSTREAM.items()
        ):
            expected = runs.get(dep_family)
            actual = upstream.get(dep_family)
            if expected is None:
                continue
            if actual is None:
                reasons.append(f"{family} run {run_id} has no {column}; expected {expected}")
            elif int(actual) != int(expected):
                reasons.append(
                    f"{family} run {run_id} references {dep_family} run {actual}, "
                    f"but the bundle declares {expected}"
                )

    if required_versions:
        for family, wanted in required_versions.items():
            got = versions.get(family)
            if got is not None and got != wanted:
                reasons.append(
                    f"{DIAG_UNSUPPORTED_MODEL_VERSION}: {family} model_version {got!r} != "
                    f"required {wanted!r}"
                )

    # The data snapshot identity is VALIDATED where the bundle declares one: two
    # different snapshot identities are two different predictive worlds.
    if (
        expected_data_snapshot_sha256 is not None
        and data_snapshot_sha256 is not None
        and str(data_snapshot_sha256) != str(expected_data_snapshot_sha256)
    ):
        reasons.append(
            f"bundle data_snapshot_sha256 {data_snapshot_sha256} != certification artifact "
            f"{expected_data_snapshot_sha256}"
        )

    if len(context_hashes) > 1:
        reasons.append(f"planning_context_hash differs across families: {sorted(context_hashes)}")

    if code_snapshot_sha256 is not None and code_snapshots:
        if code_snapshots != {str(code_snapshot_sha256)}:
            reasons.append(
                f"family code snapshots {sorted(code_snapshots)} != bundle code snapshot "
                f"{code_snapshot_sha256}"
            )
    elif len(code_snapshots) > 1:
        reasons.append(
            f"code snapshot differs across families (all must come from one code revision): "
            f"{sorted(code_snapshots)}"
        )
    # A DATA snapshot identity is recorded by the certification artifact (and carried
    # on the bundle for the decision engine); it is deliberately not compared against
    # the code fingerprint column.

    if reasons:
        raise BundleIncoherent(reasons)

    return CertifiedBundle(
        event=int(event),
        cutoff=str(cutoff),
        runs={k: int(v) for k, v in runs.items() if k in BUNDLE_DEPENDENCIES},
        model_versions=versions,
        code_snapshot_sha256=code_snapshot_sha256,
        data_snapshot_sha256=data_snapshot_sha256,
        planning_context_hash=next(iter(context_hashes)) if len(context_hashes) == 1 else None,
        required_versions=dict(required_versions or {}),
        zero_fixture_event=blank_event,
    )


def certified_bundle_from_explicit_ids(
    conn: sqlite3.Connection,
    *,
    event: int,
    cutoff: str,
    runs: Mapping[str, int],
    data_snapshot_sha256: str | None = None,
    code_snapshot_sha256: str | None = None,
    required_versions: Mapping[str, str] | None = None,
    expected_data_snapshot_sha256: str | None = None,
) -> CertifiedBundle:
    """Certify an explicitly supplied bundle (the decision engine's entry point).

    There is deliberately no "discover the latest run per family" helper here:
    silent rediscovery is exactly the failure mode this module exists to prevent.
    """

    return validate_certified_bundle(
        conn,
        event=int(event),
        cutoff=str(cutoff),
        runs=runs,
        required_versions=required_versions,
        data_snapshot_sha256=data_snapshot_sha256,
        code_snapshot_sha256=code_snapshot_sha256,
        expected_data_snapshot_sha256=expected_data_snapshot_sha256,
    )


def certify_horizon_bundles(
    conn: sqlite3.Connection,
    *,
    events: Sequence[int],
    cutoff: str,
    runs_by_event: Mapping[int, Mapping[str, int]],
    data_snapshot_sha256: str | None = None,
    required_versions: Mapping[str, str] | None = None,
    expected_data_snapshot_sha256: str | None = None,
) -> dict[int, CertifiedBundle]:
    """Certify every event's bundle, raising on the first incoherent event."""

    certified: dict[int, CertifiedBundle] = {}
    failures: dict[int, list[str]] = {}
    for event in events:
        try:
            certified[int(event)] = certified_bundle_from_explicit_ids(
                conn,
                event=int(event),
                cutoff=cutoff,
                runs=runs_by_event[int(event)],
                data_snapshot_sha256=data_snapshot_sha256,
                required_versions=required_versions,
                expected_data_snapshot_sha256=expected_data_snapshot_sha256,
            )
        except BundleIncoherent as exc:
            failures[int(event)] = exc.reasons
    if failures:
        flat = [f"GW{event}: {reason}" for event, reasons in sorted(failures.items()) for reason in reasons]
        raise BundleIncoherent(flat)
    return certified


# ===========================================================================
# PE-9 — certification integration
# ===========================================================================
#
# PE-9 decides which prediction artifacts may enter the CERTIFIED PREDICTION
# BUNDLE and therefore become consumable by downstream FPL decision logic.  It
# changes no predictive quantity: every model and every calibration stays frozen
# and authoritative, nothing is promoted, and no model or calibration version is
# bumped.
#
# The vocabulary below is deliberately small and fail-closed.  A state is NEVER
# collapsed into ``certified: true/false``, because insufficient evidence and
# incoherence are different facts, and a legitimate non-failure must not be
# reported as one.

# --- per-bundle states ------------------------------------------------------

#: Every structural gate passes and every required evidence artifact is present.
STATE_CERTIFIED_COHERENT = "CERTIFIED_COHERENT"
#: Structurally valid, but the evidence cannot support a calibration CLAIM.
#: This is NOT a failure and is never reported as one.
STATE_EVIDENCE_LIMITED = "EVIDENCE_LIMITED"
#: The surface has no calibration question (for example an identity mapping).
#: Explicitly not a defect.
STATE_CALIBRATION_NOT_APPLICABLE = "CALIBRATION_NOT_APPLICABLE"
#: A required artifact or field is absent.  A refusal, never a default.
STATE_EVIDENCE_MISSING = DIAG_EVIDENCE_MISSING
#: Code / context / dependency identity disagreement (the existing token).
STATE_PREDICTIVE_BUNDLE_INCOHERENT = DIAG_PREDICTIVE_BUNDLE_INCOHERENT
#: A family's model version is not the expected one.
STATE_UNSUPPORTED_MODEL_VERSION = DIAG_UNSUPPORTED_MODEL_VERSION
#: A disclosed limitation blocks the claim being made.
STATE_CERTIFICATION_BLOCKER_UNRESOLVED = DIAG_CERTIFICATION_BLOCKER_UNRESOLVED

PER_BUNDLE_STATES: tuple[str, ...] = (
    STATE_CERTIFIED_COHERENT,
    STATE_EVIDENCE_LIMITED,
    STATE_CALIBRATION_NOT_APPLICABLE,
    STATE_EVIDENCE_MISSING,
    STATE_PREDICTIVE_BUNDLE_INCOHERENT,
    STATE_UNSUPPORTED_MODEL_VERSION,
    STATE_CERTIFICATION_BLOCKER_UNRESOLVED,
)

#: States that REFUSE the bundle.  ``EVIDENCE_MISSING``,
#: ``PREDICTIVE_BUNDLE_INCOHERENT`` and ``UNSUPPORTED_MODEL_VERSION`` are refusal
#: tokens; ``CERTIFICATION_BLOCKER_UNRESOLVED`` blocks the CLAIM being made.
BLOCKING_BUNDLE_STATES: tuple[str, ...] = (
    STATE_PREDICTIVE_BUNDLE_INCOHERENT,
    STATE_UNSUPPORTED_MODEL_VERSION,
    STATE_EVIDENCE_MISSING,
    STATE_CERTIFICATION_BLOCKER_UNRESOLVED,
)

#: The declared, fail-closed precedence one bundle's single state is rolled up
#: under.  The most blocking fact wins, so a roll-up can never hide a refusal
#: behind a legitimate non-failure.
BUNDLE_STATE_PRECEDENCE: tuple[str, ...] = (
    STATE_PREDICTIVE_BUNDLE_INCOHERENT,
    STATE_UNSUPPORTED_MODEL_VERSION,
    STATE_EVIDENCE_MISSING,
    STATE_CERTIFICATION_BLOCKER_UNRESOLVED,
    STATE_EVIDENCE_LIMITED,
    STATE_CALIBRATION_NOT_APPLICABLE,
    STATE_CERTIFIED_COHERENT,
)

#: The LEGITIMATE, distinct, NON-FAILURE states.  PE-8 established that
#: ``NOT_FITTED``, ``NO_MATERIAL_DEFECT_DETECTED`` and
#: ``INSUFFICIENT_FOR_DIAGNOSIS`` are real outcomes rather than defects, and
#: these states carry that meaning into certification.
NON_FAILURE_BUNDLE_STATES: tuple[str, ...] = (
    STATE_CERTIFIED_COHERENT,
    STATE_EVIDENCE_LIMITED,
    STATE_CALIBRATION_NOT_APPLICABLE,
)

# --- fail-closed diagnostics ------------------------------------------------


class CertificationRefused(RuntimeError):
    """A PE-9 certification gate refused a bundle, a horizon or a decision.

    EVERY refusal carries a specific token, so "absent", "contradictory",
    "unknown identity" and "incoherent" can never be confused with one another.
    """

    def __init__(self, token: str, reasons: Sequence[str]) -> None:
        self.token = str(token)
        self.reasons = [str(reason) for reason in reasons]
        super().__init__(f"{self.token}: " + "; ".join(self.reasons))


DIAG_CERTIFICATION_ARTIFACT_ABSENT = "CERTIFICATION_ARTIFACT_ABSENT"
DIAG_CERTIFICATION_ARTIFACT_CONTRADICTORY = "CERTIFICATION_ARTIFACT_CONTRADICTORY"
DIAG_CERTIFICATION_ARTIFACT_UNVALIDATED = "CERTIFICATION_ARTIFACT_UNVALIDATED"
DIAG_CERTIFICATION_ARTIFACT_MUTATED = "CERTIFICATION_ARTIFACT_MUTATED"
DIAG_CALIBRATION_IDENTITY_UNKNOWN = "CALIBRATION_IDENTITY_UNKNOWN"
DIAG_CALIBRATION_WORLD_MISMATCH = "CALIBRATION_WORLD_MISMATCH"
DIAG_CERTIFICATION_BOUNDARY_REQUIRED = "CERTIFICATION_BOUNDARY_REQUIRED"


class CertificationArtifactAbsent(CertificationRefused):
    def __init__(self, reasons: Sequence[str]) -> None:
        super().__init__(DIAG_CERTIFICATION_ARTIFACT_ABSENT, reasons)


class CertificationArtifactContradictory(CertificationRefused):
    def __init__(self, reasons: Sequence[str]) -> None:
        super().__init__(DIAG_CERTIFICATION_ARTIFACT_CONTRADICTORY, reasons)


class CertificationArtifactUnvalidated(CertificationRefused):
    """No AUTHORISATION: the mapping was never validated by the canonical loader.

    A third fact beside absent and contradictory.  An artifact-shaped mapping that
    carries self-consistent bundles but not the fields the loader's contract
    requires (the schema, the causal and dependency status, the authorisation flag,
    the completeness audit, the wiring identity, an identity that binds its own
    bundles) is not an authorisation to load predictive data, however coherent its
    own run ids are.
    """

    def __init__(self, reasons: Sequence[str]) -> None:
        super().__init__(DIAG_CERTIFICATION_ARTIFACT_UNVALIDATED, reasons)


class CertificationArtifactMutated(CertificationArtifactContradictory):
    """A VALIDATED artifact whose bytes no longer digest to its minted identity.

    A fourth fact beside absent, contradictory and unvalidated: this value WAS
    validated and was then edited.  It is a subclass of the contradictory family -- a
    mutated authorisation contradicts its own validated bytes -- so every consumer that
    already refuses a contradiction keeps refusing, while a consumer that wants to name
    the tamper has this token.
    """

    def __init__(self, reasons: Sequence[str]) -> None:
        CertificationRefused.__init__(self, DIAG_CERTIFICATION_ARTIFACT_MUTATED, reasons)


# --- phase terminal state ---------------------------------------------------

PHASE_READY_FOR_MERGE = "READY_FOR_MERGE"
PHASE_OPEN = "OPEN"
PHASE_TERMINAL_STATES: tuple[str, ...] = (PHASE_READY_FOR_MERGE, PHASE_OPEN)

#: The PE-3 limitation that stays CARRIED, not resolved.  PE-9 must not present
#: any figure as if this limitation did not apply, so certification discloses it
#: unconditionally -- including for a bundle whose calibration evidence is clean.
CONTINUOUS_PROXY_TIE_LIMITATION = "CONTINUOUS_PROXY_TIE_LIMITATION"
DISCLOSED_LIMITATIONS: tuple[str, ...] = (CONTINUOUS_PROXY_TIE_LIMITATION,)


def declared_required_versions() -> dict[str, str]:
    """The ONE declared source of the model versions a certification requires.

    Every certification call site reads the expected version of each family from
    here and nowhere else, so a run produced by an unexpected model version fails
    with ``UNSUPPORTED_MODEL_VERSION`` instead of certifying silently.  The values
    are the FROZEN accepted constants of the family modules -- this reads them, it
    never restates them.
    """

    from . import minutes_model, monte_carlo, player_rates, team_model, xpts

    return {
        "minutes_v1": str(minutes_model.MINUTES_MODEL_VERSION),
        "team_strength_v1": str(team_model.TEAM_MODEL_VERSION),
        "player_rates_v1": str(player_rates.PLAYER_RATE_MODEL_VERSION),
        "xpts_v1": str(xpts.XPTS_MODEL_VERSION),
        "monte_carlo_v1": str(monte_carlo.MONTE_CARLO_MODEL_VERSION),
    }


def bundle_state_from_reasons(reasons: Sequence[str]) -> str:
    """The ONE per-bundle state a set of refusal reasons rolls up to.

    The declared precedence is scanned in order, so the most blocking fact wins
    and a refusal can never be hidden behind a legitimate non-failure.  Reasons
    that carry no specific token are a plain coherence failure, which is the
    existing ``PREDICTIVE_BUNDLE_INCOHERENT`` token.
    """

    if not reasons:
        return STATE_CERTIFIED_COHERENT
    for state in BUNDLE_STATE_PRECEDENCE:
        if state == STATE_PREDICTIVE_BUNDLE_INCOHERENT:
            continue  # the default for any reason, not a discriminating prefix
        if any(str(reason).startswith(state) for reason in reasons):
            return state
    return STATE_PREDICTIVE_BUNDLE_INCOHERENT


# ---------------------------------------------------------------------------
# Calibration evidence gates
# ---------------------------------------------------------------------------
#
# Certification may CONSULT the PE-8 calibration artifact, but the separation is
# absolute:
#
# * a STRUCTURAL gate may fail a bundle (coherence, identity, version);
# * an EVIDENCE gate may NOT fail a bundle for insufficiency -- insufficient
#   evidence yields ``EVIDENCE_LIMITED``, never a refusal and never a pass;
# * no NEW numerical threshold is introduced: PE-9 reuses PE-8's declared
#   tolerances rather than inventing materiality of its own;
# * a calibration identity must be KNOWN and DECLARED (an unknown or mismatched
#   identity fails closed, an absent one where one is required fails closed, and
#   ``NOT_FITTED`` is distinct from both);
# * evidence belonging to a different PREDICTION WORLD is refused.

#: The identity block PE-9 reads off a PE-8 artifact.  It is a pure function of
#: fields PE-8 itself declares, so a consumer verifies it without any code-drift
#: assumption -- the same design as ``certification_identity_of``.
CALIBRATION_EVIDENCE_SCHEMA = "fpl_brain.pe9_calibration_evidence.v1"


def calibration_tokens() -> dict[str, str]:
    """PE-8's declared outcome tokens, READ from PE-8 rather than restated.

    Certification decides WHICH certification state a PE-8 outcome maps to.  It
    never decides what PE-8's outcomes ARE, and it never re-spells them: a token
    restated here would silently become a different token the day PE-8 renamed it,
    and the branch that recognises the outcome would quietly stop matching.  That
    is a fail-OPEN drift -- an unrecognised outcome would read as "no diagnosis"
    instead of as the diagnosis PE-8 actually made -- so the tokens are read from
    their declaring modules on every call.
    """

    from . import calibration_evaluation as ce
    from . import walk_forward_scoreboard as sb

    return {
        "miscalibrated": str(ce.DIAGNOSIS_MISCALIBRATED),
        "no_material_defect": str(ce.DIAGNOSIS_NO_MATERIAL_DEFECT),
        "insufficient": str(ce.DIAGNOSIS_INSUFFICIENT),
        "not_fitted": str(ce.STATUS_NOT_FITTED),
        "unreachable": str(ce.STATUS_UNREACHABLE),
        "population_mismatch": str(ce.STATUS_POPULATION_MISMATCH),
        "descriptive_only": str(sb.SAMPLE_DESCRIPTIVE_ONLY),
        "sample_insufficient": str(sb.SAMPLE_INSUFFICIENT),
    }


#: The calibration artifact's identity fields.  An artifact naming anything not
#: known to this engine fails closed with ``CALIBRATION_IDENTITY_UNKNOWN``.
CALIBRATION_IDENTITY_FIELDS = ("schema", "evaluation_version", "certification_identity", "planning_cutoff")


def known_calibration_versions() -> dict[str, str]:
    """The PE-8 schema / evaluation versions this engine knows, read from PE-8."""

    from . import calibration_evaluation as ce

    return {"schema": str(ce.PE8_SCHEMA_VERSION), "evaluation_version": str(ce.PE8_EVALUATION_VERSION)}


def calibration_identity(calibration: Mapping[str, Any]) -> str | None:
    """The declared identity of a PE-8 calibration artifact, or None.

    ``None`` means the artifact declares no identity at all, which is a DIFFERENT
    fact from an identity that is unknown or that contradicts the bundle.
    """

    identity = calibration.get("identity")
    if not isinstance(identity, Mapping):
        return None
    declared = {
        "schema": calibration.get("schema"),
        "evaluation_version": calibration.get("evaluation_version"),
        "certification_identity": identity.get("certification_identity"),
        "planning_cutoff": identity.get("planning_cutoff"),
    }
    if not declared["schema"] or not declared["certification_identity"]:
        return None
    return "sha256:" + hashlib.sha256(
        json.dumps(
            {"schema": CALIBRATION_EVIDENCE_SCHEMA, **declared},
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode()
    ).hexdigest()


def _calibration_per_event_runs(calibration: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    identity = calibration.get("identity") or {}
    runs = identity.get("per_event_runs") or {}
    resolved: dict[int, Mapping[str, Any]] = {}
    for event, record in (runs.items() if isinstance(runs, Mapping) else ()):
        try:
            resolved[int(event)] = record if isinstance(record, Mapping) else {}
        except (TypeError, ValueError):
            continue
    return resolved


def validate_calibration_evidence(
    *,
    calibration: Mapping[str, Any] | None,
    event: int,
    cutoff: str,
    runs: Mapping[str, int],
    certification_identity: str | None,
    require_calibration: bool = False,
) -> dict[str, Any]:
    """Bind a PE-8 calibration artifact to the exact bundle it would certify.

    Fails closed when the artifact is REQUIRED and absent, when its identity is
    unknown to this engine, or when its anchor / cutoff / run ids belong to a
    different predictive world.  It NEVER fails because the evidence is
    insufficient: insufficiency is reported as a state, not a refusal.
    """

    if calibration is None:
        if require_calibration:
            raise CertificationRefused(
                STATE_EVIDENCE_MISSING,
                [
                    "a required PE-8 calibration evidence artifact is absent; an unevidenced "
                    "calibration claim is not certified"
                ],
            )
        return {
            "consulted": False,
            "identity": None,
            "certification_identity": None,
            "world_bound": None,
            "declared": False,
        }

    known = known_calibration_versions()
    declared_schema = str(calibration.get("schema") or "")
    declared_evaluation = str(calibration.get("evaluation_version") or "")
    identity_block = calibration.get("identity") or {}
    declared_certification = str(identity_block.get("certification_identity") or "")
    declared_cutoff = str(identity_block.get("planning_cutoff") or "")

    identity = calibration_identity(calibration)
    if identity is None:
        absent = [
            name
            for name in CALIBRATION_IDENTITY_FIELDS
            if not (
                calibration.get(name)
                if name in {"schema", "evaluation_version"}
                else (calibration.get("identity") or {}).get(name)
            )
        ]
        raise CertificationRefused(
            STATE_EVIDENCE_MISSING,
            [
                "the calibration artifact declares no identity; the absent field(s) "
                f"{sorted(absent)} are required, and an unattributable artifact cannot certify a "
                "bundle"
            ],
        )
    if declared_schema != known["schema"] or declared_evaluation != known["evaluation_version"]:
        raise CertificationRefused(
            DIAG_CALIBRATION_IDENTITY_UNKNOWN,
            [
                f"the calibration artifact declares schema {declared_schema!r} / evaluation "
                f"{declared_evaluation!r}, which is not the known PE-8 identity "
                f"{known['schema']!r} / {known['evaluation_version']!r}"
            ],
        )

    # The evidence must describe THIS bundle's predictive world, by exact identity.
    mismatches: list[str] = []
    if certification_identity is not None and declared_certification != str(certification_identity):
        mismatches.append(
            f"calibration certification identity {declared_certification or '<none>'} != "
            f"certified bundle identity {certification_identity}"
        )
    if declared_cutoff and declared_cutoff != str(cutoff):
        mismatches.append(f"calibration cutoff {declared_cutoff} != bundle cutoff {cutoff}")
    per_event_runs = _calibration_per_event_runs(calibration)
    event_runs = per_event_runs.get(int(event))
    if event_runs is not None:
        # PE-8 declares ``per_event_runs[event]`` keyed by MODEL FAMILY
        # (``xpts_v1`` / ``monte_carlo_v1``).  Reading some other spelling would
        # silently find nothing and let evidence from a different prediction world
        # certify this bundle, so the declared family names are what is read; the
        # explicit ``*_run_id`` spelling is accepted alongside them because the
        # binding, not the key name, is what the check is about.
        for family in ("xpts_v1", "monte_carlo_v1"):
            declared = event_runs.get(family)
            if declared is None:
                declared = event_runs.get(f"{family}_run_id")
            if declared is None:
                continue
            bundle_run = runs.get(family)
            if bundle_run is None:
                continue
            if int(declared) != int(bundle_run):
                mismatches.append(
                    f"calibration evidence scored {family} run {int(declared)} for GW{int(event)}, "
                    f"but the bundle is certified on run {int(bundle_run)}"
                )
    if mismatches:
        raise CertificationRefused(DIAG_CALIBRATION_WORLD_MISMATCH, mismatches)

    return {
        "consulted": True,
        "identity": identity,
        "certification_identity": declared_certification or None,
        "planning_cutoff": declared_cutoff or None,
        "declared_schema": declared_schema,
        "declared_evaluation_version": declared_evaluation,
        # PE-8's OWN terminal state travels with the reading, so a consumer can see
        # that certification did not upgrade the claim PE-8 declined to make.
        "pe8_terminal_state": calibration_terminal_state(calibration)["state"],
        "pe8_terminal_reasons": calibration_terminal_state(calibration)["reasons"],
        "world_bound": True,
        "declared": True,
    }


def _calibration_surface_state(surface: Mapping[str, Any]) -> dict[str, Any]:
    """One PE-8 probability surface's certification reading.

    The mapping is a READING, not a judgement: PE-8 already diagnosed, and the
    only thing decided here is whether the surface's evidence can support a
    calibration claim (``EVIDENCE_LIMITED``), whether it has no calibration
    question at all (``CALIBRATION_NOT_APPLICABLE``), whether a disclosed defect
    blocks the claim (``CERTIFICATION_BLOCKER_UNRESOLVED``) or whether it is clean.

    ``NOT_FITTED``, ``NO_MATERIAL_DEFECT_DETECTED`` and
    ``INSUFFICIENT_FOR_DIAGNOSIS`` are DISTINCT legitimate outcomes, so each is
    recognised by its own token and none of them is treated as incoherence.
    """

    tokens = calibration_tokens()
    metric = str(surface.get("metric") or "<surface>")
    status = str(surface.get("status") or "")
    diagnosis = str((surface.get("diagnosis") or {}).get("status") or "")
    sample = str(((surface.get("sample") or {}).get("sample_interpretation")) or "")
    challenger = str((surface.get("causal_challenger") or {}).get("status") or "")
    scored = int((surface.get("population") or {}).get("scored") or 0)
    observations = int(((surface.get("sample") or {}).get("observations")) or 0)
    not_fitted = challenger == tokens["not_fitted"]
    reasons: list[str] = []
    if status in {tokens["unreachable"], tokens["population_mismatch"]}:
        # PE-8 could not reach this surface, or could not match its population.
        # Its evidence is therefore absent for a CLAIM -- which is not a defect.
        state = STATE_EVIDENCE_LIMITED
        reasons.append(f"{metric}: PE-8 reports {status}, so no calibration claim is supported here")
    elif scored == 0 or observations == 0:
        state = STATE_EVIDENCE_LIMITED
        reasons.append(f"{metric}: no scored observation on this surface")
    elif diagnosis == tokens["insufficient"] or sample == tokens["sample_insufficient"]:
        # PE-8's OWN semantics, reused rather than reinterpreted: it lists a sample
        # that is NOT ``SUFFICIENT_FOR_DESCRIPTIVE_REPORTING_ONLY`` as a reason its
        # terminal state stays OPEN.  A descriptive-only sample is therefore the
        # acceptable sample verdict -- it is what PE-8 asks for -- and it does not by
        # itself withhold the diagnosis.
        state = STATE_EVIDENCE_LIMITED
        reasons.append(
            f"{metric}: the evidence is insufficient for a calibration claim "
            f"(diagnosis {diagnosis or '<none>'}, sample {sample or '<none>'})"
        )
    elif diagnosis == tokens["miscalibrated"]:
        state = STATE_CERTIFICATION_BLOCKER_UNRESOLVED
        reasons.append(
            f"{metric}: a material calibration gap is diagnosed and PE-8 promoted nothing; "
            f"the causal transform is {challenger or '<none>'}"
        )
    elif diagnosis == tokens["no_material_defect"]:
        # A clean diagnosis is a diagnosis.  A transform that could not be fitted
        # does not change that, and is reported beside it rather than as a failure.
        state = STATE_CERTIFIED_COHERENT
        reasons.append(
            f"{metric}: no material defect is detected within the declared tolerance"
            + (
                "; no causal transform is fitted (NOT_FITTED), which is a legitimate outcome"
                if not_fitted
                else ""
            )
        )
    else:
        state = STATE_EVIDENCE_LIMITED
        reasons.append(
            f"{metric}: PE-8 reports no diagnosis this certification recognises "
            f"(status {status or '<none>'}, diagnosis {diagnosis or '<none>'})"
        )
    return {
        "surface": metric,
        "status": status or None,
        "state": state,
        "diagnosis": diagnosis or None,
        "sample_interpretation": sample or None,
        "causal_challenger": challenger or None,
        "not_fitted": not_fitted,
        "scored": scored,
        "reasons": reasons,
    }


def calibration_surface_states(calibration: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Every declared PE-8 surface's certification reading, plus the declared
    non-applicable surfaces reported rather than silently skipped."""

    surfaces = ((calibration.get("probability_calibration") or {}).get("surfaces")) or []
    readings = [_calibration_surface_state(surface) for surface in surfaces]
    for entry in calibration.get("excluded_surfaces") or []:
        if not isinstance(entry, Mapping):
            continue
        readings.append(
            {
                "surface": str(entry.get("surface") or "<excluded>"),
                "state": STATE_CALIBRATION_NOT_APPLICABLE,
                "diagnosis": None,
                "sample_interpretation": None,
                "causal_challenger": None,
                "not_fitted": False,
                "scored": 0,
                "reasons": [
                    f"{entry.get('surface')}: no calibration question is asked on this surface "
                    f"({entry.get('reason')})"
                ],
            }
        )
    return readings


def calibration_terminal_state(calibration: Mapping[str, Any]) -> dict[str, Any]:
    """PE-8's OWN declared terminal state, read rather than re-derived.

    PE-8 declares ``READY_FOR_MERGE`` or ``OPEN`` and says ``OPEN`` is not a failure:
    it is the correct state when the diagnosis is supported but the evidence does not
    support a promotion, or when the sample supports nothing beyond a description.
    Certification must not UPGRADE that: a PE-9 state that claimed more than the
    artifact it consulted would be asserting a calibration claim PE-8 explicitly did
    not make.
    """

    declared = calibration.get("terminal_state")
    if not isinstance(declared, Mapping):
        return {"state": None, "reasons": [], "declared": False}
    return {
        "state": str(declared.get("state") or "") or None,
        "reasons": [str(reason) for reason in (declared.get("reasons") or [])],
        "promotion_performed": bool(declared.get("promotion_performed")),
        "declared": True,
    }


def calibration_evidence_state(calibration: Mapping[str, Any], readings: Sequence[Mapping[str, Any]]) -> str:
    """Roll a consulted calibration artifact up to ONE non-refusal state.

    ``EVIDENCE_LIMITED`` and ``CALIBRATION_NOT_APPLICABLE`` are NOT failures and
    are never reported as such.  ``CERTIFICATION_BLOCKER_UNRESOLVED`` is a
    disclosed limitation blocking the calibration CLAIM, not bundle incoherence.

    PE-8's own ``OPEN`` terminal state is honoured rather than upgraded: where PE-8
    says the evidence supports no more than a description, certification does not
    claim more.
    """

    scored = [reading for reading in readings if reading.get("state") != STATE_CALIBRATION_NOT_APPLICABLE]
    if any(reading.get("state") == STATE_CERTIFICATION_BLOCKER_UNRESOLVED for reading in readings):
        return STATE_CERTIFICATION_BLOCKER_UNRESOLVED
    if not scored:
        return STATE_CALIBRATION_NOT_APPLICABLE
    if any(reading.get("state") == STATE_EVIDENCE_LIMITED for reading in readings):
        return STATE_EVIDENCE_LIMITED
    terminal = calibration_terminal_state(calibration)
    if terminal["declared"] and terminal["state"] == "OPEN":
        # PE-8's declared OPEN is the honest ceiling on the claim here.
        return STATE_EVIDENCE_LIMITED
    return STATE_CERTIFIED_COHERENT


def disclosure_block(calibration: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The limitations certification CARRIES, including PE-3's, unresolved.

    A clean calibration figure must never be presented as if
    ``CONTINUOUS_PROXY_TIE_LIMITATION`` did not apply, so this disclosure is
    emitted unconditionally and is carried into the persisted result.
    """

    carried = list(DISCLOSED_LIMITATIONS)
    source = "PE-3 structural proxy, carried by the engine"
    if calibration is not None:
        for item in calibration.get("limitations") or []:
            text = str(item)
            if text not in carried:
                carried.append(text)
        source = "PE-3 declaration, carried through the PE-8 artifact consulted here"
    return {
        "carried": carried,
        "unresolved": list(DISCLOSED_LIMITATIONS),
        "resolution": "NOT_ATTEMPTED",
        "source": source,
        "note": (
            "certification discloses these limitations rather than resolving them: no figure "
            "here is presented as if the limitation did not apply"
        ),
    }


# ---------------------------------------------------------------------------
# Per-event and per-horizon certification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CertifiedEventBundle:
    """One event's certification: the certified bundle plus its states."""

    event: int
    cutoff: str
    state: str
    structural_state: str
    evidence_state: str
    reasons: tuple[str, ...]
    bundle_identity: str
    runs: Mapping[str, int]
    model_versions: Mapping[str, str]
    data_snapshot_sha256: str | None
    code_snapshot_sha256: str | None
    planning_context_hash: str | None
    calibration: Mapping[str, Any] = field(default_factory=dict)
    surfaces: tuple[Mapping[str, Any], ...] = ()
    evidence_reasons: tuple[str, ...] = ()
    #: PE-4's zero-fixture (BLANK) event world: reported, never a defect.
    zero_fixture_event: bool = False

    @property
    def certified(self) -> bool:
        return self.state in NON_FAILURE_BUNDLE_STATES

    def as_dict(self) -> dict[str, Any]:
        return {
            "event": int(self.event),
            "cutoff": self.cutoff,
            "state": self.state,
            "structural_state": self.structural_state,
            "evidence_state": self.evidence_state,
            "reasons": list(self.reasons),
            "evidence_reasons": list(self.evidence_reasons),
            "bundle_identity": self.bundle_identity,
            "runs": {str(k): int(v) for k, v in self.runs.items()},
            "model_versions": {str(k): str(v) for k, v in self.model_versions.items()},
            "data_snapshot_sha256": self.data_snapshot_sha256,
            "code_snapshot_sha256": self.code_snapshot_sha256,
            "planning_context_hash": self.planning_context_hash,
            "calibration": dict(self.calibration),
            "surfaces": [dict(surface) for surface in self.surfaces],
            "zero_fixture_event": bool(self.zero_fixture_event),
        }


def certify_event_bundle(
    conn: sqlite3.Connection,
    *,
    event: int,
    cutoff: str,
    runs: Mapping[str, int],
    required_versions: Mapping[str, str] | None = None,
    data_snapshot_sha256: str | None = None,
    code_snapshot_sha256: str | None = None,
    expected_data_snapshot_sha256: str | None = None,
    calibration: Mapping[str, Any] | None = None,
    require_calibration: bool = False,
    certification_identity: str | None = None,
) -> CertifiedEventBundle:
    """Certify ONE event's bundle structurally, then read its evidence.

    Structural gates are absolute (identity and coherence, never predictive
    quality), so they need no empirical threshold and they REFUSE.  Evidence gates
    never refuse for insufficiency: they yield a state.

    An artifact that is REQUIRED and absent refuses (``EVIDENCE_MISSING``).  An
    artifact that was simply not supplied, where the engine does ask a calibration
    question, yields ``EVIDENCE_LIMITED``: the claim is not made, and that is not
    the same fact as "there is no calibration question here".
    """

    runs = {str(family): int(run_id) for family, run_id in runs.items()}
    structural_state = STATE_CERTIFIED_COHERENT
    reasons: list[str] = []
    try:
        bundle = certified_bundle_from_explicit_ids(
            conn,
            event=int(event),
            cutoff=str(cutoff),
            runs=runs,
            required_versions=required_versions,
            data_snapshot_sha256=data_snapshot_sha256,
            code_snapshot_sha256=code_snapshot_sha256,
            expected_data_snapshot_sha256=expected_data_snapshot_sha256,
        )
    except BundleIncoherent as failure:
        structural_state = bundle_state_from_reasons(failure.reasons)
        bundle = None
        reasons.extend(failure.reasons)

    calibration_block: dict[str, Any] = {"consulted": False, "identity": None, "world_bound": None}
    surfaces: list[dict[str, Any]] = []
    evidence_reasons: list[str] = []
    evidence_state = STATE_CERTIFIED_COHERENT
    if bundle is not None:
        calibration_block = validate_calibration_evidence(
            calibration=calibration,
            event=int(event),
            cutoff=str(cutoff),
            runs=runs,
            certification_identity=certification_identity,
            require_calibration=require_calibration,
        )
        if calibration is not None:
            surfaces = calibration_surface_states(calibration)
            evidence_state = calibration_evidence_state(calibration, surfaces)
            evidence_reasons = [
                reason for reading in surfaces for reason in (reading.get("reasons") or [])
            ]
            if (
                evidence_state == STATE_EVIDENCE_LIMITED
                and not any(
                    reading.get("state") == STATE_EVIDENCE_LIMITED for reading in surfaces
                )
            ):
                # The surfaces are clean but PE-8's own terminal state is OPEN, so the
                # cap on the claim comes from PE-8 rather than from any single surface.
                terminal = calibration_terminal_state(calibration)
                evidence_reasons.append(
                    "PE-8 declares terminal state OPEN, so no calibration claim beyond PE-8's own "
                    "declared evidence is made"
                    + (" (" + "; ".join(terminal["reasons"]) + ")" if terminal["reasons"] else "")
                )
        else:
            # No calibration evidence was consulted.  The surfaces that ask a
            # calibration question were therefore not answered, so no calibration
            # CLAIM is made -- which is not a defect and not a refusal.
            evidence_state = STATE_EVIDENCE_LIMITED
            evidence_reasons = [
                "no PE-8 calibration evidence artifact was supplied, so no calibration claim is "
                "made for this certified bundle"
            ]

    state = structural_state if structural_state in BLOCKING_BUNDLE_STATES else evidence_state
    return CertifiedEventBundle(
        event=int(event),
        cutoff=str(cutoff),
        state=state,
        structural_state=structural_state,
        evidence_state=evidence_state,
        reasons=tuple(reasons),
        bundle_identity=bundle.bundle_identity() if bundle is not None else "",
        runs=dict(bundle.runs) if bundle is not None else dict(runs),
        model_versions=dict(bundle.model_versions) if bundle is not None else {},
        data_snapshot_sha256=(bundle.data_snapshot_sha256 if bundle is not None else data_snapshot_sha256),
        code_snapshot_sha256=(bundle.code_snapshot_sha256 if bundle is not None else code_snapshot_sha256),
        planning_context_hash=(bundle.planning_context_hash if bundle is not None else None),
        calibration=calibration_block,
        surfaces=tuple(surfaces),
        evidence_reasons=tuple(evidence_reasons),
        zero_fixture_event=bool(bundle.zero_fixture_event) if bundle is not None else False,
    )


def per_event_runs_from_bundles(bundles: Mapping[Any, Mapping[str, Any]]) -> dict[int, dict[str, int]]:
    """Every event's exact certified run ids, from certified bundle payloads."""

    resolved: dict[int, dict[str, int]] = {}
    for event, bundle in (bundles or {}).items():
        runs = (bundle or {}).get("runs") or {}
        resolved[int(event)] = {str(family): int(run_id) for family, run_id in runs.items()}
    return resolved


def certified_bundle_payload(
    bundles: Mapping[Any, Mapping[str, Any]], event: int
) -> Mapping[str, Any]:
    """One event's certified bundle payload, under EITHER key spelling.

    ``certified_bundles`` arrives from persisted JSON (string keys) and from
    in-memory payloads (integer keys).  Reading one spelling only would silently
    find nothing, and an event certified without the snapshots its own payload
    declares is not the identity the certifier minted; so both spellings resolve
    to the same payload, and a payload that is present but empty stays empty.
    """

    if not isinstance(bundles, Mapping):
        return {}
    for key in (str(int(event)), int(event)):
        payload = bundles.get(key)
        if isinstance(payload, Mapping) and payload:
            return payload
    return {}


def certify_decision_horizon(
    conn: sqlite3.Connection,
    *,
    certification: Mapping[str, Any],
    events: Sequence[int],
    cutoff: str,
    required_versions: Mapping[str, str] | None = None,
    calibration: Mapping[str, Any] | None = None,
    require_calibration: bool = False,
    last_event: int | None = None,
    identified_bypasses: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Certify a WHOLE horizon from exactly ONE certification artifact.

    The horizon is certified AS a horizon: every event's support comes from the
    one artifact, and the requested horizon must equal the certified one -- a
    horizon assembled from two artifacts, or from a subset/superset, is refused
    before any bundle is read.  One bad event makes the horizon
    ``DECISION_HORIZON_INCOMPLETE``; a partial horizon is never padded.

    ``identified_bypasses`` defaults to the DECLARED register
    (:func:`disclosed_bypasses`), so the disclosure is emitted by default rather
    than only when a caller remembers to pass it.
    """

    from . import four_gw_decision as fg

    if identified_bypasses is None:
        identified_bypasses = disclosed_bypasses()

    if certification is None:
        raise CertificationArtifactAbsent(["no certification artifact was supplied"])
    bundles = certification.get("certified_bundles") or {}
    if not bundles:
        raise CertificationArtifactContradictory(
            ["the certification artifact carries no certified bundles"]
        )
    requested = fg.canonical_event_horizon(events)
    certified = fg.canonical_event_horizon(bundles)
    if requested != certified:
        raise CertificationArtifactContradictory(
            [
                f"the requested horizon {list(requested)} is not the certified horizon "
                f"{list(certified)}; one artifact certifies exactly its own events"
            ]
        )
    artifact_snapshot = certification.get("data_snapshot_sha256")
    artifact_identity = certification.get("four_gw_certification_identity")
    versions = dict(required_versions) if required_versions else dict(
        certification.get("required_model_versions") or {}
    )

    per_event: dict[int, dict[str, Any]] = {}
    failures: list[str] = []
    support: dict[int, dict[str, Any]] = {}
    # The exact certified run ids per event, resolved by the ONE helper that reads
    # them off certified bundle payloads.
    runs_by_event = per_event_runs_from_bundles(bundles)
    for event in events:
        payload = certified_bundle_payload(bundles, int(event))
        runs = runs_by_event.get(int(event), {})
        # The certified event's identity must be EXACTLY the identity the artifact
        # recorded, so the payload's own declared snapshots are what is used -- not
        # the artifact's, which would silently mint a different identity for the same
        # bundle and break the round trip.  A payload that declares no code identity
        # while the artifact declares one cannot be bound to it, and fails closed.
        payload_code = payload.get("code_snapshot_sha256")
        artifact_code = certification.get("code_snapshot_sha256")
        if artifact_code and not payload_code:
            raise CertificationRefused(
                STATE_EVIDENCE_MISSING,
                [
                    f"GW{int(event)}: the certification artifact records code snapshot "
                    f"{artifact_code}, but the certified bundle for this event declares none; the "
                    "bundle cannot be bound to the code identity it was certified under"
                ],
            )
        try:
            certified_event = certify_event_bundle(
                conn,
                event=int(event),
                cutoff=str(cutoff),
                runs=runs,
                required_versions=versions or None,
                data_snapshot_sha256=payload.get("data_snapshot_sha256") or artifact_snapshot,
                code_snapshot_sha256=payload_code or artifact_code,
                expected_data_snapshot_sha256=artifact_snapshot,
                calibration=calibration,
                require_calibration=require_calibration,
                certification_identity=str(artifact_identity) if artifact_identity else None,
            )
        except CertificationRefused as refusal:
            failures.append(
                f"GW{int(event)}: {refusal.token}: {'; '.join(refusal.reasons)}"
            )
            # ONE key spelling for every event's record, refusal or not: the roll-up
            # below sorts those keys, and a mixture of integer and string spellings
            # cannot be ordered at all -- which would turn a structured refusal into a
            # TypeError instead of the token a consumer reads.
            per_event[str(int(event))] = {
                "event": int(event),
                "cutoff": str(cutoff),
                "state": refusal.token,
                "structural_state": refusal.token,
                "evidence_state": refusal.token,
                "reasons": list(refusal.reasons),
                "evidence_reasons": list(refusal.reasons),
                "bundle_identity": None,
                "runs": {str(k): int(v) for k, v in runs.items()},
                "model_versions": {},
                "data_snapshot_sha256": payload.get("data_snapshot_sha256") or artifact_snapshot,
                "code_snapshot_sha256": payload_code or artifact_code,
                "planning_context_hash": payload.get("planning_context_hash"),
                "calibration": {"consulted": calibration is not None},
                "surfaces": [],
            }
            support[int(event)] = {"supported": False, "data_cutoff": None, "missing_families": []}
            continue
        per_event[str(int(event))] = certified_event.as_dict()
        support[int(event)] = {
            "supported": certified_event.state not in BLOCKING_BUNDLE_STATES,
            "data_cutoff": str(cutoff),
            "missing_families": [],
            "state": certified_event.state,
            "bundle_identity": certified_event.bundle_identity,
            "matched_runs": dict(certified_event.runs),
        }

    horizon = fg.evaluate_horizon(
        # The horizon is built from the CANONICAL event set, whose equality with the
        # requested set is what authorised this call.  Reading the caller's first
        # element instead would re-anchor the horizon to whatever position it happened
        # to occupy, so an equal-but-reordered request could be reported as a
        # different (shorter) horizon and a season-end carve-out could be claimed on
        # the wrong events.
        planning_event=int(certified[0]) if certified else int(events[0]),
        support_by_event=support,
        cutoff=str(cutoff),
        last_event=(
            int(last_event)
            if last_event is not None
            else fg.season_last_event_from_db(conn)
        ),
    )
    states = {
        str(event): record.get("state")
        for event, record in sorted(per_event.items(), key=lambda item: str(item[0]))
    }
    open_reasons: list[str] = []
    if horizon["status"] == fg.DECISION_HORIZON_INCOMPLETE:
        open_reasons.append(
            f"{fg.DECISION_HORIZON_INCOMPLETE}: blocked events {horizon['blocked_events']}"
        )
    for event, state in states.items():
        if state == STATE_EVIDENCE_LIMITED:
            open_reasons.append(f"GW{event}: EVIDENCE_LIMITED (a calibration claim is not made)")
        elif state == STATE_CERTIFICATION_BLOCKER_UNRESOLVED:
            open_reasons.append(f"GW{event}: CERTIFICATION_BLOCKER_UNRESOLVED")
    # The SAME declared predicate the register uses, so the phase state cannot
    # disagree with the register it is computed from.
    unresolved = [
        dict(entry)
        for entry in identified_bypasses
        if str(entry.get("status")) not in {"CLOSED", "CLOSED_BY_REFUSAL"}
    ]
    if unresolved:
        open_reasons.append(
            "an identified downstream bypass is unresolved: "
            + "; ".join(str(entry.get("reference")) for entry in unresolved)
        )
    phase_state = PHASE_READY_FOR_MERGE if not open_reasons else PHASE_OPEN
    return {
        "schema": "fpl_brain.pe9_certification.v1",
        "phase": "PE-9",
        "certification_artifact_identity": artifact_identity,
        "certification_artifact_schema": certification.get("schema"),
        "planning_cutoff": str(cutoff),
        # The CERTIFIED horizon in its canonical order.  The requested set already
        # proved equal to it, so this is the same horizon -- and the persisted result
        # identity is then a function of WHAT was certified rather than of the order
        # the caller happened to list it in.
        "events": [int(event) for event in certified],
        "data_snapshot_sha256": artifact_snapshot,
        "code_snapshot_sha256": certification.get("code_snapshot_sha256"),
        "required_model_versions": {str(k): str(v) for k, v in versions.items()},
        "per_event": per_event,
        "per_event_bundle_identity": {
            str(event): record.get("bundle_identity")
            for event, record in sorted(per_event.items(), key=lambda item: str(item[0]))
        },
        "horizon_state": str(horizon["status"]),
        "horizon": horizon,
        "phase_terminal_state": phase_state,
        "phase_open_reasons": open_reasons,
        "calibration_identity": calibration_identity(calibration) if calibration else None,
        "calibration_consulted": calibration is not None,
        # PE-8's own terminal state, carried beside the certification state so the two
        # claims cannot be confused: certification never claims more than PE-8 declared.
        "calibration_pe8_terminal_state": (
            calibration_terminal_state(calibration)["state"] if calibration else None
        ),
        "disclosure": disclosure_block(calibration),
        "identified_bypasses": [dict(entry) for entry in identified_bypasses],
        "disclosed_limitations": disclosed_limitations(),
        "unresolved_bypasses": unresolved,
        "refusals": failures,
        "no_promotion": True,
        "promotion": (
            "NOT PERFORMED; every incumbent stays authoritative, no model version is bumped and no "
            "calibration version is replaced"
        ),
        "production_wiring": (
            "NONE; no PE-8 transform is wired into a production decision path by certification"
        ),
        "flags": [
            CONTINUOUS_PROXY_TIE_LIMITATION,
            "NO_NEW_NUMERICAL_THRESHOLD",
            "ONE_CERTIFICATION_ARTIFACT_PER_DECISION",
        ],
    }


def require_certification_artifact(certification: Any) -> Mapping[str, Any]:
    """Exactly ONE certification artifact per decision.

    Absent and contradictory are DIFFERENT facts and carry different tokens: an
    absent artifact is a missing authorisation, while an artifact that does not
    describe this decision is a contradiction.  A decision that would need two
    artifacts is refused rather than merged.
    """

    if certification is None:
        raise CertificationArtifactAbsent(["a production decision requires a certification artifact"])
    candidates = (
        list(certification)
        if isinstance(certification, (list, tuple, set, frozenset))
        else [certification]
    )
    candidates = [item for item in candidates if item is not None]
    if not candidates:
        raise CertificationArtifactAbsent(["a production decision requires a certification artifact"])
    if len(candidates) > 1:
        raise CertificationArtifactContradictory(
            [
                f"{len(candidates)} certification artifacts were supplied for one decision; a "
                "horizon assembled from two artifacts is refused"
            ]
        )
    if not isinstance(candidates[0], Mapping):
        raise CertificationArtifactContradictory(
            [f"the certification artifact is {type(candidates[0]).__name__}, not an object"]
        )
    if isinstance(candidates[0], ValidatedCertificationArtifact):
        # An authorisation that was already minted has its content-bound digest
        # re-verified here, so every consumer of "exactly one artifact per decision"
        # gets the same byte-level check the loader boundary performs: an artifact
        # edited after validation is refused before anything is read from it.
        assert_certification_artifact_bytes_unchanged(candidates[0])
    return candidates[0]


#: A validated certification artifact is the AUTHORISATION a predictive load consumes.
#: It is built by COMPOSITION over a genuinely read-only snapshot -- a ``Mapping``
#: facade over private dicts wrapped in ``MappingProxyType``, sequences frozen to
#: tuples and sets to frozensets -- rather than by subclassing ``dict``/``list``.  A
#: dict subclass only REFUSES the mutators it overrides while ``dict.__setitem__``
#: still edits the value underneath, so subclassing is not immutability.  Nothing
#: mutable is reachable from the snapshot, so the bytes a load was authorised by
#: cannot be changed after the check.
#:
#: The mint token is a closure variable of :func:`_validated_certification_artifact_capability`
#: and is deliberately NOT a module attribute: the class is exposed so a consumer can
#: RECOGNISE the authorisation it was handed, but no caller can construct one through
#: normal module access.  The only constructor is the validation function created
#: beside the token, and it applies the complete contract before minting.

_IMMUTABLE_ARTIFACT_REFUSAL = (
    "a validated certification artifact is IMMUTABLE: it is a read-only snapshot minted by "
    "certified_bundle.validate_certification_artifact, so the bytes a load was authorised "
    "by cannot be changed after the check"
)


def _canonical_certification_bytes(document: Mapping[str, Any]) -> bytes:
    """The canonical byte form of an artifact's own content.

    ONE encoding, shared by the mint and by every re-verification, so "these bytes have
    not changed" is a pure function of the CONTENT rather than a property of an object
    identity a caller could keep alive.  A value the canonical encoding cannot represent
    is refused instead of stringified: an encoding that silently stringified its input
    could not detect a change.
    """

    def encode(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(key): encode(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [encode(item) for item in value]
        if isinstance(value, (set, frozenset)):
            return sorted((encode(item) for item in value), key=repr)
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        raise TypeError(
            "a validated certification artifact must carry values the canonical digest "
            f"can represent; found {type(value).__name__}"
        )

    return json.dumps(
        encode(document), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def certification_artifact_digest(document: Mapping[str, Any]) -> str:
    """The CONTENT-BOUND digest of an artifact's own bytes, from the ONE algorithm.

    Every boundary recomputes it and compares it against the identity a validated
    artifact was minted with, so an artifact whose content changed after validation is
    refused rather than read.
    """

    try:
        payload = _canonical_certification_bytes(document)
    except TypeError as failure:
        raise CertificationArtifactUnvalidated([str(failure)]) from failure
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _readonly_certification_value(value: Any) -> Any:
    """A recursively READ-ONLY snapshot of a certification value.

    Mappings become ``MappingProxyType`` over a private dict, sequences become tuples
    and sets become frozensets, so no mutable container is reachable from the result:
    there is no base-class mutator to call and no ``copy``-and-edit route back into the
    artifact the boundary validated.
    """

    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _readonly_certification_value(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_readonly_certification_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_readonly_certification_value(item) for item in value)
    return value


def plain_certification_value(value: Any) -> Any:
    """A PLAIN (dict/list/scalar) copy of a certification value.

    The validated artifact is a read-only snapshot, so a consumer that must SERIALIZE
    what it read -- an identity hash over the artifact's own fields, a persisted
    payload -- converts through this ONE function instead of walking the snapshot
    itself.  Conversion is lossless: it copies the CONTENT and never its container, so
    the value a consumer hashes is the value the artifact carries.  Integer mapping
    keys become strings, exactly as ``json`` would render them.
    """

    if isinstance(value, Mapping):
        return {str(key): plain_certification_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain_certification_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((plain_certification_value(item) for item in value), key=repr)
    return value


def assert_certification_artifact_bytes_unchanged(artifact: Any) -> None:
    """Refuse a validated artifact whose bytes no longer digest to its minted identity.

    This is the boundary half of the authorisation contract: a validated artifact
    carries the digest of the bytes the canonical loader validated, and every boundary
    re-verifies it before anything predictive is read.  A mismatch means the value was
    edited through a route the type system does not police (``object.__setattr__`` and
    friends) after validation -- the case the digest exists to catch -- and it is
    refused with its own token instead of being read.
    """

    recorded = getattr(artifact, "content_digest", None)
    if not recorded:
        raise CertificationArtifactUnvalidated(
            [
                "the value carries no content-bound digest, so the bytes the canonical loader "
                "validated cannot be re-verified"
            ]
        )
    recomputed = certification_artifact_digest(artifact)
    if str(recorded) != str(recomputed):
        raise CertificationArtifactMutated(
            [
                f"the validated certification artifact's bytes digest to {recomputed[:23]}..., not "
                f"the identity it was minted under ({str(recorded)[:23]}...); an artifact edited "
                "after validation is not the authorisation the loader issued"
            ]
        )


def _validated_certification_artifact_capability() -> tuple[type, Any]:
    """Own the ONE mint of a validated certification artifact.

    The mint token is a CLOSURE variable and is never bound as a module attribute: the
    class is exposed so a consumer can recognise the authorisation it was handed, but
    no caller can CONSTRUCT one through normal module access.  The only constructor
    created here is the validation function, which applies the complete contract to the
    document before minting -- so an artifact-shaped mapping the contract has not
    passed can never become an authorisation.
    """

    token = object()

    class ValidatedCertificationArtifact(Mapping):
        """An IMMUTABLE certification artifact that PASSED the complete contract.

        This is the authorisation a predictive load consumes.  It is built by
        COMPOSITION over a private, recursively read-only snapshot -- ``MappingProxyType``
        for mappings, tuples for sequences -- so there is no mutable container to reach
        and no base-class mutator that bypasses the class's own.  It is a mapping, so
        every existing reader (``artifact.get(...)``, ``dict(artifact)``,
        ``candidate_universe.jsonable``) works unchanged.

        ``validated_by`` names the function that applied the contract,
        ``validated_identity`` is the artifact's own identity, and ``content_digest`` is
        the content-bound digest of the bytes that were validated -- re-verified at every
        boundary by :func:`assert_certification_artifact_bytes_unchanged`, so a value
        edited after minting is refused rather than read.
        """

        __slots__ = ("_document", "_content_digest", "_validated_by", "_validated_identity")

        def __init__(
            self,
            document: Mapping[str, Any] | None = None,
            *,
            validated_by: str = "",
            _token: Any = None,
        ) -> None:
            if _token is not token:
                raise CertificationArtifactUnvalidated(
                    [
                        "a validated certification artifact is minted only by "
                        "certified_bundle.validate_certification_artifact, which applies the "
                        "complete authorization contract; this mapping was never validated"
                    ]
                )
            snapshot = _readonly_certification_value(dict(document or {}))
            object.__setattr__(self, "_document", snapshot)
            object.__setattr__(self, "_content_digest", certification_artifact_digest(snapshot))
            object.__setattr__(self, "_validated_by", str(validated_by))
            object.__setattr__(
                self,
                "_validated_identity",
                str(
                    snapshot.get("four_gw_certification_identity")
                    or snapshot.get("certification_identity")
                    or ""
                ),
            )

        # -- the read-only mapping facade ------------------------------------

        def __getitem__(self, key: Any) -> Any:
            return self._document[key]

        def __iter__(self) -> Any:
            return iter(self._document)

        def __len__(self) -> int:
            return len(self._document)

        def __contains__(self, key: Any) -> bool:
            return key in self._document

        def __eq__(self, other: Any) -> Any:
            # Equality is a property of the CONTENT, not of the object: a copy of the
            # same bytes is the same artifact, and the canonical encoding compares a
            # frozen tuple with the list it was parsed from.
            if not isinstance(other, Mapping):
                return NotImplemented
            try:
                return self._content_digest == certification_artifact_digest(other)
            except CertificationRefused:
                return NotImplemented

        def __setattr__(self, name: str, value: Any) -> None:
            raise TypeError(_IMMUTABLE_ARTIFACT_REFUSAL)

        def __delattr__(self, name: str) -> None:
            raise TypeError(_IMMUTABLE_ARTIFACT_REFUSAL)

        def __copy__(self) -> "ValidatedCertificationArtifact":
            return self

        def __deepcopy__(self, memo: Any) -> "ValidatedCertificationArtifact":
            return self

        def __reduce__(self) -> Any:
            # An authorisation is never reconstructed from bytes: a consumer either
            # holds the value the loader minted or passes through the contract.
            raise TypeError(_IMMUTABLE_ARTIFACT_REFUSAL)

        @property
        def validated_by(self) -> str:
            return self._validated_by

        @property
        def validated_identity(self) -> str:
            return self._validated_identity

        @property
        def content_digest(self) -> str:
            return self._content_digest

        def __repr__(self) -> str:  # pragma: no cover - diagnostics only
            return (
                f"ValidatedCertificationArtifact(validated_by={self.validated_by!r}, "
                f"identity="
                f"{self.validated_identity[:23] + '...' if self.validated_identity else None}, "
                f"events={list(self.get('events') or [])})"
            )

    def validate_certification_artifact(document: Any) -> "ValidatedCertificationArtifact":
        """The CANONICAL loader-owned validation of a certification artifact.

        A raw mapping is NOT an authorisation, however self-consistent its own run ids or
        bundle identities are: the artifact must carry the fields the certification
        loader requires before anything predictive may be read from it.  This function is
        the ONE place that contract is applied at a load boundary:

        * an already-validated artifact has its content-bound digest RE-VERIFIED against
          the bytes it was minted from and is returned unchanged, so the authorisation a
          caller loaded once is checked -- not merely trusted -- at every boundary it
          reaches;
        * a path is read through ``four_gw_decision.load_certification_artifact`` -- the
          production loader -- which mints the validated value;
        * a raw mapping has the SAME contract re-run over it and is then minted, so a
          mapping that does carry every required authorization field is accepted exactly
          as the loader would accept its file form, and a mapping that does not is refused
          with ``CERTIFICATION_ARTIFACT_UNVALIDATED`` (never silently trusted, never
          defaulted, never substituted).

        The value returned is a read-only snapshot carrying the digest of its own bytes:
        what the boundary validated is what the load reads.
        """

        from . import four_gw_decision as fg

        if isinstance(document, ValidatedCertificationArtifact):
            assert_certification_artifact_bytes_unchanged(document)
            return document
        if isinstance(document, (str, Path)):
            return fg.load_certification_artifact(document)
        if document is None:
            raise CertificationArtifactAbsent(
                ["a predictive load requires a certification artifact"]
            )
        if not isinstance(document, Mapping) or not document:
            raise CertificationArtifactUnvalidated(
                [
                    f"the certification artifact is {type(document).__name__}, not an artifact the "
                    "canonical loader produced"
                ]
            )
        try:
            payload = fg.validate_certification_artifact(document)
        except CertificationRefused:
            raise
        except BundleIncoherent as failure:
            # A RECORDED incoherent dependency status is a fact about the predictive
            # world and keeps the token that fact already has; an artifact that records
            # no status at all simply never carried an authorisation.
            recorded = str((document or {}).get("dependency_validation") or "").upper()
            if recorded and recorded != "COHERENT":
                raise CertificationRefused(
                    bundle_state_from_reasons(failure.reasons), failure.reasons
                ) from failure
            raise CertificationArtifactUnvalidated(
                [
                    "the certification artifact does not carry the authorisation the canonical "
                    f"loader's contract requires: {failure}"
                ]
            ) from failure
        except Exception as failure:
            detail = str(getattr(failure, "detail", failure))
            # A mapping whose own declared records contradict each other is a
            # CONTRADICTION, whatever else it is missing: that is a different fact from a
            # mapping that simply never carried an authorisation, and the two carry
            # different tokens.
            contradicting = (
                fg.DIAG_CERTIFICATION_BUNDLE_IDENTITY_MISMATCH in detail
                or fg.DIAG_CERTIFICATION_ARTIFACT_CONTRADICTORY in detail
            )
            if contradicting:
                raise CertificationArtifactContradictory(
                    [f"the certification artifact contradicts its own records: {detail}"]
                ) from failure
            raise CertificationArtifactUnvalidated(
                [
                    "the certification artifact does not carry the authorisation the canonical "
                    f"loader's contract requires: {type(failure).__name__}: {detail}"
                ]
            ) from failure
        return ValidatedCertificationArtifact(
            payload,
            validated_by="four_gw_decision.validate_certification_artifact",
            _token=token,
        )

    return ValidatedCertificationArtifact, validate_certification_artifact


ValidatedCertificationArtifact, validate_certification_artifact = (
    _validated_certification_artifact_capability()
)


def bundle_identity_payload(
    *,
    event: int,
    cutoff: str | None,
    runs: Mapping[str, int],
    model_versions: Mapping[str, str],
    code_snapshot_sha256: str | None = None,
    data_snapshot_sha256: str | None = None,
    planning_context_hash: str | None = None,
) -> dict[str, Any]:
    """The canonical identity payload of ONE certified bundle.

    Producers that hand a bundle to a downstream loader build it from here, so the
    identity they declare is computed by the ONE shared algorithm.
    """

    return {
        "event": int(event),
        "cutoff": cutoff,
        "runs": {str(family): int(run_id) for family, run_id in runs.items()},
        "model_versions": {str(family): str(version) for family, version in model_versions.items()},
        "code_snapshot_sha256": code_snapshot_sha256,
        "data_snapshot_sha256": data_snapshot_sha256,
        "planning_context_hash": planning_context_hash,
    }


def recorded_model_versions(conn: sqlite3.Connection, runs: Mapping[str, int]) -> dict[str, str]:
    """The model version RECORDED for each of these run ids, if the row is present.

    A bundle that is handed to a predictive loader must declare the version of every
    family it names, and those declarations are checked against the rows themselves
    -- the recorded run metadata is the ground truth, never a caller's assertion.

    A family whose run row is absent from THIS database is omitted rather than
    reported as an empty version: a certified generation loaded from its
    content-addressed cache, or a fixture world, legitimately has no row here, and
    an empty string would be a defaulted version -- exactly what the caller must not
    read.  Absence is therefore a different fact from a blank declaration.
    """

    recorded: dict[str, str] = {}
    for family, run_id in sorted(runs.items()):
        row = _run(conn, int(run_id))
        if row is not None:
            recorded[str(family)] = str(row["model_version"])
    return recorded


def certified_bundle_identity_for(
    *,
    event: int,
    cutoff: str | None,
    runs: Mapping[str, int],
    model_versions: Mapping[str, str],
    code_snapshot_sha256: str | None = None,
    data_snapshot_sha256: str | None = None,
    planning_context_hash: str | None = None,
) -> str:
    """The identity a loader-bound bundle must declare, from the ONE algorithm.

    Producers call this instead of hashing anything themselves, so the identity a
    bundle declares and the identity a consumer recomputes cannot drift.
    """

    return canonical_bundle_identity(
        bundle_identity_payload(
            event=int(event),
            cutoff=cutoff,
            runs=runs,
            model_versions=model_versions,
            code_snapshot_sha256=code_snapshot_sha256,
            data_snapshot_sha256=data_snapshot_sha256,
            planning_context_hash=planning_context_hash,
        )
    )


def certified_bundle_artifact_record(
    certification: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
    event: int,
) -> dict[str, Any]:
    """The artifact's recorded CERTIFIED bundle for ONE event, or a refusal.

    This is the half of the boundary only the ARTIFACT can answer: which bundle the
    certification committed to, for which event, with which family set, which
    cutoff and which identity.  Reading it in ONE place is what stops a
    caller-minted, self-consistent hash from standing in for an authorisation, and
    it also stops an artifact that recorded a REFUSAL (or a non-coherent predictive
    bundle) from authorising a load.

    ``CERTIFICATION_ARTIFACT_ABSENT``, ``CERTIFICATION_ARTIFACT_CONTRADICTORY`` and
    ``EVIDENCE_MISSING`` are different facts with different tokens: a missing
    authorisation, an artifact that contradicts the request, and an artifact that
    simply does not record this event.  A raw mapping is not an authorisation either:
    the artifact must pass the canonical loader's complete contract, so the
    self-consistent but unauthorised mapping is refused with
    ``CERTIFICATION_ARTIFACT_UNVALIDATED`` before any predictive data is read.
    """

    artifact = require_certification_artifact(certification)
    artifact = validate_certification_artifact(artifact)
    payload = certified_bundle_payload(artifact.get("certified_bundles") or {}, int(event))
    if not payload:
        raise CertificationRefused(
            STATE_EVIDENCE_MISSING,
            [
                f"the certification artifact records no certified bundle for event {int(event)}; "
                "an event the certification did not commit to is never loaded"
            ],
        )
    recorded_event = payload.get("event")
    if recorded_event is None or int(recorded_event) != int(event):
        raise CertificationArtifactContradictory(
            [
                f"the bundle recorded under event {int(event)} was certified for event "
                f"{recorded_event!r}; a bundle is bound to its own event, not to a container key"
            ]
        )
    runs = {str(family): int(run_id) for family, run_id in (payload.get("runs") or {}).items()}
    if not runs:
        raise CertificationRefused(
            STATE_EVIDENCE_MISSING,
            [f"the bundle certified for event {int(event)} names no projection run"],
        )
    missing = [family for family in LOAD_REQUIRED_FAMILIES if family not in runs]
    if missing:
        raise CertificationRefused(
            STATE_EVIDENCE_MISSING,
            [
                f"the bundle certified for event {int(event)} names no run for {sorted(missing)}; "
                "a world load is authorised by the exact certified run ids of the families it "
                "reads, and a load is never authorised by a partial record"
            ],
        )
    cutoff = payload.get("cutoff") or artifact.get("planning_cutoff")
    if not cutoff:
        raise CertificationRefused(
            STATE_EVIDENCE_MISSING,
            [f"the certification records no data cutoff for event {int(event)}"],
        )
    # A certification STATUS authorises a load only when it is a non-blocking one.
    # The certified bundle payloads the certifier writes carry no status (they are
    # the validated bundle itself), so an ABSENT status is "not recorded" rather
    # than a defect; a RECORDED blocking one is a refusal, never a cached hit.
    recorded_states = [
        str(payload.get("state") or ""),
        str(artifact.get("predictive_bundle_status") or ""),
    ]
    blocking = [state for state in recorded_states if state in BLOCKING_BUNDLE_STATES]
    if blocking:
        raise CertificationRefused(
            blocking[0],
            [
                f"the certification records {blocking[0]} for event {int(event)}"
                + (
                    ": " + "; ".join(str(reason) for reason in (payload.get("reasons") or []))
                    if payload.get("reasons")
                    else ""
                )
            ],
        )
    return {
        "event": int(event),
        "payload": dict(payload),
        "identity": canonical_bundle_identity(payload),
        "runs": runs,
        "model_versions": {
            str(family): str(version)
            for family, version in (payload.get("model_versions") or {}).items()
        },
        "cutoff": str(cutoff),
        "state": next((state for state in recorded_states if state), None),
        "artifact_identity": artifact.get("four_gw_certification_identity"),
    }


def assert_event_bundle_certified(
    conn: sqlite3.Connection | None,
    bundle: Any,
    *,
    event: int,
    certification: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    required_versions: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Prove a routing/decision bundle is the CERTIFIED one before it is loaded.

    This is the downstream half of the contract: ``route_optimizer`` turns a bundle
    into simulated worlds, so the bundle must BE the one the certification
    committed to -- never a caller's rediscovery of "the latest run per family",
    and never a self-consistent identity a caller minted.  A validated
    certification artifact is therefore REQUIRED here, on every database load and
    on every content-addressed cache hit: a bundle that cannot present the
    authorisation is REFUSED, never defaulted, substituted or silently reloaded.

    What is enforced, in the order that keeps the most specific token:

    * the bundle declares a certified identity at all, and that identity BINDS its
      own run ids (the shared algorithm, not a caller's hash);
    * every family it names declares a model version, and each declared version is
      the AUTHORITATIVE required version of that frozen family -- with the
      artifact's own ``required_model_versions`` cross-checked against that same
      source, so a certification minted under some other pin is refused;
    * the artifact's recorded bundle for this event exists, was recorded FOR this
      event, names the run of every family the load reads
      (:data:`LOAD_REQUIRED_FAMILIES`), carries a cutoff, and is not a recorded
      refusal;
    * the bundle's identity, event and cutoff are the artifact's recorded ones;
    * where the run rows are present in this database, the recorded closure is
      re-proven from the rows themselves -- family, planning event, ``status ==
      complete``, exact data cutoff and the xPts/Monte Carlo dependency edges --
      because a certification artifact is a claim about those rows, not a
      substitute for them.
    """

    record = certified_bundle_artifact_record(certification, int(event))
    payload = getattr(bundle, "as_identity_payload", None)
    if payload is None:
        raise CertificationRefused(
            DIAG_CERTIFICATION_BOUNDARY_REQUIRED,
            [
                f"event {int(event)}: the bundle cannot declare its certified identity; a bundle "
                "without certification provenance is never loaded"
            ],
        )
    identity_payload = payload()
    declared_identity = getattr(bundle, "certified_bundle_identity", None)
    if not declared_identity:
        raise CertificationRefused(
            DIAG_CERTIFICATION_BOUNDARY_REQUIRED,
            [
                f"event {int(event)}: the bundle declares no certified_bundle_identity; the exact "
                "certified run ids are the only authorisation to load predictive data"
            ],
        )
    recomputed = canonical_bundle_identity(identity_payload)
    if str(declared_identity) != recomputed:
        raise CertificationRefused(
            STATE_PREDICTIVE_BUNDLE_INCOHERENT,
            [
                f"event {int(event)}: the declared certified_bundle_identity "
                f"{str(declared_identity)[:23]}... does not bind the bundle's own ids (recomputed "
                f"{recomputed[:23]}...)"
            ],
        )
    declared_versions = dict(getattr(bundle, "model_versions", None) or {})
    runs = {str(family): int(run_id) for family, run_id in (identity_payload.get("runs") or {}).items()}
    missing = sorted(set(runs) - set(declared_versions))
    if missing:
        raise CertificationRefused(
            STATE_EVIDENCE_MISSING,
            [
                f"event {int(event)}: the bundle names run(s) for {', '.join(missing)} but declares "
                "no model version for them; certified provenance requires every family's version"
            ],
        )

    # The AUTHORITATIVE required versions.  Every certification call site records
    # them from ``declared_required_versions``; the loader enforces the same values
    # and cross-checks the artifact's own record against them, so a bundle from an
    # unexpected version -- or a certification minted under a version nobody pins --
    # cannot be loaded.
    versions = {
        str(family): str(version)
        for family, version in dict(required_versions or declared_required_versions()).items()
    }
    artifact_pins = {
        str(family): str(version)
        for family, version in (
            dict(certification or {}).get("required_model_versions") or {}
        ).items()
    }
    for family, pinned in sorted(artifact_pins.items()):
        wanted = versions.get(family)
        if wanted is not None and str(pinned) != str(wanted):
            raise CertificationRefused(
                STATE_UNSUPPORTED_MODEL_VERSION,
                [
                    f"event {int(event)}: the certification requires {family} {pinned!r}, which is "
                    f"not the authoritative version {wanted!r}"
                ],
            )
    for family, wanted in sorted(versions.items()):
        declared = declared_versions.get(family)
        if declared is not None and str(declared) != str(wanted):
            raise CertificationRefused(
                STATE_UNSUPPORTED_MODEL_VERSION,
                [
                    f"event {int(event)}: {family} version {declared!r} != required {wanted!r}"
                ],
            )

    declared_cutoff = identity_payload.get("cutoff")
    if declared_cutoff is not None and str(declared_cutoff) != str(record["cutoff"]):
        raise CertificationArtifactContradictory(
            [
                f"event {int(event)}: the bundle declares cutoff {declared_cutoff} but the "
                f"certification recorded {record['cutoff']}"
            ]
        )
    if str(declared_identity) != str(record["identity"]):
        raise CertificationArtifactContradictory(
            [
                f"event {int(event)}: the bundle's identity {str(declared_identity)[:23]}... is not "
                f"the one the certification artifact records ({str(record['identity'])[:23]}...); a "
                "self-consistent identity is not an authorisation"
            ]
        )

    verified_against_runs = False
    if conn is not None:
        # The version each run RECORDS is the ground truth, read from the rows in one
        # place.  A family whose row is not in this database (a certified generation
        # loaded from its content-addressed cache, or a fixture world) is simply not
        # returned, and the DECLARED provenance is what authorises that load: the
        # strict "the run must exist" gate belongs to the certification path, which
        # proves it against the run rows it certifies.
        rows = {family: _run(conn, int(run_id)) for family, run_id in record["runs"].items()}
        present = sorted(family for family, row in rows.items() if row is not None)
        if present:
            # Where the rows ARE here, the RECORDED closure is re-proven from them:
            # family and planning event identity, ``status == complete``, the exact
            # data cutoff and the xPts / Monte Carlo dependency edges.  A
            # certification artifact is a CLAIM about those rows, never a substitute
            # for them.
            verified_against_runs = True
            try:
                validate_certified_bundle(
                    conn,
                    event=int(event),
                    cutoff=str(record["cutoff"]),
                    runs=record["runs"],
                    required_versions=versions,
                    data_snapshot_sha256=identity_payload.get("data_snapshot_sha256"),
                    code_snapshot_sha256=identity_payload.get("code_snapshot_sha256"),
                    expected_data_snapshot_sha256=(dict(certification or {}).get("data_snapshot_sha256")),
                    families=present,
                )
            except BundleIncoherent as failure:
                raise CertificationRefused(
                    bundle_state_from_reasons(failure.reasons), failure.reasons
                ) from failure
        else:
            for family, recorded in sorted(recorded_model_versions(conn, record["runs"]).items()):
                run_id = record["runs"][family]
                if recorded != str(declared_versions[family]):
                    raise CertificationRefused(
                        STATE_UNSUPPORTED_MODEL_VERSION,
                        [
                            f"event {int(event)}: {family} run {int(run_id)} recorded {recorded!r}, "
                            f"but the bundle declares {declared_versions[family]!r}"
                        ],
                    )
    return {
        "event": int(event),
        "certified_bundle_identity": str(declared_identity),
        "model_versions": {str(k): str(v) for k, v in declared_versions.items()},
        "required_model_versions": versions,
        "certification_artifact_identity": record["artifact_identity"],
        "certification_state": record["state"],
        "data_cutoff": str(record["cutoff"]),
        "verified_against_runs": verified_against_runs,
        "verified_against_certification": True,
    }


def certification_result_identity(payload: Mapping[str, Any]) -> str:
    """The persisted certification result's OWN identity.

    The persisted identity ROUND-TRIPS exactly: recomputing it from the stored
    bytes with this one shared algorithm yields the same value.  It is built from
    explicit run ids and their recorded identities, never from current rows, so a
    later mutation of ``players`` / ``fixtures`` / ``events`` / ``player_gameweeks``
    cannot rewrite a historical certified identity.
    """

    material = {
        "schema": payload.get("schema"),
        "certification_artifact_identity": payload.get("certification_artifact_identity"),
        "planning_cutoff": payload.get("planning_cutoff"),
        "events": [int(event) for event in (payload.get("events") or [])],
        "data_snapshot_sha256": payload.get("data_snapshot_sha256"),
        "code_snapshot_sha256": payload.get("code_snapshot_sha256"),
        "per_event_bundle_identity": {
            str(event): value
            for event, value in sorted(
                (payload.get("per_event_bundle_identity") or {}).items(), key=lambda item: str(item[0])
            )
        },
        "required_model_versions": {
            str(k): str(v) for k, v in (payload.get("required_model_versions") or {}).items()
        },
        "per_event_model_versions": {
            str(event): {str(k): str(v) for k, v in (record.get("model_versions") or {}).items()}
            for event, record in sorted(
                (payload.get("per_event") or {}).items(), key=lambda item: str(item[0])
            )
        },
        "per_event_state": {
            str(event): record.get("state")
            for event, record in sorted(
                (payload.get("per_event") or {}).items(), key=lambda item: str(item[0])
            )
        },
        "calibration_identity": payload.get("calibration_identity"),
    }
    return "sha256:" + hashlib.sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


# ---------------------------------------------------------------------------
# Disclosed downstream bypasses
# ---------------------------------------------------------------------------
#
# PE-9 closes the bypasses it can close at the boundary, and NAMES the ones it
# does not rewrite rather than assuming they are gone.  The statuses below are
# declared here so the certification result carries the disclosure mechanically
# instead of the reason being spread across commit messages and docstrings.
#
# CLOSED
#   The bypass can no longer be exercised: the consumer's entry point refuses a
#   bundle that cannot prove it is the certified one.
# CLOSED_BY_REFUSAL
#   The consumer still RESOLVES runs by latest-per-family, but the hand-assembled
#   bundle it feeds to the boundary is refused there, so no uncertified run id can
#   reach a prediction load.  The rediscovery itself is left in place deliberately:
#   rewriting a historical replay script is a different work item.
# DISCLOSED
#   A named limitation that is reported rather than removed, because it is a
#   property of the artefact and not a bypass: the missing value is labelled and
#   reported, never silently accepted as a prediction.

DISCLOSED_BYPASS_REFERENCE = "reference"
DISCLOSED_BYPASS_STATUS = "status"

CLOSED_BYPASS_DISCLOSURES: tuple[Mapping[str, str], ...] = (
    {
        "reference": "fpl_brain.route_optimizer.build_event_worlds caller-supplied bundles",
        "status": "CLOSED",
        "detail": (
            "the loader REQUIRES a VALIDATED artifact: the artifact must pass the canonical "
            "loader's complete contract (schema, causal and dependency status, the authorisation "
            "flag, the completeness audit, the wiring identity and an identity that binds its own "
            "bundles), so a raw mapping carrying self-consistent bundles but no authorisation is "
            "refused with CERTIFICATION_ARTIFACT_UNVALIDATED.  The authorisation itself is a "
            "READ-ONLY snapshot -- composition over MappingProxyType and tuples, never a "
            "dict/list subclass whose base-class mutators still edit it -- and every boundary "
            "re-verifies the content-bound digest it was minted with, so an artifact edited after "
            "validation is refused with CERTIFICATION_ARTIFACT_MUTATED rather than read.  The "
            "bundle must then be the one that "
            "artifact recorded for that event (identity, event and cutoff compared against the "
            "recorded bundle), every family it names must declare the authoritative model version, "
            "the recorded family / event / status / cutoff / dependency closure is re-proven from "
            "the run rows, and the check runs BEFORE the content-addressed cache is read -- so a "
            "self-consistent bundle built from arbitrary existing runs is refused on the database "
            "path and on a warm cache hit alike.  Worlds that were never loaded from a prediction "
            "run enter only through the declared route_optimizer.NonProductionWorlds interface, "
            "which is refused outright when an artifact is presented"
        ),
    },
    {
        "reference": "fpl_brain.route_optimizer.optimize injected and prebuilt worlds",
        "status": "CLOSED",
        "detail": (
            "optimize takes the same two declared doors: a validated certification artifact "
            "authorises the bundles, while injected worlds must be declared through "
            "NonProductionWorlds and are refused beside an artifact.  A PREBUILT matrix is admitted "
            "only through a loader-owned, content-bound capability: the matrix's canonical content "
            "identity -- every semantic block exact evaluation consumes -- must be one this loader "
            "ISSUED, for that event and against the certified bundle identity the artifact records "
            "(or, on the declared door, under that same declaration).  The capability is held "
            "INSIDE the loader that issues it: there is no module-global writable registry and no "
            "module-level issuer, so a caller cannot record content of its own and the two doors "
            "cannot be merged -- a registry a caller injects is inert, and the declared-door "
            "recorder (which the declared door, by design, needs) records no certified bundle "
            "identity and can never satisfy the certified door.  The provenance stamp on a "
            "matrix is caller-writable and therefore authorises nothing, so a hand-built matrix -- "
            "including one carrying the correct copied stamp -- is refused"
        ),
    },
    {
        "reference": "fpl_brain.route_comparator.compare_routes direct DB branch",
        "status": "CLOSED",
        "detail": (
            "the comparator's own predictive-data load crosses the same boundary and now requires "
            "the artifact; a bundle that is not the recorded one -- or whose recorded closure "
            "disagrees with the run rows -- is refused instead of simulated, and an injected world "
            "must be declared through route_optimizer.NonProductionWorlds"
        ),
    },
    {
        "reference": "fpl_brain.free_hit_request_adapter.load_certified_route_worlds",
        "status": "CLOSED",
        "detail": (
            "the Free Hit route-world load presents the LOADED artifact (carried on the decision "
            "authority derived from it) to the canonical loader, so the certified run ids it takes "
            "from the artifact's bundle map are compared against the artifact's own recorded "
            "bundles and a substituted, self-consistent bundle is refused"
        ),
    },
    {
        "reference": "fpl_brain.finalist_refinement.refine_finalists and route_stability.run_ladder",
        "status": "CLOSED",
        "detail": (
            "both forward the certification artifact (or the declared non-production source) into "
            "the loader and the optimizer they call, so neither can obtain a world by a route the "
            "boundary has not authorised"
        ),
    },
    {
        "reference": "fpl_brain.manager_worlds.build_manager_worlds",
        "status": "CLOSED",
        "detail": (
            "a predictive load beside the optimizer's: it simulated the shared worlds for the "
            "manager policy matrix straight from a hand-assembled run-id set.  It now requires a "
            "validated certification artifact and refuses unless the run ids it is handed are the "
            "ones that artifact recorded for the event, and its only caller -- "
            "scripts/build_manager_packet.py, a Phase-6A descriptive packet -- loads the artifact "
            "through the canonical loader, forwards it to the load, and validates every supplied "
            "event and run id against the artifact's own record instead of defaulting one"
        ),
    },
    {
        "reference": "scripts/build_route_comparison.py",
        "status": "CLOSED_BY_REFUSAL",
        "detail": (
            "still resolves runs with newest-id-per-family, but the hand-built bundle it passes to "
            "compare_routes is refused at the boundary, so no uncertified run reaches a load"
        ),
    },
    {
        "reference": "scripts/build_route_optimizer.py and scripts/build_route_stability.py",
        "status": "CLOSED_BY_REFUSAL",
        "detail": (
            "both hand-assemble EventBundles from hardcoded run ids and call build_event_worlds; the "
            "loader refuses them for want of an authorisation, so the prebuilt matrices they would "
            "feed to optimize are never produced"
        ),
    },
    {
        "reference": "scripts/build_final_acceptance.py and scripts/recertify_final_acceptance.py",
        "status": "CLOSED_BY_REFUSAL",
        "detail": (
            "hand-assembled EventBundles refused by build_event_worlds; the acceptance and "
            "re-certification replays are not rewritten here"
        ),
    },
    {
        "reference": "scripts/final_operational_refresh_gw04.py",
        "status": "CLOSED_BY_REFUSAL",
        "detail": (
            "still builds a hand-assembled EventBundle, which build_event_worlds now refuses; the "
            "script's own event_support_from_db readiness horizon is not consumed by any decision, "
            "and the matrix it would have re-used is presented under an explicit non-production "
            "declaration rather than as certified evidence"
        ),
    },
    {
        "reference": "scripts/gw4_current_final_board.py",
        "status": "CLOSED_BY_REFUSAL",
        "detail": "hand-assembled EventBundle refused by build_event_worlds",
    },
    {
        "reference": "scripts/live_fire_gw04.py",
        "status": "CLOSED_BY_REFUSAL",
        "detail": (
            "hand-assembled EventBundle with hardcoded run ids refused by build_event_worlds; the "
            "script is a historical GW4 replay and is not rewritten here"
        ),
    },
    {
        "reference": "scripts/run_four_gw_decision.py event_support_from_db readiness path",
        "status": "CLOSED",
        "detail": (
            "the readiness view stays a declared NON-PRODUCTION rediscovery and is never consumed by "
            "the decision, which loads only certified run ids and refuses if the discovery and "
            "exact-evaluation generations differ; PE-9 additionally RECONCILES the two published "
            "views, recording any divergence in the persisted decision provenance instead of leaving "
            "the readiness artifact and the decision artifact free to disagree"
        ),
    },
    {
        "reference": "fpl_brain.monte_carlo.load_fixture_inputs 'or {}' upstream lookup",
        "status": "CLOSED",
        "detail": (
            "an absent Minutes run now refuses with INPUT_RUN_ABSENT; reading it as empty metadata "
            "parsed the version as (0,0,0) and silently selected the legacy primitive-reconstruction "
            "path for a run whose metadata was never read"
        ),
    },
)

DISCLOSED_LIMITATION_REFERENCES: tuple[Mapping[str, str], ...] = (
    {
        "reference": "fpl_brain.candidate_universe._event_feature MISSING_PROJECTION",
        "status": "DISCLOSED",
        "detail": (
            "a player with a fixture but no certified projection row still carries 0.0 in the value, "
            "because that is what the frozen aggregation boundary computes; the zero is LABELLED "
            "MISSING_PROJECTION and reported row-by-row by unresolved_predictive_support_rows, so it "
            "is never presented as a predicted zero, and a DGW that is short rows is additionally "
            "marked partial.  PE-9 removes no value from the frozen aggregation and does not resolve "
            "this limitation."
        ),
    },
)


def disclosed_bypasses() -> list[dict[str, str]]:
    """The downstream bypasses PE-9 identified, with their declared status.

    Declared, not inferred: the certification result carries this list so the
    disclosure travels with the artifact rather than living only in review notes.
    """

    return [dict(entry) for entry in CLOSED_BYPASS_DISCLOSURES]


def disclosed_limitations() -> list[dict[str, str]]:
    """The named limitations PE-9 REPORTS rather than removes.

    A bypass can be closed; a limitation is a property of the frozen artifact, so it
    is disclosed and carried instead.  It travels with the certification result for
    the same reason the bypass register does: a reader of the persisted decision must
    be able to see that the labelled zero exists, not have to find it in a docstring.
    """

    return [dict(entry) for entry in DISCLOSED_LIMITATION_REFERENCES]


def unresolved_bypasses() -> list[dict[str, str]]:
    """The disclosed bypasses that are NOT closed, if any.

    Empty means every identified bypass is either closed or closed by refusal.  A
    non-empty result is what makes the phase terminal state ``OPEN``.
    """

    return [
        dict(entry)
        for entry in CLOSED_BYPASS_DISCLOSURES
        if str(entry.get("status")) not in {"CLOSED", "CLOSED_BY_REFUSAL"}
    ]


__all__ = [
    "BLOCKING_BUNDLE_STATES",
    "BUNDLE_DEPENDENCIES",
    "BUNDLE_STATE_PRECEDENCE",
    "BundleIncoherent",
    "CONTINUOUS_PROXY_TIE_LIMITATION",
    "CertificationArtifactAbsent",
    "CertificationArtifactContradictory",
    "CertificationArtifactMutated",
    "CertificationArtifactUnvalidated",
    "CertificationRefused",
    "CertifiedBundle",
    "CertifiedEventBundle",
    "DIAG_CERTIFICATION_ARTIFACT_ABSENT",
    "DIAG_CERTIFICATION_ARTIFACT_CONTRADICTORY",
    "DIAG_CERTIFICATION_ARTIFACT_MUTATED",
    "DIAG_CERTIFICATION_ARTIFACT_UNVALIDATED",
    "DIAG_CERTIFICATION_BOUNDARY_REQUIRED",
    "DIAG_CALIBRATION_IDENTITY_UNKNOWN",
    "DIAG_CALIBRATION_WORLD_MISMATCH",
    "DIAG_PREDICTIVE_BUNDLE_INCOHERENT",
    "LOAD_REQUIRED_FAMILIES",
    "NON_FAILURE_BUNDLE_STATES",
    "PER_BUNDLE_STATES",
    "PHASE_OPEN",
    "PHASE_READY_FOR_MERGE",
    "STATE_CALIBRATION_NOT_APPLICABLE",
    "STATE_CERTIFICATION_BLOCKER_UNRESOLVED",
    "STATE_CERTIFIED_COHERENT",
    "STATE_EVIDENCE_LIMITED",
    "STATE_EVIDENCE_MISSING",
    "STATE_PREDICTIVE_BUNDLE_INCOHERENT",
    "STATE_UNSUPPORTED_MODEL_VERSION",
    "assert_certification_artifact_bytes_unchanged",
    "assert_event_bundle_certified",
    "bundle_identity_payload",
    "bundle_state_from_reasons",
    "calibration_evidence_state",
    "calibration_identity",
    "calibration_surface_states",
    "calibration_terminal_state",
    "calibration_tokens",
    "certification_artifact_digest",
    "certification_result_identity",
    "certified_bundle_artifact_record",
    "certified_bundle_from_explicit_ids",
    "certified_bundle_payload",
    "certify_decision_horizon",
    "certify_event_bundle",
    "certify_horizon_bundles",
    "declared_required_versions",
    "disclosed_bypasses",
    "disclosed_limitations",
    "disclosure_block",
    "per_event_runs_from_bundles",
    "plain_certification_value",
    "require_certification_artifact",
    "unresolved_bypasses",
    "validate_calibration_evidence",
    "validate_certification_artifact",
    "validate_certified_bundle",
    "ValidatedCertificationArtifact",
]
