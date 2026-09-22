"""PE-6 availability / minutes challenger — challenger-first, never promotion.

WHAT THIS IS
------------
PE-6 improves the prediction of *whether and how long a player plays*.  The
incumbent is frozen: ``minutes_v1.8.0`` / ``minutes_v1.2.0`` / ``minutes_v1.5.2``
are its identity and this module neither renames nor re-points them, nor writes
through them.  A refinement must earn a promotion at PE-6's terminal boundary;
until then it lives here, under its own identity:

    ``AVAILABILITY_MINUTES_CHALLENGER_VERSION`` + ``AvailabilityMinutesChallengerConfig.config_hash()``

WHAT IT CHANGES, AND WHAT IT DELIBERATELY DOES NOT
--------------------------------------------------
The incumbent publishes four families of availability/minutes values as
**modelling assumptions rather than calibrated probabilities**.  PE-6 does not
preserve them blindly and does not replace them blindly; each family below is
either refined by *evidence that was already observable at the cutoff*, or left
declared-unchanged with that outcome stated:

* ``official_status_availability_defaults`` -> REFINED (family 1 below)
* ``bounded_scouting_modifiers`` -> UNCHANGED (see below)
* ``bounded_return_from_injury_modifiers`` -> REFINED (family 2 below)
* ``bounded_rotation_risk_modifiers`` -> REFINED (family 3 below)

*Family 1 — official status availability defaults.*  The declared table is a
statement about ``P(play | status)``.  Where a player's own causal window
contains completed rows that were played under the same **restricting** status
(``d``, ``i``, ``u``), that observed frequency replaces the declared number,
shrunk toward it with a declared prior strength.  Hard statuses (``s``, ``n``)
and the event-specific ``chance_of_playing_this_round`` signal keep their
precedence exactly as the incumbent defines them, and a scouting availability
override is never second-guessed.  Status ``a`` is deliberately NOT refined: a
0-minute row under status ``a`` is available-but-unused, which the incumbent
already models as *role* evidence (``AVAILABLE_NON_START_ZERO_MINUTES``);
treating it as unavailability would double-count non-selection.

*Family 2 — bounded return-from-injury modifiers.*  The incumbent discounts
every fixture of a returning player (``i``/``d``) by fixed factors.  The
challenger ATTENUATES those same factors by how much the player has already
demonstrated while carrying that status: the observed minutes of his own
returning-status appearances.  With no such evidence the attenuation is exactly
zero and the incumbent's modifiers stand untouched (bit-for-bit).

*Family 3 — bounded rotation-risk modifiers.*  The challenger does not re-tune
the scout-note bands.  Rotation and role uncertainty are refined through the
player's own **observed** non-selection record: the cameo rate is estimated with
the incumbent's own recency weighting (the start-rate path is recency-weighted;
the cameo path was not), and the cameo 60+ tail is estimated from the player's
own cameo appearances instead of being taken from the pool prior alone.

*``bounded_scouting_modifiers`` — UNCHANGED.*  PE-6 reports NO CHANGE for the
note bands themselves: the only available channel is the scout note, and
re-scaling the bands from the notes they are derived from would be tuning, not
evidence.  This is a PE-6 outcome, not an omission.

HOW IT IS BUILT (and why the numbers are comparable)
----------------------------------------------------
The incumbent's projection function is called UNCHANGED, once per player-fixture
with ``include_conditionals=True``, from the same PE-1 causal boundary the
incumbent uses (``analytics.completed_rows_as_of`` / ``snapshot_as_of`` /
``snapshot_history_as_of`` / ``repo.scouting_current_rows_as_of``).  No second
causal predicate exists in this module.  The challenger then applies its
declared refinements to those published marginals and re-derives the coherent
distribution with the incumbent's own coherence equations
(:func:`coherent_chain`), so ``P(available) -> P(start | available) ->
P(cameo | not start, available) -> P(0) -> P(1-59) -> P(60+) -> P(80+) ->
expected_minutes`` stays mathematically coherent in every row.

Status evidence is restricted to rows the incumbent attributes with an
EVENT-SPECIFIC status (``availability_basis == "event_context"``).  A row whose
status was only assumed from a trail that never varied proves nothing about that
row, so it is counted and excluded rather than used.

PLAYER IDENTITY IS RESOLVED AS OF THE CUTOFF
--------------------------------------------
``players.team_id`` / ``players.element_type`` / ``players.is_active`` are
CURRENT state: the store keeps one row per player and overwrites it, so a
post-cutoff transfer, position change or retirement would silently rewrite which
fixtures an earlier cutoff is projecting.  PE-6 therefore resolves every
candidate's membership, club and position from the **latest ACCEPTED official
bootstrap generation at or before the cutoff** (``bootstrap_generations``), never
from the current row:

* MEMBERSHIP is the generation's own element id set -- the official pool as it
  stood at that cutoff.  A player the cutoff places in the pool stays a candidate
  for that cutoff after he has been marked inactive, and a player only the
  persisted pool lists is not a candidate at that cutoff at all (the divergence
  is counted and reported, never silently absorbed);
* CLUB and POSITION come from that generation's OWN snapshot rows
  (``player_snapshots.raw_json`` filtered to the generation's fetch run), the
  payload the official bootstrap returned -- not from a later capture, not from
  the persisted row;
* the generation is read through ``outcome_ledger.accepted_generation_at``, the
  rule PE-5 certifies a freeze with, reused rather than restated; the id set is
  verified against the generation's recorded digest, so a corrupted identity
  fails closed instead of resolving a pool nobody recorded;
* FAIL CLOSED.  A cutoff with no usable accepted generation at or before it has
  no official pool, so no candidate is projected from the current row: the
  evaluation excludes that event's candidate enumeration at EVENT scope
  (``CANDIDATE_IDENTITY_UNAVAILABLE_AT_CUTOFF``) and scores nothing.  The live
  fallback survives only as an EXPLICIT, recorded opt-in
  (``allow_live_fallback=True``), never as a default;
* a generation member whose own snapshot row is missing or leaves the club or
  position unstated is UNRESOLVED, and an unresolved candidate is excluded with
  its reason rather than filled from the current row.

THE PRIORS ARE CONSTRUCTED HERE, FROM CUTOFF-STABLE IDENTITY
------------------------------------------------------------
``minutes_model.league_pools`` pools its position and team-position priors by
joining every historical row to the PERSISTED ``players`` row, so a later
transfer or position change moves a historical prior.  That read is the
incumbent's frozen behaviour and is not rewritten.  Instead this adapter builds
the same two tables (:func:`cutoff_stable_league_pools`) from the SAME canonical
PE-1 boundary (``historical.OBSERVATION_SQL_CLAUSES``), attributing each pooled
row by the identity resolved at the cutoff, and hands the result to the incumbent
through its own ``LeaguePools`` contract -- the argument
``project_player_fixture`` already takes.  The incumbent's own ``league_pools``
read is therefore not on this path at all.

WHERE THE NUMBERS COME FROM
---------------------------
A row on which no refinement moves a chain input REPUBLISHES the base
projection's own derived values verbatim, so an unrefined row is bit-identical to
the incumbent's.  A row that IS refined re-derives the whole chain from the
published 6-decimal marginals (:func:`coherent_chain`), so its coherence
identities hold by construction.  Either way the challenger never re-fits, re-
anchors or re-pools anything it did not declare: the only quantities it may move
are ``P(available)``, ``P(cameo | not start)``, ``P(60+ | cameo)`` and the
return-ramp factors, and every one of those movements is recorded per family
with its evidence and sample size.  The suite pins both properties.

DETERMINISM
-----------
No randomness of any kind: no RNG, no Monte Carlo, no clock-dependent branch.
The same inputs and the same cutoff produce byte-identical rows apart from
``generated_at``.  The Monte Carlo draw order is untouched, and this module is
not wired into any production decision path.
"""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass, field, fields, replace
from typing import Any, Iterable, Mapping, Sequence

from . import analytics
from . import historical_observations as historical
from . import minutes_model as incumbent
from . import outcome_ledger
from . import repositories as repo
from .utils import utc_now

AVAILABILITY_MINUTES_CHALLENGER_VERSION = "availability_minutes_challenger_v0.1.2"
# v0.1.2: review pass.  The official pool is the latest ACCEPTED bootstrap
# generation at or before the cutoff, and club/position come from that
# generation's OWN snapshot rows; a cutoff with no usable generation fails closed
# instead of falling back to the mutable current row.  The league and
# team-position priors are constructed here from cutoff-stable identity and handed
# to the incumbent through its own LeaguePools contract.
# v0.1.1: player membership, club and position were resolved per player from the
# official element capture the cutoff could see, instead of from the current
# players row.
# v0.1.0: the first PE-6 challenger.  A challenger identity, not an incumbent
# bump: nothing here is promoted, and the incumbent's own version strings are
# read-only in this module.

AVAILABILITY_MINUTES_CHALLENGER_FAMILY = "minutes_availability_challenger_pe6"

#: The incumbent identities this phase may not rename, re-point or overwrite.
FROZEN_INCUMBENT_VERSIONS: dict[str, str] = {
    "minutes_model": "minutes_v1.8.0",
    "minutes_coherent_model": "minutes_v1.2.0",
    "joint_minutes_model": "minutes_v1.5.2",
}


def frozen_incumbent_identity() -> dict[str, Any]:
    """The three frozen identities as the running code actually declares them.

    Read from the modules, never restated: the point of the check is to fail
    closed if a later change re-points one of them while claiming not to.
    """

    from . import joint_minutes

    observed = {
        "minutes_model": incumbent.MINUTES_MODEL_VERSION,
        "minutes_coherent_model": incumbent.MINUTES_COHERENT_MODEL_VERSION,
        "joint_minutes_model": joint_minutes.JOINT_MINUTES_MODEL_VERSION,
    }
    changed = sorted(
        name for name, declared in FROZEN_INCUMBENT_VERSIONS.items() if observed.get(name) != declared
    )
    return {
        "declared": dict(FROZEN_INCUMBENT_VERSIONS),
        "observed": observed,
        "unchanged": not changed,
        "changed": changed,
    }


# ---------------------------------------------------------------------------
# Refinement families.
# ---------------------------------------------------------------------------

FAMILY_AVAILABILITY_STATUS_EVIDENCE = "availability_status_evidence"
FAMILY_RETURN_FROM_INJURY_RAMP = "return_from_injury_ramp"
FAMILY_CAMEO_RATE_RECENCY = "cameo_rate_recency"
FAMILY_CAMEO_TAIL_EVIDENCE = "cameo_tail_evidence"

#: The families the challenger can turn on, in report order.
REFINEMENT_FAMILIES: tuple[str, ...] = (
    FAMILY_AVAILABILITY_STATUS_EVIDENCE,
    FAMILY_RETURN_FROM_INJURY_RAMP,
    FAMILY_CAMEO_RATE_RECENCY,
    FAMILY_CAMEO_TAIL_EVIDENCE,
)

#: The contract's four assumption families -> the PE-6 treatment of each.
ASSUMPTION_FAMILY_TREATMENT: dict[str, str] = {
    "official_status_availability_defaults": "REFINED",
    "bounded_scouting_modifiers": "UNCHANGED_BY_CHALLENGER",
    "bounded_return_from_injury_modifiers": "REFINED",
    "bounded_rotation_risk_modifiers": "REFINED",
}

#: ``contract assumption family`` -> the refinement family that addresses it.
ASSUMPTION_FAMILY_REFINEMENT: dict[str, str | None] = {
    "official_status_availability_defaults": FAMILY_AVAILABILITY_STATUS_EVIDENCE,
    "bounded_scouting_modifiers": None,
    "bounded_return_from_injury_modifiers": FAMILY_RETURN_FROM_INJURY_RAMP,
    "bounded_rotation_risk_modifiers": FAMILY_CAMEO_RATE_RECENCY,
}

#: The incumbent's status-at-event attribution basis for a row whose status is
#: genuinely event-specific.  Only such rows may be refined evidence: a status
#: attributed to a row merely because the trail never varied
#: (``assumed_constant_status``) is not proof that the status applied then.
EVENT_SPECIFIC_ATTRIBUTION = "event_context"

#: Only a *restricting* soft status measures ``P(play | status)``.  Status ``a``
#: is excluded on purpose: a 0-minute row under ``a`` is available-but-unused,
#: which is role evidence, not unavailability (see the module docstring).
SOFT_RESTRICTING_STATUSES: tuple[str, ...] = ("d", "i", "u")

#: Availability-source categories, derived structurally from the incumbent's own
#: published ``availability_source_summary``, never from its prose label.
BASIS_HARD_STATUS = "HARD_UNAVAILABLE_STATUS"
BASIS_EVENT_CHANCE = "EVENT_SPECIFIC_CHANCE_SIGNAL"
BASIS_DECLARED_STATUS_DEFAULT = "DECLARED_STATUS_AVAILABILITY_DEFAULT"
BASIS_NO_RESTRICTION = "NO_OFFICIAL_RESTRICTION"
BASIS_SCOUTING_OVERRIDE = "SCOUTING_AVAILABILITY_OVERRIDE"

AVAILABILITY_BASES: tuple[str, ...] = (
    BASIS_HARD_STATUS,
    BASIS_EVENT_CHANCE,
    BASIS_DECLARED_STATUS_DEFAULT,
    BASIS_NO_RESTRICTION,
    BASIS_SCOUTING_OVERRIDE,
)

#: The incumbent's scouting flag for a high injury-uncertainty note.  Mirrored
#: here as a named constant; the suite pins it against the incumbent's own fold.
SCOUT_RETURN_RAMP_FLAG = "RETURN_RAMP_SCOUT_HIGH"

#: Per-family outcome of one row, so an audit can tell "no evidence" from
#: "switched off" from "applied".
REFINEMENT_APPLIED = "APPLIED"
REFINEMENT_NOT_APPLICABLE = "NOT_APPLICABLE"
REFINEMENT_NO_EVIDENCE = "NO_EVIDENCE_FALLBACK_TO_INCUMBENT"
REFINEMENT_DISABLED = "DISABLED_IN_THIS_ARM"

#: Evidence classes that prove the player was available around the event.  Built
#: from the incumbent's public class constants; the suite pins it against the
#: incumbent's own (private) tuple so the two cannot silently drift.
START_OBSERVATION_CLASSES: tuple[str, ...] = (
    incumbent.EVIDENCE_STARTED,
    incumbent.EVIDENCE_BENCH_APPEARANCE,
    incumbent.EVIDENCE_AVAILABLE_NON_START_ZERO_MINUTES,
    incumbent.EVIDENCE_UNUSED_SUB_AVAILABLE,
    incumbent.EVIDENCE_NOT_IN_MATCHDAY_SQUAD,
)


class ChallengerInconsistencyError(RuntimeError):
    """The challenger and the incumbent disagree about the evidence they share.

    Fail closed: a disagreement means one of them is describing a different
    world, and a silently reconciled number is exactly what PE-6 forbids.
    """


# ---------------------------------------------------------------------------
# Configuration and identity.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AvailabilityMinutesChallengerConfig:
    """Every challenger tunable in one versioned structure; no magic numbers.

    ``None`` means "inherit the incumbent's declared value", so the challenger
    introduces no second knob for a quantity the incumbent already declares.
    """

    #: Prior weight (in rows) pulling a status estimate back to the declared
    #: assumption.  Four rows of prior per row of evidence is a deliberately
    #: cautious reading of a thin per-player status history.
    status_evidence_prior_strength: float = 4.0
    #: Fewer observed rows than this and the declared assumption stands: a
    #: single observation is not a calibration.
    status_evidence_min_rows: int = 2
    #: Observed minutes at which the return ramp is fully attenuated.
    ramp_recovery_reference_minutes: float = 90.0
    #: Appearances required before the ramp is attenuated at all.
    ramp_recovery_min_rows: int = 1
    #: ``None`` -> the incumbent's ``start_recency_half_life_matches``.
    cameo_recency_half_life_matches: float | None = None
    #: ``None`` -> the incumbent's ``cameo_p60_prior_strength``.
    cameo_tail_prior_strength: float | None = None

    def config_hash(self) -> str:
        values = {item.name: getattr(self, item.name) for item in fields(self)}
        return analytics.canonical_hash({"model": AVAILABILITY_MINUTES_CHALLENGER_VERSION, **values})

    def resolved(self, incumbent_config: "incumbent.MinutesModelConfig") -> dict[str, float]:
        """The effective values, with ``None`` inheriting from the incumbent."""

        return {
            "status_evidence_prior_strength": float(self.status_evidence_prior_strength),
            "status_evidence_min_rows": float(self.status_evidence_min_rows),
            "ramp_recovery_reference_minutes": float(self.ramp_recovery_reference_minutes),
            "ramp_recovery_min_rows": float(self.ramp_recovery_min_rows),
            "cameo_recency_half_life_matches": float(
                self.cameo_recency_half_life_matches
                if self.cameo_recency_half_life_matches is not None
                else incumbent_config.start_recency_half_life_matches
            ),
            "cameo_tail_prior_strength": float(
                self.cameo_tail_prior_strength
                if self.cameo_tail_prior_strength is not None
                else incumbent_config.cameo_p60_prior_strength
            ),
        }


def challenger_identity(
    config: AvailabilityMinutesChallengerConfig | None = None,
    incumbent_config: "incumbent.MinutesModelConfig | None" = None,
) -> dict[str, Any]:
    """The challenger's own identity, carried by every artifact it produces."""

    config = config or AvailabilityMinutesChallengerConfig()
    incumbent_config = incumbent_config or incumbent.MinutesModelConfig()
    return {
        "family": AVAILABILITY_MINUTES_CHALLENGER_FAMILY,
        "challenger_version": AVAILABILITY_MINUTES_CHALLENGER_VERSION,
        "challenger_config_hash": config.config_hash(),
        "resolved_parameters": config.resolved(incumbent_config),
        "refinement_families": list(REFINEMENT_FAMILIES),
        "assumption_family_treatment": dict(ASSUMPTION_FAMILY_TREATMENT),
        "incumbent_identity": frozen_incumbent_identity(),
        "incumbent_config_hash": incumbent_config.config_hash(),
        "promotion": "NOT_PERFORMED_CHALLENGER_ONLY",
    }


# ---------------------------------------------------------------------------
# Cutoff-stable player identity (membership, club, position).
# ---------------------------------------------------------------------------

#: Where a candidate's identity came from, in the order of preference.
IDENTITY_BASIS_CUTOFF_GENERATION = "CUTOFF_ACCEPTED_BOOTSTRAP_GENERATION"
IDENTITY_BASIS_LIVE_FALLBACK = "CURRENT_PLAYER_ROW_FALLBACK_NOT_CUTOFF_SAFE"
IDENTITY_BASIS_UNRESOLVED = "NO_IDENTITY_EVIDENCE"

IDENTITY_BASES: tuple[str, ...] = (
    IDENTITY_BASIS_CUTOFF_GENERATION,
    IDENTITY_BASIS_LIVE_FALLBACK,
    IDENTITY_BASIS_UNRESOLVED,
)

#: The official bootstrap element-payload fields the identity is read from.
OFFICIAL_ELEMENT_CLUB_FIELD = "team"
OFFICIAL_ELEMENT_POSITION_FIELD = "element_type"

#: Why a cutoff has no usable official pool generation.  Each is a fail-closed
#: refusal, never a reason to read the persisted row.
GENERATION_UNAVAILABLE_NO_ACCEPTED_GENERATION = (
    "NO_ACCEPTED_OFFICIAL_BOOTSTRAP_GENERATION_AT_OR_BEFORE_THE_CUTOFF"
)
GENERATION_UNAVAILABLE_EMPTY_POOL = "GENERATION_RECORDS_AN_EMPTY_OFFICIAL_ELEMENT_ID_SET"
GENERATION_UNAVAILABLE_ID_SET_DIGEST_MISMATCH = (
    "GENERATION_ELEMENT_ID_SET_DOES_NOT_MATCH_ITS_RECORDED_DIGEST"
)

#: The generation's OWN snapshot rows, selected by the fetch run the generation
#: was recorded from.  ``?`` order: fetch_run_id, cutoff.
GENERATION_SNAPSHOT_SQL = (
    "SELECT player_id, id, captured_at, fetch_run_id, raw_json FROM player_snapshots "
    "WHERE fetch_run_id = ? AND captured_at <= ? ORDER BY player_id, id DESC"
)

#: The same rows for a generation recorded without a fetch run: the capture
#: moment is then the only identity the generation carries.  ``?`` order:
#: captured_at, cutoff.
GENERATION_SNAPSHOT_BY_CAPTURE_SQL = (
    "SELECT player_id, id, captured_at, fetch_run_id, raw_json FROM player_snapshots "
    "WHERE captured_at = ? AND captured_at <= ? ORDER BY player_id, id DESC"
)

#: Every recorded generation attempt at or before the cutoff, whatever its
#: acceptance, newest first (the caller takes the first row).  It is what makes
#: "the latest ACCEPTED generation" auditable rather than asserted: a newer
#: attempt is visible, with the reasons it was refused.  ``?`` order: cutoff.
GENERATION_ATTEMPT_SQL = (
    "SELECT id, captured_at, accepted, rejection_reasons_json, acceptance_rule "
    "FROM bootstrap_generations WHERE captured_at <= ? ORDER BY captured_at DESC, id DESC"
)

#: The historical rows the cutoff-stable priors are pooled from.  The predicate
#: is NOT restated: ``historical.OBSERVATION_SQL_CLAUSES`` is the PE-1 boundary,
#: interpolated exactly as the incumbent's own pooling query interpolates it, so
#: PE-6 pools over the same window the incumbent does.  ``?`` order comes from
#: ``historical.boundary_params``.
PRIORS_SQL_TEMPLATE = (
    "SELECT pg.player_id AS player_id, pg.starts AS started, pg.minutes AS minutes "
    "FROM player_gameweeks pg JOIN fixtures f ON f.id=pg.fixture_id "
    "WHERE {boundary} AND pg.starts IS NOT NULL AND pg.minutes IS NOT NULL"
)


def _int_or_none(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _json_list(text: Any) -> list[Any]:
    """A JSON list column, or an empty list.  Unparseable is empty, never a guess."""

    if isinstance(text, (list, tuple)):
        return list(text)
    if isinstance(text, str) and text.strip():
        try:
            parsed = json.loads(text)
        except (TypeError, ValueError):
            return []
        return list(parsed) if isinstance(parsed, list) else []
    return []


def _generation_element_ids(generation: Mapping[str, Any]) -> list[int]:
    """The generation's recorded element id set, sorted and de-duplicated.

    ``element_ids_json`` is the column ``bootstrap_generations`` stores it in; a
    caller that read the row through ``repositories.latest_accepted_bootstrap_generation``
    arrives with it already parsed.  Both shapes are accepted, and neither is
    guessed at: an unparseable set is an EMPTY set, which fails closed.
    """

    raw = generation.get("element_ids")
    if raw is None:
        raw = generation.get("element_ids_json")
    return sorted({int(pid) for pid in _json_list(raw)})


def _official_element_payload(capture: Mapping[str, Any] | None) -> dict[str, Any]:
    """The official element payload a capture row carries, or an empty mapping.

    ``raw_json`` is the bootstrap element the capture was parsed from, stored as
    text by the repository layer.  Anything unparseable yields no fields rather
    than a guess: a missing identity is missing, never zero.
    """

    raw = (capture or {}).get("raw_json")
    if isinstance(raw, Mapping):
        return dict(raw)
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError):
            return {}
        return dict(parsed) if isinstance(parsed, Mapping) else {}
    return {}


@dataclass(frozen=True)
class PlayerIdentityAsOf:
    """One player's membership, club and position as known at one cutoff."""

    player_id: int
    team_id: int | None
    element_type: int | None
    in_official_pool: bool
    basis: str
    #: The accepted generation the pool and its snapshot rows were read from.
    generation_id: int | None
    generation_captured_at: str | None
    #: The generation snapshot row the club/position were read off, if any.
    snapshot_captured_at: str | None
    #: True only when membership, club and position ALL came from the cutoff
    #: generation, so a later change to the current row cannot move this identity.
    cutoff_safe: bool
    live_row_disagreement: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "player_id": int(self.player_id),
            "team_id": None if self.team_id is None else int(self.team_id),
            "element_type": None if self.element_type is None else int(self.element_type),
            "in_official_pool": bool(self.in_official_pool),
            "basis": str(self.basis),
            "generation_id": None if self.generation_id is None else int(self.generation_id),
            "generation_captured_at": self.generation_captured_at,
            "snapshot_captured_at": self.snapshot_captured_at,
            "cutoff_safe": bool(self.cutoff_safe),
            "live_row_disagreement": list(self.live_row_disagreement),
            "notes": list(self.notes),
        }


def _identity_from_generation(
    player_id: int,
    generation: Mapping[str, Any],
    capture: Mapping[str, Any] | None,
    live_row: Mapping[str, Any] | None,
    *,
    allow_live_fallback: bool,
) -> PlayerIdentityAsOf:
    """Resolve one generation member's identity, preferring its own capture row.

    MEMBERSHIP is already settled by the generation (he is in its element id
    set); this resolves the club and the position.  The capture's own ``team`` /
    ``element_type`` are the only cutoff-safe source.  A capture that is missing,
    unparseable or silent about a field resolves NOTHING unless the caller has
    explicitly opted into the live fallback, and a fallback-filled identity is
    never reported as cutoff-safe.

    ``live_row`` is the CURRENT ``players`` row, supplied explicitly by a caller
    that has one.  It is read for two reasons only: as that recorded opt-in
    fallback, and to RECORD how far the persisted state diverges from the
    cutoff's (``live_row_disagreement``).  It never decides an identity by
    default.
    """

    generation_id = _int_or_none(generation.get("id"))
    generation_captured_at = (
        None if generation.get("captured_at") is None else str(generation["captured_at"])
    )
    notes: list[str] = []
    from_current_row: list[str] = []
    snapshot_captured_at: str | None = None
    if capture is None:
        team: int | None = None
        position: int | None = None
        notes.append("THE_CUTOFF_GENERATION_HOLDS_NO_SNAPSHOT_ROW_FOR_THIS_PLAYER")
    else:
        payload = _official_element_payload(capture)
        team = _int_or_none(payload.get(OFFICIAL_ELEMENT_CLUB_FIELD))
        position = _int_or_none(payload.get(OFFICIAL_ELEMENT_POSITION_FIELD))
        captured_at = capture.get("captured_at")
        snapshot_captured_at = None if captured_at is None else str(captured_at)
        if team is None:
            notes.append("GENERATION_CAPTURE_CARRIES_NO_CLUB_FIELD")
        if position is None:
            notes.append("GENERATION_CAPTURE_CARRIES_NO_POSITION_FIELD")

    live_team = _int_or_none((live_row or {}).get("team_id"))
    live_position = _int_or_none((live_row or {}).get("element_type"))
    if (team is None or position is None) and live_row is not None and allow_live_fallback:
        if team is None:
            team = live_team
            from_current_row.append("club")
        if position is None:
            position = live_position
            from_current_row.append("position")
        if from_current_row:
            notes.append("FILLED_FROM_CURRENT_ROW:" + "+".join(from_current_row))

    if team is not None and position is not None:
        if from_current_row:
            basis = IDENTITY_BASIS_LIVE_FALLBACK
        else:
            basis = IDENTITY_BASIS_CUTOFF_GENERATION
    else:
        basis = IDENTITY_BASIS_UNRESOLVED
        if live_row is not None and not allow_live_fallback:
            notes.append("THE_CURRENT_ROW_IS_NOT_AN_ADMITTED_SUBSTITUTE")

    disagreement: list[str] = []
    if live_row is None:
        disagreement.append("membership:in_pool->absent_from_persisted_pool")
    else:
        if live_team is not None and team is not None and live_team != team:
            disagreement.append(f"club:{team}->{live_team}")
        if live_position is not None and position is not None and live_position != position:
            disagreement.append(f"position:{position}->{live_position}")
        if not live_row.get("is_active"):
            # The cutoff generation places him in the pool; the current row says
            # he is gone.  The cutoff wins, and the disagreement is recorded.
            disagreement.append("membership:in_pool->inactive")
    return PlayerIdentityAsOf(
        player_id=int(player_id),
        team_id=team,
        element_type=position,
        in_official_pool=True,
        basis=basis,
        generation_id=generation_id,
        generation_captured_at=generation_captured_at,
        snapshot_captured_at=snapshot_captured_at,
        cutoff_safe=basis == IDENTITY_BASIS_CUTOFF_GENERATION,
        live_row_disagreement=tuple(disagreement),
        notes=tuple(notes),
    )


def generation_snapshot_rows(
    conn: sqlite3.Connection, generation: Mapping[str, Any], cutoff: str
) -> dict[int, dict[str, Any]]:
    """The generation's own snapshot rows, newest first per player.

    ``fetch_run_id`` is the generation's own record of the fetch it was parsed
    from, so its snapshot rows are selected by that run.  A generation recorded
    without one is selected by its capture moment instead: that is the only
    identity such a row carries.  Either way the rows must have been observable
    at the cutoff.
    """

    fetch_run_id = _int_or_none(generation.get("fetch_run_id"))
    captured_at = "" if generation.get("captured_at") is None else str(generation["captured_at"])
    if fetch_run_id is None:
        rows = conn.execute(
            GENERATION_SNAPSHOT_BY_CAPTURE_SQL, (captured_at, str(cutoff))
        ).fetchall()
    else:
        rows = conn.execute(GENERATION_SNAPSHOT_SQL, (fetch_run_id, str(cutoff))).fetchall()
    captures: dict[int, dict[str, Any]] = {}
    for row in rows:
        player_id = int(row["player_id"])
        if player_id in captures:  # the first row per player is the as-of winner
            continue
        captures[player_id] = {
            "player_id": player_id,
            "id": row["id"],
            "captured_at": row["captured_at"],
            "fetch_run_id": row["fetch_run_id"],
            "raw_json": row["raw_json"],
        }
    return captures


def official_pool_generation_at_cutoff(conn: sqlite3.Connection, cutoff: str) -> dict[str, Any]:
    """The official pool generation a prediction at ``cutoff`` must resolve from.

    The reader is ``outcome_ledger.accepted_generation_at``: the accepted
    generation at or before ``observed_at``, which is the rule PE-5 certifies a
    freeze with.  It is reused rather than restated, so "latest accepted at or
    before the cutoff" has exactly one definition in the codebase.

    FAIL CLOSED: the returned ``usable`` is False -- with the reason -- when there
    is no accepted generation at all at or before the cutoff, when the recorded
    element id set is empty, or when that set does not hash to the digest the
    generation recorded.  A caller must then project nothing rather than read the
    mutable current row.
    """

    generation = outcome_ledger.accepted_generation_at(conn, observed_at=str(cutoff))
    attempt_row = conn.execute(GENERATION_ATTEMPT_SQL, (str(cutoff),)).fetchone()
    attempt: dict[str, Any] | None = None
    if attempt_row is not None:
        record = dict(attempt_row)
        attempt = {
            "id": _int_or_none(record.get("id")),
            "captured_at": None if record.get("captured_at") is None else str(record["captured_at"]),
            "accepted": bool(record.get("accepted")),
            "acceptance_rule": record.get("acceptance_rule"),
            "rejection_reasons": _json_list(record.get("rejection_reasons_json")),
        }
    reasons: list[str] = []
    element_ids: list[int] = []
    digest_matches: bool | None = None
    generation_block: dict[str, Any] | None = None
    if generation is None:
        reasons.append(GENERATION_UNAVAILABLE_NO_ACCEPTED_GENERATION)
    else:
        element_ids = _generation_element_ids(generation)
        declared_digest = str(generation.get("element_ids_sha256") or "")
        observed_digest = repo.element_ids_identity(element_ids)[1]
        digest_matches = bool(declared_digest) and declared_digest == observed_digest
        if not element_ids:
            reasons.append(GENERATION_UNAVAILABLE_EMPTY_POOL)
        elif not digest_matches:
            reasons.append(GENERATION_UNAVAILABLE_ID_SET_DIGEST_MISMATCH)
        generation_block = {
            "id": _int_or_none(generation.get("id")),
            "captured_at": (
                None if generation.get("captured_at") is None else str(generation["captured_at"])
            ),
            "fetch_run_id": _int_or_none(generation.get("fetch_run_id")),
            "accepted": True,
            "official_element_count": _int_or_none(generation.get("official_element_count")),
            "element_ids_count": len(element_ids),
            "element_ids_sha256": declared_digest or None,
            "element_ids_sha256_matches_the_id_set": digest_matches,
            "acceptance_rule": generation.get("acceptance_rule"),
            "acceptance_rule_version": generation.get("acceptance_rule_version"),
        }
    return {
        "cutoff": str(cutoff),
        "reader": (
            "outcome_ledger.accepted_generation_at: the latest ACCEPTED official bootstrap "
            "generation at or before the cutoff"
        ),
        "usable": not reasons,
        "reasons": reasons,
        "generation": generation_block,
        "element_ids": element_ids,
        "newest_attempt": attempt,
        # None when there is no attempt to compare; a False here means a newer
        # attempt at or before the cutoff was REFUSED, which is exactly why the
        # accepted one is not the newest row in the table.
        "newest_attempt_is_the_accepted_generation": (
            None
            if attempt is None or generation_block is None
            else _int_or_none(attempt.get("id")) == generation_block["id"]
        ),
    }


def generation_artifact_block(pool_generation: Mapping[str, Any]) -> dict[str, Any]:
    """The pool-generation summary as an artifact block, without the raw id list."""

    block = {key: value for key, value in dict(pool_generation).items() if key != "element_ids"}
    block["element_ids_count"] = len(pool_generation.get("element_ids") or [])
    block["accepted_generation"] = block.pop("generation", None)
    return block


def resolve_candidate_pool(
    conn: sqlite3.Connection,
    cutoff: str,
    *,
    generation: Mapping[str, Any] | None = None,
    live_rows: Sequence[Mapping[str, Any]] | None = None,
    allow_live_fallback: bool = False,
) -> dict[str, Any]:
    """One pass: the candidate pool, the unresolved candidates and the summary.

    The candidate set IS the official pool at the cutoff: the element ids of the
    latest ACCEPTED bootstrap generation at or before it.  Club and position come
    from that generation's own snapshot rows.  A player only the persisted pool
    lists is not a candidate here -- the generation did not place him in the
    official pool at that cutoff -- and the divergence between the two is
    reported in the summary rather than absorbed.

    A generation member whose own row is missing, unparseable, or leaves the club
    or position unstated is returned in ``unresolved`` rather than dropped, so a
    caller can classify it instead of losing it silently.  The persisted row is
    read ONLY through the explicit ``allow_live_fallback`` opt-in, and such a row
    is never reported as cutoff-safe.

    ``identity_available`` False means the cutoff has no usable official pool at
    all and the caller must fail closed: nothing is projected.
    """

    cutoff = str(cutoff)
    pool_generation = (
        dict(generation) if generation is not None else official_pool_generation_at_cutoff(conn, cutoff)
    )
    live = {
        int(row["player_id"]): dict(row)
        for row in (live_rows if live_rows is not None else analytics.projectable_players(conn))
    }
    if not pool_generation.get("usable"):
        return {
            "cutoff": cutoff,
            "identity_available": False,
            "generation": generation_artifact_block(pool_generation),
            "players": [],
            "unresolved": [],
            "summary": identity_resolution_summary(
                [],
                pool_generation=pool_generation,
                unresolved=[],
                persisted_pool_ids=sorted(live),
                identity_available=False,
            ),
        }
    generation_row = pool_generation.get("generation") or {}
    captures = generation_snapshot_rows(conn, generation_row, cutoff)
    players: list[dict[str, Any]] = []
    unresolved: list[dict[str, Any]] = []
    for player_id in [int(pid) for pid in (pool_generation.get("element_ids") or [])]:
        identity = _identity_from_generation(
            player_id,
            generation_row,
            captures.get(player_id),
            live.get(player_id),
            allow_live_fallback=allow_live_fallback,
        )
        if identity.team_id is None or identity.element_type is None:
            unresolved.append(identity.as_dict())
            continue
        snapshot = captures.get(player_id) or {}
        players.append(
            {
                "player_id": int(player_id),
                "team_id": int(identity.team_id),
                "element_type": int(identity.element_type),
                "is_active": 1,
                "identity": identity.as_dict(),
                "identity_basis": identity.basis,
                "identity_cutoff_safe": bool(identity.cutoff_safe),
                "identity_generation_id": identity.generation_id,
                "identity_generation_captured_at": identity.generation_captured_at,
                "identity_snapshot_id": _int_or_none(snapshot.get("id")),
                "identity_snapshot_captured_at": identity.snapshot_captured_at,
            }
        )
    return {
        "cutoff": cutoff,
        "identity_available": True,
        "generation": generation_artifact_block(pool_generation),
        "players": players,
        "unresolved": unresolved,
        "summary": identity_resolution_summary(
            players,
            pool_generation=pool_generation,
            unresolved=unresolved,
            persisted_pool_ids=sorted(live),
            identity_available=True,
        ),
    }


def projectable_players_as_of(
    conn: sqlite3.Connection,
    cutoff: str,
    *,
    generation: Mapping[str, Any] | None = None,
    live_rows: Sequence[Mapping[str, Any]] | None = None,
    allow_live_fallback: bool = False,
) -> list[dict[str, Any]]:
    """The candidate pool with every identity resolved as of ``cutoff``."""

    return resolve_candidate_pool(
        conn,
        cutoff,
        generation=generation,
        live_rows=live_rows,
        allow_live_fallback=allow_live_fallback,
    )["players"]


def unresolved_candidates_as_of(
    conn: sqlite3.Connection,
    cutoff: str,
    *,
    generation: Mapping[str, Any] | None = None,
    live_rows: Sequence[Mapping[str, Any]] | None = None,
    allow_live_fallback: bool = False,
) -> list[dict[str, Any]]:
    """Generation members the cutoff cannot place, with their reasons."""

    return resolve_candidate_pool(
        conn,
        cutoff,
        generation=generation,
        live_rows=live_rows,
        allow_live_fallback=allow_live_fallback,
    )["unresolved"]


def identity_basis_counts(players: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """How many candidates were resolved from each identity basis."""

    counts: dict[str, int] = {}
    for player in players:
        basis = str(player.get("identity_basis") or IDENTITY_BASIS_UNRESOLVED)
        counts[basis] = counts.get(basis, 0) + 1
    return dict(sorted(counts.items()))


def identity_resolution_summary(
    players: Sequence[Mapping[str, Any]],
    *,
    pool_generation: Mapping[str, Any] | None = None,
    unresolved: Sequence[Mapping[str, Any]] = (),
    persisted_pool_ids: Sequence[int] = (),
    identity_available: bool = True,
) -> dict[str, Any]:
    """The identity provenance of one candidate pool, for the artifact."""

    rows = list(players)
    safe = [row for row in rows if row.get("identity_cutoff_safe")]
    reasons: dict[str, int] = {}
    for item in unresolved:
        for note in item.get("notes") or []:
            reasons[str(note)] = reasons.get(str(note), 0) + 1
    generation_ids = {
        int(pid) for pid in ((pool_generation or {}).get("element_ids") or [])
    }
    persisted_ids = {int(pid) for pid in persisted_pool_ids}
    divergence = (
        None
        if not identity_available or not generation_ids
        else {
            "cutoff_generation_pool_size": len(generation_ids),
            "persisted_pool_size": len(persisted_ids),
            "in_cutoff_generation_not_in_persisted_pool": len(generation_ids - persisted_ids),
            "in_persisted_pool_not_in_cutoff_generation": len(persisted_ids - generation_ids),
            "rule": (
                "the cutoff generation defines the official pool at that cutoff: a player only "
                "the persisted pool lists is NOT a candidate here (the generation did not place "
                "him in the pool), and a player only the generation lists IS a candidate whatever "
                "the persisted row says now"
            ),
            "scope": "reported, never scored: this divergence describes the store, not the model",
        }
    )
    return {
        "identity_available": bool(identity_available),
        "rule": (
            "membership comes from the latest ACCEPTED official bootstrap generation at or before "
            "the cutoff; club and position come from that generation's own snapshot rows; the "
            "persisted players row is never used unless a caller explicitly opts in"
        ),
        "generation": None if pool_generation is None else generation_artifact_block(pool_generation),
        "rows": len(rows),
        "basis_counts": identity_basis_counts(rows),
        "cutoff_safe_rows": len(safe),
        "cutoff_safe_share": (round(len(safe) / len(rows), 6) if rows else None),
        "unresolved_candidates": len(unresolved),
        "unresolved_reasons": dict(sorted(reasons.items())),
        "persisted_pool_divergence": divergence,
    }


# ---------------------------------------------------------------------------
# Cutoff-stable league / team-position priors.
# ---------------------------------------------------------------------------


def cutoff_stable_league_pools(
    conn: sqlite3.Connection,
    planning_event: int,
    cutoff: str,
    identities: Mapping[int, tuple[int, int]],
) -> tuple["incumbent.LeaguePools", dict[str, Any]]:
    """The incumbent's priors, pooled from cutoff-stable identity.

    ``minutes_model.league_pools`` joins every historical row it pools to the
    PERSISTED ``players`` row for that row's club and position, so a post-cutoff
    transfer or position change silently moves a historical prior.  That read is
    the incumbent's frozen behaviour; this adapter does not rewrite it.  Instead
    it builds the same two tables -- the position pools and the team-position
    pools, with the incumbent's own keys, formulas and rounding -- over the SAME
    canonical PE-1 boundary, attributing each pooled row by the identity resolved
    at the cutoff, and returns them through the incumbent's own ``LeaguePools``
    contract, which ``project_player_fixture`` already accepts.

    ``identities`` maps ``player_id -> (team_id, element_type)`` as of the cutoff.
    A row whose player has no cutoff identity is not pooled (the incumbent's own
    query drops such rows by its inner join too) and is COUNTED, so the share of
    the window this affects is visible rather than implied.
    """

    rows = conn.execute(
        PRIORS_SQL_TEMPLATE.format(boundary=historical.OBSERVATION_SQL_CLAUSES),
        historical.boundary_params(cutoff, planning_event=int(planning_event)),
    ).fetchall()
    position_pools: dict[int, dict[str, float]] = {}
    team_position: dict[tuple[int, int], dict[str, float]] = {}
    pooled_players: set[int] = set()
    unattributed_rows = 0
    for row in rows:
        player_id = int(row["player_id"])
        identity = identities.get(player_id)
        if identity is None:
            unattributed_rows += 1
            continue
        team_id = int(identity[0])
        position = int(identity[1])
        started = bool(row["started"])
        minutes = float(row["minutes"])
        pool = position_pools.setdefault(
            position,
            {
                "starts": 0.0,
                "rows": 0.0,
                "start_minutes": 0.0,
                "p60": 0.0,
                "p80": 0.0,
                "cameo_rows": 0.0,
                "cameo_minutes_total": 0.0,
                "cameo60": 0.0,
            },
        )
        pool["rows"] += 1
        if started:
            pool["starts"] += 1
            pool["start_minutes"] += minutes
            pool["p60"] += 1 if minutes >= 60 else 0
            pool["p80"] += 1 if minutes >= 80 else 0
        elif minutes > 0:
            pool["cameo_rows"] += 1
            pool["cameo_minutes_total"] += minutes
            pool["cameo60"] += 1 if minutes >= 60 else 0
        team_pool = team_position.setdefault((team_id, position), {"rows": 0.0, "starts": 0.0})
        team_pool["rows"] += 1
        team_pool["starts"] += 1 if started else 0
        pooled_players.add(player_id)

    position: dict[int, dict[str, float]] = {}
    for pos, pool in position_pools.items():
        starts = float(pool["starts"])
        rows_count = float(pool["rows"])
        not_start_rows = rows_count - starts
        cameo_rows = float(pool["cameo_rows"])
        cameo_minutes = pool["cameo_minutes_total"] / cameo_rows if cameo_rows else 15.0
        position[pos] = {
            "start_rows": int(rows_count),
            "p_start": round(starts / rows_count, 6) if rows_count else 0.0,
            "minutes_if_start": round(pool["start_minutes"] / starts, 6) if starts else 0.0,
            "p60_if_start": round(pool["p60"] / starts, 6) if starts else 0.0,
            "p80_if_start": round(pool["p80"] / starts, 6) if starts else 0.0,
            "not_start_rows": int(not_start_rows),
            "cameo_rate": round(cameo_rows / not_start_rows, 6) if not_start_rows else 0.0,
            "cameo_rows": int(cameo_rows),
            "cameo_minutes": round(cameo_minutes, 6),
            "cameo_p60": round(pool["cameo60"] / cameo_rows, 6) if cameo_rows else 0.0,
        }
    disclosure = {
        "construction": "PE-6 ADAPTER: cutoff_stable_league_pools",
        "boundary": "historical_observations.OBSERVATION_SQL_CLAUSES (the PE-1 canonical boundary)",
        "identity_source": (
            "the accepted official bootstrap generation at or before the cutoff, per pooled row"
        ),
        "incumbent_league_pools_read_used": False,
        "incumbent_read_note": (
            "minutes_model.league_pools joins its pooled rows to the PERSISTED players row; that "
            "frozen read is not on this path"
        ),
        "pooled_rows": len(rows),
        "pooled_rows_attributed": len(rows) - unattributed_rows,
        "pooled_rows_without_cutoff_identity": unattributed_rows,
        "pooled_players": len(pooled_players),
        "position_keys": sorted(position),
        "team_position_keys": len(team_position),
    }
    return incumbent.LeaguePools(position=position, team_position=team_position), disclosure


# ---------------------------------------------------------------------------
# Small math mirrors.
#
# These three formulas are the incumbent's, restated here rather than imported
# from its private namespace.  The suite pins each mirror against the
# incumbent's own function on a grid, so they cannot drift silently.
# ---------------------------------------------------------------------------


def _shrink(evidence_total: float, evidence_count: float, prior: float, strength: float) -> float:
    """``(evidence + strength * prior) / (count + strength)`` (incumbent's form)."""

    return (evidence_total + strength * prior) / (evidence_count + strength)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _round(value: float, digits: int = 6) -> float:
    return round(value, digits)


def _recency_weights(count: int, half_life: float) -> list[float]:
    """``2 ** -(index / half_life)``, newest first (incumbent's form)."""

    return [
        math.pow(2.0, -(index / max(float(half_life), 1e-6))) for index in range(int(count))
    ]


def _availability_basis(source_summary: Mapping[str, Any], config: "incumbent.MinutesModelConfig") -> str:
    """Which availability rule produced the incumbent's ``p_available``.

    Read from the incumbent's STRUCTURED summary fields.  The prose ``rule``
    string is never parsed: a display label must not be load-bearing.
    """

    if source_summary.get("scouting_override"):
        return BASIS_SCOUTING_OVERRIDE
    status = source_summary.get("status")
    chance = source_summary.get("chance_of_playing_this_round")
    if status in config.hard_unavailable_statuses:
        return BASIS_HARD_STATUS
    if isinstance(chance, (int, float)) and not isinstance(chance, bool):
        return BASIS_EVENT_CHANCE
    if status in config.official_availability_signal_defaults:
        return BASIS_DECLARED_STATUS_DEFAULT
    return BASIS_NO_RESTRICTION


# ---------------------------------------------------------------------------
# The coherent chain.
# ---------------------------------------------------------------------------


DERIVED_CHAIN_FIELDS: tuple[str, ...] = (
    "p_start",
    "p_cameo",
    "p_zero",
    "p_1_59",
    "p_60_plus",
    "p_80_plus",
    "expected_minutes",
)

#: How a challenger row's derived values were obtained.
DERIVED_BY_BASE_REPUBLICATION = "BASE_PROJECTION_REPUBLISHED"
DERIVED_BY_CHAIN_REDERIVATION = "CHAIN_REDERIVED_FROM_PUBLISHED_MARGINALS"


def coherent_chain(
    *,
    p_available: float,
    p_start_given_available: float,
    p_cameo_given_not_start: float,
    p_60_given_start: float,
    p_80_given_start: float,
    p_60_given_cameo: float,
    expected_minutes_if_start: float,
    expected_minutes_if_cameo: float,
) -> dict[str, float]:
    """Re-derive the coherent distribution, with the incumbent's own equations.

    Mirrors ``minutes_model.project_player_fixture``'s derivation exactly:

        P(start)  = P(available) * P(start | available)
        P(cameo)  = P(available) * (1 - P(start | available)) * P(cameo | not start)
        P(0)      = 1 - P(start) - P(cameo)
        P(60+)    = P(start) * P(60+ | start) + P(cameo) * P(60+ | cameo)
        P(80+)    = P(start) * P(80+ | start), never above P(60+)
        E[minutes] = P(start) * E[minutes | start] + P(cameo) * E[minutes | cameo]

    The cameo tail is folded into ``P(1-59)`` exactly as the incumbent declares
    (``p80_if_cameo == 0``), and the same range guards are applied, so an
    impossible distribution raises instead of being published.
    """

    p_start = _clamp(p_available * p_start_given_available)
    not_start_adj = (1.0 - p_start_given_available) if p_available > 0 else 1.0
    p_cameo = _clamp(p_available * not_start_adj * p_cameo_given_not_start)
    p_zero = _clamp(1.0 - p_start - p_cameo)
    p_60_plus = _clamp(p_start * p_60_given_start + p_cameo * p_60_given_cameo)
    p_80_plus = min(_clamp(p_start * p_80_given_start), p_60_plus)
    p_1_59 = _clamp(1.0 - p_zero - p_60_plus)
    expected_minutes = p_start * expected_minutes_if_start + p_cameo * expected_minutes_if_cameo

    for name, value in (
        ("p_available", p_available),
        ("p_start_given_available", p_start_given_available),
        ("p_cameo_given_not_start", p_cameo_given_not_start),
        ("p_60_given_start", p_60_given_start),
        ("p_80_given_start", p_80_given_start),
        ("p_60_given_cameo", p_60_given_cameo),
        ("p_start", p_start),
        ("p_cameo", p_cameo),
        ("p_zero", p_zero),
        ("p_1_59", p_1_59),
        ("p_60_plus", p_60_plus),
        ("p_80_plus", p_80_plus),
    ):
        if not 0.0 - 1e-9 <= value <= 1.0 + 1e-9:
            raise ValueError(f"challenger probability out of range: {name}={value}")
    if not 0.0 <= expected_minutes <= 90.0 + 1e-9:
        raise ValueError(f"challenger expected minutes out of range: {expected_minutes}")
    if abs((p_zero + p_1_59 + p_60_plus) - 1.0) > 1e-6:
        # Same repair the incumbent makes: the three body states must sum to one.
        p_zero = _clamp(1.0 - p_1_59 - p_60_plus)

    return {
        "p_start": _round(p_start),
        "p_cameo": _round(p_cameo),
        "p_zero": _round(p_zero),
        "p_1_59": _round(p_1_59),
        "p_60_plus": _round(p_60_plus),
        "p_80_plus": _round(p_80_plus),
        "expected_minutes": _round(expected_minutes),
    }


# ---------------------------------------------------------------------------
# Evidence.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChallengerEvidence:
    """Every causal quantity the refinements are allowed to read.

    Each block carries its own sample size, so a thin sample is visible as a
    thin sample rather than as a confident number.
    """

    availability_basis: str
    status_at_cutoff: Any
    chance_at_cutoff: Any
    incumbent_p_available: float
    row_attribution_bases: Mapping[str, int]
    event_specific_rows: int
    assumed_attribution_rows_excluded: int
    status_declared_default: float | None
    status_evidence_observations: int
    status_evidence_appearances: int
    status_estimate: float | None
    ramp_applies: bool
    ramp_attenuation: float
    ramp_recovery_observations: int
    ramp_recovery_mean_minutes: float | None
    ramp_effective_modifiers: Mapping[str, float]
    cameo_rate_pool_prior: float
    cameo_rate_observations: int
    cameo_rate_recent_observations: int
    cameo_rate_estimate: float | None
    cameo_tail_prior: float
    cameo_tail_observations: int
    cameo_tail_estimate: float | None
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        payload = {item.name: getattr(self, item.name) for item in fields(self)}
        payload["ramp_effective_modifiers"] = dict(self.ramp_effective_modifiers)
        payload["row_attribution_bases"] = dict(self.row_attribution_bases)
        payload["notes"] = list(self.notes)
        return payload


def build_player_evidence(
    evidence_rows: Sequence[Mapping[str, Any]],
    snapshot: Mapping[str, Any] | None,
    snapshot_history: Sequence[Mapping[str, Any]] | None,
    scout_notes: Sequence[Mapping[str, Any]],
    *,
    config: "incumbent.MinutesModelConfig",
    challenger_config: AvailabilityMinutesChallengerConfig,
    pools: "incumbent.LeaguePools",
    position: int,
    cutoff: str,
    incumbent_p_available: float,
    incumbent_source_summary: Mapping[str, Any],
) -> ChallengerEvidence:
    """Assemble the causal evidence for one player at one cutoff.

    Uses the incumbent's own row classification (``classify_evidence_rows``) and
    its own status-at-event attribution (``availability_status_at_event``, via
    that classification), so no second history predicate exists anywhere in
    PE-6.  The availability rule is read from the incumbent's published
    structured summary and cross-checked against the snapshot the incumbent read;
    a disagreement stops the build rather than being reconciled silently.
    """

    classification = incumbent.classify_evidence_rows(
        evidence_rows, snapshot_history=snapshot_history, config=config
    )
    # Only rows whose status is event-specific may be refined evidence.
    classified = [
        row for row in classification if row.get("availability_basis") == EVENT_SPECIFIC_ATTRIBUTION
    ]
    attribution_bases: dict[str, int] = {}
    for row in classification:
        basis_name = str(row.get("availability_basis"))
        attribution_bases[basis_name] = attribution_bases.get(basis_name, 0) + 1
    assumed_excluded = len(classification) - len(classified)
    status = (snapshot or {}).get("status")
    chance = (snapshot or {}).get("chance_of_playing_this_round")
    basis = _availability_basis(incumbent_source_summary, config)
    snapshot_basis = _availability_basis(
        {"status": status, "chance_of_playing_this_round": chance}, config
    )
    if basis != snapshot_basis and basis != BASIS_SCOUTING_OVERRIDE:
        raise ChallengerInconsistencyError(
            f"availability basis disagrees with the incumbent's own summary: "
            f"{basis!r} (published) vs {snapshot_basis!r} (snapshot at the cutoff)"
        )
    notes: list[str] = []

    # --- family 1: P(play | restricting status) from the player's own rows ----
    declared_default: float | None = None
    status_observations = 0
    status_appearances = 0
    status_estimate: float | None = None
    if basis != BASIS_DECLARED_STATUS_DEFAULT:
        notes.append(f"STATUS_ESTIMATE_NOT_APPLICABLE:{basis}")
    elif status not in SOFT_RESTRICTING_STATUSES:
        notes.append("STATUS_ESTIMATE_NOT_APPLICABLE:STATUS_IS_NOT_A_RESTRICTION")
    else:
        declared_default = float(config.official_availability_signal_defaults[str(status)])
        rows_for_status = [
            row for row in classified if row.get("availability_status_at_event") == status
        ]
        status_observations = len(rows_for_status)
        status_appearances = sum(
            1 for row in rows_for_status if float(row.get("minutes") or 0) > 0
        )
        if status_observations < challenger_config.status_evidence_min_rows:
            if status_observations == 0:
                notes.append(
                    "STATUS_ESTIMATE_NO_EVENT_SPECIFIC_ATTRIBUTION"
                    if assumed_excluded
                    else "STATUS_ESTIMATE_NO_ROWS_FOR_STATUS"
                )
            else:
                notes.append("STATUS_ESTIMATE_BELOW_MINIMUM_ROWS")
        else:
            status_estimate = _shrink(
                float(status_appearances),
                float(status_observations),
                declared_default,
                float(challenger_config.status_evidence_prior_strength),
            )

    # --- family 2: attenuate the return ramp by demonstrated recovery ---------
    returning_statuses = tuple(str(value) for value in config.returning_status_values)
    _adjustments, _modifier_ids, scout_flags = incumbent.fold_scout_modifiers(scout_notes, cutoff, config)
    ramp_applies = (snapshot is not None and status in returning_statuses) or (
        SCOUT_RETURN_RAMP_FLAG in scout_flags
    )
    ramp_rows = [
        row for row in classified if row.get("availability_status_at_event") in returning_statuses
    ]
    ramp_appearances = [
        float(row["minutes"]) for row in ramp_rows if float(row.get("minutes") or 0) > 0
    ]
    ramp_mean_minutes = (
        (sum(ramp_appearances) / len(ramp_appearances)) if ramp_appearances else None
    )
    ramp_attenuation = 0.0
    if ramp_applies and len(ramp_appearances) >= int(challenger_config.ramp_recovery_min_rows):
        if ramp_mean_minutes is not None:
            ramp_attenuation = _clamp(
                ramp_mean_minutes / float(challenger_config.ramp_recovery_reference_minutes)
            )
    effective_modifiers = {
        "returning_start_modifier": _attenuated(config.returning_start_modifier, ramp_attenuation),
        "returning_upper_minutes_modifier": _attenuated(
            config.returning_upper_minutes_modifier, ramp_attenuation
        ),
        "returning_minutes_if_start_modifier": _attenuated(
            config.returning_minutes_if_start_modifier, ramp_attenuation
        ),
    }

    # --- families 3 and 4: observed rotation / cameo evidence -----------------
    # These need only the row's own observed ``starts`` / ``minutes`` facts, not
    # its attributed status, so every classified row is usable here.
    observation_rows = [
        row for row in classification if row["evidence_class"] in START_OBSERVATION_CLASSES
    ]
    weights = _recency_weights(
        len(observation_rows), float(challenger_config.resolved(config)["cameo_recency_half_life_matches"])
    )
    not_start_rows = [
        weight for weight, row in zip(weights, observation_rows) if not row.get("starts")
    ]
    cameo_rows = [
        (weight, row)
        for weight, row in zip(weights, observation_rows)
        if not row.get("starts") and float(row.get("minutes") or 0) > 0
    ]
    not_start_weight = sum(not_start_rows)
    cameo_weight = sum(weight for weight, _row in cameo_rows)
    pool = pools.position.get(int(position)) or incumbent.structural_no_history_priors(config)
    pool_cameo_rate = float(pool.get("cameo_rate") or 0.5)
    cameo_rate_estimate = (
        _shrink(
            cameo_weight,
            not_start_weight,
            pool_cameo_rate,
            float(config.cameo_rate_prior_strength),
        )
        if not_start_weight > 0
        else None
    )
    cameo_tail_prior = max(float(pool.get("cameo_p60") or 0.0), float(config.cameo_p60_prior))
    cameo_tail_60_weight = sum(
        weight for weight, row in cameo_rows if float(row.get("minutes") or 0) >= 60
    )
    cameo_tail_estimate = _shrink(
        cameo_tail_60_weight,
        cameo_weight,
        cameo_tail_prior,
        float(
            challenger_config.cameo_tail_prior_strength
            if challenger_config.cameo_tail_prior_strength is not None
            else config.cameo_p60_prior_strength
        ),
    )

    return ChallengerEvidence(
        availability_basis=basis,
        status_at_cutoff=status,
        chance_at_cutoff=chance,
        incumbent_p_available=float(incumbent_p_available),
        row_attribution_bases=attribution_bases,
        event_specific_rows=len(classified),
        assumed_attribution_rows_excluded=assumed_excluded,
        status_declared_default=declared_default,
        status_evidence_observations=status_observations,
        status_evidence_appearances=status_appearances,
        status_estimate=(None if status_estimate is None else _round(status_estimate)),
        ramp_applies=bool(ramp_applies),
        ramp_attenuation=_round(ramp_attenuation),
        ramp_recovery_observations=len(ramp_appearances),
        ramp_recovery_mean_minutes=(
            None if ramp_mean_minutes is None else _round(ramp_mean_minutes)
        ),
        ramp_effective_modifiers={key: _round(value) for key, value in effective_modifiers.items()},
        cameo_rate_pool_prior=_round(pool_cameo_rate),
        cameo_rate_observations=len(not_start_rows),
        cameo_rate_recent_observations=len(cameo_rows),
        cameo_rate_estimate=(None if cameo_rate_estimate is None else _round(cameo_rate_estimate)),
        cameo_tail_prior=_round(cameo_tail_prior),
        cameo_tail_observations=len(cameo_rows),
        cameo_tail_estimate=_round(cameo_tail_estimate),
        notes=tuple(notes),
    )


def _attenuated(modifier: float, attenuation: float) -> float:
    """Move a declared discount toward 1.0 by ``attenuation`` in ``[0, 1]``."""

    return float(modifier) + (1.0 - float(modifier)) * _clamp(attenuation)


# ---------------------------------------------------------------------------
# Row refinement.
# ---------------------------------------------------------------------------


def refine_row(
    base_row: Mapping[str, Any],
    evidence: ChallengerEvidence,
    *,
    config: "incumbent.MinutesModelConfig",
    challenger_config: AvailabilityMinutesChallengerConfig,
    enabled_families: Iterable[str],
    position_id: int = 0,
    player_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply the enabled refinements to one incumbent row and re-derive it.

    ``base_row`` must be the incumbent's row (optionally produced under the
    challenger's attenuated return-ramp configuration) with
    ``include_conditionals=True``.  Every family block records whether the
    refinement actually MOVED the published value (``changed``), so a no-op is
    never reported as an improvement.
    """

    enabled = frozenset(enabled_families)
    unknown = sorted(enabled - set(REFINEMENT_FAMILIES))
    if unknown:
        raise ValueError(f"unknown refinement family(ies): {unknown}")
    missing = [
        name
        for name in (
            "p_available",
            "p_start_given_available",
            "p_cameo_given_not_start",
            "p_60_given_start",
            "p_80_given_start",
            "p_60_given_cameo",
            "expected_minutes_if_start",
            "expected_minutes_if_cameo",
        )
        if name not in base_row
    ]
    if missing:
        raise ChallengerInconsistencyError(
            "the base row lacks the conditional quantities the coherent chain needs "
            f"({', '.join(missing)}); project it with include_conditionals=True"
        )

    p_available = float(base_row["p_available"])
    cameo_rate = float(base_row["p_cameo_given_not_start"])
    p_60_given_cameo = float(base_row["p_60_given_cameo"])
    refinements: dict[str, Any] = {}

    if FAMILY_AVAILABILITY_STATUS_EVIDENCE not in enabled:
        refinements[FAMILY_AVAILABILITY_STATUS_EVIDENCE] = {"status": REFINEMENT_DISABLED}
    elif evidence.status_estimate is None:
        refinements[FAMILY_AVAILABILITY_STATUS_EVIDENCE] = {
            "status": REFINEMENT_NOT_APPLICABLE
            if evidence.status_declared_default is None
            else REFINEMENT_NO_EVIDENCE,
            "basis": evidence.availability_basis,
            "observations": evidence.status_evidence_observations,
            "required_rows": int(challenger_config.status_evidence_min_rows),
            "declared_default": evidence.status_declared_default,
            "incumbent_p_available": evidence.incumbent_p_available,
            "assumed_attribution_rows_excluded": evidence.assumed_attribution_rows_excluded,
            "changed": False,
        }
    else:
        p_available = float(evidence.status_estimate)
        refinements[FAMILY_AVAILABILITY_STATUS_EVIDENCE] = {
            "status": REFINEMENT_APPLIED,
            "basis": evidence.availability_basis,
            "observations": evidence.status_evidence_observations,
            "appearances": evidence.status_evidence_appearances,
            "declared_default": evidence.status_declared_default,
            "estimate": evidence.status_estimate,
            "incumbent_p_available": evidence.incumbent_p_available,
            "prior_strength": float(challenger_config.status_evidence_prior_strength),
            "assumed_attribution_rows_excluded": evidence.assumed_attribution_rows_excluded,
            "changed": abs(p_available - float(base_row["p_available"])) > 1e-9,
        }

    if FAMILY_RETURN_FROM_INJURY_RAMP not in enabled:
        refinements[FAMILY_RETURN_FROM_INJURY_RAMP] = {"status": REFINEMENT_DISABLED}
    elif not evidence.ramp_applies:
        refinements[FAMILY_RETURN_FROM_INJURY_RAMP] = {
            "status": REFINEMENT_NOT_APPLICABLE,
            "reason": "no return-from-injury ramp applied to this row",
            "changed": False,
        }
    elif evidence.ramp_attenuation <= 0.0:
        refinements[FAMILY_RETURN_FROM_INJURY_RAMP] = {
            "status": REFINEMENT_NO_EVIDENCE,
            "recovery_observations": evidence.ramp_recovery_observations,
            "required_appearances": int(challenger_config.ramp_recovery_min_rows),
            "assumed_attribution_rows_excluded": evidence.assumed_attribution_rows_excluded,
            "changed": False,
        }
    else:
        refinements[FAMILY_RETURN_FROM_INJURY_RAMP] = {
            "status": REFINEMENT_APPLIED,
            "attenuation": evidence.ramp_attenuation,
            "recovery_observations": evidence.ramp_recovery_observations,
            "recovery_mean_minutes": evidence.ramp_recovery_mean_minutes,
            "reference_minutes": float(challenger_config.ramp_recovery_reference_minutes),
            "effective_modifiers": dict(evidence.ramp_effective_modifiers),
            "applied_in_projection_config": True,
            "assumed_attribution_rows_excluded": evidence.assumed_attribution_rows_excluded,
            "changed": True,
        }

    if FAMILY_CAMEO_RATE_RECENCY not in enabled:
        refinements[FAMILY_CAMEO_RATE_RECENCY] = {"status": REFINEMENT_DISABLED}
    elif evidence.cameo_rate_estimate is None:
        refinements[FAMILY_CAMEO_RATE_RECENCY] = {
            "status": REFINEMENT_NO_EVIDENCE,
            "observations": evidence.cameo_rate_observations,
            "incumbent_p_cameo_given_not_start": float(base_row["p_cameo_given_not_start"]),
            "changed": False,
        }
    else:
        cameo_rate = float(evidence.cameo_rate_estimate)
        refinements[FAMILY_CAMEO_RATE_RECENCY] = {
            "status": REFINEMENT_APPLIED,
            "observations": evidence.cameo_rate_observations,
            "cameo_observations": evidence.cameo_rate_recent_observations,
            "pool_prior": evidence.cameo_rate_pool_prior,
            "estimate": evidence.cameo_rate_estimate,
            "incumbent_p_cameo_given_not_start": float(base_row["p_cameo_given_not_start"]),
            "half_life_matches": float(
                challenger_config.resolved(config)["cameo_recency_half_life_matches"]
            ),
            "changed": abs(cameo_rate - float(base_row["p_cameo_given_not_start"])) > 1e-9,
        }

    if FAMILY_CAMEO_TAIL_EVIDENCE not in enabled:
        refinements[FAMILY_CAMEO_TAIL_EVIDENCE] = {"status": REFINEMENT_DISABLED}
    elif evidence.cameo_tail_observations == 0:
        # With no cameo appearance there is no observation to shrink, and the
        # estimate is exactly the incumbent's prior: report the fallback, not a
        # refinement that never happened.
        refinements[FAMILY_CAMEO_TAIL_EVIDENCE] = {
            "status": REFINEMENT_NO_EVIDENCE,
            "prior": evidence.cameo_tail_prior,
            "incumbent_p_60_given_cameo": float(base_row["p_60_given_cameo"]),
            "changed": False,
        }
    else:
        p_60_given_cameo = float(evidence.cameo_tail_estimate)
        refinements[FAMILY_CAMEO_TAIL_EVIDENCE] = {
            "status": REFINEMENT_APPLIED,
            "observations": evidence.cameo_tail_observations,
            "prior": evidence.cameo_tail_prior,
            "estimate": evidence.cameo_tail_estimate,
            "incumbent_p_60_given_cameo": float(base_row["p_60_given_cameo"]),
            "changed": abs(p_60_given_cameo - float(base_row["p_60_given_cameo"])) > 1e-9,
        }

    chain_inputs_changed = (
        abs(p_available - float(base_row["p_available"])) > 1e-9
        or abs(cameo_rate - float(base_row["p_cameo_given_not_start"])) > 1e-9
        or abs(p_60_given_cameo - float(base_row["p_60_given_cameo"])) > 1e-9
    )
    if chain_inputs_changed:
        chain = coherent_chain(
            p_available=p_available,
            p_start_given_available=float(base_row["p_start_given_available"]),
            p_cameo_given_not_start=cameo_rate,
            p_60_given_start=float(base_row["p_60_given_start"]),
            p_80_given_start=float(base_row["p_80_given_start"]),
            p_60_given_cameo=p_60_given_cameo,
            expected_minutes_if_start=float(base_row["expected_minutes_if_start"]),
            expected_minutes_if_cameo=float(base_row["expected_minutes_if_cameo"]),
        )
        derived_from = DERIVED_BY_CHAIN_REDERIVATION
    else:
        # No chain input moved: republish the base projection's OWN derived
        # values.  Re-deriving them from its published 6-decimal marginals would
        # shift them by the storage rounding (up to ~1e-5 minutes) and turn a
        # provable no-op into an approximation.
        chain = {name: float(base_row[name]) for name in DERIVED_CHAIN_FIELDS}
        derived_from = DERIVED_BY_BASE_REPUBLICATION

    row: dict[str, Any] = {
        "player_id": int(base_row["player_id"]),
        "fixture_id": int(base_row["fixture_id"]),
        "event": int(base_row["event"]),
        "model_version": AVAILABILITY_MINUTES_CHALLENGER_VERSION,
        "model_family": AVAILABILITY_MINUTES_CHALLENGER_FAMILY,
        "challenger_config_hash": challenger_config.config_hash(),
        "p_available": _round(p_available),
        "p_start_given_available": _round(float(base_row["p_start_given_available"])),
        "p_cameo_given_not_start": _round(cameo_rate),
        "p_60_given_start": _round(float(base_row["p_60_given_start"])),
        "p_80_given_start": _round(float(base_row["p_80_given_start"])),
        "p_60_given_cameo": _round(p_60_given_cameo),
        "expected_minutes_if_start": _round(float(base_row["expected_minutes_if_start"])),
        "expected_minutes_if_cameo": _round(float(base_row["expected_minutes_if_cameo"])),
        **chain,
        "derived_chain_source": derived_from,
        "enabled_families": sorted(enabled),
        "refinements": refinements,
        "evidence": evidence.as_dict(),
        "risk_flags": list(base_row.get("risk_flags") or []),
        "availability_basis": evidence.availability_basis,
        "status_at_cutoff": evidence.status_at_cutoff,
        "position_id": int(position_id),
        # Provenance: where this row's membership, club and position came from.
        # The resolved identity is what the projection used, so a later change to
        # the current row cannot have moved this row.
        "player_identity": dict(player_identity or {}),
        # Provenance: which incumbent artifact and which projection config this
        # row's marginals came from.  Never a claim that the incumbent's version
        # identifier is the challenger's.
        "base_row_model_version": str(base_row.get("model_version")),
        "base_row_availability_source_summary": dict(
            base_row.get("availability_source_summary") or {}
        ),
        "base_row": dict(base_row),
        "generated_at": utc_now(),
    }
    return row


# ---------------------------------------------------------------------------
# Arms.
# ---------------------------------------------------------------------------

ARM_ALL_REFINEMENTS = "challenger_all_refinements"


def arm_name_for_family(family: str) -> str:
    return f"challenger_only::{family}"


def default_arm_definitions() -> dict[str, frozenset[str]]:
    """The full challenger plus one single-family ablation per refinement."""

    definitions: dict[str, frozenset[str]] = {
        ARM_ALL_REFINEMENTS: frozenset(REFINEMENT_FAMILIES)
    }
    for family in REFINEMENT_FAMILIES:
        definitions[arm_name_for_family(family)] = frozenset({family})
    return definitions


@dataclass
class ChallengerArms:
    """One incumbent arm and the challenger arms, over the same key set."""

    planning_event: int
    cutoff: str
    incumbent_rows: dict[tuple[int, int], dict[str, Any]] = field(default_factory=dict)
    arms: dict[str, dict[tuple[int, int], dict[str, Any]]] = field(default_factory=dict)
    evidence: dict[tuple[int, int], ChallengerEvidence] = field(default_factory=dict)
    identity: dict[str, Any] = field(default_factory=dict)
    #: The cutoff-resolved candidate pool every arm was projected from, so the
    #: evaluation classifies exactly the population the arms cover.
    players: list[dict[str, Any]] = field(default_factory=list)
    #: Generation members the cutoff could not place, with reasons.
    unresolved: list[dict[str, Any]] = field(default_factory=list)
    #: The cut-off-stable priors this run was projected against, and how they
    #: were built (no persisted-row identity anywhere in the pooling).
    pool_priors: dict[str, Any] = field(default_factory=dict)

    def keys(self) -> list[tuple[int, int]]:
        return sorted(self.incumbent_rows)

    def arm_keys(self, arm: str) -> list[tuple[int, int]]:
        return sorted(self.arms[arm])

    def arm_names(self) -> tuple[str, ...]:
        return tuple(sorted(self.arms))

    def verify_same_population(self) -> None:
        """Every arm covers exactly the incumbent's key set, or PE-6 stops."""

        from . import walk_forward as wf

        incumbent_keys = self.keys()
        for arm in self.arm_names():
            wf.assert_same_population(incumbent_keys, self.arm_keys(arm))


def build_challenger_arms(
    conn: sqlite3.Connection,
    planning_event: int,
    cutoff: str,
    *,
    incumbent_config: "incumbent.MinutesModelConfig | None" = None,
    challenger_config: AvailabilityMinutesChallengerConfig | None = None,
    arm_definitions: Mapping[str, frozenset[str]] | None = None,
    resolution: Mapping[str, Any] | None = None,
    allow_live_identity_fallback: bool = False,
) -> ChallengerArms:
    """Project every player-fixture of the event for the incumbent and the arms.

    The incumbent projection function is called unchanged.  The deviations are
    all recorded per row: ``include_conditionals=True`` (the coherent layer needs
    the same conditionals), the CUTOFF-RESOLVED player identity (membership, club
    and position as the cutoff's accepted generation records them rather than as
    the persisted row now claims), the cutoff-stable ``LeaguePools`` the incumbent
    is handed through its own contract, and -- where the return-ramp refinement is
    in play -- a per-player ``MinutesModelConfig`` carrying the attenuated ramp
    factors.

    ``resolution`` lets a caller supply a pool it has already resolved
    (:func:`resolve_candidate_pool`), so the evaluation's classification and its
    arms cannot disagree about who is who.  It is required to be one that
    RESOLVED its identity: a cutoff with no usable official pool generation stops
    here rather than being projected under a guess.
    """

    config = incumbent_config or incumbent.MinutesModelConfig()
    challenger_config = challenger_config or AvailabilityMinutesChallengerConfig()
    definitions = dict(arm_definitions or default_arm_definitions())
    requested_families = frozenset().union(*definitions.values()) if definitions else frozenset()
    ramp_needed = FAMILY_RETURN_FROM_INJURY_RAMP in requested_families

    resolved = (
        dict(resolution)
        if resolution is not None
        else resolve_candidate_pool(
            conn, cutoff, allow_live_fallback=allow_live_identity_fallback
        )
    )
    if not resolved.get("identity_available"):
        raise ChallengerInconsistencyError(
            "the cutoff has no usable official pool generation, so no candidate identity can be "
            "resolved at it: "
            + "; ".join((resolved.get("summary") or {}).get("generation", {}).get("reasons") or [])
        )
    pool = [dict(row) for row in resolved["players"]]
    refused = [dict(row) for row in resolved["unresolved"]]

    identities = {
        int(row["player_id"]): (int(row["team_id"]), int(row["element_type"])) for row in pool
    }
    pools, pool_priors = cutoff_stable_league_pools(
        conn, int(planning_event), cutoff, identities
    )
    fixtures_by_team = analytics.event_fixture_map(conn, int(planning_event))
    identity = challenger_identity(challenger_config, config)
    identity["player_identity"] = dict(resolved["summary"])
    identity["pool_priors"] = pool_priors
    result = ChallengerArms(
        planning_event=int(planning_event),
        cutoff=cutoff,
        identity=identity,
        players=pool,
        unresolved=refused,
        pool_priors=pool_priors,
    )
    for arm in definitions:
        result.arms[arm] = {}

    for player in sorted(pool, key=lambda row: int(row["player_id"])):
        team_fixtures = fixtures_by_team.get(int(player["team_id"]))
        if not team_fixtures:
            continue
        player_id = int(player["player_id"])
        evidence_rows = analytics.completed_rows_as_of(conn, player_id, cutoff, int(planning_event))
        snapshot = analytics.snapshot_as_of(conn, player_id, cutoff)
        snapshot_history = analytics.snapshot_history_as_of(conn, player_id, cutoff)
        scout_notes = repo.scouting_current_rows_as_of(conn, cutoff, [player_id])
        position = int(player.get("element_type") or 0)
        for fixture in team_fixtures:
            base_row = incumbent.project_player_fixture(
                conn,
                pools,
                player,
                evidence_rows,
                fixture,
                snapshot,
                scout_notes,
                config,
                cutoff,
                include_conditionals=True,
                snapshot_history=snapshot_history,
            )
            key = (player_id, int(fixture["id"]))
            evidence = build_player_evidence(
                evidence_rows,
                snapshot,
                snapshot_history,
                scout_notes,
                config=config,
                challenger_config=challenger_config,
                pools=pools,
                position=position,
                cutoff=cutoff,
                incumbent_p_available=float(base_row["p_available"]),
                incumbent_source_summary=base_row.get("availability_source_summary") or {},
            )
            # The challenger must never disagree with the incumbent about whether
            # a return ramp is in play: the flag is the incumbent's own record.
            incumbent_ramp = "RETURN_RAMP" in set(base_row.get("risk_flags") or [])
            if evidence.ramp_applies != incumbent_ramp:
                raise ChallengerInconsistencyError(
                    f"player {player_id} fixture {key[1]}: challenger ramp state "
                    f"{evidence.ramp_applies} != incumbent RETURN_RAMP {incumbent_ramp}"
                )
            result.incumbent_rows[key] = base_row
            result.evidence[key] = evidence

            attenuated_row = None
            if ramp_needed and evidence.ramp_applies and evidence.ramp_attenuation > 0.0:
                attenuated_config = replace(
                    config,
                    returning_start_modifier=evidence.ramp_effective_modifiers["returning_start_modifier"],
                    returning_upper_minutes_modifier=evidence.ramp_effective_modifiers[
                        "returning_upper_minutes_modifier"
                    ],
                    returning_minutes_if_start_modifier=evidence.ramp_effective_modifiers[
                        "returning_minutes_if_start_modifier"
                    ],
                )
                attenuated_row = incumbent.project_player_fixture(
                    conn,
                    pools,
                    player,
                    evidence_rows,
                    fixture,
                    snapshot,
                    scout_notes,
                    attenuated_config,
                    cutoff,
                    include_conditionals=True,
                    snapshot_history=snapshot_history,
                )

            for arm, families in definitions.items():
                base = (
                    attenuated_row
                    if (attenuated_row is not None and FAMILY_RETURN_FROM_INJURY_RAMP in families)
                    else base_row
                )
                result.arms[arm][key] = refine_row(
                    base,
                    evidence,
                    config=config,
                    challenger_config=challenger_config,
                    enabled_families=families,
                    position_id=position,
                    player_identity=player.get("identity"),
                )

    result.verify_same_population()
    return result


def build_challenger_rows(
    conn: sqlite3.Connection,
    planning_event: int,
    cutoff: str,
    *,
    incumbent_config: "incumbent.MinutesModelConfig | None" = None,
    challenger_config: AvailabilityMinutesChallengerConfig | None = None,
    enabled_families: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """The challenger's rows for one event, sorted ``(player, fixture)``."""

    families = (
        frozenset(REFINEMENT_FAMILIES)
        if enabled_families is None
        else frozenset(enabled_families)
    )
    arms = build_challenger_arms(
        conn,
        planning_event,
        cutoff,
        incumbent_config=incumbent_config,
        challenger_config=challenger_config,
        arm_definitions={ARM_ALL_REFINEMENTS: families},
    )
    return [arms.arms[ARM_ALL_REFINEMENTS][key] for key in arms.keys()]
