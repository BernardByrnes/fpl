"""Chip decision core — one action space, one result contract, shared infrastructure.

FOOTBALL SCORING IS AN INPUT CONTRACT
-------------------------------------
A chip evaluator may consume only *stable, certified* inputs: per-world player
score and minute series, manager squad and transfer/bank state, chip
availability, the planning horizon, and the certification identity.  It must
never read or recreate the football model — no xPts coefficients, no
player-rate internals, no xG/xA or DEFCON formulas, no team-model formulas.
That is what lets an improved projection/world generator be dropped in without
touching chip decision logic, and it is enforced here structurally: this module
imports nothing from the predictive stack (see the input-contract test).

NOTHING HERE IS CALIBRATED
--------------------------
The final PLAY_CHIP threshold for any chip depends on a backtested, point-in-time
calibration that does not exist yet.  ``UncalibratedReservation`` therefore
returns ``value=None`` and the arbiter refuses to emit ``PLAY_CHIP`` while the
reservation value is unknown: a positive mean uplift yields
``CHIP_CANDIDATE_RECHECK_REQUIRED``, never an endorsement.  ``PLAY_CHIP`` is
reachable only by supplying a reservation implementation that declares itself
``CALIBRATED``.

ONE ACTIVE CHIP PER GAMEWEEK
----------------------------
``decide_chip_action`` returns exactly one action from
``NO_CHIP | TC | BB | FH | WC`` and refuses an input state that claims more than
one chip already played in the planning event (canonical season rule).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from .season_rules import CANONICAL_CHIP_NAMES, normalise_chip_name

CHIP_DECISION_VERSION = "chip_decision_v1.0.0"

# --- action space ----------------------------------------------------------

CHIP_ACTION_NO_CHIP = "NO_CHIP"
CHIP_ACTION_TC = "TC"
CHIP_ACTION_BB = "BB"
CHIP_ACTION_FH = "FH"
CHIP_ACTION_WC = "WC"
#: The complete top-level decision space.  Order is the deterministic tie-break.
CHIP_ACTIONS = (CHIP_ACTION_NO_CHIP, CHIP_ACTION_TC, CHIP_ACTION_BB, CHIP_ACTION_FH, CHIP_ACTION_WC)
PLAYABLE_CHIP_ACTIONS = (CHIP_ACTION_TC, CHIP_ACTION_BB, CHIP_ACTION_FH, CHIP_ACTION_WC)

#: Canonical mapping onto the official chip names held by ``season_rules``.
CHIP_ACTION_TO_OFFICIAL_NAME = {
    CHIP_ACTION_TC: "3xc",
    CHIP_ACTION_BB: "bboost",
    CHIP_ACTION_FH: "freehit",
    CHIP_ACTION_WC: "wildcard",
}

# --- statuses --------------------------------------------------------------

STATUS_PLAY_CHIP = "PLAY_CHIP"
STATUS_CHIP_CANDIDATE_RECHECK_REQUIRED = "CHIP_CANDIDATE_RECHECK_REQUIRED"
STATUS_CHIP_REVIEW_REQUIRED = "CHIP_REVIEW_REQUIRED"
STATUS_NO_CHIP = "NO_CHIP"
STATUS_INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
CHIP_STATUSES = (
    STATUS_PLAY_CHIP,
    STATUS_CHIP_CANDIDATE_RECHECK_REQUIRED,
    STATUS_CHIP_REVIEW_REQUIRED,
    STATUS_NO_CHIP,
    STATUS_INSUFFICIENT_EVIDENCE,
)

# --- calibration -----------------------------------------------------------

CALIBRATION_UNCALIBRATED = "UNCALIBRATED"
CALIBRATION_UNCALIBRATED_PROVISIONAL = "UNCALIBRATED_PROVISIONAL"
CALIBRATION_CALIBRATED = "CALIBRATED"
CALIBRATION_STATUSES = (CALIBRATION_UNCALIBRATED, CALIBRATION_UNCALIBRATED_PROVISIONAL, CALIBRATION_CALIBRATED)

# --- fail-closed tokens ----------------------------------------------------

DIAG_CHIP_CERTIFICATION_REQUIRED = "CHIP_CERTIFICATION_REQUIRED"
DIAG_CHIP_HORIZON_INCOMPLETE = "CHIP_DECISION_HORIZON_INCOMPLETE"
DIAG_CHIP_MULTIPLE_ACTIONS = "CHIP_MULTIPLE_ACTIONS_IN_GAMEWEEK"
DIAG_CHIP_GAMEWEEK_ALREADY_USED = "CHIP_GAMEWEEK_ALREADY_USED"
DIAG_CHIP_MANAGER_STATE_INCOMPLETE = "CHIP_MANAGER_STATE_INCOMPLETE"
DIAG_CHIP_UNAVAILABLE = "CHIP_UNAVAILABLE_FOR_EVENT"
DIAG_CHIP_WORLD_CONTRACT_INCOMPLETE = "CHIP_WORLD_INPUT_CONTRACT_INCOMPLETE"
DIAG_CHIP_WORLD_PARTITION_INVALID = "CHIP_WORLD_PARTITION_INVALID"
DIAG_CHIP_EVALUATOR_NOT_IMPLEMENTED = "CHIP_EVALUATOR_NOT_IMPLEMENTED"
DIAG_CHIP_RESERVATION_UNCALIBRATED = "CHIP_RESERVATION_VALUE_UNCALIBRATED"
DIAG_CHIP_UPLIFT_NON_POSITIVE = "CHIP_UPLIFT_NON_POSITIVE"
DIAG_CHIP_UPLIFT_NOT_MATERIAL = "CHIP_UPLIFT_NOT_MATERIAL"
DIAG_CHIP_UPLIFT_POSITIVE = "CHIP_UPLIFT_POSITIVE"
DIAG_CHIP_HORIZON_NOT_CANONICAL = "CHIP_HORIZON_NOT_CANONICAL"
DIAG_CHIP_PLANNING_EVENT_MISMATCH = "CHIP_PLANNING_EVENT_MISMATCH"
DIAG_CHIP_CERTIFICATION_HORIZON_MISMATCH = "CHIP_CERTIFICATION_HORIZON_MISMATCH"
DIAG_CHIP_EVALUATION_CONTEXT_MISMATCH = "CHIP_EVALUATION_CONTEXT_MISMATCH"
#: Raised when the disagreement is specifically the DATA snapshot identity, so a
#: cross-snapshot arbitration is distinguishable from a horizon mismatch.
DIAG_CHIP_DATA_SNAPSHOT_MISMATCH = "CHIP_DATA_SNAPSHOT_MISMATCH"
DIAG_CHIP_EVALUATION_ACTION_MISMATCH = "CHIP_EVALUATION_ACTION_MISMATCH"
DIAG_CHIP_EVALUATOR_UNCALIBRATED = "CHIP_EVALUATOR_VALUE_MODEL_UNCALIBRATED"
DIAG_CHIP_NO_ACTIVE_WINDOW = "CHIP_NO_ACTIVE_WINDOW_FOR_ACTION"
DIAG_CHIP_WINDOW_SELECTION_AMBIGUOUS = "CHIP_RESERVATION_WINDOW_SELECTION_AMBIGUOUS"


class ChipDecisionError(ValueError):
    """A chip decision could not be produced from the supplied evidence."""

    def __init__(self, detail: str, *, reasons: Sequence[str] = ()) -> None:
        super().__init__(detail)
        self.reasons = list(reasons)


class ChipInputError(ChipDecisionError):
    """The input contract is incomplete, so the evaluator fails closed."""


# --- the input contract ----------------------------------------------------


@dataclass(frozen=True)
class ChipWorldInputs:
    """Certified per-world score/minute series — the ONLY football input.

    Every series is length ``worlds`` and every declared player has one, exactly
    as the certified manager-world matrix guarantees.  A truncated or partially
    captured matrix fails closed here rather than being silently padded.
    """

    worlds: int
    player_ids: tuple[int, ...]
    minutes: Mapping[int, Sequence[float]]
    core: Mapping[int, Sequence[float]]
    planning_event: int
    horizon_events: tuple[int, ...]
    certification_identity: str
    data_snapshot_sha256: str
    world_seed: int
    world_identity: str
    code_snapshot_sha256: str | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> int:
        """Return the world count, raising ``ChipInputError`` on any gap."""

        if int(self.worlds) <= 0:
            raise ChipInputError(
                f"{DIAG_CHIP_WORLD_CONTRACT_INCOMPLETE}: worlds={self.worlds}",
                reasons=[DIAG_CHIP_WORLD_CONTRACT_INCOMPLETE],
            )
        if not self.player_ids:
            raise ChipInputError(
                f"{DIAG_CHIP_WORLD_CONTRACT_INCOMPLETE}: no players declared",
                reasons=[DIAG_CHIP_WORLD_CONTRACT_INCOMPLETE],
            )
        for block_name, block in (("minutes", self.minutes), ("core", self.core)):
            missing = sorted(pid for pid in self.player_ids if int(pid) not in block)
            if missing:
                raise ChipInputError(
                    f"{DIAG_CHIP_WORLD_CONTRACT_INCOMPLETE}: {block_name} has no series for {missing[:8]}",
                    reasons=[DIAG_CHIP_WORLD_CONTRACT_INCOMPLETE],
                )
            wrong = sorted(pid for pid in self.player_ids if len(block[int(pid)]) != int(self.worlds))
            if wrong:
                raise ChipInputError(
                    f"{DIAG_CHIP_WORLD_CONTRACT_INCOMPLETE}: {block_name} series length != {self.worlds} "
                    f"for {wrong[:8]}",
                    reasons=[DIAG_CHIP_WORLD_CONTRACT_INCOMPLETE],
                )
        return int(self.worlds)

    @classmethod
    def from_world_matrix(
        cls,
        matrix: Mapping[str, Any],
        *,
        planning_event: int,
        horizon_events: Sequence[int],
        certification_identity: str | None,
        data_snapshot_sha256: str | None,
        world_seed: int,
        world_identity: str,
        code_snapshot_sha256: str | None = None,
    ) -> "ChipWorldInputs":
        """Build the contract from a certified manager-world matrix verbatim."""

        if not certification_identity:
            raise ChipInputError(
                f"{DIAG_CHIP_CERTIFICATION_REQUIRED}: the world inputs carry no certification identity",
                reasons=[DIAG_CHIP_CERTIFICATION_REQUIRED],
            )
        if not data_snapshot_sha256:
            raise ChipInputError(
                f"{DIAG_CHIP_CERTIFICATION_REQUIRED}: the world inputs carry no data snapshot identity",
                reasons=[DIAG_CHIP_CERTIFICATION_REQUIRED],
            )
        player_ids = tuple(int(pid) for pid in matrix.get("player_ids") or ())
        minutes = {int(pid): tuple(float(v) for v in series) for pid, series in (matrix.get("minutes") or {}).items()}
        core = {int(pid): tuple(float(v) for v in series) for pid, series in (matrix.get("core") or {}).items()}
        return cls(
            worlds=int(matrix.get("worlds") or 0),
            player_ids=player_ids,
            minutes=minutes,
            core=core,
            planning_event=int(planning_event),
            horizon_events=tuple(int(event) for event in horizon_events),
            certification_identity=str(certification_identity),
            data_snapshot_sha256=str(data_snapshot_sha256),
            world_seed=int(world_seed),
            world_identity=str(world_identity),
            code_snapshot_sha256=None if code_snapshot_sha256 is None else str(code_snapshot_sha256),
        )

    def appeared(self, player_id: int) -> tuple[bool, ...]:
        """Per-world appearance (minutes > 0) — the certified appearance rule."""

        series = self.minutes.get(int(player_id))
        if series is None:
            raise ChipInputError(
                f"{DIAG_CHIP_WORLD_CONTRACT_INCOMPLETE}: no series for player {player_id}",
                reasons=[DIAG_CHIP_WORLD_CONTRACT_INCOMPLETE],
            )
        return tuple(float(value) > 0.0 for value in series)

    def evidence(self) -> dict[str, Any]:
        """Provenance an evaluator must carry into its result verbatim."""

        return {
            "certification_identity": self.certification_identity,
            "data_snapshot_sha256": self.data_snapshot_sha256,
            "code_snapshot_sha256": self.code_snapshot_sha256,
            "world_identity": self.world_identity,
            "world_seed": int(self.world_seed),
            "worlds": int(self.worlds),
            "planning_event": int(self.planning_event),
            "horizon_events": list(self.horizon_events),
        }


@runtime_checkable
class ChipWorldProvider(Protocol):
    """A source of certified world inputs.

    Any conforming provider can feed the evaluators unchanged, which is the
    interface half of the input contract: a replacement projection/world
    generator implements this protocol and no chip logic changes.
    """

    def provide(self, *, planning_event: int) -> ChipWorldInputs:
        ...


# --- selection / valuation worlds ------------------------------------------


@dataclass(frozen=True)
class WorldPartition:
    """A deterministic, non-overlapping split of ONE certified world set.

    Selection worlds CHOOSE the candidate; valuation worlds REPORT its value.
    Reporting a value on the same sample that chose it is the winner's-curse
    bias this partition exists to prevent.  The partition is a pure function of
    ``(seed, salt, world index)``, so world *k* is the same physical world in
    every run and prefix/common-random-number guarantees are preserved — no
    extra simulation is generated for either side.
    """

    selection: tuple[int, ...]
    valuation: tuple[int, ...]
    basis: str
    selection_worlds: int
    valuation_worlds: int

    def validate(self) -> "WorldPartition":
        if not self.selection or not self.valuation:
            raise ChipInputError(
                f"{DIAG_CHIP_WORLD_PARTITION_INVALID}: selection={len(self.selection)} "
                f"valuation={len(self.valuation)}",
                reasons=[DIAG_CHIP_WORLD_PARTITION_INVALID],
            )
        if set(self.selection) & set(self.valuation):
            raise ChipInputError(
                f"{DIAG_CHIP_WORLD_PARTITION_INVALID}: selection and valuation worlds overlap",
                reasons=[DIAG_CHIP_WORLD_PARTITION_INVALID],
            )
        if len(self.selection) + len(self.valuation) != self.selection_worlds + self.valuation_worlds:
            raise ChipInputError(
                f"{DIAG_CHIP_WORLD_PARTITION_INVALID}: partition does not cover the world set",
                reasons=[DIAG_CHIP_WORLD_PARTITION_INVALID],
            )
        return self


def _partition_bucket(seed: int, salt: str, index: int) -> int:
    payload = f"{int(seed)}|{salt}|{int(index)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % 2


def partition_worlds(worlds: int, *, seed: int, salt: str = "chip-selection-valuation") -> WorldPartition:
    """Deterministic SELECTION/VALUATION split of one certified world set."""

    count = int(worlds)
    if count < 2:
        raise ChipInputError(
            f"{DIAG_CHIP_WORLD_PARTITION_INVALID}: {count} world(s) cannot be split",
            reasons=[DIAG_CHIP_WORLD_PARTITION_INVALID],
        )
    selection = tuple(index for index in range(count) if _partition_bucket(seed, salt, index) == 0)
    valuation = tuple(index for index in range(count) if _partition_bucket(seed, salt, index) == 1)
    return WorldPartition(
        selection=selection,
        valuation=valuation,
        basis=f"sha256:{salt}",
        selection_worlds=len(selection),
        valuation_worlds=len(valuation),
    ).validate()


# --- reservation value (play now vs save) ----------------------------------

#: Reason codes a reservation implementation may carry.
DIAG_RESERVATION_TERMINAL_ZERO = "CHIP_RESERVATION_TERMINAL_VALUE_ZERO"
DIAG_RESERVATION_NO_FUTURE_MODEL = "CHIP_RESERVATION_FUTURE_MODEL_ABSENT"


@dataclass(frozen=True)
class ReservationEstimate:
    """What saving the chip is worth, in the same units as the H1 uplift.

    ``value=None`` means "unknown", and the arbiter treats unknown as
    not-endorsable.  ``terminal_value`` is the value the chip carries at expiry
    (0 — an unused chip that expires is worth nothing).
    """

    value: float | None
    calibration_status: str
    terminal_value: float
    weeks_to_expiry: int | None
    reason_codes: tuple[str, ...] = ()
    conditional_on: tuple[str, ...] = ()


@runtime_checkable
class ReservationValue(Protocol):
    """The generic PLAY-NOW vs SAVE interface.

    The comparison is deliberately between two NON-ANTICIPATIVE policies:
    play now, or save and decide again later with the information available
    then.  It is never ``current value vs E[max future realised value]``, which
    would be clairvoyant.  A conforming implementation may condition its future
    opportunity estimate on weeks remaining to expiry, chip availability,
    squad quality, player availability/role uncertainty, known fixture quality
    and manager state — ``conditional_on`` reports which it actually used.
    """

    def estimate(
        self,
        *,
        action: str,
        planning_event: int,
        expiry_event: int | None,
        state: Mapping[str, Any],
    ) -> ReservationEstimate:
        ...


@dataclass(frozen=True)
class UncalibratedReservation:
    """The honest default: no future-opportunity model exists, so it is UNKNOWN.

    It still implements the full interface (including the zero terminal value at
    expiry) so that a calibrated model can replace it without any chip-decision
    change.  It never manufactures certainty: ``value`` stays ``None``.
    """

    calibration_status: str = CALIBRATION_UNCALIBRATED

    def estimate(
        self,
        *,
        action: str,
        planning_event: int,
        expiry_event: int | None,
        state: Mapping[str, Any],
    ) -> ReservationEstimate:
        weeks = None if expiry_event is None else max(0, int(expiry_event) - int(planning_event))
        return ReservationEstimate(
            value=None,
            calibration_status=self.calibration_status,
            terminal_value=0.0,
            weeks_to_expiry=weeks,
            reason_codes=(DIAG_CHIP_RESERVATION_UNCALIBRATED, DIAG_RESERVATION_TERMINAL_ZERO,
                          DIAG_RESERVATION_NO_FUTURE_MODEL),
            conditional_on=(),
        )


# --- results ---------------------------------------------------------------


@dataclass(frozen=True)
class ChipEvaluation:
    """One chip's evaluation: candidate metrics, uncertainty, calibration, reasons."""

    action: str
    evaluator_version: str
    candidate_metrics: Mapping[str, Any] = field(default_factory=dict)
    uncertainty: Mapping[str, Any] = field(default_factory=dict)
    reason_codes: tuple[str, ...] = ()
    calibration_status: str = CALIBRATION_UNCALIBRATED
    evidence: Mapping[str, Any] = field(default_factory=dict)
    #: Whether THIS evaluation's own value model is sound enough to execute on.
    #: Independent of the reservation: an executable play needs BOTH.  Defaults
    #: to True so every pre-existing evaluator (notably Triple Captain) keeps its
    #: exact established semantics -- only an evaluator that knows its own value
    #: model is uncalibrated sets this False.
    execution_permitted: bool = True

    @property
    def mean_uplift(self) -> float | None:
        value = self.candidate_metrics.get("mean_paired_uplift")
        return None if value is None else float(value)


@dataclass(frozen=True)
class ChipDecision:
    """The ONE canonical chip decision result."""

    recommended_action: str
    status: str
    planning_event: int
    chip_availability: tuple[Mapping[str, Any], ...]
    calibration_status: str
    evidence: Mapping[str, Any]
    reason_codes: tuple[str, ...]
    uncertainty: Mapping[str, Any]
    candidate_metrics: Mapping[str, Any]
    evaluations_considered: tuple[str, ...]
    one_chip_rule_enforced: bool = True
    selection_worlds: int = 0
    valuation_worlds: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "fpl_brain.chip_decision.v1",
            "version": CHIP_DECISION_VERSION,
            "recommended_action": self.recommended_action,
            "status": self.status,
            "planning_event": int(self.planning_event),
            "chip_availability": [dict(row) for row in self.chip_availability],
            "calibration_status": self.calibration_status,
            "evidence": dict(self.evidence),
            "reason_codes": list(self.reason_codes),
            "uncertainty": dict(self.uncertainty),
            "candidate_metrics": dict(self.candidate_metrics),
            "evaluations_considered": list(self.evaluations_considered),
            "one_chip_rule_enforced": bool(self.one_chip_rule_enforced),
            "selection_worlds": int(self.selection_worlds),
            "valuation_worlds": int(self.valuation_worlds),
            "no_execution": True,
        }


# --- the arbiter -----------------------------------------------------------


#: Chip decisions are bound to the canonical FOUR-Gameweek decision window.
CHIP_HORIZON_LENGTH = 4


def canonical_chip_horizon(planning_event: int, *, last_event: int | None = None) -> tuple[int, ...]:
    """The canonical certified decision horizon for a planning event.

    Delegates to the ONE definition of the rolling four-Gameweek window used by
    the decision engine rather than restating it, so the chip layer cannot drift
    from the certified horizon.  Imported lazily because the decision engine is a
    consumer of this action space, not a dependency of the data model.
    """

    from . import four_gw_decision as fg

    if last_event is None:
        return tuple(fg.decision_events(int(planning_event), length=CHIP_HORIZON_LENGTH))
    return tuple(
        fg.decision_events(int(planning_event), length=CHIP_HORIZON_LENGTH, last_event=int(last_event))
    )


@dataclass(frozen=True)
class ChipHorizonBinding:
    """The exact certified four-event identity a chip decision is authorised for.

    Four arbitrary (or duplicated) integers are NOT a certified horizon: the
    events must be the canonical rolling window for the planning event, the world
    inputs must declare the same window and planning event, and the certification
    identity must be the one authorised for that window.
    """

    planning_event: int
    horizon_events: tuple[int, ...]
    certification_identity: str
    data_snapshot_sha256: str | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> "ChipHorizonBinding":
        events = tuple(int(event) for event in self.horizon_events)
        expected = canonical_chip_horizon(int(self.planning_event))
        if len(events) != CHIP_HORIZON_LENGTH or len(set(events)) != CHIP_HORIZON_LENGTH:
            raise ChipInputError(
                f"{DIAG_CHIP_HORIZON_NOT_CANONICAL}: a chip decision needs {CHIP_HORIZON_LENGTH} DISTINCT "
                f"events, got {list(events)}",
                reasons=[DIAG_CHIP_HORIZON_NOT_CANONICAL],
            )
        if events != expected:
            raise ChipInputError(
                f"{DIAG_CHIP_HORIZON_NOT_CANONICAL}: {list(events)} is not the canonical certified horizon "
                f"{list(expected)} for planning event {int(self.planning_event)}",
                reasons=[DIAG_CHIP_HORIZON_NOT_CANONICAL],
            )
        if not self.certification_identity:
            raise ChipInputError(
                f"{DIAG_CHIP_CERTIFICATION_REQUIRED}: the horizon binding carries no certification identity",
                reasons=[DIAG_CHIP_CERTIFICATION_REQUIRED],
            )
        return self

    @classmethod
    def from_certification(
        cls, artifact: Mapping[str, Any], *, planning_event: int | None = None
    ) -> "ChipHorizonBinding":
        """Build the binding from a certification artifact's own fields."""

        events = tuple(int(event) for event in (artifact.get("events") or ()))
        event = int(planning_event) if planning_event is not None else (events[0] if events else 0)
        identity = artifact.get("four_gw_certification_identity") or artifact.get("certification_identity")
        return cls(
            planning_event=event,
            horizon_events=events,
            certification_identity=str(identity or ""),
            data_snapshot_sha256=artifact.get("data_snapshot_sha256"),
        )

    def matches_worlds(self, worlds: "ChipWorldInputs") -> list[str]:
        """Every way the supplied world inputs disagree with this binding."""

        problems: list[str] = []
        if tuple(int(event) for event in worlds.horizon_events) != tuple(int(e) for e in self.horizon_events):
            problems.append(
                f"worlds horizon {list(worlds.horizon_events)} != certified {list(self.horizon_events)}"
            )
        if int(worlds.planning_event) != int(self.planning_event):
            problems.append(
                f"worlds planning_event {int(worlds.planning_event)} != certified {int(self.planning_event)}"
            )
        if str(worlds.certification_identity) != str(self.certification_identity):
            problems.append("worlds certification identity is not the authorised one")
        # The DATA snapshot is part of the certified context, not decoration: a
        # world set built from snapshot A may not be evaluated against a binding
        # authorised for snapshot D, even when the event window and the
        # certification identity agree.  Compared whenever both sides declare one
        # (a binding that carries no snapshot is not a claim about any snapshot).
        if self.data_snapshot_sha256 and worlds.data_snapshot_sha256:
            if str(worlds.data_snapshot_sha256) != str(self.data_snapshot_sha256):
                problems.append("worlds data snapshot is not the certified one")
        return problems


def _eligible_from_rows(
    rows: Sequence[Mapping[str, Any]], *, planning_event: int
) -> tuple[bool, list[dict[str, Any]]]:
    """Derive eligibility from the canonical chip state, per definition row.

    A chip type carries one definition row per half-season window, so the action
    is playable when ANY row is genuinely eligible.  Eligibility is computed from
    the canonical fields -- in window for THIS event, not already used, not
    expired -- rather than trusting an ``available_for_event`` flag on its own,
    and it is never derived by collapsing rows with one guessed AND/OR.
    """

    details: list[dict[str, Any]] = []
    eligible = False
    for row in rows:
        start = row.get("window_start_event")
        stop = row.get("window_stop_event")
        if start is not None and stop is not None:
            in_window = int(start) <= int(planning_event) <= int(stop)
        else:
            in_window = bool(row.get("available_for_event"))
        used = bool(row.get("used"))
        expired = bool(row.get("expired"))
        row_eligible = bool(row.get("available_for_event")) and in_window and not used and not expired
        eligible = eligible or row_eligible
        details.append(
            {
                "window": row.get("window"),
                "window_start_event": start,
                "window_stop_event": stop,
                "available_for_event": bool(row.get("available_for_event")),
                "in_window_for_event": bool(in_window),
                "used": used,
                "expired": expired,
                "eligible": row_eligible,
            }
        )
    return eligible, details


def _active_definition(
    definitions: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any] | None, str | None]:
    """The ONE definition row that establishes CURRENT eligibility for the action.

    A chip type carries one definition row per seasonal window, and
    ``planning.chips_state`` orders those rows by ``start_event``.  The first row
    is therefore the EARLIEST window: taking it blindly hands a calibrated
    reservation the ``stop_event`` of a window that has already closed.  For a
    chip with windows GW2-GW19 and GW20-GW38 the action is correctly eligible at
    GW20, but the reservation would be told it expires at 19 and see zero weeks
    remaining instead of the real live horizon.

    The active row is selected with the SAME per-row predicate the arbiter
    already used to admit the action (``_eligible_from_rows``), so the
    reservation can never disagree with the eligibility that admitted it.

    Returns ``(row, None)`` when exactly one row is active.  When no row is
    active, or several are, the state is not decidable: ``(None, token)``.  Two
    simultaneously-active windows have no precedence in ``season_rules``, so
    they fail closed rather than being resolved by row order.
    """

    active = [row for row in definitions if bool(row.get("eligible"))]
    if len(active) == 1:
        return active[0], None
    if not active:
        return None, DIAG_CHIP_NO_ACTIVE_WINDOW
    return None, DIAG_CHIP_WINDOW_SELECTION_AMBIGUOUS


def _availability_by_action(
    chip_availability: Sequence[Mapping[str, Any]], *, planning_event: int
) -> dict[str, dict[str, Any]]:
    """Map canonical chip availability rows onto the action space.

    ``planning.chips_state`` reports availability per official chip DEFINITION (a
    chip with two windows yields two rows).  Each action's eligibility is the OR
    of its rows' own eligibility; ``used``/``expired`` are reported as separate
    any/all flags so a stale flag can never mask the canonical state.
    """

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in chip_availability:
        name = normalise_chip_name(str(row.get("name") or ""))
        action = next(
            (a for a, official in CHIP_ACTION_TO_OFFICIAL_NAME.items() if normalise_chip_name(official) == name),
            None,
        )
        if action is None:
            continue
        grouped.setdefault(action, []).append(row)

    merged: dict[str, dict[str, Any]] = {}
    for action, rows in grouped.items():
        eligible, details = _eligible_from_rows(rows, planning_event=int(planning_event))
        merged[action] = {
            "action": action,
            "name": normalise_chip_name(str(rows[0].get("name") or "")),
            "eligible": eligible,
            "definitions": details,
            "windows": sorted(str(row.get("window")) for row in rows if row.get("window") is not None),
            "used_any": any(bool(row.get("used")) for row in rows),
            "expired_any": any(bool(row.get("expired")) for row in rows),
            "expired_all": all(bool(row.get("expired")) for row in rows),
        }
    return {action: merged[action] for action in merged}


def decide_chip_action(
    *,
    horizon_binding: ChipHorizonBinding,
    chip_availability: Sequence[Mapping[str, Any]],
    evaluations: Mapping[str, ChipEvaluation] | None = None,
    reservation: ReservationValue | None = None,
    certification_valid: bool = False,
    manager_state: Mapping[str, Any] | None = None,
    chips_already_played_for_event: Sequence[str] = (),
    materiality: float = 0.0,
) -> ChipDecision:
    """Choose exactly ONE chip action, or refuse.

    The decision is BOUND to one certified four-event identity: the horizon
    binding fixes the planning event, the canonical event window and the
    authorised certification identity, and nothing may be decided outside it.

    Fail-closed order (each returns ``INSUFFICIENT_EVIDENCE``):

    1. certification missing/invalid;
    2. the horizon is not the exact canonical certified window (duplicated or
       arbitrary four-event lists are refused);
    3. manager state incomplete;
    4. an evaluation's mapping key disagrees with its own action;
    5. an evaluation was produced for a different horizon/certification;
    6. the input claims more than one chip played in this Gameweek.

    A Gameweek whose single chip slot is already used returns ``NO_CHIP``: no
    further chip may be played in it.  An action is eligible only when the
    canonical chip state says so (in window for this event, not used, not
    expired); an ineligible action can never be recommended, and no action may
    borrow another action's availability.  With an uncalibrated reservation value
    a positive mean uplift yields ``CHIP_CANDIDATE_RECHECK_REQUIRED`` (or
    ``CHIP_REVIEW_REQUIRED`` when the interval still straddles zero) -- never
    ``PLAY_CHIP``, which needs a reservation that declares itself CALIBRATED.
    """

    binding = horizon_binding
    planning_event = int(binding.planning_event)
    events = tuple(int(event) for event in binding.horizon_events)
    certification_identity = binding.certification_identity
    data_snapshot_sha256 = binding.data_snapshot_sha256
    availability_rows = tuple(dict(row) for row in chip_availability)
    availability = _availability_by_action(availability_rows, planning_event=planning_event)
    reasons: list[str] = []

    def refuse(
        token: str,
        detail: str,
        *,
        status: str = STATUS_INSUFFICIENT_EVIDENCE,
        extra: Sequence[str] = (),
    ) -> ChipDecision:
        return ChipDecision(
            recommended_action=CHIP_ACTION_NO_CHIP,
            status=status,
            planning_event=planning_event,
            chip_availability=availability_rows,
            calibration_status=CALIBRATION_UNCALIBRATED,
            evidence={
                "certification_identity": certification_identity,
                "data_snapshot_sha256": data_snapshot_sha256,
                "horizon_events": list(events),
                "refusal_detail": detail,
            },
            reason_codes=tuple(sorted({token, *reasons, *extra})),
            uncertainty={},
            candidate_metrics={},
            evaluations_considered=(),
        )

    # 1. certification
    if not certification_valid or not certification_identity:
        return refuse(
            DIAG_CHIP_CERTIFICATION_REQUIRED,
            "a valid certification artifact is required to decide a chip",
        )
    if not data_snapshot_sha256:
        return refuse(DIAG_CHIP_CERTIFICATION_REQUIRED, "the certification carries no data snapshot identity")

    # 2. horizon: the exact canonical certified window, not merely four integers
    if len(events) != CHIP_HORIZON_LENGTH or len(set(events)) != CHIP_HORIZON_LENGTH:
        return refuse(
            DIAG_CHIP_HORIZON_NOT_CANONICAL,
            f"a chip decision needs {CHIP_HORIZON_LENGTH} distinct events, got {list(events)}",
        )
    if events != canonical_chip_horizon(planning_event):
        return refuse(
            DIAG_CHIP_HORIZON_NOT_CANONICAL,
            f"{list(events)} is not the canonical certified horizon "
            f"{list(canonical_chip_horizon(planning_event))} for planning event {planning_event}",
        )

    # 3. manager state
    squad_ids = tuple(int(pid) for pid in ((manager_state or {}).get("squad_ids") or ()))
    if manager_state is None or not squad_ids:
        return refuse(DIAG_CHIP_MANAGER_STATE_INCOMPLETE, "manager squad state is required to decide a chip")

    # 4/5. every evaluation must match its key, its action AND this certified context
    supplied = dict(evaluations or {})
    considered = tuple(sorted(supplied))
    mismatched_keys = sorted(key for key, evaluation in supplied.items() if str(key) != str(evaluation.action))
    if mismatched_keys:
        return refuse(
            DIAG_CHIP_EVALUATION_ACTION_MISMATCH,
            f"evaluation mapping key disagrees with the evaluation's own action for {mismatched_keys}",
        )
    foreign = sorted(
        action
        for action, evaluation in supplied.items()
        if str(evaluation.evidence.get("certification_identity") or "") != str(certification_identity)
        or tuple(int(event) for event in (evaluation.evidence.get("horizon_events") or ())) != events
        or int(evaluation.evidence.get("planning_event") or -1) != planning_event
        # An evaluation that was produced from a DIFFERENT data snapshot may not
        # be arbitrated under this binding, even when the horizon and the
        # certification identity agree.  Compared whenever the evaluation
        # declares a snapshot; an evaluation that declares none makes no claim to
        # contradict (every production evaluator emits its worlds' own value).
        or (
            bool(evaluation.evidence.get("data_snapshot_sha256"))
            and str(evaluation.evidence.get("data_snapshot_sha256")) != str(data_snapshot_sha256)
        )
    )
    if foreign:
        snapshot_foreign = sorted(
            action
            for action in foreign
            if bool(supplied[action].evidence.get("data_snapshot_sha256"))
            and str(supplied[action].evidence.get("data_snapshot_sha256")) != str(data_snapshot_sha256)
        )
        return refuse(
            DIAG_CHIP_EVALUATION_CONTEXT_MISMATCH,
            f"evaluation(s) {foreign} were produced for a different horizon, certification identity "
            f"or data snapshot",
            extra=(DIAG_CHIP_DATA_SNAPSHOT_MISMATCH,) if snapshot_foreign else (),
        )

    # 6. one chip per Gameweek
    played = tuple(str(chip) for chip in chips_already_played_for_event)
    if len(set(played)) > 1:
        return refuse(
            DIAG_CHIP_MULTIPLE_ACTIONS,
            f"{len(set(played))} chips are recorded as played in event {planning_event}: {sorted(set(played))}",
        )
    if played:
        # The Gameweek's single chip slot is spent: no further chip is playable.
        return ChipDecision(
            recommended_action=CHIP_ACTION_NO_CHIP,
            status=STATUS_NO_CHIP,
            planning_event=planning_event,
            chip_availability=availability_rows,
            calibration_status=CALIBRATION_UNCALIBRATED,
            evidence={
                "certification_identity": certification_identity,
                "data_snapshot_sha256": data_snapshot_sha256,
                "horizon_events": list(events),
                "chip_already_played": sorted(set(played)),
            },
            reason_codes=(DIAG_CHIP_GAMEWEEK_ALREADY_USED,),
            uncertainty={},
            candidate_metrics={},
            evaluations_considered=(),
        )

    # Always report which chips have no evaluator: an unimplemented chip is never
    # silently treated as evaluated-and-bad.
    unimplemented = [action for action in PLAYABLE_CHIP_ACTIONS if action not in supplied]
    if unimplemented:
        reasons.append(f"{DIAG_CHIP_EVALUATOR_NOT_IMPLEMENTED}:{','.join(unimplemented)}")

    # Eligibility comes from the canonical chip state for the SAME action that may
    # be returned: no action can borrow another action's availability.
    eligible: list[ChipEvaluation] = []
    for action, evaluation in sorted(supplied.items()):
        row = availability.get(action)
        if evaluation.action not in CHIP_ACTIONS or row is None or not row.get("eligible"):
            reasons.append(f"{DIAG_CHIP_UNAVAILABLE}:{action}")
            continue
        eligible.append(evaluation)

    if not eligible:
        available_any = any(bool(row.get("eligible")) for row in availability.values())
        return ChipDecision(
            recommended_action=CHIP_ACTION_NO_CHIP,
            status=STATUS_NO_CHIP if not available_any else STATUS_CHIP_REVIEW_REQUIRED,
            planning_event=planning_event,
            chip_availability=availability_rows,
            calibration_status=CALIBRATION_UNCALIBRATED,
            evidence={
                "certification_identity": certification_identity,
                "data_snapshot_sha256": data_snapshot_sha256,
                "horizon_events": list(events),
                "available_actions": sorted(a for a, row in availability.items() if row.get("eligible")),
            },
            reason_codes=tuple(sorted(set(reasons))),
            uncertainty={},
            candidate_metrics={},
            evaluations_considered=considered,
        )

    # Deterministic selection: net uplift first, then the canonical action order.
    def rank(evaluation: ChipEvaluation) -> tuple[float, int]:
        uplift = evaluation.mean_uplift
        value = float("-inf") if uplift is None else float(uplift)
        return (-value, CHIP_ACTIONS.index(evaluation.action))

    chosen = sorted(eligible, key=rank)[0]

    chosen_definitions = availability.get(chosen.action, {}).get("definitions") or []
    active_definition, window_problem = _active_definition(chosen_definitions)
    if window_problem is not None:
        # The action was admitted by some row, yet no single row establishes a
        # live window: the horizon the reservation would be valued against is
        # unknown, so refuse rather than guess one.
        n_active = sum(1 for row in chosen_definitions if row.get("eligible"))
        return refuse(
            window_problem,
            f"{chosen.action} is eligible but its active window is not uniquely "
            f"identifiable ({n_active} active definition rows of {len(chosen_definitions)})",
        )
    # The reservation provider receives the squad plus, when the evaluator
    # supplies one, its AUTHORITATIVE post-SAVE state (post-route squad, bank, FT,
    # chip availability).  This is how a chip whose decision turns on a projected
    # future state hands that state onward WITHOUT the evaluator ever calling the
    # reservation itself -- the arbiter remains the single seam.  An evaluator
    # that supplies no such state (notably Triple Captain) is unaffected: its
    # payload is exactly what it always was.
    reservation_state: dict[str, Any] = {"squad_ids": list(squad_ids)}
    supplied_state = (chosen.evidence.get("save_policy") or {}).get("post_save_state_for_reservation")
    if isinstance(supplied_state, Mapping):
        reservation_state.update(supplied_state)
    estimate = (reservation or UncalibratedReservation()).estimate(
        action=chosen.action,
        planning_event=planning_event,
        expiry_event=active_definition.get("window_stop_event"),
        state=reservation_state,
    )
    uplift = chosen.mean_uplift
    net = None if (uplift is None or estimate.value is None) else float(uplift) - float(estimate.value)
    lower = chosen.uncertainty.get("paired_interval_low")
    material = net is not None and float(net) > float(materiality)

    if uplift is None:
        status = STATUS_INSUFFICIENT_EVIDENCE
        verdicts = [DIAG_CHIP_UPLIFT_NOT_MATERIAL]
    elif float(uplift) <= float(materiality):
        # The play arm is not even better in H1, and saving keeps the chip.
        status = STATUS_NO_CHIP
        verdicts = [DIAG_CHIP_UPLIFT_NON_POSITIVE]
    elif estimate.calibration_status != CALIBRATION_CALIBRATED or net is None:
        # Positive evidence, but the future-opportunity term is unknown: the
        # system must not endorse.  A materially positive interval asks for a
        # recheck; one that still straddles zero asks for review.
        status = (
            STATUS_CHIP_REVIEW_REQUIRED
            if (lower is None or float(lower) <= 0.0)
            else STATUS_CHIP_CANDIDATE_RECHECK_REQUIRED
        )
        verdicts = [DIAG_CHIP_UPLIFT_POSITIVE, DIAG_CHIP_RESERVATION_UNCALIBRATED]
    elif not chosen.execution_permitted:
        # The evaluation's OWN value model is not fit to execute on, so a
        # calibrated reservation is NOT sufficient: the system still must not
        # endorse.  A positive interval asks for a recheck; a straddling one for
        # review.  Keying on this dedicated field (default True) rather than on
        # ``calibration_status`` is what leaves every pre-existing evaluator --
        # notably Triple Captain -- with its exact established semantics.
        status = (
            STATUS_CHIP_REVIEW_REQUIRED
            if (lower is None or float(lower) <= 0.0)
            else STATUS_CHIP_CANDIDATE_RECHECK_REQUIRED
        )
        verdicts = [DIAG_CHIP_UPLIFT_POSITIVE, DIAG_CHIP_EVALUATOR_UNCALIBRATED]
    elif material:
        status = STATUS_PLAY_CHIP
        verdicts = [DIAG_CHIP_UPLIFT_POSITIVE]
    else:
        status = STATUS_NO_CHIP
        verdicts = [DIAG_CHIP_UPLIFT_NOT_MATERIAL]

    return ChipDecision(
        recommended_action=(
            CHIP_ACTION_NO_CHIP if status in {STATUS_NO_CHIP, STATUS_INSUFFICIENT_EVIDENCE} else chosen.action
        ),
        status=status,
        planning_event=planning_event,
        chip_availability=availability_rows,
        calibration_status=estimate.calibration_status,
        evidence={
            "certification_identity": certification_identity,
            "data_snapshot_sha256": data_snapshot_sha256,
            "horizon_events": list(events),
            "evaluator_version": chosen.evaluator_version,
            **dict(chosen.evidence),
        },
        reason_codes=tuple(sorted({*verdicts, *chosen.reason_codes, *reasons, *estimate.reason_codes})),
        uncertainty=dict(chosen.uncertainty),
        candidate_metrics={
            **dict(chosen.candidate_metrics),
            "reservation_value": estimate.value,
            "reservation_terminal_value": estimate.terminal_value,
            "reservation_weeks_to_expiry": estimate.weeks_to_expiry,
            "reservation_conditional_on": list(estimate.conditional_on),
            "net_of_reservation": net,
            "materiality": float(materiality),
            "eligible_actions": sorted(evaluation.action for evaluation in eligible),
        },
        evaluations_considered=considered,
        selection_worlds=int(chosen.candidate_metrics.get("selection_worlds") or 0),
        valuation_worlds=int(chosen.candidate_metrics.get("valuation_worlds") or 0),
    )


def canonical_chip_names() -> tuple[str, ...]:
    """The canonical official chip names, straight from ``season_rules``."""

    return tuple(CANONICAL_CHIP_NAMES)
