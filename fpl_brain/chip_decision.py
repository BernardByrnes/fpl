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


def _availability_by_action(chip_availability: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    """Map canonical chip availability rows onto the action space.

    ``planning.chips_state`` reports availability per official chip definition
    (a chip with two windows yields two rows); an action is available when ANY
    of its definition rows is available for the event.
    """

    merged: dict[str, dict[str, Any]] = {}
    for row in chip_availability:
        name = normalise_chip_name(str(row.get("name") or ""))
        action = next(
            (a for a, official in CHIP_ACTION_TO_OFFICIAL_NAME.items() if normalise_chip_name(official) == name),
            None,
        )
        if action is None:
            continue
        entry = merged.setdefault(
            action,
            {"name": name, "available_for_event": False, "used": False, "expired": False, "windows": []},
        )
        entry["available_for_event"] = bool(entry["available_for_event"] or row.get("available_for_event"))
        entry["used"] = bool(entry["used"] or row.get("used"))
        entry["expired"] = bool(entry["expired"] and row.get("expired"))
        if row.get("window") is not None:
            entry["windows"].append(row.get("window"))
    for action, entry in merged.items():
        entry["windows"] = sorted(entry["windows"])
        entry["action"] = action
    return {action: merged[action] for action in merged}


def decide_chip_action(
    *,
    planning_event: int,
    chip_availability: Sequence[Mapping[str, Any]],
    evaluations: Mapping[str, ChipEvaluation] | None = None,
    reservation: ReservationValue | None = None,
    horizon_events: Sequence[int] | None = None,
    required_horizon_length: int = 4,
    certification_identity: str | None = None,
    data_snapshot_sha256: str | None = None,
    certification_valid: bool = False,
    manager_state: Mapping[str, Any] | None = None,
    chips_already_played_for_event: Sequence[str] = (),
    materiality: float = 0.0,
) -> ChipDecision:
    """Choose exactly ONE chip action, or refuse.

    Fail-closed order (each returns ``INSUFFICIENT_EVIDENCE``):

    1. certification missing/invalid;
    2. the decision horizon is not the exact required length;
    3. manager state incomplete;
    4. the input claims more than one chip played in this Gameweek (the canonical
       one-chip-per-Gameweek rule).

    A Gameweek whose single chip slot is already used returns ``NO_CHIP``: no
    further chip may be played in it.  A chip that is unavailable for the event
    can never be recommended.  With an uncalibrated reservation value a positive
    mean uplift yields ``CHIP_CANDIDATE_RECHECK_REQUIRED`` (or
    ``CHIP_REVIEW_REQUIRED`` when the interval still straddles zero) — never
    ``PLAY_CHIP``, which needs a reservation that declares itself CALIBRATED.
    """

    availability_rows = tuple(dict(row) for row in chip_availability)
    availability = _availability_by_action(availability_rows)
    reasons: list[str] = []

    def refuse(token: str, detail: str, *, status: str = STATUS_INSUFFICIENT_EVIDENCE) -> ChipDecision:
        return ChipDecision(
            recommended_action=CHIP_ACTION_NO_CHIP,
            status=status,
            planning_event=int(planning_event),
            chip_availability=availability_rows,
            calibration_status=CALIBRATION_UNCALIBRATED,
            evidence={
                "certification_identity": certification_identity,
                "data_snapshot_sha256": data_snapshot_sha256,
                "horizon_events": list(horizon_events or ()),
                "refusal_detail": detail,
            },
            reason_codes=tuple(sorted({token, *reasons})),
            uncertainty={},
            candidate_metrics={},
            evaluations_considered=(),
        )

    # 1. certification
    if not certification_valid or not certification_identity:
        return refuse(DIAG_CHIP_CERTIFICATION_REQUIRED,
                      "a valid certification artifact is required to decide a chip")
    if not data_snapshot_sha256:
        return refuse(DIAG_CHIP_CERTIFICATION_REQUIRED, "the certification carries no data snapshot identity")

    # 2. horizon: exact length, no approximation and no fifth event
    events = tuple(int(event) for event in (horizon_events or ()))
    if len(events) != int(required_horizon_length):
        return refuse(
            DIAG_CHIP_HORIZON_INCOMPLETE,
            f"the chip decision needs the exact {int(required_horizon_length)}-event horizon, got {list(events)}",
        )

    # 3. manager state
    squad_ids = tuple(int(pid) for pid in ((manager_state or {}).get("squad_ids") or ()))
    if manager_state is None or not squad_ids:
        return refuse(DIAG_CHIP_MANAGER_STATE_INCOMPLETE, "manager squad state is required to decide a chip")

    # 4. one chip per Gameweek
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
            planning_event=int(planning_event),
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

    supplied = dict(evaluations or {})
    considered = tuple(sorted(supplied))
    # Always report which chips have no evaluator: an unimplemented chip is
    # never silently treated as evaluated-and-bad.
    unimplemented = [action for action in PLAYABLE_CHIP_ACTIONS if action not in supplied]
    if unimplemented:
        reasons.append(f"{DIAG_CHIP_EVALUATOR_NOT_IMPLEMENTED}:{','.join(unimplemented)}")
    # A chip that is not available for this event can never be recommended.
    eligible: list[ChipEvaluation] = []
    for action in CHIP_ACTIONS:
        if action not in supplied:
            continue
        row = availability.get(action)
        if row is None or not row.get("available_for_event"):
            reasons.append(f"{DIAG_CHIP_UNAVAILABLE}:{action}")
            continue
        eligible.append(supplied[action])

    if not eligible:
        available_any = any(bool(row.get("available_for_event")) for row in availability.values())
        return ChipDecision(
            recommended_action=CHIP_ACTION_NO_CHIP,
            status=STATUS_NO_CHIP if not available_any else STATUS_CHIP_REVIEW_REQUIRED,
            planning_event=int(planning_event),
            chip_availability=availability_rows,
            calibration_status=CALIBRATION_UNCALIBRATED,
            evidence={
                "certification_identity": certification_identity,
                "data_snapshot_sha256": data_snapshot_sha256,
                "horizon_events": list(events),
                "available_actions": sorted(a for a, row in availability.items() if row.get("available_for_event")),
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

    estimate = (reservation or UncalibratedReservation()).estimate(
        action=chosen.action,
        planning_event=int(planning_event),
        expiry_event=availability.get(chosen.action, {}).get("window_stop_event"),
        state={"squad_ids": list(squad_ids)},
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
        planning_event=int(planning_event),
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
