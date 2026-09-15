"""Bench Boost evaluator — all fifteen score, inside the exact four-GW policy.

WHAT THE CHIP DOES
------------------
Bench Boost makes the FOUR bench players' scores count in addition to the XI.
It does NOT promote the bench into the XI, does NOT give anyone a second
armband copy, and does NOT change who is captain.

    NORMAL world = the accepted manager-lineup resolution
                   (starting XI + legal autosubs, armband from the original XI)
    BB world     = every one of the fifteen scores, armband rule UNCHANGED

Both arms are resolved on the certified per-world minute/score series.  There is
no ``bench_xpts`` shortcut and no "bench uplift coefficient" anywhere.

THE ONE IDENTITY
----------------
Per world, with the armband term identical across arms (it is a function of the
ORIGINAL XI's captain/vice and their minutes, so it cancels exactly):

    BB_uplift = BB_total - normal_total
              = bench_raw - normal_autosub_points

``bench_raw`` is the bench's own scoring (bench GK + three outfield bench, each
appearance-gated) and ``normal_autosub_points`` is what the NORMAL arm already
recovered from that same bench through legal autosubs.  The identity is what
makes double-counting impossible by construction: a bench player who autosubs
in is credited ONCE, because the chip's contribution is his own score minus the
copy the normal arm already had.

It also settles the three cases the contract calls out:

  * a starter who does not play contributes nothing to either arm (the certified
    engine gives a non-appearing player a zero core series), so an absent
    starter neither adds to BB nor disappears from NORMAL;
  * a bench player who COULD not legally autosub in (formation-constrained) is
    not lost: he never entered ``normal_autosub_points``, so his full score
    survives in the uplift;
  * the bench GK is worth exactly his score when the normal arm had to play the
    starting GK, and exactly zero when the normal arm already used him.

``bench_raw - normal_autosub_points == BB_total - normal_total`` is pinned
per-world by test, so a future change to either arm cannot silently break it.

APPEARANCE RULE
---------------
A player scores only if he appears (``minutes > 0``) — the same appearance rule
``manager_lineup.resolve_world`` uses.  The certified engine additionally emits
an explicit zero core for a non-appearing player, so the gated sum and the raw
sum agree on every real matrix; the gate is written out anyway, because the FPL
rule is about appearing, not about the matrix happening to agree.

PLAY vs SAVE
------------
This evaluator reports the RAW H1 paired uplift.  It never calls the reservation:
saving the chip is valued by the arbiter's reservation seam, exactly once.  It
supplies no ``post_save_state_for_reservation``, because playing Bench Boost
changes no squad, bank, free-transfer or chip state — so the arbiter's default
payload is already the correct post-SAVE state (Triple Captain's shape).

REVIEW-ONLY
-----------
``execution_permitted=False``.  No Bench Boost future-opportunity value model
exists, so a positive uplift may be reported but never executed; the arbiter
turns it into CHIP_CANDIDATE_RECHECK_REQUIRED / CHIP_REVIEW_REQUIRED.  This
mirrors the Wildcard's review-only status and is asserted by test.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from . import manager_lineup
from .chip_decision import (
    CALIBRATION_UNCALIBRATED,
    CHIP_ACTION_BB,
    DIAG_CHIP_HORIZON_NOT_CANONICAL,
    DIAG_CHIP_PLANNING_EVENT_MISMATCH,
    DIAG_CHIP_WORLD_CONTRACT_INCOMPLETE,
    ChipEvaluation,
    ChipHorizonBinding,
    ChipInputError,
    ChipWorldInputs,
)

BENCH_BOOST_EVALUATOR_VERSION = "chip_bb_v1.0.0"

#: Interval half-width constant, matching the engine's near-tie convention.
PAIRED_INTERVAL_K = 1.96

#: The units every number in this evaluator is expressed in.
VALUE_BASIS = "CORE_POINTS"

#: Which players Bench Boost makes score.  The whole squad — there is no
#: formation constraint and no bench-order priority under the chip.
BENCH_BOOST_SCORED_SLOTS = "ALL_FIFTEEN_SQUAD_SLOTS"

DIAG_BB_EVALUATED = "CHIP_BB_EVALUATED"
DIAG_BB_BENCH_ALREADY_RECOVERED = "CHIP_BB_BENCH_ALREADY_SCORED_VIA_AUTOSUB"
DIAG_BB_NO_BENCH_VALUE = "CHIP_BB_BENCH_CONTRIBUTES_NOTHING"
DIAG_BB_CAPTAIN_APPEARANCE_UNCERTAIN = "CHIP_BB_CAPTAIN_APPEARANCE_UNCERTAIN"
DIAG_BB_VICE_FALLBACK_MATERIAL = "CHIP_BB_VICE_FALLBACK_MATERIAL"
DIAG_BB_NO_ARMBAND_POSSIBLE = "CHIP_BB_NO_ARMBAND_POSSIBLE"
DIAG_BB_INPUT_UNCERTAINTY = "CHIP_BB_INPUT_UNCERTAINTY_PROPAGATED"
DIAG_BB_REVIEW_ONLY = "CHIP_BB_REVIEW_ONLY_UNCALIBRATED"


@dataclass(frozen=True)
class BenchBoostRequest:
    """Everything the evaluator may consume — nothing from the football model.

    ``policy`` is the manager's AUTHORITATIVE fifteen: the accepted lineup
    resolution (XI, bench GK, ordered outfield bench, captain, vice) for the
    planning event.  Bench Boost is evaluated on that squad — it is not a squad
    optimizer and it searches no transfers.
    """

    worlds: ChipWorldInputs
    horizon_binding: ChipHorizonBinding
    policy: manager_lineup.ManagerPolicy
    positions: Mapping[int, str]
    chip_available: bool = True
    calibration_status: str = CALIBRATION_UNCALIBRATED
    #: Upstream uncertainty flags to propagate verbatim (role, minutes, staleness).
    input_uncertainty_flags: Sequence[str] = field(default_factory=tuple)

    @property
    def planning_event(self) -> int:
        return int(self.horizon_binding.planning_event)

    @property
    def horizon_events(self) -> tuple[int, ...]:
        return tuple(int(event) for event in self.horizon_binding.horizon_events)


def squad_ids_of(policy: manager_lineup.ManagerPolicy) -> tuple[int, ...]:
    """The fifteen the chip scores: XI + bench GK + outfield bench, ascending."""

    return tuple(
        sorted(
            {int(pid) for pid in policy.starter_ids}
            | {int(policy.bench_gk_id)}
            | {int(pid) for pid in policy.bench_outfield_order}
        )
    )


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def _std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    """Same convention as the certified lineup engine's interval helper."""

    if not sorted_values:
        return float("nan")
    index = min(len(sorted_values) - 1, max(0, int(round(q * (len(sorted_values) - 1)))))
    return float(sorted_values[index])


def normal_world_value(
    policy: manager_lineup.ManagerPolicy,
    positions: Mapping[int, str],
    minutes: Mapping[int, float],
    core: Mapping[int, float],
    *,
    require_player_ids: Sequence[int] | set[int] | None = None,
) -> tuple[float, manager_lineup.WorldOutcome, float, str]:
    """The NORMAL arm for one world, resolved by the ACCEPTED engine.

    Delegates to ``manager_lineup.resolve_world`` (legal autosubs, formation
    constraints, bench priority) and ``manager_lineup.captain_multiplier``
    (captain -> vice -> nobody).  This arm is never re-implemented here: the
    certified engine is the oracle, so "when Bench Boost is not active nothing
    changed" is true by construction rather than by review.

    Returns ``(total, outcome, armband_extra, armband_source)``.
    """

    outcome = manager_lineup.resolve_world(
        policy, positions, minutes, core, require_player_ids=require_player_ids
    )
    base = sum(float(core[int(pid)]) for pid in outcome.counted_ids)
    extra, source = manager_lineup.captain_multiplier(
        policy, minutes, core, require_player_ids=require_player_ids
    )
    return base + float(extra), outcome, float(extra), str(source)


def bench_boost_world_value(
    policy: manager_lineup.ManagerPolicy,
    positions: Mapping[int, str],
    minutes: Mapping[int, float],
    core: Mapping[int, float],
    *,
    require_player_ids: Sequence[int] | set[int] | None = None,
) -> tuple[float, float, str]:
    """The BENCH BOOST arm for one world: all fifteen score, armband unchanged.

    No autosub resolution and no formation constraint apply — every one of the
    fifteen contributes his own score.  The armband is still the manager's
    captain (else vice, else nobody), exactly as in the normal arm, and it is
    taken from ``manager_lineup.captain_multiplier`` so BB cannot drift into a
    different captaincy rule.

    Returns ``(total, armband_extra, armband_source)``.
    """

    if require_player_ids is not None:
        manager_lineup.validate_scores_cover_players(
            minutes, core, require_player_ids, context="bench_boost_world_value"
        )
    total = 0.0
    for pid in squad_ids_of(policy):
        # FPL awards nothing to a player who does not appear; the certified
        # engine's appearance rule is minutes > 0.
        if float(minutes.get(int(pid), 0.0)) > 0.0:
            total += float(core.get(int(pid), 0.0))
    extra, source = manager_lineup.captain_multiplier(
        policy, minutes, core, require_player_ids=require_player_ids
    )
    return total + float(extra), float(extra), str(source)


def bench_raw_value(
    policy: manager_lineup.ManagerPolicy,
    minutes: Mapping[int, float],
    core: Mapping[int, float],
) -> float:
    """The bench's OWN scoring for one world (bench GK + three outfield bench).

    Appearance-gated, and independent of autosubs: this is the quantity the chip
    adds, before the points the NORMAL arm already recovered from the same bench
    are removed.
    """

    total = 0.0
    for pid in (int(policy.bench_gk_id), *(int(p) for p in policy.bench_outfield_order)):
        if float(minutes.get(pid, 0.0)) > 0.0:
            total += float(core.get(pid, 0.0))
    return total


def evaluate_bench_boost(request: BenchBoostRequest) -> ChipEvaluation:
    """PLAY_BB_NOW vs the SAVE policy -> the arbiter's ``ChipEvaluation`` shape.

    Raises ``ChipInputError`` when the request is not bound to the certified
    context, when the manager's fifteen is not a legal FPL lineup, or when the
    world inputs do not cover it.  A missing series is never read as zero.
    """

    worlds = request.worlds
    total_worlds = worlds.validate()
    binding = request.horizon_binding

    problems = binding.matches_worlds(worlds)
    if problems:
        raise ChipInputError(
            f"{DIAG_CHIP_PLANNING_EVENT_MISMATCH}: the request is not bound to the certified context: "
            + "; ".join(problems),
            reasons=[DIAG_CHIP_PLANNING_EVENT_MISMATCH],
        )
    events = request.horizon_events
    if events != binding.validate().horizon_events:
        raise ChipInputError(
            f"{DIAG_CHIP_HORIZON_NOT_CANONICAL}: horizon {list(events)} is not the bound certified horizon",
            reasons=[DIAG_CHIP_HORIZON_NOT_CANONICAL],
        )

    policy = request.policy
    positions = request.positions
    legality = manager_lineup.policy_legality_errors(policy, positions)
    if legality:
        raise ChipInputError(
            f"{DIAG_CHIP_WORLD_CONTRACT_INCOMPLETE}: the manager's fifteen is not a legal FPL lineup: "
            f"{legality}",
            reasons=[DIAG_CHIP_WORLD_CONTRACT_INCOMPLETE],
        )

    squad = squad_ids_of(policy)
    declared = {int(pid) for pid in worlds.player_ids}
    missing = sorted(pid for pid in squad if pid not in declared)
    if missing:
        raise ChipInputError(
            f"{DIAG_CHIP_WORLD_CONTRACT_INCOMPLETE}: the world inputs do not cover squad player(s) "
            f"{missing[:8]}",
            reasons=[DIAG_CHIP_WORLD_CONTRACT_INCOMPLETE],
        )
    # Every player either arm can score must have a real series.  A player the
    # matrix never captured fails closed here instead of being scored as zero.
    required = set(squad)
    for pid in required:
        for block_name, block in (("minutes", worlds.minutes), ("core", worlds.core)):
            series = block.get(int(pid))
            if series is None or len(series) != int(total_worlds):
                raise ChipInputError(
                    f"{DIAG_CHIP_WORLD_CONTRACT_INCOMPLETE}: {block_name} is not complete for squad "
                    f"player {pid}",
                    reasons=[DIAG_CHIP_WORLD_CONTRACT_INCOMPLETE],
                )

    # --- per-world resolution of both arms onto the SAME world indices -------
    bb_series: list[float] = []
    normal_series: list[float] = []
    paired: list[float] = []
    bench_raw_series: list[float] = []
    autosub_series: list[float] = []
    armband_series: list[float] = []
    armband_source_counts = {"CAPTAIN": 0, "VICE": 0, "NONE": 0}
    autosub_counts: list[float] = []
    gk_used_flags: list[float] = []

    for world in range(int(total_worlds)):
        w_minutes = {int(pid): float(worlds.minutes[int(pid)][world]) for pid in squad}
        w_core = {int(pid): float(worlds.core[int(pid)][world]) for pid in squad}
        normal_total, outcome, extra_normal, source_normal = normal_world_value(
            policy, positions, w_minutes, w_core, require_player_ids=squad
        )
        bb_total, extra_bb, source_bb = bench_boost_world_value(
            policy, positions, w_minutes, w_core, require_player_ids=squad
        )
        # The armband term is a function of the ORIGINAL XI's captain/vice and
        # their minutes only, so both arms must report the identical extra and
        # the identical source.  If they ever disagree, the chip has started
        # changing captaincy and the uplift is not measurable as a bench effect.
        if extra_normal != extra_bb or source_normal != source_bb:
            raise ChipInputError(
                f"{DIAG_CHIP_WORLD_CONTRACT_INCOMPLETE}: the armband is not identical across arms in world "
                f"{world} ({source_normal}/{extra_normal} vs {source_bb}/{extra_bb})",
                reasons=[DIAG_CHIP_WORLD_CONTRACT_INCOMPLETE],
            )
        raw = bench_raw_value(policy, w_minutes, w_core)
        autosub = float(outcome.autosub_points)
        uplift = float(bb_total) - float(normal_total)
        # The audit identity.  ``raw - autosub`` is the same number reached from
        # the bench side alone; a divergence means an arm was resolved wrongly.
        if abs((raw - autosub) - uplift) > 1e-9:
            raise ChipInputError(
                f"{DIAG_CHIP_WORLD_CONTRACT_INCOMPLETE}: world {world} breaks the Bench Boost identity "
                f"(bench_raw - autosub = {raw - autosub!r} != BB - normal = {uplift!r})",
                reasons=[DIAG_CHIP_WORLD_CONTRACT_INCOMPLETE],
            )
        bb_series.append(float(bb_total))
        normal_series.append(float(normal_total))
        paired.append(uplift)
        bench_raw_series.append(raw)
        autosub_series.append(autosub)
        armband_series.append(float(extra_bb))
        armband_source_counts[source_bb] = armband_source_counts.get(source_bb, 0) + 1
        autosub_counts.append(float(outcome.autosub_count))
        gk_used_flags.append(1.0 if outcome.gk_used else 0.0)

    mean_uplift = _mean(paired)
    paired_se = _std(paired) / math.sqrt(len(paired)) if paired else float("nan")
    ordered = sorted(paired)
    mean_raw = _mean(bench_raw_series)
    mean_autosub = _mean(autosub_series)
    captain = int(policy.captain_id)
    vice = int(policy.vice_captain_id)
    p_captain_appears = _mean(
        [1.0 if float(worlds.minutes[captain][world]) > 0.0 else 0.0 for world in range(int(total_worlds))]
    )

    flags: list[str] = [DIAG_BB_EVALUATED, DIAG_BB_REVIEW_ONLY]
    if mean_autosub > 0.0:
        # Part of the bench was already scoring through legal autosubs; the chip
        # is worth the REMAINDER, and that must be visible rather than assumed.
        flags.append(DIAG_BB_BENCH_ALREADY_RECOVERED)
    if mean_raw <= 0.0:
        flags.append(DIAG_BB_NO_BENCH_VALUE)
    if 0.0 < p_captain_appears < 1.0:
        flags.append(DIAG_BB_CAPTAIN_APPEARANCE_UNCERTAIN)
    if armband_source_counts.get("VICE", 0) / max(1, int(total_worlds)) > 0.05:
        flags.append(DIAG_BB_VICE_FALLBACK_MATERIAL)
    if armband_source_counts.get("NONE", 0) > 0:
        flags.append(DIAG_BB_NO_ARMBAND_POSSIBLE)
    propagated = tuple(str(flag) for flag in request.input_uncertainty_flags)
    if propagated:
        flags.append(DIAG_BB_INPUT_UNCERTAINTY)

    return ChipEvaluation(
        action=CHIP_ACTION_BB,
        evaluator_version=BENCH_BOOST_EVALUATOR_VERSION,
        candidate_metrics={
            "mean_paired_uplift": round(mean_uplift, 6),
            "mean_bench_boost_value": round(_mean(bb_series), 6),
            "mean_normal_value": round(_mean(normal_series), 6),
            # The audit identity, both sides, so the report can show it.
            "mean_bench_raw_value": round(mean_raw, 6),
            "mean_normal_autosub_points": round(mean_autosub, 6),
            "mean_bench_raw_minus_autosub": round(mean_raw - mean_autosub, 6),
            "identity_residual": round((mean_raw - mean_autosub) - mean_uplift, 9),
            "paired_se": round(paired_se, 6),
            "value_basis": VALUE_BASIS,
            "scored_slots": BENCH_BOOST_SCORED_SLOTS,
            "bench_boost_arm_resolution": "ALL_FIFTEEN_APPEARANCE_GATED",
            "normal_arm_resolution": manager_lineup.MANAGER_LINEUP_VERSION,
            "captain_multiplier_authority": "manager_lineup.captain_multiplier",
            "autosub_authority": "manager_lineup.resolve_world",
            "armband_identical_across_arms": True,
            "mean_armband_extra": round(_mean(armband_series), 6),
            "mean_autosubs_in_normal_arm": round(_mean(autosub_counts), 6),
            "bench_gk_used_probability": round(_mean(gk_used_flags), 6),
            "horizon_events": list(events),
            "four_gw_basis": (
                "Bench Boost alters only the H1 scoring of the manager's existing fifteen; it changes no "
                "squad, transfer, free-transfer, bank or chip state, so the H2-H4 policy consequence is "
                "identical across arms and the horizon must be the exact canonical four events"
            ),
            "planning_event": int(request.planning_event),
            "chip_available": bool(request.chip_available),
            # No selection step exists: both arms are fully determined by the
            # given policy, so the paired difference is averaged over the WHOLE
            # certified world set and there is no winner's-curse partition.
            "selection_step": "NONE_THE_POLICY_IS_GIVEN",
            "selection_worlds": 0,
            "valuation_worlds": int(total_worlds),
            "worlds_used": int(total_worlds),
        },
        uncertainty={
            "p_captain_appears": round(p_captain_appears, 6),
            "armband_source_counts": dict(armband_source_counts),
            "p_armband_captain": round(armband_source_counts.get("CAPTAIN", 0) / max(1, int(total_worlds)), 6),
            "p_armband_vice": round(armband_source_counts.get("VICE", 0) / max(1, int(total_worlds)), 6),
            "p_armband_none": round(armband_source_counts.get("NONE", 0) / max(1, int(total_worlds)), 6),
            "paired_interval_low": round(mean_uplift - PAIRED_INTERVAL_K * paired_se, 6),
            "paired_interval_high": round(mean_uplift + PAIRED_INTERVAL_K * paired_se, 6),
            "paired_quantile_05": round(_quantile(ordered, 0.05), 6),
            "paired_quantile_50": round(_quantile(ordered, 0.50), 6),
            "paired_quantile_95": round(_quantile(ordered, 0.95), 6),
            "interval_k": PAIRED_INTERVAL_K,
            "input_uncertainty_flags": list(propagated),
        },
        reason_codes=tuple(sorted(flags)),
        calibration_status=request.calibration_status,
        evidence={
            **worlds.evidence(),
            "evaluator_version": BENCH_BOOST_EVALUATOR_VERSION,
            "policy_ordering_key": list(_policy_identity(policy)),
            "squad_size": len(squad),
            "review_only": True,
        },
        # REVIEW-ONLY: no Bench Boost future-opportunity value model exists, so a
        # positive uplift may be reported but can never be executed on.
        execution_permitted=False,
    )


def _policy_identity(policy: manager_lineup.ManagerPolicy) -> tuple:
    """The manager's fifteen in the certified engine's own ordering identity."""

    return policy.ordering_key()
