"""Triple Captain evaluator — world-level, inside the exact four-GW policy.

WHAT THE CHIP DOES
------------------
Normal captaincy adds ONE extra copy of the armband holder's score; Triple
Captain adds TWO.  The armband holder is world-dependent: the captain when he
appears, otherwise the vice when he appears, otherwise nobody.  So the chip is
one extra copy of

    arm_w = core[captain][w] if captain appeared in w
            else core[vice][w] if vice appeared in w
            else 0

computed per world from the certified minute/score series.  There is no
``captain_xpts * 3`` shortcut anywhere.

PLAY vs SAVE — BOTH ARMS OPTIMISE
---------------------------------
The comparison is between two LEGAL POLICIES, not between an optimised arm and
whatever pair happened to be handed in:

    PLAY_TC = best legal (captain, vice) pair under Triple Captain
    SAVE    = best legal (captain, vice) pair WITHOUT the chip

Each arm independently takes the argmax over the SAME candidate universe (every
ordered pair of distinct starting players), on the SELECTION worlds only.  Using
the request's own pair for SAVE would understate the counterfactual and inflate
the reported uplift.

INSIDE THE FOUR-GW POLICY
-------------------------
TC's direct scoring effect is H1 only, and it does not touch the squad, transfers,
free-transfer state or bank, so the H2-H4 consequence of the two arms is
identical *for the transfers and lineup*.  A caller that has measured a real
non-zero H2-H4 consequence may pass it as ``route_consequence_delta``; that scalar
is added to the paired per-world difference EXACTLY ONCE and therefore moves
``mean_paired_uplift`` (a constant shift leaves the SE and the dispersion
unchanged — it is never displayed-and-ignored).  The remaining term — what saving
the chip is worth — is the reservation abstraction in ``chip_decision``.

SELECTION vs VALUATION
----------------------
Both pairs are CHOSEN on the deterministic selection worlds and VALUED on the
disjoint valuation worlds, so the reported uplift is not the maximum of the same
sample that produced it.  No candidate is re-selected on the valuation worlds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from . import manager_lineup
from .chip_decision import (
    CALIBRATION_UNCALIBRATED,
    CHIP_ACTION_TC,
    DIAG_CHIP_HORIZON_NOT_CANONICAL,
    DIAG_CHIP_PLANNING_EVENT_MISMATCH,
    DIAG_CHIP_WORLD_CONTRACT_INCOMPLETE,
    ChipEvaluation,
    ChipHorizonBinding,
    ChipInputError,
    ChipWorldInputs,
    partition_worlds,
)

TRIPLE_CAPTAIN_EVALUATOR_VERSION = "chip_tc_v1.1.0"

#: Armband BONUS copies.  Normal captaincy adds one copy (the armband holder
#: counts twice); Triple Captain adds two (he counts three times), so the chip is
#: worth exactly one extra copy of the holder's score.
NORMAL_CAPTAIN_BONUS_COPIES = 1
TRIPLE_CAPTAIN_BONUS_COPIES = 2

#: Interval half-width constant, matching the engine's near-tie convention.
PAIRED_INTERVAL_K = 1.96

#: The contract this evaluator implements for a supplied route consequence.
ROUTE_DELTA_CONTRACT = "ADDED_ONCE_TO_THE_PAIRED_DIFFERENCE"

DIAG_TC_EVALUATED = "CHIP_TC_EVALUATED"
DIAG_TC_PLAY_ARM_DIFFERS_FROM_REQUEST = "CHIP_TC_PLAY_ARM_DIFFERS_FROM_REQUEST"
DIAG_TC_SAVE_ARM_DIFFERS_FROM_REQUEST = "CHIP_TC_SAVE_ARM_DIFFERS_FROM_REQUEST"
DIAG_TC_CAPTAIN_APPEARANCE_UNCERTAIN = "CHIP_TC_CAPTAIN_APPEARANCE_UNCERTAIN"
DIAG_TC_VICE_FALLBACK_MATERIAL = "CHIP_TC_VICE_FALLBACK_MATERIAL"
DIAG_TC_NO_ARMBAND_POSSIBLE = "CHIP_TC_NO_ARMBAND_POSSIBLE"
DIAG_TC_INPUT_UNCERTAINTY = "CHIP_TC_INPUT_UNCERTAINTY_PROPAGATED"
DIAG_TC_ROUTE_DELTA_APPLIED = "CHIP_TC_ROUTE_CONSEQUENCE_DELTA_APPLIED"


@dataclass(frozen=True)
class TripleCaptainRequest:
    """Everything the evaluator may consume — nothing from the football model."""

    worlds: ChipWorldInputs
    horizon_binding: ChipHorizonBinding
    policy: manager_lineup.ManagerPolicy
    positions: Mapping[int, str]
    chip_available: bool = True
    calibration_status: str = CALIBRATION_UNCALIBRATED
    #: Optional measured H2-H4 consequence difference between the two arms, in the
    #: same units as the H1 uplift.  Added to the paired per-world difference
    #: exactly once (see ``ROUTE_DELTA_CONTRACT``); 0.0 by construction because TC
    #: does not change any H2-H4 input.
    route_consequence_delta: float = 0.0
    #: Upstream uncertainty flags to propagate verbatim (role, minutes, staleness).
    input_uncertainty_flags: Sequence[str] = field(default_factory=tuple)

    @property
    def planning_event(self) -> int:
        return int(self.horizon_binding.planning_event)

    @property
    def horizon_events(self) -> tuple[int, ...]:
        return tuple(int(event) for event in self.horizon_binding.horizon_events)


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    """Same convention as the certified lineup engine's interval helper."""

    if not sorted_values:
        return float("nan")
    index = min(len(sorted_values) - 1, max(0, int(round(q * (len(sorted_values) - 1)))))
    return float(sorted_values[index])


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def _std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _armband_series(
    *,
    core: Mapping[int, Sequence[float]],
    appeared: Mapping[int, Sequence[bool]],
    captain: int,
    vice: int,
    indices: Sequence[int],
) -> list[float]:
    """Per-world armband extra for one (captain, vice) pair, on ``indices`` only.

    Mirrors ``manager_lineup.captain_multiplier`` exactly (captain, then vice,
    then nobody) — the certified engine is the oracle for that rule and the
    equivalence is pinned by test.
    """

    captain_core = core[int(captain)]
    vice_core = core[int(vice)]
    captain_appeared = appeared[int(captain)]
    vice_appeared = appeared[int(vice)]
    out: list[float] = []
    for index in indices:
        if captain_appeared[index]:
            out.append(float(captain_core[index]))
        elif vice_appeared[index]:
            out.append(float(vice_core[index]))
        else:
            out.append(0.0)
    return out


def _armband_source_counts(
    *,
    appeared: Mapping[int, Sequence[bool]],
    captain: int,
    vice: int,
    indices: Sequence[int],
) -> dict[str, int]:
    counts = {"CAPTAIN": 0, "VICE": 0, "NONE": 0}
    captain_appeared = appeared[int(captain)]
    vice_appeared = appeared[int(vice)]
    for index in indices:
        if captain_appeared[index]:
            counts["CAPTAIN"] += 1
        elif vice_appeared[index]:
            counts["VICE"] += 1
        else:
            counts["NONE"] += 1
    return counts


def captaincy_candidates(
    policy: manager_lineup.ManagerPolicy, positions: Mapping[int, str]
) -> tuple[tuple[int, int], ...]:
    """Every legal (captain, vice) pair: both in the XI, distinct.

    FPL requires the armband to be on a starting player, and the vice to be a
    different starting player, so the candidate universe is the XI's ordered
    pairs.  BOTH arms search exactly this universe.
    """

    starters = tuple(sorted(int(pid) for pid in policy.starter_ids))
    return tuple((captain, vice) for captain in starters for vice in starters if captain != vice)


def _best_pair(
    *,
    core: Mapping[int, Sequence[float]],
    appeared: Mapping[int, Sequence[bool]],
    candidates: Sequence[tuple[int, int]],
    indices: Sequence[int],
) -> tuple[tuple[int, int], float]:
    """Deterministic argmax of the armband over the candidate universe.

    One implementation, called independently for each arm, so a future change to
    either arm's objective cannot silently reuse the other arm's choice.  Ties
    break on (captain, vice) ascending.
    """

    ranked: list[tuple[float, int, int]] = []
    for candidate_captain, candidate_vice in candidates:
        series = _armband_series(
            core=core, appeared=appeared,
            captain=candidate_captain, vice=candidate_vice, indices=indices,
        )
        ranked.append((-_mean(series), candidate_captain, candidate_vice))
    ranked.sort(key=lambda row: (row[0], row[1], row[2]))
    best_mean, best_captain, best_vice = ranked[0]
    return (int(best_captain), int(best_vice)), -best_mean


def evaluate_triple_captain(request: TripleCaptainRequest) -> ChipEvaluation:
    """Evaluate PLAY_TC against an equally-optimised SAVE arm on one world set."""

    worlds = request.worlds
    total_worlds = worlds.validate()
    binding = request.horizon_binding

    # The evaluation is bound to the exact certified horizon, planning event and
    # certification identity.
    problems = binding.matches_worlds(worlds)
    if int(request.policy.captain_id) == int(request.policy.vice_captain_id):
        problems.append("captain and vice are the same player")
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
            f"{DIAG_CHIP_WORLD_CONTRACT_INCOMPLETE}: the H1 policy is not legal: {legality}",
            reasons=[DIAG_CHIP_WORLD_CONTRACT_INCOMPLETE],
        )
    requested_captain = int(policy.captain_id)
    requested_vice = int(policy.vice_captain_id)
    policy_players = set(int(pid) for pid in policy.starter_ids) | {int(policy.bench_gk_id)} | set(
        int(pid) for pid in policy.bench_outfield_order
    )
    missing = sorted(pid for pid in policy_players if pid not in set(worlds.player_ids))
    if missing:
        raise ChipInputError(
            f"{DIAG_CHIP_WORLD_CONTRACT_INCOMPLETE}: the world inputs do not cover policy player(s) "
            f"{missing[:8]}",
            reasons=[DIAG_CHIP_WORLD_CONTRACT_INCOMPLETE],
        )

    partition = partition_worlds(total_worlds, seed=int(worlds.world_seed))
    appeared: dict[int, tuple[bool, ...]] = {pid: worlds.appeared(pid) for pid in worlds.player_ids}

    candidates = captaincy_candidates(policy, positions)
    if not candidates:
        raise ChipInputError(
            f"{DIAG_CHIP_WORLD_CONTRACT_INCOMPLETE}: no legal (captain, vice) pair exists",
            reasons=[DIAG_CHIP_WORLD_CONTRACT_INCOMPLETE],
        )

    # --- SELECTION: each arm optimises INDEPENDENTLY on the selection worlds ----
    tc_pair, tc_selection_arm = _best_pair(
        core=worlds.core, appeared=appeared, candidates=candidates, indices=partition.selection
    )
    save_pair, save_selection_arm = _best_pair(
        core=worlds.core, appeared=appeared, candidates=candidates, indices=partition.selection
    )

    # --- VALUATION: value the two selected policies on the disjoint worlds ------
    arm_tc = _armband_series(
        core=worlds.core, appeared=appeared,
        captain=tc_pair[0], vice=tc_pair[1], indices=partition.valuation,
    )
    arm_save = _armband_series(
        core=worlds.core, appeared=appeared,
        captain=save_pair[0], vice=save_pair[1], indices=partition.valuation,
    )
    route_delta = float(request.route_consequence_delta)
    paired = [
        float(TRIPLE_CAPTAIN_BONUS_COPIES) * arm_tc[index]
        - float(NORMAL_CAPTAIN_BONUS_COPIES) * arm_save[index]
        + route_delta
        for index in range(len(partition.valuation))
    ]
    mean_uplift = _mean(paired)
    paired_se = _std(paired) / math.sqrt(len(paired)) if paired else float("nan")
    ordered = sorted(paired)

    sources_tc = _armband_source_counts(
        appeared=appeared, captain=tc_pair[0], vice=tc_pair[1], indices=partition.valuation
    )
    sources_save = _armband_source_counts(
        appeared=appeared, captain=save_pair[0], vice=save_pair[1], indices=partition.valuation
    )
    valuation_worlds = max(1, partition.valuation_worlds)
    p_captain_appears = _mean([1.0 if appeared[tc_pair[0]][index] else 0.0 for index in partition.valuation])

    flags: list[str] = [DIAG_TC_EVALUATED]
    if tc_pair != (requested_captain, requested_vice):
        flags.append(DIAG_TC_PLAY_ARM_DIFFERS_FROM_REQUEST)
    if save_pair != (requested_captain, requested_vice):
        flags.append(DIAG_TC_SAVE_ARM_DIFFERS_FROM_REQUEST)
    if 0.0 < p_captain_appears < 1.0:
        flags.append(DIAG_TC_CAPTAIN_APPEARANCE_UNCERTAIN)
    if sources_tc["VICE"] / valuation_worlds > 0.05:
        flags.append(DIAG_TC_VICE_FALLBACK_MATERIAL)
    if sources_tc["NONE"] > 0:
        flags.append(DIAG_TC_NO_ARMBAND_POSSIBLE)
    if route_delta != 0.0:
        flags.append(DIAG_TC_ROUTE_DELTA_APPLIED)
    propagated = tuple(str(flag) for flag in request.input_uncertainty_flags)
    if propagated:
        flags.append(DIAG_TC_INPUT_UNCERTAINTY)

    return ChipEvaluation(
        action=CHIP_ACTION_TC,
        evaluator_version=TRIPLE_CAPTAIN_EVALUATOR_VERSION,
        candidate_metrics={
            "mean_paired_uplift": round(mean_uplift, 6),
            "mean_tc_armband_bonus_selection": round(
                tc_selection_arm * float(TRIPLE_CAPTAIN_BONUS_COPIES), 6
            ),
            "mean_save_armband_bonus_selection": round(
                save_selection_arm * float(NORMAL_CAPTAIN_BONUS_COPIES), 6
            ),
            "mean_tc_armband_bonus_valuation": round(_mean(arm_tc) * TRIPLE_CAPTAIN_BONUS_COPIES, 6),
            "mean_save_armband_bonus_valuation": round(_mean(arm_save) * NORMAL_CAPTAIN_BONUS_COPIES, 6),
            "paired_se": round(paired_se, 6),
            "armband_bonus_copies": {
                "normal": float(NORMAL_CAPTAIN_BONUS_COPIES),
                "triple_captain": float(TRIPLE_CAPTAIN_BONUS_COPIES),
            },
            "route_consequence_delta": route_delta,
            "route_consequence_contract": ROUTE_DELTA_CONTRACT,
            "four_gw_basis": (
                "TC alters only the H1 armband and the chip ledger, so the H2-H4 policy consequence is "
                "identical across arms and the horizon must be the exact canonical four events"
            ),
            "horizon_events": list(events),
            "captain_id": int(tc_pair[0]),
            "vice_captain_id": int(tc_pair[1]),
            "save_captain_id": int(save_pair[0]),
            "save_vice_captain_id": int(save_pair[1]),
            "save_arm_optimised": True,
            "requested_captain_id": requested_captain,
            "requested_vice_captain_id": requested_vice,
            "selected_candidate": f"{int(tc_pair[0])}/{int(tc_pair[1])}",
            "selected_save_candidate": f"{int(save_pair[0])}/{int(save_pair[1])}",
            "candidate_pairs_evaluated": len(candidates),
            "selection_worlds": partition.selection_worlds,
            "valuation_worlds": partition.valuation_worlds,
            "partition_basis": partition.basis,
            "chip_available": bool(request.chip_available),
        },
        uncertainty={
            "p_captain_appears": round(p_captain_appears, 6),
            "p_armband_captain": round(sources_tc["CAPTAIN"] / valuation_worlds, 6),
            "p_armband_vice": round(sources_tc["VICE"] / valuation_worlds, 6),
            "p_armband_none": round(sources_tc["NONE"] / valuation_worlds, 6),
            "armband_source_counts": dict(sources_tc),
            "save_armband_source_counts": dict(sources_save),
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
            "evaluator_version": TRIPLE_CAPTAIN_EVALUATOR_VERSION,
            "selection_valuation_disjoint": True,
        },
    )
