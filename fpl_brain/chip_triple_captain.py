"""Triple Captain evaluator — world-level, inside the exact four-GW policy.

WHAT THE CHIP DOES
------------------
Normal captaincy adds ONE extra copy of the armband holder's score; Triple
Captain adds TWO.  The armband holder is world-dependent: the captain when he
appears, otherwise the vice when he appears, otherwise nobody.  So the whole
chip is one extra copy of

    arm_w = core[captain][w] if captain appeared in w
            else core[vice][w] if vice appeared in w
            else 0

and this module computes it per world from the certified minute/score series.
There is no ``captain_xpts * 3`` shortcut anywhere: captain, vice, appearance
fallback and the armband-source probabilities all come out of the same worlds.

INSIDE THE FOUR-GW POLICY
-------------------------
TC's direct scoring effect is H1 only, and it does not touch the squad, transfers,
free-transfer state or bank.  Therefore the H2–H4 consequence of a TC policy is
*identical* to the SAVE policy's by construction, and the four-GW comparison is

    play TC  =  H1 armband uplift  +  H2-H4 route consequence  -  future opportunity
    save     =  0                  +  H2-H4 route consequence

The two route terms cancel, which is why the evaluator requires the exact
four-event horizon and carries an explicit ``route_consequence_delta`` field
(zero by construction, and available for a caller that has measured something
else) rather than silently assuming it away.  The remaining term — what saving
the chip is worth — is the reservation abstraction in ``chip_decision``.  No
final numeric reservation threshold is invented here.

SELECTION vs VALUATION
----------------------
The (captain, vice) pair is CHOSEN on the deterministic selection worlds and its
value is REPORTED on the disjoint valuation worlds, so the reported uplift is not
the maximum of the same sample that produced it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from . import manager_lineup
from .chip_decision import (
    CALIBRATION_UNCALIBRATED,
    CHIP_ACTION_TC,
    DIAG_CHIP_HORIZON_INCOMPLETE,
    DIAG_CHIP_WORLD_CONTRACT_INCOMPLETE,
    ChipEvaluation,
    ChipInputError,
    ChipWorldInputs,
    partition_worlds,
)

TRIPLE_CAPTAIN_EVALUATOR_VERSION = "chip_tc_v1.0.0"

#: Armband BONUS copies.  Normal captaincy adds one copy (the armband holder
#: counts twice); Triple Captain adds two (he counts three times).  The chip is
#: therefore worth exactly one copy of the holder's score, which is what the
#: paired difference reports.
NORMAL_CAPTAIN_BONUS_COPIES = 1
TRIPLE_CAPTAIN_BONUS_COPIES = 2

#: Interval half-width constant, matching the engine's near-tie convention.
PAIRED_INTERVAL_K = 1.96

DIAG_TC_EVALUATED = "CHIP_TC_EVALUATED"
DIAG_TC_CAPTAIN_CHANGED = "CHIP_TC_CAPTAIN_UNCHANGED_BY_CHIP"
DIAG_TC_CAPTAIN_APPEARANCE_UNCERTAIN = "CHIP_TC_CAPTAIN_APPEARANCE_UNCERTAIN"
DIAG_TC_VICE_FALLBACK_MATERIAL = "CHIP_TC_VICE_FALLBACK_MATERIAL"
DIAG_TC_NO_ARMband_POSSIBLE = "CHIP_TC_NO_ARMband_POSSIBLE"
DIAG_TC_INPUT_UNCERTAINTY = "CHIP_TC_INPUT_UNCERTAINTY_PROPAGATED"


@dataclass(frozen=True)
class TripleCaptainRequest:
    """Everything the evaluator may consume — nothing from the football model."""

    worlds: ChipWorldInputs
    policy: manager_lineup.ManagerPolicy
    positions: Mapping[int, str]
    planning_event: int
    chip_available: bool = True
    calibration_status: str = CALIBRATION_UNCALIBRATED
    #: Optional measured H2-H4 consequence difference between the two arms.  Left
    #: at 0.0 because TC provably does not change any H2-H4 input; a caller that
    #: has measured otherwise may pass it.
    route_consequence_delta: float = 0.0
    #: Upstream uncertainty flags to propagate verbatim (role, minutes, staleness).
    input_uncertainty_flags: Sequence[str] = field(default_factory=tuple)
    required_horizon_length: int = 4


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
    worlds: int,
    core: Mapping[int, Sequence[float]],
    appeared: Mapping[int, Sequence[bool]],
    captain: int,
    vice: int,
    indices: Sequence[int],
) -> list[float]:
    """Per-world armband extra for one (captain, vice) pair, on ``indices`` only.

    Mirrors ``manager_lineup.captain_multiplier`` exactly (captain, then vice,
    then none) — the certified engine is the oracle for that rule, and the
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


def captaincy_candidates(policy: manager_lineup.ManagerPolicy, positions: Mapping[int, str]) -> tuple[tuple[int, int], ...]:
    """Every legal (captain, vice) pair: both in the XI, distinct.

    FPL requires the armband to be on a starting player, and the vice to be a
    different starting player, so the candidate set is the XI's ordered pairs.
    """

    starters = tuple(sorted(int(pid) for pid in policy.starter_ids))
    return tuple((captain, vice) for captain in starters for vice in starters if captain != vice)


def evaluate_triple_captain(request: TripleCaptainRequest) -> ChipEvaluation:
    """Evaluate TC against the SAVE policy on one certified world set."""

    worlds = request.worlds
    total_worlds = worlds.validate()

    # The exact four-GW horizon is required: no fifth event, no shortened horizon.
    events = tuple(int(event) for event in worlds.horizon_events)
    if len(events) != int(request.required_horizon_length):
        raise ChipInputError(
            f"{DIAG_CHIP_HORIZON_INCOMPLETE}: Triple Captain needs the exact "
            f"{int(request.required_horizon_length)}-event horizon, got {list(events)}",
            reasons=[DIAG_CHIP_HORIZON_INCOMPLETE],
        )

    policy = request.policy
    positions = request.positions
    legality = manager_lineup.policy_legality_errors(policy, positions)
    if legality:
        raise ChipInputError(
            f"{DIAG_CHIP_WORLD_CONTRACT_INCOMPLETE}: the H1 policy is not legal: {legality}",
            reasons=[DIAG_CHIP_WORLD_CONTRACT_INCOMPLETE],
        )
    captain = int(policy.captain_id)
    vice = int(policy.vice_captain_id)
    if captain == vice:
        raise ChipInputError(
            f"{DIAG_CHIP_WORLD_CONTRACT_INCOMPLETE}: captain and vice are the same player {captain}",
            reasons=[DIAG_CHIP_WORLD_CONTRACT_INCOMPLETE],
        )
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

    # --- SELECTION: choose the armband pair on the selection worlds only -------
    ranked: list[tuple[float, int, int, list[float]]] = []
    for candidate_captain, candidate_vice in candidates:
        series = _armband_series(
            worlds=total_worlds, core=worlds.core, appeared=appeared,
            captain=candidate_captain, vice=candidate_vice, indices=partition.selection,
        )
        ranked.append((-_mean(series), candidate_captain, candidate_vice, series))
    ranked.sort(key=lambda row: (row[0], row[1], row[2]))
    best_mean_selection, best_captain, best_vice, _ = ranked[0]

    # --- VALUATION: report the chosen pair on the disjoint worlds --------------
    arm_tc = _armband_series(
        worlds=total_worlds, core=worlds.core, appeared=appeared,
        captain=best_captain, vice=best_vice, indices=partition.valuation,
    )
    arm_save = _armband_series(
        worlds=total_worlds, core=worlds.core, appeared=appeared,
        captain=captain, vice=vice, indices=partition.valuation,
    )
    paired = [
        float(TRIPLE_CAPTAIN_BONUS_COPIES) * arm_tc[index]
        - float(NORMAL_CAPTAIN_BONUS_COPIES) * arm_save[index]
        for index in range(len(partition.valuation))
    ]
    mean_uplift = _mean(paired)
    paired_se = _std(paired) / math.sqrt(len(paired)) if paired else float("nan")
    ordered = sorted(paired)

    sources = _armband_source_counts(
        appeared=appeared, captain=best_captain, vice=best_vice, indices=partition.valuation
    )
    valuation_worlds = max(1, partition.valuation_worlds)
    p_captain_appears = _mean([1.0 if appeared[best_captain][index] else 0.0 for index in partition.valuation])
    p_captain_chosen = sources["CAPTAIN"] / valuation_worlds
    p_vice_armband = sources["VICE"] / valuation_worlds
    p_no_armband = sources["NONE"] / valuation_worlds

    flags: list[str] = [DIAG_TC_EVALUATED]
    if int(best_captain) != captain:
        flags.append(DIAG_TC_CAPTAIN_CHANGED)
    if 0.0 < p_captain_appears < 1.0:
        flags.append(DIAG_TC_CAPTAIN_APPEARANCE_UNCERTAIN)
    if p_vice_armband > 0.05:
        flags.append(DIAG_TC_VICE_FALLBACK_MATERIAL)
    if p_no_armband > 0.0:
        flags.append(DIAG_TC_NO_ARMband_POSSIBLE)
    propagated = tuple(str(flag) for flag in request.input_uncertainty_flags)
    if propagated:
        flags.append(DIAG_TC_INPUT_UNCERTAINTY)

    return ChipEvaluation(
        action=CHIP_ACTION_TC,
        evaluator_version=TRIPLE_CAPTAIN_EVALUATOR_VERSION,
        candidate_metrics={
            "mean_paired_uplift": round(mean_uplift, 6),
            "mean_tc_armband_bonus_selection": round(
                -best_mean_selection * float(TRIPLE_CAPTAIN_BONUS_COPIES), 6
            ),
            "mean_tc_armband_bonus_valuation": round(
                _mean(arm_tc) * float(TRIPLE_CAPTAIN_BONUS_COPIES), 6
            ),
            "mean_save_armband_bonus_valuation": round(
                _mean(arm_save) * float(NORMAL_CAPTAIN_BONUS_COPIES), 6
            ),
            "paired_se": round(paired_se, 6),
            "armband_bonus_copies": {
                "normal": float(NORMAL_CAPTAIN_BONUS_COPIES),
                "triple_captain": float(TRIPLE_CAPTAIN_BONUS_COPIES),
            },
            "route_consequence_delta": float(request.route_consequence_delta),
            "four_gw_basis": (
                "TC alters only the H1 armband and the chip ledger, so the H2-H4 policy consequence is "
                "identical across arms; the horizon is required to be exactly four events"
            ),
            "horizon_events": list(events),
            "captain_id": int(best_captain),
            "vice_captain_id": int(best_vice),
            "save_captain_id": captain,
            "save_vice_captain_id": vice,
            "selected_candidate": f"{int(best_captain)}/{int(best_vice)}",
            "candidate_pairs_evaluated": len(candidates),
            "selection_worlds": partition.selection_worlds,
            "valuation_worlds": partition.valuation_worlds,
            "partition_basis": partition.basis,
            "chip_available": bool(request.chip_available),
        },
        uncertainty={
            "p_captain_appears": round(p_captain_appears, 6),
            "p_armband_captain": round(p_captain_chosen, 6),
            "p_armband_vice": round(p_vice_armband, 6),
            "p_armband_none": round(p_no_armband, 6),
            "armband_source_counts": dict(sources),
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
