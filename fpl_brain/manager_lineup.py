"""Phase 6A — deterministic fixed-15 manager lineup evaluation.

This module is PURE: it never touches the database, never changes a football
equation, and never consumes randomness.  It consumes a compact manager world
matrix produced by the frozen Phase-5 Monte Carlo engine (full-universe joint
simulation) and evaluates a fixed 15-manager policy:

    starting XI + reserve GK + ordered outfield bench + captain + vice-captain

Scoring basis is CORE plus, where the matrix carries it, the per-player
DETERMINISTIC expected bonus.  The armband objective maximises the value the
captain rule actually multiplies, so it includes that bonus; ``captain_value_basis``
records whether a given matrix supported that, and the XI/bench objective stays
CORE.  No transfer, chip, or route logic lives here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations, permutations
from typing import Any, Mapping, Sequence

MANAGER_LINEUP_VERSION = "manager_lineup_v6.0.0"

#: The authoritative captain-value bases.  The armband is a multiplier applied to
#: the player's actual FPL points, which include bonus, so a CORE-only basis is a
#: declared limitation rather than an equivalent definition.
CAPTAIN_VALUE_BASIS_TOTAL = "CORE_PLUS_DETERMINISTIC_EXPECTED_BONUS"
CAPTAIN_VALUE_BASIS_CORE_ONLY = "CORE_ONLY_BONUS_UNAVAILABLE"

#: A player that is being scored but is ABSENT from the world matrix that is
#: scoring it.  This is never a blank Gameweek: the Monte Carlo engine
#: pre-populates an explicit zero series for every captured player, so a
#: missing key always means the player was never captured.
ROUTE_WORLD_PLAYER_MISSING = "ROUTE_WORLD_PLAYER_MISSING"


class RouteWorldPlayerMissing(ValueError):
    """A scored player has no series in the world matrix (fail closed)."""


def policy_player_ids(policy: "ManagerPolicy") -> set[int]:
    """Every player id a policy can score: XI, bench, captain and vice."""

    return (
        {int(pid) for pid in policy.starter_ids}
        | {int(policy.bench_gk_id)}
        | {int(pid) for pid in policy.bench_outfield_order}
        | {int(policy.captain_id)}
        | {int(policy.vice_captain_id)}
    )


def validate_world_matrix(matrix: Mapping[str, Any], *, context: str = "") -> int:
    """Validate the structure of a world matrix and return its world count.

    Checks that the four required blocks exist and that ``minutes`` and ``core``
    cover exactly the declared ``player_ids`` with the declared world length.
    A truncated or partially-captured matrix fails here rather than being
    silently padded with zeros downstream.
    """

    for key in ("worlds", "player_ids", "minutes", "core"):
        if key not in matrix:
            raise RouteWorldPlayerMissing(
                f"{ROUTE_WORLD_PLAYER_MISSING}: world matrix is missing {key!r}"
                + (f" ({context})" if context else "")
            )
    worlds = int(matrix["worlds"])
    player_ids = {int(pid) for pid in matrix["player_ids"]}
    for block_name in ("minutes", "core"):
        block = matrix[block_name]
        missing = sorted(pid for pid in player_ids if pid not in block)
        if missing:
            raise RouteWorldPlayerMissing(
                f"{ROUTE_WORLD_PLAYER_MISSING}: {block_name} has no series for captured player(s) "
                f"{missing[:8]}{'...' if len(missing) > 8 else ''} ({len(missing)} of {len(player_ids)})"
                + (f" ({context})" if context else "")
            )
        wrong = sorted(pid for pid in player_ids if len(block[pid]) != worlds)
        if wrong:
            raise RouteWorldPlayerMissing(
                f"{ROUTE_WORLD_PLAYER_MISSING}: {block_name} series length != worlds ({worlds}) for "
                f"player(s) {wrong[:8]}{'...' if len(wrong) > 8 else ''}"
                + (f" ({context})" if context else "")
            )
    # ``expected_bonus`` is OPTIONAL, but a malformed one fails closed: it decides
    # the armband, and a silently ignored block would make the captain objective
    # CORE-only while the caller believed bonus was included.
    if "expected_bonus" in matrix:
        block = matrix["expected_bonus"]
        if not isinstance(block, Mapping):
            raise RouteWorldPlayerMissing(
                f"{ROUTE_WORLD_PLAYER_MISSING}: expected_bonus must be a mapping of player -> value"
                + (f" ({context})" if context else "")
            )
        missing = sorted(pid for pid in player_ids if pid not in block)
        if missing:
            raise RouteWorldPlayerMissing(
                f"{ROUTE_WORLD_PLAYER_MISSING}: expected_bonus has no value for captured player(s) "
                f"{missing[:8]}{'...' if len(missing) > 8 else ''} ({len(missing)} of {len(player_ids)})"
                + (f" ({context})" if context else "")
            )
        for pid in sorted(player_ids):
            try:
                value = float(block[pid])
            except (TypeError, ValueError) as exc:
                raise RouteWorldPlayerMissing(
                    f"{ROUTE_WORLD_PLAYER_MISSING}: expected_bonus for player {pid} is not numeric"
                    + (f" ({context})" if context else "")
                ) from exc
            if not math.isfinite(value) or value < 0.0:
                raise RouteWorldPlayerMissing(
                    f"{ROUTE_WORLD_PLAYER_MISSING}: expected_bonus for player {pid} is {value!r}, "
                    "which is not a finite non-negative expected bonus"
                    + (f" ({context})" if context else "")
                )
    return worlds


def expected_bonus_map(matrix: Mapping[str, Any]) -> dict[int, float]:
    """The per-player deterministic expected bonus carried by a world matrix.

    Returns an empty mapping when the matrix carries none, which means the captain
    value is CORE-only; :func:`captain_value_basis` names that state so a caller
    can never mistake it for the total-value armband.
    """

    block = matrix.get("expected_bonus")
    if not isinstance(block, Mapping):
        return {}
    return {int(pid): float(value) for pid, value in block.items()}


def captain_value_basis(matrix: Mapping[str, Any]) -> str:
    """Which quantity the armband objective maximised for this matrix."""

    return CAPTAIN_VALUE_BASIS_TOTAL if expected_bonus_map(matrix) else CAPTAIN_VALUE_BASIS_CORE_ONLY


def validate_matrix_covers_players(
    matrix: Mapping[str, Any],
    required_ids: Sequence[int] | set[int],
    *,
    context: str = "",
) -> None:
    """Fail closed unless every required player has a real series in the matrix.

    A legitimate blank/no-fixture player IS present here with an explicit
    all-zero series; only a genuinely uncaptured player is absent.
    """

    validate_world_matrix(matrix, context=context)
    declared = {int(pid) for pid in matrix["player_ids"]}
    missing = sorted({int(pid) for pid in required_ids} - declared)
    if missing:
        raise RouteWorldPlayerMissing(
            f"{ROUTE_WORLD_PLAYER_MISSING}: player(s) {missing} are absent from the world matrix "
            f"capture set (worlds={int(matrix['worlds'])}, captured={len(declared)})"
            + (f" ({context})" if context else "")
            + ". A missing player is never treated as a blank/zero series."
        )


def validate_scores_cover_players(
    minutes: Mapping[int, Any],
    core: Mapping[int, Any],
    required_ids: Sequence[int] | set[int],
    *,
    context: str = "",
) -> None:
    """Same contract for the per-world scalar mappings passed to ``resolve_world``."""

    for name, block in (("minutes", minutes), ("core", core)):
        missing = sorted({int(pid) for pid in required_ids if int(pid) not in block})
        if missing:
            raise RouteWorldPlayerMissing(
                f"{ROUTE_WORLD_PLAYER_MISSING}: {name} is missing player(s) {missing}"
                + (f" ({context})" if context else "")
                + ". A missing player is never treated as a non-appearance."
            )


SQUAD_SIZE = 15
XI_SIZE = 11
BENCH_SIZE = 4
OUTFIELD_BENCH_SIZE = 3

FORMATION_MIN = {"DEF": 3, "MID": 2, "FWD": 1}
FORMATION_MAX = {"DEF": 5, "MID": 5, "FWD": 3}
OUTFIELD_POSITIONS = ("DEF", "MID", "FWD")


@dataclass(frozen=True)
class ManagerPolicy:
    """One immutable fixed-15 manager decision."""

    starter_ids: tuple[int, ...]
    bench_gk_id: int
    bench_outfield_order: tuple[int, ...]
    captain_id: int
    vice_captain_id: int
    names: Mapping[int, str] | None = field(default=None, compare=False)

    def sorted_starter_ids(self) -> tuple[int, ...]:
        return tuple(sorted(int(pid) for pid in self.starter_ids))

    def ordering_key(self) -> tuple:
        """Deterministic ranking-independent identity (do not use dict order)."""

        return (self.sorted_starter_ids(), int(self.bench_gk_id),
                tuple(int(pid) for pid in self.bench_outfield_order),
                int(self.captain_id), int(self.vice_captain_id))

    def as_dict(self, names: Mapping[int, str] | None = None) -> dict[str, Any]:
        label = names or self.names or {}
        return {
            "starter_ids": self.sorted_starter_ids(),
            "bench_gk_id": int(self.bench_gk_id),
            "bench_outfield_order": tuple(int(pid) for pid in self.bench_outfield_order),
            "captain_id": int(self.captain_id),
            "vice_captain_id": int(self.vice_captain_id),
            "starter_names": [label.get(int(pid), str(pid)) for pid in self.sorted_starter_ids()],
            "bench_names": [label.get(int(pid), str(pid)) for pid in self.bench_outfield_order],
            "bench_gk_name": label.get(int(self.bench_gk_id), str(self.bench_gk_id)),
            "captain_name": label.get(int(self.captain_id), str(self.captain_id)),
            "vice_captain_name": label.get(int(self.vice_captain_id), str(self.vice_captain_id)),
        }


def policy_legality_errors(policy: ManagerPolicy, positions: Mapping[int, str]) -> list[str]:
    """Return every FPL legality violation; empty means legal."""

    errors: list[str] = []
    starters = [int(pid) for pid in policy.starter_ids]
    bench_gk = int(policy.bench_gk_id)
    bench_out = [int(pid) for pid in policy.bench_outfield_order]

    if len(starters) != XI_SIZE:
        errors.append(f"XI_IS_NOT_11: {len(starters)}")
    if len(set(starters)) != len(starters):
        errors.append("DUPLICATE_STARTER")
    if len(set(bench_out)) != OUTFIELD_BENCH_SIZE:
        errors.append(f"OUTFIELD_BENCH_NOT_3: {len(bench_out)}")

    squad = set(starters) | {bench_gk} | set(bench_out)
    if len(squad) != SQUAD_SIZE:
        errors.append(f"SQUAD_NOT_15_DISTINCT: {len(squad)}")
    for pid, label in [(p, "STARTER") for p in starters] + [(bench_gk, "BENCH_GK")] + [
        (p, "OUTFIELD_BENCH") for p in bench_out
    ]:
        if str(positions.get(pid) or "") not in ("GKP", "DEF", "MID", "FWD"):
            errors.append(f"{label}_UNKNOWN_POSITION: {pid}")

    starting_gks = [pid for pid in starters if positions.get(pid) == "GKP"]
    if len(starting_gks) != 1:
        errors.append(f"XI_NOT_EXACTLY_ONE_GK: {len(starting_gks)}")
    if positions.get(bench_gk) != "GKP":
        errors.append("BENCH_GK_IS_NOT_GKP")
    for pid in bench_out:
        if positions.get(pid) == "GKP":
            errors.append(f"OUTFIELD_BENCH_CONTAINS_GK: {pid}")

    counts = _position_counts(starters, positions)
    for position in OUTFIELD_POSITIONS:
        value = counts.get(position, 0)
        if value < FORMATION_MIN[position] or value > FORMATION_MAX[position]:
            errors.append(f"XI_FORMATION_ILLEGAL: {position}={value}")

    if len(starters) + len(bench_out) + 1 != SQUAD_SIZE and not errors:
        errors.append("SQUAD_PARTITION_MISMATCH")
    if set(starters) & ({bench_gk} | set(bench_out)):
        errors.append("STARTER_ALSO_ON_BENCH")

    captain = int(policy.captain_id)
    vice = int(policy.vice_captain_id)
    if captain not in starters:
        errors.append("CAPTAIN_NOT_IN_XI")
    if vice not in starters:
        errors.append("VICE_NOT_IN_XI")
    if captain == vice:
        errors.append("CAPTAIN_EQUALS_VICE")
    return errors


def _position_counts(player_ids: Sequence[int], positions: Mapping[int, str]) -> dict[str, int]:
    counts = {position: 0 for position in (*OUTFIELD_POSITIONS, "GKP")}
    for pid in player_ids:
        position = str(positions.get(int(pid)) or "")
        if position in counts:
            counts[position] += 1
    return counts


def formation_is_legal(counts: Mapping[str, int]) -> bool:
    return all(
        FORMATION_MIN[position] <= counts.get(position, 0) <= FORMATION_MAX[position]
        for position in OUTFIELD_POSITIONS
    )


@dataclass(frozen=True)
class WorldOutcome:
    counted_ids: tuple[int, ...]
    entrants: tuple[int, ...]
    gk_used: bool
    autosub_count: int
    autosub_points: float


def resolve_world(
    policy: ManagerPolicy,
    positions: Mapping[int, str],
    minutes: Mapping[int, float],
    core: Mapping[int, float],
    *,
    require_player_ids: Sequence[int] | set[int] | None = None,
) -> WorldOutcome:
    """Resolve one world's autosubs and return the counted team.

    Appearance is ``GW minutes > 0`` only.  A starter who plays (even one minute,
    even for negative points) is never replaced.  Bench priority dominates; the
    chosen bench subset maximises the number of absent starters replaced subject
    to a legal final formation, and ties are broken by the earliest bench order.

    ``require_player_ids`` (production callers pass the policy's players) makes
    a player absent from ``minutes``/``core`` a hard error instead of an
    implicit non-appearance.  A legitimate blank player is PRESENT with an
    explicit zero, so absence always means the player was never captured.
    """

    if require_player_ids is not None:
        validate_scores_cover_players(minutes, core, require_player_ids,
                                      context="resolve_world")

    def appeared(pid: int) -> bool:
        return float(minutes.get(int(pid), 0.0)) > 0.0

    starters = [int(pid) for pid in policy.starter_ids]
    starter_gk = next(pid for pid in starters if positions.get(pid) == "GKP")
    bench_gk = int(policy.bench_gk_id)
    gk_used = (not appeared(starter_gk)) and appeared(bench_gk)
    counted_gk = bench_gk if gk_used else starter_gk

    starters_out = [pid for pid in starters if positions.get(pid) != "GKP"]
    appearing_starters = [pid for pid in starters_out if appeared(pid)]
    absent_count = len(starters_out) - len(appearing_starters)
    bench_out = [int(pid) for pid in policy.bench_outfield_order]
    appearing_bench = [pid for pid in bench_out if appeared(pid)]

    chosen: tuple[int, ...] = ()
    best: tuple[tuple[int, ...], int] | None = None
    # Exhaustive over <= 3 bench players: exact and tiny.
    for size in range(len(appearing_bench), -1, -1):
        legal_options: list[tuple[int, ...]] = []
        for candidate in combinations(range(len(appearing_bench)), size):
            if size > absent_count:
                continue
            ids = [appearing_bench[index] for index in candidate]
            counts = _position_counts(appearing_starters + ids, positions)
            if formation_is_legal(counts):
                legal_options.append(candidate)
        if legal_options:
            # Lexicographically earliest bench-priority subsequence.
            chosen = min(legal_options)
            best = (chosen, size)
            break
    entrants = tuple(appearing_bench[index] for index in (best[0] if best else ()))

    counted = sorted(appearing_starters + list(entrants) + [counted_gk])
    autosub_points = sum(float(core.get(pid, 0.0)) for pid in entrants)
    if gk_used:
        autosub_points += float(core.get(bench_gk, 0.0))
    return WorldOutcome(
        counted_ids=tuple(counted),
        entrants=entrants,
        gk_used=gk_used,
        autosub_count=len(entrants) + (1 if gk_used else 0),
        autosub_points=autosub_points,
    )


def captain_multiplier(
    policy: ManagerPolicy, minutes: Mapping[int, float], core: Mapping[int, float],
    *, require_player_ids: Sequence[int] | set[int] | None = None,
) -> tuple[float, str]:
    """Extra (doubled) points and which armband applies, from the ORIGINAL XI."""

    if require_player_ids is not None:
        validate_scores_cover_players(minutes, core, require_player_ids,
                                      context="captain_multiplier")

    captain = int(policy.captain_id)
    vice = int(policy.vice_captain_id)
    if float(minutes.get(captain, 0.0)) > 0.0:
        return float(core.get(captain, 0.0)), "CAPTAIN"
    if float(minutes.get(vice, 0.0)) > 0.0:
        return float(core.get(vice, 0.0)), "VICE"
    return 0.0, "NONE"


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    index = min(len(sorted_values) - 1, max(0, int(round(q * (len(sorted_values) - 1)))))
    return float(sorted_values[index])


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def evaluate_policy(
    policy: ManagerPolicy,
    world_matrix: Mapping[str, Any],
    positions: Mapping[int, str],
) -> dict[str, Any]:
    """Evaluate one policy over the joint manager worlds (CORE basis)."""

    # Every player this policy can score must be in the matrix that scores it.
    validate_matrix_covers_players(world_matrix, policy_player_ids(policy),
                                   context="evaluate_policy")
    worlds = int(world_matrix["worlds"])
    core = world_matrix["core"]
    minutes = world_matrix["minutes"]
    player_ids = [int(pid) for pid in world_matrix["player_ids"]]
    required = policy_player_ids(policy)

    scores: list[float] = []
    extras: list[float] = []
    autosub_points: list[float] = []
    autosub_counts: list[float] = []
    any_autosub: list[float] = []
    slot_used = {0: 0, 1: 0, 2: 0}
    gk_used_flags: list[float] = []
    counted_counts: dict[int, int] = {pid: 0 for pid in player_ids}

    def world_value(mapping: Mapping[int, Sequence[float]], pid: int, index: int) -> float:
        series = mapping.get(int(pid))
        return float(series[index]) if series is not None else 0.0

    for world in range(worlds):
        w_minutes = {pid: world_value(minutes, pid, world) for pid in player_ids}
        w_core = {pid: world_value(core, pid, world) for pid in player_ids}
        outcome = resolve_world(policy, positions, w_minutes, w_core,
                                require_player_ids=required)
        base = sum(w_core[pid] for pid in outcome.counted_ids)
        extra, _armband = captain_multiplier(policy, w_minutes, w_core,
                                             require_player_ids=required)
        scores.append(base + extra)
        extras.append(extra)
        autosub_points.append(outcome.autosub_points)
        autosub_counts.append(float(outcome.autosub_count))
        any_autosub.append(1.0 if outcome.autosub_count > 0 else 0.0)
        gk_used_flags.append(1.0 if outcome.gk_used else 0.0)
        for pid in outcome.counted_ids:
            counted_counts[pid] = counted_counts.get(pid, 0) + 1
        bench_out = [int(pid) for pid in policy.bench_outfield_order]
        for slot, pid in enumerate(bench_out):
            if pid in outcome.entrants:
                slot_used[slot] += 1

    ordered = sorted(scores)
    captain = int(policy.captain_id)
    vice = int(policy.vice_captain_id)
    captain_appears = [1.0 if world_value(minutes, captain, w) > 0 else 0.0 for w in range(worlds)]
    vice_takes = [
        1.0 if (world_value(minutes, captain, w) <= 0 and world_value(minutes, vice, w) > 0) else 0.0
        for w in range(worlds)
    ]
    no_multiplier = [
        1.0 if (world_value(minutes, captain, w) <= 0 and world_value(minutes, vice, w) <= 0) else 0.0
        for w in range(worlds)
    ]

    return {
        "worlds": worlds,
        "mean_core": _mean(scores),
        "median_core": _quantile(ordered, 0.5),
        "std_core": _std(scores),
        "q10_core": _quantile(ordered, 0.10),
        "q25_core": _quantile(ordered, 0.25),
        "q75_core": _quantile(ordered, 0.75),
        "q90_core": _quantile(ordered, 0.90),
        "p_any_autosub": _mean(any_autosub),
        "expected_autosub_points_added": _mean(autosub_points),
        "expected_number_of_autosubs": _mean(autosub_counts),
        "p_captain_appears": _mean(captain_appears),
        "p_vice_takes_captaincy": _mean(vice_takes),
        "p_no_captain_multiplier": _mean(no_multiplier),
        "expected_captain_bonus_core": _mean(extras),
        "bench_slot_1_used_probability": slot_used[0] / worlds if worlds else float("nan"),
        "bench_slot_2_used_probability": slot_used[1] / worlds if worlds else float("nan"),
        "bench_slot_3_used_probability": slot_used[2] / worlds if worlds else float("nan"),
        "bench_gk_used_probability": _mean(gk_used_flags),
        "final_counted_player_probability": {
            pid: (counted_counts.get(pid, 0) / worlds if worlds else float("nan")) for pid in player_ids
        },
    }


def _std(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    mean = sum(values) / len(values)
    return (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5


# ---------------------------------------------------------------------------
# Exact fixed-15 enumeration.
# ---------------------------------------------------------------------------


def enumerate_skeletons(squad_ids: Sequence[int], positions: Mapping[int, str]):
    """Yield (starter_ids, bench_gk_id, bench_outfield_order) for legal XIs.

    Deterministic iteration: goalkeepers ascending, outfield combinations in the
    given squad order, bench orders in permutation order.
    """

    squad = sorted(int(pid) for pid in squad_ids)
    gks = [pid for pid in squad if positions.get(pid) == "GKP"]
    outfield = [pid for pid in squad if positions.get(pid) != "GKP"]
    for starting_gk in gks:
        bench_gk = next(pid for pid in gks if pid != starting_gk)
        for combo in combinations(outfield, XI_SIZE - 1):
            counts = _position_counts(combo, positions)
            if not formation_is_legal(counts):
                continue
            bench_out = tuple(pid for pid in outfield if pid not in combo)
            for order in permutations(bench_out):
                yield tuple(sorted(combo + (starting_gk,))), bench_gk, order


def rank_policies(
    squad_ids: Sequence[int],
    positions: Mapping[int, str],
    world_matrix: Mapping[str, Any],
    *,
    top_k: int = 50,
    skeleton_stats=None,
) -> dict[str, Any]:
    """Exact enumeration of every legal policy, ranked by expected CORE.

    Uses structure: the autosub (base) outcome depends only on the XI/bench
    skeleton and each world's appearance mask, while the captaincy bonus is an
    additive per-player quantity, so all ~377k policy means are obtained without
    evaluating each policy over all worlds.  World-level distributions are
    computed exactly for the top ``top_k`` policies.
    """

    from .manager_worlds import base_skeleton_stats, captain_terms  # local import: no cycle at module load

    # The squad being ranked must be fully captured by the matrix; otherwise an
    # uncaptured player would surface as a raw KeyError deep inside the skeleton
    # statistics instead of the explicit fail-closed diagnostic.
    validate_matrix_covers_players(world_matrix, {int(pid) for pid in squad_ids},
                                   context="rank_policies")
    captain_a, captain_c = captain_terms(world_matrix)
    skeletons = list(enumerate_skeletons(squad_ids, positions))
    base = skeleton_stats or base_skeleton_stats(skeletons, positions, world_matrix)

    heap: list[tuple] = []
    evaluated = 0
    for index, (starter_ids, bench_gk, order) in enumerate(skeletons):
        mean_base = base[index]["mean_core_base"]
        for captain in starter_ids:
            for vice in starter_ids:
                if vice == captain:
                    continue
                evaluated += 1
                mean_total = mean_base + captain_a.get(captain, 0.0) + captain_c.get(vice, {}).get(captain, 0.0)
                policy = ManagerPolicy(starter_ids, bench_gk, order, captain, vice)
                # Rank by mean desc; deterministic tie-breaks (cap, vice, sorted XI, bench order).
                sort_key = (-mean_total, int(captain), int(vice), starter_ids, order)
                if len(heap) < top_k:
                    heap.append((sort_key, policy, mean_total))
                    heap.sort(key=lambda item: item[0])
                elif sort_key < heap[-1][0]:
                    heap[-1] = (sort_key, policy, mean_total)
                    heap.sort(key=lambda item: item[0])
    top = [entry[1] for entry in heap]
    top_means = {entry[1].ordering_key(): entry[2] for entry in heap}
    return {
        "skeletons": len(skeletons),
        "evaluated_policies": evaluated,
        "top_policies": top,
        "top_mean_by_key": top_means,
        "base_stats": base,
    }
