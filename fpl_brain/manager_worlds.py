"""Phase 6A — build joint manager worlds from the FROZEN Phase-5 run.

The full football universe is simulated exactly as Phase 5 requires (run 71
inputs, run-71 config, run-71 seed).  Only afterwards is the manager's 15-player
GW matrix extracted; changing the XI, bench, captain, vice or the manager squad
subset can never alter a football outcome.

No football equation, RNG identity or draw order is changed: the Monte Carlo
engine is only asked to additionally record a compact per-world GW matrix for
the squad (a pure side effect).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from . import manager_lineup, monte_carlo
from .scoring_rules import POSITION_IDS

MANAGER_WORLDS_VERSION = "manager_worlds_v6.0.0"


def resolve_squad(context, conn) -> dict[str, Any]:
    """Resolve the fixed 15 from the canonical PlanningContext (never hardcoded)."""

    players = list((context.squad or {}).get("players") or [])
    squad_ids = sorted(int(row["player_id"]) for row in players)
    rows = conn.execute(
        "SELECT id, element_type, team_id, web_name, full_name FROM players WHERE id IN (%s)"
        % ",".join("?" for _ in squad_ids),
        squad_ids,
    ).fetchall() if squad_ids else []
    meta = {int(row["id"]): dict(row) for row in rows}
    positions = {
        pid: POSITION_IDS.get(int(meta.get(pid, {}).get("element_type")))
        for pid in squad_ids
    }
    names = {
        pid: (meta.get(pid, {}).get("full_name") or meta.get(pid, {}).get("web_name") or f"Player {pid}")
        for pid in squad_ids
    }
    return {
        "squad_ids": squad_ids,
        "positions": positions,
        "names": names,
        "player_count": len(squad_ids),
        "squad_state": (context.squad or {}).get("squad_state"),
        "squad_source": (context.squad or {}).get("squad_source"),
    }


def build_manager_worlds(
    conn,
    *,
    planning_event: int,
    minutes_run_id: int,
    xpts_run_id: int,
    team_run_id: int,
    squad_ids: Sequence[int],
    simulations: int = 10_000,
    seed: int = 20260911,
    occupancy_audit: bool = True,
) -> dict[str, Any]:
    """Simulate the full frozen universe and extract the squad GW matrix."""

    fixtures = monte_carlo.load_fixture_inputs(
        conn, event=int(planning_event), xpts_run_id=int(xpts_run_id),
        minutes_run_id=int(minutes_run_id), team_run_id=int(team_run_id),
    )
    config = monte_carlo.MonteCarloConfig(
        simulations=int(simulations), seed=int(seed), occupancy_audit=bool(occupancy_audit)
    )
    result = monte_carlo.simulate(fixtures, config, capture_player_ids=squad_ids)
    return {
        "world_matrix": result["world_matrix"],
        "simulation": {
            "worlds": int(config.simulations),
            "seed": int(config.seed),
            "config_hash": config.config_hash(),
            "occupancy_audit": bool(config.occupancy_audit),
            "occupancy_violations": int(result.get("occupancy_violations") or 0),
            "minute_mass_violations": int(result.get("minute_mass_violations") or 0),
            "invalid_lineups": int(result.get("invalid_lineups") or 0),
            "mc_model_version": monte_carlo.MONTE_CARLO_MODEL_VERSION,
        },
        "input_run_ids": {
            "minutes": int(minutes_run_id), "xpts": int(xpts_run_id), "team": int(team_run_id),
        },
        "fixtures": sorted(int(fixture_id) for fixture_id in fixtures),
    }


def _appearance_groups(world_matrix: Mapping[str, Any]):
    """Group worlds by appearance mask; keep counts and per-player core sums.

    Memoised per matrix PROVENANCE identity: the result depends only on the matrix,
    never on the squad, so within one run a second squad ranked against the same
    event's matrix must not pay for it again.  Returned objects are read-only for
    callers (``base_skeleton_stats`` only reads them).
    """

    _MATRIX_MEMO_STATS["appearance_groups_calls"] += 1
    identity = matrix_identity(world_matrix)
    if identity is not None:
        cached = _MATRIX_MEMO.get(("appearance_groups", identity))
        if cached is not None:
            _MATRIX_MEMO_STATS["appearance_groups_hits"] += 1
            return cached

    _MATRIX_MEMO_STATS["appearance_groups_computes"] += 1
    worlds = int(world_matrix["worlds"])
    player_ids = [int(pid) for pid in world_matrix["player_ids"]]
    # Fail closed: every declared player must have a real minutes AND core
    # series of the declared length.  A missing key is never a blank Gameweek
    # (a blank player is present with an explicit zero series).
    manager_lineup.validate_world_matrix(world_matrix, context="_appearance_groups")
    minutes = world_matrix["minutes"]
    core = world_matrix["core"]
    index = {pid: position for position, pid in enumerate(player_ids)}
    groups: dict[int, dict[str, Any]] = {}
    for world in range(worlds):
        mask = 0
        for pid in player_ids:
            if float(minutes[pid][world]) > 0.0:
                mask |= 1 << index[pid]
        group = groups.get(mask)
        if group is None:
            group = {"count": 0, "core_sum": [0.0] * len(player_ids)}
            groups[mask] = group
        group["count"] += 1
        for pid in player_ids:
            group["core_sum"][index[pid]] += float(core[pid][world])
    result = (player_ids, index, groups, worlds)
    if identity is not None:
        _MATRIX_MEMO[("appearance_groups", identity)] = result
    return result


_POSITION_CODE = {"DEF": 0, "MID": 1, "FWD": 2, "GKP": 3}
_MIN_POS = (3, 2, 1)
_MAX_POS = (5, 5, 3)

#: Provenance identity key stamped on a real world matrix by
#: ``route_optimizer.build_event_worlds``.  Two matrices carrying the same value are
#: the same football worlds, so anything that depends ONLY on the matrix can be
#: computed once per identity instead of once per squad.
MATRIX_IDENTITY_KEY = "_p2_matrix_identity"

#: Run-scoped, matrix-keyed memos.  Never persisted, never keyed on a
#: certification identity from a different run (the key IS the certified cache
#: identity of the matrix that is already in memory).
_MATRIX_MEMO: dict[tuple, Any] = {}
_MATRIX_MEMO_STATS: dict[str, int] = {
    "captain_terms_calls": 0, "captain_terms_hits": 0, "captain_terms_computes": 0,
    "appearance_groups_calls": 0, "appearance_groups_hits": 0, "appearance_groups_computes": 0,
    "base_skeleton_stats_calls": 0, "entrant_cache_hits": 0, "entrant_cache_misses": 0,
}


def matrix_identity(world_matrix: Mapping[str, Any]) -> Any | None:
    """The stamped provenance identity of a world matrix, or ``None``.

    ``None`` means "unidentified": memos are skipped rather than keyed on a guess
    (an ``id()``-keyed memo is not sound, because an address can be reused).
    """

    try:
        value = world_matrix.get(MATRIX_IDENTITY_KEY)
    except (AttributeError, TypeError):
        return None
    return value or None


def clear_matrix_memos() -> None:
    """Drop every matrix-keyed memo (tests; also a clean run boundary)."""

    _MATRIX_MEMO.clear()


def matrix_memo_stats() -> dict[str, int]:
    return dict(_MATRIX_MEMO_STATS)


def reset_matrix_memo_stats() -> None:
    for key in _MATRIX_MEMO_STATS:
        _MATRIX_MEMO_STATS[key] = 0


def _select_entrants(app_out_counts, absent_count, appearing_bench):
    """Lexicographically earliest max-size bench subset with a legal formation.

    ``appearing_bench`` is a list of ``(bench_slot, position_code)`` in bench
    priority order.  ``app_out_counts`` is (DEF, MID, FWD) among APPEARING
    outfield starters.
    """

    from itertools import combinations as _combinations

    limit = min(len(appearing_bench), absent_count)
    for size in range(limit, -1, -1):
        for combo in _combinations(range(len(appearing_bench)), size):
            d, m, f = app_out_counts
            for slot in combo:
                code = appearing_bench[slot][1]
                if code == 0:
                    d += 1
                elif code == 1:
                    m += 1
                else:
                    f += 1
            if _MIN_POS[0] <= d <= _MAX_POS[0] and _MIN_POS[1] <= m <= _MAX_POS[1] and _MIN_POS[2] <= f <= _MAX_POS[2]:
                return combo
    return ()


def base_skeleton_stats_reference(skeletons: Sequence[tuple], positions: Mapping[int, str],
                                  world_matrix: Mapping[str, Any]) -> list[dict[str, Any]]:
    """REFERENCE implementation of the exact base-skeleton statistics.

    Kept verbatim as the equivalence oracle for
    :func:`base_skeleton_stats`.  It recomputes, for every (skeleton, distinct
    appearance-mask) pair, the absent-starter counts, the appearing bench, the
    entrant selection and the goalkeeper swap, then accumulates five weighted
    sums in appearance-group order.
    """

    player_ids, index, groups, worlds = _appearance_groups(world_matrix)
    codes = [_POSITION_CODE.get(str(positions.get(pid) or ""), 3) for pid in player_ids]
    mask_items = []
    for mask, group in groups.items():
        count = group["count"]
        mean_core = tuple(group["core_sum"][position] / count for position in range(len(player_ids)))
        appeared = tuple(bool(mask >> position & 1) for position in range(len(player_ids)))
        mask_items.append((count / worlds, appeared, mean_core))

    stats: list[dict[str, Any]] = []
    for starter_ids, bench_gk, order in skeletons:
        starter_idx = tuple(index[int(pid)] for pid in starter_ids)
        bench_idx = tuple(index[int(pid)] for pid in order)
        gk_idx = index[int(bench_gk)]
        starter_out_idx = tuple(i for i in starter_idx if codes[i] != 3)
        weighted_base = 0.0
        any_autosub = 0.0
        autosub_points = 0.0
        autosub_count = 0.0
        gk_used = 0.0
        slot_used = [0.0, 0.0, 0.0]
        for weight, appeared, mean_core in mask_items:
            absent = 0
            sd = sm = sf = 0
            for i in starter_out_idx:
                if appeared[i]:
                    code = codes[i]
                    if code == 0:
                        sd += 1
                    elif code == 1:
                        sm += 1
                    else:
                        sf += 1
                else:
                    absent += 1
            appearing_bench = []
            for slot, i in enumerate(bench_idx):
                if appeared[i]:
                    appearing_bench.append((slot, codes[i]))
            chosen = _select_entrants((sd, sm, sf), absent, appearing_bench)
            # GK handling
            starter_gk_idx = next(i for i in starter_idx if codes[i] == 3)
            bench_gk_appeared = appeared[gk_idx]
            gk_swap = (not appeared[starter_gk_idx]) and bench_gk_appeared
            # Base core: every starter's mean core (0 when absent) + entrants + swapped GK.
            base = sum(mean_core[i] for i in starter_idx)
            for position in chosen:
                base += mean_core[bench_idx[appearing_bench[position][0]]]
            if gk_swap:
                base += mean_core[gk_idx]
            added = sum(mean_core[bench_idx[appearing_bench[position][0]]] for position in chosen) + (
                mean_core[gk_idx] if gk_swap else 0.0
            )
            count = len(chosen) + (1 if gk_swap else 0)
            weighted_base += weight * base
            autosub_points += weight * added
            autosub_count += weight * count
            any_autosub += weight * (1.0 if count > 0 else 0.0)
            gk_used += weight * (1.0 if gk_swap else 0.0)
            for position in chosen:
                slot_used[appearing_bench[position][0]] += weight
        stats.append({
            "mean_core_base": weighted_base,
            "p_any_autosub": any_autosub,
            "expected_autosub_points_added": autosub_points,
            "expected_number_of_autosubs": autosub_count,
            "bench_gk_used_probability": gk_used,
            "bench_slot_1_used_probability": slot_used[0],
            "bench_slot_2_used_probability": slot_used[1],
            "bench_slot_3_used_probability": slot_used[2],
        })
    return stats


def base_skeleton_stats(skeletons: Sequence[tuple], positions: Mapping[int, str],
                        world_matrix: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Exact base (autosub-only, no captaincy) stats per XI/bench skeleton.

    BIT-PRESERVING speed-up of :func:`base_skeleton_stats_reference`.  The reference
    recomputes the same (absent counts, appearing bench, entrant selection, GK swap)
    tuple once per (skeleton, distinct appearance mask).  That tuple is a pure
    function of the skeleton and the appearance bits of the skeleton's OWN 15
    players, so it is computed once per distinct such pattern and reused.

    What is deliberately NOT changed:

    * the outer loop still walks the appearance groups in the same order;
    * every weighted sum is still accumulated in that order, with the same operand
      order and the same starting value ``0``/``0.0``;
    * the per-group ``mean_core`` is still the mean over that group's worlds, so no
      world-level term is re-associated.

    The result is therefore bit-identical to the reference, not merely close.
    """

    _MATRIX_MEMO_STATS["base_skeleton_stats_calls"] += 1
    player_ids, index, groups, worlds = _appearance_groups(world_matrix)
    codes = [_POSITION_CODE.get(str(positions.get(pid) or ""), 3) for pid in player_ids]
    mask_items = []
    for mask, group in groups.items():
        count = group["count"]
        mean_core = tuple(group["core_sum"][position] / count for position in range(len(player_ids)))
        appeared = tuple(bool(mask >> position & 1) for position in range(len(player_ids)))
        mask_items.append((count / worlds, appeared, mean_core))

    stats: list[dict[str, Any]] = []
    for starter_ids, bench_gk, order in skeletons:
        starter_idx = tuple(index[int(pid)] for pid in starter_ids)
        bench_idx = tuple(index[int(pid)] for pid in order)
        gk_idx = index[int(bench_gk)]
        starter_out_idx = tuple(i for i in starter_idx if codes[i] != 3)
        # Loop-invariant in the reference (it recomputes it for every group).
        starter_gk_idx = next(i for i in starter_idx if codes[i] == 3)
        # The appearance bits this skeleton's autosub decision can depend on.
        watch = starter_out_idx + bench_idx + (starter_gk_idx, gk_idx)
        choice_cache: dict[tuple, tuple] = {}
        weighted_base = 0.0
        any_autosub = 0.0
        autosub_points = 0.0
        autosub_count = 0.0
        gk_used = 0.0
        slot_used = [0.0, 0.0, 0.0]
        for weight, appeared, mean_core in mask_items:
            key = tuple(appeared[i] for i in watch)
            decision = choice_cache.get(key)
            if decision is None:
                _MATRIX_MEMO_STATS["entrant_cache_misses"] += 1
                absent = 0
                sd = sm = sf = 0
                for i in starter_out_idx:
                    if appeared[i]:
                        code = codes[i]
                        if code == 0:
                            sd += 1
                        elif code == 1:
                            sm += 1
                        else:
                            sf += 1
                    else:
                        absent += 1
                appearing_bench = []
                for slot, i in enumerate(bench_idx):
                    if appeared[i]:
                        appearing_bench.append((slot, codes[i]))
                chosen = _select_entrants((sd, sm, sf), absent, appearing_bench)
                gk_swap = (not appeared[starter_gk_idx]) and appeared[gk_idx]
                entrant_idx = tuple(bench_idx[appearing_bench[position][0]] for position in chosen)
                entrant_slots = tuple(appearing_bench[position][0] for position in chosen)
                decision = (entrant_idx, entrant_slots, gk_swap)
                choice_cache[key] = decision
            else:
                _MATRIX_MEMO_STATS["entrant_cache_hits"] += 1
            entrant_idx, entrant_slots, gk_swap = decision
            # ``sum(...)`` is kept EXACTLY as the reference wrote it, including the
            # generator form.  This is not stylistic: CPython 3.12+ implements
            # ``sum()`` over floats with Neumaier compensated summation, so replacing
            # it with a naive ``total += x`` loop changes the last ulp (measured:
            # ``sum([4.1]*11) == 45.099999999999994`` against a naive ``45.10000000000001``).
            # A bit-preserving fast path must therefore keep ``sum``.
            base = sum(mean_core[i] for i in starter_idx)
            for i in entrant_idx:
                base += mean_core[i]
            if gk_swap:
                base += mean_core[gk_idx]
            added = sum(mean_core[i] for i in entrant_idx) + (
                mean_core[gk_idx] if gk_swap else 0.0
            )
            count = len(entrant_idx) + (1 if gk_swap else 0)
            weighted_base += weight * base
            autosub_points += weight * added
            autosub_count += weight * count
            any_autosub += weight * (1.0 if count > 0 else 0.0)
            gk_used += weight * (1.0 if gk_swap else 0.0)
            for slot in entrant_slots:
                slot_used[slot] += weight
        stats.append({
            "mean_core_base": weighted_base,
            "p_any_autosub": any_autosub,
            "expected_autosub_points_added": autosub_points,
            "expected_number_of_autosubs": autosub_count,
            "bench_gk_used_probability": gk_used,
            "bench_slot_1_used_probability": slot_used[0],
            "bench_slot_2_used_probability": slot_used[1],
            "bench_slot_3_used_probability": slot_used[2],
        })
    return stats


def captain_terms(world_matrix: Mapping[str, Any]) -> tuple[dict[int, float], dict[int, dict[int, float]]]:
    """Additive captaincy terms (independent of the XI/bench skeleton).

    ``A[p] = E[core_p * appeared_p]`` and ``C[v][c] = E[core_v * appeared_v * not appeared_c]``,
    so the expected captaincy bonus for a legal pair (c, v) is ``A[c] + C[v][c]``.

    The terms depend ONLY on the world matrix — never on the squad — so the result is
    memoised per matrix PROVENANCE identity.  The 165x164 pair loop over every world
    is otherwise repeated once per squad ranked against the same event's matrix.

    The cached inner dicts are shared; callers must treat them as read-only (the only
    production caller, ``manager_lineup.rank_policies``, reads two levels deep and
    mutates nothing).  The two outer dicts are copied on the way out so a caller
    cannot change how the cache is keyed.
    """

    _MATRIX_MEMO_STATS["captain_terms_calls"] += 1
    identity = matrix_identity(world_matrix)
    if identity is not None:
        cached = _MATRIX_MEMO.get(("captain_terms", identity))
        if cached is not None:
            _MATRIX_MEMO_STATS["captain_terms_hits"] += 1
            a_cached, c_cached = cached
            return dict(a_cached), dict(c_cached)

    _MATRIX_MEMO_STATS["captain_terms_computes"] += 1
    worlds = int(world_matrix["worlds"])
    player_ids = [int(pid) for pid in world_matrix["player_ids"]]
    manager_lineup.validate_world_matrix(world_matrix, context="captain_terms")
    minutes = world_matrix["minutes"]
    core = world_matrix["core"]
    appeared = {pid: [float(minutes[pid][w]) > 0.0 for w in range(worlds)] for pid in player_ids}
    core_series = {pid: [float(core[pid][w]) for w in range(worlds)] for pid in player_ids}
    a_terms: dict[int, float] = {}
    for pid in player_ids:
        a_terms[pid] = sum(
            core_series[pid][w] for w in range(worlds) if appeared[pid][w]
        ) / worlds
    c_terms: dict[int, dict[int, float]] = {}
    for vice in player_ids:
        inner: dict[int, float] = {}
        for captain in player_ids:
            if captain == vice:
                continue
            inner[captain] = sum(
                core_series[vice][w]
                for w in range(worlds)
                if appeared[vice][w] and not appeared[captain][w]
            ) / worlds
        c_terms[vice] = inner
    if identity is not None:
        _MATRIX_MEMO[("captain_terms", identity)] = (a_terms, c_terms)
        return dict(a_terms), dict(c_terms)
    return a_terms, c_terms
