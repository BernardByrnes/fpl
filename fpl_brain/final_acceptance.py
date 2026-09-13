"""Phase 8C — final end-to-end acceptance and high-fidelity shortlist confirmation.

Narrow acceptance pass: it takes the already-certified Phase-8B.1 optimizer result,
builds a compact finalist set, confirms GW4 at the certified 10,000-draw run-71
semantics, recomputes SUPPORTED_3GW, and audits end-to-end coherence.  It runs no
new search, adds no chip/price model, and makes no recommendation.
"""

from __future__ import annotations

import hashlib
from typing import Any, Iterable, Mapping, Sequence

from . import manager_lineup as ml
from . import route_comparator as rc
from . import route_optimizer as ro
from . import transfer_state as ts

PHASE8C_VERSION = "final_acceptance_v8c_1.0.0"
GW4_DRAWS = 10_000
GW5_DRAWS = 2_000
GW6_DRAWS = 2_000
CONFIRMATION_MATERIAL_CORE = 0.50  # pre-declared, unchanged
SUPPORTED_EVENTS = (4, 5, 6)
H4_H6_FLAG = ro.CANONICAL_UNSUPPORTED_FLAG
CROSS_GW_FLAG = rc.CROSS_GW_FLAG
PRICE_SCENARIO = "FLAT_CURRENT_PRICE"
PRICE_FLAG = "SCENARIO_ASSUMPTION_NOT_PRICE_FORECAST"
FRONTIER_NAME = "CONFIRMED_SHORTLIST_PARETO_FRONTIER"


# ---------------------------------------------------------------------------
# Finalist selection (from certified results only)
# ---------------------------------------------------------------------------


def _strategic_shape(actions: Sequence[Mapping[str, Any]], *, drop_events: Iterable[int] = ()) -> str:
    """Shape ignoring the listed events (used to collapse near-equivalent variants)."""

    parts = []
    for action in actions:
        if int(action["event"]) in set(drop_events):
            continue
        moves = sorted(f"{int(t['out'])}-{int(t['in'])}" for t in action.get("transfers") or [])
        parts.append(f"E{int(action['event'])}:{'|'.join(moves) if moves else 'ROLL'}")
    return ";".join(parts)


def select_finalists(certified_result: Mapping[str, Any], *, max_families: int = 12,
                     near_leader_core: float = CONFIRMATION_MATERIAL_CORE) -> dict[str, Any]:
    """Compact, materially distinct finalist families from certified routes only."""

    families: dict[str, dict[str, Any]] = {}
    for record in (certified_result.get("routes") or {}).values():
        signature = str(record.get("canonical_family_signature") or record.get("family_signature"))
        current = families.get(signature)
        if current is None or record["supported_3gw_net_core"] > current["supported_3gw_net_core"]:
            families[signature] = dict(record)
    if not families:
        raise ValueError("certified result contains no routes")

    roll_sig = next((sig for sig, rec in families.items()
                     if all(a["kind"] == "ROLL" for a in rec["actions"])), None)
    leader_3gw = max(sorted(families), key=lambda sig: families[sig]["supported_3gw_net_core"])
    leader_h1 = max(sorted(families), key=lambda sig: families[sig]["h1_net_core"])
    frontier_3gw = {str(r.get("canonical_family_signature") or r.get("family_signature"))
                    for name in (certified_result.get("supported_3gw_frontier") or [])
                    for r in [certified_result["routes"][name]] if name in certified_result["routes"]}
    frontier_h1 = {str(r.get("canonical_family_signature") or r.get("family_signature"))
                   for name in (certified_result.get("h1_frontier") or [])
                   for r in [certified_result["routes"][name]] if name in certified_result["routes"]}

    selected: dict[str, list[str]] = {}

    def add(signature: str | None, reason: str) -> None:
        if signature and signature in families:
            selected.setdefault(signature, []).append(reason)

    add(roll_sig, "ALL_ROLL_BASELINE")
    add(leader_3gw, "CERTIFIED_SUPPORTED_3GW_LEADER")
    add(leader_h1, "CERTIFIED_H1_LEADER")
    for signature in sorted(frontier_3gw):
        add(signature, "SUPPORTED_3GW_FRONTIER")
    for signature in sorted(families):
        value = families[signature]["supported_3gw_net_core"]
        if value >= families[leader_3gw]["supported_3gw_net_core"] - near_leader_core:
            add(signature, "WITHIN_NEAR_TIED_3GW")

    # H1 frontier: collapse near-equivalent variants by strategic shape and keep the
    # best H1 per shape within the declared distance of the H1 leader.
    best_by_shape: dict[str, str] = {}
    for signature in sorted(frontier_h1):
        shape = _strategic_shape(families[signature]["actions"], drop_events=(SUPPORTED_EVENTS[2],))
        current = best_by_shape.get(shape)
        if current is None or families[signature]["h1_net_core"] > families[current]["h1_net_core"]:
            best_by_shape[shape] = signature
    h1_leader_value = families[leader_h1]["h1_net_core"]
    for signature in sorted(best_by_shape.values(), key=lambda s: -families[s]["h1_net_core"]):
        if families[signature]["h1_net_core"] >= h1_leader_value - near_leader_core:
            add(signature, "H1_STRATEGIC_TRADEOFF")

    ordered = sorted(selected, key=lambda sig: (
        -families[sig]["supported_3gw_net_core"], -families[sig]["h1_net_core"], sig))
    if len(ordered) > max_families:
        must_keep = [sig for sig in ordered if "ALL_ROLL_BASELINE" in selected[sig]
                     or "CERTIFIED_SUPPORTED_3GW_LEADER" in selected[sig]
                     or "CERTIFIED_H1_LEADER" in selected[sig]]
        remaining = [sig for sig in ordered if sig not in must_keep]
        ordered = must_keep + remaining[: max(0, max_families - len(must_keep))]

    finalists = [{
        "canonical_signature": signature,
        "selection_reasons": sorted(set(selected[signature])),
        "actions": families[signature]["actions"],
        "recorded": {
            "h1_net_core": float(families[signature]["h1_net_core"]),
            "supported_3gw_net_core": float(families[signature]["supported_3gw_net_core"]),
            "cumulative_hits": int(families[signature]["cumulative_hits"]),
            "terminal_ft": int(families[signature]["terminal_ft"]),
            "terminal_bank_tenths": int(families[signature]["terminal_bank_tenths"]),
            "per_event": families[signature]["per_event"],
        },
    } for signature in ordered]
    return {
        "finalists": finalists,
        "count": len(finalists),
        "available_families": len(families),
        "rule": {
            "max_families": int(max_families), "near_leader_core": float(near_leader_core),
            "h1_cluster_events_dropped": [int(SUPPORTED_EVENTS[2])],
            "reasons": ["ALL_ROLL_BASELINE", "CERTIFIED_SUPPORTED_3GW_LEADER", "CERTIFIED_H1_LEADER",
                        "SUPPORTED_3GW_FRONTIER", "WITHIN_NEAR_TIED_3GW", "H1_STRATEGIC_TRADEOFF"],
        },
        "leader_signatures": {"supported_3gw": leader_3gw, "h1": leader_h1, "roll": roll_sig},
    }


# ---------------------------------------------------------------------------
# Phase-7A route replay + verification
# ---------------------------------------------------------------------------


def replay_route(*, initial_state: ts.RouteState, actions: Sequence[Mapping[str, Any]],
                 scenario: rc.PriceScenario, player_meta: Mapping[int, ts.PlayerMeta]) -> dict[str, Any]:
    """Replay the deterministic Phase-7A transitions and verify the recorded state."""

    state = initial_state
    events: list[dict[str, Any]] = []
    errors: list[str] = []
    for action in actions:
        event = int(action["event"])
        snapshot = scenario.snapshot_for(event)
        if snapshot is None:
            errors.append(f"NO_PRICE_SNAPSHOT_FOR_EVENT: {event}")
            break
        batch = ts.TransferBatch(tuple(ts.TransferAction(int(t["out"]), int(t["in"]))
                                       for t in action.get("transfers") or []))
        result = ts.apply_transfer_batch(state, batch, snapshot, player_meta)
        if not result.ok:
            errors.append(f"REPLAY_ILLEGAL: event {event}: {'; '.join(result.errors)}")
            break
        recorded_squad = tuple(sorted(int(pid) for pid in action.get("squad_ids") or ()))
        actual_squad = tuple(sorted(int(p.player_id) for p in result.squad_after.players))
        if recorded_squad and recorded_squad != actual_squad:
            errors.append(f"REPLAY_SQUAD_MISMATCH: event {event}")
        if int(action.get("hit_points", result.hit_points)) != int(result.hit_points):
            errors.append(f"REPLAY_HIT_MISMATCH: event {event}")
        if "bank_after" in action and int(action["bank_after"]) != int(result.bank_after_tenths):
            errors.append(f"REPLAY_BANK_MISMATCH: event {event}")
        if "ft_after" in action and int(action["ft_after"]) != int(result.next_event_state.free_transfers):
            errors.append(f"REPLAY_FT_MISMATCH: event {event}")
        purchase_basis = {int(p.player_id): int(p.purchase_price_tenths) for p in result.squad_after.players}
        events.append({
            "event": event, "kind": action["kind"], "hit_points": int(result.hit_points),
            "sale_proceeds_tenths": int(result.sale_proceeds_tenths),
            "purchase_cost_tenths": int(result.purchase_cost_tenths),
            "bank_after_tenths": int(result.bank_after_tenths),
            "ft_after": int(result.next_event_state.free_transfers),
            "squad_ids": list(actual_squad), "purchase_basis": purchase_basis,
        })
        state = result.next_event_state
    return {"events": events, "final_state": state, "errors": errors, "ok": not errors}


# ---------------------------------------------------------------------------
# GW4 high-fidelity confirmation
# ---------------------------------------------------------------------------


def gw4_policy_cache_key(event: int, squad_ids: Sequence[int], *, draws: int | None = None,
                         seed: int | None = None) -> str:
    """Phase-6 policy cache identity: event + post-transfer squad + world identity.

    Deliberately excludes route id, family signature and later-event (GW5/GW6)
    actions, which cannot affect GW4 manager scoring.
    """

    payload = "|".join([f"event={int(event)}",
                        f"draws={'' if draws is None else int(draws)}",
                        f"seed={'' if seed is None else int(seed)}",
                        ",".join(str(int(pid)) for pid in sorted(int(p) for p in squad_ids))])
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def confirm_gw4(*, finalist_gw4_squads: Mapping[str, Sequence[int]], worlds,
                positions_of, config, cache: dict) -> dict[str, Any]:
    """Reselect the Phase-6 policy at 10k for each unique GW4 squad, then score."""

    results: dict[str, dict[str, Any]] = {}
    for signature, squad_ids in finalist_gw4_squads.items():
        key = gw4_policy_cache_key(4, squad_ids, draws=int(worlds["worlds"]))
        entry = cache.get(key)
        if entry is None:
            positions = positions_of(squad_ids)
            selection = ro._subsample(worlds, int(config.policy_selection_worlds))
            ranked = ml.rank_policies(list(squad_ids), positions, selection, top_k=1)
            if not ranked["top_policies"]:
                raise ValueError(f"no legal manager policy for finalist {signature}")
            policy = ranked["top_policies"][0]
            scores = rc.policy_world_scores(policy, worlds, positions)
            metrics = ml.evaluate_policy(policy, worlds, positions)
            entry = {"policy": policy, "positions": positions, "scores": scores, "metrics": metrics}
            cache[key] = entry
        results[signature] = {
            "squad_hash": key,
            "policy": {
                "starter_ids": list(entry["policy"].starter_ids),
                "bench_gk_id": int(entry["policy"].bench_gk_id),
                "bench_outfield_order": list(entry["policy"].bench_outfield_order),
                "captain_id": int(entry["policy"].captain_id),
                "vice_captain_id": int(entry["policy"].vice_captain_id),
            },
            "mean_gross_core": float(entry["metrics"]["mean_core"]),
            "median_core": float(entry["metrics"]["median_core"]),
            "q10": float(entry["metrics"]["q10_core"]),
            "q25": float(entry["metrics"]["q25_core"]),
            "q75": float(entry["metrics"]["q75_core"]),
            "q90": float(entry["metrics"]["q90_core"]),
            "std_core": float(entry["metrics"]["std_core"]),
            "p_any_autosub": float(entry["metrics"]["p_any_autosub"]),
            "expected_autosub_points_added": float(entry["metrics"]["expected_autosub_points_added"]),
            "p_vice_takes_captaincy": float(entry["metrics"]["p_vice_takes_captaincy"]),
            "p_no_captain_multiplier": float(entry["metrics"]["p_no_captain_multiplier"]),
            "world_scores": entry["scores"],
            "scoring_basis": "CORE",
        }
    return results


def paired_vs_roll(confirmations: Mapping[str, Mapping[str, Any]], roll_signature: str) -> list[dict[str, Any]]:
    """Common-world paired comparison of each finalist's GW4 net scores vs ALL-ROLL."""

    roll = confirmations.get(roll_signature)
    if roll is None:
        return []
    out = []
    for signature, entry in sorted(confirmations.items()):
        if signature == roll_signature:
            continue
        diffs = [a - b for a, b in zip(entry["world_scores"], roll["world_scores"])]
        n = len(diffs)
        mean = sum(diffs) / n
        variance = sum((d - mean) ** 2 for d in diffs) / (n - 1) if n > 1 else 0.0
        se = (variance / n) ** 0.5 if n > 1 else 0.0
        out.append({
            "finalist": signature, "reference": roll_signature, "worlds": n,
            "mean_difference": mean, "paired_se": se,
            "ci95_low": mean - 1.96 * se, "ci95_high": mean + 1.96 * se,
            "p_finalist_gt_roll": sum(1 for d in diffs if d > 0) / n,
            "near_tied": (abs(mean) <= 1.96 * se) if se > 0 else mean == 0.0,
            "flags": [CROSS_GW_FLAG] if False else [],
        })
    return out


def paired_between(confirmations: Mapping[str, Mapping[str, Any]],
                   pairs: Sequence[Sequence[str]]) -> list[dict[str, Any]]:
    out = []
    for first, second in pairs:
        if first not in confirmations or second not in confirmations:
            continue
        diffs = [a - b for a, b in zip(confirmations[first]["world_scores"],
                                      confirmations[second]["world_scores"])]
        n = len(diffs)
        mean = sum(diffs) / n
        variance = sum((d - mean) ** 2 for d in diffs) / (n - 1) if n > 1 else 0.0
        se = (variance / n) ** 0.5 if n > 1 else 0.0
        out.append({"route_a": first, "route_b": second, "worlds": n, "mean_difference": mean,
                    "paired_se": se, "p_a_gt_b": sum(1 for d in diffs if d > 0) / n,
                    "near_tied": (abs(mean) <= 1.96 * se) if se > 0 else mean == 0.0})
    return out


def recompute_supported_3gw(gw4_net_core: float, gw5_net_core: float, gw6_net_core: float) -> float:
    """SUPPORTED_3GW = GW4 + GW5 + GW6 expected net CORE, undiscounted."""

    return float(gw4_net_core) + float(gw5_net_core) + float(gw6_net_core)


def confirmed_frontier(confirmed: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """CONFIRMED_SHORTLIST_PARETO_FRONTIER (finalist set only, not a global search)."""

    def vector(entry: Mapping[str, Any]) -> tuple:
        net = entry.get("confirmed_3gw_net_core", entry.get("supported_3gw_net_core"))
        return (float(net), -int(entry.get("cumulative_hits") or 0),
                int(entry.get("terminal_ft") or 0), int(entry.get("terminal_bank_tenths") or 0))

    items = [(sig, vector(entry)) for sig, entry in confirmed.items()]
    return sorted(
        sig for sig, vec in items
        if not any(all(a >= b for a, b in zip(other, vec)) and any(a > b for a, b in zip(other, vec))
                   for other_sig, other in items if other_sig != sig)
    )
