"""Phase 8B — bounded multi-Gameweek transfer route optimizer.

Turns the Phase-8A candidate universe into legal transfer routes over the
certified contiguous decision window (whatever ``OptimizerConfig.events`` is),
evaluates them through the Phase-7B comparator (shared football worlds, Phase-6
exact manager scoring) and returns an auditable Pareto set.  The window is
always supplied by the caller; there is no GW4 default and no event-number
hardcoding anywhere in this module.

BOUNDED and transparent only: never a black-box global optimizer, never a
recommendation or an execution, never a chip model.  Every pruning step that can
in principle remove the true optimum is explicitly labelled
``BOUNDED_SEARCH_PRUNING``; the cheap search proxy is labelled
``SEARCH_HEURISTIC_ONLY`` and is never reported as a route value.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from . import manager_lineup, route_comparator as rc, transfer_state as ts
from .season_rules import window_flag

PHASE8B_VERSION = "route_optimizer_v8b_1.0.0"
CACHE_SCHEMA_VERSION = "manager_worlds_cache_v1"

CANONICAL_UNSUPPORTED_FLAG = "CANONICAL_H4_H6_UNSUPPORTED"
CROSS_GW_FLAG = rc.CROSS_GW_FLAG
HEURISTIC_LABEL = "SEARCH_HEURISTIC_ONLY"
BOUNDED_PRUNING_LABEL = "BOUNDED_SEARCH_PRUNING"
BEST_WITHIN_SEARCH_LABEL = "BEST_ROUTE_FOUND_WITHIN_CONFIGURED_SEARCH"
HIGHER_HIT_FLAG = "HIGHER_HIT_ROUTES_OUTSIDE_AUTO_SEARCH"
SEARCH_STABLE = "SEARCH_BUDGET_STABLE"
SEARCH_UNSTABLE = "SEARCH_NOT_STABLE_AT_CURRENT_BUDGET"
#: Rescue lenses are named for what they rank, not for a Gameweek number: the
#: first decision event and the window sum respectively.
RESCUE_FIRST_EVENT = "FULL_UNIVERSE_SINGLE_RESCUE_FIRST_EVENT"
RESCUE_WINDOW_SUM = "FULL_UNIVERSE_SINGLE_RESCUE_WINDOW_SUM"
POSITIONS = ("GKP", "DEF", "MID", "FWD")

PARETO_LENS = "PARETO"
PER_EVENT_MARGINAL_LENS = "PER_EVENT_MARGINAL"
TOP_NET_PROXY_LENS = "TOP_NET_PROXY"

#: The declared retention lens set (R4B.2a).  Every lens selects from the FULL
#: generated state set G; the union is then fairly capped.  REQUIRED routes are
#: NOT a lens: they are unioned AFTER the cap so they can never be displaced
#: (see ``_retain(required=...)``).
DEFAULT_RETENTION_LENSES: tuple = (
    TOP_NET_PROXY_LENS, "TOP_H1", "LOW_HITS", "BANK_DELTA", "FT_PRESERVATION",
    PARETO_LENS, PER_EVENT_MARGINAL_LENS,
)
REQUIRED_PASS_THROUGH = "REQUIRED"

#: Lens ordering is FIXED and declared: it determines the round-robin order of
#: the fair cap, so it must never depend on set/dict iteration.
RETENTION_LENS_ORDER: tuple = DEFAULT_RETENTION_LENSES

_KNOWN_RETENTION_LENSES = RETENTION_LENS_ORDER + (REQUIRED_PASS_THROUGH,)


@dataclass(frozen=True)
class OptimizerConfig:
    #: The decision window is REQUIRED.  There is deliberately no GW4 default:
    #: a silent default would let a GW5+ run score a GW4 window.
    events: tuple = ()
    search_draws: int = 2000
    seed: int = 20260911
    beam_width: int = 12
    exact_evaluation_budget: int = 30
    policy_selection_worlds: int = 300
    max_auto_hit_points_per_event: int = 4
    singles_per_out: int = 6
    max_transfers_per_event: int = 3
    rescue_top_k_per_position: int = 25
    search_n_per_criterion: int = 20
    retention_lenses: tuple = DEFAULT_RETENTION_LENSES

    def as_dict(self) -> dict[str, Any]:
        labels = [HEURISTIC_LABEL, BOUNDED_PRUNING_LABEL, BEST_WITHIN_SEARCH_LABEL, HIGHER_HIT_FLAG,
                  CANONICAL_UNSUPPORTED_FLAG, CROSS_GW_FLAG]
        if self.events:
            labels.insert(5, window_flag(self.events))
        return {
            "events": list(self.events), "search_draws": int(self.search_draws), "seed": int(self.seed),
            "beam_width": int(self.beam_width), "exact_evaluation_budget": int(self.exact_evaluation_budget),
            "policy_selection_worlds": int(self.policy_selection_worlds),
            "max_auto_hit_points_per_event": int(self.max_auto_hit_points_per_event),
            "singles_per_out": int(self.singles_per_out),
            "max_transfers_per_event": int(self.max_transfers_per_event),
            "rescue_top_k_per_position": int(self.rescue_top_k_per_position),
            "search_n_per_criterion": int(self.search_n_per_criterion),
            "retention_lenses": list(self.retention_lenses),
            "labels": labels,
        }


# ---------------------------------------------------------------------------
# Candidate pool + rescue
# ---------------------------------------------------------------------------


def first_event_core(row: Mapping[str, Any]) -> float:
    """Expected CORE of the FIRST decision event of this row's window.

    Positional by construction: ``build_universe`` builds ``row["events"]`` in
    the order of the decision events, so index 0 is the window's first event for
    ANY window (GW4-GW7, GW5-GW8, GW7-GW10, ...).  Looking the event up by
    NUMBER instead silently returns 0.0 whenever the window does not contain
    that number, which collapses the criterion to a price/id tiebreak.
    """

    features = row.get("events") or ()
    if not features:
        return 0.0
    return _feature_core(features[0])


def search_view_ids(rows: Mapping[int, Mapping[str, Any]], n: int) -> set[int]:
    """Reproduce the Phase-8A search view (same five criteria) at any N.

    Mirrors ``candidate_universe._fill_search_view`` exactly, including its
    positional first-event criterion, so both implementations of the view agree
    for every window.
    """

    view: set[int] = set()
    for position in POSITIONS:
        group = [row for row in rows.values() if row["position"] == position]
        if not group:
            continue
        rankings = (
            sorted(group, key=lambda r: (-first_event_core(r), r["current_market_price_tenths"], r["player_id"])),
            sorted(group, key=lambda r: (-r["supported_3gw_expected_core"], r["current_market_price_tenths"], r["player_id"])),
            sorted(group, key=lambda r: (-r["supported_3gw_expected_minutes"], r["current_market_price_tenths"], r["player_id"])),
            sorted(group, key=lambda r: (r["current_market_price_tenths"], r["player_id"])),
            sorted(group, key=lambda r: (-(r["descriptive_value"] if r["descriptive_value"] is not None else float("-inf")), r["player_id"])),
        )
        for ranked in rankings:
            for row in ranked[: int(n)]:
                view.add(int(row["player_id"]))
    return view


def build_search_pool(universe: Mapping[str, Any], owned_ids: Iterable[int],
                      config: OptimizerConfig) -> dict[str, Any]:
    rows = {int(row["player_id"]): row for row in universe["universe"]}
    owned = {int(pid) for pid in owned_ids}
    pool = set(search_view_ids(rows, int(config.search_n_per_criterion))) | owned
    base_view = set(pool)

    by_position: dict[str, list[tuple[float, float, int]]] = {position: [] for position in POSITIONS}
    for edge in universe.get("replacement_edges") or []:
        if not edge["currently_legal_single_transfer"]:
            continue
        out_row, in_row = rows.get(int(edge["out_player_id"])), rows.get(int(edge["in_player_id"]))
        if out_row is None or in_row is None:
            continue
        delta_gw4 = _feature_core(in_row["events"][0]) - _feature_core(out_row["events"][0])
        delta_3gw = float(in_row["supported_3gw_expected_core"]) - float(out_row["supported_3gw_expected_core"])
        by_position[in_row["position"]].append((delta_gw4, delta_3gw, int(in_row["player_id"])))

    rescue_reasons: dict[int, list[str]] = {}
    k = int(config.rescue_top_k_per_position)
    for entries in by_position.values():
        for _d4, _d3, pid in sorted(entries, key=lambda item: (-item[0], item[2]))[:k]:
            rescue_reasons.setdefault(pid, []).append(RESCUE_FIRST_EVENT)
        for _d4, _d3, pid in sorted(entries, key=lambda item: (-item[1], item[2]))[:k]:
            rescue_reasons.setdefault(pid, []).append(RESCUE_WINDOW_SUM)
    for pid, reasons in rescue_reasons.items():
        pool.add(pid)
        row = rows[pid]
        for reason in sorted(set(reasons)):
            if reason not in row["inclusion_reasons"]:
                row["inclusion_reasons"].append(reason)
        row["inclusion_reasons"].sort()

    return {
        "pool_ids": sorted(pool),
        "base_view_ids": sorted(base_view),
        "rescued_ids": sorted(rescue_reasons),
        "rescued_outside_view": sorted(pid for pid in rescue_reasons if pid not in base_view),
        "rescue_reasons": {str(pid): sorted(set(reasons)) for pid, reasons in sorted(rescue_reasons.items())},
        "pool_hash": hashlib.sha256(",".join(str(pid) for pid in sorted(pool)).encode()).hexdigest()[:16],
    }


# ---------------------------------------------------------------------------
# Action generation (Phase-7A legality only)
# ---------------------------------------------------------------------------


def _feature_core(feature) -> float:
    if isinstance(feature, Mapping):
        return float(feature.get("expected_core") or 0.0)
    return float(feature.expected_core)


def _delta(rows, out_id, in_id) -> tuple[float, float]:
    out_row, in_row = rows[out_id], rows[in_id]
    return (_feature_core(in_row["events"][0]) - _feature_core(out_row["events"][0]),
            float(in_row["supported_3gw_expected_core"]) - float(out_row["supported_3gw_expected_core"]))


def _action(event, kind, batch, transition, hit, delta3):
    return {"event": int(event), "kind": kind, "batch": batch, "transition": transition,
            "hit_points": int(hit), "delta_3gw": float(delta3)}


def generate_actions(*, state, rows, pool_ids, positions, price_snapshot, player_meta, config, event):
    owned = [int(p.player_id) for p in state.players]
    owned_set = set(owned)
    pool = [pid for pid in pool_ids if pid not in owned_set]

    singles = []
    generated = 0
    for out_id in owned:
        candidates = [pid for pid in pool if positions.get(pid) == positions.get(out_id)]
        ranked = sorted(candidates, key=lambda pid: (-_delta(rows, out_id, pid)[1], -_delta(rows, out_id, pid)[0], pid))
        for in_id in ranked[: int(config.singles_per_out)]:
            generated += 1
            batch = ts.TransferBatch((ts.TransferAction(out_id, in_id),))
            result = ts.apply_transfer_batch(state, batch, price_snapshot, player_meta)
            if result.ok:
                d4, d3 = _delta(rows, out_id, in_id)
                singles.append((d3, d4, batch, result))
    singles.sort(key=lambda item: (-item[0], -item[1], item[2].out_ids(), item[2].in_ids()))

    actions = [_action(event, "ROLL", ts.TransferBatch.roll(), None, 0, 0.0)]
    for d3, d4, batch, result in singles:
        actions.append(_action(event, "SINGLE", batch, result, result.hit_points, d3))

    ft_before = int(state.free_transfers)
    max_paid = int(config.max_auto_hit_points_per_event) // ts.HIT_POINTS_PER_EXTRA_TRANSFER
    ceiling = min(int(config.max_transfers_per_event), max(1, ft_before) + max_paid)

    doubles = triples = 0
    if ceiling >= 2:
        top = singles[: max(1, int(config.singles_per_out)) * 12]
        for i in range(len(top)):
            for j in range(i + 1, len(top)):
                outs = top[i][2].out_ids() + top[j][2].out_ids()
                ins = top[i][2].in_ids() + top[j][2].in_ids()
                if len(set(outs)) != 2 or len(set(ins)) != 2:
                    continue
                doubles += 1
                batch = ts.TransferBatch(top[i][2].actions + top[j][2].actions)
                result = ts.apply_transfer_batch(state, batch, price_snapshot, player_meta)
                if result.ok:
                    actions.append(_action(event, "DOUBLE", batch, result, result.hit_points,
                                           top[i][0] + top[j][0]))
    if ceiling >= 3:
        top = singles[: max(1, int(config.singles_per_out)) * 6]
        for i in range(len(top)):
            for j in range(i + 1, len(top)):
                for k in range(j + 1, len(top)):
                    outs = top[i][2].out_ids() + top[j][2].out_ids() + top[k][2].out_ids()
                    ins = top[i][2].in_ids() + top[j][2].in_ids() + top[k][2].in_ids()
                    if len(set(outs)) != 3 or len(set(ins)) != 3:
                        continue
                    triples += 1
                    batch = ts.TransferBatch(top[i][2].actions + top[j][2].actions + top[k][2].actions)
                    result = ts.apply_transfer_batch(state, batch, price_snapshot, player_meta)
                    if result.ok:
                        actions.append(_action(event, "HIT", batch, result, result.hit_points,
                                               top[i][0] + top[j][0] + top[k][0]))
    return {
        "actions": actions,
        "ceiling": ceiling,
        "generated": {"singles_attempted": generated, "singles_legal": len(singles),
                      "doubles_attempted": doubles, "triples_attempted": triples},
        "counts": {kind: sum(1 for a in actions if a["kind"] == kind) for kind in ("ROLL", "SINGLE", "DOUBLE", "HIT")},
    }


# ---------------------------------------------------------------------------
# Proxy / dedupe / retention
# ---------------------------------------------------------------------------


def player_event_core(row: Mapping[str, Any], event: int) -> float:
    for feature in row["events"]:
        value = feature.get("event") if isinstance(feature, Mapping) else feature.event
        if int(value) == int(event):
            return _feature_core(feature)
    return 0.0


def window_proxy(rows, squad_ids, events) -> float:
    return sum(player_event_core(rows[pid], event) for pid in squad_ids for event in events if pid in rows)


def state_key(state: ts.RouteState) -> tuple:
    return (
        int(state.event),
        tuple(sorted((int(p.player_id), int(p.purchase_price_tenths)) for p in state.players)),
        int(state.bank_tenths), int(state.free_transfers),
        tuple(sorted((str(c.get("name")), int(c.get("number") or 0)) for c in state.chip_state)),
    )


@dataclass
class PartialRoute:
    state: ts.RouteState
    actions: tuple
    h1_proxy: float
    window_proxy: float
    hits: int
    #: Per-event single-event proxy of the squad carried from each event, aligned
    #: with the decision events.  Used ONLY by the PER_EVENT_MARGINAL coverage
    #: lens so a route that is weak early but strong later can be selected on a
    #: remaining-window suffix.  It never feeds the objective.
    event_proxies: tuple = ()

    def signature(self) -> tuple:
        return tuple((a["kind"], a["batch"].out_ids(), a["batch"].in_ids(), a["event"]) for a in self.actions)

    def net_proxy(self) -> float:
        return self.window_proxy - self.hits

    def suffix_proxy(self, start_index: int) -> float:
        """Sum of per-event proxies from ``start_index`` to the end of the window."""

        return float(sum(self.event_proxies[start_index:])) if self.event_proxies else 0.0


def _pareto(items: Sequence[Any], dims: Callable[[Any], tuple]) -> list[Any]:
    vectors = [(item, dims(item)) for item in items]
    frontier = []
    for item, vector in vectors:
        if not any(
            all(a >= b for a, b in zip(other, vector)) and any(a > b for a, b in zip(other, vector))
            for candidate, other in vectors if candidate is not item
        ):
            frontier.append(item)
    return frontier


#: Retention lenses are REAL configuration: ``_retain`` iterates
#: ``config.retention_lenses`` and applies each named lens.  An unknown name is
#: an error rather than a silent no-op, so a recorded config can never describe
#: a policy that was not applied.
def retention_budgets(config: OptimizerConfig) -> dict[str, int]:
    """Declared lens budgets, derived from ``beam_width`` only.

    * ``KN`` — primary net-proxy budget; identical to the pre-R4B.2a prefilter
      size, so today's primary search behaviour is preserved exactly.
    * ``KD`` — small-k diversity budget for each rescue lens.
    * ``K_MAX`` — cap on the NON-required retained set.  Required/inherited
      routes are unioned after this cap and do not compete for slots.
    """

    width = int(config.beam_width)
    primary = max(4 * width, 24)
    return {"KN": primary, "KD": max(4, width // 2), "K_MAX": 2 * primary}


def _total_tiebreak(s: PartialRoute) -> tuple:
    """Total deterministic ordering; signature last so input order never matters.

    Signs follow the module convention: ``net_proxy``/``h1_proxy``/bank/FT are
    higher-is-better and are negated for an ascending sort; ``hits`` is
    lower-is-better and stays positive.
    """

    return (-s.net_proxy(), -s.h1_proxy, s.hits, -s.state.bank_tenths,
            -s.state.free_transfers, s.signature())


def _primary_key(s: PartialRoute) -> tuple:
    """The primary football ordering, plus a deterministic final tie-break.

    The first three components are the pre-R4B.2a primary ordering and are
    UNCHANGED: ``-net_proxy``, ``-h1_proxy``, ``hits``.  ``signature()`` is
    appended ONLY to break exact ties deterministically, so the primary top set
    never depends on input order.  Objective values are not affected: this key
    orders routes, it does not score them.
    """

    return (-s.net_proxy(), -s.h1_proxy, s.hits, s.signature())


def _per_event_marginal(states: list[PartialRoute], events: Sequence[int], k: int) -> list[PartialRoute]:
    """Small-k selection on each remaining-window SUFFIX of the decision window.

    For ``[e0, e1, e2, e3]`` the suffixes are ``e1:e3``, ``e2:e3`` and ``e3``, so
    a route that is weak early but strong later can be retained without touching
    the four-GW objective.  Suffix start 0 (the whole window) is deliberately
    excluded: that is the primary lens.
    """

    window = [int(event) for event in events]
    picks: list[PartialRoute] = []
    if not window:
        return picks
    for start in range(1, len(window)):
        picks.extend(sorted(
            states,
            key=lambda s, start=start: (-s.suffix_proxy(start), -s.net_proxy(), s.signature()),
        )[: int(k)])
    return picks


def _lens_picks(
    states: list[PartialRoute],
    lens: str,
    *,
    events: Sequence[int],
    initial_bank_tenths: int,
    kn: int,
    kd: int,
) -> list[PartialRoute]:
    """Ordered candidates one lens selects from the FULL state set G."""

    if lens == PARETO_LENS:
        # Frontier over G, never a prefiltered subset.  Boundedness is the final
        # cap's job, so the frontier itself is not truncated.
        return _pareto(states, lambda s: (s.net_proxy(), -s.hits, s.state.bank_tenths,
                                          s.state.free_transfers))
    if lens == PER_EVENT_MARGINAL_LENS:
        return _per_event_marginal(states, events, kd)
    if lens == TOP_NET_PROXY_LENS:
        return sorted(states, key=_primary_key)[:kn]
    if lens == "TOP_H1":
        return sorted(states, key=lambda s: (-s.h1_proxy, -s.net_proxy(), s.hits, s.signature()))[:kd]
    if lens == "LOW_HITS":
        return sorted(states, key=lambda s: (s.hits, -s.net_proxy(), s.signature()))[:kd]
    if lens == "BANK_DELTA":
        # BANK_DELTA = terminal bank - initial bank: pure future purchasing power.
        return sorted(states, key=lambda s: (
            -(s.state.bank_tenths - int(initial_bank_tenths)), -s.net_proxy(), s.signature()))[:kd]
    if lens == "FT_PRESERVATION":
        return sorted(states, key=lambda s: (
            -s.state.free_transfers, s.hits, -s.state.bank_tenths, s.signature()))[:kd]
    raise ValueError(
        f"unknown retention lens {lens!r}; known lenses are {list(_KNOWN_RETENTION_LENSES)}"
    )


def _retain(
    states: list[PartialRoute],
    config: OptimizerConfig,
    *,
    events: Sequence[int] = (),
    initial_bank_tenths: int = 0,
    required: Sequence[PartialRoute] = (),
    coverage: dict[str, Any] | None = None,
) -> list[PartialRoute]:
    """Bounded FULL-SET retention lenses; heuristic and explicitly labelled.

    Shape (the R4B.2a full-set lens principle)::

        G -> each lens independently selects from G -> fair union -> cap -> + required

    There is deliberately NO primary top-K prefilter gating the other lenses: a
    route removed before the lenses ran could never be rescued by the final cap.
    The primary net-proxy ordering is one lens among several, not a gate.

    The union is capped at ``K_MAX`` by a deterministic round-robin over the
    declared lens lists, so no single lens can consume the budget and the same
    state is counted once.  REQUIRED routes are unioned AFTER the cap, hence
    ``len(retained) <= K_MAX + len(required)``.
    """

    states = list(states)
    budgets = retention_budgets(config)
    kn, kd, k_max = budgets["KN"], budgets["KD"], budgets["K_MAX"]

    lenses = tuple(config.retention_lenses)
    unknown = [lens for lens in lenses if lens not in _KNOWN_RETENTION_LENSES]
    if unknown:
        raise ValueError(
            f"unknown retention lens {unknown!r}; known lenses are {list(_KNOWN_RETENTION_LENSES)}"
        )

    # The primary set, computed once, used only to identify RESCUED routes.
    primary = sorted(states, key=_primary_key)[:kn]
    primary_keys = {state_key(s.state) for s in primary}

    per_lens: dict[str, list[PartialRoute]] = {}
    lens_stats: dict[str, dict[str, int]] = {}
    for lens in lenses:
        picks = _lens_picks(states, lens, events=events, initial_bank_tenths=initial_bank_tenths,
                            kn=kn, kd=kd)
        per_lens[lens] = picks
        lens_stats[lens] = {"considered": len(states), "selected": len(picks), "unique_added": 0}

    # Fair deterministic round-robin: one candidate per lens per pass, in the
    # declared lens order, until the non-required cap is reached.
    retained: dict[tuple, PartialRoute] = {}
    cursor = {lens: 0 for lens in lenses}
    progress = True
    while progress and len(retained) < k_max:
        progress = False
        for lens in lenses:
            picks = per_lens[lens]
            index = cursor[lens]
            if index >= len(picks):
                continue
            progress = True
            cursor[lens] = index + 1
            candidate = picks[index]
            key = state_key(candidate.state)
            if key in retained:
                continue
            retained[key] = candidate
            lens_stats[lens]["unique_added"] += 1
            if len(retained) >= k_max:
                break

    non_required_cap = min(len(retained), k_max)

    # REQUIRED pass-through: unioned AFTER the cap so it can never be displaced.
    required_added = 0
    for route in required:
        key = state_key(route.state)
        if key not in retained:
            retained[key] = route
            required_added += 1

    final = sorted(retained.values(), key=_total_tiebreak)

    if coverage is not None:
        lens_member_keys = {
            lens: {state_key(s.state) for s in per_lens[lens]} for lens in lenses
        }
        coverage.update({
            "states_considered": len(states),
            "states_retained": len(final),
            "non_required_cap": non_required_cap,
            "required_count": len(required),
            "required_added": required_added,
            "heuristic_pruned": max(0, len(states) - non_required_cap),
            "budgets": budgets,
            "lenses": lens_stats,
            "primary_net_proxy": {
                "considered": len(states),
                "selected": len(primary),
                "unique_added": sum(1 for key in primary_keys if key in retained),
                "note": "one lens among several; it is not a gate around the others",
            },
            "rescued_routes": sorted(
                (
                    {
                        "signature": str(route.signature()),
                        "rescued_by": sorted(
                            lens for lens in lenses
                            if state_key(route.state) in lens_member_keys[lens]
                        ) or [REQUIRED_PASS_THROUGH],
                        "primary_rank": next(
                            (rank for rank, s in enumerate(primary, start=1)
                             if state_key(s.state) == state_key(route.state)), None),
                        "retained": True,
                    }
                    for route in final
                    if state_key(route.state) not in primary_keys
                ),
                key=lambda item: item["signature"],
            ),
        })
    return final


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def _event_proxy_parts(rows, squad_ids, events, level_index, inherited):
    """Per-event proxy parts for the PER_EVENT_MARGINAL lens.

    Index ``i`` holds the squad carried from event ``i`` valued at event ``i``,
    so ``suffix_proxy(start)`` is a genuine remaining-window suffix.  Events
    before ``level_index`` are inherited from the parent partial route.
    """

    window = [int(event) for event in events]
    parts = [window_proxy(rows, squad_ids, [event]) for event in window[level_index:]]
    if len(inherited) >= level_index:
        return tuple(inherited[:level_index]) + tuple(parts)
    return tuple(parts)


def run_search(*, initial_state, events, rows, pool_ids, positions, scenario, player_meta, config,
               nested_prior_levels=None):
    events = [int(e) for e in events]
    # Per-event proxy parts are computed ONLY when the lens that consumes them is
    # configured, so the default cost of the other lenses is unchanged.
    needs_event_parts = PER_EVENT_MARGINAL_LENS in tuple(config.retention_lenses)
    initial_bank_tenths = int(initial_state.bank_tenths)
    states = [PartialRoute(state=initial_state, actions=(), h1_proxy=0.0, window_proxy=0.0, hits=0)]
    stats = {"levels": [], "partial_states": 1, "safe_dedup": 0, "heuristic_retained": 0,
             "heuristic_pruned": 0, "nested_inherited": 0, "invalid": {}}
    coverage: dict[int, dict[str, Any]] = {}
    coverage_levels: list[dict[str, Any]] = []
    level_survivors: list[list[PartialRoute]] = []
    nested_prior_levels = list(nested_prior_levels or [])
    roll_partial = states[0]
    for level_index, event in enumerate(events):
        by_key: dict[tuple, PartialRoute] = {}
        action_stats = {"generated": 0, "legal": 0, "kinds": {}, "generated_kinds": {}}
        for partial in states:
            snapshot = scenario.snapshot_for(event)
            if snapshot is None:
                continue
            action_set = generate_actions(state=partial.state, rows=rows, pool_ids=pool_ids,
                                          positions=positions, price_snapshot=snapshot,
                                          player_meta=player_meta, config=config, event=event)
            action_stats["generated"] += action_set["generated"]["singles_attempted"] + 1
            for kind, count in action_set["counts"].items():
                action_stats["kinds"][kind] = action_stats["kinds"].get(kind, 0) + count
            for name, value in action_set["generated"].items():
                action_stats["generated_kinds"][name] = action_stats["generated_kinds"].get(name, 0) + value
            for action in action_set["actions"]:
                transition = action["transition"]
                if transition is not None and not transition.ok:
                    action_stats["invalid"] = action_stats.get("invalid", 0) + 1
                    stats["invalid"][";".join(transition.errors)] = stats["invalid"].get(
                        ";".join(transition.errors), 0) + 1
                    continue
                new_state = (transition.next_event_state if transition is not None
                             else ts.apply_transfer_batch(partial.state, action["batch"], snapshot, player_meta).next_event_state)
                action_stats["legal"] += 1
                squad_ids = [int(p.player_id) for p in new_state.players]
                action["squad_ids"] = tuple(sorted(squad_ids))
                action["ft_after"] = int(new_state.free_transfers)
                action["bank_after"] = int(new_state.bank_tenths)
                remaining = [e for e in events if e >= event]
                candidate = PartialRoute(
                    state=new_state, actions=partial.actions + (action,),
                    h1_proxy=window_proxy(rows, squad_ids, [events[0]]),
                    window_proxy=window_proxy(rows, squad_ids, remaining),
                    hits=partial.hits + int(action["hit_points"]),
                    event_proxies=(_event_proxy_parts(rows, squad_ids, events, level_index,
                                                      partial.event_proxies)
                                   if needs_event_parts else ()),
                )
                key = state_key(new_state)
                existing = by_key.get(key)
                if existing is None:
                    by_key[key] = candidate
                else:
                    stats["safe_dedup"] += 1
                    if (candidate.net_proxy(), candidate.signature()) > (existing.net_proxy(), existing.signature()):
                        by_key[key] = candidate
        generated = list(by_key.values())
        stats["partial_states"] += len(generated)
        level_coverage: dict[str, Any] = {}
        retained = _retain(generated, config, events=events,
                           initial_bank_tenths=initial_bank_tenths,
                           coverage=level_coverage)
        coverage_levels.append({"event": int(event), **level_coverage})
        # ROLL must never be pruned by a heuristic: keep its exact chain alive.
        roll_snapshot = scenario.snapshot_for(event)
        if roll_snapshot is not None:
            roll_transition = ts.apply_transfer_batch(roll_partial.state, ts.TransferBatch.roll(),
                                                      roll_snapshot, player_meta)
            roll_state = roll_transition.next_event_state
            roll_squad = [int(p.player_id) for p in roll_state.players]
            remaining = [e for e in events if e >= event]
            roll_partial = PartialRoute(
                state=roll_state, actions=roll_partial.actions + ({"event": int(event), "kind": "ROLL",
                    "batch": ts.TransferBatch.roll(), "transition": roll_transition, "hit_points": 0,
                    "delta_3gw": 0.0, "squad_ids": tuple(sorted(roll_squad)),
                    "ft_after": int(roll_state.free_transfers), "bank_after": int(roll_state.bank_tenths)},),
                h1_proxy=window_proxy(rows, roll_squad, [events[0]]),
                window_proxy=window_proxy(rows, roll_squad, remaining), hits=roll_partial.hits)
        if roll_partial.state.event == int(event) + 1 and not any(
                all(a["kind"] == "ROLL" for a in state.actions) for state in retained):
            retained = list(retained) + [roll_partial]
        own_retained = len(retained)
        # NESTED-BUDGET MODE: an expanded budget inherits every survivor of the
        # immediately smaller-budget run at the same depth, and only removes one
        # via SAFE identical-state dedupe (never via a heuristic lens).
        inherited = 0
        if level_index < len(nested_prior_levels):
            merged = {state_key(item.state): item for item in retained}
            for prior_state in nested_prior_levels[level_index]:
                key = state_key(prior_state.state)
                current = merged.get(key)
                if current is None:
                    merged[key] = prior_state
                    inherited += 1
                elif (prior_state.net_proxy(), prior_state.signature()) > (current.net_proxy(), current.signature()):
                    merged[key] = prior_state
            retained = list(merged.values())
        level_survivors.append(retained)
        stats["heuristic_retained"] += own_retained
        stats["heuristic_pruned"] += max(0, len(generated) - own_retained)
        stats["nested_inherited"] = stats.get("nested_inherited", 0) + inherited
        coverage[int(event)] = {
            "generated_kinds": action_stats["generated_kinds"], "legal_kinds": action_stats["kinds"],
            "generated": action_stats["generated"], "legal": action_stats["legal"],
            "invalid": action_stats.get("invalid", 0),
            "states_generated": len(generated), "states_retained": len(retained),
            "own_retained": own_retained, "nested_inherited": inherited,
        }
        stats["levels"].append({"event": int(event), "states_generated": len(generated),
                                "retained": len(retained), "own_retained": own_retained,
                                "nested_inherited": inherited})
        states = retained
    return {"final_states": states, "stats": stats, "coverage": coverage,
            "level_survivors": level_survivors, "coverage_levels": coverage_levels}


# ---------------------------------------------------------------------------
# World generation + optional downstream cache
# ---------------------------------------------------------------------------


def route_player_ids(partial: PartialRoute) -> set[int]:
    """Every player id a route touches: its squad plus both sides of each move.

    Used to widen the world capture union so a forced (required/inherited) route
    can never be exact-evaluated against a world matrix that omits one of its
    players.
    """

    ids = {int(player.player_id) for player in partial.state.players}
    for action in partial.actions:
        for move in action["batch"].actions:
            ids.add(int(move.out_player_id))
            ids.add(int(move.in_player_id))
    return ids


def union_player_ids(initial_state: ts.RouteState, pool_ids: Sequence[int]) -> list[int]:
    return sorted({int(p.player_id) for p in initial_state.players} | {int(pid) for pid in pool_ids})


def world_cache_key(*, event, bundle, config, union_ids) -> str:
    """Cache identity for a world matrix.

    Includes every input that would make two matrices semantically different:
    the exact certified run ids, the simulation count, the seed, the union, and
    the AUTHORITATIVE Monte Carlo model identity.  A presentation-only label is
    deliberately not part of the key.
    """

    from . import monte_carlo

    payload = {
        "schema": CACHE_SCHEMA_VERSION, "event": int(event),
        "minutes": int(bundle.minutes_run_id), "team": int(bundle.team_run_id),
        "rate": int(bundle.rate_run_id), "xpts": int(bundle.xpts_run_id),
        "mc_model": str(monte_carlo.MONTE_CARLO_MODEL_VERSION),
        "simulations": int(config.search_draws), "seed": int(config.seed),
        "union": hashlib.sha256(",".join(map(str, sorted(int(u) for u in union_ids))).encode()).hexdigest()[:16],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def build_event_worlds(conn, bundles, event, union_ids, config, *, cache_dir: Path | None = None,
                       world_provider=None):
    if world_provider is not None:
        return world_provider(event, union_ids), {"source": "injected"}
    from . import monte_carlo

    bundle = bundles[int(event)]
    key = world_cache_key(event=int(event), bundle=bundle, config=config, union_ids=union_ids)
    if cache_dir is not None:
        path = Path(cache_dir) / f"{key}.json"
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            return {"worlds": raw["worlds"], "player_ids": raw["player_ids"],
                    "core": {int(k): v for k, v in raw["core"].items()},
                    "minutes": {int(k): v for k, v in raw["minutes"].items()}}, {"source": "cache", "key": key}
    fixtures = monte_carlo.load_fixture_inputs(
        conn, event=int(event), xpts_run_id=int(bundle.xpts_run_id),
        minutes_run_id=int(bundle.minutes_run_id), team_run_id=int(bundle.team_run_id),
    )
    mc_config = monte_carlo.MonteCarloConfig(simulations=int(config.search_draws), seed=int(config.seed),
                                             occupancy_audit=True)
    result = monte_carlo.simulate(fixtures, mc_config, capture_player_ids=list(union_ids))
    matrix = result["world_matrix"]
    if cache_dir is not None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        (Path(cache_dir) / f"{key}.json").write_text(json.dumps({
            "worlds": matrix["worlds"], "player_ids": matrix["player_ids"],
            "core": {str(k): v for k, v in matrix["core"].items()},
            "minutes": {str(k): v for k, v in matrix["minutes"].items()},
        }), encoding="utf-8")
    return matrix, {"source": "generated", "key": key}


# ---------------------------------------------------------------------------
# Exact evaluation (Phase-6)
# ---------------------------------------------------------------------------


def _subsample(matrix, limit: int):
    worlds = int(matrix["worlds"])
    if limit <= 0 or limit >= worlds:
        return matrix
    step = max(1, worlds // limit)
    indices = list(range(0, worlds, step))[:limit]
    return {"worlds": len(indices), "player_ids": list(matrix["player_ids"]),
            "core": {pid: [matrix["core"][pid][i] for i in indices] for pid in matrix["player_ids"]},
            "minutes": {pid: [matrix["minutes"][pid][i] for i in indices] for pid in matrix["player_ids"]}}


def route_event_squads(partial: PartialRoute) -> list[tuple[int, tuple[int, ...]]]:
    return [(int(a["event"]), tuple(int(pid) for pid in a["squad_ids"])) for a in partial.actions]


def exact_evaluate(partial: PartialRoute, *, worlds_by_event, positions_of, cache, config,
                   events) -> dict[str, Any]:
    per_event = []
    event_scores: dict[int, list[float]] = {}
    for event, squad_ids in route_event_squads(partial):
        key = (int(event), hashlib.sha256(",".join(map(str, squad_ids)).encode()).hexdigest()[:16],
               int(config.search_draws), int(config.seed))
        entry = cache.get(key)
        if entry is None:
            positions = positions_of(squad_ids)
            selection = _subsample(worlds_by_event[event], int(config.policy_selection_worlds))
            ranked = manager_lineup.rank_policies(list(squad_ids), positions, selection, top_k=1)
            if not ranked["top_policies"]:
                raise ValueError(f"no legal manager policy for squad at event {event}")
            policy = ranked["top_policies"][0]
            scores = rc.policy_world_scores(policy, worlds_by_event[event], positions)
            entry = {"policy": policy, "scores": scores, "mean_gross": sum(scores) / len(scores)}
            cache[key] = entry
        action = next(a for a in partial.actions if int(a["event"]) == int(event))
        hit = int(action["hit_points"])
        event_scores[int(event)] = entry["scores"]
        per_event.append({
            "event": int(event), "kind": action["kind"], "hit_points": hit,
            "mean_gross_core": float(entry["mean_gross"]), "mean_net_core": float(entry["mean_gross"]) - hit,
            "policy": {
                "starter_ids": list(entry["policy"].starter_ids), "bench_gk_id": int(entry["policy"].bench_gk_id),
                "bench_outfield_order": list(entry["policy"].bench_outfield_order),
                "captain_id": int(entry["policy"].captain_id), "vice_captain_id": int(entry["policy"].vice_captain_id),
            },
        })
    gross = sum(item["mean_gross_core"] for item in per_event)
    hits = sum(item["hit_points"] for item in per_event)
    return {"per_event": per_event, "event_scores": event_scores, "gross_core": gross,
            "cumulative_hits": hits, "net_core": gross - hits}


# ---------------------------------------------------------------------------
# Families, near ties
# ---------------------------------------------------------------------------


def family_signature(partial: PartialRoute) -> tuple:
    return tuple(
        tuple(sorted((int(a["batch"].actions[i].out_player_id), int(a["batch"].actions[i].in_player_id))
                     for i in range(len(a["batch"].actions))))
        for a in partial.actions
    )


def canonical_family_signature(partial: PartialRoute) -> str:
    """Canonical, route-id-free family identity: ordered per-event action sets.

    Example: ``E4:165-249|329-305;E5:ROLL;E6:334-449``.  Transfer sets are
    deterministically canonicalised (sorted ``out-in`` pairs), so a change in
    ephemeral ``route_###`` numbering can never change a stability verdict.
    """

    parts = []
    for action in partial.actions:
        transfers = sorted(f"{int(x.out_player_id)}-{int(x.in_player_id)}" for x in action["batch"].actions)
        body = "|".join(transfers) if transfers else "ROLL"
        parts.append(f"E{int(action['event'])}:{body}")
    return ";".join(parts)


def canonical_signature_from_actions(actions: Sequence[Mapping[str, Any]]) -> str:
    """Same identity computed from serialized route actions."""

    parts = []
    transfers = []
    current_event = None
    for action in actions:
        event = int(action["event"])
        moves = sorted(f"{int(t['out'])}-{int(t['in'])}" for t in action.get("transfers") or [])
        parts.append(f"E{event}:{'|'.join(moves) if moves else 'ROLL'}")
    return ";".join(parts)


def paired_diagnostics(entries: Sequence[tuple[str, Mapping[int, Sequence[float]]]], key: int,
                       *, first_event: int | None = None,
                       k: float = rc.NEAR_TIE_K) -> list[dict[str, Any]]:
    """Paired per-event route differences.

    ``first_event`` is the window's first decision event.  A comparison at any
    other event is flagged as a cross-Gameweek comparison; when
    ``first_event`` is omitted the flag is not applied rather than being
    computed against a hard-coded Gameweek.
    """

    results = []
    for i in range(len(entries)):
        for j in range(i + 1, len(entries)):
            name_a, scores_a = entries[i]
            name_b, scores_b = entries[j]
            if key not in scores_a or key not in scores_b:
                continue
            diffs = [a - b for a, b in zip(scores_a[key], scores_b[key])]
            n = len(diffs)
            mean = sum(diffs) / n
            variance = sum((d - mean) ** 2 for d in diffs) / (n - 1) if n > 1 else 0.0
            se = (variance / n) ** 0.5 if n > 1 else 0.0
            cross_gw = first_event is not None and int(key) != int(first_event)
            results.append({
                "route_a": name_a, "route_b": name_b, "worlds": n, "mean_difference": mean, "paired_se": se,
                "p_a_gt_b": sum(1 for d in diffs if d > 0) / n,
                "near_tied": (abs(mean) <= k * se) if se > 0 else mean == 0.0,
                "flags": [CROSS_GW_FLAG] if cross_gw else [],
            })
    return results


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

MATERIAL_FRONTIER_CHANGE_CORE = 0.25  # defined before seeing any result


def _serialize_route(partial: PartialRoute) -> list[dict[str, Any]]:
    return [
        {"event": int(a["event"]), "kind": a["kind"], "hit_points": int(a["hit_points"]),
         "transfers": [{"out": int(x.out_player_id), "in": int(x.in_player_id)} for x in a["batch"].actions],
         "squad_ids": list(a["squad_ids"]), "ft_after": int(a["ft_after"]), "bank_after": int(a["bank_after"])}
        for a in partial.actions
    ]


def _frontier_names(records: Mapping[str, Mapping[str, Any]], dims) -> list[str]:
    items = [{"id": name, "vec": dims(record)} for name, record in records.items()]
    return sorted(
        item["id"] for item in items
        if not any(all(a >= b for a, b in zip(other["vec"], item["vec"]))
                   and any(a > b for a, b in zip(other["vec"], item["vec"]))
                   for other in items if other["id"] != item["id"])
    )


def _select_promoted(final_states: Sequence[PartialRoute], config: OptimizerConfig, *,
                     required: Sequence[PartialRoute] = (),
                     inherited: Sequence[PartialRoute] = ()) -> list[PartialRoute]:
    """Promote by window proxy and H1 proxy, family-diverse; ROLL and any
    explicitly required / inherited-smaller-budget routes are always included."""

    ordered = sorted(final_states, key=lambda s: (-s.net_proxy(), -s.h1_proxy, s.hits, s.signature()))
    h1_ordered = sorted(final_states, key=lambda s: (-s.h1_proxy, -s.net_proxy(), s.hits, s.signature()))
    budget = int(config.exact_evaluation_budget)
    seen_families: set = set()
    promoted: list[PartialRoute] = []

    def push(state: PartialRoute) -> bool:
        signature = family_signature(state)
        if signature in seen_families:
            return False
        promoted.append(state)
        seen_families.add(signature)
        return True

    # 1) always include all-ROLL; 2) required (prior leaders/frontiers);
    # 3) inherited smaller-budget promoted routes; 4) own proxy promotion.
    roll = next((s for s in ordered if all(a["kind"] == "ROLL" for a in s.actions)), None)
    if roll is not None:
        push(roll)
    for state in required:
        push(state)
    for state in inherited:
        push(state)
    for source in (ordered, h1_ordered):
        for state in source:
            if len(promoted) >= budget + len(required) + len(inherited):
                break
            push(state)
    return promoted


def optimize(
    *,
    universe: Mapping[str, Any],
    initial_state: ts.RouteState,
    scenario: rc.PriceScenario,
    player_meta: Mapping[int, ts.PlayerMeta],
    bundles: Mapping[int, Any] | None = None,
    conn=None,
    config: OptimizerConfig | None = None,
    cache_dir: Path | None = None,
    world_provider=None,
    search_n_override: int | None = None,
    exact_cache: dict | None = None,
    prebuilt_worlds: Mapping[int, Any] | None = None,
    nested_prior: Mapping[str, Any] | None = None,
    required_routes: Sequence[PartialRoute] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bounded search, exact scoring, Pareto frontiers and coverage/rescue audits.

    ``nested_prior`` carries the immediately smaller-budget run's per-level
    survivors and promoted routes so this run is a true nested superset (subject
    only to safe identical-state dedupe).  ``required_routes`` forces routes
    (prior leaders, prior frontier families, ROLL) into exact evaluation.

    ``provenance`` supplies the decision's causal identity (planning cutoff,
    planning-context hash, certification identity, data snapshot, decision
    events) so the search artifact can name the deterministic inputs it used
    instead of recording ``None``.
    """

    import time

    config = config or OptimizerConfig()
    if search_n_override is not None:
        config = _with_n(config, int(search_n_override))
    events = [int(e) for e in config.events]
    if not events:
        raise ValueError(
            "OptimizerConfig.events is empty: a decision window must be supplied explicitly "
            "(there is no GW4 default)"
        )
    rows = {int(row["player_id"]): row for row in universe["universe"]}
    owned = [int(p.player_id) for p in initial_state.players]
    positions_by_id = {pid: str(row["position"]) for pid, row in rows.items()}

    started = time.time()
    pool = build_search_pool(universe, owned, config)
    pool_construction_s = time.time() - started
    # Every route that will be exact-evaluated must be inside the world capture
    # union, otherwise its players would be absent from the world matrix.
    required = tuple(required_routes or ())
    inherited = tuple((nested_prior or {}).get("promoted") or ())
    forced_ids: set[int] = set()
    for partial in required + inherited:
        forced_ids |= route_player_ids(partial)
    union = union_player_ids(initial_state, sorted(set(pool["pool_ids"]) | forced_ids))

    world_info: dict[str, Any] = {}
    worlds_by_event: dict[int, Mapping[str, Any]] = {}
    for event in events:
        if prebuilt_worlds is not None and event in prebuilt_worlds:
            matrix, info = prebuilt_worlds[event], {"source": "prebuilt"}
        else:
            matrix, info = build_event_worlds(conn, bundles, event, union, config,
                                              cache_dir=cache_dir, world_provider=world_provider)
        worlds_by_event[event] = matrix
        world_info[str(event)] = {**info, "worlds": int(matrix["worlds"]), "union_players": len(union)}

    search = run_search(initial_state=initial_state, events=events, rows=rows,
                        pool_ids=pool["pool_ids"], positions=positions_by_id, scenario=scenario,
                        player_meta=player_meta, config=config,
                        nested_prior_levels=(nested_prior or {}).get("level_survivors"))

    exact_cache = exact_cache if exact_cache is not None else {}

    def positions_of(squad_ids):
        return {int(pid): positions_by_id[int(pid)] for pid in squad_ids}

    promoted = _select_promoted(
        search["final_states"], config,
        required=required,
        inherited=inherited,
    )

    records: dict[str, dict[str, Any]] = {}
    exact_evaluations = 0
    for index, partial in enumerate(promoted):
        exact = exact_evaluate(partial, worlds_by_event=worlds_by_event, positions_of=positions_of,
                               cache=exact_cache, config=config, events=events)
        exact_evaluations += len(partial.actions)
        name = f"route_{index:03d}"
        h1_event = exact["per_event"][0]
        records[name] = {
            "route_id": name,
            "family_signature": family_signature(partial),
            "canonical_family_signature": canonical_family_signature(partial),
            "actions": _serialize_route(partial),
            "per_event": exact["per_event"],
            "event_scores": exact["event_scores"],
            "h1_net_core": float(h1_event["mean_net_core"]),
            "h1_hits": int(h1_event["hit_points"]),
            "h1_ft": int(partial.actions[0]["ft_after"]),
            "h1_bank": int(partial.actions[0]["bank_after"]),
            "supported_3gw_net_core": float(exact["net_core"]),
            "supported_3gw_gross_core": float(exact["gross_core"]),
            "cumulative_hits": int(exact["cumulative_hits"]),
            "terminal_ft": int(partial.state.free_transfers),
            "terminal_bank_tenths": int(partial.state.bank_tenths),
            "valid": True,
            "price_scenario_status": str(scenario.scenario_id),
            "uses_rescued_player": _uses_rescue(partial, pool["rescued_ids"]),
            "proxy_window": float(partial.window_proxy),
            "proxy_h1": float(partial.h1_proxy),
            "proxy_label": HEURISTIC_LABEL,
        }

    h1_frontier = _frontier_names(
        records, lambda r: (r["h1_net_core"], -r["h1_hits"], r["h1_ft"], r["h1_bank"]))
    window_frontier = _frontier_names(
        records, lambda r: (r["supported_3gw_net_core"], -r["cumulative_hits"], r["terminal_ft"],
                            r["terminal_bank_tenths"]))

    families: dict[Any, dict[str, Any]] = {}
    for name, record in records.items():
        signature = record["family_signature"]
        best = families.get(signature)
        if best is None or record["supported_3gw_net_core"] > best["supported_3gw_net_core"]:
            families[signature] = {
                "representative_route_id": name, "family_signature": signature,
                "supported_3gw_net_core": record["supported_3gw_net_core"],
                "h1_net_core": record["h1_net_core"], "cumulative_hits": record["cumulative_hits"],
                "terminal_ft": record["terminal_ft"], "terminal_bank_tenths": record["terminal_bank_tenths"],
                "actions": record["actions"], "member_route_ids": [],
            }
        families[signature]["member_route_ids"].append(name)

    h1_pairs = paired_diagnostics([(name, rec["event_scores"]) for name, rec in records.items()], events[0],
                                  first_event=events[0])
    window_pairs = _window_paired(records, events)

    roll_name = next(
        (name for name, rec in records.items() if all(a["kind"] == "ROLL" for a in rec["actions"])), None)
    roll_baseline = None if roll_name is None else {
        "route_id": roll_name, "h1_net_core": records[roll_name]["h1_net_core"],
        "supported_3gw_net_core": records[roll_name]["supported_3gw_net_core"],
        "cumulative_hits": records[roll_name]["cumulative_hits"],
        "terminal_ft": records[roll_name]["terminal_ft"],
        "terminal_bank_tenths": records[roll_name]["terminal_bank_tenths"],
        "per_event": records[roll_name]["per_event"],
    }

    rescued_users = {name for name, rec in records.items() if rec["uses_rescued_player"]}
    rescue_audit = {
        "rescued_count": len(pool["rescued_ids"]),
        "rescued_outside_search_view": len(pool["rescued_outside_view"]),
        "rescued_outside_search_view_ids": pool["rescued_outside_view"],
        "routes_using_rescued": sorted(rescued_users),
        "rescued_in_top20_h1": sorted(
            name for name in sorted(records, key=lambda n: -records[n]["h1_net_core"])[:20] if name in rescued_users),
        "rescued_in_top20_3gw": sorted(
            name for name in sorted(records, key=lambda n: -records[n]["supported_3gw_net_core"])[:20]
            if name in rescued_users),
        "rescued_on_frontier": sorted(set(h1_frontier + window_frontier) & rescued_users),
    }

    prov = dict(provenance or {})
    return {
        "phase_version": PHASE8B_VERSION,
        "planning_context_hash": prov.get("planning_context_hash"),
        "planning_cutoff": prov.get("planning_cutoff"),
        "four_gw_certification_identity": prov.get("four_gw_certification_identity"),
        "data_snapshot_sha256": prov.get("data_snapshot_sha256"),
        "certified_runs_by_event": prov.get("certified_runs_by_event"),
        "supported_events": events,
        "decision_horizon_length": len(events),
        "flags": [window_flag(events), CANONICAL_UNSUPPORTED_FLAG, HEURISTIC_LABEL, BOUNDED_PRUNING_LABEL,
                  BEST_WITHIN_SEARCH_LABEL, HIGHER_HIT_FLAG, CROSS_GW_FLAG],
        "score_basis": "CORE",
        "price_scenario_id": scenario.scenario_id,
        "price_scenario_flags": list(scenario.flags),
        "full_universe_count": len(rows),
        "base_search_view_count": len(pool["base_view_ids"]),
        "search_pool_count": len(pool["pool_ids"]),
        "rescue": rescue_audit,
        "search_coverage": _search_coverage_summary(
            search.get("coverage_levels") or [], config, search["stats"]
        ),
        "config": config.as_dict(),
        "search_stats": search["stats"],
        "coverage": {str(event): value for event, value in search["coverage"].items()},
        "world_info": world_info,
        "union_players": union,
        "exact_evaluations": exact_evaluations,
        "exact_cache_entries": len(exact_cache),
        "promoted_route_count": len(records),
        "routes": records,
        "h1_frontier": h1_frontier,
        "supported_3gw_frontier": window_frontier,
        "families": [
            {**family, "family_signature": str(family["family_signature"])}
            for family in sorted(families.values(), key=lambda f: -f["supported_3gw_net_core"])
        ],
        "paired_h1": h1_pairs,
        "paired_supported_3gw": window_pairs,
        "roll_baseline": roll_baseline,
        "timing_s": {"pool_construction": round(pool_construction_s, 3),
                     "total": round(time.time() - started, 3)},
        "level_survivors": search["level_survivors"],
        "promoted_routes": promoted,
        "nested_inherited": int(search["stats"].get("nested_inherited", 0)),
        "no_recommendation": True,
    }


def _search_coverage_summary(
    coverage_levels: Sequence[Mapping[str, Any]],
    config: OptimizerConfig,
    stats: Mapping[str, Any],
) -> dict[str, Any]:
    """Aggregate per-level lens coverage into a truthful search-coverage block.

    ``rescued`` means exactly this: the route was NOT in the primary TOP_NET_PROXY
    set but survived because of a declared lens.  It is a COVERAGE statement, not
    a quality claim — a rescued route is not asserted to be better.
    """

    lens_names = list(RETENTION_LENS_ORDER)
    lenses: dict[str, dict[str, int]] = {
        name: {"considered": 0, "selected": 0, "unique_added": 0} for name in lens_names
    }
    lenses[REQUIRED_PASS_THROUGH] = {"considered": 0, "selected": 0, "unique_added": 0}
    totals = {"states_generated": 0, "states_retained": 0, "heuristic_pruned": 0,
              "non_required_retained_total": 0, "required_count": 0}
    levels: list[dict[str, Any]] = []
    rescued: list[dict[str, Any]] = []
    for level in coverage_levels:
        totals["states_generated"] += int(level.get("states_considered") or 0)
        totals["states_retained"] += int(level.get("states_retained") or 0)
        totals["heuristic_pruned"] += int(level.get("heuristic_pruned") or 0)
        totals["non_required_retained_total"] += int(level.get("non_required_cap") or 0)
        totals["required_count"] += int(level.get("required_count") or 0)
        levels.append({
            "event": int(level.get("event") or 0),
            "states_generated": int(level.get("states_considered") or 0),
            "states_retained": int(level.get("states_retained") or 0),
            "heuristic_pruned": int(level.get("heuristic_pruned") or 0),
            # The PER-LEVEL cap actually enforced at this level.
            "non_required_cap": int(level.get("non_required_cap") or 0),
            "k_max_per_level": int(level.get("budgets", {}).get("K_MAX") or 0),
            "required_count": int(level.get("required_count") or 0),
            "required_added": int(level.get("required_added") or 0),
        })
        for name, values in (level.get("lenses") or {}).items():
            if name in lenses:
                for key in ("considered", "selected", "unique_added"):
                    lenses[name][key] += int(values.get(key) or 0)
        # NOTE: the primary lens is already part of ``level["lenses"]`` (it is a
        # lens like any other), so its counters are summed above like the rest.
        # ``primary_net_proxy`` is reported separately as documentation; it must
        # NOT overwrite the summed lens counters.
        for item in level.get("rescued_routes") or []:
            rescued.append({"event": int(level.get("event") or 0), **item})

    budgets = retention_budgets(config)
    k_max_per_level = budgets["K_MAX"]
    return {
        "global_optimality_claimed": False,
        "coverage_is_not_quality": True,
        "counts_are_aggregated_across_levels": True,
        "states_generated": totals["states_generated"],
        "states_retained": totals["states_retained"],
        "heuristic_pruned": totals["heuristic_pruned"],
        # Renamed from ``non_required_cap``: this is the SUM over levels, not a
        # single cap.  The cap enforced at any one level is ``k_max_per_level``.
        "non_required_retained_total": totals["non_required_retained_total"],
        "k_max_per_level": k_max_per_level,
        "required_count": totals["required_count"],
        "k_max": k_max_per_level,
        "k_n": budgets["KN"],
        "k_d": budgets["KD"],
        "lens_order": list(RETENTION_LENS_ORDER),
        "cap_policy": "deterministic round-robin across lens lists in lens_order; duplicates counted once; applied PER LEVEL",
        "levels": levels,
        "lenses": lenses,
        "rescued_routes": rescued,
        "state_count_retained_final": int(stats.get("heuristic_retained") or 0),
    }


def _with_n(config: OptimizerConfig, n: int) -> OptimizerConfig:
    return OptimizerConfig(
        events=config.events, search_draws=config.search_draws, seed=config.seed,
        beam_width=config.beam_width, exact_evaluation_budget=config.exact_evaluation_budget,
        policy_selection_worlds=config.policy_selection_worlds,
        max_auto_hit_points_per_event=config.max_auto_hit_points_per_event,
        singles_per_out=config.singles_per_out, max_transfers_per_event=config.max_transfers_per_event,
        rescue_top_k_per_position=config.rescue_top_k_per_position, search_n_per_criterion=int(n),
        retention_lenses=config.retention_lenses,
    )


def _uses_rescue(partial: PartialRoute, rescued_ids: Sequence[int]) -> bool:
    rescued = {int(pid) for pid in rescued_ids}
    for action in partial.actions:
        if any(int(x.in_player_id) in rescued for x in action["batch"].actions):
            return True
    return False


def _window_paired(records: Mapping[str, Mapping[str, Any]], events: Sequence[int]) -> list[dict[str, Any]]:
    names = sorted(records)
    results = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            scores_a = records[a]["event_scores"]
            scores_b = records[b]["event_scores"]
            if any(e not in scores_a or e not in scores_b for e in events):
                continue
            first = scores_a[events[0]]
            diffs = [sum(scores_a[e][w] - scores_b[e][w] for e in events) for w in range(len(first))]
            n = len(diffs)
            mean = sum(diffs) / n
            variance = sum((d - mean) ** 2 for d in diffs) / (n - 1) if n > 1 else 0.0
            se = (variance / n) ** 0.5 if n > 1 else 0.0
            results.append({"route_a": a, "route_b": b, "worlds": n, "mean_difference": mean, "paired_se": se,
                            "p_a_gt_b": sum(1 for d in diffs if d > 0) / n,
                            "near_tied": (abs(mean) <= rc.NEAR_TIE_K * se) if se > 0 else mean == 0.0,
                            "flags": [CROSS_GW_FLAG]})
    return results


TRANSIENT_KEYS = ("level_survivors", "promoted_routes")


def strip_transient(result: Mapping[str, Any]) -> dict[str, Any]:
    """Drop in-process-only fields (live PartialRoute objects) before JSON."""

    return {key: value for key, value in result.items() if key not in TRANSIENT_KEYS}


def _record_signature(record: Mapping[str, Any]) -> str:
    """Canonical, route-id-free family identity of a route record."""

    return str(record.get("canonical_family_signature") or record.get("family_signature"))


def route_family_scores(result: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """canonical signature -> exact scores (best representative per family)."""

    families: dict[str, dict[str, Any]] = {}
    for record in (result.get("routes") or {}).values():
        signature = _record_signature(record)
        value = float(record.get("supported_3gw_net_core") or 0.0)
        current = families.get(signature)
        if current is None or value > current["supported_3gw_net_core"]:
            families[signature] = {
                "h1_net_core": float(record.get("h1_net_core") or 0.0),
                "supported_3gw_net_core": value,
                "cumulative_hits": int(record.get("cumulative_hits") or 0),
                "terminal_ft": int(record.get("terminal_ft") or 0),
                "terminal_bank_tenths": int(record.get("terminal_bank_tenths") or 0),
                "route_id": record.get("route_id"),
            }
    return families


def best_route_family(result: Mapping[str, Any], objective: str = "supported_3gw_net_core") -> dict[str, Any] | None:
    families = route_family_scores(result)
    if not families:
        return None
    signature = max(sorted(families), key=lambda key: families[key][objective])
    return {"signature": signature, **families[signature]}


def frontier_family_signatures(result: Mapping[str, Any]) -> list[str]:
    lookup = {
        str(record.get("route_id", key)): _record_signature(record)
        for key, record in (result.get("routes") or {}).items()
    }
    names = set(result.get("h1_frontier") or []) | set(result.get("supported_3gw_frontier") or [])
    return sorted({lookup[name] for name in names if name in lookup})


def required_routes_from(result: Mapping[str, Any]) -> list[Any]:
    """Prior leaders + prior frontier families (+ ROLL) for exact re-evaluation."""

    promoted = list(result.get("promoted_routes") or [])
    by_signature = {canonical_family_signature(item): item for item in promoted}
    wanted: list[Any] = []
    seen: set[str] = set()

    def add(signature: str):
        if signature and signature in by_signature and signature not in seen:
            wanted.append(by_signature[signature])
            seen.add(signature)

    for objective in ("h1_net_core", "supported_3gw_net_core"):
        best = best_route_family(result, objective)
        if best:
            add(best["signature"])
    for signature in frontier_family_signatures(result):
        add(signature)
    for item in promoted:
        if all(a["kind"] == "ROLL" for a in item.actions):
            add(canonical_family_signature(item))
    return wanted


def nested_budget_view(result: Mapping[str, Any]) -> dict[str, Any]:
    """Carry-forward bundle for the next (larger) budget run."""

    return {"level_survivors": list(result.get("level_survivors") or []),
            "promoted": list(result.get("promoted_routes") or [])}


def monotonic_check(smaller: Mapping[str, Any], larger: Mapping[str, Any],
                    *, tolerance: float = 1e-9) -> dict[str, Any]:
    """best exact objective must not decrease as the nested budget grows."""

    report = {}
    ok = True
    for objective in ("h1_net_core", "supported_3gw_net_core"):
        small = best_route_family(smaller, objective) or {}
        large = best_route_family(larger, objective) or {}
        small_value = float(small.get(objective, 0.0))
        large_value = float(large.get(objective, 0.0))
        violated = large_value < small_value - tolerance
        ok = ok and not violated
        report[objective] = {
            "smaller_best": small_value, "larger_best": large_value,
            "smaller_signature": small.get("signature"), "larger_signature": large.get("signature"),
            "violated": violated,
        }
    report["tolerance"] = tolerance
    report["status"] = "PASS" if ok else "FAIL"
    return report


def cross_budget_score_identity(first: Mapping[str, Any], second: Mapping[str, Any],
                                *, tolerance: float = 1e-9) -> dict[str, Any]:
    """A family present in both runs must have identical exact scores."""

    a = route_family_scores(first)
    b = route_family_scores(second)
    shared = sorted(set(a) & set(b))
    worst = 0.0
    mismatches = []
    for signature in shared:
        for field in ("h1_net_core", "supported_3gw_net_core"):
            delta = abs(a[signature][field] - b[signature][field])
            worst = max(worst, delta)
            if delta > tolerance:
                mismatches.append({"signature": signature, "field": field, "delta": delta})
        for field in ("cumulative_hits", "terminal_ft", "terminal_bank_tenths"):
            if a[signature][field] != b[signature][field]:
                mismatches.append({"signature": signature, "field": field,
                                   "delta": abs(a[signature][field] - b[signature][field])})
    return {"shared_families": len(shared), "max_score_discrepancy": worst,
            "mismatches": mismatches, "status": "PASS" if not mismatches else "FAIL"}


def search_stability(base: Mapping[str, Any], expanded: Mapping[str, Any]) -> dict[str, Any]:
    """Compare two optimizer results by CANONICAL family signature (never route ids)."""

    def best_family(result):
        best = best_route_family(result, "supported_3gw_net_core")
        return best["signature"] if best else None

    def best_value(result):
        best = best_route_family(result, "supported_3gw_net_core")
        return float(best["supported_3gw_net_core"]) if best else 0.0

    leader_changed = best_family(base) != best_family(expanded)
    frontier_changed = frontier_family_signatures(base) != frontier_family_signatures(expanded)
    value_delta = abs(best_value(base) - best_value(expanded))
    material = leader_changed or frontier_changed or value_delta > MATERIAL_FRONTIER_CHANGE_CORE
    return {
        "best_3gw_family_base": best_family(base), "best_3gw_family_expanded": best_family(expanded),
        "frontier_families_base": frontier_family_signatures(base),
        "frontier_families_expanded": frontier_family_signatures(expanded),
        "best_3gw_value_base": best_value(base), "best_3gw_value_expanded": best_value(expanded),
        "value_delta": value_delta, "material_threshold_core": MATERIAL_FRONTIER_CHANGE_CORE,
        "leader_changed": leader_changed, "frontier_changed": frontier_changed,
        "status": SEARCH_UNSTABLE if material else SEARCH_STABLE,
    }
