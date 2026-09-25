"""Phase 7B — multi-Gameweek transfer route comparator.

Compares EXPLICIT transfer routes supplied by the caller.  It does not generate
candidates, does not optimize, and makes no recommendation.

Non-negotiable properties:

* the transfer accounting engine is Phase-7A's ``apply_transfer_batch`` only;
* football worlds are generated ONCE per event and shared by every route, from
  the union of route-relevant players (route identity never enters football RNG);
* manager policy (XI/bench/captain/vice) is selected EX ANTE per (event, squad)
  from that event's predictive world distribution and then applied unchanged;
* future transfers require an explicit ``PriceSnapshot``/``PriceScenario``;
* cumulative expected means are undiscounted and valid; multi-GW tails carry
  ``CROSS_GW_AVAILABILITY_PERSISTENCE_UNMODELLED``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

from . import certified_bundle, manager_lineup, monte_carlo, transfer_state as ts
from .season_rules import CHIP_NAME_KEYWORDS

PHASE7B_VERSION = "route_comparator_v7b_1.0.0"

HORIZON_STEPS = {"H1": 1, "H4": 4, "H6": 6}
CROSS_GW_FLAG = "CROSS_GW_AVAILABILITY_PERSISTENCE_UNMODELLED"
FLAT_PRICE_ASSUMPTION = "SCENARIO_ASSUMPTION_NOT_PRICE_FORECAST"
NEAR_TIE_K = 1.96
CHIP_ROUTE_FLAG = "CHIP_ROUTE_NOT_MODELLED"
#: Recognised chip spellings, from the canonical list in ``season_rules``.
CHIP_KEYWORDS = CHIP_NAME_KEYWORDS


# ---------------------------------------------------------------------------
# Structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RouteStep:
    event: int
    transfer_batch: ts.TransferBatch
    requested_chip: str | None = None


@dataclass(frozen=True)
class TransferRoute:
    route_id: str
    steps: tuple[RouteStep, ...]
    label: str | None = None

    def step_for(self, event: int) -> RouteStep | None:
        for step in self.steps:
            if int(step.event) == int(event):
                return step
        return None


@dataclass(frozen=True)
class PriceScenario:
    """Explicit per-event snapshots.  No price model, no forecasting."""

    scenario_id: str
    event_snapshots: Mapping[int, ts.PriceSnapshot]
    flags: tuple[str, ...] = ()

    def snapshot_for(self, event: int) -> ts.PriceSnapshot | None:
        return self.event_snapshots.get(int(event))


def flat_current_price_scenario(base: ts.PriceSnapshot, events: Iterable[int]) -> PriceScenario:
    """Engineering validation only: copy the current prices forward unchanged.

    Each event snapshot is stamped with its canonical price identity so
    ``PriceSnapshot.identity()`` is O(1) instead of re-serialising the whole price
    map for every candidate transfer batch.  The stamped string is exactly the value
    ``identity()`` already produced for that content, so every recorded
    ``price_snapshot_id`` is unchanged.
    """

    snapshots = {
        int(event): ts.PriceSnapshot(
            event=int(event), prices=dict(base.prices),
            snapshot_id=ts.price_snapshot_identity(int(event), dict(base.prices)),
        )
        for event in events
    }
    return PriceScenario(
        scenario_id="FLAT_CURRENT_PRICE", event_snapshots=snapshots, flags=(FLAT_PRICE_ASSUMPTION,)
    )


@dataclass(frozen=True)
class EventBundle:
    event: int
    minutes_run_id: int
    team_run_id: int
    rate_run_id: int
    xpts_run_id: int
    mc_run_id: int | None = None
    simulations: int = 2000
    seed: int = 20260911
    planning_cutoff: str | None = None
    #: The DATA snapshot identity this bundle was built from (the immutable
    #: execution snapshot captured at ``planning_cutoff``), never a code fingerprint.
    source_snapshot_sha256: str | None = None
    #: The CODE identity the certified runs came from.
    code_snapshot_sha256: str | None = None
    planning_context_hash: str | None = None
    #: The model version each family's run carries, as DECLARED by the producer.
    #: ``certified_bundle.assert_event_bundle_certified`` checks every one of them
    #: against the version recorded on the run's own row.
    model_versions: Mapping[str, str] = field(default_factory=dict)
    #: The canonical identity of the CERTIFIED bundle these exact run ids came from.
    #: A bundle without it cannot be loaded: see
    #: ``certified_bundle.assert_event_bundle_certified``.
    certified_bundle_identity: str | None = None

    def certified_runs(self) -> dict[str, int]:
        """The bundle's exact run ids under their model-family names."""

        return {
            "minutes_v1": int(self.minutes_run_id),
            "team_strength_v1": int(self.team_run_id),
            "player_rates_v1": int(self.rate_run_id),
            "xpts_v1": int(self.xpts_run_id),
            **(
                {}
                if self.mc_run_id is None
                else {"monte_carlo_v1": int(self.mc_run_id)}
            ),
        }

    def as_identity_payload(self) -> dict:
        """The canonical identity payload of this bundle, in its persisted shape."""

        from . import certified_bundle as cb

        return cb.bundle_identity_payload(
            event=int(self.event),
            cutoff=self.planning_cutoff,
            runs=self.certified_runs(),
            model_versions=self.model_versions,
            code_snapshot_sha256=self.code_snapshot_sha256,
            data_snapshot_sha256=self.source_snapshot_sha256,
            planning_context_hash=self.planning_context_hash,
        )


def certified_event_bundle(
    *,
    event: int,
    runs: Mapping[str, int],
    cutoff: str,
    model_versions: Mapping[str, str],
    simulations: int = 2000,
    seed: int = 20260911,
    code_snapshot_sha256: str | None = None,
    data_snapshot_sha256: str | None = None,
    planning_context_hash: str | None = None,
) -> EventBundle:
    """Build an ``EventBundle`` that DECLARES its certified provenance.

    The ONE constructor a producer may use to hand exact certified run ids to a
    predictive loader: the identity is computed by the single shared algorithm, and
    the declared model versions are the ones the certification recorded.  There is
    deliberately no convenience default that would let a bundle reach the loader
    without them.
    """

    from . import certified_bundle as cb

    resolved_versions = {str(family): str(version) for family, version in model_versions.items()}
    runs = {str(family): int(run_id) for family, run_id in runs.items()}
    return EventBundle(
        event=int(event),
        minutes_run_id=runs["minutes_v1"],
        team_run_id=runs["team_strength_v1"],
        rate_run_id=runs["player_rates_v1"],
        xpts_run_id=runs["xpts_v1"],
        mc_run_id=runs.get("monte_carlo_v1"),
        simulations=int(simulations),
        seed=int(seed),
        planning_cutoff=str(cutoff),
        source_snapshot_sha256=data_snapshot_sha256,
        code_snapshot_sha256=code_snapshot_sha256,
        planning_context_hash=planning_context_hash,
        model_versions=resolved_versions,
        certified_bundle_identity=cb.certified_bundle_identity_for(
            event=int(event),
            cutoff=str(cutoff),
            runs=runs,
            model_versions=resolved_versions,
            code_snapshot_sha256=code_snapshot_sha256,
            data_snapshot_sha256=data_snapshot_sha256,
            planning_context_hash=planning_context_hash,
        ),
    )


# ---------------------------------------------------------------------------
# Route spec parsing / validation
# ---------------------------------------------------------------------------


class RouteSpecError(ValueError):
    pass


def parse_route_spec(spec: Mapping[str, Any]) -> TransferRoute:
    """Parse a JSON route spec; player identity is ALWAYS the integer id."""

    route_id = spec.get("route_id")
    if not route_id or not isinstance(route_id, str):
        raise RouteSpecError("route spec requires a string route_id")
    steps = []
    for raw in spec.get("steps") or []:
        if "event" not in raw:
            raise RouteSpecError(f"route {route_id}: step missing event")
        transfers = raw.get("transfers") or []
        actions = []
        for move in transfers:
            if "out" not in move or "in" not in move:
                raise RouteSpecError(f"route {route_id} event {raw['event']}: transfer needs out and in ids")
            actions.append(ts.TransferAction(int(move["out"]), int(move["in"])))
        chip = raw.get("chip")
        steps.append(RouteStep(event=int(raw["event"]), transfer_batch=ts.TransferBatch(tuple(actions)),
                               requested_chip=None if chip is None else str(chip)))
    return TransferRoute(route_id=route_id, steps=tuple(sorted(steps, key=lambda s: int(s.event))),
                         label=spec.get("label"))


def validate_routes(routes: Sequence[TransferRoute], events: Sequence[int]) -> dict[str, list[str]]:
    """Every route must have exactly one step for every evaluated event."""

    problems: dict[str, list[str]] = {}
    seen_ids: set[str] = set()
    for route in routes:
        errors: list[str] = problems.setdefault(route.route_id, [])
        if route.route_id in seen_ids:
            errors.append(f"DUPLICATE_ROUTE_ID: {route.route_id}")
        seen_ids.add(route.route_id)
        for event in events:
            matching = [s for s in route.steps if int(s.event) == int(event)]
            if len(matching) == 0:
                errors.append(f"MISSING_STEP_FOR_EVENT: {event}")
            elif len(matching) > 1:
                errors.append(f"DUPLICATE_STEP_FOR_EVENT: {event}")
        for step in route.steps:
            if step.requested_chip and _is_chip(step.requested_chip):
                errors.append(f"{CHIP_ROUTE_FLAG}: event {step.event} requests chip {step.requested_chip}")
            elif step.requested_chip:
                errors.append(f"UNKNOWN_CHIP_REQUEST: {step.requested_chip}")
    return problems


def _is_chip(name: str) -> bool:
    normalised = str(name).lower().replace(" ", "").replace("-", "")
    return any(keyword in normalised for keyword in CHIP_KEYWORDS)


# ---------------------------------------------------------------------------
# Real-state helpers
# ---------------------------------------------------------------------------


def load_player_meta(conn, player_ids: Iterable[int]) -> dict[int, ts.PlayerMeta]:
    from .scoring_rules import POSITION_IDS

    ids = sorted({int(pid) for pid in player_ids})
    if not ids:
        return {}
    rows = conn.execute(
        "SELECT id, element_type, team_id FROM players WHERE id IN (%s)" % ",".join("?" for _ in ids), ids
    ).fetchall()
    meta = {}
    for row in rows:
        position = POSITION_IDS.get(int(row["element_type"])) if row["element_type"] is not None else None
        meta[int(row["id"])] = ts.PlayerMeta(int(row["id"]), str(position), int(row["team_id"]))
    return meta


def build_route_state(conn, context, squad, price_snapshot: ts.PriceSnapshot | None = None) -> ts.RouteState:
    """Derive the initial Phase-7A RouteState from the canonical PlanningContext."""

    meta = load_player_meta(conn, squad["squad_ids"])
    selling = {int(row["player_id"]): row for row in context.selling_prices}
    players = []
    for pid in squad["squad_ids"]:
        evidence = selling.get(int(pid), {})
        purchase = evidence.get("purchase_price")
        if purchase is None:
            raise RouteSpecError(f"missing purchase price evidence for player {pid}")
        players.append(ts.RoutePlayer(int(pid), squad["positions"][pid], int(meta[int(pid)].club_id), int(purchase)))
    manager = context.manager_state or {}
    return ts.RouteState(
        event=int(context.planning_event), players=tuple(players),
        bank_tenths=int(manager.get("bank") or 0),
        free_transfers=int(manager.get("free_transfers") or 0),
        chip_state=tuple(
            {"name": chip.get("name"), "number": chip.get("number")} for chip in (context.chips or [])
        ),
        # Explicit provenance for the Wildcard/Free-Hit FT transition; None when
        # the operator has not stated it (never inferred from the remaining FT).
        event_start_free_transfers=(
            None if manager.get("event_start_free_transfers") is None
            else int(manager["event_start_free_transfers"])
        ),
    )


# ---------------------------------------------------------------------------
# Per-world scoring (reuses the Phase-6 manager process)
# ---------------------------------------------------------------------------


def policy_world_scores(policy, world_matrix, positions) -> list[float]:
    """Score one policy across the shared worlds.

    Fails closed when the matrix cannot supply a real series for every player
    the policy can score.  The old ``minutes.get(pid, empty)`` default silently
    turned an uncaptured player into an all-zero (non-appearing) player, which
    then triggered an autosub instead of raising.
    """

    worlds = int(world_matrix["worlds"])
    core = world_matrix["core"]
    minutes = world_matrix["minutes"]
    pids = [int(pid) for pid in world_matrix["player_ids"]]
    required = manager_lineup.policy_player_ids(policy)
    manager_lineup.validate_matrix_covers_players(world_matrix, required,
                                                  context="policy_world_scores")
    series_minutes = {pid: minutes[pid] for pid in pids}
    series_core = {pid: core[pid] for pid in pids}
    scores: list[float] = []
    for world in range(worlds):
        w_minutes = {pid: float(series_minutes[pid][world]) for pid in pids}
        w_core = {pid: float(series_core[pid][world]) for pid in pids}
        outcome = manager_lineup.resolve_world(policy, positions, w_minutes, w_core,
                                               require_player_ids=required)
        base = sum(w_core[pid] for pid in outcome.counted_ids)
        extra, _armband = manager_lineup.captain_multiplier(policy, w_minutes, w_core,
                                                            require_player_ids=required)
        scores.append(base + extra)
    return scores


def _best_policy(squad_ids, positions, world_matrix, *, top_k: int = 1):
    ranked = manager_lineup.rank_policies(squad_ids, positions, world_matrix, top_k=top_k)
    if not ranked["top_policies"]:
        raise RouteSpecError("no legal manager policy for squad")
    policy = ranked["top_policies"][0]
    scores = policy_world_scores(policy, world_matrix, positions)
    return policy, scores


# ---------------------------------------------------------------------------
# Comparator
# ---------------------------------------------------------------------------


def compare_routes(
    *,
    bundles: Mapping[int, EventBundle],
    routes: Sequence[TransferRoute],
    initial_state: ts.RouteState,
    scenario: PriceScenario,
    player_meta: Mapping[int, ts.PlayerMeta],
    conn=None,
    world_provider: Callable[[int], Mapping[str, Any]] | None = None,
    simulations: int = 2000,
    seed: int = 20260911,
    planning_cutoff: str | None = None,
) -> dict[str, Any]:
    """Evaluate every explicit route in shared per-event football worlds."""

    events = sorted(int(event) for event in bundles)
    route_problems = validate_routes(routes, events)

    union_ids = {int(p.player_id) for p in initial_state.players}
    for route in routes:
        for step in route.steps:
            union_ids.update(int(a.out_player_id) for a in step.transfer_batch.actions)
            union_ids.update(int(a.in_player_id) for a in step.transfer_batch.actions)

    # ONE football world set per event, shared by every route.
    worlds_by_event: dict[int, Mapping[str, Any]] = {}
    worlds_generation: list[dict[str, Any]] = []
    for event in events:
        if world_provider is not None:
            matrix = world_provider(event)
        else:
            if conn is None:
                raise RouteSpecError("compare_routes needs a connection or a world_provider")
            bundle = bundles[event]
            # The comparator's own DB branch is a predictive-data loader too, so it
            # crosses the SAME certification boundary: a bundle that cannot declare
            # its certified provenance (and whose declared versions disagree with the
            # runs it names) is refused rather than simulated.
            certified_bundle.assert_event_bundle_certified(conn, bundle, event=int(event))
            fixtures = monte_carlo.load_fixture_inputs(
                conn, event=event, xpts_run_id=int(bundle.xpts_run_id),
                minutes_run_id=int(bundle.minutes_run_id), team_run_id=int(bundle.team_run_id),
            )
            config = monte_carlo.MonteCarloConfig(
                simulations=int(bundle.simulations), seed=int(bundle.seed), occupancy_audit=True
            )
            result = monte_carlo.simulate(fixtures, config, capture_player_ids=sorted(union_ids))
            matrix = result["world_matrix"]
            worlds_generation.append({
                "event": event, "worlds": int(config.simulations),
                "occupancy_violations": int(result.get("occupancy_violations") or 0),
                "minute_mass_violations": int(result.get("minute_mass_violations") or 0),
                "union_players": len(union_ids),
                "config_hash": config.config_hash(),
            })
        worlds_by_event[event] = matrix

    policy_cache: dict[tuple[int, str], dict[str, Any]] = {}
    route_results: dict[str, dict[str, Any]] = {}
    per_event_scores: dict[tuple[str, int], list[float]] = {}
    per_event_net: dict[tuple[str, int], list[float]] = {}

    for route in routes:
        problems = list(route_problems.get(route.route_id, []))
        state = initial_state
        records: list[dict[str, Any]] = []
        valid = True
        valid_through = int(initial_state.event) - 1
        failure: str | None = None
        for event in events:
            if not valid:
                break
            step = route.step_for(event)
            if step is None:
                valid, failure = False, f"MISSING_STEP_FOR_EVENT: {event}"
                break
            if step.requested_chip and _is_chip(step.requested_chip):
                valid, failure = False, f"{CHIP_ROUTE_FLAG}: event {event} requests chip {step.requested_chip}"
                break
            snapshot = scenario.snapshot_for(event)
            if snapshot is None:
                valid, failure = False, f"NO_PRICE_SNAPSHOT_FOR_EVENT: {event}"
                break
            transition = ts.apply_transfer_batch(state, step.transfer_batch, snapshot, player_meta)
            if not transition.ok:
                valid, failure = False, "; ".join(transition.errors)
                break
            squad_after = transition.squad_after
            squad_ids = tuple(int(p.player_id) for p in squad_after.players)
            positions = {int(p.player_id): p.position for p in squad_after.players}
            matrix = worlds_by_event[event]
            key = (event, squad_after.squad_hash())
            cached = policy_cache.get(key)
            if cached is None:
                policy, scores = _best_policy(squad_ids, positions, matrix, top_k=1)
                cached = {
                    "policy": policy,
                    "squad_ids": squad_ids,
                    "positions": positions,
                    "gross_world_scores": scores,
                    "mean_gross_core": sum(scores) / len(scores),
                }
                policy_cache[key] = cached
            gross = cached["gross_world_scores"]
            net = [value - transition.hit_points for value in gross]
            per_event_scores[(route.route_id, event)] = gross
            per_event_net[(route.route_id, event)] = net
            records.append({
                "event": event,
                "batch_size": len(step.transfer_batch),
                "hit_points": int(transition.hit_points),
                "free_transfers_before": int(transition.ft_before),
                "free_transfers_used": int(transition.ft_used),
                "paid_transfers": int(transition.paid_transfers),
                "bank_before_tenths": int(transition.bank_before_tenths),
                "bank_after_tenths": int(transition.bank_after_tenths),
                "next_free_transfers": int(transition.next_event_state.free_transfers),
                "next_bank_tenths": int(transition.next_event_state.bank_tenths),
                "mean_gross_core": float(cached["mean_gross_core"]),
                "mean_net_core": float(sum(net) / len(net)),
                "selected_policy": {
                    "starter_ids": list(cached["policy"].starter_ids),
                    "bench_gk_id": int(cached["policy"].bench_gk_id),
                    "bench_outfield_order": list(cached["policy"].bench_outfield_order),
                    "captain_id": int(cached["policy"].captain_id),
                    "vice_captain_id": int(cached["policy"].vice_captain_id),
                },
                "squad_hash": squad_after.squad_hash(),
            })
            state = transition.next_event_state
            valid_through = event
        route_results[route.route_id] = {
            "route_id": route.route_id,
            "label": route.label,
            "valid": valid and not problems,
            "valid_through_event": valid_through,
            "failure": failure or ("; ".join(problems) if problems else None),
            "errors": problems,
            "events": records,
            "terminal_state": {
                "event": int(state.event),
                "free_transfers": int(state.free_transfers),
                "bank_tenths": int(state.bank_tenths),
                "squad_hash": state.squad_hash(),
            },
        }

    _attach_horizons(route_results, events)
    paired, cumulative_paired = _paired_comparisons(routes, route_results, per_event_scores, per_event_net, events)
    pareto = _pareto_frontier(route_results, events)
    # Horizons require a CONTIGUOUS run of events from the initial event; a missing
    # intermediate event must never be silently skipped to form an "H4".
    supported = [label for label, steps in HORIZON_STEPS.items() if _contiguous_prefix(events) >= steps]
    unsupported = [label for label in HORIZON_STEPS if label not in supported]

    flags = [CROSS_GW_FLAG]
    if FLAT_PRICE_ASSUMPTION in (scenario.flags or ()):
        flags.append(FLAT_PRICE_ASSUMPTION)

    return {
        "phase_version": PHASE7B_VERSION,
        "planning_cutoff": planning_cutoff,
        "price_scenario_id": scenario.scenario_id,
        "price_scenario_flags": list(scenario.flags),
        "events": events,
        "supported_horizons": supported,
        "unsupported_horizons": unsupported,
        "union_player_count": len(union_ids),
        "routes": route_results,
        "route_count": len(routes),
        "valid_route_count": sum(1 for r in route_results.values() if r["valid"]),
        "invalid_route_count": sum(1 for r in route_results.values() if not r["valid"]),
        "paired_event_differences": paired,
        "cumulative_paired_differences": cumulative_paired,
        "pareto_frontier": pareto,
        "policy_cache_entries": len(policy_cache),
        "worlds_generation": worlds_generation,
        "flags": flags,
        "near_tie_k": NEAR_TIE_K,
        "no_recommendation": True,
    }


def _contiguous_prefix(events: Sequence[int]) -> int:
    """Length of the contiguous run starting at the first event (no gaps)."""

    if not events:
        return 0
    count = 1
    for previous, current in zip(events, list(events)[1:]):
        if int(current) == int(previous) + 1:
            count += 1
        else:
            break
    return count


def _attach_horizons(route_results: dict[str, dict[str, Any]], events: Sequence[int]) -> None:
    contiguous = _contiguous_prefix(events)
    for result in route_results.values():
        horizons: dict[str, Any] = {}
        records = result["events"]
        for label, steps in HORIZON_STEPS.items():
            if len(events) < steps or contiguous < steps or len(records) < steps:
                horizons[label] = {"supported": False, "reason": "INSUFFICIENT_CONTIGUOUS_EVENTS"}
                continue
            window = list(range(steps))
            cumulative = records[:steps]
            gross = sum(record["mean_gross_core"] for record in cumulative)
            hit = sum(record["hit_points"] for record in cumulative)
            horizons[label] = {
                "supported": True,
                "events": [int(events[i]) for i in window],
                "gross_core": gross,
                "cumulative_hits": hit,
                "net_core": gross - hit,
                # Terminal manager state AFTER the horizon's last event's deadline.
                "terminal_ft": int(cumulative[-1]["next_free_transfers"]),
                "terminal_bank_tenths": int(cumulative[-1]["next_bank_tenths"]),
            }
        result["horizons"] = horizons


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return float(ordered[index])


def _paired_comparisons(routes, route_results, per_event_scores, per_event_net, events):
    paired: list[dict[str, Any]] = []
    cumulative: list[dict[str, Any]] = []
    valid_routes = [r.route_id for r in routes if route_results[r.route_id]["valid"]]
    for index, route_a in enumerate(valid_routes):
        for route_b in valid_routes[index + 1:]:
            for event in events:
                scores_a = per_event_scores.get((route_a, event))
                scores_b = per_event_scores.get((route_b, event))
                if scores_a is None or scores_b is None:
                    continue
                paired.append(_paired_record("event", event, route_a, route_b, scores_a, scores_b,
                                             flag=None))
            contiguous = _contiguous_prefix(events)
            for label, steps in HORIZON_STEPS.items():
                if len(events) < steps or contiguous < steps:
                    continue
                window = events[:steps]
                if any((route_a, e) not in per_event_net or (route_b, e) not in per_event_net for e in window):
                    continue
                cum_a = [sum(per_event_net[(route_a, e)][w] for e in window) for w in range(len(per_event_net[(route_a, window[0])]))]
                cum_b = [sum(per_event_net[(route_b, e)][w] for e in window) for w in range(len(per_event_net[(route_b, window[0])]))]
                record = _paired_record("horizon", label, route_a, route_b, cum_a, cum_b, flag=CROSS_GW_FLAG)
                cumulative.append(record)
    return paired, cumulative


def _paired_record(kind, key, route_a, route_b, scores_a, scores_b, *, flag):
    differences = [a - b for a, b in zip(scores_a, scores_b)]
    n = len(differences)
    mean = sum(differences) / n
    variance = sum((d - mean) ** 2 for d in differences) / (n - 1) if n > 1 else 0.0
    se = math.sqrt(variance / n) if n > 1 else 0.0
    wins = sum(1 for d in differences if d > 0)
    near_tied = abs(mean) <= NEAR_TIE_K * se if se > 0 else mean == 0.0
    return {
        "kind": kind, "key": key, "route_a": route_a, "route_b": route_b,
        "worlds": n, "mean_difference": mean, "paired_se": se,
        "q10_difference": _quantile(differences, 0.10), "q50_difference": _quantile(differences, 0.50),
        "q90_difference": _quantile(differences, 0.90), "p_a_gt_b": wins / n,
        "near_tied": bool(near_tied),
        "flags": [flag] if flag else [],
    }


def _pareto_frontier(route_results, events):
    frontier: dict[str, list[str]] = {}
    contiguous = _contiguous_prefix(events)
    for label, steps in HORIZON_STEPS.items():
        if len(events) < steps or contiguous < steps:
            continue
        candidates = [rid for rid, res in route_results.items() if res["valid"]]
        dimensions = {}
        for rid in candidates:
            horizon = route_results[rid]["horizons"][label]
            dimensions[rid] = (
                horizon["net_core"], -horizon["cumulative_hits"],
                horizon["terminal_ft"], horizon["terminal_bank_tenths"],
            )
        frontier[label] = [
            rid for rid in sorted(candidates)
            if not any(
                _dominates(dimensions[other], dimensions[rid])
                for other in candidates if other != rid
            )
        ]
    return frontier


def _dominates(a, b) -> bool:
    """a dominates b: no worse on every dimension and strictly better on one."""

    no_worse = all(x >= y for x, y in zip(a, b))
    strictly_better = any(x > y for x, y in zip(a, b))
    return no_worse and strictly_better
