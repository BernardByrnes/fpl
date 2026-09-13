"""Canonical joint role-conditioned minutes kernel (single source of truth).

Both consumers use the SAME process:

* **Minutes v1.5** integrates the kernel to obtain player marginal summaries.
* **Monte Carlo** uses kernel draws as part of the full football-event simulation.

The kernel realises one side-world: it draws the starting XI with the accepted
randomised fixed-size sampler, draws the substitution count ``K`` and each
event time from the frozen ``TeamSubstitutionProfile``, and then -- crucially --
selects the **exiting** player only from players who ACTUALLY START in that
world and the **entering** player only from players ACTUALLY ON THE BENCH.  A
player with ``P(start) = 0.8`` therefore only receives entry opportunity in the
20% of worlds where they are benched, which is what the earlier marginal
allocation could not express.

Physical invariants per world (no red cards modelled):
``11`` players and exactly ``1`` goalkeeper on the pitch at every instant, every
departure paired with an arrival at the same time, total team player-minutes
exactly ``990``, and at most the verified number of ordinary substitutions.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, fields
from typing import Any, Mapping, Sequence

from . import analytics

JOINT_MINUTES_MODEL_VERSION = "minutes_v1.5.2"
_JOINT_SEED_NAMESPACE = "minutes_joint_v1.5"
MC_SEED_NAMESPACE = "mc_v1.1"

MAX_ORDINARY_SUBSTITUTIONS = 5


@dataclass(frozen=True)
class JointMinutesConfig:
    """Tunables of the shared kernel (also hashed into every v1.5 run)."""

    max_substitute_entrants: int = MAX_ORDINARY_SUBSTITUTIONS
    substitution_minute_jitter: tuple = (-10.0, -5.0, 0.0, 5.0, 10.0)
    substitution_minute_jitter_weights: tuple = (0.15, 0.20, 0.30, 0.20, 0.15)
    substitution_minute_min: float = 30.0
    substitution_minute_max: float = 89.0
    gk_substitution_ceiling: float = 0.06
    integration_draws: int = 20000
    integration_seed: int = 20260911

    def config_hash(self) -> str:
        values = {item.name: getattr(self, item.name) for item in fields(self)}
        return analytics.canonical_hash({"model": JOINT_MINUTES_MODEL_VERSION, **values})


# ---------------------------------------------------------------------------
# Sampling primitives.
# ---------------------------------------------------------------------------


def weighted_choice(weights: Sequence[float], rng: random.Random) -> int:
    total = sum(max(0.0, w) for w in weights)
    if total <= 0:
        return 0
    draw = rng.random() * total
    cumulative = 0.0
    for index, weight in enumerate(weights):
        cumulative += max(0.0, weight)
        if draw < cumulative:
            return index
    return len(weights) - 1


def stochastic_round(value: float, u: float) -> int:
    floor = int(math.floor(value))
    return floor + 1 if u < (value - floor) else floor


def entity_order_key(namespace: str, *parts: Any) -> float:
    """Deterministic per-entity ordering key, independent of roster composition.

    A player's key depends only on its own identity and the world's namespace, so
    adding or removing an unrelated (e.g. zero-probability container) entity never
    reorders the shared players.  This replaces a Fisher-Yates permutation over
    list positions, which shifted every induced pairwise structure whenever a
    roster row was added.
    """

    return random.Random("|".join([namespace, *(str(part) for part in parts)])).random()


def joint_order_keys(
    seed_namespace: str,
    fixture_id: int,
    team_id: int,
    draw_index: int,
    player_ids: Sequence[int],
) -> list[float]:
    """CANONICAL per-draw traversal order for the joint sampler.

    Minutes integration, scorer/assist calibration and production Monte Carlo all
    call this with their own seed namespace and their own draw/state index, so the
    three consumers describe the SAME distribution over traversal order while
    their individual draws stay statistically independent.  Ordering is
    randomised afresh for every draw/state (the fixed-order process is retired).
    """

    return [
        entity_order_key(seed_namespace, fixture_id, team_id, draw_index, int(player_id))
        for player_id in player_ids
    ]


def systematic_select(probabilities: Sequence[float], count: int, u: float, *,
                      randomised: bool = False, rng: random.Random | None = None,
                      order_keys: Sequence[float] | None = None) -> list[int]:
    """Fixed-size systematic sampling: exactly ``count`` distinct picks.

    When ``order_keys`` are supplied the traversal order is the deterministic
    key order (roster-composition invariant); otherwise the legacy randomised
    Fisher-Yates order is used.
    """

    total = sum(probabilities)
    if count <= 0 or total <= 0:
        return []
    scaled = [min(1.0, max(0.0, p) * count / total) for p in probabilities]
    if order_keys is not None:
        order = sorted(range(len(scaled)), key=lambda i: (float(order_keys[i]), i))
    else:
        order = list(range(len(scaled)))
        if randomised and rng is not None:
            for i in range(len(order) - 1, 0, -1):
                j = rng.randrange(i + 1)
                order[i], order[j] = order[j], order[i]
    targets = [u + k for k in range(count)]
    cumulative = 0.0
    selected: set[int] = set()
    target_index = 0
    for index in order:
        upper = cumulative + scaled[index]
        while target_index < len(targets) and targets[target_index] < upper and len(selected) < count:
            selected.add(index)
            target_index += 1
        cumulative = upper
    return sorted(selected)


def check_world_occupancy(intervals: Mapping[int, tuple[float, float]], gk_player_ids: Sequence[int]) -> tuple[bool, str]:
    """Exactly 11 players and 1 goalkeeper on the pitch at every instant."""

    boundaries = {0.0, 90.0}
    for low, high in intervals.values():
        boundaries.update((low, high))
    gk_set = set(gk_player_ids)
    for time in sorted(boundaries):
        probe = min(89.999999, time + 1e-6)
        on_pitch = [pid for pid, (low, high) in intervals.items() if low <= probe <= high]
        if len(on_pitch) != 11:
            return False, f"{len(on_pitch)} on pitch at {time}"
        if len([pid for pid in on_pitch if pid in gk_set]) != 1:
            return False, f"goalkeeper count wrong at {time}"
    return True, ""


# ---------------------------------------------------------------------------
# The one kernel.
# ---------------------------------------------------------------------------


def sample_side_world(
    players: Sequence[Mapping[str, Any]],
    profile: Mapping[str, Any] | None,
    config: JointMinutesConfig,
    rng: random.Random,
    *,
    order_keys: Sequence[float] | None = None,
    rng_lineup: random.Random | None = None,
    rng_substitutions: random.Random | None = None,
) -> dict[str, Any]:
    """One realisation of a side's XI and paired substitution timeline.

    ``players`` entries carry ``player_id``, ``position``, ``p_start``,
    ``p_available``, ``exit_propensity``, ``entry_propensity`` and
    ``expected_minutes_if_cameo``.  Exit selection is conditioned on the
    realised starting XI and entry selection on the realised bench.

    ``rng_lineup`` / ``rng_substitutions`` let a caller split the two categories
    into separate deterministic substreams (common random numbers).  When they
    are omitted both fall back to ``rng``, which keeps the integration and the
    simulation the same process.
    """

    lineup_rng = rng_lineup if rng_lineup is not None else rng
    sub_rng = rng_substitutions if rng_substitutions is not None else rng
    count = len(players)
    positions = [p.get("position") for p in players]
    gk_indices = [i for i in range(count) if positions[i] == "GKP"]
    out_indices = [i for i in range(count) if positions[i] != "GKP"]
    p_start = [float(p.get("p_start") or 0.0) for p in players]
    exit_prop = [max(0.0, float(p.get("exit_propensity") or 0.0)) for p in players]
    entry_prop = [max(0.0, float(p.get("entry_propensity") or 0.0)) for p in players]
    cameo_minutes = [float(p.get("expected_minutes_if_cameo") or 0.0) for p in players]
    out_order_keys = [float(order_keys[i]) for i in out_indices] if order_keys is not None else None

    starters: set[int] = set()
    gk_weights = [p_start[i] for i in gk_indices]
    if sum(gk_weights) > 0:
        starters.add(gk_indices[weighted_choice(gk_weights, lineup_rng)])
    for offset in systematic_select([p_start[i] for i in out_indices], 10, lineup_rng.random(),
                                    randomised=True, rng=lineup_rng, order_keys=out_order_keys):
        starters.add(out_indices[offset])

    profile = profile or {}
    exiting_pool = [i for i in sorted(starters) if positions[i] != "GKP"]
    if profile.get("p_sub_count"):
        counts = [float(p) for p in profile["p_sub_count"]]
        n_events = min(config.max_substitute_entrants, len(exiting_pool), weighted_choice(counts, sub_rng))
        band_masses = [float(m) for m in profile.get("event_time_masses", [])]
        band_bounds = profile.get("event_time_bands") or []
    else:
        # No frozen profile (standalone use): fall back to the bench's own
        # entry-propensity mass so the kernel remains usable on its own.
        n_events = min(
            config.max_substitute_entrants, len(exiting_pool),
            stochastic_round(sum(entry_prop[i] for i in range(count) if i not in starters), sub_rng.random()),
        )
        band_masses, band_bounds = [], []

    events: list[dict[str, Any]] = []
    # An outfield starter can never be replaced by a goalkeeper, so keepers
    # are excluded from the ordinary bench pool (GK events are separate).
    bench = [i for i in range(count) if i not in starters and positions[i] != "GKP"]
    for _ in range(max(0, n_events)):
        if not exiting_pool or not bench:
            break
        exit_offset = weighted_choice([max(1e-6, exit_prop[i]) for i in exiting_pool], sub_rng)
        exiting = exiting_pool.pop(exit_offset)
        entry_weights = [max(0.0, entry_prop[i]) for i in bench]
        if sum(entry_weights) <= 0:
            entry_weights = [1.0] * len(bench)
        entering = bench.pop(weighted_choice(entry_weights, sub_rng))
        events.append(_event(exiting, entering, cameo_minutes[entering], sub_rng, config, band_masses, band_bounds))

    gk_starter = next(iter(starters & set(gk_indices)), None)
    gk_backups = [i for i in gk_indices if i not in starters]
    if profile:
        gk_marginal = float(profile.get("gk_event_mass") or 0.0)
    else:
        gk_marginal = sum(max(0.0, entry_prop[i]) for i in gk_backups)
    gk_probability = min(gk_marginal, config.gk_substitution_ceiling)
    if (gk_starter is not None and gk_backups and len(events) < config.max_substitute_entrants
            and sub_rng.random() < gk_probability):
        weights = [max(0.0, entry_prop[i]) for i in gk_backups]
        entering = gk_backups[weighted_choice(weights, sub_rng)] if sum(weights) > 0 else gk_backups[0]
        events.append(_event(gk_starter, entering, cameo_minutes[entering], sub_rng, config, [], []))

    intervals = {players[i]["player_id"]: (0.0, 90.0) for i in starters}
    for event in events:
        intervals[players[event["exiting"]]["player_id"]] = (0.0, event["time"])
        intervals[players[event["entering"]]["player_id"]] = (event["time"], 90.0)
    minutes = {pid: max(0.0, high - low) for pid, (low, high) in intervals.items()}
    for player in players:
        minutes.setdefault(player["player_id"], 0.0)
    return {
        "starters": starters,
        "substitutes": {event["entering"] for event in events},
        "events": events,
        "intervals": intervals,
        "minutes": minutes,
        "gk_starters": len(starters & set(gk_indices)),
        "outfield_starters": len(starters & set(out_indices)),
        "substitute_entrants": len(events),
        "gk_substitution": any(positions[event["entering"]] == "GKP" for event in events),
        "suppressed_gk_cameo_mass": max(0.0, gk_marginal - gk_probability),
    }


def _event(exiting, entering, cameo_mean, rng, config, band_masses, band_bounds):
    """One paired substitution at a time drawn uniformly over its full band.

    The whole configured band is reachable: a 75-89 band permits every minute
    75..89 with equal probability.  There is no midpoint-with-jitter narrowing,
    which used to leave the band edges unreachable.
    """

    if band_masses and band_bounds and sum(band_masses) > 0:
        band_index = weighted_choice(band_masses, rng)
        low, high = band_bounds[band_index]
        # Profile bands are the audited empirical bands; use them as given so a
        # 0-29 band is not silently collapsed to the standalone fallback floor.
        low = max(0.0, float(low))
        high = min(config.substitution_minute_max, float(high))
    else:
        low, high = config.substitution_minute_min, config.substitution_minute_max
    if high < low:
        low, high = high, low
    low_int, high_int = int(round(low)), int(round(high))
    if high_int < low_int:
        high_int = low_int
    # Genuinely uniform INTEGER minute over the whole allowed band, endpoints
    # included with equal weight.
    time = low_int + int(rng.random() * (high_int - low_int + 1))
    return {"exiting": exiting, "entering": entering, "time": float(time)}


# ---------------------------------------------------------------------------
# Minutes v1.5 marginal integration.
# ---------------------------------------------------------------------------


# Finer compact minute quadrature for the deterministic nonlinear components
# (DefCon, saves).  The Monte Carlo always uses TRUE sampled minutes; these bins
# exist only so the analytic expectation integrates the same distribution with a
# small within-bin Jensen gap.
MINUTE_STATE_BOUNDS = (
    (0.0, 0.0), (1.0, 9.0), (10.0, 19.0), (20.0, 29.0), (30.0, 39.0),
    (40.0, 49.0), (50.0, 59.0), (60.0, 69.0), (70.0, 79.0), (80.0, 89.0), (90.0, 90.0),
)
MINUTE_STATE_LABELS = ("0", "1-9", "10-19", "20-29", "30-39", "40-49", "50-59",
                       "60-69", "70-79", "80-89", "90")


def _minute_state(minutes: float) -> int:
    """Bucket a sampled minute value into the frozen compact quadrature."""

    value = float(minutes)
    if value <= 0.0:
        return 0
    if value >= 90.0:
        return len(MINUTE_STATE_LABELS) - 1
    return 1 + int(value) // 10


def integrate_side_marginals(
    players: Sequence[Mapping[str, Any]],
    profile: Mapping[str, Any] | None,
    config: JointMinutesConfig,
    *,
    fixture_id: int,
    team_id: int,
    draws: int | None = None,
) -> dict[int, dict[str, Any]]:
    """Integrate the joint kernel to obtain per-player marginal summaries.

    A dedicated seed namespace (``minutes_joint_v1.5``) keeps the integration
    statistically independent of the Monte Carlo certification run.
    """

    draws = int(draws or config.integration_draws)
    rng = random.Random(f"{_JOINT_SEED_NAMESPACE}:{config.integration_seed}:{fixture_id}:{team_id}")
    player_ids = [p["player_id"] for p in players]
    bin_count = len(MINUTE_STATE_LABELS)
    stats = {
        p["player_id"]: {
            "start": 0, "cameo": 0, "zero": 0, "m60": 0, "m80": 0,
            "m60_start": 0, "m80_start": 0, "m60_cameo": 0,
            "minutes": 0.0, "minutes_start": 0.0, "minutes_cameo": 0.0,
            # Frozen compact minute quadrature (see MINUTE_STATE_BOUNDS).
            "state_counts": [0] * bin_count, "state_minutes": [0.0] * bin_count,
        }
        for p in players
    }
    for draw_index in range(draws):
        # Fresh deterministic randomised traversal order EVERY draw, exactly as
        # production Monte Carlo and the calibration library do.
        order_keys = joint_order_keys(
            _JOINT_SEED_NAMESPACE, fixture_id, team_id, draw_index, player_ids
        )
        world = sample_side_world(players, profile, config, rng, order_keys=order_keys)
        for player in players:
            pid = player["player_id"]
            entry = stats[pid]
            value = world["minutes"][pid]
            entry["minutes"] += value
            state = _minute_state(value)
            entry["state_counts"][state] += 1
            entry["state_minutes"][state] += value
            if value <= 0:
                entry["zero"] += 1
                continue
            if value >= 90.0 - 1e-9:
                entry["start"] += 1
            # role in this world decides whether the appearance was a start
            index = players.index(player)
            if index in world["starters"]:
                entry["start"] += 1 if value < 90.0 - 1e-9 else 0
                entry["minutes_start"] += value
                if value >= 60.0:
                    entry["m60"] += 1
                    entry["m60_start"] += 1
                if value >= 80.0:
                    entry["m80"] += 1
                    entry["m80_start"] += 1
            else:
                entry["cameo"] += 1
                entry["minutes_cameo"] += value
                if value >= 60.0:
                    entry["m60"] += 1
                    entry["m60_cameo"] += 1
    out: dict[int, dict[str, Any]] = {}
    for pid, entry in stats.items():
        start, cameo = entry["start"], entry["cameo"]
        out[pid] = {
            "draws": draws,
            "p_start": start / draws,
            "p_cameo": cameo / draws,
            "p_zero": entry["zero"] / draws,
            "p_60_plus": entry["m60"] / draws,
            "p_80_plus": entry["m80"] / draws,
            "expected_minutes": entry["minutes"] / draws,
            "expected_minutes_if_start": (entry["minutes_start"] / start) if start else 0.0,
            "expected_minutes_if_cameo": (entry["minutes_cameo"] / cameo) if cameo else 0.0,
            "p_60_given_start": (entry["m60_start"] / start) if start else 0.0,
            "p_80_given_start": (entry["m80_start"] / start) if start else 0.0,
            "p_60_given_cameo": (entry["m60_cameo"] / cameo) if cameo else 0.0,
            "start_standard_error": math.sqrt(max(0.0, (start / draws) * (1 - start / draws) / draws)),
            "minute_state_distribution": [
                {
                    "state": MINUTE_STATE_LABELS[index],
                    "probability": entry["state_counts"][index] / draws,
                    "conditional_mean_minutes": (
                        entry["state_minutes"][index] / entry["state_counts"][index]
                    ) if entry["state_counts"][index] else 0.0,
                }
                for index in range(bin_count)
            ],
        }
    return out


def build_minutes_v15(
    conn,
    planning_event: int,
    cutoff: str,
    *,
    joint_config: JointMinutesConfig | None = None,
    model_config=None,
    coherence_config=None,
    sub_config=None,
):
    """Minutes v1.5: the frozen projection is the marginal of the joint process.

    Reuses the accepted positional-coherent start marginals (as *targets*), the
    accepted substitution evidence/profile, and integrates the shared kernel.
    Player conditional quantities are OUTPUTS of the integration, not inputs.
    """

    from . import minutes_coherence, minutes_model, substitution_model

    joint_config = joint_config or JointMinutesConfig()
    model_config = model_config or minutes_model.MinutesModelConfig()
    coherence_config = coherence_config or minutes_coherence.MinutesCoherenceConfig()
    sub_config = sub_config or substitution_model.SubstitutionConfig()

    base_rows, _records = minutes_coherence.build_minutes_predictions_coherent(
        conn, planning_event, cutoff, model_config, coherence_config, positional=True
    )
    evidence = substitution_model.audit_substitution_evidence(conn, planning_event, cutoff)

    player_team = {
        int(r["id"]): int(r["team_id"])
        for r in conn.execute("SELECT id, team_id FROM players WHERE is_active=1 AND team_id IS NOT NULL")
    }
    sides = {}
    for row in base_rows:
        sides.setdefault((int(row["fixture_id"]), player_team[int(row["player_id"])]), []).append(row)

    rows_out = []
    profiles_out = []
    for key in sorted(sides):
        payloads = sorted(sides[key], key=lambda r: int(r["player_id"]))
        kernel_players = []
        for p in payloads:
            p_start = float(p.get("p_start") or 0.0)
            kernel_players.append({
                "player_id": int(p["player_id"]),
                "position": p.get("position_id") and {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}.get(int(p["position_id"]), "MID"),
                "p_start": p_start,
                "p_available": float(p.get("p_available") or 0.0),
                "exit_propensity": min(3.0, max(0.02, 1.0 - float(p.get("p_80_given_start") or 0.0))),
                "entry_propensity": float(p.get("p_cameo") or 0.0) / max(1e-9, 1.0 - p_start),
                "expected_minutes_if_cameo": float(p.get("expected_minutes_if_cameo") or 0.0),
            })
        profile_side = [
            {"p_start": kp["p_start"], "p_available": kp["p_available"],
             "p80_given_start": 1.0 - kp["exit_propensity"],
             "cameo_propensity": kp["entry_propensity"],
             "expected_minutes_if_start": float(payloads[i].get("expected_minutes_if_start") or 0.0)}
            for i, kp in enumerate(kernel_players)
        ]
        profile = substitution_model.build_team_substitution_profile(
            profile_side, evidence, sub_config, fixture_id=key[0], team_id=key[1]
        )
        if evidence.get("team_fixtures", 0) < 100:
            profile["risk_flags"] = sorted(set(profile["risk_flags"] + ["SUBSTITUTION_PRIOR_EARLY_SEASON"]))
        profile["risk_flags"] = sorted(set(profile["risk_flags"] + (
            ["STOPPAGE_TIME_EXIT_UNOBSERVABLE"] if evidence.get("stoppage_time_substitutions") else []
        )))
        profiles_out.append(profile)

        marginals = integrate_side_marginals(
            kernel_players, profile, joint_config, fixture_id=key[0], team_id=key[1]
        )
        for index, payload in enumerate(payloads):
            pid = int(payload["player_id"])
            m = marginals[pid]
            row = dict(payload)
            kernel_player = kernel_players[index]
            row.update({
                # --- the EXACT primitive parameter vector the joint kernel used.
                # Monte Carlo reads these verbatim; it must never reconstruct
                # propensities from integrated OUTPUTS (p_cameo,
                # p_80_given_start, expected_minutes), which would silently make
                # the integration and the simulation two different models.
                "joint_start_target": round(float(kernel_player["p_start"]), 9),
                "joint_availability": round(float(kernel_player["p_available"]), 9),
                "joint_exit_propensity": round(float(kernel_player["exit_propensity"]), 9),
                "joint_entry_propensity": round(float(kernel_player["entry_propensity"]), 9),
                "joint_position": kernel_player["position"],
                "joint_expected_minutes_if_cameo": round(float(kernel_player["expected_minutes_if_cameo"]), 9),
                "primitive_source_version": JOINT_MINUTES_MODEL_VERSION,
                "target_p_start": round(float(payload.get("p_start") or 0.0), 9),
                "p_start": round(m["p_start"], 9),
                "p_cameo": round(m["p_cameo"], 9),
                "p_zero": round(m["p_zero"], 9),
                "p_60_plus": round(m["p_60_plus"], 9),
                "p_80_plus": round(m["p_80_plus"], 9),
                "p_1_59": round(max(0.0, 1.0 - m["p_zero"] - m["p_60_plus"]), 9),
                "expected_minutes": round(m["expected_minutes"], 9),
                "expected_minutes_if_start": round(m["expected_minutes_if_start"], 9),
                "expected_minutes_if_cameo": round(m["expected_minutes_if_cameo"], 9),
                "p_60_given_start": round(m["p_60_given_start"], 9),
                "p_80_given_start": round(m["p_80_given_start"], 9),
                "p_60_given_cameo": round(m["p_60_given_cameo"], 9),
                # Frozen compact minute-state distribution, shared with the
                # deterministic DefCon/save mixture in xPts v1.4.0.
                "minute_state_distribution": [
                    {
                        "state": entry["state"],
                        "probability": round(entry["probability"], 9),
                        "conditional_mean_minutes": round(entry["conditional_mean_minutes"], 9),
                    }
                    for entry in m["minute_state_distribution"]
                ],
                "p_start_standard_error": round(m["start_standard_error"], 9),
                "p_start_absolute_error": round(m["p_start"] - float(payload.get("p_start") or 0.0), 9),
                "integration_draws": m["draws"],
                "integration_seed_policy": f"{_JOINT_SEED_NAMESPACE}:{{seed}}:{{fixture}}:{{team}}",
                "team_substitution_profile": {
                    "expected_substitutions": profile["expected_substitutions"],
                    "p_sub_count": profile["p_sub_count"],
                    "event_time_bands": profile["event_time_bands"],
                    "event_time_masses": profile["event_time_masses"],
                    "gk_event_mass": profile["gk_event_mass"],
                    "risk_flags": profile["risk_flags"],
                },
                "model_version": JOINT_MINUTES_MODEL_VERSION,
            })
            rows_out.append(row)
    return rows_out, profiles_out


def minutes_output_diagnostics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Minutes v1.5.1 output diagnostics by position (reporting only).

    Reports the realised cameo/zero/long-start marginals against the audited
    substitution evidence shape.  These are diagnostics, not targets, and no
    individual player is tuned to them.
    """

    buckets: dict[str, dict[str, float]] = {}
    for row in rows:
        position = str(row.get("joint_position") or row.get("position") or "UNK")
        bucket = buckets.setdefault(
            position,
            {"players": 0.0, "p_cameo": 0.0, "p_zero": 0.0, "p_60_given_start": 0.0,
             "p_80_given_start": 0.0, "expected_minutes_if_start": 0.0,
             "expected_minutes_if_cameo": 0.0},
        )
        bucket["players"] += 1.0
        bucket["p_cameo"] += float(row.get("p_cameo") or 0.0)
        bucket["p_zero"] += float(row.get("p_zero") or 0.0)
        bucket["p_60_given_start"] += float(row.get("p_60_given_start") or 0.0)
        bucket["p_80_given_start"] += float(row.get("p_80_given_start") or 0.0)
        bucket["expected_minutes_if_start"] += float(row.get("expected_minutes_if_start") or 0.0)
        bucket["expected_minutes_if_cameo"] += float(row.get("expected_minutes_if_cameo") or 0.0)
    out: dict[str, Any] = {"by_position": {}, "sample_players": len(rows)}
    for position, bucket in sorted(buckets.items()):
        count = max(1.0, bucket.pop("players"))
        out["by_position"][position] = {
            **{key: round(value / count, 6) for key, value in bucket.items()},
            "players": int(count),
        }
    return out
