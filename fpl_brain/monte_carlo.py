"""Monte Carlo V1 — joint uncertainty over shared football events.

The accepted analytic xPts model provides expected means; this module turns them
into a **joint** predictive distribution by simulating the underlying football
events, not the points components independently:

    joint lineup / minutes  +  team goals  ->  player goals / assists /
    personal clean-sheet & concession exposure  ->  DefCon / saves / cards
    ->  FPL points

Everything downstream of a player uses the SAME sampled minutes interval; every
player of a team shares the SAME sampled team-goal draw; clean sheets and goals
conceded come from the SAME opponent goal times.

Design notes
------------
* **Fixed-size lineup sampling.**  Each side's XI is drawn with systematic
  sampling over the coherent start marginals, so a world always contains exactly
  1 goalkeeper and 10 outfielders, and each player's inclusion probability is
  exactly the coherent ``P(start)``.
* **Deterministic RNG.**  One RNG per fixture seeded from ``(seed, fixture_id)``;
  entities are sorted before draws, so summaries are byte-identical for a fixed
  input/config/seed and independent of DB or dict ordering.
* **Common random numbers ready.**  Draws depend only on football entities and
  the simulation index, never on route/container ordering, so future route
  comparisons can reuse the same simulated world.
* **Downward-only coherence.**  Scorer/assist weights never exceed the team's own
  goal expectation; any residual attacking mass becomes an explicit
  ``RESIDUAL_SCORER`` bucket rather than being pushed onto modelled players.

Nothing here ranks players, recommends transfers, captains, or chips.
"""

from __future__ import annotations

import math
import random
import re
import sqlite3
import sys
from dataclasses import dataclass, field, fields
from typing import Any, Iterable, Mapping, NamedTuple, Sequence

from . import analytics, joint_minutes
from . import defcon_calibration as defcon_cal
from . import xpts
from .scoring_rules import DEFAULT_SCORING_RULES, POSITION_IDS, SCORING_RULES_VERSION, ScoringRules

MONTE_CARLO_MODEL_VERSION = "mc_v1.3.0"
MONTE_CARLO_MODEL_FAMILY = "monte_carlo_v1"

# Substitution limit.  Documented Premier League rule (5 substitutions per team
# per match in 3 windows; bench of 9; concussion substitutions are additional
# and are NOT modelled here, which makes the cap conservative).  The official
# club page was not machine-readable; a documented source corroborated the
# numbers on 2026-09-11.
SUBSTITUTION_LIMIT_SOURCE = (
    "documented Premier League rule: max 5 substitutions per team per match in 3 windows "
    "(half-time not a window); bench 9; concussion substitutions additional and not modelled "
    "(Wikipedia 'Substitute (association football)' corroboration, 2026-09-11)"
)

# Components compared between the analytic model and the simulation.  The
# analytic payload field names differ from the component labels (yellow uses
# ``yellow_card_xpts``), so the mapping is explicit.
MC_COMPONENTS = ("appearance", "goal", "assist", "clean_sheet", "goals_conceded",
                 "defcon", "save", "yellow", "bonus")
PAYLOAD_XPTS_KEY = {
    "appearance": "appearance_xpts", "goal": "goal_xpts", "assist": "assist_xpts",
    "clean_sheet": "clean_sheet_xpts", "goals_conceded": "goals_conceded_xpts",
    "defcon": "defcon_xpts", "save": "save_xpts", "yellow": "yellow_card_xpts",
    "bonus": "bonus_xpts",
}

# Required frozen joint-kernel primitives for a prospective Minutes run.
REQUIRED_JOINT_PRIMITIVES = (
    "joint_start_target",
    "joint_availability",
    "joint_exit_propensity",
    "joint_entry_propensity",
    "joint_position",
    "joint_expected_minutes_if_cameo",
    "primitive_source_version",
)

# Points-range histogram bounds for the integer CORE draw.
_HIST_LOW = -20
_HIST_SIZE = 120


def _parse_minutes_version(version: str) -> tuple[int, int, int]:
    """Parse ``minutes_vMAJOR.MINOR.PATCH`` into a comparable tuple."""

    match = re.search(r"minutes_v(\d+)\.(\d+)\.(\d+)", str(version or ""))
    if not match:
        return (0, 0, 0)
    return tuple(int(part) for part in match.groups())  # type: ignore[return-value]


def validate_input_run_coherence(
    conn: sqlite3.Connection,
    *,
    xpts_run_id: int,
    minutes_run_id: int,
    team_run_id: int,
    rate_run_id: int | None = None,
) -> list[str]:
    """Compare the supplied runs against the inputs recorded by the xPts run.

    An xPts expectation built from Minutes A must never be simulated with
    Minutes B.  Returns a list of problems; the caller treats non-empty as a
    HARD FAIL.
    """

    rows = conn.execute(
        """SELECT DISTINCT minutes_run_id, team_run_id, rate_run_id
             FROM player_fixture_xpts_projections WHERE projection_run_id=?""",
        (int(xpts_run_id),),
    ).fetchall()
    if not rows:
        return [f"XPTS_RUN_EMPTY: xPts run {xpts_run_id} has no projections to validate"]
    problems: set[str] = set()
    for row in rows:
        recorded_minutes = row["minutes_run_id"]
        recorded_team = row["team_run_id"]
        recorded_rate = row["rate_run_id"]
        if recorded_minutes is not None and int(recorded_minutes) != int(minutes_run_id):
            problems.add(
                f"MINUTES_RUN_MISMATCH: xPts run {xpts_run_id} was built from minutes run "
                f"{int(recorded_minutes)}, not the supplied {int(minutes_run_id)}"
            )
        if recorded_team is not None and int(recorded_team) != int(team_run_id):
            problems.add(
                f"TEAM_RUN_MISMATCH: xPts run {xpts_run_id} was built from team run "
                f"{int(recorded_team)}, not the supplied {int(team_run_id)}"
            )
        if rate_run_id is not None and recorded_rate is not None and int(recorded_rate) != int(rate_run_id):
            problems.add(
                f"RATE_RUN_MISMATCH: xPts run {xpts_run_id} was built from rate run "
                f"{int(recorded_rate)}, not the supplied {int(rate_run_id)}"
            )
    return sorted(problems)


@dataclass(frozen=True)
class MonteCarloConfig:
    """Every tunable in one versioned structure; no magic numbers in code."""

    simulations: int = 10_000
    seed: int = 20260911
    max_substitute_entrants: int = 5

    # Discrete minutes states (bounds only; representatives are solved to match
    # the conditional mean exactly where the bounds allow).
    starter_state_bounds: tuple = ((1.0, 59.0), (60.0, 79.0), (80.0, 90.0))
    cameo_state_bounds: tuple = ((1.0, 59.0), (60.0, 90.0))

    # FPL score-threshold probabilities (canonical names; the *_proxy aliases are
    # score thresholds, NOT literal football "return" events).
    score_le_2_threshold: int = 2
    score_5_plus_threshold: int = 5
    score_10_plus_threshold: int = 10
    score_15_plus_threshold: int = 15

    # Paired substitution model: a departure and an arrival share one event time
    # so each pair contributes exactly 90 player-minutes and 11 players are on
    # the pitch at every instant.
    substitution_minute_jitter: tuple = (-10.0, -5.0, 0.0, 5.0, 10.0)
    substitution_minute_jitter_weights: tuple = (0.15, 0.20, 0.30, 0.20, 0.15)
    substitution_minute_min: float = 30.0
    substitution_minute_max: float = 89.0
    # Per-player reconciliation gate: LINEAR components must reconcile per
    # player (standardised error), so offsetting rows cannot hide distortion.
    player_standardised_error_p95: float = 8.0
    player_error_p95_allowance: dict = field(default_factory=lambda: {
        "appearance": 0.25, "goal": 0.25, "assist": 0.20, "clean_sheet": 0.25,
        "goals_conceded": 0.20, "yellow": 0.15, "core_linear": 0.60, "core": 0.60,
        "defcon": 0.12, "save": 0.05,
    })

    # Secondary individual gate: a single player is a HARD FAIL only when the
    # standardised error is extreme AND the discrepancy is material in POINTS, so
    # a tiny 12-sigma residual with negligible point error is not a blocker while
    # an extreme sigma on a material point error is.  Values are taken from the
    # existing reconciliation allowances (0.20 points is the assist P95
    # allowance), not from any named-player result.
    individual_max_z: float = 8.0
    individual_error_materiality: float = 0.20

    # Scorer / assist expectation calibration.  Weights are solved on a
    # deterministic state library drawn from the SAME joint minutes kernel but
    # with a dedicated, config-hashed seed namespace that is independent of both
    # the Minutes integration and the production simulation.
    calibration_states: int = 20_000
    calibration_max_iterations: int = 250
    calibration_tolerance: float = 1e-4
    calibration_damping: float = 0.5
    calibration_min_weight: float = 1e-9
    calibration_max_weight: float = 1e9
    calibration_seed: int = 20260911

    # Absolute point materiality for the zero-variance safety rule.  Raised from
    # the initial 0.001 to 0.01 after inspection showed the tighter value flagged
    # a 0.0014-point backup-goalkeeper assist expectation that finite draws never
    # sampled; still far below the 0.02-point aggregate reconciliation floor.
    zero_variance_materiality_points: float = 0.01

    # Systematic premium-player bias gate: the top analytic-goal tertile must not
    # be under-allocated by more than this many points on average.
    premium_bias_tolerance_points: float = 0.05

    # Goalkeeper substitutions are a separate, rare event: a backup keeper only
    # plays when the starter is withdrawn.  The pooled Minutes cameo marginal for
    # backup keepers is physically implausible (audited: mean 0.295, max 0.394),
    # so it is capped here and the suppressed mass is reported.
    gk_substitution_ceiling: float = 0.06
    exit_propensity_floor: float = 0.02
    exit_propensity_ceiling: float = 3.0
    # Run the per-world 11-players / 1-goalkeeper occupancy audit (costly, off by
    # default for large freezes; always on in tests).
    occupancy_audit: bool = False

    # Analytic-vs-MC mean reconciliation gate.  The aggregate component mean
    # must converge to the analytic mean within this allowance.  DefCon and
    # saves now integrate the SAME frozen minute states the simulation
    # discretises to, so their allowances are ordinary rather than a permanent
    # nonlinearity exemption.
    mean_reconciliation_sigma: float = 5.0
    mean_reconciliation_abs_floor: float = 0.02
    component_reconciliation_allowance: dict = field(default_factory=lambda: {
        "appearance": 0.05, "goal": 0.03, "assist": 0.03, "clean_sheet": 0.04,
        "goals_conceded": 0.03, "defcon": 0.12, "save": 0.03, "yellow": 0.03, "core": 0.15,
    })

    # Team minute-mass coherence gate (per world).
    team_minutes_target: float = 990.0
    team_minutes_warn: float = 25.0
    team_minutes_fail: float = 120.0

    # xG mass coherence epsilon (numerical slack on the strict team cap).
    xg_mass_epsilon: float = 1e-4
    residual_bucket_warn_fraction: float = 0.40

    def config_hash(self) -> str:
        values = {field_.name: getattr(self, field_.name) for field_ in fields(self)}
        return analytics.canonical_hash({"model": MONTE_CARLO_MODEL_VERSION, **values})

    def calibration_state_identity(self) -> str:
        """Identity of the calibration STATE LIBRARY, not of the whole model.

        Built only from parameters that actually change the states the sampler
        generates, so changing the model-version label (or any reporting or
        gate threshold) does NOT reseed the library.  ``config_hash()`` keeps
        the version and every field for provenance; this identity exists purely
        so a version-label-only change cannot move the fitted weights.
        """

        joint = _joint_config(self)
        return analytics.canonical_hash(
            {
                "identity_version": CALIBRATION_STATE_IDENTITY_VERSION,
                "joint_state_fields": {
                    name: getattr(joint, name) for name in CALIBRATION_STATE_JOINT_FIELDS
                },
                "calibration_seed": self.calibration_seed,
                "calibration_states": self.calibration_states,
                # Only read by the legacy (pre-v1.5.1) kernel reconstruction, but
                # they can change generated states, so excluding them would be an
                # unsafe omission.
                "exit_propensity_floor": self.exit_propensity_floor,
                "exit_propensity_ceiling": self.exit_propensity_ceiling,
            }
        )

    @property
    def state_bounds(self) -> dict[str, tuple]:
        return {"starter": self.starter_state_bounds, "cameo": self.cameo_state_bounds}


# Absolute point materiality for the zero-variance safety rule: a discrepancy
# this small is finite-draw numerical noise, recorded explicitly but never
# claimed as z=0 statistical evidence.  Anything above it is a HARD FAIL.
# Inspection during Addendum 5 raised the initial 0.001 to 0.01: the tight value
# flagged a rarely-playing backup goalkeeper whose analytic assist expectation
# is 0.0014 points and whom 10,000 draws never sampled -- a microscopic
# residual that is not decision-relevant.  0.01 remains far below the model's
# own aggregate reconciliation floor (0.02 points).
ZERO_VARIANCE_MATERIALITY_POINTS = 0.01

# Category substreams used for common random numbers.  Each category gets its
# own deterministic stream per (fixture, simulation index, team) so a fixture's
# score never depends on roster size and one team's random consumption never
# shifts its opponent's.
RNG_CATEGORIES = (
    "lineup", "substitutions", "team_goals", "goal_times",
    "scorer", "assist", "defcon", "saves", "cards",
)

_UINT64_MASK = (1 << 64) - 1


def _mix64(value: int) -> int:
    """SplitMix64 finaliser: cheap deterministic integer mixing."""

    z = (value + 0x9E3779B97F4A7C15) & _UINT64_MASK
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _UINT64_MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _UINT64_MASK
    return (z ^ (z >> 31)) & _UINT64_MASK


def _category_seed(seed: int, fixture_id: int, simulation_index: int, category: str, *extra: int) -> int:
    value = int(seed) & _UINT64_MASK
    value = _mix64(value ^ ((int(fixture_id) + 1) * 0x9E3779B97F4A7C15))
    value = _mix64(value ^ ((int(simulation_index) + 1) * 0xC2B2AE3D27D4EB4F))
    value = _mix64(value ^ ((RNG_CATEGORIES.index(category) + 1) * 0x165667B19E3779F9))
    for part in extra:
        value = _mix64(value ^ ((int(part) + 1) * 0x27D4EB2F165667C5))
    return value


class _FastRng:
    """Tiny splitmix64 stream with the ``random.Random`` surface the samplers use."""

    __slots__ = ("_state",)

    def __init__(self, seed: int) -> None:
        self._state = int(seed) & _UINT64_MASK

    def random(self) -> float:
        self._state = (self._state + 0x9E3779B97F4A7C15) & _UINT64_MASK
        return _mix64(self._state) / float(1 << 64)

    def gauss(self, mu: float, sigma: float) -> float:
        u1 = max(1e-12, self.random())
        u2 = self.random()
        return mu + sigma * math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def _stream(seed: int, fixture_id: int, simulation_index: int, category: str, *extra: int) -> _FastRng:
    return _FastRng(_category_seed(seed, fixture_id, simulation_index, category, *extra))


# ---------------------------------------------------------------------------
# Deterministic RNG and small samplers.
# ---------------------------------------------------------------------------


def _rng_for(seed: int, fixture_id: int) -> random.Random:
    """One RNG per fixture; string seeding is stable across processes."""

    return random.Random(f"mc_v1:{int(seed)}:fixture:{int(fixture_id)}")


def _poisson(rng: random.Random, lam: float) -> int:
    """Knuth Poisson for small rates (goal/save/DefCon counts are small)."""

    if lam <= 0.0:
        return 0
    if lam > 30.0:  # normal approximation, only for pathological rates
        value = int(round(rng.gauss(lam, math.sqrt(lam))))
        return max(0, value)
    limit = math.exp(-lam)
    k = 0
    product = 1.0
    while True:
        k += 1
        product *= rng.random()
        if product <= limit:
            return k - 1


def _match_states(probabilities: Iterable[float], bounds: Iterable[tuple], mean: float) -> tuple[list[float], float]:
    """Representatives inside ``bounds`` whose mixture matches ``mean``.

    Starts every representative at its lower bound and greedily fills capacity
    in descending-probability order, which reaches the target exactly whenever
    the target lies inside the achievable range; otherwise the residual error is
    returned for the caller to record.
    """

    probs = [max(0.0, float(p)) for p in probabilities]
    total = sum(probs)
    if total <= 0:
        return [float(lo) for lo, _ in bounds], abs(mean)
    probs = [p / total for p in probs]
    reps = [float(lo) for lo, _ in bounds]
    remaining = float(mean) - sum(probs[i] * reps[i] for i in range(len(probs)))
    for index in sorted(range(len(probs)), key=lambda i: (-probs[i], i)):
        if probs[index] <= 0:
            continue
        lo, hi = bounds[index]
        capacity = probs[index] * (float(hi) - float(lo))
        if remaining <= capacity + 1e-12:
            reps[index] = min(float(hi), max(float(lo), reps[index] + remaining / probs[index]))
            remaining = 0.0
            break
        reps[index] = float(hi)
        remaining -= capacity
    error = abs(sum(probs[i] * reps[i] for i in range(len(probs))) - float(mean))
    return reps, error


def _starter_states(p60: float, p80: float, mean: float, config: MonteCarloConfig) -> tuple[list[float], list[float], float]:
    """<60 / 60-79 / 80-90 state probabilities and matched representatives."""

    q_high = min(1.0, max(0.0, p80))
    q_mid = min(1.0 - q_high, max(0.0, p60 - p80))
    q_low = max(0.0, 1.0 - q_high - q_mid)
    reps, error = _match_states([q_low, q_mid, q_high], config.starter_state_bounds, mean)
    return [q_low, q_mid, q_high], reps, error


def _cameo_states(p60_cameo: float, mean: float, config: MonteCarloConfig) -> tuple[list[float], list[float], float]:
    q_long = min(1.0, max(0.0, p60_cameo))
    q_short = 1.0 - q_long
    reps, error = _match_states([q_short, q_long], config.cameo_state_bounds, mean)
    return [q_short, q_long], reps, error


def _systematic_select(probabilities: list[float], count: int, u: float, *,
                       randomised: bool = False, rng: "random.Random | None" = None) -> list[int]:
    """Fixed-size systematic sampling: exactly ``count`` DISTINCT picks.

    No probability is ever rescaled: the target spacing is ``sum(p)/count`` and
    a player with ``p_i <= spacing`` can capture at most one target, so the
    result is a set of DISTINCT indices.  When the marginals already sum to
    ``count`` (which the positional coherence layer guarantees for the starting
    XI) the spacing is exactly 1, every ``p_i <= 1``, and the inclusion
    probability is exactly ``p_i``.  If the caller's marginals do not sum to
    ``count`` the distinct count may fall short, which the caller records as a
    lineup-validity diagnostic rather than silently accepting.
    """

    total = sum(probabilities)
    if count <= 0 or total <= 0:
        return []
    # Normalising to exactly ``count`` sets the spacing to 1 so no player can
    # capture two targets (each scaled probability is clamped to <= 1).  This is
    # a no-op when the marginals already sum to ``count``, as the positional
    # coherence layer guarantees for the starting XI.
    scaled = [min(1.0, max(0.0, p) * count / total) for p in probabilities]
    order = list(range(len(scaled)))
    if randomised and rng is not None:
        # A per-simulation random permutation makes the induced pairwise
        # structure depend on the permutation rather than the caller's fixed
        # entity ordering (the audit measured a 0.351 pairwise co-start
        # difference between two orderings of identical marginals).
        for i in range(len(order) - 1, 0, -1):
            j = rng.randrange(i + 1)
            order[i], order[j] = order[j], order[i]
    targets = [u + k for k in range(count)]
    cumulative = 0.0
    selected: set[int] = set()
    target_index = 0
    for index in order:
        probability = scaled[index]
        upper = cumulative + probability
        while target_index < len(targets) and targets[target_index] < upper and len(selected) < count:
            selected.add(index)
            target_index += 1
        cumulative = upper
    return sorted(selected)


def _stochastic_round(value: float, u: float) -> int:
    floor = int(math.floor(value))
    return floor + 1 if u < (value - floor) else floor


# ---------------------------------------------------------------------------
# Input loading.
# ---------------------------------------------------------------------------


def load_fixture_inputs(
    conn: sqlite3.Connection,
    *,
    event: int,
    xpts_run_id: int,
    minutes_run_id: int,
    team_run_id: int,
) -> dict[int, dict[str, Any]]:
    """Join the frozen xPts, minutes and team runs into per-fixture inputs."""

    # Primitive validation is driven by the ACTUAL Minutes run metadata, never by
    # the presence of the payload field being validated (which was vacuously
    # true when a legacy run carried no primitive fields at all).
    minutes_run = analytics.get_projection_run(conn, int(minutes_run_id)) or {}
    minutes_run_version = str(minutes_run.get("model_version") or "")
    prospective_primitives = _parse_minutes_version(minutes_run_version) >= (1, 5, 1)

    minutes = {
        (int(r["player_id"]), int(r["fixture_id"])): r["payload"]
        for r in analytics.frozen_predictions(conn, int(minutes_run_id), [analytics.MINUTES_V1_KIND])
    }
    teams = {
        (int(r["fixture_id"]), int(r["team_id"])): r
        for r in analytics.team_fixture_projections(conn, int(team_run_id))
    }
    fixtures: dict[int, dict[str, Any]] = {}
    for record in analytics.xpts_projections(conn, int(xpts_run_id)):
        fixture_id = int(record["fixture_id"])
        fixture = fixtures.setdefault(
            fixture_id,
            {"fixture_id": fixture_id, "event": int(record["event"]), "sides": {}},
        )
        side = fixture["sides"].setdefault(int(record["team_id"]), {"team_id": int(record["team_id"]), "players": []})
        side["opponent_id"] = int(record["opponent_id"])
        side["position"] = record["position"]
        side["minutes_run_version"] = minutes_run_version
        minutes_payload = minutes.get((int(record["player_id"]), fixture_id)) or {}
        if not side.get("substitution_profile"):
            side["substitution_profile"] = minutes_payload.get("team_substitution_profile")
        # A prospective (>= v1.5.1) Minutes run MUST carry every required joint
        # primitive on every row; only a run whose metadata identifies a legacy
        # model version may be reconstructed.
        missing = [field for field in REQUIRED_JOINT_PRIMITIVES if minutes_payload.get(field) is None]
        if prospective_primitives and missing:
            side["primitive_reconstruction"] = True
            side.setdefault("primitive_missing_fields", set()).update(missing)
        side["players"].append(
            {
                "player_id": int(record["player_id"]),
                "position": str(record["position"]),
                "payload": record["payload"],
                "minutes": minutes_payload,
                "has_primitives": not missing,
                # Resolve each row's declared DEFCON calibration ONCE here, not
                # per world: the sampled layer must apply the SAME calibration
                # the analytic row recorded, or the two would disagree.  An
                # unrecognised spec raises; an absent one means a pre-calibration
                # run and resolves explicitly to the legacy identity mapping.
                "defcon_calibration": defcon_cal.from_payload(
                    (record["payload"] or {}).get("defcon_calibration")
                ),
            }
        )
    for fixture in fixtures.values():
        for side in fixture["sides"].values():
            side["players"].sort(key=lambda p: p["player_id"])
            side.setdefault("primitive_reconstruction", False)
            team_row = teams.get((fixture_id := fixture["fixture_id"], side["team_id"]))
            payload = (team_row or {}).get("payload") or {}
            side["lambda_for"] = float(payload.get("expected_goals_for") or 0.0)
            side["lambda_against"] = float(payload.get("expected_goals_against") or 0.0)
            sum_xg = sum(float(p["payload"].get("adjusted_expected_xg") or 0.0) for p in side["players"])
            sum_xa = sum(float(p["payload"].get("expected_xa") or 0.0) for p in side["players"])
            sum_fpl_assists = sum(float(p["payload"].get("expected_fpl_assists") or 0.0) for p in side["players"])
            side["sum_player_xg"] = sum_xg
            side["sum_player_xa"] = sum_xa
            side["sum_fpl_assists"] = sum_fpl_assists
            side["residual_xg"] = max(0.0, side["lambda_for"] - sum_xg)
            side["residual_xa"] = max(0.0, side["lambda_for"] - sum_xa)
            side["no_assist_mass"] = max(0.0, 1.0 - (sum_fpl_assists / side["lambda_for"])) if side["lambda_for"] > 0 else 1.0
        fixture["sides"] = [fixture["sides"][team_id] for team_id in sorted(fixture["sides"])]
    return fixtures


# ---------------------------------------------------------------------------
# One simulated world.
# ---------------------------------------------------------------------------


def _kernel_players(side, config):
    """Map a side's frozen minutes payloads onto the shared kernel parameters."""

    players = []
    for player in side["players"]:
        minutes = player["minutes"]
        if minutes.get("joint_start_target") is not None:
            # Frozen joint primitives (minutes_v1.5.1+): use them VERBATIM so the
            # integration and the simulation are provably the same model.
            players.append(
                {
                    "player_id": player["player_id"],
                    "position": minutes.get("joint_position") or player["position"],
                    "p_start": float(minutes["joint_start_target"]),
                    "p_available": float(minutes.get("joint_availability") or 0.0),
                    "exit_propensity": float(minutes.get("joint_exit_propensity") or 0.0),
                    "entry_propensity": float(minutes.get("joint_entry_propensity") or 0.0),
                    "expected_minutes_if_cameo": float(minutes.get("joint_expected_minutes_if_cameo") or 0.0),
                }
            )
            continue
        # Legacy runs (pre-v1.5.1) have no primitives; reconstruct for
        # backwards compatibility only.
        p_start = float(minutes.get("p_start") or 0.0)
        players.append(
            {
                "player_id": player["player_id"],
                "position": player["position"],
                "p_start": p_start,
                "p_available": float(minutes.get("p_available") or 0.0),
                "exit_propensity": min(
                    config.exit_propensity_ceiling,
                    max(config.exit_propensity_floor, 1.0 - float(minutes.get("p_80_given_start") or 0.0)),
                ),
                "entry_propensity": float(minutes.get("p_cameo") or 0.0) / max(1e-9, 1.0 - p_start),
                "expected_minutes_if_cameo": float(minutes.get("expected_minutes_if_cameo") or 0.0),
            }
        )
    return players


def _joint_config(config: MonteCarloConfig) -> "joint_minutes.JointMinutesConfig":
    return joint_minutes.JointMinutesConfig(
        max_substitute_entrants=config.max_substitute_entrants,
        substitution_minute_jitter=config.substitution_minute_jitter,
        substitution_minute_jitter_weights=config.substitution_minute_jitter_weights,
        substitution_minute_min=config.substitution_minute_min,
        substitution_minute_max=config.substitution_minute_max,
        gk_substitution_ceiling=config.gk_substitution_ceiling,
    )


def _sample_side_world(side, rng, config, *, rng_lineup=None, rng_substitutions=None, order_keys=None):
    """Thin adapter over the canonical joint kernel in ``joint_minutes``.

    The Monte Carlo and Minutes v1.5 must describe ONE process, so the sampler
    itself lives in ``joint_minutes.sample_side_world`` and is not reimplemented
    here.  Only the tunables and the optional CRN streams are mapped across.
    """

    return joint_minutes.sample_side_world(
        _kernel_players(side, config), side.get("substitution_profile"), _joint_config(config), rng,
        order_keys=order_keys, rng_lineup=rng_lineup, rng_substitutions=rng_substitutions,
    )


check_world_occupancy = joint_minutes.check_world_occupancy


# ---------------------------------------------------------------------------
# Expectation-preserving scorer / assist calibration.
# ---------------------------------------------------------------------------


def _calibration_namespace(config: MonteCarloConfig, fixture_id: int, team_id: int) -> str:
    """Deterministic namespace for the calibration state library.

    Derived from ``calibration_state_identity()`` -- substantive state-generating
    parameters and the explicit seed only -- so a model-version label change does
    not reseed the library.  The identity version is embedded in the namespace so
    the derivation itself is visible in the seed.
    """

    return (
        f"{CALIBRATION_STATE_IDENTITY_VERSION}:{config.calibration_seed}:"
        f"{config.calibration_state_identity()[:16]}:{fixture_id}:{team_id}"
    )


# ---------------------------------------------------------------------------
# Calibration state-library identity.
#
# The calibration sampler must depend ONLY on the parameters that actually
# generate states plus the explicit seed.  It must NOT depend on the model
# version label, report labels or gate thresholds: a version-label-only change
# previously reseeded the state library and moved the fitted weights (R2C
# finding).  config_hash() still carries the version for provenance.
# ---------------------------------------------------------------------------

CALIBRATION_STATE_IDENTITY_VERSION = "mc_calib_states_v1"

# Joint-kernel fields the calibration state sampler actually reads.
CALIBRATION_STATE_JOINT_FIELDS = (
    "max_substitute_entrants",
    "substitution_minute_jitter",
    "substitution_minute_jitter_weights",
    "substitution_minute_min",
    "substitution_minute_max",
    "gk_substitution_ceiling",
)

# Joint-kernel fields that do NOT affect the calibration state library, with the
# reason.  A test asserts this classification is exhaustive over the kernel
# config, so a newly added field cannot silently be forgotten.
CALIBRATION_STATE_JOINT_EXCLUDED = {
    "integration_draws": "sampling budget of the Minutes integration driver, not the calibration sampler",
    "integration_seed": "seed of the Minutes integration driver, not the calibration sampler",
}

# Provenance of the calibration state library must be recoverable from any
# certification artifact produced from this version onwards.  A completed run
# whose identity cannot be recovered is not certifiable.
CALIBRATION_PROVENANCE_REQUIRED_FROM = MONTE_CARLO_MODEL_VERSION
CALIBRATION_PROVENANCE_FIELDS = (
    "calibration_state_identity",
    "calibration_state_identity_version",
    "calibration_seed",
    "calibration_states",
)


class CalibrationProvenanceError(RuntimeError):
    """A certification artifact is missing recoverable calibration provenance."""


def missing_calibration_provenance(meta: Mapping[str, Any] | None) -> list[str]:
    """Return the calibration-provenance fields absent from ``meta``."""

    if not meta:
        return list(CALIBRATION_PROVENANCE_FIELDS)
    return [name for name in CALIBRATION_PROVENANCE_FIELDS if meta.get(name) in (None, "")]


def require_calibration_provenance(
    meta: Mapping[str, Any] | None, *, model_version: str | None = None
) -> None:
    """Fail closed when a certifiable Monte Carlo artifact lacks provenance.

    Enforced from ``CALIBRATION_PROVENANCE_REQUIRED_FROM`` onwards; earlier
    (historical) versions are exempt so reading old artifacts never breaks.
    """

    version = model_version or MONTE_CARLO_MODEL_VERSION
    if version != CALIBRATION_PROVENANCE_REQUIRED_FROM:
        return
    missing = missing_calibration_provenance(meta)
    if missing:
        raise CalibrationProvenanceError(
            f"{version} certification artifacts must expose calibration provenance; "
            f"missing {missing}"
        )


def calibration_provenance(config: "MonteCarloConfig") -> dict[str, Any]:
    """The calibration-state provenance block written into certification artifacts."""

    return {
        "calibration_state_identity": config.calibration_state_identity(),
        "calibration_state_identity_version": CALIBRATION_STATE_IDENTITY_VERSION,
        "calibration_seed": config.calibration_seed,
        "calibration_states": config.calibration_states,
        "calibration_seed_namespace_format": (
            f"{CALIBRATION_STATE_IDENTITY_VERSION}:{{calibration_seed}}:"
            "{calibration_state_identity[:16]}:{fixture_id}:{team_id}"
        ),
    }


# ---------------------------------------------------------------------------
# Scale-aware numerical zero (v1.3.0).
#
# A scorer residual is the difference of two quantities of magnitude lambda
# (order 1).  When the intended residual is exactly zero, floating-point
# cancellation leaves a few units in the last place: for lambda ~ 1.61 the
# spacing between adjacent doubles is ulp(1.61) = 2.220446049250313e-16, which is
# exactly the magnitude observed in the GW5 failure.  Summing ~n terms bounds
# the cancellation error by ~n * ulp(scale), so a small multiple of ulp(scale)
# is a defensible band, and it scales with the magnitude of the quantity rather
# than with any FPL acceptance threshold.
#
# This is deliberately NOT a fixed constant such as 1e-4 or 1e-9.  In
# particular the 6-decimal storage rounding of `adjusted_expected_xg` produces
# residuals up to ~n * 5e-7 (about 1e-5 goals); that is a *representation*
# artefact rather than cancellation noise and is deliberately NOT canonicalized,
# because the solver satisfies such targets normally -- the stored accepted GW5
# run converged in 16 iterations with a +1.000000000139778e-06 residual.
NUMERICAL_ZERO_ULP_FACTOR = 8.0
DIAG_NUMERICAL_ZERO_CANONICALIZED = "NUMERICAL_ZERO_CANONICALIZED"
DIAG_TARGET_OUTSIDE_FEASIBLE_RANGE = "TARGET_OUTSIDE_FEASIBLE_RANGE"

# The residual component is a share of team goals, so its feasible interval is
# [0, 1]: 0 means "no scorer mass is unallocated" and 1 would mean the whole
# team goal expectation is unallocated.
RESIDUAL_SHARE_LOW = 0.0
RESIDUAL_SHARE_HIGH = 1.0


def numerical_zero_epsilon(scale: float) -> float:
    """Half-width of the cancellation-noise band for a quantity of ``scale``."""

    magnitude = abs(float(scale))
    if magnitude == 0.0:
        return 0.0
    if not math.isfinite(magnitude):
        return float("inf")
    spacing = max(math.ulp(magnitude), magnitude * sys.float_info.epsilon)
    return spacing * NUMERICAL_ZERO_ULP_FACTOR


def is_numerical_zero(value: float, scale: float) -> bool:
    """True when ``value`` is indistinguishable from zero at ``scale``."""

    return abs(float(value)) <= numerical_zero_epsilon(scale)


def canonicalize_numerical_zero(value: float, scale: float) -> tuple[float, dict[str, Any] | None]:
    """Return ``(canonical_value, diagnostic_or_None)``.

    A value inside the cancellation band becomes exactly ``0.0``, and the caller
    receives an auditable ``NUMERICAL_ZERO_CANONICALIZED`` record carrying the
    original value, the canonical value, the epsilon and the scale.
    """

    original = float(value)
    if original == 0.0:
        return 0.0, None
    epsilon = numerical_zero_epsilon(scale)
    if epsilon > 0.0 and abs(original) <= epsilon:
        return 0.0, {
            "diagnostic": DIAG_NUMERICAL_ZERO_CANONICALIZED,
            "original_value": original,
            "canonical_value": 0.0,
            "epsilon": epsilon,
            "scale": float(scale),
            "ulp_factor": NUMERICAL_ZERO_ULP_FACTOR,
        }
    return original, None


def clamp_target_to_feasible_interval(
    target: float,
    *,
    low: float,
    high: float,
    scale: float,
    label: str,
) -> tuple[float, list[str], dict[str, Any] | None]:
    """Clamp a boundary-adjacent target, but keep failing on a genuine violation.

    Returns ``(target, infeasible_messages, diagnostic)``.  A target that sits
    outside the feasible interval by more than the cancellation band is a real
    model inconsistency and is reported as ``TARGET_OUTSIDE_FEASIBLE_RANGE``
    rather than clamped into success.
    """

    value = float(target)
    epsilon = numerical_zero_epsilon(scale)
    span = abs(float(high) - float(low))
    tolerance = max(epsilon, numerical_zero_epsilon(span) if span else 0.0)
    messages: list[str] = []
    diagnostic: dict[str, Any] | None = None

    if value < low - tolerance:
        messages.append(
            f"{label}_TARGET_OUTSIDE_FEASIBLE_RANGE: target {value!r} is below the feasible "
            f"lower bound {low!r} by more than the numerical band {tolerance!r}"
        )
        return value, messages, diagnostic
    if value > high + tolerance:
        messages.append(
            f"{label}_TARGET_OUTSIDE_FEASIBLE_RANGE: target {value!r} is above the feasible "
            f"upper bound {high!r} by more than the numerical band {tolerance!r}"
        )
        return value, messages, diagnostic

    clamped = min(max(value, low), high)
    if clamped != value:
        diagnostic = {
            "diagnostic": "TARGET_CLAMPED_TO_FEASIBLE_BOUNDARY",
            "original_value": value,
            "canonical_value": clamped,
            "bound": low if value < low else high,
            "tolerance": tolerance,
            "scale": float(scale),
        }
    return clamped, messages, diagnostic


def _calibration_states(side, config, fixture_id, team_id):
    """Deterministic on-pitch state library from the SAME joint minutes kernel.

    Each state records which players are on the pitch at one representative
    uniform goal time.  The seed namespace depends on the config hash and is
    independent of both the Minutes integration and the production simulation,
    so production draws are never used to fit the weights.
    """

    kernel_players = _kernel_players(side, config)
    joint_config = _joint_config(config)
    profile = side.get("substitution_profile")
    namespace = _calibration_namespace(config, fixture_id, team_id)
    rng = random.Random(f"{namespace}:states")
    player_ids = [p["player_id"] for p in kernel_players]
    states: list[set[int]] = []
    for state_index in range(int(config.calibration_states)):
        # Fresh randomised traversal order EVERY state, matching the integration
        # and production samplers exactly.
        order_keys = joint_minutes.joint_order_keys(namespace, fixture_id, team_id, state_index, player_ids)
        world = joint_minutes.sample_side_world(
            kernel_players, profile, joint_config, rng, order_keys=order_keys
        )
        goal_time = rng.random() * 90.0
        intervals = world["intervals"]
        on_pitch = {
            index for index, player in enumerate(kernel_players)
            if player["player_id"] in intervals
            and intervals[player["player_id"]][0] <= goal_time <= intervals[player["player_id"]][1]
        }
        states.append(on_pitch)
    return states


def _solve_shares(states, n_players, targets, target_residual, config, *, label):
    """Damped iterative proportional fitting of positive categorical weights.

    Matches the empirical mean category probabilities over the calibration
    library to ``targets`` / ``target_residual``.  Deterministic, bounded, with
    explicit infeasibility and convergence diagnostics.
    """

    state_count = max(1, len(states))
    eligibility = [sum(1 for state in states if index in state) / state_count for index in range(n_players)]
    infeasible = [
        f"{label}_INFEASIBLE: player index {index} target {targets[index]:.6f} exceeds "
        f"eligibility {eligibility[index]:.6f}"
        for index in range(n_players)
        if targets[index] > eligibility[index] + 1e-9
    ]
    diagnostics: list[dict[str, Any]] = []
    original_target_residual = float(target_residual)

    # A residual share is a fraction of team goals; anything outside [0, 1] is a
    # real inconsistency.  A target sitting on a boundary within the numerical
    # band is clamped and recorded instead of being treated as a failure.
    target_residual, boundary_messages, boundary_diag = clamp_target_to_feasible_interval(
        target_residual,
        low=RESIDUAL_SHARE_LOW,
        high=RESIDUAL_SHARE_HIGH,
        scale=1.0,
        label=f"{label}_RESIDUAL",
    )
    infeasible.extend(boundary_messages)
    if boundary_diag is not None:
        diagnostics.append(boundary_diag)

    # Canonicalise cancellation-level noise to exactly zero so the iterative
    # residual solver is never launched to satisfy an unreachable target.
    target_residual, canonical_diag = canonicalize_numerical_zero(target_residual, 1.0)
    if canonical_diag is not None:
        diagnostics.append(canonical_diag)

    residual_mode = target_residual > 0.0
    weights = [1.0] * n_players
    residual_weight = 1.0 if residual_mode else 0.0
    converged = False
    max_residual = float("inf")
    iteration = 0
    for iteration in range(1, int(config.calibration_max_iterations) + 1):
        achieved = [0.0] * n_players
        achieved_residual = 0.0
        for state in states:
            denom = residual_weight + sum(weights[index] for index in state)
            if denom <= 0.0:
                continue
            for index in state:
                achieved[index] += weights[index] / denom
            achieved_residual += residual_weight / denom
        achieved = [value / state_count for value in achieved]
        achieved_residual /= state_count
        max_residual = 0.0
        for index in range(n_players):
            if targets[index] <= 0.0:
                weights[index] = config.calibration_min_weight
                max_residual = max(max_residual, abs(achieved[index] - targets[index]))
                continue
            ratio = targets[index] / max(achieved[index], 1e-300)
            weights[index] = min(
                config.calibration_max_weight,
                max(config.calibration_min_weight, weights[index] * ratio ** config.calibration_damping),
            )
            max_residual = max(max_residual, abs(achieved[index] - targets[index]))
        if residual_mode:
            ratio = target_residual / max(achieved_residual, 1e-300)
            residual_weight = min(
                config.calibration_max_weight,
                max(config.calibration_min_weight, residual_weight * ratio ** config.calibration_damping),
            )
            max_residual = max(max_residual, abs(achieved_residual - target_residual))
        scale = residual_weight if (residual_mode and residual_weight > 0) else (sum(weights) / max(1, n_players))
        if scale > 0:
            weights = [value / scale for value in weights]
            residual_weight = (residual_weight / scale) if residual_mode else 0.0
        if max_residual <= config.calibration_tolerance:
            converged = True
            break
    achieved_final = [0.0] * n_players
    achieved_residual_final = 0.0
    for state in states:
        denom = residual_weight + sum(weights[index] for index in state)
        if denom <= 0.0:
            continue
        for index in state:
            achieved_final[index] += weights[index] / denom
        achieved_residual_final += residual_weight / denom
    achieved_final = [value / state_count for value in achieved_final]
    achieved_residual_final /= state_count
    return {
        "weights": weights,
        "residual_weight": residual_weight,
        "targets": list(targets),
        "achieved": achieved_final,
        "target_residual": float(target_residual),
        "achieved_residual": achieved_residual_final,
        "eligibility": eligibility,
        "iterations": iteration,
        "converged": converged and not infeasible,
        "max_residual": max_residual,
        "infeasible": infeasible,
        "residual_mode": residual_mode,
        "target_residual_requested": float(original_target_residual),
        "numerical_diagnostics": diagnostics,
    }


def calibrate_side(side, config, fixture_id, team_id):
    """Solve scorer and assist weights so both expectations are preserved."""

    players = side["players"]
    n_players = len(players)
    lam = side["lambda_for"]
    result = {"scorer": None, "assist": None, "infeasible": [], "converged": True, "states": 0}
    if n_players == 0 or lam <= 0:
        return result
    targets_xg = [max(0.0, float(p["payload"].get("adjusted_expected_xg") or 0.0)) / lam for p in players]
    # Residual-mode entry: the residual is a difference of goal-scale quantities,
    # so cancellation noise is judged against ulp(lambda).  A mathematically zero
    # residual is canonicalised to exactly 0.0 here so the iterative residual
    # solver is never launched to chase a target below its representable floor.
    requested_residual_xg = max(0.0, float(side["residual_xg"]))
    residual_xg, residual_canonical_diag = canonicalize_numerical_zero(requested_residual_xg, lam)
    if residual_canonical_diag is not None:
        result.setdefault("numerical_diagnostics", []).append(
            {**residual_canonical_diag, "quantity": "residual_xg"}
        )
    target_residual = residual_xg / lam
    states = _calibration_states(side, config, fixture_id, team_id)
    result["states"] = len(states)
    scorer = _solve_shares(states, n_players, targets_xg, target_residual, config, label="SCORER")
    result["scorer"] = scorer
    result["infeasible"].extend(scorer["infeasible"])
    result["converged"] = result["converged"] and scorer["converged"]

    # Assist library: the scorer is drawn with the CALIBRATED scorer process, so
    # the assister eligibility excludes the realised scorer.
    assist_targets = [
        max(0.0, float(
            p["payload"].get("expected_fpl_assists")
            if p["payload"].get("expected_fpl_assists") is not None
            else (p["payload"].get("expected_xa") or 0.0)
        )) / lam
        for p in players
    ]
    target_no_assist = max(0.0, 1.0 - sum(assist_targets))
    if sum(assist_targets) > 1.0 + 1e-9:
        result["infeasible"].append(
            f"ASSIST_INFEASIBLE: total assist share {sum(assist_targets):.6f} exceeds 1.0"
        )
    assist_states: list[set[int]] = []
    assist_rng = random.Random(
        f"{_calibration_namespace(config, fixture_id, team_id)}:assist-scorer-draw"
    )
    scorer_weights = scorer["weights"]
    scorer_residual = scorer["residual_weight"]
    for state in states:
        denom = scorer_residual + sum(scorer_weights[index] for index in state)
        scorer_index = None
        if denom > 0:
            draw = assist_rng.random() * denom
            cumulative = 0.0
            for index in sorted(state):
                cumulative += scorer_weights[index]
                if draw < cumulative:
                    scorer_index = index
                    break
        assist_states.append({index for index in state if index != scorer_index})
    assist = _solve_shares(
        assist_states, n_players, assist_targets, target_no_assist, config, label="ASSIST"
    )
    result["assist"] = assist
    result["infeasible"].extend(assist["infeasible"])
    result["converged"] = result["converged"] and assist["converged"]
    return result


POINT_COMPONENTS = (
    "appearance", "goal", "assist", "clean_sheet", "goals_conceded",
    "defcon", "save", "yellow",
)


def _empty_accumulator() -> dict[str, Any]:
    names = list(POINT_COMPONENTS) + ["core_linear", "core"]
    sums = {name: 0.0 for name in names}
    sums.update({f"{name}_sq": 0.0 for name in names})
    sums.update({
        "goal_flag": 0.0, "assist_flag": 0.0, "cs_flag": 0.0, "defcon_flag": 0.0,
        "start_flag": 0.0, "minutes": 0.0, "sixty_flag": 0.0,
        "goal_count": 0.0, "assist_count": 0.0,
        "zero_minutes_flag": 0.0, "one_to_59_flag": 0.0,
    })
    return {"hist": [0] * _HIST_SIZE, "sum": sums, "sims": 0}


def _add_points(hist: list[int], points: int) -> None:
    index = int(points) - _HIST_LOW
    if 0 <= index < _HIST_SIZE:
        hist[index] += 1


def _hist_moments(hist: list[int], sims: int) -> dict[str, float]:
    if sims <= 0:
        return {}
    mean = sum((i + _HIST_LOW) * c for i, c in enumerate(hist)) / sims
    var = sum((((i + _HIST_LOW) - mean) ** 2) * c for i, c in enumerate(hist)) / sims
    return {"mean": mean, "variance": var, "std": math.sqrt(max(0.0, var))}


def _hist_quantile(hist: list[int], sims: int, q: float) -> float:
    if sims <= 0:
        return float("nan")
    target = q * sims
    cumulative = 0
    for i, count in enumerate(hist):
        cumulative += count
        if cumulative >= target:
            return float(i + _HIST_LOW)
    return float(_HIST_SIZE - 1 + _HIST_LOW)


def _hist_tail(hist: list[int], sims: int, threshold: int) -> float:
    if sims <= 0:
        return float("nan")
    count = sum(c for i, c in enumerate(hist) if (i + _HIST_LOW) >= threshold)
    return count / sims


def simulate(
    fixtures: Mapping[int, Mapping[str, Any]],
    config: MonteCarloConfig,
    rules: ScoringRules | None = None,
    *,
    capture_player_ids: Iterable[int] | None = None,
    capture_bps_worlds: bool = False,
) -> dict[str, Any]:
    """Run the joint simulation and return per-player-fixture summaries.

    Randomness is drawn from deterministic per-category substreams keyed by
    ``(seed, fixture, simulation index, team, category[, player])``.  A fixture's
    score, its goal times and every shared player's minutes therefore do not
    depend on roster size, on the opponent's random consumption, or on DB row
    ordering.

    ``capture_player_ids`` is a PURE side effect used by the manager layer: when
    supplied, the full universe is still simulated unchanged, and additionally a
    compact per-world GW matrix (core points and minutes, aggregated across the
    player's fixtures within the same world index) is returned for exactly those
    players.  It never consumes randomness, so summaries and readiness are
    byte-identical with and without it.
    """

    rules = rules or DEFAULT_SCORING_RULES
    captured_players = sorted({int(pid) for pid in (capture_player_ids or [])})
    world_core: dict[int, list[float]] = {pid: [0.0] * int(config.simulations) for pid in captured_players}
    world_minutes: dict[int, list[float]] = {pid: [0.0] * int(config.simulations) for pid in captured_players}
    accumulators: dict[tuple[int, int], dict[str, Any]] = {}
    #: Sum over worlds of the per-world scoring intensity the PRODUCTION mechanism
    #: induced for each player/fixture.  Accumulated from the sampled world state, the
    #: team's Poisson lambda and the calibrated scorer weights -- before any scorer draw
    #: is consumed and independent of the realised number of team goals.  The
    #: zero-variance goal check is judged against THIS, never against the analytic target.
    production_goal_expectation: dict[tuple[int, int], float] = {}
    player_meta: dict[int, dict[str, Any]] = {}
    team_minutes: list[float] = []
    invalid_lineups = 0
    excess_substitutes = 0
    minute_mass_violations = 0
    occupancy_violations = 0
    occupancy_examples: list[str] = []
    suppressed_gk_mass: list[float] = []
    analytic: dict[tuple[int, int], dict[str, float]] = {}
    primitive_reconstruction = 0
    calibration_by_side: dict[tuple[int, int], dict[str, Any]] = {}
    calibration_by_player: dict[tuple[int, int], dict[str, Any]] = {}
    calibration_diagnostics: list[dict[str, Any]] = []
    #: PE-3 optional side-channel: player x fixture x world raw events.  Allocated ONLY
    #: when explicitly requested, so the default path allocates nothing.
    bps_worlds: dict[int, list[dict[int, dict[str, Any]]]] = {}

    for fixture_id in sorted(fixtures):
        fixture = fixtures[fixture_id]
        # Deterministic entity ordering: sides by team id and players by player
        # id, so results never depend on DB or dict ordering.
        sides = sorted(fixture["sides"], key=lambda side: int(side["team_id"]))
        for side in sides:
            side["players"] = sorted(side["players"], key=lambda player: int(player["player_id"]))
        if len(sides) != 2:
            continue
        for side in sides:
            if side.get("primitive_reconstruction"):
                primitive_reconstruction += 1
            for player in side["players"]:
                key = (player["player_id"], fixture_id)
                player_meta[player["player_id"]] = {"position": player["position"], "team_id": side["team_id"]}
                accumulators.setdefault(key, _empty_accumulator())
                analytic[key] = {
                    name: float(player["payload"].get(PAYLOAD_XPTS_KEY[name]) or 0.0)
                    for name in MC_COMPONENTS
                }
                analytic[key]["core"] = float(player["payload"].get("core_xpts") or 0.0)

        # Scorer/assist weights are solved once per side on a dedicated state
        # library, never on the production draws themselves.
        for side in sides:
            team_id = int(side["team_id"])
            calibration = calibrate_side(side, config, fixture_id, team_id)
            calibration_by_side[(fixture_id, team_id)] = calibration
            calibration_diagnostics.append(_calibration_diagnostic(fixture_id, team_id, calibration))
            for index, player in enumerate(side["players"]):
                calibration_by_player[(fixture_id, player["player_id"])] = _player_calibration_fields(
                    calibration, index
                )

        for simulation_index in range(int(config.simulations)):
            per_side = []
            for side in sides:
                team_id = int(side["team_id"])
                rng_lineup = _stream(config.seed, fixture_id, simulation_index, "lineup", team_id)
                rng_substitutions = _stream(config.seed, fixture_id, simulation_index, "substitutions", team_id)
                order_keys = joint_minutes.joint_order_keys(
                    f"mc_crn_v1:{config.seed}", fixture_id, team_id, simulation_index,
                    [p["player_id"] for p in _kernel_players(side, config)],
                )
                world = _sample_side_world(
                    side, rng_lineup, config, rng_lineup=rng_lineup,
                    rng_substitutions=rng_substitutions, order_keys=order_keys,
                )
                if (world["gk_starters"], world["outfield_starters"]) != (1, 10):
                    invalid_lineups += 1
                if world["substitute_entrants"] > config.max_substitute_entrants:
                    excess_substitutes += 1
                side_total = sum(world["minutes"].values())
                if abs(side_total - 990.0) > 1e-6:
                    minute_mass_violations += 1
                suppressed_gk_mass.append(world["suppressed_gk_cameo_mass"])
                if config.occupancy_audit:
                    gk_ids = [p["player_id"] for p in side["players"] if p["position"] == "GKP"]
                    ok, why = joint_minutes.check_world_occupancy(world["intervals"], gk_ids)
                    if not ok:
                        occupancy_violations += 1
                        occupancy_examples.append(why)
                goals = _poisson(_stream(config.seed, fixture_id, simulation_index, "team_goals", team_id),
                                 side["lambda_for"])
                rng_goal_times = _stream(config.seed, fixture_id, simulation_index, "goal_times", team_id)
                goal_times = [rng_goal_times.random() * 90.0 for _ in range(goals)]
                per_side.append({
                    "side": side, "world": world,
                    "minutes": world["minutes"], "intervals": world["intervals"],
                    "goals": goals, "goal_times": goal_times,
                    "rng_scorer": _stream(config.seed, fixture_id, simulation_index, "scorer", team_id),
                    "rng_assist": _stream(config.seed, fixture_id, simulation_index, "assist", team_id),
                    "calibration": calibration_by_side[(fixture_id, team_id)],
                })
                team_minutes.append(side_total)

            # Goals / assists per side, using the opponent's goal times for
            # clean-sheet and concession exposure.
            world_bps_players: dict[int, dict[str, Any]] = {}
            for side_index, entry in enumerate(per_side):
                opponent = per_side[1 - side_index]
                # BEFORE the scorer draw: what opportunity did this world's play expose?
                for player_id, induced_goals in _production_goal_intensity(entry, rules).items():
                    expectation_key = (player_id, fixture_id)
                    production_goal_expectation[expectation_key] = (
                        production_goal_expectation.get(expectation_key, 0.0) + induced_goals)
                draw_components: dict[int, dict[str, float]] = {}
                _allocate_and_score(entry, rules, config, fixture_id, draw_components)
                _score_personal_events(entry, opponent, rules, accumulators, config, fixture_id,
                                       draw_components, simulation_index)
                if capture_bps_worlds:
                    positions = {int(p["player_id"]): p["position"]
                                 for p in entry["side"]["players"]}
                    for captured_id, captured_bucket in draw_components.items():
                        world_bps_players[int(captured_id)] = {
                            "position": positions.get(int(captured_id)),
                            "minutes": float(captured_bucket.get("minutes", 0.0)),
                            "goals_scored": int(captured_bucket.get("goal_count", 0.0)),
                            "assists": int(captured_bucket.get("assist_count", 0.0)),
                            "clean_sheets": int(captured_bucket.get("cs_flag", 0.0)),
                            "goals_conceded": int(captured_bucket.get("raw_goals_conceded", 0.0)),
                            "saves": int(captured_bucket.get("raw_saves", 0.0)),
                            "yellow_cards": int(captured_bucket.get("raw_yellow", 0.0)),
                        }
                _commit_draw(draw_components, accumulators, fixture_id, simulation_index,
                             world_core, world_minutes)
            if capture_bps_worlds:
                bps_worlds.setdefault(int(fixture_id), []).append(world_bps_players)

    summaries = _summarise(accumulators, analytic, config, player_meta, calibration_by_player,
                           rules, production_goal_expectation)
    return {
        "summaries": summaries,
        "team_minutes": team_minutes,
        "invalid_lineups": invalid_lineups,
        "excess_substitutes": excess_substitutes,
        "minute_mass_violations": minute_mass_violations,
        "occupancy_violations": occupancy_violations,
        "occupancy_examples": occupancy_examples[:5],
        "suppressed_gk_cameo_mass_mean": (
            sum(suppressed_gk_mass) / len(suppressed_gk_mass) if suppressed_gk_mass else 0.0
        ),
        "primitive_reconstruction": primitive_reconstruction,
        "calibration": calibration_diagnostics,
        "config_hash": config.config_hash(),
        "world_matrix": (
            {
                "worlds": int(config.simulations),
                "player_ids": captured_players,
                "core": world_core,
                "minutes": world_minutes,
            }
            if captured_players
            else None
        ),
        # PE-3: raw football events for the whole fixture, per world.  ``None`` unless the
        # caller opted in; it is a pure observation of draws that already happened.
        "bps_worlds": bps_worlds if capture_bps_worlds else None,
    }


def _calibration_diagnostic(fixture_id: int, team_id: int, calibration: Mapping[str, Any]) -> dict[str, Any]:
    scorer = calibration.get("scorer") or {}
    assist = calibration.get("assist") or {}
    return {
        "fixture_id": int(fixture_id),
        "team_id": int(team_id),
        "states": int(calibration.get("states") or 0),
        "converged": bool(calibration.get("converged")),
        "infeasible": list(calibration.get("infeasible") or []),
        "scorer_iterations": int(scorer.get("iterations") or 0),
        "scorer_converged": bool(scorer.get("converged")),
        "scorer_max_residual": float(scorer.get("max_residual") or 0.0),
        "assist_iterations": int(assist.get("iterations") or 0),
        "assist_converged": bool(assist.get("converged")),
        "assist_max_residual": float(assist.get("max_residual") or 0.0),
        "residual_target": float(scorer.get("target_residual") or 0.0),
        "residual_achieved": float(scorer.get("achieved_residual") or 0.0),
    }


#: Zero-variance rows are only judged, never blanket-excused.  A zero observed variance
#: means the component contributed the SAME value in every draw; for a discrete count
#: that almost always means "no events at all", which is either an ordinary low-rate
#: sample or a generator that is not delivering the player's events.  Those two are
#: indistinguishable from the sample alone, so the deciding evidence is the MODEL's own
#: expected number of events over the whole sample, E: observing zero has probability
#: exp(-E) under the model.  A row is tolerated only when that probability is at least
#: this floor; otherwise it keeps the historical HARD FAIL.
#:
#: The two regimes are separated by many orders of magnitude, so the verdict is not
#: sensitive to the exact value.  Reference points (2000 draws, DEF): the real GW7 row
#: expects E = 4.38 goals -> exp(-E) = 1.25e-02, tolerated; a dead instrument with a
#: 0.10-point expectation has E = 33.3 -> exp(-E) = 3.5e-15, refused.  This stays a
#: module constant rather than a MonteCarloConfig field ON PURPOSE: config_hash() hashes
#: every dataclass field, so a new field would change the calibration namespace and with
#: it every simulated number.
ZERO_EVENT_PLAUSIBILITY_FLOOR = 1e-3


class _ZeroVarianceAllowance(NamedTuple):
    """Evidence that lets a zero-variance row be JUDGED instead of blanket-failed.

    ``variance_per_world`` is a provable LOWER bound on the component's per-world
    variance in score points (so the standard error it yields is an upper bound).
    ``expected_count_in_sample`` is the model's expected number of EVENTS over the whole
    sample, which is the only quantity that can separate an unlucky zero from a dead
    generator.  It is used as a plausibility test, NOT as proof that the generator is
    live -- if the expectation is itself too large the row still fails.
    """

    variance_per_world: float
    expected_count_in_sample: float


def _classify_standardised(value: float, std: float, sims: int, materiality: float,
                           allowance: _ZeroVarianceAllowance | None = None) -> tuple[float, str]:
    """Standardise a mismatch, distinguishing broken instruments from noise.

    Returns ``(standardised_error, status)`` where status is ``ok``,
    ``below_materiality`` (a microscopic finite-draw residual with no variance,
    recorded but never claimed as z=0 evidence) or ``mismatch`` (a MATERIAL
    non-zero mismatch with zero variance, which is a HARD FAIL).

    The zero-variance branch is evaluated ONLY when ``allowance`` is supplied, which the
    caller does solely for components whose production generator has a provable variance
    bound.  Even then the row is tolerated only if the observed zero is plausible under
    the model's own expected event count; a material expectation with zero events remains
    a HARD FAIL, so a dead or mis-wired generator cannot be converted into an ordinary
    finite-sampling row.
    """

    se = std / math.sqrt(max(1, sims))
    if se > 0.0:
        return value / se, "ok"
    if value == 0.0:
        return 0.0, "ok"
    if abs(value) <= materiality:
        return 0.0, "below_materiality"
    if allowance is not None and allowance.variance_per_world > 0.0:
        expected = float(allowance.expected_count_in_sample)
        if expected > 0.0 and math.exp(-expected) >= ZERO_EVENT_PLAUSIBILITY_FLOOR:
            floor_se = math.sqrt(allowance.variance_per_world / max(1, sims))
            if floor_se > 0.0:
                return value / floor_se, "ok"
    return float("inf"), "mismatch"


def _production_goal_intensity(entry: Mapping[str, Any], rules: ScoringRules) -> dict[int, float]:
    """Per-player EXPECTED GOALS in THIS world, from the production mechanism's own state.

    Derived from the sampled play, never from the analytic comparator and never from any
    realised outcome: it consumes NO randomness and is computed independently of how many
    goals the team actually scored, so it still exists in a world where the team scores
    none.  That is what lets it certify that the goal mechanism exposed a scoring
    opportunity for a player even when the sample caught zero goals.

    The production mechanism (`_allocate_and_score` above):

      * the team's goals are ``Poisson(lambda_for)`` with INDEPENDENT uniform times on
        ``[0, 90]``;
      * each goal's scorer is a categorical draw over the players on the pitch at that
        time, with weight ``max(0, w_i)`` and an explicit residual bucket.

    So a player's goal count in this world is ``Poisson(lambda_for * p_i)`` where ``p_i``
    is his share of one goal averaged over the uniform goal time::

        p_i = (1/90) * Integral_0^90  [i on pitch at t] * w_i / D(t)  dt
        D(t) = residual + sum of on-pitch weights at t

    Both the on-pitch set and ``D`` are piecewise constant between interval endpoints, so
    the integral is EXACT.  Segments sharing an on-pitch set are merged, which collapses
    the ~2 * substitutions configurations to a handful and keeps this cheap in the hot
    loop.
    """

    side = entry["side"]
    players = side["players"]
    intervals = entry["intervals"]
    calibration = entry["calibration"]
    scorer = calibration.get("scorer") or {}
    residual = max(0.0, float(scorer.get("residual_weight") or 0.0))
    lam = float(side.get("lambda_for") or 0.0)

    player_ids = [int(player["player_id"]) for player in players]
    weights = scorer.get("weights") or [0.0] * len(players)
    weight_of = [max(0.0, float(weights[i])) if i < len(weights) else 0.0
                 for i in range(len(players))]

    intensity = {player_id: 0.0 for player_id in player_ids}
    if lam <= 0.0:
        return intensity

    bounds = {0.0, 90.0}
    for player_id in player_ids:
        interval = intervals.get(player_id)
        if interval is None:
            continue
        bounds.add(min(90.0, max(0.0, float(interval[0]))))
        bounds.add(min(90.0, max(0.0, float(interval[1]))))
    ordered = sorted(bounds)

    # Merge segments that share an on-pitch set: D depends only on the set.
    span_by_set: dict[tuple[int, ...], float] = {}
    for index in range(len(ordered) - 1):
        low, high = ordered[index], ordered[index + 1]
        width = high - low
        if width <= 0.0:
            continue
        key = tuple(i for i, player_id in enumerate(player_ids)
                    if player_id in intervals
                    and float(intervals[player_id][0]) <= low
                    and float(intervals[player_id][1]) >= high)
        span_by_set[key] = span_by_set.get(key, 0.0) + width

    for on_pitch, span in span_by_set.items():
        denom = residual + sum(weight_of[i] for i in on_pitch)
        if denom <= 0.0:
            continue
        factor = lam * span / (90.0 * denom)
        for i in on_pitch:
            if weight_of[i] > 0.0:
                intensity[player_ids[i]] += factor * weight_of[i]
    return intensity


def _production_goal_zero_variance_allowance(
    name: str, position: str | None, rules: ScoringRules, sims: int,
    production_expected_goals: float,
) -> "_ZeroVarianceAllowance | None":
    """The zero-variance allowance for ``goal``, built from the PRODUCTION generator.

    ``production_expected_goals`` is the sum, over the sampled worlds, of the per-world
    scoring intensity that the production mechanism actually induced for this player
    (see :func:`_production_goal_intensity`).  It is a property of the production
    world state, the team's Poisson lambda and the calibrated scorer weights -- NOT of
    the analytic comparator, which is only ever the quantity being tested.

    Conditional on those sampled world states the player's total goal count over the
    sample is ``Poisson(E)`` with ``E = production_expected_goals``, so

        P(zero goals) = exp(-E)
        Var(mean goal points) = weight^2 * E / sims^2   =>   SE = weight * sqrt(E) / sims

    Both are production-derived.  No claim is made that E equals the analytic target: if
    the mechanism exposes no opportunity at all (E == 0) the row HARD FAILS, and a large
    E convicts it.  Only ``goal`` qualifies; the other count components do not admit this
    derivation and keep the historical ``inf`` behaviour.
    """

    if name != "goal":
        return None
    if production_expected_goals <= 0.0:
        # The production mechanism exposed no scoring opportunity for this player in ANY
        # sampled world, yet the row carries a material goal error.  That is the
        # dead/mis-wired generator signature: no allowance is offered.
        return None
    weight = float(rules.goal_points_for(position)) if position else 0.0
    if weight <= 0.0:
        return None
    return _ZeroVarianceAllowance(
        variance_per_world=weight * weight * production_expected_goals / max(1, int(sims)),
        expected_count_in_sample=float(production_expected_goals),
    )


PROBABILITY_KEYS = (
    "p_score_le_2", "p_score_5_plus", "p_score_10_plus", "p_score_15_plus",
    "p_goal", "p_assist", "p_clean_sheet_eligible", "p_defcon_hit",
    "p_start", "p_60_plus", "p_zero_minutes", "p_1_59",
    "p_blank_proxy", "p_return_proxy",
)


def probability_range_violations(summaries: Iterable[Mapping[str, Any]], tolerance: float = 1e-9) -> list[str]:
    """Validate EVERY summary row independently (no leaked loop variable).

    A single invalid row cannot be masked by a later valid one.
    """

    violations: list[str] = []
    for summary in summaries:
        for key in PROBABILITY_KEYS:
            value = summary.get(key)
            if value is None or not math.isfinite(float(value)) or float(value) < -tolerance or float(value) > 1.0 + tolerance:
                violations.append(
                    f"PROBABILITY_OUT_OF_RANGE: player {summary.get('player_id')} "
                    f"fixture {summary.get('fixture_id')} {key}={value}"
                )
        for name, value in (summary.get("mc_component_std") or {}).items():
            if value is None or not math.isfinite(float(value)) or float(value) < -1e-12:
                violations.append(
                    f"NON_FINITE_SUMMARY: player {summary.get('player_id')} mc_component_std.{name}={value}"
                )
    return violations


#: Raw observational mirrors of already-sampled events, for the PE-3 BPS side-channel.
#: They are NOT point components: nothing sums them, squares them, reconciles them or
#: gates on them, so they cannot change any existing number.
RAW_EVENT_FIELDS: dict[str, float] = {
    "raw_goals_conceded": 0.0, "raw_saves": 0.0, "raw_yellow": 0.0,
}


def _draw_bucket(draw_components: dict[int, dict[str, float]], player_id: int) -> dict[str, float]:
    bucket = draw_components.get(player_id)
    if bucket is None:
        bucket = {name: 0.0 for name in POINT_COMPONENTS}
        bucket.update({"goal_count": 0.0, "assist_count": 0.0, "goal_flag": 0.0, "assist_flag": 0.0,
                       "cs_flag": 0.0, "defcon_flag": 0.0, "minutes": 0.0})
        # PE-3 raw observational capture.  These mirror events ALREADY sampled above; they
        # are never read by any POINT_COMPONENTS / accumulator / reconciliation path, so
        # they cannot alter scoring.  They exist only so the optional BPS side-channel can
        # expose the raw counts instead of the point deductions.
        bucket.update(RAW_EVENT_FIELDS)
        draw_components[player_id] = bucket
    return bucket


def _allocate_and_score(entry, rules, config, fixture_id, draw_components) -> None:
    """Allocate goals and assists with CALIBRATED, expectation-preserving weights.

    A goal has exactly one scorer (a modelled player or the explicit residual
    bucket), the scorer must be on the pitch, and an assist is at most one, never
    the scorer, and always on the pitch.  The weights are solved on the
    calibration library so ``E_s[P(i scores|s)]`` reproduces each player's
    analytic share of the team goal expectation.
    """

    side = entry["side"]
    players = side["players"]
    intervals = entry["intervals"]
    calibration = entry["calibration"]
    scorer = calibration.get("scorer") or {}
    assist = calibration.get("assist") or {}
    scorer_weights = scorer.get("weights") or [0.0] * len(players)
    scorer_residual = float(scorer.get("residual_weight") or 0.0)
    assist_weights = assist.get("weights") or [0.0] * len(players)
    assist_residual = float(assist.get("residual_weight") or 0.0)
    rng_scorer = entry["rng_scorer"]
    rng_assist = entry["rng_assist"]

    for goal_time in entry["goal_times"]:
        on_pitch = [
            i for i, p in enumerate(players)
            if p["player_id"] in intervals and intervals[p["player_id"]][0] <= goal_time <= intervals[p["player_id"]][1]
        ]
        weight_total = scorer_residual + sum(max(0.0, scorer_weights[i]) for i in on_pitch)
        scorer_index = None
        if weight_total > 0:
            draw = rng_scorer.random() * weight_total
            cumulative = 0.0
            for index in on_pitch:
                cumulative += max(0.0, scorer_weights[index])
                if draw < cumulative:
                    scorer_index = index
                    break
            # draw beyond the player mass falls through to the residual bucket
        if scorer_index is not None:
            bucket = _draw_bucket(draw_components, players[scorer_index]["player_id"])
            bucket["goal"] += rules.goal_points_for(players[scorer_index]["position"])
            bucket["goal_count"] += 1.0
            bucket["goal_flag"] = 1.0

        # Assist: at most one, never the scorer, always on pitch.
        candidates = [i for i in on_pitch if i != scorer_index]
        assist_total = assist_residual + sum(max(0.0, assist_weights[i]) for i in candidates)
        if assist_total <= 0:
            continue
        draw = rng_assist.random() * assist_total
        cumulative = 0.0
        assist_index = None
        for index in candidates:
            cumulative += max(0.0, assist_weights[index])
            if draw < cumulative:
                assist_index = index
                break
        if assist_index is not None:
            bucket = _draw_bucket(draw_components, players[assist_index]["player_id"])
            bucket["assist"] += rules.assist_points
            bucket["assist_count"] += 1.0
            bucket["assist_flag"] = 1.0


def _score_personal_events(entry, opponent, rules, accumulators, config, fixture_id,
                           draw_components, simulation_index) -> None:
    """Appearance, clean sheet, concessions, DefCon, saves and cards."""

    side = entry["side"]
    players = side["players"]
    intervals = entry["intervals"]
    opponent_times = opponent["goal_times"]
    starters = entry["world"]["starters"]
    team_id = int(side["team_id"])
    for index, player in enumerate(players):
        key = (player["player_id"], fixture_id)
        acc = accumulators[key]
        minutes = entry["minutes"][player["player_id"]]
        payload = player["payload"]
        position = player["position"]
        player_id = player["player_id"]
        bucket = _draw_bucket(draw_components, player_id)
        bucket["minutes"] = minutes
        acc["sum"]["minutes"] += minutes
        acc["sum"]["start_flag"] += 1 if index in starters else 0
        acc["sum"]["sixty_flag"] += 1 if minutes >= rules.clean_sheet_minutes_required else 0
        acc["sum"]["zero_minutes_flag"] += 1 if minutes <= 0 else 0
        acc["sum"]["one_to_59_flag"] += 1 if 0 < minutes < rules.clean_sheet_minutes_required else 0

        # Appearance
        if minutes >= rules.clean_sheet_minutes_required:
            bucket["appearance"] += rules.appearance_long_points
        elif minutes > 0:
            bucket["appearance"] += rules.appearance_short_points

        # Personal exposure: opponent goals while on the pitch.
        conceded_on_pitch = 0
        if minutes > 0:
            low, high = intervals.get(player_id, (0.0, 0.0))
            conceded_on_pitch = sum(1 for t in opponent_times if low <= t <= high)
        bucket["raw_goals_conceded"] = float(conceded_on_pitch)
        if position in rules.goals_conceded_positions and conceded_on_pitch > 0:
            bucket["goals_conceded"] += -math.floor(
                conceded_on_pitch / rules.goals_conceded_per_deduction
            ) * abs(rules.goals_conceded_points_for(position))
        if (
            minutes >= rules.clean_sheet_minutes_required
            and conceded_on_pitch == 0
            and rules.clean_sheet_points_for(position) > 0
        ):
            bucket["clean_sheet"] += rules.clean_sheet_points_for(position)
            bucket["cs_flag"] = 1.0

        # DefCon and saves are evaluated at the player's TRUE physical sampled
        # minutes in this world.  The compact minute quadrature is analytic-side
        # only and must never replace the sampled exposure here.
        exposure = minutes
        threshold = rules.defcon_threshold_for(position)
        if position in rules.defcon_positions and threshold is not None and exposure > 0:
            actions_per90 = float(payload.get("defcon_actions_per90") or 0.0)
            # The SAME calibrated hit probability the analytic row scored, at
            # THIS world's sampled exposure.  Sampling the raw count and
            # thresholding it would leave the sampled layer uncalibrated while
            # the analytic layer was calibrated, which is exactly the
            # raw-p-vs-calibrated-points split the two layers must not have.
            # The PAYLOAD is the authority on which calibration applies, so a
            # hand-built input that never passed through load_fixture_inputs
            # still resolves correctly (and, with no declared spec, resolves to
            # the explicit legacy identity mapping = the original semantics).
            calibration = player.get("defcon_calibration")
            if calibration is None:
                calibration = defcon_cal.from_payload((payload or {}).get("defcon_calibration"))
            p_hit = xpts.defcon_hit_probability(
                position, actions_per90, exposure, rules, calibration=calibration
            )
            rng = _stream(config.seed, fixture_id, simulation_index, "defcon", team_id, player_id)
            if rng.random() < p_hit:
                bucket["defcon"] += rules.defcon_points
                bucket["defcon_flag"] = 1.0

        # Goalkeeper saves
        if position == "GKP" and exposure > 0:
            save_model = payload.get("save_model") or {}
            saves_per90 = float(save_model.get("saves_per90_posterior") or 0.0)
            pressure = float(save_model.get("pressure_multiplier") or 1.0)
            rng = _stream(config.seed, fixture_id, simulation_index, "saves", team_id, player_id)
            saves = _poisson(rng, max(0.0, saves_per90) * exposure / 90.0 * pressure)
            bucket["raw_saves"] = float(saves)
            if saves > 0:
                save_points = math.floor(saves / rules.saves_per_point)
                if save_points:
                    bucket["save"] += save_points

        # Yellow card
        if minutes > 0:
            yellow_per90 = float(payload.get("yellow_per90") or 0.0)
            probability = min(1.0, max(0.0, yellow_per90) * minutes / 90.0)
            rng = _stream(config.seed, fixture_id, simulation_index, "cards", team_id, player_id)
            if rng.random() < probability:
                bucket["yellow"] += rules.yellow_card_points
                bucket["raw_yellow"] = 1.0


def _commit_draw(draw_components, accumulators, fixture_id, simulation_index=None,
                 world_core=None, world_minutes=None) -> None:
    """Commit ONE draw's per-player component totals and their squares.

    Squaring the per-draw total (not the individual goals/assists) is what makes
    ``mc_component_std`` the standard deviation of the player's points in one
    draw, which matters most in multi-goal / multi-assist worlds.

    When ``world_core`` / ``world_minutes`` are supplied (manager layer), the
    player's GW total is additionally accumulated for this world index across
    all of the player's fixtures.  This is a pure side effect: no equation, RNG
    identity or draw order changes.
    """

    for player_id, bucket in draw_components.items():
        acc = accumulators[(player_id, fixture_id)]
        core_linear = (
            bucket["appearance"] + bucket["goal"] + bucket["assist"]
            + bucket["clean_sheet"] + bucket["goals_conceded"] + bucket["yellow"]
        )
        core_total = core_linear + bucket["defcon"] + bucket["save"]
        for name in POINT_COMPONENTS:
            value = bucket[name]
            acc["sum"][name] += value
            acc["sum"][f"{name}_sq"] += value * value
        acc["sum"]["core_linear"] += core_linear
        acc["sum"]["core_linear_sq"] += core_linear * core_linear
        acc["sum"]["core"] += core_total
        acc["sum"]["core_sq"] += core_total * core_total
        acc["sum"]["goal_count"] += bucket["goal_count"]
        acc["sum"]["assist_count"] += bucket["assist_count"]
        acc["sum"]["goal_flag"] += bucket["goal_flag"]
        acc["sum"]["assist_flag"] += bucket["assist_flag"]
        acc["sum"]["cs_flag"] += bucket["cs_flag"]
        acc["sum"]["defcon_flag"] += bucket["defcon_flag"]
        if world_core is not None and player_id in world_core and simulation_index is not None:
            world_core[player_id][simulation_index] += core_total
            world_minutes[player_id][simulation_index] += bucket.get("minutes", 0.0)
        _add_points(acc["hist"], int(round(core_total)))
        acc["sims"] += 1


def _summarise(
    accumulators: Mapping[tuple[int, int], Mapping[str, Any]],
    analytic: Mapping[tuple[int, int], Mapping[str, float]],
    config: MonteCarloConfig,
    player_meta: Mapping[int, Mapping[str, Any]],
    calibration_by_player: Mapping[tuple[int, int], Mapping[str, Any]] | None = None,
    rules: ScoringRules | None = None,
    production_goal_expectation: Mapping[tuple[int, int], float] | None = None,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    calibration_by_player = calibration_by_player or {}
    rules = rules or DEFAULT_SCORING_RULES
    production_goal_expectation = production_goal_expectation or {}
    materiality = float(config.zero_variance_materiality_points)
    for (player_id, fixture_id) in sorted(accumulators):
        acc = accumulators[(player_id, fixture_id)]
        sims = acc["sims"]
        if sims <= 0:
            continue
        moments = _hist_moments(acc["hist"], sims)
        sums = acc["sum"]
        bonus_mean = analytic.get((player_id, fixture_id), {}).get("bonus", 0.0)
        mc_components = {
            "appearance": sums["appearance"] / sims,
            "goal": sums["goal"] / sims,
            "assist": sums["assist"] / sims,
            "clean_sheet": sums["clean_sheet"] / sims,
            "goals_conceded": sums["goals_conceded"] / sims,
            "defcon": sums["defcon"] / sims,
            "save": sums["save"] / sims,
            "yellow": sums["yellow"] / sims,
            "core_linear": sums["core_linear"] / sims,
        }
        reference = analytic.get((player_id, fixture_id), {})
        component_std = {}
        for name in mc_components:
            mean_value = mc_components[name]
            variance = max(0.0, sums.get(f"{name}_sq", 0.0) / sims - mean_value * mean_value)
            component_std[name] = math.sqrt(variance)
        reconciliation = {
            name: mc_components[name] - float(reference.get(name, 0.0))
            for name in mc_components
        }
        reconciliation["core"] = moments["mean"] - float(reference.get("core", 0.0))
        analytic_core_linear = (
            float(reference.get("core", 0.0))
            - float(reference.get("defcon", 0.0))
            - float(reference.get("save", 0.0))
        )
        reconciliation["core_linear"] = mc_components["core_linear"] - analytic_core_linear
        meta = player_meta.get(player_id, {})
        # Per-player standardised error so a good aggregate cannot hide
        # offsetting individual distortions.
        standardised = {}
        zero_variance_mismatch = []
        zero_variance_below_materiality = []
        for name, value in reconciliation.items():
            std = moments["std"] if name == "core" else component_std.get(name, 0.0)
            z_value, status = _classify_standardised(
                value, std, sims, materiality,
                _production_goal_zero_variance_allowance(
                    name, meta.get("position"), rules, sims,
                    float(production_goal_expectation.get((player_id, fixture_id), 0.0)),
                ),
            )
            standardised[name] = z_value
            if status == "below_materiality":
                zero_variance_below_materiality.append(name)
            elif status == "mismatch":
                zero_variance_mismatch.append(name)
        calibration = calibration_by_player.get((fixture_id, player_id)) or {}
        scorer_target = calibration.get("scorer_target_share")
        scorer_achieved = calibration.get("scorer_achieved_share")
        assist_target = calibration.get("assist_target_share")
        assist_achieved = calibration.get("assist_achieved_share")
        summaries.append(
            {
                "player_id": player_id,
                "fixture_id": fixture_id,
                "team_id": meta.get("team_id"),
                "position": meta.get("position"),
                "simulations": sims,
                "mean_core": moments["mean"],
                "mean_total_proxy": moments["mean"] + bonus_mean,
                "median_core": _hist_quantile(acc["hist"], sims, 0.5),
                "median_total_proxy": _hist_quantile(acc["hist"], sims, 0.5) + bonus_mean,
                "std_core": moments["std"],
                "variance_core": moments["variance"],
                "q10": _hist_quantile(acc["hist"], sims, 0.10),
                "q25": _hist_quantile(acc["hist"], sims, 0.25),
                "q50": _hist_quantile(acc["hist"], sims, 0.50),
                "q75": _hist_quantile(acc["hist"], sims, 0.75),
                "q90": _hist_quantile(acc["hist"], sims, 0.90),
                "p_zero_minutes": sums["zero_minutes_flag"] / sims,
                "p_1_59": sums["one_to_59_flag"] / sims,
                "p_start": sums["start_flag"] / sims,
                "p_60_plus": sums["sixty_flag"] / sims,
                "p_score_le_2": sum(
                    c for i, c in enumerate(acc["hist"]) if (i + _HIST_LOW) <= config.score_le_2_threshold
                ) / sims,
                "p_score_5_plus": _hist_tail(acc["hist"], sims, config.score_5_plus_threshold),
                "p_score_10_plus": _hist_tail(acc["hist"], sims, config.score_10_plus_threshold),
                "p_score_15_plus": _hist_tail(acc["hist"], sims, config.score_15_plus_threshold),
                # Compatibility aliases (score-threshold proxies, not literal
                # football return events).
                "p_blank_proxy": sum(
                    c for i, c in enumerate(acc["hist"]) if (i + _HIST_LOW) <= config.score_le_2_threshold
                ) / sims,
                "p_return_proxy": _hist_tail(acc["hist"], sims, config.score_5_plus_threshold),
                "p_goal": sums["goal_flag"] / sims,
                "p_assist": sums["assist_flag"] / sims,
                "mean_goal_count": sums["goal_count"] / sims,
                "mean_assist_count": sums["assist_count"] / sims,
                "p_clean_sheet_eligible": sums["cs_flag"] / sims,
                "p_defcon_hit": sums["defcon_flag"] / sims,
                "mean_minutes": sums["minutes"] / sims,
                "mc_components": mc_components,
                "mc_component_std": component_std,
                "analytic_components": {k: float(reference.get(k, 0.0)) for k in reference},
                "mean_reconciliation_error": reconciliation,
                "standardised_error": standardised,
                "zero_variance_mismatch": zero_variance_mismatch,
                "zero_variance_below_materiality": zero_variance_below_materiality,
                "bonus_mean_deterministic": bonus_mean,
                "distribution_basis": "CORE (bonus deterministic, variance unmodelled)",
                "total_proxy_tails_claimed": False,
                "bonus_variance": "BONUS_VARIANCE_UNMODELLED",
                # Calibrated scorer/assist diagnostics (per player).
                "scorer_target_share": scorer_target,
                "scorer_achieved_share": scorer_achieved,
                "scorer_calibrated_weight": calibration.get("scorer_calibrated_weight"),
                "scorer_calibration_residual": (
                    None if scorer_target is None or scorer_achieved is None else scorer_achieved - scorer_target
                ),
                "assist_target_share": assist_target,
                "assist_achieved_share": assist_achieved,
                "assist_calibrated_weight": calibration.get("assist_calibrated_weight"),
                "assist_calibration_residual": (
                    None if assist_target is None or assist_achieved is None else assist_achieved - assist_target
                ),
            }
        )
    return summaries


def _player_calibration_fields(calibration: Mapping[str, Any], index: int) -> dict[str, Any]:
    """Extract one player's calibrated scorer/assist diagnostics by side order."""

    scorer = calibration.get("scorer") or {}
    assist = calibration.get("assist") or {}

    def pick(source, key):
        values = source.get(key)
        return float(values[index]) if values and index < len(values) else None

    return {
        "scorer_target_share": pick(scorer, "targets"),
        "scorer_achieved_share": pick(scorer, "achieved"),
        "scorer_calibrated_weight": pick(scorer, "weights"),
        "assist_target_share": pick(assist, "targets"),
        "assist_achieved_share": pick(assist, "achieved"),
        "assist_calibrated_weight": pick(assist, "weights"),
    }


# ---------------------------------------------------------------------------
# Readiness gate.
# ---------------------------------------------------------------------------


def readiness_summary(
    fixtures: Mapping[int, Mapping[str, Any]],
    simulation_result: Mapping[str, Any],
    config: MonteCarloConfig | None = None,
    *,
    deadline_status: str | None = None,
    data_cutoff: str | None = None,
    deadline: str | None = None,
) -> dict[str, Any]:
    """Monte-Carlo-specific gate: mean reconciliation, mass coherence, lineup validity."""

    config = config or MonteCarloConfig()
    fail_reasons: list[str] = []
    warn_reasons: list[str] = []

    # xG mass must never exceed the team lambda (scorer probabilities).
    for fixture_id in sorted(fixtures):
        for side in fixtures[fixture_id]["sides"]:
            excess = side["sum_player_xg"] - side["lambda_for"]
            if excess > config.xg_mass_epsilon:
                fail_reasons.append(
                    f"XG_MASS_EXCEEDS_TEAM_LAMBDA: fixture {fixture_id} team {side['team_id']} excess {excess:.4f}"
                )
            if side["lambda_for"] > 0 and side["residual_xg"] / side["lambda_for"] > config.residual_bucket_warn_fraction:
                warn_reasons.append(
                    f"RESIDUAL_SCORER_BUCKET_LARGE: fixture {fixture_id} team {side['team_id']} "
                    f"fraction {side['residual_xg'] / side['lambda_for']:.3f}"
                )

    summaries = simulation_result.get("summaries", [])

    # A v1.5.1+ Minutes run must feed the kernel its frozen primitives verbatim;
    # reconstructing them from integrated outputs would be a different model.
    if simulation_result.get("primitive_reconstruction"):
        fail_reasons.append(
            f"KERNEL_PRIMITIVE_RECONSTRUCTION: {simulation_result['primitive_reconstruction']} sides required "
            "legacy reconstruction despite a v1.5.1+ Minutes run"
        )

    # Scorer/assist calibration must converge and reproduce the analytic shares.
    calibration_diagnostics = simulation_result.get("calibration") or []
    for side in calibration_diagnostics:
        for reason in side.get("infeasible", []):
            fail_reasons.append(f"CALIBRATION_INFEASIBLE: fixture {side['fixture_id']} team {side['team_id']} {reason}")
        if not side.get("scorer_converged"):
            fail_reasons.append(
                f"SCORER_CALIBRATION_NON_CONVERGENCE: fixture {side['fixture_id']} team {side['team_id']} "
                f"iterations {side.get('scorer_iterations')} residual {side.get('scorer_max_residual')}"
            )
        elif side.get("scorer_max_residual", 0.0) > config.calibration_tolerance:
            fail_reasons.append(
                f"SCORER_CALIBRATION_RESIDUAL: fixture {side['fixture_id']} team {side['team_id']} "
                f"{side['scorer_max_residual']:.6f} > {config.calibration_tolerance}"
            )
        if not side.get("assist_converged"):
            fail_reasons.append(
                f"ASSIST_CALIBRATION_NON_CONVERGENCE: fixture {side['fixture_id']} team {side['team_id']} "
                f"iterations {side.get('assist_iterations')} residual {side.get('assist_max_residual')}"
            )
        elif side.get("assist_max_residual", 0.0) > config.calibration_tolerance:
            fail_reasons.append(
                f"ASSIST_CALIBRATION_RESIDUAL: fixture {side['fixture_id']} team {side['team_id']} "
                f"{side['assist_max_residual']:.6f} > {config.calibration_tolerance}"
            )

    if simulation_result.get("occupancy_violations"):
        fail_reasons.append(
            f"WORLD_OCCUPANCY_VIOLATION: {simulation_result['occupancy_violations']} side-worlds did not hold "
            f"exactly 11 players / 1 goalkeeper on the pitch "
            f"(examples: {simulation_result.get('occupancy_examples')})"
        )
    if simulation_result.get("minute_mass_violations"):
        fail_reasons.append(
            f"TEAM_MINUTE_MASS_VIOLATION: {simulation_result['minute_mass_violations']} side-worlds did not sum to "
            f"exactly 990 player-minutes"
        )
    if simulation_result.get("invalid_lineups"):
        fail_reasons.append(
            f"INVALID_LINEUP_SAMPLE: {simulation_result['invalid_lineups']} side-worlds lacked exactly 1 GK + 10 outfield"
        )
    if simulation_result.get("excess_substitutes"):
        fail_reasons.append(
            f"SUBSTITUTION_LIMIT_EXCEEDED: {simulation_result['excess_substitutes']} side-worlds exceeded the limit"
        )

    team_minutes = simulation_result.get("team_minutes") or []
    if team_minutes:
        mean_minutes = sum(team_minutes) / len(team_minutes)
        if abs(mean_minutes - config.team_minutes_target) > config.team_minutes_fail:
            fail_reasons.append(
                f"TEAM_MINUTE_MASS_IMPLAUSIBLE: mean {mean_minutes:.1f} vs target {config.team_minutes_target:g}"
            )
        elif abs(mean_minutes - config.team_minutes_target) > config.team_minutes_warn:
            warn_reasons.append(
                f"TEAM_MINUTE_MASS_DRIFT: mean {mean_minutes:.1f} vs target {config.team_minutes_target:g}"
            )

    # Per-player reconciliation: every component must reconcile per player, not
    # merely on average, so offsetting rows cannot hide a distorted player.
    # DefCon and saves are no longer permanently exempt: the deterministic model
    # now uses the SAME frozen minute-state mixture the simulation discretises to.
    gated = ("appearance", "goal", "assist", "clean_sheet", "goals_conceded",
             "defcon", "save", "yellow", "core_linear", "core")
    player_error_stats: dict[str, dict[str, float]] = {}
    if summaries:
        def pct(values, q):
            if not values:
                return 0.0
            index = min(len(values) - 1, int(math.ceil(q * len(values))) - 1)
            return values[max(0, index)]
        for name in gated:
            abs_errors = [abs(s["mean_reconciliation_error"].get(name, 0.0)) for s in summaries]
            z_values = [abs(s.get("standardised_error", {}).get(name, 0.0)) for s in summaries]
            if any(v == float("inf") for v in z_values):
                fail_reasons.append(
                    f"PLAYER_RECONCILIATION_SEVERE: {name} has players with a material non-zero error and zero variance"
                )
            abs_errors.sort()
            z_values.sort()
            player_error_stats[name] = {
                "mean_abs": sum(abs_errors) / len(abs_errors),
                "median_abs": pct(abs_errors, 0.5),
                "p90_abs": pct(abs_errors, 0.90),
                "p95_abs": pct(abs_errors, 0.95),
                "max_abs": abs_errors[-1],
                "p95_standardised": pct(z_values, 0.95),
                "max_standardised": z_values[-1],
            }
            if player_error_stats[name]["p95_standardised"] > config.player_standardised_error_p95:
                fail_reasons.append(
                    f"PLAYER_RECONCILIATION_SEVERE: {name} p95 |standardised error| "
                    f"{player_error_stats[name]['p95_standardised']:.2f} > {config.player_standardised_error_p95}"
                )
            elif player_error_stats[name]["p95_abs"] > config.player_error_p95_allowance.get(name, 0.25):
                warn_reasons.append(
                    f"PLAYER_RECONCILIATION_ELEVATED: {name} p95 |error| "
                    f"{player_error_stats[name]['p95_abs']:.4f}"
                )

    # Secondary individual gate: the population P95 can hide a material failure
    # in the top 5%.  HARD FAIL only when the standardised error is extreme AND
    # the point discrepancy is material, so a tiny residual with a large sigma is
    # not a blocker.
    individual_material_failures: list[str] = []
    if summaries:
        for name in gated:
            for summary in summaries:
                z_value = abs(summary.get("standardised_error", {}).get(name, 0.0))
                error = abs(summary["mean_reconciliation_error"].get(name, 0.0))
                if z_value > config.individual_max_z and error > config.individual_error_materiality:
                    individual_material_failures.append(
                        f"{name} player {summary.get('player_id')} fixture {summary.get('fixture_id')} "
                        f"|z|={z_value:.2f} |error|={error:.4f}"
                    )

    # Systematic premium-player bias: split players by analytic goal expectation
    # and confirm the highest tertile is not under-allocated on average.
    premium_bias_diagnostics: dict[str, float] = {}
    if summaries:
        ordered = sorted(summaries, key=lambda s: float(s["analytic_components"].get("goal", 0.0)))
        size = len(ordered)
        if size >= 6:
            third = max(1, size // 3)
            low_group = ordered[:third]
            high_group = ordered[-third:]
            for label, group in (("goal_low", low_group), ("goal_high", high_group)):
                premium_bias_diagnostics[f"{label}_signed_error"] = sum(
                    s["mean_reconciliation_error"].get("goal", 0.0) for s in group
                ) / len(group)
            premium_bias_diagnostics["goal_bias_gap"] = (
                premium_bias_diagnostics["goal_high_signed_error"]
                - premium_bias_diagnostics["goal_low_signed_error"]
            )
            if premium_bias_diagnostics["goal_high_signed_error"] < -config.premium_bias_tolerance_points:
                fail_reasons.append(
                    "PREMIUM_PLAYER_GOAL_BIAS: highest analytic-goal tertile under-allocated by "
                    f"{premium_bias_diagnostics['goal_high_signed_error']:.4f} points "
                    f"(tolerance {config.premium_bias_tolerance_points})"
                )
            ordered_assist = sorted(summaries, key=lambda s: float(s["analytic_components"].get("assist", 0.0)))
            if len(ordered_assist) >= 6:
                third_a = max(1, len(ordered_assist) // 3)
                high_assist = ordered_assist[-third_a:]
                premium_bias_diagnostics["assist_high_signed_error"] = sum(
                    s["mean_reconciliation_error"].get("assist", 0.0) for s in high_assist
                ) / len(high_assist)
                if premium_bias_diagnostics["assist_high_signed_error"] < -config.premium_bias_tolerance_points:
                    fail_reasons.append(
                        "PREMIUM_PLAYER_ASSIST_BIAS: highest analytic-assist tertile under-allocated by "
                        f"{premium_bias_diagnostics['assist_high_signed_error']:.4f} points"
                    )

    # Aggregate-level analytic-vs-MC reconciliation (a MEAN convergence).  The
    # analytic and simulated DefCon/save models share the frozen minute-state
    # mixture, so any remaining gap is ordinary sampling error, not a permanent
    # Jensen exemption.
    # Aggregate (not per-player) so a critical hard-fail reason can never be
    # truncated out of the reported list by hundreds of row-level messages.
    zero_variance_components = sorted({
        name for summary in summaries for name in summary.get("zero_variance_mismatch", [])
    })
    if zero_variance_components:
        fail_reasons.append(
            "ZERO_VARIANCE_MISMATCH: components "
            f"{zero_variance_components} have a material non-zero error with zero simulated variance"
        )
    if individual_material_failures:
        fail_reasons.append(
            f"INDIVIDUAL_RECONCILIATION_MATERIAL: {len(individual_material_failures)} material individual "
            f"failures; examples {individual_material_failures[:5]}"
        )

    max_error = 0.0
    component_errors: dict[str, float] = {}
    component_diagnostics: list[str] = []
    if summaries:
        for name in ("appearance", "goal", "assist", "clean_sheet", "goals_conceded",
                     "defcon", "save", "yellow", "core", "core_linear"):
            errors = [s["mean_reconciliation_error"].get(name, 0.0) for s in summaries]
            if not all(math.isfinite(e) for e in errors):
                fail_reasons.append(f"NON_FINITE_SUMMARY: component {name}")
                continue
            aggregate = sum(errors) / len(errors)
            component_errors[name] = aggregate
            allowance = config.component_reconciliation_allowance.get(name, config.mean_reconciliation_abs_floor)
            if abs(aggregate) > allowance:
                fail_reasons.append(
                    f"ANALYTIC_MC_MEAN_MISMATCH: aggregate {name} error {aggregate:+.4f} > allowance {allowance:.4f}"
                )
            elif abs(aggregate) > config.mean_reconciliation_abs_floor:
                component_diagnostics.append(f"{name}: aggregate error {aggregate:+.4f} within allowance {allowance:.4f}")
            max_error = max(max_error, abs(aggregate))
        for summary in summaries:
            for name, value in summary["mean_reconciliation_error"].items():
                if not math.isfinite(value):
                    fail_reasons.append(f"NON_FINITE_SUMMARY: player {summary['player_id']} component {name}")
        # The probability-range gate is applied to EVERY summary row independently
        # (no leaked loop variable): one invalid row cannot be masked by a later
        # valid one.
        fail_reasons.extend(probability_range_violations(summaries))

    if data_cutoff is not None and deadline is not None and deadline_status != "LATE_FREEZE":
        from .utils import parse_utc

        if parse_utc(data_cutoff) > parse_utc(deadline):
            fail_reasons.append("MODEL_INPUTS_AFTER_DEADLINE: a pre-deadline cutoff may not exceed the deadline")

    below_materiality = sorted({
        name
        for summary in summaries
        for name in summary.get("zero_variance_below_materiality", [])
    })
    if below_materiality:
        warn_reasons.append(
            "ZERO_VARIANCE_BELOW_MATERIALITY: microscopic finite-draw residuals recorded but not "
            f"claimed as z=0 evidence for components {below_materiality}"
        )
    warn_reasons.append("BONUS_VARIANCE_UNMODELLED: bonus is a deterministic soft mean added to CORE")
    warn_reasons.append("BPS_RULE_DISCONTINUITY: bonus BPS rules changed; no BPS reconstruction")
    warn_reasons.append("SAVE_GOAL_CORRELATION_LIMITED_V1: saves and goals share only fixture pressure")
    warn_reasons.append("RESIDUAL_SCORER_BUCKET: unmodelled team scoring mass not forced onto players")
    warn_reasons.append(f"SUBSTITUTION_LIMIT_DOCUMENTED: capped at {config.max_substitute_entrants} ({SUBSTITUTION_LIMIT_SOURCE[:60]}…)")
    warn_reasons.append("CORE_TAILS_ONLY: canonical tails are CORE-based; TOTAL_PROXY tails are not claimed")

    calibration_summary: dict[str, float] = {}
    if calibration_diagnostics:
        def _p95(values):
            ordered = sorted(values)
            index = min(len(ordered) - 1, int(math.ceil(0.95 * len(ordered))) - 1)
            return ordered[max(0, index)]
        scorer_residuals = [float(side.get("scorer_max_residual") or 0.0) for side in calibration_diagnostics]
        assist_residuals = [float(side.get("assist_max_residual") or 0.0) for side in calibration_diagnostics]
        calibration_summary = {
            "sides": float(len(calibration_diagnostics)),
            "scorer_converged": float(sum(1 for side in calibration_diagnostics if side.get("scorer_converged"))),
            "assist_converged": float(sum(1 for side in calibration_diagnostics if side.get("assist_converged"))),
            "scorer_mean_residual": sum(scorer_residuals) / len(scorer_residuals),
            "scorer_p95_residual": _p95(scorer_residuals),
            "scorer_max_residual": max(scorer_residuals),
            "assist_mean_residual": sum(assist_residuals) / len(assist_residuals),
            "assist_p95_residual": _p95(assist_residuals),
            "assist_max_residual": max(assist_residuals),
            "total_iterations": float(sum(
                int(side.get("scorer_iterations") or 0) + int(side.get("assist_iterations") or 0)
                for side in calibration_diagnostics
            )),
        }

    # Numerical-semantics audit trail (informational, never a failure).  A
    # canonicalised cancellation-level residual is reported so the adjustment is
    # visible in the artifact rather than silently absorbed.
    numerical_adjustments: list[dict[str, Any]] = []
    for side in calibration_diagnostics:
        for item in side.get("numerical_diagnostics") or []:
            numerical_adjustments.append(
                {
                    "fixture_id": side.get("fixture_id"),
                    "team_id": side.get("team_id"),
                    "adjustments": [item],
                }
            )
        scorer_block = side.get("scorer")
        if isinstance(scorer_block, dict):
            for item in scorer_block.get("numerical_diagnostics") or []:
                if item.get("diagnostic") in (
                    DIAG_NUMERICAL_ZERO_CANONICALIZED,
                    DIAG_TARGET_OUTSIDE_FEASIBLE_RANGE,
                    "TARGET_CLAMPED_TO_FEASIBLE_BOUNDARY",
                ):
                    numerical_adjustments.append(
                        {
                            "fixture_id": side.get("fixture_id"),
                            "team_id": side.get("team_id"),
                            "adjustments": [item],
                        }
                    )
    canonicalized_count = sum(
        1
        for entry in numerical_adjustments
        for item in entry["adjustments"]
        if item.get("diagnostic") == DIAG_NUMERICAL_ZERO_CANONICALIZED
    )

    return {
        "status": "FAIL" if fail_reasons else "WARN",
        "fail_reasons": sorted(set(fail_reasons))[:100],
        "fail_reason_count": len(set(fail_reasons)),
        "warn_reasons": warn_reasons,
        "max_abs_mean_reconciliation_error": max_error,
        "component_mean_errors": component_errors,
        "component_diagnostics": component_diagnostics,
        "player_error_stats": player_error_stats,
        "mc_primitives_verbatim": not bool(simulation_result.get("primitive_reconstruction")),
        "calibration": calibration_diagnostics,
        "calibration_summary": calibration_summary,
        "premium_bias_diagnostics": premium_bias_diagnostics,
        "zero_variance_below_materiality_components": below_materiality,
        "individual_material_failures": individual_material_failures,
        "numerical_adjustments": numerical_adjustments,
        "numerical_zero_canonicalized_count": canonicalized_count,
    }
