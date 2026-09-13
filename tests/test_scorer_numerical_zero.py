"""R2C: scorer reconciliation numerical correctness.

Deterministic unit tests for the scale-aware numerical-zero policy, the
residual-mode entry rule, the feasible-target interval, and a fixture-41-shaped
regression.  No Monte Carlo production run, no projection writes.
"""

from __future__ import annotations

import math
import sys

import pytest

from fpl_brain import monte_carlo as mc

TOLERANCE = 1e-4
LAMBDA = 1.610637  # the GW5 fixture-41 Brentford side


def _solve(target_residual, *, targets, states, config=None, label="SCORER"):
    return mc._solve_shares(
        states, len(targets), targets, target_residual, config or mc.MonteCarloConfig(), label=label
    )


def _trivial_states(n_players=3, n_states=40):
    """Every player eligible in every state: a well-posed, always-feasible library."""

    return [set(range(n_players)) for _ in range(n_states)]


# ---------------------------------------------------------------------------
# Numerical-zero policy
# ---------------------------------------------------------------------------


def test_epsilon_is_derived_from_machine_precision_not_a_model_threshold():
    eps = mc.numerical_zero_epsilon(LAMBDA)
    assert eps == pytest.approx(math.ulp(LAMBDA) * mc.NUMERICAL_ZERO_ULP_FACTOR, rel=1e-12)
    # Far below the solver tolerance and far below any acceptance threshold.
    assert eps < 1e-12
    assert eps < TOLERANCE


def test_g_epsilon_scales_with_magnitude():
    small = mc.numerical_zero_epsilon(0.01)
    large = mc.numerical_zero_epsilon(100.0)
    assert small < mc.numerical_zero_epsilon(1.0) < large
    assert mc.numerical_zero_epsilon(0.0) == 0.0
    # Roughly proportional to magnitude across orders of magnitude.
    assert mc.numerical_zero_epsilon(1e6) / mc.numerical_zero_epsilon(1.0) > 1e5


def test_a_positive_boundary_noise_is_numerical_zero():
    value = 2.220446049250313e-16
    assert value == math.ulp(LAMBDA)
    assert mc.is_numerical_zero(value, LAMBDA)
    canonical, diagnostic = mc.canonicalize_numerical_zero(value, LAMBDA)
    assert canonical == 0.0
    assert diagnostic["diagnostic"] == mc.DIAG_NUMERICAL_ZERO_CANONICALIZED
    assert diagnostic["original_value"] == value
    assert diagnostic["canonical_value"] == 0.0
    assert diagnostic["scale"] == LAMBDA
    assert diagnostic["epsilon"] == mc.numerical_zero_epsilon(LAMBDA)


def test_b_negative_boundary_noise_is_numerical_zero():
    value = -2.220446049250313e-16
    assert mc.is_numerical_zero(value, LAMBDA)
    canonical, diagnostic = mc.canonicalize_numerical_zero(value, LAMBDA)
    assert canonical == 0.0
    assert diagnostic is not None


def test_f_large_residual_is_never_epsilon_clamped():
    for value in (1.0e-3, 1.0e-2, 0.166159, 0.5):
        canonical, diagnostic = mc.canonicalize_numerical_zero(value, LAMBDA)
        assert canonical == value, value
        assert diagnostic is None, value


def test_exactly_zero_is_a_no_op_without_a_diagnostic():
    canonical, diagnostic = mc.canonicalize_numerical_zero(0.0, LAMBDA)
    assert canonical == 0.0
    assert diagnostic is None


def test_representation_rounding_residual_is_not_canonicalized():
    """The 6-decimal storage-rounding band (~1e-5 goals) is not cancellation noise."""

    assert not mc.is_numerical_zero(1.000000000139778e-06, LAMBDA)
    assert mc.canonicalize_numerical_zero(1.000000000139778e-06, LAMBDA) == (
        1.000000000139778e-06,
        None,
    )


# ---------------------------------------------------------------------------
# Residual-mode entry
# ---------------------------------------------------------------------------


def test_h_exact_zero_never_enters_the_residual_solver():
    result = _solve(0.0, targets=[0.6, 0.4, 0.0], states=_trivial_states())
    assert result["residual_mode"] is False
    assert result["residual_weight"] == 0.0


def test_a_solver_not_launched_for_ulp_scale_noise():
    result = _solve(math.ulp(LAMBDA) / LAMBDA, targets=[0.6, 0.4, 0.0], states=_trivial_states())
    assert result["residual_mode"] is False
    assert result["converged"] is True
    canon = [
        d for d in result["numerical_diagnostics"] if d["diagnostic"] == mc.DIAG_NUMERICAL_ZERO_CANONICALIZED
    ]
    assert canon, "the canonicalisation must be auditable"


def test_c_genuinely_positive_residual_still_runs_the_solver():
    result = _solve(0.15, targets=[0.5, 0.35, 0.0], states=_trivial_states())
    assert result["residual_mode"] is True
    assert result["converged"] is True
    assert result["achieved_residual"] == pytest.approx(0.15, abs=1e-3)


# ---------------------------------------------------------------------------
# Feasible target interval
# ---------------------------------------------------------------------------


def test_d_target_below_the_feasible_lower_bound_fails():
    result = _solve(-0.25, targets=[0.6, 0.4, 0.0], states=_trivial_states())
    assert result["converged"] is False
    assert any(mc.DIAG_TARGET_OUTSIDE_FEASIBLE_RANGE in message for message in result["infeasible"])


def test_e_materially_impossible_target_hard_fails():
    result = _solve(1.75, targets=[0.6, 0.4, 0.0], states=_trivial_states())
    assert result["converged"] is False
    assert any(mc.DIAG_TARGET_OUTSIDE_FEASIBLE_RANGE in message for message in result["infeasible"])


def test_target_above_eligibility_still_hard_fails():
    # Player index 2 is on the pitch in only 2 of 40 states (eligibility 0.05),
    # so a 0.5 share is unreachable for him: a genuine infeasibility that must
    # keep failing rather than being clamped into success.
    states = [set(range(3)) for _ in range(40)]
    for index in range(38):
        states[index] = {0, 1}
    result = _solve(0.0, targets=[0.4, 0.3, 0.5], states=states)
    assert result["converged"] is False
    assert any("exceeds eligibility" in message for message in result["infeasible"])


def test_boundary_adjacent_target_is_clamped_and_recorded():
    canonical, messages, diagnostic = mc.clamp_target_to_feasible_interval(
        1.0 + math.ulp(1.0), low=0.0, high=1.0, scale=1.0, label="SCORER_RESIDUAL"
    )
    assert canonical == 1.0
    assert messages == []
    assert diagnostic is not None and diagnostic["diagnostic"] == "TARGET_CLAMPED_TO_FEASIBLE_BOUNDARY"


def test_materially_out_of_range_target_is_not_clamped():
    canonical, messages, diagnostic = mc.clamp_target_to_feasible_interval(
        -0.2, low=0.0, high=1.0, scale=1.0, label="SCORER_RESIDUAL"
    )
    assert canonical == -0.2
    assert any(mc.DIAG_TARGET_OUTSIDE_FEASIBLE_RANGE in m for m in messages)
    assert diagnostic is None


# ---------------------------------------------------------------------------
# Canonicalisation must not move the team target
# ---------------------------------------------------------------------------


def test_i_canonicalization_does_not_alter_the_team_target():
    targets = [0.55, 0.30, 0.15]
    states = _trivial_states()
    baseline = _solve(0.0, targets=targets, states=states)
    noisy = _solve(math.ulp(LAMBDA) / LAMBDA, targets=targets, states=states)
    assert baseline["targets"] == noisy["targets"] == targets
    assert baseline["achieved"] == noisy["achieved"]
    assert sum(noisy["achieved"]) == pytest.approx(sum(targets), abs=1e-6)


def test_j_replay_is_deterministic():
    targets = [0.5, 0.3, 0.2]
    first = _solve(0.0, targets=targets, states=_trivial_states())
    second = _solve(0.0, targets=targets, states=_trivial_states())
    assert first["achieved"] == second["achieved"]
    assert first["weights"] == second["weights"]
    assert first["iterations"] == second["iterations"]


# ---------------------------------------------------------------------------
# Fixture-41-shaped regression (no club hardcoded)
# ---------------------------------------------------------------------------


def _brentford_shaped_targets():
    """A side whose stored xG sums exactly to lambda through the cap.

    Mirrors the fixture-41 shape: the team cap binds, so the normalised player
    shares sum to 1.0 and the intended residual is exactly zero.
    """

    raw = [0.62, 0.34, 0.18, 0.12, 0.09, 0.20, 0.05, 0.29, 0.06, 0.11]
    scale = 1.0 / sum(raw)
    return [value * scale for value in raw]


def test_regression_capped_side_with_ulp_noise_does_not_stall():
    targets = _brentford_shaped_targets()
    n = len(targets)
    states = [set(range(n)) for _ in range(60)]
    config = mc.MonteCarloConfig()

    # Pre-fix semantics: any strictly positive residual enters residual mode.
    noise_share = math.ulp(LAMBDA) / LAMBDA
    assert noise_share > 0.0

    fixed = mc._solve_shares(states, n, targets, noise_share, config, label="SCORER")
    assert fixed["residual_mode"] is False, "ulp-scale noise must not enable residual mode"
    assert fixed["converged"] is True
    assert fixed["max_residual"] <= config.calibration_tolerance
    assert sum(fixed["achieved"]) == pytest.approx(1.0, abs=1e-6)
    # Team scorer mass stays coherent: every share is preserved, none invented.
    for achieved, target in zip(fixed["achieved"], targets):
        assert achieved == pytest.approx(target, abs=1e-3)
    # No player-level material failure caused by the numerical defect.
    for achieved, target in zip(fixed["achieved"], targets):
        assert abs(achieved - target) < config.individual_error_materiality


def test_regression_material_residual_is_still_reported():
    """The fix must not silence a genuine non-zero residual."""

    # Player shares sum to 0.6 and a 0.4 residual completes the goal mass.
    targets = [0.3, 0.2, 0.1]
    states = [set(range(3)) for _ in range(40)]
    result = _solve(0.4, targets=targets, states=states)
    assert result["residual_mode"] is True
    assert result["converged"] is True
    assert result["achieved_residual"] == pytest.approx(0.4, abs=2e-3)
    # The player shares still match their own targets, and the held-back
    # residual plus the player mass accounts for the whole team expectation.
    assert sum(result["achieved"]) == pytest.approx(sum(targets), abs=2e-3)
    assert sum(result["achieved"]) + result["achieved_residual"] == pytest.approx(1.0, abs=5e-3)


def test_audit_diagnostic_carries_original_canonical_epsilon_and_scale():
    result = _solve(math.ulp(LAMBDA) / LAMBDA, targets=[0.6, 0.4, 0.0], states=_trivial_states())
    record = next(
        d for d in result["numerical_diagnostics"] if d["diagnostic"] == mc.DIAG_NUMERICAL_ZERO_CANONICALIZED
    )
    assert set(record) >= {"original_value", "canonical_value", "epsilon", "scale", "ulp_factor"}
    assert record["original_value"] > 0.0
    assert record["canonical_value"] == 0.0
    assert result["target_residual_requested"] == record["original_value"]
    assert result["target_residual"] == 0.0


def test_version_was_bumped_for_this_change():
    assert mc.MONTE_CARLO_MODEL_VERSION == "mc_v1.3.0"
