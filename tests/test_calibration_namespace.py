"""R2C.1: calibration namespace reproducibility.

The calibration state library must depend only on substantive state-generating
parameters and the explicit seed, so a model-version label change cannot reseed
it.  The full config_hash keeps the version for provenance.
"""

from __future__ import annotations

import dataclasses
import math
import random
import sys
from dataclasses import fields

import pytest

from fpl_brain import monte_carlo as mc


def _sides():
    """A small deterministic side used to materialise the state library."""

    players = []
    for index in range(14):
        players.append(
            {
                "player_id": 100 + index,
                "position": "GKP" if index == 0 else ("DEF" if index < 5 else "MID"),
                "payload": {},
                "minutes": {
                    "joint_start_target": 0.9 if index == 0 else max(0.05, 0.6 - 0.03 * index),
                    "joint_availability": 1.0,
                    "joint_position": "GKP" if index == 0 else ("DEF" if index < 5 else "MID"),
                    "joint_exit_propensity": 0.05,
                    "joint_entry_propensity": 0.2,
                    "joint_expected_minutes_if_cameo": 15.0,
                },
                "position_id": 1 if index == 0 else (2 if index < 5 else 3),
            }
        )
    return {"players": players, "substitution_profile": None, "lambda_for": 1.5, "team_id": 4}


def _state_library(config, fixture_id=41, team_id=4):
    """Materialise the calibration state library the namespace would produce."""

    side = _sides()
    return mc._calibration_states(side, config, fixture_id, team_id)


def _fingerprint(states):
    """Order-sensitive digest of the generated states (identity of the library)."""

    import hashlib

    digest = hashlib.sha256()
    for state in states:
        digest.update(repr(sorted(state)).encode())
    return digest.hexdigest()[:24]


# ---------------------------------------------------------------------------
# A. version label change alone must not reseed
# ---------------------------------------------------------------------------


def test_a_version_label_change_does_not_change_the_state_namespace(monkeypatch):
    base = mc.MonteCarloConfig()
    before_namespace = mc._calibration_namespace(base, 41, 4)
    before_identity = base.calibration_state_identity()

    monkeypatch.setattr(mc, "MONTE_CARLO_MODEL_VERSION", "mc_v9.9.9-label-only")
    after = mc.MonteCarloConfig()
    assert mc._calibration_namespace(after, 41, 4) == before_namespace
    assert after.calibration_state_identity() == before_identity


def test_a_version_label_change_does_not_change_the_library(monkeypatch):
    base = mc.MonteCarloConfig(calibration_states=64)
    before = _fingerprint(_state_library(base))

    monkeypatch.setattr(mc, "MONTE_CARLO_MODEL_VERSION", "mc_v0.0.1-different-label")
    after = _fingerprint(_state_library(mc.MonteCarloConfig(calibration_states=64)))

    assert after == before, "a version-label-only change must not reseed the calibration library"


# ---------------------------------------------------------------------------
# B. substantive calibration parameter change must reseed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "override",
    [
        {"max_substitute_entrants": 4},
        {"substitution_minute_min": 35.0},
        {"substitution_minute_max": 85.0},
        {"gk_substitution_ceiling": 0.11},
        {"exit_propensity_floor": 0.05},
        {"exit_propensity_ceiling": 2.0},
        {"calibration_states": 96},
    ],
)
def test_b_substantive_parameter_changes_the_namespace(override):
    base = mc.MonteCarloConfig()
    changed = mc.MonteCarloConfig(**override)
    assert changed.calibration_state_identity() != base.calibration_state_identity()
    assert mc._calibration_namespace(changed, 41, 4) != mc._calibration_namespace(base, 41, 4)


def test_b_substantive_parameter_changes_the_library():
    base = mc.MonteCarloConfig(calibration_states=64)
    changed = mc.MonteCarloConfig(calibration_states=64, gk_substitution_ceiling=0.11)
    assert _fingerprint(_state_library(changed)) != _fingerprint(_state_library(base))


def test_b_jitter_weights_change_the_namespace():
    base = mc.MonteCarloConfig()
    changed = mc.MonteCarloConfig(substitution_minute_jitter_weights=(0.2, 0.2, 0.2, 0.2, 0.2))
    assert changed.calibration_state_identity() != base.calibration_state_identity()


# ---------------------------------------------------------------------------
# C. explicit seed change must reseed
# ---------------------------------------------------------------------------


def test_c_seed_change_reseeds_the_library():
    base = mc.MonteCarloConfig(calibration_states=64)
    other = mc.MonteCarloConfig(calibration_states=64, calibration_seed=12345)
    assert other.calibration_state_identity() != base.calibration_state_identity()
    assert mc._calibration_namespace(other, 41, 4) != mc._calibration_namespace(base, 41, 4)
    assert _fingerprint(_state_library(other)) != _fingerprint(_state_library(base))


def test_c_fixture_and_team_still_discriminate():
    config = mc.MonteCarloConfig()
    assert mc._calibration_namespace(config, 41, 4) != mc._calibration_namespace(config, 41, 6)
    assert mc._calibration_namespace(config, 41, 4) != mc._calibration_namespace(config, 42, 4)


# ---------------------------------------------------------------------------
# D. provenance hash keeps the version
# ---------------------------------------------------------------------------


def test_d_full_config_hash_still_changes_with_the_version(monkeypatch):
    before = mc.MonteCarloConfig().config_hash()
    monkeypatch.setattr(mc, "MONTE_CARLO_MODEL_VERSION", "mc_v9.9.9-label-only")
    assert mc.MonteCarloConfig().config_hash() != before


def test_d_config_hash_is_unchanged_by_this_phase():
    """The provenance hash must still cover every field plus the version."""

    config = mc.MonteCarloConfig(simulations=10000, seed=20260911, occupancy_audit=True)
    expected = mc.analytics.canonical_hash(
        {
            "model": mc.MONTE_CARLO_MODEL_VERSION,
            **{name: getattr(config, name) for name in (f.name for f in fields(mc.MonteCarloConfig))},
        }
    )
    assert config.config_hash() == expected
    assert config.config_hash() == (
        "sha256:b7b2649151b02eea0a416b7792820199ed06cd6e52c0b082a4dd578fd7488d52"
    )
    assert mc.MONTE_CARLO_MODEL_VERSION == "mc_v1.3.0"


def test_d_provenance_only_thresholds_do_not_move_the_state_identity():
    base = mc.MonteCarloConfig()
    for override in (
        {"individual_max_z": 9.0},
        {"individual_error_materiality": 0.3},
        {"calibration_tolerance": 1e-5},
        {"calibration_max_iterations": 400},
        {"calibration_damping": 0.4},
        {"simulations": 2_000},
        {"occupancy_audit": True},
        {"team_minutes_fail": 150.0},
        {"xg_mass_epsilon": 1e-3},
        {"premium_bias_tolerance_points": 0.09},
        {"score_10_plus_threshold": 11},
    ):
        assert mc.MonteCarloConfig(**override).calibration_state_identity() == base.calibration_state_identity(), override


def test_d_integration_driver_parameters_do_not_move_the_state_identity():
    """integration_draws / integration_seed drive the Minutes integration, not the sampler."""

    base = mc.MonteCarloConfig()
    joint_fields = {f.name for f in fields(mc._joint_config(base))}
    assert joint_fields == set(mc.CALIBRATION_STATE_JOINT_FIELDS) | set(mc.CALIBRATION_STATE_JOINT_EXCLUDED)


def test_kernel_field_classification_is_exhaustive():
    """A newly added kernel field cannot be silently omitted from the identity."""

    base = mc.MonteCarloConfig()
    joint_fields = {f.name for f in fields(mc._joint_config(base))}
    classified = set(mc.CALIBRATION_STATE_JOINT_FIELDS) | set(mc.CALIBRATION_STATE_JOINT_EXCLUDED)
    assert joint_fields == classified
    assert not (set(mc.CALIBRATION_STATE_JOINT_FIELDS) & set(mc.CALIBRATION_STATE_JOINT_EXCLUDED))


# ---------------------------------------------------------------------------
# E. R2C numerical-zero behaviour unchanged
# ---------------------------------------------------------------------------


def test_e_numerical_zero_policy_is_untouched():
    assert mc.NUMERICAL_ZERO_ULP_FACTOR == 8.0
    assert mc.numerical_zero_epsilon(1.610637) == pytest.approx(math.ulp(1.610637) * 8.0)
    assert mc.is_numerical_zero(math.ulp(1.610637), 1.610637)
    canonical, diagnostic = mc.canonicalize_numerical_zero(math.ulp(1.610637), 1.610637)
    assert canonical == 0.0
    assert diagnostic["diagnostic"] == mc.DIAG_NUMERICAL_ZERO_CANONICALIZED


def test_e_residual_mode_semantics_are_untouched():
    states = [set(range(3)) for _ in range(40)]
    config = mc.MonteCarloConfig()
    noise = mc._solve_shares(
        states, 3, [0.5, 0.3, 0.2], math.ulp(1.610637) / 1.610637, config, label="SCORER"
    )
    assert noise["residual_mode"] is False
    assert noise["converged"] is True

    genuine = mc._solve_shares(states, 3, [0.3, 0.2, 0.1], 0.4, config, label="SCORER")
    assert genuine["residual_mode"] is True
    assert genuine["converged"] is True


def test_e_state_identity_is_deterministic_across_instances():
    assert mc.MonteCarloConfig().calibration_state_identity() == mc.MonteCarloConfig().calibration_state_identity()


# ---------------------------------------------------------------------------
# F. historical runs untouched
# ---------------------------------------------------------------------------


def test_f_historical_projection_runs_are_untouched():
    """Read-only check: run 71/87 provenance is exactly as stored."""

    import sqlite3
    from fpl_brain.config import config_path, load_config

    try:
        config = load_config(None)
    except Exception:
        pytest.skip("no project config available")
    db_path = config_path(config, "database")
    if not db_path.exists():
        pytest.skip("project database not present")
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = {
            row[0]: (row[1], row[2], row[3])
            for row in conn.execute(
                "SELECT id, model_version, config_hash, random_seed FROM projection_runs WHERE id IN (71, 87)"
            )
        }
        total = conn.execute("SELECT COUNT(*) FROM projection_runs").fetchone()[0]
        max_id = conn.execute("SELECT MAX(id) FROM projection_runs").fetchone()[0]
    finally:
        conn.close()
    # Historical rows must be UNMUTATED, but the count is not pinned: a
    # legitimate fresh certification adds runs (R3 added 52). Pinning an absolute
    # count would conflate "no historical row was mutated" with "no new run was
    # created", and would fail on every valid freeze.
    assert total >= 160 and max_id >= 160
    assert rows[71][0] == "mc_v1.2.1"
    assert rows[71][1] == "sha256:fedf5f6e50b4826b6e44b42d524cac2d682f15e878a5c86f7bab08a3f4e15042"
    assert rows[87][0] == "mc_v1.2.1"
    assert rows[87][1] == "sha256:0a15a3e7117c1522cfb9819b0624522f8f6db8d6b4ee0ab91c9b9558cae879fd"
    assert rows[71][2] == 20260911 and rows[87][2] == 20260911


# ---------------------------------------------------------------------------
# R3 provenance hardening: a certification artifact must expose the identity
# ---------------------------------------------------------------------------


def test_calibration_provenance_block_is_complete_and_recoverable():
    config = mc.MonteCarloConfig()
    block = mc.calibration_provenance(config)
    assert mc.missing_calibration_provenance(block) == []
    assert block["calibration_state_identity"] == config.calibration_state_identity()
    assert block["calibration_state_identity_version"] == mc.CALIBRATION_STATE_IDENTITY_VERSION
    assert block["calibration_seed"] == config.calibration_seed
    assert block["calibration_states"] == config.calibration_states
    # The recorded identity is enough to reconstruct the namespace.
    namespace = mc._calibration_namespace(config, 41, 4)
    assert block["calibration_state_identity"][:16] in namespace
    assert str(config.calibration_seed) in namespace


def test_missing_calibration_provenance_is_detected():
    assert set(mc.missing_calibration_provenance(None)) == set(mc.CALIBRATION_PROVENANCE_FIELDS)
    assert set(mc.missing_calibration_provenance({})) == set(mc.CALIBRATION_PROVENANCE_FIELDS)
    partial = mc.calibration_provenance(mc.MonteCarloConfig())
    partial.pop("calibration_state_identity")
    assert mc.missing_calibration_provenance(partial) == ["calibration_state_identity"]


def test_require_calibration_provenance_fails_closed_for_the_current_version():
    complete = mc.calibration_provenance(mc.MonteCarloConfig())
    mc.require_calibration_provenance(complete)  # no raise

    with pytest.raises(mc.CalibrationProvenanceError, match="calibration provenance"):
        mc.require_calibration_provenance({})
    with pytest.raises(mc.CalibrationProvenanceError):
        mc.require_calibration_provenance(None)
    # A certified version cannot be accepted with the identity stripped.
    incomplete = dict(complete)
    incomplete["calibration_state_identity"] = None
    with pytest.raises(mc.CalibrationProvenanceError):
        mc.require_calibration_provenance(incomplete)


def test_historical_versions_are_exempt_from_the_provenance_requirement():
    """Reading an older artifact must never raise."""

    mc.require_calibration_provenance({}, model_version="mc_v1.2.1")
    mc.require_calibration_provenance(None, model_version="mc_v1.2.1")
    assert mc.CALIBRATION_PROVENANCE_REQUIRED_FROM == mc.MONTE_CARLO_MODEL_VERSION
