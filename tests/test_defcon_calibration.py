"""DEFCON probability calibration — versioned spec, oracle pin, fail-closed.

The calibration changes production DEFCON numbers, so these tests do three
separate jobs:

1. PIN THE VALIDATED ORACLE.  The shipped constants must reproduce the mapping
   that was validated out-of-sample, value for value.  Nothing here is re-fitted.
2. PROVE THE SAFETY PROPERTIES the promotion depends on: finite at the extremes,
   bounded to [0, 1], monotone non-decreasing, and never a rank reversal.
3. PROVE THE FAIL-CLOSED CONTRACT: an unknown, malformed or non-finite spec must
   abort rather than silently emit a raw probability, while a pre-calibration
   historical run must still resolve to its original raw semantics.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import pytest

from fpl_brain import analytics
from fpl_brain import defcon_calibration as dc
from fpl_brain import scoring_rules as sr
from fpl_brain import xpts

RULES = sr.DEFAULT_SCORING_RULES

#: The validated mapping, recomputed here independently of the implementation.
def _validated_oracle(p_raw: float) -> float:
    raw = float(p_raw)
    # the boundaries are preserved exactly, so that an impossible event never
    # acquires probability mass and a certain one never loses it
    if raw <= 0.0:
        return 0.0
    if raw >= 1.0:
        return 1.0
    floor = 1e-6
    p = min(1.0 - floor, max(floor, raw))
    z = -0.6683015553121952 + 0.6353806194567421 * math.log(p / (1.0 - p))
    return 1.0 / (1.0 + math.exp(-z)) if z >= 0.0 else math.exp(z) / (1.0 + math.exp(z))


#: (raw, calibrated) pinned from the validated formula and frozen for production.
PINNED = (
    (0.0, 0.0),
    (1e-09, 7.8966444734568267e-05),
    (1e-06, 7.8966444734568267e-05),
    (0.01, 0.026910705938498335),
    (0.05, 0.073159397159664033),
    (0.1, 0.11260785953397956),
    (0.25, 0.20321205910101875),
    (0.5, 0.33887725523471457),
    (0.75, 0.5074333026023562),
    (0.9, 0.67431624158962),
    (0.99, 0.90476679406375016),
    (1.0, 1.0),
)


# ---------------------------------------------------------------------------
# 1. spec / version
# ---------------------------------------------------------------------------


def test_the_shipped_spec_is_the_validated_version():
    spec = dc.DEFCON_PLATT_V1
    assert spec.version == "defcon_platt_v1.0.0"
    assert dc.DEFCON_CALIBRATION_VERSION == "defcon_platt_v1.0.0"
    assert spec.method == dc.CALIBRATION_METHOD_PLATT
    assert spec.intercept == -0.6683015553121952
    assert spec.slope == 0.6353806194567421
    assert spec.clip_floor == 1e-6


def test_representative_values_are_pinned_to_the_validated_oracle():
    spec = dc.DEFCON_PLATT_V1
    for raw, expected in PINNED:
        assert spec.apply(raw) == expected, raw
        assert spec.apply(raw) == _validated_oracle(raw), raw


def test_the_oracle_agrees_across_the_whole_range():
    spec = dc.DEFCON_PLATT_V1
    for i in range(1, 1000):
        p = i / 1000.0
        assert spec.apply(p) == pytest.approx(_validated_oracle(p), rel=0, abs=1e-15)


# ---------------------------------------------------------------------------
# 2. numerical safety
# ---------------------------------------------------------------------------


def test_extremes_stay_finite_and_inside_the_unit_interval():
    spec = dc.DEFCON_PLATT_V1
    for raw in (0.0, 1e-12, 1e-6, 0.5, 1.0 - 1e-6, 1.0, -1.0, 2.0):
        value = spec.apply(raw)
        assert math.isfinite(value)
        assert 0.0 <= value <= 1.0


def test_the_mapping_is_monotone_non_decreasing():
    spec = dc.DEFCON_PLATT_V1
    values = [spec.apply(i / 5000.0) for i in range(5001)]
    assert all(b >= a for a, b in zip(values, values[1:]))


def test_no_strict_rank_reversal_for_distinct_unclipped_inputs():
    spec = dc.DEFCON_PLATT_V1
    raw = [1e-4, 1e-3, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    cal = [spec.apply(p) for p in raw]
    for i in range(len(raw)):
        for j in range(len(raw)):
            if raw[i] < raw[j]:
                assert cal[i] <= cal[j]


def test_clipped_inputs_may_tie_but_never_invert():
    spec = dc.DEFCON_PLATT_V1
    # strictly positive inputs below the floor all collapse to the same value
    below = [spec.apply(p) for p in (1e-12, 1e-9, 1e-7, 1e-6)]
    assert all(v == below[0] for v in below)
    assert below[0] < spec.apply(2e-6)  # and the next step still increases


def test_impossible_events_gain_no_probability_mass():
    """Preserving p_raw == 0 matters: a player with no chance of playing must
    not acquire a non-zero expected DEFCON score from the calibration."""

    spec = dc.DEFCON_PLATT_V1
    assert spec.apply(0.0) == 0.0
    assert spec.apply(-0.5) == 0.0
    assert spec.apply(1.0) == 1.0
    assert spec.apply(1.5) == 1.0
    # just inside the interval, the calibration is in force
    assert 0.0 < spec.apply(1e-12) < 1.0
    assert 0.0 < spec.apply(1.0 - 1e-12) < 1.0


# ---------------------------------------------------------------------------
# 3. the identity calibration is the original raw semantics
# ---------------------------------------------------------------------------


def test_the_legacy_identity_calibration_reproduces_raw_semantics():
    legacy = dc.DEFCON_CALIBRATION_LEGACY
    assert legacy.version == dc.LEGACY_DEFCON_CALIBRATION_VERSION
    assert legacy.method == dc.CALIBRATION_METHOD_IDENTITY
    # the identity path is an EXACT passthrough, boundaries included — it must
    # not clip, or a historical replay would not reproduce its own numbers
    for raw in (0.0, 1e-12, 0.05, 0.3, 0.5, 0.9, 0.999999, 1.0):
        assert legacy.apply(raw) == raw


# ---------------------------------------------------------------------------
# 4. fail-closed contract
# ---------------------------------------------------------------------------


def test_resolve_rejects_an_unknown_version():
    with pytest.raises(dc.DefconCalibrationError):
        dc.resolve("defcon_platt_v9.9.9")


def test_resolve_treats_a_missing_version_as_the_legacy_path():
    assert dc.resolve(None) is dc.DEFCON_CALIBRATION_LEGACY


@pytest.mark.parametrize(
    "payload",
    [
        {"version": "defcon_platt_v1.0.0", "method": "PLATT", "intercept": float("nan"), "slope": 0.5},
        {"version": "defcon_platt_v1.0.0", "method": "PLATT", "intercept": 0.0, "slope": float("inf")},
        {"version": "defcon_platt_v1.0.0", "method": "PLATT", "intercept": 0.0, "slope": -0.5},
        {"version": "defcon_platt_v1.0.0", "method": "PLATT", "intercept": "x", "slope": 0.5},
        {"version": "defcon_platt_v1.0.0", "method": "", "intercept": 0.0, "slope": 0.5},
        {"version": "defcon_platt_v1.0.0", "method": "PLATT", "intercept": 0.0, "slope": 0.5, "clip_floor": 0.9},
        "not-a-mapping",
    ],
)
def test_from_payload_fails_closed_on_a_malformed_spec(payload):
    with pytest.raises(dc.DefconCalibrationError):
        dc.from_payload(payload)


def test_from_payload_rejects_a_well_formed_but_unregistered_version():
    """A different calibration wearing a familiar shape is still not this one."""

    with pytest.raises(dc.DefconCalibrationError):
        dc.from_payload({"version": "defcon_platt_v2.0.0", "method": "PLATT", "intercept": -1.0, "slope": 0.4})


def test_from_payload_rejects_a_registered_version_with_tampered_parameters():
    with pytest.raises(dc.DefconCalibrationError):
        dc.from_payload(
            {"version": "defcon_platt_v1.0.0", "method": "PLATT", "intercept": -0.5, "slope": 0.5, "clip_floor": 1e-6}
        )


def test_from_payload_round_trips_and_treats_absence_as_legacy():
    assert dc.from_payload(dc.DEFCON_PLATT_V1.as_dict()) == dc.DEFCON_PLATT_V1
    assert dc.from_payload(None) is dc.DEFCON_CALIBRATION_LEGACY


# ---------------------------------------------------------------------------
# 5. consistency between defcon_p_hit and defcon_xpts
# ---------------------------------------------------------------------------


def test_defcon_xpts_is_exactly_twice_the_calibrated_hit_probability():
    payload = {
        "expected_minutes": 78.0,
        "minute_state_distribution": [
            {"state": "0", "probability": 0.2, "conditional_mean_minutes": 0.0},
            {"state": "60-79", "probability": 0.5, "conditional_mean_minutes": 70.0},
            {"state": "80+", "probability": 0.3, "conditional_mean_minutes": 88.0},
        ],
    }
    for position in ("DEF", "MID", "FWD"):
        value, hit, used = xpts.defcon_xpts_with_mixture(
            position, 8.0, payload, RULES, calibration=dc.DEFCON_PLATT_V1
        )
        assert used is True
        assert hit > 0.0
        assert value == pytest.approx(RULES.defcon_points * hit, rel=0, abs=1e-15)


def test_the_calibrated_mixture_is_the_calibration_of_the_raw_mixture():
    """Same states, same weights — only the per-state probability is mapped."""

    payload = {
        "expected_minutes": 70.0,
        "minute_state_distribution": [
            {"state": "0", "probability": 0.3, "conditional_mean_minutes": 0.0},
            {"state": "80+", "probability": 0.7, "conditional_mean_minutes": 85.0},
        ],
    }
    _, raw_hit, _ = xpts.defcon_xpts_with_mixture(
        "MID", 9.0, payload, RULES, calibration=dc.DEFCON_CALIBRATION_LEGACY
    )
    _, cal_hit, _ = xpts.defcon_xpts_with_mixture(
        "MID", 9.0, payload, RULES, calibration=dc.DEFCON_PLATT_V1
    )
    manual = 0.3 * _validated_oracle(0.0) + 0.7 * _validated_oracle(
        xpts.poisson_tail_probability(9.0 * 85.0 / 90.0, RULES.defcon_threshold_for("MID"))
    )
    assert cal_hit == pytest.approx(manual, rel=0, abs=1e-15)
    # calibration can only lower a probability that was over-confident
    assert cal_hit < raw_hit


def test_the_per_exposure_helper_is_the_single_definition():
    raw = xpts._defcon_p_hit("MID", 9.0, 80.0, RULES)
    assert xpts.defcon_hit_probability(
        "MID", 9.0, 80.0, RULES, calibration=dc.DEFCON_CALIBRATION_LEGACY
    ) == pytest.approx(raw)
    assert xpts.defcon_hit_probability(
        "MID", 9.0, 80.0, RULES, calibration=dc.DEFCON_PLATT_V1
    ) == pytest.approx(dc.DEFCON_PLATT_V1.apply(raw))


# ---------------------------------------------------------------------------
# 6. reproducibility plumbing
# ---------------------------------------------------------------------------


def test_config_hash_pins_the_full_calibration_spec():
    base = xpts.XPtsConfig()
    assert base.config_hash() == xpts.XPtsConfig().config_hash()
    dated = xpts.XPtsConfig(defcon_calibration_version=dc.LEGACY_DEFCON_CALIBRATION_VERSION)
    assert dated.config_hash() != base.config_hash()
    with pytest.raises(dc.DefconCalibrationError):
        xpts.XPtsConfig(defcon_calibration_version="defcon_platt_v9.9.9").config_hash()


def test_the_calibration_module_is_in_the_certified_source_set():
    assert "fpl_brain/defcon_calibration.py" in analytics.SOURCE_SNAPSHOT_FILES


def test_the_production_call_site_passes_an_explicit_calibration():
    """A prediction must never inherit raw semantics by omission.

    The library default is the legacy identity mapping so historical replays and
    primitive-level tests keep their original meaning; production must therefore
    opt IN explicitly.  A comment spelling the call is not evidence, so this
    reads the real call site.
    """

    source = Path("fpl_brain/xpts.py").read_text(encoding="utf-8")
    assert "calibration=config.defcon_calibration" in source
    # the raw helper must still exist and still be the raw generator
    assert re.search(r"def _defcon_p_hit\(", source)
    raw_body = source.split("def _defcon_p_hit(")[1].split("\ndef ")[0]
    assert "poisson_tail_probability" in raw_body
    assert "apply(" not in raw_body


def test_monte_carlo_applies_the_rows_own_declared_calibration():
    source = Path("fpl_brain/monte_carlo.py").read_text(encoding="utf-8")
    assert "defcon_cal.from_payload(" in source
    assert "xpts.defcon_hit_probability(" in source
    # the raw count sampling must be gone from the defcon branch
    assert "_poisson(rng, max(0.0, actions_per90) * exposure / 90.0)" not in source
