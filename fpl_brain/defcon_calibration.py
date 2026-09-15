"""Versioned calibration of the DEFCON threshold probability.

WHAT THIS IS
------------
The raw DEFCON probability ``p_raw`` comes from the frozen action-rate model:
``P(Poisson(shrunk defensive_contribution per 90 * minutes / 90) >= threshold)``.
This module does NOT touch that model.  It maps ``p_raw`` to a calibrated
probability, because the raw signal was measured to be systematically
over-confident:

    p_raw -> clip -> logit -> intercept + slope * logit(p_raw) -> sigmoid

The mapping is fitted on the PRIOR season only and frozen as a versioned spec.
A historical run must never silently change meaning when a future calibration
version is introduced, so the spec is identified by version, persisted with the
run, and hashed into the run's configuration identity.

FAIL-CLOSED
-----------
``resolve()`` raises for an unknown version, a malformed spec, or a non-finite
parameter.  It never guesses and never silently degrades to the raw probability.
The ONE exception is ``DEFCON_CALIBRATION_LEGACY``, which is an explicitly
declared identity mapping that reproduces the original raw semantics for
historical runs written before calibration existed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Any, Mapping

# --- versions ---------------------------------------------------------------

CALIBRATION_METHOD_PLATT = "PLATT"
CALIBRATION_METHOD_IDENTITY = "IDENTITY"

#: Current production calibration.
DEFCON_CALIBRATION_VERSION = "defcon_platt_v1.0.0"
#: Explicit legacy identity mapping — the original, uncalibrated semantics.
LEGACY_DEFCON_CALIBRATION_VERSION = "defcon_identity_v0.0.0"

#: Clip floor applied before the logit.  Defined as a spec field, not a global.
DEFAULT_CLIP_FLOOR = 1e-6


class DefconCalibrationError(ValueError):
    """The DEFCON calibration spec is missing, malformed or unknown.

    The caller must fail closed: an uncalibrated DEFCON probability may not be
    presented as if it were calibrated.
    """


@dataclass(frozen=True)
class DefconCalibration:
    """One versioned calibration of the DEFCON threshold probability."""

    version: str
    method: str
    intercept: float
    slope: float
    clip_floor: float = DEFAULT_CLIP_FLOOR

    def __post_init__(self) -> None:
        for name in ("version", "method"):
            if not str(getattr(self, name) or "").strip():
                raise DefconCalibrationError(f"calibration {name} is empty")
        for name in ("intercept", "slope", "clip_floor"):
            value = getattr(self, name)
            try:
                as_float = float(value)
            except (TypeError, ValueError) as exc:
                raise DefconCalibrationError(f"calibration {name} is not numeric: {value!r}") from exc
            if not math.isfinite(as_float):
                raise DefconCalibrationError(f"calibration {name} is not finite: {value!r}")
        if not 0.0 < float(self.clip_floor) < 0.5:
            raise DefconCalibrationError(f"clip_floor must lie in (0, 0.5), got {self.clip_floor!r}")
        if self.method == CALIBRATION_METHOD_PLATT and float(self.slope) <= 0.0:
            # A non-positive slope is not monotone increasing, so it cannot be a
            # usable probability calibration.
            raise DefconCalibrationError(f"PLATT requires slope > 0, got {self.slope!r}")

    # --- application -------------------------------------------------------

    def clip(self, p_raw: float) -> float:
        floor = float(self.clip_floor)
        return min(1.0 - floor, max(floor, float(p_raw)))

    def apply(self, p_raw: float) -> float:
        """Calibrated probability.  Monotone non-decreasing in ``p_raw``.

        The boundaries are preserved EXACTLY.  A raw probability of 0 means the
        model says the event is impossible (a player with no chance of playing
        cannot reach the threshold); mapping it to 7.9e-5 would fabricate
        probability mass and give every DEFCON-eligible player a non-zero
        expected DEFCON score.  Symmetrically a raw 1 stays 1.  The clip only
        applies strictly INSIDE the interval, where the logit is defined.
        """

        raw = float(p_raw)
        if self.method == CALIBRATION_METHOD_IDENTITY:
            # No clipping: the identity path exists precisely to reproduce the
            # original raw semantics, boundaries included.
            return raw
        if raw <= 0.0:
            return 0.0
        if raw >= 1.0:
            return 1.0
        p = self.clip(raw)
        logit = math.log(p / (1.0 - p))
        z = float(self.intercept) + float(self.slope) * logit
        if z >= 0.0:
            return 1.0 / (1.0 + math.exp(-z))
        e = math.exp(z)
        return e / (1.0 + e)

    # --- identity / persistence -------------------------------------------

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": str(self.version),
            "method": str(self.method),
            "intercept": float(self.intercept),
            "slope": float(self.slope),
            "clip_floor": float(self.clip_floor),
        }

    def identity(self) -> str:
        """Canonical hash of the full spec — the run-level provenance tie."""

        from .analytics import canonical_hash

        return canonical_hash(self.as_dict())


#: The reviewed and validated production calibration, fitted on 2025/26 only.
DEFCON_PLATT_V1 = DefconCalibration(
    version=DEFCON_CALIBRATION_VERSION,
    method=CALIBRATION_METHOD_PLATT,
    intercept=-0.6683015553121952,
    slope=0.6353806194567421,
    clip_floor=DEFAULT_CLIP_FLOOR,
)

#: The explicit legacy identity mapping (raw semantics for pre-calibration runs).
DEFCON_CALIBRATION_LEGACY = DefconCalibration(
    version=LEGACY_DEFCON_CALIBRATION_VERSION,
    method=CALIBRATION_METHOD_IDENTITY,
    intercept=0.0,
    slope=1.0,
    clip_floor=DEFAULT_CLIP_FLOOR,
)

#: Registry of every calibration a run may legitimately declare.
KNOWN_DEFCON_CALIBRATIONS: dict[str, DefconCalibration] = {
    DEFCON_PLATT_V1.version: DEFCON_PLATT_V1,
    DEFCON_CALIBRATION_LEGACY.version: DEFCON_CALIBRATION_LEGACY,
}


def resolve(version: str | None) -> DefconCalibration:
    """The spec for ``version``, or fail closed.

    ``None`` resolves to the LEGACY identity mapping so that a historical run
    written before calibration existed keeps its original meaning.  Any OTHER
    unknown or malformed version raises rather than degrading to raw.
    """

    if version is None:
        return DEFCON_CALIBRATION_LEGACY
    key = str(version).strip()
    if key not in KNOWN_DEFCON_CALIBRATIONS:
        raise DefconCalibrationError(
            f"unknown DEFCON calibration version {key!r}; known: {sorted(KNOWN_DEFCON_CALIBRATIONS)}"
        )
    return KNOWN_DEFCON_CALIBRATIONS[key]


def from_payload(payload: Mapping[str, Any] | None) -> DefconCalibration:
    """Read a persisted spec back, fail-closed on anything unusable."""

    if payload is None:
        return DEFCON_CALIBRATION_LEGACY
    if not isinstance(payload, Mapping):
        raise DefconCalibrationError(f"DEFCON calibration payload must be a mapping, got {type(payload).__name__}")
    spec = DefconCalibration(
        version=str(payload.get("version") or ""),
        method=str(payload.get("method") or ""),
        intercept=payload.get("intercept"),
        slope=payload.get("slope"),
        clip_floor=payload.get("clip_floor", DEFAULT_CLIP_FLOOR),
    )
    # A persisted spec must also be a KNOWN spec; an unregistered version is a
    # different calibration wearing a familiar shape.
    known = KNOWN_DEFCON_CALIBRATIONS.get(spec.version)
    if known is None:
        raise DefconCalibrationError(f"persisted DEFCON calibration version {spec.version!r} is not recognised")
    if known.as_dict() != spec.as_dict():
        raise DefconCalibrationError(
            f"persisted DEFCON calibration {spec.version!r} disagrees with the registered spec"
        )
    return spec


def as_dict(spec: DefconCalibration) -> dict[str, Any]:
    return asdict(spec)
