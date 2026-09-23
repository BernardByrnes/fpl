"""Probability calibration: reliability binning, versioned transforms, causal fits.

WHAT THIS IS
------------
PE-8 asks whether the engine's persisted probabilities mean what they say.  The
frozen tree already provides Brier, a Brier reference score and the quantile
coverage metrics, but two pieces of machinery did not exist and are built here:

* **RELIABILITY BINNING.**  :func:`reliability_bins` adds the observed-frequency
  view that Brier alone cannot give.  The bin edges, the per-bin sample floor and
  the interval convention are DECLARED CONSTANTS in this module, never chosen
  from the data being described, and every bin carries its own sample size.
* **CAUSAL FITTING.**  :func:`fit_platt_causal` and
  :func:`fit_assist_mapping_causal` fit a transform on outcomes finalised
  STRICTLY BEFORE the origin it is applied at.  The frozen tree contained no
  fitting routine at all: the DefCon calibration's intercept and slope are frozen
  constants that were fitted offline and only ever read from a spec.

THE TRANSFORM PRECEDENT
-----------------------
``fpl_brain.defcon_calibration`` establishes the shape this module follows: a
versioned spec with a validated method, a canonical identity hash, and a
resolution path that fails closed.  Three differences are deliberate.

* **No legacy identity fallback.**  ``defcon_calibration`` resolves ``None`` to
  an identity mapping because runs written before calibration existed must keep
  their original meaning.  PE-8 has no such history: an absent, unknown or
  malformed calibration RAISES.  A missing calibration is never a licence to
  emit the raw number, and there is no silent degradation to identity.
* **A fitted version can never be redefined in place.**  A fitted transform's
  version string carries its own fit-basis digest, so two fits over different
  bases cannot share a version, the same basis always yields the same version,
  and re-pointing a version at different parameters is impossible by
  construction rather than by discipline.
* **A method must state its own monotonicity.**  :meth:`ProbabilityCalibration.monotonicity`
  is a required declaration, and :meth:`verify_monotone` checks the declaration
  numerically.  A transform that cannot state its own monotonicity is not
  admissible as a probability calibration.

BOUNDARIES ARE EXACT
--------------------
A raw probability of exactly ``0`` stays ``0`` and exactly ``1`` stays ``1``.
Mapping a stated impossibility to a small positive number would fabricate
probability mass and give every structurally ineligible player a non-zero
expectation.  The interior clip is declared as a spec field (``clip_floor``,
bounded inside ``(0, 0.5)``) and applies strictly inside the interval, where the
logit is defined.

THE BASIS IS POINT-IN-TIME EVIDENCE
-----------------------------------
The fitting routines here take observations; the CALLER decides which ones.
PE-8's caller builds them from PE-5's append-only, point-in-time observation
captures: a basis row must carry a FINAL official observation whose official
finality AND capture both fall strictly before the origin's certified cutoff,
and a row whose timing cannot be proven is excluded and counted rather than
assumed known.  A later correction is a later capture and therefore cannot enter
an earlier basis.

A FITTED PAYLOAD CARRIES ITS PROVENANCE
---------------------------------------
A fitted spec's payload records its canonical identity AND the applicable
policy versions -- the calibration policy it was produced under and, for a
causally fitted version, the fit policy it was fitted under.  :func:`from_payload`
REQUIRES all three and fails closed when any is absent or disagrees, so a
calibration cannot be resolved from a payload that does not say what produced
it.  The identity hash covers the fit policy too, so a re-pointed provenance
changes the identity.

DETERMINISM
-----------
Every fit sorts its observations into one canonical order before any arithmetic,
so the fitted parameters and the fit-basis digest are invariant to the order the
rows arrive in.  Nothing here reads the clock and nothing here draws a random
number.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from . import analytics

PROBABILITY_CALIBRATION_POLICY_VERSION = "prob_calibration_policy_v1.0.0"
RELIABILITY_POLICY_VERSION = "prob_reliability_v1.0.0"
CAUSAL_FIT_POLICY_VERSION = "prob_causal_fit_v1.0.0"

# --- declared methods -------------------------------------------------------

#: The declared calibration methods.  A method that is not in this tuple cannot
#: be constructed, so an unknown method fails closed at the spec boundary.
METHOD_IDENTITY = "IDENTITY"
METHOD_PLATT = "PLATT"
CALIBRATION_METHODS = (METHOD_IDENTITY, METHOD_PLATT)

PLATT_FIT_METHOD = "PLATT_LOGIT_LEAST_SQUARES"
ASSIST_MAPPING_FIT_METHOD = "RATIO_OF_SUMS"

#: The declared incumbent: the persisted probability, unchanged.  This is a
#: DECLARED transform, not a fallback -- a missing or unknown spec raises.
INCUMBENT_CALIBRATION_VERSION = "prob_identity_v0.0.0"

#: Prefix of every causally fitted Platt transform.  The fitted version is
#: ``<prefix>.<fit-basis digest prefix>``, so the version identifies the exact
#: basis a transform was fitted from.
PLATT_FIT_VERSION_PREFIX = "prob_platt_wf_v1.0.0"

#: Clip floor applied before the logit.  A spec field, not a global.
DEFAULT_CLIP_FLOOR = 1e-6

# --- declared reliability policy --------------------------------------------

#: Ten fixed bins on [0, 1].  Declared a priori and never selected from data.
RELIABILITY_BIN_EDGES: tuple[float, ...] = tuple(index / 10.0 for index in range(11))
RELIABILITY_BIN_CONVENTION = "left-closed [lower, upper); the final bin is closed at 1.0"
RELIABILITY_GAP_CONVENTION = (
    "observed frequency minus mean stated probability; positive means the event happened "
    "more often than the stored probability said"
)
#: A declared disclosure floor for a bin, NOT a significance threshold.  A bin
#: under it is reported as insufficient rather than as a calibration claim.
RELIABILITY_MIN_BIN_N = 50

# --- declared fit policy ----------------------------------------------------

#: Below this many strictly-earlier observations no transform is fitted at all.
MIN_FIT_OBSERVATIONS = 200
#: A Platt fit states a slope, so both realised outcomes must be present in the
#: basis; a basis that has seen no positive (or no negative) outcome cannot
#: support one.
MIN_FIT_CLASS_OBSERVATIONS = 25

#: Degeneracy guard for the assist-mapping ratio of sums: a coefficient outside
#: this band means the denominator was too small to estimate a mapping from, not
#: that football changed.  A band, not a prior.
ASSIST_MAPPING_FIT_BOUNDS = (0.25, 4.0)

# --- bin statuses -----------------------------------------------------------

BIN_OK = "OK"
BIN_EMPTY_SAMPLE = "EMPTY_SAMPLE"
BIN_INSUFFICIENT_SAMPLE = "INSUFFICIENT_SAMPLE"

# --- fit unavailability reasons ---------------------------------------------

FIT_INSUFFICIENT_OBSERVATIONS = "INSUFFICIENT_OBSERVATIONS"
FIT_INSUFFICIENT_CLASS_SUPPORT = "INSUFFICIENT_CLASS_SUPPORT"
FIT_ZERO_VARIANCE_INPUT = "ZERO_VARIANCE_INPUT"
FIT_NON_MONOTONE = "NON_MONOTONE_FIT"
FIT_ZERO_EXPECTED_TOTAL = "ZERO_EXPECTED_TOTAL"
FIT_OUT_OF_DECLARED_BOUNDS = "OUT_OF_DECLARED_BOUNDS"


class ProbabilityCalibrationError(ValueError):
    """The calibration spec is missing, malformed or unknown.

    The caller must fail closed: an uncalibrated probability may not be
    presented as if it were calibrated.
    """


class CausalFitError(ValueError):
    """A fit was asked to use an outcome that was not finalised before its origin.

    This is a CONTRACT VIOLATION, not a data condition: a transform fitted on
    events ``1..k`` and applied at event ``k`` describes a world in which the
    target was already known.  It is refused rather than approximated.
    """


class ReliabilityInputError(ValueError):
    """The inputs cannot produce a reliability view; refuse rather than guess."""


# ---------------------------------------------------------------------------
# The versioned spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProbabilityCalibration:
    """One versioned probability calibration.

    ``version`` names the spec, ``identity()`` hashes its full content, and both
    travel with every payload the calibration produced so a consumer resolves the
    calibration from the payload rather than from a global default.
    """

    version: str
    method: str
    intercept: float
    slope: float
    clip_floor: float = DEFAULT_CLIP_FLOOR
    #: The policy a FITTED version was fitted under.  ``None`` on a declared spec
    #: such as the incumbent, which was not fitted here at all.  It is part of
    #: ``as_dict`` and therefore part of the canonical identity.
    fit_policy_version: str | None = None

    def __post_init__(self) -> None:
        for name in ("version", "method"):
            if not str(getattr(self, name) or "").strip():
                raise ProbabilityCalibrationError(f"calibration {name} is empty")
        if self.method not in CALIBRATION_METHODS:
            raise ProbabilityCalibrationError(
                f"unknown calibration method {self.method!r}; declared: {list(CALIBRATION_METHODS)}"
            )
        if self.fit_policy_version is not None:
            declared = str(self.fit_policy_version).strip()
            if declared != CAUSAL_FIT_POLICY_VERSION:
                raise ProbabilityCalibrationError(
                    f"unknown fit policy version {self.fit_policy_version!r}; declared: "
                    f"{CAUSAL_FIT_POLICY_VERSION!r}"
                )
        for name in ("intercept", "slope", "clip_floor"):
            value = getattr(self, name)
            try:
                as_float = float(value)
            except (TypeError, ValueError) as exc:
                raise ProbabilityCalibrationError(
                    f"calibration {name} is not numeric: {value!r}"
                ) from exc
            if not math.isfinite(as_float):
                raise ProbabilityCalibrationError(f"calibration {name} is not finite: {value!r}")
        if not 0.0 < float(self.clip_floor) < 0.5:
            raise ProbabilityCalibrationError(
                f"clip_floor must lie in (0, 0.5), got {self.clip_floor!r}"
            )
        if self.method == METHOD_PLATT and float(self.slope) <= 0.0:
            # A non-positive slope is not monotone non-decreasing, so it cannot be
            # a probability calibration at all.  Rejected at construction, not
            # caught later by whoever happens to look at the curve.
            raise ProbabilityCalibrationError(
                f"PLATT requires slope > 0 (monotone non-decreasing), got {self.slope!r}"
            )

    # --- application -------------------------------------------------------

    def clip(self, p_raw: float) -> float:
        floor = float(self.clip_floor)
        return min(1.0 - floor, max(floor, float(p_raw)))

    def apply(self, p_raw: float) -> float:
        """The calibrated probability.  Monotone non-decreasing in ``p_raw``.

        The boundaries are preserved EXACTLY: raw ``0`` returns ``0`` and raw
        ``1`` returns ``1``.  The declared ``clip_floor`` applies strictly
        inside the interval, where the logit is defined, so no stated
        impossibility acquires fabricated probability mass.
        """

        raw = float(p_raw)
        if not math.isfinite(raw):
            raise ProbabilityCalibrationError(f"raw probability is not finite: {p_raw!r}")
        if self.method == METHOD_IDENTITY:
            # The identity path reproduces the raw semantics exactly, boundaries
            # included, and ignores the clip floor entirely.
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

    def monotonicity(self) -> dict[str, Any]:
        """The method's own monotonicity declaration.

        Required of every admissible transform: a method that cannot state its
        own monotonicity cannot be a probability calibration, because the whole
        point of one is that it preserves the order of the stated probabilities.
        """

        if self.method == METHOD_IDENTITY:
            return {
                "method": METHOD_IDENTITY,
                "monotone_non_decreasing": True,
                "basis": "the identity map is non-decreasing by definition",
                "preserves_boundaries_exactly": True,
            }
        return {
            "method": METHOD_PLATT,
            "monotone_non_decreasing": bool(float(self.slope) > 0.0),
            "basis": (
                "logit is strictly increasing on (0, 1); slope > 0 keeps the affine map "
                "non-decreasing; the logistic function is strictly increasing on the reals, "
                "so the composition is non-decreasing.  The interior clip is non-decreasing "
                "and the exact boundary short-circuits are non-decreasing."
            ),
            "preserves_boundaries_exactly": True,
        }

    def verify_monotone(self, *, subdivisions: int = 1000) -> dict[str, Any]:
        """Numeric check of the declaration, on a declared grid.

        Scans ``[0, 1]`` at ``subdivisions`` equal steps plus the two boundaries
        and reports every non-monotone step and any boundary shift.  A check, not
        a proof: the declaration above carries the proof.
        """

        steps = max(1, int(subdivisions))
        previous = self.apply(0.0)
        violations: list[list[float]] = []
        if previous != 0.0:
            violations.append([0.0, previous])
        for index in range(1, steps + 1):
            point = index / steps
            value = self.apply(point)
            if value < previous:
                violations.append([point, value])
            previous = value
        if self.apply(1.0) != 1.0:
            violations.append([1.0, self.apply(1.0)])
        return {
            "grid": f"0.0 to 1.0 in {steps} equal steps, plus both boundaries",
            "monotone_non_decreasing": not violations,
            "boundaries_exact": self.apply(0.0) == 0.0 and self.apply(1.0) == 1.0,
            "violations": violations[:10],
        }

    # --- identity / persistence -------------------------------------------

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": str(self.version),
            "method": str(self.method),
            "intercept": float(self.intercept),
            "slope": float(self.slope),
            "clip_floor": float(self.clip_floor),
            "fit_policy_version": (
                None if self.fit_policy_version is None else str(self.fit_policy_version)
            ),
        }

    def identity(self) -> str:
        """Canonical hash of the full spec -- the provenance tie."""

        return analytics.canonical_hash(self.as_dict())

    def as_payload(self) -> dict[str, Any]:
        """The spec, its canonical identity and the policy versions it was produced under."""

        payload = self.as_dict()
        payload["identity"] = self.identity()
        payload["policy_version"] = PROBABILITY_CALIBRATION_POLICY_VERSION
        return payload


#: The declared incumbent: the persisted probability, unchanged.
INCUMBENT_IDENTITY = ProbabilityCalibration(
    version=INCUMBENT_CALIBRATION_VERSION,
    method=METHOD_IDENTITY,
    intercept=0.0,
    slope=1.0,
    clip_floor=DEFAULT_CLIP_FLOOR,
)

#: Registry of the specs a version string alone can resolve.  A causally fitted
#: transform is deliberately NOT registered: it is resolved from its own payload,
#: where its identity can be recomputed and checked.
KNOWN_CALIBRATIONS: dict[str, ProbabilityCalibration] = {
    INCUMBENT_IDENTITY.version: INCUMBENT_IDENTITY,
}

#: Version namespaces this module will accept from a payload.  Anything else is
#: an unknown calibration wearing a familiar shape.
DECLARED_VERSION_NAMESPACES: tuple[str, ...] = (
    INCUMBENT_CALIBRATION_VERSION,
    PLATT_FIT_VERSION_PREFIX + ".",
)


def is_declared_version(version: Any) -> bool:
    """Whether ``version`` names a spec this module is willing to resolve."""

    key = str(version or "").strip()
    if not key:
        return False
    if key in KNOWN_CALIBRATIONS:
        return True
    return any(key.startswith(namespace) and len(key) > len(namespace) for namespace in DECLARED_VERSION_NAMESPACES)


def is_fitted_version(version: Any) -> bool:
    """Whether ``version`` names a causally fitted transform rather than a declared spec."""

    return str(version or "").strip().startswith(PLATT_FIT_VERSION_PREFIX + ".")


def resolve(version: str | None) -> ProbabilityCalibration:
    """The registered spec for ``version``, or fail closed.

    Unlike ``defcon_calibration.resolve`` there is NO ``None`` fallback to an
    identity mapping.  A missing version is an unknown calibration, and an
    unknown calibration stops the caller.
    """

    if version is None:
        raise ProbabilityCalibrationError(
            "no probability calibration version was declared; there is no implicit identity mapping"
        )
    key = str(version).strip()
    if key not in KNOWN_CALIBRATIONS:
        raise ProbabilityCalibrationError(
            f"unknown probability calibration version {key!r}; registered: {sorted(KNOWN_CALIBRATIONS)}"
        )
    return KNOWN_CALIBRATIONS[key]


def from_payload(payload: Mapping[str, Any] | None) -> ProbabilityCalibration:
    """Read a persisted spec back, fail-closed on anything unusable.

    A causally fitted transform resolves here, not through :func:`resolve`: the
    payload carries the parameters and the canonical identity, and the identity
    is RECOMPUTED and compared.  A payload is refused when

    * it carries no canonical ``identity``, because a spec that cannot be tied
      to its own content is not a version anyone can resolve;
    * its ``policy_version`` is absent or is not the declared calibration policy;
    * a FITTED version names no ``fit_policy_version``, or names one that is not
      the declared fit policy (and a declared spec claims one it was not fitted
      under);
    * its identity disagrees with its own parameters;
    * it disagrees with a registered spec of the same version;
    * its version namespace is unknown.
    """

    if payload is None:
        raise ProbabilityCalibrationError(
            "the payload carries no probability calibration; refusing to assume the identity mapping"
        )
    if not isinstance(payload, Mapping):
        raise ProbabilityCalibrationError(
            f"probability calibration payload must be a mapping, got {type(payload).__name__}"
        )
    spec = ProbabilityCalibration(
        version=str(payload.get("version") or ""),
        method=str(payload.get("method") or ""),
        intercept=payload.get("intercept"),
        slope=payload.get("slope"),
        clip_floor=payload.get("clip_floor", DEFAULT_CLIP_FLOOR),
        fit_policy_version=payload.get("fit_policy_version"),
    )
    if not is_declared_version(spec.version):
        raise ProbabilityCalibrationError(
            f"persisted probability calibration version {spec.version!r} is not recognised"
        )

    recorded = payload.get("identity")
    if recorded is None or not str(recorded).strip():
        raise ProbabilityCalibrationError(
            f"persisted probability calibration {spec.version!r} carries no canonical identity; a "
            "calibration that cannot be tied to the spec that produced it is not resolvable"
        )

    declared_policy = str(payload.get("policy_version") or "").strip()
    if not declared_policy:
        raise ProbabilityCalibrationError(
            f"persisted probability calibration {spec.version!r} carries no policy_version; the "
            "applicable policy that produced it must travel with it"
        )
    if declared_policy != PROBABILITY_CALIBRATION_POLICY_VERSION:
        raise ProbabilityCalibrationError(
            f"persisted probability calibration {spec.version!r} declares policy_version "
            f"{declared_policy!r}, not the declared {PROBABILITY_CALIBRATION_POLICY_VERSION!r}"
        )

    fit_policy = str(payload.get("fit_policy_version") or "").strip()
    if is_fitted_version(spec.version):
        if fit_policy != CAUSAL_FIT_POLICY_VERSION:
            raise ProbabilityCalibrationError(
                f"fitted probability calibration {spec.version!r} declares fit_policy_version "
                f"{fit_policy or None!r}, not the declared {CAUSAL_FIT_POLICY_VERSION!r}"
            )
    elif fit_policy:
        raise ProbabilityCalibrationError(
            f"declared probability calibration {spec.version!r} claims fit_policy_version "
            f"{fit_policy!r}, but it is a declared spec and not a causal fit"
        )

    if str(recorded) != spec.identity():
        raise ProbabilityCalibrationError(
            f"persisted probability calibration {spec.version!r} records identity {recorded!r} "
            f"but its parameters hash to {spec.identity()}"
        )

    known = KNOWN_CALIBRATIONS.get(spec.version)
    if known is not None and known.as_dict() != spec.as_dict():
        raise ProbabilityCalibrationError(
            f"persisted probability calibration {spec.version!r} disagrees with the registered spec"
        )
    return spec


# ---------------------------------------------------------------------------
# Reliability binning
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReliabilityBin:
    """One stated-probability bin and the observed frequency inside it."""

    index: int
    lower: float
    upper: float
    n: int
    stated_mean: float | None
    observed_frequency: float | None
    gap: float | None
    status: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": int(self.index),
            "lower": float(self.lower),
            "upper": float(self.upper),
            "n": int(self.n),
            "stated_mean": None if self.stated_mean is None else round(float(self.stated_mean), 6),
            "observed_frequency": (
                None if self.observed_frequency is None else round(float(self.observed_frequency), 6)
            ),
            "gap": None if self.gap is None else round(float(self.gap), 6),
            "status": str(self.status),
        }


def _validate_probabilities_and_outcomes(
    probabilities: Iterable[Any],
    outcomes: Iterable[Any],
    *,
    error: type[ValueError] = ReliabilityInputError,
) -> tuple[list[float], list[float]]:
    p: list[float] = []
    for value in probabilities:
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise error(f"probability is not numeric: {value!r}") from exc
        if not math.isfinite(number):
            raise error(f"probability is not finite: {value!r}")
        if number < 0.0 or number > 1.0:
            # Fail closed.  Clipping would describe a different question than the
            # one the producer answered.
            raise error(f"probability {number!r} is outside [0, 1]")
        p.append(number)
    y: list[float] = []
    for value in outcomes:
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise error(f"realised outcome is not numeric: {value!r}") from exc
        if number not in (0.0, 1.0):
            raise error(f"realised outcome {number!r} is not binary")
        y.append(number)
    if len(p) != len(y):
        raise error(f"the probabilities and outcomes are not aligned: {len(p)} vs {len(y)}")
    return p, y


def _logit(p_clipped: float) -> float:
    return math.log(p_clipped / (1.0 - p_clipped))


def _clip_probability(value: float, floor: float) -> float:
    """The declared interior clip, applied strictly inside ``(0, 1)``."""

    return min(1.0 - float(floor), max(float(floor), float(value)))


def _bin_index(probability: float, edges: Sequence[float]) -> int:
    """The declared bin for one probability: ``[lower, upper)``, last closed."""

    if probability >= edges[-1]:
        return len(edges) - 2
    for index in range(len(edges) - 1):
        if edges[index] <= probability < edges[index + 1]:
            return index
    return len(edges) - 2


def reliability_bins(
    probabilities: Sequence[float],
    outcomes: Sequence[float],
    *,
    edges: Sequence[float] = RELIABILITY_BIN_EDGES,
    floor: int = RELIABILITY_MIN_BIN_N,
) -> list[ReliabilityBin]:
    """Observed frequency per stated-probability bin, with its own sample size.

    An empty bin keeps ``n = 0`` and NULL values -- a sample of zero is not a
    frequency of zero, and rendering it as ``0.0`` would invent a perfectly
    calibrated-looking bin out of nothing.  A bin under the declared floor is
    reported as insufficient, not as a calibration claim.
    """

    declared = [float(edge) for edge in edges]
    if len(declared) < 2 or declared[0] != 0.0 or declared[-1] != 1.0:
        raise ReliabilityInputError("bin edges must start at 0.0 and end at 1.0")
    if any(b <= a for a, b in zip(declared, declared[1:])):
        raise ReliabilityInputError("bin edges must be strictly increasing")
    if int(floor) < 1:
        raise ReliabilityInputError("the bin sample floor must be at least one observation")

    p, y = _validate_probabilities_and_outcomes(probabilities, outcomes)
    buckets: list[list[int]] = [[] for _ in range(len(declared) - 1)]
    for index, probability in enumerate(p):
        buckets[_bin_index(probability, declared)].append(index)

    bins: list[ReliabilityBin] = []
    for index, bucket in enumerate(buckets):
        stated = [p[i] for i in bucket]
        realised = [y[i] for i in bucket]
        if not bucket:
            bins.append(
                ReliabilityBin(
                    index=index,
                    lower=declared[index],
                    upper=declared[index + 1],
                    n=0,
                    stated_mean=None,
                    observed_frequency=None,
                    gap=None,
                    status=BIN_EMPTY_SAMPLE,
                )
            )
            continue
        stated_mean = sum(stated) / len(stated)
        observed = sum(realised) / len(realised)
        status = BIN_OK if len(bucket) >= int(floor) else BIN_INSUFFICIENT_SAMPLE
        bins.append(
            ReliabilityBin(
                index=index,
                lower=declared[index],
                upper=declared[index + 1],
                n=len(bucket),
                stated_mean=stated_mean,
                observed_frequency=observed,
                gap=observed - stated_mean,
                status=status,
            )
        )
    return bins


def reliability_view(
    probabilities: Sequence[float],
    outcomes: Sequence[float],
    *,
    edges: Sequence[float] = RELIABILITY_BIN_EDGES,
    floor: int = RELIABILITY_MIN_BIN_N,
) -> dict[str, Any]:
    """A reliability view plus the declared policy that produced it."""

    bins = reliability_bins(probabilities, outcomes, edges=edges, floor=floor)
    informative = [entry for entry in bins if entry.status == BIN_OK]
    return {
        "policy": {
            "policy_version": RELIABILITY_POLICY_VERSION,
            "bin_edges": [float(edge) for edge in edges],
            "bin_convention": RELIABILITY_BIN_CONVENTION,
            "gap_convention": RELIABILITY_GAP_CONVENTION,
            "min_bin_n": int(floor),
            "statuses": {
                BIN_OK: "at or above the declared bin sample floor",
                BIN_EMPTY_SAMPLE: "no observation in this bin; values are NULL, never 0.0",
                BIN_INSUFFICIENT_SAMPLE: "below the declared bin sample floor; not a calibration claim",
            },
            "basis": "declared disclosure floor, NOT a significance threshold",
        },
        "n": len(outcomes),
        "bins": [entry.as_dict() for entry in bins],
        "bins_over_floor": len(informative),
        "max_abs_gap_over_floor": (
            max(abs(float(entry.gap)) for entry in informative) if informative else None
        ),
        "claims": (
            "observed frequency per stated-probability bin; no skill score is derived and no "
            "arm is ranked from this view"
        ),
    }


# ---------------------------------------------------------------------------
# Causal fitting -- strictly earlier outcomes only
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CalibrationObservation:
    """One strictly-earlier outcome available to a fit.

    ``event`` is the target event the outcome belongs to and ``key`` is the
    row's own identity within its grain, so the fit basis names the exact rows
    that produced a transform rather than a count of them.
    """

    event: int
    key: tuple[int, ...]
    probability: float
    outcome: float


@dataclass(frozen=True)
class AssistMappingObservation:
    """One strictly-earlier expected/realised assist pair for the mapping fit."""

    event: int
    key: tuple[int, ...]
    expected_assists: float
    realised_assists: float


@dataclass(frozen=True)
class FitBasis:
    """The rows a transform was fitted from, named and hashed."""

    surface: str
    grain: str
    origin_event: int
    events: tuple[int, ...]
    observations: int
    digest: str
    detail: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "surface": str(self.surface),
            "grain": str(self.grain),
            "origin_event": int(self.origin_event),
            "events": [int(event) for event in self.events],
            "strictly_before_origin": all(int(event) < int(self.origin_event) for event in self.events),
            "observations": int(self.observations),
            "digest": str(self.digest),
            "policy_version": CAUSAL_FIT_POLICY_VERSION,
            "detail": {str(k): v for k, v in sorted(self.detail.items())},
        }


@dataclass(frozen=True)
class CausalFit:
    """The declared outcome of one causal fit: a transform, or a stated reason."""

    available: bool
    basis: FitBasis
    method: str
    spec: ProbabilityCalibration | None
    reason: str | None = None
    detail: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.available and self.spec is None:
            raise CausalFitError("an available fit must carry its spec")
        if not self.available and self.spec is not None:
            raise CausalFitError("an unavailable fit must not carry a spec")

    def as_dict(self) -> dict[str, Any]:
        return {
            "available": bool(self.available),
            "reason": None if self.reason is None else str(self.reason),
            "method": str(self.method),
            "basis": self.basis.as_dict(),
            "spec": None if self.spec is None else self.spec.as_dict(),
            "provenance": None if self.spec is None else self.spec.as_payload(),
            "detail": {str(k): v for k, v in sorted(self.detail.items())},
        }


def _canonical_rows(
    rows: Sequence[tuple[int, tuple[int, ...], float, ...]]
) -> list[tuple[int, tuple[int, ...], float, ...]]:
    """One canonical order, so arithmetic and digests are row-order invariant."""

    return sorted(rows, key=lambda row: (int(row[0]), tuple(int(p) for p in row[1]), *map(float, row[2:])))


def _basis(
    *,
    surface: str,
    grain: str,
    origin_event: int,
    rows: Sequence[tuple[int, tuple[int, ...], float, ...]],
    detail: Mapping[str, Any],
) -> FitBasis:
    events = sorted({int(row[0]) for row in rows})
    digest = analytics.canonical_hash(
        {
            "policy_version": CAUSAL_FIT_POLICY_VERSION,
            "surface": str(surface),
            "grain": str(grain),
            "origin_event": int(origin_event),
            "rows": [
                [int(event), [int(part) for part in key], float(a), float(b)]
                for event, key, a, b in rows
            ],
        }
    )
    return FitBasis(
        surface=str(surface),
        grain=str(grain),
        origin_event=int(origin_event),
        events=tuple(events),
        observations=len(rows),
        digest=digest,
        detail=detail,
    )


def _refuse_late_outcomes(rows: Sequence[Any], *, origin_event: int, surface: str) -> None:
    """Fail closed when a fit basis contains an outcome at or after the origin."""

    late = sorted({int(row.event) for row in rows if int(row.event) >= int(origin_event)})
    if late:
        raise CausalFitError(
            f"{surface}: the fit basis for origin event {int(origin_event)} contains outcome(s) from "
            f"event(s) {late}; a transform fitted on events 1..k and applied at event k is a contract "
            "violation, not a conservative approximation"
        )


def fit_platt_causal(
    observations: Sequence[CalibrationObservation],
    *,
    origin_event: int,
    surface: str,
    grain: str,
    clip_floor: float = DEFAULT_CLIP_FLOOR,
) -> CausalFit:
    """Fit a Platt transform on outcomes finalised STRICTLY BEFORE ``origin_event``.

    The declared method is :data:`PLATT_FIT_METHOD`: least squares of the binary
    outcome on the clipped logit of the stated probability, i.e.

        slope     = cov(logit p, y) / var(logit p)
        intercept = mean(y) - slope * mean(logit p)

    The fit REFUSES to use any observation whose event is not strictly earlier
    than the origin, and it returns an explicitly unavailable result -- never a
    guessed transform -- when the basis is too small, single-class, constant, or
    yields a non-positive slope (which would not be monotone, and therefore not a
    probability calibration).
    """

    rows = _canonical_rows(
        [
            (int(entry.event), tuple(int(part) for part in entry.key), float(entry.probability), float(entry.outcome))
            for entry in observations
        ]
    )
    probabilities = [row[2] for row in rows]
    outcomes = [row[3] for row in rows]
    probabilities, outcomes = _validate_probabilities_and_outcomes(
        probabilities, outcomes, error=CausalFitError
    )
    _refuse_late_outcomes(observations, origin_event=int(origin_event), surface=str(surface))

    positives = sum(1 for value in outcomes if value == 1.0)
    negatives = len(outcomes) - positives
    detail = {
        "positive_outcomes": positives,
        "negative_outcomes": negatives,
        "min_observations": MIN_FIT_OBSERVATIONS,
        "min_class_observations": MIN_FIT_CLASS_OBSERVATIONS,
        "input_clip_floor": float(clip_floor),
        "estimator": PLATT_FIT_METHOD,
    }
    basis = _basis(
        surface=surface,
        grain=grain,
        origin_event=int(origin_event),
        rows=rows,
        detail={"positive_outcomes": positives, "negative_outcomes": negatives},
    )
    if len(rows) < MIN_FIT_OBSERVATIONS:
        return CausalFit(
            available=False, basis=basis, method=PLATT_FIT_METHOD, spec=None,
            reason=FIT_INSUFFICIENT_OBSERVATIONS,
            detail={**detail, "observations": len(rows)},
        )
    if positives < MIN_FIT_CLASS_OBSERVATIONS or negatives < MIN_FIT_CLASS_OBSERVATIONS:
        return CausalFit(
            available=False, basis=basis, method=PLATT_FIT_METHOD, spec=None,
            reason=FIT_INSUFFICIENT_CLASS_SUPPORT,
        )

    logits = [_logit(_clip_probability(value, clip_floor)) for value in probabilities]
    count = len(logits)
    mean_x = sum(logits) / count
    mean_y = sum(outcomes) / count
    variance = sum((x - mean_x) ** 2 for x in logits) / count
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in zip(logits, outcomes)) / count
    if variance <= 0.0:
        return CausalFit(
            available=False, basis=basis, method=PLATT_FIT_METHOD, spec=None,
            reason=FIT_ZERO_VARIANCE_INPUT,
            detail={**detail, "logit_variance": variance},
        )
    slope = covariance / variance
    intercept = mean_y - slope * mean_x
    if not math.isfinite(slope) or not math.isfinite(intercept) or slope <= 0.0:
        return CausalFit(
            available=False, basis=basis, method=PLATT_FIT_METHOD, spec=None,
            reason=FIT_NON_MONOTONE,
            detail={**detail, "slope": slope, "intercept": intercept},
        )

    # The fitted version carries its own basis digest, so a version can never be
    # re-pointed at a different parameter set and two different bases can never
    # share a version.
    version = f"{PLATT_FIT_VERSION_PREFIX}.{basis.digest.split(':', 1)[-1][:16]}"
    spec = ProbabilityCalibration(
        version=version,
        method=METHOD_PLATT,
        intercept=intercept,
        slope=slope,
        clip_floor=clip_floor,
        fit_policy_version=CAUSAL_FIT_POLICY_VERSION,
    )
    check = spec.verify_monotone()
    if not check["monotone_non_decreasing"] or not check["boundaries_exact"]:
        # Belt and braces: the declaration says this cannot happen, and if it ever
        # did, the transform must not be returned as admissible.
        return CausalFit(
            available=False, basis=basis, method=PLATT_FIT_METHOD, spec=None,
            reason=FIT_NON_MONOTONE, detail={**detail, "monotone_check": check},
        )
    return CausalFit(
        available=True, basis=basis, method=PLATT_FIT_METHOD, spec=spec,
        detail={**detail, "slope": slope, "intercept": intercept, "monotone_check": check},
    )


def _assist_mapping_body(
    *,
    surface: str,
    grain: str,
    coefficient: float,
    basis_digest: str,
    events: Sequence[int],
    origin_event: int,
    observations: int,
) -> dict[str, Any]:
    """One definition of the bytes an assist-mapping identity hashes over.

    The fit writes them and the reader recomputes them, so the two cannot drift
    into hashing different things.
    """

    return {
        "surface": str(surface),
        "grain": str(grain),
        "method": ASSIST_MAPPING_FIT_METHOD,
        "coefficient": float(coefficient),
        "declared_bounds": [float(bound) for bound in ASSIST_MAPPING_FIT_BOUNDS],
        "basis_digest": str(basis_digest),
        "fit_basis_events": [int(event) for event in events],
        "fit_basis_origin_event": int(origin_event),
        "fit_basis_observations": int(observations),
        "policy_version": CAUSAL_FIT_POLICY_VERSION,
    }


@dataclass(frozen=True)
class AssistMappingFit:
    """The declared outcome of one causal assist-mapping fit."""

    available: bool
    basis: FitBasis
    method: str
    coefficient: float | None = None
    reason: str | None = None
    detail: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.available and self.coefficient is None:
            raise CausalFitError("an available assist-mapping fit must carry its coefficient")
        if not self.available and self.coefficient is not None:
            raise CausalFitError("an unavailable assist-mapping fit must not carry a coefficient")

    def as_dict(self) -> dict[str, Any]:
        return {
            "available": bool(self.available),
            "reason": None if self.reason is None else str(self.reason),
            "method": str(self.method),
            "coefficient": None if self.coefficient is None else float(self.coefficient),
            "in_bounds": (
                None
                if self.coefficient is None
                else bool(
                    ASSIST_MAPPING_FIT_BOUNDS[0] <= float(self.coefficient) <= ASSIST_MAPPING_FIT_BOUNDS[1]
                )
            ),
            "declared_bounds": [float(bound) for bound in ASSIST_MAPPING_FIT_BOUNDS],
            "basis": self.basis.as_dict(),
            "provenance": None if not self.available else self.as_payload(),
            "detail": {str(k): v for k, v in sorted(self.detail.items())},
        }

    def _body(self) -> dict[str, Any]:
        if not self.available or self.coefficient is None:
            raise CausalFitError("an unavailable assist-mapping fit has no provenance to bind")
        return _assist_mapping_body(
            surface=self.basis.surface,
            grain=self.basis.grain,
            coefficient=float(self.coefficient),
            basis_digest=self.basis.digest,
            events=self.basis.events,
            origin_event=self.basis.origin_event,
            observations=self.basis.observations,
        )

    def identity(self) -> str:
        """Canonical hash of the fitted payload -- the provenance tie."""

        return analytics.canonical_hash(self._body())

    def as_payload(self) -> dict[str, Any]:
        """The fitted coefficient, its basis and the policy it was fitted under, hashed."""

        payload = self._body()
        payload["identity"] = analytics.canonical_hash(self._body())
        return payload


def assist_mapping_from_payload(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    """Read a fitted assist-mapping payload back, fail-closed on anything unusable.

    A fitted coefficient is only usable when the payload says which fit produced
    it: the canonical ``identity``, the applicable ``policy_version``, the
    declared method and the basis digest the fit was computed over.  Any of them
    absent, unknown or disagreeing raises rather than degrading to a coefficient
    nobody can attribute.
    """

    if payload is None:
        raise CausalFitError(
            "a fitted assist-mapping payload is required; refusing to use an unattributed coefficient"
        )
    if not isinstance(payload, Mapping):
        raise CausalFitError(
            f"assist-mapping provenance must be a mapping, got {type(payload).__name__}"
        )
    recorded = payload.get("identity")
    if recorded is None or not str(recorded).strip():
        raise CausalFitError(
            "the fitted assist-mapping payload carries no canonical identity, so the coefficient "
            "cannot be tied to the fit that produced it"
        )
    declared_policy = str(payload.get("policy_version") or "").strip()
    if declared_policy != CAUSAL_FIT_POLICY_VERSION:
        raise CausalFitError(
            f"the fitted assist-mapping payload declares policy_version {declared_policy or None!r}, "
            f"not the declared {CAUSAL_FIT_POLICY_VERSION!r}"
        )
    if str(payload.get("method") or "") != ASSIST_MAPPING_FIT_METHOD:
        raise CausalFitError(
            f"the fitted assist-mapping payload declares method {payload.get('method')!r}, not the "
            f"declared {ASSIST_MAPPING_FIT_METHOD!r}"
        )
    if not str(payload.get("basis_digest") or "").strip():
        raise CausalFitError("the fitted assist-mapping payload carries no fit-basis digest")
    try:
        coefficient = float(payload.get("coefficient"))
        events = [int(event) for event in (payload.get("fit_basis_events") or [])]
        origin_event = int(payload.get("fit_basis_origin_event"))
        observations = int(payload.get("fit_basis_observations"))
    except (TypeError, ValueError) as exc:
        raise CausalFitError(
            f"the fitted assist-mapping payload carries no readable fit basis or coefficient: {payload!r}"
        ) from exc
    body = _assist_mapping_body(
        surface=str(payload.get("surface") or ""),
        grain=str(payload.get("grain") or ""),
        coefficient=coefficient,
        basis_digest=str(payload.get("basis_digest")),
        events=events,
        origin_event=origin_event,
        observations=observations,
    )
    recomputed = analytics.canonical_hash(body)
    if str(recorded) != recomputed:
        raise CausalFitError(
            f"the fitted assist-mapping payload records identity {recorded!r} but its content hashes "
            f"to {recomputed}"
        )
    return {**body, "identity": str(recorded)}


def fit_assist_mapping_causal(
    observations: Sequence[AssistMappingObservation],
    *,
    origin_event: int,
    surface: str = "FPL_ASSIST_MAPPING",
    grain: str = "player_fixture",
) -> AssistMappingFit:
    """Fit the FPL assist-mapping coefficient on strictly earlier outcomes.

    The declared method is :data:`ASSIST_MAPPING_FIT_METHOD`: the ratio of summed
    realised assists to summed expected assists over the basis, which is the
    moment estimator for the single multiplicative coefficient the production
    formula applies (``expected_fpl_assists = expected_xa * coefficient``).

    A basis that saw no positive expected assists, or that yields a coefficient
    outside the declared degeneracy band, is reported unavailable with its
    reason.  Nothing here re-points the production constant.
    """

    rows = _canonical_rows(
        [
            (
                int(entry.event),
                tuple(int(part) for part in entry.key),
                float(entry.expected_assists),
                float(entry.realised_assists),
            )
            for entry in observations
        ]
    )
    for _event, _key, expected, realised in rows:
        if not math.isfinite(expected) or not math.isfinite(realised):
            raise CausalFitError(f"{surface}: a non-finite assist observation: {expected!r}, {realised!r}")
        if expected < 0.0 or realised < 0.0:
            # A negative expected assist is a producer defect, and a negative
            # realised count is not a count.  Either way, fail closed.
            raise CausalFitError(
                f"{surface}: a negative assist observation: expected {expected!r}, realised {realised!r}"
            )
    _refuse_late_outcomes(observations, origin_event=int(origin_event), surface=str(surface))

    expected_total = sum(row[2] for row in rows)
    realised_total = sum(row[3] for row in rows)
    detail = {
        "method_estimator": ASSIST_MAPPING_FIT_METHOD,
        "expected_assists_total": expected_total,
        "realised_assists_total": realised_total,
        "min_observations": MIN_FIT_OBSERVATIONS,
        "declared_bounds": [float(bound) for bound in ASSIST_MAPPING_FIT_BOUNDS],
    }
    basis = _basis(
        surface=surface,
        grain=grain,
        origin_event=int(origin_event),
        rows=rows,
        detail={
            "expected_assists_total": expected_total,
            "realised_assists_total": realised_total,
        },
    )
    if len(rows) < MIN_FIT_OBSERVATIONS:
        return AssistMappingFit(
            available=False, basis=basis, method=ASSIST_MAPPING_FIT_METHOD, coefficient=None,
            reason=FIT_INSUFFICIENT_OBSERVATIONS, detail={**detail, "observations": len(rows)},
        )
    if expected_total <= 0.0:
        return AssistMappingFit(
            available=False, basis=basis, method=ASSIST_MAPPING_FIT_METHOD, coefficient=None,
            reason=FIT_ZERO_EXPECTED_TOTAL, detail=detail,
        )
    coefficient = realised_total / expected_total
    if not math.isfinite(coefficient):
        return AssistMappingFit(
            available=False, basis=basis, method=ASSIST_MAPPING_FIT_METHOD, coefficient=None,
            reason=FIT_ZERO_EXPECTED_TOTAL, detail=detail,
        )
    if not ASSIST_MAPPING_FIT_BOUNDS[0] <= coefficient <= ASSIST_MAPPING_FIT_BOUNDS[1]:
        return AssistMappingFit(
            available=False, basis=basis, method=ASSIST_MAPPING_FIT_METHOD, coefficient=None,
            reason=FIT_OUT_OF_DECLARED_BOUNDS, detail={**detail, "coefficient": coefficient},
        )
    return AssistMappingFit(
        available=True, basis=basis, method=ASSIST_MAPPING_FIT_METHOD, coefficient=coefficient,
        detail=detail,
    )


def policy() -> dict[str, Any]:
    """The declared policy this module applies, as a serialisable record."""

    return {
        "policy_version": PROBABILITY_CALIBRATION_POLICY_VERSION,
        "causal_fit_policy_version": CAUSAL_FIT_POLICY_VERSION,
        "reliability_policy_version": RELIABILITY_POLICY_VERSION,
        "declared_methods": list(CALIBRATION_METHODS),
        "incumbent": INCUMBENT_IDENTITY.as_payload(),
        "platt_fit_method": PLATT_FIT_METHOD,
        "platt_fit_version_prefix": PLATT_FIT_VERSION_PREFIX,
        "assist_mapping_fit_method": ASSIST_MAPPING_FIT_METHOD,
        "assist_mapping_fit_bounds": [float(bound) for bound in ASSIST_MAPPING_FIT_BOUNDS],
        "min_fit_observations": MIN_FIT_OBSERVATIONS,
        "min_fit_class_observations": MIN_FIT_CLASS_OBSERVATIONS,
        "bin_edges": [float(edge) for edge in RELIABILITY_BIN_EDGES],
        "bin_convention": RELIABILITY_BIN_CONVENTION,
        "min_bin_n": RELIABILITY_MIN_BIN_N,
        "causality": (
            "a transform applied at origin event E is fitted only on outcomes finalised strictly "
            "before E; a basis containing an outcome at or after E is refused, not approximated"
        ),
        "fit_basis_evidence": (
            "PE-5 append-only point-in-time observation captures only: a basis row must carry a FINAL "
            "official observation whose official finality AND capture both fall strictly before the "
            "origin's certified cutoff, and a row whose timing cannot be proven is excluded and counted "
            "rather than assumed known"
        ),
        "fitted_payload_provenance": (
            "a fitted payload carries the canonical identity AND the applicable policy versions -- the "
            "calibration policy and, for a fitted version, the fit policy -- and the reader fails closed "
            "when any is absent or disagrees; the identity hash covers the fit policy, so a re-pointed "
            "provenance changes the identity"
        ),
        "diagnosis_before_fit": (
            "a surface is diagnosed BEFORE anything is fitted, and no challenger is constructed or "
            "reported unless the diagnosis is MISCALIBRATED"
        ),
        "ranking_behaviour": {            "within_event_ordering": (
                "preserved: a monotone non-decreasing transform cannot reorder two players' stated "
                "probabilities, so a ranking-based decision is unchanged by it"
            ),
            "expected_value_effect": (
                "a transform DOES change the expected value it feeds, so an expected-value-based "
                "decision can change; the two are not the same claim"
            ),
            "production_use": "NONE; no transform fitted here is wired into a decision path",
        },
        "mandate": (
            "a transform exists to correct a MEASURED miscalibration on a declared surface; where no "
            "defect is diagnosed the fitted parameters are reported as a descriptive sensitivity and "
            "are not a candidate calibration"
        ),
        "identity_fallback": (
            "NONE; an absent or unknown calibration version raises rather than degrading to identity"
        ),
        "boundaries": "raw 0 stays 0 and raw 1 stays 1; the declared clip applies strictly inside (0, 1)",
    }
