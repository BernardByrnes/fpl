"""Decision-confidence classification (R2B, corrected in R4B.2c).

This module **does not change the optimizer objective and does not rank routes**.
It reads the paired common-random-number near-tie result produced by route
comparison, plus role evidence from the CERTIFIED minutes generation, and states
how strong a recommendation is.

R4B.2c corrections
------------------
1. **One canonical near-tie definition.** ``near_tie`` is CONSUMED from the paired
   CRN test (``abs(paired_delta_mean) <= NEAR_TIE_K * paired_delta_se``, with
   ``NEAR_TIE_K = 1.96``).  It is never recomputed from a fixed CORE margin.  The
   fixed ``MATERIAL_FRONTIER_CHANGE_CORE = 0.25`` is retained for MATERIALITY only
   and is reported as ``material_margin``.
2. **A fifth state, ``ROLE_UNCERTAINTY``.** Previously role uncertainty forced
   ``NEAR_TIE_WITH_ROLE_UNCERTAINTY`` regardless of ``near_tie``, so the label could
   claim a near tie while ``near_tie`` was false.  The states are now:
   ``MODEL_EVIDENCE_CONFLICT`` > ``NEAR_TIE_WITH_ROLE_UNCERTAINTY`` >
   ``ROLE_UNCERTAINTY`` > ``NEAR_TIE`` > ``STRONG_RECOMMENDATION``, which restores
   the invariants listed in ``NEAR_TIE_INVARIANTS``.
3. **No production caller before this phase.** The module existed but was wired
   into no decision artifact; the decision runner now consumes it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .route_comparator import NEAR_TIE_K

CONFIDENCE_STRONG = "STRONG_RECOMMENDATION"
CONFIDENCE_NEAR_TIE = "NEAR_TIE"
CONFIDENCE_NEAR_TIE_ROLE_UNCERTAINTY = "NEAR_TIE_WITH_ROLE_UNCERTAINTY"
CONFIDENCE_ROLE_UNCERTAINTY = "ROLE_UNCERTAINTY"
CONFIDENCE_MODEL_EVIDENCE_CONFLICT = "MODEL_EVIDENCE_CONFLICT"
# Cannot be classified at all: the canonical paired CRN diagnostic is missing.
CONFIDENCE_INCOMPLETE = "INCOMPLETE_CONFIDENCE_EVIDENCE"

#: Explicit diagnostic emitted when the required paired diagnostic is absent.
DIAG_PAIRED_DIAGNOSTIC_REQUIRED = "PAIRED_DIAGNOSTIC_REQUIRED"

#: Canonical key under which route comparison must publish the paired CRN record
#: for the FINAL preferred leader versus its relevant runner-up/comparator.
CANONICAL_PAIRED_DIAGNOSTIC_KEY = "canonical_paired_near_tie"

# Order is the precedence order (highest first).
CONFIDENCE_STATES = (
    CONFIDENCE_INCOMPLETE,
    CONFIDENCE_MODEL_EVIDENCE_CONFLICT,
    CONFIDENCE_NEAR_TIE_ROLE_UNCERTAINTY,
    CONFIDENCE_ROLE_UNCERTAINTY,
    CONFIDENCE_NEAR_TIE,
    CONFIDENCE_STRONG,
)

#: Required invariants (asserted by tests):
#:   NEAR_TIE                        => near_tie is True
#:   NEAR_TIE_WITH_ROLE_UNCERTAINTY  => near_tie is True
#:   ROLE_UNCERTAINTY                => near_tie is False
NEAR_TIE_INVARIANTS = {
    CONFIDENCE_NEAR_TIE: True,
    CONFIDENCE_NEAR_TIE_ROLE_UNCERTAINTY: True,
    CONFIDENCE_ROLE_UNCERTAINTY: False,
}

#: The state names that contain the phrase "NEAR_TIE".
NEAR_TIE_LABELLED_STATES = (CONFIDENCE_NEAR_TIE, CONFIDENCE_NEAR_TIE_ROLE_UNCERTAINTY)

#: Materiality threshold for declaring a frontier change.  NOT a near-tie test.
MATERIAL_FRONTIER_CHANGE_CORE = 0.25

# Role evidence that counts as a genuine model/evidence contradiction.
_CONFLICT_FLAGS = frozenset(
    {
        "MODEL_PRIOR_CONFLICT_WITH_CURRENT_ROLE",
        "SCOUTING_ROLE_CONFLICT",
    }
)


@dataclass(frozen=True)
class DecisionConfidenceConfig:
    """Materiality threshold only.

    ``material_margin`` is the frontier-change materiality threshold.  It is
    deliberately NOT a near-tie threshold: near-tie comes from the paired CRN test.
    """

    material_margin: float = MATERIAL_FRONTIER_CHANGE_CORE


def paired_near_tie(paired: Mapping[str, Any] | None) -> dict[str, Any]:
    """Read the paired CRN near-tie result.  Never recomputes it from a margin."""

    if not paired:
        return {
            "available": False,
            "near_tie": None,
            "paired_delta_mean": None,
            "paired_delta_se": None,
            "near_tie_k": NEAR_TIE_K,
            "source": "route_comparison.paired_near_merge",
        }
    mean = paired.get("mean_difference", paired.get("paired_delta_mean"))
    se = paired.get("paired_se", paired.get("paired_delta_se"))
    if "near_tied" in paired:
        verdict = bool(paired["near_tied"])
    elif mean is not None and se is not None:
        verdict = (abs(float(mean)) <= NEAR_TIE_K * float(se)) if float(se) > 0 else float(mean) == 0.0
    else:
        verdict = None
    return {
        "available": verdict is not None,
        "near_tie": verdict,
        "paired_delta_mean": None if mean is None else float(mean),
        "paired_delta_se": None if se is None else float(se),
        "near_tie_k": NEAR_TIE_K,
        "source": "route_comparison.paired_near_merge",
    }


def classify_decision_confidence(
    *,
    paired: Mapping[str, Any] | None = None,
    role_evidence: Mapping[int, Mapping[str, Any]] | None = None,
    focus_player_ids: Iterable[int] | None = None,
    role_evidence_source: Mapping[str, Any] | None = None,
    config: DecisionConfidenceConfig | None = None,
    require_paired: bool = True,
    # Deprecated compatibility inputs: accepted so existing callers still run, but
    # they do NOT define near_tie any more.
    selected_core: float | None = None,
    alternative_cores: Mapping[str, float] | Iterable[float] | None = None,
    starter_ids: Iterable[int] | None = None,
) -> dict[str, Any]:
    """Classify recommendation strength.  Ranking is never affected.

    ``paired`` is a route-comparison paired record (``mean_difference`` /
    ``paired_se`` / ``near_tied``).  ``role_evidence`` maps player_id to the
    minutes payload's ``role_evidence`` block, and only the players in
    ``focus_player_ids`` are considered.
    """

    settings = config or DecisionConfidenceConfig()
    near = paired_near_tie(paired)

    focus_source = focus_player_ids if focus_player_ids is not None else starter_ids
    focus = {int(pid) for pid in focus_source} if focus_source is not None else None

    conflicts: list[dict[str, Any]] = []
    uncertain: list[dict[str, Any]] = []
    for pid, block in (role_evidence or {}).items():
        pid = int(pid)
        if focus is not None and pid not in focus:
            continue
        flags = set(block.get("conflict_flags") or [])
        if flags & _CONFLICT_FLAGS or block.get("prior_role_discontinuity") is True:
            conflicts.append({"player_id": pid, "flags": sorted(flags)})
        elif str(block.get("role_confidence") or "").upper() == "LOW":
            uncertain.append(
                {"player_id": pid, "reasons": list(block.get("uncertainty_reasons") or [])}
            )

    margin = None
    if selected_core is not None and alternative_cores is not None:
        cores = (
            {str(k): float(v) for k, v in alternative_cores.items()}
            if isinstance(alternative_cores, Mapping)
            else {f"alt_{i}": float(v) for i, v in enumerate(alternative_cores)}
        )
        if cores:
            margin = float(selected_core) - max(cores.values())

    near_tie = near["near_tie"] if near["near_tie"] is not None else False
    material = margin is not None and abs(margin) > settings.material_margin
    paired_required_but_missing = bool(require_paired and not near["available"])

    # Precedence.  A MISSING paired diagnostic is a procedural blocker, not an
    # inferred "not near tied": without the canonical paired test the near-tie
    # dimension is unmeasured, so no decisive label is defensible.  It therefore
    # outranks every substantive state, while the conflicts that were found are
    # still reported in the payload.
    if require_paired and not near["available"]:
        state = CONFIDENCE_INCOMPLETE
    elif conflicts:
        state = CONFIDENCE_MODEL_EVIDENCE_CONFLICT
    elif near_tie and uncertain:
        state = CONFIDENCE_NEAR_TIE_ROLE_UNCERTAINTY
    elif uncertain:
        state = CONFIDENCE_ROLE_UNCERTAINTY
    elif near_tie:
        state = CONFIDENCE_NEAR_TIE
    else:
        state = CONFIDENCE_STRONG

    return {
        "state": state,
        "near_tie": bool(near_tie),
        "paired_required": bool(require_paired),
        "decisive": not paired_required_but_missing,
        "confidence_diagnostic": (
            DIAG_PAIRED_DIAGNOSTIC_REQUIRED if paired_required_but_missing else None
        ),
        "paired_delta_mean": near["paired_delta_mean"],
        "paired_delta_se": near["paired_delta_se"],
        "near_tie_k": near["near_tie_k"],
        "paired_source": near["source"],
        "paired_available": near["available"],
        "material_margin": settings.material_margin,
        "material_frontier_change": material,
        "margin_core": margin,
        "role_conflicts": conflicts,
        "role_uncertainties": uncertain,
        "role_evidence_source": dict(role_evidence_source or {}),
        # Confidence is computed AFTER ranking and never influences it.
        "affects_ranking": False,
    }


def assert_confidence_invariants(payload: Mapping[str, Any]) -> None:
    """Raise if a produced classification violates the documented invariants."""

    state = str(payload.get("state"))
    near_tie = bool(payload.get("near_tie"))
    if state in NEAR_TIE_INVARIANTS:
        required = NEAR_TIE_INVARIANTS[state]
        if near_tie is not required:
            raise AssertionError(
                f"confidence invariant violated: state {state!r} requires near_tie={required} "
                f"but near_tie={near_tie}"
            )
    if state not in CONFIDENCE_STATES:
        raise AssertionError(f"unknown confidence state {state!r}")
    if payload.get("affects_ranking") is not False:
        raise AssertionError("confidence must never affect ranking")


__all__ = [
    "CANONICAL_PAIRED_DIAGNOSTIC_KEY",
    "CONFIDENCE_INCOMPLETE",
    "CONFIDENCE_MODEL_EVIDENCE_CONFLICT",
    "CONFIDENCE_NEAR_TIE",
    "CONFIDENCE_NEAR_TIE_ROLE_UNCERTAINTY",
    "CONFIDENCE_ROLE_UNCERTAINTY",
    "CONFIDENCE_STATES",
    "CONFIDENCE_STRONG",
    "DIAG_PAIRED_DIAGNOSTIC_REQUIRED",
    "DecisionConfidenceConfig",
    "MATERIAL_FRONTIER_CHANGE_CORE",
    "NEAR_TIE_INVARIANTS",
    "NEAR_TIE_LABELLED_STATES",
    "assert_confidence_invariants",
    "classify_decision_confidence",
    "paired_near_tie",
]
