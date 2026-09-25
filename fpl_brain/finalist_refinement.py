"""R4B.2b — finalist refinement and the paired search-stability gate.

Stage 1 (unchanged, ``route_optimizer.optimize``) is the accepted bounded search
at ``STAGE1_DRAWS`` shared worlds per event.  This module adds:

* **finalist selection** — the Stage-1 preferred leader, the ROLL/no-transfer
  baseline, and every route that is PAIRED-NEAR-TIED with that leader under the
  common-random-number criterion ``abs(mean) <= NEAR_TIE_K * paired_se``.  A route
  rescued by an R4B.2a structural lens is *not* promoted by this rule; only the
  paired near-tie criterion can promote it.
* **Stage-2 refinement** — the finalists alone are re-evaluated in the SAME
  football world family (same seed, same certified runs, same route legality) at
  the explicitly higher ``STAGE2_DRAWS`` budget.  This is a precision
  refinement, not a new football world, and it never mutates the discovery
  universe.
* **one canonical paired record** — after the final ranking is known, exactly one
  ``canonical_paired_near_tie`` record is published for the final preferred
  leader versus the final relevant runner-up.  It is the diagnostic the
  R4B.2c decision-confidence classifier consumes.  When it cannot be produced the
  decision must not become ``STRONG_RECOMMENDATION``.
* **the paired search-stability gate** — at most ONE bounded escalation to the
  next supported search budget (the next beam width in
  ``route_stability.LADDER_BUDGETS``), followed by a documented
  ``SEARCH_BUDGET_STABLE`` / ``SEARCH_NOT_STABLE_AT_CURRENT_BUDGET`` verdict.

The four-GW objective is never changed here: ``DECISION_OBJECTIVE_KEY`` is the
optimizer's name for the quantity ``four_gw_decision`` reports as
``four_gw_net_core`` (sum of per-event mean gross CORE minus hits).  Stability is
a decision-safety gate; it never adds or removes objective points, and confidence
is always computed after ranking.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from . import route_optimizer as ro
from . import route_stability as rs
from . import route_comparator as rc

PHASE_R4B2B_VERSION = "finalist_refinement_r4b2b_1.0.0"

#: The optimizer's name for the four-GW objective.  For a decision window of four
#: events this is exactly ``four_gw_decision.four_gw_net_core`` (per-event mean
#: gross CORE summed over the window, minus hits incurred in the window).
DECISION_OBJECTIVE_KEY = "supported_3gw_net_core"

#: Stage-1 (screening) shared-world budget per event.  Must match
#: ``route_optimizer.OptimizerConfig.search_draws`` and the production runner.
STAGE1_DRAWS = 2_000

#: Stage-2 (finalist precision) budget per event.  Explicitly higher; the artifact
#: always reports the count that ACTUALLY ran.
STAGE2_DRAWS = 10_000

#: Paired near-tie criterion.  Consumed from the paired record when the route
#: comparison already published ``near_tied``.
NEAR_TIE_K = rc.NEAR_TIE_K

#: Frontier-change MATERIALITY threshold.  Never a near-tie test.
MATERIAL_FRONTIER_CHANGE_CORE = ro.MATERIAL_FRONTIER_CHANGE_CORE

SEARCH_STABLE = ro.SEARCH_STABLE
SEARCH_NOT_STABLE = ro.SEARCH_UNSTABLE

#: The paired-record table the optimizer publishes for the decision window.
PAIRED_SOURCE_KEY = "paired_supported_3gw"

#: Explicit diagnostics.
DIAG_PAIRED_DIAGNOSTIC_UNAVAILABLE = "PAIRED_DIAGNOSTIC_REQUIRED"
DIAG_FINALIST_PARTIAL_UNAVAILABLE = "FINALIST_PARTIAL_UNAVAILABLE"
DIAG_PREFIX_INVARIANCE_FAILED = "DECISION_WORLD_PREFIX_INVARIANCE_FAILED"
DIAG_ESCALATION_FAILED = "SEARCH_ESCALATION_FAILED"
DIAG_ESCALATED_LEADER_UNPAIRED = "ESCALATED_LEADER_PAIRED_EVIDENCE_UNAVAILABLE"

#: Finalist selection reasons.  Declared, non-overlapping, and recorded per route.
REASON_STAGE1_LEADER = "STAGE1_PREFERRED_LEADER"
REASON_ROLL_BASELINE = "ROLL_BASELINE"
REASON_PAIRED_NEAR_TIE = "PAIRED_NEAR_TIED_WITH_LEADER"

#: The scope of the experimentally-proven world-prefix property.  Deliberately
#: narrower than "every internal RNG stream": only the captured surfaces route
#: scoring actually consumes are compared.
PREFIX_INVARIANCE_SCOPE = "DECISION_RELEVANT_CAPTURED_SURFACES"
PREFIX_STATUS_PASS = "PASS"
PREFIX_STATUS_FAIL = "FAIL"

#: Leader-change direction labels.
DIRECTION_UNCHANGED = "UNCHANGED"
DIRECTION_REFINED_LEADER_BETTER = "REFINED_LEADER_BETTER"
DIRECTION_STAGE1_LEADER_BETTER = "STAGE1_LEADER_BETTER"


# ---------------------------------------------------------------------------
# Identity helpers
# ---------------------------------------------------------------------------


def canonical_route_signature(record: Mapping[str, Any]) -> str:
    """Route-id-free family identity of a route record."""

    return str(record.get("canonical_family_signature") or record.get("family_signature"))


def decision_rank_key(record: Mapping[str, Any]) -> tuple:
    """The four-GW ranking key, identical to ``four_gw_decision.rank_routes_by_four_gw``.

    Kept as a local copy so this module does not depend on the decision layer;
    ``test_r4b2b_finalist_stability`` asserts the two orderings agree.
    """

    return (
        -float(record.get(DECISION_OBJECTIVE_KEY, float("-inf"))),
        -int(record.get("terminal_ft", 0) or 0),
        -int(record.get("terminal_bank_tenths", 0) or 0),
        str(record.get("route_id", "")),
    )


def ranked_route_ids(result: Mapping[str, Any]) -> list[str]:
    """Route ids of one optimizer result, ordered by the four-GW objective."""

    records = result.get("routes") or {}
    return [
        str(name)
        for name, _record in sorted(
            records.items(),
            key=lambda item: decision_rank_key({**item[1], "route_id": item[0]}),
        )
    ]


def _route_record(result: Mapping[str, Any], route_id: str) -> Mapping[str, Any]:
    return (result.get("routes") or {}).get(str(route_id)) or {}


def _signature_to_route_id(result: Mapping[str, Any]) -> dict[str, str]:
    return {
        canonical_route_signature(record): str(name)
        for name, record in (result.get("routes") or {}).items()
    }


def _route_id_to_signature(result: Mapping[str, Any]) -> dict[str, str]:
    return {
        str(name): canonical_route_signature(record)
        for name, record in (result.get("routes") or {}).items()
    }


def runner_up_for(result: Mapping[str, Any], leader_route_id: str | None) -> str | None:
    """The highest-ranked route in ``result`` other than ``leader_route_id``.

    Used to name the FINAL relevant comparator once the decision layer has named
    the preferred leader, so the canonical record's ``route_b`` is a real
    alternative and never an arbitrary pick.
    """

    ranked = ranked_route_ids(result)
    for route_id in ranked:
        if str(route_id) != str(leader_route_id):
            return route_id
    return None


def _roll_route_id(result: Mapping[str, Any]) -> str | None:
    baseline = result.get("roll_baseline") or {}
    if baseline.get("route_id"):
        return str(baseline["route_id"])
    for name, record in (result.get("routes") or {}).items():
        actions = record.get("actions") or []
        if actions and all(str(action.get("kind")) == "ROLL" for action in actions):
            return str(name)
    return None


# ---------------------------------------------------------------------------
# Paired-record reading (never recomputes the near-tie verdict from a margin)
# ---------------------------------------------------------------------------


def paired_route_index(result: Mapping[str, Any], key: str = PAIRED_SOURCE_KEY) -> dict[frozenset, dict]:
    """``frozenset({route_a, route_b}) -> paired record`` for one result."""

    index: dict[frozenset, dict] = {}
    for record in result.get(key) or []:
        a, b = record.get("route_a"), record.get("route_b")
        if a is None or b is None:
            continue
        index[frozenset((str(a), str(b)))] = dict(record)
    return index


def paired_between(result: Mapping[str, Any], route_a: str, route_b: str,
                   key: str = PAIRED_SOURCE_KEY) -> dict | None:
    return paired_route_index(result, key).get(frozenset((str(route_a), str(route_b))))


def near_tie_verdict(paired: Mapping[str, Any] | None, *, k: float = NEAR_TIE_K) -> dict[str, Any]:
    """Read (never invent) the near-tie verdict of one paired record.

    The verdict is CONSUMED from the published ``near_tied`` field.  Only when a
    record lacks that field is the criterion recomputed, and then the source is
    reported as ``RECOMPUTED`` so a reader can tell the two apart.
    """

    if not paired:
        return {"available": False, "near_tied": None, "source": "MISSING", "near_tie_k": float(k)}
    if "near_tied" in paired:
        return {"available": True, "near_tied": bool(paired["near_tied"]), "source": "PUBLISHED",
                "near_tie_k": float(k)}
    mean, se = paired.get("mean_difference"), paired.get("paired_se")
    if mean is None or se is None:
        return {"available": False, "near_tied": None, "source": "INCOMPLETE", "near_tie_k": float(k)}
    mean, se = float(mean), float(se)
    verdict = (abs(mean) <= float(k) * se) if se > 0 else mean == 0.0
    return {"available": True, "near_tied": bool(verdict), "source": "RECOMPUTED", "near_tie_k": float(k)}


# ---------------------------------------------------------------------------
# Finalist selection
# ---------------------------------------------------------------------------


def select_finalists(result: Mapping[str, Any], *, near_tie_k: float = NEAR_TIE_K,
                     paired_key: str = PAIRED_SOURCE_KEY) -> dict[str, Any]:
    """The canonical finalist set of one Stage-1 result.

    Membership is EXACTLY:
      A. the Stage-1 preferred leader (rank 1 by the four-GW objective);
      B. the ROLL/no-transfer baseline;
      C. every other route PAIRED-NEAR-TIED with the Stage-1 leader.

    Nothing else.  In particular a route that survived only because an R4B.2a
    structural lens rescued it is NOT a finalist unless criterion C holds.
    """

    routes = result.get("routes") or {}
    if not routes:
        return {
            "phase_version": PHASE_R4B2B_VERSION,
            "finalists": [],
            "finalist_route_ids": [],
            "leader_route_id": None,
            "leader_signature": None,
            "runner_up_route_id": None,
            "runner_up_signature": None,
            "leader_value": None,
            "near_tie_k": float(near_tie_k),
            "near_tied_route_ids": [],
            "rescue_lens_routes_not_promoted": [],
            "paired_source_key": paired_key,
            "global_optimality_claimed": False,
            "diagnostic": DIAG_FINALIST_PARTIAL_UNAVAILABLE,
        }

    ranked = ranked_route_ids(result)
    leader_id = ranked[0]
    leader_signature = canonical_route_signature(_route_record(result, leader_id))

    reasons: dict[str, list[str]] = {}
    reasons.setdefault(leader_id, []).append(REASON_STAGE1_LEADER)

    roll_id = _roll_route_id(result)
    if roll_id is not None and roll_id in routes:
        reasons.setdefault(roll_id, []).append(REASON_ROLL_BASELINE)

    near_tied_ids: list[str] = []
    near_tie_sources: dict[str, str] = {}
    for route_id in ranked:
        if route_id == leader_id:
            continue
        paired = paired_between(result, leader_id, route_id, paired_key)
        verdict = near_tie_verdict(paired, k=near_tie_k)
        if verdict["near_tied"] is True:
            near_tied_ids.append(route_id)
            near_tie_sources[route_id] = str(verdict["source"])
            reasons.setdefault(route_id, []).append(REASON_PAIRED_NEAR_TIE)

    finalist_ids = sorted(reasons, key=lambda rid: decision_rank_key({**routes[rid], "route_id": rid}))
    finalists = [
        {
            "route_id": route_id,
            "canonical_family_signature": canonical_route_signature(routes[route_id]),
            "reasons": sorted(reasons[route_id]),
            "objective_value": float(routes[route_id].get(DECISION_OBJECTIVE_KEY, 0.0)),
            "near_tie_source": near_tie_sources.get(route_id),
        }
        for route_id in finalist_ids
    ]

    runner_up_id = finalist_ids[1] if len(finalist_ids) > 1 else None

    # Routes promoted by the optimizer that are NOT finalists, for the audit
    # trail.  ``uses_rescued_player`` is the optimizer's own flag; reporting them
    # proves that lens rescue does not by itself promote a route.
    not_promoted = [
        {
            "route_id": route_id,
            "uses_rescued_player": bool(routes[route_id].get("uses_rescued_player")),
            "near_tied_with_leader": bool(paired_between(result, leader_id, route_id, paired_key)
                                          and near_tie_verdict(paired_between(result, leader_id, route_id, paired_key),
                                                               k=near_tie_k)["near_tied"] is True),
        }
        for route_id in ranked
        if route_id not in reasons
    ]

    return {
        "phase_version": PHASE_R4B2B_VERSION,
        "finalists": finalists,
        "finalist_route_ids": [item["route_id"] for item in finalists],
        "leader_route_id": leader_id,
        "leader_signature": leader_signature,
        "runner_up_route_id": runner_up_id,
        "runner_up_signature": (
            None if runner_up_id is None else canonical_route_signature(_route_record(result, runner_up_id))
        ),
        "leader_value": float(_route_record(result, leader_id).get(DECISION_OBJECTIVE_KEY, 0.0)),
        "near_tie_k": float(near_tie_k),
        "near_tied_route_ids": near_tied_ids,
        "rescue_lens_routes_not_promoted": not_promoted,
        "paired_source_key": paired_key,
        "global_optimality_claimed": False,
        "diagnostic": None,
    }


# ---------------------------------------------------------------------------
# Canonical paired record
# ---------------------------------------------------------------------------


def canonical_paired_record(result: Mapping[str, Any], *, leader_route_id: str,
                            runner_up_route_id: str,
                            paired_key: str = PAIRED_SOURCE_KEY) -> dict[str, Any] | None:
    """The ONE authoritative paired record: final leader vs final runner-up.

    ``route_a`` is ALWAYS the final preferred leader and ``route_b`` the runner-up,
    with the sign of ``mean_difference`` normalised to that orientation.  Returns
    ``None`` when the paired comparison never ran for that pair, so a caller can
    fail closed with ``PAIRED_DIAGNOSTIC_REQUIRED`` instead of inferring
    "not near tied".
    """

    paired = paired_between(result, leader_route_id, runner_up_route_id, paired_key)
    if not paired:
        return None
    mean = paired.get("mean_difference")
    se = paired.get("paired_se")
    worlds = paired.get("worlds")
    if mean is None or se is None or worlds is None:
        return None
    mean = float(mean)
    normalised = str(paired.get("route_a")) != str(leader_route_id)
    if normalised:
        mean = -mean
    verdict = near_tie_verdict(paired)
    return {
        "route_a": str(leader_route_id),
        "route_b": str(runner_up_route_id),
        "mean_difference": mean,
        "paired_se": float(se),
        "worlds": int(worlds),
        "near_tied": bool(verdict["near_tied"]),
        "near_tie_k": float(NEAR_TIE_K),
        "near_tie_source": str(verdict["source"]),
        "orientation_normalised": bool(normalised),
        "paired_source_key": paired_key,
        "role": "FINAL_PREFERRED_LEADER_VS_FINAL_RUNNER_UP",
    }


# ---------------------------------------------------------------------------
# Leader change
# ---------------------------------------------------------------------------


def analyze_leader_change(stage1_result: Mapping[str, Any], refined_result: Mapping[str, Any], *,
                          material_core: float = MATERIAL_FRONTIER_CHANGE_CORE,
                          near_tie_k: float = NEAR_TIE_K,
                          paired_key: str = PAIRED_SOURCE_KEY) -> dict[str, Any]:
    """Stage-2 leader vs Stage-1 leader, compared on the Stage-2 paired worlds.

    A numerical reorder is not a meaningful leader change.  The change is MATERIAL
    only when it is BOTH statistically material
    (``abs(mean) > near_tie_k * se``) AND practically material
    (``abs(mean) > material_core``).
    """

    stage1_ranked = ranked_route_ids(stage1_result)
    stage2_ranked = ranked_route_ids(refined_result)
    stage1_leader = stage1_ranked[0] if stage1_ranked else None
    stage2_leader = stage2_ranked[0] if stage2_ranked else None
    stage1_signature = (
        None if stage1_leader is None
        else canonical_route_signature(_route_record(stage1_result, stage1_leader))
    )
    stage2_signature = (
        None if stage2_leader is None
        else canonical_route_signature(_route_record(refined_result, stage2_leader))
    )

    report: dict[str, Any] = {
        "stage1_leader_route_id": stage1_leader,
        "stage1_leader_signature": stage1_signature,
        "refined_leader_route_id": stage2_leader,
        "refined_leader_signature": stage2_signature,
        "changed": stage1_signature != stage2_signature,
        "statistically_material": False,
        "practically_material": False,
        "accepted": False,
        "direction": DIRECTION_UNCHANGED,
        "material_threshold_core": float(material_core),
        "near_tie_k": float(near_tie_k),
        "paired": None,
        "paired_available": False,
        "diagnostic": None,
    }
    if not report["changed"]:
        return report

    # The Stage-1 leader is always a finalist, so it is present in the refined
    # result; locate it by canonical signature (route ids are ephemeral).
    signature_to_id = _signature_to_route_id(refined_result)
    stage1_leader_in_refined = signature_to_id.get(str(stage1_signature))
    if stage1_leader_in_refined is None or stage2_leader is None:
        report["diagnostic"] = DIAG_ESCALATED_LEADER_UNPAIRED
        report["statistically_material"] = True
        report["accepted"] = True
        report["direction"] = DIRECTION_REFINED_LEADER_BETTER
        return report

    paired = paired_between(refined_result, stage2_leader, stage1_leader_in_refined, paired_key)
    if not paired:
        report["diagnostic"] = DIAG_ESCALATED_LEADER_UNPAIRED
        report["statistically_material"] = True
        report["accepted"] = True
        report["direction"] = DIRECTION_REFINED_LEADER_BETTER
        return report

    mean = float(paired["mean_difference"])
    se = float(paired.get("paired_se") or 0.0)
    # route_a is the refined leader in ``refined_result``; a positive mean means
    # the refined leader outscores the Stage-1 leader in the same worlds.
    statistically = abs(mean) > float(near_tie_k) * se
    practically = abs(mean) > float(material_core)
    report.update({
        "paired": {
            "route_a": stage2_leader,
            "route_b": stage1_leader_in_refined,
            "mean_difference": mean,
            "paired_se": se,
            "worlds": int(paired.get("worlds") or 0),
            "near_tied": bool(near_tie_verdict(paired, k=near_tie_k)["near_tied"]),
        },
        "paired_available": True,
        "statistically_material": bool(statistically),
        "practically_material": bool(practically),
        "accepted": bool(statistically and practically),
        "direction": (
            DIRECTION_REFINED_LEADER_BETTER if mean > 0 else DIRECTION_STAGE1_LEADER_BETTER
        ),
    })
    return report


def final_ranking_after_escalation(
    *,
    stage2_result: Mapping[str, Any],
    escalated_result: Mapping[str, Any] | None = None,
    preferred_route_id: str | None = None,
    paired_key: str = PAIRED_SOURCE_KEY,
) -> dict[str, Any]:
    """The FINAL ranking, and its canonical paired record, from the FINAL result.

    When the single escalation ran, the widened-search result IS the final
    ranking: §4 requires the numerically preferred route at the final supported
    evaluation budget to be the final preferred route, and §5 requires
    ``canonical_paired_near_tie.route_b`` to be the actual next-ranked route in
    that FINAL ranking — never the old Stage-2 finalist runner-up.

    ``preferred_route_id`` lets the caller pass the route the DECISION layer
    prefers, so ``route_a`` is the final preferred route even if the decision
    layer were to exclude the numerically first record.  This function re-ranks
    nothing and scores nothing: it only names the final leader/comparator and
    reads the paired CRN record that already exists for that pair.
    """

    escalated = escalated_result is not None
    result = escalated_result if escalated else stage2_result
    stage2_ranked = ranked_route_ids(stage2_result)
    final_ranked = ranked_route_ids(result)
    stage2_leader = stage2_ranked[0] if stage2_ranked else None
    final_rank1 = final_ranked[0] if final_ranked else None
    leader = preferred_route_id or final_rank1
    comparator = runner_up_for(result, leader)
    canonical = (
        None if leader is None or comparator is None
        else canonical_paired_record(result, leader_route_id=leader, runner_up_route_id=comparator,
                                     paired_key=paired_key)
    )
    return {
        "ranking_source": "ESCALATED_FINAL_RANKING" if escalated else "STAGE2_FINALIST_RANKING",
        "escalated": bool(escalated),
        "result": result,
        "stage2_leader_route_id": stage2_leader,
        "final_rank_1_route_id": final_rank1,
        "final_ranking": {"preferred_route_id": leader, "runner_up_route_id": comparator},
        "canonical_paired_near_tie": canonical,
        "canonical_alignment": (
            "DECISION_PREFERRED_IS_RANK_1" if leader == final_rank1
            else "ALIGNED_TO_DECISION_PREFERRED"
        ),
        "comparator_source": (
            "FINAL_WIDENED_RANKING" if escalated else "STAGE2_FINALIST_RANKING"
        ),
        "diagnostic": None if canonical is not None else DIAG_PAIRED_DIAGNOSTIC_UNAVAILABLE,
    }


# ---------------------------------------------------------------------------
# Search-stability gate (at most ONE bounded escalation)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StabilityGateConfig:
    """Search-breadth ladder configuration.

    ``budgets`` are BEAM WIDTHS (search breadth), never Monte Carlo draws; see
    ``route_stability.LADDER_BUDGET_SEMANTICS``.
    """

    budgets: tuple = rs.LADDER_BUDGETS
    current_beam: int = ro.OptimizerConfig.beam_width  # 8
    near_tie_k: float = NEAR_TIE_K
    material_core: float = MATERIAL_FRONTIER_CHANGE_CORE


def assess_search_stability(
    *,
    refined_result: Mapping[str, Any],
    leader_change: Mapping[str, Any],
    canonical_paired: Mapping[str, Any] | None,
    escalation: Callable[[int], Mapping[str, Any]] | None = None,
    escalation_beam_override: int | None = None,
    config: StabilityGateConfig | None = None,
    paired_key: str = PAIRED_SOURCE_KEY,
    escalated_result_sink: dict | None = None,
    cancel_probe: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Decide ``SEARCH_BUDGET_STABLE`` / ``SEARCH_NOT_STABLE_AT_CURRENT_BUDGET``.

    The gate performs AT MOST ONE escalation, and only when the refined result is
    unresolved or the leader moved statistically.  ``escalation(beam)`` must run
    the SAME football inputs at the next supported search budget and return that
    run's result; the caller is responsible for forcing the refined finalists in
    so the comparison is well defined.  A broader search never changes the
    objective, and nothing here re-ranks routes.

    ``escalated_result_sink`` is an optional out-parameter: when supplied, the gate
    sets ``sink["result"]`` to the escalated run's result (or ``None`` when no
    escalation ran).  It is the ONLY way the caller can obtain the widened result
    without putting a world matrix into the serialized stability report.  The key
    is always written, so a caller can never read a stale value.
    """

    if escalated_result_sink is not None:
        escalated_result_sink["result"] = None

    settings = config or StabilityGateConfig()
    ranked = ranked_route_ids(refined_result)
    refined_leader = ranked[0] if ranked else None
    refined_signature = (
        None if refined_leader is None
        else canonical_route_signature(_route_record(refined_result, refined_leader))
    )

    resolved = canonical_paired is not None and not bool(canonical_paired.get("near_tied"))
    practically_material = (
        canonical_paired is not None
        and abs(float(canonical_paired.get("mean_difference", 0.0))) > float(settings.material_core)
    )
    unresolved = not resolved
    # A statistically resolved but PRACTICALLY immaterial margin is still worth one
    # breadth check: stability must never rest on "one beam happened to return one
    # leader", and it must never be denied merely because the margin is small.
    immaterial_margin = bool(resolved and not practically_material)
    leader_moved = bool(leader_change.get("statistically_material"))
    needs_breadth_check = unresolved or immaterial_margin or leader_moved

    budget_sequence = [int(settings.current_beam)]
    report: dict[str, Any] = {
        "phase_version": PHASE_R4B2B_VERSION,
        "search_budget_semantics": "BEAM_WIDTH_SEARCH_BREADTH_NOT_MONTE_CARLO_DRAWS",
        "search_budget_sequence": budget_sequence,
        "escalation_used": False,
        "escalation_bounded_to_one": True,
        "escalation_trigger": None,
        "escalation_beam": None,
        "breadth_check_performed": False,
        "stability_basis": None,
        "leaders_agree": None,
        "escalated_leader_route_id": None,
        "escalated_leader_signature": None,
        "canonical_paired_present": canonical_paired is not None,
        "canonical_paired_near_tied": (None if canonical_paired is None
                                      else bool(canonical_paired.get("near_tied"))),
        "material_threshold_core": float(settings.material_core),
        "near_tie_k": float(settings.near_tie_k),
        "diagnostic": None,
        "state": None,
    }

    next_beam = escalation_beam_override
    if next_beam is None:
        next_beam = rs.next_ladder_budget(int(settings.current_beam), settings.budgets)

    can_escalate = escalation is not None and next_beam is not None and needs_breadth_check

    if not needs_breadth_check:
        # The refined paired comparison already separates the leader from its
        # runner-up BOTH statistically and practically, and the leader did not move.
        # That is stronger evidence than "one beam happened to return one leader",
        # and it is stated as such.
        report.update({
            "stability_basis": "DECISIVE_REFINED_PAIRED_WIN",
            "state": SEARCH_STABLE,
        })
        return report

    report["escalation_trigger"] = (
        "PAIRED_NEAR_TIE_OR_MISSING" if unresolved
        else ("IMMATERIAL_PAIRED_MARGIN" if immaterial_margin else "STATISTICALLY_MATERIAL_LEADER_CHANGE")
    )

    if not can_escalate:
        report.update({
            "stability_basis": "NO_LARGER_SUPPORTED_SEARCH_BUDGET",
            "state": SEARCH_NOT_STABLE,
            "diagnostic": DIAG_PAIRED_DIAGNOSTIC_UNAVAILABLE if canonical_paired is None else None,
        })
        return report

    # ---- the single permitted escalation ------------------------------------
    report["escalation_used"] = True
    report["escalation_beam"] = int(next_beam)
    budget_sequence.append(int(next_beam))
    if cancel_probe is not None:
        # Safe boundary: nothing is in flight immediately before the escalation.
        cancel_probe()
    try:
        escalated = escalation(int(next_beam))
    except Exception as error:  # fail closed: an unusable escalation is not stability
        report.update({
            "escalation_used": True,
            "breadth_check_performed": False,
            "stability_basis": "ESCALATION_FAILED",
            "state": SEARCH_NOT_STABLE,
            "diagnostic": DIAG_ESCALATION_FAILED,
            "escalation_error": f"{type(error).__name__}: {error}",
        })
        return report

    report["breadth_check_performed"] = True
    if escalated_result_sink is not None:
        escalated_result_sink["result"] = escalated
    escalated_ranked = ranked_route_ids(escalated)
    escalated_leader = escalated_ranked[0] if escalated_ranked else None
    escalated_signature = (
        None if escalated_leader is None
        else canonical_route_signature(_route_record(escalated, escalated_leader))
    )
    report["escalated_leader_route_id"] = escalated_leader
    report["escalated_leader_signature"] = escalated_signature
    leaders_agree = escalated_signature is not None and escalated_signature == refined_signature
    report["leaders_agree"] = bool(leaders_agree)

    if leaders_agree:
        report.update({"stability_basis": "ESCALATED_SEARCH_REPRODUCES_LEADER", "state": SEARCH_STABLE})
        return report

    # A different leader emerged at the wider budget: material only if the wider
    # search's leader beats the refined leader materially in the shared worlds.
    signature_to_id = _signature_to_route_id(escalated)
    refined_leader_in_escalated = signature_to_id.get(str(refined_signature))
    paired = (
        None if refined_leader_in_escalated is None
        else paired_between(escalated, escalated_leader, refined_leader_in_escalated, paired_key)
    )
    if not paired:
        report.update({
            "stability_basis": "ESCALATED_LEADER_UNPAIRED",
            "state": SEARCH_NOT_STABLE,
            "diagnostic": DIAG_ESCALATED_LEADER_UNPAIRED,
        })
        return report

    mean = float(paired["mean_difference"])
    se = float(paired.get("paired_se") or 0.0)
    statistically = abs(mean) > float(settings.near_tie_k) * se
    practically = abs(mean) > float(settings.material_core)
    report["escalated_paired"] = {
        "route_a": escalated_leader,
        "route_b": refined_leader_in_escalated,
        "mean_difference": mean,
        "paired_se": se,
        "worlds": int(paired.get("worlds") or 0),
    }
    report["escalated_leader_statistically_material"] = bool(statistically)
    report["escalated_leader_practically_material"] = bool(practically)

    if statistically and practically:
        report.update({"stability_basis": "ESCALATED_LEADER_MATERIALLY_DIFFERENT", "state": SEARCH_NOT_STABLE})
    else:
        # A numerical reorder at the wider budget, not a meaningful change.
        report.update({"stability_basis": "ESCALATED_REORDER_NOT_MATERIAL", "state": SEARCH_STABLE})
    return report


# ---------------------------------------------------------------------------
# World-prefix invariance
# ---------------------------------------------------------------------------


def compare_world_prefix(low: Mapping[str, Any], high: Mapping[str, Any],
                         *, surfaces: Sequence[str] = ("core", "minutes")) -> dict[str, Any]:
    """Prove the first ``low["worlds"]`` captured worlds of ``high`` are identical.

    Compares only the DECISION-RELEVANT CAPTURED SURFACES (player CORE and player
    minutes) for the player ids both matrices captured.  It deliberately does NOT
    claim that every hidden internal RNG stream is identical.
    """

    low_worlds = int(low.get("worlds") or 0)
    high_worlds = int(high.get("worlds") or 0)
    shared_ids = sorted({int(pid) for pid in low.get("player_ids") or []}
                        & {int(pid) for pid in high.get("player_ids") or []})
    worlds_compared = min(low_worlds, high_worlds)
    mismatches: list[dict[str, Any]] = []
    for surface in surfaces:
        low_series = low.get(surface) or {}
        high_series = high.get(surface) or {}
        for pid in shared_ids:
            a = list(low_series.get(pid) or [])
            b = list(high_series.get(pid) or [])
            if len(a) < worlds_compared or len(b) < worlds_compared:
                mismatches.append({"surface": surface, "player_id": pid, "reason": "SERIES_TOO_SHORT"})
                continue
            for index in range(worlds_compared):
                if a[index] != b[index]:
                    mismatches.append({
                        "surface": surface, "player_id": pid, "world_index": index,
                        "low": a[index], "high": b[index],
                    })
                    break
    return {
        "status": PREFIX_STATUS_PASS if not mismatches else PREFIX_STATUS_FAIL,
        "scope": PREFIX_INVARIANCE_SCOPE,
        "surfaces": list(surfaces),
        "worlds_compared": int(worlds_compared),
        "low_worlds": low_worlds,
        "high_worlds": high_worlds,
        "shared_captured_players": len(shared_ids),
        "mismatch_count": len(mismatches),
        "first_mismatches": mismatches[:5],
        "claim": "captured player CORE and player minutes only; NOT every hidden RNG stream",
    }


# ---------------------------------------------------------------------------
# Stage-2 refinement orchestrator
# ---------------------------------------------------------------------------


def finalist_partials(result: Mapping[str, Any], finalist_selection: Mapping[str, Any]) -> list[Any]:
    """Live ``PartialRoute`` objects for a finalist selection.

    ``optimize`` keeps these in-process only (``promoted_routes`` is a transient
    field).  A missing object is a hard error rather than a silent skip: the
    refinement must not quietly evaluate a different route set.
    """

    signatures = list(finalist_selection.get("finalist_route_ids") or [])
    by_signature = {
        ro.canonical_family_signature(partial): partial
        for partial in (result.get("promoted_routes") or [])
    }
    signature_by_id = _route_id_to_signature(result)
    wanted = []
    missing = []
    for route_id in signatures:
        signature = signature_by_id.get(str(route_id))
        partial = by_signature.get(str(signature))
        if partial is None:
            missing.append(str(route_id))
        else:
            wanted.append(partial)
    if missing:
        raise FinalistRefinementError(
            f"{DIAG_FINALIST_PARTIAL_UNAVAILABLE}: {len(missing)} finalist route(s) have no live "
            f"PartialRoute in the Stage-1 result (first: {missing[0]}); the refinement must be run "
            "in the same process as the Stage-1 optimizer call"
        )
    return wanted


class FinalistRefinementError(RuntimeError):
    """Raised when the refinement cannot be performed truthfully."""


def refine_finalists(
    *,
    universe: Mapping[str, Any],
    initial_state: Any,
    scenario: Any,
    player_meta: Mapping[int, Any],
    base_config: ro.OptimizerConfig,
    stage1_result: Mapping[str, Any],
    bundles: Mapping[int, Any] | None = None,
    conn=None,
    stage2_draws: int = STAGE2_DRAWS,
    finalist_selection: Mapping[str, Any] | None = None,
    non_production_worlds: Any | None = None,
    certification: Mapping[str, Any] | None = None,
    prebuilt_worlds: Mapping[int, Any] | None = None,
    cache_dir=None,
    verify_prefix: bool = True,
    optimizer: Callable[..., Mapping[str, Any]] | None = None,
    exact_cache: dict | None = None,
    cancel_probe: Callable[[], None] | None = None,
    parallel_workers: int | None = None,
) -> dict[str, Any]:
    """Re-evaluate ONLY the finalists at ``stage2_draws`` and rank them.

    Same seed, same certified bundles, same route legality, same discovery
    universe; the only changed input is the number of shared football worlds.

    Predictive data enters through exactly two declared doors, and the refinement
    crosses the same boundary as the optimizer it calls: a ``certification``
    artifact authorises the ``bundles``' exact certified run ids -- the loader
    compares each event's bundle identity, event and cutoff against the bundle the
    artifact recorded -- while worlds that were never loaded from a prediction run
    must be declared through ``route_optimizer.NonProductionWorlds``, which names
    who is exercising that interface.  A ``prebuilt_worlds`` mapping is the
    certified loader's own output and is checked against the artifact.

    ``exact_cache`` lets a caller share ONE exact-evaluation cache between this
    refinement and a later evaluation over the same certified worlds (the stability
    escalation).  Reuse is keyed on the complete evaluation identity — event,
    canonical squad identity, draw count, seed and world provenance — so a hit can
    only occur for a literally identical evaluation.  ``cancel_probe``, when given,
    is called at safe boundaries only (see ``optimize``/``run_search``).

    ``parallel_workers`` is SCHEDULING only and is forwarded verbatim to
    ``route_optimizer.optimize``; ``None`` keeps the library default (sequential).  The
    production runner opts in explicitly and auditable so that a caller reading the
    runner can see the choice; nothing here defaults it to a pool.
    """

    selection = dict(finalist_selection or select_finalists(stage1_result))
    finalists = selection.get("finalists") or []
    if not finalists:
        raise FinalistRefinementError(
            f"{DIAG_FINALIST_PARTIAL_UNAVAILABLE}: the Stage-1 result produced no finalists"
        )
    partials = finalist_partials(stage1_result, selection)

    if prebuilt_worlds is None:
        pool = ro.build_search_pool(universe, [int(p.player_id) for p in initial_state.players], base_config)
        forced: set[int] = set()
        for partial in partials:
            forced |= ro.route_player_ids(partial)
        union = ro.union_player_ids(initial_state, sorted(set(pool["pool_ids"]) | forced))
        world_config = ro.OptimizerConfig(
            events=base_config.events, search_draws=int(stage2_draws), seed=base_config.seed,
            beam_width=base_config.beam_width,
            exact_evaluation_budget=base_config.exact_evaluation_budget,
            policy_selection_worlds=base_config.policy_selection_worlds,
        )
        if non_production_worlds is None and certification is None:
            raise FinalistRefinementError(
                "refine_finalists needs a certification artifact, a declared non-production world "
                "source, or prebuilt_worlds"
            )
        prebuilt_worlds = {}
        for event in base_config.events:
            if cancel_probe is not None:
                # Safe boundary: the previous build_event_worlds call has fully returned
                # (its cache file is written and no SQLite transaction is open).
                cancel_probe()
            matrix, _info = ro.build_event_worlds(
                conn, bundles, int(event), union, world_config, cache_dir=cache_dir,
                non_production_worlds=non_production_worlds, certification=certification,
            )
            prebuilt_worlds[int(event)] = matrix

    prefix_report: dict[str, Any] | None = None
    if verify_prefix:
        prefix_report = _measure_prefix_invariance(
            universe=universe, initial_state=initial_state, bundles=bundles, conn=conn,
            base_config=base_config, stage2_draws=int(stage2_draws), partials=partials,
            prebuilt_worlds=prebuilt_worlds, non_production_worlds=non_production_worlds,
            certification=certification, cache_dir=cache_dir, cancel_probe=cancel_probe,
        )

    refined_config = ro.OptimizerConfig(
        events=base_config.events, search_draws=int(stage2_draws), seed=base_config.seed,
        beam_width=base_config.beam_width,
        # Only the finalists (plus ROLL) are exact-evaluated: no other route may
        # enter the final ranking.
        exact_evaluation_budget=0,
        policy_selection_worlds=base_config.policy_selection_worlds,
        max_auto_hit_points_per_event=base_config.max_auto_hit_points_per_event,
        singles_per_out=base_config.singles_per_out,
        max_transfers_per_event=base_config.max_transfers_per_event,
        rescue_top_k_per_position=base_config.rescue_top_k_per_position,
        search_n_per_criterion=base_config.search_n_per_criterion,
        retention_lenses=base_config.retention_lenses,
    )
    run = optimizer or ro.optimize
    if cancel_probe is not None:
        cancel_probe()
    refined = run(
        universe=universe, initial_state=initial_state, scenario=scenario, player_meta=player_meta,
        bundles=bundles, conn=conn, config=refined_config, cache_dir=None,
        non_production_worlds=non_production_worlds, certification=certification,
        prebuilt_worlds=prebuilt_worlds,
        required_routes=partials, nested_prior=None, exact_cache=exact_cache,
        cancel_probe=cancel_probe, parallel_workers=parallel_workers,
    )
    refined = dict(refined)

    ranked = ranked_route_ids(refined)
    leader = ranked[0] if ranked else None
    runner_up = ranked[1] if len(ranked) > 1 else None
    canonical = (
        None if leader is None or runner_up is None
        else canonical_paired_record(refined, leader_route_id=leader, runner_up_route_id=runner_up)
    )

    fidelity = {
        "stage1": {
            "draws_per_event": int(base_config.search_draws),
            "seed": int(base_config.seed),
            "beam_width": int(base_config.beam_width),
            "route_count": len((stage1_result.get("routes") or {})),
        },
        "finalists": {
            "route_ids": list(selection.get("finalist_route_ids") or []),
            "selection_reasons": {
                item["route_id"]: list(item["reasons"]) for item in finalists
            },
        },
        "stage2": {
            "draws_per_event": int(stage2_draws),
            "seed": int(base_config.seed),
            "beam_width": int(base_config.beam_width),
            "route_count": len((refined.get("routes") or {})),
            "evaluates_finalists_only": True,
            "prefix_invariance": (
                prefix_report if prefix_report is not None
                else {"status": "NOT_MEASURED", "scope": PREFIX_INVARIANCE_SCOPE}
            ),
        },
        "final_ranking": {"preferred_route_id": leader, "runner_up_route_id": runner_up},
        "canonical_paired_near_tie": canonical,
        "global_optimality_claimed": False,
    }
    if prefix_report is not None and prefix_report["status"] != PREFIX_STATUS_PASS:
        fidelity["stage2"]["prefix_invariance"]["diagnostic"] = DIAG_PREFIX_INVARIANCE_FAILED

    # Publish the canonical diagnostic ON the result itself, so the decision
    # runner consumes it through the documented key instead of inferring one.
    refined["canonical_paired_near_tie"] = canonical
    refined["simulation_fidelity"] = fidelity

    return {
        "phase_version": PHASE_R4B2B_VERSION,
        "finalist_selection": selection,
        "refined": refined,
        "final_ranking": {"preferred_route_id": leader, "runner_up_route_id": runner_up},
        "canonical_paired_near_tie": canonical,
        "simulation_fidelity": fidelity,
        "prefix_invariance": prefix_report,
        # In-process only: the exact shared worlds Stage 2 used, so the ONE
        # stability escalation can run in the SAME worlds (common random numbers).
        # Never serialized into an artifact.
        "prebuilt_worlds": prebuilt_worlds,
        "global_optimality_claimed": False,
    }


def _measure_prefix_invariance(*, universe, initial_state, bundles, conn, base_config,
                               stage2_draws, partials, prebuilt_worlds,
                               non_production_worlds, certification, cache_dir=None,
                               cancel_probe: Callable[[], None] | None = None) -> dict[str, Any]:
    """Re-run the prefix experiment against the live 10k matrices.

    A 2,000-draw matrix is generated for the SAME union and seed, and its captured
    CORE/minutes series must equal the first 2,000 columns of the higher-draw
    matrix.  This is the experiment the artifact's ``prefix_invariance`` claim
    rests on; it is re-run rather than assumed.
    """

    pool = ro.build_search_pool(universe, [int(p.player_id) for p in initial_state.players], base_config)
    forced: set[int] = set()
    for partial in partials:
        forced |= ro.route_player_ids(partial)
    union = ro.union_player_ids(initial_state, sorted(set(pool["pool_ids"]) | forced))
    low_config = ro.OptimizerConfig(
        events=base_config.events, search_draws=int(STAGE1_DRAWS), seed=base_config.seed,
        beam_width=base_config.beam_width, exact_evaluation_budget=0,
        policy_selection_worlds=base_config.policy_selection_worlds,
    )
    reports: dict[str, Any] = {}
    overall = PREFIX_STATUS_PASS
    for event in base_config.events:
        event = int(event)
        if cancel_probe is not None:
            cancel_probe()  # safe boundary: each iteration is one cache read + a pure compare
        # ``cache_dir`` is the Stage-1 world cache: the 2,000-draw matrix for the
        # SAME union and seed is already there, so the prefix check compares
        # Stage 2's worlds against the worlds Stage 1 actually scored.
        low, _info = ro.build_event_worlds(
            conn, bundles, event, union, low_config, cache_dir=cache_dir,
            non_production_worlds=non_production_worlds, certification=certification,
        )
        high = prebuilt_worlds[event]
        if int(low["worlds"]) >= int(stage2_draws):
            # A low-draw matrix must be the strictly smaller prefix.
            reports[str(event)] = {"status": PREFIX_STATUS_FAIL, "reason": "LOW_NOT_SMALLER"}
            overall = PREFIX_STATUS_FAIL
            continue
        report = compare_world_prefix(low, high)
        reports[str(event)] = report
        if report["status"] != PREFIX_STATUS_PASS:
            overall = PREFIX_STATUS_FAIL
    return {
        "status": overall,
        "scope": PREFIX_INVARIANCE_SCOPE,
        "low_draws": int(STAGE1_DRAWS),
        "high_draws": int(stage2_draws),
        "per_event": reports,
        "claim": "captured player CORE and player minutes only; NOT every hidden RNG stream",
    }


__all__ = [
    "DECISION_OBJECTIVE_KEY",
    "DIAG_ESCALATED_LEADER_UNPAIRED",
    "DIAG_ESCALATION_FAILED",
    "DIAG_FINALIST_PARTIAL_UNAVAILABLE",
    "DIAG_PAIRED_DIAGNOSTIC_UNAVAILABLE",
    "DIAG_PREFIX_INVARIANCE_FAILED",
    "DIRECTION_REFINED_LEADER_BETTER",
    "DIRECTION_STAGE1_LEADER_BETTER",
    "DIRECTION_UNCHANGED",
    "FinalistRefinementError",
    "MATERIAL_FRONTIER_CHANGE_CORE",
    "NEAR_TIE_K",
    "PAIRED_SOURCE_KEY",
    "PHASE_R4B2B_VERSION",
    "PREFIX_INVARIANCE_SCOPE",
    "PREFIX_STATUS_FAIL",
    "PREFIX_STATUS_PASS",
    "REASON_PAIRED_NEAR_TIE",
    "REASON_ROLL_BASELINE",
    "REASON_STAGE1_LEADER",
    "SEARCH_NOT_STABLE",
    "SEARCH_STABLE",
    "STAGE1_DRAWS",
    "STAGE2_DRAWS",
    "StabilityGateConfig",
    "analyze_leader_change",
    "assess_search_stability",
    "canonical_paired_record",
    "canonical_route_signature",
    "compare_world_prefix",
    "decision_rank_key",
    "final_ranking_after_escalation",
    "finalist_partials",
    "near_tie_verdict",
    "paired_between",
    "paired_route_index",
    "ranked_route_ids",
    "refine_finalists",
    "runner_up_for",
    "select_finalists",
]
