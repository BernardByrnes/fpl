# Prediction Engine V1 — PE-6 Availability / Minutes Refinement

**STATUS: NEXT — CONTRACT PRESENT, IMPLEMENTATION NOT STARTED**

## Purpose

PE-6 is a finite challenger/refinement phase. It improves the prediction of
whether and how long a player plays, using the evidence infrastructure already
frozen in PE-1, PE-2 and PE-5.

PE-6 concerns availability and minutes only. It does not tune attacking rates,
team strength, bonus/BPS, transfers, chips, or calibration.

## In scope

- availability;
- `P(start | available)`;
- cameo probability;
- expected minutes;
- `P(60+)`;
- `P(80+)`;
- return-from-injury and ramp behaviour;
- rotation and role uncertainty.

## Out of scope

- attacking rates and expected goals/assists;
- team attack/defence and player attack refinement (PE-7);
- bonus / BPS structure;
- transfer policy, chip policy and decision horizons;
- calibration (PE-8);
- certification thresholds (PE-9);
- Monte Carlo RNG and draw ordering.

## Frozen incumbent

The incumbent remains authoritative throughout PE-6 and must not be silently
mutated:

- `fpl_brain/minutes_model.py` — `MINUTES_MODEL_VERSION = "minutes_v1.8.0"`
- `fpl_brain/minutes_model.py` — `MINUTES_COHERENT_MODEL_VERSION = "minutes_v1.2.0"`
- `fpl_brain/joint_minutes.py` — `JOINT_MINUTES_MODEL_VERSION = "minutes_v1.5.2"`

These version identifiers are the incumbent's identity. PE-6 may not overwrite,
rename, or re-point them by mere experimentation. Promotion, if it happens at
all, occurs only at PE-6's accepted terminal boundary and is an explicit,
reviewed act.

## Evidence boundary

PE-6 MUST reuse the frozen evidence infrastructure:

- PE-1 `fpl_brain/historical_observations.py` — the causal historical boundary;
- PE-2 walk-forward population and evaluation semantics, including the
  sample-size policy;
- PE-5 finalized outcome ledger and provenance where appropriate.

The following are contract requirements, not preferences:

- no second causal-history predicate may be introduced;
- no current-state leakage into historical cutoffs;
- missing evidence is not zero;
- a scheduled placeholder is not a DNP;
- a genuine completed zero-minute DNP remains real evidence.

## Assumptions PE-6 must scrutinize

The incumbent explicitly labels several availability values as **modelling
assumptions rather than calibrated probabilities**. PE-6 evaluates them; it does
not blindly preserve them and does not blindly replace them.

At minimum these are in scope for scrutiny:

- the official-status availability defaults
  (`OFFICIAL_STATUS_AVAILABILITY_DEFAULTS`, `minutes_model.py`), documented in
  code as "INITIAL MODELLING ASSUMPTIONS for the official availability signal"
  and surfaced as "initial modelling assumption, not calibrated";
- bounded scouting modifiers;
- bounded return-from-injury modifiers;
- bounded rotation-risk modifiers.

No arbitrary tuning merely because a number "looks better". Every proposed
change must be justified by the evaluation contract below, and a change that
cannot be distinguished from noise on an adequate sample is not an improvement.

## Challenger-first rule

Proposed refinements are built and evaluated as a **challenger**. The incumbent
remains untouched until evidence justifies promotion.

A legitimate PE-6 result is exactly one of:

- an accepted challenger;
- a partially accepted refinement;
- **NO CHANGE**, because the evidence is insufficient or the challenger fails.

Do not force an improvement. "No change" is a successful PE-6 outcome when the
evidence does not support a change.

## Evaluation contract

Incumbent and challenger are compared on **identical walk-forward
populations**. At minimum, report:

- Brier score for `P(start)`;
- Brier score for `P(60+)`;
- expected-minutes MAE;
- expected-minutes bias;
- start-rate prediction vs realised;
- 60+ prediction vs realised;
- useful position strata;
- availability/status strata where the sample allows;
- returning-player / rotation-risk strata where the sample allows.

Report sample sizes prominently. Respect the existing PE-2 sample-size policy.
Insufficient evidence must be labelled as insufficient rather than converted
into a model-selection claim.

## Causality

Every feature the challenger uses must have been observable at the prediction
cutoff. PE-6 must explicitly test that:

- post-cutoff injury or news cannot affect earlier projections;
- later outcomes cannot alter earlier prediction reads;
- future lineup or start evidence cannot leak backward;
- the same inputs and cutoff produce deterministic outputs.

## Coherence

Preserve mathematical coherence across:

`P(available)` → `P(start | available)` → `P(cameo | not start, available)` →
`P(0)` → `P(1-59)` → `P(60+)` → `P(80+)` → `expected_minutes`

Requirements:

- every probability remains in `[0, 1]`;
- no impossible player/minutes distributions are produced;
- existing team-level coherence requirements are preserved unless a failing
  test proves a concrete defect.

## Versioning

The challenger gets its own explicit model identity and configuration hash. Do
not bump or overwrite the incumbent primary version merely by experimenting.
Promotion, if justified, occurs only at PE-6's accepted terminal boundary.

## RNG

Minutes refinement remains deterministic unless there is an explicit, approved
reason otherwise. Do not change the Monte Carlo RNG or its draw ordering.

## Required hard tests

PE-6 includes at minimum:

1. hard unavailable status stays unavailable;
2. available-player probability remains bounded;
3. injury/doubt evidence is cutoff-safe;
4. a post-cutoff status update is ignored;
5. a genuine DNP is retained;
6. a scheduled placeholder is excluded;
7. repeated starts strengthen start evidence appropriately;
8. repeated available zero-minute non-starts affect role evidence appropriately;
9. return-from-injury behaviour;
10. cameo behaviour;
11. 60+/80+ coherence;
12. expected-minutes coherence;
13. no-history fallback remains structurally possible;
14. position/team coherence preserved;
15. deterministic repeat;
16. same-population incumbent/challenger evaluation;
17. a later PE-5 outcome append cannot change an earlier prediction;
18. no attacking, team-strength or bonus model changes.

## Terminal boundary

PE-6 ends at senior review. Its final state is exactly one of:

- `READY_FOR_MERGE`
- `OPEN`

No PE-7 work. No optional hardening after `READY_FOR_MERGE`.
