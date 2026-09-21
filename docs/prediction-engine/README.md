# Prediction Engine V1 roadmap — post-PE-4 authority

This document is product and engineering authority for the Prediction Engine V1
phase sequence. Each phase is finite and is completed only at its stated review
boundary.

## Roadmap

### PE-1 — Data hygiene / causal historical boundary

**STATUS: FROZEN**

### PE-2 — Walk-forward evaluation

**STATUS: FROZEN**

### MC zero-variance repair

**STATUS: FROZEN**

### PE-3 — Structural Bonus / BPS

**STATUS: FROZEN**

### PE-4 — DGW / Blank World Aggregation

**STATUS: FROZEN**

reviewed source: `c75efb5b3359f154c786f171ad6e4780ea9c5c15`

merge: `4849299d1653e75a62fbdf2aa8ade4493c181195`

### PE-5 — Outcome Capture / Prediction-to-Reality Ledger

**STATUS: FROZEN**

reviewed source: `8e039e96235cc322a1f661ac213b1a6d9cdb0c9b`

merge: `63d9cb0f8756cf1d47445ff2c71d31365488647f`

The PE-5 contract is defined in
[PE-5-OUTCOME-CAPTURE-PREDICTION-TO-REALITY-LEDGER.md](PE-5-OUTCOME-CAPTURE-PREDICTION-TO-REALITY-LEDGER.md).

### PE-6 — Availability / Minutes Refinement

**STATUS: NEXT**

The PE-6 contract is defined in
[PE-6-AVAILABILITY-MINUTES-REFINEMENT.md](PE-6-AVAILABILITY-MINUTES-REFINEMENT.md).
The contract is present; PE-6 implementation has not started.

### PE-7 — Team Attack/Defence + Player Attack Refinement

**STATUS: BLOCKED on PE-6**

### PE-8 — Calibration

**STATUS: BLOCKED on PE-7**

### PE-9 — Certification Integration

**STATUS: BLOCKED on PE-8**

### PE-10 — End-to-End Acceptance / Prediction Engine V1 Freeze

**STATUS: BLOCKED on PE-9**

## Phase policy

- each phase is finite
- one phase at a time
- later phase begins only after the previous phase is reviewed, human-approved, merged and frozen
- an accepted phase is not reopened without concrete regression
- no optional hardening after `READY_FOR_MERGE`
- the phase merge target is `feature/prediction-engine-v1`
- merge is human authorized
- the orchestrator never auto-merges

## Normal transfer contract

- current Gameweek plus the next 3 exactly
- no H1-only transfer recommendation
- incomplete fresh coherent horizon -> `DECISION_HORIZON_INCOMPLETE`
- lineup/captain are H1 only
- exhaustive official player discovery
- no silent zero for missing projections

## Carry-forward items

### PE-5

- append-only / point-in-time outcome history
- generation-certified prediction-freeze provenance

### PE-8 / PE-9

- `CONTINUOUS_PROXY_TIE_LIMITATION` from PE-3

These accepted contracts are unchanged by the post-PE-4 readiness repairs.
PE-4's frozen aggregation contract is defined in
[PE-4-DGW-BLANK-WORLD-AGGREGATION.md](PE-4-DGW-BLANK-WORLD-AGGREGATION.md).
