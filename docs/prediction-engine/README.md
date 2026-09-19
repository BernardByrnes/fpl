# Prediction Engine V1 roadmap

This document is product and engineering authority for the Prediction Engine V1 phase sequence. Each phase is finite and is completed only at its stated review boundary.

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

**STATUS: NEXT**

### PE-5 — Outcome Capture / Prediction-to-Reality Ledger

**STATUS: BLOCKED on PE-4 merge**

### PE-6 — Availability / Minutes Refinement

**STATUS: BLOCKED on PE-5**

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
- later phase begins only after previous phase is reviewed, human-approved, merged and frozen
- accepted phase is not reopened without concrete regression
- no optional hardening after `READY_FOR_MERGE`
- phase merge target is `feature/prediction-engine-v1`
- merge is human authorized
- orchestrator never auto-merges

## Normal transfer contract

- current GW + next 3 exactly
- no H1-only transfer recommendation
- incomplete fresh coherent horizon -> `DECISION_HORIZON_INCOMPLETE`
- lineup/captain H1 only
- exhaustive official player discovery
- no silent zero for missing projections

## Carry-forward items

### PE-5

- append-only / PIT outcome history
- generation-certified prediction-freeze provenance

### PE-8 / PE-9

- `CONTINUOUS_PROXY_TIE_LIMITATION` from PE-3

These accepted contracts are unchanged by this bootstrap. The PE-4 contract is defined in [PE-4-DGW-BLANK-WORLD-AGGREGATION.md](PE-4-DGW-BLANK-WORLD-AGGREGATION.md).
