# Prediction Engine V1 roadmap — post-PE-7 authority

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

**STATUS: FROZEN**

reviewed source: `688f9c43b0e670abfc3ba6728519469bf1145b21`

merge: `37e7dfa8bf3bbe06771d54c7446474b4b8f1fa6d`

human outcome: **NO CHANGE** — incumbent minutes models remain authoritative;
the challenger was not promoted because available evidence was insufficient for
model selection.

The PE-6 contract is defined in
[PE-6-AVAILABILITY-MINUTES-REFINEMENT.md](PE-6-AVAILABILITY-MINUTES-REFINEMENT.md).

### PE-7 — Team Attack/Defence + Player Attack Refinement

**STATUS: FROZEN**

reviewed source: `79f75a23cad3004a5029cca13e45ea653e000fd7`

merge: `8e5a9d6c6bf0bc9c26e60043d49f0fa57c65db9d`

human outcome: **NO PROMOTION** — the incumbent team attack/defence and player
attack models remain authoritative; the PE-7 challengers were not promoted.

The PE-7 contract is defined in
[PE-7-TEAM-PLAYER-ATTACK-REFINEMENT.md](PE-7-TEAM-PLAYER-ATTACK-REFINEMENT.md).

### PE-8 — Calibration

**STATUS: NEXT**

The PE-8 contract is defined in
[PE-8-CALIBRATION.md](PE-8-CALIBRATION.md).

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

### PE-8 / PE-9

- `CONTINUOUS_PROXY_TIE_LIMITATION` from PE-3

These accepted contracts remain unchanged by the post-PE-7 authority rollover.
PE-4's frozen aggregation contract is defined in
[PE-4-DGW-BLANK-WORLD-AGGREGATION.md](PE-4-DGW-BLANK-WORLD-AGGREGATION.md).
