# Prediction Engine V1 roadmap — post-PE-8 authority

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

**STATUS: FROZEN**

reviewed source: `491b8ae58c3b570ab60d783cf4ec21a2b987d821`

merge: `95b6f71afbe6996b975f7815de854449edf892ff`

calibration outcome: **NO PROMOTION** — no model version was bumped and no calibration
version was replaced; the merged evidence artifact declares `promotion: NOT PERFORMED`
and `production_wiring: NONE`, so no calibration transform is wired into a production
decision path.

The PE-8 contract is defined in
[PE-8-CALIBRATION.md](PE-8-CALIBRATION.md).

### PE-9 — Certification Integration

**STATUS: CLOSED — ACCEPTED FOR THIS CANDIDATE**

Bernard's dated decision accepts the PE-9 closure and supersedes this roadmap's
historical NEXT label for this candidate. The decision is retained separately
with its capture time; see the approved final packet SHA-256
120377EB09246809C7D462CC70B5F52C43EEAD365DE003220482F829E90DFD83 and the
decision record SHA-256 974B4D151ABF468D13DF52E23F7E8B868065C99EAFB4E9E896F504F4CE59408A.
The recorded code-closure point is d27de7164900ef7900828bb7f1ac12710200b08b;
subsequent retention and publication corrections remain in their original history.
This proposal preserves earlier roadmap snapshots and authorization deviations.

The PE-9 contract is defined in
[PE-9-CERTIFICATION-INTEGRATION.md](PE-9-CERTIFICATION-INTEGRATION.md).

### PE-10 — End-to-End Acceptance / Prediction Engine V1 Freeze

**STATUS: MERGED — CANDIDATE ACCEPTED; FREEZE PENDING SEPARATE APPROVAL**

Bernard's dated decision accepts candidate 226ac4544bfa3889840da5d77801a68b42914f5f
(tree d31da0a127b16993ad49fe6e9c9aa5ee82911b7f) and supersedes the historical
BLOCKED on PE-9 label for this candidate. The candidate was promoted by normal
fast-forward from ca251f0aba18ae5287a3885fe886c08569f0694d; Git and GitHub ref
readbacks confirmed main at the candidate SHA and the exact tree.

Evidence: exact-SHA CI run 36604095548 and unit job 109528571747 succeeded on
this SHA, including the authoritative wrapper. The retained raw pytest result has
two accepted failures (test_13_manager_state_prose_derives_from_actual_state and
test_13b_zero_ft_prose_is_also_derived) and no unexpected failures. Sol High's
follow-up review approves acceptance of this exact SHA and tree; see the final
packet and retained review evidence.

This is a merge and candidate-acceptance record only. The operational runtime
concern remains unaccepted; PE-10 freeze and deployment remain unapproved, and
PE-11 remains unapproved. The timing-only exception remains limited to the scope
already recorded in PE-10 section 16. The earlier Sol stop and authorization
deviations remain historical evidence and are not retroactively authorized.

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

These accepted contracts remain unchanged by the post-PE-8 authority rollover.
PE-4's frozen aggregation contract is defined in
[PE-4-DGW-BLANK-WORLD-AGGREGATION.md](PE-4-DGW-BLANK-WORLD-AGGREGATION.md).
