# Prediction Engine V1 — PE-5 Outcome Capture / Prediction-to-Reality Ledger

**STATUS: NEXT — CONTRACT PRESENT, IMPLEMENTATION NOT STARTED**

## Purpose

PE-5 defines the append-only ledger that relates a frozen prediction generation
to the later official football outcome. It creates an auditable prediction-to-
reality boundary for future evaluation and calibration work.

PE-5 does not change the PE-4 event-world equations, minutes model, scoring
rules, RNG, manager state, transfer policy, or certification thresholds.

## Authority and temporal boundary

Every prediction-side record must identify the exact prediction freeze it came
from. At minimum, its provenance must preserve:

- planning event and planning cutoff;
- prediction generation identity;
- source snapshot identity and data cutoff;
- projection run identities;
- model version, configuration hash, and random seed;
- immutable prediction artifact identity and content digest.

An outcome may be captured only after its official source and event-finalization
state are recorded. A provisional observation may be retained as provisional,
but it must not be treated as the final truth for evaluation.

## Append-only outcome history

Outcome history is point-in-time and append-only:

- previously captured observations are never updated in place;
- a correction is a new observation with its own capture time and source
  provenance;
- supersession is explicit and auditable;
- deletion of historical observations is not a correction mechanism;
- every observation remains attributable to the source response or certified
  source artifact from which it was read.

The ledger must distinguish the time the football event occurred, the time the
official outcome became final, and the time the repository captured it. These
times must not be collapsed into one timestamp.

## Join identity

Prediction and outcome records join through stable football identity, not player
names or display text. The contract requires the identity needed to distinguish:

- event/Gameweek;
- fixture;
- club/team;
- player/element;
- the prediction generation and the outcome observation.

Missing, blank, zero-minute, postponed, and not-yet-final outcomes must remain
explicit states. A missing outcome must never be silently converted to zero.

## Captured result surface

The outcome side must preserve the official facts needed to evaluate the frozen
prediction surface, including the finalized event/fixture status, player
appearance/minutes, and the applicable official scoring inputs. The ledger may
carry additional source fields, but it must not rewrite them into a modelled
expectation.

Prediction values remain the frozen values that were actually issued. PE-5 must
not regenerate predictions while capturing reality, and it must not backfill a
new model value into an old freeze.

## Evaluation boundary

PE-5 records facts and provenance. It does not, by itself:

- recalibrate a model;
- promote a certification;
- alter a prediction artifact;
- rewrite historical data;
- change the accepted PE-4 contract;
- implement PE-6 availability/minutes refinement;
- implement PE-8 calibration or PE-9 certification integration.

Any later score, calibration, or certification consumer must declare which
prediction freeze, finalized outcome observation, and supersession policy it
used.

## Required contract tests before implementation review

The future PE-5 implementation must test, at minimum:

1. prediction-freeze provenance is preserved exactly;
2. finalized and provisional outcomes are distinguishable;
3. outcome capture is append-only;
4. corrections create explicit superseding observations;
5. event, fixture, club, and player joins are stable and name-independent;
6. blank, zero-minute, postponed, and missing outcomes are not conflated;
7. prediction capture never regenerates or mutates the frozen prediction;
8. point-in-time capture metadata is retained;
9. source provenance and content identity are auditable;
10. no PE-4 semantic, model, RNG, manager-state, or certification regression is
    introduced.

## Terminal boundary

PE-5 is the next phase only. This document records authority for its future
implementation; it does not authorize implementation, merge, calibration, or
certification work.
