# Prediction Engine V1 — PE-5 Outcome Capture / Prediction-to-Reality Ledger

**STATUS: NEXT — CONTRACT PRESENT, IMPLEMENTATION NOT STARTED**

## Purpose

PE-5 defines the append-only ledger that relates a frozen prediction generation
to the later official football outcome. It creates an auditable prediction-to-
reality boundary for future evaluation and calibration work.

PE-5 does not change the PE-4 event-world equations, minutes model, scoring
rules, RNG, manager state, transfer policy, or certification thresholds.

## Existing storage — audit and reuse before creating anything

The repository ALREADY contains outcome and provenance storage. PE-5 must audit
these and reuse or extend them where appropriate:

- `player_gameweeks` — current official player/fixture performance rows
- `outcome_observations` — outcome observation storage
- `bootstrap_generations` — accepted official-player generation records
- `projection_runs` — frozen predictive run records with official provenance
- `fpl_brain/historical_observations.py` — the PE-1 causal historical boundary

**Do not blindly create a duplicate parallel outcome or provenance system.**

`player_gameweeks` is CURRENT / LATEST-STATE storage. It is insufficient by
itself for durable point-in-time history, because the same logical
player/event/fixture row may later be refreshed in place, silently replacing
what an earlier read observed. PE-5 exists to close exactly that gap.

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

## Generation-certified prediction freezes

*This is a core PE-5 carry-forward requirement.*

Every NEW predictive freeze that claims official-player-pool provenance must be
able to prove the exact accepted bootstrap generation that supplied that state.

The repository already has `bootstrap_generations`, carrying at least:

- `accepted`
- `fetch_run_id`
- `captured_at`
- `official_element_count`
- `element_ids_sha256`
- `element_ids_json`
- `acceptance_rule_version`

and `projection_runs` carry official fetch/run provenance.

For a NEW freeze to be **generation-certified** it must prove all of:

1. the referenced official fetch exists;
2. the referenced bootstrap generation exists;
3. the bootstrap generation has `accepted = 1`;
4. the bootstrap generation's `fetch_run_id` matches the freeze's recorded
   official fetch identity;
5. the bootstrap generation was observable at or before the prediction cutoff;
6. the official player element-set identity/digest is retained.

Missing, malformed or disagreeing provenance is **NOT CERTIFIABLE**.

**Never substitute the newest accepted generation merely because it is newest.**
The generation that certified a freeze is the one recorded on that freeze; a
later accepted generation does not retroactively certify an earlier freeze.

## Legacy provenance policy

Do NOT rewrite historical projection runs to manufacture provenance.

Pre-PE-5 runs may remain explicitly `LEGACY_PROVENANCE`, or an equivalent
truthful not-fully-generation-certified state.

Do not fabricate `bootstrap_generations` records for old runs.

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

PE-5 must provide durable **append-only point-in-time observation history**:

- two captures of the same player/event/fixture at different observation times
  must BOTH be preservable;
- a later official refresh must never erase earlier evidence;
- idempotent ingest of the SAME observation may no-op;
- accepted historical facts must reject ordinary `UPDATE` and `DELETE`.

## No fake backfill

Do NOT take current `player_gameweeks` state and pretend it existed at an
earlier cutoff.

Historical backfill is permitted only when an actual archived source proves:

- exact source/payload identity;
- real capture/fetch identity;
- actual capture/observable time.

Otherwise historical point-in-time evidence is **unavailable**.

**Missing history is not zero.**

## Canonical causal semantics

Reuse the frozen PE-1 historical observation boundary. Do not create a second
placeholder predicate.

- scheduled placeholder != zero-minute DNP
- a genuine completed official DNP with zero minutes/points is a REAL
  observation

Reuse the canonical event/fixture finality semantics. Partial or provisional
events must not silently enter final evaluation.

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

## Outcome grains

Two distinct grains exist and must not be conflated:

**PLAYER x EVENT**

- the headline total FPL points for the event

**PLAYER x FIXTURE**

- fixture-natural outcomes: minutes, starts, goals, assists, clean sheets, goals
  conceded, saves, defensive contribution, bonus and BPS

Rules:

- do not duplicate an event-grain value once per DGW fixture;
- do not collapse fixture-grain evidence when fixture diagnostics require it.

## Outcome fields

Where supplied by the official source, PE-5 must preserve and expose at least:

```text
minutes
starts
total_points
goals_scored
assists
clean_sheets
goals_conceded
saves
bonus
bps
yellow_cards
red_cards
penalties_saved
penalties_missed
own_goals
defensive_contribution
```

Do not fabricate unsupported fields.

## Captured result surface

The outcome side must preserve the official facts needed to evaluate the frozen
prediction surface, including the finalized event/fixture status, player
appearance/minutes, and the applicable official scoring inputs. The ledger may
carry additional source fields, but it must not rewrite them into a modelled
expectation.

Prediction values remain the frozen values that were actually issued. PE-5 must
not regenerate predictions while capturing reality, and it must not backfill a
new model value into an old freeze.

## The prediction-to-reality ledger

The deterministic ledger must be able to answer, for every row:

- the exact prediction run / freeze;
- the prediction cutoff;
- the exact accepted official-player generation;
- the player/event or player/fixture key;
- the frozen predicted value;
- the final official outcome;
- the observation time;
- the source / provenance;
- the evaluation / exclusion state;
- the explicit exclusion reason.

Reuse existing walk-forward terminology where canonical.

State discipline:

- missing prediction != zero
- missing outcome != zero
- placeholder != zero
- blank Gameweek != model failure
- real official zero == real zero

## No leakage into causal reads

Later outcomes are **evaluation evidence only**.

Appending future outcome evidence must not alter a predictive read that claims
an earlier cutoff.

Do not weaken `historical_observations.py`.

## Migration and live data

Schema work must be additive. No destructive migration.

Implementation and tests use temporary databases.

Do NOT migrate or mutate the authoritative live FPL database during PE-5
implementation or review. Any live migration requires separate operational
authorization after acceptance.

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

The future PE-5 implementation must test, at minimum, all of the following HARD
CONTRACTS. This list is the authoritative minimum.

### Point-in-time integrity

1. two captures of the same player/event/fixture are both retained
2. a later capture cannot overwrite an earlier observation
3. identical ingest is idempotent
4. scheduled placeholder is excluded
5. a genuine completed DNP zero is retained
6. a genuine zero bonus/BPS is retained
7. a missing bonus remains missing
8. a partial event cannot enter final evaluation
9. a final event can enter evaluation

### Grain

10. DGW fixture outcomes remain distinct
11. headline event points aggregate once
12. a blank is explicit rather than missing

### Causality

13. an observation after the cutoff is excluded from an earlier as-of read
14. an observation before kickoff cannot become historical evidence
15. a later outcome append cannot change an earlier causal read

### Generation certification

16. an accepted bootstrap generation certifies a new freeze
17. a rejected generation cannot certify
18. a missing generation cannot certify
19. a generation/fetch mismatch fails
20. a generation after the prediction cutoff fails
21. the newest generation cannot replace the recorded generation
22. a legacy run remains legacy
23. the official element digest is retained and deterministic

### Ledger

24. the prediction/outcome population is deterministic
25. a missing prediction is explicit
26. a missing outcome is explicit
27. a placeholder is explicit
28. a genuine zero is evaluated
29. accepted history rejects `UPDATE`
30. accepted history rejects `DELETE`
31. a repeated ledger build is deterministic
32. database row order does not change the ledger or any digest

### Contract themes (narrative, retained)

The original summary of what these tests are for:

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

## Out of scope

PE-5 does NOT tune:

- minutes
- availability
- team strength
- player attacking rates
- the bonus/BPS proxy
- calibration

It does not alter transfer or chip strategy.

It does not start PE-6.

## Terminal boundary

PE-5 is the next phase only. This document records authority for its future
implementation; it does not authorize implementation, merge, calibration, or
certification work.
