# Prediction Engine V1 — PE-7 Team Attack/Defence + Player Attack Refinement

**STATUS: NEXT — CONTRACT PRESENT, IMPLEMENTATION NOT STARTED**

## Purpose

PE-7 is a finite challenger/refinement phase. It improves the football-event
expectation layer that sits after PE-6: how many goals a team is expected to
score and concede in a given fixture, and how a player's attacking contribution
is expected to arise from that environment.

PE-7 is not an automatic rewrite. The incumbents remain authoritative until
evidence justifies promotion, and `NO CHANGE` is a legitimate PE-7 outcome.

PE-7 does not tune minutes, calibration, certification, transfer policy or chips.

## In scope

- team attack and defence parameter estimation;
- league scoring level and home advantage;
- recency weighting;
- shrinkage and prior strength;
- sparse-team handling;
- expected-goals fixture environment;
- player xG/90 and xA/90 attacking rates;
- historical-rate prior strength;
- role-change evidence;
- low-history and newly established players;
- coherent allocation of attacking opportunity between team and players;
- explicit residual attacking mass where player-level estimates do not exhaust
  the team's expectation;
- the evaluation infrastructure required to compare incumbent and challenger.

## Out of scope

- PE-6 availability/minutes promotion;
- Bonus/BPS restructuring;
- DefCon redesign;
- scoring-rule changes;
- FPL assist calibration;
- PE-8 probability / xPts calibration;
- PE-9 certification thresholds;
- transfers, captaincy or chips;
- Monte Carlo RNG or draw-order changes;
- PE-8, PE-9 or PE-10 implementation.

**Penalty and non-penalty xG must not be fabricated as separate components**
unless the stored evidence can genuinely support that separation. The current
truthfulness around embedded penalty xG is preserved.

## Frozen incumbent identities

Verified in the merged source at the PE-7 base. These are the incumbents PE-7
must not silently rewrite:

| Constant | Value | Location |
|---|---|---|
| `TEAM_MODEL_VERSION` | `team_strength_v1.1.0` | `fpl_brain/team_model.py` |
| `TEAM_BASELINE_MODEL_VERSION` | `team_naive_v1.1.0` | `fpl_brain/team_model.py` |
| `PLAYER_RATE_MODEL_VERSION` | `player_rates_v1.0.0` | `fpl_brain/player_rates.py` |
| `PLAYER_RATE_BASELINE_MODEL_VERSION` | `player_rate_baseline_v1.0.0` | `fpl_brain/player_rates.py` |

Downstream identities that PE-7 must not silently rewrite:

| Constant | Value | Location |
|---|---|---|
| `XPTS_MODEL_VERSION` | `xpts_v1.4.1` | `fpl_brain/xpts.py` |
| `MONTE_CARLO_MODEL_VERSION` | `mc_v1.3.0` | `fpl_brain/monte_carlo.py` |

The PE-6 incumbent minutes trio remains frozen and is **outside PE-7's
promotion target**:

| Constant | Value | Location |
|---|---|---|
| `MINUTES_MODEL_VERSION` | `minutes_v1.8.0` | `fpl_brain/minutes_model.py` |
| `MINUTES_COHERENT_MODEL_VERSION` | `minutes_v1.2.0` | `fpl_brain/minutes_model.py` |
| `JOINT_MINUTES_MODEL_VERSION` | `minutes_v1.5.2` | `fpl_brain/joint_minutes.py` |

No incumbent version may be bumped merely to run an experiment.

## Evidence / causality boundary

PE-7 reuses the canonical PE-1 point-in-time evidence boundary. It must not
introduce a second historical-observation predicate.

PE-7 must assume that mutable current tables can leak history until proven
otherwise, and must explicitly audit every use of:

- current `players.team_id`;
- current `players.element_type`;
- current `players.is_active`;
- current club identity;
- current role/scouting evidence;
- player/team joins used for historical priors.

A post-cutoff transfer, position change, activation/deactivation, scouting
update or data write must not change an earlier prediction.

Preserved truthfulness:

- missing xG/xA evidence is missing, never zero;
- scheduled placeholders are not played matches;
- later realised outcomes are evaluation evidence only and may not flow backward
  into prediction inputs.

## Challenger-first rule

Incumbents remain untouched during experimentation. Challenger team and
player-rate models receive explicit model identity, an explicit configuration
hash, explicit provenance, and deterministic output.

A legitimate PE-7 result is one of:

- an accepted team challenger;
- an accepted player-rate challenger;
- a partially accepted refinement;
- **NO CHANGE**, where the evidence does not justify promotion.

Do not force both model families to change.

## Evaluation contract

Incumbent and challenger are compared on **identical causal walk-forward
populations**. Report sample sizes prominently. Respect PE-2 sample-size honesty:
insufficient evidence stays descriptive and cannot become a promotion claim.

### Team model — primary target

The stored official realised **team-side xG** is the primary continuous target,
because the model estimates scoring environment rather than literal match-result
luck. At minimum report:

- team-side xG MAE;
- team-side xG RMSE;
- team-side xG bias;
- home and away strata;
- team/evidence-volume strata where the sample permits;
- predicted vs realised mean team xG.

Actual goals may be reported as a **secondary descriptive check**, but must not
replace xG as the primary refinement target.

### Player rates — primary target

Player-rate evaluation isolates attack-rate quality from PE-6 minutes
uncertainty. Where possible, evaluate expected player xG/xA using the player's
**realised exposure/minutes** for the target fixture rather than allowing
predicted-minutes error to dominate the rate comparison. At minimum report:

- xG prediction MAE / bias;
- xA prediction MAE / bias;
- useful position strata;
- prior-evidence / low-history strata;
- role-change strata where the sample permits;
- sample sizes for every reported stratum.

Integrated xPts may be reported as a **secondary diagnostic only**. PE-8 owns
calibration, and PE-7 must not promote a model merely because one downstream
xPts sample happens to improve.

## Team → player coherence

PE-7 must explicitly test that:

- team attacking expectation is finite and non-negative;
- player attacking expectations are finite and non-negative;
- player goal/assist mass cannot silently exceed the team's football-event
  environment;
- any unallocated attacking mass is explicit rather than silently redistributed;
- player allocation is stable under row/database ordering;
- double gameweeks remain fixture-atomic;
- blanks produce no invented fixture attack.

Do not alter Monte Carlo draw ordering merely to accommodate a challenger.

## Historical priors

Preserve current truthfulness:

- no previous-season team prior may be invented if the repository does not
  contain causal team-match history;
- older seasons with unavailable xG-family fields remain missing rather than
  genuine zero;
- unverifiable club portability must not be replaced with a fabricated
  team-ratio adjustment;
- role change alters prior trust only through cutoff-observable structured
  evidence.

## Minimum hard tests

PE-7 includes at least:

1. post-cutoff team xG evidence cannot change an earlier team prediction;
2. post-cutoff player xG/xA evidence cannot change an earlier player-rate prediction;
3. a later team transfer cannot reattribute a historical fixture;
4. a later player transfer cannot alter an earlier player-rate prior;
5. a later position change cannot alter an earlier population/prior;
6. a later active/inactive state cannot alter an earlier candidate;
7. scheduled placeholders are excluded;
8. missing xG/xA remains missing, not zero;
9. a low-history player remains structurally projectable through declared shrinkage;
10. the no-history fallback remains finite and explicit;
11. previous-season non-xG-bearing history is not interpreted as zero xG/xA;
12. home/away evidence remains causal;
13. role-change evidence is cutoff-safe;
14. team and player outputs are finite and non-negative;
15. team → player attacking-mass coherence;
16. identical-population incumbent/challenger evaluation;
17. deterministic repeat / row-order invariance;
18. DGW fixture atomicity and blank handling;
19. PE-6 minutes identities unchanged;
20. xPts / MC / scoring / bonus / calibration paths unchanged unless a concrete
    PE-7 integration defect proves otherwise.

## Versioning and RNG

- challenger versions must be explicit;
- no incumbent version bump for experimentation;
- no RNG introduction is required for PE-7;
- do not change the Monte Carlo seed namespace or draw ordering.

## Terminal boundary

PE-7 ends at senior review. Its final state is exactly one of:

- `READY_FOR_MERGE`
- `OPEN`

No PE-8 implementation. No optional hardening after `READY_FOR_MERGE`.
