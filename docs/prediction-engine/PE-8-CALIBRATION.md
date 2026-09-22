# Prediction Engine V1 — PE-8 Calibration

**STATUS: NEXT — CONTRACT PRESENT, IMPLEMENTATION NOT STARTED**

## Purpose

PE-8 answers one question:

> Do the engine's persisted probabilities and expected values mean what they say —
> does a stated probability match the realised frequency of the event it names,
> and is a stated expectation unbiased — and if not, can a causal, versioned,
> validated calibration transform be fitted that makes them mean what they say?

PE-8 is a calibration phase, not a model phase. It does not search for a better
model, a better feature, or a better parameter estimate. It asks whether the
quantities the engine already produces are probabilistically honest, and it
either corrects them through a declared transform or records that the available
evidence does not justify one.

PE-8 is not an automatic rewrite. Every incumbent remains authoritative until
evidence justifies a change, and **NO CHANGE** is a legitimate PE-8 outcome.

PE-8 does not tune minutes, team strength, player rates, certification,
transfer policy, chips, or scoring rules.

## In scope

- calibration diagnosis of the persisted probability surfaces;
- calibration diagnosis of the persisted expected-value surfaces;
- causal, versioned calibration transforms for a diagnosed defect;
- reliability and coverage reporting at declared grains;
- sample-size honesty on every reported calibration figure;
- the FPL assist mapping constant, if and only if evidence supports a fit;
- explicit resolution of the DefCon calibration's single-definition requirement.

## Out of scope

- any change to the PE-6 minutes models;
- any change to the PE-7 team or player attack models;
- any change to `XPTS_MODEL_VERSION` or `MONTE_CARLO_MODEL_VERSION`;
- ranking-policy or selection-policy changes;
- chip, transfer or lineup policy changes;
- certification integration (PE-9);
- end-to-end acceptance (PE-10);
- any new model family, feature set or parameter estimator.

## Central boundary — no in-sample calibration

This is the phase's defining constraint.

**A calibration transform scored on a target event must be fitted only on
outcomes that were finalised strictly before that event.** No same-event
fit-and-score. No in-sample fit. No fit on the evaluation window it is judged on.

PE-8 evaluation is **causal walk-forward (prequential) only**:

- for each scored target event, the transform in force is the one fitted from the
  events finalised strictly before it;
- the fitted parameters for a given origin are frozen once computed, and a later
  realised outcome must not retroactively change an earlier transform;
- a transform fitted on events `1..k` and scored on event `k` is a contract
  violation, not a conservative approximation.

A calibration figure produced by fitting and scoring on the same event set is
**not evidence** and must not be reported as if it were.

## Same-population requirement

A calibration figure is only interpretable when every quantity in it covers the
**same rows**.

- The incumbent and any alternative calibration are compared on an **identical
  causal walk-forward population**, identified by its own digest, not by a count.
- A surface is scored only for rows where the realised outcome exists, the row is
  not a scheduled placeholder, and the required outcome column is non-null. Two
  figures computed over two different row sets are not comparable, however
  similar their `N`.
- A row whose input probability is absent is **excluded and counted**, never
  imputed. Imputing `0.0` would manufacture a well-calibrated-looking zero from a
  missing model output.
- An item that cannot cover the comparison population is reported as
  **unreachable**, carrying the reason, rather than scored on the subset it
  happens to cover.
- Every figure states its grain. A fixture-grain probability figure and an
  event-grain points figure cover different populations in a double gameweek.

## Frozen incumbent identities

Verified in the merged source at the PE-8 base. These are the incumbents PE-8
must not silently rewrite:

| Constant | Value | Location |
|---|---|---|
| `XPTS_MODEL_VERSION` | `xpts_v1.4.1` | `fpl_brain/xpts.py` |
| `MONTE_CARLO_MODEL_VERSION` | `mc_v1.3.0` | `fpl_brain/monte_carlo.py` |

The PE-6 minutes trio remains frozen and is **outside PE-8's promotion target**:

| Constant | Value | Location |
|---|---|---|
| `MINUTES_MODEL_VERSION` | `minutes_v1.8.0` | `fpl_brain/minutes_model.py` |
| `MINUTES_COHERENT_MODEL_VERSION` | `minutes_v1.2.0` | `fpl_brain/minutes_model.py` |
| `JOINT_MINUTES_MODEL_VERSION` | `minutes_v1.5.2` | `fpl_brain/joint_minutes.py` |

The PE-7 team/player attack models were accepted at **NO PROMOTION** and remain
outside PE-8's promotion target:

| Constant | Value | Location |
|---|---|---|
| `TEAM_MODEL_VERSION` | `team_strength_v1.1.0` | `fpl_brain/team_model.py` |
| `PLAYER_RATE_MODEL_VERSION` | `player_rates_v1.0.0` | `fpl_brain/player_rates.py` |

The DefCon calibration is a frozen incumbent too:

| Constant | Value | Location |
|---|---|---|
| `DEFCON_CALIBRATION_VERSION` | `defcon_platt_v1.0.0` | `fpl_brain/defcon_calibration.py` |

No incumbent model version may be bumped merely to run an experiment, and no
incumbent calibration version may be replaced merely to run an experiment.

## Existing infrastructure PE-8 must reuse

PE-8 does **not** invent a metric layer. The PE-2 slice already provides the
measurement machinery, and PE-8 must extend it rather than bypass it.

Declared policy versions already in the frozen source:

| Policy | Version | Location |
|---|---|---|
| `WALK_FORWARD_VERSION` | `walk_forward_v1.0.0` | `fpl_brain/walk_forward.py` |
| `MISSING_DATA_POLICY_VERSION` | `wf_missing_policy_v1.0.0` | `fpl_brain/walk_forward.py` |
| `METRIC_POLICY_VERSION` | `wf_metrics_v1.0.0` | `fpl_brain/walk_forward_metrics.py` |
| `SAMPLE_POLICY_VERSION` | `wf_sample_policy_v1.0.0` | `fpl_brain/walk_forward_scoreboard.py` |
| quantile policy | `wf_quantile_policy_v1.0.0` | `fpl_brain/walk_forward_scoreboard.py` |

Declared grains already in the source:

- `GRAIN_PLAYER_EVENT` — the headline points grain; a double gameweek is one
  summed observation, never one per fixture;
- `GRAIN_PLAYER_FIXTURE` — the grain of the probability metrics and of the
  quantile coverage.

**Every PE-8 figure must state its grain.** A fixture-grain probability figure
and an event-grain points figure are not interchangeable, and in a double
gameweek they cover different populations.

Declared metric functions already available:

- `mean_absolute_error`, `root_mean_squared_error`, `mean_bias`,
  `median_absolute_error`;
- `spearman_rank_correlation`, `top_k_hit_rate`;
- `brier_score`, `brier_reference_score`;
- `central_interval_coverage`, `mean_interval_width`.

The declared bias convention is
`mean(predicted - actual); positive means overprediction`. PE-8 must reuse it
rather than define a second convention.

PE-8 must also reuse the existing calibration-transform precedent rather than
invent a new one. `fpl_brain/defcon_calibration.py` already establishes the
pattern: a versioned spec with a validated method, a canonical identity hash, a
registry that fails closed on an unknown version, and a **spec persisted into
every payload that used it** so a consumer resolves the calibration from the
payload rather than from a global constant. Any PE-8 transform follows that
pattern exactly.

## Calibration surfaces in scope

PE-8 may only evaluate a surface that is genuinely persisted and causally
evaluable in the frozen source. The following are persisted, and PE-8 must not
broaden beyond them without a concrete, evidenced reason.

### Persisted probabilities

The four scored probability surfaces, each persisted in the anchor `xpts_v1`
payload and each already declared with its realised outcome:

| Metric | Persisted field | Realised outcome | Grain |
|---|---|---|---|
| `BRIER_P_START` | `p_start` | `player_gameweeks.starts = 1` | player × fixture |
| `BRIER_P_60_PLUS` | `p_60_plus` | `player_gameweeks.minutes >= 60` | player × fixture |
| `BRIER_CLEAN_SHEET` | `clean_sheet_probability` | clean-sheet points earned | player × fixture |
| `BRIER_DEFCON` | `defcon_p_hit` | `defensive_contribution >= defcon_threshold_for(position)` | player × fixture |

The population policy is already declared: fixture grain, restricted to the
anchor `xpts_v1` run of each target event, for played fixtures of FINAL target
events; a row is scored only when the realised `player_gameweeks` row exists for
the same `(player, fixture)`, is not a scheduled placeholder, has the required
outcome column non-null, and the position defines the required threshold.

### Deliberately excluded surfaces

PE-8 must **not** silently widen onto the following:

- `p_goal` and `p_assist` exist only in `monte_carlo_distributions`, a different
  run family with its own dependency closure. Scoring them would widen the
  provenance surface with no V1 requirement behind it.
- Every `*_xpts` component is an **expected points value, not a probability**. A
  Brier score computed from expected points is meaningless, and PE-8 must not
  compute one.
- The Monte Carlo artifact stores **quantiles, never draws**. Nothing in PE-8 may
  assume a draw-level or full-distribution artifact exists.

If a future slice genuinely needs one of these, it arrives as its own evidenced
artifact-contract change, not as a silent broadening of PE-8.

## Probability calibration reporting

For each of the four probability surfaces, PE-8 reports at minimum:

- Brier score, against the declared realised outcome;
- the **Brier reference score** for the same outcome set, so the number is
  interpretable rather than bare;
- a **reliability** view — the observed frequency in stated probability bins —
  with the **sample size carried on every bin**;
- the population size `N` and the population digest, so the figure is tied to an
  identity rather than to a count;
- the grain, explicitly.

Rules that hold for every reported figure:

- a row whose probability is **absent is excluded and counted**. It is never
  scored as `0.0`;
- a scored probability outside `[0, 1]` **fails closed** — the evaluation stops
  rather than reporting a number;
- a bin below the declared sample floor is reported as **insufficient**, not as a
  calibration claim;
- no arm is ranked, and no skill score is derived from these measurements. PE-8
  measures; it does not declare a winner.

## Expected-value diagnostics

The headline expected-value surface is the persisted model expected points scored
against realised total points at the player × event grain, with MAE, RMSE, bias
and median absolute error already available.

Beyond the headline, PE-8 must report expected-value diagnostics **per persisted
component**, because an aggregate bias can hide a compensating pair of component
biases. The `xpts_v1` payload persists these components:

`appearance_xpts`, `goal_xpts`, `assist_xpts`, `clean_sheet_xpts`,
`goals_conceded_xpts`, `save_xpts`, `bonus_xpts`, `defcon_xpts`,
`yellow_card_xpts`, `soft_xpts`, `core_xpts`, `total_xpts`.

Every component figure must state the grain at which that component is actually
persisted. A component that is persisted per fixture is not reported at event
grain, and components are never summed into an event figure unless the source
does so.

A component whose bias is not causally interpretable — because the component is
an approximation, a proxy, or is flagged as such at production time — must carry
that flag alongside the number.

## Monte Carlo uncertainty

The Monte Carlo artifact persists a **quantile grid carrying the CORE components
only**, with `q10`, `q25`, `q50`, `q75`, `q90` per player-fixture. Bonus is
deterministic in that kernel.

Therefore:

- PE-8 reports **quantile coverage** — `CENTRAL_50_QUANTILE_COVERAGE` over
  `q25`/`q75` and `CENTRAL_80_QUANTILE_COVERAGE` over `q10`/`q90` — at **fixture
  grain**;
- **coverage is not full-distribution calibration.** A band that contains the
  realised value at the stated rate does not certify the distribution's shape,
  its tails, or its dependence structure. PE-8 must never describe a coverage
  figure as a calibrated predictive interval;
- quantiles are **never summed into an event figure**;
- the interval is **closed**: a realised value exactly equal to a bound counts as
  covered;
- a **reversed** quantile pair fails closed — it cannot describe an interval, and
  silently swapping the bounds would report coverage for an interval nobody
  produced;
- **CRPS is not required and must not be claimed.** The artifact stores quantiles,
  never draws, so there is no total-points distribution to score. Introducing
  CRPS would first require persisting a draw-level or total-points distribution
  artifact, which is an artifact-contract change outside PE-8's minimal scope.
- There is no total-points quantile distribution. A total-proxy figure may be
  reported only if it is labelled as a proxy and never as a quantile.

## Calibration transforms

A transform is admissible only when all of the following hold.

1. **A diagnosed defect.** A transform exists to correct a measured miscalibration
   on a declared surface, with the diagnosis reported first. A transform is never
   fitted speculatively.
2. **A causal fit.** Fitted only on outcomes finalised strictly before the origin
   it is applied at.
3. **A declared method.** The method is named and validated. A method that cannot
   state its own monotonicity is not admissible.
4. **Monotone non-decreasing in the input probability.** A transform that is not
   monotone cannot be a probability calibration.
5. **Exact boundary preservation.** A raw probability of exactly `0` stays `0`,
   and exactly `1` stays `1`. Mapping a stated impossibility to a small positive
   number fabricates probability mass and would give every structurally
   ineligible player a non-zero expectation. Any interior clipping is declared
   explicitly and bounded.
6. **A versioned spec with a canonical identity.** The transform is a declared
   artifact with a version string, a validated spec, and a canonical identity
   hash.
7. **Identity bound into provenance.** Every payload produced under a transform
   records which calibration produced it, so a consumer resolves the calibration
   from the payload rather than assuming a global default.
8. **Fail closed on an unknown version.** An unregistered or unrecognised
   calibration version stops the evaluation. There is no silent fallback to the
   identity mapping.
9. **Ranking behaviour stated.** A monotone transform preserves within-event
   ordering, so it must not change a ranking-based decision; it does change an
   expected value, so it can change an expected-value-based decision. PE-8 must
   state which of the two it is claiming, and must not wire a transform into a
   production decision path on unvalidated evidence.

## Sample-size honesty

The PE-2 disclosure floors are already declared and are **not** significance
thresholds:

- `MIN_TARGET_EVENTS_FOR_DESCRIPTIVE = 10`;
- `MIN_OBSERVATIONS_FOR_DESCRIPTIVE = 2000`.

They gate **descriptive reporting only**. Nothing in PE-8 tests a hypothesis and
no arm is ever called better.

PE-8 must therefore additionally declare:

- every figure carries its `N` alongside its value;
- a figure below a floor is reported as **insufficient for descriptive
  reporting**, not as a result;
- an **empty** count is never rendered as `0.0` — a sample of zero is not a score
  of zero;
- a **missing** value is never rendered as `0.0`;
- a figure whose population differs from the comparison population is reported as
  **unreachable** rather than scored;
- a calibration **promotion** requires materially more evidence than a
  descriptive report, because a promotion changes production behaviour.

`OPEN` is the correct terminal outcome when the available evidence can support a
diagnosis but cannot support a promotion.

## FPL assist mapping

The assist mapping constant is currently uncalibrated in production:

| Field | Value | Location |
|---|---|---|
| `XPtsConfig.fpl_assist_mapping_coefficient` | `1.0` | `fpl_brain/xpts.py` |
| `XPtsConfig.assist_mapping_calibrated` | `False` | `fpl_brain/xpts.py` |

The coefficient multiplies expected xA in the player assist expectation on both
the model path and the baseline path, and when the mapping is not calibrated the
production path appends an `FPL_ASSIST_MAPPING_UNCALIBRATED` risk flag.

PE-8 rules for this constant:

- **the coefficient remains `1.0` unless a causal walk-forward fit on finalised
  outcomes supports a different value.** A plausible-sounding correction derived
  from the football rules, with no measured evidence behind it, is not a
  calibration;
- `assist_mapping_calibrated` may become `True` **only** alongside a declared,
  versioned, validated mapping whose provenance is bound into the payload;
- the `FPL_ASSIST_MAPPING_UNCALIBRATED` flag may stop being emitted **only** when
  the mapping is genuinely calibrated. It is a truthful disclosure of the current
  state, and PE-8 must not silence a truthful flag to make an output look
  finished;
- if the evidence does not support a fit, PE-8 records that and leaves both values
  unchanged. That is a legitimate and expected outcome.

## DefCon

The DefCon threshold probability already carries a versioned calibration
(`defcon_platt_v1.0.0`, a PLATT map with a positive slope, plus the explicit
`defcon_identity_v0.0.0` legacy mapping), and it already satisfies the
single-definition requirement PE-8 would otherwise have to impose: the spec is
persisted into the payload, the canonical identity is recorded alongside it, and
the Monte Carlo layer resolves the calibration **from the payload** rather than
from a global default.

PE-8 must therefore:

- **preserve the one-definition rule.** `defcon_p_hit` and the DefCon expected
  points are produced from the same calibrated probability, and every consumer
  must use the same declared calibration. A second, independently derived DefCon
  probability is a defect, not a refinement;
- **not re-fit or replace `defcon_platt_v1.0.0`** merely to run an experiment;
- report the DefCon calibration **identity** alongside any DefCon calibration
  figure, so the number is tied to the spec that produced it;
- respect the existing exclusion: the evaluation's `defcon_threshold_for`
  returns `None` for GKP, so GKP rows are excluded from the DefCon probability
  population rather than scored against a fabricated threshold;
- keep boundary behaviour: a raw probability of exactly `0` or exactly `1` is
  preserved rather than clipped into the interior.

## PE-3 carry-forward

`CONTINUOUS_PROXY_TIE_LIMITATION` from PE-3 remains an open, accepted limitation.
PE-8 must not represent it as resolved, and must not present a calibration figure
that depends on continuous-proxy tie ordering as if the tie limitation did not
apply.

## Minimum hard tests

PE-8 includes at least:

1. no same-event fit-and-score;
2. a transform scoring target event `E` is fitted only on outcomes finalised
   strictly before `E`, so a post-cutoff outcome cannot change an earlier
   transform;
3. the transform is a declared versioned artifact with a canonical identity hash,
   and that identity is bound into the provenance of every payload it produced;
4. an unregistered or unknown calibration version fails closed, with no silent
   fallback to identity;
5. a probability transform is monotone non-decreasing in the raw probability, or
   it is rejected;
6. a transform preserves the boundaries exactly: raw `0` stays `0`, raw `1`
   stays `1`;
7. `p_start` scored by Brier and by reliability, on the declared anchor
   population;
8. `p_60_plus` scored by Brier and by reliability, on the same population;
9. `clean_sheet_probability` scored by Brier and by reliability, on the same
   population;
10. `defcon_p_hit` scored by Brier and by reliability, with GKP excluded because
    the position defines no threshold;
11. a row whose probability is absent is excluded and counted, never scored as
    `0.0`;
12. a scored probability outside `[0, 1]` fails closed;
13. every reliability bin carries its sample size, and a sub-floor bin is reported
    as insufficient rather than as a calibration claim;
14. expected-points bias is reported per persisted component under the declared
    bias convention, and an aggregate figure alone is not accepted;
15. every component figure states the grain at which the component is persisted;
16. Monte Carlo quantile coverage is reported as coverage, never as a calibrated
    predictive interval;
17. coverage is computed at fixture grain and quantiles are never summed into an
    event figure;
18. a reversed quantile pair fails closed;
19. no CRPS claim is made unless a draw-level artifact is genuinely persisted;
20. the FPL assist mapping coefficient remains `1.0` unless causal evidence
    supports a fitted value;
21. `assist_mapping_calibrated` may only become `True` alongside a declared,
    versioned, validated mapping, and the `FPL_ASSIST_MAPPING_UNCALIBRATED` flag
    stays truthful until then;
22. the DefCon calibration is applied through a single definition shared by every
    producer, and its identity is carried on every DefCon calibration figure;
23. incumbent and challenger or incumbent and alternative calibration are compared
    on identical causal walk-forward populations, with sample sizes reported, and a
    population mismatch is reported as unreachable rather than scored;
24. deterministic repeat and row-order invariance; double-gameweek fixture
    atomicity and blank handling; and unchanged incumbent model identities and
    DefCon calibration identity unless a concrete PE-8 integration defect proves
    otherwise.

## Versioning and RNG

- calibration versions must be explicit;
- no incumbent model version bump for experimentation;
- no incumbent calibration version replacement for experimentation;
- PE-8 does not require introducing RNG, and must not change the Monte Carlo seed
  namespace or draw ordering;
- a newly fitted calibration receives a new version string and a new identity
  hash; an existing version is never redefined in place, because a redefined
  version silently invalidates every artifact that recorded it.

## Terminal boundary

PE-8 ends at senior review. Its final state is exactly one of:

- `READY_FOR_MERGE`
- `OPEN`

`OPEN` is the correct state when the diagnosis is supported but the evidence does
not support a promotion, or when the available sample is insufficient for
anything beyond a descriptive report.

No PE-9 implementation. No optional hardening after `READY_FOR_MERGE`.
