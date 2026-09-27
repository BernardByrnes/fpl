# Prediction Engine V1 — PE-9 Certification Integration

**STATUS: NEXT — CONTRACT PRESENT, IMPLEMENTATION NOT STARTED**

## Purpose

PE-9 answers one question:

> Which prediction artifacts are permitted to enter the certified prediction bundle,
> and therefore become consumable by downstream FPL decision logic?

PE-9 is an **integration and certification** phase. It is not another football-model
phase. It decides what may be *used*, not how well any model predicts.

Every model and every calibration in this engine is frozen and remains authoritative.
PE-9 changes no predictive quantity. `NO CHANGE` — meaning the existing certification
machinery is extended rather than replaced, and no model or calibration is promoted —
is the expected outcome.

## Authority and base

| | |
|---|---|
| Base (authority) commit | `95b6f71afbe6996b975f7815de854449edf892ff` |
| Base tree | `31240944c2b52336455ce05eeece224a8af86730` |
| Base is the PE-8 merge | reviewed `491b8ae58c3b570ab60d783cf4ec21a2b987d821`, reviewed tree `31240944…` |
| Base branch | `feature/prediction-engine-v1` |
| Base CI | run `36003236203` — completed/success, `unit-tests` success |

PE-1 through PE-8 are FROZEN. PE-9 does not reopen them absent a concrete regression.

## Current architecture (verified in the base source)

Prediction artifacts live in SQLite. One run per model family is written to
`projection_runs`, and each family persists its own rows:

| Family | Table |
|---|---|
| `minutes_v1` | `frozen_predictions` |
| `team_strength_v1` | `team_fixture_projections` |
| `player_rates_v1` | `player_rate_projections` |
| `xpts_v1` | `player_fixture_xpts_projections` |
| `monte_carlo_v1` | `monte_carlo_distributions` |

The declared dependency graph is `fpl_brain/certified_bundle.py::BUNDLE_DEPENDENCIES`:

```
minutes_v1 ─┐
team_strength_v1 ─┼──> xpts_v1 ──> monte_carlo_v1
player_rates_v1 ─┘
```

`fpl_brain/certified_bundle.py` (341 lines) is the canonical certification module.
It exposes exactly three entry points — `validate_certified_bundle`,
`certified_bundle_from_explicit_ids`, `certify_horizon_bundles` — and deliberately
contains **no** "discover the latest run per family" helper.

The single production certification boundary is
`fpl_brain/four_gw_decision.py::event_support_from_certification`, which takes a
*certification artifact* and returns per-event support derived from its exact
certified run ids. `scripts/run_four_gw_decision.py` refuses to produce a production
decision without `--certification`.

## Existing certification capabilities — verified, and NOT to be duplicated

Each of the following was confirmed directly in the base source. PE-9 must **extend**
this path, never create a second certification system beside it.

| Property | Where | Enforced? |
|---|---|---|
| Explicit run ids per family, no discovery helper | `certified_bundle.py` (module scope) | **Yes** |
| Run exists | `validate_certified_bundle` | **Yes** |
| Family identity matches the declared family | `validate_certified_bundle` | **Yes** |
| Exact planning event | `validate_certified_bundle` | **Yes** |
| `status == complete` | `validate_certified_bundle` | **Yes** |
| Exact data cutoff (string equality across families) | `validate_certified_bundle` | **Yes** |
| Dependency-edge coherence (xPts → 3 upstreams; MC → 4 upstreams) | `validate_certified_bundle` | **Yes** |
| `planning_context_hash` identical across families | `validate_certified_bundle` | **Yes** |
| Code snapshot identical across families | `validate_certified_bundle` | **Yes** |
| One deterministic bundle identity, shared by producer and consumer | `canonical_bundle_identity` | **Yes** |
| All incoherences reported at once | `BundleIncoherent(reasons)` | **Yes** |
| Horizon-level certification (raises on the first incoherent event) | `certify_horizon_bundles` | **Yes** |
| Required model versions | `required_versions` parameter | **API only — never supplied** |

A subtle but important verified property: `projection_runs.source_snapshot_sha256` is
the **code** fingerprint, not a data identity. The validator says so explicitly and
never compares it against a data snapshot. A caller-supplied `data_snapshot_sha256` is
recorded on the bundle and carried to the decision engine, but is **not** validated
against anything.

## Proven gaps

These were established by reading the base source, not assumed. Each is a concrete
PE-9 target.

1. **Model-version pinning is unenforced.** `required_versions` exists on
   `validate_certified_bundle`, `certified_bundle_from_explicit_ids` and
   `certify_horizon_bundles`, but **no caller in `fpl_brain/` or `scripts/` ever
   supplies it**. The bundle records whatever versions the runs happen to carry. A run
   produced by an unexpected model version certifies today.

2. **Certification is mandatory at the runner, not in the library.**
   `scripts/run_four_gw_decision.py` refuses without `--certification`, and
   `event_support_from_certification` is the canonical boundary — but
   `route_optimizer.build_event_worlds` accepts a caller-supplied `bundles` mapping and
   never validates it. Several scripts construct bundles directly and feed the full
   decision path with no certification at all, including
   `scripts/build_route_comparison.py` and `scripts/final_operational_refresh_gw04.py`,
   both of which resolve runs by *latest per family*.

3. **A second, uncertified readiness path runs in production.**
   `scripts/run_four_gw_decision.py` computes the horizon from
   `four_gw_decision.event_support_from_db` — newest-per-family rediscovery, documented
   in its own docstring as non-production — *before* the certification gate, and the
   certification result is never reconciled with it. The readiness view and the decision
   view can therefore describe different predictive worlds.

4. **No calibration artifact participates in certification.** The merged PE-8 artifact
   is not an input to `certified_bundle` or to the certification boundary in any way.

5. **A missing upstream run degrades silently.**
   `fpl_brain/monte_carlo.py` does
   `analytics.get_projection_run(conn, int(minutes_run_id)) or {}`, so an unknown
   minutes run yields empty metadata instead of failing closed.

6. **A missing projection is still summed as zero at the arithmetic site.**
   `fpl_brain/candidate_universe.py` returns an event feature with `0.0` values and a
   `MISSING_PROJECTION` label, and for a double gameweek sums whichever fixture rows
   exist — recording `partial` but understating the value. The label exists; the zero is
   already in the value.

7. **The data snapshot identity is unvalidated.** It is supplied by the caller and
   never checked against stored evidence.

## PE-8 handoff

The merged PE-8 artifact (`fpl_brain/calibration_evaluation.py`, entered through
`evaluate(conn, artifact=..., events=...)`) exposes, for the certified anchor:

- **terminal state**: `READY_FOR_MERGE` or `OPEN`.
- **per-surface status**: `OK`, `UNREACHABLE`, `POPULATION_MISMATCH`, `NOT_FITTED`.
- **per-surface diagnosis**: `MISCALIBRATED`, `NO_MATERIAL_DEFECT_DETECTED`,
  `INSUFFICIENT_FOR_DIAGNOSIS`.
- **probability surfaces covered**: `p_start`, `p_60_plus`, `clean_sheet_probability`,
  `defcon_p_hit` — each at player × fixture grain, each with Brier, a reference score,
  a reliability view, population `N` and its digest.
- **applicable vs non-applicable surfaces**: per-origin applicability is decided by the
  certification, and inapplicable events are reported rather than silently skipped.
- **fitted vs not-fitted**: a transform is reported `NOT_FITTED` when no basis is
  admissible, which is distinct from any failure.
- **diagnosed vs insufficient evidence**: `MISCALIBRATED` and
  `NO_MATERIAL_DEFECT_DETECTED` are both diagnoses; `INSUFFICIENT_FOR_DIAGNOSIS` is the
  explicit absence of one. All three are distinct from an incoherent or missing
  artifact.
- **causal fit-basis provenance**: seven declared exclusion codes
  (`POINT_IN_TIME_CAPTURE_ABSENT`, `CAPTURE_NOT_BEFORE_CUTOFF`,
  `OBSERVATION_PROVISIONAL_AT_CUTOFF`, `OFFICIAL_FINALITY_UNPROVABLE`,
  `OFFICIAL_FINALITY_NOT_BEFORE_CUTOFF`, `POINT_IN_TIME_OUTCOME_UNAVAILABLE`,
  `FROZEN_PREDICTION_FIELD_ABSENT`), with a declared precedence.
- **declared material tolerances**: `MATERIAL_CALIBRATION_GAP_TOLERANCE = 0.05`,
  `MATERIAL_EV_BIAS_TOLERANCE_POINTS = 0.25`,
  `MATERIAL_COMPONENT_BIAS_TOLERANCE_POINTS = 0.25`. These are PE-8's declared,
  versioned thresholds; PE-9 must **reuse** them and must not invent new ones.
- **assist mapping**: `NO_CHANGE` or `CANDIDATE_FOR_REVIEW`, with the truthful
  `FPL_ASSIST_MAPPING_UNCALIBRATED` flag still emitted.
- **Monte Carlo diagnostics**: quantile coverage only, at fixture grain; CRPS is not
  implemented because the artifact stores quantiles, never draws.
- **expected-value diagnostics**: per persisted component, each carrying the production
  flags that make its bias a description rather than a causal claim.
- **unresolved limitations**: `CONTINUOUS_PROXY_TIE_LIMITATION` carried, not resolved.

**PE-8 established no promotion and no wiring.** The artifact declares, verbatim:

- `promotion`: *"NOT PERFORMED; every incumbent stays authoritative, no model version is
  bumped and no calibration version is replaced"*
- `production_wiring`: *"NONE; no transform fitted here is wired into a production
  decision path…"*

PE-9 must not claim otherwise, and must not treat the presence of a *fitted* transform
as authorisation to use it. `NOT_FITTED`, `NO_MATERIAL_DEFECT_DETECTED` and
`INSUFFICIENT_FOR_DIAGNOSIS` are **legitimate, distinct, non-failure states** and must
carry that meaning into certification.

## Certification states

A boolean is not enough. The vocabulary below is deliberately small, fail-closed, and
reuses tokens already established in the engine.

**Per-bundle structural state** (extending the existing module):

| State | Meaning |
|---|---|
| `CERTIFIED_COHERENT` | every structural gate passes and every required evidence artifact is present |
| `EVIDENCE_LIMITED` | structurally valid; PE-8 evidence is insufficient for a calibration *claim*, which is not a failure |
| `CALIBRATION_NOT_APPLICABLE` | the surface has no calibration question (e.g. an identity mapping); explicitly not a defect |
| `EVIDENCE_MISSING` | a required artifact or field is absent |
| `PREDICTIVE_BUNDLE_INCOHERENT` | the existing token: families do not form one predictive world |
| `UNSUPPORTED_MODEL_VERSION` | a family's version is not the expected one |
| `CERTIFICATION_BLOCKER_UNRESOLVED` | a disclosed limitation blocks the claim being made |

**Per-horizon state**: `DECISION_HORIZON_COMPLETE`, `SEASON_END_SHORT_HORIZON`, or the
existing `DECISION_HORIZON_INCOMPLETE`.

**Phase terminal state**: `READY_FOR_MERGE` or `OPEN`.

Rules:

- a state is never collapsed into `certified: true/false`;
- `EVIDENCE_LIMITED` and `CALIBRATION_NOT_APPLICABLE` are **not** failures, and must not
  be reported as such;
- `EVIDENCE_MISSING` and `PREDICTIVE_BUNDLE_INCOHERENT` are **not** the same thing;
  insufficient evidence is not incoherence.

## Structural gates

Structural integrity gates are **absolute** — they are properties of identity and
coherence, not of predictive quality, so they need no empirical threshold.

PE-9 keeps every existing gate listed above and adds:

1. **Required model versions are supplied.** Every certification call site passes
   `required_versions` from a single declared source, so gap 1 closes and a run from an
   unexpected version fails with `UNSUPPORTED_MODEL_VERSION`.
2. **The data snapshot identity is validated**, not merely recorded: where the bundle
   declares one it must match the certification artifact's, and a mismatch is
   incoherence.
3. **The horizon is certified as a horizon.** All required events are certified from
   one certification artifact; a horizon assembled from two artifacts is refused.
4. **One certification artifact per decision.** A decision consumes exactly one
   artifact, and every event it needs comes from that artifact.

No empirical threshold is introduced by any of these.

## Evidence and calibration gates

Certification may *consult* the PE-8 artifact, but the following separation is absolute:

- **A structural gate may fail a bundle.** Coherence, identity and version are
  properties of the artifact's construction.
- **An evidence gate may not fail a bundle for insufficiency.** Insufficient evidence
  yields `EVIDENCE_LIMITED`, never a refusal and never a pass.
- **No new numerical threshold is introduced.** If PE-9 needs a materiality test it
  uses PE-8's declared tolerances, at PE-8's declared grain, with PE-8's causal basis
  and boundary behaviour. Inventing a threshold because the phase is called
  "certification" is prohibited.
- **A calibration identity must be known and declared.** An artifact naming an unknown
  or mismatched calibration identity fails closed; an absent identity where one is
  required fails closed. `NOT_FITTED` (no admissible basis) is distinct from both.
- **Evidence from a different prediction world is refused.** A calibration artifact
  whose anchor, cutoff or run ids do not belong to the bundle being certified cannot
  certify it.

## Horizon rule

The normal-transfer horizon remains, unchanged:

> current Gameweek plus the next 3 exactly — four events.

- All four events must have one fresh coherent certification context, from one
  certification artifact.
- If the required horizon cannot be certified, the result is
  `DECISION_HORIZON_INCOMPLETE`. There is no H1-only transfer recommendation.
- Lineup and captain remain H1 only.
- The existing season-end carve-out (`SEASON_END_SHORT_HORIZON`) is preserved exactly as
  it is today; PE-9 does not extend or reinterpret it.
- One bad event makes the horizon incomplete. Partial horizons are never silently
  padded or truncated.

## Freshness and coherence contract

"Fresh" is defined by **exact identity**, never by a vague recency comparison:

- the planning event,
- the data cutoff (exact string equality, already enforced),
- each family's explicit run id,
- the bundle's canonical identity,
- the code snapshot identity,
- the planning context hash,
- the model versions,
- the calibration identity where one applies.

One event or family from a different predictive world fails closed. A
"recent enough timestamp" rule is deliberately not introduced: the exact identities
above are strictly stronger, and adding a timestamp rule would weaken them.

## Downstream consumer contract

The canonical certification boundary for the decision path is:

```
certification artifact
  -> four_gw_decision.event_support_from_certification      (mandatory gate)
  -> route_optimizer.build_event_worlds
  -> transfer decisions / captain / lineup / chip consumers
```

Requirements:

- downstream code consumes **exact certified ids and identities**, never a rediscovered
  prediction run;
- `build_event_worlds` and its consumers must not obtain run ids by any other route;
- any consumer that currently reaches prediction data without passing the boundary must
  be identified explicitly before it is changed. Known bypasses, to be named in the
  implementation's evidence rather than fixed by assumption:
  `route_optimizer.build_event_worlds`'s caller-supplied `bundles`;
  `scripts/build_route_comparison.py`;
  `scripts/final_operational_refresh_gw04.py`;
  `scripts/gw4_current_final_board.py`;
  `scripts/live_fire_gw04.py`;
  the `event_support_from_db` readiness path in `scripts/run_four_gw_decision.py`;
  `fpl_brain/monte_carlo.py`'s `or {}` upstream lookup;
  `fpl_brain/candidate_universe.py`'s zero-valued missing feature.
- a consumer that cannot be certified must **refuse**, not degrade.

This contract states the boundary. PE-9 implementation closes the bypasses; this
authority task does not.

## DGW / blank / event grain

PE-4 is preserved exactly.

- Certification must never collapse player × fixture predictive evidence into
  player × event before the frozen aggregation boundary.
- Double-gameweek fixture atomicity remains intact: a fixture is certified or refused as
  a unit.
- A blank event remains a valid zero-fixture event world where the frozen rules say so.
  A blank is **not** missing data, and must not be fabricated into rows or refused as if
  a projection were absent.
- The distinction between "no fixture" and "fixture with no projection row" is
  preserved and reported, not flattened into a single zero.

## Persistence and provenance

- The certification result is persisted with the decision artifact, carrying at
  minimum: the certification artifact identity, each event's canonical bundle identity,
  the per-family run ids and model versions, the cutoff, the code and data snapshot
  identities, the planning context hash, the certification state per bundle, and any
  calibration identity consulted.
- The persisted identity **round-trips exactly**: recomputing it from the stored bytes
  yields the same value, using the one shared algorithm.
- Mutation of current tables — `players`, `fixtures`, `events`,
  `player_gameweeks` — cannot rewrite a historical certified identity, because the
  identity is built from explicit run ids and their recorded identities, never from
  current rows.

## Fail-closed diagnostics

Every refusal carries a specific token. The following are required to exist and to be
distinguishable:

- `PREDICTIVE_BUNDLE_INCOHERENT` (existing) — with the offending reasons.
- `UNSUPPORTED_MODEL_VERSION` — naming the family, the found version and the required
  one.
- `EVIDENCE_MISSING` — naming the absent artifact or field.
- `CERTIFICATION_BLOCKER_UNRESOLVED` — naming the disclosed limitation.
- `DECISION_HORIZON_INCOMPLETE` (existing).
- a certification-artifact mismatch token distinguishing *absent* from *contradictory*.

There must be no path in which a missing artifact, a missing row, or a mismatched
identity produces a numeric zero, a defaulted version, or a silently chosen substitute
run.

## Hard test cases

PE-9 includes at least:

1. a coherent complete bundle is accepted;
2. a missing family is refused;
3. a wrong planning event is refused;
4. a wrong cutoff is refused;
5. an incomplete run is refused;
6. xPts referencing a different upstream run is refused;
7. Monte Carlo referencing a different xPts or upstream run is refused;
8. an unsupported model version is refused once `required_versions` is supplied;
9. code-snapshot incoherence across families is refused;
10. a planning-context mismatch is refused;
11. a required PE-8 calibration artifact that is absent is refused;
12. an unknown calibration identity is refused;
13. `NOT_FITTED` and `NO_MATERIAL_DEFECT_DETECTED` are handled as legitimate,
    non-failure outcomes and are distinguishable from incoherence;
14. PE-8 insufficient evidence is handled distinctly from incoherence and yields
    `EVIDENCE_LIMITED`;
15. calibration evidence belonging to a different prediction world is refused;
16. DGW fixture-grain coherence is preserved through certification;
17. a blank event is treated per PE-4 — a valid zero-fixture world, not missing data;
18. one bad event makes the required four-event normal-transfer horizon incomplete;
19. downstream code cannot rediscover a different "latest" run;
20. a missing projection never becomes zero;
21. the persisted certification identity round-trips exactly;
22. mutating `players`, `fixtures`, `events` or `player_gameweeks` cannot rewrite a
    historical certified identity;
23. `CONTINUOUS_PROXY_TIE_LIMITATION` remains disclosed after certification;
24. chip, transfer, scoring and RNG behaviour is unchanged by certification
    integration alone.

## Out of scope

PE-9 does not:

- redesign minutes, team strength, player rates, xPts, Monte Carlo, structural
  bonus/BPS, or any PE-8 calibration machinery;
- change transfer optimization, captaincy, chip strategy, or FPL scoring rules;
- promote, replace or retune any model or calibration — PE-8 established
  `promotion: NOT PERFORMED` and `production_wiring: NONE`, and PE-9 does not overturn
  that;
- invent numerical calibration or performance thresholds;
- introduce arbitrary future discounting;
- alter the exhaustive official player discovery contract;
- change the Monte Carlo seed namespace or draw ordering;
- add a second certification system alongside `certified_bundle.py`.

## Terminal boundary

PE-9 ends at senior review. Its final state is exactly one of:

- `READY_FOR_MERGE`
- `OPEN`

`OPEN` is the correct state when the integration is complete but a disclosed
limitation, an identified bypass, or insufficient evidence prevents a certification
claim from being made. Certification results *inside* the artifact carry the concrete
per-bundle states above.

No PE-10 implementation. No optional hardening after `READY_FOR_MERGE`.
