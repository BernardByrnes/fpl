# PE-10 — End-to-End V1 Acceptance

**Status:** acceptance evidence recorded; V1 is **READY_TO_FREEZE, pending the separate freeze
merge**, which remains a Product Owner decision this document does not take.  One timing-evidence
exception is recorded below with explicit Product Owner approval (§16): per-family Minutes and
Monte Carlo durations were **not captured**, so no shares are claimed anywhere in this record.
The exception waives the timing-evidence requirement only — no certification, provenance,
manager-state, decision or safety gate is waived.

## 1. Identity of this acceptance

| Fact | Value |
| --- | --- |
| Acceptance code at the predictive run | `78ea86853ba3f13829bb3be0e211352a75ce9119` (branch `feature/pe10-v1-acceptance`) |
| Predictive code identity (frozen at certification) | `983386e1674cf5d6c74fa08ee2c5fb4488e9c92ce9cca88da8462ebf089b02d5` |
| Repair commit (the code this document records) | `ff18c2f491dfc5767d26c76dbf7a21b3a98ba00c` (tree `2e7d7eac09ae8da3504dad1ec1762de97660df57`, parent `78ea86853ba3f13829bb3be0e211352a75ce9119`) |
| Acceptance-document commit | `deafe84cfed774129f2987424ffa6b5415b48ea5` (tree `969bf0bac87a49175d5a7ba783481f018504d717`) |
| This candidate (exception + corrections) | filled by the commit that introduces the §16 corrections |
| Repair scope | `scripts/run_four_gw_decision.py` (+cutoff guard, cache location), `tests/test_pe9_production_decision.py` regressions; **0 files under `fpl_brain/`** |
| Decision runner identity (executed) | `scripts/run_four_gw_decision.py:route_optimizer_v8b_1.0.0` / `sha256:f29ed44a2e65159489e8a354f51979ed1df2e32f06c7151bcf97619331f37c62` |
| Authorization fingerprint (carried forward, identity-only) | `5b253d0f3b6078e6d25228cb89a6b54c5340371cfc51ace16ecb3abeaa9a822e` |

The predictive identity and the decision-runner identity are recorded **separately**: the
generation binds the code that produced the predictions, the decision record binds the code
that took the decision.  A repair to the decision runner does not restate, and did not
change, the predictive identity.

## 2. Horizon and planning instant

| Fact | Value |
| --- | --- |
| Planning event | GW6 |
| Horizon | GW6, GW7, GW8, GW9 (`FOUR_GW`, `horizon_length = 4`) |
| Horizon state | `DECISION_HORIZON_COMPLETE` |
| Planning cutoff | `2026-09-28T15:25:31Z` |
| Execution UUID | `76660eb4-9d1d-4ef7-a493-dd26555821dd` |
| Pinned snapshot | `c29cf985a4ab3a5ca517d967b9d3d8b678afd6f7e2afd6bf91de94b291c00d4a` |
| Snapshot size / capture / consistency window | 1,116,467,200 bytes / **9.635668 s** capture / 10.000 s consistency window |
| Certificate generation | `sha256:454c51f289b0011620628ee3c818eda47c3e602f478cb7df46bc750e2cd76172` |
| Previous generation (historical evidence only) | `sha256:27ae1b0ace157be36ef73758a635121d5a93b44f9cf26bdaa7a2c4085d48d4a0` |

## 3. Manager-state provenance

Product Owner confirmed the live manager state; it was committed through the canonical
confirmation path (`scripts/confirm_manager_state.py`), never by direct table writes.

| Fact | Value |
| --- | --- |
| Entry / event | 241392 / GW6 |
| Free transfers remaining | 3 |
| Bank | £0.7m (7 tenths) |
| Event-start free transfers | `null` — not inferred, and permitted on this non-chip path |
| Confirmation source label | `pe10_product_owner_confirmed_gw6` |
| Confirmed at | 2026-09-28T15:16:59Z |
| Resolved authority | `user_confirmed_override` (`bank_source = manual`, `free_transfers_source = manual`) |
| Squad | 15 players / 15 unique, from the acquisition ledger; **cross-checked 15/15 against the visible live squad** |
| Selling-price basis | official market value 1003, realisable selling value 994, no gaps, no mismatches |

The previous generation's snapshot carried no manager evidence (`data_gap`), which is why
`MANAGER_STATE_EVIDENCE_MISSING` refused it.  This acceptance uses a **new** snapshot that
contains the confirmed state, and the decisive pre-Monte-Carlo check proved it from the
pinned snapshot itself: bank 7, free transfers 3, both `manual`, squad 15, route state 7/3.

## 4. Certified predictive bundle

| Event | minutes run | minutes version | team run | rates run | xPts run | Monte Carlo run | dependency closure | PE-8 evidence state |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GW6 | 569 | minutes_v1.5.2 | 570 | 572 | 575 | 576 | PASS | EVIDENCE_LIMITED |
| GW7 | 582 | minutes_v1.5.2 | 583 | 585 | 588 | 589 | PASS | EVIDENCE_LIMITED |
| GW8 | 595 | minutes_v1.5.2 | 596 | 598 | 601 | 602 | PASS | EVIDENCE_LIMITED |
| GW9 | 608 | minutes_v1.5.2 | 609 | 611 | 614 | 615 | PASS | EVIDENCE_LIMITED |

Required versions (the ONE declared authoritative source): `{"minutes_v1": "minutes_v1.5.2", "monte_carlo_v1": "mc_v1.3.0", "player_rates_v1": "player_rates_v1.0.0", "team_strength_v1": "team_strength_v1.1.0", "xpts_v1": "xpts_v1.4.1"}`

For every event, `xpts.minutes_run_id == monte_carlo.minutes_run_id == bundle["minutes_v1"]`,
and the certified minutes lineage is `minutes_v1.5.2` — the joint kernel the production
freeze consumes — with exact team/rates/xPts/Monte Carlo closure.  Selection is
dependency-derived; creation order is not consulted.

## 5. Generation verification

`verify_generation sha256:454c51f289b0011620628ee3c818eda47c3e602f478cb7df46bc750e2cd76172` → `verified = true`, with every claimed boundary reproduced from
retained evidence:

`runs_exist`, `runs_complete`, `versions_valid`, `dependency_closure_reproduced`,
`snapshot_identity = VERIFIED`, `snapshot_source_identity_reproduced`,
`planning_context_reproduced_from_snapshot`, `code_identity_reproduced_from_runs`,
`bundle_identities_reproduced`, `manifest_digest_matches`, `horizon_state`.

PE-8 evidence is reported truthfully: `NOT_CONSULTED` — no PE-8
reference or reproduction is claimed, exactly as for the previously accepted generation.

## 6. Production decision

| Fact | Value |
| --- | --- |
| Decision id | `sha256:bf93b936cd73ce1f00fbf519710860598db59fa5af1bf1b8fafcd0d5d41404c1` |
| Generation consumed | `sha256:454c51f289b0011620628ee3c818eda47c3e602f478cb7df46bc750e2cd76172` |
| Result digest | `sha256:1e03814633816eb065f618e478d6dc6a4eac61122f29ce59bb9e9515af9466fc` |
| Manager context digest | `sha256:833967288ac8e2aebcaf6d6154a523ff88f51d37096b32a1f414c200f5b46425` — recomputed from the consumed state, equal |
| Cutoff guard | `PASS` — `override_captured_at = 2026-09-28T15:16:59Z`, cutoff `2026-09-28T15:25:31Z`, delta 512.0 s |
| Decision artifact file | `<runtime>/pe9_decisions/gw06/1e03814633816eb065f618e478d6dc6a.json`, sha256 `cff53ba58d617b194fe501f190431b574b4222ed8b96ba58ea89ad988f2cb28e` |
| Suppressions | none (`suppression_reasons: []`) |
| Chip result | **NOT APPLICABLE** — no chip was evaluated on this four-GW decision path and none was forced.  Grounded in the retained run summary and artifact: `predictive_certification_only: true`, `wildcard_evaluated: false`, `transfers_or_chips_executed: 0`, and the decision payload's `wildcard_screen: null` with no chip block emitted. |

**Exact generation binding:** the identifier is identical across the verified generation, the
generation row, `engine_decision_records.generation_id`, and the artifact's own provenance
(`provenance.generation_id`, `provenance.readiness_generation_id`, and per-event
`world_info.*.generation_id`).  PASS.

**Manager attribution:** the snapshot-derived canonical state, the state actually consumed,
the retained `consumed_manager_state` (bank 7 / FT 3 / event-start FT `null`), the
`manager_context_sha256` input and the verifier's re-derivation are identical.

## 7. Decision verification

`verify_decision sha256:bf93b936cd73ce1f00fbf519710860598db59fa5af1bf1b8fafcd0d5d41404c1` → `verified = true`, with `generation_verified`,
`record_digest_recomputed`, `manager_packet_digest_verified`, `manager_context_digest_verified`,
`request_digest_verified`, `result_digest_verified`, `runner_identity_verified`,
`runner_code_identity_bound`, `decision_artifact = VERIFIED` at
`cff53ba58d617b194fe501f190431b574b4222ed8b96ba58ea89ad988f2cb28e`.

Replay boundary, stated truthfully: `DECISION_REPLAY_NOT_PERFORMED` — the generation and the
manager context re-derive from retained evidence; the decision is **not** claimed to have been
re-played.

## 8. Reproducibility and cache non-authority

| Execution | created at | decision id | result digest | seconds |
| --- | --- | --- | --- | ---: |
| warm run 1 | 2026-09-28T18:42:25Z | `sha256:bf93b936cd73ce1f00fbf519710860598db59fa5af1bf1b8fafcd0d5d41404c1` | `sha256:1e03814633816eb065f618e478d6dc6a4eac61122f29ce59bb9e9515af9466fc` | 1,965.2 |
| warm repeat | 2026-09-28T19:18:11Z | `sha256:9de2a7f220777ef456173a62a17cb33bc85b1486c2e4664c6807ff2ade67b86e` | `sha256:1e03814633816eb065f618e478d6dc6a4eac61122f29ce59bb9e9515af9466fc` | 2,015.1 |
| cold cache | 2026-09-28T20:46:57Z | `sha256:f186062538a1abaf419b44bb02b51bbfb96573d7baaabd0e394aa3ac10eb7403` | `sha256:1e03814633816eb065f618e478d6dc6a4eac61122f29ce59bb9e9515af9466fc` | 5,075.7 |

**Three** independent executions of the same certified generation, manager packet, profile and
request produced the **same** `result_sha256`
(`sha256:1e03814633816eb065f618e478d6dc6a4eac61122f29ce59bb9e9515af9466fc`) and the same
published artifact digest.  Decision-record ids differ because the record binds its own creation
context; the decision result does not change.  Runner identity for all three:
`scripts/run_four_gw_decision.py:route_optimizer_v8b_1.0.0`.

Cache non-authority: the second and third executions used the content-addressed world cache
(`world_cache/manager_worlds`, keyed by generation id + run ids + Monte Carlo identity + draws
+ seed + union); the cache is an optimisation only.  The cold run re-derived the worlds after
the cache was removed (8 files, contents preserved as evidence) and produced the **same** result digest.

## 9. Fail-closed refusals (representative)

Exercised against a **copy** of the accepted runtime; the accepted generation was untouched.

| Case | Outcome |
| --- | --- |
| unknown generation | REFUSED -> UNKNOWN_GENERATION_ID |
| manager-state mismatch (asserted bank differs) | REFUSED -> MANAGER_STATE_MISMATCH |
| nested predictive injection (bundles in the packet) | REFUSED -> PRODUCTION_DESCRIPTOR_ONLY |
| nested cache handle injection (cache_dir in the packet) | REFUSED -> PRODUCTION_DESCRIPTOR_ONLY |
| caller executor injection is not an argument | REFUSED -> PRODUCTION_DESCRIPTOR_ONLY |
| tampering with a certified generation row | REFUSED by the database -> PE9_APPEND_ONLY: a certified generation is immutable |
| missing snapshot evidence (gate) | REFUSED -> GENERATION_SNAPSHOT_UNVERIFIED |

Tampering with a certified generation row is refused by the database itself
(`PE9_APPEND_ONLY`: a certified generation is immutable), so the missing-snapshot case was
exercised directly through the same gate `make_decision` calls first
(`GENERATION_SNAPSHOT_UNVERIFIED`).

## 10. Operational timing

Every value below is read from retained evidence; the source is named so the boundary of each
number is unambiguous.

| Stage | Seconds | Retained source / boundary |
| --- | ---: | --- |
| Official refresh | 2.0 (stage) / 2.943 (process) | `fetch_runs` id 49, 15:25:02→15:25:04Z; 2.943 s is the outer process measurement |
| Immutable snapshot capture | 9.635668 | `snapshot_capture_seconds` (10.000 s is the consistency window, a different boundary) |
| Freeze GW6 (minutes + 4 variants + team + rates + xPts + Monte Carlo) | 757.9 | `per_event.6.seconds`, ledger 15:25:47→15:38:25Z |
| Freeze GW7 | 497.8 | `per_event.7.seconds` |
| Freeze GW8 | 497.0 | `per_event.8.seconds` |
| Freeze GW9 | 487.4 | `per_event.9.seconds` |
| Horizon certification + generation commit | 18 | stage ledger 16:03:07→16:03:25Z |
| **Predictive E2E** (refresh → certification) | **2,281** | harness `started_at` 15:25:31Z → `finished_at` 16:03:32Z; matches an independent 2,281.0 s measurement |
| Production decision, warm run 1 | 1,965.2 | decision log (search 694.3 s inside) |
| Production decision, warm repeat | 2,015.1 | repeat log |
| Production decision, cold cache | 5,075.7 | cold log |
| `verify_generation` / `verify_decision` | 18.273 / 20.700 | measured verifier invocations |
| **Decision-inclusive E2E (warm)** | **4,246.2 ≈ 70.8 min** | defined as predictive E2E 2,281 s + warm decision 1,965.2 s |

**PRIMARY E2E TOTAL (warm): 4,246.2 s**, defined as the predictive cycle plus one warm
decision.  Earlier figures of "4,259 s" did not reconcile with retained values (≈13 s
unattributed) and are superseded by this definition.

**Slowest stage: the production decision (1,965.2 s).  Second slowest: Freeze GW6 (757.9 s).**
Within the predictive cycle alone, the ordering is Freeze GW6 757.9 s then Freeze GW7 497.8 s.

**Per-family Minutes and Monte Carlo durations and shares: NOT CAPTURED** for this run (see
§16) — no share is claimed, and no dominance of one family over another is asserted.

**PE10_OPERATIONAL_RUNTIME_CONCERN: YES.**  A full acceptance cycle (refresh → snapshot →
predictions → certification → decision) is ~4,246 s (~70.8 min) with a warm cache and ~7,357 s
(~122.6 min) with a cold cache, and the decision alone is ~33 min warm / ~85 min cold.  No
optimisation was attempted during acceptance.

## 11. Test evidence

- Manager-state preparation and preflight: the confirmation dry-run (`DRY_RUN_OK`) and the
  dedicated checks recorded in `failed_decision_evidence/`.
- Focused suites on the repaired candidate: **77 passed** (`test_pe9_production_decision.py`,
  `test_pe10_optional_override.py`, `test_pe9_generation_store.py`).
- The repair adds a focused regression that reaches certified-executor artifact assembly and
  asserts the override cutoff guard (the site that previously raised `NameError`), plus an
  ordering regression proving a failing guard refuses before any search.
- Minutes-lineage/authority accounting preserved exactly: 15 tests in the dedicated repair
  file + 1 updated PE-9 authority test = 16; optional-override repair = 5.  These are not
  merged into one number.
- Full authoritative wrapper (`scripts/ci_run_pytest.py`, local): **2 failed (the two accepted
  ids below), 2598 passed, 31 skipped** in 3,985.28 s (1:06:25); `unexpected failures: none`.
- Hosted CI on the acceptance head `78ea8685` (run `36421953185`, job `unit-tests`, SUCCESS):
  **2 failed (the same two accepted ids), 2578 passed, 49 skipped** in 3,260.77 s.
- Hosted CI on the repair candidate `deafe84c` (run `36494119652`, job `unit-tests`, SUCCESS):
  **2 failed (the same two accepted ids), 2580 passed, 49 skipped** in 3,272.11 s.

## 12. Skips and accepted failures

- **Accepted failures (by exact node id; the authoritative wrapper accepts them BY ID and the
  job still succeeds, in BOTH environments):**
  1. `tests/test_causal_bundle_integrity.py::test_13_manager_state_prose_derives_from_actual_state`
  2. `tests/test_causal_bundle_integrity.py::test_13b_zero_ft_prose_is_also_derived`
  A different failure is unexpected even if the total is two.
- **Skip classification (measured, not inferred):** 49 skips hosted versus 31 local — a
  difference of **18** — and the pass counts differ by exactly the same **18** (2598 local vs
  2580 hosted), while both environments fail the same two accepted ids.  The 18 are the tests
  gated on local machine configuration or host prerequisites that a CI checkout does not have
  (`config.json` at its default path, local data directories); they execute locally and skip
  hosted.  No acceptance-critical test (certification, generation store, decision boundary,
  optimizer, comparator, Free Hit, manager worlds, four-GW runner) is skipped in either
  environment.

## 13. Frozen semantics

**UNCHANGED.**  The repair's diff touches no `fpl_brain/` module: minutes (including the
joint kernel), team model, player rates, xPts, Monte Carlo distributions/RNG/seed/draws,
scoring, transfers, hits, FT progression, bank and selling-price semantics, chips and
Wildcard/Free Hit policy, DGW/blank handling, the four-event horizon, the H1 lineup/captain
policy, the optimizer ranking and every PE-9 certification guarantee are byte-identical to
the accepted code.  The decision output shows the frozen surface intact: transfers over
GW6–GW9 (`FOUR_GW_NET_CORE`), lineup and captain restricted to current-GW H1
(`CURRENT_GW_H1`, captain 426, vice 427), no chip forced, no suppression.

## 14. Source safety

`K:\FPL\fpl.db` and `K:\FPL\config.json` hashes are recomputed before and after every
stage and are unchanged; no real FPL account action (transfer, lineup, captain, chip) was
submitted.  All writes went to the isolated runtime
`C:\Users\USER\agent-workspaces\fpl-pe10-runtime`.

## 15. Carry-forward note

The authorization fingerprint is a function of identity fields only (`project`, `work_item`,
`status`, `authorized_by`, `authorized_on`), so it is carried forward unchanged by this
repair; the rejected candidates, the exhausted PE-9 repair ledger, the four Sol code reviews,
both design-review rounds and the superseded service architecture all remain recorded and are
not rewritten by this document.

## 16. Approved timing-evidence exception — `PER_FAMILY_STAGE_TIMINGS_NOT_RETAINED`

**Approved by the Product Owner (PE-10 only).**  The requirement to record separately measured
Minutes and Monte Carlo durations (and any shares derived from them) is waived for this
acceptance.  The waiver covers timing evidence only: no certification, provenance,
manager-state, decision, or safety gate is waived, and every identity and digest in this record
is unaffected by it.

**Not captured:** per-family durations for Minutes and Monte Carlo anywhere in the retained
artifacts — not in `certification_run.json` (`per_event.*` carries only `seconds`,
`simulations`, `certified`, `freeze_return_code`), not in the stage ledger (`detail` is null on
all five rows), not in the four `prediction_freeze_run*.json` artifacts (no timing fields), and
not in the harness log (zero matches for per-family durations).  Each `FREEZE_GWx` stage wraps
minutes (+ four variants), team, rates, xPts and Monte Carlo together, so the families cannot be
separated from the retained evidence.  **No share is estimated, and no per-family dominance is
claimed** — the previously drafted "Monte Carlo remains the dominant predictive stage" sentence
and its "~48 min" figure were carried over from an earlier invocation's report and have been
removed, not restated.

**Retained instead:** the per-event freeze durations, the per-stage ledger, the predictive E2E
boundary, the three decision timings and the two verifier timings (§10), which are sufficient for
the stage-level runtime assessment and the runtime concern stated above.

**Earlier-tree citation in Sol's review.**  Sol's final approval cites tree `969bf0ba…`, which is
the `deafe84c` document commit rather than this candidate's tree `bcf50506`.  The delta between
those two commits is **this document alone** (`git diff deafe84c 7ef63634` → one file,
`docs/prediction-engine/PE-10-V1-ACCEPTANCE.md`): the code, tests and evidence Sol reviewed are
byte-identical.  This correction is accompanied by a fresh Sol High review of this exact
candidate, which resolves the citation literally rather than by inference.
