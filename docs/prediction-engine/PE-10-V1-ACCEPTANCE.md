# PE-10 — End-to-End V1 Acceptance

**Status:** acceptance evidence recorded; V1 is **READY_TO_FREEZE, pending the separate freeze
merge**, which remains a Product Owner decision this document does not take.  One timing-evidence
exception is recorded below with explicit Product Owner approval (§16): per-family Minutes and
Monte Carlo durations were **not captured**, so no shares are claimed anywhere in this record.
The exception waives the timing-evidence requirement only — no certification, provenance,
manager-state, decision or safety gate is waived.
A second, independent defect was found by the review that followed those corrections: a
decision-artifact **retention collision** that let a repeat execution overwrite the evidence an
earlier record still bound.  It is reported, repaired and re-evidenced in §17; the decision
evidence in §§6–8 is the **rebuilt** evidence, and the historical records it replaces are
recorded there as superseded.

## 1. Identity of this acceptance

| Fact | Value |
| --- | --- |
| Acceptance code at the predictive run | `78ea86853ba3f13829bb3be0e211352a75ce9119` (branch `feature/pe10-v1-acceptance`) |
| Predictive code identity (frozen at certification) | `983386e1674cf5d6c74fa08ee2c5fb4488e9c92ce9cca88da8462ebf089b02d5` |
| Earlier repair commit (override cutoff guard + cache location) | `ff18c2f491dfc5767d26c76dbf7a21b3a98ba00c` (tree `2e7d7eac09ae8da3504dad1ec1762de97660df57`, parent `78ea86853ba3f13829bb3be0e211352a75ce9119`) |
| Acceptance-document commit | `deafe84cfed774129f2987424ffa6b5415b48ea5` (tree `969bf0bac87a49175d5a7ba783481f018504d717`) |
| Exception + correction chain (§16) | `7ef636345d8d90cf4ed517a91ff29084a27da24f` (tree `bcf50506…`) → `2b0e7a838e3a79c0aa8ddf7a80c23cbe67355ba6` (tree `86d24ae9…`) → `96578e8dc981613e59a3899d6e45c7d6f72d5500` (tree `d419a9c3…`) |
| Retention repair commit (the code this document records) | `6261a91d945101486b1e926eb60a0c7b1162886b` (tree `c18d262f65418e1ae7292db7223ca103c68a1f23`, parent `96578e8dc981613e59a3899d6e45c7d6f72d5500`) — `fpl_brain/generation_store.py` and the `tests/test_pe9_production_decision.py` regressions; see §17 |
| Atomic-publication repair commit (the code this document records) | `1b79bcdf8688554e3dee9614a9586b92c76fa300` (tree `f9cb3afafb6fda6f60fd9db0095e0d1247febed8`, parent `efd1ce5eeb8705bde3616ab543cc78bbe89210e5`) — `fpl_brain/generation_store.py` (staging under cleanup protection, atomic no-replace publication only) and `tests/test_pe9_production_decision.py` (+2 failure-path regressions) |
| Decision runner identity (P2 rebuild, §§6–8) | `scripts/run_four_gw_decision.py:route_optimizer_v8b_1.0.0` / `sha256:f79e80df1b0811e4f810d22b452a452d5e5a941d954c09656a27e6cc409c679f` |
| Decision runner identity (intermediate rebuild, superseded — §6b) | `scripts/run_four_gw_decision.py:route_optimizer_v8b_1.0.0` / `sha256:5dcee9e8774c4dd83ca9fae860d7c668f46e9279ff00cfb55982e5f9da9c0c05` |
| Decision runner identity (the 2026-09-28 historical executions — §6b) | `scripts/run_four_gw_decision.py:route_optimizer_v8b_1.0.0` / `sha256:f29ed44a2e65159489e8a354f51979ed1df2e32f06c7151bcf97619331f37c62` |
| Source-guard follow-up | `17682cab2bb3bf1b58f0875c9b7c14599624cdfe` (parent `6261a91d945101486b1e926eb60a0c7b1162886b`) — `tests/test_r4b2b_finalist_stability.py`: the two source-literal guards that the wrapper caught are updated to the renamed helper (§11, §17) |
| This candidate (publication repair + P2-rebuilt decision) | the **documentation-only child** of `1b79bcdf` that is the tip of `feature/pe10-v1-acceptance`; the row you are reading is its only delta, so this candidate differs from the P2 repair by this document alone |
| Repair scope (earlier, `ff18c2f4`) | `scripts/run_four_gw_decision.py` (+cutoff guard, cache location), `tests/test_pe9_production_decision.py` regressions; 0 files under `fpl_brain/` |
| Repair scope (retention, `6261a91d`) | `fpl_brain/generation_store.py` (artifact retention only: naming + exclusive publish) and `tests/test_pe9_production_decision.py`; the predictive source set is untouched (§13) |
| Repair scope (publication, `1b79bcdf`) | `fpl_brain/generation_store.py` (staging under cleanup protection; atomic no-replace publication only; refusal where linking is unavailable) and `tests/test_pe9_production_decision.py`; no model, optimizer, scoring or manager-state change (§13) |
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

The decision below is the **P2-rebuilt** primary execution (§8, arm 1), taken through
`generation_store.make_decision` on the atomic-publication code.

| Fact | Value |
| --- | --- |
| Decision id | `sha256:d9dc9be8f51a6746a903e5745d963acf6ebc0a563f8f511833682afe6ec4276b` |
| Generation consumed | `sha256:454c51f289b0011620628ee3c818eda47c3e602f478cb7df46bc750e2cd76172` |
| Result digest | `sha256:1e895a9cb06880fa5c96d20764f2ef26b62323e3cb26cdf7ca121a8e229d301d` |
| Manager context digest | `sha256:833967288ac8e2aebcaf6d6154a523ff88f51d37096b32a1f414c200f5b46425` — recomputed from the consumed state, equal |
| Cutoff guard | `PASS` — `override_captured_at = 2026-09-28T15:16:59Z`, cutoff `2026-09-28T15:25:31Z`, delta 512.0 s |
| Retained decision artifact | `C:\Users\USER\agent-workspaces\fpl-pe10-runtime\pe9_decisions\gw06\1e895a9cb06880fa5c96d20764f2ef26-c67cc9f1b4a13904a89317528a71104d877cbcb9981efe393cea2599caf91082.json`, 5,129,776 bytes, sha256 `c67cc9f1b4a13904a89317528a71104d877cbcb9981efe393cea2599caf91082` — content-addressed, one file per execution, published by atomic no-replace hard link (§17) |
| Published runner artifact | `C:\Users\USER\agent-workspaces\fpl-pe10-runtime\data\exports\pe10_p2_20260929\arm1_primary_warm\gw06\four_gw_decision.json`, sha256 `1ec85fbab92b6daaceb5568bbcbbc01a8ad930e227f8917cf1a0ce0485d9c242` (the runner's own copy; the retained artifact above is the one the record binds) |
| Suppressions | none (`suppression_reasons: []`) |
| Chip result | **NOT APPLICABLE** — no chip was evaluated on this four-GW decision path and none was forced.  Grounded in the retained run summary and artifact: `predictive_certification_only: true`, `wildcard_evaluated: false`, `transfers_or_chips_executed: 0`, and the decision payload's `wildcard_screen: null` with no chip block emitted. |

**Exact generation binding:** the identifier is identical across the verified generation, the
generation row, `engine_decision_records.generation_id`, and the artifact's own provenance
(`provenance.generation_id`, `provenance.readiness_generation_id`, and per-event
`world_info.*.generation_id`).  PASS.

**Manager attribution:** the snapshot-derived canonical state, the state actually consumed,
the retained `consumed_manager_state` (bank 7 / FT 3 / event-start FT `null`), the
`manager_context_sha256` input and the verifier's re-derivation are identical.

### 6b. Superseded decision generations (2026-09-28 records and the 2026-09-29 first rebuild)

Three earlier records were taken against the **same** certified generation.  Their retained
artifacts were destroyed by the retention collision described in §17 — all three pointed at one
result-keyed path and the later executions overwrote it — so they are recorded here as
**superseded**: not evidence of this acceptance, and not claimed to verify.

A second generation of records — the **first rebuild** (four arms, 2026-09-29 early UTC) — is also
superseded by this acceptance: the atomic-publication repair changed the runner identity again, so
the acceptance binds the **P2 rebuild** (§8).  Those four records are **not** damaged — immutable
per-execution retention means every one of them still verifies against its own artifact, which is
itself the strongest evidence the retention defect is fixed — but they are not this acceptance's
evidence and are not relied on.

| Historical record | Executed | State now |
| --- | --- | --- |
| `sha256:bf93b936…d41404c1` (warm run 1 — the record the earlier draft of this document bound) | 2026-09-28T18:42:25Z | **SUPERSEDED — `DECISION_RECORD_INVALID`**: the retained file holds the last writer's bytes (`8d6896a4…`) and this record's own artifact bytes (`cff53ba5…`) are retained nowhere |
| `sha256:9de2a7f2…de67b86e` (warm repeat) | 2026-09-28T19:18:11Z | **SUPERSEDED — `DECISION_RECORD_INVALID`**: same cause (`7aeca6da…` retained nowhere) |
| `sha256:f1860625…10eb7403` (cold cache) | 2026-09-28T20:46:57Z | **SUPERSEDED**; it happens to be the last writer, so its artifact binding still verifies — it is nevertheless not this acceptance's evidence, because it predates the repair and its siblings are unrecoverable |
| first rebuild, 4 records (`84b3b60d…`, `5c6a8603…`, `ffa76b7d…`, `d7dcbdb8…`) | 2026-09-29T04:29–07:04Z | **SUPERSEDED, and all four still verify** against their own distinct artifacts (`7d2679ad…`, `f9b10c20…`, `4268db4c…`, `40d57759…`) — superseded because the publication repair changed the runner identity, not because anything was lost |

The records are append-only and were deliberately **not** rewritten, re-pointed or deleted: they
stand exactly as written, and the verifier's refusal is the honest state of their evidence.
Re-running cannot restore the 2026-09-28 bytes — only new records with immutable paths can, which
is what the P2-rebuilt evidence in §8 is.

## 7. Decision verification

`verify_decision sha256:d9dc9be8f51a6746a903e5745d963acf6ebc0a563f8f511833682afe6ec4276b` → `verified = true`, with `generation_verified`,
`record_digest_recomputed`, `manager_packet_digest_verified`, `manager_context_digest_verified`,
`request_digest_verified`, `result_digest_verified`, `runner_identity_verified`,
`runner_code_identity_bound`, `decision_artifact = VERIFIED` at
`c67cc9f1b4a13904a89317528a71104d877cbcb9981efe393cea2599caf91082`.

Every P2 record was re-verified **after the cold execution** (§8), not merely when it was written:
all three report `verified = true`, each bound to its **own** artifact file.  The earlier
generations of records behave exactly as §6b states — the first rebuild's four records still
verify, the 2026-09-28 pair still refuses with its original recorded digests.

Replay boundary, stated truthfully: `DECISION_REPLAY_NOT_PERFORMED` — the generation and the
manager context re-derive from retained evidence; the decision is **not** claimed to have been
re-played.

## 8. Reproducibility and cache non-authority

| Execution | window (UTC) | seconds | decision id | retained artifact byte digest | cache |
| --- | --- | ---: | --- | --- | --- |
| arm 1 — primary | 2026-09-29T11:05:14Z → 2026-09-29T11:38:57Z | 2,023 | `sha256:d9dc9be8f51a6746a903e5745d963acf6ebc0a563f8f511833682afe6ec4276b` | `c67cc9f1b4a13904a89317528a71104d877cbcb9981efe393cea2599caf91082` | warm |
| arm 2 — identical repeat | 2026-09-29T11:38:57Z → 2026-09-29T12:11:16Z | 1,939 | `sha256:3bbf3e15cdc152030a4d5c27ae2ee9a7c30cf84f71eec6b80d86022eab74bce0` | `ca8f8a0ba11a13f346d4cca3ba6c824e984a82717dd67f6175249d1dfdcf4ef7` | warm |
| arm 3 — cold cache | 2026-09-29T12:11:16Z → 2026-09-29T13:36:45Z | 5,129 | `sha256:1906c576079d48f4f379c174c83c1bd879d0c5b568a5de56185149145ae5af45` | `676d1b1fe9167ed24922b2c16f5a87e939da1771d2b2b8c29f9fd73747887353` | **cold** (cache cleared) |
**Three** independent executions of the same certified generation, manager packet, profile and
request produced the **same** `result_sha256` (`sha256:1e895a9cb06880fa5c96d20764f2ef26b62323e3cb26cdf7ca121a8e229d301d`) and the same decision
content, while each kept its **own** retained artifact: three distinct paths, three distinct byte
digests (table above), published by atomic no-replace hard link.  The **identical repeat** is the
arm the original defect destroyed: under the pre-repair store it overwrote arm 1's artifact.

The P2 result digest differs from the first rebuild's `sha256:e04324c834b806f72af406e7018d277e09252796f7f56f41dc54d8990a49db42` in exactly
**one** projection field — `runner_code_identity` — measured field by field (§13); the first
rebuild's differed from the 2026-09-28 historical `sha256:1e03814633816eb065f618e478d6dc6a4eac61122f29ce59bb9e9515af9466fc` for the same reason.  Every
semantic field is identical across all three generations of decisions.

Cache non-authority, re-established: arm 3 ran with the cache **cleared** (8 files removed against
a byte-verified manifest), re-derived every world, and produced the **same** result digest and the
same decision content at 5,129 s against arms 1–2 at 2,023 s and 1,939 s; the cache was then
restored byte-verified.  The cache changes runtime only, and every arm's artifact is retained
separately.

## 9. Fail-closed refusals (representative)

Exercised against a **copy** of the accepted runtime; the accepted generation was untouched.
The last two rows are the retention repairs' own regression cases (§11), exercised in-process on a
synthetic certified store.

| Case | Outcome |
| --- | --- |
| unknown generation | REFUSED -> UNKNOWN_GENERATION_ID |
| manager-state mismatch (asserted bank differs) | REFUSED -> MANAGER_STATE_MISMATCH |
| nested predictive injection (bundles in the packet) | REFUSED -> PRODUCTION_DESCRIPTOR_ONLY |
| nested cache handle injection (cache_dir in the packet) | REFUSED -> PRODUCTION_DESCRIPTOR_ONLY |
| caller executor injection is not an argument | REFUSED -> PRODUCTION_DESCRIPTOR_ONLY |
| tampering with a certified generation row | REFUSED by the database -> PE9_APPEND_ONLY: a certified generation is immutable |
| an occupied decision-artifact path whose bytes differ (retention collision) | REFUSED -> `DecisionRecordInvalid`; the occupied bytes are left exactly as found and no record is appended |
| a retained decision artifact edited under its record | REFUSED -> `DECISION_RECORD_INVALID` (digest binding), for that record only — its siblings still verify |
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
| Production decision, P2 arm 1 (primary, warm) | 2,023 | `pe10_p2_20260929/arms_summary.tsv` 2026-09-29T11:05:14Z → 2026-09-29T11:38:57Z |
| Production decision, P2 arm 2 (identical repeat, warm) | 1,939 | 2026-09-29T11:38:57Z → 2026-09-29T12:11:16Z |
| Production decision, P2 arm 3 (cold, cache cleared) | 5,129 | 2026-09-29T12:11:16Z → 2026-09-29T13:36:45Z |
| (superseded — §6b) first rebuild, 4 arms | 2,233 / 2,284 / 4,983 / 2,000 | 2026-09-29 pre-P2 code |
| (historical — §6b) warm 1 / repeat / cold | 1,965.2 / 2,015.1 / 5,075.7 | the pre-repair executions of 2026-09-28 |
| `verify_generation` / `verify_decision` | 18.273 / 20.700 | measured verifier invocations |
| **Decision-inclusive E2E (warm)** | **4,246.2 ≈ 70.8 min** | defined as predictive E2E 2,281 s + warm decision 1,965.2 s |

**PRIMARY E2E TOTAL (warm): 4,246.2 s**, defined as the predictive cycle (2,281 s) plus the warm
decision of that cycle (1,965.2 s).  Earlier figures of "4,259 s" did not reconcile with retained
values (≈13 s unattributed) and are superseded by this definition.  The **P2 rebuild's**
decision-inclusive observed figure is 2,281 + 2,023 = **4,304 s** (§8 arm 1); the
predictive half was not re-run, so 4,304 s mixes a historical predictive span with a P2 decision
and is quoted only as an observed figure, not as the acceptance total.

**Slowest stage: the production decision — P2 arm 3 (cold) at 5,129 s, then arm 1 (warm) at
2,023 s and arm 2 at 1,939 s (the superseded earlier rebuilds at 1,965.2–2,284 s warm).
Second slowest warm stage: Freeze GW6 (757.9 s).**
Within the predictive cycle alone, the ordering is Freeze GW6 757.9 s then Freeze GW7 497.8 s.

**Per-family Minutes and Monte Carlo durations and shares: NOT CAPTURED** for this run (see
§16) — no share is claimed, and no dominance of one family over another is asserted.

**PE10_OPERATIONAL_RUNTIME_CONCERN: YES.**  A full acceptance cycle (refresh → snapshot →
predictions → certification → decision) is ~4,246 s (~70.8 min) with a warm cache and ~7,357 s
(~122.6 min) with a cold cache, and the decision alone is ~33 min warm / ~85 min cold.  The
P2 decision arms measure the same shape: 2,023 s and 1,939 s warm
(identical repeat) against 5,129 s cold, with the same result digest across all three.
No optimisation was attempted during acceptance, and none is claimed by these repairs — retention
and publication add a single staged 5 MB write plus one link per decision (well under a second
against a ~2,000 s decision), so the arm-to-arm differences above are run-to-run variance, not
repair cost.

## 11. Test evidence

- Manager-state preparation and preflight: the confirmation dry-run (`DRY_RUN_OK`) and the
  dedicated checks recorded in `failed_decision_evidence/`.
- Focused suites on the earlier repair (cutoff guard + cache location): **77 passed** (`test_pe9_production_decision.py`,
  `test_pe10_optional_override.py`, `test_pe9_generation_store.py`).
- The repair adds a focused regression that reaches certified-executor artifact assembly and
  asserts the override cutoff guard (the site that previously raised `NameError`), plus an
  ordering regression proving a failing guard refuses before any search.
- The retention repair adds three regressions in `test_pe9_production_decision.py`, and each was
  **shown to fail against the pre-repair store** before being accepted:
  1. a repeated decision keeps **both** executions' artifacts and **both** records still verify;
  2. tampering with either artifact fails verification for **that** record only, and restoring the
     bytes restores verification;
  3. an occupied retention path is **refused, not replaced** (the pre-repair store returned
     `DID NOT RAISE` here), the foreign bytes are left exactly as found, and no second record is
     appended.
  4. a **staging failure** (forced `ENOSPC` on the staged temporary) leaves **no final artifact, no
     temporary and no decision record** — the pre-P2 implementation returned `DID NOT RAISE` here,
     because it never fsynced the staged file at all;
  5. a filesystem **without hard links** refuses the decision (`DecisionRecordInvalid`) instead of
     streaming the payload into the final path — the pre-P2 implementation left the final artifact
     in place here.
- Focused suites run on the P2 store: **117 passed, 1 skipped** (`test_pe9_production_decision.py`,
  `test_r4b2b_finalist_stability.py`, `test_pe9_generation_store.py`, `test_pe9_verify_commands.py`)
  and **127 passed, 1 skipped** (`test_pe9_certification_integration.py`,
  `test_pe9_manager_packet_cli.py`, `test_pe10_optional_override.py`,
  `test_r4b1_decision_correctness.py`, `test_free_hit_request_adapter.py`).
- Focused suites on the first rebuild: **209 passed, 1 skipped** across
  `test_pe9_production_decision.py`, `test_pe9_generation_store.py`, `test_pe9_verify_commands.py`,
  `test_pe9_certification_integration.py`, `test_pe9_manager_packet_cli.py`,
  `test_pe10_optional_override.py`, `test_r4b1_decision_correctness.py` and
  `test_free_hit_request_adapter.py`.
- Minutes-lineage/authority accounting preserved exactly: 15 tests in the dedicated repair
  file + 1 updated PE-9 authority test = 16; optional-override repair = 5.  These are not
  merged into one number.
- Full authoritative wrapper (`scripts/ci_run_pytest.py`, local): **2 failed (the two accepted
  ids below), 2598 passed, 31 skipped** in 3,985.28 s (1:06:25); `unexpected failures: none`.
- Wrapper on the **retention repair commit** `6261a91d`: **4 failed, 2599 passed, 31 skipped** —
  the two accepted ids plus **two unexpected** (`test_r4b2b_finalist_stability.py`'s source-literal
  guards, which assert the store's publish line as text and stopped matching when the helper was
  renamed).  Reported, not hidden: fixed in the follow-up `17682cab` (§1, §17), whose intent is
  unchanged — the guards still assert that suppression is applied before the artifact is assembled
  and that the store publishes the artifact reference from its own helper.
- Wrapper on the guarded repair `17682cab` (the pre-P2 code state): **2 failed (the two accepted
  ids below), 2601 passed, 31 skipped** in 4,090.57 s (1:08:10); `unexpected failures: none`.  The
  extra three passes over the earlier local run are the three retention regressions.
- Wrapper on the **atomic-publication repair** `1b79bcdf` (the code+test state of this candidate):
  **2 failed (the two accepted ids below), 2603 passed, 31 skipped** in 4,019.16 s (1:06:59);
  `unexpected failures: none`.  The extra two passes are the two failure-path regressions.
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

## 13. Frozen semantics — audited by behaviour

This repair is the first PE-10 commit to change a `fpl_brain/` file, so the frozen-semantics claim
is established **by what the engine actually does**, not by a file list.

**1. The predictive identity is untouched, and the certified generation still verifies.**
`fpl_brain/generation_store.py` is not part of the certified source set
(`analytics.SOURCE_SNAPSHOT_FILES`, 18 named files covering the models, the scoring rules, the
calibration, the DEFCON term, the point-in-time boundary and the certification entry points), so
the predictive code identity is unchanged.  Checked **before** any decision was re-run, with the
repaired code in place — before the first rebuild and again before the P2 rebuild:

```
{"verified": true, "code_identity_reproduced_from_runs": true, "snapshot_identity": "VERIFIED", "planning_context_reproduced_from_snapshot": true, "bundle_identities_reproduced": true, "dependency_closure_reproduced": true, "manifest_digest_matches": true, "versions_valid": true}
```

`verify_generation sha256:454c51f289b0011620628ee3c818eda47c3e602f478cb7df46bc750e2cd76172` → `verified = true`, including
`code_identity_reproduced_from_runs` and `snapshot_identity = VERIFIED`.

**2. The rebuilt decision is the same decision, measured field by field.**  The decision-result
identity projection of the rebuilt arm 1 was compared with the historical artifact's, field by
field (route table, decision block, `decision_confidence`, `suppression_reasons`,
`fixture_horizon`, `decision_events`, `planning_event`, `planning_cutoff`, `generation_id`,
`manager_context_sha256`, `runner_identity`, `runner_code_identity`, `schema`):

| Projection field | Historical | Rebuilt |
| --- | --- | --- |
| `runner_code_identity` | `sha256:f29ed44a2e65159489e8a354f51979ed1df2e32f06c7151bcf97619331f37c62` (2026-09-28) | `sha256:5dcee9e8774c4dd83ca9fae860d7c668f46e9279ff00cfb55982e5f9da9c0c05` (first rebuild) → `sha256:f79e80df1b0811e4f810d22b452a452d5e5a941d954c09656a27e6cc409c679f` (P2 rebuild) |
| every other field | — | **identical** |

So the rebuilt result digest changes for one reason only: the decision-result identity binds the
code that took the decision, and that code changed (the decision runner identity fingerprints all
`fpl_brain/*.py` plus the declared runner module).  The change is recorded in every new record,
not hidden.

**3. The decision output is unchanged.**  4-GW net ranking `route_001` 186.879 > `route_004`
186.511 > `route_003` 186.489 > `route_002` 186.433 > `route_005` 186.311 > `route_000` 177.860;
0 hits on every route; terminal FT/bank identical; transfers over GW6–GW9 (`FOUR_GW_NET_CORE`);
lineup and captain restricted to current-GW H1 (`CURRENT_GW_H1`, captain 426, vice 427); no chip
forced; no suppression (`suppression_reasons: []`).  Minutes (including the joint kernel), team
model, player rates, xPts, Monte Carlo distributions/RNG/seed/draws, scoring, transfers, hits, FT
progression, bank and selling-price semantics, chips and Wildcard/Free Hit policy, DGW/blank
handling, the four-event horizon, the H1 lineup/captain policy, the optimizer ranking and every
PE-9 certification guarantee are untouched by the repair: its diff is one store module (the
artifact-retention helper and its publish site) and tests.

**4. What did change, stated plainly.**  The decision runner code identity moved twice — once for
the retention repair (`sha256:f29ed44a2e65159489e8a354f51979ed1df2e32f06c7151bcf97619331f37c62` → `sha256:5dcee9e8774c4dd83ca9fae860d7c668f46e9279ff00cfb55982e5f9da9c0c05`) and once for the
publication repair (`sha256:5dcee9e8774c4dd83ca9fae860d7c668f46e9279ff00cfb55982e5f9da9c0c05` → `sha256:f79e80df1b0811e4f810d22b452a452d5e5a941d954c09656a27e6cc409c679f`) — and, with it, each rebuild's result
digests.  Nothing else in the projection moved, measured against both preceding generations.

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
boundary, the per-arm decision timings of both decision rebuilds and the two verifier timings
(§10), which are sufficient for the stage-level runtime assessment and the runtime concern stated
above.

**Earlier-tree citation in Sol's review.**  Sol's final approval cites tree `969bf0ba…`, which is
the `deafe84c` document commit rather than the tree of the candidate it reviewed (`bcf50506…`,
commit `7ef63634`).  The delta between
those two commits is **this document alone** (`git diff deafe84c 7ef63634` → one file,
`docs/prediction-engine/PE-10-V1-ACCEPTANCE.md`): the code, tests and evidence Sol reviewed are
byte-identical.  This correction is accompanied by a fresh Sol High review of this exact
candidate, which resolves the citation literally rather than by inference.

**Correction chain.**  Each step is separated so that its delta can be checked rather than
taken on trust:
`deafe84c` (tree `969bf0ba…`, the document commit Sol reviewed) → `7ef63634`
(tree `bcf50506…`, the hosted-CI classification correction; `git diff deafe84c 7ef63634` =
this document alone) → `2b0e7a8` (tree `86d24ae9…`, the §16 exception and the §6/§8/§10
corrections; `git diff 7ef63634 2b0e7a8` = this document alone) → the **documentation-only
child** that completes the §1 candidate row, whose only delta is that one row.  That child was
the tip of `feature/pe10-v1-acceptance` and the exact candidate Sol High reviewed, and its review
returned `FIX_REQUIRED / P1` on the retention defect of §17 — so the chain continues:
`6261a91d945101486b1e926eb60a0c7b1162886b` (tree `c18d262f…`, the **retention repair**) →
`17682cab2bb3bf1b58f0875c9b7c14599624cdfe` (the **source-guard follow-up**: the two
`test_r4b2b_finalist_stability.py` guards that assert the store's publish line as source TEXT are
updated to the renamed helper — disclosed in §11 and §17, acknowledged by the Product Owner) →
`efd1ce5eeb8705bde3616ab543cc78bbe89210e5` (the **retention-evidence document**, the candidate
Sol High reviewed at `FIX_REQUIRED / P2`) →
`1b79bcdf8688554e3dee9614a9586b92c76fa300` (tree `f9cb3afa…`, the **atomic-publication repair**
Sol prescribed: staging under cleanup protection, publish only by atomic no-replace hard link,
refuse where linking is unavailable) → the documentation-only child that completes this document,
which is the final candidate.  No commit in this chain touches model, optimizer, scoring or
manager-state code, and none touches evidence.

## 17. Decision-artifact retention — `DECISION_ARTIFACT_RETENTION_NOT_IMMUTABLE` (found, repaired, re-evidenced)

**Found by the Product-Owner-mandated Sol High review of the exception candidate `96578e8d`
(`FIX_REQUIRED / P1`), then reproduced read-only before any repair was written.**

**The defect.**  `fpl_brain/generation_store.py` retained a decision artifact at a path keyed by the
**result** digest, while the append-only record bound the digest of the **bytes that execution
wrote**.  Because the decision is deterministic, every repeat of the same generation + manager packet
+ profile produced the same result digest and therefore the **same path**, while the artifact bytes
differ (volatile telemetry); `_write_decision_artifact` then replaced the file.  The second and third
executions of 2026-09-28 overwrote the first, so `verify_decision` could no longer reproduce the
earlier records' evidence and the primary record's bytes were gone.  The database's append-only
trigger protected the **rows**, not the files those rows point at.

**The repair (`6261a91d`, §1).**

- the retention name is **content-addressed by the artifact bytes themselves**
  (`<result digest prefix>-<sha256 of the retained bytes>.json`), computed from the same buffer that
  is written, so the name cannot describe content other than what was retained;
- the payload is **staged under cleanup protection** — written, flushed and fsynced to a private
  temporary inside the `try/finally` that always discards it — so no staging failure (a full disk,
  a permission error) can leave a temporary behind, and a staging failure is reported as a
  refusal, not as a bare `OSError`;
- publication is an **atomic no-replace hard link**: it either creates the final name or fails
  without touching it, and an occupied path is **confirmed or refused, never replaced** —
  identical bytes are reused (so an identical repeated decision stays idempotent) and different
  bytes raise `DecisionRecordInvalid`;
- a filesystem that cannot hard-link is a **refusal, not a fallback write**: the decision is
  refused (`DecisionRecordInvalid`) with the final path untouched, because a decision artifact is
  never streamed into its retained name piece by piece.  There is no non-atomic publication path
  left (`_create_exclusively` was deleted).

Sol's review of the first repair (`FIX_REQUIRED / P2`) found that its fallback streamed into the
final path and that the staging sat outside the cleanup block; both are fixed here, and the two
failure-path regressions were shown to fail against that previous implementation before being
accepted (§11).

**Regressions (§11)** prove all three behaviours, and all three were shown to fail against the
pre-repair store.

**The rename was caught by the wrapper, not by me.**  The first authoritative wrapper run on the
repair commit reported **two unexpected failures** — `test_r4b2b_finalist_stability.py`'s two
source-literal guards, which assert the store's publish line as TEXT and therefore stopped
matching when the helper was renamed.  They guard a real property (suppression is applied before
the artifact is assembled, and the store publishes the artifact reference from its own helper
inside the canonical entrypoint), and that property is unchanged; only the literal moved.  Fixed
in the follow-up commit `17682cab` (§1), after which the wrapper reports only the two
accepted node ids.

**How the historical records are treated.**  The three 2026-09-28 records are **not** rewritten,
re-pointed or deleted — append-only means append-only.  Two of them can no longer verify (§6b) and
their bytes are retained nowhere; no re-run can bring those bytes back.  What the repair restores is
the **future**: every execution now keeps its own artifact.

**The P2-rebuilt evidence (§8).**  Three executions on the unchanged certified generation —
primary warm, identical repeat warm, and cold after the cache was cleared — one result digest
`sha256:1e895a9cb06880fa5c96d20764f2ef26b62323e3cb26cdf7ca121a8e229d301d`, three distinct retained artifact paths with three distinct
byte digests, and every record re-verified **after the cold execution**:

| Execution | Decision record | Result digest | Retained artifact (distinct per execution) | Verified after the cold run |
| --- | --- | --- | --- | --- |
| arm 1 — primary, warm | `sha256:d9dc9be8f51a6746a903e5745d963acf6ebc0a563f8f511833682afe6ec4276b` | `sha256:1e895a9cb06880fa5c96d20764f2ef26b62323e3cb26cdf7ca121a8e229d301d` | `…-c67cc9f1b4a13904….json` | `verified = true` |
| arm 2 — identical repeat, warm | `sha256:3bbf3e15cdc152030a4d5c27ae2ee9a7c30cf84f71eec6b80d86022eab74bce0` | `sha256:1e895a9cb06880fa5c96d20764f2ef26b62323e3cb26cdf7ca121a8e229d301d` | `…-ca8f8a0ba11a13f3….json` | `verified = true` |
| arm 3 — cold, cache cleared | `sha256:1906c576079d48f4f379c174c83c1bd879d0c5b568a5de56185149145ae5af45` | `sha256:1e895a9cb06880fa5c96d20764f2ef26b62323e3cb26cdf7ca121a8e229d301d` | `…-676d1b1fe9167ed2….json` | `verified = true` |

The first rebuild's four records (runner identity `…{RET_CODE[7:15]}…`) remain retained and still
verify against their own artifacts (§6b); they are the intermediate evidence between the retention
repair and this publication repair.

Full artifact paths, byte sizes and digests, the arm timings and the retained world cache manifest
are recorded under `<runtime>/data/exports/pe10_p2_20260929/` (`arms_evidence.json`,
`arms_summary.tsv`, `decision_evidence.json`, `world_cache_warm_manifest.json`,
`frozen_semantics_audit.json`), with the first rebuild's evidence retained under
`<runtime>/data/exports/pe10_repair_20260929/`.

**What this changes about the acceptance.**  The acceptance's decision evidence is now the rebuilt
set; the pre-repair records stand as superseded history (§6b) and are not relied on.  The timing
exception of §16 continues to apply, unchanged: nothing here waives a certification, provenance,
manager-state, decision or safety gate, and every identity in §§2–5 is untouched.
