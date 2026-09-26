# PE-9 Certification Integration — Authority Amendment 2: content-addressed generation store

**Status:** AMENDMENT — governing authority for the PE-9 generation remediation. PE-9 remains **OPEN**.
**Supersedes**, before any implementation, the isolated-service design approved in design review 4 of
`PE-9-REMEDIATION-DESIGN.md`. That document and its review history remain historical evidence.
**Authorised by:** Product Owner, 2026-09-26, on the evidence of the four rejected candidates, the four
Sol code reviews, and independent architecture review (Qwen; GLM 5.3 Max).
**Remediation base:** `feature/pe9-certification-remediation` at
`a4e744d0acd187d1c9eee858f01c6143c9861c2b` (tree `9359531a92f63c8e35a94dda373c85c4c54d8ba5`), whose
ancestry includes the rejected candidate `41eef48d7cf8d68ffaeeb54fdb9412cd0c04497d` (tree
`5f98bb9eed3c0273b18aa4c5de962536f6e86a2a`) and the PE-9 authority `d4a5b132…`.

This is **not** "repair 4" of the exhausted PE-9 loop. The historical terminal state
(`HUMAN_REQUIRED` / `REPAIR_ATTEMPT_LIMIT_EXCEEDED`, `repair_attempt` 4) is preserved exactly and is
never reopened, reset or rewritten. The remediation runs as a **fresh work item** with fresh state,
authorization, invocation history and repair counter.

## 1. Preserved history

| Stage | SHA | Sol verdict | CI |
| --- | --- | --- | --- |
| authority | `d4a5b132d17bb71369e93eeb5d394fb70e19382e` | — | success |
| original candidate | `9c8f333798bfd94ecc1f066e66bfbcb22b803ad0` | FIX_REQUIRED / P1 | `36081104050` success |
| repair 1 | `14172b1e42888b736275a61c42ee1ffb0e0b4057` | FIX_REQUIRED / P1 | `36161965191` success |
| repair 2 | `7fc8326ac166e386c5392ee7bdd95a5d553e770b` | FIX_REQUIRED / P1 | `36179945670` success |
| repair 3 | `41eef48d7cf8d68ffaeeb54fdb9412cd0c04497d` | FIX_REQUIRED / P1 | `36201728611` success |

The four code-review findings, verbatim, are preserved in the phase evidence
(`pe9_failure_history.md`): caller-minted bundle identity; raw mappings and a caller-writable matrix
stamp; a mutable `ValidatedCertificationArtifact` whose base-class mutators were still effective plus a
writable registry and a callable issuer; and finally closure-recoverable mint tokens and issuers
(`__init__.__closure__`, `build_event_worlds.__closure__`).

The service-design review history is preserved: design reviews 1–3 `FIX_REQUIRED` / P1, review 4
`APPROVED` / NONE for an isolated certified service process. **Supersession reason:** independent
architecture review (Qwen; GLM 5.3 Max) showed the service boundary is unnecessary for this threat
model — it adds an operational process, a fleet of new interfaces and a large migration, while the
threats PE-9 actually defends against (wrong run, wrong event, wrong cutoff, wrong version, stale
cache, missing upstream, replay leakage, mutated snapshots, future bypass) are all answered by
**persisted, content-addressed evidence** in the existing store. Simpler and stronger wins.

## 2. Threat model (frozen before coding)

PE-9 **protects against**: wrong run; wrong event; wrong cutoff; wrong model version; incomplete
dependency closure; stale cache; missing upstream data; silent zero/default degradation; replay/test
leakage into production; mutated snapshots; future developers or agents bypassing certification;
decisions whose predictive provenance cannot be reproduced.

PE-9 **does not claim to defend against**: arbitrary hostile code already executing inside the trusted
Python process; monkey-patching trusted production code; patching `hashlib`; replacing the DB driver;
arbitrary same-process code execution; malicious direct writes to the authoritative store.

**Do not reintroduce security theatre that tries to make in-process Python objects "unforgeable".**
No closures-as-secrets, no capability objects, no registries-as-authority, no `isinstance` trust.

**Governing invariant:** production decision logic may consume predictive data only through a
**persisted certified generation whose exact provenance can be independently re-derived and verified
from authoritative persisted evidence**.

## 3. Final architecture

**Content-addressed generation store + descriptor-only production API + persisted immutable decision
records + internal non-authoritative cache + structurally separate replay/test path.**

There is **no** mandatory certified-service process. Do not implement
`python -m fpl_brain.certified_service` as the PE-9 trust boundary. The authority is persisted evidence
in the existing authoritative store plus immutable pinned snapshots — not Python object identity and
not a service socket.

## 4. Reuse the existing certification system (do not build a second one)

Extend, wherever applicable: `fpl_brain/certified_bundle.py` (canonical bundle identity,
`validate_certified_bundle`, `certified_bundle_from_explicit_ids`, `certify_horizon_bundles`,
`declared_required_versions`, the bypass register); `fpl_brain/execution_snapshot.py`
(`capture_execution_snapshot`, `open_snapshot`, `assert_snapshot_unchanged`, `source_db_identity`,
`snapshot_consistency_window`, `require_live_cutoff_matches_snapshot`); the causality machinery;
explicit run-ID validation; `planning_context_hash`; the existing SHA/content-addressed world cache
(`route_optimizer.world_cache_key` / `cache_dir`); the execution controller, leases and stages
(`fpl_brain/execution.py`, `open_certification_source`, `decide_search_permission`); PE-8 evidence
machinery. Do not duplicate these invariants in a second implementation.

## 5. Remove the failed authority mechanisms

Remove as **authority**: `ValidatedCertificationArtifact` authority (`certified_bundle.py`, the
`Mapping` subclass and its closure mint token); `isinstance`-based trust; closure-captured tokens and
issuers; matrix stamps (`_p2_certified_bundle_identity` and successors); matrix capability registries
(`_ISSUED_WORLD_MATRICES`, `IssuedWorldMatrix`); module-global authorization registries;
`require_certified_prebuilt_matrix` and `_issue_world_matrix` / `_issue`; caller-carried certification
digests or authority objects; certified caller-prebuilt matrices on production paths. Update the
bypass register to describe only paths that still exist.

Plain dataclasses and types may remain for ordinary data ergonomics **if useful**, but they confer
**zero** authority.

## 6. Schema (smallest that fits; extensions of the existing store)

Bump the schema version (currently `database.SCHEMA_VERSION = 17`) with a migration following existing
conventions, and add three tables — no more than the architecture needs:

- **`generation`** — append-only certified generation: `generation_id` (PK), the canonical semantic
  manifest (exact bytes hashed), `planning_event`, `horizon_kind`, `cutoff`, snapshot path/identity,
  `source_db_identity`, `execution_run_uuid`, `created_at`.
  `generation_id = sha256(canonical_semantic_manifest)`; the **manifest identity excludes volatile
  fields** such as creation timestamps; canonicalization rules are frozen and documented; no
  floating-point values in identity-bearing content (stable strings/integers/fixed forms only);
  recomputing the canonical manifest digest must reproduce `generation_id`.
- **`current_generation`** — the small mutable selector: `(planning_event, horizon_kind) ->
  generation_id`. A convenience selector, **not** evidence.
- **`engine_decision_records`** — append-only production decision attribution: `decision_id`,
  `generation_id`, manager packet digest, request digest, result digest, runner/code identity,
  evidence references, decision artifact reference, `created_at`.

## 7. Append-only discipline

Once written, these must not silently mutate: completed predictive runs, certified generation
manifests, production decision records, historical certification evidence. Where practical in SQLite,
use mechanical `UPDATE`/`DELETE` refusal (triggers) on the new PE-9 records. This is protection against
**accidental** mutation, not a hostile-DB security claim. Corrections create new evidence/generations;
they never rewrite history.

## 8. Certification lifecycle (extend the existing controller)

```
acquire existing writer lease
-> capture/reuse immutable execution snapshot
-> resolve exact required predictive runs
-> validate each bundle with the existing canonical validators
-> validate required model versions (authoritative in-library source only)
-> validate dependency closure
-> validate planning_context_hash
-> validate code snapshot
-> validate data snapshot identity
-> consult/link PE-8 evidence
-> apply the four-event horizon gate
-> construct the canonical semantic manifest
-> compute generation_id
-> ONE TRANSACTION: insert generation idempotently + update current_generation
```

A generation row may exist **only** when certification has passed: do not create a half-certified
generation and later toggle a mutable `CERTIFIED` flag. For V1, *existence of a valid generation row +
digest == generation_id* means certified. A crash before commit leaves no generation and the old
pointer. Re-certifying identical semantic evidence resolves to the **same** `generation_id`.

## 9. Close all seven original PE-9 gaps

1. **`required_versions`** must come from one declared authoritative in-library source
   (`certified_bundle.declared_required_versions`) and be supplied to certification validation; never
   rely on callers. In particular `four_gw_decision.py`'s
   `certification.get("required_model_versions")` path must stop being caller-authoritative.
2. **Mandatory production boundary / caller bundles.** Production predictive consumption goes through
   the generation path; remove production acceptance of caller-supplied bundle mappings from
   `build_event_worlds` (or make such parameters explicitly non-production and unreachable from the
   production API). Audit and fix the known bypass scripts: `scripts/build_route_comparison.py`,
   `scripts/final_operational_refresh_gw04.py`, `scripts/gw4_current_final_board.py`,
   `scripts/live_fire_gw04.py`. No production-adjacent script may rediscover runs latest-per-family and
   silently construct an alternate predictive world.
3. **Second readiness path.** Remove `event_support_from_db` (or equivalent uncertified DB
   rediscovery) from the production readiness/decision path; readiness resolves the certified
   generation/pointer. Non-production diagnostic use may remain, explicitly labelled.
4. **PE-8 evidence** participates in the generation manifest/certification evidence. Preserve the
   frozen semantics: `EVIDENCE_LIMITED` and `CALIBRATION_NOT_APPLICABLE` are **not** automatically
   failures. No PE-8 model promotion.
5. **Missing Monte Carlo upstream** must refuse. Remove `get_projection_run(...) or {}` or equivalent
   silent degradation (note: `monte_carlo.require_input_run` already refuses; the remaining silent
   fallbacks elsewhere in the MC path must go).
6. **Missing projection / zero-fill.** Missing predictive evidence never silently becomes zero.
   Preserve the distinction: legitimate blank event ≠ missing projection. Partial DGW evidence must not
   silently understate a player.
7. **Data snapshot identity** must be *validated*, not merely recorded: verify the actual
   snapshot/data identity the generation used.

## 10. Descriptor-only production API

One canonical production entrypoint, conceptually:

```python
make_decision(manager_packet, planning_event: int, generation_id: str | None = None) -> DecisionResult | Refusal
```

`generation_id=None` resolves the `current_generation` pointer; an explicit `generation_id` loads that
exact certified historical generation. **`generation_id` is a selector, not a capability.**

The production caller **may** provide: manager state / manager packet, planning event, generation
selector, ordinary decision parameters/profile where already supported, and a request tracing ID.

The production caller **must not** provide: matrices, bundles, predictive run mappings, dependency
mappings, certification artifact objects, validation objects, caller-computed manifest digests, cache
handles, registry entries, or authority tokens.

## 11. Production decision lifecycle

```
resolve generation id
-> load canonical generation manifest
-> recompute digest and require digest == generation_id
-> verify event/horizon compatibility
-> verify exact cutoff
-> verify pinned snapshot identity
-> open snapshot using existing read-only/snapshot validation
-> re-run existing per-event bundle validation against the pinned runs
-> verify required versions and dependencies
-> build worlds internally
-> execute the frozen decision logic
-> append engine_decision_record
-> return decision + decision_record_id + generation_id + provenance
```

The decision stays pinned to the generation selected at the start even if `current_generation` changes
concurrently.

## 12. Cache

Keep and adapt the existing content-addressed world cache. Cache is strictly **optimization, not
authority**. The cache identity must commit to at least `generation_id`, the relevant world
parameters, the relevant Monte Carlo/config identity, and the runner/derived-artifact identity where
needed. On a hit, verify persisted content digest; on mismatch, treat as a **miss**, record evidence
where existing conventions support it, and rebuild. Deleting the entire cache must never affect
correctness. New generation content naturally produces a new cache key.

## 13. Manager-specific context

Keep global prediction certification distinct from manager-specific state. The **generation** is the
global certified predictive world. The **decision record** carries `generation_id` + manager
packet/context digest + decision request digest + result digest. Do not put mutable manager-specific
state into global generation identity unless it genuinely belongs to prediction generation.

## 14. Replay / research / test path

Matrix injection is allowed only through an explicitly separate non-production interface/module
(`NON_PRODUCTION_REPLAY_ONLY` / `REPLAY_ONLY` / `TEST_ONLY`). Non-production injected worlds cannot
mint generation rows, cannot update `current_generation`, cannot write `engine_decision_records` as
production decisions, and cannot be selected by production `make_decision`. Historical certified replay
does **not** need injection: `make_decision(..., generation_id=<historical>)` re-derives from retained
evidence/snapshot.

## 15. Re-derivation audit

Add one canonical verification command/function pair (`verify generation <generation_id>`,
`verify decision <decision_id>`). Generation verification proves: manifest digest == generation_id;
exact runs exist; runs complete; versions valid; cutoff/event valid; dependencies reproduce; snapshot
identity verifies; PE-8 evidence refs reproduce; horizon state reproduces. Decision verification
proves: the referenced generation verifies; manager/request digests verify; the predictive world
re-derives; the derived/result evidence matches the recorded decision digest. Where full deterministic
decision re-execution is genuinely deterministic, perform it; where a component is not safely
byte-for-byte replayable, verify the strongest frozen persisted artifact identity available and
**document that boundary truthfully**. Do not invent determinism the engine does not have.

## 16. `build_manager_packet.py`

Keep the useful repair; adapt it to the generation architecture. Its production output exposes
provenance: `generation_id`, exact certified run identities, snapshot identity, relevant evidence refs,
manager packet digest. Add a real CLI/output regression test.

## 17. Required tests

Preserve all original 24 PE-9 hard cases. Repurpose the adversarial tests away from impossible
"Python object unforgeability". **Refusal tests:** unknown generation ID; manifest mutated after the
generation ID was calculated; wrong event; wrong cutoff; unsupported model version; missing
dependency; missing projection; partial DGW missing fixture evidence; mutated snapshot; mismatched data
snapshot; inconsistent `planning_context_hash`; missing required PE-8 evidence; stale/invalid cache
content; replay/test world presented to production; production caller attempting to pass a matrix,
bundles, run mappings or a certification object; latest-per-family bypass attempt; current mutable fact
tables changed after a historical certification. **Success tests:** canonical four-event certified
generation; idempotent re-certification produces the same generation ID; generation + pointer
transaction succeeds atomically; decision resolves the current generation; explicit historical
generation decision works; valid cache hit; cache deletion/rebuild preserves decision correctness;
canonical Free Hit path; canonical manager-world path; `build_manager_packet` real CLI path; generation
verify passes; decision verify passes. **API-surface tests** proving production entrypoints do not
accept matrices, bundles, run-ID mappings, certification objects or cache handles.

## 18. Snapshot retention

Do not delete evidence required by persisted generations/decisions. Minimum safe rule: snapshots
referenced by surviving generation/decision evidence must not be garbage-collected; unreferenced
temporary snapshots may be cleaned per existing conventions. No complex archival subsystem in PE-9.

## 19. Frozen constraints — no football-logic change

No change to: team model; player-rate model; minutes model; xPts equations; Monte Carlo
distributions/RNG; scoring rules; transfer policy; chip policy; Wildcard status; calibration
thresholds; PE-8 promotion state; the current + next 3 transfer horizon; H1-only lineup/captain
horizon; DGW fixture grain; blank-event semantics. Frozen model identities remain frozen. This is
certification/provenance/integration only.

## 20. File-level plan (inspect first, then confirm)

**KEEP** — `fpl_brain/execution_snapshot.py`; `fpl_brain/certified_bundle.py`'s canonical identity,
validators, `certified_bundle_from_explicit_ids`, `declared_required_versions`, `certify_horizon_bundles`
and the bypass register (updated); `fpl_brain/monte_carlo.require_input_run` and the refusal
discipline; `fpl_brain/causality.py`; `route_optimizer.world_cache_key` (extended);
`fpl_brain/manager_worlds.py`'s version enforcement; the Free Hit certification plumbing; the
`build_manager_packet` repair; the PE-9 adversarial test infrastructure and the 24 hard cases.

**REMOVE** — `ValidatedCertificationArtifact` authority and its closure mint token;
`route_optimizer`'s `_issue_world_matrix` / `_issue` / `_ISSUED_WORLD_MATRICES` / `IssuedWorldMatrix` /
`require_certified_prebuilt_matrix`; caller-prebuilt matrix and caller bundle parameters on production
paths; `four_gw_decision.py`'s caller-supplied `required_model_versions`; `event_support_from_db` from
production readiness; remaining `get_projection_run(...) or {}` silent fallbacks; the bypass-register
entries describing removed mechanisms.

**MODIFY** — `fpl_brain/database.py` (schema 17 → 18 + migration + append-only triggers);
`fpl_brain/certified_bundle.py` (generation construction, manifest canonicalization,
`certify_horizon_bundles` → one transaction with the pointer); `fpl_brain/four_gw_decision.py`
(generation-based readiness and decision path); `fpl_brain/route_optimizer.py` (cache key commits to
`generation_id`; no matrix admission); `fpl_brain/free_hit_request_adapter.py`,
`fpl_brain/manager_worlds.py`, `fpl_brain/finalist_refinement.py`, `fpl_brain/route_stability.py`
(generation-carrying signatures); `scripts/build_manager_packet.py`,
`scripts/run_four_gw_decision.py` (descriptor-only, provenance output); the four bypass scripts
(GAP 2 audit).

**ADD** — the generation store accessors and `make_decision` (descriptor-only);
`verify generation` / `verify decision`; the non-production replay module boundary; the new tests
(refusals, successes, API surface).

## 21. Validation order

1. schema/migration tests; 2. generation canonicalization/content-address tests; 3. original PE-9
certification tests; 4. the seven proven-gap tests; 5. production API-surface tests; 6. snapshot/data
provenance tests; 7. cache tests; 8. optimizer/comparator/refinement/stability; 9. Free Hit;
10. manager worlds; 11. `build_manager_packet` CLI; 12. four-GW decision path; 13. verify/re-derivation
tests; 14. full suite. **Do not knowingly commit a remediation-caused red suite.**

## 22. Out of scope

Do not implement the superseded service process or its socket; do not implement security theatre that
tries to make in-process Python objects unforgeable; do not change football logic; do not promote
PE-8; do not start PE-10; do not merge anything.
