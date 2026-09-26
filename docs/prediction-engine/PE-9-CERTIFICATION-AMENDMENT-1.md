# PE-9 Certification Integration — Authority Amendment 1

**Status:** AMENDMENT — governing authority for the PE-9 remediation. PE-9 remains **OPEN**.
**Revision 3** — incorporates Sol High design reviews 1 and 2 (P1 blockers: in-process trust boundary;
manifest immutability, lifecycle and payload pinning; code-identity ordering; cache identity).
Amends `docs/prediction-engine/PE-9-CERTIFICATION-INTEGRATION.md` (authority `d4a5b132`).
**Issued by:** Product Owner instruction, 2026-09-26, on the evidence of Sol reviews 1–4.
**Base for remediation:** `41eef48d7cf8d68ffaeeb54fdb9412cd0c04497d` (tree `5f98bb9eed3c0273b18aa4c5de962536f6e86a2a`).

This amendment does **not** authorise implementation. It changes the certification contract and
requires a design review by Sol High before any code is written (see §12).

## 1. Evidence base and preserved history

Every candidate produced under the original authority was **rejected**; none was merged, and
`feature/prediction-engine-v1` remains at the frozen PE-8 merge `95b6f71a`.

| Stage | SHA | Sol verdict | CI |
| --- | --- | --- | --- |
| authority | `d4a5b132d17bb71369e93eeb5d394fb70e19382e` | — | success |
| original candidate | `9c8f333798bfd94ecc1f066e66bfbcb22b803ad0` | `FIX_REQUIRED` / P1 (review 1) | `36081104050` success |
| repair 1 | `14172b1e42888b736275a61c42ee1ffb0e0b4057` | `FIX_REQUIRED` / P1 (review 2) | `36161965191` success |
| repair 2 | `7fc8326ac166e386c5392ee7bdd95a5d553e770b` | `FIX_REQUIRED` / P1 (review 3) | `36179945670` success |
| repair 3 | `41eef48d7cf8d68ffaeeb54fdb9412cd0c04497d` | `FIX_REQUIRED` / P1 (review 4) | `36201728611` success |

Terminal orchestrator state after repair 3: `HUMAN_REQUIRED`, reason `REPAIR_ATTEMPT_LIMIT_EXCEEDED`,
`repair_attempt` 4. Every review's summary is preserved verbatim in the phase evidence
(`pe9_failure_history.md`), together with the lifecycle events and durable state snapshots. The
historical record is not rewritten: a rejected candidate is never described as approved.

## 2. Root-cause finding

**The previous implementations tried to make Python objects themselves carry certification
authority. That is architecturally wrong, and Sol defeated every variant of it.**

Rejected mechanisms: caller-computed canonical identity; mutable artifact mappings;
`isinstance(value, ValidatedCertificationArtifact)`; `dict`/`list` subclass "immutability";
caller-writable matrix stamps; module-global issuance registries (`_ISSUED_WORLD_MATRICES`);
callable/private issuers (`_issue_world_matrix`, `_issue`); closure-captured mint tokens/issuers
(`__init__.__closure__`, `build_event_worlds.__closure__`); caller-replaceable content digests
(`_document` / `_content_digest`).

**Therefore: Python-level secrecy, type identity, private attributes, closure capture, mutable
registries, or caller-carried capabilities MUST NOT be treated as production certification
authority** — and, since Sol design review 1, **an in-process library boundary is itself insufficient
while ordinary production callers share that process**. The trust boundary is a process boundary.

## 3. New canonical invariant (governing rule)

> A production decision may consume predictive data only when the certified production boundary
> itself can **reproduce certification from authoritative persisted evidence**, inside a process the
> caller does not control. No caller-supplied object, type, attribute, digest, capability, registry
> entry, closure value, matrix, cache object, connection, or previously validated Python instance
> constitutes authorization.

Certification is established from persisted evidence: exact event set; exact per-event cutoff; exact
per-family run IDs; model-family identity; authoritative model versions; run status = complete;
`planning_context_hash`; code snapshot identity; data snapshot identity; dependency edges / closure;
**a digest of each run's complete predictive payload**; execution / certification state; PE-8
evidence identity and state where applicable.

### 3.1 The immutable, event-sourced certification manifest is the authority record

"Persisted evidence" means an **immutable, append-only certification manifest**, not "the newest rows
that happen to match":

- a manifest pins the exact event set, per-event cutoff, per-family run IDs with their recorded
  family, version, status and cutoff, the authoritative required versions, the dependency closure,
  `planning_context_hash`, code and data snapshot identities, per-event bundle identities, PE-8
  evidence identity, and **`payload_digest` for every pinned run** — a cryptographic digest over that
  run's complete serialized predictive payload;
- the manifest store is **append-only and event-sourced**: certification, supersession and revocation
  are separate immutable events. No manifest row's status is ever mutated in place;
- **effective state is computed at resolution**: a manifest may enter production only while it has a
  `CERTIFIED` event and neither a `REVOKED` nor a `SUPERSEDED_BY` event. Revoked manifests are always
  rejected. Superseded manifests are usable **only** in an explicitly requested replay mode, and the
  certified production service refuses replay mode for production decisions;
- revalidation compares the manifest's pinned facts **and payload digests** against the current store
  before any matrix is generated or read from cache. Any metadata, payload, edge, or lifecycle
  difference is `HISTORICAL_EVIDENCE_DRIFT` and is refused, never silently re-resolved;
- every decision records the `manifest_id`, the resolved code and data snapshots, and the evidence
  used, so a decision remains reproducible after later changes to current tables.

### 3.2 The trust boundary is an isolated certified process

No in-process mechanism can make a matrix trustworthy inside a process the caller controls. Therefore:

1. **Certified production decisions are produced by an isolated certified service process.** Its only
   external input is a **serialized descriptor request** (no Python objects, no caller connection); it
   opens the authoritative stores itself with **read-only credentials** (`mode=ro` / `query_only`) and
   never receives them from a caller.
2. **Manifest resolution, code verification, payload hashing, matrix generation/cache, decision
   consumption and certified-result persistence all happen inside that process.** The service returns
   decisions and evidence identifiers; it never returns a world matrix.
3. **In-process library calls are non-production.** Results computed by library entry points are
   stamped `NON_PRODUCTION` and can never become certified: certified decision records are written
   only by the service process through its own store credentials, so there is no token, stamp or
   capability for a caller to mint — the certified artifact *is* the persisted record.
4. **Code identity is verified against the manifest, after resolving it**, before any certification
   act: the service hashes the exact source and loaded code objects of the certification modules and
   requires equality with the manifest's recorded code snapshot. A mismatch is
   `CERTIFICATION_CODE_IDENTITY_MISMATCH`. This detects tampering inside a process whose start-up was
   authorised; it is **not** claimed as impossibility.
5. An adversary able to execute arbitrary code inside the certified service process itself, before or
   during certification, cannot be contained by any in-process design. The amendment's answer is the
   process boundary plus the code-identity gate plus the append-only manifest: producing an
   uncertified certified decision requires tampering with the certified process itself, which is
   detectable and auditable, rather than calling a library function. **Monkey-patching is inside the
   threat model** (§12 requires the review to attempt it, §9 requires tests for it).

## 4. Artifact semantics

The certification artifact is **evidence/data, not a security credential**, and the authoritative
artifact is the persisted manifest plus the service-written decision record. Presented artifact
*content* (for audit, replay, or offline verification) must be canonicalised, re-run through the
complete certification contract, reconciled with authoritative persisted evidence, and checked for
bundle/event/run identities, cutoff, dependency closure, model versions, code/data/planning
identities, `decision_search_permitted`, and temporal/finality state. Validation never short-circuits
on `isinstance(...)`; a previously valid instance is never trusted because it was valid earlier; type
membership confers **zero** authority. Validation is content/evidence based.

## 5. Certified prebuilt matrices are removed from production

The production contract no longer contains *caller creates matrix → caller mints a capability →
certified decision consumes matrix*. Production flow is:

```
serialized descriptor request -> certified service process
  -> resolve manifest -> verify code identity -> verify payload digests
  -> generate/load matrix internally -> decide -> persist decision + evidence
  -> return decisions (never matrices)
```

The matrix is a **consequence** of the certified predictive world, never an authorization input.
There must be: no certified matrix issuer; no certified matrix registry; no caller-writable
certification stamp; no closure token; no hidden capability; **no production API that returns a world
matrix**. Tests, historical replay, benchmarking and research use an explicitly named
`NON_PRODUCTION` / `TEST_ONLY` / `REPLAY_ONLY` interface that cannot enter the production path.

## 6. Cache contract

Caching is allowed; caller authority is not. Production cache lookup stays **inside the certified
service**:

1. the service resolves and revalidates the manifest (including payload digests) first;
2. the cache identity is a digest that **commits to the complete serialized matrix content together
   with the manifest and bundle identity** — not merely a bundle identifier;
3. a hit must match that identity, and the loaded bytes must hash to the digest the identity commits
   to;
4. a mismatch (stale, tampered, wrong event) is a miss → regenerate from the certified bundle; a
   mismatch is never "repaired" silently.

Caller-supplied matrices, cache objects, registry entries, connections or "certified" stamps are
never proof.

## 7. Production paths governed by this contract

All converge on the **one** certified service boundary; no module implements its own certification
logic: `route_optimizer.build_event_worlds`, `route_optimizer.optimize`,
`route_comparator.compare_routes`, `finalist_refinement`, `route_stability`, Free Hit predictive
loading (`free_hit_request_adapter.load_certified_route_worlds`),
`manager_worlds.build_manager_worlds`, `scripts/build_manager_packet.py`,
`scripts/run_four_gw_decision.py`, warm/content cache paths, and any other decision or chip path that
loads persisted predictive runs.

## 8. `build_manager_packet.py`

The Repair-3 repair is **kept**: the command remains functional, requests certification through the
certified service, validates the exact run IDs and payload digests it consumes, reports the
`manifest_id` it used, and never bypasses certification. A real CLI/output regression test is
required.

## 9. Adversarial acceptance tests (required)

| Attack | Required outcome |
| --- | --- |
| self-consistent raw artifact with an incomplete authority contract | REFUSED |
| valid artifact, later mutated | REFUSED |
| alternate run IDs inserted into a previously valid artifact | REFUSED |
| `isinstance`-compatible fake or modified object | REFUSED |
| artifact constructor/token extracted through `__closure__` | provides NO authority |
| content *and* caller-provided digest changed together | REFUSED |
| **predictive payload changed beneath unchanged run metadata** | `HISTORICAL_EVIDENCE_DRIFT` |
| **dependency edge changed beneath unchanged run metadata** | `HISTORICAL_EVIDENCE_DRIFT` |
| arbitrary prebuilt matrix | production surface has no admission path and returns no matrix |
| copied matrix identity/stamp | REFUSED / impossible through the certified API |
| direct former issuer call | provides NO production admission |
| mutation of the former registry | provides NO production admission |
| closure issuer/token extraction | provides NO production admission |
| cache injection, or a cache whose bytes do not match its identity | REFUSED |
| monkey-patching a library/boundary function in the caller's process | cannot produce a certified decision: certification happens in the service process and the caller never supplies a connection or a matrix |
| monkey-patching certification code inside the service | `CERTIFICATION_CODE_IDENTITY_MISMATCH` |
| superseded or revoked manifest used for a production decision | REFUSED; superseded is replay-only |
| `EXACT_ID` naming a manifest in a disallowed effective state | REFUSED |
| current tables mutated after certification | `HISTORICAL_EVIDENCE_DRIFT`; the manifest identity is unchanged |
| canonical persisted runs + valid manifest (service path) | ACCEPTED |
| legitimate internal cache hit for the same certified world | ACCEPTED |
| Free Hit canonical path | ACCEPTED |
| `build_manager_packet` real CLI path | ACCEPTED |

Plus: changing mutable current tables **after** a historical certification must not rewrite the
historical certified identity. All original 24 PE-9 hard cases are preserved.

## 10. Frozen constraints (unchanged)

No change to: predictive quantities; team model; player-rate model; minutes model; xPts model; Monte
Carlo RNG; scoring rules; transfer policy; chip policy; Wildcard status (review-only / uncalibrated as
frozen); PE-8 calibrator promotion state; calibration thresholds; the current + next 3 transfer
horizon; H1-only lineup/captain semantics; DGW fixture grain; blank-event semantics.

This remains a **certification/integration remediation only**.

## 11. Reuse of Repair-3 work

The remediation branches from `41eef48` and must not rebuild PE-9 from scratch.

**Keep:** required model-version enforcement; horizon certification; PE-8 evidence integration
(without promotion); missing-upstream refusal; data-snapshot validation; certification propagation;
the `build_manager_packet` repair; the adversarial test infrastructure; the provenance and
bounded-dirty-work gates.

**Remove / simplify (the flawed authority mechanisms only):** capability secrecy; closure-based
authorization; registry-based matrix authorization; type-based artifact authority; certified
caller-prebuilt matrix admission; matrix-returning production library surfaces; in-process results
being treated as certified.

## 12. Design review gate (mandatory before implementation)

No implementation call is authorised until Sol High reviews the **design, not code**, and reports no
P1/P2 architectural blocker. The review must attempt: `__closure__` access and constructor/token
extraction; monkey-patching (including the code-identity gate); object mutation, base-class mutators,
attribute assignment on frozen dataclasses; fake types; alternate run IDs; raw mappings; replaced
digests; payload and dependency-edge substitution beneath unchanged metadata; direct private-function
calls; former issuer/registry resurrection; cache injection and cache-content substitution; arbitrary
prebuilt matrices; stamp copying; evidence-record reuse with a substituted matrix; manifest
substitution, supersession/revocation confusion, and replay-mode leakage into production; and mutation
of current tables after certification.

The question put to Sol:

> Can ordinary code running in the same Python process cause uncertified predictive data or a
> caller-created world matrix to enter the production-certified decision path without the boundary
> reproducing certification from persisted authoritative evidence?

## 13. Terminal states of the design review

- Sol reports **NONE** for P1/P2 architectural blockers → `PE-9 REMEDIATION DESIGN APPROVED —
  READY_FOR_IMPLEMENTATION`, then stop and await Product Owner authorization.
- Sol finds a P1/P2 design flaw → revise the design/authority **only** and review again. Do not spend
  an implementer cycle on a design Sol can already break.

## 14. PE-10 remains blocked

Until this remediation is implemented, exact-SHA CI green, Sol-approved, merged, and the merge CI is
green — with PE-9 **FROZEN** — there is no PE-10 branch, no PE-10 authority, and no PE-10
implementation.
