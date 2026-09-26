# PE-9 Certification Integration — Authority Amendment 1

**Status:** AMENDMENT — governing authority for the PE-9 remediation. PE-9 remains **OPEN**.
**Revision 2** — incorporates Sol High design review 1 (two P1 blockers: no in-process trust
boundary; no immutable authoritative manifest). Amends `docs/prediction-engine/PE-9-CERTIFICATION-INTEGRATION.md` (authority `d4a5b132`).
**Issued by:** Product Owner instruction, 2026-09-26, on the evidence of Sol reviews 1–4.
**Base for remediation:** `41eef48d7cf8d68ffaeeb54fdb9412cd0c04497d` (candidate tree `5f98bb9eed3c0273b18aa4c5de962536f6e86a2a`).

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
(`pe9_failure_history.md`), together with the lifecycle events and the durable state snapshots. The
historical record is not rewritten: a rejected candidate is never described as approved.

## 2. Root-cause finding

**The previous implementations tried to make Python objects themselves carry certification
authority. That is architecturally wrong, and Sol defeated every variant of it.**

Rejected mechanisms, each proven defeatable by ordinary code in the same process:

- caller-computed canonical identity (`canonical_bundle_identity` over caller-supplied values);
- mutable artifact mappings passed as "the artifact";
- `isinstance(value, ValidatedCertificationArtifact)` as validation;
- `dict` / `list` subclass "immutability";
- caller-writable matrix stamps (`_p2_certified_bundle_identity` and successors);
- module-global issuance registries (`_ISSUED_WORLD_MATRICES`);
- callable / private issuer functions (`_issue_world_matrix`, `_issue`);
- closure-captured mint tokens and issuers (`__init__.__closure__`, `build_event_worlds.__closure__`);
- caller-replaceable content digests (`_document` / `_content_digest` replacement).

**Therefore: Python-level secrecy, type identity, private attributes, closure capture, mutable
registries, or caller-carried capabilities MUST NOT be treated as production certification
authority.** Any design whose authority depends on a caller not being able to reach, copy, forge
or mutate a Python object is rejected on sight.

## 3. New canonical invariant (governing rule)

> A production decision may consume predictive data only when the certified production boundary
> itself can **reproduce certification from authoritative persisted evidence**. No caller-supplied
> object, type, attribute, digest, capability, registry entry, closure value, matrix, cache object,
> or previously validated Python instance constitutes authorization.

Certification is established from persisted evidence:

- exact event (and the exact horizon event set);
- exact cutoff;
- exact per-family run IDs;
- model-family identity;
- authoritative model versions;
- run status = complete (final, non-provisional);
- `planning_context_hash`;
- code snapshot identity;
- data snapshot identity;
- dependency edges / closure;
- execution / certification state;
- PE-8 evidence identity and state where applicable.

The boundary must **reproduce or revalidate these facts at the point of use**. Passing through a
function is not validation; only re-derivation from persisted evidence is.

### 3.1 The historical certification manifest is the authority record

"Persisted evidence" means an **immutable, versioned certification manifest**, not "the newest rows
that happen to match". A manifest is an append-only record that selects and pins, for one certified
decision world:

- the exact event set and exact per-event cutoff;
- the exact per-family run IDs, with each run's recorded version, status and cutoff at certification
  time;
- the authoritative required model versions;
- the dependency edges / closure;
- `planning_context_hash`, code snapshot identity, data snapshot identity;
- execution / certification state and PE-8 evidence identity where it applies;
- the certified bundle identity derived from the above;
- the manifest's own identity and the identity of the certification act that produced it.

Because it is append-only and itself identified, it cannot be rewritten by later mutation of current
tables: a decision revalidates **against the manifest**, and a current row that no longer agrees with
the manifest is *drift* (`HISTORICAL_EVIDENCE_DRIFT`), refused rather than silently re-resolved. Two
manifests for the same event set are distinguished by identity, and the decision records which
manifest it used, so a decision is reproducible after the fact.

### 3.2 The trust boundary is a process boundary, not in-process secrecy

No in-process mechanism — closures, private names, type identity, capabilities, registries, digests,
or guards that an attacker can also patch — can make a matrix trustworthy *inside a process the
attacker controls*. The amendment therefore fixes the trust boundary architecturally:

1. **Certified production decisions execute in a dedicated certified entry point** (the production
   CLI/module boundary) that performs manifest resolution, revalidation, matrix generation and
   decision consumption in one process, and returns **decisions and their evidence**, never matrices,
   to anything outside it.
2. **Matrices never cross into caller-controlled code** on a production path. There is no supported
   production call that hands a world matrix to a caller, so monkey-patching a caller cannot
   manufacture a certified input: the certified process never reads a matrix from its caller.
3. **Code identity is part of the evidence.** The manifest records the certification code snapshot,
   and the certified entry point verifies the code it is actually running (source digests and loaded
   code objects of the certification modules) against it before certifying anything; a mismatch is
   refused (`CERTIFICATION_CODE_IDENTITY_MISMATCH`). This is detection of tampering within a process
   whose start-up was authorised — it is **not** claimed as impossibility.
4. An adversary able to execute arbitrary code *inside the certified process itself, before or
   during certification*, cannot be contained by any in-process design; the amendment requires the
   process boundary, the code-identity gate and the manifest so that such an adversary must tamper
   with the certified process itself (a detectable, auditable act) rather than merely call a library
   function. This is stated as the residual, and **monkey-patching is no longer excluded from the
   threat model**: §12 requires the design review to attempt it, and §9 requires tests that exercise
   it against the design.

## 4. Artifact semantics

The certification artifact is **evidence/data, not a security credential**. A caller may present
artifact *content*; the production boundary must:

1. canonicalise the presented snapshot;
2. re-run the complete certification contract over it;
3. reconcile it with authoritative persisted evidence;
4. verify the claimed bundle / event / run identities;
5. verify cutoff and dependency closure;
6. verify model versions;
7. verify code / data / planning identities;
8. verify `decision_search_permitted` and temporal/finality state.

Validation must never short-circuit on `isinstance(...)`, and a previously valid instance is never
trusted merely because it was valid earlier. A strongly typed immutable representation remains
acceptable **for ergonomics only**; type membership confers **zero** authority. Validation is
content/evidence based.

## 5. Certified prebuilt matrices are removed from production

The production contract no longer contains the flow *caller creates matrix → caller mints/stamps a
capability → certified decision consumes matrix*. That flow is the defect.

Production flow is now:

```
certification evidence (descriptors)
  -> revalidate against authoritative persisted runs
  -> canonical internal loader / cache / generator
  -> world matrix (internal)
  -> decision logic
```

The matrix is a **consequence** of the certified predictive world, never an authorization input.
There must be:

- no certified matrix issuer;
- no certified matrix registry;
- no caller-writable certification stamp;
- no closure token;
- no hidden capability.

**Certified production APIs do not accept caller-supplied prebuilt world matrices at all.** Where
tests, historical replay, benchmarking or research need injected matrices, they must use an
explicitly named `NON_PRODUCTION` / `TEST_ONLY` / `REPLAY_ONLY` interface that **cannot** enter the
production-certified path.

## 6. Cache contract

Caching is allowed; caller authority is not. Production cache lookup stays **internal to the
certified loader**:

1. the boundary re-establishes the certified predictive world from persisted evidence;
2. it derives the expected content/cache identity internally;
3. it loads the corresponding cached matrix;
4. it verifies the cached content against that expected certified world.

A cache hit is therefore accepted **because the loader re-proved the world**, not because a caller
presented a cache object. Caller-supplied matrices, cache objects, registry entries or "certified"
stamps are never proof.

## 7. Production paths governed by this contract

Every one of these must converge on the **same** canonical certification-and-load boundary; no
module may implement its own certification logic:

`route_optimizer.build_event_worlds`, `route_optimizer.optimize`, `route_comparator.compare_routes`,
`finalist_refinement` (refinement / escalation), `route_stability` (ladder / stability gate),
Free Hit predictive loading (`free_hit_request_adapter.load_certified_route_worlds`),
`manager_worlds.build_manager_worlds`, `scripts/build_manager_packet.py`,
`scripts/run_four_gw_decision.py`, warm/content cache paths, and any other decision or chip path
that loads persisted predictive runs.

## 8. `build_manager_packet.py`

The Repair-3 repair of `scripts/build_manager_packet.py` is **kept**: the command must remain
functional, load canonical certification evidence, validate the exact run IDs it consumes, and
never bypass certification. A real CLI/output regression test is required (Sol review 4).

## 9. Adversarial acceptance tests (required by this amendment)

The amended authority requires these to be permanent, executable tests:

| Attack | Required outcome |
| --- | --- |
| self-consistent raw artifact with an incomplete authority contract | REFUSED |
| valid artifact, later mutated | REFUSED |
| alternate run IDs inserted into a previously valid artifact | REFUSED |
| `isinstance`-compatible fake or modified object | REFUSED |
| artifact constructor/token extracted through `__closure__` | provides NO authority |
| content *and* caller-provided digest changed together | REFUSED |
| arbitrary prebuilt matrix | production certified API has no admission path |
| copied matrix identity/stamp | REFUSED / impossible through the certified API |
| direct invocation of the former issuer | provides NO production admission |
| mutation of the former registry | provides NO production admission |
| closure issuer/token extraction | provides NO production admission |
| cache injection | REFUSED |
| canonical persisted runs + valid certification evidence | ACCEPTED |
| legitimate internal cache hit for the same certified world | ACCEPTED |
| Free Hit canonical path | ACCEPTED |
| `build_manager_packet` real CLI path | ACCEPTED |
| monkey-patched boundary/loader function in the caller's process | cannot produce a certified decision: the certified entry point runs the certification, revalidation and matrix generation itself and returns only decisions/evidence |
| monkey-patched certification module inside the certified process | refused by the code-identity gate against the manifest's code snapshot (`CERTIFICATION_CODE_IDENTITY_MISMATCH`) |
| current rows mutated after certification | refused/reported as `HISTORICAL_EVIDENCE_DRIFT`; the manifest identity is unchanged |
| two candidate manifests for one event set | distinguished by identity; the decision records the manifest it used |

Plus: changing mutable current tables **after** a historical certification must not rewrite the
historical certified identity. All original 24 PE-9 hard cases are preserved.

## 10. Frozen constraints (unchanged by this amendment)

No change to: predictive quantities; team model; player-rate model; minutes model; xPts model;
Monte Carlo RNG; scoring rules; transfer policy; chip policy; Wildcard status (review-only /
uncalibrated as frozen); PE-8 calibrator promotion state; calibration thresholds; the current + next
3 transfer horizon; H1-only lineup/captain semantics; DGW fixture grain; blank-event semantics.

This remains a **certification/integration remediation only**.

## 11. Reuse of Repair-3 work

The remediation branches from `41eef48` — it already contains substantial valid PE-9 work — and must
**not** rebuild PE-9 from scratch.

**Keep:** required model-version enforcement; horizon certification; PE-8 evidence integration
(without promotion); missing-upstream refusal; data-snapshot validation; certification propagation;
the `build_manager_packet` repair; the adversarial test infrastructure; the bounded dirty-work and
provenance gates.

**Remove / simplify (the flawed authority mechanisms only):** capability secrecy; closure-based
authorization; registry-based matrix authorization; type-based artifact authority; certified
caller-prebuilt matrix admission.

## 12. Design review gate (mandatory before implementation)

No implementation call is authorised until Sol High reviews the **design, not code**, and reports no
P1/P2 architectural blocker. The design review must attempt to break the proposed design using:
`__closure__` access; monkey-patching; object mutation; base-class mutators; fake types; constructor
extraction; alternate run IDs; raw mappings; replaced digests; direct private-function calls; cache
injection; arbitrary matrices; registry mutation; module introspection.

The question put to Sol:

> Can ordinary code running in the same Python process cause uncertified predictive data or a
> caller-created world matrix to enter the production-certified decision path without the boundary
> reproducing certification from persisted authoritative evidence?

## 13. Terminal states of the design review

- Sol reports **NONE** for P1/P2 architectural blockers → `PE-9 REMEDIATION DESIGN APPROVED —
  READY_FOR_IMPLEMENTATION`, then stop and await Product Owner authorization.
- Sol finds a P1/P2 design flaw → revise the design/authority **only** and review again. Do not
  spend an implementer cycle on a design Sol can already break.

## 14. PE-10 remains blocked

Until this remediation is implemented, exact-SHA CI green, Sol-approved, merged, and the merge CI is
green — with PE-9 **FROZEN** — there is no PE-10 branch, no PE-10 authority, and no PE-10
implementation.
