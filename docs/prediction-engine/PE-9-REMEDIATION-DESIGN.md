# PE-9 Remediation Design — certification reproduced from persisted evidence

**Status:** PROPOSED DESIGN — for Sol High adversarial design review. No implementation is authorised
until that review reports no P1/P2 architectural blocker (`PE-9-CERTIFICATION-AMENDMENT-1.md` §12).
**Revision 4** — closes Sol design review 3's remaining blocker: the cache now has an authoritative, service-owned expectation instead of a caller-recomputable digest. Closes Sol design review 1 and review 2. The production boundary is now an isolated
**service process** rather than a library call; the manifest is **event-sourced and append-only** with
**pinned payload digests**; code identity is checked **after** manifest resolution and against the
**manifest's** snapshot; and the cache identity commits to the complete serialized matrix.
**Base:** `41eef48d7cf8d68ffaeeb54fdb9412cd0c04497d` (repair-3 candidate, tree `5f98bb9e…`).
**Scope:** certification/integration only. No predictive quantity, model, RNG, scoring, chip,
transfer, horizon or calibration change.

## 0. How each prior design-review blocker is closed

| Prior finding | Where closed |
| --- | --- |
| in-process library boundary; caller-supplied connection; production callers share the process | §1 certified service process; §2 flow; §3.3 |
| in-process results could become certified by carrying a manifest id | §1.3 certified records are written only by the service; library results are `NON_PRODUCTION` |
| manifest authority not immutable; status mutated by supersession | §5 event-sourced append-only lifecycle; effective state computed at resolution |
| `EXACT_ID` accepted on existence alone | §5 allowed effective states per mode; revoked always refused |
| superseded manifests could reach production | §5 replay-only, and the service refuses replay for production kinds |
| predictive values changeable beneath unchanged run metadata | §4 payload pinning; `HISTORICAL_EVIDENCE_DRIFT` |
| code identity compared against the caller's requested snapshot, before manifest resolution | §2 step 2-3; §3.4 ordering and reference |
| cache identity did not commit to the matrix content | §6 (revision 3: content-committing `cache_identity`) |
| cache injection: arbitrary bytes plus a recomputed public identity | §6 (revision 4: the index entry, written only when the service generated the matrix, is the authoritative expectation; a public key over caller bytes is only a label) |
| monkey-patching excluded as a residual | §3.5 residual restated around the process boundary; §10 test rows |

## 1. The certified service process

Certification is produced by `fpl_brain/certified_service.py`, run as its own OS process
(`python -m fpl_brain.certified_service --request <file.json> --out <file.json>`).

```python
# serialized request: the ONLY external input. No Python objects cross the boundary.
{
  "kind": "DECISION" | "MANAGER_PACKET" | "REPLAY",       # REPLAY is non-production
  "event": 5,
  "horizon": [5, 6, 7, 8],
  "required_versions": {"minutes": "...", "team": "...", "player": "...", "xpts": "..."},
  "planning_context_hash": "...",
  "claimed_code_snapshot": "...",     # a CLAIM to be checked, never the reference
  "claimed_data_snapshot": "...",     # a CLAIM to be checked
  "manifest_policy": {"kind": "LATEST_CERTIFIED_FOR_EVENT_SET"} # or {"kind": "EXACT_ID", "manifest_id": "..."}
  "decision": {"kind": "FOUR_GW_SEARCH", ...}
}
```

The service, **inside its own process**:

1. **opens the authoritative stores itself** with read-only credentials for the predictive store
   (`file:<path>?mode=ro`, `PRAGMA query_only=ON`) and writes only the evidence/decision store it
   owns;
2. **resolves the authoritative manifest** first (`resolve_manifest`) — before any code-identity
   comparison — and computes its **effective state** from the append-only event log;
3. **verifies code identity** by hashing the exact source bytes and loaded code objects of the
   certification modules and requiring equality with the **manifest's** `code_snapshot`; the request's
   `claimed_code_snapshot` is compared as a claim, and any disagreement is refused;
4. **verifies payload digests**: every pinned run's complete serialized predictive payload and its
   dependency edges are re-read and hashed and must equal the manifest's pins; any difference is
   `HISTORICAL_EVIDENCE_DRIFT`;
5. **generates or loads** the world matrix internally, keyed by `cache_identity` (§6);
6. **runs the decision** (search / comparator / refinement / stability / Free Hit / manager packet);
7. **persists** the decision and its evidence (manifest id, code snapshot, data snapshot, run ids,
   payload digests, cache identity) through its own store credentials — this persisted record **is**
   the certified artifact;
8. **returns decisions and evidence identifiers** — never a world matrix.

No module-name or attribute secrecy is relied on anywhere: there is no minting function, no registry,
no token, no stamp, and no capability object. A caller cannot manufacture a certified record because
certified records are written by the service process through credentials the library path never opens.

**Production entry points become clients:**

- `scripts/run_four_gw_decision.py` and `scripts/build_manager_packet.py` build the serialized
  descriptor request from their own CLI arguments, invoke the service, and report the decision and its
  evidence (including the `manifest_id` and the run ids consumed);
- library-level `certified_worlds` / `certify_*` helpers remain **internal** and are marked
  `NON_PRODUCTION`: their results are never persisted as certified, and the decision writer accepts
  only records that the service store already contains.

## 2. Data flow (production)

```
production CLI -> serialized request -> [ certified service process ]
                                          resolve manifest (effective state)
                                          verify code identity vs manifest.code_snapshot
                                          hash every pinned payload + dependency edge
                                          generate/load matrix (cache_identity verified)
                                          run decision
                                          persist decision + evidence  <-- the certified artifact
                                        <- decisions + evidence ids
production CLI -> report / consume decisions (never matrices)
```

## 3. Why the design resists each class of attack

### 3.1 Caller-supplied objects, types, digests, capabilities

Nothing crosses the service boundary but serialized descriptors. There is no `isinstance` check, no
minting constructor, no registry, no stamp, no caller digest and no capability on the authority path,
so there is nothing to impersonate, copy or extract. A fabricated descriptor request can only ever
ask the service to certify something the store does not support — and then it is refused.

### 3.2 Evidence reuse, substituted matrices, cache substitution

Matrices are produced inside the service and never handed out on a production path, so a caller cannot
substitute one. Cache content is admitted only when the loaded bytes hash to the digest the
`cache_identity` commits to (§6).

### 3.3 Monkey-patching the caller's process

Patching anything in the caller's process has no effect on certification: the caller supplies a
serialized request and a path, not a connection, not a matrix, and not a code object. The certified
path never reads a matrix or an evidence object from its caller.

### 3.4 Monkey-patching inside the service

The code-identity gate (step 3) hashes the exact source bytes **and the loaded code objects** of the
certification modules and requires equality with the manifest's recorded `code_snapshot`. A patched
module or function is detected before certification. The comparison happens **after** manifest
resolution, so the reference is the manifest's snapshot, not a value supplied with the request.

### 3.5 The stated residual

An adversary able to execute arbitrary code inside the certified service process **before or during
certification** — including patching the code-identity gate itself — cannot be contained by any
in-process design, and the design does not claim otherwise. The response is structural: producing an
uncertified "certified" decision then requires tampering with the certified process itself, which the
code-identity gate, the append-only manifest log and the persisted decision record make detectable and
auditable. The trust root is explicit: **the append-only manifest store, the service's store
credentials, and the code identity of the service at the moment it certified.** No Python object
handed around at run time is part of it.

## 4. Payload pinning (closes "values changed beneath unchanged metadata")

For every pinned run the manifest records `payload_digest`:

```
payload_digest = sha256( canonical( run_id, family, version, status, cutoff, event,
                                     every predictive row the loader will read ) )
closure_digest = sha256( canonical( dependency edges of the run, by family ) )
```

Before any matrix generation or cache read, the service re-reads each run's payload and edges and
requires both digests to equal the manifest's pins. Any difference — metadata, payload, edge or
lifecycle — is `HISTORICAL_EVIDENCE_DRIFT` and the decision is refused.

## 5. Manifest lifecycle (event-sourced, append-only)

The manifest store is append-only and holds **events**, never mutable rows:

```
MANIFEST_CERTIFIED(manifest_id, pins..., certified_at, certifying_actor)
MANIFEST_SUPERSEDED(manifest_id, by_manifest_id, at)
MANIFEST_REVOKED(manifest_id, reason, at)
```

- **Creation:** `certify_horizon_bundles` / `scripts/certify_gw5_gw8.py` derive the certified world
  from persisted runs, compute the identity and payload digests, and append `MANIFEST_CERTIFIED`. The
  store rejects duplicate ids and any update-in-place.
- **Effective state is computed at resolution** from the events: `REVOKED` → always rejected;
  `SUPERSEDED` → usable only in explicit `REPLAY` mode; otherwise `CERTIFIED`.
- **Production kinds refuse replay mode**, so a superseded or revoked manifest can never back a
  production decision, and `EXACT_ID` must name a manifest whose effective state is allowed for the
  requested mode.
- Every decision records the `manifest_id` used, so a historical decision is reproducible by id even
  after supersession.

## 6. Cache contract

The cache is a performance cache, never an authorization input, and a **self-consistent digest is not
evidence of origin**: a caller can always hash arbitrary bytes and compute any public key over them.
The design therefore separates the public key from the authoritative expectation.

```
certified_input_key = H( "pe9-cache-v1", manifest_id, bundle_identity, canonical(union_ids),
                         canonical(config), generation_identity )
```

- `certified_input_key` is public and caller-computable; it is a **label**, not authorization. It names
  the deterministic certified inputs (manifest, bundle, union, configuration, generation identity).
- The **authoritative expectation** lives in a **service-owned cache index** in the store the service
  writes: `certified_input_key -> matrix_digest`, where `matrix_digest = sha256(serialized matrix)`
  is recorded **at the moment the service itself generated that matrix**. Nothing else may create an
  index entry; the index is not in the caller-writable cache directory.
- Lookup order, only after manifest resolution, code-identity verification and payload verification:
  1. compute `certified_input_key` from the verified inputs;
  2. read the index entry from the service store — **absent entry ⇒ miss**, never "accept anyway";
  3. load the cached bytes and require `sha256(bytes) == index[certified_input_key]`;
  4. equal ⇒ hit; unequal ⇒ miss **and** a `CACHE_CONTENT_MISMATCH` diagnostic is persisted (the
     inconsistency is reported, never silently repaired).
- On a miss the service regenerates the matrix from the verified certified bundle and records the new
  `matrix_digest` in the index as part of the same certified act.
- `cache_dir` is a location, not authority: pointing it elsewhere can only cause misses. Injecting
  bytes there cannot produce a hit, because no caller can create the authoritative index entry that
  would admit them.

## 7. Production entry-point inventory

| Entry point | Role under this design | Evidence revalidated |
| --- | --- | --- |
| `certified_service` (process) | the only certification act: resolve → verify code → verify payloads → generate/load → decide → persist | all of §2 |
| `scripts/run_four_gw_decision.py` | client: serialized request → service → report | via the service |
| `scripts/build_manager_packet.py` | client: serialized request → service → packet output + run-id manifest report | via the service + exact run-ID report check |
| `route_optimizer.optimize` / `build_event_worlds` | internal to the service; no matrix parameter; issuer, registry and closure deleted | inside the service before any world exists |
| `route_comparator.compare_routes` | internal to the service | idem |
| `finalist_refinement` (refine / escalate) | internal to the service | idem |
| `route_stability` (ladder / gate) | internal to the service | idem |
| `free_hit_request_adapter.load_certified_route_worlds` | internal to the service | idem |
| `manager_worlds.build_manager_worlds` | internal to the service | idem |
| warm / content cache | internal, `cache_identity`-verified | §6 |
| chip paths (Free Hit, manager packet) | clients or internal consumers as above | via the service |

## 8. `build_manager_packet.py`

Keeps its Repair-3 repair; becomes a client of the service: build the serialized request from CLI
arguments, invoke the service, print the packet plus the `manifest_id` and the per-family run ids and
payload digests actually consumed, and fail closed on any mismatch. A real CLI/output regression test
runs the command end to end and asserts the reported run ids and digests equal the manifest pins.

## 9. What survives from `41eef48`, and what is removed

**Survives:** required model-version enforcement; horizon certification; PE-8 evidence integration (no
promotion); missing-upstream refusal (never a silent zero); data-snapshot validation;
`validate_certified_bundle`; `certify_horizon_bundles` (extended to append manifest events);
`certified_bundle_from_explicit_ids`; the `canonical_bundle_identity` *function* (a pure content
function over boundary-derived content); the `build_manager_packet` repair; the adversarial test
infrastructure and the 24 PE-9 hard cases; the provenance and bounded-dirty-work gates.

**Removed:** `ValidatedCertificationArtifact` and its `__closure__`-captured mint token;
`_ISSUED_WORLD_MATRICES`; `_issue_world_matrix` / `_issue` and the `build_event_worlds` closure;
`require_certified_prebuilt_matrix`; `IssuedWorldMatrix`; `prebuilt_worlds` on every production
signature; `certification: Mapping` parameters on production signatures; matrix-returning production
surfaces; in-process results being treated as certified; the bypass-register entries describing those
mechanisms.

**Added:** the certified service process; the append-only event-sourced manifest store with payload
digests; the code-identity gate (after manifest resolution, against the manifest's snapshot); the
service-owned cache index keyed by `certified_input_key`; the non-production replay interface.

**Renamed/moved:** `NonProductionWorlds` → `ReplayWorlds` in `fpl_brain/nonproduction_worlds.py`,
reachable only from `*_replay_only` entry points and refused by the service for production kinds.

## 10. Adversarial test matrix (design-level refutation)

| Attack | Why the design refuses it |
| --- | --- |
| self-consistent raw artifact, incomplete contract | authority derives from the manifest; a presented artifact is never an input |
| valid artifact later mutated | no artifact object is an input |
| alternate run IDs inserted into a previous artifact | the manifest pins run ids; inserted ids contradict it |
| `isinstance`-compatible fake | no `isinstance` check exists on the authority path |
| constructor/token via `__closure__` | no minting constructor and no closure-captured secret exist |
| content + caller digest changed together | no caller digest is consulted |
| predictive payload changed beneath unchanged metadata | payload digests are pinned and re-hashed before use |
| dependency edge changed beneath unchanged metadata | `closure_digest` pins the edges |
| arbitrary prebuilt matrix | the production surface returns no matrix and accepts none |
| copied matrix identity/stamp | no stamp exists; identity is manifest-derived |
| direct former issuer call / registry mutation / closure extraction | those mechanisms are deleted |
| cache injection or byte substitution | the authoritative expectation is the service-owned index entry recorded at generation time; absent entry is a miss, and bytes are hashed against that expectation on read |
| monkey-patching the caller's process | certification happens in the service; the caller supplies only serialized descriptors |
| monkey-patching inside the service | code-identity gate vs the manifest's `code_snapshot` |
| superseded/revoked manifest used for production | effective state computed at resolution; revoked refused; superseded is replay-only |
| `EXACT_ID` in a disallowed state | refused by mode/effective-state check |
| current rows mutated after certification | `HISTORICAL_EVIDENCE_DRIFT`; manifest identity unchanged |
| canonical runs + valid manifest via the service | ACCEPTED |
| legitimate cache hit for the same certified world | ACCEPTED |
| Free Hit canonical path | ACCEPTED |
| `build_manager_packet` real CLI path | ACCEPTED |
