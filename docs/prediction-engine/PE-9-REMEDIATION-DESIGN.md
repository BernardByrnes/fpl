# PE-9 Remediation Design — certification reproduced from persisted evidence

**Status:** PROPOSED DESIGN — for Sol High adversarial design review. No implementation is authorised
until that review reports no P1/P2 architectural blocker (see `PE-9-CERTIFICATION-AMENDMENT-1.md` §12).
**Revision 2** — addresses Sol High design review 1: (P1) the trust boundary is now a *process*
boundary and monkey-patching is no longer excluded from the threat model; (P1) an immutable,
versioned certification manifest now selects and pins the certified world instead of "matching
current rows".
**Base:** `41eef48d7cf8d68ffaeeb54fdb9412cd0c04497d` (repair-3 candidate, tree `5f98bb9e…`).
**Scope:** certification/integration only. No predictive quantity, model, RNG, scoring, chip,
transfer, horizon or calibration change.

## 1. The one boundary

All production predictive loading funnels through a single module, `fpl_brain/certification_boundary.py`,
which is the **only** place certification is established. Modules do not implement their own
certification logic; they call this boundary.

```python
@dataclass(frozen=True, slots=True)
class ManifestPolicy:
    """Which immutable certification manifest a request is resolved against."""
    kind: Literal["LATEST_CERTIFIED_FOR_EVENT_SET", "EXACT_ID"]
    manifest_id: str | None = None          # required for EXACT_ID (replay / reproducibility)

@dataclass(frozen=True, slots=True)
class CertificationRequest:
    """Descriptors only: what the caller wants to decide, never what it claims is certified."""
    event: int
    horizon: tuple[int, ...]                # exact event set, e.g. (e, e+1, e+2, e+3)
    required_versions: tuple[tuple[str, str], ...]
    planning_context_hash: str
    code_snapshot: str
    data_snapshot: str
    policy: ManifestPolicy = ManifestPolicy("LATEST_CERTIFIED_FOR_EVENT_SET")

@dataclass(frozen=True, slots=True)
class CertificationManifest:
    """The immutable authority record. Append-only; never updated in place."""
    manifest_id: str                         # content identity of the pinned facts below
    event_set: tuple[int, ...]
    cutoff_by_event: Mapping[int, str]
    runs_by_event: Mapping[int, Mapping[str, RunPin]]   # event -> family -> pinned run facts
    required_versions: Mapping[str, str]
    dependency_closure: Mapping[str, Mapping[str, int | None]]
    planning_context_hash: str
    code_snapshot: str
    data_snapshot: str
    bundle_identity_by_event: Mapping[int, str]
    pe8_evidence: Mapping[str, Any] | None
    certified_at: str
    certifying_actor: str
    status: Literal["CERTIFIED", "SUPERSEDED", "REVOKED"]

@dataclass(frozen=True, slots=True)
class CertifiedEvidence:
    """What the BOUNDARY re-derived and reconciled against the manifest. Data, not authority."""
    request: CertificationRequest
    manifest: CertificationManifest
    reconciled: Mapping[int, Mapping[str, RunFacts]]    # manifest facts vs current rows
    drift: tuple[str, ...]                              # any historical-evidence drift, if observed
    identity: str                                       # recomputed; never accepted from a caller

def resolve_manifest(conn, request: CertificationRequest) -> CertificationManifest: ...
def open_certified_evidence(conn, request: CertificationRequest) -> CertifiedEvidence: ...
def certified_worlds(conn, evidence: CertifiedEvidence, *, event: int, union_ids, config,
                     cache_dir: Path | None = None) -> tuple[Matrix, Mapping[str, Any]]: ...

def certified_decision(conn, request: CertificationRequest, *, kind: DecisionKind, **options) -> DecisionResult:
    """The production entry point: certify, generate, decide, return DECISIONS + evidence.

    The world matrix never leaves this call. This function is what production callers invoke; it is
    the only supported path to a certified decision.
    """
```

The boundary takes **descriptors + a live `sqlite3.Connection`**, and nothing else. No function in the
design accepts a matrix, and none accepts an object that claims to be certified.

## 2. Data flow (production)

```
production CLI / module entry point (e.g. scripts/run_four_gw_decision.py, scripts/build_manager_packet.py)
        |
        |  builds CertificationRequest from its own arguments (event, horizon, versions,
        |  planning context, snapshots) - never from a caller-supplied object graph
        v
certified_decision(conn, request, kind=...)                    <- the ONLY certification act
        |   0. CODE IDENTITY GATE: verify the certification modules actually loaded in THIS process
        |      against request.code_snapshot (source digests + loaded code objects);
        |      mismatch -> CertificationRefused("CERTIFICATION_CODE_IDENTITY_MISMATCH")
        |   1. resolve_manifest(conn, request): the immutable manifest for this event set
        |      (LATEST_CERTIFIED_FOR_EVENT_SET, or EXACT_ID for reproducibility)
        |   2. reconcile manifest facts against current rows: every pinned run's id, family,
        |      version, status, cutoff, event and dependency edge must still hold; any drift is
        |      reported and refused (HISTORICAL_EVIDENCE_DRIFT) rather than silently re-resolved
        |   3. recompute the certified bundle identity from the MANIFEST's pinned facts
        |      (not from "newest matching rows") and require it to equal the manifest identity
        |   4. generate or load (cache) the world matrix internally, keyed by that identity
        |   5. run the requested decision (search / comparator / refinement / stability / Free Hit)
        v
DecisionResult (decisions, identities, manifest_id, evidence)  -> persisted by the entry point
```

`certified_worlds` remains available to **internal** boundary consumers (the decision steps inside
`certified_decision`) and to non-production callers, but no production path returns a matrix to a
caller: the production surface hands back decisions and their evidence.

## 3. Why the design resists each class of attack

### 3.1 Caller-supplied objects, types, digests, capabilities

`CertificationRequest` and `CertifiedEvidence` are caller-constructible frozen dataclasses, and that
is harmless because **every entry point re-derives from the database and the manifest**:

- `resolve_manifest` ignores any supplier of a manifest: it reads the append-only manifest store and
  selects by policy; `EXACT_ID` must name a manifest that exists.
- `certified_decision` re-derives run facts from the manifest's pins and reconciles them against the
  current rows. A hand-built evidence record contributes only descriptors; its `identity` is never
  consulted, and its claimed facts must match what the boundary derives.
- There is no `isinstance` check, no minting constructor, no registry, no stamp and no caller digest
  anywhere on the authority path, so there is nothing to impersonate, copy or extract.

### 3.2 Reusing a valid evidence record with a different matrix

The matrix is not carried in the evidence record and is not an input. `certified_decision` generates
or loads it internally from the manifest-pinned bundle, so a caller holding a perfect evidence record
still receives only the boundary's own matrix — or, on the production surface, only decisions.

### 3.3 Monkey-patching (no longer excluded)

Monkey-patching is in the threat model, and the design answers it structurally rather than by
secrecy:

- the **certified production surface returns decisions, not matrices**, and performs certification,
  matrix generation and decision consumption inside one call, so patching a *caller* cannot inject a
  matrix into the certified path — the certified path never reads a matrix from its caller;
- the **code-identity gate** (step 0) verifies the certification modules actually loaded in the
  certified process against the manifest's recorded `code_snapshot` — source bytes and loaded code
  objects — and refuses on mismatch, so an in-process patch of the certification code is detected
  before any certification occurs;
- every decision records the `manifest_id`, the resolved code snapshot and the evidence used, so a
  tampered process is auditable after the fact.

### 3.4 The stated residual

An adversary able to execute arbitrary code **inside the certified process before or during
certification** — including patching the code-identity gate itself — cannot be contained by any
in-process design. The amendment's response is to make that adversary tamper with the certified
process, not merely call a library function: the process boundary, the manifest and the gate mean the
only producible certified decisions come from the authorised entry point running authorised code over
authorised evidence. The trust root is therefore explicit: **the persisted manifest store and the
code identity of the certified entry point.** Both are auditable, and neither is a Python object
handed around at run time.

## 4. Certified prebuilt matrices: removed, with no replacement

`route_optimizer.optimize` and every other production entry point lose `prebuilt_worlds`,
`certification` (a raw mapping), `IssuedWorldMatrix`, `require_certified_prebuilt_matrix`,
`_issue_world_matrix`, `_issue`, `_ISSUED_WORLD_MATRICES`, and the closure that exposed the issuer.
No issuer, registry, stamp, token or capability replaces them — the design deliberately has nothing
to steal.

Non-production needs (tests, historical replay, benchmarking) are served by a separate module,
`fpl_brain/nonproduction_worlds.py`, exporting an explicitly named interface:

```python
REPLAY_ONLY_DECLARATION = "NON_PRODUCTION_REPLAY_ONLY"

@dataclass(frozen=True, slots=True)
class ReplayWorlds:
    declaration: str = REPLAY_ONLY_DECLARATION        # must equal the constant, else refused
    matrices: Mapping[int, Any] = field(default_factory=dict)

def optimize_replay_only(*, worlds: ReplayWorlds, ...) -> dict: ...
def compare_routes_replay_only(*, worlds: ReplayWorlds, ...) -> dict: ...
```

Rules that keep it out of production:

- production functions never accept `ReplayWorlds` (no such parameter, and the boundary cannot
  consume one — it takes descriptors and a connection);
- the replay entry points are separate functions, not flags on the production ones, so no production
  call site can reach them by argument;
- every replay result is stamped `NON_PRODUCTION_REPLAY_ONLY`, and the decision/evidence writer
  accepts only results produced by `certified_decision` (which carry a resolved `manifest_id`);
- tests assert the production signatures contain no matrix-valued parameter and that calling a
  production entry point with `ReplayWorlds` raises `TypeError`/`CertificationRefused`.

## 5. Immutable manifest lifecycle

- **Creation:** `certify_horizon_bundles` / `scripts/certify_gw5_gw8.py` derive the certified world
  from persisted runs, compute the manifest identity, and **append** the manifest with status
  `CERTIFIED`; the store rejects update-in-place and duplicate ids.
- **Supersession:** a newer certification for the same event set appends a new manifest; the older
  one becomes `SUPERSEDED` but remains readable, so a historical decision can still be reproduced by
  `EXACT_ID`.
- **Use:** live decisions resolve `LATEST_CERTIFIED_FOR_EVENT_SET`; replay and audit resolve
  `EXACT_ID`. Every decision records which one it used.
- **Drift:** revalidation compares the manifest's pinned run facts with the current rows. Any
  difference is refused and surfaced (`HISTORICAL_EVIDENCE_DRIFT`); nothing is silently re-resolved
  and no historical identity is rewritten by later mutation of current tables.

## 6. Cache contract

Caching remains, entirely inside the boundary:

1. `certified_decision` resolves and reconciles the manifest first — this happens on every call;
2. the cache key **is** the manifest-derived bundle identity;
3. a hit is read from the cache directory and its content digest must equal that identity;
4. a mismatch (stale, tampered, wrong event) is a miss → regenerate from the certified bundle; a
   mismatch is never "repaired" silently.

No function accepts a cache object, a matrix, or a registry entry. `cache_dir` is a location, not
authority: pointing it elsewhere can only cause misses or *verified* hits.

## 7. Production entry-point inventory

| Entry point | Receives | Evidence revalidated | Refusal token |
| --- | --- | --- | --- |
| `certified_decision` (boundary) | request + conn | manifest resolved, reconciled, identity recomputed, code-identity gate | `CERTIFICATION_*` |
| `scripts/run_four_gw_decision.py` | CLI args → request → `certified_decision` | idem; returns decisions + evidence | idem |
| `scripts/build_manager_packet.py` | CLI args → request → `certified_decision` | idem + exact run-ID report validation | idem |
| `route_optimizer.optimize` / `build_event_worlds` | evidence from the boundary call; no matrix parameter; issuer/closure deleted | revalidated inside the boundary before any world exists | `CERTIFICATION_EVIDENCE_MISMATCH` |
| `route_comparator.compare_routes` | idem | idem | idem |
| `finalist_refinement` (refine / escalate) | idem | idem | idem |
| `route_stability` (ladder / gate) | idem | idem | idem |
| `free_hit_request_adapter.load_certified_route_worlds` | evidence + event set | idem | idem |
| `manager_worlds.build_manager_worlds` | manifest-pinned bundle + run ids | idem | idem |
| warm / content cache | internal | identity-verified on read | `CACHE_CONTENT_MISMATCH` |
| chip paths (Free Hit, manager packet) | as above | idem | idem |

## 8. `build_manager_packet.py`

Keeps its Repair-3 repair and gains the boundary call: build a `CertificationRequest` from the CLI
arguments, call `certified_decision` (or `open_certified_evidence` + `certified_worlds` for the
packet's non-decisional output), print the per-family run IDs actually consumed and the `manifest_id`
used, and fail closed on any mismatch. A real CLI/output regression test runs the command end to end
and asserts the reported run IDs equal the manifest-pinned ones.

## 9. What survives from `41eef48`, and what is removed

**Survives:** required model-version enforcement; horizon certification; PE-8 evidence integration
(no promotion); missing-upstream refusal (never a silent zero); data-snapshot validation;
`validate_certified_bundle`; `certify_horizon_bundles` (extended to append manifests);
`certified_bundle_from_explicit_ids`; the `canonical_bundle_identity` *function* (demoted to a pure
content function over boundary-derived content); the `build_manager_packet` repair; the adversarial
test infrastructure and the 24 PE-9 hard cases; the provenance/bounded-dirty-work gates.

**Removed:** `ValidatedCertificationArtifact` and its `__closure__`-captured mint token;
`_ISSUED_WORLD_MATRICES`; `_issue_world_matrix` / `_issue` and the `build_event_worlds` closure;
`require_certified_prebuilt_matrix`; `IssuedWorldMatrix`; `prebuilt_worlds` on every production
signature; `certification: Mapping` parameters on production signatures; matrix-returning production
library surfaces (production returns decisions); the bypass-register entries describing those
mechanisms.

**Added:** the manifest store and its lifecycle; the code-identity gate; `certified_decision` as the
production surface; `resolve_manifest` / `open_certified_evidence` / `certified_worlds` as internal
boundary API.

**Renamed/moved:** `NonProductionWorlds` → `ReplayWorlds` in `fpl_brain/nonproduction_worlds.py`,
reachable only from `*_replay_only` entry points.

## 10. Adversarial test matrix (design-level refutation)

| Attack | Why the design refuses it |
| --- | --- |
| self-consistent raw artifact, incomplete contract | never an input: authority derives from the manifest, not from a presented artifact |
| valid artifact later mutated | same: no artifact object is an input to the authority path |
| alternate run IDs inserted into a previous artifact | the manifest pins the run IDs; inserted IDs contradict it |
| `isinstance`-compatible fake | no `isinstance` check exists on the authority path |
| constructor/token via `__closure__` | no minting constructor and no closure-captured secret exists |
| content + caller digest changed together | no caller digest is consulted; identity is recomputed from the manifest |
| arbitrary prebuilt matrix | no production signature accepts a matrix; the production surface returns decisions |
| copied matrix identity/stamp | no stamp exists; identity is manifest-derived |
| direct former issuer call | the issuer is deleted |
| registry mutation | the registry is deleted |
| closure issuer/token extraction | no closure issuer exists |
| cache injection | cache content is verified against the manifest-derived identity |
| monkey-patched boundary/loader in the caller's process | the certified path never reads a matrix from its caller; certification, generation and decision happen inside the boundary call |
| monkey-patched certification code inside the certified process | refused by the code-identity gate against the manifest's `code_snapshot` |
| current rows mutated after certification | `HISTORICAL_EVIDENCE_DRIFT` refusal; manifest identity unchanged |
| two manifests for one event set | distinguished by id; the decision records the one used |
| canonical persisted runs + valid manifest | ACCEPTED (happy path) |
| legitimate internal cache hit | ACCEPTED (identity-verified) |
| Free Hit canonical path | ACCEPTED |
| `build_manager_packet` CLI | ACCEPTED |
