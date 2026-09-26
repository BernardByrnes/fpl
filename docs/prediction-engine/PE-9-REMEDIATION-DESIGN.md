# PE-9 Remediation Design — certification reproduced from persisted evidence

**Status:** PROPOSED DESIGN — for Sol High adversarial design review. No implementation is authorised
until that review reports no P1/P2 architectural blocker (see `PE-9-CERTIFICATION-AMENDMENT-1.md` §12).
**Base:** `41eef48d7cf8d68ffaeeb54fdb9412cd0c04497d` (repair-3 candidate, tree `5f98bb9e…`).
**Scope:** certification/integration only. No predictive quantity, model, RNG, scoring, chip,
transfer, horizon or calibration change.

## 1. The one boundary

All production predictive loading funnels through a single module, `fpl_brain/certification_boundary.py`,
which is the **only** place certification is established. Modules do not implement their own
certification logic; they call this boundary and consume what it returns.

```python
@dataclass(frozen=True, slots=True)
class CertificationRequest:
    """Descriptors only: what the caller wants to decide, never what it claims is certified."""
    event: int
    horizon: tuple[int, ...]              # exact event set, e.g. (e, e+1, e+2, e+3)
    required_versions: tuple[tuple[str, str], ...]   # (family, authoritative version) from the model registry
    planning_context_hash: str
    code_snapshot: str
    data_snapshot: str

class CertificationRefused(RuntimeError): ...      # carries a stable token + reasons

@dataclass(frozen=True, slots=True)
class CertifiedEvidence:
    """What the BOUNDARY re-derived from persisted rows. Data, not authority."""
    request: CertificationRequest
    bundle_by_event: Mapping[int, BundleIdentity]     # event -> per-family run ids + versions
    cutoff: Mapping[int, str]
    dependency_closure: Mapping[str, Mapping[str, int | None]]
    snapshot: DataSnapshotIdentity
    pe8_evidence: Mapping[str, Any] | None
    identity: str                                     # recomputed from the above, never accepted from a caller

def open_certified_evidence(conn, request: CertificationRequest) -> CertifiedEvidence: ...
def revalidate_certified_evidence(conn, evidence: CertifiedEvidence) -> CertifiedEvidence: ...
def certified_worlds(conn, evidence: CertifiedEvidence, *, event: int, union_ids, config,
                     cache_dir: Path | None = None) -> tuple[Matrix, Mapping[str, Any]]: ...
```

The boundary takes **descriptors + a live `sqlite3.Connection`**, and nothing else. Matrices are
produced *inside* the boundary (`certified_worlds`) and returned; no function in the design accepts a
matrix, and none accepts an object that claims to be certified.

## 2. Data flow (production)

```
CLI / caller builds CertificationRequest (event, horizon, versions, planning context, snapshots)
        |
        v
open_certified_evidence(conn, request)          <- the ONLY certification act
        |   reject unless, for every event in the horizon and every required family:
        |     * a run row exists with the authoritative version,
        |     * run status is COMPLETE (final, non-provisional),
        |     * cutoff and event match the request exactly,
        |     * dependency closure resolves to the same family/version set,
        |     * planning_context_hash, code snapshot and data snapshot match the request,
        |     * PE-8 evidence identity/state agrees where applicable,
        |     * decision_search_permitted / temporal / finality gates pass;
        |   then recompute the canonical identity from the RE-DERIVED rows
        v
CertifiedEvidence (data)
        |
        v
certified_worlds(conn, evidence, event=..., union_ids=..., config=..., cache_dir=...)
        |   1. revalidate_certified_evidence(conn, evidence)  (re-derives and compares; see §3)
        |   2. identity <- recompute from re-derived rows
        |   3. cache lookup keyed by that identity; on hit verify file content digest == identity
        |   4. on miss: regenerate Monte Carlo worlds from the certified bundle, then store
        v
world matrix (internal)  ->  decision logic (search, comparator, refinement, stability, Free Hit)
```

Every consumer that needs worlds calls `certified_worlds(...)` with the evidence it holds. There is
no other way to obtain a world matrix on a production path.

## 3. Why a fabricated `CertifiedEvidence` gains nothing

`CertifiedEvidence` is caller-constructible (it is a plain frozen dataclass). That is deliberate and
harmless, because **every use re-derives from the database**:

- `certified_worlds` calls `revalidate_certified_evidence(conn, evidence)` first. That call ignores
  the object's `identity` entirely: it re-runs the §2 derivation from `conn` using the object's
  `request` descriptors, then requires the freshly derived run ids, versions, cutoff, closure,
  snapshot identities and PE-8 state to equal what the object records. A hand-built object with
  invented run ids cannot match a derivation the caller does not control.
- Even a **perfectly copied** evidence record yields no advantage: the matrix is then produced by the
  boundary from the *database*, not read out of the object. The object carries descriptors and derived
  facts only, never the matrix. Fabricating one is equivalent to calling the boundary with the same
  descriptors — the outcome is identical and equally certified.
- `identity` is recomputed internally at each use. A digest supplied by a caller is never consulted,
  so "change content and digest together" has no attack surface: there is no caller digest in the
  authority path at all.

The trust root is therefore explicit and stated: **the persisted authoritative evidence in the
database**. A process that can rewrite those rows can mislead the boundary; that is a deliberate
scope boundary (equivalent to signing/HSM territory that the frozen authority does not require), and
it is recorded as a residual in §9 rather than papered over.

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

- production functions never accept `ReplayWorlds` (no parameter of that type, and the evidence
  boundary cannot consume one — it takes descriptors and a connection);
- the replay entry points are separate functions, not flags on the production ones, so no production
  call site can reach them by argument;
- every replay result is stamped `NON_PRODUCTION_REPLAY_ONLY` in its output and is refused by the
  decision layer's persistence path (the decision/evidence writer accepts only results produced by
  production entry points, which carry the boundary's recomputed identity);
- a test asserts the production signature set contains no matrix-valued parameter and that calling a
  production entry point with a `ReplayWorlds` raises `TypeError`/`CertificationRefused`.

## 5. Cache contract

Caching remains, entirely inside the boundary:

1. `certified_worlds` re-derives the certified world (§2) — this happens on every call;
2. the cache key **is** the internally recomputed identity;
3. a hit is read from the cache directory and its content digest must equal that identity;
4. a mismatch (stale, tampered, wrong event) is treated as a miss → regenerate from the certified
   bundle; a mismatch is never "repaired" silently.

No function accepts a cache object, a matrix, or a registry entry. `cache_dir` is a location, not
authority: pointing it elsewhere can only cause misses or *verified* hits.

## 6. Production entry-point inventory (evidence revalidated at each)

| Entry point | Under the amendment it receives | Evidence revalidated | Refusal token |
| --- | --- | --- | --- |
| `route_optimizer.optimize` | `CertificationRequest` (or `CertifiedEvidence` from the same call chain) + conn | inside `certified_worlds` before any world is used | `CERTIFICATION_EVIDENCE_MISMATCH` |
| `route_optimizer.build_event_worlds` | same; the issuing wrapper and its closure are deleted | idem | idem |
| `route_comparator.compare_routes` | same; no `non_production_worlds` on the production signature | idem | idem |
| `finalist_refinement` (refine / escalate) | the evidence already validated by the caller's boundary call; it revalidates before refining | idem | idem |
| `route_stability` (ladder / stability gate) | idem | idem | idem |
| `free_hit_request_adapter.load_certified_route_worlds` | evidence + event set | idem | idem |
| `manager_worlds.build_manager_worlds` | evidence-derived bundle + run ids | idem | idem |
| `scripts/build_manager_packet.py` | CLI args → `CertificationRequest` → boundary | idem + exact run-ID report validation | idem |
| `scripts/run_four_gw_decision.py` | CLI args → `CertificationRequest` → boundary | idem | idem |
| warm / content cache | internal to the boundary | idem | idem |
| chip paths (Free Hit, manager packet) | as above | idem | idem |

## 7. `build_manager_packet.py`

Keeps its Repair-3 repair and gains the boundary call: build a `CertificationRequest` from the CLI
arguments (event, horizon, versions, planning context, snapshots), call
`open_certified_evidence` + `certified_worlds`, print the per-family run IDs it actually consumed,
and fail closed on any mismatch. A real CLI/output regression test runs the command end to end and
asserts the reported run IDs equal the boundary-derived ones.

## 8. What survives from `41eef48`, and what is removed

**Survives:** required model-version enforcement; horizon certification; PE-8 evidence integration
(no promotion); missing-upstream refusal (never a silent zero); data-snapshot validation;
`validate_certified_bundle`; `certify_horizon_bundles`; `certified_bundle_from_explicit_ids` (used by
the boundary and by `scripts/certify_gw5_gw8.py`); the `canonical_bundle_identity` *function*
(demoted to a pure content function that only ever runs over boundary-derived content); the
`build_manager_packet` repair; the adversarial test infrastructure and the 24 PE-9 hard cases; the
provenance/bounded-dirty-work gates.

**Removed:** `ValidatedCertificationArtifact` and its `__closure__`-captured mint token;
`_ISSUED_WORLD_MATRICES`; `_issue_world_matrix` / `_issue` and the `build_event_worlds` closure;
`require_certified_prebuilt_matrix`; `IssuedWorldMatrix`; `prebuilt_worlds` on every production
signature; `certification: Mapping` parameters on production signatures; the bypass-register entries
that describe those mechanisms (the register keeps only paths that still exist).

**Renamed/moved:** `NonProductionWorlds` → `ReplayWorlds` in `fpl_brain/nonproduction_worlds.py`,
reachable only from `*_replay_only` entry points.

## 9. Adversarial test matrix (design-level refutation)

| Attack | Why the design refuses it |
| --- | --- |
| self-consistent raw artifact, incomplete contract | the boundary never accepts an artifact as authority; it derives from rows, so a raw mapping is not even an input |
| valid artifact later mutated | same: no artifact object is an input to the authority path |
| alternate run IDs inserted into a previous artifact | re-derivation is from rows; inserted IDs have no row support |
| `isinstance`-compatible fake | there is no `isinstance` check anywhere in the authority path |
| constructor/token via `__closure__` | there is no minting constructor and no closure-captured secret; a test executes the former attack and asserts the decision path still refuses |
| content + caller digest changed together | no caller digest is consulted; identity is recomputed internally |
| arbitrary prebuilt matrix | no production signature accepts a matrix |
| copied matrix identity/stamp | no stamp exists; identity is recomputed |
| direct former issuer call | the issuer is deleted; a test asserts an equivalent call cannot produce an admitted matrix |
| registry mutation | the registry is deleted |
| closure issuer/token extraction | no closure issuer exists |
| cache injection | cache content is verified against the internally recomputed identity |
| canonical runs + valid evidence | ACCEPTED (happy path) |
| legitimate internal cache hit | ACCEPTED (identity-verified) |
| Free Hit canonical path | ACCEPTED |
| `build_manager_packet` CLI | ACCEPTED |
| mutable current tables after a historical certification | historical certified identity is recomputed from historical rows and cannot be rewritten by later mutations of current tables |

**Explicit residual (not claimed as protection):** a process able to rewrite the authoritative rows
themselves, or able to monkey-patch the boundary module in memory, is outside this contract. The
invariant governs authority smuggled through the supported API — objects, types, digests,
capabilities, registries, closures, matrices, caches — not an adversary with arbitrary code execution
in the process, which no in-process design can contain.
