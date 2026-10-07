# FH/WC prospective evidence — implementation checkpoint

Status: bounded FH/WC implementation and focused verification complete in the isolated worktree; root review remains pending and no commit was made.

## Source and boundary

- Accepted source anchor: commit `0e01d3ea1bbeb4a65dd562a533b297a6f2e16007`, tree `1f59e6d3cd4fb380e217496b2087e5eb802fbc02`.
- Current worktree base: commit `b131075e9f63eaa5a253ee56603cd921ddbaa556`, tree `b8f453e387febe4447ed76fe9d65fbdc332c524c` (child of the accepted source with the already-approved BB/TC collector/finality commits).
- Worktree: `C:\Users\USER\agent-workspaces\fpl-wc-fh-evidence-orchestration`.
- `K:\FPL`, production/live DB/config, reports, accounts and original evidence were not accessed or modified. Integration data and artifacts are isolated under pytest temporary directories.
- No operational/production world generation, normal decision, calibration, readiness, account action, full suite, CI, Sol review, merge, commit or push was run. Test setup explicitly materialized synthetic fixture cache entries via `route_optimizer.build_event_worlds(..., allow_materialization=True)` before the public-composer no-simulation spies were installed. Those calls were confined to pytest temporary directories; no public FH/WC composer materialized a missing cache and no operational cache was created.
- Exact tracked patch source is the current unstaged `git diff HEAD`; new files are listed below. This checkpoint records the parent/tree and the SHA-256 of each implementation/test/doc file so the exact state can be verified.

## Implemented behavior

FH/WC collection is additive over the accepted BB/TC causal workflow:

- `fpl_brain/chip_horizon_evidence_assembly.py` composes typed future FH/WC requests only from an explicit verified normal origin, exact continuation/value generations and verified cache hits. It does not build missing matrices. FH and WC preserve their accepted scoring/evaluator calls; WC keeps its six-to-ten-event valuation horizon separate from its four-event chip-decision slice.
- `fpl_brain/chip_horizon_evidence_ops.py` adds action-specific pinned chip-window registration, prospective multi-opportunity forecast retention, strict preflight/calculation/issuance availability checks, per-event outcome receipt verification, and full-horizon maturation manifests/recovery.
- `fpl_brain/chip_evidence_ops.py` shares FH/WC action binding with the existing capture/maturation path while keeping BB/TC player extraction and their one-event lifecycle unchanged. FH/WC require capture availability and official finality strictly after `forecast_available_at`; the original raw `forecast_made_at` remains unchanged. The accepted calibration validator/schema was not widened.
- Each FH/WC maturity receipt seals exact event receipt references and receipt/file digests, operation and fetch IDs, required event-player unions, receipt capture digests and a separate selected-row manifest. A later plausible but unbound ledger row cannot stand in for a receipt. Retries reverify exact time, receipt, schedule and digest bindings; missing, extra, replaced or tampered bindings refuse. The deterministic maturation key remains `(observation_id, selected_event)`.
- `scripts/chip_evidence.py` now has explicit `register-horizon-origin` and cache-only `horizon-forecast` commands. Existing capture/mature commands accept FH/WC as well as BB/TC. WC CLI input requires an explicit value-generation ID for every opportunity event. The command assembles/verifies the coverage product but does not certify products, generate predictions, rerun decisions or materialize worlds.
- Registration, forecast and maturation use read-only/query-only DB connections. Capture is the only CLI logical DB writer and appends official `player_gameweeks_final` event-grain rows through the existing SQLite writer transaction; it performs no migration or FPL refresh.
- `docs/chip-evidence-operations.md` documents the implemented commands, exact upstream blockers, strict timing/finality, retry/receipt behavior and stop conditions. It does not claim that FH/WC are available through the existing BB/TC-only calibration handoff.

### GW6 WC free-transfer semantics

The actual mismatch was `WildcardRequest.event_start_free_transfers=3` versus the accepted `WildcardSaveRouteEvent.free_transfers=4` at GW6. These are not competing facts: the first is event-start FT from the canonical route transition's `before_state`; the route DTO's `free_transfers` is its post-event route value after rollover. The request is not rewritten to 4, and no accepted scorer/producer/state rule was changed. The additive assembler now proves `before_state == projected target state`, retains separate `save_route_start_event_state` and `save_route_target_event_dto_free_transfers` fields, and the operator layer checks the request against that canonical pre-event binding. The producer reservation state continues to be built exactly from its verified SAVE DTO. Regression asserts GW6 start FT 3 and DTO value 4 separately.

## Focused validation

Native results from the isolated worktree:

| Command | Result |
|---|---|
| `python -m pytest -q tests/test_chip_evidence_ops.py::test_capture_retry_recovers_receipt_without_duplicate_rows_and_later_capture_is_distinct tests/test_chip_evidence_ops.py::test_canonical_bb_tc_origin_forecast_final_capture_and_maturation_lifecycle` | 3 passed in 82.66s; existing BB/TC extraction/lifecycle remains passing. |
| `python -m pytest -q tests/test_chip_horizon_evidence_assembly.py::test_public_future_fh_composer_requires_cache_then_reconstructs_routes` | 1 passed in 140.70s before the independent-review follow-ups described below. It covered final/missing/unbound event evidence, strict time, receipt/source/action/timing tamper, full four-event schedule, changed/extra receipt refusal, and recovery of the immutable matured pair without finalizing again. |
| `python -m pytest -q tests/test_chip_horizon_evidence_assembly.py::test_public_future_wc_composer_refuses_missing_cache_without_materializing_worlds` | 1 passed in 83.17s before the independent-review follow-ups. Covers two opportunity events (GW5/GW6), six-event value schedules, no materialization by the public composer on a miss, successful full schedule capture/maturation/retry, exact GW6 start FT 3 vs post-event DTO FT 4, and real CLI execution against an existing temporary on-disk DB. |
| `python -m pytest -q tests/test_chip_evidence_cli.py` | 18 passed in 0.34s. Includes actual command validation for duplicate WC event mappings plus explicit-argument and readonly routing checks. |
| `python -m compileall -q fpl_brain scripts tests/test_chip_horizon_evidence_assembly.py tests/test_chip_evidence_cli.py` | Exit 0. |
| `git diff --check` | Exit 0. Git emitted only its configured LF-to-CRLF notice for the rewritten operations guide. |

Final independent-review follow-up native results (same isolated temporary fixtures):

| Command | Result |
|---|---|
| `python -m pytest -q tests/test_chip_horizon_evidence_assembly.py::test_public_future_fh_composer_requires_cache_then_reconstructs_routes` | 1 passed, native exit 0. This is the strengthened FH test with coherent finality/availability equality archives, the newer unbound selected row, partial-pair refusal/recovery, and no-finalizer checks. |
| `python -m pytest -q tests/test_chip_horizon_evidence_assembly.py::test_public_future_wc_composer_refuses_missing_cache_without_materializing_worlds` | 1 passed, native exit 0. In addition to the original WC flow, it now retains a second verified FH forecast at the same observation and selected event and proves a real cross-action maturation-key collision refusal. |
| `python -m compileall -q fpl_brain scripts tests/test_chip_horizon_evidence_assembly.py` | Exit 0 after the final test changes. |
| `git diff --check` | Exit 0 after the final test changes. A separate trailing-whitespace scan of the untracked checkpoint/test returned no matches; Git emitted only its existing LF-to-CRLF notice for `docs/chip-evidence-operations.md`. |

The final focused pass set is 23 passing cases across the preserved BB/TC lifecycle (3), CLI tests (18), strengthened FH lifecycle (1) and final WC lifecycle (1). This is not a full-project test result. No full pytest suite, wrapper, CI or Sol review was run.

The WC integration invokes `register-horizon-origin`, `horizon-forecast`, `capture-outcome` and `mature` against the real existing temporary SQLite file. It records `PRAGMA schema_version` and `schema_meta.schema_version` before/after and proves both unchanged; official/test DB rows are not migrated. Capture appends event-grain rows and an identical capture retry does not increase the ledger count. Forecast, registration and maturation operate through actual readonly connections. No production DB is involved.

### Independent-review follow-ups

- Public FH capture tests now construct coherent retained archives for official-finality equality and capture-availability equality. Since a valid archive enforces `official_final_at <= available_at`, availability equality necessarily also has non-later official finality and is rejected at the earlier finality guard; a valid public archive cannot isolate the later availability-only guard.
- The FH integration now inserts a genuinely archived, newer FINAL player-event ledger row with actual payload digest, archive capture ID, and fetch-run provenance but no capture-operation receipt. It verifies that this current row supersedes the older receipt-bound row, maturation refuses before finalization, and no artifacts/receipt are written or changed. A later proper capture receipt restores a matureable state.
- From the same fully captured temporary FH scenario, the test now creates one valid immutable outcome/evidence pair, temporarily parks one member to model an ambiguous partial pair, and verifies refusal without a second finalizer call or artifact mutation. Restoring the pair allows receipt recovery without rerunning the finalizer.
- The WC integration now registers two independently pinned test-source chip windows (FH through GW5; WC through GW6), retains both valid forecasts with the same observation ID and selected GW5, verifies both action-specific forecast loaders and complete capture manifests, and matures WC first. The legitimate competing FH maturation then refuses against the occupied deterministic `(observation_id, realization_event)` receipt key because its action binding differs. The finalizer spy remains uncalled and all retained outcome/evidence/receipt bytes remain unchanged. No production code, scoring, schema or key semantics changed.
- The tests' synthetic cache preparation is explicitly distinguished from operational cache materialization: `_materialize_synthetic_fixture_caches` invokes `route_optimizer.build_event_worlds(..., allow_materialization=True)` only in pytest temporary fixture directories, before composer materialization spies are installed. The public FH/WC composer paths themselves remain cache-only.

The valid-source timing cases are intentionally honest: for public archives, `official_final_at <= available_at`. Therefore an archive with `available_at == forecast_available_at` is also at-or-before the forecast in official-finality time and is rejected by the earlier finality guard. Reaching only the later availability equality guard would require impossible archive chronology, so no such source was fabricated.

No full project pytest suite or CI run was performed; the authorized checks were focused only.

## Observed test-only failures and corrections

1. An earlier GW6 check incorrectly compared pre-event request FT 3 directly with the post-event SAVE DTO FT 4. This was rejected as a false equality; the separate canonical `before_state` binding described above is now tested.
2. An early capture fixture's archived synthetic bootstrap had only `events`; official archive validation correctly rejected missing required `elements`/`teams`. The temporary helper now retains credible element IDs/web names and team IDs/names. Latest WC end-to-end test passes.
3. The FH test inserted a synthetic unbound ledger row but left its transaction open, so the next real collector transaction correctly failed with `cannot start a transaction within a transaction`. The fixture now commits that deliberately pre-existing row before testing refusal; no production transaction semantics changed.
4. The FH test's final extra-receipt assertion expected a longer error substring than the correct refusal message. The assertion now matches `does not preserve forecast timing and the verified capture manifest`; the complete focused test passes.
5. An initial CLI test expected `parse_args()` itself to reject duplicate `--value-generation` events although duplicate semantics are command validation. It now invokes the actual command boundary and verifies the refusal.
6. The first review-boundary run collided two synthetic bootstrap archive observations at the same second (`RawArchiveCollision`); the archive correctly refused replacement. The temporary helper now advances its default observation time beyond the newest retained observation. Explicit equality timestamps are unchanged.
7. One baseline assertion selected the first receipt for the late event, which was the intentionally incomplete run, instead of the complete-run receipt. It now selects the exact receipt by fetch-run ID before comparing the selected ledger row's operation binding.
8. A test-local missing `candidate_universe` import caused `NameError`; the function now imports it locally. No production code changed.
9. The first cross-action fixture expected FH expiry GW5 while the temporary pinned chip definition said GW6. The fixture now seeds distinct, internally consistent windows before creating the generation snapshot: FH through GW5 and WC through GW6.
10. With both synthetic chip windows initially extending through GW6, the canonical FH evaluator correctly refused the GW6 toy frontier (`FREE_HIT_OPTIMIZER_FRONTIER_EMPTY`). The test was bounded to a valid one-event FH opportunity through GW5 while retaining WC's GW5–GW6 window. This is fixture scope, not a producer/scoring defect; no product repair was proposed or applied.

The independent supervisor retained the failed-attempt timestamps, call identities and bounded output tails in `K:\zcode-orchestrator-bridge\evidence\fpl_fhwc_bounded_native_supervisor_receipts_20261004T1757Z.json`; these fixture failures were not rerun solely to recover logs.

## Remaining operational limitations

- GW6 is permanently ineligible as the first prospective calibration origin because no qualifying forecast was retained before its outcomes. The accepted GW6 hypothetical assessment and its cache remain assessment-only.
- The original event-start FT remains unknown. The test fixture's known FT is not a confirmation or write to operational evidence.
- First valid FH/WC capture still requires a fresh eligible origin before outcomes, timely known event-start FT, accepted search permission, pinned unused chip/expiry, exact same-origin `CHIP_RESERVATION` continuation, WC `WILDCARD_VALUE` products where applicable, full official player/run/config/world compatibility, and all required cache hits. This work created none of those operational products.
- If a required world matrix is missing, `horizon-forecast` refuses; any preparation/materialization must be separately authorized. No such computation was run here.
- `load_verified_calibration_handoff` remains BB/TC-only. FH/WC receipts/audit rows are retained, but are not yet accepted into calibration intake/training/readiness. No calibration, evaluator permission or PLAY/SAVE endorsement is established.
- The earliest possible future origin is the first later planning event whose fresh origin and every prerequisite pass before all relevant events start. GW7 is only a candidate if its own deadline and point-in-time checks still permit a prospective issue; this checkpoint does not assert that it does.

## Current worktree files

Implementation/docs/tests changed or added in this worktree:

- `fpl_brain/chip_evidence_ops.py`
- `fpl_brain/chip_horizon_evidence_assembly.py` (new)
- `fpl_brain/chip_horizon_evidence_ops.py` (new)
- `scripts/chip_evidence.py`
- `docs/chip-evidence-operations.md`
- `tests/test_chip_horizon_evidence_assembly.py` (new)
- `tests/test_chip_evidence_cli.py`
- `docs/chip-evidence-fhwc-checkpoint-2026-10-04.md` (this checkpoint)

The worktree also has previously accepted dirty cache-reader changes in `fpl_brain/route_optimizer.py` and `tests/test_route_optimizer.py`; this follow-up did not alter them. All changes remain unstaged/uncommitted. Final targeted test total: 23 passed across the four focused invocations above. The full project suite remains intentionally unrun.

Relative to the prior checkpoint, this review follow-up changed only `tests/test_chip_horizon_evidence_assembly.py` and this checkpoint document. No production source was changed. The final focused test total above excludes historical pre-follow-up FH/WC passes and is not a claim that the full suite passed.

Current checkout remains detached at `b131075e9f63eaa5a253ee56603cd921ddbaa556` / tree `b8f453e387febe4447ed76fe9d65fbdc332c524c`; all ten listed implementation/docs/test paths remain unstaged and uncommitted. The full final `git status --short` was: modified `docs/chip-evidence-operations.md`, `fpl_brain/chip_evidence_ops.py`, `fpl_brain/route_optimizer.py`, `scripts/chip_evidence.py`, `tests/test_chip_evidence_cli.py`, `tests/test_route_optimizer.py`; untracked `docs/chip-evidence-fhwc-checkpoint-2026-10-04.md`, `fpl_brain/chip_horizon_evidence_assembly.py`, `fpl_brain/chip_horizon_evidence_ops.py`, `tests/test_chip_horizon_evidence_assembly.py`. Only the test and checkpoint document changed during this follow-up.

## SHA-256 source manifest

Hashes are for the final files in this checkpoint; the checkpoint file itself is intentionally omitted because adding its own digest would invalidate it.

| File | SHA-256 |
|---|---|
| `fpl_brain/chip_evidence_ops.py` | `A8F724DCC340AD1D039BCE62E476E3F34B8157F253608C805E2E5BA819D82A21` |
| `fpl_brain/chip_horizon_evidence_assembly.py` | `A4068B4D0D48CF2274D93D372414722A4B4CF04EC0547C6041CE04EA3A543392` |
| `fpl_brain/chip_horizon_evidence_ops.py` | `8124B47EB69CE3040F45CC2B8BB677D86A46E52BB3867C668CBA2B16EA7585B2` |
| `scripts/chip_evidence.py` | `5D22CBE847022CB055E9F6942DC5F2ABF17E54C0938C556244928856780600DE` |
| `docs/chip-evidence-operations.md` | `DA6D888673B9D5B9CF0E044E2DC50D40581BD4050CCE91543D56F352D56297AD` |
| `tests/test_chip_horizon_evidence_assembly.py` | `74D2A123C71F1032E5EFD9B303BD29D45DFE062436C1E99E56563AD7CF00B926` |
| `tests/test_chip_evidence_cli.py` | `2A8199B76A108BD3D92A4E4E8ADA9EA245F29E8A689621EE76B314AFD207C6E0` |
| previously accepted `fpl_brain/route_optimizer.py` | `EC2D3BDBC21741005E81CF3960B188339748EAE599CBAE96A0B0C189B1B0A151` |
| previously accepted `tests/test_route_optimizer.py` | `B929758854DA5E5424C4E174BFB8C0D23EBE8DD899A95224358F5ED078857D9C` |
