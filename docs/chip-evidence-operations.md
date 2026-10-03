# Prospective BB/TC evidence collection

This is an operational collector for prospective Bench Boost (BB) and Triple Captain (TC) calibration evidence. It does not generate official facts, predictions, normal decisions, certified generations, or production chip decisions. FH/WC orchestration, calibration orchestration, readiness orchestration, dashboards, and convenience tooling are out of scope.

The collector accepts explicit source identities only. It never resolves a latest generation, decision, route, fetch run, or evidence root. The completed GW6 normal decision and `route_041` hypothetical BB/TC assessment remain closed milestones; neither assessment is a prospective forecast, and the hypothetical assessment/cache cannot be relabelled as one or promoted to production authority. No planning event is declared eligible in advance: the accepted producer requires scored opportunity events to be later than the planning event, and each requested event must still be prospective when a properly bound forecast is issued. Eligibility therefore depends on the exact origin, pinned chip state, certified coverage, search permission, cache/scoring inputs, and official outcome availability passing their checks. The completed GW6 assessment does not establish those conditions either way.

Event-start free transfers remain unknown unless the manager provides a genuine confirmation of the number available at the start of the event before transfers. The collector does not infer, backdate, or fill that field.

## Required origin evidence

Before a first valid prospective origin can be registered and forecast:

1. A freshly captured and verified normal four-event generation, a verified normal decision bound to it, and the exact verified route selected from that decision must exist. Record and pass all three IDs explicitly, together with the route ID, planning event, cutoff, database, entry ID, and evidence root.
2. The accepted production-search permission check for that exact generation must pass. Registration repeats the generation, decision, route, manager-context, pinned snapshot, official chip-state, and permission checks.
3. The pinned official bootstrap/manager snapshot must show exactly one eligible, unused BB or TC chip and its official expiry. The requested action and expiry must match that pinned state.
4. A complete, certified, same-origin `CHIP_RESERVATION` generation must cover each event from planning-event + 1 through the chip expiry. It must retain the normal four-event generation as its exact run-ID prefix, and use the same planning event, cutoff, data snapshot, and predictive-code identity. The forecast producer requires this coverage product even when expiry falls inside the normal four-event window; without it, the accepted forecast builder cannot mark future coverage complete.
5. Events beyond the normal route horizon additionally need verified origin-pinned season rules and the accepted no-transfer continuation model. Any unavailable rules, event bundle, or world matrix leaves the forecast incomplete; no value is imputed.
6. Every required world matrix must bind the certified event run IDs, captured official player pool, configured seed/draw count, and semantic scoring/minutes/bonus/role blocks. A cache filename or file count alone is not proof.
7. At fresh forecast preflight, immediately before scoring, and immediately before publication, every covered event must still classify as `SCHEDULED`, with `events.finished=0`, `events.data_checked=0`, at least one fully identified fixture, and every fixture explicitly `started=0` and `finished=0`. No event-grain outcome capture or canonical completed player-fixture performance may exist. Each fixture must have a known kickoff later than the completed issue time; kickoff is only a timing guard, never a completion signal. Null/unknown kickoff, a started/finished/provisional fixture, stale-at-issuance kickoff, or unavailable source provenance refuses.
8. The live event row and target fixtures must match intact raw official captures from completed `success`/`partial` FPL fetch runs: event status from `bootstrap_static`/`bootstrap-static`, fixture schedule/state from `fixtures`/`fixtures`. Their `updated_at` timestamps must match the corresponding archive observation times and payload fields. These verified source captures may legitimately predate the origin cutoff; the collector applies no invented age threshold and does not require `updated_at >= cutoff`. The fetch interval must be ordered (`started_at <= finished_at`, with `finished_at` after the observed capture and before the check); the accepted collector may timestamp the shared raw capture slightly before the recorded run start. The future-kickoff and official-state checks are repeated at issue time so old unstarted flags cannot authorize a forecast after kickoff.

The accepted code has a library producer for `CHIP_RESERVATION` certification (`generation_store.certify_chip_reservation_generation`) but no dedicated operator CLI to prepare that product. Therefore valid-origin collection cannot begin under the present operator workflow until that product is separately prepared through an approved execution/writer-lease procedure. The collector refuses rather than creating predictions, certifying a missing product, rerunning a decision, selecting latest data, or manufacturing a compatible-looking cache entry. Do not call the library function as an undocumented shortcut.

The accepted GW6 generation/decision and hypothetical calculation do not remove this blocker. In particular, the hypothetical world cache is not promoted to production authority and must not be reused as a forecast cache unless the canonical cache key, full semantic content digest, official player population, seed, draw count, and source-run binding all reproduce for the prospective origin.

## Commands and artifacts

All four commands require the common explicit arguments:

```text
--db <existing-db> --entry-id <entry>
--generation-id <exact-generation-id> --decision-id <exact-decision-id>
--route-id <exact-route-id> --planning-event <event>
--origin-cutoff <ISO-8601-cutoff> --evidence-root <retained-root>
```

Register the immutable origin after upstream requirements are verified:

```powershell
python scripts/chip_evidence.py register-origin `
  --db <db> --entry-id <entry> --generation-id <generation> `
  --decision-id <decision> --route-id <route> --planning-event <event> `
  --origin-cutoff <cutoff> --evidence-root <evidence-root>
```

Retain one prospective BB or TC forecast, with an explicit same-origin coverage generation and cache directory:

```powershell
python scripts/chip_evidence.py forecast `
  --db <db> --entry-id <entry> --generation-id <generation> `
  --decision-id <decision> --route-id <route> --planning-event <event> `
  --origin-cutoff <cutoff> --evidence-root <evidence-root> `
  --action BB --expiry-event <pinned-expiry> --observation-id <unique-id> `
  --continuation-generation-id <same-origin-chip-reservation-generation> `
  --cache-dir <origin-specific-cache-dir>
```

Replace `BB` with `TC` for a Triple Captain observation. Missing or invalid cache entries fail closed before scoring by default. `--materialize-worlds` is a separate explicit opt-in and is allowed only in `<evidence-root>/.chip-evidence/world-cache/<origin-id>`. It invokes the accepted canonical event-world builder for missing matrices and then scores the BB/TC opportunity; it is not a cache-only operation. Treat that as a potentially substantial CPU, memory, and disk task, authorize and schedule it separately, and record its measured duration. It does not create projection runs or a generation.

After official outcomes are final, use the exact successful FPL fetch-run ID and the raw-archive root named by that fetch run:

```powershell
python scripts/chip_evidence.py capture-outcome `
  --db <db> --entry-id <entry> --generation-id <generation> `
  --decision-id <decision> --route-id <route> --planning-event <event> `
  --origin-cutoff <cutoff> --evidence-root <evidence-root> `
  --action BB --observation-id <same-observation-id> `
  --realization-event <forecast-selected-event> --fetch-run-id <successful-run-id> `
  --raw-root <that-run-raw-archive-root>
```

Only after capture receipt verification, mature the same observation/event:

```powershell
python scripts/chip_evidence.py mature `
  --db <db> --entry-id <entry> --generation-id <generation> `
  --decision-id <decision> --route-id <route> --planning-event <event> `
  --origin-cutoff <cutoff> --evidence-root <evidence-root> `
  --action BB --observation-id <same-observation-id> `
  --realization-event <forecast-selected-event>
```

The official refresh itself must use the project's normal FPL workflow, for example `python scripts/fetch_fpl.py --config config.json --summaries all --live --gw <event>`. The collector does not invoke it. `capture-outcome` accepts no provisional result: it requires the exact successful fetch run, its three archived official endpoints (`bootstrap-static`, `fixtures`, `event/<E>/live`), shared archive observation time, successful run completion time, final/data-checked event, and all target-event fixtures finished. `finished_provisional=true` is not an independent veto after those official finality conditions pass. Raw archive observation time and fetch completion/availability time are retained separately; the conservative ledger `captured_at` is the successful fetch's `finished_at`.

The flat evidence root contains immutable content-addressed producer artifacts such as `chip-origin-<digest>.json`, `chip-event-opportunity-<digest>.json`, `chip-reservation-forecast-<digest>.json`, `chip-causal-evidence-<digest>.json`, paired arm files, and matured outcome files. `.chip-evidence/` contains sealed operation receipts/intents/publication records, locks, and the origin-specific world cache. Receipts name flat artifact filenames and include content/identity digests; filenames and digests are rechecked before reuse. Keep the whole root and referenced raw archives together. Do not move or edit individual files after issuance.

The only logical database write in this workflow is `capture-outcome`, which appends event-grain rows to the existing append-only `outcome_observation_captures` table with `source_name=player_gameweeks_final` and `source_identity=player_gameweeks:<E>`. It performs no migration or collector refresh. Register, forecast, and mature use read-only, query-only database connections; they write their operation artifacts only to the explicit evidence root. Maturation validates/reuses the ledger captures and produces retained JSON artifacts; it does not update FPL facts or manager state.

## Timing, finality, and retry contract

- The origin receipt binds the explicit entry, normal generation, decision, route, planning event, cutoff, database path, and evidence root. Re-registration re-verifies that source rather than trusting a pointer.
- Forecast evidence retains the exact input cutoff, snapshot consistency observation time, later snapshot-file availability time, calculation start, completed issue time, action/expiry, evaluator and world identities, policies, source provenance, and uncertainty/diagnostic context. The source snapshot is immutable and remains the point-in-time input.
- A new prospective forecast is refused unless every event through expiry is demonstrably still scheduled and unplayed at preflight, calculation start, and issuance. `planning.event_data_state` is necessary but insufficient: each target fixture's explicit started/finished flags, event-grain outcome ledger, canonical completed-performance rows, future kickoff, and matching intact official source archives are checked as well. This catches a partially completed event that still classifies as `SCHEDULED`. Source observations can predate the pinned cutoff if their exact official archive and completed fetch-run provenance verify; missing or ambiguous provenance fails closed. Forecasts never learn from later outcomes.
- A completed publication is recovered from its sealed intent/publication state witnesses and verified source archives without rechecking today's mutable event/fixture state and without rescoring. This preserves a valid prospective issue if official finality arrives after publication but before receipt recovery. An intent with no complete publication remains an ambiguous incomplete attempt and is not replayed; use a new observation ID after resolving it.
- BB uses its paired route policy; TC retains the evaluator's independent selected PLAY/SAVE captain policy and deterministic selection procedure.
- Outcome capture requires `fetch_runs.status='success'`, valid non-reversed `started_at`/`finished_at`, and exactly one bootstrap, fixture, and target event-live archive for that run. Their common raw `observed_at` may precede `started_at`; it must not postdate successful `finished_at`. `captured_at` is that run's completion/availability time, while the raw observation time remains in the receipt.
- Outcomes use event grain `player_gameweeks:<E>`, not fixture-grain substitutes. Missing players and unavailable fields remain missing and are listed in the capture receipt; they are never turned into zeros. Existing provisional captures remain distinct in the append-only ledger and cannot satisfy final maturation.
- An identical capture retry verifies and reuses its exact operation receipt and does not duplicate rows. A later fetch/observation has its actual later timestamps and a distinct operation identity. A partial set of committed rows without a complete verifiable receipt refuses.
- The maturation receipt key is deterministically derived from `(observation_id, realization_event)`. A retry first verifies and returns the existing receipt. If receipt publication failed after finalization, a retry may recover exactly one complete, mutually bound content-addressed causal-label/outcome pair, but only after revalidating it against the original causal source, selected event, official capture receipts, source positions, and calibration validator; it publishes the single deterministic receipt without calling the finalizer again. Its `matured_at` records the actual successful publication/recovery time; the original forecast issue and label-availability times remain unchanged. Partial pairs, duplicate/competing artifacts, or artifacts that do not bind the origin are refused. Identical maturation retries therefore cannot add duplicate handoff rows.
- The calibration handoff contains only receipt-verified, unique matured observations. It is evidence for later separately authorized calibration, not calibration itself and not evaluator permission.

## Weekly operator procedure and gates

1. Separately authorize and complete the normal official data refresh, fresh normal four-event certification, and normal decision. Preserve their exact IDs and cutoff; do not run these as a side effect of evidence collection.
2. Confirm the generation snapshot contains the intended official chip state and preserve unknown event-start FT as null unless there is a genuine timely confirmation.
3. Prepare the exact same-origin `CHIP_RESERVATION` generation and complete opportunity coverage through the pinned expiry using its approved writer-lease workflow. This is the current operator blocker: no dedicated CLI exists in this checkout.
4. Inspect exact cache requirements. If all required canonical entries verify, run `register-origin`, then `forecast` before any forecast event's outcomes. If entries are missing, stop or obtain separate authorization for `--materialize-worlds`; do not invent a cache identity or silently run the builder.
5. After the normal official collector has a successful final fetch, verify its event, fixtures, endpoint archives, timestamps, run ID, and raw root. Run `capture-outcome` with those exact values.
6. Run `mature` for the selected event. Confirm the deterministic maturation receipt and then read the verified handoff. Resolve partial files only by inspection and an explicit recovery decision; do not delete or rewrite valid historical evidence.

Forecast calibration maturity and production evaluator permission are separate. A growing set of properly prospective, causally matured BB/TC pairs can later support a separately validated calibration process; it does not itself authorize PLAY/SAVE. No calibration, readiness orchestration, production assessment, FH/WC workflow, or account action is implemented here.
