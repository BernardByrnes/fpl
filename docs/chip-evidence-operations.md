# Prospective chip evidence collection

This is a thin operator workflow for retaining prospective BB/TC/FH/WC forecasts and causally maturing later official outcomes. It is not a prediction collector, normal decision runner, certified-product builder, calibration engine, readiness service, production chip arbiter, or account-action tool. It makes no schema migration. The completed GW6 normal decision and hypothetical `route_041` BB/TC assessment remain closed milestones; neither is a prospective calibration origin, and the separately materialized hypothetical cache is not promoted or reused as production authority.

All source identities are explicit. The commands never choose a latest generation, decision, route, fetch run, or evidence root. They preserve actual issue/availability and capture times. Event-start free transfers are never inferred: a prospective origin requiring an event-start FT refuses if the exact value is unknown. The retained GW6 event-start FT remains unknown; a genuine historical confirmation would not create a forecast that was never issued before outcomes.

## First-origin prerequisites

The first eligible prospective origin requires all of the following before the target outcomes happen:

1. A freshly verified normal four-event generation, its verified decision, and the exact verified route. The entry, generation ID, decision ID, route ID, planning event, cutoff, database and evidence root are all supplied explicitly.
2. Search permission for that exact normal generation, with its source decision/route and immutable pinned snapshot intact.
3. The pinned snapshot proves exactly one unused, eligible action chip and its official window start/expiry. The action and expiry passed to the operator must agree. The action-specific FH/WC origin receipt records that eligibility beside the common immutable origin.
4. A genuine, timely event-start free-transfer value where the future policy needs it. No post-event value, current value, route guess or backdated confirmation substitutes.
5. A verified same-origin `CHIP_RESERVATION` continuation whose exact four-event normal-run prefix, cutoff, data snapshot and predictive-code identity match the origin. Its coverage must span every opportunity event through the pinned expiry and every required scoring horizon.
6. For FH, each candidate event needs its four-event chip decision window. For WC, each candidate event additionally needs one explicit compatible `WILDCARD_VALUE` generation with a contiguous 6–10-event horizon. The value generations must share the continuation's same-origin per-event runs and dependency closures.
7. Every required world cache entry must verify against the exact certified generation, event run IDs, full official player population, config/seed/draw count and semantic payload. A cache filename or file count is not evidence. The FH/WC CLI path is cache-only: it refuses missing matrices and does not offer a materialization switch.
8. Every covered opportunity and valuation event must still be demonstrably unplayed at preflight, calculation start and issue. The accepted checks inspect official event/fixture state, existing performance/outcome evidence, source provenance and future kickoffs. Kickoff is a timing guard, never proof of completion. A started, finished, unknown, ambiguous or already-final event blocks a new prospective forecast.

The accepted code can certify `CHIP_RESERVATION` and WC value products when called under the existing execution writer-lease workflow, but this collector does not create those products. The first-origin operator blocker is therefore any missing same-origin continuation, WC value generation, or required cache. Prepare each separately through the accepted producer and lease procedure; the CLI refuses instead of silently generating predictions, rerunning a decision, building a generation or materializing worlds.

The completed GW6 route cannot be calibration origin #1: no qualifying prospective forecast was retained before its outcomes. The earliest possible origin is the first later planning event for which a fresh normal capture and all prerequisites above are completed before any covered event begins. GW7 is only a candidate if its own deadline/outcome status still passes those point-in-time checks; otherwise the first later eligible event is the earliest candidate. This is not established by the old GW6 assessment or cache.

## Commands

All commands share this explicit identity:

```text
--db <existing-db> --entry-id <entry>
--generation-id <exact-generation-id> --decision-id <exact-decision-id>
--route-id <exact-route-id> --planning-event <event>
--origin-cutoff <ISO-8601-cutoff> --evidence-root <retained-root>
```

Register a prospective BB/TC origin or an FH/WC-specific eligible chip window:

```powershell
python scripts/chip_evidence.py register-origin `
  --db <db> --entry-id <entry> --generation-id <generation> `
  --decision-id <decision> --route-id <route> --planning-event <event> `
  --origin-cutoff <cutoff> --evidence-root <evidence-root>

python scripts/chip_evidence.py register-horizon-origin `
  --db <db> --entry-id <entry> --generation-id <generation> `
  --decision-id <decision> --route-id <route> --planning-event <event> `
  --origin-cutoff <cutoff> --evidence-root <evidence-root> --action FH
```

Use `--action WC` for the Wildcard. This registration also re-verifies/retains the common origin; it does not certify missing continuation products.

Retain one forecast. BB/TC use `forecast`; FH/WC use `horizon-forecast`:

```powershell
python scripts/chip_evidence.py forecast `
  --db <db> --entry-id <entry> --generation-id <generation> `
  --decision-id <decision> --route-id <route> --planning-event <event> `
  --origin-cutoff <cutoff> --evidence-root <evidence-root> `
  --action BB --expiry-event <pinned-expiry> --observation-id <unique-id> `
  --continuation-generation-id <same-origin-chip-reservation-generation> `
  --cache-dir <origin-specific-cache-dir>

python scripts/chip_evidence.py horizon-forecast `
  --db <db> --entry-id <entry> --generation-id <generation> `
  --decision-id <decision> --route-id <route> --planning-event <event> `
  --origin-cutoff <cutoff> --evidence-root <evidence-root> `
  --action WC --expiry-event <pinned-expiry> --observation-id <unique-id> `
  --continuation-generation-id <same-origin-chip-reservation-generation> `
  --cache-dir <origin-specific-cache-dir> --value-horizon-length 6 `
  --value-generation <event-1>=<exact-value-generation-1> `
  --value-generation <event-2>=<exact-value-generation-2>
```

For FH, use `--action FH` and omit `--value-horizon-length` and all `--value-generation` arguments. WC needs exactly one explicit value-generation ID for each opportunity event from planning event + 1 through expiry. The command recomputes and verifies an in-memory coverage product from the named continuation; it does not certify that generation. The horizon forecast is cache-only and cannot create worlds. BB/TC retain their existing separate opt-in `--materialize-worlds`, restricted to their origin-isolated cache path; that may require substantial CPU, memory and disk and must be separately authorized and timed.

After the forecast-selected outcome or full FH/WC schedule is officially final, capture each required event-grain outcome using the successful official run that archived it:

```powershell
python scripts/chip_evidence.py capture-outcome `
  --db <db> --entry-id <entry> --generation-id <generation> `
  --decision-id <decision> --route-id <route> --planning-event <event> `
  --origin-cutoff <cutoff> --evidence-root <evidence-root> `
  --action WC --observation-id <same-observation-id> `
  --realization-event <scheduled-event> --fetch-run-id <successful-run-id> `
  --raw-root <that-run-raw-archive-root>
```

Repeat `capture-outcome` once for every event in the verified FH/WC PLAY/SAVE schedule union, each with that event's final official fetch. BB/TC capture only the one selected event. Then mature the selected realization event (the forecast-selected event; for FH/WC it binds the completed multi-event causal schedule):

```powershell
python scripts/chip_evidence.py mature `
  --db <db> --entry-id <entry> --generation-id <generation> `
  --decision-id <decision> --route-id <route> --planning-event <event> `
  --origin-cutoff <cutoff> --evidence-root <evidence-root> `
  --action WC --observation-id <same-observation-id> `
  --realization-event <forecast-selected-event>
```

Use the same identity and observation ID across the forecast, captures and maturation. Replace `WC` with `FH`, `BB` or `TC` as appropriate. The supported official refresh is a separate operator action, for example `python scripts/fetch_fpl.py --config config.json --summaries all --live --gw <event>`; this collector never invokes it.

## Evidence and timing

FH/WC prospective collection is fail-closed to opportunities whose complete PLAY and SAVE schedules have zero source-replayed transfer-hit cost at every event. The exact selected FH route is retained and replayed; the collector never substitutes a zero-hit alternative. Active FH/WC events use their accepted chip transaction and FT rules, while ordinary route events are replayed through the verified transfer engine. WC prefix DTO free transfers and carried-tail start free transfers remain distinct. The issuance gate replays every covered opportunity and both complete arms before reporting all nonzero hits, and it does so before retention; the same proof is checked on every publication load, capture, maturation, retry and recovery. A paid-hit schedule is refused because the current official realized-score finalizer is gross-only; canonical paid-hit calibration remains unsupported.
The forecast retains input cutoff, snapshot observation and availability, calculation start, issue and forecast-availability times, chip action/window/expiry, exact generation and run identities, cache/world identities, policies, evaluator version, uncertainty/diagnostic context and causal source. The actual `forecast_available_at` is distinct from raw `forecast_made_at`; FH/WC maturation carries both unchanged into its operator receipt/calibration row. The general canonical calibration validator/schema is not widened.

For every capture, use the exact successful `fetch_runs` ID and its raw archive root. The run must have exactly one archived bootstrap, fixtures and target `event/<E>/live` response with shared raw observation time, successful completion time, final/data-checked event and every target fixture `finished=true`. `finished_provisional=true` is not an independent veto after official event finality and all finished fixtures pass. Source provenance and hashes are checked before parsing. Capture availability and official finality must both be strictly later than the retained FH/WC forecast availability. Outcomes are event-grain `player_gameweeks:<E>` from `player_gameweeks_final`; missing players/fields remain missing, not zeros.

FH/WC maturation requires receipts for the exact complete event schedule. It seals, for every event, the exact capture receipt references and receipt/file digests, operation/fetch IDs, required player union and per-receipt player capture digests; a separate selected-row manifest records which official receipt supplied each retained player outcome. A later plausible ledger row without a unique verified receipt binding cannot complete the label. Retries reverify the whole manifest and refuse extra, missing, replaced, tampered, provisional or unbound rows.

Capture receipts use deterministic operation identities that bind origin, action, observation, realization event, fetch run, live payload and actual availability. An identical retry reuses its verified receipt and appends no duplicate rows. A later source observation keeps its later times and is a different operation. Rows append through the existing `outcome_observation_captures` transaction; no FPL fact is updated or overwritten. Capture intents support only exact commit-before-receipt recovery; ambiguous partial state refuses. Pre-action-bound legacy capture artifacts remain preserved but cannot be reused or treated as proof.

The maturation receipt key is deterministically derived from `(observation_id, realization_event)`. Before finalizing, the collector checks and verifies any existing receipt. If publication failed after a complete immutable outcome/causal-evidence pair exists, a retry recovers and verifies that exact pair without calling the finalizer again. If artifacts exist without one verifiable receipt or complete mutually bound pair, it fails closed. Provisional and later-final captures stay distinct; only a verified final receipt can mature. These outputs are evidence for separately authorized future calibration, not calibration itself and not production PLAY/SAVE permission.

## Storage and database access

Producer JSON is kept as flat content-addressed files under the explicit evidence root; sealed intents, receipts, locks and publication records live below `.chip-evidence/`. Receipts name flat artifact files and bind their content digests, identity and operation keys. Keep the evidence root and referenced raw archives together; never edit or move individual files after issuance. FH/WC cache references remain evidence about this operation, not production authority.

`register-origin`, `register-horizon-origin`, `forecast`, `horizon-forecast` and `mature` use read-only, query-only database connections. `capture-outcome` is the sole logical DB writer: it appends only official event-grain capture rows to the existing ledger under a SQLite writer transaction. No command migrates the schema, refreshes official data, edits manager state, or writes scouting/watchlist/decision records.

## Weekly operator sequence and stop conditions

1. Separately complete the normal official refresh, fresh normal four-event certification and normal decision; retain their exact IDs and cutoff.
2. Verify timely event-start FT evidence and the pinned snapshot's unused chip window. Unknown FT stays unknown; stop if the canonical future state cannot be proven.
3. Separately prepare the exact same-origin `CHIP_RESERVATION` generation and, for WC, every explicitly named `WILDCARD_VALUE` generation through expiry under the accepted execution writer-lease workflow. Confirm compatibility and cache readiness; this collector will not create missing products or worlds.
4. Register the action-specific horizon origin. Run `horizon-forecast` before any opportunity/valuation event starts. A failed preflight, late event, missing generation, incompatible cache, unknown state or ambiguous source is a stop, not a reason to rerun a normal decision implicitly.
5. After each required official event is final, refresh through the project's ordinary collector, verify the exact successful run and archive root, and issue one `capture-outcome` for each schedule event.
6. Run `mature` for the forecast-selected event, then verify its deterministic receipt, sealed multi-event capture manifest and retained calibration audit row. `load_verified_calibration_handoff` remains BB/TC-only; FH/WC rows are not yet exposed to calibration intake, training or readiness. Any compatible FH/WC handoff is a separate future change/authorization. Keep calibration and production evaluator permission as separate gates.
