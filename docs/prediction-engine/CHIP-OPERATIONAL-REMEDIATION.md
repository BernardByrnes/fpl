# CHIP operational remediation

## Scope and state

This isolated candidate implements the production plumbing and evidence checks
for a complete four-chip assessment. It does not change the frozen V1 branch,
rewrite retained assessments, run an FPL account action, merge the candidate, or
start PE-11.

The accepted remediation milestone is commit
`e960ca08d34802d519dc761289317b88227b9356` (tree
`2d425f68cbd2a1e2d295d81eae66469480f6458d`) on
`codex/chip-operational-remediation`. The work below is a dirty, unpinned
replacement candidate based on that milestone. It is not accepted until its
focused gates, authoritative wrapper, exact-SHA CI and Sol review finish. Do
not describe fixture evidence as production validation.

## Requirement-to-implementation matrix

| Requirement | Implemented and accepted at `e960ca0` | Current replacement candidate | Remaining dependency or restriction |
| --- | --- | --- | --- |
| 1. FH/WC future opportunities | Existing canonical chip and maturation contracts | Dedicated FH/WC producers create action-specific PLAY/SAVE future-event opportunities, retain them at origin and feed maturation/calibration. The FH producer re-derives its decision authority from the verified origin CHIP_RESERVATION generation and compares all selected bundle contexts. Event 7/8 lifecycle fixtures assert policies, world/source identities, outcome maturation, FH permanent-state restoration and WC transfer-state semantics. | Fixtures are not a live production assessment. Fresh cutoff-consistent manager, generation and outcome inputs are still required. |
| 2. Reservation through expiry | Normal decision remains FOUR_GW; WC value horizon remains separate at 6–10 events | A verified CHIP_RESERVATION product reuses the exact four-event prefix and binds continuation runs to the same cutoff, snapshot and predictive identity. BB/TC continuation carries the route terminal squad and bank with no transfers and a per-event ranked lineup. Future WC 6–10 event windows are certified only from the same origin product, with exact run IDs and dependency closures preserved; FH authority is loaded from that verified product. Origin-pinned bootstrap rules are digest-checked and retained. | A real product, bootstrap archive capture and each event opportunity must be available through confirmed expiry. Missing or unverifiable rules keep beyond-route coverage incomplete. No-transfer continuation is a forecast model, not observed future manager behavior. |
| 3. Evaluator readiness | Existing execution-permission field is preserved | Action-specific readiness evidence can clear BB/FH/WC review-only gates only after retained prospective causal observations pass independent criteria. Before applying it, the current evaluation must also carry a finite value, allowlisted success reasons, the matching evaluator version, canonical horizon, certification/snapshot identities and a required snapshot-bound flag. The readiness artifact's evaluation cutoff must be at or before the current assessment cutoff; a later artifact cannot authorize an earlier assessment. Refused evaluations remain unchanged. Fixtures demonstrate permitted and refused paths. | No production readiness evidence exists. BB/FH/WC remain execution-blocked until compatible real evidence verifies. TC retains its established gate. |
| 4. Arbiter ranking | Canonical action and reservation contracts remain in place | Each eligible candidate is compared after its reservation in shared units; an unknown reservation or blocked evaluator remains unrankable. Regression covers 10−9 versus 8−2 and chooses net 6. | Requires a calibrated reservation and readiness for every eligible executable candidate. |
| 5. Forecast timestamps | Origin cutoff and outcome-availability safeguards remain enforced | Inputs must be as-of the origin cutoff; `made_at` records actual forecast issuance and may be later. Historical replay is excluded from evaluator-readiness evidence. | No forecasts or labels are backdated. Live validation still needs a fresh certified snapshot and later official final captures. |

### Current candidate status groups

**Implemented and accepted:** the `e960ca0` milestone preserves the existing
outcome scoring, tamper refusal, historical manager-state boundary, normal
FOUR_GW contract, separate WC value horizon, scenario identities and canonical
Free Hit permanent-squad restoration checks. None of the historical evidence is
rewritten.

**Implemented in this candidate, awaiting acceptance:** FH/WC future-event
production and the origin-to-maturation lifecycle; BB/TC certified expiry
coverage and continuation; action-specific readiness; net-after-reservation
ranking; and later-issued prospective forecast timestamps. The focused evidence
and exact source paths are recorded below.

**Still unimplemented or not production-validated:** a real certified
continuation generation and real future-event opportunity records through the
manager-confirmed expiry; an operational adapter invocation using those exact
retained inputs; and production readiness/calibration evidence. No expensive
operational computation has been run to discover missing prerequisites.

**Missing certified inputs:** current planning event, complete permanent squad,
bank, current FT, separately confirmed event-start FT, chip availability/expiry,
and a fresh manager observation captured before its snapshot and cutoff. For
future continuation, the required same-origin certified event runs and pinned
season rules must also be available. Team `241392` is known; no other current
manager facts are confirmed.

**Missing calibration evidence:** the retained inventory has no matured paired
chip reservation outcomes suitable for real calibration. Synthetic fixtures
exercise the calibration path only.

**Evaluator readiness restrictions:** BB, FH and WC stay blocked without their
own action-specific evidence. Reservation calibration cannot grant that
permission. TC keeps its existing execution gate, and reservation calibration
is still required before any `PLAY_CHIP` endorsement.

## Context-resume checklist

- [x] Accepted base recorded as `e960ca0` / tree `2d425f68`.
- [x] FH/WC event 7/8 producer-to-calibration fixtures completed.
- [x] Certified CHIP_RESERVATION extension and BB/TC continuation contracts implemented with focused fixtures.
- [x] Readiness pass/refusal fixtures, net-ranking regression and timestamp safeguards implemented.
- [x] FH/WC selected-event producer → retained-origin → maturation → calibration fixtures cover event substitution and chip-specific state restoration.
- [x] Reservation expiry fixtures cover WC-6/WC-10, unknown expiry and incomplete FH horizon; a no-rule-evidence continuation remains incomplete with null value.
- [x] Pinned season-rule resolver selects only an accepted bootstrap capture at/before the origin cutoff, verifies raw archive bytes, and refuses a tampered archive.
- [x] Readiness refuses to elevate a current action refusal even when its separate readiness artifact passes; permit/refuse paths remain fixture-only.
- [x] Readiness application requires the current assessment cutoff and refuses a verified artifact whose evaluation cutoff is later, preventing temporal lookahead.
- [x] Earlier contract-focused set passed: 127 tests in 45.64 seconds on 2026-10-01; includes readiness, FH/WC lifecycle, forecast, season-rule provenance, continuation and WC evaluator tests.
- [x] Refreshed cross-contract set passed: 230 tests in 286.10 seconds, including PE-9 production decision, chip operational remediation, FH/WC, readiness and season-rule provenance.
- [x] FH preflight refusal without origin-pinned season rules, BB continuation, and season-rule provenance follow-up passed: 10 tests in 0.93 seconds.
- [x] Readiness permit/refuse and assessment-cutoff temporal-boundary tests passed: 15 tests in 0.61 seconds.
- [x] Earlier five-gap focused integration bundle passed: 165 tests in 47.46 seconds before future-WC certification and FH continuation-authority revalidation were added.
- [x] Future WC 6- and 10-event value windows certify from one origin continuation product without creating a future FOUR_GW decision pointer; an incomplete-expiry product refuses.
- [x] Future FH authority is re-derived from the verified CHIP_RESERVATION product, exact event runs are asserted, and a gapped future FH horizon refuses.
- [x] FH/WC lifecycle and future-WC certification focused integration passed: 5 tests in 16.57 seconds; FH substituted bundle-context refusal passed separately in the lifecycle bundle.
- [x] Refreshed five-gap integration bundle passed after future-WC certification and FH authority-source verification: 189 tests in 56.49 seconds.
- [x] Complete candidate diff and documentation review; code and documentation are final before pinning.
- [ ] Pin one replacement candidate only after code and documentation are complete; run the wrapper on that exact candidate.
- [ ] Run exact-SHA CI and final Sol review; do not merge or begin PE-11.
- [ ] Report production inputs, calibration and evaluator-readiness restrictions separately from code acceptance.

Current resume point: the replacement candidate is based on accepted milestone
`e960ca08d34802d519dc761289317b88227b9356` on
`codex/chip-operational-remediation`. Code and documentation are complete, the
diff has been reviewed, and the refreshed five-gap focused integration bundle
passed after future-WC certification and FH authority-source verification.
Next: pin one replacement candidate, run the authoritative wrapper against that
exact candidate, then exact-SHA CI and Sol review. Do not merge or begin PE-11.
Keep production inputs, calibration and readiness evidence as separate
prerequisites; do not merge or begin PE-11.

## Historical manager-state boundary

Live Run #1 uses cutoff `2026-09-29T20:27:01Z`. Its retained
`event_start_free_transfers` is null. The separate facts `free_transfers = 3`
and `bank = 7` do not establish that value. No later confirmation is read into
that cutoff. The historical outcomes remain:

| Chip | Historical refusal |
| --- | --- |
| Free Hit | `FREE_HIT_PRODUCTION_MANAGER_STATE_MISSING` |
| Wildcard | `WILDCARD_PRODUCTION_MANAGER_STATE_MISSING` |

The production preflight requires explicit manager confirmation in the pinned
snapshot, including event-start FT and its capture time at or before the
generation cutoff. A future assessment needs those facts confirmed before its
fresh snapshot and generation are made. No confirmation has been backdated.

Focused evidence: `test_historical_run1_missing_event_start_ft_stays_blocked_after_later_observation`
and `test_production_preflight_rejects_missing_or_late_event_start_confirmation`.

### Confirmation form for a future operational run

Team ID `241392` was supplied by the manager. The attached screenshot visibly
shows GW6, a 15/15 squad, £0.7m bank, 3 free transfers, and Play controls for
Wildcard and Free Hit. Its capture time is unknown, so these screenshot values
are unconfirmed observations and must be re-confirmed before they can enter a
new snapshot. The screenshot shows only part of the squad and does not establish
chip expiry or the saved event-start FT value. It is not evidence for a current
production run.

The local database's latest manager-state observation for this team is GW4 at
`2026-09-14T08:13:41Z`; it has no GW6 squad. That observation is stale and is not
prefilled below. Player IDs for any confirmed names must be resolved through the
supported authoritative FPL source, with source and observation time retained.

| Field | Value to confirm for the fresh run | Source/status |
| --- | --- | --- |
| Team ID | `241392` | Manager supplied; identity only |
| Current planning gameweek | GW6? | Screenshot; capture time unknown, confirm |
| Permanent 15-player squad | Names for all 15 players: **confirm** | Screenshot is partial; resolve names to IDs from the authoritative source |
| Bank | £0.7m? | Screenshot; capture time unknown, confirm |
| Free transfers remaining | 3? | Screenshot; capture time unknown, confirm |
| Event-start free transfers | **confirm separately** | Unknown; never infer from current FT remaining |
| Chip availability and expiry | **confirm each chip and expiry** | Screenshot controls alone do not establish expiry |
| Confirmation captured at | Record the actual time of the manager's response | Fill only when confirmation occurs; never prefill or backdate |

“Event-start free transfers” means the free-transfer bank entering that gameweek,
before any transfers are made in that gameweek. It is distinct from the number
of free transfers remaining after transfers. The manager has said fresh facts
will be provided later; development uses labeled fixtures until then.

## Wildcard value horizon

Wildcard now has a separate `WILDCARD_VALUE` generation kind for a contiguous
6–10 event value horizon. The ordinary decision generation remains `FOUR_GW`
with its four-event contract. Certification requires the Wildcard generation
to reuse the exact four-event run IDs and dependency closure from the verified
normal generation, then binds every event to the same cutoff, pinned snapshot,
execution UUID and predictive code identity. It verifies again after
certification. A sequence assembled from later GW7/GW8 planning runs cannot
replace the missing GW6 prefix.

For future Wildcard opportunity windows, `certify_future_wildcard_value_generation`
selects a contiguous 6–10 event slice from the verified origin
`CHIP_RESERVATION` product. It obtains the run IDs and snapshot from that
product, rechecks the root FOUR_GW prefix and cutoff/snapshot/code identity, and
compares every selected event's dependency closure after certification. It
cannot accept caller-supplied run IDs or later planning runs, and it does not
publish a future FOUR_GW decision product.

Focused evidence covers 6- and 10-event products, the unchanged four-event
product, short/gapped/length-mismatched horizons, refusal of substituted
prefix runs, and the production builder's separate value generation bound to
the exact normal four-event prefix. The connected manager-state comparison
also has a production-adapter regression. These are fixture-backed results,
not a production Wildcard generation. The available prediction inventory
reaches GW8; verified same-cutoff projection runs for any needed later events,
including GW9/GW10 when requested, remain a data prerequisite.

## BB/TC expiry continuation

For expiry beyond the normal route's four-event horizon, the supported
prediction input is a separate `CHIP_RESERVATION` generation. Certification
requires a contiguous origin-starting event sequence, the exact normal
four-event run prefix, and matching cutoff, pinned data snapshot and predictive
code identity. It does not widen the normal decision horizon. The coverage
product binds this generation and the exact runs through the known expiry.

BB/TC event opportunities after the normal horizon start from the replayed
normal route's terminal permanent squad, bank, purchase-price basis and chip
state. The declared continuation model makes no transfers in intervening
events, advances FT using the supplied season rules, and ranks a legal lineup
for each future event from that event's certified continuation worlds. The
event record binds the continuation generation, event bundle/runs, coverage
product, manager state, lineup, rule payload and world identity. The rules must
be resolved by the caller from the origin-pinned official snapshot. This is a
forecast assumption, not an observation of later manager behavior.

The reservation builder produces only opportunities supported by the verified
product. If the product, continuation rules or any event opportunity is
missing, the forecast remains `INCOMPLETE` with `raw_value = null`; a complete
prediction product alone cannot force a numeric forecast. Focused evidence:
`test_chip_reservation_product_extends_verified_four_event_prefix_through_expiry`,
`test_bb_continuation_opportunity_binds_terminal_state_and_per_event_lineup`,
and `test_complete_prediction_product_with_missing_event_forecast_stays_unknown`.
These are fixture-backed product and model checks. No real continuation
generation exists for a confirmed current manager cutoff.

## Proposed BB/TC scenario

Production BB and TC construction takes one explicit route from a verified
normal decision. The route is replayed from the pinned actual squad and prices;
the resulting proposed squad, bank, free transfers, lineup, bench and armband
are kept distinct from actual ownership. Both evaluators receive the same
proposed route and certified H1 worlds, and their evidence carries matching
scenario and world identities.

The historical captured-lineup assessment selected TC. A separate hypothetical
`route_001` assessment selected BB. Those remain two distinct assessments;
neither is rewritten or treated as calibrated `PLAY_CHIP` support.

Focused evidence: `test_verified_route_replay_rebuilds_canonical_state_and_refuses_mutation`,
`test_bb_tc_evaluations_share_one_proposed_route_scenario_and_worlds`, and
`test_bb_tc_assessment_refuses_mixed_hypothetical_scenarios`.

## Free Hit counterfactual arms

The production builder verifies the normal SAVE route and generation, reads
manager and price authority from that generation's pinned snapshot, and builds
both arms. SAVE replays the normal four-event route. PLAY starts from the
permanent squad restored at H2 and optimizes only the genuinely different H2–H4
tail. The retained evidence records both route configs and actions, source
decision and generation identities, shared manager/world identities, and the
H2 restoration state.

The immutable record verifier checks the two arms use the same certified
generation, cutoff, snapshot, certification, canonical manager digest and H1
world identity. It checks the restoration manifest against the canonical
manager state, then checks SAVE begins from current H1 squad/bank/current FT and
acquisition basis; PLAY begins at H2 with permanent squad/bank/acquisition
basis restored and event-start FT preserved. It also verifies the route event
sequences, configs, source decision hashes and arm digest.

Focused evidence: `test_free_hit_h2_state_uses_event_start_ft_not_current_remaining_ft`,
`test_free_hit_production_builder_emits_retained_play_save_arms`,
`test_assessment_store_verifies_both_retained_free_hit_arms_and_restoration`,
and forged PLAY-start/restoration-state refusals in those tests. The complete
production assembly has not been run against a fresh, factually confirmed
snapshot or a new production decision. That run stays deferred until its inputs
are ready.

## Calibration and evaluator execution gates

Reservation calibration now accepts only retained, content-addressed causal
evidence containing separate PLAY/SAVE arms for the same scenario and worlds,
plus a matured future outcome. BB/TC require the same proposed squad and lineup;
FH/WC use separate legal arms with explicit action and transfer-state semantics.
Each outcome is a
versioned paired record bound to the observation, chip action, planning origin,
scenario, world, source decision/generation, and both arm IDs and artifact
digests. The outcome scorer is versioned and derives arm weights from the
action, legal 15-player lineup, pinned player positions, official appearance
minutes, and captured event points. It uses the normal lineup engine's legal
autosubs and captain/vice fallback, scores all appearing players for Bench
Boost, and applies the extra captain copy for Triple Captain. Caller-supplied
weights are checked against those reconstructed weights; changing a weight and
rehashing the artifacts cannot change the label. The canonical matured-outcome
scorer supports BB, TC, FH and WC. `finalize_causal_observation` derives
PLAY/SAVE weights only after later official final captures are present in the
append-only ledger; it retains the outcome and evidence as immutable
content-addressed artifacts. Focused tests exercise finalization from official
ledger captures, different FH/WC arm squads, and refusal of missing or nonfinal
captures. The origin writer freezes the paired policies belonging to the
forecast's selected future event. Both maturation and later calibration
validation bind every PLAY and SAVE arm's declared event, action and role to
that selected forecast event; BB/TC arms are also bound to the exact selected
forecast policies. FH restoration evidence must be complete, typed, bound to
the SAVE arm's permanent squad/bank/acquisition basis/event-start FT, and
restore in H2 after the actual PLAY event. Regressions cover FH event
substitution during maturation and validation, TC arm substitution during
maturation and validation, incomplete FH restoration, and production position
resolution across the union of different FH PLAY/SAVE squads. These checks
reject a substituted TC captain policy and incomplete or self-asserted Free
Hit restoration evidence.

Every capture must be event-grain, officially final, and carry the declared
official player-gameweek provenance, including minutes and total points. At
production load time, the evidence reader resolves every capture digest
against `outcome_observation_captures`, recomputes its content digest, checks
the retained fields and points, and verifies the arm's player positions against
the source generation's pinned snapshot. For different PLAY/SAVE squads, the
resolver loads the union of both proposed squads before each arm is checked
against its own pinned positions. Outcome availability is derived from
the latest capture time and must match both the row and label; captures must
postdate official finality and the forecast origin. The pre-registered
expanding-origin protocol then excludes outcomes not mature by its evaluation
cutoff and by each training origin. Loading a calibrated artifact re-reads
each causal record and outcome through that production verifier and
reproduces the dataset, reports and model under the locked criteria; a
self-hashed `CALIBRATED` label alone is rejected.

The local read-only inventory found zero `outcome_observations` and zero
`calibration_records`; the available `player_gameweeks` are point outcomes,
not paired chip reservation trials. Synthetic fixtures test the calibration
implementation only. The reservation-forecast artifact contract verifies
per-event point-in-time expected-opportunity records bound to a certified
source and exact SAVE state. A route-backed BB/TC producer now evaluates each
future event available in the verified normal four-event generation and, with
the separately certified continuation product, through known expiry. It
retains event-specific paired policies and exact source identities. Both
current BB/TC assessments and their future reservation forecasts bind the same
verified proposed route; the SAVE identity includes the post-H1 squad,
acquisition-price basis, bank, free transfers, chip state and route identity.
FH/WC also have dedicated future-event producers using their canonical typed
requests and chip-specific route/generation machinery. Fixture lifecycle tests
cover production, origin retention, maturation and calibration for selected
event substitution at events 7 and 8. These fixtures are not a live forecast.
The production assessor can consume retained forecasts, and the CLI accepts
their verified artifact directory. No production forecast currently has
complete coverage through an actual manager-confirmed expiry.

Reservation calibration still requires validated, matured causal paired
outcomes; none exist in the retained inventory. Reservation therefore remains
uncalibrated and cannot support a production `PLAY_CHIP` endorsement.

The candidate now reads action-specific evaluator-readiness artifacts built
from independently revalidated prospective causal observations. Different
criteria apply to BB, FH and WC, and the readiness artifact must match the
action and evaluator version before its execution permission can be applied.
The artifact's evaluation cutoff must also be no later than the current
assessment cutoff, preventing later matured evidence from enabling a historical
assessment. Fixture tests demonstrate both permission and refusal for BB/FH/WC,
including insufficient samples, incompatible evaluator versions, historical
replay and an artifact that postdates the assessment. These simulated
observations are explicitly not production evidence. In the
absence of real readiness evidence BB/FH/WC remain execution-blocked; TC keeps
its existing gate. Reservation calibration remains a separate requirement and
cannot by itself grant evaluator execution permission.

## Assessment retention and remaining dependencies

Production orchestration requires an explicit normal route, pre-cutoff manager
facts, verified certified worlds, and the separate Wildcard value generation
when Wildcard is eligible. It emits a structured evaluated/blocked/unavailable
disposition for BB, TC, FH and WC. Per-run records are retained atomically with
no-replace semantics and verified against their digest and chip-specific
contracts.

Focused test results, the authoritative wrapper, exact-SHA CI, and independent
review are reported against the exact candidate commit in the acceptance
report. The authoritative acceptance recognizes only these two
configuration-dependent test IDs:
`tests/test_causal_bundle_integrity.py::test_13_manager_state_prose_derives_from_actual_state`
and
`tests/test_causal_bundle_integrity.py::test_13b_zero_ft_prose_is_also_derived`.
No operational run has been used to discover missing prerequisites. The
outstanding operational facts are:

1. Manager confirmation of the current planning GW, permanent squad, bank,
   current FT, event-start FT, and chip availability/expiry, captured before a
   new snapshot and its cutoff. Team `241392` is known; the screenshot's visible
   GW6/£0.7m/3 FT values remain unconfirmed because its capture time is unknown.
2. A normal four-event certified generation/decision and canonical FH
   certification matching that same snapshot.
3. One consistent 6–10 event Wildcard product with all required event runs at
   that same cutoff, snapshot, execution UUID and code identity.
4. A same-origin CHIP_RESERVATION generation with its exact four-event prefix
   and sufficient certified events through every known expiry, plus all
   action-specific event opportunities. The matching snapshot must also retain
   the accepted bootstrap capture and intact archive payload needed to resolve
   season rules. The code path exists; production products and complete forecast
   artifacts are missing.
5. Validated retained paired chip causal outcomes for reservation calibration
   and action-specific evaluator readiness. These are separate evidence gates;
   neither has real retained production samples.

The validation report must distinguish fixture-backed implementation from
operational evidence, preserve the historical Run #1 refusals and the two
separate BB/TC scenarios, and report results against the exact candidate SHA.
