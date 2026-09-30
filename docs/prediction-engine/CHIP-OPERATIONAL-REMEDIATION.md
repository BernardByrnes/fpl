# CHIP operational remediation

## Scope and state

This isolated candidate implements the production plumbing and evidence checks
for a complete four-chip assessment. It does not change the frozen V1 branch,
rewrite retained assessments, run an FPL account action, merge the candidate, or
start PE-11.

The worktree started at frozen commit
`1c5b8348a463261cb70f626b4637dfa118b36e63` (tree
`d31da0a127b16993ad49fe6e9c9aa5ee82911b7f`) on
`codex/chip-operational-remediation`. Candidate SHA and validation results are
tracked against the exact code commit in the acceptance report; this document
describes the implementation and its operational prerequisites.

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

## Wildcard value horizon

Wildcard now has a separate `WILDCARD_VALUE` generation kind for a contiguous
6–10 event value horizon. The ordinary decision generation remains `FOUR_GW`
with its four-event contract. Certification requires the Wildcard generation
to reuse the exact four-event run IDs and dependency closure from the verified
normal generation, then binds every event to the same cutoff, pinned snapshot,
execution UUID and predictive code identity. It verifies again after
certification. A sequence assembled from later GW7/GW8 planning runs cannot
replace the missing GW6 prefix.

Focused evidence covers 6- and 10-event products, the unchanged four-event
product, short/gapped/length-mismatched horizons, refusal of substituted
prefix runs, and the production builder's separate value generation bound to
the exact normal four-event prefix. The connected manager-state comparison
also has a production-adapter regression. These are fixture-backed results,
not a production Wildcard generation. The available prediction inventory
reaches GW8; verified same-cutoff projection runs for any needed later events,
including GW9/GW10 when requested, remain a data prerequisite.

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
evidence containing separate PLAY/SAVE arms for the same proposed squad,
lineup, scenario and worlds, plus a matured future outcome. Each outcome is a
versioned paired record bound to the observation, chip action, planning origin,
scenario, world, source decision/generation, and both arm IDs and artifact
digests. Its fixed label definition is recomputed from the captured event
points and each retained arm's explicit scoring weights; a caller-supplied
`observed_points` value alone is insufficient. Every capture must be event
grain, officially final, and carry the declared official player-gameweek
provenance. At production load time, the evidence reader resolves every
capture digest against `outcome_observation_captures`, recomputes its content
digest, and checks the retained fields and points. Outcome availability is
derived from the latest capture time and must match both the row and label;
captures must postdate official finality and the forecast origin. The
pre-registered expanding-origin protocol then excludes outcomes not mature by
its evaluation cutoff and by each training origin. Loading a calibrated
artifact re-reads each causal record and outcome through that production
verifier and reproduces the dataset, reports and model under the locked
criteria; a self-hashed `CALIBRATED` label alone is rejected.

The local read-only inventory found zero `outcome_observations` and zero
`calibration_records`; the available `player_gameweeks` are point outcomes,
not paired chip reservation trials. Synthetic fixtures test the calibration
implementation only. There is not enough retained causal evidence to create a
production calibration, and no current chip evaluator supplies a raw future
reservation forecast. Reservation therefore remains uncalibrated and cannot
support a production `PLAY_CHIP` endorsement.

The evaluator gates remain independent: BB is review-only, TC retains its
existing permitted gate, and FH/WC are review-only. Reservation calibration
cannot override an evaluator whose `execution_permitted` is false. It also
cannot supply the missing raw forecast. The gate behavior is covered by the
focused remediation test and the existing BB, FH and WC review-only tests.

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

1. A manager-confirmed event-start FT, current FT, bank and squad captured
   before a new snapshot and its cutoff.
2. A normal four-event certified generation/decision and canonical FH
   certification matching that same snapshot.
3. One consistent 6–10 event Wildcard product with all required event runs at
   that same cutoff, snapshot, execution UUID and code identity.
4. Validated retained paired chip causal outcomes and a point-in-time raw
   reservation forecast before any reservation-calibrated recommendation can
   be supported.

The validation report must distinguish fixture-backed implementation from
operational evidence, preserve the historical Run #1 refusals and the two
separate BB/TC scenarios, and report results against the exact candidate SHA.
