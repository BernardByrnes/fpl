# Production search permission

Status: implementation candidate documentation
Authority: PE-9 Amendment 2, plus the forward production policy adopted for this implementation
Date: 2026-10-02

## Policy

A persisted generation that passes its Amendment 2 integrity checks is necessary for a production search. It is not sufficient by itself. A new production search also requires the canonical search-permission conditions to pass, including a complete completed-event history audit over the generation's pinned origin snapshot and cutoff.

Permission is derived inside the production boundary. No caller boolean, certificate file, audit mapping, authority object or earlier decision is accepted as permission. Failed or unevaluable required evidence returns `PRODUCTION_SEARCH_PERMISSION_DENIED` with the underlying reasons. A denied decision search does not invoke the decision executor or create a decision artifact or record.

For a future chip opportunity, the audit origin is planning event **G** and its pinned cutoff. The selected opportunity event **E** is forecast metadata and never moves the history audit forward. The normal generation remains four events; Wildcard value and chip-continuation products retain their own validators and horizon contracts.

This is a forward policy. It does not change the historical interpretation of Live Run #1. Its retained decision, history audit and FH/WC refusals remain as recorded; no conclusion that the run was unauthorized is inferred from a standalone certificate field.

## Authority and shared rule

PE-9 Amendment 2 remains the production authority: a persisted, content-addressed generation is loaded and independently reverified before use. The implementation does not add a separate certification system, mandatory certificate argument, service process, permission table or caller-carried capability.

The production gate is `fpl_brain.search_permission.require_search_permission`. It re-verifies the selected generation, opens the exact pinned snapshot read-only, audits completed-event history at the generation's planning event/cutoff, derives the existing dependency and horizon results, then calls `fpl_brain.four_gw_decision.decide_search_permission`. `scripts.certify_four_gw.decide_search_permission` remains as a compatibility entry point that delegates to the shared rule. An audit exception, snapshot-open error, missing causal evidence or incomplete audit is represented as unresolved/incomplete and refuses.

## Causal evidence

Generation integrity verification already rederives each declared bundle, dependency closure, model versions, snapshot identity, planning context and horizon. It also binds every predictive run to the pinned snapshot digest and execution UUID. Those integrity checks do not, by themselves, establish the temporal relation needed for a new search.

The missing retained primitive was the generation's reproducible timing chain. `execution_snapshot.derive_generation_causality` now reads the canonical execution snapshot sidecar and the matching `execution_runs.started_at` row. It verifies the sidecar's path, snapshot digest, size, source-database identity and execution UUID; requires explicit lock, capture-start, consistency and completion timestamps; then reuses `require_live_cutoff_matches_snapshot`, `snapshot_for_cutoff` and `causality.assert_causal_cutoff` to establish that the origin cutoff is represented by the pinned consistency point and both the completed snapshot capture and origin cutoff precede execution start. The origin run's planning event and cutoff must also match the generation. No current wall clock is used to infer `CAUSAL`. A canonical digest binds the relevant timing facts and snapshot identity while excluding filesystem paths, so moving an identical retained snapshot does not change its generation identity.

New generation manifests retain the reproducible result in `search_permission_causality`. A missing sidecar or execution row produces `UNRESOLVED`; it does not invalidate the generation's historical integrity projection by itself, but it cannot authorize a new search. Verification reproduces a retained `CAUSAL` block and fails if its inputs or digest no longer match.

## Enforced production paths

The gate runs before production worlds, caches, lineup ranking or optimizer execution in these paths:

- `generation_store.make_decision`, before the declared decision executor.
- `free_hit_production.build_free_hit_production_request`, before FH cache and route/world assembly.
- `wildcard_production.build_wildcard_production_request`, before opening the normal source snapshot or assembling cached worlds.
- `chip_route_assembly.build_future_event_chip_opportunity`, before BB/TC world-cache reads, continuation-world construction or lineup ranking.
- `free_hit_production.build_future_free_hit_event_opportunity` and `wildcard_production.build_future_wildcard_event_opportunity`, before typed future-opportunity evaluation.
- `chip_reservation_forecast.build_evaluated_event_opportunity_record`, the direct production evaluation boundary.

Each chip path names the origin FOUR_GW generation. Its caller-supplied source identity is used only as a restrictive consistency assertion. The gate audits the generation's origin event/cutoff, never the selected future event. A prior successful decision does not grant permission for another search.

When the canonical decision command has an `ExecutionController`, its existing refusal handling marks that run failed with the structured gate reason. Direct API refusals retain the token and reasons on the raised exception. No denied path writes a decision artifact or decision record.

## Retained evidence

New decision artifacts carry a versioned `search_permission_evaluation` in provenance. Its canonical digest binds:

- the origin generation ID and manifest digest;
- origin planning event, cutoff, horizon kind and event list;
- pinned data snapshot digest and execution UUID;
- predictive source identity;
- reproduced causal evidence digest;
- dependency and horizon results;
- snapshot identity/error state; and
- the complete origin history audit, including its reasons.

New decision records use evidence schema `fpl_brain.pe9_engine_decision_record.v2` and retain a compact binding to the evaluation digest and origin identities. `verify_decision` compares that binding with the artifact block and rederives permission from the retained generation and snapshot. Removing, substituting or rebinding the new evidence makes verification fail.

Produced future chip-opportunity artifacts retain a versioned permission-evaluation block and bind it into their artifact digest. Opportunity verification checks its digest and agreement with the source generation, origin event/cutoff, snapshot and completed history audit. The reservation forecast's retained artifact reference continues to bind the exact opportunity bytes.

Historical decision records with evidence schema v1 remain on their historical verifier path. Historical generations without the new causal block remain integrity-verifiable. Neither historical verifier grants permission for a new search; every new search recomputes permission.

## Identity consequences

The shared predicate is implemented in `fpl_brain/four_gw_decision.py`; that module is covered by the decision-runner identity. `scripts/certify_four_gw.py` remains the compatibility entry point and delegates to the shared rule. That script is a member of `analytics.SOURCE_SNAPSHOT_FILES`, so its delegation change intentionally changes the predictive source fingerprint even though prediction equations and history semantics are unchanged. In this local Windows checkout, the predictive identity changes from `983386e1674cf5d6c74fa08ee2c5fb4488e9c92ce9cca88da8462ebf089b02d5` at the accepted baseline to `f91e08f707a02a38193669ba2dece69cb6f862dd7aeaa21cd5e530ba5452469f` after this change. The decision-runner identity changes from `sha256:15b670bab87ba0f4aa47d6e525e0f32f6b7637051da5fa3eb350daba343b7886` to `sha256:b50cf0f293e980a6e78c56378034602481a6b5b15f0f97cc76e24cb1afd36900`. Runs and generations produced under the new certifier identity must carry the new fingerprint. Existing generations keep their original identity and remain available for integrity verification; this implementation does not rewrite them or run new predictions.

The production gate, shared predicate and integrations live in `fpl_brain/*.py`; `_decision_runner_code_identity` fingerprints every such module plus the declared runner. The decision runner-code identity changes and new decision evidence binds that identity. New generation manifests also include causal timing evidence, so newly certified generation IDs reflect that added evidence. Legacy semantic projections omit the new optional block and retain their previous identities.

The accepted predictive generation and historical decisions are not rewritten. An older generation without reproducible causal evidence can still be inspected and verified under its historical integrity contract, but a new search requires a generation whose timing evidence can be reproduced.

## Frozen behavior

This change only adds a fail-closed production permission boundary and retained provenance. It does not alter prediction equations, scoring, RNG, history-audit semantics, manager-state authority, chip rules, calibration thresholds, evaluator-readiness rules or horizon lengths. Existing zero/DNP/blank-history semantics remain governed by the unchanged canonical history audit.

No official ingestion, snapshot capture, prediction generation, production decision, chip computation, merge, PE-11 work or account action is part of this implementation.
