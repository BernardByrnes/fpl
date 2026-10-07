# Base-skeleton multi-squad benchmark — 79 retained keys

Date: 2026-10-03 UTC. This report extends the earlier eight-key profile at `C:\Users\USER\agent-workspaces\gw06-base-skeleton-phase2-scratch-20261003\performance-phase2-report.md`; that report is preserved unchanged. This report supersedes its “no 79-key benchmark” status. It does not revive the withdrawn 12–20-minute whole-decision estimate from the earlier captain-term prototype.

## Scope and retained inputs

- Accepted baseline: `0e01d3ea1bbeb4a65dd562a533b297a6f2e16007`, tree `1f59e6d3cd4fb380e217496b2087e5eb802fbc02`.
- Isolated experiment: `C:\Users\USER\agent-workspaces\fpl-gw06-base-skeleton-phase2`, branch `codex/gw06-base-skeleton-phase2`. The accepted checkout and the prior captain-term prototype were not used as edit targets.
- Retained certified generation: `sha256:66b1557ce6680c878d08dbc03de853d9b462b35e461c354c1e274737b32ad98d`; generation verification passed using the runtime source DB read-only.
- Four pinned 10,000-world matrices, GW6–GW9, were read from the existing runtime cache and matched their verified cache keys and content digests. No worlds, predictions, DB rows, or decisions were created.
- Inputs: retained Stage 2 finalist set (18 unique exact keys) followed by the 61 missing keys in the retained escalation route table (79 unique keys total). The exact evaluator order, config, canonical four-worker scheduler, four-event contract, and shared parent cache were the same in both arms. Each phase used a fresh four-worker process pool; memo reuse within each pool was retained.
- Benchmark harness and worker-instrumentation SHA-256 values at freeze: `72d1265e1c42b127944a41ba1f17b55b186b1a6a57278b4ac1677f27bfa1a63` and `34925a4214f70f2035b2785f274184735f77e63e217fd1883d90298d3318eed4`. Harness and input identities matched their own start/end values and were equal across variants.

This is offline exact scoring of retained certified route inputs. It did not rerun route search or the optimizer and does not establish unchanged ranking for a complete production search.

## One isolated change

The candidate changes only `fpl_brain/manager_worlds.py` in the production implementation: each skeleton computes a watched-bit mask once and uses `appearance_mask & watch_mask` as its local autosub-choice dictionary key. The key contains exactly the same watched appearance bits as the prior boolean tuple. The dictionary remains local to one skeleton. XI, ordered bench, player positions, goalkeeper flags, entrant selection, appearance-group order, policy enumeration, and floating-point accumulation order remain unchanged. No unrelated optimization was added.

The focused oracle tests cover high player indices, irrelevant bits, skeleton-local cache separation, bench ordering, goalkeeper availability/swap boundaries, and entrant scoring. The reference implementation is the equivalence oracle.

## Measured batch results

Unprofiled scheduler wall time is the performance measure. The per-call instrumentation is diagnostic and was not used as batch wall time.

| Phase | Baseline wall | Integer-key candidate | Change |
|---|---:|---:|---:|
| Stage 2, 18 unique keys | 1,037.188 s | 822.722 s | 20.68% faster |
| Escalation, 61 missing keys; 18 parent-cache hits | 3,383.541 s | 2,645.180 s | 21.82% faster |
| Total | 4,420.730 s (73m 41s) | 3,467.902 s (57m 48s) | 952.828 s saved (21.55%) |

Actual task distribution was balanced. Stage 2: baseline 4/5/5/4 and candidate 5/4/4/5 tasks per worker. Escalation: baseline 15/15/16/15 and candidate 16/15/15/15. Stage 2 dispatched 18 jobs; escalation dispatched 61 jobs and reused all 18 parent-cache entries. No fallback or cancelled worker jobs occurred.

Across both phases, each variant processed the same 245,850 skeletons, 27,043,500 policies, and 790,000 appearance-group visits. Summed `base_skeleton_stats` call time was 15,762.103 s baseline versus 11,990.043 s candidate, 23.93% lower. These summed worker-call times overlap in wall-clock time and must not be read as batch wall time.

### First and subsequent tasks

`ordinal=1` is the first task handled by a worker in that phase’s fresh pool. Subsequent tasks are grouped separately. Counts by skeleton size are shown so the smaller first-task calls do not bias the comparison.

| Phase / skeletons | Baseline first → subsequent mean | Candidate first → subsequent mean |
|---|---:|---:|
| Stage 2 / 1,650 | 110.131 s (n=3) → 110.898 s (n=3) | 83.096 s (n=3) → 85.429 s (n=3) |
| Stage 2 / 3,300 | 213.593 s (n=1) → 210.671 s (n=11) | 162.241 s (n=1) → 155.436 s (n=11) |
| Escalation / 1,650 | 106.987 s (n=2) → 104.592 s (n=1) | 81.168 s (n=2) → 89.328 s (n=1) |
| Escalation / 3,300 | 215.704 s (n=2) → 211.037 s (n=56) | 167.543 s (n=2) → 161.173 s (n=56) |

The small 1,650-skeleton subsequent samples should not be overinterpreted. For the common 3,300-skeleton calls, the integer key reduced the per-call time for both first and later tasks. The task instrumentation shows process-local appearance-group/captain-term memo reuse across squads in the same worker: 2 hits of each in Stage 2 and 46 of each in escalation, in both variants. Matrix loads were 16 and 15 by phase respectively. These are process-local matrix-level memo hits. They are distinct from the autosub-choice cache, which is created anew for each skeleton.

### Autosub-choice cache and earlier counters

Both variants recorded exactly the same local choice-cache partition: 2,237,896,650 hits and 220,603,350 misses (2,458,500,000 lookups, about 91.0% hits). The aggregate totals are across all 79 calls and remain scoped to the per-skeleton dictionaries. They are not process-wide or cross-squad memo hits.

The earlier `29,445,900` hits / `3,554,100` misses came from two `base_skeleton_stats` calls in the cold/warm `route_002` ranking probe (`base_skeleton_stats_calls=2`), not a single call. Together they account for 33 million lookups, consistent with two 1,650-skeleton calls. The separate one-million-lookup key test measured tuple construction plus dict get/insert at 2.176 s median and integer projection plus the same dict operations at 0.314 s median, with identical hit/miss partitions.

The bounded cProfile sample found repeated Python construction of appearance/core tuples inside the target loop. `_appearance_groups` and this path use Python lists, tuples and dictionaries; no NumPy operation occurs in the measured hot path. Peak allocated bytes and isolated costs for every inner Python operation were not measured, so those components remain unquantified.

## Exact equivalence

Both arms passed against the retained artifact, and the cross-arm comparison passed:

- All 184 retained route/event score rows, policies, per-world score digests, and means matched exactly.
- Stage 2 finalist totals and paired comparisons matched exactly.
- Escalation route totals and all paired comparisons matched exactly; the canonical 10,000-world near-tie statistic reproduced as mean difference `0.0817`, paired SE `0.10106296989198536`, and `near_tied=true`.
- Start/end source, harness, and input identities were unchanged within each arm; retained input and harness identities were identical between arms.

The raw manifests, worker logs, and compact summaries are preserved in `C:\Users\USER\agent-workspaces\gw06-base-skeleton-phase2-scratch-20261003` as `baseline_79.json`, `candidate_79.json`, `baseline_workers.jsonl`, `candidate_workers.jsonl`, `comparison_79.json`, `task_metrics_79.json`, and `first_subsequent_by_skeleton_79.json`.

## Limits and acceptance state

The measured 21.55% batch improvement covers the retained GW6–GW9 79-key workload, not a complete optimizer search or production decision. No whole-decision time estimate follows from it. The earlier captain-term 12–20-minute whole-run extrapolation remains withdrawn.

Both variants ran sequentially on the same host without another heavy benchmark started concurrently. Ambient machine load was not independently sampled, so small timing differences may still include host variation. No predictions, new worlds, decisions, broad regression, or production actions were run.

The integer-key prototype remains isolated from the accepted branch. This report records benchmark and focused-equivalence evidence only; authoritative wrapper, exact-SHA CI, and Sol High review are separate acceptance gates and are recorded outside this report once complete.
