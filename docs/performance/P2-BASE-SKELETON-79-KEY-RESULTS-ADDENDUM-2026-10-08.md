# Addendum — combined base-skeleton optimization scope

Date: 2026-10-08 UTC. This addendum corrects the scope description in the 2026-10-03 report; it does not alter or replace its retained benchmark measurements.

The combined candidate reviewed at commit `f7bd00647a6ed74d64da8ee247457278f091c526` contains two base-skeleton changes in `fpl_brain/manager_worlds.py`:

1. The watched-appearance integer key (`appearance_mask & watch_mask`) described in the original report.
2. Starter-sum reuse, keyed by the exact ordered `starter_idx` and computed with the same `sum(mean_core[i] for i in starter_idx)` expression used by the reference path (`manager_worlds.py:469-482`, consumed at `:537`).

The existing oracle coverage in `tests/test_base_skeleton_integer_keys.py:139-217` checks exact equality and cache isolation for the combined code path. This is correctness/equivalence evidence; it is not a new combined-variant batch benchmark.

The original report's 21.55% measured batch improvement and 184 retained route/event equivalence rows were produced for the integer-key variant only. They must not be attributed to, or represented as performance/equivalence batch evidence for, the combined variant. No new benchmark was run for this addendum.
