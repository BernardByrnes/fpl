"""Vacuous-guard harness: bypass one guard at a time and require the test to go red.

For every load-bearing guard, neutralise it in the real source, run the test that
claims to protect it, and require that test to FAIL.  A guard whose test still
passes is decorative.  Files are restored from a byte-for-byte backup and the
restoration is verified by digest.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path("K:/FPL-pe1")
METRICS = ROOT / "fpl_brain" / "walk_forward_metrics.py"
SCOREBOARD = ROOT / "fpl_brain" / "walk_forward_scoreboard.py"
WF = ROOT / "fpl_brain" / "walk_forward.py"
RULES = ROOT / "fpl_brain" / "scoring_rules.py"

#: The position clause of the canonical clean-sheet scoring rule.  It is the ONE
#: clause separating "the side did not concede while he was on" from "he was
#: awarded something for it", and two independent consumers depend on it.
_CS_POSITION_CLAUSE = (
    RULES,
    "            and self.clean_sheet_points_for(position) > 0\n        )",
    "            # BYPASS: position clause removed\n        )",
)
#: The predecessor's exact defect, re-injected: collapse the per-event run map into
#: one family-keyed map, so only the last run per family survives.
_PER_EVENT_COLLAPSE = (
    WF,
    "        return {\n"
    "            str(event): self.runs_for_event(event)\n"
    "            for event in sorted({int(e) for e, _family, _run in self.per_event_runs})\n"
    "        }",
    "        return {family: run for _event, family, run in self.per_event_runs}  # BYPASS",
)
#: The scoreboard's cross-check of each population's recorded runs against the
#: runs it resolved from the anchor.
_WIRING_VERIFY_RUNS = (
    SCOREBOARD,
    "    _verify_recorded_runs(\n"
    "        model_population, populations, xpts_runs=xpts_runs, baseline_runs=baseline_runs\n"
    "    )",
    "    None  # BYPASS: the recorded-run cross-check is never called",
)
_AGREE_GUARD = (
    SCOREBOARD,
    "            if int(recorded) != int(run_id):",
    "            if False:  # BYPASS",
)

MULTI_BYPASS_OWN_CHECKS = [
    (
        "PE2E-P2-01: the shared helper's position clause drives the CALIBRATION target",
        [_CS_POSITION_CLAUSE],
        "tests/test_pe2e_evaluation_contracts.py::test_the_mc_clean_sheet_brier_target_matches_the_producer",
    ),
    (
        "PE2E-P2-01: the SAME clause drives the SCOREBOARD target (one definition)",
        [_CS_POSITION_CLAUSE],
        "tests/test_pe2e_evaluation_contracts.py::test_the_scoreboard_clean_sheet_target_is_zero_for_a_forward",
    ),
    (
        "PE2E-P2-01: a forward is never a clean-sheet scoring event",
        [_CS_POSITION_CLAUSE],
        "tests/test_pe2e_evaluation_contracts.py::test_a_forward_clean_sheet_is_not_a_clean_sheet_scoring_event",
    ),
    (
        "PE2E-P2-02: the shape-agnostic multi-event regression catches the collapse",
        [_PER_EVENT_COLLAPSE],
        "tests/test_pe2e_evaluation_contracts.py::test_a_real_multi_event_build_preserves_every_events_run",
    ),
    (
        "PE2E-P2-02: the exact-shape multi-event regression catches the collapse",
        [_PER_EVENT_COLLAPSE],
        "tests/test_pe2e_evaluation_contracts.py::test_a_multi_event_identity_preserves_every_event_run",
    ),
    (
        "PE2E-P2-02: the scoreboard's digest moves when a run changes",
        [_PER_EVENT_COLLAPSE],
        "tests/test_pe2e_evaluation_contracts.py::test_the_scoreboard_artifact_changes_when_one_events_run_changes",
    ),
    (
        "PE2E-P2-02: the recorded-run cross-check is WIRED into build_scoreboard",
        [_WIRING_VERIFY_RUNS],
        "tests/test_pe2e_evaluation_contracts.py::test_the_scoreboard_verifies_the_populations_recorded_runs",
    ),
    (
        "PE2E-P2-02: the disagreement guard itself raises",
        [_AGREE_GUARD],
        "tests/test_pe2e_evaluation_contracts.py::test_the_scoreboard_rejects_a_recorded_run_that_disagrees_with_the_anchor",
    ),
]

CHECKS = [
    (
        "same-population gate: an incomplete arm must produce no metric",
        SCOREBOARD,
        "            wf.assert_same_population(scorable, covered)",
        "            None  # BYPASS",
        "tests/test_walk_forward_scoreboard.py::test_an_arm_that_cannot_cover_the_population_gets_no_metric_at_all",
    ),
    (
        "same-population gate: the other arms must be named too",
        SCOREBOARD,
        "            wf.assert_same_population(scorable, covered)",
        "            None  # BYPASS",
        "tests/test_walk_forward_scoreboard.py::test_the_other_arms_are_still_measured_when_one_arm_is_unavailable",
    ),
    (
        "population digest: a constant digest must not satisfy the mismatch test",
        WF,
        '    lines = sorted("|".join(str(int(part)) for part in key) for key in keys)',
        '    lines = []  # BYPASS',
        "tests/test_walk_forward_scoreboard.py::test_a_mismatch_that_only_changes_the_digest_is_still_caught",
    ),
    (
        "arm alignment: mismatched sequence lengths must be refused",
        METRICS,
        "    if len(predicted) != len(actual):",
        "    if False:  # BYPASS",
        "tests/test_walk_forward_metrics.py::test_misaligned_arms_are_refused",
    ),
    (
        "spearman: a constant side must not divide by zero",
        METRICS,
        "    if spread_x == 0.0 or spread_y == 0.0:",
        "    if False:  # BYPASS",
        "tests/test_walk_forward_metrics.py::test_a_constant_side_makes_the_correlation_undefined_not_zero_or_one",
    ),
    (
        "spearman: end-to-end undefined state on a constant baseline arm",
        METRICS,
        "    if spread_x == 0.0 or spread_y == 0.0:",
        "    if False:  # BYPASS",
        "tests/test_walk_forward_scoreboard.py::test_a_constant_baseline_has_an_undefined_correlation_but_real_error",
    ),
    (
        "brier: an out-of-range probability must raise, not be clipped",
        METRICS,
        "        if value < 0.0 or value > 1.0:",
        "        if False:  # BYPASS",
        "tests/test_walk_forward_metrics.py::test_an_out_of_range_probability_fails_closed_instead_of_being_clipped",
    ),
    (
        "brier: the scoreboard must fail closed on a stored probability outside [0, 1]",
        SCOREBOARD,
        "                if not 0.0 <= probability <= 1.0:",
        "                if False:  # BYPASS",
        "tests/test_walk_forward_scoreboard.py::test_a_probability_outside_the_unit_interval_fails_closed",
    ),
    (
        "brier: a non-binary realised outcome must raise",
        METRICS,
        '        if value not in (0.0, 1.0):\n'
        '            raise MetricInputError(f"realised outcome {value!r} is not binary")\n'
        '    if not p:',
        '        if False:  # BYPASS\n'
        '            raise MetricInputError(f"realised outcome {value!r} is not binary")\n'
        '    if not p:',
        "tests/test_walk_forward_metrics.py::test_a_non_binary_outcome_fails_closed",
    ),
    (
        "quantile: the coverage function rejects a reversed interval",
        METRICS,
        '        if lo > hi:\n'
        '            raise MetricInputError(f"lower quantile {lo!r} exceeds upper quantile {hi!r}")\n'
        '    if not value:',
        '        if False:  # BYPASS\n'
        '            raise MetricInputError(f"lower quantile {lo!r} exceeds upper quantile {hi!r}")\n'
        '    if not value:',
        "tests/test_walk_forward_metrics.py::test_a_reversed_interval_fails_closed",
    ),
    (
        "quantile: the width function rejects a reversed pair",
        METRICS,
        '        if lo > hi:\n'
        '            raise MetricInputError(f"lower quantile {lo!r} exceeds upper quantile {hi!r}")\n'
        '    if not low:',
        '        if False:  # BYPASS\n'
        '            raise MetricInputError(f"lower quantile {lo!r} exceeds upper quantile {hi!r}")\n'
        '    if not low:',
        "tests/test_walk_forward_metrics.py::test_mean_interval_width_rejects_a_reversed_pair",
    ),
    (
        "top-k: a population smaller than the declared k must be unavailable, not rescaled",
        METRICS,
        "    if population < k:",
        "    if False:  # BYPASS",
        "tests/test_walk_forward_metrics.py::test_a_population_smaller_than_the_declared_k_is_unavailable_not_rescaled",
    ),
    (
        "top-k: a duplicate player in one ranked set must be refused",
        METRICS,
        "        if identifier in seen:",
        "        if False:  # BYPASS",
        "tests/test_walk_forward_metrics.py::test_a_duplicate_player_in_one_ranked_set_is_refused",
    ),
    (
        "non-finite: NaN must raise rather than be serialised",
        METRICS,
        "        if not math.isfinite(number):",
        "        if False:  # BYPASS",
        "tests/test_walk_forward_metrics.py::test_a_non_finite_value_is_refused_rather_than_propagated",
    ),
    (
        "immutability: the same identity with different bytes must fail closed",
        SCOREBOARD,
        '        raise ScoreboardIdentityCollision(\n'
        '            f"{kind} artifact {target.name} already exists with different bytes; "\n'
        '            "one identity must always describe one content"\n'
        '        )',
        "        return target  # BYPASS",
        "tests/test_walk_forward_scoreboard.py::test_the_same_identity_with_different_bytes_fails_closed",
    ),
    (
        "minutes dependency: an ambiguous closure must raise",
        SCOREBOARD,
        "    if len(resolved) > 1:",
        "    if False:  # BYPASS",
        "tests/test_walk_forward_scoreboard.py::test_an_ambiguous_minutes_dependency_fails_closed",
    ),
    (
        "placeholder: a scheduled placeholder must not be scored",
        WF,
        "    if any((int(key[0]), int(key[1]), int(fid)) in placeholder_pairs for fid in fixtures):",
        "    if False:  # BYPASS",
        "tests/test_walk_forward_population.py::test_D_a_scheduled_placeholder_outcome_is_excluded_not_zeroed",
    ),
    (
        "top-k tie policy: the ascending-id rule must be the one in force",
        METRICS,
        "    parsed.sort(key=lambda pair: (-pair[0], pair[1]))",
        "    parsed.sort(key=lambda pair: (-pair[0], -pair[1]))  # BYPASS",
        "tests/test_walk_forward_metrics.py::test_the_tie_policy_is_load_bearing_not_decorative",
    ),
]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


#: Checks that need more than one guard neutralised at once.  A path protected by
#: two independent guards is only provably live when BOTH are bypassed; a
#: single-bypass run passes for the wrong reason and reads as a vacuous guard.
_COVERAGE_REVERSAL = (
    METRICS,
    '        if lo > hi:\n'
    '            raise MetricInputError(f"lower quantile {lo!r} exceeds upper quantile {hi!r}")\n'
    '    if not value:',
    '        if False:  # BYPASS\n'
    '            raise MetricInputError(f"lower quantile {lo!r} exceeds upper quantile {hi!r}")\n'
    '    if not value:',
)
_WIDTH_REVERSAL = (
    METRICS,
    '        if lo > hi:\n'
    '            raise MetricInputError(f"lower quantile {lo!r} exceeds upper quantile {hi!r}")\n'
    '    if not low:',
    '        if False:  # BYPASS\n'
    '            raise MetricInputError(f"lower quantile {lo!r} exceeds upper quantile {hi!r}")\n'
    '    if not low:',
)

MULTI_BYPASS_CHECKS = [
    (
        "quantile: the scoreboard fails closed on a reversed stored interval "
        "(both independent guards neutralised)",
        [_COVERAGE_REVERSAL, _WIDTH_REVERSAL],
        "tests/test_walk_forward_scoreboard.py::test_a_reversed_stored_interval_fails_closed",
    ),
]


def run(selector: str) -> bool:
    # Bytecode invalidation is not optional here.  CPython's ``.pyc`` header stores
    # the source mtime in whole SECONDS, and two bypasses of the same guard can
    # produce the same file SIZE; writing both within one second therefore lets the
    # second run import the FIRST bypass's bytecode and report a live guard as
    # vacuous.  Clearing the caches makes every run read the bytes on disk.
    for cache in ROOT.rglob("__pycache__"):
        shutil.rmtree(cache, ignore_errors=True)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", selector, "-q", "--no-header", "-p", "no:cacheprovider"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    return result.returncode == 0


def main() -> int:
    originals = {path: path.read_bytes() for path in {METRICS, SCOREBOARD, WF, RULES}}
    digests = {path: digest(path) for path in originals}
    failures: list[str] = []
    checks = (
        [(label, [(path, needle, replacement)], selector)
         for label, path, needle, replacement, selector in CHECKS]
        + MULTI_BYPASS_CHECKS
        + MULTI_BYPASS_OWN_CHECKS
    )
    try:
        for label, edits, selector in checks:
            # Anchors are written with LF.  Two of the files under test are checked
            # out with CRLF (core.autocrlf), so matching happens on a newline-
            # normalised copy and each file's own convention is restored on write.
            newlines = {
                path: "\r\n" if b"\r\n" in originals[path] else "\n" for path in originals
            }
            originals_text = {
                path: originals[path].decode("utf-8").replace("\r\n", "\n") for path in originals
            }
            bodies = dict(originals_text)
            ok = True
            for path, needle, replacement in edits:
                if bodies[path].count(needle) != 1:
                    failures.append(
                        f"{label}: anchor found {bodies[path].count(needle)} times in {path.name}, need exactly 1"
                    )
                    ok = False
                    break
                bodies[path] = bodies[path].replace(needle, replacement)
            if not ok:
                continue
            for path, body in bodies.items():
                text = body.replace("\n", newlines[path]) if newlines[path] == "\r\n" else body
                path.write_bytes(text.encode("utf-8"))
            try:
                passed = run(selector)
            finally:
                for path, body in originals.items():
                    path.write_bytes(body)
            verdict = "RED (guard is live)" if not passed else "STILL GREEN (guard is VACUOUS)"
            print(f"[{verdict}] {label}")
            if passed:
                failures.append(label)
    finally:
        for path, body in originals.items():
            path.write_bytes(body)
        for path, expected in digests.items():
            actual = digest(path)
            status = "restored" if actual == expected else "NOT RESTORED"
            print(f"  {status}: {path.name} {actual[:16]}")
            if actual != expected:
                failures.append(f"{path.name} was not restored")

    print()
    if failures:
        print("VACUOUS-GUARD FAILURES:")
        for item in failures:
            print("  -", item)
        return 1
    total = len(CHECKS) + len(MULTI_BYPASS_CHECKS) + len(MULTI_BYPASS_OWN_CHECKS)
    print(f"all {total} guards are live and every file was restored byte-identically")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
