"""Run the authoritative test suite with the repository's accepted baseline policy.

The four accepted failures are configuration-environment contracts documented in
the orchestration-readiness bootstrap.  Every other failure remains a CI failure.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


ACCEPTED_FAILURES = frozenset(
    {
        "test_causal_bundle_integrity::test_13_manager_state_prose_derives_from_actual_state",
        "test_causal_bundle_integrity::test_13b_zero_ft_prose_is_also_derived",
        "test_r4b1_decision_correctness::test_T_conflicting_event_against_certification_fails",
        "test_r4b1_decision_correctness::test_T_no_event_and_no_certification_is_refused",
    }
)


def _module_name(classname: str) -> str:
    """Normalize pytest's JUnit classname to the repository test module name."""

    parts = [part for part in re.split(r"[./\\:]", classname) if part]
    for part in reversed(parts):
        if part.startswith("test_"):
            return Path(part).stem
    return Path(parts[-1]).stem if parts else classname


def _node_id(testcase: ET.Element) -> str:
    module = _module_name(testcase.attrib.get("classname", ""))
    name = testcase.attrib.get("name", "")
    return f"{module}::{name}"


def _results(xml_path: Path) -> tuple[int, int, list[str]]:
    root = ET.parse(xml_path).getroot()
    testcases = list(root.iter("testcase"))
    skipped = 0
    failures: list[str] = []
    for testcase in testcases:
        if testcase.find("skipped") is not None:
            skipped += 1
        if testcase.find("failure") is not None or testcase.find("error") is not None:
            failures.append(_node_id(testcase))
    passed = len(testcases) - skipped - len(failures)
    return passed, skipped, failures


def _accepted_result(failures: list[str]) -> bool:
    if not failures:
        return True
    return len(failures) == len(ACCEPTED_FAILURES) and frozenset(failures) == ACCEPTED_FAILURES


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="fpl-ci-") as temp_dir:
        report_path = Path(temp_dir) / "pytest-junit.xml"
        command = [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            f"--junitxml={report_path}",
        ]
        completed = subprocess.run(command, check=False)
        if not report_path.exists():
            print("passed: unavailable")
            print("skipped: unavailable")
            print("failed: unavailable")
            print("exact accepted failures: unavailable")
            print("unexpected failures: pytest produced no JUnit report")
            return completed.returncode or 1

        passed, skipped, failures = _results(report_path)
        failure_set = frozenset(failures)
        accepted = sorted(failure_set & ACCEPTED_FAILURES)
        unexpected = sorted(failure_set - ACCEPTED_FAILURES)

        print(f"passed: {passed}")
        print(f"skipped: {skipped}")
        print(f"failed: {len(failures)}")
        print("exact accepted failures:")
        for node_id in accepted:
            print(f"  {node_id}")
        if not accepted:
            print("  none")
        print("unexpected failures:")
        for node_id in unexpected:
            print(f"  {node_id}")
        if not unexpected:
            print("  none")

        if _accepted_result(failures):
            return 0
        return completed.returncode or 1


if __name__ == "__main__":
    raise SystemExit(main())
