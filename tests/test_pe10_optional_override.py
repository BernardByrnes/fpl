from __future__ import annotations

import pytest

from fpl_brain import four_gw_decision as fg
from fpl_brain import generation_store as gs
from scripts.run_four_gw_decision import _override_captured_at


def test_absent_override_key_normalizes_to_none():
    assert _override_captured_at({"bank": 17}) is None


def test_explicit_none_override_normalizes_to_none():
    assert _override_captured_at({"override": None}) is None


def test_mapping_override_preserves_captured_at():
    captured_at = "2026-09-28T10:09:00Z"
    assert _override_captured_at({"override": {"captured_at": captured_at}}) == captured_at


def test_non_mapping_override_fails_closed_with_structured_refusal():
    with pytest.raises(gs.DecisionRecordInvalid) as refusal:
        _override_captured_at({"override": "bad"})

    assert refusal.value.token == gs.DIAG_DECISION_RECORD_INVALID
    assert "manager_state.override" in str(refusal.value)


def test_real_override_still_obeys_cutoff_guard():
    captured_at = _override_captured_at(
        {"override": {"captured_at": "2026-09-28T10:09:00Z"}}
    )

    stale = fg.verify_cutoff_covers_override(
        planning_cutoff="2026-09-28T10:08:59Z",
        override_captured_at=captured_at,
    )
    fresh = fg.verify_cutoff_covers_override(
        planning_cutoff="2026-09-28T10:10:08Z",
        override_captured_at=captured_at,
    )

    assert stale["pass"] is False
    assert stale["status"] == fg.CUTOFF_PRECEDES_OVERRIDE
    assert fresh["pass"] is True
    assert fresh["status"] == "PASS"
