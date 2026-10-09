from __future__ import annotations

import pytest

from fpl_brain.chip_assessment_store import ChipAssessmentStoreError, _validate_run_id


@pytest.mark.parametrize("value", ["../outside", "subdir/name", r"subdir\name", "..", "", "bad id"])
def test_assessment_run_id_rejects_path_components(value):
    with pytest.raises(ChipAssessmentStoreError, match="run_id"):
        _validate_run_id(value)


@pytest.mark.parametrize("value", ["run-123", "run_123", "4b1d79a6-7a1d-4b9c-8a23-ef67a62d2eb0"])
def test_assessment_run_id_accepts_safe_identifiers(value):
    assert _validate_run_id(value) == value
