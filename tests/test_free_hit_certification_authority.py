"""Genuine certification authority — exercised through the REAL loader.

The decisive property is that the authority is DERIVED from a canonical
certification artifact loaded by ``four_gw_decision.load_certification_artifact``
and verified by RECOMPUTING ``certification_identity_of``, never trusted from a
caller or from a digest string.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import free_hit_certification_fixtures as cf  # noqa: E402
from fpl_brain import certified_bundle as cb  # noqa: E402
from fpl_brain import four_gw_decision as fg  # noqa: E402
from fpl_brain import free_hit_decision_authority as fa  # noqa: E402


def _write(tmp_path, payload=None, name="cert.json"):
    return cf.write_artifact(tmp_path, payload, name=name)


# ---------------------------------------------------------------------------
# A — coherent canonical certification
# ---------------------------------------------------------------------------


def test_A_a_real_artifact_loads_through_the_canonical_loader(tmp_path):
    path = _write(tmp_path)
    artifact = fg.load_certification_artifact(path)          # the REAL loader
    assert artifact["events"] == [5, 6, 7, 8]

    authority = fa.load_decision_authority(path)
    assert authority.certified_events == (5, 6, 7, 8)
    assert authority.planning_cutoff == cf.CUTOFF
    assert authority.data_snapshot_sha256 == cf.DATA_SNAPSHOT
    assert authority.problems() == []
    assert authority.loaded_from.endswith("cert.json")


def test_A_the_authority_is_anchored_to_the_certified_bundle(tmp_path):
    authority = fa.load_decision_authority(_write(tmp_path))
    bundle = authority.bundle_for(5)
    assert bundle is not None
    # A world identity conforming to THAT bundle has no disagreements at all.
    assert authority.disagreements_with(cf.identity_for(bundle), event=5) == []
    # And a bundle carrying a different certified event identity still validates.
    assert set(authority.certified_bundle_identity) == {5, 6, 7, 8}


# ---------------------------------------------------------------------------
# C — the certified event set is EXACT
# ---------------------------------------------------------------------------


def test_C_a_different_four_event_window_refuses(tmp_path):
    """GW6-GW9 is a legal four-event horizon and is still the WRONG one."""

    authority = fa.load_decision_authority(_write(tmp_path))
    assert authority.certified_events_problems((5, 6, 7, 8)) == []
    problems = authority.certified_events_problems((6, 7, 8, 9))
    assert problems and problems[0].startswith(fa.FH_CERTIFIED_EVENTS_MISMATCH)
    assert authority.certified_events_problems((5, 6, 7)) != []
    assert authority.certified_events_problems((5, 6, 7, 8, 9)) != []


# ---------------------------------------------------------------------------
# G / I / J — identity recomputation over the real artifact components
# ---------------------------------------------------------------------------


def test_J_a_copied_digest_on_an_altered_artifact_refuses(tmp_path):
    """A recognised identity on a different artifact cannot pass."""

    payload = cf.certification_artifact()
    copied = dict(payload)
    copied["planning_cutoff"] = "2026-09-16T12:00:00Z"        # altered component
    # The digest would no longer be the one this artifact's own fields produce.
    assert fg.certification_identity_of(copied) != payload["four_gw_certification_identity"]
    with pytest.raises(fa.FreeHitAuthorityError) as caught:
        fa.load_decision_authority(_write(tmp_path, copied))
    assert caught.value.reasons[0] == fa.FH_DECISION_AUTHORITY_REQUIRED


def test_G_mutating_one_certified_event_bundle_refuses(tmp_path):
    """The declared bundle identity no longer binds the bundle it names."""

    payload = cf.certification_artifact()
    doctored = json.loads(json.dumps(payload))
    doctored["certified_bundles"]["7"]["runs"] = {"xpts": 999}   # one event's run swapped
    with pytest.raises(Exception) as caught:
        fa.load_decision_authority(_write(tmp_path, doctored))
    assert "bundle" in str(caught.value).lower()


def test_I_mutating_the_committed_code_snapshot_invalidates_the_identity(tmp_path):
    """``code_snapshot_sha256`` is identity-bearing through the bundle identity."""

    payload = cf.certification_artifact()
    doctored = json.loads(json.dumps(payload))
    doctored["certified_bundles"]["6"]["code_snapshot_sha256"] = "sha256:" + "9" * 64
    with pytest.raises(Exception):
        fa.load_decision_authority(_write(tmp_path, doctored))


def test_the_identity_is_recomputed_not_trusted(tmp_path):
    payload = cf.certification_artifact()
    payload["four_gw_certification_identity"] = "sha256:" + "0" * 64   # merely asserted
    with pytest.raises(fa.FreeHitAuthorityError) as caught:
        fa.FreeHitDecisionAuthority.from_certification(payload, loaded_from="<test>")
    assert "declared identity is not the one its own" in str(caught.value)


def test_G2_a_correctly_rehashed_but_doctored_event_list_still_refuses(tmp_path):
    """Rehashing does not help: the declared events must BE the bundle events."""

    payload = cf.certification_artifact()
    doctored = json.loads(json.dumps(payload))
    doctored["events"] = [6, 7, 8, 9]
    doctored["four_gw_certification_identity"] = fg.certification_identity_of(doctored)
    with pytest.raises(Exception) as caught:
        fa.load_decision_authority(_write(tmp_path, doctored))
    assert "do not match the certified bundle events" in str(caught.value)


# ---------------------------------------------------------------------------
# D — cutoff authority
# ---------------------------------------------------------------------------


def test_D_the_cutoff_comes_from_the_certification(tmp_path):
    authority = fa.load_decision_authority(_write(tmp_path))
    assert authority.planning_cutoff == cf.CUTOFF
    other = cf.certification_artifact(cutoff="2026-09-16T09:30:00Z")
    moved = fa.load_decision_authority(_write(tmp_path, other, name="earlier.json"))
    assert moved.planning_cutoff == "2026-09-16T09:30:00Z"
    assert moved.planning_cutoff != authority.planning_cutoff


# ---------------------------------------------------------------------------
# E / F — data snapshot and planning context
# ---------------------------------------------------------------------------


def test_E_a_world_from_another_data_snapshot_refuses(tmp_path):
    authority = fa.load_decision_authority(_write(tmp_path))
    bundle = authority.bundle_for(5)
    foreign = cf.identity_for({**bundle, "data_snapshot_sha256": "sha256:" + "b" * 64})
    assert authority.disagreements_with(foreign, event=5) == ["identity data_snapshot_sha256"]


def test_F_a_world_from_another_planning_context_refuses(tmp_path):
    """Same events, same snapshot, different planning context: a different certificate."""

    authority = fa.load_decision_authority(_write(tmp_path))
    other = cf.certification_artifact(context="sha256:" + "q" * 64)
    moved = fa.load_decision_authority(_write(tmp_path, other, name="other-context.json"))
    assert moved.planning_context_hash != authority.planning_context_hash
    assert moved.certification_identity != authority.certification_identity


# ---------------------------------------------------------------------------
# H — run / model authority
# ---------------------------------------------------------------------------


def test_H_a_world_from_another_prediction_run_or_model_refuses(tmp_path):
    authority = fa.load_decision_authority(_write(tmp_path))
    bundle = authority.bundle_for(5)

    wrong_run = cf.identity_for({**bundle, "runs": {"xpts": 999, "minutes": 102, "team": 103}})
    assert authority.disagreements_with(wrong_run, event=5) == ["identity generation"]

    wrong_model = cf.identity_for(
        {**bundle, "model_versions": {"xpts": "xpts_v9.9.9", "minutes": "minutes_v1.6.0"}}
    )
    assert authority.disagreements_with(wrong_model, event=5) == ["identity model_config_identity"]

    wrong_code = cf.identity_for({**bundle, "code_snapshot_sha256": "sha256:" + "7" * 64})
    assert authority.disagreements_with(wrong_code, event=5) == ["identity source_snapshot_sha256"]

    wrong_snapshot = cf.identity_for({**bundle, "cutoff": "2026-09-16T12:00:00Z"})
    assert authority.disagreements_with(wrong_snapshot, event=5) == ["identity cutoff"]


# ---------------------------------------------------------------------------
# THE LOADER IS THE AUTHORITY BOUNDARY
# ---------------------------------------------------------------------------


def test_an_absent_artifact_refuses(tmp_path):
    with pytest.raises(fa.FreeHitAuthorityError) as caught:
        fa.load_decision_authority(tmp_path / "missing.json")
    assert caught.value.reasons[0] == fa.FH_DECISION_AUTHORITY_REQUIRED


def test_an_artifact_that_is_not_permitted_to_decide_refuses(tmp_path):
    payload = cf.certification_artifact()
    payload["decision_search_permitted"] = False
    with pytest.raises(fa.FreeHitAuthorityError):
        fa.load_decision_authority(_write(tmp_path, payload))


def test_an_artifact_without_a_completeness_audit_refuses(tmp_path):
    payload = cf.certification_artifact()
    payload.pop("history_completeness")
    payload["four_gw_certification_identity"] = fg.certification_identity_of(payload)
    with pytest.raises(fa.FreeHitAuthorityError):
        fa.load_decision_authority(_write(tmp_path, payload))


def test_provenance_distinguishes_a_loaded_authority_from_a_directly_built_one(tmp_path):
    """The production boundary is the LOADER, and provenance records which was used.

    A directly constructed authority is legitimate for a pure evaluator test, so it
    is not refused by a string marker -- markers are forgeable and would be
    security theatre.  What matters structurally is that the PRODUCTION adapter
    calls ``load_decision_authority`` and therefore consumes an artifact the
    canonical loader accepted.  This test pins that the two paths are
    distinguishable in the evidence.
    """

    path = _write(tmp_path)
    loaded = fa.load_decision_authority(path)
    assert loaded.loaded_from.endswith("cert.json")
    assert loaded.problems() == []

    artifact = fg.load_certification_artifact(path)
    built = fa.FreeHitDecisionAuthority.from_certification(artifact, loaded_from="")
    assert built.loaded_from == "in-memory artifact"
    assert built.certified_events == loaded.certified_events
    assert built.certification_identity == loaded.certification_identity


# ---------------------------------------------------------------------------
# CANONICAL LABELS — ONE definition shared by producer and verifier
# ---------------------------------------------------------------------------


def test_the_runs_and_model_labels_are_canonical_and_order_independent():
    assert fa.runs_label({"xpts": 1, "minutes": 2}) == fa.runs_label({"minutes": 2, "xpts": 1})
    assert fa.runs_label({"xpts": 1}) != fa.runs_label({"xpts": 2})
    assert fa.model_label({"a": "1", "b": "2"}) == fa.model_label({"b": "2", "a": "1"})
    assert fa.model_label({"a": "1"}) != fa.model_label({"a": "2"})
    # A list-shaped model_versions (the artifact's own shape) matches the mapping.
    assert fa.model_label([{"model_family": "a", "model_version": "1"}]) == fa.model_label({"a": "1"})


def test_the_bundle_identity_algorithm_is_the_canonical_one(tmp_path):
    """The fixture binds bundles with the SAME algorithm the certifier uses."""

    payload = cf.certification_artifact()
    for event, row in payload["certified_bundles"].items():
        assert payload["certified_bundle_identity"][event] == cb.canonical_bundle_identity(row)
