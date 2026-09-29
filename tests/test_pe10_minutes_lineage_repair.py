from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

import certify_four_gw as certifier  # noqa: E402
import freeze_predictions  # noqa: E402
import generation_fixtures as gf  # noqa: E402
import test_pe9_certification_integration as pe9  # noqa: E402
from fpl_brain import (  # noqa: E402
    certified_bundle as cb,
    joint_minutes,
    minutes_coherence,
    minutes_model,
    substitution_model,
)


def _production_shape_world(
    *,
    inverted_minutes_write_order: bool = False,
    joint_minutes_version: str | None = None,
    mc_minutes_run_id: int | None = None,
    include_xpts_rows: bool = True,
    include_mc_rows: bool = True,
    fixtures: int = 1,
):
    """Build the frozen production family shape with several minutes variants."""

    conn = pe9.connect_database(":memory:")
    pe9._base_world(conn)
    event = 6
    fixture_ids = [1000 + index for index in range(fixtures)]
    with conn:
        pe9._add_event(conn, event)
        for fixture_id in fixture_ids:
            pe9._add_fixture(conn, fixture_id, event, 1, 2)
    gf.prepare_fixture_snapshot(conn, [event], cutoff=pe9.CUTOFF)

    if inverted_minutes_write_order:
        minute_ids = {
            "joint": 100,
            "coherent": 101,
            "positional": 102,
            "substitution": 103,
            "plain": 104,
        }
    else:
        minute_ids = {
            "plain": 100,
            "coherent": 101,
            "positional": 102,
            "substitution": 103,
            "joint": 104,
        }
    ids = {
        "minutes_plain": minute_ids["plain"],
        "minutes_coherent": minute_ids["coherent"],
        "minutes_positional": minute_ids["positional"],
        "minutes_substitution": minute_ids["substitution"],
        "minutes_joint": minute_ids["joint"],
        "team_strength_v1": 105,
        "player_rates_v1": 106,
        "xpts_v1": 107,
        "monte_carlo_v1": 108,
    }
    minute_versions = {
        "minutes_plain": minutes_model.MINUTES_MODEL_VERSION,
        "minutes_coherent": minutes_model.MINUTES_COHERENT_MODEL_VERSION,
        "minutes_positional": minutes_coherence.MINUTES_POSITIONAL_MODEL_VERSION,
        "minutes_substitution": substitution_model.MINUTES_SUBSTITUTION_MODEL_VERSION,
        "minutes_joint": joint_minutes_version or joint_minutes.JOINT_MINUTES_MODEL_VERSION,
    }
    execution_run_uuid = gf.fixture_run_provenance(conn, event)["execution_run_uuid"]

    with conn:
        for name in (
            "minutes_plain",
            "minutes_coherent",
            "minutes_positional",
            "minutes_substitution",
            "minutes_joint",
        ):
            pe9._run(
                conn,
                ids[name],
                "minutes_v1",
                event,
                version=minute_versions[name],
            )
        pe9._run(conn, ids["team_strength_v1"], "team_strength_v1", event)
        pe9._run(conn, ids["player_rates_v1"], "player_rates_v1", event)
        pe9._run(conn, ids["xpts_v1"], "xpts_v1", event)
        pe9._run(conn, ids["monte_carlo_v1"], "monte_carlo_v1", event)

        if include_xpts_rows:
            for fixture_id in fixture_ids:
                pe9._xpts_row(
                    conn,
                    ids["xpts_v1"],
                    fixture_id,
                    event,
                    minutes_run_id=ids["minutes_joint"],
                    team_run_id=ids["team_strength_v1"],
                    rate_run_id=ids["player_rates_v1"],
                )
        if include_mc_rows:
            for fixture_id in fixture_ids:
                pe9._mc_row(
                    conn,
                    ids["monte_carlo_v1"],
                    fixture_id,
                    event,
                    xpts_run_id=ids["xpts_v1"],
                    minutes_run_id=(
                        ids["minutes_joint"] if mc_minutes_run_id is None else mc_minutes_run_id
                    ),
                    team_run_id=ids["team_strength_v1"],
                    rate_run_id=ids["player_rates_v1"],
                )
    return conn, ids, fixture_ids, execution_run_uuid


def _selected_bundle(conn, execution_run_uuid: str) -> dict[str, int]:
    return certifier.certified_bundle_runs(
        conn,
        event=6,
        cutoff=pe9.CUTOFF,
        execution_run_uuid=execution_run_uuid,
    )


def _validate_selected_bundle(conn, runs: dict[str, int]):
    snapshot = gf.fixture_snapshot(conn)
    return cb.certified_bundle_from_explicit_ids(
        conn,
        event=6,
        cutoff=pe9.CUTOFF,
        runs=runs,
        required_versions=cb.declared_required_versions(),
        data_snapshot_sha256=str(snapshot["sha256"]),
    )


def test_production_shape_selects_joint_minutes_and_exact_xpts_mc_closure():
    conn, ids, _fixtures, execution_uuid = _production_shape_world()
    try:
        runs = _selected_bundle(conn, execution_uuid)
        bundle = _validate_selected_bundle(conn, runs)
        xpts_inputs = cb._upstream_run_ids(conn, "xpts_v1", runs["xpts_v1"])
        mc_inputs = cb._upstream_run_ids(conn, "monte_carlo_v1", runs["monte_carlo_v1"])

        assert runs == {
            "minutes_v1": ids["minutes_joint"],
            "team_strength_v1": ids["team_strength_v1"],
            "player_rates_v1": ids["player_rates_v1"],
            "xpts_v1": ids["xpts_v1"],
            "monte_carlo_v1": ids["monte_carlo_v1"],
        }
        assert xpts_inputs["minutes_v1"] == mc_inputs["minutes_v1"] == runs["minutes_v1"]
        assert bundle.model_versions["minutes_v1"] == joint_minutes.JOINT_MINUTES_MODEL_VERSION
    finally:
        conn.close()


def test_joint_minutes_is_selected_when_plain_minutes_has_the_higher_run_id():
    conn, ids, _fixtures, execution_uuid = _production_shape_world(
        inverted_minutes_write_order=True
    )
    try:
        assert ids["minutes_plain"] > ids["minutes_joint"]
        runs = _selected_bundle(conn, execution_uuid)
        assert runs["minutes_v1"] == ids["minutes_joint"]
        assert _validate_selected_bundle(conn, runs).model_versions["minutes_v1"] == (
            joint_minutes.JOINT_MINUTES_MODEL_VERSION
        )
    finally:
        conn.close()


def test_newer_unrelated_minutes_run_cannot_hijack_dependency_selection():
    conn, ids, _fixtures, execution_uuid = _production_shape_world(
        inverted_minutes_write_order=True
    )
    try:
        with conn:
            pe9._run(
                conn,
                109,
                "minutes_v1",
                6,
                version=minutes_model.MINUTES_MODEL_VERSION,
            )
        runs = _selected_bundle(conn, execution_uuid)
        assert runs["minutes_v1"] == ids["minutes_joint"]
        _validate_selected_bundle(conn, runs)
    finally:
        conn.close()


def test_xpts_and_monte_carlo_minutes_mismatch_is_refused():
    conn, ids, _fixtures, execution_uuid = _production_shape_world(
        inverted_minutes_write_order=True,
        mc_minutes_run_id=104
    )
    try:
        runs = _selected_bundle(conn, execution_uuid)
        assert runs["minutes_v1"] == ids["minutes_joint"]
        with pytest.raises(cb.BundleIncoherent, match="monte_carlo_v1 run"):
            _validate_selected_bundle(conn, runs)
    finally:
        conn.close()


def test_bundle_minutes_mismatch_with_xpts_dependency_is_refused():
    conn, ids, _fixtures, execution_uuid = _production_shape_world()
    try:
        runs = _selected_bundle(conn, execution_uuid)
        runs["minutes_v1"] = ids["minutes_plain"]
        with pytest.raises(cb.BundleIncoherent, match="xpts_v1 run .* references minutes_v1"):
            _validate_selected_bundle(conn, runs)
    finally:
        conn.close()


def test_unapproved_joint_minutes_version_fails_even_when_plain_v180_exists():
    conn, ids, _fixtures, execution_uuid = _production_shape_world(
        joint_minutes_version="minutes_v1.4.9"
    )
    try:
        runs = _selected_bundle(conn, execution_uuid)
        assert runs["minutes_v1"] == ids["minutes_joint"]
        assert ids["minutes_plain"] in {
            int(row[0])
            for row in conn.execute(
                "SELECT id FROM projection_runs WHERE model_family='minutes_v1'"
            )
        }
        with pytest.raises(cb.BundleIncoherent) as caught:
            _validate_selected_bundle(conn, runs)
        assert cb.DIAG_UNSUPPORTED_MODEL_VERSION in str(caught.value)
        assert "minutes_v1.4.9" in str(caught.value)
    finally:
        conn.close()


@pytest.mark.parametrize(
    ("include_xpts_rows", "include_mc_rows", "missing_family"),
    [(False, True, "xPts run"), (True, False, "Monte Carlo run")],
)
def test_incomplete_dependency_closure_is_refused(
    include_xpts_rows, include_mc_rows, missing_family
):
    conn, _ids, _fixtures, execution_uuid = _production_shape_world(
        include_xpts_rows=include_xpts_rows,
        include_mc_rows=include_mc_rows,
    )
    try:
        with pytest.raises(cb.BundleIncoherent, match=missing_family):
            _selected_bundle(conn, execution_uuid)
    finally:
        conn.close()


def test_all_xpts_rows_must_agree_on_the_dependency_closure():
    conn, ids, fixture_ids, execution_uuid = _production_shape_world(fixtures=2)
    try:
        with conn:
            pe9._xpts_row(
                conn,
                ids["xpts_v1"],
                fixture_ids[0],
                6,
                minutes_run_id=ids["minutes_plain"],
                team_run_id=ids["team_strength_v1"],
                rate_run_id=ids["player_rates_v1"],
                player_id=2,
            )
        with pytest.raises(cb.BundleIncoherent, match="different upstream combination"):
            _selected_bundle(conn, execution_uuid)
    finally:
        conn.close()


def test_all_monte_carlo_rows_must_agree_on_the_dependency_closure():
    conn, ids, fixture_ids, execution_uuid = _production_shape_world(fixtures=2)
    try:
        with conn:
            pe9._mc_row(
                conn,
                ids["monte_carlo_v1"],
                fixture_ids[0],
                6,
                xpts_run_id=ids["xpts_v1"],
                minutes_run_id=ids["minutes_plain"],
                team_run_id=ids["team_strength_v1"],
                rate_run_id=ids["player_rates_v1"],
                player_id=2,
            )
        with pytest.raises(cb.BundleIncoherent, match="different upstream combination"):
            _selected_bundle(conn, execution_uuid)
    finally:
        conn.close()


def test_blank_event_uses_unique_execution_rows_when_no_dependency_rows_exist():
    conn, ids, _fixtures, execution_uuid = _production_shape_world(fixtures=0)
    try:
        runs = _selected_bundle(conn, execution_uuid)
        bundle = _validate_selected_bundle(conn, runs)
        assert runs["minutes_v1"] == ids["minutes_joint"]
        assert bundle.zero_fixture_event
    finally:
        conn.close()


def test_startup_authority_check_refuses_drift_before_loading_config(monkeypatch, capsys):
    original = cb.declared_required_versions

    def drifted_versions():
        versions = original()
        versions["minutes_v1"] = minutes_model.MINUTES_MODEL_VERSION
        return versions

    monkeypatch.setattr(cb, "declared_required_versions", drifted_versions)
    with pytest.raises(ValueError, match="MINUTES_LINEAGE_AUTHORITY_MISMATCH"):
        certifier.assert_minutes_authority_consistency()
    assert certifier.main([]) == 2
    assert "MINUTES_LINEAGE_AUTHORITY_MISMATCH" in capsys.readouterr().err


def test_resolved_xpts_minutes_run_fails_fast_on_version_mismatch():
    conn, ids, _fixtures, _execution_uuid = _production_shape_world()
    try:
        freeze_predictions.validate_certified_minutes_run(conn, ids["minutes_joint"])
        with pytest.raises(freeze_predictions._ReadinessFailure) as caught:
            freeze_predictions.validate_certified_minutes_run(conn, ids["minutes_plain"])
        assert any("MINUTES_LINEAGE_VERSION_MISMATCH" in reason for reason in caught.value.reasons)
    finally:
        conn.close()


def test_selected_minutes_version_check_is_wired_before_xpts_build():
    source = Path(freeze_predictions.__file__).read_text(encoding="utf-8")
    selected_check = source.index("validate_certified_minutes_run(conn, int(xpts_inputs[\"minutes\"]))")
    xpts_build = source.index("built = _build_xpts(conn, event, cutoff, xpts_inputs, rules")
    assert selected_check < xpts_build


def test_historical_accepted_joint_minutes_lineage_remains_the_declared_contract():
    conn, ids, _fixtures, execution_uuid = _production_shape_world()
    try:
        assert cb.declared_required_versions()["minutes_v1"] == joint_minutes.JOINT_MINUTES_MODEL_VERSION
        runs = _selected_bundle(conn, execution_uuid)
        result = _validate_selected_bundle(conn, runs)
        assert result.runs["minutes_v1"] == ids["minutes_joint"]
        assert result.model_versions["minutes_v1"] == "minutes_v1.5.2"
    finally:
        conn.close()
