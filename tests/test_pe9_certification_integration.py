"""PE-9 — certification integration.

Deterministic and synthetic-first: every world is built in a temporary database
from explicit rows, so a certification result can be traced to the exact ids that
produced it.  Nothing here regenerates a prediction, promotes a model or changes a
calibration; PE-9 decides only which artifacts may enter the certified bundle.

The 24 hard test cases declared by the PE-9 contract are each covered by a test
named for them.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from fpl_brain import certified_bundle as cb
from fpl_brain import four_gw_decision as fg
from fpl_brain import route_comparator as rc
from fpl_brain import route_optimizer as ro
from fpl_brain.database import connect_database

CUTOFF = "2026-09-19T11:00:00Z"
OTHER_CUTOFF = "2026-09-18T11:00:00Z"
CODE_SNAPSHOT = "codehash"
DATA_SNAPSHOT = "sha256:" + "d" * 64
CONTEXT_HASH = "ctx"

# The REAL declared versions, read from the one source rather than restated, so a
# future version bump moves the fixtures with it instead of silently un-pinning.
VERSIONS = cb.declared_required_versions()

#: The four-event normal-transfer horizon the contract fixes.
HORIZON = (5, 6, 7, 8)


# ---------------------------------------------------------------------------
# Synthetic world
# ---------------------------------------------------------------------------


def _base_world(conn) -> None:
    with conn:
        for position_id, short in ((1, "GKP"), (2, "DEF"), (3, "MID"), (4, "FWD")):
            conn.execute(
                "INSERT INTO positions(id, singular_name_short, raw_json, updated_at)"
                " VALUES (?,?, '{}','2026-09-01T00:00:00Z')",
                (position_id, short),
            )
        conn.execute(
            "INSERT INTO fetch_runs(id, started_at, status, trigger)"
            " VALUES (1,'2026-09-01T00:00:00Z','success','test')"
        )
        for team_id, name in ((1, "One"), (2, "Two")):
            conn.execute(
                "INSERT INTO teams(id, name, short_name, raw_json, updated_at)"
                " VALUES (?,?,?, '{}','2026-09-01T00:00:00Z')",
                (team_id, name, name[:3].upper()),
            )
        for player_id, name in ((1, "Alpha"), (2, "Beta")):
            conn.execute(
                "INSERT INTO players(id, web_name, team_id, element_type, is_active, first_seen_at,"
                " last_seen_at, raw_json, updated_at)"
                " VALUES (?,?,1,3,1,'2026-09-01T00:00:00Z','2026-09-01T00:00:00Z','{}',"
                "'2026-09-01T00:00:00Z')",
                (player_id, name),
            )


def _add_event(conn, event: int) -> None:
    conn.execute(
        "INSERT INTO events(id, name, deadline_time, finished, raw_json, updated_at)"
        " VALUES (?,?,?,0,'{}','2026-09-01T00:00:00Z')",
        (event, f"GW{event}", f"2026-09-{20 + event}T11:30:00Z"),
    )


def _add_fixture(conn, fixture_id: int, event: int, team_h: int, team_a: int) -> None:
    conn.execute(
        "INSERT INTO fixtures(id, event, team_h, team_a, kickoff_time, finished, started, raw_json,"
        " updated_at) VALUES (?,?,?,?,?,0,0,'{}','2026-09-01T00:00:00Z')",
        (fixture_id, event, team_h, team_a, f"2026-09-{20 + event}T14:00:00Z"),
    )


def _run(
    conn,
    run_id: int,
    family: str,
    event: int,
    *,
    cutoff: str = CUTOFF,
    version: str | None = None,
    status: str = "complete",
    code_snapshot: str | None = CODE_SNAPSHOT,
    context_hash: str | None = CONTEXT_HASH,
) -> None:
    conn.execute(
        "INSERT INTO projection_runs(id, model_family, model_version, generated_at, planning_event,"
        " data_cutoff, status, source_snapshot_sha256, planning_context_hash)"
        " VALUES (?,?,?,'2026-09-19T11:01:00Z',?,?,?,?,?)",
        (
            run_id,
            family,
            version if version is not None else VERSIONS[family],
            event,
            cutoff,
            status,
            code_snapshot,
            context_hash,
        ),
    )


def _xpts_row(
    conn,
    run_id: int,
    fixture_id: int,
    event: int,
    *,
    minutes_run_id: int,
    team_run_id: int,
    rate_run_id: int,
    player_id: int = 1,
) -> None:
    conn.execute(
        "INSERT INTO player_fixture_xpts_projections(projection_run_id, player_id, fixture_id, event,"
        " team_id, opponent_id, position, minutes_run_id, team_run_id, rate_run_id, payload_json,"
        " model_version, scoring_rules_version, generated_at)"
        " VALUES (?,?,?,?,1,2,'MID',?,?,?,'{}','xpts_v1','v1','2026-09-19T11:01:00Z')",
        (run_id, player_id, fixture_id, event, minutes_run_id, team_run_id, rate_run_id),
    )


def _mc_row(
    conn,
    run_id: int,
    fixture_id: int,
    event: int,
    *,
    xpts_run_id: int,
    minutes_run_id: int,
    team_run_id: int,
    rate_run_id: int,
    player_id: int = 1,
) -> None:
    conn.execute(
        "INSERT INTO monte_carlo_distributions(projection_run_id, player_id, fixture_id, event,"
        " team_id, opponent_id, position, xpts_run_id, minutes_run_id, team_run_id, rate_run_id,"
        " payload_json, model_version, generated_at)"
        " VALUES (?,?,?,?,1,2,'MID',?,?,?,?,'{}','mc_v1','2026-09-19T11:01:00Z')",
        (run_id, player_id, fixture_id, event, xpts_run_id, minutes_run_id, team_run_id, rate_run_id),
    )


def _world(events=HORIZON, *, fixtures_per_event=1, blanks=()):
    """A coherent five-family world for every event.  Returns (conn, runs_by_event).

    ``blanks`` names events that are a BLANK Gameweek: they have their five runs but
    no fixture and therefore no player x fixture row, which is what PE-4 calls a
    valid zero-fixture world rather than missing data.
    """

    conn = connect_database(":memory:")
    _base_world(conn)
    runs_by_event: dict[int, dict[str, int]] = {}
    next_run = 100
    next_fixture = 1000
    with conn:
        for event in events:
            _add_event(conn, event)
            fixture_ids = []
            if int(event) not in {int(blank) for blank in blanks}:
                for _ in range(fixtures_per_event):
                    _add_fixture(conn, next_fixture, event, 1, 2)
                    fixture_ids.append(next_fixture)
                    next_fixture += 1
            ids = {
                "minutes_v1": next_run,
                "team_strength_v1": next_run + 1,
                "player_rates_v1": next_run + 2,
                "xpts_v1": next_run + 3,
                "monte_carlo_v1": next_run + 4,
            }
            next_run += 5
            for family, run_id in ids.items():
                _run(conn, run_id, family, event)
            for fixture_id in fixture_ids:
                _xpts_row(
                    conn,
                    ids["xpts_v1"],
                    fixture_id,
                    event,
                    minutes_run_id=ids["minutes_v1"],
                    team_run_id=ids["team_strength_v1"],
                    rate_run_id=ids["player_rates_v1"],
                )
                _mc_row(
                    conn,
                    ids["monte_carlo_v1"],
                    fixture_id,
                    event,
                    xpts_run_id=ids["xpts_v1"],
                    minutes_run_id=ids["minutes_v1"],
                    team_run_id=ids["team_strength_v1"],
                    rate_run_id=ids["player_rates_v1"],
                )
            runs_by_event[int(event)] = ids
    return conn, runs_by_event


def _artifact(conn, runs_by_event, *, cutoff=CUTOFF, events=HORIZON, code_snapshot=CODE_SNAPSHOT, **over):
    """A certification artifact of the shape the certifier mints.

    The code snapshot is recorded on the artifact AND on every bundle payload, as
    ``scripts/certify_gw5_gw8.py`` does, so the identity the ``certify_*`` entry
    points recompute from a payload is the identity the artifact recorded.

    The AUTHORIZATION fields are the ones the canonical loader's contract requires --
    the schema, the causal status, the dependency coherence, the authorisation flag,
    the completeness audit and the wiring identity.  They are not decoration: an
    artifact-shaped mapping that carries self-consistent bundles but not these fields
    is not an authorisation, and a predictive load re-runs the contract over it and
    refuses it with ``CERTIFICATION_ARTIFACT_UNVALIDATED``.  A fixture that omitted
    them would be exercising a mapping the engine never accepts, so the adversarial
    cases below STRIP them from this one to prove the refusal.
    """

    bundles = {}
    identity = {}
    for event in events:
        bundle = cb.certified_bundle_from_explicit_ids(
            conn, event=int(event), cutoff=cutoff, runs=runs_by_event[int(event)],
            required_versions=VERSIONS, data_snapshot_sha256=DATA_SNAPSHOT,
            code_snapshot_sha256=code_snapshot,
        )
        bundles[str(event)] = bundle.as_dict()
        identity[str(event)] = bundle.bundle_identity()
    artifact = {
        "schema": fg.CERTIFICATION_ARTIFACT_SCHEMA,
        "events": [int(event) for event in events],
        "planning_cutoff": cutoff,
        "data_snapshot_sha256": DATA_SNAPSHOT,
        "code_snapshot_sha256": code_snapshot,
        "required_model_versions": dict(VERSIONS),
        "certified_bundles": bundles,
        "certified_bundle_identity": identity,
        "certification_wiring": fg.certification_wiring_identity(),
        "temporal_status": "CAUSAL",
        "dependency_validation": "COHERENT",
        "history_completeness": {"schema": "fixture", "complete": True, "reasons": []},
        "route_search_executed": False,
        "transfer_execution_performed": False,
        "decision_search_permitted": True,
        "decision_search_permitted_reasons": [],
        "four_gw_certification_identity": "sha256:" + "c" * 64,
    }
    artifact.update(over)
    return artifact


#: The authorization fields the canonical loader's contract requires, and whose
#: ABSENCE must refuse a load rather than degrade it.
AUTHORIZATION_FIELDS = (
    "schema",
    "temporal_status",
    "dependency_validation",
    "history_completeness",
    "certification_wiring",
    "decision_search_permitted",
)


def _unauthorised(artifact, *, drop=AUTHORIZATION_FIELDS):
    """The SAME self-consistent artifact, with its authorization fields removed.

    The bundles, their run ids and their identities are untouched -- so every internal
    consistency check the artifact could make about itself still passes.  What it no
    longer carries is the authorisation to load predictive data.
    """

    stripped = json.loads(json.dumps(artifact))
    for field in drop:
        stripped.pop(field, None)
    return stripped


# ---------------------------------------------------------------------------
# 1-11. Structural gates
# ---------------------------------------------------------------------------


def test_1_a_coherent_complete_bundle_is_accepted():
    conn, runs = _world()
    try:
        bundle = cb.certify_event_bundle(
            conn, event=5, cutoff=CUTOFF, runs=runs[5], required_versions=VERSIONS,
            data_snapshot_sha256=DATA_SNAPSHOT,
        )
        assert bundle.structural_state == cb.STATE_CERTIFIED_COHERENT
        assert bundle.state in cb.NON_FAILURE_BUNDLE_STATES
        assert bundle.bundle_identity.startswith("sha256:")
        assert bundle.runs == runs[5]
        assert bundle.model_versions == {family: VERSIONS[family] for family in runs[5]}
    finally:
        conn.close()


def test_2_a_missing_family_is_refused():
    conn, runs = _world()
    try:
        incomplete = {family: rid for family, rid in runs[5].items() if family != "player_rates_v1"}
        with pytest.raises(cb.BundleIncoherent) as caught:
            cb.certified_bundle_from_explicit_ids(
                conn, event=5, cutoff=CUTOFF, runs=incomplete, required_versions=VERSIONS
            )
        assert any("missing family player_rates_v1" in reason for reason in caught.value.reasons)
    finally:
        conn.close()


def test_3_a_wrong_planning_event_is_refused():
    conn, runs = _world()
    try:
        with pytest.raises(cb.BundleIncoherent) as caught:
            cb.certified_bundle_from_explicit_ids(
                conn, event=99, cutoff=CUTOFF, runs=runs[5], required_versions=VERSIONS
            )
        assert any("planning_event" in reason and "expected 99" in reason for reason in caught.value.reasons)
    finally:
        conn.close()


def test_4_a_wrong_cutoff_is_refused():
    conn, runs = _world()
    try:
        with pytest.raises(cb.BundleIncoherent) as caught:
            cb.certified_bundle_from_explicit_ids(
                conn, event=5, cutoff=OTHER_CUTOFF, runs=runs[5], required_versions=VERSIONS
            )
        assert any("data_cutoff" in reason for reason in caught.value.reasons)
    finally:
        conn.close()


def test_5_an_incomplete_run_is_refused():
    conn, runs = _world()
    try:
        with conn:
            _run(conn, 900, "minutes_v1", 5, status="running")
        runs[5] = dict(runs[5], minutes_v1=900)
        with pytest.raises(cb.BundleIncoherent) as caught:
            cb.certified_bundle_from_explicit_ids(
                conn, event=5, cutoff=CUTOFF, runs=runs[5], required_versions=VERSIONS
            )
        assert any("status is 'running', not complete" in reason for reason in caught.value.reasons)
    finally:
        conn.close()


def test_6_xpts_referencing_a_different_upstream_run_is_refused():
    conn, runs = _world()
    try:
        # A same-cutoff, same-version xPts rerun wired to a DIFFERENT minutes run.
        with conn:
            _run(conn, 901, "minutes_v1", 5)
            _run(conn, 902, "xpts_v1", 5)
            _xpts_row(
                conn, 902, 1000, 5, minutes_run_id=901,
                team_run_id=runs[5]["team_strength_v1"], rate_run_id=runs[5]["player_rates_v1"],
            )
        with pytest.raises(cb.BundleIncoherent) as caught:
            cb.certified_bundle_from_explicit_ids(
                conn, event=5, cutoff=CUTOFF, runs=dict(runs[5], xpts_v1=902),
                required_versions=VERSIONS,
            )
        assert any(
            "references minutes_v1 run 901" in reason for reason in caught.value.reasons
        )
    finally:
        conn.close()


def test_7_monte_carlo_referencing_a_different_xpts_or_upstream_run_is_refused():
    conn, runs = _world()
    try:
        with conn:
            _run(conn, 903, "xpts_v1", 5)
            _xpts_row(
                conn, 903, 1000, 5, minutes_run_id=runs[5]["minutes_v1"],
                team_run_id=runs[5]["team_strength_v1"], rate_run_id=runs[5]["player_rates_v1"],
                player_id=2,
            )
            # A distinct MC run for the event, simulated from xPts run 903 rather
            # than from the xPts run the bundle declares.
            _run(conn, 907, "monte_carlo_v1", 5)
            _mc_row(
                conn, 907, 1000, 5, xpts_run_id=903,
                minutes_run_id=runs[5]["minutes_v1"], team_run_id=runs[5]["team_strength_v1"],
                rate_run_id=runs[5]["player_rates_v1"],
            )
        with pytest.raises(cb.BundleIncoherent) as caught:
            cb.certified_bundle_from_explicit_ids(
                conn, event=5, cutoff=CUTOFF,
                runs=dict(runs[5], monte_carlo_v1=907), required_versions=VERSIONS,
            )
        assert any(
            "references xpts_v1 run 903, but the bundle declares" in reason
            for reason in caught.value.reasons
        )

        # A different upstream Minutes run is refused the same way.
        with conn:
            _run(conn, 908, "minutes_v1", 5)
            _mc_row(
                conn, runs[5]["monte_carlo_v1"], 1001, 5, xpts_run_id=runs[5]["xpts_v1"],
                minutes_run_id=908, team_run_id=runs[5]["team_strength_v1"],
                rate_run_id=runs[5]["player_rates_v1"], player_id=2,
            )
        with pytest.raises(cb.BundleIncoherent) as caught:
            cb.certified_bundle_from_explicit_ids(
                conn, event=5, cutoff=CUTOFF, runs=runs[5], required_versions=VERSIONS,
            )
        assert any(
            "was built from 2 different upstream combination(s)" in reason
            or "references minutes_v1 run 908" in reason
            for reason in caught.value.reasons
        )
    finally:
        conn.close()


def test_8_an_unsupported_model_version_is_refused_once_required_versions_is_supplied():
    conn, runs = _world()
    try:
        with conn:
            _run(conn, 904, "minutes_v1", 5, version="minutes_v1.0.1")
        runs[5] = dict(runs[5], minutes_v1=904)
        with pytest.raises(cb.BundleIncoherent) as caught:
            cb.certified_bundle_from_explicit_ids(
                conn, event=5, cutoff=CUTOFF, runs=runs[5], required_versions=VERSIONS
            )
        assert cb.DIAG_UNSUPPORTED_MODEL_VERSION in str(caught.value)
        assert cb.bundle_state_from_reasons(caught.value.reasons) == cb.STATE_UNSUPPORTED_MODEL_VERSION
        # The token names the family, the found version and the required one.
        reason = next(r for r in caught.value.reasons if "UNSUPPORTED_MODEL_VERSION" in r)
        assert "minutes_v1" in reason and "minutes_v1.0.1" in reason and VERSIONS["minutes_v1"] in reason
    finally:
        conn.close()


def test_9_code_snapshot_incoherence_across_families_is_refused():
    conn, runs = _world()
    try:
        with conn:
            _run(conn, 905, "monte_carlo_v1", 5, code_snapshot="a-different-code-revision")
        runs[5] = dict(runs[5], monte_carlo_v1=905)
        with pytest.raises(cb.BundleIncoherent) as caught:
            cb.certified_bundle_from_explicit_ids(
                conn, event=5, cutoff=CUTOFF, runs=runs[5], required_versions=VERSIONS
            )
        assert any("code snapshot" in reason for reason in caught.value.reasons)
        assert cb.bundle_state_from_reasons(caught.value.reasons) == cb.STATE_PREDICTIVE_BUNDLE_INCOHERENT
    finally:
        conn.close()


def test_10_a_planning_context_mismatch_is_refused():
    conn, runs = _world()
    try:
        with conn:
            _run(conn, 906, "xpts_v1", 5, context_hash="a-different-context")
            _xpts_row(
                conn, 906, 1000, 5, minutes_run_id=runs[5]["minutes_v1"],
                team_run_id=runs[5]["team_strength_v1"], rate_run_id=runs[5]["player_rates_v1"],
            )
        runs[5] = dict(runs[5], xpts_v1=906)
        with pytest.raises(cb.BundleIncoherent) as caught:
            cb.certified_bundle_from_explicit_ids(
                conn, event=5, cutoff=CUTOFF, runs=runs[5], required_versions=VERSIONS
            )
        assert any("planning_context_hash differs" in reason for reason in caught.value.reasons)
    finally:
        conn.close()


def test_11_a_required_calibration_artifact_that_is_absent_is_refused():
    conn, runs = _world()
    try:
        with pytest.raises(cb.CertificationRefused) as caught:
            cb.certify_event_bundle(
                conn, event=5, cutoff=CUTOFF, runs=runs[5], required_versions=VERSIONS,
                calibration=None, require_calibration=True,
            )
        assert caught.value.token == cb.STATE_EVIDENCE_MISSING
        assert cb.STATE_EVIDENCE_MISSING in str(caught.value)
    finally:
        conn.close()


def test_11b_a_supplied_absent_data_snapshot_is_not_merely_recorded():
    """The data snapshot identity is VALIDATED where the bundle declares one."""

    conn, runs = _world()
    try:
        with pytest.raises(cb.BundleIncoherent) as caught:
            cb.certified_bundle_from_explicit_ids(
                conn, event=5, cutoff=CUTOFF, runs=runs[5], required_versions=VERSIONS,
                data_snapshot_sha256="sha256:" + "e" * 64,
                expected_data_snapshot_sha256=DATA_SNAPSHOT,
            )
        assert any("data_snapshot_sha256" in reason for reason in caught.value.reasons)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 12-15. Calibration identity, evidence insufficiency and world binding
# ---------------------------------------------------------------------------


def _calibration(*, cutoff=CUTOFF, runs_by_event=None, surfaces=None, excluded=None,
                 schema=None, evaluation_version=None, certification_identity="sha256:" + "c" * 64,
                 terminal_state=None):
    """A PE-8-shaped calibration artifact, carrying PE-8's own declared tokens."""

    from fpl_brain import calibration_evaluation as ce

    body = {
        "schema": schema if schema is not None else ce.PE8_SCHEMA_VERSION,
        "phase": "PE-8",
        "evaluation_version": (
            evaluation_version if evaluation_version is not None else ce.PE8_EVALUATION_VERSION
        ),
        "identity": {
            "evaluation_version": ce.PE8_EVALUATION_VERSION,
            "schema_version": ce.PE8_SCHEMA_VERSION,
            "certification_identity": certification_identity,
            "planning_cutoff": cutoff,
            "per_event_runs": {
                str(event): {
                    "xpts_v1": (runs or {}).get("xpts_v1"),
                    "monte_carlo_v1": (runs or {}).get("monte_carlo_v1"),
                }
                for event, runs in (runs_by_event or {}).items()
            },
        },
        "probability_calibration": {"surfaces": surfaces if surfaces is not None else []},
        "excluded_surfaces": list(excluded if excluded is not None else []),
        "limitations": ["CONTINUOUS_PROXY_TIE_LIMITATION from PE-3 remains an open, accepted limitation."],
    }
    if terminal_state is not None:
        body["terminal_state"] = dict(terminal_state)
    return body


def _surface(metric, *, status="OK", diagnosis, sample, challenger, scored=10, observations=10):
    return {
        "metric": metric,
        "status": status,
        "population": {"scored": scored},
        "diagnosis": {"status": diagnosis},
        "sample": {"observations": observations, "sample_interpretation": sample},
        "causal_challenger": {"status": challenger},
    }


def test_12_an_unknown_calibration_identity_is_refused():
    conn, runs = _world()
    try:
        unknown = _calibration(runs_by_event=runs, schema="pe8_calibration_v9.9.9")
        with pytest.raises(cb.CertificationRefused) as caught:
            cb.validate_calibration_evidence(
                calibration=unknown, event=5, cutoff=CUTOFF, runs=runs[5],
                certification_identity="sha256:" + "c" * 64,
            )
        assert caught.value.token == cb.DIAG_CALIBRATION_IDENTITY_UNKNOWN
    finally:
        conn.close()


def test_12b_an_absent_calibration_identity_is_refused_and_is_not_an_unknown_one():
    conn, runs = _world()
    try:
        no_identity = _calibration(runs_by_event=runs)
        no_identity.pop("identity")
        with pytest.raises(cb.CertificationRefused) as caught:
            cb.validate_calibration_evidence(
                calibration=no_identity, event=5, cutoff=CUTOFF, runs=runs[5],
                certification_identity="sha256:" + "c" * 64,
            )
        # Absent and unknown are DIFFERENT facts with different tokens.
        assert caught.value.token == cb.STATE_EVIDENCE_MISSING
        assert caught.value.token != cb.DIAG_CALIBRATION_IDENTITY_UNKNOWN
    finally:
        conn.close()


def test_13_not_fitted_and_no_material_defect_are_legitimate_non_failure_outcomes():
    """Distinguishable from incoherence, and neither is reported as a failure.

    PE-8's own terminal state lists a sample that is NOT
    ``SUFFICIENT_FOR_DESCRIPTIVE_REPORTING_ONLY`` as a reason to stay OPEN, so a
    descriptive-only sample IS the sample verdict PE-8 asks for: the clean
    diagnosis and the descriptive-only sample together are COHERENT, and the
    unfitted transform is reported beside them rather than treated as a defect.
    """

    from fpl_brain import calibration_evaluation as ce
    from fpl_brain import walk_forward_scoreboard as sb

    conn, runs = _world()
    try:
        calibration = _calibration(
            runs_by_event=runs,
            surfaces=[
                _surface("BRIER_P_START", diagnosis=ce.DIAGNOSIS_NO_MATERIAL_DEFECT,
                         sample=sb.SAMPLE_DESCRIPTIVE_ONLY, challenger=ce.STATUS_NOT_FITTED),
            ],
            excluded=[
                {"surface": "p_goal", "reason": "an identity mapping with no calibration question"},
            ],
        )
        readings = {reading["surface"]: reading for reading in cb.calibration_surface_states(calibration)}
        clean = readings["BRIER_P_START"]
        assert clean["state"] == cb.STATE_CERTIFIED_COHERENT
        assert clean["state"] in cb.NON_FAILURE_BUNDLE_STATES
        assert clean["not_fitted"] is True
        assert clean["diagnosis"] == ce.DIAGNOSIS_NO_MATERIAL_DEFECT
        assert clean["causal_challenger"] == ce.STATUS_NOT_FITTED
        # An excluded surface is a declared NON-QUESTION, reported rather than skipped.
        assert readings["p_goal"]["state"] == cb.STATE_CALIBRATION_NOT_APPLICABLE
        assert readings["p_goal"]["state"] in cb.NON_FAILURE_BUNDLE_STATES
        assert cb.calibration_evidence_state(calibration, list(readings.values())) == (
            cb.STATE_CERTIFIED_COHERENT
        )

        bundle = cb.certify_event_bundle(
            conn, event=5, cutoff=CUTOFF, runs=runs[5], required_versions=VERSIONS,
            data_snapshot_sha256=DATA_SNAPSHOT, calibration=calibration,
        )
        assert bundle.state == cb.STATE_CERTIFIED_COHERENT
        assert bundle.structural_state == cb.STATE_CERTIFIED_COHERENT
        # Neither legitimate outcome is coherent-bundle incoherence.
        assert bundle.state != cb.STATE_PREDICTIVE_BUNDLE_INCOHERENT
        assert bundle.state != cb.STATE_EVIDENCE_MISSING
        # NOT_FITTED on its own is a non-failure state, never a refusal.
        assert bundle.state in set(cb.NON_FAILURE_BUNDLE_STATES) | set(cb.BLOCKING_BUNDLE_STATES)
    finally:
        conn.close()


def test_13b_the_pe8_sample_interpretation_is_read_from_pe8_not_restated():
    """A descriptive-only sample must be recognised as descriptive-only.

    The sample token is PE-8's (``SUFFICIENT_FOR_DESCRIPTIVE_REPORTING_ONLY``).  A
    certification that re-spelled it would silently stop matching, and a
    descriptive-only sample would read as a clean surface -- a fail-OPEN drift.
    """

    from fpl_brain import calibration_evaluation as ce
    from fpl_brain import walk_forward_scoreboard as sb

    tokens = cb.calibration_tokens()
    assert tokens["descriptive_only"] == sb.SAMPLE_DESCRIPTIVE_ONLY
    assert tokens["miscalibrated"] == ce.DIAGNOSIS_MISCALIBRATED
    assert tokens["no_material_defect"] == ce.DIAGNOSIS_NO_MATERIAL_DEFECT
    assert tokens["insufficient"] == ce.DIAGNOSIS_INSUFFICIENT
    assert tokens["not_fitted"] == ce.STATUS_NOT_FITTED

    calibration = _calibration(
        surfaces=[
            _surface("BRIER_P_START", diagnosis=ce.DIAGNOSIS_NO_MATERIAL_DEFECT,
                     sample=sb.SAMPLE_DESCRIPTIVE_ONLY, challenger=ce.STATUS_NOT_FITTED),
        ],
    )
    readings = {reading["surface"]: reading for reading in cb.calibration_surface_states(calibration)}
    assert readings["BRIER_P_START"]["state"] == cb.STATE_CERTIFIED_COHERENT

    calibration = _calibration(
        surfaces=[
            _surface("BRIER_P_60", diagnosis=ce.DIAGNOSIS_NO_MATERIAL_DEFECT,
                     sample=sb.SAMPLE_INSUFFICIENT, challenger=ce.STATUS_NOT_FITTED),
        ],
    )
    readings = {reading["surface"]: reading for reading in cb.calibration_surface_states(calibration)}
    assert readings["BRIER_P_60"]["state"] == cb.STATE_EVIDENCE_LIMITED


def test_14_pe8_insufficient_evidence_is_distinct_from_incoherence_and_is_not_a_refusal():
    from fpl_brain import calibration_evaluation as ce
    from fpl_brain import walk_forward_scoreboard as sb

    conn, runs = _world()
    try:
        calibration = _calibration(
            runs_by_event=runs,
            surfaces=[
                _surface("BRIER_P_START", diagnosis=ce.DIAGNOSIS_INSUFFICIENT,
                         sample=sb.SAMPLE_INSUFFICIENT, challenger=ce.STATUS_NOT_FITTED,
                         scored=0, observations=0),
            ],
        )
        bundle = cb.certify_event_bundle(
            conn, event=5, cutoff=CUTOFF, runs=runs[5], required_versions=VERSIONS,
            data_snapshot_sha256=DATA_SNAPSHOT, calibration=calibration,
        )
        # Structurally valid: the bundle is NOT incoherent and NOT refused.
        assert bundle.structural_state == cb.STATE_CERTIFIED_COHERENT
        assert bundle.evidence_state == cb.STATE_EVIDENCE_LIMITED
        assert bundle.state == cb.STATE_EVIDENCE_LIMITED
        assert bundle.state in cb.NON_FAILURE_BUNDLE_STATES
        assert bundle.state not in cb.BLOCKING_BUNDLE_STATES
        assert bundle.state != cb.STATE_PREDICTIVE_BUNDLE_INCOHERENT
        assert bundle.state != cb.STATE_EVIDENCE_MISSING
    finally:
        conn.close()


def test_15_calibration_evidence_from_a_different_prediction_world_is_refused():
    conn, runs = _world()
    try:
        # The evidence scored a DIFFERENT xPts run for GW5 than the bundle certifies.
        foreign = _calibration(
            runs_by_event={5: {"xpts_v1": runs[5]["xpts_v1"] + 5000,
                               "monte_carlo_v1": runs[5]["monte_carlo_v1"]}},
        )
        with pytest.raises(cb.CertificationRefused) as caught:
            cb.validate_calibration_evidence(
                calibration=foreign, event=5, cutoff=CUTOFF, runs=runs[5],
                certification_identity="sha256:" + "c" * 64,
            )
        assert caught.value.token == cb.DIAG_CALIBRATION_WORLD_MISMATCH

        # A different cutoff is a different world too.
        other_world = _calibration(cutoff=OTHER_CUTOFF, runs_by_event=runs)
        with pytest.raises(cb.CertificationRefused) as caught:
            cb.validate_calibration_evidence(
                calibration=other_world, event=5, cutoff=CUTOFF, runs=runs[5],
                certification_identity="sha256:" + "c" * 64,
            )
        assert caught.value.token == cb.DIAG_CALIBRATION_WORLD_MISMATCH
    finally:
        conn.close()


def test_15b_calibration_evidence_certifying_another_bundle_identity_is_refused():
    conn, runs = _world()
    try:
        calibration = _calibration(
            runs_by_event=runs, certification_identity="sha256:" + "f" * 64
        )
        with pytest.raises(cb.CertificationRefused) as caught:
            cb.validate_calibration_evidence(
                calibration=calibration, event=5, cutoff=CUTOFF, runs=runs[5],
                certification_identity="sha256:" + "c" * 64,
            )
        assert caught.value.token == cb.DIAG_CALIBRATION_WORLD_MISMATCH
    finally:
        conn.close()


def test_14b_pe8s_own_open_terminal_state_is_not_upgraded_by_certification():
    """Certification must not claim more than the artifact it consulted.

    PE-8 declares ``OPEN`` when the evidence supports no more than a description.
    A clean set of surfaces with PE-8's own OPEN must therefore stay
    ``EVIDENCE_LIMITED`` -- a state, never a failure, and never a pass.
    """

    from fpl_brain import calibration_evaluation as ce
    from fpl_brain import walk_forward_scoreboard as sb

    conn, runs = _world()
    try:
        clean_surfaces = [
            _surface("BRIER_P_START", diagnosis=ce.DIAGNOSIS_NO_MATERIAL_DEFECT,
                     sample=sb.SAMPLE_DESCRIPTIVE_ONLY, challenger=ce.STATUS_NOT_FITTED),
        ]
        # With no declared terminal state, a clean surface set certifies coherent.
        open_ended = _calibration(runs_by_event=runs, surfaces=clean_surfaces)
        assert cb.calibration_evidence_state(
            open_ended, cb.calibration_surface_states(open_ended)
        ) == cb.STATE_CERTIFIED_COHERENT

        # PE-8's own OPEN caps the claim, and the cap is PE-8's, not a new threshold.
        pe8_open = _calibration(
            runs_by_event=runs, surfaces=clean_surfaces,
            terminal_state={
                "state": "OPEN",
                "reasons": ["the sample can support nothing beyond a descriptive report"],
                "promotion_performed": False,
            },
        )
        assert cb.calibration_terminal_state(pe8_open)["state"] == "OPEN"
        assert cb.calibration_evidence_state(
            pe8_open, cb.calibration_surface_states(pe8_open)
        ) == cb.STATE_EVIDENCE_LIMITED

        bundle = cb.certify_event_bundle(
            conn, event=5, cutoff=CUTOFF, runs=runs[5], required_versions=VERSIONS,
            data_snapshot_sha256=DATA_SNAPSHOT, calibration=pe8_open,
        )
        assert bundle.state == cb.STATE_EVIDENCE_LIMITED
        assert bundle.structural_state == cb.STATE_CERTIFIED_COHERENT
        assert bundle.state not in cb.BLOCKING_BUNDLE_STATES
        assert bundle.calibration["pe8_terminal_state"] == "OPEN"
        assert any(
            "PE-8 declares terminal state OPEN" in reason for reason in bundle.evidence_reasons
        )
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 16-18. DGW / blank / horizon grain
# ---------------------------------------------------------------------------


def test_16_dgw_fixture_grain_coherence_is_preserved_through_certification():
    """A double gameweek is certified as one unit at fixture grain.

    Both of the event's fixtures are read, and a run whose rows disagree about
    which upstream run they were built from is refused -- a fixture is certified
    or refused as a unit, never on whichever row is read first.
    """

    conn, runs = _world(fixtures_per_event=2)
    try:
        # Fixture-grain evidence is intact: one row per player x fixture, both
        # belonging to the SAME certified runs (no collapse to player x event).
        rows = conn.execute(
            "SELECT fixture_id FROM player_fixture_xpts_projections WHERE projection_run_id=?"
            " ORDER BY fixture_id",
            (runs[5]["xpts_v1"],),
        ).fetchall()
        assert [int(row[0]) for row in rows] == [1000, 1001], (
            "a DGW is two player x fixture rows, not one player x event row"
        )

        bundle = cb.certify_event_bundle(
            conn, event=5, cutoff=CUTOFF, runs=runs[5], required_versions=VERSIONS,
            data_snapshot_sha256=DATA_SNAPSHOT,
        )
        assert bundle.state in cb.NON_FAILURE_BUNDLE_STATES

        # A second PLAYER's row for one of the two fixtures is wired to a different
        # Minutes run: the run now disagrees with itself about its upstream, and the
        # whole event is refused as a unit.  (xPts rows are append-only -- the
        # immutability trigger refuses an UPDATE -- so a disagreeing row can only be
        # added, which is exactly how a mis-wired rerun would arrive.)
        with conn:
            _run(conn, 910, "minutes_v1", 5)
            _xpts_row(
                conn, runs[5]["xpts_v1"], 1001, 5, minutes_run_id=910,
                team_run_id=runs[5]["team_strength_v1"], rate_run_id=runs[5]["player_rates_v1"],
                player_id=2,
            )
        with pytest.raises(cb.BundleIncoherent) as caught:
            cb.certified_bundle_from_explicit_ids(
                conn, event=5, cutoff=CUTOFF, runs=runs[5], required_versions=VERSIONS
            )
        assert any("different upstream combination" in reason for reason in caught.value.reasons)

        # The fixture-grain rows are append-only, so a historical certified identity
        # cannot be rewritten by an edit to the evidence.
        with pytest.raises(sqlite3.IntegrityError):
            with conn:
                conn.execute(
                    "UPDATE player_fixture_xpts_projections SET minutes_run_id=999"
                    " WHERE projection_run_id=?",
                    (runs[5]["xpts_v1"],),
                )
    finally:
        conn.close()


def test_17_a_blank_event_is_a_valid_zero_fixture_world_and_is_not_missing_data():
    """PE-4 is preserved: a blank is not absent data and is not fabricated."""

    conn, runs = _world(blanks={6})
    try:
        # GW6 is a blank: no fixture, so no row in any family's table.
        assert conn.execute(
            "SELECT COUNT(*) FROM player_fixture_xpts_projections WHERE projection_run_id=?",
            (runs[6]["xpts_v1"],),
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM fixtures WHERE event=6"
        ).fetchone()[0] == 0
        blank = cb.certify_event_bundle(
            conn, event=6, cutoff=CUTOFF, runs=runs[6], required_versions=VERSIONS,
            data_snapshot_sha256=DATA_SNAPSHOT,
        )
        # A blank event's runs still form a coherent bundle: zero fixtures is a
        # valid zero-fixture world, not a projection that is absent.
        assert blank.structural_state == cb.STATE_CERTIFIED_COHERENT
        assert blank.state not in cb.BLOCKING_BUNDLE_STATES

        # "no fixture" and "fixture with no projection row" stay different facts.
        from fpl_brain import candidate_universe as cu

        blank_feature = cu._event_feature(
            player_id=1, club_id=1, event=6, events_fixtures={}, xpts_rows={}, minutes_rows={}
        )
        assert blank_feature.status == cu.NO_FIXTURE
        assert blank_feature.fixture_count == 0
        # A blank is NOT reported as missing predictive support.
        assert cu.unresolved_predictive_support_rows([{"player_id": 1, "events": [blank_feature]}]) == []
        missing_feature = cu._event_feature(
            player_id=1, club_id=1, event=6, events_fixtures={(6, 1): [1000]},
            xpts_rows={}, minutes_rows={},
        )
        assert missing_feature.status == cu.MISSING_PROJECTION
        assert missing_feature.status != cu.NO_FIXTURE
    finally:
        conn.close()


def test_18_one_bad_event_makes_the_four_event_normal_transfer_horizon_incomplete():
    conn, runs = _world()
    try:
        artifact = _artifact(conn, runs)
        # Every event certified: the horizon is complete.
        support = fg.event_support_from_certification(
            conn, artifact, events=HORIZON, cutoff=CUTOFF
        )
        complete = fg.evaluate_horizon(
            planning_event=5, support_by_event=support, cutoff=CUTOFF, last_event=38
        )
        assert complete["status"] == fg.DECISION_HORIZON_COMPLETE

        # Break GW7 only: one bad event blocks the whole horizon.
        with conn:
            _run(conn, 911, "minutes_v1", 7, version="minutes_v1.0.0")
        artifact["certified_bundles"]["7"]["runs"]["minutes_v1"] = 911
        with pytest.raises(cb.BundleIncoherent) as caught:
            fg.event_support_from_certification(conn, artifact, events=HORIZON, cutoff=CUTOFF)
        assert "GW7" in str(caught.value)
        assert cb.DIAG_UNSUPPORTED_MODEL_VERSION in str(caught.value)

        # A horizon assembled from a different event set is refused, not padded.
        with pytest.raises(fg.DecisionCertificationRequired):
            fg.event_support_from_certification(conn, artifact, events=(5, 6, 7), cutoff=CUTOFF)
    finally:
        conn.close()


def test_18c_the_horizon_is_anchored_to_the_certified_set_not_the_callers_order():
    """An equal-but-reordered horizon certifies the SAME horizon.

    ``canonical_event_horizon`` deliberately canonicalises order, so a reordered
    request is authorised.  The horizon must therefore be built from the certified
    set itself: anchoring it to the caller's first element would silently report a
    different horizon -- and, near the season end, could claim a shorter carve-out
    over the wrong events.
    """

    conn, runs = _world()
    try:
        artifact = _artifact(conn, runs)
        in_order = cb.certify_decision_horizon(
            conn, certification=artifact, events=HORIZON, cutoff=CUTOFF,
            required_versions=VERSIONS, last_event=38,
        )
        reordered = cb.certify_decision_horizon(
            conn, certification=artifact, events=(7, 5, 8, 6), cutoff=CUTOFF,
            required_versions=VERSIONS, last_event=38,
        )
        assert reordered["horizon_state"] == in_order["horizon_state"] == fg.DECISION_HORIZON_COMPLETE
        assert reordered["horizon"]["planning_event"] == in_order["horizon"]["planning_event"] == 5
        assert reordered["horizon"]["decision_events"] == in_order["horizon"]["decision_events"] == list(
            HORIZON
        )
        assert sorted(int(event) for event in reordered["horizon"]["events"]) == list(HORIZON)
        assert reordered["horizon"]["blocked_events"] == []
        assert reordered["per_event_bundle_identity"] == in_order["per_event_bundle_identity"]
        assert cb.certification_result_identity(reordered) == cb.certification_result_identity(in_order)

        # A season-end horizon is anchored the same way, so the carve-out describes
        # the events that were actually certified.
        season_conn, season_runs = _world(events=(36, 37, 38))
        try:
            short_artifact = _artifact(season_conn, season_runs, events=(36, 37, 38))
            short = cb.certify_decision_horizon(
                season_conn, certification=short_artifact, events=(37, 36, 38), cutoff=CUTOFF,
                required_versions=VERSIONS, last_event=38,
            )
            assert short["horizon_state"] == fg.SEASON_END_SHORT_HORIZON
            assert short["horizon"]["planning_event"] == 36
            assert short["horizon"]["effective_horizon_length"] == 3
            assert short["horizon"]["decision_events"] == [36, 37, 38]
        finally:
            season_conn.close()
    finally:
        conn.close()


def test_18b_the_season_end_short_horizon_carve_out_is_unchanged():
    conn, runs = _world(events=(36, 37, 38))
    try:
        artifact = _artifact(conn, runs, events=(36, 37, 38))
        support = fg.event_support_from_certification(
            conn, artifact, events=(36, 37, 38), cutoff=CUTOFF
        )
        horizon = fg.evaluate_horizon(
            planning_event=36, support_by_event=support, cutoff=CUTOFF, last_event=38
        )
        assert horizon["status"] == fg.SEASON_END_SHORT_HORIZON
        assert horizon["effective_horizon_length"] == 3
        assert fg.transfer_recommendation_allowed(horizon) is True
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 19-24. Downstream consumers, persistence and invariance
# ---------------------------------------------------------------------------


def test_19_downstream_code_cannot_rediscover_a_different_latest_run():
    """A hand-assembled 'newest run per family' bundle is refused, not simulated."""

    conn, runs = _world()
    try:
        # A newer same-cutoff rerun of xPts exists, wired to a different minutes run.
        with conn:
            _run(conn, 912, "minutes_v1", 5)
            _run(conn, 913, "xpts_v1", 5)
            _xpts_row(
                conn, 913, 1000, 5, minutes_run_id=912,
                team_run_id=runs[5]["team_strength_v1"], rate_run_id=runs[5]["player_rates_v1"],
            )
        artifact = _artifact(conn, runs)

        # The certified generation keeps the ORIGINAL xPts run even though 913 is newer.
        support = fg.event_support_from_certification(
            conn, artifact, events=HORIZON, cutoff=CUTOFF
        )
        assert support[5]["matched_runs"]["xpts_v1"] == runs[5]["xpts_v1"]

        # A bare, provenance-free bundle cannot reach a predictive load.
        bare = rc.EventBundle(
            event=5, minutes_run_id=runs[5]["minutes_v1"], team_run_id=runs[5]["team_strength_v1"],
            rate_run_id=runs[5]["player_rates_v1"], xpts_run_id=runs[5]["xpts_v1"],
            mc_run_id=runs[5]["monte_carlo_v1"], planning_cutoff=CUTOFF,
        )
        with pytest.raises(cb.CertificationRefused) as caught:
            ro.build_event_worlds(conn, {5: bare}, 5, [1], _optimizer_config())
        # With no artifact there is no authorisation AT ALL, which is a different
        # fact from a bundle that cannot declare its provenance.
        assert caught.value.token == cb.DIAG_CERTIFICATION_ARTIFACT_ABSENT

        # Nor can one that declares an identity which does not bind its own run ids.
        lying = rc.certified_event_bundle(
            event=5, runs=runs[5], cutoff=CUTOFF, model_versions=VERSIONS,
            code_snapshot_sha256=CODE_SNAPSHOT, data_snapshot_sha256=DATA_SNAPSHOT,
            planning_context_hash=CONTEXT_HASH,
        )
        object.__setattr__(lying, "xpts_run_id", 913)
        with pytest.raises(cb.CertificationRefused) as caught:
            ro.build_event_worlds(conn, {5: lying}, 5, [1], _optimizer_config(),
                                  certification=artifact)
        assert caught.value.token == cb.STATE_PREDICTIVE_BUNDLE_INCOHERENT

        # A bundle whose declared version disagrees with the run's own row is refused.
        wrong_version = rc.certified_event_bundle(
            event=5, runs=runs[5], cutoff=CUTOFF,
            model_versions={**VERSIONS, "xpts_v1": "xpts_v1.0.0"},
            code_snapshot_sha256=CODE_SNAPSHOT, data_snapshot_sha256=DATA_SNAPSHOT,
            planning_context_hash=CONTEXT_HASH,
        )
        with pytest.raises(cb.CertificationRefused) as caught:
            ro.build_event_worlds(conn, {5: wrong_version}, 5, [1], _optimizer_config(),
                                  certification=artifact)
        assert caught.value.token == cb.STATE_UNSUPPORTED_MODEL_VERSION

        # An artifact that does not record this event is a MISSING record, which is
        # a different fact from an artifact that contradicts it.
        certified_elsewhere = _artifact(conn, runs, events=(6,))
        with pytest.raises(cb.CertificationRefused) as caught:
            cb.assert_event_bundle_certified(
                conn,
                rc.certified_event_bundle(
                    event=5, runs=runs[5], cutoff=CUTOFF, model_versions=VERSIONS,
                    code_snapshot_sha256=CODE_SNAPSHOT, data_snapshot_sha256=DATA_SNAPSHOT,
                    planning_context_hash=CONTEXT_HASH,
                ),
                event=5,
                certification=certified_elsewhere,
            )
        assert caught.value.token == cb.STATE_EVIDENCE_MISSING

        # An artifact that records THIS event as a DIFFERENT bundle contradicts the
        # request, and the two facts carry different tokens.
        substituted = json.loads(json.dumps(_artifact(conn, runs)))
        substituted["certified_bundles"]["5"]["cutoff"] = OTHER_CUTOFF
        with pytest.raises(cb.CertificationArtifactContradictory):
            cb.assert_event_bundle_certified(
                conn,
                rc.certified_event_bundle(
                    event=5, runs=runs[5], cutoff=CUTOFF, model_versions=VERSIONS,
                    code_snapshot_sha256=CODE_SNAPSHOT, data_snapshot_sha256=DATA_SNAPSHOT,
                    planning_context_hash=CONTEXT_HASH,
                ),
                event=5,
                certification=substituted,
            )
    finally:
        conn.close()


def test_19b_the_certification_artifact_is_exactly_one_per_decision():
    conn, runs = _world()
    try:
        artifact = _artifact(conn, runs)
        # Absent and contradictory are different tokens.
        with pytest.raises(cb.CertificationArtifactAbsent) as caught:
            cb.require_certification_artifact(None)
        assert caught.value.token == cb.DIAG_CERTIFICATION_ARTIFACT_ABSENT
        with pytest.raises(cb.CertificationArtifactContradictory) as caught:
            cb.require_certification_artifact([artifact, artifact])
        assert caught.value.token == cb.DIAG_CERTIFICATION_ARTIFACT_CONTRADICTORY
        with pytest.raises(fg.DecisionCertificationRequired) as caught:
            fg.assert_certification_artifact_describes_decision(None, events=HORIZON, cutoff=CUTOFF)
        assert cb.DIAG_CERTIFICATION_ARTIFACT_ABSENT in str(caught.value)

        # An artifact carrying no data snapshot identity contradicts the decision.
        no_snapshot = dict(artifact)
        no_snapshot.pop("data_snapshot_sha256")
        with pytest.raises(fg.DecisionCertificationRequired) as caught:
            fg.assert_certification_artifact_describes_decision(
                no_snapshot, events=HORIZON, cutoff=CUTOFF
            )
        assert cb.DIAG_CERTIFICATION_ARTIFACT_CONTRADICTORY in str(caught.value)

        # An artifact that contradicts the decision cutoff is refused.
        with pytest.raises(fg.DecisionCertificationRequired):
            fg.assert_certification_artifact_describes_decision(
                artifact, events=HORIZON, cutoff=OTHER_CUTOFF
            )
    finally:
        conn.close()


def test_20_a_missing_projection_never_becomes_zero():
    """The label exists AND the zero is reported as unresolved, never as a prediction."""

    from fpl_brain import candidate_universe as cu

    feature = cu._event_feature(
        player_id=1, club_id=1, event=5, events_fixtures={(5, 1): [1000]},
        xpts_rows={}, minutes_rows={},
    )
    assert feature.status == cu.MISSING_PROJECTION
    assert feature.fixture_count == 1

    row = {"player_id": 1, "events": [feature]}
    unresolved = cu.unresolved_predictive_support_rows([row])
    assert len(unresolved) == 1
    assert unresolved[0]["reason"] == cu.CANDIDATE_PREDICTIVE_SUPPORT_MISSING
    assert unresolved[0]["fixture_count"] == 1
    # An explicit zero and a missing projection are different statuses.
    assert feature.status != cu.NO_FIXTURE


def test_20b_a_missing_input_run_is_refused_rather_than_defaulted():
    """A missing run row must not parse as a legacy version and silently reconstruct."""

    from fpl_brain import analytics, monte_carlo

    conn, runs = _world()
    try:
        assert analytics.get_projection_run(conn, 999999) is None
        with pytest.raises(monte_carlo.InputRunAbsent) as caught:
            monte_carlo.load_fixture_inputs(
                conn, event=5, xpts_run_id=runs[5]["xpts_v1"], minutes_run_id=999999,
                team_run_id=runs[5]["team_strength_v1"],
            )
        assert "INPUT_RUN_ABSENT" in str(caught.value)
    finally:
        conn.close()


def test_21_the_persisted_certification_identity_round_trips_exactly():
    conn, runs = _world()
    try:
        artifact = _artifact(conn, runs)
        result = cb.certify_decision_horizon(
            conn, certification=artifact, events=HORIZON, cutoff=CUTOFF,
            required_versions=VERSIONS, last_event=38,
        )
        identity = cb.certification_result_identity(result)
        # Recomputing from the STORED BYTES yields the same value.
        stored = json.loads(json.dumps(result, sort_keys=True, default=str))
        assert cb.certification_result_identity(stored) == identity
        # Recomputing after a byte-level round trip through text also matches.
        assert cb.certification_result_identity(
            json.loads(json.dumps(stored, sort_keys=True, default=str))
        ) == identity
        # The result carries the certification identity, the per-event bundle
        # identities, the run ids, the versions, the cutoff, the code and data
        # snapshots and the per-bundle states.
        assert result["certification_artifact_identity"] == artifact["four_gw_certification_identity"]
        assert set(result["per_event_bundle_identity"]) == {"5", "6", "7", "8"}
        for event in HORIZON:
            record = result["per_event"][str(event)]
            assert record["runs"] == {f: int(r) for f, r in runs[event].items()}
            assert record["model_versions"] == {f: VERSIONS[f] for f in runs[event]}
            assert record["cutoff"] == CUTOFF
            assert record["code_snapshot_sha256"] == CODE_SNAPSHOT
            assert record["data_snapshot_sha256"] == DATA_SNAPSHOT
            assert record["state"] in cb.PER_BUNDLE_STATES
        assert result["planning_cutoff"] == CUTOFF
        assert result["required_model_versions"] == VERSIONS
        # The bundle identity is the ONE shared algorithm's, not a second one.
        for event in HORIZON:
            assert result["per_event_bundle_identity"][str(event)] == cb.canonical_bundle_identity(
                artifact["certified_bundles"][str(event)]
            )
    finally:
        conn.close()


def test_22_mutating_current_tables_cannot_rewrite_a_historical_certified_identity():
    conn, runs = _world()
    try:
        artifact = _artifact(conn, runs)
        result = cb.certify_decision_horizon(
            conn, certification=artifact, events=HORIZON, cutoff=CUTOFF,
            required_versions=VERSIONS, last_event=38,
        )
        identity = cb.certification_result_identity(result)
        per_event = dict(result["per_event_bundle_identity"])

        with conn:
            conn.execute("UPDATE players SET team_id=2, web_name='Renamed' WHERE id=1")
            conn.execute("UPDATE fixtures SET team_h=2, team_a=1, event=7 WHERE id=1000")
            conn.execute("UPDATE events SET finished=1, name='GWX' WHERE id=5")
            conn.execute(
                "INSERT INTO player_gameweeks(player_id, event, fixture_id, opponent_team, was_home,"
                " minutes, starts, total_points, source, raw_json, updated_at)"
                " VALUES (1,5,1000,2,1,90,1,9,'test','{}','2026-09-01T00:00:00Z')"
            )

        # Nothing in the identity is read from the mutable current-state tables.
        assert cb.certification_result_identity(result) == identity
        assert result["per_event_bundle_identity"] == per_event
        for event in HORIZON:
            assert result["per_event"][str(event)]["runs"] == {
                f: int(r) for f, r in runs[event].items()
            }
    finally:
        conn.close()


def test_23_continuous_proxy_tie_limitation_remains_disclosed_after_certification():
    from fpl_brain import calibration_evaluation as ce
    from fpl_brain import walk_forward_scoreboard as sb

    conn, runs = _world()
    try:
        calibration = _calibration(
            runs_by_event=runs,
            surfaces=[
                _surface("BRIER_P_START", diagnosis=ce.DIAGNOSIS_NO_MATERIAL_DEFECT,
                         sample=sb.SAMPLE_DESCRIPTIVE_ONLY, challenger=ce.STATUS_NOT_FITTED),
            ],
        )
        # Disclosed whether or not an artifact was consulted.
        bare_block = cb.disclosure_block()
        assert cb.CONTINUOUS_PROXY_TIE_LIMITATION in bare_block["carried"]
        assert cb.CONTINUOUS_PROXY_TIE_LIMITATION in bare_block["unresolved"]
        assert bare_block["resolution"] == "NOT_ATTEMPTED"

        artifact = _artifact(conn, runs)
        result = cb.certify_decision_horizon(
            conn, certification=artifact, events=HORIZON, cutoff=CUTOFF,
            required_versions=VERSIONS, calibration=calibration, last_event=38,
        )
        assert cb.CONTINUOUS_PROXY_TIE_LIMITATION in result["disclosure"]["carried"]
        assert cb.CONTINUOUS_PROXY_TIE_LIMITATION in result["disclosure"]["unresolved"]
        assert cb.CONTINUOUS_PROXY_TIE_LIMITATION in result["flags"]
        # PE-9 promotes nothing and wires nothing.
        assert result["no_promotion"] is True
        assert "NOT PERFORMED" in result["promotion"]
        assert "NONE" in result["production_wiring"]
    finally:
        conn.close()


def test_24_chip_transfer_scoring_and_rng_behaviour_is_unchanged():
    """Certification integration alone changes no predictive or decision quantity."""

    from fpl_brain import monte_carlo

    conn, runs = _world()
    try:
        # The RNG seed namespace and draw ordering are untouched: same inputs and
        # same config reproduce the identical matrix with and without certification.
        artifact = _artifact(conn, runs)
        first = ro.build_event_worlds(
            conn,
            {5: rc.certified_event_bundle(
                event=5, runs=runs[5], cutoff=CUTOFF, model_versions=VERSIONS,
                code_snapshot_sha256=CODE_SNAPSHOT, data_snapshot_sha256=DATA_SNAPSHOT,
                planning_context_hash=CONTEXT_HASH,
                simulations=32,
            )},
            5, [1, 2], _optimizer_config(search_draws=32), certification=artifact,
        )[0]
        second = ro.build_event_worlds(
            conn,
            {5: rc.certified_event_bundle(
                event=5, runs=runs[5], cutoff=CUTOFF, model_versions=VERSIONS,
                code_snapshot_sha256=CODE_SNAPSHOT, data_snapshot_sha256=DATA_SNAPSHOT,
                planning_context_hash=CONTEXT_HASH,
                simulations=32,
            )},
            5, [1, 2], _optimizer_config(search_draws=32), certification=artifact,
        )[0]
        assert first["core"] == second["core"]
        assert first["minutes"] == second["minutes"]

        # The Monte Carlo seed namespace and draw ordering are unchanged by the
        # certification boundary: the simulation itself takes no certification input.
        import inspect

        parameters = set(inspect.signature(monte_carlo.simulate).parameters)
        assert not {name for name in parameters if "certif" in name or "bundle" in name}

        # Chip and transfer rules are not touched by this module.
        from fpl_brain import season_rules

        assert not [
            name for name in dir(season_rules) if "certif" in name.lower() or "bundle" in name.lower()
        ]
    finally:
        conn.close()


def test_24b_declared_versions_are_read_from_the_frozen_family_modules():
    from fpl_brain import minutes_model, monte_carlo, player_rates, team_model, xpts

    assert cb.declared_required_versions() == {
        "minutes_v1": str(minutes_model.MINUTES_MODEL_VERSION),
        "team_strength_v1": str(team_model.TEAM_MODEL_VERSION),
        "player_rates_v1": str(player_rates.PLAYER_RATE_MODEL_VERSION),
        "xpts_v1": str(xpts.XPTS_MODEL_VERSION),
        "monte_carlo_v1": str(monte_carlo.MONTE_CARLO_MODEL_VERSION),
    }


# ---------------------------------------------------------------------------
# Phase terminal state and the disclosed bypass register
# ---------------------------------------------------------------------------


def test_phase_state_and_the_declared_bypass_register():
    conn, runs = _world()
    try:
        artifact = _artifact(conn, runs)
        result = cb.certify_decision_horizon(
            conn, certification=artifact, events=HORIZON, cutoff=CUTOFF,
            required_versions=VERSIONS, last_event=38,
        )
        assert result["horizon_state"] == fg.DECISION_HORIZON_COMPLETE
        assert result["phase_terminal_state"] in cb.PHASE_TERMINAL_STATES
        # The register is emitted by DEFAULT, so the disclosure cannot be omitted
        # by a caller that forgets to pass it.
        assert result["identified_bypasses"] == cb.disclosed_bypasses()
        named = {entry["reference"] for entry in result["identified_bypasses"]}
        for required in (
            "fpl_brain.route_optimizer.build_event_worlds caller-supplied bundles",
            "scripts/build_route_comparison.py",
            "scripts/final_operational_refresh_gw04.py",
            "scripts/gw4_current_final_board.py",
            "scripts/live_fire_gw04.py",
            "scripts/run_four_gw_decision.py event_support_from_db readiness path",
            "fpl_brain.monte_carlo.load_fixture_inputs 'or {}' upstream lookup",
        ):
            assert required in named, required
        assert "MISSING_PROJECTION" in cb.DISCLOSED_LIMITATION_REFERENCES[0]["reference"]
        # The limitation travels WITH the result, not only in the module register.
        assert result["disclosed_limitations"] == cb.disclosed_limitations()
        assert any(
            "MISSING_PROJECTION" in entry["reference"] for entry in result["disclosed_limitations"]
        )
        assert cb.unresolved_bypasses() == []
        assert result["unresolved_bypasses"] == []

        # Insufficient evidence makes the phase OPEN, and is reported as such
        # rather than as a failure of the bundle.
        from fpl_brain import calibration_evaluation as ce
        from fpl_brain import walk_forward_scoreboard as sb

        insufficient = _calibration(
            runs_by_event=runs,
            surfaces=[
                _surface("BRIER_P_START", diagnosis=ce.DIAGNOSIS_INSUFFICIENT,
                         sample=sb.SAMPLE_INSUFFICIENT, challenger=ce.STATUS_NOT_FITTED,
                         scored=0, observations=0),
            ],
        )
        limited = cb.certify_decision_horizon(
            conn, certification=artifact, events=HORIZON, cutoff=CUTOFF,
            required_versions=VERSIONS, calibration=insufficient, last_event=38,
        )
        assert limited["phase_terminal_state"] == cb.PHASE_OPEN
        assert any("EVIDENCE_LIMITED" in reason for reason in limited["phase_open_reasons"])
        assert limited["per_event"]["5"]["state"] == cb.STATE_EVIDENCE_LIMITED
    finally:
        conn.close()


def test_a_two_artifact_horizon_is_refused():
    conn, runs = _world()
    try:
        first = _artifact(conn, runs)
        with pytest.raises(cb.CertificationArtifactContradictory):
            cb.certify_decision_horizon(
                conn, certification=first, events=(5, 6, 7), cutoff=CUTOFF,
                required_versions=VERSIONS, last_event=38,
            )
        with pytest.raises(cb.CertificationArtifactAbsent):
            cb.certify_decision_horizon(
                conn, certification=None, events=HORIZON, cutoff=CUTOFF, last_event=38
            )
    finally:
        conn.close()


def test_one_event_of_foreign_calibration_evidence_refuses_that_event_alone():
    """A per-event REFUSAL must stay a structured, persisted state.

    Certification reads the calibration evidence per event, so an artifact that
    belongs to another prediction world for ONE event refuses that event while the
    other three certify.  The result must therefore carry a uniform per-event record
    for every event -- a roll-up that mixed record-key spellings would raise
    ``TypeError`` instead of reporting the token, which is the opposite of
    fail-closed.
    """

    conn, runs = _world()
    try:
        artifact = _artifact(conn, runs)
        foreign = _calibration(
            runs_by_event={**runs, 7: {**runs[7], "xpts_v1": runs[7]["xpts_v1"] + 5000}}
        )
        result = cb.certify_decision_horizon(
            conn, certification=artifact, events=HORIZON, cutoff=CUTOFF,
            required_versions=VERSIONS, calibration=foreign, last_event=38,
        )
        # Every event is present under the ONE key spelling, and the refusal names
        # the token and the family/run that caused it.
        assert sorted(result["per_event"]) == ["5", "6", "7", "8"]
        refused = result["per_event"]["7"]
        assert refused["state"] == cb.DIAG_CALIBRATION_WORLD_MISMATCH
        assert refused["bundle_identity"] is None
        assert any("xpts_v1 run" in reason for reason in refused["reasons"])
        assert any(
            f"GW7: {cb.DIAG_CALIBRATION_WORLD_MISMATCH}" in failure
            for failure in result["refusals"]
        )
        # The events whose evidence IS this world's certify normally.
        for event in (5, 6, 8):
            assert result["per_event"][str(event)]["bundle_identity"] is not None
        # One refused event makes the horizon incomplete, and the phase OPEN: the
        # horizon is never padded or truncated around it.
        assert result["horizon_state"] == fg.DECISION_HORIZON_INCOMPLETE
        assert result["phase_terminal_state"] == cb.PHASE_OPEN
        # The result still round-trips, refusals included.
        assert cb.certification_result_identity(
            json.loads(json.dumps(result, sort_keys=True, default=str))
        ) == cb.certification_result_identity(result)
    finally:
        conn.close()


def test_a_material_calibration_defect_blocks_the_claim_and_the_horizon():
    """A disclosed material defect blocks the CLAIM, and a blocked event the horizon.

    ``CERTIFICATION_BLOCKER_UNRESOLVED`` is not bundle incoherence and not absent
    evidence: the bundle is structurally valid and the artifact is present, but the
    evidence discloses a defect PE-8 promoted nothing to repair.  One blocked event
    makes the required horizon incomplete, and the decision is refused rather than
    taken on the events that remain.
    """

    from fpl_brain import calibration_evaluation as ce
    from fpl_brain import walk_forward_scoreboard as sb

    conn, runs = _world()
    try:
        artifact = _artifact(conn, runs)
        miscalibrated = _calibration(
            runs_by_event=runs,
            surfaces=[
                _surface("BRIER_P_START", diagnosis=ce.DIAGNOSIS_MISCALIBRATED,
                         sample=sb.SAMPLE_DESCRIPTIVE_ONLY, challenger=ce.STATUS_NOT_FITTED),
            ],
        )
        result = cb.certify_decision_horizon(
            conn, certification=artifact, events=HORIZON, cutoff=CUTOFF,
            required_versions=VERSIONS, calibration=miscalibrated, last_event=38,
        )
        for event in HORIZON:
            record = result["per_event"][str(event)]
            assert record["state"] == cb.STATE_CERTIFICATION_BLOCKER_UNRESOLVED
            assert record["structural_state"] == cb.STATE_CERTIFIED_COHERENT
            assert record["bundle_identity"] is not None
        assert result["horizon_state"] == fg.DECISION_HORIZON_INCOMPLETE
        assert sorted(result["horizon"]["blocked_events"]) == list(HORIZON)
        assert result["phase_terminal_state"] == cb.PHASE_OPEN
        assert any(
            cb.STATE_CERTIFICATION_BLOCKER_UNRESOLVED in reason
            for reason in result["phase_open_reasons"]
        )

        # The runner consumes the certified horizon's run ids and then GATES on PE-9's
        # horizon, so a blocked event refuses the decision instead of producing one.
        from pathlib import Path

        source = (
            Path(__file__).resolve().parents[1] / "scripts" / "run_four_gw_decision.py"
        ).read_text(encoding="utf-8")
        gate = source.index('if pe9_certification["horizon_state"] == fg.DECISION_HORIZON_INCOMPLETE')
        assert source.index("certification_result_identity(pe9_certification)") < gate
    finally:
        conn.close()


def test_a_certified_bundle_payload_is_read_under_either_key_spelling():
    """``certified_bundles`` arrives from JSON (str keys) and from memory (int keys)."""

    conn, runs = _world()
    try:
        payload = {"runs": runs[5], "event": 5}
        assert cb.certified_bundle_payload({"5": payload}, 5) is payload
        assert cb.certified_bundle_payload({5: payload}, 5) is payload
        assert cb.certified_bundle_payload({}, 5) == {}
        # A present but empty payload stays empty rather than resolving to another
        # event's -- the caller must not be handed a different world's snapshots.
        assert cb.certified_bundle_payload({"5": {}}, 5) == {}

        # An integer-keyed artifact certifies to the SAME per-event identities, so
        # the spelling cannot change what an event was certified under.
        artifact = _artifact(conn, runs)
        as_strings = cb.certify_decision_horizon(
            conn, certification=artifact, events=HORIZON, cutoff=CUTOFF,
            required_versions=VERSIONS, last_event=38,
        )
        int_keyed = dict(artifact)
        int_keyed["certified_bundles"] = {
            int(event): bundle for event, bundle in artifact["certified_bundles"].items()
        }
        as_ints = cb.certify_decision_horizon(
            conn, certification=int_keyed, events=HORIZON, cutoff=CUTOFF,
            required_versions=VERSIONS, last_event=38,
        )
        assert as_ints["per_event_bundle_identity"] == as_strings["per_event_bundle_identity"]
        assert cb.certification_result_identity(as_ints) == cb.certification_result_identity(as_strings)
    finally:
        conn.close()


def test_the_loaders_boundary_is_declared_and_does_not_accept_a_bare_id_map():
    """A bundle that cannot declare provenance is refused, never substituted."""

    conn, runs = _world()
    try:
        with pytest.raises(cb.CertificationRefused) as caught:
            cb.assert_event_bundle_certified(conn, {"runs": runs[5]}, event=5)
        assert caught.value.token == cb.DIAG_CERTIFICATION_ARTIFACT_ABSENT

        # A certified bundle loaded with no connection is still required to declare
        # a version for every family it names.
        from fpl_brain.free_hit_request_adapter import _CertifiedRunIds

        partial = _CertifiedRunIds(5, {"runs": runs[5], "model_versions": {}})
        with pytest.raises(cb.CertificationRefused) as caught:
            cb.assert_event_bundle_certified(
                None, partial, event=5, certification=_artifact(conn, runs)
            )
        assert caught.value.token == cb.STATE_EVIDENCE_MISSING

        declared = _CertifiedRunIds(
            5,
            {
                "runs": runs[5],
                "cutoff": CUTOFF,
                "model_versions": VERSIONS,
                "code_snapshot_sha256": CODE_SNAPSHOT,
                "data_snapshot_sha256": DATA_SNAPSHOT,
                "planning_context_hash": CONTEXT_HASH,
            },
        )
        proof = cb.assert_event_bundle_certified(
            conn, declared, event=5, certification=_artifact(conn, runs)
        )
        assert proof["certified_bundle_identity"].startswith("sha256:")
        # The identity it declares is the ONE algorithm's identity.
        assert proof["certified_bundle_identity"] == cb.certified_bundle_identity_for(
            event=5, cutoff=CUTOFF, runs=runs[5], model_versions=VERSIONS,
            code_snapshot_sha256=CODE_SNAPSHOT, data_snapshot_sha256=DATA_SNAPSHOT,
            planning_context_hash=CONTEXT_HASH,
        )
    finally:
        conn.close()


def test_calibration_identity_is_a_pure_function_of_declared_fields():
    conn, runs = _world()
    try:
        calibration = _calibration(runs_by_event=runs)
        first = cb.calibration_identity(calibration)
        assert first.startswith("sha256:")
        # Recomputing from the stored bytes yields the same identity.
        assert cb.calibration_identity(json.loads(json.dumps(calibration))) == first
        # A reordered mapping is the same artifact.
        assert cb.calibration_identity(dict(reversed(list(calibration.items())))) == first
        # A missing identity block is None -- a DIFFERENT fact from unknown.
        assert cb.calibration_identity({"schema": "x"}) is None
    finally:
        conn.close()


def test_the_certification_module_still_exposes_exactly_three_legacy_entry_points():
    """PE-9 EXTENDS the existing module; it adds no second certification system."""

    for name in (
        "validate_certified_bundle",
        "certified_bundle_from_explicit_ids",
        "certify_horizon_bundles",
    ):
        assert callable(getattr(cb, name))
    # There is still deliberately no "discover the latest run per family" helper.
    assert not [
        name
        for name in dir(cb)
        if "latest" in name.lower() or "newest" in name.lower() or "discover" in name.lower()
    ]
    # PE-9 introduces no numerical threshold of its own.
    assert not [
        name
        for name in dir(cb)
        if name.isupper() and ("TOLERANCE" in name or "THRESHOLD" in name)
    ]


def test_the_registry_of_states_is_fail_closed_and_never_a_boolean():
    assert len(cb.PER_BUNDLE_STATES) == 7
    assert set(cb.NON_FAILURE_BUNDLE_STATES) < set(cb.PER_BUNDLE_STATES)
    assert set(cb.BLOCKING_BUNDLE_STATES) < set(cb.PER_BUNDLE_STATES)
    assert not set(cb.NON_FAILURE_BUNDLE_STATES) & set(cb.BLOCKING_BUNDLE_STATES)
    # The precedence is declared, and the most blocking fact wins.
    assert cb.bundle_state_from_reasons([]) == cb.STATE_CERTIFIED_COHERENT
    assert cb.bundle_state_from_reasons(["plain incoherence"]) == cb.STATE_PREDICTIVE_BUNDLE_INCOHERENT
    for state in cb.BLOCKING_BUNDLE_STATES:
        assert cb.bundle_state_from_reasons([f"{state}: something"]) == state
    # A legitimate non-failure never outranks a refusal.
    assert cb.bundle_state_from_reasons([
        f"{cb.STATE_EVIDENCE_LIMITED}: thin", f"{cb.STATE_UNSUPPORTED_MODEL_VERSION}: old",
    ]) == cb.STATE_UNSUPPORTED_MODEL_VERSION


def test_bundle_identity_ignores_required_versions_but_binds_run_ids():
    conn, runs = _world()
    try:
        first = cb.certified_bundle_from_explicit_ids(
            conn, event=5, cutoff=CUTOFF, runs=runs[5], required_versions=VERSIONS
        )
        second = cb.certified_bundle_from_explicit_ids(
            conn, event=5, cutoff=CUTOFF, runs=runs[5], required_versions=None
        )
        # ``required_versions`` is a gate, not a predictive quantity, so it is not
        # part of the identity -- the recorded versions are.
        assert first.bundle_identity() == second.bundle_identity()
        assert first.model_versions == second.model_versions

        other = cb.certified_bundle_from_explicit_ids(
            conn, event=6, cutoff=CUTOFF, runs=runs[6], required_versions=VERSIONS
        )
        assert other.bundle_identity() != first.bundle_identity()
    finally:
        conn.close()


def test_the_decision_runner_certifies_and_persists_the_certification_result():
    """The production runner must CERTIFY the horizon and persist the result.

    Source-level, because exercising it end-to-end needs a full production
    database.  What is asserted is that the runner crosses the PE-9 boundary, that
    the required model versions come from the ONE declared source, and that the
    certification result is persisted WITH the decision rather than narrated.
    """

    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    source = (root / "scripts" / "run_four_gw_decision.py").read_text(encoding="utf-8")
    assert "certify_decision_horizon(" in source
    assert '"certification": pe9_certification' in source
    assert "certification_result_identity(" in source
    # The required versions reach the boundary from the artifact the certifier minted.
    assert 'certification.get("required_model_versions") or None' in source
    # A PE-8 calibration artifact can be consulted at the boundary (gap 4), and its
    # absence yields a STATE rather than a refusal.
    assert '"--calibration"' in source
    assert "calibration=calibration_artifact" in source
    # The readiness view is reconciled against the certified generation, and the
    # divergence is persisted rather than left implicit.
    assert "readiness_generation_divergence" in source
    assert "event_support_from_db" in source
    # The NON-PRODUCTION readiness path is still labelled as such at the source, so
    # it cannot be mistaken for the certified generation.
    assert "NON_PRODUCTION" in source or "NON-PRODUCTION" in source

    certifier = (root / "scripts" / "certify_gw5_gw8.py").read_text(encoding="utf-8")
    # Every certification call site supplies the required versions and records the
    # code identity on the bundle payload the consumer recomputes the identity from.
    assert "required_versions=required_versions" in certifier
    assert "code_snapshot_sha256=certification_code_snapshot" in certifier
    assert "declared_required_versions()" in certifier


def _optimizer_config(**over):
    base = dict(events=(5,), search_draws=8, seed=20260911)
    base.update(over)
    return ro.OptimizerConfig(**base)


# ---------------------------------------------------------------------------
# Adversarial: a SELF-CONSISTENT bundle built from arbitrary existing runs
# ---------------------------------------------------------------------------
#
# The attack this section models is the one a certification boundary exists to stop:
# a caller assembles a bundle from run ids that EXIST, mints its identity with the
# shared algorithm (so the identity binds its own run ids), declares the
# authoritative model version of every family, and presents it at each boundary.  It
# is internally perfect and it is still not the CERTIFIED bundle, so every load
# boundary must refuse it -- including a warm cache directory, because a cache hit is
# a predictive load like any other.


def _alternative_world(conn, event: int, *, first_run: int = 920) -> dict[str, int]:
    """A second, COMPLETE and COHERENT five-family world for one event.

    Later reruns of every family, wired to each other, at the same cutoff, with the
    same model versions and the same planning context as the certified generation:
    everything a self-consistency check can see is in order.  It is simply a
    different predictive world, which is the fact only the ARTIFACT can authorise.
    """

    with conn:
        ids = {
            "minutes_v1": first_run,
            "team_strength_v1": first_run + 1,
            "player_rates_v1": first_run + 2,
            "xpts_v1": first_run + 3,
            "monte_carlo_v1": first_run + 4,
        }
        for family, run_id in ids.items():
            _run(conn, run_id, family, event)
        fixture_id = int(
            conn.execute("SELECT id FROM fixtures WHERE event=?", (int(event),)).fetchone()["id"]
        )
        _xpts_row(
            conn, ids["xpts_v1"], fixture_id, event,
            minutes_run_id=ids["minutes_v1"], team_run_id=ids["team_strength_v1"],
            rate_run_id=ids["player_rates_v1"],
        )
        _mc_row(
            conn, ids["monte_carlo_v1"], fixture_id, event,
            xpts_run_id=ids["xpts_v1"], minutes_run_id=ids["minutes_v1"],
            team_run_id=ids["team_strength_v1"], rate_run_id=ids["player_rates_v1"],
        )
    return ids


def _self_consistent_bundle(runs, *, event: int = 5, **over):
    """A bundle that satisfies every self-consistency property and no authorisation.

    The identity is minted by the ONE shared algorithm from these exact ids, so it
    BINDS them; the versions are the authoritative ones; the snapshots and the
    context hash are the certified ones.  Nothing here is self-contradictory -- only
    unrecorded.
    """

    return rc.certified_event_bundle(
        event=int(event), runs=runs, cutoff=CUTOFF, model_versions=VERSIONS,
        code_snapshot_sha256=CODE_SNAPSHOT, data_snapshot_sha256=DATA_SNAPSHOT,
        planning_context_hash=CONTEXT_HASH, **over,
    )


def test_adversarial_a_self_consistent_bundle_is_refused_by_the_optimizer(tmp_path):
    """The optimizer's loader refuses a coherent but UNCERTIFIED prediction world."""

    from test_route_optimizer import _scenario, _universe

    conn, runs = _world()
    try:
        artifact = _artifact(conn, runs)
        alternative = _self_consistent_bundle(_alternative_world(conn, 5))

        # The bundle really is self-consistent: its identity binds its own run ids
        # and it declares the authoritative version of every family it names.
        assert alternative.certified_bundle_identity == cb.canonical_bundle_identity(
            alternative.as_identity_payload()
        )
        assert alternative.certified_runs() != runs[5]

        universe, state, meta = _universe()
        with pytest.raises(cb.CertificationArtifactContradictory):
            ro.optimize(
                universe=universe, initial_state=state, scenario=_scenario(), player_meta=meta,
                bundles={5: alternative}, conn=conn,
                config=ro.OptimizerConfig(events=(5,), search_draws=6, seed=20260911),
                certification=artifact, cache_dir=tmp_path, exact_cache={},
            )
    finally:
        conn.close()


def test_adversarial_a_warm_cache_does_not_authorise_a_self_consistent_bundle(tmp_path):
    """A cache HIT is a predictive load, so it crosses the boundary too.

    The cache is warmed for the alternative bundle's own key -- the key a caller
    would compute -- so a boundary that trusted the key, or that checked identity
    after the lookup, would hand back worlds the certification never authorised.  The
    refusal must come first.
    """

    conn, runs = _world()
    try:
        artifact = _artifact(conn, runs)
        alternative = _self_consistent_bundle(_alternative_world(conn, 5))
        config = _optimizer_config()
        union = [1, 2]
        key = ro.world_cache_key(event=5, bundle=alternative, config=config, union_ids=union)
        payload = json.dumps({
            "worlds": 2, "player_ids": [1, 2],
            "core": {"1": [99.0, 99.0], "2": [99.0, 99.0]},
            "minutes": {"1": [90.0, 90.0], "2": [90.0, 90.0]},
            "expected_bonus": {"1": 0.0, "2": 0.0},
            "role_actionability": {"1": False, "2": False},
        })
        (tmp_path / f"{key}.json").write_text(payload, encoding="utf-8")

        with pytest.raises(cb.CertificationArtifactContradictory):
            ro.build_event_worlds(
                conn, {5: alternative}, 5, union, config, cache_dir=tmp_path,
                certification=artifact,
            )

        # The same call with the CERTIFIED bundle is a cache hit, so the refusal
        # above is about provenance and not about the cache being unusable.
        certified = _self_consistent_bundle(runs[5])
        certified_key = ro.world_cache_key(
            event=5, bundle=certified, config=config, union_ids=union
        )
        (tmp_path / f"{certified_key}.json").write_text(payload, encoding="utf-8")
        matrix, info = ro.build_event_worlds(
            conn, {5: certified}, 5, union, config, cache_dir=tmp_path, certification=artifact,
        )
        assert info["source"] == "cache"
        assert matrix["core"][1] == [99.0, 99.0]
    finally:
        conn.close()


def test_adversarial_the_comparator_refuses_a_self_consistent_bundle():
    """The comparator's DB branch is a predictive loader, so it refuses one too."""

    from fpl_brain import transfer_state as ts

    conn, runs = _world()
    try:
        artifact = _artifact(conn, runs)
        alternative = _self_consistent_bundle(_alternative_world(conn, 5))
        state = ts.RouteState(
            event=5, players=(ts.RoutePlayer(1, "MID", 1, 50),), bank_tenths=0, free_transfers=1,
        )
        route = rc.TransferRoute(
            route_id="roll",
            steps=(rc.RouteStep(event=5, transfer_batch=ts.TransferBatch(())),),
        )
        with pytest.raises(cb.CertificationRefused) as caught:
            rc.compare_routes(
                bundles={5: alternative}, routes=[route], initial_state=state,
                scenario=rc.flat_current_price_scenario(
                    ts.PriceSnapshot(event=5, prices={1: 50, 2: 50}), [5]
                ),
                player_meta={}, conn=conn, certification=artifact,
            )
        assert caught.value.token == cb.DIAG_CERTIFICATION_ARTIFACT_CONTRADICTORY
    finally:
        conn.close()


def test_adversarial_free_hit_route_worlds_refuse_a_self_consistent_bundle(tmp_path):
    """Free Hit route worlds present the artifact, so agreeing run ids are not enough."""

    from fpl_brain import chip_free_hit as fh
    from fpl_brain import free_hit_request_adapter as fha

    conn, runs = _world()
    try:
        artifact = _artifact(conn, runs)
        alternative = _self_consistent_bundle(_alternative_world(conn, 5))
        # Built the way the adapter builds a bundle from a certified record: the row
        # carries its own runs, versions and snapshots, so its identity binds its own
        # run ids -- and it is still not the recorded bundle.
        row = dict(alternative.as_identity_payload())
        row["certified_bundle_identity"] = alternative.certified_bundle_identity
        with pytest.raises(fha.FreeHitAdapterError) as caught:
            fha.load_certified_route_worlds(
                conn, {5: fha._CertifiedRunIds(5, row)}, arm="SAVE", expected_events=(5,),
                union_ids=(1, 2), config=_optimizer_config(), cache_dir=tmp_path,
                certification=artifact,
            )
        assert fh.FH_DECISION_AUTHORITY_REQUIRED in str(caught.value)

        # The artifact's OWN bundle is accepted through the same call, so the refusal
        # is about the substituted bundle and nothing else.
        certified_row = {
            "runs": dict(runs[5]), "cutoff": CUTOFF, "model_versions": VERSIONS,
            "code_snapshot_sha256": CODE_SNAPSHOT, "data_snapshot_sha256": DATA_SNAPSHOT,
            "planning_context_hash": CONTEXT_HASH,
        }
        matrix, _info = ro.build_event_worlds(
            conn, {5: fha._CertifiedRunIds(5, certified_row)}, 5, (1, 2), _optimizer_config(),
            certification=artifact,
        )
        assert matrix["worlds"] == 8
    finally:
        conn.close()


def _defective_world(defect: str, value) -> sqlite3.Connection:
    """The SAME run ids ``_world()`` assigns, with ONE family's evidence wrong.

    A completed projection run is immutable, so a defect is never produced by
    mutating a certified row: it is inserted the way an incoherent generation would
    arrive in the first place.
    """

    ids = {"minutes_v1": 100, "team_strength_v1": 101, "player_rates_v1": 102,
           "xpts_v1": 103, "monte_carlo_v1": 104}
    conn = connect_database(":memory:")
    _base_world(conn)
    with conn:
        _add_event(conn, 5)
        _add_fixture(conn, 1000, 5, 1, 2)
        for family, run_id in ids.items():
            over: dict = {}
            if family == "xpts_v1":
                if defect == "status":
                    over["status"] = value
                elif defect == "planning_event":
                    over["event"] = value
                elif defect == "data_cutoff":
                    over["cutoff"] = value
                elif defect == "model_version":
                    over["version"] = value
            _run(conn, run_id, family, over.pop("event", 5), **over)
        _xpts_row(
            conn, ids["xpts_v1"], 1000, 5,
            minutes_run_id=int(value) if defect == "dependency_edge" else ids["minutes_v1"],
            team_run_id=ids["team_strength_v1"], rate_run_id=ids["player_rates_v1"],
        )
        _mc_row(
            conn, ids["monte_carlo_v1"], 1000, 5, xpts_run_id=ids["xpts_v1"],
            minutes_run_id=ids["minutes_v1"], team_run_id=ids["team_strength_v1"],
            rate_run_id=ids["player_rates_v1"],
        )
    return conn


def test_the_load_boundary_re_proves_the_recorded_closure_from_the_rows():
    """A certification artifact is a CLAIM about the run rows, never a substitute.

    The artifact records that these families were certified together; the load
    boundary re-reads the rows themselves, so a family that is at another event,
    incomplete, cut at another cutoff, re-versioned or wired to a different upstream
    run cannot authorise a prediction.  The recorded closure is re-proven at every
    load rather than assumed from the artifact.
    """

    conn, runs = _world()
    try:
        artifact = _artifact(conn, runs)
        certified = _self_consistent_bundle(runs[5])

        # The artifact authorises these ids, and a generation carrying them loads.
        matrix, info = ro.build_event_worlds(
            conn, {5: certified}, 5, [1, 2], _optimizer_config(), certification=artifact,
        )
        assert info["source"] == "generated" and matrix["worlds"] == 8

        for defect, value, token in (
            ("status", "running", cb.STATE_PREDICTIVE_BUNDLE_INCOHERENT),
            ("planning_event", 6, cb.STATE_PREDICTIVE_BUNDLE_INCOHERENT),
            ("data_cutoff", OTHER_CUTOFF, cb.STATE_PREDICTIVE_BUNDLE_INCOHERENT),
            ("dependency_edge", 912, cb.STATE_PREDICTIVE_BUNDLE_INCOHERENT),
            ("model_version", "xpts_v0.0.0", cb.STATE_UNSUPPORTED_MODEL_VERSION),
        ):
            other = _defective_world(defect, value)
            try:
                with pytest.raises(cb.CertificationRefused) as caught:
                    ro.build_event_worlds(
                        other, {5: certified}, 5, [1, 2], _optimizer_config(),
                        certification=artifact,
                    )
                assert caught.value.token == token, (defect, caught.value.token)
            finally:
                other.close()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Adversarial: a self-consistent artifact-shaped MAPPING with no authorisation
# ---------------------------------------------------------------------------
#
# The second attack: the caller presents something that LOOKS like a certification
# artifact -- every bundle in it is real, every identity in it binds its own run ids,
# every declared version is the authoritative one -- and simply omits the fields the
# canonical loader's contract requires (the state that admits the bundles, the
# authorisation flag, the audited status, the wiring identity).  A boundary that only
# checked the bundles would load predictive data on the strength of a mapping the
# engine never authorised, so every predictive-load boundary applies the contract
# itself: a validated artifact is passed through untouched, and a raw mapping has the
# SAME contract re-run over it before anything predictive is read.


def test_adversarial_an_unauthorised_mapping_is_refused_by_the_optimizer_and_cache(tmp_path):
    conn, runs = _world()
    try:
        artifact = _artifact(conn, runs)
        unauthorised = _unauthorised(artifact)
        # Self-consistent: the bundles still bind their own identities.
        assert unauthorised["certified_bundle_identity"] == artifact["certified_bundle_identity"]
        assert "decision_search_permitted" not in unauthorised

        certified = _self_consistent_bundle(runs[5])
        config = _optimizer_config()
        union = [1, 2]

        with pytest.raises(cb.CertificationRefused) as caught:
            ro.build_event_worlds(
                conn, {5: certified}, 5, union, config, certification=unauthorised,
            )
        assert caught.value.token == cb.DIAG_CERTIFICATION_ARTIFACT_UNVALIDATED

        # A WARM cache does not authorise it either: the contract is applied before the
        # cache is read, so a cache hit cannot stand in for the authorisation.
        key = ro.world_cache_key(event=5, bundle=certified, config=config, union_ids=union)
        (tmp_path / f"{key}.json").write_text(json.dumps({
            "worlds": 2, "player_ids": [1, 2],
            "core": {"1": [99.0, 99.0], "2": [99.0, 99.0]},
            "minutes": {"1": [90.0, 90.0], "2": [90.0, 90.0]},
            "expected_bonus": {"1": 0.0, "2": 0.0},
            "role_actionability": {"1": False, "2": False},
        }), encoding="utf-8")
        with pytest.raises(cb.CertificationRefused) as caught:
            ro.build_event_worlds(
                conn, {5: certified}, 5, union, config, cache_dir=tmp_path,
                certification=unauthorised,
            )
        assert caught.value.token == cb.DIAG_CERTIFICATION_ARTIFACT_UNVALIDATED

        from test_route_optimizer import _scenario, _universe

        universe, state, meta = _universe()
        with pytest.raises(cb.CertificationRefused) as caught:
            ro.optimize(
                universe=universe, initial_state=state, scenario=_scenario(), player_meta=meta,
                bundles={5: certified}, conn=conn, config=config, cache_dir=tmp_path,
                certification=unauthorised, exact_cache={},
            )
        assert caught.value.token == cb.DIAG_CERTIFICATION_ARTIFACT_UNVALIDATED

        # The AUTHORISED artifact loads the same world through the same call, so the
        # refusal above is about the missing authorisation and nothing else.
        matrix, info = ro.build_event_worlds(
            conn, {5: certified}, 5, union, config, cache_dir=tmp_path, certification=artifact,
        )
        assert info["source"] == "cache" and matrix["core"][1] == [99.0, 99.0]
    finally:
        conn.close()


def test_adversarial_an_unauthorised_mapping_is_refused_by_the_comparator():
    from fpl_brain import transfer_state as ts

    conn, runs = _world()
    try:
        artifact = _artifact(conn, runs)
        unauthorised = _unauthorised(artifact)
        certified = _self_consistent_bundle(runs[5])
        state = ts.RouteState(
            event=5, players=(ts.RoutePlayer(1, "MID", 1, 50),), bank_tenths=0, free_transfers=1,
        )
        route = rc.TransferRoute(
            route_id="roll",
            steps=(rc.RouteStep(event=5, transfer_batch=ts.TransferBatch(())),),
        )
        scenario = rc.flat_current_price_scenario(
            ts.PriceSnapshot(event=5, prices={1: 50, 2: 50}), [5]
        )
        with pytest.raises(cb.CertificationRefused) as caught:
            rc.compare_routes(
                bundles={5: certified}, routes=[route], initial_state=state,
                scenario=scenario, player_meta={}, conn=conn, certification=unauthorised,
            )
        assert caught.value.token == cb.DIAG_CERTIFICATION_ARTIFACT_UNVALIDATED

        accepted = rc.compare_routes(
            bundles={5: certified}, routes=[route], initial_state=state,
            scenario=scenario, player_meta={}, conn=conn, certification=artifact,
            simulations=4,
        )
        assert accepted["routes"]
    finally:
        conn.close()


def test_adversarial_an_unauthorised_mapping_is_refused_by_the_free_hit_loader(tmp_path):
    from fpl_brain import chip_free_hit as fh
    from fpl_brain import free_hit_request_adapter as fha

    conn, runs = _world()
    try:
        artifact = _artifact(conn, runs)
        unauthorised = _unauthorised(artifact)
        row = {
            "runs": dict(runs[5]), "cutoff": CUTOFF, "model_versions": VERSIONS,
            "code_snapshot_sha256": CODE_SNAPSHOT, "data_snapshot_sha256": DATA_SNAPSHOT,
            "planning_context_hash": CONTEXT_HASH,
        }
        with pytest.raises(fha.FreeHitAdapterError) as caught:
            fha.load_certified_route_worlds(
                conn, {5: fha._CertifiedRunIds(5, row)}, arm="SAVE", expected_events=(5,),
                union_ids=(1, 2), config=_optimizer_config(), cache_dir=tmp_path,
                certification=unauthorised,
            )
        assert fh.FH_DECISION_AUTHORITY_REQUIRED in str(caught.value)
        assert cb.DIAG_CERTIFICATION_ARTIFACT_UNVALIDATED in str(caught.value)

        # The authorised artifact loads the SAME certified runs through the same call.
        matrix, _info = ro.build_event_worlds(
            conn, {5: fha._CertifiedRunIds(5, row)}, 5, (1, 2), _optimizer_config(),
            certification=artifact,
        )
        assert matrix["worlds"] == 8
    finally:
        conn.close()


def test_adversarial_an_unauthorised_mapping_is_refused_by_refinement_and_stability(tmp_path):
    """Both stages forward the artifact into the loader, so both apply the contract."""

    from fpl_brain import finalist_refinement as fr
    from fpl_brain import route_stability as rs
    from test_route_optimizer import EVENTS, _config as _small_config, _provider, _scenario, _universe

    universe, state, meta = _universe()
    scenario = _scenario()
    config = _small_config()
    # The Stage-1/Stage-2 fixtures score event 4, so the certified world is event 4's.
    conn, runs = _world(events=EVENTS)
    try:
        artifact = _artifact(conn, runs, events=EVENTS)
        unauthorised = _unauthorised(artifact)
        certified = _self_consistent_bundle(runs[int(EVENTS[0])], event=int(EVENTS[0]))
        bundles = {int(EVENTS[0]): certified}

        # Stage 1 in the declared non-production worlds, so the refinement has a
        # Stage-1 result to refine and the ONLY thing under test is the artifact the
        # refinement and the ladder forward to the certified loader.
        stage1 = ro.optimize(
            universe=universe, initial_state=state, scenario=scenario, player_meta=meta,
            config=config, non_production_worlds=ro.NonProductionWorlds(
                declaration="tests: pe-9 adversarial stage 1", provider=_provider()
            ),
        )
        with pytest.raises(cb.CertificationRefused) as caught:
            fr.refine_finalists(
                universe=universe, initial_state=state, scenario=scenario, player_meta=meta,
                bundles=bundles, conn=conn, base_config=config, stage1_result=stage1,
                stage2_draws=config.search_draws * 2, certification=unauthorised,
                cache_dir=tmp_path, verify_prefix=False, exact_cache={},
            )
        assert caught.value.token == cb.DIAG_CERTIFICATION_ARTIFACT_UNVALIDATED

        with pytest.raises(cb.CertificationRefused) as caught:
            rs.run_ladder(
                universe=universe, initial_state=state, scenario=scenario, player_meta=meta,
                base_config=config, budgets=[2], bundles=bundles, conn=conn,
                certification=unauthorised, cache_dir=tmp_path, exact_cache={},
            )
        assert caught.value.token == cb.DIAG_CERTIFICATION_ARTIFACT_UNVALIDATED
    finally:
        conn.close()


def test_the_validated_artifact_is_immutable_and_a_self_consistent_mapping_is_revalidated():
    """The capability can be neither forged nor edited, and it round-trips."""

    conn, runs = _world()
    try:
        artifact = _artifact(conn, runs)
        validated = cb.validate_certification_artifact(artifact)
        assert isinstance(validated, cb.ValidatedCertificationArtifact)
        # Idempotent: the authorisation itself is what a caller passes on.
        assert cb.validate_certification_artifact(validated) is validated
        assert validated["certified_bundles"]["5"]["runs"] == runs[5]

        # Minting one from a raw mapping is refused, so the type IS the authorisation.
        with pytest.raises(cb.CertificationArtifactUnvalidated):
            cb.ValidatedCertificationArtifact(artifact)
        # ... and it cannot be edited after the check.
        with pytest.raises(TypeError):
            validated["certified_bundles"]["5"]["runs"]["minutes_v1"] = 999
        with pytest.raises(TypeError):
            validated["events"] = [9]
        with pytest.raises(TypeError):
            validated["certified_bundles"]["5"]["cutoff"] = OTHER_CUTOFF
        # The artifact the boundary validated is unchanged by those attempts.
        assert validated["certified_bundles"]["5"]["cutoff"] == CUTOFF

        # A raw mapping that DOES carry every authorization field is revalidated and
        # accepted -- the contract is re-run, never assumed from the mapping's shape.
        revalidated = cb.validate_certification_artifact(json.loads(json.dumps(artifact)))
        assert isinstance(revalidated, cb.ValidatedCertificationArtifact)
        assert revalidated.validated_identity == validated.validated_identity

        # Dropping ONE authorization field is enough to turn it back into a mapping
        # that is not an authorisation.
        for field in AUTHORIZATION_FIELDS:
            with pytest.raises(cb.CertificationArtifactUnvalidated):
                cb.validate_certification_artifact(_unauthorised(artifact, drop=(field,)))
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Adversarial: an arbitrary matrix carrying the CORRECT copied stamp
# ---------------------------------------------------------------------------
#
# The third attack: a caller builds its own worlds (or edits a loaded matrix) and
# copies the certified-bundle stamp onto them.  A stamp is caller-writable, so it
# authorises nothing; what admits a prebuilt matrix is the loader's own record of the
# content it produced, verified by canonical content identity and bound to the event
# and to the certified bundle the artifact records.


def _stamped_by_hand(matrix, identity: str):
    """A caller-built matrix carrying the CERTIFIED bundle stamp it copied."""

    from fpl_brain.manager_worlds import MATRIX_CERTIFIED_BUNDLE_KEY

    stamped = {key: value for key, value in matrix.items() if not str(key).startswith("_p2_")}
    stamped[MATRIX_CERTIFIED_BUNDLE_KEY] = str(identity)
    return stamped


def _certified_matrix_loads(conn, runs, artifact, tmp_path, event: int):
    """The loader's OWN 8-world matrix for one event, plus its certified identity."""

    certified = _self_consistent_bundle(runs[int(event)], event=int(event))
    matrix, info = ro.build_event_worlds(
        conn, {int(event): certified}, int(event), (1, 2), _optimizer_config(events=(int(event),)),
        certification=artifact, cache_dir=tmp_path,
    )
    return matrix, info


def test_adversarial_a_copied_stamp_does_not_authorise_a_matrix(tmp_path):
    """Every door that admits a prebuilt matrix refuses a stamped hand-built one."""

    from fpl_brain.manager_worlds import MATRIX_CERTIFIED_BUNDLE_KEY
    from fpl_brain import finalist_refinement as fr
    from fpl_brain import route_stability as rs
    from test_route_optimizer import (
        EVENTS, _config as _small_config, _provider, _scenario, _universe,
    )

    event = int(EVENTS[0])
    conn, runs = _world(events=EVENTS)
    try:
        artifact = _artifact(conn, runs, events=EVENTS)
        certified = _self_consistent_bundle(runs[event], event=event)
        matrix, info = _certified_matrix_loads(conn, runs, artifact, tmp_path, event)
        identity = info["certified_bundle_identity"]

        # The loader's own output carries the stamp, and IS admitted.
        assert str(matrix[MATRIX_CERTIFIED_BUNDLE_KEY]) == str(identity)

        universe, state, meta = _universe()
        scenario = _scenario()
        config = _small_config()

        def _optimize_with(worlds, **over):
            return ro.optimize(
                universe=universe, initial_state=state, scenario=scenario, player_meta=meta,
                bundles={event: certified}, conn=conn, config=ro.OptimizerConfig(
                    events=EVENTS, search_draws=8, seed=20260911, policy_selection_worlds=6,
                ),
                certification=artifact, prebuilt_worlds=worlds, exact_cache={}, **over,
            )

        # (1) A hand-built matrix carrying the CORRECT copied stamp is refused.
        hand_built = _stamped_by_hand(
            {"worlds": 8, "player_ids": [1, 2],
             "core": {1: [99.0] * 8, 2: [99.0] * 8},
             "minutes": {1: [90.0] * 8, 2: [90.0] * 8},
             "expected_bonus": {1: 0.0, 2: 0.0},
             "role_actionability": {1: False, 2: False}},
            identity,
        )
        assert str(hand_built[MATRIX_CERTIFIED_BUNDLE_KEY]) == str(identity)
        with pytest.raises(cb.CertificationRefused) as caught:
            _optimize_with({event: hand_built})
        assert caught.value.token == ro.DIAG_WORLD_MATRIX_NOT_LOADER_ISSUED

        # (2) EDITING a loaded matrix keeps its stamp but not its content identity.
        edited = {key: value for key, value in matrix.items()}
        edited["core"] = {1: [99.0] * int(edited["worlds"]), 2: [99.0] * int(edited["worlds"])}
        assert str(edited[MATRIX_CERTIFIED_BUNDLE_KEY]) == str(identity)
        with pytest.raises(cb.CertificationRefused) as caught:
            _optimize_with({event: edited})
        assert caught.value.token == ro.DIAG_WORLD_MATRIX_NOT_LOADER_ISSUED

        # (3) A loaded matrix offered for a DIFFERENT event's authorisation is refused:
        # the capability is bound to the event it was issued for.
        with pytest.raises(cb.CertificationRefused):
            ro.require_certified_prebuilt_matrix(
                matrix, event=event + 1, certified_bundle_identity=str(identity)
            )

        # (4) Refinement and the stability ladder forward prebuilt worlds to the SAME
        # door, so neither admits one either.
        with pytest.raises(cb.CertificationRefused) as caught:
            fr.refine_finalists(
                universe=universe, initial_state=state, scenario=scenario, player_meta=meta,
                bundles={event: certified}, conn=conn, base_config=config,
                stage1_result=ro.optimize(
                    universe=universe, initial_state=state, scenario=scenario, player_meta=meta,
                    config=config, non_production_worlds=ro.NonProductionWorlds(
                        declaration="tests: pe-9 adversarial stage 1", provider=_provider()
                    ),
                ),
                stage2_draws=config.search_draws * 2, certification=artifact,
                prebuilt_worlds={event: hand_built}, verify_prefix=False, exact_cache={},
            )
        assert caught.value.token == ro.DIAG_WORLD_MATRIX_NOT_LOADER_ISSUED

        with pytest.raises(cb.CertificationRefused) as caught:
            rs.run_ladder(
                universe=universe, initial_state=state, scenario=scenario, player_meta=meta,
                base_config=config, budgets=[2], bundles={event: certified}, conn=conn,
                certification=artifact, prebuilt_worlds={event: hand_built}, exact_cache={},
            )
        assert caught.value.token == ro.DIAG_WORLD_MATRIX_NOT_LOADER_ISSUED
    finally:
        conn.close()


def test_adversarial_the_manager_world_load_requires_the_certified_run_ids():
    """The manager policy matrix is a decision input, so its worlds cross the boundary.

    ``manager_worlds.build_manager_worlds`` reads the Minutes / team-strength / xPts
    runs and simulates the shared worlds the manager policy is scored in.  A
    hand-assembled run-id set is refused, and so is the same set dressed in an
    artifact-shaped mapping that carries no authorisation.
    """

    from fpl_brain import manager_worlds as mw

    conn, runs = _world()
    try:
        artifact = _artifact(conn, runs)
        certified = runs[5]
        with pytest.raises(cb.CertificationRefused) as caught:
            mw.build_manager_worlds(
                conn, planning_event=5, minutes_run_id=certified["minutes_v1"],
                xpts_run_id=certified["xpts_v1"], team_run_id=certified["team_strength_v1"],
                squad_ids=[1], simulations=4,
            )
        assert caught.value.token == cb.DIAG_CERTIFICATION_ARTIFACT_ABSENT

        with pytest.raises(cb.CertificationRefused) as caught:
            mw.build_manager_worlds(
                conn, planning_event=5, minutes_run_id=certified["minutes_v1"],
                xpts_run_id=certified["xpts_v1"], team_run_id=certified["team_strength_v1"],
                squad_ids=[1], simulations=4, certification=_unauthorised(artifact),
            )
        assert caught.value.token == cb.DIAG_CERTIFICATION_ARTIFACT_UNVALIDATED

        # A run id the artifact did not record for this event is refused, however
        # complete the world it names is.
        other = _alternative_world(conn, 5)
        with pytest.raises(cb.CertificationRefused) as caught:
            mw.build_manager_worlds(
                conn, planning_event=5, minutes_run_id=other["minutes_v1"],
                xpts_run_id=other["xpts_v1"], team_run_id=other["team_strength_v1"],
                squad_ids=[1], simulations=4, certification=artifact,
            )
        assert caught.value.token == cb.STATE_PREDICTIVE_BUNDLE_INCOHERENT

        # The CERTIFIED run ids load, so the refusals above are about provenance.
        built = mw.build_manager_worlds(
            conn, planning_event=5, minutes_run_id=certified["minutes_v1"],
            xpts_run_id=certified["xpts_v1"], team_run_id=certified["team_strength_v1"],
            squad_ids=[1], simulations=4, certification=artifact,
        )
        assert built["world_matrix"]["worlds"] == 4
    finally:
        conn.close()


def test_the_loader_issued_matrix_is_content_bound_not_stamp_bound(tmp_path):
    """The capability binds CONTENT: re-issuing the same worlds is admitted, a stamp
    copied onto different worlds is not -- and the stamp itself plays no part."""

    from test_route_optimizer import EVENTS

    event = int(EVENTS[0])
    conn, runs = _world(events=EVENTS)
    try:
        artifact = _artifact(conn, runs, events=EVENTS)
        certified = _self_consistent_bundle(runs[event], event=event)
        first, info = _certified_matrix_loads(conn, runs, artifact, tmp_path, event)
        identity = info["certified_bundle_identity"]

        # A second, INDEPENDENT load of the same certified world is issued too: the
        # same content, produced by the same loader, is the same authorisation.
        second, info2 = ro.build_event_worlds(
            conn, {event: certified}, event, (1, 2), _optimizer_config(events=(event,)),
            certification=artifact,
        )
        assert info2["certified_bundle_identity"] == identity
        assert ro.issued_world_matrix(second) is not None
        assert (
            ro.issued_world_matrix(second).content_identity
            == ro.issued_world_matrix(first).content_identity
        )
        assert ro.require_certified_prebuilt_matrix(
            second, event=event, certified_bundle_identity=str(identity)
        ).event == event

        # A matrix with no semantic blocks at all -- the shape a caller reaches for when
        # it only wants the stamp check to pass -- is not issued, and cannot be.
        assert ro.issued_world_matrix({"worlds": 8, "player_ids": [1, 2]}) is None
        assert ro.issued_world_matrix(
            _stamped_by_hand({"worlds": 8, "player_ids": [1, 2]}, identity)
        ) is None
    finally:
        conn.close()
