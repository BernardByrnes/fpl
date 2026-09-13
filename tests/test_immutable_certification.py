"""R4A.1: immutable execution snapshot + certified-decision enforcement.

Deterministic.  Uses temporary databases and a temporary snapshot directory; the
only live reads are read-only immutability checks.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from fpl_brain import causality as cx
from fpl_brain import certified_bundle as cb
from fpl_brain import execution_snapshot as es
from fpl_brain import four_gw_decision as fg
from fpl_brain import repositories as repo
from fpl_brain.database import connect_database

NOW = datetime(2026, 9, 13, 12, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# World builder
# ---------------------------------------------------------------------------


def _world(path: Path):
    conn = connect_database(path)
    with conn:
        conn.execute("INSERT INTO positions(id, singular_name_short, raw_json, updated_at)"
                     " VALUES (2,'DEF','{}','2026-09-01T00:00:00Z')")
        conn.execute("INSERT INTO teams(id, name, short_name, raw_json, updated_at)"
                     " VALUES (1,'One','ONE','{}','2026-09-01T00:00:00Z')")
        conn.execute("INSERT INTO teams(id, name, short_name, raw_json, updated_at)"
                     " VALUES (2,'Two','TWO','{}','2026-09-01T00:00:00Z')")
        conn.execute("INSERT INTO events(id, name, deadline_time, finished, raw_json, updated_at)"
                     " VALUES (5,'GW5','2026-09-18T17:30:00Z',0,'{}','2026-09-01T00:00:00Z')")
        conn.execute("INSERT INTO fixtures(id, event, team_h, team_a, kickoff_time, finished, started,"
                     " raw_json, updated_at)"
                     " VALUES (48,5,1,2,'2026-09-19T14:00:00Z',0,0,'{}','2026-09-01T00:00:00Z')")
        conn.execute("INSERT INTO fetch_runs(id, started_at, status, trigger)"
                     " VALUES (1,'2026-09-01T00:00:00Z','success','test')")
        conn.execute("INSERT INTO players(id, web_name, team_id, element_type, is_active, first_seen_at,"
                     " last_seen_at, raw_json, updated_at)"
                     " VALUES (1,'Alpha',1,2,1,'2026-09-01T00:00:00Z','2026-09-01T00:00:00Z','{}','2026-09-01T00:00:00Z')")
        conn.execute("INSERT INTO players(id, web_name, team_id, element_type, is_active, first_seen_at,"
                     " last_seen_at, raw_json, updated_at)"
                     " VALUES (2,'Beta',1,2,1,'2026-09-01T00:00:00Z','2026-09-01T00:00:00Z','{}','2026-09-01T00:00:00Z')")
        conn.execute("INSERT INTO player_snapshots(id, player_id, fetch_run_id, captured_at, now_cost, raw_json)"
                     " VALUES (1,1,1,'2026-09-01T00:00:00Z',50,'{}')")
        conn.execute("INSERT INTO scouting_imports(id, source_file, file_sha256, imported_at)"
                     " VALUES (1,'world','world-hash','2026-09-01T00:00:00Z')")
        conn.execute("INSERT INTO scouting_notes(id, import_id, player_id, key, category, value_text,"
                     " confidence, observed_at, expires_at, created_at)"
                     " VALUES (1,1,1,'rotation_risk','minutes','high','medium','2026-09-02T09:00:00Z',NULL,"
                     "'2026-09-02T09:00:00Z')".replace("'high','medium'", "'high','medium'"))
    return conn


def _capture(tmp_path, conn, cutoff="2026-09-12T19:00:00Z", clock=None):
    snap = es.capture_execution_snapshot(
        tmp_path / "live.db",
        directory=tmp_path / "snap",
        execution_run_uuid="run-1",
        planning_cutoff=cutoff,
        clock=clock or (lambda: NOW),
    )
    return snap


# ---------------------------------------------------------------------------
# A. one immutable snapshot
# ---------------------------------------------------------------------------


def test_a_certification_takes_one_immutable_db_snapshot(tmp_path):
    conn = _world(tmp_path / "live.db")
    snap = _capture(tmp_path, conn)
    assert Path(snap.path).exists()
    assert snap.size_bytes > 0
    assert len(snap.data_snapshot_sha256) == 64
    assert snap.source_db_identity["projection_runs_max_id"] in (0, None)
    assert Path(snap.manifest_path).exists()
    manifest = json.loads(Path(snap.manifest_path).read_text(encoding="utf-8"))
    assert manifest["data_snapshot_sha256"] == snap.data_snapshot_sha256
    conn.close()


def test_a_snapshot_is_read_only(tmp_path):
    conn = _world(tmp_path / "live.db")
    snap = _capture(tmp_path, conn)
    reader = es.open_snapshot(snap)
    try:
        reader.execute("SELECT COUNT(*) FROM fixtures").fetchone()
        with pytest.raises(sqlite3.OperationalError):
            reader.execute("CREATE TABLE nope (id INTEGER)")
    finally:
        reader.close()
        conn.close()


def test_a_snapshot_represents_a_captured_state_not_a_recomputed_hash(tmp_path):
    """The identity is of the snapshot FILE, so it is an actual captured state."""

    conn = _world(tmp_path / "live.db")
    snap = _capture(tmp_path, conn)
    assert es.file_sha256(snap.path) == snap.data_snapshot_sha256
    # Rewriting the snapshot changes its identity -> hard fail.
    with open(snap.path, "r+b") as handle:
        handle.seek(0)
        handle.write(b"\x00")
    with pytest.raises(es.SnapshotError) as caught:
        es.assert_snapshot_unchanged(snap)
    assert es.DIAG_SNAPSHOT_MUTATED in str(caught.value)
    conn.close()


def test_a_missing_snapshot_fails_closed(tmp_path):
    conn = _world(tmp_path / "live.db")
    snap = _capture(tmp_path, conn)
    Path(snap.path).unlink()
    with pytest.raises(es.SnapshotError) as caught:
        es.open_snapshot(snap)
    assert es.DIAG_SNAPSHOT_MISSING in str(caught.value)
    with pytest.raises(es.SnapshotError):
        es.load_snapshot_manifest(tmp_path / "no-such-dir")
    conn.close()


# ---------------------------------------------------------------------------
# B. all events share one snapshot identity
# ---------------------------------------------------------------------------


def test_b_all_events_read_the_same_snapshot_identity(tmp_path):
    conn = _world(tmp_path / "live.db")
    snap = _capture(tmp_path, conn)
    identities = set()
    for _event in (5, 6, 7, 8):
        reader = es.open_snapshot(snap)
        try:
            identities.add(es.file_sha256(snap.path))
            assert reader.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 1
        finally:
            reader.close()
    assert len(identities) == 1
    conn.close()


# ---------------------------------------------------------------------------
# C-F. later live mutations cannot change the prediction source
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mutation,probe",
    [
        ("UPDATE fixtures SET kickoff_time='2027-01-01T00:00:00Z' WHERE id=48",
         "SELECT kickoff_time FROM fixtures WHERE id=48"),
        ("UPDATE players SET team_id=2, is_active=0 WHERE id=1",
         "SELECT team_id, is_active FROM players WHERE id=1"),
        ("UPDATE player_snapshots SET now_cost=99 WHERE id=1",
         "SELECT now_cost FROM player_snapshots WHERE id=1"),
        ("UPDATE scouting_notes SET value_text='low' WHERE id=1",
         "SELECT value_text FROM scouting_notes WHERE id=1"),
    ],
    ids=["fixture", "bootstrap", "price", "scouting"],
)
def test_cdef_later_live_mutation_cannot_change_the_snapshot(tmp_path, mutation, probe):
    conn = _world(tmp_path / "live.db")
    snap = _capture(tmp_path, conn)
    reader = es.open_snapshot(snap)
    before = reader.execute(probe).fetchone()
    reader.close()

    with conn:  # mutate the LIVE database only
        conn.execute(mutation)
    es.assert_snapshot_unchanged(snap)  # the snapshot itself is untouched

    reader = es.open_snapshot(snap)
    try:
        after = reader.execute(probe).fetchone()
    finally:
        reader.close()
    assert tuple(before) == tuple(after), f"{mutation!r} must not reach the snapshot"

    # ...while the live row did change, and that is only an informational signal.
    live = conn.execute(probe).fetchone()
    assert tuple(live) != tuple(before)
    drift = es.live_source_drift(snap)
    assert drift["available"] is True and drift["informational"] is True
    conn.close()


# ---------------------------------------------------------------------------
# G. same-event acquisition temporal leak
# ---------------------------------------------------------------------------


def test_g_same_event_later_transfer_cannot_leak_into_earlier_cutoff(tmp_path):
    """Monday cutoff: A owned.  Wednesday (same GW): A sold, B bought."""

    conn = _world(tmp_path / "live.db")
    with conn:
        conn.execute(
            "INSERT INTO manager_player_acquisitions(id, entry_id, player_id, acquired_event,"
            " purchase_price, sold_event, source, created_at, updated_at)"
            " VALUES (1,241392,1,1,50,5,'manual','2026-09-07T10:00:00Z','2026-09-12T10:00:00Z')"
        )
        conn.execute(
            "INSERT INTO manager_player_acquisitions(id, entry_id, player_id, acquired_event,"
            " purchase_price, sold_event, source, created_at, updated_at)"
            " VALUES (2,241392,2,5,45,NULL,'manual','2026-09-12T10:00:00Z','2026-09-12T10:00:00Z')"
        )

    monday = "2026-09-14T09:00:00Z"      # before the Wednesday transfer (10:00)
    wednesday = "2026-09-16T11:00:00Z"   # after it

    # Timestamps are absent for the initial-squad rows, so a timestamp-as-of
    # reconstruction is REPORTED AS UNSUPPORTED rather than faked.
    coverage = repo.acquisition_timestamp_coverage(conn, 241392)
    assert coverage["timestamps_complete"] is False
    assert coverage["diagnostic"] == repo.DIAG_ACQUISITION_TIMESTAMPS_INCOMPLETE

    resolved = repo.active_manager_acquisitions_as_of_cutoff(conn, 241392, monday)
    assert resolved["basis"] == "unavailable"
    assert resolved["diagnostic"] == repo.DIAG_ACQUISITION_TIMESTAMPS_INCOMPLETE
    assert resolved["authoritative_alternative"]

    # The event-level resolver CANNOT distinguish the two: both are GW<=5.
    event_level = repo.active_manager_acquisitions_as_of(conn, 241392, 5)
    assert {row["player_id"] for row in event_level} == {2}, (
        "event-level logic returns the CURRENT owner (B) for a Monday cutoff -- the leak"
    )
    # The CURRENT ledger also shows only B.
    assert {row["player_id"] for row in repo.active_manager_acquisitions(conn, 241392)} == {2}

    # The authoritative mechanism for a cutoff-accurate decision is the immutable
    # snapshot / explicit manual manager state captured at the cutoff.  The capture
    # clock is after the Monday cutoff, satisfying planning_cutoff <= created_at.
    snap = _capture(
        tmp_path, conn, cutoff=monday,
        clock=lambda: datetime(2026, 9, 17, 12, 0, 0, tzinfo=timezone.utc),
    )
    # With complete timestamps the cutoff resolver DOES return A for Monday.
    with conn:
        conn.execute("UPDATE manager_player_acquisitions SET acquired_at='2026-09-07T10:00:00Z'"
                     " WHERE id=1")
        conn.execute("UPDATE manager_player_acquisitions SET sold_at='2026-09-16T10:00:00Z'"
                     " WHERE id=1")
        conn.execute("UPDATE manager_player_acquisitions SET acquired_at='2026-09-16T10:00:00Z'"
                     " WHERE id=2")
    assert repo.acquisition_timestamp_coverage(conn, 241392)["timestamps_complete"] is True
    monday_rows = repo.active_manager_acquisitions_as_of_cutoff(conn, 241392, monday)
    assert monday_rows["basis"] == "transaction_timestamps"
    assert {row["player_id"] for row in monday_rows["rows"]} == {1}, (
        "a Monday cutoff must return A, not the Wednesday buy"
    )
    wednesday_rows = repo.active_manager_acquisitions_as_of_cutoff(conn, 241392, wednesday)
    assert {row["player_id"] for row in wednesday_rows["rows"]} == {2}
    conn.close()


def test_g_snapshot_preserves_the_pre_transfer_ledger(tmp_path):
    """A snapshot taken at the Monday cutoff keeps the Monday ledger."""

    conn = _world(tmp_path / "live.db")
    with conn:
        conn.execute(
            "INSERT INTO manager_player_acquisitions(id, entry_id, player_id, acquired_event,"
            " purchase_price, sold_event, source, acquired_at, sold_at, created_at, updated_at)"
            " VALUES (1,241392,1,1,50,NULL,'manual','2026-09-07T10:00:00Z',NULL,"
            "'2026-09-07T10:00:00Z','2026-09-07T10:00:00Z')"
        )
    snap = _capture(tmp_path, conn)
    with conn:  # Wednesday: sell A, buy B
        conn.execute("UPDATE manager_player_acquisitions SET sold_event=5, sold_at='2026-09-16T10:00:00Z'"
                     " WHERE id=1")
        conn.execute(
            "INSERT INTO manager_player_acquisitions(id, entry_id, player_id, acquired_event,"
            " purchase_price, sold_event, source, acquired_at, sold_at, created_at, updated_at)"
            " VALUES (2,241392,2,5,45,NULL,'manual','2026-09-16T10:00:00Z',NULL,"
            "'2026-09-16T10:00:00Z','2026-09-16T10:00:00Z')"
        )
    reader = es.open_snapshot(snap)
    try:
        rows = list(reader.execute(
            "SELECT player_id, sold_event FROM manager_player_acquisitions ORDER BY id"))
        assert [(r["player_id"], r["sold_event"]) for r in rows] == [(1, None)], (
            "the snapshot must still show A owned and unsold"
        )
        # The cutoff resolver, run AGAINST THE SNAPSHOT, returns A for Monday.
        resolved = repo.active_manager_acquisitions_as_of_cutoff(reader, 241392, "2026-09-14T09:00:00Z")
        assert resolved["basis"] == "transaction_timestamps"
        assert {row["player_id"] for row in resolved["rows"]} == {1}
    finally:
        reader.close()
        conn.close()


# ---------------------------------------------------------------------------
# H-I. production decision runner enforcement
# ---------------------------------------------------------------------------


def _base_world_for_bundles(conn):
    with conn:
        conn.execute("INSERT INTO positions(id, singular_name_short, raw_json, updated_at)"
                     " VALUES (2,'DEF','{}','2026-09-01T00:00:00Z')")
        conn.execute("INSERT INTO teams(id, name, short_name, raw_json, updated_at)"
                     " VALUES (1,'One','ONE','{}','2026-09-01T00:00:00Z')")
        conn.execute("INSERT INTO teams(id, name, short_name, raw_json, updated_at)"
                     " VALUES (2,'Two','TWO','{}','2026-09-01T00:00:00Z')")
        conn.execute("INSERT INTO events(id, name, deadline_time, finished, raw_json, updated_at)"
                     " VALUES (5,'GW5','2026-09-18T17:30:00Z',0,'{}','2026-09-01T00:00:00Z')")
        conn.execute("INSERT INTO fixtures(id, event, team_h, team_a, kickoff_time, finished, started,"
                     " raw_json, updated_at)"
                     " VALUES (48,5,1,2,'2026-09-19T14:00:00Z',0,0,'{}','2026-09-01T00:00:00Z')")

        def run(run_id, family, version):
            conn.execute(
                "INSERT INTO projection_runs(id, model_family, model_version, generated_at,"
                " planning_event, data_cutoff, status, source_snapshot_sha256, planning_context_hash)"
                " VALUES (?,?,?,'2026-09-12T19:01:00Z',5,'2026-09-12T19:00:00Z','complete',"
                "'codehash','ctx')",
                (run_id, family, version),
            )

        run(1, "minutes_v1", "minutes_v1.6.0")
        run(2, "team_strength_v1", "team_strength_v1.0.0")
        run(3, "player_rates_v1", "player_rates_v1.0.0")
        run(4, "xpts_v1", "xpts_v1.4.1")
        run(5, "monte_carlo_v1", "mc_v1.3.0")
        run(6, "minutes_v1", "minutes_v1.6.0")
        run(7, "xpts_v1", "xpts_v1.4.1")
        conn.execute(
            "INSERT INTO player_fixture_xpts_projections(projection_run_id, player_id, fixture_id, event,"
            " team_id, opponent_id, position, minutes_run_id, team_run_id, rate_run_id, payload_json,"
            " model_version, scoring_rules_version, generated_at)"
            " VALUES (4,1,48,5,1,2,'MID',1,2,3,'{}','xpts_v1.4.1','v1','2026-09-12T19:01:00Z')"
        )
        conn.execute(
            "INSERT INTO player_fixture_xpts_projections(projection_run_id, player_id, fixture_id, event,"
            " team_id, opponent_id, position, minutes_run_id, team_run_id, rate_run_id, payload_json,"
            " model_version, scoring_rules_version, generated_at)"
            " VALUES (7,1,48,5,1,2,'MID',6,2,3,'{}','xpts_v1.4.1','v1','2026-09-12T19:01:00Z')"
        )
        conn.execute(
            "INSERT INTO monte_carlo_distributions(projection_run_id, player_id, fixture_id, event, team_id,"
            " opponent_id, position, xpts_run_id, minutes_run_id, team_run_id, rate_run_id, payload_json,"
            " model_version, generated_at)"
            " VALUES (5,1,48,5,1,2,'MID',4,1,2,3,'{}','mc_v1.3.0','2026-09-12T19:01:00Z')"
        )


def _artifact(**overrides):
    payload = {
        "schema": fg.CERTIFICATION_ARTIFACT_SCHEMA,
        "temporal_status": "CAUSAL",
        "dependency_validation": "COHERENT",
        "data_snapshot_sha256": "d" * 64,
        "code_snapshot_sha256": "codehash",
        # R4A.2 tightened the contract: the artifact must carry the computed
        # authorisation flag and the factual execution fields.
        "route_search_executed": False,
        "transfer_execution_performed": False,
        "decision_search_permitted": True,
        "certified_bundles": {
            "5": {
                "runs": {"minutes_v1": 1, "team_strength_v1": 2, "player_rates_v1": 3,
                         "xpts_v1": 4, "monte_carlo_v1": 5},
                "data_snapshot_sha256": "d" * 64,
            }
        },
    }
    payload.update(overrides)
    return payload


def test_h_decision_runner_refuses_missing_certification_artifact(tmp_path):
    missing = tmp_path / "nope.json"
    with pytest.raises(fg.DecisionCertificationRequired) as caught:
        fg.load_certification_artifact(missing)
    assert fg.DIAG_DECISION_CERTIFICATION_REQUIRED in str(caught.value)


def test_h_malformed_or_non_causal_artifact_is_refused(tmp_path):
    path = tmp_path / "art.json"
    for overrides in (
        {"schema": "wrong"},
        {"temporal_status": "TEMPORAL_PROVENANCE_SUPERSEDED"},
        {"dependency_validation": "INCOHERENT"},
        {"data_snapshot_sha256": None},
        {"certified_bundles": {}},
    ):
        path.write_text(json.dumps(_artifact(**overrides)), encoding="utf-8")
        with pytest.raises((fg.DecisionCertificationRequired, cb.BundleIncoherent)):
            fg.load_certification_artifact(path)
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(fg.DecisionCertificationRequired):
        fg.load_certification_artifact(path)


def test_i_decision_runner_refuses_a_mismatched_certified_bundle(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    _base_world_for_bundles(conn)
    path = tmp_path / "art.json"
    # The artifact names xpts run 7, whose minutes edge points at run 6 while the
    # bundle declares run 1.
    path.write_text(json.dumps(_artifact()), encoding="utf-8")
    bad = json.loads(path.read_text(encoding="utf-8"))
    bad["certified_bundles"]["5"]["runs"]["xpts_v1"] = 7
    path.write_text(json.dumps(bad), encoding="utf-8")
    artifact = fg.load_certification_artifact(path)
    with pytest.raises(cb.BundleIncoherent) as caught:
        fg.event_support_from_certification(conn, artifact, events=[5], cutoff="2026-09-12T19:00:00Z")
    assert cb.DIAG_PREDICTIVE_BUNDLE_INCOHERENT in str(caught.value)
    conn.close()


def test_i_valid_artifact_yields_exact_certified_run_ids(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    _base_world_for_bundles(conn)
    path = tmp_path / "art.json"
    path.write_text(json.dumps(_artifact()), encoding="utf-8")
    artifact = fg.load_certification_artifact(path)
    support = fg.event_support_from_certification(conn, artifact, events=[5], cutoff="2026-09-12T19:00:00Z")
    assert support[5]["matched_runs"]["xpts_v1"] == 4, "the newer same-cutoff run 7 must be ignored"
    assert support[5]["bundle_identity"].startswith("sha256:")
    conn.close()


def test_i_artifact_missing_an_event_is_refused(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    _base_world_for_bundles(conn)
    path = tmp_path / "art.json"
    path.write_text(json.dumps(_artifact()), encoding="utf-8")
    artifact = fg.load_certification_artifact(path)
    with pytest.raises(fg.DecisionCertificationRequired, match="covers no bundle"):
        fg.event_support_from_certification(conn, artifact, events=[5, 6], cutoff="2026-09-12T19:00:00Z")
    conn.close()


# ---------------------------------------------------------------------------
# J. the production decision path does not use the legacy rediscovery function
# ---------------------------------------------------------------------------


def test_j_production_decision_path_does_not_use_legacy_discovery():
    source = Path("scripts/run_four_gw_decision.py").read_text(encoding="utf-8")
    # The search branch must use the certified support.
    assert "certified_support[int(event)]" in source
    # The legacy call survives only for the readiness display, and is labelled.
    assert "NON-PRODUCTION" in source
    legacy_line = [
        line for line in source.splitlines() if "event_support_from_db(" in line and "fg." in line
    ]
    assert legacy_line, "the legacy readiness call should still exist for diagnostics"
    # It must appear BEFORE the search branch, i.e. only in the readiness stage.
    legacy_index = source.index(legacy_line[0])
    search_index = source.index("certified_support[int(event)]")
    assert legacy_index < search_index, "legacy discovery must not feed the search stage"
    # And the legacy function itself is documented as non-production.
    fg_source = Path("fpl_brain/four_gw_decision.py").read_text(encoding="utf-8")
    assert "NON-PRODUCTION (diagnostics and tests only)" in fg_source


# ---------------------------------------------------------------------------
# K. newer same-cutoff family run cannot replace certified ids
# ---------------------------------------------------------------------------


def test_k_newer_same_cutoff_run_cannot_replace_certified_ids(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    _base_world_for_bundles(conn)
    artifact = _artifact()
    support = fg.event_support_from_certification(
        conn, artifact, events=[5], cutoff="2026-09-12T19:00:00Z"
    )
    assert support[5]["matched_runs"] == {
        "minutes_v1": 1, "team_strength_v1": 2, "player_rates_v1": 3, "xpts_v1": 4, "monte_carlo_v1": 5
    }
    # The legacy path WOULD have picked run 7 for xpts.
    legacy = fg.event_support_from_db(conn, [5], "2026-09-12T19:00:00Z")
    assert legacy[5]["matched_runs"]["xpts_v1"] == 7
    conn.close()


# ---------------------------------------------------------------------------
# L. runs 1-212 immutable
# ---------------------------------------------------------------------------


def test_l_runs_1_to_212_remain_immutable():
    from fpl_brain.config import config_path, load_config

    try:
        config = load_config(None)
    except Exception:
        pytest.skip("no project config available")
    db_path = config_path(config, "database")
    if not db_path.exists():
        pytest.skip("project database not present")
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        total, max_id = conn.execute("SELECT COUNT(*), MAX(id) FROM projection_runs").fetchone()
        non_complete = conn.execute(
            "SELECT COUNT(*) FROM projection_runs WHERE status!='complete'"
        ).fetchone()[0]
        run71 = conn.execute(
            "SELECT model_version, config_hash FROM projection_runs WHERE id=71"
        ).fetchone()
    finally:
        conn.close()
    assert total >= 212 and max_id >= 212
    assert non_complete == 0
    assert run71[0] == "mc_v1.2.1"
    assert run71[1] == "sha256:fedf5f6e50b4826b6e44b42d524cac2d682f15e878a5c86f7bab08a3f4e15042"


def test_snapshot_capture_refuses_a_cutoff_after_the_capture(tmp_path):
    conn = _world(tmp_path / "live.db")
    with pytest.raises(cx.PlanningCutoffInFuture):
        es.capture_execution_snapshot(
            tmp_path / "live.db",
            directory=tmp_path / "snap",
            planning_cutoff="2027-01-01T00:00:00Z",
            clock=lambda: NOW,
        )
    conn.close()
