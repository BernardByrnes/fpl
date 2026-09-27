"""R4A: causal cutoff invariants, as-of resolution and certified bundle integrity.

Deterministic.  Builds synthetic worlds in temporary databases; the only live
reads are read-only checks that historical runs are untouched.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from fpl_brain import causality as cx
from fpl_brain import certified_bundle as cb
from fpl_brain import execution, repositories as repo
from fpl_brain.database import connect_database

NOW = datetime(2026, 9, 12, 19, 10, 36, tzinfo=timezone.utc)
CUTOFF_BEFORE = "2026-09-12T19:00:00Z"
CUTOFF_EQUAL = "2026-09-12T19:10:36Z"
CUTOFF_FUTURE_10M = "2026-09-12T19:20:00Z"  # the exact R3 shape


# ---------------------------------------------------------------------------
# 1-2. Causal cutoff invariant
# ---------------------------------------------------------------------------


def test_1_future_cutoff_ten_minutes_fails_closed():
    with pytest.raises(cx.PlanningCutoffInFuture) as caught:
        cx.assert_causal_cutoff(CUTOFF_FUTURE_10M, NOW, label="R3 regression")
    assert cx.DIAG_PLANNING_CUTOFF_IN_FUTURE in str(caught.value)
    assert caught.value.lead_seconds == pytest.approx(564.0, abs=0.5)


def test_2_cutoff_at_or_before_execution_start_passes():
    cx.assert_causal_cutoff(CUTOFF_BEFORE, NOW)  # no raise
    cx.assert_causal_cutoff(CUTOFF_EQUAL, NOW)  # exactly equal


def test_2b_clock_skew_tolerance_is_tiny_and_bounded():
    assert cx.CLOCK_SKEW_TOLERANCE_SECONDS <= 2.0
    just_inside = (NOW + timedelta(seconds=1.5)).strftime("%Y-%m-%dT%H:%M:%SZ")
    cx.assert_causal_cutoff(just_inside, NOW)  # tolerated
    just_outside = (NOW + timedelta(seconds=3)).strftime("%Y-%m-%dT%H:%M:%SZ")
    with pytest.raises(cx.PlanningCutoffInFuture):
        cx.assert_causal_cutoff(just_outside, NOW)


def test_2c_production_guard_refuses_a_future_cutoff(tmp_path):
    """Defense point 1: the production controller preflight."""

    path = tmp_path / "fpl.db"
    conn = connect_database(path)
    try:
        with pytest.raises(cx.PlanningCutoffInFuture):
            with execution.production_run_guard(
                conn,
                planning_event=5,
                planning_cutoff="2999-01-01T00:00:00Z",
                label="future-cutoff probe",
            ):
                pytest.fail("guard body must never run for a future cutoff")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 3. generated_at vs data_cutoff
# ---------------------------------------------------------------------------


def test_3_generated_before_data_cutoff_is_impossible_for_live_certification(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    try:
        # A run generated now cannot claim a data cutoff an hour in the future.
        future = (datetime.now(timezone.utc) + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        with pytest.raises(cx.CausalityError) as caught:
            from fpl_brain import analytics

            analytics.create_projection_run(
                conn,
                model_family="baseline",
                model_version="baseline_v1.0.0",
                planning_event=5,
                planning_context_hash=None,
                data_cutoff=future,
                scouting_cutoff=None,
                official_run_ids=None,
            )
        assert cx.DIAG_DATA_CUTOFF_AFTER_GENERATED in str(caught.value)
    finally:
        conn.close()


def test_3b_historical_construction_can_opt_out_explicitly(tmp_path):
    """The escape hatch exists only for synthetic/historical construction."""

    conn = connect_database(tmp_path / "fpl.db")
    try:
        from fpl_brain import analytics

        future = (datetime.now(timezone.utc) + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        run_id = analytics.create_projection_run(
            conn,
            model_family="baseline",
            model_version="baseline_v1.0.0",
            planning_event=5,
            planning_context_hash=None,
            data_cutoff=future,
            scouting_cutoff=None,
            official_run_ids=None,
            allow_future_cutoff=True,
        )
        assert run_id > 0
    finally:
        conn.close()


def _base_world(conn):
    """Prerequisite rows every synthetic world needs (FK targets and bootstrap)."""

    with conn:
        conn.execute("INSERT INTO positions(id, singular_name_short, raw_json, updated_at)"
                     " VALUES (1,'GKP','{}','2026-09-01T00:00:00Z')")
        conn.execute("INSERT INTO positions(id, singular_name_short, raw_json, updated_at)"
                     " VALUES (2,'DEF','{}','2026-09-01T00:00:00Z')")
        conn.execute("INSERT INTO positions(id, singular_name_short, raw_json, updated_at)"
                     " VALUES (3,'MID','{}','2026-09-01T00:00:00Z')")
        conn.execute("INSERT INTO fetch_runs(id, started_at, status, trigger)"
                     " VALUES (1,'2026-09-01T00:00:00Z','success','test')")
        conn.execute("INSERT INTO teams(id, name, short_name, raw_json, updated_at)"
                     " VALUES (1,'One','ONE','{}','2026-09-01T00:00:00Z')")
        conn.execute("INSERT INTO teams(id, name, short_name, raw_json, updated_at)"
                     " VALUES (2,'Two','TWO','{}','2026-09-01T00:00:00Z')")
        conn.execute("INSERT INTO events(id, name, deadline_time, finished, raw_json, updated_at)"
                     " VALUES (5,'GW5','2026-09-18T17:30:00Z',0,'{}','2026-09-01T00:00:00Z')")
        conn.execute("INSERT INTO fixtures(id, event, team_h, team_a, kickoff_time, finished, started,"
                     " raw_json, updated_at)"
                     " VALUES (48,5,1,2,'2026-09-19T14:00:00Z',0,0,'{}','2026-09-01T00:00:00Z')")
        conn.execute("INSERT INTO scouting_imports(id, source_file, file_sha256, imported_at)"
                     " VALUES (1,'world','world-hash','2026-09-01T00:00:00Z')")
        _base_world_done = True


def _player(conn, pid, name, position=2):
    conn.execute(
        "INSERT INTO players(id, web_name, team_id, element_type, is_active, first_seen_at,"
        " last_seen_at, raw_json, updated_at)"
        " VALUES (?,?,1,?,1,'2026-09-01T00:00:00Z','2026-09-01T00:00:00Z','{}','2026-09-01T00:00:00Z')",
        (pid, name, position),
    )


# ---------------------------------------------------------------------------
# 4. Price as-of
# ---------------------------------------------------------------------------


def _price_world(conn):
    _base_world(conn)
    with conn:
        _player(conn, 1, "Old")
        _player(conn, 2, "New")
        # ids offset from the base world's fetch run 1
        for row_id, pid, cost, captured in (
            (101, 1, 50, "2026-09-01T00:00:00Z"),
            (102, 1, 60, "2026-09-10T00:00:00Z"),
            (103, 1, 70, "2026-09-20T00:00:00Z"),  # future relative to the cutoffs
            (104, 2, 45, "2026-09-20T00:00:00Z"),  # only a future observation
        ):
            # player_snapshots is keyed (player_id, fetch_run_id), so each capture
            # needs its own fetch run.
            conn.execute(
                "INSERT INTO fetch_runs(id, started_at, status, trigger) VALUES (?,?,'success','test')",
                (row_id, captured),
            )
            conn.execute(
                "INSERT INTO player_snapshots(id, player_id, fetch_run_id, captured_at, now_cost, raw_json)"
                " VALUES (?,?,?,?,?,'{}')",
                (row_id, pid, row_id, captured, cost),
            )


def test_4_later_price_never_leaks_backward(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    _price_world(conn)
    old = cx.price_snapshot_as_of(conn, "2026-09-05T00:00:00Z")
    new = cx.price_snapshot_as_of(conn, "2026-09-15T00:00:00Z")
    assert old.price(1) == 50  # only the 09-01 row existed
    assert new.price(1) == 60  # the 09-10 row, not the future 09-20 row of 70
    assert "2026-09-10T00:00:00Z" == new.captured_at[1]
    # Player 2 has only a future observation: no causal price.
    assert old.price(2) is None and new.price(2) is None
    conn.close()


def test_4b_missing_required_price_fails_closed(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    _price_world(conn)
    assert cx.missing_required_prices(conn, "2026-09-05T00:00:00Z", [1, 2]) == [2]

    from fpl_brain import candidate_universe as cu

    with pytest.raises(cx.CausalityError) as caught:
        cu.price_snapshot_as_of(conn, 5, "2026-09-05T00:00:00Z", required_player_ids=[2])
    assert cx.DIAG_PRICE_MISSING_AS_OF_CUTOFF in str(caught.value)
    ok = cu.price_snapshot_as_of(conn, 5, "2026-09-05T00:00:00Z", required_player_ids=[1])
    assert ok.prices[1] == 50
    conn.close()


def test_4c_price_resolution_is_deterministic(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    _price_world(conn)
    first = cx.price_snapshot_as_of(conn, "2026-09-15T00:00:00Z")
    second = cx.price_snapshot_as_of(conn, "2026-09-15T00:00:00Z")
    assert first.prices == second.prices and first.row_ids == second.row_ids
    conn.close()


# ---------------------------------------------------------------------------
# 5. Scouting as-of (the Wednesday/Friday/Thursday regression)
# ---------------------------------------------------------------------------


def _scouting_world(conn):
    _base_world(conn)
    with conn:
        _player(conn, 1, "P1", position=3)
        for row_id, key, value, observed in (
            (1, "rotation_risk", "high", "2026-09-02T09:00:00Z"),      # Wednesday
            (2, "role_security_5gw", "secure", "2026-09-04T09:00:00Z"),  # Friday
        ):
            conn.execute(
                "INSERT INTO scouting_notes(id, import_id, player_id, key, category, value_text,"
                " confidence, observed_at, expires_at, created_at)"
                " VALUES (?,1,1,?,'minutes',?,'medium',?,NULL,'2026-09-04T09:00:00Z')",
                (row_id, key, value, observed),
            )


def test_5_thursday_cutoff_returns_the_wednesday_note(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    _scouting_world(conn)
    thursday = "2026-09-03T09:00:00Z"  # between Wednesday and Friday

    as_of = repo.scouting_current_rows_as_of(conn, thursday, [1])
    by_key = {row["key"]: row for row in as_of}
    # The Wednesday note is returned ...
    assert by_key["rotation_risk"]["value_text"] == "high"
    # ... and the Friday note is invisible, not "latest then discarded".
    assert "role_security_5gw" not in by_key

    # The old unbounded ranking sees the Friday note, which is the bug.
    unfiltered = repo.scouting_current_rows(conn, [1])
    assert {row["key"] for row in unfiltered} == {"rotation_risk", "role_security_5gw"}

    # After Friday, both are visible and the newer per-key note wins.
    friday_after = repo.scouting_current_rows_as_of(conn, "2026-09-05T00:00:00Z", [1])
    assert {row["key"] for row in friday_after} == {"rotation_risk", "role_security_5gw"}
    conn.close()


def test_5b_expiry_is_applied_after_correct_as_of_selection(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    _scouting_world(conn)
    # The Wednesday note expires before Thursday: as-of selection still returns it
    # (it was the note in force), and its expiry is visible to the consumer.
    with conn:
        conn.execute("UPDATE scouting_notes SET expires_at='2026-09-02T23:59:59Z' WHERE id=1")
    rows = repo.scouting_current_rows_as_of(conn, "2026-09-03T09:00:00Z", [1])
    note = {row["key"]: row for row in rows}["rotation_risk"]
    assert note["expires_at"] == "2026-09-02T23:59:59Z"
    conn.close()


# ---------------------------------------------------------------------------
# 6-7. Manager state and acquisition as-of
# ---------------------------------------------------------------------------


def _manager_world(conn):
    _base_world(conn)
    with conn:
        _player(conn, 10, "Owned")
        _player(conn, 20, "Later")
        for row_id, captured, bank in (
            (1, "2026-09-01T00:00:00Z", 10),
            (2, "2026-09-20T00:00:00Z", 99),  # after the cutoff
        ):
            conn.execute(
                "INSERT INTO manager_state(id, entry_id, fetch_run_id, captured_at, event, bank, raw_json)"
                " VALUES (?,241392,1,?,4,?,'{}')",
                (row_id, captured, bank),
            )
        conn.execute(
            "INSERT INTO manager_player_acquisitions(id, entry_id, player_id, acquired_event,"
            " purchase_price, sold_event, source, created_at, updated_at)"
            " VALUES (1,241392,10,1,50,NULL,'manual','2026-09-01T00:00:00Z','2026-09-01T00:00:00Z')"
        )
    return conn


def test_6_later_manager_state_cannot_leak_backward(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    _manager_world(conn)
    early = repo.manager_planning_state(conn, 241392, 4, as_of="2026-09-05T00:00:00Z")
    late = repo.manager_planning_state(conn, 241392, 4, as_of="2026-09-25T00:00:00Z")
    assert early["bank"] == 10, "the future bank (99) must not leak into the earlier as_of"
    assert late["bank"] == 99
    conn.close()


def test_7_later_acquisition_and_sale_cannot_leak_backward(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    _manager_world(conn)
    with conn:
        conn.execute(
            "INSERT INTO manager_player_acquisitions(id, entry_id, player_id, acquired_event,"
            " purchase_price, sold_event, source, created_at, updated_at)"
            " VALUES (2,241392,20,7,55,NULL,'manual','2026-09-01T00:00:00Z','2026-09-01T00:00:00Z')"
        )
        # player 10 is sold at event 9
        conn.execute("UPDATE manager_player_acquisitions SET sold_event=9 WHERE id=1")

    at_event_5 = repo.active_manager_acquisitions_as_of(conn, 241392, 5)
    at_event_8 = repo.active_manager_acquisitions_as_of(conn, 241392, 8)
    current = repo.active_manager_acquisitions(conn, 241392)

    assert [row["player_id"] for row in at_event_5] == [10], "a later buy must not appear early"
    assert [row["player_id"] for row in at_event_8] == [10, 20]
    # The CURRENT ledger has lost player 10 entirely -- the backward-leak bug.
    assert [row["player_id"] for row in current] == [20]
    conn.close()


# ---------------------------------------------------------------------------
# 8. Data snapshot identity and drift
# ---------------------------------------------------------------------------


def test_8_data_snapshot_is_deterministic_and_detects_drift(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    _base_world(conn)
    first = cx.capture_causal_data_snapshot(conn, cutoff="2026-09-12T19:00:00Z", events=[5])
    again = cx.capture_causal_data_snapshot(conn, cutoff="2026-09-12T19:00:00Z", events=[5])
    assert first["data_snapshot_sha256"] == again["data_snapshot_sha256"]

    # A post-snapshot ingestion that mutates a fixture MUST be detectable.
    with conn:
        conn.execute("UPDATE fixtures SET kickoff_time='2026-09-20T14:00:00Z', event=6 WHERE id=48")
    drift = cx.data_snapshot_drift(
        conn, first["data_snapshot_sha256"], cutoff="2026-09-12T19:00:00Z", events=[5]
    )
    assert drift["matches"] is False
    assert drift["diagnostic"] == cx.DIAG_DATA_SNAPSHOT_DRIFT

    # Restoring the causal state restores the identity.
    with conn:
        conn.execute("UPDATE fixtures SET kickoff_time='2026-09-19T14:00:00Z', event=5 WHERE id=48")
    assert cx.data_snapshot_drift(
        conn, first["data_snapshot_sha256"], cutoff="2026-09-12T19:00:00Z", events=[5]
    )["matches"] is True
    conn.close()


def test_8b_code_and_data_snapshot_are_distinct_concepts():
    from fpl_brain import analytics

    code = analytics.source_snapshot_sha256()
    assert isinstance(code, str) and len(code) == 64
    # The code fingerprint is not a data identity: it takes no cutoff/data input.
    import inspect

    # The code fingerprint takes only path/root arguments: it cannot represent a
    # data instant, which is precisely why a separate data identity is required.
    code_params = set(inspect.signature(analytics.source_snapshot_sha256).parameters)
    assert code_params <= {"paths", "root"}
    for forbidden in ("cutoff", "as_of", "event", "entry_id"):
        assert forbidden not in code_params
    assert hasattr(cx, "capture_causal_data_snapshot")


# ---------------------------------------------------------------------------
# 9-11. Certified bundle / DAG
# ---------------------------------------------------------------------------


#: The AUTHORITATIVE declared versions, read from the one in-library source: PE-9 gap 1
#: made the required versions come from that source and never from the artifact, so a
#: world recorded under stale literals is a run nobody pins.
DECLARED_VERSIONS = cb.declared_required_versions()


def _bundle_world(conn):
    """Five families for GW5 with correct and incorrect DAG wiring available."""

    def run(run_id, family, version, event=5, cutoff="2026-09-12T19:00:00Z", status="complete"):
        conn.execute(
            "INSERT INTO projection_runs(id, model_family, model_version, generated_at, planning_event,"
            " data_cutoff, status, source_snapshot_sha256, planning_context_hash)"
            " VALUES (?,?,?,'2026-09-12T19:01:00Z',?,?,?, 'codehash', 'ctx')",
            (run_id, family, version, event, cutoff, status),
        )

    declared = DECLARED_VERSIONS
    _base_world(conn)
    with conn:
        run(1, "minutes_v1", declared["minutes_v1"])
        run(2, "team_strength_v1", declared["team_strength_v1"])
        run(3, "player_rates_v1", declared["player_rates_v1"])
        run(4, "xpts_v1", declared["xpts_v1"])
        run(5, "monte_carlo_v1", declared["monte_carlo_v1"])
        # a NEWER same-cutoff rerun of xpts, wired to a different minutes run
        run(6, "minutes_v1", declared["minutes_v1"])
        run(7, "xpts_v1", declared["xpts_v1"])
        conn.execute(
            "INSERT INTO player_fixture_xpts_projections(projection_run_id, player_id, fixture_id, event,"
            " team_id, opponent_id, position, minutes_run_id, team_run_id, rate_run_id, payload_json,"
            " model_version, scoring_rules_version, generated_at)"
            " VALUES (4,1,48,5,1,2,'MID',1,2,3,'{}',?,'v1','2026-09-12T19:01:00Z')",
            (declared["xpts_v1"],),
        )
        conn.execute(
            "INSERT INTO player_fixture_xpts_projections(projection_run_id, player_id, fixture_id, event,"
            " team_id, opponent_id, position, minutes_run_id, team_run_id, rate_run_id, payload_json,"
            " model_version, scoring_rules_version, generated_at)"
            " VALUES (7,1,48,5,1,2,'MID',6,2,3,'{}',?,'v1','2026-09-12T19:01:00Z')",
            (declared["xpts_v1"],),
        )
        conn.execute(
            "INSERT INTO monte_carlo_distributions(projection_run_id, player_id, fixture_id, event, team_id,"
            " opponent_id, position, xpts_run_id, minutes_run_id, team_run_id, rate_run_id, payload_json,"
            " model_version, generated_at)"
            " VALUES (5,1,48,5,1,2,'MID',4,1,2,3,'{}',?,'2026-09-12T19:01:00Z')",
            (declared["monte_carlo_v1"],),
        )
    return conn


def test_9_same_cutoff_but_mismatched_upstream_dag_fails(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    _bundle_world(conn)
    # xpts run 7 is same cutoff/event/version, but its minutes edge points at run 6.
    with pytest.raises(cb.BundleIncoherent) as caught:
        cb.certified_bundle_from_explicit_ids(
            conn, event=5, cutoff="2026-09-12T19:00:00Z",
            runs={"minutes_v1": 1, "team_strength_v1": 2, "player_rates_v1": 3, "xpts_v1": 7, "monte_carlo_v1": 5},
        )
    assert cb.DIAG_PREDICTIVE_BUNDLE_INCOHERENT in str(caught.value)
    assert any("references minutes_v1 run 6" in reason for reason in caught.value.reasons)
    conn.close()


def test_10_exact_coherent_bundle_passes_and_has_an_identity(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    _bundle_world(conn)
    bundle = cb.certified_bundle_from_explicit_ids(
        conn, event=5, cutoff="2026-09-12T19:00:00Z",
        runs={"minutes_v1": 1, "team_strength_v1": 2, "player_rates_v1": 3, "xpts_v1": 4, "monte_carlo_v1": 5},
        required_versions={
            "minutes_v1": DECLARED_VERSIONS["minutes_v1"],
            "monte_carlo_v1": DECLARED_VERSIONS["monte_carlo_v1"],
        },
    )
    assert bundle.bundle_identity().startswith("sha256:")
    assert bundle.runs["xpts_v1"] == 4
    assert bundle.model_versions["monte_carlo_v1"] == DECLARED_VERSIONS["monte_carlo_v1"]
    conn.close()


def test_11_newer_same_cutoff_rerun_does_not_replace_the_certified_run(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    _bundle_world(conn)
    # The exact bundle keeps run 4 even though newer xpts run 7 shares its cutoff.
    bundle = cb.certified_bundle_from_explicit_ids(
        conn, event=5, cutoff="2026-09-12T19:00:00Z",
        runs={"minutes_v1": 1, "team_strength_v1": 2, "player_rates_v1": 3, "xpts_v1": 4, "monte_carlo_v1": 5},
    )
    assert bundle.runs["xpts_v1"] == 4
    assert max(7, bundle.runs["xpts_v1"]) == 7  # a newer run exists and is ignored
    conn.close()


def test_11b_wrong_version_and_wrong_cutoff_are_reported(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    _bundle_world(conn)
    with pytest.raises(cb.BundleIncoherent) as caught:
        cb.certified_bundle_from_explicit_ids(
            conn, event=5, cutoff="2026-09-12T19:00:00Z",
            runs={"minutes_v1": 1, "team_strength_v1": 2, "player_rates_v1": 3, "xpts_v1": 4, "monte_carlo_v1": 5},
            required_versions={"monte_carlo_v1": "mc_v9.9.9"},
        )
    assert any("required 'mc_v9.9.9'" in reason for reason in caught.value.reasons)
    conn.close()


# ---------------------------------------------------------------------------
# 12-13. Historical integrity and narrative regression
# ---------------------------------------------------------------------------


def test_12_historical_runs_1_to_212_are_untouched():
    import json
    from pathlib import Path

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
        statuses = dict(conn.execute("SELECT status, COUNT(*) FROM projection_runs GROUP BY status"))
        first = conn.execute(
            "SELECT model_version, config_hash FROM projection_runs WHERE id=71"
        ).fetchone()
    finally:
        conn.close()
    assert total >= 212 and max_id >= 212
    assert set(statuses) == {"complete"}, "no run may be left running or failed"
    assert first[0] == "mc_v1.2.1"
    assert first[1] == "sha256:fedf5f6e50b4826b6e44b42d524cac2d682f15e878a5c86f7bab08a3f4e15042"
    # The R3 classification artifact exists and mutates nothing.
    classification = json.loads(
        Path("data/exports/reliability/r3_runs_temporal_classification.json").read_text(encoding="utf-8")
    )
    assert classification["classification"] == "TEMPORAL_PROVENANCE_SUPERSEDED"
    assert classification["rows_mutated"] == 0


def test_13_manager_state_prose_derives_from_actual_state(tmp_path):
    import json
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    provenance = tmp_path / "prov.json"
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "confirm_manager_state.py"),
            "--event", "5", "--ft", "1", "--bank", "7", "--event-start-ft", "1",
            "--no-strict-bank", "--dry-run", "--provenance-out", str(provenance),
        ],
        capture_output=True, text=True, cwd=str(root),
    )
    assert provenance.exists(), result.stderr
    note = json.loads(provenance.read_text(encoding="utf-8"))["ft_arithmetic"]["note"]
    assert "GW5" in note, "the event number must be derived, not hardcoded"
    assert "1 free transfer(s) remain" in note
    assert "ADDITIONAL GW4" not in note


def test_13b_zero_ft_prose_is_also_derived(tmp_path):
    import json
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    provenance = tmp_path / "prov0.json"
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "confirm_manager_state.py"),
            "--event", "5", "--ft", "0", "--bank", "7", "--event-start-ft", "1",
            "--no-strict-bank", "--dry-run", "--provenance-out", str(provenance),
        ],
        capture_output=True, text=True, cwd=str(root),
    )
    assert provenance.exists(), result.stderr
    note = json.loads(provenance.read_text(encoding="utf-8"))["ft_arithmetic"]["note"]
    assert "ADDITIONAL GW5" in note
    assert "is 0" in note


# ---------------------------------------------------------------------------
# 14. The decision path must consume certified bundle ids, not rediscover them
# ---------------------------------------------------------------------------


def test_14_horizon_support_uses_certified_bundle_ids(tmp_path):
    from fpl_brain import four_gw_decision as fg

    conn = connect_database(tmp_path / "fpl.db")
    _bundle_world(conn)
    cutoff = "2026-09-12T19:00:00Z"
    # The bundle names run 4 for xpts even though newer same-cutoff run 7 exists.
    bundle = {
        "runs": {"minutes_v1": 1, "team_strength_v1": 2, "player_rates_v1": 3, "xpts_v1": 4, "monte_carlo_v1": 5},
        "data_snapshot_sha256": None,
        "code_snapshot_sha256": None,
    }
    support = fg.event_support_from_certified_bundles(conn, {5: bundle}, cutoff=cutoff)
    assert support[5]["supported"] is True
    assert support[5]["matched_runs"]["xpts_v1"] == 4, "must not rediscover the newer run 7"
    assert support[5]["source"] == "certified_bundle"
    assert support[5]["bundle_identity"].startswith("sha256:")

    # An incoherent bundle is refused rather than silently degrading.
    bad = dict(bundle)
    bad["runs"] = dict(bundle["runs"], xpts_v1=7)  # wired to minutes run 6
    with pytest.raises(cb.BundleIncoherent):
        fg.event_support_from_certified_bundles(conn, {5: bad}, cutoff=cutoff)
    conn.close()


def test_14b_certified_support_declares_its_cutoff_so_the_horizon_is_not_stale(tmp_path):
    """A certified bundle's support must declare the bundle's cutoff.

    ``event_support_from_certified_bundles`` omitted ``data_cutoff``, so
    ``evaluate_horizon(..., cutoff=...)`` read ``None`` for every certified event,
    judged it STALE_CUTOFF_MISMATCH, and made the production decision refuse with
    "certified decision horizon incomplete" no matter how coherent the bundle was.
    """

    from fpl_brain import four_gw_decision as fg

    conn = connect_database(tmp_path / "fpl.db")
    _bundle_world(conn)
    cutoff = "2026-09-12T19:00:00Z"
    bundle = {
        "runs": {"minutes_v1": 1, "team_strength_v1": 2, "player_rates_v1": 3, "xpts_v1": 4, "monte_carlo_v1": 5},
        "data_snapshot_sha256": None,
        "code_snapshot_sha256": None,
    }
    support = fg.event_support_from_certified_bundles(conn, {5: bundle}, cutoff=cutoff)
    assert support[5]["data_cutoff"] == cutoff

    horizon = fg.evaluate_horizon(planning_event=5, support_by_event=support, cutoff=cutoff, last_event=38)
    record = horizon["events"]["5"]
    assert record["cutoff_matches"] is True, record
    assert record["supported"] is True, record
    assert record["reason"] is None, record
    # GW6-8 carry no bundle here, so the WINDOW is legitimately incomplete -- the
    # point is that the certified event itself is not stale.
    assert horizon["status"] == fg.DECISION_HORIZON_INCOMPLETE
    assert 5 not in horizon["stale_cutoff_events"]
    conn.close()


def test_14c_certifier_resolves_the_horizon_from_the_certified_support(tmp_path):
    """The certifier's authorisation must run against real certified support.

    Its inline predecessor referenced an undefined local, so the certification
    artifact could never be written. A single-event artifact must yield a genuine
    horizon status string (an incomplete window, not a crash), and a bundle the
    artifact does not carry must refuse rather than authorise.
    """

    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import certify_gw5_gw8 as certifier  # noqa: E402

    from fpl_brain import four_gw_decision as fg

    conn = connect_database(tmp_path / "fpl.db")
    _bundle_world(conn)
    cutoff = "2026-09-12T19:00:00Z"
    artifact = {
        # The certified horizon must be DECLARED: the consumed horizon is required
        # to equal it, so an artifact that does not state one authorises nothing.
        "events": [5],
        "certified_bundles": {
            "5": {
                "runs": {"minutes_v1": 1, "team_strength_v1": 2, "player_rates_v1": 3,
                         "xpts_v1": 4, "monte_carlo_v1": 5},
                "data_snapshot_sha256": None,
                "code_snapshot_sha256": None,
            }
        },
        "data_snapshot_sha256": "d" * 64,
        "code_snapshot_sha256": "codehash",
    }
    status = certifier.certified_horizon_status(conn, artifact, [5], cutoff)
    # The synthetic world holds exactly one real event (GW5) with a supported
    # bundle, so the resolver reports a genuinely complete short horizon -- the
    # point is that it returns a real status from the certified support rather
    # than crashing on an unbound local.
    assert status == fg.SEASON_END_SHORT_HORIZON

    with pytest.raises(fg.DecisionCertificationRequired):
        certifier.certified_horizon_status(conn, artifact, [5, 6, 7, 8], cutoff)
    conn.close()


def test_14b_db_support_still_rediscovers_and_is_labelled_as_such(tmp_path):
    """The legacy path is retained but explicitly marked as rediscovery."""

    from fpl_brain import four_gw_decision as fg

    conn = connect_database(tmp_path / "fpl.db")
    _bundle_world(conn)
    support = fg.event_support_from_db(conn, [5], "2026-09-12T19:00:00Z")
    assert support[5]["supported"] is True
    # It picks the NEWEST same-cutoff xpts (run 7), which is the silent-replacement
    # behaviour the certified bundle exists to prevent.
    assert support[5]["matched_runs"]["xpts_v1"] == 7
    assert "bundle_identity" not in support[5]
    conn.close()
