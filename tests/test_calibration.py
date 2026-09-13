"""Analytics spine tests: immutability, outcomes, calibration, comparisons."""

from __future__ import annotations

import json
import math
import sqlite3

import pytest

from fpl_brain import analytics, calibration, minutes_model
from fpl_brain.database import connect_database

from test_minutes_model import CUTOFF, EVENT_DEADLINE, _world, _freeze_context


def test_frozen_predictions_are_immutable_at_storage_level(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    _world(conn)
    context = _freeze_context(conn)
    run_id, _counts = analytics.freeze_baselines_for_event(conn, context, CUTOFF, event=4, deadline_status="PRE_DEADLINE")
    rows = analytics.frozen_predictions(conn, run_id, [analytics.EP_NEXT_KIND])
    assert rows
    target_id = rows[0]["id"]
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        conn.execute("UPDATE frozen_predictions SET payload_json='{}' WHERE id=?", (target_id,))
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        conn.execute("DELETE FROM frozen_predictions WHERE id=?", (target_id,))
    # projection run rows are also canonical: only allowed status transitions.
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        conn.execute("UPDATE frozen_predictions SET model_version='x' WHERE id=?", (target_id,))
    conn.close()


def test_projection_run_requires_family_and_version(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    _world(conn)
    run_id = analytics.create_projection_run(
        conn,
        model_family=analytics.MINUTES_MODEL_FAMILY,
        model_version=minutes_model.MINUTES_MODEL_VERSION,
        planning_event=4,
        planning_context_hash="hash",
        data_cutoff=CUTOFF,
        scouting_cutoff=None,
        official_run_ids={},
    )
    record = analytics.get_projection_run(conn, run_id)
    assert record["status"] == "running" and record["model_family"] == "minutes_v1"
    analytics.finish_projection_run(conn, run_id, "complete")
    assert analytics.get_projection_run(conn, run_id)["status"] == "complete"
    conn.close()


def test_same_inputs_freeze_deterministic_prediction_payloads(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    _world(conn)
    context = _freeze_context(conn)
    config = minutes_model.MinutesModelConfig()
    run_a = analytics.create_projection_run(
        conn,
        model_family=analytics.MINUTES_MODEL_FAMILY,
        model_version=minutes_model.MINUTES_MODEL_VERSION,
        planning_event=4,
        planning_context_hash="ref",
        data_cutoff=CUTOFF,
        scouting_cutoff=None,
        official_run_ids={},
        config_hash=config.config_hash(),
        deadline_status="PRE_DEADLINE",
    )
    run_b = analytics.create_projection_run(
        conn,
        model_family=analytics.MINUTES_MODEL_FAMILY,
        model_version=minutes_model.MINUTES_MODEL_VERSION,
        planning_event=4,
        planning_context_hash="ref",
        data_cutoff=CUTOFF,
        scouting_cutoff=None,
        official_run_ids={},
        config_hash=config.config_hash(),
        deadline_status="PRE_DEADLINE",
    )
    for run_id in (run_a, run_b):
        with conn:
            for row in minutes_model.build_minutes_predictions(conn, 4, CUTOFF, config):
                analytics.freeze_prediction(
                    conn,
                    run_id,
                    kind=analytics.MINUTES_V1_KIND,
                    player_id=int(row["player_id"]),
                    event=4,
                    fixture_id=int(row["fixture_id"]),
                    payload=row,
                    model_version=minutes_model.MINUTES_MODEL_VERSION,
                )
    payload_a = {(r["player_id"], r["fixture_id"]): r["payload"] for r in analytics.frozen_predictions(conn, run_a, [analytics.MINUTES_V1_KIND])}
    payload_b = {(r["player_id"], r["fixture_id"]): r["payload"] for r in analytics.frozen_predictions(conn, run_b, [analytics.MINUTES_V1_KIND])}
    strip_at = lambda payload: {  # noqa: E731
        key: value for key, value in payload.items() if key not in {"generated_at"}
    }
    assert {key: strip_at(p) for key, p in payload_a.items()} == {key: strip_at(p) for key, p in payload_b.items()}
    conn.close()


def test_outcome_observation_requires_official_finalisation(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    _world(conn)
    with pytest.raises(ValueError, match="outcomes may only be observed"):
        calibration.observe_event_outcomes(conn, 4)  # pending event
    observed = calibration.observe_event_outcomes(conn, 3)
    assert observed > 0
    count = conn.execute("SELECT COUNT(*) FROM outcome_observations WHERE event=3").fetchone()[0]
    assert count == observed
    # Idempotent: re-observation updates in place, never duplicates.
    calibration.observe_event_outcomes(conn, 3)
    assert conn.execute("SELECT COUNT(*) FROM outcome_observations WHERE event=3").fetchone()[0] == count
    conn.close()


def test_outcomes_have_started_minutes_and_zero_buckets(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    _world(conn)
    calibration.observe_event_outcomes(conn, 3)
    rows = {
        (int(row["player_id"]), int(row["fixture_id"])): dict(row)
        for row in conn.execute("SELECT * FROM outcome_observations WHERE event=3")
    }
    starter = rows[(10, 3)]   # minutes 45 (started, not 60+)
    assert starter["actual_started"] == 1 and starter["actual_minutes"] == 45
    assert starter["actual_60_plus"] == 0 and starter["actual_zero_minutes"] == 0
    benched = rows[(12, 3)]   # minutes 0
    assert benched["actual_started"] == 0 and benched["actual_minutes"] == 0
    assert benched["actual_zero_minutes"] == 1 and benched["actual_60_plus"] == 0
    conn.close()


def test_calibration_metrics_match_hand_computation(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    _world(conn, override={"pgw": {10: [(1, 1, 90, 1)], 12: []}})
    # Freeze a tiny deterministic minutes world for event 4 with two rows:
    # player 10 (started every completed row) and player 12 (never started).
    context = _freeze_context(conn)
    config = minutes_model.MinutesModelConfig(start_prior_strength=0.5, cameo_rate_prior_strength=0.5, cameo_minutes_prior_strength=0.5, p60_if_start_prior_strength=0.01, p80_if_start_prior_strength=0.01, minutes_if_start_prior_strength=0.01)
    run_id = analytics.create_projection_run(
        conn,
        model_family=analytics.MINUTES_MODEL_FAMILY,
        model_version=minutes_model.MINUTES_MODEL_VERSION,
        planning_event=4,
        planning_context_hash="ref",
        data_cutoff=CUTOFF,
        scouting_cutoff=None,
        official_run_ids={},
        config_hash=config.config_hash(),
        deadline_status="PRE_DEADLINE",
    )
    rows = [row for row in minutes_model.build_minutes_predictions(conn, 4, CUTOFF, config) if row["player_id"] in (10, 12)]
    for row in rows:
        analytics.freeze_prediction(
            conn,
            run_id,
            kind=analytics.MINUTES_V1_KIND,
            player_id=int(row["player_id"]),
            event=4,
            fixture_id=int(row["fixture_id"]),
            payload=row,
            model_version=minutes_model.MINUTES_MODEL_VERSION,
        )
    # Observed outcome: player 10 started and played 45; player 12: zero again.
    with conn:
        conn.execute(
            "INSERT INTO outcome_observations(event, player_id, fixture_id, actual_started, actual_minutes, actual_60_plus, actual_zero_minutes, observed_at, source)"
            " VALUES (4, 10, 4, 1, 45, 0, 0, '2026-09-16T00:00:00Z', 'test')"
        )
        conn.execute(
            "INSERT INTO outcome_observations(event, player_id, fixture_id, actual_started, actual_minutes, actual_60_plus, actual_zero_minutes, observed_at, source)"
            " VALUES (4, 12, 4, 0, 0, 0, 1, '2026-09-16T00:00:00Z', 'test')"
        )
    metrics = calibration.evaluate_minutes_run(conn, run_id, 4)
    by_key = {(row["player_id"], row["fixture_id"]): row for row in rows}
    row10 = by_key[(10, 4)]
    row12 = by_key[(12, 4)]
    expected_brier = ((row10["p_start"] - 1) ** 2 + (row12["p_start"] - 0) ** 2) / 2
    assert metrics["brier_start"] == pytest.approx(expected_brier, abs=1e-9)
    assert metrics["sample_count"] == 2
    assert metrics["mae_minutes"] == pytest.approx(
        (abs(row10["expected_minutes"] - 45) + abs(row12["expected_minutes"] - 0)) / 2, abs=1e-6
    )
    expected_p60_brier = ((row10["p_60_plus"] - 0) ** 2 + (row12["p_60_plus"] - 0) ** 2) / 2
    assert metrics["brier_p60"] == pytest.approx(expected_p60_brier, abs=1e-9)
    expected_p0_brier = ((row10["p_zero"] - 0) ** 2 + (row12["p_zero"] - 1) ** 2) / 2
    assert metrics["brier_p0"] == pytest.approx(expected_p0_brier, abs=1e-9)
    # multiclass Brier must satisfy the coherence bound of 2 with 3 classes.
    assert 0.0 <= metrics["multiclass_brier"] <= 2.0
    assert 0.0 < metrics["multiclass_log_loss"] < math.inf
    conn.close()


def test_log_loss_edge_clamping(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    _world(conn, override={"pgw": {10: [(1, 1, 0, 0)]}})
    # A zero-probability actual-0 case must not explode log(0).
    value = calibration._log_value(0.0, 0)
    assert value < math.inf
    assert value == pytest.approx(-math.log(1 - calibration._LOG_CLAMP), rel=1e-6)
    conn.close()


def test_calibration_records_written_and_tied_to_runs(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    _world(conn, override={"pgw": {10: [(1, 1, 90, 1)], 12: []}})
    context = _freeze_context(conn)
    baseline_run, _ = analytics.freeze_baselines_for_event(conn, context, CUTOFF, event=4, deadline_status="PRE_DEADLINE")
    minutes_run = analytics.create_projection_run(
        conn,
        model_family=analytics.MINUTES_MODEL_FAMILY,
        model_version=minutes_model.MINUTES_MODEL_VERSION,
        planning_event=4,
        planning_context_hash="ref",
        data_cutoff=CUTOFF,
        scouting_cutoff=None,
        official_run_ids={},
        config_hash=minutes_model.MinutesModelConfig().config_hash(),
        deadline_status="PRE_DEADLINE",
    )
    rows = [row for row in minutes_model.build_minutes_predictions(conn, 4, CUTOFF) if row["player_id"] in (10, 12)]
    for row in rows:
        analytics.freeze_prediction(
            conn,
            minutes_run,
            kind=analytics.MINUTES_V1_KIND,
            player_id=int(row["player_id"]),
            event=4,
            fixture_id=int(row["fixture_id"]),
            payload=row,
            model_version=minutes_model.MINUTES_MODEL_VERSION,
        )
    calibration.observe_event_outcomes(conn, 3)
    # Direct outcome evidence for the frozen event 4 rows:
    with conn:
        conn.execute(
            "INSERT INTO outcome_observations(event, player_id, fixture_id, actual_started, actual_minutes, actual_60_plus, actual_zero_minutes, observed_at, source)"
            " VALUES (4, 10, 4, 1, 60, 1, 0, '2026-09-16T00:00:00Z', 'test')"
        )
    summary = calibration.calibrate_event(conn, minutes_run, baseline_run, event=4)
    # The baseline comparator uses NAIVE_MINUTES rows for the same join.
    assert summary["minutes_run_id"] == minutes_run
    assert summary["naive_baseline"] is not None
    counts = conn.execute(
        "SELECT COUNT(*) FROM calibration_records WHERE projection_run_id IN (?,?)",
        (minutes_run, baseline_run),
    ).fetchone()[0]
    assert counts >= 2
    assert summary["minutes_v1"].get("brier_start") is not None
    conn.close()
