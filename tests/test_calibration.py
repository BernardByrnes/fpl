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


# ---------------------------------------------------------------------------


# PE2-OUTCOME-PLACEHOLDER
#
# A finished fixture is necessary but not sufficient for an outcome.  When the
# refresh sequence never reaches the element-summary endpoint, a completed
# fixture keeps pre-round schedule rows: minutes=0 and every performance column
# NULL.  The outcome reader must not record those as "he did not play", and must
# still record a genuine zero-minute non-appearance, which carries explicit
# zeros rather than absent evidence.
# ---------------------------------------------------------------------------


def _placeholder_row(conn, *, player_id: int, event: int, fixture_id: int) -> None:
    """Insert a canonical scheduled placeholder: no observation at all."""

    from fpl_brain import repositories as repo
    from fpl_brain.models import PlayerGameweekRecord
    from test_minutes_model import OBSERVED_AT

    with conn:
        repo.upsert_player_gameweeks(
            conn,
            [PlayerGameweekRecord(player_id=player_id, event=event, fixture_id=fixture_id,
                                  minutes=0, source="element_summary", raw_json={})],
            OBSERVED_AT,
        )


def _stored_row(conn, *, player_id: int, event: int, fixture_id: int) -> dict:
    row = conn.execute(
        "SELECT pg.* FROM player_gameweeks pg WHERE pg.player_id=? AND pg.event=? AND pg.fixture_id=?",
        (player_id, event, fixture_id),
    ).fetchone()
    assert row is not None, "the seeded row is missing"
    return dict(row)


def _outcome_keys(conn, event: int) -> set:
    return {(int(r[0]), int(r[1])) for r in conn.execute(
        "SELECT player_id, fixture_id FROM outcome_observations WHERE event=?", (event,))}


def test_A_scheduled_placeholder_is_never_observed_as_an_outcome(tmp_path):
    """CASE A. The row is placeholder-shaped and the event is final, so nothing
    except the canonical placeholder signature can keep it out."""

    from fpl_brain import repositories as repo

    conn = connect_database(tmp_path / "fpl.db")
    _world(conn, override={"players": {30: (1, 3, "a")}})
    _placeholder_row(conn, player_id=30, event=3, fixture_id=3)

    stored = _stored_row(conn, player_id=30, event=3, fixture_id=3)
    assert repo.row_is_scheduled_placeholder(stored) is True, (
        "the constructed row is not placeholder-shaped, so this test would not "
        "exercise the defect it exists for"
    )
    assert stored["minutes"] == 0 and stored["total_points"] is None

    calibration.observe_event_outcomes(conn, 3)
    assert (30, 3) not in _outcome_keys(conn, 3), (
        "a scheduled placeholder was recorded as a zero-minute outcome"
    )
    conn.close()


def test_BC_placeholders_are_excluded_while_genuine_zeros_are_retained(tmp_path):
    """CASES B and C.  Exclusion is the placeholder SIGNATURE, not minutes == 0:
    in the same event a genuine zero-minute DNP survives beside an excluded
    placeholder, and an ordinary played player is written normally."""

    from fpl_brain import repositories as repo

    conn = connect_database(tmp_path / "fpl.db")
    _world(conn, override={"players": {30: (1, 3, "a")}})
    _placeholder_row(conn, player_id=30, event=3, fixture_id=3)

    assert repo.row_is_scheduled_placeholder(_stored_row(conn, player_id=30, event=3, fixture_id=3)) is True
    dnp = _stored_row(conn, player_id=12, event=3, fixture_id=3)   # minutes 0 with explicit zeros
    assert repo.row_is_scheduled_placeholder(dnp) is False
    assert dnp["minutes"] == 0 and dnp["total_points"] == 0, "the control DNP is not a real zero"

    calibration.observe_event_outcomes(conn, 3)
    rows = {(int(r["player_id"]), int(r["fixture_id"])): dict(r) for r in conn.execute(
        "SELECT * FROM outcome_observations WHERE event=3")}

    assert (30, 3) not in rows, "placeholder admitted"
    # CASE B: the genuine DNP is retained with unchanged semantics.
    assert (12, 3) in rows, "a genuine zero-minute non-appearance was erased"
    assert rows[(12, 3)]["actual_minutes"] == 0
    assert rows[(12, 3)]["actual_zero_minutes"] == 1
    assert rows[(12, 3)]["actual_started"] == 0
    # CASE C: the ordinary played player is written normally.
    assert rows[(10, 3)]["actual_minutes"] == 45 and rows[(10, 3)]["actual_started"] == 1
    assert rows[(10, 3)]["actual_zero_minutes"] == 0
    conn.close()


def test_E_a_mixed_event_persists_only_genuine_observations(tmp_path):
    """CASE E.  Placeholder + genuine DNP + played player in ONE event: the
    stored outcome set is exactly the genuine observations, with no silent zero
    substitution for the placeholder."""

    from fpl_brain import repositories as repo
    from fpl_brain.models import FixtureRecord, PlayerRecord

    conn = connect_database(tmp_path / "fpl.db")
    _world(conn, override={"players": {30: (1, 3, "a")}})
    with conn:
        # A second finished GW3 fixture whose rows were never refreshed: every
        # row on it is a scheduled placeholder.
        repo.upsert_fixtures(conn, [FixtureRecord(id=9, event=3, team_h=1, team_a=2, finished=1, started=1,
                                                  kickoff_time="2026-09-06T15:30:00Z", raw_json={})])
        repo.upsert_players(conn, [PlayerRecord(id=31, web_name="P31", full_name="Player 31",
                                                team_id=1, element_type=3)])
    _placeholder_row(conn, player_id=31, event=3, fixture_id=9)
    assert repo.row_is_scheduled_placeholder(_stored_row(conn, player_id=31, event=3, fixture_id=9)) is True

    observed = calibration.observe_event_outcomes(conn, 3)
    keys = _outcome_keys(conn, 3)

    assert (31, 9) not in keys, "the placeholder on the unrefreshed fixture was admitted"
    assert (12, 3) in keys and (10, 3) in keys
    assert observed == len(keys)
    null_points = conn.execute(
        "SELECT COUNT(*) FROM outcome_observations WHERE actual_points IS NULL").fetchone()[0]
    assert null_points == 0, "a placeholder-shaped row reached the outcome table"
    conn.close()


def test_D_a_non_final_event_blocks_and_writes_nothing(tmp_path):
    """CASE D.  GW4-class state: fixtures finished while the EVENT is not final.
    The finalisation gate must refuse and leave the outcome table untouched."""

    conn = connect_database(tmp_path / "fpl.db")
    _world(conn)
    before = conn.execute("SELECT COUNT(*) FROM outcome_observations").fetchone()[0]
    with pytest.raises(ValueError, match="outcomes may only be observed"):
        calibration.observe_event_outcomes(conn, 4)
    after = conn.execute("SELECT COUNT(*) FROM outcome_observations").fetchone()[0]
    assert after == before == 0, "a blocked event still wrote outcomes"
    conn.close()


def test_the_shared_completed_row_boundary_applies_the_placeholder_contract(tmp_path):
    """The evaluation side and the history side must agree on what an observation
    is, and they agree by REUSING the one definition rather than restating it."""

    from fpl_brain import repositories as repo

    conn = connect_database(tmp_path / "fpl.db")
    _world(conn, override={"players": {30: (1, 3, "a")}})
    _placeholder_row(conn, player_id=30, event=3, fixture_id=3)

    assert repo.gameweek_has_performance_sql("p") in repo._COMPLETED_PERFORMANCE_WHERE, (
        "the completed-row boundary no longer composes the canonical predicate"
    )

    rows = repo.completed_player_fixture_rows(conn, event=3)
    keys = {(int(r["player_id"]), int(r["fixture_id"])) for r in rows}
    assert (30, 3) not in keys, "a placeholder passed the completed-row boundary"
    assert (12, 3) in keys and (10, 3) in keys, "genuine rows were dropped"

    # The genuine DNP and the placeholder differ ONLY by the signature, so the
    # boundary is discriminating on the right thing.
    assert _stored_row(conn, player_id=12, event=3, fixture_id=3)["minutes"] == 0
    assert _stored_row(conn, player_id=30, event=3, fixture_id=3)["minutes"] == 0
    conn.close()


def test_the_baseline_arm_carries_the_code_fingerprint(tmp_path):
    """Every scored comparison arm in a certified bundle carries the code fingerprint.

    ``certified_bundle`` compares each family's recorded
    ``source_snapshot_sha256`` against the bundle's code snapshot.  A NULL is not
    a weaker claim, it is the absence of one: the family is skipped by that check
    entirely, so the arm is certified without its implementation being bound.
    """

    conn = connect_database(tmp_path / "fpl.db")
    _world(conn)
    context = _freeze_context(conn)
    run_id, _counts = analytics.freeze_baselines_for_event(
        conn, context, CUTOFF, event=4, deadline_status="PRE_DEADLINE"
    )
    row = conn.execute(
        "SELECT model_family, source_snapshot_sha256 FROM projection_runs WHERE id=?", (run_id,)
    ).fetchone()
    assert row[0] == analytics.BASELINE_MODEL_FAMILY
    assert row[1], "the baseline arm records no code fingerprint"
    assert row[1] == analytics.source_snapshot_sha256()
    conn.close()
