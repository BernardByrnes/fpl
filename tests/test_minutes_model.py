"""Minutes Model v1 tests: coherence, shrinkage, modifiers, no-lookahead, grain."""

from __future__ import annotations

import copy

import pytest

from fpl_brain import analytics, minutes_model
from fpl_brain.database import connect_database
from fpl_brain.models import (
    EventRecord,
    FixtureRecord,
    PlayerGameweekRecord,
    PlayerRecord,
    PlayerSnapshotRecord,
    TeamRecord,
)

CUTOFF = "2026-09-10T12:00:00Z"
EVENT_DEADLINE = "2026-09-12T12:30:00Z"


def _world(conn, *, override: dict | None = None):
    seed = {
        "pgw": {
            # player: [(event, fixture, minutes, starts)]
            10: [(1, 1, 90, 1), (2, 2, 90, 1), (3, 3, 45, 1)],      # regular starter
            11: [(1, 1, 90, 1), (2, 2, 90, 1), (3, 3, 80, 1)],      # heavy starter
            12: [(1, 1, 0, 0), (2, 2, 0, 0), (3, 3, 0, 0)],         # bench regular
            13: [(1, 1, 90, 1), (2, 2, 0, 0)],                      # injured since last start
            14: [(1, 1, 90, 1)],                                    # suspended
            15: [(1, 1, 90, 1), (2, 2, 90, 1), (3, 3, 90, 1), (3, 6, 85, 1)],  # midweek heavy
            22: [(1, 1, 0, 0), (2, 2, 0, 0), (3, 3, 0, 0)],         # background MID rows (team 1)
        },
        "players": {
            10: (1, 2, "a"),  # (team, position, status): DEF starter
            11: (1, 3, "a"),
            12: (1, 3, "a"),
            13: (1, 2, "i"),
            14: (1, 2, "s"),
            15: (2, 3, "a"),
            22: (1, 3, "a"),
            20: (3, 2, "a"),  # team 3 has NO event-4 fixture (blank week)
        },
        "snapshots": {10: ("a", None), 11: ("a", None), 12: ("a", None), 13: ("i", 25), 14: ("s", 0), 15: ("a", None), 22: ("a", None), 20: ("a", None)},
        "scout_notes": {10: ("rotation_risk", "high")},
    }
    if override:
        for key, values in override.items():
            seed[key].update(values)
    with conn:
        analytics_repo_upserts(conn, seed)
    return seed


def analytics_repo_upserts(conn, seed):  # noqa: N802 (test-local helper)
    from fpl_brain import repositories as repo
    from fpl_brain.models import PositionRecord

    repo.upsert_teams(conn, [TeamRecord(id=1, name="One"), TeamRecord(id=2, name="Two"), TeamRecord(id=3, name="Three")])
    repo.upsert_positions(
        conn,
        [PositionRecord(id=pos, singular_name_short=name) for pos, name in ((1, "GKP"), (2, "DEF"), (3, "MID"), (4, "FWD"))],
    )
    repo.upsert_players(
        conn,
        [
            PlayerRecord(id=pid, web_name=f"P{pid}", full_name=f"Player {pid}", team_id=team, element_type=pos)
            for pid, (team, pos, _status) in seed["players"].items()
        ],
    )
    repo.upsert_events(
        conn,
        [
            EventRecord(id=1, finished=1, data_checked=1, deadline_time="2026-08-21T17:30:00Z", raw_json={}),
            EventRecord(id=2, finished=1, data_checked=1, deadline_time="2026-08-28T17:30:00Z", raw_json={}),
            EventRecord(id=3, finished=1, data_checked=1, deadline_time="2026-09-04T17:30:00Z", raw_json={}),
            EventRecord(id=4, finished=0, data_checked=0, deadline_time=EVENT_DEADLINE, raw_json={}),
        ],
    )
    run = repo.create_fetch_run(conn, "fetch_fpl")
    snapshots = []
    for player_id, (status, chance) in seed["snapshots"].items():
        snapshots.append(
            PlayerSnapshotRecord(
                player_id=player_id,
                captured_at="2026-09-10T08:00:00Z",
                now_cost=50,
                status=status,
                chance_of_playing_this_round=chance,
                raw_json={},
            )
        )
    repo.insert_snapshots(conn, snapshots, run)
    repo.upsert_fixtures(
        conn,
        [
            FixtureRecord(id=1, event=1, team_h=1, team_a=2, finished=1, started=1, kickoff_time="2026-08-22T14:00:00Z", raw_json={}),
            FixtureRecord(id=2, event=2, team_h=1, team_a=2, finished=1, started=1, kickoff_time="2026-08-29T14:00:00Z", raw_json={}),
            FixtureRecord(id=3, event=3, team_h=1, team_a=2, finished=1, started=1, kickoff_time="2026-09-05T14:00:00Z", raw_json={}),
            FixtureRecord(id=4, event=4, team_h=1, team_a=2, finished=0, started=0, kickoff_time="2026-09-13T14:00:00Z", raw_json={}),
            FixtureRecord(id=5, event=4, team_h=2, team_a=1, finished=0, started=0, kickoff_time="2026-09-14T14:00:00Z", raw_json={}),
            # team 2 also had a completed midweek fixture ~3.75 days before fixture 4
            FixtureRecord(id=6, event=3, team_h=2, team_a=3, finished=1, started=1, kickoff_time="2026-09-09T19:45:00Z", raw_json={}),
        ],
    )
    gameweeks = []
    for player_id, rows in seed["pgw"].items():
        for event, fixture, minutes, starts in rows:
            gameweeks.append(
                PlayerGameweekRecord(
                    player_id=player_id,
                    event=event,
                    fixture_id=fixture,
                    minutes=minutes,
                    starts=starts,
                    total_points=2 if starts else 0,
                    source="element_summary",
                    raw_json={},
                )
            )
    repo.upsert_player_gameweeks(conn, gameweeks)
    for player_id, (key, value) in seed.get("scout_notes", {}).items():
        import_id = repo.insert_scouting_import(conn, {"source_file": "world", "file_sha256": f"world-{player_id}", "players_total": 1, "players_resolved": 1, "notes_inserted": 1})
        repo.insert_scouting_note(
            conn,
            {
                "import_id": import_id,
                "player_id": player_id,
                "key": key,
                "value_text": value,
                "confidence": "medium",
                "observed_at": "2026-09-09T09:00:00Z",
                "expires_at": "2026-09-30T00:00:00Z",
            },
        )


def test_probability_validity_and_coherence(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    _world(conn)
    rows = minutes_model.build_minutes_predictions(conn, 4, CUTOFF)
    for row in rows:
        values = {
            "p_available": row["p_available"],
            "p_start": row["p_start"],
            "p_cameo": row["p_cameo"],
            "p_zero": row["p_zero"],
            "p_1_59": row["p_1_59"],
            "p_60_plus": row["p_60_plus"],
            "p_80_plus": row["p_80_plus"],
        }
        for name, value in values.items():
            assert 0.0 <= value <= 1.0, (row["player_id"], name, value)
        assert row["p_start"] <= row["p_available"] + 1e-9
        assert row["p_60_plus"] <= row["p_start"] + row["p_cameo"] + 1e-9
        assert row["p_80_plus"] <= row["p_60_plus"] + 1e-9
        assert abs((row["p_zero"] + row["p_1_59"] + row["p_60_plus"]) - 1.0) <= 3e-6  # 6dp storage rounding
        expected = row["p_start"] * row["expected_minutes_if_start"] + row["p_cameo"] * row["expected_minutes_if_cameo"]
        assert abs(expected - row["expected_minutes"]) <= 1 + 1e-6
    conn.close()


def test_hard_out_player_is_zeroed_with_trace_row(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    _world(conn)
    rows = {row["player_id"]: row for row in minutes_model.build_minutes_predictions(conn, 4, CUTOFF)}
    suspended = rows[14]
    assert suspended["p_available"] == 0.0
    assert suspended["p_start"] == 0.0 and suspended["p_cameo"] == 0.0
    assert suspended["expected_minutes"] == 0.0
    assert suspended["p_zero"] == 1.0
    conn.close()


def test_blank_week_has_no_projection_row(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    _world(conn)
    rows = [row for row in minutes_model.build_minutes_predictions(conn, 4, CUTOFF)]
    assert any(row["player_id"] == 20 for row in rows) is False
    conn.close()


def test_dgw_player_gets_two_independent_projections(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    # Team 1 plays fixtures 4 AND 5 in event 4 in the base world.
    _world(conn)
    rows = [row for row in minutes_model.build_minutes_predictions(conn, 4, CUTOFF) if row["player_id"] == 10]
    assert sorted(row["fixture_id"] for row in rows) == [4, 5]
    assert all(row["event"] == 4 for row in rows)
    conn.close()


def test_posterior_moves_toward_evidence_and_small_samples_shrink(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    config = minutes_model.MinutesModelConfig()
    _world(conn)
    regular = minutes_model.build_minutes_predictions(conn, 4, CUTOFF, config)[0]
    # Player 12 never starts: observed rate 0 vs player 10 rate 2/3.
    bench = [row for row in minutes_model.build_minutes_predictions(conn, 4, CUTOFF, config) if row["player_id"] == 12][0]
    assert regular["p_start_given_available"] > bench["p_start_given_available"]
    # Small sample: probe 12's world with exactly one start in history.
    solo = connect_database(tmp_path / "solo.db")
    _world(solo, override={"pgw": {12: [(1, 1, 90, 1)], 10: [(1, 1, 0, 0)]}})
    solo_rows = [row for row in minutes_model.build_minutes_predictions(solo, 4, CUTOFF, config) if row["player_id"] == 12][0]
    prior = solo_rows["start_evidence"]["prior_start_rate"]
    p_sga = solo_rows["p_start_given_available"]
    assert prior < p_sga < 1.0  # evidence moves it above prior, shrink keeps it below reality
    assert p_sga < regular["p_start_given_available"]
    solo.close()
    conn.close()


def test_recent_start_dominates_older_bench(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    # Newest observation starts (index 0 most recent): rows newest-first.
    _world(conn)
    evidence = [
        {"minutes": 90, "starts": 1},
        {"minutes": 0, "starts": 0},
    ]
    weights = minutes_model._recency_weight_list(len(evidence), 1.0)
    posterior = (sum(w * s for w, s in zip(weights, (1, 0))) + 5.0 * 0.5) / (sum(weights) + 5.0)
    inverse = minutes_model._recency_weight_list(len(evidence), 1.0)
    posterior_reversed = (sum(w * s for w, s in zip(reversed(weights), (1, 0))) + 5.0 * 0.5) / (sum(reversed(weights)) + 5.0)
    assert posterior > posterior_reversed  # recency weighting favours the startest evidence
    assert weights[0] > weights[-1] and inverse[-2] == weights[0]
    conn.close()


def test_injury_return_reduces_upper_minutes_and_flags(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    config = minutes_model.MinutesModelConfig()
    _world(conn)
    base_rows = {row["player_id"]: row for row in minutes_model.build_minutes_predictions(conn, 4, CUTOFF, config)}
    returning = base_rows[13]
    healthy_analog = connect_database(tmp_path / "healthy.db")
    _world(healthy_analog, override={"snapshots": {13: ("a", None)}})
    analog = [row for row in minutes_model.build_minutes_predictions(healthy_analog, 4, CUTOFF, config) if row["player_id"] == 13][0]
    assert returning["p_80_plus"] <= analog["p_80_plus"] + 1e-9
    assert "RETURN_RAMP" in returning["risk_flags"]
    healthy_analog.close()
    conn.close()


def test_midweek_heavy_and_rested_modifiers(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    config = minutes_model.MinutesModelConfig()
    # Player 11's team fixture 4 kicks off 2026-09-13; team 2's midweek fixture 6
    # is 2026-09-10 (within the 4-day window) with heavy minutes for player 15.
    _world(conn)
    rows = {(row["player_id"], row["fixture_id"]): row for row in minutes_model.build_minutes_predictions(conn, 4, CUTOFF, config)}
    heavy = rows[(15, 4)]
    assert "MIDWEEK_HEAVY" in heavy["risk_flags"]
    # The later DGW fixture (fixture 5, 4.75 rest days) is outside the window.
    assert "MIDWEEK_HEAVY" not in rows[(15, 5)]["risk_flags"]
    # Player 11 (team 1) had no fixture near the window -> no congestion flags.
    no_congestion = rows[(11, 4)]
    assert "MIDWEEK_HEAVY" not in no_congestion["risk_flags"] and "MIDWEEK_DATA_UNAVAILABLE" not in no_congestion["risk_flags"]
    conn.close()


def test_scouting_modifier_is_bounded_deterministic_and_traceable(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    _world(conn)
    rows = {row["player_id"]: row for row in minutes_model.build_minutes_predictions(conn, 4, CUTOFF)}
    assert rows[10]["modifier_ids"], "a scouting note should contribute a traceable modifier id"
    assert all(0.0 <= p <= 1.0 for p in (rows[10]["raw_p_start"], rows[10]["adjusted_p_start"]))
    # raw vs adjusted differ only by the bounded delta
    assert abs(rows[10]["raw_p_start"] - rows[10]["adjusted_p_start"]) <= minutes_model.MinutesModelConfig().scout_p_start_delta_bound + 1e-9
    conn.close()


def test_no_lookahead_post_deadline_evidence_cannot_leak(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    _world(conn)
    config = minutes_model.MinutesModelConfig()
    baseline_rows = minutes_model.build_minutes_predictions(conn, 4, CUTOFF, config)
    with conn:
        # Inject a post-deadline observation and snapshot AFTER the cutoff:
        # player 12 started and played 90 minutes in event 4 (post-deadline).
        from fpl_brain import repositories as repo

        repo.upsert_player_gameweeks(
            conn,
            [PlayerGameweekRecord(player_id=12, event=4, fixture_id=4, minutes=90, starts=1, total_points=9, source="event_live", raw_json={})],
        )
        run = repo.create_fetch_run(conn, "fetch_fpl")
        repo.insert_snapshots(
            conn,
            [PlayerSnapshotRecord(player_id=12, captured_at="2026-09-13T08:00:00Z", now_cost=50, status="a", raw_json={})],
            run,
        )
    after_rows = minutes_model.build_minutes_predictions(conn, 4, CUTOFF, config)
    strip = lambda rows: [{k: v for k, v in r.items() if k != "generated_at"} for r in rows]  # noqa: E731
    assert strip(baseline_rows) == strip(after_rows)
    conn.close()


def test_monetary_changes_never_change_minutes(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    seed = _world(conn)
    config = minutes_model.MinutesModelConfig()
    before = [dict(r) for r in minutes_model.build_minutes_predictions(conn, 4, CUTOFF, config)]
    with conn:
        from fpl_brain import repositories as repo
        from fpl_brain.models import PlayerSnapshotRecord as Snap

        run = repo.create_fetch_run(conn, "fetch_fpl")
        repo.insert_snapshots(
            conn,
            [
                Snap(
                    player_id=pid,
                    captured_at="2026-09-10T10:00:00Z",
                    now_cost=99,
                    raw_json={},
                    status=seed["snapshots"][pid][0],
                    chance_of_playing_this_round=seed["snapshots"][pid][1],
                )
                for pid in sorted(seed["snapshots"])
            ],
            run,
        )
        repo.insert_manager_acquisition(conn, 241392, 10, 4, 55, source="manual", created_at="2026-09-10T10:00:00Z")
        repo.upsert_manager_selling_prices(conn, 241392, 4, {10: 55}, captured_at="2026-09-10T10:00:00Z")
    after = minutes_model.build_minutes_predictions(conn, 4, CUTOFF, config)
    strip = lambda rows: [{k: v for k, v in r.items() if k != "generated_at"} for r in rows]  # noqa: E731
    assert strip(before) == strip(after)
    conn.close()


def test_freeze_then_outcomes_do_not_mutate_frozen_rows(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    _world(conn)
    from fpl_brain.database import connect_database as connect  # noqa: F401
    from fpl_brain.planning import get_planning_context

    context = _freeze_context(conn)
    run_id, counts = analytics.freeze_baselines_for_event(conn, context, CUTOFF, event=4, deadline_status="PRE_DEADLINE")
    model_config = minutes_model.MinutesModelConfig()
    minutes_run = analytics.create_projection_run(
        conn,
        model_family=analytics.MINUTES_MODEL_FAMILY,
        model_version=minutes_model.MINUTES_MODEL_VERSION,
        planning_event=4,
        planning_context_hash=analytics.planning_context_reference(context),
        data_cutoff=CUTOFF,
        scouting_cutoff=context.scouting_cutoff,
        official_run_ids=context.official_runs,
        config_hash=model_config.config_hash(),
        deadline_status="PRE_DEADLINE",
    )
    rows = minutes_model.build_minutes_predictions(conn, 4, CUTOFF, model_config)
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
    before = analytics.frozen_predictions(conn, minutes_run)
    with conn:
        from fpl_brain import calibration

        # Simulate finalised outcomes existing; must not touch frozen rows.
        calibration.observe_event_outcomes(conn, 3)
    after = analytics.frozen_predictions(conn, minutes_run)
    assert before == after
    conn.close()


def _freeze_context(conn):
    class SimpleContext:
        entry_id = 241392
        season = "2026/27"
        planning_event = 4
        as_of = CUTOFF
        event_data_state = "SCHEDULED"
        deadline = EVENT_DEADLINE
        scouting_cutoff = "2026-09-02T10:49:00Z"
        official_runs = {}
        manager_state = {}
        squad = {}
        official_price_freshness = {}
        health = {"status": "PASS"}

    return SimpleContext()


def test_config_hash_changes_with_tunable(tmp_path):
    base = minutes_model.MinutesModelConfig()
    changed = minutes_model.MinutesModelConfig(start_prior_strength=6.0)
    assert base.config_hash() != changed.config_hash()
    assert base.config_hash() == minutes_model.MinutesModelConfig().config_hash()


def test_readiness_gate(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    _world(conn)
    rows = minutes_model.build_minutes_predictions(conn, 4, CUTOFF)
    summary = minutes_model.readiness_summary(
        None, rows, deadline_status="PRE_DEADLINE", data_cutoff=CUTOFF, deadline=EVENT_DEADLINE
    )
    assert summary["status"] in {"PASS", "WARN"}
    late = minutes_model.readiness_summary(
        None, rows, deadline_status="PRE_DEADLINE", data_cutoff="2026-09-13T00:00:00Z", deadline=EVENT_DEADLINE
    )
    assert late["status"] == "FAIL" and any("MODEL_INPUTS_AFTER_DEADLINE" in reason for reason in late["fail_reasons"])
    marked_late = minutes_model.readiness_summary(
        None, rows, deadline_status="LATE_FREEZE", data_cutoff="2026-09-13T00:00:00Z", deadline=EVENT_DEADLINE
    )
    assert marked_late["status"] != "FAIL"
    conn.close()
