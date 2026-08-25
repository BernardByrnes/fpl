from __future__ import annotations

from fpl_brain import repositories as repo
from fpl_brain.database import connect_database
from fpl_brain.metrics import (
    attacking_and_defensive_outlook,
    fixture_outlook,
    minutes_reliability,
    points_per_million,
    trend,
)
from fpl_brain.models import FixtureRecord, PlayerGameweekRecord, PlayerRecord, PlayerSnapshotRecord, PositionRecord, TeamRecord


def test_metrics_are_transparent_and_handle_blanks_doubles(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    with conn:
        repo.upsert_teams(
            conn,
            [
                TeamRecord(id=1, name="One", strength_defence_home=10, strength_defence_away=11, strength_attack_home=20, strength_attack_away=21),
                TeamRecord(id=2, name="Two", strength_defence_home=30, strength_defence_away=31, strength_attack_home=40, strength_attack_away=41),
                TeamRecord(id=3, name="Three", strength_defence_home=50, strength_defence_away=51, strength_attack_home=60, strength_attack_away=61),
            ],
        )
        repo.upsert_positions(conn, [PositionRecord(id=3, singular_name_short="MID")])
        repo.upsert_players(conn, [PlayerRecord(id=1, web_name="Player", full_name="Player", team_id=1, element_type=3)])
        repo.upsert_fixtures(
            conn,
            [
                FixtureRecord(id=1, event=1, team_h=1, team_a=2, team_h_difficulty=2, team_a_difficulty=4, raw_json={}),
                FixtureRecord(id=2, event=2, team_h=3, team_a=1, team_h_difficulty=3, team_a_difficulty=5, raw_json={}),
                FixtureRecord(id=3, event=3, team_h=1, team_a=2, team_h_difficulty=2, team_a_difficulty=4, raw_json={}),
                FixtureRecord(id=4, event=3, team_h=3, team_a=1, team_h_difficulty=3, team_a_difficulty=5, raw_json={}),
            ],
        )
        run1 = repo.create_fetch_run(conn, "fetch_fpl")
        repo.insert_snapshots(conn, [PlayerSnapshotRecord(player_id=1, captured_at="2026-08-01T00:00:00Z", now_cost=80, selected_by_percent=10, total_points=100, raw_json={})], run1)
        run2 = repo.create_fetch_run(conn, "fetch_fpl")
        repo.insert_snapshots(conn, [PlayerSnapshotRecord(player_id=1, captured_at="2026-08-10T00:00:00Z", now_cost=85, selected_by_percent=12, total_points=100, raw_json={})], run2)
    outlook = fixture_outlook(conn, 1, 1, 5)
    assert outlook["fixture_count"] == 4
    assert outlook["blank_count"] == 1
    assert outlook["double_events"] == [3]
    assert outlook["mean_fdr"] == 3.5
    assert points_per_million(conn, 1) == 100 / 8.5
    movement = trend(conn, 1, "now_cost", 30)
    assert movement["absolute_delta"] == 5
    assert movement["sample_count"] == 2
    assert attacking_and_defensive_outlook(conn, 1, 1, 1)["fixture_count"] == 1
    conn.close()


def test_minutes_reliability_ignores_future_schedule_rows_and_uses_completed_events(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    with conn:
        repo.upsert_players(conn, [PlayerRecord(id=1, web_name="Player", full_name="Player")])
        repo.upsert_fixtures(
            conn,
            [
                FixtureRecord(id=1, event=1, finished=1, raw_json={}),
                FixtureRecord(id=2, event=2, finished=1, raw_json={}),
                FixtureRecord(id=3, event=3, finished=0, raw_json={}),
                FixtureRecord(id=4, event=4, finished=0, raw_json={}),
                FixtureRecord(id=5, event=5, finished=0, raw_json={}),
                FixtureRecord(id=6, event=6, finished=0, raw_json={}),
                FixtureRecord(id=7, event=7, finished=0, raw_json={}),
                FixtureRecord(id=8, event=8, finished=0, raw_json={}),
            ],
        )
        repo.upsert_player_gameweeks(
            conn,
            [
                PlayerGameweekRecord(player_id=1, event=1, fixture_id=1, minutes=90, starts=1, total_points=6, raw_json={}),
                PlayerGameweekRecord(player_id=1, event=2, fixture_id=2, minutes=0, starts=0, total_points=0, raw_json={}),
                PlayerGameweekRecord(player_id=1, event=3, fixture_id=3, minutes=0, kickoff_time="future", raw_json={}),
                PlayerGameweekRecord(player_id=1, event=4, fixture_id=4, minutes=0, kickoff_time="future", raw_json={}),
                PlayerGameweekRecord(player_id=1, event=5, fixture_id=5, minutes=0, kickoff_time="future", raw_json={}),
                PlayerGameweekRecord(player_id=1, event=6, fixture_id=6, minutes=0, kickoff_time="future", raw_json={}),
                PlayerGameweekRecord(player_id=1, event=7, fixture_id=7, minutes=0, kickoff_time="future", raw_json={}),
                PlayerGameweekRecord(player_id=1, event=8, fixture_id=8, minutes=0, kickoff_time="future", raw_json={}),
            ],
        )
    rows = repo.gameweek_rows(conn, 1, 5)
    assert [row["event"] for row in rows] == [1, 2]
    reliability = minutes_reliability(conn, 1, 5)
    assert reliability == {"starts": 1, "appearances": 1, "total_minutes": 90, "mean_minutes": 45, "events_sampled": 2}
    conn.close()


def test_minutes_reliability_returns_none_for_pure_preseason_schedule_minutes_zero(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    with conn:
        repo.upsert_players(conn, [PlayerRecord(id=1, web_name="Player", full_name="Player")])
        repo.upsert_fixtures(
            conn,
            [FixtureRecord(id=fixture_id, event=fixture_id, finished=0, raw_json={}) for fixture_id in range(1, 5)],
        )
        repo.upsert_player_gameweeks(
            conn,
            [
                PlayerGameweekRecord(player_id=1, event=fixture_id, fixture_id=fixture_id, minutes=0, source="element_summary", raw_json={"minutes": 0})
                for fixture_id in range(1, 5)
            ],
        )
    assert repo.gameweek_rows(conn, 1, 5) == []
    assert minutes_reliability(conn, 1, 5) is None
    conn.close()
