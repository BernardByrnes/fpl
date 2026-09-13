from __future__ import annotations

from fpl_brain import repositories as repo
from fpl_brain.database import connect_database
from fpl_brain.metrics import (
    attacking_and_defensive_outlook,
    calculate_selling_price,
    effective_selling_price,
    fixture_outlook,
    minutes_reliability,
    points_per_million,
    realisable_squad_value,
    transfer_affordability,
    trend,
)
from fpl_brain.models import EntryRecord, EventRecord, FixtureRecord, HistoryRow, PickRecord, PlayerGameweekRecord, PlayerRecord, PlayerSnapshotRecord, PositionRecord, TeamRecord


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


def test_future_outlook_excludes_completed_and_started_fixtures_and_uses_official_state_not_kickoff(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    with conn:
        repo.upsert_teams(
            conn,
            [
                TeamRecord(id=1, name="One", strength_defence_home=10, strength_defence_away=11, strength_attack_home=20, strength_attack_away=21),
                TeamRecord(id=2, name="Two", strength_defence_home=30, strength_defence_away=40, strength_attack_home=50, strength_attack_away=60),
                TeamRecord(id=3, name="Three", strength_defence_home=70, strength_defence_away=80, strength_attack_home=90, strength_attack_away=100),
            ],
        )
        repo.upsert_fixtures(
            conn,
            [
                FixtureRecord(id=20, event=2, team_h=2, team_a=1, team_a_difficulty=5, started=1, finished=1, raw_json={}),
                FixtureRecord(id=21, event=2, team_h=1, team_a=3, team_h_difficulty=4, started=1, finished=0, raw_json={}),
                # The stale-looking kickoff is intentionally retained: the
                # official state flags, not wall-clock time, govern inclusion.
                FixtureRecord(id=22, event=3, team_h=2, team_a=1, team_a_difficulty=2, kickoff_time="2026-07-01T00:00:00Z", started=0, finished=0, raw_json={}),
                FixtureRecord(id=23, event=4, team_h=1, team_a=2, team_h_difficulty=3, started=0, finished=0, raw_json={}),
                FixtureRecord(id=24, event=5, team_h=3, team_a=1, team_a_difficulty=1, started=0, finished=0, raw_json={}),
                FixtureRecord(id=25, event=6, team_h=1, team_a=2, team_h_difficulty=5, started=0, finished=0, raw_json={}),
            ],
        )

    outlook = fixture_outlook(conn, 1, 2, 3)
    assert [item["fixture_id"] for item in outlook["fixtures"]] == [22, 23, 24]
    assert [item["event"] for item in outlook["fixtures"]] == [3, 4, 5]
    assert outlook["mean_fdr"] == 2
    assert 20 not in [item["fixture_id"] for item in outlook["fixtures"]]
    assert 21 not in [item["fixture_id"] for item in outlook["fixtures"]]
    assert attacking_and_defensive_outlook(conn, 1, 2, 3) == {
        "attacking_outlook": 46.666666666666664,
        "defensive_outlook": 66.66666666666667,
        "mean_opponent_defence": 46.666666666666664,
        "mean_opponent_attack": 66.66666666666667,
        "fixture_count": 3,
    }
    conn.close()


def test_future_outlook_places_old_event_fixture_by_rescheduled_kickoff_without_horizon_contamination(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    with conn:
        repo.upsert_teams(conn, [TeamRecord(id=1, name="One"), TeamRecord(id=2, name="Two")])
        repo.upsert_events(
            conn,
            [
                EventRecord(id=3, deadline_time="2026-09-01T00:00:00Z"),
                EventRecord(id=8, deadline_time="2026-10-01T00:00:00Z"),
                EventRecord(id=9, deadline_time="2026-10-08T00:00:00Z"),
                EventRecord(id=10, deadline_time="2026-10-15T00:00:00Z"),
                EventRecord(id=11, deadline_time="2026-10-22T00:00:00Z"),
                EventRecord(id=12, deadline_time="2026-10-29T00:00:00Z"),
                EventRecord(id=13, deadline_time="2026-11-05T00:00:00Z"),
                EventRecord(id=14, deadline_time="2026-11-12T00:00:00Z"),
            ],
        )
        repo.upsert_fixtures(
            conn,
            [
                # The source event is old, but the confirmed reschedule belongs
                # inside the GW8–GW10 deadline window.
                FixtureRecord(id=40, event=3, team_h=1, team_a=2, team_h_difficulty=4, kickoff_time="2026-10-10T12:00:00Z", started=0, finished=0, raw_json={}),
                # This old-event pending row is safely known to be around GW13,
                # so it must not contaminate the GW8–GW10 FDR horizon.
                FixtureRecord(id=41, event=3, team_h=1, team_a=2, team_h_difficulty=1, kickoff_time="2026-11-06T12:00:00Z", started=0, finished=0, raw_json={}),
                FixtureRecord(id=42, event=3, team_h=1, team_a=2, team_h_difficulty=5, kickoff_time="2026-10-10T12:00:00Z", started=1, finished=1, raw_json={}),
                FixtureRecord(id=43, event=3, team_h=1, team_a=2, team_h_difficulty=5, kickoff_time="2026-10-10T12:00:00Z", started=1, finished=0, raw_json={}),
                FixtureRecord(id=50, event=8, team_h=1, team_a=2, team_h_difficulty=2, kickoff_time="2026-10-02T12:00:00Z", started=0, finished=0, raw_json={}),
                FixtureRecord(id=51, event=9, team_h=1, team_a=2, team_h_difficulty=3, kickoff_time="2026-10-09T12:00:00Z", started=0, finished=0, raw_json={}),
                FixtureRecord(id=52, event=10, team_h=1, team_a=2, team_h_difficulty=5, kickoff_time="2026-10-16T12:00:00Z", started=0, finished=0, raw_json={}),
                FixtureRecord(id=53, event=11, team_h=1, team_a=2, team_h_difficulty=2, kickoff_time="2026-10-23T12:00:00Z", started=0, finished=0, raw_json={}),
                FixtureRecord(id=54, event=12, team_h=1, team_a=2, team_h_difficulty=3, kickoff_time="2026-10-30T12:00:00Z", started=0, finished=0, raw_json={}),
                FixtureRecord(id=55, event=13, team_h=1, team_a=2, team_h_difficulty=4, kickoff_time="2026-11-06T19:00:00Z", started=0, finished=0, raw_json={}),
            ],
        )

    outlook = fixture_outlook(conn, 1, 8, 3)
    assert [item["fixture_id"] for item in outlook["fixtures"]] == [50, 51, 40, 52]
    assert [item["event"] for item in outlook["fixtures"]] == [8, 9, 3, 10]
    assert [item["horizon_event"] for item in outlook["fixtures"]] == [8, 9, 9, 10]
    assert outlook["mean_fdr"] == 3.5
    assert outlook["double_events"] == [9]
    assert 41 not in [item["fixture_id"] for item in outlook["fixtures"]]
    assert 42 not in [item["fixture_id"] for item in outlook["fixtures"]]
    assert 43 not in [item["fixture_id"] for item in outlook["fixtures"]]

    later_outlook = fixture_outlook(conn, 1, 11, 3)
    later_fixture = next(item for item in later_outlook["fixtures"] if item["fixture_id"] == 41)
    assert later_fixture["event"] == 3
    assert later_fixture["horizon_event"] == 13
    conn.close()


def test_future_outlook_excludes_old_pending_fixture_with_unknown_schedule_and_surfaces_it(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    with conn:
        repo.upsert_teams(conn, [TeamRecord(id=1, name="One"), TeamRecord(id=2, name="Two")])
        repo.upsert_events(
            conn,
            [
                EventRecord(id=3, deadline_time="2026-09-01T00:00:00Z"),
                EventRecord(id=8, deadline_time="2026-10-01T00:00:00Z"),
                EventRecord(id=9, deadline_time="2026-10-08T00:00:00Z"),
                EventRecord(id=10, deadline_time="2026-10-15T00:00:00Z"),
                EventRecord(id=11, deadline_time="2026-10-22T00:00:00Z"),
            ],
        )
        repo.upsert_fixtures(
            conn,
            [
                FixtureRecord(id=60, event=3, team_h=1, team_a=2, team_h_difficulty=5, started=0, finished=0, raw_json={}),
                FixtureRecord(id=61, event=8, team_h=1, team_a=2, team_h_difficulty=2, kickoff_time="2026-10-02T12:00:00Z", started=0, finished=0, raw_json={}),
                FixtureRecord(id=62, event=9, team_h=1, team_a=2, team_h_difficulty=3, kickoff_time="2026-10-09T12:00:00Z", started=0, finished=0, raw_json={}),
            ],
        )

    outlook = fixture_outlook(conn, 1, 8, 3)
    assert [item["fixture_id"] for item in outlook["fixtures"]] == [61, 62]
    assert outlook["mean_fdr"] == 2.5
    assert [row["id"] for row in repo.unplaced_pending_fixture_rows(conn, 8, 3)] == [60]
    conn.close()


def test_future_outlook_places_old_pending_fixture_in_open_ended_final_event_window(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    with conn:
        repo.upsert_teams(conn, [TeamRecord(id=1, name="One"), TeamRecord(id=2, name="Two")])
        repo.upsert_events(
            conn,
            [
                EventRecord(id=3, deadline_time="2026-09-01T00:00:00Z"),
                EventRecord(id=37, deadline_time="2027-05-01T00:00:00Z"),
                EventRecord(id=38, deadline_time="2027-05-08T00:00:00Z"),
            ],
        )
        repo.upsert_fixtures(
            conn,
            [
                FixtureRecord(id=90, event=3, team_h=1, team_a=2, team_h_difficulty=5, kickoff_time="2027-05-09T12:00:00Z", started=0, finished=0, raw_json={}),
                FixtureRecord(id=91, event=37, team_h=1, team_a=2, team_h_difficulty=2, kickoff_time="2027-05-02T12:00:00Z", started=0, finished=0, raw_json={}),
                FixtureRecord(id=92, event=38, team_h=1, team_a=2, team_h_difficulty=3, kickoff_time="2027-05-09T10:00:00Z", started=0, finished=0, raw_json={}),
            ],
        )

    outlook = fixture_outlook(conn, 1, 37, 3)
    assert [item["fixture_id"] for item in outlook["fixtures"]] == [91, 92, 90]
    old_fixture = outlook["fixtures"][-1]
    assert old_fixture["event"] == 3
    assert old_fixture["horizon_event"] == 38
    assert outlook["fixture_count"] == 3
    assert outlook["mean_fdr"] == 10 / 3
    assert outlook["double_events"] == [38]
    conn.close()


def test_future_outlook_preserves_normal_finalized_gameweek_sequences(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    with conn:
        repo.upsert_teams(
            conn,
            [
                TeamRecord(id=1, name="Arsenal"),
                TeamRecord(id=2, name="Aston Villa"),
                TeamRecord(id=3, name="Chelsea"),
                TeamRecord(id=4, name="Liverpool"),
                TeamRecord(id=5, name="Nottingham Forest"),
                TeamRecord(id=6, name="Ipswich"),
                TeamRecord(id=7, name="Man City"),
                TeamRecord(id=8, name="Crystal Palace"),
                TeamRecord(id=9, name="Coventry"),
            ],
        )
        repo.upsert_fixtures(
            conn,
            [
                FixtureRecord(id=70, event=2, team_h=2, team_a=1, started=1, finished=1, raw_json={}),
                FixtureRecord(id=71, event=2, team_h=5, team_a=4, started=1, finished=1, raw_json={}),
                FixtureRecord(id=72, event=2, team_h=8, team_a=7, started=1, finished=1, raw_json={}),
                FixtureRecord(id=73, event=3, team_h=1, team_a=3, started=0, finished=0, raw_json={}),
                FixtureRecord(id=74, event=3, team_h=6, team_a=4, started=0, finished=0, raw_json={}),
                FixtureRecord(id=75, event=3, team_h=7, team_a=9, started=0, finished=0, raw_json={}),
            ],
        )

    first_fixture = {
        team_id: fixture_outlook(conn, team_id, 2, 3)["fixtures"][0]
        for team_id in (1, 4, 7)
    }
    assert (first_fixture[1]["opponent_team"], first_fixture[1]["home"]) == (3, True)
    assert (first_fixture[4]["opponent_team"], first_fixture[4]["home"]) == (6, False)
    assert (first_fixture[7]["opponent_team"], first_fixture[7]["home"]) == (9, True)
    conn.close()


def test_future_outlook_keeps_remaining_current_gameweek_fixture(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    with conn:
        repo.upsert_teams(conn, [TeamRecord(id=1, name="One"), TeamRecord(id=2, name="Two")])
        repo.upsert_fixtures(
            conn,
            [
                FixtureRecord(id=30, event=2, team_h=1, team_a=2, started=1, finished=1, raw_json={}),
                FixtureRecord(id=31, event=2, team_h=2, team_a=1, started=0, finished=0, raw_json={}),
                FixtureRecord(id=32, event=3, team_h=1, team_a=2, started=0, finished=0, raw_json={}),
            ],
        )
    outlook = fixture_outlook(conn, 1, 2, 2)
    assert [item["fixture_id"] for item in outlook["fixtures"]] == [31, 32]
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


def test_transfer_affordability_uses_selling_prices_and_official_market_prices(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    with conn:
        repo.upsert_players(
            conn,
            [
                PlayerRecord(id=10, web_name="Maguire", full_name="Harry Maguire"),
                PlayerRecord(id=11, web_name="Rodon", full_name="Joe Rodon"),
                PlayerRecord(id=20, web_name="DeCuyper", full_name="Maxim De Cuyper"),
                PlayerRecord(id=21, web_name="Other", full_name="Other Defender"),
            ],
        )
        run = repo.create_fetch_run(conn, "fetch_fpl")
        repo.insert_snapshots(
            conn,
            [
                PlayerSnapshotRecord(player_id=10, captured_at="2026-09-07T00:00:00Z", now_cost=51, raw_json={}),
                PlayerSnapshotRecord(player_id=11, captured_at="2026-09-07T00:00:00Z", now_cost=45, raw_json={}),
                PlayerSnapshotRecord(player_id=20, captured_at="2026-09-07T00:00:00Z", now_cost=47, raw_json={}),
                PlayerSnapshotRecord(player_id=21, captured_at="2026-09-07T00:00:00Z", now_cost=47, raw_json={}),
            ],
            run,
        )
        repo.upsert_manual_manager_state(conn, 241392, 4, 3, 0, captured_at="2026-09-07T12:00:00Z")
        repo.upsert_manager_selling_prices(
            conn,
            241392,
            4,
            {10: 50, 11: 44},
            captured_at="2026-09-07T12:00:00Z",
        )

    one_for_one = transfer_affordability(conn, 241392, 4, [10], [20])
    assert one_for_one["status"] == "AFFORDABLE"
    assert one_for_one["bank"] == 0
    assert one_for_one["selling_prices"] == {10: 50}
    assert one_for_one["official_market_prices"] == {20: 47}
    assert one_for_one["available_funds"] == 3

    insufficient = transfer_affordability(conn, 241392, 4, [11], [20])
    assert insufficient["status"] == "INSUFFICIENT_FUNDS"
    assert insufficient["available_funds"] == -3
    assert insufficient["shortfall"] == 3

    two_for_two = transfer_affordability(conn, 241392, 4, [10, 11], [20, 21])
    assert two_for_two["status"] == "AFFORDABLE"
    assert two_for_two["available_funds"] == 0
    assert two_for_two["outgoing_selling_value"] == 94
    assert two_for_two["incoming_market_value"] == 94
    conn.close()


def test_transfer_affordability_does_not_fallback_to_official_price_for_missing_selling_price(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    with conn:
        repo.upsert_players(
            conn,
            [
                PlayerRecord(id=30, web_name="Outgoing", full_name="Outgoing Player"),
                PlayerRecord(id=31, web_name="Incoming", full_name="Incoming Player"),
            ],
        )
        run = repo.create_fetch_run(conn, "fetch_fpl")
        repo.insert_snapshots(
            conn,
            [
                PlayerSnapshotRecord(player_id=30, captured_at="2026-09-07T00:00:00Z", now_cost=50, raw_json={}),
                PlayerSnapshotRecord(player_id=31, captured_at="2026-09-07T00:00:00Z", now_cost=47, raw_json={}),
            ],
            run,
        )
        repo.upsert_manual_manager_state(conn, 241392, 4, 3, 0, captured_at="2026-09-07T12:00:00Z")

    result = transfer_affordability(conn, 241392, 4, [30], [31])
    assert result["status"] == "DATA_GAP"
    assert result["available_funds"] is None
    assert any("selling price unavailable" in message for message in result["data_gaps"])
    conn.close()


def test_selling_price_formula_and_stale_manual_snapshot(tmp_path):
    assert [calculate_selling_price(55, market) for market in (55, 56, 57, 58, 59)] == [55, 55, 56, 56, 57]
    assert [calculate_selling_price(55, market) for market in (55, 54, 53)] == [55, 54, 53]
    conn = connect_database(tmp_path / "fpl.db")
    with conn:
        repo.upsert_players(conn, [PlayerRecord(id=1, web_name="Player", full_name="Player")])
        run = repo.create_fetch_run(conn, "fetch_fpl")
        repo.insert_snapshots(conn, [PlayerSnapshotRecord(player_id=1, captured_at="2026-09-01T00:00:00Z", now_cost=57, raw_json={})], run)
        repo.insert_manager_acquisition(conn, 99, 1, 1, 55, source="verified_initial_squad")
        repo.upsert_manager_selling_prices(conn, 99, 4, {1: 56}, market_prices_at_capture={1: 57}, captured_at="2026-09-01T12:00:00Z")
    assert effective_selling_price(conn, 99, 4, 1)["status"] == "VERIFIED_BY_MANUAL"
    with conn:
        run = repo.create_fetch_run(conn, "fetch_fpl")
        repo.insert_snapshots(conn, [PlayerSnapshotRecord(player_id=1, captured_at="2026-09-02T00:00:00Z", now_cost=59, raw_json={})], run)
    value = effective_selling_price(conn, 99, 4, 1)
    assert value["status"] == "AUTO_CALCULATED"
    assert value["calculated_selling_price"] == 57
    assert value["effective_selling_price"] == 57
    conn.close()


def test_same_market_manual_mismatch_is_not_silently_overridden(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    with conn:
        repo.upsert_players(conn, [PlayerRecord(id=1, web_name="Player", full_name="Player")])
        run = repo.create_fetch_run(conn, "fetch_fpl")
        repo.insert_snapshots(conn, [PlayerSnapshotRecord(player_id=1, captured_at="2026-09-01T00:00:00Z", now_cost=57, raw_json={})], run)
        repo.insert_manager_acquisition(conn, 99, 1, 1, 55, source="verified_initial_squad")
        repo.upsert_manager_selling_prices(conn, 99, 4, {1: 55}, market_prices_at_capture={1: 57})
    result = effective_selling_price(conn, 99, 4, 1)
    assert result["status"] == "MISMATCH"
    assert result["effective_selling_price"] is None
    conn.close()


def test_manager_planning_state_precedence_is_event_scoped(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    with conn:
        repo.insert_manager_state(
            conn,
            EntryRecord(entry_id=99, bank=30, team_value=1000, total_transfers=1),
            HistoryRow(event=4, bank=30, total_transfers=1),
            None,
            1,
            None,
            4,
            {},
            "2026-09-07T00:00:00Z",
        )
        repo.upsert_manual_manager_state(conn, 99, 4, 3, 0, captured_at="2026-09-07T12:00:00Z")
    manual = repo.manager_planning_state(conn, 99, 4)
    assert manual["free_transfers"] == 3 and manual["free_transfers_source"] == "manual"
    assert manual["bank"] == 0 and manual["bank_source"] == "manual"
    with conn:
        repo.upsert_manual_manager_state(conn, 99, 4, None, None, captured_at="2026-09-07T13:00:00Z")
    unavailable = repo.manager_planning_state(conn, 99, 4)
    assert unavailable["free_transfers"] is None and unavailable["bank"] is None
    with conn:
        conn.execute("DELETE FROM manager_manual_state WHERE entry_id=99 AND event=4")
    legacy = repo.manager_planning_state(conn, 99, 4)
    assert legacy["free_transfers"] == 1 and legacy["free_transfers_source"] == "legacy_manager_state"
    assert legacy["bank"] == 30 and legacy["bank_source"] == "manager_state"
    conn.close()


def test_price_change_updates_realisable_value_and_affordability_without_new_manual_input(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    with conn:
        repo.upsert_positions(conn, [PositionRecord(id=1, singular_name_short="GKP")])
        repo.upsert_players(
            conn,
            [PlayerRecord(id=player_id, web_name=f"P{player_id}", full_name=f"Player {player_id}", element_type=1) for player_id in range(1, 16)]
            + [PlayerRecord(id=20, web_name="Incoming", full_name="Incoming", element_type=1)],
        )
        run = repo.create_fetch_run(conn, "fetch_fpl")
        repo.insert_snapshots(
            conn,
            [PlayerSnapshotRecord(player_id=player_id, captured_at="2026-09-01T00:00:00Z", now_cost=57 if player_id == 1 else 55, raw_json={}) for player_id in range(1, 16)]
            + [PlayerSnapshotRecord(player_id=20, captured_at="2026-09-01T00:00:00Z", now_cost=57, raw_json={})],
            run,
        )
        repo.upsert_squad_picks(conn, 99, 4, [PickRecord(player_id=player_id, position=player_id, raw_json={}) for player_id in range(1, 16)])
        for player_id in range(1, 16):
            repo.insert_manager_acquisition(conn, 99, player_id, 1, 55, source="verified_initial_squad")
        repo.upsert_manual_manager_state(conn, 99, 4, 3, 0)
        repo.upsert_manager_selling_prices(conn, 99, 4, {1: 56}, market_prices_at_capture={1: 57})
    old_value = realisable_squad_value(conn, 99, 4)
    assert old_value["realisable_selling_value"] == 56 + (14 * 55)
    with conn:
        run = repo.create_fetch_run(conn, "fetch_fpl")
        repo.insert_snapshots(conn, [PlayerSnapshotRecord(player_id=1, captured_at="2026-09-02T00:00:00Z", now_cost=59, raw_json={})], run)
    new_value = realisable_squad_value(conn, 99, 4)
    assert new_value["realisable_selling_value"] == 57 + (14 * 55)
    route = transfer_affordability(conn, 99, 4, [1], [20])
    assert route["status"] == "AFFORDABLE"
    assert route["selling_prices"] == {1: 57}
    assert route["available_funds"] == 0
    conn.close()
