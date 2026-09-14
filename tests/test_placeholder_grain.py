"""Fixture-grain data health: DGW, blank GWs, reschedules, stale placeholders.

The canonical performance grain is `(player_id, event, fixture_id)`.  These
tests prove the grain never loses per-fixture facts, never abandons a blank
gameweek, and cannot be corrupted by schedule placeholders left behind by an
older official payload (for example after a player's club context changed).
"""

from __future__ import annotations

from fpl_brain import repositories as repo
from fpl_brain.database import connect_database
from fpl_brain.models import (
    EventRecord,
    FixtureRecord,
    PlayerGameweekRecord,
    PlayerRecord,
    TeamRecord,
)


def _seed_world(conn):
    with conn:
        repo.upsert_teams(
            conn,
            [
                TeamRecord(id=1, name="One"),
                TeamRecord(id=2, name="Two"),
                TeamRecord(id=3, name="Three"),
                TeamRecord(id=10, name="Ten"),
                TeamRecord(id=11, name="Eleven"),
            ],
        )
        repo.upsert_players(conn, [PlayerRecord(id=10, web_name="DGW Player", full_name="DGW Player", team_id=1)])
        repo.upsert_events(
            conn,
            [
                EventRecord(id=1, finished=1, data_checked=1, raw_json={}),
                EventRecord(id=2, finished=1, data_checked=1, raw_json={}),
                EventRecord(id=3, finished=0, data_checked=0, raw_json={}),
            ],
        )
        # Team 1 plays one fixture in GW2, one in GW3, and a DGW pair in GW1.
        repo.upsert_fixtures(
            conn,
            [
                FixtureRecord(id=101, event=1, team_h=1, team_a=2, finished=1, started=1, raw_json={}),
                FixtureRecord(id=102, event=1, team_h=2, team_a=1, finished=1, started=1, raw_json={}),
                FixtureRecord(id=103, event=2, team_h=1, team_a=2, finished=1, started=1, raw_json={}),
                FixtureRecord(id=104, event=3, team_h=1, team_a=2, finished=0, raw_json={}),
            ],
        )


def test_dgw_player_two_fixture_rows_are_both_preserved(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    _seed_world(conn)
    rows = [
        PlayerGameweekRecord(
            player_id=10, event=1, fixture_id=101, minutes=90, total_points=6,
            expected_goals=0.4, expected_assists=0.1, source="element_summary", raw_json={},
        ),
        PlayerGameweekRecord(
            player_id=10, event=1, fixture_id=102, minutes=75, total_points=3,
            expected_goals=0.2, expected_assists=0.0, source="element_summary", raw_json={},
        ),
    ]
    with conn:
        repo.upsert_player_gameweeks(conn, rows + rows)
    stored = conn.execute(
        "SELECT fixture_id, minutes, total_points FROM player_gameweeks WHERE player_id=10 AND event=1 ORDER BY fixture_id"
    ).fetchall()
    assert [(row[0], row[1], row[2]) for row in stored] == [(101, 90, 6), (102, 75, 3)]
    completed = repo.completed_player_fixture_rows(conn, player_id=10, event=1)
    assert sorted(row["fixture_id"] for row in completed) == [101, 102]
    # Per-fixture xG must remain individually addressable (never merged).
    assert abs(sum(row["expected_goals"] for row in completed) - 0.6) < 1e-9
    conn.close()


def test_bgw_event_stores_no_performance_rows(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    _seed_world(conn)
    # A blank GW for team 1 has no stored fixture and therefore no rows:
    # the completed-performance boundary stays empty instead of inventing one.
    assert repo.completed_player_fixture_rows(conn, player_id=10, event=99) == []
    sentinel = PlayerGameweekRecord(player_id=10, event=2, fixture_id=-1, minutes=0, total_points=0, source="event_live", raw_json={})
    with conn:
        repo.upsert_player_gameweeks(conn, [sentinel])
        # A real completed summary later replaces the sentinel, never coexists.
        repo.upsert_player_gameweeks(
            conn,
            [PlayerGameweekRecord(player_id=10, event=2, fixture_id=103, minutes=90, total_points=4, source="element_summary", raw_json={})],
        )
    count = conn.execute(
        "SELECT COUNT(*) FROM player_gameweeks WHERE player_id=10 AND event=2 AND fixture_id=-1"
    ).fetchone()[0]
    assert count == 0
    assert repo.completed_player_fixture_rows(conn, player_id=10, event=2)[0]["fixture_id"] == 103
    conn.close()


def test_future_placeholder_never_downgrades_performance(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    _seed_world(conn)
    with conn:
        repo.upsert_player_gameweeks(
            conn,
            [PlayerGameweekRecord(player_id=10, event=2, fixture_id=103, minutes=90, total_points=4, source="element_summary", raw_json={})],
        )
        # GW3 schedule placeholder (fixture unfinished) must not become an appearance.
        repo.upsert_player_gameweeks(
            conn,
            [PlayerGameweekRecord(player_id=10, event=3, fixture_id=104, minutes=0, source="element_summary", raw_json={})],
        )
    assert repo.completed_player_fixture_rows(conn, player_id=10, event=3) == []
    assert len(repo.completed_player_fixture_rows(conn, player_id=10, event=2)) == 1
    conn.close()


def test_rescheduled_fixture_placeholder_grain_migrates(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    _seed_world(conn)
    with conn:
        # Old payload claimed fixture 104 in GW3 (pending); officially it is
        # later reassigned to GW4. The new payload claims (4, 104).
        repo.upsert_player_gameweeks(
            conn,
            [PlayerGameweekRecord(player_id=10, event=3, fixture_id=104, minutes=0, source="element_summary", raw_json={})],
        )
        repo.upsert_fixtures(
            conn,
            [FixtureRecord(id=104, event=4, team_h=1, team_a=2, finished=0, raw_json={})],
        )
        repo.prune_stale_element_summary_placeholders(conn, 10, {(4, 104)})
        repo.upsert_player_gameweeks(
            conn,
            [PlayerGameweekRecord(player_id=10, event=4, fixture_id=104, minutes=0, source="element_summary", raw_json={})],
        )
    assert conn.execute(
        "SELECT COUNT(*) FROM player_gameweeks WHERE player_id=10 AND event=3 AND fixture_id=104"
    ).fetchone()[0] == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM player_gameweeks WHERE player_id=10 AND event=4 AND fixture_id=104"
    ).fetchone()[0] == 1
    conn.close()


def test_club_change_leaves_no_stale_orphan_placeholders(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    _seed_world(conn)
    with conn:
        # Older syncs stored team 2 placeholders from a previous club context.
        repo.upsert_fixtures(
            conn,
            [
                FixtureRecord(id=201, event=2, team_h=2, team_a=10, finished=1, raw_json={}),
                FixtureRecord(id=202, event=3, team_h=2, team_a=11, finished=0, raw_json={}),
            ],
        )
        repo.upsert_fixtures(
            conn,
            [FixtureRecord(id=102, event=2, team_h=1, team_a=2, finished=1, started=1, raw_json={})],
        )
        repo.upsert_player_gameweeks(
            conn,
            [
                PlayerGameweekRecord(player_id=10, event=2, fixture_id=201, minutes=0, source="element_summary", raw_json={}),
                PlayerGameweekRecord(player_id=10, event=3, fixture_id=202, minutes=0, source="element_summary", raw_json={}),
                # Real performance row for the current club is never pruned even unclaimed.
                PlayerGameweekRecord(player_id=10, event=2, fixture_id=102, minutes=90, total_points=4, source="element_summary", raw_json={}),
            ],
        )
        changed = repo.prune_stale_element_summary_placeholders(conn, 10, {(3, 103)})
    assert changed >= 2
    assert conn.execute(
        "SELECT COUNT(*) FROM player_gameweeks WHERE player_id=10 AND fixture_id IN (201, 202)"
    ).fetchone()[0] == 0
    # The performance row survives and a legitimate current-club placeholder stays.
    assert conn.execute(
        "SELECT COUNT(*) FROM player_gameweeks WHERE player_id=10 AND fixture_id=102"
    ).fetchone()[0] == 1
    conn.close()


def test_claimed_placeholder_and_blob_rows_are_never_pruned(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    _seed_world(conn)
    with conn:
        repo.upsert_player_gameweeks(
            conn,
            [
                PlayerGameweekRecord(player_id=10, event=3, fixture_id=104, minutes=0, source="element_summary", raw_json={}),
                PlayerGameweekRecord(player_id=10, event=4, fixture_id=-1, minutes=0, total_points=0, source="event_live", raw_json={}),
                PlayerGameweekRecord(player_id=10, event=3, fixture_id=None, minutes=0, total_points=0, source="element_summary", raw_json={}),
            ],
        )
        deleted = repo.prune_stale_element_summary_placeholders(conn, 10, {(3, 104)})
    assert deleted == 0
    conn.close()


# ---------------------------------------------------------------------------
# F1 — a started fixture's placeholder must not vanish without replacement.
#
# A placeholder holds no observation, so its deletion normally only removes a
# stale *slot*.  Once the fixture has started the slot is the sole surviving
# evidence that the player-fixture observation is still owed, and a row that does
# not exist cannot be reported as a gap -- the certified-history audit scans rows,
# not absences.  These three cases pin the replacement-before-delete rule.
# ---------------------------------------------------------------------------


def _placeholder_row(conn, event, fixture_id):
    with conn:
        repo.upsert_player_gameweeks(
            conn,
            [PlayerGameweekRecord(player_id=10, event=event, fixture_id=fixture_id, minutes=0,
                                  source="element_summary", raw_json={})],
        )


def _row_count(conn, event, fixture_id):
    return conn.execute(
        "SELECT COUNT(*) FROM player_gameweeks WHERE player_id=10 AND event=? AND fixture_id=?",
        (event, fixture_id),
    ).fetchone()[0]


def test_started_current_club_placeholder_without_replacement_is_retained(tmp_path):
    """Case C: fixture 103 (event 2, finished+started) is the player's own club.

    The payload now labels it under event 2 and offers NO replacement row, so the
    stale event-3 slot must survive -- otherwise the completed GW2 observation
    becomes invisible to the audit.
    """

    conn = connect_database(tmp_path / "fpl.db")
    _seed_world(conn)
    _placeholder_row(conn, 3, 103)
    with conn:
        deleted = repo.prune_stale_element_summary_placeholders(conn, 10, {(2, 103)})
    assert deleted == 0, "a started current-club placeholder must not be pruned without replacement"
    assert _row_count(conn, 3, 103) == 1
    conn.close()


def test_started_current_club_placeholder_with_replacement_is_pruned(tmp_path):
    """Case B: the same stale label may go once the observation is represented."""

    conn = connect_database(tmp_path / "fpl.db")
    _seed_world(conn)
    _placeholder_row(conn, 3, 103)
    _placeholder_row(conn, 2, 103)          # authoritative replacement at the claimed pair
    with conn:
        deleted = repo.prune_stale_element_summary_placeholders(conn, 10, {(2, 103)})
    assert deleted == 1
    assert _row_count(conn, 3, 103) == 0
    assert _row_count(conn, 2, 103) == 1
    conn.close()


def test_unstarted_placeholder_is_pruned_normally(tmp_path):
    """Case A: fixture 104 has not started, so the slot is not yet owed."""

    conn = connect_database(tmp_path / "fpl.db")
    _seed_world(conn)
    _placeholder_row(conn, 3, 104)
    with conn:
        repo.upsert_fixtures(conn, [FixtureRecord(id=104, event=4, team_h=1, team_a=2, finished=0, raw_json={})])
        deleted = repo.prune_stale_element_summary_placeholders(conn, 10, {(4, 104)})
    assert deleted == 1
    assert _row_count(conn, 3, 104) == 0
    conn.close()


def test_started_placeholder_for_another_club_is_still_pruned(tmp_path):
    """A former club's slot is not this player's required evidence; it may go."""

    conn = connect_database(tmp_path / "fpl.db")
    _seed_world(conn)
    with conn:
        # fixture 301 involves teams 2 and 3; the player is at team 1.
        repo.upsert_fixtures(conn, [FixtureRecord(id=301, event=2, team_h=2, team_a=3,
                                                  finished=1, started=1, raw_json={})])
    _placeholder_row(conn, 2, 301)
    with conn:
        deleted = repo.prune_stale_element_summary_placeholders(conn, 10, {(3, 103)})
    assert deleted == 1
    assert _row_count(conn, 2, 301) == 0
    conn.close()
