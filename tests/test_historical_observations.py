"""PE-1 — the point-in-time historical boundary.

Cases A–G from the phase brief, plus the observability clause the old predicate
lacked.  Seeded with explicit timestamps so each clause can be isolated.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fpl_brain import historical_observations as ho
from fpl_brain.database import connect_database

def _plus_hours(stamp: str, hours: int) -> str:
    from datetime import timedelta

    from fpl_brain.utils import parse_utc, utc_now

    parsed = parse_utc(stamp)
    return (parsed + timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ") if parsed is not None else stamp


EARLY = "2026-08-20T00:00:00Z"     # before everything
WRITTEN = "2026-09-10T20:39:41Z"   # when the placeholder rows were written
LATE = "2026-09-16T00:00:00Z"      # after everything


def _seed(conn, *, fixtures, observations, event=5):
    """Seed fixtures and player_gameweeks rows with explicit timestamps."""

    with conn:
        from fpl_brain.models import PlayerRecord, PositionRecord, TeamRecord
        from fpl_brain import repositories as repo

        repo.upsert_positions(conn, [
            PositionRecord(id=1, singular_name="Goalkeeper", singular_name_short="GKP"),
            PositionRecord(id=2, singular_name="Defender", singular_name_short="DEF"),
            PositionRecord(id=3, singular_name="Midfielder", singular_name_short="MID"),
            PositionRecord(id=4, singular_name="Forward", singular_name_short="FWD"),
        ])

        repo.upsert_teams(conn, [TeamRecord(id=t, name=f"Team {t}") for t in (1, 2, 3, 4)])

        repo.upsert_players(conn, [
            PlayerRecord(id=int(row["player_id"]), web_name=f"P{int(row['player_id'])}",
                         full_name=f"Player {int(row['player_id'])}", team_id=1, element_type=3)
            for row in observations
        ])
        for row in fixtures:
            conn.execute(
                """INSERT INTO fixtures(id, event, team_h, team_a, kickoff_time, started, finished,
                   raw_json, updated_at) VALUES (?,?,?,?,?,?,?,?,?)""",
                (row["id"], row["event"], row.get("team_h", 1), row.get("team_a", 2),
                 row["kickoff_time"], row.get("started", 1), row.get("finished", 1), "{}",
                 "2026-09-01T00:00:00Z"),
            )
        kickoff_of = {int(f["id"]): str(f["kickoff_time"]) for f in fixtures}
        for row in observations:
            # A realistic write time: the post-match fetch, three hours after
            # kickoff, unless the test deliberately sets one to isolate a clause.
            written = row.get("updated_at") or _plus_hours(kickoff_of[int(row["fixture_id"])], 3)
            conn.execute(
                """INSERT INTO player_gameweeks(player_id, event, fixture_id, minutes, starts,
                   total_points, expected_goals, defensive_contribution, source, updated_at, raw_json)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (row["player_id"], row["event"], row["fixture_id"], row.get("minutes"),
                 row.get("starts"), row.get("total_points"), row.get("expected_goals"),
                 row.get("defensive_contribution"), row.get("source", "element_summary"),
                 written, "{}"),
            )


@pytest.fixture
def conn(tmp_path):
    connection = connect_database(tmp_path / "pe1.db")
    try:
        yield connection
    finally:
        connection.close()


def _read(conn, as_of, planning_event=None):
    return ho.historical_player_fixtures(conn, as_of=as_of, planning_event=planning_event)


# ---------------------------------------------------------------------------
# CASE A / B — future placeholder
# ---------------------------------------------------------------------------


def test_case_A_a_future_fixture_placeholder_is_excluded(conn):
    """Kickoff after as_of, 0-minute row: not evidence."""

    _seed(
        conn,
        fixtures=[{"id": 10, "event": 5, "kickoff_time": "2026-09-20T14:00:00Z", "started": 0, "finished": 0}],
        observations=[{"player_id": 1, "event": 5, "fixture_id": 10, "minutes": 0, "updated_at": WRITTEN}],
    )
    assert _read(conn, LATE, 6) == []


def test_case_B_the_live_pre_round_placeholder_on_a_finished_fixture_is_excluded(conn):
    """THE MEASURED HAZARD: the fixture finished, the row was never refreshed.

    This is the real live-store shape — written 2026-09-10 for a fixture that
    kicked off 2026-09-12 — and the old predicate accepted it.
    """

    _seed(
        conn,
        fixtures=[{"id": 36, "event": 4, "kickoff_time": "2026-09-12T14:00:00Z"}],
        observations=[{"player_id": 1, "event": 4, "fixture_id": 36, "minutes": 0,
                       "total_points": None, "updated_at": "2026-09-10T20:39:41Z"}],
    )
    # The fixture IS finished, so the old predicate would accept it.
    assert conn.execute(
        "SELECT COUNT(*) FROM player_gameweeks pg JOIN fixtures f ON f.id=pg.fixture_id "
        "WHERE f.finished=1 AND f.started=1 AND (f.kickoff_time IS NULL OR f.kickoff_time <= ?)",
        (LATE,),
    ).fetchone()[0] == 1
    # The boundary does not.
    assert _read(conn, LATE, 5) == []


def test_the_write_before_kickoff_clause_is_independent_of_the_value_signature(conn):
    """A row written BEFORE kickoff is not evidence whatever its columns hold."""

    _seed(
        conn,
        fixtures=[{"id": 40, "event": 4, "kickoff_time": "2026-09-12T14:00:00Z"}],
        observations=[{"player_id": 1, "event": 4, "fixture_id": 40, "minutes": 77,
                       "starts": 1, "total_points": 5, "expected_goals": 0.4,
                       "updated_at": "2026-09-11T20:00:00Z"}],   # before kickoff
    )
    assert _read(conn, LATE, 5) == []


# ---------------------------------------------------------------------------
# CASE C / D — genuine observations
# ---------------------------------------------------------------------------


def test_case_C_a_completed_starter_before_as_of_is_included(conn):
    _seed(
        conn,
        fixtures=[{"id": 20, "event": 2, "kickoff_time": "2026-08-29T14:00:00Z"}],
        observations=[{"player_id": 2, "event": 2, "fixture_id": 20, "minutes": 90, "starts": 1,
                       "total_points": 6, "expected_goals": 0.5, "updated_at": WRITTEN}],
    )
    rows = _read(conn, LATE, 5)
    assert len(rows) == 1 and rows[0]["minutes"] == 90


def test_case_D_a_genuine_zero_minute_non_appearance_is_INCLUDED(conn):
    """A real DNP is an observation and must never be erased.

    This is the clause that forbids scattering ``WHERE minutes > 0``.
    """

    _seed(
        conn,
        fixtures=[{"id": 21, "event": 2, "kickoff_time": "2026-08-29T14:00:00Z"}],
        observations=[{"player_id": 3, "event": 2, "fixture_id": 21, "minutes": 0, "starts": 0,
                       "total_points": 0, "expected_goals": 0.0, "updated_at": WRITTEN}],
    )
    rows = _read(conn, LATE, 5)
    assert len(rows) == 1, "a genuine non-appearance was erased"
    assert rows[0]["minutes"] == 0 and rows[0]["total_points"] == 0


def test_case_D2_zero_minutes_with_no_performance_columns_is_a_placeholder(conn):
    """The discriminator: the same minutes, but no official observation at all."""

    _seed(
        conn,
        fixtures=[{"id": 22, "event": 2, "kickoff_time": "2026-08-29T14:00:00Z"}],
        observations=[{"player_id": 4, "event": 2, "fixture_id": 22, "minutes": 0,
                       "total_points": None, "updated_at": WRITTEN}],
    )
    assert _read(conn, LATE, 5) == []


# ---------------------------------------------------------------------------
# CASE E — same nominal Gameweek, kickoff after as_of
# ---------------------------------------------------------------------------


def test_case_E_a_same_event_fixture_kicking_off_after_as_of_is_excluded(conn):
    """Event numbering is not a clock: the cutoff is."""

    _seed(
        conn,
        fixtures=[
            {"id": 30, "event": 3, "kickoff_time": "2026-09-04T19:00:00Z"},
            {"id": 31, "event": 3, "kickoff_time": "2026-09-06T15:30:00Z"},
        ],
        observations=[
            {"player_id": 5, "event": 3, "fixture_id": 30, "minutes": 90, "total_points": 4},
            {"player_id": 5, "event": 3, "fixture_id": 31, "minutes": 90, "total_points": 9},
        ],
    )
    rows = _read(conn, "2026-09-05T00:00:00Z", 4)
    assert [r["fixture_id"] for r in rows] == [30], "a fixture after the cutoff leaked in"


# ---------------------------------------------------------------------------
# CASE F — a double Gameweek, one fixture before and one after
# ---------------------------------------------------------------------------


def test_case_F_a_double_gameweek_only_exposes_the_causally_available_fixture(conn):
    _seed(
        conn,
        fixtures=[
            {"id": 50, "event": 3, "kickoff_time": "2026-09-05T14:00:00Z"},
            {"id": 51, "event": 3, "kickoff_time": "2026-09-09T19:00:00Z"},
        ],
        observations=[
            {"player_id": 6, "event": 3, "fixture_id": 50, "minutes": 90, "total_points": 3},
            {"player_id": 6, "event": 3, "fixture_id": 51, "minutes": 90, "total_points": 7},
        ],
    )
    rows = _read(conn, "2026-09-07T00:00:00Z", 4)
    assert [r["fixture_id"] for r in rows] == [50]
    both = _read(conn, "2026-09-10T00:00:00Z", 4)
    assert sorted(r["fixture_id"] for r in both) == [50, 51]


# ---------------------------------------------------------------------------
# CASE G — a rescheduled fixture does not leak through old event numbering
# ---------------------------------------------------------------------------


def test_case_G_a_postponed_fixture_does_not_leak_by_event_number(conn):
    """The row's event is EARLY but its kickoff is LATE: the clock decides."""

    _seed(
        conn,
        fixtures=[{"id": 60, "event": 2, "kickoff_time": "2026-09-15T19:00:00Z"}],
        observations=[{"player_id": 7, "event": 2, "fixture_id": 60, "minutes": 90,
                       "total_points": 8, "updated_at": "2026-09-15T22:00:00Z"}],
    )
    # Event 2 is well before the cutoff, and the old predicate would accept it.
    assert _read(conn, "2026-09-14T00:00:00Z", 5) == []
    # Once it has actually been played, it becomes evidence.
    assert len(_read(conn, "2026-09-16T00:00:00Z", 5)) == 1


# ---------------------------------------------------------------------------
# AS-OF IS REQUIRED — fail closed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("missing", [None, "", "   "])
def test_the_boundary_refuses_without_an_explicit_as_of(conn, missing):
    with pytest.raises(ho.HistoricalBoundaryError):
        ho.historical_player_fixtures(conn, as_of=missing)
    with pytest.raises(ho.HistoricalBoundaryError):
        ho.historical_player_fixture_rows(conn, as_of=missing)
    with pytest.raises(ho.HistoricalBoundaryError):
        ho.is_historical_observation({"fixture_kickoff": WRITTEN, "updated_at": WRITTEN}, as_of=missing)


def test_a_row_written_after_the_cutoff_is_not_visible_at_the_cutoff(conn):
    """The observability clause: what we know now is not what we knew then."""

    _seed(
        conn,
        fixtures=[{"id": 70, "event": 2, "kickoff_time": "2026-08-29T14:00:00Z"}],
        observations=[{"player_id": 8, "event": 2, "fixture_id": 70, "minutes": 90,
                       "total_points": 5, "updated_at": "2026-09-10T20:39:41Z"}],
    )
    assert _read(conn, "2026-09-01T00:00:00Z", 5) == []      # written later
    assert len(_read(conn, "2026-09-11T00:00:00Z", 5)) == 1


def test_one_predicate_definition_is_exposed_for_evidence(conn):
    window = ho.ObservationWindow(as_of=LATE, planning_event=5)
    assert "updated_at >= kickoff_time" in window.clauses
    assert "updated_at <= as_of" in window.clauses
    assert window.as_dict()["boundary_version"] == ho.HISTORICAL_OBSERVATION_BOUNDARY_VERSION
