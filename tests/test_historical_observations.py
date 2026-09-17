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


# ---------------------------------------------------------------------------
# CONSUMERS — a model prior reader must not re-derive the window locally
# ---------------------------------------------------------------------------


def test_the_minutes_position_priors_use_the_boundary_not_a_local_predicate(conn):
    """``positional_pooled_engagement`` feeds the level-4 minutes priors.

    Its old predicate pruned on event number and kickoff time but never asked
    when the row was written, so a scheduled placeholder written before its own
    kickoff was read as an observation.  The two rows share one position, so the
    survivor proves the exclusion is the boundary clause and not an empty window.
    """

    from fpl_brain import analytics

    _seed(
        conn,
        fixtures=[
            # Its row is written after its own kickoff: causally valid.
            {"id": 60, "event": 2, "kickoff_time": "2026-08-29T14:00:00Z"},
            # Its row is written BEFORE its own kickoff: a scheduled placeholder.
            {"id": 61, "event": 3, "kickoff_time": "2026-09-05T14:00:00Z"},
        ],
        observations=[
            {"player_id": 5, "event": 2, "fixture_id": 60, "minutes": 90, "starts": 1,
             "total_points": 6, "updated_at": WRITTEN},
            {"player_id": 6, "event": 3, "fixture_id": 61, "minutes": 0, "starts": 0,
             "total_points": 0, "updated_at": "2026-09-05T09:00:00Z"},
        ],
    )

    pools = analytics.positional_pooled_engagement(conn, 5, LATE)
    assert 3 in pools, "the causally valid starter must still reach the priors"
    assert pools[3]["start_rows"] == 1, "the pre-kickoff placeholder was counted as an observation"
    assert pools[3]["p_start"] == 1.0, "the pre-kickoff placeholder deflated the start-rate prior to 0.5"
    assert pools[3]["minutes_if_start"] == 90.0

    # A cutoff before either row was written leaves the priors empty: the
    # boundary is a clock, not a shape, and it fails closed.
    assert analytics.positional_pooled_engagement(conn, 5, EARLY) == {}


def test_a_row_written_after_the_cutoff_never_reaches_the_minutes_priors(conn):
    """The look-ahead gate.

    The row's fixture has kicked off and finished before the cutoff, so the old
    local predicate accepted it; only its write time is in the future relative
    to the planning instant.  It must not be readable as history.
    """

    from fpl_brain import analytics

    _seed(
        conn,
        fixtures=[{"id": 62, "event": 2, "kickoff_time": "2026-08-29T14:00:00Z"}],
        observations=[
            {"player_id": 7, "event": 2, "fixture_id": 62, "minutes": 90, "starts": 1,
             "total_points": 9, "updated_at": "2026-09-30T00:00:00Z"},
        ],
    )

    # The row's own fixture state satisfies every clause the old predicate had.
    accepted_by_old = conn.execute(
        """SELECT COUNT(*) FROM player_gameweeks pg JOIN fixtures f ON f.id=pg.fixture_id
            WHERE f.finished=1 AND f.started=1 AND pg.event<? AND f.event<?
              AND f.event IS NOT NULL AND pg.starts IS NOT NULL AND pg.minutes IS NOT NULL
              AND (f.kickoff_time IS NULL OR f.kickoff_time <= ?)""",
        (5, 5, LATE),
    ).fetchone()[0]
    assert accepted_by_old == 1, "the old predicate is expected to accept this row"

    # The reader must not.
    assert analytics.positional_pooled_engagement(conn, 5, LATE) == {}


def test_the_naive_minutes_baseline_uses_the_boundary_not_a_local_predicate(conn):
    """``positional_pooled_minutes`` feeds the naive baseline projection run.

    Its old predicate had no write-time clause and no ``starts`` filter, so the
    scheduled placeholders were averaged in as genuine zero-minute appearances.
    Against the certified data that understated every positional mean by 4.8 to
    8.9 minutes, because all 456 placeholders passed every clause it had.
    """

    from fpl_brain import analytics

    _seed(
        conn,
        fixtures=[
            {"id": 70, "event": 2, "kickoff_time": "2026-08-29T14:00:00Z"},
            {"id": 71, "event": 3, "kickoff_time": "2026-09-05T14:00:00Z"},
        ],
        observations=[
            {"player_id": 8, "event": 2, "fixture_id": 70, "minutes": 90, "starts": 1,
             "total_points": 6, "updated_at": WRITTEN},
            # A scheduled placeholder: written before its own kickoff.
            {"player_id": 9, "event": 3, "fixture_id": 71, "minutes": 0,
             "total_points": None, "updated_at": "2026-09-05T09:00:00Z"},
        ],
    )

    assert analytics.positional_pooled_minutes(conn, 5, LATE)[3] == 90.0, (
        "the placeholder was averaged in as a zero-minute appearance"
    )
    # A cutoff before any row was written leaves the window empty: it fails closed
    # rather than reaching for the rows it can see but may not use.
    assert analytics.positional_pooled_minutes(conn, 5, EARLY) == {}


# ---------------------------------------------------------------------------
# PE1-P1-02 — ONE scheduled-placeholder contract, in Python and in SQL
# ---------------------------------------------------------------------------

#: Case B is the shape senior review found missing: a placeholder written AFTER
#: its own kickoff satisfies the write-after-kickoff clause, so only the value
#: signature can reject it.  Every case is written into the same fixture so the
#: only difference between them is the row itself.
_KICKOFF = "2026-08-29T14:00:00Z"
_AFTER_KICKOFF_BEFORE_CUTOFF = "2026-09-01T09:00:00Z"   # < LATE, > kickoff
_PLACEHOLDER_CASES = [
    # (name, minutes, explicit performance, updated_at, expect_included)
    ("A placeholder written BEFORE kickoff", 0, None, "2026-08-29T09:00:00Z", False),
    ("B placeholder written AFTER kickoff", 0, None, _AFTER_KICKOFF_BEFORE_CUTOFF, False),
    ("C genuine zero-minute DNP", 0, {"starts": 0, "total_points": 0, "expected_goals": 0.0}, WRITTEN, True),
    ("D normal played observation", 90, {"starts": 1, "total_points": 6, "expected_goals": 0.5}, WRITTEN, True),
    ("E observation written AFTER the cutoff", 90, {"starts": 1, "total_points": 6}, "2026-09-30T00:00:00Z", False),
]


def _seed_placeholder_cases(conn):
    observations = []
    for index, (_name, minutes, perf, written, _expected) in enumerate(_PLACEHOLDER_CASES):
        row = {"player_id": 200 + index, "event": 2, "fixture_id": 80, "minutes": minutes,
               "updated_at": written, "source": "element_summary"}
        row.update(perf or {})
        observations.append(row)
    _seed(
        conn,
        fixtures=[{"id": 80, "event": 2, "kickoff_time": _KICKOFF}],
        observations=observations,
    )


def test_the_placeholder_signature_is_one_definition_in_python_and_sql(conn):
    """SQL and Python must classify every shape identically.

    The SQL form is derived from the same column tuple and the same two
    branches, so this is a drift alarm rather than a second opinion.
    """

    from fpl_brain import repositories as repo

    _seed_placeholder_cases(conn)
    rows = conn.execute("SELECT pg.*, pg.rowid AS rid FROM player_gameweeks pg ORDER BY pg.rowid").fetchall()
    assert len(rows) == len(_PLACEHOLDER_CASES)

    for (name, _m, _p, _w, _e), row in zip(_PLACEHOLDER_CASES, rows):
        values = dict(row)
        python_says = repo.row_is_scheduled_placeholder(values)
        sql_says = conn.execute(
            f"SELECT ({repo.scheduled_placeholder_sql('pg')}) FROM player_gameweeks pg WHERE pg.rowid = ?",
            (int(values["rid"]),),
        ).fetchone()[0]
        assert sql_says is not None, f"{name}: SQL returned NULL, which silently drops rows"
        assert bool(sql_says) == python_says, f"{name}: SQL and Python disagree"


def test_exclusion_is_not_minutes_zero_and_not_starts_zero(conn):
    """A genuine zero-minute DNP survives; the two placeholders do not.

    This is the property that forbids replacing the signature with
    ``minutes == 0`` or ``starts == 0`` anywhere in the boundary.
    """

    _seed_placeholder_cases(conn)
    included = {(int(r["player_id"])) for r in _read(conn, LATE, 5)}
    dnp_player = 200 + 2   # case C
    played_player = 200 + 3  # case D
    assert dnp_player in included, "a genuine zero-minute non-appearance was erased"
    assert played_player in included
    assert (200 + 0) not in included, "case A: pre-kickoff placeholder leaked"
    assert (200 + 1) not in included, "case B: post-kickoff placeholder leaked"
    assert (200 + 4) not in included, "case E: post-cutoff observation leaked"


def test_the_composed_boundary_and_the_filtered_helper_return_the_same_rows(conn):
    """The composable SQL contract and the fully-filtered helper must be ONE contract."""

    _seed_placeholder_cases(conn)
    composed = {
        (int(r[0]), int(r[1])) for r in conn.execute(
            f"SELECT pg.player_id, pg.fixture_id FROM player_gameweeks pg "
            f"JOIN fixtures f ON f.id = pg.fixture_id WHERE {ho.OBSERVATION_SQL_CLAUSES}",
            ho.boundary_params(LATE, planning_event=5),
        ).fetchall()
    }
    helper = {
        (int(r["player_id"]), int(r["fixture_id"]))
        for r in ho.historical_player_fixture_rows(conn, as_of=LATE, planning_event=5)
    }
    assert composed == helper, (
        "a reader composing OBSERVATION_SQL_CLAUSES does not receive the same "
        "evidence as the fully-filtered helper"
    )


def test_the_post_kickoff_placeholder_is_blocked_from_the_minutes_baseline(conn):
    """PE1-P1-02 at a real consumer: ``analytics.positional_pooled_minutes``."""

    from fpl_brain import analytics

    _seed_placeholder_cases(conn)
    # Only cases C and D are evidence, so the position mean is (0 + 90) / 2.
    assert analytics.positional_pooled_minutes(conn, 5, LATE)[3] == 45.0, (
        "a placeholder changed the naive positional mean"
    )
    # Case B alone, with no genuine observation present, leaves the window empty.
    assert analytics.positional_pooled_minutes(conn, 5, "2026-09-02T00:00:00Z") == {}
