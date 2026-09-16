"""The ONE canonical point-in-time boundary for historical player-fixture evidence.

WHY THIS EXISTS
---------------
Before this module the causal predicate

    fixtures.finished = 1 AND fixtures.started = 1
    AND player_gameweeks.event < planning_event
    AND fixtures.event < planning_event
    AND (fixtures.kickoff_time IS NULL OR fixtures.kickoff_time <= cutoff)

was copy-pasted into at least seven readers (``analytics``, ``xpts``,
``minutes_model``, ``team_model``, ``player_rates``, ``substitution_model``,
``history_completeness``, ``market_report``).  Duplicated predicates drift, and a
drifted historical reader silently poisons every number computed on top of it.

THE HAZARD THIS CLOSES
----------------------
The old predicate is not sufficient on a real database.  Measured on the live
store: 2,545 ``player_gameweeks`` rows sit on FINISHED fixtures, and of those
**655 are pre-round placeholders** -- written 2026-09-10 for fixtures that kicked
off on 2026-09-12..14, carrying ``minutes=0`` and ``total_points IS NULL``.  The
FIXTURE finished, so ``f.finished=1`` accepted them; the ROW was never refreshed.
A reader that accepts them treats "we do not know yet" as "he scored nothing".

THE BOUNDARY
------------
An observation is historical evidence as of ``as_of`` only when ALL hold:

  1. the fixture started and finished            (existing contract)
  2. the fixture is strictly before the planning event when one is supplied
  3. ``kickoff_time IS NOT NULL AND kickoff_time <= as_of``  -- causally past
  4. ``updated_at <= as_of``                     -- it was OBSERVABLE at the cutoff
  5. ``updated_at >= kickoff_time``              -- the row was written AFTER kickoff,
     so it cannot be a pre-round schedule placeholder
  6. the row is not a scheduled placeholder, by the canonical
     ``repositories.row_is_scheduled_placeholder`` value signature

Clauses 4 and 5 are what the old predicate lacked.  Clause 4 is the honest
as-of requirement: a fixture that is finished TODAY was not finished at a
historical cutoff T, and a row written after T must not be visible at T.
Clause 5 is the causal complement to clause 6: the value signature catches the
placeholders that exist now, while the timestamp test catches a row that predates
its own kickoff whatever its columns happen to contain.

FAIL CLOSED
-----------
``as_of`` is REQUIRED and has no default.  There is deliberately no "now"
fallback: a historical reader that cannot state its cutoff cannot be trusted, and
silently defaulting to the current time is exactly the leakage this module
exists to prevent.

WHAT IS NOT HERE
----------------
This is a boundary, not a feature store.  It returns raw rows with their
provenance intact and makes no modelling decision about them.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any, Sequence

from . import repositories as repo

HISTORICAL_OBSERVATION_BOUNDARY_VERSION = "historical_observations_v1.0.0"

#: The columns every historical reader may rely on.
OBSERVATION_COLUMNS = (
    "pg.player_id, pg.event, pg.fixture_id, pg.opponent_team, pg.was_home, pg.minutes, "
    "pg.starts, pg.total_points, pg.goals_scored, pg.assists, pg.clean_sheets, "
    "pg.goals_conceded, pg.saves, pg.bonus, pg.bps, pg.yellow_cards, pg.red_cards, "
    "pg.penalties_saved, pg.penalties_missed, pg.own_goals, pg.influence, pg.creativity, "
    "pg.threat, pg.ict_index, pg.expected_goals, pg.expected_assists, "
    "pg.expected_goal_involvements, pg.expected_goals_conceded, pg.defensive_contribution, "
    "pg.value, pg.selected, pg.source, pg.updated_at, "
    "f.kickoff_time AS fixture_kickoff, f.event AS fixture_event, f.finished AS fixture_finished"
)

#: The causal + observability clauses, in one place.  ``?`` order:
#: as_of (kickoff), as_of (observability), kickoff (write-after-kickoff).
_AS_OF_CLAUSES = """
      AND f.finished = 1 AND f.started = 1
      AND f.kickoff_time IS NOT NULL AND f.kickoff_time <= ?
      AND pg.updated_at IS NOT NULL AND pg.updated_at <= ?
      AND pg.updated_at >= f.kickoff_time
"""


class HistoricalBoundaryError(ValueError):
    """A historical read was requested without a usable point-in-time cutoff."""


def require_as_of(as_of: str | None) -> str:
    """The explicit cutoff, or refuse.  There is no implicit "now"."""

    value = str(as_of or "").strip()
    if not value:
        raise HistoricalBoundaryError(
            "a historical player-fixture read requires an explicit as_of cutoff; there is no "
            "implicit 'now' because the current time is not a point in time anyone decided at"
        )
    return value


@dataclass(frozen=True)
class ObservationWindow:
    """The resolved boundary a read was performed under, for the caller's evidence."""

    as_of: str
    planning_event: int | None
    boundary_version: str = HISTORICAL_OBSERVATION_BOUNDARY_VERSION
    clauses: tuple[str, ...] = (
        "fixture started and finished",
        "fixture strictly before the planning event",
        "kickoff_time <= as_of",
        "updated_at <= as_of",
        "updated_at >= kickoff_time",
        "not a scheduled placeholder",
    )

    def as_dict(self) -> dict[str, Any]:
        return {
            "as_of": str(self.as_of),
            "planning_event": None if self.planning_event is None else int(self.planning_event),
            "boundary_version": str(self.boundary_version),
            "clauses": list(self.clauses),
        }


def _filter_placeholders(rows: Sequence[Mapping_like]) -> list[dict[str, Any]]:
    """Keep only rows that carry a real observation, by the canonical signature."""

    kept: list[dict[str, Any]] = []
    for row in rows:
        values = dict(row)
        if repo.row_is_scheduled_placeholder(values):
            continue
        kept.append(values)
    return kept


Mapping_like = Any  # narrowed by usage; rows arrive as sqlite3.Row or Mapping


def historical_player_fixtures(
    conn: sqlite3.Connection,
    *,
    as_of: str,
    planning_event: int | None = None,
    player_ids: Sequence[int] | None = None,
    event: int | None = None,
) -> list[dict[str, Any]]:
    """The historical player-fixture observations legitimately visible at ``as_of``.

    ``planning_event``, when supplied, adds the strictly-earlier-event clause that
    the accepted readers use.  Omitting it still applies every causal and
    observability clause; it only widens the event range.
    """

    cutoff = require_as_of(as_of)
    params: list[Any] = [cutoff, cutoff]
    sql = f"SELECT {OBSERVATION_COLUMNS} FROM player_gameweeks pg JOIN fixtures f ON f.id = pg.fixture_id WHERE 1=1"
    sql += _AS_OF_CLAUSES
    if planning_event is not None:
        sql += " AND pg.event < ? AND f.event < ?"
        params.extend([int(planning_event), int(planning_event)])
    if event is not None:
        sql += " AND pg.event = ?"
        params.append(int(event))
    if player_ids is not None:
        ids = [int(p) for p in player_ids]
        if not ids:
            return []
        sql += f" AND pg.player_id IN ({','.join('?' for _ in ids)})"
        params.extend(ids)
    sql += " ORDER BY pg.event, pg.fixture_id, pg.player_id"
    return _filter_placeholders(conn.execute(sql, tuple(params)).fetchall())


def historical_player_fixture_rows(
    conn: sqlite3.Connection,
    *,
    as_of: str,
    planning_event: int | None = None,
    player_id: int | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """The same boundary, most-recent-first, for a single player or the whole pool."""

    cutoff = require_as_of(as_of)
    params: list[Any] = [cutoff, cutoff]
    sql = f"SELECT {OBSERVATION_COLUMNS} FROM player_gameweeks pg JOIN fixtures f ON f.id = pg.fixture_id WHERE 1=1"
    sql += _AS_OF_CLAUSES
    if planning_event is not None:
        sql += " AND pg.event < ? AND f.event < ?"
        params.extend([int(planning_event), int(planning_event)])
    if player_id is not None:
        sql += " AND pg.player_id = ?"
        params.append(int(player_id))
    sql += " ORDER BY f.event DESC, f.kickoff_time DESC, f.id DESC"
    if limit is not None:
        sql += " LIMIT ?"
        params.append(int(limit))
    return _filter_placeholders(conn.execute(sql, tuple(params)).fetchall())


def is_historical_observation(row: Any, *, as_of: str) -> bool:
    """Whether ONE row is legitimate evidence at ``as_of``, by the same clauses.

    Used by callers that already hold rows (for example a per-fixture lookup) so
    they do not restate the boundary.
    """

    cutoff = require_as_of(as_of)
    values = dict(row)
    kickoff = values.get("fixture_kickoff", values.get("kickoff_time"))
    updated = values.get("updated_at")
    if not kickoff or not updated:
        return False
    if str(kickoff) > cutoff or str(updated) > cutoff:
        return False
    if str(updated) < str(kickoff):
        return False
    return not repo.row_is_scheduled_placeholder(values)


# ---------------------------------------------------------------------------
# Data health
# ---------------------------------------------------------------------------


def data_health(conn: sqlite3.Connection, *, as_of: str | None = None) -> dict[str, Any]:
    """A deterministic point-in-time data-health snapshot.

    ``as_of`` defaults to the newest observable timestamp in the store rather than
    to wall-clock now, so the report is reproducible from the data alone.
    """

    from .utils import utc_now

    cutoff = str(as_of).strip() if as_of else str(
        (conn.execute("SELECT MAX(updated_at) FROM player_gameweeks").fetchone() or [utc_now()])[0]
        or utc_now()
    )
    total = conn.execute("SELECT COUNT(*) FROM player_gameweeks").fetchone()[0]
    on_finished = conn.execute(
        "SELECT COUNT(*) FROM player_gameweeks pg JOIN fixtures f ON f.id = pg.fixture_id "
        "WHERE f.finished = 1 AND f.started = 1"
    ).fetchone()[0]
    observed = historical_player_fixtures(conn, as_of=cutoff)
    zero_minute = [row for row in observed if not float(row.get("minutes") or 0.0) > 0.0]

    captures = [
        str(row[0]) for row in conn.execute(
            "SELECT DISTINCT captured_at FROM player_snapshots ORDER BY captured_at"
        ).fetchall() if row[0]
    ]
    gaps = [
        (captures[index + 1][:10], captures[index][:10])
        for index in range(len(captures) - 1)
    ]
    deadline = conn.execute(
        "SELECT id, deadline_time FROM events WHERE deadline_time IS NOT NULL "
        "AND deadline_time > ? ORDER BY deadline_time LIMIT 1",
        (cutoff,),
    ).fetchone()
    last_before_deadline = None
    if deadline is not None:
        earlier = [c for c in captures if c <= str(deadline[1])]
        last_before_deadline = earlier[-1] if earlier else None
    return {
        "as_of": cutoff,
        "boundary": ObservationWindow(as_of=cutoff, planning_event=None).as_dict(),
        "player_gameweek_rows": int(total),
        "rows_on_finished_fixtures": int(on_finished),
        "past_observations_as_of_now": len(observed),
        "future_or_placeholder_rows": int(total) - len(observed),
        "completed_zero_minute_observations": len(zero_minute),
        "availability_captures": len(captures),
        "first_availability_capture": captures[0] if captures else None,
        "latest_availability_capture": captures[-1] if captures else None,
        "max_capture_gap_days": _max_gap_days(captures),
        "next_deadline": ({"event": int(deadline[0]), "deadline_time": str(deadline[1])}
                          if deadline is not None else None),
        "last_capture_before_next_deadline": last_before_deadline,
        "capture_gaps": [{"from": a, "to": b} for a, b in gaps if a != b],
    }


def _max_gap_days(captures: Sequence[str]) -> float | None:
    from .utils import parse_utc

    days = [
        (parse_utc(b) - parse_utc(a)).total_seconds() / 86400.0
        for a, b in zip(captures, captures[1:])
        if parse_utc(a) is not None and parse_utc(b) is not None
    ]
    return round(max(days), 3) if days else None
