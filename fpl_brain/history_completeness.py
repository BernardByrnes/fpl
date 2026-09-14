"""Certified historical-input completeness.

WHY THIS EXISTS
---------------
A predictive certification may only be minted when the models can actually read
the completed history they claim to consume.  The R5 post-decision audit proved
the two official channels can disagree: the bootstrap aggregate
(``player_snapshots``) advances as soon as an official payload reports it, while
the event-specific history (``player_gameweeks``) is only refreshed by the
element-summary / event-live ingest.  A refresh sequence that skips those
endpoints leaves pre-round placeholder rows behind, and a placeholder is
indistinguishable from an unused player to a model that only looks at minutes.

THE BOUNDARY DOES NOT CHANGE
----------------------------
A Gameweek that is IN_PROGRESS or PROVISIONAL stays excluded from historical
evidence exactly as before; an aggregate field advancing never promotes an
in-progress round into training data.  What is enforced here is the weaker,
causal invariant:

    WHEN an event/fixture is officially complete and therefore eligible as
    historical predictive evidence, its required event-specific player
    observations must actually be present.

Three outcomes, in canonical vocabulary::

    IN-PROGRESS event          -> excluded, not required  -> PASS
    FINAL event + real rows    -> consumed normally       -> PASS
    FINAL event + placeholders -> CERTIFIED_PREDICTION_INPUT_HISTORY_INCOMPLETE

Only ``minutes`` is a HARD reconciled field.  Minutes are never revised by the
provider, whereas ``total_points`` demonstrably is: on the certified R5 snapshot
four players hold event rows whose points exceed the current official aggregate
because bonus was reallocated after the event (e.g. player 153: 4 stored points,
aggregate 1).  Points/goals/assists are therefore reported as diagnostics only,
never as blockers, so the invariant cannot fail on a legitimate revision.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Iterable, Mapping, Sequence

from . import repositories as repo

HISTORY_COMPLETENESS_SCHEMA = "fpl_brain.history_completeness.v1"

# The canonical top-level blocker.  Exact token; other systems grep for it.
CERTIFIED_PREDICTION_INPUT_HISTORY_INCOMPLETE = "CERTIFIED_PREDICTION_INPUT_HISTORY_INCOMPLETE"
# Internal subtypes, exposed in ``reasons`` for a precise diagnosis.
DIAG_COMPLETED_EVENT_PLACEHOLDER_ROW = "COMPLETED_EVENT_PLACEHOLDER_ROW"
DIAG_AGGREGATE_HISTORY_RECONCILIATION_FAILED = "AGGREGATE_HISTORY_RECONCILIATION_FAILED"
DIAG_HISTORY_EXCEEDS_OFFICIAL_AGGREGATE = "HISTORY_EXCEEDS_OFFICIAL_AGGREGATE"

# Minutes are the only field the provider never revises, so they are the only
# field allowed to block.  See the module docstring for the counterexamples.
HARD_RECONCILED_FIELD = "minutes"
DIAGNOSTIC_RECONCILED_FIELDS = ("total_points", "goals_scored", "assists")

# Bounded reporting: a broken refresh can touch every player, and an artifact
# must stay readable.  Counts are always exact; the id lists are capped.
MAX_REPORTED_PLAYERS = 40
MAX_REPORTED_ROWS = 40


def _import_analytics():
    from . import analytics

    return analytics


def _event_state_final() -> str:
    from .planning import EVENT_STATE_FINAL

    return EVENT_STATE_FINAL


def event_data_state(conn: sqlite3.Connection, event_id: int) -> tuple[str, list[str]]:
    """The official finality of one Gameweek (delegates to the planning spine)."""

    from .planning import event_data_state as _state

    return _state(conn, int(event_id))


# ---------------------------------------------------------------------------
# Which events are required, and what the models can actually see.
# ---------------------------------------------------------------------------


def required_completed_events(
    conn: sqlite3.Connection, *, planning_event: int, cutoff: str
) -> list[int]:
    """Events strictly before the planning event that are officially COMPLETE.

    FINAL requires ``events.finished=1`` AND ``events.data_checked=1``.  An event
    whose last kickoff is after the cutoff is not required either: the snapshot
    cannot have observed a completion that had not happened yet.
    """

    final = _event_state_final()
    out: list[int] = []
    rows = conn.execute(
        "SELECT id FROM events WHERE id < ? ORDER BY id", (int(planning_event),)
    ).fetchall()
    for row in rows:
        event_id = int(row[0])
        state, _basis = event_data_state(conn, event_id)
        if str(state) != final:
            continue
        latest = conn.execute(
            "SELECT MAX(kickoff_time) FROM fixtures WHERE event=?", (event_id,)
        ).fetchone()[0]
        if latest is not None and str(latest) > str(cutoff):
            continue
        out.append(event_id)
    return out


def in_progress_events(
    conn: sqlite3.Connection, *, planning_event: int, cutoff: str
) -> list[int]:
    """Pre-cutoff events that have started but are not officially complete."""

    final = _event_state_final()
    out: list[int] = []
    rows = conn.execute(
        "SELECT id FROM events WHERE id < ? ORDER BY id", (int(planning_event),)
    ).fetchall()
    for row in rows:
        event_id = int(row[0])
        state, _basis = event_data_state(conn, event_id)
        if str(state) == final:
            continue
        started = conn.execute(
            "SELECT COUNT(*) FROM fixtures WHERE event=? AND started=1"
            " AND (kickoff_time IS NULL OR kickoff_time <= ?)",
            (event_id, str(cutoff)),
        ).fetchone()[0]
        if int(started or 0) > 0:
            out.append(event_id)
    return out


def latest_observed_event(
    conn: sqlite3.Connection, *, planning_event: int, cutoff: str
) -> int | None:
    """The latest event the models actually hold a real (non-placeholder) row for.

    This is the honest counterpart to ``latest_required_completed_event``: it
    reports what the prediction could have read, not what was scheduled.
    """

    rows = conn.execute(
        """SELECT pg.player_id, pg.event, pg.fixture_id, pg.minutes, pg.starts,
                  pg.total_points, pg.goals_scored, pg.assists, pg.clean_sheets,
                  pg.goals_conceded, pg.saves, pg.bonus, pg.bps, pg.yellow_cards,
                  pg.red_cards, pg.penalties_saved, pg.penalties_missed, pg.own_goals,
                  pg.influence, pg.creativity, pg.threat, pg.ict_index,
                  pg.expected_goals, pg.expected_assists, pg.expected_goal_involvements,
                  pg.expected_goals_conceded, pg.defensive_contribution, pg.source
             FROM player_gameweeks pg JOIN fixtures f ON f.id = pg.fixture_id
            WHERE f.finished = 1 AND f.started = 1
              AND pg.event < ? AND f.event < ?
              AND (f.kickoff_time IS NULL OR f.kickoff_time <= ?)
            ORDER BY pg.event DESC""",
        (int(planning_event), int(planning_event), str(cutoff)),
    ).fetchall()
    for row in rows:
        if not repo.row_is_scheduled_placeholder(dict(row)):
            return int(row["event"])
    return None


# ---------------------------------------------------------------------------
# Structural check: the models' OWN row predicate must not return placeholders.
# ---------------------------------------------------------------------------


def structural_audit(
    conn: sqlite3.Connection, *, planning_event: int, cutoff: str
) -> dict[str, Any]:
    """Placeholder rows inside the exact row set the models may consume.

    The predicate is deliberately identical to ``analytics.completed_rows_as_of``
    (``f.finished=1 AND f.started=1``, event strictly before planning, kickoff not
    after the cutoff).  Using the consumers' own scope is what makes this
    false-positive free: if the models cannot read a row, its state is not
    asserted here.
    """

    rows = conn.execute(
        """SELECT pg.player_id, pg.event, pg.fixture_id, pg.minutes, pg.starts,
                  pg.total_points, pg.goals_scored, pg.assists, pg.clean_sheets,
                  pg.goals_conceded, pg.saves, pg.bonus, pg.bps, pg.yellow_cards,
                  pg.red_cards, pg.penalties_saved, pg.penalties_missed, pg.own_goals,
                  pg.influence, pg.creativity, pg.threat, pg.ict_index,
                  pg.expected_goals, pg.expected_assists, pg.expected_goal_involvements,
                  pg.expected_goals_conceded, pg.defensive_contribution, pg.source
             FROM player_gameweeks pg JOIN fixtures f ON f.id = pg.fixture_id
            WHERE f.finished = 1 AND f.started = 1
              AND pg.event < ? AND f.event < ?
              AND (f.kickoff_time IS NULL OR f.kickoff_time <= ?)
            ORDER BY pg.event, pg.fixture_id, pg.player_id""",
        (int(planning_event), int(planning_event), str(cutoff)),
    ).fetchall()

    placeholders = [dict(row) for row in rows if repo.row_is_scheduled_placeholder(dict(row))]
    by_event: dict[int, int] = {}
    for row in placeholders:
        by_event[int(row["event"])] = by_event.get(int(row["event"]), 0) + 1
    affected = sorted(
        (
            {"player_id": int(r["player_id"]), "event": int(r["event"]), "fixture_id": int(r["fixture_id"])}
            for r in placeholders
        ),
        key=lambda item: (item["event"], item["fixture_id"], item["player_id"]),
    )
    return {
        "rows_scanned": len(rows),
        "placeholder_rows": len(placeholders),
        "placeholder_rows_by_event": {str(k): v for k, v in sorted(by_event.items())},
        "affected_players": sorted({item["player_id"] for item in affected}),
        "affected_rows": affected[:MAX_REPORTED_ROWS],
        "affected_rows_truncated": max(0, len(affected) - MAX_REPORTED_ROWS),
        "complete": not placeholders,
        "detail": (
            "every row the models may consume carries an official observation"
            if not placeholders
            else "completed-event rows carry no official observation; a placeholder must never "
                 "be read as 'the player did not play'"
        ),
    }


# ---------------------------------------------------------------------------
# Reconciliation: official aggregate vs completed-event history.
# ---------------------------------------------------------------------------


def anchored_residual_players(
    conn: sqlite3.Connection, *, planning_event: int, cutoff: str
) -> set[int]:
    """Players who hold their OWN row on a started-but-unfinished pre-cutoff fixture.

    This is the only admissible explanation for an aggregate that runs ahead of
    the completed history: the residual must come from a fixture the player
    himself has a slot in.  The anchor is derived from the player's own rows, not
    from his current club, so nothing about historical membership is inferred
    from the present squad — a player whose history was never ingested at all has
    no anchor and therefore cannot have a residual excused.
    """

    rows = conn.execute(
        """SELECT DISTINCT pg.player_id
             FROM player_gameweeks pg JOIN fixtures f ON f.id = pg.fixture_id
            WHERE f.finished != 1 AND f.started = 1
              AND pg.event < ? AND (f.kickoff_time IS NULL OR f.kickoff_time <= ?)""",
        (int(planning_event), str(cutoff)),
    ).fetchall()
    return {int(row[0]) for row in rows}


def started_unfinished_team_ids(
    conn: sqlite3.Connection, *, planning_event: int, cutoff: str
) -> set[int]:
    """Teams with a pre-cutoff fixture that has started but is not finished.

    Informational only.  A club having such a fixture is NOT sufficient to excuse
    a residual -- the player must hold his own row on it
    (``anchored_residual_players``) -- because an unfetched or pruned observation
    would otherwise be masked by a team-mate's unfinished fixture.
    """

    rows = conn.execute(
        """SELECT team_h AS team FROM fixtures
            WHERE event < ? AND finished != 1 AND started = 1
              AND (kickoff_time IS NULL OR kickoff_time <= ?)
           UNION
           SELECT team_a AS team FROM fixtures
            WHERE event < ? AND finished != 1 AND started = 1
              AND (kickoff_time IS NULL OR kickoff_time <= ?)""",
        (int(planning_event), str(cutoff), int(planning_event), str(cutoff)),
    ).fetchall()
    return {int(row[0]) for row in rows}


def _row_sums(conn: sqlite3.Connection, sql: str, params: Sequence[Any]) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    for row in conn.execute(sql, tuple(params)):
        out[int(row["player_id"])] = {
            field: float(row[field] or 0.0) for field in (HARD_RECONCILED_FIELD,) + DIAGNOSTIC_RECONCILED_FIELDS
        }
    return out


def reconciliation_audit(
    conn: sqlite3.Connection, *, planning_event: int, cutoff: str
) -> dict[str, Any]:
    """Official aggregate minus the history the models can actually read.

    The residual is permitted only when the PLAYER HIMSELF holds a row on a
    started-but-unfinished pre-cutoff fixture, which is the one thing that can
    legitimately put minutes into the aggregate without putting them into the
    history.  A club-level explanation is deliberately NOT enough: it would mask a
    pruned or never-ingested observation behind a team-mate's unfinished fixture.
    An unexplained positive residual (history missing) and a negative residual
    (history exceeding the official cumulative total) both fail closed.
    """

    analytics = _import_analytics()
    fields = (HARD_RECONCILED_FIELD,) + DIAGNOSTIC_RECONCILED_FIELDS
    select = ", ".join(f"SUM(pg.{f}) AS {f}" for f in fields)
    # The row predicate is the consumer one (analytics.completed_rows_as_of): both
    # the player's event label AND the fixture's event must sit before the planning
    # event, so the reconciliation measures exactly the rows the models can read.
    params = (int(planning_event), int(planning_event), str(cutoff))

    history = _row_sums(
        conn,
        f"""SELECT pg.player_id, {select}
              FROM player_gameweeks pg JOIN fixtures f ON f.id = pg.fixture_id
             WHERE f.finished = 1 AND f.started = 1
               AND pg.event < ? AND f.event < ?
               AND (f.kickoff_time IS NULL OR f.kickoff_time <= ?)
             GROUP BY pg.player_id""",
        params,
    )
    in_progress = _row_sums(
        conn,
        f"""SELECT pg.player_id, {select}
              FROM player_gameweeks pg JOIN fixtures f ON f.id = pg.fixture_id
             WHERE f.finished != 1 AND f.started = 1
               AND pg.event < ? AND f.event < ?
               AND (f.kickoff_time IS NULL OR f.kickoff_time <= ?)
             GROUP BY pg.player_id""",
        params,
    )
    anchored = anchored_residual_players(conn, planning_event=planning_event, cutoff=cutoff)
    explainable_teams = started_unfinished_team_ids(conn, planning_event=planning_event, cutoff=cutoff)

    teams = {
        int(row["player_id"]): (int(row["team_id"]) if row["team_id"] is not None else None)
        for row in analytics.projectable_players(conn)
    }

    per_field: dict[str, dict[str, Any]] = {}
    for field in fields:
        missing: list[dict[str, Any]] = []
        exceed: list[dict[str, Any]] = []
        permitted = 0
        checked = 0
        for player_id in sorted(teams):
            snapshot = analytics.snapshot_as_of(conn, player_id, cutoff)
            if snapshot is None or snapshot.get(field) is None:
                continue
            checked += 1
            aggregate = float(snapshot[field])
            accounted = history.get(player_id, {}).get(field, 0.0) + in_progress.get(player_id, {}).get(field, 0.0)
            residual = round(aggregate - accounted, 6)
            if abs(residual) < 1e-6:
                continue
            team = teams.get(player_id)
            entry = {
                "player_id": player_id,
                "team_id": team,
                "aggregate": aggregate,
                "completed_history": history.get(player_id, {}).get(field, 0.0),
                "in_progress_history": in_progress.get(player_id, {}).get(field, 0.0),
                "residual": residual,
            }
            if residual < 0:
                exceed.append(entry)
            elif player_id in anchored:
                permitted += 1
            else:
                missing.append(entry)
        per_field[field] = {
            "enforced": field == HARD_RECONCILED_FIELD,
            "players_checked": checked,
            "residuals_permitted_by_own_anchor_row": permitted,
            "unexplained_players": len(missing),
            "history_exceeds_aggregate_players": len(exceed),
            "unexplained": missing[:MAX_REPORTED_PLAYERS],
            "unexplained_truncated": max(0, len(missing) - MAX_REPORTED_PLAYERS),
            "history_exceeds_aggregate": exceed[:MAX_REPORTED_PLAYERS],
            "complete": not missing and not exceed,
        }

    hard = per_field[HARD_RECONCILED_FIELD]
    return {
        "hard_field": HARD_RECONCILED_FIELD,
        "diagnostic_fields": list(DIAGNOSTIC_RECONCILED_FIELDS),
        "explainable_team_ids": sorted(explainable_teams),
        "anchored_player_count": len(anchored),
        "per_field": per_field,
        "complete": bool(hard["complete"]),
        "detail": (
            "the official aggregate is fully accounted for by completed history plus each "
            "player's own started-but-unfinished fixture rows"
            if hard["complete"]
            else "the official aggregate advanced beyond the completed history with no "
                 "unfinished fixture of the player's own to explain it"
        ),
    }


# ---------------------------------------------------------------------------
# The audit the certification gate consumes.
# ---------------------------------------------------------------------------


def audit_history_completeness(
    conn: sqlite3.Connection, *, planning_event: int, cutoff: str
) -> dict[str, Any]:
    """Deterministic, side-effect-free completeness audit of one cutoff.

    Ordering is fixed (events ascending, then fixture id, then player id; reason
    tokens sorted), so the same database always produces the same artifact bytes.
    """

    required = required_completed_events(conn, planning_event=planning_event, cutoff=cutoff)
    structural = structural_audit(conn, planning_event=planning_event, cutoff=cutoff)
    reconciliation = reconciliation_audit(conn, planning_event=planning_event, cutoff=cutoff)

    reasons: list[str] = []
    if not structural["complete"]:
        reasons.append(DIAG_COMPLETED_EVENT_PLACEHOLDER_ROW)
    if not reconciliation["complete"]:
        if reconciliation["per_field"][HARD_RECONCILED_FIELD]["unexplained_players"]:
            reasons.append(DIAG_AGGREGATE_HISTORY_RECONCILIATION_FAILED)
        if reconciliation["per_field"][HARD_RECONCILED_FIELD]["history_exceeds_aggregate_players"]:
            reasons.append(DIAG_HISTORY_EXCEEDS_OFFICIAL_AGGREGATE)
    reasons = sorted(set(reasons))

    return {
        "schema": HISTORY_COMPLETENESS_SCHEMA,
        "planning_event": int(planning_event),
        "cutoff": str(cutoff),
        "required_completed_events": required,
        "latest_required_completed_event": required[-1] if required else None,
        "in_progress_events": in_progress_events(conn, planning_event=planning_event, cutoff=cutoff),
        "latest_observed_event": latest_observed_event(conn, planning_event=planning_event, cutoff=cutoff),
        "structural": structural,
        "reconciliation": reconciliation,
        "complete": not reasons,
        "blocker": None if not reasons else CERTIFIED_PREDICTION_INPUT_HISTORY_INCOMPLETE,
        "reasons": reasons,
        "detail": (
            "required completed-event history is present and reconciles with the official aggregate"
            if not reasons
            else "required completed-event history is incomplete: "
                 + ", ".join(reasons)
        ),
    }


def blocking_reason_token(audit: Mapping[str, Any]) -> str | None:
    """The canonical token to place in a permission-reason list, or None."""

    if audit.get("complete"):
        return None
    return str(audit.get("blocker") or CERTIFIED_PREDICTION_INPUT_HISTORY_INCOMPLETE)


# ---------------------------------------------------------------------------
# REPAIR A: obtain the required completed-event history through official ingest.
# ---------------------------------------------------------------------------


class HistoryRefreshIncomplete(RuntimeError):
    """The refresh ran but the required completed-event history is still missing."""


def planning_event_from_db(conn: sqlite3.Connection) -> int:
    """The event the models plan for: the official next event, else current+1."""

    row = conn.execute("SELECT id FROM events WHERE is_next=1 ORDER BY id LIMIT 1").fetchone()
    if row is not None:
        return int(row[0])
    row = conn.execute("SELECT MAX(id) FROM events WHERE is_current=1").fetchone()
    if row is not None and row[0] is not None:
        return int(row[0]) + 1
    row = conn.execute("SELECT MAX(id) FROM events").fetchone()
    if row is not None and row[0] is not None:
        return int(row[0]) + 1
    raise HistoryRefreshIncomplete("no events are stored, so no planning event can be derived")


def refresh_completed_event_history(
    config: Mapping[str, Any],
    *,
    planning_event: int | None = None,
    cutoff: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Ensure completed-event history exists before a certification is minted.

    The refresh reuses the canonical official ingest (``ingest.run_fetch``) rather
    than a bespoke downloader, and asks for ``summaries="all"`` because xG/xA --
    which ``player_rates`` fits on -- exist only in the element-summary payload;
    the per-event live endpoint carries minutes and points but no xG.  It is a
    no-op when the history is already complete, and it FAILS CLOSED when the
    history is still incomplete afterwards: no caller may proceed to certify on
    stale placeholders.
    """

    from .config import config_path
    from .database import connect_database

    database = config_path(config, "database")
    conn = connect_database(database)
    try:
        event = int(planning_event) if planning_event is not None else planning_event_from_db(conn)
        effective_cutoff = str(cutoff) if cutoff is not None else _latest_captured_at(conn)
        before = audit_history_completeness(conn, planning_event=event, cutoff=effective_cutoff)
        needed = not before["complete"]
        latest_required = before["latest_required_completed_event"]
    finally:
        conn.close()

    refresh: dict[str, Any] = {
        "performed": False,
        "summaries": "all",
        "live_event": latest_required,
        "reason": "completed-event history already present" if not needed else "refreshing required history",
    }
    fetch_result: dict[str, Any] | None = None
    if needed and not dry_run:
        from . import ingest

        fetch_result = ingest.run_fetch(config, "all", latest_required, False)
        refresh.update(
            {
                "performed": True,
                "run_id": fetch_result.get("run_id"),
                "status": fetch_result.get("status"),
                "endpoints_ok": list(fetch_result.get("endpoints_ok") or []),
                "endpoints_failed": list(fetch_result.get("endpoints_failed") or []),
            }
        )
    elif needed and dry_run:
        refresh["reason"] = "dry run: required history is missing and would be refreshed"

    conn = connect_database(database)
    try:
        after = audit_history_completeness(conn, planning_event=event, cutoff=effective_cutoff)
    finally:
        conn.close()

    report = {
        "schema": "fpl_brain.history_refresh.v1",
        "planning_event": event,
        "cutoff": effective_cutoff,
        "dry_run": bool(dry_run),
        "history_required": needed,
        "refresh": refresh,
        "before": _compact(before),
        "after": _compact(after),
        "complete": bool(after["complete"]),
        "blocker": blocking_reason_token(after),
    }
    if needed and not dry_run and not after["complete"]:
        raise HistoryRefreshIncomplete(
            f"{CERTIFIED_PREDICTION_INPUT_HISTORY_INCOMPLETE}: completed-event history is still "
            f"incomplete after refresh (reasons={after['reasons']}); refusing to proceed"
        )
    return report


def _latest_captured_at(conn: sqlite3.Connection) -> str:
    row = conn.execute("SELECT MAX(captured_at) FROM player_snapshots").fetchone()
    if row is None or row[0] is None:
        row = conn.execute("SELECT MAX(updated_at) FROM fixtures").fetchone()
    if row is None or row[0] is None:
        raise HistoryRefreshIncomplete("no official capture exists, so no cutoff can be derived")
    return str(row[0])


def _compact(audit: Mapping[str, Any]) -> dict[str, Any]:
    """The audit without the bulk row lists, for embedding in a report."""

    structural = dict(audit["structural"])
    structural.pop("affected_rows", None)
    reconciliation = {
        "hard_field": audit["reconciliation"]["hard_field"],
        "complete": audit["reconciliation"]["complete"],
        "per_field": {
            field: {k: v for k, v in payload.items() if k not in {"unexplained", "history_exceeds_aggregate"}}
            for field, payload in audit["reconciliation"]["per_field"].items()
        },
    }
    return {
        "required_completed_events": audit["required_completed_events"],
        "latest_required_completed_event": audit["latest_required_completed_event"],
        "in_progress_events": audit["in_progress_events"],
        "latest_observed_event": audit["latest_observed_event"],
        "structural": structural,
        "reconciliation": reconciliation,
        "complete": audit["complete"],
        "blocker": audit["blocker"],
        "reasons": audit["reasons"],
    }
