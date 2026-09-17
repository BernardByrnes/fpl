"""Typed database reads and writes for each logical table."""

from __future__ import annotations

import json
import sqlite3
from typing import Any, Iterable, Mapping, Sequence

from .models import (
    ChipRecord,
    EntryRecord,
    EventRecord,
    FixtureRecord,
    HistoryRow,
    PickRecord,
    PlayerGameweekRecord,
    PlayerRecord,
    PlayerSnapshotRecord,
    PositionRecord,
    TeamRecord,
)
from .utils import json_text, normalise_name, parse_utc, utc_now


def _raw(value: Any) -> str:
    return json_text(value)


# ---------------------------------------------------------------------------
# Certified official-player-pool identity (R4B.1.1)
# ---------------------------------------------------------------------------


def record_bootstrap_generation(
    conn: sqlite3.Connection,
    *,
    captured_at: str,
    accepted: bool,
    official_element_count: int,
    parsed_count: int,
    persisted_count: int,
    element_ids: Sequence[int],
    element_ids_sha256: str,
    acceptance_rule: str,
    acceptance_rule_version: str,
    fetch_run_id: int | None = None,
    availability_counts: Mapping[str, Any] | None = None,
    club_player_counts: Mapping[str, Any] | None = None,
    rejection_reasons: Sequence[str] | None = None,
) -> int:
    """Persist one bootstrap generation INSIDE SQLite.

    Persisting in the database (rather than only as an external JSON report) is
    what makes the identity travel into the immutable execution snapshot, which
    is the causal source a decision reads.
    """

    cursor = conn.execute(
        """INSERT INTO bootstrap_generations(
               fetch_run_id, captured_at, accepted, official_element_count, parsed_count,
               persisted_count, element_ids_sha256, element_ids_json, availability_counts_json,
               club_player_counts_json, acceptance_rule, rejection_reasons_json,
               acceptance_rule_version, recorded_at
           ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            None if fetch_run_id is None else int(fetch_run_id),
            str(captured_at),
            1 if accepted else 0,
            int(official_element_count),
            int(parsed_count),
            int(persisted_count),
            str(element_ids_sha256),
            _raw(sorted(int(pid) for pid in element_ids)),
            _raw(dict(availability_counts or {})),
            _raw(dict(club_player_counts or {})),
            str(acceptance_rule),
            _raw(list(rejection_reasons or [])),
            str(acceptance_rule_version),
            utc_now(),
        ),
    )
    return int(cursor.lastrowid)


def latest_accepted_bootstrap_generation(conn: sqlite3.Connection) -> dict[str, Any] | None:
    """The latest ACCEPTED official bootstrap generation, or None.

    ``WHERE accepted=1`` is the whole point: a rejected generation is retained
    for audit but must never be read as the authoritative pool identity.
    """

    row = conn.execute(
        "SELECT * FROM bootstrap_generations WHERE accepted=1 ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if row is None:
        return None
    record = dict(row)
    try:
        record["element_ids"] = [int(pid) for pid in json.loads(record.get("element_ids_json") or "[]")]
    except (TypeError, ValueError):
        record["element_ids"] = []
    return record


def active_player_ids(conn: sqlite3.Connection) -> list[int]:
    """Every currently active player id, sorted (the persisted official pool)."""

    return sorted(int(row[0]) for row in conn.execute("SELECT id FROM players WHERE is_active=1"))


def active_player_ids_sha256(conn: sqlite3.Connection) -> tuple[int, str]:
    """(count, deterministic digest) of the persisted active player id set."""

    return element_ids_identity(active_player_ids(conn))


def latest_bootstrap_generation_attempt(conn: sqlite3.Connection) -> dict[str, Any] | None:
    """The latest generation attempt of ANY status (accepted or rejected)."""

    row = conn.execute("SELECT * FROM bootstrap_generations ORDER BY id DESC LIMIT 1").fetchone()
    return dict(row) if row is not None else None


def element_ids_identity(element_ids: Sequence[int]) -> tuple[int, str]:
    """(count, sha256) of the sorted, de-duplicated official element id set.

    Delegates to the single canonical implementation in
    :mod:`fpl_brain.ingest_provenance` so the writer, the reader and the
    decision-time assertion can never drift apart.
    """

    from .ingest_provenance import element_id_sha256

    ordered = sorted({int(pid) for pid in element_ids})
    return len(ordered), element_id_sha256(ordered)


def _rows(cursor: sqlite3.Cursor) -> list[dict[str, Any]]:
    return [dict(row) for row in cursor.fetchall()]


def create_fetch_run(
    conn: sqlite3.Connection,
    trigger: str,
    raw_dir: str | None = None,
    started_at: str | None = None,
) -> int:
    cursor = conn.execute(
        "INSERT INTO fetch_runs(started_at, status, trigger, raw_dir) VALUES (?, 'running', ?, ?)",
        (started_at or utc_now(), trigger, raw_dir),
    )
    return int(cursor.lastrowid)


def finish_fetch_run(
    conn: sqlite3.Connection,
    run_id: int,
    status: str,
    current_event: int | None = None,
    endpoints_ok: Sequence[str] | None = None,
    endpoints_failed: Sequence[dict[str, Any]] | None = None,
    error_message: str | None = None,
) -> None:
    conn.execute(
        """UPDATE fetch_runs
           SET finished_at=?, status=?, current_event=?, endpoints_ok=?, endpoints_failed=?, error_message=?
           WHERE id=?""",
        (
            utc_now(),
            status,
            current_event,
            _raw(list(endpoints_ok or [])),
            _raw(list(endpoints_failed or [])),
            error_message,
            run_id,
        ),
    )


def latest_fetch_run(conn: sqlite3.Connection, trigger: str = "fetch_fpl") -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT * FROM fetch_runs WHERE trigger=? ORDER BY id DESC LIMIT 1",
        (trigger,),
    ).fetchone()
    return dict(row) if row else None


def upsert_events(conn: sqlite3.Connection, records: Iterable[EventRecord], updated_at: str | None = None) -> None:
    timestamp = updated_at or utc_now()
    for record in records:
        conn.execute(
            """INSERT INTO events(
                id,name,deadline_time,deadline_time_epoch,finished,data_checked,is_previous,is_current,is_next,
                average_entry_score,highest_score,most_selected,most_transferred_in,most_captained,
                most_vice_captained,top_element,transfers_made,released,raw_json,updated_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET
                name=excluded.name, deadline_time=excluded.deadline_time,
                deadline_time_epoch=excluded.deadline_time_epoch, finished=excluded.finished,
                data_checked=excluded.data_checked, is_previous=excluded.is_previous,
                is_current=excluded.is_current, is_next=excluded.is_next,
                average_entry_score=excluded.average_entry_score, highest_score=excluded.highest_score,
                most_selected=excluded.most_selected, most_transferred_in=excluded.most_transferred_in,
                most_captained=excluded.most_captained, most_vice_captained=excluded.most_vice_captained,
                top_element=excluded.top_element, transfers_made=excluded.transfers_made,
                released=excluded.released, raw_json=excluded.raw_json, updated_at=excluded.updated_at""",
            (
                record.id, record.name, record.deadline_time, record.deadline_time_epoch, record.finished,
                record.data_checked, record.is_previous, record.is_current, record.is_next,
                record.average_entry_score, record.highest_score, record.most_selected,
                record.most_transferred_in, record.most_captained, record.most_vice_captained,
                record.top_element, record.transfers_made, record.released, _raw(record.raw_json), timestamp,
            ),
        )


def upsert_chips(conn: sqlite3.Connection, records: Iterable[ChipRecord], updated_at: str | None = None) -> None:
    timestamp = updated_at or utc_now()
    for record in records:
        conn.execute(
            """INSERT INTO chip_definitions(id,name,number,chip_type,start_event,stop_event,updated_at)
            VALUES (?,?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET name=excluded.name, number=excluded.number,
              chip_type=excluded.chip_type, start_event=excluded.start_event,
              stop_event=excluded.stop_event, updated_at=excluded.updated_at""",
            (record.id, record.name, record.number, record.chip_type, record.start_event, record.stop_event, timestamp),
        )


def upsert_teams(conn: sqlite3.Connection, records: Iterable[TeamRecord], updated_at: str | None = None) -> None:
    timestamp = updated_at or utc_now()
    for record in records:
        if record.name is None:
            continue
        conn.execute(
            """INSERT INTO teams(
                id,code,name,short_name,strength,strength_overall_home,strength_overall_away,
                strength_attack_home,strength_attack_away,strength_defence_home,strength_defence_away,
                raw_json,updated_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET code=excluded.code, name=excluded.name,
              short_name=excluded.short_name, strength=excluded.strength,
              strength_overall_home=excluded.strength_overall_home, strength_overall_away=excluded.strength_overall_away,
              strength_attack_home=excluded.strength_attack_home, strength_attack_away=excluded.strength_attack_away,
              strength_defence_home=excluded.strength_defence_home, strength_defence_away=excluded.strength_defence_away,
              raw_json=excluded.raw_json, updated_at=excluded.updated_at""",
            (
                record.id, record.code, record.name, record.short_name, record.strength,
                record.strength_overall_home, record.strength_overall_away, record.strength_attack_home,
                record.strength_attack_away, record.strength_defence_home, record.strength_defence_away,
                _raw(record.raw_json), timestamp,
            ),
        )


def upsert_positions(conn: sqlite3.Connection, records: Iterable[PositionRecord], updated_at: str | None = None) -> None:
    timestamp = updated_at or utc_now()
    for record in records:
        conn.execute(
            """INSERT INTO positions(
                id,singular_name,singular_name_short,plural_name,squad_select,squad_min_play,squad_max_play,raw_json,updated_at
            ) VALUES (?,?,?,?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET singular_name=excluded.singular_name,
              singular_name_short=excluded.singular_name_short, plural_name=excluded.plural_name,
              squad_select=excluded.squad_select, squad_min_play=excluded.squad_min_play,
              squad_max_play=excluded.squad_max_play, raw_json=excluded.raw_json, updated_at=excluded.updated_at""",
            (
                record.id, record.singular_name, record.singular_name_short, record.plural_name,
                record.squad_select, record.squad_min_play, record.squad_max_play, _raw(record.raw_json), timestamp,
            ),
        )


def upsert_players(conn: sqlite3.Connection, records: Iterable[PlayerRecord], captured_at: str | None = None) -> set[int]:
    timestamp = captured_at or utc_now()
    seen: set[int] = set()
    for record in records:
        if not record.web_name:
            continue
        seen.add(record.id)
        conn.execute(
            """INSERT INTO players(
                id,code,first_name,second_name,web_name,full_name,norm_name,team_id,element_type,squad_number,
                opta_code,first_seen_at,last_seen_at,is_active,raw_json,updated_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET code=excluded.code, first_name=excluded.first_name,
              second_name=excluded.second_name, web_name=excluded.web_name, full_name=excluded.full_name,
              norm_name=excluded.norm_name, team_id=excluded.team_id, element_type=excluded.element_type,
              squad_number=excluded.squad_number, opta_code=excluded.opta_code, last_seen_at=excluded.last_seen_at,
              is_active=1, raw_json=excluded.raw_json, updated_at=excluded.updated_at""",
            (
                record.id, record.code, record.first_name, record.second_name, record.web_name, record.full_name,
                record.norm_name, record.team_id, record.element_type, record.squad_number, record.opta_code,
                timestamp, timestamp, 1, _raw(record.raw_json), timestamp,
            ),
        )
    return seen


def mark_absent_players(conn: sqlite3.Connection, seen: set[int]) -> None:
    if not seen:
        # An empty set is never credible evidence that the whole live catalog
        # disappeared.  Preserve activity and let the caller record the
        # validation failure instead.
        return
    placeholders = ",".join("?" for _ in seen)
    conn.execute(f"UPDATE players SET is_active=0 WHERE id NOT IN ({placeholders})", tuple(sorted(seen)))


def insert_snapshots(
    conn: sqlite3.Connection,
    records: Iterable[PlayerSnapshotRecord],
    fetch_run_id: int,
) -> int:
    columns = (
        "player_id,fetch_run_id,captured_at,event_context,now_cost,cost_change_event,cost_change_start,"
        "selected_by_percent,transfers_in_event,transfers_out_event,transfers_in,transfers_out,status,news,news_added,"
        "chance_of_playing_this_round,chance_of_playing_next_round,total_points,event_points,points_per_game,form,minutes,"
        "starts,goals_scored,assists,clean_sheets,goals_conceded,saves,bonus,bps,yellow_cards,red_cards,penalties_saved,"
        "penalties_missed,influence,creativity,threat,ict_index,expected_goals,expected_assists,expected_goal_involvements,"
        "expected_goals_conceded,expected_goals_per_90,expected_assists_per_90,expected_goal_involvements_per_90,"
        "expected_goals_conceded_per_90,defensive_contribution,clearances_blocks_interceptions,recoveries,tackles,"
        "ep_this,ep_next,value_form,value_season,raw_json"
    )
    sql = f"INSERT INTO player_snapshots({columns}) VALUES ({','.join('?' for _ in columns.split(','))})"
    count = 0
    for record in records:
        values = (
            record.player_id, fetch_run_id, record.captured_at, record.event_context, record.now_cost,
            record.cost_change_event, record.cost_change_start, record.selected_by_percent,
            record.transfers_in_event, record.transfers_out_event, record.transfers_in, record.transfers_out,
            record.status, record.news, record.news_added, record.chance_of_playing_this_round,
            record.chance_of_playing_next_round, record.total_points, record.event_points, record.points_per_game,
            record.form, record.minutes, record.starts, record.goals_scored, record.assists, record.clean_sheets,
            record.goals_conceded, record.saves, record.bonus, record.bps, record.yellow_cards, record.red_cards,
            record.penalties_saved, record.penalties_missed, record.influence, record.creativity, record.threat,
            record.ict_index, record.expected_goals, record.expected_assists, record.expected_goal_involvements,
            record.expected_goals_conceded, record.expected_goals_per_90, record.expected_assists_per_90,
            record.expected_goal_involvements_per_90, record.expected_goals_conceded_per_90,
            record.defensive_contribution, record.clearances_blocks_interceptions, record.recoveries,
            record.tackles, record.ep_this, record.ep_next, record.value_form, record.value_season, _raw(record.raw_json),
        )
        conn.execute(sql, values)
        count += 1
    return count


def upsert_fixtures(conn: sqlite3.Connection, records: Iterable[FixtureRecord], updated_at: str | None = None) -> int:
    timestamp = updated_at or utc_now()
    count = 0
    for record in records:
        conn.execute(
            """INSERT INTO fixtures(
              id,code,event,kickoff_time,team_h,team_a,team_h_score,team_a_score,team_h_difficulty,
              team_a_difficulty,started,finished,finished_provisional,provisional_start_time,minutes,
              stats_json,raw_json,updated_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET code=excluded.code,event=excluded.event,kickoff_time=excluded.kickoff_time,
              team_h=excluded.team_h,team_a=excluded.team_a,team_h_score=excluded.team_h_score,
              team_a_score=excluded.team_a_score,team_h_difficulty=excluded.team_h_difficulty,
              team_a_difficulty=excluded.team_a_difficulty,started=excluded.started,finished=excluded.finished,
              finished_provisional=excluded.finished_provisional,provisional_start_time=excluded.provisional_start_time,
              minutes=excluded.minutes,stats_json=excluded.stats_json,raw_json=excluded.raw_json,updated_at=excluded.updated_at""",
            (
                record.id, record.code, record.event, record.kickoff_time, record.team_h, record.team_a,
                record.team_h_score, record.team_a_score, record.team_h_difficulty, record.team_a_difficulty,
                record.started, record.finished, record.finished_provisional, record.provisional_start_time,
                record.minutes, _raw(record.stats_json), _raw(record.raw_json), timestamp,
            ),
        )
        count += 1
    return count


def delete_event_live_sentinels(conn: sqlite3.Connection, player_id: int, event: int) -> None:
    conn.execute(
        "DELETE FROM player_gameweeks WHERE player_id=? AND event=? AND fixture_id=-1",
        (player_id, event),
    )


_GAMEWEEK_DATA_COLUMNS = (
    "opponent_team", "was_home", "kickoff_time", "minutes", "starts", "total_points",
    "goals_scored", "assists", "clean_sheets", "goals_conceded", "saves", "bonus", "bps",
    "yellow_cards", "red_cards", "penalties_saved", "penalties_missed", "own_goals",
    "influence", "creativity", "threat", "ict_index", "expected_goals", "expected_assists",
    "expected_goal_involvements", "expected_goals_conceded", "defensive_contribution", "value",
    "selected", "transfers_in", "transfers_out", "transfers_balance", "source", "raw_json", "updated_at",
)
_GAMEWEEK_PERFORMANCE_COLUMNS = (
    "minutes", "starts", "total_points", "goals_scored", "assists", "clean_sheets", "goals_conceded",
    "saves", "bonus", "bps", "yellow_cards", "red_cards", "penalties_saved", "penalties_missed",
    "own_goals", "influence", "creativity", "threat", "ict_index", "expected_goals", "expected_assists",
    "expected_goal_involvements", "expected_goals_conceded", "defensive_contribution",
)


def _gameweek_has_performance(values: dict[str, Any]) -> bool:
    non_minutes = any(
        values.get(column) is not None
        for column in _GAMEWEEK_PERFORMANCE_COLUMNS
        if column != "minutes"
    )
    if non_minutes:
        return True
    minutes = values.get("minutes")
    if minutes is None:
        return False
    # Some scheduled summary placeholders expose minutes=0 but no completed
    # history fields.  A zero-minute completed history row normally also has
    # total_points/starts, which is covered above; do not let the placeholder
    # downgrade an existing live observation.
    if values.get("source") == "element_summary" and minutes == 0:
        return False
    return True


# Public alias: completeness/provenance code must read the SAME column set the
# upsert path uses, so the two definitions cannot drift apart.
GAMEWEEK_PERFORMANCE_COLUMNS = _GAMEWEEK_PERFORMANCE_COLUMNS


def gameweek_has_performance_sql(alias: str = "pg") -> str:
    """``_gameweek_has_performance`` as a WHERE predicate for one row alias.

    This is not a second definition of the signature, it is the same one
    written for SQL: the field list comes from ``_GAMEWEEK_PERFORMANCE_COLUMNS``
    and the two branches mirror ``_gameweek_has_performance`` exactly, so the
    Python and SQL forms cannot drift without the parity test failing.  A
    caller that wants the placeholder predicate applies ``NOT`` to this, which
    is literally how :func:`row_is_scheduled_placeholder` defines it.

    ``COALESCE(source, '')`` is load-bearing.  Without it, SQL three-valued
    logic makes ``source IS NULL AND minutes = 0`` evaluate to NULL, which
    excludes the row, while the Python predicate includes it -- a divergence no
    amount of column-list sharing would repair.
    """

    non_minutes = [column for column in _GAMEWEEK_PERFORMANCE_COLUMNS if column != "minutes"]
    # The outer parentheses are load-bearing: without them the top-level ``OR``
    # binds looser than a composing ``AND``, so ``a AND <predicate> AND b``
    # would parse as ``(a AND (A)) OR ((B) AND b)`` -- silently dropping the
    # caller's remaining predicates on one branch and fanning the row set out.
    return (
        "((" + " OR ".join(f"{alias}.{column} IS NOT NULL" for column in non_minutes) + ")"
        f" OR ({alias}.minutes IS NOT NULL AND NOT "
        f"(COALESCE({alias}.source, '') = 'element_summary' AND {alias}.minutes = 0)))"
    )


def scheduled_placeholder_sql(alias: str = "pg") -> str:
    """The placeholder predicate: the exact negation of the above."""

    return f"NOT ({gameweek_has_performance_sql(alias)})"


def row_is_scheduled_placeholder(values: Mapping[str, Any]) -> bool:
    """True when a player-fixture row carries no official observation at all.

    Measured against the certified R5 source snapshot: a REAL official row
    populates every performance column, because a genuine did-not-play is all
    explicit zeros (``minutes=0``, ``starts=0``, ``total_points=0`` ...,
    ``expected_goals=0.0``).  A schedule placeholder instead has ``minutes``
    0/NULL and every remaining performance column NULL.  "No performance
    evidence" is therefore exactly the placeholder signature, and it is never a
    claim that the player did not play.
    """

    return not _gameweek_has_performance(dict(values))


def _gameweek_record_values(record: PlayerGameweekRecord, fixture_id: int, timestamp: str) -> dict[str, Any]:
    values = {column: getattr(record, column) for column in _GAMEWEEK_DATA_COLUMNS if column != "updated_at"}
    values.update({"player_id": record.player_id, "event": record.event, "fixture_id": fixture_id, "updated_at": timestamp})
    values["raw_json"] = _raw(record.raw_json)
    return values


def upsert_player_gameweeks(
    conn: sqlite3.Connection,
    records: Iterable[PlayerGameweekRecord],
    updated_at: str | None = None,
) -> int:
    timestamp = updated_at or utc_now()
    count = 0
    for record in records:
        fixture_id = record.fixture_id if record.fixture_id is not None else -1
        incoming = _gameweek_record_values(record, fixture_id, timestamp)
        if record.source == "element_summary" and fixture_id != -1 and _gameweek_has_performance(incoming):
            # A completed summary row is a real fixture-level observation and
            # can supersede an older event-total sentinel.  Future schedule
            # rows do not enter this branch and therefore cannot erase it.
            delete_event_live_sentinels(conn, record.player_id, record.event)
        existing = conn.execute(
            "SELECT * FROM player_gameweeks WHERE player_id=? AND event=? AND fixture_id=?",
            (record.player_id, record.event, fixture_id),
        ).fetchone()
        if existing is None:
            columns = ["player_id", "event", "fixture_id", *_GAMEWEEK_DATA_COLUMNS]
            conn.execute(
                f"INSERT INTO player_gameweeks({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",
                tuple(incoming[column] for column in columns),
            )
        else:
            previous = dict(existing)
            merged = {
                column: incoming[column] if incoming[column] is not None else previous[column]
                for column in _GAMEWEEK_DATA_COLUMNS
            }
            incoming_has_performance = _gameweek_has_performance(incoming)
            previous_has_performance = _gameweek_has_performance(previous)
            if not incoming_has_performance and previous_has_performance:
                for column in _GAMEWEEK_PERFORMANCE_COLUMNS:
                    merged[column] = previous[column]
                merged["source"] = previous["source"]
                merged["raw_json"] = previous["raw_json"]
            elif incoming_has_performance:
                merged["source"] = record.source
                merged["raw_json"] = incoming["raw_json"]
            conn.execute(
                "UPDATE player_gameweeks SET "
                + ",".join(f"{column}=?" for column in _GAMEWEEK_DATA_COLUMNS)
                + " WHERE player_id=? AND event=? AND fixture_id=?",
                tuple(merged[column] for column in _GAMEWEEK_DATA_COLUMNS)
                + (record.player_id, record.event, fixture_id),
            )
        count += 1
    return count


def _official_club_at(conn: sqlite3.Connection, player_id: int, moment: str | None) -> int | None:
    """The player's official club as of ``moment``, from CAUSALLY VALID captures.

    Destruction of a started placeholder may only be authorised by evidence that
    was genuinely available: an official observation qualifies only when

    A. the response that carried it COMPLETED no later than ``moment`` -- the run's
       ``finished_at``, never ``captured_at``.  ``run_fetch`` takes ``captured_at``
       before the HTTP request is issued, so a response received after a fixture
       kicked off can still carry a pre-kickoff ``captured_at`` (the accepted
       snapshot shows windows of ~5-8 minutes), and request-start time is not
       availability; and
    B. it belongs to an ACCEPTED bootstrap generation -- the canonical official
       state.  Snapshots from rejected generations are recorded for audit only and
       are never authoritative.

    Returns ``None`` whenever either condition cannot be proven, and the caller
    MUST fail safe on ``None``.  The player's CURRENT club is never consulted: it
    says nothing about membership at kickoff.
    """

    if moment is None:
        return None
    row = conn.execute(
        """SELECT ps.raw_json
             FROM player_snapshots ps
             JOIN fetch_runs fr ON fr.id = ps.fetch_run_id
             JOIN bootstrap_generations bg
               ON bg.fetch_run_id = ps.fetch_run_id AND bg.accepted = 1
            WHERE ps.player_id = ?
              AND fr.finished_at IS NOT NULL
              AND fr.finished_at <= ?
            ORDER BY fr.finished_at DESC, ps.id DESC
            LIMIT 1""",
        (int(player_id), str(moment)),
    ).fetchone()
    if row is None or not row["raw_json"]:
        return None
    try:
        payload = json.loads(row["raw_json"])
    except (TypeError, ValueError):
        return None
    team = payload.get("team") if isinstance(payload, dict) else None
    if team is None:
        return None
    try:
        return int(team)
    except (TypeError, ValueError):
        return None


def prune_stale_element_summary_placeholders(
    conn: sqlite3.Connection,
    player_id: int,
    claimed_pairs: Iterable[tuple[int, int]],
) -> int:
    """Delete stale schedule-placeholder rows after a fresh element-summary fetch.

    An official element-summary response claims exactly the `(event, fixture)`
    pairs in its `history` and `fixtures` rows.  Pure schedule placeholders
    (source `element_summary`, no performance data) for pairs the latest
    payload no longer claims are stale — usually because the player's club
    context changed between syncs, or because the fixture was officially
    reassigned to another event — and otherwise stay in the canonical grain
    forever, where they corrupt appearance and fixture counts.  Performance
    rows and sentinel/blank rows are never touched.

    PROSPECTIVE vs OWED slots
    -------------------------
    For a fixture that has NOT started the slot is prospective: the pre-existing
    rule stands and an unclaimed placeholder belonging to the player's current
    club is retained while any other is pruned.

    Once the fixture HAS STARTED the placeholder is the only surviving evidence
    that this player-fixture observation is unresolved, and a row that does not
    exist cannot be reported as a gap.  Deletion then requires causally valid
    evidence, and the player's CURRENT club is NOT such evidence — it is not proof
    of membership at kickoff:

      * point-in-time official captures prove the player belonged to a club that
        did not play this fixture at kickoff -> safely pruned (the placeholder was
        already invalid before the fixture began);
      * authoritative replacement exists at a pair the payload itself claims
        -> the stale label is removed because the observation is represented;
      * otherwise -> RETAIN (fail safe), so an owed observation surfaces later as
        ``COMPLETED_EVENT_PLACEHOLDER_ROW`` instead of disappearing from the audit.

    Nothing is fabricated: no zeros, no synthetic observations, and no row is ever
    relocated to a club the player is not currently at.
    """

    claimed = {(int(event), int(fixture_id)) for event, fixture_id in claimed_pairs}
    claimed_fixtures = {fixture_id for _, fixture_id in claimed}
    team_row = conn.execute("SELECT team_id FROM players WHERE id=?", (int(player_id),)).fetchone()
    current_team = int(team_row[0]) if team_row and team_row[0] is not None else None
    deleted = 0
    for stale in conn.execute(
        "SELECT event, fixture_id FROM player_gameweeks WHERE player_id=? AND source='element_summary'",
        (int(player_id),),
    ).fetchall():
        event = stale["event"]
        fixture_id = stale["fixture_id"]
        if fixture_id in (None, -1):
            continue
        if (int(event), int(fixture_id)) in claimed:
            continue
        stored = conn.execute(
            "SELECT * FROM player_gameweeks WHERE player_id=? AND event=? AND fixture_id=?",
            (int(player_id), int(event), int(fixture_id)),
        ).fetchone()
        if stored is None or _gameweek_has_performance(dict(stored)):
            continue
        fixture = conn.execute(
            "SELECT team_h, team_a, started, finished, kickoff_time FROM fixtures WHERE id=?",
            (int(fixture_id),),
        ).fetchone()
        if fixture is None:
            # Cannot establish anything about this fixture; retaining is the
            # recoverable choice, deleting is not.
            continue
        involves_current_club = current_team is not None and current_team in (
            fixture["team_h"], fixture["team_a"]
        )
        fixture_started = bool(fixture["started"]) or bool(fixture["finished"])
        if int(fixture_id) in claimed_fixtures:
            # The payload itself says this fixture belongs to another event, so the
            # local (event, fixture) label is stale.  For a STARTED fixture the
            # observation must survive the relabel, so the stale label may go only
            # once the observation is represented at one of the claimed pairs.
            if fixture_started:
                present = {
                    int(row[0])
                    for row in conn.execute(
                        "SELECT event FROM player_gameweeks WHERE player_id=? AND fixture_id=?",
                        (int(player_id), int(fixture_id)),
                    ).fetchall()
                }
                replacement_events = {pair[0] for pair in claimed if pair[1] == int(fixture_id)}
                if not (replacement_events & present):
                    continue
            conn.execute(
                "DELETE FROM player_gameweeks WHERE player_id=? AND event=? AND fixture_id=?",
                (int(player_id), int(event), int(fixture_id)),
            )
            deleted += 1
            continue
        if not fixture_started:
            # Prospective slot: unchanged behaviour (the current club's planned
            # fixture is kept, anything else stale is removed).
            if involves_current_club:
                continue
            conn.execute(
                "DELETE FROM player_gameweeks WHERE player_id=? AND event=? AND fixture_id=?",
                (int(player_id), int(event), int(fixture_id)),
            )
            deleted += 1
            continue
        # The fixture HAS started, so the placeholder is the only surviving evidence
        # of an unresolved observation.  Membership at kickoff is decided by
        # point-in-time official captures, NEVER by today's club.
        club_at_kickoff = _official_club_at(conn, int(player_id), fixture["kickoff_time"])
        if club_at_kickoff is not None and club_at_kickoff not in (
            fixture["team_h"], fixture["team_a"]
        ):
            # Provably not this player's fixture before it began.
            conn.execute(
                "DELETE FROM player_gameweeks WHERE player_id=? AND event=? AND fixture_id=?",
                (int(player_id), int(event), int(fixture_id)),
            )
            deleted += 1
            continue
        # Owed, or membership at kickoff unprovable: fail safe (retain).
    return deleted


def enrich_gameweek_context(
    conn: sqlite3.Connection,
    records: Iterable[PlayerGameweekRecord],
) -> list[PlayerGameweekRecord]:
    """Fill live fixture context from already-persisted fixture/team facts."""

    enriched: list[PlayerGameweekRecord] = []
    for record in records:
        if record.fixture_id in (None, -1):
            enriched.append(record)
            continue
        fixture = conn.execute(
            "SELECT team_h,team_a,kickoff_time FROM fixtures WHERE id=?",
            (record.fixture_id,),
        ).fetchone()
        player = conn.execute("SELECT team_id FROM players WHERE id=?", (record.player_id,)).fetchone()
        if fixture is None or player is None or player[0] is None:
            enriched.append(record)
            continue
        team_id = int(player[0])
        was_home = record.was_home
        opponent = record.opponent_team
        if was_home is None and fixture[0] is not None and fixture[1] is not None:
            if int(fixture[0]) == team_id:
                was_home = 1
            elif int(fixture[1]) == team_id:
                was_home = 0
        if opponent is None:
            if was_home == 1:
                opponent = fixture[1]
            elif was_home == 0:
                opponent = fixture[0]
        enriched.append(
            record.__class__(
                **{
                    **record.__dict__,
                    "was_home": was_home,
                    "opponent_team": opponent,
                    "kickoff_time": record.kickoff_time or fixture[2],
                }
            )
        )
    return enriched


def insert_manager_state(
    conn: sqlite3.Connection,
    entry: EntryRecord,
    history_row: HistoryRow | None,
    active_chip: str | None,
    free_transfers_manual: int | None,
    fetch_run_id: int | None,
    event: int | None,
    raw_json: Any,
    captured_at: str | None = None,
) -> int:
    row = history_row or HistoryRow()
    cursor = conn.execute(
        """INSERT INTO manager_state(
          entry_id,fetch_run_id,captured_at,event,player_name,team_name,summary_overall_points,summary_overall_rank,
          summary_event_points,summary_event_rank,bank,team_value,total_transfers,event_transfers,event_transfers_cost,
          points_on_bench,active_chip,free_transfers_manual,raw_json
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            entry.entry_id, fetch_run_id, captured_at or utc_now(), event, entry.player_name, entry.team_name,
            entry.summary_overall_points, entry.summary_overall_rank, entry.summary_event_points, entry.summary_event_rank,
            entry.bank if entry.bank is not None else row.bank,
            entry.team_value if entry.team_value is not None else row.value,
            entry.total_transfers if entry.total_transfers is not None else row.total_transfers,
            row.event_transfers, row.event_transfers_cost, row.points_on_bench, active_chip,
            free_transfers_manual, _raw(raw_json),
        ),
    )
    return int(cursor.lastrowid)


def upsert_manual_manager_state(
    conn: sqlite3.Connection,
    entry_id: int,
    event: int,
    free_transfers: int | None,
    bank: int | None,
    *,
    source: str = "manual",
    captured_at: str | None = None,
    event_start_free_transfers: int | None = None,
) -> dict[str, Any]:
    """Store the manager-entered transfer state for one entry and event.

    The current-state row keeps its exact `(entry_id, event)` scope, and every
    entry is additionally appended to `manager_state_observations` so multiple
    valid states inside one event are never lost to the upsert.

    ``event_start_free_transfers`` is the FT bank at the START of the Gameweek
    (before any current-event transfer).  It is stored explicitly because a
    Wildcard played after transfers were already made must preserve that value
    rather than the current remaining count; NULL means "not stated".
    """

    if isinstance(free_transfers, bool) or (free_transfers is not None and not isinstance(free_transfers, int)):
        raise ValueError("free_transfers must be an integer or null")
    if isinstance(bank, bool) or (bank is not None and not isinstance(bank, int)):
        raise ValueError("bank must be an integer tenths value or null")
    if free_transfers is not None and free_transfers < 0:
        raise ValueError("free_transfers must not be negative")
    if bank is not None and bank < 0:
        raise ValueError("bank must not be negative")
    if event_start_free_transfers is not None and (
        isinstance(event_start_free_transfers, bool) or not isinstance(event_start_free_transfers, int)
        or event_start_free_transfers < 0
    ):
        raise ValueError("event_start_free_transfers must be a non-negative integer or null")
    if not source or not source.strip():
        raise ValueError("source must not be empty")
    timestamp = captured_at or utc_now()
    conn.execute(
        """INSERT INTO manager_manual_state(entry_id,event,free_transfers,bank,source,captured_at,
             event_start_free_transfers)
           VALUES (?,?,?,?,?,?,?)
           ON CONFLICT(entry_id,event) DO UPDATE SET
             free_transfers=excluded.free_transfers,
             bank=excluded.bank,
             source=excluded.source,
             captured_at=excluded.captured_at,
             event_start_free_transfers=COALESCE(excluded.event_start_free_transfers,
                                                 manager_manual_state.event_start_free_transfers)""",
        (int(entry_id), int(event), free_transfers, bank, source, timestamp, event_start_free_transfers),
    )
    conn.execute(
        """INSERT INTO manager_state_observations(entry_id,event,free_transfers,bank,source,captured_at,
             created_at,event_start_free_transfers)
           VALUES (?,?,?,?,?,?,?,?)""",
        (int(entry_id), int(event), free_transfers, bank, source, timestamp, timestamp,
         event_start_free_transfers),
    )
    stored = get_manual_manager_state(conn, entry_id, event)
    if stored is None:
        raise RuntimeError("manual manager state was not stored")
    return stored


def list_manager_state_observations(
    conn: sqlite3.Connection,
    entry_id: int,
    event: int,
    *,
    as_of: str | None = None,
) -> list[dict[str, Any]]:
    """Return the append-only manual state history for one event, oldest first."""

    if as_of is None:
        rows = conn.execute(
            """SELECT * FROM manager_state_observations
               WHERE entry_id=? AND event=?
               ORDER BY captured_at, id""",
            (int(entry_id), int(event)),
        )
    else:
        rows = conn.execute(
            """SELECT * FROM manager_state_observations
               WHERE entry_id=? AND event=? AND captured_at<=?
               ORDER BY captured_at, id""",
            (int(entry_id), int(event), as_of),
        )
    return _rows(rows)


def manual_manager_state_as_of(
    conn: sqlite3.Connection,
    entry_id: int,
    event: int,
    as_of: str | None = None,
) -> dict[str, Any] | None:
    """Latest manual state observation for the event, optionally as of a timestamp."""

    observations = list_manager_state_observations(conn, entry_id, event, as_of=as_of)
    if as_of is None:
        row = conn.execute(
            """SELECT * FROM manager_manual_state WHERE entry_id=? AND event=?""",
            (int(entry_id), int(event)),
        ).fetchone()
        return dict(row) if row else None
    return observations[-1] if observations else None


def get_manual_manager_state(
    conn: sqlite3.Connection,
    entry_id: int,
    event: int,
) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT * FROM manager_manual_state WHERE entry_id=? AND event=?",
        (int(entry_id), int(event)),
    ).fetchone()
    return dict(row) if row else None


def upsert_manager_selling_prices(
    conn: sqlite3.Connection,
    entry_id: int,
    event: int,
    prices: Mapping[int, int],
    *,
    source: str = "manual",
    captured_at: str | None = None,
    market_prices_at_capture: Mapping[int, int] | None = None,
) -> int:
    """Store exact manager selling prices without changing official snapshots."""

    if not source or not source.strip():
        raise ValueError("source must not be empty")
    timestamp = captured_at or utc_now()
    count = 0
    for player_id, selling_price in prices.items():
        if isinstance(selling_price, bool) or not isinstance(selling_price, int) or selling_price < 0:
            raise ValueError("selling_price must be a non-negative integer tenths value")
        market_price = None if market_prices_at_capture is None else market_prices_at_capture.get(int(player_id))
        if market_price is not None and (isinstance(market_price, bool) or not isinstance(market_price, int) or market_price < 0):
            raise ValueError("market_price_at_capture must be a non-negative integer tenths value")
        conn.execute(
            """INSERT INTO manager_selling_prices(
                 entry_id,event,player_id,selling_price,source,captured_at,market_price_at_capture
               ) VALUES (?,?,?,?,?,?,?)
               ON CONFLICT(entry_id,event,player_id) DO UPDATE SET
                 selling_price=excluded.selling_price,
                 source=excluded.source,
                 captured_at=excluded.captured_at,
                 market_price_at_capture=COALESCE(excluded.market_price_at_capture, manager_selling_prices.market_price_at_capture)""",
            (int(entry_id), int(event), int(player_id), selling_price, source, timestamp, market_price),
        )
        conn.execute(
            """INSERT INTO manager_selling_price_observations(
                 entry_id,event,player_id,selling_price,market_price_at_capture,source,captured_at,created_at
               ) VALUES (?,?,?,?,?,?,?,?)""",
            (int(entry_id), int(event), int(player_id), selling_price, market_price, source, timestamp, timestamp),
        )
        count += 1
    return count


def list_manager_selling_price_observations(
    conn: sqlite3.Connection,
    entry_id: int,
    event: int,
    player_id: int | None = None,
    *,
    as_of: str | None = None,
) -> list[dict[str, Any]]:
    """Return append-only manual selling-price observations, oldest first."""

    clauses = ["entry_id=?", "event=?"]
    params: list[Any] = [int(entry_id), int(event)]
    if player_id is not None:
        clauses.append("player_id=?")
        params.append(int(player_id))
    if as_of is not None:
        clauses.append("captured_at<=?")
        params.append(as_of)
    return _rows(
        conn.execute(
            f"""SELECT * FROM manager_selling_price_observations
                WHERE {' AND '.join(clauses)}
                ORDER BY player_id, captured_at, id""",
            tuple(params),
        )
    )


def get_manager_selling_prices(
    conn: sqlite3.Connection,
    entry_id: int,
    event: int,
) -> list[dict[str, Any]]:
    return _rows(
        conn.execute(
            """SELECT * FROM manager_selling_prices
               WHERE entry_id=? AND event=?
               ORDER BY player_id""",
            (int(entry_id), int(event)),
        )
    )


def get_manager_selling_price(
    conn: sqlite3.Connection,
    entry_id: int,
    event: int,
    player_id: int,
) -> dict[str, Any] | None:
    row = conn.execute(
        """SELECT * FROM manager_selling_prices
           WHERE entry_id=? AND event=? AND player_id=?""",
        (int(entry_id), int(event), int(player_id)),
    ).fetchone()
    return dict(row) if row else None


def upsert_player_season_histories(
    conn: sqlite3.Connection,
    records: Iterable[Any],
    observed_at: str | None = None,
) -> int:
    """Store official prior-season rows (append/upsert by player + season)."""

    timestamp = observed_at or utc_now()
    count = 0
    seen_per_player_season: set[tuple[int, str]] = set()
    for record in records:
        player_id = int(record.player_id)
        season_name = str(record.season_name)
        key = (player_id, season_name)
        if key in seen_per_player_season:
            continue
        seen_per_player_season.add(key)
        conn.execute(
            """INSERT INTO player_season_histories(
                 player_id, season_name, minutes, starts, total_points,
                 goals_scored, assists, clean_sheets, bonus, saves,
                 source, observed_at, raw_json
               ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(player_id, season_name) DO UPDATE SET
                 minutes=excluded.minutes, starts=excluded.starts,
                 total_points=excluded.total_points, goals_scored=excluded.goals_scored,
                 assists=excluded.assists, clean_sheets=excluded.clean_sheets,
                 bonus=excluded.bonus, saves=excluded.saves,
                 source=excluded.source, observed_at=excluded.observed_at,
                 raw_json=excluded.raw_json""",
            (
                player_id, season_name, record.minutes, record.starts, record.total_points,
                record.goals_scored, record.assists, record.clean_sheets, record.bonus, record.saves,
                "element_summary_history_past", timestamp, _raw(record.raw_json),
            ),
        )
        count += 1
    return count


def player_season_histories(
    conn: sqlite3.Connection,
    player_id: int | None = None,
) -> list[dict[str, Any]]:
    """Previous-season official history rows, most recent season first."""

    if player_id is None:
        rows = conn.execute(
            "SELECT * FROM player_season_histories ORDER BY season_name DESC, player_id"
        ).fetchall()
        return [dict(row) for row in rows]
    rows = conn.execute(
        "SELECT * FROM player_season_histories WHERE player_id=? ORDER BY season_name DESC",
        (int(player_id),),
    ).fetchall()
    return [dict(row) for row in rows]


def insert_manager_acquisition(
    conn: sqlite3.Connection,
    entry_id: int,
    player_id: int,
    acquired_event: int,
    purchase_price: int,
    *,
    source: str,
    acquired_at: str | None = None,
    sold_event: int | None = None,
    sold_at: str | None = None,
    created_at: str | None = None,
) -> int:
    if source not in {"verified_initial_squad", "official_transfer_history", "manual", "reconciled"}:
        raise ValueError("unsupported acquisition source")
    if isinstance(purchase_price, bool) or not isinstance(purchase_price, int) or purchase_price < 0:
        raise ValueError("purchase_price must be a non-negative integer tenths value")
    timestamp = created_at or utc_now()
    cursor = conn.execute(
        """INSERT INTO manager_player_acquisitions(
             entry_id,player_id,acquired_event,purchase_price,sold_event,source,acquired_at,sold_at,created_at,updated_at
           ) VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (int(entry_id), int(player_id), int(acquired_event), int(purchase_price), sold_event, source, acquired_at, sold_at, timestamp, timestamp),
    )
    return int(cursor.lastrowid)


def close_manager_acquisition(
    conn: sqlite3.Connection,
    acquisition_id: int,
    sold_event: int,
    sold_at: str | None = None,
    updated_at: str | None = None,
) -> None:
    conn.execute(
        "UPDATE manager_player_acquisitions SET sold_event=?, sold_at=?, updated_at=? WHERE id=?",
        (int(sold_event), sold_at, updated_at or utc_now(), int(acquisition_id)),
    )


def list_manager_acquisitions(
    conn: sqlite3.Connection,
    entry_id: int,
    player_id: int | None = None,
) -> list[dict[str, Any]]:
    clauses = ["entry_id=?"]
    params: list[Any] = [int(entry_id)]
    if player_id is not None:
        clauses.append("player_id=?")
        params.append(int(player_id))
    return _rows(
        conn.execute(
            "SELECT * FROM manager_player_acquisitions WHERE " + " AND ".join(clauses) + " ORDER BY acquired_event,id",
            tuple(params),
        )
    )


def active_manager_acquisitions(conn: sqlite3.Connection, entry_id: int) -> list[dict[str, Any]]:
    return _rows(
        conn.execute(
            "SELECT * FROM manager_player_acquisitions WHERE entry_id=? AND sold_event IS NULL ORDER BY player_id",
            (int(entry_id),),
        )
    )


DIAG_ACQUISITION_TIMESTAMPS_INCOMPLETE = "ACQUISITION_TIMESTAMPS_INCOMPLETE"
DIAG_ACQUISITION_CUTOFF_BASIS = "ACQUISITION_CUTOFF_BASIS"


def acquisition_timestamp_coverage(conn: sqlite3.Connection, entry_id: int) -> dict[str, Any]:
    """Whether the acquisition ledger can support cutoff-based as-of resolution.

    A transfer inside the SAME Gameweek cannot be distinguished by event numbers
    alone, so cutoff-accurate ownership needs transaction timestamps.  This
    reports exactly how much of the ledger has them instead of guessing.
    """

    rows = _rows(
        conn.execute(
            "SELECT acquired_event, sold_event, acquired_at, sold_at, created_at FROM"
            " manager_player_acquisitions WHERE entry_id=?",
            (int(entry_id),),
        )
    )
    total = len(rows)
    with_acquired = sum(1 for row in rows if row.get("acquired_at"))
    sold = [row for row in rows if row.get("sold_event") is not None]
    with_sold = sum(1 for row in sold if row.get("sold_at"))
    complete = total > 0 and with_acquired == total and with_sold == len(sold)
    return {
        "total_rows": total,
        "rows_with_acquired_at": with_acquired,
        "sold_rows": len(sold),
        "sold_rows_with_sold_at": with_sold,
        "timestamps_complete": complete,
        "diagnostic": None if complete else DIAG_ACQUISITION_TIMESTAMPS_INCOMPLETE,
    }


def active_manager_acquisitions_as_of_cutoff(
    conn: sqlite3.Connection, entry_id: int, cutoff: str
) -> dict[str, Any]:
    """Ownership as of a CUTOFF instant, using transaction timestamps.

    Returns ``{"rows": [...], "basis": "transaction_timestamps"|"unavailable",
    "diagnostic": ...}``.  When the ledger lacks timestamps this does NOT fake an
    as-of reconstruction: it reports ``ACQUISITION_TIMESTAMPS_INCOMPLETE`` and the
    caller must use the authoritative source (the immutable execution snapshot /
    explicit user-confirmed manager state captured at the cutoff).
    """

    coverage = acquisition_timestamp_coverage(conn, entry_id)
    if not coverage["timestamps_complete"]:
        return {
            "rows": [],
            "basis": "unavailable",
            "coverage": coverage,
            "diagnostic": DIAG_ACQUISITION_TIMESTAMPS_INCOMPLETE,
            "authoritative_alternative": (
                "immutable execution snapshot / explicit user-confirmed manager state captured at "
                "the cutoff"
            ),
        }
    rows = _rows(
        conn.execute(
            "SELECT * FROM manager_player_acquisitions WHERE entry_id=?"
            " AND acquired_at<=? AND (sold_at IS NULL OR sold_at>?) ORDER BY player_id",
            (int(entry_id), str(cutoff), str(cutoff)),
        )
    )
    return {
        "rows": rows,
        "basis": "transaction_timestamps",
        "coverage": coverage,
        "diagnostic": None,
    }


def active_manager_acquisitions_as_of(
    conn: sqlite3.Connection, entry_id: int, event: int
) -> list[dict[str, Any]]:
    """Ownership reconstructed as of the START of ``event``.

    ``active_manager_acquisitions`` uses the CURRENT ledger, so a later sale
    removes a player from the reconstruction of an earlier planning state -- a
    later transfer leaking backward.  As-of semantics keep a player who was
    acquired at or before the event and had not yet been sold by it.
    """

    return _rows(
        conn.execute(
            "SELECT * FROM manager_player_acquisitions WHERE entry_id=?"
            " AND acquired_event<=? AND (sold_event IS NULL OR sold_event>?)"
            " ORDER BY player_id",
            (int(entry_id), int(event), int(event)),
        )
    )


def manager_chip_events(conn: sqlite3.Connection, entry_id: int, chip_name: str) -> set[int]:
    """Official events on which this entry played the named chip."""

    return {
        int(row["event"])
        for row in conn.execute(
            "SELECT DISTINCT event FROM manager_chips WHERE entry_id=? AND name=? AND event IS NOT NULL",
            (int(entry_id), chip_name),
        ).fetchall()
    }


def _complete_squad_rows(rows: list[dict[str, Any]], entry_id: int, event: int) -> bool:
    if len(rows) != 15:
        return False
    try:
        return (
            all(int(row["entry_id"]) == int(entry_id) and int(row["event"]) == int(event) for row in rows)
            and sorted(int(row["position"]) for row in rows) == list(range(1, 16))
            and len({int(row["player_id"]) for row in rows}) == 15
        )
    except (KeyError, TypeError, ValueError):
        return False


def complete_squad_rows(conn: sqlite3.Connection, entry_id: int, event: int) -> list[dict[str, Any]]:
    rows = squad_rows(conn, int(entry_id), int(event))
    return rows if _complete_squad_rows(rows, int(entry_id), int(event)) else []


def latest_complete_squad_rows(
    conn: sqlite3.Connection,
    entry_id: int,
    event: int,
    *,
    ascending: bool = False,
    exclude_events: set[int] | frozenset[int] | None = None,
) -> tuple[int, list[dict[str, Any]]] | None:
    """Latest complete squad at or before ``event``, optionally skipping events.

    Chip weeks (for example a used Free Hit) hold temporary squads, so callers
    that need a *permanent* squad base pass those events in ``exclude_events``.
    """
    events = conn.execute(
        "SELECT DISTINCT event FROM squad_picks WHERE entry_id=? AND event<=? ORDER BY event " + ("ASC" if ascending else "DESC"),
        (int(entry_id), int(event)),
    ).fetchall()
    excluded = {int(value) for value in (exclude_events or frozenset())}
    for row in events:
        candidate_event = int(row[0])
        if candidate_event in excluded:
            continue
        rows = complete_squad_rows(conn, int(entry_id), candidate_event)
        if rows:
            return candidate_event, rows
    return None


def manager_transfer_history(conn: sqlite3.Connection, entry_id: int) -> tuple[bool, list[dict[str, Any]]]:
    state = latest_manager_state(conn, int(entry_id))
    if state is None:
        return False, []
    try:
        raw = json.loads(state.get("raw_json") or "{}")
    except (TypeError, ValueError, json.JSONDecodeError):
        return False, []
    transfers = raw.get("transfers")
    return bool(raw.get("transfers_endpoint_available")), [dict(row) for row in transfers if isinstance(row, dict)] if isinstance(transfers, list) else []


def manager_planning_state(
    conn: sqlite3.Connection,
    entry_id: int,
    event: int,
    as_of: str | None = None,
) -> dict[str, Any]:
    """Resolve event-scoped manual state before legacy/API manager state."""

    manual = manual_manager_state_as_of(conn, int(entry_id), int(event), as_of)
    # The official/legacy fallback is bounded by the same as_of instant.  Reading
    # the newest manager_state regardless of the cutoff let state captured AFTER
    # the planning cutoff supply bank/free transfers to an earlier plan.
    if as_of:
        exact_api = conn.execute(
            "SELECT * FROM manager_state WHERE entry_id=? AND event=? AND captured_at<=?"
            " ORDER BY captured_at DESC,id DESC LIMIT 1",
            (int(entry_id), int(event), str(as_of)),
        ).fetchone()
        latest = conn.execute(
            "SELECT * FROM manager_state WHERE entry_id=? AND captured_at<=?"
            " ORDER BY captured_at DESC, id DESC LIMIT 1",
            (int(entry_id), str(as_of)),
        ).fetchone()
        latest = dict(latest) if latest else None
    else:
        exact_api = conn.execute(
            "SELECT * FROM manager_state WHERE entry_id=? AND event=? ORDER BY captured_at DESC,id DESC LIMIT 1",
            (int(entry_id), int(event)),
        ).fetchone()
        latest = latest_manager_state(conn, int(entry_id))
    api = dict(exact_api) if exact_api else None
    free_transfers = None
    free_source = "data_gap"
    bank = None
    bank_source = "data_gap"
    if manual is not None:
        free_transfers = manual.get("free_transfers")
        free_source = "manual" if free_transfers is not None else "data_gap"
        bank = manual.get("bank")
        bank_source = "manual" if bank is not None else "data_gap"
    if manual is None:
        legacy_api = api or latest
        if free_transfers is None and legacy_api is not None and legacy_api.get("free_transfers_manual") is not None:
            free_transfers = legacy_api.get("free_transfers_manual")
            free_source = "legacy_manager_state"
        if bank is None and api is not None and api.get("bank") is not None:
            bank = api.get("bank")
            bank_source = "manager_state"
    official_api = None if api is None else {
        "free_transfers": api.get("free_transfers_manual"),
        "bank": api.get("bank"),
        "captured_at": api.get("captured_at"),
        "active_chip": api.get("active_chip"),
        "event_transfers": api.get("event_transfers"),
        "event_transfers_cost": api.get("event_transfers_cost"),
    }
    stale_official_overridden = bool(
        manual is not None
        and official_api is not None
        and (
            (official_api["free_transfers"] is not None and manual.get("free_transfers") is not None
             and int(official_api["free_transfers"]) != int(manual["free_transfers"]))
            or (official_api["bank"] is not None and manual.get("bank") is not None
                and int(official_api["bank"]) != int(manual["bank"]))
        )
    )
    if manual is not None:
        authoritative = "user_confirmed_override"
    elif free_transfers is not None or bank is not None:
        authoritative = "official_manager_state"
    else:
        authoritative = "data_gap"
    # Event-start FT is EXPLICIT provenance (never inferred): it is required for
    # the Wildcard/Free-Hit FT transition, which preserves the FT bank from the
    # start of the Gameweek even when the chip is played after transfers.
    event_start_ft = manual.get("event_start_free_transfers") if manual is not None else None
    return {
        "entry_id": int(entry_id),
        "event": int(event),
        "free_transfers": free_transfers,
        "free_transfers_source": free_source,
        "bank": bank,
        "bank_source": bank_source,
        "event_start_free_transfers": event_start_ft,
        "event_start_free_transfers_source": "manual" if event_start_ft is not None else "data_gap",
        "manual": manual,
        "api": api,
        "latest_api": latest,
        "authoritative_source": authoritative,
        "override": None if manual is None else {
            "free_transfers": manual.get("free_transfers"),
            "bank": manual.get("bank"),
            "event_start_free_transfers": manual.get("event_start_free_transfers"),
            "source": manual.get("source"),
            "captured_at": manual.get("captured_at"),
        },
        "official_api": official_api,
        "field_provenance": {
            "free_transfers": free_source,
            "bank": bank_source,
            "event_start_free_transfers": "manual" if event_start_ft is not None else "data_gap",
        },
        "stale_official_state_overridden": stale_official_overridden,
        "override_authoritative": manual is not None,
    }


def _transfer_parts(value: Any, fallback_entry_id: int) -> dict[str, Any]:
    if hasattr(value, "element_in"):
        return {
            "entry_id": int(value.entry_id),
            "element_in": int(value.element_in),
            "element_out": int(value.element_out),
            "event": int(value.event),
            "time": value.time,
            "element_in_cost": int(value.element_in_cost),
            "element_out_cost": int(value.element_out_cost),
        }
    if not isinstance(value, dict):
        raise ValueError("transfer history row must be an object")
    keys = ("element_in", "element_out", "event", "element_in_cost", "element_out_cost")
    if any(value.get(key) is None for key in keys):
        raise ValueError("transfer history row is missing an exact required field")
    return {
        "entry_id": int(value.get("entry", fallback_entry_id)),
        "element_in": int(value["element_in"]),
        "element_out": int(value["element_out"]),
        "event": int(value["event"]),
        "time": value.get("time"),
        "element_in_cost": int(value["element_in_cost"]),
        "element_out_cost": int(value["element_out_cost"]),
    }


def _calculated_selling_price(purchase_price: int, market_price: int) -> int:
    return market_price if market_price <= purchase_price else purchase_price + (market_price - purchase_price) // 2


def reconcile_manager_acquisitions(
    conn: sqlite3.Connection,
    entry_id: int,
    event: int,
    transfer_records: Sequence[Any] | None,
    transfers_available: bool,
    *,
    captured_at: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Reconcile exact transfer history into append-only acquisition stints.

    With ``dry_run=True`` the reconciliation is validated and the identical
    result structure returned without writing any acquisition rows.
    """

    transfer_parts = [_transfer_parts(row, int(entry_id)) for row in (transfer_records or [])]
    if any(part["entry_id"] != int(entry_id) for part in transfer_parts):
        return {"status": "data_gap", "message": "transfer history entry ID mismatch", "inserted": 0, "closed": 0}
    if not transfers_available:
        return {"status": "data_gap", "message": "ACQUISITION PRICE DATA GAP - transfer history unavailable", "inserted": 0, "closed": 0}
    transfer_parts.sort(key=lambda row: (row["event"], row.get("time") or ""))
    # Official Free Hit transfers are temporary squad overlays: the FPL entry
    # history records them, but they must never open or close permanent
    # acquisition stints. Wildcard transfers are permanent and pass through.
    free_hit_events = manager_chip_events(conn, int(entry_id), "freehit")
    permanent_parts = [part for part in transfer_parts if part["event"] not in free_hit_events]
    snapshot = latest_complete_squad_rows(conn, int(entry_id), int(event), ascending=True, exclude_events=free_hit_events)
    latest_snapshot_rows = latest_complete_squad_rows(conn, int(entry_id), int(event), ascending=False, exclude_events=free_hit_events)
    if snapshot is None or latest_snapshot_rows is None:
        if free_hit_events:
            return {
                "status": "data_gap",
                "message": "only temporary Free-Hit squad snapshots available; permanent squad base unavailable",
                "inserted": 0,
                "closed": 0,
            }
        return {"status": "data_gap", "message": "verified complete squad snapshot unavailable", "inserted": 0, "closed": 0}

    existing = list_manager_acquisitions(conn, int(entry_id))
    active = {int(row["player_id"]): dict(row) for row in existing if row.get("sold_event") is None}
    planned_close: list[tuple[int, int, str | None, int]] = []
    planned_insert: list[dict[str, Any]] = []
    planned_confirm: list[tuple[int, str | None]] = []
    inserted_initial = 0
    mismatches: list[str] = []
    base_event, base_rows = snapshot

    if not existing:
        for row in base_rows:
            player_id = int(row["player_id"])
            current = latest_snapshot(conn, player_id)
            if current is None or current.get("now_cost") is None or current.get("cost_change_start") is None:
                return {"status": "data_gap", "message": f"purchase price unavailable for player {player_id}", "inserted": 0, "closed": 0}
            purchase_price = int(current["now_cost"]) - int(current["cost_change_start"])
            if purchase_price < 0:
                return {"status": "data_gap", "message": f"invalid reconstructed purchase price for player {player_id}", "inserted": 0, "closed": 0}
            planned_insert.append(
                {
                    "player_id": player_id,
                    "acquired_event": 1 if base_event == 1 else base_event,
                    "purchase_price": purchase_price,
                    "source": "verified_initial_squad",
                    "acquired_at": None,
                }
            )
            active[player_id] = planned_insert[-1]
            inserted_initial += 1

        if not transfer_parts:
            manual = {int(row["player_id"]): row for row in get_manager_selling_prices(conn, int(entry_id), int(event))}
            latest_ids = {int(row["player_id"]) for row in latest_snapshot_rows[1]}
            base_ids = {int(row["player_id"]) for row in base_rows}
            if latest_ids == base_ids and manual:
                for player_id, acquisition in active.items():
                    market = latest_snapshot(conn, player_id)
                    manual_row = manual.get(player_id)
                    if market is None or market.get("now_cost") is None or manual_row is None:
                        continue
                    captured_market = manual_row.get("market_price_at_capture")
                    if captured_market is not None and int(captured_market) != int(market["now_cost"]):
                        continue
                    expected = _calculated_selling_price(int(acquisition["purchase_price"]), int(market["now_cost"]))
                    if int(manual_row["selling_price"]) != expected:
                        mismatches.append(
                            f"player {player_id}: manual selling {manual_row['selling_price']} != calculated {expected}"
                        )
        if mismatches:
            return {"status": "mismatch", "message": "SELLING PRICE MISMATCH", "mismatches": mismatches, "inserted": 0, "closed": 0}

    existing_matches = {
        (int(row["player_id"]), int(row["acquired_event"]), int(row["purchase_price"]), row.get("acquired_at"))
        for row in existing
    }
    for transfer in permanent_parts:
        incoming = int(transfer["element_in"])
        outgoing = int(transfer["element_out"])
        incoming_key = (incoming, int(transfer["event"]), int(transfer["element_in_cost"]), transfer.get("time"))
        if incoming_key in existing_matches:
            continue
        # A user-verified manual/fallback acquisition (recorded because the
        # public endpoint lagged the real move) is confirmed by this exact
        # official history row: upgrade provenance instead of creating a
        # duplicate stint or re-open/closing anything.
        manual_confirmation = next(
            (
                row
                for row in existing
                if int(row["player_id"]) == incoming
                and int(row["acquired_event"]) == int(transfer["event"])
                and int(row["purchase_price"]) == int(transfer["element_in_cost"])
                and row.get("source") == "manual"
            ),
            None,
        )
        if manual_confirmation is not None:
            planned_confirm.append((int(manual_confirmation["id"]), transfer.get("time")))
            manual_confirmation["source"] = "reconciled"
            manual_confirmation["acquired_at"] = transfer.get("time")
            existing_matches.add(incoming_key)
            outgoing_row = active.get(outgoing)
            if outgoing_row is not None and int(outgoing_row.get("id", 0)):
                planned_close.append((int(outgoing_row["id"]), int(transfer["event"]), transfer.get("time"), outgoing))
                active.pop(outgoing, None)
            continue
        outgoing_row = active.get(outgoing)
        if outgoing_row is None:
            return {"status": "data_gap", "message": f"outgoing acquisition unavailable for player {outgoing}", "inserted": 0, "closed": 0}
        if incoming in active:
            return {"status": "data_gap", "message": f"incoming player {incoming} already active", "inserted": 0, "closed": 0}
        planned_close.append((int(outgoing_row.get("id", 0)), int(transfer["event"]), transfer.get("time"), outgoing))
        active.pop(outgoing, None)
        new_row = {
            "player_id": incoming,
            "acquired_event": int(transfer["event"]),
            "purchase_price": int(transfer["element_in_cost"]),
            "source": "official_transfer_history",
            "acquired_at": transfer.get("time"),
        }
        planned_insert.append(new_row)
        active[incoming] = new_row
        existing_matches.add(incoming_key)

    expected_ids = {int(row["player_id"]) for row in latest_snapshot_rows[1]}
    latest_event = int(latest_snapshot_rows[0])
    for transfer in permanent_parts:
        if int(transfer["event"]) > latest_event:
            expected_ids.discard(int(transfer["element_out"]))
            expected_ids.add(int(transfer["element_in"]))
    if set(active) != expected_ids:
        return {"status": "data_gap", "message": "acquisition ledger does not reconcile to the latest complete squad", "inserted": 0, "closed": 0}

    timestamp = captured_at or utc_now()
    if dry_run:
        return {
            "status": "success",
            "message": "acquisition history reconciled (dry-run validation only)",
            "inserted": 0,
            "closed": len(planned_close),
            "confirmed_manual": len(planned_confirm),
            "active": len(active),
            "purchase_prices": {int(player_id): int(row["purchase_price"]) for player_id, row in active.items()},
        }
    for confirmation_id, official_time in planned_confirm:
        conn.execute(
            "UPDATE manager_player_acquisitions SET source='reconciled', acquired_at=?, updated_at=? WHERE id=?",
            (official_time, timestamp, int(confirmation_id)),
        )
    for row in planned_insert:
        if "id" not in row and row.get("source") == "verified_initial_squad":
            row["id"] = insert_manager_acquisition(
                conn,
                int(entry_id),
                int(row["player_id"]),
                int(row["acquired_event"]),
                int(row["purchase_price"]),
                source=str(row["source"]),
                acquired_at=row.get("acquired_at"),
                created_at=timestamp,
            )
    for acquisition_id, sold_event, sold_at, player_id in planned_close:
        if not acquisition_id:
            # The closed stint may itself be planned inside this same
            # reconciliation (same-event churn).  Insert it first so the
            # close can resolve the real acquisition id.
            pending = next(
                (row for row in planned_insert if int(row["player_id"]) == int(player_id) and "id" not in row),
                None,
            )
            if pending is not None:
                pending["id"] = insert_manager_acquisition(
                    conn,
                    int(entry_id),
                    int(pending["player_id"]),
                    int(pending["acquired_event"]),
                    int(pending["purchase_price"]),
                    source=str(pending["source"]),
                    acquired_at=pending.get("acquired_at"),
                    created_at=timestamp,
                )
                acquisition_id = int(pending["id"])
        if not acquisition_id:
            acquisition_id = next(
                int(row["id"])
                for row in planned_insert
                if int(row["player_id"]) == int(player_id) and row.get("id") is not None
            )
        close_manager_acquisition(conn, acquisition_id, sold_event, sold_at, timestamp)
    for row in planned_insert:
        if "id" not in row:
            insert_manager_acquisition(
                conn,
                int(entry_id),
                int(row["player_id"]),
                int(row["acquired_event"]),
                int(row["purchase_price"]),
                source=str(row["source"]),
                acquired_at=row.get("acquired_at"),
                created_at=timestamp,
            )
    return {
        "status": "success",
        "message": "acquisition history reconciled",
        "inserted": inserted_initial + len(planned_insert) - inserted_initial,
        "closed": len(planned_close),
        "confirmed_manual": len(planned_confirm),
        "active": len(active),
        "purchase_prices": {int(player_id): int(row["purchase_price"]) for player_id, row in active.items()},
    }


def upsert_manager_chips(conn: sqlite3.Connection, entry_id: int, records: Iterable[Any]) -> int:
    count = 0
    for record in records:
        conn.execute(
            """INSERT INTO manager_chips(entry_id,name,event,time,updated_at) VALUES (?,?,?,?,?)
            ON CONFLICT(entry_id,name,event) DO UPDATE SET time=excluded.time,updated_at=excluded.updated_at""",
            (entry_id, record.name, record.event, record.time, utc_now()),
        )
        count += 1
    return count


def upsert_squad_picks(
    conn: sqlite3.Connection,
    entry_id: int,
    event: int,
    records: Iterable[PickRecord],
    synced_at: str | None = None,
) -> int:
    timestamp = synced_at or utc_now()
    records = list(records)
    if not records:
        # Empty or malformed picks are “no data”, not an instruction to clear
        # a previously stored roster.
        return 0
    conn.execute("SAVEPOINT squad_roster_replace")
    count = 0
    try:
        # Exact entry/event replacement keeps roster removals visible while
        # leaving every other gameweek untouched.  The savepoint makes a
        # malformed insert roll back to the prior roster.
        conn.execute("DELETE FROM squad_picks WHERE entry_id=? AND event=?", (entry_id, event))
        for record in records:
            conn.execute(
                """INSERT INTO squad_picks(
                  entry_id,event,player_id,position,multiplier,is_captain,is_vice_captain,is_starting,synced_at,raw_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    entry_id, event, record.player_id, record.position, record.multiplier, record.is_captain,
                    record.is_vice_captain, 1 if record.position <= 11 else 0, timestamp, _raw(record.raw_json),
                ),
            )
            count += 1
        conn.execute("RELEASE SAVEPOINT squad_roster_replace")
    except Exception:
        conn.execute("ROLLBACK TO SAVEPOINT squad_roster_replace")
        conn.execute("RELEASE SAVEPOINT squad_roster_replace")
        raise
    return count


def insert_scouting_import(conn: sqlite3.Connection, values: dict[str, Any]) -> int:
    cursor = conn.execute(
        """INSERT INTO scouting_imports(
          source_file,file_sha256,schema_version,agent,generated_at,gameweek,players_total,players_resolved,
          notes_inserted,unresolved_json,imported_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        (
            values["source_file"], values["file_sha256"], values.get("schema_version"), values.get("agent"),
            values.get("generated_at"), values.get("gameweek"), values.get("players_total"),
            values.get("players_resolved"), values.get("notes_inserted"), _raw(values.get("unresolved", [])),
            values.get("imported_at", utc_now()),
        ),
    )
    return int(cursor.lastrowid)


def insert_scouting_note(conn: sqlite3.Connection, values: dict[str, Any]) -> int:
    cursor = conn.execute(
        """INSERT INTO scouting_notes(
          import_id,player_id,key,category,value_text,value_num,value_unit,confidence,evidence_json,observation,
          gameweek_context,observed_at,expires_at,agent,created_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            values.get("import_id"), values["player_id"], values["key"], values.get("category"),
            values.get("value_text"), values.get("value_num"), values.get("value_unit"), values.get("confidence"),
            _raw(values.get("evidence", [])), values.get("observation"), values.get("gameweek_context"),
            values["observed_at"], values.get("expires_at"), values.get("agent"), values.get("created_at", utc_now()),
        ),
    )
    return int(cursor.lastrowid)


def has_scouting_hash(conn: sqlite3.Connection, file_sha256: str) -> bool:
    return conn.execute("SELECT 1 FROM scouting_imports WHERE file_sha256=? LIMIT 1", (file_sha256,)).fetchone() is not None


def add_watchlist(
    conn: sqlite3.Connection,
    player_id: int,
    status: str,
    reason: str | None = None,
    target_event_from: int | None = None,
    target_event_to: int | None = None,
    notes: str | None = None,
) -> int:
    now = utc_now()
    cursor = conn.execute(
        """INSERT INTO watchlist(player_id,status,reason,target_event_from,target_event_to,notes,created_at,updated_at)
        VALUES (?,?,?,?,?,?,?,?)""",
        (player_id, status, reason, target_event_from, target_event_to, notes, now, now),
    )
    return int(cursor.lastrowid)


def update_watchlist(conn: sqlite3.Connection, watchlist_id: int, **values: Any) -> None:
    allowed = {"status", "reason", "target_event_from", "target_event_to", "notes", "is_active"}
    changes = [(key, value) for key, value in values.items() if key in allowed]
    if not changes:
        return
    assignments = ",".join(f"{key}=?" for key, _ in changes)
    conn.execute(f"UPDATE watchlist SET {assignments}, updated_at=? WHERE id=?", [value for _, value in changes] + [utc_now(), watchlist_id])


def add_decision(conn: sqlite3.Connection, values: dict[str, Any]) -> int:
    cursor = conn.execute(
        """INSERT INTO decisions(
          event,action,player_in_id,player_out_id,captain_id,vice_captain_id,chip,reasoning,confidence,
          assumptions_json,invalidators_json,expected_cost,created_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            values["event"], values["action"], values.get("player_in_id"), values.get("player_out_id"),
            values.get("captain_id"), values.get("vice_captain_id"), values.get("chip"), values.get("reasoning"),
            values.get("confidence"), _raw(values.get("assumptions", [])), _raw(values.get("invalidators", [])),
            values.get("expected_cost"), values.get("created_at", utc_now()),
        ),
    )
    return int(cursor.lastrowid)


def review_decision(conn: sqlite3.Connection, decision_id: int, review_notes: str) -> None:
    conn.execute("UPDATE decisions SET review_notes=?, reviewed_at=? WHERE id=?", (review_notes, utc_now(), decision_id))


def add_strategy(conn: sqlite3.Connection, values: dict[str, Any]) -> int:
    cursor = conn.execute(
        """INSERT INTO strategy(event,wildcard_horizon,bench_boost_plan,free_hit_plan,triple_captain_plan,risk_posture,notes,created_at)
        VALUES (?,?,?,?,?,?,?,?)""",
        (
            values.get("event"), values.get("wildcard_horizon"), values.get("bench_boost_plan"),
            values.get("free_hit_plan"), values.get("triple_captain_plan"), values.get("risk_posture"),
            values.get("notes"), values.get("created_at", utc_now()),
        ),
    )
    return int(cursor.lastrowid)


def get_player(conn: sqlite3.Connection, player_id: int) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM players WHERE id=?", (player_id,)).fetchone()
    return dict(row) if row else None


def find_players_by_norm_name(conn: sqlite3.Connection, norm_name: str) -> list[dict[str, Any]]:
    return _rows(conn.execute("SELECT * FROM players WHERE is_active=1 AND norm_name=?", (norm_name,)))


def find_players_by_web_name(conn: sqlite3.Connection, norm_name: str) -> list[dict[str, Any]]:
    rows = _rows(conn.execute("SELECT * FROM players WHERE is_active=1"))
    return [row for row in rows if normalise_name(row.get("web_name")) == norm_name]


def player_candidates(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    return _rows(
        conn.execute(
            """SELECT p.*, t.name AS team_name, t.short_name AS team_short_name,
                      pos.singular_name_short AS position_short_name
               FROM players p LEFT JOIN teams t ON t.id=p.team_id
               LEFT JOIN positions pos ON pos.id=p.element_type
              WHERE p.is_active=1 ORDER BY p.id"""
        )
    )


def list_active_players(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    return _rows(conn.execute("SELECT * FROM players WHERE is_active=1 ORDER BY web_name"))


def latest_snapshot(conn: sqlite3.Connection, player_id: int) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT * FROM player_snapshots WHERE player_id=? ORDER BY captured_at DESC, id DESC LIMIT 1", (player_id,)
    ).fetchone()
    return dict(row) if row else None


def latest_snapshots(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    return _rows(
        conn.execute(
            """SELECT s.* FROM player_snapshots s
               JOIN (SELECT player_id, MAX(captured_at) AS captured_at FROM player_snapshots GROUP BY player_id) latest
                 ON latest.player_id=s.player_id AND latest.captured_at=s.captured_at
               ORDER BY s.player_id"""
        )
    )


def snapshot_history(conn: sqlite3.Connection, player_id: int, field: str, since: str | None = None) -> list[dict[str, Any]]:
    allowed = {"now_cost", "selected_by_percent", "transfers_in_event", "transfers_out_event"}
    if field not in allowed:
        raise ValueError(f"Unsupported trend field: {field}")
    if since is None:
        rows = conn.execute(
            f"SELECT captured_at, {field} AS value FROM player_snapshots WHERE player_id=? ORDER BY captured_at, id",
            (player_id,),
        )
    else:
        rows = conn.execute(
            f"SELECT captured_at, {field} AS value FROM player_snapshots WHERE player_id=? AND captured_at>=? ORDER BY captured_at, id",
            (player_id, since),
        )
    return _rows(rows)


def fixture_rows(conn: sqlite3.Connection, from_event: int, horizon: int) -> list[dict[str, Any]]:
    return _rows(
        conn.execute(
            """SELECT * FROM fixtures WHERE event IS NOT NULL AND event>=? AND event<?
               ORDER BY event, kickoff_time, id""",
            (from_event, from_event + horizon),
        )
    )


def _fixture_horizon_selection(
    conn: sqlite3.Connection,
    from_event: int,
    horizon: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return fixtures placed in an event horizon and pending fixtures left unplaced."""

    if horizon <= 0:
        return [], []

    rows = _rows(conn.execute("SELECT * FROM fixtures ORDER BY event, kickoff_time, id"))
    pending = [row for row in rows if row.get("finished") != 1 and row.get("started") != 1]
    scheduled_pending = [
        row for row in pending
        if row.get("event") is not None and int(row["event"]) >= int(from_event)
    ]
    older_pending = [
        row for row in pending
        if row.get("event") is not None and int(row["event"]) < int(from_event)
    ]
    unplaced = [row for row in pending if row.get("event") is None]
    if not scheduled_pending:
        return [], sorted(older_pending + unplaced, key=_fixture_sort_key)

    first_event = min(int(row["event"]) for row in scheduled_pending)
    last_event_exclusive = first_event + int(horizon)
    placed = [
        {**row, "horizon_event": int(row["event"])}
        for row in scheduled_pending
        if int(row["event"]) < last_event_exclusive
    ]
    if not older_pending:
        return sorted(placed, key=_fixture_sort_key), sorted(unplaced, key=_fixture_sort_key)

    deadline_rows = _rows(
        conn.execute(
            """SELECT id, deadline_time FROM events
               WHERE id>=? AND id<=? ORDER BY id""",
            (first_event, last_event_exclusive),
        )
    )
    deadlines = {int(row["id"]): parse_utc(row.get("deadline_time")) for row in deadline_rows}
    max_event = conn.execute("SELECT MAX(id) FROM events").fetchone()[0]
    max_event = int(max_event) if max_event is not None else None
    horizon_start = deadlines.get(first_event)
    horizon_end = deadlines.get(last_event_exclusive)
    cutoff_is_expected = max_event is not None and last_event_exclusive <= max_event
    last_horizon_event = min(last_event_exclusive - 1, max_event) if max_event is not None else None

    for row in older_pending:
        kickoff = parse_utc(row.get("kickoff_time"))
        if kickoff is None or horizon_start is None:
            unplaced.append(row)
            continue
        if kickoff < horizon_start:
            # Official fixture state says pending, but its recorded schedule is
            # before this horizon. Keep it visible as a data-quality gap rather
            # than treating its old event as an immediate fixture.
            unplaced.append(row)
            continue
        if cutoff_is_expected and horizon_end is None:
            unplaced.append(row)
            continue
        if horizon_end is not None and kickoff >= horizon_end:
            # The known reschedule is beyond this horizon. It will be evaluated
            # again when a later horizon reaches its kickoff window.
            continue

        placement_event = _horizon_event_for_kickoff(
            kickoff,
            deadlines,
            first_event,
            last_event_exclusive,
            last_horizon_event,
            max_event,
        )
        if placement_event is None:
            unplaced.append(row)
            continue
        placed.append({**row, "horizon_event": placement_event})

    return sorted(placed, key=_fixture_sort_key), sorted(unplaced, key=_fixture_sort_key)


def _horizon_event_for_kickoff(
    kickoff: Any,
    deadlines: dict[int, Any],
    first_event: int,
    last_event_exclusive: int,
    last_horizon_event: int | None,
    max_event: int | None,
) -> int | None:
    """Map a known pending kickoff without inferring state or inventing an event."""

    if last_horizon_event is None:
        return None
    for event_id in range(first_event, min(last_event_exclusive, last_horizon_event + 1)):
        event_start = deadlines.get(event_id)
        if event_start is None:
            return None
        next_start = deadlines.get(event_id + 1)
        if next_start is None:
            if event_id == last_horizon_event and (
                event_id == max_event or event_id + 1 >= last_event_exclusive
            ):
                return event_id
            return None
        if event_start <= kickoff < next_start:
            return event_id
    return None


def _fixture_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    """Order placed fixtures by derived event window, then scheduled kickoff."""

    kickoff = parse_utc(row.get("kickoff_time"))
    event = row.get("horizon_event", row.get("event"))
    return (
        int(event) if event is not None else 10**9,
        kickoff is None,
        kickoff.isoformat() if kickoff is not None else "",
        int(row["id"]),
    )


def future_fixture_rows(conn: sqlite3.Connection, from_event: int, horizon: int) -> list[dict[str, Any]]:
    """Return state-pending fixtures that can be placed in the next event horizon.

    The normal horizon starts at the earliest pending FPL event at or after
    ``from_event``. A pending fixture retained against an older event is not
    automatically treated as immediate: its known kickoff must fall inside
    the horizon's official event-deadline windows. Such a row retains its
    source ``event`` and receives a derived ``horizon_event`` for ordering and
    DGW accounting. The final official event is an open-ended window when no
    later official event exists; no new event is invented. A missing or
    contradictory schedule is excluded from numeric future metrics and exposed
    by ``unplaced_pending_fixture_rows``.

    Only ``started=1`` or ``finished=1`` proves that a fixture is no longer
    future-facing. Kickoff time is used solely to place an already-pending
    old-event fixture; it never infers completion.
    """

    return _fixture_horizon_selection(conn, from_event, horizon)[0]


def unplaced_pending_fixture_rows(conn: sqlite3.Connection, from_event: int, horizon: int) -> list[dict[str, Any]]:
    """Return pending fixtures that cannot be safely placed in an event horizon."""

    return _fixture_horizon_selection(conn, from_event, horizon)[1]


def team_row(conn: sqlite3.Connection, team_id: int) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM teams WHERE id=?", (team_id,)).fetchone()
    return dict(row) if row else None


def event_rows(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    return _rows(conn.execute("SELECT * FROM events ORDER BY id"))


def current_or_next_event(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT id FROM events WHERE is_current=1 ORDER BY id LIMIT 1").fetchone()
    if row:
        return int(row[0])
    row = conn.execute("SELECT id FROM events WHERE is_next=1 ORDER BY id LIMIT 1").fetchone()
    return int(row[0]) if row else 1


def latest_manager_state(conn: sqlite3.Connection, entry_id: int | None = None) -> dict[str, Any] | None:
    if entry_id is None:
        row = conn.execute("SELECT * FROM manager_state ORDER BY captured_at DESC, id DESC LIMIT 1").fetchone()
    else:
        row = conn.execute(
            "SELECT * FROM manager_state WHERE entry_id=? ORDER BY captured_at DESC, id DESC LIMIT 1", (entry_id,)
        ).fetchone()
    return dict(row) if row else None


def squad_rows(conn: sqlite3.Connection, entry_id: int, event: int) -> list[dict[str, Any]]:
    return _rows(
        conn.execute(
            """SELECT q.*, p.web_name, p.full_name, p.team_id, p.element_type, t.name AS team_name,
                      pos.singular_name_short
               FROM squad_picks q JOIN players p ON p.id=q.player_id
               LEFT JOIN teams t ON t.id=p.team_id LEFT JOIN positions pos ON pos.id=p.element_type
              WHERE q.entry_id=? AND q.event=? ORDER BY q.position""",
            (entry_id, event),
        )
    )


def active_watchlist_rows(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    return _rows(
        conn.execute(
            """SELECT w.*, p.web_name, p.full_name, p.team_id, t.name AS team_name
               FROM watchlist w JOIN players p ON p.id=w.player_id
               LEFT JOIN teams t ON t.id=p.team_id WHERE w.is_active=1 ORDER BY w.id"""
        )
    )


def scouting_current_rows_as_of(
    conn: sqlite3.Connection,
    cutoff: str,
    player_ids: Sequence[int],
) -> list[dict[str, Any]]:
    """Scouting notes ranked **within the rows eligible at** ``cutoff``.

    The v_scouting_current_ranked view ranks over all time and then keeps rank 1,
    so a note observed after the cutoff was selected first and could only be
    discarded afterwards ("rank latest, then discard").  Correct as-of semantics
    rank ONLY rows with ``observed_at <= cutoff`` and then take the latest per
    (player, key).  Expiry is applied afterwards, so an expired-but-latest note
    still reports as expired rather than silently reverting to an older note.
    """

    if not player_ids:
        return []
    placeholders = ",".join("?" for _ in player_ids)
    return _rows(
        conn.execute(
            f"""SELECT s.*, p.web_name, p.full_name, p.team_id, t.name AS team_name
                FROM (
                  SELECT notes.*,
                         ROW_NUMBER() OVER (
                           PARTITION BY notes.player_id, notes.key
                           ORDER BY notes.observed_at DESC, notes.id DESC
                         ) AS current_rank
                  FROM scouting_notes notes
                  WHERE notes.observed_at <= ?
                ) s
                JOIN players p ON p.id=s.player_id
                LEFT JOIN teams t ON t.id=p.team_id
                WHERE s.current_rank = 1 AND s.player_id IN ({placeholders})
                ORDER BY p.web_name, s.key""",
            (str(cutoff), *[int(pid) for pid in player_ids]),
        )
    )


def scouting_current_rows(conn: sqlite3.Connection, player_ids: Sequence[int] | None = None) -> list[dict[str, Any]]:
    current_view = "v_scouting_current_ranked"
    if player_ids is None:
        return _rows(
            conn.execute(
                f"""SELECT s.*, p.web_name, p.full_name, p.team_id, t.name AS team_name
                   FROM {current_view} s JOIN players p ON p.id=s.player_id
                   LEFT JOIN teams t ON t.id=p.team_id ORDER BY p.web_name, s.key"""
            )
        )
    if not player_ids:
        return []
    placeholders = ",".join("?" for _ in player_ids)
    return _rows(
        conn.execute(
            f"""SELECT s.*, p.web_name, p.full_name, p.team_id, t.name AS team_name
                FROM {current_view} s JOIN players p ON p.id=s.player_id
               LEFT JOIN teams t ON t.id=p.team_id
               WHERE s.player_id IN ({placeholders}) ORDER BY p.web_name, s.key""",
            tuple(player_ids),
        )
    )


def decision_rows(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    return _rows(conn.execute("SELECT * FROM decisions ORDER BY created_at, id"))


def strategy_row(conn: sqlite3.Connection) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM strategy ORDER BY created_at DESC, id DESC LIMIT 1").fetchone()
    return dict(row) if row else None


_COMPLETED_PERFORMANCE_FROM = "FROM player_gameweeks p JOIN fixtures f ON f.id=p.fixture_id AND f.finished=1"
#: A finished fixture is NECESSARY but not sufficient: the same refresh sequence
#: that skips the element-summary endpoint leaves pre-round schedule rows behind
#: on a completed fixture, and those rows carry no observation at all.  The
#: canonical placeholder signature is therefore part of this boundary, so every
#: consumer of completed rows -- outcomes, completeness audits and the market
#: report -- agrees with the causal-history readers on what an observation is.
_COMPLETED_PERFORMANCE_WHERE = (
    "p.fixture_id IS NOT NULL AND p.fixture_id != -1 AND " + gameweek_has_performance_sql("p")
)


def completed_player_fixture_rows(
    conn: sqlite3.Connection,
    *,
    player_id: int | None = None,
    event: int | None = None,
    before_event: int | None = None,
    events: Sequence[int] | None = None,
    fixture_ids: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    """Return only player-fixture rows that carry a real realised observation.

    This is the single reusable completed-performance boundary. A future
    element-summary schedule can contain minutes=0, while a completed player can
    legitimately also have zero minutes; neither value is a completion marker.
    Sentinel/null fixture references are excluded, and a row must additionally
    satisfy the canonical scheduled-placeholder signature -- a finished fixture
    whose rows were never refreshed holds placeholders, not outcomes, and
    admitting them would record "we do not know yet" as "he scored nothing".
    """

    if event is not None and events is not None:
        raise ValueError("event and events cannot both be supplied")
    if before_event is not None and (event is not None or events is not None):
        raise ValueError("before_event cannot be combined with event/events")
    clauses = [_COMPLETED_PERFORMANCE_WHERE]
    params: list[Any] = []
    if player_id is not None:
        clauses.append("p.player_id=?")
        params.append(int(player_id))
    if event is not None:
        clauses.append("p.event=?")
        params.append(int(event))
    if before_event is not None:
        clauses.append("p.event<?")
        params.append(int(before_event))
    if events is not None:
        event_values = [int(value) for value in events]
        if not event_values:
            return []
        clauses.append("p.event IN (" + ",".join("?" for _ in event_values) + ")")
        params.extend(event_values)
    if fixture_ids is not None:
        fixture_values = [int(value) for value in fixture_ids]
        if not fixture_values:
            return []
        clauses.append("p.fixture_id IN (" + ",".join("?" for _ in fixture_values) + ")")
        params.extend(fixture_values)
    where = " AND ".join(clauses)
    return _rows(
        conn.execute(
            f"""SELECT p.*,
                       f.event AS fixture_event,
                       f.team_h AS fixture_team_h,
                       f.team_a AS fixture_team_a,
                       f.finished AS fixture_finished,
                       f.finished_provisional AS fixture_finished_provisional,
                       f.started AS fixture_started
                {_COMPLETED_PERFORMANCE_FROM}
                WHERE {where}
                ORDER BY p.event,p.fixture_id,p.player_id""",
            tuple(params),
        )
    )


def gameweek_rows(conn: sqlite3.Connection, player_id: int, last_n_events: int | None = None) -> list[dict[str, Any]]:
    # Keep the public minute-reliability query on the shared authoritative
    # fixture-completion boundary.
    if last_n_events is None:
        return completed_player_fixture_rows(conn, player_id=player_id)
    event_rows = conn.execute(
        f"""SELECT DISTINCT p.event
            {_COMPLETED_PERFORMANCE_FROM}
            WHERE {_COMPLETED_PERFORMANCE_WHERE} AND p.player_id=?
            ORDER BY p.event DESC LIMIT ?""",
        (int(player_id), int(last_n_events)),
    ).fetchall()
    if not event_rows:
        return []
    events = [int(row[0]) for row in event_rows]
    return completed_player_fixture_rows(conn, player_id=player_id, events=events)


def count_rows(conn: sqlite3.Connection, table: str) -> int:
    allowed = {
        "teams",
        "players",
        "fixtures",
        "player_snapshots",
        "player_gameweeks",
        "manager_state",
        "manager_manual_state",
        "manager_selling_prices",
        "manager_player_acquisitions",
        "squad_picks",
    }
    if table not in allowed:
        raise ValueError("Unsupported count table")
    return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
