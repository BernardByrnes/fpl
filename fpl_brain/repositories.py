"""Typed database reads and writes for each logical table."""

from __future__ import annotations

import json
import sqlite3
from typing import Any, Iterable, Sequence

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
_COMPLETED_PERFORMANCE_WHERE = "p.fixture_id IS NOT NULL AND p.fixture_id != -1"


def completed_player_fixture_rows(
    conn: sqlite3.Connection,
    *,
    player_id: int | None = None,
    event: int | None = None,
    before_event: int | None = None,
    events: Sequence[int] | None = None,
    fixture_ids: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    """Return only player-fixture rows proven complete by their fixture.

    This is the single reusable completed-performance boundary. A future
    element-summary schedule can contain minutes=0, while a completed player
    can legitimately also have zero minutes; neither value is a completion
    marker. Sentinel/null fixture references are excluded and fixture
    finished=1 is authoritative.
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
    allowed = {"teams", "players", "fixtures", "player_snapshots", "player_gameweeks", "manager_state", "squad_picks"}
    if table not in allowed:
        raise ValueError("Unsupported count table")
    return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
