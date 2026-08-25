"""Fetch and manager-sync orchestration."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any

from .api import FplApiError, FplClient
from .config import config_path
from .database import connect_database
from .parsers import (
    parse_bootstrap,
    parse_element_summary,
    parse_entry,
    parse_entry_history,
    parse_entry_picks,
    parse_event_live,
    parse_fixtures,
    validate_bootstrap_payload,
)
from . import repositories as repo
from .utils import ensure_directory, utc_now

LOGGER = logging.getLogger(__name__)


def _event_context(records: Any) -> int:
    events = records.events
    current = [event.id for event in events if event.is_current == 1]
    if current:
        return min(current)
    next_events = [event.id for event in events if event.is_next == 1]
    return min(next_events) if next_events else 1


def _selection_from_db(conn: sqlite3.Connection, entry_id: int | None, event: int) -> set[int]:
    selected: set[int] = set()
    if entry_id is not None:
        selected.update(row["player_id"] for row in repo.squad_rows(conn, entry_id, event))
    selected.update(row["player_id"] for row in repo.active_watchlist_rows(conn))
    return selected


def _result_counts(conn: sqlite3.Connection | None) -> dict[str, int]:
    if conn is None:
        return {}
    return {
        "teams": repo.count_rows(conn, "teams"),
        "players": repo.count_rows(conn, "players"),
        "fixtures": repo.count_rows(conn, "fixtures"),
        "snapshots": repo.count_rows(conn, "player_snapshots"),
        "gameweeks": repo.count_rows(conn, "player_gameweeks"),
    }


def run_fetch(
    config: dict[str, Any],
    summaries: str = "none",
    live_gw: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run bootstrap/fixture ingest and optional detail fan-out."""

    if summaries not in {"none", "squad", "all"}:
        raise ValueError("summaries must be none, squad, or all")
    captured_at = utc_now()
    raw_dir = config_path(config, "raw_dir")
    database_path = config_path(config, "database")
    conn: sqlite3.Connection | None = None
    run_id: int | None = None
    endpoints_ok: list[str] = []
    endpoint_failures: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    if not dry_run:
        ensure_directory(raw_dir)
        conn = connect_database(database_path)
        with conn:
            run_id = repo.create_fetch_run(conn, "fetch_fpl", str(raw_dir))
    client = FplClient(config, raw_dir=None if dry_run else raw_dir, run_id=run_id, dry_run=dry_run)
    try:
        try:
            bootstrap_payload = client.get_bootstrap_static()
            first_records = parse_bootstrap(bootstrap_payload, captured_at)
            validate_bootstrap_payload(bootstrap_payload, first_records)
            current_event = _event_context(first_records)
            records = parse_bootstrap(bootstrap_payload, captured_at, current_event)
        except Exception as exc:
            if conn is not None and run_id is not None:
                with conn:
                    repo.finish_fetch_run(conn, run_id, "failed", endpoints_ok=endpoints_ok, endpoints_failed=[{"endpoint": "bootstrap-static", "error": str(exc)}], error_message=str(exc))
            raise
        endpoints_ok.append("bootstrap-static")
        if not dry_run and conn is not None and run_id is not None:
            try:
                with conn:
                    repo.upsert_events(conn, records.events, captured_at)
                    repo.upsert_chips(conn, records.chips, captured_at)
                    repo.upsert_teams(conn, records.teams, captured_at)
                    repo.upsert_positions(conn, records.positions, captured_at)
                    seen = repo.upsert_players(conn, records.players, captured_at)
                    repo.mark_absent_players(conn, seen)
                    snapshot_count = repo.insert_snapshots(conn, records.snapshots, run_id)
                LOGGER.info("bootstrap stored teams=%s players=%s snapshots=%s", len(records.teams), len(seen), snapshot_count)
            except Exception as exc:
                with conn:
                    repo.finish_fetch_run(conn, run_id, "failed", current_event, endpoints_ok, [{"endpoint": "bootstrap-static", "error": str(exc)}], str(exc))
                raise
        else:
            snapshot_count = len(records.snapshots)

        try:
            fixture_payload = client.get_fixtures()
            fixtures = parse_fixtures(fixture_payload)
            endpoints_ok.append("fixtures")
            if not dry_run and conn is not None:
                with conn:
                    fixture_count = repo.upsert_fixtures(conn, fixtures, captured_at)
                LOGGER.info("fixtures stored=%s", fixture_count)
            else:
                fixture_count = len(fixtures)
        except Exception as exc:
            endpoint_failures.append({"endpoint": "fixtures", "error": str(exc)})
            LOGGER.error("fixtures stage failed: %s", exc)
            fixture_count = 0

        if summaries != "none":
            if summaries == "all":
                selected_ids = {player.id for player in records.players}
            elif conn is not None:
                selected_ids = _selection_from_db(conn, config.get("fpl_entry_id"), current_event)
            else:
                selected_ids = set()
            summary_count = 0
            for player_id in sorted(selected_ids):
                try:
                    payload = client.get_element_summary(player_id)
                    detail_rows = parse_element_summary(payload, player_id)
                    summary_count += len(detail_rows)
                    endpoints_ok.append(f"element-summary/{player_id}")
                    if not dry_run and conn is not None:
                        with conn:
                            repo.upsert_player_gameweeks(conn, detail_rows, captured_at)
                except FplApiError as exc:
                    endpoint_failures.append({"endpoint": f"element-summary/{player_id}", "error": str(exc)})
                    LOGGER.warning("element summary failed for %s: %s", player_id, exc)
            LOGGER.info("element summaries stored rows=%s players=%s", summary_count, len(selected_ids))
        else:
            summary_count = 0

        if live_gw is not None:
            try:
                live_payload = client.get_event_live(live_gw)
                live_rows = parse_event_live(live_payload, live_gw)
                if not dry_run and conn is not None:
                    live_rows = repo.enrich_gameweek_context(conn, live_rows)
                endpoints_ok.append(f"event/{live_gw}/live")
                if not dry_run and conn is not None:
                    with conn:
                        repo.upsert_player_gameweeks(conn, live_rows, captured_at)
                LOGGER.info("event live stored rows=%s", len(live_rows))
            except FplApiError as exc:
                endpoint_failures.append({"endpoint": f"event/{live_gw}/live", "error": str(exc)})
                LOGGER.warning("event live failed: %s", exc)

        status = "partial" if endpoint_failures else "success"
        if not dry_run and conn is not None and run_id is not None:
            with conn:
                repo.finish_fetch_run(conn, run_id, status, current_event, endpoints_ok, endpoint_failures)
            counts = _result_counts(conn)
        return {
            "status": status,
            "run_id": run_id,
            "current_event": current_event,
            "teams": len(records.teams),
            "players": len(records.players),
            "fixtures": fixture_count,
            "snapshots": snapshot_count,
            "summaries": summary_count,
            "counts": counts,
            "endpoints_failed": endpoint_failures,
        }
    finally:
        client.close()
        if conn is not None:
            conn.close()


def run_manager_sync(
    config: dict[str, Any],
    event: int | None = None,
    all_events: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Sync public manager summary/history and any available squad picks."""

    entry_id = config.get("fpl_entry_id")
    if entry_id is None:
        return {
            "status": "not_configured",
            "message": "Team ID not configured. Find it in the URL on the FPL Points tab and set fpl_entry_id.",
        }
    raw_dir = config_path(config, "raw_dir")
    database_path = config_path(config, "database")
    captured_at = utc_now()
    conn: sqlite3.Connection | None = None
    run_id: int | None = None
    failures: list[dict[str, Any]] = []
    endpoints_ok: list[str] = []
    if not dry_run:
        ensure_directory(raw_dir)
        conn = connect_database(database_path)
        with conn:
            run_id = repo.create_fetch_run(conn, "sync_manager", str(raw_dir))
    client = FplClient(config, raw_dir=None if dry_run else raw_dir, run_id=run_id, dry_run=dry_run)
    try:
        try:
            entry_payload = client.get_entry(int(entry_id))
            history_payload = client.get_entry_history(int(entry_id))
            entry = parse_entry(entry_payload)
            history = parse_entry_history(history_payload)
            if entry is None:
                raise ValueError("Entry response had no valid ID")
            endpoints_ok.extend([f"entry/{entry_id}", f"entry/{entry_id}/history"])
        except Exception as exc:
            if conn is not None and run_id is not None:
                with conn:
                    repo.finish_fetch_run(conn, run_id, "failed", endpoints_ok=endpoints_ok, endpoints_failed=[{"endpoint": "entry", "error": str(exc)}], error_message=str(exc))
            raise

        target_event = event or entry.current_event or (repo.current_or_next_event(conn) if conn is not None else 1)
        if all_events:
            target_events = list(range(1, int(target_event) + 1))
        else:
            target_events = [int(target_event)]
        parsed_picks: list[tuple[int, Any]] = []
        picks_written = 0
        for target in target_events:
            try:
                picks_payload = client.get_entry_picks(int(entry_id), target)
                if picks_payload is None:
                    LOGGER.info("no data yet for entry/%s/event/%s/picks (deadline has not passed)", entry_id, target)
                    continue
                picks = parse_entry_picks(picks_payload)
                endpoints_ok.append(f"entry/{entry_id}/event/{target}/picks")
                parsed_picks.append((target, picks))
            except (FplApiError, ValueError, TypeError) as exc:
                failures.append({"endpoint": f"entry/{entry_id}/event/{target}/picks", "error": str(exc)})
                LOGGER.warning("picks failed for event %s: %s", target, exc)
        if not dry_run and conn is not None:
            with conn:
                fallback = history.current[-1] if history.current else next(
                    (picks.entry_history for _, picks in parsed_picks if picks.entry_history is not None), None
                )
                active_chip = next((picks.active_chip for _, picks in parsed_picks if picks.active_chip), None)
                repo.insert_manager_state(
                    conn,
                    entry,
                    fallback,
                    active_chip,
                    config.get("manual_overrides", {}).get("free_transfers"),
                    run_id,
                    int(target_event),
                    {"entry": entry_payload, "history": history_payload},
                    captured_at,
                )
                repo.upsert_manager_chips(conn, int(entry_id), history.chips)
                for target, picks in parsed_picks:
                    picks_written += repo.upsert_squad_picks(conn, int(entry_id), target, picks.picks, captured_at)
        status = "partial" if failures else "success"
        if not dry_run and conn is not None and run_id is not None:
            with conn:
                repo.finish_fetch_run(conn, run_id, status, int(target_event), endpoints_ok, failures)
        return {
            "status": status,
            "entry_id": int(entry_id),
            "event": int(target_event),
            "picks_written": picks_written,
            "endpoints_failed": failures,
            "run_id": run_id,
        }
    finally:
        client.close()
        if conn is not None:
            conn.close()
