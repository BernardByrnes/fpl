"""Database-only Markdown and JSON evidence-pack renderers."""

from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any

from . import repositories as repo
from .metrics import attacking_and_defensive_outlook, fixture_outlook, minutes_reliability, points_per_million, trend
from .utils import json_text, parse_utc, utc_now

SECTION_ORDER = [
    "HOW TO READ THIS REPORT",
    "MANAGER  [FACT unless noted]",
    "STRATEGY  [MANUAL]",
    "CURRENT SQUAD  [FACT]",
    "PLAYER DATA  [FACT + DERIVED]",
    "CURRENT FIXTURES  [FACT]",
    "SCOUTING CONTEXT  [SCOUTING]",
    "RISKS",
    "WATCHLIST  [MANUAL + FACT]",
    "PRICE / OWNERSHIP MOVEMENTS  [DERIVED from FACT snapshots]",
    "PREVIOUS DECISIONS  [MANUAL]",
    "DATA GAPS AND CAVEATS",
]


def _money(value: Any) -> str:
    return "unavailable" if value is None else f"£{float(value) / 10:.1f}m"


def _value(value: Any, missing: str = "—") -> str:
    return missing if value is None else str(value)


def _percent(value: Any) -> str:
    return "—" if value is None else f"{float(value):.1f}%"


def _player_name(row: dict[str, Any]) -> str:
    return row.get("full_name") or row.get("web_name") or f"Player {row.get('id')}"


def _event_context(conn: sqlite3.Connection, requested: int | None) -> tuple[int, dict[str, Any] | None]:
    event_id = requested or repo.current_or_next_event(conn)
    row = next((row for row in repo.event_rows(conn) if row["id"] == event_id), None)
    return event_id, row


def _freshness(conn: sqlite3.Connection) -> str:
    values = [
        ("bootstrap", conn.execute("SELECT MAX(updated_at) FROM events").fetchone()[0]),
        ("fixtures", conn.execute("SELECT MAX(updated_at) FROM fixtures").fetchone()[0]),
    ]
    manager = conn.execute("SELECT MAX(captured_at) FROM manager_state").fetchone()[0]
    scouting = conn.execute("SELECT MAX(observed_at) FROM scouting_notes").fetchone()[0]
    values.append(("manager", manager or "never synced"))
    values.append(("scouting", scouting or "never"))
    return " | ".join(f"{name} {value or 'never'}" for name, value in values)


def _team_label(conn: sqlite3.Connection, team_id: Any) -> str:
    if team_id is None:
        return "TEAM"
    team = repo.team_row(conn, int(team_id))
    if not team:
        return f"Team {team_id}"
    return str(team.get("short_name") or team.get("name") or f"Team {team_id}")


def _fetch_health_lines(conn: sqlite3.Connection) -> list[str]:
    run = repo.latest_fetch_run(conn, "fetch_fpl")
    if not run:
        return ["[FACT] Latest fetch_fpl run: never recorded."]
    try:
        ok = set(json.loads(run.get("endpoints_ok") or "[]"))
        failed = json.loads(run.get("endpoints_failed") or "[]")
    except (TypeError, ValueError, json.JSONDecodeError):
        ok, failed = set(), []
    failed_names = {
        str(item.get("endpoint"))
        for item in failed
        if isinstance(item, dict) and item.get("endpoint")
    }
    bootstrap_state = "success" if "bootstrap-static" in ok else "failed" if "bootstrap-static" in failed_names else "not-run"
    fixtures_state = "success" if "fixtures" in ok else "failed" if "fixtures" in failed_names else "not-run"
    lines = [
        f"[FACT] Latest fetch_fpl run {run['id']}: status {run.get('status')}; bootstrap-static {bootstrap_state}; fixtures {fixtures_state}.",
    ]
    if run.get("status") in {"partial", "failed", "running"}:
        lines.append(f"[FACT WARNING] Latest fetch_fpl run is {run.get('status')}; some FACT data may be stale.")
    return lines


def _player_rows(conn: sqlite3.Connection, player_ids: list[int]) -> list[dict[str, Any]]:
    if not player_ids:
        return []
    placeholders = ",".join("?" for _ in player_ids)
    rows = conn.execute(
        f"""SELECT p.*, t.name AS team_name, t.short_name AS team_short_name,
                   pos.singular_name_short AS position_short_name,
                   s.now_cost, s.selected_by_percent, s.form, s.points_per_game, s.total_points,
                   s.minutes, s.starts, s.goals_scored, s.assists, s.bonus, s.bps,
                   s.expected_goals, s.expected_assists, s.expected_goal_involvements,
                   s.expected_goals_conceded, s.defensive_contribution, s.status, s.news,
                   s.captured_at
              FROM players p LEFT JOIN teams t ON t.id=p.team_id
              LEFT JOIN positions pos ON pos.id=p.element_type
              LEFT JOIN player_snapshots s ON s.id=(
                SELECT latest.id FROM player_snapshots latest
                 WHERE latest.player_id=p.id ORDER BY latest.captured_at DESC, latest.id DESC LIMIT 1
              )
             WHERE p.id IN ({placeholders}) ORDER BY p.id""",
        tuple(player_ids),
    ).fetchall()
    return [dict(row) for row in rows]


def _section(lines: list[str]) -> dict[str, Any]:
    return {"lines": lines}


def build_report(conn: sqlite3.Connection, config: dict[str, Any], gw: int | None = None) -> dict[str, Any]:
    event_id, event = _event_context(conn, gw)
    generated = utc_now()
    deadline = event.get("deadline_time") if event else None
    entry_id = config.get("fpl_entry_id")
    manager = repo.latest_manager_state(conn, entry_id)
    watchlist = repo.active_watchlist_rows(conn)
    squad = repo.squad_rows(conn, int(entry_id), event_id) if entry_id is not None else []
    selected_ids = [int(row["player_id"]) for row in squad]
    selected_ids.extend(int(row["player_id"]) for row in watchlist if int(row["player_id"]) not in selected_ids)
    if config["report"].get("include_all_players") or config["report"].get("player_data_selection") == "all":
        selected_ids = [int(row["id"]) for row in repo.list_active_players(conn)]
    player_rows = _player_rows(conn, selected_ids)
    player_by_id = {int(row["id"]): row for row in player_rows}
    scouting_rows = repo.scouting_current_rows(conn, selected_ids)
    scouting_by_player: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in scouting_rows:
        scouting_by_player[int(row["player_id"])].append(row)
    sections: dict[str, dict[str, Any]] = {}

    sections[SECTION_ORDER[0]] = _section(
        [
            "[FACT] = official FPL API data.",
            "[SCOUTING] = researched inference, with confidence and observed timestamp. May be wrong.",
            "[DERIVED] = computed from FACTs by a named, transparent metric.",
            "[MANUAL] = entered by the manager.",
            "Anything marked STALE is older than the configured freshness window — discount it.",
            *_fetch_health_lines(conn),
        ]
    )

    manager_lines: list[str] = []
    if manager:
        manager_lines.append(
            f"Entry ID: {manager['entry_id']} | FPL team: {_value(manager.get('team_name'))} | "
            f"Manager: {_value(manager.get('player_name'))}"
        )
        manager_lines.append(f"Bank: {_money(manager.get('bank'))} | Team value: {_money(manager.get('team_value'))}")
        manager_lines.append(f"Overall rank: {_value(manager.get('summary_overall_rank'))} (pre-season if unavailable)")
    elif entry_id is None:
        manager_lines.append("No manager data — Team ID not configured")
    else:
        manager_lines.append(f"No manager data synced for Team ID {entry_id}")
    manual_free_transfers = config.get("manual_overrides", {}).get("free_transfers")
    manager_lines.append(f"Free transfers: {_value(manual_free_transfers)}  [MANUAL — not exposed by public API]")
    chips_used = conn.execute("SELECT name,event FROM manager_chips WHERE entry_id=? ORDER BY event,name", (entry_id or -1,)).fetchall()
    manager_lines.append("Chips used: " + (", ".join(f"{row['name']} GW{row['event']}" for row in chips_used) if chips_used else "none"))
    chips = conn.execute("SELECT * FROM chip_definitions ORDER BY start_event,id").fetchall()
    available = ", ".join(
        f"{str(row['name']).upper()}#{row['number'] or 1} (GW{row['start_event']}–{row['stop_event']})" for row in chips
    ) or "unavailable"
    manager_lines.append(f"Chips available: {available}  [FACT]")
    sections[SECTION_ORDER[1]] = _section(manager_lines)

    strategy = repo.strategy_row(conn)
    strategy_source = strategy or config.get("strategy", {})
    sections[SECTION_ORDER[2]] = _section(
        [
            f"Risk posture: {strategy_source.get('risk_posture') or 'not set'}",
            f"Wildcard horizon: {strategy_source.get('wildcard_horizon') or 'not set'}",
            f"Bench Boost plan: {strategy_source.get('bench_boost_plan') or 'not set'}",
            f"Free Hit plan: {strategy_source.get('free_hit_plan') or 'not set'}",
            f"Triple Captain plan: {strategy_source.get('triple_captain_plan') or 'not set'}",
        ]
    )

    squad_lines: list[str] = []
    if entry_id is None:
        squad_lines.append("No squad data — Team ID not configured")
    elif not squad:
        squad_lines.append(f"GW{event_id} deadline has not passed; picks endpoint returns no data yet")
    else:
        squad_lines.append("XI:")
        for row in squad:
            latest = repo.latest_snapshot(conn, int(row["player_id"])) or {}
            squad_lines.append(
                f"  {row.get('singular_name_short') or 'POS'}  {_player_name(row)} ({row.get('team_name') or 'TEAM'}) "
                f"{_money(latest.get('now_cost'))}  own {_percent(latest.get('selected_by_percent'))}  status: {_value(latest.get('status'), 'unknown')}"
            )
        bench = [row for row in squad if int(row["position"]) > 11]
        squad_lines.append("Bench (order): " + (" ".join(f"{row['position']}. {_player_name(row)}" for row in bench) or "none"))
        captain = next((row for row in squad if row.get("is_captain") == 1), None)
        vice = next((row for row in squad if row.get("is_vice_captain") == 1), None)
        squad_lines.append(f"Captain: {_player_name(captain) if captain else '—'} | Vice: {_player_name(vice) if vice else '—'}")
        structure = Counter(row.get("singular_name_short") or "?" for row in squad)
        starting = Counter(row.get("singular_name_short") or "?" for row in squad if int(row["position"]) <= 11)
        squad_lines.append(
            f"Structure: {structure.get('GKP', 0)}-{structure.get('DEF', 0)}-{structure.get('MID', 0)}-{structure.get('FWD', 0)} | "
            f"Formation of XI: {starting.get('GKP', 0)}-{starting.get('DEF', 0)}-{starting.get('MID', 0)}-{starting.get('FWD', 0)}"
        )
    sections[SECTION_ORDER[3]] = _section(squad_lines)

    player_lines: list[str] = [f"For squad + watchlist players (config: {config['report'].get('player_data_selection')}):"]
    if not player_rows:
        player_lines.append("  No squad or watchlist player data selected.")
    for row in player_rows:
        player_lines.extend(
            [
                f"  {_player_name(row)} ({row.get('team_short_name') or row.get('team_name') or 'TEAM'}, {row.get('position_short_name') or 'POS'}) {_money(row.get('now_cost'))}",
                f"    Ownership {_percent(row.get('selected_by_percent'))} | Form {_value(row.get('form'))} | PPG {_value(row.get('points_per_game'))} | Total pts {_value(row.get('total_points'))}",
                f"    Minutes {_value(row.get('minutes'))} | Starts {_value(row.get('starts'))} | Goals {_value(row.get('goals_scored'))} | Assists {_value(row.get('assists'))} | Bonus {_value(row.get('bonus'))} | BPS {_value(row.get('bps'))}",
                f"    xG {_value(row.get('expected_goals'))} | xA {_value(row.get('expected_assists'))} | xGI {_value(row.get('expected_goal_involvements'))} | xGC {_value(row.get('expected_goals_conceded'))} | DEFCON {_value(row.get('defensive_contribution'))}",
                f"    Status: {_value(row.get('status'), 'unknown')} | News: {_value(row.get('news'))}",
            ]
        )
        ppm = points_per_million(conn, int(row["id"]))
        reliability = minutes_reliability(conn, int(row["id"]), 5)
        fdr_values = []
        for horizon in config["report"].get("fixture_horizons", [3, 5, 8]):
            outlook = fixture_outlook(conn, int(row["team_id"]), event_id, int(horizon)) if row.get("team_id") else {}
            fdr_values.append("—" if outlook.get("mean_fdr") is None else f"{outlook['mean_fdr']:.1f}")
        reliability_text = "no GW data yet" if reliability is None else f"{reliability['starts']} starts/{reliability['appearances']} apps, {reliability['total_minutes']} min"
        player_lines.append(f"    [DERIVED] Points per £m: {_value(None if ppm is None else f'{ppm:.2f}')} | Minutes reliability: {reliability_text}")
        player_lines.append(f"    [DERIVED] FDR next horizons: {' / '.join(fdr_values)}")
        strength = attacking_and_defensive_outlook(conn, int(row["team_id"]), event_id, 8) if row.get("team_id") else {}
        attacking_strength = strength.get("attacking_outlook") if strength else None
        defensive_strength = strength.get("defensive_outlook") if strength else None
        attacking_text = "—" if attacking_strength is None else f"{attacking_strength:.1f}"
        defensive_text = "—" if defensive_strength is None else f"{defensive_strength:.1f}"
        player_lines.append(
            "    [DERIVED] Strength outlook next 8 (home/away aware): "
            f"attacking uses opponent strength_defence = {attacking_text}; "
            f"defensive/clean-sheet uses opponent strength_attack = {defensive_text}; "
            "not xG or probability."
        )
    player_lines.append(
        "  [DERIVED] Unimplemented metrics: expected_minutes, start_probability, defcon_potential, "
        "route_diversity, role_security, squad_flexibility, future_transfer_pressure, replacement_options — not implemented"
    )
    sections[SECTION_ORDER[4]] = _section(player_lines)

    fixture_lines: list[str] = [f"GW{event_id} fixtures:"]
    current_fixtures = conn.execute("SELECT * FROM fixtures WHERE event=? ORDER BY kickoff_time,id", (event_id,)).fetchall()
    if not current_fixtures:
        fixture_lines.append("  No fixtures stored for this gameweek.")
    for row in current_fixtures:
        fixture_lines.append(
            f"  {row['kickoff_time'] or 'kickoff unknown'} — {_team_label(conn, row['team_h'])} vs {_team_label(conn, row['team_a'])} "
            f"(H FDR {_value(row['team_h_difficulty'])}, A FDR {_value(row['team_a_difficulty'])})"
        )
    team_ids = sorted({int(row["team_id"]) for row in player_rows if row.get("team_id") is not None})
    for team_id in team_ids:
        team = repo.team_row(conn, team_id)
        outlook = fixture_outlook(conn, team_id, event_id, 8)
        fixture_lines.append(
            f"  {team.get('name') if team else team_id} next 8: "
            + (", ".join(f"{_team_label(conn, item['opponent_team'])}({'H' if item['home'] else 'A'}, {item['difficulty'] or '—'})" for item in outlook["fixtures"]) or "none")
        )
        if outlook["fixture_count"] < 8:
            fixture_lines.append(f"  [DERIVED] Blank flagged: only {outlook['fixture_count']} fixtures in the next 8-event horizon.")
        if outlook["double_events"]:
            fixture_lines.append(f"  [DERIVED] Double flagged in GW{', GW'.join(map(str, outlook['double_events']))}.")
    sections[SECTION_ORDER[5]] = _section(fixture_lines)

    scouting_lines: list[str] = []
    if not scouting_rows:
        scouting_lines.append("No scouting notes for the selected players.")
    stale_days = int(config["report"].get("scouting_stale_after_days", 14))
    now = datetime.now(timezone.utc)
    for player_id, rows in scouting_by_player.items():
        row = player_by_id.get(player_id) or repo.get_player(conn, player_id) or {"id": player_id}
        scouting_lines.append(f"  {_player_name(row)} ({row.get('team_name') or 'TEAM'})")
        for note in rows:
            observed = note.get("observed_at")
            expires = note.get("expires_at")
            observed_dt = parse_utc(observed)
            stale = bool(expires and parse_utc(expires) and parse_utc(expires) < now)
            stale = stale or bool(observed_dt and (now - observed_dt).days > stale_days)
            value = note.get("value_text") if note.get("value_text") is not None else note.get("value_num")
            suffix = " STALE" if stale else ""
            scouting_lines.append(
                f"    [SCOUTING] {note['key']}: {_value(value)} (confidence {note.get('confidence') or 'unknown'}, observed {observed or 'unknown'}){suffix}"
            )
            if note.get("observation"):
                scouting_lines.append(f"      Note: {note['observation']}")
    sections[SECTION_ORDER[6]] = _section(scouting_lines)

    risk_lines: list[str] = []
    availability = []
    for row in player_rows:
        if row.get("status") not in (None, "a"):
            availability.append(f"[FACT] {_player_name(row)}: status {row['status']}; news: {row.get('news') or '—'}")
    risk_lines.extend(availability or ["[FACT] No unavailable or doubtful selected players."])
    for player_id, rows in scouting_by_player.items():
        for note in rows:
            if note.get("key") in {"rotation_risk", "transfer_exit_risk", "role_security_5gw", "injury_uncertainty"} or note.get("confidence") == "low":
                row = player_by_id.get(player_id) or {"id": player_id}
                value = note.get("value_text") if note.get("value_text") is not None else note.get("value_num")
                risk_lines.append(f"[SCOUTING] {_player_name(row)} {note['key']}: {_value(value)} (confidence {note.get('confidence')}, observed {note.get('observed_at')})")
    club_counts = Counter(row.get("team_id") for row in squad)
    concentrated = [team_id for team_id, count in club_counts.items() if team_id is not None and count > 2]
    if concentrated:
        for team_id in concentrated:
            count = club_counts[team_id]
            risk_lines.append(
                f"[DERIVED] Concentration risk: {count} players from {_team_label(conn, team_id)} — maximum club allocation reached; "
                f"another {_team_label(conn, team_id)} target would require selling one."
            )
    else:
        risk_lines.append("[DERIVED] Structural: no club has reached the three-player maximum.")
    sections[SECTION_ORDER[7]] = _section(risk_lines)

    watchlist_lines = []
    if not watchlist:
        watchlist_lines.append("No active watchlist entries.")
    for row in watchlist:
        player = player_by_id.get(int(row["player_id"])) or repo.get_player(conn, int(row["player_id"])) or row
        snapshot = repo.latest_snapshot(conn, int(row["player_id"])) or {}
        fdr = fixture_outlook(conn, int(player["team_id"]), event_id, 3).get("mean_fdr") if player.get("team_id") else None
        watchlist_lines.append(
            f"  {_player_name(player)} ({row.get('team_name') or 'TEAM'}) — {row['status']} — target GW{row.get('target_event_from') or event_id}–{row.get('target_event_to') or event_id} — reason: {row.get('reason') or '—'} — {_money(snapshot.get('now_cost'))}, own {_percent(snapshot.get('selected_by_percent'))}, FDR3 {'—' if fdr is None else f'{fdr:.1f}'}"
        )
    sections[SECTION_ORDER[8]] = _section(watchlist_lines)

    movement_lines = []
    if not selected_ids:
        movement_lines.append("insufficient snapshot history")
    for row in player_rows:
        cost = trend(conn, int(row["id"]), "now_cost", 30)
        ownership = trend(conn, int(row["id"]), "selected_by_percent", 30)
        if cost is None and ownership is None:
            movement_lines.append(f"  {_player_name(row)}: insufficient snapshot history")
            continue
        movement_lines.append(
            f"  {_player_name(row)}: {_money(cost['first'] if cost else None)} → {_money(cost['last'] if cost else None)} "
            f"({cost['absolute_delta'] if cost else '—'}) | own {_value(ownership['first'] if ownership else None)} → {_value(ownership['last'] if ownership else None)} "
            f"({ownership['absolute_delta'] if ownership else '—'}pp) | snapshots {(cost or ownership)['sample_count']}"
        )
    sections[SECTION_ORDER[9]] = _section(movement_lines)

    decision_lines = []
    decisions = repo.decision_rows(conn)
    if not decisions:
        decision_lines.append("No previous decisions logged.")
    for decision in decisions:
        assumptions = json.loads(decision.get("assumptions_json") or "[]")
        invalidators = json.loads(decision.get("invalidators_json") or "[]")
        decision_lines.extend(
            [
                f"  GW{decision['event']} — {decision['action']} — confidence {decision.get('confidence') or 'unknown'} — {decision['created_at']}",
                f"    Reasoning: {decision.get('reasoning') or '—'}",
                f"    Assumptions: {json.dumps(assumptions, ensure_ascii=False)}",
                f"    Would be invalidated by: {json.dumps(invalidators, ensure_ascii=False)}",
                f"    Review: {decision.get('review_notes') or '(not yet reviewed)'}",
            ]
        )
    sections[SECTION_ORDER[10]] = _section(decision_lines)

    gaps = []
    latest_run = repo.latest_fetch_run(conn, "fetch_fpl")
    if latest_run and latest_run.get("status") in {"partial", "failed", "running"}:
        gaps.append(f"[FACT WARNING] Latest fetch_fpl run {latest_run['id']} is {latest_run.get('status')}; some FACT data may be stale.")
    if conn.execute("SELECT COUNT(*) FROM player_gameweeks WHERE minutes IS NOT NULL").fetchone()[0] == 0:
        gaps.append("No GW performance data: season has not started or no per-GW data has been synced.")
    if entry_id is None:
        gaps.append("No manager or squad data: Team ID is not configured.")
    if manual_free_transfers is None:
        gaps.append("Free transfers are manual; verify them in the FPL UI.")
    missing_scouting = [_player_name(row) for row in player_rows if not scouting_by_player.get(int(row["id"]))]
    if missing_scouting:
        gaps.append("No scouting notes for: " + ", ".join(missing_scouting) + ".")
    gaps.append(
        f"event/{event_id}/live may return zero elements pre-season; no live performance rows are present."
        if not conn.execute("SELECT COUNT(*) FROM player_gameweeks WHERE event=? AND minutes IS NOT NULL", (event_id,)).fetchone()[0]
        else "Per-GW performance rows are present for the selected event."
    )
    sections[SECTION_ORDER[11]] = _section([f"  - {gap}" for gap in gaps] or ["  - No known data gaps."])

    return {
        "title": f"FPL BRAIN — GW{event_id}",
        "generated": generated,
        "freshness": _freshness(conn),
        "season": config.get("season", "2026/27"),
        "deadline": deadline,
        "sections": sections,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# {report['title']}",
        f"Generated: {report['generated']}",
        f"Data freshness: {report['freshness']}",
        f"Season: {report['season']} | Deadline: {report['deadline'] or 'unavailable'}",
        "",
    ]
    for heading in SECTION_ORDER:
        lines.append(f"## {heading}")
        lines.extend(report["sections"].get(heading, {}).get("lines", []))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_json(report: dict[str, Any]) -> str:
    return json.dumps(report, indent=2, ensure_ascii=False) + "\n"


render_markdown_report = render_markdown
render_json_report = render_json
