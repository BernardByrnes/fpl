"""Deterministic canonical decision packets.

The packet is produced from the database through the canonical planning
context, serialised to canonical JSON, and rendered to Markdown by a pure
function of the packet only. Numeric evidence is never manually restated: any
hand-edited Markdown disagrees with the canonical packet when re-rendered.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import Any

from . import metrics, repositories as repo
from .planning import get_planning_context
from .utils import json_text

PACKET_SCHEMA_VERSION = "fpl_brain_decision_packet.v1"


def _money(value: Any) -> str:
    """Render integer-tenths money exactly like the shared report helpers."""

    return "unavailable" if value is None else f"£{float(value) / 10:.1f}m"


def _value(value: Any, missing: str = "—") -> Any:
    return missing if value is None else str(value)


def build_decision_packet(
    conn: sqlite3.Connection,
    config: dict[str, Any],
    event: int | None = None,
    as_of: str | None = None,
    *,
    route_out: list[int] | None = None,
    route_in: list[int] | None = None,
    manager: dict[str, Any] | None = None,
    routes: dict[str, Any] | None = None,
    candidate_universe: dict[str, Any] | None = None,
    optimizer: dict[str, Any] | None = None,
    final_acceptance: dict[str, Any] | None = None,
    four_gw_decision: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the canonical decision packet plus its planning context.

    ``manager`` is the optional Phase-6A manager-lineup section, ``routes`` the
    optional Phase-7B route-comparison section, and ``four_gw_decision`` the
    rolling four-Gameweek decision section; all are attached verbatim without
    touching the planning context or any predictive input.
    """

    conn.row_factory = sqlite3.Row
    try:
        planning_event = int(event)
    except (TypeError, ValueError) as exc:
        raise ValueError("event must be a positive integer") from exc
    entry_id = config.get("fpl_entry_id")
    context = get_planning_context(
        conn,
        int(entry_id),
        planning_event,
        as_of=as_of,
        season=config.get("season"),
        route_out=route_out,
        route_in=route_in,
        scouting_stale_after_days=int(config.get("report", {}).get("scouting_stale_after_days", 14)),
        official_price_stale_after_hours=config.get("report", {}).get("official_price_stale_after_hours"),
    )

    squad_player_ids = [int(pid) for pid in context.acquisitions.get("active_players", [])]
    exact_squad_rows = repo.squad_rows(conn, int(entry_id), planning_event)
    if len(exact_squad_rows) == 15:
        squad_player_ids = [int(row["player_id"]) for row in exact_squad_rows]
    owned = _player_metric_rows(conn, squad_player_ids, entry_id, planning_event)
    watchlist = repo.active_watchlist_rows(conn)
    candidate_ids = [int(row["player_id"]) for row in watchlist if int(row["player_id"]) not in squad_player_ids]
    candidates = _player_metric_rows(conn, candidate_ids, entry_id, planning_event)

    team_ids = sorted({row.get("team_id") for row in owned if row.get("team_id") is not None})
    fixtures = []
    for team_id in team_ids:
        outlook = metrics.fixture_outlook(conn, int(team_id), planning_event, 8)
        fixtures.append(
            {
                "team_id": int(team_id),
                "fixtures": [
                    {
                        "fixture_id": item["fixture_id"],
                        "event": item["event"],
                        "horizon_event": item.get("horizon_event"),
                        "opponent_team": item["opponent_team"],
                        "opponent": _team_label(conn, item["opponent_team"]),
                        "home": item["home"],
                        "difficulty": item["difficulty"],
                        "kickoff_time": item["kickoff_time"],
                    }
                    for item in outlook["fixtures"]
                ],
                "fixture_count": outlook["fixture_count"],
                "blank_count": outlook["blank_count"],
                "double_events": outlook["double_events"],
            }
        )

    scouting_rows = repo.scouting_current_rows(conn, squad_player_ids + candidate_ids)
    scouting: dict[int, list[dict[str, Any]]] = {}
    for row in scouting_rows:
        scouting.setdefault(int(row["player_id"]), []).append(
            {
                "key": row.get("key"),
                "value": row.get("value_text") if row.get("value_text") is not None else row.get("value_num"),
                "confidence": row.get("confidence"),
                "observed_at": row.get("observed_at"),
                "expires_at": row.get("expires_at"),
                "observation": row.get("observation"),
            }
        )

    packet = {
        "schema_version": PACKET_SCHEMA_VERSION,
        "packet_type": "decision_packet",
        "metadata": {
            "entry_id": context.entry_id,
            "season": context.season,
            "event": context.planning_event,
            "as_of": context.as_of,
            "event_data_state": context.event_data_state,
            "event_data_state_basis": context.event_data_state_basis,
            "deadline": context.deadline,
            "health": context.health,
            "official_runs": context.official_runs,
            "official_price_freshness": (context.official_price_freshness or {}).get("freshness"),
            "latest_official_price_run_at": (context.official_price_freshness or {}).get("latest_official_price_run_at"),
            "official_price_run_id": (context.official_price_freshness or {}).get("official_price_run_id"),
            "official_price_run_age_hours": (context.official_price_freshness or {}).get("official_price_run_age_hours"),
            "official_price_stale_after_hours": (context.official_price_freshness or {}).get("stale_after_hours"),
        },
        "planning_context": context.to_dict(),
        "owned_players": owned,
        "candidates": candidates,
        "fixtures": fixtures,
        "scouting_evidence": {str(player_id): rows for player_id, rows in sorted(scouting.items())},
        "route_inputs": context.route_inputs,
        "manager": manager,
        "routes": routes,
        "candidate_universe": candidate_universe,
        "optimizer": optimizer,
        "final_acceptance": final_acceptance,
        "four_gw_decision": four_gw_decision,
    }
    return {"packet": packet, "context": context}


def _team_label(conn: sqlite3.Connection, team_id: Any) -> str:
    if team_id is None:
        return "TEAM"
    team = repo.team_row(conn, int(team_id))
    if not team:
        return f"Team {team_id}"
    return str(team.get("short_name") or team.get("name") or f"Team {team_id}")


def _player_name(row: dict[str, Any]) -> str:
    return row.get("full_name") or row.get("web_name") or f"Player {row.get('id')}"


def _player_metric_rows(
    conn: sqlite3.Connection,
    player_ids: list[int],
    entry_id: int,
    event: int,
) -> list[dict[str, Any]]:
    """Canonical per-player context rows for the packet (no rendering here)."""

    disclosure: list[dict[str, Any]] = []
    for player_id in player_ids:
        player = repo.get_player(conn, int(player_id)) or {"id": int(player_id)}
        snapshot = repo.latest_snapshot(conn, int(player_id)) or {}
        acquisition = next(
            (
                row
                for row in repo.active_manager_acquisitions(conn, int(entry_id))
                if int(row["player_id"]) == int(player_id)
            ),
            None,
        )
        value = metrics.effective_selling_price(conn, int(entry_id), int(event), int(player_id))
        disclosure.append(
            {
                "player_id": int(player_id),
                "name": _player_name(player),
                "team": _team_label(conn, player.get("team_id")),
                "team_id": player.get("team_id"),
                "position": player.get("element_type"),
                "position_short": next(
                    (
                        str(pos_row["singular_name_short"])
                        for pos_row in conn.execute(
                            "SELECT singular_name_short FROM positions WHERE id=?", (player.get("element_type"),)
                        ).fetchall()
                    ),
                    "POS",
                ),
                "official_market_price": snapshot.get("now_cost"),
                "selected_by_percent": snapshot.get("selected_by_percent"),
                "form": snapshot.get("form"),
                "points_per_game": snapshot.get("points_per_game"),
                "total_points": snapshot.get("total_points"),
                "minutes": snapshot.get("minutes"),
                "starts": snapshot.get("starts"),
                "goals_scored": snapshot.get("goals_scored"),
                "assists": snapshot.get("assists"),
                "bonus": snapshot.get("bonus"),
                "bps": snapshot.get("bps"),
                "expected_goals": snapshot.get("expected_goals"),
                "expected_assists": snapshot.get("expected_assists"),
                "expected_goal_involvements": snapshot.get("expected_goal_involvements"),
                "expected_goals_conceded": snapshot.get("expected_goals_conceded"),
                "defensive_contribution": snapshot.get("defensive_contribution"),
                "status": snapshot.get("status"),
                "news": snapshot.get("news"),
                "snapshot_captured_at": snapshot.get("captured_at"),
                "acquisition": None
                if acquisition is None
                else {
                    "acquired_event": acquisition.get("acquired_event"),
                    "purchase_price": acquisition.get("purchase_price"),
                    "source": acquisition.get("source"),
                },
                "selling_price": value,
            }
        )
    return disclosure


def packet_to_dict(packet: dict[str, Any]) -> dict[str, Any]:
    """Canonical packet document ready for JSON serialisation."""

    return packet


def packet_to_json(packet: dict[str, Any]) -> str:
    """Canonical JSON of one packet. The source artifact for all rendering."""

    return json.dumps(packet, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def packet_sha256(packet: dict[str, Any]) -> str:
    digest = hashlib.sha256(json_text(packet).encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def render_packet_markdown(packet: dict[str, Any]) -> str:
    """Deterministic Markdown rendered from the canonical packet only."""

    metadata = packet["metadata"]
    context = packet["planning_context"]
    health = metadata.get("health") or {}
    lines = [
        f"# FPL Brain — Decision Packet — GW{metadata['event']:02d}",
        "",
        f"Entry: {metadata.get('entry_id')} | Season: {metadata.get('season')} | As of: {metadata.get('as_of')}",
        f"Event data state: {metadata.get('event_data_state')} | Deadline: {_value(metadata.get('deadline'))}",
        f"Data health: {health.get('status')}",
        "Numbers in this Markdown are rendered from the canonical JSON packet only;",
        "any number that disagrees with `<same-name>.json` is unverified.",
        "",
    ]
    for reason in health.get("fail_reasons") or []:
        lines.append(f"- HEALTH FAIL: {reason}")
    for reason in health.get("warn_reasons") or []:
        lines.append(f"- HEALTH WARN: {reason}")

    lines.extend(
        [
            "",
            "## Official Price Freshness",
            "",
            f"- Official price freshness: {metadata.get('official_price_freshness')}",
            f"- Latest official price run: {_value(metadata.get('latest_official_price_run_at'))} "
            f"(run id {_value(metadata.get('official_price_run_id'))}, "
            f"age at packet build: {_value(metadata.get('official_price_run_age_hours'))} h, "
            f"stale threshold: {_value(metadata.get('official_price_stale_after_hours'))} h)",
            "- Staleness is surfaced, never silently rendered as current; official prices are only refreshed by an official fetch, never overwritten manually.",
        ]
    )

    lines.extend(["", "## Manager State", ""])
    manager = context.get("manager_state") or {}
    lines.append(f"- Bank: {_money(manager.get('bank'))} (source: {manager.get('bank_source')})")
    lines.append(
        f"- Free transfers: {_value(manager.get('free_transfers'))} (source: {manager.get('free_transfers_source')})"
    )
    lines.append(f"- Latest manual observation: {_value(manager.get('latest_observation'))}")
    lines.append(
        f"- Verification counter: {_value(manager.get('observation_count'))} manual state observation(s)"
    )
    acquisitions = context.get("acquisitions") or {}
    lines.append(f"- Acquisition reconciliation: {acquisitions.get('reconciliation_status')}")
    lines.append(f"- Active acquisitions: {len(acquisitions.get('active_players') or [])} players")

    lines.extend(["", "## Verified Squad & Selling Prices", ""])
    squad = context.get("squad") or {}
    lines.append(
        f"- Squad state: {squad.get('squad_state')} (squad event: {_value(squad.get('squad_event'))}, "
        f"players: {_value(squad.get('player_count'))})"
    )
    for player in packet["owned_players"]:
        sell = player.get("selling_price") or {}
        lines.append(
            f"  {player['name']} ({player['team']}, {player['position_short']}): "
            f"purchase {_money(sell.get('purchase_price'))} | market {_money(sell.get('official_market_price'))} | "
            f"effective sell {_money(sell.get('effective_selling_price'))} | source {sell.get('source')} | "
            f"status {sell.get('status')}"
        )

    lines.extend(["", "## Fixture Horizon (8)", ""])
    for entry in packet["fixtures"]:
        summary = f"{entry['team_id']} — {len(entry['fixtures'])} fixtures"
        if entry["double_events"]:
            summary += f" | double GW{', GW'.join(map(str, entry['double_events']))}"
        if entry["blank_count"]:
            summary += f" | {entry['blank_count']} blank"
        lines.append(f"- {summary}")
        for item in entry["fixtures"]:
            lines.append(
                f"  GW{item['event']}: {item['opponent']} ({'H' if item['home'] else 'A'}, FDR {_value(item['difficulty'])})"
            )

    lines.extend(["", "## Chips", ""])
    for chip in context.get("chips") or []:
        if chip.get("available_for_event"):
            availability = "available"
        elif chip.get("used"):
            availability = f"used GW{chip.get('used_event')}"
        elif chip.get("expired"):
            availability = "EXPIRED (window closed)"
        else:
            availability = "outside window"
        lines.append(f"- {chip['name']}#{chip.get('number') or 1} {chip.get('window')}: {availability}")

    lines.extend(["", "## Candidate Markets", ""])
    if not packet["candidates"]:
        lines.append("- No watchlist candidates.")
    for player in packet["candidates"]:
        sell = player.get("selling_price") or {}
        lines.append(
            f"  {player['name']} ({player['team']}, {player['position_short']}): market {_money(sell.get('official_market_price'))} | "
            f"xG {_value(player.get('expected_goals'))} | xA {_value(player.get('expected_assists'))} | "
            f"xGI {_value(player.get('expected_goal_involvements'))} | status {_value(player.get('status'))}"
        )

    lines.extend(["", "## Scouting Evidence", ""])
    if not packet["scouting_evidence"]:
        lines.append("- No scouting evidence stored.")
    for player_id, notes in sorted(packet["scouting_evidence"].items(), key=lambda item: int(item[0])):
        names = [
            player["name"]
            for player in packet["owned_players"] + packet["candidates"]
            if str(player["player_id"]) == str(player_id)
        ]
        label = names[0] if names else f"Player {player_id}"
        lines.append(f"- {label}:")
        for note in notes:
            lines.append(
                f"    {note.get('key')}: {_value(note.get('value'))} (confidence {note.get('confidence')}, observed {note.get('observed_at')})"
            )

    if packet.get("route_inputs"):
        route = packet["route_inputs"]
        lines.extend(["", "## Route Inputs", ""])
        lines.append(
            f"- Out: {route.get('transfer_out_ids')} | In: {route.get('transfer_in_ids')} | Legal: {_value(route.get('legal'))}"
        )
        affordability = route.get("affordability") or {}
        lines.append(f"- Affordability: {affordability.get('status')} | Remaining bank: {_money(affordability.get('remaining_bank'))}")

    if packet.get("manager"):
        manager_lineup = packet["manager"]
        lines.extend(["", "## Manager Lineup (Phase 6A, CORE-based, descriptive only)", ""])
        lines.append(
            f"- Phase: {manager_lineup.get('phase_version')} | Scoring basis: {manager_lineup.get('scoring_basis')} "
            f"| Bonus handling: {manager_lineup.get('bonus_handling_status')}"
        )
        runs = manager_lineup.get("predictive_runs") or {}
        lines.append(
            f"- Predictive runs: minutes {runs.get('minutes')} ({(runs.get('minutes_version'))}), "
            f"xPts {runs.get('xpts')} ({(runs.get('xpts_version'))}), MC {runs.get('monte_carlo')} "
            f"({(runs.get('monte_carlo_version'))})"
        )
        lines.append(
            f"- Manager worlds: {manager_lineup.get('worlds')} | Legal policies evaluated: "
            f"{manager_lineup.get('evaluated_policy_count')} | XI/bench skeletons: {manager_lineup.get('skeleton_count')}"
        )
        lines.append("- This is descriptive validation output only; no lineup change, transfer or chip is executed.")
        lines.append("")
        lines.append("| # | mean CORE | median | q10 | q90 | autosub pts | P(vice) | captain | vice |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for index, row in enumerate(manager_lineup.get("top_policies") or [], start=1):
            lines.append(
                f"| {index} | {row['mean_core']:.3f} | {row['median_core']:.1f} | {row['q10_core']:.1f} | "
                f"{row['q90_core']:.1f} | {row['expected_autosub_points_added']:.3f} | "
                f"{row['p_vice_takes_captaincy']:.3f} | {row['captain_name']} | {row['vice_captain_name']} |"
            )

    if packet.get("routes"):
        routes = packet["routes"]
        lines.extend(["", "## Route Comparison (Phase 7B, descriptive only)", ""])
        lines.append(
            f"- Phase: {routes.get('phase_version')} | Price scenario: {routes.get('price_scenario_id')} "
            f"| Planning cutoff: {_value(routes.get('planning_cutoff'))} | Worlds/event: {routes.get('simulations')}"
        )
        lines.append(
            f"- Horizons supported: {', '.join(routes.get('supported_horizons') or []) or 'none'}"
            f" (unsupported: {', '.join(routes.get('unsupported_horizons') or []) or 'none'})"
        )
        lines.append("- Descriptive comparison only; no transfer is recommended or executed.")
        lines.append("")
        lines.append("| route | valid | H1 net | H4 net | H6 net | cum hits | terminal FT | terminal bank |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for row in routes.get("route_table") or []:
            lines.append(
                f"| {row['route_id']} | {row['valid']} | {_value(row.get('H1_net'))} | {_value(row.get('H4_net'))} | "
                f"{_value(row.get('H6_net'))} | {_value(row.get('cumulative_hits'))} | "
                f"{_value(row.get('terminal_ft'))} | {_value(row.get('terminal_bank_tenths'))} |"
            )
        lines.append("")
        for label, members in (routes.get("pareto_frontier") or {}).items():
            lines.append(f"- Pareto frontier {label}: {', '.join(members) or 'none'}")

    if packet.get("candidate_universe"):
        universe_section = packet["candidate_universe"]
        lines.extend(["", "## Candidate Universe (Phase 8A, metadata only)", ""])
        lines.append(
            f"- Phase: {universe_section.get('phase_version')} | Score basis: {universe_section.get('score_basis')} "
            f"| Events: {universe_section.get('supported_events')} | Cutoff: {_value(universe_section.get('planning_cutoff'))}"
        )
        lines.append(
            f"- Universe: {universe_section.get('universe_count')} {universe_section.get('counts_by_position')} | "
            f"search view: {universe_section.get('search_view_count')} {universe_section.get('search_view_counts_by_position')}"
        )
        lines.append(
            f"- Replacement edges: {universe_section.get('replacement_edge_count')} "
            f"(legal single {universe_section.get('legal_single_transfer_count')}, "
            f"illegal single {universe_section.get('illegal_single_transfer_count')}) | "
            f"Squad coverage: {universe_section.get('squad_coverage')}"
        )
        lines.append(f"- Full rows: {universe_section.get('candidate_universe_artifact')}")
        lines.append("- Candidates only; this is not a transfer recommendation.")

    if packet.get("optimizer"):
        optimizer = packet["optimizer"]
        lines.extend(["", "## Route Optimizer (Phase 8B, bounded search, descriptive only)", ""])
        lines.append(
            f"- Phase: {optimizer.get('phase_version')} | Scope: {optimizer.get('search_scope')} "
            f"| Events: {optimizer.get('supported_events')} | Cutoff: {_value(optimizer.get('planning_cutoff'))}"
        )
        lines.append(
            f"- Universe {optimizer.get('full_universe_count')} | pool {optimizer.get('search_pool_count')} "
            f"| promoted routes {optimizer.get('promoted_route_count')} | exact evaluations {optimizer.get('exact_evaluations')}"
        )
        lines.append(f"- H1 frontier: {optimizer.get('h1_frontier')} | SUPPORTED_3GW frontier: {optimizer.get('supported_3gw_frontier')}")
        stability = optimizer.get("stability") or {}
        lines.append(f"- Search stability: {stability.get('overall_flag')} "
                     f"(budget {(stability.get('budget_expansion') or {}).get('status')}, "
                     f"pool {(stability.get('candidate_pool_expansion') or {}).get('status')})")
        lines.append(f"- Full search tree: {optimizer.get('optimizer_artifact')}")
        lines.append("- Bounded engineering search only; not a transfer recommendation.")

    if packet.get("final_acceptance"):
        acceptance = packet["final_acceptance"]
        lines.extend(["", "## Final Acceptance (Phase 8C, engineering acceptance only)", ""])
        lines.append(
            f"- Phase: {acceptance.get('phase_version')} | Cutoff: {_value(acceptance.get('planning_cutoff'))} "
            f"| Events: {acceptance.get('supported_events')} | Score basis: {acceptance.get('score_basis')}"
        )
        lines.append(
            f"- Draw fidelity: {acceptance.get('draw_fidelity')} | Price scenario: {acceptance.get('price_scenario')}"
        )
        lines.append(
            f"- Finalists: {acceptance.get('finalist_count')} | Confirmed frontier: "
            f"{acceptance.get('confirmed_shortlist_pareto_frontier')}"
        )
        high = acceptance.get("high_fidelity") or {}
        lines.append(
            f"- H1 leader 2k -> 10k: {high.get('h1_leader_2k')} -> {high.get('h1_leader_10k')} "
            f"(changed={high.get('h1_leader_changed')}) | HIGH_FIDELITY_H1_LEADER_CHANGED="
            f"{high.get('HIGH_FIDELITY_H1_LEADER_CHANGED')}"
        )
        lines.append(f"- Cross-section audit: {(acceptance.get('cross_section_audit') or {}).get('status')} | "
                     f"Acceptance: {acceptance.get('acceptance')}")
        lines.append(f"- Full artifact: {acceptance.get('final_acceptance_artifact')}")
        lines.append("- Engineering acceptance only; no transfer is recommended or executed.")

    lines.extend(["", "## Data Gaps", ""])
    if context.get("data_gaps"):
        lines.extend(f"- {gap}" for gap in context["data_gaps"])
    else:
        lines.append("- No remaining data gaps.")

    lines.extend(
        [
            "",
            "## Provenance",
            "",
            f"- Packet: {packet.get('schema_version')} | Packet hash: {packet_sha256(packet)}",
            f"- Official runs: {json_text(metadata.get('official_runs'))}",
            f"- Scouting cutoff: {metadata.get('official_runs', {}).get('scouting_cutoff') or 'never'}",
            "- Source: local SQLite FACT tables via canonical PlanningContext; no wall-clock values other than as_of.",
            "",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def verify_packet_markdown(markdown: str, packet_dict: dict[str, Any]) -> bool:
    """Provenance rule: the Markdown must be exactly the packet's re-render."""

    return markdown == render_packet_markdown(packet_dict)
