"""Canonical point-in-time planning context with an explicit data-health gate.

One coherent planning state (verified squad, manager bank/free transfers,
selling prices, chip availability, scouting cutoff, official run provenance,
explicit data gaps) plus a deterministic PASS/WARN/FAIL health gate. Decision
packets, affordability checks, and future projection engines read this instead
of independently re-deriving squad/prices/state from raw tables.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from . import metrics, repositories as repo
from .utils import parse_utc, utc_now

# Deterministic event data semantics.  FPL may revise performance rows during
# post-match review before official finalisation, so reports must not treat
# non-final events as final.
EVENT_STATE_FINAL = "FINAL"

EVENT_STATE_PROVISIONAL = "PROVISIONAL"
EVENT_STATE_IN_PROGRESS = "IN_PROGRESS"
EVENT_STATE_SCHEDULED = "SCHEDULED"
EVENT_STATE_UNKNOWN = "UNKNOWN"

HEALTH_FAIL = "FAIL"
HEALTH_WARN = "WARN"
HEALTH_PASS = "PASS"

DEFAULT_MANUAL_STATE_STALE_DAYS = 7
DEFAULT_OFFICIAL_PRICE_STALE_HOURS = 48

OFFICIAL_PRICE_CURRENT = "CURRENT"
OFFICIAL_PRICE_STALE = "STALE"
OFFICIAL_PRICE_UNKNOWN = "UNKNOWN"


def event_data_state(conn: sqlite3.Connection, event_id: int) -> tuple[str, list[str]]:
    """Classify the stored official finality of one gameweek.

    FINAL requires both ``events.finished=1`` and ``events.data_checked=1``.
    All-finished fixtures with an unfinished event are exactly the observed
    post-match review state and are PROVISIONAL, not final.
    """

    event = next((row for row in repo.event_rows(conn) if int(row["id"]) == int(event_id)), None)
    basis: list[str] = []
    if event is None:
        return EVENT_STATE_UNKNOWN, ["event not present in events table"]
    finished = event.get("finished")
    data_checked = event.get("data_checked")
    fixtures = list(conn.execute("SELECT * FROM fixtures WHERE event=?", (int(event_id),)).fetchall())
    finished_count = sum(1 for row in fixtures if row["finished"] == 1)
    started_unfinished = sum(1 for row in fixtures if row["started"] == 1 and row["finished"] != 1)
    basis.append(f"events.finished={finished}")
    basis.append(f"events.data_checked={data_checked}")
    if fixtures:
        basis.append(f"fixtures {finished_count}/{len(fixtures)} finished")
    if finished == 1 and data_checked == 1:
        return EVENT_STATE_FINAL, basis
    if started_unfinished:
        basis.append(f"{started_unfinished} fixtures started but unfinished")
        return EVENT_STATE_IN_PROGRESS, basis
    if fixtures and finished_count == len(fixtures):
        basis.append("all fixtures finished while event is not officially finalised")
        return EVENT_STATE_PROVISIONAL, basis
    return EVENT_STATE_SCHEDULED, basis


def chips_state(conn: sqlite3.Connection, entry_id: int, planning_event: int) -> list[dict[str, Any]]:
    """Chip availability with the season's two chip sets and their windows.

    Availability is deterministic: inside the definition's `[start_event,
    stop_event]` window and not yet used. A chip whose window has passed has
    expired and can no longer be saved for a later opportunity.
    """

    used = [
        (str(row["name"]), int(row["event"]))
        for row in conn.execute(
            "SELECT name, event FROM manager_chips WHERE entry_id=? AND event IS NOT NULL", (int(entry_id),)
        ).fetchall()
    ]
    definitions = conn.execute(
        """SELECT name, number, chip_type, start_event, stop_event FROM chip_definitions
           ORDER BY start_event, id"""
    ).fetchall()
    state: list[dict[str, Any]] = []
    for row in definitions:
        name = str(row["name"] or "")
        stop_event = row["stop_event"]
        start_event = row["start_event"]
        used_event = next(
            (
                event
                for (chip_name, event) in used
                if chip_name == name
                and start_event is not None
                and stop_event is not None
                and int(start_event) <= event <= int(stop_event)
            ),
            None,
        )
        in_window = (
            start_event is not None
            and stop_event is not None
            and int(start_event) <= int(planning_event) <= int(stop_event)
        )
        state.append(
            {
                "name": name,
                "number": row["number"],
                "chip_type": row["chip_type"],
                "window_start_event": start_event,
                "window_stop_event": stop_event,
                "window": f"GW{start_event}-GW{stop_event}" if start_event is not None and stop_event is not None else None,
                "used": used_event is not None,
                "used_event": used_event,
                "available_for_event": in_window and used_event is None,
                "expired": bool(stop_event is not None and int(planning_event) > int(stop_event)),
            }
        )
    return state


def scouting_cutoff(conn: sqlite3.Connection) -> str | None:
    row = conn.execute("SELECT MAX(observed_at) FROM scouting_notes").fetchone()
    return row[0] if row and row[0] else None


def official_price_freshness(
    conn: sqlite3.Connection,
    player_ids: Sequence[int] | None = None,
    *,
    as_of: str | None = None,
    stale_after_hours: float = DEFAULT_OFFICIAL_PRICE_STALE_HOURS,
) -> dict[str, Any]:
    """Age/provenance of the official market-price snapshot that would feed planning.

    CURRENT / STALE / UNKNOWN by configured age semantics: the timestamp is
    the newest official player-price snapshot (`player_snapshots.captured_at`,
    restricted to the resolved squad when available), so a lagging price run
    is always identifiable without ever overwriting official price facts.
    """

    if player_ids:
        placeholders = ",".join("?" for _ in set(player_ids))
        ids = tuple(set(player_ids))
        row = conn.execute(
            f"SELECT MAX(captured_at) FROM player_snapshots WHERE player_id IN ({placeholders})", ids
        ).fetchone()
    else:
        row = conn.execute("SELECT MAX(captured_at) FROM player_snapshots").fetchone()
    latest_run_at = row[0] if row and row[0] else None
    run_id: int | None = None
    age_hours: float | None = None
    freshness = OFFICIAL_PRICE_UNKNOWN
    if latest_run_at is not None:
        run_row = conn.execute(
            "SELECT fetch_run_id FROM player_snapshots WHERE captured_at=? AND fetch_run_id IS NOT NULL ORDER BY player_id LIMIT 1",
            (latest_run_at,),
        ).fetchone()
        run_id = int(run_row[0]) if run_row and run_row[0] is not None else None
        baseline = parse_utc(as_of) or parse_utc(utc_now())
        run_dt = parse_utc(latest_run_at)
        if baseline is not None and run_dt is not None:
            age_hours = (baseline - run_dt).total_seconds() / 3600.0
            freshness = OFFICIAL_PRICE_CURRENT if age_hours <= float(stale_after_hours) else OFFICIAL_PRICE_STALE
    return {
        "freshness": freshness,
        "latest_official_price_run_at": latest_run_at,
        "official_price_run_id": run_id,
        "official_price_run_age_hours": round(age_hours, 2) if age_hours is not None else None,
        "stale_after_hours": float(stale_after_hours),
    }


@dataclass
class PlanningContext:
    """Coherent planning state at one point in time, never silently mixed."""

    entry_id: int | None
    season: str | None
    planning_event: int
    as_of: str | None
    event_data_state: str
    event_data_state_basis: list[str]
    deadline: str | None
    official_runs: dict[str, Any] = field(default_factory=dict)
    squad: dict[str, Any] = field(default_factory=dict)
    manager_state: dict[str, Any] = field(default_factory=dict)
    acquisitions: dict[str, Any] = field(default_factory=dict)
    selling_prices: list[dict[str, Any]] = field(default_factory=list)
    chips: list[dict[str, Any]] = field(default_factory=list)
    scouting_cutoff: str | None = None
    scouting_stale: bool = False
    official_price_freshness: dict[str, Any] = field(default_factory=dict)
    route_inputs: dict[str, Any] | None = None
    data_gaps: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    health: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "season": self.season,
            "planning_event": self.planning_event,
            "as_of": self.as_of,
            "event_data_state": self.event_data_state,
            "event_data_state_basis": self.event_data_state_basis,
            "deadline": self.deadline,
            "official_runs": self.official_runs,
            "squad": self.squad,
            "manager_state": self.manager_state,
            "acquisitions": self.acquisitions,
            "selling_prices": self.selling_prices,
            "chips": self.chips,
            "scouting_cutoff": self.scouting_cutoff,
            "scouting_stale": self.scouting_stale,
            "official_price_freshness": self.official_price_freshness,
            "route_inputs": self.route_inputs,
            "data_gaps": self.data_gaps,
            "warnings": self.warnings,
            "health": self.health,
        }


def _official_runs(conn: sqlite3.Connection) -> dict[str, Any]:
    runs: dict[str, Any] = {}
    fetch_run = repo.latest_fetch_run(conn, "fetch_fpl")
    sync_run = repo.latest_fetch_run(conn, "sync_manager")
    runs["fetch"] = None if fetch_run is None else {
        "run_id": fetch_run.get("id"),
        "started_at": fetch_run.get("started_at"),
        "finished_at": fetch_run.get("finished_at"),
        "status": fetch_run.get("status"),
        "current_event": fetch_run.get("current_event"),
    }
    runs["manager_sync"] = None if sync_run is None else {
        "run_id": sync_run.get("id"),
        "started_at": sync_run.get("started_at"),
        "finished_at": sync_run.get("finished_at"),
        "status": sync_run.get("status"),
        "current_event": sync_run.get("current_event"),
    }
    runs["scouting_cutoff"] = scouting_cutoff(conn)
    return runs


def _ledger_squad_value_state(
    conn: sqlite3.Connection,
    entry_id: int,
    event: int,
) -> dict[str, Any] | None:
    """Ownership-value state derived from the verified acquisition ledger.

    When official target-event picks are stale or absent (pre-deadline, or a
    user-verified transfer that the public endpoint has not yet reflected),
    the reconcile-certified acquisition ledger IS current ownership — its
    active stints include purchases whose provenance is explicitly manual.
    Returns None when the ledger cannot represent a 15-player squad.
    """

    ledger = repo.active_manager_acquisitions_as_of(conn, int(entry_id), int(event))
    if len(ledger) != 15:
        return None
    players: list[dict[str, Any]] = []
    data_gaps: list[str] = []
    mismatches: list[str] = []
    official_total = 0
    selling_total = 0
    for acquisition in sorted(ledger, key=lambda row: int(row["player_id"])):
        player_id = int(acquisition["player_id"])
        value = metrics.effective_selling_price(conn, int(entry_id), int(event), player_id)
        player = repo.get_player(conn, player_id) or {}
        players.append(
            {
                "player_id": player_id,
                "web_name": player.get("web_name"),
                "full_name": player.get("full_name") or player.get("web_name"),
                "purchase_price": value["purchase_price"],
                "purchase_source": acquisition.get("source"),
                "official_market_price": value["official_market_price"],
                "calculated_selling_price": value["calculated_selling_price"],
                "manual_selling_price": value["manual_selling_price"],
                "effective_selling_price": value["effective_selling_price"],
                "source": value["source"],
                "status": value["status"],
            }
        )
        if value["official_market_price"] is None:
            data_gaps.append(f"official market price missing for player {player_id}")
        else:
            official_total += int(value["official_market_price"])
        if value["status"] == "MISMATCH":
            mismatches.append(f"player {player_id}: SELLING PRICE MISMATCH")
        if value["effective_selling_price"] is None:
            if value["status"] != "MISMATCH":
                data_gaps.append(f"effective selling price missing for player {player_id}")
        else:
            selling_total += int(value["effective_selling_price"])
    status = "SELLING_PRICE_MISMATCH" if mismatches else "DATA_GAP" if data_gaps else "OK"
    return {
        "status": status,
        "entry_id": int(entry_id),
        "event": int(event),
        "squad_state": "acquisition_ledger",
        "squad_source": "acquisition_ledger",
        "squad_event": None,
        "player_count": len(players),
        "official_market_value": official_total if not data_gaps else None,
        "realisable_selling_value": selling_total if not data_gaps and not mismatches else None,
        "players": players,
        "data_gaps": data_gaps,
        "mismatches": mismatches,
        "warnings": [],
    }


def _squad_state_value_state(conn: sqlite3.Connection, entry_id: int, event: int) -> dict[str, Any]:
    exact = repo.complete_squad_rows(conn, int(entry_id), int(event))
    free_hit_events = repo.manager_chip_events(conn, int(entry_id), "freehit")
    if exact:
        value_state = metrics.realisable_squad_value(conn, int(entry_id), int(event))
        squad_state = "temp_chip_overlay" if int(event) in free_hit_events else "exact_target_event"
        value_state["squad_state"] = squad_state
        if squad_state == "temp_chip_overlay":
            value_state.setdefault("warnings", []).append(
                "target-event squad is a temporary Free-Hit overlay; permanent squad differs"
            )
        return value_state
    ledger_state = _ledger_squad_value_state(conn, int(entry_id), int(event))
    if ledger_state is not None:
        return ledger_state
    value_state = metrics.realisable_squad_value(conn, int(entry_id), int(event))
    value_state["squad_state"] = value_state.get("squad_source") or "missing"
    return value_state


def _manager_state_context(
    conn: sqlite3.Connection,
    entry_id: int,
    event: int,
    as_of: str | None,
    season: str | None,
) -> tuple[dict[str, Any], list[str], list[str]]:
    gaps: list[str] = []
    warnings: list[str] = []
    planning = repo.manager_planning_state(conn, int(entry_id), int(event), as_of=as_of)
    observations = repo.list_manager_state_observations(conn, int(entry_id), int(event), as_of=as_of)
    planning["observation_count"] = len(observations)
    planning["earliest_observation"] = observations[0].get("captured_at") if observations else None
    planning["latest_observation"] = observations[-1].get("captured_at") if observations else None
    if planning.get("free_transfers") is None:
        gaps.append(f"free transfers are a DATA GAP for entry {entry_id}, GW{event}")
    if planning.get("bank") is None:
        gaps.append(f"bank is a DATA GAP for entry {entry_id}, GW{event}")
    latest_observation = observations[-1] if observations else None
    if latest_observation and latest_observation.get("captured_at"):
        as_of_baseline = as_of or utc_now()
        captured_age_days = (
            parse_utc(as_of_baseline) - parse_utc(latest_observation["captured_at"])
        ).days
        if captured_age_days > DEFAULT_MANUAL_STATE_STALE_DAYS:
            warnings.append(
                f"manual manager state for GW{event} is {max(0, captured_age_days)} days old but still reconcilable"
            )
            planning["manual_state_stale"] = True
    else:
        planning["manual_state_stale"] = False
    planning["season"] = season
    return planning, gaps, warnings


def _assess_route(
    conn: sqlite3.Connection,
    entry_id: int,
    event: int,
    route_out: Sequence[int] | None,
    route_in: Sequence[int] | None,
) -> tuple[dict[str, Any], list[str]]:
    route: dict[str, Any] = {
        "transfer_out_ids": sorted(int(pid) for pid in (route_out or [])),
        "transfer_in_ids": sorted(int(pid) for pid in (route_in or [])),
        "legal": None,
        "affordability": None,
    }
    if not route["transfer_out_ids"] and not route["transfer_in_ids"]:
        return route, []
    squad = repo.squad_rows(conn, int(entry_id), int(event))
    if not squad:
        return route, [f"transfer route cannot be validated: no stored squad for entry {entry_id}, GW{event}"]
    fail_reasons = []
    owned = [int(row["player_id"]) for row in squad]
    out_ids = [int(pid) for pid in (route_out or [])]
    for player_id in out_ids:
        if player_id not in owned:
            fail_reasons.append(f"transfer route illegal: player {player_id} is not in the verified GW{event} squad")
    affordability = metrics.transfer_affordability(conn, int(entry_id), int(event), out_ids, list(route_in or []))
    route["affordability"] = {
        "status": affordability["status"],
        "data_gaps": affordability.get("data_gaps", []),
        "mismatches": affordability.get("mismatches", []),
        "remaining_bank": affordability.get("remaining_bank"),
    }
    if fail_reasons:
        route["legal"] = False
        for reason in fail_reasons:
            route.setdefault("data_gaps", []).append(reason)
        return route, fail_reasons
    route["legal"] = True
    if affordability["status"] == "SELLING_PRICE_MISMATCH":
        return route, ["transfer route unresolved: " + "; ".join(affordability.get("mismatches") or [])]
    if affordability["status"] in {"DATA_GAP", "INSUFFICIENT_FUNDS"}:
        return route, ["transfer route unresolved: " + "; ".join(affordability.get("data_gaps") or []) or affordability["status"]]
    return route, []


def health_gate(
    *,
    squad_state: dict[str, Any],
    acquisition_status: str | None,
    event_state: str,
    scouting_stale: bool,
    route_unresolved: list[str],
    manual_state_stale: bool,
    unplaced_pending_fixtures: int,
    official_price_freshness: str | None = None,
    unresolved_override: bool = False,
) -> dict[str, Any]:
    """Deterministic PASS/WARN/FAIL gate over the assembled planning state."""

    fail_reasons: list[str] = []
    warn_reasons: list[str] = []
    value_status = squad_state.get("status")
    if value_status is None or squad_state.get("player_count") != 15:
        fail_reasons.append("no verified 15-player squad")
    elif value_status == "SELLING_PRICE_MISMATCH":
        fail_reasons.append("selling price MISMATCH: " + "; ".join(squad_state.get("mismatches") or []))
    if acquisition_status in {"mismatch"}:
        fail_reasons.append("manager state irreconcilable: acquisition reconciliation reported a mismatch")
    if unresolved_override:
        fail_reasons.append(
            "manager state cannot be resolved from official data and no explicit user-confirmed override "
            "exists for this event; refusing to fall back to stale official state"
        )
    value_gaps = squad_state.get("data_gaps") or []
    if value_gaps and value_status not in {"SELLING_PRICE_MISMATCH"}:
        if any("effective selling price" in gap or "official market price" in gap for gap in value_gaps):
            fail_reasons.append("outgoing selling price DATA GAP; transfers cannot be priced safely")
        else:
            fail_reasons.append("planning-state reconciliation failed: " + "; ".join(value_gaps))
    for reason in route_unresolved:
        fail_reasons.append(reason)
    if event_state in {EVENT_STATE_PROVISIONAL, EVENT_STATE_IN_PROGRESS}:
        warn_reasons.append(f"event performance data is {event_state}; await official finalisation and re-fetch")
    if scouting_stale:
        warn_reasons.append("scouting evidence is stale relative to the configured freshness window")
    if manual_state_stale:
        warn_reasons.append("manual manager state is old but still reconcilable")
    if official_price_freshness in {OFFICIAL_PRICE_STALE, OFFICIAL_PRICE_UNKNOWN}:
        warn_reasons.append(
            "official price data is not verifiably current "
            f"({official_price_freshness}); market prices may have moved — refresh from the official API before pre-deadline affordability decisions"
        )
    if unplaced_pending_fixtures:
        warn_reasons.append(
            f"{unplaced_pending_fixtures} pending fixtures cannot be placed in the fixture horizon and are excluded from numeric context"
        )
    status = HEALTH_FAIL if fail_reasons else (HEALTH_WARN if warn_reasons else HEALTH_PASS)
    return {"status": status, "fail_reasons": fail_reasons, "warn_reasons": warn_reasons}


def get_planning_context(
    conn: sqlite3.Connection,
    entry_id: int,
    event: int,
    as_of: str | None = None,
    *,
    season: str | None = None,
    route_out: Sequence[int] | None = None,
    route_in: Sequence[int] | None = None,
    scouting_stale_after_days: int | None = None,
    official_price_stale_after_hours: float | None = None,
) -> PlanningContext:
    """Resolve a coherent, gap-aware planning state for one entry and event.

    Manual observations are resolved as of ``as_of`` when given; absent
    explicit evidence is surfaced through ``data_gaps`` and never silently
    mixed with newer or older states.
    """

    conn.row_factory = sqlite3.Row
    event_state, event_basis = event_data_state(conn, int(event))
    deadline = next(
        (row.get("deadline_time") for row in repo.event_rows(conn) if int(row["id"]) == int(event)), None
    )
    managerial, state_gaps, state_warnings = _manager_state_context(conn, int(entry_id), int(event), as_of, season)
    value_state = _squad_state_value_state(conn, int(entry_id), int(event))
    acquisition_status: str | None = None
    transfers_available, transfer_rows = repo.manager_transfer_history(conn, int(entry_id))
    if transfers_available:
        for row in transfer_rows:
            if isinstance(row, dict) and row.get("entry") is not None and int(row["entry"]) != int(entry_id):
                acquisition_status = "mismatch"
    reconcile = repo.reconcile_manager_acquisitions(
        conn,
        int(entry_id),
        int(event),
        transfer_rows,
        transfers_available,
        captured_at=utc_now(),
        dry_run=True,
    )
    if reconcile.get("status") in {"mismatch"}:
        acquisition_status = "mismatch"
    elif reconcile.get("status") == "data_gap":
        acquisition_status = "data_gap"
    provenance_lag_warning: list[str] = []
    if reconcile.get("status") == "data_gap" and "does not reconcile" in str(reconcile.get("message") or ""):
        provenance_lag_warning.append(
            "verified acquisition ledger is ahead of official transfer history: a user-verified manual "
            "transfer is not yet present in public data; the manual observation takes precedence until "
            "the official row appears and is then reconciled into the ledger"
        )
    active = repo.active_manager_acquisitions_as_of(conn, int(entry_id), int(event))
    selling_prices: list[dict[str, Any]] = []
    for acquisition in active:
        player_id = int(acquisition["player_id"])
        value = metrics.effective_selling_price(conn, int(entry_id), int(event), player_id)
        manual_observations = repo.list_manager_selling_price_observations(
            conn, int(entry_id), int(event), player_id
        )
        selling_prices.append(
            {
                **value,
                "observation_count": len(manual_observations),
                "latest_manual_observation": manual_observations[-1].get("captured_at") if manual_observations else None,
            }
        )
    chips = chips_state(conn, int(entry_id), int(event))
    cutoff = scouting_cutoff(conn)
    stale = False
    if cutoff is not None:
        cutoff_dt = parse_utc(cutoff)
        now = parse_utc(as_of) or parse_utc(utc_now())
        if cutoff_dt is not None and now is not None and (now - cutoff_dt).days > (scouting_stale_after_days or 14):
            stale = True
    route_inputs = None
    route_unresolved: list[str] = []
    if route_out is not None or route_in is not None:
        route_inputs, route_unresolved = _assess_route(conn, int(entry_id), int(event), route_out, route_in)
    unplaced = repo.unplaced_pending_fixture_rows(conn, int(event), 8)
    price_threshold = (
        float(official_price_stale_after_hours)
        if official_price_stale_after_hours is not None
        else DEFAULT_OFFICIAL_PRICE_STALE_HOURS
    )
    squad_player_ids = [int(row["player_id"]) for row in (value_state.get("players") or [])] or None
    price_freshness = official_price_freshness(
        conn, squad_player_ids, as_of=as_of, stale_after_hours=price_threshold
    )
    context = PlanningContext(
        entry_id=int(entry_id),
        season=season,
        planning_event=int(event),
        as_of=as_of or utc_now(),
        event_data_state=event_state,
        event_data_state_basis=event_basis,
        deadline=deadline,
        official_runs=_official_runs(conn),
        squad={
            **{key: value_state.get(key) for key in (
                "status", "squad_state", "squad_event", "squad_source", "player_count",
                "official_market_value", "realisable_selling_value", "data_gaps", "mismatches", "warnings",
            )},
            "players": [
                {"player_id": int(row["player_id"]), "web_name": row.get("web_name"), "purchase_source": row.get("purchase_source")}
                for row in (value_state.get("players") or [])
            ],
        },
        manager_state=managerial,
        acquisitions={
            "active_players": [int(acquisition["player_id"]) for acquisition in active],
            "reconciliation_status": acquisition_status or reconcile.get("status"),
            "reconciliation_message": reconcile.get("message"),
        },
        selling_prices=selling_prices,
        chips=chips,
        scouting_cutoff=cutoff,
        scouting_stale=stale,
        official_price_freshness=price_freshness,
        route_inputs=route_inputs,
        data_gaps=list(state_gaps) + list(value_state.get("data_gaps") or []),
        warnings=list(state_warnings) + list(value_state.get("warnings") or []) + provenance_lag_warning,
    )
    context.health = health_gate(
        squad_state=value_state,
        acquisition_status=acquisition_status,
        event_state=event_state,
        scouting_stale=stale,
        route_unresolved=route_unresolved,
        manual_state_stale=bool(managerial.get("manual_state_stale")),
        unplaced_pending_fixtures=len(unplaced),
        official_price_freshness=price_freshness.get("freshness"),
        # Only the CURRENT decision event's own transfer state must be explicitly
        # confirmed.  A strictly future event (e.g. GW5 when the confirmed moves
        # happened in GW4) has no decision state to protect yet — its predictions
        # do not consume free transfers or bank — so the ledger-ahead-of-official
        # condition must not block its freeze.
        unresolved_override=(
            bool(provenance_lag_warning)
            and managerial.get("override_authoritative") is not True
            and int(event) <= max((int(row.get("acquired_event") or 0) for row in active), default=0)
        ),
    )
    return context
