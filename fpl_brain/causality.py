"""Causal cutoff invariants and as-of (point-in-time) resolvers (R4A).

Three distinct snapshots exist and must never be conflated:

``code_snapshot_sha256``
    A fingerprint of the source code / worktree that produced a run.  This is
    what ``analytics.source_snapshot_sha256()`` has always computed and what
    ``projection_runs.source_snapshot_sha256`` has always stored.

``data_snapshot_sha256``
    A content hash of the **causal source data** a certification actually read:
    events, fixtures, event fixture assignment, the active-player bootstrap, the
    as-of official price/status rows, the as-of scouting rows and the as-of
    manager state.  Two certifications with the same data snapshot identity read
    the same causal inputs.

``planning_cutoff``
    The wall-clock instant a planning run claims to represent.  It may not be in
    the future relative to the execution that claims it, because everything the
    run reads must have existed at that instant.

The mutable-state problem is not solved by pretending ``fixtures`` or
``players`` can be replayed historically -- they cannot (see the R4A
verification table).  It is solved by (a) refusing any cutoff that post-dates
execution, and (b) materialising and hashing the causal rows so later ingestion
is detectable rather than silently absorbed.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Mapping, Sequence

from .utils import parse_utc

# A cutoff may lead the execution clock by at most this much.  It exists only so
# that a cutoff minted a moment before the run starts does not fail spuriously;
# it is deliberately tiny and is NOT a budget for any real lead time.
CLOCK_SKEW_TOLERANCE_SECONDS = 2.0

DIAG_PLANNING_CUTOFF_IN_FUTURE = "PLANNING_CUTOFF_IN_FUTURE"
DIAG_DATA_CUTOFF_AFTER_GENERATED = "DATA_CUTOFF_AFTER_GENERATED_AT"
DIAG_DATA_SNAPSHOT_DRIFT = "DATA_SNAPSHOT_DRIFT"
DIAG_PRICE_MISSING_AS_OF_CUTOFF = "PRICE_MISSING_AS_OF_CUTOFF"


class CausalityError(RuntimeError):
    """A causal ordering invariant was violated."""


class PlanningCutoffInFuture(CausalityError):
    """The planning cutoff post-dates the execution that claims it."""

    def __init__(self, cutoff: str, reference_utc: str, lead_seconds: float, label: str) -> None:
        super().__init__(
            f"{DIAG_PLANNING_CUTOFF_IN_FUTURE}: {label} planning cutoff {cutoff} leads the "
            f"execution clock {reference_utc} by {lead_seconds:.1f}s "
            f"(tolerance {CLOCK_SKEW_TOLERANCE_SECONDS:g}s)"
        )
        self.cutoff = cutoff
        self.reference_utc = reference_utc
        self.lead_seconds = lead_seconds
        self.label = label


def _aware(value: datetime | str, *, label: str) -> datetime:
    if isinstance(value, datetime):
        moment = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
        return moment.astimezone(timezone.utc)
    parsed = parse_utc(str(value))
    if parsed is None:
        raise CausalityError(f"{label} is not a valid UTC timestamp: {value!r}")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def assert_causal_cutoff(
    cutoff: datetime | str | None,
    reference_utc: datetime | str,
    *,
    label: str = "planning",
    tolerance_seconds: float = CLOCK_SKEW_TOLERANCE_SECONDS,
) -> None:
    """Fail closed when ``cutoff`` post-dates ``reference_utc``.

    Raises :class:`PlanningCutoffInFuture`.  A ``None`` cutoff is a no-op so
    callers that legitimately have no cutoff are unaffected.
    """

    if cutoff is None:
        return
    cutoff_dt = _aware(cutoff, label=f"{label} cutoff")
    reference_dt = _aware(reference_utc, label=f"{label} reference")
    lead = (cutoff_dt - reference_dt).total_seconds()
    if lead > float(tolerance_seconds):
        raise PlanningCutoffInFuture(
            str(cutoff), str(reference_utc), lead, label
        )


def assert_data_cutoff_not_after_generated(
    data_cutoff: datetime | str | None,
    generated_at: datetime | str | None,
    *,
    tolerance_seconds: float = CLOCK_SKEW_TOLERANCE_SECONDS,
) -> None:
    """A run may not claim a data cutoff later than the moment it was produced."""

    if data_cutoff is None or generated_at is None:
        return
    cutoff_dt = _aware(data_cutoff, label="data cutoff")
    generated_dt = _aware(generated_at, label="generated_at")
    lead = (cutoff_dt - generated_dt).total_seconds()
    if lead > float(tolerance_seconds):
        raise CausalityError(
            f"{DIAG_DATA_CUTOFF_AFTER_GENERATED}: data cutoff {data_cutoff} is {lead:.1f}s after "
            f"generated_at {generated_at}"
        )


# ---------------------------------------------------------------------------
# As-of price resolution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PriceAsOf:
    """Official prices resolved strictly at or before a cutoff."""

    cutoff: str
    event: int | None
    prices: Mapping[int, int]
    captured_at: Mapping[int, str]
    row_ids: Mapping[int, int]

    def price(self, player_id: int) -> int | None:
        return self.prices.get(int(player_id))


def price_snapshot_as_of(
    conn: sqlite3.Connection,
    cutoff: datetime | str,
    *,
    event: int | None = None,
    required_player_ids: Iterable[int] | None = None,
    fallback_to_earliest: bool = True,
) -> PriceAsOf:
    """Resolve each player's official price as of ``cutoff``, deterministically.

    For every player the eligible row is the latest one with
    ``captured_at <= cutoff`` (and ``event_context <= event`` when an event is
    supplied), tie-broken by ``captured_at DESC, id DESC``.

    A player whose first observation is *after* the cutoff has no causal price.
    Such a player is reported as missing rather than silently priced from the
    future; callers that must price them can opt into the earliest-known row via
    ``fallback_to_earliest``, and every fallback is visible in the result.
    """

    cutoff_text = str(cutoff)
    if required_player_ids is not None and not isinstance(required_player_ids, (list, tuple, set)):
        required_player_ids = list(required_player_ids)

    sql = [
        "SELECT id, player_id, now_cost, captured_at, event_context FROM player_snapshots",
        "WHERE now_cost IS NOT NULL AND captured_at <= ?",
    ]
    params: list[Any] = [cutoff_text]
    if event is not None:
        sql.append("AND (event_context IS NULL OR event_context <= ?)")
        params.append(int(event))
    sql.append("ORDER BY player_id, captured_at DESC, id DESC")

    prices: dict[int, int] = {}
    captured: dict[int, str] = {}
    row_ids: dict[int, int] = {}
    for row in conn.execute(" ".join(sql), tuple(params)):
        player_id = int(row["player_id"])
        if player_id in prices:
            continue  # first row per player is the as-of winner
        prices[player_id] = int(row["now_cost"])
        captured[player_id] = str(row["captured_at"])
        row_ids[player_id] = int(row["id"])

    required = [int(pid) for pid in (required_player_ids or ())]
    missing = [pid for pid in required if pid not in prices]
    if missing and fallback_to_earliest:
        # Never invent a price.  Use the earliest KNOWN row for a player who has
        # no causal observation, and record it so the substitution is auditable.
        placeholders = ",".join("?" for _ in missing)
        for row in conn.execute(
            f"SELECT id, player_id, now_cost, captured_at FROM player_snapshots "
            f"WHERE now_cost IS NOT NULL AND player_id IN ({placeholders}) "
            f"ORDER BY player_id, captured_at ASC, id ASC",
            tuple(missing),
        ):
            player_id = int(row["player_id"])
            if player_id in prices:
                continue
            prices[player_id] = int(row["now_cost"])
            captured[player_id] = f"FALLBACK_EARLIEST:{row['captured_at']}"
            row_ids[player_id] = int(row["id"])
        missing = [pid for pid in required if pid not in prices]

    return PriceAsOf(
        cutoff=cutoff_text,
        event=int(event) if event is not None else None,
        prices=prices,
        captured_at=captured,
        row_ids=row_ids,
    )


def missing_required_prices(
    conn: sqlite3.Connection,
    cutoff: datetime | str,
    player_ids: Sequence[int],
    *,
    event: int | None = None,
) -> list[int]:
    """Players among ``player_ids`` with no official price at or before ``cutoff``."""

    snapshot = price_snapshot_as_of(conn, cutoff, event=event, fallback_to_earliest=False)
    return [int(pid) for pid in player_ids if int(pid) not in snapshot.prices]


# ---------------------------------------------------------------------------
# Data snapshot identity
# ---------------------------------------------------------------------------

_DATA_SNAPSHOT_VERSION = "fpl_data_snapshot_v1"


def _rows(conn: sqlite3.Connection, sql: str, params: Sequence[Any] = ()) -> list[list[Any]]:
    return [[_jsonable(value) for value in row] for row in conn.execute(sql, tuple(params))]


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, type(None), bool)):
        return value
    if isinstance(value, bytes):
        return value.hex()
    return str(value)


def capture_causal_data_snapshot(
    conn: sqlite3.Connection,
    *,
    cutoff: datetime | str,
    events: Sequence[int],
    entry_id: int | None = None,
) -> dict[str, Any]:
    """Materialise the causal source rows a certification reads, and hash them.

    Scope is deliberately narrow: only what a GW5-GW8 predictive certification
    actually consumes, resolved strictly at or before ``cutoff``.
    """

    cutoff_text = str(cutoff)
    event_ids = sorted(int(event) for event in events)
    event_placeholders = ",".join("?" for _ in event_ids) or "NULL"

    snapshot: dict[str, Any] = {
        "snapshot_version": _DATA_SNAPSHOT_VERSION,
        "cutoff": cutoff_text,
        "events": event_ids,
        "entry_id": int(entry_id) if entry_id is not None else None,
    }

    # events: identity, deadline and completion state as known at the cutoff
    snapshot["events"] = _rows(
        conn,
        f"SELECT id, name, deadline_time, finished, is_next, is_current FROM events "
        f"WHERE id IN ({event_placeholders}) ORDER BY id",
        event_ids,
    )
    # fixtures: the full row, because kickoff/event/started/finished are mutable
    snapshot["fixtures"] = _rows(
        conn,
        f"SELECT id, event, team_h, team_a, kickoff_time, finished, started "
        f"FROM fixtures WHERE event IN ({event_placeholders}) ORDER BY id",
        event_ids,
    )
    # complete fixtures that any completed-rows-as-of read can reach
    snapshot["completed_fixtures"] = _rows(
        conn,
        "SELECT id, event, team_h, team_a, kickoff_time, finished, started FROM fixtures "
        "WHERE finished=1 AND started=1 AND event IS NOT NULL AND event < ? "
        "AND (kickoff_time IS NULL OR kickoff_time <= ?) ORDER BY id",
        (max(event_ids) + 1 if event_ids else 0, cutoff_text),
    )
    # active-player bootstrap (mutable: team_id, element_type, is_active)
    snapshot["players"] = _rows(
        conn,
        "SELECT id, team_id, element_type, is_active FROM players ORDER BY id",
    )
    # official price/status rows at or before the cutoff
    snapshot["player_snapshots_as_of"] = _rows(
        conn,
        "SELECT id, player_id, captured_at, event_context, now_cost, status, news,"
        " chance_of_playing_this_round FROM player_snapshots WHERE captured_at <= ? "
        "ORDER BY player_id, captured_at DESC, id DESC",
        (cutoff_text,),
    )
    # scouting rows eligible at the cutoff
    snapshot["scouting_notes_as_of"] = _rows(
        conn,
        "SELECT id, player_id, key, value_num, value_text, observed_at, expires_at "
        "FROM scouting_notes WHERE observed_at <= ? ORDER BY player_id, key, observed_at DESC, id DESC",
        (cutoff_text,),
    )
    # manual manager state at or before the cutoff
    snapshot["manager_manual_state_as_of"] = _rows(
        conn,
        "SELECT entry_id, event, free_transfers, bank, source, captured_at,"
        " event_start_free_transfers FROM manager_manual_state"
        " WHERE captured_at <= ? ORDER BY captured_at ASC, entry_id",
        (cutoff_text,),
    )
    # acquisition ledger (current state; recorded so drift is detectable)
    snapshot["manager_acquisitions"] = _rows(
        conn,
        "SELECT id, entry_id, player_id, acquired_event, purchase_price, sold_event FROM"
        " manager_player_acquisitions ORDER BY id",
    )

    payload = json.dumps(snapshot, sort_keys=True, separators=(",", ":"), default=str)
    snapshot_sha256 = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return {"data_snapshot_sha256": snapshot_sha256, "payload": snapshot}


def data_snapshot_drift(
    conn: sqlite3.Connection,
    expected_sha256: str,
    *,
    cutoff: datetime | str,
    events: Sequence[int],
    entry_id: int | None = None,
) -> dict[str, Any]:
    """Recompute the data snapshot identity and report whether it drifted."""

    current = capture_causal_data_snapshot(conn, cutoff=cutoff, events=events, entry_id=entry_id)
    matches = str(current["data_snapshot_sha256"]) == str(expected_sha256)
    return {
        "matches": matches,
        "expected_sha256": str(expected_sha256),
        "current_sha256": current["data_snapshot_sha256"],
        "diagnostic": None if matches else DIAG_DATA_SNAPSHOT_DRIFT,
    }


__all__ = [
    "CLOCK_SKEW_TOLERANCE_SECONDS",
    "CausalityError",
    "DIAG_DATA_CUTOFF_AFTER_GENERATED",
    "DIAG_DATA_SNAPSHOT_DRIFT",
    "DIAG_PLANNING_CUTOFF_IN_FUTURE",
    "DIAG_PRICE_MISSING_AS_OF_CUTOFF",
    "PlanningCutoffInFuture",
    "PriceAsOf",
    "assert_causal_cutoff",
    "assert_data_cutoff_not_after_generated",
    "capture_causal_data_snapshot",
    "data_snapshot_drift",
    "missing_required_prices",
    "price_snapshot_as_of",
]
