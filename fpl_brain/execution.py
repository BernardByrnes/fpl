"""Production execution controller (R2A).

Operational infrastructure only.  This module never touches the predictive
models and never writes to the projection spine; it governs *how* a production
planning run executes.

The five guarantees
-------------------
1. **Run identity** — one explicit ``run_uuid`` with an auditable lifecycle
   (``CREATED → RUNNING → {COMPLETE | FAILED | CANCEL_REQUESTED → CANCELLED}``).
2. **Single-run / single-writer lease** — enforced in the database by a partial
   unique index over ``status='ACTIVE'``, not by a lock file.  A stale lease is
   reclaimable only through one explicit, audited rule.
3. **Hard wall-clock budget** — ``hard_stop_at`` is executable: nothing starts
   after it, and a stage whose declared minimum exceeds the remaining budget is
   refused with ``STAGE_SKIPPED_INSUFFICIENT_TIME`` rather than started.
4. **Cooperative cancellation** — a token persisted on the run row so a *separate
   process* can request cancellation; checked before every stage and at the
   points production code asks for it (each event, each family, MC, route
   search, between optimizer batches).
5. **Owned process trees** — children are registered (``pid -> argv``) and
   terminated as a tree; unrelated processes are never signalled.

Design note on ``projection_runs``: the projection spine is append-only
predictive provenance and is deliberately **not** reused here.  The controller
records execution provenance in ``execution_*`` tables and only ever *reads*
``projection_runs``, to decide whether completed semantic work can be reused.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import socket
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Iterable, Mapping

from .causality import assert_causal_cutoff
from .process_control import OwnedProcessTree

LOGGER = logging.getLogger(__name__)

# --- lifecycle statuses ----------------------------------------------------
RUN_CREATED = "CREATED"
RUN_RUNNING = "RUNNING"
RUN_CANCEL_REQUESTED = "CANCEL_REQUESTED"
RUN_CANCELLED = "CANCELLED"
RUN_FAILED = "FAILED"
RUN_COMPLETE = "COMPLETE"
RUN_STATUSES = (
    RUN_CREATED,
    RUN_RUNNING,
    RUN_CANCEL_REQUESTED,
    RUN_CANCELLED,
    RUN_FAILED,
    RUN_COMPLETE,
)
TERMINAL_RUN_STATUSES = (RUN_CANCELLED, RUN_FAILED, RUN_COMPLETE)

# --- stage statuses --------------------------------------------------------
STAGE_NOT_STARTED = "NOT_STARTED"
STAGE_RUNNING = "RUNNING"
STAGE_COMPLETE = "COMPLETE"
STAGE_FAILED = "FAILED"
STAGE_CANCELLED = "CANCELLED"
STAGE_SKIPPED_INSUFFICIENT_TIME = "SKIPPED_INSUFFICIENT_TIME"

# --- restart classifications ----------------------------------------------
REUSE_COMPLETE = "REUSE_COMPLETE"
RETRY_SAFE = "RETRY_SAFE"
BLOCKED_PARTIAL_STATE = "BLOCKED_PARTIAL_STATE"
NOT_STARTED = "NOT_STARTED"

# --- canonical production stages ------------------------------------------
PREFLIGHT = "PREFLIGHT"
OFFICIAL_FETCH = "OFFICIAL_FETCH"
BASELINE = "BASELINE"
MINUTES = "MINUTES"
TEAM_STRENGTH = "TEAM_STRENGTH"
PLAYER_RATES = "PLAYER_RATES"
XPTS = "XPTS"
MONTE_CARLO = "MONTE_CARLO"
ROUTE_SEARCH = "ROUTE_SEARCH"
DECISION_BOARD = "DECISION_BOARD"
PERSIST = "PERSIST"

PRODUCTION_STAGES = (
    PREFLIGHT,
    OFFICIAL_FETCH,
    BASELINE,
    MINUTES,
    TEAM_STRENGTH,
    PLAYER_RATES,
    XPTS,
    MONTE_CARLO,
    ROUTE_SEARCH,
    DECISION_BOARD,
    PERSIST,
)

# Minimum seconds a stage needs to be worth starting.  These are declarations
# of intent, not measurements: the point is that the controller *refuses* to
# start a stage it cannot finish before the hard stop.
STAGE_MINIMUM_SECONDS: dict[str, float] = {
    PREFLIGHT: 5.0,
    OFFICIAL_FETCH: 30.0,
    BASELINE: 30.0,
    MINUTES: 60.0,
    TEAM_STRENGTH: 30.0,
    PLAYER_RATES: 30.0,
    XPTS: 30.0,
    MONTE_CARLO: 180.0,
    ROUTE_SEARCH: 120.0,
    DECISION_BOARD: 60.0,
    PERSIST: 10.0,
}

# Stages that only read external/recorded state are safe to re-run.  Stages that
# write projection provenance are reused when a completed run exists, and are
# otherwise blocked if a previous attempt died mid-write.
RETRY_SAFE_STAGES = frozenset({PREFLIGHT, OFFICIAL_FETCH, DECISION_BOARD})

# Stage -> projection model family, for the pre-generation semantic check.
STAGE_PROJECTION_FAMILY: dict[str, str] = {
    BASELINE: "baseline",
    MINUTES: "minutes_v1",
    TEAM_STRENGTH: "team_strength_v1",
    PLAYER_RATES: "player_rates_v1",
    XPTS: "xpts_v1",
    MONTE_CARLO: "monte_carlo_v1",
}

LEASE_KIND_RUN = "run"
LEASE_KIND_WRITER = "writer"
LEASE_ACTIVE = "ACTIVE"
LEASE_RELEASED = "RELEASED"
LEASE_EXPIRED = "EXPIRED"
LEASE_RECLAIMED = "RECLAIMED"


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ExecutionError(RuntimeError):
    """Base class for controller failures."""


class LeaseError(ExecutionError):
    """A lease could not be acquired, or was lost."""


class RunCancelled(ExecutionError):
    """Cooperative cancellation was requested; no new work may start."""

    def __init__(self, message: str = "run cancellation requested") -> None:
        super().__init__(message)


class DeadlineExceeded(ExecutionError):
    """The hard wall-clock stop has been reached."""

    def __init__(self, stage: str, remaining: float) -> None:
        super().__init__(
            f"hard stop reached before stage {stage!r} (remaining {remaining:.1f}s)"
        )
        self.stage = stage
        self.remaining = remaining


class StageSkippedInsufficientTime(ExecutionError):
    """A stage was refused because it cannot finish inside the remaining budget."""

    def __init__(self, stage: str, remaining: float, required: float) -> None:
        super().__init__(
            f"STAGE_SKIPPED_INSUFFICIENT_TIME: stage {stage!r} needs {required:.1f}s "
            f"but only {remaining:.1f}s remain"
        )
        self.stage = stage
        self.remaining = remaining
        self.required = required


# ---------------------------------------------------------------------------
# UTC helpers (timezone-aware by construction)
# ---------------------------------------------------------------------------


def utc_now_dt() -> datetime:
    """Current time as an aware UTC datetime."""

    return datetime.now(timezone.utc)


def format_utc(moment: datetime) -> str:
    """Serialise an **aware** datetime to canonical ``...Z`` text.

    Refuses naive datetimes outright: a naive value is ambiguous local wall
    clock, and suffixing ``Z`` onto it would silently mislabel it as UTC.
    """

    if not isinstance(moment, datetime):
        raise TypeError(f"expected datetime, got {type(moment).__name__}")
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise ValueError(
            "refusing to format a naive datetime as UTC; pass an aware datetime "
            "(use utc_now_dt() or parse with an explicit timezone)"
        )
    return moment.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_utc_dt(value: str | None) -> datetime | None:
    """Parse stored UTC text into an aware datetime; naive input is read as UTC."""

    if not value:
        return None
    text = str(value).strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def add_seconds(moment: datetime, seconds: float) -> datetime:
    return moment + timedelta(seconds=float(seconds))


# ---------------------------------------------------------------------------
# Semantic identity
# ---------------------------------------------------------------------------


def canonical_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def semantic_run_key(
    planning_event: int | None,
    planning_cutoff: str | None,
    *,
    families: Iterable[str] | None = None,
    config_hash: str | None = None,
    source_snapshot_sha256: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> str:
    """One identity for a unit of production work.

    Two runners that would compute the same semantic work produce the same key,
    which is what lets the lease block the second one and lets restart detect a
    completed unit instead of recomputing it.
    """

    payload: dict[str, Any] = {
        "planning_event": int(planning_event) if planning_event is not None else None,
        "planning_cutoff": planning_cutoff,
        "families": sorted(str(f) for f in (families or ())),
        "config_hash": config_hash,
        "source_snapshot_sha256": source_snapshot_sha256,
    }
    if extra:
        payload["extra"] = {str(k): extra[k] for k in sorted(extra)}
    return canonical_hash(payload)


def stage_semantic_key(run_key: str, stage: str, discriminator: str | None = None) -> str:
    """Identity of one stage inside a run, optionally discriminated.

    ``discriminator`` exists precisely because the R1 audit found two
    semantically different rate-baseline runs sharing one indistinguishable
    identity; a discriminator keeps such cases apart.
    """

    return canonical_hash(
        {"run_key": run_key, "stage": str(stage), "discriminator": discriminator}
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExecutionConfig:
    heartbeat_ttl_seconds: float = 120.0
    lease_ttl_seconds: float = 900.0
    grace_seconds: float = 5.0
    safety_margin_seconds: float = 0.0
    stage_minimums: Mapping[str, float] = field(
        default_factory=lambda: dict(STAGE_MINIMUM_SECONDS)
    )

    def minimum_for(self, stage: str) -> float:
        return float(self.stage_minimums.get(str(stage), 0.0))


@dataclass
class RunIdentity:
    run_uuid: str
    planning_event: int | None
    planning_cutoff: str | None
    semantic_run_key: str
    status: str
    hard_stop_at: str | None = None
    started_at: str | None = None
    heartbeat_at: str | None = None
    current_stage: str | None = None
    owner_pid: int | None = None
    owner_host: str | None = None
    cancel_requested_at: str | None = None
    finished_at: str | None = None
    failure_reason: str | None = None
    label: str | None = None

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "RunIdentity":
        return cls(
            run_uuid=row["run_uuid"],
            planning_event=row["planning_event"],
            planning_cutoff=row["planning_cutoff"],
            semantic_run_key=row["semantic_run_key"],
            status=row["status"],
            hard_stop_at=row["hard_stop_at"],
            started_at=row["started_at"],
            heartbeat_at=row["heartbeat_at"],
            current_stage=row["current_stage"],
            owner_pid=row["owner_pid"],
            owner_host=row["owner_host"],
            cancel_requested_at=row["cancel_requested_at"],
            finished_at=row["finished_at"],
            failure_reason=row["failure_reason"],
            label=row["label"],
        )

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


# ---------------------------------------------------------------------------
# Projection provenance probe (read-only)
# ---------------------------------------------------------------------------


def find_completed_projection_run(
    conn: sqlite3.Connection,
    *,
    model_family: str,
    planning_event: int,
    data_cutoff: str,
    source_snapshot_sha256: str | None = None,
) -> int | None:
    """Read-only lookup of a completed accepted run for the same semantic work.

    This is the pre-generation check: it never writes, never deletes, and never
    alters the append-only projection spine.
    """

    sql = (
        "SELECT id FROM projection_runs WHERE model_family=? AND planning_event=? "
        "AND data_cutoff=? AND status='complete'"
    )
    params: list[Any] = [str(model_family), int(planning_event), str(data_cutoff)]
    if source_snapshot_sha256 is not None:
        sql += " AND source_snapshot_sha256=?"
        params.append(str(source_snapshot_sha256))
    sql += " ORDER BY id LIMIT 1"
    row = conn.execute(sql, tuple(params)).fetchone()
    return int(row[0]) if row else None


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------


class ExecutionController:
    """Owns one production run: identity, lease, budget, cancellation, children."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        *,
        run_uuid: str | None = None,
        config: ExecutionConfig | None = None,
        wall_clock: Callable[[], datetime] | None = None,
        monotonic: Callable[[], float] | None = None,
        pid: int | None = None,
        host: str | None = None,
        tree: OwnedProcessTree | None = None,
    ) -> None:
        self.conn = conn
        self.config = config or ExecutionConfig()
        self.run_uuid = run_uuid or str(uuid.uuid4())
        self._wall_clock = wall_clock or utc_now_dt
        self._monotonic = monotonic or time.monotonic
        self.pid = int(pid) if pid is not None else os.getpid()
        self.host = host or socket.gethostname()
        self.tree = tree or OwnedProcessTree(label=self.run_uuid)
        self._lease_ids: list[int] = []
        self._stage_ids: dict[str, int] = {}

    # -- clock -------------------------------------------------------------
    def now_dt(self) -> datetime:
        moment = self._wall_clock()
        if moment.tzinfo is None or moment.utcoffset() is None:
            raise ValueError("wall_clock must return an aware datetime")
        return moment.astimezone(timezone.utc)

    def now(self) -> str:
        return format_utc(self.now_dt())

    def _mono(self) -> float:
        return float(self._monotonic())

    # -- run lifecycle -----------------------------------------------------
    def create_run(
        self,
        *,
        planning_event: int | None = None,
        planning_cutoff: str | None = None,
        hard_stop_at: datetime | str,
        label: str | None = None,
        families: Iterable[str] | None = None,
        config_hash: str | None = None,
        source_snapshot_sha256: str | None = None,
        semantic_key: str | None = None,
    ) -> RunIdentity:
        if isinstance(hard_stop_at, datetime):
            hard_stop_text = format_utc(hard_stop_at)
        else:
            parsed = parse_utc_dt(hard_stop_at)
            if parsed is None:
                raise ValueError(f"hard_stop_at is not a valid UTC timestamp: {hard_stop_at!r}")
            hard_stop_text = format_utc(parsed)
        key = semantic_key or semantic_run_key(
            planning_event,
            planning_cutoff,
            families=families,
            config_hash=config_hash,
            source_snapshot_sha256=source_snapshot_sha256,
        )
        now = self.now()
        self.conn.execute(
            """INSERT INTO execution_runs(
                 run_uuid, label, planning_event, planning_cutoff, semantic_run_key,
                 status, current_stage, owner_pid, owner_host, hard_stop_at,
                 heartbeat_at, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                self.run_uuid,
                label,
                int(planning_event) if planning_event is not None else None,
                planning_cutoff,
                key,
                RUN_CREATED,
                None,
                self.pid,
                self.host,
                hard_stop_text,
                now,
                now,
                now,
            ),
        )
        self.conn.commit()
        return self.run()

    def run(self) -> RunIdentity:
        row = self.conn.execute(
            "SELECT * FROM execution_runs WHERE run_uuid=?", (self.run_uuid,)
        ).fetchone()
        if row is None:
            raise ExecutionError(f"run {self.run_uuid} not found")
        return RunIdentity.from_row(row)

    def _update_run(self, **fields: Any) -> None:
        if not fields:
            return
        fields["updated_at"] = self.now()
        assignments = ", ".join(f"{name}=?" for name in fields)
        self.conn.execute(
            f"UPDATE execution_runs SET {assignments} WHERE run_uuid=?",
            (*fields.values(), self.run_uuid),
        )
        self.conn.commit()

    def start(self) -> RunIdentity:
        current = self.run()
        if current.status != RUN_CREATED:
            raise ExecutionError(f"cannot start run in status {current.status}")
        now = self.now()
        self._update_run(status=RUN_RUNNING, started_at=now, heartbeat_at=now)
        return self.run()

    def heartbeat(self) -> str:
        now_dt = self.now_dt()
        now = format_utc(now_dt)
        expires = format_utc(add_seconds(now_dt, self.config.lease_ttl_seconds))
        self._update_run(heartbeat_at=now)
        for lease_id in list(self._lease_ids):
            self.conn.execute(
                "UPDATE execution_leases SET heartbeat_at=?, expires_at=? "
                "WHERE id=? AND status='ACTIVE'",
                (now, expires, lease_id),
            )
        self.conn.commit()
        return now

    def set_stage(self, stage: str) -> None:
        self._update_run(current_stage=str(stage))

    def finish(self, status: str, failure_reason: str | None = None) -> RunIdentity:
        if status not in TERMINAL_RUN_STATUSES:
            raise ExecutionError(f"{status} is not a terminal run status")
        self._update_run(status=status, finished_at=self.now(), failure_reason=failure_reason)
        self.release_all_leases()
        return self.run()

    # -- cancellation ------------------------------------------------------
    def request_cancel(self, reason: str | None = None) -> RunIdentity:
        """Request cancellation.  Safe to call from a *different* process."""

        current = self.run()
        if current.status in TERMINAL_RUN_STATUSES:
            return current
        now = self.now()
        self._update_run(
            status=RUN_CANCEL_REQUESTED, cancel_requested_at=now, cancel_reason=reason
        )
        return self.run()

    def cancel_requested(self) -> bool:
        """Re-read the run row so another process's request is observed."""

        current = self.run()
        return bool(current.cancel_requested_at) or current.status in (
            RUN_CANCEL_REQUESTED,
            RUN_CANCELLED,
        )

    def check_cancel(self) -> None:
        """Cooperative checkpoint.  Raises ``RunCancelled`` when requested."""

        if self.cancel_requested():
            raise RunCancelled(f"run {self.run_uuid} cancellation requested")

    def acknowledge_cancel(self) -> RunIdentity:
        """Move CANCEL_REQUESTED -> CANCELLED once work has actually stopped."""

        current = self.run()
        if current.status == RUN_CANCELLED:
            return current
        if current.status != RUN_CANCEL_REQUESTED:
            raise ExecutionError(f"cannot acknowledge cancellation in status {current.status}")
        return self.finish(RUN_CANCELLED)

    # -- wall-clock budget -------------------------------------------------
    def hard_stop(self) -> datetime:
        current = self.run()
        parsed = parse_utc_dt(current.hard_stop_at)
        if parsed is None:
            raise ExecutionError(f"run {self.run_uuid} has no hard_stop_at")
        return parsed

    def remaining_seconds(self) -> float:
        remaining = (self.hard_stop() - self.now_dt()).total_seconds()
        return remaining - float(self.config.safety_margin_seconds)

    def ensure_can_start(self, stage: str) -> float:
        """Refuse to start a stage that cannot finish before the hard stop.

        Raises ``DeadlineExceeded`` when the hard stop has passed, and
        ``StageSkippedInsufficientTime`` when the remaining budget is smaller
        than the stage's declared minimum.
        """

        remaining = self.remaining_seconds()
        if remaining <= 0:
            raise DeadlineExceeded(str(stage), remaining)
        required = self.config.minimum_for(stage)
        if remaining < required:
            raise StageSkippedInsufficientTime(str(stage), remaining, required)
        return remaining

    # -- leases ------------------------------------------------------------
    def _insert_lease(self, lease_kind: str, lease_key: str, ttl_seconds: float, note: str | None = None):
        now_dt = self.now_dt()
        now = format_utc(now_dt)
        expires = format_utc(add_seconds(now_dt, ttl_seconds))
        try:
            cursor = self.conn.execute(
                """INSERT INTO execution_leases(
                     lease_kind, lease_key, run_uuid, owner_pid, owner_host,
                     acquired_at, heartbeat_at, expires_at, status, note)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    lease_kind,
                    lease_key,
                    self.run_uuid,
                    self.pid,
                    self.host,
                    now,
                    now,
                    expires,
                    LEASE_ACTIVE,
                    note,
                ),
            )
            self.conn.commit()
        except sqlite3.IntegrityError as exc:
            self.conn.rollback()
            raise LeaseError(
                f"lease already held: kind={lease_kind} key={lease_key[:16]}…"
            ) from exc
        lease_id = int(cursor.lastrowid)
        self._lease_ids.append(lease_id)
        LOGGER.info(
            "acquired %s lease id=%s key=%s owner_pid=%s expires=%s",
            lease_kind,
            lease_id,
            lease_key[:16],
            self.pid,
            expires,
        )
        return lease_id

    def acquire_run_lease(
        self, lease_key: str | None = None, *, ttl_seconds: float | None = None
    ) -> int:
        """Acquire the single-run lease for the run's semantic identity."""

        key = lease_key or self.run().semantic_run_key
        return self._insert_lease(
            LEASE_KIND_RUN,
            key,
            float(ttl_seconds if ttl_seconds is not None else self.config.lease_ttl_seconds),
        )

    def acquire_writer_lease(
        self, lease_key: str = "sqlite-writer", *, ttl_seconds: float | None = None
    ) -> int:
        """Acquire the single-writer lease guarding SQLite writes."""

        return self._insert_lease(
            LEASE_KIND_WRITER,
            lease_key,
            float(ttl_seconds if ttl_seconds is not None else self.config.lease_ttl_seconds),
        )

    def active_lease(self, lease_kind: str, lease_key: str) -> Mapping[str, Any] | None:
        return self.conn.execute(
            "SELECT * FROM execution_leases WHERE lease_kind=? AND lease_key=? "
            "AND status='ACTIVE' ORDER BY id DESC LIMIT 1",
            (str(lease_kind), str(lease_key)),
        ).fetchone()

    def release_lease(self, lease_id: int, *, status: str = LEASE_RELEASED) -> None:
        self.conn.execute(
            "UPDATE execution_leases SET status=?, released_at=? WHERE id=? AND status='ACTIVE'",
            (str(status), self.now(), int(lease_id)),
        )
        self.conn.commit()
        if lease_id in self._lease_ids:
            self._lease_ids.remove(lease_id)

    def release_all_leases(self) -> None:
        for lease_id in list(self._lease_ids):
            self.release_lease(lease_id)

    def reclaim_stale_leases(
        self,
        lease_kind: str,
        lease_key: str,
        *,
        require_dead_owner: bool = True,
        liveness_probe: Callable[[int], bool] | None = None,
    ) -> list[int]:
        """Reclaim expired leases through one explicit, auditable rule.

        A lease is reclaimable only when **both** hold:

        1. its ``expires_at`` is in the past (the TTL lapsed), and
        2. its owning PID is not alive on this host (when
           ``require_dead_owner`` is true and the lease was taken on this host).

        A live owner that simply missed a heartbeat is therefore **not**
        reclaimable, so a slow-but-working run cannot be silently duplicated.
        Every reclaim is recorded with ``status='RECLAIMED'`` and the reason.
        """

        probe = liveness_probe or OwnedProcessTree.is_alive
        now = self.now_dt()
        reclaimed: list[int] = []
        rows = self.conn.execute(
            "SELECT * FROM execution_leases WHERE lease_kind=? AND lease_key=? AND status='ACTIVE'",
            (str(lease_kind), str(lease_key)),
        ).fetchall()
        for row in rows:
            expires = parse_utc_dt(row["expires_at"])
            if expires is None or expires > now:
                continue
            reason = "ttl_expired"
            if require_dead_owner and row["owner_host"] == self.host:
                if probe(int(row["owner_pid"])):
                    LOGGER.info(
                        "lease id=%s expired but owner pid=%s is alive; not reclaiming",
                        row["id"],
                        row["owner_pid"],
                    )
                    continue
                reason = "ttl_expired_owner_dead"
            self.conn.execute(
                "UPDATE execution_leases SET status=?, note=? WHERE id=? AND status='ACTIVE'",
                (LEASE_RECLAIMED, f"reclaimed:{reason}", int(row["id"])),
            )
            reclaimed.append(int(row["id"]))
        if reclaimed:
            self.conn.commit()
        return reclaimed

    # -- stages ------------------------------------------------------------
    @contextmanager
    def stage(self, stage: str, *, discriminator: str | None = None, detail: Mapping[str, Any] | None = None):
        """Run one stage under budget and cancellation control.

        Refuses to start (recording ``SKIPPED_INSUFFICIENT_TIME``) when the
        remaining budget cannot cover the stage minimum, and refuses to start at
        all once cancellation was requested.
        """

        name = str(stage)
        key = stage_semantic_key(self.run().semantic_run_key, name, discriminator)
        try:
            self.check_cancel()
            self.ensure_can_start(name)
        except StageSkippedInsufficientTime as exc:
            self._record_stage(name, key, STAGE_SKIPPED_INSUFFICIENT_TIME, detail={"reason": str(exc)})
            raise
        except RunCancelled:
            self._record_stage(name, key, STAGE_CANCELLED, detail={"reason": "cancel_requested"})
            raise
        except DeadlineExceeded:
            self._record_stage(name, key, STAGE_FAILED, detail={"reason": "deadline_exceeded"})
            raise

        self.set_stage(name)
        self._record_stage(name, key, STAGE_RUNNING, detail=detail, start=True)
        try:
            yield self
        except RunCancelled:
            self._finish_stage(name, STAGE_CANCELLED, detail={"reason": "cancel_requested"})
            raise
        except BaseException as exc:
            self._finish_stage(name, STAGE_FAILED, detail={"error": f"{type(exc).__name__}: {exc}"})
            raise
        else:
            self._finish_stage(name, STAGE_COMPLETE)
        finally:
            self.set_stage(None)

    def _record_stage(
        self,
        stage: str,
        semantic_key: str,
        status: str,
        *,
        detail: Mapping[str, Any] | None = None,
        start: bool = False,
    ) -> int:
        now = self.now()
        cursor = self.conn.execute(
            """INSERT INTO execution_stages(
                 run_uuid, stage, semantic_key, status, started_at, finished_at,
                 detail_json, created_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (
                self.run_uuid,
                str(stage),
                semantic_key,
                status,
                now if start else None,
                None if start else now,
                json.dumps(dict(detail), sort_keys=True, default=str) if detail else None,
                now,
            ),
        )
        self.conn.commit()
        stage_id = int(cursor.lastrowid)
        self._stage_ids[str(stage)] = stage_id
        return stage_id

    def _finish_stage(
        self, stage: str, status: str, *, detail: Mapping[str, Any] | None = None
    ) -> None:
        stage_id = self._stage_ids.get(str(stage))
        now = self.now()
        if stage_id is None:
            self._record_stage(
                stage,
                stage_semantic_key(self.run().semantic_run_key, str(stage)),
                status,
                detail=detail,
            )
            return
        self.conn.execute(
            "UPDATE execution_stages SET status=?, finished_at=?, detail_json=? WHERE id=?",
            (
                str(status),
                now,
                json.dumps(dict(detail), sort_keys=True, default=str) if detail else None,
                stage_id,
            ),
        )
        self.conn.commit()

    def stages(self) -> list[Mapping[str, Any]]:
        return list(
            self.conn.execute(
                "SELECT * FROM execution_stages WHERE run_uuid=? ORDER BY id", (self.run_uuid,)
            ).fetchall()
        )

    # -- restart semantics -------------------------------------------------
    def classify_stage(
        self,
        stage: str,
        *,
        discriminator: str | None = None,
        projection_probe: Callable[[str], int | None] | None = None,
    ) -> str:
        """Classify a stage for restart: reuse, retry, block, or not started.

        Never deletes or rewrites predictive provenance — a completed unit is
        *reused*, and an interrupted writer is *blocked* for explicit review
        rather than blindly retried over a possible partial write.
        """

        name = str(stage)
        run_key = self.run().semantic_run_key
        key = stage_semantic_key(run_key, name, discriminator)

        completed = self.conn.execute(
            "SELECT run_uuid FROM execution_stages WHERE semantic_key=? AND stage=? "
            "AND status='COMPLETE' LIMIT 1",
            (key, name),
        ).fetchone()
        if completed is not None:
            return REUSE_COMPLETE

        if projection_probe is not None:
            family = STAGE_PROJECTION_FAMILY.get(name)
            if family is not None and projection_probe(family) is not None:
                return REUSE_COMPLETE

        prior = self.conn.execute(
            "SELECT status FROM execution_stages WHERE semantic_key=? AND stage=? ORDER BY id DESC LIMIT 1",
            (key, name),
        ).fetchone()
        if prior is None:
            return NOT_STARTED
        if prior["status"] == STAGE_COMPLETE:
            return REUSE_COMPLETE
        if prior["status"] == STAGE_RUNNING:
            # A writer died mid-stage: do not assume the write was atomic.
            return BLOCKED_PARTIAL_STATE
        if name in RETRY_SAFE_STAGES:
            return RETRY_SAFE
        if prior["status"] == STAGE_FAILED:
            return RETRY_SAFE
        return NOT_STARTED

    def plan_restart(
        self,
        stages: Iterable[str] = PRODUCTION_STAGES,
        *,
        discriminator: str | None = None,
        projection_probe: Callable[[str], int | None] | None = None,
    ) -> dict[str, str]:
        return {
            str(stage): self.classify_stage(
                stage, discriminator=discriminator, projection_probe=projection_probe
            )
            for stage in stages
        }

    # -- owned children ----------------------------------------------------
    def register_child(self, pid: int, argv: Iterable[str]) -> Mapping[str, Any]:
        child = self.tree.register(pid, tuple(argv))
        self.conn.execute(
            "INSERT INTO execution_children(run_uuid, pid, argv_json, registered_at) VALUES (?,?,?,?)",
            (self.run_uuid, int(pid), json.dumps(list(child.argv)), child.registered_at),
        )
        self.conn.commit()
        return child.as_dict()

    def child_audit_list(self) -> list[dict]:
        return self.tree.audit_list()

    def terminate_children(self, grace_seconds: float | None = None) -> list[dict]:
        grace = self.config.grace_seconds if grace_seconds is None else grace_seconds
        audit = self.tree.terminate_all(grace)
        for entry in audit:
            if entry.get("terminated_at"):
                self.conn.execute(
                    "UPDATE execution_children SET terminated_at=?, exit_status=? "
                    "WHERE run_uuid=? AND pid=?",
                    (entry["terminated_at"], entry.get("exit_status"), self.run_uuid, entry["pid"]),
                )
        self.conn.commit()
        self.tree.close()
        return audit

    def close(self) -> None:
        self.tree.close()

    def __enter__(self) -> "ExecutionController":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


@contextmanager
def production_run_guard(
    conn: sqlite3.Connection,
    *,
    planning_event: int,
    planning_cutoff: str | None = None,
    label: str = "production",
    families: Iterable[str] | None = None,
    hard_stop_at: datetime | str | None = None,
    max_wall_seconds: float = 6 * 3600,
    official_deadline: str | None = None,
    allow_late: bool = False,
    wall_clock: Callable[[], datetime] | None = None,
    config: ExecutionConfig | None = None,
):
    """The single entry point every production writer must go through.

    Acquires the run lease **and** the SQLite writer lease for the semantic
    identity, derives the hard stop, registers for cancellation, and releases
    everything on the way out.  A second invocation for the same semantic work —
    including one launched by a helper script that spawns a duplicate top-level
    runner — fails here with ``LeaseError`` instead of colliding in SQLite.

    The hard stop is ``min(official_deadline, now + max_wall_seconds)`` unless
    ``allow_late`` is set, in which case only the wall-clock ceiling applies.
    That makes the official FPL deadline an executable boundary rather than
    advisory text.
    """

    settings = config or ExecutionConfig()
    clock = wall_clock or utc_now_dt
    controller = ExecutionController(conn, config=settings, wall_clock=clock)

    now_dt = controller.now_dt()
    # Preflight: the planning cutoff may not post-date this execution.
    assert_causal_cutoff(planning_cutoff, now_dt, label=f"{label} guard")
    candidates = [add_seconds(now_dt, float(max_wall_seconds))]
    if official_deadline and not allow_late:
        parsed = parse_utc_dt(official_deadline)
        if parsed is not None:
            candidates.append(parsed)
    hard_stop = min(candidates)
    if hard_stop_at is not None:
        parsed = (
            hard_stop_at
            if isinstance(hard_stop_at, datetime)
            else parse_utc_dt(hard_stop_at)
        )
        if parsed is None:
            raise ValueError(f"hard_stop_at is not a valid UTC timestamp: {hard_stop_at!r}")
        hard_stop = min(hard_stop, parsed)

    controller.create_run(
        planning_event=int(planning_event),
        planning_cutoff=planning_cutoff,
        hard_stop_at=hard_stop,
        label=label,
        families=sorted(str(f) for f in (families or ())),
    )
    controller.start()
    acquired: list[int] = []
    try:
        acquired.append(controller.acquire_run_lease())
        acquired.append(controller.acquire_writer_lease())
        LOGGER.info(
            "production guard held run=%s event=%s hard_stop=%s",
            controller.run_uuid,
            planning_event,
            controller.run().hard_stop_at,
        )
        yield controller
    except RunCancelled:
        if controller.run().status == RUN_CANCEL_REQUESTED:
            controller.acknowledge_cancel()
        raise
    except BaseException as exc:
        if controller.run().status not in TERMINAL_RUN_STATUSES:
            controller.finish(RUN_FAILED, f"{type(exc).__name__}: {exc}")
        raise
    else:
        if controller.run().status not in TERMINAL_RUN_STATUSES:
            controller.finish(RUN_COMPLETE)
    finally:
        controller.terminate_children()
        controller.close()


@contextmanager
def event_run_guard(
    conn: sqlite3.Connection,
    *,
    planning_event: int,
    cutoff: str | None,
    label: str = "production",
    families: Iterable[str] | None = None,
    max_wall_seconds: float = 6 * 3600,
    wall_clock: Callable[[], datetime] | None = None,
    config: ExecutionConfig | None = None,
):
    """``production_run_guard`` bound to an event's official deadline.

    A cutoff at or after the deadline is a deliberate late reproduction, so the
    deadline is not imposed as the hard stop in that case; otherwise the earlier
    of the deadline and the wall-clock ceiling wins.  This keeps the official
    deadline an executable boundary without outlawing authorised late work.
    """

    clock = wall_clock or utc_now_dt
    row = conn.execute(
        "SELECT deadline_time FROM events WHERE id=?", (int(planning_event),)
    ).fetchone()
    official_deadline = None
    if row is not None:
        value = row["deadline_time"] if isinstance(row, sqlite3.Row) else row[0]
        if value:
            official_deadline = str(value)

    cutoff_text = str(cutoff) if cutoff else format_utc(clock())
    allow_late = False
    if official_deadline:
        cutoff_dt = parse_utc_dt(cutoff_text)
        deadline_dt = parse_utc_dt(official_deadline)
        if cutoff_dt is not None and deadline_dt is not None:
            allow_late = cutoff_dt > deadline_dt

    with production_run_guard(
        conn,
        planning_event=int(planning_event),
        planning_cutoff=cutoff_text,
        label=label,
        families=families,
        official_deadline=official_deadline,
        allow_late=allow_late,
        max_wall_seconds=max_wall_seconds,
        wall_clock=wall_clock,
        config=config,
    ) as controller:
        yield controller


__all__ = [
    "BASELINE",
    "BLOCKED_PARTIAL_STATE",
    "DECISION_BOARD",
    "DeadlineExceeded",
    "ExecutionConfig",
    "ExecutionController",
    "ExecutionError",
    "LEASE_ACTIVE",
    "LEASE_EXPIRED",
    "LEASE_KIND_RUN",
    "LEASE_KIND_WRITER",
    "LEASE_RECLAIMED",
    "LEASE_RELEASED",
    "LeaseError",
    "MINUTES",
    "MONTE_CARLO",
    "NOT_STARTED",
    "OFFICIAL_FETCH",
    "PERSIST",
    "PLAYER_RATES",
    "PREFLIGHT",
    "PRODUCTION_STAGES",
    "RETRY_SAFE",
    "RETRY_SAFE_STAGES",
    "REUSE_COMPLETE",
    "ROUTE_SEARCH",
    "RUN_CANCELLED",
    "RUN_CANCEL_REQUESTED",
    "RUN_COMPLETE",
    "RUN_CREATED",
    "RUN_FAILED",
    "RUN_RUNNING",
    "RUN_STATUSES",
    "RunCancelled",
    "RunIdentity",
    "STAGE_COMPLETE",
    "STAGE_FAILED",
    "STAGE_MINIMUM_SECONDS",
    "STAGE_NOT_STARTED",
    "STAGE_RUNNING",
    "STAGE_SKIPPED_INSUFFICIENT_TIME",
    "STAGE_PROJECTION_FAMILY",
    "StageSkippedInsufficientTime",
    "TEAM_STRENGTH",
    "TERMINAL_RUN_STATUSES",
    "XPTS",
    "find_completed_projection_run",
    "format_utc",
    "event_run_guard",
    "parse_utc_dt",
    "production_run_guard",
    "semantic_run_key",
    "stage_semantic_key",
    "utc_now_dt",
]
