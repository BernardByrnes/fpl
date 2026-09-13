"""Immutable execution-scoped source-database snapshot (R4A.1).

R4A produced a *drift-detecting causal manifest*: a content hash over the rows a
certification would read, recomputed from live tables.  That detects a later
ingestion but does not prevent it, so it was NOT equivalent to "later ingestion
cannot change an in-progress certification".

R4A.1 supersedes it with a **physically captured, immutable database snapshot**:

1. the controller starts and the causal cutoff is established;
2. a real SQLite copy of the source database is taken at that instant;
3. the snapshot is opened READ-ONLY by every predictive stage;
4. all four events read the same snapshot identity;
5. ingestion continues against the live database, untouched and unaffected.

The snapshot identity is the sha256 of the snapshot FILE, so it is an identity of
an actual captured database state -- not a hash recomputed from live tables.

R4A.3 makes the captured state EXACT rather than bounded.  The source database is
held under SQLite writer exclusion (``BEGIN IMMEDIATE``) for the whole capture, so
no commit can occur between lock acquisition and release and the data state is
provably constant throughout the copy.  ``snapshot_consistency_at`` is therefore
the LOCK ACQUISITION instant, not the file-completion instant -- the file may
finish much later without changing what state was captured.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .causality import CausalityError, _aware, assert_causal_cutoff
from .utils import utc_now

DIAG_SNAPSHOT_MISSING = "CERTIFICATION_SNAPSHOT_MISSING"
DIAG_HISTORICAL_SNAPSHOT_REQUIRED = "HISTORICAL_SNAPSHOT_REQUIRED"
DIAG_SOURCE_LOCK_FAILED = "CERTIFICATION_SOURCE_LOCK_FAILED"
DIAG_SNAPSHOT_MUTATED = "CERTIFICATION_SNAPSHOT_MUTATED"
DIAG_SNAPSHOT_AFTER_CUTOFF = "CERTIFICATION_SNAPSHOT_AFTER_CUTOFF"
DIAG_LIVE_SOURCE_DRIFT = "LIVE_SOURCE_DRIFT_DETECTED"

SNAPSHOT_FILENAME = "execution_source_snapshot.db"
SNAPSHOT_MANIFEST = "execution_source_snapshot.json"


class SnapshotError(CausalityError):
    """The execution snapshot is missing, unreadable, or no longer immutable."""


def _fmt(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def file_sha256(path: str | Path, *, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def source_db_identity(path: str | Path) -> dict[str, Any]:
    """Identity of the live source database at capture time.

    ``schema_version`` and the ``projection_runs`` high-water mark describe the
    state a snapshot was taken from without hashing the whole live file (which
    would be large and would change under concurrent ingestion).
    """

    conn = sqlite3.connect(f"file:{Path(path)}?mode=ro", uri=True)
    identity: dict[str, Any] = {"path": str(path)}
    try:
        try:
            schema_version = conn.execute(
                "SELECT value FROM schema_meta WHERE key='schema_version'"
            ).fetchone()
            identity["schema_version"] = str(schema_version[0]) if schema_version else None
        except sqlite3.Error:
            identity["schema_version"] = None
        try:
            runs = conn.execute("SELECT COUNT(*), MAX(id) FROM projection_runs").fetchone()
            identity["projection_runs_count"] = int(runs[0])
            identity["projection_runs_max_id"] = int(runs[1]) if runs[1] is not None else None
        except sqlite3.Error:
            identity["projection_runs_count"] = None
            identity["projection_runs_max_id"] = None
        identity["page_count"] = int(conn.execute("PRAGMA page_count").fetchone()[0])
        return identity
    finally:
        conn.close()


@dataclass(frozen=True)
class ExecutionSnapshot:
    """One physically captured, immutable source database."""

    path: str
    data_snapshot_sha256: str
    # THREE distinct instants -- "created_at" is retained as the causal
    # consistency instant for backwards compatibility, but the three fields below
    # are the authoritative record.
    snapshot_lock_acquired_at: str
    snapshot_capture_started_at: str
    snapshot_consistency_at: str
    snapshot_capture_completed_at: str
    snapshot_capture_seconds: float
    source_db_identity: dict[str, Any]
    execution_run_uuid: str | None
    planning_cutoff: str | None
    size_bytes: int
    manifest_path: str | None = None

    @property
    def created_at(self) -> str:
        """Backwards-compatible alias for the causal consistency instant."""

        return self.snapshot_consistency_at

    def as_dict(self) -> dict[str, Any]:
        return {
            "data_snapshot_path": self.path,
            "data_snapshot_sha256": self.data_snapshot_sha256,
            "data_snapshot_created_at": self.snapshot_consistency_at,
            "snapshot_lock_acquired_at": self.snapshot_lock_acquired_at,
            "snapshot_capture_started_at": self.snapshot_capture_started_at,
            "snapshot_consistency_at": self.snapshot_consistency_at,
            "snapshot_capture_completed_at": self.snapshot_capture_completed_at,
            "snapshot_capture_seconds": self.snapshot_capture_seconds,
            "data_snapshot_source_db_identity": self.source_db_identity,
            "execution_run_uuid": self.execution_run_uuid,
            "planning_cutoff": self.planning_cutoff,
            "data_snapshot_size_bytes": self.size_bytes,
            "data_snapshot_manifest": self.manifest_path,
        }


def capture_execution_snapshot(
    source_db: str | Path,
    *,
    directory: str | Path,
    execution_run_uuid: str | None = None,
    planning_cutoff: str | None = None,
    clock=None,
    timeout_seconds: float = 30.0,
) -> ExecutionSnapshot:
    """Physically capture the source database and record its identity.

    Uses SQLite ``VACUUM INTO``, which produces a consistent single-file copy of
    the database as of the transaction that runs it -- one actual captured state,
    not a recomputed hash.
    """

    _clock = clock or (lambda: datetime.now(timezone.utc))

    # The snapshot may not predate the planning cutoff it is used for.  The check is
    # against the instant the caller invokes capture, which is the earliest the
    # locked state can be established.
    if planning_cutoff is not None:
        assert_causal_cutoff(planning_cutoff, _clock(), label="execution snapshot capture")

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    snapshot_path = directory / SNAPSHOT_FILENAME
    if snapshot_path.exists():
        snapshot_path.unlink()

    source = str(source_db)
    if source == ":memory:":
        raise SnapshotError("cannot snapshot an in-memory source database")

    # --- writer exclusion: make the source state EXACTLY fixed ----------------
    # BEGIN IMMEDIATE takes SQLite's RESERVED write lock, so no other connection
    # can commit for as long as we hold it.  A separate read connection can still
    # run VACUUM INTO (verified experimentally: E2/E3), so the copy is taken from a
    # database state that provably cannot change.  Because the state is constant
    # across the whole capture, any internal SQLite read-snapshot instant
    # represents exactly the same data state as lock acquisition.
    try:
        locker = sqlite3.connect(source, timeout=timeout_seconds)
    except sqlite3.Error as exc:
        raise SnapshotError(
            f"{DIAG_SOURCE_LOCK_FAILED}: could not open the source database {source} to acquire "
            f"writer exclusion: {exc}"
        ) from exc
    try:
        locker.execute(f"PRAGMA busy_timeout={int(timeout_seconds * 1000)}")
    except sqlite3.Error:
        pass
    try:
        locker.execute("BEGIN IMMEDIATE")
    except sqlite3.Error as exc:
        locker.close()
        raise SnapshotError(
            f"{DIAG_SOURCE_LOCK_FAILED}: could not acquire SQLite writer exclusion on {source}: {exc}"
        ) from exc
    lock_acquired = _clock()
    lock_acquired_at = _fmt(lock_acquired)
    capture_started = lock_acquired

    try:
        reader = sqlite3.connect(source)
        try:
            reader.execute("PRAGMA busy_timeout=30000")
            reader.execute("VACUUM INTO ?", (str(snapshot_path),))
        finally:
            reader.close()
    finally:
        # The exclusion is held only long enough to capture, never across generation.
        try:
            locker.execute("ROLLBACK")
        finally:
            locker.close()

    capture_completed = _clock()
    completed_at = _fmt(capture_completed)
    capture_seconds = (capture_completed - capture_started).total_seconds()
    identity = source_db_identity(source)
    digest = file_sha256(snapshot_path)
    manifest_path = directory / SNAPSHOT_MANIFEST
    snapshot = ExecutionSnapshot(
        path=str(snapshot_path),
        data_snapshot_sha256=digest,
        # EXACT CONSISTENCY INSTANT.
        #
        # The source is under SQLite writer exclusion (BEGIN IMMEDIATE) for the whole
        # capture, so no commit can occur between lock acquisition and release: the
        # database state is CONSTANT throughout the copy.  Any internal read-snapshot
        # instant therefore represents exactly the same data state as lock
        # acquisition, and the causal instant is the LOCK ACQUISITION time -- not the
        # file-completion time.
        #
        # Experimentally verified (see data/exports/reliability/exp_r4a3_sqlite_lock.py):
        # E1 shows an unprotected VACUUM INTO excludes a commit that lands during the
        # copy, so calling the COMPLETION time the data instant would over-claim;
        # E2/E3 show BEGIN IMMEDIATE blocks competing writers for the whole capture
        # while a separate read connection still runs VACUUM INTO successfully.
        snapshot_lock_acquired_at=lock_acquired_at,
        snapshot_capture_started_at=lock_acquired_at,
        snapshot_consistency_at=lock_acquired_at,
        snapshot_capture_completed_at=completed_at,
        snapshot_capture_seconds=capture_seconds,
        source_db_identity=identity,
        execution_run_uuid=execution_run_uuid,
        planning_cutoff=str(planning_cutoff) if planning_cutoff else None,
        size_bytes=int(os.path.getsize(snapshot_path)),
        manifest_path=str(manifest_path),
    )
    manifest_path.write_text(
        json.dumps(snapshot.as_dict(), indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    return snapshot


def open_snapshot(snapshot: ExecutionSnapshot) -> sqlite3.Connection:
    """Open the snapshot READ-ONLY, refusing anything that is not immutable.

    Writing to the snapshot is what would break "later ingestion cannot change an
    in-progress certification", so the connection is opened ``mode=ro`` with
    ``query_only=ON`` and its file hash is verified on every open.
    """

    path = Path(snapshot.path)
    if not path.exists():
        raise SnapshotError(f"{DIAG_SNAPSHOT_MISSING}: snapshot {path} does not exist")
    current = file_sha256(path)
    if current != snapshot.data_snapshot_sha256:
        raise SnapshotError(
            f"{DIAG_SNAPSHOT_MUTATED}: snapshot {path} hash {current} != recorded "
            f"{snapshot.data_snapshot_sha256}"
        )
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    return conn


def load_snapshot_manifest(directory: str | Path) -> ExecutionSnapshot:
    """Load a previously captured snapshot manifest, failing closed if absent."""

    manifest = Path(directory) / SNAPSHOT_MANIFEST
    if not manifest.exists():
        raise SnapshotError(
            f"{DIAG_SNAPSHOT_MISSING}: no certification snapshot manifest at {manifest}"
        )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    return ExecutionSnapshot(
        path=payload["data_snapshot_path"],
        data_snapshot_sha256=payload["data_snapshot_sha256"],
        snapshot_lock_acquired_at=payload.get("snapshot_lock_acquired_at")
        or payload.get("snapshot_capture_started_at")
        or payload["data_snapshot_created_at"],
        snapshot_capture_started_at=payload.get("snapshot_capture_started_at")
        or payload["data_snapshot_created_at"],
        snapshot_consistency_at=payload.get("snapshot_consistency_at")
        or payload["data_snapshot_created_at"],
        snapshot_capture_completed_at=payload.get("snapshot_capture_completed_at")
        or payload["data_snapshot_created_at"],
        snapshot_capture_seconds=float(payload.get("snapshot_capture_seconds") or 0.0),
        source_db_identity=payload.get("data_snapshot_source_db_identity") or {},
        execution_run_uuid=payload.get("execution_run_uuid"),
        planning_cutoff=payload.get("planning_cutoff"),
        size_bytes=int(payload.get("data_snapshot_size_bytes") or 0),
        manifest_path=str(manifest),
    )


def assert_snapshot_unchanged(snapshot: ExecutionSnapshot) -> None:
    """Hard fail if the snapshot itself changed.  Live drift is NOT checked here."""

    current = file_sha256(snapshot.path)
    if current != snapshot.data_snapshot_sha256:
        raise SnapshotError(
            f"{DIAG_SNAPSHOT_MUTATED}: snapshot hash changed from "
            f"{snapshot.data_snapshot_sha256} to {current}"
        )


def live_source_drift(
    snapshot: ExecutionSnapshot, *, live_db: str | Path | None = None
) -> dict[str, Any]:
    """Informational: how far the live database has moved since capture.

    This must NOT contaminate an active certification -- the certification reads
    the snapshot.  It is reported so an operator can see that ingestion continued.
    """

    path = live_db or snapshot.source_db_identity.get("path")
    if not path or not Path(path).exists():
        return {"available": False}
    current = source_db_identity(path)
    before = snapshot.source_db_identity
    return {
        "available": True,
        "informational": True,
        "diagnostic": None
        if current.get("projection_runs_max_id") == before.get("projection_runs_max_id")
        and current.get("schema_version") == before.get("schema_version")
        else DIAG_LIVE_SOURCE_DRIFT,
        "at_capture": before,
        "now": current,
        "note": (
            "live-source drift is informational only: the certification reads the immutable "
            "snapshot, so later ingestion cannot change its prediction inputs"
        ),
    }


# Mechanical tolerance for "the requested cutoff equals the snapshot consistency
# instant".  The cutoff is normally MINTED from the consistency instant, so this
# only absorbs timestamp formatting/rounding, never a real time difference.
CUTOFF_MATCH_TOLERANCE_SECONDS = 2.0


def require_live_cutoff_matches_snapshot(
    snapshot_consistency_at: str,
    requested_cutoff: str | None,
    *,
    tolerance_seconds: float = CUTOFF_MATCH_TOLERANCE_SECONDS,
) -> str:
    """The LIVE cutoff rule: ``planning_cutoff == snapshot_consistency_at``.

    A caller must not request an older cutoff and have the system silently capture
    a fresh live snapshot: with only 2 of 17 acquisition rows timestamped, the
    ledger cannot reconstruct an earlier ownership state, so a stale-but-close
    cutoff against a fresh snapshot would misrepresent a same-Gameweek transfer.

    A materially older cutoff therefore fails closed with
    ``HISTORICAL_SNAPSHOT_REQUIRED``, naming the only acceptable remedy: an
    already-existing immutable snapshot captured for that cutoff.
    """

    consistency = _aware(snapshot_consistency_at, label="snapshot consistency")
    if requested_cutoff is None:
        return snapshot_consistency_at
    requested = _aware(requested_cutoff, label="requested cutoff")
    delta = (consistency - requested).total_seconds()
    if abs(delta) <= float(tolerance_seconds):
        return snapshot_consistency_at
    if delta > 0:
        raise SnapshotError(
            f"{DIAG_HISTORICAL_SNAPSHOT_REQUIRED}: requested cutoff {requested_cutoff} predates the "
            f"captured snapshot consistency instant {snapshot_consistency_at} by {delta:.1f}s. Live "
            "certification cannot represent an earlier instant from a freshly captured snapshot; "
            "supply an existing immutable snapshot captured for that cutoff."
        )
    raise SnapshotError(
        f"{DIAG_HISTORICAL_SNAPSHOT_REQUIRED}: requested cutoff {requested_cutoff} post-dates the "
        f"snapshot consistency instant {snapshot_consistency_at} by {-delta:.1f}s. A snapshot cannot "
        "represent a later instant than the one it was captured at."
    )


def snapshot_consistency_window(snapshot: ExecutionSnapshot) -> float:
    """Seconds between capture start and capture completion (the read window)."""

    started = _aware(snapshot.snapshot_capture_started_at, label="capture started")
    completed = _aware(snapshot.snapshot_capture_completed_at, label="capture completed")
    return (completed - started).total_seconds()


def snapshot_for_cutoff(created_at: str, planning_cutoff: str) -> None:
    """Enforce ``planning_cutoff <= snapshot_created_at`` (the production rule).

    The cutoff may equal the capture instant or immediately precede it, but a
    cutoff minted AFTER the capture would claim knowledge the snapshot cannot
    contain.
    """

    assert_causal_cutoff(planning_cutoff, _aware(created_at, label="snapshot created_at"), label="snapshot")


__all__ = [
    "CUTOFF_MATCH_TOLERANCE_SECONDS",
    "DIAG_SOURCE_LOCK_FAILED",
    "DIAG_HISTORICAL_SNAPSHOT_REQUIRED",
    "DIAG_LIVE_SOURCE_DRIFT",
    "DIAG_SNAPSHOT_AFTER_CUTOFF",
    "DIAG_SNAPSHOT_MISSING",
    "DIAG_SNAPSHOT_MUTATED",
    "ExecutionSnapshot",
    "SNAPSHOT_FILENAME",
    "SNAPSHOT_MANIFEST",
    "SnapshotError",
    "assert_snapshot_unchanged",
    "capture_execution_snapshot",
    "file_sha256",
    "live_source_drift",
    "load_snapshot_manifest",
    "open_snapshot",
    "require_live_cutoff_matches_snapshot",
    "snapshot_consistency_window",
    "snapshot_for_cutoff",
    "source_db_identity",
]
