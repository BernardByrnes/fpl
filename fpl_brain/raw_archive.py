"""Immutable archive of the exact upstream bytes the repository ingested.

WHY THIS EXISTS
---------------
The fetch layer persists each response into ``raw_dir/<fetch_run_id>/``.  That
is convenient and, by accident of run-scoping, a later fetch rarely clobbers an
earlier one -- but nothing *enforces* it.  There is no hash of the received
bytes, no observation time in the file's identity, no atomic write, and no
refusal when an identity is reused with different bytes.  A payload can
therefore be lost or silently replaced and nobody could prove it afterwards,
which is exactly what point-in-time modelling needs to be able to do.

WHAT AN ARCHIVE RECORD IS
-------------------------
One record is one OBSERVATION of one source at one instant:

    capture_id       source @ observed_at # payload_sha256
    source           the endpoint slug the fetch layer used
    observed_at      the UTC instant the capture was taken
    payload_sha256   over the exact bytes archived, nothing re-serialised
    relative_path    the archived blob, relative to the archive root
    byte_count       the length of those bytes
    event            the Gameweek, for event-scoped sources
    run_id           the fetch run that produced it

An identical payload fetched at two different instants is TWO observations:
availability being unchanged is itself evidence, so the observation time is part
of the blob's path and identical bytes at different times are never collapsed.
No content-addressed store is built: the path carries time *and* hash, which is
enough to be unambiguous without a deduplication layer.

ATOMICITY AND FAIL-CLOSED COLLISION
-----------------------------------
The blob is written to a temporary file in its destination directory, flushed,
``fsync``-ed and then ``os.replace``-d into place, so a crash can never leave a
truncated file that looks like a capture.  The manifest is published the same
way -- the complete new manifest is written to a temp file and replaced over
the live ``manifest.jsonl`` -- so a crash can never leave a truncated live
record either.  Identity read, collision decision, blob publication and
manifest publication all happen under one cross-process publication lock, so
overlapping publishers cannot both observe "identity absent" and publish
contradictory bytes.  If a crash separates blob publication from manifest
publication, the orphan blob is verified and recovered on retry, never
deleted; a manifest record whose blob is missing or corrupted fails closed
rather than reporting an observation with no intact bytes.  If an identity is
presented again:

* with the SAME bytes, the observation is already recorded -- nothing is
  rewritten and the existing record is returned;
* with DIFFERENT bytes, that is a contradiction and raises.  Archive bytes are
  never silently overwritten.

WHAT IS NOT HERE
----------------
This module preserves evidence; it does not read it back.  No prediction path
consumes the archive in this phase -- the existing "latest" raw file and the
database remain the production read path.  The archive exists so a future
reader can reconstruct what was actually visible at an instant.

HISTORICAL GAP
--------------
The archive begins honestly at the first capture made under this contract.
Nothing was backfilled and no historical ``observed_at`` was invented for data
downloaded later: point-in-time captures that were never taken cannot be
recovered, and pretending otherwise would be worse than the gap.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from .utils import ensure_directory

try:
    import fcntl as _fcntl
except ImportError:  # Windows has no fcntl; msvcrt is used instead.
    _fcntl = None
try:
    import msvcrt as _msvcrt
except ImportError:  # POSIX has no msvcrt; fcntl is used instead.
    _msvcrt = None

RAW_ARCHIVE_VERSION = "raw_archive_v1.0.0"

#: Directory under the configured raw root that holds the archive and its
#: manifest.  The ``raw_dir/<fetch_run_id>/`` tree is left exactly as it was;
#: those files are an operational convenience, the archive is the authority.
ARCHIVE_DIRNAME = "archive"
MANIFEST_FILENAME = "manifest.jsonl"

#: Every endpoint the client fetches is persisted into the database as evidence
#: (bootstrap -> players/snapshots, fixtures -> fixtures, element-summary and
#: event-live -> player_gameweeks, entry/picks -> manager state, event-status ->
#: events).  There is no fetched source that is not historical evidence, which
#: is why the archive is wired at the single successful-response site rather
#: than per endpoint: a fetch that is not evidence does not exist.
ARCHIVE_REQUIRED_SOURCES = (
    "bootstrap_static",
    "fixtures",
    "element_summary",
    "event_live",
    "entry",
    "entry_history",
    "entry_transfers",
    "entry_picks",
    "event_status",
)

#: Sources whose identity includes the Gameweek they describe.
EVENT_SCOPED_SOURCES = ("fixtures", "event_live", "entry_picks")

#: Lock file (inside the archive root) guarding the publication critical
#: section: identity read, collision decision, blob recovery/publication and
#: manifest publication.  Held only for that metadata step, never for network
#: fetching.
ARCHIVE_LOCK_FILENAME = ".manifest.lock"

#: How long a publisher waits for the publication lock before failing closed.
ARCHIVE_LOCK_TIMEOUT_SECONDS = 30.0

# Serializes the threads of THIS process.  OS file locks are process-scoped on
# some platforms, so threads need their own guard in addition to the file lock.
_archive_thread_lock = threading.Lock()


class RawArchiveError(RuntimeError):
    """The archive contract was violated."""


class RawArchiveCollision(RawArchiveError):
    """One archive identity was presented with two different payloads."""


class RawArchiveBusy(RawArchiveError):
    """The archive publication lock could not be acquired in time."""


def _try_file_lock(fd: int) -> bool:
    """One non-blocking attempt at an exclusive cross-process file lock."""

    if os.name == "nt" and _msvcrt is not None:
        try:
            _msvcrt.locking(fd, _msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False
    if _fcntl is not None:
        try:
            _fcntl.flock(fd, _fcntl.LOCK_EX | _fcntl.LOCK_NB)
            return True
        except OSError:
            return False
    # No OS primitive available: threads remain serialized by the module lock;
    # processes are best-effort.  Every supported platform provides one of the
    # branches above, so this is a fallback, not the contract.
    return True


def _unlock_file(fd: int) -> None:
    """Release the lock from ``_try_file_lock``; never raises."""

    try:
        if os.name == "nt" and _msvcrt is not None:
            _msvcrt.locking(fd, _msvcrt.LK_UNLCK, 1)
        elif _fcntl is not None:
            _fcntl.flock(fd, _fcntl.LOCK_UN)
    except OSError:
        pass


@contextlib.contextmanager
def _publication_lock(root: Path) -> Iterator[None]:
    """The ONE process-safe critical section for archive publication.

    Identity read, collision decision, blob recovery/publication and manifest
    publication all happen under this lock, so two overlapping publishers --
    threads or processes -- cannot both observe "identity absent" and then
    publish contradictory bytes for it.  Same identity + same payload stays
    idempotent; same identity + different payload lets exactly one publisher
    establish the identity while the other fails closed.

    The OS lock is released automatically if the holder crashes, so a crash
    can never wedge the archive.  The lock file itself is left in place
    deliberately: removing it would race with a publisher opening it.
    """

    ensure_directory(root)
    lock_path = Path(root) / ARCHIVE_LOCK_FILENAME
    deadline = time.monotonic() + ARCHIVE_LOCK_TIMEOUT_SECONDS
    with _archive_thread_lock:
        fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o644)
        try:
            while not _try_file_lock(fd):
                if time.monotonic() >= deadline:
                    raise RawArchiveBusy(
                        f"could not acquire the archive publication lock at {lock_path} "
                        f"within {ARCHIVE_LOCK_TIMEOUT_SECONDS}s; refusing to publish outside it"
                    )
                time.sleep(0.01)
            try:
                yield
            finally:
                _unlock_file(fd)
        finally:
            os.close(fd)


def archive_root(raw_dir: str | Path) -> Path:
    return Path(raw_dir) / ARCHIVE_DIRNAME


def manifest_path(root: str | Path) -> Path:
    return Path(root) / MANIFEST_FILENAME


def sha256_of(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _compact(observed_at: str) -> str:
    return "".join(char for char in str(observed_at) if char.isalnum())


def capture_id(source: str, observed_at: str, payload_sha256: str) -> str:
    return f"{source}@{observed_at}#{payload_sha256[:12]}"


def relative_blob_path(source: str, observed_at: str, payload_sha256: str, event: int | None) -> str:
    """``<source>/[event_<n>/]<observed_at>_<sha12>.json``.

    The observation time is in the name, so identical bytes at two instants are
    two files; the hash is in the name, so two different payloads at one instant
    cannot share a path.  Together they make an ambiguous identity unrepresentable
    rather than merely unlikely.
    """

    parts = [str(source)]
    if event is not None:
        parts.append(f"event_{int(event)}")
    parts.append(f"{_compact(observed_at)}_{payload_sha256[:12]}.json")
    return "/".join(parts)


@dataclass(frozen=True)
class RawCapture:
    capture_id: str
    source: str
    observed_at: str
    payload_sha256: str
    relative_path: str
    byte_count: int
    event: int | None = None
    run_id: str | None = None
    archive_version: str = RAW_ARCHIVE_VERSION

    def as_dict(self) -> dict[str, Any]:
        return {
            "capture_id": self.capture_id,
            "source": self.source,
            "observed_at": str(self.observed_at),
            "payload_sha256": self.payload_sha256,
            "relative_path": self.relative_path,
            "byte_count": int(self.byte_count),
            "event": None if self.event is None else int(self.event),
            "run_id": None if self.run_id is None else str(self.run_id),
            "archive_version": self.archive_version,
        }


def _write_atomically(target: Path, body: bytes) -> None:
    """temp file -> flush -> fsync -> os.replace, in the destination directory."""

    ensure_directory(target.parent)
    temp = target.with_name(f".{target.name}.tmp-{os.getpid()}-{os.urandom(4).hex()}")
    try:
        with open(temp, "wb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, target)
    except BaseException:
        # A failed write must leave no candidate archive entry behind.
        try:
            temp.unlink()
        except OSError:
            pass
        raise


def _append_manifest(root: Path, record: RawCapture) -> None:
    """Publish one manifest line without ever exposing a partial live manifest.

    The format stays JSONL and stays logically append-only, but physically the
    complete new manifest (existing bytes plus the new line) is written to a
    temp file, flushed, fsync-ed and ``os.replace``-d over the live file.  An
    interrupted write can therefore only leave a temp file behind -- never a
    truncated live record -- and concurrent publishers cannot interleave lines,
    because every publication happens under :func:`_publication_lock`.
    """

    ensure_directory(root)
    path = manifest_path(root)
    line = json.dumps(record.as_dict(), sort_keys=True, separators=(",", ":"))
    existing = path.read_bytes() if path.exists() else b""
    if existing and not existing.endswith(b"\n"):
        existing += b"\n"
    _write_atomically(path, existing + (line + "\n").encode("utf-8"))


def _require_recorded_blob(root: Path, item: Mapping[str, Any]) -> None:
    """Fail closed unless the manifest record's blob exists and matches its digest.

    An idempotent return is only valid for an observation whose bytes are
    actually present and intact.  A record pointing at a missing or corrupted
    blob reports success for evidence nobody can read back, so it raises
    instead of returning.
    """

    blob = root / str(item["relative_path"])
    digest = str(item["payload_sha256"])
    if not blob.exists():
        raise RawArchiveError(
            f"archive manifest records {item.get('capture_id')} at {item.get('relative_path')} "
            f"with digest {digest[:12]} but the blob is missing; refusing to report "
            "an observation with no bytes behind it"
        )
    if sha256_of(blob.read_bytes()) != digest:
        raise RawArchiveError(
            f"archive blob {item.get('relative_path')} no longer hashes to its recorded "
            f"digest {digest[:12]}; refusing to report corrupted evidence as intact"
        )


def archive_raw_capture(
    raw_dir: str | Path,
    *,
    source: str,
    observed_at: str,
    body: bytes,
    event: int | None = None,
    run_id: int | str | None = None,
) -> RawCapture:
    """Preserve ``body`` as one observation of ``source`` at ``observed_at``.

    ``body`` must be the exact bytes received.  They are hashed and stored as
    given: nothing is re-serialised, re-encoded or pretty-printed first, so the
    digest always describes the archived file itself.

    Endpoint validation happens BEFORE this call (the fetch layer runs the
    endpoint's own full validation contract first), so anything reaching here
    has already been accepted as a valid capture.

    Publication is crash-safe and cross-process serialized under
    :func:`_publication_lock`, covering identity read, collision decision, blob
    recovery/publication and manifest publication as one critical section:

    * a blob left without its manifest record by an interrupted publication is
      verified and recovered on retry -- never silently dropped, never
      duplicated, and never deleted merely because metadata publication failed;
    * an idempotent return is only made after verifying the recorded blob
      exists and still hashes to its recorded digest, otherwise it fails
      closed.
    """

    if not str(source).strip():
        raise RawArchiveError("an archive capture requires a source identity")
    if not str(observed_at).strip():
        raise RawArchiveError("an archive capture requires an explicit observed_at")
    if not isinstance(body, (bytes, bytearray)):
        raise RawArchiveError("archive bodies must be bytes, not decoded text")
    payload = bytes(body)

    digest = sha256_of(payload)
    relative = relative_blob_path(str(source), str(observed_at), digest, event)
    root = archive_root(raw_dir)
    target = root / relative
    record = RawCapture(
        capture_id=capture_id(str(source), str(observed_at), digest),
        source=str(source),
        observed_at=str(observed_at),
        payload_sha256=digest,
        relative_path=relative,
        byte_count=len(payload),
        event=None if event is None else int(event),
        run_id=None if run_id is None else str(run_id),
    )

    with _publication_lock(root):
        # The manifest is the identity register.  A blob path already encodes the
        # digest, so a path clash alone cannot detect a contradiction; the recorded
        # (source, observed_at, event) triple is what must not be reused with
        # different content.
        same_identity = [
            item for item in load_manifest(root)
            if str(item.get("source")) == record.source
            and str(item.get("observed_at")) == record.observed_at
            and (None if item.get("event") is None else int(item["event"]))
            == (None if record.event is None else int(record.event))
        ]
        if same_identity:
            if all(str(item.get("payload_sha256")) == digest for item in same_identity):
                # The same observation offered twice: already recorded, not
                # rewritten -- but only after proving the bytes are still there.
                for item in same_identity:
                    _require_recorded_blob(root, item)
                return record
            raise RawArchiveCollision(
                f"{record.source} at {record.observed_at} is already archived with different bytes "
                f"({str(same_identity[0].get('payload_sha256'))[:12]} on record vs {digest[:12]} presented); "
                "refusing to overwrite an archived observation"
            )

        if target.exists():
            existing = sha256_of(target.read_bytes())
            if existing != digest:
                raise RawArchiveCollision(
                    f"archive path {relative} already holds different bytes "
                    f"({existing[:12]} on disk vs {digest[:12]} presented); refusing to overwrite"
                )
            # A previous attempt published this exact blob but crashed before
            # its manifest record.  The bytes are verified above, so recover by
            # publishing the missing record; the valid blob is never deleted.
            _append_manifest(root, record)
            return record

        _write_atomically(target, payload)
        try:
            _append_manifest(root, record)
        except BaseException:
            # The blob is valid evidence; a manifest failure must not destroy
            # it.  A retry re-enters above, verifies these bytes and publishes
            # the missing record.
            raise
        return record


# ---------------------------------------------------------------------------
# Reading the manifest
# ---------------------------------------------------------------------------


def load_manifest(root: str | Path) -> list[dict[str, Any]]:
    """Every archived capture, oldest first.  Missing manifest reads as empty."""

    path = manifest_path(Path(root) / ARCHIVE_DIRNAME) if Path(root).name != ARCHIVE_DIRNAME else manifest_path(root)
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        records.append(json.loads(stripped))
    records.sort(key=lambda item: (str(item.get("observed_at") or ""), str(item.get("source") or "")))
    return records


def captures_by_source(root: str | Path) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in load_manifest(root):
        grouped.setdefault(str(record.get("source")), []).append(record)
    return grouped


def verify_archived_blob(root: str | Path, record: Mapping[str, Any]) -> bool:
    """Whether the stored blob still hashes to its recorded digest."""

    path = Path(root) / ARCHIVE_DIRNAME / str(record["relative_path"])
    if not path.exists():
        return False
    return sha256_of(path.read_bytes()) == str(record["payload_sha256"])


# ---------------------------------------------------------------------------
# Cadence health
# ---------------------------------------------------------------------------


def _instant_before(first: str, second: str) -> bool:
    """Chronological ``first < second`` for capture instants, defensively parsed."""

    from .utils import parse_utc

    parsed_first, parsed_second = parse_utc(first), parse_utc(second)
    if parsed_first is not None and parsed_second is not None:
        return parsed_first < parsed_second
    return str(first) < str(second)


def _era_start(bootstrap_times: set[str], generation_times: set[str]) -> str | None:
    """Earliest instant of the archive/generation era, or None before the contract."""

    candidates = [stamp for stamp in {*bootstrap_times, *generation_times} if stamp]
    if not candidates:
        return None
    earliest = candidates[0]
    for candidate in candidates[1:]:
        if _instant_before(candidate, earliest):
            earliest = candidate
    return earliest


def _gap_hours(stamps: Sequence[str]) -> list[float]:
    from .utils import parse_utc

    hours: list[float] = []
    for earlier, later in zip(stamps, stamps[1:]):
        first, second = parse_utc(earlier), parse_utc(later)
        if first is not None and second is not None:
            hours.append((second - first).total_seconds() / 3600.0)
    return hours


def capture_cadence_health(
    conn: sqlite3.Connection, *, raw_dir: str | Path | None = None, now: str | None = None
) -> dict[str, Any]:
    """Facts about the availability-capture cadence.  No thresholds, no verdicts.

    Reports what was captured and what is missing.  It deliberately applies no
    "stale after N hours" rule: no such policy exists in the product, and
    inventing one here would turn a report into an unowned gate.

    Pairing is bootstrap-authoritative: player snapshots derive from
    ``bootstrap_static``, so ``snapshot_without_bootstrap`` uses only archive
    records with ``source == "bootstrap_static"`` -- a fixtures / event-live /
    element-summary record sharing the instant never satisfies the bootstrap
    side.  ``bootstrap_without_complete_snapshot`` is the reverse failure: a
    bootstrap archive whose database side holds no accepted COMPLETE
    generation for the same instant.  Captures that predate the
    archive/generation contract are reported as historically unverifiable,
    never fabricated as complete and never condemned as fresh gaps.
    """

    import statistics

    from .utils import parse_utc, utc_now

    rows = [
        (str(captured), int(count))
        for captured, count in conn.execute(
            "SELECT captured_at, COUNT(*) FROM player_snapshots WHERE captured_at IS NOT NULL "
            "GROUP BY captured_at ORDER BY captured_at"
        ).fetchall()
    ]
    stamps = [captured for captured, _ in rows]
    counts = [count for _, count in rows]
    gaps = _gap_hours(stamps)
    try:
        generated_runs = [
            (str(captured), int(official))
            for captured, official in conn.execute(
                "SELECT captured_at, official_element_count FROM bootstrap_generations ORDER BY captured_at"
            ).fetchall()
        ]
    except sqlite3.OperationalError:
        # Databases that predate the bootstrap_generations provenance table
        # have no generation records; their captures are reported as
        # historically unverifiable below, never fabricated as complete.
        generated_runs = []
    try:
        generation_attempts = [
            {
                "captured_at": str(captured),
                "accepted": bool(accepted),
                "official_element_count": int(official),
                "parsed_count": int(parsed),
                "persisted_count": int(persisted),
            }
            for captured, accepted, official, parsed, persisted in conn.execute(
                "SELECT captured_at, accepted, official_element_count, parsed_count, persisted_count "
                "FROM bootstrap_generations ORDER BY captured_at, id"
            ).fetchall()
        ]
    except sqlite3.OperationalError:
        generation_attempts = []
    observed_now = str(now).strip() if now else utc_now()
    latest = stamps[-1] if stamps else None
    age_hours = None
    if latest:
        parsed_latest, parsed_now = parse_utc(latest), parse_utc(observed_now)
        if parsed_latest is not None and parsed_now is not None:
            age_hours = round((parsed_now - parsed_latest).total_seconds() / 3600.0, 3)

    # A capture whose snapshot population is smaller than the payload's own
    # element count is partial.  Only captures that recorded that count can be
    # judged -- absence of the record is reported as such, never assumed complete.
    expected = {captured: official for captured, official in generated_runs}
    partial: list[dict[str, Any]] = []
    unverified: list[str] = []
    for captured, count in rows:
        if captured not in expected:
            unverified.append(captured)
        elif count != expected[captured]:
            partial.append({"captured_at": captured, "snapshot_rows": count,
                            "official_element_count": expected[captured]})

    report: dict[str, Any] = {
        "captures": len(stamps),
        "earliest_capture": stamps[0] if stamps else None,
        "latest_capture": latest,
        "now": observed_now,
        "latest_age_hours": age_hours,
        "min_gap_hours": round(min(gaps), 3) if gaps else None,
        "median_gap_hours": round(statistics.median(gaps), 3) if gaps else None,
        "max_gap_hours": round(max(gaps), 3) if gaps else None,
        "players_per_capture_min": min(counts) if counts else None,
        "players_per_capture_median": int(statistics.median(counts)) if counts else None,
        "players_per_capture_max": max(counts) if counts else None,
        "partial_captures": partial,
        "captures_without_a_generation_record": unverified,
    }
    if raw_dir is not None:
        grouped = captures_by_source(raw_dir)
        report["raw_captures_by_source"] = {source: len(items) for source, items in sorted(grouped.items())}
        report["latest_raw_capture_by_source"] = {
            source: (items[-1]["observed_at"] if items else None) for source, items in sorted(grouped.items())
        }
        # Player snapshots derive from bootstrap_static: only a bootstrap_static
        # archive at the same observed_at pairs with a snapshot capture.  A
        # fixtures / event-live / element-summary record sharing the instant
        # proves nothing about the bootstrap side.
        bootstrap_times = {
            str(item["observed_at"]) for item in grouped.get("bootstrap_static", [])
            if item.get("observed_at") is not None
        }
        generations_by_capture: dict[str, list[dict[str, Any]]] = {}
        for attempt in generation_attempts:
            generations_by_capture.setdefault(attempt["captured_at"], []).append(attempt)
        era_start = _era_start(bootstrap_times, set(generations_by_capture))
        snapshot_without_bootstrap: list[str] = []
        historically_unverifiable: list[str] = []
        for captured in stamps:
            if captured in bootstrap_times:
                continue
            if captured in generations_by_capture:
                # A new-era database capture with no bootstrap archive: the
                # archive side of the pair is genuinely missing.
                snapshot_without_bootstrap.append(captured)
            elif era_start is None or _instant_before(captured, era_start):
                # No generation record and predating (or entirely without) the
                # archive/generation era: historically unverifiable, never
                # fabricated as a fresh gap and never silently called healthy
                # because some other source happens to share the instant.
                historically_unverifiable.append(captured)
            else:
                snapshot_without_bootstrap.append(captured)
        report["snapshot_without_bootstrap"] = snapshot_without_bootstrap
        # Legacy alias kept for compatibility; same corrected bootstrap-only content.
        report["captures_without_an_archived_bootstrap"] = list(snapshot_without_bootstrap)
        report["historically_unverifiable_captures"] = historically_unverifiable
        # Reverse failure: a bootstrap archive whose database side holds no
        # accepted COMPLETE generation for the same instant.  The archive proves
        # the capture happened in the archive era, so a missing generation is a
        # genuine gap, not history.
        snapshot_counts = dict(zip(stamps, counts))
        bootstrap_without_complete_snapshot: list[str] = []
        for stamp in sorted(bootstrap_times):
            accepted = [item for item in generations_by_capture.get(stamp, []) if item["accepted"]]
            if not accepted:
                bootstrap_without_complete_snapshot.append(stamp)
                continue
            latest = accepted[-1]
            official = latest["official_element_count"]
            complete = (
                official > 0
                and latest["parsed_count"] == official
                and latest["persisted_count"] == official
                and snapshot_counts.get(stamp, 0) == official
            )
            if not complete:
                bootstrap_without_complete_snapshot.append(stamp)
        report["bootstrap_without_complete_snapshot"] = bootstrap_without_complete_snapshot
    return report
