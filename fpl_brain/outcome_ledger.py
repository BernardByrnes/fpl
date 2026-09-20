"""PE-5 -- outcome capture and the deterministic prediction-to-reality ledger.

WHY THIS EXISTS
---------------
The repository already stored outcomes, but only as CURRENT state.
``player_gameweeks`` is refreshed in place, and ``outcome_observations`` is
keyed ``UNIQUE(event, player_id, fixture_id)`` and upserts in place.  Both answer
"what is the outcome now"; neither can answer "what did we know at time T",
because the row a read observed yesterday is the row today's refresh overwrites.
A frozen prediction could therefore never be evaluated against the outcome that
was knowable when it was frozen.

This module adds the missing half: durable, append-only, POINT-IN-TIME outcome
history, and a deterministic ledger that relates one frozen prediction
generation to the later official reality.

THREE TIMES, NEVER ONE
----------------------
``event_time``       when the football event occurred (fixture kickoff)
``official_final_at``when the official result became final
``captured_at``      when THIS repository read it

They are separate columns on purpose.  Collapsing them into one timestamp is
what makes a backfilled or a provisional number indistinguishable from a final
one.

MISSING IS NOT ZERO
-------------------
The ledger never converts an absence into a value.  A player with no stored
observation, a scheduled placeholder, a fixture that has not kicked off, an
event that is not officially final, and a provisional capture are five distinct
states, each with its own canonical reason code, and NONE of them is zero
points.  A genuine completed did-not-play -- all explicit zeros -- is the
opposite case: it IS an observation and it IS evaluated.

A BLANK IS NOT A MODEL FAILURE
------------------------------
A blank Gameweek is an explicit ``TARGET_NO_FIXTURE`` row, not a missing one.

REUSE, NOT REINVENTION
----------------------
* the PE-1 boundary (:mod:`fpl_brain.historical_observations`) remains the ONLY
  causal historical predicate; nothing here weakens or duplicates it;
* the scheduled-placeholder signature is ``repositories``' single definition;
* the evaluation reason codes are :mod:`fpl_brain.walk_forward`'s canonical
  vocabulary, so a ledger exclusion and a scoreboard exclusion mean the same
  thing.

WHAT THIS MODULE DOES NOT DO
----------------------------
It does not regenerate, recalibrate or alter a prediction, and it does not
recompute an outcome from a model.  Prediction values in the ledger are the
frozen values that were actually issued; outcome values are the official facts
that were actually read, copied field-for-field, with a missing field left
missing.  Nothing here is a modelled expectation.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from . import repositories as repo
from .ingest_provenance import element_id_sha256
from .utils import parse_utc, utc_now
from . import walk_forward as wf

OUTCOME_LEDGER_VERSION = "outcome_ledger_v1.0.0"
OBSERVATION_CAPTURE_VERSION = "outcome_observation_v1.0.0"
FREEZE_CERTIFICATION_VERSION = "prediction_freeze_certification_v1.0.0"

#: How a key with several retained captures is resolved into the one the ledger
#: reports.  Declared, versioned and recorded on every ledger, so a later
#: consumer can say which supersession policy it used instead of assuming one.
SUPERSESSION_POLICY_VERSION = "latest_final_capture_then_latest_provisional_v1"

GRAIN_PLAYER_EVENT = "player_event"
GRAIN_PLAYER_FIXTURE = "player_fixture"
GRAINS = (GRAIN_PLAYER_EVENT, GRAIN_PLAYER_FIXTURE)

# -- freeze provenance states ------------------------------------------------
GENERATION_CERTIFIED = "GENERATION_CERTIFIED"
LEGACY_PROVENANCE = "LEGACY_PROVENANCE"
NOT_CERTIFIABLE = "NOT_CERTIFIABLE"

# -- observation states ------------------------------------------------------
OBSERVATION_FINAL = "FINAL"
OBSERVATION_PROVISIONAL = "PROVISIONAL"

# -- evaluation, reusing the walk-forward vocabulary where canonical ---------
EVALUATED = wf.EVALUATED
EXCLUDED = "EXCLUDED"
MODEL_PROJECTION_MISSING = wf.MODEL_PROJECTION_MISSING
INPUT_EVIDENCE_UNAVAILABLE = wf.INPUT_EVIDENCE_UNAVAILABLE
OUTCOME_NOT_FINALISED = wf.OUTCOME_NOT_FINALISED
OUTCOME_PLACEHOLDER_EXCLUDED = wf.OUTCOME_PLACEHOLDER_EXCLUDED
TARGET_NO_FIXTURE = wf.TARGET_NO_FIXTURE
TARGET_FIXTURE_NOT_PLAYED = wf.TARGET_FIXTURE_NOT_PLAYED
PLAYER_NOT_IN_OFFICIAL_POOL = wf.PLAYER_NOT_IN_OFFICIAL_POOL

#: The official per-fixture fields PE-5 preserves where the source supplies
#: them.  This is the contract's minimum set; the list is also the placeholder
#: signature ``repositories`` defines, which is why a capture carrying none of
#: these is refused rather than stored as "he scored nothing".
OFFICIAL_OUTCOME_FIELDS = (
    "minutes",
    "starts",
    "total_points",
    "goals_scored",
    "assists",
    "clean_sheets",
    "goals_conceded",
    "saves",
    "bonus",
    "bps",
    "yellow_cards",
    "red_cards",
    "penalties_saved",
    "penalties_missed",
    "own_goals",
    "defensive_contribution",
)

#: The grain at which each frozen kind makes its claim.  An event-grain
#: baseline may NEVER be expanded onto every DGW fixture: that would invent a
#: per-fixture claim the baseline never made and double-count it.
KIND_GRAINS: dict[str, str] = {
    "OFFICIAL_FPL_EP_NEXT": GRAIN_PLAYER_EVENT,
    "RECENT_POINTS_BASELINE": GRAIN_PLAYER_EVENT,
    "NAIVE_P90_BASELINE": GRAIN_PLAYER_EVENT,
    "NAIVE_MINUTES_BASELINE": GRAIN_PLAYER_EVENT,
    "MINUTES_V1": GRAIN_PLAYER_FIXTURE,
}

#: Where each kind's frozen value lives inside its own payload.  A kind absent
#: from this map is reported with an explicit missing value, never guessed.
KIND_VALUE_KEYS: dict[str, str] = {
    "OFFICIAL_FPL_EP_NEXT": "value",
    "RECENT_POINTS_BASELINE": "value",
    "NAIVE_P90_BASELINE": "value",
    "NAIVE_MINUTES_BASELINE": "value",
    "MINUTES_V1": "expected_minutes",
}

#: Fields aggregated to the headline event total.  ``None`` means "not stated
#: by the source", and a sum of only-stated values is only reported when EVERY
#: contributing fixture stated the field -- otherwise the total is withheld.
SUNNABLE_OUTCOME_FIELDS = OFFICIAL_OUTCOME_FIELDS


class OutcomeLedgerError(ValueError):
    """A capture or ledger build was refused because the contract forbade it."""


class PlaceholderObservationError(OutcomeLedgerError):
    """A scheduled placeholder was submitted as if it were an observation."""


class BackfillEvidenceError(OutcomeLedgerError):
    """Historical point-in-time history was claimed without an archived source."""


class FreezeAlreadyCertified(OutcomeLedgerError):
    """The freeze already carries provenance; provenance is append-only."""


# ---------------------------------------------------------------------------
# Canonical digests
# ---------------------------------------------------------------------------


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(namespace: str, lines: Iterable[str]) -> str:
    """SHA-256 over the namespace line plus the sorted payload lines.

    Sorting is what makes a digest independent of database row order; the
    namespace makes a fixture-grain digest unrepresentable as an event-grain one
    or as a capture digest.
    """

    body = "\n".join([namespace, *sorted(str(line) for line in lines)])
    return "sha256:" + hashlib.sha256(body.encode("utf-8")).hexdigest()


def _grain_namespace(grain: str) -> str:
    if grain not in GRAINS:
        raise OutcomeLedgerError(f"unknown ledger grain {grain!r}")
    return f"{OUTCOME_LEDGER_VERSION}:{grain}"


def _key_text(row: Mapping[str, Any]) -> str:
    fixture = "" if row.get("fixture_id") is None else str(int(row["fixture_id"]))
    return "|".join(
        (
            str(row.get("model_family") or ""),
            str(row.get("prediction_kind") or ""),
            str(int(row["event"])),
            str(int(row["player_id"])),
            fixture,
        )
    )


def destination_identity(grain: str, event: int, player_id: int, fixture_id: int | None) -> str:
    """The stable football key of one ledger row at ``grain``.

    Joined on ids only: a player/element id, an event, a fixture.  Names and
    display text are never part of the identity, so a rename cannot silently
    re-point a row at different football facts.
    """

    if grain not in GRAINS:
        raise OutcomeLedgerError(f"unknown ledger grain {grain!r}")
    if grain == GRAIN_PLAYER_FIXTURE and fixture_id is None:
        raise OutcomeLedgerError("a player_fixture ledger row requires a fixture id")
    parts = [grain, str(int(event)), str(int(player_id))]
    if fixture_id is not None:
        parts.append(str(int(fixture_id)))
    return ":".join(parts)


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CaptureResult:
    """What an ingest did: a new retained observation, or an idempotent no-op."""

    capture_digest: str
    inserted: bool
    observation_state: str
    grain: str
    event: int
    player_id: int
    fixture_id: int | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "capture_digest": self.capture_digest,
            "inserted": bool(self.inserted),
            "observation_state": self.observation_state,
            "grain": self.grain,
            "event": int(self.event),
            "player_id": int(self.player_id),
            "fixture_id": None if self.fixture_id is None else int(self.fixture_id),
        }


def event_finality(conn: sqlite3.Connection, event: int) -> str:
    """The stored official finality of one gameweek, by the accepted definition."""

    from . import planning

    state, _basis = planning.event_data_state(conn, int(event))
    return str(state)


def _event_updated_at(conn: sqlite3.Connection, event: int) -> str | None:
    row = conn.execute("SELECT updated_at FROM events WHERE id=?", (int(event),)).fetchone()
    if row is None:
        return None
    value = row["updated_at"] if isinstance(row, sqlite3.Row) else row[0]
    return str(value) if value else None


def _fixture_row(conn: sqlite3.Connection, fixture_id: int) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM fixtures WHERE id=?", (int(fixture_id),)).fetchone()
    return dict(row) if row is not None else None


def _observed_value(value: Any) -> Any:
    """A field as the source stated it: ``None`` stays ``None``, never 0."""

    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        numeric = float(value)
        return int(numeric) if numeric.is_integer() else numeric
    if isinstance(value, int):
        return int(value)
    return value


def _capture_fields(fields: Mapping[str, Any]) -> dict[str, Any]:
    """The official fields as the source stated them, in the contract's order.

    The required columns come first and every additional supplied field is kept
    alongside them rather than dropped, so "the source said more than PE-5's
    minimum" is preserved evidence instead of a lossy projection.
    """

    preserved: dict[str, Any] = {name: None for name in OFFICIAL_OUTCOME_FIELDS}
    for name, value in fields.items():
        if name in OFFICIAL_OUTCOME_FIELDS:
            preserved[name] = _observed_value(value)
    for name, value in fields.items():
        if name not in OFFICIAL_OUTCOME_FIELDS:
            preserved[str(name)] = value
    return preserved


def capture_digest_for(
    *,
    grain: str,
    event: int,
    player_id: int,
    fixture_id: int | None,
    captured_at: str,
    observation_state: str,
    source_name: str,
    source_identity: str | None,
    source_payload_sha256: str | None,
    archive_capture_id: str | None,
    payload: Mapping[str, Any],
) -> str:
    """The idempotency key of one observation.

    Content-derived on purpose: it is the same digest whether the row was the
    first or the thousandth inserted, so a ledger digest built from these never
    depends on database insertion order.  ``supersedes_capture_id`` is excluded
    because a supersession link is an audit relationship between two
    observations, not part of the observation itself.
    """

    return _digest(
        OBSERVATION_CAPTURE_VERSION,
        (
            "|".join(
                (
                    str(grain),
                    str(int(event)),
                    str(int(player_id)),
                    "" if fixture_id is None else str(int(fixture_id)),
                    str(captured_at),
                    str(observation_state),
                    str(source_name),
                    str(source_identity or ""),
                    str(source_payload_sha256 or ""),
                    str(archive_capture_id or ""),
                )
            ),
            _canonical_json(dict(payload)),
        ),
    )


def _assert_backfill_evidence(
    *,
    backfill: bool,
    captured_at: str | None,
    source_identity: str | None,
    source_payload_sha256: str | None,
    archive_capture_id: str | None,
) -> None:
    """No fake backfill: point-in-time history needs a proven archived source.

    The rule is deliberately narrow.  A capture made now needs nothing beyond
    its own provenance; a capture that claims a PAST observation time must name
    the archived source it was read from -- exact payload identity, the real
    capture identity, and the actual observable time.  Without all three the
    historical evidence is simply unavailable, and "unavailable" is the honest
    answer rather than a number manufactured from current state.
    """

    if not backfill:
        return
    missing = [
        label
        for label, value in (
            ("captured_at (the actual observable time)", captured_at),
            ("source_identity", source_identity),
            ("source_payload_sha256", source_payload_sha256),
            ("archive_capture_id", archive_capture_id),
        )
        if not str(value or "").strip()
    ]
    if missing:
        raise BackfillEvidenceError(
            "a historical backfill requires the archived source to prove " + ", ".join(missing)
        )


def _verify_archive_capture(
    conn: sqlite3.Connection,
    archive_root: Any,
    *,
    archive_capture_id: str,
    source_payload_sha256: str,
) -> None:
    """Confirm the named capture really is in the archive, with that payload."""

    from . import raw_archive

    records = {str(item.get("capture_id")): item for item in raw_archive.load_manifest(archive_root)}
    record = records.get(str(archive_capture_id))
    if record is None:
        raise BackfillEvidenceError(f"archive capture {archive_capture_id!r} is not in the manifest")
    recorded = str(record.get("payload_sha256") or "")
    if recorded != str(source_payload_sha256):
        raise BackfillEvidenceError(
            f"archive capture {archive_capture_id!r} holds payload {recorded[:12]!r}, "
            f"not the claimed {str(source_payload_sha256)[:12]!r}"
        )
    if not raw_archive.verify_archived_blob(archive_root, record):
        raise BackfillEvidenceError(
            f"archive capture {archive_capture_id!r} no longer matches its recorded digest"
        )


def capture_observation(
    conn: sqlite3.Connection,
    *,
    grain: str,
    event: int,
    player_id: int,
    fields: Mapping[str, Any],
    source_name: str,
    fixture_id: int | None = None,
    team_id: int | None = None,
    opponent_team_id: int | None = None,
    event_time: str | None = None,
    official_final_at: str | None = None,
    observation_state: str | None = None,
    source_identity: str | None = None,
    source_payload_sha256: str | None = None,
    fetch_run_id: int | None = None,
    archive_capture_id: str | None = None,
    captured_at: str | None = None,
    backfill: bool = False,
    supersedes_capture_id: int | None = None,
    correction_reason: str | None = None,
    archive_root: Any = None,
) -> CaptureResult:
    """Append one point-in-time observation; never overwrite an earlier one.

    Fails closed on every ambiguity:

    * a scheduled placeholder payload is REFUSED, not stored as a zero;
    * a capture claiming the football event had not happened yet is REFUSED;
    * a capture may not claim ``FINAL`` before the event is officially final;
    * a capture that reads an event whose official finality is not yet
      observable is downgraded to ``PROVISIONAL`` and stays provisional;
    * a claimed past observation time requires the archived source that proves it.
    """

    grain = str(grain)
    if grain not in GRAINS:
        raise OutcomeLedgerError(f"unknown observation grain {grain!r}")
    fixture = None if fixture_id is None else int(fixture_id)
    if grain == GRAIN_PLAYER_FIXTURE:
        if fixture is None or fixture <= 0:
            raise OutcomeLedgerError(
                "a player_fixture observation requires a real fixture id; a sentinel is not a fixture"
            )
    elif fixture is not None:
        raise OutcomeLedgerError(
            "a player_event observation must not carry a fixture id: that is the conflated grain"
        )

    event = int(event)
    player_id = int(player_id)
    payload = _capture_fields(fields)

    # The canonical placeholder signature, from repositories' single definition.
    # Everything the contract requires is inside the performance column set, so
    # the same predicate that guards every historical reader guards this write.
    signature = dict(fields)
    signature.setdefault("source", str(source_name))
    if repo.row_is_scheduled_placeholder(signature):
        raise PlaceholderObservationError(
            "the supplied evidence is a scheduled placeholder, which is 'not yet known' "
            "rather than an observation; PE-5 does not record it as a zero"
        )

    if fixture is not None:
        fixture_row = _fixture_row(conn, fixture)
        player = _player_record(conn, player_id)
        if team_id is None and player is not None:
            team_id = player.get("team_id")
        if fixture_row is not None:
            if event_time is None:
                event_time = fixture_row.get("kickoff_time")
            if opponent_team_id is None and team_id is not None:
                opponent_team_id = _opponent_of(fixture_row, int(team_id))

    _assert_backfill_evidence(
        backfill=backfill,
        captured_at=captured_at,
        source_identity=source_identity,
        source_payload_sha256=source_payload_sha256,
        archive_capture_id=archive_capture_id,
    )
    if backfill and archive_root is not None:
        _verify_archive_capture(
            conn,
            archive_root,
            archive_capture_id=str(archive_capture_id),
            source_payload_sha256=str(source_payload_sha256),
        )

    moment = str(captured_at).strip() if captured_at else utc_now()
    finality = event_finality(conn, event)
    state = _resolve_observation_state(
        requested=observation_state,
        event_finality=finality,
        event=event,
        moment=moment,
        official_final_at=official_final_at,
        conn=conn,
    )
    final_at = official_final_at
    if state == OBSERVATION_FINAL and final_at is None:
        final_at = _event_updated_at(conn, event)

    kickoff = parse_utc(str(event_time)) if event_time else None
    if kickoff is not None and parse_utc(moment) is not None and parse_utc(moment) < kickoff:
        raise OutcomeLedgerError(
            f"capture time {moment} precedes the football event at {event_time}; an observation "
            "cannot be evidence of an event that had not happened yet"
        )

    digest = capture_digest_for(
        grain=grain,
        event=event,
        player_id=player_id,
        fixture_id=fixture,
        captured_at=moment,
        observation_state=state,
        source_name=str(source_name),
        source_identity=source_identity,
        source_payload_sha256=source_payload_sha256,
        archive_capture_id=archive_capture_id,
        payload=payload,
    )
    columns = (
        "capture_digest", "grain", "event", "player_id", "fixture_id", "team_id",
        "opponent_team_id", "event_time", "official_final_at", "captured_at",
        "observation_state", "supersedes_capture_id", "correction_reason",
        "source_name", "source_identity", "source_payload_sha256", "fetch_run_id",
        "archive_capture_id", "backfill_evidence_json", "payload_json",
        *OFFICIAL_OUTCOME_FIELDS, "created_at",
    )
    values = (
        digest, grain, event, player_id, fixture,
        None if team_id is None else int(team_id),
        None if opponent_team_id is None else int(opponent_team_id),
        None if event_time is None else str(event_time),
        None if final_at is None else str(final_at),
        moment, state,
        None if supersedes_capture_id is None else int(supersedes_capture_id),
        correction_reason, str(source_name), source_identity, source_payload_sha256,
        None if fetch_run_id is None else int(fetch_run_id),
        archive_capture_id,
        None if not backfill else _canonical_json(
            {
                "captured_at": moment,
                "source_identity": source_identity,
                "source_payload_sha256": source_payload_sha256,
                "archive_capture_id": archive_capture_id,
                "archive_root": None if archive_root is None else str(archive_root),
            }
        ),
        _canonical_json(payload),
        *(payload[name] for name in OFFICIAL_OUTCOME_FIELDS),
        utc_now(),
    )
    cursor = conn.execute(
        f"INSERT INTO outcome_observation_captures({','.join(columns)}) "
        f"VALUES ({','.join('?' for _ in columns)}) ON CONFLICT(capture_digest) DO NOTHING",
        values,
    )
    return CaptureResult(
        capture_digest=digest,
        inserted=bool(cursor.rowcount),
        observation_state=state,
        grain=grain,
        event=event,
        player_id=player_id,
        fixture_id=fixture,
    )


def _opponent_of(fixture_row: Mapping[str, Any], team_id: int) -> int | None:
    home = fixture_row.get("team_h")
    away = fixture_row.get("team_a")
    if home is not None and int(home) == int(team_id):
        return None if away is None else int(away)
    if away is not None and int(away) == int(team_id):
        return None if home is None else int(home)
    return None


def _resolve_observation_state(
    *,
    requested: str | None,
    event_finality: str,
    event: int,
    moment: str,
    official_final_at: str | None,
    conn: sqlite3.Connection,
) -> str:
    """FINAL only when the stored finality and the capture moment both allow it."""

    if requested is not None and str(requested) not in (OBSERVATION_FINAL, OBSERVATION_PROVISIONAL):
        raise OutcomeLedgerError(f"unknown observation state {requested!r}")
    if str(event_finality) != "FINAL":
        if requested == OBSERVATION_FINAL:
            raise OutcomeLedgerError(
                f"event {event} is {event_finality}; a FINAL observation cannot be captured before "
                "the official source records the event as finalised"
            )
        return OBSERVATION_PROVISIONAL
    final_at = official_final_at or _event_updated_at(conn, event)
    if final_at is not None:
        known = parse_utc(str(final_at))
        read = parse_utc(moment)
        if known is not None and read is not None and read < known:
            # The event is final TODAY; this read happened BEFORE that was
            # observable, so what it saw was provisional whatever it looks like now.
            return OBSERVATION_PROVISIONAL
    return OBSERVATION_FINAL


def capture_completed_player_fixtures(
    conn: sqlite3.Connection,
    event: int,
    *,
    player_id: int | None = None,
    captured_at: str | None = None,
    source_name: str = "player_gameweeks_final",
    fetch_run_id: int | None = None,
    official_final_at: str | None = None,
) -> list[CaptureResult]:
    """Capture every completed official player-fixture row for one event.

    Rows come from ``repositories.completed_player_fixture_rows``, which already
    applies the canonical placeholder signature, so a finished fixture whose
    rows were never refreshed contributes NOTHING here rather than a set of
    fabricated zeros.  The whole event's official fields are copied verbatim:
    the required ones explicitly, and every additional source field alongside.
    """

    results: list[CaptureResult] = []
    for row in repo.completed_player_fixture_rows(conn, event=int(event), player_id=player_id):
        values = dict(row)
        fields = {
            key: value
            for key, value in values.items()
            if key not in {"player_id", "event", "fixture_id", "source", "raw_json", "updated_at"}
            and not key.startswith("fixture_")
        }
        fields["minutes"] = values.get("minutes")
        results.append(
            capture_observation(
                conn,
                grain=GRAIN_PLAYER_FIXTURE,
                event=int(event),
                player_id=int(values["player_id"]),
                fixture_id=int(values["fixture_id"]),
                fields=fields,
                source_name=str(source_name),
                opponent_team_id=values.get("opponent_team"),
                event_time=values.get("kickoff_time") or values.get("fixture_kickoff"),
                official_final_at=official_final_at,
                source_identity=f"player_gameweeks:{int(event)}:{int(values['fixture_id'])}",
                fetch_run_id=fetch_run_id,
                captured_at=captured_at,
            )
        )
    return results


# ---------------------------------------------------------------------------
# Point-in-time reads
# ---------------------------------------------------------------------------


def observation_captures(
    conn: sqlite3.Connection,
    *,
    grain: str | None = None,
    event: int | None = None,
    events: Sequence[int] | None = None,
    player_id: int | None = None,
    as_of: str | None = None,
) -> list[dict[str, Any]]:
    """Every capture visible at ``as_of``, in a deterministic order.

    ``as_of`` is the honest point-in-time boundary: a capture recorded after it
    did not exist yet and is invisible, which is exactly what stops a later
    outcome append from changing an earlier causal read.
    """

    clauses: list[str] = []
    params: list[Any] = []
    if grain is not None:
        if str(grain) not in GRAINS:
            raise OutcomeLedgerError(f"unknown observation grain {grain!r}")
        clauses.append("grain=?")
        params.append(str(grain))
    if event is not None:
        clauses.append("event=?")
        params.append(int(event))
    if events is not None:
        wanted = sorted({int(value) for value in events})
        if not wanted:
            return []
        clauses.append("event IN (" + ",".join("?" for _ in wanted) + ")")
        params.extend(wanted)
    if player_id is not None:
        clauses.append("player_id=?")
        params.append(int(player_id))
    if as_of is not None:
        clauses.append("captured_at<=?")
        params.append(str(as_of))
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    rows = conn.execute(
        "SELECT * FROM outcome_observation_captures"
        + where
        + " ORDER BY event, player_id, fixture_id IS NULL, fixture_id, captured_at, capture_digest",
        tuple(params),
    ).fetchall()
    return [_capture_record(row) for row in rows]


def _capture_record(row: Any) -> dict[str, Any]:
    record = dict(row)
    raw = record.pop("payload_json", None)
    try:
        record["payload"] = json.loads(raw) if raw else {}
    except (TypeError, ValueError):
        record["payload"] = {}
    if record.get("supersedes_capture_id") is not None:
        record["supersedes_capture_id"] = int(record["supersedes_capture_id"])
    return record


def select_capture(captures: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """The one capture a ledger reports for a key, by the declared policy.

    A finalised capture always wins over a provisional one.  Within the winning
    class the latest capture time wins, and the tie-break is the content digest
    rather than the row id, so the choice cannot depend on insertion order.
    """

    if not captures:
        return None
    finals = [row for row in captures if str(row.get("observation_state")) == OBSERVATION_FINAL]
    pool = finals or list(captures)
    return max(pool, key=lambda row: (str(row.get("captured_at") or ""), str(row.get("capture_digest") or "")))


# ---------------------------------------------------------------------------
# Freeze provenance and generation certification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FreezeCertification:
    """The outcome of certifying one prediction freeze."""

    freeze_identity: str
    provenance_state: str
    planning_event: int
    planning_cutoff: str
    bootstrap_generation_id: int | None
    official_fetch_run_id: int | None
    bootstrap_element_ids_sha256: str | None
    prediction_artifact_identity: str
    prediction_artifact_sha256: str
    reasons: tuple[str, ...]
    projection_run_ids: tuple[int, ...]

    @property
    def certified(self) -> bool:
        return self.provenance_state == GENERATION_CERTIFIED

    def as_dict(self) -> dict[str, Any]:
        return {
            "freeze_identity": self.freeze_identity,
            "provenance_state": self.provenance_state,
            "planning_event": int(self.planning_event),
            "planning_cutoff": self.planning_cutoff,
            "bootstrap_generation_id": self.bootstrap_generation_id,
            "official_fetch_run_id": self.official_fetch_run_id,
            "bootstrap_element_ids_sha256": self.bootstrap_element_ids_sha256,
            "prediction_artifact_identity": self.prediction_artifact_identity,
            "prediction_artifact_sha256": self.prediction_artifact_sha256,
            "reasons": list(self.reasons),
            "projection_run_ids": [int(rid) for rid in self.projection_run_ids],
        }


def prediction_artifact_digest(conn: sqlite3.Connection, run_ids: Sequence[int]) -> str:
    """Content digest of the immutable prediction artifact a freeze issued.

    One canonical line per frozen row, sorted, so the digest identifies WHAT was
    predicted and cannot drift with storage order.
    """

    ordered = sorted({int(rid) for rid in run_ids})
    if not ordered:
        return _digest(FREEZE_CERTIFICATION_VERSION, ())
    placeholders = ",".join("?" for _ in ordered)
    lines = [
        "|".join(
            (
                str(row["projection_run_id"]),
                str(row["kind"]),
                str(row["player_id"]),
                "" if row["fixture_id"] is None else str(row["fixture_id"]),
                str(row["event"]),
                str(row["model_version"]),
                str(row["payload_json"]),
            )
        )
        for row in conn.execute(
            "SELECT projection_run_id, kind, player_id, fixture_id, event, model_version, payload_json "
            f"FROM frozen_predictions WHERE projection_run_id IN ({placeholders})",
            tuple(ordered),
        ).fetchall()
    ]
    return _digest(FREEZE_CERTIFICATION_VERSION, lines)


def _declared_official_fetch_run_id(
    runs: Sequence[Mapping[str, Any]], explicit: int | None
) -> tuple[int | None, str | None]:
    """The official fetch identity a freeze recorded, from the runs themselves.

    ``projection_runs.official_run_ids`` is the existing provenance channel; the
    freeze's fetch identity is read out of it rather than invented.  Runs that
    name DIFFERENT fetches leave the freeze with no single recorded identity --
    one is not chosen for them, because doing so would manufacture agreement.
    """

    if explicit is not None:
        return int(explicit), None
    seen: set[int] = set()
    for run in runs:
        raw = run.get("official_run_ids")
        if not raw:
            continue
        try:
            payload = json.loads(raw) if isinstance(raw, str) else dict(raw)
        except (TypeError, ValueError):
            continue
        fetch = payload.get("fetch") if isinstance(payload, Mapping) else None
        if isinstance(fetch, Mapping) and fetch.get("run_id") is not None:
            seen.add(int(fetch["run_id"]))
    if len(seen) == 1:
        return seen.pop(), None
    if len(seen) > 1:
        return None, f"the freeze's runs record different official fetch runs: {sorted(seen)}"
    return None, None


def _bootstrap_generation(conn: sqlite3.Connection, generation_id: int) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT * FROM bootstrap_generations WHERE id=?", (int(generation_id),)
    ).fetchone()
    return dict(row) if row is not None else None


def accepted_generation_at(
    conn: sqlite3.Connection,
    *,
    observed_at: str,
    fetch_run_id: int | None = None,
) -> dict[str, Any] | None:
    """The accepted official generation a freeze at ``observed_at`` could have read.

    This is NOT "the newest accepted row".  A generation captured after the
    cutoff was not observable at the cutoff, so it is invisible here however new
    it is; and when the freeze recorded an official fetch identity, only the
    generation from THAT fetch can have supplied it.  Resolving the generation
    this way is what lets an already-frozen PE-4-era prediction run be
    certified from provenance it already carries, without rewriting
    ``projection_runs`` and without threading a new parameter through the
    certified prediction pipeline.
    """

    clauses = ["accepted = 1", "captured_at <= ?"]
    params: list[Any] = [str(observed_at)]
    if fetch_run_id is not None:
        clauses.append("fetch_run_id = ?")
        params.append(int(fetch_run_id))
    row = conn.execute(
        "SELECT * FROM bootstrap_generations WHERE "
        + " AND ".join(clauses)
        + " ORDER BY captured_at DESC, id DESC LIMIT 1",
        tuple(params),
    ).fetchone()
    return dict(row) if row is not None else None


def certify_prediction_freeze(
    conn: sqlite3.Connection,
    *,
    projection_run_ids: Sequence[int],
    bootstrap_generation_id: int | None = None,
    official_fetch_run_id: int | None = None,
    certified_at: str | None = None,
) -> FreezeCertification:
    """Certify that a NEW freeze proves the official generation that supplied it.

    A freeze is generation-certified only when ALL of these are proven from
    persisted rows, never from a name that happens to look right:

    1. the referenced official fetch exists;
    2. the referenced bootstrap generation exists;
    3. that generation is accepted;
    4. its ``fetch_run_id`` is the freeze's recorded official fetch identity;
    5. it was observable at or before the prediction cutoff;
    6. the official element-set identity and digest are retained and the digest
       recomputes from the retained id set.

    Missing, malformed or disagreeing provenance is NOT CERTIFIABLE.  A freeze
    that declares no generation and no official fetch at all is honestly
    LEGACY_PROVENANCE: pre-PE-5 runs are never rewritten to manufacture
    provenance, and no ``bootstrap_generations`` row is fabricated for them.

    The generation recorded here is the one certified.  A later accepted
    generation does not retroactively certify this freeze, and re-certification
    is refused outright, because provenance is itself point-in-time.
    """

    ordered = sorted({int(rid) for rid in projection_run_ids})
    if not ordered:
        raise OutcomeLedgerError("a freeze must name at least one projection run")
    runs: list[dict[str, Any]] = []
    for run_id in ordered:
        row = conn.execute("SELECT * FROM projection_runs WHERE id=?", (run_id,)).fetchone()
        if row is None:
            raise OutcomeLedgerError(f"projection run {run_id} does not exist")
        runs.append(dict(row))

    reasons: list[str] = []
    events = sorted({int(run["planning_event"]) for run in runs})
    cutoffs = sorted({str(run["data_cutoff"]) for run in runs})
    if len(events) > 1:
        reasons.append(f"the freeze's runs disagree on the planning event: {events}")
    if len(cutoffs) > 1:
        reasons.append(f"the freeze's runs disagree on the data cutoff: {cutoffs}")
    families = sorted((str(run["model_family"]), str(run["model_version"])) for run in runs)
    if len(set(families)) != len(families):
        reasons.append(f"the freeze names the same model family twice: {families}")

    planning_event = events[0] if len(events) == 1 else min(events)
    planning_cutoff = cutoffs[0] if len(cutoffs) == 1 else ""
    artifact_sha = prediction_artifact_digest(conn, ordered)
    artifact_identity = "|".join(
        (
            f"event={planning_event}",
            f"cutoff={planning_cutoff}",
            "runs=" + ",".join(str(rid) for rid in ordered),
        )
    )
    freeze_identity = _digest(
        FREEZE_CERTIFICATION_VERSION,
        (
            f"event={planning_event}",
            f"cutoff={planning_cutoff}",
            "runs=" + ",".join(str(rid) for rid in ordered),
            f"artifact={artifact_sha}",
        ),
    )

    fetch_run_id, fetch_reason = _declared_official_fetch_run_id(runs, official_fetch_run_id)
    if fetch_reason:
        reasons.append(fetch_reason)
    generation = (
        None if bootstrap_generation_id is None else _bootstrap_generation(conn, int(bootstrap_generation_id))
    )
    generation_id = None if bootstrap_generation_id is None else int(bootstrap_generation_id)
    if generation_id is None and fetch_run_id is not None and planning_cutoff:
        # The freeze recorded WHICH official fetch supplied it; the generation is
        # then the accepted one from that fetch that was observable at the
        # cutoff.  Deriving it here rather than at read time is the point: the
        # resolved id is pinned on the freeze and never re-resolved later.
        derived = accepted_generation_at(
            conn, observed_at=planning_cutoff, fetch_run_id=int(fetch_run_id)
        )
        if derived is not None:
            generation = derived
            generation_id = int(derived["id"])
    element_ids: list[int] = []
    element_digest: str | None = None
    element_count: int | None = None

    if generation_id is None:
        if fetch_run_id is None:
            # No generation and no official fetch was ever recorded.  That is the
            # truthful legacy state, not a failed certification.
            return _record_freeze(
                conn, runs=runs, ordered=ordered, freeze_identity=freeze_identity,
                provenance_state=LEGACY_PROVENANCE, planning_event=planning_event,
                planning_cutoff=planning_cutoff, artifact_identity=artifact_identity,
                artifact_sha=artifact_sha, reasons=reasons, official_fetch_run_id=None,
                generation=None, element_ids=[], element_digest=None, element_count=None,
                certified_at=certified_at or utc_now(),
            )
        reasons.append(
            "the freeze records an official fetch but no bootstrap generation, so the accepted "
            "official-player generation that supplied it cannot be proven"
        )
    else:
        if generation is None:
            reasons.append(f"bootstrap generation {generation_id} does not exist")
        else:
            if int(generation.get("accepted") or 0) != 1:
                reasons.append(
                    f"bootstrap generation {generation_id} was rejected and never certified the official pool"
                )
            recorded_fetch = generation.get("fetch_run_id")
            if recorded_fetch is None:
                reasons.append(f"bootstrap generation {generation_id} records no official fetch identity")
            elif fetch_run_id is None:
                reasons.append(
                    "the freeze records no official fetch identity to match against the generation"
                )
            elif int(recorded_fetch) != int(fetch_run_id):
                reasons.append(
                    f"bootstrap generation {generation_id} came from fetch run {int(recorded_fetch)}, "
                    f"not the freeze's recorded fetch run {int(fetch_run_id)}"
                )
            elif not _fetch_run_exists(conn, int(fetch_run_id)):
                reasons.append(f"the referenced official fetch run {int(fetch_run_id)} does not exist")
            captured_at = str(generation.get("captured_at") or "")
            if not captured_at:
                reasons.append(f"bootstrap generation {generation_id} records no capture time")
            elif not planning_cutoff:
                reasons.append("the freeze records no prediction cutoff to compare the generation against")
            elif str(captured_at) > str(planning_cutoff):
                reasons.append(
                    f"bootstrap generation {generation_id} was captured at {captured_at}, after the "
                    f"prediction cutoff {planning_cutoff}; a later generation cannot certify an earlier freeze"
                )
            element_ids, element_digest, element_count, digest_reasons = _retained_element_identity(generation)
            reasons.extend(digest_reasons)

    state = NOT_CERTIFIABLE if reasons else GENERATION_CERTIFIED
    return _record_freeze(
        conn, runs=runs, ordered=ordered, freeze_identity=freeze_identity,
        provenance_state=state, planning_event=planning_event, planning_cutoff=planning_cutoff,
        artifact_identity=artifact_identity, artifact_sha=artifact_sha, reasons=reasons,
        official_fetch_run_id=fetch_run_id, generation=generation, element_ids=element_ids,
        element_digest=element_digest, element_count=element_count,
        certified_at=certified_at or utc_now(),
    )


def _fetch_run_exists(conn: sqlite3.Connection, fetch_run_id: int) -> bool:
    row = conn.execute("SELECT 1 FROM fetch_runs WHERE id=?", (int(fetch_run_id),)).fetchone()
    return row is not None


def _retained_element_identity(
    generation: Mapping[str, Any],
) -> tuple[list[int], str | None, int | None, list[str]]:
    """Condition 6: the official element-set identity is retained AND deterministic.

    A digest that does not recompute from the id set it claims to describe is
    NOT recorded as if it verified: the id set is kept for audit and the digest
    is withheld, because storing an unverified digest is exactly the false
    certificate this contract exists to prevent.
    """

    reasons: list[str] = []
    raw = generation.get("element_ids_json")
    try:
        element_ids = sorted({int(pid) for pid in json.loads(raw)}) if raw else []
    except (TypeError, ValueError):
        return [], None, None, [
            f"bootstrap generation {generation.get('id')} retains a malformed official element id set"
        ]
    if not element_ids:
        return [], None, None, [
            f"bootstrap generation {generation.get('id')} retains no official element id set"
        ]
    recorded = str(generation.get("element_ids_sha256") or "")
    recomputed = element_id_sha256(element_ids)
    if not recorded:
        reasons.append(
            f"bootstrap generation {generation.get('id')} retains no official element digest"
        )
        return element_ids, None, len(element_ids), reasons
    if recorded != recomputed:
        reasons.append(
            f"bootstrap generation {generation.get('id')} records element digest {recorded[:12]} but its "
            f"retained id set hashes to {recomputed[:12]}"
        )
        return element_ids, None, len(element_ids), reasons
    return element_ids, recorded, len(element_ids), reasons


def _record_freeze(
    conn: sqlite3.Connection,
    *,
    runs: Sequence[Mapping[str, Any]],
    ordered: Sequence[int],
    freeze_identity: str,
    provenance_state: str,
    planning_event: int,
    planning_cutoff: str,
    artifact_identity: str,
    artifact_sha: str,
    reasons: Sequence[str],
    official_fetch_run_id: int | None,
    generation: Mapping[str, Any] | None,
    element_ids: Sequence[int],
    element_digest: str | None,
    element_count: int | None,
    certified_at: str,
) -> FreezeCertification:
    existing = conn.execute(
        "SELECT id FROM prediction_freeze_provenance WHERE freeze_identity=?", (freeze_identity,)
    ).fetchone()
    if existing is not None:
        raise FreezeAlreadyCertified(
            f"freeze {freeze_identity} already carries recorded provenance; provenance is append-only "
            "and a later generation does not retroactively re-certify an earlier freeze"
        )
    versions = sorted({str(run["model_version"]) for run in runs})
    hashes = sorted({str(run["config_hash"]) for run in runs if run.get("config_hash")})
    seeds = sorted({int(run["random_seed"]) for run in runs if run.get("random_seed") is not None})
    snapshots = sorted(
        {str(run["source_snapshot_sha256"]) for run in runs if run.get("source_snapshot_sha256")}
    )
    revisions = sorted({str(run["code_revision"]) for run in runs if run.get("code_revision")})
    runs_json = _canonical_json(
        [
            {
                "run_id": int(run["id"]),
                "model_family": str(run["model_family"]),
                "model_version": str(run["model_version"]),
                "config_hash": run.get("config_hash"),
                "random_seed": None if run.get("random_seed") is None else int(run["random_seed"]),
            }
            for run in sorted(runs, key=lambda item: int(item["id"]))
        ]
    )
    # The provenance row and its run membership are ONE fact.  Written
    # together because both tables refuse UPDATE and DELETE: a half-written
    # certificate could never be completed, and a freeze whose certificate
    # cannot be finished is worse than a freeze with none.
    with conn:
        cursor = conn.execute(
            """INSERT INTO prediction_freeze_provenance(
                 freeze_identity, provenance_state, planning_event, planning_cutoff,
                 model_version, config_hash, random_seed, source_snapshot_sha256, code_revision,
                 official_fetch_run_id, bootstrap_generation_id, bootstrap_captured_at,
                 bootstrap_element_count, bootstrap_element_ids_sha256, bootstrap_element_ids_json,
                 bootstrap_acceptance_rule_version, runs_json, prediction_artifact_identity,
                 prediction_artifact_sha256, reasons_json, certification_version, certified_at
               ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                freeze_identity,
                provenance_state,
                int(planning_event),
                str(planning_cutoff),
                versions[0] if len(versions) == 1 else None,
                hashes[0] if len(hashes) == 1 else None,
                seeds[0] if len(seeds) == 1 else None,
                snapshots[0] if len(snapshots) == 1 else None,
                revisions[0] if len(revisions) == 1 else None,
                official_fetch_run_id,
                None if generation is None else int(generation["id"]),
                None if generation is None else str(generation.get("captured_at") or "") or None,
                element_count,
                element_digest,
                None if not element_ids else _canonical_json(sorted(int(pid) for pid in element_ids)),
                None if generation is None else str(generation.get("acceptance_rule_version") or "") or None,
                runs_json,
                artifact_identity,
                artifact_sha,
                _canonical_json(list(reasons)),
                FREEZE_CERTIFICATION_VERSION,
                str(certified_at),
            ),
        )
        freeze_id = int(cursor.lastrowid)
        for run in sorted(runs, key=lambda item: int(item["id"])):
            conn.execute(
                """INSERT INTO prediction_freeze_runs(
                     freeze_id, projection_run_id, model_family, model_version, config_hash, random_seed
                   ) VALUES (?,?,?,?,?,?)""",
                (
                    freeze_id,
                    int(run["id"]),
                    str(run["model_family"]),
                    str(run["model_version"]),
                    run.get("config_hash"),
                    None if run.get("random_seed") is None else int(run["random_seed"]),
                ),
            )

    return FreezeCertification(
        freeze_identity=freeze_identity,
        provenance_state=provenance_state,
        planning_event=int(planning_event),
        planning_cutoff=str(planning_cutoff),
        bootstrap_generation_id=None if generation is None else int(generation["id"]),
        official_fetch_run_id=official_fetch_run_id,
        bootstrap_element_ids_sha256=element_digest,
        prediction_artifact_identity=artifact_identity,
        prediction_artifact_sha256=artifact_sha,
        reasons=tuple(str(reason) for reason in reasons),
        projection_run_ids=tuple(int(rid) for rid in ordered),
    )


def freeze_provenance(conn: sqlite3.Connection, freeze_identity: str) -> dict[str, Any] | None:
    """The provenance AS RECORDED, never re-resolved from the newest generation.

    ``bootstrap_generation_id`` is authoritative ONLY when ``provenance_state``
    is ``GENERATION_CERTIFIED``.  A ``NOT_CERTIFIABLE`` record keeps the
    generation it EXAMINED, together with the reasons it failed, so a reviewer
    can see what was offered and why it was refused -- which is not the same
    claim as "this generation supplied this freeze".
    """

    row = conn.execute(
        "SELECT * FROM prediction_freeze_provenance WHERE freeze_identity=?", (str(freeze_identity),)
    ).fetchone()
    if row is None:
        return None
    record = dict(row)
    for key in ("reasons_json", "bootstrap_element_ids_json", "runs_json"):
        raw = record.get(key)
        try:
            record[key.removesuffix("_json")] = json.loads(raw) if raw else []
        except (TypeError, ValueError):
            record[key.removesuffix("_json")] = []
    return record


def freeze_for_run(conn: sqlite3.Connection, projection_run_id: int) -> dict[str, Any] | None:
    """The provenance of the freeze a projection run belongs to, if certified."""

    row = conn.execute(
        "SELECT freeze_id FROM prediction_freeze_runs WHERE projection_run_id=?",
        (int(projection_run_id),),
    ).fetchone()
    if row is None:
        return None
    record = freeze_provenance(conn, _freeze_identity_of(conn, int(row["freeze_id"])))
    return record


def _freeze_identity_of(conn: sqlite3.Connection, freeze_id: int) -> str:
    row = conn.execute(
        "SELECT freeze_identity FROM prediction_freeze_provenance WHERE id=?", (int(freeze_id),)
    ).fetchone()
    return "" if row is None else str(row["freeze_identity"])


# ---------------------------------------------------------------------------
# The prediction-to-reality ledger
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RealityLedger:
    """A deterministic statement of what was predicted and what actually happened."""

    ledger_version: str
    grain: str
    freeze_identity: str
    provenance_state: str
    planning_event: int
    planning_cutoff: str
    as_of: str | None
    supersession_policy_version: str
    bootstrap_generation_id: int | None
    bootstrap_element_ids_sha256: str | None
    rows: tuple[dict[str, Any], ...]
    population_digest: str = ""
    ledger_digest: str = ""
    state_counts: tuple[tuple[str, int], ...] = field(default_factory=tuple)

    def evaluated(self) -> tuple[dict[str, Any], ...]:
        return tuple(row for row in self.rows if row["evaluation_state"] == EVALUATED)

    def excluded(self) -> tuple[dict[str, Any], ...]:
        return tuple(row for row in self.rows if row["evaluation_state"] == EXCLUDED)

    def as_dict(self, *, include_rows: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ledger_version": self.ledger_version,
            "grain": self.grain,
            "freeze_identity": self.freeze_identity,
            "provenance_state": self.provenance_state,
            "planning_event": int(self.planning_event),
            "planning_cutoff": self.planning_cutoff,
            "as_of": self.as_of,
            "supersession_policy_version": self.supersession_policy_version,
            "bootstrap_generation_id": self.bootstrap_generation_id,
            "bootstrap_element_ids_sha256": self.bootstrap_element_ids_sha256,
            "row_count": len(self.rows),
            "population_digest": self.population_digest,
            "ledger_digest": self.ledger_digest,
            "state_counts": {code: count for code, count in self.state_counts},
        }
        if include_rows:
            payload["rows"] = [dict(row) for row in self.rows]
        return payload


#: Fields that carry CONTENT into a ledger digest.  Surrogate database keys are
#: excluded on purpose: the digest must be identical for two stores holding the
#: same content, whatever order the rows were written in.
_DIGEST_ROW_FIELDS = (
    "destination_key",
    "model_family",
    "model_version",
    "prediction_kind",
    "prediction_state",
    "grain",
    "event",
    "player_id",
    "fixture_id",
    "team_id",
    "opponent_team_id",
    "event_time",
    "official_final_at",
    "observation_time",
    "observation_state",
    "observation_capture_digests",
    "observation_sources",
    "event_finality",
    "outcome_evidence",
    "outcome_state",
    "frozen_predicted_value",
    "official_outcome",
    "evaluation_state",
    "exclusion_reason",
    "exclusion_detail",
)


def build_reality_ledger(
    conn: sqlite3.Connection,
    *,
    freeze_identity: str,
    grain: str = GRAIN_PLAYER_EVENT,
    as_of: str | None = None,
) -> RealityLedger:
    """Relate one frozen prediction generation to the official reality.

    Deterministic by construction: the population is the union of the freeze's
    frozen prediction keys and the observable outcome keys, both sorted; every
    read has an explicit ORDER BY; and the digests are taken over content
    fields, never over surrogate row ids.  Building the same ledger twice, on
    two stores whose rows were written in different orders, yields identical
    rows, identical counts and identical digests.

    Nothing is dropped.  An excluded key is RETURNED with its reason code, so
    coverage is visible instead of implied, and a missing value is never
    converted into a zero.
    """

    if str(grain) not in GRAINS:
        raise OutcomeLedgerError(f"unknown ledger grain {grain!r}")
    provenance = freeze_provenance(conn, str(freeze_identity))
    if provenance is None:
        raise OutcomeLedgerError(
            f"freeze {freeze_identity} has no recorded provenance; capture it before building a ledger"
        )
    freeze_id = int(provenance["id"])
    runs = {
        int(row["projection_run_id"]): dict(row)
        for row in conn.execute(
            "SELECT * FROM prediction_freeze_runs WHERE freeze_id=? ORDER BY projection_run_id",
            (freeze_id,),
        ).fetchall()
    }
    predictions = _freeze_predictions(conn, sorted(runs), grain)
    target_events = _target_events(conn, provenance, predictions)
    captures = observation_captures(conn, events=target_events, as_of=as_of)

    keys: set[tuple[int, int, int | None]] = set()
    for row in predictions:
        keys.add((int(row["event"]), int(row["player_id"]), row["fixture_id"]))
    for row in captures:
        if str(row["grain"]) == grain:
            keys.add((int(row["event"]), int(row["player_id"]), row["fixture_id"]))
        elif grain == GRAIN_PLAYER_EVENT and str(row["grain"]) == GRAIN_PLAYER_FIXTURE:
            # Fixture-natural evidence is why an event total can be built at all,
            # so a fixture-grain capture alone still puts the player in the
            # event population -- with no prediction, which is explicit.
            keys.add((int(row["event"]), int(row["player_id"]), None))

    rows: list[dict[str, Any]] = []
    for event, player_id, fixture_id in sorted(
        keys, key=lambda item: (item[0], item[1], -1 if item[2] is None else int(item[2]))
    ):
        for row in _ledger_rows_for_key(
            conn,
            grain=str(grain),
            event=int(event),
            player_id=int(player_id),
            fixture_id=fixture_id,
            runs=runs,
            predictions=predictions,
            captures=captures,
            provenance=provenance,
        ):
            rows.append(row)

    rows.sort(key=_row_sort_key)
    digest_rows = [
        _canonical_json({name: row[name] for name in _DIGEST_ROW_FIELDS}) for row in rows
    ]
    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row["exclusion_reason"])] = counts.get(str(row["exclusion_reason"]), 0) + 1
    return RealityLedger(
        ledger_version=OUTCOME_LEDGER_VERSION,
        grain=str(grain),
        freeze_identity=str(freeze_identity),
        provenance_state=str(provenance["provenance_state"]),
        planning_event=int(provenance["planning_event"]),
        planning_cutoff=str(provenance["planning_cutoff"]),
        as_of=None if as_of is None else str(as_of),
        supersession_policy_version=SUPERSESSION_POLICY_VERSION,
        bootstrap_generation_id=(
            None
            if provenance.get("bootstrap_generation_id") is None
            else int(provenance["bootstrap_generation_id"])
        ),
        bootstrap_element_ids_sha256=provenance.get("bootstrap_element_ids_sha256"),
        rows=tuple(rows),
        population_digest=_digest(_grain_namespace(str(grain)), (_key_text(row) for row in rows)),
        ledger_digest=_digest(
            f"{OUTCOME_LEDGER_VERSION}:{grain}:{SUPERSESSION_POLICY_VERSION}", digest_rows
        ),
        state_counts=tuple(sorted(counts.items())),
    )


def _row_sort_key(row: Mapping[str, Any]) -> tuple:
    fixture = -1 if row.get("fixture_id") is None else int(row["fixture_id"])
    return (
        int(row["event"]),
        int(row["player_id"]),
        fixture,
        str(row.get("model_family") or ""),
        str(row.get("prediction_kind") or ""),
    )


def _freeze_predictions(
    conn: sqlite3.Connection, run_ids: Sequence[int], grain: str
) -> list[dict[str, Any]]:
    """The freeze's frozen predictions that make a claim at ``grain``."""

    wanted = [kind for kind, kind_grain in KIND_GRAINS.items() if kind_grain == grain]
    if not run_ids or not wanted:
        return []
    placeholders = ",".join("?" for _ in run_ids)
    kinds = ",".join("?" for _ in wanted)
    rows = conn.execute(
        "SELECT * FROM frozen_predictions "
        f"WHERE projection_run_id IN ({placeholders}) AND kind IN ({kinds}) "
        "ORDER BY event, player_id, fixture_id IS NULL, fixture_id, kind, projection_run_id",
        (*[int(rid) for rid in run_ids], *wanted),
    ).fetchall()
    records: list[dict[str, Any]] = []
    for row in rows:
        record = dict(row)
        try:
            record["payload"] = json.loads(record.pop("payload_json"))
        except (TypeError, ValueError):
            record["payload"] = {}
        records.append(record)
    return records


def _target_events(
    conn: sqlite3.Connection, provenance: Mapping[str, Any], predictions: Sequence[Mapping[str, Any]]
) -> list[int]:
    events = {int(provenance["planning_event"])}
    events.update(int(row["event"]) for row in predictions)
    return sorted(events)


def _ledger_rows_for_key(
    conn: sqlite3.Connection,
    *,
    grain: str,
    event: int,
    player_id: int,
    fixture_id: int | None,
    runs: Mapping[int, Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    captures: Sequence[Mapping[str, Any]],
    provenance: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if grain == GRAIN_PLAYER_FIXTURE:
        matching = [
            row
            for row in predictions
            if int(row["event"]) == event
            and int(row["player_id"]) == player_id
            and row["fixture_id"] is not None
            and int(row["fixture_id"]) == int(fixture_id)
        ]
    else:
        # An event-grain claim is never expanded onto a fixture: the kinds that
        # make a per-fixture claim simply are not in this population.
        matching = [
            row
            for row in predictions
            if int(row["event"]) == event and int(row["player_id"]) == player_id
        ]
    matching = sorted(
        ({str(row["kind"]): row for row in matching}.values()),
        key=lambda row: (
            str(row["kind"]),
            str(runs[int(row["projection_run_id"])]["model_family"]),
            str(runs[int(row["projection_run_id"])]["model_version"]),
        ),
    )
    context = _KeyContext(
        conn=conn, grain=grain, event=event, player_id=player_id, fixture_id=fixture_id,
        runs=runs, captures=captures, provenance=provenance,
    )
    outcome = context.resolve_outcome(has_prediction=bool(matching))
    if not matching:
        return [
            context.row(
                prediction=None,
                frozen_predicted_value=None,
                prediction_state="MISSING",
                outcome=outcome,
            )
        ]
    rows: list[dict[str, Any]] = []
    for prediction in matching:
        family = str(runs[int(prediction["projection_run_id"])]["model_family"])
        rows.append(
            context.row(
                prediction={**prediction, "model_family": family},
                frozen_predicted_value=context.predicted_value(prediction),
                prediction_state="PRESENT",
                outcome=outcome,
            )
        )
    return rows


class _KeyContext:
    """One football-key join, shared by every prediction row it emits."""

    def __init__(
        self,
        *,
        conn: sqlite3.Connection,
        grain: str,
        event: int,
        player_id: int,
        fixture_id: int | None,
        runs: Mapping[int, Mapping[str, Any]],
        captures: Sequence[Mapping[str, Any]],
        provenance: Mapping[str, Any],
    ) -> None:
        self.conn = conn
        self.grain = grain
        self.event = event
        self.player_id = player_id
        self.fixture_id = fixture_id
        self.runs = runs
        self.provenance = provenance
        self.captures = captures
        self._fixtures = _event_fixtures(conn, event)
        self._player = _player_record(conn, player_id)
        # Resolved ONCE per key, from the accepted finality definition, and
        # reported on every row the key emits: the ledger must state the
        # finalization the outcome was judged under, not only the verdict.
        self.event_state = event_finality(conn, event)

    # -- prediction side ----------------------------------------------------

    def predicted_value(self, prediction: Mapping[str, Any]) -> float | None:
        key = KIND_VALUE_KEYS.get(str(prediction["kind"]))
        value = (prediction.get("payload") or {}).get(key) if key else None
        return None if value is None else float(value)

    # -- outcome side -------------------------------------------------------

    def resolve_outcome(self, *, has_prediction: bool) -> dict[str, Any]:
        """What reality says, and separately why the key may not be scored.

        The two are computed independently on purpose.  ``outcome_state`` and the
        official values describe what was OBSERVED -- they do not disappear just
        because the freeze issued no projection for the key -- while
        ``exclusion_reason`` follows the canonical precedence, so a key is
        excluded for the first reason in the order that actually applies.
        """

        resolved = self._resolve_official_outcome()
        if has_prediction:
            return resolved
        if resolved["reason_code"] in _PRECEDES_MISSING_PROJECTION:
            return resolved
        return {
            **resolved,
            "reason_code": MODEL_PROJECTION_MISSING,
            "detail": "the freeze issued no projection for this player and event",
        }

    def _resolve_official_outcome(self) -> dict[str, Any]:
        """The official-outcome half: event/fixture truth and the retained evidence."""

        if self._player is None or not int(self._player.get("is_active") or 0):
            return _outcome(PLAYER_NOT_IN_OFFICIAL_POOL, "player is not in the stored official pool")
        state = self.event_state
        if str(state) != "FINAL":
            return _outcome(OUTCOME_NOT_FINALISED, f"event {self.event} is {state}")

        fixtures = sorted(
            fid
            for fid, row in self._fixtures.items()
            if int(row.get("team_h") or 0) == int(self._player["team_id"] or 0)
            or int(row.get("team_a") or 0) == int(self._player["team_id"] or 0)
        )
        if self.grain == GRAIN_PLAYER_FIXTURE:
            if self.fixture_id is None or int(self.fixture_id) not in fixtures:
                return _outcome(
                    TARGET_NO_FIXTURE,
                    f"the player's club has no fixture {self.fixture_id} in event {self.event}",
                )
            fixtures = [int(self.fixture_id)]
        elif not fixtures:
            return _outcome(
                TARGET_NO_FIXTURE, f"the player's club has no fixture in event {self.event}"
            )

        unplayed = [fid for fid in fixtures if int(self._fixtures[fid].get("finished") or 0) != 1]
        if unplayed:
            return _outcome(TARGET_FIXTURE_NOT_PLAYED, f"fixture(s) not played: {sorted(unplayed)}")
        if self.grain == GRAIN_PLAYER_EVENT:
            return self._resolve_event_outcome(fixtures)
        return self._resolve_fixture_outcome(int(self.fixture_id))

    def _capture_for(self, fixture_id: int) -> Mapping[str, Any] | None:
        return select_capture(
            [
                row
                for row in self.captures
                if str(row["grain"]) == GRAIN_PLAYER_FIXTURE
                and int(row["event"]) == self.event
                and int(row["player_id"]) == self.player_id
                and row["fixture_id"] is not None
                and int(row["fixture_id"]) == int(fixture_id)
            ]
        )

    def _is_placeholder(self, fixture_id: int) -> bool:
        rows = [
            dict(row)
            for row in self.conn.execute(
                "SELECT pg.* FROM player_gameweeks pg WHERE pg.event=? AND pg.player_id=? AND pg.fixture_id=?",
                (self.event, self.player_id, int(fixture_id)),
            ).fetchall()
        ]
        return bool(rows) and all(repo.row_is_scheduled_placeholder(row) for row in rows)

    def _resolve_fixture_outcome(self, fixture_id: int) -> dict[str, Any]:
        capture = self._capture_for(fixture_id)
        if capture is None:
            if self._is_placeholder(fixture_id):
                return _outcome(
                    OUTCOME_PLACEHOLDER_EXCLUDED,
                    f"the stored row for fixture {fixture_id} is a scheduled placeholder",
                )
            return _outcome(
                INPUT_EVIDENCE_UNAVAILABLE,
                f"no retained observation for fixture {fixture_id} at or before this read",
            )
        digests = (str(capture["capture_digest"]),)
        values = _official_values(capture)
        if str(capture["observation_state"]) != OBSERVATION_FINAL:
            return _outcome(
                OUTCOME_NOT_FINALISED,
                f"the only retained observation for fixture {fixture_id} is provisional",
                capture=capture,
                values=values,
                evidence="FIXTURE_CAPTURE",
                digests=digests,
            )
        return _outcome(
            EVALUATED,
            "",
            capture=capture,
            values=values,
            evidence="FIXTURE_CAPTURE",
            digests=digests,
        )

    def _resolve_event_outcome(self, fixtures: Sequence[int]) -> dict[str, Any]:
        direct = select_capture(
            [
                row
                for row in self.captures
                if str(row["grain"]) == GRAIN_PLAYER_EVENT
                and int(row["event"]) == self.event
                and int(row["player_id"]) == self.player_id
            ]
        )
        if direct is not None:
            values = _official_values(direct)
            digests = (str(direct["capture_digest"]),)
            if str(direct["observation_state"]) != OBSERVATION_FINAL:
                return _outcome(
                    OUTCOME_NOT_FINALISED,
                    "the only retained event observation is provisional",
                    capture=direct,
                    values=values,
                    evidence="EVENT_CAPTURE",
                    digests=digests,
                )
            return _outcome(
                EVALUATED,
                "",
                capture=direct,
                values=values,
                evidence="EVENT_CAPTURE",
                digests=digests,
            )

        chosen: list[Mapping[str, Any]] = []
        for fixture_id in fixtures:
            capture = self._capture_for(fixture_id)
            if capture is None:
                if self._is_placeholder(fixture_id):
                    return _outcome(
                        OUTCOME_PLACEHOLDER_EXCLUDED,
                        f"the stored row for fixture {fixture_id} is a scheduled placeholder",
                    )
                return _outcome(
                    INPUT_EVIDENCE_UNAVAILABLE,
                    f"no retained observation for fixture {fixture_id}; an event total built from "
                    "partial fixture evidence would present an incomplete headline total as complete",
                )
            chosen.append(capture)
        values, digests = _aggregate_fixture_values(chosen)
        if any(str(row["observation_state"]) != OBSERVATION_FINAL for row in chosen):
            return _outcome(
                OUTCOME_NOT_FINALISED,
                "at least one retained fixture observation for this event is provisional",
                capture=chosen[0],
                values=values,
                evidence="AGGREGATED_FIXTURE_CAPTURES",
                digests=digests,
                captures=chosen,
            )
        return _outcome(
            EVALUATED,
            "",
            capture=chosen[0],
            values=values,
            evidence="AGGREGATED_FIXTURE_CAPTURES",
            digests=digests,
            captures=chosen,
        )

    # -- row assembly -------------------------------------------------------

    def row(
        self,
        *,
        prediction: Mapping[str, Any] | None,
        frozen_predicted_value: float | None,
        prediction_state: str,
        outcome: Mapping[str, Any],
    ) -> dict[str, Any]:
        capture = outcome.get("capture")
        run = None if prediction is None else self.runs[int(prediction["projection_run_id"])]
        reason = str(outcome["reason_code"])
        evaluation = EVALUATED if reason == EVALUATED and prediction is not None else EXCLUDED
        return {
            "destination_key": destination_identity(
                self.grain, self.event, self.player_id, self.fixture_id
            ),
            "freeze_identity": str(self.provenance["freeze_identity"]),
            "provenance_state": str(self.provenance["provenance_state"]),
            "planning_cutoff": str(self.provenance["planning_cutoff"]),
            "bootstrap_generation_id": (
                None
                if self.provenance.get("bootstrap_generation_id") is None
                else int(self.provenance["bootstrap_generation_id"])
            ),
            "bootstrap_element_ids_sha256": self.provenance.get("bootstrap_element_ids_sha256"),
            "projection_run_id": None if run is None else int(run["projection_run_id"]),
            "model_family": None if run is None else str(run["model_family"]),
            "model_version": None if run is None else str(run["model_version"]),
            "config_hash": None if run is None else run.get("config_hash"),
            "random_seed": None if run is None else run.get("random_seed"),
            "prediction_kind": None if prediction is None else str(prediction["kind"]),
            "prediction_state": prediction_state,
            "grain": self.grain,
            "event": self.event,
            "player_id": self.player_id,
            "fixture_id": self.fixture_id,
            "team_id": None if self._player is None else self._player.get("team_id"),
            "opponent_team_id": None if capture is None else capture.get("opponent_team_id"),
            "event_time": None if capture is None else capture.get("event_time"),
            "official_final_at": None if capture is None else capture.get("official_final_at"),
            "observation_time": None if capture is None else capture.get("captured_at"),
            "observation_state": None if capture is None else capture.get("observation_state"),
            "observation_capture_digests": list(outcome.get("digests") or ()),
            "observation_sources": [dict(row) for row in outcome.get("sources") or ()],
            "event_finality": str(self.event_state),
            "outcome_evidence": str(outcome["evidence"]),
            "outcome_state": str(outcome["outcome_state"]),
            "frozen_predicted_value": frozen_predicted_value,
            "official_outcome": outcome.get("values"),
            "evaluation_state": evaluation,
            "exclusion_reason": reason,
            "exclusion_detail": "" if evaluation == EVALUATED else str(outcome["detail"]),
        }


def _source_record(capture: Mapping[str, Any]) -> dict[str, Any]:
    """The exact source one observation was read from, for the ledger's provenance.

    A digest alone proves WHICH bytes were read but not WHAT was read; the ledger
    must name the source too, so a reviewer can go back to the response or the
    certified artifact an outcome came from without reverse-engineering a hash.
    """

    return {
        "capture_digest": str(capture.get("capture_digest") or ""),
        "source_name": str(capture.get("source_name") or ""),
        "source_identity": capture.get("source_identity"),
        "source_payload_sha256": capture.get("source_payload_sha256"),
        "archive_capture_id": capture.get("archive_capture_id"),
        "fetch_run_id": capture.get("fetch_run_id"),
        "observation_state": str(capture.get("observation_state") or ""),
    }


def _outcome(
    reason_code: str,
    detail: str,
    *,
    capture: Mapping[str, Any] | None = None,
    values: Mapping[str, Any] | None = None,
    evidence: str = "NONE",
    digests: Sequence[str] | None = None,
    captures: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    # Sorted by digest, never by row id: the source attribution of one key must
    # not depend on the order its captures were written in.
    contributing = sorted(
        captures if captures is not None else ([capture] if capture is not None else []),
        key=lambda row: str(row.get("capture_digest") or ""),
    )
    return {
        "reason_code": str(reason_code),
        "detail": str(detail),
        "capture": capture,
        "values": None if values is None else dict(values),
        "evidence": str(evidence),
        "digests": tuple(str(value) for value in (digests or ())),
        "sources": tuple(_source_record(row) for row in contributing),
        "outcome_state": _outcome_state(str(reason_code)),
    }


#: Reasons that are decided BEFORE "the freeze issued no projection", in the
#: canonical precedence order: an event that is not final, a blank Gameweek and
#: an unplayed fixture are all known independently of any model output, so they
#: are the honest reason even for a key the freeze never covered.
_PRECEDES_MISSING_PROJECTION = frozenset(
    (PLAYER_NOT_IN_OFFICIAL_POOL, OUTCOME_NOT_FINALISED, TARGET_NO_FIXTURE, TARGET_FIXTURE_NOT_PLAYED)
)


def _outcome_state(reason_code: str) -> str:
    return {
        EVALUATED: "OBSERVED",
        MODEL_PROJECTION_MISSING: "MISSING",
        INPUT_EVIDENCE_UNAVAILABLE: "MISSING",
        OUTCOME_PLACEHOLDER_EXCLUDED: "PLACEHOLDER",
        OUTCOME_NOT_FINALISED: "NOT_FINALISED",
        TARGET_NO_FIXTURE: "BLANK",
        TARGET_FIXTURE_NOT_PLAYED: "NOT_PLAYED",
        PLAYER_NOT_IN_OFFICIAL_POOL: "NOT_IN_POOL",
    }.get(str(reason_code), "EXCLUDED")


def _official_values(capture: Mapping[str, Any]) -> dict[str, Any]:
    """Every official field one capture preserved, missing fields left missing.

    This is the source's own payload, not a modelled expectation: no field is
    recomputed, defaulted or dropped, which is what lets a later consumer check
    the ledger's arithmetic against the official evidence.
    """

    payload = capture.get("payload")
    if not isinstance(payload, Mapping):
        return {name: capture.get(name) for name in OFFICIAL_OUTCOME_FIELDS}
    return {str(name): value for name, value in payload.items()}


def _aggregate_fixture_values(
    captures: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """The headline event total, summed ONCE across the event's fixtures.

    A field is reported only when every contributing fixture stated it: summing
    the fixtures that happened to answer would silently present an incomplete
    total as a complete one, which is the same missing-is-not-zero error one
    grain down.  Blank and zero are therefore kept distinct.
    """

    ordered = sorted(captures, key=lambda row: str(row["capture_digest"]))
    values: dict[str, Any] = {}
    for name in SUNNABLE_OUTCOME_FIELDS:
        stated = [row.get(name) for row in ordered]
        if any(value is None for value in stated):
            values[name] = None
        else:
            values[name] = sum(int(value) for value in stated)
    return values, tuple(str(row["capture_digest"]) for row in ordered)


def _event_fixtures(conn: sqlite3.Connection, event: int) -> dict[int, dict[str, Any]]:
    return {
        int(row["id"]): dict(row)
        for row in conn.execute(
            "SELECT * FROM fixtures WHERE event=? AND id > 0 ORDER BY id", (int(event),)
        ).fetchall()
    }


def _player_record(conn: sqlite3.Connection, player_id: int) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM players WHERE id=?", (int(player_id),)).fetchone()
    return dict(row) if row is not None else None
