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
  thing;
* the event-finality rule is :mod:`fpl_brain.planning`'s; what changes is which
  evidence answers it, never the rule;
* an archived payload is parsed by :mod:`fpl_brain.parsers`' own canonical
  readers, so a backfilled observation says what the archive CONTAINS rather
  than what the caller asserted about it.

PINNED EVIDENCE, NOT CURRENT TABLES
-----------------------------------
``players``, ``fixtures``, ``events`` and ``player_gameweeks`` are refreshed in
place, so they describe NOW and nothing else.  A read that names a cutoff
therefore resolves the pool, the clubs, the fixtures and the event finality from
pinned or versioned evidence -- the generation the freeze proved, the accepted
generation observable at that cutoff, and the raw archive's verified captures --
and when that evidence does not exist it says so.  Substituting today's row for a
past instant is the same mistake as substituting zero for a missing outcome, one
table over.  A read that claims no cutoff is the present-tense read and may
answer from the live tables, which it then records as its evidence source.

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
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import planning as planning_state
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
    archive_root: Any,
    captured_at: str | None,
    source_identity: str | None,
    source_payload_sha256: str | None,
    archive_capture_id: str | None,
    fetch_run_id: int | None,
) -> None:
    """No fake backfill: point-in-time history needs a proven archived source.

    The rule is deliberately narrow, and asserted identifiers are not proof.
    A capture made now needs nothing beyond its own provenance; a capture that
    claims a PAST observation time must name the archived source it was read
    from -- exact payload identity, the real capture/fetch identity, and the
    actual observable time -- AND that source must be present in the archive
    with those bytes, which is checked in
    :func:`_verify_archive_capture`.  Without the archive itself the historical
    evidence is simply unavailable, and "unavailable" is the honest answer
    rather than a number manufactured from current state.
    """

    if not backfill:
        return
    missing = [
        label
        for label, value in (
            ("the archive it was read from (archive_root)", archive_root),
            ("captured_at (the actual observable time)", captured_at),
            ("source_identity", source_identity),
            ("source_payload_sha256", source_payload_sha256),
            ("archive_capture_id", archive_capture_id),
            ("fetch_run_id (the real capture identity)", fetch_run_id),
        )
        if value is None or not str(value).strip()
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
    source_identity: str,
    captured_at: str,
    fetch_run_id: int,
) -> tuple[dict[str, Any], Any]:
    """Confirm the named capture really is in the archive, with those bytes.

    The manifest record is the identity: its source, its observation time, its
    payload digest and its blob must all agree with what the caller asserted.
    A digest that matches while the blob does not hash to it is refused, and an
    identifier that is merely asserted -- present in the arguments but absent
    from the archive -- proves nothing, so it is refused too.

    What the record CANNOT prove is what the bytes say, so the verified payload
    is returned for the caller to bind the observation to.  Manifest metadata is
    identity, never content.
    """

    from . import raw_archive

    records = {str(item.get("capture_id")): item for item in raw_archive.load_manifest(archive_root)}
    record = records.get(str(archive_capture_id))
    if record is None:
        raise BackfillEvidenceError(
            f"archive capture {archive_capture_id!r} is not in the manifest; an asserted identifier is "
            "not archived evidence"
        )
    recorded = str(record.get("payload_sha256") or "")
    if recorded != str(source_payload_sha256):
        raise BackfillEvidenceError(
            f"archive capture {archive_capture_id!r} holds payload {recorded[:12]!r}, "
            f"not the claimed {str(source_payload_sha256)[:12]!r}"
        )
    if str(record.get("source") or "") != str(source_identity):
        raise BackfillEvidenceError(
            f"archive capture {archive_capture_id!r} came from source {str(record.get('source'))!r}, "
            f"not the claimed {str(source_identity)!r}"
        )
    if str(record.get("observed_at") or "") != str(captured_at):
        raise BackfillEvidenceError(
            f"archive capture {archive_capture_id!r} was observed at {str(record.get('observed_at'))!r}, "
            f"not the claimed observation time {str(captured_at)!r}"
        )
    recorded_run = record.get("run_id")
    if recorded_run is None or str(recorded_run) != str(int(fetch_run_id)):
        raise BackfillEvidenceError(
            f"archive capture {archive_capture_id!r} records fetch run {recorded_run!r}, "
            f"not the claimed {int(fetch_run_id)}"
        )
    if not raw_archive.verify_archived_blob(archive_root, record):
        raise BackfillEvidenceError(
            f"archive capture {archive_capture_id!r} no longer matches its recorded digest; the archived "
            "blob is required to prove the observation, so a claim with no intact bytes is refused"
        )
    payload = _archived_payload(archive_root, record)
    if payload is None:
        raise BackfillEvidenceError(
            f"archive capture {archive_capture_id!r} holds no readable JSON payload, so there is no "
            "archived content the observation can be bound to"
        )
    return dict(record), payload


#: The archived sources that carry per-player, per-fixture observation rows.
#: A backfilled observation is bound to the rows one of these parses; a source
#: with no such rows -- a schedule, a bootstrap pool, a manager's picks -- cannot
#: witness an observation at all, so it is refused rather than trusted.
_OBSERVATION_ARCHIVE_SOURCES = ("element_summary", "event_live")


def _archive_source_kind(source: str) -> tuple[str, str]:
    """``(kind, suffix)`` for one archive source slug.

    The fetch layer names a per-player capture ``element_summary_<player_id>``
    and a per-event capture ``event_live_<event>``; the bare source name is
    accepted too, and the identity then has to come out of the archived rows.
    """

    text = str(source)
    for kind in _OBSERVATION_ARCHIVE_SOURCES:
        if text == kind:
            return kind, ""
        if text.startswith(kind + "_"):
            return kind, text[len(kind) + 1 :]
    return "", ""


def _row_raw(row: Any) -> Mapping[str, Any]:
    raw = getattr(row, "raw_json", None)
    return raw if isinstance(raw, Mapping) else {}


def _row_stated_element(row: Any) -> int | None:
    """The player identity an archived row states about ITSELF, if it states one.

    "About itself" is the whole point: a row that never names an element is not
    evidence about anybody, however confidently a parser -- which was told whose
    summary it was reading -- has labelled it.  The parsers retain the element id
    they actually read for both sources, so this is a statement the payload
    made, never the claim that was being checked.
    """

    for key in ("element", "id"):
        value = _row_raw(row).get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    return None


def _row_stated_field(row: Any, *keys: str) -> Any:
    """One archived value, from the raw payload or the row it was parsed into."""

    raw = _row_raw(row)
    for key in keys:
        if key in raw:
            return raw[key]
    for key in keys:
        value = getattr(row, key, None)
        if value is not None:
            return value
    return None


def _archived_row_identity(row: Any) -> dict[str, Any]:
    """The football identity one archived row states: club, opponent, kickoff.

    Read from the raw payload when it states these, and otherwise from the row
    the canonical parser normalised them into -- the live endpoint keeps the
    club pair and the kickoff on its ``explain`` leg rather than on the element,
    so the parsed record is where that identity survives.
    """

    was_home = _row_stated_field(row, "was_home", "is_home")
    home = _row_stated_field(row, "team_h")
    away = _row_stated_field(row, "team_a")
    team: int | None = None
    opponent: int | None = None
    if was_home is not None and home is not None and away is not None:
        team, opponent = (int(home), int(away)) if int(bool(was_home)) else (int(away), int(home))
    stated_opponent = _row_stated_field(row, "opponent_team")
    if stated_opponent is not None:
        opponent = int(stated_opponent)
    kickoff = _row_stated_field(row, "kickoff_time")
    return {
        "team_id": team,
        "opponent_team_id": opponent,
        "event_time": None if kickoff is None else str(kickoff),
    }


def _archived_observation_values(rows: Sequence[Any], *, grain: str) -> dict[str, Any]:
    """The official values the archived rows state, at the grain being captured.

    Fixture grain takes the one archived row.  Event grain takes the canonical
    event aggregation of its rows -- summing ONCE, and leaving a field missing
    when any contributing fixture left it missing -- which is the same rule the
    ledger applies in reverse, so the two cannot disagree about a total.
    """

    per_row = [
        {name: getattr(row, name, None) for name in OFFICIAL_OUTCOME_FIELDS} for row in rows
    ]
    if grain == GRAIN_PLAYER_EVENT:
        values, _ = _aggregate_fixture_values(per_row)
        return values
    return per_row[0]


def _bind_backfilled_observation(
    *,
    record: Mapping[str, Any],
    payload: Any,
    grain: str,
    event: int,
    player_id: int,
    fixture_id: int | None,
    claimed: Mapping[str, Any],
    team_id: int | None,
    opponent_team_id: int | None,
    event_time: str | None,
) -> dict[str, Any]:
    """Bind a backfilled observation to the ARCHIVED content, not the manifest.

    The manifest record proves WHICH bytes were read and when; it says nothing
    about what those bytes contain.  A caller could otherwise name a real
    archived capture, keep its identity and its digest, and then attach whatever
    fields, player, fixture or club it liked: every one of those claims would
    agree with the manifest while the observation itself was fabricated.

    So the archived payload is parsed with the canonical parser for its source
    and the claim must be exactly what that parsed content states -- the same
    player, event, fixture and club identity, the same official fields, and the
    same additional source fields.  Anything the archive does not state is
    refused, never stored, and identity the caller left unstated is filled from
    the archive rather than from a table that is refreshed in place.
    """

    from . import parsers

    capture_id = str(record.get("capture_id"))
    kind, suffix = _archive_source_kind(str(record.get("source")))
    if not kind:
        raise BackfillEvidenceError(
            f"archived capture {capture_id!r} comes from source {str(record.get('source'))!r}, which "
            "carries no per-player observation rows; an observation cannot be bound to its content"
        )
    if kind == "event_live":
        recorded_event = record.get("event")
        archived_event = (
            int(recorded_event)
            if recorded_event is not None
            else (int(suffix) if suffix.isdigit() else None)
        )
        if archived_event is None or archived_event != int(event):
            raise BackfillEvidenceError(
                f"archived capture {capture_id!r} does not state the claimed event {int(event)}; an "
                "event-live capture is evidence only for the event it was taken for"
            )
        parsed = parsers.parse_event_live(payload, int(event))
    else:
        archived_player = int(suffix) if suffix.isdigit() else None
        if archived_player is not None and archived_player != int(player_id):
            raise BackfillEvidenceError(
                f"archived capture {capture_id!r} is the element summary of player {archived_player}, "
                f"not of the claimed player {int(player_id)}"
            )
        parsed = parsers.parse_element_summary(payload, int(player_id))

    # PLAYER identity comes first, from the rows themselves.  A parser is TOLD
    # which player it is reading, so its records carry the caller's id whatever
    # the payload says; taking that at face value would file one player's
    # numbers under another player's id whenever the two happened to score the
    # same.  The element a row states about itself is therefore what selects
    # rows, and rows naming nobody are not evidence about anybody.
    in_event = [row for row in parsed if int(row.event) == int(event)]
    with_element = [(row, _row_stated_element(row)) for row in in_event]
    named = {identity for _, identity in with_element if identity is not None}
    # A bulk payload legitimately carries other players -- an event-live response
    # serves the whole pool -- so those rows are simply not this claim's.  The
    # question of WHOSE the payload is comes before the question of which of his
    # fixtures it holds.
    ours = [row for row, identity in with_element if identity == int(player_id)]
    if not ours and named:
        # The payload names players, but not this one: it is not evidence about
        # the claimant at all.
        raise BackfillEvidenceError(
            f"archived capture {capture_id!r} states player identities {sorted(named)} at event "
            f"{int(event)}, none of which is the claimed player {int(player_id)}; an archived payload "
            "is evidence only for the players it names"
        )
    if not ours and in_event:
        # Rows exist for this event and none of them says whose they are, so
        # there is nothing in the bytes that can bind the claim to a player.
        raise BackfillEvidenceError(
            f"archived capture {capture_id!r} states no player identity for the claimed player "
            f"{int(player_id)} at event {int(event)}, so the claim cannot be bound to its content"
        )
    matched = [row for row in ours if fixture_id is None or row.fixture_id == int(fixture_id)]
    if not matched:
        raise BackfillEvidenceError(
            f"archived capture {capture_id!r} holds no observation for player {int(player_id)}, event "
            f"{int(event)}" + ("" if fixture_id is None else f", fixture {int(fixture_id)}")
        )
    if fixture_id is not None and len(matched) > 1:
        raise BackfillEvidenceError(
            f"archived capture {capture_id!r} holds {len(matched)} rows for fixture {int(fixture_id)}; a "
            "fixture-grain observation must resolve to exactly one archived row"
        )
    stated_fixtures = {int(row.fixture_id) for row in matched}
    if len(stated_fixtures) != len(matched):
        raise BackfillEvidenceError(
            f"archived capture {capture_id!r} holds {len(matched)} rows for player {int(player_id)} at "
            f"event {int(event)} but only {len(stated_fixtures)} distinct fixtures; a row that names no "
            "fixture cannot witness which fixture was played"
        )

    archived = _archived_observation_values(matched, grain=grain)
    for name in OFFICIAL_OUTCOME_FIELDS:
        if _observed_value(claimed.get(name)) != _observed_value(archived[name]):
            raise BackfillEvidenceError(
                f"archived capture {capture_id!r} states {name}={archived[name]!r} for player "
                f"{int(player_id)} event {int(event)}, not the claimed {claimed.get(name)!r}; a backfilled "
                "observation is what the archive says, not what the caller asserts"
            )
    for name in claimed:
        if name in OFFICIAL_OUTCOME_FIELDS:
            continue
        for row in matched:
            raw = _row_raw(row)
            if name not in raw or _observed_value(raw.get(name)) != _observed_value(claimed[name]):
                raise BackfillEvidenceError(
                    f"archived capture {capture_id!r} does not state {str(name)!r} with the claimed value "
                    f"{claimed[name]!r}; an additional field is archived evidence only when the archive "
                    "states it"
                )

    # Identity, at the grain being captured.  FIXTURE grain has exactly one row,
    # so that row's club, opponent and kickoff ARE the observation's.  EVENT
    # grain may span two fixtures, and a double gameweek has no single club,
    # opponent or kickoff: taking one row's identity would file fixture 1's
    # opponent on a row that also sums fixture 2, which is a fabrication dressed
    # as a fact.  So the event grain keeps a field only when every contributing
    # row agrees on it, and otherwise leaves it NULL -- "not applicable at this
    # grain" rather than an arbitrary member of a set.
    stated = [_archived_row_identity(row) for row in matched]
    agreed: dict[str, Any] = {}
    for field in ("team_id", "opponent_team_id", "event_time"):
        values = {_observed_value(item[field]) for item in stated}
        agreed[field] = values.pop() if len(values) == 1 else None

    for label, claimed_value, archived_value in (
        ("club/team", team_id, agreed["team_id"]),
        ("opponent", opponent_team_id, agreed["opponent_team_id"]),
    ):
        if claimed_value is None:
            continue
        if archived_value is None:
            raise BackfillEvidenceError(
                f"archived capture {capture_id!r} states no {label} for player {int(player_id)} event "
                f"{int(event)}, so the claimed {claimed_value!r} is not archived evidence"
            )
        if _observed_value(claimed_value) != _observed_value(archived_value):
            raise BackfillEvidenceError(
                f"archived capture {capture_id!r} states the {label} as {archived_value!r} for player "
                f"{int(player_id)} event {int(event)}, not the claimed {claimed_value!r}"
            )
    if event_time is not None:
        claimed_kickoff = parse_utc(str(event_time))
        stated_kickoffs = {str(item["event_time"]) for item in stated if item["event_time"] is not None}
        if not stated_kickoffs:
            # No archived row says when this was played, so nothing in the
            # evidence supports the claim -- and the contract's rule is that a
            # caller-supplied value is not evidence just because no row
            # contradicts it.
            raise BackfillEvidenceError(
                f"archived capture {capture_id!r} states no kickoff time for player {int(player_id)} "
                f"event {int(event)}, so the claimed {event_time} is not archived evidence"
            )
        if claimed_kickoff not in {parse_utc(value) for value in stated_kickoffs}:
            raise BackfillEvidenceError(
                f"archived capture {capture_id!r} was played at {sorted(stated_kickoffs)}, not at the "
                f"claimed {event_time}"
            )
    if grain == GRAIN_PLAYER_FIXTURE:
        return {
            "team_id": agreed["team_id"] if team_id is None else team_id,
            "opponent_team_id": (
                agreed["opponent_team_id"] if opponent_team_id is None else opponent_team_id
            ),
            "event_time": agreed["event_time"] if event_time is None else str(event_time),
        }
    # Event grain: a club may be carried (one player belongs to one club for a
    # whole event, so agreement is the rule), but a per-fixture identity may
    # not.  A kickoff time belongs to a fixture rather than to an event, so the
    # headline row keeps none: filling it from one of two legs would state a
    # per-fixture fact on an aggregate that spans both.
    return {
        "team_id": agreed["team_id"],
        "opponent_team_id": agreed["opponent_team_id"],
        "event_time": None,
    }


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
    * an explicitly ``PROVISIONAL`` capture is never promoted, whatever the
      event's finality looks like by the time the row is written;
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

    if fixture is not None and not backfill:
        # A capture made NOW may read the fixture and the player from the live
        # tables: now is exactly what those tables hold.  A backfill may not --
        # see below, where its identity comes from the archived row instead.
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
        archive_root=archive_root,
        captured_at=captured_at,
        source_identity=source_identity,
        source_payload_sha256=source_payload_sha256,
        archive_capture_id=archive_capture_id,
        fetch_run_id=fetch_run_id,
    )
    if backfill:
        # An asserted identifier is not evidence: the archived blob itself must be
        # present and must still hash to the claimed digest, and the observation
        # must be bound to what those bytes SAY -- the same player, event,
        # fixture, club and fields -- rather than to the manifest that names them.
        record, archived_payload = _verify_archive_capture(
            conn,
            archive_root,
            archive_capture_id=str(archive_capture_id),
            source_payload_sha256=str(source_payload_sha256),
            source_identity=str(source_identity),
            captured_at=str(captured_at),
            fetch_run_id=int(fetch_run_id),
        )
        bound = _bind_backfilled_observation(
            record=record,
            payload=archived_payload,
            grain=grain,
            event=event,
            player_id=player_id,
            fixture_id=fixture,
            claimed=payload,
            team_id=team_id,
            opponent_team_id=opponent_team_id,
            event_time=event_time,
        )
        team_id = bound["team_id"]
        opponent_team_id = bound["opponent_team_id"]
        event_time = bound["event_time"]

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
    if requested == OBSERVATION_PROVISIONAL:
        # The caller's own statement about what it read is evidence too.  A
        # capture that says "provisional" is NEVER promoted later by the arrival
        # of an official finality it did not observe: that silent upgrade is
        # exactly the false finality this module exists to refuse.  A later,
        # genuinely final read is a NEW observation, not a rewrite of this one.
        return OBSERVATION_PROVISIONAL
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


#: The one key ``projection_runs.official_run_ids`` records an official fetch
#: under.  This is the existing provenance channel; PE-5 reads it, never invents it.
PROVENANCE_FETCH_KEY = "fetch"

#: How one run's persisted provenance read out: a fetch run id, or why there is
#: none.  ``None`` means the record is intact and names a fetch; ``"absent"``
#: means the run records nothing about an official fetch -- which is the
#: truthful pre-PE-5 state; anything else describes a record that EXISTS and
#: cannot be read, which is a broken claim rather than an absence.
_PROVENANCE_ABSENT = "absent"


def _persisted_fetch_run_id(run: Mapping[str, Any]) -> tuple[int | None, str | None]:
    """The official fetch ONE run recorded, or why its record does not answer.

    Absence and malformation are deliberately different answers: a pre-PE-5 run
    that records nothing is honestly legacy, while a record that is present and
    unreadable must not be read as if the run had recorded nothing either.
    """

    raw = run.get("official_run_ids")
    if raw is None or not str(raw).strip():
        return None, _PROVENANCE_ABSENT
    try:
        payload = json.loads(raw) if isinstance(raw, (str, bytes, bytearray)) else dict(raw)
    except (TypeError, ValueError):
        return None, "records an official_run_ids value that is not readable JSON"
    if not isinstance(payload, Mapping):
        return None, "records an official_run_ids value that is not an object"
    fetch = payload.get(PROVENANCE_FETCH_KEY)
    if fetch is None:
        return None, _PROVENANCE_ABSENT
    if not isinstance(fetch, Mapping):
        return None, f"records an official {PROVENANCE_FETCH_KEY} provenance that is not an object"
    value = fetch.get("run_id")
    if value is None:
        return None, "records an official fetch provenance with no run_id"
    try:
        return int(value), None
    except (TypeError, ValueError):
        return None, f"records a non-integer official fetch run_id {value!r}"


def _declared_official_fetch_run_id(
    runs: Sequence[Mapping[str, Any]], explicit: int | None
) -> tuple[int | None, list[str]]:
    """The official fetch identity a freeze recorded, from the runs themselves.

    ``projection_runs.official_run_ids`` is the existing provenance channel; the
    freeze's fetch identity is read out of it rather than invented.  The identity
    must be recorded by EVERY run of the freeze and every recorded identity must
    agree: a fetch only some of the runs state is not the freeze's provenance,
    and choosing one of several recorded fetches would manufacture agreement
    the runs do not have.

    An explicitly supplied fetch identity is an ASSERTION against that channel,
    never a substitute for it.  It is accepted only when every run's persisted
    provenance names exactly that fetch; a fetch the runs contradict -- or a
    fetch no run records at all -- is not evidence about this freeze, and
    accepting it would let a caller attach a provenance the runs never carried.

    A freeze whose runs ALL record nothing (and which declares nothing) keeps the
    truthful legacy answer: no fetch identity, and no fabricated generation.
    """

    recorded: dict[int, list[int]] = {}
    absent: list[int] = []
    unreadable: list[str] = []
    for run in runs:
        run_id = int(run["id"])
        fetch, pathology = _persisted_fetch_run_id(run)
        if pathology is None:
            recorded.setdefault(int(fetch), []).append(run_id)
        elif pathology == _PROVENANCE_ABSENT:
            absent.append(run_id)
        else:
            unreadable.append(f"projection run {run_id} {pathology}")

    reasons: list[str] = list(unreadable)
    if absent and (recorded or explicit is not None):
        reasons.append(
            f"projection run(s) {absent} record no official fetch provenance while the freeze is certified "
            "from one; a fetch identity only some of the runs state is not this freeze's provenance"
        )
    if len(recorded) > 1:
        reasons.append(f"the freeze's runs record different official fetch runs: {sorted(recorded)}")
    if reasons:
        return None, reasons
    if not recorded:
        # Every run is honestly pre-PE-5: no official fetch was ever recorded, and
        # no generation may be manufactured for it.
        return None, []
    fetch_run_id = next(iter(recorded))
    if explicit is not None and int(explicit) != int(fetch_run_id):
        return None, [
            f"the declared official fetch run {int(explicit)} disagrees with every run's persisted "
            f"provenance ({[int(fetch_run_id)]}); a fetch the runs contradict is not their provenance"
        ]
    return int(fetch_run_id), []


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

    Two properties of the RUNS themselves are required as well, because a
    certificate describes a closed artifact:

    * every referenced run is ``complete``.  A ``running`` run's frozen set can
      still grow, so certifying it would certify a prediction artifact that is
      not yet what it will be;
    * at most one run per model family.  One family makes one claim per key, so
      two runs of the same family cannot both belong to one freeze, and the
      contract's ledger resolves a key's claims per family.

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
    incomplete = sorted(str(int(run["id"])) for run in runs if str(run.get("status")) != "complete")
    if incomplete:
        reasons.append(
            f"projection run(s) {incomplete} are not complete; a run that is still running can add frozen "
            "predictions, so its artifact is not yet the artifact a certificate would describe"
        )
    families = [str(run["model_family"]) for run in runs]
    duplicated = sorted({family for family in families if families.count(family) > 1})
    if duplicated:
        reasons.append(
            f"the freeze names the same model family twice: {duplicated}; one model family makes one claim "
            "per key, so a second run of it cannot belong to the same freeze"
        )

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

    fetch_run_id, fetch_reasons = _declared_official_fetch_run_id(runs, official_fetch_run_id)
    reasons.extend(fetch_reasons)
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
            # truthful legacy state -- but ONLY when nothing else about the runs
            # is wrong.  Legacy is a statement about missing provenance, not a
            # blanket that hides a run that is still open or a freeze that names
            # one model family twice; those are refusals, and a refusal the
            # legacy branch swallowed would never reach a reviewer.
            state = NOT_CERTIFIABLE if reasons else LEGACY_PROVENANCE
            return _record_freeze(
                conn, runs=runs, ordered=ordered, freeze_identity=freeze_identity,
                provenance_state=state, planning_event=planning_event,
                planning_cutoff=planning_cutoff, artifact_identity=artifact_identity,
                artifact_sha=artifact_sha, reasons=reasons, official_fetch_run_id=None,
                generation=None, element_ids=[], element_digest=None, element_count=None,
                certified_at=certified_at or utc_now(),
                closure=_closure_evidence(conn, ordered, None),
                decided_generation=None,
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
        closure=_closure_evidence(conn, ordered, generation_id),
        decided_generation=generation,
    )


#: The ``bootstrap_generations`` columns a certificate's every claim is read
#: from.  The fingerprint is the whole set, so a row edited between the decision
#: and the write cannot pass the closure check by moving a field the certificate
#: does not itself echo -- ``official_element_count`` and
#: ``acceptance_rule_version`` are recorded on the certificate and therefore
#: must be part of what has to hold still.
_GENERATION_CLOSURE_FIELDS = (
    "id",
    "accepted",
    "fetch_run_id",
    "captured_at",
    "official_element_count",
    "element_ids_sha256",
    "element_ids_json",
    "acceptance_rule_version",
)


def _generation_fingerprint(generation: Mapping[str, Any] | None) -> tuple[Any, ...] | None:
    """One generation's certification-relevant identity, in canonical form."""

    if generation is None:
        return None
    values: list[Any] = []
    for name in _GENERATION_CLOSURE_FIELDS:
        value = generation.get(name)
        if name in ("official_element_count",):
            values.append(None if value is None else int(value))
        elif name in ("id", "fetch_run_id", "accepted"):
            values.append(None if value is None else int(value))
        else:
            values.append(None if value is None else str(value))
    return tuple(values)


def _closure_evidence(
    conn: sqlite3.Connection, ordered: Sequence[int], generation_id: int | None
) -> tuple[Any, ...]:
    """Everything a certificate's claims are read from, as ONE observation.

    The digest of the frozen artifact and the run evidence it is certified with
    are read here as a single unit, and then read AGAIN inside the closure's own
    transaction.  A certificate may only be written when both reads agree, which
    is what makes hashing and closure atomic: the artifact the certificate
    describes is the artifact that existed at the instant it was written, not
    the one a decision read saw a moment earlier.

    The generation is re-read BY ID here rather than fingerprinted from whatever
    dictionary the decision happened to hold.  A certificate records the
    generation that certified it, so the row that has to be still is the row in
    the table; a fingerprint taken from a stale copy would happily certify a
    generation that had since been edited underneath it.
    """

    runs: list[tuple[Any, ...]] = []
    for run_id in sorted({int(value) for value in ordered}):
        row = conn.execute(
            "SELECT id, status, model_family, planning_event, data_cutoff FROM projection_runs WHERE id=?",
            (int(run_id),),
        ).fetchone()
        runs.append(
            (int(run_id), None)
            if row is None
            else (
                int(row["id"]),
                str(row["status"]),
                str(row["model_family"]),
                int(row["planning_event"]),
                str(row["data_cutoff"]),
            )
        )
    generation = None if generation_id is None else _bootstrap_generation(conn, int(generation_id))
    return (
        tuple(runs),
        _generation_fingerprint(generation),
        prediction_artifact_digest(conn, ordered),
    )


def _closure_difference(before: tuple[Any, ...], after: tuple[Any, ...]) -> str:
    """Which part of the certificate's evidence moved, in plain words."""

    labels = ("run evidence", "bootstrap generation", "frozen artifact digest")
    changed = [
        label for label, left, right in zip(labels, before, after) if left != right
    ]
    return "changed: " + ", ".join(changed) if changed else "no observable difference"


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
    closure: tuple[Any, ...],
    decided_generation: Mapping[str, Any] | None,
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
    # The provenance row and its run membership are ONE fact, so they are written
    # under one savepoint: both tables refuse UPDATE and DELETE, and a
    # half-written certificate could never be completed.  A SAVEPOINT, not a
    # transaction: the connection's transaction belongs to the CALLER, and
    # committing it here would publish whatever else the caller had in flight.
    conn.execute("SAVEPOINT pe5_record_freeze")
    try:
        # Re-read the evidence INSIDE the closure's own transaction, so the
        # recorded artifact digest, the run evidence and the certificate are one
        # atomic fact.  Another connection that put a frozen prediction, a run
        # status or a generation row in between the decision and this write is
        # caught here and the closure is refused, because the artifact it would
        # describe is not the artifact that existed when the decision was made.
        observed = _closure_evidence(
            conn, ordered, None if decided_generation is None else int(decided_generation["id"])
        )
        # The certificate is decided FROM a generation read and records THAT
        # generation.  Comparing the decision's own reading with the row that
        # exists now is what makes the two the same fact: a row retracted,
        # rejected or rebuilt in between would otherwise be certified from a
        # truth that has already been withdrawn.  The closure's own read alone
        # cannot see this, because by then it reads whatever the row says.
        decided_generation_fingerprint = _generation_fingerprint(decided_generation)
        if decided_generation_fingerprint != observed[1]:
            raise OutcomeLedgerError(
                f"freeze {freeze_identity} cannot be certified: the bootstrap generation it was decided "
                f"from ({decided_generation_fingerprint}) is not the generation that exists now "
                f"({observed[1]}); a certificate records the row that supplied the pool, so a row that "
                "moved is refused and nothing was recorded"
            )
        if observed[2] != artifact_sha:
            raise OutcomeLedgerError(
                f"freeze {freeze_identity} cannot be certified: its frozen artifact digested to "
                f"{artifact_sha} when the certificate was decided and to {observed[2]} when it would have "
                "been written; a freeze whose artifact is still moving is not closed, so nothing was recorded"
            )
        if observed != closure:
            raise OutcomeLedgerError(
                f"freeze {freeze_identity} cannot be certified: its evidence changed while the "
                "certificate was being written ("
                + _closure_difference(closure, observed)
                + "); hashing and closure are one atomic step, so nothing was recorded"
            )
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
    except BaseException:
        conn.execute("ROLLBACK TO pe5_record_freeze")
        conn.execute("RELEASE pe5_record_freeze")
        raise
    conn.execute("RELEASE pe5_record_freeze")

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
# The football world a read is resolved against
# ---------------------------------------------------------------------------

#: How a ledger resolved the pool, the clubs, the fixtures and the event
#: finality.  ``PINNED_EVIDENCE`` means every one of them came from versioned or
#: immutable evidence (the freeze's certified generation, the accepted
#: generation observable at the read, the raw archive).  ``CURRENT_STATE`` means
#: at least one of them came from a table that is refreshed in place, which a
#: read may only do when it claims no cutoff.  ``UNAVAILABLE`` means the pinned
#: evidence a historical read needs does not exist, and the honest answer is
#: that the evidence is unavailable rather than the current value of a mutable
#: table.
WORLD_PINNED = "PINNED_EVIDENCE"
WORLD_CURRENT = "CURRENT_STATE"
WORLD_UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True)
class WorldEvidence:
    """The football world a ledger read was resolved against, and from where."""

    source: str
    as_of: str | None
    moment: str
    pool_ids: frozenset[int] | None
    pool_basis: str
    clubs: Mapping[int, int]
    fixtures: Mapping[int, Mapping[str, Any]]
    fixture_events: frozenset[int]
    event_rows: Mapping[int, Mapping[str, Any]]
    evidence: tuple[str, ...]
    unavailable: tuple[str, ...]

    @property
    def historical(self) -> bool:
        return self.as_of is not None

    def club_of(self, player_id: int) -> int | None:
        value = self.clubs.get(int(player_id))
        return None if value is None else int(value)

    def fixtures_of(self, event: int) -> tuple[int, ...] | None:
        """The event's fixtures, or ``None`` when that schedule is unavailable."""

        if int(event) not in self.fixture_events:
            return None
        return tuple(
            sorted(
                int(fixture_id)
                for fixture_id, row in self.fixtures.items()
                if int(row.get("event") or 0) == int(event)
            )
        )

    def fixture_row(self, fixture_id: int) -> Mapping[str, Any] | None:
        return self.fixtures.get(int(fixture_id))

    def event_state(self, event: int) -> str:
        """The canonical finality, applied to THIS read's evidence.

        The rule is :func:`fpl_brain.planning.event_data_state`'s, unchanged --
        ``finished`` and ``data_checked`` on the event, plus the fixtures' own
        finished/started counts -- but the rows it is applied to are the pinned
        ones when the read has them.  A second policy is not invented here; the
        only difference is which evidence answers.
        """

        row = self.event_rows.get(int(event))
        if row is None:
            return planning_state.EVENT_STATE_UNKNOWN
        fixtures = [
            item for item in self.fixtures.values() if int(item.get("event") or 0) == int(event)
        ]
        finished = row.get("finished")
        data_checked = row.get("data_checked")
        if finished == 1 and data_checked == 1:
            return planning_state.EVENT_STATE_FINAL
        if any(int(item.get("started") or 0) == 1 and item.get("finished") != 1 for item in fixtures):
            return planning_state.EVENT_STATE_IN_PROGRESS
        if fixtures and all(item.get("finished") == 1 for item in fixtures):
            return planning_state.EVENT_STATE_PROVISIONAL
        return planning_state.EVENT_STATE_SCHEDULED


def _archive_records(archive_root: Any, *, moment: str) -> list[dict[str, Any]]:
    """Every archived capture observable at or before ``moment``."""

    from . import raw_archive

    records = [
        dict(record)
        for record in raw_archive.load_manifest(archive_root)
        if _observed_at_or_before(str(record.get("observed_at") or ""), moment)
    ]
    records.sort(key=lambda item: (str(item.get("observed_at") or ""), str(item.get("capture_id") or "")))
    return records


def _observed_at_or_before(observed_at: str, moment: str) -> bool:
    left, right = parse_utc(observed_at), parse_utc(moment)
    if left is not None and right is not None:
        return left <= right
    return str(observed_at) <= str(moment)


def _archive_record_for(
    records: Sequence[Mapping[str, Any]], source: str, *, event: int | None = None
) -> Mapping[str, Any] | None:
    """The latest archived capture of ``source`` at/before the moment, for the event."""

    matching = [
        record
        for record in records
        if str(record.get("source")) == str(source)
        and (event is None or record.get("event") is None or int(record["event"]) == int(event))
    ]
    if not matching:
        return None
    return max(matching, key=lambda item: (str(item.get("observed_at") or ""), str(item.get("capture_id") or "")))


def _archived_payload(archive_root: Any, record: Mapping[str, Any]) -> Any | None:
    """The archived bytes, verified against their recorded digest before use.

    An archive record whose blob is missing or no longer hashes to its recorded
    digest is not evidence, so it resolves to ``None`` (unavailable) rather than
    being parsed as if the bytes were intact.
    """

    from . import raw_archive

    if not raw_archive.verify_archived_blob(archive_root, record):
        return None
    root = Path(str(archive_root))
    base = root if root.name == raw_archive.ARCHIVE_DIRNAME else root / raw_archive.ARCHIVE_DIRNAME
    try:
        return json.loads((base / str(record["relative_path"])).read_bytes())
    except (OSError, TypeError, ValueError):
        return None


def _pinned_items(payload: Any, key: str) -> list[Any]:
    """The named array from an archived payload, in either documented shape."""

    if isinstance(payload, Mapping):
        value = payload.get(key)
        return list(value) if isinstance(value, list) else []
    if key == "fixtures" and isinstance(payload, list):
        return list(payload)
    return []


def _pinned_evidence(
    archive_root: Any, *, moment: str, events: Sequence[int]
) -> dict[str, Any]:
    """The pinned world from the raw archive: pool, clubs, fixtures, finality."""

    records = _archive_records(archive_root, moment=moment)
    pooled: dict[str, Any] = {
        "elements": {},
        "event_rows": {},
        "fixtures": {},
        "fixture_events": set(),
        "evidence": [],
        "unavailable": [],
    }
    bootstrap = _archive_record_for(records, "bootstrap_static")
    if bootstrap is None:
        pooled["unavailable"].append(
            f"no archived bootstrap_static capture at or before {moment}: the official pool and the "
            "official club membership are unavailable as of that instant"
        )
    else:
        payload = _archived_payload(archive_root, bootstrap)
        if payload is None:
            pooled["unavailable"].append(
                f"archived bootstrap_static capture {bootstrap.get('capture_id')} has no intact bytes, "
                "so it is not evidence"
            )
        else:
            for element in _pinned_items(payload, "elements"):
                if isinstance(element, Mapping) and element.get("id") is not None:
                    pooled["elements"][int(element["id"])] = dict(element)
            for row in _pinned_items(payload, "events"):
                if isinstance(row, Mapping) and row.get("id") is not None:
                    pooled["event_rows"][int(row["id"])] = dict(row)
            pooled["evidence"].append(str(bootstrap.get("capture_id")))
    for event in sorted({int(value) for value in events}):
        record = _archive_record_for(records, "fixtures", event=event)
        if record is None:
            pooled["unavailable"].append(
                f"no archived fixtures capture at or before {moment} for event {event}: the fixture "
                "schedule is unavailable as of that instant"
            )
            continue
        payload = _archived_payload(archive_root, record)
        if payload is None:
            pooled["unavailable"].append(
                f"archived fixtures capture {record.get('capture_id')} has no intact bytes, so it is "
                "not evidence"
            )
            continue
        rows = [
            dict(row)
            for row in _pinned_items(payload, "fixtures")
            if isinstance(row, Mapping) and row.get("id") is not None
        ]
        for row in rows:
            if int(row.get("event") or 0) == int(event):
                pooled["fixtures"][int(row["id"])] = row
        pooled["fixture_events"].add(int(event))
        pooled["evidence"].append(str(record.get("capture_id")))
    return pooled


def resolve_world_evidence(
    conn: sqlite3.Connection,
    *,
    provenance: Mapping[str, Any],
    as_of: str | None = None,
    archive_root: Any = None,
    events: Sequence[int] = (),
) -> WorldEvidence:
    """Resolve pool, club, fixtures and event finality for one ledger read.

    Three rules, in order:

    1. **Pinned first.**  The freeze's certified generation is pinned ON the
       freeze, so it always wins for the pool; an accepted generation observable
       at the read's moment is pinned by its own capture time; and the raw
       archive supplies the pool, the clubs, the fixtures and the events as they
       were at or before the read's moment.
    2. **Mutable tables only for a read that claims no cutoff.**  ``as_of=None``
       asks "what is true now", and now is exactly what those tables hold.  A
       read that names a cutoff may NOT consult them: a fixture that finished
       today was not finished at a historical cutoff, and a player who is
       inactive today was not inactive then.
    3. **Otherwise unavailable.**  A historical read with no pinned evidence
       reports that the evidence is unavailable -- never the current value of a
       mutable table, which is precisely the substitution "missing history is
       not zero" forbids.
    """

    historical = as_of is not None
    moment = str(as_of) if historical else utc_now()
    event_ids = sorted({int(value) for value in events})
    evidence: list[str] = []
    unavailable: list[str] = []
    current_used = False

    pinned = (
        _pinned_evidence(archive_root, moment=moment, events=event_ids)
        if archive_root is not None
        else {"elements": {}, "event_rows": {}, "fixtures": {}, "fixture_events": set(),
              "evidence": [], "unavailable": []}
    )
    if archive_root is None and historical:
        unavailable.append(
            f"no raw archive was supplied for the read at {moment}: pool, club, fixture and finality "
            "evidence for a historical read must come from archived captures"
        )
    evidence.extend(pinned["evidence"])
    unavailable.extend(pinned["unavailable"])

    # -- pool ---------------------------------------------------------------
    pool_ids: frozenset[int] | None = None
    pool_basis = ""
    if provenance.get("provenance_state") == GENERATION_CERTIFIED and provenance.get(
        "bootstrap_element_ids"
    ):
        pool_ids = frozenset(int(pid) for pid in provenance["bootstrap_element_ids"])
        pool_basis = (
            f"the freeze's certified bootstrap generation {provenance.get('bootstrap_generation_id')}"
        )
    if pool_ids is None:
        generation = accepted_generation_at(conn, observed_at=moment)
        if generation is not None:
            try:
                ids = sorted({int(pid) for pid in json.loads(generation.get("element_ids_json") or "[]")})
            except (TypeError, ValueError):
                ids = []
            if ids:
                pool_ids = frozenset(ids)
                pool_basis = (
                    f"accepted bootstrap generation {generation.get('id')} observable at {moment}"
                )
                evidence.append(f"bootstrap_generation:{int(generation['id'])}")
    if pool_ids is None and pinned["elements"]:
        pool_ids = frozenset(int(pid) for pid in pinned["elements"])
        pool_basis = f"the archived bootstrap_static element set observable at {moment}"
    if pool_ids is None and not historical:
        ids = sorted(int(pid) for pid in repo.active_player_ids(conn))
        pool_ids = frozenset(ids)
        pool_basis = "the current active official pool"
        current_used = True
    if pool_ids is None:
        unavailable.append(
            f"no accepted official-player generation and no archived bootstrap capture is observable at "
            f"{moment}, so the pool that supplied this freeze cannot be read from evidence"
        )

    # -- clubs --------------------------------------------------------------
    clubs: dict[int, int] = {}
    for player_id, element in pinned["elements"].items():
        team = element.get("team")
        if team is not None:
            clubs[int(player_id)] = int(team)
    if not historical:
        for row in conn.execute(
            "SELECT id, team_id FROM players WHERE team_id IS NOT NULL ORDER BY id"
        ).fetchall():
            if int(row["id"]) not in clubs:
                clubs[int(row["id"])] = int(row["team_id"])
                current_used = True

    # -- fixtures and event rows -------------------------------------------
    fixtures: dict[int, dict[str, Any]] = dict(pinned["fixtures"])
    fixture_events: set[int] = set(pinned["fixture_events"])
    event_rows: dict[int, dict[str, Any]] = dict(pinned["event_rows"])
    if not historical:
        for event in event_ids:
            if event in fixture_events:
                continue
            found = False
            for row in conn.execute(
                "SELECT * FROM fixtures WHERE event=? AND id > 0 ORDER BY id", (int(event),)
            ).fetchall():
                fixtures[int(row["id"])] = dict(row)
                found = True
            if found:
                fixture_events.add(int(event))
                current_used = True
        for event in event_ids:
            if event in event_rows:
                continue
            row = conn.execute("SELECT * FROM events WHERE id=?", (int(event),)).fetchone()
            if row is not None:
                event_rows[int(event)] = dict(row)
                current_used = True

    if not event_rows and event_ids:
        unavailable.append(
            f"the official event rows at {moment} are unavailable, so no event's finality can be judged "
            "from evidence"
        )

    pinned_complete = bool(fixtures) and bool(event_rows) and all(
        event in fixture_events and event in event_rows for event in event_ids
    )
    if current_used:
        source = WORLD_CURRENT
    elif historical and not pinned_complete:
        source = WORLD_UNAVAILABLE
    elif pool_ids is None and not pinned_complete:
        source = WORLD_UNAVAILABLE
    else:
        source = WORLD_PINNED
    return WorldEvidence(
        source=source,
        as_of=as_of,
        moment=moment,
        pool_ids=pool_ids,
        pool_basis=pool_basis,
        clubs=clubs,
        fixtures=fixtures,
        fixture_events=frozenset(fixture_events),
        event_rows=event_rows,
        evidence=tuple(dict.fromkeys(evidence)),
        unavailable=tuple(dict.fromkeys(unavailable)),
    )


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
    prediction_artifact_sha256: str = ""
    artifact_verified: bool = False
    world_source: str = ""
    world_pool_basis: str = ""
    world_evidence: tuple[str, ...] = field(default_factory=tuple)
    world_unavailable: tuple[str, ...] = field(default_factory=tuple)

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
            "prediction_artifact_sha256": self.prediction_artifact_sha256,
            "artifact_verified": bool(self.artifact_verified),
            "world_source": self.world_source,
            "world_pool_basis": self.world_pool_basis,
            "world_evidence": list(self.world_evidence),
            "world_unavailable": list(self.world_unavailable),
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
    "prediction_artifact_sha256",
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
    "world_source",
    "world_evidence",
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
    archive_root: Any = None,
) -> RealityLedger:
    """Relate one frozen prediction generation to the official reality.

    Deterministic by construction: the population is the union of the freeze's
    frozen prediction keys and the observable outcome keys, both sorted; every
    read has an explicit ORDER BY; and the digests are taken over content
    fields, never over surrogate row ids.  Building the same ledger twice, on
    two stores whose rows were written in different orders, yields identical
    rows, identical counts and identical digests.

    Three things are proven rather than assumed:

    * the frozen artifact still digests to the digest the freeze was certified
      with.  A store whose runs grew after certification is a different artifact
      wearing a certificate for the old one, so the ledger refuses to build
      instead of reporting a prediction set nobody certified;
    * the pool, the clubs, the fixtures and the event finality come from pinned
      evidence -- the freeze's certified generation, the accepted generation
      observable at the read, or the raw archive -- and a read that names a
      cutoff never falls back to a table that is refreshed in place;
    * nothing is dropped.  An excluded key is RETURNED with its reason code, so
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
    if not runs:
        raise OutcomeLedgerError(f"freeze {freeze_identity} has no recorded runs")
    incomplete = sorted(
        str(int(row["id"]))
        for row in conn.execute(
            "SELECT id, status FROM projection_runs WHERE id IN ("
            + ",".join("?" for _ in runs)
            + ")",
            tuple(sorted(runs)),
        ).fetchall()
        if str(row["status"]) != "complete"
    )
    if incomplete:
        raise OutcomeLedgerError(
            f"freeze {freeze_identity} includes projection run(s) {incomplete} that are not complete; a "
            "certified freeze is readable only while every run it certifies is closed"
        )
    artifact_sha = prediction_artifact_digest(conn, sorted(runs))
    recorded_artifact = str(provenance.get("prediction_artifact_sha256") or "")
    if artifact_sha != recorded_artifact:
        raise OutcomeLedgerError(
            f"freeze {freeze_identity} was certified over artifact {recorded_artifact}, but its frozen "
            f"predictions now digest to {artifact_sha}; the frozen artifact changed after certification, so "
            "this freeze no longer certifies what is stored"
        )
    predictions = _freeze_predictions(conn, sorted(runs), grain)
    events = _target_events(conn, provenance, predictions)
    world = resolve_world_evidence(
        conn, provenance=provenance, as_of=as_of, archive_root=archive_root, events=events
    )
    captures = observation_captures(conn, events=events, as_of=as_of)

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
            world=world,
            artifact_sha=artifact_sha,
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
        prediction_artifact_sha256=artifact_sha,
        artifact_verified=True,
        world_source=world.source,
        world_pool_basis=world.pool_basis,
        world_evidence=world.evidence,
        world_unavailable=world.unavailable,
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
    world: WorldEvidence,
    artifact_sha: str,
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
    # Every distinct prediction is retained.  De-duplicating on the kind alone
    # would silently drop one family's claim whenever two families freeze the
    # same kind for the same key, so the identity is the whole claim: the kind
    # AND the run that issued it.
    distinct: dict[tuple, Mapping[str, Any]] = {}
    for row in matching:
        run = runs[int(row["projection_run_id"])]
        key = (
            str(row["kind"]),
            str(run["model_family"]),
            str(run["model_version"]),
            int(row["projection_run_id"]),
        )
        distinct.setdefault(key, row)
    matching = [distinct[key] for key in sorted(distinct)]
    context = _KeyContext(
        conn=conn, grain=grain, event=event, player_id=player_id, fixture_id=fixture_id,
        runs=runs, captures=captures, provenance=provenance, world=world, artifact_sha=artifact_sha,
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
        world: WorldEvidence,
        artifact_sha: str,
    ) -> None:
        self.conn = conn
        self.grain = grain
        self.event = event
        self.player_id = player_id
        self.fixture_id = fixture_id
        self.runs = runs
        self.provenance = provenance
        self.captures = captures
        self.world = world
        self.artifact_sha = artifact_sha
        # The club is pinned evidence too.  The observation's own recorded club
        # is the strongest source (it is what the source stated when it was
        # read) and the archived element payload is the other one; today's
        # players table is consulted only by a read that claims no cutoff.
        recorded_clubs = {
            int(row["team_id"])
            for row in captures
            if int(row["player_id"]) == player_id
            and int(row["event"]) == event
            and row.get("team_id") is not None
        }
        if len(recorded_clubs) == 1:
            self.team_id = recorded_clubs.pop()
        elif len(recorded_clubs) > 1:
            # Two retained observations disagree about the club.  That is an
            # ambiguity, not a number to pick a winner from.
            self.team_id = None
        else:
            self.team_id = world.club_of(player_id)
        # Resolved ONCE per key, from the accepted finality definition applied to
        # THIS read's evidence, and reported on every row the key emits: the
        # ledger must state the finalization the outcome was judged under, not
        # only the verdict.
        self.event_state = world.event_state(event)

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

        pool = self.world.pool_ids
        if pool is None:
            return _outcome(
                INPUT_EVIDENCE_UNAVAILABLE,
                "the official-player pool this freeze is judged against is unavailable in the evidence "
                f"read at {self.world.moment}; the current pool is not evidence about a past cutoff",
            )
        if int(self.player_id) not in pool:
            return _outcome(
                PLAYER_NOT_IN_OFFICIAL_POOL,
                f"player {self.player_id} is not in {(self.world.pool_basis or 'the official pool')}",
            )
        state = self.event_state
        if str(state) == planning_state.EVENT_STATE_UNKNOWN:
            return _outcome(
                INPUT_EVIDENCE_UNAVAILABLE,
                f"the official state of event {self.event} is unavailable in the evidence read at "
                f"{self.world.moment}, so its finality cannot be judged",
            )
        if str(state) != "FINAL":
            return _outcome(OUTCOME_NOT_FINALISED, f"event {self.event} is {state}")

        schedule = self.world.fixtures_of(self.event)
        if schedule is None:
            return _outcome(
                INPUT_EVIDENCE_UNAVAILABLE,
                f"the fixture schedule of event {self.event} is unavailable in the evidence read at "
                f"{self.world.moment}, so no club's fixtures can be resolved",
            )
        if self.team_id is None:
            return _outcome(
                INPUT_EVIDENCE_UNAVAILABLE,
                f"the club player {self.player_id} belonged to at {self.world.moment} is unavailable in "
                "the evidence read, and a club is what a fixture is joined through",
            )
        fixtures = sorted(
            fixture_id
            for fixture_id in schedule
            if int((self.world.fixture_row(fixture_id) or {}).get("team_h") or 0) == int(self.team_id)
            or int((self.world.fixture_row(fixture_id) or {}).get("team_a") or 0) == int(self.team_id)
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

        unplayed = [
            fid
            for fid in fixtures
            if int((self.world.fixture_row(fid) or {}).get("finished") or 0) != 1
        ]
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
        if self.world.historical:
            # A row in today's player_gameweeks is not evidence about what was
            # observable before the read's cutoff: the schedule rows are
            # refreshed in place.  A historical read therefore has no placeholder
            # evidence and says so, instead of reading today's table backwards.
            return False
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
            "prediction_artifact_sha256": str(self.artifact_sha),
            "grain": self.grain,
            "event": self.event,
            "player_id": self.player_id,
            "fixture_id": self.fixture_id,
            "team_id": self.team_id,
            "opponent_team_id": None if capture is None else capture.get("opponent_team_id"),
            "event_time": None if capture is None else capture.get("event_time"),
            "official_final_at": None if capture is None else capture.get("official_final_at"),
            "observation_time": None if capture is None else capture.get("captured_at"),
            "observation_state": None if capture is None else capture.get("observation_state"),
            "observation_capture_digests": list(outcome.get("digests") or ()),
            "observation_sources": [dict(row) for row in outcome.get("sources") or ()],
            "world_source": str(self.world.source),
            "world_evidence": list(self.world.evidence),
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

    ordered = sorted(captures, key=lambda row: str(row.get("capture_digest") or ""))
    values: dict[str, Any] = {}
    for name in SUNNABLE_OUTCOME_FIELDS:
        stated = [row.get(name) for row in ordered]
        if any(value is None for value in stated):
            values[name] = None
        else:
            values[name] = sum(int(value) for value in stated)
    return values, tuple(str(row.get("capture_digest") or "") for row in ordered)


def _player_record(conn: sqlite3.Connection, player_id: int) -> dict[str, Any] | None:
    """The player row as the CAPTURE path reads it: at capture time, now is the truth."""

    row = conn.execute("SELECT * FROM players WHERE id=?", (int(player_id),)).fetchone()
    return dict(row) if row is not None else None
