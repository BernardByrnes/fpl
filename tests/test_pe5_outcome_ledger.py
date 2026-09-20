"""PE-5 — outcome capture and prediction-to-reality ledger contracts.

The authoritative minimum is the 32 hard contracts in
``docs/prediction-engine/PE-5-OUTCOME-CAPTURE-PREDICTION-TO-REALITY-LEDGER.md``.
They are grouped below in the contract's own order, and the docstring of each
test names the contract it pins.

TWO THINGS THESE TESTS ARE NOT
------------------------------
They do not migrate or read the authoritative live database: every case builds
its own temporary SQLite file.  And they do not re-tune anything: PE-5 records
facts and provenance, so where a contract is already satisfied by existing
storage (the scheduled-placeholder signature, the PE-1 causal boundary, the
walk-forward reason codes) the test PINS that reuse rather than restating it.

The synthetic world is deliberately small and its timestamps are derived from
``utc_now()`` rather than hard-coded, so the causality cases stay meaningful
whenever the suite runs.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import timedelta
from pathlib import Path

import pytest

from fpl_brain import analytics
from fpl_brain import historical_observations as historical
from fpl_brain import outcome_ledger as ol
from fpl_brain import planning
from fpl_brain import repositories as repo
from fpl_brain.database import connect_database
from fpl_brain.ingest_provenance import element_id_sha256
from fpl_brain.models import (
    EventRecord,
    FixtureRecord,
    PlayerGameweekRecord,
    PlayerRecord,
    PositionRecord,
    TeamRecord,
)
from fpl_brain.utils import parse_utc, utc_now

# ---------------------------------------------------------------------------
# Synthetic world
# ---------------------------------------------------------------------------

NOW = parse_utc(utc_now())


def _t(days: float = 0.0) -> str:
    """An ISO-8601 UTC instant ``days`` before the run, with no microseconds."""

    return (NOW - timedelta(days=days)).replace(microsecond=0).isoformat().replace("+00:00", "Z")


TEAM_A, TEAM_B, TEAM_C, TEAM_D = 1, 2, 3, 4
DEF, MID = 2, 3
POSITIONS = ((1, "GKP"), (2, "DEF"), (3, "MID"), (4, "FWD"))

#: Team A plays TWICE in the target event (a real double gameweek); teams B and
#: C play once; team D has no fixture at all, so its player's week is a blank.
DGW_PLAYER = 10          # team A, two fixtures
SGW_A_PLAYER = 11        # team A, also two fixtures
SGW_B_PLAYER = 20        # team B, one fixture
BLANK_PLAYER = 30        # team D, no fixture -> blank, not missing
NO_OUTCOME_PLAYER = 40   # team B, played fixture, no stored observation at all
PLACEHOLDER_PLAYER = 50  # team B, played fixture, only a scheduled placeholder row
INACTIVE_PLAYER = 60     # team C, not in the official pool

FIXTURE_1 = 400          # event 4, team A v team B
FIXTURE_2 = 401          # event 4, team A v team C  (the DGW second leg)
FIXTURE_OTHER = 402      # event 3, team B v team D  (history, not the target event)
FIXTURE_PARTIAL = 500    # event 5, finished but the EVENT is not final
FIXTURE_UNPLAYED = 501   # event 5, not started

KICKOFF_1 = _t(8)
KICKOFF_2 = _t(7)
EVENT_ROWS_WRITTEN = _t(6)   # when the repository recorded official finality
CUTOFF = _t(9)               # the prediction data cutoff
GENERATION_CAPTURED = _t(10)  # the accepted generation, before the cutoff
CAPTURE_1 = _t(5)
CAPTURE_2 = _t(4)
AS_OF = _t(5)

EVENT = 4
EVENT_PARTIAL = 5
OTHER_EVENT = 3

#: The official element ids the accepted generation certifies: every synthetic
#: player that IS in the official pool.  ``INACTIVE_PLAYER`` is deliberately
#: absent, which is the pinned statement of "he was not in the pool", as opposed
#: to today's ``players.is_active`` flag.
SYNTHETIC_POOL = (
    DGW_PLAYER, SGW_A_PLAYER, SGW_B_PLAYER, BLANK_PLAYER, NO_OUTCOME_PLAYER, PLACEHOLDER_PLAYER,
)

FAMILY_POINTS = analytics.BASELINE_MODEL_FAMILY
FAMILY_MINUTES = analytics.MINUTES_MODEL_FAMILY
KIND_POINTS = analytics.EP_NEXT_KIND
KIND_MINUTES = analytics.MINUTES_V1_KIND


def _players(conn) -> None:
    repo.upsert_players(
        conn,
        [
            PlayerRecord(id=DGW_PLAYER, web_name="DGW", team_id=TEAM_A, element_type=MID),
            PlayerRecord(id=SGW_A_PLAYER, web_name="A2", team_id=TEAM_A, element_type=DEF),
            PlayerRecord(id=SGW_B_PLAYER, web_name="B1", team_id=TEAM_B, element_type=MID),
            PlayerRecord(id=BLANK_PLAYER, web_name="BLK", team_id=TEAM_D, element_type=MID),
            PlayerRecord(id=NO_OUTCOME_PLAYER, web_name="NIL", team_id=TEAM_B, element_type=DEF),
            PlayerRecord(id=PLACEHOLDER_PLAYER, web_name="PH", team_id=TEAM_B, element_type=DEF),
            PlayerRecord(id=INACTIVE_PLAYER, web_name="OLD", team_id=TEAM_C, element_type=DEF),
        ],
    )
    conn.execute("UPDATE players SET is_active=0 WHERE id=?", (INACTIVE_PLAYER,))


def _world(conn, *, fixtures: bool = True, events: bool = True) -> None:
    repo.upsert_teams(conn, [TeamRecord(id=tid, name=f"T{tid}") for tid in (TEAM_A, TEAM_B, TEAM_C, TEAM_D)])
    repo.upsert_positions(conn, [PositionRecord(id=pid, singular_name_short=name) for pid, name in POSITIONS])
    _players(conn)
    if events:
        repo.upsert_events(
            conn,
            [
                EventRecord(id=OTHER_EVENT, finished=1, data_checked=1, deadline_time=_t(12), raw_json={}),
                EventRecord(id=EVENT, finished=1, data_checked=1, deadline_time=_t(9), raw_json={}),
                # All its fixtures are over, but the event is NOT officially
                # finalised -- the real partial/provisional review state.
                EventRecord(id=EVENT_PARTIAL, finished=0, data_checked=0, deadline_time=_t(2), raw_json={}),
            ],
            updated_at=EVENT_ROWS_WRITTEN,
        )
    if fixtures:
        repo.upsert_fixtures(
            conn,
            [
                FixtureRecord(id=FIXTURE_OTHER, event=OTHER_EVENT, team_h=TEAM_B, team_a=TEAM_D,
                              started=1, finished=1, kickoff_time=_t(13), raw_json={}),
                FixtureRecord(id=FIXTURE_1, event=EVENT, team_h=TEAM_A, team_a=TEAM_B,
                              started=1, finished=1, kickoff_time=KICKOFF_1, raw_json={}),
                FixtureRecord(id=FIXTURE_2, event=EVENT, team_h=TEAM_A, team_a=TEAM_C,
                              started=1, finished=1, kickoff_time=KICKOFF_2, raw_json={}),
                FixtureRecord(id=FIXTURE_PARTIAL, event=EVENT_PARTIAL, team_h=TEAM_A, team_a=TEAM_B,
                              started=1, finished=1, kickoff_time=_t(6.5), raw_json={}),
                FixtureRecord(id=FIXTURE_UNPLAYED, event=EVENT_PARTIAL, team_h=TEAM_A, team_a=TEAM_C,
                              started=0, finished=0, kickoff_time=_t(-1), raw_json={}),
            ],
        )


def _fixture_kickoff(conn, fixture_id: int) -> str:
    row = conn.execute("SELECT kickoff_time FROM fixtures WHERE id=?", (fixture_id,)).fetchone()
    return str(row[0])


def _freeze(
    conn,
    *,
    family: str,
    version: str,
    kind: str,
    values: dict[int, float],
    event: int = EVENT,
    cutoff: str = CUTOFF,
    official_run_ids=None,
    config_hash: str | None = "cfg",
    random_seed: int | None = 7,
    fixture_of: dict[int, int] | None = None,
    finish: bool = True,
) -> int:
    """Freeze one transparent claim per player, exactly as a model run would."""

    run_id = analytics.create_projection_run(
        conn,
        model_family=family,
        model_version=version,
        planning_event=event,
        planning_context_hash="pe5-context",
        data_cutoff=cutoff,
        scouting_cutoff=None,
        official_run_ids={} if official_run_ids is None else official_run_ids,
        config_hash=config_hash,
        random_seed=random_seed,
        deadline_status="PRE_DEADLINE",
    )
    for player_id, value in sorted(values.items()):
        analytics.freeze_prediction(
            conn,
            run_id,
            kind=kind,
            player_id=player_id,
            event=event,
            fixture_id=None if fixture_of is None else fixture_of[player_id],
            payload={"value": value, "expected_minutes": value},
            model_version=version,
        )
    if finish:
        analytics.finish_projection_run(conn, run_id, "complete")
    return run_id


def _archive(tmp_path, *, source, observed_at, payload, event=None, run_id=1, raw_dir=None):
    """Archive one payload as the immutable evidence a historical read needs."""

    from fpl_brain import raw_archive

    root = Path(raw_dir) if raw_dir is not None else Path(tmp_path) / "raw"
    body = json.dumps(payload, sort_keys=True).encode("utf-8")
    record = raw_archive.archive_raw_capture(
        root, source=source, observed_at=observed_at, body=body, event=event, run_id=run_id
    )
    return root, record


def _archive_world(tmp_path, *, observed_at, fixture_events=(EVENT,), event_rows=None, raw_dir=None):
    """The pinned world at ``observed_at``: the pool, the clubs and the fixtures.

    Exactly the shapes the fetch layer archives: ``bootstrap_static`` carries the
    elements (with their clubs), the events and the teams; ``fixtures`` carries
    the event's schedule with its published status.
    """

    root = Path(raw_dir) if raw_dir is not None else None
    root, _ = _archive(
        tmp_path,
        source="bootstrap_static",
        observed_at=observed_at,
        payload={
            "elements": [
                {"id": DGW_PLAYER, "team": TEAM_A, "status": "a"},
                {"id": SGW_A_PLAYER, "team": TEAM_A, "status": "a"},
                {"id": SGW_B_PLAYER, "team": TEAM_B, "status": "a"},
                {"id": BLANK_PLAYER, "team": TEAM_D, "status": "a"},
                {"id": NO_OUTCOME_PLAYER, "team": TEAM_B, "status": "a"},
                {"id": PLACEHOLDER_PLAYER, "team": TEAM_B, "status": "a"},
                {"id": INACTIVE_PLAYER, "team": TEAM_C, "status": "a"},
            ],
            "events": event_rows
            if event_rows is not None
            else [
                {"id": OTHER_EVENT, "finished": 1, "data_checked": 1},
                {"id": EVENT, "finished": 1, "data_checked": 1},
            ],
            "teams": [{"id": tid, "name": f"T{tid}"} for tid in (TEAM_A, TEAM_B, TEAM_C, TEAM_D)],
        },
        raw_dir=root,
        run_id=1,
    )
    schedule = [
        {"id": FIXTURE_1, "event": EVENT, "team_h": TEAM_A, "team_a": TEAM_B,
         "kickoff_time": KICKOFF_1, "started": 1, "finished": 1},
        {"id": FIXTURE_2, "event": EVENT, "team_h": TEAM_A, "team_a": TEAM_C,
         "kickoff_time": KICKOFF_2, "started": 1, "finished": 1},
    ]
    for event in fixture_events:
        rows = [row for row in schedule if row["event"] == event]
        root, _ = _archive(
            tmp_path, source="fixtures", observed_at=observed_at, payload=rows, event=event,
            run_id=1, raw_dir=root,
        )
    return root


def _generation(
    conn,
    *,
    fetch_run_id: int | None,
    captured_at: str = GENERATION_CAPTURED,
    accepted: bool = True,
    element_ids=SYNTHETIC_POOL,
) -> int:
    return repo.record_bootstrap_generation(
        conn,
        captured_at=captured_at,
        accepted=accepted,
        official_element_count=len(element_ids),
        parsed_count=len(element_ids),
        persisted_count=len(element_ids),
        element_ids=list(element_ids),
        element_ids_sha256=element_id_sha256(list(element_ids)),
        acceptance_rule="pe5-test",
        acceptance_rule_version="BOOTSTRAP_GENERATION_ACCEPTANCE v1",
        fetch_run_id=fetch_run_id,
    )


def _fetch_run(conn) -> int:
    return int(repo.create_fetch_run(conn, "fetch_fpl"))


def _capture(
    conn,
    *,
    player_id: int,
    fixture_id: int,
    captured_at: str | None = CAPTURE_1,
    minutes: int = 90,
    total_points: int = 6,
    bonus: int | None = 1,
    bps: int | None = 30,
    starts: int = 1,
    source_name: str = "player_gameweeks_final",
    grain: str = ol.GRAIN_PLAYER_FIXTURE,
    event: int = EVENT,
    official_final_at: str | None = EVENT_ROWS_WRITTEN,
    backfill: bool = False,
    source_identity: str | None = None,
    source_payload_sha256: str | None = None,
    archive_capture_id: str | None = None,
    archive_root: object = None,
    fetch_run_id: int | None = None,
    supersedes_capture_id: int | None = None,
    correction_reason: str | None = None,
    team_id: int | None = None,
    opponent_team_id: int | None = None,
    event_time: str | None = None,
    **extra,
):
    """One official player-fixture observation, with every required field stated.

    A field is stated as ``None`` to model "the source did not supply it", which
    is deliberately different from stating zero.
    """

    fields = {
        "minutes": minutes,
        "starts": starts,
        "total_points": total_points,
        "goals_scored": extra.pop("goals_scored", 1),
        "assists": extra.pop("assists", 0),
        "clean_sheets": extra.pop("clean_sheets", 0),
        "goals_conceded": extra.pop("goals_conceded", 1),
        "saves": extra.pop("saves", 0),
        "bonus": bonus,
        "bps": bps,
        "yellow_cards": extra.pop("yellow_cards", 0),
        "red_cards": extra.pop("red_cards", 0),
        "penalties_saved": extra.pop("penalties_saved", 0),
        "penalties_missed": extra.pop("penalties_missed", 0),
        "own_goals": extra.pop("own_goals", 0),
        "defensive_contribution": extra.pop("defensive_contribution", 0),
    }
    fields.update(extra)
    return ol.capture_observation(
        conn,
        grain=grain,
        event=event,
        player_id=player_id,
        fixture_id=fixture_id,
        fields=fields,
        source_name=source_name,
        captured_at=captured_at,
        official_final_at=official_final_at,
        backfill=backfill,
        source_identity=source_identity,
        source_payload_sha256=source_payload_sha256,
        archive_capture_id=archive_capture_id,
        archive_root=archive_root,
        fetch_run_id=fetch_run_id,
        supersedes_capture_id=supersedes_capture_id,
        correction_reason=correction_reason,
        team_id=team_id,
        opponent_team_id=opponent_team_id,
        event_time=event_time,
    )


def _capture_all(conn, *, captured_at: str = CAPTURE_1) -> None:
    """The full official outcome of event 4, per fixture, for the DGW pair."""

    _capture(conn, player_id=DGW_PLAYER, fixture_id=FIXTURE_1, captured_at=captured_at,
             minutes=90, total_points=6, bonus=1, bps=30)
    _capture(conn, player_id=DGW_PLAYER, fixture_id=FIXTURE_2, captured_at=captured_at,
             minutes=62, total_points=3, bonus=0, bps=19)
    _capture(conn, player_id=SGW_A_PLAYER, fixture_id=FIXTURE_1, captured_at=captured_at,
             minutes=90, total_points=2, bonus=0, bps=12)
    _capture(conn, player_id=SGW_A_PLAYER, fixture_id=FIXTURE_2, captured_at=captured_at,
             minutes=0, total_points=0, bonus=0, bps=0, starts=0,
             goals_scored=0, assists=0, clean_sheets=0, goals_conceded=0)
    _capture(conn, player_id=SGW_B_PLAYER, fixture_id=FIXTURE_1, captured_at=captured_at,
             minutes=90, total_points=9, bonus=3, bps=41)
    _capture(conn, player_id=PLACEHOLDER_PLAYER, fixture_id=FIXTURE_1, captured_at=captured_at,
             minutes=45, total_points=1, bonus=0, bps=8, starts=1)


def _certified_freeze(conn, *, values=None, with_minutes_run: bool = True):
    """A generation-certified freeze with one event-grain and one fixture-grain run."""

    fetch = _fetch_run(conn)
    generation_id = _generation(conn, fetch_run_id=fetch)
    claims = values if values is not None else {
        DGW_PLAYER: 8.0, SGW_A_PLAYER: 3.0, SGW_B_PLAYER: 5.0,
        BLANK_PLAYER: 4.0, NO_OUTCOME_PLAYER: 2.0, PLACEHOLDER_PLAYER: 3.0,
    }
    runs = [
        _freeze(
            conn,
            family=FAMILY_POINTS,
            version=analytics.BASELINE_MODEL_VERSION,
            kind=KIND_POINTS,
            values=claims,
            official_run_ids={"fetch": {"run_id": fetch}},
        )
    ]
    if with_minutes_run:
        runs.append(
            _freeze(
                conn,
                family=FAMILY_MINUTES,
                version="minutes_v1.4.0",
                kind=KIND_MINUTES,
                values=claims,
                official_run_ids={"fetch": {"run_id": fetch}},
                fixture_of={player: FIXTURE_1 for player in claims},
            )
        )
    certification = ol.certify_prediction_freeze(
        conn, projection_run_ids=runs, bootstrap_generation_id=generation_id
    )
    assert certification.provenance_state == ol.GENERATION_CERTIFIED, certification.reasons
    return certification


def _archived_element_row(
    *,
    player_id: int = DGW_PLAYER,
    event: int = EVENT,
    fixture_id: int = FIXTURE_1,
    was_home: int = 1,
    team_h: int = TEAM_A,
    team_a: int = TEAM_B,
    kickoff_time: str = KICKOFF_1,
    minutes: int = 90,
    starts: int = 1,
    total_points: int = 6,
    bonus: int | None = 1,
    bps: int | None = 30,
    **extra,
) -> dict:
    """One ``element-summary`` history row, exactly as the official source states it.

    Its values are the same official facts ``_capture`` stores, so a backfill of
    this row is a faithful copy -- the only kind of backfill an archive witnesses.
    """

    row = {
        "element": player_id,
        "event": event,
        "fixture": fixture_id,
        "opponent_team": team_a if was_home else team_h,
        "was_home": was_home,
        "team_h": team_h,
        "team_a": team_a,
        "kickoff_time": kickoff_time,
        "minutes": minutes,
        "starts": starts,
        "total_points": total_points,
        "goals_scored": 1,
        "assists": 0,
        "clean_sheets": 0,
        "goals_conceded": 1,
        "saves": 0,
        "bonus": bonus,
        "bps": bps,
        "yellow_cards": 0,
        "red_cards": 0,
        "penalties_saved": 0,
        "penalties_missed": 0,
        "own_goals": 0,
        "defensive_contribution": 0,
    }
    row.update(extra)
    return row


def _archived_observation(
    tmp_path,
    *,
    rows,
    source: str = "element_summary",
    observed_at: str | None = None,
    event: int | None = None,
    run_id: int = 7,
    raw_dir=None,
):
    """Archive one observation payload, returning ``(root, record)``."""

    return _archive(
        tmp_path,
        source=source,
        observed_at=observed_at or _t(6),
        payload={"history": list(rows)},
        event=event,
        run_id=run_id,
        raw_dir=raw_dir,
    )


def _live_element(
    *,
    player_id: int = DGW_PLAYER,
    fixture_id: int = FIXTURE_1,
    stats: dict | None = None,
    explain: list | None = None,
) -> dict:
    """One element of an ``event/live`` response: stats totals plus explain legs.

    The live endpoint nests each fixture's own numbers under ``explain``, which
    is the only fixture-specific part of the payload -- the top-level ``stats``
    object is the player's EVENT total.  A double gameweek therefore has to be
    read out of the explain legs, or one fixture's numbers get copied onto both.
    """

    return {
        "id": player_id,
        "stats": stats
        if stats is not None
        else {
            "minutes": 90, "starts": 1, "total_points": 6, "goals_scored": 1,
            "assists": 0, "clean_sheets": 0, "goals_conceded": 1, "saves": 0,
            "bonus": 1, "bps": 30, "yellow_cards": 0, "red_cards": 0,
            "penalties_saved": 0, "penalties_missed": 0, "own_goals": 0,
            "defensive_contribution": 0, "fixture": fixture_id,
        },
        "explain": explain if explain is not None else [],
    }


def _live_leg(
    *,
    fixture_id: int = FIXTURE_1,
    opponent_team: int = TEAM_B,
    was_home: bool = True,
    kickoff_time: str = KICKOFF_1,
    minutes: int = 90,
    total_points: int = 6,
    bonus: int | None = 1,
    bps: int | None = 30,
    **extra,
) -> dict:
    """One ``explain`` entry: a fixture's own stat list, exactly as it is served."""

    stats = [
        {"identifier": "minutes", "points": 0, "value": minutes},
        {"identifier": "starts", "points": 0, "value": 1 if minutes else 0},
        {"identifier": "total_points", "points": total_points, "value": total_points},
        {"identifier": "goals_scored", "points": 0, "value": 1},
        {"identifier": "assists", "points": 0, "value": 0},
        {"identifier": "clean_sheets", "points": 0, "value": 0},
        {"identifier": "goals_conceded", "points": 0, "value": 1},
        {"identifier": "saves", "points": 0, "value": 0},
        {"identifier": "bonus", "points": 0, "value": bonus},
        {"identifier": "bps", "points": 0, "value": bps},
        {"identifier": "yellow_cards", "points": 0, "value": 0},
        {"identifier": "red_cards", "points": 0, "value": 0},
        {"identifier": "penalties_saved", "points": 0, "value": 0},
        {"identifier": "penalties_missed", "points": 0, "value": 0},
        {"identifier": "own_goals", "points": 0, "value": 0},
        {"identifier": "defensive_contribution", "points": 0, "value": 0},
    ]
    for name in extra:
        stats.append({"identifier": name, "points": 0, "value": extra[name]})
    return {
        "fixture": fixture_id,
        "opponent_team": opponent_team,
        "was_home": was_home,
        "kickoff_time": kickoff_time,
        "stats": stats,
    }


def _archived_live(
    tmp_path,
    *,
    elements,
    event: int = EVENT,
    observed_at: str | None = None,
    run_id: int = 7,
    raw_dir=None,
):
    """Archive one ``event/live`` response, returning ``(root, record)``."""

    return _archive(
        tmp_path,
        source=f"event_live_{int(event)}",
        observed_at=observed_at or _t(6),
        payload={"elements": list(elements)},
        event=int(event),
        run_id=run_id,
        raw_dir=raw_dir,
    )


def _backfill(conn, *, root, record, source_identity: str | None = None, **overrides):
    """The backfill claim for one archived capture, carrying the archive's own values."""

    arguments = {
        "player_id": DGW_PLAYER,
        "fixture_id": FIXTURE_1,
        "captured_at": _t(6),
        "source_name": "archived_element_summary",
        "source_identity": source_identity or str(record.source),
        "source_payload_sha256": record.payload_sha256,
        "archive_capture_id": record.capture_id,
        "fetch_run_id": 7,
        "backfill": True,
        "archive_root": root,
    }
    arguments.update(overrides)
    return _capture(conn, **arguments)


# ---------------------------------------------------------------------------
# Point-in-time integrity
# ---------------------------------------------------------------------------


def test_01_two_captures_of_the_same_key_are_both_retained(tmp_path):
    """Contract 1: two captures of the same player/event/fixture both survive."""

    conn = connect_database(tmp_path / "fpl.db")
    _world(conn)
    first = _capture(conn, player_id=DGW_PLAYER, fixture_id=FIXTURE_1, captured_at=CAPTURE_1)
    second = _capture(conn, player_id=DGW_PLAYER, fixture_id=FIXTURE_1, captured_at=CAPTURE_2,
                      total_points=9)
    assert first.inserted and second.inserted
    assert first.capture_digest != second.capture_digest
    rows = ol.observation_captures(conn, event=EVENT, player_id=DGW_PLAYER)
    assert len(rows) == 2
    assert [row["captured_at"] for row in rows] == [CAPTURE_1, CAPTURE_2]
    assert [row["total_points"] for row in rows] == [6, 9]
    conn.close()


def test_02_a_later_capture_cannot_overwrite_an_earlier_observation(tmp_path):
    """Contract 2: a later read of the same key never rewrites the earlier one."""

    conn = connect_database(tmp_path / "fpl.db")
    _world(conn)
    _capture(conn, player_id=DGW_PLAYER, fixture_id=FIXTURE_1, captured_at=CAPTURE_1, total_points=6)
    _capture(conn, player_id=DGW_PLAYER, fixture_id=FIXTURE_1, captured_at=CAPTURE_2, total_points=9)
    as_of_first = ol.observation_captures(conn, event=EVENT, as_of=CAPTURE_1)
    assert [row["total_points"] for row in as_of_first] == [6]
    as_of_later = ol.observation_captures(conn, event=EVENT, as_of=CAPTURE_2)
    assert [row["total_points"] for row in as_of_later] == [6, 9]
    # The later value is a correction channel, not a replacement.
    chosen = ol.select_capture(as_of_later)
    assert chosen is not None and chosen["total_points"] == 9
    conn.close()


def test_03_identical_ingest_is_idempotent(tmp_path):
    """Contract 3: re-ingesting the SAME observation is a no-op."""

    conn = connect_database(tmp_path / "fpl.db")
    _world(conn)
    first = _capture(conn, player_id=DGW_PLAYER, fixture_id=FIXTURE_1)
    repeat = _capture(conn, player_id=DGW_PLAYER, fixture_id=FIXTURE_1)
    assert first.inserted is True and repeat.inserted is False
    assert first.capture_digest == repeat.capture_digest
    assert conn.execute("SELECT COUNT(*) FROM outcome_observation_captures").fetchone()[0] == 1
    conn.close()


def test_04_scheduled_placeholder_is_excluded(tmp_path):
    """Contract 4: a scheduled placeholder is never recorded as an observation."""

    conn = connect_database(tmp_path / "fpl.db")
    _world(conn)
    with pytest.raises(ol.PlaceholderObservationError):
        ol.capture_observation(
            conn,
            grain=ol.GRAIN_PLAYER_FIXTURE,
            event=EVENT,
            player_id=PLACEHOLDER_PLAYER,
            fixture_id=FIXTURE_1,
            # The canonical placeholder signature: minutes known, nothing else.
            fields={"minutes": 0},
            source_name="element_summary",
            captured_at=CAPTURE_1,
        )
    assert conn.execute("SELECT COUNT(*) FROM outcome_observation_captures").fetchone()[0] == 0
    conn.close()


def test_05_genuine_completed_dnp_zero_is_retained(tmp_path):
    """Contract 5: a real completed DNP with zero minutes is a real observation."""

    conn = connect_database(tmp_path / "fpl.db")
    _world(conn)
    result = _capture(
        conn, player_id=SGW_A_PLAYER, fixture_id=FIXTURE_2, captured_at=CAPTURE_1,
        minutes=0, starts=0, total_points=0, bonus=0, bps=0,
        goals_scored=0, assists=0, clean_sheets=0, goals_conceded=0,
        saves=0, yellow_cards=0, red_cards=0, penalties_saved=0,
        penalties_missed=0, own_goals=0, defensive_contribution=0,
    )
    assert result.inserted
    row = ol.observation_captures(conn, event=EVENT, player_id=SGW_A_PLAYER)[0]
    assert row["minutes"] == 0 and row["total_points"] == 0 and row["starts"] == 0
    # Zero minutes is a value, not an absence: it must not read as a placeholder.
    assert repo.row_is_scheduled_placeholder(row) is False
    conn.close()


def test_06_genuine_zero_bonus_and_bps_are_retained(tmp_path):
    """Contract 6: a genuine zero bonus/BPS is retained as zero."""

    conn = connect_database(tmp_path / "fpl.db")
    _world(conn)
    _capture(conn, player_id=DGW_PLAYER, fixture_id=FIXTURE_2, bonus=0, bps=0)
    row = ol.observation_captures(conn, event=EVENT, player_id=DGW_PLAYER)[0]
    assert row["bonus"] == 0
    assert row["bps"] == 0
    assert row["bonus"] is not None and row["bps"] is not None
    conn.close()


def test_07_missing_bonus_remains_missing(tmp_path):
    """Contract 7: a field the source never stated stays missing, not zero."""

    conn = connect_database(tmp_path / "fpl.db")
    _world(conn)
    _capture(conn, player_id=DGW_PLAYER, fixture_id=FIXTURE_2, bonus=None, bps=None)
    row = ol.observation_captures(conn, event=EVENT, player_id=DGW_PLAYER)[0]
    assert row["bonus"] is None
    assert row["bps"] is None
    assert row["total_points"] == 6  # the fields that WERE stated are untouched
    conn.close()


def test_08_partial_event_cannot_enter_final_evaluation(tmp_path):
    """Contract 8: a fixture that is over inside a not-yet-final event is excluded."""

    conn = connect_database(tmp_path / "fpl.db")
    _world(conn)
    claims = {DGW_PLAYER: 5.0}
    runs = [_freeze(conn, family=FAMILY_POINTS, version=analytics.BASELINE_MODEL_VERSION,
                    kind=KIND_POINTS, values=claims, event=EVENT_PARTIAL)]
    certification = ol.certify_prediction_freeze(conn, projection_run_ids=runs)
    _capture(conn, player_id=DGW_PLAYER, fixture_id=FIXTURE_PARTIAL, event=EVENT_PARTIAL,
             captured_at=CAPTURE_1)
    ledger = ol.build_reality_ledger(conn, freeze_identity=certification.freeze_identity)
    assert ledger.state_counts == ((ol.OUTCOME_NOT_FINALISED, 1),)
    assert ledger.evaluated() == ()
    assert ledger.rows[0]["outcome_state"] == "NOT_FINALISED"
    conn.close()


def test_09_final_event_can_enter_evaluation(tmp_path):
    """Contract 9: the same row enters evaluation once the event IS final."""

    conn = connect_database(tmp_path / "fpl.db")
    _world(conn)
    certification = _certified_freeze(conn, values={DGW_PLAYER: 6.0})
    _capture(conn, player_id=DGW_PLAYER, fixture_id=FIXTURE_1)
    _capture(conn, player_id=DGW_PLAYER, fixture_id=FIXTURE_2, total_points=3)
    ledger = ol.build_reality_ledger(conn, freeze_identity=certification.freeze_identity)
    evaluated = ledger.evaluated()
    assert len(evaluated) == 1
    assert evaluated[0]["exclusion_reason"] == ol.EVALUATED
    assert evaluated[0]["official_outcome"]["total_points"] == 9
    conn.close()


# ---------------------------------------------------------------------------
# Grain
# ---------------------------------------------------------------------------


def test_10_dgw_fixture_outcomes_remain_distinct(tmp_path):
    """Contract 10: a double gameweek keeps two distinct fixture-grain rows."""

    conn = connect_database(tmp_path / "fpl.db")
    _world(conn)
    certification = _certified_freeze(conn, values={DGW_PLAYER: 6.0})
    _capture(conn, player_id=DGW_PLAYER, fixture_id=FIXTURE_1, minutes=90, total_points=6)
    _capture(conn, player_id=DGW_PLAYER, fixture_id=FIXTURE_2, minutes=62, total_points=3)
    ledger = ol.build_reality_ledger(
        conn, freeze_identity=certification.freeze_identity, grain=ol.GRAIN_PLAYER_FIXTURE
    )
    points = {
        row["fixture_id"]: row["official_outcome"]["total_points"]
        for row in ledger.rows
        if row["player_id"] == DGW_PLAYER
    }
    assert points == {FIXTURE_1: 6, FIXTURE_2: 3}
    # ...and the fixture evidence is not collapsed into one number.
    assert len([row for row in ledger.rows if row["player_id"] == DGW_PLAYER]) == 2
    conn.close()


def test_11_headline_event_points_aggregate_once(tmp_path):
    """Contract 11: the event total is summed once, never once per fixture."""

    conn = connect_database(tmp_path / "fpl.db")
    _world(conn)
    certification = _certified_freeze(conn, values={DGW_PLAYER: 6.0})
    _capture(conn, player_id=DGW_PLAYER, fixture_id=FIXTURE_1, minutes=90, total_points=6)
    _capture(conn, player_id=DGW_PLAYER, fixture_id=FIXTURE_2, minutes=62, total_points=3)
    ledger = ol.build_reality_ledger(conn, freeze_identity=certification.freeze_identity)
    rows = [row for row in ledger.rows if row["player_id"] == DGW_PLAYER]
    assert len(rows) == 1
    assert rows[0]["fixture_id"] is None
    assert rows[0]["official_outcome"]["total_points"] == 9
    assert rows[0]["official_outcome"]["minutes"] == 152
    assert rows[0]["outcome_evidence"] == "AGGREGATED_FIXTURE_CAPTURES"
    assert len(rows[0]["observation_capture_digests"]) == 2
    conn.close()


def test_12_blank_is_explicit_rather_than_missing(tmp_path):
    """Contract 12: a blank Gameweek is an explicit state, not a missing row."""

    conn = connect_database(tmp_path / "fpl.db")
    _world(conn)
    certification = _certified_freeze(conn, values={BLANK_PLAYER: 4.0})
    ledger = ol.build_reality_ledger(conn, freeze_identity=certification.freeze_identity)
    rows = [row for row in ledger.rows if row["player_id"] == BLANK_PLAYER]
    assert len(rows) == 1
    assert rows[0]["outcome_state"] == "BLANK"
    assert rows[0]["exclusion_reason"] == ol.TARGET_NO_FIXTURE
    assert rows[0]["evaluation_state"] == ol.EXCLUDED
    conn.close()


# ---------------------------------------------------------------------------
# Causality
# ---------------------------------------------------------------------------


def test_13_observation_after_cutoff_is_excluded_from_earlier_as_of_read(tmp_path):
    """Contract 13: a later observation is invisible to an earlier as-of read."""

    conn = connect_database(tmp_path / "fpl.db")
    _world(conn)
    certification = _certified_freeze(conn, values={SGW_B_PLAYER: 5.0})
    _capture(conn, player_id=SGW_B_PLAYER, fixture_id=FIXTURE_1, captured_at=CAPTURE_2,
             total_points=9)
    early = ol.build_reality_ledger(
        conn, freeze_identity=certification.freeze_identity, as_of=AS_OF
    )
    assert early.rows[0]["exclusion_reason"] == ol.INPUT_EVIDENCE_UNAVAILABLE
    assert early.rows[0]["official_outcome"] is None
    later = ol.build_reality_ledger(conn, freeze_identity=certification.freeze_identity)
    assert later.rows[0]["official_outcome"]["total_points"] == 9
    conn.close()


def test_14_observation_before_kickoff_cannot_become_historical_evidence(tmp_path):
    """Contract 14: a capture claiming a time before kickoff is refused."""

    conn = connect_database(tmp_path / "fpl.db")
    _world(conn)
    assert parse_utc(_fixture_kickoff(conn, FIXTURE_1)) > parse_utc(_t(9))
    with pytest.raises(ol.OutcomeLedgerError, match="precedes the football event"):
        _capture(conn, player_id=DGW_PLAYER, fixture_id=FIXTURE_1, captured_at=_t(9))
    assert conn.execute("SELECT COUNT(*) FROM outcome_observation_captures").fetchone()[0] == 0
    conn.close()


def test_15_later_outcome_append_cannot_change_an_earlier_causal_read(tmp_path):
    """Contract 15: appending a later outcome leaves the earlier read byte-identical."""

    conn = connect_database(tmp_path / "fpl.db")
    _world(conn)
    certification = _certified_freeze(conn, values={DGW_PLAYER: 6.0, SGW_B_PLAYER: 5.0})
    _capture(conn, player_id=DGW_PLAYER, fixture_id=FIXTURE_1, captured_at=CAPTURE_1)
    _capture(conn, player_id=DGW_PLAYER, fixture_id=FIXTURE_2, captured_at=CAPTURE_1)
    before = ol.build_reality_ledger(
        conn, freeze_identity=certification.freeze_identity, as_of=AS_OF
    )
    _capture(conn, player_id=SGW_B_PLAYER, fixture_id=FIXTURE_1, captured_at=CAPTURE_2, total_points=9)
    after = ol.build_reality_ledger(
        conn, freeze_identity=certification.freeze_identity, as_of=AS_OF
    )
    assert before.ledger_digest == after.ledger_digest
    assert before.population_digest == after.population_digest
    assert before.rows == after.rows
    conn.close()


def test_15b_capture_never_regenerates_or_mutates_the_frozen_prediction(tmp_path):
    """Contract theme 7: outcome capture never rewrites a frozen prediction value."""

    conn = connect_database(tmp_path / "fpl.db")
    _world(conn)
    certification = _certified_freeze(conn, values={DGW_PLAYER: 6.0})
    before = ol.build_reality_ledger(conn, freeze_identity=certification.freeze_identity)
    predictions_before = conn.execute(
        "SELECT id, payload_json, model_version FROM frozen_predictions ORDER BY id"
    ).fetchall()
    _capture_all(conn)
    after = ol.build_reality_ledger(conn, freeze_identity=certification.freeze_identity)
    predictions_after = conn.execute(
        "SELECT id, payload_json, model_version FROM frozen_predictions ORDER BY id"
    ).fetchall()
    assert [tuple(row) for row in predictions_before] == [tuple(row) for row in predictions_after]
    before_by_key = {(r["player_id"], r["prediction_kind"]): r for r in before.rows}
    for row in after.rows:
        key = (row["player_id"], row["prediction_kind"])
        if key not in before_by_key:
            continue
        assert row["frozen_predicted_value"] == before_by_key[key]["frozen_predicted_value"]
    assert all(row["frozen_predicted_value"] is not None for row in before.evaluated())
    conn.close()


# ---------------------------------------------------------------------------
# Generation certification
# ---------------------------------------------------------------------------


def test_16_accepted_generation_certifies_a_new_freeze(tmp_path):
    """Contract 16: an accepted generation certifies a new freeze, in full."""

    conn = connect_database(tmp_path / "fpl.db")
    _world(conn)
    certification = _certified_freeze(conn)
    assert certification.provenance_state == ol.GENERATION_CERTIFIED
    assert certification.reasons == ()
    recorded = ol.freeze_provenance(conn, certification.freeze_identity)
    assert recorded is not None
    generation = repo.latest_accepted_bootstrap_generation(conn)
    assert recorded["bootstrap_generation_id"] == generation["id"]
    assert recorded["official_fetch_run_id"] == generation["fetch_run_id"]
    assert recorded["planning_cutoff"] == CUTOFF
    assert recorded["bootstrap_element_ids_sha256"] == generation["element_ids_sha256"]
    assert recorded["bootstrap_element_ids"] == json.loads(generation["element_ids_json"])
    assert recorded["prediction_artifact_sha256"].startswith("sha256:")
    assert len(recorded["runs"]) == 2
    assert {run["model_family"] for run in recorded["runs"]} == {FAMILY_POINTS, FAMILY_MINUTES}
    assert all(run["config_hash"] == "cfg" and run["random_seed"] == 7 for run in recorded["runs"])
    conn.close()


def test_16b_official_fetch_without_a_generation_cannot_certify(tmp_path):
    """A freeze that declares an official fetch but no generation cannot certify."""

    conn = connect_database(tmp_path / "fpl.db")
    _world(conn)
    fetch = _fetch_run(conn)
    run = _freeze(
        conn, family=FAMILY_POINTS, version=analytics.BASELINE_MODEL_VERSION,
        kind=KIND_POINTS, values={DGW_PLAYER: 6.0},
        official_run_ids={"fetch": {"run_id": fetch}},
    )
    certification = ol.certify_prediction_freeze(conn, projection_run_ids=[run])
    assert certification.provenance_state == ol.NOT_CERTIFIABLE
    assert any("no bootstrap generation" in reason for reason in certification.reasons)
    conn.close()


def test_16c_a_run_that_recorded_its_official_fetch_is_certified_from_it(tmp_path):
    """A run whose existing provenance names the fetch can prove its generation.

    The generation is derived from the fetch identity the run ALREADY recorded
    and from the cutoff, then pinned on the freeze.  Nothing in the certified
    prediction pipeline has to change for a new freeze to be certifiable.
    """

    conn = connect_database(tmp_path / "fpl.db")
    _world(conn)
    fetch = _fetch_run(conn)
    generation = _generation(conn, fetch_run_id=fetch, captured_at=GENERATION_CAPTURED)
    run = _freeze(
        conn, family=FAMILY_POINTS, version=analytics.BASELINE_MODEL_VERSION,
        kind=KIND_POINTS, values={DGW_PLAYER: 6.0},
        official_run_ids={"fetch": {"run_id": fetch}},
    )
    certification = ol.certify_prediction_freeze(conn, projection_run_ids=[run])
    assert certification.provenance_state == ol.GENERATION_CERTIFIED, certification.reasons
    assert certification.bootstrap_generation_id == generation
    assert certification.official_fetch_run_id == fetch
    # A generation captured AFTER the cutoff is not eligible to be derived, so
    # the same run would not certify against it.
    other = _fetch_run(conn)
    _generation(conn, fetch_run_id=other, captured_at=_t(1))
    late_run = _freeze(
        conn, family=FAMILY_POINTS, version=analytics.BASELINE_MODEL_VERSION,
        kind=KIND_POINTS, values={DGW_PLAYER: 6.0},
        official_run_ids={"fetch": {"run_id": other}},
    )
    assert ol.accepted_generation_at(conn, observed_at=CUTOFF, fetch_run_id=other) is None
    late = ol.certify_prediction_freeze(conn, projection_run_ids=[late_run])
    assert late.provenance_state == ol.NOT_CERTIFIABLE
    assert any("no bootstrap generation" in reason for reason in late.reasons)
    conn.close()


def test_17_rejected_generation_cannot_certify(tmp_path):
    """Contract 17: a rejected generation never certifies the official pool."""

    conn = connect_database(tmp_path / "fpl.db")
    _world(conn)
    fetch = _fetch_run(conn)
    rejected = _generation(conn, fetch_run_id=fetch, accepted=False)
    run = _freeze(
        conn, family=FAMILY_POINTS, version=analytics.BASELINE_MODEL_VERSION,
        kind=KIND_POINTS, values={DGW_PLAYER: 6.0},
        official_run_ids={"fetch": {"run_id": fetch}},
    )
    certification = ol.certify_prediction_freeze(
        conn, projection_run_ids=[run], bootstrap_generation_id=rejected
    )
    assert certification.provenance_state == ol.NOT_CERTIFIABLE
    assert any("rejected" in reason for reason in certification.reasons)
    # The generation that was EXAMINED is recorded for audit; it is not
    # authoritative, which is what the NOT_CERTIFIABLE state exists to say.
    recorded = ol.freeze_provenance(conn, certification.freeze_identity)
    assert recorded["bootstrap_generation_id"] == rejected
    assert recorded["provenance_state"] == ol.NOT_CERTIFIABLE
    conn.close()


def test_18_missing_generation_cannot_certify(tmp_path):
    """Contract 18: a generation that does not exist cannot certify."""

    conn = connect_database(tmp_path / "fpl.db")
    _world(conn)
    fetch = _fetch_run(conn)
    run = _freeze(
        conn, family=FAMILY_POINTS, version=analytics.BASELINE_MODEL_VERSION,
        kind=KIND_POINTS, values={DGW_PLAYER: 6.0},
        official_run_ids={"fetch": {"run_id": fetch}},
    )
    certification = ol.certify_prediction_freeze(
        conn, projection_run_ids=[run], bootstrap_generation_id=987654
    )
    assert certification.provenance_state == ol.NOT_CERTIFIABLE
    assert any("does not exist" in reason for reason in certification.reasons)
    conn.close()


def test_19_generation_fetch_mismatch_fails(tmp_path):
    """Contract 19: the generation must come from the freeze's recorded fetch."""

    conn = connect_database(tmp_path / "fpl.db")
    _world(conn)
    other_fetch = _fetch_run(conn)
    generation = _generation(conn, fetch_run_id=other_fetch + 7)
    declared = _fetch_run(conn)
    run = _freeze(
        conn, family=FAMILY_POINTS, version=analytics.BASELINE_MODEL_VERSION,
        kind=KIND_POINTS, values={DGW_PLAYER: 6.0},
        official_run_ids={"fetch": {"run_id": declared}},
    )
    certification = ol.certify_prediction_freeze(
        conn, projection_run_ids=[run], bootstrap_generation_id=generation
    )
    assert certification.provenance_state == ol.NOT_CERTIFIABLE
    assert any("not the freeze's recorded fetch run" in reason for reason in certification.reasons)
    conn.close()


def test_20_generation_after_the_prediction_cutoff_fails(tmp_path):
    """Contract 20: a generation observable only AFTER the cutoff cannot certify."""

    conn = connect_database(tmp_path / "fpl.db")
    _world(conn)
    fetch = _fetch_run(conn)
    late = _generation(conn, fetch_run_id=fetch, captured_at=_t(1))
    run = _freeze(
        conn, family=FAMILY_POINTS, version=analytics.BASELINE_MODEL_VERSION,
        kind=KIND_POINTS, values={DGW_PLAYER: 6.0},
        official_run_ids={"fetch": {"run_id": fetch}},
    )
    certification = ol.certify_prediction_freeze(
        conn, projection_run_ids=[run], bootstrap_generation_id=late
    )
    assert certification.provenance_state == ol.NOT_CERTIFIABLE
    assert any("after the prediction cutoff" in reason for reason in certification.reasons)
    conn.close()


def test_21_newest_generation_cannot_replace_the_recorded_generation(tmp_path):
    """Contract 21: a later accepted generation does not re-certify an earlier freeze."""

    conn = connect_database(tmp_path / "fpl.db")
    _world(conn)
    certification = _certified_freeze(conn)
    recorded_first = ol.freeze_provenance(conn, certification.freeze_identity)["bootstrap_generation_id"]
    assert recorded_first is not None
    # A NEWER accepted generation arrives; the freeze's certificate must not move.
    newer = _generation(conn, fetch_run_id=_fetch_run(conn), captured_at=_t(3),
                        element_ids=(*SYNTHETIC_POOL, 61))
    assert int(newer) != int(recorded_first)
    assert repo.latest_accepted_bootstrap_generation(conn)["id"] == newer
    recorded_again = ol.freeze_provenance(conn, certification.freeze_identity)
    assert recorded_again["bootstrap_generation_id"] == recorded_first
    # ...and the freeze cannot be re-certified against the newer generation.
    with pytest.raises(ol.FreezeAlreadyCertified):
        ol.certify_prediction_freeze(
            conn, projection_run_ids=certification.projection_run_ids,
            bootstrap_generation_id=newer,
        )
    conn.close()


def test_22_legacy_run_remains_legacy(tmp_path):
    """Contract 22: a pre-PE-5 run stays honestly LEGACY_PROVENANCE."""

    conn = connect_database(tmp_path / "fpl.db")
    _world(conn)
    legacy = _freeze(
        conn, family=FAMILY_POINTS, version=analytics.BASELINE_MODEL_VERSION,
        kind=KIND_POINTS, values={DGW_PLAYER: 6.0}, official_run_ids={},
    )
    certification = ol.certify_prediction_freeze(conn, projection_run_ids=[legacy])
    assert certification.provenance_state == ol.LEGACY_PROVENANCE
    assert certification.bootstrap_generation_id is None
    # No bootstrap_generations row was fabricated to make it look certified.
    assert conn.execute("SELECT COUNT(*) FROM bootstrap_generations").fetchone()[0] == 0
    # A later accepted generation does not retroactively promote it.
    _generation(conn, fetch_run_id=_fetch_run(conn), captured_at=_t(2))
    assert ol.freeze_provenance(conn, certification.freeze_identity)["provenance_state"] == (
        ol.LEGACY_PROVENANCE
    )
    conn.close()


def test_23_official_element_digest_is_retained_and_deterministic(tmp_path):
    """Contract 23: the element-set identity is retained and recomputes exactly."""

    conn = connect_database(tmp_path / "fpl.db")
    _world(conn)
    first = _certified_freeze(conn)
    recorded = ol.freeze_provenance(conn, first.freeze_identity)
    retained = json.loads(recorded["bootstrap_element_ids_json"])
    assert retained == sorted(SYNTHETIC_POOL)
    assert retained == sorted(retained)
    assert element_id_sha256(retained) == recorded["bootstrap_element_ids_sha256"]

    # A second, independent generation over the SAME id set digests identically,
    # and the id set is order-insensitive by construction.
    same = _generation(conn, fetch_run_id=_fetch_run(conn), captured_at=_t(10),
                       element_ids=tuple(reversed(SYNTHETIC_POOL)))
    row = conn.execute("SELECT element_ids_sha256, element_ids_json FROM bootstrap_generations WHERE id=?",
                       (same,)).fetchone()
    assert row[0] == recorded["bootstrap_element_ids_sha256"]
    assert json.loads(row[1]) == retained
    conn.close()


def test_23b_malformed_retained_element_identity_fails_certification(tmp_path):
    """Condition 6 is enforced, not assumed: a broken retained set is not certifiable."""

    conn = connect_database(tmp_path / "fpl.db")
    _world(conn)
    fetch = _fetch_run(conn)
    generation = _generation(conn, fetch_run_id=fetch)
    conn.execute(
        "UPDATE bootstrap_generations SET element_ids_sha256='deadbeef' WHERE id=?", (generation,)
    )
    run = _freeze(
        conn, family=FAMILY_POINTS, version=analytics.BASELINE_MODEL_VERSION,
        kind=KIND_POINTS, values={DGW_PLAYER: 6.0},
        official_run_ids={"fetch": {"run_id": fetch}},
    )
    certification = ol.certify_prediction_freeze(
        conn, projection_run_ids=[run], bootstrap_generation_id=generation
    )
    assert certification.provenance_state == ol.NOT_CERTIFIABLE
    assert any("retained id set hashes to" in reason for reason in certification.reasons)
    conn.close()


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------


def test_24_prediction_outcome_population_is_deterministic(tmp_path):
    """Contract 24: the population is the sorted union of both sides."""

    conn = connect_database(tmp_path / "fpl.db")
    _world(conn)
    certification = _certified_freeze(conn)
    _capture_all(conn)
    # An outcome with NO prediction in the freeze still belongs to the population.
    _capture(conn, player_id=INACTIVE_PLAYER, fixture_id=FIXTURE_2, captured_at=CAPTURE_1,
             event=EVENT, total_points=1)
    ledger = ol.build_reality_ledger(conn, freeze_identity=certification.freeze_identity)
    keys = [
        (row["event"], row["player_id"], row["fixture_id"], row["prediction_kind"])
        for row in ledger.rows
    ]
    assert keys == sorted(keys, key=lambda k: (k[0], k[1], -1 if k[2] is None else k[2], str(k[3] or "")))
    assert len(keys) == len(set(keys))
    assert ledger.population_digest.startswith("sha256:")
    conn.close()


def test_25_missing_prediction_is_explicit(tmp_path):
    """Contract 25: an outcome with no frozen prediction says so, it is not dropped."""

    conn = connect_database(tmp_path / "fpl.db")
    _world(conn)
    certification = _certified_freeze(conn, values={DGW_PLAYER: 6.0, SGW_A_PLAYER: 3.0})
    _capture_all(conn)
    ledger = ol.build_reality_ledger(conn, freeze_identity=certification.freeze_identity)
    rows = [row for row in ledger.rows if row["player_id"] == SGW_B_PLAYER]
    assert len(rows) == 1
    assert rows[0]["prediction_state"] == "MISSING"
    assert rows[0]["prediction_kind"] is None
    assert rows[0]["frozen_predicted_value"] is None
    assert rows[0]["exclusion_reason"] == ol.MODEL_PROJECTION_MISSING
    assert rows[0]["official_outcome"]["total_points"] == 9
    conn.close()


def test_26_missing_outcome_is_explicit(tmp_path):
    """Contract 26: a prediction with no stored observation is explicit, never zero."""

    conn = connect_database(tmp_path / "fpl.db")
    _world(conn)
    certification = _certified_freeze(conn)
    _capture_all(conn)
    ledger = ol.build_reality_ledger(conn, freeze_identity=certification.freeze_identity)
    rows = [row for row in ledger.rows if row["player_id"] == NO_OUTCOME_PLAYER]
    assert rows and all(row["evaluation_state"] == ol.EXCLUDED for row in rows)
    assert {row["exclusion_reason"] for row in rows} == {ol.INPUT_EVIDENCE_UNAVAILABLE}
    assert all(row["official_outcome"] is None for row in rows)
    assert all(row["outcome_state"] == "MISSING" for row in rows)
    conn.close()


def test_27_placeholder_is_explicit(tmp_path):
    """Contract 27: a placeholder row is its own state, not a zero-minute appearance."""

    conn = connect_database(tmp_path / "fpl.db")
    _world(conn)
    # Exactly the canonical placeholder signature: minutes 0, every other
    # performance column NULL, written as a schedule row before kickoff.
    conn.execute(
        """INSERT INTO player_gameweeks(
             player_id, event, fixture_id, minutes, source, raw_json, updated_at)
           VALUES (?,?,?,?,?,?,?)""",
        (PLACEHOLDER_PLAYER, EVENT, FIXTURE_1, 0, "element_summary", "{}", _t(10)),
    )
    certification = _certified_freeze(conn)
    ledger = ol.build_reality_ledger(conn, freeze_identity=certification.freeze_identity)
    rows = [row for row in ledger.rows if row["player_id"] == PLACEHOLDER_PLAYER]
    assert rows
    assert {row["outcome_state"] for row in rows} == {"PLACEHOLDER"}
    assert {row["exclusion_reason"] for row in rows} == {ol.OUTCOME_PLACEHOLDER_EXCLUDED}
    conn.close()


def test_28_genuine_zero_is_evaluated(tmp_path):
    """Contract 28: a real official zero is EVALUATED, not excluded."""

    conn = connect_database(tmp_path / "fpl.db")
    _world(conn)
    certification = _certified_freeze(conn, values={SGW_B_PLAYER: 2.0})
    _capture(
        conn, player_id=SGW_B_PLAYER, fixture_id=FIXTURE_1, captured_at=CAPTURE_1,
        minutes=0, starts=0, total_points=0, bonus=0, bps=4,
        goals_scored=0, assists=0, clean_sheets=0, goals_conceded=2,
    )
    ledger = ol.build_reality_ledger(conn, freeze_identity=certification.freeze_identity)
    assert [row["exclusion_reason"] for row in ledger.rows] == [ol.EVALUATED]
    row = ledger.rows[0]
    assert row["evaluation_state"] == ol.EVALUATED
    assert row["official_outcome"]["total_points"] == 0
    assert row["official_outcome"]["minutes"] == 0
    assert row["frozen_predicted_value"] == 2.0
    conn.close()


def test_29_accepted_history_rejects_update(tmp_path):
    """Contract 29: accepted history rejects ordinary UPDATE."""

    conn = connect_database(tmp_path / "fpl.db")
    _world(conn)
    _capture(conn, player_id=DGW_PLAYER, fixture_id=FIXTURE_1)
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        conn.execute("UPDATE outcome_observation_captures SET total_points=99")
    assert conn.execute("SELECT total_points FROM outcome_observation_captures").fetchone()[0] == 6
    conn.close()


def test_30_accepted_history_rejects_delete(tmp_path):
    """Contract 30: accepted history rejects DELETE -- deletion is not a correction."""

    conn = connect_database(tmp_path / "fpl.db")
    _world(conn)
    _capture(conn, player_id=DGW_PLAYER, fixture_id=FIXTURE_1)
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        conn.execute("DELETE FROM outcome_observation_captures")
    assert conn.execute("SELECT COUNT(*) FROM outcome_observation_captures").fetchone()[0] == 1
    # The freeze provenance is equally uncorrectable by rewrite.
    certification = _certified_freeze(conn, values={DGW_PLAYER: 6.0})
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        conn.execute("UPDATE prediction_freeze_provenance SET provenance_state='LEGACY_PROVENANCE'")
    with pytest.raises(sqlite3.IntegrityError, match="immutable"):
        conn.execute("DELETE FROM prediction_freeze_provenance")
    assert ol.freeze_provenance(conn, certification.freeze_identity)["provenance_state"] == (
        ol.GENERATION_CERTIFIED
    )
    conn.close()


def test_31_repeated_ledger_build_is_deterministic(tmp_path):
    """Contract 31: building the same ledger twice yields identical everything."""

    conn = connect_database(tmp_path / "fpl.db")
    _world(conn)
    certification = _certified_freeze(conn)
    _capture_all(conn)
    first = ol.build_reality_ledger(conn, freeze_identity=certification.freeze_identity)
    second = ol.build_reality_ledger(conn, freeze_identity=certification.freeze_identity)
    assert first.rows == second.rows
    assert first.population_digest == second.population_digest
    assert first.ledger_digest == second.ledger_digest
    assert first.state_counts == second.state_counts
    fixture_first = ol.build_reality_ledger(
        conn, freeze_identity=certification.freeze_identity, grain=ol.GRAIN_PLAYER_FIXTURE
    )
    fixture_second = ol.build_reality_ledger(
        conn, freeze_identity=certification.freeze_identity, grain=ol.GRAIN_PLAYER_FIXTURE
    )
    assert fixture_first.ledger_digest == fixture_second.ledger_digest
    # The two grains are namespaced, so one digest can never be read as the other.
    assert fixture_first.ledger_digest != first.ledger_digest
    conn.close()


def test_32_database_row_order_does_not_change_the_ledger_or_any_digest(tmp_path):
    """Contract 32: two stores with the same content in a different row order agree."""

    forward = _ledger_from(connect_database(tmp_path / "forward.db"), reverse=False)
    backward = _ledger_from(connect_database(tmp_path / "backward.db"), reverse=True)
    assert forward[0] == backward[0]                       # ledger digest
    assert forward[1] == backward[1]                        # population digest
    assert forward[2] == backward[2]                        # state counts
    assert forward[3] == backward[3]                        # digest-bearing row content
    assert forward[4] == backward[4]                        # fixture-grain ledger digest


def _ledger_from(conn, *, reverse: bool) -> tuple:
    _world(conn)
    claims = {
        DGW_PLAYER: 8.0, SGW_A_PLAYER: 3.0, SGW_B_PLAYER: 5.0,
        BLANK_PLAYER: 4.0, NO_OUTCOME_PLAYER: 2.0, PLACEHOLDER_PLAYER: 3.0,
    }
    fetch = _fetch_run(conn)
    generation = _generation(conn, fetch_run_id=fetch)
    order = sorted(claims.items(), reverse=reverse)
    run = _freeze(
        conn, family=FAMILY_POINTS, version=analytics.BASELINE_MODEL_VERSION,
        kind=KIND_POINTS, values=dict(order),
        official_run_ids={"fetch": {"run_id": fetch}},
    )
    certification = ol.certify_prediction_freeze(
        conn, projection_run_ids=[run], bootstrap_generation_id=generation
    )
    for player_id, fixture_id in sorted(
        ((DGW_PLAYER, FIXTURE_1), (DGW_PLAYER, FIXTURE_2), (SGW_B_PLAYER, FIXTURE_1)),
        key=lambda pair: (pair[0], pair[1]),
        reverse=reverse,
    ):
        _capture(conn, player_id=player_id, fixture_id=fixture_id, captured_at=CAPTURE_1)
    ledger = ol.build_reality_ledger(conn, freeze_identity=certification.freeze_identity)
    fixture_ledger = ol.build_reality_ledger(
        conn, freeze_identity=certification.freeze_identity, grain=ol.GRAIN_PLAYER_FIXTURE
    )
    content = tuple(
        json.dumps(
            {name: row[name] for name in ol._DIGEST_ROW_FIELDS},  # noqa: SLF001 - contract surface
            sort_keys=True,
        )
        for row in ledger.rows
    )
    conn.close()
    return (
        ledger.ledger_digest,
        ledger.population_digest,
        ledger.state_counts,
        content,
        fixture_ledger.ledger_digest,
    )


# ---------------------------------------------------------------------------
# Contract themes that are not one of the 32 numbered cases
# ---------------------------------------------------------------------------


def test_T_no_fake_backfill_without_an_archived_source(tmp_path):
    """Theme 8: historical point-in-time history needs a PROVEN archived source.

    Asserted identifiers are not proof.  The backfill must name the archive it
    was read from, and the archive must hold that exact capture: the source, the
    observation time, the fetch run and the payload digest must all agree, and
    the archived bytes must still hash to the recorded digest.  Absent that proof
    the honest answer is "unavailable", never a number manufactured from today's
    current state.  What the manifest cannot settle -- what the bytes SAY -- is
    bound separately, by parsing the archived payload.
    """

    from fpl_brain import raw_archive

    conn = connect_database(tmp_path / "fpl.db")
    _world(conn)

    def _attempt(**overrides):
        arguments = {
            "player_id": DGW_PLAYER,
            "fixture_id": FIXTURE_1,
            "captured_at": _t(6),
            "source_name": "archived_element_summary",
            "source_identity": "element_summary",
            "source_payload_sha256": "a" * 64,
            "archive_capture_id": "capture-7",
            "fetch_run_id": 7,
            "backfill": True,
        }
        arguments.update(overrides)
        return _capture(conn, **arguments)

    # 1. no archive at all: the identifiers are asserted, so the capture is refused.
    with pytest.raises(ol.BackfillEvidenceError, match="archive it was read from"):
        _attempt()
    # 2. no observation time: the real observable instant is missing.
    with pytest.raises(ol.BackfillEvidenceError, match="actual observable time"):
        _attempt(captured_at=None)

    # 3. an archive that does not contain the claimed capture proves nothing.
    root, record = _archived_observation(tmp_path, rows=[_archived_element_row()])
    with pytest.raises(ol.BackfillEvidenceError, match="not in the manifest"):
        _attempt(archive_root=root)

    # 4. a claimed digest that disagrees with the archived bytes is refused.
    with pytest.raises(ol.BackfillEvidenceError, match="holds payload"):
        _attempt(
            archive_root=root,
            archive_capture_id=record.capture_id,
            source_payload_sha256="b" * 64,
        )
    # 5. a claimed source or observation time the archive contradicts is refused.
    with pytest.raises(ol.BackfillEvidenceError, match="came from source"):
        _attempt(
            archive_root=root,
            archive_capture_id=record.capture_id,
            source_payload_sha256=record.payload_sha256,
            source_identity="element_summary_v2",
        )
    with pytest.raises(ol.BackfillEvidenceError, match="was observed at"):
        _attempt(
            archive_root=root,
            archive_capture_id=record.capture_id,
            source_payload_sha256=record.payload_sha256,
            captured_at=_t(6.5),
        )
    # 6. a fetch run the archive contradicts is refused.
    with pytest.raises(ol.BackfillEvidenceError, match="records fetch run"):
        _attempt(
            archive_root=root,
            archive_capture_id=record.capture_id,
            source_payload_sha256=record.payload_sha256,
            fetch_run_id=8,
        )
    assert conn.execute("SELECT COUNT(*) FROM outcome_observation_captures").fetchone()[0] == 0

    # 7. the SAME claim, backed by the archived blob, is admitted -- and it is the
    #    archived evidence, not the caller's word, that made it admissible.  The
    #    club identity is the archive's too: the live player table is moved to
    #    another club first, and the stored observation still names the club the
    #    archived row was played for.
    conn.execute("UPDATE players SET team_id=? WHERE id=?", (TEAM_C, DGW_PLAYER))
    result = _attempt(
        archive_root=root,
        archive_capture_id=record.capture_id,
        source_payload_sha256=record.payload_sha256,
    )
    assert result.inserted
    stored = ol.observation_captures(conn, event=EVENT, player_id=DGW_PLAYER)[0]
    assert stored["archive_capture_id"] == record.capture_id
    assert stored["fetch_run_id"] == 7
    assert stored["team_id"] == TEAM_A
    assert stored["opponent_team_id"] == TEAM_B
    assert stored["event_time"] == KICKOFF_1
    assert stored["total_points"] == 6
    assert json.loads(stored["backfill_evidence_json"])["archive_root"] == str(root)

    # 8. once the archived bytes are gone, the same claim is refused again: the
    #    archive record alone, with no intact blob, is not evidence.
    blob = Path(root) / raw_archive.ARCHIVE_DIRNAME / record.relative_path
    body = blob.read_bytes()
    blob.unlink()
    try:
        with pytest.raises(ol.BackfillEvidenceError, match="no longer matches its recorded digest"):
            _attempt(
                player_id=SGW_B_PLAYER,
                archive_root=root,
                archive_capture_id=record.capture_id,
                source_payload_sha256=record.payload_sha256,
            )
    finally:
        blob.write_bytes(body)
    conn.close()


# ---------------------------------------------------------------------------
# Hardening cases the review demonstrated
# ---------------------------------------------------------------------------


def test_H1_a_certified_freeze_refuses_new_frozen_predictions(tmp_path):
    """The certified artifact is closed: no prediction may be added to it."""

    conn = connect_database(tmp_path / "fpl.db")
    _world(conn)
    certification = _certified_freeze(conn, values={DGW_PLAYER: 6.0})
    run_id = certification.projection_run_ids[0]
    before = conn.execute(
        "SELECT COUNT(*) FROM frozen_predictions WHERE projection_run_id=?", (run_id,)
    ).fetchone()[0]
    with pytest.raises(sqlite3.IntegrityError, match="closed to new frozen predictions"):
        analytics.freeze_prediction(
            conn,
            run_id,
            kind=KIND_POINTS,
            player_id=SGW_A_PLAYER,
            event=EVENT,
            payload={"value": 1.0},
            model_version=analytics.BASELINE_MODEL_VERSION,
        )
    assert conn.execute(
        "SELECT COUNT(*) FROM frozen_predictions WHERE projection_run_id=?", (run_id,)
    ).fetchone()[0] == before
    conn.close()


def test_H2_a_running_run_cannot_be_certified(tmp_path):
    """An open run's frozen set can still grow, so it is not a closed artifact."""

    conn = connect_database(tmp_path / "fpl.db")
    _world(conn)
    fetch = _fetch_run(conn)
    generation = _generation(conn, fetch_run_id=fetch)
    running = _freeze(
        conn, family=FAMILY_POINTS, version=analytics.BASELINE_MODEL_VERSION,
        kind=KIND_POINTS, values={DGW_PLAYER: 6.0},
        official_run_ids={"fetch": {"run_id": fetch}}, finish=False,
    )
    certification = ol.certify_prediction_freeze(
        conn, projection_run_ids=[running], bootstrap_generation_id=generation
    )
    assert certification.provenance_state == ol.NOT_CERTIFIABLE
    assert any("not complete" in reason for reason in certification.reasons)
    assert ol.freeze_provenance(conn, certification.freeze_identity) is not None
    conn.close()


def test_H3_a_ledger_verifies_the_certified_artifact_digest(tmp_path):
    """Building a ledger re-derives the artifact digest instead of trusting it."""

    conn = connect_database(tmp_path / "fpl.db")
    _world(conn)
    certification = _certified_freeze(conn, values={DGW_PLAYER: 6.0})
    ledger = ol.build_reality_ledger(conn, freeze_identity=certification.freeze_identity)
    assert ledger.artifact_verified is True
    assert ledger.prediction_artifact_sha256 == certification.prediction_artifact_sha256

    # A store whose runs grew AFTER certification: reachable on a database
    # written before the m017 trigger existed, so the trigger is dropped here to
    # demonstrate exactly that case.
    conn.execute("DROP TRIGGER frozen_predictions_no_insert_after_certification")
    analytics.freeze_prediction(
        conn,
        certification.projection_run_ids[0],
        kind=KIND_POINTS,
        player_id=SGW_A_PLAYER,
        event=EVENT,
        payload={"value": 9.0},
        model_version=analytics.BASELINE_MODEL_VERSION,
    )
    with pytest.raises(ol.OutcomeLedgerError, match="changed after certification"):
        ol.build_reality_ledger(conn, freeze_identity=certification.freeze_identity)
    conn.close()


def test_H4_an_explicit_fetch_that_contradicts_every_run_is_rejected(tmp_path):
    """A fetch the runs contradict is not this freeze's provenance."""

    conn = connect_database(tmp_path / "fpl.db")
    _world(conn)
    recorded = _fetch_run(conn)
    declared = _fetch_run(conn)
    generation = _generation(conn, fetch_run_id=declared)
    run = _freeze(
        conn, family=FAMILY_POINTS, version=analytics.BASELINE_MODEL_VERSION,
        kind=KIND_POINTS, values={DGW_PLAYER: 6.0},
        official_run_ids={"fetch": {"run_id": recorded}},
    )
    certification = ol.certify_prediction_freeze(
        conn, projection_run_ids=[run], bootstrap_generation_id=generation,
        official_fetch_run_id=declared,
    )
    assert certification.provenance_state == ol.NOT_CERTIFIABLE
    assert any("disagrees with every run's persisted provenance" in r for r in certification.reasons)
    assert certification.official_fetch_run_id is None
    # The same generation certifies when the declared fetch IS the recorded one.
    again = _freeze(
        conn, family=FAMILY_POINTS, version=analytics.BASELINE_MODEL_VERSION,
        kind=KIND_POINTS, values={DGW_PLAYER: 6.0},
        official_run_ids={"fetch": {"run_id": declared}},
    )
    accepted = ol.certify_prediction_freeze(
        conn, projection_run_ids=[again], bootstrap_generation_id=generation,
        official_fetch_run_id=declared,
    )
    assert accepted.provenance_state == ol.GENERATION_CERTIFIED, accepted.reasons
    conn.close()


def test_H5_one_run_per_model_family_is_enforced(tmp_path):
    """Two runs of one family make two claims about one key, so they cannot certify."""

    conn = connect_database(tmp_path / "fpl.db")
    _world(conn)
    fetch = _fetch_run(conn)
    generation = _generation(conn, fetch_run_id=fetch)
    first = _freeze(
        conn, family=FAMILY_POINTS, version="baseline_v1.0.0", kind=KIND_POINTS,
        values={DGW_PLAYER: 6.0}, official_run_ids={"fetch": {"run_id": fetch}},
    )
    second = _freeze(
        conn, family=FAMILY_POINTS, version="baseline_v1.1.0", kind=KIND_POINTS,
        values={SGW_A_PLAYER: 3.0}, official_run_ids={"fetch": {"run_id": fetch}},
    )
    certification = ol.certify_prediction_freeze(
        conn, projection_run_ids=[first, second], bootstrap_generation_id=generation
    )
    assert certification.provenance_state == ol.NOT_CERTIFIABLE
    assert any("same model family twice" in reason for reason in certification.reasons)
    conn.close()


def test_H6_every_distinct_prediction_is_retained(tmp_path):
    """Two families freezing the same kind are two claims, not one overwritten key."""

    conn = connect_database(tmp_path / "fpl.db")
    _world(conn)
    fetch = _fetch_run(conn)
    generation = _generation(conn, fetch_run_id=fetch)
    runs = [
        _freeze(
            conn, family=FAMILY_POINTS, version=analytics.BASELINE_MODEL_VERSION,
            kind=KIND_POINTS, values={DGW_PLAYER: 6.0},
            official_run_ids={"fetch": {"run_id": fetch}},
        ),
        _freeze(
            conn, family="xpts_v1", version="xpts_v1.0.0", kind=KIND_POINTS,
            values={DGW_PLAYER: 4.0}, official_run_ids={"fetch": {"run_id": fetch}},
        ),
    ]
    certification = ol.certify_prediction_freeze(
        conn, projection_run_ids=runs, bootstrap_generation_id=generation
    )
    assert certification.provenance_state == ol.GENERATION_CERTIFIED, certification.reasons
    _capture(conn, player_id=DGW_PLAYER, fixture_id=FIXTURE_1)
    _capture(conn, player_id=DGW_PLAYER, fixture_id=FIXTURE_2, total_points=3)
    ledger = ol.build_reality_ledger(conn, freeze_identity=certification.freeze_identity)
    rows = [row for row in ledger.rows if row["player_id"] == DGW_PLAYER]
    assert len(rows) == 2
    assert {row["model_family"] for row in rows} == {FAMILY_POINTS, "xpts_v1"}
    assert {row["frozen_predicted_value"] for row in rows} == {6.0, 4.0}
    assert all(row["evaluation_state"] == ol.EVALUATED for row in rows)
    conn.close()


def test_H7_certification_does_not_commit_the_callers_transaction(tmp_path):
    """The connection's transaction belongs to the caller, not to a certificate."""

    conn = connect_database(tmp_path / "fpl.db")
    _world(conn)
    fetch = _fetch_run(conn)
    generation = _generation(conn, fetch_run_id=fetch)
    run = _freeze(
        conn, family=FAMILY_POINTS, version=analytics.BASELINE_MODEL_VERSION,
        kind=KIND_POINTS, values={DGW_PLAYER: 6.0},
        official_run_ids={"fetch": {"run_id": fetch}},
    )
    assert conn.in_transaction, "the caller's own work is in flight"
    certification = ol.certify_prediction_freeze(
        conn, projection_run_ids=[run], bootstrap_generation_id=generation
    )
    assert certification.provenance_state == ol.GENERATION_CERTIFIED
    assert conn.in_transaction, "certification must not close the caller's transaction"
    assert conn.execute("SELECT COUNT(*) FROM prediction_freeze_runs").fetchone()[0] == 1
    # Rolling the caller's transaction back discards the certificate with it: had
    # certification committed, this row would survive as an orphan.
    conn.rollback()
    assert conn.execute("SELECT COUNT(*) FROM prediction_freeze_provenance").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM prediction_freeze_runs").fetchone()[0] == 0
    conn.close()


def test_H8_an_explicitly_provisional_capture_is_never_promoted(tmp_path):
    """A capture that says PROVISIONAL stays provisional, whatever arrived since."""

    conn = connect_database(tmp_path / "fpl.db")
    _world(conn)
    certification = _certified_freeze(conn, values={SGW_B_PLAYER: 5.0})
    result = ol.capture_observation(
        conn,
        grain=ol.GRAIN_PLAYER_FIXTURE,
        event=EVENT,
        player_id=SGW_B_PLAYER,
        fixture_id=FIXTURE_1,
        fields={"minutes": 90, "starts": 1, "total_points": 6, "bonus": 1, "bps": 30},
        source_name="event_live_pre_review",
        captured_at=CAPTURE_1,
        observation_state=ol.OBSERVATION_PROVISIONAL,
    )
    assert result.observation_state == ol.OBSERVATION_PROVISIONAL
    stored = ol.observation_captures(conn, event=EVENT, player_id=SGW_B_PLAYER)[0]
    # The event IS officially final and the capture time is after finalisation...
    assert ol.event_finality(conn, EVENT) == planning.EVENT_STATE_FINAL
    assert stored["official_final_at"] is None
    # ...and it is STILL provisional, so it cannot enter final evaluation.
    assert stored["observation_state"] == ol.OBSERVATION_PROVISIONAL
    ledger = ol.build_reality_ledger(conn, freeze_identity=certification.freeze_identity)
    row = next(entry for entry in ledger.rows if entry["player_id"] == SGW_B_PLAYER)
    assert row["exclusion_reason"] == ol.OUTCOME_NOT_FINALISED
    assert row["evaluation_state"] == ol.EXCLUDED
    assert row["observation_state"] == ol.OBSERVATION_PROVISIONAL
    conn.close()


def test_H12_a_ledger_refuses_a_run_that_is_no_longer_complete(tmp_path):
    """A reopenable run is not a closed artifact, so its ledger is refused too."""

    conn = connect_database(tmp_path / "fpl.db")
    _world(conn)
    certification = _certified_freeze(conn, values={DGW_PLAYER: 6.0})
    run_id = certification.projection_run_ids[0]
    # A store that recorded a certificate while its run was still open -- only
    # reachable before m006/m017, so the completed-run lock is dropped here to
    # show the case the ledger must refuse rather than read.
    conn.execute("DROP TRIGGER projection_runs_completed_immutable")
    conn.execute("UPDATE projection_runs SET status='running' WHERE id=?", (run_id,))
    with pytest.raises(ol.OutcomeLedgerError, match="not complete"):
        ol.build_reality_ledger(conn, freeze_identity=certification.freeze_identity)
    conn.close()


def test_H13_every_run_must_record_a_readable_agreeing_fetch(tmp_path):
    """The freeze's fetch identity is required from EVERY run, and read out of it.

    A fetch only some of the runs recorded is not the freeze's provenance; a
    record that exists and cannot be read is not the same thing as an absence;
    and an explicitly declared fetch is only ever an ASSERTION against what the
    runs persisted -- with nothing persisted there is nothing to assert against,
    so a caller cannot attach a provenance the runs never carried.
    """

    conn = connect_database(tmp_path / "fpl.db")
    _world(conn)
    fetch = _fetch_run(conn)
    generation = _generation(conn, fetch_run_id=fetch)

    def _run(*, family: str, version: str, run_ids, finish: bool = True) -> int:
        return _freeze(
            conn, family=family, version=version,
            kind=KIND_POINTS if family == FAMILY_POINTS else KIND_MINUTES,
            values={DGW_PLAYER: 6.0}, official_run_ids=run_ids, finish=finish,
        )

    def _run_recording(raw: str) -> int:
        """A complete run whose persisted provenance was written as ``raw``."""

        run_id = _run(family=FAMILY_POINTS, version=analytics.BASELINE_MODEL_VERSION, run_ids={}, finish=False)
        conn.execute("UPDATE projection_runs SET official_run_ids=? WHERE id=?", (raw, run_id))
        analytics.finish_projection_run(conn, run_id, "complete")
        return run_id

    # PARTIAL: one run records the fetch, the other records nothing at all.
    partial = ol.certify_prediction_freeze(
        conn,
        projection_run_ids=[
            _run(
                family=FAMILY_POINTS, version=analytics.BASELINE_MODEL_VERSION,
                run_ids={"fetch": {"run_id": fetch}},
            ),
            _run(family=FAMILY_MINUTES, version="minutes_v1.4.0", run_ids={}),
        ],
        bootstrap_generation_id=generation,
    )
    assert partial.provenance_state == ol.NOT_CERTIFIABLE
    assert partial.official_fetch_run_id is None
    assert any("record no official fetch provenance" in reason for reason in partial.reasons)

    # An explicit fetch cannot stand in for the provenance no run recorded: it is
    # an assertion ABOUT persisted provenance, not a substitute for it.
    declared = ol.certify_prediction_freeze(
        conn,
        projection_run_ids=[
            _run(family=FAMILY_POINTS, version=analytics.BASELINE_MODEL_VERSION, run_ids={})
        ],
        official_fetch_run_id=fetch,
    )
    assert declared.provenance_state == ol.NOT_CERTIFIABLE
    assert declared.official_fetch_run_id is None
    assert any("only some of the runs state" in reason for reason in declared.reasons)

    # MALFORMED: a record that exists and cannot be read is not an absence, so it
    # is refused rather than quietly read as "this run recorded nothing".
    for raw, expected in (
        ('{"fetch": "not-an-object"}', "is not an object"),
        ('{"fetch": {"run_id":', "not readable JSON"),
        ('{"fetch": {}}', "no run_id"),
        ('{"fetch": {"run_id": "seven"}}', "non-integer"),
        ("[]", "not an object"),
    ):
        broken = ol.certify_prediction_freeze(
            conn, projection_run_ids=[_run_recording(raw)], bootstrap_generation_id=generation
        )
        assert broken.provenance_state == ol.NOT_CERTIFIABLE
        assert broken.official_fetch_run_id is None
        assert any(expected in reason for reason in broken.reasons), (raw, broken.reasons)

    # ABSENT: not one run recorded a fetch, and the freeze claims certification
    # from a generation anyway.  A generation cannot supply a provenance the runs
    # never carried, so the claim is refused rather than read as a freeze that
    # happened to record nothing yet was certified regardless.
    nothing_recorded = ol.certify_prediction_freeze(
        conn,
        projection_run_ids=[
            _run(family=FAMILY_POINTS, version=analytics.BASELINE_MODEL_VERSION, run_ids={})
        ],
        bootstrap_generation_id=generation,
    )
    assert nothing_recorded.provenance_state == ol.NOT_CERTIFIABLE
    assert nothing_recorded.official_fetch_run_id is None
    assert any("no official fetch identity to match" in reason for reason in nothing_recorded.reasons)
    assert ol.freeze_provenance(conn, nothing_recorded.freeze_identity)["provenance_state"] == (
        ol.NOT_CERTIFIABLE
    )

    # DISAGREEING: every run recorded a fetch, but not the same one, so there is
    # no single recorded identity to certify from -- and the refusal stands even
    # with no generation to compare against, because the legacy branch answers
    # for MISSING provenance and this is provenance the runs contradict.
    other_fetch = _fetch_run(conn)
    disagreeing = ol.certify_prediction_freeze(
        conn,
        projection_run_ids=[
            _run(
                family=FAMILY_POINTS, version=analytics.BASELINE_MODEL_VERSION,
                run_ids={"fetch": {"run_id": fetch}},
            ),
            _run(
                family=FAMILY_MINUTES, version="minutes_v1.4.0",
                run_ids={"fetch": {"run_id": other_fetch}},
            ),
        ],
    )
    assert disagreeing.provenance_state == ol.NOT_CERTIFIABLE
    assert disagreeing.official_fetch_run_id is None
    assert any("record different official fetch runs" in reason for reason in disagreeing.reasons)

    # The intact case still certifies: every run records the same fetch.
    good = ol.certify_prediction_freeze(
        conn,
        projection_run_ids=[
            _run(
                family=FAMILY_POINTS, version=analytics.BASELINE_MODEL_VERSION,
                run_ids={"fetch": {"run_id": fetch}},
            )
        ],
        bootstrap_generation_id=generation,
    )
    assert good.provenance_state == ol.GENERATION_CERTIFIED, good.reasons
    assert good.official_fetch_run_id == fetch
    conn.close()


def test_H14_a_provenance_free_running_run_is_refused_not_legacied(tmp_path):
    """The legacy branch may not swallow an incomplete-run or duplicate-family refusal.

    Legacy is a statement about MISSING PROVENANCE, not a blanket over every run
    that happens to record none.  A run that is still open, or a freeze that
    names one model family twice, is a refusal -- and a refusal the legacy branch
    hid would never reach a reviewer.
    """

    conn = connect_database(tmp_path / "fpl.db")
    _world(conn)
    running = _freeze(
        conn, family=FAMILY_POINTS, version=analytics.BASELINE_MODEL_VERSION,
        kind=KIND_POINTS, values={DGW_PLAYER: 6.0}, official_run_ids={}, finish=False,
    )
    certification = ol.certify_prediction_freeze(conn, projection_run_ids=[running])
    assert certification.provenance_state == ol.NOT_CERTIFIABLE
    assert any("not complete" in reason for reason in certification.reasons)
    assert ol.freeze_provenance(conn, certification.freeze_identity)["provenance_state"] == (
        ol.NOT_CERTIFIABLE
    )

    first = _freeze(
        conn, family=FAMILY_POINTS, version="baseline_v1.0.0", kind=KIND_POINTS,
        values={DGW_PLAYER: 6.0}, official_run_ids={},
    )
    second = _freeze(
        conn, family=FAMILY_POINTS, version="baseline_v1.1.0", kind=KIND_POINTS,
        values={SGW_A_PLAYER: 3.0}, official_run_ids={},
    )
    duplicated = ol.certify_prediction_freeze(conn, projection_run_ids=[first, second])
    assert duplicated.provenance_state == ol.NOT_CERTIFIABLE
    assert any("same model family twice" in reason for reason in duplicated.reasons)
    # ...while a provenance-free freeze that is genuinely sound stays legacy.
    legacy = _freeze(
        conn, family=FAMILY_MINUTES, version="minutes_v1.4.0", kind=KIND_MINUTES,
        values={DGW_PLAYER: 6.0}, official_run_ids={},
    )
    assert ol.certify_prediction_freeze(
        conn, projection_run_ids=[legacy]
    ).provenance_state == ol.LEGACY_PROVENANCE
    conn.close()


def test_H15_a_concurrent_insertion_during_certification_is_refused(tmp_path, monkeypatch):
    """Hashing and closure are ONE atomic step, proved with two connections.

    The first connection decides the certificate from the artifact it read.  A
    second connection then appends to that very artifact -- in the window
    between the decision read and the write -- and commits.  The certificate
    must be refused rather than written with a digest that stopped describing
    the artifact the moment the other connection touched it; once the artifact
    has settled, certifying it again is fine, which is what shows the refusal is
    about the race and not about the extra row.
    """

    path = tmp_path / "fpl.db"
    conn = connect_database(path)
    other = connect_database(path)
    _world(conn)
    fetch = _fetch_run(conn)
    generation = _generation(conn, fetch_run_id=fetch)
    run = _freeze(
        conn, family=FAMILY_POINTS, version=analytics.BASELINE_MODEL_VERSION,
        kind=KIND_POINTS, values={DGW_PLAYER: 6.0},
        official_run_ids={"fetch": {"run_id": fetch}},
    )
    conn.commit()

    real = ol.prediction_artifact_digest
    racing = {"inserted": False}

    def _digest_racing_with_a_second_connection(target, run_ids):
        result = real(target, run_ids)
        if not racing["inserted"]:
            racing["inserted"] = True
            analytics.freeze_prediction(
                other,
                run,
                kind=KIND_POINTS,
                player_id=SGW_A_PLAYER,
                event=EVENT,
                payload={"value": 9.0},
                model_version=analytics.BASELINE_MODEL_VERSION,
            )
            other.commit()
        return result

    monkeypatch.setattr(ol, "prediction_artifact_digest", _digest_racing_with_a_second_connection)
    with pytest.raises(ol.OutcomeLedgerError, match="still moving"):
        ol.certify_prediction_freeze(
            conn, projection_run_ids=[run], bootstrap_generation_id=generation
        )
    assert racing["inserted"] is True
    assert conn.execute("SELECT COUNT(*) FROM prediction_freeze_provenance").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM prediction_freeze_runs").fetchone()[0] == 0

    # The settled artifact certifies cleanly, and the ledger can verify it.
    certification = ol.certify_prediction_freeze(
        conn, projection_run_ids=[run], bootstrap_generation_id=generation
    )
    assert certification.provenance_state == ol.GENERATION_CERTIFIED, certification.reasons
    ledger = ol.build_reality_ledger(conn, freeze_identity=certification.freeze_identity)
    assert ledger.artifact_verified is True
    other.close()
    conn.close()


def test_H16_a_second_connection_cannot_write_inside_the_closure(tmp_path, monkeypatch):
    """The closure re-reads its evidence inside its own transaction.

    The first connection certifies inside a transaction the CALLER owns.  While
    the closure re-reads what it is certifying, a second connection tries to
    append to the same artifact: the database refuses the interleave, so the
    certificate can only ever be written from the state its own transaction
    observed.  The caller's transaction is preserved throughout -- it is still
    the caller's to commit or discard, certificate included.
    """

    path = tmp_path / "fpl.db"
    conn = connect_database(path)
    other = connect_database(path)
    other.execute("PRAGMA busy_timeout=0")
    _world(conn)
    fetch = _fetch_run(conn)
    generation = _generation(conn, fetch_run_id=fetch)
    run = _freeze(
        conn, family=FAMILY_POINTS, version=analytics.BASELINE_MODEL_VERSION,
        kind=KIND_POINTS, values={DGW_PLAYER: 6.0},
        official_run_ids={"fetch": {"run_id": fetch}},
    )

    real = ol.prediction_artifact_digest
    calls = {"count": 0}
    refused: list[str] = []

    def _digest_with_a_second_connection_writing(target, run_ids):
        calls["count"] += 1
        if calls["count"] == 3:
            # The third read is the closure's own, inside its savepoint.
            try:
                analytics.freeze_prediction(
                    other,
                    run,
                    kind=KIND_POINTS,
                    player_id=SGW_A_PLAYER,
                    event=EVENT,
                    payload={"value": 9.0},
                    model_version=analytics.BASELINE_MODEL_VERSION,
                )
                other.commit()
            except sqlite3.OperationalError as failure:
                refused.append(str(failure))
        return real(target, run_ids)

    monkeypatch.setattr(ol, "prediction_artifact_digest", _digest_with_a_second_connection_writing)
    certification = ol.certify_prediction_freeze(
        conn, projection_run_ids=[run], bootstrap_generation_id=generation
    )
    assert calls["count"] == 3, "the closure re-reads the evidence it is certifying"
    assert refused and "database is locked" in refused[0], refused
    assert certification.provenance_state == ol.GENERATION_CERTIFIED, certification.reasons
    assert conn.execute(
        "SELECT COUNT(*) FROM frozen_predictions WHERE projection_run_id=?", (run,)
    ).fetchone()[0] == 1
    # The caller's transaction is still open, and still owns the certificate.
    assert conn.in_transaction
    assert ol.build_reality_ledger(conn, freeze_identity=certification.freeze_identity).artifact_verified
    conn.rollback()
    assert conn.execute("SELECT COUNT(*) FROM prediction_freeze_provenance").fetchone()[0] == 0
    other.close()
    conn.close()


def test_H17_a_backfill_is_bound_to_the_archived_content(tmp_path):
    """A valid archive capture proves identity and BYTES, never the claim.

    A manifest record can be satisfied perfectly while the observation attached
    to it is invented, because the manifest says which bytes were read and never
    what they say.  So every part of the claim is bound to the parsed archived
    content: each official field, each additional source field, the player, the
    fixture, the event and the club identity all have to be what the archive
    states, and the headline event total has to be that content's aggregation.
    """

    conn = connect_database(tmp_path / "fpl.db")
    _world(conn)
    row = _archived_element_row()
    root, record = _archived_observation(tmp_path, rows=[row], source="element_summary_10")
    assert _backfill(conn, root=root, record=record).inserted
    # An archive that states MORE than PE-5's minimum is exactly the case the
    # additional-field rule has to bind, so it gets its own capture.
    rich_root, rich_record = _archived_observation(
        tmp_path,
        rows=[_archived_element_row(influence=12.5, ict_index=8.1)],
        source="element_summary_10",
        observed_at=_t(5),
        raw_dir=tmp_path / "raw5",
    )

    def _refused(match: str, *, capture=None, **overrides):
        target_root, target_record = capture or (root, record)
        before = conn.execute("SELECT COUNT(*) FROM outcome_observation_captures").fetchone()[0]
        with pytest.raises(ol.BackfillEvidenceError, match=match):
            _backfill(conn, root=target_root, record=target_record, **overrides)
        assert conn.execute("SELECT COUNT(*) FROM outcome_observation_captures").fetchone()[0] == before

    # An official field the archive states differently is not the archive's fact,
    # and neither is one the archive leaves missing.
    _refused("not the claimed", total_points=9)
    _refused("not the claimed", bps=99)
    _refused("not the claimed", bonus=None)
    # An additional field the archive never states cannot be attached to it...
    _refused("does not state 'creates'", creates=3, capture=(rich_root, rich_record), captured_at=_t(5))
    # ...and one it does state cannot be re-valued.
    _refused(
        "does not state 'influence'", influence=7.0, capture=(rich_root, rich_record), captured_at=_t(5)
    )
    # A fixture the archived capture holds no row for.
    _refused("holds no observation", fixture_id=FIXTURE_2)
    # A club identity the archived row contradicts.
    _refused("states the club/team", team_id=TEAM_C)
    _refused("states the opponent", opponent_team_id=TEAM_A)
    _refused("was played at", event_time=KICKOFF_2)
    # A player the archived capture never mentions: the source slug is another
    # element's summary, so the claim is not about the bytes being offered.
    _refused("is the element summary of player", player_id=SGW_A_PLAYER)
    # The additional fields the archive DOES state are carried through verbatim.
    stored = _backfill(
        conn, root=rich_root, record=rich_record, captured_at=_t(5), influence=12.5, ict_index=8.1
    )
    assert stored.inserted
    kept = next(
        row
        for row in ol.observation_captures(conn, event=EVENT, player_id=DGW_PLAYER)
        if row["archive_capture_id"] == rich_record.capture_id
    )
    assert kept["payload"]["influence"] == 12.5
    assert kept["payload"]["ict_index"] == 8.1

    # Without the player in the source slug, the ROW's identity is what binds the
    # claim -- and a payload whose rows state no player at all cannot prove whose
    # observation it is.
    other_root, other_record = _archived_observation(
        tmp_path, rows=[_archived_element_row(player_id=SGW_A_PLAYER)], raw_dir=tmp_path / "raw2",
    )
    assert _backfill(
        conn, root=other_root, record=other_record, player_id=SGW_A_PLAYER
    ).inserted
    anonymous = _archived_element_row()
    anonymous.pop("element")
    anon_root, anon_record = _archived_observation(
        tmp_path, rows=[anonymous], source="element_summary", observed_at=_t(5), raw_dir=tmp_path / "raw3",
    )
    with pytest.raises(ol.BackfillEvidenceError, match="states no player identity"):
        _backfill(conn, root=anon_root, record=anon_record, captured_at=_t(5))

    # Event grain: the archived double gameweek is summed ONCE, so the headline
    # total cannot be one fixture's value copied into the event row.
    dgw_root, dgw_record = _archived_observation(
        tmp_path,
        rows=[
            _archived_element_row(fixture_id=FIXTURE_1),
            _archived_element_row(
                fixture_id=FIXTURE_2, kickoff_time=KICKOFF_2, team_a=TEAM_C, opponent_team=TEAM_C,
            ),
        ],
        source="element_summary_10",
        observed_at=_t(4),
        raw_dir=tmp_path / "raw4",
    )
    event_claim = {
        "minutes": 180, "starts": 2, "total_points": 12, "goals_scored": 2, "assists": 0,
        "clean_sheets": 0, "goals_conceded": 2, "saves": 0, "bonus": 2, "bps": 60,
        "yellow_cards": 0, "red_cards": 0, "penalties_saved": 0, "penalties_missed": 0,
        "own_goals": 0, "defensive_contribution": 0,
    }

    def _event_claim(**overrides):
        arguments = {
            "player_id": DGW_PLAYER,
            "fixture_id": None,
            "grain": ol.GRAIN_PLAYER_EVENT,
            "captured_at": _t(4),
            "source_name": "archived_element_summary",
            "source_identity": "element_summary_10",
            "source_payload_sha256": dgw_record.payload_sha256,
            "archive_capture_id": dgw_record.capture_id,
            "fetch_run_id": 7,
            "backfill": True,
            "archive_root": dgw_root,
        }
        arguments.update(event_claim)
        arguments.update(overrides)
        return _capture(conn, **arguments)

    assert _event_claim().inserted
    with pytest.raises(ol.BackfillEvidenceError, match="not the claimed"):
        _event_claim(total_points=6)
    conn.close()


def test_H18_a_backfill_is_bound_to_the_player_the_archive_names(tmp_path):
    """The rows must belong to the claimed player, whatever the payload is called.

    A parser is TOLD which player it is reading, so its records carry the
    caller's id no matter whose numbers they contain: the ``element`` a row
    states about itself is therefore the only thing that can bind a claim to a
    player.  Without that check the live endpoint -- which serves every player
    in one response -- would let one player's afternoon be filed as another's,
    and the file would look perfectly well formed while being about the wrong
    footballer.
    """

    conn = connect_database(tmp_path / "fpl.db")
    _world(conn)

    # The live payload belongs to SGW_B_PLAYER, who scored nine.  Claiming it for
    # the DGW player is claiming another footballer's fact.
    root, record = _archived_live(
        tmp_path, elements=[_live_element(player_id=SGW_B_PLAYER)], raw_dir=tmp_path / "live1"
    )
    with pytest.raises(ol.BackfillEvidenceError, match="none of which is the claimed player"):
        _backfill(conn, root=root, record=record, player_id=DGW_PLAYER)
    assert conn.execute("SELECT COUNT(*) FROM outcome_observation_captures").fetchone()[0] == 0

    # The same rule for an element summary: a payload named for one player whose
    # rows state another is not that player's evidence either.
    summary_root, summary_record = _archived_observation(
        tmp_path,
        rows=[_archived_element_row(player_id=SGW_B_PLAYER, total_points=9, bonus=3, bps=41)],
        source="element_summary_10",
        raw_dir=tmp_path / "summary1",
    )
    with pytest.raises(ol.BackfillEvidenceError, match="none of which is the claimed player"):
        _backfill(conn, root=summary_root, record=summary_record, player_id=DGW_PLAYER)

    # A payload that states the claimant at a DIFFERENT event is no evidence for
    # THIS event, so the archive offers nothing to bind the claim to.
    elsewhere_root, elsewhere_record = _archived_observation(
        tmp_path,
        rows=[_archived_element_row(event=OTHER_EVENT, fixture_id=FIXTURE_OTHER)],
        source="element_summary_10",
        observed_at=_t(12),
        raw_dir=tmp_path / "summary2",
    )
    with pytest.raises(ol.BackfillEvidenceError, match="holds no observation"):
        _backfill(conn, root=elsewhere_root, record=elsewhere_record, captured_at=_t(12))

    # The player's own rows are still exactly the evidence that binds, including
    # when the same response carries other players alongside him.
    both_root, both_record = _archived_live(
        tmp_path,
        elements=[
            _live_element(player_id=SGW_B_PLAYER, stats={"fixture": FIXTURE_1}),
            _live_element(player_id=DGW_PLAYER),
        ],
        observed_at=_t(5),
        raw_dir=tmp_path / "live2",
    )
    assert _backfill(
        conn, root=both_root, record=both_record, captured_at=_t(5),
        source_identity="event_live_4", fetch_run_id=7,
    ).inserted
    conn.close()


def test_H19_a_claim_the_archive_never_states_is_refused(tmp_path):
    """Silence in the archive is not evidence, for identity of any kind.

    A caller may name a real archived capture and then supply the parts it does
    not state -- a kickoff time, a club -- and the manifest will agree with every
    word, because the manifest knows which bytes were read and never what they
    say.  Identity the archive does not state is therefore refused rather than
    stored, which is the same rule that already governs the official fields.
    """

    conn = connect_database(tmp_path / "fpl.db")
    _world(conn)
    silent = _archived_element_row()
    silent.pop("kickoff_time")
    root, record = _archived_observation(
        tmp_path, rows=[silent], source="element_summary_10", raw_dir=tmp_path / "silent"
    )

    # The archived row states no kickoff, so a claimed event_time is an assertion.
    with pytest.raises(ol.BackfillEvidenceError, match="states no kickoff time"):
        _backfill(conn, root=root, record=record, event_time=KICKOFF_1)
    # ...and one that happens to equal the fixture's CURRENT kickoff is refused
    # too: the live fixture table is not the archive.
    with pytest.raises(ol.BackfillEvidenceError, match="states no kickoff time"):
        _backfill(conn, root=root, record=record, event_time=_fixture_kickoff(conn, FIXTURE_1))

    # A club the archived row never states is refused for the same reason.
    clubless = _archived_element_row()
    for key in ("team_h", "team_a", "opponent_team", "was_home"):
        clubless.pop(key)
    club_root, club_record = _archived_observation(
        tmp_path, rows=[clubless], source="element_summary_10",
        observed_at=_t(5), raw_dir=tmp_path / "clubless",
    )
    with pytest.raises(ol.BackfillEvidenceError, match="states no club/team"):
        _backfill(conn, root=club_root, record=club_record, captured_at=_t(5), team_id=TEAM_A)
    with pytest.raises(ol.BackfillEvidenceError, match="states no opponent"):
        _backfill(
            conn, root=club_root, record=club_record, captured_at=_t(5), opponent_team_id=TEAM_B
        )

    # Stating none of it is still a faithful copy of what the archive says.
    stored = _backfill(conn, root=root, record=record)
    assert stored.inserted
    kept = next(
        row
        for row in ol.observation_captures(conn, event=EVENT, player_id=DGW_PLAYER)
        if row["archive_capture_id"] == record.capture_id
    )
    assert kept["event_time"] is None
    assert conn.execute("SELECT COUNT(*) FROM outcome_observation_captures").fetchone()[0] == 1
    conn.close()


def test_H20_a_nested_event_live_backfill_reads_the_explain_legs(tmp_path):
    """A real live payload backfills from its explain legs, not its event totals.

    ``event/live`` serves the whole pool in one response, with each player's
    EVENT totals at the top level and the fixture-specific numbers nested under
    ``explain``.  A double gameweek therefore has exactly one correct reading --
    one row per explain leg -- and copying the top-level totals onto a fixture
    row would score the same afternoon twice.
    """

    conn = connect_database(tmp_path / "fpl.db")
    _world(conn)
    root, record = _archived_live(
        tmp_path,
        elements=[
            _live_element(
                player_id=DGW_PLAYER,
                # The event totals: what the two legs below add up to.  Nothing
                # may be read from these.
                stats={"fixture": FIXTURE_1, "minutes": 152, "total_points": 9, "bonus": 1, "bps": 49},
                explain=[
                    _live_leg(fixture_id=FIXTURE_1, kickoff_time=KICKOFF_1),
                    _live_leg(
                        fixture_id=FIXTURE_2, opponent_team=TEAM_C, kickoff_time=KICKOFF_2,
                        minutes=62, total_points=3, bonus=0, bps=19,
                    ),
                ],
            )
        ],
    )

    def _live_claim(**overrides):
        arguments = {
            "player_id": DGW_PLAYER,
            "fixture_id": FIXTURE_1,
            "captured_at": _t(6),
            "source_name": "archived_event_live",
            "source_identity": "event_live_4",
            "source_payload_sha256": record.payload_sha256,
            "archive_capture_id": record.capture_id,
            "fetch_run_id": 7,
            "backfill": True,
            "archive_root": root,
        }
        arguments.update(overrides)
        return _capture(conn, **arguments)

    # The FIRST leg's explain entry is the fixture-grain fact: 90 minutes, 6
    # points, 30 bps -- not the event totals that also cover the second leg.
    assert _live_claim().inserted
    # The second leg is its own row, with its own numbers.  The live payload
    # states the OPPONENT of each leg (it carries no home/away team pair), so
    # that is what a claim may name.
    assert _live_claim(
        fixture_id=FIXTURE_2, opponent_team_id=TEAM_C,
        minutes=62, total_points=3, bonus=0, bps=19, event_time=KICKOFF_2,
    ).inserted
    # ...and the event totals cannot be pinned onto either leg.
    with pytest.raises(ol.BackfillEvidenceError, match="not the claimed"):
        _live_claim(fixture_id=FIXTURE_2, minutes=152)

    stored = {
        int(row["fixture_id"]): row
        for row in ol.observation_captures(conn, event=EVENT, player_id=DGW_PLAYER)
        if row["archive_capture_id"] == record.capture_id
    }
    assert sorted(stored) == [FIXTURE_1, FIXTURE_2]
    assert (stored[FIXTURE_1]["minutes"], stored[FIXTURE_1]["total_points"]) == (90, 6)
    assert (stored[FIXTURE_2]["minutes"], stored[FIXTURE_2]["total_points"]) == (62, 3)

    # Event grain sums the legs ONCE, and carries no per-fixture identity: the
    # headline row spans two fixtures whose opponents and kickoffs differ, so
    # stating either leg's would be asserting a fixture fact about an event.
    event_root, event_record = _archived_live(
        tmp_path,
        elements=[
            _live_element(
                player_id=DGW_PLAYER,
                stats={},
                explain=[
                    _live_leg(fixture_id=FIXTURE_1, kickoff_time=KICKOFF_1),
                    _live_leg(
                        fixture_id=FIXTURE_2, opponent_team=TEAM_C, kickoff_time=KICKOFF_2,
                        minutes=62, total_points=3, bonus=0, bps=19,
                    ),
                ],
            )
        ],
        observed_at=_t(4),
        raw_dir=tmp_path / "live_event",
    )
    assert _capture(
        conn,
        grain=ol.GRAIN_PLAYER_EVENT,
        player_id=DGW_PLAYER,
        fixture_id=None,
        minutes=152, starts=2, total_points=9, goals_scored=2, assists=0, clean_sheets=0,
        goals_conceded=2, saves=0, bonus=1, bps=49, yellow_cards=0, red_cards=0,
        penalties_saved=0, penalties_missed=0, own_goals=0, defensive_contribution=0,
        captured_at=_t(4),
        source_name="archived_event_live",
        source_identity="event_live_4",
        source_payload_sha256=event_record.payload_sha256,
        archive_capture_id=event_record.capture_id,
        fetch_run_id=7,
        backfill=True,
        archive_root=event_root,
    ).inserted
    headline = next(
        row
        for row in ol.observation_captures(conn, event=EVENT, player_id=DGW_PLAYER)
        if row["grain"] == ol.GRAIN_PLAYER_EVENT
    )
    assert headline["total_points"] == 9
    # The legs disagree about the opponent and the kickoff, so the headline row
    # states neither: an event total is not a fixture.
    assert headline["event_time"] is None
    assert headline["opponent_team_id"] is None
    assert headline["team_id"] is None
    # A per-fixture identity cannot be attached to the headline row at all.
    with pytest.raises(ol.BackfillEvidenceError, match="not archived evidence"):
        _capture(
            conn,
            grain=ol.GRAIN_PLAYER_EVENT,
            player_id=DGW_PLAYER,
            fixture_id=None,
            minutes=152, starts=2, total_points=9, goals_scored=2, assists=0, clean_sheets=0,
            goals_conceded=2, saves=0, bonus=1, bps=49, yellow_cards=0, red_cards=0,
            penalties_saved=0, penalties_missed=0, own_goals=0, defensive_contribution=0,
            captured_at=_t(4),
            source_name="archived_event_live",
            source_identity="event_live_4",
            source_payload_sha256=event_record.payload_sha256,
            archive_capture_id=event_record.capture_id,
            fetch_run_id=7,
            backfill=True,
            archive_root=event_root,
            opponent_team_id=TEAM_B,
        )
    conn.close()


def test_H21_a_generation_that_moves_under_the_certificate_is_refused(tmp_path, monkeypatch):
    """The closure re-reads the generation BY ID, not from its own decision copy.

    A certificate records which accepted official-player generation supplied a
    freeze, so the row that has to hold still while the certificate is written is
    the row in the table.  Storing the decision's own dictionary and comparing it
    with itself would prove nothing: a generation edited in between -- its
    element set, its count, its acceptance -- would be certified from a truth
    that no longer exists anywhere.  Each case below moves exactly one field of
    that row in the window between the decision read and the closure's own, and
    every one of them is refused with nothing written.
    """

    cases = {
        "element ids": ("element_ids_json", json.dumps([DGW_PLAYER, SGW_B_PLAYER])),
        "element digest": ("element_ids_sha256", element_id_sha256([DGW_PLAYER, SGW_B_PLAYER])),
        "official element count": ("official_element_count", 2),
        "acceptance": ("accepted", 0),
        "acceptance rule": ("acceptance_rule_version", "BOOTSTRAP_GENERATION_ACCEPTANCE v2"),
    }
    for label, (column, moved) in cases.items():
        # One database per case: a certificate is recorded once per freeze, so a
        # shared world would turn the later cases into duplicate-provenance
        # refusals instead of the race this test is about.
        path = tmp_path / str(column) / "fpl.db"
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = connect_database(path)
        other = connect_database(path)
        _world(conn)
        fetch = _fetch_run(conn)
        generation = _generation(conn, fetch_run_id=fetch)
        run = _freeze(
            conn, family=FAMILY_POINTS, version=analytics.BASELINE_MODEL_VERSION,
            kind=KIND_POINTS, values={DGW_PLAYER: 6.0},
            official_run_ids={"fetch": {"run_id": fetch}},
        )
        conn.commit()

        real = ol._bootstrap_generation
        reads = {"count": 0}

        def _moving(target, generation_id, _column=column, _moved=moved, _real=real):
            record = _real(target, generation_id)
            reads["count"] += 1
            if reads["count"] == 2:
                # The second read is the closure's own, so the row moves after
                # the certificate was decided from it and before it is written.
                other.execute(
                    f"UPDATE bootstrap_generations SET {_column}=? WHERE id=?",
                    (_moved, int(generation_id)),
                )
                other.commit()
            return record

        monkeypatch.setattr(ol, "_bootstrap_generation", _moving)
        with pytest.raises(ol.OutcomeLedgerError, match="bootstrap generation"):
            ol.certify_prediction_freeze(
                conn, projection_run_ids=[run], bootstrap_generation_id=generation
            )
        monkeypatch.undo()
        assert reads["count"] >= 2, f"the {label} mutation never ran"
        assert conn.execute("SELECT COUNT(*) FROM prediction_freeze_provenance").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM prediction_freeze_runs").fetchone()[0] == 0
        # The row that moved is the row that was examined; restoring it lets the
        # very same freeze certify, which shows the refusal was about the change.
        other.execute(
            f"UPDATE bootstrap_generations SET {column}=? WHERE id=?",
            (
                json.dumps(list(SYNTHETIC_POOL)) if column == "element_ids_json"
                else element_id_sha256(list(SYNTHETIC_POOL)) if column == "element_ids_sha256"
                else len(SYNTHETIC_POOL) if column == "official_element_count"
                else 1 if column == "accepted"
                else "BOOTSTRAP_GENERATION_ACCEPTANCE v1",
                int(generation),
            ),
        )
        other.commit()
        certification = ol.certify_prediction_freeze(
            conn, projection_run_ids=[run], bootstrap_generation_id=generation
        )
        assert certification.provenance_state == ol.GENERATION_CERTIFIED, certification.reasons
        conn.close()
        other.close()


def test_H22_the_closure_reread_catches_a_generation_the_caller_cached(tmp_path, monkeypatch):
    """A dictionary the caller already holds is not evidence about the table.

    The same race seen from the other side: the row is moved first, and the
    certificate is then requested with a generation id whose row no longer says
    what it said when the decision was made.  The closure's own read is what
    notices, because the value it would record came from the table.
    """

    path = tmp_path / "fpl.db"
    conn = connect_database(path)
    other = connect_database(path)
    _world(conn)
    fetch = _fetch_run(conn)
    generation = _generation(conn, fetch_run_id=fetch)
    run = _freeze(
        conn, family=FAMILY_POINTS, version=analytics.BASELINE_MODEL_VERSION,
        kind=KIND_POINTS, values={DGW_PLAYER: 6.0},
        official_run_ids={"fetch": {"run_id": fetch}},
    )
    conn.commit()

    # The generation is read once to decide the certificate and once more inside
    # the closure, and the SECOND connection retires the row after the decision
    # was taken from it.  The certificate must be refused, because the generation
    # it would record is not the generation that exists.
    real = ol._bootstrap_generation
    reads = {"count": 0}

    def _moving_between_the_decision_and_the_closure(target, generation_id):
        record = real(target, generation_id)
        reads["count"] += 1
        if reads["count"] == 2:
            # The second read is the closure's own; the row moves after the
            # decision read it and before the certificate is written.
            other.execute(
                "UPDATE bootstrap_generations SET accepted=0 WHERE id=?", (int(generation_id),)
            )
            other.commit()
        return record

    monkeypatch.setattr(ol, "_bootstrap_generation", _moving_between_the_decision_and_the_closure)
    with pytest.raises(ol.OutcomeLedgerError, match="bootstrap generation"):
        ol.certify_prediction_freeze(
            conn, projection_run_ids=[run], bootstrap_generation_id=generation
        )
    assert reads["count"] >= 2, "the closure re-read the generation it was certified with"
    assert conn.execute("SELECT COUNT(*) FROM prediction_freeze_provenance").fetchone()[0] == 0
    conn.close()
    other.close()


def test_H9_a_historical_read_resolves_from_pinned_evidence(tmp_path):
    """Pool, club, fixtures and finality at a cutoff come from pinned evidence.

    The demonstrated case: a ledger read at a cutoff, then the mutable tables
    move on -- a fixture is un-finished, an event is un-finalised, a player is
    deactivated and transferred, a placeholder row appears.  The same historical
    read must be byte-identical afterwards, and must not have seen any of it.
    """

    conn = connect_database(tmp_path / "fpl.db")
    _world(conn)
    certification = _certified_freeze(conn, values={DGW_PLAYER: 6.0, BLANK_PLAYER: 4.0})
    _capture(conn, player_id=DGW_PLAYER, fixture_id=FIXTURE_1, captured_at=CAPTURE_1, minutes=90,
             total_points=6, bonus=1, bps=30)
    _capture(conn, player_id=DGW_PLAYER, fixture_id=FIXTURE_2, captured_at=CAPTURE_1, minutes=62,
             total_points=3, bonus=0, bps=19)
    archive = _archive_world(tmp_path, observed_at=AS_OF)

    before = ol.build_reality_ledger(
        conn, freeze_identity=certification.freeze_identity, as_of=AS_OF, archive_root=archive
    )
    assert before.world_source == ol.WORLD_PINNED
    assert before.world_unavailable == ()
    scored = next(row for row in before.rows if row["player_id"] == DGW_PLAYER)
    assert scored["evaluation_state"] == ol.EVALUATED
    assert scored["official_outcome"]["total_points"] == 9
    assert scored["team_id"] == TEAM_A

    # Everything the mutable tables can get wrong about that instant, moved on.
    conn.execute("UPDATE fixtures SET finished=0, started=0 WHERE id IN (?,?)",
                 (FIXTURE_1, FIXTURE_2))
    conn.execute("UPDATE events SET finished=0, data_checked=0 WHERE id=?", (EVENT,))
    conn.execute("UPDATE players SET is_active=0, team_id=? WHERE id=?", (TEAM_C, DGW_PLAYER))
    conn.execute(
        """INSERT INTO player_gameweeks(player_id, event, fixture_id, minutes, source, raw_json, updated_at)
           VALUES (?,?,?,?,?,?,?)""",
        (DGW_PLAYER, EVENT, FIXTURE_1, 0, "element_summary", "{}", _t(1)),
    )

    after = ol.build_reality_ledger(
        conn, freeze_identity=certification.freeze_identity, as_of=AS_OF, archive_root=archive
    )
    assert after.rows == before.rows
    assert after.ledger_digest == before.ledger_digest
    assert after.population_digest == before.population_digest
    assert after.world_source == ol.WORLD_PINNED
    # The read is not merely stable: it never saw the mutations.
    scored_after = next(row for row in after.rows if row["player_id"] == DGW_PLAYER)
    assert scored_after["evaluation_state"] == ol.EVALUATED
    assert scored_after["team_id"] == TEAM_A
    assert scored_after["world_evidence"] != []
    blank = next(row for row in after.rows if row["player_id"] == BLANK_PLAYER)
    assert blank["outcome_state"] == "BLANK"

    # The control: the SAME store read at "now" is a different, current-state
    # ledger, which is what makes the pinned read above meaningful rather than
    # an artefact of the mutations being invisible.
    current = ol.build_reality_ledger(conn, freeze_identity=certification.freeze_identity)
    assert current.world_source == ol.WORLD_CURRENT
    current_row = next(row for row in current.rows if row["player_id"] == DGW_PLAYER)
    assert current_row["evaluation_state"] == ol.EXCLUDED
    assert current_row["exclusion_reason"] in {
        ol.TARGET_FIXTURE_NOT_PLAYED, ol.OUTCOME_NOT_FINALISED, ol.PLAYER_NOT_IN_OFFICIAL_POOL,
    }
    conn.close()


def test_H10_a_historical_read_without_pinned_evidence_is_unavailable(tmp_path):
    """No archive at the cutoff means unavailable, never today's value."""

    conn = connect_database(tmp_path / "fpl.db")
    _world(conn)
    certification = _certified_freeze(conn, values={DGW_PLAYER: 6.0})
    _capture(conn, player_id=DGW_PLAYER, fixture_id=FIXTURE_1, captured_at=CAPTURE_1)
    _capture(conn, player_id=DGW_PLAYER, fixture_id=FIXTURE_2, captured_at=CAPTURE_1, total_points=3)

    historical = ol.build_reality_ledger(
        conn, freeze_identity=certification.freeze_identity, as_of=AS_OF
    )
    # The fixtures ARE finished today, and the event IS final today; neither is
    # evidence about the cutoff, so the read reports the gap instead of a number.
    assert ol.event_finality(conn, EVENT) == planning.EVENT_STATE_FINAL
    assert historical.world_source == ol.WORLD_UNAVAILABLE
    assert historical.world_unavailable
    assert {row["exclusion_reason"] for row in historical.rows} == {ol.INPUT_EVIDENCE_UNAVAILABLE}
    assert all(row["official_outcome"] is None for row in historical.rows)
    assert historical.evaluated() == ()

    # The present-tense read is the one that may answer from the live tables.
    current = ol.build_reality_ledger(conn, freeze_identity=certification.freeze_identity)
    assert current.world_source == ol.WORLD_CURRENT
    assert current.evaluated()
    conn.close()


def test_H11_the_reads_finality_rule_is_the_canonical_rule(tmp_path):
    """The pinned resolution applies the accepted definition, not a second one."""

    conn = connect_database(tmp_path / "fpl.db")
    _world(conn)
    certification = _certified_freeze(conn, values={DGW_PLAYER: 6.0})
    provenance = ol.freeze_provenance(conn, certification.freeze_identity)
    assert provenance is not None
    events = (EVENT, EVENT_PARTIAL, OTHER_EVENT)
    world = ol.resolve_world_evidence(conn, provenance=provenance, as_of=None, events=events)
    for event in events:
        assert world.event_state(event) == planning.event_data_state(conn, event)[0]
    # ...and the same rule applied to pinned rows reports the pinned state.
    archive = _archive_world(
        tmp_path,
        observed_at=AS_OF,
        event_rows=[{"id": EVENT, "finished": 0, "data_checked": 0}],
    )
    pinned = ol.resolve_world_evidence(
        conn, provenance=provenance, as_of=AS_OF, archive_root=archive, events=[EVENT]
    )
    assert pinned.event_state(EVENT) == planning.EVENT_STATE_PROVISIONAL
    conn.close()



def test_T_pe1_causal_boundary_is_reused_and_not_weakened(tmp_path):
    """Theme 10: PE-5 reuses the frozen PE-1 boundary and changes none of it."""

    conn = connect_database(tmp_path / "fpl.db")
    _world(conn)
    # The boundary still fails closed with no cutoff...
    with pytest.raises(historical.HistoricalBoundaryError):
        historical.historical_player_fixtures(conn, as_of="")
    # ...and the PE-5 placeholder gate is literally that same definition.
    assert repo.row_is_scheduled_placeholder({"minutes": 0, "source": "element_summary"}) is True
    assert repo.row_is_scheduled_placeholder(
        {"minutes": 0, "starts": 0, "total_points": 0, "source": "element_summary"}
    ) is False
    # A PE-5 capture is invisible to the PE-1 reader: outcome evidence is not
    # player-fixture history, and appending it changes no causal read.
    before = historical.data_health(conn, as_of=AS_OF)
    _capture(conn, player_id=DGW_PLAYER, fixture_id=FIXTURE_1, captured_at=CAPTURE_1)
    after = historical.data_health(conn, as_of=AS_OF)
    assert before == after
    conn.close()


def test_T_corrections_create_explicit_superseding_observations(tmp_path):
    """Theme 4: a correction is a new observation naming what it supersedes."""

    conn = connect_database(tmp_path / "fpl.db")
    _world(conn)
    first = _capture(conn, player_id=DGW_PLAYER, fixture_id=FIXTURE_1, captured_at=CAPTURE_1,
                     total_points=6)
    first_id = conn.execute(
        "SELECT id FROM outcome_observation_captures WHERE capture_digest=?", (first.capture_digest,)
    ).fetchone()[0]
    corrected = _capture(
        conn,
        player_id=DGW_PLAYER,
        fixture_id=FIXTURE_1,
        captured_at=CAPTURE_2,
        total_points=8,
        bonus=2,
        bps=34,
        supersedes_capture_id=first_id,
        correction_reason="official points correction applied by the provider",
    )
    rows = ol.observation_captures(conn, event=EVENT)
    assert [row["total_points"] for row in rows] == [6, 8]
    assert rows[0]["supersedes_capture_id"] is None
    assert rows[1]["supersedes_capture_id"] == first_id
    assert rows[1]["correction_reason"].startswith("official points correction")
    # The superseded value is still there; supersession never erased it.
    assert rows[0]["total_points"] == 6
    ledger = ol.build_reality_ledger(
        conn,
        freeze_identity=ol.certify_prediction_freeze(
            conn,
            projection_run_ids=[
                _freeze(conn, family=FAMILY_MINUTES, version="minutes_v1.4.0",
                        kind=KIND_MINUTES, values={DGW_PLAYER: 6.0},
                        fixture_of={DGW_PLAYER: FIXTURE_1})
            ],
        ).freeze_identity,
        grain=ol.GRAIN_PLAYER_FIXTURE,
    )
    scored = [row for row in ledger.rows if row["player_id"] == DGW_PLAYER]
    assert len(scored) == 1
    assert scored[0]["official_outcome"]["total_points"] == 8
    assert scored[0]["exclusion_reason"] == ol.EVALUATED
    conn.close()


def test_T_join_identity_is_stable_and_name_independent(tmp_path):
    """Theme 5: the join is ids and grains, never display text."""

    conn = connect_database(tmp_path / "fpl.db")
    _world(conn)
    certification = _certified_freeze(conn, values={DGW_PLAYER: 6.0})
    _capture(conn, player_id=DGW_PLAYER, fixture_id=FIXTURE_1)
    _capture(conn, player_id=DGW_PLAYER, fixture_id=FIXTURE_2, total_points=3)
    before = ol.build_reality_ledger(conn, freeze_identity=certification.freeze_identity)
    conn.execute("UPDATE players SET web_name='RENAMED', full_name='A Different Person' WHERE id=?",
                 (DGW_PLAYER,))
    after = ol.build_reality_ledger(conn, freeze_identity=certification.freeze_identity)
    assert before.ledger_digest == after.ledger_digest
    assert after.rows[0]["destination_key"] == f"player_event:{EVENT}:{DGW_PLAYER}"
    conn.close()


def test_T_ledger_states_the_finalization_it_judged_under(tmp_path):
    """Theme 2: finalized and provisional are distinguishable ON the row itself.

    The exclusion reason says a row was refused; ``event_finality`` says what the
    official source had actually recorded, so a reviewer can tell "the event was
    not over yet" from "the event is over but the provider has not checked it".
    """

    conn = connect_database(tmp_path / "fpl.db")
    _world(conn)
    final = _certified_freeze(conn, values={DGW_PLAYER: 6.0})
    _capture(conn, player_id=DGW_PLAYER, fixture_id=FIXTURE_1)
    _capture(conn, player_id=DGW_PLAYER, fixture_id=FIXTURE_2, total_points=3)
    ledger = ol.build_reality_ledger(conn, freeze_identity=final.freeze_identity)
    row = next(entry for entry in ledger.rows if entry["player_id"] == DGW_PLAYER)
    assert row["event_finality"] == planning.EVENT_STATE_FINAL
    assert row["outcome_state"] == "OBSERVED"

    # The same player in an event whose fixtures are ALL over but which the
    # official source has not finalised is PROVISIONAL, and the row says so.
    conn.execute(
        "UPDATE fixtures SET started=1, finished=1 WHERE id=?", (FIXTURE_UNPLAYED,)
    )
    partial_run = _freeze(
        conn, family=FAMILY_POINTS, version=analytics.BASELINE_MODEL_VERSION,
        kind=KIND_POINTS, values={DGW_PLAYER: 6.0}, event=EVENT_PARTIAL,
    )
    partial = ol.certify_prediction_freeze(conn, projection_run_ids=[partial_run])
    _capture(conn, player_id=DGW_PLAYER, fixture_id=FIXTURE_PARTIAL, event=EVENT_PARTIAL)
    partial_ledger = ol.build_reality_ledger(conn, freeze_identity=partial.freeze_identity)
    partial_row = next(entry for entry in partial_ledger.rows if entry["player_id"] == DGW_PLAYER)
    # The reported state IS the canonical definition, not a second opinion about it.
    assert planning.event_data_state(conn, EVENT_PARTIAL)[0] == planning.EVENT_STATE_PROVISIONAL
    assert partial_row["event_finality"] == planning.EVENT_STATE_PROVISIONAL
    assert partial_row["outcome_state"] == "NOT_FINALISED"
    assert partial_row["evaluation_state"] == ol.EXCLUDED
    # Finalized and provisional are different content, not a display difference.
    assert partial_row["event_finality"] != row["event_finality"]
    conn.close()


def test_T_ledger_requires_recorded_freeze_provenance(tmp_path):
    """A consumer must name the freeze it used; an unknown one is refused."""

    conn = connect_database(tmp_path / "fpl.db")
    _world(conn)
    with pytest.raises(ol.OutcomeLedgerError, match="no recorded provenance"):
        ol.build_reality_ledger(conn, freeze_identity="sha256:nothing")
    conn.close()


def test_T_ledger_rows_name_the_source_they_were_read_from(tmp_path):
    """Theme 9: every row answers "the source / provenance", not only a digest."""

    conn = connect_database(tmp_path / "fpl.db")
    _world(conn)
    certification = _certified_freeze(conn, values={DGW_PLAYER: 6.0})
    _capture(
        conn,
        player_id=DGW_PLAYER,
        fixture_id=FIXTURE_1,
        captured_at=CAPTURE_1,
        source_name="official_element_summary",
        source_identity=f"element_summary:{DGW_PLAYER}:{EVENT}",
        source_payload_sha256="c" * 64,
    )
    _capture(
        conn,
        player_id=DGW_PLAYER,
        fixture_id=FIXTURE_2,
        captured_at=CAPTURE_1,
        total_points=3,
        source_name="official_element_summary",
        source_identity=f"element_summary:{DGW_PLAYER}:{EVENT}",
        source_payload_sha256="d" * 64,
    )
    ledger = ol.build_reality_ledger(conn, freeze_identity=certification.freeze_identity)
    row = next(entry for entry in ledger.rows if entry["player_id"] == DGW_PLAYER)
    sources = row["observation_sources"]
    # The headline total was built from TWO fixture captures, and the row names
    # both of them rather than only the first.
    assert len(sources) == 2
    assert {source["source_payload_sha256"] for source in sources} == {"c" * 64, "d" * 64}
    assert {source["source_name"] for source in sources} == {"official_element_summary"}
    assert set(row["observation_capture_digests"]) == {
        source["capture_digest"] for source in sources
    }
    # The order is by digest, not by discovery order, so it cannot drift.
    assert [source["capture_digest"] for source in sources] == sorted(
        source["capture_digest"] for source in sources
    )
    # Every named capture really is the retained source, with that identity.
    retained = {
        capture["capture_digest"]: capture
        for capture in ol.observation_captures(conn, event=EVENT, player_id=DGW_PLAYER)
    }
    for source in sources:
        assert retained[source["capture_digest"]]["source_identity"] == (
            f"element_summary:{DGW_PLAYER}:{EVENT}"
        )
    conn.close()


def test_T_observations_from_different_sources_are_separately_attributable(tmp_path):
    """Theme 9 + 3: the same fact read from TWO sources is two attributable rows."""

    conn = connect_database(tmp_path / "fpl.db")
    _world(conn)
    first = _capture(
        conn,
        player_id=DGW_PLAYER,
        fixture_id=FIXTURE_1,
        captured_at=CAPTURE_1,
        source_name="official_element_summary",
        source_identity="element_summary:batch-1",
        source_payload_sha256="e" * 64,
    )
    second = _capture(
        conn,
        player_id=DGW_PLAYER,
        fixture_id=FIXTURE_1,
        captured_at=CAPTURE_1,
        source_name="cached_bootstrap_fixtures",
        source_identity="bootstrap:batch-9",
        source_payload_sha256="f" * 64,
    )
    # Same key, same time, same values -- but a different source is a DIFFERENT
    # observation, not an idempotent repeat of the first.
    assert first.inserted and second.inserted
    assert first.capture_digest != second.capture_digest
    rows = ol.observation_captures(conn, event=EVENT, player_id=DGW_PLAYER)
    assert sorted(row["source_name"] for row in rows) == [
        "cached_bootstrap_fixtures",
        "official_element_summary",
    ]
    certification = _certified_freeze(conn, values={DGW_PLAYER: 6.0, SGW_A_PLAYER: 3.0})
    ledger = ol.build_reality_ledger(
        conn, freeze_identity=certification.freeze_identity, grain=ol.GRAIN_PLAYER_FIXTURE
    )
    scored = [row for row in ledger.rows if row["player_id"] == DGW_PLAYER]
    assert len(scored) == 1
    # The scored row names the ONE source it reports, and the other retained
    # observation is still there to be inspected.
    assert len(scored[0]["observation_sources"]) == 1
    named = scored[0]["observation_sources"][0]
    assert named["capture_digest"] in {row["capture_digest"] for row in rows}
    conn.close()


def test_T_capture_completed_rows_reuses_the_canonical_completed_boundary(tmp_path):
    """Theme 3: the ingest path reads the SAME completed rows every reader trusts.

    A finished fixture whose rows were never refreshed holds placeholders, and
    the canonical completed boundary already rejects them; the capture path must
    inherit that decision rather than restate it.
    """

    conn = connect_database(tmp_path / "fpl.db")
    _world(conn)
    repo.upsert_player_gameweeks(
        conn,
        [
            PlayerGameweekRecord(
                player_id=DGW_PLAYER, event=EVENT, fixture_id=FIXTURE_1, minutes=90, starts=1,
                total_points=6, goals_scored=1, bonus=1, bps=30, source="element_summary", raw_json={},
            ),
            # The placeholder signature: minutes 0, nothing else stated.
            PlayerGameweekRecord(
                player_id=SGW_A_PLAYER, event=EVENT, fixture_id=FIXTURE_1, minutes=0,
                source="element_summary", raw_json={},
            ),
            # A GENUINE did-not-play is all explicit zeros and IS captured.
            PlayerGameweekRecord(
                player_id=SGW_B_PLAYER, event=EVENT, fixture_id=FIXTURE_1, minutes=0, starts=0,
                total_points=0, goals_scored=0, assists=0, clean_sheets=0, goals_conceded=2,
                saves=0, bonus=0, bps=4, yellow_cards=0, red_cards=0, penalties_saved=0,
                penalties_missed=0, own_goals=0, defensive_contribution=0,
                source="element_summary", raw_json={},
            ),
        ],
        _t(5.5),
    )
    captured = ol.capture_completed_player_fixtures(conn, EVENT, captured_at=CAPTURE_1)
    assert sorted(result.player_id for result in captured) == [DGW_PLAYER, SGW_B_PLAYER]
    rows = {(row["player_id"], row["fixture_id"]): row for row in ol.observation_captures(conn, event=EVENT)}
    assert set(rows) == {(DGW_PLAYER, FIXTURE_1), (SGW_B_PLAYER, FIXTURE_1)}
    assert rows[(DGW_PLAYER, FIXTURE_1)]["observation_state"] == ol.OBSERVATION_FINAL
    assert rows[(DGW_PLAYER, FIXTURE_1)]["team_id"] == TEAM_A
    assert rows[(DGW_PLAYER, FIXTURE_1)]["opponent_team_id"] == TEAM_B
    assert rows[(DGW_PLAYER, FIXTURE_1)]["event_time"] == KICKOFF_1
    # The genuine DNP is a retained zero-minute observation; the placeholder has
    # no observation at all.
    assert rows[(SGW_B_PLAYER, FIXTURE_1)]["minutes"] == 0
    assert rows[(SGW_B_PLAYER, FIXTURE_1)]["bps"] == 4
    assert (SGW_A_PLAYER, FIXTURE_1) not in rows
    conn.close()


def test_T_a_run_resolves_to_its_recorded_freeze(tmp_path):
    """Every prediction-side record can name the exact freeze it came from."""

    conn = connect_database(tmp_path / "fpl.db")
    _world(conn)
    certification = _certified_freeze(conn, values={DGW_PLAYER: 6.0})
    for run_id in certification.projection_run_ids:
        record = ol.freeze_for_run(conn, run_id)
        assert record is not None
        assert record["freeze_identity"] == certification.freeze_identity
        assert record["bootstrap_element_ids_sha256"] == certification.bootstrap_element_ids_sha256
    assert ol.freeze_for_run(conn, 999999) is None
    conn.close()


def test_T_outcome_evidence_never_leaks_into_a_causal_read(tmp_path):
    """Theme 10 / contract theme 1: later outcomes are evaluation evidence ONLY.

    Appending a finalized outcome must leave every predictive read that claims an
    earlier cutoff exactly as it was -- including the PE-1 boundary, which is
    unchanged and unreplaced.
    """

    conn = connect_database(tmp_path / "fpl.db")
    _world(conn)
    repo.upsert_player_gameweeks(
        conn,
        [
            PlayerGameweekRecord(
                player_id=SGW_B_PLAYER, event=OTHER_EVENT, fixture_id=FIXTURE_OTHER,
                minutes=90, starts=1, total_points=6, source="element_summary", raw_json={},
            )
        ],
        _t(12),
    )
    read_as_of = _t(11)
    before = historical.historical_player_fixtures(
        conn, as_of=read_as_of, planning_event=EVENT
    )
    assert [row["fixture_id"] for row in before] == [FIXTURE_OTHER]
    _capture_all(conn)
    _capture(conn, player_id=SGW_A_PLAYER, fixture_id=FIXTURE_1, captured_at=CAPTURE_2)
    after = historical.historical_player_fixtures(conn, as_of=read_as_of, planning_event=EVENT)
    assert after == before
    # The outcome is only reachable through the outcome boundary.
    assert len(ol.observation_captures(conn, events=[EVENT], as_of=read_as_of)) == 0
    assert len(ol.observation_captures(conn, events=[EVENT])) > 0
    conn.close()


def test_T_schema_migration_is_additive_and_leaves_existing_tables_alone(tmp_path):
    """Theme 10: m016 adds tables and triggers; it moves and rewrites nothing."""

    conn = connect_database(tmp_path / "fpl.db")
    _world(conn)
    from fpl_brain.database import SCHEMA_VERSION

    version = conn.execute("SELECT value FROM schema_meta WHERE key='schema_version'").fetchone()[0]
    assert int(version) == SCHEMA_VERSION >= 16
    names = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    assert {
        "outcome_observation_captures",
        "prediction_freeze_provenance",
        "prediction_freeze_runs",
        # the pre-existing outcome/provenance storage PE-5 reuses rather than
        # duplicating: it is still here, unchanged.
        "outcome_observations",
        "bootstrap_generations",
        "projection_runs",
        "player_gameweeks",
    } <= names
    # Re-running every migration is a no-op, so an existing live store upgrades
    # without a rebuild.
    from fpl_brain.database import initialize_database

    initialize_database(conn)
    assert conn.execute("SELECT COUNT(*) FROM player_gameweeks").fetchone()[0] == 0
    conn.close()
