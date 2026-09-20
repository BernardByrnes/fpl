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
    analytics.finish_projection_run(conn, run_id, "complete")
    return run_id


def _generation(
    conn,
    *,
    fetch_run_id: int | None,
    captured_at: str = GENERATION_CAPTURED,
    accepted: bool = True,
    element_ids=(101, 102, 103),
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
    supersedes_capture_id: int | None = None,
    correction_reason: str | None = None,
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
        supersedes_capture_id=supersedes_capture_id,
        correction_reason=correction_reason,
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
    newer = _generation(conn, fetch_run_id=_fetch_run(conn), captured_at=_t(3), element_ids=(101, 102, 103, 104))
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
    assert retained == [101, 102, 103]
    assert retained == sorted(retained)
    assert element_id_sha256(retained) == recorded["bootstrap_element_ids_sha256"]

    # A second, independent generation over the SAME id set digests identically,
    # and the id set is order-insensitive by construction.
    same = _generation(conn, fetch_run_id=_fetch_run(conn), captured_at=_t(10), element_ids=(103, 101, 102))
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
    """Theme 8: historical point-in-time history needs a proven archived source.

    Absent that proof the honest answer is "unavailable", never a number
    manufactured from today's current state.
    """

    conn = connect_database(tmp_path / "fpl.db")
    _world(conn)
    with pytest.raises(ol.BackfillEvidenceError):
        _capture(conn, player_id=DGW_PLAYER, fixture_id=FIXTURE_1, captured_at=_t(6), backfill=True)
    assert conn.execute("SELECT COUNT(*) FROM outcome_observation_captures").fetchone()[0] == 0
    # With the archive proof present the same capture is admitted...
    result = _capture(
        conn,
        player_id=DGW_PLAYER,
        fixture_id=FIXTURE_1,
        captured_at=_t(6),
        source_name="archived_player_gameweeks",
        source_identity="fetch:7:player_gameweeks",
        source_payload_sha256="a" * 64,
        archive_capture_id="capture-7",
        backfill=True,
    )
    assert result.inserted
    # ...while a capture claiming no observable time at all is still refused.
    with pytest.raises(ol.BackfillEvidenceError):
        _capture(
            conn,
            player_id=DGW_PLAYER,
            fixture_id=FIXTURE_2,
            captured_at=None,
            source_name="archived_player_gameweeks",
            source_identity="fetch:7:player_gameweeks",
            source_payload_sha256="b" * 64,
            archive_capture_id="capture-8",
            backfill=True,
        )
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
