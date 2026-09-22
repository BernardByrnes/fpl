"""PE-6 availability / minutes refinement — hard tests.

The contract's required hard cases are covered here in order, plus the
challenger identity, the evaluation contract and the PE-2 sample-size policy.
The incumbent is never mutated: every assertion about the frozen model reads it
through its own entry points, and the challenger is checked *against* it rather
than in place of it.
"""

from __future__ import annotations

import inspect
import json

import pytest

from fpl_brain import analytics
from fpl_brain import availability_minutes_challenger as ch
from fpl_brain import availability_minutes_evaluation as ev
from fpl_brain import bonus_allocation
from fpl_brain import bonus_challenger
from fpl_brain import bps_rules
from fpl_brain import historical_observations as historical
from fpl_brain import minutes_coherence
from fpl_brain import minutes_model as incumbent
from fpl_brain import monte_carlo
from fpl_brain import outcome_ledger
from fpl_brain import repositories as repo
from fpl_brain import team_model
from fpl_brain import walk_forward as wf
from fpl_brain import xpts
from fpl_brain.database import connect_database
from fpl_brain.models import (
    EventRecord,
    FixtureRecord,
    PlayerGameweekRecord,
    PlayerRecord,
    PlayerSnapshotRecord,
    PositionRecord,
    TeamRecord,
)

#: The planning cutoff for event 4 (the event being predicted).
CUTOFF = "2026-09-10T12:00:00Z"
#: When the seeded historical rows were OBSERVED.  Observation time is part of
#: the PE-1 boundary, so it is stated explicitly rather than left to the ambient
#: clock (which would place the write after the cutoff it is evidence for).
OBSERVED_AT = {
    1: "2026-08-23T08:00:00Z",
    2: "2026-08-31T08:00:00Z",
    3: "2026-09-06T08:00:00Z",
}
EVENT_DEADLINES = {
    1: "2026-08-21T17:30:00Z",
    2: "2026-08-28T17:30:00Z",
    3: "2026-09-04T17:30:00Z",
    4: "2026-09-12T12:30:00Z",
}
EVENT_KICKOFFS = {
    1: "2026-08-22T14:00:00Z",
    2: "2026-08-29T14:00:00Z",
    3: "2026-09-05T14:00:00Z",
    4: "2026-09-13T14:00:00Z",
}

#: Fixtures: (fixture_id, event, home, away).  Event 2 is a double gameweek for
#: team 2; event 3 is blank for team 2; event 4 is blank for team 3.
FIXTURES = (
    (1, 1, 1, 2),
    (2, 2, 1, 2),
    (3, 2, 2, 3),
    (4, 3, 1, 3),
    (5, 4, 1, 2),
)
FIXTURE_BY_EVENT_TEAM = {
    (1, 1): 1,
    (1, 2): 1,
    (2, 1): 2,
    (2, 2): 2,
    (2, 3): 3,
    (3, 1): 4,
    (3, 3): 4,
    (4, 1): 5,
    (4, 2): 5,
}

#: team -> position -> player ids.  Teams 1 and 2 each carry a side large enough
#: for the frozen team-coherence layer to solve (a full XI plus a bench), so team
#: coherence can be checked on top of the challenger's own rows.
SQUADS = {
    1: {1: [9], 2: [10, 13, 14, 18, 20, 21, 24], 3: [11, 12, 15, 17, 22, 23], 4: [16, 19]},
    2: {1: [29], 2: [30, 31, 32, 38, 40, 41], 3: [33, 34, 35, 39, 42, 43], 4: [36, 37]},
    3: {1: [48], 2: [49], 3: [50, 51], 4: [52]},
}

#: Minutes and starts per player per event.  ``None`` means "no row": a blank
#: gameweek holds no row at all, exactly as the official data behaves.
PGW: dict[int, dict[int, tuple[int, int]]] = {
    # --- team 1 -------------------------------------------------------------
    9: {1: (90, 1), 2: (90, 1), 3: (90, 1)},
    10: {1: (90, 1), 2: (90, 1), 3: (90, 1)},          # regular starter
    11: {1: (90, 1), 2: (45, 1), 3: (90, 1)},          # reduced minutes
    12: {1: (0, 0), 2: (0, 0), 3: (0, 0)},             # bench regular, no minutes
    13: {1: (90, 1), 2: (0, 0), 3: (90, 1)},           # injured, then back at full minutes
    14: {1: (90, 1), 2: (0, 0), 3: (0, 0)},            # hard unavailable at the cutoff
    15: {1: (60, 1), 2: (0, 0), 3: (90, 1)},           # rotation-risk note
    16: {1: (75, 0), 2: (20, 0), 3: (0, 0)},           # cameo record, incl. a 60+ cameo
    17: {1: (90, 1), 2: (90, 1), 3: (90, 1)},
    18: {1: (0, 0), 2: (0, 0), 3: (90, 1)},            # two available zero-minute non-starts
    19: {1: (30, 0), 2: (90, 1), 3: (90, 1)},
    20: {1: (90, 1), 2: (90, 1), 3: (90, 1)},
    21: {1: (90, 1), 2: (70, 1), 3: (0, 0)},
    22: {1: (20, 0), 2: (90, 1), 3: (90, 1)},
    23: {2: (90, 1), 3: (90, 1)},                      # no event-1 row at all
    # --- team 2 -------------------------------------------------------------
    29: {1: (90, 1), 2: (90, 1), 3: (0, 0)},
    30: {1: (90, 1), 2: (90, 1), 3: (45, 0)},
    31: {1: (90, 1), 2: (60, 1), 3: (0, 0)},
    32: {1: (0, 0), 2: (0, 0), 3: (90, 1)},
    33: {1: (90, 1), 2: (90, 1), 3: (12, 0)},
    34: {1: (0, 0), 2: (0, 0), 3: (0, 0)},
    35: {1: (90, 1), 2: (90, 1), 3: (70, 1)},
    36: {1: (25, 0), 2: (80, 0), 3: (0, 0)},
    37: {1: (90, 1), 2: (90, 1), 3: (90, 1)},
    38: {1: (0, 0), 2: (0, 0), 3: (55, 1)},
    39: {1: (90, 1), 2: (30, 0), 3: (90, 1)},
    40: {1: (90, 1), 2: (90, 1), 3: (90, 1)},
    41: {1: (90, 1), 2: (90, 1), 3: (0, 0)},
    42: {1: (0, 0), 2: (15, 0), 3: (90, 1)},
    43: {1: (90, 1), 2: (90, 1), 3: (90, 1)},
    # --- team 3 (blank in event 4) ------------------------------------------
    48: {2: (90, 1), 3: (90, 1)},
    49: {2: (0, 0), 3: (0, 0)},
    50: {2: (45, 1), 3: (90, 1)},
    51: {2: (0, 0), 3: (0, 0)},
    52: {2: (0, 0), 3: (20, 0)},
}

#: player -> [(event_context, status, chance)] official trail, oldest first.
STATUS_TRAILS: dict[int, list[tuple[int, str, int | None]]] = {
    12: [(1, "a", None), (2, "a", None), (3, "a", None)],
    13: [(1, "a", None), (2, "i", None), (3, "i", None)],
    14: [(3, "s", 0)],
    18: [(1, "a", None), (2, "a", None), (3, "a", None)],
}
#: player -> the status/chance snapshot in force at the cutoff.
CURRENT_STATUS: dict[int, tuple[str, int | None]] = {
    12: ("a", None),
    13: ("i", None),
    14: ("s", 0),
    18: ("a", None),
    15: ("a", None),
    16: ("a", None),
}
#: player -> (scouting key, value text) note in force at the cutoff.
SCOUT_NOTES: dict[int, tuple[str, str]] = {15: ("rotation_risk", "high")}

#: The capture of the official element list that every player is in.  It sits
#: before event 1's deadline, so the accepted bootstrap generation recorded from
#: it is the official pool generation every cutoff in this world can see.
IDENTITY_CAPTURE = "2026-08-20T08:00:00Z"

#: Every player the seeded world knows, sorted.
ALL_PLAYERS = sorted(
    player for positions in SQUADS.values() for players in positions.values() for player in players
)


def _position_of(player_id):
    return next(
        position
        for positions in SQUADS.values()
        for position, players in positions.items()
        if player_id in players
    )


def _official_element(player_id, *, team=None, position=None):
    """The official bootstrap element a capture was parsed from.

    The real payload carries the club (``team``) and the position
    (``element_type``), which is what makes the generation's own snapshot rows
    the cutoff-safe record of a player's identity.
    """

    return {
        "id": player_id,
        "team": _team_of(player_id) if team is None else team,
        "element_type": _position_of(player_id) if position is None else position,
    }


def _seed_generation(
    conn,
    captured_at,
    element_ids,
    *,
    accepted=True,
    missing=(),
    payloads=None,
    run=None,
):
    """One bootstrap generation with its OWN snapshot rows.

    A generation is resolvable only if its rows carry the official payload, so
    they are written from ONE fetch run and the generation records that run --
    exactly the shape the ingest path produces.  ``missing`` leaves an element id
    in the recorded pool with no snapshot row at all, and ``payloads`` overrides
    a row's payload (a partial capture, for instance).
    """

    ids = sorted({int(pid) for pid in element_ids})
    omitted = {int(value) for value in missing}
    record_run = repo.create_fetch_run(conn, "fetch_fpl") if run is None else int(run)
    records = [
        PlayerSnapshotRecord(
            player_id=player_id,
            captured_at=captured_at,
            now_cost=50,
            raw_json=(
                _official_element(player_id)
                if (payloads or {}).get(player_id) is None
                else (payloads or {})[player_id]
            ),
        )
        for player_id in ids
        if player_id not in omitted
    ]
    with conn:
        repo.insert_snapshots(conn, records, record_run)
        generation_id = repo.record_bootstrap_generation(
            conn,
            captured_at=captured_at,
            accepted=bool(accepted),
            official_element_count=len(ids),
            parsed_count=len(ids),
            persisted_count=len(ids),
            element_ids=ids,
            element_ids_sha256=repo.element_ids_identity(ids)[1],
            acceptance_rule="pe6-test",
            acceptance_rule_version="BOOTSTRAP_GENERATION_ACCEPTANCE v1",
            fetch_run_id=record_run,
        )
    return generation_id, record_run


def _seed(
    conn,
    *,
    pgw_overrides=None,
    drop_events=(),
    extra_snapshots=(),
    placeholder_rows=(),
    generation_accepted=True,
    generation_missing_players=(),
    generation_payloads=None,
):
    """A two-sided world with official status trails, scouting notes and outcomes."""

    minutes_table = {player: dict(rows) for player, rows in PGW.items()}
    for player, rows in (pgw_overrides or {}).items():
        minutes_table[player] = dict(rows)
    with conn:
        repo.upsert_teams(
            conn,
            [TeamRecord(id=1, name="One"), TeamRecord(id=2, name="Two"), TeamRecord(id=3, name="Three")],
        )
        repo.upsert_positions(
            conn,
            [
                PositionRecord(id=position, singular_name_short=name)
                for position, name in ((1, "GKP"), (2, "DEF"), (3, "MID"), (4, "FWD"))
            ],
        )
        repo.upsert_players(
            conn,
            [
                PlayerRecord(
                    id=player_id,
                    web_name=f"P{player_id}",
                    full_name=f"Player {player_id}",
                    team_id=team,
                    element_type=position,
                )
                for team, positions in SQUADS.items()
                for position, players in positions.items()
                for player_id in players
            ],
        )
        repo.upsert_events(
            conn,
            [
                EventRecord(
                    id=event,
                    finished=1 if event <= 3 else 0,
                    data_checked=1 if event <= 3 else 0,
                    deadline_time=EVENT_DEADLINES[event],
                    raw_json={},
                )
                for event in (1, 2, 3, 4)
            ],
        )
        repo.upsert_fixtures(
            conn,
            [
                FixtureRecord(
                    id=fixture_id,
                    event=event,
                    team_h=home,
                    team_a=away,
                    finished=1 if event <= 3 else 0,
                    started=1 if event <= 3 else 0,
                    kickoff_time=EVENT_KICKOFFS[event],
                    raw_json={},
                )
                for fixture_id, event, home, away in FIXTURES
            ],
        )

    player_teams = {
        player: team
        for team, positions in SQUADS.items()
        for players in positions.values()
        for player in players
    }
    # Every fixture of the event the player's club plays, so a double gameweek
    # player really carries two rows (the grain the official data has).
    fixtures_of: dict[tuple[int, int], list[int]] = {}
    for fixture_id, event, home, away in FIXTURES:
        fixtures_of.setdefault((event, home), []).append(fixture_id)
        fixtures_of.setdefault((event, away), []).append(fixture_id)
    rows_by_event: dict[int, list[PlayerGameweekRecord]] = {}
    for player_id, per_event in sorted(minutes_table.items()):
        team = player_teams[player_id]
        for event, (minutes, starts) in sorted(per_event.items()):
            if event in drop_events:
                continue
            for fixture_id in fixtures_of.get((event, team), []):
                rows_by_event.setdefault(event, []).append(
                    PlayerGameweekRecord(
                        player_id=player_id,
                        event=event,
                        fixture_id=fixture_id,
                        minutes=minutes,
                        starts=starts,
                        total_points=6 if starts and minutes >= 60 else (2 if minutes > 0 else 0),
                        source="element_summary",
                        raw_json={},
                    )
                )
    with conn:
        for event, rows in sorted(rows_by_event.items()):
            repo.upsert_player_gameweeks(conn, rows, OBSERVED_AT[event])
        # A scheduled placeholder: minutes 0 with every other performance column
        # NULL.  It is a schedule artefact, never a did-not-play.
        for player_id, event, fixture_id in placeholder_rows:
            repo.upsert_player_gameweeks(
                conn,
                [
                    PlayerGameweekRecord(
                        player_id=player_id,
                        event=event,
                        fixture_id=fixture_id,
                        minutes=0,
                        source="element_summary",
                        raw_json={},
                    )
                ],
                OBSERVED_AT[event],
            )

    # One fetch run per (player, status snapshot): the store's uniqueness is
    # (player_id, fetch_run_id), so a trail lives in successive fetches.
    def insert_snapshot(player_id, captured_at, status, chance, event_context=None, element=None):
        run = repo.create_fetch_run(conn, "fetch_fpl")
        repo.insert_snapshots(
            conn,
            [
                PlayerSnapshotRecord(
                    player_id=player_id,
                    captured_at=captured_at,
                    event_context=event_context,
                    now_cost=50,
                    status=status,
                    chance_of_playing_this_round=chance,
                    raw_json=_official_element(player_id) if element is None else element,
                )
            ],
            run,
        )

    with conn:
        for player_id, trail in sorted(STATUS_TRAILS.items()):
            for event_context, status, chance in trail:
                insert_snapshot(
                    player_id,
                    f"2026-08-{20 + event_context:02d}T08:00:00Z",
                    status,
                    chance,
                    event_context,
                )
        for player_id, (status, chance) in sorted(CURRENT_STATUS.items()):
            insert_snapshot(player_id, "2026-09-10T08:00:00Z", status, chance)
        for player_id, event, status, chance, captured_at in extra_snapshots:
            insert_snapshot(player_id, captured_at, status, chance, event)
        # The official pool EVERY cutoff in this world resolves from: one ACCEPTED
        # bootstrap generation, one fetch run, one snapshot row per element whose
        # payload carries the club and the position.  A player with no status
        # capture of his own has this row as his freshest capture, which states no
        # status (None) and is therefore read by the incumbent exactly as it read
        # "no snapshot at all": p_available 1.0 and an unattributable 0-minute row.
        _seed_generation(
            conn,
            IDENTITY_CAPTURE,
            ALL_PLAYERS,
            accepted=generation_accepted,
            missing=generation_missing_players,
            payloads=generation_payloads,
        )
        for player_id, (key, value) in sorted(SCOUT_NOTES.items()):
            import_id = repo.insert_scouting_import(
                conn,
                {
                    "source_file": "world",
                    "file_sha256": f"world-{player_id}",
                    "players_total": 1,
                    "players_resolved": 1,
                    "notes_inserted": 1,
                },
            )
            repo.insert_scouting_note(
                conn,
                {
                    "import_id": import_id,
                    "player_id": player_id,
                    "key": key,
                    "value_text": value,
                    "confidence": "medium",
                    "observed_at": "2026-09-09T09:00:00Z",
                    "expires_at": "2026-09-30T00:00:00Z",
                },
            )
    return conn


def _db(tmp_path, name="fpl.db", **kwargs):
    return _seed(connect_database(tmp_path / name), **kwargs)


def _strip_value(value):
    if isinstance(value, dict):
        return {key: _strip_value(item) for key, item in value.items() if key != "generated_at"}
    if isinstance(value, list):
        return [_strip_value(item) for item in value]
    return value


def _strip(rows):
    """Rows with the ambient clock removed, nested provenance included."""

    return [_strip_value(row) for row in rows]


def _projection(row):
    """One row with the identity PROVENANCE removed.

    ``player_identity`` records where the identity was resolved from and how far
    the persisted row has diverged from it, so it is the one field that must
    change when the current row later disagrees with the cutoff.  Everything else
    is the projection, and must not move.
    """

    return _strip_value({key: value for key, value in row.items() if key != "player_identity"})


#: The ONE artifact section that is deliberately store-dependent: it reports how
#: far the persisted pool has moved from the cutoff's official pool.  Everything
#: else must be byte-identical when the persisted row is mutated.
_STORE_DIVERGENCE_SECTION = "store_divergence"


def _scoring_artifact(artifact):
    """The artifact with that one declared store-divergence section removed."""

    return {
        key: value for key, value in artifact.items() if key != _STORE_DIVERGENCE_SECTION
    }


def _fixture_for(event, team):
    return FIXTURE_BY_EVENT_TEAM[(event, team)]


def _row(arms, key, arm=ch.ARM_ALL_REFINEMENTS):
    return arms.arms[arm][key]


def _team_of(player_id):
    return next(
        team
        for team, positions in SQUADS.items()
        if any(player_id in players for players in positions.values())
    )


# ---------------------------------------------------------------------------
# Frozen incumbent identity and the challenger's own identity.
# ---------------------------------------------------------------------------


def test_frozen_incumbent_identity_is_untouched():
    identity = ch.frozen_incumbent_identity()
    assert identity["unchanged"] is True, identity["changed"]
    assert identity["observed"] == ch.FROZEN_INCUMBENT_VERSIONS
    assert incumbent.MINUTES_MODEL_VERSION == "minutes_v1.8.0"
    assert incumbent.MINUTES_COHERENT_MODEL_VERSION == "minutes_v1.2.0"
    from fpl_brain import joint_minutes

    assert joint_minutes.JOINT_MINUTES_MODEL_VERSION == "minutes_v1.5.2"


def test_challenger_identity_is_its_own_and_hashed():
    base = ch.AvailabilityMinutesChallengerConfig()
    other = ch.AvailabilityMinutesChallengerConfig(status_evidence_prior_strength=5.0)
    assert base.config_hash() == ch.AvailabilityMinutesChallengerConfig().config_hash()
    assert base.config_hash() != other.config_hash()
    assert base.config_hash() != incumbent.MinutesModelConfig().config_hash()
    identity = ch.challenger_identity(base)
    assert identity["challenger_version"] == ch.AVAILABILITY_MINUTES_CHALLENGER_VERSION
    assert identity["family"] == ch.AVAILABILITY_MINUTES_CHALLENGER_FAMILY
    assert identity["promotion"] == "NOT_PERFORMED_CHALLENGER_ONLY"
    assert identity["assumption_family_treatment"] == ch.ASSUMPTION_FAMILY_TREATMENT
    assert set(ch.ASSUMPTION_FAMILY_TREATMENT) == {
        "official_status_availability_defaults",
        "bounded_scouting_modifiers",
        "bounded_return_from_injury_modifiers",
        "bounded_rotation_risk_modifiers",
    }
    assert ch.ASSUMPTION_FAMILY_TREATMENT["bounded_scouting_modifiers"] == "UNCHANGED_BY_CHALLENGER"
    assert identity["resolved_parameters"]["cameo_recency_half_life_matches"] == pytest.approx(
        incumbent.MinutesModelConfig().start_recency_half_life_matches
    )


# ---------------------------------------------------------------------------
# Required hard tests 1-2: hard unavailability; bounded probabilities.
# ---------------------------------------------------------------------------


def test_hard_unavailable_status_stays_unavailable(tmp_path):
    """Contract case 1: a hard status is never refined into availability."""

    # The trail claims the player carried the hard status across two events, and
    # his own rows show 90 minutes in one of them: the strongest possible
    # temptation to "refine" the status.  It must not move.
    conn = _db(
        tmp_path,
        extra_snapshots=[
            (14, 1, "s", 0, "2026-08-21T08:00:00Z"),
            (14, 2, "s", 0, "2026-08-28T08:00:00Z"),
        ],
    )
    arms = ch.build_challenger_arms(conn, 4, CUTOFF)
    key = (14, _fixture_for(4, 1))
    row = _row(arms, key)
    assert row["availability_basis"] == ch.BASIS_HARD_STATUS
    assert row["p_available"] == 0.0
    assert (row["p_start"], row["p_cameo"], row["p_60_plus"], row["p_80_plus"]) == (0.0, 0.0, 0.0, 0.0)
    assert row["expected_minutes"] == 0.0 and row["p_zero"] == 1.0
    block = row["refinements"][ch.FAMILY_AVAILABILITY_STATUS_EVIDENCE]
    assert block["status"] == ch.REFINEMENT_NOT_APPLICABLE
    assert block["basis"] == ch.BASIS_HARD_STATUS
    conn.close()


def test_available_player_probability_remains_bounded(tmp_path):
    """Contract case 2: every published probability stays inside [0, 1]."""

    conn = _db(tmp_path)
    arms = ch.build_challenger_arms(conn, 4, CUTOFF)
    assert arms.keys()
    probabilities = (
        "p_available",
        "p_start_given_available",
        "p_cameo_given_not_start",
        "p_60_given_start",
        "p_80_given_start",
        "p_60_given_cameo",
        "p_start",
        "p_cameo",
        "p_zero",
        "p_1_59",
        "p_60_plus",
        "p_80_plus",
    )
    for arm in arms.arm_names():
        for key, row in arms.arms[arm].items():
            for name in probabilities:
                assert 0.0 <= row[name] <= 1.0, (arm, key, name, row[name])
            assert 0.0 <= row["expected_minutes"] <= 90.0
    conn.close()


# ---------------------------------------------------------------------------
# Required hard tests 3-6: causality, genuine DNP, placeholders.
# ---------------------------------------------------------------------------


def test_injury_doubt_evidence_is_cutoff_safe(tmp_path):
    """Contract case 3: injury/doubt evidence comes from pre-cutoff rows only."""

    conn = _db(tmp_path)
    arms = ch.build_challenger_arms(conn, 4, CUTOFF)
    key = (13, _fixture_for(4, 1))
    evidence = arms.evidence[key]
    assert evidence.availability_basis == ch.BASIS_DECLARED_STATUS_DEFAULT
    assert evidence.status_evidence_observations >= 2
    rows = analytics.completed_rows_as_of(conn, 13, CUTOFF, 4)
    assert rows, "the causal window must hold the seeded rows"
    for row in rows:
        assert int(row["event"]) < 4
        assert str(row["fixture_kickoff"]) <= CUTOFF
        assert str(row["updated_at"]) <= CUTOFF
    assert evidence.row_attribution_bases.get(ch.EVENT_SPECIFIC_ATTRIBUTION) == len(rows)
    conn.close()


def test_post_cutoff_status_update_is_ignored(tmp_path):
    """Contract case 4: a status update after the cutoff changes nothing."""

    conn = _db(tmp_path)
    arms = ch.build_challenger_arms(conn, 4, CUTOFF)
    before = {key: _strip([_row(arms, key)])[0] for key in sorted(arms.keys())}
    with conn:
        run = repo.create_fetch_run(conn, "fetch_fpl")
        repo.insert_snapshots(
            conn,
            [
                PlayerSnapshotRecord(
                    player_id=13,
                    captured_at="2026-09-11T08:00:00Z",
                    event_context=4,
                    now_cost=50,
                    status="a",
                    raw_json={},
                )
            ],
            run,
        )
    after = ch.build_challenger_arms(conn, 4, CUTOFF)
    assert sorted(after.keys()) == sorted(before)
    for key in before:
        assert _strip([_row(after, key)])[0] == before[key], key
    conn.close()


# ---------------------------------------------------------------------------
# Cutoff-resolved player identity (review pass).
#
# ``players.team_id`` / ``element_type`` / ``is_active`` are CURRENT state: the
# store keeps one row per player and overwrites it.  A later transfer, position
# change or retirement must not be able to rewrite what an earlier cutoff
# projected, so PE-6 resolves the identity from the latest ACCEPTED official
# bootstrap generation at or before the cutoff: its element id set is the
# official pool, and its OWN snapshot rows carry the club and the position.
# ---------------------------------------------------------------------------


def _resolved_identity(arms):
    return {
        int(player["player_id"]): (int(player["team_id"]), int(player["element_type"]))
        for player in arms.players
    }


def _resolved_identity_rows(resolution):
    return {
        int(player["player_id"]): (int(player["team_id"]), int(player["element_type"]))
        for player in resolution["players"]
    }


def _resolved_identity_of(resolution, player_id):
    return next(
        player for player in resolution["players"] if int(player["player_id"]) == player_id
    )["identity"]


def _player(arms, player_id):
    return next(player for player in arms.players if int(player["player_id"]) == player_id)


def test_the_official_pool_is_the_accepted_generation_at_the_cutoff(tmp_path):
    """Membership, club and position all come from the generation's own rows."""

    conn = _db(tmp_path)
    generation = ch.official_pool_generation_at_cutoff(conn, CUTOFF)
    assert generation["usable"] is True, generation["reasons"]
    assert generation["generation"]["accepted"] is True
    assert generation["generation"]["captured_at"] == IDENTITY_CAPTURE
    assert generation["generation"]["element_ids_sha256_matches_the_id_set"] is True
    assert generation["element_ids"] == ALL_PLAYERS
    assert generation["newest_attempt_is_the_accepted_generation"] is True
    # The rows it resolves from are its OWN fetch run's rows, one per element.
    rows = ch.generation_snapshot_rows(conn, generation["generation"], CUTOFF)
    assert sorted(rows) == ALL_PLAYERS
    assert {str(row["captured_at"]) for row in rows.values()} == {IDENTITY_CAPTURE}
    assert len({int(row["fetch_run_id"]) for row in rows.values()}) == 1

    pool = ch.resolve_candidate_pool(conn, CUTOFF)
    assert pool["identity_available"] is True
    assert [(p["player_id"], p["team_id"], p["element_type"]) for p in pool["players"]] == sorted(
        (player_id, _team_of(player_id), _position_of(player_id)) for player_id in ALL_PLAYERS
    )
    assert pool["summary"]["basis_counts"] == {
        ch.IDENTITY_BASIS_CUTOFF_GENERATION: len(ALL_PLAYERS)
    }
    assert pool["summary"]["cutoff_safe_rows"] == len(ALL_PLAYERS)
    assert pool["summary"]["unresolved_candidates"] == 0

    # A REJECTED generation is never read as authoritative, however new it is.
    with conn:
        _seed_generation(
            conn,
            "2026-09-08T08:00:00Z",
            [11],
            accepted=False,
            payloads={11: _official_element(11, team=3)},
        )
    rejected = ch.official_pool_generation_at_cutoff(conn, CUTOFF)
    assert rejected["element_ids"] == ALL_PLAYERS, "a refused generation must not redefine the pool"
    assert rejected["newest_attempt_is_the_accepted_generation"] is False
    assert rejected["newest_attempt"]["accepted"] is False
    assert ch.resolve_candidate_pool(conn, CUTOFF)["players"] == pool["players"]

    # A generation captured AFTER the cutoff is invisible to it.
    with conn:
        _seed_generation(
            conn, "2026-09-11T08:00:00Z", [11], payloads={11: _official_element(11, team=3)}
        )
    assert ch.official_pool_generation_at_cutoff(conn, CUTOFF)["element_ids"] == ALL_PLAYERS
    later = "2026-09-11T09:00:00Z"
    assert ch.official_pool_generation_at_cutoff(conn, later)["element_ids"] == [11]
    conn.close()


def test_a_cutoff_without_a_usable_generation_fails_closed(tmp_path):
    """No accepted generation means no pool: nothing is read from the current row."""

    conn = _db(tmp_path, generation_accepted=False)
    generation = ch.official_pool_generation_at_cutoff(conn, CUTOFF)
    assert generation["usable"] is False
    assert generation["reasons"] == [ch.GENERATION_UNAVAILABLE_NO_ACCEPTED_GENERATION]
    assert generation["generation"] is None
    resolution = ch.resolve_candidate_pool(conn, CUTOFF)
    assert resolution["identity_available"] is False
    assert resolution["players"] == [] and resolution["unresolved"] == []
    assert resolution["summary"]["identity_available"] is False
    # A projection attempt stops rather than falling back to the mutable row.
    with pytest.raises(ch.ChallengerInconsistencyError):
        ch.build_challenger_arms(conn, 4, CUTOFF)

    artifact = ev.evaluate_events(conn, [3])
    population = artifact["population"]
    assert population["scored"] == 0
    assert population["excluded_by_status"][ev.STATUS_IDENTITY_UNAVAILABLE_AT_CUTOFF] > 0
    assert population["accounting"]["reconciles"] is True
    assert population["accounting"]["status_totals_reconcile"] is True
    block = artifact["events"][0]
    assert block["status"] == "NOT_EVALUATABLE"
    assert block["exclusion_scope"] == ev.SCOPE_EVENT
    assert block["identity"]["generation"]["usable"] is False
    exit_item = next(
        item
        for item in artifact["exclusions"]
        if item["status"] == ev.STATUS_IDENTITY_UNAVAILABLE_AT_CUTOFF
    )
    assert exit_item["scope"] == ev.SCOPE_EVENT
    assert ch.GENERATION_UNAVAILABLE_NO_ACCEPTED_GENERATION in exit_item["detail"]
    conn.close()


def test_a_generation_id_set_that_contradicts_its_digest_is_refused(tmp_path):
    """Identity is proven by the id set, so a corrupt one fails closed."""

    conn = _db(tmp_path)
    with conn:
        conn.execute("UPDATE bootstrap_generations SET element_ids_sha256='deadbeef'")
    generation = ch.official_pool_generation_at_cutoff(conn, CUTOFF)
    assert generation["usable"] is False
    assert ch.GENERATION_UNAVAILABLE_ID_SET_DIGEST_MISMATCH in generation["reasons"]
    assert generation["generation"]["element_ids_sha256_matches_the_id_set"] is False
    assert ch.resolve_candidate_pool(conn, CUTOFF)["identity_available"] is False
    conn.close()


def test_a_player_absent_from_the_cutoff_generation_is_not_a_candidate(tmp_path):
    """The generation, not the persisted pool, decides membership."""

    conn = _db(tmp_path)
    # Player 23 exists in the persisted pool and has a full record, but the
    # generation a LATER cutoff resolves from does not place him in the pool.
    with conn:
        _seed_generation(
            conn,
            "2026-09-08T08:00:00Z",
            [player_id for player_id in ALL_PLAYERS if player_id != 23],
        )
        conn.execute("UPDATE players SET is_active=1 WHERE id=23")
    assert 23 in {int(row["player_id"]) for row in analytics.projectable_players(conn)}

    resolution = ch.resolve_candidate_pool(conn, CUTOFF)
    assert 23 not in _resolved_identity_rows(resolution)
    arms = ch.build_challenger_arms(conn, 4, CUTOFF, resolution=resolution)
    assert 23 not in {int(key[0]) for key in arms.keys()}
    # The divergence is REPORTED, with counts, rather than absorbed.
    divergence = resolution["summary"]["persisted_pool_divergence"]
    assert divergence["cutoff_generation_pool_size"] == len(ALL_PLAYERS) - 1
    assert divergence["in_persisted_pool_not_in_cutoff_generation"] >= 1
    assert "NOT a candidate" in divergence["rule"]

    # An EARLIER cutoff (the generation the world had then) still places him in
    # the pool, so membership is resolved per cutoff rather than globally.
    earlier = "2026-09-07T00:00:00Z"
    earlier_resolution = ch.resolve_candidate_pool(conn, earlier)
    assert earlier_resolution["generation"]["accepted_generation"]["captured_at"] == IDENTITY_CAPTURE
    assert earlier_resolution["generation"]["element_ids_count"] == len(ALL_PLAYERS)
    assert 23 in _resolved_identity_rows(earlier_resolution)

    # And an earlier projection is NOT moved by the later generation's pool.
    artifact = ev.evaluate_events(conn, [3])
    assert artifact["population"]["scored"] > 0
    conn.close()


def test_later_team_position_and_active_state_cannot_move_an_earlier_projection(tmp_path):
    """THE regression: mutate team, position AND active state for players with history."""

    conn = _db(tmp_path)
    arms = ch.build_challenger_arms(conn, 4, CUTOFF)
    artifact_before = ev.evaluate_events(conn, [2, 3])
    before = {key: _projection(_row(arms, key)) for key in sorted(arms.keys())}
    before_identity = _resolved_identity(arms)
    # Every mutated player CARRIES HISTORICAL OBSERVATIONS, so each of them is a
    # row in the league window the priors are pooled from -- exactly the case the
    # persisted-row pooling could not survive.
    assert all(player_id in before_identity for player_id in (10, 13, 23, 40))
    with conn:
        conn.execute("UPDATE players SET team_id=3, element_type=4 WHERE id=10")
        conn.execute("UPDATE players SET team_id=2, element_type=4, is_active=0 WHERE id=13")
        conn.execute("UPDATE players SET team_id=2, element_type=1, is_active=0 WHERE id=23")
        conn.execute("UPDATE players SET team_id=3, element_type=3, is_active=0 WHERE id=40")

    after = ch.build_challenger_arms(conn, 4, CUTOFF)
    assert sorted(after.keys()) == sorted(before), "the candidate set is unchanged"
    for key in before:
        assert _projection(_row(after, key)) == before[key], key
    assert _resolved_identity(after) == before_identity
    # The mutation is real in the persisted store ...
    assert conn.execute(
        "SELECT team_id, element_type, is_active FROM players WHERE id=13"
    ).fetchone()[:] == (2, 4, 0)
    # ... and the persisted pool really lost them, while the cutoff pool did not.
    live_ids = {int(row["player_id"]) for row in analytics.projectable_players(conn)}
    pool_ids = {int(player["player_id"]) for player in after.players}
    assert {13, 23, 40} & live_ids == set()
    assert {13, 23, 40} <= pool_ids

    # The ARTIFACT is byte-identical too, not just the projections: every metric,
    # stratum, exclusion, population count and digest.  The ONLY section allowed
    # to differ is the declared store-divergence report, which exists to show that
    # the persisted row moved and is an input to nothing.
    artifact_after = ev.evaluate_events(conn, [2, 3])
    assert json.dumps(_scoring_artifact(artifact_after), sort_keys=True, default=str) == json.dumps(
        _scoring_artifact(artifact_before), sort_keys=True, default=str
    )
    assert artifact_after["population"]["population_digest"] == artifact_before["population"][
        "population_digest"
    ]
    before_generation = artifact_before["population"]["identity"]["cutoff_generations"]["3"][
        "accepted_generation"
    ]
    after_generations = artifact_after["population"]["identity"]["cutoff_generations"]
    assert set(after_generations) == {"2", "3"}
    assert after_generations["3"]["accepted_generation"] == before_generation
    assert all(block["usable"] is True for block in after_generations.values())
    assert all(
        block["accepted_generation"]["captured_at"] == IDENTITY_CAPTURE
        for block in after_generations.values()
    )
    # The divergence itself DID change, and it says exactly why.
    assert artifact_after["store_divergence"] != artifact_before["store_divergence"]
    diverged = artifact_after["store_divergence"]["per_event"]
    assert set(diverged) == {"2", "3"}
    assert all(
        block["in_cutoff_generation_not_in_persisted_pool"] >= 3 for block in diverged.values()
    )
    assert all(
        block["in_persisted_pool_not_in_cutoff_generation"] == 0 for block in diverged.values()
    )
    assert artifact_after["store_divergence"]["scope"] == "reported, never scored"
    for block in artifact_after["population"]["pool_priors"]["per_event"].values():
        assert block["incumbent_league_pools_read_used"] is False
        assert block["pooled_rows_without_cutoff_identity"] == 0

    # The divergence is recorded on the row rather than absorbed: the persisted
    # pool no longer lists him at all, while the cutoff's pool still does.
    identity10 = _player(after, 10)["identity"]
    assert identity10["basis"] == ch.IDENTITY_BASIS_CUTOFF_GENERATION
    assert identity10["cutoff_safe"] is True
    assert (identity10["team_id"], identity10["element_type"]) == (1, 2)
    assert "club:1->3" in identity10["live_row_disagreement"]
    assert "position:2->4" in identity10["live_row_disagreement"]
    identity13 = _player(after, 13)["identity"]
    assert identity13["basis"] == ch.IDENTITY_BASIS_CUTOFF_GENERATION
    assert identity13["cutoff_safe"] is True
    assert (identity13["team_id"], identity13["element_type"]) == (1, 2)
    assert "membership:in_pool->absent_from_persisted_pool" in identity13["live_row_disagreement"]
    identity23 = _player(after, 23)["identity"]
    assert "membership:in_pool->absent_from_persisted_pool" in identity23["live_row_disagreement"]
    # And a caller that hands the resolution a row it has kept with is_active=0
    # gets the divergence named for what it is.
    inactive_row = dict(
        next(row for row in analytics.projectable_players(conn) if int(row["player_id"]) == 11)
    )
    inactive_row["is_active"] = 0
    with_inactive = ch.resolve_candidate_pool(conn, CUTOFF, live_rows=[inactive_row])
    assert "membership:in_pool->inactive" in _resolved_identity_of(with_inactive, 11)[
        "live_row_disagreement"
    ]
    # And the projection really used the cutoff club: player 13's event-4 fixture
    # is team 1's, the one his cutoff identity gives him.
    assert (13, _fixture_for(4, 1)) in after.keys()
    conn.close()


def test_the_priors_mirror_the_incumbent_when_identity_matches_the_current_rows(tmp_path):
    """The pooling mirror is pinned against the incumbent itself."""

    conn = _db(tmp_path)
    identities = {
        int(row["player_id"]): (int(row["team_id"]), int(row["element_type"]))
        for row in analytics.projectable_players(conn)
    }
    pools, disclosure = ch.cutoff_stable_league_pools(conn, 4, CUTOFF, identities)
    reference = incumbent.league_pools(conn, 4, CUTOFF)
    assert pools.position == reference.position
    assert pools.team_position == reference.team_position
    assert disclosure["incumbent_league_pools_read_used"] is False
    assert "OBSERVATION_SQL_CLAUSES" in disclosure["boundary"]
    assert disclosure["pooled_rows"] > 0
    assert disclosure["pooled_rows_attributed"] == disclosure["pooled_rows"]
    assert disclosure["pooled_rows_without_cutoff_identity"] == 0
    assert disclosure["pooled_players"] > 0
    # A player with no cutoff identity is not pooled, and the rows are COUNTED.
    trimmed = {key: value for key, value in identities.items() if key != 10}
    _without_10, counted = ch.cutoff_stable_league_pools(conn, 4, CUTOFF, trimmed)
    assert counted["pooled_rows_without_cutoff_identity"] > 0
    assert counted["pooled_rows"] - counted["pooled_rows_attributed"] == (
        counted["pooled_rows_without_cutoff_identity"]
    )
    conn.close()


def test_identity_disagreement_with_the_current_row_is_recorded(tmp_path):
    """A club/position change is recorded, and the cutoff still wins."""

    conn = _db(tmp_path)
    with conn:
        conn.execute("UPDATE players SET team_id=2, element_type=4 WHERE id=23")
    resolution = ch.resolve_candidate_pool(conn, CUTOFF)
    player23 = next(row for row in resolution["players"] if int(row["player_id"]) == 23)
    assert (player23["team_id"], player23["element_type"]) == (1, 3)
    assert player23["identity_cutoff_safe"] is True
    assert "club:1->2" in player23["identity"]["live_row_disagreement"]
    assert "position:3->4" in player23["identity"]["live_row_disagreement"]
    # The event-4 fixture is team 1's, not the club the current row claims.
    arms = ch.build_challenger_arms(conn, 4, CUTOFF, resolution=resolution)
    assert (23, _fixture_for(4, 1)) in arms.keys()
    assert all(int(key[1]) == _fixture_for(4, 1) for key in arms.keys() if int(key[0]) == 23)
    conn.close()


def test_post_cutoff_capture_cannot_move_an_earlier_identity(tmp_path):
    """A later capture, and a later accepted generation, are invisible."""

    conn = _db(tmp_path)
    arms = ch.build_challenger_arms(conn, 4, CUTOFF)
    before = {key: _projection(_row(arms, key)) for key in sorted(arms.keys())}
    assert _resolved_identity(arms)[11] == (1, 3)
    with conn:
        run = repo.create_fetch_run(conn, "fetch_fpl")
        # A capture after the cutoff, carrying a different club and position.
        repo.insert_snapshots(
            conn,
            [
                PlayerSnapshotRecord(
                    player_id=11,
                    captured_at="2026-09-11T08:00:00Z",
                    event_context=4,
                    now_cost=50,
                    status="a",
                    raw_json=_official_element(11, team=2, position=4),
                )
            ],
            run,
        )
    after = ch.build_challenger_arms(conn, 4, CUTOFF)
    assert _resolved_identity(after)[11] == (1, 3)
    for key in before:
        assert _projection(_row(after, key)) == before[key], key

    # A LATER ACCEPTED generation moves nobody at this cutoff either: it was not
    # observable at it.
    with conn:
        _seed_generation(
            conn,
            "2026-09-11T09:00:00Z",
            ALL_PLAYERS,
            payloads={11: _official_element(11, team=2, position=4)},
        )
        conn.execute("UPDATE players SET team_id=2, element_type=4 WHERE id=11")
    last = ch.build_challenger_arms(conn, 4, CUTOFF)
    assert _resolved_identity(last)[11] == (1, 3)
    for key in before:
        assert _projection(_row(last, key)) == before[key], key
    # The later generation DOES govern a later cutoff.
    later = ch.official_pool_generation_at_cutoff(conn, "2026-09-11T10:00:00Z")
    assert later["generation"]["captured_at"] == "2026-09-11T09:00:00Z"
    later_pool = ch.resolve_candidate_pool(conn, "2026-09-11T10:00:00Z")
    assert _resolved_identity_rows(later_pool)[11] == (2, 4)
    conn.close()


def test_a_missing_or_partial_generation_capture_is_refused_not_filled(tmp_path):
    """Missing evidence is not filled from the mutable row by default."""

    # 1. No snapshot row in the generation for this player at all.
    conn = _db(tmp_path, "missing.db", generation_missing_players=(11,))
    resolution = ch.resolve_candidate_pool(conn, CUTOFF)
    assert resolution["identity_available"] is True
    assert 11 not in _resolved_identity_rows(resolution)
    unresolved = next(item for item in resolution["unresolved"] if int(item["player_id"]) == 11)
    assert unresolved["basis"] == ch.IDENTITY_BASIS_UNRESOLVED
    assert unresolved["in_official_pool"] is True
    assert "THE_CUTOFF_GENERATION_HOLDS_NO_SNAPSHOT_ROW_FOR_THIS_PLAYER" in unresolved["notes"]
    assert any(
        reason.startswith("THE_CUTOFF_GENERATION_HOLDS_NO_SNAPSHOT_ROW_FOR_THIS_PLAYER")
        for reason in resolution["summary"]["unresolved_reasons"]
    )
    artifact = ev.evaluate_events(conn, [3])
    assert artifact["population"]["excluded_by_status"][ev.STATUS_IDENTITY_UNRESOLVED_AT_CUTOFF] == 1
    assert artifact["population"]["scored"] > 0
    assert artifact["population"]["accounting"]["reconciles"] is True
    assert artifact["population"]["identity"]["scored_rows_by_basis"] == {
        ch.IDENTITY_BASIS_CUTOFF_GENERATION: artifact["population"]["scored"]
    }
    assert artifact["population"]["identity"]["live_fallback_allowed"] is False
    # The excluded candidate is named, with its identity evidence, and it is not
    # scored anywhere.
    item = next(
        entry
        for entry in artifact["exclusions"]
        if entry["status"] == ev.STATUS_IDENTITY_UNRESOLVED_AT_CUTOFF
    )
    assert int(item["player_id"]) == 11
    assert item["candidates"] == 1
    assert item["identity"]["basis"] == ch.IDENTITY_BASIS_UNRESOLVED

    # 2. The row exists but leaves the club or the position unstated.
    conn2 = _db(tmp_path, "partial.db", generation_payloads={11: {"id": 11, "team": 1}})
    partial = ch.resolve_candidate_pool(conn2, CUTOFF)
    assert 11 not in _resolved_identity_rows(partial)
    partial_row = next(item for item in partial["unresolved"] if int(item["player_id"]) == 11)
    assert "GENERATION_CAPTURE_CARRIES_NO_POSITION_FIELD" in partial_row["notes"]
    assert ev.evaluate_events(conn2, [3])["population"]["excluded_by_status"].get(
        ev.STATUS_IDENTITY_UNRESOLVED_AT_CUTOFF
    ) == 1
    conn.close()
    conn2.close()


def test_the_live_fallback_is_an_explicit_opt_in(tmp_path):
    """The persisted row may fill a gap only when a caller asks for it."""

    conn = _db(tmp_path, generation_missing_players=(11,))
    # Default: refused, and the candidate is counted as unresolved.
    default = ch.resolve_candidate_pool(conn, CUTOFF)
    assert 11 not in _resolved_identity_rows(default)
    # Opt-in: filled from the current row, and never reported as cutoff-safe.
    opted_in = ch.resolve_candidate_pool(conn, CUTOFF, allow_live_fallback=True)
    assert _resolved_identity_rows(opted_in)[11] == (_team_of(11), _position_of(11))
    row11 = next(player for player in opted_in["players"] if int(player["player_id"]) == 11)
    assert row11["identity_basis"] == ch.IDENTITY_BASIS_LIVE_FALLBACK
    assert row11["identity_cutoff_safe"] is False
    assert row11["identity_generation_id"] is not None
    assert any(
        note.startswith("FILLED_FROM_CURRENT_ROW") for note in row11["identity"]["notes"]
    )
    assert opted_in["summary"]["basis_counts"] == {
        ch.IDENTITY_BASIS_CUTOFF_GENERATION: len(ALL_PLAYERS) - 1,
        ch.IDENTITY_BASIS_LIVE_FALLBACK: 1,
    }

    artifact = ev.evaluate_events(conn, [3], allow_live_identity_fallback=True)
    identity = artifact["population"]["identity"]
    assert identity["live_fallback_allowed"] is True
    assert identity["scored_rows_by_basis"][ch.IDENTITY_BASIS_LIVE_FALLBACK] > 0
    assert identity["scored_rows_cutoff_safe"] < identity["scored_rows"]
    assert identity["cutoff_safe_share"] < 1.0
    strict = ev.evaluate_events(conn, [3])
    assert strict["population"]["identity"]["live_fallback_allowed"] is False
    assert strict["population"]["identity"]["scored_rows_cutoff_safe"] == (
        strict["population"]["identity"]["scored_rows"]
    )
    conn.close()


def test_artifact_reports_where_every_identity_came_from(tmp_path):
    conn = _db(tmp_path)
    cutoff, _reasons = ev.event_cutoff(conn, 3)
    artifact = ev.evaluate_events(conn, [3])
    block = artifact["population"]["identity"]
    scored = artifact["population"]["scored"]
    assert scored > 0
    assert block["scored_rows"] == scored
    assert block["scored_rows_by_basis"] == {ch.IDENTITY_BASIS_CUTOFF_GENERATION: scored}
    assert block["scored_rows_cutoff_safe"] == scored
    assert block["cutoff_safe_share"] == 1.0
    assert block["live_fallback_allowed"] is False
    assert block["basis_vocabulary"] == list(ch.IDENTITY_BASES)
    assert "accepted_generation_at" in block["generation_reader"]
    assert "EVENT scope" in block["fail_closed_rule"]
    generation = block["cutoff_generations"][str(artifact["events"][0]["event"])]
    assert generation["accepted_generation"]["captured_at"] == IDENTITY_CAPTURE
    assert generation["element_ids_count"] == len(ALL_PLAYERS)
    assert "element_ids" not in generation, "the raw id list is not dumped into the artifact"

    # The priors construction is disclosed, and it is not the incumbent's read.
    priors = artifact["population"]["pool_priors"]
    assert priors["incumbent_league_pools_read_used"] is False
    assert "LeaguePools" in priors["note"] and "not used on this path" in priors["note"]
    assert "cutoff_stable_league_pools" in priors["constructed_by"]

    event_identity = artifact["events"][0]["identity"]
    assert event_identity["basis_counts"] == {ch.IDENTITY_BASIS_CUTOFF_GENERATION: len(ALL_PLAYERS)}
    assert event_identity["cutoff_safe_rows"] == event_identity["rows"] == len(ALL_PLAYERS)
    assert event_identity["unresolved_candidates"] == 0
    assert event_identity["generation"]["element_ids_count"] == len(ALL_PLAYERS)
    # Each scored row carries its own provenance, so a reviewer can audit one.
    records, _excluded, _counts, _reasons = ev.build_event_records(
        conn, 3, cutoff=cutoff, arms=ch.build_challenger_arms(conn, 3, cutoff)
    )
    assert len(records) == scored
    for item in records:
        assert item["identity_basis"] == ch.IDENTITY_BASIS_CUTOFF_GENERATION
        assert item["identity_cutoff_safe"] is True
        assert int(item["player_identity"]["team_id"]) == int(item["team_id"])
        assert int(item["player_identity"]["element_type"]) == int(item["position_id"])
        assert str(item["player_identity"]["snapshot_captured_at"]) <= cutoff
    conn.close()


def test_genuine_dnp_is_retained(tmp_path):
    """Contract case 5: a real completed zero-minute row stays real evidence."""

    conn = _db(tmp_path)
    arms = ch.build_challenger_arms(conn, 4, CUTOFF)
    key = (18, _fixture_for(4, 1))
    base = arms.incumbent_rows[key]
    assert base["evidence_classes"][incumbent.EVIDENCE_AVAILABLE_NON_START_ZERO_MINUTES] == 2
    assert base["start_evidence"]["available_non_start_zero_minute_observations"] == 2
    assert incumbent.FLAG_ZERO_MINUTE_NON_START_EVIDENCE_STRONG in base["risk_flags"]
    evidence = arms.evidence[key]
    assert evidence.cameo_rate_observations >= 2, "a genuine DNP enters the cameo denominator"
    # And it is scored as a real realisation, never excluded.
    cutoff, _reasons = ev.event_cutoff(conn, 3)
    records, _excluded, _counts, _reasons = ev.build_event_records(
        conn, 3, cutoff=cutoff, arms=ch.build_challenger_arms(conn, 3, cutoff)
    )
    dnps = [item for item in records if item["realised"]["minutes"] == 0.0]
    assert dnps, "a completed zero-minute row must be a scored observation"
    assert all(item["realised"]["started"] == 0 for item in dnps)
    conn.close()


def test_scheduled_placeholder_is_excluded(tmp_path):
    """Contract case 6: a schedule artefact is never evidence and never a zero."""

    conn = _db(tmp_path, placeholder_rows=[(23, 1, 1)])
    stored = conn.execute(
        "SELECT COUNT(*) FROM player_gameweeks WHERE player_id=23"
    ).fetchone()[0]
    assert stored == 3, "the placeholder row is really in the store"
    rows = analytics.completed_rows_as_of(conn, 23, CUTOFF, 4)
    assert {int(row["event"]) for row in rows} == {2, 3}, "the placeholder must not enter the window"
    assert all(row["minutes"] is not None for row in rows)
    arms = ch.build_challenger_arms(conn, 4, CUTOFF)
    assert sorted(arms.evidence.keys()) == sorted(arms.keys())
    assert arms.evidence[(23, _fixture_for(4, 1))].cameo_rate_observations == 0
    # An outcome-side placeholder excludes the key instead of scoring it as zero.
    # Player 24 has no official row for the target event at all, so the stored
    # placeholder is the only record and it must not be read as a 0-minute DNP.
    conn2 = _db(tmp_path, "outcome.db", placeholder_rows=[(24, 3, 4)])
    outcome, status = ev.realised_outcome(conn2, 24, 4)
    assert outcome is None and status == ev.STATUS_OUTCOME_PLACEHOLDER
    assert status == wf.OUTCOME_PLACEHOLDER_EXCLUDED
    artifact = ev.evaluate_events(conn2, [3])
    assert artifact["population"]["excluded_by_status"].get(ev.STATUS_OUTCOME_PLACEHOLDER) == 1
    assert artifact["population"]["scored"] > 0
    conn.close()
    conn2.close()


# ---------------------------------------------------------------------------
# Required hard tests 7-10: role evidence, return ramp, cameo behaviour.
# ---------------------------------------------------------------------------


def test_repeated_starts_strengthen_start_evidence(tmp_path):
    """Contract case 7: more starts -> stronger start evidence, in both arms."""

    conn = _db(tmp_path)
    arms = ch.build_challenger_arms(conn, 4, CUTOFF)
    starter = _row(arms, (10, _fixture_for(4, 1)))
    reserve = _row(arms, (12, _fixture_for(4, 1)))
    assert starter["p_start_given_available"] > reserve["p_start_given_available"]
    # One start added, everything else unchanged, must raise the estimate.
    conn2 = _db(tmp_path, "more.db", pgw_overrides={12: {1: (90, 1), 2: (0, 0), 3: (0, 0)}})
    stronger = _row(ch.build_challenger_arms(conn2, 4, CUTOFF), (12, _fixture_for(4, 1)))
    assert stronger["p_start_given_available"] > reserve["p_start_given_available"]
    assert stronger["p_start"] > reserve["p_start"]
    conn.close()
    conn2.close()


def test_repeated_available_zero_minute_non_starts_affect_role_evidence(tmp_path):
    """Contract case 8: repeated available non-starts are negative role evidence."""

    conn = _db(tmp_path)
    arms = ch.build_challenger_arms(conn, 4, CUTOFF)
    key = (18, _fixture_for(4, 1))
    row = _row(arms, key)
    regular = _row(arms, (10, _fixture_for(4, 1)))
    assert row["p_start_given_available"] < regular["p_start_given_available"]
    base = arms.incumbent_rows[key]
    assert base["role_evidence"]["available_non_start_zero_minute_observations"] == 2
    assert "repeated_available_zero_minute_non_start" in base["role_evidence"]["uncertainty_reasons"]
    assert arms.evidence[key].cameo_rate_observations == 2
    conn.close()


def test_return_from_injury_behaviour(tmp_path):
    """Contract case 9: the ramp is attenuated by demonstrated recovery only."""

    conn = _db(tmp_path)
    arms = ch.build_challenger_arms(conn, 4, CUTOFF)
    key = (13, _fixture_for(4, 1))
    base = arms.incumbent_rows[key]
    row = _row(arms, key)
    assert "RETURN_RAMP" in base["risk_flags"]
    ramp = row["refinements"][ch.FAMILY_RETURN_FROM_INJURY_RAMP]
    assert ramp["status"] == ch.REFINEMENT_APPLIED
    assert ramp["recovery_observations"] >= 1
    assert ramp["attenuation"] == pytest.approx(1.0)
    for modifier in ramp["effective_modifiers"].values():
        assert modifier == pytest.approx(1.0), "a demonstrated recovery removes the discount"
    assert row["p_start"] > base["p_start"]
    assert row["p_60_plus"] > base["p_60_plus"]
    assert row["expected_minutes"] > base["expected_minutes"]

    # With no recovery evidence at all the ramp movement is exactly zero, and an
    # arm carrying only that family reproduces the incumbent for this player.
    conn2 = _db(tmp_path, "no_recovery.db", pgw_overrides={13: {1: (0, 0), 2: (0, 0), 3: (0, 0)}})
    arms2 = ch.build_challenger_arms(
        conn2,
        4,
        CUTOFF,
        arm_definitions={ch.arm_name_for_family(ch.FAMILY_RETURN_FROM_INJURY_RAMP): frozenset({ch.FAMILY_RETURN_FROM_INJURY_RAMP})},
    )
    arm = ch.arm_name_for_family(ch.FAMILY_RETURN_FROM_INJURY_RAMP)
    row2 = arms2.arms[arm][key]
    base2 = arms2.incumbent_rows[key]
    assert row2["refinements"][ch.FAMILY_RETURN_FROM_INJURY_RAMP]["status"] == ch.REFINEMENT_NO_EVIDENCE
    assert row2["evidence"]["ramp_attenuation"] == 0.0
    assert row2["evidence"]["ramp_effective_modifiers"] == pytest.approx(
        {
            "returning_start_modifier": 0.8,
            "returning_upper_minutes_modifier": 0.75,
            "returning_minutes_if_start_modifier": 0.85,
        }
    )
    assert row2["p_start"] == base2["p_start"]
    assert row2["p_60_plus"] == base2["p_60_plus"]
    conn.close()
    conn2.close()


def test_cameo_behaviour(tmp_path):
    """Contract case 10: cameo evidence is used, recency-weighted and bounded."""

    conn = _db(tmp_path)
    arms = ch.build_challenger_arms(conn, 4, CUTOFF)
    key = (16, _fixture_for(4, 1))
    row = _row(arms, key)
    recency = row["refinements"][ch.FAMILY_CAMEO_RATE_RECENCY]
    tail = row["refinements"][ch.FAMILY_CAMEO_TAIL_EVIDENCE]
    assert recency["status"] == ch.REFINEMENT_APPLIED
    assert recency["observations"] >= 1 and recency["cameo_observations"] >= 2
    assert 0.0 <= row["p_cameo_given_not_start"] <= 1.0
    assert tail["status"] == ch.REFINEMENT_APPLIED
    assert tail["observations"] >= 2
    assert row["p_60_given_cameo"] > tail["prior"], "an observed 60+ cameo must raise the tail"

    # A player who never reached 60 from the bench has the pool prior pulled down.
    conn2 = _db(tmp_path, "short_cameo.db", pgw_overrides={16: {1: (20, 0), 2: (20, 0), 3: (0, 0)}})
    arms2 = ch.build_challenger_arms(conn2, 4, CUTOFF)
    row2 = arms2.arms[ch.ARM_ALL_REFINEMENTS][key]
    assert row2["p_60_given_cameo"] < arms2.incumbent_rows[key]["p_60_given_cameo"]
    assert row2["p_60_given_cameo"] < row["p_60_given_cameo"]
    conn.close()
    conn2.close()


# ---------------------------------------------------------------------------
# Required hard tests 11-14: coherence.
# ---------------------------------------------------------------------------


def test_60_and_80_coherence(tmp_path):
    """Contract case 11: the minutes states remain one coherent distribution."""

    conn = _db(tmp_path)
    arms = ch.build_challenger_arms(conn, 4, CUTOFF)
    for arm in arms.arm_names():
        for key, row in arms.arms[arm].items():
            assert row["p_80_plus"] <= row["p_60_plus"] + 1e-9, (arm, key)
            assert row["p_60_plus"] <= row["p_start"] + row["p_cameo"] + 1e-9, (arm, key)
            assert row["p_start"] <= row["p_available"] + 1e-9, (arm, key)
            assert abs((row["p_zero"] + row["p_1_59"] + row["p_60_plus"]) - 1.0) <= 3e-6, (arm, key)
            expected_60 = row["p_start"] * row["p_60_given_start"] + row["p_cameo"] * row["p_60_given_cameo"]
            assert row["p_60_plus"] == pytest.approx(expected_60, abs=1e-4), (arm, key)
            expected_80 = row["p_start"] * row["p_80_given_start"]
            assert row["p_80_plus"] == pytest.approx(min(expected_80, expected_60), abs=1e-4), (arm, key)
    conn.close()


def test_expected_minutes_coherence(tmp_path):
    """Contract case 12: expected minutes follow the published marginals."""

    conn = _db(tmp_path)
    arms = ch.build_challenger_arms(conn, 4, CUTOFF)
    for arm in arms.arm_names():
        for key, row in arms.arms[arm].items():
            expected = (
                row["p_start"] * row["expected_minutes_if_start"]
                + row["p_cameo"] * row["expected_minutes_if_cameo"]
            )
            assert row["expected_minutes"] == pytest.approx(expected, abs=1e-4), (arm, key)
            assert 0.0 <= row["expected_minutes"] <= 90.0, (arm, key)
            if row["p_start"] == 0.0 and row["p_cameo"] == 0.0:
                assert row["expected_minutes"] == 0.0, (arm, key)
    conn.close()


def test_no_history_fallback_remains_structurally_possible(tmp_path):
    """Contract case 13: an empty causal window still produces a coherent row."""

    conn = _db(tmp_path, drop_events=(1, 2, 3))
    arms = ch.build_challenger_arms(conn, 4, CUTOFF)
    assert arms.keys()
    for key in arms.keys():
        base = arms.incumbent_rows[key]
        row = _row(arms, key)
        assert "NO_LEAGUE_START_EVIDENCE" in base["risk_flags"]
        assert "NO_START_MINUTES_EVIDENCE" in base["risk_flags"]
        assert row["derived_chain_source"] == ch.DERIVED_BY_BASE_REPUBLICATION
        assert row["expected_minutes"] == base["expected_minutes"]
        assert 0.0 <= row["p_start"] <= 1.0
    # A population with no scored observation reports the metric as undefined,
    # never as a zero.
    artifact = ev.evaluate_events(conn, [3])
    assert artifact["sample"]["player_fixture_observations"] == 0
    assert artifact["incumbent_metrics"]["brier_p_start"]["status"] == "NO_SAMPLE"
    assert artifact["incumbent_metrics"]["brier_p_start"]["value"] is None
    assert artifact["measurement_summary"]["token"] == ev.MEASUREMENT_INSUFFICIENT_SAMPLE
    assert ev.MEASUREMENT_BASIS_INSUFFICIENT_SAMPLE in artifact["measurement_summary"]["basis"]
    conn.close()


def test_position_and_team_coherence_preserved(tmp_path):
    """Contract case 14: the frozen team-coherence layer still solves on top."""

    conn = _db(tmp_path)
    arms = ch.build_challenger_arms(conn, 4, CUTOFF)
    fixture_id = _fixture_for(4, 1)
    for team_id in (1, 2):
        payloads = []
        for (player_id, row_fixture), row in sorted(arms.arms[ch.ARM_ALL_REFINEMENTS].items()):
            if row_fixture != fixture_id or _team_of(player_id) != team_id:
                continue
            payloads.append(dict(row))
        assert len(payloads) >= 11, "each side carries a full XI plus a bench"
        coherent, record = minutes_coherence.solve_team_coherence(
            payloads, fixture_id=fixture_id, team_id=team_id, positional=True
        )
        assert len(coherent) == len(payloads)
        assert record["status"] == "COHERENT"
        assert record["start_residual"] == pytest.approx(0.0, abs=1e-6)
        assert record["minutes_residual"] == pytest.approx(0.0, abs=1e-6)
        assert record["gk_start_sum"] == pytest.approx(1.0, abs=1e-6)
        readiness = minutes_coherence.coherence_readiness([record])
        assert readiness["status"] == "PASS", readiness
    # The frozen coherent builder is unaffected and still runs on the same world.
    rows, records = minutes_coherence.build_minutes_predictions_coherent(conn, 4, CUTOFF, positional=True)
    assert rows and records
    assert minutes_coherence.coherence_readiness(records)["status"] == "PASS"
    conn.close()


# ---------------------------------------------------------------------------
# Required hard tests 15-17: determinism, same population, PE-5 append.
# ---------------------------------------------------------------------------


def test_deterministic_repeat(tmp_path):
    """Contract case 15: identical inputs and cutoff produce identical output."""

    conn = _db(tmp_path)
    first = ch.build_challenger_arms(conn, 4, CUTOFF)
    second = ch.build_challenger_arms(conn, 4, CUTOFF)
    assert first.arm_names() == second.arm_names()
    for arm in first.arm_names():
        assert _strip([first.arms[arm][key] for key in sorted(first.keys())]) == _strip(
            [second.arms[arm][key] for key in sorted(second.keys())]
        )
    first_artifact = ev.evaluate_events(conn, [2, 3])
    second_artifact = ev.evaluate_events(conn, [2, 3])
    assert json.dumps(first_artifact, sort_keys=True, default=str) == json.dumps(
        second_artifact, sort_keys=True, default=str
    )
    conn.close()


def test_same_population_incumbent_challenger_evaluation(tmp_path):
    """Contract case 16: both arms are scored on one identical population."""

    conn = _db(tmp_path)
    cutoff, _reasons = ev.event_cutoff(conn, 3)
    arms = ch.build_challenger_arms(conn, 3, cutoff)
    for arm in arms.arm_names():
        assert arms.arm_keys(arm) == arms.keys()
    arms.verify_same_population()
    with pytest.raises(wf.PopulationMismatch):
        wf.assert_same_population(arms.keys(), arms.keys()[:-1])

    artifact = ev.evaluate_events(conn, [2, 3])
    population = artifact["population"]
    assert population["shared_population_across_arms"] is True
    assert population["shared_population_enforced_by"] == "walk_forward.assert_same_population"
    assert population["scored"] > 0
    for arm, payload in artifact["arms"].items():
        assert payload["metrics"]["n"] == population["scored"], arm
    assert artifact["incumbent_metrics"]["n"] == population["scored"]
    conn.close()


def test_later_pe5_outcome_append_cannot_change_an_earlier_prediction(tmp_path):
    """Contract case 17: a later PE-5 append leaves earlier projections alone."""

    conn = _db(tmp_path)
    arms = ch.build_challenger_arms(conn, 4, CUTOFF)
    before = {key: _strip([_row(arms, key)])[0] for key in sorted(arms.keys())}
    # A real PE-5 outcome append for the event at the cutoff ...
    outcome_ledger.capture_observation(
        conn,
        grain=outcome_ledger.GRAIN_PLAYER_FIXTURE,
        event=3,
        player_id=13,
        fixture_id=4,
        fields={
            "minutes": 90,
            "starts": 1,
            "total_points": 9,
            "goals_scored": 1,
            "assists": 0,
            "clean_sheets": 1,
            "goals_conceded": 0,
            "saves": 0,
            "bonus": 3,
            "bps": 40,
            "yellow_cards": 0,
            "red_cards": 0,
            "penalties_saved": 0,
            "penalties_missed": 0,
            "own_goals": 0,
            "defensive_contribution": 0,
        },
        source_name="element_summary",
        captured_at="2026-09-18T10:00:00Z",
        event_time=EVENT_KICKOFFS[3],
    )
    # ... and a later official observation of the target event itself.
    with conn:
        repo.upsert_player_gameweeks(
            conn,
            [
                PlayerGameweekRecord(
                    player_id=13,
                    event=4,
                    fixture_id=5,
                    minutes=90,
                    starts=1,
                    total_points=9,
                    source="event_live",
                    raw_json={},
                )
            ],
            "2026-09-15T08:00:00Z",
        )
    after = {key: _strip([_row(ch.build_challenger_arms(conn, 4, CUTOFF), key)])[0] for key in before}
    assert after == before
    conn.close()


# ---------------------------------------------------------------------------
# Required hard test 18: nothing outside availability / minutes changed.
# ---------------------------------------------------------------------------


def test_no_attacking_team_strength_or_bonus_model_changes():
    """Contract case 18: PE-6 touches availability and minutes only."""

    assert xpts.XPTS_MODEL_VERSION == "xpts_v1.4.1"
    assert team_model.TEAM_MODEL_VERSION == "team_strength_v1.1.0"
    assert team_model.TEAM_BASELINE_MODEL_VERSION == "team_naive_v1.1.0"
    assert bonus_challenger.BONUS_BPS_MODEL_VERSION == "bonus_bps_v1.0.0"
    assert bonus_allocation.BONUS_ALLOCATOR_VERSION == "fpl_bonus_allocation_v1.0.0"
    assert bps_rules.BPS_RULES_VERSION == "fpl_bps_2026_27_v1.0.0"
    assert monte_carlo.MONTE_CARLO_MODEL_VERSION == "mc_v1.3.0"
    source = inspect.getsource(ch) + inspect.getsource(ev)
    for forbidden in (
        "xpts",
        "team_model",
        "bonus_challenger",
        "bonus_allocation",
        "bps_rules",
        "player_rates",
        "import random",
        "monte_carlo",
    ):
        assert forbidden not in source, f"PE-6 must not reach into {forbidden}"


# ---------------------------------------------------------------------------
# Internal consistency: mirrors, no-op identity, evidence taxonomy.
# ---------------------------------------------------------------------------


def test_math_mirrors_match_the_incumbent():
    for total in (0.0, 1.0, 3.0, 7.5):
        for count in (0.0, 2.0, 9.0):
            for prior in (0.0, 0.25, 0.7, 1.0):
                for strength in (1.0, 4.0, 8.0):
                    assert ch._shrink(total, count, prior, strength) == incumbent._shrink(
                        total, count, prior, strength
                    )
    for count in (0, 1, 3, 7):
        for half_life in (1.0, 4.0, 12.5):
            assert ch._recency_weights(count, half_life) == incumbent._recency_weight_list(
                count, half_life
            )
    assert ch.START_OBSERVATION_CLASSES == incumbent._START_OBSERVATION_CLASSES
    assert "expected_minutes" in ch.DERIVED_CHAIN_FIELDS and "p_start" in ch.DERIVED_CHAIN_FIELDS


def test_scout_return_ramp_flag_mirrors_the_incumbent():
    notes = [
        {
            "id": 1,
            "key": "injury_uncertainty",
            "value_num": 80.0,
            "observed_at": "2026-09-09T09:00:00Z",
            "expires_at": "2026-09-30T00:00:00Z",
        }
    ]
    _adjustments, _ids, flags = incumbent.fold_scout_modifiers(
        notes, CUTOFF, incumbent.MinutesModelConfig()
    )
    assert ch.SCOUT_RETURN_RAMP_FLAG in flags


def test_availability_basis_matches_the_incumbent_declared_assumption(tmp_path):
    conn = _db(tmp_path)
    arms = ch.build_challenger_arms(conn, 4, CUTOFF)
    declared_key = (13, _fixture_for(4, 1))
    declared = arms.incumbent_rows[declared_key]
    assert "initial modelling assumption, not calibrated" in declared["availability_source_summary"]["rule"]
    assert arms.evidence[declared_key].availability_basis == ch.BASIS_DECLARED_STATUS_DEFAULT
    hard_key = (14, _fixture_for(4, 1))
    assert "hard official status" in arms.incumbent_rows[hard_key]["availability_source_summary"]["rule"]
    assert arms.evidence[hard_key].availability_basis == ch.BASIS_HARD_STATUS
    # An event-specific chance signal takes precedence over any status estimate.
    conn2 = _db(
        tmp_path,
        "chance.db",
        extra_snapshots=[(13, 4, "i", 75, "2026-09-10T09:00:00Z")],
    )
    arms2 = ch.build_challenger_arms(conn2, 4, CUTOFF)
    assert arms2.evidence[declared_key].availability_basis == ch.BASIS_EVENT_CHANCE
    assert arms2.evidence[declared_key].status_estimate is None
    conn.close()
    conn2.close()


def test_an_arm_with_no_refinements_reproduces_the_incumbent_exactly(tmp_path):
    conn = _db(tmp_path)
    arms = ch.build_challenger_arms(conn, 4, CUTOFF, arm_definitions={"replay": frozenset()})
    for key in arms.keys():
        replayed = arms.arms["replay"][key]
        base = arms.incumbent_rows[key]
        assert replayed["derived_chain_source"] == ch.DERIVED_BY_BASE_REPUBLICATION
        for name in (
            *ch.DERIVED_CHAIN_FIELDS,
            "p_available",
            "p_start_given_available",
            "p_cameo_given_not_start",
            "p_60_given_start",
            "p_80_given_start",
            "p_60_given_cameo",
            "expected_minutes_if_start",
            "expected_minutes_if_cameo",
        ):
            assert replayed[name] == base[name], (key, name)
    conn.close()


def test_assumed_attribution_is_not_refined_evidence(tmp_path):
    """A trail that never varied proves nothing about the row it is attributed to."""

    conn = _db(tmp_path, pgw_overrides={11: {1: (90, 1), 2: (0, 0), 3: (0, 0)}})
    with conn:
        run = repo.create_fetch_run(conn, "fetch_fpl")
        # One snapshot only, no event context: the incumbent attributes it to
        # every row ("assumed_constant_status").
        repo.insert_snapshots(
            conn,
            [
                PlayerSnapshotRecord(
                    player_id=11,
                    captured_at="2026-09-10T09:00:00Z",
                    now_cost=50,
                    status="d",
                    raw_json={},
                )
            ],
            run,
        )
    arms = ch.build_challenger_arms(conn, 4, CUTOFF)
    evidence = arms.evidence[(11, _fixture_for(4, 1))]
    assert evidence.availability_basis == ch.BASIS_DECLARED_STATUS_DEFAULT
    assert evidence.assumed_attribution_rows_excluded >= 1
    assert evidence.status_estimate is None, "an assumed trail is not status evidence"
    assert "STATUS_ESTIMATE_NO_EVENT_SPECIFIC_ATTRIBUTION" in evidence.notes
    conn.close()


def test_cameo_recency_weighting_follows_the_declared_half_life(tmp_path):
    """Recent cameo evidence outweighs stale cameo evidence."""

    recent = _db(tmp_path, "recent.db", pgw_overrides={12: {1: (0, 0), 2: (0, 0), 3: (30, 0)}})
    stale = _db(tmp_path, "stale.db", pgw_overrides={12: {1: (30, 0), 2: (0, 0), 3: (0, 0)}})
    key = (12, _fixture_for(4, 1))
    recent_arms = ch.build_challenger_arms(recent, 4, CUTOFF)
    stale_arms = ch.build_challenger_arms(stale, 4, CUTOFF)
    recent_row = _row(recent_arms, key)
    stale_row = _row(stale_arms, key)
    assert recent_row["p_cameo_given_not_start"] > stale_row["p_cameo_given_not_start"]
    assert recent_row["evidence"]["cameo_rate_estimate"] > stale_row["evidence"]["cameo_rate_estimate"]
    # The same one-cameo-in-three record: the incumbent cannot tell the two apart,
    # because its cameo evidence is unweighted.
    assert (
        recent_arms.incumbent_rows[key]["p_cameo_given_not_start"]
        == stale_arms.incumbent_rows[key]["p_cameo_given_not_start"]
    )
    recent.close()
    stale.close()


# ---------------------------------------------------------------------------
# The evaluation contract.
# ---------------------------------------------------------------------------


def test_evaluation_reports_the_declared_metric_set_with_sample_sizes(tmp_path):
    conn = _db(tmp_path)
    artifact = ev.evaluate_events(conn, [2, 3])
    metrics = artifact["incumbent_metrics"]
    for name in (
        "brier_p_start",
        "brier_p_60",
        "brier_p_80",
        "brier_p_cameo",
        "brier_p_appearance",
        "expected_minutes_mae",
        "expected_minutes_bias",
        "brier_p_start_reference",
        "brier_p_60_reference",
    ):
        assert name in metrics, name
        assert metrics[name]["n"] == metrics["n"], name
        assert metrics[name]["status"] in {"OK", "NO_SAMPLE", "INSUFFICIENT_SAMPLE"}
    for name in (
        "start_rate_predicted",
        "start_rate_realised",
        "p60_rate_predicted",
        "p60_rate_realised",
        "appearance_rate_predicted",
        "appearance_rate_realised",
    ):
        assert metrics[name] is not None, name
    assert metrics["n"] > 0
    policy = artifact["sample"]["policy"]
    assert policy["policy_version"] == ev.wfs.SAMPLE_POLICY_VERSION
    assert policy["min_target_events_for_descriptive_reporting"] == ev.wfs.MIN_TARGET_EVENTS_FOR_DESCRIPTIVE
    assert policy["min_observations_for_descriptive_reporting"] == ev.wfs.MIN_OBSERVATIONS_FOR_DESCRIPTIVE
    conn.close()


def test_evaluation_refuses_to_claim_anything_on_an_insufficient_sample(tmp_path):
    """PE-2's descriptive floor gates a DESCRIPTION, never a model selection."""

    conn = _db(tmp_path)
    artifact = ev.evaluate_events(conn, [3])
    assert artifact["sample"]["sample_interpretation"] == ev.wfs.SAMPLE_INSUFFICIENT
    assert artifact["sample"]["player_fixture_observations"] < ev.wfs.MIN_OBSERVATIONS_FOR_DESCRIPTIVE
    for arm, payload in artifact["arms"].items():
        assert payload["measurement"]["token"] == ev.MEASUREMENT_INSUFFICIENT_SAMPLE, arm
        assert ev.MEASUREMENT_BASIS_INSUFFICIENT_SAMPLE in payload["measurement"]["basis"]
    summary = artifact["measurement_summary"]
    assert summary["token"] == ev.MEASUREMENT_INSUFFICIENT_SAMPLE
    assert summary["model_selection"] == ev.SELECTION_CLAIM_NOT_MADE
    assert summary["review_required"] is True
    assert artifact["identity"]["push_or_promotion_performed"] is False
    assert "evidence for senior review" in artifact["identity"]["promotion_note"]
    # Nothing in the artifact reads as an acceptance, and the policy says so.
    blob = json.dumps(artifact, default=str, sort_keys=True)
    for stale in (
        "CHALLENGER_ACCEPTED",
        "PARTIALLY_ACCEPTED",
        "accepted_refinement_families",
        '"recommendation"',
    ):
        assert stale not in blob, stale
    conn.close()


def test_evaluation_strata_are_labeled_with_their_sample_size(tmp_path):
    conn = _db(tmp_path)
    artifact = ev.evaluate_events(conn, [2, 3])
    for dimension in ev.STRATUM_DIMENSIONS:
        section = artifact["strata"][dimension]
        assert section["min_observations_for_stratum_reporting"] == ev.MIN_STRATUM_OBSERVATIONS
        for key, block in section["strata"].items():
            assert block["n"] > 0, (dimension, key)
            assert block["sample_interpretation"] in {
                "SUFFICIENT_FOR_STRATUM_DESCRIPTIVE_REPORTING",
                "INSUFFICIENT_STRATUM_SAMPLE",
            }
            assert {"comparison", "incumbent", "challenger"} <= set(block)
    assert {"1", "2", "3", "4"} <= set(artifact["strata"]["by_position"]["strata"])
    assert set(artifact["strata"]["by_return_ramp"]["strata"]) <= {"RETURN_RAMP", "NO_RETURN_RAMP"}
    assert set(artifact["strata"]["by_rotation_risk"]["strata"]) <= {"ROTATION_RISK", "NO_ROTATION_RISK"}
    assert "DECLARED_STATUS_AVAILABILITY_DEFAULT" in artifact["strata"]["by_availability_basis"]["strata"]
    conn.close()


def test_evaluation_excludes_are_labelled_and_never_scored(tmp_path):
    # Player 24 carries no official row for the target events; the stored
    # placeholder is the only record for his event-3 fixture.
    conn = _db(tmp_path, placeholder_rows=[(24, 3, 4)])
    artifact = ev.evaluate_events(conn, [2, 3])
    statuses = artifact["population"]["excluded_by_status"]
    assert statuses.get(ev.STATUS_BLANK_NO_FIXTURE, 0) > 0
    assert statuses.get(ev.STATUS_OUTCOME_PLACEHOLDER, 0) == 1
    assert statuses.get(ev.STATUS_OUTCOME_MISSING, 0) >= 1
    assert artifact["population"]["scored"] > 0
    for item in artifact["exclusions"]:
        assert item["status"] in ev.EXCLUSION_STATUSES
        assert item["detail"]
        assert item["scope"] in {ev.SCOPE_PLAYER, ev.SCOPE_EVENT}
        assert item["candidates"] >= 1
    conn.close()


def test_double_gameweek_fixtures_are_retained_and_scored(tmp_path):
    """Contract grain: minutes are a player-fixture quantity, DGW included."""

    conn = _db(tmp_path)
    # Event 2 is a double gameweek for team 2 (fixtures 2 and 3).
    cutoff, _reasons = ev.event_cutoff(conn, 2)
    arms = ch.build_challenger_arms(conn, 2, cutoff)
    records, excluded, _counts, _reasons = ev.build_event_records(
        conn, 2, cutoff=cutoff, arms=arms
    )
    dgw = [item for item in records if int(item["player_id"]) in (29, 30, 31)]
    assert dgw, "the double gameweek's fixtures must be scored, not excluded"
    assert {int(item["fixture_id"]) for item in dgw} == {2, 3}
    assert not any(
        item["status"] == "MULTI_FIXTURE_EVENT_POINT_AGGREGATION_UNSPECIFIED" for item in excluded
    )
    # Both fixtures of the same player are separate observations with their own
    # realised outcome, and each keeps its own projection.
    by_key = {(int(item["player_id"]), int(item["fixture_id"])): item for item in records}
    for player_id in (29, 30, 31):
        assert (player_id, 2) in by_key and (player_id, 3) in by_key

    artifact = ev.evaluate_events(conn, [2])
    assert artifact["grain"] == wf.GRAIN_PLAYER_FIXTURE
    assert artifact["grain"] == ev.GRAIN_PLAYER_FIXTURE
    assert artifact["sample"]["player_fixture_observations"] == artifact["population"]["scored"]
    assert artifact["population"]["scored"] >= len(dgw)
    assert "scored: a double gameweek is two player-fixture observations" in json.dumps(
        artifact["population_rule"]
    )
    # PE-2 declared this grain for minutes; PE-6 reuses the same constant.
    assert artifact["grain"] == ev.wfs.wf.GRAIN_PLAYER_FIXTURE
    conn.close()


def test_repeated_event_ids_cannot_inflate_any_count(tmp_path):
    """A repeated event id is normalized before anything is projected or scored."""

    conn = _db(tmp_path)
    once = ev.evaluate_events(conn, [2, 3])
    twice = ev.evaluate_events(conn, [3, 2, 3, 2, 3])
    assert twice["population"]["target_events"] == once["population"]["target_events"] == [2, 3]
    assert twice["population"]["scored"] == once["population"]["scored"]
    assert twice["sample"]["player_fixture_observations"] == once["sample"]["player_fixture_observations"]
    assert twice["sample"]["target_events"] == once["sample"]["target_events"]
    assert twice["population"]["population_digest"] == once["population"]["population_digest"]
    assert twice["population"]["candidates"] == once["population"]["candidates"]
    assert twice["population"]["excluded_by_status"] == once["population"]["excluded_by_status"]
    for arm, payload in twice["arms"].items():
        assert payload["metrics"]["n"] == once["arms"][arm]["metrics"]["n"], arm
    request = twice["population"]["event_request"]
    assert request["requested"] == [3, 2, 3, 2, 3]
    assert request["normalized"] == [2, 3]
    assert request["duplicates_removed"] == 3
    assert request["repeated_events"] == {"2": 2, "3": 3}
    assert once["population"]["event_request"]["duplicates_removed"] == 0
    conn.close()


def test_evaluation_accounting_reconciles_when_a_cutoff_is_unavailable(tmp_path):
    """An EVENT-scope exclusion accounts for its whole enumeration, so the totals add up."""

    conn = _db(tmp_path)
    with conn:
        repo.upsert_events(
            conn, [EventRecord(id=9, finished=1, data_checked=1, deadline_time=None, raw_json={})]
        )
    pool = len(analytics.projectable_players(conn))
    artifact = ev.evaluate_events(conn, [3, 9])
    population = artifact["population"]
    accounting = population["accounting"]
    assert population["candidates"] == population["scored"] + population["excluded"]
    assert accounting["enumerated_candidate_slots"] == (
        accounting["scored_rows"] + accounting["excluded_candidates"]
    )
    assert accounting["reconciles"] is True
    assert accounting["status_totals_reconcile"] is True
    assert accounting["per_event_reconciles"] is True
    # The whole unavailable-cutoff pool is accounted for, and exactly once.
    assert population["excluded_by_status"][ev.STATUS_EVENT_CUTOFF_UNAVAILABLE] == pool
    assert accounting["event_scope_exclusions"] == 1
    assert accounting["event_scope_candidates"] == pool
    assert sum(population["excluded_by_status"].values()) == population["excluded"]
    assert accounting["excluded_by_status_total"] == population["excluded"]
    event_block = next(block for block in artifact["events"] if int(block["event"]) == 9)
    assert event_block["status"] == "NOT_EVALUATED"
    assert event_block["candidates"] == pool
    assert event_block["candidate_slots_enumerated"] == pool
    assert event_block["excluded_candidates"] == pool
    assert event_block["scored_observations"] == 0
    # The scored event is untouched by the unavailable one.
    assert population["scored"] == ev.evaluate_events(conn, [3])["population"]["scored"]
    conn.close()


def test_candidate_slots_are_enumerated_independently_of_the_scored_rows(tmp_path):
    """The reconciliation compares an INDEPENDENT enumeration, not its own sum."""

    conn = _db(tmp_path)
    artifact = ev.evaluate_events(conn, [2, 3])
    accounting = artifact["population"]["accounting"]
    assert accounting["reconciles"] is True
    assert accounting["status_totals_reconcile"] is True
    assert accounting["per_event_reconciles"] is True
    assert accounting["enumerated_candidate_slots"] == (
        accounting["scored_rows"] + accounting["excluded_candidates"]
    )
    assert accounting["excluded_by_status_total"] == accounting["excluded_candidates"]
    assert accounting["scored_rows"] == artifact["population"]["scored"]

    # The enumeration is drawn from the pool and the fixtures, so it can be
    # recomputed by hand: every candidate is one slot per fixture of his club in
    # the event, or one slot when his club has none.
    for event in (2, 3):
        cutoff, _reasons = ev.event_cutoff(conn, event)
        resolution = ch.resolve_candidate_pool(conn, cutoff)
        enumeration = ev.candidate_slot_enumeration(conn, event, resolution=resolution)
        fixtures_by_team = analytics.event_fixture_map(conn, event)
        expected = sum(
            len(fixtures_by_team.get(int(player["team_id"])) or []) or 1
            for player in resolution["players"]
        ) + len(resolution["unresolved"])
        assert enumeration["slots"] == expected
        assert enumeration["resolved_candidates"] == len(resolution["players"])
        assert enumeration["unresolved_candidates"] == len(resolution["unresolved"])
        block = next(item for item in artifact["events"] if int(item["event"]) == event)
        assert block["candidate_slots_enumerated"] == expected
        assert block["candidate_slot_enumeration"] == enumeration
        assert block["candidate_slots_enumerated"] == (
            block["scored_observations"] + block["excluded_candidates"]
        )
    # Event 2 is a double gameweek for team 2, so its slots really are counted per
    # fixture rather than per player.
    cutoff2, _reasons = ev.event_cutoff(conn, 2)
    resolution2 = ch.resolve_candidate_pool(conn, cutoff2)
    enum2 = ev.candidate_slot_enumeration(conn, 2, resolution=resolution2)
    team2_fixtures = analytics.event_fixture_map(conn, 2)[2]
    assert len(team2_fixtures) == 2
    assert enum2["slots"] > enum2["resolved_candidates"]
    # The totals are the sum of the per-event enumerations, not of the records.
    assert accounting["enumerated_candidate_slots"] == sum(
        int(block["candidate_slots_enumerated"]) for block in artifact["events"]
    )
    conn.close()


def test_an_unplayed_fixture_is_excluded_not_scored_as_zero(tmp_path):
    """A finalised event whose fixture has not been played scores nothing."""

    conn = _db(tmp_path)
    with conn:
        repo.upsert_fixtures(
            conn,
            [
                FixtureRecord(
                    id=4,
                    event=3,
                    team_h=1,
                    team_a=3,
                    finished=0,
                    started=0,
                    kickoff_time=EVENT_KICKOFFS[3],
                    raw_json={},
                )
            ],
        )
    artifact = ev.evaluate_events(conn, [3])
    assert artifact["population"]["excluded_by_status"].get(ev.STATUS_FIXTURE_NOT_PLAYED, 0) > 0
    assert artifact["population"]["scored"] == 0
    assert artifact["incumbent_metrics"]["brier_p_start"]["status"] == "NO_SAMPLE"
    assert artifact["population"]["accounting"]["reconciles"] is True
    conn.close()


def test_evaluation_requires_a_causal_cutoff(tmp_path):
    conn = _db(tmp_path)
    with conn:
        repo.upsert_events(
            conn, [EventRecord(id=9, finished=1, data_checked=1, deadline_time=None, raw_json={})]
        )
    cutoff, reasons = ev.event_cutoff(conn, 9)
    assert cutoff is None and reasons
    artifact = ev.evaluate_events(conn, [9])
    assert artifact["sample"]["player_fixture_observations"] == 0
    assert artifact["population"]["events_excluded"]
    assert artifact["events"][0]["status"] == "NOT_EVALUATED"
    conn.close()


def test_evaluation_population_digest_covers_the_scored_keys(tmp_path):
    conn = _db(tmp_path)
    artifact = ev.evaluate_events(conn, [3])
    cutoff, _reasons = ev.event_cutoff(conn, 3)
    arms = ch.build_challenger_arms(conn, 3, cutoff)
    records, _excluded, _counts, _reasons = ev.build_event_records(conn, 3, cutoff=cutoff, arms=arms)
    keys = [
        (int(item["event"]), int(item["player_id"]), int(item["fixture_id"])) for item in records
    ]
    assert artifact["population"]["population_digest"] == wf.canonical_population_digest(
        keys, grain=ev.GRAIN_PLAYER_FIXTURE
    )
    assert artifact["population"]["scored"] == len(keys)
    assert artifact["population"]["coverage_share"] is not None
    conn.close()


def test_evaluation_units_are_declared(tmp_path):
    conn = _db(tmp_path)
    artifact = ev.evaluate_events(conn, [3])
    assert artifact["cutoff_policy"] == "TARGET_EVENT_DEADLINE_TIME"
    assert artifact["grain"] == ev.GRAIN_PLAYER_FIXTURE == wf.GRAIN_PLAYER_FIXTURE
    assert artifact["missing_data_policy_version"] == wf.MISSING_DATA_POLICY_VERSION
    assert artifact["population_rule"]["never_scored_as_zero"]
    assert artifact["population_rule"]["exclusion_statuses"] == list(ev.EXCLUSION_STATUSES)
    comparison = artifact["arms"][ch.ARM_ALL_REFINEMENTS]["comparison"]
    assert "challenger - incumbent" in comparison["delta_convention"]
    assert comparison["metrics"]["expected_minutes_bias"]["status"] in {"OK", "UNDEFINED"}
    assert comparison["references"]["brier_p_start_reference"]["incumbent"]["status"] == "OK"
    summary = artifact["measurement_summary"]
    assert summary["measurement_vocabulary"] == list(ev.MEASUREMENT_VOCABULARY)
    assert summary["primary_criteria"] == list(ev.PRIMARY_CRITERIA)
    assert summary["improvement_floors"]["brier"] == ev.DELTA_BRIER_IMPROVEMENT_MIN
    rule = artifact["measurement_rule"]
    assert rule["primary_criteria"] == list(ev.PRIMARY_CRITERIA)
    assert rule["criteria_declared_a_priori"] is True
    assert rule["selected_from_the_data"] is False
    assert "NOT significance tests" in rule["floors"]["basis"]
    # The outcome reader is declared, and PE-5's role in PE-6 is stated.
    evidence = artifact["outcome_evidence"]
    assert "FINAL" in evidence["event_finality_predicate"]
    assert evidence["pe5_ledger"] == "NOT_THE_READER_FOR_A_WALK_FORWARD_EVENT_SET"
    assert "capture_observation" in evidence["pe5_reuse"]
    # The phase's terminal state belongs to senior review, and says so.
    boundary = artifact["terminal_boundary"]
    assert boundary["state_vocabulary"] == ["READY_FOR_MERGE", "OPEN"]
    assert boundary["phase_state_before_review"] == "OPEN"
    assert boundary["set_by"] == "senior review"
    conn.close()


def test_refine_row_refuses_a_base_row_without_conditionals(tmp_path):
    """The coherent chain needs the conditionals; a bare row fails closed."""

    conn = _db(tmp_path)
    arms = ch.build_challenger_arms(conn, 4, CUTOFF)
    base = dict(arms.incumbent_rows[(10, _fixture_for(4, 1))])
    for name in ("p_60_given_start", "p_80_given_start", "p_60_given_cameo"):
        base.pop(name)
    with pytest.raises(ch.ChallengerInconsistencyError):
        ch.refine_row(
            base,
            arms.evidence[(10, _fixture_for(4, 1))],
            config=incumbent.MinutesModelConfig(),
            challenger_config=ch.AvailabilityMinutesChallengerConfig(),
            enabled_families=ch.REFINEMENT_FAMILIES,
        )
    with pytest.raises(ValueError):
        ch.refine_row(
            arms.incumbent_rows[(10, _fixture_for(4, 1))],
            arms.evidence[(10, _fixture_for(4, 1))],
            config=incumbent.MinutesModelConfig(),
            challenger_config=ch.AvailabilityMinutesChallengerConfig(),
            enabled_families=("not_a_family",),
        )
    conn.close()


def test_coherent_chain_rejects_an_impossible_distribution():
    with pytest.raises(ValueError):
        ch.coherent_chain(
            p_available=1.5,
            p_start_given_available=1.0,
            p_cameo_given_not_start=0.0,
            p_60_given_start=1.0,
            p_80_given_start=1.0,
            p_60_given_cameo=0.0,
            expected_minutes_if_start=90.0,
            expected_minutes_if_cameo=15.0,
        )
    chain = ch.coherent_chain(
        p_available=1.0,
        p_start_given_available=0.8,
        p_cameo_given_not_start=0.5,
        p_60_given_start=0.9,
        p_80_given_start=0.7,
        p_60_given_cameo=0.05,
        expected_minutes_if_start=85.0,
        expected_minutes_if_cameo=15.0,
    )
    assert chain["p_zero"] + chain["p_1_59"] + chain["p_60_plus"] == pytest.approx(1.0, abs=3e-6)
    assert chain["p_80_plus"] <= chain["p_60_plus"]


def test_measurement_rule_is_a_declared_pure_function():
    """Insufficient evidence is never converted into a model-selection claim."""

    def block(start, sixty, mae):
        """``start``/``sixty``/``mae`` are the challenger's deltas on the incumbent."""

        return {
            "metrics": {
                "brier_p_start": {
                    "status": "OK",
                    "value": 0.2 + start,
                    "preferred": ev._preference("brier_p_start", 0.2, 0.2 + start),
                },
                "brier_p_60": {
                    "status": "OK",
                    "value": 0.2 + sixty,
                    "preferred": ev._preference("brier_p_60", 0.2, 0.2 + sixty),
                },
                "expected_minutes_mae": {
                    "status": "OK",
                    "value": 10.0 + mae,
                    "preferred": ev._preference("expected_minutes_mae", 10.0, 10.0 + mae),
                },
            }
        }

    all_better = ev.arm_measurement(block(-0.01, -0.01, -0.5), sample_sufficient=True)
    assert all_better["token"] == ev.MEASUREMENT_CHALLENGER_PREFERRED
    assert ev.MEASUREMENT_BASIS_ALL in all_better["basis"]
    partial = ev.arm_measurement(block(-0.01, 0.0, 0.0), sample_sufficient=True)
    assert partial["token"] == ev.MEASUREMENT_PARTIAL
    noise = ev.arm_measurement(block(0.0005, 0.0005, 0.005), sample_sufficient=True)
    assert noise["token"] == ev.MEASUREMENT_WITHIN_NOISE
    assert ev.MEASUREMENT_BASIS_WITHIN_NOISE in noise["basis"]
    worse = ev.arm_measurement(block(0.01, -0.01, -0.5), sample_sufficient=True)
    assert worse["token"] == ev.MEASUREMENT_INCUMBENT_PREFERRED
    assert ev.MEASUREMENT_BASIS_CHALLENGER_WORSE in worse["basis"]
    insufficient = ev.arm_measurement(block(-0.01, -0.01, -0.5), sample_sufficient=False)
    assert insufficient["token"] == ev.MEASUREMENT_INSUFFICIENT_SAMPLE
    assert ev.MEASUREMENT_BASIS_INSUFFICIENT_SAMPLE in insufficient["basis"]
    undefined = ev.arm_measurement(
        {"metrics": {"brier_p_start": {"status": "UNDEFINED"}}}, sample_sufficient=True
    )
    assert undefined["token"] == ev.MEASUREMENT_UNDEFINED
    assert ev.MEASUREMENT_BASIS_UNDEFINED in undefined["basis"]
    # Every token is a statement about numbers; none of them is an acceptance.
    for token in ev.MEASUREMENT_VOCABULARY:
        assert "ACCEPT" not in token and "PROMOT" not in token and "CERTIF" not in token, token
    assert ev.SELECTION_CLAIM_NOT_MADE not in ev.MEASUREMENT_VOCABULARY
    assert "MODEL_SELECTION" in ev.CLAIMS_NOT_ALLOWED
    assert "DESCRIPTIVE_REPORTING_ONLY" in ev.CLAIMS_ALLOWED


def test_a_multi_family_subset_is_only_reported_through_its_own_arm(tmp_path):
    """Combining single-family measurements would be a different model."""

    conn = _db(tmp_path)
    pairs = frozenset({ch.FAMILY_AVAILABILITY_STATUS_EVIDENCE, ch.FAMILY_CAMEO_RATE_RECENCY})
    definitions = dict(ch.default_arm_definitions())
    definitions["challenger_proposed_pair"] = pairs
    artifact = ev.evaluate_events(conn, [3], arm_definitions=definitions)
    policy = artifact["multi_family_subset_policy"]
    subsets = policy["declared_subsets"]
    assert set(subsets) == {ch.ARM_ALL_REFINEMENTS, "challenger_proposed_pair"}
    assert all(block["status"] == "EVALUATED_AS_ITS_OWN_ARM" for block in subsets.values())
    assert subsets["challenger_proposed_pair"]["families"] == sorted(pairs)
    assert subsets[ch.ARM_ALL_REFINEMENTS]["headline"] is True
    assert subsets["challenger_proposed_pair"]["headline"] is False
    # The subset has its own metrics and its own measurement, not a union.
    payload = artifact["arms"]["challenger_proposed_pair"]
    assert payload["families"] == sorted(pairs)
    assert payload["metrics"]["n"] == artifact["population"]["scored"]
    assert payload["measurement"]["token"] in ev.MEASUREMENT_VOCABULARY
    for family in pairs:
        solo = artifact["arms"][ch.arm_name_for_family(family)]
        assert solo["families"] == [family]
        assert solo["measurement"] is not payload["measurement"]
    assert "no combination is assembled" in policy["rule"]
    # A subset that was never evaluated can never be claimed.
    assert ev.multi_family_subset_policy(
        {"challenger_proposed_pair": pairs}, {}
    )["declared_subsets"]["challenger_proposed_pair"]["status"] == "NOT_EVALUATED"
    conn.close()


def test_no_second_causal_predicate_exists_in_pe6():
    """PE-6 reuses the frozen boundary and the canonical generation reader."""

    challenger_source = inspect.getsource(ch)
    # Every database read in the challenger is declared as a named constant, and
    # each one reads exactly one thing:
    #   * the pool identity, from the generation the cutoff could see;
    #   * which generation attempts existed at the cutoff (audit, never identity);
    #   * the historical rows the priors are pooled from, under the PE-1 clause.
    assert challenger_source.count("SELECT") == 4, "four declared reads"
    for statement in (ch.GENERATION_SNAPSHOT_SQL, ch.GENERATION_SNAPSHOT_BY_CAPTURE_SQL):
        assert "FROM player_snapshots" in statement
        assert "fetch_run_id = ?" in statement or "captured_at = ?" in statement
        assert "captured_at <= ?" in statement
        for forbidden in ("player_gameweeks", "FROM fixtures", "updated_at", "finished", "kickoff_time"):
            assert forbidden not in statement, forbidden
    assert "FROM bootstrap_generations WHERE captured_at <= ?" in ch.GENERATION_ATTEMPT_SQL
    # The pooling read does NOT restate the causal boundary: it interpolates the
    # canonical clause, and takes its parameters from the canonical helper.
    assert "{boundary}" in ch.PRIORS_SQL_TEMPLATE
    assert "historical.OBSERVATION_SQL_CLAUSES" in challenger_source
    assert "historical.boundary_params" in challenger_source
    assert "updated_at" not in ch.PRIORS_SQL_TEMPLATE
    assert "FROM players" not in challenger_source, (
        "the persisted players row may be read for a divergence REPORT and through the explicit "
        "fallback, never to resolve an identity by default"
    )
    # No second definition of "the latest accepted generation at or before the
    # cutoff": PE-5's canonical reader is reused.
    assert "outcome_ledger.accepted_generation_at" in challenger_source
    assert "analytics.completed_rows_as_of" in challenger_source
    assert "analytics.snapshot_as_of" in challenger_source
    assert "analytics.snapshot_history_as_of" in challenger_source
    # The priors mirror the incumbent's own aggregation, which the suite pins
    # against the incumbent itself rather than against a restated formula.
    assert "incumbent.LeaguePools" in challenger_source
    evaluation_source = inspect.getsource(ev)
    # The evaluation's only reads are the exact-key outcome row, the event
    # deadline and the fixture kickoffs: never a window predicate.
    assert "FROM player_gameweeks WHERE player_id=? AND fixture_id=?" in evaluation_source
    assert "updated_at" not in evaluation_source
    assert "FROM player_snapshots" not in evaluation_source
    # The frozen chain the challenger delegates to is the PE-1 reader.
    assert "historical.historical_player_fixture_rows" in inspect.getsource(
        analytics.completed_rows_as_of
    )
    assert historical.HISTORICAL_OBSERVATION_BOUNDARY_VERSION == "historical_observations_v1.0.0"
