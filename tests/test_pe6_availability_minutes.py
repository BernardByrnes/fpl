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


def _seed(conn, *, pgw_overrides=None, drop_events=(), extra_snapshots=(), placeholder_rows=()):
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
    rows_by_event: dict[int, list[PlayerGameweekRecord]] = {}
    for player_id, per_event in sorted(minutes_table.items()):
        team = player_teams[player_id]
        for event, (minutes, starts) in sorted(per_event.items()):
            if event in drop_events:
                continue
            fixture_id = FIXTURE_BY_EVENT_TEAM.get((event, team))
            if fixture_id is None:
                continue
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
    def insert_snapshot(player_id, captured_at, status, chance, event_context=None):
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
                    raw_json={},
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
    assert artifact["sample"]["player_event_observations"] == 0
    assert artifact["incumbent_metrics"]["brier_p_start"]["status"] == "NO_SAMPLE"
    assert artifact["incumbent_metrics"]["brier_p_start"]["value"] is None
    assert artifact["recommendation"]["token"] == ev.OUTCOME_NO_CHANGE
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


def test_evaluation_refuses_model_selection_on_an_insufficient_sample(tmp_path):
    conn = _db(tmp_path)
    artifact = ev.evaluate_events(conn, [3])
    assert artifact["sample"]["sample_interpretation"] == ev.wfs.SAMPLE_INSUFFICIENT
    assert artifact["sample"]["player_event_observations"] < ev.wfs.MIN_OBSERVATIONS_FOR_DESCRIPTIVE
    for arm, payload in artifact["arms"].items():
        assert payload["outcome"]["token"] == ev.OUTCOME_NO_CHANGE, arm
        assert ev.NO_CHANGE_BASIS_INSUFFICIENT_SAMPLE in payload["outcome"]["basis"]
    assert artifact["recommendation"]["token"] == ev.OUTCOME_NO_CHANGE
    assert artifact["recommendation"]["review_required"] is True
    assert artifact["identity"]["push_or_promotion_performed"] is False
    assert "evidence for senior review" in artifact["identity"]["promotion_note"]
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
    assert statuses.get(ev.STATUS_MULTI_FIXTURE_UNSPECIFIED, 0) > 0
    assert statuses.get(ev.STATUS_BLANK_NO_FIXTURE, 0) > 0
    assert statuses.get(ev.STATUS_OUTCOME_PLACEHOLDER, 0) == 1
    assert statuses.get(ev.STATUS_OUTCOME_MISSING, 0) >= 1
    assert artifact["population"]["scored"] > 0
    for item in artifact["exclusions"]:
        assert item["status"] in ev.EXCLUSION_STATUSES
        assert item["detail"]
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
    assert artifact["sample"]["player_event_observations"] == 0
    assert artifact["population"]["events_excluded"]
    assert artifact["events"][0]["status"] == "NOT_EVALUATED"
    conn.close()


def test_evaluation_population_digest_covers_the_scored_keys(tmp_path):
    conn = _db(tmp_path)
    artifact = ev.evaluate_events(conn, [3])
    cutoff, _reasons = ev.event_cutoff(conn, 3)
    arms = ch.build_challenger_arms(conn, 3, cutoff)
    records, _excluded, _counts, _reasons = ev.build_event_records(conn, 3, cutoff=cutoff, arms=arms)
    keys = [(int(item["event"]), int(item["player_id"])) for item in records]
    assert artifact["population"]["population_digest"] == wf.canonical_population_digest(
        keys, grain=ev.GRAIN_PLAYER_EVENT
    )
    assert artifact["population"]["scored"] == len(keys)
    assert artifact["population"]["coverage_share"] is not None
    conn.close()


def test_evaluation_units_are_declared(tmp_path):
    conn = _db(tmp_path)
    artifact = ev.evaluate_events(conn, [3])
    assert artifact["cutoff_policy"] == "TARGET_EVENT_DEADLINE_TIME"
    assert artifact["grain"] == ev.GRAIN_PLAYER_EVENT
    assert artifact["missing_data_policy_version"] == wf.MISSING_DATA_POLICY_VERSION
    assert artifact["population_rule"]["never_scored_as_zero"]
    assert artifact["population_rule"]["exclusion_statuses"] == list(ev.EXCLUSION_STATUSES)
    comparison = artifact["arms"][ch.ARM_ALL_REFINEMENTS]["comparison"]
    assert "challenger - incumbent" in comparison["delta_convention"]
    assert comparison["metrics"]["expected_minutes_bias"]["status"] in {"OK", "UNDEFINED"}
    assert comparison["references"]["brier_p_start_reference"]["incumbent"]["status"] == "OK"
    assert artifact["recommendation"]["outcome_vocabulary"] == list(ev.OUTCOME_VOCABULARY)
    assert artifact["recommendation"]["primary_criteria"] == list(ev.PRIMARY_CRITERIA)
    assert artifact["recommendation"]["improvement_floors"]["brier"] == ev.DELTA_BRIER_IMPROVEMENT_MIN
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


def test_outcome_rule_is_a_declared_pure_function():
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

    all_better = ev.arm_outcome(block(-0.01, -0.01, -0.5), sample_sufficient=True)
    assert all_better["token"] == ev.OUTCOME_CHALLENGER_ACCEPTED
    partial = ev.arm_outcome(block(-0.01, 0.0, 0.0), sample_sufficient=True)
    assert partial["token"] == ev.OUTCOME_PARTIALLY_ACCEPTED
    noise = ev.arm_outcome(block(0.0005, 0.0005, 0.005), sample_sufficient=True)
    assert noise["token"] == ev.OUTCOME_NO_CHANGE
    assert ev.NO_CHANGE_BASIS_WITHIN_NOISE in noise["basis"]
    worse = ev.arm_outcome(block(0.01, -0.01, -0.5), sample_sufficient=True)
    assert worse["token"] == ev.OUTCOME_NO_CHANGE
    assert ev.NO_CHANGE_BASIS_CHALLENGER_FAILS in worse["basis"]
    insufficient = ev.arm_outcome(block(-0.01, -0.01, -0.5), sample_sufficient=False)
    assert insufficient["token"] == ev.OUTCOME_NO_CHANGE
    assert ev.NO_CHANGE_BASIS_INSUFFICIENT_SAMPLE in insufficient["basis"]
    undefined = ev.arm_outcome(
        {"metrics": {"brier_p_start": {"status": "UNDEFINED"}}}, sample_sufficient=True
    )
    assert undefined["token"] == ev.OUTCOME_NO_CHANGE
    assert ev.NO_CHANGE_BASIS_UNDEFINED in undefined["basis"]


def test_no_second_causal_predicate_exists_in_pe6():
    """PE-6 reuses the frozen boundary; it never restates it."""

    challenger_source = inspect.getsource(ch)
    assert "SELECT" not in challenger_source, "the challenger holds no history predicate"
    assert "analytics.completed_rows_as_of" in challenger_source
    assert "analytics.snapshot_as_of" in challenger_source
    assert "analytics.snapshot_history_as_of" in challenger_source
    evaluation_source = inspect.getsource(ev)
    # The evaluation's only reads are the exact-key outcome row, the event
    # deadline and the fixture kickoffs: never a window predicate.
    assert "FROM player_gameweeks WHERE player_id=? AND fixture_id=?" in evaluation_source
    assert "updated_at" not in evaluation_source
    # The frozen chain the challenger delegates to is the PE-1 reader.
    assert "historical.historical_player_fixture_rows" in inspect.getsource(
        analytics.completed_rows_as_of
    )
    assert historical.HISTORICAL_OBSERVATION_BOUNDARY_VERSION == "historical_observations_v1.0.0"
