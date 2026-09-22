"""PE-7 team attack/defence + player attack refinement — hard tests.

The contract's required hard cases are covered here in order, plus the challenger
identities, the evaluation contract (identical populations, sample-size honesty,
strata sample sizes) and the team -> player coherence layer.  The incumbents are
never mutated: every assertion about a frozen model reads it through its own entry
points, and each challenger is checked *against* it rather than in place of it.

Contract hard-test map
----------------------
1  post-cutoff team xG cannot enter an earlier prediction      -> test_1
2  post-cutoff player xG/xA cannot enter an earlier rate       -> test_2, test_2b, test_2c
3  a later team transfer cannot reattribute a fixture          -> test_3, test_3b
4  a later player transfer cannot alter an earlier prior       -> test_4, test_4b
5  a later position change cannot alter a population/prior     -> test_5
6  a later active/inactive state cannot alter a candidate      -> test_6
7  scheduled placeholders are excluded                         -> test_7
8  missing xG/xA stays missing, never zero                     -> test_8, test_8b
9  a low-history player stays projectable through shrinkage    -> test_9
10 the no-history fallback stays finite and explicit           -> test_10
11 non-xG-bearing prior seasons are not read as zero xG/xA     -> test_11
12 home/away evidence remains causal                           -> test_12
13 role-change evidence is cutoff-safe                         -> test_13, test_13b, test_13c
14 team and player outputs are finite and non-negative         -> test_14
15 team -> player attacking-mass coherence                     -> test_15..15c
16 identical-population incumbent/challenger evaluation        -> test_16, test_16b
17 deterministic repeat / row-order invariance                 -> test_17, test_17b
18 DGW fixture atomicity and blank handling                    -> test_18, test_18b
19 PE-6 minutes identities unchanged                           -> test_19
20 xPts / MC / scoring / bonus / calibration paths unchanged   -> test_20

Beyond the contract: challenger identity and provenance (21, 22), the empty-family
equivalence with the incumbent (23, 24, 25), the incumbent COMPARISON arm's own
cutoff safety and fidelity (2c, 4b, 25b), the derived estimators in both directions,
at their fail-closed edges and on their numerical scale (26, 26b, 27, 27b, 28), the
evaluation artifact's own limits (29-36), the declared disclosure vocabulary firing
(15d, 15e, 37), and a season-opening cutoff with no team evidence (32b).
"""

from __future__ import annotations

import json
import math
import random

import pytest

from fpl_brain import analytics
from fpl_brain import availability_minutes_challenger as identity_resolution
from fpl_brain import bonus_allocation
from fpl_brain import bps_rules
from fpl_brain import joint_minutes
from fpl_brain import minutes_model
from fpl_brain import monte_carlo
from fpl_brain import player_attack_challenger as player_ch
from fpl_brain import player_rates
from fpl_brain import repositories as repo
from fpl_brain import scoring_rules
from fpl_brain import team_attack_challenger as team_ch
from fpl_brain import team_model
from fpl_brain import team_player_attack_coherence as coherence
from fpl_brain import team_player_attack_evaluation as ev
from fpl_brain import walk_forward as wf
from fpl_brain import walk_forward_scoreboard as wf_scoreboard
from fpl_brain.database import connect_database
from fpl_brain.models import (
    EventRecord,
    FixtureRecord,
    PlayerGameweekRecord,
    PlayerRecord,
    PlayerSeasonHistoryRecord,
    PlayerSnapshotRecord,
    PositionRecord,
    TeamRecord,
)

#: The planning event and its official deadline: the cutoff every prediction for
#: event 4 is taken at (the last moment a manager could have acted).
PLANNING_EVENT = 4
CUTOFF = "2026-09-12T12:30:00Z"
#: When the completed-fixture rows were OBSERVED.  Observation time is part of the
#: PE-1 boundary, so it is stated explicitly: after every event-1..3 kickoff and
#: before the cutoff, which is the window the model is allowed to read.
OBSERVED_AT = "2026-09-10T08:00:00Z"
#: The cutoff's accepted official bootstrap generation.
IDENTITY_CAPTURE = "2026-09-01T08:00:00Z"
#: Prior-season rows observed before the season began.
HISTORY_OBSERVED_AT = "2026-07-15T08:00:00Z"
#: When the target event's own rows were written: after the cutoff, because they
#: are outcomes.  They are evaluation evidence and never a model input.
OUTCOME_OBSERVED_AT = "2026-09-15T08:00:00Z"

EVENT_DEADLINES = {
    1: "2026-08-21T17:30:00Z",
    2: "2026-08-28T17:30:00Z",
    3: "2026-09-04T17:30:00Z",
    4: CUTOFF,
}
#: (fixture_id, event, home, away).  Event 4 is a DOUBLE gameweek for team 1 and a
#: BLANK for teams 4, 5 and 6.
FIXTURES = (
    (1, 1, 1, 2),
    (2, 1, 3, 4),
    (3, 1, 5, 6),
    (4, 2, 2, 3),
    (5, 2, 4, 5),
    (6, 2, 6, 1),
    (7, 3, 3, 1),
    (8, 3, 5, 2),
    (9, 3, 6, 4),
    (10, 4, 1, 2),
    (11, 4, 3, 1),
)
FIXTURE_KICKOFF = {
    1: "2026-08-22T14:00:00Z",
    2: "2026-08-22T16:00:00Z",
    3: "2026-08-23T14:00:00Z",
    4: "2026-08-29T14:00:00Z",
    5: "2026-08-29T16:00:00Z",
    6: "2026-08-30T14:00:00Z",
    7: "2026-09-05T14:00:00Z",
    8: "2026-09-05T16:00:00Z",
    9: "2026-09-06T14:00:00Z",
    10: "2026-09-13T14:00:00Z",
    11: "2026-09-14T14:00:00Z",
}
EVENT_FIXTURES = {1: (1, 2, 3), 2: (4, 5, 6), 3: (7, 8, 9), 4: (10, 11)}
TEAMS = (1, 2, 3, 4, 5, 6)
#: One GKP, two DEF, two MID, one FWD per club.
SQUAD_TEMPLATE = (1, 2, 2, 3, 3, 4)
POSITION_WEIGHTS = {1: 0.0, 2: 0.06, 3: 0.22, 4: 0.40}


def _player_id(team: int, index: int) -> int:
    return 100 + team * 10 + index


def _squads() -> dict[int, list[tuple[int, int]]]:
    return {
        team: [(_player_id(team, index + 1), position) for index, position in enumerate(SQUAD_TEMPLATE)]
        for team in TEAMS
    }


ALL_PLAYERS = sorted(player_id for squad in _squads().values() for player_id, _pos in squad)
PLAYER_TEAM = {player_id: team for team, squad in _squads().items() for player_id, _pos in squad}
PLAYER_POSITION = {player_id: position for squad in _squads().values() for player_id, position in squad}
#: A player with a confirmed role change whose signal precedes his latest row, one
#: whose signal arrives after the cutoff, one with no prior-season row, and one who
#: is used for the missing-evidence and no-history cases.
ROLE_CHANGE_PLAYER = _player_id(1, 5)
POST_CUTOFF_SIGNAL_PLAYER = _player_id(1, 6)
NO_PRIOR_PLAYER = _player_id(2, 6)
NO_HISTORY_PLAYER = _player_id(2, 5)
#: The role signal is observed between the player's earlier rows and his latest one,
#: so the segmentation family has something to segment.
ROLE_SIGNAL_OBSERVED_AT = "2026-09-02T08:00:00Z"


def _side_xg(fixture_id: int, team_id: int, was_home: bool) -> float:
    """The designed team-side xG of one fixture side (with a real home advantage)."""

    base = 1.05 + 0.22 * ((fixture_id * 7 + team_id * 3) % 5)
    return round(base + (0.30 if was_home else 0.0), 4)


def _player_xg(fixture_id: int, player_id: int, was_home: bool) -> float:
    """A player's own xG for one fixture: his side's total, split and perturbed.

    The split is by position, so a forward carries more of his side's expectation
    than a defender.  The per-player, per-fixture perturbation is what gives the
    split-half estimators a non-zero sampling variance to measure.
    """

    team = PLAYER_TEAM[player_id]
    total = _side_xg(fixture_id, team, was_home)
    squad = _squads()[team]
    weights = {pid: POSITION_WEIGHTS[pos] for pid, pos in squad}
    weight_total = sum(weights.values())
    share = weights[player_id] / weight_total if weight_total else 0.0
    multiplier = 0.75 + 0.10 * ((fixture_id + player_id) % 4)
    return round(total * share * multiplier, 5)


def _world(
    conn,
    *,
    extra_gameweeks=None,
    extra_gameweek_observed_at=OBSERVED_AT,
    drop_gameweek_players=(),
    drop_gameweek_rows=(),
    row_observed_at=None,
    history_omit=(),
    raise_history=None,
    history_extra=(),
    history_observed_at=HISTORY_OBSERVED_AT,
    notes=None,
    generation_missing=(),
    generation_extra_ids=(),
    after_seed=None,
    shuffle_fixtures=False,
    shuffle_gameweeks=False,
):
    """A six-club league with official xG, a cutoff identity and prior history.

    ``drop_gameweek_players`` removes a player's generated rows (so a case can put
    its own in), ``row_observed_at`` moves individual ``(player, event)`` rows to
    another observation time, and ``after_seed`` runs inside the same transaction —
    which is how the mutation tests simulate a post-cutoff write (a transfer, a
    position change, an activation change) without rebuilding the world.
    """

    squads = _squads()
    fixtures = [
        {"id": fixture_id, "event": event, "h": home, "a": away}
        for fixture_id, event, home, away in FIXTURES
    ]
    if shuffle_fixtures:
        random.Random(7).shuffle(fixtures)
    rows: list[PlayerGameweekRecord] = []
    for fixture in fixtures:
        for team_id, was_home in ((fixture["h"], True), (fixture["a"], False)):
            for player_id, _position in squads[team_id]:
                if player_id in drop_gameweek_players:
                    continue
                if (player_id, fixture["id"]) in set(drop_gameweek_rows):
                    continue
                xg = _player_xg(fixture["id"], player_id, was_home)
                rows.append(
                    PlayerGameweekRecord(
                        player_id=player_id,
                        event=fixture["event"],
                        fixture_id=fixture["id"],
                        was_home=1 if was_home else 0,
                        minutes=90,
                        starts=1,
                        total_points=2,
                        goals_scored=0,
                        assists=0,
                        clean_sheets=0,
                        goals_conceded=1,
                        saves=0,
                        bonus=0,
                        bps=10,
                        expected_goals=xg,
                        expected_assists=round(xg * 0.35, 5),
                        expected_goals_conceded=_side_xg(fixture["id"], team_id, not was_home),
                        source="element_summary",
                        raw_json={},
                    )
                )
    if shuffle_gameweeks:
        random.Random(11).shuffle(rows)
    buckets: dict[str, list[PlayerGameweekRecord]] = {}
    for row in rows:
        default = (
            OUTCOME_OBSERVED_AT if int(row.event) == PLANNING_EVENT else OBSERVED_AT
        )
        key = str((row_observed_at or {}).get((row.player_id, row.event), default))
        buckets.setdefault(key, []).append(row)
    for row in extra_gameweeks or ():
        buckets.setdefault(str(extra_gameweek_observed_at), []).append(row)

    if notes is None:
        notes = (
            (ROLE_CHANGE_PLAYER, "role_change", "confirmed", ROLE_SIGNAL_OBSERVED_AT),
            (POST_CUTOFF_SIGNAL_PLAYER, "role_change", "confirmed", "2026-09-14T08:00:00Z"),
        )

    with conn:
        repo.upsert_teams(conn, [TeamRecord(id=team, name=f"Team {team}") for team in TEAMS])
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
                for team, squad in sorted(squads.items())
                for player_id, position in squad
            ],
        )
        repo.upsert_events(
            conn,
            [
                EventRecord(
                    id=event,
                    finished=1,
                    data_checked=1,
                    deadline_time=EVENT_DEADLINES[event],
                    raw_json={},
                )
                for event in sorted(EVENT_DEADLINES)
            ],
        )
        repo.upsert_fixtures(
            conn,
            [
                FixtureRecord(
                    id=fixture["id"],
                    event=fixture["event"],
                    team_h=fixture["h"],
                    team_a=fixture["a"],
                    finished=1,
                    started=1,
                    team_h_score=1,
                    team_a_score=1,
                    kickoff_time=FIXTURE_KICKOFF[fixture["id"]],
                    raw_json={},
                )
                for fixture in fixtures
            ],
        )
        for observed_at in sorted(buckets):
            repo.upsert_player_gameweeks(conn, buckets[observed_at], observed_at)
        run_id = repo.create_fetch_run(conn, "fetch_fpl")
        ids = sorted({*ALL_PLAYERS, *generation_extra_ids})
        repo.insert_snapshots(
            conn,
            [
                PlayerSnapshotRecord(
                    player_id=player_id,
                    captured_at=IDENTITY_CAPTURE,
                    now_cost=50,
                    raw_json={
                        "team": PLAYER_TEAM.get(player_id),
                        "element_type": PLAYER_POSITION.get(player_id),
                    },
                )
                for player_id in ids
                if player_id not in set(generation_missing) and player_id in PLAYER_TEAM
            ],
            run_id,
        )
        repo.record_bootstrap_generation(
            conn,
            captured_at=IDENTITY_CAPTURE,
            accepted=True,
            official_element_count=len(ids),
            parsed_count=len(ids),
            persisted_count=len(ids),
            element_ids=ids,
            element_ids_sha256=repo.element_ids_identity(ids)[1],
            acceptance_rule="pe7-test",
            acceptance_rule_version="BOOTSTRAP_GENERATION_ACCEPTANCE v1",
            fetch_run_id=run_id,
        )
        histories = [
            PlayerSeasonHistoryRecord(
                player_id=player_id,
                season_name="2025/26",
                minutes=1500 + (player_id % 401),
                starts=17,
                raw_json={
                    "expected_goals": 3.0 + (player_id % 7) * 0.4,
                    "expected_assists": 1.4 + (player_id % 5) * 0.3,
                },
            )
            for player_id in ALL_PLAYERS
            if player_id not in set(history_omit)
        ]
        histories.extend(history_extra)
        if raise_history is not None:
            histories.append(raise_history)
        if histories:
            repo.upsert_player_season_histories(conn, histories, history_observed_at)
        for player_id, key, value, observed in notes:
            repo.insert_scouting_note(
                conn,
                {
                    "player_id": player_id,
                    "key": key,
                    "value_text": value,
                    "observed_at": observed,
                    "source": "world",
                    "observation": "seeded",
                },
            )
        if after_seed is not None:
            after_seed(conn)


def _pool(conn, cutoff=CUTOFF):
    resolution = identity_resolution.resolve_candidate_pool(conn, cutoff)
    assert resolution["identity_available"] is True
    return resolution


def _identities(resolution) -> dict[int, tuple[int, int]]:
    return {
        int(player["player_id"]): (int(player["team_id"]), int(player["element_type"]))
        for player in resolution["players"]
    }


def _rows_by_key(rows) -> dict:
    return {(int(row["fixture_id"]), int(row["team_id"])): row for row in rows}


def _player_rows_by_key(rows) -> dict:
    return {(int(row["player_id"]), str(row["component"])): row for row in rows}


#: A row for an ALREADY COMPLETED fixture, of the shape a post-cutoff refresh
#: writes: the official endpoint re-sends a finished match when it is corrected.
LATE_OBSERVED_AT = "2026-09-13T08:00:00Z"


def _corrected_team_row(player_id: int, event: int, fixture_id: int, was_home: int):
    return PlayerGameweekRecord(
        player_id=player_id,
        event=event,
        fixture_id=fixture_id,
        was_home=was_home,
        minutes=90,
        starts=1,
        total_points=2,
        goals_scored=1,
        assists=0,
        clean_sheets=0,
        goals_conceded=1,
        saves=0,
        bonus=0,
        bps=9,
        expected_goals=4.5,
        expected_assists=3.5,
        source="element_summary",
        raw_json={},
    )


def _challenger_player_rows(conn, resolution, families=None):
    kwargs = {} if families is None else {"families": families}
    return _player_rows_by_key(
        player_ch.build_challenger_player_rate_projections(
            conn,
            PLANNING_EVENT,
            CUTOFF,
            players=resolution["players"],
            identities=_identities(resolution),
            **kwargs,
        )
    )


# ===========================================================================
# 1. post-cutoff team xG evidence cannot change an earlier team prediction
# ===========================================================================


def test_1_post_cutoff_team_xg_evidence_cannot_enter_an_earlier_team_prediction():
    """A row the cutoff could not see never becomes evidence AT the cutoff.

    Two directions, both stated because the store admits only one of them:

    * a post-cutoff observation of a fixture the window held NO evidence for
      cannot create evidence at the cutoff — the row is refused outright;
    * a post-cutoff REFRESH of a row the window did hold does lose that row: the
      boundary reads a row's own observation time, and a store that keeps one row
      per (player, fixture) cannot show what the earlier version said.  The two
      consequences are (a) the window can only shrink, never gain, and (b) the
      refreshed fixture is treated exactly as if the row had never been observed
      at the cutoff.  Admitting the post-cutoff CONTENT is what the boundary
      exists to prevent, and it never happens.
    """

    # NO_PRIOR_PLAYER is a team-2 player, so fixture 1 (team 1 at home) is his away match.
    missing_row_key = (NO_PRIOR_PLAYER, 1)
    corrected_row = lambda: _corrected_team_row(NO_PRIOR_PLAYER, 1, 1, 0)  # noqa: E731
    baseline_conn = connect_database(":memory:")
    _world(baseline_conn)
    baseline = _rows_by_key(
        team_ch.build_challenger_team_projections(baseline_conn, PLANNING_EVENT, CUTOFF)[0]
    )

    # (a) the window had no evidence for this player+fixture: nothing can appear.
    without_row_conn = connect_database(":memory:")
    _world(without_row_conn, drop_gameweek_rows=(missing_row_key,))
    without_row = _rows_by_key(
        team_ch.build_challenger_team_projections(without_row_conn, PLANNING_EVENT, CUTOFF)[0]
    )
    control_conn = connect_database(":memory:")
    _world(
        control_conn,
        drop_gameweek_rows=(missing_row_key,),
        extra_gameweeks=(corrected_row(),),
        extra_gameweek_observed_at=OBSERVED_AT,
    )
    control = _rows_by_key(
        team_ch.build_challenger_team_projections(control_conn, PLANNING_EVENT, CUTOFF)[0]
    )
    late_conn = connect_database(":memory:")
    _world(
        late_conn,
        drop_gameweek_rows=(missing_row_key,),
        extra_gameweeks=(corrected_row(),),
        extra_gameweek_observed_at=LATE_OBSERVED_AT,
    )
    late = _rows_by_key(
        team_ch.build_challenger_team_projections(late_conn, PLANNING_EVENT, CUTOFF)[0]
    )
    assert any(
        control[key]["expected_goals_for"] != without_row[key]["expected_goals_for"]
        for key in without_row
    ), "the control row must be able to move the model, or this test proves nothing"
    for key in sorted(without_row):
        assert late[key]["expected_goals_for"] == without_row[key]["expected_goals_for"]
        assert late[key]["expected_goals_against"] == without_row[key]["expected_goals_against"]

    # (b) a post-cutoff REFRESH of a row the window held: treated as if the row had
    # never been observed at the cutoff, and never as post-cutoff content.
    refreshed_conn = connect_database(":memory:")
    _world(
        refreshed_conn,
        extra_gameweeks=(corrected_row(),),
        extra_gameweek_observed_at=LATE_OBSERVED_AT,
    )
    refreshed = _rows_by_key(
        team_ch.build_challenger_team_projections(refreshed_conn, PLANNING_EVENT, CUTOFF)[0]
    )
    assert any(
        refreshed[key]["expected_goals_for"] != baseline[key]["expected_goals_for"]
        for key in baseline
    ), "the row was observable at the cutoff: losing it must be visible, not silent"
    for key in sorted(without_row):
        assert (
            refreshed[key]["expected_goals_for"] == without_row[key]["expected_goals_for"]
        ), "a post-cutoff refresh is invisible, never admitted as post-cutoff content"


# ===========================================================================
# 2. post-cutoff player xG/xA evidence cannot change an earlier player-rate read
# ===========================================================================


def test_2_post_cutoff_player_xg_evidence_cannot_enter_an_earlier_player_rate():
    """The same two directions as test 1, at player-rate grain."""

    # NO_HISTORY_PLAYER is a team-2 player, so fixture 4 (team 2 at home) is his home match.
    missing_row_key = (NO_HISTORY_PLAYER, 4)
    corrected_row = lambda: _corrected_team_row(NO_HISTORY_PLAYER, 2, 4, 1)  # noqa: E731
    baseline_conn = connect_database(":memory:")
    _world(baseline_conn)
    baseline = _challenger_player_rows(baseline_conn, _pool(baseline_conn))

    without_row_conn = connect_database(":memory:")
    _world(without_row_conn, drop_gameweek_rows=(missing_row_key,))
    without_row = _challenger_player_rows(without_row_conn, _pool(without_row_conn))
    control_conn = connect_database(":memory:")
    _world(
        control_conn,
        drop_gameweek_rows=(missing_row_key,),
        extra_gameweeks=(corrected_row(),),
        extra_gameweek_observed_at=OBSERVED_AT,
    )
    control = _challenger_player_rows(control_conn, _pool(control_conn))
    late_conn = connect_database(":memory:")
    _world(
        late_conn,
        drop_gameweek_rows=(missing_row_key,),
        extra_gameweeks=(corrected_row(),),
        extra_gameweek_observed_at=LATE_OBSERVED_AT,
    )
    late = _challenger_player_rows(late_conn, _pool(late_conn))

    key = (NO_HISTORY_PLAYER, player_rates.COMPONENT_XG)
    assert control[key]["current_total"] != without_row[key]["current_total"]
    assert late[key]["current_total"] == without_row[key]["current_total"]
    assert late[key]["current_played_rows"] == without_row[key]["current_played_rows"]
    assert late[key]["posterior_mean"] == without_row[key]["posterior_mean"]

    refreshed_conn = connect_database(":memory:")
    _world(
        refreshed_conn,
        extra_gameweeks=(corrected_row(),),
        extra_gameweek_observed_at=LATE_OBSERVED_AT,
    )
    refreshed = _challenger_player_rows(refreshed_conn, _pool(refreshed_conn))
    for key in sorted(without_row):
        assert refreshed[key]["current_total"] == without_row[key]["current_total"]
        assert refreshed[key]["posterior_mean"] == without_row[key]["posterior_mean"]
        assert (
            refreshed[key]["posterior_mean"] != baseline[key]["posterior_mean"]
            or without_row[key]["posterior_mean"] == baseline[key]["posterior_mean"]
        )


def test_2b_post_cutoff_prior_season_write_cannot_enter_an_earlier_player_rate():
    """A prior-season read is as-of too, and the store's own overwrite is stated.

    ``player_season_histories`` keeps ONE row per (player, season) with its own
    observation time, so the two directions are the same two as test 1/2:

    * a prior-season row written after the cutoff — a new season row, or the official
      endpoint re-sending one — is refused, so it can never enter an earlier read;
    * a post-cutoff refresh of a row the cutoff could see loses that row: it is
      treated exactly as if the row had never been observed at the cutoff, which is
      what "not admitted" means on a store with no row history.
    """

    key = (ROLE_CHANGE_PLAYER, player_rates.COMPONENT_XG)
    baseline_conn = connect_database(":memory:")
    _world(baseline_conn)
    baseline = _challenger_player_rows(baseline_conn, _pool(baseline_conn))
    assert baseline[key]["prior_source"] == "prev_season_same_player"

    # (a) a NEW row for a season the player has no row for, written after the cutoff.
    new_row_conn = connect_database(":memory:")
    _world(new_row_conn)
    with new_row_conn:
        repo.upsert_player_season_histories(
            new_row_conn,
            [
                PlayerSeasonHistoryRecord(
                    player_id=ROLE_CHANGE_PLAYER,
                    season_name="2024/25",
                    minutes=2400,
                    raw_json={"expected_goals": 30.0, "expected_assists": 20.0},
                )
            ],
            LATE_OBSERVED_AT,
        )
    with_new_row = _challenger_player_rows(new_row_conn, _pool(new_row_conn))
    assert with_new_row[key]["prior_mean"] == baseline[key]["prior_mean"]
    assert with_new_row[key]["posterior_mean"] == baseline[key]["posterior_mean"]
    assert player_ch.FLAG_POST_CUTOFF_HISTORY_EXCLUDED in with_new_row[key]["risk_flags"]
    assert with_new_row[key]["exposure_detail"]["history_rows_excluded_post_cutoff"] == 1

    # (b) a post-cutoff REFRESH of the row the cutoff did see: invisible, never
    # admitted — the read matches a store where that row was never observable.
    refreshed_conn = connect_database(":memory:")
    _world(refreshed_conn)
    with refreshed_conn:
        refreshed_conn.execute(
            "UPDATE player_season_histories SET observed_at=? WHERE player_id=? AND season_name='2025/26'",
            (LATE_OBSERVED_AT, ROLE_CHANGE_PLAYER),
        )
    refreshed = _challenger_player_rows(refreshed_conn, _pool(refreshed_conn))

    absent_conn = connect_database(":memory:")
    _world(absent_conn)
    with absent_conn:
        absent_conn.execute(
            "DELETE FROM player_season_histories WHERE player_id=? AND season_name='2025/26'",
            (ROLE_CHANGE_PLAYER,),
        )
    absent = _challenger_player_rows(absent_conn, _pool(absent_conn))

    assert refreshed[key]["prior_source"] != "prev_season_same_player"
    assert refreshed[key]["prior_mean"] == absent[key]["prior_mean"]
    assert refreshed[key]["posterior_mean"] == absent[key]["posterior_mean"]
    assert refreshed[key]["posterior_mean"] != baseline[key]["posterior_mean"]


def test_2c_a_post_cutoff_season_write_cannot_move_the_incumbent_comparison_arm():
    """The arm a challenger is scored against reads the cutoff as well.

    The baseline of every PE-7 delta is the incumbent's own prediction, so a
    comparison arm built by reaching straight for ``player_rates.player_prior``
    would leak here: that read carries no observation time at all, so the row below
    would enter the earlier arm and move the baseline — corrupting every comparison
    without touching a single challenger number.

    The control is the SAME row observed BEFORE the cutoff, where it must move the
    arm; without it this test would pass on an arm that ignored prior-season
    evidence entirely.
    """

    def arms(conn):
        resolution = _pool(conn)
        return player_ch.build_challenger_player_arms(
            conn,
            PLANNING_EVENT,
            CUTOFF,
            players=resolution["players"],
            identities=_identities(resolution),
        )

    key = (NO_PRIOR_PLAYER, player_rates.COMPONENT_XG)
    extra_row = PlayerSeasonHistoryRecord(
        player_id=NO_PRIOR_PLAYER,
        season_name="2024/25",
        minutes=2100,
        raw_json={"expected_goals": 12.0, "expected_assists": 6.0},
    )

    baseline_conn = connect_database(":memory:")
    _world(baseline_conn, history_omit=(NO_PRIOR_PLAYER,))
    baseline = arms(baseline_conn)
    assert baseline.incumbent_rows[key]["prior_source"] == "position_pooled"

    # (a) the same row, written AFTER the cutoff: invisible to the arm.
    late_conn = connect_database(":memory:")
    _world(late_conn, history_omit=(NO_PRIOR_PLAYER,))
    with late_conn:
        repo.upsert_player_season_histories(late_conn, [extra_row], LATE_OBSERVED_AT)
    late = arms(late_conn)
    assert late.incumbent_rows[key]["prior_source"] == "position_pooled"
    assert (
        late.incumbent_rows[key]["posterior_mean"]
        == baseline.incumbent_rows[key]["posterior_mean"]
    )
    assert (
        late.incumbent_rows[key]["prior_ess"] == baseline.incumbent_rows[key]["prior_ess"]
    )
    assert player_ch.FLAG_POST_CUTOFF_HISTORY_EXCLUDED in late.incumbent_rows[key]["risk_flags"]
    assert late.incumbent_arm["history_rows_excluded_post_cutoff"] == 1
    assert late.incumbent_arm["construction"] == player_ch.INCUMBENT_ARM_CONSTRUCTION
    for arm in late.arm_names():
        assert late.row(arm, key)["posterior_mean"] == baseline.row(arm, key)["posterior_mean"]

    # The frozen incumbent's OWN read does take the row: that is exactly why the
    # comparison arm may not be built through it.
    published_late = _player_rows_by_key(
        player_rates.build_player_rate_projections(late_conn, PLANNING_EVENT, CUTOFF)
    )
    assert published_late[key]["prior_source"] == "multi_season_same_player"
    assert published_late[key]["prior_mean"] != late.incumbent_rows[key]["prior_mean"]

    # (b) control: the same row, observable at the cutoff, DOES move both the arm and
    # every challenger, so the assertion above is not vacuous.
    early_conn = connect_database(":memory:")
    _world(early_conn, history_omit=(NO_PRIOR_PLAYER,), history_extra=(extra_row,))
    early = arms(early_conn)
    assert early.incumbent_rows[key]["prior_source"] == "multi_season_same_player"
    assert (
        early.incumbent_rows[key]["posterior_mean"] != baseline.incumbent_rows[key]["posterior_mean"]
    )
    for arm in early.arm_names():
        assert early.row(arm, key)["posterior_mean"] != baseline.row(arm, key)["posterior_mean"]


# ===========================================================================
# 3. a later team transfer cannot reattribute a historical fixture
# ===========================================================================


def test_3_a_later_team_transfer_cannot_reattribute_a_historical_fixture():
    conn = connect_database(":memory:")
    _world(conn)
    before = _rows_by_key(
        team_ch.build_challenger_team_projections(conn, PLANNING_EVENT, CUTOFF)[0]
    )
    with conn:
        conn.execute("UPDATE players SET team_id=6 WHERE id=?", (ROLE_CHANGE_PLAYER,))
    after = _rows_by_key(
        team_ch.build_challenger_team_projections(conn, PLANNING_EVENT, CUTOFF)[0]
    )
    assert sorted(before) == sorted(after)
    for key in sorted(before):
        assert before[key]["expected_goals_for"] == after[key]["expected_goals_for"]
        assert before[key]["attack_rating"] == after[key]["attack_rating"]


def test_3b_fixture_side_comes_from_was_home_not_from_the_current_club():
    conn = connect_database(":memory:")
    _world(conn)
    fixture_id, _event, home, away = FIXTURES[6]  # fixture 7: team 3 home, team 1 away
    expected = {
        "home": round(sum(_player_xg(fixture_id, pid, True) for pid, _pos in _squads()[home]), 6),
        "away": round(sum(_player_xg(fixture_id, pid, False) for pid, _pos in _squads()[away]), 6),
    }
    with conn:
        for player_id, _pos in _squads()[away]:
            conn.execute("UPDATE players SET team_id=? WHERE id=?", (home, player_id))
    sides = team_model.fixture_side_xg(conn, fixture_id, as_of=CUTOFF, planning_event=PLANNING_EVENT)
    assert float(sides["home_xg"]) == pytest.approx(expected["home"], abs=1e-6)
    assert float(sides["away_xg"]) == pytest.approx(expected["away"], abs=1e-6)


# ===========================================================================
# 4. a later player transfer cannot alter an earlier player-rate prior
# ===========================================================================


def test_4_a_later_player_transfer_cannot_alter_an_earlier_player_rate_prior():
    conn = connect_database(":memory:")
    _world(conn)
    resolution = _pool(conn)
    before = _challenger_player_rows(conn, resolution)
    with conn:
        conn.execute("UPDATE players SET team_id=5 WHERE id=?", (ROLE_CHANGE_PLAYER,))
        conn.execute("UPDATE players SET team_id=5 WHERE id=?", (NO_HISTORY_PLAYER,))
    after = _challenger_player_rows(conn, resolution)
    assert sorted(before) == sorted(after)
    for key in sorted(before):
        assert before[key]["posterior_mean"] == after[key]["posterior_mean"]
        assert before[key]["prior_mean"] == after[key]["prior_mean"]


def test_4b_a_later_identity_change_cannot_move_the_incumbent_comparison_arm():
    """The same five mutations, measured on the ARM the delta is taken against.

    A mutation that moved only the incumbent arm would corrupt every PE-7
    comparison while leaving every challenger number untouched, so the guard has to
    be stated on that arm specifically.  The control is the incumbent's own
    persisted-row read (``pooled_rates``): it DOES move — that is its frozen
    behaviour and the reason this arm is built by an adapter.
    """

    key = (NO_PRIOR_PLAYER, player_rates.COMPONENT_XG)

    def arms(conn, resolution):
        return player_ch.build_challenger_player_arms(
            conn,
            PLANNING_EVENT,
            CUTOFF,
            players=resolution["players"],
            identities=_identities(resolution),
        )

    conn = connect_database(":memory:")
    _world(conn, history_omit=(NO_PRIOR_PLAYER,))
    resolution = _pool(conn)
    before = arms(conn, resolution)
    published_before = _player_rows_by_key(
        player_rates.build_player_rate_projections(conn, PLANNING_EVENT, CUTOFF)
    )
    assert before.incumbent_rows[key]["prior_source"] == "position_pooled"

    # A transfer AND a position change, both persisted after the cutoff.  The
    # position write lands in the pooled bucket the mutating player's own prior rows
    # sit in, so the incumbent's persisted-row pooling really does move.
    with conn:
        conn.execute("UPDATE players SET team_id=5 WHERE id=?", (ROLE_CHANGE_PLAYER,))
        conn.execute("UPDATE players SET element_type=4 WHERE id=?", (ROLE_CHANGE_PLAYER,))
    after = arms(conn, resolution)
    published_after = _player_rows_by_key(
        player_rates.build_player_rate_projections(conn, PLANNING_EVENT, CUTOFF)
    )
    assert (
        published_before[key]["prior_mean"] != published_after[key]["prior_mean"]
    ), "the persisted-row read must move, or this test proves nothing"
    assert sorted(before.incumbent_rows) == sorted(after.incumbent_rows)
    for row_key in sorted(before.incumbent_rows):
        assert before.incumbent_rows[row_key]["posterior_mean"] == after.incumbent_rows[row_key]["posterior_mean"]
        assert before.incumbent_rows[row_key]["prior_mean"] == after.incumbent_rows[row_key]["prior_mean"]
        assert before.incumbent_rows[row_key]["prior_ess"] == after.incumbent_rows[row_key]["prior_ess"]
    for arm in before.arm_names():
        assert before.arm_keys(arm) == after.arm_keys(arm)
        for row_key in before.arm_keys(arm):
            assert before.row(arm, row_key)["posterior_mean"] == after.row(arm, row_key)["posterior_mean"]

    # And the activation write: the arm keeps projecting the candidate the cutoff's
    # accepted generation placed in the pool.
    with conn:
        conn.execute("UPDATE players SET is_active=0 WHERE id=?", (NO_PRIOR_PLAYER,))
    deactivated = arms(conn, resolution)
    assert sorted(deactivated.incumbent_rows) == sorted(before.incumbent_rows)
    for row_key in sorted(before.incumbent_rows):
        assert (
            deactivated.incumbent_rows[row_key]["posterior_mean"]
            == before.incumbent_rows[row_key]["posterior_mean"]
        )
    assert key in deactivated.incumbent_rows


# ===========================================================================
# 5. a later position change cannot alter an earlier population/prior
# ===========================================================================


def test_5_a_later_position_change_cannot_alter_an_earlier_population_or_prior():
    conn = connect_database(":memory:")
    _world(conn, history_omit=(NO_PRIOR_PLAYER,))
    resolution = _pool(conn)
    assert _identities(resolution)[NO_PRIOR_PLAYER][1] == PLAYER_POSITION[NO_PRIOR_PLAYER]
    before = _challenger_player_rows(conn, resolution)

    # The mutated position belongs to a player who DOES carry prior-season rows, so
    # the incumbent's persisted-row pools really move; the candidate read below is
    # what must not.
    with conn:
        conn.execute("UPDATE players SET element_type=1 WHERE id=?", (ROLE_CHANGE_PLAYER,))
    after_resolution = _pool(conn)
    after = _challenger_player_rows(conn, after_resolution)
    key = (NO_PRIOR_PLAYER, player_rates.COMPONENT_XG)
    assert before[key]["prior_source"] == after[key]["prior_source"] == "position_pooled"
    assert before[key]["prior_mean"] == after[key]["prior_mean"]
    assert before[key]["posterior_mean"] == after[key]["posterior_mean"]
    assert _identities(after_resolution)[NO_PRIOR_PLAYER][1] == PLAYER_POSITION[NO_PRIOR_PLAYER]

    # The incumbent's own pools DO move with the persisted row: that is its frozen
    # behaviour, and exactly why this challenger builds its own from the cutoff.
    incumbent_after = player_rates.pooled_rates(conn, player_rates.PlayerRatesConfig())
    with conn:
        conn.execute("UPDATE players SET element_type=2 WHERE id=?", (ROLE_CHANGE_PLAYER,))
    incumbent_moved = player_rates.pooled_rates(conn, player_rates.PlayerRatesConfig())
    assert incumbent_after["position"] != incumbent_moved["position"]
    # The cutoff-stable pools ignore the same write: every pooled row is attributed
    # by the identity the CUTOFF resolves, so the pools are identical.
    pools_before, _disclosure = player_ch.cutoff_stable_rate_pools(conn, _identities(resolution), cutoff=CUTOFF)
    pools_after, _disclosure2 = player_ch.cutoff_stable_rate_pools(conn, _identities(after_resolution), cutoff=CUTOFF)
    assert pools_after == pools_before


# ===========================================================================
# 6. a later active/inactive state cannot alter an earlier candidate
# ===========================================================================


def test_6_a_later_active_inactive_state_cannot_alter_an_earlier_candidate():
    conn = connect_database(":memory:")
    _world(conn)
    before = _pool(conn)
    before_keys = sorted(int(player["player_id"]) for player in before["players"])
    with conn:
        conn.execute("UPDATE players SET is_active=0 WHERE id=?", (ROLE_CHANGE_PLAYER,))
    after = _pool(conn)
    assert sorted(int(player["player_id"]) for player in after["players"]) == before_keys
    identity = next(
        player for player in after["players"] if int(player["player_id"]) == ROLE_CHANGE_PLAYER
    )
    assert identity["identity_cutoff_safe"] is True
    # The cutoff placed him in the official pool; the persisted pool (active only)
    # no longer lists him at all.  The disagreement is recorded and the CUTOFF wins.
    assert identity["identity"]["live_row_disagreement"] == [
        "membership:in_pool->absent_from_persisted_pool"
    ]
    assert after["summary"]["cutoff_safe_rows"] == len(after["players"])
    # The persisted pool is read only to REPORT the divergence, never to decide it.
    assert after["summary"]["persisted_pool_divergence"] is not None


# ===========================================================================
# 7. scheduled placeholders are excluded
# ===========================================================================


def test_7_scheduled_placeholders_are_excluded():
    placeholder = PlayerGameweekRecord(
        player_id=NO_HISTORY_PLAYER,
        event=3,
        fixture_id=9,
        was_home=0,
        minutes=0,
        source="element_summary",
        raw_json={},
    )
    conn = connect_database(":memory:")
    _world(conn, extra_gameweeks=(placeholder,))
    raw_rows = conn.execute(
        "SELECT COUNT(*) FROM player_gameweeks pg JOIN fixtures f ON f.id=pg.fixture_id "
        "WHERE pg.player_id=? AND f.finished=1 AND f.started=1",
        (NO_HISTORY_PLAYER,),
    ).fetchone()[0]
    resolution = _pool(conn)
    rows = _challenger_player_rows(conn, resolution)
    key = (NO_HISTORY_PLAYER, player_rates.COMPONENT_XG)
    # The placeholder is stored on a finished fixture, so a fixture-joined read
    # would take it: the canonical boundary refuses it before this model sees it.
    # Four real rows for this player (three completed fixtures before the cutoff plus
    # his target-event outcome) and the placeholder: the fixture-joined read sees it,
    # the canonical boundary does not.
    assert raw_rows == 5
    assert rows[key]["current_played_rows"] == 3
    assert all(
        fixture_id != 9
        for fixture_id in rows[key]["exposure_detail"]["current_fixtures"]
    )
    from fpl_brain import historical_observations as historical

    observed_ids = {
        int(row["fixture_id"])
        for row in historical.historical_player_fixtures(
            conn,
            as_of=OBSERVED_AT,
            planning_event=PLANNING_EVENT,
            player_ids=[NO_HISTORY_PLAYER],
        )
    }
    assert 9 not in observed_ids


# ===========================================================================
# 8. missing xG/xA remains missing, not zero
# ===========================================================================


def test_8_missing_xg_remains_missing_and_is_never_counted_as_zero():
    # A PLAYED row (90 minutes, real points) with no xG value at all.
    missing_row = PlayerGameweekRecord(
        player_id=NO_HISTORY_PLAYER,
        event=3,
        fixture_id=8,
        was_home=0,
        minutes=90,
        starts=1,
        total_points=1,
        goals_scored=0,
        assists=0,
        clean_sheets=0,
        goals_conceded=2,
        saves=0,
        bonus=0,
        bps=5,
        expected_goals=None,
        expected_assists=None,
        source="element_summary",
        raw_json={},
    )
    conn = connect_database(":memory:")
    _world(conn, drop_gameweek_rows=((NO_HISTORY_PLAYER, 8),), extra_gameweeks=(missing_row,))
    resolution = _pool(conn)
    rows = _challenger_player_rows(conn, resolution)
    xg_row = rows[(NO_HISTORY_PLAYER, player_rates.COMPONENT_XG)]
    xa_row = rows[(NO_HISTORY_PLAYER, player_rates.COMPONENT_XA)]
    # The two rows that carry a value contribute exposure; the third is a GAP, and
    # 90 minutes of invented zero xG would be a fabrication, not a measurement.
    assert xg_row["current_played_rows"] == 2
    assert xa_row["current_played_rows"] == 2
    assert xg_row["exposure_detail"]["missing_value_rows"] == 1
    assert xa_row["exposure_detail"]["missing_value_rows"] == 1
    assert "CURRENT_XG_DATA_GAP" in xg_row["risk_flags"]
    assert float(xg_row["current_minutes"]) == pytest.approx(180.0)
    assert float(xa_row["current_minutes"]) == pytest.approx(180.0)
    assert xg_row["current_rate"] is not None and xg_row["current_rate"] > 0.0
    # The exposure the model reports is exactly the sum of the two real rows, so no
    # zero had to be invented to make the arithmetic work.
    assert float(xg_row["current_total"]) == pytest.approx(
        _player_xg(1, NO_HISTORY_PLAYER, False) + _player_xg(4, NO_HISTORY_PLAYER, True), abs=1e-6
    )


def test_8b_unplayed_target_fixture_rows_are_not_scored():
    conn = connect_database(":memory:")
    _world(conn)
    with conn:
        conn.execute("UPDATE fixtures SET finished=0, started=0 WHERE id=10")
    artifact = ev.evaluate_events(conn, [PLANNING_EVENT])
    # Fixture 10 is not a played match, so its sides are excluded WITH A STATUS
    # rather than scored as a zero, and fixture 11 still scores.
    team_excluded = artifact["events"][0]["excluded_by_status"]["team"]
    assert team_excluded.get(ev.STATUS_FIXTURE_NOT_PLAYED) == 2
    assert artifact["population"]["team"]["scored"] == 2
    # Fixture 11 is still scored, on both sides, and the accounting still closes.
    assert artifact["population"]["team"]["candidate_slots"] == 4
    assert artifact["population"]["team"]["accounting"]["reconciles"] is True


# ===========================================================================
# 9. a low-history player remains structurally projectable through shrinkage
# ===========================================================================


def test_9_a_low_history_player_remains_structurally_projectable():
    conn = connect_database(":memory:")
    _world(
        conn,
        history_omit=(NO_HISTORY_PLAYER,),
        drop_gameweek_players=(NO_HISTORY_PLAYER,),
    )
    with conn:
        repo.upsert_player_gameweeks(
            conn,
            [
                PlayerGameweekRecord(
                    player_id=NO_HISTORY_PLAYER,
                    event=2,
                    fixture_id=4,
                    was_home=0,
                    minutes=23,
                    starts=0,
                    total_points=1,
                    goals_scored=0,
                    assists=0,
                    clean_sheets=0,
                    goals_conceded=1,
                    saves=0,
                    bonus=0,
                    bps=3,
                    expected_goals=0.25,
                    expected_assists=0.1,
                    source="element_summary",
                    raw_json={},
                )
            ],
            OBSERVED_AT,
        )
    resolution = _pool(conn)
    rows = _challenger_player_rows(conn, resolution)
    row = rows[(NO_HISTORY_PLAYER, player_rates.COMPONENT_XG)]
    assert row["prior_source"] == "position_pooled"
    assert float(row["prior_ess"]) > 0.0
    assert float(row["current_minutes"]) == pytest.approx(23.0)
    assert row["posterior_mean"] is not None
    assert row["posterior_mean"] > 0.0
    # The posterior is a genuine blend: the player's own exposure has weight and so
    # does the declared prior, so a thin sample is shrunk rather than dropped.
    assert 0.0 < row["posterior_uncertainty"]["prior_share"] < 1.0
    assert 0.0 < row["posterior_uncertainty"]["current_share"] < 1.0
    assert "NO_HISTORICAL_PLAYER_PRIOR" in row["risk_flags"]


# ===========================================================================
# 10. the no-history fallback remains finite and explicit
# ===========================================================================


def test_10_the_no_history_fallback_is_finite_and_explicit():
    conn = connect_database(":memory:")
    _world(conn, history_omit=(NO_HISTORY_PLAYER,), drop_gameweek_players=(NO_HISTORY_PLAYER,))
    resolution = _pool(conn)
    rows = _challenger_player_rows(conn, resolution)
    row = rows[(NO_HISTORY_PLAYER, player_rates.COMPONENT_XG)]
    # No same-player history and no current exposure: the fallback is the pooled
    # prior, and its provenance says so.
    assert row["prior_source"] == "position_pooled"
    assert row["posterior_mean"] is not None
    assert float(row["posterior_mean"]) == pytest.approx(float(row["prior_mean"]))
    assert row["posterior_mean"] >= 0.0
    assert "NO_CURRENT_EVIDENCE" in row["risk_flags"]
    assert any("pooled prior used" in gap for gap in row["data_gaps"])

    # With nothing at all in the store the fallback is None plus an explicit gap:
    # a missing prior is missing, never a fabricated zero.
    empty = connect_database(":memory:")
    bare = player_ch.challenger_player_prior(
        NO_HISTORY_PLAYER,
        3,
        player_rates.COMPONENT_XG,
        player_ch.PlayerAttackChallengerConfig().resolved(player_rates.PlayerRatesConfig()),
        {"league": {player_rates.COMPONENT_XG: {"rate": None, "minutes": 0.0, "total": 0.0}}, "position": {}},
        {"seasons": {}, "excluded": {"post_cutoff_rows": 0, "undated_rows": 0}, "rows_visible": []},
    )
    assert bare["prior_rate"] is None
    assert "LEAGUE_POOLED_PRIOR" in bare["flags"]
    assert empty.execute("SELECT 1").fetchone() is not None


# ===========================================================================
# 11. previous-season non-xG-bearing history is not interpreted as zero
# ===========================================================================


def test_11_non_xg_bearing_previous_season_is_not_read_as_zero_xg():
    conn = connect_database(":memory:")
    _world(
        conn,
        history_omit=(NO_PRIOR_PLAYER,),
        # The official endpoint returns "0.00" for the whole xG family on these
        # seasons; that is MISSING evidence, not a genuine zero.
        history_extra=(
            PlayerSeasonHistoryRecord(
                player_id=NO_PRIOR_PLAYER,
                season_name="2021/22",
                minutes=2100,
                starts=24,
                raw_json={"expected_goals": 0.0, "expected_assists": 0.0},
            ),
        ),
    )
    resolution = _pool(conn)
    rows = _challenger_player_rows(conn, resolution)
    row = rows[(NO_PRIOR_PLAYER, player_rates.COMPONENT_XG)]
    assert row["prior_source"] != "prev_season_same_player"
    assert row["prior_source"] != "multi_season_same_player"
    assert row["prior_source"] == "position_pooled"
    assert float(row["prior_mean"]) > 0.0
    assert float(row["prior_minutes"]) > 0.0


# ===========================================================================
# 12. home/away evidence remains causal
# ===========================================================================


def test_12_home_away_evidence_remains_causal():
    conn = connect_database(":memory:")
    _world(conn)
    params = team_ch.fit_challenger_team_strength(conn, PLANNING_EVENT, CUTOFF)
    # Only fixtures STRICTLY before the planning event can contribute, whatever
    # their stored finality says: the target event's own played fixtures are
    # outcomes, not evidence about the environment that produced them.
    pre_event_rows = team_model.team_match_rows(conn, PLANNING_EVENT, CUTOFF)
    assert params["match_count"] == len({row["fixture_id"] for row in pre_event_rows}) == 9
    assert all(int(row["event"]) < PLANNING_EVENT for row in pre_event_rows)
    assert all(int(row["fixture_id"]) not in EVENT_FIXTURES[PLANNING_EVENT] for row in pre_event_rows)

    # And the venue split is measured on those rows only: dropping the target
    # event's fixtures from the store cannot move the fitted parameters.
    with conn:
        conn.execute("DELETE FROM player_gameweeks WHERE event=?", (PLANNING_EVENT,))
        conn.execute("DELETE FROM fixtures WHERE event=?", (PLANNING_EVENT,))
    after = team_ch.fit_challenger_team_strength(conn, PLANNING_EVENT, CUTOFF)
    assert after["home_advantage"] == params["home_advantage"]
    assert after["league_log_baseline"] == params["league_log_baseline"]
    assert after["attack"] == params["attack"]


# ===========================================================================
# 13. role-change evidence is cutoff-safe
# ===========================================================================


def test_13_role_change_evidence_is_cutoff_safe():
    conn = connect_database(":memory:")
    _world(
        conn,
        row_observed_at={
            (ROLE_CHANGE_PLAYER, 1): "2026-08-25T08:00:00Z",
            (ROLE_CHANGE_PLAYER, 2): "2026-08-31T08:00:00Z",
            (ROLE_CHANGE_PLAYER, 3): OBSERVED_AT,
        },
    )
    resolution = _pool(conn)
    rows = _challenger_player_rows(conn, resolution)
    changed = rows[(ROLE_CHANGE_PLAYER, player_rates.COMPONENT_XG)]
    # The signal was observed at the cutoff, so it was eligible, and its own
    # observation time is the boundary the player's exposure is segmented at.
    assert changed["role_modifier_applied"] is True
    assert changed["role_segmentation"]["boundary_observed_at"] == ROLE_SIGNAL_OBSERVED_AT
    assert changed["role_segmentation"]["applied"] is True
    assert changed["exposure_detail"]["superseded_fixtures"] == [1, 6]
    assert player_ch.FLAG_ROLE_SEGMENTED_EXPOSURE in changed["risk_flags"]
    assert changed["exposure_detail"]["superseded_rate"] is not None

    # A signal observed AFTER the cutoff proves nothing about that cutoff: the
    # player's own role discount stays untouched and his exposure is not segmented.
    later = rows[(POST_CUTOFF_SIGNAL_PLAYER, player_rates.COMPONENT_XG)]
    assert later["role_modifier_applied"] is False
    assert later["role_segmentation"]["applied"] is False
    assert later["role_segmentation"]["boundary_observed_at"] is None


def test_13c_a_post_cutoff_role_note_cannot_move_either_arm():
    """The role evidence is an as-of read on BOTH sides of the comparison.

    The incumbent's own eligibility rule is called by both arms, so a note written
    after the cutoff has to be refused for the incumbent arm as well: an arm that
    discounted its prior on a post-cutoff note would have been projected on evidence
    the cutoff could not see.  The control is the same note observed BEFORE the
    cutoff, which must move both.
    """

    key = (ROLE_CHANGE_PLAYER, player_rates.COMPONENT_XG)

    def arms(conn):
        resolution = _pool(conn)
        return player_ch.build_challenger_player_arms(
            conn,
            PLANNING_EVENT,
            CUTOFF,
            players=resolution["players"],
            identities=_identities(resolution),
        )

    none_conn = connect_database(":memory:")
    _world(none_conn, notes=())
    none = arms(none_conn)

    late_conn = connect_database(":memory:")
    _world(
        late_conn,
        notes=(
            (ROLE_CHANGE_PLAYER, "role_change", "confirmed", "2026-09-14T08:00:00Z"),
        ),
    )
    late = arms(late_conn)
    assert late.incumbent_rows[key]["prior_ess"] == none.incumbent_rows[key]["prior_ess"]
    assert late.incumbent_rows[key]["posterior_mean"] == none.incumbent_rows[key]["posterior_mean"]
    assert late.incumbent_rows[key]["role_modifier_applied"] is False
    for arm in late.arm_names():
        assert late.row(arm, key)["posterior_mean"] == none.row(arm, key)["posterior_mean"]

    early_conn = connect_database(":memory:")
    _world(
        early_conn,
        notes=((ROLE_CHANGE_PLAYER, "role_change", "confirmed", ROLE_SIGNAL_OBSERVED_AT),),
    )
    early = arms(early_conn)
    assert early.incumbent_rows[key]["role_modifier_applied"] is True
    assert early.incumbent_rows[key]["prior_ess"] < none.incumbent_rows[key]["prior_ess"]
    assert early.incumbent_rows[key]["posterior_mean"] != none.incumbent_rows[key]["posterior_mean"]


def test_13b_role_signal_never_moves_the_rate_of_a_player_without_one():
    conn = connect_database(":memory:")
    _world(conn)
    resolution = _pool(conn)
    rows = _challenger_player_rows(conn, resolution)
    untouched = rows[(NO_PRIOR_PLAYER, player_rates.COMPONENT_XG)]
    assert untouched["role_modifier_applied"] is False
    assert untouched["role_segmentation"]["applied"] is False
    assert untouched["prior_mean_alignment"]["applied"] is False


# ===========================================================================
# 14. team and player outputs are finite and non-negative
# ===========================================================================


def test_14_team_and_player_outputs_are_finite_and_non_negative():
    conn = connect_database(":memory:")
    _world(conn)
    team_rows, _meta = team_ch.build_challenger_team_projections(conn, PLANNING_EVENT, CUTOFF)
    incumbent_team_rows, _meta2 = team_model.build_team_fixture_projections(
        conn, PLANNING_EVENT, CUTOFF
    )
    resolution = _pool(conn)
    player_rows = list(
        player_ch.build_challenger_player_rate_projections(
            conn,
            PLANNING_EVENT,
            CUTOFF,
            players=resolution["players"],
            identities=_identities(resolution),
        )
    )
    incumbent_player_rows = player_rates.build_player_rate_projections(
        conn, PLANNING_EVENT, CUTOFF
    )
    team_fields = ("expected_goals_for", "expected_goals_against", "p_goals_0", "p_clean_sheet")
    for rows in (team_rows, incumbent_team_rows):
        assert coherence.validate_attack_expectations(rows, team_fields) == []
    for rows in (player_rows, incumbent_player_rows):
        assert rows
        assert [row for row in rows if row["posterior_mean"] is None] == []
        assert [
            row
            for row in rows
            if not math.isfinite(float(row["posterior_mean"])) or float(row["posterior_mean"]) < 0.0
        ] == []
    # The incumbent's own readiness gate accepts the challenger's rows too: the
    # challenger publishes the same field contract, not a parallel one.
    gate = team_model.readiness_summary(None, team_rows)
    assert gate["status"] in {"PASS", "WARN"}
    rate_gate = player_rates.readiness_summary(None, player_rows)
    assert rate_gate["status"] in {"PASS", "WARN"}


# ===========================================================================
# 15. team -> player attacking-mass coherence
# ===========================================================================


def test_15_team_to_player_attacking_mass_coherence():
    conn = connect_database(":memory:")
    _world(conn)
    resolution = ev.resolve_event(conn, PLANNING_EVENT)
    team_arms = team_ch.build_challenger_arms(conn, PLANNING_EVENT, CUTOFF)
    player_arms = player_ch.build_challenger_player_arms(
        conn,
        PLANNING_EVENT,
        CUTOFF,
        players=resolution["players"],
        identities=_identities(resolution),
    )
    block = ev.coherence_for_event(conn, resolution, team_arms, player_arms)
    for arm, summary in block["summary"].items():
        checks = summary["checks"]
        assert checks["team_expectations_finite_and_non_negative"] is True
        assert checks["player_expectations_finite_and_non_negative"] is True
        assert checks["unallocated_mass_is_explicit"] is True
        assert checks["no_invented_fixture_attack"] is True
        assert checks["fixture_atomic"] is True
        assert checks["redistribution"] == coherence.REDISTRIBUTION_NONE
        # Where the player level DOES exceed the environment, the block says which
        # fixture sides it happened on: exceeding is reported, never silent.
        if checks["no_component_exceeds_the_team_environment"] is False:
            assert summary["exceeds_environment"], "an exceeded environment must be named"
            assert all(
                any(
                    "PLAYER_MASS_EXCEEDS_TEAM_ENVIRONMENT" in flag
                    for flag in record["flags"]
                )
                for block_entry in summary["exceeds_environment"]
                for record in block["arms"][arm]["records"]
                if record["status"] == coherence.STATUS_EXCEEDS
            )
    # The pairings are declared data, not an assumption: the frozen pair, the
    # headline pair, and every ablation against the other family's incumbent.
    assert "incumbent" in block["summary"]
    assert team_ch.ARM_ALL_REFINEMENTS in block["summary"]
    assert len(block["pairings"]) == 1 + 1 + (len(team_arms.arm_names()) - 1) + (
        len(player_arms.arm_names()) - 1
    )
    for pairing in block["pairings"]:
        entry = block["arms"][pairing["id"]]
        assert entry["pairing"]["team_arm"] == pairing["team_arm"]
        assert entry["pairing"]["player_arm"] == pairing["player_arm"]

    detailed = block["arms"][team_ch.ARM_ALL_REFINEMENTS]
    assert detailed["records"]
    # The mass is real: a block that measured nothing would "pass" every check
    # vacuously, so the total is asserted before the checks are believed.
    assert sum(record["allocated"]["xG_per90"] for record in detailed["records"]) > 0.0
    assert sum(record["players_contributing"]["xG_per90"] for record in detailed["records"]) >= 4
    for record in detailed["records"]:
        for component in coherence.ATTACK_COMPONENTS:
            residual = record["residual"][component]
            allocated = record["allocated"][component]
            expectation = record["team_expectation"]
            assert residual == pytest.approx(expectation - allocated, abs=1e-6)
            assert record["residual_share"][component] is not None
        assert record["redistribution"] == coherence.REDISTRIBUTION_NONE


def test_15b_an_excess_is_reported_and_never_clamped_or_redistributed():
    team_rows = [
        {"fixture_id": 1, "event": 4, "team_id": 1, "venue": "home", "expected_goals_for": 1.0},
        {"fixture_id": 1, "event": 4, "team_id": 2, "venue": "away", "expected_goals_for": 2.0},
    ]
    fixtures = {1: {"id": 1, "team_h": 1, "team_a": 2}}
    mass = [
        {"fixture_id": 1, "team_id": 1, "component": "xG_per90", "mass": 1.8, "players_contributing": 3},
        {"fixture_id": 1, "team_id": 1, "component": "xA_per90", "mass": 0.2, "players_contributing": 2},
    ]
    block = coherence.attacking_mass_allocation(team_rows, mass, fixtures=fixtures)
    assert block["checks"]["no_component_exceeds_the_team_environment"] is False
    assert block["status"] == coherence.STATUS_EXCEEDS
    assert {"fixture_id": 1, "team_id": 1} in block["exceeds_environment"]
    record = next(item for item in block["records"] if item["team_id"] == 1)
    # The mass is reported exactly as it came in: not clamped to the environment,
    # and not rescaled onto the other component.
    assert record["allocated"]["xG_per90"] == 1.8
    assert record["residual"]["xG_per90"] == pytest.approx(-0.8)
    assert record["redistribution"] == coherence.REDISTRIBUTION_NONE
    other = next(item for item in block["records"] if item["team_id"] == 2)
    assert other["allocated"]["xG_per90"] == 0.0
    assert other["status"] == coherence.STATUS_NO_EXPOSURE


def test_15c_an_excess_in_the_artifact_is_reported_and_never_clamped():
    """An excess is a reported fact, not a failure to hide.

    The contract forbids player mass from SILENTLY exceeding the team environment.
    Where it does exceed it, this artifact names the fixture side, keeps the mass
    exactly as the model produced it, and never rescales it onto another player or
    another component.
    """

    conn = connect_database(":memory:")
    _world(conn)
    artifact = ev.evaluate_events(conn, [PLANNING_EVENT])
    pairings = artifact["coherence"]["per_event"][str(PLANNING_EVENT)]["arms"]
    for pairing_id, entry in artifact["coherence"]["checks"]["arms"].items():
        # A reported excess is allowed to be present; nothing ELSE may be dirty.
        unexpected = [
            item
            for item in entry["dirty"]
            if not item.startswith("no_component_exceeds_the_team_environment@")
        ]
        assert unexpected == [], (pairing_id, unexpected)
        for block in entry["exceeds_environment"]:
            assert block["records"], "an excess must name the fixture sides it applies to"
        # And the mass is exactly what the arm's own rows produce.
        detail = pairings[pairing_id]
        for record in detail["records"]:
            assert record["redistribution"] == coherence.REDISTRIBUTION_NONE
            assert record["residual"]["xG_per90"] == pytest.approx(
                record["team_expectation"] - record["allocated"]["xG_per90"], abs=1e-6
            )
    assert artifact["coherence"]["checks"]["arms"]["incumbent"]["events"] == [PLANNING_EVENT]


def test_15d_the_declared_disclosure_vocabulary_actually_fires():
    """Every declared token names a condition that can occur.

    A vocabulary a reader cannot trigger is worse than no vocabulary: it implies a
    check that never runs.  The coherence module's flags and its coherent-status
    token are exercised here on purpose-built inputs, and the mass is asserted to
    come back exactly as it went in — a negative expectation is REPORTED, never
    clamped, exactly like an excess.
    """

    fixtures = {1: {"id": 1, "team_h": 1, "team_a": 2}}
    team_rows = [
        {"fixture_id": 1, "event": 4, "team_id": 1, "venue": "home", "expected_goals_for": 1.5},
        {"fixture_id": 1, "event": 4, "team_id": 2, "venue": "away", "expected_goals_for": 1.0},
        {"fixture_id": 99, "event": 4, "team_id": 7, "venue": "home", "expected_goals_for": 1.0},
    ]
    block = coherence.attacking_mass_allocation(
        team_rows,
        [
            {
                "fixture_id": 1,
                "team_id": 1,
                "component": coherence.COMPONENT_XG,
                "mass": -0.4,
                "players_contributing": 2,
            }
        ],
        fixtures=fixtures,
    )
    assert block["status"] == coherence.STATUS_INVALID
    assert coherence.FLAG_PLAYER_EXPECTATION_INVALID in block["flags"]
    player_record = next(item for item in block["records"] if item["team_id"] == 1)
    assert coherence.FLAG_PLAYER_EXPECTATION_INVALID in player_record["flags"]
    assert player_record["allocated"][coherence.COMPONENT_XG] == -0.4
    assert [item["value"] for item in block["violations"] if item["kind"] == "player_mass"] == [-0.4]

    # A side the fixture list does not contain is an invented fixture attack, named
    # on the record AND on the block.
    assert coherence.FLAG_INVENTED_FIXTURE_ATTACK in block["flags"]
    assert block["checks"]["no_invented_fixture_attack"] is False
    invented_record = next(item for item in block["records"] if item["fixture_id"] == 99)
    assert coherence.FLAG_INVENTED_FIXTURE_ATTACK in invented_record["flags"]

    # A clean block reports the DECLARED coherent token rather than a bare literal,
    # and carries no flag at all.
    clean = coherence.attacking_mass_allocation(
        team_rows[:2],
        [
            {
                "fixture_id": 1,
                "team_id": 2,
                "component": coherence.COMPONENT_XG,
                "mass": 0.25,
                "players_contributing": 1,
            }
        ],
        fixtures=fixtures,
    )
    assert clean["status"] == coherence.STATUS_COHERENT
    assert clean["flags"] == []

    # An invalid team expectation is a violation, and its record keeps every flag it
    # accumulated rather than only the one that fired last.
    invalid_team = coherence.attacking_mass_allocation(
        [{"fixture_id": 99, "event": 4, "team_id": 7, "venue": "home", "expected_goals_for": None}],
        [],
        fixtures=fixtures,
    )
    assert invalid_team["status"] == coherence.STATUS_INVALID
    assert coherence.FLAG_TEAM_EXPECTATION_INVALID in invalid_team["flags"]
    assert invalid_team["records"][0]["flags"] == sorted(
        {coherence.FLAG_INVENTED_FIXTURE_ATTACK, coherence.FLAG_TEAM_EXPECTATION_INVALID}
    )


def test_15e_a_mass_that_is_not_a_number_has_no_magnitude_and_is_not_a_zero():
    """A missing or unparseable player mass contributes nothing and says so.

    ``None`` is not a zero: it is an unusable value, so it is refused with a
    violation and its own flag rather than silently entering the sum as 0.0.
    """

    fixtures = {1: {"id": 1, "team_h": 1, "team_a": 2}}
    team_rows = [{"fixture_id": 1, "event": 4, "team_id": 1, "venue": "home", "expected_goals_for": 1.5}]
    for unusable in (None, "not-a-number"):
        block = coherence.attacking_mass_allocation(
            team_rows,
            [
                {
                    "fixture_id": 1,
                    "team_id": 1,
                    "component": coherence.COMPONENT_XG,
                    "mass": unusable,
                    "players_contributing": 1,
                }
            ],
            fixtures=fixtures,
        )
        assert block["status"] == coherence.STATUS_INVALID
        # The violation publishes the value EXACTLY as it arrived, so an unusable
        # mass is visible rather than replaced by the zero it contributed.
        assert [item["value"] for item in block["violations"]] == [unusable]
        assert coherence.FLAG_PLAYER_EXPECTATION_INVALID in block["flags"]


# ===========================================================================
# 16. identical-population incumbent/challenger evaluation
# ===========================================================================


def test_16_incumbent_and_challenger_are_scored_on_identical_populations():
    conn = connect_database(":memory:")
    _world(conn)
    resolution = identity_resolution.resolve_candidate_pool(conn, CUTOFF)
    team_arms = team_ch.build_challenger_arms(conn, PLANNING_EVENT, CUTOFF)
    team_arms.verify_same_population()
    assert set(team_arms.arm_names()) == set(team_ch.default_arm_definitions())
    for arm in team_arms.arm_names():
        assert team_arms.arm_keys(arm) == team_arms.keys()

    player_arms = player_ch.build_challenger_player_arms(
        conn,
        PLANNING_EVENT,
        CUTOFF,
        players=resolution["players"],
        identities=_identities(resolution),
    )
    player_arms.verify_same_population()
    for arm in player_arms.arm_names():
        assert player_arms.arm_keys(arm) == player_arms.keys()

    artifact = ev.evaluate_events(conn, [PLANNING_EVENT])
    assert artifact["population"]["same_population"] is True
    assert artifact["population"]["team"]["population_digest"]
    assert artifact["population"]["player"]["population_digest"]
    assert artifact["population"]["team"]["scored"] == artifact["population"]["team"][
        "accounting"
    ]["scored_rows"]
    assert artifact["population"]["player"]["accounting"]["reconciles"] is True


def test_16b_every_arm_covers_every_scored_record():
    conn = connect_database(":memory:")
    _world(conn)
    artifact = ev.evaluate_events(conn, [PLANNING_EVENT])
    team_arms = set(artifact["population"]["team"]["arms_scored"])
    player_arms = set(artifact["population"]["player"]["arms_scored"])
    for family, names in (("team", team_arms), ("player", player_arms)):
        for arm in names:
            payload = artifact[family]["arms"][arm]
            assert payload["metrics"]["n"] == payload["incumbent_metrics"]["n"]
    # The measurement is drawn under the declared rule, and the sample-size policy
    # is PE-2's, carried by token rather than re-derived.
    for section in (artifact["team"]["arms"], artifact["player"]["arms"]):
        for payload in section.values():
            assert payload["measurement"]["token"] in ev.MEASUREMENT_VOCABULARY


# ===========================================================================
# 17. deterministic repeat / row-order invariance
# ===========================================================================


def test_17_repeat_builds_are_deterministic():
    conn = connect_database(":memory:")
    _world(conn)
    first = ev.evaluate_events(conn, [PLANNING_EVENT])
    second = ev.evaluate_events(conn, [PLANNING_EVENT])
    assert first["artifact_digest"] == second["artifact_digest"]
    assert (
        first["population"]["team"]["population_digest"]
        == second["population"]["team"]["population_digest"]
    )
    assert (
        first["population"]["player"]["population_digest"]
        == second["population"]["player"]["population_digest"]
    )


def test_17b_row_order_cannot_change_the_artifact():
    forward = connect_database(":memory:")
    _world(forward)
    artifacts = [ev.evaluate_events(forward, [PLANNING_EVENT])]

    shuffled = connect_database(":memory:")
    _world(shuffled, shuffle_fixtures=True, shuffle_gameweeks=True)
    artifacts.append(ev.evaluate_events(shuffled, [PLANNING_EVENT]))

    digests = {artifact["artifact_digest"] for artifact in artifacts}
    assert len(digests) == 1
    # The coherence block has its own canonical digest over sorted records, and it is
    # the same digest in every build: the allocation does not depend on row order.
    per_build = [
        {
            (event, pairing): block["digest"]
            for event, event_block in artifact["coherence"]["per_event"].items()
            for pairing, block in event_block["arms"].items()
        }
        for artifact in artifacts
    ]
    assert len(per_build) >= 2
    assert all(entry == per_build[0] for entry in per_build)
    assert all(value.startswith("sha256:") for value in per_build[0].values())


# ===========================================================================
# 18. DGW fixture atomicity and blank handling
# ===========================================================================


def test_18_double_gameweek_stays_fixture_atomic_and_blanks_invent_nothing():
    conn = connect_database(":memory:")
    _world(conn)
    artifact = ev.evaluate_events(conn, [PLANNING_EVENT])
    # The target event's two fixtures each contribute their own sides, and team 1's
    # double gameweek is TWO records rather than one collapsed number.
    team_keys = {
        (record["fixture_id"], record["team_id"])
        for block in artifact["coherence"]["per_event"].values()
        for record in block["arms"][team_ch.ARM_ALL_REFINEMENTS]["records"]
    }
    assert (10, 1) in team_keys and (11, 1) in team_keys
    assert sum(1 for key in team_keys if key[1] == 1) == 2

    event_block = artifact["coherence"]["per_event"][str(PLANNING_EVENT)]["arms"][
        team_ch.ARM_ALL_REFINEMENTS
    ]
    aggregate = event_block["event_aggregates"][str(PLANNING_EVENT)]
    assert aggregate["atomic_records"] == 4
    assert aggregate["fixture_atomic"] is True
    assert aggregate["fixtures"] == [10, 11]
    team_one = aggregate["teams"]["1"]
    assert team_one["fixtures"] == [10, 11]
    assert "AGGREGATE" in aggregate["note"]

    # Teams 4, 5 and 6 are blank in the event: no fixture, so no fixture attack.
    for team in (4, 5, 6):
        assert str(team) not in aggregate["teams"]
    assert not any(key[1] in {4, 5, 6} for key in team_keys)


def test_18b_blank_team_candidates_are_counted_and_never_projected():
    conn = connect_database(":memory:")
    _world(conn)
    resolution = identity_resolution.resolve_candidate_pool(conn, CUTOFF)
    event_block = ev.build_event_records(conn, PLANNING_EVENT, ev.resolve_event(conn, PLANNING_EVENT))
    blank_players = {player_id for player_id, team in PLAYER_TEAM.items() if team in (4, 5, 6)}
    excluded = {
        int(item["player_id"])
        for item in event_block["player_excluded"]
        if item["status"] == ev.STATUS_BLANK_NO_FIXTURE
    }
    assert excluded == blank_players
    assert all(
        int(record["player_id"]) not in blank_players for record in event_block["player_records"]
    )
    assert all(
        int(record["team_id"]) not in {4, 5, 6} for record in event_block["team_records"]
    )


# ===========================================================================
# 19. PE-6 minutes identities unchanged
# ===========================================================================


def test_19_pe6_minutes_identities_are_unchanged():
    identity = ev.frozen_incumbent_identity()
    assert identity["unchanged"] is True
    assert identity["mismatches"] == {}
    assert minutes_model.MINUTES_MODEL_VERSION == "minutes_v1.8.0"
    assert minutes_model.MINUTES_COHERENT_MODEL_VERSION == "minutes_v1.2.0"
    assert joint_minutes.JOINT_MINUTES_MODEL_VERSION == "minutes_v1.5.2"
    # The PE-7 evaluation reads no minutes projection at all: the player side is
    # scored on REALISED exposure, so a minutes change cannot move a PE-7 number.
    assert identity["values"]["minutes_model.MINUTES_MODEL_VERSION"] == "minutes_v1.8.0"
    assert identity["values"]["minutes_model.MINUTES_COHERENT_MODEL_VERSION"] == "minutes_v1.2.0"
    assert identity["values"]["joint_minutes.JOINT_MINUTES_MODEL_VERSION"] == "minutes_v1.5.2"
    assert identity["pe6_minutes_trio"] == [
        "minutes_model.MINUTES_MODEL_VERSION",
        "minutes_model.MINUTES_COHERENT_MODEL_VERSION",
        "joint_minutes.JOINT_MINUTES_MODEL_VERSION",
    ]
    assert all(name in identity["values"] for name in identity["pe6_minutes_trio"])


# ===========================================================================
# 20. downstream identities unchanged (xPts / MC / scoring / bonus / calibration)
# ===========================================================================


def test_20_downstream_paths_are_unchanged():
    identity = ev.frozen_incumbent_identity()
    assert identity["values"]["xpts.XPTS_MODEL_VERSION"] == "xpts_v1.4.1"
    assert identity["values"]["monte_carlo.MONTE_CARLO_MODEL_VERSION"] == "mc_v1.3.0"
    assert identity["values"]["team_model.TEAM_MODEL_VERSION"] == "team_strength_v1.1.0"
    assert identity["values"]["team_model.TEAM_BASELINE_MODEL_VERSION"] == "team_naive_v1.1.0"
    assert identity["values"]["player_rates.PLAYER_RATE_MODEL_VERSION"] == "player_rates_v1.0.0"
    assert (
        identity["values"]["player_rates.PLAYER_RATE_BASELINE_MODEL_VERSION"]
        == "player_rate_baseline_v1.0.0"
    )
    # The scoring / BPS / bonus allocation identities are not PE-7's to move.
    assert scoring_rules.SCORING_RULES_VERSION == "fpl_scoring_2026_27_v1.0.0"
    assert bps_rules.BPS_RULES_VERSION == "fpl_bps_2026_27_v1.0.0"
    assert bonus_allocation.BONUS_ALLOCATOR_VERSION == "fpl_bonus_allocation_v1.0.0"
    # And the evaluation computes no integrated xPts diagnostic at all: PE-8 owns
    # calibration and PE-7 may not promote on a downstream xPts sample.
    conn = connect_database(":memory:")
    _world(conn)
    artifact = ev.evaluate_events(conn, [PLANNING_EVENT])
    assert artifact["xpts_diagnostic"]["status"] == ev.XPTS_DIAGNOSTIC_STATUS
    assert "PE-8" in artifact["xpts_diagnostic"]["reason"]
    assert artifact["identity"]["promotion"] == "NOT_PERFORMED_THIS_ARTIFACT_MEASURES_ONLY"


# ===========================================================================
# Challenger identity, provenance and the empty-family equivalence
# ===========================================================================


def test_21_the_team_challenger_carries_its_own_identity_and_provenance():
    identity = team_ch.challenger_identity()
    assert identity["challenger_version"] == "team_attack_challenger_v0.1.0"
    assert identity["incumbent_identity"]["unchanged"] is True
    assert identity["promotion"] == "NOT_PERFORMED_CHALLENGER_ONLY"
    assert set(identity["refinement_families"]) == set(team_ch.REFINEMENT_FAMILIES)
    assert (
        team_ch.challenger_identity(team_ch.TeamAttackChallengerConfig(iterations=3))[
            "challenger_config_hash"
        ]
        != identity["challenger_config_hash"]
    )
    # The incumbent's version identifiers are read, never written: the challenger
    # publishes its own and leaves theirs alone.
    assert team_model.TEAM_MODEL_VERSION == "team_strength_v1.1.0"
    assert team_model.TEAM_BASELINE_MODEL_VERSION == "team_naive_v1.1.0"

    conn = connect_database(":memory:")
    _world(conn)
    rows, meta = team_ch.build_challenger_team_projections(conn, PLANNING_EVENT, CUTOFF)
    assert meta["model_version"] == team_ch.TEAM_ATTACK_CHALLENGER_VERSION
    assert all(row["model_version"] == team_ch.TEAM_ATTACK_CHALLENGER_VERSION for row in rows)
    for row in rows:
        assert row["provenance"]["challenger_families"] == sorted(team_ch.REFINEMENT_FAMILIES)
        assert row["provenance"]["promotion"] == "NOT_PERFORMED_CHALLENGER_ONLY"
        assert row["provenance"]["side_attribution"] == "player_gameweeks.was_home"


def test_22_the_player_challenger_carries_its_own_identity_and_penalty_discipline():
    identity = player_ch.challenger_identity()
    assert identity["challenger_version"] == "player_attack_challenger_v0.1.0"
    assert identity["incumbent_identity"]["unchanged"] is True
    assert identity["penalties"] == "embedded_in_xG"
    assert identity["npxg_separation"] is False
    assert identity["promotion"] == "NOT_PERFORMED_CHALLENGER_ONLY"
    assert player_rates.PLAYER_RATE_MODEL_VERSION == "player_rates_v1.0.0"

    conn = connect_database(":memory:")
    _world(conn)
    rows = _challenger_player_rows(conn, _pool(conn))
    assert rows
    for row in rows.values():
        provenance = row["provenance"]
        assert provenance["penalties"] == "embedded_in_xG"
        assert provenance["npxg_separation"] is False
        assert "NPxG" in provenance["penalty_note"]
        assert provenance["position_basis"] == player_ch.POSITION_BASIS
        assert row["model_version"] == player_ch.PLAYER_ATTACK_CHALLENGER_VERSION
        # The published component names stay xG/xA: no penalty split is fabricated.
        assert row["component"] in (player_rates.COMPONENT_XG, player_rates.COMPONENT_XA)


def test_23_the_team_challenger_with_no_families_reproduces_the_incumbent_exactly():
    conn = connect_database(":memory:")
    _world(conn)
    incumbent_params = team_model.fit_team_strength(conn, PLANNING_EVENT, CUTOFF)
    challenger_params = team_ch.fit_challenger_team_strength(
        conn, PLANNING_EVENT, CUTOFF, families=frozenset()
    )
    for field in (
        "league_log_baseline",
        "home_advantage",
        "attack",
        "defence",
        "team_match_counts",
        "team_weight_totals",
        "home_match_weight",
        "match_count",
        "total_team_weight",
    ):
        assert challenger_params[field] == incumbent_params[field], field

    incumbent_rows = _rows_by_key(
        team_model.build_team_fixture_projections(conn, PLANNING_EVENT, CUTOFF)[0]
    )
    challenger_rows = _rows_by_key(
        team_ch.build_challenger_team_projections(
            conn, PLANNING_EVENT, CUTOFF, families=frozenset()
        )[0]
    )
    assert sorted(incumbent_rows) == sorted(challenger_rows)
    for key in sorted(incumbent_rows):
        for field in (
            "expected_goals_for",
            "expected_goals_against",
            "p_goals_0",
            "p_goals_1",
            "p_goals_2_plus",
            "p_clean_sheet",
            "attack_rating",
            "opponent_defence_rating",
            "current_evidence_weight",
            "current_evidence_share",
            "risk_flags",
        ):
            assert incumbent_rows[key][field] == challenger_rows[key][field], (key, field)


def test_24_the_player_challenger_with_no_families_reproduces_the_incumbent_on_a_clean_store():
    conn = connect_database(":memory:")
    _world(conn)
    resolution = _pool(conn)
    pools, disclosure = player_ch.cutoff_stable_rate_pools(
        conn, _identities(resolution), cutoff=CUTOFF
    )
    assert disclosure["history_rows_written_after_the_cutoff"] == 0
    assert disclosure["history_rows_without_an_observation_time"] == 0

    challenger = _player_rows_by_key(
        player_ch.build_challenger_player_rate_projections(
            conn,
            PLANNING_EVENT,
            CUTOFF,
            players=resolution["players"],
            identities=_identities(resolution),
            families=frozenset(),
            pools=pools,
        )
    )
    incumbent = _player_rows_by_key(
        player_ch.build_challenger_player_arms(
            conn,
            PLANNING_EVENT,
            CUTOFF,
            players=resolution["players"],
            identities=_identities(resolution),
        ).incumbent_rows.values()
    )
    assert sorted(challenger) == sorted(incumbent)
    for key in sorted(challenger):
        for field in (
            "prior_mean",
            "prior_source",
            "prior_ess_before_discounts",
            "prior_minutes",
            "current_minutes",
            "current_total",
            "current_rate",
            "posterior_mean",
            "posterior_ess",
            "club_portability_discount",
        ):
            assert challenger[key][field] == incumbent[key][field], (key, field)


def test_25_the_incumbent_arm_is_the_incumbents_own_projection_code():
    conn = connect_database(":memory:")
    _world(conn)
    resolution = _pool(conn)
    arm = player_ch.build_challenger_player_arms(
        conn,
        PLANNING_EVENT,
        CUTOFF,
        players=resolution["players"],
        identities=_identities(resolution),
    )
    pool_ids = {int(player["player_id"]) for player in resolution["players"]}
    published = {
        (int(row["player_id"]), str(row["component"])): row
        for row in player_rates.build_player_rate_projections(conn, PLANNING_EVENT, CUTOFF)
        if int(row["player_id"]) in pool_ids
    }
    assert set(published) == set(arm.incumbent_rows)
    for key in sorted(published):
        assert published[key]["posterior_mean"] == arm.incumbent_rows[key]["posterior_mean"]
        assert published[key]["prior_source"] == arm.incumbent_rows[key]["prior_source"]


def test_25b_the_incumbent_arm_is_the_incumbents_own_numbers_field_by_field():
    """The adapter is the incumbent, not a look-alike.

    On a store whose prior-season rows were all observable at the cutoff and whose
    cutoff identity agrees with the persisted row, the arm must equal the
    incumbent's OWN published projection for every field the incumbent publishes —
    the whole hierarchy, not just the posterior — with exactly two declared
    differences: the pooled-prior flag is renamed (the arm attributes the pool by
    the cutoff's identity, so it may not claim the incumbent's persisted-squad
    wording) and the adapter's own construction block is added.
    """

    conn = connect_database(":memory:")
    # Two players carry no prior-season row of their own, so the pooled prior — and
    # with it the renamed flag — is exercised rather than assumed away.
    _world(conn, history_omit=(NO_PRIOR_PLAYER, ROLE_CHANGE_PLAYER))
    resolution = _pool(conn)
    arm = player_ch.build_challenger_player_arms(
        conn,
        PLANNING_EVENT,
        CUTOFF,
        players=resolution["players"],
        identities=_identities(resolution),
    )
    pool_ids = {int(player["player_id"]) for player in resolution["players"]}
    published = {
        (int(row["player_id"]), str(row["component"])): row
        for row in player_rates.build_player_rate_projections(conn, PLANNING_EVENT, CUTOFF)
        if int(row["player_id"]) in pool_ids
    }
    assert set(published) == set(arm.incumbent_rows)

    renamed = 0
    for key in sorted(published):
        expected = published[key]
        actual = arm.incumbent_rows[key]
        for field in expected:
            if field in {"risk_flags", "provenance"}:
                continue
            assert actual[field] == expected[field], (key, field)
        # Every published provenance entry is the incumbent's, unchanged.
        for name, value in expected["provenance"].items():
            assert actual["provenance"][name] == value, (key, name)
        # The flags differ ONLY by the declared renaming.
        added = set(actual["risk_flags"]) - set(expected["risk_flags"])
        removed = set(expected["risk_flags"]) - set(actual["risk_flags"])
        allowed_additions = set(player_ch.INCUMBENT_ARM_FLAG_ADDITIONS) | set(
            player_ch.INCUMBENT_ARM_FLAG_SUBSTITUTIONS.values()
        )
        assert added <= allowed_additions, (key, added)
        assert removed <= set(player_ch.INCUMBENT_ARM_FLAG_SUBSTITUTIONS), (key, removed)
        if "POSITION_FROM_CURRENT_SQUAD" in expected["risk_flags"]:
            # The incumbent publishes this flag as a bare literal, so the test pins
            # the literal rather than a constant that does not exist.
            assert player_ch.POSITION_BASIS in actual["risk_flags"]
            renamed += 1
        # The arm carries its identity and its own construction, not a rename of the
        # incumbent's version string.
        assert actual["model_version"] == player_rates.PLAYER_RATE_MODEL_VERSION
        assert actual["arm"] == player_ch.ARM_INCUMBENT
        assert actual["arm_construction"]["construction"] == player_ch.INCUMBENT_ARM_CONSTRUCTION
        assert actual["arm_construction"]["incumbent_module_modified"] is False
        assert actual["arm_construction"]["incumbent_pooled_rates_read_used"] is False
    assert renamed, "the pooled-prior path must be exercised for the renaming to mean anything"


# ===========================================================================
# The derived estimators, in both directions and at their fail-closed edges
# ===========================================================================


def _synthetic_population(between_spread: float, noise: float, players: int = 30):
    """Two played rows per player, each 90 minutes, with a designed spread/noise."""

    rows: dict[int, list[dict[str, float]]] = {}
    for index in range(players):
        base = 0.15 + between_spread * index
        rows[1000 + index] = [
            {"minutes": 90.0, "value": round(base * (1.0 - noise), 5)},
            {"minutes": 90.0, "value": round(base * (1.0 + noise), 5)},
        ]
    return rows


def _designed_population(
    players: int = 40, rows_per_player: int = 20, spread: float = 0.00866, noise: float = 0.147
):
    """A population whose between-player spread and per-match noise are DECLARED.

    Each player has ``rows_per_player`` rows of ninety minutes, alternating
    ``base * (1 - noise)`` and ``base * (1 + noise)``, so:

    * his pooled per-90 rate is exactly ``base`` (the alternation cancels);
    * his split-half difference is exactly ``2 * base * noise`` over two halves of
      ``rows_per_player / 2 * 90`` minutes, which fixes his per-minute variance at
      ``2 * base^2 * noise^2 / (rows_per_player * 90)``;
    * ``base`` runs over an arithmetic sequence, so the between-player variance of
      the true rates is the closed form below.

    The design is what makes a NUMERICAL assertion possible rather than a
    direction-only one: every quantity the estimator estimates is known in advance.
    """

    rows: dict[int, list[dict[str, float]]] = {}
    for index in range(players):
        base = 0.30 + spread * index
        rows[2000 + index] = [
            {"minutes": 90.0, "value": base * (1.0 + (noise if step % 2 else -noise))}
            for step in range(rows_per_player)
        ]
    return rows


def _designed_expectations(population, rows_per_player: int = 20):
    """The same quantities re-derived independently, in the DECLARED units."""

    sigma: list[float] = []
    rates: dict[int, tuple[float, float]] = {}
    for player_id, rows in population.items():
        first, second = rows[0::2], rows[1::2]
        minutes_first = sum(row["minutes"] for row in first)
        minutes_second = sum(row["minutes"] for row in second)
        rate_first = sum(row["value"] for row in first) / minutes_first * 90.0
        rate_second = sum(row["value"] for row in second) / minutes_second * 90.0
        sigma.append((rate_first - rate_second) ** 2 / (8100.0 * (1.0 / minutes_first + 1.0 / minutes_second)))
        rates[player_id] = (
            sum(row["value"] for row in rows),
            sum(row["minutes"] for row in rows),
        )
    sampling_variance_per_minute = sum(sigma) / len(sigma)
    total_minutes = sum(minutes for _total, minutes in rates.values())
    mean_rate = sum(
        (total / minutes * 90.0) * (minutes / total_minutes) for total, minutes in rates.values()
    )
    observed_variance = sum(
        ((total / minutes * 90.0 - mean_rate) ** 2) * (minutes / total_minutes)
        for total, minutes in rates.values()
    )
    expected_sampling = sum(
        (8100.0 * sampling_variance_per_minute / minutes) * (minutes / total_minutes)
        for _total, minutes in rates.values()
    )
    return {
        "sampling_variance_per_minute": sampling_variance_per_minute,
        "mean_rate": mean_rate,
        "observed_rate_variance": observed_variance,
        "between_player_rate_variance_per90": observed_variance - expected_sampling,
    }


def test_26b_the_derived_prior_strength_is_numerically_the_method_of_moments_value():
    """The SCALE, not the direction.  A direction test cannot see a unit error.

    ``ESS = 8100 * sigma_per_minute / between_player_rate_variance_per90`` is an
    exposure weight in minutes, and the 8100 is the per-90 scale (90^2) that turns a
    per-minute variance into one comparable with a per-90 spread.  Writing 90 there
    instead shrinks every player in the league by two orders of magnitude and moves
    every direction-only assertion the same way, so this test pins the VALUE: on a
    designed population whose true spread and true noise are known, the published
    ESS must equal the closed form — and must not equal the 90-scaled one.
    """

    config = player_ch.PlayerAttackChallengerConfig()
    incumbent_config = player_rates.PlayerRatesConfig()
    resolved = config.resolved(incumbent_config)
    population = _designed_population()
    result = player_ch.derived_prior_strength(
        population,
        resolved,
        component=player_rates.COMPONENT_XG,
        declared_ess=config.declared_ess(player_rates.COMPONENT_XG, incumbent_config),
    )
    assert result["status"] == player_ch.STRENGTH_DERIVED
    assert result["basis"] == player_ch.ESS_BASIS_SPLIT_HALF_METHOD_OF_MOMENTS

    # (a) the primaries are the units they are named for: the test's own re-derivation
    # of each one, from the same rows, in minutes and in per-90 squares.
    expected = _designed_expectations(population)
    assert result["sampling_variance_per_minute"] == pytest.approx(
        expected["sampling_variance_per_minute"], rel=1e-9
    )
    assert result["mean_rate"] == pytest.approx(expected["mean_rate"], rel=1e-9)
    assert result["observed_rate_variance"] == pytest.approx(
        expected["observed_rate_variance"], rel=1e-9
    )
    assert result["between_player_rate_variance_per90"] == pytest.approx(
        expected["between_player_rate_variance_per90"], rel=1e-9
    )
    assert result["ess_units"] == "MINUTES_OF_EXPOSURE"
    assert result["per90_variance_scale"] == 8100.0
    assert result["ess_derivation"] == player_ch.ESS_DERIVATION

    # (b) the closed form of the DESIGN: an arithmetic sequence of true per-90 rates,
    # a declared alternation per match, twenty matches of ninety minutes each.  Each
    # split half is 900 minutes, so a player's per-minute variance is
    # ``(2 * base * noise)^2 / (8100 * (2 / 900))``.
    players, rows_per_player, spread, noise = 40, 20, 0.00866, 0.147
    half_minutes = rows_per_player / 2 * 90.0
    between_true = spread**2 * (players**2 - 1) / 12.0
    mean_true = 0.30 + spread * (players - 1) / 2.0
    mean_square_true = between_true + mean_true**2
    sigma_design = mean_square_true * noise**2 * half_minutes / 4050.0
    between_design = between_true - 8100.0 * sigma_design / (rows_per_player * 90.0)
    design_ess = 8100.0 * sigma_design / between_design
    assert result["prior_ess_minutes"] == pytest.approx(design_ess, rel=1e-6)

    # (c) the value on the MINUTES scale, and never the 90-scaled one.  A league whose
    # per-90 spread and per-minute noise are these implies an exposure weight
    # comparable to the incumbent's declared 850 minutes; the wrong constant would
    # report about twenty minutes and shrink the prior to ~2% of its declared weight.
    assert result["prior_ess_minutes"] == pytest.approx(
        8100.0 * result["sampling_variance_per_minute"] / result["between_player_rate_variance_per90"],
        rel=1e-9,
    )
    wrong_scale = (
        90.0
        * result["sampling_variance_per_minute"]
        / result["between_player_rate_variance_per90"]
    )
    assert result["prior_ess_minutes"] != pytest.approx(wrong_scale, rel=1e-3)
    assert result["prior_ess_minutes"] == pytest.approx(wrong_scale * 90.0, rel=1e-9)
    assert 300.0 <= result["prior_ess_minutes"] <= 3000.0


def test_26_the_derived_player_prior_strength_moves_in_both_directions():
    config = player_ch.PlayerAttackChallengerConfig()
    resolved = config.resolved(player_rates.PlayerRatesConfig())
    spread = player_ch.derived_prior_strength(
        _synthetic_population(0.02, 0.05),
        resolved,
        component=player_rates.COMPONENT_XG,
        declared_ess=config.declared_ess(player_rates.COMPONENT_XG),
    )
    tight = player_ch.derived_prior_strength(
        _synthetic_population(0.006, 0.05),
        resolved,
        component=player_rates.COMPONENT_XG,
        declared_ess=config.declared_ess(player_rates.COMPONENT_XG),
    )
    assert spread["basis"] == tight["basis"] == player_ch.ESS_BASIS_SPLIT_HALF_METHOD_OF_MOMENTS
    assert spread["status"] == player_ch.STRENGTH_DERIVED
    # A population whose players genuinely differ is shrunk LESS: the prior earns
    # less weight when the between-player spread dominates the sampling noise.
    assert tight["prior_ess_minutes"] > spread["prior_ess_minutes"]
    assert 0.0 <= spread["prior_ess_minutes"] <= resolved["ess_max_minutes"]


def test_27_the_derived_player_prior_strength_fails_closed_with_its_reason():
    config = player_ch.PlayerAttackChallengerConfig()
    resolved = config.resolved(player_rates.PlayerRatesConfig())
    declared = config.declared_ess(player_rates.COMPONENT_XG)

    thin = player_ch.derived_prior_strength(
        _synthetic_population(0.02, 0.05, players=4),
        resolved,
        component=player_rates.COMPONENT_XG,
        declared_ess=declared,
    )
    assert thin["status"] == player_ch.STRENGTH_FALLBACK
    assert thin["prior_ess_minutes"] == declared
    assert thin["reason"] == player_ch.ESS_REASON_TOO_FEW_PLAYERS

    # A store whose values are rounded to two decimals cannot show a positive
    # sampling variance, and zero noise would mean "no shrinkage at all": refused.
    degenerate = player_ch.derived_prior_strength(
        _synthetic_population(0.02, 0.0),
        resolved,
        component=player_rates.COMPONENT_XG,
        declared_ess=declared,
    )
    assert degenerate["status"] == player_ch.STRENGTH_FALLBACK
    assert degenerate["reason"] == player_ch.ESS_REASON_NO_SAMPLING_VARIANCE
    assert degenerate["prior_ess_minutes"] == declared

    # A population with no spread at all has no between-player variance to divide by.
    flat = player_ch.derived_prior_strength(
        _synthetic_population(0.0, 0.05),
        resolved,
        component=player_rates.COMPONENT_XG,
        declared_ess=declared,
    )
    assert flat["status"] == player_ch.STRENGTH_FALLBACK
    assert flat["reason"] == player_ch.ESS_REASON_NON_POSITIVE_BETWEEN_VARIANCE
    assert flat["prior_ess_minutes"] == declared


def test_27b_the_fallback_and_comparison_ess_come_from_the_resolved_incumbent_config():
    """An incumbent run under a non-default config must be measured against IT.

    The derived prior strength falls back to, and is compared against, the
    incumbent's DECLARED prior strength.  Reading that off a fresh
    ``PlayerRatesConfig()`` instead of the config the run actually resolved would
    silently re-point both the fallback and the above/below-declared verdicts at
    published defaults the caller never asked for — so the assertions below are
    stated on a config whose declared strength is nothing like the default.
    """

    non_default = player_rates.PlayerRatesConfig(xg_prior_ess_minutes=1.0, xa_prior_ess_minutes=1.0)
    default = player_rates.PlayerRatesConfig()
    config = player_ch.PlayerAttackChallengerConfig()
    assert config.declared_ess(player_rates.COMPONENT_XG) == default.xg_prior_ess_minutes
    assert (
        config.declared_ess(player_rates.COMPONENT_XG, non_default)
        == non_default.xg_prior_ess_minutes
    )
    assert (
        config.declared_ess(player_rates.COMPONENT_XA, non_default)
        == non_default.xa_prior_ess_minutes
    )
    # A challenger-level override of the component governs when it is set, whatever
    # the incumbent declares.
    assert (
        player_ch.PlayerAttackChallengerConfig(xg_prior_ess_minutes=333.0).declared_ess(
            player_rates.COMPONENT_XG, non_default
        )
        == 333.0
    )

    # (a) the fallback path: a window too thin to derive anything uses the RESOLVED
    # declared strength, and the comparison basis recorded with it is the same number.
    thin = player_ch.derived_prior_strength(
        _synthetic_population(0.02, 0.05, players=4),
        config.resolved(non_default),
        component=player_rates.COMPONENT_XG,
        declared_ess=config.declared_ess(player_rates.COMPONENT_XG, non_default),
    )
    assert thin["status"] == player_ch.STRENGTH_FALLBACK
    assert thin["prior_ess_minutes"] == non_default.xg_prior_ess_minutes
    assert thin["declared_fallback_prior_ess_minutes"] == non_default.xg_prior_ess_minutes

    # (b) end to end: the arms, and the incumbent comparison arm, are built against the
    # resolved incumbent config rather than the published defaults.
    conn = connect_database(":memory:")
    _world(conn)
    resolution = _pool(conn)
    arm = player_ch.build_challenger_player_arms(
        conn,
        PLANNING_EVENT,
        CUTOFF,
        players=resolution["players"],
        identities=_identities(resolution),
        incumbent_config=non_default,
    )
    key = (ROLE_CHANGE_PLAYER, player_rates.COMPONENT_XG)
    # Every player in this world carries a prior-season row above the tiny-sample
    # threshold and no role signal moves his ESS scale, so the arm's base ESS IS the
    # declared one — the number the whole comparison is weighted by.
    assert arm.incumbent_rows[key]["prior_ess_before_discounts"] == non_default.xg_prior_ess_minutes
    assert arm.incumbent_arm["incumbent_config_hash"] == non_default.config_hash()
    # The derived estimate is far ABOVE this config's declared 1 minute, and the
    # verdict is taken against the RESOLVED value: read off the defaults it would say
    # the opposite, which is the defect this pins.
    assert arm.ess_estimates[player_rates.COMPONENT_XG]["prior_ess_minutes"] > 1.0
    assert arm.ess_estimates[player_rates.COMPONENT_XG][
        "declared_fallback_prior_ess_minutes"
    ] == non_default.xg_prior_ess_minutes
    checked = 0
    for family in arm.arm_names():
        if player_ch.FAMILY_DATA_DERIVED_PRIOR_STRENGTH not in arm.families[family]:
            continue
        checked += 1
        row = arm.row(family, key)
        assert player_ch.FLAG_DERIVED_PRIOR_ESS_ABOVE_DECLARED in row["risk_flags"], family
        assert player_ch.FLAG_DERIVED_PRIOR_ESS_BELOW_DECLARED not in row["risk_flags"], family
        assert row["prior_ess_estimate"]["declared_fallback_prior_ess_minutes"] == 1.0
    assert checked, "an arm carrying the derived-strength family must be exercised"

    # (c) a config the window CANNOT support: every estimate falls back, and what it
    # falls back to is the resolved config's own declared strength.
    starved = player_ch.PlayerAttackChallengerConfig(ess_min_players=10_000)
    fallback_arm = player_ch.build_challenger_player_arms(
        conn,
        PLANNING_EVENT,
        CUTOFF,
        players=resolution["players"],
        identities=_identities(resolution),
        incumbent_config=non_default,
        challenger_config=starved,
    )
    assert (
        fallback_arm.ess_estimates[player_rates.COMPONENT_XG]["status"] == player_ch.STRENGTH_FALLBACK
    )
    assert fallback_arm.incumbent_rows[key]["prior_ess_before_discounts"] == 1.0
    checked = 0
    for family in fallback_arm.arm_names():
        if player_ch.FAMILY_DATA_DERIVED_PRIOR_STRENGTH not in fallback_arm.families[family]:
            continue
        checked += 1
        row = fallback_arm.row(family, key)
        assert row["prior_ess_basis"] == player_ch.ESS_BASIS_INCUMBENT_FALLBACK
        assert player_ch.FLAG_DERIVED_PRIOR_ESS_UNAVAILABLE in row["risk_flags"], family
    assert checked


def test_28_the_team_reliability_estimator_moves_in_both_directions():
    config = team_ch.TeamAttackChallengerConfig()
    resolved = config.resolved(team_model.TeamStrengthConfig())
    declared = float(team_model.TeamStrengthConfig().attack_prior_strength)

    def residuals(noise: float, teams: int = 12):
        rows = {}
        for team in range(teams):
            rows[team] = [
                {
                    "age_matches": age,
                    "weight": 1.0,
                    "signal": team * 0.30 + noise * ((age % 2) * 2 - 1),
                }
                for age in range(4)
            ]
        return rows

    reliable = team_ch.split_half_reliability(
        residuals(0.01), resolved, component="attack", declared_fallback=declared
    )
    noisy = team_ch.split_half_reliability(
        residuals(0.60), resolved, component="attack", declared_fallback=declared
    )
    assert reliable["status"] == team_ch.STRENGTH_DERIVED
    assert reliable["reliability_full_window"] > 0.9
    assert reliable["prior_strength"] < declared
    # When the spread between teams is mostly noise, the estimator never claims more
    # precision: it either falls back or lands on a larger strength.
    assert (
        noisy["status"] == team_ch.STRENGTH_FALLBACK
        or noisy["prior_strength"] > reliable["prior_strength"]
    )
    if noisy["status"] == team_ch.STRENGTH_FALLBACK:
        assert noisy["prior_strength"] == declared
        assert noisy["reason"] in {
            team_ch.RELIABILITY_REASON_NON_POSITIVE_BETWEEN_VARIANCE,
            team_ch.RELIABILITY_REASON_NO_HALF_NOISE,
            team_ch.RELIABILITY_REASON_TOO_FEW_TEAMS,
        }

    thin = team_ch.split_half_reliability(
        residuals(0.01, teams=4), resolved, component="attack", declared_fallback=declared
    )
    assert thin["status"] == team_ch.STRENGTH_FALLBACK
    assert thin["reason"] == team_ch.RELIABILITY_REASON_TOO_FEW_TEAMS
    assert thin["prior_strength"] == declared


# ===========================================================================
# The evaluation contract: sample-size honesty, limits, strata, normalization
# ===========================================================================


def test_29_the_sample_size_policy_is_pe2s_and_insufficient_stays_descriptive():
    conn = connect_database(":memory:")
    _world(conn)
    artifact = ev.evaluate_events(conn, [PLANNING_EVENT])
    sample = artifact["sample"]
    for family in ("team", "player"):
        block = sample[family]
        assert block["policy"]["policy_version"] == wf_scoreboard.SAMPLE_POLICY_VERSION
        assert (
            block["policy"]["min_observations_for_descriptive_reporting"]
            == wf_scoreboard.MIN_OBSERVATIONS_FOR_DESCRIPTIVE
        )
        assert block["observations"] == artifact["population"][family]["scored"]
        assert block["observations"] < wf_scoreboard.MIN_OBSERVATIONS_FOR_DESCRIPTIVE
        assert block["sample_interpretation"] == wf_scoreboard.SAMPLE_INSUFFICIENT
    assert sample["headline_interpretation"] == wf_scoreboard.SAMPLE_INSUFFICIENT
    # And no arm may convert that into a model-selection claim.
    for family in ("team", "player"):
        for arm, payload in artifact[family]["arms"].items():
            assert payload["measurement"]["token"] == ev.MEASUREMENT_INSUFFICIENT_SAMPLE
            assert payload["measurement"]["basis"] == [ev.MEASUREMENT_BASIS_INSUFFICIENT_SAMPLE]
    assert artifact["claims"]["selection_claim"] == ev.SELECTION_CLAIM_NOT_MADE
    assert artifact["claims"]["allowed"] == list(ev.CLAIMS_ALLOWED)
    assert set(ev.CLAIMS_NOT_ALLOWED).issubset(set(artifact["claims"]["not_allowed"]))


def test_30_every_stratum_reports_its_own_sample_size():
    conn = connect_database(":memory:")
    _world(conn)
    artifact = ev.evaluate_events(conn, [PLANNING_EVENT])
    for family, dimensions in (
        ("team", ev.TEAM_STRATUM_DIMENSIONS),
        ("player", ev.PLAYER_STRATUM_DIMENSIONS),
    ):
        strata = artifact[family]["strata"]
        assert set(strata) == set(dimensions)
        for dimension, block in strata.items():
            assert block["strata"], dimension
            for key, entry in block["strata"].items():
                assert entry["n"] > 0, (dimension, key)
                assert entry["sample_interpretation"] in {
                    "SUFFICIENT_FOR_STRATUM_DESCRIPTIVE_REPORTING",
                    "INSUFFICIENT_STRATUM_SAMPLE",
                }
                for arm in artifact[family]["arms"]:
                    assert entry["arms"][arm]["metrics"]["n"] == entry["n"]
    # The declared dimensions are the ones the contract asks for, populated by data.
    team_strata = artifact["team"]["strata"]
    assert set(team_strata["by_venue"]["strata"]) == {"home", "away"}
    assert len(team_strata["by_team"]["strata"]) == 3  # teams 1, 2 and 3 play
    player_strata = artifact["player"]["strata"]
    assert len(player_strata["by_position"]["strata"]) >= 2
    assert len(player_strata["by_prior_evidence"]["strata"]) >= 1
    assert len(player_strata["by_history_coverage"]["strata"]) >= 1


def test_31_duplicate_target_events_are_normalized_not_double_counted():
    conn = connect_database(":memory:")
    _world(conn)
    single = ev.evaluate_events(conn, [PLANNING_EVENT])
    doubled = ev.evaluate_events(conn, [PLANNING_EVENT, PLANNING_EVENT])
    assert doubled["population"]["normalization"]["duplicates_removed"] == 1
    assert doubled["population"]["normalization"]["normalized"] == [PLANNING_EVENT]
    assert doubled["population"]["normalization"]["repeated_events"] == {str(PLANNING_EVENT): 2}
    assert (
        doubled["population"]["team"]["population_digest"]
        == single["population"]["team"]["population_digest"]
    )
    assert (
        doubled["population"]["player"]["population_digest"]
        == single["population"]["player"]["population_digest"]
    )
    assert doubled["population"]["team"]["scored"] == single["population"]["team"]["scored"]


def test_32_an_event_without_a_causal_cutoff_reports_no_population_at_all():
    conn = connect_database(":memory:")
    _world(conn)
    with conn:
        conn.execute("UPDATE events SET deadline_time=NULL WHERE id=?", (PLANNING_EVENT,))
    artifact = ev.evaluate_events(conn, [PLANNING_EVENT])
    cutoff, reasons = ev.event_cutoff(conn, PLANNING_EVENT)
    assert cutoff is None and reasons
    event = artifact["events"][0]
    assert event["cutoff"] is None
    assert event["excluded_by_status"]["team"][ev.STATUS_EVENT_CUTOFF_UNAVAILABLE] == 1
    assert event["excluded_by_status"]["player"][ev.STATUS_EVENT_CUTOFF_UNAVAILABLE] == 1
    for family in ("team", "player"):
        block = artifact["population"][family]
        assert block["candidate_slots"] is None
        assert block["candidate_count_available"] is False
        assert block["scored"] == 0
        assert block["accounting"]["reconciles"] is True


def test_32b_an_event_whose_cutoff_carries_no_team_evidence_is_excluded_never_projected():
    """A season-opening cutoff is a legitimate walk-forward state, not a crash.

    The incumbent is fitted on completed fixtures of STRICTLY earlier events, so
    the first event of a season has a well-formed cutoff and no team evidence at
    all: no level, no attack, no defence.  PE-7 must not project such an event,
    must not score its sides as zero, and must not reach the incumbent's
    projection path — whose ``lambda_for`` is undefined on an empty evidence set —
    because a walk-forward over a season's event list has to produce an artifact
    rather than a traceback.  Every fixture side is excluded with a status and the
    accounting still closes.
    """

    conn = connect_database(":memory:")
    _world(conn)
    first, first_cutoff = 1, EVENT_DEADLINES[1]

    # The condition the exclusion exists for, asserted before it is relied on.
    assert team_model.team_match_rows(conn, first, first_cutoff) == []
    empty = team_model.fit_team_strength(conn, first, first_cutoff)
    assert empty["league_log_baseline"] is None
    assert empty["data_gaps"] == ["no completed fixtures with official xG before the cutoff"]
    # The frozen incumbent's own projection path cannot be asked for these sides,
    # which is why PE-7 excludes them instead of calling it.
    with pytest.raises(TypeError):
        team_model.build_team_fixture_projections(conn, first, first_cutoff)

    artifact = ev.evaluate_events(conn, [first, PLANNING_EVENT])
    opening = next(entry for entry in artifact["events"] if entry["event"] == first)
    assert opening["team_evidence"]["available"] is False
    assert opening["team_evidence"]["matches"] == 0
    assert opening["team_evidence"]["data_gaps"]
    sides = 2 * len(EVENT_FIXTURES[first])
    assert opening["excluded_by_status"]["team"] == {ev.STATUS_TEAM_EVIDENCE_UNAVAILABLE: sides}
    team_population = artifact["population"]["team"]
    # The team side is still ENUMERABLE from the fixture list, so the population is
    # counted and every side is excluded with its status rather than the comparison
    # quietly narrowing.
    assert team_population["candidate_slots"] == sides + 2 * len(EVENT_FIXTURES[PLANNING_EVENT])
    assert team_population["accounting"]["reconciles"] is True
    # The event that DOES carry evidence still scores, on both its fixtures.
    assert team_population["scored"] == 2 * len(EVENT_FIXTURES[PLANNING_EVENT])
    # No team environment means no coherence comparison, and the omission is stated
    # rather than reported as a pass.
    coherence_block = artifact["coherence"]["per_event"][str(first)]
    assert coherence_block["arms"] == {}
    assert "unavailable at this cutoff" in coherence_block["note"]
    # And the whole artifact is produced, with its digest, over a season-opening
    # list: the CLI's own --events 1..N entry point cannot raise.
    artifact_all = ev.evaluate_events(conn, sorted(EVENT_DEADLINES))
    assert artifact_all["artifact_digest"].startswith("sha256:")
    assert artifact_all["population"]["team"]["accounting"]["reconciles"] is True
    assert artifact_all["population"]["player"]["accounting"]["reconciles"] is True


def test_33_an_unaccepted_pool_generation_excludes_candidates_at_event_scope():
    conn = connect_database(":memory:")
    _world(conn)
    with conn:
        conn.execute("UPDATE bootstrap_generations SET accepted=0")
    artifact = ev.evaluate_events(conn, [PLANNING_EVENT])
    resolution = ev.resolve_event(conn, PLANNING_EVENT)
    assert resolution["identity_available"] is False
    assert resolution["players"] == []
    player_population = artifact["population"]["player"]
    # No pool means no candidate COUNT: the mutable players table is not enumerated
    # to invent one.
    assert player_population["candidate_slots"] is None
    assert player_population["candidate_count_available"] is False
    assert player_population["events_without_a_candidate_count"] == 1
    assert player_population["scored"] == 0
    assert (
        artifact["events"][0]["excluded_by_status"]["player"][
            ev.STATUS_IDENTITY_UNAVAILABLE_AT_CUTOFF
        ]
        == 1
    )
    # The team side needs no player identity: a team side is a fixture side.
    assert artifact["population"]["team"]["scored"] == 4


def test_34_the_artifact_declares_its_multi_family_arms_and_its_digest():
    conn = connect_database(":memory:")
    _world(conn)
    artifact = ev.evaluate_events(conn, [PLANNING_EVENT])
    for family in ("team", "player"):
        policy = artifact[family]["multi_family_subset_policy"]
        subsets = policy["declared_subsets"]
        assert list(subsets) == ["challenger_all_refinements"]
        entry = subsets["challenger_all_refinements"]
        assert entry["headline"] is True
        assert entry["status"] == "EVALUATED_AS_ITS_OWN_ARM"
        assert entry["measurement"]["token"] in ev.MEASUREMENT_VOCABULARY
        assert artifact[family]["measurement_rule"]["criteria_declared_a_priori"] is True
        assert artifact[family]["measurement_rule"]["selected_from_the_data"] is False
        assert (
            policy["single_family_arms_are_not_a_subset"] is True
        )
    assert artifact["artifact_digest"].startswith("sha256:")
    assert artifact["schema"] == ev.ARTIFACT_SCHEMA


def test_35_the_evaluation_cli_prints_the_artifact(tmp_path):
    """The CLI is exercised the way an operator runs it: as its own process.

    A subprocess keeps this smoke test honest about what the CLI does with a
    read-only database connection, and independent of anything the suite has done
    to this process' file handles by the time it runs.
    """

    import subprocess
    import sys
    from pathlib import Path

    database = tmp_path / "fpl.db"
    conn = connect_database(str(database))
    _world(conn)
    conn.commit()
    conn.close()
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "season": "2026/27",
                "paths": {
                    "database": str(database),
                    "raw_dir": str(tmp_path / "raw"),
                    "exports_dir": str(tmp_path / "exports"),
                },
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "pe7.json"
    script = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_pe7_team_player_attack.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--config",
            str(config),
            "--events",
            str(PLANNING_EVENT),
            "--json-out",
            str(out),
        ],
        capture_output=True,
        text=True,
        cwd=str(script.parents[1]),
        check=False,
    )
    assert completed.returncode == 0, (completed.returncode, completed.stdout[-2000:], completed.stderr[-2000:])
    assert "same population" in completed.stdout
    assert "causality audit" in completed.stdout
    assert "xPts diagnostic : NOT_COMPUTED" in completed.stdout
    artifact = json.loads(out.read_text(encoding="utf-8"))
    assert artifact["schema"] == ev.ARTIFACT_SCHEMA
    assert artifact["population"]["same_population"] is True
    assert artifact["identity"]["promotion"] == "NOT_PERFORMED_THIS_ARTIFACT_MEASURES_ONLY"

def test_36_the_artifact_carries_an_explicit_causality_audit():
    conn = connect_database(":memory:")
    _world(conn)
    artifact = ev.evaluate_events(conn, [PLANNING_EVENT])
    audit = artifact["causality"]["audit"]
    assert len(audit) == artifact["causality"]["audited_surface_count"] >= 8
    surfaces = {entry["surface"] for entry in audit}
    for required in (
        "players.team_id (candidate club)",
        "players.team_id (historical fixture side)",
        "players.element_type",
        "players.is_active",
        "current club identity (player/team joins)",
        "current role / scouting evidence",
        "player/team joins for historical priors",
        # The arm the delta is measured against is audited like any other read: an
        # audit that covered the challenger alone would leave the baseline unstated.
        "frozen incumbent player-rate arm (the comparison)",
    ):
        assert required in surfaces, required
    for entry in audit:
        assert entry["resolution"].strip(), entry
        assert entry["used_for"].strip(), entry
    discipline = artifact["causality"]["discipline"]
    assert discipline["second_predicate_introduced"] is False
    assert discipline["missing_is_zero"] is False
    assert "historical_observations" in discipline["boundary"]
    assert "incumbent_player_rate_rows" in discipline["incumbent_arm"]
    # The disclosure is on the artifact, not only in the module: the identity block
    # and every event name the incumbent arm's construction.
    assert (
        artifact["identity"]["player_incumbent_arm_construction"]
        == player_ch.INCUMBENT_ARM_CONSTRUCTION
    )
    assert artifact["identity"]["player_incumbent_arm_boundary"] == player_ch.INCUMBENT_ARM_BOUNDARY
    assert artifact["player"]["incumbent_arm"]["construction"] == player_ch.INCUMBENT_ARM_CONSTRUCTION
    assert artifact["player"]["incumbent_arm"]["boundary"] == player_ch.INCUMBENT_ARM_BOUNDARY
    event = artifact["events"][0]
    incumbent_arm = event["incumbent_arm"]
    assert incumbent_arm["construction"] == player_ch.INCUMBENT_ARM_CONSTRUCTION
    assert incumbent_arm["incumbent_module_modified"] is False
    assert incumbent_arm["incumbent_pooled_rates_read_used"] is False
    assert incumbent_arm["model_version"] == player_rates.PLAYER_RATE_MODEL_VERSION
    assert incumbent_arm["incumbent_config_hash"] == player_rates.PlayerRatesConfig().config_hash()
    assert incumbent_arm["rows"] == incumbent_arm["players"] * len(player_rates.COMPONENTS)
    assert incumbent_arm["players"] > 0


def test_37_every_emitted_exclusion_status_is_declared_and_never_scored_as_zero():
    """The exclusion vocabulary is published, and an emitted status belongs to it.

    A count whose status a reader cannot look up is not a disclosure, so the
    declared vocabulary travels with the artifact and every status the evaluation
    actually emits is checked against it — the same contract PE-6's artifact
    carries for its own exclusions.
    """

    conn = connect_database(":memory:")
    _world(conn)
    # A season-opening list: the target events include cutoffs that hold no team
    # evidence and no accepted pool generation, so several statuses fire at once.
    artifact = ev.evaluate_events(conn, sorted(EVENT_DEADLINES))
    population = artifact["population"]
    assert population["exclusion_statuses"] == list(ev.EXCLUSION_STATUSES)
    assert set(population["never_scored_as_zero"])

    emitted: set[str] = set()
    for family in ("team", "player"):
        emitted.update(population[family]["excluded_by_status"])
    for entry in artifact["events"]:
        for family in ("team", "player"):
            emitted.update(entry["excluded_by_status"][family])
    assert emitted, "the walk-forward must exclude something, or this proves nothing"
    assert emitted.issubset(set(ev.EXCLUSION_STATUSES)), sorted(
        emitted - set(ev.EXCLUSION_STATUSES)
    )
    # The events with no team evidence at their cutoff are among them, excluded with
    # PE-2's own token and never scored as a zero.
    assert ev.STATUS_TEAM_EVIDENCE_UNAVAILABLE in emitted
    assert population["team"]["accounting"]["reconciles"] is True
    assert population["player"]["accounting"]["reconciles"] is True
