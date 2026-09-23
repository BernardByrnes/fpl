"""PE-8 calibration: the contract's hard cases, on hand-controlled worlds.

Every expected number below was derived by hand from the declared definition
before the implementation was run.

Reference world (events 5 and 6 are DOUBLE gameweeks for teams 1 and 2)
---------------------------------------------------------------------
players: 10 GKP team 1, 11 DEF team 1, 12 MID team 1, 13 FWD team 2,
         14 MID team 2, 15 MID team 3 (no fixture), 16 MID team 1 (no projection)
fixtures: event 5 -> 100 = 1 v 2, 101 = 2 v 1; event 6 -> 102 = 1 v 2, 103 = 2 v 1

The realised side is generated from the frozen scoring rules, so the world is
self-consistent by construction: ``total_points`` is the realised CORE plus bonus,
and each persisted component's realised counterpart is the same quantity the
evaluation reconstructs.

The ``_big_world`` factory builds a population large enough to cross the DECLARED
sample floors (10 target events, 2000 observations), because those floors are
policy and are not lowered for a test: it is the only way to exercise the
READY_FOR_MERGE branch, the multi-origin causal fit, and the population-wide
reliability bins honestly.
"""

from __future__ import annotations

import inspect
import json
import pathlib
import re
import sqlite3
from typing import Any, Mapping

import pytest

from fpl_brain import analytics
from fpl_brain import calibration as mc_calibration
from fpl_brain import calibration_evaluation as ce
from fpl_brain import defcon_calibration as defcon_cal
from fpl_brain import four_gw_decision as fg
from fpl_brain import monte_carlo as mc_module
from fpl_brain import probability_calibration as pc
from fpl_brain import walk_forward as wf
from fpl_brain import walk_forward_metrics as wm
from fpl_brain import walk_forward_scoreboard as sb
from fpl_brain import xpts as xpts_module
from fpl_brain.database import connect_database
from fpl_brain.scoring_rules import DEFAULT_SCORING_RULES, POSITION_IDS

CUTOFF = "2026-09-10T12:00:00Z"
CODE_SNAPSHOT = "sha256:" + "c" * 64
XPTS_RUN_VERSION = xpts_module.XPTS_MODEL_VERSION
MC_RUN_VERSION = mc_module.MONTE_CARLO_MODEL_VERSION
MINUTES_RUN_VERSION = "minutes_v1.5.2"
#: The registered DefCon calibration, written into every persisted DefCon row.
DEFCON_PAYLOAD = defcon_cal.DEFCON_PLATT_V1.as_dict()
DISTRIBUTION_BASIS = "CORE (bonus deterministic, variance unmodelled)"

TEAM_OF = {10: 1, 11: 1, 12: 1, 13: 2, 14: 2, 15: 3, 16: 1}
POSITION_OF = {10: "GKP", 11: "DEF", 12: "MID", 13: "FWD", 14: "MID", 15: "MID", 16: "MID"}
ELEMENT_TYPE_OF = {10: 1, 11: 2, 12: 3, 13: 4, 14: 3, 15: 3, 16: 3}

#: ``{event: {fixture: (home, away)}}`` -- the double gameweek shape.
EVENT_FIXTURES = {
    5: {100: (1, 2), 101: (2, 1)},
    6: {102: (1, 2), 103: (2, 1)},
}

#: Realised facts, ``(event, player) -> (minutes, starts, goals, assists,
#: conceded, saves, yellow, bonus)``.  The clean-sheet flag is derived from the
#: frozen rule rather than stored as an independent claim.
FACTS = {
    (5, 10): (90, 1, 0, 0, 0, 0, 0, 0),
    (5, 11): (90, 1, 0, 0, 0, 0, 0, 0),
    (5, 12): (30, 0, 0, 0, 0, 0, 0, 0),
    (5, 13): (90, 1, 1, 0, 0, 0, 1, 0),
    (5, 14): (20, 0, 0, 0, 0, 0, 0, 0),
    (6, 10): (90, 1, 0, 0, 0, 0, 0, 0),
    (6, 11): (45, 0, 0, 0, 1, 0, 0, 0),
    (6, 12): (90, 1, 0, 1, 0, 0, 0, 0),
    (6, 13): (90, 1, 0, 0, 0, 0, 0, 0),
    # A genuine non-appearance: every performance column is an explicit zero, so
    # it is a real zero-minute outcome rather than a scheduled placeholder.
    (6, 14): (0, 0, 0, 0, 0, 0, 0, 0),
}

#: DefCon defensive-contribution counts, chosen to reach or miss each position's
#: declared threshold (DEF 10, MID/PWD 12; GKP has none).
DEFCON_CONTRIBUTION = {
    (5, 10): 15,
    (5, 11): 14,
    (5, 12): 4,
    (5, 13): 2,
    (5, 14): 3,
    (6, 10): 15,
    (6, 11): 7,
    (6, 12): 5,
    (6, 13): 20,
    (6, 14): 0,
}

#: Stated probabilities per player, shared by every fixture of an event unless
#: overridden.  ``p_start`` is chosen so the event-5 Brier is hand-computable.
PROBABILITIES = {
    10: {"p_start": 0.9, "p_60_plus": 0.85, "clean_sheet_probability": 0.4, "defcon_p_hit": 0.5},
    11: {"p_start": 0.6, "p_60_plus": 0.6, "clean_sheet_probability": 0.4, "defcon_p_hit": 0.4},
    12: {"p_start": 0.3, "p_60_plus": 0.2, "clean_sheet_probability": 0.3, "defcon_p_hit": 0.3},
    13: {"p_start": 0.8, "p_60_plus": 0.8, "clean_sheet_probability": 0.05, "defcon_p_hit": 0.6},
    14: {"p_start": 0.1, "p_60_plus": 0.05, "clean_sheet_probability": 0.2, "defcon_p_hit": 0.2},
    15: {"p_start": 0.5, "p_60_plus": 0.5, "clean_sheet_probability": 0.3, "defcon_p_hit": 0.5},
}

#: The four surfaces scored in the reference world, in declared order.
SURFACES = ("BRIER_P_START", "BRIER_P_60_PLUS", "BRIER_CLEAN_SHEET", "BRIER_DEFCON")

#: Rows the declared population offers per event: players 10-14 have a fixture
#: and a projection, so 5 players x 2 fixtures.
ROWS_PER_EVENT = 10

PRODUCTION_FLAGS = [
    "BONUS_LOW_CONFIDENCE",
    "BONUS_SOFT",
    "BPS_RULE_DISCONTINUITY",
    "CS_MINUTES_APPROX_V1",
    "FPL_ASSIST_MAPPING_UNCALIBRATED",
    "PENALTIES_EMBEDDED",
]

EXPECTED_XA = {10: 0.2, 11: 0.2, 12: 0.2, 13: 0.2, 14: 0.2, 15: 0.1, 16: 0.1}


# ---------------------------------------------------------------------------
# The reference world
# ---------------------------------------------------------------------------


def _realised_row(event: int, player_id: int) -> dict:
    """One realised ``player_gameweeks`` row, with the CORE derived from the rules.

    The realised side is the WORLD's truth: a model offset injected by a test
    moves the persisted payload only, never the official column, so a component
    bias is visible in exactly the place the model claims it.
    """

    minutes, starts, goals, assists, conceded, saves, yellow, bonus = FACTS[(event, player_id)]
    position = POSITION_OF[player_id]
    contribution = DEFCON_CONTRIBUTION[(event, player_id)]
    earned_clean_sheet = DEFAULT_SCORING_RULES.earns_clean_sheet_points(position, minutes, conceded)
    outcome = {
        "minutes": minutes,
        "starts": starts,
        "goals_scored": goals,
        "assists": assists,
        "clean_sheets": 1 if earned_clean_sheet else 0,
        "goals_conceded": conceded,
        "saves": saves,
        "bonus": bonus,
        "yellow_cards": yellow,
        "defensive_contribution": contribution,
    }
    core = mc_calibration.actual_modelled_core(outcome, position, DEFAULT_SCORING_RULES)
    outcome["total_points"] = int(core + bonus)
    outcome["_core"] = core
    return outcome


def _component_payload(event: int, player_id: int, *, goal_bias: float = 0.0, assist_bias: float = 0.0) -> dict:
    """The persisted ``xpts_v1`` payload: every component, plus the four probabilities."""

    outcome = _realised_row(event, player_id)
    position = POSITION_OF[player_id]
    rules = DEFAULT_SCORING_RULES
    minutes = outcome["minutes"]
    appearance = (
        float(rules.appearance_long_points)
        if minutes >= rules.clean_sheet_minutes_required
        else (float(rules.appearance_short_points) if minutes > 0 else 0.0)
    )
    threshold = rules.defcon_threshold_for(position)
    defcon_hit = (
        threshold is not None
        and position in rules.defcon_positions
        and outcome["defensive_contribution"] >= threshold
    )
    clean_sheet = float(rules.clean_sheet_points_for(position)) if outcome["clean_sheets"] else 0.0
    conceded_steps = (
        outcome["goals_conceded"] // rules.goals_conceded_per_deduction
        if position in rules.goals_conceded_positions
        else 0
    )
    goals_conceded_xpts = -float(conceded_steps) * abs(rules.goals_conceded_points_for(position))
    save_xpts = (
        float(outcome["saves"] // rules.saves_per_point) if position == "GKP" else 0.0
    )
    yellow_xpts = float(outcome["yellow_cards"] * rules.yellow_card_points)
    goal_xpts = float(outcome["goals_scored"] * rules.goal_points_for(position)) + goal_bias
    assist_xpts = float(outcome["assists"] * rules.assist_points) + assist_bias
    defcon_xpts = float(rules.defcon_points) if defcon_hit else 0.0
    bonus_xpts = float(outcome["bonus"])
    core = (
        appearance
        + goal_xpts
        + assist_xpts
        + clean_sheet
        + goals_conceded_xpts
        + defcon_xpts
        + save_xpts
        + yellow_xpts
    )
    payload = {
        **PROBABILITIES[player_id],
        "p_appearance": 0.9,
        "expected_xa": EXPECTED_XA[player_id],
        "appearance_xpts": appearance,
        "goal_xpts": goal_xpts,
        "assist_xpts": assist_xpts,
        "clean_sheet_xpts": clean_sheet,
        "goals_conceded_xpts": goals_conceded_xpts,
        "defcon_xpts": defcon_xpts,
        "save_xpts": save_xpts,
        "yellow_card_xpts": yellow_xpts,
        "bonus_xpts": bonus_xpts,
        "soft_xpts": bonus_xpts,
        "core_xpts": core,
        "total_xpts": core + bonus_xpts,
        "defcon_actions_per90": 12.0,
        "risk_flags": list(PRODUCTION_FLAGS),
        "defcon_calibration": DEFCON_PAYLOAD,
    }
    return payload


def _quantiles(event: int, player_id: int) -> dict:
    outcome = _realised_row(event, player_id)
    core = outcome["_core"]
    return {
        "q10": core - 2.0,
        "q25": core - 1.0,
        "q50": core,
        "q75": core + 1.0,
        "q90": core + 2.0,
        "mean_core": core,
        "distribution_basis": DISTRIBUTION_BASIS,
    }


def _small_world(
    conn: sqlite3.Connection,
    *,
    probability_overrides: dict[tuple[int, int], dict] | None = None,
    payload_overrides: dict[tuple[int, int], dict] | None = None,
    quantile_overrides: dict[tuple[int, int], dict] | None = None,
    drop_probability_field: str | None = None,
    include_event_6: bool = True,
    reverse_inserts: bool = False,
    goal_bias: float = 0.0,
    assist_bias: float = 0.0,
    defcon_payload_override: dict | None = None,
) -> dict:
    """Build the reference league and return the certified anchor artifact."""

    from fpl_brain.models import EventRecord, FixtureRecord, PlayerRecord, PositionRecord, TeamRecord
    from fpl_brain import repositories as repo

    probability_overrides = probability_overrides or {}
    payload_overrides = payload_overrides or {}
    quantile_overrides = quantile_overrides or {}
    events = [5] + ([6] if include_event_6 else [])
    fixtures = {
        fixture_id: pair
        for event, block in EVENT_FIXTURES.items()
        if event in events
        for fixture_id, pair in block.items()
    }

    with conn:
        repo.upsert_teams(conn, [TeamRecord(id=team, name=f"Team {team}") for team in (1, 2, 3)])
        repo.upsert_positions(
            conn,
            [
                PositionRecord(id=element_type, singular_name_short=name)
                for element_type, name in POSITION_IDS.items()
            ],
        )
        repo.upsert_players(
            conn,
            [
                PlayerRecord(
                    id=player_id,
                    web_name=f"P{player_id}",
                    full_name=f"Player {player_id}",
                    team_id=TEAM_OF[player_id],
                    element_type=ELEMENT_TYPE_OF[player_id],
                )
                for player_id in TEAM_OF
            ],
        )
        repo.upsert_events(
            conn,
            [
                EventRecord(
                    id=event,
                    finished=1,
                    data_checked=1,
                    deadline_time="2026-09-05T17:30:00Z",
                    raw_json={},
                )
                for event in events
            ],
        )
        repo.upsert_fixtures(
            conn,
            [
                FixtureRecord(
                    id=fixture_id,
                    event=event,
                    team_h=pair[0],
                    team_a=pair[1],
                    finished=1,
                    started=1,
                    kickoff_time="2026-09-06T14:00:00Z",
                    raw_json={},
                )
                for event, block in EVENT_FIXTURES.items()
                if event in events
                for fixture_id, pair in block.items()
            ],
        )

    rows: list[tuple[int, int, int, dict]] = []
    for event in events:
        for fixture_id, (home, away) in sorted(EVENT_FIXTURES[event].items()):
            for player_id, team in sorted(TEAM_OF.items()):
                if team not in (home, away) or (event, player_id) not in FACTS:
                    continue
                rows.append((event, fixture_id, player_id, _realised_row(event, player_id)))
    if reverse_inserts:
        rows = list(reversed(rows))
    with conn:
        for event, fixture_id, player_id, outcome in rows:
            conn.execute(
                "INSERT INTO player_gameweeks(player_id, event, fixture_id, minutes, starts, total_points,"
                " goals_scored, assists, clean_sheets, goals_conceded, saves, bonus, yellow_cards,"
                " defensive_contribution, source, updated_at, raw_json)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,'element_summary','2026-09-07T09:00:00Z','{}')",
                (
                    player_id,
                    event,
                    fixture_id,
                    outcome["minutes"],
                    outcome["starts"],
                    outcome["total_points"],
                    outcome["goals_scored"],
                    outcome["assists"],
                    outcome["clean_sheets"],
                    outcome["goals_conceded"],
                    outcome["saves"],
                    outcome["bonus"],
                    outcome["yellow_cards"],
                    outcome["defensive_contribution"],
                ),
            )
        # Player 15's team has no fixture at all; his stored row is the canonical
        # scheduled placeholder, which must never be scored.
        conn.execute(
            "INSERT INTO player_gameweeks(player_id, event, fixture_id, minutes, source, updated_at, raw_json)"
            " VALUES (15,5,100,0,'element_summary','2026-09-05T08:00:00Z','{}')"
        )

    xpts_runs = {
        event: analytics.create_projection_run(
            conn,
            model_family="xpts_v1",
            model_version=XPTS_RUN_VERSION,
            planning_event=event,
            planning_context_hash="ctx",
            data_cutoff=CUTOFF,
            scouting_cutoff=None,
            official_run_ids={},
            source_snapshot_sha256=CODE_SNAPSHOT,
        )
        for event in events
    }
    mc_runs = {
        event: analytics.create_projection_run(
            conn,
            model_family="monte_carlo_v1",
            model_version=MC_RUN_VERSION,
            planning_event=event,
            planning_context_hash="ctx",
            data_cutoff=CUTOFF,
            scouting_cutoff=None,
            official_run_ids={},
            source_snapshot_sha256=CODE_SNAPSHOT,
        )
        for event in events
    }
    minutes_runs = {
        event: analytics.create_projection_run(
            conn,
            model_family="minutes_v1",
            model_version=MINUTES_RUN_VERSION,
            planning_event=event,
            planning_context_hash="ctx",
            data_cutoff=CUTOFF,
            scouting_cutoff=None,
            official_run_ids={},
            source_snapshot_sha256=CODE_SNAPSHOT,
        )
        for event in events
    }

    projection_rows = [
        (event, fixture_id, player_id)
        for event in events
        for fixture_id, (home, away) in fixtures.items()
        if fixture_id in EVENT_FIXTURES[event]
        for player_id, team in sorted(TEAM_OF.items())
        if team in (home, away) and player_id != 16 and (event, player_id) in FACTS
    ]
    if reverse_inserts:
        projection_rows = list(reversed(projection_rows))
    with conn:
        for event, fixture_id, player_id in projection_rows:
            home, away = fixtures[fixture_id]
            team_id = TEAM_OF[player_id]
            opponent = away if team_id == home else home
            payload = _component_payload(event, player_id, goal_bias=goal_bias, assist_bias=assist_bias)
            payload.update(probability_overrides.get((player_id, fixture_id), {}))
            payload.update(probability_overrides.get((player_id, 0), {}))
            payload.update(payload_overrides.get((event, player_id), {}))
            if drop_probability_field:
                payload.pop(drop_probability_field, None)
            conn.execute(
                "INSERT INTO player_fixture_xpts_projections(projection_run_id, player_id, fixture_id,"
                " event, team_id, opponent_id, position, minutes_run_id, team_run_id, rate_run_id,"
                " payload_json, model_version, scoring_rules_version, generated_at)"
                " VALUES (?,?,?,?,?,?,?,?,1,1,?,?, 'fpl_scoring_2026_27_v1.0.0','2026-09-10T12:00:00Z')",
                (
                    xpts_runs[event],
                    player_id,
                    fixture_id,
                    event,
                    team_id,
                    opponent,
                    POSITION_OF[player_id],
                    minutes_runs[event],
                    json.dumps(payload),
                    XPTS_RUN_VERSION,
                ),
            )
            quantiles = _quantiles(event, player_id)
            quantiles.update(quantile_overrides.get((player_id, fixture_id), {}))
            conn.execute(
                "INSERT INTO monte_carlo_distributions(projection_run_id, player_id, fixture_id, event,"
                " team_id, opponent_id, position, xpts_run_id, minutes_run_id, team_run_id, rate_run_id,"
                " payload_json, model_version, generated_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,1,1,?,?,'2026-09-10T12:00:00Z')",
                (
                    mc_runs[event],
                    player_id,
                    fixture_id,
                    event,
                    team_id,
                    opponent,
                    POSITION_OF[player_id],
                    xpts_runs[event],
                    minutes_runs[event],
                    json.dumps(quantiles),
                    MC_RUN_VERSION,
                ),
            )
    with conn:
        for run_id in list(xpts_runs.values()) + list(mc_runs.values()) + list(minutes_runs.values()):
            conn.execute("UPDATE projection_runs SET status='complete' WHERE id=?", (int(run_id),))

    artifact = {
        "schema": "four_gw_certification_v1",
        "four_gw_certification_identity": "sha256:" + "d" * 64,
        "planning_cutoff": CUTOFF,
        "certified_bundles": {
            str(event): {
                "event": event,
                "cutoff": CUTOFF,
                "runs": {
                    "xpts_v1": xpts_runs[event],
                    "minutes_v1": minutes_runs[event],
                    "monte_carlo_v1": mc_runs[event],
                },
                "model_versions": {
                    "xpts_v1": XPTS_RUN_VERSION,
                    "minutes_v1": MINUTES_RUN_VERSION,
                    "monte_carlo_v1": MC_RUN_VERSION,
                },
                "code_snapshot_sha256": CODE_SNAPSHOT,
            }
            for event in events
        },
    }
    return {
        "events": events,
        "artifact": artifact,
        "xpts_runs": xpts_runs,
        "monte_carlo_runs": mc_runs,
        "minutes_runs": minutes_runs,
    }


@pytest.fixture()
def world(tmp_path):
    conn = connect_database(tmp_path / "pe8.db")
    try:
        built = _small_world(conn)
        yield conn, built
    finally:
        conn.close()


# --- the population large enough to cross the DECLARED sample floors ---------

BIG_EVENTS = tuple(range(5, 15))
BIG_PLAYERS = tuple(range(1000, 1200))
BIG_TEAM_OF = {player_id: (1 if player_id < 1100 else 2) for player_id in BIG_PLAYERS}
BIG_GROUP_A = tuple(player_id for player_id in BIG_PLAYERS if player_id % 2 == 0)
BIG_GROUP_B = tuple(player_id for player_id in BIG_PLAYERS if player_id % 2 == 1)


def _big_fixtures(event: int) -> dict[int, tuple[int, int]]:
    base = 1000 + 2 * (int(event) - BIG_EVENTS[0])
    return {base: (1, 2), base + 1: (2, 1)}


def _big_facts(
    *,
    index: int,
    group_rows: int,
    start_share: float,
    cs_share: float,
    assist: bool,
    zero: bool,
) -> dict:
    """One row of the calibrated design, so every aggregate bias is exactly zero."""

    if zero:
        return {
            "minutes": 0, "starts": 0, "goals": 0, "assists": 0, "conceded": 0,
            "contribution": 0, "bonus": 0, "sixty": False,
        }
    sixty = index < int(round(group_rows * start_share))
    clean_sheet = index < int(round(group_rows * cs_share))
    return {
        "minutes": 90 if sixty else 30,
        "starts": 1 if sixty else 0,
        "goals": 0,
        "assists": 1 if assist else 0,
        "conceded": 0 if clean_sheet else (1 if sixty else 0),
        "contribution": 15 if sixty else 3,
        "bonus": 0,
        "sixty": sixty,
    }


def _big_payload(player_id: int, facts: Mapping[str, Any], *, expected_xa: float) -> dict:
    """The persisted payload for one big-world row, designed to be unbiased overall."""

    rules = DEFAULT_SCORING_RULES
    group_a = player_id % 2 == 0
    p_start = 0.58 if group_a else 0.52
    cs_probability = 0.22 if group_a else 0.28
    minutes = facts["minutes"]
    appearance = (
        float(rules.appearance_long_points)
        if minutes >= rules.clean_sheet_minutes_required
        else (float(rules.appearance_short_points) if minutes > 0 else 0.0)
    )
    earned_clean_sheet = 1.0 if (facts["conceded"] == 0 and minutes >= 60) else 0.0
    defcon_hit = facts["contribution"] >= int(rules.defcon_threshold_for("MID"))
    defcon_xpts = float(rules.defcon_points) if defcon_hit else 0.0
    assist_points = float(facts["assists"] * rules.assist_points)
    predicted_assists = float(expected_xa) * rules.assist_points
    predicted_core = appearance + cs_probability + 2.0 * p_start + predicted_assists
    payload = {
        "p_start": p_start,
        "p_60_plus": p_start,
        "p_appearance": 1.0,
        "clean_sheet_probability": cs_probability,
        "defcon_p_hit": p_start,
        "expected_xa": float(expected_xa),
        "appearance_xpts": 1.0 + p_start,
        "goal_xpts": 0.0,
        "assist_xpts": predicted_assists,
        "clean_sheet_xpts": cs_probability,
        "goals_conceded_xpts": 0.0,
        "defcon_xpts": 2.0 * p_start,
        "save_xpts": 0.0,
        "yellow_card_xpts": 0.0,
        "bonus_xpts": 0.0,
        "soft_xpts": 0.0,
        "core_xpts": predicted_core,
        "total_xpts": predicted_core,
        "risk_flags": ["BONUS_SOFT", "PENALTIES_EMBEDDED", "CS_MINUTES_APPROX_V1"],
        "defcon_calibration": DEFCON_PAYLOAD,
        "_realised_core": appearance + earned_clean_sheet + defcon_xpts + assist_points,
        "_earned_clean_sheet": earned_clean_sheet,
    }
    return payload


def _big_world(
    conn: sqlite3.Connection,
    *,
    start_share_a: float = 0.60,
    start_share_b: float = 0.50,
    cs_share_a: float = 0.20,
    cs_share_b: float = 0.30,
    expected_xa: float = 0.0,
    assist_every: int = 0,
    alt_last_event: bool = False,
) -> dict:
    """A calibrated population over ten events: 4000 fixture rows and 2000 event rows.

    The stated probabilities and the realised frequencies agree exactly at the bin
    level, and every component's predicted mean equals its realised mean, so
    READY_FOR_MERGE is reachable without weakening a declared floor.  ``assist_every``
    places one realised assist every N rows, so the assist mapping has both a
    positive expected total and a coefficient inside its declared bounds.

    ``alt_last_event`` rewrites ONLY the last event's realised outcomes.  No
    origin's fit basis contains its own event, so every fitted transform must be
    unmoved while the last event's scored figures change: the causality contract,
    checked at the evaluation level rather than only inside the fit.
    """

    from fpl_brain.models import EventRecord, FixtureRecord, PlayerRecord, PositionRecord, TeamRecord
    from fpl_brain import repositories as repo

    with conn:
        repo.upsert_teams(conn, [TeamRecord(id=team, name=f"Team {team}") for team in (1, 2)])
        repo.upsert_positions(
            conn,
            [
                PositionRecord(id=element_type, singular_name_short=name)
                for element_type, name in POSITION_IDS.items()
            ],
        )
        repo.upsert_players(
            conn,
            [
                PlayerRecord(
                    id=player_id,
                    web_name=f"B{player_id}",
                    full_name=f"Big Player {player_id}",
                    team_id=BIG_TEAM_OF[player_id],
                    element_type=3,
                )
                for player_id in BIG_PLAYERS
            ],
        )
        repo.upsert_events(
            conn,
            [
                EventRecord(id=event, finished=1, data_checked=1,
                            deadline_time="2026-09-05T17:30:00Z", raw_json={})
                for event in BIG_EVENTS
            ],
        )
        repo.upsert_fixtures(
            conn,
            [
                FixtureRecord(id=fixture_id, event=event, team_h=pair[0], team_a=pair[1],
                              finished=1, started=1, kickoff_time="2026-09-06T14:00:00Z", raw_json={})
                for event in BIG_EVENTS
                for fixture_id, pair in sorted(_big_fixtures(event).items())
            ],
        )

    def _run(family: str, version: str, event: int) -> int:
        return analytics.create_projection_run(
            conn,
            model_family=family,
            model_version=version,
            planning_event=event,
            planning_context_hash="pe8-big",
            data_cutoff=CUTOFF,
            scouting_cutoff=None,
            official_run_ids={},
            source_snapshot_sha256=CODE_SNAPSHOT,
        )

    xpts_runs = {event: _run("xpts_v1", XPTS_RUN_VERSION, event) for event in BIG_EVENTS}
    mc_runs = {event: _run("monte_carlo_v1", MC_RUN_VERSION, event) for event in BIG_EVENTS}
    minutes_runs = {event: _run("minutes_v1", MINUTES_RUN_VERSION, event) for event in BIG_EVENTS}

    row_number = 0
    with conn:
        for event in BIG_EVENTS:
            fixtures = sorted(_big_fixtures(event).items())
            last = alt_last_event and event == BIG_EVENTS[-1]
            for players, start_share, cs_share in (
                (BIG_GROUP_A, start_share_a, cs_share_a),
                (BIG_GROUP_B, start_share_b, cs_share_b),
            ):
                group_rows = len(players) * 2
                for rank, player_id in enumerate(players):
                    for slot, (fixture_id, (home, away)) in enumerate(fixtures):
                        assist = (
                            bool(assist_every)
                            and row_number % int(assist_every) == 0
                            and not last
                        )
                        facts = _big_facts(
                            index=rank * 2 + slot,
                            group_rows=group_rows,
                            start_share=start_share,
                            cs_share=cs_share,
                            assist=assist,
                            zero=last,
                        )
                        payload = _big_payload(player_id, facts, expected_xa=expected_xa)
                        realised_core = payload.pop("_realised_core")
                        earned_clean_sheet = payload.pop("_earned_clean_sheet")
                        team_id = BIG_TEAM_OF[player_id]
                        opponent = away if team_id == home else home
                        conn.execute(
                            "INSERT INTO player_gameweeks(player_id, event, fixture_id, minutes, starts,"
                            " total_points, goals_scored, assists, clean_sheets, goals_conceded, saves,"
                            " bonus, yellow_cards, defensive_contribution, source, updated_at, raw_json)"
                            " VALUES (?,?,?,?,?,?,?,?,?,?,0,?,0,?,'element_summary',"
                            "'2026-09-07T09:00:00Z','{}')",
                            (
                                player_id,
                                event,
                                fixture_id,
                                facts["minutes"],
                                facts["starts"],
                                int(realised_core + facts["bonus"]),
                                facts["goals"],
                                facts["assists"],
                                int(earned_clean_sheet),
                                facts["conceded"],
                                facts["bonus"],
                                facts["contribution"],
                            ),
                        )
                        conn.execute(
                            "INSERT INTO player_fixture_xpts_projections(projection_run_id, player_id,"
                            " fixture_id, event, team_id, opponent_id, position, minutes_run_id,"
                            " team_run_id, rate_run_id, payload_json, model_version,"
                            " scoring_rules_version, generated_at)"
                            " VALUES (?,?,?,?,?,?,'MID',?,1,1,?,?, 'fpl_scoring_2026_27_v1.0.0',"
                            "'2026-09-10T12:00:00Z')",
                            (
                                xpts_runs[event],
                                player_id,
                                fixture_id,
                                event,
                                team_id,
                                opponent,
                                minutes_runs[event],
                                json.dumps(payload),
                                XPTS_RUN_VERSION,
                            ),
                        )
                        conn.execute(
                            "INSERT INTO monte_carlo_distributions(projection_run_id, player_id,"
                            " fixture_id, event, team_id, opponent_id, position, xpts_run_id,"
                            " minutes_run_id, team_run_id, rate_run_id, payload_json, model_version,"
                            " generated_at)"
                            " VALUES (?,?,?,?,?,?,'MID',?,?,1,1,?,?,'2026-09-10T12:00:00Z')",
                            (
                                mc_runs[event],
                                player_id,
                                fixture_id,
                                event,
                                team_id,
                                opponent,
                                xpts_runs[event],
                                minutes_runs[event],
                                json.dumps(
                                    {
                                        "q10": realised_core - 2.0,
                                        "q25": realised_core - 1.0,
                                        "q50": realised_core,
                                        "q75": realised_core + 1.0,
                                        "q90": realised_core + 2.0,
                                        "mean_core": realised_core,
                                        "distribution_basis": DISTRIBUTION_BASIS,
                                    }
                                ),
                                MC_RUN_VERSION,
                            ),
                        )
                        row_number += 1
        for run_id in list(xpts_runs.values()) + list(mc_runs.values()) + list(minutes_runs.values()):
            conn.execute("UPDATE projection_runs SET status='complete' WHERE id=?", (int(run_id),))

    artifact = {
        "schema": fg.CERTIFICATION_ARTIFACT_SCHEMA_V1,
        "four_gw_certification_identity": "sha256:" + "e" * 64,
        "planning_cutoff": CUTOFF,
        "certified_bundles": {
            str(event): {
                "event": event,
                "cutoff": CUTOFF,
                "runs": {
                    "xpts_v1": xpts_runs[event],
                    "minutes_v1": minutes_runs[event],
                    "monte_carlo_v1": mc_runs[event],
                },
                "model_versions": {
                    "xpts_v1": XPTS_RUN_VERSION,
                    "minutes_v1": MINUTES_RUN_VERSION,
                    "monte_carlo_v1": MC_RUN_VERSION,
                },
                "code_snapshot_sha256": CODE_SNAPSHOT,
            }
            for event in BIG_EVENTS
        },
    }
    return {"events": list(BIG_EVENTS), "artifact": artifact, "xpts_runs": xpts_runs}


@pytest.fixture(scope="module")
def big_world(tmp_path_factory):
    """The calibrated population, built once: every big-world test only reads it."""

    conn = connect_database(tmp_path_factory.mktemp("pe8-big") / "big.db")
    try:
        yield conn, _big_world(conn)
    finally:
        conn.close()


def _single_use_world(tmp_path, **kwargs):
    conn = connect_database(tmp_path / "pe8-variant.db")
    try:
        built = _big_world(conn, **kwargs)
    except Exception:
        conn.close()
        raise
    return conn, built


def _evaluate(conn, world, **kwargs):
    kwargs.setdefault("events", world["events"])
    return ce.evaluate(conn, artifact=world["artifact"], **kwargs)


def _surface(artifact, metric):
    for entry in artifact["probability_calibration"]["surfaces"]:
        if entry["metric"] == metric:
            return entry
    raise AssertionError(f"surface {metric} is absent")


def _component(artifact, name):
    for entry in artifact["expected_value"]["component_diagnostics"]["components"]:
        if entry["component"] == name:
            return entry
    raise AssertionError(f"component {name} is absent")


def _scoring_rows(conn, built, *, metric):
    population = sb.player_fixture_population(
        conn,
        events=built["events"],
        xpts_runs={
            event: built["xpts_runs"][event] for event in built["events"]
        },
    )
    scored = sb.probability_scoring_rows(population["rows"])
    return [row for row in scored["rows"] if row.metric == metric], population, scored


# ---------------------------------------------------------------------------
# 1-6: the transform contract
# ---------------------------------------------------------------------------


def _observations(rows):
    return [
        pc.CalibrationObservation(
            event=event, key=(index, 0), probability=probability, outcome=outcome
        )
        for index, (event, probability, outcome) in enumerate(rows)
    ]


def _platt_basis(*, events=(5, 6), per_event=150, positives=0.6, spread=0.4):
    """A basis large enough to cross the declared fit floor, with both classes."""

    rows = []
    for event in events:
        for index in range(per_event):
            strong = index % 2 == 0
            probability = 0.5 + (spread / 2 if strong else -spread / 2)
            hit = index < int(per_event * (positives if strong else positives - 0.2))
            rows.append((event, probability, 1.0 if hit else 0.0))
    return _observations(rows)


def test_no_same_event_fit_and_score_is_refused():
    """Hard test 1: a transform may not be fitted on the event it is applied at."""

    basis = _observations([(5, 0.5, 1.0), (5, 0.4, 0.0)])
    with pytest.raises(pc.CausalFitError) as error:
        pc.fit_platt_causal(
            basis, origin_event=5, surface="BRIER_P_START", grain=wf.GRAIN_PLAYER_FIXTURE
        )
    assert "contract violation" in str(error.value)
    # The same basis is legitimate one event later: the refusal is about the
    # origin, not about the rows.
    later = pc.fit_platt_causal(
        basis, origin_event=6, surface="BRIER_P_START", grain=wf.GRAIN_PLAYER_FIXTURE
    )
    assert later.basis.as_dict()["strictly_before_origin"] is True
    assert later.available is False, "two observations cannot support a fit; it is refused, not guessed"


def _scoring_row(event: int, index: int, probability: float, outcome: float) -> sb.ProbabilityScoringRow:
    """One synthetic scoring row, for the causal wiring tests that need no database."""

    return sb.ProbabilityScoringRow(
        metric="BRIER_P_START",
        event=int(event),
        player_id=1000 + int(index),
        fixture_id=100,
        position="MID",
        probability=float(probability),
        outcome=float(outcome),
        xpts_run_id=1,
        risk_flags=(),
    )


def _row_basis(*, events=(5, 6), per_event=150, positives=0.6, spread=0.4):
    """Scoring rows large enough to cross the declared fit floor, both classes present."""

    rows: list[sb.ProbabilityScoringRow] = []
    for event in events:
        for index in range(per_event):
            strong = index % 2 == 0
            probability = 0.5 + (spread / 2 if strong else -spread / 2)
            hit = index < int(per_event * (positives if strong else positives - 0.2))
            rows.append(_scoring_row(event, index, probability, 1.0 if hit else 0.0))
    return rows


def test_no_same_event_fit_and_score_is_refused():
    """Hard test 1: a transform may not be fitted on the event it is applied at."""

    basis = _observations([(5, 0.5, 1.0), (5, 0.4, 0.0)])
    with pytest.raises(pc.CausalFitError) as error:
        pc.fit_platt_causal(
            basis, origin_event=5, surface="BRIER_P_START", grain=wf.GRAIN_PLAYER_FIXTURE
        )
    assert "contract violation" in str(error.value)
    # The same basis is legitimate one event later: the refusal is about the
    # origin, not about the rows.
    later = pc.fit_platt_causal(
        basis, origin_event=6, surface="BRIER_P_START", grain=wf.GRAIN_PLAYER_FIXTURE
    )
    assert later.basis.as_dict()["strictly_before_origin"] is True
    assert later.available is False, "two observations cannot support a fit; it is refused, not guessed"


def test_a_post_origin_outcome_cannot_change_an_earlier_transform(monkeypatch):
    """Hard test 2: the fit basis is strictly earlier, and later rows cannot move it."""

    early_rows = _row_basis(events=(5, 6, 7))
    early = ce.causal_origins(early_rows, surface="BRIER_P_START")[7]
    assert early.available is True
    assert tuple(early.basis.events) == (5, 6)
    identity = early.spec.identity()
    version = early.spec.version

    # The SAME population plus two later events: origin 7's transform is unmoved,
    # because events 8 and 9 are not in its basis, and the basis is what the
    # version and identity are derived from.
    with_later = early_rows + _row_basis(events=(8, 9), positives=0.95)
    unchanged = ce.causal_origins(with_later, surface="BRIER_P_START")[7]
    assert unchanged.spec.identity() == identity
    assert unchanged.spec.version == version
    assert unchanged.basis.digest == early.basis.digest
    assert tuple(unchanged.basis.events) == (5, 6), "events 8 and 9 are not in origin 7's basis"

    # And a CHANGED outcome INSIDE the basis does move it: the invariance above is
    # causality, not a digest that ignores its inputs.
    mutated = ce.causal_origins(
        _row_basis(events=(5, 6, 7), positives=0.75), surface="BRIER_P_START"
    )[7]
    assert mutated.spec.identity() != identity

    # A caller that hands the fit a late outcome instead of filtering is refused.
    late = [pc.CalibrationObservation(event=7, key=(1, 1), probability=0.6, outcome=1.0)]
    with pytest.raises(pc.CausalFitError):
        pc.fit_platt_causal(
            _observations([(5, 0.5, 1.0)]) + late,
            origin_event=7,
            surface="BRIER_P_START",
            grain=wf.GRAIN_PLAYER_FIXTURE,
        )


def test_the_evaluation_records_a_strictly_earlier_basis_for_every_origin():
    """Hard test 2, at the evaluation's own wiring: origins score under their own fit."""

    rows = _row_basis(events=(5, 6, 7), per_event=220, spread=0.1)
    fits = ce.causal_origins(rows, surface="BRIER_P_START")
    assert sorted(fits) == [5, 6, 7]
    assert fits[5].available is False and fits[5].reason == pc.FIT_INSUFFICIENT_OBSERVATIONS
    for origin, fit in fits.items():
        assert fit.basis.as_dict()["strictly_before_origin"] is True
        assert all(event < origin for event in fit.basis.events)
    assert tuple(fits[7].basis.events) == (5, 6)
    assert fits[7].basis.observations == 440
    assert fits[7].available is True

    block, covered = ce._challenger_block(rows, surface="BRIER_P_START", grain=wf.GRAIN_PLAYER_FIXTURE)
    assert block["status"] == ce.STATUS_OK, "origins 6 and 7 have a basis; origin 5 does not"
    assert block["rows_without_origin_transform"] == 220
    assert block["excluded_by_reason"] == {pc.FIT_INSUFFICIENT_OBSERVATIONS: 220}
    assert len(covered) == 440
    assert block["population"]["n"] == 440
    assert block["population"]["rows_without_origin_transform"] == 220
    assert block["gate"]["same_population"] is True
    assert block["incumbent_on_comparison_population"]["n"] == 440
    assert block["candidate"]["n"] == 440
    assert [entry["rows_scored_at_origin"] for entry in block["origins"]] == [0, 220, 220]
    assert block["origins"][0]["fit"]["reason"] == pc.FIT_INSUFFICIENT_OBSERVATIONS
    assert block["origins"][1]["fit"]["available"] is True
    # The transform in force at each origin is the one fitted for that origin, and
    # the two origins have different bases, so they are different transforms.
    assert block["origins"][1]["fit"]["spec"]["version"] != block["origins"][2]["fit"]["spec"]["version"]
    assert block["versions_in_force"] == sorted(
        {
            block["origins"][1]["fit"]["spec"]["version"],
            block["origins"][2]["fit"]["spec"]["version"],
        }
    )


def test_the_transform_is_a_versioned_artifact_with_canonical_identity_and_provenance():
    """Hard test 3: version, validated spec, canonical identity, bound into the payload."""

    fit = pc.fit_platt_causal(
        _platt_basis(events=(5, 6)),
        origin_event=7,
        surface="BRIER_P_START",
        grain=wf.GRAIN_PLAYER_FIXTURE,
    )
    assert fit.available is True
    spec = fit.spec
    assert spec.version.startswith(pc.PLATT_FIT_VERSION_PREFIX + ".")
    assert spec.method == pc.METHOD_PLATT
    assert spec.monotonicity()["monotone_non_decreasing"] is True
    payload = spec.as_payload()
    assert payload["identity"] == spec.identity()
    assert payload["policy_version"] == pc.PROBABILITY_CALIBRATION_POLICY_VERSION
    # The persisted spec reads back through the fail-closed reader, identity
    # included, and the fit record carries it as provenance.
    round_tripped = pc.from_payload(payload)
    assert round_tripped.as_dict() == spec.as_dict()
    assert fit.as_dict()["provenance"]["identity"] == spec.identity()
    assert spec.version.endswith(fit.basis.digest.split(":")[-1][:16])


def test_a_fitted_version_can_never_be_redefined_in_place():
    """Two different bases cannot share a version; the same basis always does."""

    first = pc.fit_platt_causal(
        _platt_basis(events=(5, 6)), origin_event=7, surface="BRIER_P_START", grain=wf.GRAIN_PLAYER_FIXTURE
    )
    same = pc.fit_platt_causal(
        _platt_basis(events=(5, 6)), origin_event=7, surface="BRIER_P_START", grain=wf.GRAIN_PLAYER_FIXTURE
    )
    other = pc.fit_platt_causal(
        _platt_basis(events=(5, 6), positives=0.7), origin_event=7, surface="BRIER_P_START",
        grain=wf.GRAIN_PLAYER_FIXTURE,
    )
    assert first.spec.version == same.spec.version
    assert first.spec.identity() == same.spec.identity()
    assert other.spec.version != first.spec.version
    assert other.spec.identity() != first.spec.identity()


def test_an_unknown_calibration_version_fails_closed_with_no_identity_fallback():
    """Hard test 4: no silent degradation to identity, by version or by payload."""

    with pytest.raises(pc.ProbabilityCalibrationError):
        pc.resolve(None)
    with pytest.raises(pc.ProbabilityCalibrationError):
        pc.resolve("prob_platt_wf_v1.0.0.deadbeefdeadbeef")
    with pytest.raises(pc.ProbabilityCalibrationError):
        pc.resolve("")
    with pytest.raises(pc.ProbabilityCalibrationError):
        pc.from_payload(None)
    with pytest.raises(pc.ProbabilityCalibrationError):
        pc.from_payload({"version": "some_other_calibration_v1.0.0", "method": pc.METHOD_IDENTITY})
    with pytest.raises(pc.ProbabilityCalibrationError):
        pc.from_payload(
            {
                "version": pc.INCUMBENT_CALIBRATION_VERSION,
                "method": pc.METHOD_PLATT,
                "intercept": 0.0,
                "slope": 1.0,
                "clip_floor": 1e-6,
            }
        )
    # A tampered identity is refused even when the version namespace is declared.
    spec = pc.ProbabilityCalibration(
        version=pc.PLATT_FIT_VERSION_PREFIX + ".0123456789abcdef",
        method=pc.METHOD_PLATT,
        intercept=-0.4,
        slope=1.1,
    )
    tampered = spec.as_payload()
    tampered["slope"] = 1.2
    with pytest.raises(pc.ProbabilityCalibrationError) as error:
        pc.from_payload(tampered)
    assert "records identity" in str(error.value)
    # The one spec a version CAN resolve is the declared incumbent, and it is
    # declared rather than assumed.
    assert pc.resolve(pc.INCUMBENT_CALIBRATION_VERSION) is pc.INCUMBENT_IDENTITY
    assert pc.is_declared_version(pc.INCUMBENT_CALIBRATION_VERSION) is True
    assert pc.is_declared_version(pc.PLATT_FIT_VERSION_PREFIX) is False
    assert pc.is_declared_version("prob_platt_wf_v1.0.0") is False


def test_a_probability_transform_is_monotone_or_it_is_rejected():
    """Hard test 5: a non-monotone map is not a probability calibration."""

    for slope in (0.0, -1.0):
        with pytest.raises(pc.ProbabilityCalibrationError):
            pc.ProbabilityCalibration(
                version=pc.PLATT_FIT_VERSION_PREFIX + ".deadbeefdeadbeef",
                method=pc.METHOD_PLATT,
                intercept=0.0,
                slope=slope,
            )
    inverted = pc.fit_platt_causal(
        _observations([(5, 0.6, 0.0)] * 120 + [(5, 0.4, 1.0)] * 120),
        origin_event=6,
        surface="BRIER_P_START",
        grain=wf.GRAIN_PLAYER_FIXTURE,
    )
    assert inverted.available is False
    assert inverted.reason == pc.FIT_NON_MONOTONE
    assert inverted.spec is None

    spec = pc.ProbabilityCalibration(
        version=pc.PLATT_FIT_VERSION_PREFIX + ".0123456789abcdef",
        method=pc.METHOD_PLATT,
        intercept=-0.7,
        slope=0.9,
    )
    check = spec.verify_monotone(subdivisions=200)
    assert check["monotone_non_decreasing"] is True
    assert check["boundaries_exact"] is True


def test_boundaries_are_preserved_exactly():
    """Hard test 6: raw 0 stays 0 and raw 1 stays 1, interior clip or not."""

    spec = pc.ProbabilityCalibration(
        version=pc.PLATT_FIT_VERSION_PREFIX + ".0123456789abcdef",
        method=pc.METHOD_PLATT,
        intercept=-1.3,
        slope=0.7,
    )
    assert spec.apply(0.0) == 0.0
    assert spec.apply(1.0) == 1.0
    assert pc.INCUMBENT_IDENTITY.apply(0.0) == 0.0
    assert pc.INCUMBENT_IDENTITY.apply(1.0) == 1.0
    assert 0.0 < spec.apply(1e-12) < 1.0
    assert 0.0 < spec.apply(1.0 - 1e-12) < 1.0
    assert spec.clip_floor == pc.DEFAULT_CLIP_FLOOR
    with pytest.raises(pc.ProbabilityCalibrationError):
        pc.ProbabilityCalibration(
            version=pc.PLATT_FIT_VERSION_PREFIX + ".0123456789abcdef",
            method=pc.METHOD_PLATT,
            intercept=0.0,
            slope=1.0,
            clip_floor=0.6,
        )


def test_a_monotone_transform_preserves_within_event_ordering_but_not_expected_values():
    """Transform rule 9, stated rather than implied."""

    spec = pc.ProbabilityCalibration(
        version=pc.PLATT_FIT_VERSION_PREFIX + ".0123456789abcdef",
        method=pc.METHOD_PLATT,
        intercept=-0.2,
        slope=1.4,
    )
    raw = [0.05, 0.2, 0.2, 0.55, 0.9]
    mapped = [spec.apply(value) for value in raw]
    assert mapped == sorted(mapped)
    assert len(set(mapped)) == len(set(raw)), "distinct stated probabilities stay distinct"
    assert mapped[3] != raw[3], "a transform does change the expected value it feeds"
    assert "NONE" in ce.claims_block()["production_wiring"]


# ---------------------------------------------------------------------------
# 7-13: the probability surfaces
# ---------------------------------------------------------------------------


def test_the_declared_probability_population_is_the_scoreboards_own(world):
    """The population and its digest come from PE-2's declared rows, not a copy."""

    conn, built = world
    artifact = _evaluate(conn, built)
    rows, population, _scored = _scoring_rows(conn, built, metric="BRIER_P_START")
    assert population["candidates"] == 2 * ROWS_PER_EVENT
    assert population["excluded"] == {"fixture_not_played": 0, "outcome_row_missing": 0}
    surface = _surface(artifact, "BRIER_P_START")
    assert surface["population"]["scored"] == len(rows) == 2 * ROWS_PER_EVENT
    assert surface["population"]["population_digest"] == wf.canonical_population_digest(
        [row.key for row in rows], grain=wf.GRAIN_PLAYER_FIXTURE
    )
    assert surface["population"]["policy"] == sb.PROBABILITY_POPULATION_POLICY
    assert surface["grain"] == wf.GRAIN_PLAYER_FIXTURE


def test_p_start_is_scored_by_brier_and_reliability_hand_computed(world):
    """Hard test 7: p_start on the declared anchor population, event 5, by hand."""

    conn, built = world
    artifact = ce.evaluate(conn, artifact=built["artifact"], events=[5])
    surface = _surface(artifact, "BRIER_P_START")
    figure = surface["incumbent"]["figure"]
    assert surface["population"]["scored"] == 2 * 5, "five players with a fixture and a projection, two fixtures"
    # (0.9-1)^2 and (0.9-1)^2  -> 0.01 x 2   player 10
    # (0.6-1)^2 and (0.6-1)^2  -> 0.16 x 2   player 11
    # (0.3-0)^2 and (0.3-0)^2  -> 0.09 x 2   player 12
    # (0.8-1)^2 and (0.8-1)^2  -> 0.04 x 2   player 13
    # (0.1-0)^2 and (0.1-0)^2  -> 0.01 x 2   player 14
    # total 0.62 over 10 rows -> 0.062
    assert figure["brier"]["value"] == pytest.approx(0.062, abs=1e-9)
    assert figure["n"] == 10
    # The base rate is 6/10, so the reference Brier is 0.6 x 0.4 = 0.24.
    assert figure["brier_reference"]["value"] == pytest.approx(0.24, abs=1e-9)
    assert figure["observed_rate"] == pytest.approx(0.6, abs=1e-9)
    bins = {entry["index"]: entry for entry in figure["reliability"]["bins"]}
    assert bins[1]["n"] == 2 and bins[1]["stated_mean"] == pytest.approx(0.1)
    assert bins[1]["observed_frequency"] == 0.0 and bins[1]["gap"] == pytest.approx(-0.1)
    assert bins[3]["observed_frequency"] == 0.0 and bins[3]["gap"] == pytest.approx(-0.3)
    assert bins[6]["observed_frequency"] == 1.0 and bins[6]["gap"] == pytest.approx(0.4)
    assert bins[9]["observed_frequency"] == 1.0
    assert surface["incumbent"]["calibration"]["version"] == pc.INCUMBENT_CALIBRATION_VERSION
    assert surface["incumbent"]["calibration"]["identity"] == pc.INCUMBENT_IDENTITY.identity()


def test_p_60_plus_and_clean_sheet_are_scored_on_the_same_population(world):
    """Hard tests 8 and 9: the two remaining non-DefCon surfaces, same rows."""

    conn, built = world
    artifact = ce.evaluate(conn, artifact=built["artifact"], events=[5])
    reference_rows, _population, _scored = _scoring_rows(conn, built, metric="BRIER_P_START")
    reference_keys = {row.key for row in reference_rows}
    for metric in ("BRIER_P_60_PLUS", "BRIER_CLEAN_SHEET"):
        surface = _surface(artifact, metric)
        rows, _population, _scored = _scoring_rows(conn, built, metric=metric)
        assert {row.key for row in rows} == reference_keys, "the same rows, not merely the same N"
        assert surface["population"]["scored"] == 10
        assert surface["incumbent"]["figure"]["brier"]["status"] == wm.METRIC_OK
        assert surface["incumbent"]["figure"]["brier_reference"]["status"] == wm.METRIC_OK
        assert len(surface["incumbent"]["figure"]["reliability"]["bins"]) == 10
        assert surface["grain"] == wf.GRAIN_PLAYER_FIXTURE
    # p_60_plus: minutes >= 60 for players 10, 11, 13 in event 5, and not for 12, 14.
    sixty = _surface(artifact, "BRIER_P_60_PLUS")["incumbent"]["figure"]
    expected = (
        (0.85 - 1) ** 2 * 2
        + (0.6 - 1) ** 2 * 2
        + (0.2 - 0) ** 2 * 2
        + (0.8 - 1) ** 2 * 2
        + (0.05 - 0) ** 2 * 2
    ) / 10
    assert sixty["brier"]["value"] == pytest.approx(expected, abs=1e-9)
    # clean sheet: GKP 10 and DEF 11 play 90 minutes with no goal conceded, so they
    # earn it; MID 12 (30 minutes) and MID 14 (20 minutes) do not reach the
    # boundary; FWD 13 plays 90 minutes of a clean sheet and still earns NOTHING,
    # because the canonical rule requires the position to receive clean-sheet
    # points at all.  Two of five players, so four of ten rows.
    sheet = _surface(artifact, "BRIER_CLEAN_SHEET")["incumbent"]["figure"]
    assert sheet["observed_rate"] == pytest.approx(0.4, abs=1e-9)
    assert DEFAULT_SCORING_RULES.clean_sheet_points_for("FWD") == 0


def test_defcon_is_scored_and_gkp_is_excluded_not_fabricated(world):
    """Hard test 10: GKP has no threshold, so its rows are excluded and counted."""

    conn, built = world
    artifact = _evaluate(conn, built)
    surface = _surface(artifact, "BRIER_DEFCON")
    rows, _population, _scored = _scoring_rows(conn, built, metric="BRIER_DEFCON")
    assert surface["population"]["scored"] == len(rows) == 16
    assert surface["population"]["outcome_unavailable"] == 4, "the four GKP rows define no threshold"
    assert all(row.position != "GKP" for row in rows)
    assert DEFAULT_SCORING_RULES.defcon_threshold_for("GKP") is None
    assert "GKP" in surface["population"]["gkp_excluded_reason"]
    # event 5: DEF 11 (14 >= 10) reaches it; MID 12 (4), FWD 13 (2) and MID 14 (3) do not.
    # event 6: FWD 13 (20 >= 12) reaches it; DEF 11 (7), MID 12 (5) and MID 14 (0) do not.
    figure = surface["incumbent"]["figure"]
    assert figure["observed_rate"] == pytest.approx(0.25, abs=1e-9)
    assert figure["brier"]["value"] == pytest.approx(
        ((0.4 - 1) ** 2 * 2 + (0.3 - 0) ** 2 * 2 + (0.6 - 0) ** 2 * 2 + (0.2 - 0) ** 2 * 2
         + (0.4 - 0) ** 2 * 2 + (0.3 - 0) ** 2 * 2 + (0.6 - 1) ** 2 * 2 + (0.2 - 0) ** 2 * 2) / 16,
        abs=1e-9,
    )


def test_defcon_calibration_identity_is_carried_and_single_definition(world):
    """Hard test 22: the DefCon figure is tied to the spec that produced it."""

    conn, built = world
    artifact = ce.evaluate(conn, artifact=built["artifact"], events=[5])
    block = _surface(artifact, "BRIER_DEFCON")["defcon_calibration"]
    registered = defcon_cal.resolve(defcon_cal.DEFCON_CALIBRATION_VERSION)
    assert block["status"] == ce.STATUS_OK
    assert block["version"] == defcon_cal.DEFCON_CALIBRATION_VERSION
    assert block["identity"] == registered.identity()
    assert block["single_definition"] is True
    assert block["carried_on_every_row"] is True
    assert block["boundaries"] == {"raw_zero_stays_zero": True, "raw_one_stays_one": True}
    # The registry is the ONE definition every producer resolves: the xPts config
    # resolves the same version to the same identity, and the Monte Carlo kernel
    # reads it back from the payload rather than from a global.
    assert xpts_module.XPtsConfig().defcon_calibration.identity() == registered.identity()
    assert defcon_cal.from_payload(DEFCON_PAYLOAD).identity() == registered.identity()
    assert "defcon_cal.from_payload" in inspect.getsource(mc_module)


def test_a_disagreeing_defcon_calibration_on_the_rows_fails_closed(world):
    """A second, independently derived DefCon probability is a defect, not a refinement."""

    conn, built = world
    override = dict(DEFCON_PAYLOAD)
    override["intercept"] = DEFCON_PAYLOAD["intercept"] + 0.5
    other = defcon_cal.DefconCalibration(
        version=DEFCON_PAYLOAD["version"],
        method=DEFCON_PAYLOAD["method"],
        intercept=override["intercept"],
        slope=DEFCON_PAYLOAD["slope"],
    )
    # A different parameter set under the SAME version is refused by the frozen
    # reader, before PE-8's own single-definition check is even reached.
    with pytest.raises(defcon_cal.DefconCalibrationError):
        defcon_cal.from_payload(other.as_dict())
    row = sb.ProbabilityScoringRow(
        metric="BRIER_DEFCON",
        event=5,
        player_id=1,
        fixture_id=100,
        position="DEF",
        probability=0.5,
        outcome=1.0,
        xpts_run_id=built["xpts_runs"][5],
        risk_flags=(),
        defcon_calibration_payload={"version": "defcon_unknown_v9.9.9", "method": "PLATT",
                                    "intercept": 0.0, "slope": 1.0, "clip_floor": 1e-6},
    )
    with pytest.raises(defcon_cal.DefconCalibrationError):
        ce.defcon_calibration_block([row])


def test_an_absent_probability_is_excluded_and_counted_never_scored_as_zero():
    """Hard test 11: missing is counted, never imputed."""

    conn = connect_database(":memory:")
    try:
        built = _small_world(conn, drop_probability_field="clean_sheet_probability")
        artifact = _evaluate(conn, built)
    finally:
        conn.close()
    surface = _surface(artifact, "BRIER_CLEAN_SHEET")
    assert surface["population"]["scored"] == 0
    assert surface["population"]["probability_absent"] == 20, "every candidate row, both events"
    figure = surface["incumbent"]["figure"]
    assert figure["n"] == 0
    assert figure["brier"]["value"] is None
    assert figure["brier"]["status"] == wm.METRIC_NO_SAMPLE
    assert figure["brier_reference"]["value"] is None
    assert figure["observed_rate"] is None
    assert all(entry["observed_frequency"] is None for entry in figure["reliability"]["bins"])
    # The other surfaces are unaffected: this is a per-field exclusion.
    assert _surface(artifact, "BRIER_P_START")["population"]["scored"] == 20


def test_a_probability_outside_the_unit_interval_fails_closed():
    """Hard test 12: fail closed rather than clip into a different question."""

    conn = connect_database(":memory:")
    try:
        built = _small_world(conn, probability_overrides={(12, 0): {"p_start": 1.5}})
        with pytest.raises(sb.ScoreboardError) as error:
            _evaluate(conn, built)
    finally:
        conn.close()
    assert "outside [0, 1]" in str(error.value)
    with pytest.raises(pc.ReliabilityInputError):
        pc.reliability_bins([1.5], [1.0])
    with pytest.raises(pc.ReliabilityInputError):
        pc.reliability_bins([-0.1], [0.0])
    with pytest.raises(pc.ReliabilityInputError):
        pc.reliability_bins([0.5], [0.5]), "a realised outcome must be binary"


def test_every_reliability_bin_carries_its_sample_size(world):
    """Hard test 13: per-bin N, empty bins are null, sub-floor bins are insufficient."""

    conn, built = world
    artifact = _evaluate(conn, built)
    surface = _surface(artifact, "BRIER_P_START")
    reliability = surface["incumbent"]["figure"]["reliability"]
    bins = reliability["bins"]
    assert len(bins) == len(pc.RELIABILITY_BIN_EDGES) - 1 == 10
    assert all("n" in entry for entry in bins)
    assert sum(entry["n"] for entry in bins) == surface["population"]["scored"]
    populated = [entry for entry in bins if entry["n"] > 0]
    empty = [entry for entry in bins if entry["n"] == 0]
    assert populated and empty
    assert all(entry["status"] == pc.BIN_INSUFFICIENT_SAMPLE for entry in populated), (
        "every bin here is far below the declared floor, so none supports a calibration claim"
    )
    assert all(
        entry["observed_frequency"] is None
        and entry["stated_mean"] is None
        and entry["gap"] is None
        and entry["status"] == pc.BIN_EMPTY_SAMPLE
        for entry in empty
    )
    assert reliability["policy"]["min_bin_n"] == pc.RELIABILITY_MIN_BIN_N
    assert reliability["bins_over_floor"] == 0
    assert reliability["max_abs_gap_over_floor"] is None
    assert surface["diagnosis"]["status"] == ce.DIAGNOSIS_INSUFFICIENT
    # A bin does NOT become a calibration claim by being non-empty: the sub-floor
    # rule holds even where the gap is large (here -0.3 and +0.4).
    assert {entry["status"] for entry in populated} == {pc.BIN_INSUFFICIENT_SAMPLE}


def test_a_bin_over_the_floor_reports_insufficient_the_sub_floor_does_not_apply(world):
    """The declared floor decides: bins over it can carry a diagnosis."""

    conn, built = world
    artifact = _evaluate(conn, built)
    reliability = _surface(artifact, "BRIER_P_START")["incumbent"]["figure"]["reliability"]
    assert reliability["policy"]["basis"] == "declared disclosure floor, NOT a significance threshold"
    # Same bins, a lowered floor: now the large gaps are visible as gaps, and the
    # diagnosis (not the number) is what changes.
    figure = pc.reliability_bins(
        [row.probability for row in _scoring_rows(conn, built, metric="BRIER_P_START")[0]],
        [row.outcome for row in _scoring_rows(conn, built, metric="BRIER_P_START")[0]],
        floor=1,
    )
    over = [entry for entry in figure if entry.status == pc.BIN_OK]
    assert over and max(abs(float(entry.gap)) for entry in over) > ce.MATERIAL_CALIBRATION_GAP_TOLERANCE


# ---------------------------------------------------------------------------
# 14-19: expected value and coverage
# ---------------------------------------------------------------------------


def test_expected_points_bias_is_reported_per_component_not_only_in_aggregate():
    """Hard test 14: a compensating pair of component biases cannot hide."""

    conn = connect_database(":memory:")
    try:
        built = _small_world(conn, goal_bias=1.0, assist_bias=-1.0)
        artifact = _evaluate(conn, built)
    finally:
        conn.close()
    headline = artifact["expected_value"]["headline"]
    # The headline is event grain: model total_xpts summed over the event's
    # fixtures against the realised total_points, and the two component errors
    # cancel there.
    assert headline["status"] == ce.STATUS_OK
    assert headline["grain"] == wf.GRAIN_PLAYER_EVENT
    assert headline["bias"]["value"] == pytest.approx(0.0, abs=1e-9)
    assert _component(artifact, "goal_xpts")["bias"]["value"] == pytest.approx(1.0, abs=1e-9)
    assert _component(artifact, "assist_xpts")["bias"]["value"] == pytest.approx(-1.0, abs=1e-9)
    assert _component(artifact, "core_xpts")["bias"]["value"] == pytest.approx(0.0, abs=1e-9)
    # The sign convention is PE-2's, reused rather than restated.
    assert artifact["expected_value"]["bias_convention"] == sb.BIAS_CONVENTION
    assert "positive means overprediction" in sb.BIAS_CONVENTION
    assert "an aggregate bias can hide" in artifact["expected_value"]["aggregate_warning"]
    # Requirement: an aggregate figure alone is not accepted, so every declared
    # component is present with its own number.
    reported = {
        entry["component"] for entry in artifact["expected_value"]["component_diagnostics"]["components"]
    }
    assert reported == set(xpts_module.CORE_COMPONENTS) | set(xpts_module.SOFT_COMPONENTS) | {
        "soft_xpts",
        "core_xpts",
        "total_xpts",
    }


def test_every_component_figure_states_the_grain_it_is_persisted_at(world):
    """Hard test 15: components are persisted per player x fixture, and say so."""

    conn, built = world
    artifact = _evaluate(conn, built)
    diagnostics = artifact["expected_value"]["component_diagnostics"]
    assert diagnostics["grain"] == wf.GRAIN_PLAYER_FIXTURE
    for entry in diagnostics["components"]:
        assert entry["grain"] == wf.GRAIN_PLAYER_FIXTURE
        assert "player x fixture" in entry["grain_note"]
        assert entry["interpretation"]
        assert entry["realised_target"]
        assert entry["n"] > 0, "the reference world persists every component"
    assert artifact["identity"]["grains"]["expected_value_components"] == wf.GRAIN_PLAYER_FIXTURE
    assert artifact["identity"]["grains"]["expected_value_headline"] == wf.GRAIN_PLAYER_EVENT
    assert "not the event-grain claim" in artifact["expected_value"]["grain_policy"]


def test_a_component_whose_bias_is_not_causally_interpretable_carries_its_flag(world):
    """Hard test 14/15: an approximation carries the production flag beside the number."""

    conn, built = world
    artifact = _evaluate(conn, built)
    bonus = _component(artifact, "bonus_xpts")
    assert bonus["n"] > 0
    assert "BONUS_SOFT" in bonus["risk_flags"]
    assert "BONUS_SOFT" in bonus["non_causal_flags"]
    assert bonus["causally_interpretable"] is False
    assert bonus["bias"]["value"] == pytest.approx(0.0, abs=1e-9), "still reported, just not as causal"
    appearance = _component(artifact, "appearance_xpts")
    assert appearance["risk_flags"], "the flags the rows carried are shown even when none disqualify"
    assert appearance["non_causal_flags"] == []
    assert appearance["causally_interpretable"] is True
    goal = _component(artifact, "goal_xpts")
    assert "PENALTIES_EMBEDDED" in goal["non_causal_flags"]
    assert goal["causally_interpretable"] is False


def test_monte_carlo_quantile_coverage_is_reported_as_coverage(world):
    """Hard test 16: coverage, never a calibrated predictive interval."""

    conn, built = world
    artifact = _evaluate(conn, built)
    block = artifact["monte_carlo_coverage"]
    coverage = block["coverage"]
    assert {entry["metric"] for entry in coverage["metrics"]} == {
        "CENTRAL_50_QUANTILE_COVERAGE",
        "CENTRAL_80_QUANTILE_COVERAGE",
    }
    assert block["policy"]["claim"].startswith("quantile coverage only")
    assert "NOT a statement of complete distribution calibration" in block["policy"]["claim"]
    claims = block["claims"]
    assert claims["coverage_label"] == "QUANTILE COVERAGE, never a calibrated predictive interval"
    assert "shape, its tails" in claims["not_full_distribution_calibration"]
    assert coverage["population"]["scored"] == 20
    fifty = next(
        entry for entry in coverage["metrics"] if entry["metric"] == "CENTRAL_50_QUANTILE_COVERAGE"
    )
    # Every realised CORE lies inside [core-1, core+1] by construction, so the
    # central 50 band covers everything here; that is a coverage fact only.
    assert fifty["value"]["value"] == pytest.approx(1.0, abs=1e-9)
    assert fifty["nominal_coverage"] == 0.5
    assert fifty["value"]["n"] == 20


def test_coverage_is_fixture_grain_and_quantiles_are_never_summed(world):
    """Hard test 17: a double gameweek is two coverage observations, not one."""

    conn, built = world
    artifact = _evaluate(conn, built)
    coverage = artifact["monte_carlo_coverage"]
    assert coverage["grain"] == wf.GRAIN_PLAYER_FIXTURE
    assert coverage["claims"]["saturation"] == "quantiles are never summed into an event figure"
    assert coverage["policy"]["aggregation"] == "fixture grain; quantiles are never summed into an event figure"
    assert coverage["coverage"]["population"]["scored"] == 20, (
        "five players x two fixtures x two events: a double gameweek keeps every fixture"
    )
    # The headline expected-value figure at event grain covers a different
    # population, and the two are never conflated.
    assert artifact["expected_value"]["headline"]["n"] == 10
    assert artifact["identity"]["grains"]["quantile_coverage"] == wf.GRAIN_PLAYER_FIXTURE


def test_a_reversed_quantile_pair_fails_closed():
    """Hard test 18: bounds are never silently swapped."""

    conn = connect_database(":memory:")
    try:
        built = _small_world(conn, quantile_overrides={(10, 100): {"q25": 9.0, "q75": 1.0}})
        with pytest.raises(wm.MetricInputError) as error:
            _evaluate(conn, built)
    finally:
        conn.close()
    assert "exceeds upper quantile" in str(error.value)


def test_no_crps_claim_is_made_without_a_draw_level_artifact(world):
    """Hard test 19: no CRPS, because no draw-level artifact is persisted."""

    conn, built = world
    artifact = _evaluate(conn, built)
    assert artifact["monte_carlo_coverage"]["claims"]["crps"].startswith("NOT CLAIMED")
    assert sb.QUANTILE_POLICY["crps"].startswith("NOT_IMPLEMENTED")
    assert artifact["claims"]["crps"].startswith("NOT CLAIMED")
    metrics = {
        entry["metric"] for entry in artifact["monte_carlo_coverage"]["coverage"]["metrics"]
    }
    assert not any("CRPS" in name for name in metrics)
    assert "CRPS" not in json.dumps(artifact["monte_carlo_coverage"]["coverage"]["metrics"])


def test_the_headline_is_unreachable_rather_than_scored_when_no_event_is_final():
    """An item that cannot cover the comparison population says so."""

    conn = connect_database(":memory:")
    try:
        built = _small_world(conn)
        conn.execute("UPDATE events SET finished=0, data_checked=0 WHERE id=5")
        conn.commit()
        artifact = ce.evaluate(conn, artifact=built["artifact"], events=[5])
    finally:
        conn.close()
    assert artifact["identity"]["events_evaluated"] == []
    assert artifact["identity"]["events_excluded"][0]["state"] != "FINAL"
    headline = artifact["expected_value"]["headline"]
    assert headline["status"] == ce.STATUS_UNREACHABLE
    assert headline["reason"]
    assert headline["metrics"] == []
    for surface in artifact["probability_calibration"]["surfaces"]:
        assert surface["population"]["scored"] == 0
        assert surface["incumbent"]["figure"]["brier"]["value"] is None
        assert surface["incumbent"]["figure"]["observed_rate"] is None
        assert surface["population"]["excluded_events"][0]["event"] == 5
    assert artifact["terminal_state"]["state"] == ce.TERMINAL_OPEN


def test_a_blank_event_contributes_no_rows_and_is_not_a_zero_score():
    """Hard test 24, blank handling: a blank Gameweek is not a zero-score one."""

    conn = connect_database(":memory:")
    try:
        built = _small_world(conn, include_event_6=False)
        with conn:
            conn.execute(
                "INSERT INTO events(id, finished, data_checked, deadline_time, raw_json, updated_at)"
                " VALUES (7,1,1,'2026-09-05T17:30:00Z','{}','2026-09-05T17:30:00Z')"
            )
        # The certified anchor declares the blank event too; its xPts run simply
        # produced no row for it, which is a blank, not a zero.
        built["artifact"]["certified_bundles"]["7"] = {
            **built["artifact"]["certified_bundles"]["5"],
            "event": 7,
        }
        artifact = ce.evaluate(conn, artifact=built["artifact"], events=[5, 7])
        expected_rows = {
            surface["metric"]: len(_scoring_rows(conn, built, metric=surface["metric"])[0])
            for surface in artifact["probability_calibration"]["surfaces"]
        }
    finally:
        conn.close()
    assert artifact["identity"]["target_events"] == [5, 7]
    assert artifact["identity"]["events_evaluated"] == [5, 7]
    for surface in artifact["probability_calibration"]["surfaces"]:
        expected = expected_rows[surface["metric"]]
        assert surface["population"]["scored"] == expected, "event 7 contributes no rows"
        assert expected > 0
        assert surface["sample"]["target_events"] == 2
        assert surface["sample"]["target_events_with_observations"] == 1
    headline = artifact["expected_value"]["headline"]
    assert headline["population"]["excluded_by_status"][wf.TARGET_NO_FIXTURE] >= 1


# ---------------------------------------------------------------------------
# 20-21: the FPL assist mapping
# ---------------------------------------------------------------------------


def test_the_assist_mapping_coefficient_remains_one_without_causal_evidence(world):
    """Hard test 20: a plausible-sounding correction with no evidence is not a calibration."""

    conn, built = world
    artifact = _evaluate(conn, built)
    assist = artifact["assist_mapping"]
    config = xpts_module.XPtsConfig()
    assert config.fpl_assist_mapping_coefficient == 1.0
    assert assist["incumbent"]["coefficient"] == 1.0
    assert assist["incumbent"]["assist_mapping_calibrated"] is False
    assert assist["decision"]["outcome"] == ce.ASSIST_MAPPING_NO_CHANGE
    assert assist["decision"]["coefficient_after"] == 1.0
    assert assist["decision"]["promotion_performed"] is False
    assert assist["decision"]["candidate_coefficients"] == []
    assert assist["decision"]["reasons"], "leaving the constant alone is recorded with its reason"
    assert any("floors" in reason for reason in assist["decision"]["reasons"])
    assert assist["pooled_evidence"]["observations"] > 0


def test_assist_mapping_calibrated_only_alongside_a_declared_versioned_mapping():
    """Hard test 21: the flag stays truthful until a mapping is genuinely declared."""

    assert xpts_module.XPtsConfig().assist_mapping_calibrated is False
    source = inspect.getsource(xpts_module)
    assert re.search(
        r'if not config\.assist_mapping_calibrated:\s*\n\s+flags\.append\("FPL_ASSIST_MAPPING_UNCALIBRATED"\)',
        source,
    ), "the truthful disclosure is emitted while the mapping is uncalibrated"

    admissible = {"available": True, "in_bounds": True, "coefficient": 0.83}
    origins = [{"origin_event": 14, "fit": admissible}]
    pooled = {"observations": sb.MIN_OBSERVATIONS_FOR_DESCRIPTIVE, "events": sb.MIN_TARGET_EVENTS_FOR_DESCRIPTIVE}
    candidate = ce.assist_mapping_decision(origins, pooled)
    assert candidate["outcome"] == ce.ASSIST_MAPPING_CANDIDATE
    assert candidate["promotion_performed"] is False
    assert candidate["coefficient_after"] == 1.0, "PE-8 does not re-point the production constant"
    assert candidate["assist_mapping_calibrated_after"] is False
    assert candidate["flag_after"] == ce.ASSIST_MAPPING_FLAG
    assert candidate["candidate_coefficients"] == [0.83]
    # Below the floors the same admissible fit changes nothing at all.
    thin = ce.assist_mapping_decision(origins, {"observations": 10, "events": 2})
    assert thin["outcome"] == ce.ASSIST_MAPPING_NO_CHANGE
    assert thin["coefficient_after"] == 1.0
    # And no fit at all is a NO_CHANGE with its own reason.
    assert ce.assist_mapping_decision([], pooled)["outcome"] == ce.ASSIST_MAPPING_NO_CHANGE


# ---------------------------------------------------------------------------
# 23: the same-population gate
# ---------------------------------------------------------------------------


def test_a_population_mismatch_is_unreachable_not_scored(world):
    """Hard test 23: PE-2's gate decides, and a mismatch carries both sides."""

    gate = ce.comparison_gate([(5, 10, 100), (5, 11, 100)], [(5, 10, 100), (5, 12, 100)])
    assert gate["status"] == ce.STATUS_POPULATION_MISMATCH
    assert gate["same_population"] is False
    assert gate["population_digest"] is None, "no digest for a comparison that is not made"
    assert gate["incumbent_only_n"] == 1 and gate["arm_only_n"] == 1
    assert gate["incumbent_only_example"] == [[5, 11, 100]]
    assert gate["arm_only_example"] == [[5, 12, 100]]
    ok = ce.comparison_gate([(5, 10, 100)], [(5, 10, 100)])
    assert ok["status"] == ce.STATUS_OK
    assert ok["population_digest"] == wf.canonical_population_digest(
        [(5, 10, 100)], grain=wf.GRAIN_PLAYER_FIXTURE
    )

    conn, built = world
    artifact = _evaluate(conn, built)
    surface = _surface(artifact, "BRIER_P_START")
    challenger = surface["causal_challenger"]
    assert challenger["status"] == ce.STATUS_UNREACHABLE, "the reference world has too little history"
    assert challenger["candidate"] is None and challenger["incumbent_on_comparison_population"] is None
    assert challenger["excluded_by_reason"] == {pc.FIT_INSUFFICIENT_OBSERVATIONS: 20}
    assert challenger["population"]["n"] == 0
    assert challenger["population"]["population_digest"] is None
    assert "cannot cover the comparison population" in challenger["reason"]


def test_an_incumbent_and_its_causal_challenger_share_one_population_digest(big_world):
    """Hard test 23: identical digests, or no comparison."""

    conn, built = big_world
    artifact = _evaluate(conn, built)
    for surface in artifact["probability_calibration"]["surfaces"]:
        challenger = surface["causal_challenger"]
        assert challenger["status"] == ce.STATUS_OK, surface["metric"]
        assert challenger["gate"]["same_population"] is True
        assert challenger["population"]["population_digest"] == challenger["gate"]["population_digest"]
        incumbent_figure = challenger["incumbent_on_comparison_population"]
        candidate_figure = challenger["candidate"]
        assert incumbent_figure["n"] == candidate_figure["n"] == challenger["population"]["n"]
        assert challenger["population"]["n"] + challenger["rows_without_origin_transform"] == (
            surface["population"]["scored"]
        )
        assert incumbent_figure["brier"]["n"] == candidate_figure["brier"]["n"]
        assert len(challenger["versions_in_force"]) >= 1
        # Both arms cover the identical keys, and the digest is the fixture grain one.
        assert challenger["population"]["population_digest"] != surface["population"]["population_digest"], (
            "the comparison population is the challenger's causal coverage, not the whole surface"
        )


# ---------------------------------------------------------------------------
# 24: determinism, grain atomicity, frozen identities
# ---------------------------------------------------------------------------


def test_deterministic_repeat_and_row_order_invariance():
    """Hard test 24: same runs and outcomes, same bytes; insert order is irrelevant."""

    digests = []
    for reverse in (False, True):
        conn = connect_database(":memory:")
        try:
            built = _small_world(conn, reverse_inserts=reverse)
            artifact = _evaluate(conn, built)
            digests.append(ce.artifact_digest(artifact))
            if not reverse:
                again = _evaluate(conn, built)
                assert ce.canonical_bytes(again) == ce.canonical_bytes(artifact)
                assert ce.artifact_digest(again) == digests[0]
        finally:
            conn.close()
    assert digests[0] == digests[1], "the database row order must not change a single figure"
    assert digests[0].startswith("sha256:")


def test_a_double_gameweek_is_fixture_atomic_where_it_should_be_and_event_where_it_should_be(world):
    """Hard test 24: DGW handling, stated per grain."""

    conn, built = world
    artifact = _evaluate(conn, built)
    # Two fixtures per team per event, so every player has two fixture rows.
    for surface in artifact["probability_calibration"]["surfaces"]:
        assert surface["population"]["scored"] == (
            2 * ROWS_PER_EVENT if surface["metric"] != "BRIER_DEFCON" else 2 * (ROWS_PER_EVENT - 2)
        )
    assert artifact["expected_value"]["component_diagnostics"]["population"]["rows"] == 2 * ROWS_PER_EVENT
    # The headline is one observation per player per event: the fixtures are summed
    # by the source, and only by the source.
    assert artifact["expected_value"]["headline"]["n"] == 5 * len(built["events"])
    assert artifact["identity"]["grains"]["expected_value_headline"] == wf.GRAIN_PLAYER_EVENT
    assert "never one per fixture" in sb.TARGET_GRAIN_NOTE


def test_incumbent_model_and_calibration_identities_are_unchanged(world):
    """Hard test 24: PE-8 re-points nothing."""

    conn, built = world
    artifact = _evaluate(conn, built)
    frozen = artifact["identity"]["frozen_incumbents"]
    assert frozen["unchanged"] is True, frozen["mismatches"]
    assert frozen["mismatches"] == []
    assert frozen["entries"]["XPTS_MODEL_VERSION"]["actual"] == "xpts_v1.4.1"
    assert frozen["entries"]["MONTE_CARLO_MODEL_VERSION"]["actual"] == "mc_v1.3.0"
    assert frozen["entries"]["MINUTES_MODEL_VERSION"]["actual"] == "minutes_v1.8.0"
    assert frozen["entries"]["JOINT_MINUTES_MODEL_VERSION"]["actual"] == "minutes_v1.5.2"
    assert frozen["entries"]["TEAM_MODEL_VERSION"]["actual"] == "team_strength_v1.1.0"
    assert frozen["entries"]["PLAYER_RATE_MODEL_VERSION"]["actual"] == "player_rates_v1.0.0"
    assert frozen["entries"]["DEFCON_CALIBRATION_VERSION"]["actual"] == "defcon_platt_v1.0.0"
    assert artifact["terminal_state"]["promotion_performed"] is False
    assert artifact["terminal_state"]["incumbent_authoritative"] is True
    assert artifact["claims"]["promotion"].startswith("NOT PERFORMED")
    assert artifact["claims"]["ranking"].startswith("NONE")


def test_the_pe3_carry_forward_is_declared_unresolved(world):
    """Hard test 24-adjacent: an accepted limitation is carried, not quietly closed."""

    conn, built = world
    artifact = _evaluate(conn, built)
    carried = artifact["claims"]["continuous_proxy_tie_limitation"]
    assert carried.startswith("CARRIED, NOT RESOLVED")
    assert "CONTINUOUS_PROXY_TIE_LIMITATION" in carried
    assert any("CONTINUOUS_PROXY_TIE_LIMITATION" in line for line in artifact["limitations"])


# ---------------------------------------------------------------------------
# Terminal boundary
# ---------------------------------------------------------------------------


def test_the_terminal_state_is_open_on_an_insufficient_sample(world):
    """The correct state when the sample supports only a descriptive report."""

    conn, built = world
    artifact = _evaluate(conn, built)
    state = artifact["terminal_state"]
    assert state["state"] == ce.TERMINAL_OPEN
    assert state["promotion_performed"] is False
    assert any("INSUFFICIENT_FOR_STRONG_MODEL_SELECTION" in reason for reason in state["reasons"])
    assert state["terminal_boundary"].startswith("PE-8 ends at senior review")
    assert artifact["claims"]["promotion"].startswith("NOT PERFORMED")
    # The reference world's components are unbiased by construction, so the sample
    # size is the only thing standing between it and READY_FOR_MERGE -- which is
    # exactly why the declared floors are not lowered to make a green test.
    assert all(
        abs(float(entry["bias"]["value"])) <= ce.MATERIAL_COMPONENT_BIAS_TOLERANCE_POINTS
        for entry in artifact["expected_value"]["component_diagnostics"]["components"]
    )
    assert all(
        surface["diagnosis"]["status"] != ce.DIAGNOSIS_MISCALIBRATED
        for surface in artifact["probability_calibration"]["surfaces"]
    )


def test_a_material_expected_value_bias_opens_the_phase():
    """A diagnosed expected-value defect has no PE-8 transform, and is reported."""

    conn = connect_database(":memory:")
    try:
        built = _small_world(conn, goal_bias=1.0)
        artifact = _evaluate(conn, built)
    finally:
        conn.close()
    state = artifact["terminal_state"]
    assert _component(artifact, "goal_xpts")["bias"]["value"] == pytest.approx(1.0, abs=1e-9)
    assert artifact["expected_value"]["headline"]["bias"]["value"] > ce.MATERIAL_EV_BIAS_TOLERANCE_POINTS
    assert state["state"] == ce.TERMINAL_OPEN
    assert any("headline expected-value bias" in reason for reason in state["reasons"])
    assert any("component bias beyond the declared tolerance" in reason for reason in state["reasons"])
    # No probability transform is claimed to fix an expected-value defect.
    assert "no probability transform addresses an expected-value defect" in " ".join(state["reasons"])


def test_the_cli_needs_a_certification_artifact_and_reports_read_only(tmp_path):
    """The CLI is a read-only evidence surface and fails closed without authority."""

    import importlib.util

    from fpl_brain import four_gw_decision as fg

    script = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "evaluate_pe8_calibration.py"
    spec = importlib.util.spec_from_file_location("pe8_cli", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    with pytest.raises(fg.DecisionCertificationRequired):
        module.main(["--certification", str(tmp_path / "absent.json")])


# ---------------------------------------------------------------------------
# The declared floors, the causal fit and the terminal boundary at scale
# ---------------------------------------------------------------------------


def test_the_terminal_state_is_ready_for_merge_on_a_calibrated_population(big_world):
    """With the declared floors met and no defect, the phase is READY_FOR_MERGE."""

    conn, built = big_world
    artifact = _evaluate(conn, built)
    assert artifact["expected_value"]["headline"]["n"] == len(BIG_PLAYERS) * len(BIG_EVENTS) == 2000
    assert artifact["expected_value"]["headline"]["bias"]["value"] == pytest.approx(0.0, abs=1e-9)
    for surface in artifact["probability_calibration"]["surfaces"]:
        sample = surface["sample"]
        assert sample["target_events"] == len(BIG_EVENTS) == 10
        assert sample["observations"] == 4000
        assert sample["sample_interpretation"] == sb.SAMPLE_DESCRIPTIVE_ONLY
        reliability = surface["incumbent"]["figure"]["reliability"]
        assert reliability["bins_over_floor"] == 1
        assert reliability["max_abs_gap_over_floor"] <= ce.MATERIAL_CALIBRATION_GAP_TOLERANCE
        assert surface["diagnosis"]["status"] == ce.DIAGNOSIS_NO_MATERIAL_DEFECT
        assert surface["causal_challenger"]["status"] == ce.STATUS_OK
    for entry in artifact["expected_value"]["component_diagnostics"]["components"]:
        assert entry["n"] == 4000
        assert abs(float(entry["bias"]["value"])) <= ce.MATERIAL_COMPONENT_BIAS_TOLERANCE_POINTS, (
            f"{entry['component']} is biased in a world designed to be unbiased"
        )
    state = artifact["terminal_state"]
    assert state["state"] == ce.TERMINAL_READY_FOR_MERGE
    assert state["reasons"] == []
    assert state["promotion_performed"] is False
    assert artifact["assist_mapping"]["decision"]["outcome"] == ce.ASSIST_MAPPING_NO_CHANGE


def test_a_diagnosed_miscalibration_opens_the_phase_without_a_promotion(tmp_path):
    """A diagnosed defect with no admissible transform is OPEN, never a silent pass."""

    conn, built = _single_use_world(tmp_path, start_share_a=0.30)
    try:
        artifact = _evaluate(conn, built)
    finally:
        conn.close()
    surface = _surface(artifact, "BRIER_P_START")
    # Group A states 0.58 and realises 0.30; group B states 0.52 and realises 0.50,
    # so the pool states 0.55 and realises 0.40.
    assert surface["incumbent"]["figure"]["observed_rate"] == pytest.approx(0.40, abs=1e-9)
    assert surface["diagnosis"]["status"] == ce.DIAGNOSIS_MISCALIBRATED
    assert surface["diagnosis"]["max_abs_gap_over_floor"] == pytest.approx(0.15, abs=1e-9)
    # The realised rate now falls as the stated probability rises, so no monotone
    # transform is admissible: the challenger says so rather than fitting one.
    assert surface["causal_challenger"]["status"] == ce.STATUS_UNREACHABLE
    assert pc.FIT_NON_MONOTONE in surface["causal_challenger"]["exclusion_reasons"]
    state = artifact["terminal_state"]
    assert state["state"] == ce.TERMINAL_OPEN
    assert any("BRIER_P_START" in reason and "material calibration gap" in reason for reason in state["reasons"])
    assert state["promotion_performed"] is False
    assert "incumbent stays authoritative" in artifact["claims"]["promotion"]


def test_a_causal_assist_mapping_fit_is_a_candidate_that_still_changes_nothing(tmp_path):
    """Hard tests 20 and 21 at scale: an admissible fit is evidence, not a change."""

    conn, built = _single_use_world(tmp_path, expected_xa=0.05, assist_every=20)
    try:
        artifact = _evaluate(conn, built)
        decision = artifact["assist_mapping"]["decision"]
        pooled = artifact["assist_mapping"]["pooled_evidence"]
    finally:
        conn.close()
    assert pooled["observations"] == 4000 and pooled["events"] == 10
    assert pooled["expected_assists_total"] == pytest.approx(200.0, abs=1e-6)
    assert pooled["realised_assists_total"] == pytest.approx(200.0, abs=1e-6)
    assert decision["outcome"] == ce.ASSIST_MAPPING_CANDIDATE
    assert decision["promotion_performed"] is False
    assert decision["coefficient_after"] == 1.0
    assert decision["assist_mapping_calibrated_after"] is False
    assert decision["flag_after"] == ce.ASSIST_MAPPING_FLAG
    # The fitted coefficient is the ratio of sums over a strictly earlier basis.
    coefficients = decision["candidate_coefficients"]
    assert coefficients and all(abs(value - 1.0) < 0.05 for value in coefficients)
    assert artifact["assist_mapping"]["incumbent"]["coefficient"] == 1.0
    assert xpts_module.XPtsConfig().assist_mapping_calibrated is False
    assert any("belongs to review" in reason for reason in artifact["terminal_state"]["reasons"])
    assert artifact["terminal_state"]["state"] == ce.TERMINAL_OPEN


def test_a_later_realised_outcome_leaves_every_earlier_transform_untouched(big_world, tmp_path):
    """Hard tests 2 and 23 at scale: the last event's outcomes move nothing earlier."""

    conn, built = big_world
    reference = _evaluate(conn, built)

    conn_alt, built_alt = _single_use_world(tmp_path, alt_last_event=True)
    try:
        altered = _evaluate(conn_alt, built_alt)
    finally:
        conn_alt.close()

    last_event = BIG_EVENTS[-1]
    for reference_surface, altered_surface in zip(
        reference["probability_calibration"]["surfaces"],
        altered["probability_calibration"]["surfaces"],
    ):
        assert reference_surface["metric"] == altered_surface["metric"]
        reference_fits = {
            int(entry["origin_event"]): (
                None if entry["fit"]["spec"] is None else entry["fit"]["spec"]["version"]
            )
            for entry in reference_surface["causal_challenger"]["origins"]
        }
        altered_fits = {
            int(entry["origin_event"]): (
                None if entry["fit"]["spec"] is None else entry["fit"]["spec"]["version"]
            )
            for entry in altered_surface["causal_challenger"]["origins"]
        }
        assert reference_fits == altered_fits, (
            f"{reference_surface['metric']}: rewriting event {last_event}'s outcomes changed a transform"
        )
        assert reference_fits[last_event] is not None, "the last origin has a fitted transform"
        # The basis of the last origin is every earlier event, and never its own.
        last_basis = next(
            entry["fit"]["basis"]
            for entry in altered_surface["causal_challenger"]["origins"]
            if int(entry["origin_event"]) == last_event
        )
        assert last_basis["events"] == [int(event) for event in BIG_EVENTS[:-1]]
        assert last_basis["strictly_before_origin"] is True
        # The last event's SCORED figure does move: causality, not a frozen file.
        assert (
            reference_surface["incumbent"]["figure"]["observed_rate"]
            != altered_surface["incumbent"]["figure"]["observed_rate"]
        )
    assert ce.artifact_digest(reference) != ce.artifact_digest(altered)


def test_the_out_of_scope_surfaces_are_declared_rather_than_silently_skipped(world):
    """PE-8 must not widen onto a surface the contract excludes, and must say so."""

    conn, built = world
    artifact = _evaluate(conn, built)
    excluded = {entry["surface"]: entry["reason"] for entry in artifact["excluded_surfaces"]}
    assert set(excluded) == {
        "p_goal",
        "p_assist",
        "*_xpts components",
        "monte_carlo draws / total-points distribution",
    }
    assert all(reason for reason in excluded.values())
    assert artifact["claims"]["probability_outside_unit_interval"].startswith("fails closed")
    # No expected-points component is scored as a probability anywhere in the artifact.
    scored_metrics = {
        surface["metric"] for surface in artifact["probability_calibration"]["surfaces"]
    }
    assert scored_metrics == set(SURFACES)
    assert not any("xpts" in metric for metric in scored_metrics)
    serialised = json.dumps(artifact["probability_calibration"])
    assert "BRIER_GOAL" not in serialised and "BRIER_ASSIST" not in serialised
    # And the declared grains are carried by token, so a reader never has to infer one.
    grains = artifact["identity"]["grains"]
    assert grains["probability_surfaces"] == wf.GRAIN_PLAYER_FIXTURE
    assert grains["expected_value_headline"] == wf.GRAIN_PLAYER_EVENT
    assert artifact["identity"]["walk_forward_identity"] == wf.WALK_FORWARD_VERSION
    assert artifact["identity"]["missing_data_policy_version"] == wf.MISSING_DATA_POLICY_VERSION
    assert artifact["identity"]["sample_policy_version"] == sb.SAMPLE_POLICY_VERSION
    assert artifact["identity"]["metric_policy_version"] == wm.METRIC_POLICY_VERSION


def test_a_transform_has_a_mandate_only_from_a_diagnosed_defect(big_world, tmp_path):
    """Transform rule 1: a transform corrects a MEASURED defect, never speculatively."""

    conn, built = big_world
    artifact = _evaluate(conn, built)
    for surface in artifact["probability_calibration"]["surfaces"]:
        mandate = surface["transform_mandate"]
        assert mandate["mandate"] == "NO_DEFECT_DIAGNOSED"
        assert mandate["promotable_by_pe8"] is False
        assert mandate["transform_status"] == ce.STATUS_OK, (
            "the transform is still fitted and reported: it is the sensitivity of a clean surface"
        )
        assert "descriptive sensitivity" in mandate["note"]
        assert "promotes none" in mandate["rule"]

    conn_alt, built_alt = _single_use_world(tmp_path, start_share_a=0.30)
    try:
        diagnosed = _evaluate(conn_alt, built_alt)
    finally:
        conn_alt.close()
    surface = _surface(diagnosed, "BRIER_P_START")
    mandate = surface["transform_mandate"]
    assert mandate["mandate"] == "DIAGNOSED_DEFECT"
    assert mandate["promotable_by_pe8"] is False
    assert mandate["transform_status"] == ce.STATUS_UNREACHABLE
    assert "still not promoted" in mandate["note"]

    # The ranking-behaviour statement the contract asks for, in one place.
    ranking = pc.policy()["ranking_behaviour"]
    assert ranking["within_event_ordering"].startswith("preserved")
    assert "DOES change the expected value" in ranking["expected_value_effect"]
    assert ranking["production_use"].startswith("NONE")
    assert pc.policy()["mandate"].startswith("a transform exists to correct a MEASURED miscalibration")


def test_the_cli_report_prints_the_evidence_it_is_given(world, tmp_path, capsys):
    """The report path is exercised on a real artifact, including the JSON output."""

    import importlib.util

    script = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "evaluate_pe8_calibration.py"
    spec = importlib.util.spec_from_file_location("pe8_cli_report", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    conn, built = world
    artifact = _evaluate(conn, built)
    out = tmp_path / "pe8.json"
    assert module.report(artifact, json_out=str(out)) == 0
    printed = capsys.readouterr().out
    assert "terminal state" in printed
    assert "BRIER_P_START" in printed
    assert "defcon_platt_v1.0.0" in printed
    assert "assist mapping" in printed
    assert ce.artifact_digest(artifact) in printed
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["schema"] == ce.PE8_SCHEMA_VERSION
    assert written["terminal_state"] == artifact["terminal_state"]
    assert ce.canonical_bytes(written) == ce.canonical_bytes(artifact)


def test_every_reported_figure_carries_its_sample_size_and_a_floor_label(big_world, world):
    """Every figure states its N, and a figure below a declared floor says so."""

    conn, built = big_world
    artifact = _evaluate(conn, built)
    headline = artifact["expected_value"]["headline"]
    assert headline["sample"]["observations"] == headline["n"] == 2000
    assert headline["sample"]["target_events_with_observations"] == len(BIG_EVENTS)
    assert headline["sample"]["sample_interpretation"] == sb.SAMPLE_DESCRIPTIVE_ONLY
    assert headline["sample"]["policy"]["policy_version"] == sb.SAMPLE_POLICY_VERSION
    for entry in artifact["expected_value"]["component_diagnostics"]["components"]:
        assert entry["sample"]["observations"] == entry["n"] == 4000
        assert entry["sample"]["sample_interpretation"] == sb.SAMPLE_DESCRIPTIVE_ONLY
    coverage = artifact["monte_carlo_coverage"]["sample"]
    assert coverage["observations"] == 4000
    assert coverage["sample_interpretation"] == sb.SAMPLE_DESCRIPTIVE_ONLY
    assert "same declared disclosure floor" in coverage["note"]
    assert "NOT a significance threshold" in headline["sample"]["policy"]["basis"]

    # The reference world is below the floor, and says so on every surface rather
    # than presenting a thin number as a result.
    small_conn, small_built = world
    small = _evaluate(small_conn, small_built)
    for surface in small["probability_calibration"]["surfaces"]:
        assert surface["sample"]["sample_interpretation"] == sb.SAMPLE_INSUFFICIENT
        assert surface["sample"]["observations"] == surface["population"]["scored"]
    for entry in small["expected_value"]["component_diagnostics"]["components"]:
        assert entry["sample"]["sample_interpretation"] == sb.SAMPLE_INSUFFICIENT
    assert small["monte_carlo_coverage"]["sample"]["sample_interpretation"] == sb.SAMPLE_INSUFFICIENT
