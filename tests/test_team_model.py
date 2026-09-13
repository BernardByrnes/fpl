"""Team Attack/Defence Model v1 tests: coherence, shrinkage, grain, no-lookahead."""

from __future__ import annotations

import math

from fpl_brain import team_model
from fpl_brain.database import connect_database
from fpl_brain.models import (
    EventRecord,
    FixtureRecord,
    PlayerGameweekRecord,
    PlayerRecord,
    PositionRecord,
    TeamRecord,
)

CUTOFF = "2026-09-10T12:00:00Z"
EVENT_DEADLINE = "2026-09-12T12:30:00Z"

# Completed season: team 1 plays home/away/home; team 6 has a single match.
BASE_FIXTURES = [
    {"id": 1, "event": 1, "h": 1, "a": 2, "kickoff": "2026-08-22T14:00:00Z"},
    {"id": 2, "event": 1, "h": 3, "a": 5, "kickoff": "2026-08-22T16:00:00Z"},
    {"id": 3, "event": 2, "h": 5, "a": 1, "kickoff": "2026-08-29T14:00:00Z"},
    {"id": 4, "event": 2, "h": 2, "a": 3, "kickoff": "2026-08-29T16:00:00Z"},
    {"id": 5, "event": 3, "h": 1, "a": 6, "kickoff": "2026-09-05T14:00:00Z"},
    {"id": 6, "event": 3, "h": 2, "a": 5, "kickoff": "2026-09-05T16:00:00Z"},
]
# Planning event 4: team 1 double gameweek; team 6 blank.
PLANNING_FIXTURES = [
    {"id": 7, "event": 4, "h": 1, "a": 2, "kickoff": "2026-09-13T14:00:00Z", "finished": 0, "started": 0},
    {"id": 8, "event": 4, "h": 3, "a": 5, "kickoff": "2026-09-13T16:00:00Z", "finished": 0, "started": 0},
    {"id": 9, "event": 4, "h": 1, "a": 3, "kickoff": "2026-09-14T14:00:00Z", "finished": 0, "started": 0},
]
TEAMS = (1, 2, 3, 5, 6)
CARRIER = {team: 100 + team for team in TEAMS}


def _world(conn, *, xg=None, xgc=None, extra_fixtures=(), gameweeks_extra=()):
    """Seed a small league whose team-fixture xG is carried by one player/team."""

    from fpl_brain import repositories as repo

    xg = xg or {}
    xgc = xgc or {}
    fixtures = BASE_FIXTURES + PLANNING_FIXTURES + list(extra_fixtures)
    with conn:
        repo.upsert_teams(conn, [TeamRecord(id=team, name=f"Team {team}") for team in TEAMS])
        repo.upsert_positions(
            conn,
            [PositionRecord(id=pos, singular_name_short=name)
             for pos, name in ((1, "GKP"), (2, "DEF"), (3, "MID"), (4, "FWD"))],
        )
        repo.upsert_players(
            conn,
            [PlayerRecord(id=CARRIER[team], web_name=f"C{team}", full_name=f"Carrier {team}", team_id=team, element_type=3)
             for team in TEAMS],
        )
        repo.upsert_events(
            conn,
            [EventRecord(id=e, finished=1 if e <= 3 else 0, data_checked=1 if e <= 3 else 0,
                         deadline_time=EVENT_DEADLINE if e == 4 else f"2026-0{e}-01T12:00:00Z", raw_json={})
             for e in range(1, 5)],
        )
        repo.upsert_fixtures(
            conn,
            [FixtureRecord(id=fx["id"], event=fx["event"], team_h=fx["h"], team_a=fx["a"],
                           finished=fx.get("finished", 1), started=fx.get("started", 1),
                           team_h_score=fx.get("hs", 1 if fx.get("finished", 1) else None),
                           team_a_score=fx.get("as", 1 if fx.get("finished", 1) else None),
                           kickoff_time=fx["kickoff"], raw_json={})
             for fx in fixtures],
        )
        rows = []
        for fx in fixtures:
            if not fx.get("finished", 1):
                continue
            hxg, axg = xg.get(fx["id"], (1.6, 1.2))
            hxgc, axgc = xgc.get(fx["id"], (axg, hxg))
            rows.append(PlayerGameweekRecord(player_id=CARRIER[fx["h"]], event=fx["event"], fixture_id=fx["id"],
                                             was_home=1, minutes=90, starts=1,
                                             expected_goals=hxg, expected_goals_conceded=hxgc,
                                             source="element_summary", raw_json={}))
            rows.append(PlayerGameweekRecord(player_id=CARRIER[fx["a"]], event=fx["event"], fixture_id=fx["id"],
                                             was_home=0, minutes=90, starts=1,
                                             expected_goals=axg, expected_goals_conceded=axgc,
                                             source="element_summary", raw_json={}))
        rows.extend(gameweeks_extra)
        repo.upsert_player_gameweeks(conn, rows)
    return fixtures


def _project(conn, config=None, cutoff=CUTOFF):
    rows, _meta = team_model.build_team_fixture_projections(conn, 4, cutoff, config)
    return rows


def _find(rows, fixture_id, team_id):
    return next(row for row in rows if row["fixture_id"] == fixture_id and row["team_id"] == team_id)


def _signature(rows):
    return sorted(
        (r["fixture_id"], r["team_id"], r["expected_goals_for"], r["expected_goals_against"]) for r in rows
    )


# 1 ------------------------------------------------------------------


def test_probabilities_valid_and_sum_to_one():
    conn = connect_database(":memory:")
    _world(conn)
    for row in _project(conn):
        probs = [row["p_goals_0"], row["p_goals_1"], row["p_goals_2_plus"], row["p_clean_sheet"]]
        assert all(0.0 <= p <= 1.0 for p in probs)
        assert abs(row["p_goals_0"] + row["p_goals_1"] + row["p_goals_2_plus"] - 1.0) < 1e-5


# 2 ------------------------------------------------------------------


def test_lambda_positive_and_finite():
    conn = connect_database(":memory:")
    _world(conn)
    for row in _project(conn):
        for key in ("expected_goals_for", "expected_goals_against"):
            assert math.isfinite(row[key]) and row[key] > 0.0


# 3 ------------------------------------------------------------------


def test_stronger_attack_does_not_reduce_lambda_for():
    base = connect_database(":memory:")
    _world(base)
    strong = connect_database(":memory:")
    # Team 1 raises its own xG in every completed match (home and away).
    _world(strong, xg={1: (2.6, 1.2), 3: (1.2, 2.6), 5: (2.6, 1.2)})
    base_row = _find(_project(base), 7, 1)
    strong_row = _find(_project(strong), 7, 1)
    assert strong_row["expected_goals_for"] >= base_row["expected_goals_for"]
    assert strong_row["attack_rating"] >= base_row["attack_rating"]


# 4 ------------------------------------------------------------------


def test_weaker_opponent_defence_does_not_reduce_lambda_for():
    base = connect_database(":memory:")
    _world(base)
    weak = connect_database(":memory:")
    # Team 2 (the opponent in fixture 7) concedes more in every match, while
    # its own attack is unchanged (fixtures 1, 4 and 6 opponent xG only).
    _world(weak, xg={1: (2.6, 1.2), 4: (1.2, 1.6), 6: (1.2, 1.6)})
    base_row = _find(_project(base), 7, 1)
    weak_row = _find(_project(weak), 7, 1)
    assert weak_row["opponent_defence_rating"] >= base_row["opponent_defence_rating"]
    assert weak_row["expected_goals_for"] >= base_row["expected_goals_for"]


# 5 ------------------------------------------------------------------


def test_home_advantage_affects_expected_direction():
    conn = connect_database(":memory:")
    _world(conn)
    params = team_model.fit_team_strength(conn, 4, CUTOFF)
    assert params["home_advantage"] > 0.0
    lam_home = team_model.lambda_for(1, 2, "home", params)
    lam_away = team_model.lambda_for(1, 2, "away", params)
    assert lam_home > lam_away


# 6 ------------------------------------------------------------------


def test_stronger_opponent_attack_does_not_raise_clean_sheet():
    base = connect_database(":memory:")
    _world(base)
    strong_opp = connect_database(":memory:")
    # Team 2 (opponent) raises its attack: xG in fixtures 1 (away), 4 and 6 (home).
    _world(strong_opp, xg={1: (1.6, 2.6), 4: (2.6, 1.2), 6: (2.6, 1.2)})
    base_row = _find(_project(base), 7, 1)
    opp_row = _find(_project(strong_opp), 7, 1)
    assert opp_row["expected_goals_against"] >= base_row["expected_goals_against"]
    assert opp_row["p_clean_sheet"] <= base_row["p_clean_sheet"] + 1e-12


# 7 ------------------------------------------------------------------


def test_clean_sheet_equals_opponent_p0_from_same_latent_lambda():
    conn = connect_database(":memory:")
    _world(conn)
    rows = _project(conn)
    for row in rows:
        opponent = _find(rows, row["fixture_id"], row["opponent_id"])
        assert abs(row["p_clean_sheet"] - opponent["p_goals_0"]) < 1e-9


# 8 ------------------------------------------------------------------


def test_double_gameweek_produces_distinct_projection_per_fixture():
    conn = connect_database(":memory:")
    _world(conn)
    team1 = [row for row in _project(conn) if row["team_id"] == 1]
    fixtures = sorted(row["fixture_id"] for row in team1)
    assert fixtures == [7, 9]
    assert len({row["fixture_id"] for row in team1}) == 2


# 9 ------------------------------------------------------------------


def test_blank_gameweek_produces_no_projection():
    conn = connect_database(":memory:")
    _world(conn)
    assert [row for row in _project(conn) if row["team_id"] == 6] == []


# 10 -----------------------------------------------------------------


def test_price_changes_do_not_affect_team_projections():
    conn = connect_database(":memory:")
    _world(conn)
    before = _signature(_project(conn))
    from fpl_brain import repositories as repo
    from fpl_brain.models import PlayerSnapshotRecord

    with conn:
        run = repo.create_fetch_run(conn, "fetch_fpl")
        repo.insert_snapshots(conn, [
            PlayerSnapshotRecord(player_id=CARRIER[t], captured_at="2026-09-10T08:00:00Z", now_cost=200, raw_json={})
            for t in TEAMS
        ], run)
    assert _signature(_project(conn)) == before


# 11 -----------------------------------------------------------------


def test_manager_ownership_does_not_affect_team_projections():
    conn = connect_database(":memory:")
    _world(conn)
    before = _signature(_project(conn))
    with conn:
        conn.execute(
            """INSERT INTO squad_picks(entry_id, event, player_id, position, synced_at, raw_json)
               VALUES (241392, 4, ?, 1, '2026-09-10T08:00:00Z', '{}')""",
            (CARRIER[1],),
        )
        conn.execute(
            """INSERT INTO manager_state(entry_id, captured_at, bank, raw_json)
               VALUES (241392, '2026-09-10T08:00:00Z', 0, '{}')"""
        )
    assert _signature(_project(conn)) == before


# 12 -----------------------------------------------------------------


def test_no_lookahead_excludes_future_and_post_cutoff_evidence():
    conn = connect_database(":memory:")
    _world(conn)
    before = _signature(_project(conn))
    # A completed fixture in the planning event itself and a fixture whose
    # kickoff is after the cutoff must both be ignored.
    extra = [
        {"id": 90, "event": 4, "h": 1, "a": 5, "kickoff": "2026-09-12T14:00:00Z", "finished": 1, "started": 1},
        {"id": 91, "event": 3, "h": 3, "a": 6, "kickoff": "2026-09-11T14:00:00Z", "finished": 1, "started": 1},
    ]
    _world(conn, extra_fixtures=extra)
    # Fixture 90 leaks nothing (event == planning); fixture 91 is post-cutoff.
    after = {(r[0], r[1]): r for r in _signature(_project(conn))}
    before_map = {(r[0], r[1]): r for r in before}
    for key, value in before_map.items():
        assert after[key] == value


# 13 -----------------------------------------------------------------


def test_deterministic_for_same_inputs():
    conn = connect_database(":memory:")
    _world(conn)
    first = _project(conn)
    second = _project(conn)
    strip = lambda rows: [{k: v for k, v in row.items() if k != "generated_at"} for row in rows]
    assert strip(first) == strip(second)


# 14 -----------------------------------------------------------------


def test_config_change_changes_config_hash():
    default = team_model.TeamStrengthConfig()
    changed = team_model.TeamStrengthConfig(current_match_half_life=6.0)
    assert default.config_hash() != changed.config_hash()


# 15 -----------------------------------------------------------------


def test_naive_baseline_stored_separately_without_team_signal():
    conn = connect_database(":memory:")
    _world(conn)
    model_rows = _project(conn)
    naive_rows = team_model.build_naive_team_projections(conn, 4, CUTOFF)
    assert len(naive_rows) == len(model_rows) == 2 * len(PLANNING_FIXTURES)
    for row in naive_rows:
        assert row["attack_rating"] is None and row["opponent_defence_rating"] is None
        assert row["provenance"]["baseline_kind"] == "LEAGUE_AVERAGE_VENUE"
    # Same home lambda for every home side (no team-specific signal).
    homes = {row["expected_goals_for"] for row in naive_rows if row["venue"] == "home"}
    assert len(homes) == 1


# 16 -----------------------------------------------------------------


def test_extreme_small_sample_remains_strongly_shrunk():
    conn = connect_database(":memory:")
    # Team 6 has exactly one completed match (with extreme xG); give it an
    # event-4 fixture so a projection exists to inspect.
    _world(
        conn,
        xg={5: (1.6, 5.0)},
        extra_fixtures=[{"id": 11, "event": 4, "h": 6, "a": 2, "kickoff": "2026-09-13T18:00:00Z", "finished": 0, "started": 0}],
    )
    params = team_model.fit_team_strength(conn, 4, CUTOFF)
    assert params["team_match_counts"][6] == 1
    assert abs(params["attack"][6]) < 0.5
    assert "HIGH_REGULARISATION_DOMINANCE" in _find(_project(conn), 11, 6)["risk_flags"]


# 17 -----------------------------------------------------------------


def test_xg_source_reconciliation_surfaces_discrepancy():
    conn = connect_database(":memory:")
    # In fixture 1 the away side's xGC is fabricated 0.9 above the home side's
    # true xG sum, so the home team's independent check must flag it.
    _world(conn, xgc={1: (1.2, 2.5)})
    home_row = _find(_project(conn), 7, 1)
    assert "XG_XGC_RECONCILIATION_DISCREPANCY" in home_row["risk_flags"]
    assert home_row["provenance"]["reconciliation_status"] == "DISCREPANCY"
