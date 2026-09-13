"""Player Attacking-Rate Shrinkage v1 tests: priors, ESS, role, grain, immutability."""

from __future__ import annotations

import pytest

from fpl_brain import analytics, player_rates
from fpl_brain.database import connect_database
from fpl_brain.models import (
    EventRecord,
    FixtureRecord,
    PlayerGameweekRecord,
    PlayerRecord,
    PlayerSeasonHistoryRecord,
    PositionRecord,
    TeamRecord,
)

CUTOFF = "2026-09-10T12:00:00Z"
DEADLINE = "2026-09-12T12:30:00Z"
KICKOFFS = ["2026-08-22T14:00:00Z", "2026-08-29T14:00:00Z", "2026-09-05T14:00:00Z"]
COMPONENT = player_rates.COMPONENT_XG


def _seed(conn, *, players, histories=(), fixtures=(), gameweeks=(), notes=()):
    """Seed players, prior-season rows, fixtures, gameweeks and scouting notes."""

    from fpl_brain import repositories as repo

    teams = sorted(
        {team for team, _pos in players.values()}
        | {team for _fid, _event, h, a, _kickoff, _finished in fixtures for team in (h, a)}
    )
    with conn:
        repo.upsert_teams(conn, [TeamRecord(id=team, name=f"Team {team}") for team in teams])
        repo.upsert_positions(
            conn,
            [PositionRecord(id=pos, singular_name_short=name)
             for pos, name in ((1, "GKP"), (2, "DEF"), (3, "MID"), (4, "FWD"))],
        )
        repo.upsert_players(
            conn,
            [PlayerRecord(id=pid, web_name=f"P{pid}", full_name=f"Player {pid}", team_id=team, element_type=pos)
             for pid, (team, pos) in players.items()],
        )
        repo.upsert_events(
            conn,
            [EventRecord(id=e, finished=1 if e <= 3 else 0, data_checked=1 if e <= 3 else 0,
                         deadline_time=DEADLINE if e == 4 else f"2026-0{e}-01T12:00:00Z", raw_json={})
             for e in range(1, 5)],
        )
        if fixtures:
            repo.upsert_fixtures(
                conn,
                [FixtureRecord(id=fid, event=event, team_h=h, team_a=a, finished=finished, started=finished,
                               kickoff_time=kickoff, team_h_score=1 if finished else None,
                               team_a_score=1 if finished else None, raw_json={})
                 for fid, event, h, a, kickoff, finished in fixtures],
            )
        if histories:
            repo.upsert_player_season_histories(
                conn,
                [PlayerSeasonHistoryRecord(player_id=pid, season_name=season, minutes=minutes,
                                           starts=minutes // 90, raw_json={"expected_goals": xg, "expected_assists": xa})
                 for pid, season, minutes, xg, xa in histories],
            )
        if gameweeks:
            repo.upsert_player_gameweeks(
                conn,
                [PlayerGameweekRecord(player_id=pid, event=event, fixture_id=fid, minutes=minutes,
                                      starts=1 if minutes and minutes >= 45 else 0,
                                      expected_goals=xg, expected_assists=xa,
                                      source="element_summary", raw_json={})
                 for pid, event, fid, minutes, xg, xa in gameweeks],
            )
        for pid, key, value_text, observed_at, expires_at in notes:
            conn.execute(
                """INSERT INTO scouting_notes(player_id, key, value_text, confidence, observed_at, expires_at, created_at)
                   VALUES (?,?,?, 'medium', ?, ?, ?)""",
                (pid, key, value_text, observed_at, expires_at, observed_at),
            )


def _history(pid, minutes, xg, xa, season="2025/26"):
    return (pid, season, minutes, xg, xa)


def _row(conn, player_id, component=COMPONENT, config=None):
    config = config or player_rates.PlayerRatesConfig()
    rows = player_rates.build_player_rate_projections(conn, 4, CUTOFF, config)
    return next(row for row in rows if row["player_id"] == player_id and row["component"] == component)


def _strip(rows):
    return [{k: v for k, v in row.items() if k != "generated_at"} for row in rows]


# 1 ------------------------------------------------------------------


def test_zero_or_minimal_current_minutes_stays_near_prior():
    conn = connect_database(":memory:")
    _seed(conn, players={10: (1, 3)}, histories=[_history(10, 2000, 20.0, 10.0)],
          fixtures=[(1, 1, 1, 2, KICKOFFS[0], 1)], gameweeks=[(10, 1, 1, 10, 0.5, 0.1)])
    row = _row(conn, 10)
    assert abs(row["posterior_mean"] - row["prior_mean"]) < 0.2 * abs(row["current_rate"] - row["prior_mean"])


# 2 ------------------------------------------------------------------


def test_more_current_minutes_moves_posterior_toward_current():
    players = {10: (1, 3), 11: (1, 3)}
    fixtures = [(fid, (fid + 1) // 2, 1, 2, KICKOFFS[0], 1) for fid in range(1, 7)]
    gameweeks = [(10, 1, 1, 90, 1.35, 0.1)] + [(11, e, e, 90, 1.35, 0.1) for e in range(1, 6)]
    conn = connect_database(":memory:")
    _seed(conn, players=players, histories=[_history(10, 2000, 20.0, 10.0), _history(11, 2000, 20.0, 10.0)],
          fixtures=fixtures, gameweeks=gameweeks)
    few, many = _row(conn, 10), _row(conn, 11)
    assert many["current_minutes"] > few["current_minutes"]
    assert abs(many["posterior_mean"] - many["current_rate"]) < abs(few["posterior_mean"] - few["current_rate"])


# 3 ------------------------------------------------------------------


def test_extreme_short_sample_is_strongly_shrunk():
    conn = connect_database(":memory:")
    _seed(conn, players={10: (1, 3)}, histories=[_history(10, 2000, 6.67, 3.0)],
          fixtures=[(1, 1, 1, 2, KICKOFFS[0], 1), (2, 2, 2, 1, KICKOFFS[1], 1)],
          gameweeks=[(10, 1, 1, 90, 2.5, 0.0), (10, 2, 2, 90, 2.5, 0.0)])
    row = _row(conn, 10)
    assert row["current_rate"] > 2.0
    assert row["posterior_mean"] < 0.6 * row["current_rate"]
    assert abs(row["posterior_mean"] - row["prior_mean"]) < abs(row["posterior_mean"] - row["current_rate"])


# 4 ------------------------------------------------------------------


def test_stable_long_prior_has_more_ess_than_tiny_prior():
    conn = connect_database(":memory:")
    _seed(conn, players={10: (1, 3), 11: (1, 3)},
          histories=[_history(10, 2000, 20.0, 10.0), _history(11, 200, 2.0, 1.0)],
          fixtures=[(1, 1, 1, 2, KICKOFFS[0], 1)],
          gameweeks=[(10, 1, 1, 90, 0.9, 0.1), (11, 1, 1, 90, 0.9, 0.1)])
    assert _row(conn, 10)["prior_ess"] > _row(conn, 11)["prior_ess"]


# 5 ------------------------------------------------------------------


def test_tiny_prior_season_sample_is_heavily_discounted():
    conn = connect_database(":memory:")
    _seed(conn, players={10: (1, 3)}, histories=[_history(10, 90, 1.0, 0.5)],
          fixtures=[(1, 1, 1, 2, KICKOFFS[0], 1)], gameweeks=[(10, 1, 1, 90, 0.9, 0.1)])
    row = _row(conn, 10)
    assert "TINY_HISTORICAL_SAMPLE" in row["risk_flags"]
    assert row["prior_ess"] < 0.5 * player_rates.PlayerRatesConfig().xg_prior_ess_minutes


# 6 ------------------------------------------------------------------


def test_no_prior_falls_back_to_pooled_with_warning():
    conn = connect_database(":memory:")
    # Player 20 has no history at all; player 10 supplies the MID pooled prior.
    _seed(conn, players={10: (1, 3), 20: (2, 3)}, histories=[_history(10, 2000, 20.0, 10.0)],
          fixtures=[(1, 1, 1, 2, KICKOFFS[0], 1)], gameweeks=[(20, 1, 1, 90, 0.0, 0.0)])
    row = _row(conn, 20)
    assert row["prior_source"] == "position_pooled"
    assert "NO_HISTORICAL_PLAYER_PRIOR" in row["risk_flags"]
    assert row["prior_mean"] is not None


# 7 ------------------------------------------------------------------


def test_xg_and_xa_use_distinct_ess_parameters():
    config = player_rates.PlayerRatesConfig()
    assert config.xg_prior_ess_minutes != config.xa_prior_ess_minutes
    conn = connect_database(":memory:")
    _seed(conn, players={10: (1, 3)}, histories=[_history(10, 2000, 20.0, 10.0)],
          fixtures=[(1, 1, 1, 2, KICKOFFS[0], 1)], gameweeks=[(10, 1, 1, 90, 0.9, 0.4)])
    xg = _row(conn, 10, player_rates.COMPONENT_XG)
    xa = _row(conn, 10, player_rates.COMPONENT_XA)
    assert xg["prior_ess"] != xa["prior_ess"]


# 8 ------------------------------------------------------------------


def test_role_change_reduces_prior_ess_without_adding_rate():
    conn = connect_database(":memory:")
    _seed(conn, players={10: (1, 3)}, histories=[_history(10, 2000, 20.0, 10.0)],
          fixtures=[(1, 1, 1, 2, KICKOFFS[0], 1)], gameweeks=[(10, 1, 1, 90, 0.9, 0.4)],
          notes=[(10, "role_change", "confirmed", "2026-09-06T10:00:00Z", None)])
    row = _row(conn, 10)
    assert "ROLE_CHANGE_CONFIRMED" in row["risk_flags"]
    assert row["role_modifier_applied"] is True
    assert row["prior_ess"] < 0.5 * row["prior_ess_before_discounts"]
    # The posterior is exactly the closed form with the reduced ESS, never a bump.
    expected = (row["current_minutes"] * row["current_rate"] + row["prior_ess"] * row["prior_mean"]) / (
        row["current_minutes"] + row["prior_ess"]
    )
    assert row["posterior_mean"] == pytest.approx(expected, abs=1e-6)


# 9 ------------------------------------------------------------------


def test_monetary_input_does_not_affect_rates():
    from fpl_brain import repositories as repo
    from fpl_brain.models import PlayerSnapshotRecord

    conn = connect_database(":memory:")
    _seed(conn, players={10: (1, 3)}, histories=[_history(10, 2000, 20.0, 10.0)],
          fixtures=[(1, 1, 1, 2, KICKOFFS[0], 1)], gameweeks=[(10, 1, 1, 90, 0.9, 0.4)])
    before = _strip(player_rates.build_player_rate_projections(conn, 4, CUTOFF))
    with conn:
        run = repo.create_fetch_run(conn, "fetch_fpl")
        repo.insert_snapshots(conn, [PlayerSnapshotRecord(player_id=10, captured_at="2026-09-10T08:00:00Z", now_cost=200, raw_json={})], run)
    assert _strip(player_rates.build_player_rate_projections(conn, 4, CUTOFF)) == before


# 10 -----------------------------------------------------------------


def test_manager_ownership_does_not_affect_rates():
    conn = connect_database(":memory:")
    _seed(conn, players={10: (1, 3)}, histories=[_history(10, 2000, 20.0, 10.0)],
          fixtures=[(1, 1, 1, 2, KICKOFFS[0], 1)], gameweeks=[(10, 1, 1, 90, 0.9, 0.4)])
    before = _strip(player_rates.build_player_rate_projections(conn, 4, CUTOFF))
    with conn:
        conn.execute("""INSERT INTO squad_picks(entry_id, event, player_id, position, synced_at, raw_json)
                        VALUES (241392, 4, 10, 1, '2026-09-10T08:00:00Z', '{}')""")
    assert _strip(player_rates.build_player_rate_projections(conn, 4, CUTOFF)) == before


# 11 -----------------------------------------------------------------


def test_double_gameweek_evidence_includes_both_fixtures():
    conn = connect_database(":memory:")
    _seed(conn, players={10: (1, 3)}, histories=[_history(10, 2000, 20.0, 10.0)],
          fixtures=[(1, 3, 1, 2, KICKOFFS[2], 1), (2, 3, 2, 1, KICKOFFS[2], 1)],
          gameweeks=[(10, 3, 1, 90, 1.0, 0.2), (10, 3, 2, 90, 0.0, 0.4)])
    row = _row(conn, 10)
    assert row["current_minutes"] == 180
    assert row["current_played_rows"] == 2
    assert row["current_total"] == pytest.approx(1.0)


# 12 -----------------------------------------------------------------


def test_missing_xg_is_a_data_gap_not_zero():
    conn = connect_database(":memory:")
    _seed(conn, players={10: (1, 3)}, histories=[_history(10, 2000, 20.0, 10.0)],
          fixtures=[(1, 1, 1, 2, KICKOFFS[0], 1), (2, 2, 2, 1, KICKOFFS[1], 1)],
          gameweeks=[(10, 1, 1, 90, 0.9, 0.2), (10, 2, 2, 90, None, 0.3)])
    row = _row(conn, 10)
    assert "CURRENT_XG_DATA_GAP" in row["risk_flags"]
    assert row["current_minutes"] == 90  # the missing-xG row is excluded, not zeroed
    assert row["current_total"] == pytest.approx(0.9)


# 13 -----------------------------------------------------------------


def test_no_penalty_component_and_rates_named_without_npxg():
    conn = connect_database(":memory:")
    _seed(conn, players={10: (1, 3)}, histories=[_history(10, 2000, 20.0, 10.0)],
          fixtures=[(1, 1, 1, 2, KICKOFFS[0], 1)], gameweeks=[(10, 1, 1, 90, 0.9, 0.4)])
    rows = player_rates.build_player_rate_projections(conn, 4, CUTOFF)
    assert {row["component"] for row in rows} == {"xG_per90", "xA_per90"}
    row = _row(conn, 10)
    assert row["provenance"]["npxg_separation"] is False
    assert row["provenance"]["penalties"] == "embedded_in_xG"
    assert not any("NPxG" in key for key in row)


# 14 -----------------------------------------------------------------


def test_deterministic_for_same_inputs():
    conn = connect_database(":memory:")
    _seed(conn, players={10: (1, 3)}, histories=[_history(10, 2000, 20.0, 10.0)],
          fixtures=[(1, 1, 1, 2, KICKOFFS[0], 1)], gameweeks=[(10, 1, 1, 90, 0.9, 0.4)])
    assert _strip(player_rates.build_player_rate_projections(conn, 4, CUTOFF)) == _strip(
        player_rates.build_player_rate_projections(conn, 4, CUTOFF)
    )


# 15 -----------------------------------------------------------------


def test_frozen_rate_predictions_are_immutable():
    conn = connect_database(":memory:")
    _seed(conn, players={10: (1, 3)}, histories=[_history(10, 2000, 20.0, 10.0)],
          fixtures=[(1, 1, 1, 2, KICKOFFS[0], 1)], gameweeks=[(10, 1, 1, 90, 0.9, 0.4)])
    row = _row(conn, 10)
    with conn:
        run_id = analytics.create_projection_run(
            conn, model_family=analytics.PLAYER_RATES_MODEL_FAMILY,
            model_version=player_rates.PLAYER_RATE_MODEL_VERSION, planning_event=4,
            planning_context_hash="test", data_cutoff=CUTOFF, scouting_cutoff=None,
            official_run_ids={}, config_hash=player_rates.PlayerRatesConfig().config_hash(),
            deadline_status="PRE_DEADLINE",
        )
        analytics.freeze_player_rate_projection(
            conn, run_id, player_id=10, component=row["component"], event=4,
            payload={k: v for k, v in row.items() if k not in {"player_id", "component", "event"}},
            model_version=player_rates.PLAYER_RATE_MODEL_VERSION,
        )
        analytics.finish_projection_run(conn, run_id, "complete")
    with pytest.raises(Exception):
        conn.execute("UPDATE player_rate_projections SET component='x' WHERE projection_run_id=?", (run_id,))
    with pytest.raises(Exception):
        conn.execute("DELETE FROM player_rate_projections WHERE projection_run_id=?", (run_id,))
    assert len(analytics.player_rate_projections(conn, run_id)) == 1


# 16 -----------------------------------------------------------------


def test_no_lookahead_ignores_post_cutoff_evidence():
    conn = connect_database(":memory:")
    _seed(conn, players={10: (1, 3)}, histories=[_history(10, 2000, 20.0, 10.0)],
          fixtures=[(1, 1, 1, 2, KICKOFFS[0], 1)], gameweeks=[(10, 1, 1, 90, 0.9, 0.4)])
    before = _row(conn, 10)
    # A completed fixture after the cutoff and a post-cutoff scouting note.
    _seed(conn, players={10: (1, 3)}, fixtures=[(9, 2, 2, 1, "2026-09-11T14:00:00Z", 1)],
          gameweeks=[(10, 2, 9, 90, 5.0, 3.0)],
          notes=[(10, "role_change", "confirmed", "2026-09-11T09:00:00Z", None)])
    after = _row(conn, 10)
    assert after["current_minutes"] == before["current_minutes"]
    assert after["posterior_mean"] == before["posterior_mean"]
    assert after["prior_ess"] == before["prior_ess"]
