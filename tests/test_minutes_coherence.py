"""Minutes team-coherence (v1.2.0) tests: solver identities, bounds, integration."""

from __future__ import annotations

import copy
import hashlib

import pytest

from fpl_brain import analytics, minutes_coherence as mc, xpts
from fpl_brain.database import connect_database
from fpl_brain.models import (
    EventRecord,
    FixtureRecord,
    PlayerRecord,
    PlayerSnapshotRecord,
    PositionRecord,
    TeamRecord,
)

CUTOFF = "2026-09-10T12:00:00Z"
DEADLINE = "2026-09-12T12:30:00Z"
EVENT = 4
CFG = mc.MinutesCoherenceConfig()


# ---------------------------------------------------------------------------
# Synthetic side helper (pure solver tests).
# ---------------------------------------------------------------------------


def _payload(pid, *, avail=1.0, sga=0.5, cgr=0.5, m_start=90.0, m_cameo=15.0,
             p60s=0.9, p80s=0.7, p60c=0.04):
    raw_start = avail * sga
    raw_cameo = avail * (1.0 - sga) * cgr
    raw_zero = 1.0 - raw_start - raw_cameo
    raw_p60 = raw_start * p60s + raw_cameo * p60c
    raw_p80 = min(raw_start * p80s, raw_p60)
    return {
        "player_id": pid,
        "p_available": avail,
        "p_start_given_available": sga,
        "p_cameo_given_not_start": cgr,
        "expected_minutes_if_start": m_start,
        "expected_minutes_if_cameo": m_cameo,
        "p_60_given_start": p60s,
        "p_80_given_start": p80s,
        "p_60_given_cameo": p60c,
        "p_start": raw_start,
        "p_cameo": raw_cameo,
        "p_zero": raw_zero,
        "p_1_59": 1.0 - raw_zero - raw_p60,
        "p_60_plus": raw_p60,
        "p_80_plus": raw_p80,
        "expected_minutes": raw_start * m_start + raw_cameo * m_cameo,
    }


def _side(n_players=20, start_id=1000, **over):
    return [_payload(start_id + i, **over) for i in range(n_players)]


def _solve(payloads, fixture_id=1, team_id=1, config=CFG):
    return mc.solve_team_coherence(payloads, fixture_id=fixture_id, team_id=team_id, config=config)


# ---------------------------------------------------------------------------
# Database world (integration tests): 12 players per side so a side is fieldable.
# ---------------------------------------------------------------------------


def _world(conn):
    from fpl_brain import repositories as repo
    from fpl_brain.models import PlayerGameweekRecord

    squad = 25
    with conn:
        repo.upsert_teams(conn, [TeamRecord(id=t, name=f"Team {t}") for t in (1, 2, 3)])
        repo.upsert_positions(
            conn,
            [PositionRecord(id=p, singular_name_short=n)
             for p, n in ((1, "GKP"), (2, "DEF"), (3, "MID"), (4, "FWD"))],
        )
        players = []
        for i in range(1, squad + 1):
            players.append(PlayerRecord(id=i, web_name=f"T1P{i}", full_name=f"Team1 Player {i}", team_id=1,
                                        element_type=(1 if i == 1 else (2 if i <= 8 else (3 if i <= 18 else 4)))))
        for i in range(1, squad + 1):
            players.append(PlayerRecord(id=100 + i, web_name=f"T2P{i}", full_name=f"Team2 Player {i}", team_id=2,
                                        element_type=(1 if i == 1 else (2 if i <= 8 else (3 if i <= 18 else 4)))))
        players.append(PlayerRecord(id=300, web_name="T3P1", full_name="Team3 Player 1", team_id=3, element_type=3))
        repo.upsert_players(conn, players)
        repo.upsert_events(
            conn,
            [EventRecord(id=e, finished=1 if e == 1 else 0, data_checked=1 if e == 1 else 0,
                         deadline_time=DEADLINE if e == EVENT else "2026-08-21T12:00:00Z", raw_json={})
             for e in (1, EVENT)],
        )
        repo.upsert_fixtures(
            conn,
            [
                FixtureRecord(id=90, event=1, team_h=1, team_a=2, finished=1, started=1,
                              team_h_score=1, team_a_score=1, kickoff_time="2026-08-22T14:00:00Z", raw_json={}),
                FixtureRecord(id=100, event=EVENT, team_h=1, team_a=2, finished=0, started=0,
                              kickoff_time="2026-09-13T14:00:00Z", raw_json={}),
                FixtureRecord(id=101, event=EVENT, team_h=2, team_a=1, finished=0, started=0,
                              kickoff_time="2026-09-14T14:00:00Z", raw_json={}),
            ],
        )
        # One completed 88-minute start per player gives realistic conditional
        # minutes (E[min|start] ~ 88), so a 25-man side can supply 990 minutes.
        gameweeks = [
            PlayerGameweekRecord(player_id=p.id, event=1, fixture_id=90,
                                 was_home=1 if p.team_id == 1 else 0, minutes=88, starts=1,
                                 expected_goals=0.2, expected_assists=0.1, source="element_summary", raw_json={})
            for p in players if p.team_id in (1, 2)
        ]
        repo.upsert_player_gameweeks(conn, gameweeks)
        run = repo.create_fetch_run(conn, "fetch_fpl")
        repo.insert_snapshots(
            conn,
            [PlayerSnapshotRecord(player_id=p.id, captured_at="2026-09-09T08:00:00Z", now_cost=50,
                                  status="a", ep_next=5.0, raw_json={}) for p in players],
            run,
        )
    return [p.id for p in players]


def _freeze_runs(conn, *, minutes_rows=None, coherence_records=None, rates_players=None,
                 minutes_version="test_minutes", with_team=True):
    """Create component runs; returns run ids."""

    run_ids: dict[str, int] = {}
    with conn:
        if minutes_rows is not None:
            run_id = analytics.create_projection_run(
                conn, model_family=analytics.MINUTES_MODEL_FAMILY, model_version=minutes_version,
                planning_event=EVENT, planning_context_hash="t", data_cutoff=CUTOFF, scouting_cutoff=None,
                official_run_ids={}, config_hash="x", deadline_status="PRE_DEADLINE",
            )
            for row in minutes_rows:
                analytics.freeze_prediction(
                    conn, run_id, kind=analytics.MINUTES_V1_KIND, player_id=int(row["player_id"]), event=EVENT,
                    fixture_id=int(row["fixture_id"]),
                    payload={k: v for k, v in row.items() if k not in {"player_id", "fixture_id", "event"}},
                    model_version=minutes_version,
                )
            for record in (coherence_records or []):
                analytics.freeze_team_minutes_coherence(conn, run_id, record)
            analytics.finish_projection_run(conn, run_id, "complete")
            run_ids["minutes"] = run_id
        if with_team:
            team_payload = {"expected_goals_for": 1.4, "expected_goals_against": 1.3, "home_advantage": 0.1,
                            "opponent_defence_rating": 0.0, "attack_rating": 0.0, "league_baseline": 0.28,
                            "risk_flags": []}
            for key, family, version in (("team", analytics.TEAM_MODEL_FAMILY, "test_team"),
                                         ("team_baseline", analytics.TEAM_BASELINE_MODEL_FAMILY, "test_team_baseline")):
                run_id = analytics.create_projection_run(
                    conn, model_family=family, model_version=version, planning_event=EVENT,
                    planning_context_hash="t", data_cutoff=CUTOFF, scouting_cutoff=None, official_run_ids={},
                    config_hash="x", deadline_status="PRE_DEADLINE",
                )
                for fixture_id, team_id, opponent_id, venue in ((100, 1, 2, "home"), (100, 2, 1, "away")):
                    analytics.freeze_team_fixture_projection(
                        conn, run_id, fixture_id=fixture_id, event=EVENT, team_id=team_id,
                        opponent_id=opponent_id, venue=venue, payload=dict(team_payload), model_version=version,
                    )
                analytics.finish_projection_run(conn, run_id, "complete")
                run_ids[key] = run_id
        if rates_players:
            run_id = analytics.create_projection_run(
                conn, model_family=analytics.PLAYER_RATES_MODEL_FAMILY, model_version="test_rates",
                planning_event=EVENT, planning_context_hash="t", data_cutoff=CUTOFF, scouting_cutoff=None,
                official_run_ids={}, config_hash="x", deadline_status="PRE_DEADLINE",
            )
            rate_payload = {"prior_mean": 0.4, "prior_ess": 700.0, "prior_source": "pooled",
                            "current_minutes": 200.0, "current_total": 1.0, "current_rate": 0.45,
                            "posterior_mean": 0.42, "posterior_ess": 900.0, "risk_flags": []}
            for pid in rates_players:
                for component in ("xG_per90", "xA_per90"):
                    analytics.freeze_player_rate_projection(
                        conn, run_id, player_id=int(pid), component=component, event=EVENT,
                        payload=dict(rate_payload), model_version="test_rates",
                    )
            analytics.finish_projection_run(conn, run_id, "complete")
            run_ids["rates"] = run_id
    return run_ids


# ---------------------------------------------------------------------------
# Solver identities.
# ---------------------------------------------------------------------------


def test_adjusted_start_sum_is_exactly_eleven():
    for over in ({"sga": 0.6}, {"sga": 0.9}, {"sga": 0.2}, {"sga": 0.5, "avail": 1.0}):
        rows, record = _solve(_side(20, **over))
        assert sum(r["p_start"] for r in rows) == pytest.approx(11.0, abs=1e-4)
        assert record["start_residual"] == pytest.approx(0.0, abs=1e-6)


def test_hard_out_players_remain_zero():
    payloads = _side(20, sga=0.6)
    payloads[0]["p_available"] = 0.0
    rows, _ = _solve(payloads)
    assert rows[0]["p_start"] == 0.0
    assert rows[0]["p_cameo"] == 0.0


def test_start_probability_never_exceeds_availability():
    payloads = [{"p_available": 0.9, **{k: v for k, v in _payload(2000 + i).items() if k != "p_available"}}
                for i in range(14)]
    rows, _ = _solve(payloads)
    for row in rows:
        assert row["p_start"] <= row["p_available"] + 1e-9


def test_relative_start_odds_ordering_preserved():
    payloads = [_payload(1, sga=0.8), _payload(2, sga=0.3), _payload(3, sga=0.55)] + _side(17, start_id=2000, sga=0.5)
    rows, _ = _solve(payloads)
    by_id = {r["player_id"]: r for r in rows}
    assert by_id[1]["p_start"] > by_id[3]["p_start"] > by_id[2]["p_start"]


def test_high_confidence_starter_above_fringe_option():
    payloads = [_payload(1, sga=0.95), _payload(2, sga=0.15)] + _side(18, start_id=2000, sga=0.5)
    rows, _ = _solve(payloads)
    by_id = {r["player_id"]: r for r in rows}
    assert by_id[1]["p_start"] > by_id[2]["p_start"]


def test_adjusted_cameo_probabilities_stay_valid():
    rows, _ = _solve(_side(20, sga=0.6, cgr=0.5))
    for row in rows:
        assert 0.0 <= row["p_cameo"] <= 1.0
        assert 0.0 <= row["p_cameo_given_not_start"] <= 1.0


def test_total_expected_minutes_is_990():
    rows, record = _solve(_side(20, sga=0.6))
    assert sum(r["expected_minutes"] for r in rows) == pytest.approx(990.0, abs=1e-2)
    assert record["minutes_residual"] == pytest.approx(0.0, abs=1e-3)


def test_start_plus_cameo_within_availability():
    rows, _ = _solve(_side(20, sga=0.7, cgr=0.6))
    for row in rows:
        assert row["p_start"] + row["p_cameo"] <= row["p_available"] + 1e-9


def test_p60_and_p80_coherence_after_adjustment():
    rows, _ = _solve(_side(20, sga=0.6))
    for row in rows:
        appearance = row["p_start"] + row["p_cameo"]
        assert 0.0 <= row["p_zero"] <= 1.0
        assert 0.0 <= row["p_60_plus"] <= appearance + 1e-6
        assert 0.0 <= row["p_80_plus"] <= row["p_60_plus"] + 1e-6
        assert row["p_zero"] + row["p_1_59"] + row["p_60_plus"] == pytest.approx(1.0, abs=1e-5)


def test_impossible_availability_mass_raises():
    with pytest.raises(mc.TeamCoherenceError):
        _solve(_side(10, sga=0.5))  # 10 players cannot field 11


def test_impossible_cameo_minute_budget_raises():
    payloads = [_payload(i, avail=1.0, sga=1.0, m_start=80.0, m_cameo=0.0) for i in range(11)]
    with pytest.raises(mc.TeamCoherenceError):
        _solve(payloads)


def test_deterministic_root_solving_and_identical_results():
    payloads = _side(20, sga=0.62, cgr=0.44)
    first_rows, first_record = _solve(copy.deepcopy(payloads))
    second_rows, second_record = _solve(copy.deepcopy(payloads))
    assert first_rows == second_rows
    assert first_record == second_record


def test_config_hash_changes_with_coherence_parameters():
    base = mc.MinutesCoherenceConfig()
    assert base.config_hash() == mc.MinutesCoherenceConfig().config_hash()
    assert base.config_hash() != mc.MinutesCoherenceConfig(solver_bound=30.0).config_hash()
    assert base.config_hash() != mc.MinutesCoherenceConfig(start_tolerance=1e-6).config_hash()


# ---------------------------------------------------------------------------
# Integration.
# ---------------------------------------------------------------------------


def test_every_built_side_is_coherent_and_raw_retained():
    conn = connect_database(":memory:")
    _world(conn)
    rows, records = mc.build_minutes_predictions_coherent(conn, EVENT, CUTOFF)
    assert records and len(rows) == 25 * 4  # 25 players x 4 sides
    for record in records:
        assert record["adjusted_start_sum"] == pytest.approx(11.0, abs=1e-4)
        assert record["adjusted_minutes_sum"] == pytest.approx(990.0, abs=1e-2)
        assert abs(record["start_residual"]) <= CFG.verify_start_tolerance
        assert abs(record["minutes_residual"]) <= CFG.verify_minutes_tolerance
        assert record["raw_start_sum"] > 11.0  # the raw independent sum is retained and over-full
    assert mc.coherence_readiness(records)["status"] == "PASS"
    for row in rows:
        for key in ("p_start_raw_independent", "p_cameo_raw_independent", "p_zero_raw_independent",
                    "p_60_plus_raw_independent", "p_80_plus_raw_independent", "expected_minutes_raw_independent"):
            assert key in row
        assert row["p_zero"] + row["p_1_59"] + row["p_60_plus"] == pytest.approx(1.0, abs=1e-5)


def test_dgw_solved_separately_and_bgw_absent():
    conn = connect_database(":memory:")
    _world(conn)
    rows, records = mc.build_minutes_predictions_coherent(conn, EVENT, CUTOFF)
    # team 1 and 2 each have two fixtures -> four sides, each independently 11/990
    assert sorted((r["fixture_id"], r["team_id"]) for r in records) == [(100, 1), (100, 2), (101, 1), (101, 2)]
    for record in records:
        assert record["adjusted_start_sum"] == pytest.approx(11.0, abs=1e-4)
    # team 3 has no event-4 fixture -> its player 300 yields no projection
    assert [r for r in rows if r["player_id"] == 300] == []


def test_price_and_ownership_are_noops_for_coherent_minutes():
    from fpl_brain import repositories as repo

    conn = connect_database(":memory:")
    _world(conn)
    before = [(r["player_id"], r["p_start"], r["expected_minutes"])
              for r in mc.build_minutes_predictions_coherent(conn, EVENT, CUTOFF)[0]]
    with conn:
        run = repo.create_fetch_run(conn, "fetch_fpl")
        repo.insert_snapshots(conn, [PlayerSnapshotRecord(player_id=1, captured_at="2026-09-10T08:00:00Z",
                                                          now_cost=200, status="a", raw_json={})], run)
        conn.execute("""INSERT INTO squad_picks(entry_id, event, player_id, position, synced_at, raw_json)
                        VALUES (241392, 4, 1, 1, '2026-09-10T08:00:00Z', '{}')""")
    after = [(r["player_id"], r["p_start"], r["expected_minutes"])
             for r in mc.build_minutes_predictions_coherent(conn, EVENT, CUTOFF)[0]]
    assert after == before


def test_historical_runs_untouched_by_coherent_build():
    conn = connect_database(":memory:")
    _world(conn)
    raw_rows = mc.build_minutes_predictions_coherent(conn, EVENT, CUTOFF)[0]
    runs = _freeze_runs(conn, minutes_rows=raw_rows, minutes_version="minutes_v1.1.0")
    query = "SELECT payload_json FROM frozen_predictions WHERE projection_run_id=? ORDER BY id"
    digest_before = hashlib.sha256("|".join(r["payload_json"] for r in conn.execute(query, (runs["minutes"],))).encode()).hexdigest()
    mc.build_minutes_predictions_coherent(conn, EVENT, CUTOFF)
    _freeze_runs(conn, minutes_rows=raw_rows, minutes_version="minutes_v1.2.0", with_team=False)
    digest_after = hashlib.sha256("|".join(r["payload_json"] for r in conn.execute(query, (runs["minutes"],))).encode()).hexdigest()
    assert digest_before == digest_after


def test_no_lookahead_from_post_cutoff_evidence():
    from fpl_brain import repositories as repo
    from fpl_brain.models import PlayerGameweekRecord

    conn = connect_database(":memory:")
    _world(conn)
    before = [(r["player_id"], r["p_start"]) for r in mc.build_minutes_predictions_coherent(conn, EVENT, CUTOFF)[0]]
    with conn:
        repo.upsert_fixtures(conn, [FixtureRecord(id=200, event=1, team_h=1, team_a=2, finished=1, started=1,
                                                  kickoff_time="2026-09-11T14:00:00Z", raw_json={})])
        repo.upsert_player_gameweeks(conn, [PlayerGameweekRecord(
            player_id=2, event=1, fixture_id=200, minutes=90, starts=1, expected_goals=5.0,
            source="element_summary", raw_json={})])
    after = [(r["player_id"], r["p_start"]) for r in mc.build_minutes_predictions_coherent(conn, EVENT, CUTOFF)[0]]
    assert after == before


def test_calibration_can_address_raw_and_adjusted_variants():
    from fpl_brain import calibration

    conn = connect_database(":memory:")
    _world(conn)
    rows, records = mc.build_minutes_predictions_coherent(conn, EVENT, CUTOFF)
    runs = _freeze_runs(conn, minutes_rows=rows, coherence_records=records,
                        minutes_version=mc.MINUTES_COHERENT_MODEL_VERSION)
    with conn:
        conn.execute("""INSERT INTO outcome_observations(event, player_id, fixture_id, actual_started,
                        actual_minutes, actual_60_plus, actual_zero_minutes, observed_at, source)
                        VALUES (?,?,?,?,?,?,?,?,?)""",
                     (EVENT, 2, 100, 1, 90, 1, 0, "2026-09-11T00:00:00Z", "test"))
    coherent = calibration.evaluate_minutes_variant(conn, runs["minutes"], EVENT, raw=False)
    raw = calibration.evaluate_minutes_variant(conn, runs["minutes"], EVENT, raw=True)
    assert "brier_start" in coherent and "brier_start" in raw
    assert coherent["mae_minutes"] != raw["mae_minutes"]


def test_xpts_consumes_coherent_run_and_cap_diagnostics():
    conn = connect_database(":memory:")
    _world(conn)
    raw_rows = mc.build_minutes_predictions_coherent(conn, EVENT, CUTOFF)[0]
    # raw independent variant: strip the coherence mark and restore raw marginals
    raw_variant = []
    for row in raw_rows:
        row = dict(row)
        row["p_start"] = row["p_start_raw_independent"]
        row["p_cameo"] = row["p_cameo_raw_independent"]
        row["expected_minutes"] = row["expected_minutes_raw_independent"]
        row.pop("team_coherence", None)
        raw_variant.append(row)

    all_players = list(range(1, 26)) + list(range(101, 126))
    raw_runs = _freeze_runs(conn, minutes_rows=raw_variant, rates_players=all_players)
    coh_runs = _freeze_runs(conn, minutes_rows=raw_rows, rates_players=all_players)

    raw_built = xpts.build_xpts_projections(
        conn, event=EVENT, cutoff=CUTOFF, minutes_run_id=raw_runs["minutes"], team_run_id=raw_runs["team"],
        team_baseline_run_id=raw_runs["team_baseline"], rate_run_id=raw_runs["rates"])
    coh_built = xpts.build_xpts_projections(
        conn, event=EVENT, cutoff=CUTOFF, minutes_run_id=coh_runs["minutes"], team_run_id=coh_runs["team"],
        team_baseline_run_id=coh_runs["team_baseline"], rate_run_id=coh_runs["rates"])
    assert raw_built["meta"]["input_team_coherence"] is False
    assert coh_built["meta"]["input_team_coherence"] is True

    raw_ready = xpts.readiness_summary(None, raw_built["rows"], raw_built["coherence"], strict_coherence=True)
    coh_ready = xpts.readiness_summary(None, coh_built["rows"], coh_built["coherence"], strict_coherence=True)
    assert raw_ready["status"] == "FAIL"  # raw mass violates the strict identity
    assert coh_ready["status"] != "FAIL"  # coherent mass satisfies it

    def cap_stats(rows):
        scales = [r["xg_scale_factor"] for r in rows]
        return (sum(1 for r in rows if "TEAM_XG_CAP_APPLIED" in r["risk_flags"]),
                sum(1 for r in rows if "TEAM_XG_EXCESS_WARN" in r["risk_flags"]),
                sum(scales) / len(scales), min(scales), max(scales))

    raw_stats = cap_stats(raw_built["rows"])
    coh_stats = cap_stats(coh_built["rows"])
    assert raw_stats[2] <= 1.0 and coh_stats[2] <= 1.0  # mean scale never exceeds 1 (downward only)
    assert coh_stats[1] >= 0  # diagnostics recorded, never targeted


def test_xpts_strict_coherence_flags_violation():
    conn = connect_database(":memory:")
    _world(conn)
    rows, records = mc.build_minutes_predictions_coherent(conn, EVENT, CUTOFF)
    # A mild but material violation: total start mass 12.0 (deviation 1.0)
    # with the minute mass left coherent, so only the strict gate should fail.
    bad = []
    for row in rows:
        row = dict(row)
        row["p_start"] = round(row["p_start"] * (12.0 / 11.0), 6)
        row["team_coherence"] = {**row["team_coherence"], "applied": True}
        bad.append(row)
    runs = _freeze_runs(conn, minutes_rows=bad, rates_players=list(range(1, 26)) + list(range(101, 126)))
    built = xpts.build_xpts_projections(
        conn, event=EVENT, cutoff=CUTOFF, minutes_run_id=runs["minutes"], team_run_id=runs["team"],
        team_baseline_run_id=runs["team_baseline"], rate_run_id=runs["rates"])
    strict = xpts.readiness_summary(None, built["rows"], built["coherence"], strict_coherence=True)
    relaxed = xpts.readiness_summary(None, built["rows"], built["coherence"], strict_coherence=False)
    assert strict["status"] == "FAIL"
    assert any(reason.startswith("COHERENT_START_SUM_VIOLATION") for reason in strict["fail_reasons"])
    assert relaxed["status"] != "FAIL"


def test_gc_deduction_exact_identity():
    """Deterministic 90-minute GK/DEF: GC_xPts == -1 * E[floor(N/2)]."""

    payload = {"p_start": 1.0, "p_cameo": 0.0, "expected_minutes_if_start": 90.0,
               "expected_minutes_if_cameo": 15.0}
    for lam in (0.5, 1.3, 2.0, 3.7):
        got = xpts._goals_conceded_deduction(payload, lam, goals_per_deduction=2, points_per_deduction=1)
        assert got == pytest.approx(-1.0 * xpts.expected_floor_half(lam), abs=1e-12)
        assert got != pytest.approx(-(lam / 2.0), abs=1e-6)
