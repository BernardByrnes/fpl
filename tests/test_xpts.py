"""xPts v1 integration tests: components, coherence, caps, grain, immutability."""

from __future__ import annotations

import copy
import math
import pytest

from fpl_brain import analytics, xpts
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


# ---------------------------------------------------------------------------
# Minimal frozen-input scaffolding.
# ---------------------------------------------------------------------------


def _seed(conn):
    from fpl_brain import repositories as repo

    with conn:
        repo.upsert_teams(conn, [TeamRecord(id=t, name=f"Team {t}") for t in (1, 2)])
        repo.upsert_positions(
            conn,
            [PositionRecord(id=p, singular_name_short=n)
             for p, n in ((1, "GKP"), (2, "DEF"), (3, "MID"), (4, "FWD"))],
        )
        repo.upsert_players(
            conn,
            [
                PlayerRecord(id=10, web_name="MID1", full_name="Mid One", team_id=1, element_type=3),
                PlayerRecord(id=11, web_name="FWD1", full_name="Fwd One", team_id=1, element_type=4),
                PlayerRecord(id=12, web_name="DEF1", full_name="Def One", team_id=1, element_type=2),
                PlayerRecord(id=13, web_name="GK1", full_name="Keeper One", team_id=1, element_type=1),
                PlayerRecord(id=20, web_name="MID2", full_name="Mid Two", team_id=2, element_type=3),
            ],
        )
        repo.upsert_events(
            conn,
            [EventRecord(id=e, finished=1 if e == 1 else 0, data_checked=1 if e == 1 else 0,
                         deadline_time=DEADLINE if e == EVENT else "2026-08-21T12:00:00Z", raw_json={})
             for e in (1, EVENT)],
        )
        repo.upsert_fixtures(
            conn,
            [
                FixtureRecord(id=100, event=EVENT, team_h=1, team_a=2, finished=0, started=0,
                              kickoff_time="2026-09-13T14:00:00Z", raw_json={}),
                FixtureRecord(id=101, event=EVENT, team_h=2, team_a=1, finished=0, started=0,
                              kickoff_time="2026-09-14T14:00:00Z", raw_json={}),
            ],
        )
        run = repo.create_fetch_run(conn, "fetch_fpl")
        repo.insert_snapshots(
            conn,
            [PlayerSnapshotRecord(player_id=10, captured_at="2026-09-09T08:00:00Z", now_cost=50,
                                  status="a", ep_next=6.5, raw_json={})],
            run,
        )


def _minutes_payload(**over):
    payload = {
        "p_start": 0.8, "p_cameo": 0.1, "p_zero": 0.1, "p_60_plus": 0.75, "p_1_59": 0.15,
        "expected_minutes": 75.0, "expected_minutes_if_start": 88.0, "expected_minutes_if_cameo": 15.0,
        "p_available": 1.0,
    }
    payload.update(over)
    return payload


def _team_payload(**over):
    payload = {
        "expected_goals_for": 1.5, "expected_goals_against": 1.3, "home_advantage": 0.10,
        "opponent_defence_rating": 0.0, "attack_rating": 0.0, "league_baseline": 0.28,
        "risk_flags": [],
    }
    payload.update(over)
    return payload


def _rate_payload(**over):
    payload = {
        "prior_mean": 0.4, "prior_ess": 700.0, "prior_source": "pooled", "current_minutes": 200.0,
        "current_total": 1.0, "current_rate": 0.45, "posterior_mean": 0.42, "posterior_ess": 900.0,
        "risk_flags": [],
    }
    payload.update(over)
    return payload


def _freeze(conn, *, minutes, team, baseline, rates, event=EVENT, cutoff=CUTOFF):
    """Create immutable component runs from explicit payloads."""

    run_ids = {}
    with conn:
        for key, family, version, rows, writer in (
            ("minutes", analytics.MINUTES_MODEL_FAMILY, "test_minutes", minutes, "minutes"),
            ("team", analytics.TEAM_MODEL_FAMILY, "test_team", team, "team"),
            ("team_baseline", analytics.TEAM_BASELINE_MODEL_FAMILY, "test_team_baseline", baseline, "team"),
            ("rates", analytics.PLAYER_RATES_MODEL_FAMILY, "test_rates", rates, "rates"),
        ):
            run_id = analytics.create_projection_run(
                conn, model_family=family, model_version=version, planning_event=event,
                planning_context_hash="test", data_cutoff=cutoff, scouting_cutoff=None,
                official_run_ids={}, config_hash="test", deadline_status="PRE_DEADLINE",
            )
            for row in rows:
                if writer == "minutes":
                    analytics.freeze_prediction(
                        conn, run_id, kind=analytics.MINUTES_V1_KIND, player_id=int(row["player_id"]),
                        event=event, fixture_id=int(row["fixture_id"]),
                        payload={k: v for k, v in row.items() if k not in {"player_id", "fixture_id"}},
                        model_version=version,
                    )
                elif writer == "team":
                    analytics.freeze_team_fixture_projection(
                        conn, run_id, fixture_id=int(row["fixture_id"]), event=event,
                        team_id=int(row["team_id"]), opponent_id=int(row["opponent_id"]),
                        venue=str(row["venue"]),
                        payload={k: v for k, v in row.items()
                                 if k not in {"fixture_id", "team_id", "opponent_id", "venue"}},
                        model_version=version,
                    )
                else:
                    analytics.freeze_player_rate_projection(
                        conn, run_id, player_id=int(row["player_id"]), component=str(row["component"]),
                        event=event,
                        payload={k: v for k, v in row.items() if k not in {"player_id", "component"}},
                        model_version=version,
                    )
            analytics.finish_projection_run(conn, run_id, "complete")
            run_ids[key] = run_id
    return run_ids


def _default_inputs():
    minutes = [
        {"player_id": 10, "fixture_id": 100, **_minutes_payload()},
        {"player_id": 11, "fixture_id": 100, **_minutes_payload(p_start=0.7, p_60_plus=0.65)},
        {"player_id": 12, "fixture_id": 100, **_minutes_payload(p_start=0.75, p_60_plus=0.7)},
        {"player_id": 13, "fixture_id": 100, **_minutes_payload(p_start=0.9, p_60_plus=0.88)},
        {"player_id": 20, "fixture_id": 100, **_minutes_payload(p_start=0.6)},
    ]
    team = [
        {"fixture_id": 100, "team_id": 1, "opponent_id": 2, "venue": "home", **_team_payload()},
        {"fixture_id": 100, "team_id": 2, "opponent_id": 1, "venue": "away", **_team_payload()},
    ]
    baseline = [
        {"fixture_id": 100, "team_id": 1, "opponent_id": 2, "venue": "home", **_team_payload(expected_goals_for=1.3, expected_goals_against=1.3)},
        {"fixture_id": 100, "team_id": 2, "opponent_id": 1, "venue": "away", **_team_payload(expected_goals_for=1.3, expected_goals_against=1.3)},
    ]
    rates = []
    for pid in (10, 11, 12, 13, 20):
        rates.append({"player_id": pid, "component": "xG_per90", **_rate_payload()})
        rates.append({"player_id": pid, "component": "xA_per90", **_rate_payload(posterior_mean=0.25)})
    return {"minutes": minutes, "team": team, "baseline": baseline, "rates": rates}


def _build(conn, inputs, config=None, **over):
    run_ids = _freeze(conn, **inputs)
    result = xpts.build_xpts_projections(
        conn, event=EVENT, cutoff=CUTOFF, minutes_run_id=run_ids["minutes"], team_run_id=run_ids["team"],
        team_baseline_run_id=run_ids["team_baseline"], rate_run_id=run_ids["rates"],
        config=config or xpts.XPtsConfig(), **over,
    )
    return result, run_ids


def _row_for(result, player_id, fixture_id=100):
    return next(r for r in result["rows"] if r["player_id"] == player_id and r["fixture_id"] == fixture_id)


def _strip(rows):
    return [{k: v for k, v in row.items() if k != "generated_at"} for row in rows]


# 1 ------------------------------------------------------------------


def test_zero_playing_probability_gives_approximately_zero_xpts():
    conn = connect_database(":memory:")
    _seed(conn)
    inputs = _default_inputs()
    inputs["minutes"] = [{"player_id": 10, "fixture_id": 100,
                          **_minutes_payload(p_start=0.0, p_cameo=0.0, p_zero=1.0, p_60_plus=0.0,
                                             p_1_59=0.0, expected_minutes=0.0)}]
    result, _ = _build(conn, inputs)
    row = _row_for(result, 10)
    assert row["total_xpts"] == pytest.approx(0.0, abs=1e-6)


# 2 ------------------------------------------------------------------


def test_higher_xg_rate_raises_goal_xpts():
    conn = connect_database(":memory:")
    _seed(conn)
    inputs = _default_inputs()
    base_result, _ = _build(conn, copy.deepcopy(inputs))
    higher = copy.deepcopy(inputs)
    for row in higher["rates"]:
        if row["component"] == "xG_per90":
            row["posterior_mean"] = 0.84
    # Raise the team lambda too: v1.2.0 scales any player mass above team lambda
    # downward, so the uncapped regime is the meaningful comparison here.
    for row in higher["team"]:
        if row["team_id"] == 1:
            row["expected_goals_for"] = 4.0
    for row in higher["baseline"]:
        if row["team_id"] == 1:
            row["expected_goals_for"] = 4.0
    high_result, _ = _build(conn, higher)
    assert _row_for(high_result, 10)["goal_xpts"] > _row_for(base_result, 10)["goal_xpts"]


# 3 ------------------------------------------------------------------


def test_higher_xa_rate_raises_assist_xpts():
    conn = connect_database(":memory:")
    _seed(conn)
    inputs = _default_inputs()
    base_result, _ = _build(conn, copy.deepcopy(inputs))
    higher = copy.deepcopy(inputs)
    for row in higher["rates"]:
        if row["component"] == "xA_per90":
            row["posterior_mean"] = 0.50
    high_result, _ = _build(conn, higher)
    assert _row_for(high_result, 10)["assist_xpts"] > _row_for(base_result, 10)["assist_xpts"]


# 4 ------------------------------------------------------------------


def test_weaker_opponent_defence_does_not_reduce_attacking_expectation():
    conn = connect_database(":memory:")
    _seed(conn)
    inputs = _default_inputs()
    base_result, _ = _build(conn, copy.deepcopy(inputs))
    weaker = copy.deepcopy(inputs)
    for row in weaker["team"]:
        if row["team_id"] == 1:
            row["opponent_defence_rating"] = 0.20  # opponent concedes more
    weak_result, _ = _build(conn, weaker)
    base, weak = _row_for(base_result, 10), _row_for(weak_result, 10)
    assert weak["fixture_multiplier"] >= base["fixture_multiplier"]
    assert weak["goal_xpts"] >= base["goal_xpts"]


# 5 ------------------------------------------------------------------


def test_higher_opponent_lambda_does_not_raise_clean_sheet_xpts():
    conn = connect_database(":memory:")
    _seed(conn)
    inputs = _default_inputs()
    base_result, _ = _build(conn, copy.deepcopy(inputs))
    leaky = copy.deepcopy(inputs)
    for row in leaky["team"]:
        if row["team_id"] == 1:
            row["expected_goals_against"] = 2.6
    leaky_result, _ = _build(conn, leaky)
    assert _row_for(leaky_result, 12)["clean_sheet_xpts"] <= _row_for(base_result, 12)["clean_sheet_xpts"]


# 6 ------------------------------------------------------------------


def test_higher_p60_does_not_reduce_appearance_or_cs_component():
    conn = connect_database(":memory:")
    _seed(conn)
    inputs = _default_inputs()
    base_result, _ = _build(conn, copy.deepcopy(inputs))
    higher = copy.deepcopy(inputs)
    for row in higher["minutes"]:
        if row["player_id"] == 12:
            row["p_60_plus"] = min(1.0, row["p_60_plus"] + 0.2)
            row["p_start"] = min(1.0, row["p_start"] + 0.2)
    high_result, _ = _build(conn, higher)
    base, high = _row_for(base_result, 12), _row_for(high_result, 12)
    assert high["appearance_xpts"] >= base["appearance_xpts"]
    assert high["clean_sheet_xpts"] >= base["clean_sheet_xpts"]


# 7 ------------------------------------------------------------------


def test_price_changes_do_not_change_football_xpts():
    conn = connect_database(":memory:")
    _seed(conn)
    result, _ = _build(conn, _default_inputs())
    before = _strip(result["rows"])
    from fpl_brain import repositories as repo
    with conn:
        run = repo.create_fetch_run(conn, "fetch_fpl")
        repo.insert_snapshots(conn, [PlayerSnapshotRecord(player_id=10, captured_at="2026-09-10T08:00:00Z",
                                                           now_cost=200, raw_json={})], run)
    after = _strip(xpts.build_xpts_projections(
        conn, event=EVENT, cutoff=CUTOFF, minutes_run_id=result["meta"]["minutes_run_id"],
        team_run_id=result["meta"]["team_run_id"], team_baseline_run_id=result["meta"]["team_baseline_run_id"],
        rate_run_id=result["meta"]["rate_run_id"])["rows"])
    key = lambda r: (r["player_id"], r["fixture_id"], r["total_xpts"], r["goal_xpts"], r["appearance_xpts"])
    assert sorted(key(r) for r in after) == sorted(key(r) for r in before)


# 8 ------------------------------------------------------------------


def test_ownership_does_not_change_football_xpts():
    conn = connect_database(":memory:")
    _seed(conn)
    result, _ = _build(conn, _default_inputs())
    before = _strip(result["rows"])
    with conn:
        conn.execute("""INSERT INTO squad_picks(entry_id, event, player_id, position, synced_at, raw_json)
                        VALUES (241392, 4, 10, 1, '2026-09-10T08:00:00Z', '{}')""")
    after = _strip(xpts.build_xpts_projections(
        conn, event=EVENT, cutoff=CUTOFF, minutes_run_id=result["meta"]["minutes_run_id"],
        team_run_id=result["meta"]["team_run_id"], team_baseline_run_id=result["meta"]["team_baseline_run_id"],
        rate_run_id=result["meta"]["rate_run_id"])["rows"])
    assert [r["total_xpts"] for r in after] == [r["total_xpts"] for r in before]


# 9 ------------------------------------------------------------------


def test_double_gameweek_produces_two_rows():
    conn = connect_database(":memory:")
    _seed(conn)
    inputs = _default_inputs()
    inputs["minutes"].append({"player_id": 10, "fixture_id": 101, **_minutes_payload()})
    inputs["team"].append({"fixture_id": 101, "team_id": 1, "opponent_id": 2, "venue": "away", **_team_payload()})
    inputs["team"].append({"fixture_id": 101, "team_id": 2, "opponent_id": 1, "venue": "home", **_team_payload()})
    inputs["baseline"].append({"fixture_id": 101, "team_id": 1, "opponent_id": 2, "venue": "away", **_team_payload()})
    inputs["baseline"].append({"fixture_id": 101, "team_id": 2, "opponent_id": 1, "venue": "home", **_team_payload()})
    result, _ = _build(conn, inputs)
    fixtures = sorted(r["fixture_id"] for r in result["rows"] if r["player_id"] == 10)
    assert fixtures == [100, 101]


# 10 -----------------------------------------------------------------


def test_blank_gameweek_produces_no_rows_for_that_player():
    conn = connect_database(":memory:")
    _seed(conn)
    # Player 12 has no minutes row at all (as if their team had no fixture).
    inputs = _default_inputs()
    inputs["minutes"] = [m for m in inputs["minutes"] if m["player_id"] != 12]
    result, _ = _build(conn, inputs)
    assert [r for r in result["rows"] if r["player_id"] == 12] == []


# 11 -----------------------------------------------------------------


def test_deterministic_for_same_inputs():
    conn = connect_database(":memory:")
    _seed(conn)
    inputs = _default_inputs()
    run_ids = _freeze(conn, **inputs)
    first = xpts.build_xpts_projections(conn, event=EVENT, cutoff=CUTOFF, minutes_run_id=run_ids["minutes"],
                                        team_run_id=run_ids["team"], team_baseline_run_id=run_ids["team_baseline"],
                                        rate_run_id=run_ids["rates"])
    second = xpts.build_xpts_projections(conn, event=EVENT, cutoff=CUTOFF, minutes_run_id=run_ids["minutes"],
                                         team_run_id=run_ids["team"], team_baseline_run_id=run_ids["team_baseline"],
                                         rate_run_id=run_ids["rates"])
    assert _strip(first["rows"]) == _strip(second["rows"])


# 12 -----------------------------------------------------------------


def test_frozen_xpts_are_immutable():
    conn = connect_database(":memory:")
    _seed(conn)
    result, run_ids = _build(conn, _default_inputs())
    with conn:
        run_id = analytics.create_projection_run(
            conn, model_family=analytics.XPTS_MODEL_FAMILY, model_version=xpts.XPTS_MODEL_VERSION,
            planning_event=EVENT, planning_context_hash="t", data_cutoff=CUTOFF, scouting_cutoff=None,
            official_run_ids={}, config_hash="x", deadline_status="PRE_DEADLINE",
        )
        analytics.freeze_xpts_projection(
            conn, run_id, player_id=10, fixture_id=100, event=EVENT, team_id=1, opponent_id=2, position="MID",
            minutes_run_id=run_ids["minutes"], team_run_id=run_ids["team"], rate_run_id=run_ids["rates"],
            payload={"total_xpts": 1.0}, model_version=xpts.XPTS_MODEL_VERSION,
            scoring_rules_version="test",
        )
        analytics.finish_projection_run(conn, run_id, "complete")
    with pytest.raises(Exception):
        conn.execute("UPDATE player_fixture_xpts_projections SET position='FWD' WHERE projection_run_id=?", (run_id,))
    with pytest.raises(Exception):
        conn.execute("DELETE FROM player_fixture_xpts_projections WHERE projection_run_id=?", (run_id,))
    assert len(analytics.xpts_projections(conn, run_id)) == 1


# 13 -----------------------------------------------------------------


def test_config_change_changes_config_hash():
    assert xpts.XPtsConfig().config_hash() != xpts.XPtsConfig(fpl_assist_mapping_coefficient=0.9).config_hash()


# 14 -----------------------------------------------------------------


def test_no_lookahead_from_post_cutoff_data():
    conn = connect_database(":memory:")
    _seed(conn)
    inputs = _default_inputs()
    run_ids = _freeze(conn, **inputs)
    first = xpts.build_xpts_projections(conn, event=EVENT, cutoff=CUTOFF, minutes_run_id=run_ids["minutes"],
                                        team_run_id=run_ids["team"], team_baseline_run_id=run_ids["team_baseline"],
                                        rate_run_id=run_ids["rates"])
    # A completed fixture and a snapshot captured AFTER the cutoff must not matter.
    from fpl_brain import repositories as repo
    with conn:
        repo.upsert_fixtures(conn, [FixtureRecord(id=200, event=1, team_h=1, team_a=2, finished=1, started=1,
                                                  team_h_score=9, team_a_score=0,
                                                  kickoff_time="2026-09-11T14:00:00Z", raw_json={})])
        repo.upsert_player_gameweeks(conn, [__import__("fpl_brain.models", fromlist=["PlayerGameweekRecord"]).PlayerGameweekRecord(
            player_id=11, event=1, fixture_id=200, minutes=90, expected_goals=5.0, source="element_summary", raw_json={})])
        run = repo.create_fetch_run(conn, "fetch_fpl")
        repo.insert_snapshots(conn, [PlayerSnapshotRecord(player_id=10, captured_at="2026-09-11T08:00:00Z",
                                                          now_cost=99, status="a", ep_next=99.0, raw_json={})], run)
    second = xpts.build_xpts_projections(conn, event=EVENT, cutoff=CUTOFF, minutes_run_id=run_ids["minutes"],
                                         team_run_id=run_ids["team"], team_baseline_run_id=run_ids["team_baseline"],
                                         rate_run_id=run_ids["rates"])
    assert _strip(first["rows"]) == _strip(second["rows"])


# 15 -----------------------------------------------------------------


def test_player_xg_sum_sanity_and_residual_bucket():
    conn = connect_database(":memory:")
    _seed(conn)
    inputs = _default_inputs()
    # Team lambda far above the player mass -> a positive unmodelled residual.
    for row in inputs["team"]:
        if row["team_id"] == 1:
            row["expected_goals_for"] = 3.0
    result, _ = _build(conn, inputs)
    buckets = result["meta"]["residual_buckets"]
    key = "100:1"
    assert buckets[key]["residual_bucket"] > 0.0
    assert buckets[key]["cap_applied"] is False


# 16 -----------------------------------------------------------------


def test_material_team_xg_excess_triggers_scaling_and_flag():
    conn = connect_database(":memory:")
    _seed(conn)
    inputs = _default_inputs()
    # Tiny team lambda with large player raw mass -> explicit downward scaling.
    for row in inputs["team"]:
        if row["team_id"] == 1:
            row["expected_goals_for"] = 0.4
    for row in inputs["rates"]:
        if row["component"] == "xG_per90":
            row["posterior_mean"] = 1.5
    result, _ = _build(conn, inputs)
    capped = [r for r in result["rows"] if r["team_id"] == 1 and "TEAM_XG_CAP_APPLIED" in r["risk_flags"]]
    assert capped
    for row in capped:
        assert row["xg_scale_factor"] < 1.0
        assert row["adjusted_expected_xg"] < row["raw_expected_xg"]


# 17 -----------------------------------------------------------------


def test_never_scales_up_to_consume_residual():
    conn = connect_database(":memory:")
    _seed(conn)
    inputs = _default_inputs()
    for row in inputs["team"]:
        if row["team_id"] == 1:
            row["expected_goals_for"] = 4.0
    result, _ = _build(conn, inputs)
    for row in result["rows"]:
        assert row["xg_scale_factor"] <= 1.0
        assert row["adjusted_expected_xg"] <= row["raw_expected_xg"] + 1e-9


# 18 -----------------------------------------------------------------


def test_no_separate_penalty_component():
    conn = connect_database(":memory:")
    _seed(conn)
    result, _ = _build(conn, _default_inputs())
    row = _row_for(result, 10)
    assert not any("penalt" in key for key in row)
    assert row["provenance"]["penalties"].startswith("embedded_in_xG")
    # Non-GK never has a save component, and no penalty points exist anywhere.
    assert row["save_xpts"] == 0.0


# 19 -----------------------------------------------------------------


def test_xg_is_named_xg_not_npxg():
    conn = connect_database(":memory:")
    _seed(conn)
    result, _ = _build(conn, _default_inputs())
    row = _row_for(result, 10)
    assert "adjusted_expected_xg" in row
    assert not any("NPxG" in key for key in row)
    # NPxG separation is explicitly disclaimed, not silently assumed.
    assert row["provenance"]["penalties"].startswith("embedded_in_xG")


# 20 -----------------------------------------------------------------


def test_role_modifier_only_acts_through_frozen_rate_input():
    conn = connect_database(":memory:")
    _seed(conn)
    result, _ = _build(conn, _default_inputs())
    row = _row_for(result, 10)
    assert row["goal_xpts"] == pytest.approx(row["adjusted_expected_xg"] * 5, abs=1e-4)  # MID = 5
    assert row["adjusted_expected_xg"] == pytest.approx(
        row["raw_expected_xg"] * row["xg_scale_factor"], abs=1e-5
    )


# 21 -----------------------------------------------------------------


def test_team_minutes_coherence_diagnostics_generated():
    conn = connect_database(":memory:")
    _seed(conn)
    result, _ = _build(conn, _default_inputs())
    coherence = result["coherence"]
    assert coherence["sides"]
    side = coherence["sides"][0]
    for key in ("expected_starters_sum", "expected_appearances_sum", "expected_minutes_sum",
                "starter_deviation_from_11", "minutes_deviation_from_990", "risk_flags"):
        assert key in side


# 22 -----------------------------------------------------------------


def test_core_plus_soft_equals_total_exactly():
    conn = connect_database(":memory:")
    _seed(conn)
    result, _ = _build(conn, _default_inputs())
    for row in result["rows"]:
        assert row["core_xpts"] + row["soft_xpts"] == pytest.approx(row["total_xpts"], abs=1e-9)


# 23 -----------------------------------------------------------------


def test_team_baseline_sensitivity_generated_separately():
    conn = connect_database(":memory:")
    _seed(conn)
    result, _ = _build(conn, _default_inputs())
    row = _row_for(result, 10)
    assert row["total_xpts_without_team_model"] is not None
    assert row["team_model_sensitivity"] is not None
    assert row["team_model_sensitivity"] == pytest.approx(
        abs(row["total_xpts"] - row["total_xpts_without_team_model"]), abs=1e-9
    )


# 24 -----------------------------------------------------------------


def test_official_ep_next_baseline_stored_separately():
    conn = connect_database(":memory:")
    _seed(conn)
    result, _ = _build(conn, _default_inputs())
    row = _row_for(result, 10)
    assert row["official_ep_next"] == 6.5
    # ep_next is never used as an input to our own model.
    assert row["total_xpts"] != row["official_ep_next"]
