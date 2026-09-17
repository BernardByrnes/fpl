"""Deterministic walk-forward scoreboard: hand-computed acceptance world.

Every expected number in this file was derived by hand from the definition before
the implementation was run.  The synthetic world is designed so that each metric
lands on a different value and each degenerate case (constant predictions, a
double gameweek, a zero-minute genuine non-appearance, a scheduled placeholder, a
missing projection, a blank Gameweek) is present at least once.

Reference world (event 5 is a DOUBLE gameweek for teams 1 and 2)
----------------------------------------------------------------
players  teams: 10->1, 11->1, 12->2, 13->3 (no fixture), 14->2, 15->1, 16->1 (no projection)
fixtures:       100 = 1 v 2, 101 = 2 v 1, both played and both event 5

realised event points:  10 -> 11, 11 -> 4, 12 -> 3, 14 -> 0
model event xPts:       10 -> 7.5, 11 -> 2.0, 12 -> 4.5, 14 -> 1.0
"""

from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path

import pytest

from fpl_brain import analytics
from fpl_brain import walk_forward as wf
from fpl_brain import walk_forward_metrics as wm
from fpl_brain import walk_forward_scoreboard as sb
from fpl_brain.database import connect_database

CUTOFF = "2026-09-10T12:00:00Z"
CODE_SNAPSHOT = "sha256:" + "a" * 64
XPTS_RUN_VERSION = "xpts_v1.4.1"
CLOSURE_MINUTES_VERSION = "minutes_v1.5.2"
LABEL_MINUTES_VERSION = "minutes_v1.7.0"
MC_VERSION = "mc_v1.3.0"
DISTRIBUTION_BASIS = "CORE (bonus deterministic, variance unmodelled)"
TOP_K = 2

PLAYERS = {10: 1, 11: 1, 12: 2, 13: 3, 14: 2, 15: 1, 16: 1}
FIXTURES = {100: (1, 2), 101: (2, 1)}
#: (event, player_id), ascending — the four eligible and scored observations.
EVALUABLE = [(5, 10), (5, 11), (5, 12), (5, 14)]
MODEL_POINTS = [7.5, 2.0, 4.5, 1.0]
REALISED_POINTS = [11.0, 4.0, 3.0, 0.0]

#: total xPts per (player, fixture); a double gameweek is the SUM of these.
XPTS_TOTALS = {
    (10, 100): 3.0, (10, 101): 4.5,
    (11, 100): 1.0, (11, 101): 1.0,
    (12, 100): 2.0, (12, 101): 2.5,
    (14, 100): 0.5, (14, 101): 0.5,
    (15, 100): 1.0, (15, 101): 1.0,
}

#: The four persisted xPts probabilities, per (player, fixture), in the order
#: (p_start, p_60_plus, clean_sheet_probability, defcon_p_hit).
PROBABILITIES = {
    (10, 100): (0.9, 0.8, 0.6, 0.5),
    (10, 101): (0.8, 0.7, 0.5, 0.5),
    (11, 100): (0.2, 0.1, 0.1, 0.5),
    (11, 101): (0.7, 0.6, 0.55, 0.5),
    (12, 100): (0.85, 0.75, 0.5, 0.5),
    (12, 101): (0.3, 0.2, 0.15, 0.5),
    (14, 100): (0.1, 0.05, 0.05, 0.5),
    (14, 101): (0.1, 0.05, 0.05, 0.5),
    # Player 15's outcome row is a scheduled placeholder, so these projections are
    # present but must never be scored.
    (15, 100): (0.5, 0.5, 0.5, 0.5),
    (15, 101): (0.5, 0.5, 0.5, 0.5),
}

#: (player, fixture, minutes, starts, total_points, goals_scored, assists,
#:  clean_sheets, goals_conceded, saves, yellow_cards, defensive_contribution)
OUTCOMES = [
    (10, 100, 90, 1, 6, 0, 0, 1, 0, 0, 0, 15),
    (10, 101, 90, 1, 5, 0, 0, 0, 1, 0, 0, 8),
    (11, 100, 30, 0, 1, 0, 0, 0, 0, 0, 0, 12),
    (11, 101, 90, 1, 3, 0, 0, 0, 0, 0, 0, 20),
    (12, 100, 75, 1, 2, 0, 0, 1, 0, 0, 0, 5),
    (12, 101, 20, 0, 1, 0, 0, 0, 0, 0, 0, 13),
    (14, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    (14, 101, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
]

#: (player, fixture, q10, q25, q75, q90) for the Monte Carlo grid.
QUANTILES = {
    (10, 100): (0.0, 1.0, 3.0, 5.0),
    (10, 101): (0.0, 1.0, 3.0, 5.0),
    (11, 100): (0.0, 1.0, 3.0, 5.0),
    (11, 101): (0.0, 1.0, 3.0, 5.0),
    (12, 100): (-2.0, 1.0, 3.0, 5.0),
    (12, 101): (-2.0, 1.0, 3.0, 5.0),
    (14, 100): (-2.0, 1.0, 3.0, 5.0),
    (14, 101): (-2.0, 1.0, 3.0, 5.0),
}

#: (player, fixture) -> expected minutes, from the resolved minutes run.
EXPECTED_MINUTES = {
    (10, 100): 85.0, (10, 101): 85.0,
    (11, 100): 40.0, (11, 101): 80.0,
    (12, 100): 70.0, (12, 101): 30.0,
    (14, 100): 5.0, (14, 101): 5.0,
}

EP_NEXT_VALUES = {(5, 10): 5.0, (5, 11): 4.0, (5, 12): 2.0, (5, 14): 0.5}
RECENT_POINTS_VALUES = {(5, 10): 2.0, (5, 11): 2.0, (5, 12): 2.0, (5, 14): 2.0}
#: Deliberately short: player 14 has NO per-90 value, so this arm cannot cover the
#: shared population and must be reported without any metric.
NAIVE_P90_VALUES = {(5, 10): 4.0, (5, 11): 3.0, (5, 12): 2.0}


def _world(
    conn: sqlite3.Connection,
    *,
    reverse_inserts: bool = False,
    probability_overrides: dict[tuple[int, int], dict] | None = None,
    quantile_overrides: dict[tuple[int, int], dict] | None = None,
    minutes_run_overrides: dict[tuple[int, int], int] | None = None,
    xpts_total_overrides: dict[tuple[int, int], float] | None = None,
    extra_evaluable_player: bool = False,
    extra_projection_only: bool = False,
    ambiguous_minutes_dependency: bool = False,
):
    """Build the reference league.

    The projection tables are immutable by database trigger (no UPDATE, no DELETE),
    so every corrupt-value case is injected HERE, at INSERT time.  That is a
    property of the system worth keeping: it is why these tests cannot mutate a
    scored row after the fact.
    """

    from fpl_brain import repositories as repo
    from fpl_brain.models import (
        EventRecord,
        FixtureRecord,
        PlayerRecord,
        PositionRecord,
        TeamRecord,
    )

    probability_overrides = probability_overrides or {}
    quantile_overrides = quantile_overrides or {}
    minutes_run_overrides = minutes_run_overrides or {}
    xpts_total_overrides = xpts_total_overrides or {}

    with conn:
        repo.upsert_teams(conn, [TeamRecord(id=t, name=f"Team {t}") for t in (1, 2, 3)])
        repo.upsert_positions(
            conn,
            [PositionRecord(id=p, singular_name_short=n) for p, n in ((1, "GKP"), (2, "DEF"), (3, "MID"), (4, "FWD"))],
        )
        repo.upsert_players(
            conn,
            [
                PlayerRecord(id=pid, web_name=f"P{pid}", full_name=f"Player {pid}", team_id=team, element_type=3)
                for pid, team in PLAYERS.items()
            ],
        )
        repo.upsert_events(
            conn,
            [EventRecord(id=5, finished=1, data_checked=1, deadline_time="2026-09-05T17:30:00Z", raw_json={})],
        )
        repo.upsert_fixtures(
            conn,
            [
                FixtureRecord(id=fid, event=5, team_h=h, team_a=a, finished=1, started=1,
                              kickoff_time="2026-09-06T14:00:00Z", raw_json={})
                for fid, (h, a) in FIXTURES.items()
            ],
        )

    outcome_rows = list(OUTCOMES)
    if reverse_inserts:
        outcome_rows = list(reversed(outcome_rows))
    with conn:
        for row in outcome_rows:
            conn.execute(
                "INSERT INTO player_gameweeks(player_id, event, fixture_id, minutes, starts, total_points,"
                " goals_scored, assists, clean_sheets, goals_conceded, saves, yellow_cards,"
                " defensive_contribution, source, updated_at, raw_json)"
                " VALUES (?,5,?,?,?,?,?,?,?,?,?,?,?,'element_summary','2026-09-07T09:00:00Z','{}')",
                row,
            )
        # Player 15's stored row is the canonical scheduled placeholder: minutes 0
        # and EVERY performance column NULL.  starts=0/total_points=0 would be
        # genuine evidence of a non-appearance, not a placeholder.
        conn.execute(
            "INSERT INTO player_gameweeks(player_id, event, fixture_id, minutes, source, updated_at, raw_json)"
            " VALUES (15,5,100,0,'element_summary','2026-09-05T08:00:00Z','{}')"
        )
        if extra_evaluable_player:
            # Player 16 gains both a projection and a real outcome, so he becomes a
            # fifth eligible and scored observation.
            for fixture_id in sorted(FIXTURES):
                conn.execute(
                    "INSERT INTO player_gameweeks(player_id, event, fixture_id, minutes, starts, total_points,"
                    " goals_scored, assists, clean_sheets, goals_conceded, saves, yellow_cards,"
                    " defensive_contribution, source, updated_at, raw_json)"
                    " VALUES (16,5,?,60,1,2,0,0,0,0,0,0,0,'element_summary','2026-09-07T09:00:00Z','{}')",
                    (int(fixture_id),),
                )

    xpts_run = analytics.create_projection_run(
        conn, model_family="xpts_v1", model_version=XPTS_RUN_VERSION, planning_event=5,
        planning_context_hash="ctx", data_cutoff=CUTOFF, scouting_cutoff=None,
        official_run_ids={}, source_snapshot_sha256=CODE_SNAPSHOT,
    )
    baseline_run = analytics.create_projection_run(
        conn, model_family="baseline", model_version=analytics.BASELINE_MODEL_VERSION, planning_event=5,
        planning_context_hash="ctx", data_cutoff=CUTOFF, scouting_cutoff=None,
        official_run_ids={}, source_snapshot_sha256=CODE_SNAPSHOT,
    )
    minutes_run = analytics.create_projection_run(
        conn, model_family="minutes_v1", model_version=CLOSURE_MINUTES_VERSION, planning_event=5,
        planning_context_hash="ctx", data_cutoff=CUTOFF, scouting_cutoff=None,
        official_run_ids={}, source_snapshot_sha256=CODE_SNAPSHOT,
    )
    #: A DIFFERENT minutes run, declared by the bundle's family label.  The
    #: component layer must follow the xPts dependency closure and must not pick
    #: this one, so its version is deliberately distinguishable.
    label_minutes_run = analytics.create_projection_run(
        conn, model_family="minutes_v1", model_version=LABEL_MINUTES_VERSION, planning_event=5,
        planning_context_hash="ctx", data_cutoff=CUTOFF, scouting_cutoff=None,
        official_run_ids={}, source_snapshot_sha256=CODE_SNAPSHOT,
    )
    mc_run = analytics.create_projection_run(
        conn, model_family="monte_carlo_v1", model_version=MC_VERSION, planning_event=5,
        planning_context_hash="ctx", data_cutoff=CUTOFF, scouting_cutoff=None,
        official_run_ids={}, source_snapshot_sha256=CODE_SNAPSHOT,
    )

    pairs = list(XPTS_TOTALS)
    if reverse_inserts:
        pairs = list(reversed(pairs))
    if extra_evaluable_player or extra_projection_only:
        pairs.extend([(16, 100), (16, 101)])
    if ambiguous_minutes_dependency:
        minutes_run_overrides = {**minutes_run_overrides, (11, 100): label_minutes_run}
    with conn:
        for player_id, fixture_id in pairs:
            team_id = PLAYERS[player_id]
            opponent = FIXTURES[fixture_id][1] if FIXTURES[fixture_id][0] == team_id else FIXTURES[fixture_id][0]
            p_start, p_60, cs_prob, defcon = PROBABILITIES.get((player_id, fixture_id), (0.5, 0.5, 0.5, 0.5))
            payload = {
                "total_xpts": xpts_total_overrides.get((player_id, fixture_id),
                                                       XPTS_TOTALS.get((player_id, fixture_id), 3.0)),
                "p_start": p_start,
                "p_60_plus": p_60,
                "clean_sheet_probability": cs_prob,
                "defcon_p_hit": defcon,
            }
            payload.update(probability_overrides.get((player_id, fixture_id), {}))
            conn.execute(
                "INSERT INTO player_fixture_xpts_projections(projection_run_id, player_id, fixture_id,"
                " event, team_id, opponent_id, position, minutes_run_id, team_run_id, rate_run_id,"
                " payload_json, model_version, scoring_rules_version, generated_at)"
                " VALUES (?,?,?,5,?,?,'MID',?,1,1,?,?, 'fpl_scoring_2026_27_v1.0.0','2026-09-10T12:00:00Z')",
                (xpts_run, int(player_id), int(fixture_id), int(team_id), int(opponent),
                 int(minutes_run_overrides.get((player_id, fixture_id), minutes_run)),
                 json.dumps(payload), XPTS_RUN_VERSION),
            )
        for player_id, fixture_id in sorted(QUANTILES):
            q10, q25, q75, q90 = QUANTILES[(player_id, fixture_id)]
            team_id = PLAYERS[player_id]
            home, away = FIXTURES[fixture_id]
            opponent = away if home == team_id else home
            payload = {
                "q10": q10, "q25": q25, "q50": q25 + 1.0, "q75": q75, "q90": q90,
                "mean_core": (q25 + q75) / 2.0,
                "distribution_basis": DISTRIBUTION_BASIS,
            }
            payload.update(quantile_overrides.get((player_id, fixture_id), {}))
            conn.execute(
                "INSERT INTO monte_carlo_distributions(projection_run_id, player_id, fixture_id, event,"
                " team_id, opponent_id, position, xpts_run_id, minutes_run_id, team_run_id, rate_run_id,"
                " payload_json, model_version, generated_at)"
                " VALUES (?,?,?,5,?,?,'MID',?,?,1,1,?,?, '2026-09-10T12:00:00Z')",
                (mc_run, int(player_id), int(fixture_id), int(team_id), int(opponent),
                 xpts_run, int(minutes_run), json.dumps(payload), MC_VERSION),
            )
        for player_id, fixture_id in pairs:
            if (player_id, fixture_id) not in EXPECTED_MINUTES:
                continue
            conn.execute(
                "INSERT INTO frozen_predictions(projection_run_id, kind, player_id, fixture_id, event,"
                " payload_json, model_version, generated_at) VALUES (?,?,?,?,5,?,?,'2026-09-10T12:00:00Z')",
                (minutes_run, analytics.MINUTES_V1_KIND, int(player_id), int(fixture_id),
                 json.dumps({"expected_minutes": EXPECTED_MINUTES[(player_id, fixture_id)]}),
                 CLOSURE_MINUTES_VERSION),
            )
        for kind, values in (
            (analytics.EP_NEXT_KIND, EP_NEXT_VALUES),
            (analytics.RECENT_POINTS_KIND, RECENT_POINTS_VALUES),
            (analytics.NAIVE_P90_KIND, NAIVE_P90_VALUES),
        ):
            for (event, player_id), value in sorted(values.items()):
                conn.execute(
                    "INSERT INTO frozen_predictions(projection_run_id, kind, player_id, fixture_id, event,"
                    " payload_json, model_version, generated_at) VALUES (?,?,?,NULL,?,?,?,'2026-09-10T12:00:00Z')",
                    (baseline_run, kind, int(player_id), int(event), json.dumps({"value": float(value)}),
                     analytics.BASELINE_MODEL_VERSION),
                )
    with conn:
        for run_id in (xpts_run, baseline_run, minutes_run, label_minutes_run, mc_run):
            conn.execute("UPDATE projection_runs SET status='complete' WHERE id=?", (int(run_id),))

    artifact = {
        "four_gw_certification_identity": "sha256:" + "b" * 64,
        "planning_cutoff": CUTOFF,
        "certified_bundles": {
            "5": {
                "event": 5,
                "cutoff": CUTOFF,
                "runs": {
                    "xpts_v1": xpts_run,
                    "minutes_v1": label_minutes_run,
                    "monte_carlo_v1": mc_run,
                },
                "model_versions": {
                    "xpts_v1": XPTS_RUN_VERSION,
                    "minutes_v1": LABEL_MINUTES_VERSION,
                    "monte_carlo_v1": MC_VERSION,
                },
                "code_snapshot_sha256": CODE_SNAPSHOT,
            }
        },
    }
    return {
        "xpts_run": xpts_run,
        "baseline_run": baseline_run,
        "minutes_run": minutes_run,
        "label_minutes_run": label_minutes_run,
        "mc_run": mc_run,
        "artifact": artifact,
    }


@pytest.fixture()
def world(tmp_path):
    conn = connect_database(tmp_path / "scoreboard.db")
    try:
        yield conn, _world(conn)
    finally:
        conn.close()


def _scoreboard(conn, world, **kwargs):
    kwargs.setdefault("top_k", TOP_K)
    return sb.build_scoreboard(conn, artifact=world["artifact"], events=[5], **kwargs)


def _arm(scoreboard, name):
    for entry in scoreboard["arms"]:
        if entry["arm"] == name:
            return entry
    raise AssertionError(f"arm {name} is absent from the scoreboard")


# ---------------------------------------------------------------------------
# The hand-computed acceptance world
# ---------------------------------------------------------------------------


def test_the_acceptance_world_is_fully_evaluated(world):
    conn, built = world
    scoreboard = _scoreboard(conn, built)
    assert scoreboard["status"] == sb.STATUS_EVALUATED
    assert scoreboard["status_reasons"] == []
    assert scoreboard["target_events"] == [5]


def test_hand_computed_population_and_coverage(world):
    conn, built = world
    scoreboard = _scoreboard(conn, built)
    population = scoreboard["population"]
    assert population["candidates"] == 7, "ten, eleven, twelve, thirteen, fourteen, fifteen, sixteen"
    assert population["evaluated"] == 4
    assert population["excluded"] == 3
    assert population["shared_comparison_n"] == 4
    # 4/7 rounded to the declared six places
    assert population["shared_comparison_coverage"] == pytest.approx(0.571429, abs=1e-6)
    assert population["evaluated_without_recorded_points"] == 0

    by_status = population["excluded_by_status"]
    assert by_status[wf.TARGET_NO_FIXTURE] == 1, "player 13's team has no fixture"
    assert by_status[wf.OUTCOME_PLACEHOLDER_EXCLUDED] == 1, "player 15's row is a placeholder"
    assert by_status[wf.MODEL_PROJECTION_MISSING] == 1, "player 16 has no projection"
    assert population["shared_comparison_digest"] == wf.canonical_population_digest(
        EVALUABLE, grain=wf.GRAIN_PLAYER_EVENT
    )


def test_hand_computed_model_arm(world):
    conn, built = world
    arm = _arm(_scoreboard(conn, built), sb.ARM_MODEL)

    # errors (predicted - actual): -3.5, -2.0, +1.5, +1.0
    assert arm["N"] == 4
    assert arm["mae"]["value"] == pytest.approx(2.0, abs=1e-6)            # 8.0 / 4
    assert arm["rmse"]["value"] == pytest.approx(math.sqrt(4.875), abs=1e-6)   # 19.5 / 4
    assert arm["bias"]["value"] == pytest.approx(-0.75, abs=1e-6)         # -3.0 / 4
    assert arm["median_ae"]["value"] == pytest.approx(1.75, abs=1e-6)     # (1.5 + 2.0) / 2
    # predicted ranks [4,2,3,1] against actual ranks [4,3,2,1]
    # covariance 4.0, both spreads 5.0 -> 4.0 / 5.0
    assert arm["spearman"]["value"] == pytest.approx(0.8, abs=1e-6)
    assert arm["spearman"]["status"] == wm.METRIC_OK
    assert arm["version"] == XPTS_RUN_VERSION
    assert arm["run_id"] == built["xpts_run"]


def test_hand_computed_top_k_with_ties_at_the_boundary(world):
    conn, built = world
    scoreboard = _scoreboard(conn, built)
    # k=2.  Predicted order (score desc, id asc): (7.5,10),(4.5,12),(2.0,11),(1.0,14) -> {10,12}
    #       Actual order:                          (11,10),(4,11),(3,12),(0,14)         -> {10,11}
    # intersection {10} -> 1/2
    assert _arm(scoreboard, sb.ARM_MODEL)["top_k_hit_rate"]["value"] == pytest.approx(0.5, abs=1e-6)
    assert scoreboard["identity"]["top_k"] == 2
    assert scoreboard["identity"]["top_k_tie_policy"] == sb.TOP_K_TIE_POLICY


def test_hand_computed_ep_next_arm(world):
    conn, built = world
    arm = _arm(_scoreboard(conn, built), analytics.EP_NEXT_KIND)
    # errors: -6, 0, -1, +0.5
    assert arm["mae"]["value"] == pytest.approx(1.875, abs=1e-6)          # 7.5 / 4
    assert arm["rmse"]["value"] == pytest.approx(math.sqrt(9.3125), abs=1e-6)  # 37.25 / 4
    assert arm["bias"]["value"] == pytest.approx(-1.625, abs=1e-6)        # -6.5 / 4
    assert arm["median_ae"]["value"] == pytest.approx(0.75, abs=1e-6)     # (0.5 + 1.0) / 2
    # A perfect rank ordering with a persistently low level: the two measurements
    # disagree on purpose, which is why both are reported.
    assert arm["spearman"]["value"] == pytest.approx(1.0, abs=1e-6)
    assert arm["top_k_hit_rate"]["value"] == pytest.approx(1.0, abs=1e-6)


def test_a_constant_baseline_has_an_undefined_correlation_but_real_error(world):
    conn, built = world
    arm = _arm(_scoreboard(conn, built), analytics.RECENT_POINTS_KIND)
    # The arm predicts 2.0 for everyone.  Its error is perfectly well defined...
    assert arm["mae"]["value"] == pytest.approx(3.5, abs=1e-6)            # 14 / 4
    assert arm["rmse"]["value"] == pytest.approx(math.sqrt(22.5), abs=1e-6)  # 90 / 4
    assert arm["bias"]["value"] == pytest.approx(-2.5, abs=1e-6)          # -10 / 4
    assert arm["median_ae"]["value"] == pytest.approx(2.0, abs=1e-6)      # (2 + 2) / 2
    # ... but its ordering does not exist, so the correlation is explicitly
    # unavailable rather than reported as 0 or 1.
    assert arm["spearman"]["status"] == wm.METRIC_ZERO_VARIANCE
    assert arm["spearman"]["value"] is None
    # Every prediction ties, so the ascending-id tie policy chooses the top two.
    assert arm["top_k_hit_rate"]["value"] == pytest.approx(1.0, abs=1e-6)


def test_hand_computed_per_event_breakdown(world):
    conn, built = world
    scoreboard = _scoreboard(conn, built)
    rows = [row for row in scoreboard["per_event"] if row["arm"] == sb.ARM_MODEL]
    assert len(rows) == 1
    assert rows[0]["event"] == 5 and rows[0]["N"] == 4
    assert rows[0]["mae"]["value"] == pytest.approx(2.0, abs=1e-6)
    assert rows[0]["rmse"]["value"] == pytest.approx(math.sqrt(4.875), abs=1e-6)
    assert rows[0]["bias"]["value"] == pytest.approx(-0.75, abs=1e-6)
    # One row per (arm, event) for the arms that were measurable: the model and the
    # two complete baselines.  The unavailable P90 arm contributes none.
    assert len(scoreboard["per_event"]) == 3
    assert {row["arm"] for row in scoreboard["per_event"]} == {
        sb.ARM_MODEL, analytics.EP_NEXT_KIND, analytics.RECENT_POINTS_KIND
    }


# ---------------------------------------------------------------------------
# Probability metrics
# ---------------------------------------------------------------------------


def test_hand_computed_brier_scores(world):
    """Brier values are read back from the artifact, so they carry its 6-place rounding."""

    conn, built = world
    scoreboard = _scoreboard(conn, built)
    by_metric = {entry["metric"]: entry for entry in scoreboard["probability_metrics"]}

    # p_start vs realised starts [1,1,0,1,1,0,0,0]
    #   0.3125 / 8 = 0.0390625 -> serialised at six places as 0.039062
    assert by_metric["BRIER_P_START"]["brier"]["value"] == pytest.approx(0.0390625, abs=1e-6)
    assert by_metric["BRIER_P_START"]["brier"]["value"] == 0.039062
    assert by_metric["BRIER_P_START"]["observed_rate"] == pytest.approx(0.5)
    assert by_metric["BRIER_P_START"]["brier_reference"]["value"] == pytest.approx(0.25, abs=1e-6)
    assert by_metric["BRIER_P_START"]["population"]["scored"] == 8

    # p_60_plus vs realised minutes>=60 [1,1,0,1,1,0,0,0]; 0.4075 / 8 = 0.0509375
    assert by_metric["BRIER_P_60_PLUS"]["brier"]["value"] == pytest.approx(0.0509375, abs=1e-6)

    # clean_sheet_probability vs realised clean-sheet POINTS [1,0,0,1,1,0,0,0]
    assert by_metric["BRIER_CLEAN_SHEET"]["brier"]["value"] == pytest.approx(0.1125, abs=1e-6)
    assert by_metric["BRIER_CLEAN_SHEET"]["observed_rate"] == pytest.approx(0.375)
    assert by_metric["BRIER_CLEAN_SHEET"]["brier_reference"]["value"] == pytest.approx(0.234375, abs=1e-6)

    # a constant 0.5 against a 0.5 base rate: exactly 0.25 per observation
    assert by_metric["BRIER_DEFCON"]["brier"]["value"] == pytest.approx(0.25, abs=1e-6)
    assert by_metric["BRIER_DEFCON"]["population"]["scored"] == 8

    for entry in scoreboard["probability_metrics"]:
        assert entry["range_check"] and entry["missing_policy"]
        assert entry["grain"] == wf.GRAIN_PLAYER_FIXTURE


def test_the_clean_sheet_target_is_clean_sheet_points_not_a_raw_clean_sheet_count(world):
    """Player 11 fixture 100 played 30 minutes, so no clean-sheet points were earned.

    A raw ``clean_sheets`` column would not distinguish this from an earned clean
    sheet; the declared outcome requires the 60-minute boundary as well.
    """

    conn, built = world
    scoreboard = _scoreboard(conn, built)
    entry = {item["metric"]: item for item in scoreboard["probability_metrics"]}["BRIER_CLEAN_SHEET"]
    assert "clean_sheet_points_for(position) > 0" in entry["realised_outcome"]
    assert "minutes >= 60" in entry["realised_outcome"]
    # 3 of 8, not the 4 that a raw-conceded-zero rule would give.
    assert entry["observed_rate"] == pytest.approx(0.375)


def test_a_probability_outside_the_unit_interval_fails_closed(tmp_path):
    conn = connect_database(tmp_path / "bad_prob.db")
    built = _world(conn, probability_overrides={(10, 100): {"p_start": 1.4}})
    with pytest.raises(sb.ScoreboardError) as failure:
        _scoreboard(conn, built)
    assert "outside [0, 1]" in str(failure.value)
    conn.close()


def test_a_placeholder_row_is_never_scored_in_a_probability_metric(world):
    conn, built = world
    scoreboard = _scoreboard(conn, built)
    # Player 15 has projections for both fixtures but a placeholder outcome row, so
    # the scored population is the eight real rows, never ten.
    for entry in scoreboard["probability_metrics"]:
        assert entry["population"]["scored"] == 8


# ---------------------------------------------------------------------------
# Quantile coverage
# ---------------------------------------------------------------------------


def test_hand_computed_quantile_coverage(world):
    conn, built = world
    scoreboard = _scoreboard(conn, built)
    block = scoreboard["quantile_coverage"]
    by_metric = {entry["metric"]: entry for entry in block["metrics"]}

    # realised CORE, hand-derived (appearance + clean sheet + DefCon):
    #   [5, 2, 3, 5, 3, 3, 0, 0]
    # q25=1, q75=3 for every row -> inside for the four values 2, 3, 3, 3 -> 4/8
    assert by_metric["CENTRAL_50_QUANTILE_COVERAGE"]["value"]["value"] == pytest.approx(0.5, abs=1e-6)
    assert by_metric["CENTRAL_50_QUANTILE_COVERAGE"]["mean_interval_width"]["value"] == pytest.approx(2.0)
    assert by_metric["CENTRAL_50_QUANTILE_COVERAGE"]["nominal_coverage"] == 0.50

    # q90=5 everywhere and q10 is 0 for the first four rows, -2 for the last four.
    # Realised 5 sits exactly on q90 and realised 0 sits exactly on q10 for the
    # first four rows: the interval is closed, so both count as covered -> 8/8.
    assert by_metric["CENTRAL_80_QUANTILE_COVERAGE"]["value"]["value"] == pytest.approx(1.0, abs=1e-6)
    # widths 5 for four rows and 7 for four rows -> 48/8
    assert by_metric["CENTRAL_80_QUANTILE_COVERAGE"]["mean_interval_width"]["value"] == pytest.approx(6.0)
    assert block["population"]["scored"] == 8
    assert block["population"]["quantiles_absent"] == 0
    assert block["distribution_basis"] == [DISTRIBUTION_BASIS]
    assert block["versions"] == [MC_VERSION]


def test_the_distribution_limitation_travels_with_the_coverage(world):
    conn, built = world
    scoreboard = _scoreboard(conn, built)
    qpolicy = scoreboard["identity"]["quantile_policy"]
    assert qpolicy["crps"] == "NOT_IMPLEMENTED (the artifact stores quantiles, never draws)"
    assert "NOT a statement of complete distribution calibration" in qpolicy["claim"]
    assert any("CRPS" in limitation for limitation in scoreboard["limitations"])
    assert any(
        "QUANTILE COVERAGE" in limitation for limitation in scoreboard["limitations"]
    )
    assert "CENTRAL_50_QUANTILE_COVERAGE" in qpolicy["metrics"]
    assert "CENTRAL_80_QUANTILE_COVERAGE" in qpolicy["metrics"]


def test_a_reversed_stored_interval_fails_closed(tmp_path):
    conn = connect_database(tmp_path / "bad_q.db")
    built = _world(conn, quantile_overrides={(10, 100): {"q25": 6.0, "q75": 1.0}})
    with pytest.raises(wm.MetricInputError):
        _scoreboard(conn, built)
    conn.close()


# ---------------------------------------------------------------------------
# Component diagnostics and the minutes dependency closure
# ---------------------------------------------------------------------------


def test_hand_computed_minutes_component_diagnostics(world):
    conn, built = world
    scoreboard = _scoreboard(conn, built)
    block = scoreboard["component_diagnostics"]
    by_metric = {entry["metric"]: entry for entry in block["metrics"]}
    # errors (expected - realised): -5,-5,+10,-10,-5,+10,+5,+5
    assert by_metric["MINUTES_MAE"]["value"]["value"] == pytest.approx(6.875, abs=1e-6)   # 55/8
    assert by_metric["MINUTES_BIAS"]["value"]["value"] == pytest.approx(0.625, abs=1e-6)  # 5/8
    assert block["population"]["scored"] == 8
    assert block["grain"] == wf.GRAIN_PLAYER_FIXTURE


def test_the_minutes_run_comes_from_the_dependency_closure_not_the_bundle_label(world):
    """The bundle's ``minutes_v1`` member is a LABEL, and it can be a different run.

    Here the bundle declares ``minutes_v1.7.0`` while the xPts rows were actually
    produced from ``minutes_v1.5.2``.  Resolving by family label would silently
    score the wrong run, so the closure is the authority and this proves it.
    """

    conn, built = world
    assert built["label_minutes_run"] != built["minutes_run"]
    resolved = sb.resolve_component_run(conn, built["xpts_run"])
    assert resolved == built["minutes_run"], "the bundle label was used instead of the closure"
    assert resolved != built["label_minutes_run"]

    block = _scoreboard(conn, built)["component_diagnostics"]
    assert block["run_ids_by_event"] == {5: built["minutes_run"]}
    assert block["versions"] == [CLOSURE_MINUTES_VERSION]
    assert block["bundle_label_used"] is False
    assert "dependency closure" in block["resolved_from"]

    identity_runs = _scoreboard(conn, built)["identity"]["per_event_runs"]
    assert identity_runs["minutes_v1"] == {5: built["minutes_run"]}
    assert identity_runs["xpts_v1"] == {5: built["xpts_run"]}
    assert identity_runs["monte_carlo_v1"] == {5: built["mc_run"]}


def test_an_ambiguous_minutes_dependency_fails_closed(tmp_path):
    """Two distinct minutes runs inside one xPts run cannot both be the dependency."""

    conn = connect_database(tmp_path / "ambiguous.db")
    built = _world(conn, ambiguous_minutes_dependency=True)
    with pytest.raises(sb.ScoreboardError) as failure:
        sb.resolve_component_run(conn, built["xpts_run"])
    assert "more than one minutes run" in str(failure.value)
    conn.close()


# ---------------------------------------------------------------------------
# The same-population gate, per comparison
# ---------------------------------------------------------------------------


def test_an_arm_that_cannot_cover_the_population_gets_no_metric_at_all(world):
    conn, built = world
    scoreboard = _scoreboard(conn, built)
    arm = _arm(scoreboard, analytics.NAIVE_P90_KIND)

    assert arm["status"] == sb.ARM_POPULATION_MISMATCH
    assert arm["N"] is None
    for field in ("mae", "rmse", "bias", "median_ae", "spearman", "top_k_hit_rate"):
        assert arm[field]["value"] is None, f"{field} was reported for a mismatched arm"
        assert arm[field]["status"] == sb.ARM_POPULATION_MISMATCH
    # The gate names the key that differs and how many, rather than leaving two
    # different N values to be spotted by eye.
    assert arm["coverage"]["model_only_n"] == 1
    assert arm["coverage"]["arm_only_n"] == 0
    assert arm["coverage"]["model_only_example"] == [[5, 14]]
    assert arm["coverage"]["shared_comparison_n"] == 4
    assert arm["coverage"]["population_digest"] != scoreboard["population"]["shared_comparison_digest"]


def test_the_other_arms_are_still_measured_when_one_arm_is_unavailable(world):
    conn, built = world
    scoreboard = _scoreboard(conn, built)
    measured = {arm["arm"] for arm in scoreboard["arms"] if arm["status"] == sb.ARM_OK}
    assert measured == {sb.ARM_MODEL, analytics.EP_NEXT_KIND, analytics.RECENT_POINTS_KIND}
    assert analytics.NAIVE_P90_KIND not in measured


def test_a_mismatch_that_only_changes_the_digest_is_still_caught(tmp_path):
    """Guards against a digest comparison that can never fail.

    Every arm is built from the same EVALUATED rows, so a digest taken over those
    rows would agree trivially.  This compares the SCORED key sets -- the keys an
    arm actually has a value for -- so a single absent value changes the digest and
    trips the gate.
    """

    conn = connect_database(tmp_path / "digest_gate.db")
    built = _world(conn)
    scoreboard = _scoreboard(conn, built)
    model_digest = scoreboard["population"]["shared_comparison_digest"]
    assert model_digest == wf.canonical_population_digest(EVALUABLE, grain=wf.GRAIN_PLAYER_EVENT)
    arm = _arm(scoreboard, analytics.NAIVE_P90_KIND)
    covered = [key for key in EVALUABLE if key != (5, 14)]
    assert arm["coverage"]["population_digest"] == wf.canonical_population_digest(
        covered, grain=wf.GRAIN_PLAYER_EVENT
    )
    assert arm["coverage"]["population_digest"] != model_digest
    conn.close()


def test_the_gate_names_both_sides_when_they_diverge():
    mismatch = wf.PopulationMismatch("x", model_only=[(5, 12)], baseline_only=[(5, 14)])
    assert mismatch.model_only == ((5, 12),)
    assert mismatch.baseline_only == ((5, 14),)


# ---------------------------------------------------------------------------
# Sample-size honesty
# ---------------------------------------------------------------------------


def test_a_tiny_sample_is_declared_insufficient_for_model_selection(world):
    conn, built = world
    scoreboard = _scoreboard(conn, built)
    sample = scoreboard["sample"]
    assert sample["target_events"] == 1
    assert sample["target_events_with_observations"] == 1
    assert sample["player_event_observations"] == 4
    assert sample["sample_interpretation"] == sb.SAMPLE_INSUFFICIENT
    assert sample["policy"]["policy_version"] == sb.SAMPLE_POLICY_VERSION
    assert "NOT a significance test" in sample["policy"]["basis"]


def test_no_superiority_language_appears_anywhere_in_the_artifact(world):
    conn, built = world
    payload = _scoreboard(conn, built)
    text = sb.canonical_bytes(payload).decode("utf-8").lower()
    # Claim PHRASES, not bare words: the sample policy legitimately contains the
    # negation ("no arm is described as better, superior, proven or calibrated"),
    # so a substring scan for "superior" would flag the disclaimer itself.
    for forbidden in (
        "is superior", "is better than", "outperforms", "winner", "best arm",
        "proven better", "is calibrated", "beats the",
    ):
        assert forbidden not in text, f"{forbidden!r} leaked into the artifact"
    assert "no arm is described as better, superior, proven or calibrated" in text
    assert "not as a calibrated predictive interval" in text
    assert payload["identity"]["ranking_policy"].startswith("NONE")
    assert payload["sample"]["sample_interpretation"] == sb.SAMPLE_INSUFFICIENT


# ---------------------------------------------------------------------------
# Per-event runs and identity bindings
# ---------------------------------------------------------------------------


def test_the_identity_binds_everything_needed_to_reproduce_the_measurement(world):
    conn, built = world
    identity = _scoreboard(conn, built)["identity"]
    assert identity["scoreboard_schema_version"] == sb.SCOREBOARD_SCHEMA_VERSION
    assert identity["metric_policy_version"] == wm.METRIC_POLICY_VERSION
    assert identity["planning_cutoff"] == CUTOFF
    assert identity["certification_identity"] == "sha256:" + "b" * 64
    assert identity["bias_convention"] == sb.BIAS_CONVENTION
    assert identity["top_k"] == TOP_K
    assert identity["headline_grain"] == wf.GRAIN_PLAYER_EVENT
    assert identity["baseline_kinds"] == list(wf.HEADLINE_POINTS_BASELINE_KINDS)
    assert identity["walk_forward_identity"]["missing_data_policy_version"] == wf.MISSING_DATA_POLICY_VERSION
    assert {d["field"] for d in identity["probability_metric_definitions"]} == {
        "p_start", "p_60_plus", "clean_sheet_probability", "defcon_p_hit"
    }
    assert "NONE" in identity["ranking_policy"]


def test_the_bias_convention_is_predicted_minus_actual_end_to_end(world):
    conn, built = world
    arm = _arm(_scoreboard(conn, built), sb.ARM_MODEL)
    predicted = sum(MODEL_POINTS)
    actual = sum(REALISED_POINTS)
    # The model under-predicts overall, so the signed bias must be negative.
    assert (predicted - actual) < 0
    assert arm["bias"]["value"] < 0
    assert arm["bias"]["value"] == pytest.approx((predicted - actual) / 4, abs=1e-6)


def test_the_headline_grain_sums_a_double_gameweek_and_never_expands_a_baseline(world):
    conn, built = world
    scoreboard = _scoreboard(conn, built)
    population = scoreboard["population"]
    assert population["shared_comparison_n"] == 4, "a DGW must not produce two rows"
    assert population["grain_note"] == sb.TARGET_GRAIN_NOTE
    arm = _arm(scoreboard, analytics.RECENT_POINTS_KIND)
    # The event-grain baseline predicts 2.0 once per player; a per-fixture expansion
    # would have made its error four points larger in magnitude.
    assert arm["bias"]["value"] == pytest.approx(-2.5, abs=1e-6)


# ---------------------------------------------------------------------------
# Determinism and immutability
# ---------------------------------------------------------------------------


def test_the_same_world_produces_byte_identical_canonical_json(world):
    conn, built = world
    first = sb.canonical_bytes(_scoreboard(conn, built))
    second = sb.canonical_bytes(_scoreboard(conn, built))
    assert first == second
    assert sb.scoreboard_digest(_scoreboard(conn, built)) == sb.scoreboard_digest(_scoreboard(conn, built))


def test_a_different_insertion_order_produces_the_same_digest(tmp_path):
    forward = connect_database(tmp_path / "forward.db")
    reverse = connect_database(tmp_path / "reverse.db")
    built_forward = _world(forward)
    built_reverse = _world(reverse, reverse_inserts=True)
    assert sb.scoreboard_digest(_scoreboard(forward, built_forward)) == sb.scoreboard_digest(
        _scoreboard(reverse, built_reverse)
    )
    forward.close()
    reverse.close()


def test_one_changed_prediction_changes_the_artifact_digest(tmp_path):
    """Same population, same outcomes, one prediction moved: the artifact must move."""

    conn = connect_database(tmp_path / "p1.db")
    before = sb.scoreboard_digest(_scoreboard(conn, _world(conn)))
    conn.close()

    conn = connect_database(tmp_path / "p2.db")
    after = sb.scoreboard_digest(_scoreboard(conn, _world(conn, xpts_total_overrides={(12, 100): 9.0})))
    conn.close()
    assert before != after


def test_one_changed_eligible_key_changes_both_digests(tmp_path):
    """Adding a genuinely eligible observation moves the population AND the artifact.

    Player 16 needs BOTH a projection and a real outcome: a projection alone would
    only become INPUT_EVIDENCE_UNAVAILABLE, which is deliberately not an eligible
    key, so the digest would (correctly) not move.
    """

    conn = connect_database(tmp_path / "k2.db")
    before = _scoreboard(conn, _world(conn))
    conn.close()

    conn = connect_database(tmp_path / "k3.db")
    after = _scoreboard(conn, _world(conn, extra_evaluable_player=True))
    try:
        assert before["population"]["shared_comparison_n"] == 4
        assert after["population"]["shared_comparison_n"] == 5
        assert before["population"]["shared_comparison_digest"] != after["population"]["shared_comparison_digest"]
        assert sb.scoreboard_digest(before) != sb.scoreboard_digest(after)
        assert _arm(after, sb.ARM_MODEL)["N"] == 5
    finally:
        conn.close()


def test_a_projection_without_an_outcome_is_not_an_eligible_key(tmp_path):
    """A projection with no stored observation is NOT scoreable and must not be.

    Player 16 has projections for both fixtures but no ``player_gameweeks`` row, so
    he cannot be scored and the digest does not move.
    """

    conn = connect_database(tmp_path / "k4.db")
    scoreboard = _scoreboard(conn, _world(conn, extra_projection_only=True))
    statuses = scoreboard["population"]["excluded_by_status"]
    assert statuses[wf.INPUT_EVIDENCE_UNAVAILABLE] == 1
    assert scoreboard["population"]["shared_comparison_n"] == 4
    assert _arm(scoreboard, sb.ARM_MODEL)["N"] == 4
    conn.close()


def test_a_changed_metric_policy_version_changes_the_digest(world, monkeypatch):
    conn, built = world
    before = sb.scoreboard_digest(_scoreboard(conn, built))
    monkeypatch.setattr(wm, "METRIC_POLICY_VERSION", "wf_metrics_v9.9.9")
    after_scoreboard = _scoreboard(conn, built)
    assert after_scoreboard["identity"]["metric_policy_version"] == "wf_metrics_v9.9.9"
    assert sb.scoreboard_digest(after_scoreboard) != before


def test_the_artifact_carries_no_wall_clock_timestamp(world):
    conn, built = world
    payload = json.loads(sb.canonical_bytes(_scoreboard(conn, built)))
    assert "generated_at" not in sb.canonical_bytes(_scoreboard(conn, built)).decode("utf-8")
    # The only time present is the supplied observation.
    assert sb.__dict__  # keep the module imported for clarity
    assert payload["identity"]["planning_cutoff"] == CUTOFF


def test_write_is_idempotent_for_the_same_identity(world, tmp_path):
    conn, built = world
    scoreboard = _scoreboard(conn, built)
    digest, json_path, md_path = sb.write_scoreboard(scoreboard, directory=tmp_path)
    assert json_path.exists() and md_path.exists()
    assert json_path.name == f"{digest.replace(':', '-')}.json"
    assert json_path.read_bytes() == sb.canonical_bytes(scoreboard)
    assert sb.scoreboard_digest(scoreboard) == digest
    # Writing the identical content again is a no-op, not an error.
    again_digest, again_json, _again_md = sb.write_scoreboard(scoreboard, directory=tmp_path)
    assert (again_digest, again_json) == (digest, json_path)
    assert again_json.read_bytes() == sb.canonical_bytes(scoreboard)


def test_the_same_identity_with_different_bytes_fails_closed(world, tmp_path):
    conn, built = world
    scoreboard = _scoreboard(conn, built)
    digest, json_path, _md = sb.write_scoreboard(scoreboard, directory=tmp_path)
    json_path.write_bytes(b"different bytes at the same identity")
    with pytest.raises(sb.ScoreboardIdentityCollision) as failure:
        sb.write_scoreboard(scoreboard, directory=tmp_path)
    assert digest in str(failure.value) or json_path.name in str(failure.value)


def test_the_artifact_path_is_derived_from_the_digest(tmp_path):
    digest = "sha256:" + "d" * 64
    assert sb.artifact_path(digest, directory=tmp_path).name == f"sha256-{'d' * 64}.json"
    assert sb.artifact_path(digest, directory=tmp_path, suffix=".md").name == f"sha256-{'d' * 64}.md"


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def test_the_markdown_rendering_exposes_the_required_sections(world):
    conn, built = world
    scoreboard = _scoreboard(conn, built)
    text = sb.render_markdown(scoreboard)
    assert "# Walk-forward scoreboard — EVALUATED" in text
    for expected in (
        "## Population",
        "### Excluded by status",
        "## Arms (measurements, not a ranking)",
        "| arm | version | N | MAE | RMSE | BIAS | MEDIAN_AE | SPEARMAN | TOP_K_HIT_RATE |",
        "## Per-event metrics",
        "## Probability metrics (Brier)",
        "## Quantile coverage",
        "## Component diagnostics",
        "## Limitations",
        scoreboard["population"]["shared_comparison_digest"],
        sb.STATUS_EVALUATED,
    ):
        assert expected in text, f"{expected!r} missing from the Markdown rendering"
    # The JSON is authoritative; the rendering is derived from it, never the reverse.
    assert sb.render_markdown(scoreboard) == sb.render_markdown(scoreboard)


# ---------------------------------------------------------------------------
# The blocked state: an unplayed event carries no metric at all
# ---------------------------------------------------------------------------


def test_an_unplayed_event_is_not_yet_evaluatable_and_reports_no_metric(tmp_path):
    conn = connect_database(tmp_path / "unplayed.db")
    from fpl_brain import repositories as repo
    from fpl_brain.models import EventRecord, FixtureRecord, PlayerRecord, PositionRecord, TeamRecord

    with conn:
        repo.upsert_teams(conn, [TeamRecord(id=1, name="T1"), TeamRecord(id=2, name="T2")])
        repo.upsert_positions(conn, [PositionRecord(id=3, singular_name_short="MID")])
        repo.upsert_players(conn, [PlayerRecord(id=10, web_name="P10", full_name="P10", team_id=1, element_type=3)])
        repo.upsert_events(conn, [EventRecord(id=5, finished=0, data_checked=0,
                                             deadline_time="2026-09-20T17:30:00Z", raw_json={})])
        repo.upsert_fixtures(conn, [FixtureRecord(id=100, event=5, team_h=1, team_a=2, finished=0, started=0,
                                                 kickoff_time="2026-09-20T14:00:00Z", raw_json={})])
    xpts_run = analytics.create_projection_run(
        conn, model_family="xpts_v1", model_version=XPTS_RUN_VERSION, planning_event=5,
        planning_context_hash="ctx", data_cutoff=CUTOFF, scouting_cutoff=None,
        official_run_ids={}, source_snapshot_sha256=CODE_SNAPSHOT,
    )
    with conn:
        conn.execute("UPDATE projection_runs SET status='complete' WHERE id=?", (int(xpts_run),))
    artifact = {
        "four_gw_certification_identity": "sha256:" + "c" * 64,
        "planning_cutoff": CUTOFF,
        "certified_bundles": {
            "5": {"event": 5, "cutoff": CUTOFF, "runs": {"xpts_v1": xpts_run},
                  "model_versions": {"xpts_v1": XPTS_RUN_VERSION}}
        },
    }
    scoreboard = sb.build_scoreboard(conn, artifact=artifact, events=[5], top_k=TOP_K)

    assert scoreboard["status"] == sb.STATUS_NOT_YET_EVALUATABLE
    assert scoreboard["arms"] == [], "an unplayed event must not produce an arm row"
    assert scoreboard["per_event"] == []
    assert scoreboard["probability_metrics"] == []
    assert scoreboard["quantile_coverage"] == {}
    assert scoreboard["component_diagnostics"] == {}
    assert scoreboard["sample"]["player_event_observations"] == 0
    assert scoreboard["sample"]["sample_interpretation"] == sb.SAMPLE_INSUFFICIENT
    assert scoreboard["population"]["shared_comparison_n"] == 0
    # Nothing scored, out of one candidate: 0.0 with the count beside it, not a
    # hidden blank and not a plausible-looking ratio.
    assert scoreboard["population"]["candidates"] == 1
    assert scoreboard["population"]["evaluated"] == 0
    assert scoreboard["population"]["shared_comparison_coverage"] == 0.0
    assert any("not a zero-score Gameweek" in reason for reason in scoreboard["limitations"])
    assert scoreboard["status_reasons"] and any(
        wf.OUTCOME_NOT_FINALISED in reason for reason in scoreboard["status_reasons"]
    )
    # No numeric metric anywhere: no zero MAE, no empty row pretending to be a result.
    text = sb.canonical_bytes(scoreboard).decode("utf-8")
    assert '"mae"' not in text and '"rmse"' not in text and '"bias"' not in text
    assert "NaN" not in text and "Infinity" not in text
    conn.close()


def test_a_partially_final_target_reports_the_events_it_could_score(tmp_path):
    conn = connect_database(tmp_path / "partial.db")
    built = _world(conn)
    from fpl_brain import repositories as repo
    from fpl_brain.models import EventRecord, FixtureRecord

    with conn:
        repo.upsert_events(conn, [EventRecord(id=6, finished=0, data_checked=0,
                                             deadline_time="2026-09-20T17:30:00Z", raw_json={})])
        repo.upsert_fixtures(conn, [FixtureRecord(id=200, event=6, team_h=1, team_a=2, finished=0, started=0,
                                                 kickoff_time="2026-09-20T14:00:00Z", raw_json={})])
    artifact = json.loads(json.dumps({**built["artifact"]}))
    artifact["certified_bundles"]["6"] = {
        "event": 6, "cutoff": CUTOFF,
        "runs": {"xpts_v1": built["xpts_run"]}, "model_versions": {"xpts_v1": XPTS_RUN_VERSION},
    }
    scoreboard = sb.build_scoreboard(conn, artifact=artifact, events=[5, 6], top_k=TOP_K)
    assert scoreboard["status"] == sb.STATUS_PARTIALLY_EVALUATABLE
    assert scoreboard["sample"]["target_events"] == 2
    assert scoreboard["sample"]["target_events_with_observations"] == 1
    # The evaluable event is still measured, and only that event contributes rows.
    events = {row["event"] for row in scoreboard["per_event"]}
    assert events == {5}
    assert any("only some target events" in limitation.lower() for limitation in scoreboard["limitations"])
    conn.close()


# ---------------------------------------------------------------------------
# Live smoke: the canonical GW5 anchor is unplayed and must stay unscored
# ---------------------------------------------------------------------------

LIVE_ARTIFACT = Path("K:/FPL/data/exports/four_gw/gw05/certification_artifact.json")
LIVE_DB = Path("K:/FPL/fpl.db")


@pytest.mark.skipif(
    not (LIVE_ARTIFACT.exists() and LIVE_DB.exists()),
    reason="the canonical anchor or the live database is not present",
)
def test_the_canonical_gw5_scoreboard_is_not_yet_evaluatable_and_fabricates_nothing():
    """Read-only.  No artifact is written and the live database is never modified."""

    from fpl_brain import four_gw_decision as fg

    conn = sqlite3.connect(f"file:{LIVE_DB}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        artifact = fg.load_certification_artifact(LIVE_ARTIFACT)
        assert artifact["four_gw_certification_identity"] == (
            "sha256:d39d02d3f244b35ea6e80397e4f1037572f35e733d761cb5cbb10d5358af6aae"
        )
        scoreboard = sb.build_scoreboard(conn, artifact=artifact, events=[5])

        assert scoreboard["status"] == sb.STATUS_NOT_YET_EVALUATABLE
        assert scoreboard["arms"] == []
        assert scoreboard["per_event"] == []
        assert scoreboard["sample"]["player_event_observations"] == 0
        assert scoreboard["population"]["shared_comparison_n"] == 0
        assert scoreboard["population"]["evaluated"] == 0
        assert scoreboard["population"]["candidates"] == 659
        assert scoreboard["population"]["excluded_by_status"] == {wf.OUTCOME_NOT_FINALISED: 659}

        text = sb.canonical_bytes(scoreboard).decode("utf-8")
        assert '"mae"' not in text and '"rmse"' not in text and '"bias"' not in text
        assert "NaN" not in text and "Infinity" not in text

        # Nothing was scored, so nothing can have been fabricated.
        assert conn.execute("SELECT COUNT(*) FROM outcome_observations").fetchone()[0] == 0
        assert conn.execute("SELECT finished FROM events WHERE id=5").fetchone()[0] == 0
    finally:
        conn.close()
