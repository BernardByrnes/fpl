"""Deterministic walk-forward population core: eligibility, gate, digest, identity.

Every behavioural test is synthetic and hermetic.  The single live-database test
is the read-only canonical-anchor smoke, which must classify an unplayed event
as NOT YET EVALUATABLE rather than inventing a zero.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from fpl_brain import analytics, walk_forward as wf
from fpl_brain.database import connect_database

CUTOFF = "2026-09-10T12:00:00Z"
CODE_SNAPSHOT = "sha256:" + "a" * 64

#: Players and the fixtures they are projected for, by event.  Team 3 never has a
#: fixture, so player 13 is the blank-Gameweek case.
PLAYERS = {10: 1, 11: 1, 12: 2, 13: 3, 14: 1}
XS_RUN = 900
BASELINE_RUN = 901


def _world(
    conn: sqlite3.Connection,
    *,
    xpts_by_player_fixture: dict[tuple[int, int], float] | None = None,
    xpts_run_version: str = "xpts_v1.4.1",
    xpts_run_snapshot: str | None = CODE_SNAPSHOT,
    xpts_run_cutoff: str = CUTOFF,
    baseline_values: dict[tuple[int, int], float] | None = None,
    baseline_snapshot: str | None = CODE_SNAPSHOT,
    omit_model_pairs: set[tuple[int, int]] | None = None,
):
    """A tiny league with one synthetic double gameweek, one single, one unplayed.

    Event 5 is a DOUBLE gameweek for teams 1 and 2 (fixtures 100 and 101).
    Event 6 is a single gameweek (fixture 102).  Event 7 is scheduled, not
    played (fixture 103).  Team 3 has no fixture in any event.
    """

    from fpl_brain import repositories as repo
    from fpl_brain.models import (
        EventRecord,
        FixtureRecord,
        PlayerRecord,
        PositionRecord,
        TeamRecord,
    )

    omit_model_pairs = omit_model_pairs or set()
    xpts_by_player_fixture = xpts_by_player_fixture or {}

    with conn:
        repo.upsert_teams(conn, [TeamRecord(id=t, name=f"Team {t}") for t in (1, 2, 3)])
        repo.upsert_positions(conn, [PositionRecord(id=p, singular_name_short=n)
                                     for p, n in ((1, "GKP"), (2, "DEF"), (3, "MID"), (4, "FWD"))])
        repo.upsert_players(conn, [PlayerRecord(id=pid, web_name=f"P{pid}", full_name=f"Player {pid}",
                                                team_id=team, element_type=3)
                                   for pid, team in PLAYERS.items()])
        repo.upsert_events(conn, [
            EventRecord(id=5, finished=1, data_checked=1, deadline_time="2026-09-05T17:30:00Z", raw_json={}),
            EventRecord(id=6, finished=1, data_checked=1, deadline_time="2026-09-12T17:30:00Z", raw_json={}),
            EventRecord(id=7, finished=0, data_checked=0, deadline_time="2026-09-19T17:30:00Z", raw_json={}),
        ])
        repo.upsert_fixtures(conn, [
            FixtureRecord(id=100, event=5, team_h=1, team_a=2, finished=1, started=1,
                          kickoff_time="2026-09-04T14:00:00Z", raw_json={}),
            FixtureRecord(id=101, event=5, team_h=2, team_a=1, finished=1, started=1,
                          kickoff_time="2026-09-06T14:00:00Z", raw_json={}),
            FixtureRecord(id=102, event=6, team_h=1, team_a=2, finished=1, started=1,
                          kickoff_time="2026-09-11T14:00:00Z", raw_json={}),
            FixtureRecord(id=103, event=7, team_h=1, team_a=2, finished=0, started=0,
                          kickoff_time="2026-09-19T14:00:00Z", raw_json={}),
        ])
        # Outcomes for the two finalised events.
        conn.execute(
            "INSERT INTO player_gameweeks(player_id, event, fixture_id, minutes, starts, total_points,"
            " expected_goals, source, updated_at, raw_json) VALUES "
            "(10,5,100,90,1,6,0.4,'element_summary','2026-09-05T09:00:00Z','{}'),"
            "(10,5,101,88,1,5,0.3,'element_summary','2026-09-07T09:00:00Z','{}'),"
            # Canonical placeholder: minutes=0 and EVERY performance column NULL.
            # starts=0/total_points=0 would be genuine evidence, not a placeholder.
            "(11,5,100,0,NULL,NULL,NULL,'element_summary','2026-09-05T08:00:00Z','{}'),"
            "(12,5,100,90,1,2,0.1,'element_summary','2026-09-05T09:00:00Z','{}'),"
            "(12,5,101,90,1,3,0.2,'element_summary','2026-09-07T09:00:00Z','{}'),"
            "(14,5,100,0,0,0,0.0,'element_summary','2026-09-05T09:00:00Z','{}'),"    # genuine DNP
            "(14,5,101,0,0,0,0.0,'element_summary','2026-09-07T09:00:00Z','{}'),"
            "(10,6,102,90,1,7,0.5,'element_summary','2026-09-12T09:00:00Z','{}'),"
            "(12,6,102,90,1,4,0.1,'element_summary','2026-09-12T09:00:00Z','{}')"
        )
        conn.execute(
            "INSERT INTO player_gameweeks(player_id, event, fixture_id, minutes, starts, total_points,"
            " source, updated_at, raw_json) VALUES "
            "(10,7,103,0,0,0,'element_summary','2026-09-19T09:00:00Z','{}')"
        )

    xpts_run = analytics.create_projection_run(
        conn, model_family="xpts_v1", model_version=xpts_run_version, planning_event=5,
        planning_context_hash="ctx", data_cutoff=xpts_run_cutoff, scouting_cutoff=None,
        official_run_ids={}, source_snapshot_sha256=xpts_run_snapshot,
    )
    baseline_run = analytics.create_projection_run(
        conn, model_family="baseline", model_version=analytics.BASELINE_MODEL_VERSION, planning_event=5,
        planning_context_hash="ctx", data_cutoff=CUTOFF, scouting_cutoff=None,
        official_run_ids={}, source_snapshot_sha256=baseline_snapshot,
    )
    with conn:
        for (player_id, fixture_id), value in sorted(xpts_by_player_fixture.items()):
            if (player_id, fixture_id) in omit_model_pairs:
                continue
            fixture_event = conn.execute(
                "SELECT event, team_h, team_a FROM fixtures WHERE id=?", (fixture_id,)).fetchone()
            event, team_h, team_a = int(fixture_event[0]), int(fixture_event[1]), int(fixture_event[2])
            team_id = PLAYERS[player_id]
            opponent = team_a if team_id == team_h else team_h
            conn.execute(
                "INSERT INTO player_fixture_xpts_projections(projection_run_id, player_id, fixture_id,"
                " event, team_id, opponent_id, position, minutes_run_id, team_run_id, rate_run_id,"
                " payload_json, model_version, scoring_rules_version, generated_at)"
                " VALUES (?,?,?,?,?,?,'MID',1,1,1,?,?, 'fpl_scoring_2026_27_v1.0.0','2026-09-10T12:00:00Z')",
                (xpts_run, int(player_id), int(fixture_id), event, int(team_id), int(opponent),
                 json.dumps({"total_xpts": float(value)}), xpts_run_version),
            )
        for (event, player_id), value in sorted((baseline_values or {}).items()):
            conn.execute(
                "INSERT INTO frozen_predictions(projection_run_id, kind, player_id, fixture_id, event,"
                " payload_json, model_version, generated_at) VALUES (?,?,?,NULL,?,?,?,'2026-09-10T12:00:00Z')",
                (baseline_run, analytics.RECENT_POINTS_KIND, int(player_id), int(event),
                 json.dumps({"value": float(value)}), analytics.BASELINE_MODEL_VERSION),
            )
    with conn:
        for run_id in (xpts_run, baseline_run):
            conn.execute("UPDATE projection_runs SET status='complete' WHERE id=?", (int(run_id),))
    return {
        "xpts_run": xpts_run,
        "baseline_run": baseline_run,
        "artifact": {
            "four_gw_certification_identity": "sha256:" + "b" * 64,
            "planning_cutoff": CUTOFF,
            "certified_bundles": {
                "5": {"event": 5, "cutoff": CUTOFF, "runs": {"xpts_v1": xpts_run},
                      "model_versions": {"xpts_v1": xpts_run_version}, "code_snapshot_sha256": CODE_SNAPSHOT},
                "6": {"event": 6, "cutoff": CUTOFF, "runs": {"xpts_v1": xpts_run},
                      "model_versions": {"xpts_v1": xpts_run_version}, "code_snapshot_sha256": CODE_SNAPSHOT},
                "7": {"event": 7, "cutoff": CUTOFF, "runs": {"xpts_v1": xpts_run},
                      "model_versions": {"xpts_v1": xpts_run_version}, "code_snapshot_sha256": CODE_SNAPSHOT},
            },
        },
    }


def _dense_xpts() -> dict[tuple[int, int], float]:
    """Every pool player projected on both of their event-5 fixtures (or none)."""

    return {
        (10, 100): 3.0, (10, 101): 4.5,
        (11, 100): 1.0, (11, 101): 1.0,
        (12, 100): 2.0, (12, 101): 2.5,
        (14, 100): 0.5, (14, 101): 0.5,
    }


def _population(conn, world, *, events=(5,), baseline_kind=None, **kwargs):
    anchor = wf.discover_certified_anchor(conn, world["artifact"], events=list(events))
    return wf.build_event_population(conn, anchor=anchor, events=list(events),
                                     baseline_kind=baseline_kind, **kwargs)


def _by_player(population):
    return {(int(r["event"]), int(r["player_id"])): r for r in population.rows}


def _status_by_player(population):
    return {(int(r["event"]), int(r["player_id"])): r["status"] for r in population.excluded}


# ---------------------------------------------------------------------------
# A / B / P — aggregation
# ---------------------------------------------------------------------------


def test_B_a_double_gameweek_sums_to_one_event_prediction(tmp_path):
    """Two fixture projections in one event become ONE summed event prediction."""

    conn = connect_database(tmp_path / "wf.db")
    world = _world(conn, xpts_by_player_fixture=_dense_xpts())
    population = _population(conn, world)

    row = _by_player(population)[(5, 10)]
    assert row["model_xpts"] == pytest.approx(7.5), "3.0 + 4.5"
    assert row["model_fixtures"] == [100, 101], "both contributing fixtures are recorded"
    assert population.eligible_keys.count((5, 10)) == 1, "exactly one event row for the player"
    conn.close()


def test_A_a_single_gameweek_is_one_fixture(tmp_path):
    conn = connect_database(tmp_path / "wf.db")
    world = _world(conn, xpts_by_player_fixture={**_dense_xpts(), (10, 102): 7.0})
    population = _population(conn, world, events=(6,))
    row = _by_player(population)[(6, 10)]
    assert row["model_xpts"] == pytest.approx(7.0)
    assert row["model_fixtures"] == [102]
    conn.close()


def test_P_the_event_baseline_is_not_duplicated_across_a_double_gameweek(tmp_path):
    """One event-grain baseline value stays one value on a DGW.

    The stored baseline is 4.0 for player 10 in event 5.  If anyone expanded it
    per fixture the row would carry 8.0, or two rows would appear.
    """

    conn = connect_database(tmp_path / "wf.db")
    world = _world(conn, xpts_by_player_fixture=_dense_xpts(),
                   baseline_values={(5, 10): 4.0, (5, 11): 1.0, (5, 12): 2.0, (5, 14): 0.5})
    population = _population(conn, world, baseline_kind=analytics.RECENT_POINTS_KIND)
    rows = [r for r in population.rows if int(r["player_id"]) == 10]
    assert len(rows) == 1, "a DGW must not produce one evaluated row per fixture"
    assert rows[0]["baseline_value"] == pytest.approx(4.0), "the baseline was duplicated across fixtures"
    assert rows[0]["model_xpts"] == pytest.approx(7.5), "the model side legitimately sums"
    conn.close()


# ---------------------------------------------------------------------------
# C / D / E / F / G — statuses
# ---------------------------------------------------------------------------


def test_C_a_blank_gameweek_is_classified_not_scored(tmp_path):
    conn = connect_database(tmp_path / "wf.db")
    world = _world(conn, xpts_by_player_fixture=_dense_xpts())
    population = _population(conn, world)
    assert _status_by_player(population)[(5, 13)] == wf.TARGET_NO_FIXTURE
    assert (5, 13) not in _by_player(population)
    conn.close()


def test_D_a_scheduled_placeholder_outcome_is_excluded_not_zeroed(tmp_path):
    conn = connect_database(tmp_path / "wf.db")
    world = _world(conn, xpts_by_player_fixture=_dense_xpts())
    population = _population(conn, world)
    assert _status_by_player(population)[(5, 11)] == wf.OUTCOME_PLACEHOLDER_EXCLUDED
    assert (5, 11) not in _by_player(population), "a placeholder must not be scored as 0 points"
    conn.close()


def test_E_a_genuine_zero_minute_dnp_is_eligible(tmp_path):
    conn = connect_database(tmp_path / "wf.db")
    world = _world(conn, xpts_by_player_fixture=_dense_xpts())
    population = _population(conn, world)
    row = _by_player(population)[(5, 14)]
    assert row["outcome"]["minutes"] == 0
    assert row["outcome"]["total_points"] == 0
    assert row["model_xpts"] == pytest.approx(1.0), "0.5 + 0.5"
    conn.close()


def test_F_a_non_final_event_is_not_evaluatable(tmp_path):
    conn = connect_database(tmp_path / "wf.db")
    world = _world(conn, xpts_by_player_fixture=_dense_xpts())
    population = _population(conn, world, events=(7,))
    assert population.rows == []
    assert set(population.status_counts) == {wf.OUTCOME_NOT_FINALISED}
    assert population.identity.as_dict()["outcome_state"] == {7: "SCHEDULED"}
    conn.close()


def test_G_a_missing_model_projection_is_not_zero_points(tmp_path):
    conn = connect_database(tmp_path / "wf.db")
    world = _world(conn, xpts_by_player_fixture=_dense_xpts(),
                   omit_model_pairs={(12, 100), (12, 101)})   # the whole event, not one fixture
    population = _population(conn, world)
    statuses = _status_by_player(population)
    assert statuses[(5, 12)] == wf.MODEL_PROJECTION_MISSING
    assert (5, 12) not in _by_player(population), "missing must not become a scored zero"
    conn.close()


# ---------------------------------------------------------------------------
# H / I — the same-population gate
# ---------------------------------------------------------------------------


def test_H_a_missing_baseline_value_never_becomes_zero(tmp_path):
    conn = connect_database(tmp_path / "wf.db")
    # Player 14 is ELIGIBLE but the baseline run has no value for them.  The
    # arms would be compared on different rows, so this must fail closed rather
    # than treating the absent baseline as zero.
    world = _world(conn, xpts_by_player_fixture=_dense_xpts(),
                   baseline_values={(5, 10): 4.0, (5, 12): 2.0})
    with pytest.raises(wf.PopulationMismatch) as failure:
        _population(conn, world, baseline_kind=analytics.RECENT_POINTS_KIND)
    assert failure.value.model_only == ((5, 14),), "the model-only key must be named"
    assert not failure.value.baseline_only
    conn.close()


def test_I_a_key_mismatch_fails_closed_and_names_both_sides(tmp_path):
    with pytest.raises(wf.PopulationMismatch) as failure:
        wf.assert_same_population([(5, 10), (5, 12)], [(5, 10), (5, 14)])
    assert failure.value.model_only == ((5, 12),)
    assert failure.value.baseline_only == ((5, 14),)
    # Identical key sets, in different orders, are the same population.
    wf.assert_same_population([(5, 12), (5, 10)], [(5, 10), (5, 12)])
    conn = None
    assert conn is None


# ---------------------------------------------------------------------------
# J / K / L — the digest
# ---------------------------------------------------------------------------


def test_J_key_order_does_not_change_the_digest():
    keys = [(5, 12), (5, 10), (5, 14)]
    assert wf.canonical_population_digest(keys, grain=wf.GRAIN_PLAYER_EVENT) == \
        wf.canonical_population_digest(reversed(keys), grain=wf.GRAIN_PLAYER_EVENT)


def test_K_changing_one_key_changes_the_digest():
    base = wf.canonical_population_digest([(5, 10), (5, 12)], grain=wf.GRAIN_PLAYER_EVENT)
    changed = wf.canonical_population_digest([(5, 10), (5, 13)], grain=wf.GRAIN_PLAYER_EVENT)
    assert base != changed


def test_L_event_and_fixture_grain_digests_are_not_conflated():
    event_keys = [(5, 10), (5, 12)]
    fixture_keys = [(5, 10), (5, 12)]     # same pairs, different meaning
    assert wf.canonical_population_digest(event_keys, grain=wf.GRAIN_PLAYER_EVENT) != \
        wf.canonical_population_digest(fixture_keys, grain=wf.GRAIN_PLAYER_FIXTURE)
    with pytest.raises(wf.WalkForwardError):
        wf.canonical_population_digest(event_keys, grain="not_a_grain")


def test_the_digest_is_stable_across_repeated_construction(tmp_path):
    conn = connect_database(tmp_path / "wf.db")
    world = _world(conn, xpts_by_player_fixture=_dense_xpts())
    first = _population(conn, world)
    second = _population(conn, world)
    assert first.digest == second.digest
    assert [r["player_id"] for r in first.rows] == [r["player_id"] for r in second.rows]
    conn.close()


# ---------------------------------------------------------------------------
# M / N — semantic eligibility
# ---------------------------------------------------------------------------


def test_M_a_legacy_semantic_run_is_excluded(tmp_path):
    """A pre-repair run cannot enter a corrected scoreboard.

    The anchor carries the current identity; the legacy run is judged against
    it and fails on BOTH its version and its recorded code fingerprint, so a
    compatible-looking model name is not enough.
    """

    conn = connect_database(tmp_path / "wf.db")
    world = _world(conn, xpts_by_player_fixture=_dense_xpts())
    legacy_run = analytics.create_projection_run(
        conn, model_family="xpts_v1", model_version="xpts_v1.0.0", planning_event=5,
        planning_context_hash="ctx", data_cutoff=CUTOFF, scouting_cutoff=None,
        official_run_ids={}, source_snapshot_sha256="sha256:" + "c" * 64,
    )
    anchor = wf.discover_certified_anchor(conn, world["artifact"], events=[5])
    entry = anchor.for_event(5)
    assert entry.code_snapshot_sha256 == CODE_SNAPSHOT

    current, reasons = wf.run_is_current_semantics(
        conn, legacy_run, expected_cutoff=entry.cutoff,
        expected_code_snapshot=entry.code_snapshot_sha256)
    assert current is False
    joined = " | ".join(reasons)
    assert "xpts_v1.0.0" in joined, "the version gate must fire"
    assert "fingerprint" in joined, "the code-identity gate must fire"
    conn.close()


def test_a_run_with_no_code_fingerprint_cannot_be_verified(tmp_path):
    """An unverifiable run is excluded rather than trusted by name."""

    conn = connect_database(tmp_path / "wf.db")
    world = _world(conn, xpts_by_player_fixture=_dense_xpts())
    unverifiable = analytics.create_projection_run(
        conn, model_family="xpts_v1", model_version="xpts_v1.4.1", planning_event=5,
        planning_context_hash="ctx", data_cutoff=CUTOFF, scouting_cutoff=None,
        official_run_ids={}, source_snapshot_sha256=None,
    )
    anchor = wf.discover_certified_anchor(conn, world["artifact"], events=[5])
    entry = anchor.for_event(5)
    current, reasons = wf.run_is_current_semantics(
        conn, unverifiable, expected_cutoff=entry.cutoff,
        expected_code_snapshot=entry.code_snapshot_sha256)
    assert current is False and any("no code fingerprint" in reason for reason in reasons)
    conn.close()


def test_N_a_current_semantic_run_is_accepted(tmp_path):
    conn = connect_database(tmp_path / "wf.db")
    world = _world(conn, xpts_by_player_fixture=_dense_xpts())
    anchor = wf.discover_certified_anchor(conn, world["artifact"], events=[5])
    entry = anchor.for_event(5)
    assert entry.code_snapshot_sha256 == CODE_SNAPSHOT, "the anchor's runs agree on their fingerprint"
    current, reasons = wf.run_is_current_semantics(
        conn, world["xpts_run"], expected_cutoff=entry.cutoff,
        expected_code_snapshot=entry.code_snapshot_sha256)
    assert current is True and reasons == ()
    conn.close()


def test_a_wrong_cutoff_is_not_current_semantics(tmp_path):
    conn = connect_database(tmp_path / "wf.db")
    world = _world(conn, xpts_by_player_fixture=_dense_xpts(), xpts_run_cutoff="2026-09-09T00:00:00Z")
    anchor = wf.discover_certified_anchor(conn, world["artifact"], events=[5])
    entry = anchor.for_event(5)
    current, reasons = wf.run_is_current_semantics(
        conn, world["xpts_run"], expected_cutoff=entry.cutoff,
        expected_code_snapshot=entry.code_snapshot_sha256)
    assert current is False and any("cutoff" in reason for reason in reasons)
    conn.close()


def test_the_anchor_is_discovered_from_the_artifact_not_from_the_newest_run(tmp_path):
    """A newer run outside the bundle must not become the evaluation anchor."""

    conn = connect_database(tmp_path / "wf.db")
    world = _world(conn, xpts_by_player_fixture=_dense_xpts())
    newer = analytics.create_projection_run(
        conn, model_family="xpts_v1", model_version="xpts_v1.4.1", planning_event=5,
        planning_context_hash="ctx", data_cutoff=CUTOFF, scouting_cutoff=None,
        official_run_ids={}, source_snapshot_sha256=CODE_SNAPSHOT,
    )
    assert newer > world["xpts_run"]
    anchor = wf.discover_certified_anchor(conn, world["artifact"], events=[5])
    assert anchor.for_event(5).run_id("xpts_v1") == world["xpts_run"], "the newest run was chosen"
    conn.close()


# ---------------------------------------------------------------------------
# Baseline arm classification
# ---------------------------------------------------------------------------


def test_only_points_baselines_are_eligible_for_the_headline_score():
    assert analytics.EP_NEXT_KIND in wf.HEADLINE_POINTS_BASELINE_KINDS
    assert analytics.RECENT_POINTS_KIND in wf.HEADLINE_POINTS_BASELINE_KINDS
    assert analytics.NAIVE_P90_KIND in wf.HEADLINE_POINTS_BASELINE_KINDS
    assert analytics.NAIVE_MINUTES_KIND in wf.NON_POINTS_BASELINE_KINDS
    assert analytics.NAIVE_MINUTES_KIND not in wf.HEADLINE_POINTS_BASELINE_KINDS, (
        "a minutes baseline must not be scored as if minutes were FPL points"
    )


def test_coverage_is_reported_not_implied(tmp_path):
    conn = connect_database(tmp_path / "wf.db")
    world = _world(conn, xpts_by_player_fixture=_dense_xpts(),
                   omit_model_pairs={(12, 100), (12, 101)})
    population = _population(conn, world)
    coverage = population.coverage()
    assert coverage["candidates"] == len(population.rows) + len(population.excluded)
    assert coverage["excluded"] >= 2
    assert coverage["by_status"][wf.MODEL_PROJECTION_MISSING] == 1
    assert 0.0 < coverage["evaluated_share"] < 1.0
    conn.close()


def test_the_identity_carries_everything_needed_to_reproduce_the_population(tmp_path):
    conn = connect_database(tmp_path / "wf.db")
    world = _world(conn, xpts_by_player_fixture=_dense_xpts(),
                   baseline_values={(5, 10): 4.0, (5, 11): 1.0, (5, 12): 2.0, (5, 14): 0.5})
    population = _population(conn, world, baseline_kind=analytics.RECENT_POINTS_KIND)
    identity = population.identity.as_dict()
    assert identity["evaluation_version"] == wf.WALK_FORWARD_VERSION
    assert identity["grain"] == wf.GRAIN_PLAYER_EVENT
    assert identity["target_events"] == [5]
    assert identity["planning_cutoff"] == CUTOFF
    assert identity["per_event_runs"]["5"]["xpts_v1"] == world["xpts_run"]
    assert identity["per_event_runs"]["5"][wf.BASELINE_FAMILY] == world["baseline_run"]
    # Run identity is recorded PER EVENT: one family can name a different run in
    # every target event, so a family-keyed map would keep only the last one.
    assert population.identity.run_id_for(5, "xpts_v1") == world["xpts_run"]
    assert identity["eligible_population_digest"] == population.digest
    assert identity["code_snapshot_sha256"] == CODE_SNAPSHOT
    assert identity["missing_data_policy_version"] == wf.MISSING_DATA_POLICY_VERSION
    assert identity["outcome_state"] == {5: "FINAL"}
    conn.close()


# ---------------------------------------------------------------------------
# O — the live canonical anchor is not yet evaluatable
# ---------------------------------------------------------------------------

LIVE_ARTIFACT = Path("K:/FPL/data/exports/four_gw/gw05/certification_artifact.json")
LIVE_DB = Path("K:/FPL/fpl.db")


@pytest.mark.skipif(not (LIVE_ARTIFACT.exists() and LIVE_DB.exists()),
                    reason="the canonical anchor or the live database is not present")
def test_O_the_canonical_gw5_anchor_is_not_yet_evaluatable():
    """Read-only.  An unplayed event must be classified, never scored as zero."""

    from fpl_brain import four_gw_decision as fg

    conn = sqlite3.connect(f"file:{LIVE_DB}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        artifact = fg.load_certification_artifact(LIVE_ARTIFACT)
        anchor = wf.discover_certified_anchor(conn, artifact, events=[5])
        entry = anchor.for_event(5)
        assert artifact["four_gw_certification_identity"] == \
            "sha256:d39d02d3f244b35ea6e80397e4f1037572f35e733d761cb5cbb10d5358af6aae"
        assert entry.cutoff == "2026-09-17T13:46:43Z"
        current, reasons = wf.run_is_current_semantics(
            conn, entry.run_id("xpts_v1"), expected_cutoff=entry.cutoff,
            expected_code_snapshot=entry.code_snapshot_sha256)
        assert current is True, reasons

        population = wf.build_event_population(conn, anchor=anchor, events=[5], baseline_kind=None)
        assert population.rows == [], "an unplayed event produced scored rows"
        assert set(population.status_counts) == {wf.OUTCOME_NOT_FINALISED}
        assert population.status_counts[wf.OUTCOME_NOT_FINALISED] == 659
        assert conn.execute("SELECT COUNT(*) FROM outcome_observations").fetchone()[0] == 0
    finally:
        conn.close()
