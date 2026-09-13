"""Minutes v1.1.0 acceptance: previous-season prior, availability provenance,
historical-availability semantics, and completed-run immutability."""

from __future__ import annotations

import pytest

from fpl_brain import minutes_model
from fpl_brain.database import connect_database
from fpl_brain.models import PlayerSeasonHistoryRecord
from fpl_brain.parsers import parse_element_history_past

from test_minutes_model import analytics_repo_upserts, CUTOFF, EVENT_DEADLINE


def _prior_world(conn, *, season_rows=None, minutes_seed=None, snapshots_seed=None):
    seed = {
        "players": {
            30: (1, 3, "a"),  # established primary (strong prev season)
            31: (1, 3, "a"),  # sub-heavy rotation profile
            32: (1, 3, "a"),  # tiny prior exposure
            33: (1, 3, "a"),  # established + role-weakened note
            34: (1, 4, "u"),  # soft unavailable signal
            35: (1, 2, "a"),  # current-season injury absence scenario
            36: (1, 2, "s"),  # suspended: the only genuine hard-out
        },
        "snapshots": {**{pid: ("a", None) for pid in (30, 31, 32, 33, 35)}, 34: ("u", None), 36: ("s", 0)},
        "pgw": minutes_seed or {},  # current-season completed rows
        "scout_notes": {33: ("role_security_5gw", "very_low")},
    }
    rows = [
        PlayerSeasonHistoryRecord(player_id=player_id, season_name="2025/26", **fields, raw_json={})
        for player_id, fields in (season_rows or {}).items()
    ]
    with conn:
        analytics_repo_upserts(conn, seed)
        if rows:
            from fpl_brain import repositories as repo

            repo.upsert_player_season_histories(conn, rows, observed_at="2026-09-10T08:00:00Z")
    return seed


def _freeze_rows(conn, config, player_ids=(30, 31, 32, 33, 34, 35)):
    rows = minutes_model.build_minutes_predictions(conn, 4, CUTOFF, config)
    return {(row["player_id"], row["fixture_id"]): row for row in rows if row["player_id"] in player_ids}


def test_established_primary_stays_above_generic_prior(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    _prior_world(conn, season_rows={30: {"minutes": 2880, "starts": 32}})
    rows = _freeze_rows(conn, minutes_model.MinutesModelConfig())
    primary = rows[(30, 4)]
    prior = primary["start_evidence"]["prior_start_rate"]
    prev = primary["prev_season_prior"]
    assert prev["effective_sample_size"] == pytest.approx(12.0)
    assert prev["start_rate"] == pytest.approx(1.0)
    # The prev-season prior lifts the generic anchor, not just the output.
    assert prior > 0.6 and prev["start_rate"] == 1.0
    assert primary["p_start"] > 0.5  # meaningfully above the generic position prior (~0.3-0.4)
    conn.close()


def test_rotation_profile_stays_below_nailed_prior(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    _prior_world(conn, season_rows={30: {"minutes": 2880, "starts": 32}, 31: {"minutes": 2400, "starts": 19}})
    rows = _freeze_rows(conn, minutes_model.MinutesModelConfig())
    primary, rotation = rows[(30, 4)], rows[(31, 4)]
    assert rotation["prev_season_prior"]["start_rate"] == pytest.approx(0.5, abs=0.02)
    assert primary["start_evidence"]["prior_start_rate"] > rotation["start_evidence"]["prior_start_rate"]
    assert primary["p_start"] > rotation["p_start"]
    conn.close()


def test_small_current_evidence_gradually_outweighs_prior(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    _prior_world(conn, season_rows={30: {"minutes": 2000, "starts": 10}})
    config = minutes_model.MinutesModelConfig()
    no_current = _freeze_rows(conn, config)
    with conn:
        from fpl_brain import repositories as repo
        from fpl_brain.models import PlayerGameweekRecord as Pgw

        # Strong new-season evidence: five recent completed starts.
        repo.upsert_player_gameweeks(
            conn,
            [
                Pgw(player_id=30, event=event, fixture_id=fixture, minutes=90, starts=1, total_points=3, source="element_summary", raw_json={})
                for event, fixture in ((1, 1), (2, 2), (3, 3), (2, 2), (3, 3))
            ],
        )
    with_current = _freeze_rows(conn, config)
    baseline_p = no_current[(30, 4)]["p_start_given_available"]
    raised_p = with_current[(30, 4)]["p_start_given_available"]
    assert raised_p > baseline_p
    # Five strong observed starts move the posterior well beyond the prior.
    assert raised_p - baseline_p >= 0.05
    conn.close()


def test_tiny_prev_season_sample_is_heavily_shrunk(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    bare = connect_database(tmp_path / "bare.db")
    _prior_world(conn, season_rows={32: {"minutes": 85, "starts": 1}})
    _prior_world(bare, season_rows={})
    config = minutes_model.MinutesModelConfig()
    tiny = _freeze_rows(conn, config)
    none = _freeze_rows(bare, config)
    tiny_prior = tiny[(32, 4)]["start_evidence"]["prior_start_rate"]
    generic_prior = none[(32, 4)]["start_evidence"]["prior_start_rate"]
    assert tiny[(32, 4)]["prev_season_prior"]["effective_sample_size"] < 1.0
    assert "thin_prev_exposure" in " ".join(tiny[(32, 4)]["prev_season_prior"]["ess_discounts"])
    # One appearance last season barely moves the generic prior.
    assert abs(tiny_prior - generic_prior) < 0.05
    bare.close()
    conn.close()


def test_role_change_weakens_ess_never_confers_start_probability(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    _prior_world(conn, season_rows={33: {"minutes": 2880, "starts": 32}})
    config = minutes_model.MinutesModelConfig()
    rows = _freeze_rows(conn, config)
    prev = rows[(33, 4)]["prev_season_prior"]
    assert prev["effective_sample_size"] == pytest.approx(12.0 * config.prev_season_role_change_ess_discount)
    assert "role_change_prev_ess" in prev["ess_discounts"]
    # The weaker prior cannot push the start probability above the unweakened case.
    conn2 = connect_database(tmp_path / "conn2.db")
    _prior_world(conn2, season_rows={33: {"minutes": 2880, "starts": 32}, 30: {"minutes": 2880, "starts": 32}, 31: {"minutes": 2400, "starts": 19}, 32: {"minutes": 85, "starts": 1}, 35: {"minutes": 1710, "starts": 19}})
    rows_strong = _freeze_rows(conn2, config)
    assert rows_strong[(33, 4)]["start_evidence"]["prior_start_rate"] < rows_strong[(30, 4)]["start_evidence"]["prior_start_rate"]
    conn2.close()
    conn.close()


def test_availability_signal_defaults_are_config_resolved(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    _prior_world(conn, season_rows={})
    default_config = minutes_model.MinutesModelConfig()
    custom = minutes_model.MinutesModelConfig(
        official_availability_signal_defaults={"a": 1.0, "u": 0.35},
    )
    assert default_config.config_hash() != custom.config_hash()
    rows_default = _freeze_rows(conn, default_config)
    rows_custom = _freeze_rows(conn, custom)
    # status 'u' player 34: default signal 0.2 vs custom 0.35 — modelled multiplier
    # comes from the (hashed) config, not hidden module magic.
    assert rows_default[(34, 4)]["p_available"] == pytest.approx(0.2)
    assert rows_custom[(34, 4)]["p_available"] == pytest.approx(0.35)
    assert "initial modelling assumption" in rows_custom[(34, 4)]["availability_source_summary"]["rule"]
    conn.close()


def test_only_genuine_hard_out_forces_zero(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    _prior_world(conn, season_rows={})
    config = minutes_model.MinutesModelConfig()
    hard_rows = {row["player_id"]: row for row in minutes_model.build_minutes_predictions(conn, 4, CUTOFF, config)}
    # Player 36 is status 's': the ONLY genuine hard-out (config, not hidden magic).
    suspended = hard_rows[36]
    assert suspended["p_available"] == 0.0
    assert suspended["expected_minutes"] == 0.0
    # A soft 'u' status player stays a signal multiplier, never a hard zero.
    assert 0.0 < hard_rows[34]["p_available"] < 1.0
    conn.close()


def test_season_history_ingestion_roundtrip(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    _prior_world(conn, season_rows={})
    with conn:
        from fpl_brain import repositories as repo

        payload = {
            "history_past": [
                {"season_name": "2025/26", "starts": 32, "minutes": 2835, "total_points": 120},
                {"season_name": "2024/25", "starts": 37, "minutes": 3195, "total_points": 111},
            ]
        }
        records = parse_element_history_past(payload, 30)
        assert len(records) == 2
        repo.upsert_player_season_histories(conn, records, observed_at="2026-09-10T08:00:00Z")
    rows = repo.player_season_histories(conn, 30)
    assert [r["season_name"] for r in rows] == ["2025/26", "2024/25"]
    assert rows[0]["starts"] == 32 and rows[0]["minutes"] == 2835
    conn.close()


def test_injury_absence_does_not_destroy_tactical_prior(tmp_path):
    """Historical availability: a *proven* injury absence must not drag the
    player's start rate toward a false tactical non-selection record.

    R2B correction: this test previously seeded player 35 as status 'a' while
    asserting that his 0-minute GW3 row had no effect.  That encoded the R1
    defect — an *available* non-selection is negative role evidence.  The
    fixture now carries an event-attributed status trail (available GW1-2,
    injured GW3) so the name, the docstring and the data agree.  The companion
    case — an available 0-minute row DOES lower p_start — is covered by
    ``test_available_non_start_evidence_lowers_p_start_materially`` in
    ``test_minutes_role_evidence.py``.
    """

    conn = connect_database(tmp_path / "fpl.db")
    _prior_world(conn, season_rows={})
    config = minutes_model.MinutesModelConfig()
    with conn:
        from fpl_brain import repositories as repo
        from fpl_brain.models import PlayerGameweekRecord as Pgw
        from fpl_brain.models import PlayerSnapshotRecord as Snap

        # Event-attributed availability trail for player 35: available for GW1
        # and GW2, injured for GW3.  player_snapshots is keyed on
        # (player_id, fetch_run_id), so each capture needs its own run.
        run = repo.create_fetch_run(conn, "fetch_fpl")
        for event, captured_at, status in (
            (1, "2026-08-20T08:00:00Z", "a"),
            (2, "2026-08-27T08:00:00Z", "a"),
            (3, "2026-09-03T08:00:00Z", "i"),
        ):
            snapshot_run = repo.create_fetch_run(conn, "fetch_fpl")
            repo.insert_snapshots(
                conn,
                [
                    Snap(
                        player_id=35,
                        captured_at=captured_at,
                        event_context=event,
                        now_cost=50,
                        status=status,
                        raw_json={},
                    )
                ],
                snapshot_run,
            )
        # Stable pooled background so the positional pool is not one-row fragile.
        repo.upsert_players(
            conn,
            [
                PlayerRecordFactory(pid)
                for pid in (40, 41, 42, 43)
            ],
        )
        repo.insert_snapshots(
            conn,
            [PlayerSnapshotFactory(pid) for pid in (40, 41, 42, 43)],
            run,
        )
        background = []
        for player_id in (40, 41, 42, 43):
            for event, fixture, minutes, starts in ((1, 1, 90, 1), (2, 2, 0, 0), (3, 3, 0, 0)):
                background.append(Pgw(player_id=player_id, event=event, fixture_id=fixture, minutes=minutes, starts=starts, total_points=2 if starts else 0, source="element_summary", raw_json={}))
        repo.upsert_player_gameweeks(conn, background)
        # Player 35 started GW1 and GW2, then sat out GW3 (observed, minutes 0).
        repo.upsert_player_gameweeks(
            conn,
            [
                Pgw(player_id=35, event=1, fixture_id=1, minutes=90, starts=1, total_points=3, source="element_summary", raw_json={}),
                Pgw(player_id=35, event=2, fixture_id=2, minutes=90, starts=1, total_points=3, source="element_summary", raw_json={}),
                Pgw(player_id=35, event=3, fixture_id=3, minutes=0, starts=0, total_points=0, source="element_summary", raw_json={}),
            ],
        )
    with_absence = minutes_model.build_minutes_predictions(conn, 4, CUTOFF, config)
    pick = lambda rows: [r for r in rows if r["player_id"] == 35][0]  # noqa: E731
    with_row = pick(with_absence)["p_start_given_available"]
    evidence = pick(with_absence)["start_evidence"]
    classes = pick(with_absence)["evidence_classes"]
    assert evidence["observed_rows"] == 3 and evidence["known_available_rows"] == 2
    # The GW3 row is attributed to injury, so it is not role evidence at all.
    assert classes[minutes_model.EVIDENCE_UNAVAILABLE] == 1
    assert classes[minutes_model.EVIDENCE_AVAILABLE_NON_START_ZERO_MINUTES] == 0
    assert evidence["available_observation_rows"] == 2
    with conn:
        conn.execute("DELETE FROM player_gameweeks WHERE player_id=35 AND event=3")
    without_row = pick(minutes_model.build_minutes_predictions(conn, 4, CUTOFF, config))["p_start_given_available"]
    # Only a small positional-pool effect may remain (13-row synthetic pool);
    # the player's OWN evidence rate is the clean 2-of-2 started record either
    # way — the injury absence never became non-start evidence for the player.
    assert abs(with_row - without_row) < 0.03
    assert with_row > pick(with_absence)["start_evidence"]["prior_start_rate"]
    conn.close()


def PlayerRecordFactory(pid):  # noqa: N802 (local factory)
    from fpl_brain.models import PlayerRecord

    return PlayerRecord(id=pid, web_name=f"P{pid}", full_name=f"Player {pid}", team_id=1, element_type=2)


def PlayerSnapshotFactory(pid):  # noqa: N802 (local factory)
    from fpl_brain.models import PlayerSnapshotRecord

    return PlayerSnapshotRecord(player_id=pid, captured_at="2026-09-10T08:00:00Z", now_cost=50, status="a", raw_json={})


def test_completed_projection_runs_immutable_but_lifecycle_allowed(tmp_path):
    conn = connect_database(tmp_path / "fpl.db")
    _prior_world(conn, season_rows={})
    from fpl_brain import analytics

    run_id = analytics.create_projection_run(
        conn,
        model_family=analytics.MINUTES_MODEL_FAMILY,
        model_version=minutes_model.MINUTES_MODEL_VERSION,
        planning_event=4,
        planning_context_hash="ref",
        data_cutoff=CUTOFF,
        scouting_cutoff=None,
        official_run_ids={},
        deadline_status="PRE_DEADLINE",
    )
    # Lifecycle transition allowed while running.
    analytics.finish_projection_run(conn, run_id, "complete")
    for sql in (
        "UPDATE projection_runs SET model_family='baseline' WHERE id=?",
        "UPDATE projection_runs SET model_version='minutes_v9.9.9' WHERE id=?",
        "UPDATE projection_runs SET config_hash='x' WHERE id=?",
        "UPDATE projection_runs SET deadline_status='LATE_FREEZE' WHERE id=?",
        "UPDATE projection_runs SET status='running' WHERE id=?",
    ):
        with pytest.raises(Exception, match="completed projection runs are immutable"):
            conn.execute(sql, (run_id,))
    with pytest.raises(Exception, match="projection runs are provenance and cannot be deleted"):
        conn.execute("DELETE FROM projection_runs WHERE id=?", (run_id,))
    conn.close()
