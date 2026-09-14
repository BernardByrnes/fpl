"""Certified historical-input completeness repair — targeted tests.

Covers the R5 post-decision defect: an officially completed Gameweek whose
event-specific player observations were never refreshed, so the models read
stale schedule placeholders while the official aggregate had already advanced.

The boundary itself is NOT changed: an IN_PROGRESS/PROVISIONAL Gameweek stays
excluded from historical evidence.  Only the completeness of REQUIRED completed
history is enforced.

Deterministic.  Temporary databases only; the single live read is the read-only
R5 snapshot non-regression proof, which skips when the snapshot is absent.
"""

from __future__ import annotations

import copy
import json
import sqlite3
import sys
from pathlib import Path

import pytest

from fpl_brain import analytics
from fpl_brain import history_completeness as hc
from fpl_brain import minutes_model
from fpl_brain import player_rates
from fpl_brain import repositories as repo
from fpl_brain.config import DEFAULT_CONFIG
from fpl_brain.database import connect_database

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import certify_gw5_gw8 as certifier  # noqa: E402

PLANNING_EVENT = 5
CUTOFF = "2026-09-14T08:14:22Z"
R5_SNAPSHOT = Path(
    "K:/FPL/data/exports/four_gw/gw05/snapshots/"
    "1e7fd22e-8b2d-44d6-9442-789d147287b7/execution_source_snapshot.db"
)

# A complete official observation: EVERY performance column is populated.
# Measured on the certified R5 snapshot: 1890 completed-fixture rows, all of them
# fully populated -- including the 961 did-not-plays, which are all explicit
# zeros.  So "all performance columns NULL" is the placeholder signature and can
# never be a legitimate observation.
COMPLETE_ROW = {
    "minutes": 90, "starts": 1, "total_points": 6, "goals_scored": 1, "assists": 0,
    "clean_sheets": 0, "goals_conceded": 1, "saves": 0, "bonus": 2, "bps": 30,
    "yellow_cards": 0, "red_cards": 0, "penalties_saved": 0, "penalties_missed": 0,
    "own_goals": 0, "influence": 20.0, "creativity": 5.0, "threat": 30.0, "ict_index": 5.5,
    "expected_goals": 0.5, "expected_assists": 0.1, "expected_goal_involvements": 0.6,
    "expected_goals_conceded": 0.8, "defensive_contribution": 2,
}
# A GENUINE did-not-play is all explicit zeros -- a real observation.
DNP_ROW = {key: (0.0 if isinstance(value, float) else 0) for key, value in COMPLETE_ROW.items()}
# A pure schedule placeholder: minutes 0 and every other performance column NULL.
PLACEHOLDER_ROW = {"minutes": 0, **{key: None for key in COMPLETE_ROW if key != "minutes"}}

PLAYERS = (1, 2)


def _config(tmp_path):
    config = copy.deepcopy(DEFAULT_CONFIG)
    config["paths"]["database"] = str(tmp_path / "fpl.db")
    config["paths"]["raw_dir"] = str(tmp_path / "raw")
    config["paths"]["exports_dir"] = str(tmp_path / "exports")
    Path(config["paths"]["database"]).parent.mkdir(parents=True, exist_ok=True)
    return config


def _seed(
    tmp_path,
    *,
    event4_final: bool,
    event4_row: dict,
    aggregate_minutes: int,
    fixture4_finished: bool,
    rows_events_1_3: dict | None = None,
):
    """Two players (both in team 1), four events, one fixture each.

    Events 1-3 are final.  Event 4 is materialised per the knobs, so the same
    builder can express: in-progress, provisional, stale-final and fresh-final.
    """

    config = _config(tmp_path)
    conn = connect_database(config["paths"]["database"])
    with conn:
        for position_id in (1, 2, 3, 4):
            conn.execute(
                "INSERT OR IGNORE INTO positions(id, singular_name, singular_name_short, plural_name,"
                " squad_select, squad_min_play, squad_max_play, raw_json, updated_at)"
                " VALUES (?,?,?,?,0,0,0,'{}',?)",
                (position_id, f"Pos{position_id}", f"P{position_id}", f"Pos{position_id}s", CUTOFF),
            )
        for team_id in (1, 2):
            conn.execute(
                "INSERT INTO teams(id, name, short_name, raw_json, updated_at) VALUES (?,?,?,'{}',?)",
                (team_id, f"Team {team_id}", f"T{team_id}", CUTOFF),
            )
        conn.execute("INSERT INTO events(id, name, finished, data_checked, is_current, is_next, raw_json, updated_at) VALUES (5,'Gameweek 5',0,0,0,1,'{}',?)", (CUTOFF,))
        for event in (1, 2, 3, 4):
            finished = 1 if event < 4 else (1 if event4_final else 0)
            data_checked = finished
            conn.execute(
                "INSERT INTO events(id, name, finished, data_checked, raw_json, updated_at) VALUES (?,?,?,?,'{}',?)",
                (event, f"Gameweek {event}", finished, data_checked, CUTOFF),
            )
        for event in (1, 2, 3, 4):
            finished = 1 if event < 4 else (1 if fixture4_finished else 0)
            started = 1
            conn.execute(
                "INSERT INTO fixtures(id, event, kickoff_time, team_h, team_a, started, finished, raw_json, updated_at)"
                " VALUES (?,?,?,1,2,?,?,'{}',?)",
                (event, event, f"2026-09-0{event}T14:00:00Z", started, finished, CUTOFF),
            )
        for player_id in PLAYERS:
            conn.execute(
                "INSERT INTO players(id, code, web_name, team_id, element_type, is_active, first_seen_at,"
                " last_seen_at, raw_json, updated_at) VALUES (?,?,?,1,3,1,?,?, '{}', ?)",
                (player_id, player_id, f"Player {player_id}", CUTOFF, CUTOFF, CUTOFF),
            )
        fetch_run = repo.create_fetch_run(conn, "history_test", str(tmp_path / "raw"), CUTOFF)
        for player_id in PLAYERS:
            for event in (1, 2, 3, 4):
                values = dict(rows_events_1_3 or COMPLETE_ROW) if event < 4 else dict(event4_row)
                columns = ["player_id", "event", "fixture_id", *(repo.GAMEWEEK_PERFORMANCE_COLUMNS), "source", "raw_json", "updated_at"]
                placeholders = ",".join("?" for _ in columns)
                conn.execute(
                    f"INSERT INTO player_gameweeks({','.join(columns)}) VALUES ({placeholders})",
                    (player_id, event, event, *[values.get(c) for c in repo.GAMEWEEK_PERFORMANCE_COLUMNS], "element_summary", "{}", CUTOFF),
                )
            conn.execute(
                "INSERT INTO player_snapshots(player_id, fetch_run_id, captured_at, minutes, total_points,"
                " raw_json) VALUES (?,?,?,?,0,'{}')",
                (player_id, fetch_run, CUTOFF, aggregate_minutes),
            )
    return config, conn


def _audit(conn):
    return hc.audit_history_completeness(conn, planning_event=PLANNING_EVENT, cutoff=CUTOFF)


# ---------------------------------------------------------------------------
# A. completed event + complete player row -> PASS
# ---------------------------------------------------------------------------


def test_A_completed_event_with_complete_rows_passes(tmp_path):
    _config_, conn = _seed(
        tmp_path, event4_final=True, event4_row=COMPLETE_ROW, aggregate_minutes=360, fixture4_finished=True
    )
    audit = _audit(conn)
    assert audit["required_completed_events"] == [1, 2, 3, 4]
    assert audit["latest_required_completed_event"] == 4
    assert audit["structural"]["placeholder_rows"] == 0
    assert audit["reconciliation"]["per_field"]["minutes"]["unexplained_players"] == 0
    assert audit["complete"] is True, audit["reasons"]
    assert audit["blocker"] is None
    conn.close()


# ---------------------------------------------------------------------------
# B. completed event + placeholder row -> FAIL closed
# ---------------------------------------------------------------------------


def test_B_completed_event_with_placeholder_row_fails_closed(tmp_path):
    _config_, conn = _seed(
        tmp_path, event4_final=True, event4_row=PLACEHOLDER_ROW, aggregate_minutes=360, fixture4_finished=True
    )
    audit = _audit(conn)
    assert audit["structural"]["placeholder_rows"] == len(PLAYERS)
    assert audit["structural"]["placeholder_rows_by_event"] == {"4": len(PLAYERS)}
    assert hc.DIAG_COMPLETED_EVENT_PLACEHOLDER_ROW in audit["reasons"]
    assert audit["complete"] is False
    assert audit["blocker"] == hc.CERTIFIED_PREDICTION_INPUT_HISTORY_INCOMPLETE
    conn.close()


# ---------------------------------------------------------------------------
# C. completed event + genuine official DNP -> PASS (zeros are an observation)
# ---------------------------------------------------------------------------


def test_C_genuine_dnp_is_not_a_placeholder(tmp_path):
    _config_, conn = _seed(
        tmp_path, event4_final=True, event4_row=DNP_ROW, aggregate_minutes=270, fixture4_finished=True
    )
    audit = _audit(conn)
    assert audit["structural"]["placeholder_rows"] == 0
    assert audit["reconciliation"]["per_field"]["minutes"]["unexplained_players"] == 0
    assert repo.row_is_scheduled_placeholder({"source": "element_summary", **DNP_ROW}) is False
    assert repo.row_is_scheduled_placeholder({"source": "element_summary", **PLACEHOLDER_ROW}) is True
    assert audit["complete"] is True, audit["reasons"]
    conn.close()


# ---------------------------------------------------------------------------
# D. in-progress event + placeholder rows -> PASS (not required evidence)
# ---------------------------------------------------------------------------


def test_D_in_progress_event_placeholders_are_not_required_evidence(tmp_path):
    _config_, conn = _seed(
        tmp_path, event4_final=False, event4_row=PLACEHOLDER_ROW, aggregate_minutes=360, fixture4_finished=False
    )
    audit = _audit(conn)
    assert audit["required_completed_events"] == [1, 2, 3]
    assert audit["latest_required_completed_event"] == 3
    assert audit["in_progress_events"] == [4]
    assert audit["latest_observed_event"] == 3
    # The unfinished fixture keeps the placeholders out of the consumers' scope.
    assert audit["structural"]["placeholder_rows"] == 0
    assert audit["complete"] is True, audit["reasons"]
    conn.close()


# ---------------------------------------------------------------------------
# E. all fixtures finished but the event is NOT officially finalised
# ---------------------------------------------------------------------------


def test_E_provisional_event_preserves_existing_semantics(tmp_path):
    from fpl_brain import planning

    _config_, conn = _seed(
        tmp_path, event4_final=False, event4_row=COMPLETE_ROW, aggregate_minutes=360, fixture4_finished=True
    )
    state, basis = planning.event_data_state(conn, 4)
    assert state == planning.EVENT_STATE_PROVISIONAL, basis
    audit = _audit(conn)
    # Not finalised => not a REQUIRED completed event, even though its fixtures ended.
    assert 4 not in audit["required_completed_events"]
    assert audit["latest_required_completed_event"] == 3
    assert audit["complete"] is True, audit["reasons"]
    conn.close()


def test_E2_provisional_event_with_placeholder_rows_is_reported(tmp_path):
    """A fixture-complete event's rows stay in the consumers' scope.

    The fixture-level ``finished=1`` boundary is unchanged, so a placeholder
    sitting on a finished fixture is readable by the models and must be caught
    even when the EVENT is not yet data-checked.
    """

    _config_, conn = _seed(
        tmp_path, event4_final=False, event4_row=PLACEHOLDER_ROW, aggregate_minutes=360, fixture4_finished=True
    )
    audit = _audit(conn)
    assert audit["structural"]["placeholder_rows"] == len(PLAYERS)
    assert audit["complete"] is False
    conn.close()


# ---------------------------------------------------------------------------
# F. the aggregate advanced because of an unfinished current event -> PASS
# ---------------------------------------------------------------------------


def test_F_aggregate_advanced_by_unfinished_fixture_is_permitted(tmp_path):
    _config_, conn = _seed(
        tmp_path, event4_final=False, event4_row=PLACEHOLDER_ROW, aggregate_minutes=360, fixture4_finished=False
    )
    reconciliation = hc.reconciliation_audit(conn, planning_event=PLANNING_EVENT, cutoff=CUTOFF)
    minutes = reconciliation["per_field"]["minutes"]
    assert minutes["enforced"] is True
    # 360 aggregate vs 270 completed history: the 90-minute residual is permitted
    # because team 1 has a started-but-unfinished fixture.
    assert minutes["residuals_permitted_by_unfinished_fixtures"] == len(PLAYERS)
    assert minutes["unexplained_players"] == 0
    assert reconciliation["complete"] is True
    conn.close()


# ---------------------------------------------------------------------------
# G. completed event whose history went stale -> reconciliation FAILS
# ---------------------------------------------------------------------------


def test_G_aggregate_ahead_of_completed_history_fails(tmp_path):
    _config_, conn = _seed(
        tmp_path, event4_final=True, event4_row=PLACEHOLDER_ROW, aggregate_minutes=360, fixture4_finished=True
    )
    reconciliation = hc.reconciliation_audit(conn, planning_event=PLANNING_EVENT, cutoff=CUTOFF)
    minutes = reconciliation["per_field"]["minutes"]
    assert minutes["unexplained_players"] == len(PLAYERS)
    assert minutes["complete"] is False
    assert reconciliation["complete"] is False
    offenders = {row["player_id"] for row in minutes["unexplained"]}
    assert offenders == set(PLAYERS)
    assert all(row["residual"] == 90.0 for row in minutes["unexplained"])
    audit = _audit(conn)
    assert hc.DIAG_AGGREGATE_HISTORY_RECONCILIATION_FAILED in audit["reasons"]
    conn.close()


def test_G2_history_exceeding_the_official_aggregate_fails(tmp_path):
    """History can never exceed the official cumulative total."""

    _config_, conn = _seed(
        tmp_path, event4_final=True, event4_row=COMPLETE_ROW, aggregate_minutes=200, fixture4_finished=True
    )
    audit = _audit(conn)
    minutes = audit["reconciliation"]["per_field"]["minutes"]
    assert minutes["history_exceeds_aggregate_players"] == len(PLAYERS)
    assert hc.DIAG_HISTORY_EXCEEDS_OFFICIAL_AGGREGATE in audit["reasons"]
    assert audit["complete"] is False
    conn.close()


# ---------------------------------------------------------------------------
# H. certification permission cannot become TRUE while the blocker exists
# ---------------------------------------------------------------------------


def _permission_base(**overrides):
    base = dict(
        temporal_status="CAUSAL",
        dependency_validation="COHERENT",
        horizon_status=certifier.fg.DECISION_HORIZON_COMPLETE,
        data_snapshot_sha256="d" * 64,
    )
    base.update(overrides)
    return base


def test_H_permission_refused_while_history_is_incomplete(tmp_path):
    _config_, conn = _seed(
        tmp_path, event4_final=True, event4_row=PLACEHOLDER_ROW, aggregate_minutes=360, fixture4_finished=True
    )
    incomplete = _audit(conn)
    permitted, reasons = certifier.decide_search_permission(
        **_permission_base(history_completeness=incomplete)
    )
    assert permitted is False
    assert any(hc.CERTIFIED_PREDICTION_INPUT_HISTORY_INCOMPLETE in reason for reason in reasons), reasons
    assert any(hc.DIAG_COMPLETED_EVENT_PLACEHOLDER_ROW in reason for reason in reasons), reasons
    conn.close()


def test_H2_permission_granted_when_history_is_complete(tmp_path):
    _config_, conn = _seed(
        tmp_path, event4_final=True, event4_row=COMPLETE_ROW, aggregate_minutes=360, fixture4_finished=True
    )
    permitted, reasons = certifier.decide_search_permission(
        **_permission_base(history_completeness=_audit(conn))
    )
    assert permitted is True, reasons
    assert reasons == []
    conn.close()


def test_H3_permission_unchanged_when_the_audit_is_not_supplied():
    """Backwards compatible: the rule stays callable with no database."""

    permitted, reasons = certifier.decide_search_permission(**_permission_base())
    assert permitted is True, reasons


def test_H4_r5_shaped_audit_does_not_refuse_permission(tmp_path):
    """The accepted R5 shape (in-progress GW4) must still be permitted."""

    _config_, conn = _seed(
        tmp_path, event4_final=False, event4_row=PLACEHOLDER_ROW, aggregate_minutes=360, fixture4_finished=False
    )
    permitted, reasons = certifier.decide_search_permission(
        **_permission_base(history_completeness=_audit(conn))
    )
    assert permitted is True, reasons
    conn.close()


def test_H5_decision_runner_refuses_an_artifact_with_an_incomplete_audit(tmp_path):
    """Defence in depth in the CONSUMER, not just the producer."""

    from fpl_brain import four_gw_decision as fg

    _config_, conn = _seed(
        tmp_path, event4_final=True, event4_row=PLACEHOLDER_ROW, aggregate_minutes=360, fixture4_finished=True
    )
    incomplete = _audit(conn)
    conn.close()

    def _artifact(history_completeness):
        return {
            "schema": "fpl_brain.certification_artifact.v1",
            "temporal_status": "CAUSAL",
            "dependency_validation": "COHERENT",
            "certified_bundles": {"5": {"runs": {}}},
            "data_snapshot_sha256": "d" * 64,
            "decision_search_permitted": True,
            "route_search_executed": False,
            "transfer_execution_performed": False,
            "history_completeness": history_completeness,
        }

    path = tmp_path / "artifact.json"
    path.write_text(json.dumps(_artifact(incomplete)), encoding="utf-8")
    with pytest.raises(fg.DecisionCertificationRequired) as failure:
        fg.load_certification_artifact(path)
    assert hc.CERTIFIED_PREDICTION_INPUT_HISTORY_INCOMPLETE in str(failure.value)

    # A complete audit, and an artifact without the block, both stay readable.
    complete = dict(incomplete, complete=True, blocker=None, reasons=[])
    path.write_text(json.dumps(_artifact(complete)), encoding="utf-8")
    assert fg.load_certification_artifact(path)["decision_search_permitted"] is True
    path.write_text(json.dumps(_artifact(None)), encoding="utf-8")
    assert fg.load_certification_artifact(path)["decision_search_permitted"] is True


# ---------------------------------------------------------------------------
# I. player_rates reports the gap instead of silently shrinking exposure
# ---------------------------------------------------------------------------


def test_I_player_rates_reports_the_gap(tmp_path):
    fresh = _seed(tmp_path / "fresh", event4_final=True, event4_row=COMPLETE_ROW, aggregate_minutes=360, fixture4_finished=True)[1]
    evidence = player_rates.current_rate_evidence(fresh, 1, "xG_per90", PLANNING_EVENT, CUTOFF)
    assert evidence["current_minutes"] == 360.0
    assert evidence["placeholder_rows"] == []
    assert hc.DIAG_COMPLETED_EVENT_PLACEHOLDER_ROW not in evidence["flags"]
    fresh.close()

    stale = _seed(tmp_path / "stale", event4_final=True, event4_row=PLACEHOLDER_ROW, aggregate_minutes=360, fixture4_finished=True)[1]
    evidence = player_rates.current_rate_evidence(stale, 1, "xG_per90", PLANNING_EVENT, CUTOFF)
    # The placeholder contributes no exposure (it holds none) but is not silent:
    assert evidence["current_minutes"] == 270.0
    assert evidence["played_rows"] == 3
    assert len(evidence["placeholder_rows"]) == 1
    assert hc.DIAG_COMPLETED_EVENT_PLACEHOLDER_ROW in evidence["flags"]
    stale.close()


def test_I2_player_rates_dnp_is_not_a_gap(tmp_path):
    _config_, conn = _seed(
        tmp_path, event4_final=True, event4_row=DNP_ROW, aggregate_minutes=270, fixture4_finished=True
    )
    evidence = player_rates.current_rate_evidence(conn, 1, "xG_per90", PLANNING_EVENT, CUTOFF)
    assert evidence["current_minutes"] == 270.0
    assert evidence["placeholder_rows"] == []
    assert hc.DIAG_COMPLETED_EVENT_PLACEHOLDER_ROW not in evidence["flags"]
    conn.close()


# ---------------------------------------------------------------------------
# J. minutes evidence reports the gap instead of reading it as a DNP
# ---------------------------------------------------------------------------


def test_J_minutes_evidence_reports_the_gap(tmp_path):
    _config_, conn = _seed(
        tmp_path, event4_final=True, event4_row=PLACEHOLDER_ROW, aggregate_minutes=360, fixture4_finished=True
    )
    rows = analytics.completed_rows_as_of(conn, 1, CUTOFF, PLANNING_EVENT)
    assert len(rows) == 4
    flagged = [row for row in rows if row["history_placeholder"]]
    assert len(flagged) == 1
    assert flagged[0]["event"] == 4

    classified = minutes_model.classify_evidence_rows(rows)
    placeholder = next(row for row in classified if row["history_placeholder"])
    assert placeholder["evidence_class"] == minutes_model.EVIDENCE_UNKNOWN
    assert "placeholder" in placeholder["evidence_reason"]
    # A placeholder is NOT negative selection evidence.
    assert placeholder["evidence_class"] not in minutes_model._START_OBSERVATION_CLASSES
    conn.close()


def test_J2_minutes_dnp_still_classifies_as_an_observation(tmp_path):
    _config_, conn = _seed(
        tmp_path, event4_final=True, event4_row=DNP_ROW, aggregate_minutes=270, fixture4_finished=True
    )
    rows = analytics.completed_rows_as_of(conn, 1, CUTOFF, PLANNING_EVENT)
    assert all(row["history_placeholder"] is False for row in rows)
    classified = minutes_model.classify_evidence_rows(rows)
    dnp = next(row for row in classified if int(row["event"]) == 4)
    # A genuine did-not-play is a real observation, not a completeness gap: it keeps
    # its normal taxonomy path rather than the placeholder reason.
    assert "placeholder" not in dnp["evidence_reason"]
    assert dnp["evidence_class"] in minutes_model.EVIDENCE_CLASSES
    conn.close()


# ---------------------------------------------------------------------------
# K. the accepted R5 snapshot is NOT retroactively invalidated
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not R5_SNAPSHOT.exists(), reason="certified R5 snapshot is not present in this checkout")
def test_K_r5_snapshot_passes_history_completeness():
    conn = sqlite3.connect(f"file:{R5_SNAPSHOT.as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        audit = hc.audit_history_completeness(conn, planning_event=5, cutoff="2026-09-14T08:14:22Z")
    finally:
        conn.close()
    # GW4 was IN_PROGRESS at the cutoff, so GW3 is the latest required event and
    # GW4 is correctly not required.
    assert audit["required_completed_events"] == [1, 2, 3]
    assert audit["latest_required_completed_event"] == 3
    assert 4 in audit["in_progress_events"]
    assert audit["latest_observed_event"] == 3
    assert audit["structural"]["placeholder_rows"] == 0
    minutes = audit["reconciliation"]["per_field"]["minutes"]
    assert minutes["unexplained_players"] == 0
    assert minutes["history_exceeds_aggregate_players"] == 0
    assert audit["complete"] is True, audit["reasons"]
    assert audit["blocker"] is None


# ---------------------------------------------------------------------------
# L. deterministic ordering
# ---------------------------------------------------------------------------


def test_L_audit_is_deterministic_and_ordered(tmp_path):
    _config_, conn = _seed(
        tmp_path, event4_final=True, event4_row=PLACEHOLDER_ROW, aggregate_minutes=360, fixture4_finished=True
    )
    first, second = _audit(conn), _audit(conn)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert first["reasons"] == sorted(first["reasons"])
    assert first["required_completed_events"] == sorted(first["required_completed_events"])
    assert first["structural"]["affected_players"] == sorted(first["structural"]["affected_players"])
    assert first["reconciliation"]["explainable_team_ids"] == sorted(
        first["reconciliation"]["explainable_team_ids"]
    )
    conn.close()


def test_L2_reason_tokens_are_sorted_when_several_fire(tmp_path):
    _config_, conn = _seed(
        tmp_path, event4_final=True, event4_row=PLACEHOLDER_ROW, aggregate_minutes=999, fixture4_finished=True
    )
    audit = _audit(conn)
    assert audit["reasons"] == sorted(audit["reasons"])
    assert len(audit["reasons"]) >= 2
    conn.close()


# ---------------------------------------------------------------------------
# REPAIR A: the refresh contract before certification
# ---------------------------------------------------------------------------


def _fake_refresh(config, calls, *, populate: bool):
    """A stand-in for the canonical ingest that can either fix or fail the gap."""

    def _run_fetch(cfg, summaries="none", live_gw=None, dry_run=False):
        calls.append({"summaries": summaries, "live_gw": live_gw, "dry_run": dry_run})
        if populate:
            conn = connect_database(cfg["paths"]["database"])
            with conn:
                conn.execute(
                    "UPDATE player_gameweeks SET minutes=90.0, starts=1, total_points=6, goals_scored=1,"
                    " assists=0, clean_sheets=0, goals_conceded=1, saves=0, bonus=2, bps=30,"
                    " yellow_cards=0, red_cards=0, penalties_saved=0, penalties_missed=0, own_goals=0,"
                    " influence=20.0, creativity=5.0, threat=30.0, ict_index=5.5, expected_goals=0.5,"
                    " expected_assists=0.1, expected_goal_involvements=0.6, expected_goals_conceded=0.8,"
                    " defensive_contribution=2 WHERE event=4"
                )
            conn.close()
        return {"status": "success", "run_id": 7, "endpoints_ok": ["bootstrap-static", "fixtures", f"element-summary/all"],
                "endpoints_failed": []}

    return _run_fetch


def test_refresh_is_a_noop_when_history_is_complete(tmp_path, monkeypatch):
    config, conn = _seed(
        tmp_path, event4_final=True, event4_row=COMPLETE_ROW, aggregate_minutes=360, fixture4_finished=True
    )
    conn.close()
    calls: list[dict] = []
    import fpl_brain.ingest as ingest

    monkeypatch.setattr(ingest, "run_fetch", _fake_refresh(config, calls, populate=False))
    report = hc.refresh_completed_event_history(config, planning_event=PLANNING_EVENT, cutoff=CUTOFF)
    assert calls == []
    assert report["history_required"] is False
    assert report["refresh"]["performed"] is False
    assert report["complete"] is True


def test_refresh_requests_the_canonical_ingest_and_succeeds(tmp_path, monkeypatch):
    config, conn = _seed(
        tmp_path, event4_final=True, event4_row=PLACEHOLDER_ROW, aggregate_minutes=360, fixture4_finished=True
    )
    conn.close()
    calls: list[dict] = []
    import fpl_brain.ingest as ingest

    monkeypatch.setattr(ingest, "run_fetch", _fake_refresh(config, calls, populate=True))
    report = hc.refresh_completed_event_history(config, planning_event=PLANNING_EVENT, cutoff=CUTOFF)
    # xG/xA only exist in the element-summary payload, so the refresh must ask for
    # the full summaries fan-out, scoped to the latest REQUIRED completed event.
    assert calls == [{"summaries": "all", "live_gw": 4, "dry_run": False}]
    assert report["before"]["complete"] is False
    assert report["after"]["complete"] is True
    assert report["refresh"]["performed"] is True
    assert report["complete"] is True


def test_refresh_fails_closed_when_history_stays_incomplete(tmp_path, monkeypatch):
    config, conn = _seed(
        tmp_path, event4_final=True, event4_row=PLACEHOLDER_ROW, aggregate_minutes=360, fixture4_finished=True
    )
    conn.close()
    calls: list[dict] = []
    import fpl_brain.ingest as ingest

    monkeypatch.setattr(ingest, "run_fetch", _fake_refresh(config, calls, populate=False))
    with pytest.raises(hc.HistoryRefreshIncomplete) as failure:
        hc.refresh_completed_event_history(config, planning_event=PLANNING_EVENT, cutoff=CUTOFF)
    assert hc.CERTIFIED_PREDICTION_INPUT_HISTORY_INCOMPLETE in str(failure.value)
    assert len(calls) == 1


def test_refresh_dry_run_never_ingests(tmp_path, monkeypatch):
    config, conn = _seed(
        tmp_path, event4_final=True, event4_row=PLACEHOLDER_ROW, aggregate_minutes=360, fixture4_finished=True
    )
    conn.close()
    calls: list[dict] = []
    import fpl_brain.ingest as ingest

    monkeypatch.setattr(ingest, "run_fetch", _fake_refresh(config, calls, populate=True))
    report = hc.refresh_completed_event_history(config, planning_event=PLANNING_EVENT, cutoff=CUTOFF, dry_run=True)
    assert calls == []
    assert report["dry_run"] is True
    assert report["history_required"] is True
    assert report["complete"] is False


def test_planning_event_derivation(tmp_path):
    _config_, conn = _seed(
        tmp_path, event4_final=False, event4_row=PLACEHOLDER_ROW, aggregate_minutes=0, fixture4_finished=False
    )
    assert hc.planning_event_from_db(conn) == 5
    conn.close()
